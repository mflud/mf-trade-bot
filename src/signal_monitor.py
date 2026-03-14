"""
Multi-Instrument Real-Time Signal Monitor.

Polls TopstepX every 30 seconds, builds the latest 5-min bar, and displays
side-by-side panels for each configured instrument:
  - Volatility regime (σ in bps/points, annualised vol, regime label)
  - Current bar status (OHLCV, scaled return, volume ratio)
  - Signal status: GREEN (long), RED (short), YELLOW (watching)
  - If signal: entry, target, stop in points + expiry countdown

Instruments and their optimal parameters (from backtest):
  MYM  Micro Dow      stop=2.0σ  target=2.5σ  $0.50/pt
  MES  Micro S&P 500  stop=1.5σ  target=2.5σ  $5.00/pt

Run modes:
  python src/signal_monitor.py          # live (requires .env credentials)
  python src/signal_monitor.py --demo   # static demo with synthetic data
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, "src")

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Strategy parameters ────────────────────────────────────────────────────────

TF_MINUTES    = 5
TRAILING_BARS = 100
SIGNAL_SIGMA  = 3.0
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 15
BARS_PER_YEAR = 252 * 23 * 60

REGIME_THRESHOLDS = [
    (0.10, "QUIET",    "dim"),
    (0.15, "NORMAL",   "cyan"),
    (0.20, "ELEVATED", "yellow"),
    (0.30, "ACTIVE",   "orange1"),
    (1.00, "HIGH VOL", "red"),
]


@dataclass
class InstrumentConfig:
    symbol:      str
    search_term: str          # passed to search_contracts()
    stop_sigma:  float        # stop loss in σ units
    target_sigma: float       # profit target in σ units
    point_value: float        # $ per point (for display only)
    ev_sigma:    float        # expected EV per signal in σ (from backtest)


INSTRUMENTS = [
    InstrumentConfig("MYM", "MYM", stop_sigma=2.0, target_sigma=2.5,
                     point_value=0.50, ev_sigma=0.484),
    InstrumentConfig("MES", "MES", stop_sigma=1.5, target_sigma=2.5,
                     point_value=5.00, ev_sigma=0.081),
]

console = Console()


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts:     datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class Signal:
    cfg:        InstrumentConfig
    direction:  int            # +1 long / -1 short
    entry:      float
    sigma:      float
    sigma_pts:  float
    scaled:     float
    vol_ratio:  float
    bar_ts:     datetime
    target:     float = field(init=False)
    stop:       float = field(init=False)
    expires_at: datetime = field(init=False)

    def __post_init__(self):
        self.target     = self.entry + self.direction * self.cfg.target_sigma * self.sigma_pts
        self.stop       = self.entry - self.direction * self.cfg.stop_sigma   * self.sigma_pts
        self.expires_at = self.bar_ts + timedelta(minutes=MAX_HOLD_MIN)

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)


@dataclass
class RecentSignal:
    symbol:  str
    signal:  Signal
    outcome: str        # "TARGET", "STOPPED", "TIME EXIT", "OPEN"
    pnl_pts: float


@dataclass
class InstrumentState:
    cfg:           InstrumentConfig
    cid:           str = ""
    cname:         str = ""
    bars:          list[Bar] = field(default_factory=list)
    sigma:         float = 0.0
    sigma_pts:     float = 0.0
    mean_vol:      float = 1.0
    active_signal: Signal | None = None
    history:       list[RecentSignal] = field(default_factory=list)
    error:         str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def annualised_vol(sigma: float) -> float:
    return sigma * math.sqrt(BARS_PER_YEAR / TF_MINUTES)


def regime_label(ann_vol: float) -> tuple[str, str]:
    for thresh, label, style in REGIME_THRESHOLDS:
        if ann_vol < thresh:
            return label, style
    return REGIME_THRESHOLDS[-1][1], REGIME_THRESHOLDS[-1][2]


def _next_bar_close(now: datetime) -> datetime:
    epoch_min = int(now.timestamp() // 60)
    next_close_min = ((epoch_min // TF_MINUTES) + 1) * TF_MINUTES
    return datetime.fromtimestamp(next_close_min * 60, tz=timezone.utc)


# ── Panel builders ─────────────────────────────────────────────────────────────

def build_regime_panel(state: InstrumentState) -> Panel:
    sigma, sigma_pts = state.sigma, state.sigma_pts
    ann = annualised_vol(sigma)
    label, style = regime_label(ann)
    cur_vol   = state.bars[-1].volume if state.bars else 0.0
    vol_ratio = cur_vol / state.mean_vol if state.mean_vol else 0.0

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=16)
    t.add_column(width=24)

    t.add_row("σ per bar:",
              f"[bold]{sigma * 10000:.2f} bps[/]  │  {sigma_pts:.2f} pts")
    t.add_row("Ann. vol:",
              f"[bold]{ann*100:.1f}%[/]")
    t.add_row("Regime:",
              f"[bold {style}]{label}[/]")
    t.add_row("Avg volume:",
              f"{state.mean_vol:,.0f}")
    t.add_row("Cur volume:",
              f"{cur_vol:,.0f}  ({vol_ratio:.1f}× avg)")

    return Panel(t, title="[bold]VOL REGIME[/]",
                 border_style="blue", padding=(0, 1))


def build_bar_panel(state: InstrumentState) -> Panel:
    bar   = state.bars[-1]
    sigma = state.sigma
    ret    = math.log(bar.close / bar.open) if bar.open else 0.0
    scaled = ret / sigma if sigma else 0.0
    vol_ratio = bar.volume / state.mean_vol if state.mean_vol else 0.0

    sc_style = ("green" if scaled > 0 else "red") if abs(scaled) >= SIGNAL_SIGMA \
               else ("yellow" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "white")
    vr_style = "green" if vol_ratio >= VOL_RATIO_MIN else \
               ("yellow" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "white")
    sc_check = "✓" if abs(scaled) >= SIGNAL_SIGMA else \
               ("~" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "✗")
    vr_check = "✓" if vol_ratio >= VOL_RATIO_MIN else \
               ("~" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "✗")

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=10)
    t.add_column(width=30)

    t.add_row("Open:",   f"{bar.open:,.2f}")
    t.add_row("High:",   f"[green]{bar.high:,.2f}[/]")
    t.add_row("Low:",    f"[red]{bar.low:,.2f}[/]")
    t.add_row("Close:",  f"[bold]{bar.close:,.2f}[/]")
    t.add_row("Volume:", f"{bar.volume:,.0f}  [{vr_style}]{vol_ratio:.2f}× "
                         f"[thr {VOL_RATIO_MIN:.1f}×] {vr_check}[/]")
    t.add_row("Scaled:", f"[{sc_style}]{scaled:+.2f}σ  "
                         f"[thr {SIGNAL_SIGMA:.0f}σ] {sc_check}[/]")

    return Panel(t, title=f"[bold]LAST {TF_MINUTES}-MIN BAR[/]",
                 border_style="blue", padding=(0, 1))


def build_signal_panel(state: InstrumentState, now: datetime) -> Panel:
    signal = state.active_signal
    cfg    = state.cfg

    if signal is None:
        bar    = state.bars[-1]
        sigma  = state.sigma
        scaled = math.log(bar.close / bar.open) / sigma if sigma and bar.open else 0.0
        vol_ratio = bar.volume / state.mean_vol if state.mean_vol else 0.0
        sc_pct = min(abs(scaled)  / SIGNAL_SIGMA * 100, 100)
        vr_pct = min(vol_ratio    / VOL_RATIO_MIN * 100, 100)
        bar_sc = "█" * int(sc_pct / 5) + "░" * (20 - int(sc_pct / 5))
        bar_vr = "█" * int(vr_pct / 5) + "░" * (20 - int(vr_pct / 5))

        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", width=16)
        t.add_column()
        t.add_row("Scaled return:",
                  f"[yellow]{bar_sc}[/] {abs(scaled):.2f}σ / {SIGNAL_SIGMA:.0f}σ")
        t.add_row("Volume ratio:",
                  f"[yellow]{bar_vr}[/] {vol_ratio:.2f}× / {VOL_RATIO_MIN:.1f}×")
        t.add_row("", "")
        t.add_row("", f"[dim]Need ≥{SIGNAL_SIGMA:.0f}σ + ≥{VOL_RATIO_MIN:.1f}× vol[/]")

        return Panel(t, title="[bold yellow]⬤  WATCHING[/]",
                     border_style="yellow", padding=(0, 2))

    direction_str = "LONG  ▲" if signal.direction == 1 else "SHORT ▼"
    color         = "green"  if signal.direction == 1 else "red"
    remaining     = signal.expires_at - now
    rem_str       = f"{int(remaining.total_seconds() // 60)}m " \
                    f"{int(remaining.total_seconds() % 60):02d}s" \
                    if remaining.total_seconds() > 0 else "[blink]EXPIRED[/]"
    expires_str   = signal.expires_at.astimezone(
                        datetime.now().astimezone().tzinfo
                    ).strftime("%H:%M:%S %Z")
    rr = signal.target_pts() / signal.stop_pts() if signal.stop_pts() else 0.0

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=10)
    t.add_column()

    t.add_row("Entry:",
              f"[bold]{signal.entry:,.2f}[/]")
    t.add_row("Target:",
              f"[bold {color}]{signal.target:,.2f}[/]  "
              f"([{color}]+{signal.target_pts():.2f} pts[/] │ +{cfg.target_sigma:.1f}σ)")
    t.add_row("Stop:",
              f"[bold red]{signal.stop:,.2f}[/]  "
              f"([red]−{signal.stop_pts():.2f} pts[/] │ −{cfg.stop_sigma:.1f}σ)")
    t.add_row("R:R / EV:",
              f"{rr:.2f}:1  (EV ≈ +{cfg.ev_sigma:.2f}σ / signal)")
    t.add_row("Expires:",
              f"{expires_str}  [dim]({rem_str})[/]")
    t.add_row("Trigger:",
              f"[dim]scaled={signal.scaled:+.2f}σ  vol={signal.vol_ratio:.2f}×[/]")

    return Panel(t,
                 title=f"[bold {color}]⬤  {direction_str}  SIGNAL[/]",
                 border_style=color, padding=(0, 2))


def build_instrument_column(state: InstrumentState, now: datetime) -> Table:
    """Vertical stack of panels for one instrument."""
    col = Table.grid(padding=(0, 0))
    col.add_column()
    col.add_row(Panel(
        Text(state.cname or state.cfg.symbol, style="bold cyan", justify="center"),
        border_style="dark_blue", padding=(0, 1),
    ))
    col.add_row(build_regime_panel(state))
    col.add_row(build_bar_panel(state))
    col.add_row(build_signal_panel(state, now))
    return col


def build_history_table(history: list[RecentSignal]) -> Panel:
    t = Table(box=box.SIMPLE, padding=(0, 1), show_header=True,
              header_style="bold dim")
    t.add_column("Time",    width=8)
    t.add_column("Sym",     width=5)
    t.add_column("Dir",     width=6)
    t.add_column("Entry",   width=10, justify="right")
    t.add_column("Target",  width=10, justify="right")
    t.add_column("Stop",    width=10, justify="right")
    t.add_column("Outcome", width=12)
    t.add_column("P&L",     width=10, justify="right")

    for rs in reversed(history[-8:]):
        s  = rs.signal
        ts = s.bar_ts.astimezone(datetime.now().astimezone().tzinfo)
        dir_str = "[green]LONG[/]" if s.direction == 1 else "[red]SHORT[/]"

        if rs.outcome == "TARGET":
            out_str = "[green]HIT TARGET[/]"
            pnl_str = f"[green]+{rs.pnl_pts:.2f}[/]"
        elif rs.outcome == "STOPPED":
            out_str = "[red]STOPPED[/]"
            pnl_str = f"[red]−{abs(rs.pnl_pts):.2f}[/]"
        elif rs.outcome == "TIME EXIT":
            pnl_col = "green" if rs.pnl_pts >= 0 else "red"
            sign    = "+" if rs.pnl_pts >= 0 else "−"
            out_str = "[yellow]TIME EXIT[/]"
            pnl_str = f"[{pnl_col}]{sign}{abs(rs.pnl_pts):.2f}[/]"
        else:
            out_str = "[bold yellow]OPEN[/]"
            pnl_str = "[dim]—[/]"

        t.add_row(
            ts.strftime("%H:%M"),
            rs.symbol,
            dir_str,
            f"{s.entry:,.2f}",
            f"{s.target:,.2f}",
            f"{s.stop:,.2f}",
            out_str,
            pnl_str,
        )

    return Panel(t, title="[bold]RECENT SIGNALS[/]",
                 border_style="dim", padding=(0, 1))


def build_header(now: datetime) -> Panel:
    local = now.astimezone(datetime.now().astimezone().tzinfo)
    t = Text(justify="center")
    t.append("  SIGNAL MONITOR  ", style="bold white on dark_blue")
    t.append("  MYM & MES  ", style="bold cyan")
    t.append("│  ")
    t.append(local.strftime("%a %Y-%m-%d  %H:%M:%S %Z"), style="dim")
    t.append("  │  next bar: ")
    nb_local = _next_bar_close(now).astimezone(datetime.now().astimezone().tzinfo)
    t.append(nb_local.strftime("%H:%M:%S"), style="yellow")
    return Panel(t, border_style="dark_blue", padding=(0, 0))


def render(states: list[InstrumentState],
           history: list[RecentSignal],
           now: datetime) -> Table:
    root = Table.grid(padding=(0, 0))
    root.add_column()

    root.add_row(build_header(now))

    cols = Columns(
        [build_instrument_column(s, now) for s in states],
        equal=True, padding=(0, 1),
    )
    root.add_row(cols)

    if history:
        root.add_row(build_history_table(history))

    return root


# ── Live mode ──────────────────────────────────────────────────────────────────

def evaluate(state: InstrumentState) -> Signal | None:
    bars    = state.bars
    closes  = np.array([b.close  for b in bars])
    volumes = np.array([b.volume for b in bars])

    trail = np.log(closes[-TRAILING_BARS:] / closes[-TRAILING_BARS - 1:-1]) \
            if len(closes) > TRAILING_BARS else np.array([])
    sigma     = float(np.std(trail, ddof=1)) if len(trail) >= 2 else 0.0
    sigma_pts = sigma * closes[-1]
    mean_vol  = float(volumes[-TRAILING_BARS - 1:-1].mean()) if len(volumes) > 1 else 1.0

    state.sigma     = sigma
    state.sigma_pts = sigma_pts
    state.mean_vol  = mean_vol

    last      = bars[-1]
    bar_ret   = math.log(last.close / last.open) if last.open else 0.0
    scaled    = bar_ret / sigma if sigma else 0.0
    vol_ratio = last.volume / mean_vol

    if abs(scaled) >= SIGNAL_SIGMA and vol_ratio >= VOL_RATIO_MIN:
        return Signal(cfg=state.cfg,
                      direction=1 if scaled > 0 else -1,
                      entry=last.close, sigma=sigma, sigma_pts=sigma_pts,
                      scaled=scaled, vol_ratio=vol_ratio, bar_ts=last.ts)
    return None


def run_live():
    from topstep_client import TopstepClient

    client = TopstepClient()
    client.login()

    states: list[InstrumentState] = []
    for cfg in INSTRUMENTS:
        contracts = client.search_contracts(cfg.search_term)
        if not contracts:
            console.print(f"[red]No contract found for {cfg.symbol}[/]")
            continue
        c = contracts[0]
        st = InstrumentState(cfg=cfg, cid=c["id"], cname=c["name"])
        states.append(st)
        console.print(f"  {cfg.symbol}: {c['name']}  id={c['id']}")

    combined_history: list[RecentSignal] = []

    def fetch_bars(state: InstrumentState):
        end   = datetime.now(timezone.utc)
        start = end - timedelta(minutes=TF_MINUTES * (TRAILING_BARS + 5))
        raw = client.get_bars(contract_id=state.cid, start=start, end=end,
                              unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
                              limit=TRAILING_BARS + 10)
        raw = list(reversed(raw))
        state.bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                          open=b["o"], high=b["h"], low=b["l"],
                          close=b["c"], volume=b["v"]) for b in raw]

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            now = datetime.now(timezone.utc)

            for state in states:
                fetch_bars(state)
                if len(state.bars) < TRAILING_BARS + 1:
                    continue

                new_sig = evaluate(state)

                # Resolve expired active signal
                if state.active_signal and now >= state.active_signal.expires_at:
                    last_close = state.bars[-1].close
                    pnl = (last_close - state.active_signal.entry) * state.active_signal.direction
                    combined_history.append(
                        RecentSignal(state.cfg.symbol, state.active_signal, "TIME EXIT", pnl)
                    )
                    state.active_signal = None

                if new_sig and (state.active_signal is None or
                                new_sig.bar_ts != state.active_signal.bar_ts):
                    if state.active_signal:
                        combined_history.append(
                            RecentSignal(state.cfg.symbol, state.active_signal, "OPEN", 0)
                        )
                    state.active_signal = new_sig

            live.update(render(states, combined_history, now))
            time.sleep(30)


# ── Demo mode ──────────────────────────────────────────────────────────────────

def run_demo():
    now = datetime.now(timezone.utc)

    # MYM synthetic state
    mym_cfg   = INSTRUMENTS[0]
    mym_sigma = 0.000721       # ~19% annualised at 5-min bars
    mym_price = 46_500.0
    mym_sp    = mym_sigma * mym_price   # ≈ 33.5 pts per 1σ

    mym_state = InstrumentState(cfg=mym_cfg, cname="MYMH6")
    mym_state.sigma     = mym_sigma
    mym_state.sigma_pts = mym_sp
    mym_state.mean_vol  = 4_200.0
    mym_state.bars = [Bar(ts=now - timedelta(minutes=5),
                          open=46_430, high=46_560, low=46_415,
                          close=46_500, volume=9_800)]
    mym_state.active_signal = Signal(
        cfg=mym_cfg, direction=1, entry=46_500,
        sigma=mym_sigma, sigma_pts=mym_sp,
        scaled=+3.91, vol_ratio=2.33, bar_ts=now - timedelta(minutes=5),
    )

    # MES synthetic state — watching
    mes_cfg   = INSTRUMENTS[1]
    mes_sigma = 0.000721
    mes_price = 6_625.0
    mes_sp    = mes_sigma * mes_price   # ≈ 4.78 pts per 1σ

    mes_state = InstrumentState(cfg=mes_cfg, cname="MESH6")
    mes_state.sigma     = mes_sigma
    mes_state.sigma_pts = mes_sp
    mes_state.mean_vol  = 8_450.0
    mes_state.bars = [Bar(ts=now - timedelta(minutes=5),
                          open=6_622.0, high=6_627.5, low=6_620.5,
                          close=6_625.0, volume=6_820)]

    history = [
        RecentSignal("MYM", Signal(mym_cfg, -1, 46_550, mym_sigma, mym_sp,
                                   -3.5, 2.1, now - timedelta(minutes=75)),
                     "TARGET",    mym_sp * 2.5),
        RecentSignal("MES", Signal(mes_cfg, +1, 6_610, mes_sigma, mes_sp,
                                   +4.1, 1.9, now - timedelta(minutes=130)),
                     "STOPPED",  -mes_sp * 1.5),
        RecentSignal("MYM", Signal(mym_cfg, +1, 46_380, mym_sigma, mym_sp,
                                   +3.3, 1.7, now - timedelta(minutes=215)),
                     "TIME EXIT", mym_sp * 0.6),
        RecentSignal("MES", Signal(mes_cfg, -1, 6_645, mes_sigma, mes_sp,
                                   -3.8, 2.4, now - timedelta(minutes=280)),
                     "TARGET",    mes_sp * 2.5),
    ]

    console.print()
    console.print("[bold underline]DEMO — MYM: LONG SIGNAL  │  MES: WATCHING[/]",
                  justify="center")
    console.print(render([mym_state, mes_state], history, now))

    # Frame 2: MES short, MYM watching
    mes_state2 = InstrumentState(cfg=mes_cfg, cname="MESH6")
    mes_state2.sigma     = mes_sigma
    mes_state2.sigma_pts = mes_sp
    mes_state2.mean_vol  = 8_450.0
    mes_state2.bars = [Bar(ts=now - timedelta(minutes=5),
                           open=6_640.0, high=6_641.0, low=6_618.5,
                           close=6_620.0, volume=21_800)]
    mes_state2.active_signal = Signal(
        cfg=mes_cfg, direction=-1, entry=6_620.0,
        sigma=mes_sigma, sigma_pts=mes_sp,
        scaled=-4.12, vol_ratio=2.58, bar_ts=now - timedelta(minutes=5),
    )

    mym_state2 = InstrumentState(cfg=mym_cfg, cname="MYMH6")
    mym_state2.sigma     = mym_sigma
    mym_state2.sigma_pts = mym_sp
    mym_state2.mean_vol  = 4_200.0
    mym_state2.bars = [Bar(ts=now - timedelta(minutes=5),
                           open=46_490, high=46_505, low=46_475,
                           close=46_490, volume=3_100)]

    console.print()
    console.print("[bold underline]DEMO — MYM: WATCHING  │  MES: SHORT SIGNAL[/]",
                  justify="center")
    console.print(render([mym_state2, mes_state2], history, now))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Show sample output without API calls")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_live()
