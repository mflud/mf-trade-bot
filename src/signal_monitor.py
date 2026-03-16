"""
Multi-Instrument Real-Time Signal Monitor.

Polls TopstepX every 30 seconds, builds the latest 5-min bar, and displays
side-by-side panels for each configured instrument:
  - Volatility regime (σ in bps/points, annualised vol, regime label)
  - Current bar status (OHLCV, scaled return, volume ratio)
  - Signal status: GREEN (long), RED (short), YELLOW (watching)
  - If signal: entry, target, stop in points + expiry countdown

Instruments and their optimal parameters (from backtest):
  MYM  Micro Dow      stop=2.0σ  target=3.0σ  $0.50/pt
  MES  Micro S&P 500  stop=2.0σ  target=3.0σ  $5.00/pt

Run modes:
  python src/signal_monitor.py          # live (requires .env credentials)
  python src/signal_monitor.py --demo   # static demo with synthetic data
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
TRAILING_BARS = 20    # 20 × 5-min = 100 min (optimal from signal_window_grid)
GK_VOL_BARS   = 20   # 20 × 5-min = 100-min window; Garman-Klass estimator
MOM_BARS      = 8    # 8 × 5-min = 40 min momentum window
CSR_THRESHOLD = 1.5  # CSR = Cumulative Scaled Return; min value aligned with signal direction
SIGNAL_SIGMA  = 3.0
MAX_SCALED    = 5.0   # ignore extreme event spikes above this
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 15
BARS_PER_YEAR = 252 * 23 * 60

# Time-of-day blackout windows (ET local time).  Signals are suppressed during these periods.
#   08:00–09:00 ET — economic data release window (e.g. 8:30 ET NFP/CPI): P(stop)>0.40, EV negative
ET = ZoneInfo("America/New_York")
BLACKOUT_WINDOWS_ET = [
    (8,  0, 9,  0),   # (start_h, start_m, end_h, end_m)  8:00–9:00 ET (DST-aware)
]

LOG_PATH = Path("logs/signals.csv")
LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "entry", "target", "stop",
    "sigma_pts", "scaled", "vol_ratio", "csr",
    "outcome", "pnl_pts", "pnl_sigma",
]

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
    InstrumentConfig("MYM", "MYM", stop_sigma=2.0, target_sigma=3.0,
                     point_value=0.50, ev_sigma=0.073),  # trail=20 pending MYM retest
    InstrumentConfig("MES", "MES", stop_sigma=2.0, target_sigma=3.0,
                     point_value=5.00, ev_sigma=0.073),  # trail=20, no filter, -2σ/+3σ
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
    csr:        float
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
    gk_ann_vol:    float = 0.0
    csr:           float = 0.0   # cumulative scaled return (40 min, direction-adjusted)
    mean_vol:           float | None = None
    active_signal:      Signal | None = None
    history:            list[RecentSignal] = field(default_factory=list)
    error:              str | None = None
    last_evaluated_ts:  datetime | None = None   # ts of last bar evaluated for signals


# ── Helpers ────────────────────────────────────────────────────────────────────

def annualised_vol(sigma: float) -> float:
    return sigma * math.sqrt(BARS_PER_YEAR / TF_MINUTES)


def gk_annualised_vol(bars: list) -> float:
    """Garman-Klass annualised vol from the last GK_VOL_BARS 5-min bars."""
    sample = bars[-GK_VOL_BARS:] if len(bars) >= GK_VOL_BARS else bars
    if len(sample) < 2:
        return float("nan")
    ln_hl = np.log(np.array([b.high / b.low   for b in sample]))
    ln_co = np.log(np.array([b.close / b.open for b in sample]))
    gk = 0.5 * ln_hl ** 2 - (2 * math.log(2) - 1) * ln_co ** 2
    var = float(np.mean(gk))
    return math.sqrt(var * BARS_PER_YEAR / TF_MINUTES) if var > 0 else float("nan")


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
    vol_ratio = (cur_vol / state.mean_vol) if state.mean_vol is not None else None

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=16)
    t.add_column(width=24)

    gk  = state.gk_ann_vol
    gk_label, gk_style = regime_label(gk) if gk > 0 else (label, style)

    t.add_row("σ per bar:",
              f"[bold]{sigma * 10000:.2f} bps[/]  │  {sigma_pts:.2f} pts")
    t.add_row("Ann. vol (CR):",
              f"{ann*100:.1f}%  [dim](close-return, 100 bars)[/]")
    t.add_row("Ann. vol (GK):",
              f"[bold]{gk*100:.1f}%[/]  [dim](Garman-Klass, 20 bars)[/]")
    t.add_row("Regime:",
              f"[bold {gk_style}]{gk_label}[/]  [dim](GK)[/]")
    t.add_row("Avg volume:",
              f"{state.mean_vol:,.0f}" if state.mean_vol is not None else "[dim]—[/]")
    t.add_row("Cur volume:",
              f"{cur_vol:,.0f}  ({vol_ratio:.1f}× avg)" if vol_ratio is not None else f"{cur_vol:,.0f}  [dim](warming up)[/]")

    return Panel(t, title="[bold]VOL REGIME[/]",
                 border_style="blue", padding=(0, 1))


def build_bar_panel(state: InstrumentState) -> Panel:
    bar   = state.bars[-1]
    sigma = state.sigma
    ret    = math.log(bar.close / bar.open) if bar.open else 0.0
    scaled = ret / sigma if sigma else 0.0
    vol_ratio = (bar.volume / state.mean_vol) if state.mean_vol is not None else None

    sc_style = ("green" if scaled > 0 else "red") if abs(scaled) >= SIGNAL_SIGMA \
               else ("yellow" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "white")
    vr_style = ("green" if vol_ratio >= VOL_RATIO_MIN else
                ("yellow" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "white")) if vol_ratio is not None else "dim"
    sc_check = "✓" if abs(scaled) >= SIGNAL_SIGMA else \
               ("~" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "✗")
    vr_check = ("✓" if vol_ratio >= VOL_RATIO_MIN else
                ("~" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "✗")) if vol_ratio is not None else "?"

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=10)
    t.add_column(width=30)

    t.add_row("Open:",   f"{bar.open:,.2f}")
    t.add_row("High:",   f"[green]{bar.high:,.2f}[/]")
    t.add_row("Low:",    f"[red]{bar.low:,.2f}[/]")
    t.add_row("Close:",  f"[bold]{bar.close:,.2f}[/]")
    t.add_row("Volume:", f"{bar.volume:,.0f}  [{vr_style}]{vol_ratio:.2f}× "
                         f"[thr {VOL_RATIO_MIN:.1f}×] {vr_check}[/]"
              if vol_ratio is not None else
              f"{bar.volume:,.0f}  [dim](warming up)[/]")
    val_style = "bold green" if scaled > 0 else "bold red"
    t.add_row("Scaled:", f"[{val_style}]{scaled:+.2f}σ[/]  "
                         f"[{sc_style}][thr {SIGNAL_SIGMA:.0f}σ] {sc_check}[/]")

    return Panel(t, title=f"[bold]LAST {TF_MINUTES}-MIN BAR[/]",
                 border_style="blue", padding=(0, 1))


def build_signal_panel(state: InstrumentState, now: datetime) -> Panel:
    signal = state.active_signal
    cfg    = state.cfg

    if signal is None:
        bar    = state.bars[-1]
        sigma  = state.sigma
        scaled = math.log(bar.close / bar.open) / sigma if sigma and bar.open else 0.0
        vol_ratio = (bar.volume / state.mean_vol) if state.mean_vol is not None else None
        sc_pct = min(abs(scaled) / SIGNAL_SIGMA * 100, 100)
        vr_pct = min(vol_ratio   / VOL_RATIO_MIN * 100, 100) if vol_ratio is not None else 0.0
        bar_sc = "█" * int(sc_pct / 5) + "░" * (20 - int(sc_pct / 5))
        bar_vr = "█" * int(vr_pct / 5) + "░" * (20 - int(vr_pct / 5))

        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", width=16)
        t.add_column()
        csr       = state.csr
        csr_pct   = min(abs(csr) / CSR_THRESHOLD * 100, 100)
        bar_csr   = "█" * int(csr_pct / 5) + "░" * (20 - int(csr_pct / 5))
        csr_style = "green" if csr >= CSR_THRESHOLD else \
                    ("yellow" if csr > 0 else "red")
        csr_check = "✓" if csr >= CSR_THRESHOLD else \
                    ("~" if csr > 0 else "✗")

        t.add_row("Scaled return:",
                  f"[yellow]{bar_sc}[/] {abs(scaled):.2f}σ / {SIGNAL_SIGMA:.0f}σ")
        t.add_row("Volume ratio:",
                  f"[yellow]{bar_vr}[/] {vol_ratio:.2f}× / {VOL_RATIO_MIN:.1f}×"
                  if vol_ratio is not None else
                  f"[dim]{bar_vr}[/] warming up")
        t.add_row("Momentum(40m):",
                  f"[{csr_style}]{bar_csr}[/] {csr:+.2f}σ / {CSR_THRESHOLD:.1f}σ {csr_check}")
        bar_et = state.bars[-1].ts.astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        in_blackout = any(
            (sh, sm) <= bar_hm < (eh, em)
            for sh, sm, eh, em in BLACKOUT_WINDOWS_ET
        )
        # Show BLACKOUT only when in the restricted window AND CSR doesn't lift it
        blackout_note = ("  [bold red]BLACKOUT[/]"
                         if in_blackout and csr < CSR_THRESHOLD else "")
        t.add_row("", f"[dim]Need ≥{SIGNAL_SIGMA:.0f}σ + ≥{VOL_RATIO_MIN:.1f}× vol + CSR≥{CSR_THRESHOLD:.1f}σ[/]{blackout_note}")

        # NOTRADE: show indicative SL/TP based on current price and σ
        price     = bar.close
        sigma_pts = state.sigma_pts
        nt = Table.grid(padding=(0, 1))
        nt.add_column(style="dim", width=10)
        nt.add_column()
        if sigma_pts > 0:
            long_tgt  = price + cfg.target_sigma * sigma_pts
            long_stop = price - cfg.stop_sigma   * sigma_pts
            shrt_tgt  = price - cfg.target_sigma * sigma_pts
            shrt_stop = price + cfg.stop_sigma   * sigma_pts
            nt.add_row("Price:",  f"[dim]{price:,.2f}[/]")
            nt.add_row("LONG:",   f"[dim]tgt {long_tgt:,.2f}  /  sl {long_stop:,.2f}[/]")
            nt.add_row("SHORT:",  f"[dim]tgt {shrt_tgt:,.2f}  /  sl {shrt_stop:,.2f}[/]")
        else:
            nt.add_row("", "[dim]warming up[/]")

        watching_panel = Panel(t,  title="[bold yellow]⬤  WATCHING[/]",
                               border_style="yellow", padding=(0, 2))
        notrade_panel  = Panel(nt, title="[dim]NOTRADE[/]",
                               border_style="dim",    padding=(0, 2))

        col = Table.grid()
        col.add_column()
        col.add_row(watching_panel)
        col.add_row(notrade_panel)
        return col

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
    if not state.bars:
        msg = f"[red]{state.error}[/]" if state.error else "[dim]Waiting for bars…[/]"
        col.add_row(Panel(msg, border_style="dim", padding=(0, 1)))
        return col
    if state.error:
        col.add_row(Panel(f"[yellow]⚠ {state.error} — showing last known data[/]",
                          border_style="yellow", padding=(0, 1)))
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


# ── Trade logging ──────────────────────────────────────────────────────────────

def _ensure_log():
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def _log_trade(sym: str, sig: Signal, outcome: str,
               pnl_pts: float, resolved_at: datetime):
    row = {
        "fired_at":    sig.bar_ts.isoformat(),
        "resolved_at": resolved_at.isoformat(),
        "symbol":      sym,
        "direction":   "LONG" if sig.direction == 1 else "SHORT",
        "entry":       round(sig.entry,      4),
        "target":      round(sig.target,     4),
        "stop":        round(sig.stop,       4),
        "sigma_pts":   round(sig.sigma_pts,  4),
        "scaled":      round(sig.scaled,     4),
        "vol_ratio":   round(sig.vol_ratio,  4),
        "csr":         round(sig.csr,        4),
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts,        4),
        "pnl_sigma":   round(pnl_pts / sig.sigma_pts, 4) if sig.sigma_pts else 0.0,
    }
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def _check_resolution(sig: Signal, bars: list[Bar]) -> tuple[str, float] | None:
    """Scan bars after the signal bar for target/stop hit. Returns (outcome, pnl_pts) or None."""
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if sig.direction == 1:
            if bar.high >= sig.target:
                return "TARGET",  sig.target_pts()
            if bar.low  <= sig.stop:
                return "STOPPED", -sig.stop_pts()
        else:
            if bar.low  <= sig.target:
                return "TARGET",  sig.target_pts()
            if bar.high >= sig.stop:
                return "STOPPED", -sig.stop_pts()
    return None


# ── Live mode ──────────────────────────────────────────────────────────────────

def evaluate(state: InstrumentState) -> Signal | None:
    bars    = state.bars
    closes  = np.array([b.close  for b in bars])
    volumes = np.array([b.volume for b in bars])

    trail = np.log(closes[1:] / closes[:-1])[-TRAILING_BARS:] \
            if len(closes) >= 2 else np.array([])
    sigma     = float(np.std(trail, ddof=1)) if len(trail) >= 2 else 0.0
    sigma_pts = sigma * closes[-1]
    warmed_up = len(closes) > TRAILING_BARS   # full window required for signals
    prior_vols = volumes[-TRAILING_BARS - 1:-1]
    active_vols = prior_vols[prior_vols >= 10]
    mean_vol = float(np.median(active_vols)) if len(active_vols) >= 10 else None

    state.sigma     = sigma
    state.sigma_pts = sigma_pts
    state.mean_vol  = mean_vol
    state.gk_ann_vol = gk_annualised_vol(bars)

    last      = bars[-1]
    bar_ret   = math.log(last.close / last.open) if last.open else 0.0
    scaled    = bar_ret / sigma if sigma else 0.0
    vol_ratio = (last.volume / mean_vol) if mean_vol is not None else None

    # 40-min cumulative scaled return (direction-adjusted; >CSR_THRESHOLD = with momentum)
    direction = 1 if scaled > 0 else -1
    if len(closes) >= MOM_BARS + 1:
        mom_rets = np.log(closes[-MOM_BARS:] / closes[-MOM_BARS - 1:-1])
        state.csr = float(mom_rets.sum()) / sigma * direction if sigma else 0.0
    else:
        state.csr = 0.0

    # Conditional blackout: 08:00–09:00 ET only blocks if CSR < threshold.
    # CSR≥threshold is already required below, so no separate gate needed here.
    if (warmed_up
            and abs(scaled) >= SIGNAL_SIGMA and abs(scaled) <= MAX_SCALED
            and vol_ratio is not None and vol_ratio >= VOL_RATIO_MIN
            and state.csr >= CSR_THRESHOLD):
        return Signal(cfg=state.cfg,
                      direction=1 if scaled > 0 else -1,
                      entry=last.close, sigma=sigma, sigma_pts=sigma_pts,
                      scaled=scaled, vol_ratio=vol_ratio, csr=state.csr,
                      bar_ts=last.ts)
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
    _ensure_log()

    def fetch_bars(state: InstrumentState):
        end   = datetime.now(timezone.utc)
        lookback = TRAILING_BARS + MOM_BARS + 10
        start = end - timedelta(minutes=TF_MINUTES * lookback)
        try:
            raw = client.get_bars(contract_id=state.cid, start=start, end=end,
                                  unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
                                  limit=lookback)
            raw = list(reversed(raw))
            state.bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                              open=b["o"], high=b["h"], low=b["l"],
                              close=b["c"], volume=b["v"]) for b in raw]
            state.error = None
        except Exception as e:
            state.error = f"fetch error: {e}"

    def resolve(state: InstrumentState, outcome: str, pnl_pts: float, now: datetime):
        sig = state.active_signal
        combined_history.append(RecentSignal(state.cfg.symbol, sig, outcome, pnl_pts))
        _log_trade(state.cfg.symbol, sig, outcome, pnl_pts, now)
        state.active_signal = None

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            now = datetime.now(timezone.utc)

            for state in states:
                fetch_bars(state)
                if not state.bars:
                    continue

                # Update display metrics and check for signal
                new_sig = evaluate(state)

                # Only act on a signal from a bar we haven't seen before
                last_bar_ts = state.bars[-1].ts
                if last_bar_ts == state.last_evaluated_ts:
                    new_sig = None
                else:
                    state.last_evaluated_ts = last_bar_ts

                # Check active signal for target/stop hit or expiry
                if state.active_signal:
                    hit = _check_resolution(state.active_signal, state.bars)
                    if hit:
                        resolve(state, hit[0], hit[1], now)
                    elif now >= state.active_signal.expires_at:
                        last_close = state.bars[-1].close
                        pnl = (last_close - state.active_signal.entry) * state.active_signal.direction
                        resolve(state, "TIME EXIT", pnl, now)

                if new_sig and (state.active_signal is None or
                                new_sig.bar_ts != state.active_signal.bar_ts):
                    if state.active_signal:
                        resolve(state, "SUPERSEDED", 0.0, now)
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
    mym_state.gk_ann_vol = 0.198   # ~19.8% GK vol (slightly above close-return)
    mym_state.mean_vol  = 4_200.0
    mym_state.bars = [Bar(ts=now - timedelta(minutes=5),
                          open=46_430, high=46_560, low=46_415,
                          close=46_500, volume=9_800)]
    mym_state.active_signal = Signal(
        cfg=mym_cfg, direction=1, entry=46_500,
        sigma=mym_sigma, sigma_pts=mym_sp,
        scaled=+3.91, vol_ratio=2.33, csr=1.82, bar_ts=now - timedelta(minutes=5),
    )

    # MES synthetic state — watching
    mes_cfg   = INSTRUMENTS[1]
    mes_sigma = 0.000721
    mes_price = 6_625.0
    mes_sp    = mes_sigma * mes_price   # ≈ 4.78 pts per 1σ

    mes_state = InstrumentState(cfg=mes_cfg, cname="MESH6")
    mes_state.sigma     = mes_sigma
    mes_state.sigma_pts = mes_sp
    mes_state.gk_ann_vol = 0.198
    mes_state.mean_vol  = 8_450.0
    mes_state.bars = [Bar(ts=now - timedelta(minutes=5),
                          open=6_622.0, high=6_627.5, low=6_620.5,
                          close=6_625.0, volume=6_820)]

    history = [
        RecentSignal("MYM", Signal(mym_cfg, -1, 46_550, mym_sigma, mym_sp,
                                   -3.5, 2.1, 1.91, now - timedelta(minutes=75)),
                     "TARGET",    mym_sp * 2.5),
        RecentSignal("MES", Signal(mes_cfg, +1, 6_610, mes_sigma, mes_sp,
                                   +4.1, 1.9, 2.04, now - timedelta(minutes=130)),
                     "STOPPED",  -mes_sp * 1.5),
        RecentSignal("MYM", Signal(mym_cfg, +1, 46_380, mym_sigma, mym_sp,
                                   +3.3, 1.7, 1.63, now - timedelta(minutes=215)),
                     "TIME EXIT", mym_sp * 0.6),
        RecentSignal("MES", Signal(mes_cfg, -1, 6_645, mes_sigma, mes_sp,
                                   -3.8, 2.4, 1.55, now - timedelta(minutes=280)),
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
    mes_state2.gk_ann_vol = 0.198
    mes_state2.mean_vol  = 8_450.0
    mes_state2.bars = [Bar(ts=now - timedelta(minutes=5),
                           open=6_640.0, high=6_641.0, low=6_618.5,
                           close=6_620.0, volume=21_800)]
    mes_state2.active_signal = Signal(
        cfg=mes_cfg, direction=-1, entry=6_620.0,
        sigma=mes_sigma, sigma_pts=mes_sp,
        scaled=-4.12, vol_ratio=2.58, csr=2.17, bar_ts=now - timedelta(minutes=5),
    )

    mym_state2 = InstrumentState(cfg=mym_cfg, cname="MYMH6")
    mym_state2.sigma     = mym_sigma
    mym_state2.sigma_pts = mym_sp
    mym_state2.gk_ann_vol = 0.198
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
