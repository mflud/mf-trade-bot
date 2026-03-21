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
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
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
MOM_BARS      = 8    # 8 × 5-min = 40 min momentum window (default; overridden dynamically)
CSR_THRESHOLD = 1.5  # CSR = Cumulative Scaled Return; min value aligned with signal direction
SIGNAL_SIGMA  = 3.0
MAX_SCALED    = 5.0   # ignore extreme event spikes above this
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 25
BARS_PER_YEAR = 252 * 23 * 60

PL_N_BARS = 10    # 1-min bars to look back for PL computation
PL_THRESH = 0.50  # PL_aligned ≥ this → 2× sizing

ET = ZoneInfo("America/New_York")

# ORB parameters (15-min ORB, wide-range LONG, morning + power-hour windows)
ORB_BARS      = 3          # 3 × 5-min = 15-min opening range
ORB_WIDTH_MIN = 15.25      # pts — wide tertile cutoff from backtest
ORB_STOP_SIG  = 2.0
ORB_TGT_SIG   = 2.0        # 2σ:2σ → EV ≈ +0.61R
ORB_WINDOWS   = [          # (start_h, start_m, end_h, end_m, label)
    (9,  45, 10, 30, "Morning"),
    (13, 30, 16,  0, "Power hr"),
]

LOG_PATH = Path("logs/signals.csv")
ORB_LOG_PATH = Path("logs/orb_signals.csv")
ORB_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "entry", "target", "stop",
    "orb_high", "orb_low", "orb_width", "sigma_pts",
    "window", "outcome", "pnl_pts", "pnl_r",
]
LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "entry", "target", "stop",
    "sigma_pts", "scaled", "vol_ratio", "csr",
    "pl_aligned", "contracts",
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
    # Dynamic CSR window: list of (gk_ann_vol_upper_bound, mom_bars).
    # First entry whose upper bound exceeds current GK vol is used.
    csr_vol_windows: list = field(default_factory=lambda: [(1.0, 8)])
    # Per-instrument blackout windows: (start_h, start_m, end_h, end_m, conditional).
    # conditional=True: block only when CSR < threshold; False: always block.
    blackout_windows: list = field(default_factory=list)


INSTRUMENTS = [
    InstrumentConfig("MES", "MES", stop_sigma=2.0, target_sigma=3.0,
                     point_value=5.00, ev_sigma=0.073,
                     csr_vol_windows=[(0.08, 4), (1.0, 8)],
                     blackout_windows=[
                         (8, 0, 9, 0, True),  # econ releases: block only if CSR<1.5
                     ]),
    InstrumentConfig("MYM", "MYM", stop_sigma=2.0, target_sigma=3.0,
                     point_value=0.50, ev_sigma=0.073,
                     csr_vol_windows=[(0.08, 4), (1.0, 8)],
                     blackout_windows=[
                         (9,  0,  9, 30, False),  # pre-open: EV=-0.076σ CSR-filtered
                         (15, 0, 16,  0, False),  # NYSE close: EV=-0.375σ CSR-filtered
                     ]),
]

ALERT_SOUND = "/System/Library/Sounds/Ping.aiff"


def play_alert():
    """Play alert sound in background thread (non-blocking)."""
    threading.Thread(
        target=lambda: subprocess.run(["afplay", ALERT_SOUND], check=False),
        daemon=True,
    ).start()


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
    pl_aligned: float | None = None   # set after signal fires; drives 2× sizing
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
class OrbSignal:
    entry:      float
    target:     float
    stop:       float
    orb_high:   float
    orb_low:    float
    sigma_pts:  float
    window:     str
    bar_ts:     datetime

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)
    def risk_pts(self):   return self.stop_pts()


@dataclass
class OrbState:
    session_date:    date | None = None
    orb_high:        float = 0.0
    orb_low:         float = 0.0
    orb_bars_seen:   int   = 0
    orb_complete:    bool  = False
    morning_fired:   bool  = False
    power_hr_fired:  bool  = False
    active_signal:   OrbSignal | None = None


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
    current_pl:         float | None = None      # raw 1-min PL, refreshed every poll
    orb:                OrbState = field(default_factory=OrbState)
    history:            list[RecentSignal] = field(default_factory=list)
    error:              str | None = None
    last_evaluated_ts:  datetime | None = None   # ts of last bar evaluated for signals


# ── Helpers ────────────────────────────────────────────────────────────────────

def annualised_vol(sigma: float) -> float:
    return sigma * math.sqrt(BARS_PER_YEAR / TF_MINUTES)


def get_mom_bars(gk_ann_vol: float, csr_vol_windows: list) -> int:
    """Return the CSR window (in bars) for the current GK vol regime."""
    for upper, bars in csr_vol_windows:
        if gk_ann_vol < upper:
            return bars
    return csr_vol_windows[-1][1]


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


def _pl_bar(pl: float, width: int = 20) -> str:
    """
    Render a signed [-1, +1] bar with:
      - red ░░░ fill for the [-1, -0.5] region (left quarter)
      - dim ─── fill for the [-0.5, +0.5] neutral region
      - green ░░░ fill for the [+0.5, +1] region (right quarter)
      - dim │ at the centre (pl=0 reference)
      - bold █ marker at current pl position, coloured by region
    """
    marker = min(width - 1, int((pl + 1) / 2 * width))
    center = width // 2
    parts  = []
    for i in range(width):
        in_red   = i < width // 4
        in_green = i >= width - width // 4
        if i == marker:
            style = "bold red" if in_red else ("bold green" if in_green else "bold white")
            parts.append(f"[{style}]█[/]")
        elif i == center:
            parts.append("[dim]│[/]")
        elif in_red:
            parts.append("[red]░[/]")
        elif in_green:
            parts.append("[green]░[/]")
        else:
            parts.append("[dim]─[/]")
    return "".join(parts)


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
    t.add_column(width=34)

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
        mom_bars = get_mom_bars(state.gk_ann_vol, state.cfg.csr_vol_windows)
        t.add_row(f"Momentum({mom_bars * TF_MINUTES}m):",
                  f"[{csr_style}]{bar_csr}[/] {csr:+.2f}σ / {CSR_THRESHOLD:.1f}σ {csr_check}")

        pl = state.current_pl
        if pl is not None:
            val_style = "green" if pl >= PL_THRESH else ("red" if pl <= -PL_THRESH else "white")
            t.add_row("1-min PL:",
                      f"{_pl_bar(pl)} [{val_style}]{pl:+.2f}[/]")
        else:
            t.add_row("1-min PL:", "[dim]fetching…[/]")

        bar_et = state.bars[-1].ts.astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        in_active_blackout = any(
            (sh, sm) <= bar_hm < (eh, em) and (not conditional or csr < CSR_THRESHOLD)
            for sh, sm, eh, em, conditional in state.cfg.blackout_windows
        )
        blackout_note = "  [bold red]BLACKOUT[/]" if in_active_blackout else ""
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
            nt.add_row("LONG:",   f"[dim]tgt {long_tgt:,.2f}  /  sl  {long_stop:,.2f}[/]")
            nt.add_row("SHORT:",  f"[dim]tgt {shrt_tgt:,.2f}  /  cs  {shrt_stop:,.2f}[/]")
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

    sizing_2x = signal.pl_aligned is not None and signal.pl_aligned >= PL_THRESH
    if sizing_2x:
        pl_str = f"{signal.pl_aligned:+.2f}"
        t.add_row("", "")
        t.add_row("Size:",
                  f"[bold yellow on dark_red]  ⚡ 2× CONTRACTS  PL={pl_str}  [/]")
    elif signal.pl_aligned is not None:
        t.add_row("PL:",
                  f"[dim]{signal.pl_aligned:+.2f} (1× size)[/]")

    border = "yellow" if sizing_2x else color
    return Panel(t,
                 title=f"[bold {color}]⬤  {direction_str}  SIGNAL[/]",
                 border_style=border, padding=(0, 2))


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
    if state.cfg.symbol == "MES":
        col.add_row(build_orb_panel(state, now))
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
    t.append("  MES & MYM  ", style="bold cyan")
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
        "pl_aligned":  round(sig.pl_aligned, 4) if sig.pl_aligned is not None else "",
        "contracts":   2 if (sig.pl_aligned is not None and sig.pl_aligned >= PL_THRESH) else 1,
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


# ── ORB evaluation ─────────────────────────────────────────────────────────────

def _orb_window(bar_et: datetime) -> str | None:
    hm = (bar_et.hour, bar_et.minute)
    for sh, sm, eh, em, label in ORB_WINDOWS:
        if (sh, sm) <= hm < (eh, em):
            return label
    return None


def evaluate_orb(state: InstrumentState) -> OrbSignal | None:
    """Update OrbState incrementally; return a new OrbSignal on qualifying breakout."""
    if not state.bars:
        return None

    bar    = state.bars[-1]
    bar_et = bar.ts.astimezone(ET)
    today  = bar_et.date()
    orb    = state.orb

    if orb.session_date != today:
        orb.session_date   = today
        orb.orb_high       = 0.0
        orb.orb_low        = 0.0
        orb.orb_bars_seen  = 0
        orb.orb_complete   = False
        orb.morning_fired  = False
        orb.power_hr_fired = False
        orb.active_signal  = None

    hm = (bar_et.hour, bar_et.minute)

    if (9, 30) <= hm < (9, 30 + ORB_BARS * TF_MINUTES) and not orb.orb_complete:
        if orb.orb_bars_seen == 0:
            orb.orb_high = bar.high
            orb.orb_low  = bar.low
        else:
            orb.orb_high = max(orb.orb_high, bar.high)
            orb.orb_low  = min(orb.orb_low,  bar.low)
        orb.orb_bars_seen += 1
        if orb.orb_bars_seen >= ORB_BARS:
            orb.orb_complete = True
        return None

    if not orb.orb_complete:
        return None
    if hm < (9, 30) or hm >= (16, 0):
        return None

    window = _orb_window(bar_et)
    if window is None:
        return None
    if window == "Morning"  and orb.morning_fired:
        return None
    if window == "Power hr" and orb.power_hr_fired:
        return None

    orb_width = orb.orb_high - orb.orb_low
    if orb_width < ORB_WIDTH_MIN:
        return None
    if state.sigma_pts <= 0:
        return None

    if bar.close > orb.orb_high:
        entry  = bar.close
        sig = OrbSignal(
            entry=entry,
            target=entry + ORB_TGT_SIG * state.sigma_pts,
            stop=entry   - ORB_STOP_SIG * state.sigma_pts,
            orb_high=orb.orb_high, orb_low=orb.orb_low,
            sigma_pts=state.sigma_pts, window=window, bar_ts=bar.ts,
        )
        if window == "Morning":
            orb.morning_fired  = True
        else:
            orb.power_hr_fired = True
        orb.active_signal = sig
        return sig

    return None


def _check_orb_resolution(sig: OrbSignal, bars: list[Bar]) -> tuple[str, float] | None:
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if bar.high >= sig.target:
            return "TARGET",  sig.target_pts()
        if bar.low  <= sig.stop:
            return "STOPPED", -sig.stop_pts()
    return None


def build_orb_panel(state: InstrumentState, now: datetime) -> Panel:
    orb    = state.orb
    bar_et = state.bars[-1].ts.astimezone(ET) if state.bars else None

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=14)
    t.add_column()

    if orb.orb_complete:
        width   = orb.orb_high - orb.orb_low
        w_style = "green" if width >= ORB_WIDTH_MIN else "red"
        w_flag  = " ✓" if width >= ORB_WIDTH_MIN else f" ✗ need>{ORB_WIDTH_MIN:.0f}"
        t.add_row("ORB high:", f"{orb.orb_high:,.2f}")
        t.add_row("ORB low:",  f"{orb.orb_low:,.2f}")
        t.add_row("ORB width:", f"[{w_style}]{width:.2f} pts{w_flag}[/]")
    elif orb.session_date == (bar_et.date() if bar_et else None):
        t.add_row("ORB:", f"[yellow]Building… {orb.orb_bars_seen}/{ORB_BARS} bars[/]")
    else:
        t.add_row("ORB:", "[dim]Waiting for RTH open[/]")

    if orb.active_signal:
        sig   = orb.active_signal
        rem   = (sig.bar_ts + timedelta(minutes=MAX_HOLD_MIN)) - now
        rem_s = int(rem.total_seconds())
        rem_str = f"{rem_s // 60}m {rem_s % 60:02d}s" if rem_s > 0 else "[blink]EXPIRED[/]"
        t.add_row("", "")
        t.add_row("Entry:",  f"[bold]{sig.entry:,.2f}[/]  [dim]({sig.window})[/]")
        t.add_row("Target:", f"[bold green]{sig.target:,.2f}[/]  "
                              f"([green]+{sig.target_pts():.2f} pts[/] │ +{ORB_TGT_SIG:.1f}σ)")
        t.add_row("Stop:",   f"[bold red]{sig.stop:,.2f}[/]  "
                              f"([red]−{sig.stop_pts():.2f} pts[/] │ −{ORB_STOP_SIG:.1f}σ)")
        t.add_row("EV:",     f"+0.61R ≈ +{0.61*sig.risk_pts():.1f} pts  [{rem_str}]")
        return Panel(t, title="[bold green]⬤  ORB LONG ▲[/]",
                     border_style="green", padding=(0, 1))

    if orb.orb_complete:
        width  = orb.orb_high - orb.orb_low
        window = _orb_window(bar_et) if bar_et else None
        if width < ORB_WIDTH_MIN:
            status = "[dim]ORB too narrow[/]"
        elif window:
            fired = (window == "Morning" and orb.morning_fired) or \
                    (window == "Power hr" and orb.power_hr_fired)
            status = "[dim]Already fired[/]" if fired else \
                     f"[yellow]Watch >{orb.orb_high:.2f}[/]"
        else:
            remaining = [l for sh, sm, eh, em, l in ORB_WINDOWS
                         if not ((l == "Morning" and orb.morning_fired) or
                                 (l == "Power hr" and orb.power_hr_fired))]
            status = f"[dim]Next: {remaining[0]}[/]" if remaining else "[dim]Done today[/]"
        t.add_row("Status:", status)

    border = "yellow" if (orb.orb_complete and _orb_window(bar_et) and
                          not (orb.morning_fired and orb.power_hr_fired)) else "dim"
    return Panel(t, title="[bold]ORB LONG[/]", border_style=border, padding=(0, 1))


def _log_orb(sym: str, sig: OrbSignal, outcome: str,
             pnl_pts: float, resolved_at: datetime):
    ORB_LOG_PATH.parent.mkdir(exist_ok=True)
    if not ORB_LOG_PATH.exists():
        with open(ORB_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writeheader()
    row = {
        "fired_at": sig.bar_ts.isoformat(), "resolved_at": resolved_at.isoformat(),
        "symbol": sym, "direction": "LONG",
        "entry": round(sig.entry, 4), "target": round(sig.target, 4),
        "stop": round(sig.stop, 4),
        "orb_high": round(sig.orb_high, 4), "orb_low": round(sig.orb_low, 4),
        "orb_width": round(sig.orb_high - sig.orb_low, 4),
        "sigma_pts": round(sig.sigma_pts, 4), "window": sig.window,
        "outcome": outcome, "pnl_pts": round(pnl_pts, 4),
        "pnl_r": round(pnl_pts / sig.risk_pts(), 4) if sig.risk_pts() else 0.0,
    }
    with open(ORB_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writerow(row)


# ── PL confidence sizing ───────────────────────────────────────────────────────

def fetch_1min_pl(client, contract_id: str,
                  signal_bar_ts: datetime, direction: int) -> float | None:
    """
    Fetch PL_N_BARS 1-min bars ending just before the signal 5-min bar and
    return PL_aligned = (signed path length) × direction.
    +1 = 1-min flow perfectly aligned; ≥ PL_THRESH → 2× sizing.
    Returns None on fetch error or insufficient data.
    """
    from topstep_client import TopstepClient
    end   = signal_bar_ts
    start = end - timedelta(minutes=PL_N_BARS + 5)
    try:
        raw = client.get_bars(
            contract_id=contract_id, start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=PL_N_BARS + 5,
        )
        raw = list(reversed(raw))
    except Exception:
        return None
    if len(raw) < PL_N_BARS + 1:
        return None
    closes   = np.array([b["c"] for b in raw[-(PL_N_BARS + 1):]])
    rets     = np.log(closes[1:] / closes[:-1])
    sum_absr = float(np.abs(rets).sum())
    if sum_absr == 0:
        return None
    pl = float(rets.sum()) / sum_absr
    return pl * direction


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

    # Dynamic CSR window based on current GK vol regime
    direction = 1 if scaled > 0 else -1
    mom_bars  = get_mom_bars(state.gk_ann_vol, state.cfg.csr_vol_windows)
    if len(closes) >= mom_bars + 1:
        mom_rets  = np.log(closes[-mom_bars:] / closes[-mom_bars - 1:-1])
        state.csr = float(mom_rets.sum()) / sigma * direction if sigma else 0.0
    else:
        state.csr = 0.0

    # Per-instrument blackout windows.
    bar_et = last.ts.astimezone(ET)
    bar_hm = (bar_et.hour, bar_et.minute)
    for sh, sm, eh, em, conditional in state.cfg.blackout_windows:
        if (sh, sm) <= bar_hm < (eh, em):
            if not conditional or state.csr < CSR_THRESHOLD:
                return None

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
        max_mom  = max(bars for cfg in INSTRUMENTS for _, bars in cfg.csr_vol_windows)
        lookback = TRAILING_BARS + max_mom + 10
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

                # Refresh 1-min PL for watching display
                state.current_pl = fetch_1min_pl(client, state.cid, now, 1)

                # Update display metrics and check for signal
                new_sig = evaluate(state)
                new_orb = evaluate_orb(state) if state.cfg.symbol == "MES" else None

                # Only act on a signal from a bar we haven't seen before
                last_bar_ts = state.bars[-1].ts
                if last_bar_ts == state.last_evaluated_ts:
                    new_sig = None
                    new_orb = None
                else:
                    state.last_evaluated_ts = last_bar_ts

                # Check momentum signal for target/stop hit or expiry
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
                    pl = fetch_1min_pl(client, state.cid,
                                       new_sig.bar_ts, new_sig.direction)
                    if pl is not None:
                        state.active_signal.pl_aligned = pl
                    play_alert()

                # Check ORB signal for target/stop hit or expiry (MES only)
                if state.orb.active_signal:
                    hit = _check_orb_resolution(state.orb.active_signal, state.bars)
                    if hit:
                        _log_orb(state.cfg.symbol, state.orb.active_signal,
                                 hit[0], hit[1], now)
                        state.orb.active_signal = None
                    elif now >= state.orb.active_signal.bar_ts + timedelta(minutes=MAX_HOLD_MIN):
                        pnl = state.bars[-1].close - state.orb.active_signal.entry
                        _log_orb(state.cfg.symbol, state.orb.active_signal,
                                 "TIME EXIT", pnl, now)
                        state.orb.active_signal = None

                if new_orb and state.orb.active_signal is None:
                    state.orb.active_signal = new_orb
                    play_alert()

            live.update(render(states, combined_history, now))
            time.sleep(30)


# ── Demo mode ──────────────────────────────────────────────────────────────────

def run_demo():
    now = datetime.now(timezone.utc)

    # MES synthetic state — watching, with completed wide ORB
    mes_cfg   = INSTRUMENTS[0]
    mes_sigma = 0.000721
    mes_price = 6_625.0
    mes_sp    = mes_sigma * mes_price   # ≈ 4.78 pts per 1σ
    now_et    = now.astimezone(ET)

    mes_state = InstrumentState(cfg=mes_cfg, cname="MESH6")
    mes_state.sigma      = mes_sigma
    mes_state.sigma_pts  = mes_sp
    mes_state.gk_ann_vol = 0.198
    mes_state.mean_vol   = 8_450.0
    mes_state.bars = [Bar(ts=now - timedelta(minutes=5),
                          open=6_622.0, high=6_627.5, low=6_620.5,
                          close=6_625.0, volume=6_820)]
    mes_state.orb.session_date  = now_et.date()
    mes_state.orb.orb_high      = 6_618.0
    mes_state.orb.orb_low       = 6_594.0   # width = 24 pts ✓
    mes_state.orb.orb_complete  = True

    # MYM synthetic state — long signal active
    mym_cfg   = INSTRUMENTS[1]
    mym_sigma = 0.000721
    mym_price = 46_500.0
    mym_sp    = mym_sigma * mym_price   # ≈ 33.5 pts per 1σ

    mym_state = InstrumentState(cfg=mym_cfg, cname="MYMH6")
    mym_state.sigma      = mym_sigma
    mym_state.sigma_pts  = mym_sp
    mym_state.gk_ann_vol = 0.198
    mym_state.mean_vol   = 4_200.0
    mym_state.bars = [Bar(ts=now - timedelta(minutes=5),
                          open=46_430, high=46_560, low=46_415,
                          close=46_500, volume=9_800)]
    mym_state.active_signal = Signal(
        cfg=mym_cfg, direction=1, entry=46_500,
        sigma=mym_sigma, sigma_pts=mym_sp,
        scaled=+3.91, vol_ratio=2.33, csr=1.82, bar_ts=now - timedelta(minutes=5),
    )

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
    console.print("[bold underline]DEMO — MES: WATCHING (ORB ready)  │  MYM: LONG SIGNAL[/]",
                  justify="center")
    console.print(render([mes_state, mym_state], history, now))

    # Frame 2: MES short signal, MYM watching
    mes_state2 = InstrumentState(cfg=mes_cfg, cname="MESH6")
    mes_state2.sigma      = mes_sigma
    mes_state2.sigma_pts  = mes_sp
    mes_state2.gk_ann_vol = 0.198
    mes_state2.mean_vol   = 8_450.0
    mes_state2.bars = [Bar(ts=now - timedelta(minutes=5),
                           open=6_640.0, high=6_641.0, low=6_618.5,
                           close=6_620.0, volume=21_800)]
    mes_state2.active_signal = Signal(
        cfg=mes_cfg, direction=-1, entry=6_620.0,
        sigma=mes_sigma, sigma_pts=mes_sp,
        scaled=-4.12, vol_ratio=2.58, csr=2.17, bar_ts=now - timedelta(minutes=5),
    )
    mes_state2.orb.session_date = now_et.date()
    mes_state2.orb.orb_high     = 6_635.0
    mes_state2.orb.orb_low      = 6_611.0   # width = 24 pts ✓
    mes_state2.orb.orb_complete = True

    mym_state2 = InstrumentState(cfg=mym_cfg, cname="MYMH6")
    mym_state2.sigma      = mym_sigma
    mym_state2.sigma_pts  = mym_sp
    mym_state2.gk_ann_vol = 0.198
    mym_state2.mean_vol   = 4_200.0
    mym_state2.bars = [Bar(ts=now - timedelta(minutes=5),
                           open=46_490, high=46_505, low=46_475,
                           close=46_490, volume=3_100)]

    console.print()
    console.print("[bold underline]DEMO — MES: SHORT SIGNAL (ORB ready)  │  MYM: WATCHING[/]",
                  justify="center")
    console.print(render([mes_state2, mym_state2], history, now))


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
