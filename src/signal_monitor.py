"""
Multi-Instrument Real-Time Signal Monitor.

Polls TopstepX every 30 seconds, builds the latest 5-min bar, and displays
side-by-side panels for each configured instrument:
  - Volatility regime (σ in bps/points, annualised vol, regime label)
  - Current bar status (OHLCV, scaled return, volume ratio)
  - Signal status: GREEN (long), RED (short), YELLOW (watching)
  - If signal: entry, target, stop in points + expiry countdown

Instruments and their optimal parameters (from backtest):
  MYM  Micro Dow          stop=2.0σ  target=3.0σ  $0.50/pt
  MES  Micro S&P 500      stop=2.0σ  target=3.0σ  $5.00/pt
  M2K  Micro Russell 2000 stop=2.0σ  target=3.0σ  $5.00/pt

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
from concurrent.futures import ThreadPoolExecutor
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

from trade_summary_panel import build_trade_summary_panel

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

# Trailing stop sigmas (keep in sync with trading_bot.py)
CSR_TRAIL_SIGMA    = 0.5

# VWASLR parameters (Volume-Weighted Average Scaled Log Return)
# 1-min bars: N=50 (50-min signal window), σ=500 bars (500-min σ window)
# Entry: EMA-10 crosses ±threshold.  Exit: EMA retracts below ±(threshold/2).
VWASLR_SIGMA_BARS = 500   # 500 × 1-min = 500-min σ window (slow/stable; matches backtest)
VWASLR_N          = 50    # 50 × 1-min = 50-min signal window
VWASLR_INIT_BARS  = VWASLR_SIGMA_BARS + VWASLR_N + 15  # initial 1-min fetch (565 bars)
VWASLR_EMA_SPAN   = 10    # EMA span applied to raw VWASLR (α = 2/11 ≈ 0.18)

ET = ZoneInfo("America/New_York")

# SLR_Scalp parameters (Volume Surge, immediate entry, LONG only)
# From backtest: vol≥7×, move≥12bp, no pullback, hold=15b RTH / 10b Globex
SLR_VOL_LOOKBACK = 20    # rolling median window (1-min bars)
SLR_VOL_MULT     = 7.0   # minimum volume surge multiplier
SLR_MOVE_BPS     = 12.0  # minimum surge move in basis points
SLR_TARGET_BPS   = 15.0  # profit target in basis points of entry price
SLR_STOP_BPS     = 10.0   # stop distance in basis points (scales with price)
SLR_HOLD_RTH     = 15    # max hold (minutes) during RTH
SLR_HOLD_GLOBEX  = 10    # max hold (minutes) during Globex

# ORB Character panel — thresholds from historical MES 2019-2026 analysis
ORB_CHAR_PL_STRONG  = 0.65   # 5-min PL ≥ this → trending signal (54% precision)
ORB_CHAR_MAE_CLEAN  = 0.15   # MAE/Move < this → clean move (45% precision)
ORB_CHAR_FETCH_BARS = 1500   # 1-min bars to fetch for prev_close (~25 h)

# PL_Mom parameters (keep in sync with trading_bot.py)
PL_MOM_WINDOW     = 6       # 5s bars = 30s
PL_MOM_ENTRY_PL   = 0.70
PL_MOM_MOVE_BPS   = 12.0
PL_MOM_EXIT_PL    = 0.40
PL_MOM_STOP_BPS   = 10.0
PL_MOM_MIN_HOLD_S = 10      # seconds before PL exit is checked (stop always active)
PL_MOM_MAX_HOLD_S = 120
PL_MOM_5S_FETCH   = 30      # number of 5s bars to fetch per poll (covers window + 4 extra)

# ORB parameters (15-min ORB, wide-range LONG, morning + power-hour windows)
ORB_BARS      = 3          # 3 × 5-min = 15-min opening range
# Per-instrument ORB_WIDTH_MIN stored in InstrumentConfig.orb_width_min
ORB_STOP_SIG  = 2.0
ORB_TGT_SIG   = 2.0        # 2σ:2σ → EV ≈ +0.61R
ORB_WINDOWS   = [          # (start_h, start_m, end_h, end_m, label)
    (9,  45, 10, 30, "Morning"),
]

LOG_PATH      = Path("logs/signals.csv")       # signal_monitor's own audit log (not read back)
ORB_LOG_PATH  = Path("logs/orb_signals.csv")   # signal_monitor's own audit log (not read back)
VWAS_LOG_PATH = Path("logs/vwaslr_trades.csv")
BOT_LOG_PATH     = Path("logs/bot_trades.csv")    # authoritative CSR trades written by trading_bot
BOT_ORB_LOG_PATH = Path("logs/orb_trades.csv")    # authoritative ORB trades written by trading_bot
BOT_SLR_LOG_PATH    = Path("logs/slr_trades.csv")    # authoritative SLR trades written by trading_bot
PL_MOM_TRADES_PATH  = Path("logs/pl_mom_trades.csv") # authoritative PL_mom trades written by trading_bot
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
    (0.20, "ELEVATED", "dark_orange"),
    (0.30, "ACTIVE",   "dark_orange"),
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
    # ORB: set orb_enabled=True and orb_width_pct_min to the wide-tertile cutoff from backtest.
    # Width threshold is a fraction of ORB midpoint price (e.g. 0.00354 = 0.354%).
    # Using percentage rather than fixed points keeps the filter consistent as prices change.
    orb_enabled:       bool  = False
    orb_width_pct_min: float = 0.0
    # VWASLR: 0 = disabled. n = look-back bars for the volume-weighted avg scaled return.
    # threshold = signal trigger level in σ/bar units (from backtest optimisation).
    # vwaslr_start: earliest (hour, minute) ET for VWASLR signals (default 9:30 RTH open).
    vwaslr_n:         int   = 0
    vwaslr_threshold: float = 1.0
    vwaslr_start:     tuple = (9, 30)
    # SLR_Scalp: enabled per instrument. Uses same 1-min bars as VWASLR.
    slr_enabled: bool = False
    # PL_Mom: Price Linearity Momentum on 5s bars. RTH only.
    pl_mom_enabled:   bool  = False
    pl_mom_entry_pl:  float = PL_MOM_ENTRY_PL
    pl_mom_move_bps:  float = PL_MOM_MOVE_BPS
    pl_mom_exit_pl:   float = PL_MOM_EXIT_PL
    pl_mom_stop_bps:  float = PL_MOM_STOP_BPS


INSTRUMENTS = [
    InstrumentConfig("MES", "MES", stop_sigma=2.0, target_sigma=3.0,
                     point_value=5.00, ev_sigma=0.073,
                     csr_vol_windows=[(0.08, 4), (1.0, 8)],
                     blackout_windows=[
                         (16,  0,  9,  0, False),  # trade 09:00–16:00 ET only
                     ],
                     orb_enabled=True, orb_width_pct_min=0.00354,
                     vwaslr_n=50, vwaslr_threshold=0.4, vwaslr_start=(9, 0),
                     slr_enabled=True,
                     pl_mom_enabled=True, pl_mom_entry_pl=0.80, pl_mom_move_bps=8.0,
                     pl_mom_stop_bps=7.0, pl_mom_exit_pl=0.40),
    InstrumentConfig("MNQ", "MNQ", stop_sigma=2.0, target_sigma=3.0,
                     point_value=2.00, ev_sigma=0.073,
                     csr_vol_windows=[(0.08, 4), (1.0, 8)],
                     blackout_windows=[
                         (16,  0,  9,  0, False),  # trade 09:00–16:00 ET only
                     ],
                     orb_enabled=False,
                     vwaslr_n=0, slr_enabled=True,
                     pl_mom_enabled=True,
                     pl_mom_stop_bps=8.0, pl_mom_exit_pl=0.20),
]

ALERT_SOUND = "/System/Library/Sounds/Ping.aiff"


def play_alert():
    """Sound notifications disabled."""
    return
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
    direction:  int = 1    # 1 = LONG, -1 = SHORT

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)
    def risk_pts(self):   return self.stop_pts()


@dataclass
class SLRScalpSignal:
    """Volume Surge + Shallow Pullback signal (LONG only)."""
    entry:     float
    target:    float   # entry + entry * SLR_TARGET_BPS / 10000
    stop:      float   # entry × (1 − SLR_STOP_BPS/10000)
    surge_ts:  datetime   # timestamp of the volume surge bar
    bar_ts:    datetime   # same as surge_ts (no pullback)
    vol_ratio: float   # surge bar volume / 20-bar median
    move_bps:  float   # actual move in basis points
    is_rth:    bool
    direction: int = 1    # always LONG

    def target_pts(self): return self.target - self.entry
    def stop_pts(self):   return self.entry  - self.stop   # positive

    def expires_at(self) -> datetime:
        hold = SLR_HOLD_RTH if self.is_rth else SLR_HOLD_GLOBEX
        return self.bar_ts + timedelta(minutes=hold)


@dataclass
class PLMomSignal:
    """Price Linearity Momentum signal on 5s bars."""
    direction:  int    # +1 long, -1 short
    entry:      float
    stop:       float
    pl:         float
    move_bps:   float
    bar_ts:     datetime  # ts of the last 5s bar at signal time
    is_rth:     bool

    def stop_pts(self): return abs(self.stop - self.entry)
    def expires_at(self): return self.bar_ts + timedelta(seconds=PL_MOM_MAX_HOLD_S)


@dataclass
class OrbState:
    session_date:    date | None = None
    orb_high:        float = 0.0
    orb_low:         float = 0.0
    orb_bars_seen:   int   = 0
    orb_complete:    bool  = False
    morning_fired:   bool  = False
    active_signal:   OrbSignal | None = None
    last_orb_bar_ts: datetime | None = None  # dedup: skip if we've seen this bar


@dataclass
class _HistSignal:
    """Minimal signal record reconstructed from CSV for history display."""
    bar_ts:    datetime
    direction: int      # +1 long, -1 short, 0 = ORB
    entry:     float
    target:    float
    stop:      float
    kind:      str = ""   # "ORB", "VWASLR", or "" (CSR momentum)


@dataclass
class RecentSignal:
    symbol:    str
    signal:    "Signal | OrbSignal | _HistSignal"
    outcome:   str        # "TARGET", "STOPPED", "TIME EXIT", "OPEN"
    pnl_pts:   float
    contracts: int = 1    # number of contracts traded (CSR can be 2×)
    exit_time: "datetime | None" = None


@dataclass
class InstrumentState:
    cfg:           InstrumentConfig
    cid:           str = ""
    cname:         str = ""
    bars:          list[Bar] = field(default_factory=list)
    vwaslr_bars:   list[Bar] = field(default_factory=list)  # separate 1-min bars for VWASLR
    sigma:         float = 0.0
    sigma_pts:     float = 0.0
    sigma_bar_count: int = 0      # number of returns used in sigma calc
    gk_ann_vol:    float = 0.0
    csr:           float = 0.0   # cumulative scaled return (40 min, direction-adjusted)
    mean_vol:           float | None = None
    active_signal:      Signal | None = None
    current_pl:         float | None = None      # raw 1-min PL, refreshed every poll
    current_ha_streak:  int = 0                  # 5-min HA streak: +k green, -k red, 0 = flat
    current_vwaslr:     float = 0.0             # EMA-10 of VWASLR (0.0 = not computed / warming up)
    vwaslr_ema:         float = 0.0            # running EMA-10 state (persists across polls)
    vwaslr_ema_prev:    float = 0.0            # EMA before last update (cross detection)
    vwaslr_entry:       float | None = None     # close price when EMA first crossed threshold
    has_vwaslr_position: bool = False           # True if account has an open position for this instrument
    position_size:       int  = 0              # number of open contracts (0 = flat)
    position_direction:  int  = 0              # +1 long, -1 short, 0 flat
    position_entry:      "float | None" = None # average entry price
    position_strategy:   str  = ""             # strategy that opened the position
    csr_trail_peak:     float | None = None     # most favourable price seen since CSR signal
    csr_trail_stop:     float | None = None     # current trailing stop level for CSR signal
    live_bar:           Bar | None = None        # developing 5-min bar, refreshed every poll
    orb:                OrbState = field(default_factory=OrbState)
    history:            list[RecentSignal] = field(default_factory=list)
    error:              str | None = None
    last_evaluated_ts:  datetime | None = None   # ts of last bar evaluated for signals
    pl_bars:            list[Bar] = field(default_factory=list)  # cached 1-min bars for display PL
    bars_fetch_min:     int = -1                 # UTC minute of last 5-min bar fetch (throttle)
    vwaslr_fetch_min:   int = -1                 # UTC minute of last vwaslr fetch (throttle)
    vwaslr_last_fetch:  datetime | None = None   # wall-clock time of last vwaslr fetch (stale retry)
    vwaslr_ema_bar_ts:  datetime | None = None   # bar ts when EMA was last advanced (gate per-bar)
    pl_fetch_min:       int = -1                 # UTC minute of last display PL fetch (throttle)
    live_bar_fetch_min: int = -1                 # UTC minute of last live-bar fetch (throttle)
    # SLR_Scalp state
    slr_last_surge_ts:  datetime | None = None   # surge bar ts of last fired signal (dedup)
    slr_eval_bar_ts:    datetime | None = None   # 1-min bar ts last evaluated for SLR (gate)
    active_slr_signal:  SLRScalpSignal | None = None
    # PL_Mom state
    pl_mom_5s_bars:       list[Bar] = field(default_factory=list)
    active_pl_mom_signal: "PLMomSignal | None" = None
    pl_mom_last_bar_ts:   "datetime | None" = None
    pl_mom_entry_ts:      "datetime | None" = None  # when signal fired (for min_hold check)
    pl_mom_history:       list = field(default_factory=list)   # [(ts, pl, dir_sym, move_bps), ...]
    pl_mom_last_hist_ts:  "datetime | None" = None
    # ORB character panel state
    orb_char_prev_close: float      = 0.0        # last RTH close before today's 8:30 ET
    orb_char_date:       "date | None" = None    # date prev_close was computed for
    orb_char_ovn_cache:  "dict | None" = None    # overnight metrics cached at RTH open (immutable)
    orb_char_gap_bp:     "float | None" = None   # RTH open gap vs prev_close (cached)
    orb_char_rth_open:   "float | None" = None   # RTH open price (cached)
    orb_char_ovn_low_bp: "float | None" = None   # overnight low in bp vs prev_close (cached)
    orb_char_ovn_high_bp:"float | None" = None   # overnight high in bp vs prev_close (cached)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _in_blackout(bar_hm: tuple[int, int],
                 sh: int, sm: int, eh: int, em: int) -> bool:
    """Return True if bar_hm falls inside the [start, end) window.
    Handles overnight windows where start > end (e.g. 18:00–09:00)."""
    s = sh * 60 + sm
    e = eh * 60 + em
    b = bar_hm[0] * 60 + bar_hm[1]
    return (s <= b < e) if s < e else (b >= s or b < e)

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


def _signal_bar(val: float, threshold: float, max_val: float, width: int = 20) -> str:
    """
    Centered signed bar for scaled return, CSR, etc.
    - Red   ░ region left  of -threshold
    - Green ░ region right of +threshold
    - Dim   ─ neutral zone between ±threshold
    - Dim   │ at centre (val=0)
    - Bold  █ marker coloured green (val>0) / red (val<0)
    """
    half    = width // 2
    clamped = max(-max_val, min(max_val, val))
    marker  = max(0, min(width - 1, half + int(round(clamped / max_val * half))))
    thr_off = max(1, int(round(threshold / max_val * half)))
    thr_lo  = half - thr_off   # left threshold position
    thr_hi  = half + thr_off   # right threshold position

    parts = []
    for i in range(width):
        if i == marker:
            style = "bold green" if val > 0 else ("bold red" if val < 0 else "bold white")
            parts.append(f"[{style}]█[/]")
        elif i == half:
            parts.append("[dim]│[/]")
        elif i < thr_lo:
            parts.append("[red]░[/]")
        elif i >= thr_hi:
            parts.append("[green]░[/]")
        else:
            parts.append("[dim]─[/]")
    return "".join(parts)


def _next_bar_close(now: datetime) -> datetime:
    epoch_min = int(now.timestamp() // 60)
    next_close_min = ((epoch_min // TF_MINUTES) + 1) * TF_MINUTES
    return datetime.fromtimestamp(next_close_min * 60, tz=timezone.utc)


def _ha_streak(bars: list[Bar]) -> int:
    """
    Compute the current 5-min Heiken-Ashi streak from state.bars.
    Returns +k for k consecutive green bars, -k for k consecutive red bars.
    HA state is reset at each ET calendar date boundary.
    Returns 0 if the current session has fewer than 2 bars.
    """
    if not bars:
        return 0
    session_date = bars[-1].ts.astimezone(ET).date()
    # Find first bar of current session
    session_start = len(bars) - 1
    for i in range(len(bars) - 1, -1, -1):
        if bars[i].ts.astimezone(ET).date() == session_date:
            session_start = i
        else:
            break
    sb = bars[session_start:]
    if len(sb) < 2:
        return 0
    n = len(sb)
    ha_open  = np.zeros(n)
    ha_close = np.zeros(n)
    ha_close[0] = (sb[0].open + sb[0].high + sb[0].low + sb[0].close) / 4
    ha_open[0]  = (sb[0].open + sb[0].close) / 2
    for i in range(1, n):
        b = sb[i]
        ha_close[i] = (b.open + b.high + b.low + b.close) / 4
        ha_open[i]  = (ha_open[i - 1] + ha_close[i - 1]) / 2
    green = ha_close[-1] >= ha_open[-1]
    streak = 1
    for i in range(n - 2, -1, -1):
        if (ha_close[i] >= ha_open[i]) == green:
            streak += 1
        else:
            break
    return streak if green else -streak


def _compute_vwaslr(bars: list[Bar], n_win: int,
                    sigma_bars: int = VWASLR_SIGMA_BARS) -> float:
    """
    Compute VWASLR_n (Volume-Weighted Average Scaled Log Return) for the most
    recent completed bar.

      VWASLR = Σ(ret_j / σ × vol_j) / Σ(vol_j)   for j in [i-n_win+1 .. i]

    σ is estimated from the trailing `sigma_bars` log-returns (slow 500-min window).
    Returns 0.0 if there are insufficient bars.
    """
    needed = max(n_win, sigma_bars) + 1
    if len(bars) < needed:
        return 0.0
    closes  = np.array([b.close  for b in bars], dtype=float)
    volumes = np.array([b.volume for b in bars], dtype=float)
    i = len(bars) - 1

    trail_rets = np.log(closes[i - sigma_bars + 1: i + 1]
                      / closes[i - sigma_bars:     i    ])
    sigma = float(np.std(trail_rets, ddof=1))
    if sigma == 0:
        return 0.0

    ret_win = np.log(closes[i - n_win + 1: i + 1]
                   / closes[i - n_win:     i    ])
    vol_win = volumes[i - n_win: i]
    sum_vol = float(vol_win.sum())
    if sum_vol == 0:
        return 0.0

    return float((ret_win / sigma * vol_win).sum() / sum_vol)


def evaluate_slr_scalp(state: "InstrumentState") -> "SLRScalpSignal | None":
    """
    Scan state.vwaslr_bars for vol surge (LONG only). No pullback required —
    entry fires immediately when surge bar closes. bars[-1] is the surge bar.
    """
    bars = state.vwaslr_bars
    if len(bars) < SLR_VOL_LOOKBACK + 2:
        return None

    n         = len(bars)
    surge_bar = bars[-1]
    surge_idx = n - 1

    # Skip CME settlement gap (16:00–17:00 CT = 21:00–22:00 UTC)
    surge_et = surge_bar.ts.astimezone(ET)
    if (surge_et.hour, surge_et.minute) >= (16, 0) and surge_et.hour < 17:
        return None

    # Skip if already fired on this surge bar
    if (state.slr_last_surge_ts is not None
            and surge_bar.ts <= state.slr_last_surge_ts):
        return None

    # 1) Volume surge: vol ≥ SLR_VOL_MULT × 20-bar median
    prior_vols = [bars[j].volume for j in range(surge_idx - SLR_VOL_LOOKBACK, surge_idx)]
    med_vol    = float(np.median(prior_vols)) if prior_vols else 0.0
    if med_vol <= 0:
        return None
    vol_ratio = surge_bar.volume / med_vol
    if vol_ratio < SLR_VOL_MULT:
        return None

    # 2) Bullish bar
    if surge_bar.close < surge_bar.open:
        return None

    # 3) Directional move ≥ SLR_MOVE_BPS (WO2: open[surge-1] → close[surge])
    prev_bar = bars[surge_idx - 1]
    if prev_bar.open <= 0:
        return None
    if (surge_bar.ts - prev_bar.ts) > timedelta(minutes=2):
        return None
    surge_move_bps = (surge_bar.close - prev_bar.open) / prev_bar.open * 10000
    if surge_move_bps < SLR_MOVE_BPS:
        return None

    entry     = surge_bar.close
    target    = entry * (1.0 + SLR_TARGET_BPS / 10000.0)
    sigma_bps = (state.sigma_pts / entry * 10000) if state.sigma_pts else 0.0
    stop_bps  = max(SLR_STOP_BPS, sigma_bps)
    stop      = entry * (1.0 - stop_bps / 10000.0)
    is_rth    = (9, 30) <= (surge_et.hour, surge_et.minute) < (16, 0)

    return SLRScalpSignal(
        entry=entry, target=target, stop=stop,
        surge_ts=surge_bar.ts, bar_ts=surge_bar.ts,
        vol_ratio=vol_ratio, move_bps=surge_move_bps,
        is_rth=is_rth,
    )


def _check_slr_resolution(sig: "SLRScalpSignal",
                           bars: list[Bar]) -> "tuple[str, float] | None":
    """Scan 1-min bars after the signal bar for target/stop. Stop-first (conservative)."""
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if bar.low <= sig.stop:
            return "STOPPED", -sig.stop_pts()
        if bar.high >= sig.target:
            return "TARGET",   sig.target_pts()
    return None


# ── Panel builders ─────────────────────────────────────────────────────────────

def build_regime_panel(state: InstrumentState) -> Panel:
    sigma, sigma_pts = state.sigma, state.sigma_pts
    ann = annualised_vol(sigma)
    label, style = regime_label(ann)
    cur_vol   = state.bars[-1].volume if state.bars else 0.0
    vol_ratio = (cur_vol / state.mean_vol) if state.mean_vol is not None else None

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=16, no_wrap=True)
    t.add_column(width=34)

    gk  = state.gk_ann_vol
    gk_label, gk_style = regime_label(gk) if gk > 0 else (label, style)

    t.add_row("σ per bar:",
              f"[bold]{sigma * 10000:.2f} bps[/]  │  {sigma_pts:.2f} pts")
    t.add_row("Ann. vol (CR):",
              f"{ann*100:.1f}%  [dim](closes, 100 bars)[/]")
    t.add_row("Ann. vol (GK):",
              f"[bold]{gk*100:.1f}%[/]  [dim](G-K, 20 bars)[/]")
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
               else ("dark_orange" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "white")
    vr_style = ("green" if vol_ratio >= VOL_RATIO_MIN else
                ("dark_orange" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "white")) if vol_ratio is not None else "dim"
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
        vr_pct  = min(vol_ratio / VOL_RATIO_MIN * 100, 100) if vol_ratio is not None else 0.0
        bar_vr  = "█" * int(vr_pct / 5) + "░" * (20 - int(vr_pct / 5))

        t = Table.grid(padding=(0, 1))
        t.add_column(style="green", width=12, no_wrap=True)
        t.add_column(no_wrap=True)
        csr       = state.csr                           # direction-adjusted: used for ✓ check
        raw_csr   = csr if scaled > 0 else -csr        # raw directional: +ve = market up
        csr_style = "green" if raw_csr > 0 else "red"
        csr_check = "✓" if csr >= CSR_THRESHOLD else \
                    ("~" if csr > 0 else "✗")
        sc_style  = "green" if scaled > 0 else "red"

        t.add_row("Scaled:",
                  f"{_signal_bar(scaled, SIGNAL_SIGMA, MAX_SCALED, width=15)} "
                  f"[{sc_style}]{scaled:+.2f}σ[/]/{SIGNAL_SIGMA:.0f}σ")

        now_utc   = datetime.now(timezone.utc)
        open_min  = int(now_utc.timestamp() // 60) // TF_MINUTES * TF_MINUTES
        bar_open  = datetime.fromtimestamp(open_min * 60, tz=timezone.utc)
        elapsed_m = int((now_utc - bar_open).total_seconds() // 60)
        lb = state.live_bar
        if lb is not None and sigma:
            lb_ret    = math.log(lb.close / lb.open) if lb.open else 0.0
            lb_scaled = lb_ret / sigma
            num_style = "green" if lb_scaled > 0 else "red"
            t.add_row(f"Dev ({elapsed_m}m):",
                      f"{_signal_bar(lb_scaled, SIGNAL_SIGMA, MAX_SCALED, width=15)} "
                      f"[{num_style}]{lb_scaled:+.2f}σ[/]")
        else:
            t.add_row(f"Dev ({elapsed_m}m):", _signal_bar(0.0, SIGNAL_SIGMA, MAX_SCALED, width=15))

        mom_bars = get_mom_bars(state.gk_ann_vol, state.cfg.csr_vol_windows)
        t.add_row(f"Mom({mom_bars * TF_MINUTES}m):",
                  f"{_signal_bar(raw_csr, CSR_THRESHOLD, CSR_THRESHOLD * 2, width=15)} "
                  f"[{csr_style}]{raw_csr:+.2f}σ[/]/{CSR_THRESHOLD:.1f}σ {csr_check}")
        pl = state.current_pl
        if pl is not None:
            val_style = "green" if pl >= PL_THRESH else ("red" if pl <= -PL_THRESH else "white")
            t.add_row("1-min PL:",
                      f"{_pl_bar(abs(pl), '▲' if pl >= 0 else '▼', half=7)} [{val_style}]{pl:+.2f}[/]")
        else:
            t.add_row("1-min PL:", "[dim]fetching…[/]")
        ha = state.current_ha_streak
        if ha != 0:
            abs_ha  = abs(ha)
            ha_col  = "green" if ha > 0 else "red"
            ha_dir  = "▲" if ha > 0 else "▼"
            ha_word = "green" if ha > 0 else "red"
            t.add_row("5-min HA:",
                      f"[{ha_col}]{ha_dir} {abs_ha} {ha_word} bar{'s' if abs_ha > 1 else ''}[/]")
        else:
            t.add_row("5-min HA:", "[dim]—[/]")
        t.add_row("", "")
        t.add_row("Vol ratio:",
                  f"[green]{bar_vr}[/] {vol_ratio:.2f}×/{VOL_RATIO_MIN:.1f}×"
                  if vol_ratio is not None else
                  f"[dim]{bar_vr}[/] warming up")

        bar_et = state.bars[-1].ts.astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        in_active_blackout = any(
            _in_blackout(bar_hm, sh, sm, eh, em) and (not conditional or csr < CSR_THRESHOLD)
            for sh, sm, eh, em, conditional in state.cfg.blackout_windows
        )
        blackout_note = "  [bold red]BLACKOUT[/]" if in_active_blackout else ""
        if in_active_blackout:
            t.add_row("[bold red]BLACKOUT[/]", "[dim]signals suppressed[/]")

        # NOTRADE: show indicative SL/TP based on current price and σ
        price     = bar.close
        sigma_pts = state.sigma_pts
        nt = Table.grid(padding=(0, 1))
        nt.add_column(style="dim", width=10)
        nt.add_column(min_width=38)
        if sigma_pts > 0:
            min_stop_pts   = price * SLR_STOP_BPS / 10000.0
            min_target_pts = min_stop_pts * cfg.target_sigma / cfg.stop_sigma
            stop_dist      = max(cfg.stop_sigma   * sigma_pts, min_stop_pts)
            target_dist    = max(cfg.target_sigma * sigma_pts, min_target_pts)
            long_tgt  = price + target_dist
            long_stop = price - stop_dist
            shrt_tgt  = price - target_dist
            shrt_stop = price + stop_dist
            nt.add_row("Price:",  f"[dim]{price:,.2f}[/]")
            nt.add_row("LONG:",   f"[dim]tgt {long_tgt:,.2f}  /  sl  {long_stop:,.2f}[/]")
            nt.add_row("SHORT:",  f"[dim]tgt {shrt_tgt:,.2f}  /  cs  {shrt_stop:,.2f}[/]")
        else:
            nt.add_row("", "[dim]warming up[/]")

        watching_panel = Panel(t,  title="[bold blue]⬤  WATCHING[/]",
                               border_style="blue", padding=(0, 2))
        notrade_panel  = Panel(nt, title="NOTRADE",
                               border_style="blue",   padding=(0, 2))

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

    trail_stop = state.csr_trail_stop
    trail_peak = state.csr_trail_peak

    t.add_row("Entry:",
              f"[bold]{signal.entry:,.2f}[/]")
    t.add_row("Target:",
              f"[bold {color}]{signal.target:,.2f}[/]  "
              f"([{color}]+{signal.target_pts():.2f} pts[/] │ +{cfg.target_sigma:.1f}σ)")
    if trail_stop is not None:
        trail_pnl = (trail_stop - signal.entry) * signal.direction
        trail_col = "green" if trail_pnl >= 0 else "red"
        peak_str  = f"  [dim](peak {trail_peak:,.2f})[/]" if trail_peak else ""
        t.add_row("Trail stop:",
                  f"[bold {trail_col}]{trail_stop:,.2f}[/]  "
                  f"([{trail_col}]{trail_pnl:+.2f} pts[/] │ {CSR_TRAIL_SIGMA}σ trail)"
                  f"{peak_str}")
        t.add_row("Hard stop:",
                  f"[dim]{signal.stop:,.2f}  (safety net, {cfg.stop_sigma:.1f}σ)[/]")
    else:
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
                  f"[bold green]  ⚡ 2× CONTRACTS  PL={pl_str}  [/]")
    elif signal.pl_aligned is not None:
        t.add_row("PL:",
                  f"[dim]{signal.pl_aligned:+.2f} (1× size)[/]")

    border = "gold1" if sizing_2x else color
    return Panel(t,
                 title=f"[bold {color}]⬤  {direction_str}  SIGNAL[/]",
                 border_style=border, padding=(0, 2))


def build_vwaslr_panel(state: InstrumentState) -> Panel:
    """
    Small panel showing the current VWASLR value for the instrument.
    Lights green/red when above/below ±threshold.
    """
    v   = state.current_vwaslr
    thr = state.cfg.vwaslr_threshold
    n   = state.cfg.vwaslr_n

    is_long  = v >= thr
    is_short = v <= -thr
    max_disp = max(thr * 2.0, abs(v) * 1.05)

    t = Table.grid(padding=(0, 1))
    t.add_column(width=16)
    t.add_column()

    sig_min = n * 1   # 1-min bars → sig_min = n minutes

    if v == 0.0:
        t.add_row(f"VWASLR({sig_min}m):", "warming up…")
    else:
        v_style = ("bold green" if is_long  else
                   "bold red"   if is_short else
                   "green"      if v > 0    else
                   "red"        if v < 0    else "bold")
        tz_local  = datetime.now().astimezone().tzinfo
        # Show bar close time (open ts + 1 min) so e.g. the bar that
        # opened at 15:32 and closed at 15:33 is labelled "15:33".
        cur_ts_str = (
            (state.vwaslr_bars[-1].ts + timedelta(minutes=1))
            .astimezone(tz_local).strftime("%H:%M")
            if state.vwaslr_bars else "")
        t.add_row(f"VWASLR({sig_min}m):",
                  f"{_signal_bar(v, thr, max_disp)} "
                  f"[{v_style}]{v:+.3f}[/]  / ±{thr:.1f}σ"
                  + (f"  {cur_ts_str}" if cur_ts_str else ""))

    # Five prior 1-min VWASLR values (-1m through -5m)
    # Need VWASLR_SIGMA_BARS + 1 bars for the oldest offset (offset=5 → slice_end=len-5,
    # so need len-5 >= VWASLR_SIGMA_BARS+1, i.e. len >= VWASLR_SIGMA_BARS+6).
    if len(state.vwaslr_bars) >= VWASLR_SIGMA_BARS + 6:
        tz_local = datetime.now().astimezone().tzinfo
        for offset in range(1, 6):
            slice_end = len(state.vwaslr_bars) - offset
            recent_v  = _compute_vwaslr(state.vwaslr_bars[:slice_end], n, VWASLR_SIGMA_BARS)
            # Bar close time = open ts + 1 min
            bar_ts    = (state.vwaslr_bars[slice_end - 1].ts + timedelta(minutes=1)
                         ).astimezone(tz_local)
            lbl       = f"  -{offset}m:"
            if recent_v == 0.0:
                t.add_row(lbl, "—")
            else:
                rv_style = ("bold green" if recent_v >= thr  else
                            "bold red"   if recent_v <= -thr else
                            "green"      if recent_v > 0      else
                            "red"        if recent_v < 0      else "bold")
                t.add_row(lbl,
                          f"[{rv_style}]{recent_v:+.3f}[/]"
                          f"  {bar_ts.strftime('%H:%M')}")

    if state.vwaslr_entry is not None:
        entry_style = "bold green" if is_long else "bold red"
        t.add_row("Entry:", f"[{entry_style}]{state.vwaslr_entry:,.2f}[/]")

        # Show half-zero exit level: EMA must retract below ±(thr/2) to exit
        half_thr = thr / 2
        exit_col = "cyan"
        t.add_row("Exit when:",
                  f"[{exit_col}]EMA {'<' if is_long else '>'} "
                  f"{'+'if is_long else '-'}{half_thr:.2f}σ[/]  "
                  f"[dim](now {v:+.3f}σ)[/]")

    border = "green" if is_long else ("red" if is_short else "blue")
    if is_long:
        if state.has_vwaslr_position:
            title = "[bold green]⬤  VWASLR  ▲ LONG[/]"
        else:
            title = "[green]◯  VWASLR  ▲ LONG  [dim]no position[/][/]"
    elif is_short:
        if state.has_vwaslr_position:
            title = "[bold red]⬤  VWASLR  ▼ SHORT[/]"
        else:
            title = "[red]◯  VWASLR  ▼ SHORT  [dim]no position[/][/]"
    else:
        title = "VWASLR"
    return Panel(t, title=title, border_style=border, padding=(0, 1))


def build_slr_scalp_panel(state: InstrumentState, now: datetime) -> Panel:
    """Small panel showing current SLR_Scalp signal status or recent vol ratios."""
    sig  = state.active_slr_signal
    bars = state.vwaslr_bars

    if sig is not None:
        t = Table.grid(padding=(0, 1))
        t.add_column(width=14)
        t.add_column()
        expires = sig.expires_at()
        rem     = expires - now
        rem_s   = int(rem.total_seconds())
        rem_str = (f"{rem_s // 60}m {rem_s % 60:02d}s"
                   if rem_s > 0 else "[blink]EXPIRED[/]")
        sess    = "RTH" if sig.is_rth else "GLOBEX"
        hold    = SLR_HOLD_RTH if sig.is_rth else SLR_HOLD_GLOBEX

        is_long  = sig.direction == 1
        clr      = "green" if is_long else "red"
        dirn_str = "▲ LONG" if is_long else "▼ SHORT"
        t.add_row("Entry:",   f"[bold]{sig.entry:,.2f}[/]")
        t.add_row("Target:",  f"[bold {clr}]{sig.target:,.2f}[/]  "
                              f"([{clr}]{sig.target_pts():.2f} pts[/]  {SLR_TARGET_BPS:.0f}bp)")
        t.add_row("Stop:",    f"[bold red]{sig.stop:,.2f}[/]  "
                              f"([red]{sig.stop_pts():.2f} pts[/]  {SLR_STOP_BPS:.1f}bp)")
        t.add_row("Expires:", f"{rem_str}  [dim]({sess} {hold}m hold)[/]")
        t.add_row("Trigger:", f"[dim]vol={sig.vol_ratio:.1f}×  move={sig.move_bps:.1f}bp[/]")

        return Panel(t, title=f"[bold {clr}]⬤  SLR SCALP  {dirn_str}  {state.cfg.symbol}[/]",
                     border_style=clr, padding=(0, 1))

    # No active signal — 2-column grid; use Python format strings for alignment
    t = Table.grid(padding=(0, 1))
    t.add_column(width=9)
    t.add_column(no_wrap=True)

    if len(bars) >= SLR_VOL_LOOKBACK + 2:
        tz_local = datetime.now().astimezone().tzinfo
        for i in range(1, 6):
            idx = len(bars) - i
            if idx < SLR_VOL_LOOKBACK:
                break
            b        = bars[idx]
            prior    = [bars[j].volume for j in range(idx - SLR_VOL_LOOKBACK, idx)]
            med      = float(np.median(prior)) if prior else 0.0
            vr       = b.volume / med if med > 0 else 0.0
            prev_b   = bars[idx - 1] if idx > 0 else None
            move_bps = ((b.close - prev_b.open) / prev_b.open * 10000
                        if prev_b and prev_b.open and
                           (b.ts - prev_b.ts) <= timedelta(minutes=2) and
                           abs(b.close / prev_b.open - 1) <= 0.05   # sanity: ≤5% cross-bar move
                        else 0.0)
            bar_time = (b.ts + timedelta(minutes=1)).astimezone(tz_local).strftime("%H:%M")
            lbl      = ("Latest:" if i == 1 else f"  -{i}m:")
            vr_num   = f"{vr:4.1f}"   # fixed width: " 0.5" or "14.5"
            if vr >= SLR_VOL_MULT:
                vr_str = f"[bold green]{vr_num}× vol[/]"
            elif vr >= SLR_VOL_MULT * 0.6:
                vr_str = f"[bold green]{vr_num}× vol[/]"
            else:
                vr_str = f"{vr_num}× vol"
            vol_num = f"{int(b.volume):>4}"            # right-aligned, up to 4 digits
            bp_num  = f"{move_bps:+6.1f}bp"           # fixed width: " +0.4bp" or "+12.3bp"
            bp_col  = "bold green" if move_bps >= SLR_MOVE_BPS else ("bold red" if move_bps <= -SLR_MOVE_BPS else "")
            bp_disp = (f"[{bp_col}]{bp_num}[/]" if bp_col else bp_num)
            t.add_row(lbl, f"{vr_str}  [dim]{vol_num}[/]  {bp_disp}  [dim]{bar_time}[/]")
    else:
        t.add_row("Status:", "[dim]warming up…[/]")
    t.add_row("", f"[dim]watching {SLR_VOL_MULT:.0f}× vol, {SLR_MOVE_BPS:.0f}bp surge[/]")
    return Panel(t, title=f"SLR SCALP  {state.cfg.symbol}", border_style="blue", padding=(0, 1))


def _pl_bar(pl: float, dir_sym: str, half: int = 6) -> str:
    """
    Bidirectional centered bar. Green extends right for longs, red left for shorts.
    '░░░░░░|██████'  (long)   or   '██████|░░░░░░'  (short)
    """
    filled = max(0, min(half, round(min(1.0, pl) * half)))
    empty  = half - filled
    if dir_sym == "▲":
        return f"[dim]{'░' * half}|[/][green]{'█' * filled}{'░' * empty}[/]"
    else:
        return f"[red]{'░' * empty}{'█' * filled}[/][dim]|{'░' * half}[/]"


def _orb_char_metrics(state: "InstrumentState", now: datetime) -> dict:
    """
    Compute overnight + first-2h ORB characteristics from vwaslr_bars.

    Returns a dict (may be empty if data not available) with keys:
      prev_close, gap_bp, ovn_low_bp, ovn_high_bp, ovn_char,
      pl5, mae_vs_move, phase ('pre'|'live'|'final'), elapsed_min, rth_open
    """
    prev_close = state.orb_char_prev_close
    if not prev_close or not state.vwaslr_bars:
        return {}

    today   = now.astimezone(ET).date()
    now_et  = now.astimezone(ET)
    now_hm  = now_et.hour * 60 + now_et.minute

    def _et(h: int, m: int) -> datetime:
        return datetime(today.year, today.month, today.day, h, m,
                        tzinfo=ET).astimezone(timezone.utc)

    t_rth   = _et(8, 30)
    t_orb   = _et(10, 30)
    t_ovn_s = _et(17, 0) - timedelta(days=1)   # ~5pm ET yesterday

    bars = state.vwaslr_bars

    # Use state-cached values when available (populated by refresh_orb_char
    # from full history fetch — immune to vwaslr_bars trimming).
    gap_bp      = state.orb_char_gap_bp
    rth_open    = state.orb_char_rth_open
    ovn_low_bp  = state.orb_char_ovn_low_bp  or 0.0
    ovn_high_bp = state.orb_char_ovn_high_bp or 0.0

    # Determine phase from clock, not from bar availability
    if now_hm < 8 * 60 + 30:
        phase = "pre"
    elif now_hm < 10 * 60 + 30:
        phase = "live"
    else:
        phase = "final"

    # ORB window bars still needed for PL/MAE calculation
    rth_all = [b for b in bars if b.ts >= t_rth and b.ts.date() == today]

    elapsed_min = max(0, min(120, now_hm - 8 * 60 - 30)) if phase != "pre" else 0

    # ORB window bars (8:30–10:30, or up to now if still live)
    orb_cutoff  = t_orb if phase == "final" else now
    orb_1m = [b for b in rth_all if b.ts < orb_cutoff]

    pl5 = mae_vs_move = None
    if len(orb_1m) >= 5:
        # Resample 1-min → 5-min by taking close of every 5th bar group
        closes_5m = []
        for i in range(0, len(orb_1m), 5):
            chunk = orb_1m[i:i + 5]
            if chunk:
                closes_5m.append(chunk[-1].close)

        if len(closes_5m) >= 2:
            arr  = np.array(closes_5m, dtype=float)
            rets = np.log(arr[1:] / arr[:-1])
            denom = float(np.abs(rets).sum())
            pl5   = float(abs(rets.sum()) / denom) if denom > 0 else 0.0

            w_ret = (arr[-1] - arr[0]) / arr[0]
            cum   = np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
            direction = 1 if w_ret >= 0 else -1
            if direction == 1:
                mae_frac = float(abs((cum / np.maximum.accumulate(cum) - 1.0).min()))
            else:
                mae_frac = float(abs((cum / np.minimum.accumulate(cum) - 1.0).max()))
            if abs(w_ret) > 1e-5:
                mae_vs_move = mae_frac / abs(w_ret)

    # Overnight character text
    DIP_THRESH = 15.0   # bp adverse overnight move to count as "deep"
    GAP_THRESH = 10.0   # bp open gap to be labelled gap-up / gap-down
    if gap_bp is None:
        ovn_char = "—"
    elif ovn_low_bp < -DIP_THRESH and gap_bp > -GAP_THRESH:
        recovery_bp = gap_bp - ovn_low_bp   # how much recovered from low
        ovn_char = f"V-shape dip  [dim](−{abs(ovn_low_bp):.0f}bp, rcvd {recovery_bp:.0f}bp)[/]"
    elif ovn_high_bp > DIP_THRESH and gap_bp < GAP_THRESH:
        ovn_char = f"V-shape rally  [dim](+{ovn_high_bp:.0f}bp, faded)[/]"
    elif gap_bp > GAP_THRESH:
        ovn_char = f"Gap up  [dim](+{gap_bp:.0f}bp)[/]"
    elif gap_bp < -GAP_THRESH:
        ovn_char = f"Gap down  [dim]({gap_bp:.0f}bp)[/]"
    else:
        ovn_char = f"Flat open  [dim]({gap_bp:+.0f}bp)[/]"

    return dict(prev_close=prev_close, gap_bp=gap_bp,
                ovn_low_bp=ovn_low_bp, ovn_high_bp=ovn_high_bp,
                ovn_char=ovn_char, pl5=pl5, mae_vs_move=mae_vs_move,
                phase=phase, elapsed_min=elapsed_min, rth_open=rth_open)


def _detect_position_strategy(symbol: str) -> str:
    """Scan the tail of trading_bot.log for the most recent order message for this symbol."""
    log_path = Path("logs/trading_bot.log")
    if not log_path.exists():
        return ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines[-500:]):
            if f" {symbol} " not in line:
                continue
            if "VWASLR ORDER" in line:  return "VWASLR"
            if "SLR ORDER"    in line:  return "SLR"
            if "PL MOM ORDER" in line:  return "PL MOM"
            if "ORB ORDER"    in line:  return "ORB"
            if "ORDER PLACED" in line:  return "CSR"
            if "EVE ORDER"    in line:  return "EVE"
            if "SUN ORDER"    in line:  return "SUN"
    except Exception:
        pass
    return ""


def build_positions_panel(states: "list[InstrumentState]") -> Panel:
    """Compact panel showing open contract count and strategy per instrument."""
    t = Table.grid(padding=(0, 2))
    t.add_column(width=5)                    # symbol
    t.add_column(width=4, justify="right")   # contracts
    t.add_column(width=8)                    # direction arrow
    t.add_column(width=8, justify="right")   # entry price
    t.add_column(width=8)                    # strategy
    for s in states:
        sym   = s.cfg.symbol
        sz    = s.position_size
        d     = s.position_direction
        strat = s.position_strategy
        entry = f"{s.position_entry:.2f}" if s.position_entry else ""
        if sz == 0:
            t.add_row(sym, "—", "", "", "")
        elif d == 1:
            t.add_row(sym, f"[green]{sz}[/]", "[green]▲ LONG[/]", f"[green]{entry}[/]", f"[green]{strat}[/]")
        else:
            t.add_row(sym, f"[red]{sz}[/]", "[red]▼ SHORT[/]", f"[red]{entry}[/]", f"[red]{strat}[/]")
    return Panel(t, title="POSITIONS", border_style="blue", padding=(0, 1), expand=False)


def build_orb_char_panel(states: "list[InstrumentState]", now: datetime) -> Panel:
    """
    Compact vertical panel: one row per instrument, columns for key ORB metrics.
    Narrow enough to fit inside M2K's instrument column without squeezing others.

    Columns: Sym | Ovn | Gap | PL(5m) | MAE/Mv | Verdict
    """
    metrics = [_orb_char_metrics(s, now) for s in states]

    phases = [m.get("phase", "pre") for m in metrics if m]
    phase  = phases[0] if phases else "pre"

    # Abbreviate overnight character to a short symbol
    def _ovn_sym(m: dict) -> str:
        if not m:
            return "[dim]—[/]"
        lo = m.get("ovn_low_bp",  0.0) or 0.0
        hi = m.get("ovn_high_bp", 0.0) or 0.0
        gap = m.get("gap_bp")
        DIP = 15.0; GAP = 10.0
        if gap is None:
            return "—"
        if lo < -DIP and gap > -GAP:
            return "V↓"
        if hi > DIP and gap < GAP:
            return "V↑"
        if gap > GAP:
            return "G↑"
        if gap < -GAP:
            return "G↓"
        return "~"

    t = Table.grid(padding=(0, 2))
    t.add_column(width=7)                          # symbol
    t.add_column(width=4)                          # ovn char
    t.add_column(width=7,  justify="right")        # gap
    t.add_column(width=8,  justify="right")        # PL(5m)
    t.add_column(width=8,  justify="right")        # MAE/Move ratio
    t.add_column(width=12)                         # verdict

    # Header row — plain text, no dim/bold
    pl_hdr  = "PL(5m)"
    mae_hdr = "MAE/Mv"   # ratio: max-pullback ÷ net-move (lower = cleaner)
    t.add_row("Sym", "Ovn", "Gap", pl_hdr, mae_hdr, "Verdict")

    for s, m in zip(states, metrics):
        sym = s.cfg.symbol

        gap = m.get("gap_bp")
        gap_str = "—" if gap is None else f"{gap:+.0f}bp"

        pl = m.get("pl5")
        if pl is None:
            pl_str = "—"
        elif pl >= ORB_CHAR_PL_STRONG:
            pl_str = f"[green]{pl:.3f}★[/]"
        else:
            pl_str = f"{pl:.3f}"

        mv = m.get("mae_vs_move")
        if mv is None:
            mae_str = "—"
        elif mv < ORB_CHAR_MAE_CLEAN:
            mae_str = f"[green]{mv:.2f}★[/]"
        else:
            # Cap display at 9.99 to avoid wrapping when move is tiny
            mae_str = f"{min(mv, 9.99):.2f}"

        if pl is None or mv is None:
            verd = "—"
        elif pl >= ORB_CHAR_PL_STRONG and mv < ORB_CHAR_MAE_CLEAN:
            verd = "[green]TRENDING ★★[/]"
        elif pl >= ORB_CHAR_PL_STRONG or mv < ORB_CHAR_MAE_CLEAN:
            verd = "[green]TRENDING[/]"
        else:
            verd = "MIXED"

        t.add_row(sym, _ovn_sym(m), gap_str, pl_str, mae_str, verd)

    # Progress bar during live window
    if phase == "live":
        elapsed = next((m.get("elapsed_min", 0) for m in metrics if m), 0)
        filled  = int(elapsed / 120 * 16)
        bar_str = "█" * filled + "░" * (16 - filled)
        t.add_row("", "", f"{bar_str} {elapsed}m", "", "", "")

    # Threshold hint — shows what's needed to exit MIXED
    t.add_row(
        "[dim]TREND:[/]",
        "", "",
        f"[dim]≥{ORB_CHAR_PL_STRONG:.2f}[/]",
        f"[dim]<{ORB_CHAR_MAE_CLEAN:.2f}[/]",
        "[dim](both→★★)[/]",
    )

    phase_labels = {"pre": "pre-RTH", "live": "live 8:30–10:30", "final": "final 8:30–10:30"}
    title = f"[bold]DAY CHARACTER[/]  [dim]({phase_labels.get(phase, '')})[/]"
    return Panel(t, title=title, border_style="blue", padding=(0, 1), expand=False)


def build_instrument_column(state: InstrumentState, now: datetime) -> Table:
    """Vertical stack of panels for one instrument (VOL REGIME + WATCHING only)."""
    col = Table.grid(padding=(0, 0))
    col.add_column()
    sym_text = Text(state.cfg.symbol, style="bold cyan", justify="center")
    if state.cname and state.cname != state.cfg.symbol:
        sym_text.append(f"  {state.cname}", style="dim cyan")
    col.add_row(Panel(sym_text, border_style="dark_blue", padding=(0, 1)))
    if not state.bars:
        msg = f"[red]{state.error}[/]" if state.error else "[dim]Waiting for bars…[/]"
        col.add_row(Panel(msg, border_style="dim", padding=(0, 1)))
        return col
    if state.error:
        col.add_row(Panel(f"[gold1]⚠ {state.error} — showing last known data[/]",
                          border_style="gold1", padding=(0, 1)))
    col.add_row(build_regime_panel(state))
    col.add_row(build_signal_panel(state, now))
    return col


_POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}

def build_history_table(history: list[RecentSignal], max_rows: int = 6) -> Panel:
    t = Table(box=box.SIMPLE, padding=(0, 1), show_header=True,
              header_style="bold dim")
    t.add_column("Time",    width=6)
    t.add_column("Close",   width=6)
    t.add_column("Sym",     width=5)
    t.add_column("Dir",     width=6)
    t.add_column("Entry",   width=10, justify="right")
    t.add_column("Target",  width=10, justify="right")
    t.add_column("Stop",    width=10, justify="right")
    t.add_column("Exit",    width=10, justify="right")
    t.add_column("Outcome", width=12)
    t.add_column("P&L pts", width=8,  justify="right")
    t.add_column("P&L $",   width=9,  justify="right")

    for rs in reversed(history[-max_rows:]):
        s  = rs.signal
        ts = s.bar_ts.astimezone(datetime.now().astimezone().tzinfo)
        kind = s.kind if isinstance(s, _HistSignal) else (
            "ORB" if isinstance(s, OrbSignal) else "")
        if kind == "ORB" or (isinstance(s, _HistSignal) and s.direction == 0):
            dir_str = "[green]ORB ↑[/]"
        elif kind == "VWASLR":
            dir_str = "[green]VWA ↑[/]" if s.direction == 1 else "[red]VWA ↓[/]"
        elif kind == "SLR":
            dir_str = "[green]SLR ↑[/]"
        else:
            dir_str = "[green]LONG[/]" if s.direction == 1 else "[red]SHORT[/]"

        pnl_col = "green" if rs.pnl_pts >= 0 else "red"
        pnl_sign = "+" if rs.pnl_pts >= 0 else "−"
        pnl_str = f"[{pnl_col}]{pnl_sign}{abs(rs.pnl_pts):.2f}[/]"
        if rs.outcome == "TARGET":
            out_str = "[green]HIT TARGET[/]"
        elif rs.outcome == "STOPPED":
            out_str = "[red]STOPPED[/]"
        elif rs.outcome in ("TRAIL STOP", "TRAIL"):
            out_str = "[cyan]TRAIL STOP[/]"
        elif rs.outcome == "TIME EXIT":
            out_str = "[cyan]TIME EXIT[/]"
        elif rs.outcome == "OPEN":
            out_str = "[bold cyan]OPEN[/]"
            pnl_str = "[dim]—[/]"
        else:
            out_str = f"[dim]{rs.outcome}[/]"

        if rs.outcome == "OPEN":
            exit_str = "[dim]—[/]"
        else:
            exit_price = s.entry + rs.pnl_pts * s.direction
            exit_str = f"[{pnl_col}]{exit_price:,.2f}[/]"

        pv = _POINT_VALUE.get(rs.symbol, 1.0)
        pnl_dollars = rs.pnl_pts * rs.contracts * pv
        if rs.outcome == "OPEN":
            dollar_str = "[dim]—[/]"
        else:
            d_col  = "green" if pnl_dollars >= 0 else "red"
            d_sign = "+" if pnl_dollars >= 0 else "−"
            dollar_str = f"[{d_col}]{d_sign}${abs(pnl_dollars):,.0f}[/]"

        if rs.exit_time is not None and rs.outcome != "OPEN":
            close_et = rs.exit_time.astimezone(datetime.now().astimezone().tzinfo)
            close_str = close_et.strftime("%H:%M")
        else:
            close_str = "[dim]—[/]"

        t.add_row(
            ts.strftime("%H:%M"),
            close_str,
            rs.symbol,
            dir_str,
            f"{s.entry:,.2f}",
            f"{s.target:,.2f}",
            f"{s.stop:,.2f}",
            exit_str,
            out_str,
            pnl_str,
            dollar_str,
        )

    return Panel(t, title="[bold]RECENT SIGNALS[/]",
                 border_style="blue", padding=(0, 1), expand=False)


def build_header(now: datetime) -> Panel:
    local = now.astimezone(datetime.now().astimezone().tzinfo)
    t = Text(justify="center")
    t.append("  SIGNAL MONITOR  ", style="bold white on dark_blue")
    t.append("  MES & MNQ  ", style="bold cyan")
    t.append("│  ")
    t.append(local.strftime("%a %Y-%m-%d  %H:%M:%S %Z"), style="dim")
    t.append("  │  next bar: ")
    nb_local = _next_bar_close(now).astimezone(datetime.now().astimezone().tzinfo)
    t.append(nb_local.strftime("%H:%M:%S"), style="cyan")
    return Panel(t, border_style="dark_blue", padding=(0, 0), expand=False)


def build_sizing_table(states: list[InstrumentState]) -> Panel:
    """Lot-size table: rows = $100/$200/$300/$400/$500 risk, columns = each instrument."""
    RISKS = [100, 200, 300, 400, 500]

    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    t.add_column("", justify="right")
    for s in states:
        t.add_column(s.cfg.symbol, justify="right", style="bold")

    # Sigma row
    partial = False
    for s in states:
        if s.sigma_pts and s.sigma_bar_count < TRAILING_BARS:
            partial = True

    sigma_bp_vals = []
    for s in states:
        if s.sigma_pts and s.bars:
            sigma_bp_vals.append(f"{2 * s.sigma_pts / s.bars[-1].close * 10000:.1f}")
        else:
            sigma_bp_vals.append("—")
    t.add_row("bp", *sigma_bp_vals, style="cyan")

    sigma_stop_vals = []
    for s in states:
        if s.sigma_pts:
            sigma_stop_vals.append(f"{2 * s.sigma_pts:.2f}")
        else:
            sigma_stop_vals.append("—")
    t.add_row("pts", *sigma_stop_vals, style="bold cyan")

    t.add_row("RISK", *[""] * len(states), style="")

    # Risk rows
    for risk in RISKS:
        cells = []
        for s in states:
            if s.sigma_pts and s.sigma_pts > 0:
                dollar_risk_per_lot = 2 * s.sigma_pts * s.cfg.point_value
                lots = risk / dollar_risk_per_lot
                cells.append(f"{lots:.1f}")
            else:
                cells.append("—")
        t.add_row(f"${risk}", *cells)

    title = "[bold]SIZING (2σ stop)[/]"
    if partial:
        title += "  [dim]* partial — warming up[/]"
    return Panel(t, title=title, border_style="blue", padding=(0, 1), expand=False)


def build_slr_trades_panel() -> Panel:
    return build_trade_summary_panel()


def render(states: list[InstrumentState],
           history: list[RecentSignal],
           now: datetime | None = None) -> Table:
    if now is None:
        now = datetime.now(timezone.utc)
    root = Table.grid(padding=(0, 0))
    root.add_column()

    root.add_row(build_header(now))

    slr_states    = [s for s in states if s.cfg.slr_enabled]
    orb_states    = [s for s in states if s.cfg.orb_enabled]
    vwaslr_states = [s for s in states if s.cfg.vwaslr_n > 0]

    # ── Row 2: [MES] [MNQ] [ORB] [VWASLR] [POSITIONS] ───────────────────────
    row2 = Table.grid(padding=(0, 1))
    row2.add_column()   # MES
    row2.add_column()   # MNQ
    if orb_states:
        row2.add_column()   # ORB (MES only)
    if vwaslr_states:
        row2.add_column()   # VWASLR (MES only)
    row2.add_column()   # POSITIONS

    orb_col = Table.grid(padding=(0, 0))
    orb_col.add_column()
    for s in orb_states:
        orb_col.add_row(build_orb_panel(s, now))

    vwaslr_col = Table.grid(padding=(0, 0))
    vwaslr_col.add_column()
    for s in vwaslr_states:
        vwaslr_col.add_row(build_vwaslr_panel(s))

    r2_cells = [build_instrument_column(s, now) for s in states]
    if orb_states:
        r2_cells.append(orb_col)
    if vwaslr_states:
        r2_cells.append(vwaslr_col)
    r2_cells.append(build_positions_panel(states))
    row2.add_row(*r2_cells)
    root.add_row(row2)

    # ── Row 3: [SLR MES] [SLR MNQ] ──────────────────────────────────────────
    if slr_states:
        row3 = Table.grid(padding=(0, 1), expand=True)
        for _ in slr_states:
            row3.add_column(ratio=1)
        row3.add_row(*[build_slr_scalp_panel(s, now) for s in slr_states])
        root.add_row(row3)

    # ── Row 4: [Trade Summary] [Sizing] ──────────────────────────────────────
    row4 = Table.grid(padding=(0, 1))
    row4.add_column()
    row4.add_column()
    row4.add_row(build_slr_trades_panel(), build_sizing_table(states))
    root.add_row(row4)

    # ── Row 5: Day Character ──────────────────────────────────────────────────
    if slr_states:
        root.add_row(build_orb_char_panel(slr_states, now))

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
    """Scan bars after the signal bar for target/stop hit. Returns (outcome, pnl_pts) or None.
    Conservative (adverse-first) ordering: stop checked before target within each bar."""
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if sig.direction == 1:   # LONG: low → stop, high → target
            if bar.low  <= sig.stop:
                return "STOPPED", -sig.stop_pts()
            if bar.high >= sig.target:
                return "TARGET",  sig.target_pts()
        else:                    # SHORT: high → stop, low → target
            if bar.high >= sig.stop:
                return "STOPPED", -sig.stop_pts()
            if bar.low  <= sig.target:
                return "TARGET",  sig.target_pts()
    return None


def _check_csr_trail_resolution(sig: Signal, bars: list[Bar],
                                 trail_sigma: float) -> tuple[str, float] | None:
    """
    Simulate a trailing stop through bars after the signal.
    Conservative OHLC ordering: adverse side tested before peak update within each bar.
    Returns (outcome, pnl_pts) or None if still open.
    """
    entry      = sig.entry
    d          = sig.direction
    trail_dist = trail_sigma * sig.sigma_pts
    peak       = entry
    trail_stop = entry - d * trail_dist

    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if d == 1:   # LONG
            if bar.low <= trail_stop:
                return "TRAIL STOP", trail_stop - entry
            if bar.high > peak:
                peak       = bar.high
                trail_stop = peak - trail_dist
        else:        # SHORT
            if bar.high >= trail_stop:
                return "TRAIL STOP", entry - trail_stop
            if bar.low < peak:
                peak       = bar.low
                trail_stop = peak + trail_dist
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
        orb.active_signal  = None
        orb.last_orb_bar_ts = None

    hm = (bar_et.hour, bar_et.minute)

    # ORB window: 9:30 through the close of bar 3 (9:30 + ORB_BARS * TF_MINUTES = 9:45).
    # Use <= so the 9:45-close bar (open-stamped 9:40) is counted; dedup by bar ts.
    orb_end_hm = (9, 30 + ORB_BARS * TF_MINUTES)  # (9, 45)
    if (9, 30) <= hm <= orb_end_hm and not orb.orb_complete:
        if orb.last_orb_bar_ts != bar.ts:   # only process each bar once
            orb.last_orb_bar_ts = bar.ts
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
    if window == "Morning" and orb.morning_fired:
        return None

    orb_width     = orb.orb_high - orb.orb_low
    orb_mid       = (orb.orb_high + orb.orb_low) / 2.0
    orb_width_pct = orb_width / orb_mid if orb_mid > 0 else 0.0
    if orb_width_pct < state.cfg.orb_width_pct_min:
        return None
    if state.sigma_pts <= 0:
        return None

    if bar.close > orb.orb_high:
        entry = bar.close
        sig = OrbSignal(
            entry=entry,
            target=entry + ORB_TGT_SIG * state.sigma_pts,
            stop=entry   - ORB_STOP_SIG * state.sigma_pts,
            orb_high=orb.orb_high, orb_low=orb.orb_low,
            sigma_pts=state.sigma_pts, window=window, bar_ts=bar.ts,
            direction=1,
        )
        if window == "Morning":
            orb.morning_fired = True
        orb.active_signal = sig
        return sig

    if bar.close < orb.orb_low:
        entry = bar.close
        sig = OrbSignal(
            entry=entry,
            target=entry - ORB_TGT_SIG * state.sigma_pts,
            stop=entry   + ORB_STOP_SIG * state.sigma_pts,
            orb_high=orb.orb_high, orb_low=orb.orb_low,
            sigma_pts=state.sigma_pts, window=window, bar_ts=bar.ts,
            direction=-1,
        )
        if window == "Morning":
            orb.morning_fired = True
        orb.active_signal = sig
        return sig

    return None


def _check_orb_resolution(sig: OrbSignal, bars: list[Bar]) -> tuple[str, float] | None:
    # Conservative (adverse-first): stop checked before target within each bar.
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if sig.direction == 1:   # LONG: low → stop, high → target
            if bar.low  <= sig.stop:
                return "STOPPED", -sig.stop_pts()
            if bar.high >= sig.target:
                return "TARGET",   sig.target_pts()
        else:                    # SHORT: high → stop, low → target
            if bar.high >= sig.stop:
                return "STOPPED", -sig.stop_pts()
            if bar.low  <= sig.target:
                return "TARGET",   sig.target_pts()
    return None


def build_orb_panel(state: InstrumentState, now: datetime) -> Panel:
    orb    = state.orb
    bar_et = state.bars[-1].ts.astimezone(ET) if state.bars else None

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=14, no_wrap=True)
    t.add_column(no_wrap=True)

    if orb.orb_complete:
        width     = orb.orb_high - orb.orb_low
        orb_mid   = (orb.orb_high + orb.orb_low) / 2.0
        width_pct = width / orb_mid if orb_mid > 0 else 0.0
        pct_min   = state.cfg.orb_width_pct_min
        pts_min   = pct_min * orb_mid
        wide      = width_pct >= pct_min
        w_style   = "green" if wide else "red"
        w_flag    = " ✓" if wide else f" ✗ need>{pts_min:.1f}"
        t.add_row("ORB high:", f"{orb.orb_high:,.2f}")
        t.add_row("ORB low:",  f"{orb.orb_low:,.2f}")
        t.add_row("ORB width:", f"[{w_style}]{width:.2f} pts ({width_pct*100:.3f}%){w_flag}[/]")
    elif orb.session_date == (bar_et.date() if bar_et else None):
        t.add_row("ORB:", f"Building… {orb.orb_bars_seen}/{ORB_BARS} bars")
    else:
        t.add_row("ORB:", "[dim]Waiting for RTH open[/]")

    if orb.active_signal:
        sig   = orb.active_signal
        is_long = sig.direction == 1
        rem   = (sig.bar_ts + timedelta(minutes=MAX_HOLD_MIN)) - now
        rem_s = int(rem.total_seconds())
        rem_str = f"{rem_s // 60}m {rem_s % 60:02d}s" if rem_s > 0 else "[blink]EXPIRED[/]"
        tgt_sign = "+" if is_long else "−"
        stp_sign = "−" if is_long else "+"
        ev_r     = 0.66 if is_long else 0.44
        t.add_row("", "")
        t.add_row("Entry:",  f"[bold]{sig.entry:,.2f}[/]  [dim]({sig.window})[/]")
        t.add_row("Target:", f"[bold green]{sig.target:,.2f}[/]  "
                              f"([green]{tgt_sign}{sig.target_pts():.2f} pts[/] │ {tgt_sign}{ORB_TGT_SIG:.1f}σ)")
        t.add_row("Stop:",   f"[bold red]{sig.stop:,.2f}[/]  "
                              f"([red]{stp_sign}{sig.stop_pts():.2f} pts[/] │ {stp_sign}{ORB_STOP_SIG:.1f}σ)")
        t.add_row("EV:",     f"+{ev_r:.2f}R ≈ +{ev_r*sig.risk_pts():.1f} pts  [{rem_str}]")
        title = "[bold green]⬤  ORB LONG ▲[/]" if is_long else "[bold red]⬤  ORB SHORT ▼[/]"
        border = "green" if is_long else "red"
        return Panel(t, title=title, border_style=border, padding=(0, 1))

    if orb.orb_complete:
        width     = orb.orb_high - orb.orb_low
        orb_mid_s = (orb.orb_high + orb.orb_low) / 2.0
        width_pct = width / orb_mid_s if orb_mid_s > 0 else 0.0
        window    = _orb_window(bar_et) if bar_et else None
        if width_pct < state.cfg.orb_width_pct_min:
            status = "[dim]ORB too narrow[/]"
        elif window:
            fired  = window == "Morning" and orb.morning_fired
            status = "[dim]Already fired[/]" if fired else \
                     f"Watch >{orb.orb_high:.2f} or <{orb.orb_low:.2f}"
        else:
            # All windows have passed (window is None and we're past the last one)
            last_window_end = max((eh * 60 + em) for _, _, eh, em, _ in ORB_WINDOWS)
            hm_mins = bar_et.hour * 60 + bar_et.minute if bar_et else 0
            all_windows_passed = hm_mins >= last_window_end
            status = "[dim]Done today[/]" if (orb.morning_fired or all_windows_passed) \
                     else "[dim]Waiting for window[/]"
        t.add_row("Status:", status)

    orb_wide = (orb.orb_complete and
                (orb.orb_high - orb.orb_low) / max((orb.orb_high + orb.orb_low) / 2, 1)
                >= state.cfg.orb_width_pct_min)
    border = "gold1" if (orb_wide and _orb_window(bar_et) and
                          not orb.morning_fired) else "blue"
    return Panel(t, title="[bold]ORB[/]", border_style=border, padding=(0, 1))


def _log_orb(sym: str, sig: OrbSignal, outcome: str,
             pnl_pts: float, resolved_at: datetime):
    ORB_LOG_PATH.parent.mkdir(exist_ok=True)
    if not ORB_LOG_PATH.exists():
        with open(ORB_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writeheader()
    row = {
        "fired_at": sig.bar_ts.isoformat(), "resolved_at": resolved_at.isoformat(),
        "symbol": sym, "direction": "LONG" if sig.direction == 1 else "SHORT",
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
                  signal_bar_ts: datetime, direction: int,
                  symbol: str = "") -> float | None:
    """
    Fetch PL_N_BARS 1-min bars ending just before the signal 5-min bar and
    return PL_aligned = (signed path length) × direction.
    +1 = 1-min flow perfectly aligned; ≥ PL_THRESH → 2× sizing.
    Returns None on fetch error or insufficient data.
    """
    from topstep_client import TopstepClient, get_bars_from_db, bars_db_available
    try:
        if symbol and bars_db_available():
            raw = get_bars_from_db(symbol, 1, PL_N_BARS + 5)
        else:
            end   = signal_bar_ts
            start = end - timedelta(minutes=PL_N_BARS + 5)
            raw = list(reversed(client.get_bars(
                contract_id=contract_id, start=start, end=end,
                unit=TopstepClient.MINUTE, unit_number=1,
                limit=PL_N_BARS + 5,
            )))
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


def fetch_live_bar(client, state: "InstrumentState", now: datetime) -> Bar | None:
    """
    Aggregate 1-min bars since the current 5-min bar's clock-aligned open into a
    partial bar.  Used for display only — no signal is fired from this value.
    """
    from topstep_client import TopstepClient
    epoch_min  = int(now.timestamp() // 60)
    open_min   = (epoch_min // TF_MINUTES) * TF_MINUTES
    bar_open   = datetime.fromtimestamp(open_min * 60, tz=timezone.utc)
    elapsed    = int((now - bar_open).total_seconds() // 60)
    if elapsed == 0:
        return None   # brand-new bar, nothing to show yet
    try:
        raw = client.get_bars(
            contract_id=state.cid,
            start=bar_open,
            end=now,
            unit=TopstepClient.MINUTE,
            unit_number=1,
            limit=TF_MINUTES + 1,
        )
        raw = list(reversed(raw))   # chronological order
    except Exception:
        return None
    # Keep only bars whose timestamp is >= bar_open (guard against off-by-one)
    mins = [b for b in raw if datetime.fromisoformat(b["t"]).replace(tzinfo=timezone.utc) >= bar_open]
    if not mins:
        return None
    return Bar(
        ts=bar_open,
        open=mins[0]["o"],
        high=max(b["h"] for b in mins),
        low=min(b["l"]  for b in mins),
        close=mins[-1]["c"],
        volume=sum(b["v"] for b in mins),
    )


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

    state.sigma           = sigma
    state.sigma_pts       = sigma_pts
    state.sigma_bar_count = len(trail)
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
        if _in_blackout(bar_hm, sh, sm, eh, em):
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


def backfill_orb_state(state: InstrumentState, client) -> None:
    """
    Reconstruct today's ORB range and morning-fired flag from historical bars.

    Called at startup and on session rollover so that starting signal_monitor
    after 9:45 ET still shows the correct ORB range and doesn't re-fire a
    morning signal that already happened.
    """
    if not state.cfg.orb_enabled:
        return

    now_et = datetime.now(ET)
    today  = now_et.date()

    # Nothing to backfill before the ORB window has opened
    from datetime import time as _dtime
    orb_start_et = datetime.combine(today, _dtime(9, 30), tzinfo=ET)
    if datetime.now(ET) < orb_start_et:
        return

    # Already built for today
    if state.orb.session_date == today and state.orb.orb_complete:
        return

    # Fetch from 1 min before 9:30 ET — API start is exclusive so 9:30 bar
    # would be dropped if start == bar.ts exactly.
    session_open = (datetime.combine(today, _dtime(9, 30), tzinfo=ET).astimezone(timezone.utc)
                    - timedelta(minutes=1))
    try:
        raw = client.get_bars(
            contract_id=state.cid,
            start=session_open,
            end=datetime.now(timezone.utc),
            unit=client.MINUTE,
            unit_number=TF_MINUTES,
            limit=200,
        )
    except Exception:
        return

    if not raw:
        return

    bars = sorted(
        [Bar(ts=datetime.fromisoformat(b["t"]),
             open=b["o"], high=b["h"], low=b["l"],
             close=b["c"], volume=b["v"]) for b in raw],
        key=lambda b: b.ts,
    )

    orb  = state.orb
    orb.session_date    = today
    orb.orb_high        = 0.0
    orb.orb_low         = 0.0
    orb.orb_bars_seen   = 0
    orb.orb_complete    = False
    orb.morning_fired   = False
    orb.active_signal   = None
    orb.last_orb_bar_ts = None

    for bar in bars:
        bar_et = bar.ts.astimezone(ET)
        hm     = (bar_et.hour, bar_et.minute)

        # Build ORB range 9:30–9:45 inclusive; bars are sorted so no dedup needed
        if (9, 30) <= hm <= (9, 30 + ORB_BARS * TF_MINUTES) and not orb.orb_complete:
            if orb.orb_bars_seen == 0:
                orb.orb_high = bar.high
                orb.orb_low  = bar.low
            else:
                orb.orb_high = max(orb.orb_high, bar.high)
                orb.orb_low  = min(orb.orb_low,  bar.low)
            orb.orb_bars_seen   += 1
            orb.last_orb_bar_ts  = bar.ts
            if orb.orb_bars_seen >= ORB_BARS:
                orb.orb_complete = True
            continue

        if not orb.orb_complete:
            continue

        # Check if morning window already produced a breakout (don't re-fire)
        morning_sh, morning_sm, morning_eh, morning_em = ORB_WINDOWS[0][:4]
        if (morning_sh, morning_sm) <= hm < (morning_eh, morning_em):
            if not orb.morning_fired and bar.close > orb.orb_high:
                orb.morning_fired = True


def _load_recent_history(hours: int = 24) -> list[RecentSignal]:
    """Read bot_trades.csv and orb_trades.csv and return entries from the past `hours`."""
    cutoff  = datetime.now(timezone.utc) - timedelta(hours=hours)
    entries: list[tuple[datetime, RecentSignal]] = []

    if BOT_LOG_PATH.exists():
        with open(BOT_LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    fired_at = datetime.fromisoformat(row["fired_at"])
                    if fired_at.tzinfo is None:
                        fired_at = fired_at.replace(tzinfo=timezone.utc)
                    if fired_at < cutoff:
                        continue
                    sig = _HistSignal(
                        bar_ts=fired_at,
                        direction=1 if row["direction"] == "LONG" else -1,
                        entry=float(row.get("fill_price") or row["est_entry"]),
                        target=float(row["target"]),
                        stop=float(row["stop"]),
                    )
                    exit_time = None
                    try:
                        raw = row.get("resolved_at", "")
                        if raw:
                            exit_time = datetime.fromisoformat(raw)
                            if exit_time.tzinfo is None:
                                exit_time = exit_time.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                    entries.append((fired_at, RecentSignal(
                        row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                        contracts=int(float(row.get("contracts") or 1)),
                        exit_time=exit_time)))
                except Exception:
                    continue

    if BOT_ORB_LOG_PATH.exists():
        with open(BOT_ORB_LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    fired_at = datetime.fromisoformat(row["fired_at"])
                    if fired_at.tzinfo is None:
                        fired_at = fired_at.replace(tzinfo=timezone.utc)
                    if fired_at < cutoff:
                        continue
                    sig = _HistSignal(
                        bar_ts=fired_at,
                        direction=1 if row["direction"] == "LONG" else -1,
                        entry=float(row.get("fill_price") or row["est_entry"]),
                        target=float(row["target"]),
                        stop=float(row["stop"]),
                        kind="ORB",
                    )
                    exit_time = None
                    try:
                        raw = row.get("resolved_at", "")
                        if raw:
                            exit_time = datetime.fromisoformat(raw)
                            if exit_time.tzinfo is None:
                                exit_time = exit_time.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                    entries.append((fired_at, RecentSignal(
                        row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                        exit_time=exit_time)))
                except Exception:
                    continue

    if VWAS_LOG_PATH.exists():
        with open(VWAS_LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    fired_at = datetime.fromisoformat(row["fired_at"])
                    if fired_at.tzinfo is None:
                        fired_at = fired_at.replace(tzinfo=timezone.utc)
                    if fired_at < cutoff:
                        continue
                    sig = _HistSignal(
                        bar_ts=fired_at,
                        direction=1 if row["direction"] == "LONG" else -1,
                        entry=float(row.get("fill_price") or row["est_entry"]),
                        target=float(row["target"]),
                        stop=float(row["stop"]),
                        kind="VWASLR",
                    )
                    exit_time = None
                    try:
                        raw = row.get("resolved_at", "")
                        if raw:
                            exit_time = datetime.fromisoformat(raw)
                            if exit_time.tzinfo is None:
                                exit_time = exit_time.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                    entries.append((fired_at, RecentSignal(
                        row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                        exit_time=exit_time)))
                except Exception:
                    continue

    if BOT_SLR_LOG_PATH.exists():
        with open(BOT_SLR_LOG_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    fired_at = datetime.fromisoformat(row["fired_at"])
                    if fired_at.tzinfo is None:
                        fired_at = fired_at.replace(tzinfo=timezone.utc)
                    if fired_at < cutoff:
                        continue
                    sig = _HistSignal(
                        bar_ts=fired_at,
                        direction=1,   # SLR is always LONG
                        entry=float(row.get("fill_price") or row["est_entry"]),
                        target=float(row["target"]),
                        stop=float(row["stop"]),
                        kind="SLR",
                    )
                    exit_time = None
                    try:
                        raw = row.get("resolved_at", "")
                        if raw:
                            exit_time = datetime.fromisoformat(raw)
                            if exit_time.tzinfo is None:
                                exit_time = exit_time.replace(tzinfo=timezone.utc)
                    except Exception:
                        pass
                    entries.append((fired_at, RecentSignal(
                        row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                        exit_time=exit_time)))
                except Exception:
                    continue

    entries.sort(key=lambda x: x[0])
    return [rs for _, rs in entries]


def _poll_vwaslr_new(since: datetime) -> tuple[list[RecentSignal], datetime]:
    """
    Return any VWASLR trades written to vwaslr_trades.csv with fired_at > since,
    plus the updated max timestamp (to pass as `since` on the next call).
    """
    if not VWAS_LOG_PATH.exists():
        return [], since
    new_entries: list[tuple[datetime, RecentSignal]] = []
    max_ts = since
    with open(VWAS_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fired_at = datetime.fromisoformat(row["fired_at"])
                if fired_at.tzinfo is None:
                    fired_at = fired_at.replace(tzinfo=timezone.utc)
                if fired_at <= since:
                    continue
                sig = _HistSignal(
                    bar_ts=fired_at,
                    direction=1 if row["direction"] == "LONG" else -1,
                    entry=float(row.get("fill_price") or row["est_entry"]),
                    target=float(row["target"]),
                    stop=float(row["stop"]),
                    kind="VWASLR",
                )
                exit_time = None
                try:
                    raw = row.get("resolved_at", "")
                    if raw:
                        exit_time = datetime.fromisoformat(raw)
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
                new_entries.append((fired_at, RecentSignal(
                    row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                    exit_time=exit_time)))
                if fired_at > max_ts:
                    max_ts = fired_at
            except Exception:
                continue
    new_entries.sort(key=lambda x: x[0])
    return [rs for _, rs in new_entries], max_ts


def _poll_csr_new(since: datetime) -> tuple[list[RecentSignal], datetime]:
    """Return CSR trades from bot_trades.csv with fired_at > since."""
    if not BOT_LOG_PATH.exists():
        return [], since
    new_entries: list[tuple[datetime, RecentSignal]] = []
    max_ts = since
    with open(BOT_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fired_at = datetime.fromisoformat(row["fired_at"])
                if fired_at.tzinfo is None:
                    fired_at = fired_at.replace(tzinfo=timezone.utc)
                if fired_at <= since:
                    continue
                sig = _HistSignal(
                    bar_ts=fired_at,
                    direction=1 if row["direction"] == "LONG" else -1,
                    entry=float(row.get("fill_price") or row["est_entry"]),
                    target=float(row["target"]),
                    stop=float(row["stop"]),
                )
                exit_time = None
                try:
                    raw = row.get("resolved_at", "")
                    if raw:
                        exit_time = datetime.fromisoformat(raw)
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
                new_entries.append((fired_at, RecentSignal(
                    row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                    contracts=int(float(row.get("contracts") or 1)),
                    exit_time=exit_time)))
                if fired_at > max_ts:
                    max_ts = fired_at
            except Exception:
                continue
    new_entries.sort(key=lambda x: x[0])
    return [rs for _, rs in new_entries], max_ts


def _poll_orb_new(since: datetime) -> tuple[list[RecentSignal], datetime]:
    """Return ORB trades from orb_trades.csv with fired_at > since."""
    if not BOT_ORB_LOG_PATH.exists():
        return [], since
    new_entries: list[tuple[datetime, RecentSignal]] = []
    max_ts = since
    with open(BOT_ORB_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fired_at = datetime.fromisoformat(row["fired_at"])
                if fired_at.tzinfo is None:
                    fired_at = fired_at.replace(tzinfo=timezone.utc)
                if fired_at <= since:
                    continue
                sig = _HistSignal(
                    bar_ts=fired_at,
                    direction=1 if row["direction"] == "LONG" else -1,
                    entry=float(row.get("fill_price") or row["est_entry"]),
                    target=float(row["target"]),
                    stop=float(row["stop"]),
                    kind="ORB",
                )
                exit_time = None
                try:
                    raw = row.get("resolved_at", "")
                    if raw:
                        exit_time = datetime.fromisoformat(raw)
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
                new_entries.append((fired_at, RecentSignal(
                    row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                    exit_time=exit_time)))
                if fired_at > max_ts:
                    max_ts = fired_at
            except Exception:
                continue
    new_entries.sort(key=lambda x: x[0])
    return [rs for _, rs in new_entries], max_ts


def _poll_slr_new(since: datetime) -> tuple[list[RecentSignal], datetime]:
    """Return SLR trades from slr_trades.csv with fired_at > since."""
    if not BOT_SLR_LOG_PATH.exists():
        return [], since
    new_entries: list[tuple[datetime, RecentSignal]] = []
    max_ts = since
    with open(BOT_SLR_LOG_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                fired_at = datetime.fromisoformat(row["fired_at"])
                if fired_at.tzinfo is None:
                    fired_at = fired_at.replace(tzinfo=timezone.utc)
                if fired_at <= since:
                    continue
                sig = _HistSignal(
                    bar_ts=fired_at,
                    direction=1,   # SLR is always LONG
                    entry=float(row.get("fill_price") or row["est_entry"]),
                    target=float(row["target"]),
                    stop=float(row["stop"]),
                    kind="SLR",
                )
                exit_time = None
                try:
                    raw = row.get("resolved_at", "")
                    if raw:
                        exit_time = datetime.fromisoformat(raw)
                        if exit_time.tzinfo is None:
                            exit_time = exit_time.replace(tzinfo=timezone.utc)
                except Exception:
                    pass
                new_entries.append((fired_at, RecentSignal(
                    row["symbol"], sig, row["outcome"], float(row["pnl_pts"]),
                    exit_time=exit_time)))
                if fired_at > max_ts:
                    max_ts = fired_at
            except Exception:
                continue
    new_entries.sort(key=lambda x: x[0])
    return [rs for _, rs in new_entries], max_ts


def run_live():
    from topstep_client import TopstepClient, get_bars_from_db, get_5s_bars_from_db, bars_db_available

    client = TopstepClient()
    client.use_shared_token()  # reuse bar_collector's token — avoids multiple-sessions disconnect

    accounts   = client.get_accounts()
    account_id = accounts[0]["id"] if accounts else None

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

    combined_history: list[RecentSignal] = _load_recent_history(hours=24)

    # Track the latest fired_at already loaded so live-polls only pick up new trades.
    _ts_min = datetime.min.replace(tzinfo=timezone.utc)
    csr_last_ts  = _ts_min
    orb_last_ts  = _ts_min
    vwas_last_ts = _ts_min
    slr_last_ts  = _ts_min
    for rs in combined_history:
        if not isinstance(rs.signal, _HistSignal):
            continue
        if rs.signal.kind == "VWASLR" and rs.signal.bar_ts > vwas_last_ts:
            vwas_last_ts = rs.signal.bar_ts
        elif rs.signal.kind == "ORB" and rs.signal.bar_ts > orb_last_ts:
            orb_last_ts = rs.signal.bar_ts
        elif rs.signal.kind == "SLR" and rs.signal.bar_ts > slr_last_ts:
            slr_last_ts = rs.signal.bar_ts
        elif rs.signal.kind == "" and rs.signal.bar_ts > csr_last_ts:
            csr_last_ts = rs.signal.bar_ts

    def fetch_bars(state: InstrumentState):
        max_mom  = max(bars for cfg in INSTRUMENTS for _, bars in cfg.csr_vol_windows)
        lookback = TRAILING_BARS + max_mom + 10
        try:
            now_utc  = datetime.now(timezone.utc)
            db_fresh = False
            if bars_db_available():
                raw = get_bars_from_db(state.cfg.symbol, TF_MINUTES, lookback)
                if raw:
                    db_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                                   open=b["o"], high=b["h"], low=b["l"],
                                   close=b["c"], volume=b["v"]) for b in raw]
                    state.bars = db_bars
                    # DB is fresh if newest bar is within 2 bar-widths old
                    db_fresh = (now_utc - db_bars[-1].ts).total_seconds() < TF_MINUTES * 60 * 2
            if not db_fresh:
                end   = now_utc
                start = end - timedelta(minutes=TF_MINUTES * lookback)
                raw = list(reversed(client.get_bars(
                    contract_id=state.cid, start=start, end=end,
                    unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
                    limit=lookback)))
                state.bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                                  open=b["o"], high=b["h"], low=b["l"],
                                  close=b["c"], volume=b["v"]) for b in raw]
            state.error = None
        except Exception as e:
            state.error = f"fetch error: {e}"

    def fetch_vwaslr_bars(state: InstrumentState):
        """Fetch 1-min bars for VWASLR. Uses DB when available and fresh;
        falls back to incremental REST fetching when DB is stale or absent."""
        try:
            now_utc   = datetime.now(timezone.utc)
            now_floor = datetime.fromtimestamp(
                (int(now_utc.timestamp()) // 60) * 60, tz=timezone.utc)
            db_fresh = False
            if bars_db_available():
                raw = get_bars_from_db(state.cfg.symbol, 1, VWASLR_INIT_BARS)
                db_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                               open=b["o"], high=b["h"], low=b["l"],
                               close=b["c"], volume=b["v"])
                           for b in raw
                           if datetime.fromisoformat(b["t"]) < now_floor]
                if db_bars:
                    # Only update if new bars are at least as recent as what we have
                    if (not state.vwaslr_bars
                            or db_bars[-1].ts >= state.vwaslr_bars[-1].ts):
                        state.vwaslr_bars = db_bars
                    # Consider DB fresh if newest bar is within 3 minutes
                    db_fresh = (now_utc - db_bars[-1].ts).total_seconds() < 180

            if not db_fresh:
                # DB absent, empty, or stale — use REST API
                if not state.vwaslr_bars:
                    end   = now_utc
                    start = end - timedelta(minutes=VWASLR_INIT_BARS + 30)
                    raw   = client.get_bars(contract_id=state.cid, start=start, end=end,
                                            unit=TopstepClient.MINUTE, unit_number=1,
                                            limit=VWASLR_INIT_BARS)
                    state.vwaslr_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                                             open=b["o"], high=b["h"], low=b["l"],
                                             close=b["c"], volume=b["v"])
                                         for b in reversed(raw)]
                else:
                    since = state.vwaslr_bars[-1].ts
                    raw   = client.get_bars(contract_id=state.cid, start=since, end=now_utc,
                                            unit=TopstepClient.MINUTE, unit_number=1,
                                            limit=10)
                    new_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                                    open=b["o"], high=b["h"], low=b["l"],
                                    close=b["c"], volume=b["v"])
                                for b in reversed(raw)]
                    for b in new_bars:
                        if b.ts > since:
                            state.vwaslr_bars.append(b)
                    if len(state.vwaslr_bars) > VWASLR_INIT_BARS + 200:
                        state.vwaslr_bars = state.vwaslr_bars[-(VWASLR_INIT_BARS + 100):]
        except Exception as e:
            state.error = f"vwaslr fetch error: {e}"

    def update_pl_bars(state: InstrumentState, now: datetime):
        """Fetch 1-min bars for display PL. Uses DB when available;
        falls back to incremental REST fetching."""
        try:
            if bars_db_available():
                raw = get_bars_from_db(state.cfg.symbol, 1, PL_N_BARS + 5)
                state.pl_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                                     open=b["o"], high=b["h"], low=b["l"],
                                     close=b["c"], volume=b["v"])
                                 for b in raw]
            elif not state.pl_bars:
                start = now - timedelta(minutes=PL_N_BARS + 5)
                raw   = client.get_bars(contract_id=state.cid, start=start, end=now,
                                        unit=TopstepClient.MINUTE, unit_number=1,
                                        limit=PL_N_BARS + 5)
                state.pl_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                                     open=b["o"], high=b["h"], low=b["l"],
                                     close=b["c"], volume=b["v"])
                                 for b in reversed(raw)]
            else:
                since = state.pl_bars[-1].ts
                raw   = client.get_bars(contract_id=state.cid, start=since, end=now,
                                        unit=TopstepClient.MINUTE, unit_number=1,
                                        limit=3)
                for b in reversed(raw):
                    bar = Bar(ts=datetime.fromisoformat(b["t"]),
                              open=b["o"], high=b["h"], low=b["l"],
                              close=b["c"], volume=b["v"])
                    if bar.ts > since:
                        state.pl_bars.append(bar)
                if len(state.pl_bars) > PL_N_BARS + 10:
                    state.pl_bars = state.pl_bars[-(PL_N_BARS + 5):]
            # Recompute current_pl from cached bars
            if len(state.pl_bars) >= PL_N_BARS + 1:
                closes   = np.array([b.close for b in state.pl_bars[-(PL_N_BARS + 1):]])
                rets     = np.log(closes[1:] / closes[:-1])
                sum_absr = float(np.abs(rets).sum())
                state.current_pl = float(rets.sum()) / sum_absr if sum_absr > 0 else None
        except Exception:
            pass  # keep last known value on error

    def resolve(state: InstrumentState, outcome: str, pnl_pts: float, now: datetime):
        state.active_signal = None

    def refresh_orb_char(state: InstrumentState):
        """Fetch the last RTH close and overnight metrics, cached once per day."""
        today = datetime.now(ET).date()
        if state.orb_char_date == today and state.orb_char_ovn_cache is not None:
            return
        # New day — reset overnight cache
        if state.orb_char_date != today:
            state.orb_char_ovn_cache = None
        try:
            if bars_db_available():
                raw = get_bars_from_db(state.cfg.symbol, 1, ORB_CHAR_FETCH_BARS)
            else:
                end   = datetime.now(timezone.utc)
                start = end - timedelta(minutes=ORB_CHAR_FETCH_BARS + 30)
                raw   = list(reversed(client.get_bars(
                    contract_id=state.cid, start=start, end=end,
                    unit=2, unit_number=1, limit=ORB_CHAR_FETCH_BARS)))
            if not raw:
                return
            t_rth_utc = datetime(today.year, today.month, today.day,
                                 8, 30, tzinfo=ET).astimezone(timezone.utc)
            t_ovn_s   = t_rth_utc - timedelta(hours=15, minutes=30)  # ~5pm ET yesterday
            prev_bars = [b for b in raw
                         if datetime.fromisoformat(b["t"]).replace(tzinfo=timezone.utc) < t_rth_utc]
            if not prev_bars:
                return
            state.orb_char_prev_close = prev_bars[-1]["c"]
            state.orb_char_date       = today
            # Compute and cache overnight metrics from the full bar history
            prev_close = state.orb_char_prev_close
            ovn_bars = [b for b in raw
                        if t_ovn_s <= datetime.fromisoformat(b["t"]).replace(tzinfo=timezone.utc) < t_rth_utc]
            rth_bars = [b for b in raw
                        if datetime.fromisoformat(b["t"]).replace(tzinfo=timezone.utc) >= t_rth_utc]
            if rth_bars:
                rth_open = rth_bars[0]["o"]
                state.orb_char_rth_open   = rth_open
                state.orb_char_gap_bp     = (rth_open - prev_close) / prev_close * 10000
            if ovn_bars:
                ovn_low  = min(b["l"] for b in ovn_bars)
                ovn_high = max(b["h"] for b in ovn_bars)
                state.orb_char_ovn_low_bp  = (ovn_low  - prev_close) / prev_close * 10000
                state.orb_char_ovn_high_bp = (ovn_high - prev_close) / prev_close * 10000
        except Exception:
            pass

    # Backfill at startup: fetch bars + VWASLR, run evaluate() so sigma_pts is
    # ready for the first render, then stamp the throttle mins so the first
    # loop iteration doesn't redundantly re-fetch everything.
    _startup_min = datetime.now(timezone.utc).minute
    n_instruments = len(states)
    for i, state in enumerate(states, 1):
        console.print(f"  [{i}/{n_instruments}] {state.cfg.symbol}: fetching bars…", end=" ")
        fetch_bars(state)
        state.bars_fetch_min = _startup_min
        if state.bars:
            evaluate(state)          # populate sigma_pts immediately
        if state.cfg.vwaslr_n > 0 or state.cfg.slr_enabled:
            console.print("vwaslr…", end=" ")
            fetch_vwaslr_bars(state)
            state.vwaslr_fetch_min = _startup_min
            state.vwaslr_last_fetch = datetime.now(timezone.utc)
        if state.cfg.orb_enabled:
            console.print("orb…", end=" ")
            backfill_orb_state(state, client)
        if state.cfg.slr_enabled:
            console.print("orb-char…", end=" ")
            refresh_orb_char(state)
        console.print("done")

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            now = datetime.now(timezone.utc)

            # ── Fast path: bars fetch (serial, ~6 s for 3 instruments) ───────────
            # Runs first so Scaled Return / sigma update immediately at each
            # new 5-min bar, without waiting behind pl/live_bar/vwaslr calls.
            if any(now.minute != s.bars_fetch_min for s in states):
                for state in states:
                    if now.minute != state.bars_fetch_min:
                        fetch_bars(state)
                        state.bars_fetch_min = now.minute
                        if state.bars:
                            state.current_ha_streak = _ha_streak(state.bars)
                live.update(render(states, combined_history))

            # ── Slow path: pl_bars, live_bar, vwaslr (background thread) ─────────
            def _fetch_slow(state: InstrumentState):
                if not state.bars:
                    return
                if state.cfg.slr_enabled:
                    refresh_orb_char(state)   # no-op if already done today
                if now.minute != state.pl_fetch_min:
                    update_pl_bars(state, now)
                    state.pl_fetch_min = now.minute
                if now.minute != state.live_bar_fetch_min:
                    state.live_bar           = fetch_live_bar(client, state, now)
                    state.live_bar_fetch_min = now.minute
                if state.cfg.vwaslr_n > 0 or state.cfg.slr_enabled:
                    # Re-fetch on new clock minute OR if the latest bar is stale
                    # (API sometimes takes 60+ s to publish a completed 1-min bar;
                    # retry every 15 s until the bar advances rather than waiting
                    # the full next clock minute).
                    _expected_1min = datetime.fromtimestamp(
                        (int(now.timestamp()) // 60) * 60 - 60, tz=timezone.utc)
                    _vwas_stale = (
                        bool(state.vwaslr_bars)
                        and state.vwaslr_bars[-1].ts < _expected_1min
                        and state.vwaslr_last_fetch is not None
                        and (now - state.vwaslr_last_fetch).total_seconds() >= 15
                    )
                    if now.minute != state.vwaslr_fetch_min or _vwas_stale:
                        fetch_vwaslr_bars(state)
                        state.vwaslr_last_fetch = now
                        if account_id:
                            try:
                                positions = client.get_open_positions(account_id)
                                pos = next((p for p in positions
                                            if str(p.get("contractId", "")) == str(state.cid)), None)
                                state.has_vwaslr_position = pos is not None
                                sz = int(pos.get("size", 0)) if pos else 0
                                # ProjectX API: size is always positive; type 1=Long, 2=Short
                                pos_type = int(pos.get("type", 0)) if pos else 0
                                prev_size = state.position_size
                                state.position_size      = abs(sz)
                                state.position_entry     = float(pos.get("averagePrice", 0) or 0) if pos and sz else None
                                if sz == 0:
                                    state.position_direction = 0
                                else:
                                    state.position_direction = -1 if pos_type == 2 else 1
                                if state.position_size > 0 and prev_size == 0:
                                    state.position_strategy = _detect_position_strategy(state.cfg.symbol)
                                elif state.position_size == 0:
                                    state.position_strategy = ""
                            except Exception as e:
                                pass
                        state.vwaslr_fetch_min = now.minute
                    # EMA advances once per new 1-min bar (not every loop tick).
                    # Gate on bar timestamp so history and EMA stay in sync.
                    _cur_bar_ts = state.vwaslr_bars[-1].ts if state.vwaslr_bars else None
                    if _cur_bar_ts and _cur_bar_ts != state.vwaslr_ema_bar_ts:
                        raw = _compute_vwaslr(
                            state.vwaslr_bars, state.cfg.vwaslr_n, VWASLR_SIGMA_BARS)
                        alpha = 2.0 / (VWASLR_EMA_SPAN + 1)
                        state.vwaslr_ema_prev = state.vwaslr_ema
                        state.vwaslr_ema = alpha * raw + (1.0 - alpha) * state.vwaslr_ema
                        state.current_vwaslr = state.vwaslr_ema
                        state.vwaslr_ema_bar_ts = _cur_bar_ts
                        thr      = state.cfg.vwaslr_threshold
                        half_thr = thr / 2
                        # Entry: EMA crosses ±threshold; Exit: EMA retracts below ±half_thr
                        if abs(state.vwaslr_ema) >= thr and abs(state.vwaslr_ema_prev) < thr:
                            state.vwaslr_entry = (state.vwaslr_bars[-1].close
                                                  if state.vwaslr_bars else state.bars[-1].close)
                        elif abs(state.vwaslr_ema) < half_thr and state.vwaslr_entry is not None:
                            state.vwaslr_entry = None


            import threading as _threading
            _fetch_done = _threading.Event()
            def _run_slow():
                with ThreadPoolExecutor(max_workers=len(states)) as ex:
                    list(ex.map(_fetch_slow, states))
                _fetch_done.set()
            _threading.Thread(target=_run_slow, daemon=True).start()
            while not _fetch_done.wait(timeout=1):
                live.update(render(states, combined_history))
            live.update(render(states, combined_history))

            # ── 5-min bar retry ───────────────────────────────────────────────────
            # The API takes ~20 s to publish a completed bar after the close.
            # We fire when either:
            #   • just past a 5-min boundary (< 60 s in): wait the remaining
            #     time to reach T+20 s, then retry.
            #   • approaching a boundary (≤ 20 s away): wait through it to
            #     T+20 s, then retry.  (Entered via the early display-loop break.)
            # Uses live clock so it works regardless of when `now` was captured.
            _rn        = datetime.now(timezone.utc)
            _period_s  = TF_MINUTES * 60
            _epoch_s   = int(_rn.timestamp())
            _secs_in   = _epoch_s % _period_s          # seconds since last bar close
            _secs_to   = _period_s - _secs_in          # seconds until next bar close

            if _secs_in < 60 or _secs_to <= 20:
                # Bar open-timestamp we expect to see after the (upcoming) close.
                # Pre-boundary: the bar opening at the current period start closes
                #               at the next boundary — that's what we're waiting for.
                # Post-boundary: the bar that just closed opened one period earlier.
                if _secs_to <= 20:   # pre-boundary
                    _expected_bar_ts = datetime.fromtimestamp(
                        (_epoch_s // _period_s) * _period_s, tz=timezone.utc
                    )
                else:                # post-boundary
                    _expected_bar_ts = datetime.fromtimestamp(
                        (_epoch_s // _period_s) * _period_s - _period_s, tz=timezone.utc
                    )
                needs_retry = any(
                    s.bars and s.bars[-1].ts < _expected_bar_ts for s in states
                )
                if needs_retry:
                    # Wait until T+20 s past the bar close, ticking the display.
                    # Retry up to 3 times (15 s apart) for contracts whose bar
                    # the API publishes late — stops as soon as all are updated.
                    if _secs_to <= 20:
                        _wait = _secs_to + 20   # cross boundary then 20 s more
                    else:
                        _wait = max(0, 20 - _secs_in)
                    for _attempt in range(3):
                        for _ in range(int(_wait) + 1):
                            live.update(render(states, combined_history))
                            time.sleep(1)
                        for state in states:
                            if state.bars and state.bars[-1].ts >= _expected_bar_ts:
                                continue   # already have this bar
                            fetch_bars(state)
                            if state.bars:
                                state.current_ha_streak = _ha_streak(state.bars)
                        # Update throttle only for states that now have the bar
                        _cur_min = datetime.now(timezone.utc).minute
                        for state in states:
                            if state.bars and state.bars[-1].ts >= _expected_bar_ts:
                                state.bars_fetch_min = _cur_min
                        live.update(render(states, combined_history))
                        if not any(s.bars and s.bars[-1].ts < _expected_bar_ts
                                   for s in states):
                            break          # all contracts updated — done
                        _wait = 15         # subsequent retries: 15 s apart

            for state in states:
                if not state.bars:
                    continue

                # If ORB window has closed but ORB is still incomplete, backfill now
                # (handles bar_collector downtime during the 9:30–9:45 build window)
                if (state.cfg.orb_enabled and not state.orb.orb_complete
                        and datetime.now(ET).hour * 60 + datetime.now(ET).minute
                            > 9 * 60 + 30 + ORB_BARS * TF_MINUTES):
                    backfill_orb_state(state, client)

                # Update display metrics and check for signal
                new_sig = evaluate(state)
                new_orb = evaluate_orb(state) if state.cfg.orb_enabled else None

                # Only act on a signal from a bar we haven't seen before
                last_bar_ts = state.bars[-1].ts
                if last_bar_ts == state.last_evaluated_ts:
                    new_sig = None
                    new_orb = None
                else:
                    state.last_evaluated_ts = last_bar_ts

                # Check momentum signal for target/stop hit or expiry
                if state.active_signal:
                    sig = state.active_signal

                    # Update trailing stop display state
                    trail_dist  = CSR_TRAIL_SIGMA * sig.sigma_pts
                    bars_after  = [b for b in state.bars if b.ts > sig.bar_ts]
                    if bars_after:
                        if sig.direction == 1:
                            state.csr_trail_peak = max(b.high for b in bars_after)
                        else:
                            state.csr_trail_peak = min(b.low for b in bars_after)
                        state.csr_trail_stop = (
                            state.csr_trail_peak - sig.direction * trail_dist
                        )
                    else:
                        state.csr_trail_peak = sig.entry
                        state.csr_trail_stop = sig.entry - sig.direction * trail_dist

                    hit = _check_csr_trail_resolution(sig, state.bars, CSR_TRAIL_SIGMA)
                    if hit:
                        resolve(state, hit[0], hit[1], now)
                        state.csr_trail_peak = None
                        state.csr_trail_stop = None
                    elif now >= sig.expires_at:
                        last_close = state.bars[-1].close
                        pnl = (last_close - sig.entry) * sig.direction
                        resolve(state, "TIME EXIT", pnl, now)
                        state.csr_trail_peak = None
                        state.csr_trail_stop = None

                if new_sig and (state.active_signal is None or
                                new_sig.bar_ts != state.active_signal.bar_ts):
                    if state.active_signal:
                        resolve(state, "SUPERSEDED", 0.0, now)
                        state.csr_trail_peak = None
                        state.csr_trail_stop = None
                    state.active_signal = new_sig
                    pl = fetch_1min_pl(client, state.cid,
                                       new_sig.bar_ts, new_sig.direction,
                                       symbol=state.cfg.symbol)
                    if pl is not None:
                        state.active_signal.pl_aligned = pl
                    play_alert()

                # Check ORB signal for target/stop hit or expiry (display state only)
                if state.orb.active_signal:
                    hit = _check_orb_resolution(state.orb.active_signal, state.bars)
                    if hit:
                        state.orb.active_signal = None
                    elif now >= state.orb.active_signal.bar_ts + timedelta(minutes=MAX_HOLD_MIN):
                        state.orb.active_signal = None

                if new_orb and state.orb.active_signal is None:
                    state.orb.active_signal = new_orb
                    play_alert()

                # ── SLR_Scalp detection and resolution (1-min bar gated) ────────
                if state.cfg.slr_enabled and state.vwaslr_bars:
                    _slr_bar_ts = state.vwaslr_bars[-1].ts

                    # Resolve active signal first (stop/target/time)
                    if state.active_slr_signal:
                        _slr_sig = state.active_slr_signal
                        _slr_hit = _check_slr_resolution(_slr_sig, state.vwaslr_bars)
                        if _slr_hit:
                            state.active_slr_signal = None
                        elif now >= _slr_sig.expires_at():
                            state.active_slr_signal = None

                    # Detect new signal only on a new 1-min bar
                    if _slr_bar_ts != state.slr_eval_bar_ts:
                        state.slr_eval_bar_ts = _slr_bar_ts
                        if state.active_slr_signal is None:
                            _new_slr = evaluate_slr_scalp(state)
                            if _new_slr is not None:
                                state.slr_last_surge_ts = _new_slr.surge_ts
                                state.active_slr_signal = _new_slr
                                play_alert()

            # Pick up any trades logged by trading_bot since last poll
            new_csr,  csr_last_ts  = _poll_csr_new(csr_last_ts)
            new_orb,  orb_last_ts  = _poll_orb_new(orb_last_ts)
            new_vwas, vwas_last_ts = _poll_vwaslr_new(vwas_last_ts)
            new_slr,  slr_last_ts  = _poll_slr_new(slr_last_ts)
            if new_csr or new_orb or new_vwas or new_slr:
                combined_history.extend(new_csr)
                combined_history.extend(new_orb)
                combined_history.extend(new_vwas)
                combined_history.extend(new_slr)
                combined_history.sort(key=lambda r: r.signal.bar_ts)

            # Display loop: tick every second.
            # Break at any minute change (picks up new bars quickly), AND
            # break 20 s before each 5-min bar close so the retry window
            # starts before the boundary rather than a full minute after it.
            loop_start_min = now.minute
            for _ in range(30):
                live.update(render(states, combined_history))
                time.sleep(1)
                _cur = datetime.now(timezone.utc)
                if _cur.minute != loop_start_min:
                    break
                # Pre-emptive break: 20 s before a 5-min bar close
                if (TF_MINUTES * 60) - (int(_cur.timestamp()) % (TF_MINUTES * 60)) <= 20:
                    break


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
    mes_state.current_ha_streak = 3           # demo: 3 green bars
    mes_state.current_vwaslr    = 0.72       # demo: building toward threshold
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
    mym_state.current_vwaslr = 0.63          # demo: above threshold (thr=0.5)
    mym_state.vwaslr_entry   = 46_500
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
    mes_state2.current_vwaslr = -1.18        # demo: SHORT signal (below -1.0)
    mes_state2.vwaslr_entry   = 6_620.0
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
    mym_state2.sigma        = mym_sigma
    mym_state2.sigma_pts    = mym_sp
    mym_state2.gk_ann_vol   = 0.198
    mym_state2.mean_vol     = 4_200.0
    mym_state2.current_vwaslr = -0.38       # demo: below threshold (thr=0.5), watching
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
