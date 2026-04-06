"""
Automated trading bot for the 3σ MES/MYM/M2K continuation signal.

Monitors 5-min bars every 30 seconds, detects signals using the same criteria
as signal_monitor.py, places market orders with native API bracket
stops/targets, force-closes at 15-minute expiry if brackets not hit, and
logs every trade outcome to logs/bot_trades.csv.

Signal types:

  Primary (3σ CSR momentum) — 2σ bracket stop (safety net) + 0.5σ software trailing stop:
  1. |bar_return / σ| ≥ 3.0   (σ = trailing 100-min close-return std dev)
  2. Volume ≥ 1.5× trailing mean volume
  3. 40-min CSR ≥ 1.5σ aligned with signal direction (momentum filter)
  4. Not in instrument-specific blackout window
  5. |scaled| ≤ 5.0 (extreme event filter)

  ORB (opening range breakout, MES/MYM/M2K):
  - 15-min opening range (9:30–9:45 ET); breakout in morning window (9:45–10:30 ET)
  - ORB width ≥ instrument-specific wide-range cutoff (from backtest)

  VWASLR (volume-weighted avg scaled log return, MES/MYM):
  - 1-min bars; EMA-10(VWASLR(50min, σ=500min)) crosses ±0.4σ; M2K disabled (too noisy)
  - MES/MYM: active 8:30–16:00 ET (pre-open edge confirmed)
  - Exit: EMA retracts below ±0.2σ (half-zero); bracket orders remain as safety net
  - Separate incremental 1-min bar fetch (initial 565 bars, then new bars only each poll)

Only one position per instrument at a time (all signal types share the lock).

Usage:
  python src/trading_bot.py                  # live trading (requires .env)
  python src/trading_bot.py --paper          # paper mode: signals logged, no orders placed
  python src/trading_bot.py --account 12345  # specify account ID explicitly
"""

import argparse
import csv
import logging
import math
import os
import random
import subprocess
import time
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from topstep_client import TopstepClient

# Practice account — the only account the live bot is permitted to trade on.
# Set TOPSTEP_ACCOUNT_ID in .env to override.
PRACTICE_ACCOUNT_ID = int(os.environ.get("TOPSTEP_ACCOUNT_ID", "10634862"))
PRACTICE_ACCOUNT_NAME = "PRAC-V2-88916-19336808"

# ── Parameters (keep in sync with signal_monitor.py) ────────────────────────

TF_MINUTES    = 5
TRAILING_BARS = 20     # 20 × 5-min = 100-min σ window
MOM_BARS      = 8      # 8 × 5-min = 40-min CSR window (default; overridden dynamically)
CSR_THRESHOLD = 1.5    # min cumulative scaled return aligned with signal direction
SIGNAL_SIGMA  = 3.0
MAX_SCALED    = 5.0    # ignore extreme event spikes
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 25     # force-close after this many minutes
POLL_SECONDS     = 40   # default poll interval (VWASLR on 1-min bars; 60s was too coarse)
POLL_SECONDS_ORB = 20   # faster poll during the ORB window (9:30–10:30 ET)

PL_N_BARS = 10    # 1-min bars to look back for PL computation
PL_THRESH = 0.50  # PL_aligned ≥ this → 2× sizing

# ── VWASLR parameters (keep in sync with signal_monitor.py) ─────────────────
# 1-min bars: N=50 (50-min window), σ=500 bars (500-min window)
# Entry: EMA-10 of raw VWASLR crosses ±threshold.
# Exit:  EMA retracts below ±(threshold/2) — "half-zero" signal exit.
VWASLR_SIGMA_BARS   = 500   # 500 × 1-min = 500-min σ window (slow/stable)
VWASLR_N            = 50    # 50 × 1-min = 50-min signal window
VWASLR_INIT_BARS    = VWASLR_SIGMA_BARS + VWASLR_N + 15  # initial 1-min fetch (565 bars)
VWASLR_EMA_SPAN     = 10    # EMA span applied to raw VWASLR (α = 2/11 ≈ 0.18)
VWASLR_STOP_SIGMA   = 2.0
VWASLR_TARGET_SIGMA = 3.0

# ── Trailing stops (keep in sync with signal_monitor.py) ────────────────────
# Software trailing stops replace fixed targets/stops for signal monitoring.
# The bracket stop (stop_sigma=2σ) stays as a hard safety net on the API.
# Trail fires when price retraces TRAIL_SIGMA * sigma_pts from its peak.
CSR_TRAIL_SIGMA    = 0.5   # tight — CSR momentum fades quickly after initial spike

# ── ORB parameters (keep in sync with signal_monitor.py) ────────────────────
ORB_BARS     = 3    # 3 × 5-min = 15-min opening range
ORB_STOP_SIG = 2.0
ORB_TGT_SIG  = 2.0  # 2σ:2σ → EV ≈ +0.61R
ORB_WINDOWS  = [    # (start_h, start_m, end_h, end_m, label)
    (9,  45, 10, 30, "Morning"),
]

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")

TRADING_CUTOFF_CT = (15, 10)   # TopstepX closes at 15:10 CT; no new entries after this

LOG_PATH      = Path("logs/bot_trades.csv")
ORB_LOG_PATH  = Path("logs/orb_trades.csv")
VWAS_LOG_PATH = Path("logs/vwaslr_trades.csv")
VWAS_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "target", "stop",
    "sigma_pts", "vwaslr", "outcome", "pnl_pts", "pnl_sigma",
]
ORB_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "target", "stop",
    "orb_high", "orb_low", "orb_width", "sigma_pts",
    "window", "outcome", "pnl_pts", "pnl_r",
]
LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "target", "stop",
    "sigma_pts", "scaled", "vol_ratio", "csr",
    "pl_aligned", "contracts",
    "outcome", "pnl_pts", "pnl_sigma",
]

log = logging.getLogger("bot")

TRADE_SOUND = "/System/Library/Sounds/Hero.aiff"

def play_trade_sound():
    """Play trade-execution sound non-blocking."""
    threading.Thread(
        target=lambda: subprocess.run(["afplay", TRADE_SOUND], check=False),
        daemon=True,
    ).start()


# ── Instrument config ────────────────────────────────────────────────────────

@dataclass
class BotInstrument:
    symbol:       str
    search_term:  str
    stop_sigma:   float = 2.0
    target_sigma: float = 3.0
    tick_size:    float = 0.25   # minimum price increment
    point_value:  float = 5.00  # $ per point (informational only)
    # Dynamic CSR window: list of (gk_ann_vol_upper_bound, mom_bars)
    csr_vol_windows: list = field(default_factory=lambda: [(1.0, 8)])
    # Per-instrument blackout windows: (start_h, start_m, end_h, end_m, conditional)
    # conditional=True: block only when CSR < threshold; False: always block.
    blackout_windows: list = field(default_factory=list)
    # ORB: set orb_enabled=True and orb_width_pct_min to wide-tertile cutoff from backtest.
    # Width threshold is a fraction of ORB midpoint price (e.g. 0.00354 = 0.354%).
    orb_enabled:       bool  = False
    orb_width_pct_min: float = 0.0
    # VWASLR: 0 = disabled. n = look-back bars; threshold = signal level in σ/bar units.
    # vwaslr_start: earliest (hour, minute) ET for VWASLR signals (default 9:30 RTH open).
    vwaslr_n:         int   = 0
    vwaslr_threshold: float = 1.0
    vwaslr_start:     tuple = (9, 30)


INSTRUMENTS = [
    BotInstrument("MES", "MES", tick_size=0.25, point_value=5.00,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (18,  0,  8,  0, False),  # overnight Globex: no edge, unvalidated
                      (8,   0,  9,  0, True),   # econ releases: block CSR if below 1.5σ (VWASLR skips conditional pre-9:30)
                      (15, 45, 18,  0, False),  # EOD volatility + daily break until Globex open
                  ],
                  orb_enabled=True, orb_width_pct_min=0.00354,
                  vwaslr_n=50, vwaslr_threshold=0.4, vwaslr_start=(8, 30)),  # VWASLR 8:30→16:00 ET; EMA=10 half-zero exit
    BotInstrument("MYM", "MYM", tick_size=1.00, point_value=0.50,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (18,  0,  8, 30, False),  # overnight Globex (VWASLR active from 8:30)
                      (15, 45, 18,  0, False),  # EOD volatility + daily break until Globex open
                  ],
                  orb_enabled=True, orb_width_pct_min=0.00402,
                  vwaslr_n=50, vwaslr_threshold=0.4, vwaslr_start=(8, 30)),  # VWASLR 8:30→16:00 ET; EMA=10 half-zero exit
    BotInstrument("M2K", "M2K", tick_size=0.10, point_value=5.00,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (18,  0,  8,  0, False),  # overnight Globex: no edge, unvalidated
                      (8,   0,  9,  0, True),   # econ releases: block only if CSR<1.5
                      (15, 45, 18,  0, False),  # EOD volatility + daily break until Globex open
                  ],
                  orb_enabled=True, orb_width_pct_min=0.00715,
                  vwaslr_n=0),  # VWASLR disabled — Russell 2000 too noisy at any threshold tested
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts:     datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class ActiveTrade:
    instrument:  BotInstrument
    contract_id: str
    direction:   int        # +1 long / -1 short
    est_entry:   float      # signal bar close (pre-fill estimate)
    sigma_pts:   float
    scaled:      float
    vol_ratio:   float
    csr:         float
    fired_at:    datetime
    pl_aligned:       float | None = None
    contracts:        int = 1
    order_id:         int | None = None
    fill_price:       float | None = None
    trail_peak:       float | None = None   # most favourable price seen since fill
    trail_stop_level: float | None = None   # current trailing stop price
    expires_at:       datetime = field(init=False)

    def __post_init__(self):
        self.expires_at = self.fired_at + timedelta(minutes=MAX_HOLD_MIN)

    def target_price(self) -> float:
        p = self.fill_price or self.est_entry
        return p + self.direction * self.instrument.target_sigma * self.sigma_pts

    def stop_price(self) -> float:
        p = self.fill_price or self.est_entry
        return p - self.direction * self.instrument.stop_sigma * self.sigma_pts


@dataclass
class OrbSignal:
    entry:     float
    target:    float
    stop:      float
    orb_high:  float
    orb_low:   float
    sigma_pts: float
    window:    str
    bar_ts:    datetime
    direction: int = 1    # 1 = LONG, -1 = SHORT

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


@dataclass
class VwasrlSignal:
    entry:     float
    target:    float
    stop:      float
    sigma_pts: float
    vwaslr:    float
    bar_ts:    datetime
    direction: int = 1

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)


@dataclass
class ActiveVwasrlTrade:
    instrument:  BotInstrument
    contract_id: str
    sig:         VwasrlSignal
    fired_at:    datetime
    order_id:         int | None = None
    fill_price:       float | None = None
    trail_peak:       float | None = None   # most favourable price seen since fill
    trail_stop_level: float | None = None   # current trailing stop price
    expires_at:       datetime = field(init=False)

    def __post_init__(self):
        self.expires_at = self.fired_at + timedelta(minutes=MAX_HOLD_MIN)

    def target_price(self) -> float:
        p = self.fill_price or self.sig.entry
        return p + self.sig.direction * self.sig.target_pts()

    def stop_price(self) -> float:
        p = self.fill_price or self.sig.entry
        return p - self.sig.direction * self.sig.stop_pts()


@dataclass
class ActiveOrbTrade:
    instrument:  BotInstrument
    contract_id: str
    sig:         OrbSignal
    fired_at:    datetime
    order_id:    int | None = None
    fill_price:  float | None = None
    expires_at:  datetime = field(init=False)

    def __post_init__(self):
        self.expires_at = self.fired_at + timedelta(minutes=MAX_HOLD_MIN)

    def target_price(self) -> float:
        p = self.fill_price or self.sig.entry
        return p + self.sig.direction * self.sig.target_pts()

    def stop_price(self) -> float:
        p = self.fill_price or self.sig.entry
        return p - self.sig.direction * self.sig.stop_pts()


@dataclass
class InstrumentState:
    instrument:   BotInstrument
    contract_id:  str = ""
    bars:         list[Bar] = field(default_factory=list)
    vwaslr_bars:  list[Bar] = field(default_factory=list)  # separate 1-min bar list for VWASLR
    sigma:        float = 0.0
    sigma_pts:    float = 0.0
    mean_vol:           float | None = None
    gk_ann_vol:         float = 0.0
    csr:                float = 0.0
    active_trade:         ActiveTrade | None = None
    active_orb_trade:     ActiveOrbTrade | None = None
    active_vwaslr_trade:  ActiveVwasrlTrade | None = None
    orb:                  OrbState = field(default_factory=OrbState)
    last_evaluated_ts:    datetime | None = None
    vwaslr_last_ts:       datetime | None = None
    vwaslr_fetch_min:     int = -1               # UTC minute of last vwaslr fetch (throttle)
    vwaslr_ema:           float = 0.0            # EMA-10 of raw VWASLR (updated every poll)
    vwaslr_ema_prev:      float = 0.0            # EMA value before last update (cross detection)


# ── Trade logging ────────────────────────────────────────────────────────────

def _ensure_log():
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def _log_trade(trade: ActiveTrade, outcome: str, exit_price: float, now: datetime):
    fill = trade.fill_price or trade.est_entry
    pnl_pts = (exit_price - fill) * trade.direction
    row = {
        "fired_at":    trade.fired_at.isoformat(),
        "resolved_at": now.isoformat(),
        "symbol":      trade.instrument.symbol,
        "direction":   "LONG" if trade.direction == 1 else "SHORT",
        "est_entry":   round(trade.est_entry, 4),
        "fill_price":  round(fill, 4),
        "target":      round(trade.target_price(), 4),
        "stop":        round(trade.stop_price(), 4),
        "sigma_pts":   round(trade.sigma_pts, 4),
        "scaled":      round(trade.scaled, 4),
        "vol_ratio":   round(trade.vol_ratio, 4),
        "csr":         round(trade.csr, 4),
        "pl_aligned":  round(trade.pl_aligned, 4) if trade.pl_aligned is not None else "",
        "contracts":   trade.contracts,
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
        "pnl_sigma":   round(pnl_pts / trade.sigma_pts, 4) if trade.sigma_pts else 0.0,
    }
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
    log.info(
        f"TRADE LOGGED  {trade.instrument.symbol} {row['direction']}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  "
        f"pnl={pnl_pts:+.2f}pts ({pnl_pts / trade.sigma_pts:+.3f}σ)"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _in_blackout(bar_hm: tuple[int, int],
                 sh: int, sm: int, eh: int, em: int) -> bool:
    """Return True if bar_hm falls inside the [start, end) window.
    Handles overnight windows where start > end (e.g. 18:00–09:00)."""
    s = sh * 60 + sm
    e = eh * 60 + em
    b = bar_hm[0] * 60 + bar_hm[1]
    return (s <= b < e) if s < e else (b >= s or b < e)


GK_VOL_BARS   = 20
BARS_PER_YEAR = 252 * 23 * 60

def _gk_annualised_vol(bars: list) -> float:
    sample = bars[-GK_VOL_BARS:] if len(bars) >= GK_VOL_BARS else bars
    if len(sample) < 2:
        return 0.0
    vals = []
    for b in sample:
        if b.open <= 0 or b.high <= 0 or b.low <= 0 or b.close <= 0:
            continue
        hl = math.log(b.high / b.low) ** 2
        co = math.log(b.close / b.open) ** 2
        vals.append(0.5 * hl - (2 * math.log(2) - 1) * co)
    if not vals:
        return 0.0
    return math.sqrt(max(0.0, float(np.mean(vals))) * BARS_PER_YEAR / TF_MINUTES)


def get_mom_bars(gk_ann_vol: float, csr_vol_windows: list) -> int:
    """Return CSR window (bars) for current GK vol regime."""
    for upper, bars in csr_vol_windows:
        if gk_ann_vol < upper:
            return bars
    return csr_vol_windows[-1][1]


# ── PL confidence sizing ─────────────────────────────────────────────────────

def fetch_1min_pl(client: TopstepClient, contract_id: str,
                  signal_bar_ts: datetime, direction: int) -> float | None:
    """
    Fetch PL_N_BARS 1-min bars ending just before the signal 5-min bar and
    return PL_aligned = (signed path length) × direction.
    +1 = 1-min flow perfectly aligned; ≥ PL_THRESH → 2× sizing.
    Returns None on fetch error or insufficient data.
    """
    end   = signal_bar_ts
    start = end - timedelta(minutes=PL_N_BARS + 5)
    try:
        raw = client.get_bars(
            contract_id=contract_id, start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=PL_N_BARS + 5,
        )
        raw = list(reversed(raw))
    except Exception as e:
        log.debug(f"fetch_1min_pl error for {contract_id}: {e}")
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


# ── Signal evaluation (identical logic to signal_monitor.py) ────────────────

def evaluate(state: InstrumentState) -> dict | None:
    """Return signal dict if all criteria are met, else None."""
    bars = state.bars
    if len(bars) < TRAILING_BARS + 1:
        return None

    closes  = np.array([b.close  for b in bars])
    volumes = np.array([b.volume for b in bars])

    trail     = np.log(closes[-TRAILING_BARS:] / closes[-TRAILING_BARS - 1:-1])
    sigma     = float(np.std(trail, ddof=1))
    if sigma == 0:
        return None

    sigma_pts = sigma * closes[-1]
    prior_vols   = volumes[-TRAILING_BARS - 1:-1]
    active_vols  = prior_vols[prior_vols >= 10]
    mean_vol     = float(np.median(active_vols)) if len(active_vols) >= 10 else None
    last      = bars[-1]
    bar_ret   = math.log(last.close / last.open) if last.open else 0.0
    scaled    = bar_ret / sigma
    vol_ratio = (last.volume / mean_vol) if mean_vol is not None else None
    direction = 1 if scaled > 0 else -1

    # Dynamic CSR window based on current GK vol regime
    state.gk_ann_vol = _gk_annualised_vol(bars)
    mom_bars = get_mom_bars(state.gk_ann_vol, state.instrument.csr_vol_windows)
    if len(closes) >= mom_bars + 1:
        mom_rets = np.log(closes[-mom_bars:] / closes[-mom_bars - 1:-1])
        csr = float(mom_rets.sum()) / sigma * direction
    else:
        csr = 0.0

    state.sigma     = sigma
    state.sigma_pts = sigma_pts
    state.mean_vol  = mean_vol
    state.csr       = csr

    # Per-instrument blackout windows.
    bar_et = last.ts.astimezone(ET)
    bar_hm = (bar_et.hour, bar_et.minute)
    for sh, sm, eh, em, conditional in state.instrument.blackout_windows:
        if _in_blackout(bar_hm, sh, sm, eh, em):
            if not conditional or csr < CSR_THRESHOLD:
                return None

    if (abs(scaled) >= SIGNAL_SIGMA and abs(scaled) <= MAX_SCALED
            and vol_ratio is not None and vol_ratio >= VOL_RATIO_MIN
            and csr >= CSR_THRESHOLD):
        return {
            "direction": direction,
            "entry":     last.close,
            "sigma":     sigma,
            "sigma_pts": sigma_pts,
            "scaled":    scaled,
            "vol_ratio": vol_ratio,
            "csr":       csr,
            "bar_ts":    last.ts,
        }
    return None


# ── Bar fetching ─────────────────────────────────────────────────────────────

def fetch_bars(client: TopstepClient, state: InstrumentState):
    end      = datetime.now(timezone.utc)
    max_mom  = max(bars for inst in INSTRUMENTS for _, bars in inst.csr_vol_windows)
    lookback = TRAILING_BARS + max_mom + 10
    start    = end - timedelta(minutes=TF_MINUTES * lookback)
    raw = client.get_bars(
        contract_id=state.contract_id,
        start=start, end=end,
        unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
        limit=lookback,
    )
    state.bars = [
        Bar(ts=datetime.fromisoformat(b["t"]),
            open=b["o"], high=b["h"], low=b["l"],
            close=b["c"], volume=b["v"])
        for b in reversed(raw)
    ]


def fetch_vwaslr_bars(client: TopstepClient, state: InstrumentState):
    """Fetch 1-min bars for VWASLR.  Initial call: full 565-bar history.
    Subsequent calls: only bars since the last known timestamp (incremental)."""
    if not state.vwaslr_bars:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(minutes=VWASLR_INIT_BARS + 30)
        raw   = client.get_bars(
            contract_id=state.contract_id,
            start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=VWASLR_INIT_BARS,
        )
        state.vwaslr_bars = [
            Bar(ts=datetime.fromisoformat(b["t"]),
                open=b["o"], high=b["h"], low=b["l"],
                close=b["c"], volume=b["v"])
            for b in reversed(raw)
        ]
    else:
        since = state.vwaslr_bars[-1].ts
        end   = datetime.now(timezone.utc)
        raw   = client.get_bars(
            contract_id=state.contract_id,
            start=since, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=10,
        )
        new_bars = [
            Bar(ts=datetime.fromisoformat(b["t"]),
                open=b["o"], high=b["h"], low=b["l"],
                close=b["c"], volume=b["v"])
            for b in reversed(raw)
        ]
        for b in new_bars:
            if b.ts > since:
                state.vwaslr_bars.append(b)
        # Trim to avoid unbounded growth (keep last VWASLR_INIT_BARS + buffer)
        if len(state.vwaslr_bars) > VWASLR_INIT_BARS + 200:
            state.vwaslr_bars = state.vwaslr_bars[-(VWASLR_INIT_BARS + 100):]


# ── Order placement ──────────────────────────────────────────────────────────

def place_signal(client: TopstepClient, state: InstrumentState,
                 sig: dict, account_id: int, paper: bool) -> ActiveTrade:
    inst      = state.instrument
    direction = sig["direction"]
    sigma_pts = sig["sigma_pts"]
    tick      = inst.tick_size

    # API ticks are signed relative to fill price: negative = below, positive = above.
    # Long:  stop below entry (negative), target above entry (positive)
    # Short: stop above entry (positive), target below entry (negative)
    stop_mag   = max(1, round(inst.stop_sigma   * sigma_pts / tick))
    target_mag = max(1, round(inst.target_sigma * sigma_pts / tick))
    stop_ticks   = -stop_mag   * direction
    target_ticks =  target_mag * direction
    side         = TopstepClient.BID if direction == 1 else TopstepClient.ASK
    dir_str      = "LONG" if direction == 1 else "SHORT"

    n_contracts = sig.get("contracts", 1)
    pl_aligned  = sig.get("pl_aligned")

    trade = ActiveTrade(
        instrument=inst, contract_id=state.contract_id,
        direction=direction, est_entry=sig["entry"],
        sigma_pts=sigma_pts, scaled=sig["scaled"],
        vol_ratio=sig["vol_ratio"], csr=sig["csr"],
        fired_at=sig["bar_ts"],
        pl_aligned=pl_aligned, contracts=n_contracts,
    )

    pl_note = f"  pl={pl_aligned:+.2f}" if pl_aligned is not None else ""
    size_note = f"  [⚡ 2× SIZE]" if n_contracts == 2 else ""

    if paper:
        log.info(
            f"[PAPER] {inst.symbol} {dir_str}  entry≈{sig['entry']:.2f}  "
            f"stop={stop_ticks}t ({inst.stop_sigma}σ)  "
            f"target={target_ticks}t ({inst.target_sigma}σ)  "
            f"scaled={sig['scaled']:+.2f}σ  csr={sig['csr']:+.2f}σ"
            f"{pl_note}  contracts={n_contracts}{size_note}"
        )
    else:
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=side, size=n_contracts,
            order_type=TopstepClient.ORDER_MARKET,
            stop_loss_ticks=stop_ticks,
            take_profit_ticks=target_ticks,
            custom_tag=f"bot_{inst.symbol}_{sig['bar_ts'].strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"ORDER PLACED  {inst.symbol} {dir_str}  order_id={trade.order_id}  "
            f"entry≈{sig['entry']:.2f}  stop={stop_ticks}t  target={target_ticks}t  "
            f"scaled={sig['scaled']:+.2f}σ  csr={sig['csr']:+.2f}σ"
            f"{pl_note}  contracts={n_contracts}{size_note}"
        )

    state.active_trade = trade
    return trade


# ── Position monitoring ──────────────────────────────────────────────────────

def handle_active_trade(client: TopstepClient, state: InstrumentState,
                        account_id: int, now: datetime, paper: bool):
    trade = state.active_trade

    if paper:
        # Paper mode: only simulate time exit
        if now >= trade.expires_at:
            exit_price = state.bars[-1].close if state.bars else trade.est_entry
            _log_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_trade = None
        return

    # Fetch current open positions
    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"Could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    # Update fill price once the position appears
    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"{trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    # Software trailing stop for CSR trades.
    # The bracket stop at 2σ remains as a safety net; this fires earlier.
    if pos and trade.fill_price is not None and state.bars:
        fill       = trade.fill_price
        trail_dist = CSR_TRAIL_SIGMA * trade.sigma_pts
        bars_after = [b for b in state.bars if b.ts >= trade.fired_at]
        if bars_after:
            if trade.direction == 1:   # LONG — track highest high
                new_peak = max(b.high for b in bars_after)
                trade.trail_peak = max(new_peak, fill)
                trade.trail_stop_level = trade.trail_peak - trail_dist
            else:                      # SHORT — track lowest low
                new_peak = min(b.low for b in bars_after)
                trade.trail_peak = min(new_peak, fill)
                trade.trail_stop_level = trade.trail_peak + trail_dist

            last_bar   = state.bars[-1]
            trail_stop = trade.trail_stop_level
            trail_hit  = (
                last_bar.ts > trade.fired_at and trail_stop is not None and (
                    (trade.direction ==  1 and last_bar.low  <= trail_stop) or
                    (trade.direction == -1 and last_bar.high >= trail_stop)
                )
            )
            if trail_hit:
                log.info(
                    f"{trade.instrument.symbol} TRAIL STOP  "
                    f"trail_stop={trail_stop:.2f}  peak={trade.trail_peak:.2f}  "
                    f"trail={CSR_TRAIL_SIGMA}σ={trail_dist:.2f}pts"
                )
                try:
                    n = client.cancel_all_orders(account_id)
                    if n:
                        log.info(f"{trade.instrument.symbol}: cancelled {n} bracket(s) before trail close")
                except Exception as e:
                    log.warning(f"{trade.instrument.symbol}: pre-trail cancel failed: {e}")
                try:
                    client.close_position(account_id, trade.contract_id)
                except Exception as e:
                    log.error(f"{trade.instrument.symbol}: trail close_position failed: {e}")
                    return
                _log_trade(trade, "TRAIL STOP", trail_stop, now)
                state.active_trade = None
                try:
                    client.cancel_all_orders(account_id)
                except Exception:
                    pass
                return

    if pos is None:
        # Position is gone — brackets closed it; cancel any residual OCO orders
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now)
        if exit_price is not None:
            # Actual fill from trade history — classify by proximity to brackets
            # Direction-aware: for LONG target > entry, for SHORT target < entry
            d = trade.direction
            outcome = ("TARGET" if (d == 1 and exit_price >= trade.target_price()) or
                                   (d == -1 and exit_price <= trade.target_price())
                       else "STOPPED")
        else:
            # Lookup failed — infer from bar highs/lows (accurate to ~1 tick)
            outcome, exit_price = _classify_outcome_from_bars(trade, state.bars)
        _log_trade(trade, outcome, exit_price, now)
        state.active_trade = None
        try:
            n_cancelled = client.cancel_all_orders(account_id)
            if n_cancelled:
                log.info(f"{trade.instrument.symbol} {outcome}: cancelled {n_cancelled} residual order(s)")
        except Exception as e:
            log.warning(f"{trade.instrument.symbol}: cancel_all_orders failed: {e}")
        return

    # Force time exit if max hold exceeded
    if now >= trade.expires_at:
        log.info(f"{trade.instrument.symbol} max hold reached — cancelling brackets then closing at market")
        # Cancel brackets BEFORE closing: the API treats close_position() as a new
        # market order and can attach fresh brackets to it, or leave orphan OCO legs
        # that later fill and open an unwanted opposing position.
        try:
            n_cancelled = client.cancel_all_orders(account_id)
            if n_cancelled:
                log.info(f"{trade.instrument.symbol}: cancelled {n_cancelled} bracket order(s) before time exit")
        except Exception as e:
            log.warning(f"{trade.instrument.symbol}: pre-close cancel_all failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"Failed to close position for {trade.instrument.symbol}: {e}")
            return
        exit_price = state.bars[-1].close if state.bars else (trade.fill_price or trade.est_entry)
        _log_trade(trade, "TIME EXIT", exit_price, now)
        state.active_trade = None
        # Cancel again in case the closing order itself spawned new brackets
        try:
            n_cancelled = client.cancel_all_orders(account_id)
            if n_cancelled:
                log.info(f"{trade.instrument.symbol} TIME EXIT: cancelled {n_cancelled} residual order(s) after close")
        except Exception as e:
            log.warning(f"{trade.instrument.symbol}: post-close cancel_all failed: {e}")




def _classify_outcome_from_bars(trade: ActiveTrade, bars: list[Bar]) -> tuple[str, float]:
    """
    Infer whether a bracket-closed trade hit target or stop by scanning bar
    data since the signal fired.  Returns (outcome, exit_price) using the
    bracket price as the exit, which is accurate to within one tick for
    exchange-managed OCO orders.
    """
    for bar in bars:
        if bar.ts <= trade.fired_at:
            continue
        if trade.direction == 1:   # long
            if bar.high >= trade.target_price():
                return "TARGET",  trade.target_price()
            if bar.low  <= trade.stop_price():
                return "STOPPED", trade.stop_price()
        else:                      # short
            if bar.low  <= trade.target_price():
                return "TARGET",  trade.target_price()
            if bar.high >= trade.stop_price():
                return "STOPPED", trade.stop_price()
    # No bracket found in available bars — classify by which is closer to last close
    last_close = bars[-1].close if bars else (trade.fill_price or trade.est_entry)
    if abs(last_close - trade.target_price()) <= abs(last_close - trade.stop_price()):
        return "TARGET",  trade.target_price()
    return "STOPPED", trade.stop_price()


# ── ORB log ───────────────────────────────────────────────────────────────────

def _ensure_orb_log():
    ORB_LOG_PATH.parent.mkdir(exist_ok=True)
    if not ORB_LOG_PATH.exists():
        with open(ORB_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writeheader()


def _log_orb_trade(trade: ActiveOrbTrade, outcome: str, exit_price: float, now: datetime):
    fill = trade.fill_price or trade.sig.entry
    pnl_pts = (exit_price - fill) * trade.sig.direction
    risk     = trade.sig.risk_pts()
    row = {
        "fired_at":    trade.fired_at.isoformat(),
        "resolved_at": now.isoformat(),
        "symbol":      trade.instrument.symbol,
        "direction":   "LONG" if trade.sig.direction == 1 else "SHORT",
        "est_entry":   round(trade.sig.entry, 4),
        "fill_price":  round(fill, 4),
        "target":      round(trade.target_price(), 4),
        "stop":        round(trade.stop_price(), 4),
        "orb_high":    round(trade.sig.orb_high, 4),
        "orb_low":     round(trade.sig.orb_low, 4),
        "orb_width":   round(trade.sig.orb_high - trade.sig.orb_low, 4),
        "sigma_pts":   round(trade.sig.sigma_pts, 4),
        "window":      trade.sig.window,
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
        "pnl_r":       round(pnl_pts / risk, 4) if risk else 0.0,
    }
    with open(ORB_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writerow(row)
    log.info(
        f"ORB LOGGED  {trade.instrument.symbol} LONG  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  "
        f"pnl={pnl_pts:+.2f}pts ({pnl_pts/risk:+.3f}R)" if risk else
        f"ORB LOGGED  {trade.instrument.symbol} LONG  {outcome}"
    )


# ── ORB evaluation ────────────────────────────────────────────────────────────

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
    if window == "Morning" and orb.morning_fired:
        return None

    orb_width     = orb.orb_high - orb.orb_low
    orb_mid       = (orb.orb_high + orb.orb_low) / 2.0
    orb_width_pct = orb_width / orb_mid if orb_mid > 0 else 0.0
    if orb_width_pct < state.instrument.orb_width_pct_min:
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
        orb.morning_fired = True
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
        orb.morning_fired = True
        return sig

    return None


# ── ORB order placement ───────────────────────────────────────────────────────

def place_orb_signal(client: TopstepClient, state: InstrumentState,
                     sig: OrbSignal, account_id: int, paper: bool) -> ActiveOrbTrade:
    inst      = state.instrument
    tick      = inst.tick_size
    is_long   = sig.direction == 1
    dir_label = "LONG" if is_long else "SHORT"
    stop_mag   = max(1, round(sig.stop_pts()   / tick))
    target_mag = max(1, round(sig.target_pts() / tick))
    # For LONG:  stop below entry (negative ticks), target above (positive ticks)
    # For SHORT: stop above entry (positive ticks), target below (negative ticks)
    stop_ticks   = -stop_mag   if is_long else  stop_mag
    target_ticks =  target_mag if is_long else -target_mag

    trade = ActiveOrbTrade(
        instrument=inst, contract_id=state.contract_id,
        sig=sig, fired_at=sig.bar_ts,
    )

    if paper:
        log.info(
            f"[PAPER] ORB {inst.symbol} {dir_label}  entry≈{sig.entry:.2f}  "
            f"target={sig.target:.2f} ({sig.target_pts():.2f}pts)  "
            f"stop={sig.stop:.2f} ({sig.stop_pts():.2f}pts)  "
            f"window={sig.window}  orb={sig.orb_low:.2f}–{sig.orb_high:.2f}"
        )
    else:
        order_side = TopstepClient.BID if is_long else TopstepClient.ASK
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=order_side,
            size=1,
            order_type=TopstepClient.ORDER_MARKET,
            stop_loss_ticks=stop_ticks,
            take_profit_ticks=target_ticks,
            custom_tag=f"orb_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"ORB ORDER  {inst.symbol} {dir_label}  order_id={trade.order_id}  "
            f"entry≈{sig.entry:.2f}  stop={stop_ticks}t  target={target_ticks}t  "
            f"window={sig.window}"
        )

    state.active_orb_trade = trade
    return trade


# ── ORB position monitoring ───────────────────────────────────────────────────

def handle_active_orb_trade(client: TopstepClient, state: InstrumentState,
                             account_id: int, now: datetime, paper: bool):
    trade = state.active_orb_trade

    if paper:
        if now >= trade.expires_at:
            exit_price = state.bars[-1].close if state.bars else trade.sig.entry
            _log_orb_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_orb_trade = None
        return

    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"ORB {trade.instrument.symbol}: could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"ORB {trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    if pos is None:
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now)
        if exit_price is not None:
            d = trade.sig.direction
            outcome = ("TARGET" if (d == 1 and exit_price >= trade.target_price()) or
                                   (d == -1 and exit_price <= trade.target_price())
                       else "STOPPED")
        else:
            outcome, exit_price = _classify_orb_outcome(trade, state.bars)
        _log_orb_trade(trade, outcome, exit_price, now)
        state.active_orb_trade = None
        try:
            n = client.cancel_all_orders(account_id)
            if n:
                log.info(f"ORB {trade.instrument.symbol} {outcome}: cancelled {n} residual order(s)")
        except Exception as e:
            log.warning(f"ORB {trade.instrument.symbol}: cancel_all_orders failed: {e}")
        return

    if now >= trade.expires_at:
        log.info(f"ORB {trade.instrument.symbol} max hold reached — closing")
        try:
            client.cancel_all_orders(account_id)
        except Exception as e:
            log.warning(f"ORB {trade.instrument.symbol}: pre-close cancel_all failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"ORB {trade.instrument.symbol}: failed to close position: {e}")
            return
        exit_price = state.bars[-1].close if state.bars else (trade.fill_price or trade.sig.entry)
        _log_orb_trade(trade, "TIME EXIT", exit_price, now)
        state.active_orb_trade = None
        try:
            client.cancel_all_orders(account_id)
        except Exception:
            pass


def _get_exit_price(client: TopstepClient, account_id: int,
                    fired_at: datetime, contract_id: str, now: datetime) -> float | None:
    try:
        trades = client.search_trades(account_id, fired_at, now)
        closing = [
            t for t in trades
            if t.get("contractId") == contract_id
            and datetime.fromisoformat(t.get("timestamp", "1970-01-01T00:00:00")).replace(tzinfo=timezone.utc)
                > fired_at
        ]
        if closing:
            price = float(closing[-1].get("price", 0))
            if price > 0:
                return price
    except Exception as e:
        log.warning(f"Could not fetch trade history for exit price: {e}")
    return None


def _classify_orb_outcome(trade: ActiveOrbTrade, bars: list[Bar]) -> tuple[str, float]:
    # Conservative (adverse-first): check stop before target within each bar.
    for bar in bars:
        if bar.ts <= trade.fired_at:
            continue
        if trade.sig.direction == 1:   # LONG
            if bar.low  <= trade.stop_price():
                return "STOPPED", trade.stop_price()
            if bar.high >= trade.target_price():
                return "TARGET",  trade.target_price()
        else:                          # SHORT
            if bar.high >= trade.stop_price():
                return "STOPPED", trade.stop_price()
            if bar.low  <= trade.target_price():
                return "TARGET",  trade.target_price()
    last_close = bars[-1].close if bars else (trade.fill_price or trade.sig.entry)
    if abs(last_close - trade.target_price()) <= abs(last_close - trade.stop_price()):
        return "TARGET",  trade.target_price()
    return "STOPPED", trade.stop_price()


# ── VWASLR signal ────────────────────────────────────────────────────────────

def _ensure_vwaslr_log():
    VWAS_LOG_PATH.parent.mkdir(exist_ok=True)
    if not VWAS_LOG_PATH.exists():
        with open(VWAS_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=VWAS_LOG_FIELDS).writeheader()


def _log_vwaslr_trade(trade: ActiveVwasrlTrade, outcome: str,
                      exit_price: float, now: datetime):
    fill    = trade.fill_price or trade.sig.entry
    pnl_pts = (exit_price - fill) * trade.sig.direction
    row = {
        "fired_at":    trade.fired_at.isoformat(),
        "resolved_at": now.isoformat(),
        "symbol":      trade.instrument.symbol,
        "direction":   "LONG" if trade.sig.direction == 1 else "SHORT",
        "est_entry":   round(trade.sig.entry, 4),
        "fill_price":  round(fill, 4),
        "target":      round(trade.target_price(), 4),
        "stop":        round(trade.stop_price(), 4),
        "sigma_pts":   round(trade.sig.sigma_pts, 4),
        "vwaslr":      round(trade.sig.vwaslr, 4),
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
        "pnl_sigma":   round(pnl_pts / trade.sig.sigma_pts, 4) if trade.sig.sigma_pts else 0.0,
    }
    with open(VWAS_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=VWAS_LOG_FIELDS).writerow(row)
    dir_s = "LONG" if trade.sig.direction == 1 else "SHORT"
    log.info(
        f"VWASLR LOGGED  {trade.instrument.symbol} {dir_s}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  "
        f"pnl={pnl_pts:+.2f}pts ({pnl_pts / trade.sig.sigma_pts:+.3f}σ)"
        if trade.sig.sigma_pts else
        f"VWASLR LOGGED  {trade.instrument.symbol} {dir_s}  {outcome}"
    )


def _update_vwaslr_ema(state: InstrumentState):
    """
    Compute raw VWASLR from the current 1-min bars and advance the EMA-10.
    Called every poll cycle regardless of position state so the EMA stays
    current for both entry cross-detection and half-zero exit checks.
    Updates state.vwaslr_ema_prev and state.vwaslr_ema in-place.

    EMA advances on every bar (Globex included, settlement gap excluded by the
    API).  Backtesting confirmed that using all-bars EMA with an 8:30 ET entry
    window doubles Sharpe vs RTH-only EMA.  Entry is still gated by
    inst.vwaslr_start so no trade fires during the unvalidated overnight window.
    """
    inst = state.instrument
    bars = state.vwaslr_bars
    needed = inst.vwaslr_n + VWASLR_SIGMA_BARS + 1
    if len(bars) < needed:
        return

    closes  = np.array([b.close  for b in bars], dtype=float)
    volumes = np.array([b.volume for b in bars], dtype=float)
    i = len(bars) - 1

    trail = np.log(closes[i - VWASLR_SIGMA_BARS + 1: i + 1]
                 / closes[i - VWASLR_SIGMA_BARS:     i    ])
    sigma = float(np.std(trail, ddof=1))
    if sigma == 0:
        return

    ret_win = np.log(closes[i - inst.vwaslr_n + 1: i + 1]
                   / closes[i - inst.vwaslr_n:     i    ])
    vol_win = volumes[i - inst.vwaslr_n: i]
    sum_vol = float(vol_win.sum())
    if sum_vol == 0:
        return

    raw = float((ret_win / sigma * vol_win).sum() / sum_vol)
    alpha = 2.0 / (VWASLR_EMA_SPAN + 1)
    state.vwaslr_ema_prev = state.vwaslr_ema
    state.vwaslr_ema = alpha * raw + (1.0 - alpha) * state.vwaslr_ema


def evaluate_vwaslr(state: InstrumentState) -> VwasrlSignal | None:
    """
    Return a VwasrlSignal if the EMA-10 of VWASLR just crossed ±threshold on
    the current bar, and the bar is within inst.vwaslr_start–16:00 ET.
    EMA must be updated by _update_vwaslr_ema() before calling this.
    """
    inst     = state.instrument
    thr      = inst.vwaslr_threshold
    ema      = state.vwaslr_ema
    ema_prev = state.vwaslr_ema_prev

    # Fire only on the bar where EMA first crosses the threshold
    crossed_up   = ema_prev <= thr  and ema > thr
    crossed_down = ema_prev >= -thr and ema < -thr
    if not crossed_up and not crossed_down:
        return None

    bars   = state.vwaslr_bars
    needed = inst.vwaslr_n + VWASLR_SIGMA_BARS + 1
    if len(bars) < needed:
        return None

    # σ for order sizing
    closes = np.array([b.close for b in bars], dtype=float)
    i = len(bars) - 1
    trail = np.log(closes[i - VWASLR_SIGMA_BARS + 1: i + 1]
                 / closes[i - VWASLR_SIGMA_BARS:     i    ])
    sigma = float(np.std(trail, ddof=1))
    if sigma == 0:
        return None
    sigma_pts = sigma * closes[i]

    # RTH filter: inst.vwaslr_start–16:00 ET only
    last   = bars[-1]
    bar_et = last.ts.astimezone(ET)
    bar_hm = (bar_et.hour, bar_et.minute)
    if bar_hm < inst.vwaslr_start or bar_hm >= (16, 0):
        return None

    # Respect shared blackout windows.
    # Pre-9:30 (pre-market): skip conditional blackouts — no CSR context from 5-min bars yet.
    pre_rth = bar_hm < (9, 30)
    for sh, sm, eh, em, conditional in inst.blackout_windows:
        if _in_blackout(bar_hm, sh, sm, eh, em):
            if conditional and pre_rth:
                continue  # conditional blackout irrelevant pre-market; VWASLR edge confirmed
            if not conditional or state.csr < CSR_THRESHOLD:
                return None

    direction = 1 if ema > 0 else -1
    entry     = last.close
    target    = entry + direction * VWASLR_TARGET_SIGMA * sigma_pts
    stop      = entry - direction * VWASLR_STOP_SIGMA   * sigma_pts
    return VwasrlSignal(entry=entry, target=target, stop=stop,
                        sigma_pts=sigma_pts, vwaslr=ema,
                        bar_ts=last.ts, direction=direction)


def place_vwaslr_signal(client: TopstepClient, state: InstrumentState,
                        sig: VwasrlSignal, account_id: int,
                        paper: bool) -> ActiveVwasrlTrade:
    inst      = state.instrument
    tick      = inst.tick_size
    is_long   = sig.direction == 1
    dir_label = "LONG" if is_long else "SHORT"
    stop_mag   = max(1, round(sig.stop_pts()   / tick))
    target_mag = max(1, round(sig.target_pts() / tick))
    stop_ticks   = -stop_mag   if is_long else  stop_mag
    target_ticks =  target_mag if is_long else -target_mag

    trade = ActiveVwasrlTrade(
        instrument=inst, contract_id=state.contract_id,
        sig=sig, fired_at=sig.bar_ts,
    )

    if paper:
        log.info(
            f"[PAPER] VWASLR {inst.symbol} {dir_label}  entry≈{sig.entry:.2f}  "
            f"target={sig.target:.2f} ({sig.target_pts():.2f}pts)  "
            f"stop={sig.stop:.2f} ({sig.stop_pts():.2f}pts)  "
            f"vwaslr={sig.vwaslr:+.3f}σ/bar"
        )
    else:
        order_side = TopstepClient.BID if is_long else TopstepClient.ASK
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=order_side,
            size=1,
            order_type=TopstepClient.ORDER_MARKET,
            stop_loss_ticks=stop_ticks,
            take_profit_ticks=target_ticks,
            custom_tag=f"vwas_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"VWASLR ORDER  {inst.symbol} {dir_label}  order_id={trade.order_id}  "
            f"entry≈{sig.entry:.2f}  stop={stop_ticks}t  target={target_ticks}t  "
            f"vwaslr={sig.vwaslr:+.3f}σ/bar"
        )

    state.active_vwaslr_trade = trade
    return trade


def handle_active_vwaslr_trade(client: TopstepClient, state: InstrumentState,
                                account_id: int, now: datetime, paper: bool):
    trade = state.active_vwaslr_trade

    if paper:
        if now >= trade.expires_at:
            exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                          else trade.sig.entry)
            _log_vwaslr_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_vwaslr_trade = None
        return

    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"VWASLR {trade.instrument.symbol}: could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"VWASLR {trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    # Half-zero signal exit: close when EMA-VWASLR retracts below ±(threshold/2).
    # The bracket stop at 2σ remains on the API as a hard safety net.
    half_thr = trade.instrument.vwaslr_threshold / 2
    ema = state.vwaslr_ema
    signal_exit = pos and (
        (trade.sig.direction ==  1 and ema < half_thr) or
        (trade.sig.direction == -1 and ema > -half_thr)
    )
    if signal_exit:
        log.info(
            f"VWASLR {trade.instrument.symbol} SIGNAL EXIT  "
            f"ema={ema:+.3f}  half_thr=±{half_thr:.2f}"
        )
        try:
            n = client.cancel_all_orders(account_id)
            if n:
                log.info(f"VWASLR {trade.instrument.symbol}: cancelled {n} bracket(s) before signal exit")
        except Exception as e:
            log.warning(f"VWASLR {trade.instrument.symbol}: pre-signal-exit cancel failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"VWASLR {trade.instrument.symbol}: signal exit close_position failed: {e}")
            return
        exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                      else (trade.fill_price or trade.sig.entry))
        _log_vwaslr_trade(trade, "SIGNAL EXIT", exit_price, now)
        state.active_vwaslr_trade = None
        try:
            client.cancel_all_orders(account_id)
        except Exception:
            pass
        return

    if pos is None:
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now)
        if exit_price is not None:
            d = trade.sig.direction
            outcome = ("TARGET" if (d == 1 and exit_price >= trade.target_price()) or
                                   (d == -1 and exit_price <= trade.target_price())
                       else "STOPPED")
        else:
            outcome, exit_price = _classify_vwaslr_outcome(trade, state.vwaslr_bars)
        _log_vwaslr_trade(trade, outcome, exit_price, now)
        state.active_vwaslr_trade = None
        try:
            n = client.cancel_all_orders(account_id)
            if n:
                log.info(f"VWASLR {trade.instrument.symbol} {outcome}: cancelled {n} residual order(s)")
        except Exception as e:
            log.warning(f"VWASLR {trade.instrument.symbol}: cancel_all_orders failed: {e}")
        return

    if now >= trade.expires_at:
        log.info(f"VWASLR {trade.instrument.symbol} max hold reached — closing")
        try:
            client.cancel_all_orders(account_id)
        except Exception as e:
            log.warning(f"VWASLR {trade.instrument.symbol}: pre-close cancel_all failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"VWASLR {trade.instrument.symbol}: failed to close position: {e}")
            return
        exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                      else (trade.fill_price or trade.sig.entry))
        _log_vwaslr_trade(trade, "TIME EXIT", exit_price, now)
        state.active_vwaslr_trade = None
        try:
            client.cancel_all_orders(account_id)
        except Exception:
            pass


def _classify_vwaslr_outcome(trade: ActiveVwasrlTrade,
                              bars: list[Bar]) -> tuple[str, float]:
    # Conservative (adverse-first) OHLC ordering: check stop before target within
    # each bar so that if a single bar touches both levels we record the loss.
    for bar in bars:
        if bar.ts <= trade.fired_at:
            continue
        if trade.sig.direction == 1:   # LONG: low → stop, high → target
            if bar.low  <= trade.stop_price():
                return "STOPPED", trade.stop_price()
            if bar.high >= trade.target_price():
                return "TARGET",  trade.target_price()
        else:                          # SHORT: high → stop, low → target
            if bar.high >= trade.stop_price():
                return "STOPPED", trade.stop_price()
            if bar.low  <= trade.target_price():
                return "TARGET",  trade.target_price()
    last_close = bars[-1].close if bars else (trade.fill_price or trade.sig.entry)
    if abs(last_close - trade.target_price()) <= abs(last_close - trade.stop_price()):
        return "TARGET",  trade.target_price()
    return "STOPPED", trade.stop_price()


# ── Main loop ────────────────────────────────────────────────────────────────

def run(account_id: int | None, paper: bool):
    client = TopstepClient()
    client.login()

    # Resolve and confirm account
    accounts = client.get_accounts()
    if not accounts:
        raise RuntimeError("No active accounts found.")

    # Default to the designated practice account
    target_id = account_id if account_id is not None else PRACTICE_ACCOUNT_ID
    acct = next((a for a in accounts if a["id"] == target_id), None)
    if acct is None:
        raise RuntimeError(f"Account {target_id} not found in your active accounts.")
    account_id = acct["id"]

    acct_name    = acct.get("name", "—")
    acct_balance = acct.get("balance", "unknown")

    # Safety guard: refuse live trading on any non-practice account
    if not paper and acct_name != PRACTICE_ACCOUNT_NAME:
        raise RuntimeError(
            f"LIVE TRADING BLOCKED — account '{acct_name}' (id={account_id}) is not the "
            f"designated practice account '{PRACTICE_ACCOUNT_NAME}'. "
            f"Update PRACTICE_ACCOUNT_NAME in trading_bot.py to authorise a different account."
        )

    log.info(f"Account: {acct_name}  id={account_id}  balance=${acct_balance:,.2f}"
             + ("  [PAPER]" if paper else "  [LIVE]"))

    # Initialise instrument states
    states: list[InstrumentState] = []
    for inst in INSTRUMENTS:
        contracts = client.search_contracts(inst.search_term)
        if not contracts:
            log.error(f"No contract found for {inst.symbol}")
            continue
        c = contracts[0]
        log.info(f"  {inst.symbol}: {c['name']}  id={c['id']}")
        states.append(InstrumentState(instrument=inst, contract_id=c["id"]))

    if not states:
        raise RuntimeError("No instruments initialised.")

    # Warn if positions are already open on startup
    try:
        existing = client.get_open_positions(account_id)
        open_cids = {p["contractId"] for p in existing}
        for state in states:
            if state.contract_id in open_cids:
                log.warning(
                    f"{state.instrument.symbol}: open position exists on startup — "
                    f"bot will not enter a new one until it closes"
                )
    except Exception as e:
        log.warning(f"Could not check existing positions on startup: {e}")

    _ensure_log()
    _ensure_orb_log()
    _ensure_vwaslr_log()
    mode = "PAPER MODE" if paper else "LIVE"
    log.info(
        f"Bot running — {mode}  account={account_id}  "
        f"instruments={[s.instrument.symbol for s in states]}  "
        f"poll={POLL_SECONDS}s"
    )

    while True:
        now = datetime.now(timezone.utc)

        for state in states:
            try:
                fetch_bars(client, state)
                if state.instrument.vwaslr_n > 0:
                    cur_min = now.minute
                    if cur_min != state.vwaslr_fetch_min:
                        fetch_vwaslr_bars(client, state)
                        state.vwaslr_fetch_min = cur_min
                    _update_vwaslr_ema(state)

                if state.active_trade:
                    handle_active_trade(client, state, account_id, now, paper)

                if state.active_orb_trade:
                    handle_active_orb_trade(client, state, account_id, now, paper)

                if state.active_vwaslr_trade:
                    handle_active_vwaslr_trade(client, state, account_id, now, paper)

                # Only enter new trades when no position is open on this instrument
                no_position = (not state.active_trade
                               and not state.active_orb_trade
                               and not state.active_vwaslr_trade)
                last_bar_ts = state.bars[-1].ts if state.bars else None

                # Don't enter new trades after TopstepX daily cutoff
                now_ct = now.astimezone(CT)
                past_cutoff = (now_ct.hour, now_ct.minute) >= TRADING_CUTOFF_CT

                if no_position and not past_cutoff:
                    sig = evaluate(state)
                    if sig and last_bar_ts != state.last_evaluated_ts:
                        state.last_evaluated_ts = last_bar_ts
                        pl = fetch_1min_pl(client, state.contract_id,
                                           sig["bar_ts"], sig["direction"])
                        sig["pl_aligned"] = pl
                        sig["contracts"]  = 2 if (pl is not None and pl >= PL_THRESH) else 1
                        place_signal(client, state, sig, account_id, paper)
                    elif last_bar_ts != state.last_evaluated_ts:
                        state.last_evaluated_ts = last_bar_ts

                if no_position and not past_cutoff and state.instrument.orb_enabled:
                    orb_sig = evaluate_orb(state)
                    if orb_sig:
                        place_orb_signal(client, state, orb_sig, account_id, paper)

                if no_position and not past_cutoff and state.instrument.vwaslr_n > 0:
                    vwas_sig = evaluate_vwaslr(state)
                    vwaslr_bar_ts = state.vwaslr_bars[-1].ts if state.vwaslr_bars else None
                    if vwas_sig and vwaslr_bar_ts != state.vwaslr_last_ts:
                        state.vwaslr_last_ts = vwaslr_bar_ts
                        place_vwaslr_signal(client, state, vwas_sig, account_id, paper)

            except Exception as e:
                log.error(f"{state.instrument.symbol}: {e}", exc_info=True)

        # Poll faster during the ORB window so breakouts aren't missed
        now_et = datetime.now(ET)
        hm = (now_et.hour, now_et.minute)
        in_orb_window = (9, 30) <= hm < (10, 30)
        time.sleep(POLL_SECONDS_ORB if in_orb_window else POLL_SECONDS)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/bot.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="MES/MYM 3σ continuation trading bot")
    parser.add_argument("--paper", action="store_true",
                        help="Detect signals and log them but place no real orders")
    parser.add_argument("--account", type=int, default=None,
                        help="TopstepX account ID (auto-detects first active account if omitted)")
    args = parser.parse_args()

    run(account_id=args.account, paper=args.paper)
