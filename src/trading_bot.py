"""
Automated trading bot for the 3σ MES/MNQ continuation signal.

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

  ORB (opening range breakout, MES/MNQ):
  - 15-min opening range (9:30–9:45 ET); breakout in morning window (9:45–10:30 ET)
  - ORB width ≥ instrument-specific wide-range cutoff (from backtest)

  Evening Resumption (MES only):
  - At 18:00 ET Mon–Fri, if |gap from last RTH close → first bar open| ≥ 0.2%,
    enter in gap direction; pure 30-min time exit, no bracket orders

  Sunday Open Gap (MES only):
  - At 18:00 ET Sunday (weekly Globex open), if |gap from Friday close → first
    5-min bar open| ≥ 0.3% AND first-bar volume ≥ 1.5× median of prior 8 Sundays,
    enter in gap direction; pure 30-min time exit, no bracket orders
  - fri_close and Sunday vol history persisted to logs/sun_gap_state_MES.json

  VWASLR (volume-weighted avg scaled log return, MES/MNQ):
  - 1-min bars; EMA-10(VWASLR(50min, σ=500min)) crosses ±0.4σ
  - MES/MNQ: active 8:30–16:00 ET (pre-open edge confirmed)
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
import json
import logging
import math
import os
import random
import ssl
import subprocess
import time
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import websocket

import numpy as np
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from topstep_client import TopstepClient, get_bars_from_db, get_5s_bars_from_db, bars_db_available

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
POLL_SECONDS_RTH = 5    # fast poll during RTH for PL_MOM entry detection (matches 5s bar cadence)

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

# Trading cutoff — configurable per broker via TRADING_CUTOFF_MST in .env.
# Defaults to 13:10 MST (TopstepX).  Lucid=13:45, Tradeify=13:59.
# MST (UTC-7) is 2h behind CDT (UTC-5), so CT = MST + 2h.
_cutoff_mst_str = os.environ.get("TRADING_CUTOFF_MST", "13:10")
_cm, _ch = int(_cutoff_mst_str.split(":")[1]), int(_cutoff_mst_str.split(":")[0])
_ct_total = _ch * 60 + _cm + 120   # MST → CDT (+2h)
TRADING_CUTOFF_CT = (_ct_total // 60, _ct_total % 60)

# ── SLR_Scalp parameters (keep in sync with signal_monitor.py) ──────────────
# Volume surge + shallow pullback, LONG only, MES enabled.
SLR_VOL_LOOKBACK = 20    # rolling median window (1-min bars)
SLR_VOL_MULT     = 7.0   # minimum volume surge multiplier
SLR_MOVE_BPS     = 12.0  # minimum surge move in basis points (bullish)
SLR_TARGET_BPS   = 15.0  # profit target (bps of entry price)
SLR_STOP_BPS     = 10.0   # stop distance in basis points (scales with price)
SLR_HOLD_RTH     = 15    # max hold (minutes) during RTH
SLR_HOLD_GLOBEX  = 10    # max hold (minutes) during Globex

# ── PL_Mom parameters (keep in sync with signal_monitor.py) ─────────────────
# Price Linearity Momentum on 5-second bars. RTH only.
PL_MOM_WINDOW          = 6       # 5s bars = 30s
PL_MOM_ENTRY_PL        = 0.70
PL_MOM_MOVE_BPS        = 12.0   # fixed floor (used when sigma not available)
PL_MOM_EXIT_PL         = 0.40
PL_MOM_STOP_BPS        = 10.0
PL_MOM_MIN_HOLD_S      = 10     # seconds before PL exit is checked (stop always active)
PL_MOM_MAX_HOLD_S      = 120
PL_MOM_5S_FETCH        = 130    # 5s bars to fetch: covers sigma lookback (120) + window (6) + margin
PL_MOM_SIGMA_N         = 2.0    # move threshold = max(move_bps_floor, sigma_n × σ_30s_bps)
PL_MOM_SIGMA_LOOKBACK  = 120    # 5s bars for rolling σ (10 min); backtest-optimal

# ── Evening Resumption parameters ────────────────────────────────────────────
# At 18:00 ET daily (CME resumes after 16:00–18:00 ET settlement gap):
# if |gap from prev RTH close → first bar open| ≥ EVE_GAP_THRESH, enter in
# gap direction and hold for EVE_HOLD_MINUTES (time exit only, no brackets).
EVE_GAP_THRESH   = 0.002   # 0.2% minimum gap fraction to fire
EVE_HOLD_MINUTES = 30      # time-only exit (backtest: no bracket improves EV)

# ── Sunday Open Gap parameters ────────────────────────────────────────────────
# Weekly Globex open at 18:00 ET Sunday. Signal fires after first 5-min bar
# closes (≈18:05 ET). Direction follows gap. Pure 30-min time exit.
SUN_GAP_THRESH   = 0.003   # 0.3% minimum gap fraction to fire
SUN_VOL_LOOKBACK = 8       # prior Sunday opens for median vol baseline
SUN_VOL_MULT     = 1.5     # first-bar vol must be ≥ this × median of priors
SUN_HOLD_MINUTES = 30      # time-only exit (no brackets; backtest best result)

LOG_PATH      = Path("logs/bot_trades.csv")
ORB_LOG_PATH  = Path("logs/orb_trades.csv")
VWAS_LOG_PATH = Path("logs/vwaslr_trades.csv")
SLR_LOG_PATH  = Path("logs/slr_trades.csv")
EVE_LOG_PATH  = Path("logs/eve_trades.csv")
SUN_LOG_PATH  = Path("logs/sun_gap_trades.csv")
PL_MOM_LOG_PATH = Path("logs/pl_mom_trades.csv")
SLR_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "target", "stop",
    "vol_ratio", "move_bps", "session",
    "outcome", "pnl_pts",
]
PL_MOM_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "stop", "pl", "move_bps",
    "outcome", "pnl_pts",
]
EVE_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "prev_close", "gap_pct",
    "outcome", "pnl_pts",
]
SUN_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "fri_close", "gap_pct",
    "vol_ratio", "outcome", "pnl_pts",
]
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

# ── DOM WebSocket constants ───────────────────────────────────────────────────

MARKET_HUB_URL  = "wss://rtc.topstepx.com/hubs/market"
DOM_RS          = "\x1e"   # SignalR record separator
DOM_N_LEVELS    = 10       # levels to capture each side at signal fire time
DOM_SIGNAL_LOG  = Path("logs/dom_at_signal.csv")
DOM_SIGNAL_FIELDS = [
    "ts", "strategy", "symbol", "direction",
    "entry", "best_bid", "best_ask", "spread",
    "imb_l1", "imb_l5", "imb_l10",
    "total_bid", "total_ask",
    "max_bid_wall_sz", "max_bid_wall_dist",
    "max_ask_wall_sz", "max_ask_wall_dist",
] + [f"bid_sz_{i}" for i in range(1, DOM_N_LEVELS + 1)] \
  + [f"ask_sz_{i}" for i in range(1, DOM_N_LEVELS + 1)]

_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode    = ssl.CERT_NONE


# ── DOMBook ───────────────────────────────────────────────────────────────────

@dataclass
class DOMBook:
    """Live order book maintained via GatewayDepth WebSocket updates."""
    bids:        dict = field(default_factory=dict)   # price → size
    asks:        dict = field(default_factory=dict)
    best_bid:    float | None = None
    best_ask:    float | None = None
    last_price:  float | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _lock:       threading.Lock = field(default_factory=threading.Lock)

    BID   = 4
    ASK   = 3
    RESET = 6

    def apply_depth(self, updates: list):
        with self._lock:
            for u in updates:
                price    = float(u.get("price",  0))
                volume   = float(u.get("volume", 0))
                dom_type = u.get("type", -1)
                if dom_type == self.RESET:
                    self.bids.clear(); self.asks.clear()
                elif dom_type == self.BID:
                    if volume == 0: self.bids.pop(price, None)
                    else:           self.bids[price] = volume
                elif dom_type == self.ASK:
                    if volume == 0: self.asks.pop(price, None)
                    else:           self.asks[price] = volume
            self.last_update = datetime.now(timezone.utc)

    def apply_quote(self, last, bid, ask):
        with self._lock:
            if last is not None: self.last_price = float(last)
            if bid  is not None: self.best_bid   = float(bid)
            if ask  is not None: self.best_ask   = float(ask)
            self.last_update = datetime.now(timezone.utc)

    def snapshot_features(self, n: int = DOM_N_LEVELS) -> dict:
        """Return a flat dict of DOM features for logging at signal fire time."""
        with self._lock:
            bb, ba = self.best_bid, self.best_ask
            if bb and ba:
                bids = sorted(((p, s) for p, s in self.bids.items() if p < ba), reverse=True)[:n]
                asks = sorted((p, s) for p, s in self.asks.items() if p > bb)[:n]
            else:
                bids = sorted(self.bids.items(), reverse=True)[:n]
                asks = sorted(self.asks.items())[:n]

        bid_sizes = [s for _, s in bids]
        ask_sizes = [s for _, s in asks]
        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        denom     = total_bid + total_ask

        def imb(b_list, a_list):
            b, a = sum(b_list), sum(a_list)
            d = b + a
            return round((b - a) / d, 4) if d else None

        mid = ((bb + ba) / 2) if bb and ba else None

        def wall(side_list, is_ask):
            if not side_list: return None, None
            max_sz  = max(side_list)
            max_idx = side_list.index(max_sz)
            lvls    = asks if is_ask else bids
            if max_idx < len(lvls):
                dist = abs(lvls[max_idx][0] - mid) if mid else None
                return max_sz, round(dist, 4) if dist is not None else None
            return max_sz, None

        bid_wall_sz, bid_wall_dist = wall(bid_sizes, False)
        ask_wall_sz, ask_wall_dist = wall(ask_sizes, True)

        row = {
            "best_bid":  bb,
            "best_ask":  ba,
            "spread":    round(ba - bb, 4) if bb and ba else None,
            "imb_l1":    imb(bid_sizes[:1], ask_sizes[:1]),
            "imb_l5":    imb(bid_sizes[:5], ask_sizes[:5]),
            "imb_l10":   imb(bid_sizes[:10], ask_sizes[:10]),
            "total_bid": total_bid,
            "total_ask": total_ask,
            "max_bid_wall_sz":   bid_wall_sz,
            "max_bid_wall_dist": bid_wall_dist,
            "max_ask_wall_sz":   ask_wall_sz,
            "max_ask_wall_dist": ask_wall_dist,
        }
        for i in range(1, n + 1):
            row[f"bid_sz_{i}"] = bid_sizes[i - 1] if i - 1 < len(bid_sizes) else None
            row[f"ask_sz_{i}"] = ask_sizes[i - 1] if i - 1 < len(ask_sizes) else None
        return row

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_update).total_seconds()


# ── SignalR WebSocket client (market hub) ────────────────────────────────────

class SignalRConn:
    RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]
    PING_INTERVAL    = 15

    def __init__(self, hub_url, token_factory, on_message, on_connected, name="hub"):
        self.hub_url       = hub_url
        self.token_factory = token_factory
        self.on_message    = on_message
        self.on_connected  = on_connected
        self.name          = name
        self._ws           = None
        self._running      = False
        self._send_lock    = threading.Lock()
        self._inv_id       = 0

    def start(self):
        self._running = True
        threading.Thread(target=self._run_loop, daemon=True, name=self.name).start()

    def stop(self):
        self._running = False
        if self._ws:
            try: self._ws.close()
            except: pass

    def send(self, target, arguments):
        self._inv_id += 1
        msg = json.dumps({"type": 1, "invocationId": str(self._inv_id),
                          "target": target, "arguments": arguments}) + DOM_RS
        with self._send_lock:
            if self._ws:
                try: self._ws.send(msg)
                except: pass

    def _url(self):
        return f"{self.hub_url}?access_token={self.token_factory()}"

    def _run_loop(self):
        attempt = 0
        while self._running:
            try:
                self._connect()
                attempt = 0
            except Exception as e:
                if not self._running: break
                delay = self.RECONNECT_DELAYS[min(attempt, len(self.RECONNECT_DELAYS) - 1)]
                attempt += 1
                log.warning(f"[{self.name}] disconnected ({e}), reconnecting in {delay}s")
                time.sleep(delay)

    def _connect(self):
        ws = websocket.create_connection(self._url(), sslopt={"context": _ssl_ctx}, timeout=30)
        self._ws = ws
        ws.send('{"protocol":"json","version":1}' + DOM_RS)
        raw    = ws.recv()
        frames = [f for f in raw.split(DOM_RS) if f.strip()]
        hs     = json.loads(frames[0])
        if hs.get("error"):
            raise RuntimeError(f"Handshake error: {hs['error']}")
        for f in frames[1:]:
            try: self._dispatch(json.loads(f))
            except: pass
        self.on_connected(self)

        ping_stop = threading.Event()
        threading.Thread(target=self._ping_loop, args=(ws, ping_stop), daemon=True).start()
        try:
            while self._running:
                raw = ws.recv()
                for f in raw.split(DOM_RS):
                    f = f.strip()
                    if not f: continue
                    try: self._dispatch(json.loads(f))
                    except: pass
        finally:
            ping_stop.set()
            try: ws.close()
            except: pass
            self._ws = None

    def _dispatch(self, msg):
        if msg.get("type") == 1:
            self.on_message(msg)

    def _ping_loop(self, ws, stop):
        while not stop.wait(self.PING_INTERVAL):
            try:
                with self._send_lock:
                    ws.send('{"type":6}' + DOM_RS)
            except: break


# ── DOM signal logging ────────────────────────────────────────────────────────

def _ensure_dom_signal_log():
    DOM_SIGNAL_LOG.parent.mkdir(exist_ok=True)
    if not DOM_SIGNAL_LOG.exists():
        with open(DOM_SIGNAL_LOG, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=DOM_SIGNAL_FIELDS).writeheader()


def log_dom_at_signal(strategy: str, symbol: str, direction: int, entry: float,
                      dom: DOMBook):
    """Capture current DOM book state and append to dom_at_signal.csv."""
    age = dom.age_seconds()
    if age > 30:
        return  # no live DOM feed in trading_bot; skip silently
    feats = dom.snapshot_features()
    row = {
        "ts":        datetime.now(timezone.utc).isoformat(),
        "strategy":  strategy,
        "symbol":    symbol,
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry":     entry,
    }
    row.update(feats)
    # Fill any missing fields with empty string
    for fld in DOM_SIGNAL_FIELDS:
        row.setdefault(fld, "")
    with open(DOM_SIGNAL_LOG, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=DOM_SIGNAL_FIELDS).writerow(row)


def play_trade_sound():
    """Play trade-execution sound non-blocking. Currently disabled."""
    pass


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
    # SLR_Scalp: enabled per instrument (both RTH and Globex).
    slr_enabled:   bool  = False
    slr_vol_mult:  float = SLR_VOL_MULT   # per-instrument vol surge threshold
    # Evening Resumption: gap-fade/follow at 18:00 ET daily close resume; MES only.
    eve_enabled: bool = False
    # Sunday Open Gap: weekly Globex open gap at 18:00 ET Sunday; MES only.
    sun_gap_enabled: bool = False
    # PL_Mom: Price Linearity Momentum on 5s bars. RTH only.
    pl_mom_enabled:        bool  = False
    pl_mom_entry_pl:       float = PL_MOM_ENTRY_PL
    pl_mom_move_bps:       float = PL_MOM_MOVE_BPS   # floor; adaptive = max(this, sigma_n × σ)
    pl_mom_exit_pl:        float = PL_MOM_EXIT_PL
    pl_mom_stop_bps:       float = PL_MOM_STOP_BPS
    pl_mom_sigma_lookback: int   = PL_MOM_SIGMA_LOOKBACK  # 5s bars for σ computation


INSTRUMENTS = [
    BotInstrument("MES", "MES", tick_size=0.25, point_value=5.00,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (16,  0,  9,  0, False),  # trade 09:00–16:00 ET only
                  ],
                  orb_enabled=True, orb_width_pct_min=0.00354,
                  vwaslr_n=50, vwaslr_threshold=0.4, vwaslr_start=(9, 0),
                  slr_enabled=True,
                  eve_enabled=False,
                  sun_gap_enabled=False,
                  pl_mom_enabled=True, pl_mom_entry_pl=0.80, pl_mom_move_bps=8.0,
                  pl_mom_stop_bps=7.0, pl_mom_exit_pl=0.40),
    BotInstrument("MNQ", "MNQ", tick_size=0.25, point_value=2.00,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (16,  0,  9,  0, False),  # trade 09:00–16:00 ET only
                  ],
                  orb_enabled=False,
                  vwaslr_n=0,
                  slr_enabled=True,
                  slr_vol_mult=15.0,            # raised from 7x: May analysis showed 7-10x MNQ signals have no edge
                  pl_mom_enabled=True,
                  pl_mom_stop_bps=8.0, pl_mom_exit_pl=0.20),
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
class SLRScalpSignal:
    """Volume Surge signal (LONG or SHORT). Entry fires immediately after surge bar closes."""
    entry:     float
    target:    float   # LONG: entry + target_bps; SHORT: entry - target_bps
    stop:      float   # LONG: entry - stop_bps;   SHORT: entry + stop_bps
    surge_ts:  datetime   # timestamp of the volume surge bar
    bar_ts:    datetime   # same as surge_ts (no pullback)
    vol_ratio: float
    move_bps:  float
    is_rth:    bool
    direction: int = 1    # 1 = LONG, -1 = SHORT

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)

    def expires_at(self) -> datetime:
        hold = SLR_HOLD_RTH if self.is_rth else SLR_HOLD_GLOBEX
        return self.bar_ts + timedelta(minutes=hold)


@dataclass
class ActiveSLRTrade:
    instrument:  BotInstrument
    contract_id: str
    sig:         SLRScalpSignal
    fired_at:    datetime
    order_id:    int | None = None
    fill_price:  float | None = None

    def target_price(self) -> float:
        p = self.fill_price or self.sig.entry
        return p + self.sig.direction * self.sig.target_pts()

    def stop_price(self) -> float:
        p = self.fill_price or self.sig.entry
        return p - self.sig.direction * self.sig.stop_pts()


@dataclass
class PLMomSignal:
    """Price Linearity Momentum signal on 5s bars. RTH only."""
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
class ActivePLMomTrade:
    instrument:  BotInstrument
    contract_id: str
    sig:         PLMomSignal
    fired_at:    datetime
    entry_ts:    datetime    # actual entry time (for min_hold check)
    order_id:    int | None = None
    fill_price:  float | None = None


@dataclass
class EveningResumeSignal:
    """18:00 ET gap resumption signal (MES only). Time exit, no brackets."""
    entry:      float
    prev_close: float
    gap_pct:    float    # signed fraction: positive = gap up, negative = gap down
    bar_ts:     datetime
    direction:  int = 1  # +1 = gap up (LONG), -1 = gap down (SHORT)


@dataclass
class ActiveEveningTrade:
    instrument:  BotInstrument
    contract_id: str
    sig:         EveningResumeSignal
    fired_at:    datetime
    order_id:    int | None = None
    fill_price:  float | None = None
    expires_at:  datetime = field(init=False)

    def __post_init__(self):
        self.expires_at = self.fired_at + timedelta(minutes=EVE_HOLD_MINUTES)


@dataclass
class SundayGapSignal:
    """Sunday 18:00 ET Globex open gap signal. Time exit only, no brackets."""
    entry:      float
    fri_close:  float
    gap_pct:    float    # signed fraction; positive = gap up
    vol_ratio:  float    # first_bar_vol / vol_baseline (0.0 if no baseline yet)
    bar_ts:     datetime
    direction:  int = 1  # +1 = gap up (LONG), -1 = gap down (SHORT)


@dataclass
class ActiveSundayGapTrade:
    instrument:  BotInstrument
    contract_id: str
    sig:         SundayGapSignal
    fired_at:    datetime
    order_id:    int | None = None
    fill_price:  float | None = None
    expires_at:  datetime = field(init=False)

    def __post_init__(self):
        self.expires_at = self.fired_at + timedelta(minutes=SUN_HOLD_MINUTES)


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
    active_slr_trade:     ActiveSLRTrade | None = None
    orb:                  OrbState = field(default_factory=OrbState)
    last_evaluated_ts:    datetime | None = None
    vwaslr_last_ts:       datetime | None = None
    vwaslr_fetch_min:     int = -1               # UTC minute of last vwaslr fetch (throttle)
    vwaslr_last_fetch:    datetime | None = None  # wall-clock time of last vwaslr fetch (stale retry)
    vwaslr_ema:           float = 0.0            # EMA-10 of raw VWASLR (updated every poll)
    vwaslr_ema_prev:      float = 0.0            # EMA value before last update (cross detection)
    slr_last_surge_ts:    datetime | None = None  # surge bar ts of last fired SLR signal (dedup)
    slr_last_bar_ts:      datetime | None = None  # 1-min bar ts last evaluated for SLR (gate)
    dom:                  DOMBook = field(default_factory=DOMBook)
    # PL_Mom state
    pl_mom_5s_bars:       list[Bar] = field(default_factory=list)
    active_pl_mom_trade:  "ActivePLMomTrade | None" = None
    pl_mom_last_bar_ts:   "datetime | None" = None
    pl_mom_sigma_30s_bps: float = 0.0  # rolling σ of 30s (6-bar) moves in bps
    eve_prev_close:       float | None = None     # last RTH close before 18:00 ET gap
    eve_fired_date:       date  | None = None     # ET date of last evening resumption trade
    active_evening_trade: ActiveEveningTrade | None = None
    sun_fri_close:           float | None = None  # last Friday RTH close (persisted to disk)
    sun_vol_history:         list = field(default_factory=list)  # prior Sunday 1st-bar volumes
    sun_vol_baseline:        float = 0.0          # median of sun_vol_history
    sun_fired_date:          date | None = None   # ET date of last Sunday gap trade
    sun_vol_rec_date:        date | None = None   # ET date when current Sunday vol was recorded
    active_sunday_gap_trade: ActiveSundayGapTrade | None = None


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
                  signal_bar_ts: datetime, direction: int,
                  symbol: str = "") -> float | None:
    """
    Fetch PL_N_BARS 1-min bars ending just before the signal 5-min bar and
    return PL_aligned = (signed path length) × direction.
    +1 = 1-min flow perfectly aligned; ≥ PL_THRESH → 2× sizing.
    Returns None on fetch error or insufficient data.
    """
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
    max_mom  = max(bars for inst in INSTRUMENTS for _, bars in inst.csr_vol_windows)
    lookback = TRAILING_BARS + max_mom + 10
    now_utc  = datetime.now(timezone.utc)
    db_fresh = False
    if bars_db_available():
        raw = get_bars_from_db(state.instrument.symbol, TF_MINUTES, lookback)
        if raw:
            db_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                           open=b["o"], high=b["h"], low=b["l"],
                           close=b["c"], volume=b["v"]) for b in raw]
            state.bars = db_bars
            db_fresh = (now_utc - db_bars[-1].ts).total_seconds() < TF_MINUTES * 60 * 2
    if not db_fresh:
        end   = now_utc
        start = end - timedelta(minutes=TF_MINUTES * lookback)
        raw = list(reversed(client.get_bars(
            contract_id=state.contract_id,
            start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
            limit=lookback,
        )))
        state.bars = [
            Bar(ts=datetime.fromisoformat(b["t"]),
                open=b["o"], high=b["h"], low=b["l"],
                close=b["c"], volume=b["v"])
            for b in raw
        ]


def fetch_vwaslr_bars(client: TopstepClient, state: InstrumentState):
    """Fetch 1-min bars for VWASLR. Uses DB when available and fresh;
    falls back to incremental REST fetching when DB is stale or absent."""
    now_utc   = datetime.now(timezone.utc)
    now_floor = datetime.fromtimestamp(
        (int(now_utc.timestamp()) // 60) * 60, tz=timezone.utc)
    db_fresh  = False
    if bars_db_available():
        # Fetch one extra bar; exclude the currently-forming bar so bars[-1]
        # is always a completed bar (bar_collector's 30s flush writes partial bars).
        raw = get_bars_from_db(state.instrument.symbol, 1, VWASLR_INIT_BARS + 1)
        db_bars = [
            Bar(ts=datetime.fromisoformat(b["t"]),
                open=b["o"], high=b["h"], low=b["l"],
                close=b["c"], volume=b["v"])
            for b in raw
            if datetime.fromisoformat(b["t"]) < now_floor
        ]
        if db_bars:
            state.vwaslr_bars = db_bars
            db_fresh = (now_utc - db_bars[-1].ts).total_seconds() < 180
    if not db_fresh and not state.vwaslr_bars:
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
    elif not db_fresh:
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
    min_stop_pts   = sig["entry"] * SLR_STOP_BPS / 10000.0
    stop_pts       = max(inst.stop_sigma   * sigma_pts, min_stop_pts)
    target_pts     = max(inst.target_sigma * sigma_pts,
                         min_stop_pts * inst.target_sigma / inst.stop_sigma)
    stop_mag   = max(1, round(stop_pts   / tick))
    target_mag = max(1, round(target_pts / tick))
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
        min_trail  = trade.fill_price * SLR_STOP_BPS / 10000.0
        trail_dist = max(CSR_TRAIL_SIGMA * trade.sigma_pts, min_trail)
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
                # Fetch actual fill — trail_stop is the trigger level, not the fill price.
                # Market close can fill significantly away from the trigger on fast instruments.
                actual_exit = _get_exit_price(client, account_id, trade.fired_at,
                                              trade.contract_id, now)
                if actual_exit is None:
                    actual_exit = trail_stop   # fallback to trigger level if lookup fails
                    log.warning(f"{trade.instrument.symbol}: trail exit fill not found, using trigger level {trail_stop:.2f}")
                else:
                    log.info(f"{trade.instrument.symbol}: trail exit fill={actual_exit:.2f}  trigger={trail_stop:.2f}  slip={actual_exit - trail_stop:+.2f}")
                _log_trade(trade, "TRAIL STOP", actual_exit, now)
                state.active_trade = None
                try:
                    client.cancel_all_orders(account_id)
                except Exception:
                    pass
                return

    if pos is None:
        # Position is gone — brackets closed it; cancel any residual OCO orders
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now,
                                     entry_price=trade.fill_price or trade.est_entry)
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
            custom_tag=f"orb_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
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
                    fired_at: datetime, contract_id: str, now: datetime,
                    entry_price: float | None = None) -> float | None:
    """Fetch the actual exit fill price from trade history.

    Filters to trades after fired_at on the given contract.  If entry_price is
    provided, skips fills within 1 tick of entry (avoids returning the entry fill
    as the exit when only one trade is returned by the API).
    """
    try:
        trades = client.search_trades(account_id, fired_at, now)
        closing = [
            t for t in trades
            if t.get("contractId") == contract_id
            and datetime.fromisoformat(t.get("timestamp", "1970-01-01T00:00:00")).replace(tzinfo=timezone.utc)
                > fired_at
        ]
        if not closing:
            log.debug(f"_get_exit_price: no trades found after {fired_at} for {contract_id}")
            return None
        # If multiple fills, the last one is the exit
        # If only one fill and it matches entry, it's the entry fill — skip it
        price = float(closing[-1].get("price", 0))
        if price <= 0:
            return None
        if entry_price is not None and len(closing) == 1 and abs(price - entry_price) < 1.0:
            log.debug(f"_get_exit_price: only fill matches entry ({price:.2f}) — waiting for exit fill")
            return None
        log.debug(f"_get_exit_price: found exit fill {price:.2f} ({len(closing)} fills total)")
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

    direction  = 1 if ema > 0 else -1
    entry      = last.close
    min_stop        = entry * SLR_STOP_BPS / 10000.0      # 10bp floor
    stop_dist       = max(VWASLR_STOP_SIGMA  * sigma_pts, min_stop)
    target_dist     = max(VWASLR_TARGET_SIGMA * sigma_pts,
                          min_stop * VWASLR_TARGET_SIGMA / VWASLR_STOP_SIGMA)
    stop   = entry - direction * stop_dist
    target = entry + direction * target_dist
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

    log_dom_at_signal("VWASLR", inst.symbol, sig.direction, sig.entry, state.dom)

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
            custom_tag=f"vwas_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
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


# ── SLR_Scalp signal ─────────────────────────────────────────────────────────

def _ensure_slr_log():
    SLR_LOG_PATH.parent.mkdir(exist_ok=True)
    if not SLR_LOG_PATH.exists():
        with open(SLR_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SLR_LOG_FIELDS).writeheader()


def _log_slr_trade(trade: ActiveSLRTrade, outcome: str,
                   exit_price: float, now: datetime):
    fill    = trade.fill_price or trade.sig.entry
    pnl_pts = (exit_price - fill) * trade.sig.direction
    sess    = "RTH" if trade.sig.is_rth else "GLOBEX"
    dirn    = "LONG" if trade.sig.direction == 1 else "SHORT"
    row = {
        "fired_at":    trade.fired_at.isoformat(),
        "resolved_at": now.isoformat(),
        "symbol":      trade.instrument.symbol,
        "direction":   dirn,
        "est_entry":   round(trade.sig.entry, 4),
        "fill_price":  round(fill, 4),
        "target":      round(trade.target_price(), 4),
        "stop":        round(trade.stop_price(), 4),
        "vol_ratio":   round(trade.sig.vol_ratio, 4),
        "move_bps":    round(trade.sig.move_bps, 4),
        "session":     sess,
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
    }
    with open(SLR_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SLR_LOG_FIELDS).writerow(row)
    log.info(
        f"SLR LOGGED  {trade.instrument.symbol} {dirn}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  "
        f"pnl={pnl_pts:+.2f}pts  session={sess}"
    )


def evaluate_slr_scalp(state: InstrumentState) -> SLRScalpSignal | None:
    """
    Scan state.vwaslr_bars for vol surge (LONG or SHORT). No pullback required —
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

    # 1) Volume surge: vol ≥ instrument.slr_vol_mult × 20-bar median
    prior_vols = [bars[j].volume for j in range(surge_idx - SLR_VOL_LOOKBACK, surge_idx)]
    med_vol    = float(np.median(prior_vols)) if prior_vols else 0.0
    if med_vol <= 0:
        return None
    vol_ratio = surge_bar.volume / med_vol
    if vol_ratio < state.instrument.slr_vol_mult:
        return None

    # 2) Determine direction from surge bar; flat bars are skipped
    if surge_bar.close > surge_bar.open:
        direction = 1
    elif surge_bar.close < surge_bar.open:
        direction = -1
    else:
        return None

    # 3) Directional move ≥ SLR_MOVE_BPS (WO2: open[surge-1] → close[surge])
    prev_bar = bars[surge_idx - 1]
    if prev_bar.open <= 0:
        return None
    if (surge_bar.ts - prev_bar.ts) > timedelta(minutes=2):
        return None
    surge_move_bps = direction * (surge_bar.close - prev_bar.open) / prev_bar.open * 10000
    if surge_move_bps < SLR_MOVE_BPS:
        return None

    entry     = surge_bar.close
    sigma_bps = (state.sigma_pts / entry * 10000) if state.sigma_pts else 0.0
    stop_bps  = max(SLR_STOP_BPS, sigma_bps)
    target    = entry * (1.0 + direction * SLR_TARGET_BPS / 10000.0)
    stop      = entry * (1.0 - direction * stop_bps      / 10000.0)
    is_rth    = (9, 30) <= (surge_et.hour, surge_et.minute) < (16, 0)

    return SLRScalpSignal(
        entry=entry, target=target, stop=stop,
        surge_ts=surge_bar.ts, bar_ts=surge_bar.ts,
        vol_ratio=vol_ratio, move_bps=surge_move_bps,
        is_rth=is_rth, direction=direction,
    )


def place_slr_signal(client: TopstepClient, state: InstrumentState,
                     sig: SLRScalpSignal, account_id: int,
                     paper: bool) -> ActiveSLRTrade:
    inst      = state.instrument
    tick      = inst.tick_size
    stop_mag   = max(1, round(sig.stop_pts()   / tick))
    target_mag = max(1, round(sig.target_pts() / tick))
    stop_ticks   = -sig.direction * stop_mag    # LONG: below entry; SHORT: above entry
    target_ticks =  sig.direction * target_mag  # LONG: above entry; SHORT: below entry
    side  = TopstepClient.BID if sig.direction == 1 else TopstepClient.ASK
    dirn  = "LONG" if sig.direction == 1 else "SHORT"
    sess  = "RTH" if sig.is_rth else "GLOBEX"

    trade = ActiveSLRTrade(
        instrument=inst, contract_id=state.contract_id,
        sig=sig, fired_at=sig.bar_ts,
    )

    log_dom_at_signal("SLR", inst.symbol, sig.direction, sig.entry, state.dom)

    if paper:
        log.info(
            f"[PAPER] SLR {inst.symbol} {dirn}  entry≈{sig.entry:.2f}  "
            f"target={sig.target:.2f} ({sig.target_pts():.2f}pts)  "
            f"stop={sig.stop:.2f} ({sig.stop_pts():.2f}pts)  "
            f"vol={sig.vol_ratio:.1f}×  move={sig.move_bps:.1f}bp  {sess}"
        )
    else:
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=side,
            size=1,
            order_type=TopstepClient.ORDER_MARKET,
            stop_loss_ticks=stop_ticks,
            take_profit_ticks=target_ticks,
            custom_tag=f"slr_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"SLR ORDER  {inst.symbol} {dirn}  order_id={trade.order_id}  "
            f"entry≈{sig.entry:.2f}  stop={stop_ticks}t  target={target_ticks}t  "
            f"vol={sig.vol_ratio:.1f}×  move={sig.move_bps:.1f}bp  {sess}"
        )

    state.active_slr_trade = trade
    return trade


def handle_active_slr_trade(client: TopstepClient, state: InstrumentState,
                             account_id: int, now: datetime, paper: bool):
    trade = state.active_slr_trade

    if paper:
        if now >= trade.sig.expires_at():
            exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                          else trade.sig.entry)
            _log_slr_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_slr_trade = None
        return

    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"SLR {trade.instrument.symbol}: could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"SLR {trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    if pos is None:
        # Position closed — brackets hit
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now)
        if exit_price is not None:
            if trade.sig.direction == 1:
                outcome = "TARGET" if exit_price >= trade.target_price() else "STOPPED"
            else:
                outcome = "TARGET" if exit_price <= trade.target_price() else "STOPPED"
        else:
            outcome, exit_price = _classify_slr_outcome(trade, state.vwaslr_bars)
        _log_slr_trade(trade, outcome, exit_price, now)
        state.active_slr_trade = None
        try:
            n = client.cancel_all_orders(account_id)
            if n:
                log.info(f"SLR {trade.instrument.symbol} {outcome}: cancelled {n} residual order(s)")
        except Exception as e:
            log.warning(f"SLR {trade.instrument.symbol}: cancel_all_orders failed: {e}")
        return

    if now >= trade.sig.expires_at():
        log.info(f"SLR {trade.instrument.symbol} max hold reached — closing")
        try:
            client.cancel_all_orders(account_id)
        except Exception as e:
            log.warning(f"SLR {trade.instrument.symbol}: pre-close cancel_all failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"SLR {trade.instrument.symbol}: failed to close position: {e}")
            return
        exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                      else (trade.fill_price or trade.sig.entry))
        _log_slr_trade(trade, "TIME EXIT", exit_price, now)
        state.active_slr_trade = None
        try:
            client.cancel_all_orders(account_id)
        except Exception:
            pass


def _classify_slr_outcome(trade: ActiveSLRTrade,
                           bars: list[Bar]) -> tuple[str, float]:
    """Infer bracket outcome from 1-min bar OHLC (direction-aware, stop-first)."""
    d = trade.sig.direction
    for bar in bars:
        if bar.ts <= trade.fired_at:
            continue
        if d == 1:
            if bar.low  <= trade.stop_price():   return "STOPPED", trade.stop_price()
            if bar.high >= trade.target_price():  return "TARGET",  trade.target_price()
        else:
            if bar.high >= trade.stop_price():   return "STOPPED", trade.stop_price()
            if bar.low  <= trade.target_price():  return "TARGET",  trade.target_price()
    last_close = bars[-1].close if bars else (trade.fill_price or trade.sig.entry)
    if abs(last_close - trade.target_price()) <= abs(last_close - trade.stop_price()):
        return "TARGET",  trade.target_price()
    return "STOPPED", trade.stop_price()


# ── PL_Mom ───────────────────────────────────────────────────────────────────

def _ensure_pl_mom_log():
    PL_MOM_LOG_PATH.parent.mkdir(exist_ok=True)
    if not PL_MOM_LOG_PATH.exists():
        with open(PL_MOM_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=PL_MOM_LOG_FIELDS).writeheader()


def _log_pl_mom_trade(trade: ActivePLMomTrade, outcome: str,
                      exit_price: float, now: datetime):
    fill    = trade.fill_price or trade.sig.entry
    pnl_pts = (exit_price - fill) * trade.sig.direction
    dirn    = "LONG" if trade.sig.direction == 1 else "SHORT"
    row = {
        "fired_at":    trade.fired_at.isoformat(),
        "resolved_at": now.isoformat(),
        "symbol":      trade.instrument.symbol,
        "direction":   dirn,
        "est_entry":   round(trade.sig.entry, 4),
        "fill_price":  round(fill, 4),
        "stop":        round(trade.sig.stop, 4),
        "pl":          round(trade.sig.pl, 4),
        "move_bps":    round(trade.sig.move_bps, 4),
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
    }
    with open(PL_MOM_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=PL_MOM_LOG_FIELDS).writerow(row)
    log.info(
        f"PL_MOM LOGGED  {trade.instrument.symbol} {dirn}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  pnl={pnl_pts:+.2f}pts  "
        f"pl={trade.sig.pl:.3f}  move={trade.sig.move_bps:.1f}bp"
    )


def _compute_pl_mom_sigma(bars: list, lookback: int) -> float:
    """Compute rolling σ of 30s (PL_MOM_WINDOW × 5s) net moves in bps
    using up to `lookback` 5s bars. Returns 0.0 if insufficient data."""
    closes = np.array([b.close for b in bars], dtype=float)
    if len(closes) < PL_MOM_WINDOW + 2:
        return 0.0
    # Per-5s-bar log return, then sum consecutive windows of PL_MOM_WINDOW bars
    rets = np.log(closes[1:] / closes[:-1])
    window_moves = []
    n = len(rets)
    lb = min(lookback, n - PL_MOM_WINDOW + 1)
    for i in range(n - lb, n - PL_MOM_WINDOW + 1):
        if i < 0:
            continue
        window_moves.append(abs(rets[i:i + PL_MOM_WINDOW].sum()))
    if len(window_moves) < 4:
        return 0.0
    return float(np.std(window_moves)) * 10000  # convert to bps


def fetch_pl_mom_bars(client: TopstepClient, state: InstrumentState):
    """Fetch the latest PL_MOM_5S_FETCH 5-second bars. Uses bar_collector DB
    when available; falls back to REST API. Also updates pl_mom_sigma_30s_bps."""
    try:
        if bars_db_available():
            raw = get_5s_bars_from_db(state.instrument.symbol, PL_MOM_5S_FETCH)
            if raw:
                state.pl_mom_5s_bars = [
                    Bar(ts=datetime.fromisoformat(b["t"]),
                        open=b["o"], high=b["h"], low=b["l"],
                        close=b["c"], volume=b["v"])
                    for b in raw
                ]
                state.pl_mom_sigma_30s_bps = _compute_pl_mom_sigma(
                    state.pl_mom_5s_bars, state.instrument.pl_mom_sigma_lookback)
                return
        # Fallback: REST API
        now_utc      = datetime.now(timezone.utc)
        now_floor_5s = datetime.fromtimestamp(
            (int(now_utc.timestamp()) // 5) * 5, tz=timezone.utc)
        start = now_floor_5s - timedelta(seconds=5 * (PL_MOM_5S_FETCH + 5))
        raw   = client.get_bars(
            contract_id=state.contract_id,
            start=start, end=now_floor_5s,
            unit=TopstepClient.SECOND, unit_number=5,
            limit=PL_MOM_5S_FETCH,
        )
        if raw:
            state.pl_mom_5s_bars = [
                Bar(ts=datetime.fromisoformat(b["t"]),
                    open=b["o"], high=b["h"], low=b["l"],
                    close=b["c"], volume=b["v"])
                for b in raw
            ]
            state.pl_mom_sigma_30s_bps = _compute_pl_mom_sigma(
                state.pl_mom_5s_bars, state.instrument.pl_mom_sigma_lookback)
    except Exception as e:
        log.debug(f"fetch_pl_mom_bars {state.instrument.symbol}: {e}")


def evaluate_pl_mom(state: InstrumentState) -> PLMomSignal | None:
    """
    Evaluate Price Linearity Momentum on the last PL_MOM_WINDOW completed 5s bars.
    Returns a PLMomSignal if PL ≥ entry threshold and net move ≥ move threshold.
    Skips gaps (consecutive bar timestamps more than 8s apart).
    """
    bars = state.pl_mom_5s_bars
    if len(bars) < PL_MOM_WINDOW + 1:
        return None

    window_bars = bars[-PL_MOM_WINDOW:]

    # Gap detection: consecutive timestamps must be ≤ 8s apart
    for i in range(1, len(window_bars)):
        dt = (window_bars[i].ts - window_bars[i - 1].ts).total_seconds()
        if dt > 8:
            return None

    last_bar = window_bars[-1]

    closes  = np.array([b.close for b in window_bars], dtype=float)
    rets    = np.log(closes[1:] / closes[:-1])
    sum_abs = float(np.abs(rets).sum())
    if sum_abs == 0:
        return None

    pl = float(abs(rets.sum()) / sum_abs)
    if pl < state.instrument.pl_mom_entry_pl:
        return None

    net_ret  = float(rets.sum())
    entry    = last_bar.close
    if entry <= 0:
        return None
    move_bps = abs(net_ret) * 10000

    # Sigma-adaptive threshold: max(floor, N_sigma × σ_30s_bps)
    sigma = state.pl_mom_sigma_30s_bps
    if sigma > 0:
        effective_threshold = max(state.instrument.pl_mom_move_bps,
                                  PL_MOM_SIGMA_N * sigma)
    else:
        effective_threshold = state.instrument.pl_mom_move_bps
    if move_bps < effective_threshold:
        return None

    direction = 1 if net_ret > 0 else -1
    stop      = entry * (1.0 - direction * state.instrument.pl_mom_stop_bps / 10000.0)
    bar_et    = last_bar.ts.astimezone(ET)
    is_rth    = (9, 30) <= (bar_et.hour, bar_et.minute) < (16, 0)

    return PLMomSignal(
        direction=direction, entry=entry, stop=stop,
        pl=pl, move_bps=move_bps,
        bar_ts=last_bar.ts, is_rth=is_rth,
    )


def place_pl_mom_signal(client: TopstepClient, state: InstrumentState,
                        sig: PLMomSignal, account_id: int,
                        paper: bool, now: datetime) -> ActivePLMomTrade:
    inst      = state.instrument
    tick      = inst.tick_size
    is_long   = sig.direction == 1
    dir_label = "LONG" if is_long else "SHORT"
    stop_mag  = max(1, round(sig.stop_pts() / tick))
    # Stop only — no fixed target for PL_Mom (exits on PL drop or time)
    stop_ticks = -stop_mag if is_long else stop_mag

    trade = ActivePLMomTrade(
        instrument=inst, contract_id=state.contract_id,
        sig=sig, fired_at=sig.bar_ts, entry_ts=now,
    )

    log_dom_at_signal("PL_MOM", inst.symbol, sig.direction, sig.entry, state.dom)

    sigma = state.pl_mom_sigma_30s_bps
    sigma_str = f"σ={sigma:.1f}bp  thr={max(inst.pl_mom_move_bps, PL_MOM_SIGMA_N * sigma):.1f}bp" if sigma > 0 else f"thr={inst.pl_mom_move_bps:.1f}bp"
    if paper:
        log.info(
            f"[PAPER] PL_MOM {inst.symbol} {dir_label}  entry≈{sig.entry:.2f}  "
            f"stop={sig.stop:.2f} ({sig.stop_pts():.2f}pts  {PL_MOM_STOP_BPS:.0f}bp)  "
            f"pl={sig.pl:.3f}  move={sig.move_bps:.1f}bp  {sigma_str}  max={PL_MOM_MAX_HOLD_S}s"
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
            custom_tag=f"plm_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"PL_MOM ORDER  {inst.symbol} {dir_label}  order_id={trade.order_id}  "
            f"entry≈{sig.entry:.2f}  stop={stop_ticks}t  "
            f"pl={sig.pl:.3f}  move={sig.move_bps:.1f}bp  {sigma_str}"
        )

    state.active_pl_mom_trade = trade
    return trade


def handle_active_pl_mom_trade(client: TopstepClient, state: InstrumentState,
                                account_id: int, now: datetime, paper: bool):
    trade = state.active_pl_mom_trade

    if paper:
        # Paper mode: time exit only
        if now >= trade.sig.expires_at():
            exit_price = (state.pl_mom_5s_bars[-1].close if state.pl_mom_5s_bars
                          else trade.sig.entry)
            _log_pl_mom_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_pl_mom_trade = None
        return

    # Live mode: check position, PL exit, and time exit
    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"PL_MOM {trade.instrument.symbol}: could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"PL_MOM {trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    if pos is None:
        # Position closed — stop hit (bracket order triggered)
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now,
                                     entry_price=trade.fill_price or trade.sig.entry)
        exit_price = exit_price or trade.sig.stop
        _log_pl_mom_trade(trade, "STOPPED", exit_price, now)
        state.active_pl_mom_trade = None
        try:
            n = client.cancel_all_orders(account_id)
            if n:
                log.info(f"PL_MOM {trade.instrument.symbol} STOPPED: cancelled {n} residual order(s)")
        except Exception as e:
            log.warning(f"PL_MOM {trade.instrument.symbol}: cancel_all_orders failed: {e}")
        return

    # PL exit: only check after min hold
    min_hold_ok = (now - trade.entry_ts).total_seconds() >= PL_MOM_MIN_HOLD_S
    if min_hold_ok and state.pl_mom_5s_bars:
        window = state.pl_mom_5s_bars[-PL_MOM_WINDOW:]
        if len(window) >= PL_MOM_WINDOW:
            closes  = np.array([b.close for b in window], dtype=float)
            rets    = np.log(closes[1:] / closes[:-1])
            sum_abs = float(np.abs(rets).sum())
            cur_pl  = abs(float(rets.sum()) / sum_abs) if sum_abs > 0 else 0.0
            if cur_pl <= trade.instrument.pl_mom_exit_pl:
                log.info(
                    f"PL_MOM {trade.instrument.symbol} PL EXIT  "
                    f"cur_pl={cur_pl:.3f} ≤ {trade.instrument.pl_mom_exit_pl}"
                )
                try:
                    n = client.cancel_all_orders(account_id)
                    if n:
                        log.info(f"PL_MOM {trade.instrument.symbol}: cancelled {n} bracket(s) before PL exit")
                except Exception as e:
                    log.warning(f"PL_MOM {trade.instrument.symbol}: pre-PL-exit cancel failed: {e}")
                try:
                    client.close_position(account_id, trade.contract_id)
                except Exception as e:
                    log.error(f"PL_MOM {trade.instrument.symbol}: PL exit close_position failed: {e}")
                    return
                actual_exit = _get_exit_price(client, account_id, trade.entry_ts,
                                              trade.contract_id, now,
                                              entry_price=trade.fill_price or trade.sig.entry)
                exit_price = actual_exit if actual_exit is not None else (
                    state.pl_mom_5s_bars[-1].close if state.pl_mom_5s_bars
                    else (trade.fill_price or trade.sig.entry))
                _log_pl_mom_trade(trade, "PL EXIT", exit_price, now)
                state.active_pl_mom_trade = None
                try:
                    client.cancel_all_orders(account_id)
                except Exception:
                    pass
                return

    # Time exit
    if now >= trade.sig.expires_at():
        log.info(f"PL_MOM {trade.instrument.symbol} max hold reached — closing")
        try:
            client.cancel_all_orders(account_id)
        except Exception as e:
            log.warning(f"PL_MOM {trade.instrument.symbol}: pre-close cancel_all failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"PL_MOM {trade.instrument.symbol}: failed to close position: {e}")
            return
        actual_exit = _get_exit_price(client, account_id, trade.entry_ts,
                                      trade.contract_id, now,
                                      entry_price=trade.fill_price or trade.sig.entry)
        exit_price = actual_exit if actual_exit is not None else (
            state.pl_mom_5s_bars[-1].close if state.pl_mom_5s_bars
            else (trade.fill_price or trade.sig.entry))
        _log_pl_mom_trade(trade, "TIME EXIT", exit_price, now)
        state.active_pl_mom_trade = None
        try:
            client.cancel_all_orders(account_id)
        except Exception:
            pass


# ── Evening Resumption ───────────────────────────────────────────────────────

def _ensure_eve_log():
    EVE_LOG_PATH.parent.mkdir(exist_ok=True)
    if not EVE_LOG_PATH.exists():
        with open(EVE_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=EVE_LOG_FIELDS).writeheader()


def _log_eve_trade(trade: ActiveEveningTrade, outcome: str,
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
        "prev_close":  round(trade.sig.prev_close, 4),
        "gap_pct":     round(trade.sig.gap_pct * 100, 4),
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
    }
    with open(EVE_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=EVE_LOG_FIELDS).writerow(row)
    dir_s = "LONG" if trade.sig.direction == 1 else "SHORT"
    log.info(
        f"EVE LOGGED  {trade.instrument.symbol} {dir_s}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  pnl={pnl_pts:+.2f}pts  "
        f"gap={trade.sig.gap_pct*100:+.3f}%"
    )


def _update_eve_prev_close(state: InstrumentState, now: datetime):
    """Track the most recent RTH close (bar before 16:00 ET) as prev_close."""
    bars = state.vwaslr_bars
    if not bars:
        return
    now_et = now.astimezone(ET)
    today  = now_et.date()
    for b in reversed(bars):
        b_et = b.ts.astimezone(ET)
        if b_et.date() == today and b_et.hour < 16:
            state.eve_prev_close = b.close
            return


def evaluate_evening_resumption(state: InstrumentState,
                                 now: datetime) -> EveningResumeSignal | None:
    """
    Fire once per evening session at 18:00–18:10 ET if the gap from the
    prior RTH close to the first post-gap bar open is ≥ EVE_GAP_THRESH.
    Direction: LONG if gap up, SHORT if gap down.
    """
    now_et = now.astimezone(ET)
    today  = now_et.date()

    # Only evaluate during the first 10 minutes after evening open
    if not ((18, 0) <= (now_et.hour, now_et.minute) < (18, 10)):
        return None

    # One trade per evening session
    if state.eve_fired_date == today:
        return None

    if state.eve_prev_close is None or state.eve_prev_close <= 0:
        return None

    bars = state.vwaslr_bars
    if not bars:
        return None

    # Find the first 1-min bar at or after 18:00 ET today
    evening_start = datetime(today.year, today.month, today.day,
                             18, 0, tzinfo=ET)
    first_bar = next(
        (b for b in bars if b.ts >= evening_start),
        None,
    )
    if first_bar is None:
        return None

    gap_pct = (first_bar.open - state.eve_prev_close) / state.eve_prev_close
    if abs(gap_pct) < EVE_GAP_THRESH:
        return None

    direction = 1 if gap_pct > 0 else -1
    return EveningResumeSignal(
        entry=first_bar.open,
        prev_close=state.eve_prev_close,
        gap_pct=gap_pct,
        bar_ts=first_bar.ts,
        direction=direction,
    )


def place_eve_signal(client: TopstepClient, state: InstrumentState,
                     sig: EveningResumeSignal, account_id: int,
                     paper: bool) -> ActiveEveningTrade:
    inst      = state.instrument
    is_long   = sig.direction == 1
    dir_label = "LONG" if is_long else "SHORT"

    trade = ActiveEveningTrade(
        instrument=inst, contract_id=state.contract_id,
        sig=sig, fired_at=sig.bar_ts,
    )

    if paper:
        log.info(
            f"[PAPER] EVE {inst.symbol} {dir_label}  entry≈{sig.entry:.2f}  "
            f"prev_close={sig.prev_close:.2f}  gap={sig.gap_pct*100:+.3f}%  "
            f"hold={EVE_HOLD_MINUTES}min"
        )
    else:
        order_side = TopstepClient.BID if is_long else TopstepClient.ASK
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=order_side,
            size=1,
            order_type=TopstepClient.ORDER_MARKET,
            custom_tag=f"eve_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"EVE ORDER  {inst.symbol} {dir_label}  order_id={trade.order_id}  "
            f"entry≈{sig.entry:.2f}  prev_close={sig.prev_close:.2f}  "
            f"gap={sig.gap_pct*100:+.3f}%  hold={EVE_HOLD_MINUTES}min"
        )

    state.active_evening_trade = trade
    return trade


def handle_active_evening_trade(client: TopstepClient, state: InstrumentState,
                                 account_id: int, now: datetime, paper: bool):
    trade = state.active_evening_trade

    if paper:
        if now >= trade.expires_at:
            exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                          else trade.sig.entry)
            _log_eve_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_evening_trade = None
        return

    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"EVE {trade.instrument.symbol}: could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"EVE {trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    if pos is None:
        # Position closed unexpectedly (e.g. hit exchange risk limit)
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now)
        exit_price = exit_price or (trade.fill_price or trade.sig.entry)
        _log_eve_trade(trade, "CLOSED", exit_price, now)
        state.active_evening_trade = None
        return

    # Time exit after EVE_HOLD_MINUTES
    if now >= trade.expires_at:
        log.info(f"EVE {trade.instrument.symbol} time exit — closing at market")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"EVE {trade.instrument.symbol}: close_position failed: {e}")
            return
        exit_price = (state.vwaslr_bars[-1].close if state.vwaslr_bars
                      else (trade.fill_price or trade.sig.entry))
        _log_eve_trade(trade, "TIME EXIT", exit_price, now)
        state.active_evening_trade = None


# ── Sunday Open Gap ──────────────────────────────────────────────────────────

def _sun_state_path(symbol: str) -> Path:
    return Path(f"logs/sun_gap_state_{symbol}.json")


def _load_sun_state(state: InstrumentState):
    """Load persisted Friday close and Sunday vol history from disk."""
    p = _sun_state_path(state.instrument.symbol)
    if p.exists():
        try:
            d = json.loads(p.read_text())
            state.sun_fri_close   = d.get("fri_close")
            state.sun_vol_history = d.get("vol_history", [])
            if len(state.sun_vol_history) >= 2:
                state.sun_vol_baseline = float(
                    np.median(state.sun_vol_history[-SUN_VOL_LOOKBACK:]))
        except Exception as e:
            log.debug(f"SUN_GAP {state.instrument.symbol}: could not load state: {e}")


def _save_sun_state(state: InstrumentState):
    """Persist Friday close and Sunday vol history to disk."""
    p = _sun_state_path(state.instrument.symbol)
    p.parent.mkdir(exist_ok=True)
    try:
        p.write_text(json.dumps({
            "fri_close":   state.sun_fri_close,
            "vol_history": state.sun_vol_history[-(SUN_VOL_LOOKBACK * 2):],
        }))
    except Exception as e:
        log.debug(f"SUN_GAP {state.instrument.symbol}: could not save state: {e}")


def _update_sun_fri_close(state: InstrumentState):
    """
    Track the most recent Friday RTH bar close from state.bars.
    Saves to disk whenever the value changes so restarts survive the weekend gap.
    Works in DB mode (bar_collector includes Friday bars in the lookback window
    even when called on Sunday).
    """
    for b in reversed(state.bars):
        b_et = b.ts.astimezone(ET)
        if b_et.weekday() < 5 and b_et.hour < 16:   # Mon–Fri, before RTH close
            if state.sun_fri_close != b.close:
                state.sun_fri_close = b.close
                _save_sun_state(state)
            return


def _ensure_sun_log():
    SUN_LOG_PATH.parent.mkdir(exist_ok=True)
    if not SUN_LOG_PATH.exists():
        with open(SUN_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUN_LOG_FIELDS).writeheader()


def _log_sun_gap_trade(trade: ActiveSundayGapTrade, outcome: str,
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
        "fri_close":   round(trade.sig.fri_close, 4),
        "gap_pct":     round(trade.sig.gap_pct * 100, 4),
        "vol_ratio":   round(trade.sig.vol_ratio, 4),
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
    }
    with open(SUN_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=SUN_LOG_FIELDS).writerow(row)
    dir_s = "LONG" if trade.sig.direction == 1 else "SHORT"
    log.info(
        f"SUN_GAP LOGGED  {trade.instrument.symbol} {dir_s}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  pnl={pnl_pts:+.2f}pts  "
        f"gap={trade.sig.gap_pct*100:+.3f}%  vol={trade.sig.vol_ratio:.2f}×"
    )


def evaluate_sunday_gap(state: InstrumentState, now: datetime) -> SundayGapSignal | None:
    """
    Fire once per Sunday at 18:05–18:15 ET (first 5-min bar complete).
    Gap = (first_bar.open − fri_close) / fri_close.
    Volume filter: first_bar.volume ≥ SUN_VOL_MULT × median of prior Sundays.
    Skips the vol filter if fewer than 2 prior Sunday volumes are recorded.
    """
    now_et = now.astimezone(ET)
    if now_et.weekday() != 6:                                 # Sunday only
        return None
    if not ((18, 5) <= (now_et.hour, now_et.minute) < (18, 15)):
        return None
    if state.sun_fired_date == now_et.date():                 # one trade per Sunday
        return None
    if state.sun_fri_close is None or state.sun_fri_close <= 0:
        return None

    # Find first 5-min bar at or after 18:00 ET today
    sunday_open = datetime(now_et.year, now_et.month, now_et.day, 18, 0, tzinfo=ET)
    first_bar = next((b for b in state.bars if b.ts >= sunday_open), None)
    if first_bar is None:
        return None

    # Record this Sunday's volume for future baselines (once per Sunday)
    today = now_et.date()
    if state.sun_vol_rec_date != today and first_bar.volume > 0:
        state.sun_vol_rec_date = today
        state.sun_vol_history.append(first_bar.volume)
        state.sun_vol_history = state.sun_vol_history[-(SUN_VOL_LOOKBACK * 2):]
        if len(state.sun_vol_history) >= 2:
            state.sun_vol_baseline = float(
                np.median(state.sun_vol_history[-SUN_VOL_LOOKBACK:]))
        _save_sun_state(state)

    gap_pct = (first_bar.open - state.sun_fri_close) / state.sun_fri_close
    if abs(gap_pct) < SUN_GAP_THRESH:
        return None

    # Volume filter: use history BEFORE adding today (already appended above)
    # sun_vol_baseline is updated from the full history including today; use
    # the prior-only baseline by computing from history[:-1] if we just added.
    prior_vols = state.sun_vol_history[:-1]   # exclude today's just-appended entry
    if len(prior_vols) >= 2:
        baseline = float(np.median(prior_vols[-SUN_VOL_LOOKBACK:]))
        vol_ratio = first_bar.volume / baseline if baseline > 0 else 0.0
        if vol_ratio < SUN_VOL_MULT:
            return None
    else:
        vol_ratio = 0.0   # insufficient history — skip vol filter

    direction = 1 if gap_pct > 0 else -1
    return SundayGapSignal(
        entry=first_bar.close,
        fri_close=state.sun_fri_close,
        gap_pct=gap_pct,
        vol_ratio=vol_ratio,
        bar_ts=first_bar.ts,
        direction=direction,
    )


def place_sun_gap_signal(client: TopstepClient, state: InstrumentState,
                          sig: SundayGapSignal, account_id: int,
                          paper: bool) -> ActiveSundayGapTrade:
    inst      = state.instrument
    is_long   = sig.direction == 1
    dir_label = "LONG" if is_long else "SHORT"

    trade = ActiveSundayGapTrade(
        instrument=inst, contract_id=state.contract_id,
        sig=sig, fired_at=sig.bar_ts,
    )

    if paper:
        log.info(
            f"[PAPER] SUN_GAP {inst.symbol} {dir_label}  entry≈{sig.entry:.2f}  "
            f"fri_close={sig.fri_close:.2f}  gap={sig.gap_pct*100:+.3f}%  "
            f"vol={sig.vol_ratio:.2f}×  hold={SUN_HOLD_MINUTES}min"
        )
    else:
        order_side = TopstepClient.BID if is_long else TopstepClient.ASK
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=order_side,
            size=1,
            order_type=TopstepClient.ORDER_MARKET,
            custom_tag=f"sun_{inst.symbol}_{sig.bar_ts.strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"SUN_GAP ORDER  {inst.symbol} {dir_label}  order_id={trade.order_id}  "
            f"entry≈{sig.entry:.2f}  fri_close={sig.fri_close:.2f}  "
            f"gap={sig.gap_pct*100:+.3f}%  vol={sig.vol_ratio:.2f}×  hold={SUN_HOLD_MINUTES}min"
        )

    state.active_sunday_gap_trade = trade
    return trade


def handle_active_sunday_gap_trade(client: TopstepClient, state: InstrumentState,
                                    account_id: int, now: datetime, paper: bool):
    trade = state.active_sunday_gap_trade

    if paper:
        if now >= trade.expires_at:
            exit_price = (state.bars[-1].close if state.bars else trade.sig.entry)
            _log_sun_gap_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_sunday_gap_trade = None
        return

    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"SUN_GAP {trade.instrument.symbol}: could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"SUN_GAP {trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")
        play_trade_sound()

    if pos is None:
        # Position closed unexpectedly (e.g. exchange risk limit)
        exit_price = _get_exit_price(client, account_id, trade.fired_at,
                                     trade.contract_id, now)
        exit_price = exit_price or (trade.fill_price or trade.sig.entry)
        _log_sun_gap_trade(trade, "CLOSED", exit_price, now)
        state.active_sunday_gap_trade = None
        return

    # Time exit after SUN_HOLD_MINUTES
    if now >= trade.expires_at:
        log.info(f"SUN_GAP {trade.instrument.symbol} time exit — closing at market")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"SUN_GAP {trade.instrument.symbol}: close_position failed: {e}")
            return
        exit_price = (state.bars[-1].close if state.bars
                      else (trade.fill_price or trade.sig.entry))
        _log_sun_gap_trade(trade, "TIME EXIT", exit_price, now)
        state.active_sunday_gap_trade = None


# ── Main loop ────────────────────────────────────────────────────────────────

def run(account_id: int | None, paper: bool):
    client = TopstepClient()
    client.use_shared_token()  # reuse bar_collector's token — avoids multiple-sessions disconnect

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
    _ensure_slr_log()
    _ensure_eve_log()
    _ensure_sun_log()
    _ensure_pl_mom_log()
    _ensure_dom_signal_log()
    for state in states:
        if state.instrument.sun_gap_enabled:
            _load_sun_state(state)
            log.info(
                f"SUN_GAP {state.instrument.symbol}: "
                f"fri_close={state.sun_fri_close}  "
                f"vol_history={len(state.sun_vol_history)} entries"
            )
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
                if state.instrument.sun_gap_enabled:
                    _update_sun_fri_close(state)

                if state.instrument.vwaslr_n > 0 or state.instrument.slr_enabled or state.instrument.eve_enabled:
                    cur_min = now.minute
                    _expected_1min = datetime.fromtimestamp(
                        (int(now.timestamp()) // 60) * 60 - 60, tz=timezone.utc)
                    _vwas_stale = (
                        bool(state.vwaslr_bars)
                        and state.vwaslr_bars[-1].ts < _expected_1min
                        and state.vwaslr_last_fetch is not None
                        and (now - state.vwaslr_last_fetch).total_seconds() >= 15
                    )
                    if cur_min != state.vwaslr_fetch_min or _vwas_stale:
                        fetch_vwaslr_bars(client, state)
                        state.vwaslr_last_fetch = now
                        state.vwaslr_fetch_min = cur_min
                    if state.instrument.vwaslr_n > 0:
                        _update_vwaslr_ema(state)
                    if state.instrument.eve_enabled:
                        _update_eve_prev_close(state, now)

                if state.active_trade:
                    handle_active_trade(client, state, account_id, now, paper)

                if state.active_orb_trade:
                    handle_active_orb_trade(client, state, account_id, now, paper)

                if state.active_vwaslr_trade:
                    handle_active_vwaslr_trade(client, state, account_id, now, paper)

                if state.active_slr_trade:
                    handle_active_slr_trade(client, state, account_id, now, paper)

                if state.active_evening_trade:
                    handle_active_evening_trade(client, state, account_id, now, paper)

                if state.active_sunday_gap_trade:
                    handle_active_sunday_gap_trade(client, state, account_id, now, paper)

                # PL_Mom: fetch 5s bars when trade active (needed for PL exit check)
                if state.active_pl_mom_trade and state.instrument.pl_mom_enabled:
                    fetch_pl_mom_bars(client, state)

                if state.active_pl_mom_trade:
                    handle_active_pl_mom_trade(client, state, account_id, now, paper)

                # Only enter new trades when no position is open on this instrument
                no_position = (not state.active_trade
                               and not state.active_orb_trade
                               and not state.active_vwaslr_trade
                               and not state.active_slr_trade
                               and not state.active_pl_mom_trade
                               and not state.active_evening_trade
                               and not state.active_sunday_gap_trade)
                last_bar_ts = state.bars[-1].ts if state.bars else None

                # Don't enter new trades after TopstepX daily cutoff (RTH only)
                now_ct  = now.astimezone(CT)
                now_et  = now.astimezone(ET)
                past_cutoff = (now_ct.hour, now_ct.minute) >= TRADING_CUTOFF_CT
                # SLR is also active during Globex — cutoff only applies in RTH session
                now_et_hm        = (now_et.hour, now_et.minute)
                slr_in_rth       = (9, 30) <= now_et_hm < (16, 0)
                slr_past_cutoff  = past_cutoff and slr_in_rth

                if no_position and not past_cutoff:
                    sig = evaluate(state)
                    if sig and last_bar_ts != state.last_evaluated_ts:
                        state.last_evaluated_ts = last_bar_ts
                        pl = fetch_1min_pl(client, state.contract_id,
                                           sig["bar_ts"], sig["direction"],
                                           symbol=state.instrument.symbol)
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
                        # API-side guard: confirm no open position before entering
                        # (in-memory state can be stale after a transient API failure)
                        _api_open = False
                        if not paper:
                            try:
                                _live_pos = client.get_open_positions(account_id)
                                _api_open = any(
                                    p.get("contractId") == state.contract_id
                                    for p in _live_pos
                                )
                            except Exception:
                                pass
                        if _api_open:
                            log.warning(
                                f"VWASLR {state.instrument.symbol}: "
                                f"open position detected via API — skipping duplicate entry"
                            )
                        else:
                            place_vwaslr_signal(client, state, vwas_sig, account_id, paper)

                slr_in_blackout = any(
                    _in_blackout(now_et_hm, sh, sm, eh, em)
                    for sh, sm, eh, em, _cond in state.instrument.blackout_windows
                )
                if no_position and not slr_past_cutoff and not slr_in_blackout and state.instrument.slr_enabled:
                    slr_bar_ts = state.vwaslr_bars[-1].ts if state.vwaslr_bars else None
                    if slr_bar_ts != state.slr_last_bar_ts:
                        state.slr_last_bar_ts = slr_bar_ts
                        slr_sig = evaluate_slr_scalp(state)
                        if slr_sig:
                            state.slr_last_surge_ts = slr_sig.surge_ts
                            place_slr_signal(client, state, slr_sig, account_id, paper)

                if no_position and state.instrument.eve_enabled:
                    eve_sig = evaluate_evening_resumption(state, now)
                    if eve_sig:
                        state.eve_fired_date = eve_sig.bar_ts.astimezone(ET).date()
                        place_eve_signal(client, state, eve_sig, account_id, paper)

                if no_position and state.instrument.sun_gap_enabled:
                    sun_sig = evaluate_sunday_gap(state, now)
                    if sun_sig:
                        state.sun_fired_date = sun_sig.bar_ts.astimezone(ET).date()
                        place_sun_gap_signal(client, state, sun_sig, account_id, paper)

                # PL_Mom entry: RTH only, new 5s bar gate
                # Blackout 9:30–9:40 ET: first 10 min are high-volatility churn with negative edge
                if no_position and not past_cutoff and state.instrument.pl_mom_enabled:
                    now_et_hm_chk = (now_et.hour, now_et.minute)
                    if (9, 40) <= now_et_hm_chk < (16, 0):
                        fetch_pl_mom_bars(client, state)
                        pl_mom_bar_ts = (state.pl_mom_5s_bars[-1].ts
                                         if state.pl_mom_5s_bars else None)
                        if pl_mom_bar_ts != state.pl_mom_last_bar_ts:
                            state.pl_mom_last_bar_ts = pl_mom_bar_ts
                            pl_mom_sig = evaluate_pl_mom(state)
                            if pl_mom_sig:
                                place_pl_mom_signal(client, state, pl_mom_sig,
                                                    account_id, paper, now)

            except Exception as e:
                log.error(f"{state.instrument.symbol}: {e}", exc_info=True)

        # Poll interval: 5s during RTH (PL_MOM entry cadence), 10s when PL_MOM
        # trade is active, 20s during ORB window, 40s otherwise
        now_et = datetime.now(ET)
        hm = (now_et.hour, now_et.minute)
        in_rth          = (9, 30) <= hm < (16, 0)
        in_orb_window   = (9, 30) <= hm < (10, 30)
        any_pl_active   = any(s.active_pl_mom_trade for s in states)
        any_pl_enabled  = any(s.instrument.pl_mom_enabled for s in states)
        if any_pl_active:
            sleep_s = 10
        elif in_rth and any_pl_enabled:
            sleep_s = POLL_SECONDS_RTH
        elif in_orb_window:
            sleep_s = POLL_SECONDS_ORB
        else:
            sleep_s = POLL_SECONDS
        time.sleep(sleep_s)


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

    parser = argparse.ArgumentParser(description="MES/MNQ 3σ continuation trading bot")
    parser.add_argument("--paper", action="store_true",
                        help="Detect signals and log them but place no real orders")
    parser.add_argument("--account", type=int, default=None,
                        help="TopstepX account ID (auto-detects first active account if omitted)")
    args = parser.parse_args()

    run(account_id=args.account, paper=args.paper)
