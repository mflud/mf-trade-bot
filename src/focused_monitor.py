"""
focused_monitor.py — Monitor for the focused bot (BA-BRK, BA-REV).

ORB and ORB-cls panels removed 2026-10-02 — corrected (bar.close entry)
backtests showed no validated edge for either, even after width/stop
tuning (see project_orb_cls_strategy memory). The underlying ORB/ORB-cls
code (evaluate_orb/_cls, panel builders, log-sync) is left in place but
unused, in case it's revisited.

Layout:
  ┌──────────────────┬───────────────────────────┐
  │ BA-BRK            │ Trade Summary             │
  │ BA-REV            │ Positions                 │
  └──────────────────┴───────────────────────────┘

Usage:
    python src/focused_monitor.py
"""

import json
import re
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, "src")
load_dotenv()

from topstep_client import TopstepClient, get_bars_from_db, get_5s_bars_from_db, bars_db_available
from trade_summary_panel import build_trade_summary_panel
from wall_tracker import WallTracker, log_wall_events, ensure_wall_log

ET    = ZoneInfo("America/New_York")
LOCAL = datetime.now().astimezone().tzinfo

SYMBOL      = "MES"
POINT_VALUE = 5.0

# ── VWASLR constants (mirrors trading_bot.py) ─────────────────────────────────
VWASLR_SIGMA_BARS = 500
VWASLR_N          = 50
VWASLR_EMA_SPAN   = 10
VWASLR_THRESHOLD  = 0.4
VWASLR_STOP_S     = 2.0
VWASLR_TARGET_S   = 3.0
VWASLR_BARS_FETCH = 580   # VWASLR_SIGMA_BARS + VWASLR_N + headroom

# ── PL_REV constants (mirrors trading_bot.py) ─────────────────────────────────
PL_WINDOW       = 6      # 5s bars = 30s
PL_ENTRY_PL     = 0.60
PL_MOVE_BPS     = 25.0
PL_TP_BPS       = 24.0
PL_STOP_BPS     = 15.0
PL_RESUME_PL    = 0.70
PL_MAX_HOLD_S   = 120
PL_MIN_HOLD_S   = 10
PL_5S_FETCH     = 130
PL_SIGMA_N      = 3.0
PL_SIGMA_LB     = 120
PL_HISTORY      = 10

# ── ORB constants — MNQ 1-min ORB ───────────────────────────────────────────────
ORB_MNQ_SYMBOL    = "MNQ"
ORB_MIN           = 1        # opening range = first 1-min bar (9:30 ET)
ORB_ENTRY_WIN     = 5        # minutes after ORB close to look for breakout
ORB_MNQ_WIDTH_MAX = 0.003    # ≤ 30bps
ORB_TGT_MULT      = 1.0      # 1× width
ORB_HOLD_MIN      = 10       # force-exit after 10 min (~9:40 ET)
ORB_MNQ_MAX_LOSS  = 500.0    # $500 stop cap (far-side stop)
ORB_MNQ_PV        = 2.0      # MNQ point value
ORB_BARS_FETCH    = 200

# ── ORB constants — MES 1-min ORB ───────────────────────────────────────────────
ORB_MES_SYMBOL    = "MES"
ORB_MES_WIDTH_MAX = 0.0      # no width cap
ORB_MES_MAX_LOSS  = 0.0      # no dollar cap (midpoint stop is naturally small)
ORB_MES_PV        = 5.0      # MES point value
# Compatibility aliases used by existing _update_orb / build_orb_panel
ORB_SYMBOL    = ORB_MNQ_SYMBOL
ORB_WIDTH_MAX = ORB_MNQ_WIDTH_MAX
ORB_MAX_LOSS  = ORB_MNQ_MAX_LOSS
ORB_PV        = ORB_MNQ_PV

# ── ORB-cls constants (mirrors trading_bot.py's BotInstrument.orb_cls_*) ───────
# Closing-range breakout: range = the 15:50 ET 1-min bar instead of 9:30.
# No cross-confirmation (see project_orb_cls_strategy memory) — each
# instrument fires independently, no "awaiting other leg" phase.
ORB_CLS_RANGE_HM      = (15, 50)   # ET — range bar start
ORB_CLS_TGT_MULT      = 3.25
ORB_CLS_HOLD_MIN      = 15
ORB_CLS_SAFETY_HM     = (16, 8)    # hard force-flat, 2min inside TopstepX's 16:10 ET cutoff
ORB_CLS_MES_ENTRY_WIN = 5
ORB_CLS_MNQ_ENTRY_WIN = 3

# ── DOM constants ─────────────────────────────────────────────────────────────
WALL_MULT     = 2.5
DOM_DB_PATH   = Path("data/dom.db")
DOM_DB_STALE  = 10
DOM_NEAR      = 4
DOM_BUCKET_PT = 1.0
DOM_BUCKET_N  = 10
DOM_NEAR_CAP  = 100
DOM_BKT_CAP   = 500
DOM_PRICE_COL = 10
DOM_NEAR_BAR  = 16
DOM_BKT_BAR   = 14

# ── Settlement ────────────────────────────────────────────────────────────────
SETTLE_UTC_START = 21
SETTLE_UTC_END   = 22

# ── Alert ─────────────────────────────────────────────────────────────────────
ALERT          = "/System/Library/Sounds/Pop.aiff"
ALERT_COOLDOWN = 15.0
_last_alert    = 0.0


def play_alert():
    pass  # alerts disabled


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts: datetime; open: float; high: float; low: float; close: float; volume: float


@dataclass
class VwaslrSignal:
    direction: int   # +1 LONG, -1 SHORT
    entry:     float
    sigma_pts: float
    ema:       float
    fired_at:  datetime
    expires_at: datetime


@dataclass
class PLRevSignal:
    direction: int
    entry:     float
    pl:        float
    move_bps:  float
    bar_ts:    datetime
    def expires_at(self): return self.bar_ts + timedelta(seconds=PL_MAX_HOLD_S)


WALL_STOP_PTS   = 4.0
WALL_TARGET_PTS = 12.0
WALL_HOLD_MIN   = 15

@dataclass
class WallBreakSignal:
    direction:  int      # +1 LONG (ask wall broken), -1 SHORT (bid wall broken)
    wall_price: float
    entry:      float
    target:     float
    stop:       float
    test_count: int
    peak_size:  float
    fired_at:   datetime
    expires_at: datetime


# BA-BRK (Bid/Ask Break — wall cascade): mirrors trading_bot.py's live logic
# (see BA_BRK_* constants there for the backtest this is based on).
BA_BRK_CASCADE_MIN = 3
BA_BRK_MAX_GAP_SEC = 45
BA_BRK_STOP_PTS    = 3.5
BA_BRK_TARGET_PTS  = 12.0
BA_BRK_HOLD_MIN    = 25
# Bid side, enabled live 2026-09-14 (own alignment filter + params, see trading_bot.py)
BA_BRK_BID_STOP_PTS   = 4.0
BA_BRK_BID_TARGET_PTS = 18.0
BA_BRK_BID_HOLD_MIN   = 30
BA_BRK_BUFFER_MIN  = 10  # display buffer only — cascade detection itself only cares about
                          # gaps within BA_BRK_MAX_GAP_SEC, so this just controls how much
                          # history the panel's activity table can show

@dataclass
class BaBrkSignal:
    direction:      int      # +1 LONG (ask cascade), -1 SHORT (bid cascade)
    side:           str      # "ask" | "bid"
    entry:          float
    target:         float
    stop:           float
    cascade_len:    int
    first_break_ts: datetime
    last_break_ts:  datetime
    fired_at:       datetime
    expires_at:     datetime


# BA-REV (Bid/Ask Reversal — wall rejection): mirrors trading_bot.py's live
# logic (see BA_REV_* constants there). Opposite premise from BA-BRK: bet a
# wall that survives BA_REV_TEST_COUNT touches HOLDS, not breaks. Redesigned
# 2026-09-15: free 09:40-10:30 ET, then only while still contained within the
# 07:00-10:30 ET reference range (one breakout disables the rest of the day).
BA_REV_TEST_COUNT       = 3
BA_REV_STOP_PTS         = 2.0   # stepped up from backtest-optimal 1.0pt — see trading_bot.py
BA_REV_TARGET_PTS       = 6.0
BA_REV_HOLD_MIN         = 60
BA_REV_REF_START_HM     = 7 * 60          # 07:00 ET — reference window start
BA_REV_LOCK_HM          = 10 * 60 + 30    # 10:30 ET — reference locks; trading starts here (no free window)
BA_REV_TRADE_START_HM   = BA_REV_LOCK_HM
BA_REV_SIGNAL_STALE_SEC = 60
BA_REV_RANGE_LOOKBACK_BARS = 2500

@dataclass
class BaRevSignal:
    direction:   int      # +1 LONG (bid rejection), -1 SHORT (ask rejection)
    side:        str      # "ask" | "bid" — which wall rejected
    entry:       float
    target:      float
    stop:        float
    test_count:  int
    wall_price:  float
    trigger_ts:  datetime
    fired_at:    datetime
    expires_at:  datetime


@dataclass
class ORBState:
    orb_high:    float = 0.0
    orb_low:     float = 0.0
    orb_width:   float = 0.0
    orb_mid:     float = 0.0
    valid:       bool  = False    # width passes filter
    entry_price: float = 0.0
    direction:   int   = 0       # +1 LONG, -1 SHORT candidate; 0 = no breakout yet
    target:      float = 0.0
    stop:        float = 0.0
    session_date: "object" = None   # date object
    # Cross-confirmation display (mirrors trading_bot.py's live gate — see
    # evaluate_orb's orb_cross_confirm): `direction` is set the moment this
    # instrument's own range breaks, same as before, but that's only a
    # candidate now. `confirmed` tracks whether the OTHER instrument's own
    # candidate has matched direction — only then did a real trade fire live.
    confirmed:   bool  = False


@dataclass
class DOMBook:
    bids: dict = field(default_factory=dict)
    asks: dict = field(default_factory=dict)
    last_price: "float|None" = None
    best_bid:   "float|None" = None
    best_ask:   "float|None" = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    BID = 4; ASK = 3; RESET = 6

    def apply_depth(self, updates):
        with self._lock:
            for u in updates:
                price = float(u.get("price", 0)); volume = float(u.get("volume", 0))
                t = u.get("type", -1)
                if t == self.RESET:      self.bids.clear(); self.asks.clear()
                elif t == self.BID:
                    if volume == 0: self.bids.pop(price, None)
                    else:           self.bids[price] = volume
                elif t == self.ASK:
                    if volume == 0: self.asks.pop(price, None)
                    else:           self.asks[price] = volume
            self.last_update = datetime.now(timezone.utc)

    def hybrid_snapshot(self, near, bucket_pts, bucket_n):
        with self._lock:
            bb, ba = self.best_bid, self.best_ask
            all_bids = sorted(((p, s) for p, s in self.bids.items()
                               if bb is None or ba is None or p < ba), reverse=True)
            all_asks = sorted((p, s) for p, s in self.asks.items()
                              if bb is None or ba is None or p > bb)
            last, updated = self.last_price, self.last_update

        def build_side(levels, is_ask):
            rows = []
            for p, s in levels[:near]:
                rows.append((p, s, False))
            while len(rows) < near:
                rows.append((None, 0.0, False))
            far = levels[near:]
            if far:
                grid_start = (far[0][0] // bucket_pts) * bucket_pts
                buckets: dict = {}
                for p, s in far:
                    idx = int((p - grid_start) // bucket_pts)
                    buckets[idx] = buckets.get(idx, 0.0) + s
                for k in (sorted(buckets) if is_ask else sorted(buckets, reverse=True))[:bucket_n]:
                    lo = grid_start + k * bucket_pts
                    rows.append((f"{lo:.0f}–{lo+bucket_pts:.0f}", buckets[k], True))
            while len(rows) < near + bucket_n:
                rows.append((None, 0.0, True))
            return rows

        return (build_side(all_bids, False), build_side(all_asks, True),
                last, bb, ba, updated)


@dataclass
class MonitorState:
    contract_id:     str
    mnq_contract_id: str = ""
    bars_1m:     list = field(default_factory=list)   # MES 1-min bars for VWASLR + sizing
    bars_orb:    list = field(default_factory=list)   # MNQ 1-min bars for ORB
    bars_mes_orb: list = field(default_factory=list)  # MES 1-min bars for MES ORB
    bars_5s:     list = field(default_factory=list)   # MES 5s bars for PL_REV
    bars_5s_mnq: list = field(default_factory=list)   # MNQ 5s bars for PL_REV
    dom:         DOMBook = field(default_factory=DOMBook)
    wall_tracker: "WallTracker|None" = None
    # VWASLR
    vwaslr_ema:       float = 0.0
    vwaslr_ema_prev:  float = 0.0
    vwaslr_sigma_pts: float = 0.0
    vwaslr_signal:    "VwaslrSignal|None" = None
    vwaslr_history:   list = field(default_factory=list)   # (ts, ema, fired)
    # PL_REV — MES
    pl_sig:         "PLRevSignal|None" = None
    pl_entry_ts:    "datetime|None"   = None
    pl_last_bar_ts: "datetime|None"   = None
    pl_history:     list = field(default_factory=list)    # (ts, pl, dir_sym, move_bps)
    sigma_30s_bps:  float = 0.0
    # PL_REV — MNQ
    pl_sig_mnq:         "PLRevSignal|None" = None
    pl_entry_ts_mnq:    "datetime|None"   = None
    pl_last_bar_ts_mnq: "datetime|None"   = None
    pl_history_mnq:     list = field(default_factory=list)
    sigma_30s_bps_mnq:  float = 0.0
    # ORB
    orb:     ORBState = field(default_factory=ORBState)   # MNQ ORB
    orb_mes: ORBState = field(default_factory=ORBState)   # MES ORB
    # ORB-cls (15:50 ET closing range) — separate session state, no cross-confirm
    orb_cls:     ORBState = field(default_factory=ORBState)   # MNQ ORB-cls
    orb_cls_mes: ORBState = field(default_factory=ORBState)   # MES ORB-cls
    # Wall Break
    wall_break_signal: "WallBreakSignal|None" = None
    wall_recent_events: list = field(default_factory=list)  # (ts, event, side, wall_price, test_count, entry, peak_size)
    # BA-BRK (wall cascade)
    ba_brk_signal:    "BaBrkSignal|None" = None
    ba_brk_breakouts: list = field(default_factory=list)   # raw WallEvent breakouts, both sides
    day_open:         "float|None" = None
    day_open_date:    "object|None" = None   # date this day_open was computed for — set once/day
    # BA-REV (wall rejection reversal) — reuses ba_brk_breakouts above for its
    # "test" events (same buffer already holds both event types)
    ba_rev_signal:     "BaRevSignal|None" = None
    ba_rev_ref_day:    "object|None" = None
    ba_rev_ref_high:   "float|None" = None
    ba_rev_ref_low:    "float|None" = None
    ba_rev_contained:  bool = True
    # Position — MES
    position_size:  int   = 0
    position_dir:   int   = 0
    position_entry: "float|None" = None
    position_strat: str   = ""
    # Position — MNQ
    mnq_position_size:  int   = 0
    mnq_position_dir:   int   = 0
    mnq_position_entry: "float|None" = None
    mnq_position_strat: str   = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _in_settlement(ts: datetime) -> bool:
    return SETTLE_UTC_START <= ts.astimezone(timezone.utc).hour < SETTLE_UTC_END


def _compute_sigma_5s(bars: list, lookback: int) -> float:
    closes = np.array([b.close for b in bars], dtype=float)
    if len(closes) < PL_WINDOW + 2:
        return 0.0
    rets  = np.log(closes[1:] / closes[:-1])
    n     = len(rets)
    lb    = min(lookback, n - PL_WINDOW + 1)
    moves = [abs(rets[i:i+PL_WINDOW].sum())
             for i in range(n - lb, n - PL_WINDOW + 1) if i >= 0]
    return float(np.std(moves)) * 10000 if len(moves) >= 4 else 0.0


def _day_open_price(bars: list, now: datetime) -> float | None:
    """First bar's open on `now`'s local calendar day, or None if no bar yet today."""
    today = now.astimezone(LOCAL).date()
    for bar in bars:
        if bar.ts.astimezone(LOCAL).date() == today:
            return bar.open
    return None


def _detect_cascade_signal(events: list, side: str, cascade_min: int, max_gap_sec: int,
                            after_ts: "datetime | None"):
    """Mirrors trading_bot.py's _detect_cascade_signal — chains consecutive
    same-side breakouts less than max_gap_sec apart; returns
    (trigger_event, cascade_start_ts, cascade_len) for the most recent fresh
    cascade past cascade_min, or None."""
    same_side = sorted((e for e in events if e.side == side), key=lambda e: e.ts)
    cascade_count = 0
    cascade_start = None
    last_ts = None
    signaled_this_cascade = False
    result = None
    for e in same_side:
        if last_ts is not None and (e.ts - last_ts).total_seconds() <= max_gap_sec:
            cascade_count += 1
        else:
            cascade_count = 1
            cascade_start = e.ts
            signaled_this_cascade = False
        last_ts = e.ts
        if cascade_count >= cascade_min and not signaled_this_cascade:
            signaled_this_cascade = True
            if after_ts is None or e.ts > after_ts:
                result = (e, cascade_start, cascade_count)
    return result


def _find_test_signal(events: list, side: str, test_count: int,
                       after_ts: "datetime | None"):
    """Mirrors trading_bot.py's _find_test_signal — most recent WallEvent
    matching event=='test', the given side, and exactly `test_count`."""
    matches = [e for e in events
               if e.side == side and e.event == "test" and e.test_count == test_count
               and (after_ts is None or e.ts > after_ts)]
    return max(matches, key=lambda e: e.ts) if matches else None


def _update_ba_rev_reference(state: "MonitorState", now: datetime):
    """Mirrors trading_bot.py's _update_ba_rev_reference — display-only copy.
    Locks the 07:00-10:30 ET reference range once per day, right at 10:30 ET."""
    now_et = now.astimezone(ET)
    today  = now_et.date()
    if state.ba_rev_ref_day == today:
        return
    lock_et = datetime.combine(today, dtime(BA_REV_LOCK_HM // 60, BA_REV_LOCK_HM % 60), tzinfo=ET)
    if now_et < lock_et:
        return

    state.ba_rev_ref_day   = today
    state.ba_rev_ref_high  = None
    state.ba_rev_ref_low   = None
    state.ba_rev_contained = True

    raw = get_bars_from_db(SYMBOL, 1, BA_REV_RANGE_LOOKBACK_BARS)
    if not raw:
        return

    ref_start_et = datetime.combine(today, dtime(BA_REV_REF_START_HM // 60, BA_REV_REF_START_HM % 60), tzinfo=ET)
    ref_hi = ref_lo = None
    for b in raw:
        ts_et = datetime.fromisoformat(b["t"]).astimezone(ET)
        if ref_start_et <= ts_et < lock_et:
            ref_hi = b["h"] if ref_hi is None else max(ref_hi, b["h"])
            ref_lo = b["l"] if ref_lo is None else min(ref_lo, b["l"])

    if ref_hi is None:
        return
    state.ba_rev_ref_high = ref_hi
    state.ba_rev_ref_low  = ref_lo


def _update_ba_rev_containment(state: "MonitorState", bars_1m: list):
    """After the reference locks, one bar breaking [ref_low, ref_high]
    permanently disables containment for the rest of the day."""
    if state.ba_rev_ref_high is None or not state.ba_rev_contained:
        return
    if not bars_1m:
        return
    last = bars_1m[-1]
    if last.high > state.ba_rev_ref_high or last.low < state.ba_rev_ref_low:
        state.ba_rev_contained = False


def _pl_bar(pl: float, dir_sym: str, half: int = 6) -> str:
    filled = max(0, min(half, round(min(1.0, pl) * half)))
    empty  = half - filled
    if dir_sym == "▲":
        return f"{'░'*half}|[green]{'█'*filled}{'░'*empty}[/]"
    else:
        return f"[red]{'░'*empty}{'█'*filled}[/]|{'░'*half}"


# ─── Bar fetchers ─────────────────────────────────────────────────────────────

def fetch_1min_bars(client: TopstepClient, state: MonitorState):
    now_utc   = datetime.now(timezone.utc)
    now_floor = datetime.fromtimestamp(
        (int(now_utc.timestamp()) // 60) * 60, tz=timezone.utc)
    db_fresh = False
    if bars_db_available():
        raw = get_bars_from_db(SYMBOL, 1, VWASLR_BARS_FETCH + 1)
        db_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                       open=b["o"], high=b["h"], low=b["l"],
                       close=b["c"], volume=b["v"])
                   for b in raw
                   if datetime.fromisoformat(b["t"]) < now_floor]
        if db_bars:
            if not state.bars_1m or db_bars[-1].ts >= state.bars_1m[-1].ts:
                state.bars_1m = db_bars
            db_fresh = (now_utc - db_bars[-1].ts).total_seconds() < 180
    if not db_fresh:
        try:
            end   = now_utc
            start = end - timedelta(minutes=VWASLR_BARS_FETCH + 30)
            raw   = client.get_bars(contract_id=state.contract_id, start=start, end=end,
                                    unit=TopstepClient.MINUTE, unit_number=1,
                                    limit=VWASLR_BARS_FETCH)
            state.bars_1m = [Bar(ts=datetime.fromisoformat(b["t"]),
                                  open=b["o"], high=b["h"], low=b["l"],
                                  close=b["c"], volume=b["v"])
                              for b in reversed(raw)]
        except Exception:
            pass


# ─── MNQ bar fetcher (for ORB) ────────────────────────────────────────────────

def _orb_session_cutoff_utc() -> str:
    """09:00 ET today, as an ISO UTC string — the lower bound for ORB bar
    fetches. Using a time cutoff (not a row-count LIMIT) so a mid-day
    restart of this monitor doesn't lose the 9:30 ET morning-ORB range:
    a LIMIT 200 fetch only reaches back ~3.3hrs, which silently stops
    covering 9:30 ET by early afternoon and leaves the morning ORB panel
    stuck on LOADING for the rest of the day."""
    session_start = datetime.now(ET).replace(hour=9, minute=0, second=0, microsecond=0)
    return session_start.astimezone(timezone.utc).isoformat()


def fetch_mnq_bars(state: MonitorState):
    """Load MNQ 1-min bars from bars.db for today's ORB tracking (09:00 ET onward)."""
    try:
        import sqlite3
        conn = sqlite3.connect("data/bars.db", timeout=1.0)
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE symbol='MNQ' AND minutes=1 AND ts >= ? ORDER BY ts LIMIT 600",
            (_orb_session_cutoff_utc(),)
        ).fetchall()
        conn.close()
        if rows:
            state.bars_orb = [
                Bar(ts=datetime.fromisoformat(r[0]),
                    open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5])
                for r in rows
            ]
    except Exception:
        pass


def fetch_mes_orb_bars(state: MonitorState):
    """Load MES 1-min bars from bars.db for today's MES ORB tracking (09:00 ET onward)."""
    try:
        import sqlite3
        conn = sqlite3.connect("data/bars.db", timeout=1.0)
        rows = conn.execute(
            "SELECT ts, open, high, low, close, volume FROM bars "
            "WHERE symbol='MES' AND minutes=1 AND ts >= ? ORDER BY ts LIMIT 600",
            (_orb_session_cutoff_utc(),)
        ).fetchall()
        conn.close()
        if rows:
            state.bars_mes_orb = [
                Bar(ts=datetime.fromisoformat(r[0]),
                    open=r[1], high=r[2], low=r[3], close=r[4], volume=r[5])
                for r in rows
            ]
    except Exception:
        pass


def _update_mes_orb(state: MonitorState):
    """Update MES ORB state — midpoint stop, no width cap."""
    bars = state.bars_mes_orb if state.bars_mes_orb else state.bars_1m
    if not bars:
        return
    now_et = datetime.now(ET)
    today  = now_et.date()
    orb    = state.orb_mes

    if orb.session_date != today:
        orb.__init__()
        orb.session_date = today

    orb_end_hm   = 9 * 60 + 30 + ORB_MIN   # 9:31 ET
    entry_end_hm = orb_end_hm + ORB_ENTRY_WIN

    today_rth = [b for b in bars
                 if b.ts.astimezone(ET).date() == today
                 and 9*60+30 <= b.ts.astimezone(ET).hour*60 + b.ts.astimezone(ET).minute < 16*60]
    if not today_rth:
        return

    orb_bars = [b for b in today_rth
                if b.ts.astimezone(ET).hour*60 + b.ts.astimezone(ET).minute < orb_end_hm]
    if orb_bars:
        orb.orb_high  = max(b.high  for b in orb_bars)
        orb.orb_low   = min(b.low   for b in orb_bars)
        orb.orb_mid   = (orb.orb_high + orb.orb_low) / 2
        orb.orb_width = orb.orb_high - orb.orb_low
        orb.valid     = True   # no width filter for MES

    if orb.valid and orb.direction == 0:
        after_orb = [b for b in today_rth
                     if orb_end_hm <= b.ts.astimezone(ET).hour*60 + b.ts.astimezone(ET).minute < entry_end_hm]
        for b in after_orb:
            w = orb.orb_width
            if b.close > orb.orb_high:
                orb.direction   = 1
                orb.entry_price = b.close
                orb.target      = b.close + w * ORB_TGT_MULT
                orb.stop        = b.close - w / 2.0   # midpoint stop
                break
            if b.close < orb.orb_low:
                orb.direction   = -1
                orb.entry_price = b.close
                orb.target      = b.close - w * ORB_TGT_MULT
                orb.stop        = b.close + w / 2.0   # midpoint stop
                break

    # Cross-confirmation: re-checked every call so it can flip True as soon
    # as MNQ's own candidate catches up (mirrors trading_bot.py's live gate).
    if orb.direction != 0 and not orb.confirmed:
        if state.orb.direction == orb.direction:
            orb.confirmed = True


def _update_orb_cls_generic(orb: ORBState, bars: list, entry_win: int):
    """Shared ORB-cls logic for one instrument — range = the 15:50 ET 1-min
    bar, midpoint stop, no cross-confirmation (mirrors evaluate_orb_cls in
    trading_bot.py). `confirmed` is left False here; _sync_orb_cls_from_log
    sets it once the real order log shows a fill, same pattern as the
    morning ORB panels."""
    if not bars:
        return
    now_et = datetime.now(ET)
    today  = now_et.date()

    if orb.session_date != today:
        orb.__init__()
        orb.session_date = today

    range_hm     = ORB_CLS_RANGE_HM[0] * 60 + ORB_CLS_RANGE_HM[1]
    range_end_hm = range_hm + 1              # the range bar is 1 minute wide
    entry_end_hm = range_end_hm + entry_win

    today_bars = [b for b in bars
                  if b.ts.astimezone(ET).date() == today
                  and range_hm <= b.ts.astimezone(ET).hour * 60 + b.ts.astimezone(ET).minute < 16 * 60]
    if not today_bars:
        return

    range_bar = next((b for b in today_bars
                      if b.ts.astimezone(ET).hour * 60 + b.ts.astimezone(ET).minute == range_hm), None)
    if range_bar:
        orb.orb_high  = range_bar.high
        orb.orb_low   = range_bar.low
        orb.orb_mid   = (orb.orb_high + orb.orb_low) / 2
        orb.orb_width = orb.orb_high - orb.orb_low
        orb.valid     = True   # no width filter for ORB-cls

    if orb.valid and orb.direction == 0:
        after_range = [b for b in today_bars
                       if range_end_hm <= b.ts.astimezone(ET).hour * 60 + b.ts.astimezone(ET).minute < entry_end_hm]
        for b in after_range:
            w = orb.orb_width
            if b.close > orb.orb_high:
                orb.direction   = 1
                orb.entry_price = b.close
                orb.target      = b.close + w * ORB_CLS_TGT_MULT
                orb.stop        = b.close - w / 2.0   # midpoint stop
                break
            if b.close < orb.orb_low:
                orb.direction   = -1
                orb.entry_price = b.close
                orb.target      = b.close - w * ORB_CLS_TGT_MULT
                orb.stop        = b.close + w / 2.0   # midpoint stop
                break


def _update_orb_cls(state: MonitorState):
    """MNQ ORB-cls — reuses the same MNQ 1-min bars fetched for the morning ORB."""
    bars = state.bars_orb if state.bars_orb else state.bars_1m
    _update_orb_cls_generic(state.orb_cls, bars, ORB_CLS_MNQ_ENTRY_WIN)


def _update_mes_orb_cls(state: MonitorState):
    """MES ORB-cls — reuses the same MES 1-min bars fetched for the morning ORB."""
    bars = state.bars_mes_orb if state.bars_mes_orb else state.bars_1m
    _update_orb_cls_generic(state.orb_cls_mes, bars, ORB_CLS_MES_ENTRY_WIN)


# ─── DOM reader ───────────────────────────────────────────────────────────────

def _read_dom_db(state: MonitorState):
    if not DOM_DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(str(DOM_DB_PATH), timeout=1.0)
        row  = conn.execute(
            "SELECT updated,last_price,best_bid,best_ask,bids_json,asks_json "
            "FROM dom_live WHERE symbol=?", (SYMBOL,)).fetchone()
        conn.close()
        if not row:
            return
        updated_dt = datetime.fromisoformat(row[0])
        if (datetime.now(timezone.utc) - updated_dt).total_seconds() > DOM_DB_STALE:
            return
        bids = {float(k): v for k, v in json.loads(row[4]).items()}
        asks = {float(k): v for k, v in json.loads(row[5]).items()}
        with state.dom._lock:
            state.dom.bids = bids; state.dom.asks = asks
            state.dom.last_price = row[1]; state.dom.best_bid = row[2]
            state.dom.best_ask   = row[3]; state.dom.last_update = updated_dt
    except Exception:
        pass


# ─── VWASLR computation ───────────────────────────────────────────────────────

def _update_vwaslr_ema(state: MonitorState):
    bars   = state.bars_1m
    needed = VWASLR_N + VWASLR_SIGMA_BARS + 1
    if len(bars) < needed:
        return None   # not enough bars

    closes  = np.array([b.close  for b in bars], dtype=float)
    volumes = np.array([b.volume for b in bars], dtype=float)
    i = len(bars) - 1

    trail = np.log(closes[i - VWASLR_SIGMA_BARS + 1: i + 1]
                 / closes[i - VWASLR_SIGMA_BARS:     i    ])
    sigma = float(np.std(trail, ddof=1))
    if sigma == 0:
        return None

    ret_win = np.log(closes[i - VWASLR_N + 1: i + 1]
                   / closes[i - VWASLR_N:     i    ])
    vol_win = volumes[i - VWASLR_N: i]
    sum_vol = float(vol_win.sum())
    if sum_vol == 0:
        return None

    raw   = float((ret_win / sigma * vol_win).sum() / sum_vol)
    alpha = 2.0 / (VWASLR_EMA_SPAN + 1)
    state.vwaslr_ema_prev = state.vwaslr_ema
    state.vwaslr_ema      = alpha * raw + (1.0 - alpha) * state.vwaslr_ema
    return sigma * closes[i]   # sigma_pts for sizing


def _evaluate_vwaslr(state: MonitorState) -> "VwaslrSignal|None":
    ema      = state.vwaslr_ema
    ema_prev = state.vwaslr_ema_prev
    thr      = VWASLR_THRESHOLD

    crossed_up   = ema_prev <= thr  and ema > thr
    crossed_down = ema_prev >= -thr and ema < -thr
    if not crossed_up and not crossed_down:
        return None

    bars   = state.bars_1m
    needed = VWASLR_N + VWASLR_SIGMA_BARS + 1
    if len(bars) < needed:
        return None

    last   = bars[-1]
    bar_et = last.ts.astimezone(ET)
    bar_hm = (bar_et.hour, bar_et.minute)
    if bar_hm < (9, 0) or bar_hm >= (16, 0):
        return None
    if (9, 30) <= bar_hm < (9, 40):
        return None

    closes = np.array([b.close for b in bars], dtype=float)
    i = len(bars) - 1
    trail    = np.log(closes[i - VWASLR_SIGMA_BARS + 1: i+1]
                    / closes[i - VWASLR_SIGMA_BARS:    i  ])
    sigma    = float(np.std(trail, ddof=1))
    sigma_pts = sigma * closes[i]

    direction  = 1 if crossed_up else -1
    entry      = last.close
    stop_pts   = VWASLR_STOP_S   * sigma_pts
    target_pts = VWASLR_TARGET_S * sigma_pts

    return VwaslrSignal(
        direction=direction, entry=entry, sigma_pts=sigma_pts, ema=ema,
        fired_at=last.ts,
        expires_at=last.ts + timedelta(minutes=30),  # display window only
    )


# ─── ORB tracking ─────────────────────────────────────────────────────────────

def _update_orb(state: MonitorState):
    # ORB uses MNQ 1-min bars — fetch separately if available
    bars = state.bars_orb if state.bars_orb else state.bars_1m
    if not bars:
        return
    now_et    = datetime.now(ET)
    today     = now_et.date()
    orb       = state.orb

    # Reset on new day
    if orb.session_date != today:
        orb.__init__()
        orb.session_date = today

    orb_start_hm = 9 * 60 + 30
    orb_end_hm   = orb_start_hm + ORB_MIN   # 9:31 ET (1-min ORB)

    # Collect today's RTH 1-min bars
    today_rth = [b for b in bars
                 if b.ts.astimezone(ET).date() == today
                 and 9*60+30 <= b.ts.astimezone(ET).hour*60 + b.ts.astimezone(ET).minute < 16*60]
    if not today_rth:
        return

    # Build ORB range from the 9:30 bar only (1-min ORB)
    orb_bars = [b for b in today_rth
                if b.ts.astimezone(ET).hour*60 + b.ts.astimezone(ET).minute < orb_end_hm]

    if orb_bars:
        orb.orb_high  = max(b.high  for b in orb_bars)
        orb.orb_low   = min(b.low   for b in orb_bars)
        orb.orb_mid   = (orb.orb_high + orb.orb_low) / 2
        orb.orb_width = orb.orb_high - orb.orb_low
        if orb.orb_mid > 0:
            w_pct = orb.orb_width / orb.orb_mid
            orb.valid = w_pct <= ORB_WIDTH_MAX
        else:
            orb.valid = False

    # Look for breakout (only if ORB formed and no breakout yet)
    if orb.valid and orb.direction == 0:
        entry_end_hm = orb_end_hm + ORB_ENTRY_WIN
        after_orb = [b for b in today_rth
                     if orb_end_hm <= b.ts.astimezone(ET).hour*60 + b.ts.astimezone(ET).minute < entry_end_hm]
        for b in after_orb:
            w = orb.orb_width
            if b.close > orb.orb_high:
                raw_stop = orb.orb_low
                max_pts  = ORB_MAX_LOSS / ORB_PV
                stop     = max(raw_stop, b.close - max_pts)
                orb.direction   = 1
                orb.entry_price = b.close
                orb.target      = b.close + w * ORB_TGT_MULT
                orb.stop        = stop
                break
            if b.close < orb.orb_low:
                raw_stop = orb.orb_high
                max_pts  = ORB_MAX_LOSS / ORB_PV
                stop     = min(raw_stop, b.close + max_pts)
                orb.direction   = -1
                orb.entry_price = b.close
                orb.target      = b.close - w * ORB_TGT_MULT
                orb.stop        = stop
                break

    # Cross-confirmation: re-checked every call so it can flip True as soon
    # as MES's own candidate catches up (mirrors trading_bot.py's live gate).
    if orb.direction != 0 and not orb.confirmed:
        if state.orb_mes.direction == orb.direction:
            orb.confirmed = True


# ─── PL_REV evaluation ───────────────────────────────────────────────────────

def _evaluate_pl_rev(bars: list, prev_sig, sigma: float) -> "PLRevSignal|None":
    if len(bars) < PL_WINDOW + 1:
        return None
    window = bars[-PL_WINDOW:]
    for i in range(1, len(window)):
        if (window[i].ts - window[i-1].ts).total_seconds() > 8:
            return None
    last = window[-1]
    if prev_sig and last.ts == prev_sig.bar_ts:
        return None
    closes  = np.array([b.close for b in window], dtype=float)
    rets    = np.log(closes[1:] / closes[:-1])
    sum_abs = float(np.abs(rets).sum())
    if sum_abs == 0:
        return None
    pl       = float(abs(rets.sum()) / sum_abs)
    if pl < PL_ENTRY_PL:
        return None
    net_ret  = float(rets.sum())
    move_bps = abs(net_ret) * 10000
    eff_thr  = max(PL_MOVE_BPS, PL_SIGMA_N * sigma) if sigma > 0 else PL_MOVE_BPS
    if move_bps < eff_thr:
        return None
    direction = -1 if net_ret > 0 else 1   # FADE: opposite to momentum
    entry     = last.close
    return PLRevSignal(direction=direction, entry=entry, pl=pl,
                       move_bps=move_bps, bar_ts=last.ts)


# ─── Panel builders ───────────────────────────────────────────────────────────

def build_vwaslr_panel(state: MonitorState, now: datetime) -> Panel:
    ema  = state.vwaslr_ema
    thr  = VWASLR_THRESHOLD
    sig  = state.vwaslr_signal
    bars = state.bars_1m

    warming = len(bars) < VWASLR_N + VWASLR_SIGMA_BARS + 1

    if warming:
        status = "WARMING UP"
        border = "default"
        style  = ""
    elif sig and now < sig.expires_at:
        direction = sig.direction
        status = "▲ LONG" if direction == 1 else "▼ SHORT"
        border = "green" if direction == 1 else "red"
        style  = f"bold {'green' if direction == 1 else 'red'}"
    elif ema > thr:
        status = "ABOVE THR"; border = "green"; style = "green"
    elif ema < -thr:
        status = "BELOW THR"; border = "red";   style = "red"
    else:
        status = "WATCHING";  border = "default"; style = "bold"

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")
    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")
    root.add_row("")

    # EMA gauge
    gauge_half = 8
    ema_norm   = max(-1.0, min(1.0, ema / (thr * 3)))
    filled     = int(abs(ema_norm) * gauge_half)
    empty      = gauge_half - filled
    if ema >= 0:
        gauge = f"{'░'*gauge_half}[green]{'|'}{'█'*filled}{'░'*empty}[/]"
    else:
        gauge = f"[red]{'░'*empty}{'█'*filled}{'|'}[/]{'░'*gauge_half}"

    grd = Table.grid(padding=(0, 1))
    grd.add_column(width=10, justify="right")
    grd.add_column()
    grd.add_row("EMA:", f"{gauge}  {ema:+.4f}σ")
    grd.add_row("Threshold:", f"±{thr:.2f}σ  (cross fires signal)")
    if bars and len(bars) >= 2:
        last_et = bars[-1].ts.astimezone(LOCAL)
        sp = state.vwaslr_sigma_pts
        sigma_s = f"  σ={sp:.2f}pt" if sp else ""
        grd.add_row("Last bar:", f"{last_et.strftime('%H:%M')}  "
                                 f"close={bars[-1].close:.2f}{sigma_s}")
    root.add_row(grd)
    root.add_row("")

    if sig and now < sig.expires_at:
        sdet = Table.grid(padding=(0, 1))
        sdet.add_column(width=10, justify="right")
        sdet.add_column()
        clr = "green" if sig.direction == 1 else "red"
        tgt = sig.entry + sig.direction * VWASLR_TARGET_S * sig.sigma_pts
        stp = sig.entry - sig.direction * VWASLR_STOP_S   * sig.sigma_pts
        rem = max(0, int((sig.expires_at - now).total_seconds()))
        sdet.add_row("Entry:",   f"[bold {clr}]{sig.entry:.2f}[/]")
        sdet.add_row("Target:",  f"[bold green]{tgt:.2f}[/]  ({VWASLR_TARGET_S:.0f}σ)")
        sdet.add_row("Stop:",    f"[bold red]{stp:.2f}[/]  ({VWASLR_STOP_S:.0f}σ)")
        sdet.add_row("Expires:", f"{rem}s")
        root.add_row(sdet)
        root.add_row("")

    # History (last few EMA readings)
    if state.vwaslr_history:
        ht = Table(box=None, show_header=True, padding=(0, 1), header_style="bold")
        ht.add_column("time",   justify="right")
        ht.add_column("EMA",    justify="right")
        ht.add_column("",       justify="left", no_wrap=True)
        ht.add_column("signal", justify="center")
        for ts, ema_h, fired in reversed(state.vwaslr_history[-10:]):
            t_str = ts.astimezone(LOCAL).strftime("%H:%M")
            clr   = ("green" if ema_h > thr else "red" if ema_h < -thr else "")
            ema_s = (f"[{clr}]{ema_h:+.4f}[/]" if clr else f"{ema_h:+.4f}")
            filled2 = max(0, min(8, int(abs(ema_h) / (thr * 2) * 8)))
            bar_s = ("▲" if ema_h > 0 else "▼") * max(1, filled2)
            sig_s = ("[bold green]▲ LONG[/]" if fired == 1 else
                     "[bold red]▼ SHORT[/]" if fired == -1 else "")
            ht.add_row(t_str, ema_s, bar_s, sig_s)
        root.add_row(ht)

    if warming:
        root.add_row(f"Need {VWASLR_N + VWASLR_SIGMA_BARS + 1 - len(bars)} more bars")

    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"VWASLR({VWASLR_N}min, σ={VWASLR_SIGMA_BARS}min)  EMA({VWASLR_EMA_SPAN})  thr ±{thr:.2f}σ  2σ/3σ bracket")
    root.add_row(foot)

    return Panel(root, title=f"VWASLR  {SYMBOL}", border_style=border,
                 padding=(0, 1), expand=True)


def build_pl_rev_panel(state: MonitorState, now: datetime, symbol: str = "MES") -> Panel:
    if symbol == "MNQ":
        sig      = state.pl_sig_mnq
        hist_src = state.pl_history_mnq
        bars_src = state.bars_5s_mnq
    else:
        sig      = state.pl_sig
        hist_src = state.pl_history
        bars_src = state.bars_5s

    rev_qualifies = (sig is not None
                     and sig.pl >= PL_ENTRY_PL
                     and sig.move_bps >= PL_MOVE_BPS)

    if not rev_qualifies:
        status = "WATCHING";    style = "bold";        border = "default"
    elif sig.direction == 1:
        status = "FADE LONG";   style = "bold green";  border = "green"
    else:
        status = "FADE SHORT";  style = "bold red";    border = "red"

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")
    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")
    root.add_row("")

    if rev_qualifies:
        det = Table.grid(padding=(0, 1))
        det.add_column(width=9, justify="right")
        det.add_column()
        fade_dir = sig.direction
        mom_dir  = -fade_dir
        clr      = "green" if fade_dir == 1 else "red"
        mom_sym  = "▼" if mom_dir == -1 else "▲"
        fade_sym = "▲ LONG" if fade_dir == 1 else "▼ SHORT"
        tp_pt    = sig.entry * PL_TP_BPS   / 10000.0
        stop_pt  = sig.entry * PL_STOP_BPS / 10000.0
        tgt      = sig.entry + fade_dir * tp_pt
        stp      = sig.entry - fade_dir * stop_pt
        rem_s    = max(0, int((sig.expires_at() - now).total_seconds()))
        det.add_row("", f"mom {mom_sym}  →  fade [bold {clr}]{fade_sym}[/]  "
                        f"PL={sig.pl:.3f}  {sig.move_bps:.1f}bp")
        det.add_row("Entry:",   f"[bold]{sig.entry:,.2f}[/]")
        det.add_row("Target:",  f"[bold green]{tgt:,.2f}[/]  ({tp_pt:.2f}pt  {PL_TP_BPS:.0f}bp)")
        det.add_row("Stop:",    f"[bold red]{stp:,.2f}[/]  ({stop_pt:.2f}pt  {PL_STOP_BPS:.0f}bp)")
        det.add_row("Expires:", f"{rem_s//60}m {rem_s%60:02d}s")
        root.add_row(det)
        root.add_row("")

    hist = hist_src[-PL_HISTORY:]
    if hist:
        ht = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), header_style="bold")
        ht.add_column("time",   justify="right")
        ht.add_column("mom",    justify="center")
        ht.add_column("",       justify="left",  no_wrap=True)
        ht.add_column("PL",     justify="right")
        ht.add_column("30s bp", justify="right")
        ht.add_column("fade?",  justify="center")
        for ts, pl, dir_sym, move in reversed(hist):
            qualifies = pl >= PL_ENTRY_PL and move >= PL_MOVE_BPS
            pl_sty    = "bold green" if pl >= PL_ENTRY_PL else (
                        "yellow"    if pl >= PL_ENTRY_PL * 0.85 else "")
            bp_sty    = "bold green" if move >= PL_MOVE_BPS else ""
            fade_col  = "[bold green]✓[/]" if qualifies else ""
            ht.add_row(
                ts.astimezone(LOCAL).strftime("%H:%M:%S"),
                dir_sym, _pl_bar(pl, dir_sym),
                f"[{pl_sty}]{pl:.3f}[/]" if pl_sty else f"{pl:.3f}",
                f"[{bp_sty}]{move:.1f}[/]" if bp_sty else f"{move:.1f}",
                fade_col)
        root.add_row(ht)
    elif not bars_src:
        root.add_row("warming up…")

    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"entry: PL≥{PL_ENTRY_PL:.2f} bp≥{PL_MOVE_BPS:.0f}  "
                 f"TP {PL_TP_BPS:.0f}bp  stop {PL_STOP_BPS:.0f}bp  resume≥{PL_RESUME_PL:.2f}")
    root.add_row(foot)

    return Panel(root, title=f"PL REV  {symbol}", border_style=border,
                 padding=(0, 1), expand=True)


# ─── ORB ground-truth sync ─────────────────────────────────────────────────
# This monitor recomputes ORB candidates locally (above) purely to show the
# reference range while it's forming. But that local recompute is a separate
# simulation from trading_bot.py's own live ORB state — it can diverge (e.g.
# a late bar-fetch skips the bar where trading_bot's candidate first latched,
# so this monitor latches a different, later bar/direction instead). Since
# only trading_bot.py's process actually holds the broker connection, its
# log is ground truth for whether a candidate fired, confirmed, and what
# price it's actually working at. Sync local state to that truth whenever
# it's available so "Confirmed: yes — order placed" only shows when a real
# order was actually sent.
_ORB_CANDIDATE_RE = re.compile(r"ORB (MES|MNQ): candidate (LONG|SHORT) at ([\d.]+)")
_ORB_FIRE_RE       = re.compile(r"ORB (MES|MNQ): cross-confirmed by \w+ — firing (LONG|SHORT) at ([\d.]+)")
_ORB_ORDER_RE      = re.compile(r"ORB ORDER\s+(MES|MNQ) (LONG|SHORT)\s+order_id=(\S+)\s+entry≈([\d.]+)")
_ORB_FILL_RE       = re.compile(r"ORB (MES|MNQ) fill confirmed: ([\d.]+)")
# ORB-cls fires immediately on breakout (no cross-confirm candidate phase),
# so there's no equivalent of _ORB_CANDIDATE_RE/_ORB_FIRE_RE here.
_ORB_CLS_ORDER_RE = re.compile(r"ORB_CLS ORDER\s+(MES|MNQ) (LONG|SHORT)\s+order_id=(\S+)\s+entry≈([\d.]+)")
_ORB_CLS_FILL_RE  = re.compile(r"ORB_CLS (MES|MNQ) fill confirmed: ([\d.]+)")


def _sync_orb_from_log(state: MonitorState):
    today_str = datetime.now().date().isoformat()   # log timestamps are local system time
    found: dict = {}
    for log_path in (Path("logs/focused_bot.log"), Path("logs/trading_bot.log")):
        if not log_path.exists():
            continue
        try:
            with open(log_path) as f:
                lines = f.readlines()[-500:]
        except Exception:
            continue
        for line in lines:
            if not line.startswith(today_str):
                continue
            m = _ORB_CANDIDATE_RE.search(line)
            if m:
                sym, dirw, price = m.group(1), m.group(2), float(m.group(3))
                found[sym] = {"direction": 1 if dirw == "LONG" else -1,
                              "entry": price, "confirmed": False}
                continue
            m = _ORB_FIRE_RE.search(line)
            if m:
                sym, dirw, price = m.group(1), m.group(2), float(m.group(3))
                found[sym] = {"direction": 1 if dirw == "LONG" else -1,
                              "entry": price, "confirmed": True}
                continue
            m = _ORB_ORDER_RE.search(line)
            if m:
                sym, dirw, price = m.group(1), m.group(2), float(m.group(4))
                found[sym] = {"direction": 1 if dirw == "LONG" else -1,
                              "entry": price, "confirmed": True}
                continue
            m = _ORB_FILL_RE.search(line)
            if m:
                sym, price = m.group(1), float(m.group(2))
                if sym in found:
                    found[sym]["entry"] = price
                continue

    for sym, info in found.items():
        orb = state.orb_mes if sym == "MES" else state.orb
        if orb.session_date != datetime.now(ET).date() or not orb.orb_high:
            continue   # local range not built yet for today — nothing to reconcile against
        orb.direction   = info["direction"]
        orb.entry_price = info["entry"]
        orb.confirmed   = info["confirmed"]
        w = orb.orb_width
        if sym == "MES":
            if orb.direction == 1:
                orb.target = orb.entry_price + w * ORB_TGT_MULT
                orb.stop   = orb.entry_price - w / 2.0
            else:
                orb.target = orb.entry_price - w * ORB_TGT_MULT
                orb.stop   = orb.entry_price + w / 2.0
        else:
            max_pts = ORB_MAX_LOSS / ORB_PV
            if orb.direction == 1:
                orb.target = orb.entry_price + w * ORB_TGT_MULT
                orb.stop   = max(orb.orb_low, orb.entry_price - max_pts)
            else:
                orb.target = orb.entry_price - w * ORB_TGT_MULT
                orb.stop   = min(orb.orb_high, orb.entry_price + max_pts)


def _sync_orb_cls_from_log(state: MonitorState):
    """Mirrors _sync_orb_from_log for ORB-cls. Both MES and MNQ use a plain
    midpoint stop here (no far-side/dollar-cap variant), so target/stop use
    the same formula for both symbols."""
    today_str = datetime.now().date().isoformat()
    found: dict = {}
    for log_path in (Path("logs/focused_bot.log"), Path("logs/trading_bot.log")):
        if not log_path.exists():
            continue
        try:
            with open(log_path) as f:
                lines = f.readlines()[-500:]
        except Exception:
            continue
        for line in lines:
            if not line.startswith(today_str):
                continue
            m = _ORB_CLS_ORDER_RE.search(line)
            if m:
                sym, dirw, price = m.group(1), m.group(2), float(m.group(4))
                found[sym] = {"direction": 1 if dirw == "LONG" else -1,
                              "entry": price, "confirmed": True}
                continue
            m = _ORB_CLS_FILL_RE.search(line)
            if m:
                sym, price = m.group(1), float(m.group(2))
                if sym in found:
                    found[sym]["entry"] = price
                continue

    for sym, info in found.items():
        orb = state.orb_cls_mes if sym == "MES" else state.orb_cls
        if orb.session_date != datetime.now(ET).date() or not orb.orb_high:
            continue   # local range not built yet for today — nothing to reconcile against
        orb.direction   = info["direction"]
        orb.entry_price = info["entry"]
        orb.confirmed   = info["confirmed"]
        w = orb.orb_width
        if orb.direction == 1:
            orb.target = orb.entry_price + w * ORB_CLS_TGT_MULT
            orb.stop   = orb.entry_price - w / 2.0
        else:
            orb.target = orb.entry_price - w * ORB_CLS_TGT_MULT
            orb.stop   = orb.entry_price + w / 2.0


def build_orb_panel(state: MonitorState, now: datetime) -> Panel:
    orb    = state.orb
    now_et = now.astimezone(ET)
    hm_et  = now_et.hour * 60 + now_et.minute
    orb_end_hm = 9*60 + 30 + ORB_MIN
    entry_end_hm = orb_end_hm + ORB_ENTRY_WIN

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")

    if orb.session_date is None or not orb.orb_high:
        if hm_et < 9*60+30:
            status = "PRE-MARKET"; style = ""; border = "default"
        else:
            status = "LOADING…"; style = ""; border = "default"
        root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")
        return Panel(root, title=f"ORB  MNQ  (1-min)", border_style=border,
                     padding=(0, 1), width=68)

    # Determine status
    dir_word = "LONG" if orb.direction == 1 else "SHORT"
    if hm_et < orb_end_hm:
        status = "FORMING"; style = "bold"; border = "default"
    elif not orb.valid:
        status = "INVALID WIDTH"; style = "yellow"; border = "yellow"
    elif orb.direction == 0 and hm_et >= entry_end_hm:
        status = "EXPIRED"; style = ""; border = "default"
    elif orb.direction == 0:
        status = "WATCHING"; style = "bold"; border = "blue"
    elif orb.confirmed and orb.direction == 1:
        status = "▲ BREAKOUT LONG"; style = "bold green"; border = "green"
    elif orb.confirmed:
        status = "▼ BREAKOUT SHORT"; style = "bold red"; border = "red"
    elif hm_et >= entry_end_hm:
        status = f"EXPIRED (unconfirmed {dir_word})"; style = "yellow"; border = "yellow"
    else:
        status = f"CANDIDATE {dir_word} — awaiting MES"; style = "bold yellow"; border = "yellow"

    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")
    root.add_row("")

    # ORB range info
    rng = Table.grid(padding=(0, 1))
    rng.add_column(width=10, justify="right")
    rng.add_column()

    if orb.orb_high:
        w_pct = orb.orb_width / orb.orb_mid * 100 if orb.orb_mid else 0
        valid_s = "✓" if orb.valid else f"✗ (need ≤{ORB_WIDTH_MAX*100:.2f}%)"
        v_clr   = "green" if orb.valid else "yellow"
        rng.add_row("ORB High:",  f"[bold]{orb.orb_high:.2f}[/]")
        rng.add_row("ORB Low:",   f"[bold]{orb.orb_low:.2f}[/]")
        rng.add_row("Width:",     f"{orb.orb_width:.2f}pt  {w_pct:.3f}%  [{v_clr}]{valid_s}[/]")
        rng.add_row("Midpoint:",  f"{orb.orb_mid:.2f}")

    root.add_row(rng)

    # Breakout details
    if orb.direction != 0:
        root.add_row("")
        bdet = Table.grid(padding=(0, 1))
        bdet.add_column(width=10, justify="right")
        bdet.add_column()
        clr = "green" if orb.direction == 1 else "red"
        exit_et_min = 9*60+31 + ORB_ENTRY_WIN - 1 + ORB_HOLD_MIN
        exit_h, exit_m = divmod(exit_et_min, 60)
        bdet.add_row("Confirmed:", "[green]yes — order placed[/]" if orb.confirmed
                                    else "[yellow]not yet — no trade[/]")
        bdet.add_row("Entry:",  f"[bold {clr}]{orb.entry_price:.2f}[/]"
                                 + ("" if orb.confirmed else "  (candidate level)"))
        bdet.add_row("Target:", f"[bold green]{orb.target:.2f}[/]  "
                                f"(+{abs(orb.target-orb.entry_price):.2f}pt)")
        bdet.add_row("Stop:",   f"[bold red]{orb.stop:.2f}[/]  "
                                f"({abs(orb.stop-orb.entry_price):.2f}pt)")
        bdet.add_row("Exit by:", f"~{exit_h:02d}:{exit_m:02d} ET" if orb.confirmed else "—")
        root.add_row(bdet)
    elif orb.valid and hm_et < entry_end_hm:
        root.add_row("")
        root.add_row(f"  Entry window: 9:31–9:36 ET  "
                     f"(close > {orb.orb_high:.2f} → LONG  /  "
                     f"close < {orb.orb_low:.2f} → SHORT)")

    root.add_row("")
    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"MNQ ORB  ≤{ORB_ENTRY_WIN}min  hold {ORB_HOLD_MIN}min  stop=opp(cap ${ORB_MAX_LOSS:.0f})  tgt={ORB_TGT_MULT:.0f}×  w≤{ORB_WIDTH_MAX*10000:.0f}bp  cross-confirm w/ MES")
    root.add_row(foot)

    return Panel(root, title=f"ORB  MNQ  (1-min)", border_style=border,
                 padding=(0, 1), width=68)


def build_mes_orb_panel(state: MonitorState, now: datetime) -> Panel:
    orb    = state.orb_mes
    now_et = now.astimezone(ET)
    hm_et  = now_et.hour * 60 + now_et.minute
    orb_end_hm   = 9*60 + 30 + ORB_MIN
    entry_end_hm = orb_end_hm + ORB_ENTRY_WIN

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")

    if orb.session_date is None or not orb.orb_high:
        status = "PRE-MARKET" if hm_et < 9*60+30 else "LOADING…"
        root.add_row(f"  {status}  ")
        return Panel(root, title="ORB  MES  (1-min)", border_style="default",
                     padding=(0, 1), width=72)

    dir_word = "LONG" if orb.direction == 1 else "SHORT"
    if hm_et < orb_end_hm:
        status = "FORMING"; style = "bold"; border = "default"
    elif orb.direction == 0 and hm_et >= entry_end_hm:
        status = "EXPIRED"; style = ""; border = "default"
    elif orb.direction == 0:
        status = "WATCHING"; style = "bold"; border = "blue"
    elif orb.confirmed and orb.direction == 1:
        status = "▲ BREAKOUT LONG"; style = "bold green"; border = "green"
    elif orb.confirmed:
        status = "▼ BREAKOUT SHORT"; style = "bold red"; border = "red"
    elif hm_et >= entry_end_hm:
        status = f"EXPIRED (unconfirmed {dir_word})"; style = "yellow"; border = "yellow"
    else:
        status = f"CANDIDATE {dir_word} — awaiting MNQ"; style = "bold yellow"; border = "yellow"

    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")
    root.add_row("")

    rng = Table.grid(padding=(0, 1))
    rng.add_column(width=10, justify="right")
    rng.add_column()
    if orb.orb_high:
        w_pct = orb.orb_width / orb.orb_mid * 100 if orb.orb_mid else 0
        rng.add_row("ORB High:", f"[bold]{orb.orb_high:.2f}[/]")
        rng.add_row("ORB Low:",  f"[bold]{orb.orb_low:.2f}[/]")
        rng.add_row("Width:",    f"{orb.orb_width:.2f}pt  {w_pct:.3f}%")
        rng.add_row("Midpoint:", f"{orb.orb_mid:.2f}")
    root.add_row(rng)

    if orb.direction != 0:
        root.add_row("")
        bdet = Table.grid(padding=(0, 1))
        bdet.add_column(width=10, justify="right")
        bdet.add_column()
        clr = "green" if orb.direction == 1 else "red"
        exit_et_min = 9*60+31 + ORB_ENTRY_WIN - 1 + ORB_HOLD_MIN
        exit_h, exit_m = divmod(exit_et_min, 60)
        bdet.add_row("Confirmed:", "[green]yes — order placed[/]" if orb.confirmed
                                    else "[yellow]not yet — no trade[/]")
        bdet.add_row("Entry:",   f"[bold {clr}]{orb.entry_price:.2f}[/]"
                                  + ("" if orb.confirmed else "  (candidate level)"))
        bdet.add_row("Target:",  f"[bold green]{orb.target:.2f}[/]  "
                                 f"(+{abs(orb.target-orb.entry_price):.2f}pt)")
        bdet.add_row("Stop:",    f"[bold red]{orb.stop:.2f}[/]  "
                                 f"({abs(orb.stop-orb.entry_price):.2f}pt  midpoint)")
        bdet.add_row("Exit by:", f"~{exit_h:02d}:{exit_m:02d} ET" if orb.confirmed else "—")
        root.add_row(bdet)
    elif hm_et < entry_end_hm:
        root.add_row("")
        root.add_row(f"  Entry window: 9:31–9:36 ET  "
                     f"(close > {orb.orb_high:.2f} → LONG  /  "
                     f"close < {orb.orb_low:.2f} → SHORT)")

    root.add_row("")
    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"MES ORB  ≤{ORB_ENTRY_WIN}min  hold {ORB_HOLD_MIN}min  stop=midpoint  tgt={ORB_TGT_MULT:.0f}×  no width filter  cross-confirm w/ MNQ")
    root.add_row(foot)

    return Panel(root, title="ORB  MES  (1-min)", border_style=border,
                 padding=(0, 1), width=72)


def _build_orb_cls_panel(orb: ORBState, now: datetime, symbol: str,
                         entry_win: int, panel_width: int) -> Panel:
    """Shared ORB-cls rendering for MES/MNQ — identical mechanics for both
    (midpoint stop, no cross-confirm), only symbol/entry_win/width differ."""
    now_et = now.astimezone(ET)
    hm_et  = now_et.hour * 60 + now_et.minute
    range_hm     = ORB_CLS_RANGE_HM[0] * 60 + ORB_CLS_RANGE_HM[1]
    range_end_hm = range_hm + 1
    entry_end_hm = range_end_hm + entry_win
    safety_min   = ORB_CLS_SAFETY_HM[0] * 60 + ORB_CLS_SAFETY_HM[1]

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")

    if orb.session_date is None or not orb.orb_high:
        status = "PRE-MARKET" if hm_et < range_hm else "LOADING…"
        root.add_row(f"  {status}  ")
        return Panel(root, title=f"ORB-cls  {symbol}  (15:50 ET)", border_style="default",
                     padding=(0, 1), width=panel_width)

    dir_word = "LONG" if orb.direction == 1 else "SHORT"
    if hm_et < range_end_hm:
        status = "FORMING"; style = "bold"; border = "default"
    elif orb.direction == 0 and hm_et >= entry_end_hm:
        status = "EXPIRED"; style = ""; border = "default"
    elif orb.direction == 0:
        status = "WATCHING"; style = "bold"; border = "blue"
    elif orb.confirmed and orb.direction == 1:
        status = "▲ BREAKOUT LONG"; style = "bold green"; border = "green"
    elif orb.confirmed:
        status = "▼ BREAKOUT SHORT"; style = "bold red"; border = "red"
    else:
        status = f"{dir_word} — confirming…"; style = "bold yellow"; border = "yellow"

    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")
    root.add_row("")

    rng = Table.grid(padding=(0, 1))
    rng.add_column(width=10, justify="right")
    rng.add_column()
    if orb.orb_high:
        w_pct = orb.orb_width / orb.orb_mid * 100 if orb.orb_mid else 0
        rng.add_row("Range Hi:", f"[bold]{orb.orb_high:.2f}[/]")
        rng.add_row("Range Lo:", f"[bold]{orb.orb_low:.2f}[/]")
        rng.add_row("Width:",    f"{orb.orb_width:.2f}pt  {w_pct:.3f}%")
        rng.add_row("Midpoint:", f"{orb.orb_mid:.2f}")
    root.add_row(rng)

    if orb.direction != 0:
        root.add_row("")
        bdet = Table.grid(padding=(0, 1))
        bdet.add_column(width=10, justify="right")
        bdet.add_column()
        clr = "green" if orb.direction == 1 else "red"
        nominal_exit_min = entry_end_hm - 1 + ORB_CLS_HOLD_MIN
        exit_et_min = min(nominal_exit_min, safety_min)
        exit_h, exit_m = divmod(exit_et_min, 60)
        bdet.add_row("Confirmed:", "[green]yes — order placed[/]" if orb.confirmed
                                    else "[yellow]not yet — no trade[/]")
        bdet.add_row("Entry:",   f"[bold {clr}]{orb.entry_price:.2f}[/]"
                                  + ("" if orb.confirmed else "  (candidate level)"))
        bdet.add_row("Target:",  f"[bold green]{orb.target:.2f}[/]  "
                                 f"(+{abs(orb.target-orb.entry_price):.2f}pt)")
        bdet.add_row("Stop:",    f"[bold red]{orb.stop:.2f}[/]  "
                                 f"({abs(orb.stop-orb.entry_price):.2f}pt  midpoint)")
        bdet.add_row("Exit by:", f"~{exit_h:02d}:{exit_m:02d} ET" if orb.confirmed else "—")
        root.add_row(bdet)
    elif hm_et < entry_end_hm:
        root.add_row("")
        root.add_row(f"  Entry window: 15:51–{(range_end_hm+entry_win-1)//60:02d}:{(range_end_hm+entry_win-1)%60:02d} ET  "
                     f"(close > {orb.orb_high:.2f} → LONG  /  "
                     f"close < {orb.orb_low:.2f} → SHORT)")

    root.add_row("")
    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"{symbol} ORB-cls  ≤{entry_win}min  hold {ORB_CLS_HOLD_MIN}min  stop=midpoint  "
                f"tgt={ORB_CLS_TGT_MULT:.2f}×  no cross-confirm  "
                f"safety cutoff {ORB_CLS_SAFETY_HM[0]:02d}:{ORB_CLS_SAFETY_HM[1]:02d} ET")
    root.add_row(foot)

    return Panel(root, title=f"ORB-cls  {symbol}  (15:50 ET)", border_style=border,
                 padding=(0, 1), width=panel_width)


def build_orb_cls_panel(state: MonitorState, now: datetime) -> Panel:
    return _build_orb_cls_panel(state.orb_cls, now, "MNQ", ORB_CLS_MNQ_ENTRY_WIN, 68)


def build_mes_orb_cls_panel(state: MonitorState, now: datetime) -> Panel:
    return _build_orb_cls_panel(state.orb_cls_mes, now, "MES", ORB_CLS_MES_ENTRY_WIN, 72)


def build_dom_panel(state: MonitorState) -> Panel:
    wall_tests: dict = {}
    if state.wall_tracker is not None:
        for w in state.wall_tracker.active_walls():
            if w.test_count > 0:
                wall_tests[w.price] = w.test_count

    bid_rows, ask_rows, last, bb, ba, updated = state.dom.hybrid_snapshot(
        DOM_NEAR, DOM_BUCKET_PT, DOM_BUCKET_N)

    all_sizes = [s for _, s, _ in bid_rows + ask_rows if s > 0]
    med_sz    = float(np.median(all_sizes)) if all_sizes else 1

    def sz_bar(size, is_bid, is_bucket):
        cap   = DOM_BKT_CAP if is_bucket else DOM_NEAR_CAP
        width = DOM_BKT_BAR if is_bucket else DOM_NEAR_BAR
        filled = max(1, int(round(min(size, cap) / cap * width))) if size > 0 else 0
        ch = ("▒" if is_bucket else "█") * filled + "░" * (width - filled)
        return f"[{'green' if is_bid else 'red'}]{ch}[/]"

    def is_wall(size): return size > 0 and size >= med_sz * WALL_MULT

    t = Table(box=None, show_header=False, padding=(0, 0))
    t.add_column(justify="right",  width=6)
    t.add_column(justify="right",  width=DOM_NEAR_BAR)
    t.add_column(justify="center", width=DOM_PRICE_COL)
    t.add_column(justify="left",   width=DOM_NEAR_BAR)
    t.add_column(justify="left",   width=6)

    for label, size, is_bucket in list(reversed(ask_rows[DOM_NEAR:])) + list(reversed(ask_rows[:DOM_NEAR])):
        if label is None: t.add_row("","","","",""); continue
        wall  = is_wall(size)
        style = "bold red" if wall else ("red" if size > 0 else "")
        bar   = sz_bar(size, False, is_bucket)
        wm    = "◀" if wall else " "
        tc    = wall_tests.get(label) if isinstance(label, float) else None
        tc_s  = f"[yellow]T{tc}[/]" if tc else ""
        sz_s  = f"[{style}]{size:.0f}{wm}[/]{tc_s}" if size > 0 else ""
        lbl_raw = f"{label}" if is_bucket else f"{label:.2f}"
        lbl_s = f"[{style}]{lbl_raw}[/]" if style else lbl_raw
        t.add_row("", "", lbl_s, bar, sz_s)

    spread = f"{ba-bb:.2f}" if bb and ba else "—"
    imbal_txt = ""
    if all_sizes:
        tb = sum(s for _, s, _ in bid_rows if s > 0)
        ta = sum(s for _, s, _ in ask_rows if s > 0)
        d  = tb + ta
        im = (tb - ta) / d if d else 0
        ic = "green" if im > 0.1 else ("red" if im < -0.1 else "")
        imbal_txt = f"  [{ic}]imb {im:+.2f}[/]" if ic else f"  imb {im:+.2f}"
    t.add_row("", f"spread {spread}{imbal_txt}", "", "", "")

    for label, size, is_bucket in bid_rows[:DOM_NEAR] + bid_rows[DOM_NEAR:]:
        if label is None: t.add_row("","","","",""); continue
        wall  = is_wall(size)
        style = "bold green" if wall else ("green" if size > 0 else "")
        bar   = sz_bar(size, True, is_bucket)
        wm    = "◀" if wall else " "
        tc    = wall_tests.get(label) if isinstance(label, float) else None
        tc_s  = f"[yellow]T{tc}[/]" if tc else ""
        sz_s  = f"{tc_s}[{style}]{wm}{size:.0f}[/]" if size > 0 else ""
        lbl_raw = f"{label}" if is_bucket else f"{label:.2f}"
        lbl_s = f"[{style}]{lbl_raw}[/]" if style else lbl_raw
        t.add_row(sz_s, bar, lbl_s, "", "")

    age     = (datetime.now(timezone.utc) - updated).total_seconds()
    age_col = "green" if age < 10 else ("yellow" if age < 30 else "red")
    last_s  = f"{last:.2f}" if last else "—"
    return Panel(t, title=f"DOM  {SYMBOL}  [{age_col}]{last_s}[/]  {age:.0f}s",
                 border_style="blue", padding=(0, 1), expand=True)


def build_wall_panel(state: MonitorState, now: datetime) -> Panel:
    sig = state.wall_break_signal
    active = state.wall_tracker.active_walls() if state.wall_tracker else []

    if sig and now < sig.expires_at:
        if sig.direction == 1:
            status = "▲ BREAKOUT LONG"; border = "green"; style = "bold green"
        else:
            status = "▼ BREAKOUT SHORT"; border = "red"; style = "bold red"
    elif active:
        status = "WATCHING"; border = "blue"; style = "bold"
    else:
        status = "NO WALLS"; border = "default"; style = ""

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")
    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")

    # Active signal detail
    if sig and now < sig.expires_at:
        root.add_row("")
        det = Table.grid(padding=(0, 1))
        det.add_column(width=10, justify="right")
        det.add_column()
        clr  = "green" if sig.direction == 1 else "red"
        rem  = max(0, int((sig.expires_at - now).total_seconds()))
        det.add_row("Wall:",   f"[bold]{sig.wall_price:.2f}[/]  "
                               f"peak={sig.peak_size:.0f}  tests={sig.test_count}")
        det.add_row("Entry:",  f"[bold {clr}]{sig.entry:.2f}[/]")
        det.add_row("Target:", f"[bold green]{sig.target:.2f}[/]  (+{WALL_TARGET_PTS:.0f}pt)")
        det.add_row("Stop:",   f"[bold red]{sig.stop:.2f}[/]  ({WALL_STOP_PTS:.0f}pt)")
        det.add_row("Hold:",   f"{rem//60}m {rem%60:02d}s remaining")
        root.add_row(det)

    # Recent event log (tests + breakouts) — capped at 10 rows; walls shown on DOM ladder
    notable = [(ts, ev, side, wp, tc, ep, pk)
               for ts, ev, side, wp, tc, ep, pk in reversed(state.wall_recent_events[-30:])
               if ev in ("breakout", "test")][:10]
    if notable:
        root.add_row("")
        lt = Table(box=None, show_header=True, padding=(0, 1), header_style="bold")
        lt.add_column("time",   justify="right")
        lt.add_column("event",  justify="center")
        lt.add_column("side",   justify="center")
        lt.add_column("wall",   justify="right")
        lt.add_column("peak",   justify="right")
        lt.add_column("tests",  justify="center")
        for ts, ev, side, wp, tc, ep, pk in notable:
            t_s  = ts.astimezone(LOCAL).strftime("%H:%M:%S")
            ec   = "bold yellow" if ev == "breakout" else ""
            sc   = "green" if side == "bid" else "red"
            pk_s = f"[bold green]{pk}[/]" if pk >= 100 else str(pk)
            lt.add_row(t_s, f"[{ec}]{ev}[/]" if ec else ev, f"[{sc}]{side}[/]",
                       f"{wp:.2f}", pk_s, f"T{tc}")
        root.add_row(lt)

    root.add_row("")
    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"stop {WALL_STOP_PTS:.0f}pt  target {WALL_TARGET_PTS:.0f}pt  hold {WALL_HOLD_MIN}min  peak≥100  tests≥3  10:30–13:00 ET")
    root.add_row(foot)

    return Panel(root, title=f"WALL BREAK  {SYMBOL}", border_style=border,
                 padding=(0, 1), expand=False)


def build_ba_brk_panel(state: MonitorState, now: datetime) -> Panel:
    sig = state.ba_brk_signal

    # Live cascade progress: consecutive same-side breakouts chained within
    # the gap window, ending at the most recent breakout (informational —
    # mirrors trading_bot.py's _detect_cascade_signal but ignores the dedup
    # cursor so the panel always shows current progress, not just fresh
    # triggers). Tracked for both sides since the real bot trades both.
    def _live_cascade_len(side: str):
        evs = sorted((e for e in state.ba_brk_breakouts
                      if e.side == side and e.event == "breakout"), key=lambda e: e.ts)
        gaps = [None] + [(evs[i].ts - evs[i-1].ts).total_seconds() for i in range(1, len(evs))]
        n = 0
        for gap in gaps:
            n = n + 1 if (gap is not None and gap <= BA_BRK_MAX_GAP_SEC) else 1
        return n, evs

    live_len, ask_breakouts = _live_cascade_len("ask")
    aligned = (state.day_open is not None and ask_breakouts and
               (ask_breakouts[-1].price or ask_breakouts[-1].wall_price) > state.day_open)

    bid_live_len, bid_breakouts = _live_cascade_len("bid")
    bid_aligned = (state.day_open is not None and bid_breakouts and
                   (bid_breakouts[-1].price or bid_breakouts[-1].wall_price) < state.day_open)

    if sig and now < sig.expires_at:
        if sig.direction == 1:
            status = "▲ CASCADE LONG"; border = "green"; style = "bold green"
        else:
            status = "▼ CASCADE SHORT"; border = "red"; style = "bold red"
    elif max(live_len, bid_live_len) >= 2:
        side_word = "ask" if live_len >= bid_live_len else "bid"
        status = f"BUILDING {side_word} ({max(live_len, bid_live_len)}/{BA_BRK_CASCADE_MIN})"
        border = "yellow"; style = "bold yellow"
    else:
        status = "WATCHING"; border = "blue" if state.ba_brk_breakouts else "default"; style = "bold" if state.ba_brk_breakouts else ""

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")
    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")

    if sig and now < sig.expires_at:
        root.add_row("")
        det = Table.grid(padding=(0, 1))
        det.add_column(width=10, justify="right")
        det.add_column()
        rem = max(0, int((sig.expires_at - now).total_seconds()))
        clr = "green" if sig.direction == 1 else "red"
        tgt_pts  = BA_BRK_TARGET_PTS if sig.side == "ask" else BA_BRK_BID_TARGET_PTS
        stop_pts = BA_BRK_STOP_PTS   if sig.side == "ask" else BA_BRK_BID_STOP_PTS
        det.add_row("Cascade:", f"[bold]{sig.cascade_len}[/] breaks in "
                                 f"{(sig.last_break_ts - sig.first_break_ts).total_seconds():.0f}s")
        det.add_row("Entry:",   f"[bold {clr}]{sig.entry:.2f}[/]")
        det.add_row("Target:",  f"[bold green]{sig.target:.2f}[/]  "
                                 f"({'+' if sig.direction == 1 else '-'}{tgt_pts:.0f}pt)")
        det.add_row("Stop:",    f"[bold red]{sig.stop:.2f}[/]  "
                                 f"({'-' if sig.direction == 1 else '+'}{stop_pts:.1f}pt)")
        det.add_row("Hold:",    f"{rem//60}m {rem%60:02d}s remaining")
        root.add_row(det)
    elif state.day_open is not None:
        root.add_row("")
        det = Table.grid(padding=(0, 1))
        det.add_column(width=14, justify="right")
        det.add_column()
        det.add_row("Day open:", f"{state.day_open:.2f}")
        det.add_row("Ask Aligned:", ("[green]yes[/]" if aligned else "[red]no — filtered[/]")
                                     if ask_breakouts else "—")
        det.add_row("Bid Aligned:", ("[green]yes[/]" if bid_aligned else "[red]no — filtered[/]")
                                     if bid_breakouts else "—")
        root.add_row(det)

    # Recent wall activity, both sides, test + breakout (last 12) — same
    # richer view as WALL BREAK's event log, so there's something to watch
    # even on a bid-heavy day when no ask cascade has fired yet.
    notable = sorted(state.ba_brk_breakouts, key=lambda e: e.ts, reverse=True)[:12]
    if notable:
        root.add_row("")
        lt = Table(box=None, show_header=True, padding=(0, 1), header_style="bold")
        lt.add_column("time",   justify="right")
        lt.add_column("event",  justify="center")
        lt.add_column("side",   justify="center")
        lt.add_column("wall",   justify="right")
        lt.add_column("peak",   justify="right")
        lt.add_column("tests",  justify="center")
        for e in notable:
            t_s  = e.ts.astimezone(LOCAL).strftime("%H:%M:%S")
            ec   = "bold yellow" if e.event == "breakout" else ""
            sc   = "green" if e.side == "bid" else "red"
            pk   = e.peak_size
            pk_s = f"[bold green]{pk:.0f}[/]" if pk >= 100 else f"{pk:.0f}"
            lt.add_row(t_s, f"[{ec}]{e.event}[/]" if ec else e.event, f"[{sc}]{e.side}[/]",
                       f"{e.wall_price:.2f}", pk_s, f"T{e.test_count}")
        root.add_row(lt)

    root.add_row("")
    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"{BA_BRK_CASCADE_MIN}+ breaks / {BA_BRK_MAX_GAP_SEC}s  aligned w/ day open  9:40–13:00 ET")
    foot.add_row(f"ask {BA_BRK_STOP_PTS:.1f}/{BA_BRK_TARGET_PTS:.0f}/{BA_BRK_HOLD_MIN}  "
                 f"bid {BA_BRK_BID_STOP_PTS:.1f}/{BA_BRK_BID_TARGET_PTS:.0f}/{BA_BRK_BID_HOLD_MIN}  "
                 f"(stop/target/hold)")
    root.add_row(foot)

    return Panel(root, title=f"BA-BRK  {SYMBOL}", border_style=border,
                 padding=(0, 1), width=127)


def build_ba_rev_panel(state: MonitorState, now: datetime) -> Panel:
    sig = state.ba_rev_signal
    now_et_hm = now.astimezone(ET).hour * 60 + now.astimezone(ET).minute
    past_lock = now_et_hm >= BA_REV_LOCK_HM

    # Live test-count progress per side (most recent test event, informational)
    ask_tests  = [e for e in state.ba_brk_breakouts if e.side == "ask" and e.event == "test"]
    bid_tests  = [e for e in state.ba_brk_breakouts if e.side == "bid" and e.event == "test"]
    ask_latest = max(ask_tests, key=lambda e: e.ts) if ask_tests else None
    bid_latest = max(bid_tests, key=lambda e: e.ts) if bid_tests else None

    if sig and now < sig.expires_at:
        dir_arrow = "▲" if sig.direction == 1 else "▼"
        status = f"{dir_arrow} REVERSAL {'LONG' if sig.direction == 1 else 'SHORT'}"
        border, style = "green", "bold green"
    elif past_lock and not state.ba_rev_contained:
        status = "BROKE OUT — disabled today"; border = "red"; style = "bold red"
    elif not past_lock and state.ba_rev_ref_day is None:
        status = "BUILDING REFERENCE (07:00-10:30)"; border = "blue"; style = "bold"
    elif ((ask_latest and ask_latest.test_count >= BA_REV_TEST_COUNT - 1) or
          (bid_latest and bid_latest.test_count >= BA_REV_TEST_COUNT - 1)):
        status = "BUILDING"; border = "yellow"; style = "bold yellow"
    else:
        status = "WATCHING" + (" (pre-lock)" if not past_lock else " (contained)")
        border = "blue" if state.ba_brk_breakouts else "default"
        style = "bold" if state.ba_brk_breakouts else ""

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")
    root.add_row(f"[{style}]  {status}  [/]" if style else f"  {status}  ")

    if sig and now < sig.expires_at:
        root.add_row("")
        det = Table.grid(padding=(0, 1))
        det.add_column(width=11, justify="right")
        det.add_column()
        rem = max(0, int((sig.expires_at - now).total_seconds()))
        det.add_row("Rejection:", f"[bold]{sig.side}[/] wall @ {sig.wall_price:.2f}  (tests={sig.test_count})")
        det.add_row("Entry:",  f"[bold green]{sig.entry:.2f}[/]")
        det.add_row("Target:", f"[bold green]{sig.target:.2f}[/]  ({'+' if sig.direction == 1 else '-'}{BA_REV_TARGET_PTS:.0f}pt)")
        det.add_row("Stop:",   f"[bold red]{sig.stop:.2f}[/]  ({'-' if sig.direction == 1 else '+'}{BA_REV_STOP_PTS:.1f}pt)")
        det.add_row("Hold:",   f"{rem//60}m {rem%60:02d}s remaining")
        root.add_row(det)
    else:
        root.add_row("")
        det = Table.grid(padding=(0, 1))
        det.add_column(width=14, justify="right")
        det.add_column()
        if state.ba_rev_ref_high is not None:
            det.add_row("Ref range:", f"{state.ba_rev_ref_low:.2f} – {state.ba_rev_ref_high:.2f}  "
                                        f"({state.ba_rev_ref_high - state.ba_rev_ref_low:.1f}pt)  locked")
        else:
            # Live preview: running high/low from 07:00 ET to now, updates
            # every render until it locks at 10:30 ET — useful to watch build.
            ref_start_et = datetime.combine(now.astimezone(ET).date(),
                                             dtime(BA_REV_REF_START_HM // 60, BA_REV_REF_START_HM % 60), tzinfo=ET)
            preview = [b for b in state.bars_1m if b.ts.astimezone(ET) >= ref_start_et]
            if preview:
                p_hi = max(b.high for b in preview)
                p_lo = min(b.low for b in preview)
                det.add_row("Ref range:", f"{p_lo:.2f} – {p_hi:.2f}  ({p_hi - p_lo:.1f}pt)  building…")
            else:
                det.add_row("Ref range:", "— (building, locks 10:30 ET)")
        contain_style = "green" if state.ba_rev_contained else "red"
        det.add_row("Status:", f"[{contain_style}]{'contained' if state.ba_rev_contained else 'broken out'}[/]"
                                if past_lock else "pre-lock")
        if ask_latest:
            det.add_row("Ask tests:", f"{ask_latest.test_count} @ {ask_latest.wall_price:.2f}")
        if bid_latest:
            det.add_row("Bid tests:", f"{bid_latest.test_count} @ {bid_latest.wall_price:.2f}")
        root.add_row(det)

    root.add_row("")
    foot = Table.grid(); foot.add_column(justify="center")
    foot.add_row(f"{BA_REV_TEST_COUNT} tests, no break  "
                 f"stop {BA_REV_STOP_PTS:.1f}pt  target {BA_REV_TARGET_PTS:.0f}pt  "
                 f"hold {BA_REV_HOLD_MIN}min  both sides  "
                 f"trades from 10:30 ET while contained in 07:00-10:30 range")
    root.add_row(foot)

    return Panel(root, title=f"BA-REV  {SYMBOL}", border_style=border,
                 padding=(0, 1), width=107)


def _position_row(t, symbol: str, size: int, direction: int,
                   entry_px: "float|None", strat: str):
    entry = f"{entry_px:.2f}" if entry_px else ""
    if size == 0:
        t.add_row(symbol, "—", "", "", "")
    elif direction == 1:
        t.add_row(symbol, f"[green]{size}[/]", "[green]▲ LONG[/]",
                  f"[green]{entry}[/]", f"[green]{strat}[/]")
    else:
        t.add_row(symbol, f"[red]{size}[/]", "[red]▼ SHORT[/]",
                  f"[red]{entry}[/]", f"[red]{strat}[/]")


def build_positions_panel(state: MonitorState) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(width=5); t.add_column(width=4, justify="right")
    t.add_column(width=8); t.add_column(width=9, justify="right"); t.add_column(width=10)
    _position_row(t, SYMBOL, state.position_size, state.position_dir,
                  state.position_entry, state.position_strat)
    _position_row(t, "MNQ", state.mnq_position_size, state.mnq_position_dir,
                  state.mnq_position_entry, state.mnq_position_strat)
    return Panel(t, title="POSITIONS", border_style="blue", padding=(0, 1), expand=False)


def build_header() -> Table:
    now_et  = datetime.now(ET)
    now_loc = datetime.now(LOCAL)
    hm_et   = now_et.hour * 60 + now_et.minute
    if 9*60+30 <= hm_et < 16*60:
        sess = "[bold green]RTH[/]"
    else:
        h = datetime.now(timezone.utc).hour
        sess = ("[bold red]SETTLEMENT[/]" if SETTLE_UTC_START <= h < SETTLE_UTC_END
                else "GLOBEX")
    t = Table.grid(expand=True)
    t.add_column(ratio=1); t.add_column(ratio=1, justify="center"); t.add_column(ratio=1, justify="right")
    t.add_row(
        f"[bold]Focused Bot Monitor[/]  {sess}  BA-BRK · BA-REV",
        f"{now_loc.strftime('%H:%M:%S')}  /  {now_et.strftime('%H:%M ET')}",
        "",
    )
    return t


# Strategies the focused_bot currently trades. ORB and ORB-cls removed
# 2026-10-02 — corrected (bar.close entry) backtests showed no validated
# edge even after width/stop tuning; see project_orb_cls_strategy memory.
# Restricts the Trade Summary panel to just these — see trade_summary_panel.py.
FOCUSED_STRATEGIES = {"BA-BRK", "BA-REV"}


def render(state: MonitorState) -> Table:
    now = datetime.now(timezone.utc)
    root = Table.grid(expand=True)
    root.add_column(ratio=1)
    root.add_row(build_header())

    # ── Layout ─────────────────────────────────────────────────────────────────
    # Col 1 (ratio=1): BA-BRK · BA-REV
    # Col 2 (ratio=1): Trade Summary · Positions
    col1 = Table(box=None, show_header=False, padding=(0, 0), expand=True)
    col1.add_column(ratio=1)
    col1.add_row(build_ba_brk_panel(state, now))
    col1.add_row(build_ba_rev_panel(state, now))

    # Col 2: Trade Summary, given its own column so it isn't squeezed by col1
    col2 = Table(box=None, show_header=False, padding=(0, 0), expand=True)
    col2.add_column(ratio=1)
    col2.add_row(build_trade_summary_panel(strategies=FOCUSED_STRATEGIES))
    col2.add_row(build_positions_panel(state))

    main = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    main.add_column(ratio=1)    # col1: fills remaining space
    main.add_column(ratio=1)    # col2: fills remaining space
    main.add_row(col1, col2)
    root.add_row(main)

    return root


# ─── Position strategy detection ─────────────────────────────────────────────

def _detect_position_strategy(symbol: str = SYMBOL) -> str:
    for log_path in [Path("logs/focused_bot.log"), Path("logs/trading_bot.log")]:
        if not log_path.exists():
            continue
        try:
            with open(log_path) as f:
                lines = f.readlines()
            for line in reversed(lines[-500:]):
                if f" {symbol} " not in line:
                    continue
                if "VWASLR ORDER"  in line: return "VWASLR"
                if "PL_REV ORDER"  in line: return "PL REV"
                if "ORB_CLS ORDER" in line: return "ORB-cls"
                if "ORB ORDER"     in line: return "ORB"
                if "WALL"         in line and "ORDER" in line: return "WALL BRK"
                if "BA_BRK ORDER" in line: return "BA-BRK"
                if "BA_REV ORDER" in line: return "BA-REV"
        except Exception:
            pass
    return ""


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    client = TopstepClient()
    client.use_shared_token()
    ensure_wall_log()

    contracts = client.search_contracts(SYMBOL)
    if not contracts:
        print(f"ERROR: no contract for {SYMBOL}")
        return
    cid = str(contracts[0]["id"])

    mnq_contracts = client.search_contracts("MNQ")
    mnq_cid = str(mnq_contracts[0]["id"]) if mnq_contracts else ""
    if not mnq_cid:
        print("WARNING: no contract for MNQ — MNQ position row will stay blank")

    state = MonitorState(contract_id=cid, mnq_contract_id=mnq_cid,
                          wall_tracker=WallTracker(SYMBOL))
    fetch_1min_bars(client, state)

    # DOM reader (250ms)
    def _poll_dom():
        while True:
            try: _read_dom_db(state)
            except Exception: traceback.print_exc()
            time.sleep(0.25)
    threading.Thread(target=_poll_dom, daemon=True, name="dom-reader").start()

    # 1-min bar fetch + VWASLR EMA update (60s cadence)
    def _fetch_1min_loop():
        while True:
            time.sleep(60)
            try:
                fetch_1min_bars(client, state)
                sigma_pts = _update_vwaslr_ema(state)
                if sigma_pts:
                    state.vwaslr_sigma_pts = sigma_pts
                sig = _evaluate_vwaslr(state)
                if sig and state.vwaslr_signal is None:
                    state.vwaslr_signal = sig
                    play_alert()
                # Expire signal after 30 min
                if state.vwaslr_signal and datetime.now(timezone.utc) >= state.vwaslr_signal.expires_at:
                    state.vwaslr_signal = None
                # Update history
                if state.bars_1m:
                    fired = (sig.direction if sig and state.vwaslr_signal and
                             sig.fired_at == state.vwaslr_signal.fired_at else 0)
                    state.vwaslr_history.append((state.bars_1m[-1].ts, state.vwaslr_ema, fired))
                    if len(state.vwaslr_history) > 40:
                        state.vwaslr_history = state.vwaslr_history[-40:]
            except Exception: traceback.print_exc()
    threading.Thread(target=_fetch_1min_loop, daemon=True, name="bar-fetch").start()

    def _update_pl_rev_for(bars: list, prev_sig, entry_ts, history: list, sigma: float, now):
        """Shared logic: update history + evaluate signal for one instrument's 5s bars.
        Returns (new_sig, new_entry_ts) — unchanged values if no new signal."""
        bar_ts = bars[-1].ts if bars else None
        last_ts = prev_sig.bar_ts if prev_sig else None
        if bar_ts and bar_ts != last_ts:
            if len(bars) >= PL_WINDOW:
                window  = bars[-PL_WINDOW:]
                closes  = np.array([b.close for b in window], dtype=float)
                rets    = np.log(closes[1:] / closes[:-1])
                sum_abs = float(np.abs(rets).sum())
                if sum_abs > 0:
                    net = float(rets.sum())
                    history.append((bar_ts, abs(net)/sum_abs,
                                    "▲" if net > 0 else "▼", abs(net)*10000))
                    if len(history) > 40:
                        del history[:-40]
            new_sig = _evaluate_pl_rev(bars, prev_sig, sigma)
            if new_sig and (prev_sig is None or new_sig.bar_ts != prev_sig.bar_ts):
                play_alert()
                return new_sig, now
        return prev_sig, entry_ts

    def _expire_pl_rev(bars: list, sig, entry_ts, now):
        """Expire a PL_REV signal if max-hold elapsed or trend resumed."""
        if sig is None:
            return None, None
        expired = now >= sig.expires_at()
        pl_exit = False
        min_ok  = (entry_ts is not None and
                   (now - entry_ts).total_seconds() >= PL_MIN_HOLD_S)
        if min_ok and len(bars) >= PL_WINDOW:
            window = bars[-PL_WINDOW:]
            closes = np.array([b.close for b in window], dtype=float)
            rets   = np.log(closes[1:] / closes[:-1])
            sa     = float(np.abs(rets).sum())
            cur_pl = abs(float(rets.sum())/sa) if sa > 0 else 0.0
            pl_exit = cur_pl >= PL_RESUME_PL
        if expired or pl_exit:
            return None, None
        return sig, entry_ts

    # 5s bar poll + PL_REV eval for MES and MNQ (2s cadence, RTH only)
    def _poll_5s():
        while True:
            now_et = datetime.now(ET)
            in_rth = (9, 30) <= (now_et.hour, now_et.minute) < (16, 0)
            if in_rth:
                try:
                    now = datetime.now(timezone.utc)
                    db_ok = bars_db_available()
                    floor = datetime.fromtimestamp((int(now.timestamp())//5)*5, tz=timezone.utc)
                    start = floor - timedelta(seconds=5*(PL_5S_FETCH+5))

                    # ── MES 5s bars ───────────────────────────────────────────
                    if db_ok:
                        raw = get_5s_bars_from_db(SYMBOL, PL_5S_FETCH)
                    else:
                        raw = client.get_bars(contract_id=state.contract_id, start=start, end=floor,
                                              unit=TopstepClient.SECOND, unit_number=5, limit=PL_5S_FETCH)
                    if raw:
                        state.bars_5s = [Bar(ts=datetime.fromisoformat(b["t"]),
                                             open=b["o"], high=b["h"], low=b["l"],
                                             close=b["c"], volume=b["v"]) for b in raw]
                        state.sigma_30s_bps = _compute_sigma_5s(state.bars_5s, PL_SIGMA_LB)

                    state.pl_sig, state.pl_entry_ts = _update_pl_rev_for(
                        state.bars_5s, state.pl_sig, state.pl_entry_ts,
                        state.pl_history, state.sigma_30s_bps, now)
                    state.pl_sig, state.pl_entry_ts = _expire_pl_rev(
                        state.bars_5s, state.pl_sig, state.pl_entry_ts, now)

                    # ── MNQ 5s bars ───────────────────────────────────────────
                    if True:  # always fetch; DB path needs no contract ID
                        if db_ok:
                            raw_mnq = get_5s_bars_from_db("MNQ", PL_5S_FETCH)
                        elif state.mnq_contract_id:
                            raw_mnq = client.get_bars(contract_id=state.mnq_contract_id,
                                                      start=start, end=floor,
                                                      unit=TopstepClient.SECOND,
                                                      unit_number=5, limit=PL_5S_FETCH)
                        else:
                            raw_mnq = []
                        if raw_mnq:
                            state.bars_5s_mnq = [Bar(ts=datetime.fromisoformat(b["t"]),
                                                     open=b["o"], high=b["h"], low=b["l"],
                                                     close=b["c"], volume=b["v"]) for b in raw_mnq]
                            state.sigma_30s_bps_mnq = _compute_sigma_5s(state.bars_5s_mnq, PL_SIGMA_LB)

                        state.pl_sig_mnq, state.pl_entry_ts_mnq = _update_pl_rev_for(
                            state.bars_5s_mnq, state.pl_sig_mnq, state.pl_entry_ts_mnq,
                            state.pl_history_mnq, state.sigma_30s_bps_mnq, now)
                        state.pl_sig_mnq, state.pl_entry_ts_mnq = _expire_pl_rev(
                            state.bars_5s_mnq, state.pl_sig_mnq, state.pl_entry_ts_mnq, now)

                except Exception: traceback.print_exc()
            time.sleep(2)
    threading.Thread(target=_poll_5s, daemon=True, name="pl-rev").start()

    # Wall + ORB eval thread (5s cadence)
    def _eval_walls():
        while True:
            time.sleep(5)
            now = datetime.now(timezone.utc)
            if state.wall_tracker is not None:
                with state.dom._lock:
                    bids = dict(state.dom.bids); asks = dict(state.dom.asks)
                    bb = state.dom.best_bid;     ba = state.dom.best_ask
                events = state.wall_tracker.update(bids, asks, bb, ba, now)
                if events:
                    log_wall_events(events)
                    for ev in events:
                        if ev.event in ("test", "breakout", "wall_found", "wall_pulled"):
                            entry_price = ev.price or ev.wall_price
                            state.wall_recent_events.append(
                                (ev.ts, ev.event, ev.side, ev.wall_price,
                                 ev.test_count, entry_price, ev.peak_size))
                            if len(state.wall_recent_events) > 30:
                                state.wall_recent_events = state.wall_recent_events[-30:]
                        if ev.event == "breakout":
                            entry_price = ev.price or ev.wall_price
                            # ask wall breaks up → LONG; bid wall breaks down → SHORT
                            direction = 1 if ev.side == "ask" else -1
                            expires = now + timedelta(minutes=WALL_HOLD_MIN)
                            sig = WallBreakSignal(
                                direction=direction,
                                wall_price=ev.wall_price,
                                entry=entry_price,
                                target=entry_price + direction * WALL_TARGET_PTS,
                                stop=entry_price  - direction * WALL_STOP_PTS,
                                test_count=ev.test_count,
                                peak_size=ev.peak_size,
                                fired_at=now,
                                expires_at=expires,
                            )
                            state.wall_break_signal = sig
                            play_alert()
                # Expire wall break signal
                if (state.wall_break_signal is not None and
                        now >= state.wall_break_signal.expires_at):
                    state.wall_break_signal = None

                # BA-BRK: accumulate breakout + test events (both sides, unfiltered
                # by wall size) for display, and check for a fresh ask cascade.
                # (test events are display-only context — cascade detection below
                # filters back down to breakouts, matching trading_bot.py's live logic.)
                for ev in events:
                    if ev.event in ("test", "breakout"):
                        state.ba_brk_breakouts.append(ev)
                prune_before = now - timedelta(minutes=BA_BRK_BUFFER_MIN)
                state.ba_brk_breakouts = [e for e in state.ba_brk_breakouts if e.ts >= prune_before]

                today = now.astimezone(LOCAL).date()
                if state.day_open_date != today:
                    fresh_open = _day_open_price(state.bars_mes_orb, now)
                    if fresh_open is not None:
                        state.day_open      = fresh_open
                        state.day_open_date = today

                # Mirrors trading_bot.py's evaluate_ba_brk: check both sides (ask
                # cascades are long, bid cascades are short — bid live since
                # 2026-09-14, see BA_BRK_BID_* constants), pick whichever cascade
                # triggered most recently, same as the real bot's own tie-break.
                ask_breakouts_only = [e for e in state.ba_brk_breakouts if e.event == "breakout"]
                last_ts = state.ba_brk_signal.last_break_ts if state.ba_brk_signal else None
                brk_candidates = []
                r_ask = _detect_cascade_signal(ask_breakouts_only, "ask",
                                                BA_BRK_CASCADE_MIN, BA_BRK_MAX_GAP_SEC, last_ts)
                if r_ask:
                    brk_candidates.append(("ask", *r_ask))
                r_bid = _detect_cascade_signal(ask_breakouts_only, "bid",
                                                BA_BRK_CASCADE_MIN, BA_BRK_MAX_GAP_SEC, last_ts)
                if r_bid:
                    brk_candidates.append(("bid", *r_bid))
                if brk_candidates:
                    side, trigger, cascade_start, cascade_len = max(brk_candidates, key=lambda c: c[1].ts)
                    if (now - trigger.ts).total_seconds() <= BA_BRK_MAX_GAP_SEC + 15:
                        entry_price = trigger.price or trigger.wall_price
                        if side == "ask" and state.day_open is not None and entry_price > state.day_open:
                            state.ba_brk_signal = BaBrkSignal(
                                direction=1, side="ask", entry=entry_price,
                                target=entry_price + BA_BRK_TARGET_PTS,
                                stop=entry_price - BA_BRK_STOP_PTS,
                                cascade_len=cascade_len, first_break_ts=cascade_start,
                                last_break_ts=trigger.ts,
                                fired_at=now, expires_at=now + timedelta(minutes=BA_BRK_HOLD_MIN),
                            )
                            play_alert()
                        elif side == "bid" and state.day_open is not None and entry_price < state.day_open:
                            state.ba_brk_signal = BaBrkSignal(
                                direction=-1, side="bid", entry=entry_price,
                                target=entry_price - BA_BRK_BID_TARGET_PTS,
                                stop=entry_price + BA_BRK_BID_STOP_PTS,
                                cascade_len=cascade_len, first_break_ts=cascade_start,
                                last_break_ts=trigger.ts,
                                fired_at=now, expires_at=now + timedelta(minutes=BA_BRK_BID_HOLD_MIN),
                            )
                            play_alert()
                if (state.ba_brk_signal is not None and
                        now >= state.ba_brk_signal.expires_at):
                    state.ba_brk_signal = None

                # BA-REV: wall-rejection reversal — a wall surviving
                # BA_REV_TEST_COUNT touches without breaking. Reuses the same
                # ba_brk_breakouts buffer (already holds both test+breakout
                # events). Trading starts at 10:30 ET (no free window — see
                # trading_bot.py) and only while still contained within the
                # 07:00-10:30 reference range.
                _update_ba_rev_reference(state, now)
                _update_ba_rev_containment(state, state.bars_1m)
                now_et_hm_rev = now.astimezone(ET).hour * 60 + now.astimezone(ET).minute
                rev_eligible = now_et_hm_rev >= BA_REV_LOCK_HM and state.ba_rev_contained
                if rev_eligible:
                    rev_last_ts = state.ba_rev_signal.trigger_ts if state.ba_rev_signal else None
                    rev_candidates = []
                    for side in ("ask", "bid"):
                        e = _find_test_signal(state.ba_brk_breakouts, side,
                                               BA_REV_TEST_COUNT, rev_last_ts)
                        if e:
                            rev_candidates.append((side, e))
                    if rev_candidates:
                        rside, rtrigger = max(rev_candidates, key=lambda c: c[1].ts)
                        if (now - rtrigger.ts).total_seconds() <= BA_REV_SIGNAL_STALE_SEC:
                            rdir  = -1 if rside == "ask" else 1
                            rentry = rtrigger.price or rtrigger.wall_price
                            state.ba_rev_signal = BaRevSignal(
                                direction=rdir, side=rside, entry=rentry,
                                target=rentry + rdir * BA_REV_TARGET_PTS,
                                stop=rentry   - rdir * BA_REV_STOP_PTS,
                                test_count=rtrigger.test_count, wall_price=rtrigger.wall_price,
                                trigger_ts=rtrigger.ts,
                                fired_at=now, expires_at=now + timedelta(minutes=BA_REV_HOLD_MIN),
                            )
                            play_alert()
                if (state.ba_rev_signal is not None and
                        now >= state.ba_rev_signal.expires_at):
                    state.ba_rev_signal = None
    threading.Thread(target=_eval_walls, daemon=True, name="wall-eval").start()

    # Initial VWASLR EMA after bars loaded
    time.sleep(2)
    sigma_pts = _update_vwaslr_ema(state)
    if sigma_pts:
        state.vwaslr_sigma_pts = sigma_pts

    # Resolve account ID — same logic as trading_bot (uses TOPSTEP_ACCOUNT_ID env var)
    import os as _os
    _acct_id_env = int(_os.environ.get("TOPSTEP_ACCOUNT_ID", "0"))
    _accounts    = client.get_accounts()
    _acct        = (next((a for a in _accounts if a["id"] == _acct_id_env), None)
                    if _acct_id_env else None) or (_accounts[0] if _accounts else None)
    MONITOR_ACCOUNT_ID = _acct["id"] if _acct else None

    # Position poll (30s cadence)
    def _poll_positions():
        while True:
            try:
                account_id = MONITOR_ACCOUNT_ID
                if not account_id:
                    time.sleep(30); continue
                positions  = client.get_open_positions(account_id)
                pos = next((p for p in positions
                            if str(p.get("contractId", "")) == state.contract_id), None)
                sz  = int(pos.get("size", 0)) if pos else 0
                pt  = int(pos.get("type", 0)) if pos else 0
                prev = state.position_size
                state.position_size  = abs(sz)
                state.position_entry = (float(pos.get("averagePrice", 0) or 0)
                                        if pos and sz else None)
                state.position_dir   = 0 if sz == 0 else (-1 if pt == 2 else 1)
                if state.position_size > 0 and prev == 0:
                    state.position_strat = _detect_position_strategy(SYMBOL)
                elif state.position_size == 0:
                    state.position_strat = ""

                if state.mnq_contract_id:
                    mnq_pos = next((p for p in positions
                                     if str(p.get("contractId", "")) == state.mnq_contract_id), None)
                    mnq_sz  = int(mnq_pos.get("size", 0)) if mnq_pos else 0
                    mnq_pt  = int(mnq_pos.get("type", 0)) if mnq_pos else 0
                    mnq_prev = state.mnq_position_size
                    state.mnq_position_size  = abs(mnq_sz)
                    state.mnq_position_entry = (float(mnq_pos.get("averagePrice", 0) or 0)
                                                if mnq_pos and mnq_sz else None)
                    state.mnq_position_dir   = 0 if mnq_sz == 0 else (-1 if mnq_pt == 2 else 1)
                    if state.mnq_position_size > 0 and mnq_prev == 0:
                        state.mnq_position_strat = _detect_position_strategy("MNQ")
                    elif state.mnq_position_size == 0:
                        state.mnq_position_strat = ""
            except Exception: traceback.print_exc()
            time.sleep(30)
    threading.Thread(target=_poll_positions, daemon=True, name="pos-poll").start()

    console = Console()
    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(0.25)
            try:
                live.update(render(state))
            except Exception as e:
                traceback.print_exc()


if __name__ == "__main__":
    run()
