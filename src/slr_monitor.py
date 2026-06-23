"""
slr_monitor.py — Focused SLR Scalp monitor for MES and MNQ.

Displays:
  • SLR Scalp signal status (vol ratios, active signal)
  • Live DOM walls (bid/ask depth with wall detection)
  • Open positions
  • Recent SLR trades (MES/MNQ only)

Connects to market-hub WebSocket for real-time DOM data.
Reads 1-min bars from bars.db (with REST API fallback) for SLR evaluation.

Usage:
    python src/slr_monitor.py
"""

import csv
import json
import math
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.live import Live
from rich.console import Group as RichGroup
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, "src")
load_dotenv()

from topstep_client import TopstepClient
from topstep_client import get_bars_from_db, get_5s_bars_from_db, bars_db_available
from trade_summary_panel import build_trade_summary_panel
from wall_tracker import WallTracker, log_wall_events, ensure_wall_log

# ── Timezone ──────────────────────────────────────────────────────────────────
ET  = ZoneInfo("America/New_York")
MST = ZoneInfo("America/Phoenix")

# ── Instruments ───────────────────────────────────────────────────────────────
INSTRUMENTS = ["MES", "MNQ"]
POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}

# ── SLR constants (keep in sync with trading_bot.py) ─────────────────────────
SLR_VOL_LOOKBACK = 20      # bars for rolling median
SLR_VOL_MULT     = 7.0     # minimum vol surge multiplier
SLR_MOVE_BPS     = 12      # minimum directional move in basis points
SLR_TARGET_BPS   = 15      # profit target
SLR_STOP_BPS     = 10      # minimum stop (basis points)
SLR_HOLD_RTH     = 15      # minutes to hold in RTH
SLR_HOLD_GLOBEX  = 10      # minutes to hold overnight
SLR_BARS_FETCH   = 200     # 1-min bars to maintain

# ── DOM constants ─────────────────────────────────────────────────────────────
WALL_MULT        = 2.5     # size > WALL_MULT × median → highlight as wall
DOM_DB_PATH      = Path("data/dom.db")
DOM_DB_STALE_S   = 10      # seconds before DOM is considered stale

# Hybrid DOM layout per instrument:
#   TICK_SIZE      — minimum price increment
#   NEAR_LEVELS    — tick-by-tick rows near the spread (each side)
#   BUCKET_PTS     — point size of aggregated buckets beyond the near zone
#   BUCKET_COUNT   — number of aggregated buckets beyond the near zone
DOM_CONFIG = {
    "MES": dict(tick=0.25, near_levels=4, bucket_pts=1.0,  bucket_count=10, price_col=10, near_cap=100, bucket_cap=500),
    "MNQ": dict(tick=0.25, near_levels=4, bucket_pts=5.0,  bucket_count=10, price_col=14, near_cap=25,  bucket_cap=125),
}

# ── Sizing constants ──────────────────────────────────────────────────────────
SIZING_SIGMA_BARS = 100   # 1-min bars for sigma window (100-min ≈ signal_monitor's 20×5-min)
SIZING_RISKS      = [100, 200, 300, 400, 500]

# ── CME settlement gap ────────────────────────────────────────────────────────
SETTLE_UTC_START = 21
SETTLE_UTC_END   = 22

# ── Paths ─────────────────────────────────────────────────────────────────────
SLR_TRADES_PATH = Path("logs/slr_trades.csv")

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts:     datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class SLRSignal:
    direction: int      # +1 LONG, -1 SHORT
    entry:     float
    target:    float
    stop:      float
    vol_ratio: float
    move_bps:  float
    fired_at:  datetime
    expires_at: datetime
    is_rth:    bool


@dataclass
class DOMBook:
    """Minimal real-time DOM book updated via WebSocket."""
    bids:       dict = field(default_factory=dict)   # price → size
    asks:       dict = field(default_factory=dict)
    last_price: float | None = None
    best_bid:   float | None = None
    best_ask:   float | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _lock:      threading.Lock = field(default_factory=threading.Lock)

    BID   = 4
    ASK   = 3
    RESET = 6

    def apply_depth(self, updates: list):
        with self._lock:
            for u in updates:
                price  = float(u.get("price",  0))
                volume = float(u.get("volume", 0))
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

    def hybrid_snapshot(self, near_levels: int, bucket_pts: float, bucket_count: int):
        """
        Return (bid_rows, ask_rows, last, bb, ba, updated) where each row is
        (price_label, size, is_bucket) — price_label is a float for tick rows
        or a string like "7260–7261" for bucket rows.
        Total rows each side = near_levels + bucket_count (fixed).
        """
        with self._lock:
            bb, ba = self.best_bid, self.best_ask
            all_bids = sorted(((p,s) for p,s in self.bids.items()
                                if bb is None or ba is None or p < ba), reverse=True)
            all_asks = sorted((p,s) for p,s in self.asks.items()
                               if bb is None or ba is None or p > bb)
            last, updated = self.last_price, self.last_update

        def build_side(levels, is_ask: bool):
            rows = []
            # Near zone: first near_levels tick-by-tick entries
            near = levels[:near_levels]
            for p, s in near:
                rows.append((p, s, False))
            # Pad near zone if fewer ticks available
            while len(rows) < near_levels:
                rows.append((None, 0.0, False))

            # Far zone: aggregate into bucket_pts buckets
            far = levels[near_levels:]
            if far:
                far_start = far[0][0]
                # Align bucket boundary to bucket_pts grid
                if is_ask:
                    grid_start = (far_start // bucket_pts) * bucket_pts
                else:
                    grid_start = (far_start // bucket_pts) * bucket_pts
                buckets = {}
                for p, s in far:
                    idx = int((p - grid_start) // bucket_pts)
                    buckets[idx] = buckets.get(idx, 0.0) + s
                if is_ask:
                    sorted_keys = sorted(buckets.keys())
                else:
                    sorted_keys = sorted(buckets.keys(), reverse=True)
                for k in sorted_keys[:bucket_count]:
                    lo = grid_start + k * bucket_pts
                    hi = lo + bucket_pts
                    label = f"{lo:.0f}–{hi:.0f}"
                    rows.append((label, buckets[k], True))

            # Pad far zone to bucket_count
            while len(rows) < near_levels + bucket_count:
                rows.append((None, 0.0, True))

            return rows

        bid_rows = build_side(all_bids, is_ask=False)
        ask_rows = build_side(all_asks, is_ask=True)
        return bid_rows, ask_rows, last, bb, ba, updated


@dataclass
class InstrState:
    symbol:      str
    contract_id: str
    bars:        list = field(default_factory=list)     # 1-min Bar list
    dom:         DOMBook = field(default_factory=DOMBook)
    signal:      SLRSignal | None = None
    last_surge_ts: datetime | None = None
    position_size: int = 0
    position_dir:  int = 0       # +1 LONG, -1 SHORT, 0 flat
    position_entry: float | None = None
    position_strategy: str = ""
    bars_fetch_min: int = -1
    # Wall tracker (initialised after symbol is known — see run())
    wall_tracker: WallTracker | None = None


# ── DOM DB reader (populated by dom_client.py) ────────────────────────────────

def _read_dom_db(state_list: list) -> None:
    """Read live DOM state from data/dom.db (written by dom_client) and
    update each InstrState.dom in place. No-ops silently if DB is absent."""
    if not DOM_DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(str(DOM_DB_PATH), timeout=1.0)
        syms = {s.symbol: s for s in state_list}
        rows = conn.execute(
            "SELECT symbol, updated, last_price, best_bid, best_ask, "
            "bids_json, asks_json FROM dom_live WHERE symbol IN ({})".format(
                ",".join("?" * len(syms))),
            list(syms.keys()),
        ).fetchall()
        conn.close()
        now = datetime.now(timezone.utc)
        for sym, updated, last, bb, ba, bids_json, asks_json in rows:
            state = syms.get(sym)
            if not state:
                continue
            updated_dt = datetime.fromisoformat(updated)
            if (now - updated_dt).total_seconds() > DOM_DB_STALE_S:
                continue   # stale — leave existing book intact
            bids = {float(k): v for k, v in json.loads(bids_json).items()}
            asks = {float(k): v for k, v in json.loads(asks_json).items()}
            with state.dom._lock:
                state.dom.bids        = bids
                state.dom.asks        = asks
                state.dom.last_price  = last
                state.dom.best_bid    = bb
                state.dom.best_ask    = ba
                state.dom.last_update = updated_dt
    except Exception:
        pass


# ── SLR evaluation ────────────────────────────────────────────────────────────

def _in_settlement(ts: datetime) -> bool:
    h = ts.astimezone(timezone.utc).hour
    return SETTLE_UTC_START <= h < SETTLE_UTC_END


def evaluate_slr(state: InstrState) -> SLRSignal | None:
    bars = state.bars
    if len(bars) < SLR_VOL_LOOKBACK + 2:
        return None

    surge     = bars[-1]
    surge_idx = len(bars) - 1

    if _in_settlement(surge.ts):
        return None
    if state.last_surge_ts and surge.ts <= state.last_surge_ts:
        return None

    prior_vols = [bars[j].volume for j in range(surge_idx - SLR_VOL_LOOKBACK, surge_idx)]
    med_vol    = float(np.median(prior_vols)) if prior_vols else 0.0
    if med_vol <= 0:
        return None
    vol_ratio = surge.volume / med_vol
    if vol_ratio < SLR_VOL_MULT:
        return None

    if surge.close > surge.open:    direction = 1
    elif surge.close < surge.open:  direction = -1
    else:                           return None

    prev = bars[surge_idx - 1] if surge_idx > 0 else None
    if (not prev or not prev.open
            or (surge.ts - prev.ts) > timedelta(minutes=2)
            or abs(surge.close / prev.open - 1) > 0.05):
        return None

    move_bps = (surge.close - prev.open) / prev.open * 10000 * direction
    if move_bps < SLR_MOVE_BPS:
        return None

    entry     = surge.close
    stop_bps  = max(SLR_STOP_BPS, SLR_STOP_BPS)   # could add vol-based here
    target    = entry * (1 + direction * SLR_TARGET_BPS / 10000)
    stop      = entry * (1 - direction * stop_bps   / 10000)

    surge_et  = surge.ts.astimezone(ET)
    is_rth    = (9, 30) <= (surge_et.hour, surge_et.minute) < (16, 0)
    hold      = SLR_HOLD_RTH if is_rth else SLR_HOLD_GLOBEX

    return SLRSignal(
        direction=direction, entry=entry, target=target, stop=stop,
        vol_ratio=vol_ratio, move_bps=move_bps,
        fired_at=surge.ts, expires_at=surge.ts + timedelta(minutes=hold),
        is_rth=is_rth,
    )


# ── Bar fetching ──────────────────────────────────────────────────────────────

def fetch_bars(client: TopstepClient, state: InstrState):
    now_utc  = datetime.now(timezone.utc)
    now_floor = datetime.fromtimestamp(
        (int(now_utc.timestamp()) // 60) * 60, tz=timezone.utc)
    db_fresh = False
    if bars_db_available():
        raw = get_bars_from_db(state.symbol, 1, SLR_BARS_FETCH + 1)
        db_bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                       open=b["o"], high=b["h"], low=b["l"],
                       close=b["c"], volume=b["v"])
                   for b in raw
                   if datetime.fromisoformat(b["t"]) < now_floor]
        if db_bars:
            if not state.bars or db_bars[-1].ts >= state.bars[-1].ts:
                state.bars = db_bars
            db_fresh = (now_utc - db_bars[-1].ts).total_seconds() < 180
    if not db_fresh:
        try:
            end   = now_utc
            start = end - timedelta(minutes=SLR_BARS_FETCH + 30)
            raw   = client.get_bars(
                contract_id=state.contract_id, start=start, end=end,
                unit=TopstepClient.MINUTE, unit_number=1,
                limit=SLR_BARS_FETCH)
            state.bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                               open=b["o"], high=b["h"], low=b["l"],
                               close=b["c"], volume=b["v"])
                           for b in reversed(raw)]
        except Exception:
            pass


# ── DOM panel builder ─────────────────────────────────────────────────────────

def build_dom_panel(state: InstrState) -> Panel:
    cfg = DOM_CONFIG.get(state.symbol, DOM_CONFIG["MES"])
    near   = cfg["near_levels"]
    bkpt   = cfg["bucket_pts"]
    bkcnt  = cfg["bucket_count"]

    # Build lookup: price → test_count for active walls
    wall_tests: dict[float, int] = {}
    if state.wall_tracker is not None:
        for w in state.wall_tracker.active_walls():
            if w.test_count > 0:
                wall_tests[w.price] = w.test_count

    bid_rows, ask_rows, last, bb, ba, updated = state.dom.hybrid_snapshot(
        near_levels=near, bucket_pts=bkpt, bucket_count=bkcnt)

    # Collect all non-zero sizes for bar scaling and wall detection
    all_sizes = [s for _, s, _ in bid_rows + ask_rows if s > 0]
    max_sz = max(all_sizes) if all_sizes else 1
    med_sz = float(np.median(all_sizes)) if all_sizes else 1

    def sz_bar(size, is_bid, is_bucket):
        cap    = cfg["bucket_cap"] if is_bucket else cfg["near_cap"]
        width  = 24 if is_bucket else 28
        filled = max(1, int(round(min(size, cap) / cap * width))) if size > 0 else 0
        ch     = ("▒" if is_bucket else "█") * filled + "░" * (width - filled)
        color  = "green" if is_bid else "red"
        return f"[{color}]{ch}[/]"

    def is_wall(size):
        return size >= med_sz * WALL_MULT

    def bp_dist(price, ref):
        """Distance from ref price in basis points."""
        if price is None or ref is None or ref == 0:
            return None
        return abs(price - ref) / ref * 10000

    price_col = cfg.get("price_col", 10)
    t = Table(box=None, show_header=False, padding=(0, 0))
    t.add_column(justify="right",  width=6)           # bid size
    t.add_column(justify="right",  width=30)          # bid bar
    t.add_column(justify="center", width=price_col)   # price
    t.add_column(justify="left",   width=30)          # ask bar
    t.add_column(justify="left",   width=6)           # ask size

    # Ask rows: bucket rows first (furthest), then near rows closest to spread
    # ask_rows is ordered: near[0]=closest … near[n-1], bucket[0]=next … bucket[m-1]
    # We display asks top-to-bottom: buckets reversed (furthest at top), then near reversed
    bucket_asks = list(reversed(ask_rows[near:near+bkcnt]))
    near_asks   = list(reversed(ask_rows[:near]))

    for label, size, is_bucket in bucket_asks + near_asks:
        if label is None:
            t.add_row("", "", "", "", "")
            continue
        wall  = is_wall(size) and size > 0
        style = "bold red" if wall else ("red" if size > 0 else "dim")
        bar   = sz_bar(size, False, is_bucket)
        wm    = "◀" if wall else " "
        tc    = wall_tests.get(label) if isinstance(label, float) else None
        tc_txt = f"[yellow]T{tc}[/]" if tc else ""
        if is_bucket:
            price_txt = f"[{style}]{label}[/]"
            size_txt  = f"[{style}]{size:.0f}{wm}[/]{tc_txt}" if size > 0 else ""
        else:
            price_txt = f"[{style}]{label:.2f}[/]"
            dist      = bp_dist(label, bb)
            size_txt  = f"[{style}]{size:.0f}{wm}[/]{tc_txt}" if size > 0 else ""
        t.add_row("", "", price_txt, bar, size_txt)

    # Spread row with imbalance
    spread = f"{ba - bb:.2f}" if bb and ba else "—"
    if all_sizes:
        total_bid = sum(s for _, s, _ in bid_rows if s > 0)
        total_ask = sum(s for _, s, _ in ask_rows if s > 0)
        denom     = total_bid + total_ask
        imbal     = (total_bid - total_ask) / denom if denom else 0
        imbal_col = "green" if imbal > 0.1 else ("red" if imbal < -0.1 else "dim")
        imbal_txt = f"  [{imbal_col}]imb {imbal:+.2f}[/]"
    else:
        imbal_txt = ""
    t.add_row("", f"[dim]spread {spread}[/]{imbal_txt}", "", "", "")

    # Bid rows: near first (closest to spread), then buckets
    near_bids   = bid_rows[:near]
    bucket_bids = bid_rows[near:near+bkcnt]

    for label, size, is_bucket in near_bids + bucket_bids:
        if label is None:
            t.add_row("", "", "", "", "")
            continue
        wall  = is_wall(size) and size > 0
        style = "bold green" if wall else ("green" if size > 0 else "dim")
        bar   = sz_bar(size, True, is_bucket)
        wm    = "◀" if wall else " "
        tc    = wall_tests.get(label) if isinstance(label, float) else None
        tc_txt = f"[yellow]T{tc}[/]" if tc else ""
        if is_bucket:
            price_txt = f"[{style}]{label}[/]"
            size_txt  = f"{tc_txt}[{style}]{wm}{size:.0f}[/]" if size > 0 else ""
        else:
            price_txt = f"[{style}]{label:.2f}[/]"
            size_txt  = f"{tc_txt}[{style}]{wm}{size:.0f}[/]" if size > 0 else ""
        t.add_row(size_txt, bar, price_txt, "", "")

    # Wall summary — find largest resting order each side, show bp distance
    def best_wall(rows, ref, is_ask):
        candidates = [(lbl, sz) for lbl, sz, _ in rows
                      if sz > 0 and isinstance(lbl, float)]
        if not candidates:
            candidates = [(lbl, sz) for lbl, sz, _ in rows if sz > 0]
        if not candidates:
            return None
        lbl, sz = max(candidates, key=lambda x: x[1])
        if isinstance(lbl, float) and ref:
            dist_pts = abs(lbl - ref)
            return f"{sz:.0f} @ {dist_pts:.2f}pts"
        return f"{sz:.0f}"

    wbid = best_wall(bid_rows, bb, False)
    wask = best_wall(ask_rows, ba, True)
    if wbid or wask:
        t.add_row("", f"bid wall {wbid or '—'}",
                  "", f"ask wall {wask or '—'}", "")

    age      = (datetime.now(timezone.utc) - updated).total_seconds()
    age_col  = "green" if age < 10 else ("yellow" if age < 30 else "red")
    last_str = f"{last:.2f}" if last else "—"
    title    = f"DOM  {state.symbol}  [{age_col}]{last_str}[/]  [dim]{age:.0f}s ago[/]"
    return Panel(t, title=title, border_style="blue", padding=(0, 1), expand=True)


# ── Visual bar helpers ────────────────────────────────────────────────────────

def _centered_bar(ratio: float, is_long: bool, half: int = 6) -> str:
    """
    Bidirectional bar centered on '|'. Total width = 2*half + 1.
    ratio: 0–1 fill fraction.  is_long: True → green right, False → red left.
    Example (half=6):  '      |██████'  or  '██████|      '
    """
    filled = max(0, min(half, round(ratio * half)))
    empty  = half - filled
    if is_long:
        return f"[dim]{'░' * half}|[/][green]{'█' * filled}{'░' * empty}[/]"
    else:
        return f"[red]{'░' * empty}{'█' * filled}[/][dim]|{'░' * half}[/]"


def _vol_bar(vr: float, move_bps: float) -> str:
    """Centered bar: vol ratio vs 2× threshold, direction from move_bps sign."""
    ratio   = min(1.0, vr / (SLR_VOL_MULT * 2))
    is_long = move_bps >= 0
    if vr < SLR_VOL_MULT * 0.6:
        half = 6
        return f"[dim]{'░' * half}|{'░' * half}[/]" if vr == 0 else _centered_bar(ratio, is_long)
    return _centered_bar(ratio, is_long)


def _pl_bar(pl: float, dir_sym: str) -> str:
    """Centered bar: PL 0–1, direction from dir_sym (▲=long, ▼=short)."""
    ratio   = min(1.0, pl)
    is_long = dir_sym == "▲"
    return _centered_bar(ratio, is_long)


# ── SLR Scalp panel builder ───────────────────────────────────────────────────

def build_slr_panel(state: InstrState, now: datetime) -> Panel:
    bars = state.bars
    sig  = state.signal

    # Active signal
    if sig:
        remaining = int((sig.expires_at - now).total_seconds())
        if remaining < 0: remaining = 0
        dirn  = "▲ LONG" if sig.direction == 1 else "▼ SHORT"
        color = "green"  if sig.direction == 1 else "red"
        sess  = "RTH" if sig.is_rth else "GLOBEX"

        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column(width=9); t.add_column()
        t.add_row("Entry:",   f"[bold {color}]{sig.entry:.2f}[/]")
        t.add_row("Target:",  f"[bold green]{sig.target:.2f}[/]  "
                              f"[dim]({abs(sig.target-sig.entry):.2f}pts)[/]")
        t.add_row("Stop:",    f"[bold red]{sig.stop:.2f}[/]  "
                              f"[dim]({abs(sig.stop-sig.entry):.2f}pts)[/]")
        t.add_row("Expires:", f"[dim]{remaining}s  {sess}[/]")
        t.add_row("Trigger:", f"[dim]vol={sig.vol_ratio:.1f}×  move={sig.move_bps:.1f}bp[/]")
        border = color
        title  = f"[bold {color}]⬤  SLR  {dirn}  {state.symbol}[/]"
        return Panel(t, title=title, border_style=border, padding=(0,1), expand=True)

    # No signal — show recent vol ratios
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(width=9)    # label
    t.add_column(width=13)   # centered vol bar (6+1+6)
    t.add_column(no_wrap=True)  # stats

    if len(bars) >= SLR_VOL_LOOKBACK + 2:
        tz_local = datetime.now().astimezone().tzinfo
        for i in range(1, 7):
            idx = len(bars) - i
            if idx < SLR_VOL_LOOKBACK: break
            b        = bars[idx]
            prior    = [bars[j].volume for j in range(idx - SLR_VOL_LOOKBACK, idx)]
            med      = float(np.median(prior)) if prior else 0.0
            vr       = b.volume / med if med > 0 else 0.0
            prev_b   = bars[idx-1] if idx > 0 else None
            move_bps = 0.0
            if prev_b and prev_b.open and (b.ts - prev_b.ts) <= timedelta(minutes=2):
                move_bps = (b.close - prev_b.open) / prev_b.open * 10000
            bar_time = (b.ts + timedelta(minutes=1)).astimezone(tz_local).strftime("%H:%M")
            lbl      = ("Latest:" if i == 1 else f"  -{i}m:")
            vr_num   = f"{vr:4.1f}"
            if vr >= SLR_VOL_MULT:
                vr_str = f"[bold green]{vr_num}×[/]"
            elif vr >= SLR_VOL_MULT * 0.6:
                vr_str = f"[yellow]{vr_num}×[/]"
            else:
                vr_str = f"[dim]{vr_num}×[/]"
            vol_num  = f"{int(b.volume):>5}"
            bp_col   = ("bold green" if move_bps >= SLR_MOVE_BPS
                        else ("bold red" if move_bps <= -SLR_MOVE_BPS else ""))
            bp_str   = (f"[{bp_col}]{move_bps:+6.1f}bp[/]" if bp_col
                        else f"{move_bps:+6.1f}bp")
            t.add_row(lbl, _vol_bar(vr, move_bps),
                      f"{vr_str}  [dim]{vol_num}[/]  {bp_str}  [dim]{bar_time}[/]")
    else:
        t.add_row("Status:", "", "[dim]warming up…[/]")

    t.add_row("", "", f"[dim]{SLR_VOL_MULT:.0f}× vol threshold  {SLR_MOVE_BPS:.0f}bp surge[/]")
    return Panel(t, title=f"SLR SCALP  {state.symbol}", border_style="blue",
                 padding=(0,1), expand=True)


# ── Sizing panel ─────────────────────────────────────────────────────────────

def _compute_sigma_pts(bars: list) -> tuple[float | None, int]:
    """Return (sigma_pts, bar_count_used) from 1-min bar closes."""
    if len(bars) < 2:
        return None, 0
    recent = bars[-min(SIZING_SIGMA_BARS, len(bars)):]
    closes = [b.close for b in recent]
    lrs = [np.log(closes[i] / closes[i-1])
           for i in range(1, len(closes)) if closes[i-1] > 0]
    if len(lrs) < 2:
        return None, 0
    sigma_frac = float(np.std(lrs, ddof=1))
    return sigma_frac * closes[-1], len(recent)


def build_sizing_panel(states: list) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    t.add_column("", justify="right")
    for s in states:
        t.add_column(s.symbol, justify="right", style="bold")

    sigma_data = [_compute_sigma_pts(s.bars) for s in states]
    partial = any(cnt > 0 and cnt < SIZING_SIGMA_BARS for _, cnt in sigma_data)

    bp_vals = []
    for i, (sp, _) in enumerate(sigma_data):
        close = states[i].bars[-1].close if states[i].bars else None
        if sp and close:
            bp_vals.append(f"{2 * sp / close * 10000:.1f}")
        else:
            bp_vals.append("—")
    t.add_row("bp", *bp_vals, style="cyan")

    stop_vals = []
    for sp, _ in sigma_data:
        stop_vals.append(f"{2 * sp:.2f}" if sp else "—")
    t.add_row("pts", *stop_vals, style="bold cyan")

    t.add_row("RISK", *[""] * len(states), style="")
    for risk in SIZING_RISKS:
        cells = []
        for i, s in enumerate(states):
            sp, _ = sigma_data[i]
            if sp and sp > 0:
                lots = risk / (2 * sp * POINT_VALUE[s.symbol])
                cells.append(f"{lots:.1f}")
            else:
                cells.append("—")
        t.add_row(f"${risk}", *cells)

    title = "[bold]SIZING (2σ stop)[/]"
    if partial:
        title += "  [dim]* partial[/]"
    return Panel(t, title=title, border_style="blue", padding=(0, 1), expand=False)


# ── Positions panel ───────────────────────────────────────────────────────────

def _detect_position_strategy(symbol: str) -> str:
    """Scan the tail of trading_bot.log for the most recent order for this symbol."""
    log_path = Path("logs/trading_bot.log")
    if not log_path.exists():
        return ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines[-500:]):
            if f" {symbol} " not in line:
                continue
            if "VWASLR ORDER" in line: return "VWASLR"
            if "SLR ORDER"    in line: return "SLR"
            if "PL MOM ORDER" in line: return "PL MOM"
            if "ORB ORDER"    in line: return "ORB"
            if "ORDER PLACED" in line: return "CSR"
    except Exception:
        pass
    return ""


def build_positions_panel(states: list[InstrState]) -> Panel:
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(width=5)
    t.add_column(width=4, justify="right")
    t.add_column(width=8)
    t.add_column(width=9, justify="right")
    t.add_column(width=7)
    for s in states:
        sz    = s.position_size
        d     = s.position_dir
        entry = f"{s.position_entry:.2f}" if s.position_entry else ""
        strat = s.position_strategy
        if sz == 0:
            t.add_row(s.symbol, "—", "", "", "")
        elif d == 1:
            t.add_row(s.symbol, f"[green]{sz}[/]",
                      "[green]▲ LONG[/]", f"[green]{entry}[/]", f"[green]{strat}[/]")
        else:
            t.add_row(s.symbol, f"[red]{sz}[/]",
                      "[red]▼ SHORT[/]", f"[red]{entry}[/]", f"[red]{strat}[/]")
    return Panel(t, title="POSITIONS", border_style="blue",
                 padding=(0,1), expand=False)


# ── Recent trades panel ───────────────────────────────────────────────────────

# ── Header ────────────────────────────────────────────────────────────────────

def build_header(states: list[InstrState]) -> Table:
    now_mst  = datetime.now(MST)
    now_et   = datetime.now(ET)
    hm_et    = now_et.hour * 60 + now_et.minute
    if (9*60+30) <= hm_et < 16*60:
        sess_str = "[bold green]RTH[/]"
    elif 21*60 <= hm_et or hm_et < 16*60+30:
        h = datetime.now(timezone.utc).hour
        if SETTLE_UTC_START <= h < SETTLE_UTC_END:
            sess_str = "[bold red]SETTLEMENT[/]"
        else:
            sess_str = "[dim]GLOBEX[/]"
    else:
        sess_str = "[dim]—[/]"

    bars_age = ""
    for s in states:
        if s.bars:
            age = int((datetime.now(timezone.utc) - s.bars[-1].ts).total_seconds())
            bars_age = f"  [dim]bars {age}s ago[/]"
            break

    t = Table.grid(expand=True)
    t.add_column(ratio=1)
    t.add_column(ratio=1, justify="center")
    t.add_column(ratio=1, justify="right")
    t.add_row(
        f"[bold]SLR Scalp Monitor[/]  {sess_str}",
        f"[dim]{now_mst.strftime('%H:%M:%S MST')}  /  {now_et.strftime('%H:%M ET')}[/]",
        bars_age,
    )
    return t


# ── Full render ───────────────────────────────────────────────────────────────

def render(states: list[InstrState]) -> Table:
    now = datetime.now(timezone.utc)

    root = Table.grid(expand=True)
    root.add_column(ratio=1)

    # Header
    root.add_row(build_header(states))

    # Row 1: [MES stack | MNQ stack] — full width, no right column
    mes, mnq = states[0], states[1]
    mes_col = RichGroup(build_slr_panel(mes, now), build_dom_panel(mes))
    mnq_col = RichGroup(build_slr_panel(mnq, now), build_dom_panel(mnq))

    main = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    main.add_column(ratio=1)
    main.add_column(ratio=1)
    main.add_row(mes_col, mnq_col)
    root.add_row(main)

    # Row 2: [Trade Summary + Positions stacked | Sizing]
    left_col = RichGroup(build_trade_summary_panel(), build_positions_panel(states))
    bottom = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    bottom.add_column()        # trade summary + positions
    bottom.add_column()        # sizing
    bottom.add_row(left_col, build_sizing_panel(states))
    root.add_row(bottom)

    return root


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    client = TopstepClient()
    client.use_shared_token()  # reuse bar_collector's token — avoids multiple-sessions disconnect

    ensure_wall_log()

    # Resolve contracts
    states: dict[str, InstrState] = {}
    for sym in INSTRUMENTS:
        contracts = client.search_contracts(sym)
        if not contracts:
            print(f"ERROR: no contract found for {sym}")
            sys.exit(1)
        cid = contracts[0]["id"]
        states[sym] = InstrState(symbol=sym, contract_id=cid,
                                 wall_tracker=WallTracker(sym))

    state_list = [states[s] for s in INSTRUMENTS]

    # Initial bar fetch
    for s in state_list:
        fetch_bars(client, s)

    # DOM polling thread — reads from data/dom.db (written by dom_client.py)
    # No direct WebSocket connection; dom_client holds the one market hub session.
    def _poll_dom():
        while True:
            _read_dom_db(state_list)
            time.sleep(0.25)

    threading.Thread(target=_poll_dom, daemon=True, name="dom-reader").start()

    # Position polling thread
    def _poll_positions():
        while True:
            try:
                account_id = client.get_accounts()[0]["id"]
                positions  = client.get_open_positions(account_id)
                cid_map    = {s.contract_id: s for s in state_list}
                for s in state_list:
                    pos = next((p for p in positions
                                if str(p.get("contractId","")) == str(s.contract_id)), None)
                    sz  = int(pos.get("size",0)) if pos else 0
                    pt  = int(pos.get("type",0)) if pos else 0
                    prev_size        = s.position_size
                    s.position_size  = abs(sz)
                    s.position_entry = float(pos.get("averagePrice",0) or 0) if pos and sz else None
                    s.position_dir   = 0 if sz == 0 else (-1 if pt == 2 else 1)
                    if s.position_size > 0 and prev_size == 0:
                        s.position_strategy = _detect_position_strategy(s.symbol)
                    elif s.position_size == 0:
                        s.position_strategy = ""
            except: pass
            time.sleep(30)

    threading.Thread(target=_poll_positions, daemon=True).start()

    # Bar fetch thread (every minute)
    def _fetch_bars_loop():
        while True:
            time.sleep(60)
            for s in state_list:
                try: fetch_bars(client, s)
                except: pass

    threading.Thread(target=_fetch_bars_loop, daemon=True).start()

    # SLR evaluation thread (every 5 seconds)
    def _evaluate_loop():
        while True:
            time.sleep(5)
            now = datetime.now(timezone.utc)
            for s in state_list:
                # ── Wall tracker update ───────────────────────────────────
                if s.wall_tracker is not None:
                    with s.dom._lock:
                        bids = dict(s.dom.bids)
                        asks = dict(s.dom.asks)
                        bb   = s.dom.best_bid
                        ba   = s.dom.best_ask
                    events = s.wall_tracker.update(bids, asks, bb, ba, now)
                    if events:
                        log_wall_events(events)
                # ── SLR: expire / resolve active signal ──────────────────────
                if s.signal:
                    if now >= s.signal.expires_at:
                        s.last_surge_ts = s.signal.fired_at
                        s.signal = None
                    elif s.bars:
                        last_close = s.bars[-1].close
                        if (s.signal.direction == 1 and last_close >= s.signal.target):
                            s.last_surge_ts = s.signal.fired_at
                            s.signal = None
                        elif (s.signal.direction == -1 and last_close <= s.signal.target):
                            s.last_surge_ts = s.signal.fired_at
                            s.signal = None
                        elif (s.signal.direction == 1 and last_close <= s.signal.stop):
                            s.last_surge_ts = s.signal.fired_at
                            s.signal = None
                        elif (s.signal.direction == -1 and last_close >= s.signal.stop):
                            s.last_surge_ts = s.signal.fired_at
                            s.signal = None
                # Evaluate new SLR signal
                if not s.signal and s.bars:
                    sig = evaluate_slr(s)
                    if sig:
                        s.signal = sig
                        s.last_surge_ts = sig.fired_at

    threading.Thread(target=_evaluate_loop, daemon=True).start()

    # Give connections time to establish
    time.sleep(3)

    console = Console()
    with Live(render(state_list), console=console, refresh_per_second=4,
              screen=True) as live:
        while True:
            time.sleep(0.25)
            live.update(render(state_list))


if __name__ == "__main__":
    run()
