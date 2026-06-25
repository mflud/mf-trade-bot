"""
mes_monitor.py — Unified MES monitor.

Displays:
  • PL_MOM signal (5s bars) — BUY LONG / SELL SHORT / NO TRADE
  • SLR Scalp signal (1-min bars) — vol surge scanner
  • Live DOM (bid/ask depth from dom.db)
  • Trade summary (all strategies, from trade_summary_panel)
  • Sizing table (2σ stop lot sizes)
  • Open position

Usage:
    python src/mes_monitor.py
"""

import json
import sqlite3
import subprocess
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
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, "src")
load_dotenv()

from topstep_client import TopstepClient, get_bars_from_db, get_5s_bars_from_db, bars_db_available
from trade_summary_panel import build_trade_summary_panel
from wall_tracker import WallTracker, log_wall_events, ensure_wall_log

# ── Timezone ───────────────────────────────────────────────────────────────────
ET    = ZoneInfo("America/New_York")
LOCAL = datetime.now().astimezone().tzinfo

# ── Instrument ─────────────────────────────────────────────────────────────────
SYMBOL      = "MES"
POINT_VALUE = 5.0

# ── SLR constants (keep in sync with trading_bot.py) ─────────────────────────
SLR_VOL_LOOKBACK = 20
SLR_VOL_MULT     = 7.0
SLR_MOVE_BPS     = 12.0
SLR_TARGET_BPS   = 15.0
SLR_STOP_BPS     = 10.0
SLR_HOLD_RTH     = 15
SLR_HOLD_GLOBEX  = 10
SLR_BARS_FETCH   = 200

# ── PL_MOM constants (keep in sync with trading_bot.py / pl_monitor.py) ───────
PL_MOM_WINDOW     = 6
PL_MOM_ENTRY_PL   = 0.80
PL_MOM_MOVE_BPS   = 8.0     # floor
PL_MOM_EXIT_PL    = 0.40
PL_MOM_STOP_BPS   = 7.0
PL_MOM_MIN_HOLD_S = 10
PL_MOM_MAX_HOLD_S = 120
PL_MOM_5S_FETCH   = 130
PL_MOM_SIGMA_N    = 3.0
PL_MOM_SIGMA_LB   = 120
HISTORY_BARS      = 24
CLOSE_TRD_SHOW_S  = 8

# ── DOM constants ─────────────────────────────────────────────────────────────
WALL_MULT      = 2.5
DOM_DB_PATH    = Path("data/dom.db")
DOM_DB_STALE_S = 10
DOM_NEAR       = 4
DOM_BUCKET_PT  = 1.0
DOM_BUCKET_N   = 10
DOM_NEAR_CAP   = 100
DOM_BKT_CAP    = 500
DOM_PRICE_COL  = 10
DOM_NEAR_BAR_W = 16    # narrowed from 28 to fit left-column layout
DOM_BKT_BAR_W  = 14   # narrowed from 24

# ── Sizing constants ──────────────────────────────────────────────────────────
SIZING_SIGMA_BARS = 100
SIZING_RISKS      = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# ── CME settlement gap ────────────────────────────────────────────────────────
SETTLE_UTC_START = 21
SETTLE_UTC_END   = 22

# ── Alert ─────────────────────────────────────────────────────────────────────
ALERT          = "/System/Library/Sounds/Pop.aiff"
ALERT_COOLDOWN = 15.0
_last_alert    = 0.0


def play_alert():
    global _last_alert
    t = time.time()
    if t - _last_alert < ALERT_COOLDOWN:
        return
    _last_alert = t
    threading.Thread(
        target=lambda: subprocess.run(["afplay", ALERT], check=False),
        daemon=True,
    ).start()


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
    direction:  int
    entry:      float
    target:     float
    stop:       float
    vol_ratio:  float
    move_bps:   float
    fired_at:   datetime
    expires_at: datetime
    is_rth:     bool


@dataclass
class PLMomSignal:
    direction: int
    entry:     float
    stop:      float
    pl:        float
    move_bps:  float
    bar_ts:    datetime

    def stop_pts(self): return abs(self.stop - self.entry)
    def expires_at(self): return self.bar_ts + timedelta(seconds=PL_MOM_MAX_HOLD_S)


@dataclass
class DOMBook:
    bids:        dict = field(default_factory=dict)
    asks:        dict = field(default_factory=dict)
    last_price:  float | None = None
    best_bid:    float | None = None
    best_ask:    float | None = None
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _lock:       threading.Lock = field(default_factory=threading.Lock)

    BID   = 4
    ASK   = 3
    RESET = 6

    def apply_depth(self, updates):
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

    def hybrid_snapshot(self, near: int, bucket_pts: float, bucket_n: int):
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
                buckets: dict[int, float] = {}
                for p, s in far:
                    idx = int((p - grid_start) // bucket_pts)
                    buckets[idx] = buckets.get(idx, 0.0) + s
                sorted_keys = sorted(buckets) if is_ask else sorted(buckets, reverse=True)
                for k in sorted_keys[:bucket_n]:
                    lo = grid_start + k * bucket_pts
                    rows.append((f"{lo:.0f}–{lo + bucket_pts:.0f}", buckets[k], True))
            while len(rows) < near + bucket_n:
                rows.append((None, 0.0, True))
            return rows

        return (build_side(all_bids, False), build_side(all_asks, True),
                last, bb, ba, updated)


@dataclass
class MonitorState:
    contract_id: str
    # 1-min bars (SLR + sizing)
    bars:        list = field(default_factory=list)
    # 5s bars (PL_MOM)
    bars_5s:           list = field(default_factory=list)
    last_bar_ts:       datetime | None = None
    sigma_30s_bps:     float = 0.0
    # DOM
    dom:          DOMBook = field(default_factory=DOMBook)
    wall_tracker: WallTracker | None = None
    # SLR signal
    slr_signal:        SLRSignal | None = None
    slr_last_surge_ts: datetime | None = None
    # PL_MOM signal
    pl_mom_sig:        PLMomSignal | None = None
    pl_mom_entry_ts:   datetime | None = None
    close_until:       datetime | None = None
    last_sig_bar_ts:   datetime | None = None
    pl_mom_history:    list = field(default_factory=list)   # (ts, pl, dir_sym, move_bps)
    # Position
    position_size:     int   = 0
    position_dir:      int   = 0
    position_entry:    float | None = None
    position_strategy: str   = ""


# ── DOM DB reader ─────────────────────────────────────────────────────────────

def _read_dom_db(state: MonitorState):
    if not DOM_DB_PATH.exists():
        return
    try:
        conn = sqlite3.connect(str(DOM_DB_PATH), timeout=1.0)
        row  = conn.execute(
            "SELECT updated, last_price, best_bid, best_ask, bids_json, asks_json "
            "FROM dom_live WHERE symbol=?", (SYMBOL,)
        ).fetchone()
        conn.close()
        if not row:
            return
        updated_dt = datetime.fromisoformat(row[0])
        if (datetime.now(timezone.utc) - updated_dt).total_seconds() > DOM_DB_STALE_S:
            return
        bids = {float(k): v for k, v in json.loads(row[4]).items()}
        asks = {float(k): v for k, v in json.loads(row[5]).items()}
        with state.dom._lock:
            state.dom.bids        = bids
            state.dom.asks        = asks
            state.dom.last_price  = row[1]
            state.dom.best_bid    = row[2]
            state.dom.best_ask    = row[3]
            state.dom.last_update = updated_dt
    except Exception:
        pass


# ── SLR evaluation ─────────────────────────────────────────────────────────────

def _in_settlement(ts: datetime) -> bool:
    return SETTLE_UTC_START <= ts.astimezone(timezone.utc).hour < SETTLE_UTC_END


def evaluate_slr(state: MonitorState) -> SLRSignal | None:
    bars = state.bars
    if len(bars) < SLR_VOL_LOOKBACK + 2:
        return None
    surge     = bars[-1]
    surge_idx = len(bars) - 1
    if _in_settlement(surge.ts):
        return None
    if state.slr_last_surge_ts and surge.ts <= state.slr_last_surge_ts:
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
    prev = bars[surge_idx - 1]
    if (not prev.open
            or (surge.ts - prev.ts) > timedelta(minutes=2)
            or abs(surge.close / prev.open - 1) > 0.05):
        return None
    move_bps = (surge.close - prev.open) / prev.open * 10000 * direction
    if move_bps < SLR_MOVE_BPS:
        return None
    entry    = surge.close
    target   = entry * (1 + direction * SLR_TARGET_BPS / 10000)
    stop     = entry * (1 - direction * SLR_STOP_BPS   / 10000)
    surge_et = surge.ts.astimezone(ET)
    is_rth   = (9, 30) <= (surge_et.hour, surge_et.minute) < (16, 0)
    hold     = SLR_HOLD_RTH if is_rth else SLR_HOLD_GLOBEX
    return SLRSignal(
        direction=direction, entry=entry, target=target, stop=stop,
        vol_ratio=vol_ratio, move_bps=move_bps,
        fired_at=surge.ts, expires_at=surge.ts + timedelta(minutes=hold),
        is_rth=is_rth,
    )


# ── PL_MOM helpers ─────────────────────────────────────────────────────────────

def _compute_sigma(bars: list, lookback: int) -> float:
    """σ of rolling 30s (PL_MOM_WINDOW × 5s) net moves in bps."""
    closes = np.array([b.close for b in bars], dtype=float)
    if len(closes) < PL_MOM_WINDOW + 2:
        return 0.0
    rets = np.log(closes[1:] / closes[:-1])
    n    = len(rets)
    lb   = min(lookback, n - PL_MOM_WINDOW + 1)
    moves = [abs(rets[i:i + PL_MOM_WINDOW].sum())
             for i in range(n - lb, n - PL_MOM_WINDOW + 1)
             if i >= 0]
    if len(moves) < 4:
        return 0.0
    return float(np.std(moves)) * 10000


def evaluate_pl_mom(state: MonitorState) -> PLMomSignal | None:
    bars = state.bars_5s
    if len(bars) < PL_MOM_WINDOW + 1:
        return None
    window = bars[-PL_MOM_WINDOW:]
    for i in range(1, len(window)):
        if (window[i].ts - window[i - 1].ts).total_seconds() > 8:
            return None
    last = window[-1]
    if state.pl_mom_sig and last.ts == state.pl_mom_sig.bar_ts:
        return None
    closes  = np.array([b.close for b in window], dtype=float)
    rets    = np.log(closes[1:] / closes[:-1])
    sum_abs = float(np.abs(rets).sum())
    if sum_abs == 0:
        return None
    pl       = float(abs(rets.sum()) / sum_abs)
    if pl < PL_MOM_ENTRY_PL:
        return None
    net_ret  = float(rets.sum())
    entry    = last.close
    move_bps = abs(net_ret) * 10000
    sigma    = state.sigma_30s_bps
    eff_thr  = max(PL_MOM_MOVE_BPS, PL_MOM_SIGMA_N * sigma) if sigma > 0 else PL_MOM_MOVE_BPS
    if move_bps < eff_thr:
        return None
    direction = 1 if net_ret > 0 else -1
    stop = entry * (1.0 - direction * PL_MOM_STOP_BPS / 10000.0)
    return PLMomSignal(direction=direction, entry=entry, stop=stop,
                       pl=pl, move_bps=move_bps, bar_ts=last.ts)


# ── Bar fetchers ──────────────────────────────────────────────────────────────

def fetch_1min_bars(client: TopstepClient, state: MonitorState):
    now_utc   = datetime.now(timezone.utc)
    now_floor = datetime.fromtimestamp(
        (int(now_utc.timestamp()) // 60) * 60, tz=timezone.utc)
    db_fresh = False
    if bars_db_available():
        raw = get_bars_from_db(SYMBOL, 1, SLR_BARS_FETCH + 1)
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
                unit=TopstepClient.MINUTE, unit_number=1, limit=SLR_BARS_FETCH)
            state.bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                               open=b["o"], high=b["h"], low=b["l"],
                               close=b["c"], volume=b["v"])
                           for b in reversed(raw)]
        except Exception:
            pass


# ── Visual helpers ────────────────────────────────────────────────────────────

def _pl_bar(pl: float, dir_sym: str, half: int = 6) -> str:
    filled = max(0, min(half, round(min(1.0, pl) * half)))
    empty  = half - filled
    if dir_sym == "▲":
        return f"{'░' * half}|[green]{'█' * filled}{'░' * empty}[/]"
    else:
        return f"[red]{'░' * empty}{'█' * filled}[/]|{'░' * half}"


def _centered_bar(ratio: float, is_long: bool, half: int = 6) -> str:
    filled = max(0, min(half, round(ratio * half)))
    empty  = half - filled
    if is_long:
        return f"[dim]{'░' * half}|[/][green]{'█' * filled}{'░' * empty}[/]"
    else:
        return f"[red]{'░' * empty}{'█' * filled}[/][dim]|{'░' * half}[/]"


# ── Panel builders ────────────────────────────────────────────────────────────

def build_pl_mom_panel(state: MonitorState, now: datetime) -> Panel:
    sig     = state.pl_mom_sig
    closing = state.close_until is not None and now < state.close_until

    if closing:
        status, style, border = "CLOSE TRD", "bold yellow on black", "yellow"
    elif sig is None:
        status, style, border = "NO TRADE",  "bold",                 "default"
    elif sig.direction == 1:
        status, style, border = "BUY LONG",  "bold green",           "green"
    else:
        status, style, border = "SELL SHORT", "bold red",            "red"

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")

    hdr = Table.grid()
    hdr.add_column(justify="center", min_width=28)
    hdr.add_row(f"[{style}]  {status}  [/]")
    root.add_row(hdr)
    root.add_row("")

    if sig and not closing:
        det = Table.grid(padding=(0, 1))
        det.add_column(width=8, justify="right")
        det.add_column()
        rem_s   = max(0, int((sig.expires_at() - now).total_seconds()))
        rem_str = f"{rem_s // 60}m {rem_s % 60:02d}s"
        clr     = "green" if sig.direction == 1 else "red"
        dirn    = "▲ LONG" if sig.direction == 1 else "▼ SHORT"
        det.add_row("", f"[bold {clr}]{dirn}[/]  PL={sig.pl:.3f}  {sig.move_bps:.1f}bp")
        det.add_row("Entry:",   f"[bold]{sig.entry:,.2f}[/]")
        det.add_row("Stop:",    f"[bold red]{sig.stop:,.2f}[/]  ({sig.stop_pts():.2f} pts)")
        det.add_row("Expires:", rem_str)
        root.add_row(det)
        root.add_row("")

    hist = state.pl_mom_history[-HISTORY_BARS:]
    if hist:
        ht = Table(box=box.SIMPLE, show_header=True, padding=(0, 1), header_style="bold")
        ht.add_column("time",  justify="right")
        ht.add_column("",      justify="center")
        ht.add_column("",      justify="left", no_wrap=True)
        ht.add_column("PL",    justify="right")
        ht.add_column("bp",    justify="right")
        for ts, pl, dir_sym, move in reversed(hist):
            t_str  = ts.astimezone(LOCAL).strftime("%H:%M:%S")
            pl_sty = ("bold green" if pl >= PL_MOM_ENTRY_PL else
                      "yellow"     if pl >= PL_MOM_ENTRY_PL * 0.85 else "")
            bp_sty = "bold green" if move >= PL_MOM_MOVE_BPS else ""
            ht.add_row(
                t_str, dir_sym, _pl_bar(pl, dir_sym),
                f"[{pl_sty}]{pl:.3f}[/]" if pl_sty else f"{pl:.3f}",
                f"[{bp_sty}]{move:.1f}[/]" if bp_sty else f"{move:.1f}",
            )
        root.add_row(ht)
    elif not state.bars_5s:
        root.add_row("warming up…")

    sigma   = state.sigma_30s_bps
    eff_thr = max(PL_MOM_MOVE_BPS, PL_MOM_SIGMA_N * sigma) if sigma > 0 else PL_MOM_MOVE_BPS
    if sigma > 0 and eff_thr > PL_MOM_MOVE_BPS:
        thr_str = f"bp ≥ {eff_thr:.1f} (floor {PL_MOM_MOVE_BPS:.0f}, σ={sigma:.1f})"
    elif sigma > 0:
        thr_str = f"bp ≥ {PL_MOM_MOVE_BPS:.0f} (σ={sigma:.1f})"
    else:
        thr_str = f"bp ≥ {PL_MOM_MOVE_BPS:.0f}"
    thresh = Table.grid()
    thresh.add_column(justify="center", min_width=28)
    thresh.add_row(f"entry: PL ≥ {PL_MOM_ENTRY_PL:.2f}  {thr_str}")
    thresh.add_row(f"exit:  PL ≤ {PL_MOM_EXIT_PL:.2f}  stop {PL_MOM_STOP_BPS:.0f}bp")
    root.add_row(thresh)

    return Panel(root, title=f"PL MOM  {SYMBOL}", border_style=border,
                 padding=(0, 1), expand=True)


def build_slr_panel(state: MonitorState, now: datetime) -> Panel:
    sig  = state.slr_signal
    bars = state.bars

    if sig:
        remaining = max(0, int((sig.expires_at - now).total_seconds()))
        clr  = "green" if sig.direction == 1 else "red"
        dirn = "▲ LONG" if sig.direction == 1 else "▼ SHORT"
        sess = "RTH" if sig.is_rth else "GLOBEX"
        t = Table(box=None, show_header=False, padding=(0, 1))
        t.add_column(width=9); t.add_column()
        t.add_row("Entry:",   f"[bold {clr}]{sig.entry:.2f}[/]")
        t.add_row("Target:",  f"[bold green]{sig.target:.2f}[/]  "
                              f"[dim]({abs(sig.target - sig.entry):.2f}pts)[/]")
        t.add_row("Stop:",    f"[bold red]{sig.stop:.2f}[/]  "
                              f"[dim]({abs(sig.stop - sig.entry):.2f}pts)[/]")
        t.add_row("Expires:", f"[dim]{remaining}s  {sess}[/]")
        t.add_row("Trigger:", f"[dim]vol={sig.vol_ratio:.1f}×  move={sig.move_bps:.1f}bp[/]")
        return Panel(t, title=f"[bold {clr}]⬤  SLR SCALP  {dirn}[/]",
                     border_style=clr, padding=(0, 1), expand=True)

    # No signal — show recent vol ratios
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(width=9)
    t.add_column(width=13)
    t.add_column(no_wrap=True)

    if len(bars) >= SLR_VOL_LOOKBACK + 2:
        for i in range(1, 7):
            idx = len(bars) - i
            if idx < SLR_VOL_LOOKBACK:
                break
            b      = bars[idx]
            prior  = [bars[j].volume for j in range(idx - SLR_VOL_LOOKBACK, idx)]
            med    = float(np.median(prior)) if prior else 0.0
            vr     = b.volume / med if med > 0 else 0.0
            prev_b = bars[idx - 1]
            move   = 0.0
            if (prev_b.open
                    and (b.ts - prev_b.ts) <= timedelta(minutes=2)
                    and abs(b.close / prev_b.open - 1) <= 0.05):
                move = (b.close - prev_b.open) / prev_b.open * 10000
            bar_time = (b.ts + timedelta(minutes=1)).astimezone(LOCAL).strftime("%H:%M")
            lbl = "Latest:" if i == 1 else f"  -{i}m:"
            vr_n = f"{vr:4.1f}"
            if vr >= SLR_VOL_MULT:
                vr_s = f"[bold green]{vr_n}×[/]"
            elif vr >= SLR_VOL_MULT * 0.6:
                vr_s = f"[yellow]{vr_n}×[/]"
            else:
                vr_s = f"[dim]{vr_n}×[/]"
            bp_col = ("bold green" if move >= SLR_MOVE_BPS
                      else ("bold red" if move <= -SLR_MOVE_BPS else ""))
            bp_s   = f"[{bp_col}]{move:+6.1f}bp[/]" if bp_col else f"{move:+6.1f}bp"
            cbar   = _centered_bar(min(1.0, vr / (SLR_VOL_MULT * 2)), move >= 0)
            t.add_row(lbl, cbar,
                      f"{vr_s}  [dim]{int(b.volume):>5}[/]  {bp_s}  [dim]{bar_time}[/]")
    else:
        t.add_row("Status:", "", "[dim]warming up…[/]")
    t.add_row("", "", f"[dim]{SLR_VOL_MULT:.0f}× vol threshold  {SLR_MOVE_BPS:.0f}bp surge[/]")
    return Panel(t, title=f"SLR SCALP  {SYMBOL}", border_style="blue",
                 padding=(0, 1), expand=True)


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
        cap    = DOM_BKT_CAP if is_bucket else DOM_NEAR_CAP
        width  = DOM_BKT_BAR_W if is_bucket else DOM_NEAR_BAR_W
        filled = max(1, int(round(min(size, cap) / cap * width))) if size > 0 else 0
        ch     = ("▒" if is_bucket else "█") * filled + "░" * (width - filled)
        return f"[{'green' if is_bid else 'red'}]{ch}[/]"

    def is_wall(size): return size > 0 and size >= med_sz * WALL_MULT

    t = Table(box=None, show_header=False, padding=(0, 0))
    t.add_column(justify="right",  width=6)
    t.add_column(justify="right",  width=DOM_NEAR_BAR_W)
    t.add_column(justify="center", width=DOM_PRICE_COL)
    t.add_column(justify="left",   width=DOM_NEAR_BAR_W)
    t.add_column(justify="left",   width=6)

    # Asks: buckets (far) top → near rows closest to spread
    for label, size, is_bucket in list(reversed(ask_rows[DOM_NEAR:])) + list(reversed(ask_rows[:DOM_NEAR])):
        if label is None:
            t.add_row("", "", "", "", ""); continue
        wall  = is_wall(size)
        style = "bold red" if wall else ("red" if size > 0 else "dim")
        bar   = sz_bar(size, False, is_bucket)
        wm    = "◀" if wall else " "
        tc    = wall_tests.get(label) if isinstance(label, float) else None
        tc_s  = f"[yellow]T{tc}[/]" if tc else ""
        sz_s  = f"[{style}]{size:.0f}{wm}[/]{tc_s}" if size > 0 else ""
        lbl_s = f"[{style}]{label}[/]" if is_bucket else f"[{style}]{label:.2f}[/]"
        t.add_row("", "", lbl_s, bar, sz_s)

    # Spread row
    spread = f"{ba - bb:.2f}" if bb and ba else "—"
    imbal_txt = ""
    if all_sizes:
        total_bid = sum(s for _, s, _ in bid_rows if s > 0)
        total_ask = sum(s for _, s, _ in ask_rows if s > 0)
        denom     = total_bid + total_ask
        imbal     = (total_bid - total_ask) / denom if denom else 0
        ic        = "green" if imbal > 0.1 else ("red" if imbal < -0.1 else "dim")
        imbal_txt = f"  [{ic}]imb {imbal:+.2f}[/]"
    t.add_row("", f"[dim]spread {spread}[/]{imbal_txt}", "", "", "")

    # Bids: near rows (closest to spread) then buckets
    for label, size, is_bucket in bid_rows[:DOM_NEAR] + bid_rows[DOM_NEAR:]:
        if label is None:
            t.add_row("", "", "", "", ""); continue
        wall  = is_wall(size)
        style = "bold green" if wall else ("green" if size > 0 else "dim")
        bar   = sz_bar(size, True, is_bucket)
        wm    = "◀" if wall else " "
        tc    = wall_tests.get(label) if isinstance(label, float) else None
        tc_s  = f"[yellow]T{tc}[/]" if tc else ""
        sz_s  = f"{tc_s}[{style}]{wm}{size:.0f}[/]" if size > 0 else ""
        lbl_s = f"[{style}]{label}[/]" if is_bucket else f"[{style}]{label:.2f}[/]"
        t.add_row(sz_s, bar, lbl_s, "", "")

    # Wall summary
    def best_wall(rows, ref):
        cands = [(lbl, sz) for lbl, sz, _ in rows if sz > 0 and isinstance(lbl, float)]
        if not cands:
            cands = [(lbl, sz) for lbl, sz, _ in rows if sz > 0]
        if not cands:
            return None
        lbl, sz = max(cands, key=lambda x: x[1])
        if isinstance(lbl, float) and ref:
            return f"{sz:.0f} @ {abs(lbl - ref):.2f}pts"
        return f"{sz:.0f}"

    wbid = best_wall(bid_rows, bb)
    wask = best_wall(ask_rows, ba)
    if wbid or wask:
        t.add_row("", f"bid wall {wbid or '—'}", "", f"ask wall {wask or '—'}", "")

    age     = (datetime.now(timezone.utc) - updated).total_seconds()
    age_col = "green" if age < 10 else ("yellow" if age < 30 else "red")
    last_s  = f"{last:.2f}" if last else "—"
    return Panel(t,
                 title=f"DOM  {SYMBOL}  [{age_col}]{last_s}[/]  [dim]{age:.0f}s ago[/]",
                 border_style="blue", padding=(0, 1), expand=True)


def build_sizing_panel(state: MonitorState) -> Panel:
    sigma_pts: float | None = None
    if len(state.bars) >= 2:
        recent = state.bars[-min(SIZING_SIGMA_BARS, len(state.bars)):]
        closes = [b.close for b in recent]
        lrs    = [np.log(closes[i] / closes[i - 1])
                  for i in range(1, len(closes)) if closes[i - 1] > 0]
        if len(lrs) >= 2:
            sigma_pts = float(np.std(lrs, ddof=1)) * closes[-1]

    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 2))
    t.add_column("",     justify="right")
    t.add_column(SYMBOL, justify="right", style="bold")

    if sigma_pts:
        close = state.bars[-1].close
        t.add_row("bp",  f"{2 * sigma_pts / close * 10000:.1f}", style="cyan")
        t.add_row("pts", f"{2 * sigma_pts:.2f}",                 style="bold cyan")
    else:
        t.add_row("bp",  "—", style="cyan")
        t.add_row("pts", "—", style="bold cyan")

    actual_bars = min(SIZING_SIGMA_BARS, len(state.bars))
    t.add_row("RISK", "")
    for risk in SIZING_RISKS:
        if sigma_pts and sigma_pts > 0:
            t.add_row(f"${risk}", f"{risk / (2 * sigma_pts * POINT_VALUE):.1f}")
        else:
            t.add_row(f"${risk}", "—")
    return Panel(t, title="[bold]SIZING (2σ stop)[/]",
                 subtitle=f"[dim]σ: {actual_bars} 1-min bars[/]",
                 border_style="blue", padding=(0, 1), expand=False)


def build_positions_panel(state: MonitorState) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(width=5)
    t.add_column(width=4,  justify="right")
    t.add_column(width=8)
    t.add_column(width=9,  justify="right")
    t.add_column(width=7)
    entry = f"{state.position_entry:.2f}" if state.position_entry else ""
    if state.position_size == 0:
        t.add_row(SYMBOL, "—", "", "", "")
    elif state.position_dir == 1:
        t.add_row(SYMBOL, f"[green]{state.position_size}[/]",
                  "[green]▲ LONG[/]", f"[green]{entry}[/]",
                  f"[green]{state.position_strategy}[/]")
    else:
        t.add_row(SYMBOL, f"[red]{state.position_size}[/]",
                  "[red]▼ SHORT[/]", f"[red]{entry}[/]",
                  f"[red]{state.position_strategy}[/]")
    return Panel(t, title="POSITIONS", border_style="blue", padding=(0, 1), expand=False)


def build_header() -> Table:
    now_et  = datetime.now(ET)
    now_loc = datetime.now(LOCAL)
    hm_et   = now_et.hour * 60 + now_et.minute
    if (9 * 60 + 30) <= hm_et < 16 * 60:
        sess = "[bold green]RTH[/]"
    else:
        h = datetime.now(timezone.utc).hour
        sess = ("[bold red]SETTLEMENT[/]" if SETTLE_UTC_START <= h < SETTLE_UTC_END
                else "[dim]GLOBEX[/]")
    t = Table.grid(expand=True)
    t.add_column(ratio=1)
    t.add_column(ratio=1, justify="center")
    t.add_column(ratio=1, justify="right")
    t.add_row(
        f"[bold]MES Monitor[/]  {sess}",
        f"[dim]{now_loc.strftime('%H:%M:%S')}  /  {now_et.strftime('%H:%M ET')}[/]",
        "",
    )
    return t


def render(state: MonitorState) -> Table:
    now = datetime.now(timezone.utc)

    root = Table.grid(expand=True)
    root.add_column(ratio=1)

    # Row 1: header
    root.add_row(build_header())

    # Row 2: [SLR + DOM stacked (left) | PL_MOM (right)]
    # DOM stays fixed in the left column regardless of PL_MOM height.
    left_col = Table.grid()
    left_col.add_column()
    left_col.add_row(build_slr_panel(state, now))
    left_col.add_row(build_dom_panel(state))

    row2 = Table(box=None, show_header=False, padding=(0, 1), expand=True)
    row2.add_column()        # left: SLR + DOM (natural width)
    row2.add_column(ratio=1) # right: PL_MOM (fills remaining space)
    row2.add_row(left_col, build_pl_mom_panel(state, now))
    root.add_row(row2)

    # Row 3: Trade Summary + Positions | Sizing
    left = Table.grid()
    left.add_column()
    left.add_row(build_trade_summary_panel())
    left.add_row(build_positions_panel(state))

    row3 = Table(box=None, show_header=False, padding=(0, 1), expand=False)
    row3.add_column()
    row3.add_column()
    row3.add_row(left, build_sizing_panel(state))
    root.add_row(row3)

    return root


# ── Position strategy detection ───────────────────────────────────────────────

def _detect_position_strategy() -> str:
    log_path = Path("logs/trading_bot.log")
    if not log_path.exists():
        return ""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        for line in reversed(lines[-500:]):
            if f" {SYMBOL} " not in line:
                continue
            if "VWASLR ORDER" in line: return "VWASLR"
            if "SLR ORDER"    in line: return "SLR"
            if "PL MOM ORDER" in line: return "PL MOM"
            if "ORB ORDER"    in line: return "ORB"
            if "ORDER PLACED" in line: return "CSR"
    except Exception:
        pass
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    client = TopstepClient()
    client.use_shared_token()

    ensure_wall_log()

    contracts = client.search_contracts(SYMBOL)
    if not contracts:
        print(f"ERROR: no contract for {SYMBOL}")
        return
    cid = str(contracts[0]["id"])

    state = MonitorState(contract_id=cid, wall_tracker=WallTracker(SYMBOL))

    # Initial 1-min bar fetch
    fetch_1min_bars(client, state)

    # DOM reader thread (250ms)
    def _poll_dom():
        while True:
            _read_dom_db(state)
            time.sleep(0.25)
    threading.Thread(target=_poll_dom, daemon=True, name="dom-reader").start()

    # 1-min bar fetch thread (60s cadence, for SLR + sizing)
    def _fetch_1min_loop():
        while True:
            time.sleep(60)
            try:
                fetch_1min_bars(client, state)
            except Exception:
                pass
    threading.Thread(target=_fetch_1min_loop, daemon=True, name="bar-fetch").start()

    # 5s bar fetch + PL_MOM eval (2s cadence, RTH only)
    def _poll_5s():
        while True:
            now    = datetime.now(timezone.utc)
            now_et = now.astimezone(ET)
            in_rth = (9, 30) <= (now_et.hour, now_et.minute) < (16, 0)
            if in_rth:
                try:
                    if bars_db_available():
                        raw = get_5s_bars_from_db(SYMBOL, PL_MOM_5S_FETCH)
                    else:
                        floor = datetime.fromtimestamp(
                            (int(now.timestamp()) // 5) * 5, tz=timezone.utc)
                        start = floor - timedelta(seconds=5 * (PL_MOM_5S_FETCH + 5))
                        raw   = client.get_bars(
                            contract_id=state.contract_id, start=start, end=floor,
                            unit=TopstepClient.SECOND, unit_number=5,
                            limit=PL_MOM_5S_FETCH)
                    if raw:
                        state.bars_5s = [
                            Bar(ts=datetime.fromisoformat(b["t"]),
                                open=b["o"], high=b["h"], low=b["l"],
                                close=b["c"], volume=b["v"])
                            for b in raw]
                        state.sigma_30s_bps = _compute_sigma(state.bars_5s, PL_MOM_SIGMA_LB)
                except Exception:
                    pass

                # History update on new bar
                bar_ts = state.bars_5s[-1].ts if state.bars_5s else None
                if bar_ts and bar_ts != state.last_bar_ts:
                    state.last_bar_ts = bar_ts
                    if len(state.bars_5s) >= PL_MOM_WINDOW:
                        window  = state.bars_5s[-PL_MOM_WINDOW:]
                        closes  = np.array([b.close for b in window], dtype=float)
                        rets    = np.log(closes[1:] / closes[:-1])
                        sum_abs = float(np.abs(rets).sum())
                        if sum_abs > 0:
                            net = float(rets.sum())
                            state.pl_mom_history.append((
                                bar_ts, abs(net) / sum_abs,
                                "▲" if net > 0 else "▼",
                                abs(net) * 10000,
                            ))
                            if len(state.pl_mom_history) > 40:
                                state.pl_mom_history = state.pl_mom_history[-40:]

                    # New signal?
                    new_sig = evaluate_pl_mom(state)
                    if (new_sig and state.pl_mom_sig is None
                            and new_sig.bar_ts != state.last_sig_bar_ts):
                        state.pl_mom_sig      = new_sig
                        state.pl_mom_entry_ts = now
                        state.close_until     = None
                        state.last_sig_bar_ts = new_sig.bar_ts
                        play_alert()

                # Expire / PL exit
                if state.pl_mom_sig is not None:
                    sig    = state.pl_mom_sig
                    min_ok = (state.pl_mom_entry_ts is not None and
                              (now - state.pl_mom_entry_ts).total_seconds() >= PL_MOM_MIN_HOLD_S)
                    expired = now >= sig.expires_at()
                    pl_exit = False
                    if min_ok and len(state.bars_5s) >= PL_MOM_WINDOW:
                        window  = state.bars_5s[-PL_MOM_WINDOW:]
                        closes  = np.array([b.close for b in window], dtype=float)
                        rets    = np.log(closes[1:] / closes[:-1])
                        sa      = float(np.abs(rets).sum())
                        cur_pl  = abs(float(rets.sum()) / sa) if sa > 0 else 0.0
                        pl_exit = cur_pl <= PL_MOM_EXIT_PL
                    if expired or pl_exit:
                        state.pl_mom_sig      = None
                        state.pl_mom_entry_ts = None
                        state.close_until     = now + timedelta(seconds=CLOSE_TRD_SHOW_S)
            time.sleep(2)
    threading.Thread(target=_poll_5s, daemon=True, name="pl-mom").start()

    # SLR evaluate thread (5s cadence)
    def _eval_slr():
        while True:
            time.sleep(5)
            now = datetime.now(timezone.utc)
            # Wall tracker update
            if state.wall_tracker is not None:
                with state.dom._lock:
                    bids = dict(state.dom.bids)
                    asks = dict(state.dom.asks)
                    bb   = state.dom.best_bid
                    ba   = state.dom.best_ask
                events = state.wall_tracker.update(bids, asks, bb, ba, now)
                if events:
                    log_wall_events(events)
            # Expire active SLR signal
            if state.slr_signal:
                sig = state.slr_signal
                if now >= sig.expires_at:
                    state.slr_last_surge_ts = sig.fired_at
                    state.slr_signal = None
                elif state.bars:
                    lc = state.bars[-1].close
                    hit_target = (sig.direction == 1 and lc >= sig.target) or \
                                 (sig.direction == -1 and lc <= sig.target)
                    hit_stop   = (sig.direction == 1 and lc <= sig.stop) or \
                                 (sig.direction == -1 and lc >= sig.stop)
                    if hit_target or hit_stop:
                        state.slr_last_surge_ts = sig.fired_at
                        state.slr_signal = None
            # Evaluate new SLR signal
            if not state.slr_signal and state.bars:
                sig = evaluate_slr(state)
                if sig:
                    state.slr_signal        = sig
                    state.slr_last_surge_ts = sig.fired_at
    threading.Thread(target=_eval_slr, daemon=True, name="slr-eval").start()

    # Position poll thread (30s cadence)
    def _poll_positions():
        while True:
            try:
                account_id = client.get_accounts()[0]["id"]
                positions  = client.get_open_positions(account_id)
                pos = next((p for p in positions
                            if str(p.get("contractId", "")) == state.contract_id), None)
                sz  = int(pos.get("size", 0)) if pos else 0
                pt  = int(pos.get("type", 0)) if pos else 0
                prev_size              = state.position_size
                state.position_size    = abs(sz)
                state.position_entry   = (float(pos.get("averagePrice", 0) or 0)
                                          if pos and sz else None)
                state.position_dir     = 0 if sz == 0 else (-1 if pt == 2 else 1)
                if state.position_size > 0 and prev_size == 0:
                    state.position_strategy = _detect_position_strategy()
                elif state.position_size == 0:
                    state.position_strategy = ""
            except Exception:
                pass
            time.sleep(30)
    threading.Thread(target=_poll_positions, daemon=True, name="pos-poll").start()

    time.sleep(2)
    console = Console()
    with Live(render(state), console=console, refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(0.25)
            live.update(render(state))


if __name__ == "__main__":
    run()
