"""
Globex Monitor — Sunday Open Gap and Evening Resumption Gap.

Unlike the main signal_monitor (which covers RTH), this monitor focuses on
Globex-specific opportunities that have distinct market microstructure.
Currently implemented:
  1. Sunday Open Gap    — gap momentum in the 30 min after the weekly open
  2. Evening Resumption — gap from the 17:00–18:00 ET settlement, Mon–Fri

Run modes:
  python src/globex_monitor.py          # live (requires .env credentials)
  python src/globex_monitor.py --demo   # static demo with synthetic data

Trading status:
  Sunday Open Gap     — LIVE (traded via trading_bot.py)
  Evening Resumption  — DISPLAY ONLY (not yet traded; under observation)
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from rich import box
from rich.console import Console
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from topstep_client import TopstepClient

# ── Constants ─────────────────────────────────────────────────────────────────

ET           = ZoneInfo("America/New_York")
POLL_SECONDS = 15
CSV_PATH     = "mes_hist_1min.csv"
BAR_MINUTES  = 5

# ── Sunday Open Gap parameters (from backtest_sunday_globex.py) ───────────────
SUN_GAP_THRESH        = 0.003   # 0.3% minimum
SUN_GAP_THRESH_STRONG = 0.005   # 0.5% — strong tier
SUN_VOL_LOOKBACK      = 8       # prior Sunday opens for median baseline
SUN_VOL_MULT          = 1.5     # volume must be ≥ 1.5× median
SUN_HOLD_BARS         = 6       # 6 × 5-min = 30 min

# ── Evening Resumption parameters (from backtest_evening_globex.py) ───────────
EVE_GAP_THRESH        = 0.002   # 0.2% minimum
EVE_GAP_THRESH_STRONG = 0.003   # 0.3% — strong tier
EVE_HOLD_BARS         = 6       # 30 min (best from backtest)
EVE_HIST_MIN_GAP      = 0.001   # show in history if gap ≥ 0.1% (filter noise)

# Active window: 30 min before 18:00 through 35 min after (both signals)
WINDOW_PRE_MIN  = 30
WINDOW_POST_MIN = 35

MES_POINT = 5.00
console   = Console()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SundayOpenHistory:
    ts:         datetime
    fri_close:  float
    gap_open:   float
    gap_close:  float
    volume:     float
    vol_med:    float
    signal_dir: int      # +1 / -1 / 0
    outcome:    str      # "cont" / "rev" / "open"
    pnl_pts:    float


@dataclass
class SundayOpenState:
    vol_baseline:    float = 0.0
    fri_close:       float = 0.0
    first_bar_open:  float = 0.0
    first_bar_close: float = 0.0
    first_bar_vol:   float = 0.0
    gap_open_pct:    float = 0.0
    gap_close_pct:   float = 0.0
    signal_dir:      int   = 0
    signal_entry:    float = 0.0
    candle_complete: bool  = False
    signal_fired_at: datetime | None = None
    signal_exit_at:  datetime | None = None
    history: list = field(default_factory=list)
    status: str = "INACTIVE"   # INACTIVE/STANDBY/FIRST_CANDLE/SIGNAL_EVAL/IN_TRADE/DONE


@dataclass
class EveningOpenHistory:
    ts:         datetime
    prev_close: float
    gap_open:   float
    gap_close:  float
    volume:     float
    signal_dir: int      # +1 / -1 / 0 (gap threshold only, no vol filter)
    outcome:    str      # "cont" / "rev" / "open"
    dow:        int      # 0=Mon … 4=Fri


@dataclass
class EveningOpenState:
    prev_close:      float = 0.0
    first_bar_open:  float = 0.0
    first_bar_close: float = 0.0
    first_bar_vol:   float = 0.0
    gap_open_pct:    float = 0.0
    gap_close_pct:   float = 0.0
    signal_dir:      int   = 0
    signal_entry:    float = 0.0
    candle_complete: bool  = False
    history: list = field(default_factory=list)
    status: str = "INACTIVE"   # INACTIVE/STANDBY/FIRST_CANDLE/SIGNAL_EVAL/DONE
    # Note: no IN_TRADE — evening signal is display-only, not yet traded


# ── CSV history loading ───────────────────────────────────────────────────────

def _load_df5(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Shared 5-min resampler used by both history loaders."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[df.index.minute % BAR_MINUTES == 0]
    df5 = df.resample(f"{BAR_MINUTES}min", closed="left", label="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()
    df5.index = df5.index.tz_convert(ET)
    df5 = df5.reset_index()
    df5.columns = ["ts", "open", "high", "low", "close", "volume"]
    return df5


def load_sunday_history(csv_path: str = CSV_PATH) -> tuple[list, float]:
    """Load Sunday open events. Returns (history, vol_baseline)."""
    if not Path(csv_path).exists():
        return [], 0.0

    df5 = _load_df5(csv_path)
    df5["gap_td"] = df5["ts"].diff()
    history = []

    for i, row in df5.iterrows():
        if i == 0:
            continue
        if pd.isna(row["gap_td"]) or row["gap_td"] <= timedelta(hours=24):
            continue
        if row["ts"].weekday() != 6:
            continue
        prev = df5.iloc[i - 1]
        if prev["ts"].weekday() not in (4, 5):
            continue

        gap_open  = math.log(row["open"]  / prev["close"])
        gap_close = math.log(row["close"] / prev["close"])
        direction = 1 if gap_open > 0 else -1

        exit_ts = row["ts"] + timedelta(minutes=SUN_HOLD_BARS * BAR_MINUTES)
        later   = df5[df5["ts"] >= exit_ts]
        if len(later) > 0:
            fwd_ret = math.log(float(later.iloc[0]["close"]) / row["close"])
            dir_ret = direction * fwd_ret
            outcome = "cont" if dir_ret > 0 else "rev"
            pnl_pts = dir_ret * float(row["close"])
        else:
            outcome, pnl_pts = "open", 0.0

        vols    = [h.volume for h in history[-SUN_VOL_LOOKBACK:]]
        vol_med = float(np.median(vols)) if len(vols) >= 3 else 0.0
        sig     = direction if (abs(gap_open) >= SUN_GAP_THRESH
                                and vol_med > 0
                                and row["volume"] >= SUN_VOL_MULT * vol_med) else 0

        history.append(SundayOpenHistory(
            ts=row["ts"].to_pydatetime(), fri_close=float(prev["close"]),
            gap_open=gap_open, gap_close=gap_close,
            volume=float(row["volume"]), vol_med=vol_med,
            signal_dir=sig, outcome=outcome,
            pnl_pts=pnl_pts if sig != 0 else 0.0,
        ))

    vol_baseline = 0.0
    if history:
        vol_baseline = float(np.median([h.volume for h in history[-SUN_VOL_LOOKBACK:]]))
    return history, vol_baseline


def load_evening_history(csv_path: str = CSV_PATH) -> list:
    """Load weekday 18:00 ET resumption events (gap ≥ EVE_HIST_MIN_GAP only)."""
    if not Path(csv_path).exists():
        return []

    df5 = _load_df5(csv_path)
    df5["gap_td"] = df5["ts"].diff()
    history = []

    for i, row in df5.iterrows():
        if i == 0:
            continue
        ts = row["ts"]
        if ts.hour != 18 or ts.minute != 0:
            continue
        if ts.weekday() > 4:   # skip Sat/Sun
            continue
        gap_td = row["gap_td"]
        if pd.isna(gap_td) or gap_td < timedelta(minutes=55):
            continue

        prev = df5.iloc[i - 1]
        gap_open  = math.log(float(row["open"])  / float(prev["close"]))
        gap_close = math.log(float(row["close"]) / float(prev["close"]))

        # Only record events above the display threshold
        if abs(gap_open) < EVE_HIST_MIN_GAP:
            continue

        direction = 1 if gap_open > 0 else -1
        exit_ts   = ts + timedelta(minutes=EVE_HOLD_BARS * BAR_MINUTES)
        later     = df5[df5["ts"] >= exit_ts]
        if len(later) > 0:
            fwd_ret = math.log(float(later.iloc[0]["close"]) / float(row["close"]))
            outcome = "cont" if direction * fwd_ret > 0 else "rev"
        else:
            outcome = "open"

        sig = direction if abs(gap_open) >= EVE_GAP_THRESH else 0

        history.append(EveningOpenHistory(
            ts=ts.to_pydatetime(), prev_close=float(prev["close"]),
            gap_open=gap_open, gap_close=gap_close,
            volume=float(row["volume"]),
            signal_dir=sig, outcome=outcome, dow=ts.weekday(),
        ))

    return history


# ── Live data helpers ─────────────────────────────────────────────────────────

def fetch_current_bar(client: TopstepClient, contract_id: str) -> dict | None:
    now   = datetime.now(timezone.utc)
    bars  = client.get_bars(
        contract_id=contract_id,
        start=now - timedelta(minutes=15), end=now,
        unit=TopstepClient.MINUTE, unit_number=BAR_MINUTES, limit=5,
    )
    return bars[0] if bars else None


def fetch_prev_close(client: TopstepClient, contract_id: str,
                     cutoff_et: datetime) -> float:
    """Fetch the close of the last bar before cutoff_et (handles both Fri close
    for Sunday and pre-17:00 close for evening)."""
    cutoff_utc = cutoff_et.astimezone(timezone.utc)
    bars = client.get_bars(
        contract_id=contract_id,
        start=cutoff_utc - timedelta(days=3),
        end=cutoff_utc - timedelta(minutes=BAR_MINUTES),
        unit=TopstepClient.MINUTE, unit_number=BAR_MINUTES, limit=20_000,
    )
    return float(bars[0]["c"]) if bars else 0.0


# ── Signal evaluation ─────────────────────────────────────────────────────────

def evaluate_sunday_signal(state: SundayOpenState) -> int:
    if not state.candle_complete or state.fri_close == 0:
        return 0
    gap = state.gap_open_pct
    if abs(gap) < SUN_GAP_THRESH:
        return 0
    if state.vol_baseline > 0 and state.first_bar_vol < SUN_VOL_MULT * state.vol_baseline:
        return 0
    return 1 if gap > 0 else -1


def evaluate_evening_signal(state: EveningOpenState) -> int:
    if not state.candle_complete or state.prev_close == 0:
        return 0
    gap = state.gap_open_pct
    if abs(gap) < EVE_GAP_THRESH:
        return 0
    return 1 if gap > 0 else -1


# ── TUI helpers ───────────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 20,
         pos: str = "green", neg: str = "red") -> Text:
    t    = Text()
    fill = int(min(abs(value) / max_val, 1.0) * (width // 2))
    half = width // 2
    if value >= 0:
        t.append("─" * half, style="dim")
        t.append("█" * fill, style=pos)
        t.append("░" * (half - fill), style="dim")
    else:
        t.append("░" * (half - fill), style="dim")
        t.append("█" * fill, style=neg)
        t.append("─" * half, style="dim")
    return t


def _gap_row(tbl: Table, label: str, gap: float, thresh: float, strong: float):
    color = "green" if gap > 0 else "red"
    txt   = Text(f"{gap*100:+.3f}%", style=color)
    if abs(gap) >= strong:
        txt.append("  ✓ (strong)", style="green")
    elif abs(gap) >= thresh:
        txt.append("  ✓", style="yellow")
    else:
        txt.append(f"  ✗  need {thresh*100:.1f}%", style="dim")
    tbl.add_row(label, txt)


def _outcome_text(h, show_vol: bool = False) -> tuple[Text, Text]:
    """Returns (sig_text, result_text) for a history row."""
    if h.signal_dir == 1:
        sig = Text("▲", style="green")
    elif h.signal_dir == -1:
        sig = Text("▼", style="red")
    else:
        sig = Text("—", style="dim")

    if h.outcome == "cont":
        style = "green" if h.signal_dir != 0 else "dim"
        res   = Text("cont ✓" if h.signal_dir != 0 else "cont", style=style)
    elif h.outcome == "rev":
        style = "red" if h.signal_dir != 0 else "dim"
        res   = Text("rev ✗" if h.signal_dir != 0 else "rev", style=style)
    else:
        res = Text(h.outcome, style="dim")

    return sig, res


def _signal_status_row(tbl: Table, state, hold_bars: int, now_et: datetime,
                        in_trade: bool = False):
    """Add the signal / status row to a panel table."""
    sig = state.signal_dir
    s   = state.status

    if sig == 0:
        tbl.add_row("Signal:", Text("NO SIGNAL", style="dim"))
        return "dim"

    lbl = "LONG" if sig > 0 else "SHORT"
    col = "green" if sig > 0 else "red"

    if in_trade and s == "IN_TRADE":
        elapsed = int((now_et - state.signal_fired_at.astimezone(ET))
                      .total_seconds() / 60)
        remain  = hold_bars * BAR_MINUTES - elapsed
        tbl.add_row("Signal:", Text(f"{lbl}  —  {remain}m remaining", style=col))
    elif s == "DONE" and in_trade:
        tbl.add_row("Signal:", Text(f"{lbl}  —  hold complete", style=f"{col} bold"))
        return "dim"
    elif s == "SIGNAL_EVAL":
        tbl.add_row("Signal:", Text(f"SIGNAL: {lbl}  (display only)", style=f"{col} bold"))
    else:
        tbl.add_row("Signal:", Text(f"SIGNAL: {lbl}", style=f"{col} bold"))

    if state.signal_entry:
        tbl.add_row("Entry:", f"{state.signal_entry:,.2f}")

    return col


# ── Sunday Open Gap panel ─────────────────────────────────────────────────────

def _next_sunday_open_et(now: datetime) -> datetime:
    d = now.date()
    days = (6 - d.weekday()) % 7
    if days == 0 and now.hour >= 18:
        days = 7
    t = d + timedelta(days=days)
    return datetime(t.year, t.month, t.day, 18, 0, tzinfo=ET)


def build_sunday_panel(state: SundayOpenState, now_et: datetime) -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column("label", style="dim", width=14, no_wrap=True)
    tbl.add_column("value", width=42)

    s = state.status

    if s == "INACTIVE":
        nxt   = _next_sunday_open_et(now_et)
        delta = nxt - now_et
        d, r  = divmod(int(delta.total_seconds()), 86400)
        h, r  = divmod(r, 3600)
        m     = r // 60
        tbl.add_row("Next open:",
                    Text(f"Sun {nxt.strftime('%b %d')} 18:00 ET"
                         f"  in {d}d {h:02d}h {m:02d}m", style="dim"))
        tbl.add_row("Vol baseline:", f"{state.vol_baseline:,.0f}  "
                    f"(last {SUN_VOL_LOOKBACK} opens)")
        tbl.add_row("Threshold:", f"≥ {SUN_GAP_THRESH*100:.1f}%  "
                    f"| vol ≥ {SUN_VOL_MULT}× med")
        border, title = "dim", "SUNDAY GAP  ·  inactive"

    elif s == "STANDBY":
        open_t = datetime(now_et.year, now_et.month, now_et.day, 18, 0, tzinfo=ET)
        mins   = int((open_t - now_et).total_seconds() / 60)
        tbl.add_row("Status:", Text(f"Standby — {mins}m to open", style="yellow"))
        if state.fri_close:
            tbl.add_row("Fri close:", Text(f"{state.fri_close:,.2f}", style="white"))
        tbl.add_row("Vol baseline:", f"{state.vol_baseline:,.0f}")
        tbl.add_row("Need vol ≥:", f"{state.vol_baseline * SUN_VOL_MULT:,.0f}  "
                    f"({SUN_VOL_MULT}×)")
        border, title = "yellow", "SUNDAY GAP  ·  standby"

    elif s == "FIRST_CANDLE":
        tbl.add_row("Status:", Text("First candle forming …", style="yellow bold"))
        if state.fri_close and state.first_bar_open:
            gap   = math.log(state.first_bar_open / state.fri_close)
            color = "green" if gap > 0 else "red"
            tbl.add_row("Fri close:", f"{state.fri_close:,.2f}")
            tbl.add_row("Open:", Text(f"{state.first_bar_open:,.2f}", style=color))
            _gap_row(tbl, "Gap (open):", gap, SUN_GAP_THRESH, SUN_GAP_THRESH_STRONG)
            tbl.add_row("", _bar(gap, 0.01))
        tbl.add_row("Vol baseline:", f"{state.vol_baseline:,.0f}")
        tbl.add_row("Need vol ≥:", f"{state.vol_baseline * SUN_VOL_MULT:,.0f}  "
                    f"({SUN_VOL_MULT}×)")
        tbl.add_row("Threshold:", f"≥ {SUN_GAP_THRESH*100:.1f}%  "
                    f"| vol ≥ {SUN_VOL_MULT}× med")
        border, title = "yellow", "SUNDAY GAP  ·  first candle"

    elif s in ("SIGNAL_EVAL", "IN_TRADE", "DONE"):
        go, gc = state.gap_open_pct, state.gap_close_pct
        tbl.add_row("Fri close:",   f"{state.fri_close:,.2f}")
        tbl.add_row("First open:",  f"{state.first_bar_open:,.2f}")
        tbl.add_row("First close:", f"{state.first_bar_close:,.2f}")
        _gap_row(tbl, "Gap (open):",  go, SUN_GAP_THRESH, SUN_GAP_THRESH_STRONG)
        _gap_row(tbl, "Gap (close):", gc, SUN_GAP_THRESH, SUN_GAP_THRESH_STRONG)
        tbl.add_row("", _bar(go, 0.01))

        vol_x  = (state.first_bar_vol / state.vol_baseline
                  if state.vol_baseline > 0 else 0.0)
        vol_ok = state.vol_baseline > 0 and state.first_bar_vol >= SUN_VOL_MULT * state.vol_baseline
        v_txt  = Text(f"{state.first_bar_vol:,.0f}  ({vol_x:.1f}×)")
        v_txt.append("  ✓" if vol_ok else
                     f"  ✗  need {state.vol_baseline*SUN_VOL_MULT:,.0f}",
                     style="green" if vol_ok else "dim")
        tbl.add_row("Volume:", v_txt)

        border = _signal_status_row(tbl, state, SUN_HOLD_BARS, now_et, in_trade=True)
        title  = {"SIGNAL_EVAL": "SUNDAY GAP  ·  evaluating",
                  "IN_TRADE":    "SUNDAY GAP  ·  in trade",
                  "DONE":        "SUNDAY GAP  ·  complete"}[s]
    else:
        border, title = "dim", "SUNDAY GAP"

    return Panel(tbl, title=title, border_style=border, box=box.ROUNDED)


def build_sunday_history_panel(state: SundayOpenState) -> Panel:
    tbl = Table(box=None, show_header=True, padding=(0, 1))
    tbl.add_column("Date",   style="dim", width=7)
    tbl.add_column("Gap",    justify="right", width=8)
    tbl.add_column("Vol×",   justify="right", width=6)
    tbl.add_column("Sig",    justify="center", width=4)
    tbl.add_column("Result", justify="center", width=8)

    recent = state.history[-8:] if state.history else []
    for h in reversed(recent):
        sig, res = _outcome_text(h)
        vol_x = f"{h.volume/h.vol_med:.1f}×" if h.vol_med > 0 else "—"
        tbl.add_row(h.ts.astimezone(ET).strftime("%b %d"),
                    f"{h.gap_open*100:+.2f}%", vol_x, sig, res)

    sigs  = [h for h in state.history if h.signal_dir != 0]
    conts = sum(1 for h in sigs if h.outcome == "cont")
    stats = Text(
        f"  {len(sigs)} signals  •  {conts/len(sigs)*100:.0f}% cont"
        if sigs else "  No signals fired yet", style="dim"
    )
    return Panel(Group(tbl, stats),
                 title="Sunday Opens  (last 8)",
                 border_style="dim", box=box.ROUNDED)


# ── Evening Resumption panel ──────────────────────────────────────────────────

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _side_by_side(left, right) -> Table:
    """Place two renderables side-by-side at natural width, no gap between them."""
    grid = Table.grid()
    grid.add_column()
    grid.add_column()
    grid.add_row(left, right)
    return grid


def _next_evening_open_et(now: datetime) -> datetime:
    """Next 18:00 ET on a weekday."""
    d = now.date()
    for offset in range(1, 8):
        t = d + timedelta(days=offset)
        if t.weekday() <= 4:
            return datetime(t.year, t.month, t.day, 18, 0, tzinfo=ET)
    raise ValueError("no weekday found")


def build_evening_panel(state: EveningOpenState, now_et: datetime) -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column("label", style="dim", width=14, no_wrap=True)
    tbl.add_column("value", width=42)

    s = state.status

    if s == "INACTIVE":
        nxt   = _next_evening_open_et(now_et)
        delta = nxt - now_et
        h, r  = divmod(int(delta.total_seconds()), 3600)
        m     = r // 60
        tbl.add_row("Next open:",
                    Text(f"{nxt.strftime('%a %b %d')} 18:00 ET"
                         f"  in {h}h {m:02d}m", style="dim"))
        tbl.add_row("Threshold:", f"≥ {EVE_GAP_THRESH*100:.1f}%  "
                    f"(strong: {EVE_GAP_THRESH_STRONG*100:.1f}%)")
        tbl.add_row("Trading:", Text("Display only — not yet traded", style="dim"))
        border, title = "dim", "EVENING RESUMPTION  ·  inactive"

    elif s == "STANDBY":
        open_t = datetime(now_et.year, now_et.month, now_et.day, 18, 0, tzinfo=ET)
        mins   = int((open_t - now_et).total_seconds() / 60)
        tbl.add_row("Status:", Text(f"Standby — {mins}m to open", style="yellow"))
        if state.prev_close:
            tbl.add_row("Pre-gap close:", Text(f"{state.prev_close:,.2f}", style="white"))
        tbl.add_row("Threshold:", f"≥ {EVE_GAP_THRESH*100:.1f}%")
        tbl.add_row("Trading:", Text("Display only", style="dim"))
        border, title = "yellow", "EVENING RESUMPTION  ·  standby"

    elif s == "FIRST_CANDLE":
        tbl.add_row("Status:", Text("First candle forming …", style="yellow bold"))
        if state.prev_close and state.first_bar_open:
            gap   = math.log(state.first_bar_open / state.prev_close)
            color = "green" if gap > 0 else "red"
            tbl.add_row("Pre-gap close:", f"{state.prev_close:,.2f}")
            tbl.add_row("Open:", Text(f"{state.first_bar_open:,.2f}", style=color))
            _gap_row(tbl, "Gap (open):", gap, EVE_GAP_THRESH, EVE_GAP_THRESH_STRONG)
            tbl.add_row("", _bar(gap, 0.01))
        tbl.add_row("Trading:", Text("Display only", style="dim"))
        border, title = "yellow", "EVENING RESUMPTION  ·  first candle"

    elif s in ("SIGNAL_EVAL", "DONE"):
        go, gc = state.gap_open_pct, state.gap_close_pct
        tbl.add_row("Pre-gap close:", f"{state.prev_close:,.2f}")
        tbl.add_row("First open:",    f"{state.first_bar_open:,.2f}")
        tbl.add_row("First close:",   f"{state.first_bar_close:,.2f}")
        _gap_row(tbl, "Gap (open):",  go, EVE_GAP_THRESH, EVE_GAP_THRESH_STRONG)
        _gap_row(tbl, "Gap (close):", gc, EVE_GAP_THRESH, EVE_GAP_THRESH_STRONG)
        tbl.add_row("", _bar(go, 0.01))
        tbl.add_row("Volume:", f"{state.first_bar_vol:,.0f}")

        border = _signal_status_row(tbl, state, EVE_HOLD_BARS, now_et, in_trade=False)
        tbl.add_row("Trading:", Text("Display only — not yet traded", style="dim"))
        title = {"SIGNAL_EVAL": "EVENING RESUMPTION  ·  evaluating",
                 "DONE":        "EVENING RESUMPTION  ·  complete"}[s]
    else:
        border, title = "dim", "EVENING RESUMPTION"

    return Panel(tbl, title=title, border_style=border, box=box.ROUNDED)


def build_evening_history_panel(state: EveningOpenState) -> Panel:
    tbl = Table(box=None, show_header=True, padding=(0, 1))
    tbl.add_column("Date",   style="dim", width=7)
    tbl.add_column("Day",    style="dim", width=4)
    tbl.add_column("Gap",    justify="right", width=8)
    tbl.add_column("Sig",    justify="center", width=4)
    tbl.add_column("Result", justify="center", width=8)

    recent = state.history[-8:] if state.history else []
    for h in reversed(recent):
        sig, res = _outcome_text(h)
        tbl.add_row(h.ts.astimezone(ET).strftime("%b %d"),
                    DOW_NAMES[h.dow],
                    f"{h.gap_open*100:+.2f}%",
                    sig, res)

    sigs  = [h for h in state.history if h.signal_dir != 0]
    conts = sum(1 for h in sigs if h.outcome == "cont")
    note  = Text(
        f"  {len(sigs)} signals (≥{EVE_GAP_THRESH*100:.1f}%)  •  "
        f"{conts/len(sigs)*100:.0f}% cont  •  display only"
        if sigs else "  No signals above threshold", style="dim"
    )
    return Panel(Group(tbl, note),
                 title=f"Evening Resumptions  (last 8, gap ≥ {EVE_HIST_MIN_GAP*100:.1f}%)",
                 border_style="dim", box=box.ROUNDED)


# ── Header ────────────────────────────────────────────────────────────────────

def build_header(now_et: datetime,
                 sun: SundayOpenState,
                 eve: EveningOpenState) -> Panel:
    active_statuses = [s for s in (sun.status, eve.status) if s != "INACTIVE"]
    overall = active_statuses[0] if active_statuses else "INACTIVE"
    color   = {"STANDBY": "yellow", "FIRST_CANDLE": "yellow bold",
               "SIGNAL_EVAL": "cyan bold", "IN_TRADE": "green bold",
               "DONE": "dim"}.get(overall, "dim")

    t = Text()
    t.append("  GLOBEX MONITOR  ", style="bold white on dark_blue")
    t.append("   │   ", style="dim")
    t.append(now_et.strftime("%a %Y-%m-%d  %H:%M:%S %Z"), style="dim")
    t.append("   │   ", style="dim")
    abbrev = {"INACTIVE": "·", "STANDBY": "stby", "FIRST_CANDLE": "1st",
              "SIGNAL_EVAL": "SIG", "IN_TRADE": "LIVE", "DONE": "done"}
    t.append(f"Sun:{abbrev.get(sun.status, sun.status)}  "
             f"Eve:{abbrev.get(eve.status, eve.status)}", style=color)
    return Panel(t, box=box.ROUNDED, border_style="dim", expand=False)


# ── State machine ─────────────────────────────────────────────────────────────

def _active_window(now: datetime, open_hour: int = 18) -> str:
    """Return INACTIVE/STANDBY/FIRST_CANDLE/SIGNAL_EVAL for an 18:00 open."""
    open_t = datetime(now.year, now.month, now.day, open_hour, 0, tzinfo=ET)
    if now < open_t - timedelta(minutes=WINDOW_PRE_MIN):
        return "INACTIVE"
    if now >= open_t + timedelta(minutes=WINDOW_POST_MIN):
        return "INACTIVE"
    if now < open_t:
        return "STANDBY"
    if now < open_t + timedelta(minutes=BAR_MINUTES):
        return "FIRST_CANDLE"
    return "SIGNAL_EVAL"


def update_sunday(client, contract_id: str, state: SundayOpenState, now_et: datetime):
    if state.status in ("IN_TRADE",):
        if state.signal_exit_at and now_et >= state.signal_exit_at.astimezone(ET):
            state.status = "DONE"
        return
    if state.status == "DONE":
        return
    if now_et.weekday() != 6:
        state.status = "INACTIVE"
        return

    sub = _active_window(now_et)
    if sub == "INACTIVE":
        state.status = "INACTIVE"
        return

    if sub == "STANDBY":
        if state.fri_close == 0:
            cutoff = datetime(now_et.year, now_et.month, now_et.day, 17, 0, tzinfo=ET)
            state.fri_close = fetch_prev_close(client, contract_id, cutoff)
        state.status = "STANDBY"

    elif sub == "FIRST_CANDLE":
        bar = fetch_current_bar(client, contract_id)
        if bar:
            state.first_bar_open = float(bar["o"])
            state.first_bar_vol  = float(bar.get("v", 0))
        if state.fri_close == 0:
            cutoff = datetime(now_et.year, now_et.month, now_et.day, 17, 0, tzinfo=ET)
            state.fri_close = fetch_prev_close(client, contract_id, cutoff)
        state.status = "FIRST_CANDLE"

    elif sub == "SIGNAL_EVAL" and not state.candle_complete:
        bar = fetch_current_bar(client, contract_id)
        if bar and state.fri_close:
            state.first_bar_open  = float(bar["o"])
            state.first_bar_close = float(bar["c"])
            state.first_bar_vol   = float(bar.get("v", 0))
            state.gap_open_pct    = math.log(state.first_bar_open  / state.fri_close)
            state.gap_close_pct   = math.log(state.first_bar_close / state.fri_close)
            state.candle_complete = True
            sig = evaluate_sunday_signal(state)
            state.signal_dir   = sig
            state.signal_entry = state.first_bar_close
            if sig != 0:
                state.status         = "IN_TRADE"
                state.signal_fired_at = now_et
                state.signal_exit_at  = (
                    now_et.astimezone(timezone.utc)
                    + timedelta(minutes=SUN_HOLD_BARS * BAR_MINUTES)
                ).astimezone(ET)
            else:
                state.status = "SIGNAL_EVAL"


def update_evening(client, contract_id: str, state: EveningOpenState, now_et: datetime):
    if state.status == "DONE":
        return
    if now_et.weekday() > 4:   # Sat/Sun — not an evening session
        state.status = "INACTIVE"
        return

    sub = _active_window(now_et)
    if sub == "INACTIVE":
        # Reset for next day if we're past the window
        if state.candle_complete:
            state.status = "DONE"
        else:
            state.status = "INACTIVE"
        return

    if sub == "STANDBY":
        if state.prev_close == 0:
            cutoff = datetime(now_et.year, now_et.month, now_et.day, 17, 0, tzinfo=ET)
            state.prev_close = fetch_prev_close(client, contract_id, cutoff)
        state.status = "STANDBY"

    elif sub == "FIRST_CANDLE":
        bar = fetch_current_bar(client, contract_id)
        if bar:
            state.first_bar_open = float(bar["o"])
            state.first_bar_vol  = float(bar.get("v", 0))
        if state.prev_close == 0:
            cutoff = datetime(now_et.year, now_et.month, now_et.day, 17, 0, tzinfo=ET)
            state.prev_close = fetch_prev_close(client, contract_id, cutoff)
        state.status = "FIRST_CANDLE"

    elif sub == "SIGNAL_EVAL" and not state.candle_complete:
        bar = fetch_current_bar(client, contract_id)
        if bar and state.prev_close:
            state.first_bar_open  = float(bar["o"])
            state.first_bar_close = float(bar["c"])
            state.first_bar_vol   = float(bar.get("v", 0))
            state.gap_open_pct    = math.log(state.first_bar_open  / state.prev_close)
            state.gap_close_pct   = math.log(state.first_bar_close / state.prev_close)
            state.candle_complete = True
            state.signal_dir      = evaluate_evening_signal(state)
            state.signal_entry    = state.first_bar_close
            state.status          = "SIGNAL_EVAL"


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_live():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    print("Loading history from CSV …", flush=True)
    sun_history, vol_baseline = load_sunday_history()
    eve_history               = load_evening_history()
    print(f"  Sunday:  {len(sun_history)} events  vol baseline={vol_baseline:,.0f}")
    print(f"  Evening: {len(eve_history)} significant gaps (≥{EVE_HIST_MIN_GAP*100:.1f}%)")

    sun = SundayOpenState(history=sun_history, vol_baseline=vol_baseline)
    eve = EveningOpenState(history=eve_history)

    with TopstepClient() as client:
        contracts = client.search_contracts("MES")
        if not contracts:
            print("No MES contract found.")
            return
        contract_id = contracts[0]["id"]
        print(f"  Contract: {contracts[0]['name']}  id={contract_id}", flush=True)

        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                now_et = datetime.now(ET)
                try:
                    update_sunday(client, contract_id, sun, now_et)
                    update_evening(client, contract_id, eve, now_et)
                except Exception:
                    pass

                layout = Table.grid(expand=True)
                layout.add_column(ratio=1)
                layout.add_row(build_header(now_et, sun, eve))
                layout.add_row(_side_by_side(
                    Group(build_sunday_panel(sun, now_et),
                          build_sunday_history_panel(sun)),
                    Group(build_evening_panel(eve, now_et),
                          build_evening_history_panel(eve)),
                ))

                live.update(layout)
                time.sleep(POLL_SECONDS)


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    sun_history, vol_baseline = load_sunday_history()
    eve_history               = load_evening_history()

    # ── Snapshot 1: both inactive (Thursday afternoon) ────────────────────
    sun1 = SundayOpenState(history=sun_history, vol_baseline=vol_baseline,
                            status="INACTIVE")
    eve1 = EveningOpenState(history=eve_history, status="INACTIVE")
    now1 = datetime(2026, 3, 26, 14, 30, tzinfo=ET)

    console.print("\n[bold]── Demo 1: INACTIVE  (Thu 14:30 ET) ──[/bold]")
    console.print(build_header(now1, sun1, eve1))
    console.print(_side_by_side(
        Group(build_sunday_panel(sun1, now1), build_sunday_history_panel(sun1)),
        Group(build_evening_panel(eve1, now1), build_evening_history_panel(eve1)),
    ))

    # ── Snapshot 2: Sunday first candle / Evening signal eval ─────────────
    sun2 = SundayOpenState(history=sun_history, vol_baseline=vol_baseline,
                            status="FIRST_CANDLE",
                            fri_close=5720.50, first_bar_open=5763.00)
    eve2 = EveningOpenState(
        history=eve_history, status="SIGNAL_EVAL",
        prev_close=5720.50, first_bar_open=5732.75, first_bar_close=5728.00,
        first_bar_vol=420.0,
        gap_open_pct=math.log(5732.75 / 5720.50),
        gap_close_pct=math.log(5728.00 / 5720.50),
        candle_complete=True, signal_dir=1, signal_entry=5728.00,
    )
    now2 = datetime(2026, 3, 27, 18, 6, tzinfo=ET)   # Friday 18:06

    console.print("\n[bold]── Demo 2: Sun FIRST_CANDLE  /  Eve SIGNAL_EVAL  (Fri 18:06 ET) ──[/bold]")
    console.print(build_header(now2, sun2, eve2))
    console.print(_side_by_side(
        Group(build_sunday_panel(sun2, now2), build_sunday_history_panel(sun2)),
        Group(build_evening_panel(eve2, now2), build_evening_history_panel(eve2)),
    ))

    # ── Snapshot 3: Sunday IN_TRADE ───────────────────────────────────────
    sun3 = SundayOpenState(
        history=sun_history, vol_baseline=8000.0, status="IN_TRADE",
        fri_close=5720.50, first_bar_open=5763.00, first_bar_close=5771.25,
        first_bar_vol=14200.0,
        gap_open_pct=math.log(5763.00 / 5720.50),
        gap_close_pct=math.log(5771.25 / 5720.50),
        candle_complete=True, signal_dir=1, signal_entry=5771.25,
        signal_fired_at=datetime(2026, 3, 29, 18, 5, tzinfo=ET),
        signal_exit_at=datetime(2026, 3, 29, 18, 35, tzinfo=ET),
    )
    eve3 = EveningOpenState(history=eve_history, status="INACTIVE")
    now3 = datetime(2026, 3, 29, 18, 12, tzinfo=ET)  # Sunday 18:12

    console.print("\n[bold]── Demo 3: Sun IN_TRADE  (Sun 18:12 ET) ──[/bold]")
    console.print(build_header(now3, sun3, eve3))
    console.print(_side_by_side(
        Group(build_sunday_panel(sun3, now3), build_sunday_history_panel(sun3)),
        Group(build_evening_panel(eve3, now3), build_evening_history_panel(eve3)),
    ))


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_live()
