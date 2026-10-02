"""
backtest_orb_stop_floor.py — Does a minimum stop-distance floor (or,
alternatively, a minimum width filter) help MES's midpoint-stop ORB and
ORB-cls?

Motivated by a live observation (2026-10-02): 3 MES trades (2 morning ORB,
1 ORB-cls) stopped out for ~2pt losses, all on narrow (4.25-4.75pt) ranges
— the midpoint stop on a narrow range is only ~2-2.4pt away, tight enough
to clip on routine noise. A scan of logs/orb_trades.csv's full MES history
shows the same pattern repeatedly: width <= ~9pt trades are STOPPED far
more often than wider ones, and every TARGET hit in the log has width >= 8pt.

This backtests the production config for both strategies (close entry,
midpoint stop, current target/entry-window/hold) against the full
bars.db history, sweeping:
  - stop_floor_pts: 0 (current, no floor) vs widening the stop to at
    least this many points when the raw midpoint distance is smaller.
  - width_min_pts: 0 (current, no filter) vs skipping the trade entirely
    when the range is narrower than this.
Both tested independently (floor OR filter, not combined) so the report
shows which lever actually helps, if either does.

Usage:
  python src/backtest_orb_stop_floor.py            # MES morning ORB
  python src/backtest_orb_stop_floor.py --cls       # MES ORB-cls
"""

import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = __import__("zoneinfo").ZoneInfo("America/New_York")
POINT_VALUE = 5.0
COMMISSION  = 4.0


def fetch_mes_bars() -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol='MES' AND minutes=1 ORDER BY ts"
    ).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)   # CME settlement gap
    df["et"]  = df["ts"].dt.tz_convert(ET)
    df["hm"]  = df["et"].dt.hour * 60 + df["et"].dt.minute
    df["date"] = df["et"].dt.date
    return df


def build_sessions(df: pd.DataFrame, range_hm: int) -> list[dict]:
    """range_hm: minute-of-day (ET) of the 1-min range bar — 9:30 for the
    morning ORB, 15:50 for ORB-cls."""
    sessions = []
    for idx in df.index[df["hm"] == range_hm]:
        row = df.iloc[idx]
        future = df.iloc[idx + 1: idx + 1 + 260]
        future = future[(future["ts"] - row["ts"]) <= pd.Timedelta(minutes=260)]
        if future.empty:
            continue
        sessions.append({
            "date": row["date"], "range_ts": row["ts"],
            "orb_high": float(row["h"]), "orb_low": float(row["l"]),
            "bars": future[["ts", "hm", "o", "h", "l", "c"]].to_dict("records"),
        })
    return sessions


def sim_session(sess: dict, target_mult: float, entry_win: int, hold_min: int,
                stop_floor_pts: float, width_min_pts: float) -> list[dict]:
    """direction='first' (one trade/session, matches live's morning_fired /
    orb_cls.fired latch), close-break entry, midpoint stop — mirrors
    production exactly except for the two levers under test."""
    orb_h, orb_l = sess["orb_high"], sess["orb_low"]
    width = orb_h - orb_l
    if width <= 0 or width < width_min_pts:
        return []
    mid = (orb_h + orb_l) / 2
    target_pts = width * target_mult

    for i, bar in enumerate(sess["bars"]):
        elapsed = (bar["ts"] - sess["range_ts"]).total_seconds() / 60.0
        if elapsed > entry_win:
            break
        direction = 0
        if bar["c"] > orb_h: direction = 1
        elif bar["c"] < orb_l: direction = -1
        if direction == 0:
            continue

        entry = bar["c"]
        raw_stop_pts = (orb_h - mid) if direction == 1 else (mid - orb_l)
        stop_pts = max(raw_stop_pts, stop_floor_pts)
        tgt = entry + direction * target_pts
        stp = entry - direction * stop_pts

        future = sess["bars"][i + 1:]
        window = [b for b in future if (b["ts"] - bar["ts"]).total_seconds() / 60.0 <= hold_min]
        result, exit_p = None, None
        for b in window:
            if direction == 1:
                if b["l"] <= stp: result, exit_p = "stop", stp; break
                if b["h"] >= tgt: result, exit_p = "target", tgt; break
            else:
                if b["h"] >= stp: result, exit_p = "stop", stp; break
                if b["l"] <= tgt: result, exit_p = "target", tgt; break
        if result is None:
            exit_p = window[-1]["c"] if window else entry
            result = "hold_exit"
        pnl = (exit_p - entry) * direction * POINT_VALUE - COMMISSION
        return [dict(result=result, pnl=pnl, width=width, raw_stop_pts=raw_stop_pts)]
    return []


def run_sweep(sessions: list[dict], target_mult: float, entry_win: int, hold_min: int,
             label: str):
    stop_floors = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    width_mins  = [0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    print(f"\n{'='*80}\n{label} — stop-floor sweep (width_min=0, i.e. no trades skipped)")
    print("="*80)
    rows = []
    for sf in stop_floors:
        trades = []
        for sess in sessions:
            trades.extend(sim_session(sess, target_mult, entry_win, hold_min, sf, 0.0))
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
        pf = (sum(wins)/abs(sum(losses))) if losses and sum(losses) else (999 if wins else 0)
        rows.append(dict(stop_floor=sf, trades=len(pnls),
                         wr=round(len(wins)/len(pnls), 3) if pnls else 0,
                         pf=round(pf, 3), total=round(sum(pnls), 1)))
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n{'='*80}\n{label} — width-min filter sweep (stop_floor=0, i.e. raw midpoint stop)")
    print("="*80)
    rows = []
    for wm in width_mins:
        trades = []
        for sess in sessions:
            trades.extend(sim_session(sess, target_mult, entry_win, hold_min, 0.0, wm))
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
        pf = (sum(wins)/abs(sum(losses))) if losses and sum(losses) else (999 if wins else 0)
        rows.append(dict(width_min=wm, trades=len(pnls),
                         wr=round(len(wins)/len(pnls), 3) if pnls else 0,
                         pf=round(pf, 3), total=round(sum(pnls), 1)))
    print(pd.DataFrame(rows).to_string(index=False))

    # Narrow-vs-wide split at the current (no-filter) baseline, for context
    trades0 = []
    for sess in sessions:
        trades0.extend(sim_session(sess, target_mult, entry_win, hold_min, 0.0, 0.0))
    narrow = [t for t in trades0 if t["width"] < 8]
    wide   = [t for t in trades0 if t["width"] >= 8]
    def _stats(ts):
        pnls = [t["pnl"] for t in ts]
        stopped = sum(1 for t in ts if t["result"] == "stop")
        return len(ts), stopped, round(sum(pnls), 1)
    n_n, n_s, n_t = _stats(narrow)
    w_n, w_s, w_t = _stats(wide)
    print(f"\nBaseline (no floor/filter) split by width: "
          f"narrow(<8pt) n={n_n} stopped={n_s} total={n_t}pt  |  "
          f"wide(>=8pt) n={w_n} stopped={w_s} total={w_t}pt")


def main():
    is_cls = "--cls" in sys.argv
    df = fetch_mes_bars()
    if is_cls:
        sessions = build_sessions(df, 15 * 60 + 50)
        print(f"ORB-cls sessions: {len(sessions)}")
        run_sweep(sessions, target_mult=3.25, entry_win=5, hold_min=15, label="MES ORB-cls")
    else:
        sessions = build_sessions(df, 9 * 60 + 30)
        print(f"Morning ORB sessions: {len(sessions)}")
        run_sweep(sessions, target_mult=1.0, entry_win=5, hold_min=10, label="MES morning ORB")


if __name__ == "__main__":
    main()
