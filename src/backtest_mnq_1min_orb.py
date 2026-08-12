"""
backtest_mnq_1min_orb.py — Parameter sweep for 1-minute ORB on MNQ.

Opening range = first 1-min bar (9:30–9:31 ET).
Sweeps:
  - entry_type : 'touch' (high/low touched) | 'close' (bar closes beyond)
  - stop_type  : 'midpoint' | 'far_side' (other edge of ORB)
  - target_mult: 1.0 | 1.5 | 2.0 | 2.5 | 3.0  (× ORB width)
  - entry_win  : 5 | 10 | 15 | 30 | 60  (minutes after 9:31 to look for break)
  - direction  : 'first' (first break only) | 'both' (trade both sides)
  - width_min  : minimum ORB width in points (0 = no filter)

Reports top 20 combos by profit factor, then a fixed "baseline" run.
"""

import sys
from datetime import datetime, timezone
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET          = ZoneInfo("America/New_York")
SYMBOL      = "MNQ"
POINT_VALUE = 2.0      # MNQ = $2/point
COMMISSION  = 2.0 * 2  # $2/side × 2 sides

# ── Load bars from bars.db ─────────────────────────────────────────────────────

def fetch_bars() -> pd.DataFrame:
    import sqlite3
    print("Loading MNQ 1-min bars from bars.db…", flush=True)
    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol='MNQ' AND minutes=1 ORDER BY ts"
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("No MNQ 1-min bars in bars.db")
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    # Filter settlement gap (21:00–22:00 UTC)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)
    print(f"Loaded {len(df)} bars  ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})",
          flush=True)
    return df


# ── Session builder ────────────────────────────────────────────────────────────

def build_sessions(df: pd.DataFrame) -> list[dict]:
    """
    For each RTH session, extract:
      - orb_bar   : the 9:30 ET 1-min bar (ORB range)
      - rth_bars  : subsequent 1-min bars within session
    """
    df["et"] = df["ts"].dt.tz_convert(ET)
    df["date_et"] = df["et"].dt.date
    df["hm"] = df["et"].dt.hour * 60 + df["et"].dt.minute

    sessions = []
    for date, grp in df.groupby("date_et"):
        grp = grp.sort_values("et")
        orb_rows = grp[grp["hm"] == 9 * 60 + 30]
        if orb_rows.empty:
            continue
        orb = orb_rows.iloc[0]
        rest = grp[(grp["hm"] >= 9 * 60 + 31) & (grp["hm"] < 16 * 60)]
        if rest.empty:
            continue
        sessions.append({
            "date":     date,
            "orb_open":  float(orb["o"]),
            "orb_high":  float(orb["h"]),
            "orb_low":   float(orb["l"]),
            "orb_close": float(orb["c"]),
            "orb_vol":   float(orb.get("v", 0)),
            "bars":      rest[["hm", "o", "h", "l", "c", "v"]].to_dict("records"),
        })
    return sessions


# ── Single-session trade simulator ────────────────────────────────────────────

def sim_session(sess: dict,
                entry_type: str,   # 'touch' | 'close'
                stop_type:  str,   # 'midpoint' | 'far_side'
                target_mult: float,
                entry_win:  int,   # minutes after 9:31 cutoff
                direction:  str,   # 'first' | 'both'
                width_min:  float,
                ) -> list[dict]:
    """
    Returns list of trade dicts: {dir, entry, target, stop, exit, pnl_pts, result}
    """
    orb_h = sess["orb_high"]
    orb_l = sess["orb_low"]
    width = orb_h - orb_l
    mid   = (orb_h + orb_l) / 2

    if width < width_min:
        return []

    target_pts = width * target_mult
    if stop_type == "midpoint":
        stop_pts_long  = orb_h - mid   # entry near orb_h; stop at mid
        stop_pts_short = mid - orb_l
    else:  # far_side
        stop_pts_long  = width          # entry near orb_h; stop at orb_l
        stop_pts_short = width

    cutoff_hm = 9 * 60 + 31 + entry_win  # last bar allowed to enter

    trades    = []
    long_done = False
    short_done= False

    for bar in sess["bars"]:
        hm = bar["hm"]
        if hm >= cutoff_hm:
            break

        # Check LONG break
        if not long_done and (direction == "both" or not short_done):
            triggered = False
            if entry_type == "touch":
                triggered = bar["h"] >= orb_h
            else:  # close
                triggered = bar["c"] > orb_h

            if triggered:
                entry = orb_h  # assume entry at ORB high (breakout price)
                tgt   = entry + target_pts
                stp   = entry - stop_pts_long
                # Simulate exit on remaining bars
                exit_p, result = _sim_exit(entry, tgt, stp, 1, bar, sess["bars"])
                pnl = (exit_p - entry) * POINT_VALUE * (1 if result != "stop" else -1)
                if result == "stop":
                    pnl = -(stop_pts_long * POINT_VALUE)
                else:
                    pnl = (exit_p - entry) * POINT_VALUE
                pnl -= COMMISSION
                trades.append(dict(dir=1, entry=entry, target=tgt, stop=stp,
                                   exit=exit_p, result=result, pnl=pnl))
                long_done = True
                if direction == "first":
                    break

        # Check SHORT break
        if not short_done and (direction == "both" or not long_done):
            triggered = False
            if entry_type == "touch":
                triggered = bar["l"] <= orb_l
            else:
                triggered = bar["c"] < orb_l

            if triggered:
                entry = orb_l
                tgt   = entry - target_pts
                stp   = entry + stop_pts_short
                exit_p, result = _sim_exit(entry, tgt, stp, -1, bar, sess["bars"])
                if result == "stop":
                    pnl = -(stop_pts_short * POINT_VALUE)
                else:
                    pnl = (entry - exit_p) * POINT_VALUE
                pnl -= COMMISSION
                trades.append(dict(dir=-1, entry=entry, target=tgt, stop=stp,
                                   exit=exit_p, result=result, pnl=pnl))
                short_done = True
                if direction == "first":
                    break

    return trades


def _sim_exit(entry, tgt, stp, direction, trigger_bar, all_bars):
    """Simulate exit on bars following trigger. Returns (exit_price, result)."""
    # Check remaining bars after the trigger bar for target/stop
    trigger_hm = trigger_bar["hm"]
    future = [b for b in all_bars if b["hm"] > trigger_hm and b["hm"] < 16 * 60]

    if direction == 1:
        for b in future:
            if b["l"] <= stp:
                return stp, "stop"
            if b["h"] >= tgt:
                return tgt, "target"
        # EOD exit
        return future[-1]["c"] if future else entry, "eod"
    else:
        for b in future:
            if b["h"] >= stp:
                return stp, "stop"
            if b["l"] <= tgt:
                return tgt, "target"
        return future[-1]["c"] if future else entry, "eod"


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_sweep(sessions: list[dict]) -> pd.DataFrame:
    entry_types  = ["touch", "close"]
    stop_types   = ["midpoint", "far_side"]
    target_mults = [1.0, 1.5, 2.0, 2.5, 3.0]
    entry_wins   = [5, 10, 15, 30, 60]
    directions   = ["first", "both"]
    width_mins   = [0.0, 1.0, 2.0]

    combos = list(product(entry_types, stop_types, target_mults,
                          entry_wins, directions, width_mins))
    print(f"\nRunning {len(combos)} parameter combos on {len(sessions)} sessions…", flush=True)

    results = []
    for i, (et, st, tm, ew, dr, wm) in enumerate(combos):
        if i % 100 == 0:
            print(f"  {i}/{len(combos)}…", flush=True, end="\r")

        all_trades = []
        for sess in sessions:
            all_trades.extend(sim_session(sess, et, st, tm, ew, dr, wm))

        if len(all_trades) < 20:
            continue

        pnls     = [t["pnl"] for t in all_trades]
        wins     = [p for p in pnls if p > 0]
        losses   = [p for p in pnls if p <= 0]
        total    = sum(pnls)
        wr       = len(wins) / len(pnls)
        pf       = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 999
        avg_win  = np.mean(wins)  if wins   else 0
        avg_loss = np.mean(losses) if losses else 0

        results.append(dict(
            entry_type=et, stop=st, target=tm, entry_win=ew,
            direction=dr, width_min=wm,
            trades=len(pnls), wr=round(wr, 3), pf=round(pf, 3),
            total=round(total, 0), avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
        ))

    print(f"\nDone. {len(results)} combos with ≥20 trades.", flush=True)
    return pd.DataFrame(results).sort_values("pf", ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = fetch_bars()
    sessions = build_sessions(df)
    print(f"Sessions with valid ORB bar: {len(sessions)}", flush=True)

    results = run_sweep(sessions)

    # ── Top 20 by profit factor ────────────────────────────────────────────────
    print("\n" + "="*90)
    print("TOP 20 COMBOS BY PROFIT FACTOR")
    print("="*90)
    top = results.head(20)
    print(top.to_string(index=False))

    # ── Top 20 by total P&L ───────────────────────────────────────────────────
    by_total = results.sort_values("total", ascending=False).head(20)
    print("\n" + "="*90)
    print("TOP 20 COMBOS BY TOTAL P&L")
    print("="*90)
    print(by_total.to_string(index=False))

    # ── High WR focus (WR ≥ 0.75, PF ≥ 1.5) ──────────────────────────────────
    high_wr = results[(results["wr"] >= 0.75) & (results["pf"] >= 1.5)].sort_values("total", ascending=False)
    print(f"\n{'='*90}")
    print(f"HIGH WIN-RATE COMBOS (WR≥75%, PF≥1.5)  — {len(high_wr)} found")
    print("="*90)
    if not high_wr.empty:
        print(high_wr.head(20).to_string(index=False))
    else:
        print("  none")

    # ── Monthly breakdown for best combo ──────────────────────────────────────
    best = results.iloc[0]
    print(f"\n{'='*90}")
    print(f"MONTHLY BREAKDOWN — best combo: entry={best.entry_type} stop={best.stop} "
          f"target={best.target}× win={best.entry_win}min dir={best.direction} wmin={best.width_min}pt")
    print("="*90)

    monthly: dict = {}
    for sess in sessions:
        trades = sim_session(sess, best.entry_type, best.stop, best.target,
                             int(best.entry_win), best.direction, best.width_min)
        ym = (sess["date"].year, sess["date"].month)
        if ym not in monthly:
            monthly[ym] = []
        monthly[ym].extend(trades)

    print(f"{'Month':<12} {'Trades':>7} {'WR':>7} {'Total$':>9} {'PF':>6}")
    print("-" * 46)
    for ym in sorted(monthly):
        ts = monthly[ym]
        if not ts:
            continue
        pnls = [t["pnl"] for t in ts]
        wins = [p for p in pnls if p > 0]
        lss  = [p for p in pnls if p <= 0]
        pf_m = sum(wins) / abs(sum(lss)) if lss and sum(lss) != 0 else 999
        print(f"{ym[0]}-{ym[1]:02d}       {len(pnls):>6}  {len(wins)/len(pnls):>6.1%}  "
              f"{sum(pnls):>8.0f}  {pf_m:>5.2f}")

    results.to_csv("mnq_1min_orb_sweep.csv", index=False)
    print("\nFull results saved to mnq_1min_orb_sweep.csv")


if __name__ == "__main__":
    main()
