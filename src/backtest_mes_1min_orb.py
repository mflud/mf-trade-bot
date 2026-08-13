"""
backtest_mes_1min_orb.py — Parameter sweep for 1-minute ORB on MES.

Opening range = first 1-min bar (9:30–9:31 ET).
Mirrors the MNQ 1-min ORB backtest structure.

Sweeps:
  - entry_type : 'touch' (high/low touched) | 'close' (bar closes beyond)
  - stop_type  : 'midpoint' | 'far_side' (other edge of ORB)
  - target_mult: 1.0 | 1.5 | 2.0 | 2.5 | 3.0  (× ORB width)
  - entry_win  : 5 | 10 | 15 | 30 | 60  (minutes after 9:31 to look for break)
  - direction  : 'first' (first break only) | 'both' (trade both sides)
  - width_min  : minimum ORB width in points (0 = no filter)
  - width_max_bps: maximum ORB width as % of price (0 = no filter) — same ≤30bps used for MNQ

Reports top 20 by profit factor, top 20 by P&L, high-WR filter,
and monthly breakdown for the best combo.

Usage:
  python src/backtest_mes_1min_orb.py
"""

import sys
from datetime import datetime, timezone
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET          = ZoneInfo("America/New_York")
SYMBOL      = "MES"
POINT_VALUE = 5.0      # MES = $5/point
COMMISSION  = 2.0 * 2  # $2/side × 2 sides (approximate)

# ── Load bars from bars.db ─────────────────────────────────────────────────────

def fetch_bars() -> pd.DataFrame:
    import sqlite3
    print("Loading MES 1-min bars from bars.db…", flush=True)
    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol='MES' AND minutes=1 ORDER BY ts"
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError("No MES 1-min bars in bars.db")
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    # Filter settlement gap (21:00–22:00 UTC)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)
    print(f"Loaded {len(df)} bars  ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})",
          flush=True)
    return df


# ── Session builder ────────────────────────────────────────────────────────────

def build_sessions(df: pd.DataFrame) -> list[dict]:
    df["et"]      = df["ts"].dt.tz_convert(ET)
    df["date_et"] = df["et"].dt.date
    df["hm"]      = df["et"].dt.hour * 60 + df["et"].dt.minute

    sessions = []
    for date, grp in df.groupby("date_et"):
        grp = grp.sort_values("et")
        orb_rows = grp[grp["hm"] == 9 * 60 + 30]
        if orb_rows.empty:
            continue
        orb  = orb_rows.iloc[0]
        rest = grp[(grp["hm"] >= 9 * 60 + 31) & (grp["hm"] < 16 * 60)]
        if rest.empty:
            continue
        sessions.append({
            "date":      date,
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
                entry_type: str,       # 'touch' | 'close'
                stop_type:  str,       # 'midpoint' | 'far_side'
                target_mult: float,
                entry_win:  int,       # minutes after 9:31 cutoff
                direction:  str,       # 'first' | 'both'
                width_min:  float,     # pts
                width_max_bps: float,  # max ORB width as fraction of price (0 = off)
                ) -> list[dict]:
    orb_h = sess["orb_high"]
    orb_l = sess["orb_low"]
    width = orb_h - orb_l
    mid   = (orb_h + orb_l) / 2

    if width < width_min:
        return []
    if width_max_bps > 0 and mid > 0 and width / mid > width_max_bps:
        return []

    target_pts = width * target_mult
    if stop_type == "midpoint":
        stop_pts_long  = orb_h - mid
        stop_pts_short = mid - orb_l
    else:  # far_side
        stop_pts_long  = width
        stop_pts_short = width

    cutoff_hm = 9 * 60 + 31 + entry_win

    trades     = []
    long_done  = False
    short_done = False

    for bar in sess["bars"]:
        hm = bar["hm"]
        if hm >= cutoff_hm:
            break

        # Check LONG break
        if not long_done and (direction == "both" or not short_done):
            triggered = (bar["h"] >= orb_h) if entry_type == "touch" else (bar["c"] > orb_h)
            if triggered:
                entry = orb_h
                tgt   = entry + target_pts
                stp   = entry - stop_pts_long
                exit_p, result = _sim_exit(entry, tgt, stp, 1, bar, sess["bars"])
                pnl = (exit_p - entry) * POINT_VALUE if result != "stop" else -(stop_pts_long * POINT_VALUE)
                pnl -= COMMISSION
                trades.append(dict(dir=1, entry=entry, target=tgt, stop=stp,
                                   exit=exit_p, result=result, pnl=pnl))
                long_done = True
                if direction == "first":
                    break

        # Check SHORT break
        if not short_done and (direction == "both" or not long_done):
            triggered = (bar["l"] <= orb_l) if entry_type == "touch" else (bar["c"] < orb_l)
            if triggered:
                entry = orb_l
                tgt   = entry - target_pts
                stp   = entry + stop_pts_short
                exit_p, result = _sim_exit(entry, tgt, stp, -1, bar, sess["bars"])
                pnl = (entry - exit_p) * POINT_VALUE if result != "stop" else -(stop_pts_short * POINT_VALUE)
                pnl -= COMMISSION
                trades.append(dict(dir=-1, entry=entry, target=tgt, stop=stp,
                                   exit=exit_p, result=result, pnl=pnl))
                short_done = True
                if direction == "first":
                    break

    return trades


def _sim_exit(entry, tgt, stp, direction, trigger_bar, all_bars):
    trigger_hm = trigger_bar["hm"]
    future = [b for b in all_bars if b["hm"] > trigger_hm and b["hm"] < 16 * 60]
    if direction == 1:
        for b in future:
            if b["l"] <= stp: return stp, "stop"
            if b["h"] >= tgt: return tgt, "target"
        return future[-1]["c"] if future else entry, "eod"
    else:
        for b in future:
            if b["h"] >= stp: return stp, "stop"
            if b["l"] <= tgt: return tgt, "target"
        return future[-1]["c"] if future else entry, "eod"


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_sweep(sessions: list[dict]) -> pd.DataFrame:
    entry_types    = ["touch", "close"]
    stop_types     = ["midpoint", "far_side"]
    target_mults   = [1.0, 1.5, 2.0, 2.5, 3.0]
    entry_wins     = [5, 10, 15, 30, 60]
    directions     = ["first", "both"]
    width_mins     = [0.0, 1.0, 2.0]
    width_max_bpss = [0.0, 0.003]   # 0 = no cap, 0.003 = ≤30bps (matches MNQ live)

    combos = list(product(entry_types, stop_types, target_mults,
                          entry_wins, directions, width_mins, width_max_bpss))
    print(f"\nRunning {len(combos)} parameter combos on {len(sessions)} sessions…", flush=True)

    results = []
    for i, (et, st, tm, ew, dr, wm, wb) in enumerate(combos):
        if i % 200 == 0:
            print(f"  {i}/{len(combos)}…", flush=True, end="\r")

        all_trades = []
        for sess in sessions:
            all_trades.extend(sim_session(sess, et, st, tm, ew, dr, wm, wb))

        if len(all_trades) < 20:
            continue

        pnls   = [t["pnl"] for t in all_trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total  = sum(pnls)
        wr     = len(wins) / len(pnls)
        pf     = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 999

        results.append(dict(
            entry_type=et, stop=st, target=tm, entry_win=ew,
            direction=dr, width_min=wm, width_max_bps=wb,
            trades=len(pnls), wr=round(wr, 3), pf=round(pf, 3),
            total=round(total, 0),
            avg_win=round(np.mean(wins) if wins else 0, 2),
            avg_loss=round(np.mean(losses) if losses else 0, 2),
        ))

    print(f"\nDone. {len(results)} combos with ≥20 trades.", flush=True)
    return pd.DataFrame(results).sort_values("pf", ascending=False)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = fetch_bars()
    sessions = build_sessions(df)
    print(f"Sessions with valid ORB bar: {len(sessions)}", flush=True)

    results = run_sweep(sessions)

    print("\n" + "="*95)
    print("TOP 20 COMBOS BY PROFIT FACTOR")
    print("="*95)
    print(results.head(20).to_string(index=False))

    by_total = results.sort_values("total", ascending=False).head(20)
    print("\n" + "="*95)
    print("TOP 20 COMBOS BY TOTAL P&L")
    print("="*95)
    print(by_total.to_string(index=False))

    high_wr = results[(results["wr"] >= 0.60) & (results["pf"] >= 1.3)].sort_values("total", ascending=False)
    print(f"\n{'='*95}")
    print(f"DECENT WIN-RATE COMBOS (WR≥60%, PF≥1.3)  — {len(high_wr)} found")
    print("="*95)
    if not high_wr.empty:
        print(high_wr.head(20).to_string(index=False))
    else:
        print("  none")

    # Monthly breakdown for best combo by PF
    best = results.iloc[0]
    print(f"\n{'='*95}")
    print(f"MONTHLY BREAKDOWN — best combo: entry={best.entry_type} stop={best.stop} "
          f"target={best.target}× win={best.entry_win}min dir={best.direction} "
          f"wmin={best.width_min}pt wmax={best.width_max_bps*10000:.0f}bps")
    print("="*95)

    monthly: dict = {}
    for sess in sessions:
        trades = sim_session(sess, best.entry_type, best.stop, best.target,
                             int(best.entry_win), best.direction,
                             best.width_min, best.width_max_bps)
        ym = (sess["date"].year, sess["date"].month)
        monthly.setdefault(ym, []).extend(trades)

    print(f"{'Month':<12} {'Trades':>7} {'WR':>7} {'Total$':>9} {'PF':>6}")
    print("-" * 46)
    for ym in sorted(monthly):
        ts = monthly[ym]
        if not ts: continue
        pnls = [t["pnl"] for t in ts]
        wins = [p for p in pnls if p > 0]
        lss  = [p for p in pnls if p <= 0]
        pf_m = sum(wins) / abs(sum(lss)) if lss and sum(lss) != 0 else 999
        print(f"{ym[0]}-{ym[1]:02d}       {len(pnls):>6}  {len(wins)/len(pnls):>6.1%}  "
              f"{sum(pnls):>8.0f}  {pf_m:>5.2f}")

    # Also show the MNQ-equivalent params for direct comparison
    mnq_equiv = results[
        (results["entry_type"] == "close") &
        (results["stop"] == "far_side") &
        (results["target"] == 1.0) &
        (results["entry_win"] == 5) &
        (results["direction"] == "both") &
        (results["width_max_bps"] == 0.003)
    ]
    print(f"\n{'='*95}")
    print("MNQ-EQUIVALENT PARAMS (close/far_side/1×/5min/both/≤30bps)")
    print("="*95)
    if not mnq_equiv.empty:
        print(mnq_equiv.to_string(index=False))
    else:
        print("  No results (fewer than 20 trades)")

    results.to_csv("mes_1min_orb_sweep.csv", index=False)
    print("\nFull results saved to mes_1min_orb_sweep.csv")


if __name__ == "__main__":
    main()
