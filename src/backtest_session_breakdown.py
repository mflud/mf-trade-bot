"""
backtest_session_breakdown.py — Break down trade performance by session.

Sessions:
  pre-mkt  : 08:30–09:30 ET  (pre-NYSE open, CME futures active)
  RTH      : 09:30–16:00 ET  (NYSE open → close)
  post-mkt : 16:00–18:00 ET  (post-close, still liquid)
  globex   : 18:00–08:30 ET  (overnight, thin)

Analyses all trade logs: bot_trades, slr_trades, pl_mom_trades, vwaslr_trades.

Usage:
    python src/backtest_session_breakdown.py
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ET = ZoneInfo("America/New_York")

LOGS = {
    "CSR/3σ":   Path("logs/bot_trades.csv"),
    "SLR":      Path("logs/slr_trades.csv"),
    "PL MOM":   Path("logs/pl_mom_trades.csv"),
    "VWASLR":   Path("logs/vwaslr_trades.csv"),
}

INSTRUMENTS = {"MES", "MNQ", "MYM", "M2K", "MNQ", "MES"}

def session(dt_et: datetime) -> str:
    hm = dt_et.hour * 60 + dt_et.minute
    if   8*60+30 <= hm <  9*60+30: return "pre-mkt (8:30–9:30)"
    elif 9*60+30 <= hm < 16*60:    return "RTH (9:30–16:00)"
    elif 16*60   <= hm < 18*60:    return "post-mkt (16:00–18:00)"
    else:                           return "globex (18:00–8:30)"

def load(path: Path, strat: str) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                fired = datetime.fromisoformat(r["fired_at"])
                if fired.tzinfo is None:
                    fired = fired.replace(tzinfo=timezone.utc)
                fired_et = fired.astimezone(ET)
                pnl = float(r.get("pnl_pts", 0) or 0)
                rows.append({
                    "strat":    strat,
                    "symbol":   r.get("symbol", ""),
                    "fired_et": fired_et,
                    "session":  session(fired_et),
                    "pnl_pts":  pnl,
                    "outcome":  r.get("outcome", ""),
                    "direction": r.get("direction", ""),
                })
            except Exception:
                continue
    return rows


def summarise(trades: list[dict], label: str):
    if not trades:
        return
    arr   = np.array([t["pnl_pts"] for t in trades])
    wins  = (arr > 0).sum()
    total = len(arr)
    print(f"  {label:<26}  N={total:>4}  "
          f"win={wins/total*100:>5.1f}%  "
          f"avg={arr.mean():>+7.2f}  "
          f"med={np.median(arr):>+7.2f}  "
          f"sum={arr.sum():>+8.2f}  "
          f"avg|x|={np.abs(arr).mean():>6.2f}")


def main():
    all_trades = []
    for strat, path in LOGS.items():
        trades = load(path, strat)
        all_trades.extend(trades)
        print(f"Loaded {len(trades):>4} trades  ← {path.name}")

    print()
    sessions_order = [
        "pre-mkt (8:30–9:30)",
        "RTH (9:30–16:00)",
        "post-mkt (16:00–18:00)",
        "globex (18:00–8:30)",
    ]

    # ── Overall by session ────────────────────────────────────────────────────
    print("=" * 80)
    print("ALL STRATEGIES  —  by session")
    print("=" * 80)
    print(f"  {'Session':<26}  {'N':>5}  {'win%':>6}  {'avg':>8}  {'med':>8}  "
          f"{'sum':>9}  {'avg|x|':>7}")
    print(f"  {'─'*72}")
    for sess in sessions_order:
        subset = [t for t in all_trades if t["session"] == sess]
        summarise(subset, sess)

    # ── Per strategy by session ───────────────────────────────────────────────
    for strat in LOGS:
        strat_trades = [t for t in all_trades if t["strat"] == strat]
        if not strat_trades:
            continue
        print()
        print(f"{'─'*80}")
        print(f"  {strat}  (N={len(strat_trades)})")
        print(f"  {'─'*72}")
        for sess in sessions_order:
            subset = [t for t in strat_trades if t["session"] == sess]
            if subset:
                summarise(subset, sess)

    # ── Per strategy + symbol by session ─────────────────────────────────────
    print()
    print("=" * 80)
    print("SLR + VWASLR  —  by symbol × session")
    print("=" * 80)
    for strat in ["SLR", "VWASLR"]:
        strat_trades = [t for t in all_trades if t["strat"] == strat]
        symbols = sorted({t["symbol"] for t in strat_trades})
        for sym in symbols:
            sym_trades = [t for t in strat_trades if t["symbol"] == sym]
            print(f"\n  {strat}  {sym}:")
            for sess in sessions_order:
                subset = [t for t in sym_trades if t["session"] == sess]
                if subset:
                    summarise(subset, f"  {sess}")

    # ── Pre-mkt detail: 8:30–9:00 vs 9:00–9:30 ───────────────────────────────
    print()
    print("=" * 80)
    print("PRE-MARKET SPLIT  (8:30–9:00 vs 9:00–9:30)")
    print("=" * 80)
    for label, lo, hi in [("8:30–9:00", 8*60+30, 9*60), ("9:00–9:30", 9*60, 9*60+30)]:
        subset = [t for t in all_trades
                  if lo <= t["fired_et"].hour*60 + t["fired_et"].minute < hi]
        summarise(subset, label)


if __name__ == "__main__":
    main()
