"""
Backtest: Wall Breakout signal parameter sweep.

Uses qualifying events from logs/wall_events.csv (peak_size 100–300, test_count >= 2,
RTH) and 1-min MES bars from data/bars.db to simulate bracketed trades across a grid
of (stop_pts, target_pts, hold_min) values.

Also prints MAE/MFE distributions to guide parameter selection.

Usage:
    python src/backtest_wall_break.py
"""

import sqlite3
import sys
from datetime import timedelta, timezone

import numpy as np
import pandas as pd
import pytz

# ── Config ────────────────────────────────────────────────────────────────────

WALL_EVENTS_CSV = "logs/wall_events.csv"
BARS_DB         = "data/bars.db"

WALL_MED_MIN   = 100
WALL_MED_MAX   = 300
MIN_TESTS      = 2

# Parameter grid
STOPS   = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
TARGETS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
HOLDS   = [5, 10, 15, 20]   # minutes

ET = pytz.timezone("America/New_York")

# ── Load data ─────────────────────────────────────────────────────────────────

def load_breakouts() -> pd.DataFrame:
    df = pd.read_csv(WALL_EVENTS_CSV, header=None,
        names=["ts","symbol","side","wall_price","wall_size","peak_size",
               "event","test_count","price_at_event"])
    df["ts"]             = pd.to_datetime(df["ts"], utc=True)
    df["peak_size"]      = pd.to_numeric(df["peak_size"])
    df["test_count"]     = pd.to_numeric(df["test_count"])
    df["price_at_event"] = pd.to_numeric(df["price_at_event"], errors="coerce")

    bo = df[
        (df["event"]   == "breakout") &
        (df["symbol"]  == "MES") &
        (df["peak_size"] >= WALL_MED_MIN) &
        (df["peak_size"] <  WALL_MED_MAX) &
        (df["test_count"] >= MIN_TESTS) &
        df["price_at_event"].notna()
    ].copy()

    bo["ts_et"] = bo["ts"].dt.tz_convert(ET)
    bo["hm"]    = bo["ts_et"].dt.hour * 60 + bo["ts_et"].dt.minute
    bo_rth      = bo[(bo["hm"] >= 570) & (bo["hm"] < 960)].copy()  # 9:30–16:00 ET
    # direction: ask wall broke up → long; bid wall broke down → short
    bo_rth["direction"] = bo_rth["side"].map({"ask": 1, "bid": -1})
    return bo_rth.reset_index(drop=True)


def load_bars() -> pd.DataFrame:
    conn = sqlite3.connect(BARS_DB)
    bars = pd.read_sql(
        "SELECT ts, open, high, low, close FROM bars WHERE symbol='MES' AND minutes=1 ORDER BY ts",
        conn)
    conn.close()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    return bars


# ── MAE / MFE analysis ────────────────────────────────────────────────────────

def compute_mae_mfe(breakouts: pd.DataFrame, bars: pd.DataFrame,
                    max_hold_min: int = 20) -> pd.DataFrame:
    """
    For each breakout, compute:
      mae: max adverse excursion (pts against direction) within max_hold_min bars
      mfe: max favourable excursion (pts in direction)   within max_hold_min bars
      final: net move at bar max_hold_min
    Entry = price_at_event (mid at breakout).
    """
    records = []
    for _, row in breakouts.iterrows():
        future = bars[bars["ts"] > row["ts"]].head(max_hold_min)
        if len(future) < 2:
            continue
        entry = row["price_at_event"]
        d     = row["direction"]
        highs = future["high"].values
        lows  = future["low"].values
        if d == 1:
            mae   = entry - lows.min()
            mfe   = highs.max() - entry
            final = future["close"].iloc[-1] - entry
        else:
            mae   = highs.max() - entry
            mfe   = entry - lows.min()
            final = entry - future["close"].iloc[-1]
        records.append({
            "ts":       row["ts"],
            "side":     row["side"],
            "peak_sz":  row["peak_size"],
            "tests":    row["test_count"],
            "entry":    entry,
            "mae":      mae,
            "mfe":      mfe,
            "final":    final,
            "n_bars":   len(future),
        })
    return pd.DataFrame(records)


# ── Bracket simulation ────────────────────────────────────────────────────────

def simulate_trade(row, bars, stop_pts, target_pts, hold_min):
    """
    Simulate a bracketed trade for one breakout.
    Returns (outcome, pnl_pts) where outcome is 'TARGET'|'STOPPED'|'TIME'.
    Uses bar OHLC; checks stop before target within each bar (conservative).
    """
    future = bars[bars["ts"] > row["ts"]].head(hold_min)
    if future.empty:
        return None, None
    entry = row["price_at_event"]
    d     = row["direction"]
    stop_price   = entry - d * stop_pts
    target_price = entry + d * target_pts

    for _, bar in future.iterrows():
        if d == 1:
            if bar["low"]  <= stop_price:   return "STOPPED", -stop_pts
            if bar["high"] >= target_price: return "TARGET",  +target_pts
        else:
            if bar["high"] >= stop_price:   return "STOPPED", -stop_pts
            if bar["low"]  <= target_price: return "TARGET",  +target_pts

    # Time exit at last bar close
    exit_price = future["close"].iloc[-1]
    pnl = (exit_price - entry) * d
    return "TIME", pnl


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data…")
    breakouts = load_breakouts()
    bars      = load_bars()
    print(f"  Qualifying RTH breakouts: {len(breakouts)}")
    print(f"  MES 1-min bars: {len(bars):,}  "
          f"({bars['ts'].min().date()} → {bars['ts'].max().date()})")

    # ── MAE / MFE summary ────────────────────────────────────────────────────
    mae_df = compute_mae_mfe(breakouts, bars, max_hold_min=20)
    print(f"\n{'='*60}")
    print(f"MAE / MFE distribution  (n={len(mae_df)}, 20-bar window)")
    print(f"{'='*60}")
    for pct in [25, 50, 75, 90, 95]:
        print(f"  MAE p{pct:2d}: {np.percentile(mae_df['mae'], pct):.2f} pts  "
              f"MFE p{pct:2d}: {np.percentile(mae_df['mfe'], pct):.2f} pts")
    print(f"  MAE mean: {mae_df['mae'].mean():.2f}  max: {mae_df['mae'].max():.2f}")
    print(f"  MFE mean: {mae_df['mfe'].mean():.2f}  max: {mae_df['mfe'].max():.2f}")

    # Stop survival: fraction of trades that survive a stop of size X
    print(f"\n  Stop survival (fraction not stopped within 20 bars):")
    for s in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        survived = (mae_df["mae"] < s).mean()
        print(f"    stop={s:.1f}pt: {survived:.0%} survive")

    # ── Parameter sweep ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Parameter sweep  (EV = avg pnl per trade in pts)")
    print(f"{'='*60}")

    results = []
    for hold in HOLDS:
        for stop in STOPS:
            for target in TARGETS:
                if target <= stop:
                    continue
                outcomes, pnls = [], []
                for _, row in breakouts.iterrows():
                    outcome, pnl = simulate_trade(row, bars, stop, target, hold)
                    if outcome is not None:
                        outcomes.append(outcome)
                        pnls.append(pnl)
                if not pnls:
                    continue
                n       = len(pnls)
                wr      = sum(1 for o in outcomes if o == "TARGET") / n
                ev      = np.mean(pnls)
                avg_win = np.mean([p for p in pnls if p > 0] or [0])
                avg_los = np.mean([p for p in pnls if p < 0] or [0])
                results.append({
                    "hold": hold, "stop": stop, "target": target,
                    "n": n, "wr": wr, "ev": ev,
                    "avg_win": avg_win, "avg_los": avg_los,
                    "pf": -avg_win / avg_los if avg_los < 0 else float("inf"),
                })

    r = pd.DataFrame(results)

    # Best by EV for each hold
    for hold in HOLDS:
        sub = r[r["hold"] == hold].sort_values("ev", ascending=False).head(8)
        print(f"\n── Hold {hold} min — top 8 by EV ──")
        print(f"  {'stop':>5} {'target':>7} {'n':>4} {'WR':>6} {'EV':>6} {'avg_win':>8} {'avg_los':>8} {'PF':>6}")
        for _, row in sub.iterrows():
            print(f"  {row['stop']:5.1f} {row['target']:7.1f} {row['n']:4.0f} "
                  f"{row['wr']:6.1%} {row['ev']:6.2f} {row['avg_win']:8.2f} "
                  f"{row['avg_los']:8.2f} {row['pf']:6.2f}")

    # Best overall across all holds
    best = r.sort_values("ev", ascending=False).head(15)
    print(f"\n{'='*60}")
    print("Top 15 overall by EV")
    print(f"{'='*60}")
    print(f"  {'hold':>5} {'stop':>5} {'target':>7} {'n':>4} {'WR':>6} {'EV':>6} {'avg_win':>8} {'avg_los':>8} {'PF':>6}")
    for _, row in best.iterrows():
        print(f"  {row['hold']:5.0f} {row['stop']:5.1f} {row['target']:7.1f} {row['n']:4.0f} "
              f"{row['wr']:6.1%} {row['ev']:6.2f} {row['avg_win']:8.2f} "
              f"{row['avg_los']:8.2f} {row['pf']:6.2f}")

    # ── R:R analysis at fixed stop sizes ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("EV heatmap by stop × target  (hold=10 min)")
    print(f"{'='*60}")
    hold10 = r[r["hold"] == 10]
    pivot  = hold10.pivot_table(index="stop", columns="target", values="ev")
    pd.set_option("display.float_format", "{:5.2f}".format)
    pd.set_option("display.width", 120)
    print(pivot.to_string())


if __name__ == "__main__":
    main()
