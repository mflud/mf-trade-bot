"""
Backtest: Wall Breakout with a trailing stop, vs. the fixed stop/target baseline.

Reuses the same data (logs/wall_events.csv qualifying breakouts, 1-min MES bars
from data/bars.db) and event filters as backtest_wall_break.py. Simulates three
exit schemes per trade:

  FIXED  — hard stop_pts / target_pts bracket (current live behaviour).
  TRAIL  — once price has moved `activation_pts` in favour, a stop trails
           `trail_pts` behind the highest-high (long) / lowest-low (short)
           seen since entry, for the rest of the hold. No target — trade
           only exits on trail-stop or time.
  BE_CAP — same trailing logic, but the stop is not allowed to advance past
           breakeven (entry price). Once it reaches breakeven it holds there;
           it does not continue trailing toward the target. Trade still exits
           early via TARGET if price gets there.

Usage:
    python src/backtest_wall_break_trail.py
"""

import sqlite3

import numpy as np
import pandas as pd
import pytz

from backtest_wall_break import load_breakouts, load_bars

# ── Config ────────────────────────────────────────────────────────────────────

# Current live wall-break params (trading_bot.py): stop 4-5pt, target 12pt.
BASE_STOP_MIN = 4.0
BASE_STOP_MAX = 5.0
BASE_TARGET   = 12.0
HOLD_MIN      = 20   # matches MAX_HOLD_MIN scale used for wall_break in trading_bot.py

ACTIVATIONS = [2.0, 3.0, 4.0, 5.0]   # profit pts before trailing kicks in
TRAILS      = [2.0, 3.0, 4.0, 5.0]   # trail distance behind peak, in pts

ET = pytz.timezone("America/New_York")


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_fixed(row, bars, stop_pts, target_pts, hold_min):
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
    exit_price = future["close"].iloc[-1]
    return "TIME", (exit_price - entry) * d


def simulate_trail(row, bars, stop_pts, target_pts, hold_min,
                    activation_pts, trail_pts, cap_at_breakeven):
    """
    Bar-by-bar simulation. Initial hard stop = stop_pts (matches the live
    bracket's safety net). Once favourable excursion since entry >= activation_pts,
    a trailing stop is armed and ratchets with the running peak. If
    cap_at_breakeven, the trail never advances past entry price. Target still
    exits the trade if hit before the trail does. Conservative ordering within
    a bar: stop/trail checked before target.
    """
    future = bars[bars["ts"] > row["ts"]].head(hold_min)
    if future.empty:
        return None, None
    entry  = row["price_at_event"]
    d      = row["direction"]
    target_price = entry + d * target_pts
    hard_stop    = entry - d * stop_pts

    trail_armed = False
    peak        = entry   # highest favourable price (long) / lowest (short)
    trail_stop  = None

    for _, bar in future.iterrows():
        bar_high, bar_low = bar["high"], bar["low"]

        # Update peak using this bar's favourable extreme.
        fav_extreme = bar_high if d == 1 else bar_low
        if d == 1:
            peak = max(peak, fav_extreme)
        else:
            peak = min(peak, fav_extreme)

        favourable_excursion = (peak - entry) * d
        if not trail_armed and favourable_excursion >= activation_pts:
            trail_armed = True

        if trail_armed:
            raw_trail = peak - d * trail_pts
            if cap_at_breakeven:
                # Stop never advances past entry — clamp toward entry.
                if d == 1:
                    raw_trail = min(raw_trail, entry)
                else:
                    raw_trail = max(raw_trail, entry)
            trail_stop = raw_trail if trail_stop is None else (
                max(trail_stop, raw_trail) if d == 1 else min(trail_stop, raw_trail)
            )

        active_stop = trail_stop if trail_armed else hard_stop

        if d == 1:
            if bar_low <= active_stop:
                pnl = active_stop - entry
                return ("TRAIL" if trail_armed else "STOPPED"), pnl
            if bar_high >= target_price:
                return "TARGET", target_pts
        else:
            if bar_high >= active_stop:
                pnl = entry - active_stop
                return ("TRAIL" if trail_armed else "STOPPED"), pnl
            if bar_low <= target_price:
                return "TARGET", target_pts

    exit_price = future["close"].iloc[-1]
    return "TIME", (exit_price - entry) * d


# ── Main ──────────────────────────────────────────────────────────────────────

def summarize(label, outcomes, pnls):
    n       = len(pnls)
    if n == 0:
        print(f"  {label}: no trades")
        return
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    wr      = len(wins) / n
    ev      = np.mean(pnls)
    avg_win = np.mean(wins) if wins else 0.0
    avg_los = np.mean(losses) if losses else 0.0
    pf      = -avg_win * len(wins) / (avg_los * len(losses)) if losses and avg_los < 0 else float("inf")
    print(f"  {label:<28} n={n:4d}  WR={wr:6.1%}  EV={ev:6.2f}pts  "
          f"avg_win={avg_win:6.2f}  avg_los={avg_los:6.2f}  total={sum(pnls):8.1f}pts")


def main():
    print("Loading data…")
    breakouts = load_breakouts()
    bars      = load_bars()
    print(f"  Qualifying RTH breakouts: {len(breakouts)}")
    print(f"  MES 1-min bars: {len(bars):,}  "
          f"({bars['ts'].min().date()} → {bars['ts'].max().date()})")

    stop_pts = (BASE_STOP_MIN + BASE_STOP_MAX) / 2  # 4.5, matches sigma-clamped live avg

    # ── Baseline: current fixed stop/target ──────────────────────────────────
    print(f"\n{'='*72}")
    print(f"BASELINE  stop={stop_pts:.1f}pt  target={BASE_TARGET:.0f}pt  hold={HOLD_MIN}min")
    print(f"{'='*72}")
    outcomes, pnls = [], []
    for _, row in breakouts.iterrows():
        o, p = simulate_fixed(row, bars, stop_pts, BASE_TARGET, HOLD_MIN)
        if o is not None:
            outcomes.append(o); pnls.append(p)
    summarize("FIXED (baseline)", outcomes, pnls)

    # ── Trailing stop sweep ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"TRAIL sweep  (target={BASE_TARGET:.0f}pt still caps upside, hold={HOLD_MIN}min)")
    print(f"{'='*72}")
    print(f"  {'activation':>10} {'trail':>6}   {'variant':<28} {'n':>5} {'WR':>7} {'EV':>7} {'total':>8}")

    results = []
    for cap in [False, True]:
        variant_name = "BE_CAP (stop at breakeven)" if cap else "TRAIL (full trail to target)"
        for act in ACTIVATIONS:
            for trail in TRAILS:
                outcomes, pnls = [], []
                for _, row in breakouts.iterrows():
                    o, p = simulate_trail(row, bars, stop_pts, BASE_TARGET, HOLD_MIN,
                                          activation_pts=act, trail_pts=trail,
                                          cap_at_breakeven=cap)
                    if o is not None:
                        outcomes.append(o); pnls.append(p)
                if not pnls:
                    continue
                n  = len(pnls)
                wr = sum(1 for p in pnls if p > 0) / n   # win = pnl>0, not just TARGET hits
                ev = np.mean(pnls)
                results.append({
                    "cap": cap, "variant": variant_name,
                    "activation": act, "trail": trail,
                    "n": n, "wr": wr, "ev": ev, "total": sum(pnls),
                })
                print(f"  {act:10.1f} {trail:6.1f}   {variant_name:<28} {n:5d} "
                      f"{wr:7.1%} {ev:7.2f} {sum(pnls):8.1f}")

    r = pd.DataFrame(results)

    print(f"\n{'='*72}")
    print("Top 10 by EV across both variants")
    print(f"{'='*72}")
    top = r.sort_values("ev", ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"  {row['variant']:<28} activation={row['activation']:.1f}  "
              f"trail={row['trail']:.1f}  n={row['n']:.0f}  WR={row['wr']:.1%}  "
              f"EV={row['ev']:.2f}pts  total={row['total']:.1f}pts")

    print(f"\n{'='*72}")
    print("Best FIXED-baseline params for comparison (from backtest_wall_break.py grid)")
    print(f"{'='*72}")
    print(f"  Re-run src/backtest_wall_break.py for the full stop/target/hold grid.")


if __name__ == "__main__":
    main()
