"""
Backtest: Wall Breakout with a partial-profit take, vs. the fixed 1-lot
stop/target baseline and the trailing-stop variants in backtest_wall_break_trail.py.

Wall break trades are currently size=1 (trading_bot.py place_wall_break_signal),
so "take partial profit" only means something if size is bumped to 2: one lot
scratches at a nearer partial target, the other rides the original stop/target
(optionally moved to breakeven once the partial fires). This replays the
actual 75 logged trades in logs/wall_break_trades.csv (not the loose signal
universe) at 2x size and reports EV *per contract* so it's directly comparable
to the 1-lot baseline.

Two remainder-management modes:
  KEEP_STOP — runner keeps the original hard stop after the partial fires.
  TO_BE     — runner's stop moves to breakeven once the partial fires.

Usage:
    python src/backtest_wall_break_partial.py
"""

import sqlite3

import numpy as np
import pandas as pd

WALL_BREAK_LOG = "logs/wall_break_trades.csv"
BARS_DB        = "data/bars.db"

STOP_PTS   = 4.5   # matches sigma-clamped live avg (4-5pt band)
TARGET_PTS = 12.0  # live full target
HOLD_MIN   = 25     # matches MAX_HOLD_MIN in trading_bot.py

PARTIAL_TARGETS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def load_trades():
    df = pd.read_csv(WALL_BREAK_LOG)
    df["fired_at"] = pd.to_datetime(df["fired_at"], utc=True, errors="coerce")
    df = df[df["fired_at"].notna()].copy()
    df["entry"] = df["fill_price"].fillna(df["est_entry"])
    df = df[df["entry"].notna()].copy()
    df["direction"] = df["direction"].map({"LONG": 1, "SHORT": -1})
    return df


def load_bars():
    conn = sqlite3.connect(BARS_DB)
    bars = pd.read_sql(
        "SELECT ts, high, low, close FROM bars WHERE symbol='MES' AND minutes=1 ORDER BY ts",
        conn)
    conn.close()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    return bars


def simulate_partial(row, bars, stop_pts, target_pts, hold_min,
                      partial_pts, move_to_be):
    """
    2-contract simulation. Leg A scratches at partial_pts, or stops with the
    runner if the hard stop is hit first. Leg B (the runner) keeps riding
    after the partial fires, exiting at target_pts, the (possibly moved)
    stop, or time. Bar-by-bar; stop checked before target within a bar
    (conservative, matches backtest_wall_break.py convention).
    Returns (leg_a_pnl, leg_b_pnl) in points, or (None, None) if no bars.
    """
    future = bars[bars["ts"] > row["fired_at"]].head(hold_min)
    if future.empty:
        return None, None
    entry  = row["entry"]
    d      = row["direction"]
    hard_stop    = entry - d * stop_pts
    partial_price = entry + d * partial_pts
    target_price  = entry + d * target_pts

    leg_a_open = True
    leg_a_pnl  = None
    runner_stop = hard_stop

    for _, bar in future.iterrows():
        bar_high, bar_low = bar["high"], bar["low"]

        # Leg A: exits at partial target, or stops out with the runner if
        # the hard stop is hit before the partial (checked first, conservative).
        if leg_a_open:
            hit_stop = (bar_low <= hard_stop) if d == 1 else (bar_high >= hard_stop)
            if hit_stop:
                leg_a_pnl = -stop_pts
                leg_a_open = False
            else:
                hit_partial = (bar_high >= partial_price) if d == 1 else (bar_low <= partial_price)
                if hit_partial:
                    leg_a_pnl = partial_pts
                    leg_a_open = False
                    if move_to_be:
                        runner_stop = entry

        # Runner (leg B): exits at its stop or the full target.
        hit_runner_stop = (bar_low <= runner_stop) if d == 1 else (bar_high >= runner_stop)
        if hit_runner_stop:
            leg_b_pnl = (runner_stop - entry) * d
            if leg_a_open:
                leg_a_pnl = leg_b_pnl  # both legs share the same stop until partial fires
            return leg_a_pnl, leg_b_pnl
        hit_target = (bar_high >= target_price) if d == 1 else (bar_low <= target_price)
        if hit_target:
            leg_b_pnl = target_pts
            if leg_a_open:
                leg_a_pnl = partial_pts if partial_pts <= target_pts else target_pts
            return leg_a_pnl, leg_b_pnl

    # Time exit
    exit_price = future["close"].iloc[-1]
    leg_b_pnl = (exit_price - entry) * d
    if leg_a_open:
        leg_a_pnl = leg_b_pnl
    return leg_a_pnl, leg_b_pnl


def main():
    trades = load_trades()
    bars   = load_bars()
    print(f"Logged wall-break trades: {len(trades)}")
    actual_total = trades["pnl_pts"].sum()
    print(f"Actual live total (1 lot): {actual_total:.1f}pts over {len(trades)} trades "
          f"({actual_total/len(trades):.2f}pts/trade)")

    print(f"\n{'='*78}")
    print(f"PARTIAL sweep  (stop={STOP_PTS}pt  full_target={TARGET_PTS:.0f}pt  hold={HOLD_MIN}min)")
    print(f"EV reported per contract (2-lot trade pnl / 2) for apples-to-apples vs 1-lot baseline")
    print(f"{'='*78}")
    print(f"  {'partial':>7} {'move_to_be':>10}   {'n':>4} {'total_2lot':>11} {'ev/contract':>12} {'vs_actual':>10}")

    results = []
    for move_to_be in [False, True]:
        for partial in PARTIAL_TARGETS:
            leg_a_sum, leg_b_sum, n = 0.0, 0.0, 0
            for _, row in trades.iterrows():
                a, b = simulate_partial(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN,
                                        partial, move_to_be)
                if a is None:
                    continue
                leg_a_sum += a
                leg_b_sum += b
                n += 1
            total_2lot   = leg_a_sum + leg_b_sum
            ev_per_contract = total_2lot / (2 * n) if n else float("nan")
            results.append({
                "move_to_be": move_to_be, "partial": partial, "n": n,
                "total_2lot": total_2lot, "ev_per_contract": ev_per_contract,
            })
            print(f"  {partial:7.1f} {str(move_to_be):>10}   {n:4d} {total_2lot:11.1f} "
                  f"{ev_per_contract:12.3f} {ev_per_contract - actual_total/len(trades):10.3f}")

    r = pd.DataFrame(results)
    print(f"\n{'='*78}")
    print("Top by EV per contract")
    print(f"{'='*78}")
    top = r.sort_values("ev_per_contract", ascending=False).head(8)
    for _, row in top.iterrows():
        print(f"  move_to_be={row['move_to_be']}  partial={row['partial']:.1f}pt  "
              f"n={row['n']:.0f}  ev/contract={row['ev_per_contract']:.3f}pts  "
              f"(baseline actual/contract={actual_total/len(trades):.3f}pts)")


if __name__ == "__main__":
    main()
