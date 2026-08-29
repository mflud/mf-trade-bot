"""
"Wander then resume" test for MES bp moves.

continuation_ml.py's move-detection already captures this for free: an event
is only marked "continuation" or "reversion" once price first closes/touches
one of two targets (trigger_price +/- move_threshold_bp) — until then it is,
by construction, ranging inside that band. So `bars_to_resolve` (minutes from
trigger to whichever target got hit) IS a direct measure of how long the
move "wandered" in a tight band before resolving.

This asks: does a longer wander phase before resolution shift the eventual
probability toward continuation (the user's hypothesis) vs reversion,
compared to moves that resolve quickly (little or no wandering)?

Usage:
    python src/continuation_wander.py
"""

import numpy as np
import pandas as pd

from continuation_ml import (MOVE_THRESHOLDS_BP, build_daily_state,
                              detect_moves, load_rth_bars)

WANDER_CUTOFFS_MIN = [2, 5, 10, 15]


def two_prop_ztest(p1, n1, p2, n2):
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan")
    return (p1 - p2) / se


def decile_curve(events: pd.DataFrame):
    resolved = events[events["outcome"] != "timeout"].copy()
    resolved["is_cont"] = (resolved["outcome"] == "continuation").astype(int)
    resolved["decile"] = pd.qcut(resolved["bars_to_resolve"], 10,
                                  duplicates="drop", labels=False)
    rows = []
    for d, grp in resolved.groupby("decile"):
        rows.append({
            "decile": d,
            "bars_to_resolve_range": f"{grp['bars_to_resolve'].min():.0f}-{grp['bars_to_resolve'].max():.0f}",
            "n": len(grp),
            "p_cont": grp["is_cont"].mean(),
        })
    return pd.DataFrame(rows)


def quick_vs_wander(events: pd.DataFrame, cutoff_min: int):
    resolved = events[events["outcome"] != "timeout"].copy()
    resolved["is_cont"] = (resolved["outcome"] == "continuation").astype(int)
    quick    = resolved[resolved["bars_to_resolve"] <= cutoff_min]
    wandered = resolved[resolved["bars_to_resolve"] >  cutoff_min]
    n_q, n_w = len(quick), len(wandered)
    if n_q == 0 or n_w == 0:
        return None
    p_q, p_w = quick["is_cont"].mean(), wandered["is_cont"].mean()
    z = two_prop_ztest(p_w, n_w, p_q, n_q)
    return {
        "cutoff_min": cutoff_min,
        "n_quick": n_q, "p_cont_quick": p_q,
        "n_wandered": n_w, "p_cont_wandered": p_w,
        "spread": p_w - p_q, "z": z,
    }


def main():
    print("Loading RTH 1-min MES bars from mes_hist_1min.csv…")
    df = load_rth_bars()
    print(f"  {len(df):,} RTH bars  ({df['date'].min()} -> {df['date'].max()}, "
          f"{df['date'].nunique()} trading days)")
    df = build_daily_state(df)

    for thresh in MOVE_THRESHOLDS_BP:
        events = detect_moves(df, thresh)
        n_total   = len(events)
        n_timeout = (events["outcome"] == "timeout").sum()
        resolved  = events[events["outcome"] != "timeout"]

        print(f"\n{'='*78}")
        print(f"Move threshold = {thresh:.0f}bp  |  n_events={n_total}  "
              f"|  never resolved within 30min: {n_timeout} ({n_timeout/n_total:.1%})")
        print(f"{'='*78}")

        print(f"\n  P(continuation) by wander-duration decile "
              f"(bars_to_resolve = minutes spent ranging before breaking out):")
        curve = decile_curve(events)
        print(f"  {'decile':>6} {'bars-to-resolve':>16} {'n':>7} {'P(cont)':>9}")
        for _, row in curve.iterrows():
            print(f"  {row['decile']:>6.0f} {row['bars_to_resolve_range']:>16} "
                  f"{row['n']:>7.0f} {row['p_cont']:>9.3f}")

        print(f"\n  Quick-resolve vs wandered-then-resolved, at several cutoffs:")
        print(f"  {'cutoff':>7} {'n_quick':>8} {'P(cont|quick)':>14} "
              f"{'n_wander':>9} {'P(cont|wander)':>15} {'spread':>8} {'z':>6}")
        for cutoff in WANDER_CUTOFFS_MIN:
            r = quick_vs_wander(events, cutoff)
            if r is None:
                continue
            print(f"  {r['cutoff_min']:>6.0f}m {r['n_quick']:>8d} {r['p_cont_quick']:>14.3f} "
                  f"{r['n_wandered']:>9d} {r['p_cont_wandered']:>15.3f} "
                  f"{r['spread']:>+8.3f} {r['z']:>6.2f}")


if __name__ == "__main__":
    main()
