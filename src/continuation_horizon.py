"""
"Wander then resume" test across longer horizons.

continuation_wander.py capped resolution at 30 minutes (MAX_FORWARD_MIN) and
found no continuation edge for moves that took longer than a couple minutes
to resolve. This extends the resolution window to 60/120/240 minutes to see
whether moves that specifically needed more than 30 minutes to break out
(the genuine "long wanderers" under the original test, previously scored as
unresolved timeouts) skew toward continuation once given the room to do so.

RTH day is ~390 minutes, so a 240-min horizon still leaves real room even
for triggers around midday; triggers late in the day get truncated by the
session close (tracked separately via `eod_censored`, not folded into the
"reversion" bucket).

Usage:
    python src/continuation_horizon.py
"""

import numpy as np
import pandas as pd

from continuation_ml import build_daily_state, detect_moves, load_rth_bars
from continuation_wander import two_prop_ztest

HORIZONS_MIN = [30, 60, 120, 240]
THRESHOLDS_BP = [10.0, 15.0, 20.0]   # 5bp resolves almost entirely within minutes; skip


def summarize(events: pd.DataFrame, horizon: int):
    n_total = len(events)
    timeouts = events[events["outcome"] == "timeout"]
    n_timeout = len(timeouts)
    n_eod = timeouts["eod_censored"].sum() if n_timeout else 0
    resolved = events[events["outcome"] != "timeout"]
    resolved = resolved.assign(is_cont=(resolved["outcome"] == "continuation").astype(int))

    p_all = resolved["is_cont"].mean() if len(resolved) else float("nan")

    print(f"\n  horizon={horizon:>4d}min   n_events={n_total:>6d}   "
          f"unresolved={n_timeout:>5d} ({n_timeout/n_total:.1%}, "
          f"of which {n_eod/n_timeout:.0%} hit session close rather than the horizon cap)"
          if n_timeout else
          f"\n  horizon={horizon:>4d}min   n_events={n_total:>6d}   unresolved=0")
    print(f"    P(continuation) overall (resolved only): {p_all:.3f}  (n={len(resolved)})")

    # Events that specifically needed > 30min to resolve, now that they get the room to.
    long_wanderers = resolved[resolved["bars_to_resolve"] > 30]
    quick          = resolved[resolved["bars_to_resolve"] <= 30]
    if len(long_wanderers) > 0 and len(quick) > 0:
        p_lw, n_lw = long_wanderers["is_cont"].mean(), len(long_wanderers)
        p_q,  n_q  = quick["is_cont"].mean(), len(quick)
        z = two_prop_ztest(p_lw, n_lw, p_q, n_q)
        print(f"    resolved within 30min:  n={n_q:>6d}  P(cont)={p_q:.3f}")
        print(f"    took >30min to resolve: n={n_lw:>6d}  P(cont)={p_lw:.3f}   "
              f"spread={p_lw - p_q:+.3f}  z={z:.2f}")
    else:
        print(f"    (no events needed >30min to resolve within this horizon)")


def main():
    print("Loading RTH 1-min MES bars from mes_hist_1min.csv…")
    df = load_rth_bars()
    print(f"  {len(df):,} RTH bars  ({df['date'].min()} -> {df['date'].max()}, "
          f"{df['date'].nunique()} trading days)")
    df = build_daily_state(df)

    for thresh in THRESHOLDS_BP:
        print(f"\n{'='*78}")
        print(f"Move threshold = {thresh:.0f}bp")
        print(f"{'='*78}")
        for horizon in HORIZONS_MIN:
            events = detect_moves(df, thresh, max_forward_min=horizon)
            summarize(events, horizon)


if __name__ == "__main__":
    main()
