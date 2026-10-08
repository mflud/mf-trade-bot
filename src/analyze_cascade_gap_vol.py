"""
analyze_cascade_gap_vol.py — Does the price range/vol BETWEEN breakouts #2
and #3 of a BA-BRK cascade predict whether the triggering trade continues?

User's hypothesis: a relatively LARGE range in the gap right before the 3rd
(triggering) breakout means the push is choppier/less clean, so the trade
is LESS likely to hit target (more likely to fail) than a cascade whose
pre-trigger gap was tight/orderly.

Reuses backtest_wall_cascade.py's load_breakouts/compute_day_opens/
simulate_realistic/summarize (already validated against the live BA_BRK_*
config) — only adds: (a) tracking each cascade's actual breakout timestamps
(not just the trigger) so the #2->#3 gap is known, (b) pulling 5s bars from
mes_hist_5sec.csv to measure the real price range within that gap.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_wall_cascade import (
    load_breakouts, load_bars, compute_day_opens, simulate_realistic, summarize,
    STOP_PTS, TARGET_PTS, HOLD_MIN, BA_BRK_LIVE_MAX_GAP,
)

MES_5SEC_CSV = "mes_hist_5sec.csv"


def detect_cascades_with_gaps(breakouts: pd.DataFrame, side: str, cascade_min: int,
                               max_gap_sec: int, day_opens: dict) -> pd.DataFrame:
    """Same cascade-detection semantics as backtest_wall_cascade.detect_cascades
    (one signal per cascade, fires the moment cascade_min is first reached),
    but also records the full list of breakout timestamps in that cascade so
    the pre-trigger gap(s) are available."""
    direction = 1 if side == "ask" else -1
    events = breakouts[breakouts["side"] == side].sort_values("ts")

    signals = []
    cascade_ts: list = []
    last_ts = None
    signaled_this_cascade = False

    for _, e in events.iterrows():
        if last_ts is not None and (e["ts"] - last_ts).total_seconds() <= max_gap_sec:
            cascade_ts.append(e["ts"])
        else:
            cascade_ts = [e["ts"]]
            signaled_this_cascade = False
        last_ts = e["ts"]

        if len(cascade_ts) >= cascade_min and not signaled_this_cascade:
            day_open = day_opens.get(e["ts_et"].date())
            if day_open is None:
                continue
            if side == "ask" and e["price_at_event"] <= day_open:
                continue
            if side == "bid" and e["price_at_event"] >= day_open:
                continue
            # last cascade_min breakout timestamps, in order (the ones that formed THIS trigger)
            trigger_breakouts = cascade_ts[-cascade_min:]
            signals.append({
                "ts": e["ts"], "side": side, "direction": direction,
                "entry": e["price_at_event"], "cascade_len": len(cascade_ts),
                "breakout_ts_list": trigger_breakouts,
            })
            signaled_this_cascade = True

    return pd.DataFrame(signals)


def load_5sec_bars() -> pd.DataFrame:
    df = pd.read_csv(MES_5SEC_CSV)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="mixed")
    return df.sort_values("ts").reset_index(drop=True)


def gap_range(bars5s: pd.DataFrame, ts_start: pd.Timestamp, ts_end: pd.Timestamp) -> float:
    """High-low range of all 5s bars strictly between ts_start and ts_end
    (exclusive of ts_start, inclusive of ts_end — the gap AFTER breakout #k
    up to and including breakout #k+1)."""
    window = bars5s[(bars5s["ts"] > ts_start) & (bars5s["ts"] <= ts_end)]
    if window.empty:
        return np.nan
    return window["high"].max() - window["low"].min()


def main():
    print("Loading data…")
    breakouts = load_breakouts("MES", window="settled")
    bars = load_bars("MES")
    bars5s = load_5sec_bars()
    day_opens = compute_day_opens(bars)

    sigs = detect_cascades_with_gaps(breakouts, "ask", cascade_min=3,
                                      max_gap_sec=BA_BRK_LIVE_MAX_GAP, day_opens=day_opens)
    print(f"Ask-side N=3 cascades (aligned, settled window): {len(sigs)}")

    # Restrict to the live 9:40-13:00 ET trading window to match what's actually tradeable
    sigs = sigs.copy()
    sigs["ts_et"] = sigs["ts"].dt.tz_convert("America/New_York")
    hm = sigs["ts_et"].dt.hour * 60 + sigs["ts_et"].dt.minute
    sigs = sigs[(hm >= 9 * 60 + 40) & (hm < 13 * 60)].reset_index(drop=True)
    print(f"  within live 9:40-13:00 ET window: {len(sigs)}\n")

    sigs["gap_2_3"] = sigs["breakout_ts_list"].apply(
        lambda ts_list: gap_range(bars5s, ts_list[-2], ts_list[-1]))
    sigs["gap_1_2"] = sigs["breakout_ts_list"].apply(
        lambda ts_list: gap_range(bars5s, ts_list[-3], ts_list[-2]))
    sigs["gap_total"] = sigs["gap_1_2"] + sigs["gap_2_3"]

    before = len(sigs)
    sigs = sigs.dropna(subset=["gap_2_3", "gap_1_2"]).reset_index(drop=True)
    print(f"Dropped {before - len(sigs)} signals with no 5s-bar coverage in the gap "
          f"(outside mes_hist_5sec.csv's range)\n")

    from backtest_wall_cascade import simulate_trade
    outcomes, pnls, exit_tss = [], [], []
    for _, row in sigs.iterrows():
        o, p, exit_ts = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
        outcomes.append(o)
        pnls.append(p)
        exit_tss.append(exit_ts)
    sigs["outcome"] = outcomes
    sigs["pnl"] = pnls
    sigs["exit_ts"] = exit_tss
    sigs = sigs.dropna(subset=["pnl"]).reset_index(drop=True)

    # One-position-at-a-time filter: drop any signal firing while the previous
    # trade's hold window is still open — avoids overlapping signals (which may
    # cluster differently across gap-size buckets, e.g. choppy/high-range periods
    # could produce more overlapping cascades) confounding the gap/outcome read.
    sigs = sigs.sort_values("ts").reset_index(drop=True)
    keep_idx = []
    busy_until = None
    for idx, row in sigs.iterrows():
        if busy_until is not None and row["ts"] <= busy_until:
            continue
        keep_idx.append(idx)
        busy_until = row["exit_ts"]
    print(f"Total simulated trades (raw, before overlap filter): {len(sigs)}")
    sigs = sigs.loc[keep_idx].reset_index(drop=True)
    print(f"Total trades (one-position-at-a-time, non-overlapping): {len(sigs)}\n")

    for gap_col in ["gap_2_3", "gap_1_2", "gap_total"]:
        print(f"{'='*80}\nBucketed by {gap_col}\n{'='*80}")
        terciles = sigs[gap_col].quantile([0, 1/3, 2/3, 1.0]).values
        labels = ["LOW (tight)", "MID", "HIGH (wide)"]
        for i in range(3):
            lo, hi = terciles[i], terciles[i + 1]
            mask = (sigs[gap_col] >= lo) & (sigs[gap_col] <= hi) if i == 2 else \
                   (sigs[gap_col] >= lo) & (sigs[gap_col] < hi)
            sub = sigs[mask]
            n = len(sub)
            if n == 0:
                continue
            wr = (sub["pnl"] > 0).mean()
            ev = sub["pnl"].mean()
            total = sub["pnl"].sum()
            target_rate = (sub["outcome"] == "TARGET").mean()
            stopped_rate = (sub["outcome"] == "STOPPED").mean()
            print(f"  {labels[i]:>12} (range in [{lo:6.2f}, {hi:6.2f}]pt)  n={n:>4}  "
                  f"WR={wr:>6.1%}  EV={ev:>6.2f}pt  total={total:>7.1f}pt  "
                  f"TARGET%={target_rate:>6.1%}  STOPPED%={stopped_rate:>6.1%}")
        corr = np.corrcoef(sigs[gap_col], sigs["pnl"])[0, 1]
        print(f"  correlation(gap, pnl) = {corr:.4f}")
        print()

    return sigs


def backtest_gap_filter(thresholds=(1.5, 2.0, 2.25, 2.5, 3.0, 999.0)):
    """Actual entry-filter backtest: skip any cascade whose gap_2_3 exceeds
    `threshold` entirely (don't open a position for it), then run the FULL
    one-position-at-a-time simulation over what's left. This differs from
    the bucketed sub-group stats in main() — skipping a bad signal can free
    up a slot for the next signal to fire sooner than it otherwise would
    have, so the aggregate effect has to be measured on the full sequence,
    not approximated from independent bucket averages. threshold=999
    (effectively no filter) is the baseline for comparison.

    Note: the 2.25pt threshold itself came from the full-sample tercile
    split in main() — this is a fixed constant (same style as this file's
    other wall/breakout thresholds, e.g. BREAK_PTS), not a lookahead-free
    walk-forward calibration. Treat this as confirming the direction/size
    of the effect on realistic aggregate P&L, not a final tuned parameter.
    """
    print("Loading data…")
    breakouts = load_breakouts("MES", window="settled")
    bars = load_bars("MES")
    bars5s = load_5sec_bars()
    day_opens = compute_day_opens(bars)

    sigs = detect_cascades_with_gaps(breakouts, "ask", cascade_min=3,
                                      max_gap_sec=BA_BRK_LIVE_MAX_GAP, day_opens=day_opens)
    sigs = sigs.copy()
    sigs["ts_et"] = sigs["ts"].dt.tz_convert("America/New_York")
    hm = sigs["ts_et"].dt.hour * 60 + sigs["ts_et"].dt.minute
    sigs = sigs[(hm >= 9 * 60 + 40) & (hm < 13 * 60)].reset_index(drop=True)

    sigs["gap_2_3"] = sigs["breakout_ts_list"].apply(
        lambda ts_list: gap_range(bars5s, ts_list[-2], ts_list[-1]))
    sigs = sigs.dropna(subset=["gap_2_3"]).sort_values("ts").reset_index(drop=True)

    from backtest_wall_cascade import simulate_trade
    print(f"\n{len(sigs)} candidate signals (live window, 5s coverage available)\n")
    print(f"{'gap<=':>8} | {'n':>5} {'WR':>7} {'EV':>7} {'total':>8} {'top5%':>7}")

    for thr in thresholds:
        outcomes, pnls = [], []
        busy_until = None
        for _, row in sigs.iterrows():
            if busy_until is not None and row["ts"] <= busy_until:
                continue
            if row["gap_2_3"] > thr:
                continue   # filtered out — do NOT consume a slot
            o, p, exit_ts = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
            if o is None:
                continue
            outcomes.append(o)
            pnls.append(p)
            busy_until = exit_ts
        s = summarize("gap_filter", outcomes, pnls)
        if s:
            label = f"{thr:.2f}" if thr < 999 else "none"
            print(f"{label:>8} | {s['n']:>5} {s['wr']:>7.1%} {s['ev']:>7.2f} "
                  f"{s['total']:>8.1f} {s['top5_share']:>6.0%}")


if __name__ == "__main__":
    main()
    print("\n\n" + "#" * 80 + "\n# Entry-filter backtest (full one-at-a-time simulation)\n" + "#" * 80 + "\n")
    backtest_gap_filter()
