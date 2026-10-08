"""
analyze_rev_gap_vol.py — Same gap-volatility question as
analyze_cascade_gap_vol.py, but for BA-REV: does the price range between
test #2 and test #3 of the SAME wall (the gap right before the rejection
signal fires) predict whether the trade continues?

Reuses backtest_ba_rev_resweep.py's load_bars/load_tests/compute_containment/
simulate_one/simulate_realistic/summarize (already validated against live
BA_REV_* config) — only adds matching test#1->test#2->test#3 timestamps for
the SAME wall (grouped by side+wall_price) and pulling 5s-bar range between
them from mes_hist_5sec.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from backtest_ba_rev_resweep import (
    load_bars, load_tests, compute_containment, simulate_one, simulate_realistic, summarize,
    REF_START_HM, LOCK_HM, TRADE_END_HM,
)
from analyze_cascade_gap_vol import load_5sec_bars, gap_range

LIVE_STOP, LIVE_TARGET, LIVE_HOLD = 2.0, 6.0, 60
LIVE_N = 3


def detect_rev_test_sequences(tests: pd.DataFrame, side: str, containment: dict,
                               test_count: int = LIVE_N) -> pd.DataFrame:
    """For each test==test_count event on `side`, find the SAME wall's
    (matched by exact wall_price within that side) test_count-1 and
    test_count-2 event timestamps immediately preceding it, subject to the
    live containment/window gate (mirrors build_signals in
    backtest_ba_rev_resweep.py)."""
    sub = tests[tests["side"] == side].sort_values("ts")
    signals = []

    for wall_price, grp in sub.groupby("wall_price"):
        grp = grp.sort_values("ts").reset_index(drop=True)
        for _, row in grp.iterrows():
            if row["test_count"] != test_count:
                continue
            if not (LOCK_HM <= row["hm"] < TRADE_END_HM):
                continue
            c = containment.get(row["date"])
            if c is None:
                continue
            _, _, first_breach_ts = c
            if first_breach_ts is not None and row["ts"] >= first_breach_ts:
                continue

            prior = grp[grp["ts"] < row["ts"]]
            tc2 = prior[prior["test_count"] == test_count - 1]
            tc1 = prior[prior["test_count"] == test_count - 2]
            if tc2.empty or tc1.empty:
                continue

            direction = -1 if side == "ask" else 1
            signals.append({
                "ts": row["ts"], "date": row["date"], "side": side, "direction": direction,
                "entry": row["price_at_event"], "wall_price": wall_price,
                "t1_ts": tc1["ts"].iloc[-1], "t2_ts": tc2["ts"].iloc[-1], "t3_ts": row["ts"],
            })

    return pd.DataFrame(signals).sort_values("ts").reset_index(drop=True)


def main():
    print("Loading data…")
    bars = load_bars("MES")
    tests = load_tests("MES")
    bars5s = load_5sec_bars()
    containment = compute_containment(bars)

    all_sigs = []
    for side in ["ask", "bid"]:
        s = detect_rev_test_sequences(tests, side, containment, test_count=LIVE_N)
        print(f"{side}: {len(s)} contained N={LIVE_N} test sequences")
        all_sigs.append(s)
    sigs = pd.concat(all_sigs, ignore_index=True)

    sigs["gap_2_3"] = sigs.apply(lambda r: gap_range(bars5s, r["t2_ts"], r["t3_ts"]), axis=1)
    sigs["gap_1_2"] = sigs.apply(lambda r: gap_range(bars5s, r["t1_ts"], r["t2_ts"]), axis=1)
    before = len(sigs)
    sigs = sigs.dropna(subset=["gap_2_3", "gap_1_2"]).reset_index(drop=True)
    print(f"\nDropped {before - len(sigs)} with no 5s coverage; {len(sigs)} remain\n")

    bars_ts = bars["ts"].values
    bars_high = bars["high"].values
    bars_low = bars["low"].values
    bars_close = bars["close"].values

    outcomes, pnls = [], []
    for _, row in sigs.iterrows():
        s_ts = np.datetime64(row["ts"].tz_convert("UTC").tz_localize(None))
        r = simulate_one(s_ts, row["entry"], row["direction"], bars_ts, bars_high, bars_low, bars_close,
                          LIVE_STOP, LIVE_TARGET, LIVE_HOLD)
        outcomes.append(r["outcome"] if r else None)
        pnls.append(r["pnl"] if r else None)
    sigs["outcome"] = outcomes
    sigs["pnl"] = pnls
    sigs = sigs.dropna(subset=["pnl"]).reset_index(drop=True)
    print(f"Simulated trades (raw, both sides combined): {len(sigs)}\n")

    for side in ["ask", "bid", "both"]:
        s = sigs if side == "both" else sigs[sigs["side"] == side]
        if len(s) < 30:
            print(f"{side}: too few ({len(s)})\n")
            continue
        print(f"{'='*80}\n{side.upper()} — bucketed by gap_2_3  (n={len(s)})\n{'='*80}")
        terciles = s["gap_2_3"].quantile([0, 1/3, 2/3, 1.0]).values
        labels = ["LOW", "MID", "HIGH"]
        for i in range(3):
            lo, hi = terciles[i], terciles[i + 1]
            mask = (s["gap_2_3"] >= lo) & (s["gap_2_3"] <= hi) if i == 2 else \
                   (s["gap_2_3"] >= lo) & (s["gap_2_3"] < hi)
            sub = s[mask]
            if len(sub) == 0:
                continue
            print(f"  {labels[i]:>5} [{lo:5.2f},{hi:5.2f}]  n={len(sub):>3}  "
                  f"WR={(sub['pnl']>0).mean():>6.1%}  EV={sub['pnl'].mean():>6.2f}  "
                  f"total={sub['pnl'].sum():>7.1f}  TARGET%={(sub['outcome']=='TARGET').mean():>6.1%}  "
                  f"STOPPED%={(sub['outcome']=='STOPPED').mean():>6.1%}")
        print(f"  corr(gap_2_3, pnl) = {np.corrcoef(s['gap_2_3'], s['pnl'])[0,1]:.4f}\n")

    return sigs


def backtest_filter(sigs: pd.DataFrame, bars: pd.DataFrame, side: str, thresholds):
    s = sigs if side == "both" else sigs[sigs["side"] == side]
    s = s.sort_values("ts").reset_index(drop=True)
    bars_ts = bars["ts"].values
    bars_high = bars["high"].values
    bars_low = bars["low"].values
    bars_close = bars["close"].values

    print(f"\n{side.upper()} entry-filter backtest ({len(s)} candidates)")
    print(f"{'gap<=':>8} | {'n':>5} {'WR':>7} {'EV':>7} {'total':>8} {'top5%':>7}")
    for thr in thresholds:
        busy_until = None
        outcomes, pnls = [], []
        for _, row in s.iterrows():
            s_ts = np.datetime64(row["ts"].tz_convert("UTC").tz_localize(None))
            if busy_until is not None and s_ts < busy_until:
                continue
            if row["gap_2_3"] > thr:
                continue
            r = simulate_one(s_ts, row["entry"], row["direction"], bars_ts, bars_high, bars_low, bars_close,
                              LIVE_STOP, LIVE_TARGET, LIVE_HOLD)
            if r is None:
                continue
            outcomes.append(r["outcome"]); pnls.append(r["pnl"])
            busy_until = np.datetime64(r["exit_ts"])
        n = len(pnls)
        label = f"{thr:.2f}" if thr < 999 else "none"
        if n == 0:
            print(f"{label:>8} | no trades")
            continue
        pnls_arr = np.array(pnls)
        total = pnls_arr.sum()
        top5 = np.sort(pnls_arr)[-5:].sum() if n >= 5 else pnls_arr[pnls_arr > 0].sum()
        print(f"{label:>8} | {n:>5} {(pnls_arr>0).mean():>7.1%} {pnls_arr.mean():>7.2f} "
              f"{total:>8.1f} {(top5/total if total else float('nan')):>6.0%}")


if __name__ == "__main__":
    sigs = main()
    bars = load_bars("MES")
    for side in ["ask", "bid", "both"]:
        backtest_filter(sigs, bars, side, thresholds=[1.5, 2.0, 2.5, 3.0, 999.0])
