"""
Rolling-percentile dynamic CSR window backtest.

Instead of a fixed GK vol threshold (e.g. 8%), the current bar's GK vol is
ranked against the trailing N-bar distribution.  When it falls in the bottom
quartile (< 25th percentile) the 4-bar (20-min) CSR window is used; otherwise
the 8-bar (40-min) baseline is used.

Tests several percentile thresholds (10th, 25th, 33rd, 40th) and rolling
window lengths (100, 250, 500, 1000 bars) to assess stability.

Also tests the 4-tier interpolated staircase:
  p < 25  → 4 bars
  p 25–50 → 5 bars
  p 50–75 → 6 bars
  p > 75  → 8 bars

Compares all variants against:
  - Fixed 8-bar baseline
  - Fixed absolute-threshold rule (< 8% → 4 bars)

Usage:
  python src/backtest_csr_percentile.py
  python src/backtest_csr_percentile.py --sym MES
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TF             = 5
TRAILING_BARS  = 20
MAX_BARS_HOLD  = 3
MIN_SCALED     = 3.0
MAX_SCALED     = 99.0
MIN_VOL_RATIO  = 1.5
CSR_THRESHOLD  = 1.5
GK_VOL_BARS    = 20
BLACKOUT_ET    = [(8, 0, 9, 0)]

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0

MAX_CSR_WINDOW = 8   # largest window we'll ever use

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

# Rolling window lengths to test (bars of 5-min data)
ROLL_WINDOWS = [100, 250, 500, 1000]

# Percentile thresholds for the 2-tier rule (bottom p% → 4 bars, rest → 8 bars)
PCT_THRESHOLDS = [10, 25, 33, 40]


# ── Data helpers ─────────────────────────────────────────────────────────────────

def load_1min(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    return df.sort_values("ts").reset_index(drop=True)


def make_5min_bars(df1):
    df1 = df1.copy()
    df1["gap"] = df1["ts"].diff() > pd.Timedelta(minutes=2)
    records = []
    i, n = 0, len(df1)
    while i < n and df1["ts"].iloc[i].minute % TF != 0:
        i += 1
    while i + TF <= n:
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            while i < n and df1["ts"].iloc[i].minute % TF != 0:
                i += 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[-1],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += TF
    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


def gk_val(o, h, l, c):
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return None
    return 0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2


# ── Scan: record gk_vol and prior returns for every trigger ──────────────────────

def scan(bars):
    bars_per_year = 252 * 23 * 60 / TF
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    opens   = bars["open"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)

    # Pre-compute per-bar GK variance (raw, not annualised)
    gk_var = np.array([
        gk_val(opens[j], highs[j], lows[j], closes[j]) or 0.0
        for j in range(n)
    ])

    records = []
    lookback = max(TRAILING_BARS, MAX_CSR_WINDOW) + 1

    for i in range(lookback, n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                          / closes[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or abs(scaled) > MAX_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        bar_et = ts_pd[i].astimezone(ET)
        if any((sh, sm) <= (bar_et.hour, bar_et.minute) < (eh, em)
               for sh, sm, eh, em in BLACKOUT_ET):
            continue

        direction = 1 if scaled > 0 else -1

        # GK annualised vol (20-bar window)
        gk_sample = gk_var[max(0, i - GK_VOL_BARS): i]
        gk_ann = math.sqrt(max(0.0, float(gk_sample.mean())) * bars_per_year) if len(gk_sample) else 0.0

        # Prior returns for CSR at any window up to MAX_CSR_WINDOW
        prior_rets = np.log(closes[i - MAX_CSR_WINDOW: i]
                          / closes[i - MAX_CSR_WINDOW - 1: i - 1])

        # Outcomes
        entry       = closes[i]
        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}
        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if direction == 1 and h >= tgt_prices[t]: hit_tgt[t] = j - i
                    elif direction == -1 and l <= tgt_prices[t]: hit_tgt[t] = j - i
            for s in STOPS:
                if hit_stop[s] is None:
                    if direction == 1 and l <= stop_prices[s]: hit_stop[s] = j - i
                    elif direction == -1 and h >= stop_prices[s]: hit_stop[s] = j - i

        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
            "idx":           i,
            "year":          ts_pd[i].year,
            "sigma":         sigma,
            "gk_ann":        gk_ann,
            "direction":     direction,
            "prior_rets":    prior_rets,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return records, gk_var, bars_per_year


# ── EV helpers ───────────────────────────────────────────────────────────────────

def csr_for(rec, window):
    return float(rec["prior_rets"][-window:].sum()) / rec["sigma"] * rec["direction"]


def ev_stats(sub):
    s, t = PRAC_S, PRAC_T
    if len(sub) < 5:
        return {"ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan"), "n": len(sub)}
    def first(r):
        tv = r[f"hit_tgt_{t}"]  if r[f"hit_tgt_{t}"]  is not None else 999
        sv = r[f"hit_stop_{s}"] if r[f"hit_stop_{s}"] is not None else 999
        return tv, sv
    ht_first = np.array([first(r)[0] <= first(r)[1] and r[f"hit_tgt_{t}"] is not None for r in sub])
    hs_first = np.array([first(r)[1] <  first(r)[0] and r[f"hit_stop_{s}"] is not None for r in sub])
    neither  = ~ht_first & ~hs_first
    time_rets = np.array([r["time_exit_ret"] for r in sub])
    p_tgt  = ht_first.mean()
    p_stop = hs_first.mean()
    ev_nei = time_rets[neither].mean() if neither.any() else 0.0
    ev     = p_tgt * t - p_stop * s + neither.mean() * ev_nei
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(sub)}


# ── Rolling percentile assignment ────────────────────────────────────────────────

def assign_percentile_rank(records, all_gk_vals, roll_window):
    """
    For each trigger, compute the percentile rank of its gk_ann within the
    trailing roll_window bars of ALL bars (not just trigger bars).
    Returns a list of percentile ranks (0–100) aligned with records.
    """
    ranks = []
    for rec in records:
        i = rec["idx"]
        start = max(0, i - roll_window)
        window_vals = all_gk_vals[start: i]
        if len(window_vals) == 0:
            ranks.append(50.0)
        else:
            # Percentile rank: fraction of window values strictly below current value
            rank = float(np.mean(window_vals < rec["gk_ann"])) * 100
            ranks.append(rank)
    return ranks


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None)
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, path in syms.items():
        print(f"\n{'═'*76}")
        print(f"  {sym}  —  Rolling-Percentile Dynamic CSR Window")
        print(f"{'═'*76}")

        bars = make_5min_bars(load_1min(path))
        print(f"  {len(bars):,} 5-min bars — scanning …", end=" ", flush=True)
        records, gk_var_all, bars_per_year = scan(bars)

        # Convert raw GK variance array to annualised vol for each bar
        gk_ann_all = np.sqrt(np.maximum(0, gk_var_all) * bars_per_year)
        print(f"{len(records):,} triggers")

        # ── Baselines ────────────────────────────────────────────────────────────
        fixed8  = [r for r in records if csr_for(r, 8) >= CSR_THRESHOLD]
        fixed4  = [r for r in records if csr_for(r, 4) >= CSR_THRESHOLD]
        abs_thr = [r for r in records
                   if csr_for(r, 4 if r["gk_ann"] < 0.08 else 8) >= CSR_THRESHOLD]

        p8   = ev_stats(fixed8)
        p4   = ev_stats(fixed4)
        pabs = ev_stats(abs_thr)

        print(f"\n  BASELINES")
        print(f"  {'Rule':<38}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}")
        print(f"  {'─'*64}")
        print(f"  {'Fixed 8-bar (40min)':<38}  {p8['n']:>5,}  {p8['p_tgt']:>7.3f}  {p8['p_stop']:>7.3f}  {p8['ev']:>+9.4f}σ")
        print(f"  {'Fixed 4-bar (20min)':<38}  {p4['n']:>5,}  {p4['p_tgt']:>7.3f}  {p4['p_stop']:>7.3f}  {p4['ev']:>+9.4f}σ")
        print(f"  {'Abs threshold gk<8%→4, else→8':<38}  {pabs['n']:>5,}  {pabs['p_tgt']:>7.3f}  {pabs['p_stop']:>7.3f}  {pabs['ev']:>+9.4f}σ")

        # ── 2-tier rolling percentile sweep ──────────────────────────────────────
        print(f"\n  2-TIER ROLLING PERCENTILE  (bottom p% → 4 bars, rest → 8 bars)")
        print(f"  {'Rule':<38}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}  {'vs fixed-8'}")
        print(f"  {'─'*72}")

        best_ev, best_rule = p8["ev"], "fixed-8"
        for rw in ROLL_WINDOWS:
            ranks = assign_percentile_rank(records, gk_ann_all, rw)
            for pct in PCT_THRESHOLDS:
                filtered = [r for r, rank in zip(records, ranks)
                            if csr_for(r, 4 if rank < pct else 8) >= CSR_THRESHOLD]
                p = ev_stats(filtered)
                delta = p["ev"] - p8["ev"] if not math.isnan(p["ev"]) else float("nan")
                flag  = " ◄" if not math.isnan(p["ev"]) and p["ev"] > best_ev else ""
                if flag:
                    best_ev, best_rule = p["ev"], f"roll={rw},p<{pct}"
                label = f"roll={rw:>4}bars, p<{pct:>2}% → 4 bars"
                print(f"  {label:<38}  {p['n']:>5,}  {p['p_tgt']:>7.3f}  "
                      f"{p['p_stop']:>7.3f}  {p['ev']:>+9.4f}σ  {delta:>+7.4f}σ{flag}")
            print()

        # ── 4-tier staircase (best rolling window) ───────────────────────────────
        print(f"  4-TIER STAIRCASE  (p<25→4, p25-50→5, p50-75→6, p>75→8 bars)")
        print(f"  {'Rule':<38}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}  {'vs fixed-8'}")
        print(f"  {'─'*72}")

        for rw in ROLL_WINDOWS:
            ranks = assign_percentile_rank(records, gk_ann_all, rw)
            def staircase(rank):
                if rank < 25:  return 4
                if rank < 50:  return 5
                if rank < 75:  return 6
                return 8
            filtered = [r for r, rank in zip(records, ranks)
                        if csr_for(r, staircase(rank)) >= CSR_THRESHOLD]
            p = ev_stats(filtered)
            delta = p["ev"] - p8["ev"] if not math.isnan(p["ev"]) else float("nan")
            flag  = " ◄" if not math.isnan(p["ev"]) and p["ev"] > best_ev else ""
            if flag:
                best_ev, best_rule = p["ev"], f"staircase roll={rw}"
            label = f"staircase, roll={rw:>4} bars"
            print(f"  {label:<38}  {p['n']:>5,}  {p['p_tgt']:>7.3f}  "
                  f"{p['p_stop']:>7.3f}  {p['ev']:>+9.4f}σ  {delta:>+7.4f}σ{flag}")

        print(f"\n  Best overall: {best_rule}  EV={best_ev:+.4f}σ")

        # ── Year-by-year stability for best 2-tier rule ───────────────────────────
        best_rw  = int(best_rule.split("roll=")[1].split(",")[0]) if "roll=" in best_rule else 500
        best_pct = int(best_rule.split("p<")[1].split("%")[0])    if "p<"    in best_rule else 25

        print(f"\n  YEAR-BY-YEAR STABILITY")
        print(f"  Best rule: roll={best_rw} bars, p<{best_pct}% → 4 bars, else → 8 bars")
        print(f"  vs fixed 8-bar baseline and absolute-threshold rule")
        print(f"\n  {'Year':<6}  {'n(fix8)':>7}  {'EV(fix8)':>9}  "
              f"{'n(abs)':>7}  {'EV(abs)':>9}  "
              f"{'n(pct)':>7}  {'EV(pct)':>9}  {'delta':>8}")
        print(f"  {'─'*72}")

        ranks_best = assign_percentile_rank(records, gk_ann_all, best_rw)
        years = sorted(set(r["year"] for r in records))
        for yr in years:
            yr_idx = [j for j, r in enumerate(records) if r["year"] == yr]
            yr_recs = [records[j] for j in yr_idx]
            yr_ranks = [ranks_best[j] for j in yr_idx]

            f8 = [r for r in yr_recs if csr_for(r, 8) >= CSR_THRESHOLD]
            fa = [r for r in yr_recs
                  if csr_for(r, 4 if r["gk_ann"] < 0.08 else 8) >= CSR_THRESHOLD]
            fp = [r for r, rank in zip(yr_recs, yr_ranks)
                  if csr_for(r, 4 if rank < best_pct else 8) >= CSR_THRESHOLD]

            p8y  = ev_stats(f8)
            pay  = ev_stats(fa)
            ppy  = ev_stats(fp)
            d    = ppy["ev"] - p8y["ev"] if not math.isnan(ppy["ev"]) and not math.isnan(p8y["ev"]) else float("nan")
            flag = " ◄" if not math.isnan(d) and d > 0.02 else ""

            def fe(p): return f"{p['ev']:>+9.4f}σ" if not math.isnan(p["ev"]) else f"{'—':>9}"
            print(f"  {yr:<6}  {p8y['n']:>7,}  {fe(p8y)}  "
                  f"{pay['n']:>7,}  {fe(pay)}  "
                  f"{ppy['n']:>7,}  {fe(ppy)}  {d:>+8.4f}σ{flag}")

        # ── Distribution of percentile threshold over time ────────────────────────
        print(f"\n  THRESHOLD STABILITY: what absolute GK vol level corresponds to")
        print(f"  the {best_pct}th percentile of the roll={best_rw}-bar window, by year?")
        print(f"  (Shows whether 8% is stable or whether the regime shifts)")
        print(f"\n  {'Year':<6}  {'p10':>8}  {'p25':>8}  {'p50':>8}  {'p75':>8}  {'p90':>8}  {'mean':>8}")
        print(f"  {'─'*58}")
        yr_gk = {}
        for j, r in enumerate(records):
            yr_gk.setdefault(r["year"], []).append(r["gk_ann"])
        for yr in years:
            vals = np.array(yr_gk[yr])
            print(f"  {yr:<6}  "
                  f"{np.percentile(vals, 10)*100:>7.1f}%  "
                  f"{np.percentile(vals, 25)*100:>7.1f}%  "
                  f"{np.percentile(vals, 50)*100:>7.1f}%  "
                  f"{np.percentile(vals, 75)*100:>7.1f}%  "
                  f"{np.percentile(vals, 90)*100:>7.1f}%  "
                  f"{vals.mean()*100:>7.1f}%")
