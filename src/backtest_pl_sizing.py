"""
Two follow-on analyses of the 1-min PL feature vs the 3σ signal.

Uses N=10 1-min bars before each signal bar (all 1,091 signals matched).

1. INVERSE FILTER
   Does negative/low PL_aligned predict signal failure?
   Sweep a minimum PL_aligned threshold:  drop signals below it, keep the rest.
   → Shows whether removing the worst-context signals improves EV while
     preserving most of the signal count.

2. CONFIDENCE SIZING
   Allocate 2× position to high-PL_aligned signals, 1× to the rest.
   Sweep the "high" cutoff:  top 10%, 15%, 20%, 25% by PL_aligned.
   → Shows total P&L (sum of sized R outcomes) vs equal-weight baseline.

Usage:
  python src/backtest_pl_sizing.py
  python src/backtest_pl_sizing.py --sym MYM
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Parameters (must match backtest_pl_filter.py / production) ────────────────

TF              = 5
TRAILING_BARS   = 20
MAX_BARS_HOLD   = 5
MIN_SCALED      = 3.0
MIN_VOL_RATIO   = 1.5
STOP_SIG        = 2.0
TGT_SIG         = 3.0
CSR_THRESHOLD   = 1.5
GK_VOL_BARS     = 20
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
CSR_LOW_WIN     = 4
CSR_NORM_WIN    = 8
GK_LOW_THRESH   = 0.08
BLACKOUT_ET     = (8, 0, 9, 0)

N               = 10          # 1-min lookback (all signals matched at this value)
STRONG_THRESH   = 0.5
SIGMA_1M_WIN    = 100

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
    "NKD": "nkd_hist_1min.csv",
}

# ── Data (identical to backtest_pl_filter.py) ─────────────────────────────────

def load_1min(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
    sigma_1m       = df["log_ret"].rolling(SIGMA_1M_WIN, min_periods=SIGMA_1M_WIN).std(ddof=1)
    df["sigma_1m"] = sigma_1m * df["close"]
    return df


def make_5min_bars(df1):
    df1 = df1.copy()
    df1["gap"] = df1["ts"].diff() > pd.Timedelta(minutes=2)
    records, i, n = [], 0, len(df1)
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
        records.append({"ts": chunk["ts"].iloc[-1], "open": chunk["open"].iloc[0],
                        "high": chunk["high"].max(), "low": chunk["low"].min(),
                        "close": chunk["close"].iloc[-1], "volume": chunk["volume"].sum()})
        i += TF
    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


def gk_vol(opens, highs, lows, closes):
    bpy = 252 * 23 * 60 / TF
    vals = [0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2
            for o, h, l, c in zip(opens, highs, lows, closes) if min(o, h, l, c) > 0]
    return math.sqrt(max(0.0, float(np.mean(vals))) * bpy) if vals else 0.0


def scan_5min(bars):
    MAX_CSR  = max(CSR_LOW_WIN, CSR_NORM_WIN)
    closes   = bars["close"].values;  highs  = bars["high"].values
    lows     = bars["low"].values;    opens  = bars["open"].values
    volumes  = bars["volume"].values; gaps   = bars["gap"].values
    ts_pd    = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n        = len(bars);  records = []
    lookback = max(TRAILING_BARS, MAX_CSR, GK_VOL_BARS) + 1

    for i in range(lookback, n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue
        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1] /
                            closes[i - TRAILING_BARS:     i    ])
        sigma = trail_rets.std(ddof=1)
        if sigma == 0:
            continue
        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or abs(scaled) > 99.0:
            continue
        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        if vol_ratio < MIN_VOL_RATIO:
            continue
        direction = 1 if scaled > 0 else -1
        gk = gk_vol(opens[i - GK_VOL_BARS: i], highs[i - GK_VOL_BARS: i],
                    lows[i - GK_VOL_BARS: i],  closes[i - GK_VOL_BARS: i])
        csr_win    = CSR_LOW_WIN if gk < GK_LOW_THRESH else CSR_NORM_WIN
        prior_rets = np.log(closes[i - csr_win: i] / closes[i - csr_win - 1: i - 1])
        csr        = float(prior_rets.sum()) / sigma * direction
        bar_et     = ts_pd[i].astimezone(ET)
        hm         = (bar_et.hour, bar_et.minute)
        sh, sm, eh, em = BLACKOUT_ET
        if (sh, sm) <= hm < (eh, em) and csr < CSR_THRESHOLD:
            continue
        if csr < CSR_THRESHOLD:
            continue

        entry   = closes[i]
        tgt_px  = entry * math.exp( direction * TGT_SIG  * sigma)
        stop_px = entry * math.exp(-direction * STOP_SIG * sigma)
        hit_tgt = hit_stop = False
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            if not hit_tgt and not hit_stop:
                if direction == 1:
                    if h >= tgt_px: hit_tgt = True
                    elif l <= stop_px: hit_stop = True
                else:
                    if l <= tgt_px: hit_tgt = True
                    elif h >= stop_px: hit_stop = True
            else:
                break
        time_exit_r = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        # Per-trade R outcome
        if hit_tgt and not hit_stop:
            outcome_r = TGT_SIG
        elif hit_stop and not hit_tgt:
            outcome_r = -STOP_SIG
        else:
            outcome_r = time_exit_r

        records.append({"ts": ts_pd[i], "year": ts_pd[i].year,
                        "direction": direction, "sigma": sigma,
                        "hit_tgt": hit_tgt, "hit_stop": hit_stop,
                        "time_exit_r": time_exit_r, "outcome_r": outcome_r})
    return records


def attach_pl(records, df1):
    log_ret   = df1["log_ret"].values
    sigma_1m  = df1["sigma_1m"].values
    closes_1m = df1["close"].values
    ts_ns     = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts_map    = {v: i for i, v in enumerate(ts_ns)}

    for rec in records:
        sig_ts_ns = int(rec["ts"].as_unit("ns").value)
        pre_ts_ns = sig_ts_ns - TF * 60 * 10**9
        end_idx   = ts_map.get(pre_ts_ns)
        if end_idx is None:
            rec["pl_aligned"] = float("nan")
            rec["mn"]         = float("nan")
            continue
        start_idx = end_idx - N + 1
        if start_idx < SIGMA_1M_WIN:
            rec["pl_aligned"] = float("nan")
            rec["mn"]         = float("nan")
            continue
        rets     = log_ret[start_idx: end_idx + 1]
        sig_pt   = sigma_1m[end_idx]
        cl       = closes_1m[end_idx]
        if np.any(np.isnan(rets)) or math.isnan(sig_pt) or sig_pt <= 0:
            rec["pl_aligned"] = float("nan")
            rec["mn"]         = float("nan")
            continue
        sum_absr = float(np.abs(rets).sum())
        if sum_absr == 0:
            rec["pl_aligned"] = 0.0
            rec["mn"]         = 0.0
            continue
        pl_aligned = float(rets.sum()) / sum_absr * rec["direction"]
        sig_lr     = sig_pt / cl
        mn         = int(np.sum((rets * rec["direction"]) > STRONG_THRESH * sig_lr)) / N
        rec["pl_aligned"] = pl_aligned
        rec["mn"]         = mn


# ── EV helpers ────────────────────────────────────────────────────────────────

def ev_stats(recs):
    n = len(recs)
    if n < 5:
        return {"n": n, "ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "total_r": float("nan")}
    p_tgt  = sum(1 for r in recs if r["hit_tgt"]  and not r["hit_stop"]) / n
    p_stop = sum(1 for r in recs if r["hit_stop"] and not r["hit_tgt"])  / n
    p_nei  = 1.0 - p_tgt - p_stop
    nei    = [r["time_exit_r"] for r in recs if not r["hit_tgt"] and not r["hit_stop"]]
    ev_nei = float(np.mean(nei)) if nei else 0.0
    ev     = p_tgt * TGT_SIG - p_stop * STOP_SIG + p_nei * ev_nei
    return {"n": n, "ev": ev, "p_tgt": p_tgt, "p_stop": p_stop,
            "total_r": sum(r["outcome_r"] for r in recs)}


# ── Analysis 1: Inverse filter ────────────────────────────────────────────────

def analysis_inverse_filter(records, base):
    print(f"\n{'═'*72}")
    print(f"  ANALYSIS 1 — INVERSE FILTER")
    print(f"  Drop signals where PL_aligned < threshold; keep the rest.")
    print(f"  N={N} 1-min bars before signal. Baseline n={base['n']}, "
          f"EV={base['ev']:+.4f}σ")
    print(f"{'─'*72}")

    recs = [r for r in records if not math.isnan(r["pl_aligned"])]

    thresholds = [-0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    print(f"\n  {'MinPL':>7}  {'n kept':>7}  {'kept%':>6}  "
          f"{'P(tgt)':>7}  {'P(stp)':>7}  {'EV':>9}  {'ΔEV':>8}  "
          f"{'tot R':>8}  {'Δtot R':>8}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*6}  "
          f"{'-'*7}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}")

    base_total = base["total_r"]
    for thr in thresholds:
        sub = [r for r in recs if r["pl_aligned"] >= thr]
        s   = ev_stats(sub)
        if math.isnan(s["ev"]):
            continue
        pct    = s["n"] / base["n"] * 100
        d_ev   = s["ev"]   - base["ev"]
        d_tot  = s["total_r"] - base_total
        flag   = " ◄" if d_ev > 0.02 and pct > 50 else ""
        print(f"  {thr:>+7.2f}  {s['n']:>7,}  {pct:>5.0f}%  "
              f"{s['p_tgt']:>7.3f}  {s['p_stop']:>7.3f}  "
              f"{s['ev']:>+9.4f}  {d_ev:>+8.4f}  "
              f"{s['total_r']:>+8.2f}  {d_tot:>+8.2f}{flag}")

    # Best practical threshold: first one where kept% > 50% and EV > baseline
    print(f"\n  PL_aligned distribution (of {len(recs)} matched signals):")
    vals = np.array([r["pl_aligned"] for r in recs])
    for label, pct in [("p5", 5), ("p25", 25), ("p50", 50), ("p75", 75), ("p95", 95)]:
        print(f"    {label}: {np.percentile(vals, pct):+.3f}")

    # Quartile EV breakdown
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    print(f"\n  EV by PL_aligned quartile:")
    print(f"  {'Quartile':>20}  {'n':>5}  {'P(tgt)':>7}  {'P(stp)':>7}  "
          f"{'EV':>9}  {'ΔEV':>8}")
    buckets = [
        (f"Q1 (≤{q25:+.2f})",  [r for r in recs if r["pl_aligned"] <= q25]),
        (f"Q2 ({q25:+.2f}–{q50:+.2f})", [r for r in recs if q25 < r["pl_aligned"] <= q50]),
        (f"Q3 ({q50:+.2f}–{q75:+.2f})", [r for r in recs if q50 < r["pl_aligned"] <= q75]),
        (f"Q4 (>{q75:+.2f})",  [r for r in recs if r["pl_aligned"] > q75]),
    ]
    for label, grp in buckets:
        s = ev_stats(grp)
        if math.isnan(s["ev"]):
            continue
        d = s["ev"] - base["ev"]
        flag = " ◄" if d > 0.02 else (" ✗" if d < -0.05 else "")
        print(f"  {label:>20}  {s['n']:>5,}  {s['p_tgt']:>7.3f}  "
              f"{s['p_stop']:>7.3f}  {s['ev']:>+9.4f}  {d:>+8.4f}{flag}")

    # Year-by-year for a practical threshold (drop Q1)
    print(f"\n  Year-by-year: drop Q1 (PL_aligned ≤ {q25:+.2f})  vs  baseline")
    filtered = [r for r in recs if r["pl_aligned"] > q25]
    years    = sorted(set(r["year"] for r in recs))
    print(f"  {'Year':>6}  {'n(base)':>8}  {'EV(base)':>9}  "
          f"{'n(filt)':>8}  {'EV(filt)':>9}  {'ΔEV':>8}")
    for yr in years:
        yr_base = [r for r in recs   if r["year"] == yr]
        yr_filt = [r for r in filtered if r["year"] == yr]
        sb = ev_stats(yr_base);  sf = ev_stats(yr_filt)
        if math.isnan(sb["ev"]) or math.isnan(sf["ev"]) or sb["n"] < 5:
            continue
        d    = sf["ev"] - sb["ev"]
        flag = " ◄" if d > 0.02 else ""
        print(f"  {yr:>6}  {sb['n']:>8,}  {sb['ev']:>+9.4f}  "
              f"{sf['n']:>8,}  {sf['ev']:>+9.4f}  {d:>+8.4f}{flag}")


# ── Analysis 2: Confidence sizing ─────────────────────────────────────────────

def analysis_sizing(records, base):
    print(f"\n{'═'*72}")
    print(f"  ANALYSIS 2 — CONFIDENCE SIZING  (2× on high-PL signals)")
    print(f"  Baseline: all {base['n']} signals at 1 unit.  "
          f"Total base R = {base['total_r']:+.2f}R")
    print(f"{'─'*72}")

    recs = [r for r in records if not math.isnan(r["pl_aligned"])]
    vals = np.array([r["pl_aligned"] for r in recs])

    # Sweep "top X%" cutoffs for 2× sizing
    cutoffs_pct = [10, 15, 20, 25, 30]

    print(f"\n  Sizing rule: top X% of signals by PL_aligned get 2 units; rest get 1 unit.")
    print(f"\n  {'Top%':>6}  {'n(2×)':>7}  {'PL cut':>8}  "
          f"{'2×totR':>9}  {'1×totR':>9}  {'SizedR':>9}  "
          f"{'BaseR':>9}  {'ΔtotR':>8}  {'Δ/unit':>8}")
    print(f"  {'-'*6}  {'-'*7}  {'-'*8}  "
          f"{'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*8}")

    base_total = base["total_r"]
    for top_pct in cutoffs_pct:
        cutoff   = float(np.percentile(vals, 100 - top_pct))
        big_recs = [r for r in recs if r["pl_aligned"] >= cutoff]
        sml_recs = [r for r in recs if r["pl_aligned"] <  cutoff]
        big_r    = sum(r["outcome_r"] for r in big_recs)
        sml_r    = sum(r["outcome_r"] for r in sml_recs)
        sized_r  = 2 * big_r + 1 * sml_r
        total_units = 2 * len(big_recs) + 1 * len(sml_recs)
        delta_r  = sized_r - base_total
        # per-unit EV: sized_r / total_units vs base_total / base["n"]
        delta_pu = sized_r / total_units - base_total / base["n"]
        flag     = " ◄" if delta_r > 0 and delta_pu > 0 else ""
        print(f"  {top_pct:>5}%  {len(big_recs):>7,}  {cutoff:>+8.3f}  "
              f"{big_r:>+9.2f}  {sml_r:>+9.2f}  {sized_r:>+9.2f}  "
              f"{base_total:>+9.2f}  {delta_r:>+8.2f}  {delta_pu:>+8.4f}{flag}")

    # Year-by-year for top 20%
    top_pct  = 20
    cutoff   = float(np.percentile(vals, 80))
    print(f"\n  Year-by-year sizing: top {top_pct}% (PL_aligned ≥ {cutoff:+.3f}) → 2×, rest → 1×")
    print(f"  {'Year':>6}  {'n':>5}  {'n(2×)':>6}  "
          f"{'BaseR':>8}  {'SizedR':>8}  {'ΔR':>7}  "
          f"{'EV/unit base':>13}  {'EV/unit sized':>14}")
    years = sorted(set(r["year"] for r in recs))
    for yr in years:
        yr_recs  = [r for r in recs if r["year"] == yr]
        if len(yr_recs) < 5:
            continue
        big_yr   = [r for r in yr_recs if r["pl_aligned"] >= cutoff]
        sml_yr   = [r for r in yr_recs if r["pl_aligned"] <  cutoff]
        base_r   = sum(r["outcome_r"] for r in yr_recs)
        sized_r  = 2 * sum(r["outcome_r"] for r in big_yr) + \
                   1 * sum(r["outcome_r"] for r in sml_yr)
        n_total  = len(yr_recs)
        n_big    = len(big_yr)
        units    = 2 * n_big + (n_total - n_big)
        ev_base  = base_r  / n_total
        ev_sized = sized_r / units
        d        = sized_r - base_r
        flag     = " ◄" if d > 0 else ""
        print(f"  {yr:>6}  {n_total:>5,}  {n_big:>6,}  "
              f"{base_r:>+8.2f}  {sized_r:>+8.2f}  {d:>+7.2f}  "
              f"{ev_base:>+13.4f}  {ev_sized:>+14.4f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=sorted(INSTRUMENTS.keys()))
    args = parser.parse_args()

    path = INSTRUMENTS[args.sym]
    print(f"Loading {path} …", flush=True)
    df1   = load_1min(path)
    bars5 = make_5min_bars(df1)
    print(f"  {len(bars5):,} 5-min bars", flush=True)

    print("Scanning …", flush=True)
    records = scan_5min(bars5)
    print(f"  {len(records):,} baseline triggers", flush=True)

    attach_pl(records, df1)
    matched = sum(1 for r in records if not math.isnan(r.get("pl_aligned", float("nan"))))
    print(f"  {matched:,} triggers with 1-min PL computed", flush=True)

    base = ev_stats(records)

    analysis_inverse_filter(records, base)
    analysis_sizing(records, base)
    print()


if __name__ == "__main__":
    main()
