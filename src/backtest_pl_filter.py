"""
Backtest: PL + strong-bar momentum filter layered on the 3σ continuation signal.

At each 3σ trigger, looks back N 1-min bars BEFORE the signal bar and computes:
  - PL_aligned  = (Σr_1min / Σ|r_1min|) × direction  ∈ [-1, 1]
                  positive = 1-min path was trending in signal direction
  - MN          = fraction of those bars with |r_1min| > 0.5 × σ_1min
                  in the signal direction

Baseline = production signal: 3σ + vol≥1.5× + dynamic CSR≥1.5 (4-bar if GK<8%,
           8-bar otherwise) + conditional blackout 08:00–09:00 ET.

Sweeps N ∈ {10, 20, 30}, mn_thresh ∈ {0.50, 0.60}, pl_thresh ∈ {0.50, 0.70, 0.80}.

Usage:
  python src/backtest_pl_filter.py
  python src/backtest_pl_filter.py --sym MYM
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Production signal parameters ─────────────────────────────────────────────

TF              = 5
TRAILING_BARS   = 20
MAX_BARS_HOLD   = 5          # 25 min (production setting)
MIN_SCALED      = 3.0
MIN_VOL_RATIO   = 1.5
STOP_SIG        = 2.0
TGT_SIG         = 3.0
CSR_THRESHOLD   = 1.5
GK_VOL_BARS     = 20
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

# Dynamic CSR window
CSR_LOW_WIN  = 4    # bars when GK vol < 8%
CSR_NORM_WIN = 8    # bars otherwise
GK_LOW_THRESH = 0.08

# Conditional blackout: 08:00–09:00 ET blocks only if CSR < threshold
BLACKOUT_ET = (8, 0, 9, 0)

# ── PL filter sweep parameters ────────────────────────────────────────────────

N_VALUES      = [10, 20, 30]
MN_THRESH     = [0.50, 0.60]
PL_THRESH     = [0.50, 0.70, 0.80]
STRONG_THRESH = 0.5            # |r_1min| > STRONG_THRESH × σ_1min
SIGMA_1M_WIN  = 100            # trailing 1-min bars for σ_1min

MIN_N = 30                     # suppress combos with fewer signals

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    # 1-min log returns and trailing σ
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    sigma_1m      = df["log_ret"].rolling(SIGMA_1M_WIN, min_periods=SIGMA_1M_WIN).std(ddof=1)
    df["sigma_1m"] = sigma_1m * df["close"]   # in price points
    return df


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
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


def gk_vol(opens, highs, lows, closes) -> float:
    bars_per_year = 252 * 23 * 60 / TF
    vals = []
    for o, h, l, c in zip(opens, highs, lows, closes):
        if min(o, h, l, c) <= 0:
            continue
        vals.append(0.5 * math.log(h / l) ** 2 -
                    (2 * math.log(2) - 1) * math.log(c / o) ** 2)
    return math.sqrt(max(0.0, float(np.mean(vals))) * bars_per_year) if vals else 0.0


# ── 5-min signal scanner ──────────────────────────────────────────────────────

def scan_5min(bars: pd.DataFrame) -> list[dict]:
    """
    Scan for 3σ triggers on 5-min bars.
    Returns one record per trigger containing outcome metrics and metadata
    needed for downstream PL filtering.
    """
    MAX_CSR = max(CSR_LOW_WIN, CSR_NORM_WIN)

    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    opens   = bars["open"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)
    records = []

    lookback = max(TRAILING_BARS, MAX_CSR, GK_VOL_BARS) + 1

    for i in range(lookback, n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1] /
                            closes[i - TRAILING_BARS:     i    ])
        sigma = trail_rets.std(ddof=1)
        if sigma == 0:
            continue

        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or abs(scaled) > 99.0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        if vol_ratio < MIN_VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1

        # Dynamic CSR
        gk = gk_vol(opens[i - GK_VOL_BARS: i], highs[i - GK_VOL_BARS: i],
                    lows[i - GK_VOL_BARS: i],  closes[i - GK_VOL_BARS: i])
        csr_win = CSR_LOW_WIN if gk < GK_LOW_THRESH else CSR_NORM_WIN
        prior_rets = np.log(closes[i - csr_win: i] /
                            closes[i - csr_win - 1: i - 1])
        csr = float(prior_rets.sum()) / sigma * direction

        # Conditional blackout: block 08:00–09:00 ET only if CSR < threshold
        bar_et = ts_pd[i].astimezone(ET)
        hm = (bar_et.hour, bar_et.minute)
        sh, sm, eh, em = BLACKOUT_ET
        if (sh, sm) <= hm < (eh, em) and csr < CSR_THRESHOLD:
            continue

        if csr < CSR_THRESHOLD:
            continue

        # Outcomes: first-touch OCO (stop=2σ, target=3σ), then time exit
        entry      = closes[i]
        tgt_px     = entry * math.exp( direction * TGT_SIG  * sigma)
        stop_px    = entry * math.exp(-direction * STOP_SIG * sigma)
        hit_tgt = hit_stop = False
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            if not hit_tgt and not hit_stop:
                if direction == 1:
                    if h >= tgt_px:  hit_tgt  = True
                    elif l <= stop_px: hit_stop = True
                else:
                    if l <= tgt_px:  hit_tgt  = True
                    elif h >= stop_px: hit_stop = True
            elif hit_tgt or hit_stop:
                break

        time_exit_r = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
            "ts":          ts_pd[i],
            "year":        ts_pd[i].year,
            "direction":   direction,
            "sigma":       sigma,
            "gk_vol":      gk,
            "hit_tgt":     hit_tgt,
            "hit_stop":    hit_stop,
            "time_exit_r": time_exit_r,
        })

    return records


# ── PL / M computation from 1-min bars ───────────────────────────────────────

def attach_pl_features(records: list[dict], df1: pd.DataFrame) -> None:
    """
    For each trigger record, look back N 1-min bars BEFORE the signal bar
    and attach pl_aligned_{N} and mn_{N} features in-place.

    The signal bar is a 5-min bar closing at record["ts"].
    Pre-signal 1-min bars are the TF bars ending TF minutes before that close
    (i.e., the bar just before the signal bar started), going back N bars.
    """
    log_ret   = df1["log_ret"].values
    sigma_1m  = df1["sigma_1m"].values    # in price points
    closes_1m = df1["close"].values

    # Build fast timestamp → int-index lookup.
    # Force nanosecond int64 regardless of the DataFrame's internal resolution
    # (pandas 2.0 uses datetime64[us]; Timestamp.value is always ns).
    ts_ns = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts_map = {v: i for i, v in enumerate(ts_ns)}

    for rec in records:
        sig_ts_ns = int(rec["ts"].as_unit("ns").value)   # ns since epoch
        pre_ts_ns = sig_ts_ns - TF * 60 * 10**9          # TF minutes earlier

        end_idx = ts_map.get(pre_ts_ns)
        if end_idx is None:
            continue

        direction = rec["direction"]

        for N in N_VALUES:
            start_idx = end_idx - N + 1
            if start_idx < SIGMA_1M_WIN:
                continue

            rets   = log_ret[start_idx: end_idx + 1]
            sig_pt = sigma_1m[end_idx]
            cl     = closes_1m[end_idx]

            if np.any(np.isnan(rets)) or math.isnan(sig_pt) or sig_pt <= 0:
                continue

            sum_r    = float(rets.sum())
            sum_absr = float(np.abs(rets).sum())
            if sum_absr == 0:
                continue

            pl_aligned = (sum_r / sum_absr) * direction   # positive = trending with signal

            # σ in log-return units for the strong-bar count
            sig_logret = sig_pt / cl
            strong = int(np.sum((rets * direction) > STRONG_THRESH * sig_logret))
            mn     = strong / N

            rec[f"pl_{N}"] = pl_aligned
            rec[f"mn_{N}"] = mn


# ── EV helper ─────────────────────────────────────────────────────────────────

def ev_stats(recs: list[dict]) -> dict:
    n = len(recs)
    if n < 2:
        return {"n": n, "ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan")}
    p_tgt  = sum(1 for r in recs if r["hit_tgt"]  and not r["hit_stop"]) / n
    p_stop = sum(1 for r in recs if r["hit_stop"] and not r["hit_tgt"])  / n
    p_nei  = 1.0 - p_tgt - p_stop
    neither_rets = [r["time_exit_r"] for r in recs
                    if not r["hit_tgt"] and not r["hit_stop"]]
    ev_nei = float(np.mean(neither_rets)) if neither_rets else 0.0
    ev = p_tgt * TGT_SIG - p_stop * STOP_SIG + p_nei * ev_nei
    return {"n": n, "ev": ev, "p_tgt": p_tgt, "p_stop": p_stop}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(records: list[dict], sym: str):
    base = ev_stats(records)
    sigs_per_year = base["n"] / max(1, len(set(r["year"] for r in records)))

    print(f"\n{'═'*76}")
    print(f"  {sym}  —  PL + Strong-Bar Filter on 3σ Signal")
    print(f"{'═'*76}")
    print(f"\n  Baseline (3σ + vol + dynamic CSR + blackout):  "
          f"n={base['n']:,}  (~{sigs_per_year:.0f}/yr  "
          f"~{sigs_per_year/252:.2f}/day)")
    print(f"  P(tgt)={base['p_tgt']:.3f}  P(stop)={base['p_stop']:.3f}  "
          f"EV={base['ev']:+.4f}σ")

    for N in N_VALUES:
        K = max(1, N // 10)
        eligible = [r for r in records if f"pl_{N}" in r]
        if not eligible:
            print(f"\n  N={N}: no 1-min data matched")
            continue

        print(f"\n  ── N={N} 1-min bars before signal bar  (K={K} bars fwd in prior study) ──")
        print(f"  {'MN>':>6}  {'PL>':>6}  {'n':>6}  {'↓sig%':>6}  "
              f"{'P(tgt)':>7}  {'P(stp)':>7}  {'EV':>9}  {'ΔEV':>8}  {'yr/day':>8}")
        print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  "
              f"{'-'*7}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*8}")

        for mn_t in MN_THRESH:
            for pl_t in PL_THRESH:
                sub = [r for r in eligible
                       if r[f"mn_{N}"] > mn_t and r[f"pl_{N}"] > pl_t]
                s = ev_stats(sub)
                if s["n"] < MIN_N:
                    print(f"  {mn_t:.2f}   {pl_t:.2f}   "
                          f"{'<30':>6}  {'—':>6}  {'—':>7}  {'—':>7}  {'—':>9}  {'—':>8}  {'—':>8}")
                    continue
                pct_kept = s["n"] / base["n"] * 100
                delta    = s["ev"] - base["ev"]
                spd      = s["n"] / max(1, len(set(r["year"] for r in sub))) / 252
                flag     = " ◄" if delta > 0.02 else ""
                print(f"  {mn_t:.2f}   {pl_t:.2f}   "
                      f"{s['n']:>6,}  {pct_kept:>5.0f}%  "
                      f"{s['p_tgt']:>7.3f}  {s['p_stop']:>7.3f}  "
                      f"{s['ev']:>+9.4f}  {delta:>+8.4f}  "
                      f"{spd:>7.2f}/d{flag}")

    # Year-by-year for best-looking combo per N
    for N in N_VALUES:
        eligible = [r for r in records if f"pl_{N}" in r]
        if not eligible:
            continue
        # Use MN>0.50, PL>0.70 as representative
        for mn_t, pl_t in [(0.50, 0.70), (0.60, 0.70)]:
            sub = [r for r in eligible
                   if r[f"mn_{N}"] > mn_t and r[f"pl_{N}"] > pl_t]
            if len(sub) < MIN_N:
                continue
            years = sorted(set(r["year"] for r in sub))
            print(f"\n  Year-by-year  N={N}, MN>{mn_t:.2f}, PL>{pl_t:.2f}  "
                  f"(baseline EV={base['ev']:+.4f}σ):")
            print(f"  {'Year':>6}  {'n':>5}  {'P(tgt)':>7}  {'P(stp)':>7}  "
                  f"{'EV':>9}  {'ΔEV':>8}")
            for yr in years:
                yr_sub = [r for r in sub if r["year"] == yr]
                if len(yr_sub) < 5:
                    continue
                yr_base = [r for r in records if r["year"] == yr]
                s  = ev_stats(yr_sub)
                sb = ev_stats(yr_base)
                d  = s["ev"] - sb["ev"]
                flag = " ◄" if d > 0.02 else ""
                print(f"  {yr:>6}  {len(yr_sub):>5,}  "
                      f"{s['p_tgt']:>7.3f}  {s['p_stop']:>7.3f}  "
                      f"{s['ev']:>+9.4f}  {d:>+8.4f}{flag}")
            break   # only print one combo per N to keep output manageable


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=list(INSTRUMENTS.keys()))
    args = parser.parse_args()

    path = INSTRUMENTS[args.sym]
    print(f"Loading {path} …", flush=True)
    df1 = load_1min(path)
    print(f"  {len(df1):,} 1-min bars loaded", flush=True)

    print("Building 5-min bars …", flush=True)
    bars5 = make_5min_bars(df1)
    print(f"  {len(bars5):,} 5-min bars", flush=True)

    print("Scanning for 3σ triggers …", flush=True)
    records = scan_5min(bars5)
    print(f"  {len(records):,} baseline triggers", flush=True)

    print("Computing 1-min PL features …", flush=True)
    attach_pl_features(records, df1)
    matched = sum(1 for r in records if "pl_20" in r)
    print(f"  {matched:,} triggers with 1-min data matched", flush=True)

    print_results(records, args.sym)
    print()


if __name__ == "__main__":
    main()
