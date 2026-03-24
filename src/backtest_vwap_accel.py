"""
Backtest: VWAP acceleration as a predictor of post-signal continuation.

For each 3σ signal, computes the VWAP acceleration at signal time:

  dist(t)      = price(t) - session_vwap(t)          # raw signed distance
  accel(t)     = EMA(dist, fast) - EMA(dist, slow)    # MACD of VWAP distance
  accel_aligned = accel(t) × signal_direction         # positive = accelerating
                                                       # away from VWAP in
                                                       # signal direction

Session VWAP resets at 09:30 ET each day.

Analyses:
  1. accel_aligned quartile → forward 10 / 20 / 30-min return in signal direction
  2. Sweep fast/slow EMA combos to find best parameters
  3. 2×2 matrix: accel_aligned (high/low) × PL_aligned (high/low)
  4. Sizing: 2× when accel_aligned ≥ threshold, compare vs PL sizing

Usage:
  python src/backtest_vwap_accel.py
  python src/backtest_vwap_accel.py --sym MYM
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Parameters ────────────────────────────────────────────────────────────────

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
SIGMA_1M_WIN    = 100
N_PL            = 10
PL_THRESH       = 0.50

FWD_WINDOWS     = [10, 20, 30]          # 1-min bars forward to check
EMA_COMBOS      = [(5, 20), (5, 30), (10, 20), (10, 30)]  # (fast, slow)

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def _ema_series(arr, period):
    a = 2 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1 - a) * out[i - 1]
    return out


def load_1min(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # 1-min σ
    df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
    sigma_1m       = df["log_ret"].rolling(SIGMA_1M_WIN, min_periods=SIGMA_1M_WIN).std(ddof=1)
    df["sigma_1m"] = sigma_1m * df["close"]

    # Session VWAP (reset at 09:30 ET each day)
    ts_et      = df["ts"].dt.tz_convert(ET)
    df["date"] = ts_et.dt.date
    df["hm"]   = ts_et.dt.hour * 60 + ts_et.dt.minute
    typical    = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"]  = typical * df["volume"]
    df["session_vwap"] = float("nan")
    for date, grp in df.groupby("date"):
        rth = grp[grp["hm"] >= 9 * 60 + 30]
        if rth.empty:
            continue
        cum_tpv = rth["tpv"].cumsum()
        cum_vol = rth["volume"].cumsum()
        with np.errstate(invalid="ignore"):
            vwap = cum_tpv / cum_vol.replace(0, float("nan"))
        df.loc[rth.index, "session_vwap"] = vwap.values

    # Raw VWAP distance (signed, not direction-adjusted)
    df["vwap_dist"] = df["close"] - df["session_vwap"]

    # VWAP acceleration for each EMA combo
    dist = df["vwap_dist"].values.copy()
    # fill NaN with 0 for EMA warmup (pre-market)
    dist_filled = np.where(np.isnan(dist), 0.0, dist)
    for fast, slow in EMA_COMBOS:
        fast_ema = _ema_series(dist_filled, fast)
        slow_ema = _ema_series(dist_filled, slow)
        df[f"accel_{fast}_{slow}"] = fast_ema - slow_ema
        # zero out pre-VWAP bars so they don't pollute signal lookups
        df.loc[df["session_vwap"].isna(), f"accel_{fast}_{slow}"] = float("nan")

    return df


def make_5min_bars(df1):
    df1  = df1.copy()
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


def gk_vol(opens, highs, lows, closes):
    bpy  = 252 * 23 * 60 / TF
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
                    if h >= tgt_px:    hit_tgt  = True
                    elif l <= stop_px: hit_stop = True
                else:
                    if l <= tgt_px:    hit_tgt  = True
                    elif h >= stop_px: hit_stop = True
            else:
                break

        time_exit_r = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma
        if hit_tgt and not hit_stop:
            outcome_r = TGT_SIG
        elif hit_stop and not hit_tgt:
            outcome_r = -STOP_SIG
        else:
            outcome_r = time_exit_r

        records.append({
            "ts": ts_pd[i], "year": ts_pd[i].year,
            "direction": direction, "sigma": sigma,
            "sigma_pts": sigma * closes[i], "close": closes[i],
            "hit_tgt": hit_tgt, "hit_stop": hit_stop,
            "time_exit_r": time_exit_r, "outcome_r": outcome_r,
        })
    return records


# ── Feature attachment ────────────────────────────────────────────────────────

def attach_features(records, df1):
    closes_1m = df1["close"].values
    log_ret   = df1["log_ret"].values
    sigma_1m  = df1["sigma_1m"].values
    ts_ns     = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts_map    = {v: i for i, v in enumerate(ts_ns)}
    n1m       = len(df1)

    # Precompute all accel columns as arrays
    accel_arrays = {}
    for fast, slow in EMA_COMBOS:
        accel_arrays[(fast, slow)] = df1[f"accel_{fast}_{slow}"].values

    for rec in records:
        sig_ts_ns = int(rec["ts"].as_unit("ns").value)
        sig_idx   = ts_map.get(sig_ts_ns)
        pre_ts_ns = sig_ts_ns - TF * 60 * 10**9
        pre_idx   = ts_map.get(pre_ts_ns)
        d         = rec["direction"]

        # PL_aligned
        if pre_idx is None or pre_idx - N_PL + 1 < SIGMA_1M_WIN:
            rec["pl_aligned"] = float("nan")
        else:
            rets     = log_ret[pre_idx - N_PL + 1: pre_idx + 1]
            sig_pt   = sigma_1m[pre_idx]
            if np.any(np.isnan(rets)) or math.isnan(sig_pt) or sig_pt <= 0:
                rec["pl_aligned"] = float("nan")
            else:
                sa = float(np.abs(rets).sum())
                rec["pl_aligned"] = float(rets.sum()) / sa * d if sa > 0 else 0.0

        # VWAP acceleration (direction-adjusted) for each combo
        for fast, slow in EMA_COMBOS:
            key = f"accel_{fast}_{slow}"
            if sig_idx is None:
                rec[key] = float("nan")
            else:
                val = accel_arrays[(fast, slow)][sig_idx]
                rec[key] = float(val) * d if not math.isnan(val) else float("nan")

        # Forward 1-min returns in signal direction (10, 20, 30 bars)
        if sig_idx is None:
            for k in FWD_WINDOWS:
                rec[f"fwd_{k}"] = float("nan")
        else:
            for k in FWD_WINDOWS:
                fwd_idx = sig_idx + k
                if fwd_idx >= n1m:
                    rec[f"fwd_{k}"] = float("nan")
                else:
                    fwd_r = math.log(closes_1m[fwd_idx] / closes_1m[sig_idx])
                    sp    = rec["sigma_pts"]
                    rec[f"fwd_{k}"] = fwd_r * d / (sp / rec["close"]) if sp > 0 else float("nan")


# ── Reporting ─────────────────────────────────────────────────────────────────

def ev_stats(recs):
    n = len(recs)
    if n < 5:
        return {"n": n, "ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "total_r": float("nan")}
    p_tgt  = sum(1 for r in recs if r["hit_tgt"]  and not r["hit_stop"]) / n
    p_stop = sum(1 for r in recs if r["hit_stop"] and not r["hit_tgt"])  / n
    nei    = [r["time_exit_r"] for r in recs if not r["hit_tgt"] and not r["hit_stop"]]
    ev_nei = float(np.mean(nei)) if nei else 0.0
    ev     = p_tgt * TGT_SIG - p_stop * STOP_SIG + (1 - p_tgt - p_stop) * ev_nei
    return {"n": n, "ev": ev, "p_tgt": p_tgt, "p_stop": p_stop,
            "total_r": sum(r["outcome_r"] for r in recs)}


def analyse(records, sym):
    # Pick best EMA combo by correlation with forward 20-min return
    print(f"\n{'═'*76}")
    print(f"  {sym}  VWAP Acceleration  —  EMA combo selection")
    print(f"{'─'*76}")
    print(f"\n  {'Combo':>12}  {'corr fwd10':>10}  {'corr fwd20':>10}  {'corr fwd30':>10}  "
          f"{'n valid':>8}")

    best_combo = None
    best_corr  = -999.0
    for fast, slow in EMA_COMBOS:
        key  = f"accel_{fast}_{slow}"
        recs = [r for r in records
                if not math.isnan(r.get(key, float("nan")))]
        if len(recs) < 10:
            continue
        accel_v = np.array([r[key] for r in recs])
        corrs   = []
        for k in FWD_WINDOWS:
            fwd_v = np.array([r[f"fwd_{k}"] for r in recs])
            mask  = ~np.isnan(fwd_v)
            if mask.sum() < 10:
                corrs.append(float("nan"))
            else:
                corrs.append(float(np.corrcoef(accel_v[mask], fwd_v[mask])[0, 1]))
        mean_corr = np.nanmean(corrs)
        if mean_corr > best_corr:
            best_corr  = mean_corr
            best_combo = (fast, slow)
        c_str = "  ".join(f"{c:>+.3f}" if not math.isnan(c) else "   nan" for c in corrs)
        print(f"  EMA({fast:>2},{slow:>2}):     {c_str}    {len(recs):>6,}")

    fast, slow = best_combo
    key = f"accel_{fast}_{slow}"
    print(f"\n  → Best combo: EMA({fast},{slow})")

    recs = [r for r in records
            if not math.isnan(r.get(key, float("nan")))
            and not math.isnan(r.get(f"fwd_10", float("nan")))]

    accel_v = np.array([r[key] for r in recs])
    q25, q50, q75 = np.percentile(accel_v, [25, 50, 75])

    # ── Quartile breakdown vs forward returns ─────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  accel_aligned quartiles → forward returns  (n={len(recs):,})")
    print(f"{'─'*76}")
    print(f"\n  accel_aligned = EMA({fast},{slow}) of (price−VWAP) × direction")
    print(f"  positive = accelerating away from VWAP in signal direction\n")
    print(f"  {'Quartile':>22}  {'n':>5}  {'EV(5-bar)':>10}  "
          + "  ".join(f"{'fwd'+str(k)+'m':>8}" for k in FWD_WINDOWS))
    print(f"  {'─'*22}  {'─'*5}  {'─'*10}  " + "  ".join(['─'*8]*len(FWD_WINDOWS)))

    buckets = [
        (f"Q1 (≤{q25:+.2f})", [r for r in recs if r[key] <= q25]),
        (f"Q2 ({q25:+.2f}–{q50:+.2f})", [r for r in recs if q25 < r[key] <= q50]),
        (f"Q3 ({q50:+.2f}–{q75:+.2f})", [r for r in recs if q50 < r[key] <= q75]),
        (f"Q4 (>{q75:+.2f})", [r for r in recs if r[key] > q75]),
        ("ALL", recs),
    ]
    for label, grp in buckets:
        s   = ev_stats(grp)
        fwd = []
        for k in FWD_WINDOWS:
            vs = [r[f"fwd_{k}"] for r in grp if not math.isnan(r[f"fwd_{k}"])]
            fwd.append(f"{np.mean(vs):>+8.3f}σ" if vs else "     nan")
        ev_str = f"{s['ev']:>+10.4f}" if not math.isnan(s["ev"]) else "       nan"
        print(f"  {label:>22}  {len(grp):>5,}  {ev_str}  " + "  ".join(fwd))

    # ── 2×2: accel × PL ──────────────────────────────────────────────────────
    full = [r for r in recs if not math.isnan(r.get("pl_aligned", float("nan")))]
    med_accel = np.median([r[key] for r in full])

    hh = [r for r in full if r[key] >= med_accel and r["pl_aligned"] >= PL_THRESH]
    hl = [r for r in full if r[key] >= med_accel and r["pl_aligned"] <  PL_THRESH]
    lh = [r for r in full if r[key] <  med_accel and r["pl_aligned"] >= PL_THRESH]
    ll = [r for r in full if r[key] <  med_accel and r["pl_aligned"] <  PL_THRESH]

    print(f"\n{'═'*76}")
    print(f"  2×2  accel ≥ median ({med_accel:+.2f}) × PL_aligned ≥ {PL_THRESH:.2f}")
    print(f"{'─'*76}")
    print(f"  {'':30}  {'PL ≥ +0.50':>14}  {'PL < +0.50':>14}")
    for label, ga, gb in [
        (f"Accel ≥ {med_accel:+.2f}  (strong)", hh, hl),
        (f"Accel <  {med_accel:+.2f}  (weak)",   lh, ll),
    ]:
        def fmt(g):
            s = ev_stats(g)
            return f"{len(g):>4}  {s['ev']:>+7.3f}σ" if not math.isnan(s["ev"]) else f"{'n<5':>14}"
        print(f"  {label:30}  {fmt(ga):>14}  {fmt(gb):>14}")

    # ── Sizing: accel vs PL vs combined ──────────────────────────────────────
    print(f"\n{'═'*76}")
    print(f"  SIZING COMPARISON  (EMA({fast},{slow}), accel threshold = median {med_accel:+.2f})")
    print(f"{'─'*76}")

    base_r  = sum(r["outcome_r"] for r in full)
    base_eu = base_r / len(full)

    def sized(fn):
        tot   = sum(2*r["outcome_r"] if fn(r) else r["outcome_r"] for r in full)
        units = sum(2 if fn(r) else 1 for r in full)
        return tot, units

    strategies = [
        ("Baseline (1× all)",               lambda r: False),
        (f"Accel ≥ median → 2×",             lambda r: r[key] >= med_accel),
        (f"PL ≥ +{PL_THRESH:.2f} → 2×",      lambda r: r["pl_aligned"] >= PL_THRESH),
        (f"Accel AND PL → 2×",               lambda r: r[key] >= med_accel and r["pl_aligned"] >= PL_THRESH),
        (f"Accel OR  PL → 2×",               lambda r: r[key] >= med_accel or  r["pl_aligned"] >= PL_THRESH),
    ]

    print(f"\n  {'Strategy':35}  {'n(2×)':>6}  {'TotalR':>8}  "
          f"{'ΔR':>7}  {'EV/unit':>8}  {'ΔEV/u':>8}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")
    for label, fn in strategies:
        n2x       = sum(1 for r in full if fn(r))
        tot, units = sized(fn)
        d_r   = tot - base_r
        eu    = tot / units
        d_eu  = eu - base_eu
        flag  = " ◄" if d_r > 0 and d_eu > 0 else ""
        print(f"  {label:35}  {n2x:>6,}  {tot:>+8.2f}  "
              f"{d_r:>+7.2f}  {eu:>+8.4f}  {d_eu:>+8.4f}{flag}")

    # ── Year-by-year for combined rule ────────────────────────────────────────
    combined_fn = lambda r: r[key] >= med_accel and r["pl_aligned"] >= PL_THRESH
    pl_fn       = lambda r: r["pl_aligned"] >= PL_THRESH
    print(f"\n  Year-by-year  [Accel AND PL → 2×]  vs  [PL only → 2×]  vs  baseline")
    print(f"  {'Year':>5}  {'n':>5}  {'BaseR':>8}  {'PL R':>8}  {'A+PL R':>8}  {'ΔvsP':>7}")
    years = sorted(set(r["year"] for r in full))
    for yr in years:
        yr_r  = [r for r in full if r["year"] == yr]
        if len(yr_r) < 5: continue
        br    = sum(r["outcome_r"] for r in yr_r)
        pl_r  = sum(2*r["outcome_r"] if pl_fn(r) else r["outcome_r"] for r in yr_r)
        co_r  = sum(2*r["outcome_r"] if combined_fn(r) else r["outcome_r"] for r in yr_r)
        flag  = " ◄" if co_r > pl_r else ""
        print(f"  {yr:>5}  {len(yr_r):>5,}  {br:>+8.2f}  {pl_r:>+8.2f}  "
              f"{co_r:>+8.2f}  {co_r-pl_r:>+7.2f}{flag}")
    all_pl = sum(2*r["outcome_r"] if pl_fn(r) else r["outcome_r"] for r in full)
    all_co = sum(2*r["outcome_r"] if combined_fn(r) else r["outcome_r"] for r in full)
    print(f"  {'TOTAL':>5}  {len(full):>5,}  {base_r:>+8.2f}  "
          f"{all_pl:>+8.2f}  {all_co:>+8.2f}  {all_co-all_pl:>+7.2f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=list(INSTRUMENTS.keys()))
    args = parser.parse_args()

    path = INSTRUMENTS[args.sym]
    print(f"Loading {path} …", flush=True)
    df1 = load_1min(path)

    bars5 = make_5min_bars(df1)
    print(f"  {len(bars5):,} 5-min bars  |  {len(df1):,} 1-min bars", flush=True)

    print("Scanning signals …", flush=True)
    records = scan_5min(bars5)
    print(f"  {len(records):,} baseline triggers", flush=True)

    print("Attaching features …", flush=True)
    attach_features(records, df1)

    n_valid = sum(1 for r in records
                  if not math.isnan(r.get("accel_5_20", float("nan"))))
    print(f"  {n_valid:,} triggers with VWAP accel", flush=True)

    analyse(records, args.sym)
    print()


if __name__ == "__main__":
    main()
