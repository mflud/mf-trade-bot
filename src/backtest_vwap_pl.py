"""
Backtest: PL_aligned × Session VWAP position as combined sizing filter.

Builds on backtest_pl_sizing.py. For each 3σ signal also records:
  - vwap_aligned: whether signal direction matches session VWAP position
      LONG  signal → vwap_aligned iff close > session VWAP
      SHORT signal → vwap_aligned iff close < session VWAP
  - vwap_dist_sig: (close - session_vwap) * direction / sigma_pts
      positive = aligned with VWAP, negative = fighting VWAP

Session VWAP resets at 9:30 ET each day. Signals before 9:30 ET are
excluded from VWAP analysis (pre-market, no NYSE VWAP reference yet).

Analyses:
  1. 2×2 matrix — EV by (PL high/low) × (VWAP aligned/not)
  2. Four sizing rules compared:
       baseline  : 1× all signals
       PL only   : 2× when PL_aligned ≥ PL_THRESH
       VWAP only : 2× when vwap_aligned
       PL + VWAP : 2× when PL_aligned ≥ PL_THRESH AND vwap_aligned
  3. Year-by-year for the combined rule

Usage:
  python src/backtest_vwap_pl.py
  python src/backtest_vwap_pl.py --sym MYM
"""

import argparse
import math
import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Parameters (keep in sync with production / backtest_pl_sizing.py) ─────────

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

N               = 10          # 1-min PL lookback bars
STRONG_THRESH   = 0.5
SIGMA_1M_WIN    = 100
PL_THRESH       = 0.50        # PL_aligned cutoff for "high confidence"

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data loading ───────────────────────────────────────────────────────────────

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


def add_session_vwap(df1):
    """
    Add session_vwap column: cumulative typical-price×volume VWAP resetting
    at 09:30 ET each day. NaN for bars before 09:30 ET.
    """
    df = df1.copy()
    ts_et      = df["ts"].dt.tz_convert(ET)
    df["date"] = ts_et.dt.date
    df["hm"]   = ts_et.dt.hour * 60 + ts_et.dt.minute
    typical    = (df["high"] + df["low"] + df["close"]) / 3
    df["tpv"]  = typical * df["volume"]

    df["session_vwap"] = float("nan")

    # Process each day independently
    for date, grp in df.groupby("date"):
        rth = grp[grp["hm"] >= 9 * 60 + 30]
        if rth.empty:
            continue
        cum_tpv = rth["tpv"].cumsum()
        cum_vol = rth["volume"].cumsum()
        with np.errstate(invalid="ignore"):
            vwap = cum_tpv / cum_vol.replace(0, float("nan"))
        df.loc[rth.index, "session_vwap"] = vwap.values

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


# ── Signal scan (identical to backtest_pl_sizing.py) ─────────────────────────

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
                    if h >= tgt_px:   hit_tgt  = True
                    elif l <= stop_px: hit_stop = True
                else:
                    if l <= tgt_px:   hit_tgt  = True
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
            "ts":           ts_pd[i],
            "year":         ts_pd[i].year,
            "direction":    direction,
            "sigma":        sigma,
            "sigma_pts":    sigma * closes[i],
            "close":        closes[i],
            "hit_tgt":      hit_tgt,
            "hit_stop":     hit_stop,
            "time_exit_r":  time_exit_r,
            "outcome_r":    outcome_r,
        })
    return records


# ── Feature attachment ────────────────────────────────────────────────────────

def attach_features(records, df1):
    """Attach pl_aligned and vwap_aligned to each signal record."""
    log_ret    = df1["log_ret"].values
    sigma_1m   = df1["sigma_1m"].values
    closes_1m  = df1["close"].values
    vwap_vals  = df1["session_vwap"].values
    ts_ns      = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts_map     = {v: i for i, v in enumerate(ts_ns)}

    for rec in records:
        sig_ts_ns = int(rec["ts"].as_unit("ns").value)
        pre_ts_ns = sig_ts_ns - TF * 60 * 10**9   # 1-min bar just before signal

        end_idx = ts_map.get(pre_ts_ns)

        # ── PL_aligned ───────────────────────────────────────────────────────
        if end_idx is None or end_idx - N + 1 < SIGMA_1M_WIN:
            rec["pl_aligned"] = float("nan")
        else:
            rets     = log_ret[end_idx - N + 1: end_idx + 1]
            sig_pt   = sigma_1m[end_idx]
            if np.any(np.isnan(rets)) or math.isnan(sig_pt) or sig_pt <= 0:
                rec["pl_aligned"] = float("nan")
            else:
                sum_absr = float(np.abs(rets).sum())
                rec["pl_aligned"] = (float(rets.sum()) / sum_absr * rec["direction"]
                                     if sum_absr > 0 else 0.0)

        # ── VWAP_aligned (session VWAP from 9:30 ET) ─────────────────────────
        # Look up VWAP at the signal bar's own 1-min timestamp
        sig_1m_idx = ts_map.get(sig_ts_ns)
        if sig_1m_idx is None:
            rec["vwap_aligned"]  = float("nan")
            rec["vwap_dist_sig"] = float("nan")
        else:
            vwap = vwap_vals[sig_1m_idx]
            if math.isnan(vwap):
                rec["vwap_aligned"]  = float("nan")   # pre-market, no VWAP yet
                rec["vwap_dist_sig"] = float("nan")
            else:
                close = closes_1m[sig_1m_idx]
                dist  = (close - vwap) * rec["direction"]   # + = aligned
                rec["vwap_aligned"]  = 1 if dist > 0 else 0
                rec["vwap_dist_sig"] = dist / rec["sigma_pts"] if rec["sigma_pts"] > 0 else 0.0


# ── EV helpers ────────────────────────────────────────────────────────────────

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


def sized_total(recs, flag_fn):
    """Total R under 2×/1× sizing where flag_fn(r) → True means 2×."""
    return sum(2 * r["outcome_r"] if flag_fn(r) else r["outcome_r"] for r in recs)


def sized_units(recs, flag_fn):
    return sum(2 if flag_fn(r) else 1 for r in recs)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(records, sym):
    # Filter to records with both features computed
    full = [r for r in records
            if not math.isnan(r.get("pl_aligned", float("nan")))
            and not math.isnan(r.get("vwap_aligned", float("nan")))]

    base_all  = ev_stats(records)
    base_full = ev_stats(full)

    print(f"\n{'═'*76}")
    print(f"  {sym}  PL_aligned × Session VWAP  —  combined sizing analysis")
    print(f"{'─'*76}")
    print(f"  All triggers:            n={base_all['n']:,}   EV={base_all['ev']:+.4f}σ  "
          f"total={base_all['total_r']:+.2f}R")
    print(f"  With both features:      n={len(full):,}   EV={base_full['ev']:+.4f}σ  "
          f"total={base_full['total_r']:+.2f}R")

    pl_high = [r for r in full if r["pl_aligned"] >= PL_THRESH]
    pl_low  = [r for r in full if r["pl_aligned"] <  PL_THRESH]
    vw_yes  = [r for r in full if r["vwap_aligned"] == 1]
    vw_no   = [r for r in full if r["vwap_aligned"] == 0]

    print(f"\n  Feature split:")
    print(f"    PL_aligned ≥ {PL_THRESH:.2f}:   n={len(pl_high):,}  ({len(pl_high)/len(full)*100:.0f}%)")
    print(f"    PL_aligned <  {PL_THRESH:.2f}:   n={len(pl_low):,}  ({len(pl_low)/len(full)*100:.0f}%)")
    print(f"    VWAP aligned:         n={len(vw_yes):,}  ({len(vw_yes)/len(full)*100:.0f}%)")
    print(f"    VWAP not aligned:     n={len(vw_no):,}  ({len(vw_no)/len(full)*100:.0f}%)")

    # ── 2×2 matrix ───────────────────────────────────────────────────────────
    hh = [r for r in full if r["pl_aligned"] >= PL_THRESH and r["vwap_aligned"] == 1]
    hl = [r for r in full if r["pl_aligned"] >= PL_THRESH and r["vwap_aligned"] == 0]
    lh = [r for r in full if r["pl_aligned"] <  PL_THRESH and r["vwap_aligned"] == 1]
    ll = [r for r in full if r["pl_aligned"] <  PL_THRESH and r["vwap_aligned"] == 0]

    print(f"\n  {'─'*76}")
    print(f"  2×2 MATRIX  (PL high/low × VWAP aligned/not)")
    print(f"  {'─'*76}")
    print(f"  {'':30}  {'VWAP aligned':>16}  {'VWAP NOT aligned':>16}")
    print(f"  {'':30}  {'n / P(tgt) / EV':>16}  {'n / P(tgt) / EV':>16}")
    print(f"  {'─'*30}  {'─'*16}  {'─'*16}")

    for label, grp_yes, grp_no in [
        (f"PL ≥ +{PL_THRESH:.2f}  (high conf)", hh, hl),
        (f"PL <  +{PL_THRESH:.2f}  (low conf)",  lh, ll),
    ]:
        sy = ev_stats(grp_yes)
        sn = ev_stats(grp_no)
        def fmt(s):
            if math.isnan(s["ev"]):
                return f"{'n<5':>16}"
            return f"{s['n']:>4}  {s['p_tgt']:>5.1%}  {s['ev']:>+5.3f}σ"
        print(f"  {label:30}  {fmt(sy):>16}  {fmt(sn):>16}")

    # ── Four sizing strategies ────────────────────────────────────────────────
    print(f"\n  {'─'*76}")
    print(f"  SIZING COMPARISON  (base = {len(full)} signals × 1 unit = {base_full['total_r']:+.2f}R)")
    print(f"  {'─'*76}")
    print(f"\n  {'Strategy':35}  {'n(2×)':>6}  {'Units':>6}  "
          f"{'TotalR':>8}  {'ΔR':>7}  {'EV/unit':>8}  {'ΔEV/u':>8}")
    print(f"  {'-'*35}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")

    base_r  = base_full["total_r"]
    base_eu = base_r / len(full)

    strategies = [
        ("Baseline (1× all)",
         lambda r: False),
        (f"PL ≥ +{PL_THRESH:.2f}  only  → 2×",
         lambda r: r["pl_aligned"] >= PL_THRESH),
        ("VWAP aligned  only  → 2×",
         lambda r: r["vwap_aligned"] == 1),
        (f"PL ≥ +{PL_THRESH:.2f}  AND VWAP → 2×",
         lambda r: r["pl_aligned"] >= PL_THRESH and r["vwap_aligned"] == 1),
        (f"PL ≥ +{PL_THRESH:.2f}  OR  VWAP → 2×",
         lambda r: r["pl_aligned"] >= PL_THRESH or  r["vwap_aligned"] == 1),
    ]

    for label, fn in strategies:
        n2x   = sum(1 for r in full if fn(r))
        units = sized_units(full, fn)
        tot   = sized_total(full, fn)
        d_r   = tot - base_r
        eu    = tot / units
        d_eu  = eu - base_eu
        flag  = " ◄" if d_r > 0 and d_eu > 0 else ""
        print(f"  {label:35}  {n2x:>6,}  {units:>6,}  "
              f"{tot:>+8.2f}  {d_r:>+7.2f}  {eu:>+8.4f}  {d_eu:>+8.4f}{flag}")

    # ── Year-by-year for the combined rule ────────────────────────────────────
    combined_fn = lambda r: r["pl_aligned"] >= PL_THRESH and r["vwap_aligned"] == 1
    pl_only_fn  = lambda r: r["pl_aligned"] >= PL_THRESH

    print(f"\n  {'─'*76}")
    print(f"  YEAR-BY-YEAR  —  PL + VWAP combined  vs  PL only  vs  baseline")
    print(f"  {'─'*76}")
    print(f"  {'Year':>5}  {'n':>5}  {'Base R':>8}  {'PL R':>8}  {'PL+VW R':>9}  "
          f"{'ΔvsPL':>7}  {'P(tgt)':>7}  {'P(stp)':>7}")

    years = sorted(set(r["year"] for r in full))
    for yr in years:
        yr_recs = [r for r in full if r["year"] == yr]
        if len(yr_recs) < 5:
            continue
        base_yr  = ev_stats(yr_recs)
        pl_r     = sized_total(yr_recs, pl_only_fn)
        comb_r   = sized_total(yr_recs, combined_fn)
        delta    = comb_r - pl_r
        flag     = " ◄" if comb_r > pl_r else ""
        print(f"  {yr:>5}  {len(yr_recs):>5,}  "
              f"{base_yr['total_r']:>+8.2f}  {pl_r:>+8.2f}  {comb_r:>+9.2f}  "
              f"{delta:>+7.2f}  {base_yr['p_tgt']:>7.1%}  {base_yr['p_stop']:>7.1%}{flag}")

    all_pl_r   = sized_total(full, pl_only_fn)
    all_comb_r = sized_total(full, combined_fn)
    print(f"  {'TOTAL':>5}  {len(full):>5,}  "
          f"{base_r:>+8.2f}  {all_pl_r:>+8.2f}  {all_comb_r:>+9.2f}  "
          f"{all_comb_r - all_pl_r:>+7.2f}")

    # ── VWAP distance as graded signal ────────────────────────────────────────
    print(f"\n  {'─'*76}")
    print(f"  VWAP DISTANCE (in σ) → EV  (all signals, binned by vwap_dist_sig)")
    print(f"  {'─'*76}")
    dists = np.array([r["vwap_dist_sig"] for r in full])
    edges = [-999, -3, -2, -1, 0, 1, 2, 3, 999]
    labels = ["<-3σ", "-3–-2σ", "-2–-1σ", "-1–0σ", "0–+1σ", "+1–+2σ", "+2–+3σ", ">+3σ"]
    print(f"  {'Dist from VWAP':>14}  {'n':>5}  {'P(tgt)':>7}  {'P(stp)':>7}  {'EV':>9}")
    for lbl, lo, hi in zip(labels, edges, edges[1:]):
        grp = [r for r, d in zip(full, dists) if lo < d <= hi]
        s   = ev_stats(grp)
        if math.isnan(s["ev"]):
            continue
        flag = " ◄" if s["ev"] > base_full["ev"] + 0.05 else ""
        print(f"  {lbl:>14}  {s['n']:>5,}  {s['p_tgt']:>7.1%}  "
              f"{s['p_stop']:>7.1%}  {s['ev']:>+9.4f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=list(INSTRUMENTS.keys()))
    args = parser.parse_args()

    path = INSTRUMENTS[args.sym]
    print(f"Loading {path} …", flush=True)
    df1 = load_1min(path)
    print(f"  Adding session VWAP …", flush=True)
    df1 = add_session_vwap(df1)

    bars5 = make_5min_bars(df1)
    print(f"  {len(bars5):,} 5-min bars", flush=True)

    print("Scanning signals …", flush=True)
    records = scan_5min(bars5)
    print(f"  {len(records):,} baseline triggers", flush=True)

    print("Attaching PL + VWAP features …", flush=True)
    attach_features(records, df1)
    full = [r for r in records
            if not math.isnan(r.get("pl_aligned", float("nan")))
            and not math.isnan(r.get("vwap_aligned", float("nan")))]
    print(f"  {len(full):,} triggers with both features", flush=True)

    analyse(records, args.sym)
    print()


if __name__ == "__main__":
    main()
