"""
MNQ confirmation filter backtest.

For each MES / MYM 3σ signal, checks whether MNQ's scaled return on the
same bar is aligned with the signal direction. Tests several alignment
thresholds to find the best trade-off between signal count and EV.

Usage:
  python src/backtest_mnq_confirm.py
  python src/backtest_mnq_confirm.py --sym MES
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config ──────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

TF             = 5
TRAILING_BARS  = 20
MAX_MOM_BARS   = 8          # largest CSR window used
MAX_BARS_HOLD  = 3
MIN_SCALED     = 3.0
MAX_SCALED     = 99.0
MIN_VOL_RATIO  = 1.5
CSR_THRESHOLD  = 1.5
BLACKOUT_ET    = [(8, 0, 9, 0)]   # conditional: only blocks if CSR < threshold
GK_VOL_BARS    = 20
BARS_PER_YEAR  = 252 * 23 * 60 / TF

# Dynamic CSR window: (gk_ann_vol_threshold, csr_bars)
CSR_VOL_WINDOWS = [(0.08, 4), (1.0, 8)]

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}
MNQ_PATH = "mnq_hist_1min.csv"

# MNQ alignment thresholds to test:
#   0.0 = any same-direction MNQ move (even tiny)
#   0.5 = MNQ scaled > 0.5σ in same direction
#   1.0, 1.5, 2.0 = progressively stronger confirmation required
MNQ_THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0]


# ── Data helpers ─────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    return df.sort_values("ts").reset_index(drop=True)


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


def gk_annualised(opens, highs, lows, closes):
    """Compute annualised GK vol over the last GK_VOL_BARS bars (excluding current)."""
    vals = []
    for j in range(len(opens)):
        o, h, l, c = opens[j], highs[j], lows[j], closes[j]
        if o > 0 and h > 0 and l > 0 and c > 0:
            vals.append(0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2)
    return math.sqrt(max(0.0, float(np.mean(vals))) * BARS_PER_YEAR) if vals else 0.0


def get_mom_bars(gk_ann: float) -> int:
    for threshold, bars in CSR_VOL_WINDOWS:
        if gk_ann < threshold:
            return bars
    return CSR_VOL_WINDOWS[-1][1]


def compute_mnq_scaled(bars: pd.DataFrame) -> pd.Series:
    """Compute MNQ scaled returns indexed by ts."""
    closes = bars["close"].values
    gaps   = bars["gap"].values
    n      = len(bars)
    scaled = np.full(n, np.nan)
    for i in range(TRAILING_BARS, n):
        if gaps[i - TRAILING_BARS + 1: i + 1].any():
            continue
        trail = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                     / closes[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail, ddof=1)
        if sigma == 0:
            continue
        scaled[i] = math.log(closes[i] / closes[i - 1]) / sigma
    return pd.Series(scaled, index=bars["ts"])


# ── Scan ─────────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, mnq_scaled: pd.Series) -> pd.DataFrame:
    opens   = bars["open"].values
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)
    records = []

    # Build a lookup for MNQ scaled at each bar timestamp
    mnq_lookup = mnq_scaled.to_dict()

    lookback = max(TRAILING_BARS, MAX_MOM_BARS, GK_VOL_BARS) + 1
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

        direction = 1 if scaled > 0 else -1

        # Dynamic CSR window based on GK vol regime
        gk_ann = gk_annualised(
            opens[i - GK_VOL_BARS: i], highs[i - GK_VOL_BARS: i],
            lows[i - GK_VOL_BARS: i], closes[i - GK_VOL_BARS: i])
        mom_bars = get_mom_bars(gk_ann)

        if i >= mom_bars and not gaps[i - mom_bars: i].any():
            mom_rets = np.log(closes[i - mom_bars + 1: i]
                            / closes[i - mom_bars:     i - 1])
            csr = float(mom_rets.sum()) / sigma * direction
        else:
            csr = float("nan")

        # Conditional blackout: block only if CSR < threshold
        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        in_blackout = any((sh, sm) <= bar_hm < (eh, em) for sh, sm, eh, em in BLACKOUT_ET)
        if in_blackout and (math.isnan(csr) or csr < CSR_THRESHOLD):
            continue

        # MNQ scaled return on same bar (direction-adjusted)
        ts_key = ts_pd[i]
        mnq_sc = mnq_lookup.get(ts_key, float("nan"))
        mnq_aligned = mnq_sc * direction if not math.isnan(mnq_sc) else float("nan")

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
            "year":          ts_pd[i].year,
            "csr":           csr,
            "mnq_aligned":   mnq_aligned,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV helpers ───────────────────────────────────────────────────────────────────

def ev_stats(sub: pd.DataFrame, s: float, t: float) -> dict:
    if len(sub) < 5:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(sub)}
    ht = sub[f"hit_tgt_{t}"].notna().values
    hs = sub[f"hit_stop_{s}"].notna().values
    ht_first = ht & ~(hs & (sub[f"hit_stop_{s}"].fillna(999)
                            <= sub[f"hit_tgt_{t}"].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    p_tgt  = ht_first.mean()
    p_stop = hs_first.mean()
    ev_nei = sub["time_exit_ret"].values[neither].mean() if neither.any() else 0.0
    ev     = p_tgt * t - p_stop * s + neither.mean() * ev_nei
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(sub)}


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None)
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    print("Loading MNQ …")
    mnq_bars = make_5min_bars(load_1min(MNQ_PATH))
    mnq_scaled = compute_mnq_scaled(mnq_bars)
    print(f"  MNQ: {len(mnq_bars):,} bars  "
          f"({mnq_bars['ts'].min().date()} → {mnq_bars['ts'].max().date()})")

    for sym, path in syms.items():
        print(f"\n{'═'*72}")
        print(f"  {sym}  —  MNQ Confirmation Filter Backtest")
        print(f"{'═'*72}")

        bars = make_5min_bars(load_1min(path))
        print(f"  {sym}: {len(bars):,} bars — scanning …", end=" ", flush=True)
        res = scan(bars, mnq_scaled)
        print(f"{len(res):,} raw triggers")

        # Baseline: CSR filter only (no MNQ)
        base = res.dropna(subset=["csr"])
        base = base[base["csr"] > CSR_THRESHOLD]

        print(f"\n  {'Filter':<28}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>7}  "
              f"{'EV':>9}  {'vs baseline'}")
        print(f"  {'─'*68}")

        p0 = ev_stats(base, PRAC_S, PRAC_T)
        print(f"  {'Baseline (CSR≥1.5 only)':<28}  {p0['n']:>5,}  "
              f"{p0['p_tgt']:>7.3f}  {p0['p_stop']:>7.3f}  {p0['ev']:>+9.4f}σ")

        # MNQ thresholds (no CSR required)
        print(f"\n  — MNQ aligned (no CSR filter) —")
        for thr in MNQ_THRESHOLDS:
            sub = res.dropna(subset=["mnq_aligned"])
            sub = sub[sub["mnq_aligned"] > thr]
            p = ev_stats(sub, PRAC_S, PRAC_T)
            delta = p["ev"] - p0["ev"] if not math.isnan(p["ev"]) else float("nan")
            pct   = 100 * p["n"] / p0["n"] if p0["n"] else 0
            print(f"  {'MNQ >' + f'{thr:.1f}σ':<28}  {p['n']:>5,}  "
                  f"{p['p_tgt']:>7.3f}  {p['p_stop']:>7.3f}  {p['ev']:>+9.4f}σ  "
                  f"{delta:>+7.4f}σ  ({pct:.0f}% of signals)")

        # CSR + MNQ combined
        print(f"\n  — CSR≥1.5 AND MNQ aligned —")
        for thr in MNQ_THRESHOLDS:
            sub = res.dropna(subset=["csr", "mnq_aligned"])
            sub = sub[(sub["csr"] > CSR_THRESHOLD) & (sub["mnq_aligned"] > thr)]
            p = ev_stats(sub, PRAC_S, PRAC_T)
            delta = p["ev"] - p0["ev"] if not math.isnan(p["ev"]) else float("nan")
            pct   = 100 * p["n"] / p0["n"] if p0["n"] else 0
            print(f"  {'CSR≥1.5 + MNQ >' + f'{thr:.1f}σ':<28}  {p['n']:>5,}  "
                  f"{p['p_tgt']:>7.3f}  {p['p_stop']:>7.3f}  {p['ev']:>+9.4f}σ  "
                  f"{delta:>+7.4f}σ  ({pct:.0f}% of signals)")

        # Year-by-year for best combined filter
        print(f"\n  — BY YEAR: Baseline vs CSR≥1.5+MNQ>0.0σ vs CSR≥1.5+MNQ>1.0σ —")
        print(f"  {'Year':<6}  {'n(base)':>7}  {'EV(base)':>10}  "
              f"{'n(MNQ>0)':>8}  {'EV(MNQ>0)':>10}  "
              f"{'n(MNQ>1)':>8}  {'EV(MNQ>1)':>10}")
        print(f"  {'─'*72}")
        for yr in sorted(res["year"].unique()):
            ry = res[res["year"] == yr]

            b  = ry.dropna(subset=["csr"])
            b  = b[b["csr"] > CSR_THRESHOLD]
            pb = ev_stats(b, PRAC_S, PRAC_T)

            m0 = ry.dropna(subset=["csr","mnq_aligned"])
            m0 = m0[(m0["csr"] > CSR_THRESHOLD) & (m0["mnq_aligned"] > 0.0)]
            pm0 = ev_stats(m0, PRAC_S, PRAC_T)

            m1 = ry.dropna(subset=["csr","mnq_aligned"])
            m1 = m1[(m1["csr"] > CSR_THRESHOLD) & (m1["mnq_aligned"] > 1.0)]
            pm1 = ev_stats(m1, PRAC_S, PRAC_T)

            def fmt(p):
                if math.isnan(p["ev"]): return f"{'—':>10}"
                return f"{p['ev']:>+10.4f}σ"

            print(f"  {yr:<6}  {pb['n']:>7,}  {fmt(pb)}  "
                  f"{pm0['n']:>8,}  {fmt(pm0)}  "
                  f"{pm1['n']:>8,}  {fmt(pm1)}")
