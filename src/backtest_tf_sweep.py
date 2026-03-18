"""
Timeframe sweep: compare 3-min, 4-min, and 5-min bars.

Option A (--option A): same candle counts across TFs (TRAILING_BARS=20, MOM_BARS=8)
Option B (--option B): same time windows across TFs (100-min σ, 40-min CSR, 15-min hold)
  3-min: 33 bars σ, 13 bars CSR, 5 bars hold
  4-min: 25 bars σ, 10 bars CSR, 4 bars hold
  5-min: 20 bars σ,  8 bars CSR, 3 bars hold  (baseline)

Reads the same 1-min CSV cache as regime_analysis.py and resamples.

Usage:
  python src/backtest_tf_sweep.py --sym MES           # Option A (default)
  python src/backtest_tf_sweep.py --sym MES --option B
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config ─────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

MIN_SCALED     = 3.0
MAX_SCALED     = 99.0
MIN_VOL_RATIO  = 1.5
CSR_THRESHOLD  = 1.5

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0

TIMEFRAMES = [2, 3, 4, 5, 6, 10]   # minutes

# Option A: same candle counts for all TFs
PARAMS_A = {
     2: {"trailing": 20, "mom": 8, "hold": 3},
     3: {"trailing": 20, "mom": 8, "hold": 3},
     4: {"trailing": 20, "mom": 8, "hold": 3},
     5: {"trailing": 20, "mom": 8, "hold": 3},
     6: {"trailing": 20, "mom": 8, "hold": 3},
    10: {"trailing": 20, "mom": 8, "hold": 3},
}

# Option B: same time windows (~100-min σ, ~40-min CSR, ~15-min hold)
PARAMS_B = {
     2: {"trailing": 50, "mom": 20, "hold": 8},
     3: {"trailing": 33, "mom": 13, "hold": 5},
     4: {"trailing": 25, "mom": 10, "hold": 4},
     5: {"trailing": 20, "mom":  8, "hold": 3},
     6: {"trailing": 17, "mom":  7, "hold": 3},
    10: {"trailing": 10, "mom":  4, "hold": 2},
}

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

BLACKOUT_ET = [(8, 0, 9, 0)]   # 08:00–09:00 ET (DST-aware)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_nmin_bars(df1: pd.DataFrame, n: int) -> pd.DataFrame:
    """Aggregate 1-min bars into n-min bars, skipping across session gaps."""
    records, i = [], 0
    while i + n <= len(df1):
        chunk = df1.iloc[i: i + n]
        if chunk["gap"].iloc[1:].any():
            # Skip to just after the gap
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            continue
        records.append({
            "ts":     chunk["ts"].iloc[0],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += n

    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=n)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Scan ───────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, tf: int, params: dict) -> pd.DataFrame:
    trailing_bars = params["trailing"]
    mom_bars      = params["mom"]
    max_hold      = params["hold"]

    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)

    bars_per_year = 252 * 23 * 60 / tf
    records = []

    for i in range(max(trailing_bars, mom_bars), n - max_hold):
        if gaps[i - trailing_bars + 1: i + max_hold + 1].any():
            continue

        trail_rets = np.log(closes[i - trailing_bars + 1: i + 1]
                          / closes[i - trailing_bars:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - trailing_bars: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")

        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or abs(scaled) > MAX_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1
        entry     = closes[i]
        sigma_pts = sigma * entry
        ann_vol   = sigma * math.sqrt(bars_per_year)

        # Blackout check (ET local time — tracks DST correctly)
        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        if any((sh, sm) <= bar_hm < (eh, em) for sh, sm, eh, em in BLACKOUT_ET):
            continue

        # CSR: prior mom_bars bars only
        if i >= mom_bars and not gaps[i - mom_bars: i].any():
            mom_rets = np.log(closes[i - mom_bars + 1: i]
                            / closes[i - mom_bars:     i - 1])
            csr = float(mom_rets.sum()) / sigma * direction
        else:
            csr = float("nan")

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + max_hold + 1):
            h, l = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if direction == 1 and h >= tgt_prices[t]:
                        hit_tgt[t] = j - i
                    elif direction == -1 and l <= tgt_prices[t]:
                        hit_tgt[t] = j - i
            for s in STOPS:
                if hit_stop[s] is None:
                    if direction == 1 and l <= stop_prices[s]:
                        hit_stop[s] = j - i
                    elif direction == -1 and h >= stop_prices[s]:
                        hit_stop[s] = j - i

        time_exit_ret = math.log(closes[i + max_hold] / entry) * direction / sigma

        records.append({
            "year":          ts_pd[i].year,
            "csr":           csr,
            "ann_vol":       ann_vol,
            "sigma_pts":     sigma_pts,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV helpers ─────────────────────────────────────────────────────────────────

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
    time_ret = sub["time_exit_ret"].values
    p_tgt    = ht_first.mean()
    p_stop   = hs_first.mean()
    ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
    ev       = p_tgt * t - p_stop * s + neither.mean() * ev_nei
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(sub)}


def best_ev(sub: pd.DataFrame) -> tuple[float, float, float]:
    best = -999.0
    bs = bt = PRAC_S, PRAC_T
    for s in STOPS:
        for t in TARGETS:
            st = ev_stats(sub, s, t)
            if not math.isnan(st["ev"]) and st["ev"] > best:
                best, bs, bt = st["ev"], s, t
    return best, bs, bt


# ── Reporting ──────────────────────────────────────────────────────────────────

def report_tf(sym: str, tf: int, res: pd.DataFrame, params: dict):
    n = len(res)
    hold_min      = params["hold"]     * tf
    sigma_win_min = params["trailing"] * tf
    csr_win_min   = params["mom"]      * tf

    print(f"\n{'═'*72}")
    print(f"  {sym}  {tf}-MIN BARS  |  σ window: {params['trailing']} bars ({sigma_win_min} min)  "
          f"|  CSR window: {params['mom']} bars ({csr_win_min} min)")
    print(f"  {n:,} triggers  (≥{MIN_SCALED:.0f}σ, vol≥{MIN_VOL_RATIO:.1f}×, "
          f"hold≤{hold_min} min, CSR all)")
    print(f"{'═'*72}")

    # Overall
    prac = ev_stats(res, PRAC_S, PRAC_T)
    bev, bs, bt = best_ev(res)
    print(f"\n  Overall (no CSR filter):")
    print(f"    EV={prac['ev']:+.4f}σ  P(tgt)={prac['p_tgt']:.3f}  "
          f"P(stop)={prac['p_stop']:.3f}  n={n:,}")
    print(f"    Best: -{bs:.1f}σ/+{bt:.1f}σ  EV={bev:+.4f}σ")

    # Year by year
    years = sorted(res["year"].unique())
    print(f"\n  By year  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ):")
    print(f"  {'Year':<6}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
    print(f"  {'─'*44}")
    for y in years:
        sub = res[res["year"] == y]
        p = ev_stats(sub, PRAC_S, PRAC_T)
        flag = "◄" if p["ev"] > 0 else ""
        print(f"  {y:<6}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
              f"{p['p_stop']:>8.3f}  {p['ev']:>+9.4f}σ  {flag}")

    # Momentum filter
    valid = res.dropna(subset=["csr"])
    with_mom    = valid[valid["csr"] >  CSR_THRESHOLD]
    against_mom = valid[valid["csr"] < -CSR_THRESHOLD]
    neutral_mom = valid[valid["csr"].abs() <= CSR_THRESHOLD]

    print(f"\n  Momentum filter (CSR >{CSR_THRESHOLD:.1f}σ, {csr_win_min}-min / {params['mom']}-bar window):")
    print(f"  {'Slice':<22}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
    print(f"  {'─'*56}")
    for label, sub in [("WITH momentum",    with_mom),
                       ("AGAINST momentum", against_mom),
                       ("NEUTRAL",          neutral_mom)]:
        p = ev_stats(sub, PRAC_S, PRAC_T)
        flag = "◄" if p["ev"] > 0 else ""
        ev_s = f"{p['ev']:>+9.4f}σ" if not math.isnan(p["ev"]) else f"{'—':>10}"
        print(f"  {label:<22}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
              f"{p['p_stop']:>8.3f}  {ev_s}  {flag}")

    # σ_pts distribution
    sp = res["sigma_pts"].values
    print(f"\n  σ_pts distribution (1σ = points to target/stop):")
    print(f"  {'':8}  {'1σ pts':>8}  {'tgt ({:.0f}σ) pts'.format(PRAC_T):>12}  "
          f"{'stop ({:.0f}σ) pts'.format(PRAC_S):>13}")
    for pct_v, label in [(25, "p25"), (50, "median"), (75, "p75"), (90, "p90")]:
        sp_v = float(np.percentile(sp, pct_v))
        print(f"  {label:<8}  {sp_v:>8.2f}  {PRAC_T*sp_v:>12.2f}  {PRAC_S*sp_v:>13.2f}")


def summary_table(sym: str, results: dict[int, pd.DataFrame], param_set: dict):
    """One-line comparison across TFs."""
    print(f"\n{'═'*72}")
    print(f"  {sym}  —  SUMMARY COMPARISON  (practical: -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ, "
          f"CSR≥{CSR_THRESHOLD:.1f} filter applied)")
    print(f"{'═'*72}")
    print(f"  {'TF':>4}  {'σ win':>6}  {'CSR win':>7}  {'hold':>5}  "
          f"{'n (all)':>8}  {'n (CSR+)':>9}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}  {'BestEV':>8}")
    print(f"  {'─'*82}")

    for tf, res in sorted(results.items()):
        p_tf = param_set[tf]
        valid = res.dropna(subset=["csr"])
        with_mom = valid[valid["csr"] > CSR_THRESHOLD]
        p = ev_stats(with_mom, PRAC_S, PRAC_T)
        bev, _, _ = best_ev(with_mom)
        ev_s   = f"{p['ev']:>+9.4f}σ" if not math.isnan(p["ev"]) else f"{'—':>10}"
        bev_s  = f"{bev:>+8.4f}σ" if not math.isnan(bev) else f"{'—':>9}"
        print(f"  {tf:>3}m  {p_tf['trailing']*tf:>5}m  {p_tf['mom']*tf:>6}m  "
              f"{p_tf['hold']*tf:>4}m  {len(res):>8,}  {len(with_mom):>9,}  "
              f"{p['p_tgt']:>8.3f}  {p['p_stop']:>8.3f}  {ev_s}  {bev_s}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None, help="MES or MYM (default: both)")
    parser.add_argument("--option", default="A", choices=["A", "B"],
                        help="A = same candle counts; B = same time windows (default: A)")
    args = parser.parse_args()

    param_set = PARAMS_A if args.option == "A" else PARAMS_B
    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    print(f"\nOption {args.option}: "
          + ("same candle counts (trailing=20, mom=8)" if args.option == "A"
             else "same time windows (~100-min σ, ~40-min CSR, ~15-min hold)"))

    for sym, cache in syms.items():
        print(f"\nLoading {cache} …")
        df1 = load_1min(cache)
        print(f"  {len(df1):,} 1-min bars  ({df1['ts'].min().date()} → {df1['ts'].max().date()})")

        results: dict[int, pd.DataFrame] = {}
        for tf in TIMEFRAMES:
            bars = make_nmin_bars(df1, tf)
            print(f"  {tf}-min: {len(bars):,} bars — scanning …", end=" ", flush=True)
            res = scan(bars, tf, param_set[tf])
            print(f"{len(res):,} triggers")
            results[tf] = res

        summary_table(sym, results, param_set)

        for tf in TIMEFRAMES:
            report_tf(sym, tf, results[tf], param_set[tf])
