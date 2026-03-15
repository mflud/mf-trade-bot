"""
Conditional edge analysis: does the ≥3σ + vol continuation signal work in
specific regimes?

Stratifies MES and MYM trigger results by:
  1. Volatility regime   (trailing annualised vol bucket)
  2. Session             (Overnight / NYSE hours / CME close)
  3. Intraday trend      (Price Linearity of same-day 5-min bars up to trigger)

For each slice reports: n, P(target), P(stop), EV at the practical combo
(-1.5σ stop / +2.5σ target) and at each combo's best.

Usage:
  python src/regime_analysis.py            # MES + MYM
  python src/regime_analysis.py --sym MES  # single instrument
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

# ── Config ─────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TRAILING_BARS        = 20     # 20 × 5-min = 100 min (optimal from signal_window_grid)
MOM_BARS             = 8      # 8 × 5-min = 40 min momentum window (optimal from momentum_filter)
CSR_THRESHOLD        = 1.5    # min cumulative scaled return aligned with signal direction
TF                   = 5
MAX_BARS_HOLD        = 3
MIN_SCALED           = 3.0
MAX_SCALED           = 99.0   # override via --max-scaled; filters extreme event spikes
MIN_VOL_RATIO        = 1.5
BARS_PER_YEAR        = 252 * 23 * 60

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0    # optimal combo at trail=20

# Vol regime filter (kept for stratified analysis; no filter is best at trail=20)
VOL_FILTER_LO = 0.10
VOL_FILTER_HI = 0.20

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

# Vol regime thresholds (annualised)
VOL_REGIME_BINS   = [0, 0.10, 0.15, 0.20, 0.30, 99]
VOL_REGIME_LABELS = ["QUIET (<10%)", "NORMAL (10–15%)", "ELEVATED (15–20%)",
                     "ACTIVE (20–30%)", "HIGH VOL (>30%)"]

# Session in UTC hours (approximate; CME equity index)
#   Overnight : 22:00–13:29 UTC  (Asian + European hours)
#   NYSE open : 13:30–20:00 UTC  (regular NYSE session)
#   CME close : 20:00–21:00 UTC  (post-close, before settlement gap)
SESSION_BINS   = [-1, 13, 20, 25]   # 25 wraps around midnight
SESSION_LABELS = ["Overnight (pre-NYSE)", "NYSE hours (13:30–20 UTC)",
                  "CME close (20–21 UTC)"]

# Price Linearity buckets
PL_BINS   = [0, 0.30, 0.55, 1.01]
PL_LABELS = ["Choppy (PL<0.30)", "Mixed (0.30–0.55)", "Trending (PL>0.55)"]


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


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
    records, i = [], 0
    n = TF
    while i + n <= len(df1):
        chunk = df1.iloc[i: i + n]
        internal = chunk["gap"].iloc[1:].values
        if internal.any():
            i += int(internal.argmax()) + 1
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
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Price Linearity ────────────────────────────────────────────────────────────

def price_linearity(closes: np.ndarray) -> float:
    """PL = |net move| / sum(|bar moves|).  Range [0,1]. 1=trending, 0=choppy."""
    if len(closes) < 2:
        return float("nan")
    rets = np.diff(np.log(closes))
    total = np.sum(np.abs(rets))
    return abs(rets.sum()) / total if total > 0 else float("nan")


# ── Scan with context ──────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, max_scaled: float = MAX_SCALED) -> pd.DataFrame:
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_arr  = bars["ts"].values   # numpy datetime64
    n       = len(bars)
    records = []

    # Pre-compute CME trading date for each bar (date changes at 17:00 CT = 23:00 UTC)
    ts_pd   = pd.DatetimeIndex(ts_arr, tz="UTC")
    # Shift back 23h so the 23:00 UTC open of each session maps to calendar date
    session_date = (ts_pd - pd.Timedelta(hours=23)).date

    for i in range(max(TRAILING_BARS, MOM_BARS), n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                          / closes[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")

        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or abs(scaled) > max_scaled or vol_ratio < MIN_VOL_RATIO:
            continue

        direction   = 1 if scaled > 0 else -1
        entry       = closes[i]
        sigma_pts   = sigma * entry
        ann_vol     = sigma * math.sqrt(BARS_PER_YEAR / TF)

        # 40-min cumulative scaled return (positive = momentum aligned with signal)
        if i >= MOM_BARS and not gaps[i - MOM_BARS: i].any():
            mom_rets = np.log(closes[i - MOM_BARS + 1: i]
                            / closes[i - MOM_BARS:     i - 1])
            csr = float(mom_rets.sum()) / sigma * direction
        else:
            csr = float("nan")

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
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

        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        # ── Context features ──────────────────────────────────────────────────

        # Session hour (UTC)
        bar_hour = ts_pd[i].hour + ts_pd[i].minute / 60

        # Intraday Price Linearity: same CME date, bars up to (not including) trigger
        this_date = session_date[i]
        same_day  = np.where(
            np.array(session_date[:i]) == this_date
        )[0]
        if len(same_day) >= 3:
            pl = price_linearity(closes[same_day])
        else:
            pl = float("nan")

        records.append({
            "year":          ts_pd[i].year,
            "csr_40m":       csr,
            "ann_vol":       ann_vol,
            "sigma_pts":     sigma_pts,
            "scaled":        abs(scaled),
            "vol_ratio":     vol_ratio,
            "bar_hour_utc":  bar_hour,
            "intraday_pl":   pl,
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
    best = -999
    for s in STOPS:
        for t in TARGETS:
            st = ev_stats(sub, s, t)
            if not math.isnan(st["ev"]) and st["ev"] > best:
                best, bs, bt = st["ev"], s, t
    return best, bs, bt


# ── Filtered EV grid ──────────────────────────────────────────────────────────

def filtered_ev_grid(sym: str, res: pd.DataFrame,
                     vol_lo: float, vol_hi: float):
    filt = res[(res["ann_vol"] >= vol_lo) & (res["ann_vol"] < vol_hi)]
    n_all  = len(res)
    n_filt = len(filt)
    pct    = n_filt / n_all * 100

    print(f"\n{'═'*72}")
    print(f"  {sym}  —  VOL FILTER  {vol_lo*100:.0f}%–{vol_hi*100:.0f}%  "
          f"({n_filt:,} of {n_all:,} triggers = {pct:.0f}%)")
    print(f"{'═'*72}")

    if n_filt < 30:
        print("  Insufficient triggers after filtering.")
        return

    # Baseline vs filtered
    base  = ev_stats(res,  PRAC_S, PRAC_T)
    fstat = ev_stats(filt, PRAC_S, PRAC_T)
    bev_base,  bs_base,  bt_base  = best_ev(res)
    bev_filt,  bs_filt,  bt_filt  = best_ev(filt)

    print(f"\n  Practical combo (-{PRAC_S:.1f}σ / +{PRAC_T:.1f}σ):")
    print(f"  {'':30}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
    print(f"  {'─'*62}")
    print(f"  {'Unfiltered (all vol regimes)':<30}  {n_all:>6,}  "
          f"{base['p_tgt']:>8.3f}  {base['p_stop']:>8.3f}  {base['ev']:>+9.4f}σ")
    print(f"  {'Filtered ({:.0f}%–{:.0f}% ann vol)'.format(vol_lo*100, vol_hi*100):<30}  "
          f"{n_filt:>6,}  "
          f"{fstat['p_tgt']:>8.3f}  {fstat['p_stop']:>8.3f}  {fstat['ev']:>+9.4f}σ")

    print(f"\n  Full EV grid (filtered, σ units)  — rows=stop, cols=target:")
    hdr = f"  {'Stop \\ Target':>14}" + "".join(f"  +{t:.1f}σ" for t in TARGETS)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    best_ev_val = -999
    best_s = best_t = None
    for s in STOPS:
        row = f"  -{s:.1f}σ          "
        for t in TARGETS:
            st = ev_stats(filt, s, t)
            row += f"  {st['ev']:>+5.3f}"
            if st["ev"] > best_ev_val:
                best_ev_val, best_s, best_t = st["ev"], s, t
        print(row)

    bst = ev_stats(filt, best_s, best_t)
    print(f"\n  Best:  stop={best_s:.1f}σ  target={best_t:.1f}σ  →  EV={best_ev_val:+.4f}σ")
    print(f"    P(target): {bst['p_tgt']:.3f}   P(stop): {bst['p_stop']:.3f}")

    # Points context
    sp = filt["sigma_pts"].values
    print(f"\n  In points (filtered triggers):")
    print(f"  {'':8}  {'1σ pts':>8}  {'tgt pts':>9}  {'stop pts':>9}  "
          f"{'EV(σ) prac':>11}  {'EV pts prac':>12}")
    for pct_v, label in [(25,"p25"),(50,"median"),(75,"p75"),(90,"p90")]:
        sigma_p = float(np.percentile(sp, pct_v))
        print(f"  {label:<8}  {sigma_p:>8.2f}  "
              f"{PRAC_T*sigma_p:>9.2f}  {PRAC_S*sigma_p:>9.2f}  "
              f"{fstat['ev']:>+11.4f}  {fstat['ev']*sigma_p:>+12.3f}")


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_slice_table(title: str, groups: list[tuple[str, pd.DataFrame]],
                      min_n: int = 30):
    print(f"\n  {title}")
    print(f"  {'─'*80}")
    hdr = (f"  {'Slice':<28}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  "
           f"{'EV prac':>9}  {'BestEV':>8}  {'Best combo'}")
    print(hdr)
    print(f"  {'─'*80}")

    for label, sub in groups:
        if len(sub) < min_n:
            print(f"  {label:<28}  {len(sub):>6}  [insufficient data]")
            continue
        prac = ev_stats(sub, PRAC_S, PRAC_T)
        bev, bs, bt = best_ev(sub)
        flag = "  ◄" if prac["ev"] > 0 else ""
        print(f"  {label:<28}  {prac['n']:>6,}  "
              f"{prac['p_tgt']:>8.3f}  {prac['p_stop']:>8.3f}  "
              f"{prac['ev']:>+9.4f}  {bev:>+8.4f}  "
              f"-{bs:.1f}σ/+{bt:.1f}σ{flag}")


def report(sym: str, res: pd.DataFrame):
    n = len(res)
    print(f"\n{'═'*82}")
    print(f"  {sym}  —  {n:,} triggers  "
          f"(≥{MIN_SCALED:.0f}σ, vol≥{MIN_VOL_RATIO:.1f}×, hold≤{MAX_BARS_HOLD*TF}min)")
    print(f"{'═'*82}")

    # Overall baseline
    prac = ev_stats(res, PRAC_S, PRAC_T)
    bev, bs, bt = best_ev(res)
    print(f"\n  Overall:  EV={prac['ev']:+.4f}σ  "
          f"P(tgt)={prac['p_tgt']:.3f}  P(stop)={prac['p_stop']:.3f}  "
          f"Best={bev:+.4f}σ at -{bs:.1f}σ/+{bt:.1f}σ")

    # ── 0. Year by year ───────────────────────────────────────────────────────
    years = sorted(res["year"].unique())
    year_groups = [(str(y), res[res["year"] == y]) for y in years]
    print_slice_table("0. BY YEAR", year_groups, min_n=20)

    # ── 1. Vol regime ─────────────────────────────────────────────────────────
    res["vol_regime"] = pd.cut(res["ann_vol"], bins=VOL_REGIME_BINS,
                               labels=VOL_REGIME_LABELS)
    groups = [(str(lbl), res[res["vol_regime"] == lbl])
              for lbl in VOL_REGIME_LABELS]
    print_slice_table("1. BY VOLATILITY REGIME", groups)

    # ── 2. Session ────────────────────────────────────────────────────────────
    # Map hour to session label manually (hour wraps: 21–24 = overnight)
    def session(h):
        if h >= 20:
            return SESSION_LABELS[2]    # CME close 20–21
        elif h >= 13.5:
            return SESSION_LABELS[1]    # NYSE hours
        else:
            return SESSION_LABELS[0]    # Overnight

    res["session"] = res["bar_hour_utc"].map(session)
    groups = [(lbl, res[res["session"] == lbl]) for lbl in SESSION_LABELS]
    print_slice_table("2. BY SESSION (UTC)", groups)

    # ── 3. Intraday Price Linearity ───────────────────────────────────────────
    sub_pl = res.dropna(subset=["intraday_pl"])
    sub_pl = sub_pl.copy()
    sub_pl["pl_bucket"] = pd.cut(sub_pl["intraday_pl"], bins=PL_BINS,
                                  labels=PL_LABELS)
    groups = [(str(lbl), sub_pl[sub_pl["pl_bucket"] == lbl])
              for lbl in PL_LABELS]
    print_slice_table("3. BY INTRADAY TREND (Price Linearity up to trigger)",
                      groups)

    # ── 4. 2D: Vol regime × Session (best cells only) ─────────────────────────
    print(f"\n  4. 2D: VOL REGIME × SESSION  (practical EV, n≥30)")
    print(f"  {'─'*80}")
    hdr2 = f"  {'Vol regime':<22}" + "".join(f"  {s[:16]:>20}" for s in SESSION_LABELS)
    print(hdr2)
    print(f"  {'─'*80}")
    for vr in VOL_REGIME_LABELS:
        row = f"  {vr:<22}"
        for sess in SESSION_LABELS:
            cell = res[(res["vol_regime"] == vr) & (res["session"] == sess)]
            if len(cell) < 30:
                row += f"  {'— (n=' + str(len(cell)) + ')':>20}"
            else:
                prac = ev_stats(cell, PRAC_S, PRAC_T)
                flag = "◄" if prac["ev"] > 0 else " "
                row += f"  {prac['ev']:>+8.4f} n={len(cell):<5}{flag:>3}"
        print(row)

    # ── 5. Vol regime × Trend ─────────────────────────────────────────────────
    print(f"\n  5. 2D: VOL REGIME × TREND  (practical EV, n≥30)")
    print(f"  {'─'*80}")
    hdr3 = f"  {'Vol regime':<22}" + "".join(f"  {p[:18]:>22}" for p in PL_LABELS)
    print(hdr3)
    print(f"  {'─'*80}")
    for vr in VOL_REGIME_LABELS:
        row = f"  {vr:<22}"
        for pl in PL_LABELS:
            cell = sub_pl[(sub_pl["vol_regime"] == vr) & (sub_pl["pl_bucket"] == pl)]
            if len(cell) < 30:
                row += f"  {'— (n=' + str(len(cell)) + ')':>22}"
            else:
                prac = ev_stats(cell, PRAC_S, PRAC_T)
                flag = "◄" if prac["ev"] > 0 else " "
                row += f"  {prac['ev']:>+8.4f} n={len(cell):<5}{flag:>3}"
        print(row)

    # ── 6. Momentum filter (CSR 40-min) ──────────────────────────────────────
    valid = res.dropna(subset=["csr_40m"])
    with_mom    = valid[valid["csr_40m"] >  CSR_THRESHOLD]
    against_mom = valid[valid["csr_40m"] < -CSR_THRESHOLD]
    neutral_mom = valid[valid["csr_40m"].abs() <= CSR_THRESHOLD]

    pw = ev_stats(with_mom,    PRAC_S, PRAC_T)
    pa = ev_stats(against_mom, PRAC_S, PRAC_T)
    pn = ev_stats(neutral_mom, PRAC_S, PRAC_T)

    print(f"\n  6. MOMENTUM FILTER  (CSR >{CSR_THRESHOLD:.1f}σ over {MOM_BARS*TF} min)")
    print(f"  {'─'*70}")
    print(f"  {'Slice':<28}  {'n':>6}  {'P(tgt)':>7}  {'P(stop)':>7}  {'EV prac':>10}")
    print(f"  {'─'*70}")
    for label, p in [("WITH momentum",    pw),
                     ("AGAINST momentum", pa),
                     ("NEUTRAL",          pn)]:
        flag = "  ◄" if p["ev"] > 0 else ""
        n_s  = f"{p['n']:>6,}" if p["n"] >= 5 else f"{'[n=' + str(p['n']) + ']':>6}"
        ev_s = f"{p['ev']:>+10.4f}σ" if not math.isnan(p["ev"]) else f"{'—':>11}"
        print(f"  {label:<28}  {n_s}  {p['p_tgt']:>7.3f}  {p['p_stop']:>7.3f}  {ev_s}{flag}")

    # ── 7. Filtered EV grid ───────────────────────────────────────────────────
    filtered_ev_grid(sym, res, VOL_FILTER_LO, VOL_FILTER_HI)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None,
                        help="Single instrument to analyse (MES or MYM)")
    parser.add_argument("--max-scaled", type=float, default=99.0,
                        help="Exclude triggers with |scaled| > this (e.g. 5 to drop event spikes)")
    args = parser.parse_args()
    MAX_SCALED = args.max_scaled  # override module-level default

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, cache in syms.items():
        print(f"\nLoading {cache} …")
        df1  = load_1min(cache)
        bars = make_5min_bars(df1)
        print(f"  {len(bars):,} 5-min bars")

        print("  Scanning triggers …")
        res = scan(bars, max_scaled=args.max_scaled)
        print(f"  {len(res):,} qualifying triggers")

        report(sym, res)
