"""
Dynamic CSR window backtest.

For each signal trigger, records the raw prior returns so we can compute
CSR at any window length post-hoc.  Then sweeps CSR windows (2–24 bars =
10–120 min) and breaks results down by trailing volatility regime.

Hypothesis: optimal CSR lookback may be shorter in high-vol (fast momentum)
and longer in low-vol (slow, persistent trends).

Usage:
  python src/backtest_csr_sweep.py
  python src/backtest_csr_sweep.py --sym MES
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
MAX_BARS_HOLD  = 3
MIN_SCALED     = 3.0
MAX_SCALED     = 99.0
MIN_VOL_RATIO  = 1.5
CSR_THRESHOLD  = 1.5
BLACKOUT_ET    = [(8, 0, 9, 0)]

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0

# CSR windows to test (in bars; 1 bar = 5 min)
CSR_WINDOWS = [2, 4, 6, 8, 10, 12, 16, 20, 24]

# GK vol regimes (annualised)
VOL_REGIMES = [
    ("Low   (<8%)",   0.00, 0.08),
    ("Normal(8-15%)", 0.08, 0.15),
    ("Active(15-25%)",0.15, 0.25),
    ("High  (>25%)",  0.25, 9.99),
]
GK_VOL_BARS = 20

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


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


def gk_vol(bars_slice) -> float:
    """Annualised Garman-Klass vol for a slice of bar dicts/rows."""
    bars_per_year = 252 * 23 * 60 / TF
    vals = []
    for b in bars_slice:
        o, h, l, c = b["open"], b["high"], b["low"], b["close"]
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            continue
        hl = math.log(h / l) ** 2
        co = math.log(c / o) ** 2
        vals.append(0.5 * hl - (2 * math.log(2) - 1) * co)
    if not vals:
        return 0.0
    return math.sqrt(max(0.0, np.mean(vals)) * bars_per_year)


# ── Scan ─────────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame) -> pd.DataFrame:
    """
    For each trigger bar, records:
      - outcome columns (hit_tgt_*, hit_stop_*, time_exit_ret)
      - sigma, direction
      - trailing log-returns for the last MAX_CSR bars (to compute any CSR window)
      - gk_vol (annualised, for regime bucketing)
    """
    MAX_CSR = max(CSR_WINDOWS)

    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    opens   = bars["open"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)
    records = []

    lookback = max(TRAILING_BARS, MAX_CSR) + 1

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

        # Pre-signal returns for CSR computation at any window
        # We need MAX_CSR prior *completed* bars (excluding bar i itself)
        prior_rets = np.log(closes[i - MAX_CSR: i]
                          / closes[i - MAX_CSR - 1: i - 1])

        # GK vol over last GK_VOL_BARS bars
        gk = gk_vol([{"open": opens[j], "high": highs[j],
                       "low": lows[j],  "close": closes[j]}
                     for j in range(i - GK_VOL_BARS, i)])

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

        rec = {
            "year":          ts_pd[i].year,
            "sigma":         sigma,
            "gk_vol":        gk,
            "direction":     direction,
            "prior_rets":    prior_rets,   # array of MAX_CSR returns
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        }
        records.append(rec)

    return records   # list of dicts (prior_rets is ndarray, can't go in DataFrame directly)


# ── EV helpers ───────────────────────────────────────────────────────────────────

def ev_stats(sub, s: float, t: float) -> dict:
    if len(sub) < 5:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(sub)}
    ht = np.array([r[f"hit_tgt_{t}"] is not None for r in sub])
    hs = np.array([r[f"hit_stop_{s}"] is not None for r in sub])

    def first(r):
        tv = r[f"hit_tgt_{t}"]  if r[f"hit_tgt_{t}"]  is not None else 999
        sv = r[f"hit_stop_{s}"] if r[f"hit_stop_{s}"] is not None else 999
        return tv, sv

    ht_first = np.array([first(r)[0] <= first(r)[1] and r[f"hit_tgt_{t}"] is not None
                         for r in sub])
    hs_first = np.array([first(r)[1] <  first(r)[0] and r[f"hit_stop_{s}"] is not None
                         for r in sub])
    neither  = ~ht_first & ~hs_first
    time_rets = np.array([r["time_exit_ret"] for r in sub])
    p_tgt  = ht_first.mean()
    p_stop = hs_first.mean()
    ev_nei = time_rets[neither].mean() if neither.any() else 0.0
    ev     = p_tgt * t - p_stop * s + neither.mean() * ev_nei
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(sub)}


def csr_for(rec: dict, window: int) -> float:
    """Direction-adjusted CSR for a given window (prior bars, excl. signal bar)."""
    rets = rec["prior_rets"][-window:]
    return float(rets.sum()) / rec["sigma"] * rec["direction"]


# ── Main ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None)
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, path in syms.items():
        print(f"\n{'═'*76}")
        print(f"  {sym}  —  Dynamic CSR Window Sweep")
        print(f"{'═'*76}")

        bars = make_5min_bars(load_1min(path))
        print(f"  {len(bars):,} 5-min bars — scanning …", end=" ", flush=True)
        records = scan(bars)
        print(f"{len(records):,} raw triggers (≥3σ, vol≥1.5×)")

        # ── 1. Overall EV by CSR window ──────────────────────────────────────────
        print(f"\n  OVERALL — EV by CSR window (threshold ≥{CSR_THRESHOLD:.1f}σ)")
        print(f"  {'Window':>8}  {'mins':>5}  {'n(all)':>7}  {'n(CSR+)':>7}  "
              f"{'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}  {'vs 8-bar'}")
        print(f"  {'─'*70}")

        baseline_ev = None
        for w in CSR_WINDOWS:
            filtered = [r for r in records
                        if csr_for(r, w) >= CSR_THRESHOLD]
            p = ev_stats(filtered, PRAC_S, PRAC_T)
            if w == 8:
                baseline_ev = p["ev"]
            delta = (p["ev"] - baseline_ev) if baseline_ev is not None and not math.isnan(p["ev"]) else float("nan")
            flag  = " ◄ baseline" if w == 8 else (f" {delta:>+.4f}σ" if not math.isnan(delta) else "")
            print(f"  {w:>6} bar  {w*TF:>4}m  {len(records):>7,}  {p['n']:>7,}  "
                  f"{p['p_tgt']:>7.3f}  {p['p_stop']:>7.3f}  {p['ev']:>+9.4f}σ{flag}")

        # ── 2. EV by CSR window × vol regime ────────────────────────────────────
        print(f"\n  BY VOL REGIME — best CSR window per regime")
        print(f"  (GK annualised vol, {GK_VOL_BARS}-bar window)")

        for r_label, r_lo, r_hi in VOL_REGIMES:
            regime_recs = [r for r in records if r_lo <= r["gk_vol"] < r_hi]
            if len(regime_recs) < 20:
                print(f"\n  {r_label}: too few triggers ({len(regime_recs)}), skipping")
                continue

            print(f"\n  {r_label}  (n_raw={len(regime_recs):,})")
            print(f"  {'Window':>8}  {'mins':>5}  {'n(CSR+)':>7}  "
                  f"{'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}")
            print(f"  {'─'*54}")

            best_ev, best_w = -999.0, 8
            rows = []
            for w in CSR_WINDOWS:
                filtered = [r for r in regime_recs
                            if csr_for(r, w) >= CSR_THRESHOLD]
                p = ev_stats(filtered, PRAC_S, PRAC_T)
                rows.append((w, p))
                if not math.isnan(p["ev"]) and p["ev"] > best_ev:
                    best_ev, best_w = p["ev"], w

            for w, p in rows:
                flag = " ◄ BEST" if w == best_w else (
                       " ◄ baseline" if w == 8 else "")
                if math.isnan(p["ev"]):
                    print(f"  {w:>6} bar  {w*TF:>4}m  {p['n']:>7,}  {'—':>7}  {'—':>7}  {'—':>9}{flag}")
                else:
                    print(f"  {w:>6} bar  {w*TF:>4}m  {p['n']:>7,}  "
                          f"{p['p_tgt']:>7.3f}  {p['p_stop']:>7.3f}  "
                          f"{p['ev']:>+9.4f}σ{flag}")

        # ── 3. Dynamic rule: assign window based on vol regime ───────────────────
        print(f"\n  {'─'*70}")
        print(f"  DYNAMIC RULE TEST")
        print(f"  Assign CSR window based on GK vol regime (using best-per-regime windows above)")
        print(f"  vs fixed 8-bar baseline")

        # First collect best windows per regime from the analysis above
        regime_best = {}
        for r_label, r_lo, r_hi in VOL_REGIMES:
            regime_recs = [r for r in records if r_lo <= r["gk_vol"] < r_hi]
            if len(regime_recs) < 20:
                regime_best[(r_lo, r_hi)] = 8
                continue
            best_ev2, best_w2 = -999.0, 8
            for w in CSR_WINDOWS:
                filtered = [r for r in regime_recs if csr_for(r, w) >= CSR_THRESHOLD]
                p = ev_stats(filtered, PRAC_S, PRAC_T)
                if not math.isnan(p["ev"]) and p["ev"] > best_ev2:
                    best_ev2, best_w2 = p["ev"], w
            regime_best[(r_lo, r_hi)] = best_w2

        print(f"\n  Vol regime → CSR window assigned:")
        for r_label, r_lo, r_hi in VOL_REGIMES:
            w = regime_best[(r_lo, r_hi)]
            print(f"    {r_label}: {w} bars ({w*TF} min)")

        # Apply dynamic rule
        dynamic = []
        for r in records:
            gk = r["gk_vol"]
            w = 8  # default
            for r_label, r_lo, r_hi in VOL_REGIMES:
                if r_lo <= gk < r_hi:
                    w = regime_best[(r_lo, r_hi)]
                    break
            if csr_for(r, w) >= CSR_THRESHOLD:
                dynamic.append(r)

        fixed = [r for r in records if csr_for(r, 8) >= CSR_THRESHOLD]
        p_dyn  = ev_stats(dynamic, PRAC_S, PRAC_T)
        p_fix  = ev_stats(fixed,   PRAC_S, PRAC_T)
        delta  = p_dyn["ev"] - p_fix["ev"] if not math.isnan(p_dyn["ev"]) else float("nan")

        print(f"\n  {'':28}  {'n':>6}  {'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}")
        print(f"  {'─'*58}")
        print(f"  {'Fixed 8-bar (baseline)':<28}  {p_fix['n']:>6,}  "
              f"{p_fix['p_tgt']:>7.3f}  {p_fix['p_stop']:>7.3f}  {p_fix['ev']:>+9.4f}σ")
        print(f"  {'Dynamic (best per regime)':<28}  {p_dyn['n']:>6,}  "
              f"{p_dyn['p_tgt']:>7.3f}  {p_dyn['p_stop']:>7.3f}  {p_dyn['ev']:>+9.4f}σ  "
              f"({delta:>+.4f}σ vs fixed)")

        # Year-by-year for dynamic vs fixed
        years = sorted(set(r["year"] for r in records))
        print(f"\n  By year:")
        print(f"  {'Year':<6}  {'n(fix)':>7}  {'EV(fix)':>9}  {'n(dyn)':>7}  {'EV(dyn)':>9}  {'delta':>8}")
        print(f"  {'─'*58}")
        for yr in years:
            yr_recs = [r for r in records if r["year"] == yr]
            f_yr = [r for r in yr_recs if csr_for(r, 8) >= CSR_THRESHOLD]
            d_yr = []
            for r in yr_recs:
                gk = r["gk_vol"]
                w  = 8
                for _, r_lo, r_hi in VOL_REGIMES:
                    if r_lo <= gk < r_hi:
                        w = regime_best[(r_lo, r_hi)]
                        break
                if csr_for(r, w) >= CSR_THRESHOLD:
                    d_yr.append(r)
            pf = ev_stats(f_yr, PRAC_S, PRAC_T)
            pd_ = ev_stats(d_yr, PRAC_S, PRAC_T)
            d   = pd_["ev"] - pf["ev"] if not math.isnan(pd_["ev"]) and not math.isnan(pf["ev"]) else float("nan")
            flag = " ◄" if not math.isnan(d) and d > 0.02 else ""
            print(f"  {yr:<6}  {pf['n']:>7,}  {pf['ev']:>+9.4f}σ  "
                  f"{pd_['n']:>7,}  {pd_['ev']:>+9.4f}σ  {d:>+8.4f}σ{flag}")
