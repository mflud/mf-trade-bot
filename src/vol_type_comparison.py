"""
Compare signal edge using close-return σ (original) vs Garman-Klass σ (improved).

Close-return: σ = std of 100 trailing 5-min close-to-close log returns (500 min)
GK          : σ = sqrt(mean GK estimator over 4 trailing 5-min bars) = 20 min
              (20 min is the optimal window from vol_predict_grid.py)

Both σ measures are used identically for:
  - scaled return threshold (|bar_ret / σ| ≥ 3)
  - stop/target sizing (-1.5σ stop / +2.5σ target)
  - regime classification (annualised vol buckets)

Usage:
  python src/vol_type_comparison.py            # MES
  python src/vol_type_comparison.py --sym MYM
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
TRAILING_CLOSE       = 100   # bars for close-return σ  (500 min)
GK_VOL_BARS          = 4     # bars for GK σ  (4 × 5-min = 20 min, optimal from grid)
TF                   = 5
MAX_BARS_HOLD        = 3
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5
BARS_PER_YEAR        = 252 * 23 * 60

PRAC_S, PRAC_T = 1.5, 2.5
VOL_FILTER_LO  = 0.10
VOL_FILTER_HI  = 0.20

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

VOL_REGIME_BINS   = [0, 0.10, 0.15, 0.20, 0.30, 99]
VOL_REGIME_LABELS = ["QUIET (<10%)", "NORMAL (10–15%)", "ELEVATED (15–20%)",
                     "ACTIVE (20–30%)", "HIGH VOL (>30%)"]

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


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
    while i + TF <= len(df1):
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].values.any():
            i += int(chunk["gap"].iloc[1:].values.argmax()) + 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[0],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += TF
    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Vol estimators ─────────────────────────────────────────────────────────────

def close_sigma(closes: np.ndarray) -> float:
    r = np.log(closes[1:] / closes[:-1])
    return float(np.std(r, ddof=1)) if len(r) >= 2 else 0.0


def gk_sigma(opens: np.ndarray, highs: np.ndarray,
             lows: np.ndarray, closes: np.ndarray) -> float:
    ln_hl = np.log(highs / lows)
    ln_co = np.log(closes / opens)
    gk = 0.5 * ln_hl ** 2 - (2 * math.log(2) - 1) * ln_co ** 2
    var = float(np.mean(gk))
    return math.sqrt(var) if var > 0 else 0.0


# ── Scanner ────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, vol_type: str) -> pd.DataFrame:
    """vol_type: 'close' (original) or 'gk' (Garman-Klass)."""
    opens   = bars["open"].values
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    n       = len(bars)

    lookback = TRAILING_CLOSE if vol_type == "close" else GK_VOL_BARS
    records  = []

    for i in range(lookback, n - MAX_BARS_HOLD):
        if gaps[i - lookback + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        if vol_type == "close":
            sigma = close_sigma(closes[i - TRAILING_CLOSE: i + 1])
        else:
            sigma = gk_sigma(opens[i - GK_VOL_BARS: i],
                             highs[i - GK_VOL_BARS: i],
                             lows[i - GK_VOL_BARS: i],
                             closes[i - GK_VOL_BARS: i])

        if sigma == 0:
            continue

        mean_vol  = volumes[max(i - TRAILING_CLOSE, 0): i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction  = 1 if scaled > 0 else -1
        entry      = closes[i]
        sigma_pts  = sigma * entry
        ann_vol    = sigma * math.sqrt(BARS_PER_YEAR / TF)

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if (direction == 1 and h >= tgt_prices[t]) or \
                       (direction == -1 and l <= tgt_prices[t]):
                        hit_tgt[t] = j - i
            for s in STOPS:
                if hit_stop[s] is None:
                    if (direction == 1 and l <= stop_prices[s]) or \
                       (direction == -1 and h >= stop_prices[s]):
                        hit_stop[s] = j - i

        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
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
        return {"ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan"), "n": len(sub)}
    ht = sub[f"hit_tgt_{t}"].notna().values
    hs = sub[f"hit_stop_{s}"].notna().values
    ht_first = ht & ~(hs & (sub[f"hit_stop_{s}"].fillna(999)
                            <= sub[f"hit_tgt_{t}"].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    time_ret = sub["time_exit_ret"].values
    ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
    return {
        "ev":     ht_first.mean() * t - hs_first.mean() * s + neither.mean() * ev_nei,
        "p_tgt":  ht_first.mean(),
        "p_stop": hs_first.mean(),
        "n":      len(sub),
    }


def best_ev(sub: pd.DataFrame) -> tuple[float, float, float]:
    best, bs, bt = -999.0, 0.0, 0.0
    for s in STOPS:
        for t in TARGETS:
            st = ev_stats(sub, s, t)
            if not math.isnan(st["ev"]) and st["ev"] > best:
                best, bs, bt = st["ev"], s, t
    return best, bs, bt


# ── Comparison report ──────────────────────────────────────────────────────────

def report_comparison(sym: str, res_c: pd.DataFrame, res_g: pd.DataFrame):

    def fmt_row(label: str, rc: pd.DataFrame, rg: pd.DataFrame, min_n: int = 30) -> str:
        pc = ev_stats(rc, PRAC_S, PRAC_T)
        pg = ev_stats(rg, PRAC_S, PRAC_T)
        bc, _, _ = best_ev(rc)
        bg, _, _ = best_ev(rg)
        fc = "◄" if pc["ev"] > 0 else " "
        fg = "◄" if pg["ev"] > 0 else " "

        if pc["n"] < min_n:
            col_c = f"{'[n=' + str(pc['n']) + ']':^44}"
        else:
            col_c = (f"n={pc['n']:>5,}  P(t)={pc['p_tgt']:.3f}  "
                     f"EV={pc['ev']:>+7.4f}σ  best={bc:>+6.4f}σ {fc}")
        if pg["n"] < min_n:
            col_g = f"{'[n=' + str(pg['n']) + ']':^44}"
        else:
            col_g = (f"n={pg['n']:>5,}  P(t)={pg['p_tgt']:.3f}  "
                     f"EV={pg['ev']:>+7.4f}σ  best={bg:>+6.4f}σ {fg}")
        return f"  {label:<28}  {col_c:<44}   {col_g}"

    hline = "  " + "─" * 106

    print(f"\n{'═'*110}")
    print(f"  {sym}  —  CLOSE-RETURN σ  ({TRAILING_CLOSE} bars = {TRAILING_CLOSE*TF} min)  "
          f"vs  GARMAN-KLASS σ  ({GK_VOL_BARS} bars = {GK_VOL_BARS*TF} min)")
    print(f"{'═'*110}")
    print(f"\n  {'Slice':<28}  "
          f"{'── CLOSE-RETURN ─────────────────────────────':^44}   "
          f"{'── GARMAN-KLASS ─────────────────────────────':^44}")
    print(hline)

    print(fmt_row("ALL TRIGGERS", res_c, res_g))

    fc = res_c[(res_c["ann_vol"] >= VOL_FILTER_LO) & (res_c["ann_vol"] < VOL_FILTER_HI)].copy()
    fg = res_g[(res_g["ann_vol"] >= VOL_FILTER_LO) & (res_g["ann_vol"] < VOL_FILTER_HI)].copy()
    print(fmt_row(f"VOL FILTER {VOL_FILTER_LO*100:.0f}–{VOL_FILTER_HI*100:.0f}%", fc, fg))

    print(hline)
    print(f"  BY VOLATILITY REGIME:")

    res_c["vol_regime"] = pd.cut(res_c["ann_vol"], bins=VOL_REGIME_BINS, labels=VOL_REGIME_LABELS)
    res_g["vol_regime"] = pd.cut(res_g["ann_vol"], bins=VOL_REGIME_BINS, labels=VOL_REGIME_LABELS)

    for lbl in VOL_REGIME_LABELS:
        print(fmt_row(f"  {lbl}",
                      res_c[res_c["vol_regime"] == lbl],
                      res_g[res_g["vol_regime"] == lbl]))

    # EV grid side by side
    print(f"\n{hline}")
    print(f"  EV GRID  (vol-filtered {VOL_FILTER_LO*100:.0f}–{VOL_FILTER_HI*100:.0f}%)  "
          f"rows=stop, cols=target:")
    col_hdr = "".join(f"  +{t:.1f}σ" for t in TARGETS)
    print(f"\n  {'':8}  {'CLOSE-RETURN':^36}   {'GARMAN-KLASS':^36}")
    print(f"  {'Stop':<8}" + col_hdr + "   " + col_hdr)
    print("  " + "─" * 84)
    for s in STOPS:
        line = f"  -{s:.1f}σ  "
        for t in TARGETS:
            line += f"  {ev_stats(fc, s, t)['ev']:>+5.3f}"
        line += "   "
        for t in TARGETS:
            line += f"  {ev_stats(fg, s, t)['ev']:>+5.3f}"
        print(line)

    # Summary
    ev_c = ev_stats(fc, PRAC_S, PRAC_T)
    ev_g = ev_stats(fg, PRAC_S, PRAC_T)
    sp_c = fc["sigma_pts"].median() if len(fc) else 0.0
    sp_g = fg["sigma_pts"].median() if len(fg) else 0.0

    print(f"\n{hline}")
    print(f"  PRACTICAL COMBO (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ) — vol-filtered:")
    print(f"  {'':30}  {'CLOSE-RETURN':>15}   {'GARMAN-KLASS':>15}")
    print(f"  {'n triggers':<30}  {ev_c['n']:>15,}   {ev_g['n']:>15,}")
    print(f"  {'P(target)':<30}  {ev_c['p_tgt']:>15.3f}   {ev_g['p_tgt']:>15.3f}")
    print(f"  {'P(stop)':<30}  {ev_c['p_stop']:>15.3f}   {ev_g['p_stop']:>15.3f}")
    print(f"  {'EV (σ)':<30}  {ev_c['ev']:>+15.4f}   {ev_g['ev']:>+15.4f}")
    print(f"  {'Median 1σ (pts)':<30}  {sp_c:>15.2f}   {sp_g:>15.2f}")
    print(f"  {'EV (pts)':<30}  {ev_c['ev']*sp_c:>+15.3f}   {ev_g['ev']*sp_g:>+15.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES")
    args = parser.parse_args()

    sym   = args.sym.upper()
    cache = INSTRUMENTS.get(sym)
    if not cache:
        print(f"Unknown instrument: {sym}")
        sys.exit(1)

    print(f"Loading {cache} …")
    df1  = load_1min(cache)
    bars = make_5min_bars(df1)
    print(f"  {len(bars):,} 5-min bars")

    print("Scanning with close-return σ …")
    res_c = scan(bars, "close")
    print(f"  {len(res_c):,} triggers")

    print("Scanning with Garman-Klass σ …")
    res_g = scan(bars, "gk")
    print(f"  {len(res_g):,} triggers")

    report_comparison(sym, res_c, res_g)
