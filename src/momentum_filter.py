"""
Momentum filter analysis for the 3σ continuation signal.

For each trigger bar, computes two momentum measures over windows of 20–100 min
(4–20 × 5-min bars) PRIOR to the trigger:

  1. Cumulative scaled return (CSR): sum of (bar_ret / σ) over the window
     - Same units as the trigger scaled return
     - Positive = price has been rising; negative = falling

  2. Linear regression slope (LRS): slope of OLS fit through the last N closes,
     normalised by σ per bar so it's scale-free

Triggers are split into:
  "with"    — momentum aligns with signal direction  (CSR × direction > threshold)
  "against" — momentum opposes signal direction      (CSR × direction < -threshold)
  "neutral" — |CSR| below threshold (near-zero trend)

Reports EV at -2σ/+3σ for each split and each window.

Usage:
  python src/momentum_filter.py            # MES
  python src/momentum_filter.py --sym MYM
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")

# ── Config ─────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TRAILING_BARS        = 20    # 20 × 5-min = 100-min σ window (current optimal)
TF                   = 5
MAX_BARS_HOLD        = 3
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5
BARS_PER_YEAR        = 252 * 23 * 60

PRAC_S, PRAC_T = 2.0, 3.0
STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Momentum lookback windows (in 5-min bars before the trigger)
MOM_WINDOWS = [4, 8, 12, 16, 20]   # 20, 40, 60, 80, 100 min

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data ───────────────────────────────────────────────────────────────────────

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


# ── Scanner ────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame) -> pd.DataFrame:
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    n       = len(bars)

    max_lookback = max(TRAILING_BARS, max(MOM_WINDOWS))
    records = []

    for i in range(max_lookback, n - MAX_BARS_HOLD):
        # Gap check: σ window + forward hold
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                          / closes[i - TRAILING_BARS:     i    ])
        sigma = float(np.std(trail_rets, ddof=1))
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1
        entry     = closes[i]

        # ── Momentum features ─────────────────────────────────────────────────
        mom = {}
        for w in MOM_WINDOWS:
            # Gap check for momentum window (bars i-w to i-1, not including trigger)
            if gaps[i - w: i].any():
                mom[f"csr_{w}"]  = float("nan")
                mom[f"lrs_{w}"]  = float("nan")
                continue

            window_rets = np.log(closes[i - w + 1: i]
                               / closes[i - w:     i - 1])   # w-1 returns

            # 1. Cumulative Scaled Return: sum of bar returns / σ
            csr = float(window_rets.sum()) / sigma

            # 2. Linear Regression Slope through closes, normalised by σ
            c_window = closes[i - w: i]   # w prices
            x        = np.arange(w, dtype=float)
            slope, *_ = stats.linregress(x, c_window)
            # Convert to σ-normalised units: slope is pts/bar → divide by σ_pts
            sigma_pts = sigma * entry
            lrs = slope / sigma_pts if sigma_pts > 0 else 0.0

            mom[f"csr_{w}"]  = csr * direction   # positive = "with" signal
            mom[f"lrs_{w}"]  = lrs * direction   # positive = "with" signal

        # ── Target/stop outcome ───────────────────────────────────────────────
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
            "time_exit_ret": time_exit_ret,
            **mom,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV helpers ─────────────────────────────────────────────────────────────────

def ev_stats(sub: pd.DataFrame, s: float = PRAC_S, t: float = PRAC_T) -> dict:
    if len(sub) < 5:
        return {"ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan"), "n": len(sub)}
    ht = sub[f"hit_tgt_{t}"].notna().values
    hs = sub[f"hit_stop_{s}"].notna().values
    ht_first = ht & ~(hs & (sub[f"hit_stop_{s}"].fillna(999)
                            <= sub[f"hit_tgt_{t}"].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    ev_nei   = sub["time_exit_ret"].values[neither].mean() if neither.any() else 0.0
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


def fmt(ev, n, min_n=30):
    if n < min_n or math.isnan(ev):
        return f"{'[n=' + str(n) + ']':>18}"
    mk = "◄" if ev > 0 else " "
    return f"{ev:>+8.4f}σ  n={n:<5}{mk}"


# ── Report ─────────────────────────────────────────────────────────────────────

def report(sym: str, res: pd.DataFrame):
    hline = "  " + "─" * 90

    p_all = ev_stats(res)
    bev, bs, bt = best_ev(res)
    print(f"\n{'═'*94}")
    print(f"  {sym}  —  MOMENTUM FILTER ANALYSIS  (trail=20, -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)")
    print(f"  Baseline (all triggers): EV={p_all['ev']:>+.4f}σ  n={p_all['n']:,}  "
          f"P(tgt)={p_all['p_tgt']:.3f}  best={bev:>+.4f}σ at -{bs:.1f}σ/+{bt:.1f}σ")
    print(f"{'═'*94}")

    for measure, col_prefix, label in [
        ("csr", "csr", "CUMULATIVE SCALED RETURN (sum of bar_ret/σ over window)"),
        ("lrs", "lrs", "LINEAR REGRESSION SLOPE  (normalised by σ_pts)"),
    ]:
        print(f"\n  ── {label}")
        print(hline)

        # Header: windows across columns
        win_hdr = "".join(f"  {w*TF:>3}min={w:>2}b" for w in MOM_WINDOWS)
        print(f"  {'Threshold / split':<22}" + win_hdr)
        print(hline)

        # For each threshold, split into with/against and show EV
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0]

        for thr in thresholds:
            for split_name, sign in [("WITH  momentum", 1), ("AGAINST momentum", -1)]:
                row = f"  {split_name:<14} >{thr:.1f}σ  "
                for w in MOM_WINDOWS:
                    col = f"{col_prefix}_{w}"
                    vals = res[col].dropna()
                    if split_name.startswith("WITH"):
                        mask = res[col] > thr
                    else:
                        mask = res[col] < -thr
                    sub = res[mask.fillna(False)]
                    p   = ev_stats(sub)
                    if p["n"] < 30:
                        row += f"  {'[n=' + str(p['n']) + ']':>10}"
                    else:
                        mk  = "◄" if p["ev"] > 0 else " "
                        row += f"  {p['ev']:>+7.4f}{mk}"
                print(row)

            # Neutral band
            if thr > 0:
                row = f"  {'NEUTRAL':>14} <{thr:.1f}σ  "
                for w in MOM_WINDOWS:
                    col = f"{col_prefix}_{w}"
                    mask = res[col].abs() <= thr
                    sub  = res[mask.fillna(False)]
                    p    = ev_stats(sub)
                    if p["n"] < 30:
                        row += f"  {'[n=' + str(p['n']) + ']':>10}"
                    else:
                        mk  = "◄" if p["ev"] > 0 else " "
                        row += f"  {p['ev']:>+7.4f}{mk}"
                print(row)
            print()

    # ── Best single filter: with-momentum only, optimal threshold & window ───
    print(hline)
    print(f"\n  BEST 'WITH MOMENTUM' CONFIG  (max EV with n ≥ 100):")
    print(f"  {'Measure':<8}  {'Window':>8}  {'Threshold':>10}  {'n':>6}  "
          f"{'EV':>9}  {'Best EV':>9}  {'P(tgt)':>7}  {'P(stop)':>7}")
    print(f"  {'─'*70}")

    best_configs = []
    for col_prefix in ["csr", "lrs"]:
        for w in MOM_WINDOWS:
            col = f"{col_prefix}_{w}"
            for thr in [0.0, 0.5, 1.0, 1.5, 2.0]:
                mask = res[col] > thr
                sub  = res[mask.fillna(False)]
                p    = ev_stats(sub)
                bev2, _, _ = best_ev(sub)
                if p["n"] >= 100 and not math.isnan(p["ev"]):
                    best_configs.append({
                        "measure": col_prefix.upper(), "window": w,
                        "window_min": w * TF, "threshold": thr,
                        "n": p["n"], "ev": p["ev"],
                        "best_ev": bev2, "p_tgt": p["p_tgt"], "p_stop": p["p_stop"],
                    })

    best_configs.sort(key=lambda x: x["ev"], reverse=True)
    for r in best_configs[:10]:
        flag = "◄" if r["ev"] > 0 else " "
        print(f"  {r['measure']:<8}  {r['window_min']:>4} min  "
              f"    >{r['threshold']:.1f}σ  {r['n']:>6,}  "
              f"{r['ev']:>+9.4f}σ{flag}  {r['best_ev']:>+9.4f}σ  "
              f"{r['p_tgt']:>7.3f}  {r['p_stop']:>7.3f}")


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

    print("Scanning triggers and computing momentum …")
    res = scan(bars)
    print(f"  {len(res):,} triggers")

    report(sym, res)
