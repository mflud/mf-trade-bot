"""
Intraday vol prediction for MES 1-min bars (full 24h session).

Two tests:
  A) Trailing 10h  →  next 1h  forward vol
  B) Trailing  5h  →  next 30min forward vol

Observations are strided by the forward window length so forward windows
are non-overlapping (avoids look-ahead bias in the dependent variable).
Settlement gap (21:00-22:00 UTC) bars are dropped; any window that spans
a gap is skipped.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
from topstep_client import TopstepClient

CACHE_PATH = "mes_hist_1min.csv"
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
BARS_PER_YEAR        = 252 * 23 * 60   # full 24h session

CONFIGS = [
    {"label": "A",  "trailing_bars": 10 * 60,  "forward_bars": 60,  "desc": "10h trailing  → 1h forward"},
    {"label": "B",  "trailing_bars":  5 * 60,  "forward_bars": 30,  "desc": " 5h trailing  → 30min forward"},
    {"label": "C",  "trailing_bars":      100,  "forward_bars": 10,  "desc": "100min trailing → 10min forward"},
    {"label": "D",  "trailing_bars":       50,  "forward_bars":  5,  "desc": " 50min trailing →  5min forward"},
    {"label": "E",  "trailing_bars":       30,  "forward_bars":  3,  "desc": " 30min trailing →  3min forward"},
    {"label": "F",  "trailing_bars":       20,  "forward_bars":  2,  "desc": " 20min trailing →  2min forward"},
]


# ── Data ─────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    if not os.path.exists(CACHE_PATH):
        print(f"Cache not found at {CACHE_PATH}. Run vol_predict.py first to fetch data.")
        sys.exit(1)

    df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")

    # Drop settlement gap
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # Mark positions where gap > 1 bar (session boundary / roll)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)

    print(f"Loaded {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")
    return df


# ── Backtest ──────────────────────────────────────────────────────────────────

def realised_vol(closes: np.ndarray) -> float:
    rets = np.log(closes[1:] / closes[:-1])
    if len(rets) < 1:
        return float("nan")
    ddof = 1 if len(rets) >= 2 else 0
    return float(np.std(rets, ddof=ddof) * math.sqrt(BARS_PER_YEAR))


def run(df: pd.DataFrame, trailing_bars: int, forward_bars: int) -> pd.DataFrame:
    closes = df["close"].values
    gaps   = df["gap"].values
    n      = len(df)

    records = []
    # Stride by forward_bars so forward windows don't overlap
    i = trailing_bars
    while i + forward_bars < n:
        trail_slice = slice(i - trailing_bars, i + 1)
        fwd_slice   = slice(i + 1, i + forward_bars + 1)

        # Skip if any gap falls inside the trailing or forward window
        if gaps[trail_slice].any() or gaps[fwd_slice].any():
            i += 1
            continue

        tv = realised_vol(closes[trail_slice])
        fv = realised_vol(closes[fwd_slice])

        if not (math.isnan(tv) or math.isnan(fv)):
            records.append({
                "ts":           df["ts"].iloc[i],
                "trailing_vol": tv,
                "forward_vol":  fv,
            })

        i += forward_bars   # stride

    return pd.DataFrame(records)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(res: pd.DataFrame, desc: str):
    tv = res["trailing_vol"].values
    fv = res["forward_vol"].values
    n  = len(res)

    pearson_r,  pearson_p  = stats.pearsonr(tv, fv)
    spearman_r, spearman_p = stats.spearmanr(tv, fv)
    r2 = pearson_r ** 2
    slope, intercept, *_ = stats.linregress(tv, fv)

    mae_model    = np.mean(np.abs(tv  - fv))
    mae_baseline = np.mean(np.abs(tv.mean() - fv))
    skill = 1 - mae_model / mae_baseline

    print(f"\n{'='*62}")
    print(f"  {desc}")
    print(f"  n = {n:,} non-overlapping observations  "
          f"({res['ts'].dt.date.min()} → {res['ts'].dt.date.max()})")
    print(f"{'='*62}")

    print(f"\n  Correlation")
    print(f"    Pearson  r  = {pearson_r:+.4f}   p = {pearson_p:.2e}")
    print(f"    Spearman r  = {spearman_r:+.4f}   p = {spearman_p:.2e}")
    print(f"    R²          = {r2:.4f}  ({r2*100:.1f}% of variance explained)")
    print(f"\n  Linear fit:  fwd_vol = {slope:.4f} × trailing_vol + {intercept:.4f}")

    mean_tv = tv.mean()
    print(f"\n  MAE comparison (annualised vol)")
    print(f"    Model    (trailing vol as predictor):  {mae_model*100:.2f}pp")
    print(f"    Baseline (historical mean = {mean_tv*100:.1f}%):   {mae_baseline*100:.2f}pp")
    skill_word = "better" if skill > 0 else "worse"
    print(f"    Skill score: {skill:+.3f}  ({abs(skill)*100:.1f}% {skill_word} than mean baseline)")

    res2 = res.copy()
    res2["quartile"] = pd.qcut(res2["trailing_vol"], 4,
                               labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    print(f"\n  Forward vol by trailing vol quartile:")
    print(f"    {'Quartile':<12} {'n':>5}  {'mean fwd':>10}  {'median fwd':>11}  {'std':>8}  trailing range")
    for label, grp in res2.groupby("quartile", observed=True):
        fv_g = grp["forward_vol"]
        tv_g = grp["trailing_vol"]
        print(f"    {str(label):<12} {len(grp):>5}  "
              f"{fv_g.mean()*100:>9.2f}%  "
              f"{fv_g.median()*100:>10.2f}%  "
              f"{fv_g.std()*100:>7.2f}%  "
              f"({tv_g.min()*100:.1f}–{tv_g.max()*100:.1f}%)")

    print(f"\n  Conclusion:")
    if abs(pearson_r) >= 0.5 and pearson_p < 0.05:
        strength = "strong"
    elif abs(pearson_r) >= 0.3 and pearson_p < 0.05:
        strength = "moderate"
    elif abs(pearson_r) >= 0.1 and pearson_p < 0.05:
        strength = "weak"
    else:
        strength = "negligible / not significant"
    print(f"    {strength.capitalize()} signal (r={pearson_r:.3f}, R²={r2*100:.1f}%, "
          f"p={pearson_p:.2e}).")
    if skill > 0:
        print(f"    Trailing vol beats the mean baseline by {skill*100:.1f}% on MAE.")
    else:
        print(f"    Trailing vol does NOT beat the mean baseline on MAE.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    for cfg in CONFIGS:
        res = run(df, cfg["trailing_bars"], cfg["forward_bars"])
        analyse(res, f"Test {cfg['label']}: {cfg['desc']}")
