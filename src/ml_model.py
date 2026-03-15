"""
Phase 1 ML model: predict MES signal continuation using OHLCV-derived features.

Target: does the ≥3σ + vol trigger lead to a continuation (target hit first)
        using the practical combo: -1.5σ stop / +2.5σ target, ≤15min hold.

Features (all derived from 1-min OHLCV, no DOM required):
  Trigger bar:
    scaled          — |bar return / σ|
    vol_ratio       — bar volume / 100-bar mean
    gk_ann_vol      — Garman-Klass annualised vol, 20-bar window (regime)
    bar_range_ratio — trigger bar (high-low) / trailing avg (high-low)
    close_position  — (close-low)/(high-low): 1=closed at top, 0=at bottom
  Context:
    prior_scaled    — |scaled return| of bar before trigger
    prior_vol_ratio — volume ratio of bar before trigger
    momentum_5b     — cumulative log return over 5 bars before trigger (in σ)
    gk_vol_trend    — GK ann_vol vs 20-bar-ago GK ann_vol (expanding/contracting)
  Time:
    hour_utc        — UTC hour of bar close (captures session)
    day_of_week     — 0=Mon … 4=Fri

Validation: walk-forward — train on all years up to Y-1, test on year Y.
            Minimum 2 training years before first test.

Usage:
  python src/ml_model.py            # full report
  python src/ml_model.py --sym MYM  # switch instrument
"""

import argparse
import math
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

# ── Config ─────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TRAILING_BARS        = 100
GK_VOL_BARS          = 20    # 20 × 5-min = 100 min; GK estimator optimal window
TF                   = 5
MAX_BARS_HOLD        = 3
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5
BARS_PER_YEAR        = 252 * 23 * 60

STOP_SIGMA   = 1.5
TARGET_SIGMA = 2.5

VOL_FILTER_LO = 0.10
VOL_FILTER_HI = 0.20

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


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


def _gk_ann_vol(opens: np.ndarray, highs: np.ndarray,
                lows: np.ndarray, closes: np.ndarray) -> float:
    """Garman-Klass annualised vol on 5-min bars."""
    ln_hl = np.log(highs / lows)
    ln_co = np.log(closes / opens)
    gk = 0.5 * ln_hl ** 2 - (2 * math.log(2) - 1) * ln_co ** 2
    var = float(np.mean(gk))
    return math.sqrt(var * BARS_PER_YEAR / TF) if var > 0 else float("nan")


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
    records, i = [], 0
    while i + TF <= len(df1):
        chunk = df1.iloc[i: i + TF]
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
        i += TF
    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_dataset(bars: pd.DataFrame) -> pd.DataFrame:
    opens   = bars["open"].values
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_arr  = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)

    # Trailing high-low range for range_ratio feature
    ranges = highs - lows

    records = []
    LOOKBACK = 20   # bars for vol_trend

    for i in range(TRAILING_BARS, n - MAX_BARS_HOLD):
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
        if abs(scaled) < MIN_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        ann_vol   = sigma * math.sqrt(BARS_PER_YEAR / TF)  # close-return vol (used for vol filter)
        gk_start  = max(i - GK_VOL_BARS, 0)
        gk_ann    = _gk_ann_vol(opens[gk_start:i], highs[gk_start:i],
                                lows[gk_start:i], closes[gk_start:i])
        direction = 1 if scaled > 0 else -1
        entry     = closes[i]

        # ── Target: did continuation win? ─────────────────────────────────────
        tgt_price  = entry * math.exp( direction * TARGET_SIGMA * sigma)
        stop_price = entry * math.exp(-direction * STOP_SIGMA   * sigma)
        hit_tgt = hit_stop = None
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            if hit_tgt is None:
                if direction == 1 and highs[j] >= tgt_price:
                    hit_tgt = j - i
                elif direction == -1 and lows[j] <= tgt_price:
                    hit_tgt = j - i
            if hit_stop is None:
                if direction == 1 and lows[j] <= stop_price:
                    hit_stop = j - i
                elif direction == -1 and highs[j] >= stop_price:
                    hit_stop = j - i

        if hit_tgt is not None and (hit_stop is None or hit_tgt < hit_stop):
            label = 1   # continuation: target hit first
        elif hit_stop is not None:
            label = 0   # reversal: stop hit first
        else:
            label = 0   # time exit: treat as non-continuation

        # ── Features ──────────────────────────────────────────────────────────

        # Trigger bar
        bar_range     = highs[i] - lows[i]
        avg_range     = np.mean(ranges[i - TRAILING_BARS: i])
        range_ratio   = bar_range / avg_range if avg_range > 0 else 1.0
        hl             = highs[i] - lows[i]
        close_pos     = (closes[i] - lows[i]) / hl if hl > 0 else 0.5
        # Flip close_pos for short signals so 1 always means "closed in signal direction"
        if direction == -1:
            close_pos = 1 - close_pos

        # Prior bar
        prior_ret       = math.log(closes[i - 1] / closes[i - 2]) if i >= 2 else 0.0
        prior_scaled    = abs(prior_ret / sigma)
        prior_vol_ratio = volumes[i - 1] / mean_vol if mean_vol > 0 else 1.0

        # 5-bar momentum (cumulative log return in σ, aligned with direction)
        mom_5 = math.log(closes[i - 1] / closes[i - 6]) * direction / sigma \
                if i >= 6 else 0.0

        # GK vol trend: expanding (+) or contracting (-)
        if i >= GK_VOL_BARS + LOOKBACK:
            gk_j   = i - LOOKBACK
            gk_old = _gk_ann_vol(opens[gk_j - GK_VOL_BARS: gk_j],
                                 highs[gk_j - GK_VOL_BARS: gk_j],
                                 lows[gk_j - GK_VOL_BARS:  gk_j],
                                 closes[gk_j - GK_VOL_BARS: gk_j])
            gk_trend = (gk_ann - gk_old) if not (math.isnan(gk_ann) or math.isnan(gk_old)) else 0.0
        else:
            gk_trend = 0.0

        # Time
        ts          = ts_arr[i]
        hour_utc    = ts.hour + ts.minute / 60
        day_of_week = ts.dayofweek   # 0=Mon, 4=Fri

        records.append({
            "ts":           ts,
            "year":         ts.year,
            "label":        label,
            # Trigger features
            "scaled":       abs(scaled),
            "vol_ratio":    vol_ratio,
            "ann_vol":      ann_vol,      # close-return vol (vol filter only)
            "gk_ann_vol":   gk_ann,       # Garman-Klass vol (feature)
            "range_ratio":  range_ratio,
            "close_pos":    close_pos,
            # Context features
            "prior_scaled":    prior_scaled,
            "prior_vol_ratio": prior_vol_ratio,
            "momentum_5b":     mom_5,
            "gk_vol_trend":    gk_trend,
            # Time features
            "hour_utc":     hour_utc,
            "day_of_week":  day_of_week,
        })

    return pd.DataFrame(records)


# ── Walk-forward validation ────────────────────────────────────────────────────

FEATURE_COLS = [
    "scaled", "vol_ratio", "gk_ann_vol",
    "range_ratio", "close_pos",
    "prior_scaled", "prior_vol_ratio", "momentum_5b", "gk_vol_trend",
    "hour_utc", "day_of_week",
]

MODELS = {
    "LogReg":  Pipeline([("sc", StandardScaler()),
                         ("m",  LogisticRegression(class_weight="balanced",
                                                    max_iter=1000, C=0.1))]),
    "RF":      RandomForestClassifier(n_estimators=500, max_depth=4,
                                       min_samples_leaf=20,
                                       class_weight="balanced",
                                       random_state=42, n_jobs=-1),
    "GBM":     GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                           learning_rate=0.05,
                                           subsample=0.8,
                                           min_samples_leaf=20,
                                           random_state=42),
}


def walk_forward(df: pd.DataFrame, vol_filter: bool = True):
    if vol_filter:
        df = df[(df["ann_vol"] >= VOL_FILTER_LO) &
                (df["ann_vol"] <  VOL_FILTER_HI)].copy()

    years = sorted(df["year"].unique())
    # Need at least 2 training years before first test year
    test_years = years[2:]

    results = {name: [] for name in MODELS}

    print(f"\n  Walk-forward folds  "
          f"({'vol filtered' if vol_filter else 'unfiltered'}):")
    print(f"  {'Test year':<12} {'Train n':>8} {'Test n':>8}  "
          + "  ".join(f"{name:>10}" for name in MODELS))
    print(f"  {'─'*70}")

    for test_yr in test_years:
        train = df[df["year"] < test_yr]
        test  = df[df["year"] == test_yr]
        if len(train) < 50 or len(test) < 10:
            continue

        X_tr = train[FEATURE_COLS].values
        y_tr = train["label"].values
        X_te = test[FEATURE_COLS].values
        y_te = test["label"].values

        row = f"  {test_yr:<12} {len(train):>8,} {len(test):>8,}"
        for name, model in MODELS.items():
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_tr, y_tr)
            proba = m.predict_proba(X_te)[:, 1]
            auc   = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else float("nan")
            results[name].append((test_yr, auc, proba, y_te))
            row += f"  {auc:>10.4f}"
        print(row)

    # Summary
    print(f"\n  Mean out-of-sample AUC (chance = 0.500):")
    for name, folds in results.items():
        aucs = [f[1] for f in folds if not math.isnan(f[1])]
        print(f"    {name:<8}: {np.mean(aucs):.4f}  ±{np.std(aucs):.4f}")

    return results


# ── Feature importance (RF) ────────────────────────────────────────────────────

def fit_final_rf(df: pd.DataFrame, vol_filter: bool = True) -> RandomForestClassifier:
    if vol_filter:
        df = df[(df["ann_vol"] >= VOL_FILTER_LO) &
                (df["ann_vol"] <  VOL_FILTER_HI)].copy()

    from sklearn.base import clone
    rf = clone(MODELS["RF"])
    rf.fit(df[FEATURE_COLS].values, df["label"].values)

    print(f"\n  Feature importances (full dataset RF, vol-filtered):")
    imp = sorted(zip(FEATURE_COLS, rf.feature_importances_),
                 key=lambda x: x[1], reverse=True)
    for feat, score in imp:
        bar = "█" * int(score * 200)
        print(f"    {feat:<20}  {score:.4f}  {bar}")

    return rf


# ── Probability threshold EV analysis ─────────────────────────────────────────

def threshold_ev(results: dict, sym: str):
    """
    Pool all out-of-sample predictions and compute EV at various
    probability thresholds.  EV = p_tgt*TARGET_SIGMA - p_stop*STOP_SIGMA
    (time exits treated as -0 here since their mean P&L is near 0).
    """
    name = "RF"
    all_proba, all_labels = [], []
    for _, _, proba, labels in results[name]:
        all_proba.extend(proba)
        all_labels.extend(labels)
    all_proba  = np.array(all_proba)
    all_labels = np.array(all_labels)

    print(f"\n  EV by probability threshold  (RF, out-of-sample, {sym}):")
    print(f"  {'Threshold':>10}  {'n signals':>10}  {'% of all':>9}  "
          f"{'P(cont)':>9}  {'EV (σ)':>9}")
    print(f"  {'─'*56}")

    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        mask   = all_proba >= thresh
        n      = mask.sum()
        if n < 10:
            print(f"  {thresh:>10.2f}  {n:>10}  [too few]")
            continue
        p_cont = all_labels[mask].mean()
        p_stop = 1 - p_cont            # simplified (ignores time exits)
        ev     = p_cont * TARGET_SIGMA - p_stop * STOP_SIGMA
        pct    = n / len(all_proba) * 100
        flag   = "  ◄" if ev > 0 else ""
        print(f"  {thresh:>10.2f}  {n:>10,}  {pct:>8.1f}%  "
              f"{p_cont:>9.3f}  {ev:>+9.4f}σ{flag}")


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

    print("Building feature dataset …")
    df = build_dataset(bars)
    print(f"  {len(df):,} triggers  "
          f"(positive={df['label'].mean()*100:.1f}%  "
          f"n_pos={df['label'].sum():,}  n_neg={(df['label']==0).sum():,})")

    filt = df[(df["ann_vol"] >= VOL_FILTER_LO) & (df["ann_vol"] < VOL_FILTER_HI)]
    print(f"  After vol filter: {len(filt):,} triggers  "
          f"(positive={filt['label'].mean()*100:.1f}%)")

    print(f"\n{'═'*72}")
    print(f"  WALK-FORWARD VALIDATION  —  {sym}")
    print(f"{'═'*72}")
    results = walk_forward(df, vol_filter=True)

    print(f"\n{'═'*72}")
    print(f"  FEATURE IMPORTANCES  —  {sym}")
    print(f"{'═'*72}")
    fit_final_rf(df, vol_filter=True)

    print(f"\n{'═'*72}")
    print(f"  PROBABILITY THRESHOLD ANALYSIS  —  {sym}")
    print(f"{'═'*72}")
    threshold_ev(results, sym)
