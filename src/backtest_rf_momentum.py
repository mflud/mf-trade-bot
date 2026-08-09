"""
backtest_rf_momentum.py — Random Forest momentum predictor for MES.

All features are built on the 1-min bar series (RTH only), then resampled
to 2-min for the medium-term return features. No expensive joins needed.

Features at each 2-min bar close:
  • r2m_1 … r2m_{lookback-1}  : last N 2-min log returns  (10–40 min)
  • r1m_1 … r1m_6             : last 6 1-min log returns  (~6 min)
  • ret_open                   : log return since 9:30 open
  • mins_since_open            : minutes elapsed since open (time-of-day)

Targets (12 total, swept):
  • Does close cross ±X pts within H forward 2-min bars?
    X ∈ {2, 3, 4, 5},  H ∈ {1, 2, 3}

Train/test: walk-forward — first 75% of RTH days train, last 25% test.

Data sources merged (oldest-first, duplicates dropped):
  • mes_hist_1min.csv  — Databento (2019–2026-03)
  • bars.db            — Live feed  (2026-04 onward)

Run from repo root:
    python src/backtest_rf_momentum.py
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import time
from zoneinfo import ZoneInfo
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

ET = ZoneInfo("America/New_York")
DB_PATH  = "data/bars.db"
HIST_CSV = "mes_hist_1min.csv"

LOOKBACKS  = [5, 10, 15, 20]   # 2-min lookback bars (10–40 min)
SHORT_N    = 6                  # 1-min short-term bars
THRESHOLDS = [2.0, 3.0, 4.0, 5.0]
HORIZONS   = [1, 2, 3]         # 2-min bars forward
TRAIN_FRAC = 0.75

RTH_START = time(9, 30)
RTH_END   = time(16, 0)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_rth_1min() -> pd.DataFrame:
    frames = []

    hist = Path(HIST_CSV)
    if hist.exists():
        print(f"  Reading {HIST_CSV} …", flush=True)
        df_h = pd.read_csv(hist, parse_dates=["ts"])
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= RTH_START) & (t < RTH_END)]
        frames.append(df_h)
        print(f"  Databento RTH : {len(df_h):,} bars  "
              f"({df_h['ts'].dt.date.min()} → {df_h['ts'].dt.date.max()})",
              flush=True)

    print("  Reading bars.db …", flush=True)
    db = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql(
        "SELECT ts, open, high, low, close, volume "
        "FROM bars WHERE symbol='MES' AND minutes=1 ORDER BY ts", db)
    db.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t = df_db["ts"].dt.time
    df_db = df_db[(t >= RTH_START) & (t < RTH_END)]
    frames.append(df_db)
    print(f"  Live DB RTH   : {len(df_db):,} bars  "
          f"({df_db['ts'].dt.date.min()} → {df_db['ts'].dt.date.max()})",
          flush=True)

    df = pd.concat(frames, ignore_index=True)
    df = (df.drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True))
    print(f"  Combined      : {len(df):,} bars  "
          f"({df['ts'].dt.date.min()} → {df['ts'].dt.date.max()})",
          flush=True)
    return df


# ── Vectorized feature + target construction ──────────────────────────────────

def build_dataset(df1: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (feat_df on 2-min grid, df2 for target lookup).
    All operations are vectorized on aligned DataFrames.
    """
    df1 = df1.copy()
    df1["date"] = df1["ts"].dt.date

    # ── 1-min log returns, zero at day boundaries ──────────────────────────
    df1["lret1"] = np.log(df1["close"] / df1["close"].shift(1))
    cross1 = df1["date"] != df1["date"].shift(1)
    df1.loc[cross1, "lret1"] = np.nan

    # Lagged 1-min returns (SHORT_N lags)
    for lag in range(1, SHORT_N + 1):
        s = df1["lret1"].shift(lag)
        s[df1["date"] != df1["date"].shift(lag)] = np.nan
        df1[f"r1m_{lag}"] = s

    # ── 9:30 open price per day ───────────────────────────────────────────
    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first()
                .rename("open_930"))
    df1 = df1.join(open_930, on="date")
    df1["ret_open"]      = np.log(df1["close"] / df1["open_930"])
    df1["mins_since_open"] = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute
                               - (9 * 60 + 30))

    # ── Resample to 2-min ─────────────────────────────────────────────────
    df1_idx = df1.set_index("ts")
    df2 = df1_idx[["open","high","low","close","volume"]].resample(
        "2min", label="right", closed="right"
    ).agg(open=("open","first"), high=("high","max"),
          low=("low","min"),   close=("close","last"),
          volume=("volume","sum")).dropna(subset=["close"])
    df2.index = df2.index.tz_convert(ET)
    t2 = df2.index.time
    df2 = df2[(t2 >= RTH_START) & (t2 < RTH_END)].reset_index()
    df2["date"] = df2["ts"].dt.date

    # ── 2-min log returns, zero at day boundaries ─────────────────────────
    df2["lret2"] = np.log(df2["close"] / df2["close"].shift(1))
    cross2 = df2["date"] != df2["date"].shift(1)
    df2.loc[cross2, "lret2"] = np.nan

    for lag in range(1, lookback):
        s = df2["lret2"].shift(lag)
        s[df2["date"] != df2["date"].shift(lag)] = np.nan
        df2[f"r2m_{lag}"] = s

    # ── Merge 1-min features onto 2-min grid via merge_asof ──────────────
    # Only carry over the short-term 1-min features
    r1m_cols = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols = ["ret_open", "mins_since_open"]
    df1_small = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    df2 = df2.sort_values("ts")
    df2 = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    # ── Drop incomplete rows ──────────────────────────────────────────────
    r2m_cols = [f"r2m_{i}" for i in range(1, lookback)]
    all_feat = r2m_cols + r1m_cols + scalar_cols
    df2 = df2.dropna(subset=all_feat)
    df2 = df2[df2["mins_since_open"] >= lookback * 2].reset_index(drop=True)
    df2["idx"] = df2.index

    return df2, df2   # same df used for both features and target lookup


def add_targets(df2: pd.DataFrame, threshold: float, horizon: int) -> pd.DataFrame:
    highs  = df2["high"].values
    lows   = df2["low"].values
    closes = df2["close"].values
    dates  = df2["date"].values
    n = len(df2)

    targets = np.full(n, np.nan)
    for i in range(n - horizon):
        fwd_dates = dates[i + 1 : i + 1 + horizon]
        if len(fwd_dates) < horizon or fwd_dates[0] != fwd_dates[-1]:
            continue
        if fwd_dates[0] != dates[i]:
            continue
        entry  = closes[i]
        max_h  = highs[i + 1 : i + 1 + horizon].max()
        min_l  = lows[i + 1  : i + 1 + horizon].min()
        targets[i] = int(max_h >= entry + threshold or
                         min_l <= entry - threshold)

    out = df2.copy()
    out["target"] = targets
    return out.dropna(subset=["target"])


# ── Walk-forward split ────────────────────────────────────────────────────────

def split(df: pd.DataFrame):
    dates  = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRAC)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff], cutoff


# ── Model ─────────────────────────────────────────────────────────────────────

def get_fcols(lookback: int) -> list[str]:
    r2m = [f"r2m_{i}" for i in range(1, lookback)]
    r1m = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    return r2m + r1m + ["ret_open", "mins_since_open"]


def run_rf(train: pd.DataFrame, test: pd.DataFrame, fcols: list[str]) -> dict:
    X_tr, y_tr = train[fcols].values, train["target"].astype(int).values
    X_te, y_te = test[fcols].values,  test["target"].astype(int).values

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    auc  = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else np.nan
    prec = precision_score(y_te, pred, zero_division=0)
    rec  = recall_score(y_te, pred, zero_division=0)
    imp  = dict(zip(fcols, clf.feature_importances_))
    top  = sorted(imp.items(), key=lambda x: -x[1])[:3]

    return {
        "n_train": len(train), "n_test": len(test),
        "base_rate": y_te.mean(),
        "auc": auc, "precision": prec, "recall": rec,
        "top_feats": top,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars …", flush=True)
    df1 = load_rth_1min()
    print(flush=True)

    all_results = []

    for lookback in LOOKBACKS:
        print(f"── Lookback = {lookback} 2-min bars ({lookback*2} min) ──────────",
              flush=True)
        print("   Building features …", end=" ", flush=True)
        df2, _ = build_dataset(df1, lookback)
        fcols = get_fcols(lookback)
        print(f"{len(df2):,} bars  |  {len(fcols)} features", flush=True)

        for threshold in THRESHOLDS:
            for horizon in HORIZONS:
                feat_df = add_targets(df2, threshold, horizon)
                train, test, cutoff = split(feat_df)

                if len(train) < 500 or len(test) < 100:
                    continue

                res   = run_rf(train, test, fcols)
                top   = res["top_feats"][0][0] if res["top_feats"] else "—"

                print(
                    f"  X={threshold:.0f}pt  H={horizon} ({horizon*2}min)  "
                    f"train={res['n_train']:>6,}  test={res['n_test']:>5,}  "
                    f"base={res['base_rate']*100:>5.1f}%  "
                    f"AUC={res['auc']:.3f}  "
                    f"prec={res['precision']:.3f}  rec={res['recall']:.3f}  "
                    f"top={top}",
                    flush=True,
                )

                all_results.append({
                    "lookback": lookback, "threshold": threshold, "horizon": horizon,
                    "auc": res["auc"], "precision": res["precision"],
                    "recall": res["recall"], "base_rate": res["base_rate"],
                    "n_train": res["n_train"], "n_test": res["n_test"],
                    "top_feat": top,
                })
        print(flush=True)

    if all_results:
        best = pd.DataFrame(all_results).sort_values("auc", ascending=False)
        print("── Top 10 by AUC ───────────────────────────────────────────────")
        pd.set_option("display.width", 130)
        print(best.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
