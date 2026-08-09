"""
backtest_rf_feat_v2.py — Feature ablation: drop mins_since_open, add ret_open².

Compares three feature sets on the best config (2-min, lb=15, X=5pt H=1 and X=3pt H=3):
  A) Baseline  : r2m + r1m + ret_open + mins_since_open
  B) Drop TOD  : r2m + r1m + ret_open                    (drop mins_since_open)
  C) + ret_open²: r2m + r1m + ret_open + ret_open²        (add squared term)

Also tests on lb=10 for comparison.

Run from repo root:
    python src/backtest_rf_feat_v2.py
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

SHORT_N    = 6
TRAIN_FRAC = 0.75
RTH_START  = time(9, 30)
RTH_END    = time(16, 0)

CONFIGS = [
    (15, 5.0, 1),
    (15, 3.0, 3),
    (10, 5.0, 1),
    (10, 3.0, 3),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_rth_1min() -> pd.DataFrame:
    frames = []
    hist = Path(HIST_CSV)
    if hist.exists():
        df_h = pd.read_csv(hist, parse_dates=["ts"])
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= RTH_START) & (t < RTH_END)]
        frames.append(df_h)
    db = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql(
        "SELECT ts, open, high, low, close, volume "
        "FROM bars WHERE symbol='MES' AND minutes=1 ORDER BY ts", db)
    db.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t = df_db["ts"].dt.time
    df_db = df_db[(t >= RTH_START) & (t < RTH_END)]
    frames.append(df_db)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True))
    print(f"  Loaded {len(df):,} RTH 1-min bars  "
          f"({df['ts'].dt.date.min()} → {df['ts'].dt.date.max()})", flush=True)
    return df


# ── Feature engineering ───────────────────────────────────────────────────────

def build_dataset(df1: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df1 = df1.copy()
    df1["date"] = df1["ts"].dt.date

    df1["lret1"] = np.log(df1["close"] / df1["close"].shift(1))
    df1.loc[df1["date"] != df1["date"].shift(1), "lret1"] = np.nan
    for lag in range(1, SHORT_N + 1):
        s = df1["lret1"].shift(lag)
        s[df1["date"] != df1["date"].shift(lag)] = np.nan
        df1[f"r1m_{lag}"] = s

    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df1 = df1.join(open_930, on="date")
    df1["ret_open"]       = np.log(df1["close"] / df1["open_930"])
    df1["ret_open_sq"]    = df1["ret_open"] ** 2
    df1["mins_since_open"] = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute
                               - (9 * 60 + 30))

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

    df2["lret2"] = np.log(df2["close"] / df2["close"].shift(1))
    df2.loc[df2["date"] != df2["date"].shift(1), "lret2"] = np.nan
    for lag in range(1, lookback):
        s = df2["lret2"].shift(lag)
        s[df2["date"] != df2["date"].shift(lag)] = np.nan
        df2[f"r2m_{lag}"] = s

    r1m_cols     = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols  = ["ret_open", "ret_open_sq", "mins_since_open"]
    df1_small    = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    df2          = df2.sort_values("ts")
    df2          = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    r2m_cols    = [f"r2m_{i}" for i in range(1, lookback)]
    all_feat    = r2m_cols + r1m_cols + scalar_cols
    df2         = df2.dropna(subset=all_feat)
    df2         = df2[df2["mins_since_open"] >= lookback * 2].reset_index(drop=True)
    df2["idx"]  = df2.index
    return df2


def add_targets(df2: pd.DataFrame, threshold: float, horizon: int) -> pd.DataFrame:
    highs  = df2["high"].values
    lows   = df2["low"].values
    closes = df2["close"].values
    dates  = df2["date"].values
    n      = len(df2)
    targets = np.full(n, np.nan)
    for i in range(n - horizon):
        fd = dates[i + 1 : i + 1 + horizon]
        if len(fd) < horizon or fd[0] != fd[-1] or fd[0] != dates[i]:
            continue
        targets[i] = int(highs[i+1:i+1+horizon].max() >= closes[i] + threshold or
                         lows[i+1:i+1+horizon].min()  <= closes[i] - threshold)
    df2 = df2.copy()
    df2["target"] = targets
    return df2.dropna(subset=["target"])


def split(df: pd.DataFrame):
    dates  = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRAC)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def run_rf(train, test, fcols):
    X_tr, y_tr = train[fcols].values, train["target"].astype(int).values
    X_te, y_te = test[fcols].values,  test["target"].astype(int).values
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else np.nan
    prec  = precision_score(y_te, pred, zero_division=0)
    rec   = recall_score(y_te, pred, zero_division=0)
    imp   = dict(zip(fcols, clf.feature_importances_))
    return {"auc": auc, "precision": prec, "recall": rec,
            "base_rate": y_te.mean(), "n_train": len(train),
            "n_test": len(test), "imp": imp}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars …", flush=True)
    df1 = load_rth_1min()
    print(flush=True)

    cache = {}
    all_results = []

    for lookback, threshold, horizon in CONFIGS:
        if lookback not in cache:
            print(f"── lb={lookback} (30 min back) — building features …",
                  end=" ", flush=True)
            cache[lookback] = build_dataset(df1, lookback)
            print(f"{len(cache[lookback]):,} bars", flush=True)

        df2 = cache[lookback]
        feat_df = add_targets(df2, threshold, horizon)
        train, test = split(feat_df)
        if len(train) < 500 or len(test) < 100:
            continue

        r2m_cols = [f"r2m_{i}" for i in range(1, lookback)]
        r1m_cols = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]

        feature_sets = {
            "A_baseline":    r2m_cols + r1m_cols + ["ret_open", "mins_since_open"],
            "B_drop_tod":    r2m_cols + r1m_cols + ["ret_open"],
            "C_add_sq":      r2m_cols + r1m_cols + ["ret_open", "ret_open_sq"],
        }

        print(f"\n  lb={lookback}  X={threshold:.0f}pt  H={horizon} ({horizon*2}min)  "
              f"base={feat_df['target'].mean()*100:.1f}%  "
              f"n_train={len(train):,}  n_test={len(test):,}")
        print(f"  {'Variant':<16}  {'AUC':>6}  {'Prec':>6}  {'Rec':>5}  "
              f"{'Δ AUC':>7}  ret_open%  ret_open_sq%  mins%")

        baseline_auc = None
        for label, fcols in feature_sets.items():
            res = run_rf(train, test, fcols)
            delta = f"{res['auc'] - baseline_auc:+.4f}" if baseline_auc else "   —   "
            if baseline_auc is None:
                baseline_auc = res["auc"]

            imp = res["imp"]
            ro_pct  = imp.get("ret_open",    0) * 100
            sq_pct  = imp.get("ret_open_sq", 0) * 100
            tod_pct = imp.get("mins_since_open", 0) * 100

            print(f"  {label:<16}  {res['auc']:.4f}  {res['precision']:.4f}  "
                  f"{res['recall']:.4f}  {delta}  "
                  f"{ro_pct:>8.2f}%  {sq_pct:>11.2f}%  {tod_pct:>4.2f}%",
                  flush=True)

            all_results.append({
                "lb": lookback, "X": threshold, "H": horizon,
                "variant": label, "auc": res["auc"],
                "precision": res["precision"], "recall": res["recall"],
            })

    print("\n\n── Summary table (sorted by AUC) ──────────────────────────────")
    df_res = pd.DataFrame(all_results).sort_values("auc", ascending=False)
    pd.set_option("display.width", 120)
    print(df_res.to_string(index=False))

    print("\n── Mean AUC by variant ─────────────────────────────────────────")
    print(df_res.groupby("variant")["auc"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
