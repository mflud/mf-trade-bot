"""
backtest_rf_feat_v3.py — Add volume and volatility features to RF model.

Compares feature sets on best config (2-min, lb=15, drop mins_since_open):
  A) Baseline  : r2m + r1m + ret_open
  B) + Volume  : + rel_vol_2m, vol_trend_2m, rel_vol_1m
  C) + Vol/Vol : + realized_vol, atr_ratio
  D) + All     : baseline + volume + vol/vol features

Volume features:
  rel_vol_2m     — current 2-min bar volume / 20-bar rolling mean (relative volume)
  vol_trend_2m   — log(vol_last_2m / vol_2m_avg_5bar)  (is volume picking up?)
  rel_vol_1m_1   — most recent 1-min bar relative volume

Volatility features:
  realized_vol   — std of last 10 2-min log returns (annualized not needed, raw)
  atr_ratio      — current 2-min range (high-low) / 10-bar avg range (is this bar wide?)
  vol_regime     — realized_vol percentile vs. 60-bar rolling window (0=calm, 1=volatile)

Targets: lb=15, X=5pt H=1 and X=3pt H=3 (best from prior runs).

Run from repo root:
    python src/backtest_rf_feat_v3.py
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
LOOKBACK   = 15
TRAIN_FRAC = 0.75
RTH_START  = time(9, 30)
RTH_END    = time(16, 0)

TARGETS = [
    (5.0, 1),
    (3.0, 3),
    (4.0, 1),
]

VOL_WINDOW   = 10   # bars for realized vol and ATR
REL_VOL_MEAN = 20   # bars for relative volume baseline
VOL_PCT_WIN  = 60   # bars for vol regime percentile


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

def build_dataset(df1: pd.DataFrame) -> pd.DataFrame:
    df1 = df1.copy()
    df1["date"] = df1["ts"].dt.date

    # 1-min log returns
    df1["lret1"] = np.log(df1["close"] / df1["close"].shift(1))
    df1.loc[df1["date"] != df1["date"].shift(1), "lret1"] = np.nan
    for lag in range(1, SHORT_N + 1):
        s = df1["lret1"].shift(lag)
        s[df1["date"] != df1["date"].shift(lag)] = np.nan
        df1[f"r1m_{lag}"] = s

    # 9:30 open per day
    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df1 = df1.join(open_930, on="date")
    df1["ret_open"] = np.log(df1["close"] / df1["open_930"])

    # 1-min relative volume (vs 20-bar rolling mean, same-day only)
    df1["vol_mean_1m"] = (df1.groupby("date")["volume"]
                            .transform(lambda x: x.shift(1).rolling(REL_VOL_MEAN, min_periods=5).mean()))
    df1["rel_vol_1m_1"] = np.log1p(df1["volume"].shift(1) / df1["vol_mean_1m"].clip(lower=1))

    # Resample to 2-min
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

    # 2-min log returns
    df2["lret2"] = np.log(df2["close"] / df2["close"].shift(1))
    df2.loc[df2["date"] != df2["date"].shift(1), "lret2"] = np.nan
    for lag in range(1, LOOKBACK):
        s = df2["lret2"].shift(lag)
        s[df2["date"] != df2["date"].shift(lag)] = np.nan
        df2[f"r2m_{lag}"] = s

    # ── Volume features on 2-min bars ────────────────────────────────────────
    # Relative volume: current / rolling mean (same-day)
    df2["vol_mean_2m"] = (df2.groupby("date")["volume"]
                            .transform(lambda x: x.shift(1).rolling(REL_VOL_MEAN, min_periods=5).mean()))
    df2["rel_vol_2m"] = np.log1p(df2["volume"] / df2["vol_mean_2m"].clip(lower=1))

    # Volume trend: recent 2-bar avg vs 10-bar avg (is vol accelerating?)
    vol_fast = df2.groupby("date")["volume"].transform(
        lambda x: x.shift(1).rolling(2, min_periods=2).mean())
    vol_slow = df2.groupby("date")["volume"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    df2["vol_trend_2m"] = np.log(vol_fast.clip(lower=1) / vol_slow.clip(lower=1))

    # ── Volatility features on 2-min bars ────────────────────────────────────
    # Realized vol: std of last VOL_WINDOW 2-min log returns
    df2["realized_vol"] = (df2.groupby("date")["lret2"]
                             .transform(lambda x: x.shift(1).rolling(VOL_WINDOW, min_periods=5).std()))

    # ATR ratio: current bar range vs. rolling avg range
    df2["range_2m"] = df2["high"] - df2["low"]
    avg_range = (df2.groupby("date")["range_2m"]
                   .transform(lambda x: x.shift(1).rolling(VOL_WINDOW, min_periods=5).mean()))
    df2["atr_ratio"] = df2["range_2m"] / avg_range.clip(lower=0.01)

    # Vol regime: percentile of realized_vol vs. rolling 60-bar window
    df2["vol_regime"] = (df2.groupby("date")["realized_vol"]
                           .transform(lambda x:
                               x.shift(1).rolling(VOL_PCT_WIN, min_periods=20)
                                .rank(pct=True)))

    # Merge 1-min features
    r1m_cols    = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols = ["ret_open", "rel_vol_1m_1"]
    df1_small   = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    df2 = df2.sort_values("ts")
    df2 = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    # Drop incomplete rows
    r2m_cols   = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    vol_cols   = ["rel_vol_2m", "vol_trend_2m"]
    vola_cols  = ["realized_vol", "atr_ratio", "vol_regime"]
    all_needed = r2m_cols + r1m_cols + ["ret_open"] + vol_cols + vola_cols + ["rel_vol_1m_1"]
    df2 = df2.dropna(subset=all_needed)
    df2 = df2[df2.groupby("date").cumcount() >= LOOKBACK * 2].reset_index(drop=True)
    df2["idx"] = df2.index
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


def fmt_imp(imp, cols):
    total = sum(imp.get(c, 0) for c in cols)
    return f"{total*100:5.1f}%"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars …", flush=True)
    df1 = load_rth_1min()
    print("\nBuilding features …", end=" ", flush=True)
    df2 = build_dataset(df1)
    print(f"{len(df2):,} bars\n", flush=True)

    r2m_cols  = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    r1m_cols  = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    vol_cols  = ["rel_vol_2m", "vol_trend_2m", "rel_vol_1m_1"]
    vola_cols = ["realized_vol", "atr_ratio", "vol_regime"]

    feature_sets = {
        "A_baseline": r2m_cols + r1m_cols + ["ret_open"],
        "B_+volume":  r2m_cols + r1m_cols + ["ret_open"] + vol_cols,
        "C_+volat":   r2m_cols + r1m_cols + ["ret_open"] + vola_cols,
        "D_+all":     r2m_cols + r1m_cols + ["ret_open"] + vol_cols + vola_cols,
    }

    all_results = []

    for threshold, horizon in TARGETS:
        feat_df = add_targets(df2, threshold, horizon)
        train, test = split(feat_df)
        if len(train) < 500 or len(test) < 100:
            continue

        print(f"── X={threshold:.0f}pt  H={horizon} ({horizon*2}min)  "
              f"base={feat_df['target'].mean()*100:.1f}%  "
              f"n_train={len(train):,}  n_test={len(test):,}")
        print(f"  {'Variant':<14}  {'AUC':>6}  {'Prec':>6}  {'Rec':>5}  "
              f"{'Δ AUC':>7}  "
              f"{'r2m':>5}  {'r1m':>5}  {'ret_op':>6}  "
              f"{'vol':>5}  {'vola':>5}")

        baseline_auc = None
        for label, fcols in feature_sets.items():
            res = run_rf(train, test, fcols)
            delta = f"{res['auc'] - baseline_auc:+.4f}" if baseline_auc else "   —   "
            if baseline_auc is None:
                baseline_auc = res["auc"]

            imp = res["imp"]
            r2m_tot  = fmt_imp(imp, r2m_cols)
            r1m_tot  = fmt_imp(imp, r1m_cols)
            ro_pct   = f"{imp.get('ret_open', 0)*100:5.1f}%"
            vol_tot  = fmt_imp(imp, vol_cols)
            vola_tot = fmt_imp(imp, vola_cols)

            print(f"  {label:<14}  {res['auc']:.4f}  {res['precision']:.4f}  "
                  f"{res['recall']:.4f}  {delta}  "
                  f"{r2m_tot}  {r1m_tot}  {ro_pct}  "
                  f"{vol_tot}  {vola_tot}",
                  flush=True)

            all_results.append({
                "X": threshold, "H": horizon, "variant": label,
                "auc": res["auc"], "precision": res["precision"],
                "recall": res["recall"],
            })

        # Print top features for D_+all
        print(f"\n  Top 12 features (D_+all):")
        feat_df2 = add_targets(df2, threshold, horizon)
        train2, test2 = split(feat_df2)
        res_d = run_rf(train2, test2, feature_sets["D_+all"])
        for name, val in sorted(res_d["imp"].items(), key=lambda x: -x[1])[:12]:
            bar = "█" * int(val * 400)
            print(f"    {name:<18} {val*100:5.2f}%  {bar}")
        print(flush=True)

    print("\n── Mean AUC by variant ─────────────────────────────────────────")
    df_res = pd.DataFrame(all_results)
    print(df_res.groupby("variant")["auc"].mean().sort_values(ascending=False).to_string())

    print("\n── Full results ────────────────────────────────────────────────")
    pd.set_option("display.width", 120)
    print(df_res.sort_values("auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
