"""
backtest_rf_vol_regime.py — Separate RF models by volatility regime.

Uses the best feature set from v3 (2-min lb=15, r2m + r1m + ret_open +
realized_vol + atr_ratio + vol_regime) and splits bars into three regimes
based on realized_vol percentile:
  Low    : vol_regime < 0.33
  Medium : 0.33 <= vol_regime < 0.67
  High   : vol_regime >= 0.67

For each regime, trains a separate RF and compares:
  - AUC within regime vs. combined model on same subset
  - Base rate shifts across regimes
  - Feature importance differences across regimes

Targets: X=5pt H=1 and X=3pt H=3 (best from prior runs).

Run from repo root:
    python src/backtest_rf_vol_regime.py
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

SHORT_N      = 6
LOOKBACK     = 15
TRAIN_FRAC   = 0.75
RTH_START    = time(9, 30)
RTH_END      = time(16, 0)
VOL_WINDOW   = 10
VOL_PCT_WIN  = 60

TARGETS = [(5.0, 1), (3.0, 3), (4.0, 1)]

REGIME_CUTS = [0.0, 0.33, 0.67, 1.01]
REGIME_NAMES = ["Low vol", "Med vol", "High vol"]


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

    df1["lret1"] = np.log(df1["close"] / df1["close"].shift(1))
    df1.loc[df1["date"] != df1["date"].shift(1), "lret1"] = np.nan
    for lag in range(1, SHORT_N + 1):
        s = df1["lret1"].shift(lag)
        s[df1["date"] != df1["date"].shift(lag)] = np.nan
        df1[f"r1m_{lag}"] = s

    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df1 = df1.join(open_930, on="date")
    df1["ret_open"] = np.log(df1["close"] / df1["open_930"])

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
    for lag in range(1, LOOKBACK):
        s = df2["lret2"].shift(lag)
        s[df2["date"] != df2["date"].shift(lag)] = np.nan
        df2[f"r2m_{lag}"] = s

    # Volatility features
    df2["realized_vol"] = (df2.groupby("date")["lret2"]
                             .transform(lambda x: x.shift(1)
                                        .rolling(VOL_WINDOW, min_periods=5).std()))
    df2["range_2m"] = df2["high"] - df2["low"]
    avg_range = (df2.groupby("date")["range_2m"]
                   .transform(lambda x: x.shift(1)
                              .rolling(VOL_WINDOW, min_periods=5).mean()))
    df2["atr_ratio"] = df2["range_2m"] / avg_range.clip(lower=0.01)
    df2["vol_regime"] = (df2.groupby("date")["realized_vol"]
                           .transform(lambda x:
                               x.shift(1).rolling(VOL_PCT_WIN, min_periods=20)
                                .rank(pct=True)))

    r1m_cols    = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols = ["ret_open"]
    df1_small   = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    df2 = df2.sort_values("ts")
    df2 = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    r2m_cols  = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    vola_cols = ["realized_vol", "atr_ratio", "vol_regime"]
    all_need  = r2m_cols + r1m_cols + ["ret_open"] + vola_cols
    df2 = df2.dropna(subset=all_need)
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
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return None
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_te, proba)
    prec  = precision_score(y_te, pred, zero_division=0)
    rec   = recall_score(y_te, pred, zero_division=0)
    imp   = dict(zip(fcols, clf.feature_importances_))
    return {"auc": auc, "precision": prec, "recall": rec,
            "base_rate": y_te.mean(), "n_train": len(train),
            "n_test": len(test), "imp": imp}


def print_top_feats(imp: dict, n: int = 8):
    for name, val in sorted(imp.items(), key=lambda x: -x[1])[:n]:
        bar = "█" * int(val * 300)
        print(f"      {name:<18} {val*100:5.2f}%  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars …", flush=True)
    df1 = load_rth_1min()
    print("\nBuilding features …", end=" ", flush=True)
    df2 = build_dataset(df1)
    print(f"{len(df2):,} 2-min bars\n", flush=True)

    fcols_all = ([f"r2m_{i}" for i in range(1, LOOKBACK)]
                 + [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
                 + ["ret_open", "realized_vol", "atr_ratio", "vol_regime"])

    # For regime-specific models, drop vol_regime (it's constant within regime)
    fcols_regime = ([f"r2m_{i}" for i in range(1, LOOKBACK)]
                    + [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
                    + ["ret_open", "realized_vol", "atr_ratio"])

    all_results = []

    for threshold, horizon in TARGETS:
        feat_df = add_targets(df2, threshold, horizon)
        print(f"═══════════════════════════════════════════════════════════════")
        print(f"X={threshold:.0f}pt  H={horizon} ({horizon*2}min)  "
              f"overall base={feat_df['target'].mean()*100:.1f}%  "
              f"n={len(feat_df):,}")
        print(f"═══════════════════════════════════════════════════════════════\n")

        # ── Combined model (baseline) ─────────────────────────────────────
        train_all, test_all = split(feat_df)
        res_combined = run_rf(train_all, test_all, fcols_all)
        print(f"  COMBINED model (all regimes):  "
              f"AUC={res_combined['auc']:.4f}  "
              f"prec={res_combined['precision']:.4f}  "
              f"rec={res_combined['recall']:.4f}  "
              f"base={res_combined['base_rate']*100:.1f}%  "
              f"n_test={res_combined['n_test']:,}\n")

        # ── Per-regime models ─────────────────────────────────────────────
        print(f"  {'Regime':<10}  {'n_train':>7}  {'n_test':>6}  "
              f"{'base%':>6}  {'AUC':>6}  {'Δ AUC':>7}  "
              f"{'Prec':>6}  {'Rec':>5}  "
              f"avg_rvol")

        for ri, rname in enumerate(REGIME_NAMES):
            lo = REGIME_CUTS[ri]
            hi = REGIME_CUTS[ri + 1]
            mask = (feat_df["vol_regime"] >= lo) & (feat_df["vol_regime"] < hi)
            sub  = feat_df[mask]

            avg_rvol = sub["realized_vol"].mean()

            train_r, test_r = split(sub)
            if len(train_r) < 300 or len(test_r) < 100:
                print(f"  {rname:<10}  (insufficient data)", flush=True)
                continue

            res_r = run_rf(train_r, test_r, fcols_regime)
            if res_r is None:
                continue

            delta = res_r["auc"] - res_combined["auc"]

            print(
                f"  {rname:<10}  {res_r['n_train']:>7,}  {res_r['n_test']:>6,}  "
                f"{res_r['base_rate']*100:>6.1f}%  "
                f"{res_r['auc']:>6.4f}  {delta:>+7.4f}  "
                f"{res_r['precision']:>6.4f}  {res_r['recall']:>5.4f}  "
                f"{avg_rvol*100:.4f}%",
                flush=True,
            )

            all_results.append({
                "threshold": threshold, "horizon": horizon,
                "regime": rname, "lo": lo, "hi": hi,
                "auc": res_r["auc"], "auc_delta": delta,
                "precision": res_r["precision"], "recall": res_r["recall"],
                "base_rate": res_r["base_rate"],
                "n_train": res_r["n_train"], "n_test": res_r["n_test"],
                "avg_rvol": avg_rvol,
                "imp": res_r["imp"],
            })

        # ── Feature importance comparison across regimes ───────────────────
        print(f"\n  Feature importance by regime (top 8):")
        relevant = [r for r in all_results
                    if r["threshold"] == threshold and r["horizon"] == horizon]
        for r in relevant:
            print(f"\n  [{r['regime']}]  AUC={r['auc']:.4f}  "
                  f"base={r['base_rate']*100:.1f}%  "
                  f"avg realized_vol={r['avg_rvol']*100:.4f}%")
            print_top_feats(r["imp"])

        # ── Combined model importance for comparison ─────────────────────
        print(f"\n  [Combined]  AUC={res_combined['auc']:.4f}")
        print_top_feats(res_combined["imp"])
        print(flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n\n── AUC by regime (mean across targets) ─────────────────────────")
    df_res = pd.DataFrame([{k: v for k, v in r.items() if k != "imp"}
                            for r in all_results])
    summary = (df_res.groupby("regime")[["auc", "auc_delta", "base_rate"]]
               .mean().sort_values("auc", ascending=False))
    print(summary.to_string())

    print("\n── Full results ────────────────────────────────────────────────")
    pd.set_option("display.width", 130)
    print(df_res.sort_values(["threshold","horizon","regime"])
               .drop(columns=["lo","hi"])
               .to_string(index=False))


if __name__ == "__main__":
    main()
