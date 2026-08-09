"""
backtest_rf_feat_importance.py — Feature importance analysis + bar-interval sweep.

Two analyses:
  1. Feature group importance breakdown for the best 2-min configuration
     (lb=15, X=5pt, H=1 and lb=15, X=3pt, H=3)
     Shows individual feature importances and group totals.

  2. Bar interval sweep: repeat best lookback (15 bars) across 2-min, 3-min, 4-min
     bars, using best targets from prior run. Compare AUC across intervals.

Run from repo root:
    python src/backtest_rf_feat_importance.py
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

# Configs to sweep: (bar_minutes, lookback, threshold, horizon)
SWEEP = [
    # 2-min (best from prior run)
    (2, 15, 5.0, 1),
    (2, 15, 3.0, 3),
    (2, 15, 4.0, 1),
    (2, 10, 5.0, 1),
    # 3-min
    (3, 10, 5.0, 1),
    (3, 10, 3.0, 2),
    (3, 10, 4.0, 1),
    (3,  7, 5.0, 1),
    # 4-min
    (4,  8, 5.0, 1),
    (4,  8, 3.0, 2),
    (4,  8, 4.0, 1),
    (4,  5, 5.0, 1),
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

def build_dataset(df1: pd.DataFrame, bar_min: int, lookback: int) -> pd.DataFrame:
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
    df1["ret_open"]       = np.log(df1["close"] / df1["open_930"])
    df1["mins_since_open"] = (df1["ts"].dt.hour * 60 + df1["ts"].dt.minute
                               - (9 * 60 + 30))

    # Resample to bar_min
    df1_idx = df1.set_index("ts")
    dfN = df1_idx[["open","high","low","close","volume"]].resample(
        f"{bar_min}min", label="right", closed="right"
    ).agg(open=("open","first"), high=("high","max"),
          low=("low","min"),   close=("close","last"),
          volume=("volume","sum")).dropna(subset=["close"])
    dfN.index = dfN.index.tz_convert(ET)
    tN = dfN.index.time
    dfN = dfN[(tN >= RTH_START) & (tN < RTH_END)].reset_index()
    dfN["date"] = dfN["ts"].dt.date

    # N-min log returns
    dfN["lretN"] = np.log(dfN["close"] / dfN["close"].shift(1))
    dfN.loc[dfN["date"] != dfN["date"].shift(1), "lretN"] = np.nan

    for lag in range(1, lookback):
        s = dfN["lretN"].shift(lag)
        s[dfN["date"] != dfN["date"].shift(lag)] = np.nan
        dfN[f"rNm_{lag}"] = s

    # Merge 1-min features onto N-min grid
    r1m_cols = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols = ["ret_open", "mins_since_open"]
    df1_small = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    dfN = dfN.sort_values("ts")
    dfN = pd.merge_asof(dfN, df1_small, on="ts", direction="backward")

    rNm_cols = [f"rNm_{i}" for i in range(1, lookback)]
    all_feat = rNm_cols + r1m_cols + scalar_cols
    dfN = dfN.dropna(subset=all_feat)
    dfN = dfN[dfN["mins_since_open"] >= lookback * bar_min].reset_index(drop=True)
    dfN["idx"] = dfN.index
    return dfN


def add_targets(dfN: pd.DataFrame, threshold: float, horizon: int) -> pd.DataFrame:
    highs  = dfN["high"].values
    lows   = dfN["low"].values
    closes = dfN["close"].values
    dates  = dfN["date"].values
    n = len(dfN)
    targets = np.full(n, np.nan)
    for i in range(n - horizon):
        fwd = slice(i + 1, i + 1 + horizon)
        fd  = dates[fwd]
        if len(fd) < horizon or fd[0] != fd[-1] or fd[0] != dates[i]:
            continue
        targets[i] = int(highs[fwd].max() >= closes[i] + threshold or
                         lows[fwd].min()  <= closes[i] - threshold)
    dfN = dfN.copy()
    dfN["target"] = targets
    return dfN.dropna(subset=["target"])


def split(df: pd.DataFrame):
    dates  = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRAC)]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def get_fcols(lookback: int) -> list[str]:
    return ([f"rNm_{i}" for i in range(1, lookback)]
            + [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
            + ["ret_open", "mins_since_open"])


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
    return {"n_train": len(train), "n_test": len(test),
            "base_rate": y_te.mean(), "auc": auc,
            "precision": prec, "recall": rec, "imp": imp}


def print_importance(imp: dict, lookback: int, bar_min: int):
    """Print individual importances and group totals."""
    rNm_cols    = [f"rNm_{i}" for i in range(1, lookback)]
    r1m_cols    = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]

    total = sum(imp.values())

    # Group totals
    grp_Nm  = sum(imp.get(c, 0) for c in rNm_cols)
    grp_1m  = sum(imp.get(c, 0) for c in r1m_cols)
    grp_open = imp.get("ret_open", 0)
    grp_tod  = imp.get("mins_since_open", 0)

    print(f"\n  Feature group totals:")
    print(f"    {bar_min}-min returns ({len(rNm_cols)} feats): {grp_Nm*100:5.1f}%")
    print(f"    1-min returns  ({len(r1m_cols)} feats): {grp_1m*100:5.1f}%")
    print(f"    ret_open                  : {grp_open*100:5.1f}%")
    print(f"    mins_since_open           : {grp_tod*100:5.1f}%")

    # Top 10 individual features
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
    print(f"\n  Top 10 individual features:")
    for name, val in sorted_imp[:10]:
        bar = "█" * int(val / total * 40)
        print(f"    {name:<18} {val*100:5.2f}%  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars …", flush=True)
    df1 = load_rth_1min()
    print(flush=True)

    # Cache built datasets to avoid rebuilding per target
    cache: dict[tuple, pd.DataFrame] = {}

    all_results = []

    print("═" * 80)
    print("SWEEP: bar interval × lookback × threshold × horizon")
    print("═" * 80)

    prev_bar_lb = None

    for bar_min, lookback, threshold, horizon in SWEEP:
        key = (bar_min, lookback)
        if key not in cache:
            print(f"\n── {bar_min}-min bars, lb={lookback} ({lookback*bar_min} min back) ──",
                  flush=True)
            print("   Building features …", end=" ", flush=True)
            cache[key] = build_dataset(df1, bar_min, lookback)
            print(f"{len(cache[key]):,} bars", flush=True)
        dfN = cache[key]
        fcols = get_fcols(lookback)

        feat_df = add_targets(dfN, threshold, horizon)
        train, test = split(feat_df)
        if len(train) < 500 or len(test) < 100:
            continue

        res = run_rf(train, test, fcols)
        top3 = sorted(res["imp"].items(), key=lambda x: -x[1])[:3]
        top_names = ", ".join(f"{n}({v*100:.1f}%)" for n, v in top3)

        label = f"{bar_min}min lb={lookback} X={threshold:.0f}pt H={horizon}"
        print(
            f"  {label:<28}  "
            f"train={res['n_train']:>6,}  test={res['n_test']:>5,}  "
            f"base={res['base_rate']*100:>5.1f}%  "
            f"AUC={res['auc']:.3f}  "
            f"prec={res['precision']:.3f}  rec={res['recall']:.3f}",
            flush=True,
        )

        # Print importance for the "clean" reference case
        is_reference = (bar_min == 2 and lookback == 15 and
                        threshold == 5.0 and horizon == 1)
        is_reference |= (bar_min == 2 and lookback == 15 and
                         threshold == 3.0 and horizon == 3)
        is_reference |= (bar_min == 3 and lookback == 10 and
                         threshold == 5.0 and horizon == 1)
        is_reference |= (bar_min == 4 and lookback == 8 and
                         threshold == 5.0 and horizon == 1)
        if is_reference:
            print_importance(res["imp"], lookback, bar_min)
            print(flush=True)

        all_results.append({
            "bar_min": bar_min, "lookback": lookback,
            "threshold": threshold, "horizon": horizon,
            "auc": res["auc"], "precision": res["precision"],
            "recall": res["recall"], "base_rate": res["base_rate"],
            "n_train": res["n_train"], "n_test": res["n_test"],
            "top_feat": top3[0][0] if top3 else "—",
        })

    print("\n\n" + "═" * 80)
    print("TOP RESULTS BY AUC")
    print("═" * 80)
    best = pd.DataFrame(all_results).sort_values("auc", ascending=False)
    pd.set_option("display.width", 130)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
