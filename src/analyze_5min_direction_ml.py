"""
analyze_5min_direction_ml.py — At every 5-min mark from 10:00 ET onward
(10:00, 10:05, 10:10, ...), using the day's data so far (trailing-window
stats + return/range since open), what's the probability MES closes
higher 5 minutes later?

Starts at 10:00 ET (not 9:30) per the hypothesis — already confirmed in
analyze_move_persistence.py — that moves in the first half hour mostly
revert, so there's little reason to expect a clean directional signal
there. Binary label (UP vs not), unlike the 5pt/3min project's
UP/DOWN/NEITHER — this is pure "higher or not" regardless of magnitude.

Reuses the merged ~2yr MES bar series (glbx "ellipsoid" pull + bars.db)
and the fetch_bars() cache from analyze_5pt_move_ml.py.

Usage:
  python src/analyze_5min_direction_ml.py
  python src/analyze_5min_direction_ml.py --walkforward
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

sys.path.insert(0, "src")
from analyze_5pt_move_ml import ET, fetch_bars  # noqa: E402

WINDOWS      = [5, 10, 15, 30]
STEP_MIN     = 5
START_HM     = 10 * 60        # 10:00 ET — skip the choppy opening half hour
RTH_START_HM = 9 * 60 + 30
CACHE_PATH   = Path("mes_5min_direction_cache.parquet")


def build_day_features(day: pd.DataFrame) -> pd.DataFrame:
    day = day.reset_index(drop=True)
    n = len(day)
    closes, opens, highs, lows = day["c"].values, day["o"].values, day["h"].values, day["l"].values
    hm = day["hm"].values
    log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))

    max_w = max(WINDOWS)
    rows = []
    for i in range(max_w, n - STEP_MIN):
        if hm[i] < START_HM or hm[i] % STEP_MIN != 0:
            continue
        if hm[i + STEP_MIN] - hm[i] != STEP_MIN:   # guard against any data gap
            continue

        feat = dict(ts=day["ts"].iloc[i], date=day["date"].iloc[i])
        feat["minutes_since_open"] = hm[i] - RTH_START_HM
        day_high_so_far = highs[:i + 1].max()
        day_low_so_far  = lows[:i + 1].min()
        feat["range_since_open"]   = day_high_so_far - day_low_so_far
        feat["dist_from_day_high"] = day_high_so_far - closes[i]
        feat["dist_from_day_low"]  = closes[i] - day_low_so_far
        feat["ret_since_open"]     = closes[i] - opens[0]

        for w in WINDOWS:
            win_ret   = log_ret[i - w + 1: i + 1]
            win_high  = highs[i - w + 1: i + 1]
            win_low   = lows[i - w + 1: i + 1]
            win_open  = opens[i - w + 1: i + 1]
            win_close = closes[i - w + 1: i + 1]
            feat[f"net_move_{w}"] = closes[i] - closes[i - w]
            feat[f"vol_{w}"]      = float(np.std(win_ret, ddof=1)) if w > 1 else 0.0
            feat[f"range_{w}"]    = float(win_high.max() - win_low.min())
            sum_abs = float(np.abs(win_ret).sum())
            feat[f"pl_{w}"]       = float(abs(win_ret.sum()) / sum_abs) if sum_abs > 0 else 0.0
            feat[f"up_frac_{w}"]  = float((win_close > win_open).mean())

        feat["label"] = int(closes[i + STEP_MIN] > closes[i])
        rows.append(feat)

    return pd.DataFrame(rows)


def load_data(rebuild: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not rebuild:
        print(f"Loading cached features from {CACHE_PATH}…", flush=True)
        return pd.read_parquet(CACHE_PATH)

    df = fetch_bars()
    print(f"Loaded {len(df)} MES 1-min RTH bars, {df['date'].nunique()} sessions "
          f"({df['date'].min()} -> {df['date'].max()})", flush=True)

    all_feats = []
    for date, day in df.groupby("date"):
        all_feats.append(build_day_features(day.sort_values("ts")))
    data = pd.concat(all_feats, ignore_index=True)
    data.to_parquet(CACHE_PATH)
    print(f"Built {len(data)} 5-min snapshots (10:00 ET onward), cached to {CACHE_PATH}", flush=True)
    return data


def run_fold(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], label: str = "") -> dict:
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(train[feature_cols], train["label"])
    proba = clf.predict_proba(test[feature_cols])[:, list(clf.classes_).index(1)]
    y_test = test["label"].values
    n_days = test["date"].nunique()
    base_rate = train["label"].mean()
    auc = roc_auc_score(y_test, proba) if len(set(y_test)) > 1 else float("nan")

    rows = []
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for side, mask in [("UP", proba >= thr), ("DOWN", proba <= (1 - thr))]:
            n = int(mask.sum())
            hit = float(y_test[mask].mean()) if n else None
            if side == "DOWN" and hit is not None:
                hit = 1 - hit   # hit = correctly predicted DOWN
            rows.append(dict(fold=label, side=side, threshold=thr, signals=n,
                             per_day=round(n / n_days, 2) if n_days else 0,
                             hit_rate=round(hit, 3) if hit is not None else None))
    return dict(rows=rows, auc=round(auc, 3), base_rate=round(base_rate, 3), n_days=n_days,
               train_range=(train["date"].min(), train["date"].max()),
               test_range=(test["date"].min(), test["date"].max()))


def main():
    rebuild = "--rebuild-cache" in sys.argv
    walkforward = "--walkforward" in sys.argv
    data = load_data(rebuild=rebuild)
    print(f"Overall UP base rate: {data['label'].mean():.4f}  n={len(data)}")

    feature_cols = [c for c in data.columns if c not in ("ts", "date", "label")]
    dates = sorted(data["date"].unique())

    if walkforward:
        n_folds = 5
        fold_size = len(dates) // (n_folds + 1)
        all_rows = []
        print(f"\n{'='*90}\nWALK-FORWARD: {n_folds} expanding-window folds, {len(dates)} days\n{'='*90}")
        for k in range(1, n_folds + 1):
            train_dates = set(dates[:fold_size * k])
            test_dates  = set(dates[fold_size * k: fold_size * (k + 1)])
            if not test_dates:
                break
            train = data[data["date"].isin(train_dates)]
            test  = data[data["date"].isin(test_dates)]
            result = run_fold(train, test, feature_cols, label=f"fold{k}")
            print(f"\nFold {k}: train {result['train_range'][0]}->{result['train_range'][1]} "
                  f"({len(train_dates)}d)  test {result['test_range'][0]}->{result['test_range'][1]} "
                  f"({result['n_days']}d)  AUC={result['auc']}  base_rate={result['base_rate']}")
            print(pd.DataFrame(result["rows"]).to_string(index=False))
            all_rows.extend(result["rows"])

        print(f"\n{'='*90}\nSUMMARY — threshold=0.60\n{'='*90}")
        summary = pd.DataFrame(all_rows)
        print(summary[summary["threshold"] == 0.60].to_string(index=False))
        return

    cut = int(len(dates) * 0.7)
    train_dates, test_dates = set(dates[:cut]), set(dates[cut:])
    train = data[data["date"].isin(train_dates)]
    test  = data[data["date"].isin(test_dates)]
    print(f"\nTrain: {len(train)} snapshots, {len(train_dates)} days "
          f"({min(train_dates)} -> {max(train_dates)})")
    print(f"Test:  {len(test)} snapshots, {len(test_dates)} days "
          f"({min(test_dates)} -> {max(test_dates)})")

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(train[feature_cols], train["label"])
    y_pred = clf.predict(test[feature_cols])
    print(f"\n{'='*70}\nCLASSIFICATION REPORT (test set)\n{'='*70}")
    print(classification_report(test["label"], y_pred, digits=3))
    proba = clf.predict_proba(test[feature_cols])[:, list(clf.classes_).index(1)]
    print(f"AUC: {roc_auc_score(test['label'], proba):.3f}")

    print(f"\n{'='*70}\nFEATURE IMPORTANCES (top 12)\n{'='*70}")
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(imp.head(12).round(4).to_string())


if __name__ == "__main__":
    main()
