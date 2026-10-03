"""
analyze_5pt_move_ml.py — Can trailing-window features (5/10/15min) predict
a 5pt+ MES move over the next 3 minutes, well enough to trade?

For each 1-min bar snapshot during RTH (09:30-16:00 ET), builds features
from trailing 5/10/15-min windows (net move, realized vol, range, price
linearity, up-bar fraction) plus time-since-open and position within the
day's range so far. Labels each snapshot UP / DOWN / NEITHER based on
whether price moves >=5pt in either direction over the next 3 minutes
(whichever threshold is crossed first, scanning bar-by-bar).

Trains a RandomForestClassifier on the first 70% of trading days
(chronological, not shuffled — these are autocorrelated minutes) and
evaluates on the last 30%. Reports feature importances (mechanism
sanity-check) and, more importantly, the trading-relevant question: at
various probability thresholds, how many signals/day and what fraction
actually hit (vs. the unconditional base rate).

Usage:
  python src/analyze_5pt_move_ml.py
"""

import sqlite3
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

ET = ZoneInfo("America/New_York")
MOVE_BP      = 6.0   # move threshold as basis points of price, not fixed points —
                      # keeps the task equally hard across the ~2900-7850 price
                      # range in this 7yr history (a fixed 5pt was ~17bp back at
                      # 2019 levels vs ~6.4bp today — ~3x "easier" — would have
                      # badly confounded the longer-history walk-forward otherwise)
MOVE_MIN_PTS = 2.0    # floor so the bp threshold never shrinks below a sane tick
                      # count at low historical price levels (6bp at 2900 would be
                      # only 1.74pt otherwise)
LOOKAHEAD  = 3    # minutes
WINDOWS    = [5, 10, 15]
RTH_START_HM = 9 * 60 + 30
RTH_END_HM   = 16 * 60

MES_HIST_PATH = Path("mes_hist_1min.csv")   # pre-existing continuous MES series, 2019-05 -> 2026-03
GLBX_PATH     = Path("glbx-mdp3-20240829-20260928.ohlcv-1m.csv")   # raw multi-symbol, "ellipsoid project" pull
GLBX_START_DATE  = pd.Timestamp("2024-08-29").date()
BARS_DB_MIN_DATE = pd.Timestamp("2026-04-08").date()   # bars.db's earliest MES date


def _fetch_hist_csv() -> pd.DataFrame:
    """mes_hist_1min.csv is already a continuous front-month series (verified
    tick-for-tick against glbx's MESU4 at the 2024-08-29 boundary — no gap, no
    back-adjustment offset needed). Used for dates before glbx's coverage starts."""
    if not MES_HIST_PATH.exists():
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
    df = pd.read_csv(MES_HIST_PATH)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.tz_convert(ET).dt.date
    df = df[df["date"] < GLBX_START_DATE]
    out = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    return out[["ts", "o", "h", "l", "c", "v"]].sort_values("ts").reset_index(drop=True)


def _fetch_glbx_mes_front_month() -> pd.DataFrame:
    """Continuous (non-back-adjusted) MES series from the raw glbx multi-symbol
    dump: for each calendar day, keep only the contract with the highest that-
    day volume (front month). No back-adjustment needed since every downstream
    feature/label here resets at day boundaries anyway, so a roll just lands
    on a natural session boundary rather than contaminating a trailing window."""
    if not GLBX_PATH.exists():
        return pd.DataFrame(columns=["ts", "o", "h", "l", "c", "v"])
    usecols = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]
    df = pd.read_csv(GLBX_PATH, usecols=usecols)
    df = df[df["symbol"].str.match(r"^MES[HMUZ]\d$", na=False)]
    df["ts"] = pd.to_datetime(df["ts_event"], utc=True)
    df["date"] = df["ts"].dt.tz_convert(ET).dt.date
    df = df[(df["date"] >= GLBX_START_DATE) & (df["date"] < BARS_DB_MIN_DATE)]

    daily_vol = df.groupby(["date", "symbol"])["volume"].sum()
    front = daily_vol.groupby("date").idxmax().apply(lambda x: x[1])   # date -> front-month symbol
    df = df[df["symbol"] == df["date"].map(front)]

    out = df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"})
    return out[["ts", "o", "h", "l", "c", "v"]].sort_values("ts").reset_index(drop=True)


RAW_BARS_CACHE = Path("mes_bars_merged_cache.parquet")   # hist_csv + glbx + bars.db, RTH only, ~7yr


def fetch_bars(rebuild: bool = False) -> pd.DataFrame:
    if RAW_BARS_CACHE.exists() and not rebuild:
        return pd.read_parquet(RAW_BARS_CACHE)

    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol='MES' AND minutes=1 ORDER BY ts"
    ).fetchall()
    conn.close()
    recent = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    recent["ts"] = pd.to_datetime(recent["t"], utc=True)
    recent = recent.drop(columns=["t"])

    oldest = _fetch_hist_csv()
    older  = _fetch_glbx_mes_front_month()
    df = pd.concat([oldest, older, recent], ignore_index=True)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)   # CME settlement gap
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df["et"] = df["ts"].dt.tz_convert(ET)
    df["hm"] = df["et"].dt.hour * 60 + df["et"].dt.minute
    df["date"] = df["et"].dt.date
    df = df[(df["hm"] >= RTH_START_HM) & (df["hm"] < RTH_END_HM)].reset_index(drop=True)
    df.to_parquet(RAW_BARS_CACHE)
    return df


def build_day_features(day: pd.DataFrame) -> pd.DataFrame:
    day = day.reset_index(drop=True)
    n = len(day)
    closes = day["c"].values
    opens  = day["o"].values
    highs  = day["h"].values
    lows   = day["l"].values
    log_ret = np.diff(np.log(closes), prepend=np.log(closes[0]))

    max_w = max(WINDOWS)
    rows = []
    for i in range(max_w, n - LOOKAHEAD):
        feat = dict(ts=day["ts"].iloc[i], date=day["date"].iloc[i])
        feat["minutes_since_open"] = day["hm"].iloc[i] - RTH_START_HM
        day_high_so_far = highs[:i + 1].max()
        day_low_so_far  = lows[:i + 1].min()
        feat["range_since_open"]    = day_high_so_far - day_low_so_far
        feat["dist_from_day_high"]  = day_high_so_far - closes[i]
        feat["dist_from_day_low"]   = closes[i] - day_low_so_far
        feat["ret_since_open"]      = closes[i] - opens[0]

        for w in WINDOWS:
            win_ret  = log_ret[i - w + 1: i + 1]
            win_high = highs[i - w + 1: i + 1]
            win_low  = lows[i - w + 1: i + 1]
            win_open = opens[i - w + 1: i + 1]
            win_close = closes[i - w + 1: i + 1]
            feat[f"net_move_{w}"]  = closes[i] - closes[i - w]
            feat[f"vol_{w}"]       = float(np.std(win_ret, ddof=1)) if w > 1 else 0.0
            feat[f"range_{w}"]     = float(win_high.max() - win_low.min())
            sum_abs = float(np.abs(win_ret).sum())
            feat[f"pl_{w}"]        = float(abs(win_ret.sum()) / sum_abs) if sum_abs > 0 else 0.0
            feat[f"up_frac_{w}"]   = float((win_close > win_open).mean())

        # ── Label: first >=move_pts threshold crossed in the next LOOKAHEAD bars ──
        entry = closes[i]
        move_pts = max(entry * MOVE_BP / 10000.0, MOVE_MIN_PTS)
        feat["move_pts"] = move_pts
        running_high, running_low = entry, entry
        label = "NEITHER"
        for j in range(i + 1, i + 1 + LOOKAHEAD):
            running_high = max(running_high, highs[j])
            running_low  = min(running_low, lows[j])
            if running_high - entry >= move_pts:
                label = "UP"; break
            if entry - running_low >= move_pts:
                label = "DOWN"; break
        feat["label"] = label
        rows.append(feat)

    return pd.DataFrame(rows)


CACHE_PATH = Path("mes_5pt_features_cache.parquet")


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
    print(f"Built {len(data)} snapshots, cached to {CACHE_PATH}", flush=True)
    return data


def run_fold(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str],
            label: str = "") -> dict:
    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(train[feature_cols], train["label"])
    proba = clf.predict_proba(test[feature_cols])
    classes = list(clf.classes_)
    p_up   = proba[:, classes.index("UP")]
    p_down = proba[:, classes.index("DOWN")]
    base_up   = (train["label"] == "UP").mean()
    base_down = (train["label"] == "DOWN").mean()
    n_days = test["date"].nunique()

    rows = []
    for thr in [0.4, 0.5, 0.6]:
        for side, p, base in [("UP", p_up, base_up), ("DOWN", p_down, base_down)]:
            mask = p >= thr
            n = int(mask.sum())
            hit = float((test["label"][mask] == side).mean()) if n else None
            rows.append(dict(fold=label, side=side, threshold=thr, signals=n,
                             per_day=round(n / n_days, 2) if n_days else 0,
                             hit_rate=round(hit, 3) if hit is not None else None,
                             lift=round(hit / base, 2) if (hit is not None and base > 0) else None))
    return dict(rows=rows, n_days=n_days,
               train_range=(train["date"].min(), train["date"].max()),
               test_range=(test["date"].min(), test["date"].max()))


def main():
    rebuild = "--rebuild-cache" in sys.argv
    walkforward = "--walkforward" in sys.argv
    data = load_data(rebuild=rebuild)
    print(data["label"].value_counts(normalize=True).rename("base_rate").round(4))

    if walkforward:
        dates = sorted(data["date"].unique())
        n_folds = 5
        fold_size = len(dates) // (n_folds + 1)   # first fold's worth = initial train
        feature_cols = [c for c in data.columns if c not in ("ts", "date", "label", "move_pts")]
        all_rows = []
        print(f"\n{'='*90}\nWALK-FORWARD: {n_folds} expanding-window folds, "
              f"{len(dates)} total days\n{'='*90}")
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
                  f"({result['n_days']}d)")
            print(pd.DataFrame(result["rows"]).to_string(index=False))
            all_rows.extend(result["rows"])

        print(f"\n{'='*90}\nSUMMARY ACROSS FOLDS — threshold=0.5\n{'='*90}")
        summary = pd.DataFrame(all_rows)
        summary = summary[summary["threshold"] == 0.5]
        print(summary.to_string(index=False))
        return

    dates = sorted(data["date"].unique())
    cut = int(len(dates) * 0.7)
    train_dates, test_dates = set(dates[:cut]), set(dates[cut:])
    train = data[data["date"].isin(train_dates)]
    test  = data[data["date"].isin(test_dates)]
    print(f"\nTrain: {len(train)} snapshots, {len(train_dates)} days "
          f"({min(train_dates)} -> {max(train_dates)})")
    print(f"Test:  {len(test)} snapshots, {len(test_dates)} days "
          f"({min(test_dates)} -> {max(test_dates)})")

    feature_cols = [c for c in data.columns if c not in ("ts", "date", "label", "move_pts")]
    X_train, y_train = train[feature_cols], train["label"]
    X_test,  y_test  = test[feature_cols],  test["label"]

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print(f"\n{'='*70}\nCLASSIFICATION REPORT (test set)\n{'='*70}")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (rows=actual, cols=predicted), classes:", clf.classes_)
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

    print(f"\n{'='*70}\nFEATURE IMPORTANCES (top 12)\n{'='*70}")
    imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(imp.head(12).round(4).to_string())

    # ── Trading-relevant question: signals/day and hit rate at various thresholds ──
    proba = clf.predict_proba(X_test)
    classes = list(clf.classes_)
    up_idx, down_idx = classes.index("UP"), classes.index("DOWN")
    test = test.copy()
    test["p_up"] = proba[:, up_idx]
    test["p_down"] = proba[:, down_idx]
    n_test_days = len(test_dates)

    base_up   = (train["label"] == "UP").mean()
    base_down = (train["label"] == "DOWN").mean()
    print(f"\n{'='*70}\nSIGNAL THRESHOLD ANALYSIS (test set, {n_test_days} days)\n{'='*70}")
    print(f"Base rate (train): P(UP)={base_up:.3f}  P(DOWN)={base_down:.3f}")
    print(f"{'threshold':>10} {'side':>5} {'signals':>8} {'per_day':>8} {'hit_rate':>9} {'lift_x':>7}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for side, p_col, base in [("UP", "p_up", base_up), ("DOWN", "p_down", base_down)]:
            sig = test[test[p_col] >= thr]
            if len(sig) == 0:
                continue
            hit_rate = (sig["label"] == side).mean()
            lift = hit_rate / base if base > 0 else float("nan")
            print(f"{thr:>10.2f} {side:>5} {len(sig):>8d} {len(sig)/n_test_days:>8.2f} "
                  f"{hit_rate:>9.3f} {lift:>7.2f}")


if __name__ == "__main__":
    main()
