"""
backtest_rf_direction.py — Predict continuation vs. reversal given existing move.

For each 2-min bar where price has moved >= move_threshold pts over the prior
move_lookback bars, predict whether price continues in that direction (hits
target_pts) before reversing (hits stop_pts), within horizon bars.

This is a directional signal conditioned on momentum already being present:
  - LONG signal: close is >= X pts above close N bars ago → predict continuation
  - SHORT signal: close is >= X pts below close N bars ago → predict continuation
  The features are flipped for SHORT (returns negated) so the model learns
  one unified "continuation" concept regardless of direction.

Feature set (best from v3):
  r2m_1..14  — 2-min log returns (signed relative to move direction)
  r1m_1..6   — 1-min log returns (signed)
  ret_open   — log return since 9:30 open (signed)
  realized_vol, atr_ratio, vol_regime  — volatility features (unsigned)
  move_size  — magnitude of existing move in pts
  move_bars  — number of bars over which the move occurred

Sweep:
  move_threshold ∈ {2, 3, 4} pts
  move_lookback  ∈ {1, 2, 3} bars
  target_pts / stop_pts ∈ {(2,2), (2,4), (3,3), (4,2)}  [tgt:stop]
  horizon = 6 bars (12 min)

Run from repo root:
    python src/backtest_rf_direction.py
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import time
from zoneinfo import ZoneInfo
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, brier_score_loss

ET = ZoneInfo("America/New_York")
DB_PATH  = "data/bars.db"
HIST_CSV = "mes_hist_1min.csv"

SHORT_N      = 6
LOOKBACK     = 15
TRAIN_FRAC   = 0.75
RTH_START    = time(9, 30)
RTH_END      = time(16, 0)
HORIZON      = 6       # max bars to wait for target or stop

VOL_WINDOW   = 10
REL_VOL_MEAN = 20
VOL_PCT_WIN  = 60

MOVE_THRESHOLDS = [2.0, 3.0, 4.0]
MOVE_LOOKBACKS  = [1, 2, 3]
RISK_REWARDS    = [
    (2, 2),   # 1:1
    (2, 4),   # 1:2 (tight stop, wide target)
    (3, 3),   # 1:1 wider
    (4, 2),   # 2:1 (wide stop, tight target — fade-friendly)
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

def build_base_features(df1: pd.DataFrame) -> pd.DataFrame:
    """Build 2-min bar dataset with all features (unsigned — direction applied later)."""
    df1 = df1.copy()
    df1["date"] = df1["ts"].dt.date

    # 1-min log returns
    df1["lret1"] = np.log(df1["close"] / df1["close"].shift(1))
    df1.loc[df1["date"] != df1["date"].shift(1), "lret1"] = np.nan
    for lag in range(1, SHORT_N + 1):
        s = df1["lret1"].shift(lag)
        s[df1["date"] != df1["date"].shift(lag)] = np.nan
        df1[f"r1m_{lag}"] = s

    # 9:30 open
    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df1 = df1.join(open_930, on="date")
    df1["ret_open"] = np.log(df1["close"] / df1["open_930"])

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

    # Volatility features
    df2["realized_vol"] = (df2.groupby("date")["lret2"]
                             .transform(lambda x: x.shift(1).rolling(VOL_WINDOW, min_periods=5).std()))
    df2["range_2m"] = df2["high"] - df2["low"]
    avg_range = (df2.groupby("date")["range_2m"]
                   .transform(lambda x: x.shift(1).rolling(VOL_WINDOW, min_periods=5).mean()))
    df2["atr_ratio"] = df2["range_2m"] / avg_range.clip(lower=0.01)
    df2["vol_regime"] = (df2.groupby("date")["realized_vol"]
                           .transform(lambda x:
                               x.shift(1).rolling(VOL_PCT_WIN, min_periods=20).rank(pct=True)))

    # Merge 1-min features
    r1m_cols    = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols = ["ret_open"]
    df1_small   = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    df2 = df2.sort_values("ts")
    df2 = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    # Drop incomplete rows
    r2m_cols  = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    vola_cols = ["realized_vol", "atr_ratio", "vol_regime"]
    all_need  = r2m_cols + r1m_cols + ["ret_open"] + vola_cols
    df2 = df2.dropna(subset=all_need)
    df2 = df2[df2.groupby("date").cumcount() >= LOOKBACK * 2].reset_index(drop=True)
    df2["idx"] = df2.index
    return df2


def build_directional_dataset(df2: pd.DataFrame,
                               move_threshold: float,
                               move_lookback: int,
                               target_pts: float,
                               stop_pts: float) -> pd.DataFrame:
    """
    For each bar where |close - close[N bars ago]| >= move_threshold:
      - direction = +1 (long) or -1 (short)
      - flip return features so they're always from the move's perspective
      - target: 1 if price hits +target_pts before -stop_pts within HORIZON bars
    """
    closes = df2["close"].values
    highs  = df2["high"].values
    lows   = df2["low"].values
    dates  = df2["date"].values
    n      = len(df2)

    r2m_cols  = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    r1m_cols  = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    vola_cols = ["realized_vol", "atr_ratio", "vol_regime"]

    rows = []
    for i in range(move_lookback, n - HORIZON):
        # Same-day check for lookback
        ref_i = i - move_lookback
        if dates[ref_i] != dates[i]:
            continue

        move = closes[i] - closes[ref_i]
        if abs(move) < move_threshold:
            continue

        direction = 1 if move > 0 else -1

        # Check forward window is same day
        fwd_dates = dates[i + 1 : i + 1 + HORIZON]
        if len(fwd_dates) < HORIZON or fwd_dates[0] != fwd_dates[-1]:
            continue
        if fwd_dates[0] != dates[i]:
            continue

        # Build target: does price hit continuation target before stop?
        entry = closes[i]
        target_price = entry + direction * target_pts
        stop_price   = entry - direction * stop_pts
        hit_target = False
        hit_stop   = False
        for j in range(i + 1, i + 1 + HORIZON):
            if direction == 1:
                if highs[j] >= target_price:
                    hit_target = True
                    break
                if lows[j] <= stop_price:
                    hit_stop = True
                    break
            else:
                if lows[j] <= target_price:
                    hit_target = True
                    break
                if highs[j] >= stop_price:
                    hit_stop = True
                    break

        if not hit_target and not hit_stop:
            continue   # inconclusive within horizon — skip

        target = int(hit_target)

        # Build feature row — flip return signs for SHORT so model sees
        # everything from the "continuation" perspective
        row = {}
        for c in r2m_cols:
            row[c] = df2.at[i, c] * direction
        for c in r1m_cols:
            row[c] = df2.at[i, c] * direction
        row["ret_open"]     = df2.at[i, "ret_open"] * direction
        row["realized_vol"] = df2.at[i, "realized_vol"]
        row["atr_ratio"]    = df2.at[i, "atr_ratio"]
        row["vol_regime"]   = df2.at[i, "vol_regime"]
        row["move_size"]    = abs(move)
        row["move_bars"]    = move_lookback
        row["direction"]    = direction
        row["date"]         = dates[i]
        row["target"]       = target
        rows.append(row)

    return pd.DataFrame(rows)


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
    print("\nBuilding base features …", end=" ", flush=True)
    df2 = build_base_features(df1)
    print(f"{len(df2):,} 2-min bars\n", flush=True)

    fcols = ([f"r2m_{i}" for i in range(1, LOOKBACK)]
             + [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
             + ["ret_open", "realized_vol", "atr_ratio", "vol_regime",
                "move_size", "move_bars"])

    all_results = []

    for move_thr in MOVE_THRESHOLDS:
        for move_lb in MOVE_LOOKBACKS:
            print(f"── Move ≥{move_thr:.0f}pt over {move_lb} bar(s) "
                  f"({move_lb*2} min) ──────────────────────",
                  flush=True)
            print(f"  {'tgt:stp':<8}  {'n_train':>7}  {'n_test':>6}  "
                  f"{'base%':>6}  {'AUC':>6}  {'Prec':>6}  {'Rec':>5}  "
                  f"{'top feature'}")

            for tgt, stp in RISK_REWARDS:
                df_dir = build_directional_dataset(df2, move_thr, move_lb, tgt, stp)
                if len(df_dir) < 200:
                    continue
                train, test = split(df_dir)
                if len(train) < 200 or len(test) < 50:
                    continue

                res = run_rf(train, test, fcols)
                top = sorted(res["imp"].items(), key=lambda x: -x[1])[0][0]

                rr_label = f"{tgt}pt:{stp}pt"
                print(
                    f"  {rr_label:<8}  "
                    f"{res['n_train']:>7,}  {res['n_test']:>6,}  "
                    f"{res['base_rate']*100:>6.1f}%  "
                    f"{res['auc']:>6.3f}  "
                    f"{res['precision']:>6.3f}  "
                    f"{res['recall']:>5.3f}  "
                    f"{top}",
                    flush=True,
                )
                all_results.append({
                    "move_thr": move_thr, "move_lb": move_lb,
                    "tgt": tgt, "stp": stp,
                    "rr": f"{tgt}:{stp}",
                    "auc": res["auc"], "precision": res["precision"],
                    "recall": res["recall"], "base_rate": res["base_rate"],
                    "n_train": res["n_train"], "n_test": res["n_test"],
                    "top_feat": top,
                })
            print(flush=True)

    print("\n═" * 40)
    print("TOP 15 BY AUC")
    print("═" * 40)
    df_res = pd.DataFrame(all_results).sort_values("auc", ascending=False)
    pd.set_option("display.width", 130)
    print(df_res.head(15).to_string(index=False))

    print("\n── Best feature importances for top config ─────────────────")
    best = df_res.iloc[0]
    df_dir = build_directional_dataset(
        df2, best["move_thr"], int(best["move_lb"]), best["tgt"], best["stp"])
    train, test = split(df_dir)
    res = run_rf(train, test, fcols)
    print(f"  Config: move≥{best['move_thr']:.0f}pt/{best['move_lb']:.0f}bar  "
          f"tgt={best['tgt']:.0f}pt stp={best['stp']:.0f}pt  "
          f"AUC={res['auc']:.4f}  base={res['base_rate']*100:.1f}%")
    for name, val in sorted(res["imp"].items(), key=lambda x: -x[1])[:12]:
        bar = "█" * int(val * 400)
        print(f"    {name:<18} {val*100:5.2f}%  {bar}")


if __name__ == "__main__":
    main()
