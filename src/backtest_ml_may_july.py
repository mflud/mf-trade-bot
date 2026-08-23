"""
backtest_ml_may_july.py — Out-of-sample ML RF backtest for May–July 2026.

Replicates ml_trading_bot.py exactly:
  - Same feature engineering (build_2min_features / build_directional_dataset)
  - Same RF hyperparameters
  - Same entry gates (prob >= 0.65, vol_regime >= 0.50, 9:40–15:45 ET)
  - Same direction rule (r2m_1 and ret_open must agree)
  - Same target/stop (MES: 16bp target / 9bp stop)

Train cutoff: 2026-04-30  (all data before May)
Test window:  2026-05-01 → 2026-07-31

Usage:
    python src/backtest_ml_may_july.py
"""

import math
import sqlite3
from datetime import time as dtime, date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ET = ZoneInfo("America/New_York")

HIST_CSV    = Path("mes_hist_1min.csv")
BARS_DB     = Path("data/bars.db")
PRE_START   = dtime(8, 30)
RTH_START   = dtime(9, 30)
RTH_END     = dtime(16, 0)
BLACKOUT    = dtime(9, 40)
SESSION_END = dtime(15, 45)

TRAIN_CUTOFF = date(2026, 5, 1)   # exclusive — train on < this date
DATE_START   = date(2026, 5, 1)
DATE_END     = date(2026, 7, 31)

# Model hyperparams — identical to ml_trading_bot.py
LOOKBACK    = 15
SHORT_N     = 6
VOL_WINDOW  = 10
VOL_PCT_WIN = 60
TRAIN_HORIZON = 3   # 2-min bars = 6 min

# MES production params
TARGET_BPS  = 16.0
STOP_BPS    = 9.0
MOVE_THR_BPS   = 3.0
MOVE_LB_BARS   = 2
TICK = 0.25
MES_PV = 5.0

# Entry gates
PROB_THRESHOLD = 0.65
VOL_REGIME_MIN = 0.50


def pts_from_bps(price: float, bps: float) -> float:
    """Round up to nearest tick (matches SymbolConfig.pts_from_bps)."""
    raw   = price * bps / 10000.0
    ticks = int(raw / TICK) + (1 if raw % TICK > 0 else 0)
    return round(ticks * TICK, 4)


# ─── Data loading ──────────────────────────────────────────────────────────────

def load_1min() -> pd.DataFrame:
    frames = []
    if HIST_CSV.exists():
        df_h = pd.read_csv(HIST_CSV)
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= PRE_START) & (t < RTH_END)].copy()
        frames.append(df_h)
    conn = sqlite3.connect(BARS_DB)
    df_db = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM bars "
        "WHERE symbol='MES' AND minutes=1 ORDER BY ts", conn)
    conn.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t2 = df_db["ts"].dt.time
    df_db = df_db[(t2 >= PRE_START) & (t2 < RTH_END)].copy()
    frames.append(df_db)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True))
    df["date"] = df["ts"].dt.date
    print(f"  {len(df):,} pre+RTH 1-min bars  {df['date'].min()} → {df['date'].max()}")
    return df


# ─── Feature engineering (matches ml_trading_bot.py exactly) ──────────────────

def build_2min_features(df1: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
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
    df2 = df2[(t2 >= PRE_START) & (t2 < RTH_END)].reset_index()
    df2["date"] = df2["ts"].dt.date

    df2["lret2"] = np.log(df2["close"] / df2["close"].shift(1))
    df2.loc[df2["date"] != df2["date"].shift(1), "lret2"] = np.nan
    for lag in range(1, LOOKBACK):
        s = df2["lret2"].shift(lag)
        s[df2["date"] != df2["date"].shift(lag)] = np.nan
        df2[f"r2m_{lag}"] = s

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

    r1m_cols  = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    df1_small = df1[["ts"] + r1m_cols + ["ret_open"]].sort_values("ts")
    df2 = df2.sort_values("ts")
    df2 = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    r2m_cols  = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    vola_cols = ["realized_vol", "atr_ratio", "vol_regime"]
    all_need  = r2m_cols + r1m_cols + ["ret_open"] + vola_cols
    df2 = df2.dropna(subset=all_need)
    if drop_warmup:
        df2 = df2[df2.groupby("date").cumcount() >= LOOKBACK * 2].reset_index(drop=True)
    else:
        df2 = df2.reset_index(drop=True)
    return df2


def build_directional_dataset(df2: pd.DataFrame) -> pd.DataFrame:
    df2 = df2[df2["ts"].dt.time >= BLACKOUT].reset_index(drop=True)

    closes = df2["close"].values
    highs  = df2["high"].values
    lows   = df2["low"].values
    dates  = df2["date"].values
    n      = len(df2)

    r2m_cols = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    r1m_cols = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]

    lb   = MOVE_LB_BARS
    rows = []

    for i in range(lb, n - TRAIN_HORIZON):
        if dates[i - lb] != dates[i]:
            continue
        move = closes[i] - closes[i - lb]
        thr  = closes[i] * MOVE_THR_BPS / 10000.0
        if abs(move) < thr:
            continue

        direction = 1 if move > 0 else -1

        fwd = dates[i + 1: i + 1 + TRAIN_HORIZON]
        if len(fwd) < TRAIN_HORIZON or fwd[0] != fwd[-1] or fwd[0] != dates[i]:
            continue

        entry = closes[i]
        tgt   = pts_from_bps(entry, TARGET_BPS)
        stp   = pts_from_bps(entry, STOP_BPS)

        hit_target = hit_stop = False
        for j in range(i + 1, i + 1 + TRAIN_HORIZON):
            if direction == 1:
                if highs[j] >= entry + tgt:  hit_target = True; break
                if lows[j]  <= entry - stp:  hit_stop   = True; break
            else:
                if lows[j]  <= entry - tgt:  hit_target = True; break
                if highs[j] >= entry + stp:  hit_stop   = True; break

        if not hit_target and not hit_stop:
            continue

        row = {}
        for c in r2m_cols:
            row[c] = df2.at[i, c] * direction
        for c in r1m_cols:
            row[c] = df2.at[i, c] * direction
        row["ret_open"]      = df2.at[i, "ret_open"] * direction
        row["realized_vol"]  = df2.at[i, "realized_vol"]
        row["atr_ratio"]     = df2.at[i, "atr_ratio"]
        row["vol_regime"]    = df2.at[i, "vol_regime"]
        row["move_size_bps"] = abs(move) / closes[i] * 10000.0
        row["date"]          = dates[i]
        row["ts"]            = df2["ts"].iloc[i]
        row["direction"]     = direction
        row["target"]        = int(hit_target)
        rows.append(row)

    return pd.DataFrame(rows)


def get_fcols() -> list[str]:
    return ([f"r2m_{i}" for i in range(1, LOOKBACK)]
            + [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
            + ["ret_open", "realized_vol", "atr_ratio", "vol_regime", "move_size_bps"])


# ─── Simulate live entries on test bars ────────────────────────────────────────

def simulate_live(df2_test: pd.DataFrame, clf: RandomForestClassifier,
                  fcols: list[str]) -> list[dict]:
    """
    Walk through test 2-min bars in chronological order, firing trades
    using the same entry gates as ml_trading_bot.py live loop.
    """
    df2_test = df2_test[
        (df2_test["ts"].dt.time >= BLACKOUT) &
        (df2_test["ts"].dt.time <= SESSION_END)
    ].reset_index(drop=True)

    closes = df2_test["close"].values
    highs  = df2_test["high"].values
    lows   = df2_test["low"].values
    dates  = df2_test["date"].values
    n      = len(df2_test)

    lb = MOVE_LB_BARS
    trades = []
    hold_until = -1   # bar index until which we're in a trade

    for i in range(lb, n - TRAIN_HORIZON):
        if i <= hold_until:
            continue
        if dates[i - lb] != dates[i]:
            continue

        # Vol regime gate
        vr = df2_test.at[i, "vol_regime"]
        if pd.isna(vr) or vr < VOL_REGIME_MIN:
            continue

        # Qualifying momentum move
        move = closes[i] - closes[i - lb]
        thr  = closes[i] * MOVE_THR_BPS / 10000.0
        if abs(move) < thr:
            continue

        raw_direction = 1 if move > 0 else -1

        # Direction alignment gate (r2m_1 and ret_open must agree)
        r2m1    = df2_test.at[i, "r2m_1"]
        ret_op  = df2_test.at[i, "ret_open"]
        if not ((r2m1 > 0 and ret_op > 0) or (r2m1 < 0 and ret_op < 0)):
            continue
        # Reconcile: if r2m_1 sign disagrees with move sign, skip
        if np.sign(r2m1) != raw_direction:
            continue

        # Sign-flip features for model (same as training)
        base_fcols = ([f"r2m_{k}" for k in range(1, LOOKBACK)]
                      + [f"r1m_{k}" for k in range(1, SHORT_N + 1)]
                      + ["ret_open"])
        feat_row = {}
        for c in fcols:
            if c == "move_size_bps":
                feat_row[c] = abs(move) / closes[i] * 10000.0
                continue
            val = df2_test.at[i, c]
            if pd.isna(val):
                feat_row[c] = 0.0
            elif c in base_fcols:
                feat_row[c] = val * raw_direction
            else:
                feat_row[c] = val
        X = np.array([feat_row[c] for c in fcols]).reshape(1, -1)

        prob = clf.predict_proba(X)[0, 1]
        if prob < PROB_THRESHOLD:
            continue

        # Fire trade
        direction = raw_direction
        entry     = closes[i]
        tgt_pts   = pts_from_bps(entry, TARGET_BPS)
        stp_pts   = pts_from_bps(entry, STOP_BPS)

        outcome   = "TIME"
        exit_pts  = None
        exit_i    = min(i + 20, n - 1)   # max ~40 min fallback

        for j in range(i + 1, min(i + 21, n)):
            if dates[j] != dates[i]:
                exit_pts = (closes[j-1] - entry) * direction
                exit_i   = j - 1
                outcome  = "EOD"
                break
            if direction == 1:
                if highs[j] >= entry + tgt_pts:
                    exit_pts = tgt_pts; exit_i = j; outcome = "TARGET"; break
                if lows[j]  <= entry - stp_pts:
                    exit_pts = -stp_pts; exit_i = j; outcome = "STOPPED"; break
            else:
                if lows[j]  <= entry - tgt_pts:
                    exit_pts = tgt_pts; exit_i = j; outcome = "TARGET"; break
                if highs[j] >= entry + stp_pts:
                    exit_pts = -stp_pts; exit_i = j; outcome = "STOPPED"; break

        if exit_pts is None:
            exit_pts = (closes[exit_i] - entry) * direction
            outcome  = "TIME"

        hold_until = exit_i
        pnl_pts    = exit_pts
        pnl_dol    = pnl_pts * MES_PV

        trades.append({
            "date":      dates[i],
            "ts":        df2_test["ts"].iloc[i],
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry":     round(entry, 2),
            "tgt_pts":   tgt_pts,
            "stp_pts":   stp_pts,
            "prob":      round(prob, 3),
            "vol_regime": round(float(vr), 3),
            "outcome":   outcome,
            "pnl_pts":   round(pnl_pts, 3),
            "pnl_dol":   round(pnl_dol, 2),
        })

    return trades


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'═'*65}")
    print(f"  ML RF  MES  Out-of-Sample Backtest  —  May–July 2026")
    print(f"  Train: all data < {TRAIN_CUTOFF}   Test: {DATE_START} → {DATE_END}")
    print(f"{'═'*65}\n")

    print("Loading 1-min bars …", flush=True)
    df1 = load_1min()

    # Build features on FULL dataset — vol_regime needs rolling history across all sessions
    print("Building 2-min feature matrix (full history for correct rolling context) …", flush=True)
    df2_full = build_2min_features(df1)
    df2_full["date"] = df2_full["ts"].dt.date

    # Split AFTER feature computation
    df2_train = df2_full[df2_full["date"] < TRAIN_CUTOFF].reset_index(drop=True)
    df2_test  = df2_full[(df2_full["date"] >= DATE_START) & (df2_full["date"] <= DATE_END)].reset_index(drop=True)
    # alias for rest of function
    df2 = df2_full

    print(f"  Train 2-min bars: {len(df2_train):,}  ({df2_train['date'].min()} → {df2_train['date'].max()})")
    print(f"  Test  2-min bars: {len(df2_test):,}  ({df2_test['date'].min()} → {df2_test['date'].max()})")

    # Build training dataset
    print("\nBuilding directional training dataset …", flush=True)
    df_dir = build_directional_dataset(df2_train)
    print(f"  {len(df_dir):,} qualifying directional bars  base_rate={df_dir['target'].mean():.1%}")

    # Train RF (same hyperparams as ml_trading_bot.py)
    print("\nTraining RandomForest …", flush=True)
    fcols = get_fcols()
    X = df_dir[fcols].values
    y = df_dir["target"].astype(int).values

    # Quick in-sample AUC to sanity check (last 25% of training as holdout)
    dates_tr = sorted(df_dir["date"].unique())
    cutoff75  = dates_tr[int(len(dates_tr) * 0.75)]
    tr75 = df_dir[df_dir["date"] < cutoff75]
    va25 = df_dir[df_dir["date"] >= cutoff75]

    clf_check = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf_check.fit(tr75[fcols].values, tr75["target"].astype(int).values)
    auc_val = roc_auc_score(va25["target"].astype(int).values,
                             clf_check.predict_proba(va25[fcols].values)[:, 1])
    print(f"  Validation AUC (last 25% of training): {auc_val:.3f}")
    print(f"  Train date range: {dates_tr[0]} → {dates_tr[-1]}")

    # Train final model on ALL training data
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X, y)
    print(f"  Final model trained on {len(df_dir):,} bars ({df_dir['date'].min()} → {df_dir['date'].max()})")

    # Simulate live trading on test period
    print(f"\nSimulating live trades on {DATE_START} → {DATE_END} …", flush=True)
    trades = simulate_live(df2_test, clf, fcols)

    if not trades:
        print("  No trades fired in test period.")
        return

    df_t = pd.DataFrame(trades)
    df_t["ym"] = df_t["date"].apply(lambda d: f"{d.year}-{d.month:02d}")

    n      = len(df_t)
    wr     = (df_t["pnl_pts"] > 0).mean()
    avg_pt = df_t["pnl_pts"].mean()
    tot_pt = df_t["pnl_pts"].sum()
    tot_d  = df_t["pnl_dol"].sum()

    tgt_n  = (df_t["outcome"] == "TARGET").sum()
    stp_n  = (df_t["outcome"] == "STOPPED").sum()
    oth_n  = n - tgt_n - stp_n

    print(f"\n{'─'*65}")
    print(f"  RESULTS  —  ML RF  MES  May–July 2026")
    print(f"{'─'*65}")
    print(f"  Trades: {n}  WR: {wr:.1%}  avg: {avg_pt:+.3f}pt  total: {tot_pt:+.1f}pt  ≈${tot_d:+,.0f}")
    print(f"  Outcomes:  TARGET={tgt_n}  STOPPED={stp_n}  other={oth_n}")
    print(f"  avg prob:  {df_t['prob'].mean():.3f}   avg vol_regime: {df_t['vol_regime'].mean():.3f}")

    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<8}  {'n':>4}  {'WR':>7}  {'avg pt':>8}  {'total pt':>9}  {'$':>8}")
    for ym, g in df_t.groupby("ym"):
        wr2  = (g["pnl_pts"] > 0).mean()
        av2  = g["pnl_pts"].mean()
        tot2 = g["pnl_pts"].sum()
        d2   = g["pnl_dol"].sum()
        print(f"  {ym:<8}  {len(g):4d}  {wr2:7.1%}  {av2:+8.3f}pt  {tot2:+9.2f}pt  ${d2:+8,.0f}")

    print(f"\n  Direction breakdown:")
    for dirn, g in df_t.groupby("direction"):
        wr2 = (g["pnl_pts"] > 0).mean()
        av2 = g["pnl_pts"].mean()
        print(f"    {dirn:<6}  n={len(g):3d}  WR={wr2:.1%}  avg={av2:+.3f}pt")

    print(f"\n  Outcome breakdown:")
    for out, g in df_t.groupby("outcome"):
        av2 = g["pnl_pts"].mean()
        print(f"    {out:<10}  n={len(g):3d}  avg={av2:+.3f}pt")

    print(f"\n  Feature importance (top 10):")
    imp = sorted(zip(fcols, clf.feature_importances_), key=lambda x: -x[1])
    for feat, score in imp[:10]:
        bar = "█" * int(score * 50)
        print(f"    {feat:<18}  {score:.3f}  {bar}")

    # Comparison vs May-July baseline (from unified backtest)
    print(f"\n{'═'*65}")
    print(f"  Context: May-July 2026 strategy comparison (1-lot MES)")
    print(f"  {'Strategy':<20}  {'Trades':>7}  {'WR':>7}  {'Total $':>9}")
    print(f"  {'─'*50}")
    comparisons = [
        ("ML RF",       n,    wr,     tot_d),
        ("VWASLR",      260,  0.762,  11018),
        ("SLR Scalp",   31,   0.516,    275),
        ("ORB",         10,   0.600,    451),
        ("PL_REV",      11,   0.364,    -62),
        ("Wall Break",  96,   0.427,    591),
    ]
    for name, nt, wr3, dol in comparisons:
        print(f"  {name:<20}  {nt:>7d}  {wr3:>7.1%}  ${dol:>+9,.0f}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
