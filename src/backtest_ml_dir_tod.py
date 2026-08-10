"""
backtest_ml_dir_tod.py — Directional RF model backtest with time-of-day breakdown.

Uses the exact same architecture as ml_trading_bot.py:
  - Qualifying momentum move gate (move_threshold_bps over move_lookback_bars)
  - Direction-flipped features fed to RF classifier
  - Directional TP/SL labels (TARGET hits before STOP within TRAIN_HORIZON)

Also sweeps move_threshold_bps and move_lookback_bars to find best config.

Run from repo root:
    python src/backtest_ml_dir_tod.py           # MES (default)
    python src/backtest_ml_dir_tod.py MNQ
    python src/backtest_ml_dir_tod.py MES sweep
"""

import sys
import numpy as np
import pandas as pd
from datetime import time as dtime

from ml_trading_bot import (
    SYMBOL_CONFIGS, ET,
    PRE_START, RTH_START, RTH_END, BLACKOUT_END, SESSION_END,
    LOOKBACK, SHORT_N, TRAIN_HORIZON,
    PROB_THRESHOLD, VOL_REGIME_MIN,
    load_training_data, build_2min_features,
    build_directional_dataset, get_feature_cols, train_model,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

WINDOWS = [
    ("9:40–11:00",  dtime(9, 40),  dtime(11,  0)),
    ("11:00–13:00", dtime(11,  0), dtime(13,  0)),
    ("13:00–15:45", dtime(13,  0), dtime(15, 45)),
]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(df_dir_test: pd.DataFrame, clf, fcols: list[str],
             cfg, df2_all: pd.DataFrame) -> pd.DataFrame:
    """
    Walk test rows in time order.  One trade at a time.
    df_dir_test: directional dataset rows for test period (already filtered)
    df2_all:     full 2-min bar df (needed to check move_close at test time)
    """
    # Build a ts→index lookup into df2_all for move_close retrieval
    df2_all = df2_all.reset_index(drop=True)
    ts_to_idx = {row["ts"]: i for i, row in df2_all.iterrows()}

    feat_arr  = df_dir_test[fcols].values
    proba     = clf.predict_proba(feat_arr)[:, 1]

    rows      = df_dir_test.reset_index(drop=True)
    trades    = []
    in_trade  = False

    for i, row in rows.iterrows():
        bar_t = pd.Timestamp(row["ts"]).tz_localize(None) if row["ts"].tzinfo is None \
                else row["ts"]
        bar_t = row["ts"].time() if hasattr(row["ts"], "time") else \
                pd.Timestamp(row["ts"]).time()

        # ── Manage open position ─────────────────────────────────────────────
        if in_trade:
            # Close at open if new day or past session end
            if row["date"] != trades[-1]["date"] or bar_t >= SESSION_END:
                # Mark as expired — use close of the entry bar as proxy
                trades[-1].update(outcome="EXPIRED", pnl_pts=0.0)
                in_trade = False
            else:
                # Check if this bar resolves the trade (direction-aware)
                h, l  = row["high"], row["low"]
                entry = trades[-1]["entry"]
                tgt   = trades[-1]["tgt"]
                stp   = trades[-1]["stp"]
                d     = trades[-1]["dir_int"]
                if d == 1:
                    if h >= entry + tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt); in_trade=False; continue
                    if l <= entry - stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp); in_trade=False; continue
                else:
                    if l <= entry - tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt); in_trade=False; continue
                    if h >= entry + stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp); in_trade=False; continue
                continue  # still in trade

        # ── Entry gate ───────────────────────────────────────────────────────
        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue

        prob       = proba[i]
        vol_regime = row["vol_regime"]

        if prob < PROB_THRESHOLD or vol_regime < VOL_REGIME_MIN:
            continue

        direction = int(row["direction"])
        entry     = row["close"] if "close" in row else 0
        tgt       = cfg.pts_from_bps(entry, cfg.target_bps) if entry else 0
        stp       = cfg.pts_from_bps(entry, cfg.stop_bps)   if entry else 0

        trades.append({
            "ts":        row["ts"],
            "date":      row["date"],
            "bar_time":  bar_t,
            "direction": "LONG" if direction == 1 else "SHORT",
            "dir_int":   direction,
            "entry":     entry,
            "tgt":       tgt,
            "stp":       stp,
            "prob":      prob,
            "high":      row.get("high", entry),
            "low":       row.get("low", entry),
            "outcome":   None,
            "pnl_pts":   None,
        })
        in_trade = True

    df_t = pd.DataFrame(trades)
    df_t = df_t.dropna(subset=["outcome"]).reset_index(drop=True)
    return df_t


def report_window(label: str, trades: pd.DataFrame, tick_val: float):
    n = len(trades)
    if n == 0:
        print(f"  {label:<14}  no trades")
        return
    wins    = (trades["outcome"] == "TARGET").sum()
    expired = (trades["outcome"] == "EXPIRED").sum()
    wr      = wins / n * 100
    pnl     = trades["pnl_pts"].sum()
    avg     = trades["pnl_pts"].mean()
    usd     = pnl * tick_val * 4
    exp_str = f"  exp={expired}" if expired else ""
    print(f"  {label:<14}  n={n:3d}  WR={wr:5.1f}%  "
          f"avg={avg:+.2f}pt  total={pnl:+.1f}pt  (~${usd:+,.0f}){exp_str}")


# ── Core run ──────────────────────────────────────────────────────────────────

def run_one(cfg, df1, df2, sweep_cfg: dict | None = None):
    """Build directional dataset, train, simulate, report."""
    from copy import copy
    if sweep_cfg:
        cfg = copy(cfg)
        cfg.move_threshold_bps = sweep_cfg["move_threshold_bps"]
        cfg.move_lookback_bars = sweep_cfg["move_lookback_bars"]

    print(f"\n  move≥{cfg.move_threshold_bps:.0f}bp  lb={cfg.move_lookback_bars}bar  "
          f"tgt={cfg.target_bps:.0f}bp  stp={cfg.stop_bps:.0f}bp", flush=True)

    df_dir = build_directional_dataset(df2, cfg)
    if len(df_dir) < 500:
        print(f"  Too few qualifying bars ({len(df_dir)}) — skipping")
        return None

    base = df_dir["target"].mean()
    print(f"  Qualifying bars: {len(df_dir):,}  base_rate={base*100:.1f}%", flush=True)

    fcols  = get_feature_cols()
    dates  = sorted(df_dir["date"].unique())
    cutoff = dates[int(len(dates) * 0.75)]
    train  = df_dir[df_dir["date"] <  cutoff]
    test   = df_dir[df_dir["date"] >= cutoff]

    if len(train) < 300 or len(test) < 100:
        print(f"  Insufficient train/test split — skipping")
        return None

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(train[fcols].values, train["target"].astype(int).values)
    proba = clf.predict_proba(test[fcols].values)[:, 1]
    auc   = roc_auc_score(test["target"].astype(int).values, proba)
    print(f"  AUC={auc:.4f}  test_base={test['target'].mean()*100:.1f}%  "
          f"test_n={len(test):,}", flush=True)

    # Need close/high/low in test for simulation
    # Merge back from df2 on ts
    df2_lookup = df2.set_index("ts")[["close","high","low"]].copy()
    test2 = test.copy()
    for col in ["close","high","low"]:
        if col not in test2.columns:
            test2[col] = test2["ts"].map(df2_lookup[col])

    tick_val = {"MES": 1.25, "MNQ": 0.50}[cfg.symbol]
    trades   = simulate(test2, clf, fcols, cfg, df2)

    test_days = test["date"].nunique()
    n_trades  = len(trades)
    if n_trades == 0:
        print("  No completed trades in simulation")
        return {"auc": auc, "n_trades": 0}

    total_pnl = trades["pnl_pts"].sum()
    total_wr  = (trades["outcome"] == "TARGET").mean() * 100

    print(f"\n  ── Overall ({test_days} test days) ─────────────────────────")
    print(f"  Trades: {n_trades}  ({n_trades/test_days:.1f}/day)  "
          f"WR: {total_wr:.1f}%  "
          f"Total P&L: {total_pnl:+.1f}pt  (~${total_pnl*tick_val*4:+,.0f})")

    print(f"\n  ── By Time Window ──────────────────────────────────────────")
    for label, t0, t1 in WINDOWS:
        mask = (trades["bar_time"] >= t0) & (trades["bar_time"] < t1)
        report_window(label, trades[mask], tick_val)

    print(f"\n  ── By Day-of-Week ──────────────────────────────────────────")
    trades["dow"] = pd.to_datetime(trades["date"]).dt.strftime("%a")
    for dow in ["Mon","Tue","Wed","Thu","Fri"]:
        report_window(dow, trades[trades["dow"]==dow], tick_val)

    return {
        "auc":       auc,
        "n_trades":  n_trades,
        "per_day":   n_trades / test_days,
        "wr":        total_wr,
        "pnl":       total_pnl,
        "pnl_usd":   total_pnl * tick_val * 4,
        "move_thr":  cfg.move_threshold_bps,
        "move_lb":   cfg.move_lookback_bars,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym   = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    mode  = sys.argv[2].lower() if len(sys.argv) > 2 else "single"
    cfg   = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    print(f"\n=== ML Directional TOD Backtest  ·  {sym} ===\n")
    print("Loading bars …", flush=True)
    df1 = load_training_data(cfg)

    print("Building 2-min features …", end=" ", flush=True)
    df2 = build_2min_features(df1, drop_warmup=True)
    print(f"{len(df2):,} bars\n", flush=True)

    if mode == "sweep":
        print("── Parameter sweep ─────────────────────────────────────────────")
        results = []
        for move_thr in [4.0, 5.0, 6.0, 7.0, 8.0]:
            for move_lb in [1, 2]:
                r = run_one(cfg, df1, df2,
                            {"move_threshold_bps": move_thr,
                             "move_lookback_bars": move_lb})
                if r and r["n_trades"] > 0:
                    results.append(r)

        if results:
            print("\n── Sweep Summary (sorted by P&L) ───────────────────────────")
            df_r = pd.DataFrame(results).sort_values("pnl", ascending=False)
            print(df_r[["move_thr","move_lb","n_trades","per_day",
                         "wr","pnl","pnl_usd","auc"]].to_string(index=False))
    else:
        run_one(cfg, df1, df2)


if __name__ == "__main__":
    main()
