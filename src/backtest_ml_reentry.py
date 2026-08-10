"""
backtest_ml_reentry.py — Sweep move_threshold_bps and re-entry on 2-min directional model.

Tests two levers to increase trade frequency:
  1. Lower move_threshold_bps (3bp vs 4bp vs 5bp)
  2. Re-entry allowed immediately after a trade closes (vs single trade/session)

For each combination, reports: trades/day, WR, avg P&L, total P&L, and TOD breakdown.

Run from repo root:
    python src/backtest_ml_reentry.py           # MES
    python src/backtest_ml_reentry.py MNQ
"""

import sys
import numpy as np
import pandas as pd
from copy import copy
from datetime import time as dtime

from ml_trading_bot import (
    SYMBOL_CONFIGS, ET,
    PRE_START, RTH_START, RTH_END, BLACKOUT_END, SESSION_END,
    LOOKBACK, SHORT_N, TRAIN_HORIZON,
    PROB_THRESHOLD, VOL_REGIME_MIN,
    load_training_data, build_2min_features,
    build_directional_dataset, get_feature_cols,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

WINDOWS = [
    ("9:40–11:00",  dtime(9, 40),  dtime(11,  0)),
    ("11:00–13:00", dtime(11,  0), dtime(13,  0)),
    ("13:00–15:45", dtime(13,  0), dtime(15, 45)),
]

MOVE_THRESHOLDS = [2.0, 3.0, 4.0, 5.0, 6.0]

# Vol lookback sweep — controls how many 2-min bars are used for:
#   realized_vol (std window), atr_ratio (avg range), vol_regime (pct window)
# Defaults in live bot: VOL_WINDOW=10, VOL_PCT_WIN=60
VOL_WINDOW_SWEEP  = [5, 10, 20]       # for realized_vol and atr_ratio
VOL_PCT_WIN_SWEEP = [30, 60, 120]     # for vol_regime percentile rank


# ── Simulation (supports re-entry) ────────────────────────────────────────────

def simulate(test: pd.DataFrame, clf, fcols: list[str],
             cfg, allow_reentry: bool = False) -> pd.DataFrame:
    """
    Walk test rows. allow_reentry=True: take new signal immediately after
    a trade closes. allow_reentry=False: one trade per session (original).
    """
    feat_arr = test[fcols].values
    proba    = clf.predict_proba(feat_arr)[:, 1]
    rows     = test.reset_index(drop=True)
    trades   = []
    in_trade = False
    traded_today = False
    last_date = None

    for i, row in rows.iterrows():
        bar_t = row["ts"].time()
        today = row["date"]

        # Reset per-session flag
        if today != last_date:
            traded_today = False
            last_date = today

        # ── Manage open position ─────────────────────────────────────────────
        if in_trade:
            if today != trades[-1]["date"] or bar_t >= SESSION_END:
                trades[-1].update(outcome="EXPIRED", pnl_pts=0.0)
                in_trade = False
                traded_today = not allow_reentry  # if no reentry, block rest of day
            else:
                h, l  = row["high"], row["low"]
                entry = trades[-1]["entry"]
                tgt   = trades[-1]["tgt"]
                stp   = trades[-1]["stp"]
                d     = trades[-1]["dir_int"]
                closed = False
                if d == 1:
                    if h >= entry + tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt); closed=True
                    elif l <= entry - stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp); closed=True
                else:
                    if l <= entry - tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt); closed=True
                    elif h >= entry + stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp); closed=True
                if closed:
                    in_trade = False
                    if not allow_reentry:
                        traded_today = True
                    continue
                else:
                    continue  # still in trade

        # ── Entry gate ───────────────────────────────────────────────────────
        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue
        if not allow_reentry and traded_today:
            continue
        if proba[i] < PROB_THRESHOLD or row["vol_regime"] < VOL_REGIME_MIN:
            continue

        d     = int(row["direction"])
        entry = row["close"]
        tgt   = cfg.pts_from_bps(entry, cfg.target_bps)
        stp   = cfg.pts_from_bps(entry, cfg.stop_bps)

        trades.append({
            "ts": row["ts"], "date": today, "bar_time": bar_t,
            "direction": "LONG" if d == 1 else "SHORT", "dir_int": d,
            "entry": entry, "tgt": tgt, "stp": stp,
            "high": row["high"], "low": row["low"],
            "prob": proba[i], "outcome": None, "pnl_pts": None,
        })
        in_trade = True

    if not trades:
        return pd.DataFrame()
    df_t = pd.DataFrame(trades)
    if "outcome" in df_t.columns:
        df_t = df_t.dropna(subset=["outcome"])
    return df_t.reset_index(drop=True)


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_window(label: str, trades: pd.DataFrame, tick_val: float):
    n = len(trades)
    if n == 0:
        print(f"    {label:<14}  —"); return
    wins    = (trades["outcome"] == "TARGET").sum()
    expired = (trades["outcome"] == "EXPIRED").sum()
    wr      = wins / n * 100
    pnl     = trades["pnl_pts"].sum()
    avg     = trades["pnl_pts"].mean()
    usd     = pnl * tick_val * 4
    xs      = f"  exp={expired}" if expired else ""
    print(f"    {label:<14}  n={n:4d}  WR={wr:5.1f}%  "
          f"avg={avg:+.2f}pt  total={pnl:+.1f}pt  (~${usd:+,.0f}){xs}")


def summarise(label: str, trades: pd.DataFrame, test_days: int,
              tick_val: float):
    if trades.empty or len(trades) == 0:
        print(f"  {label}: no trades"); return
    n    = len(trades)
    wr   = (trades["outcome"] == "TARGET").mean() * 100
    pnl  = trades["pnl_pts"].sum()
    avg  = trades["pnl_pts"].mean()
    usd  = pnl * tick_val * 4
    exp  = (trades["outcome"] == "EXPIRED").sum()
    print(f"  {label}:  n={n} ({n/test_days:.2f}/day)  WR={wr:.1f}%  "
          f"avg={avg:+.2f}pt  P&L={pnl:+.1f}pt (~${usd:+,.0f})"
          + (f"  exp={exp}" if exp else ""))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    cfg = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    tick_val = {"MES": 1.25, "MNQ": 0.50}[sym]

    print(f"\n=== 2-Min Directional RF  ·  Re-entry & Threshold Sweep  ·  {sym} ===\n")
    print("Loading bars …", flush=True)
    df1 = load_training_data(cfg)

    print("Building 2-min features …", end=" ", flush=True)
    df2 = build_2min_features(df1, drop_warmup=True)
    print(f"{len(df2):,} bars\n", flush=True)

    fcols = get_feature_cols()

    # Train one model per threshold (direction-flipped features, so threshold
    # affects dataset composition but we retrain for each)
    summary_rows = []

    for move_thr in MOVE_THRESHOLDS:
        cfg2 = copy(cfg)
        cfg2.move_threshold_bps = move_thr

        df_dir = build_directional_dataset(df2, cfg2)
        if len(df_dir) < 300:
            print(f"move≥{move_thr:.0f}bp: too few bars ({len(df_dir)}) — skip")
            continue

        dates  = sorted(df_dir["date"].unique())
        cutoff = dates[int(len(dates) * 0.75)]
        train  = df_dir[df_dir["date"] <  cutoff]
        test   = df_dir[df_dir["date"] >= cutoff]
        test_days = test["date"].nunique()

        if len(train) < 200 or len(test) < 50:
            continue

        # Merge close/high/low back for simulation
        df2_lu = df2.set_index("ts")[["close","high","low"]]
        test2  = test.copy()
        for col in ["close","high","low"]:
            if col not in test2.columns:
                test2[col] = test2["ts"].map(df2_lu[col])

        clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[fcols].values, train["target"].astype(int).values)
        proba = clf.predict_proba(test[fcols].values)[:, 1]
        auc   = roc_auc_score(test["target"].astype(int).values, proba)
        base  = test["target"].mean() * 100

        print(f"── move≥{move_thr:.0f}bp  (AUC={auc:.4f}  base={base:.1f}%  "
              f"train={len(train):,}  test={len(test):,}  days={test_days})")

        for reentry in [False, True]:
            label = "re-entry" if reentry else "single  "
            trades = simulate(test2, clf, fcols, cfg2, allow_reentry=reentry)
            summarise(label, trades, test_days, tick_val)

            if not trades.empty:
                summary_rows.append({
                    "move_thr":  move_thr,
                    "reentry":   reentry,
                    "auc":       auc,
                    "n":         len(trades),
                    "per_day":   len(trades) / test_days,
                    "wr":        (trades["outcome"] == "TARGET").mean() * 100,
                    "avg_pnl":   trades["pnl_pts"].mean(),
                    "total_pnl": trades["pnl_pts"].sum(),
                    "usd":       trades["pnl_pts"].sum() * tick_val * 4,
                    "expired":   (trades["outcome"] == "EXPIRED").sum(),
                    "_trades":   trades,
                })

        print()

    # ── Vol lookback sweep (fixed threshold = cfg default, re-entry=True) ────
    print("═" * 70)
    print(f"VOL LOOKBACK SWEEP  (move≥{cfg.move_threshold_bps:.0f}bp  re-entry=ON)")
    print("═" * 70)
    vol_rows = []
    for vw in VOL_WINDOW_SWEEP:
        for vpw in VOL_PCT_WIN_SWEEP:
            # Rebuild features with different vol windows
            import ml_trading_bot as _mlb
            _orig_vw, _orig_vpw = _mlb.VOL_WINDOW, _mlb.VOL_PCT_WIN
            _mlb.VOL_WINDOW  = vw
            _mlb.VOL_PCT_WIN = vpw
            df2v = build_2min_features(df1, drop_warmup=True)
            _mlb.VOL_WINDOW  = _orig_vw
            _mlb.VOL_PCT_WIN = _orig_vpw

            df_dir = build_directional_dataset(df2v, cfg)
            if len(df_dir) < 300:
                continue
            dates  = sorted(df_dir["date"].unique())
            cutoff = dates[int(len(dates) * 0.75)]
            train  = df_dir[df_dir["date"] <  cutoff]
            test   = df_dir[df_dir["date"] >= cutoff]
            test_days = test["date"].nunique()
            if len(train) < 200 or len(test) < 50:
                continue

            df2v_lu = df2v.set_index("ts")[["close","high","low"]]
            test2   = test.copy()
            for col in ["close","high","low"]:
                if col not in test2.columns:
                    test2[col] = test2["ts"].map(df2v_lu[col])

            clf2 = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            clf2.fit(train[fcols].values, train["target"].astype(int).values)
            proba2 = clf2.predict_proba(test[fcols].values)[:, 1]
            auc2   = roc_auc_score(test["target"].astype(int).values, proba2)

            trades2 = simulate(test2, clf2, fcols, cfg, allow_reentry=True)
            if trades2.empty:
                continue
            n2   = len(trades2)
            wr2  = (trades2["outcome"] == "TARGET").mean() * 100
            pnl2 = trades2["pnl_pts"].sum()
            usd2 = pnl2 * tick_val * 4
            marker = " ◄ baseline" if vw == 10 and vpw == 60 else ""
            print(f"  vol_win={vw:3d}  pct_win={vpw:3d}  "
                  f"AUC={auc2:.4f}  n={n2:3d} ({n2/test_days:.2f}/day)  "
                  f"WR={wr2:.1f}%  P&L={pnl2:+.1f}pt (~${usd2:+,.0f}){marker}")
            vol_rows.append({"vol_win": vw, "pct_win": vpw, "auc": auc2,
                             "n": n2, "per_day": n2/test_days,
                             "wr": wr2, "pnl": pnl2, "usd": usd2})
    print()

    # Full breakdown for best re-entry and best single configs
    if not summary_rows:
        print("No results."); return

    df_sum = pd.DataFrame([{k:v for k,v in r.items() if k!="_trades"}
                            for r in summary_rows])

    print("═" * 70)
    print("SUMMARY — sorted by total P&L")
    print("═" * 70)
    print(df_sum[["move_thr","reentry","n","per_day","wr","avg_pnl",
                  "total_pnl","usd","auc"]]
          .sort_values("total_pnl", ascending=False)
          .to_string(index=False, float_format="{:.2f}".format))

    # TOD detail for best single and best re-entry
    for reentry_flag, label in [(False, "single"), (True, "re-entry")]:
        subset = [r for r in summary_rows if r["reentry"] == reentry_flag]
        if not subset:
            continue
        best = max(subset, key=lambda r: r["total_pnl"])
        trades = best["_trades"]
        test_days = int(best["n"] / best["per_day"])
        print(f"\n── TOD Breakdown — best {label}: "
              f"move≥{best['move_thr']:.0f}bp  "
              f"({len(trades)} trades  {best['per_day']:.2f}/day  WR={best['wr']:.1f}%) ──")
        for win_label, t0, t1 in WINDOWS:
            mask = (trades["bar_time"] >= t0) & (trades["bar_time"] < t1)
            report_window(win_label, trades[mask], tick_val)

        print(f"\n  Monthly P&L:")
        t2 = trades.copy()
        t2["month"] = pd.to_datetime(t2["date"]).dt.to_period("M")
        monthly = t2.groupby("month").agg(
            n=("pnl_pts","count"),
            wr=("outcome", lambda x: (x=="TARGET").mean()*100),
            pnl=("pnl_pts","sum"),
        ).reset_index()
        for _, row in monthly.iterrows():
            bar = "█" * max(0, int(row["pnl"] / 2))
            neg = "░" * max(0, int(-row["pnl"] / 2))
            print(f"  {str(row['month']):<8}  n={int(row['n']):3d}  "
                  f"WR={row['wr']:5.1f}%  pnl={row['pnl']:+6.1f}pt  {bar}{neg}")
    print()


if __name__ == "__main__":
    main()
