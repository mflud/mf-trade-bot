"""
backtest_tp_sweep.py — Sweep profit-target and stop-loss on the directional 2-min model.

Same architecture as the live directional model (move gate, sign-flip features,
re-entry), but sweeps target_bps × stop_bps × train_horizon to find the best
TP/SL combination for capturing larger trend moves.

Key insight: wider TP needs a longer train_horizon so the RF label resolution
window is long enough to actually see the target hit.

Sweep:
  target_bps   : [10, 15, 20, 25, 30, 40, 50]
  stop_bps     : [8, 10, 12, 15]   (always < target)
  horizon_bars : [3, 5, 10, 20]   (2-min bars = 6/10/20/40 min)
  move_thr_bps : MES=3, MNQ=4  (best from prior sweep, fixed)
  re-entry     : always enabled

Usage (from repo root):
    python src/backtest_tp_sweep.py           # MES
    python src/backtest_tp_sweep.py MNQ
"""

import sys
import math
import numpy as np
import pandas as pd
from copy import copy
from datetime import time as dtime
from itertools import product

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ml_trading_bot import (
    SYMBOL_CONFIGS, ET,
    PRE_START, RTH_START, RTH_END, BLACKOUT_END, SESSION_END,
    LOOKBACK, SHORT_N,
    PROB_THRESHOLD, VOL_REGIME_MIN,
    load_training_data, build_2min_features,
    get_feature_cols,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ── Sweep space ────────────────────────────────────────────────────────────────

TARGET_BPS   = [10, 15, 20, 25, 30, 40, 50]
STOP_BPS     = [8, 10, 12, 15]
HORIZON_BARS = [3, 5, 10, 20]   # 2-min bars → 6/10/20/40 min to resolve label

WINDOWS = [
    ("9:40–11:00",  dtime(9, 40),  dtime(11,  0)),
    ("11:00–13:00", dtime(11,  0), dtime(13,  0)),
    ("13:00–15:45", dtime(13,  0), dtime(15, 45)),
]


# ── Dataset labeling — fully vectorized ───────────────────────────────────────

def _precompute_labels(df2: pd.DataFrame, cfg, target_bps: float,
                       stop_bps: float, horizon: int) -> np.ndarray:
    """
    Vectorized forward-scan label computation.
    Returns an int8 array: 1=target hit first, 0=stop hit first, -1=inconclusive.

    For each bar i with direction d:
      LONG:  target if any high[i+1..i+hz] >= close[i]+tgt  AND that occurs
             before any low[i+1..i+hz] <= close[i]-stp
      SHORT: symmetric

    Strategy: for each lag k in 1..horizon, check if target or stop hit.
    Use a rolling "first hit" approach via cumulative arrays.
    """
    n      = len(df2)
    closes = df2["close"].values.astype(np.float64)
    highs  = df2["high"].values.astype(np.float64)
    lows   = df2["low"].values.astype(np.float64)
    dates  = df2["date"].values

    lb   = cfg.move_lookback_bars
    thr  = cfg.move_threshold_bps / 10000.0

    # Qualifying bar mask: same-day lookback AND |move| >= threshold
    valid_lb = np.zeros(n, dtype=bool)
    valid_lb[lb:] = (dates[lb:] == dates[:n-lb])
    move    = np.where(valid_lb, closes - np.concatenate([closes[lb:], np.zeros(lb)]), 0.0)
    # Use current close as price denominator for threshold
    move_shifted = np.zeros(n)
    move_shifted[lb:] = closes[lb:] - closes[:n-lb]
    thresh   = closes * thr
    qual     = valid_lb & (np.abs(move_shifted) >= thresh)

    direction = np.where(move_shifted > 0, 1, -1).astype(np.int8)

    # tgt_pts and stp_pts per bar (vectorized, rounded to tick)
    tick = cfg.tick_size
    tgt_pts = np.ceil(closes * target_bps / 10000.0 / tick) * tick
    stp_pts = np.ceil(closes * stop_bps   / 10000.0 / tick) * tick

    labels = np.full(n, -1, dtype=np.int8)

    for k in range(1, horizon + 1):
        # Bars at distance k forward; only valid if same day and within bounds
        i_max = n - k
        if i_max <= 0:
            break
        fwd_date_ok = dates[:i_max] == dates[k:n]
        # For qualifying, undecided bars: check if k-th bar ahead resolves
        undecided = qual[:i_max] & (labels[:i_max] == -1) & fwd_date_ok

        h = highs[k:][:i_max]
        l = lows[k:][:i_max]

        # LONG bars
        long_undec = undecided & (direction[:i_max] == 1)
        tgt_hit_l  = long_undec & (h >= closes[:i_max] + tgt_pts[:i_max])
        stp_hit_l  = long_undec & (l <= closes[:i_max] - stp_pts[:i_max])
        labels[:i_max][tgt_hit_l & ~stp_hit_l] = 1
        labels[:i_max][stp_hit_l & ~tgt_hit_l] = 0
        # If both hit on same bar: whichever was closer to entry wins (assume stop)
        labels[:i_max][tgt_hit_l & stp_hit_l] = 0

        # SHORT bars
        shrt_undec = undecided & (direction[:i_max] == -1)
        tgt_hit_s  = shrt_undec & (l <= closes[:i_max] - tgt_pts[:i_max])
        stp_hit_s  = shrt_undec & (h >= closes[:i_max] + stp_pts[:i_max])
        labels[:i_max][tgt_hit_s & ~stp_hit_s] = 1
        labels[:i_max][stp_hit_s & ~tgt_hit_s] = 0
        labels[:i_max][tgt_hit_s & stp_hit_s] = 0

    # Inconclusive bars (labels == -1) kept as -1 — caller will filter
    return labels, qual, direction, move_shifted


def build_dataset(df2: pd.DataFrame, cfg, target_bps: float, stop_bps: float,
                  horizon: int) -> pd.DataFrame:
    """Build training dataset using vectorized label computation."""
    df2 = df2[df2["ts"].dt.time >= BLACKOUT_END].reset_index(drop=True)

    labels, qual, direction, move_shifted = _precompute_labels(
        df2, cfg, target_bps, stop_bps, horizon)

    # Keep only qualifying, conclusive bars
    keep = qual & (labels >= 0)
    if not keep.any():
        return pd.DataFrame()

    df_k = df2[keep].copy().reset_index(drop=True)
    dir_k = direction[keep]
    lab_k = labels[keep]
    mov_k = move_shifted[keep]

    r2m_cols = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    r1m_cols = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]

    # Sign-flip directional features
    for c in r2m_cols + r1m_cols + ["ret_open"]:
        if c in df_k.columns:
            df_k[c] = df_k[c] * dir_k

    df_k["direction"]     = dir_k
    df_k["target"]        = lab_k.astype(int)
    df_k["move_size_bps"] = np.abs(mov_k) / df_k["close"].values * 10000.0

    return df_k.reset_index(drop=True)


# ── Simulation (re-entry always on) ───────────────────────────────────────────

def simulate(test: pd.DataFrame, clf, fcols: list[str], cfg,
             target_bps: float, stop_bps: float) -> pd.DataFrame:
    """Walk test rows with re-entry; use target_bps/stop_bps for live TP/SL."""
    feat_arr = test[fcols].values
    proba    = clf.predict_proba(feat_arr)[:, 1]
    rows     = test.reset_index(drop=True)
    trades   = []
    in_trade = False
    last_date = None

    for i, row in rows.iterrows():
        bar_t = row["ts"].time()
        today = row["date"]

        if today != last_date:
            last_date = today

        if in_trade:
            if today != trades[-1]["date"] or bar_t >= SESSION_END:
                trades[-1].update(outcome="EXPIRED", pnl_pts=0.0)
                in_trade = False
            else:
                h, l  = row["high"], row["low"]
                entry = trades[-1]["entry"]
                tgt   = trades[-1]["tgt"]
                stp   = trades[-1]["stp"]
                d     = trades[-1]["dir_int"]
                closed = False
                if d == 1:
                    if h >= entry + tgt:   trades[-1].update(outcome="TARGET",  pnl_pts=tgt);  closed=True
                    elif l <= entry - stp: trades[-1].update(outcome="STOPPED", pnl_pts=-stp); closed=True
                else:
                    if l <= entry - tgt:   trades[-1].update(outcome="TARGET",  pnl_pts=tgt);  closed=True
                    elif h >= entry + stp: trades[-1].update(outcome="STOPPED", pnl_pts=-stp); closed=True
                if closed:
                    in_trade = False
                    continue
                else:
                    continue

        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue
        if proba[i] < PROB_THRESHOLD or row["vol_regime"] < VOL_REGIME_MIN:
            continue

        d     = int(row["direction"])
        entry = row["close"]
        tgt   = cfg.pts_from_bps(entry, target_bps)
        stp   = cfg.pts_from_bps(entry, stop_bps)

        trades.append({
            "ts": row["ts"], "date": today, "bar_time": bar_t,
            "direction": "LONG" if d == 1 else "SHORT", "dir_int": d,
            "entry": entry, "tgt": tgt, "stp": stp,
            "prob": proba[i], "outcome": None, "pnl_pts": None,
        })
        in_trade = True

    if not trades:
        return pd.DataFrame()
    df_t = pd.DataFrame(trades)
    if "outcome" in df_t.columns:
        df_t = df_t.dropna(subset=["outcome"])
    return df_t.reset_index(drop=True)


# ── Reporting ──────────────────────────────────────────────────────────────────

def detail_report(label: str, trades: pd.DataFrame, auc: float,
                  train_days: int, test_days: int, tick_val: float,
                  point_value: float):
    if trades.empty:
        return
    n   = len(trades)
    wr  = (trades["outcome"] == "TARGET").mean() * 100
    pnl = trades["pnl_pts"].sum()
    usd = pnl * point_value
    avg = trades["pnl_pts"].mean()
    exp = (trades["outcome"] == "EXPIRED").sum()
    print(f"\n{'═'*72}")
    print(f"  {label}")
    print(f"{'═'*72}")
    print(f"  AUC={auc:.4f}   train={train_days}d  test={test_days}d")
    print(f"  Trades: {n} ({n/test_days:.2f}/day)  WR={wr:.1f}%  "
          f"avg={avg:+.2f}pt  P&L={pnl:+.1f}pt (~${usd:+,.0f})"
          + (f"  exp={exp}" if exp else ""))

    print("\n  Time-of-day:")
    for lbl, t0, t1 in WINDOWS:
        w = trades[(trades["bar_time"] >= t0) & (trades["bar_time"] < t1)]
        if w.empty:
            print(f"    {lbl:<14} —"); continue
        nw  = len(w)
        wrw = (w["outcome"] == "TARGET").mean() * 100
        pw  = w["pnl_pts"].sum()
        uw  = pw * point_value
        print(f"    {lbl:<14} n={nw:4d}  WR={wrw:5.1f}%  pnl={pw:+.1f}pt (~${uw:+,.0f})")

    print("\n  Monthly P&L:")
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
        print(f"    {str(row['month']):<8}  n={int(row['n']):3d}  "
              f"WR={row['wr']:5.1f}%  pnl={row['pnl']:+6.1f}pt  {bar}{neg}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    cfg = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    tick_val    = {"MES": 1.25, "MNQ": 0.50}[sym]
    point_value = {"MES": 5.0,  "MNQ": 2.0}[sym]

    print(f"\n=== TP/SL Sweep  ·  Directional Re-entry Model  ·  {sym} ===\n")
    print(f"  move_thr={cfg.move_threshold_bps:.0f}bp  move_lb={cfg.move_lookback_bars}bar  re-entry=ON\n")

    print("Loading bars …", flush=True)
    df1 = load_training_data(cfg)

    print("Building 2-min features …", end=" ", flush=True)
    df2 = build_2min_features(df1, drop_warmup=True)
    print(f"{len(df2):,} bars\n", flush=True)

    fcols = get_feature_cols()

    configs = [
        (tg, st, hz)
        for tg, st, hz in product(TARGET_BPS, STOP_BPS, HORIZON_BARS)
        if st < tg
    ]
    print(f"Sweep: {len(configs)} configs "
          f"({len(TARGET_BPS)}tgt × {len(STOP_BPS)}stp × {len(HORIZON_BARS)}hz "
          f"with stp<tgt) …\n")

    summary_rows = []

    for ci, (tg, st, hz) in enumerate(configs):
        if ci % 20 == 0:
            print(f"  {ci}/{len(configs)} …", flush=True)

        df_dir = build_dataset(df2, cfg, tg, st, hz)
        if len(df_dir) < 200:
            continue

        dates  = sorted(df_dir["date"].unique())
        cutoff = dates[int(len(dates) * 0.75)]
        train  = df_dir[df_dir["date"] <  cutoff]
        test   = df_dir[df_dir["date"] >= cutoff]
        test_days  = test["date"].nunique()
        train_days = train["date"].nunique()

        if len(train) < 150 or len(test) < 50 or test_days < 50:
            continue


        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[fcols].values, train["target"].astype(int).values)

        proba = clf.predict_proba(test[fcols].values)[:, 1]
        auc   = roc_auc_score(test["target"].astype(int).values, proba)

        trades = simulate(test, clf, fcols, cfg, tg, st)
        if trades.empty:
            continue

        n_t     = len(trades)
        wr_t    = (trades["outcome"] == "TARGET").mean() * 100
        pnl_t   = trades["pnl_pts"].sum()
        usd_t   = pnl_t * point_value
        avg_t   = trades["pnl_pts"].mean()
        exp_t   = (trades["outcome"] == "EXPIRED").sum()
        n_months = pd.to_datetime(trades["date"]).dt.to_period("M").nunique()
        monthly_pnl = (trades.assign(
            month=pd.to_datetime(trades["date"]).dt.to_period("M"))
            .groupby("month")["pnl_pts"].sum())
        conc = monthly_pnl.abs().max() / (trades["pnl_pts"].abs().sum() + 1e-9)

        summary_rows.append({
            "target": tg, "stop": st, "horizon": hz,
            "n": n_t, "per_day": n_t / test_days,
            "wr": wr_t, "avg_pnl": avg_t,
            "total_pnl": pnl_t, "usd": usd_t, "auc": auc,
            "months": n_months, "conc": conc,
            "_trades": trades, "_test_days": test_days,
            "_train_days": train_days, "_clf": clf,
        })

    if not summary_rows:
        print("No results."); return

    df_sum = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                            for r in summary_rows])

    print(f"\n{'═'*90}")
    print(f"  SWEEP RESULTS  ·  {sym}  ·  move_thr={cfg.move_threshold_bps:.0f}bp  re-entry=ON")
    print(f"  Sorted by total P&L  |  conc = fraction of P&L in single best month")
    print(f"{'═'*90}")
    cols = ["target","stop","horizon","n","per_day","wr","avg_pnl",
            "total_pnl","usd","auc","months","conc"]
    df_show = df_sum[cols].sort_values("total_pnl", ascending=False).head(30)
    print(df_show.to_string(index=False, float_format="{:.2f}".format))

    # Broad configs: spread across at least N months
    broad = df_sum[df_sum["months"] >= 6].sort_values("total_pnl", ascending=False)
    if not broad.empty:
        print(f"\n{'─'*90}")
        print(f"  CONFIGS WITH ≥6 MONTHS ACTIVE (broader distribution)")
        print(f"{'─'*90}")
        print(broad[cols].head(20).to_string(index=False, float_format="{:.2f}".format))

    # Detail report for top-5 overall and top-3 broad
    top5 = sorted(summary_rows, key=lambda r: r["total_pnl"], reverse=True)[:5]
    print(f"\n{'═'*72}")
    print(f"  DETAIL REPORTS — TOP 5 BY TOTAL P&L")

    for r in top5:
        lbl = (f"tgt={r['target']}bp  stp={r['stop']}bp  hz={r['horizon']}bar"
               f"  ({r['n']} trades  {r['per_day']:.2f}/day  WR={r['wr']:.1f}%"
               f"  {r['months']}mo)")
        detail_report(lbl, r["_trades"], r["auc"],
                      r["_train_days"], r["_test_days"],
                      tick_val, point_value)

    broad_rows = [r for r in summary_rows if r["months"] >= 6]
    if broad_rows:
        broad_rows.sort(key=lambda r: r["total_pnl"], reverse=True)
        print(f"\n{'═'*72}")
        print(f"  DETAIL REPORTS — TOP 3 BROAD (≥6 MONTHS)")
        for r in broad_rows[:3]:
            lbl = (f"tgt={r['target']}bp  stp={r['stop']}bp  hz={r['horizon']}bar"
                   f"  ({r['n']} trades  {r['per_day']:.2f}/day  WR={r['wr']:.1f}%"
                   f"  {r['months']}mo)")
            detail_report(lbl, r["_trades"], r["auc"],
                          r["_train_days"], r["_test_days"],
                          tick_val, point_value)

    print()


if __name__ == "__main__":
    main()
