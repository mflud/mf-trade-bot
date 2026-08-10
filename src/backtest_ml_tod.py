"""
backtest_ml_tod.py — Time-of-day breakdown for the ML strategy.

Uses the exact same feature engineering, model, and signal gates as
ml_trading_bot.py.  On the test set, simulates trades bar-by-bar and
groups results into user-defined time windows.

Run from repo root:
    python src/backtest_ml_tod.py           # MES (default)
    python src/backtest_ml_tod.py MNQ
"""

import sys
import numpy as np
import pandas as pd
from datetime import time as dtime

# Re-use all feature engineering + model code from the live bot
from ml_trading_bot import (
    SYMBOL_CONFIGS, ET,
    PRE_START, RTH_START, RTH_END,
    BLACKOUT_END, SESSION_END,
    LOOKBACK, SHORT_N, VOL_WINDOW, VOL_PCT_WIN,
    PROB_THRESHOLD, VOL_REGIME_MIN,
    load_training_data, build_2min_features, get_feature_cols,
    add_targets, train_model,
)

# ── Time-of-day windows ───────────────────────────────────────────────────────

WINDOWS = [
    ("9:40–11:00", dtime(9, 40), dtime(11,  0)),
    ("11:00–13:00", dtime(11, 0), dtime(13,  0)),
    ("13:00–15:45", dtime(13, 0), dtime(15, 45)),
]


def pts_from_bps(price: float, bps: float, tick_size: float) -> float:
    raw   = price * bps / 10000.0
    ticks = int(raw / tick_size) + (1 if raw % tick_size > 0 else 0)
    return round(ticks * tick_size, 4)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(test: pd.DataFrame, clf, fcols: list[str],
             cfg) -> pd.DataFrame:
    """
    Walk test rows in date order.  One trade at a time — no overlapping
    positions.  Returns a DataFrame of trades with columns:
        ts, date, bar_time, direction, entry, tgt, stp, outcome, pnl_pts
    """
    feat_arr = test[fcols].values
    proba    = clf.predict_proba(feat_arr)[:, 1]

    rows     = test.reset_index(drop=True)
    trades   = []
    in_trade = False

    for i, row in rows.iterrows():
        bar_t = row["ts"].time()

        # ── Manage open position ─────────────────────────────────────────────
        if in_trade:
            # Close at market if new day or past session end
            if row["date"] != trades[-1]["date"] or bar_t >= SESSION_END:
                exit_px = row["open"]
                pnl = (exit_px - entry) if direction == "LONG" else (entry - exit_px)
                trades[-1].update(outcome="EXPIRED", pnl_pts=pnl)
                in_trade = False
                # fall through to check entry on this bar
            else:
                h, l = row["high"], row["low"]
                if direction == "LONG":
                    if h >= entry + tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt)
                        in_trade = False; continue
                    elif l <= entry - stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp)
                        in_trade = False; continue
                else:
                    if l <= entry - tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt)
                        in_trade = False; continue
                    elif h >= entry + stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp)
                        in_trade = False; continue
                continue  # still in trade, skip entry logic

        # ── Entry gate ───────────────────────────────────────────────────────
        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue

        prob       = proba[i]
        vol_regime = row["vol_regime"]
        r2m_1      = row.get("r2m_1", 0.0)

        if prob < PROB_THRESHOLD or vol_regime < VOL_REGIME_MIN:
            continue
        if np.isnan(r2m_1):
            continue

        direction = "LONG" if r2m_1 >= 0 else "SHORT"
        entry     = row["close"]
        tgt       = pts_from_bps(entry, cfg.target_bps, cfg.tick_size)
        stp       = pts_from_bps(entry, cfg.stop_bps,   cfg.tick_size)

        trades.append({
            "ts":        row["ts"],
            "date":      row["date"],
            "bar_time":  bar_t,
            "direction": direction,
            "entry":     entry,
            "tgt":       tgt,
            "stp":       stp,
            "prob":      prob,
            "outcome":   None,
            "pnl_pts":   None,
        })
        in_trade = True

    df_t = pd.DataFrame(trades)
    df_t = df_t.dropna(subset=["outcome"]).reset_index(drop=True)
    return df_t


# ── Reporting ─────────────────────────────────────────────────────────────────

def report_window(label: str, trades: pd.DataFrame, tick_val: float):
    n       = len(trades)
    if n == 0:
        print(f"  {label:<14}  no trades")
        return
    wins    = (trades["outcome"] == "TARGET").sum()
    expired = (trades["outcome"] == "EXPIRED").sum()
    wr      = wins / n * 100
    pnl     = trades["pnl_pts"].sum()
    avg_pnl = trades["pnl_pts"].mean()
    pnl_usd = pnl * tick_val * 4
    exp_str = f"  exp={expired}" if expired > 0 else ""
    print(f"  {label:<14}  trades={n:3d}  WR={wr:5.1f}%  "
          f"avg={avg_pnl:+.2f}pt  total={pnl:+.1f}pt  (~${pnl_usd:+,.0f}){exp_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    cfg = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol: {sym}.  Use MES or MNQ.")
        sys.exit(1)

    tick_val = {"MES": 1.25, "MNQ": 0.50}[sym]   # $ per tick (0.25pt)

    print(f"\n=== ML Time-of-Day Backtest  ·  {sym} ===\n")
    print("Loading bars …", flush=True)
    df1 = load_training_data(cfg)

    print("Building features …", end=" ", flush=True)
    df2 = build_2min_features(df1, drop_warmup=True)
    print(f"{len(df2):,} 2-min bars", flush=True)

    print("Adding targets …", end=" ", flush=True)
    df2_tgt = add_targets(df2, threshold_bps=cfg.threshold_bps, horizon=1)
    print(f"{len(df2_tgt):,} labeled bars", flush=True)

    # Train/test split
    fcols   = get_feature_cols()
    dates   = sorted(df2_tgt["date"].unique())
    cutoff  = dates[int(len(dates) * 0.75)]
    train   = df2_tgt[df2_tgt["date"] <  cutoff]
    test    = df2_tgt[df2_tgt["date"] >= cutoff]

    print(f"Train: {len(dates[:int(len(dates)*0.75)])} days  "
          f"({dates[0]} → {cutoff})  "
          f"Test: {len(dates) - int(len(dates)*0.75)} days  "
          f"({cutoff} → {dates[-1]})\n", flush=True)

    print("Training model …", end=" ", flush=True)
    clf, fcols, info = train_model(df1, cfg)
    print(f"AUC={info['auc']:.4f}  base_rate={info['base_rate']*100:.1f}%\n",
          flush=True)

    print("Simulating trades on test set …", flush=True)
    test_feat = test[test["date"] >= cutoff].reset_index(drop=True)
    trades    = simulate(test_feat, clf, fcols, cfg)

    total_n   = len(trades)
    total_pnl = trades["pnl_pts"].sum() if total_n > 0 else 0.0
    total_wr  = (trades["outcome"] == "TARGET").mean() * 100 if total_n > 0 else 0.0
    test_days = test["date"].nunique()

    print(f"\n── Overall ({test_days} test days) ─────────────────────────────────────")
    print(f"  Trades: {total_n}  ({total_n/test_days:.1f}/day)  "
          f"WR: {total_wr:.1f}%  "
          f"Total P&L: {total_pnl:+.1f}pt  "
          f"(~${total_pnl * tick_val * 4:+,.0f})\n")

    print("── By Time Window ──────────────────────────────────────────────────")
    for label, t_start, t_end in WINDOWS:
        mask   = (trades["bar_time"] >= t_start) & (trades["bar_time"] < t_end)
        report_window(label, trades[mask], tick_val)

    print("\n── By Day-of-Week ──────────────────────────────────────────────────")
    trades["dow"] = pd.to_datetime(trades["date"]).dt.strftime("%a")
    for dow in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        mask = trades["dow"] == dow
        report_window(dow, trades[mask], tick_val)

    print("\n── Monthly P&L trend (test set) ────────────────────────────────────")
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M")
    monthly = trades.groupby("month").agg(
        n=("pnl_pts", "count"),
        wr=("outcome", lambda x: (x == "TARGET").mean() * 100),
        pnl=("pnl_pts", "sum"),
    ).reset_index()
    for _, row in monthly.iterrows():
        bar = "█" * max(0, int(row["pnl"] / 2))
        neg = "░" * max(0, int(-row["pnl"] / 2))
        print(f"  {str(row['month']):<8}  n={int(row['n']):3d}  "
              f"WR={row['wr']:5.1f}%  pnl={row['pnl']:+6.1f}pt  "
              f"{bar}{neg}")

    print()


if __name__ == "__main__":
    main()
