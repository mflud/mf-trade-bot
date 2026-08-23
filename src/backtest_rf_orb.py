"""
backtest_rf_orb.py — Random Forest classifier to filter ORB setups on MES.

Data: mes_hist_1min.csv (2019-05-06 → 2026-03-12) + bars.db (2026-04-08 → present)
      = ~1,855 sessions, ~1,275 qualifying ORB setups (15-min range, width 0.15-0.50%)

ORB setup (per session, no direction pre-filter):
  Opening range = first 15 minutes (9:30-9:45 ET)
  Entry = first 1-min bar close that breaks ORB high (LONG) or ORB low (SHORT)
  Stop  = ORB midpoint (half-range)
  Target = 1× ORB width

Features per setup (all known before entry):
  orb_width_pct     — range / mid price (size of opening balance)
  gap_pct           — (today open − prev close) / prev close
  gap_dir           — sign of gap (+1 up, −1 down)
  against_gap       — 1 if fading gap (SHORT on gap-up, LONG on gap-down), 0 otherwise
  trade_dir         — +1 LONG, −1 SHORT
  mins_to_break     — how many minutes after ORB close before breakout (0=immediate)
  breakout_vol_ratio— volume at breakout bar / mean ORB-period volume
  orb_vol_ratio     — total ORB volume / rolling 20-session mean ORB volume
  prev_day_range_pct— prior RTH (H−L) / mid
  prev_day_ret      — prior session (close−open) / open
  realized_vol_5d   — std of last 5 daily log-returns (annualised ×√252)
  open_vs_ma20      — (today open − 20-session MA of closes) / 20-session MA
  orb_mid_vs_prev   — (ORB mid − prev close) / prev close (gap fill status)

Target: hit_target (1) = reached 1× ORB width; (0) = stopped or time-exited

Evaluation:
  Walk-forward: train on first 70% of sessions (date-ordered), test on last 30%
  Compare: no-filter baseline vs ML-filtered (prob ≥ threshold)
  Feature importance chart (text-based)

Usage:
    python src/backtest_rf_orb.py
    python src/backtest_rf_orb.py --threshold 0.60   # ML filter confidence
"""

import argparse
import math
import sqlite3
from datetime import time, date
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

ET = ZoneInfo("America/New_York")

HIST_CSV = "mes_hist_1min.csv"
BARS_DB  = "data/bars.db"

RTH_OPEN  = time(9, 30)
RTH_CLOSE = time(16, 0)
ORB_MIN   = 15           # opening range duration
ENTRY_WIN = 60           # minutes after ORB close to look for breakout
WIDTH_MIN = 0.0015
WIDTH_MAX = 0.0050
TGT_MULT  = 1.0          # target = 1× ORB width
MES_PV    = 5.0          # $ per MES point

SETTLE_H_START = 21
SETTLE_H_END   = 22


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_rth_1min() -> pd.DataFrame:
    frames = []

    csv_path = Path(HIST_CSV)
    if csv_path.exists():
        df_h = pd.read_csv(csv_path)
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= RTH_OPEN) & (t < RTH_CLOSE)].copy()
        frames.append(df_h)

    db = sqlite3.connect(BARS_DB)
    df_db = pd.read_sql(
        "SELECT ts,open,high,low,close,volume FROM bars "
        "WHERE symbol='MES' AND minutes=1 ORDER BY ts", db)
    db.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t2 = df_db["ts"].dt.time
    df_db = df_db[(t2 >= RTH_OPEN) & (t2 < RTH_CLOSE)].copy()
    frames.append(df_db)

    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True))

    df["date"] = df["ts"].dt.date
    df["hm"]   = df["ts"].dt.hour * 60 + df["ts"].dt.minute

    print(f"  {len(df):,} RTH 1-min bars  "
          f"({df['date'].min()} → {df['date'].max()})")
    return df


# ─── Per-session feature extraction ──────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each qualifying session, extract ORB setup features + trade outcome.
    Returns one row per setup (max 1 per session — first breakout direction).
    """
    sessions = sorted(df["date"].unique())
    by_date  = {d: g.reset_index(drop=True) for d, g in df.groupby("date")}

    # Rolling session context (computed as we iterate)
    prev_close: float | None  = None
    prev_open:  float | None  = None
    session_closes: list[float] = []   # for MA and vol
    orb_vols:  list[float] = []        # rolling ORB-period volumes

    records = []

    for d in sessions:
        sess = by_date[d]
        orb_hm_end = 9 * 60 + 30 + ORB_MIN

        # ─── ORB range ───
        orb_bars = sess[sess["hm"] < orb_hm_end]
        if len(orb_bars) < ORB_MIN - 1:
            # Update context and skip
            rth_close = float(sess["close"].iloc[-1])
            rth_open  = float(sess["open"].iloc[0])
            session_closes.append(rth_close)
            prev_close = rth_close
            prev_open  = rth_open
            if "volume" in orb_bars.columns:
                orb_vols.append(float(orb_bars["volume"].sum()))
            continue

        orb_high = float(orb_bars["high"].max())
        orb_low  = float(orb_bars["low"].min())
        orb_mid  = (orb_high + orb_low) / 2
        orb_width = orb_high - orb_low
        orb_width_pct = orb_width / orb_mid

        if not (WIDTH_MIN <= orb_width_pct <= WIDTH_MAX):
            rth_close = float(sess["close"].iloc[-1])
            rth_open  = float(sess["open"].iloc[0])
            session_closes.append(rth_close)
            prev_close = rth_close
            prev_open  = rth_open
            if "volume" in orb_bars.columns:
                orb_vols.append(float(orb_bars["volume"].sum()))
            continue

        rth_open_px = float(sess["open"].iloc[0])

        # ─── Context features ───
        gap_pct  = (rth_open_px - prev_close) / prev_close if prev_close else 0.0
        gap_dir  = np.sign(gap_pct)

        # 20-session MA of closes
        ma20 = float(np.mean(session_closes[-20:])) if len(session_closes) >= 5 else None
        open_vs_ma20 = (rth_open_px - ma20) / ma20 if ma20 else 0.0

        # Prior session return and range
        prev_range_pct = 0.0
        prev_day_ret   = 0.0
        if prev_close and prev_open:
            prev_range = sess  # dummy — we need prev session's bars
            prev_day_ret = (prev_close - prev_open) / prev_open if prev_open else 0.0
        # Get prev session range from the last day we saw
        if len(sessions) > 1:
            prev_d_idx = sessions.index(d) - 1
            if prev_d_idx >= 0:
                prev_sess = by_date.get(sessions[prev_d_idx])
                if prev_sess is not None and len(prev_sess) > 0:
                    prev_range_pct = (prev_sess["high"].max() - prev_sess["low"].min()) / (
                        (prev_sess["high"].max() + prev_sess["low"].min()) / 2)

        # 5-day realized vol
        if len(session_closes) >= 7:
            c = np.array(session_closes[-7:])
            rets = np.log(c[1:] / c[:-1])
            realized_vol_5d = float(np.std(rets, ddof=1)) * math.sqrt(252)
        else:
            realized_vol_5d = 0.0

        # ORB volume ratio
        orb_vol_total = float(orb_bars["volume"].sum()) if "volume" in orb_bars.columns else 0.0
        orb_vol_mean  = float(np.mean(orb_vols[-20:])) if len(orb_vols) >= 5 else orb_vol_total
        orb_vol_ratio = orb_vol_total / orb_vol_mean if orb_vol_mean > 0 else 1.0

        # ORB midpoint vs prior close
        orb_mid_vs_prev = (orb_mid - prev_close) / prev_close if prev_close else 0.0

        # ─── Find breakout ───
        entry_bars = sess[
            (sess["hm"] >= orb_hm_end) &
            (sess["hm"] <  orb_hm_end + ENTRY_WIN)
        ]

        trade_dir   = None
        entry_price = None
        entry_hm    = None
        breakout_vol = 0.0

        for _, bar in entry_bars.iterrows():
            if bar["close"] > orb_high:
                trade_dir   = 1
                entry_price = float(bar["close"])
                entry_hm    = int(bar["hm"])
                breakout_vol = float(bar.get("volume", 0))
                break
            if bar["close"] < orb_low:
                trade_dir   = -1
                entry_price = float(bar["close"])
                entry_hm    = int(bar["hm"])
                breakout_vol = float(bar.get("volume", 0))
                break

        # Update context before deciding to skip
        rth_close = float(sess["close"].iloc[-1])
        session_closes.append(rth_close)
        if "volume" in orb_bars.columns:
            orb_vols.append(orb_vol_total)
        prev_close = rth_close
        prev_open  = rth_open_px

        if trade_dir is None or entry_price is None:
            continue   # no breakout in entry window

        # ─── Compute outcome ───
        stop_dist   = orb_width / 2
        target_dist = orb_width * TGT_MULT
        stop_price   = entry_price - trade_dir * stop_dist
        target_price = entry_price + trade_dir * target_dist

        after = sess[sess["hm"] > entry_hm]
        hit_target = 0
        for _, bar in after.iterrows():
            if trade_dir == 1:
                if bar["low"]  <= stop_price:   hit_target = 0; break
                if bar["high"] >= target_price: hit_target = 1; break
            else:
                if bar["high"] >= stop_price:   hit_target = 0; break
                if bar["low"]  <= target_price: hit_target = 1; break

        # ─── Build feature row ───
        against_gap  = 1 if trade_dir != gap_dir and gap_dir != 0 else 0
        mins_to_break = entry_hm - orb_hm_end
        orb_vol_mean_per_bar = orb_vol_total / max(len(orb_bars), 1)
        breakout_vol_ratio = breakout_vol / orb_vol_mean_per_bar if orb_vol_mean_per_bar > 0 else 1.0

        records.append({
            "date":               d,
            "trade_dir":          trade_dir,
            "orb_width_pct":      orb_width_pct,
            "gap_pct":            gap_pct,
            "gap_dir":            gap_dir,
            "against_gap":        against_gap,
            "mins_to_break":      mins_to_break,
            "breakout_vol_ratio": breakout_vol_ratio,
            "orb_vol_ratio":      orb_vol_ratio,
            "prev_day_range_pct": prev_range_pct,
            "prev_day_ret":       prev_day_ret,
            "realized_vol_5d":    realized_vol_5d,
            "open_vs_ma20":       open_vs_ma20,
            "orb_mid_vs_prev":    orb_mid_vs_prev,
            "orb_width":          orb_width,
            "entry_price":        entry_price,
            "hit_target":         hit_target,
        })

    return pd.DataFrame(records)


# ─── Walk-forward evaluation ──────────────────────────────────────────────────

FEATURE_COLS = [
    "orb_width_pct",
    "gap_pct",
    "gap_dir",
    "against_gap",
    "trade_dir",
    "mins_to_break",
    "breakout_vol_ratio",
    "orb_vol_ratio",
    "prev_day_range_pct",
    "prev_day_ret",
    "realized_vol_5d",
    "open_vs_ma20",
    "orb_mid_vs_prev",
]


def evaluate(df: pd.DataFrame, threshold: float) -> None:
    df = df.dropna(subset=FEATURE_COLS + ["hit_target"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    n  = len(df)

    split = int(n * 0.70)
    train = df.iloc[:split]
    test  = df.iloc[split:]

    print(f"\n  Walk-forward split:  "
          f"train={len(train)} ({train['date'].min()} → {train['date'].max()})  "
          f"test={len(test)} ({test['date'].min()} → {test['date'].max()})")

    X_tr = train[FEATURE_COLS].values
    y_tr = train["hit_target"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["hit_target"].values

    # ── Baseline (no filter) on test set ──
    bl_wr  = y_te.mean()
    bl_n   = len(y_te)
    bl_avg = test["orb_width"].mean() * (bl_wr * TGT_MULT - (1 - bl_wr) * 0.5)
    bl_dol = bl_avg * bl_n * MES_PV

    print(f"\n  BASELINE (no ML filter)  n={bl_n}  WR={bl_wr:.1%}  "
          f"avg={bl_avg:+.3f}pt  total≈${bl_dol:+,.0f}")

    # ── Train RF ──
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    proba_rf = rf.predict_proba(X_te)[:, 1]
    auc_rf   = roc_auc_score(y_te, proba_rf)

    # ── Train GBM ──
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=10, random_state=42
    )
    gbm.fit(X_tr, y_tr)
    proba_gbm = gbm.predict_proba(X_te)[:, 1]
    auc_gbm   = roc_auc_score(y_te, proba_gbm)

    print(f"\n  Model AUC — RF: {auc_rf:.3f}   GBM: {auc_gbm:.3f}")

    # ── Threshold sweep ──
    print(f"\n  Threshold sweep  (prod params: threshold={threshold:.2f})")
    print(f"  {'Model':<5}  {'thr':>5}  {'n':>5}  {'n%':>5}  {'WR':>7}  "
          f"{'avg pt':>8}  {'total $':>9}  {'AUC':>6}")
    print(f"  {'─'*60}")

    best_rf = best_gbm = None

    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        for label, proba in [("RF", proba_rf), ("GBM", proba_gbm)]:
            mask   = proba >= thr
            sub    = test[mask]
            sub_y  = y_te[mask]
            if len(sub) < 5:
                continue
            wr2    = sub_y.mean()
            avg_w  = sub["orb_width"].mean()
            avg_pt = avg_w * (wr2 * TGT_MULT - (1 - wr2) * 0.5)
            tot_d  = avg_pt * len(sub) * MES_PV
            pct_n  = len(sub) / bl_n * 100
            flag   = " ◄" if thr == threshold else ""
            print(f"  {label:<5}  {thr:.2f}  {len(sub):5d}  {pct_n:4.0f}%  "
                  f"{wr2:7.1%}  {avg_pt:+8.3f}pt  ${tot_d:+9,.0f}{flag}")
            if thr == threshold:
                if label == "RF":  best_rf  = (sub, mask, proba)
                if label == "GBM": best_gbm = (sub, mask, proba)

    # ── Monthly breakdown at chosen threshold (RF) ──
    if best_rf:
        sub, mask, proba = best_rf
        print(f"\n  RF (threshold={threshold:.2f}) — monthly breakdown on test set:")
        print(f"  {'Month':<8}  {'n':>4}  {'WR':>7}  {'avg pt':>8}  {'total $':>9}")
        sub2 = sub.copy()
        sub2["hit"] = y_te[mask]
        sub2["ym"]  = sub2["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
        for ym, g in sub2.groupby("ym"):
            wr2   = g["hit"].mean()
            avg_w = g["orb_width"].mean()
            pt    = avg_w * (wr2 * TGT_MULT - (1 - wr2) * 0.5)
            tot   = pt * len(g) * MES_PV
            print(f"  {ym:<8}  {len(g):4d}  {wr2:7.1%}  {pt:+8.3f}pt  ${tot:+9,.0f}")

    # ── Feature importance ──
    print(f"\n  RF Feature Importance:")
    imp = sorted(zip(FEATURE_COLS, rf.feature_importances_), key=lambda x: -x[1])
    for feat, score in imp:
        bar = "█" * int(score * 40)
        print(f"  {feat:<22}  {score:.3f}  {bar}")

    # ── Direction breakdown ──
    print(f"\n  Direction breakdown (no filter, test set):")
    test2 = test.copy()
    test2["hit"] = y_te
    for d, g in test2.groupby("trade_dir"):
        label = "LONG" if d == 1 else "SHORT"
        wr2   = g["hit"].mean()
        avg_w = g["orb_width"].mean()
        pt    = avg_w * (wr2 - (1 - wr2) * 0.5)
        print(f"    {label:<6}  n={len(g):4d}  WR={wr2:.1%}  avg≈{pt:+.3f}pt")

    # ── Gap direction breakdown ──
    print(f"\n  Gap direction breakdown (no filter, test set):")
    for gap_d in [1, -1]:
        for trade_d in [1, -1]:
            g = test2[(test2["gap_dir"] == gap_d) & (test2["trade_dir"] == trade_d)]
            if len(g) < 3:
                continue
            wr2 = g["hit"].mean()
            desc = ("Gap↑→L" if gap_d==1 and trade_d==1 else
                    "Gap↑→S" if gap_d==1 and trade_d==-1 else
                    "Gap↓→L" if gap_d==-1 and trade_d==1 else "Gap↓→S")
            print(f"    {desc}  n={len(g):4d}  WR={wr2:.1%}")


# ─── Calibration ──────────────────────────────────────────────────────────────

def print_calibration(df: pd.DataFrame) -> None:
    """Show WR by gap_pct decile — reveals non-linear patterns."""
    df2 = df.copy()
    df2["gap_decile"] = pd.qcut(df2["gap_pct"], 10, labels=False, duplicates="drop")
    print(f"\n  WR by gap_pct decile:")
    for dec, g in df2.groupby("gap_decile"):
        rng = f"[{g['gap_pct'].min():+.3%}, {g['gap_pct'].max():+.3%}]"
        print(f"    d{int(dec)}  {rng:22s}  n={len(g):4d}  WR={g['hit_target'].mean():.1%}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(threshold: float):
    print(f"\n{'═'*65}")
    print(f"  ML ORB Filter  —  MES  (15-min range, 1× target, ½ stop)")
    print(f"{'═'*65}")

    print("\nLoading 1-min bars …", flush=True)
    df1 = load_rth_1min()

    print("Extracting ORB features …", flush=True)
    feat = extract_features(df1)
    print(f"  {len(feat)} qualifying ORB setups  "
          f"({feat['date'].min()} → {feat['date'].max()})")
    print(f"  Overall WR: {feat['hit_target'].mean():.1%}  "
          f"LONG: {len(feat[feat['trade_dir']==1])}  SHORT: {len(feat[feat['trade_dir']==-1])}")

    print_calibration(feat)
    evaluate(feat, threshold)

    print(f"\n{'═'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.60,
                        help="ML probability threshold for taking a trade (default: 0.60)")
    args = parser.parse_args()
    main(args.threshold)
