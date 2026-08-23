"""
backtest_pl_5s_ml.py — ML filter on top of PL_MOM qualifying bars (5s bars, 30s window).

The rule-based PL_MOM (live in trading_bot) uses:
  Entry : PL >= pl_thr  AND  |net_move| >= move_thr_bps  (30s / 6-bar window)
  Direction : sign of net move
  Exit  : rolling PL (recomputed each bar) drops to <= EXIT_PL
           OR price hits STOP_BPS from entry
           OR max hold MAX_HOLD_BARS (120s)

This script trains a Random Forest on the SAME qualifying bars to predict
which entries will end up profitable with the live exit logic.

Label = 1 if final P&L > 0 using live exit, 0 otherwise.

Features (sign-flipped by direction so model learns "continuation"):
  ret_1..ret_6   individual 5s log returns in window (ret_1 = most recent bar)
  pl             PL value at entry (already >= pl_thr)
  net_ret_bps    absolute net move in window (always +)
  realized_vol   std of 6 returns
  vol_accel      last-3 bar volume / first-3 bar volume  (>1 = accelerating)
  vol_ratio      window avg volume / daily avg volume
  ret_open_bps   log(close / open_930) × 10000, sign-flipped
  vol_regime     percentile rank of today's rvol vs 60-day history
  atr_ratio      today's avg 5s bar range / 20-day avg range
  tod_sin/cos    cyclic time-of-day

Qualifying gate sweep:
  pl_thr       : [0.60, 0.70, 0.80]
  move_thr_bps : [8, 12, 16, 20]

Exit parameters (fixed to match live strategy):
  EXIT_PL    = 0.40  (rolling PL threshold to exit)
  STOP_BPS   = 10    (bp from entry)
  MAX_HOLD   = 24    (bars = 120s)
  MIN_HOLD   = 2     (bars = 10s, before PL exit allowed)

Usage (from repo root):
  python src/backtest_pl_5s_ml.py           # MES
  python src/backtest_pl_5s_ml.py MNQ
  python src/backtest_pl_5s_ml.py MES single
  python src/backtest_pl_5s_ml.py MES vol_gate   # vol_regime sweep, no ML
"""

import math
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import time as dtime
from zoneinfo import ZoneInfo
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ──────────────────────────────────────────────────────────────────

ET           = ZoneInfo("America/New_York")
PRE_START    = dtime(8, 30)
RTH_START    = dtime(9, 30)
RTH_END      = dtime(16, 0)
BLACKOUT_END = dtime(9, 40)
SESSION_END  = dtime(15, 55)
SETTLEMENT_UTC_H = (21, 22)

SYMBOL_CONFIGS = {
    "MES": dict(csv="es_hist_5sec.csv",  tick_size=0.25, point_value=5.0),
    "MNQ": dict(csv="nq_hist_5sec.csv",  tick_size=0.25, point_value=2.0),
}

WINDOW       = 6      # bars (30s) — matches live PL_MOM
MAX_HOLD     = 24     # bars (120s) max hold
MIN_HOLD     = 2      # bars (10s) before PL exit allowed
EXIT_PL      = 0.40   # rolling PL exit threshold
STOP_BPS     = 10.0   # stop loss in bp from entry

TRAIN_FRAC   = 0.75
VOL_HIST_DAYS = 60
ATR_HIST_DAYS = 20

# Qualifying gate sweep
PL_THRS   = [0.60, 0.70, 0.80]
MOVE_THRS = [8, 12, 16, 20]

PROB_THRESHOLD = 0.55   # default ML filter

SINGLE_CFG = dict(pl_thr=0.70, move_thr=12)

# Months to exclude from ALL analysis — remove extreme-regime outliers so that
# the backtest reflects edge in normal market conditions.
# April 2025 = tariff crash (extreme vol, not representative of base rate edge).
EXCLUDE_MONTHS: set[tuple[int,int]] = {(2025, 4)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tod_frac(t: dtime) -> float:
    mins  = t.hour * 60 + t.minute
    start = 9 * 60 + 40
    end   = 15 * 60 + 55
    return max(0.0, min(1.0, (mins - start) / max(end - start, 1)))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_5s(symbol: str) -> pd.DataFrame:
    cfg = SYMBOL_CONFIGS[symbol]
    df  = pd.read_csv(cfg["csv"])
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)

    h  = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_H[0]) & (h < SETTLEMENT_UTC_H[1]))].copy()

    ts_et       = df["ts"].dt.tz_convert(ET)
    bt          = ts_et.dt.time
    df["bar_time"] = bt
    df["date"]     = ts_et.dt.date
    df = df[(bt >= PRE_START) & (bt < RTH_END)].copy().reset_index(drop=True)

    df["gap"] = (df["ts"].diff() > pd.Timedelta(seconds=10)).values
    df.iloc[0, df.columns.get_loc("gap")] = True

    if EXCLUDE_MONTHS:
        mask = df["date"].apply(lambda d: (d.year, d.month) not in EXCLUDE_MONTHS)
        df = df[mask].copy().reset_index(drop=True)
        df["gap"] = (df["ts"].diff() > pd.Timedelta(seconds=10)).values
        df.iloc[0, df.columns.get_loc("gap")] = True
        excl = ", ".join(f"{y}-{m:02d}" for y, m in sorted(EXCLUDE_MONTHS))
        print(f"  [excluded {excl}]", flush=True)

    open_930 = (df[df["bar_time"] == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df = df.join(open_930, on="date")

    print(f"  {len(df):,} 5s bars  {df['date'].min()} → {df['date'].max()}",
          flush=True)
    return df


# ── Rolling feature computation ────────────────────────────────────────────────

def _build_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Precompute all rolling features on the full RTH 5s series.
    Called ONCE; result reused for every config.
    """
    df2    = df.copy().reset_index(drop=True)
    n      = len(df2)
    closes = df2["close"].values.astype(np.float64)
    highs  = df2["high"].values.astype(np.float64)
    lows   = df2["low"].values.astype(np.float64)
    gaps   = df2["gap"].values.astype(bool)

    # Log returns, NaN at gaps
    lr = np.empty(n); lr[:] = np.nan
    lr[1:] = np.log(closes[1:] / closes[:-1])
    lr[gaps] = np.nan
    abs_lr = np.abs(lr)

    lr_s   = pd.Series(lr)
    abs_s  = pd.Series(abs_lr)
    gap_s  = pd.Series(gaps.astype(float))

    roll     = lr_s.rolling(WINDOW, min_periods=WINDOW)
    roll_abs = abs_s.rolling(WINDOW, min_periods=WINDOW)
    gap_roll = gap_s.rolling(WINDOW, min_periods=WINDOW).sum()
    has_gap  = gap_roll > 0

    net_ret = roll.sum()
    rvol    = roll.std(ddof=1)
    abs_sum = roll_abs.sum()
    pl_raw  = net_ret.abs() / abs_sum.replace(0, np.nan)

    for s in [net_ret, rvol, abs_sum, pl_raw]:
        s[has_gap] = np.nan

    df2["net_ret"]      = net_ret.values
    df2["realized_vol"] = rvol.values
    df2["pl"]           = pl_raw.values
    df2["move_bps"]     = (net_ret.abs() * 10000).values

    # Individual 5s returns (ret_1 = most recent, ret_6 = oldest)
    for lag in range(1, WINDOW + 1):
        col = lr_s.shift(lag - 1).copy()
        col[has_gap] = np.nan
        df2[f"ret_{lag}"] = col.values

    # Volume features
    if "volume" in df2.columns:
        vol_s   = df2["volume"].astype(float)
        half    = WINDOW // 2   # 3
        win_v   = vol_s.rolling(WINDOW, min_periods=WINDOW).mean()
        first_v = vol_s.rolling(half, min_periods=half).mean()
        last_v  = vol_s.rolling(half, min_periods=half).mean()
        vol_accel = last_v / (first_v.shift(half) + 1e-9)
        vol_accel[has_gap] = np.nan
        df2["vol_accel"] = vol_accel.values

        day_vol_avg = df2.groupby("date")["volume"].transform("mean")
        df2["vol_ratio"] = (win_v / (day_vol_avg + 1e-9)).values
    else:
        df2["vol_accel"] = np.nan
        df2["vol_ratio"] = np.nan

    df2["ret_open_bps"] = np.log(closes / df2["open_930"].values) * 10000

    # Daily rolling baselines for vol_regime and atr_ratio
    df2["bar_range"] = highs - lows
    day_rvol  = df2.groupby("date")["realized_vol"].mean()
    day_range = df2.groupby("date")["bar_range"].mean()
    days      = sorted(day_rvol.index)
    vol_reg_map   = {}
    atr_ratio_map = {}
    hist_vol   = []
    hist_range = []
    for d in days:
        pv = [v for v in hist_vol[-VOL_HIST_DAYS:]   if not np.isnan(v)]
        pr = [v for v in hist_range[-ATR_HIST_DAYS:] if not np.isnan(v)]
        v  = float(day_rvol.get(d, np.nan))
        r  = float(day_range.get(d, np.nan))
        vol_reg_map[d]   = (float(np.mean(np.array(pv) <= v))
                            if len(pv) >= 10 else np.nan)
        atr_ratio_map[d] = (v / float(np.mean(pr))
                            if len(pr) >= 5 and float(np.mean(pr)) > 0
                            else np.nan)
        hist_vol.append(v)
        hist_range.append(r)

    df2["vol_regime"] = df2["date"].map(vol_reg_map)
    df2["atr_ratio"]  = df2["date"].map(atr_ratio_map)

    frac = df2["bar_time"].apply(_tod_frac)
    df2["tod_sin"] = np.sin(2 * np.pi * frac)
    df2["tod_cos"] = np.cos(2 * np.pi * frac)

    return df2


# ── Live exit simulation (for labels AND simulation) ──────────────────────────

def _live_exit_pnl(i: int, direction: int, entry: float,
                   closes, highs, lows, pl_arr, dates) -> float:
    """
    Simulate one trade from bar i using the live PL_MOM exit logic:
      - Stop at STOP_BPS from entry (checked against high/low each bar)
      - After MIN_HOLD bars: exit when rolling PL <= EXIT_PL
      - Max hold MAX_HOLD bars
    Returns P&L in bps (from entry close). NaN if can't exit (end-of-data).
    """
    n        = len(closes)
    stop_pt  = entry * STOP_BPS / 10000.0
    day_0    = dates[i]

    for k in range(1, MAX_HOLD + 1):
        j = i + k
        if j >= n:
            raw = (closes[j - 1] - entry) * direction / entry * 10000.0
            return max(raw, -STOP_BPS)
        if dates[j] != day_0:
            # End-of-session exit: cap at stop loss
            raw = (closes[j - 1] - entry) * direction / entry * 10000.0
            return max(raw, -STOP_BPS)

        h_j  = highs[j]
        lv_j = lows[j]

        # Stop check (using high/low for accuracy)
        if direction == 1 and lv_j <= entry - stop_pt:
            return -STOP_BPS
        if direction == -1 and h_j >= entry + stop_pt:
            return -STOP_BPS

        # PL-based exit (only after min hold)
        cur_pl = pl_arr[j]
        if k >= MIN_HOLD and not np.isnan(cur_pl) and cur_pl <= EXIT_PL:
            return (closes[j] - entry) * direction / entry * 10000.0

    # Max hold: exit at close of bar i + MAX_HOLD (cap at stop)
    j = i + MAX_HOLD
    raw = ((closes[j] - entry) if j < n else (closes[n - 1] - entry))
    return max(raw * direction / entry * 10000.0, -STOP_BPS)


# ── Dataset builder ───────────────────────────────────────────────────────────

FEATURE_COLS = (
    [f"ret_{i}" for i in range(1, WINDOW + 1)] +
    ["pl", "net_ret_bps", "realized_vol",
     "vol_accel", "vol_ratio", "ret_open_bps",
     "vol_regime", "atr_ratio",
     "tod_sin", "tod_cos"]
)


def build_dataset(df_feat: pd.DataFrame,
                  pl_thr: float, move_thr_bps: float) -> pd.DataFrame:
    """
    Apply qualifying gate (PL >= pl_thr, |move| >= move_thr_bps),
    then simulate live exit for each qualifying bar to get labels.
    label=1 if P&L > 0, label=0 otherwise.
    """
    closes = df_feat["close"].values.astype(np.float64)
    highs  = df_feat["high"].values.astype(np.float64)
    lows   = df_feat["low"].values.astype(np.float64)
    pl_arr = df_feat["pl"].values
    dates  = df_feat["date"].values
    times  = df_feat["bar_time"].values
    net_ret = df_feat["net_ret"].values

    n = len(df_feat)

    time_ok   = np.array([BLACKOUT_END <= t < SESSION_END for t in times])
    ret_cols  = [f"ret_{i}" for i in range(1, WINDOW + 1)]
    has_feats = df_feat[ret_cols].notna().all(axis=1).values

    qual = (
        time_ok &
        has_feats &
        (pl_arr >= pl_thr) &
        (df_feat["move_bps"].values >= move_thr_bps) &
        ~np.isnan(pl_arr)
    )

    qual_idx = np.where(qual)[0]
    if len(qual_idx) == 0:
        return pd.DataFrame()

    direction = np.where(net_ret > 0, 1, -1).astype(int)

    # Compute live exit P&L for each qualifying bar
    pnl_bps = np.full(n, np.nan)
    for i in qual_idx:
        pnl_bps[i] = _live_exit_pnl(
            i, direction[i], closes[i],
            closes, highs, lows, pl_arr, dates)

    keep = qual & ~np.isnan(pnl_bps)
    if not keep.any():
        return pd.DataFrame()

    df_k  = df_feat[keep].copy().reset_index(drop=True)
    dir_k = direction[keep]

    # Sign-flip directional features
    for c in ret_cols + ["net_ret", "ret_open_bps"]:
        if c in df_k.columns:
            df_k[c] = df_k[c] * dir_k

    df_k["net_ret_bps"] = df_k["net_ret"].abs() * 10000
    df_k["direction"]   = dir_k
    df_k["pnl_bps"]     = pnl_bps[keep]
    df_k["target"]      = (pnl_bps[keep] > 0).astype(int)
    df_k["entry"]       = df_k["close"]

    return df_k.reset_index(drop=True)


# ── Simulation ────────────────────────────────────────────────────────────────

def _run_simulation(test: pd.DataFrame, df_all: pd.DataFrame,
                    qual_map: dict) -> pd.DataFrame:
    """
    Walk all RTH 5s bars in test period; enter on qualifying bars matching qual_map.
    Uses live exit logic (PL drop + stop + max hold).
    """
    if not qual_map:
        return pd.DataFrame()

    accepted = [(accept, r) for accept, r in qual_map.values() if accept and r is not None]
    if not accepted:
        return pd.DataFrame()
    t_min = min(r["date"] for _, r in accepted)
    t_max = max(r["date"] for _, r in accepted)

    # Build index into df_all for fast forward scan during exit
    all_bars = (df_all[(df_all["date"] >= t_min) &
                       (df_all["date"] <= t_max) &
                       (df_all["bar_time"] >= BLACKOUT_END) &
                       (df_all["bar_time"] < RTH_END)]
                .reset_index(drop=True))

    closes_all = all_bars["close"].values.astype(np.float64)
    highs_all  = all_bars["high"].values.astype(np.float64)
    lows_all   = all_bars["low"].values.astype(np.float64)
    pl_all     = all_bars["pl"].values   # precomputed from df_all (df_feat)
    dates_all  = all_bars["date"].values
    times_all  = all_bars["bar_time"].values
    # Use pandas Timestamps so they match qual_map keys (also pandas Timestamps)
    ts_all     = list(all_bars["ts"])

    trades   = []
    i        = 0
    n_all    = len(all_bars)

    while i < n_all:
        ts = ts_all[i]
        if ts in qual_map:
            accept, qrow = qual_map[ts]
            if accept:
                d     = int(qrow["direction"])
                entry = closes_all[i]
                today = dates_all[i]

                # Simulate live exit from bar i+1 onward
                stop_pt = entry * STOP_BPS / 10000.0
                exit_pnl_bps = None
                exit_idx = i

                for k in range(1, MAX_HOLD + 1):
                    j = i + k
                    if j >= n_all or dates_all[j] != today:
                        # End of session — cap at stop loss
                        raw = (closes_all[j - 1] - entry) * d / entry * 10000
                        exit_pnl_bps = max(raw, -STOP_BPS)
                        exit_idx = j - 1
                        break

                    if times_all[j] >= SESSION_END:
                        raw = (closes_all[j] - entry) * d / entry * 10000
                        exit_pnl_bps = max(raw, -STOP_BPS)
                        exit_idx = j
                        break

                    # Stop
                    if d == 1 and lows_all[j] <= entry - stop_pt:
                        exit_pnl_bps = -STOP_BPS
                        exit_idx = j
                        break
                    if d == -1 and highs_all[j] >= entry + stop_pt:
                        exit_pnl_bps = -STOP_BPS
                        exit_idx = j
                        break

                    # PL exit (after min hold)
                    cur_pl = pl_all[j]
                    if k >= MIN_HOLD and not np.isnan(cur_pl) and cur_pl <= EXIT_PL:
                        raw = (closes_all[j] - entry) * d / entry * 10000
                        exit_pnl_bps = max(raw, -STOP_BPS)
                        exit_idx = j
                        break

                    if k == MAX_HOLD:
                        raw = (closes_all[j] - entry) * d / entry * 10000
                        exit_pnl_bps = max(raw, -STOP_BPS)
                        exit_idx = j

                if exit_pnl_bps is None:
                    i += 1
                    continue

                trades.append({
                    "ts": ts, "date": today, "bar_time": times_all[i],
                    "direction": "LONG" if d == 1 else "SHORT",
                    "entry": entry,
                    "pnl_bps": exit_pnl_bps,
                    "pnl_pts": exit_pnl_bps * entry / 10000.0,
                    "prob": qrow.get("prob", 1.0),
                })
                # Re-entry: resume scanning from bar after exit
                i = exit_idx + 1
                continue

        i += 1

    return pd.DataFrame(trades)


def simulate_ml(test: pd.DataFrame, df_all: pd.DataFrame,
                clf, fcols: list[str],
                prob_thr: float = PROB_THRESHOLD) -> pd.DataFrame:
    """ML-filtered simulation: only enter when proba >= prob_thr."""
    if test.empty:
        return pd.DataFrame()
    test_r   = test.reset_index(drop=True)
    feat_arr = test_r[fcols].fillna(0.0).values
    proba    = clf.predict_proba(feat_arr)[:, 1]
    qual_map = {}
    for i, row in test_r.iterrows():
        p = proba[i]
        if p >= prob_thr:
            r2 = dict(row)
            r2["prob"] = p
            qual_map[row["ts"]] = (True, r2)
        else:
            qual_map[row["ts"]] = (False, None)
    return _run_simulation(test, df_all, qual_map)


def simulate_baseline(test: pd.DataFrame,
                      df_all: pd.DataFrame) -> pd.DataFrame:
    """Baseline: enter on ALL qualifying bars (no ML filter)."""
    if test.empty:
        return pd.DataFrame()
    qual_map = {}
    for _, row in test.iterrows():
        r2 = dict(row)
        r2["prob"] = 1.0
        qual_map[row["ts"]] = (True, r2)
    return _run_simulation(test, df_all, qual_map)


# ── Reporting ─────────────────────────────────────────────────────────────────

TOD_WINDOWS = [
    ("9:40–10:00",  dtime(9,  40), dtime(10,  0)),
    ("10:00–10:30", dtime(10,  0), dtime(10, 30)),
    ("10:30–11:00", dtime(10, 30), dtime(11,  0)),
    ("11:00–13:00", dtime(11,  0), dtime(13,  0)),
    ("13:00–15:55", dtime(13,  0), dtime(15, 55)),
]


def _trade_stats(trades: pd.DataFrame, test_days: int, point_value: float,
                 label: str) -> None:
    if trades.empty:
        print(f"  {label}: (no trades)"); return
    n   = len(trades)
    wr  = (trades["pnl_bps"] > 0).mean() * 100
    avg = trades["pnl_bps"].mean()
    pnl_pts = trades["pnl_pts"].sum()
    usd = pnl_pts * point_value
    ev  = avg   # avg P&L in bps per trade
    print(f"  {label}: n={n:5d}  {n/test_days:.2f}/day  "
          f"WR={wr:5.1f}%  EV={ev:+.2f}bp  "
          f"P&L={pnl_pts:+.1f}pt (~${usd:+,.0f})")


def detail_report(label: str,
                  trades_ml: pd.DataFrame, trades_base: pd.DataFrame,
                  auc: float, base_wr: float,
                  train_days: int, test_days: int, point_value: float,
                  fi: list | None = None,
                  prob_thr: float = PROB_THRESHOLD):
    print(f"\n{'═'*80}")
    print(f"  {label}")
    print(f"{'═'*80}")
    print(f"  AUC={auc:.4f}  base_wr={base_wr:.1f}%  "
          f"prob≥{prob_thr}  train={train_days}d  test={test_days}d")

    _trade_stats(trades_base, test_days, point_value, "BASELINE (no ML)")
    _trade_stats(trades_ml,   test_days, point_value, f"ML      (p≥{prob_thr})")

    if not trades_ml.empty:
        print("\n  Time-of-day (ML):")
        for lbl, t0, t1 in TOD_WINDOWS:
            w = trades_ml[(trades_ml["bar_time"] >= t0) & (trades_ml["bar_time"] < t1)]
            if w.empty:
                print(f"    {lbl:<14} —"); continue
            nw  = len(w)
            wrw = (w["pnl_bps"] > 0).mean() * 100
            ev  = w["pnl_bps"].mean()
            pts = w["pnl_pts"].sum()
            print(f"    {lbl:<14} n={nw:5d}  WR={wrw:5.1f}%  "
                  f"EV={ev:+.2f}bp  pnl={pts:+.1f}pt (~${pts*point_value:+,.0f})")

        print("\n  Monthly P&L (ML, in pts):")
        t2 = trades_ml.copy()
        t2["month"] = pd.to_datetime(t2["date"]).dt.to_period("M")
        monthly = t2.groupby("month").agg(
            n=("pnl_pts", "count"),
            wr=("pnl_bps", lambda x: (x > 0).mean() * 100),
            pnl=("pnl_pts", "sum"),
        ).reset_index()
        for _, row in monthly.iterrows():
            bar = "█" * max(0, int(row["pnl"] / 3))
            neg = "░" * max(0, int(-row["pnl"] / 3))
            print(f"    {str(row['month']):<8}  n={int(row['n']):4d}  "
                  f"WR={row['wr']:5.1f}%  pnl={row['pnl']:+7.1f}pt  {bar}{neg}")

        # Prob threshold sweep
        if "prob" in trades_ml.columns:
            print(f"\n  Prob-threshold sweep:")
            for thr in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]:
                sub = trades_ml[trades_ml["prob"] >= thr]
                if len(sub) < 5:
                    continue
                wrr = (sub["pnl_bps"] > 0).mean() * 100
                ev  = sub["pnl_bps"].mean()
                pts = sub["pnl_pts"].sum()
                usd = pts * point_value
                print(f"    p≥{thr:.2f}: n={len(sub):5d}  "
                      f"WR={wrr:5.1f}%  EV={ev:+.2f}bp  ~${usd:+,.0f}")

    if fi:
        print("\n  Feature importances:")
        for fname, fv in fi[:14]:
            bar = "█" * int(fv * 400)
            print(f"    {fname:<22} {fv*100:5.1f}%  {bar}")


# ── Vol-regime gate analysis (no ML) ─────────────────────────────────────────

VOL_GATE_THRESHOLDS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80]
VOL_GATE_PL_THR    = 0.70
VOL_GATE_MOVE_THR  = 12

# Time windows for combined sweep
VOL_GATE_TIME_WINDOWS = [
    ("all",        dtime(9,  40), dtime(15, 55)),
    ("9:40-10:00", dtime(9,  40), dtime(10,  0)),
    ("10:00-10:15",dtime(10,  0), dtime(10, 15)),
    ("10:15-10:30",dtime(10, 15), dtime(10, 30)),
    ("10:00-10:30",dtime(10,  0), dtime(10, 30)),
    ("10:30-11:00",dtime(10, 30), dtime(11,  0)),
    ("11:00-15:55",dtime(11,  0), dtime(15, 55)),
]


def _build_qual_map(df_feat, qual_mask, direction):
    """Build qual_map dict for _run_simulation from a boolean mask."""
    closes = df_feat["close"].values.astype(np.float64)
    dates  = df_feat["date"].values
    qual_map = {}
    for i in np.where(qual_mask)[0]:
        ts  = df_feat["ts"].iloc[i]
        row = {"ts": ts, "date": dates[i], "direction": int(direction[i]),
               "entry": closes[i], "prob": 1.0}
        qual_map[ts] = (True, row)
    return qual_map


def _sim_stats(trades, point_value):
    if trades.empty:
        return None
    n_days = len(pd.to_datetime(trades["date"]).dt.date.unique())
    n_t    = len(trades)
    wr     = (trades["pnl_bps"] > 0).mean() * 100
    ev     = trades["pnl_bps"].mean()
    pts    = trades["pnl_pts"].sum()
    usd    = pts * point_value
    mpnl   = (trades.assign(m=pd.to_datetime(trades["date"]).dt.to_period("M"))
              .groupby("m")["pnl_pts"].sum())
    n_mth  = mpnl.nunique()
    conc   = mpnl.abs().max() / (trades["pnl_pts"].abs().sum() + 1e-9)
    per_day = n_t / n_days if n_days else 0
    return dict(n=n_t, per_day=per_day, wr=wr, ev=ev, pts=pts,
                usd=usd, n_mth=n_mth, conc=conc, trades=trades)


def _run_vol_gate(df_feat: pd.DataFrame, df_all: pd.DataFrame,
                  point_value: float):
    """
    Phase 1: vol_regime sweep on test period (same as ML mode).
    Phase 2: 2D sweep vol_min × time_window on test period.
    Phase 3: full-dataset monthly detail for the best combined filter.
    """
    net_ret = df_feat["net_ret"].values
    pl_arr  = df_feat["pl"].values
    vol_r   = df_feat["vol_regime"].values
    times   = df_feat["bar_time"].values
    dates   = df_feat["date"].values

    ret_cols  = [f"ret_{i}" for i in range(1, WINDOW + 1)]
    has_feats = df_feat[ret_cols].notna().all(axis=1).values
    time_ok   = np.array([BLACKOUT_END <= t < SESSION_END for t in times])
    direction = np.where(net_ret > 0, 1, -1).astype(int)

    # Base qualifying gate (no vol or time filter yet)
    qual_base = (
        time_ok & has_feats &
        (pl_arr >= VOL_GATE_PL_THR) &
        (df_feat["move_bps"].values >= VOL_GATE_MOVE_THR) &
        ~np.isnan(pl_arr)
    )

    # Test-period mask (last 25%)
    all_dates = sorted(df_feat["date"].unique())
    cutoff    = all_dates[int(len(all_dates) * TRAIN_FRAC)]
    test_mask = dates >= cutoff

    def vol_mask(vmin):
        if vmin == 0.0:
            return np.ones(len(df_feat), dtype=bool)
        return (~np.isnan(vol_r)) & (vol_r >= vmin)

    def time_mask(t0, t1):
        return np.array([t0 <= t < t1 for t in times])

    def run(qual):
        qm = _build_qual_map(df_feat, qual, direction)
        if not qm:
            return None
        t = _run_simulation(df_feat, df_all, qm)
        return _sim_stats(t, point_value)

    # ── Phase 1: vol sweep on test period ──────────────────────────────────
    print(f"\n{'═'*95}")
    print(f"  PHASE 1 — vol_regime sweep  ·  PL≥{VOL_GATE_PL_THR}  "
          f"mv≥{VOL_GATE_MOVE_THR}bp  ·  test period ({cutoff} →)  ·  all hours")
    print(f"{'═'*95}")
    print(f"  {'vol_min':>8}  {'n':>6}  {'per_day':>7}  {'WR%':>6}  "
          f"{'EV(bp)':>8}  {'P&L pts':>9}  {'P&L $':>10}  {'months':>7}")
    print(f"  {'─'*75}")
    vol1_rows = []
    for vmin in VOL_GATE_THRESHOLDS:
        q = qual_base & test_mask & vol_mask(vmin)
        s = run(q)
        if s is None:
            print(f"  {vmin:>8.2f}  —"); continue
        print(f"  {vmin:>8.2f}  {s['n']:>6}  {s['per_day']:>7.2f}  "
              f"{s['wr']:>6.1f}%  {s['ev']:>+8.2f}  "
              f"{s['pts']:>+9.1f}  {s['usd']:>+10,.0f}  {s['n_mth']:>7}")
        vol1_rows.append((vmin, s))

    # ── Phase 2: 2D sweep (vol_min × time_window) on test period ───────────
    print(f"\n{'═'*130}")
    print(f"  PHASE 2 — combined sweep  ·  vol_min × time_window  ·  EV (bp)  [n trades]")
    print(f"{'═'*130}")

    win_labels = [lbl for lbl, _, _ in VOL_GATE_TIME_WINDOWS]
    hdr = f"  {'vol_min':>8}" + "".join(f"  {lbl:>16}" for lbl in win_labels)
    print(hdr)
    print(f"  {'─'*8}" + "  " + "  ".join(["─"*16] * len(win_labels)))

    best_combo = None   # (vmin, win_label, t0, t1, stats)
    best_ev    = -999.0
    combo_grid = {}

    for vmin in VOL_GATE_THRESHOLDS:
        row_str = f"  {vmin:>8.2f}"
        for lbl, t0, t1 in VOL_GATE_TIME_WINDOWS:
            q  = qual_base & test_mask & vol_mask(vmin) & time_mask(t0, t1)
            s  = run(q)
            combo_grid[(vmin, lbl)] = s
            if s is None or s["n"] < 10:
                row_str += f"  {'—':>16}"
                continue
            cell = f"{s['ev']:+.2f}bp [{s['n']}]"
            row_str += f"  {cell:>16}"
            if s["ev"] > best_ev and s["n"] >= 10:
                best_ev    = s["ev"]
                best_combo = (vmin, lbl, t0, t1, s)
        print(row_str)

    # ── Phase 3: full-dataset monthly detail for best combo ─────────────────
    # ── Phase 3: full-dataset monthly detail for robust candidates ─────────
    # Instead of auto-picking the overfit best cell, examine the 10:00-10:30
    # window (consistently positive in test period with n≥37 trades at any vol level)
    print(f"\n{'═'*95}")
    print(f"  PHASE 3 — FULL DATASET  ·  10:00–10:30 window  (robust candidate)")
    print(f"{'═'*95}")

    t0_r, t1_r = dtime(10, 0), dtime(10, 30)

    for vmin_r in [0.0, 0.50, 0.70]:
        q_full = qual_base & vol_mask(vmin_r) & time_mask(t0_r, t1_r)
        s_full = run(q_full)
        lbl_r  = f"vol≥{vmin_r:.2f}  10:00–10:30"
        if s_full is None:
            print(f"\n  {lbl_r}: no trades"); continue

        t = s_full["trades"]
        n_days_full = len(pd.to_datetime(t["date"]).dt.date.unique())
        print(f"\n  {lbl_r}")
        print(f"  Full: n={s_full['n']}  {s_full['n']/n_days_full:.2f}/day  "
              f"WR={s_full['wr']:.1f}%  EV={s_full['ev']:+.2f}bp  "
              f"P&L={s_full['pts']:+.1f}pt (~${s_full['usd']:+,.0f})  "
              f"{s_full['n_mth']}mo  conc={s_full['conc']:.2f}")
        print(f"  Test: n={combo_grid.get((vmin_r, '10:00-10:30'), {}).get('n','?')}  "
              f"EV={combo_grid.get((vmin_r, '10:00-10:30'), {}).get('ev', float('nan')):+.2f}bp")

        t2 = t.copy()
        t2["month"] = pd.to_datetime(t2["date"]).dt.to_period("M")
        monthly = (t2.groupby("month")
                   .agg(n=("pnl_pts","count"),
                        wr=("pnl_bps", lambda x: (x > 0).mean() * 100),
                        pnl=("pnl_pts","sum"))
                   .reset_index())
        for _, row in monthly.iterrows():
            bar = "█" * max(0, int(row["pnl"] / 2))
            neg = "░" * max(0, int(-row["pnl"] / 2))
            usd_m = row["pnl"] * point_value
            print(f"    {str(row['month']):<8}  n={int(row['n']):4d}  "
                  f"WR={row['wr']:5.1f}%  pnl={row['pnl']:>+7.1f}pt  "
                  f"(~${usd_m:>+7,.0f})  {bar}{neg}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym  = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "sweep"

    cfg = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    tick_size   = cfg["tick_size"]
    point_value = cfg["point_value"]

    print(f"\n=== PL_MOM ML Filter  ·  5s/30s  ·  {sym}  ·  {mode.upper()} ===\n")
    print("Loading 5s bars …", flush=True)
    df = load_5s(sym)

    print("Building rolling features …", flush=True)
    df_feat = _build_rolling_features(df)

    # For the simulation we need pl_arr on the full dataset
    # df_feat is indexed on the RTH slice; df is the same (we pass df_feat as df_all)
    df_all = df_feat   # same frame, simulation reads pl from it

    print("  done.", flush=True)

    if mode == "vol_gate":
        _run_vol_gate(df_feat, df_all, point_value)
        return

    if mode == "single":
        sc = SINGLE_CFG
        configs = [(sc["pl_thr"], sc["move_thr"])]
    else:
        configs = list(product(PL_THRS, MOVE_THRS))
        print(f"\nSweep: {len(configs)} qualifying-gate configs  "
              f"exit: pl≤{EXIT_PL} | stop={STOP_BPS}bp | hold≤{MAX_HOLD*5}s\n")

    summary_rows = []

    for ci, (pt, mt) in enumerate(configs):
        if mode != "single":
            print(f"  [{ci+1}/{len(configs)}] pl≥{pt}  mv≥{mt}bp …", flush=True)

        df_dir = build_dataset(df_feat, pt, mt)
        if len(df_dir) < 200:
            print(f"    → too few samples ({len(df_dir)}), skip"); continue

        df_dir = df_dir.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        if len(df_dir) < 200:
            continue

        dates  = sorted(df_dir["date"].unique())
        cutoff = dates[int(len(dates) * TRAIN_FRAC)]
        train  = df_dir[df_dir["date"] <  cutoff]
        test   = df_dir[df_dir["date"] >= cutoff]
        test_days  = test["date"].nunique()
        train_days = train["date"].nunique()

        if len(train) < 100 or len(test) < 50 or test_days < 20:
            print(f"    → not enough train/test days, skip"); continue

        n_trees = 300 if mode == "single" else 100
        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=6, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[FEATURE_COLS].fillna(0).values,
                train["target"].astype(int).values)

        proba   = clf.predict_proba(test[FEATURE_COLS].fillna(0).values)[:, 1]
        auc     = roc_auc_score(test["target"].astype(int).values, proba)
        base_wr = test["target"].mean() * 100

        trades_ml   = simulate_ml(test, df_all, clf, FEATURE_COLS)
        trades_base = simulate_baseline(test, df_all)

        def _stats(trd):
            if trd.empty: return {}
            n   = len(trd)
            wr  = (trd["pnl_bps"] > 0).mean() * 100
            ev  = trd["pnl_bps"].mean()
            pts = trd["pnl_pts"].sum()
            usd = pts * point_value
            mth = pd.to_datetime(trd["date"]).dt.to_period("M").nunique()
            mpnl = (trd.assign(m=pd.to_datetime(trd["date"]).dt.to_period("M"))
                    .groupby("m")["pnl_pts"].sum())
            conc = mpnl.abs().max() / (trd["pnl_pts"].abs().sum() + 1e-9)
            return dict(n=n, per_day=n/test_days, wr=wr, ev=ev,
                        pts=pts, usd=usd, months=mth, conc=conc)

        sb = _stats(trades_base)
        sm = _stats(trades_ml)

        if mode == "single":
            fi = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                        key=lambda x: -x[1])
            label = (f"pl≥{pt}  mv≥{mt}bp  "
                     f"exit: pl≤{EXIT_PL}|stop={STOP_BPS}bp|hold≤{MAX_HOLD*5}s  "
                     f"AUC={auc:.4f}  base_wr={base_wr:.1f}%")
            detail_report(label, trades_ml, trades_base, auc, base_wr,
                          train_days, test_days, point_value, fi)
        else:
            summary_rows.append({
                "pl_thr": pt, "move_thr": mt,
                "auc": auc, "base_wr": base_wr,
                **{f"b_{k}": v for k, v in sb.items()},
                **{f"m_{k}": v for k, v in sm.items()},
                "_test_days": test_days, "_train_days": train_days,
                "_clf": clf, "_trades_ml": trades_ml, "_trades_base": trades_base,
                "_df_dir": df_dir,
            })

    if mode == "sweep" and summary_rows:
        df_sum = pd.DataFrame([{k: v for k, v in r.items()
                                 if not k.startswith("_")}
                                for r in summary_rows])

        def show(title, df_s, sort_col):
            print(f"\n{'═'*100}")
            print(f"  {title}")
            print(f"{'═'*100}")
            cols = ["pl_thr","move_thr","auc","base_wr",
                    "b_n","b_per_day","b_wr","b_ev","b_usd",
                    "m_n","m_per_day","m_wr","m_ev","m_usd","m_months","m_conc"]
            avail = [c for c in cols if c in df_s.columns]
            print(df_s[avail].sort_values(sort_col, ascending=False)
                  .to_string(index=False, float_format="{:.2f}".format))

        show(f"SWEEP — sorted by ML P&L  ·  {sym}", df_sum, "m_usd")

        # Detail report for top-3 by ML P&L, re-train with 300 trees
        top = sorted(summary_rows, key=lambda r: r.get("m_usd", 0), reverse=True)[:3]
        for r in top:
            pt2, mt2 = r["pl_thr"], r["move_thr"]
            df2 = build_dataset(df_feat, pt2, mt2)
            df2 = df2.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            dates2  = sorted(df2["date"].unique())
            cutoff2 = dates2[int(len(dates2) * TRAIN_FRAC)]
            tr2 = df2[df2["date"] < cutoff2]
            te2 = df2[df2["date"] >= cutoff2]
            clf2 = RandomForestClassifier(
                n_estimators=300, max_depth=6, min_samples_leaf=20,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            clf2.fit(tr2[FEATURE_COLS].fillna(0).values,
                     tr2["target"].astype(int).values)
            proba2  = clf2.predict_proba(te2[FEATURE_COLS].fillna(0).values)[:, 1]
            auc2    = roc_auc_score(te2["target"].astype(int).values, proba2)
            base_wr2 = te2["target"].mean() * 100
            fi2     = sorted(zip(FEATURE_COLS, clf2.feature_importances_),
                             key=lambda x: -x[1])
            tm2  = simulate_ml(te2, df_all, clf2, FEATURE_COLS)
            tb2  = simulate_baseline(te2, df_all)
            lbl  = (f"pl≥{pt2}  mv≥{mt2}bp  "
                    f"(base: {r.get('b_n',0)} trades  "
                    f"EV={r.get('b_ev',0):+.2f}bp  {r.get('b_months',0)}mo  |  "
                    f"ML: {r.get('m_n',0)} trades  "
                    f"EV={r.get('m_ev',0):+.2f}bp  ~${r.get('m_usd',0):+,.0f})")
            detail_report(lbl, tm2, tb2, auc2, base_wr2,
                          r["_train_days"], r["_test_days"], point_value, fi2)

    print()


if __name__ == "__main__":
    main()
