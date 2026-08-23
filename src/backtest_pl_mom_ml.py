"""
backtest_pl_mom_ml.py — ML version of PL_MOM on 1-min bars.

Current PL_MOM uses 5s bars, 30s window, hard rules (PL ≥ 0.70, move ≥ 12bp).
This version replaces the hard rules with a Random Forest on 1-min bars with a
10–20 min lookback, predicting whether price continues target_bps in the next
5 minutes before reversing by stop_bps.

Features (all sign-flipped by direction so the model learns "continuation"):
  ret_1m_1 … ret_1m_N   — individual 1-min log returns over lookback window
  net_ret_bps            — total return over window (move magnitude, always +)
  pl                     — price linearity = |sum(rets)| / sum(|rets|)  [0,1]
  realized_vol           — std of 1-min returns in window
  vol_accel              — volume in last half of window / first half  (>1 = accel)
  vol_ratio              — window avg volume / rolling 20-day avg volume
  ret_open_bps           — log(close / open_930) × 10000, sign-flipped
  vol_regime             — percentile rank of realized_vol vs 60-day history
  atr_ratio              — window range / rolling 20-day avg range
  tod_sin, tod_cos       — cyclic time-of-day encoding

Sweep parameters:
  lookback_bars : [10, 15, 20]    1-min bars = 10/15/20 min window
  move_thr_bps  : [5, 8, 12, 20]  qualifying move gate
  target_bps    : [8, 12, 16, 20] profit target
  stop_bps      : [5, 8, 10]      stop loss (< target enforced)
  horizon_bars  : 5               fixed = 5 minutes

Re-entry is always enabled.
Train/test: 75/25 walk-forward split by date.

Usage (from repo root):
    python src/backtest_pl_mom_ml.py           # MES
    python src/backtest_pl_mom_ml.py MNQ
    python src/backtest_pl_mom_ml.py MES single  # quick test of default config
"""

import math
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
from datetime import time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ──────────────────────────────────────────────────────────────────

ET           = ZoneInfo("America/New_York")
PRE_START    = dtime(8, 30)
RTH_START    = dtime(9, 30)
RTH_END      = dtime(16, 0)
BLACKOUT_END = dtime(9, 40)   # no entries in first 10 min
SESSION_END  = dtime(15, 55)  # stop taking new entries 5 min before close

SYMBOL_CONFIGS = {
    "MES": dict(hist_csv="mes_hist_1min.csv", tick_size=0.25,
                tick_val=1.25, point_value=5.0),
    "MNQ": dict(hist_csv="mnq_hist_1min.csv", tick_size=0.25,
                tick_val=0.50, point_value=2.0),
}

DB_PATH    = "data/bars.db"
TRAIN_FRAC = 0.75
HORIZON    = 5    # 1-min bars (5 minutes, fixed)

# Rolling windows for regime/ratio features
VOL_HIST_DAYS = 60
ATR_HIST_DAYS = 20
VOL_AVG_DAYS  = 20   # for vol_ratio baseline

# Sweep space
LOOKBACK_BARS = [10, 15, 20]
MOVE_THR_BPS  = [5, 8, 12, 20]
TARGET_BPS    = [8, 12, 16, 20]
STOP_BPS      = [5, 8, 10]

PROB_THRESHOLD = 0.58
VOL_REGIME_MIN = 0.20

# Single-config default
SINGLE_CFG = dict(lookback=15, move_thr=8, target=12, stop=8)

# Months to exclude from ALL analysis — remove extreme-regime outliers.
# April 2025 = tariff crash, dominated by extreme vol, not representative.
EXCLUDE_MONTHS: set[tuple[int,int]] = {(2025, 4)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def pts_from_bps(price: float, bps: float, tick_size: float) -> float:
    raw   = price * bps / 10000.0
    ticks = math.ceil(raw / tick_size)
    return round(ticks * tick_size, 4)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_1min(symbol: str) -> pd.DataFrame:
    cfg    = SYMBOL_CONFIGS[symbol]
    frames = []
    hist = Path(cfg["hist_csv"])
    if hist.exists():
        df = pd.read_csv(hist, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
        t = df["ts"].dt.time
        df = df[(t >= PRE_START) & (t < RTH_END)]
        frames.append(df)
    db = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql(
        f"SELECT ts, open, high, low, close, volume "
        f"FROM bars WHERE symbol='{symbol}' AND minutes=1 ORDER BY ts", db)
    db.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t = df_db["ts"].dt.time
    df_db = df_db[(t >= PRE_START) & (t < RTH_END)]
    frames.append(df_db)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True))
    df["date"]     = df["ts"].dt.date
    df["bar_time"] = df["ts"].dt.time
    if EXCLUDE_MONTHS:
        mask = df["date"].apply(lambda d: (d.year, d.month) not in EXCLUDE_MONTHS)
        df = df[mask].reset_index(drop=True)
        excl = ", ".join(f"{y}-{m:02d}" for y, m in sorted(EXCLUDE_MONTHS))
        print(f"  [excluded {excl}]", flush=True)
    # 9:30 open per day
    open_930 = (df[df["bar_time"] == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df = df.join(open_930, on="date")
    print(f"  {len(df):,} 1-min bars  "
          f"{df['date'].min()} → {df['date'].max()}", flush=True)
    return df


# ── Feature + label computation ───────────────────────────────────────────────

def _compute_daily_adx(df: pd.DataFrame, period: int = 14,
                       win_start: dtime = dtime(9, 0),
                       win_end:   dtime = dtime(10, 0)) -> dict:
    """
    Compute ADX(period) on the win_start–win_end 1-min window for each date.
    Returns dict {date: adx_value} using standard Wilder-smoothed ADX (0–100).
    df must include pre-market bars (bar_time >= 8:30).
    """
    result = {}
    for d, grp in df.groupby("date"):
        w = grp[(grp["bar_time"] >= win_start) & (grp["bar_time"] < win_end)]
        h = w["high"].values.astype(float)
        l = w["low"].values.astype(float)
        c = w["close"].values.astype(float)
        n = len(c)
        if n < period * 2 + 1:
            result[d] = np.nan
            continue
        tr  = np.maximum(h[1:] - l[1:],
              np.maximum(np.abs(h[1:] - c[:-1]),
                         np.abs(l[1:] - c[:-1])))
        up   = h[1:] - h[:-1]
        dn   = l[:-1] - l[1:]
        pdm  = np.where((up > dn) & (up > 0), up, 0.0)
        ndm  = np.where((dn > up) & (dn > 0), dn, 0.0)
        # Wilder smooth: init = sum of first `period`, then rolling
        def _w(arr):
            s = np.empty(len(arr))
            s[0] = arr[:period].sum()
            for i in range(1, len(arr)):
                s[i] = s[i-1] - s[i-1] / period + arr[i]
            return s
        satr = _w(tr); spdm = _w(pdm); sndm = _w(ndm)
        pdi  = 100.0 * spdm / np.maximum(satr, 1e-12)
        ndi  = 100.0 * sndm / np.maximum(satr, 1e-12)
        dx   = 100.0 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-12)
        # ADX: Wilder-smooth DX, init as mean (keeps 0–100 range)
        adx  = np.empty(len(dx))
        adx[period - 1] = dx[:period].mean()
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        result[d] = float(adx[-1])
    return result


def _build_rolling_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Precompute all rolling features for a given lookback on the RTH 1-min series.
    Called once per lookback value; result is reused across (move_thr, tg, st) configs.
    """
    df_rth = df[(df["bar_time"] >= RTH_START) &
                (df["bar_time"] < RTH_END)].copy().reset_index(drop=True)

    # 1-min log returns (NaN at day boundaries)
    lr = np.log(df_rth["close"] / df_rth["close"].shift(1))
    lr[df_rth["date"] != df_rth["date"].shift(1)] = np.nan
    abs_lr = lr.abs()

    roll = lr.rolling(lookback, min_periods=lookback)
    roll_abs = abs_lr.rolling(lookback, min_periods=lookback)

    net_ret  = roll.sum()
    rvol     = roll.std(ddof=1)
    abs_sum  = roll_abs.sum()
    pl       = net_ret.abs() / abs_sum.replace(0, np.nan)

    # Individual lagged returns
    for lag in range(1, lookback + 1):
        df_rth[f"ret_{lag}"] = lr.shift(lag - 1)   # ret_1 = most recent

    df_rth["net_ret"]      = net_ret
    df_rth["realized_vol"] = rvol
    df_rth["pl"]           = pl

    # Volume features
    if "volume" in df_rth.columns:
        vol_s = df_rth["volume"]
        half  = max(1, lookback // 2)
        vol_first = vol_s.rolling(half, min_periods=half).mean()
        vol_last  = vol_s.shift(0).rolling(half, min_periods=half).mean()  # most recent half
        # vol_accel: last half vs first half of window
        vol_accel = vol_last / (vol_first.shift(half) + 1e-9)
        df_rth["vol_accel"] = vol_accel

        # vol_ratio: window avg vs daily avg
        win_vol_avg = vol_s.rolling(lookback, min_periods=lookback).mean()
        day_vol_avg = df_rth.groupby("date")["volume"].transform("mean")
        df_rth["vol_ratio"] = win_vol_avg / (day_vol_avg + 1e-9)
    else:
        df_rth["vol_accel"] = np.nan
        df_rth["vol_ratio"] = np.nan

    # Return since open
    df_rth["ret_open_bps"] = (np.log(df_rth["close"] / df_rth["open_930"])
                               * 10000)

    # Daily rolling baselines for vol_regime and atr_ratio
    day_vol   = df_rth.groupby("date")["realized_vol"].mean()
    day_range = (df_rth["high"] - df_rth["low"]).groupby(df_rth["date"]).mean()

    days = sorted(day_vol.index)
    vol_reg_map   = {}
    atr_ratio_map = {}
    hist_vol   = []
    hist_range = []
    for d in days:
        pv = [v for v in hist_vol[-VOL_HIST_DAYS:]   if not np.isnan(v)]
        pr = [v for v in hist_range[-ATR_HIST_DAYS:] if not np.isnan(v)]
        v  = day_vol.get(d, np.nan)
        r  = day_range.get(d, np.nan)
        vol_reg_map[d]   = (float(np.mean(np.array(pv) <= v))
                             if len(pv) >= 10 else np.nan)
        atr_ratio_map[d] = (v / float(np.mean(pr))
                             if len(pr) >= 5 and float(np.mean(pr)) > 0
                             else np.nan)
        hist_vol.append(v)
        hist_range.append(r)

    df_rth["vol_regime"] = df_rth["date"].map(vol_reg_map)
    df_rth["atr_ratio"]  = df_rth["date"].map(atr_ratio_map)

    # Time-of-day
    frac = df_rth["bar_time"].apply(_tod_frac)
    df_rth["tod_sin"] = np.sin(2 * np.pi * frac)
    df_rth["tod_cos"] = np.cos(2 * np.pi * frac)

    # ADX(14) on 9:00–10:00 1-min bars — daily opening regime indicator
    # Computed once per day; strong predictor of PL_MOM daily success
    adx_map = _compute_daily_adx(df, period=14,
                                  win_start=dtime(9, 0), win_end=dtime(10, 0))
    df_rth["adx_open"] = df_rth["date"].map(adx_map)

    return df_rth


def build_dataset(df: pd.DataFrame, lookback: int, move_thr_bps: float,
                  target_bps: float, stop_bps: float, tick_size: float,
                  df_feat: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build ML dataset from 1-min bars.

    For each RTH bar i (after BLACKOUT_END):
      1. Check qualifying move gate: |close[i] - close[i-lookback]| >= move_thr_bps
      2. Look up precomputed features from df_feat (built by _build_rolling_features)
      3. Label: 1 if +target_bps hit before -stop_bps in next HORIZON bars, 0 otherwise
         Inconclusive (neither hit) → skip

    Returns a DataFrame ready for RF training.
    df_feat: output of _build_rolling_features(); pass in to avoid recomputing per config.
    """
    df_rth = df_feat if df_feat is not None else _build_rolling_features(df, lookback)

    closes = df_rth["close"].values.astype(np.float64)
    highs  = df_rth["high"].values.astype(np.float64)
    lows   = df_rth["low"].values.astype(np.float64)
    dates  = df_rth["date"].values
    times  = df_rth["bar_time"].values
    n      = len(df_rth)

    # Qualifying move gate (vectorized)
    move     = np.zeros(n)
    thr      = closes * move_thr_bps / 10000.0
    lb_close = np.full(n, np.nan)
    lb_close[lookback:] = closes[:n - lookback]
    # Zero out cross-day
    same_day_lb = np.zeros(n, dtype=bool)
    same_day_lb[lookback:] = dates[lookback:] == dates[:n - lookback]
    move = np.where(same_day_lb, closes - lb_close, 0.0)
    qual = (np.abs(move) >= thr) & same_day_lb

    # Time gate
    time_ok = np.array([BLACKOUT_END <= t < SESSION_END for t in times])
    qual    = qual & time_ok

    # Also require all individual ret features to be non-NaN
    ret_cols = [f"ret_{i}" for i in range(1, lookback + 1)]
    has_feats = df_rth[ret_cols].notna().all(axis=1).values
    qual = qual & has_feats

    # Vectorized label computation
    tgt_pts   = np.ceil(closes * target_bps / 10000.0 / tick_size) * tick_size
    stp_pts   = np.ceil(closes * stop_bps   / 10000.0 / tick_size) * tick_size
    labels    = np.full(n, -1, dtype=np.int8)
    direction = np.where(move > 0, 1, -1).astype(np.int8)

    for k in range(1, HORIZON + 1):
        i_max = n - k
        if i_max <= 0:
            break
        same_day  = dates[:i_max] == dates[k:n]
        undecided = qual[:i_max] & (labels[:i_max] == -1) & same_day
        h  = highs[k:n][:i_max]
        lv = lows[k:n][:i_max]

        long_u = undecided & (direction[:i_max] == 1)
        tgt_l  = long_u & (h >= closes[:i_max] + tgt_pts[:i_max])
        stp_l  = long_u & (lv <= closes[:i_max] - stp_pts[:i_max])
        labels[:i_max][tgt_l & ~stp_l] = 1
        labels[:i_max][stp_l & ~tgt_l] = 0
        labels[:i_max][tgt_l & stp_l]  = 0

        shrt_u = undecided & (direction[:i_max] == -1)
        tgt_s  = shrt_u & (lv <= closes[:i_max] - tgt_pts[:i_max])
        stp_s  = shrt_u & (h >= closes[:i_max] + stp_pts[:i_max])
        labels[:i_max][tgt_s & ~stp_s] = 1
        labels[:i_max][stp_s & ~tgt_s] = 0
        labels[:i_max][tgt_s & stp_s]  = 0

    keep = qual & (labels >= 0)
    if not keep.any():
        return pd.DataFrame()

    df_k = df_rth[keep].copy().reset_index(drop=True)
    dir_k = direction[keep]

    # Sign-flip directional features
    for c in ret_cols + ["net_ret", "ret_open_bps"]:
        if c in df_k.columns:
            df_k[c] = df_k[c] * dir_k

    df_k["net_ret_bps"] = df_k["net_ret"].abs() * 10000
    df_k["direction"]   = dir_k
    df_k["target"]      = labels[keep].astype(int)
    df_k["entry"]       = df_k["close"]

    return df_k.reset_index(drop=True)


def _tod_frac(t: dtime) -> float:
    mins  = t.hour * 60 + t.minute
    start = 9 * 60 + 40
    end   = 15 * 60 + 55
    return max(0.0, min(1.0, (mins - start) / max(end - start, 1)))


def get_feature_cols(lookback: int) -> list[str]:
    ret_cols = [f"ret_{i}" for i in range(1, lookback + 1)]
    return ret_cols + [
        "net_ret_bps", "pl", "realized_vol",
        "vol_accel", "vol_ratio", "ret_open_bps",
        "vol_regime", "atr_ratio",
        "tod_sin", "tod_cos",
        "adx_open",   # ADX(14) on 9:00-10:00 1-min bars — opening regime
    ]


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(test: pd.DataFrame, df_all: pd.DataFrame,
             clf, fcols: list[str],
             target_bps: float, stop_bps: float,
             tick_size: float) -> pd.DataFrame:
    """
    Re-entry simulation. Walks ALL RTH 1-min bars in test period; checks TP/SL
    on every bar (not just qualifying bars), allows re-entry after each close.
    """
    # ts→(high,low,date,bar_time) from full 1-min series
    df_lu = (df_all[(df_all["bar_time"] >= BLACKOUT_END) &
                    (df_all["bar_time"] < RTH_END)]
             .set_index("ts")[["high","low","date","bar_time"]])

    if test.empty:
        return pd.DataFrame()

    # Probability map by ts for qualifying bars
    test_r   = test.reset_index(drop=True)
    feat_arr = test_r[fcols].fillna(0.0).values
    proba    = clf.predict_proba(feat_arr)[:, 1]
    qual_map = {row["ts"]: (proba[i], row)
                for i, row in test_r.iterrows()}

    t_min = test["date"].min()
    t_max = test["date"].max()
    all_bars = (df_all[(df_all["date"] >= t_min) &
                       (df_all["date"] <= t_max) &
                       (df_all["bar_time"] >= BLACKOUT_END) &
                       (df_all["bar_time"] < RTH_END)]
                .reset_index(drop=True))

    trades   = []
    in_trade = False

    for _, bar in all_bars.iterrows():
        bar_t = bar["bar_time"]
        today = bar["date"]

        if in_trade:
            t = trades[-1]
            if today != t["date"] or bar_t >= SESSION_END:
                t.update(outcome="EXPIRED", pnl_pts=0.0)
                in_trade = False
            else:
                h, lv = bar["high"], bar["low"]
                d, entry, tgt, stp = t["dir_int"], t["entry"], t["tgt"], t["stp"]
                closed = False
                if d == 1:
                    if h >= entry + tgt:    t.update(outcome="TARGET",  pnl_pts=tgt);  closed=True
                    elif lv <= entry - stp: t.update(outcome="STOPPED", pnl_pts=-stp); closed=True
                else:
                    if lv <= entry - tgt:   t.update(outcome="TARGET",  pnl_pts=tgt);  closed=True
                    elif h >= entry + stp:  t.update(outcome="STOPPED", pnl_pts=-stp); closed=True
                if closed:
                    in_trade = False
                else:
                    continue

        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue
        ts = bar["ts"]
        if ts not in qual_map:
            continue
        p, qrow = qual_map[ts]
        if p < PROB_THRESHOLD or qrow.get("vol_regime", 0) < VOL_REGIME_MIN:
            continue

        d     = int(qrow["direction"])
        entry = float(qrow["entry"])
        tgt   = pts_from_bps(entry, target_bps, tick_size)
        stp   = pts_from_bps(entry, stop_bps,   tick_size)

        trades.append({
            "ts": ts, "date": today, "bar_time": bar_t,
            "direction": "LONG" if d == 1 else "SHORT", "dir_int": d,
            "entry": entry, "tgt": tgt, "stp": stp,
            "prob": p, "vol_regime": qrow.get("vol_regime", 0),
            "outcome": None, "pnl_pts": None,
        })
        in_trade = True

    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).dropna(subset=["outcome"]).reset_index(drop=True)


# ── Reporting ─────────────────────────────────────────────────────────────────

WINDOWS = [
    ("9:40–11:00",  dtime(9,  40), dtime(11,  0)),
    ("11:00–13:00", dtime(11,  0), dtime(13,  0)),
    ("13:00–15:55", dtime(13,  0), dtime(15, 55)),
]


def detail_report(label: str, trades: pd.DataFrame, auc: float,
                  train_days: int, test_days: int, point_value: float,
                  fi: list | None = None):
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
        print(f"    {lbl:<14} n={nw:4d}  WR={wrw:5.1f}%  "
              f"pnl={pw:+.1f}pt (~${pw*point_value:+,.0f})")

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

    if fi:
        print("\n  Feature importances:")
        for fname, fv in fi[:12]:
            bar = "█" * int(fv * 400)
            print(f"    {fname:<22} {fv*100:5.1f}%  {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym  = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "sweep"

    cfg = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    tick_size   = cfg["tick_size"]
    point_value = cfg["point_value"]

    print(f"\n=== PL_MOM ML  ·  1-min bars  ·  {sym}  ·  {mode.upper()} ===\n")
    print("Loading 1-min bars …", flush=True)
    df = load_1min(sym)

    if mode == "single":
        sc = SINGLE_CFG
        configs = [(sc["lookback"], sc["move_thr"], sc["target"], sc["stop"])]
    else:
        configs = [
            (lb, mt, tg, st)
            for lb, mt, tg, st in product(LOOKBACK_BARS, MOVE_THR_BPS,
                                           TARGET_BPS, STOP_BPS)
            if st < tg
        ]
        n_total = len(configs)
        print(f"\nSweep: {n_total} configs "
              f"({len(LOOKBACK_BARS)}lb × {len(MOVE_THR_BPS)}mv × "
              f"{len(TARGET_BPS)}tg × {len(STOP_BPS)}st, stp<tgt)  "
              f"horizon=5min\n")

    summary_rows = []
    feat_cache: dict[int, pd.DataFrame] = {}   # lb → _build_rolling_features result

    for ci, (lb, mt, tg, st) in enumerate(configs):
        if mode != "single" and ci % 20 == 0:
            print(f"  {ci}/{len(configs)} …", flush=True)

        if lb not in feat_cache:
            print(f"  Building rolling features  lb={lb} …", flush=True)
            feat_cache[lb] = _build_rolling_features(df, lb)

        df_dir = build_dataset(df, lb, mt, tg, st, tick_size, df_feat=feat_cache[lb])
        if len(df_dir) < 200:
            continue

        fcols = get_feature_cols(lb)
        # Drop rows where any feature is NaN
        df_dir = df_dir.dropna(subset=fcols).reset_index(drop=True)
        if len(df_dir) < 200:
            continue

        dates  = sorted(df_dir["date"].unique())
        cutoff = dates[int(len(dates) * TRAIN_FRAC)]
        train  = df_dir[df_dir["date"] <  cutoff]
        test   = df_dir[df_dir["date"] >= cutoff]
        test_days  = test["date"].nunique()
        train_days = train["date"].nunique()

        if len(train) < 100 or len(test) < 50 or test_days < 50:
            continue

        n_trees = 300 if mode == "single" else 100
        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[fcols].fillna(0).values,
                train["target"].astype(int).values)

        proba = clf.predict_proba(test[fcols].fillna(0).values)[:, 1]
        auc   = roc_auc_score(test["target"].astype(int).values, proba)

        trades = simulate(test, df, clf, fcols, tg, st, tick_size)
        if trades.empty:
            continue

        n_t     = len(trades)
        wr_t    = (trades["outcome"] == "TARGET").mean() * 100
        pnl_t   = trades["pnl_pts"].sum()
        usd_t   = pnl_t * point_value
        avg_t   = trades["pnl_pts"].mean()
        n_months= pd.to_datetime(trades["date"]).dt.to_period("M").nunique()
        mpnl    = (trades.assign(month=pd.to_datetime(trades["date"]).dt.to_period("M"))
                   .groupby("month")["pnl_pts"].sum())
        conc    = mpnl.abs().max() / (trades["pnl_pts"].abs().sum() + 1e-9)

        if mode == "single":
            fi = sorted(zip(fcols, clf.feature_importances_),
                        key=lambda x: -x[1])
            label = (f"lb={lb}  mv={mt}bp  tg={tg}bp  st={st}bp  hz=5min"
                     f"  AUC={auc:.4f}  base={test['target'].mean()*100:.1f}%")
            detail_report(label, trades, auc, train_days, test_days,
                          point_value, fi)
        else:
            summary_rows.append({
                "lookback": lb, "move_thr": mt, "target": tg, "stop": st,
                "n": n_t, "per_day": n_t / test_days,
                "wr": wr_t, "avg_pnl": avg_t,
                "total_pnl": pnl_t, "usd": usd_t, "auc": auc,
                "months": n_months, "conc": conc,
                "_trades": trades, "_test_days": test_days,
                "_train_days": train_days, "_clf": clf,
            })

    if mode == "sweep" and summary_rows:
        df_sum = pd.DataFrame([{k: v for k, v in r.items()
                                 if not k.startswith("_")}
                                for r in summary_rows])

        print(f"\n{'═'*100}")
        print(f"  SWEEP RESULTS  ·  {sym}  ·  PL_MOM ML  (sorted by total_pnl)")
        print(f"{'═'*100}")
        cols = ["lookback","move_thr","target","stop","n","per_day",
                "wr","avg_pnl","total_pnl","usd","auc","months","conc"]
        print(df_sum[cols].sort_values("total_pnl", ascending=False)
              .head(30).to_string(index=False, float_format="{:.2f}".format))

        broad = df_sum[df_sum["months"] >= 6].sort_values("total_pnl", ascending=False)
        if not broad.empty:
            print(f"\n{'─'*100}")
            print(f"  BROAD CONFIGS (≥6 months active)")
            print(f"{'─'*100}")
            print(broad[cols].head(15).to_string(index=False, float_format="{:.2f}".format))

        # Detail + feature importances for top-3 broad
        broad_rows = sorted([r for r in summary_rows if r["months"] >= 6],
                            key=lambda r: r["total_pnl"], reverse=True)
        for r in (broad_rows or sorted(summary_rows,
                                        key=lambda r: r["total_pnl"],
                                        reverse=True))[:3]:
            # Re-build with 300 trees for accurate feature importance
            lb2, mt2, tg2, st2 = r["lookback"], r["move_thr"], r["target"], r["stop"]
            fcols2 = get_feature_cols(lb2)
            df2 = build_dataset(df, lb2, mt2, tg2, st2, tick_size,
                                df_feat=feat_cache.get(lb2))
            df2 = df2.dropna(subset=fcols2).reset_index(drop=True)
            dates2  = sorted(df2["date"].unique())
            cutoff2 = dates2[int(len(dates2) * TRAIN_FRAC)]
            tr2 = df2[df2["date"] < cutoff2]
            te2 = df2[df2["date"] >= cutoff2]
            clf2 = RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=15,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )
            clf2.fit(tr2[fcols2].fillna(0).values,
                     tr2["target"].astype(int).values)
            proba2 = clf2.predict_proba(te2[fcols2].fillna(0).values)[:, 1]
            auc2   = roc_auc_score(te2["target"].astype(int).values, proba2)
            fi2    = sorted(zip(fcols2, clf2.feature_importances_),
                            key=lambda x: -x[1])
            lbl = (f"lb={lb2}  mv={mt2}bp  tg={tg2}bp  st={st2}bp  hz=5min"
                   f"  ({r['n']} trades  {r['per_day']:.2f}/day  WR={r['wr']:.1f}%"
                   f"  {r['months']}mo)")
            detail_report(lbl, r["_trades"], auc2,
                          r["_train_days"], r["_test_days"],
                          point_value, fi2)

    print()


if __name__ == "__main__":
    main()
