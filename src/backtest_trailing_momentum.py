"""
backtest_trailing_momentum.py — Trailing-window momentum continuation model.

Core idea: instead of predicting direction from opening features, look at the
LAST N bars (60–120 min of 2-min bars) to ask: has price been trending cleanly?
If so, bet on continuation.

Key features:
  trail_ret_bps     — log return over trailing window (sign-flipped by direction)
  trail_vol         — realized vol of trailing log-returns
  trail_linearity   — |sum(rets)| / sum(|rets|) — regime_pl analog for the window
  trail_range_bps   — (high-low) of trailing window / current price * 10000
  atr_ratio         — trail_range vs rolling 20-day avg trail_range
  vol_regime        — percentile rank of trail_vol vs rolling 60-day history
  session_ret_bps   — log(close / open_930) * 10000, sign-flipped by direction
  tod_sin, tod_cos  — cyclic time-of-day encoding (9:40–15:45)
  r2m_1, r2m_2      — 2-min bar return at lag 1, 2 (short momentum)

Sweep parameters:
  trail_bars   : [20, 30, 45, 60]  (40/60/90/120 min of 2-min bars)
  move_thr_bps : [20, 30, 40, 60]  (qualifying trend magnitude gate)
  target_bps   : [15, 20, 30]      (profit target)
  stop_bps     : [8, 12, 20]       (stop loss, always < target)
  horizon_bars : [3, 5, 10, 20]    (2-min bars to resolve; 6/10/20/40 min)

Re-entry is always allowed (same bar the prior trade closes).
Train/test: 75/25 walk-forward split by date.

Usage (from repo root):
    python src/backtest_trailing_momentum.py           # MES
    python src/backtest_trailing_momentum.py MNQ
    python src/backtest_trailing_momentum.py MES single  # quick single-config run
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

ET        = ZoneInfo("America/New_York")
PRE_START = dtime(8, 30)
RTH_START = dtime(9, 30)
RTH_END   = dtime(16, 0)
BLACKOUT_END = dtime(9, 40)
SESSION_END  = dtime(15, 45)

SYMBOL_CONFIGS = {
    "MES": dict(hist_csv="mes_hist_1min.csv", tick_size=0.25, tick_val=1.25,
                point_value=5.0),
    "MNQ": dict(hist_csv="mnq_hist_1min.csv", tick_size=0.25, tick_val=0.50,
                point_value=2.0),
}

DB_PATH = "data/bars.db"
TRAIN_FRAC = 0.75

# Rolling history windows
VOL_HIST_DAYS   = 60   # days of history for vol_regime percentile
ATR_HIST_DAYS   = 20   # days for atr_ratio baseline

# Sweep space
TRAIL_BARS   = [20, 30, 45, 60]    # 2-min bars ≈ 40/60/90/120 min
MOVE_THR_BPS = [20, 30, 40, 60]    # qualifying gate: min trend magnitude
TARGET_BPS   = [15, 20, 30]        # profit target in bps
STOP_BPS     = [8, 12, 20]         # stop loss in bps (must be < target)
HORIZON_BARS = [3, 5, 10, 20]      # 2-min bars to hold (6/10/20/40 min)

# Single-config defaults for quick runs
SINGLE_CFG = dict(trail_bars=10, move_thr_bps=15, target_bps=15,
                  stop_bps=8, horizon_bars=5)

PROB_THRESHOLD = 0.58
VOL_REGIME_MIN = 0.20


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
        df_h = pd.read_csv(hist, parse_dates=["ts"])
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= PRE_START) & (t < RTH_END)]
        frames.append(df_h)
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
    print(f"  Loaded {len(df):,} 1-min bars  "
          f"{df['ts'].dt.date.min()} → {df['ts'].dt.date.max()}", flush=True)
    return df


def build_2min(df1: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 2-min and attach open_930 + ret_open."""
    df1 = df1.copy()
    df1["date"] = df1["ts"].dt.date

    # 9:30 open per day
    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df1 = df1.join(open_930, on="date")

    df2 = (df1.set_index("ts")[["open","high","low","close","volume"]]
             .resample("2min", label="right", closed="right")
             .agg(open=("open","first"), high=("high","max"),
                  low=("low","min"),   close=("close","last"),
                  volume=("volume","sum"))
             .dropna(subset=["close"])
             .reset_index())

    df2["date"]     = df2["ts"].dt.date
    df2["bar_time"] = df2["ts"].dt.time

    # Filter to RTH + blackout
    rth = (df2["bar_time"] >= RTH_START) & (df2["bar_time"] < RTH_END)
    df2 = df2[rth].reset_index(drop=True)

    # Attach open_930 and session return
    df2 = df2.join(open_930, on="date")
    df2["ret_open"] = np.log(df2["close"] / df2["open_930"])

    print(f"  {len(df2):,} 2-min RTH bars", flush=True)
    return df2


# ── Feature engineering ───────────────────────────────────────────────────────

def build_features(df2: pd.DataFrame, trail_bars: int) -> pd.DataFrame:
    """
    Add trailing-window features to df2 using vectorised pandas rolling ops.

    Cross-day contamination: lr is set to NaN at day boundaries so any rolling
    window that spans a day break automatically becomes NaN.
    """
    df2 = df2.copy().reset_index(drop=True)

    # 2-min log returns; NaN at day boundaries kills cross-day windows
    lr = np.log(df2["close"] / df2["close"].shift(1))
    lr[df2["date"] != df2["date"].shift(1)] = np.nan
    abs_lr = lr.abs()

    # Rolling over trail_bars bars (min_periods=trail_bars so partial windows → NaN)
    roll = lr.rolling(trail_bars, min_periods=trail_bars)
    roll_abs = abs_lr.rolling(trail_bars, min_periods=trail_bars)

    trail_ret_raw = roll.sum()                   # bps when * 10000
    trail_vol_raw = roll.std(ddof=1)
    abs_sum       = roll_abs.sum()
    trail_lin_raw = trail_ret_raw.abs() / abs_sum.replace(0, np.nan)

    # Range over trailing window
    roll_hi  = df2["high"].rolling(trail_bars, min_periods=trail_bars).max()
    roll_lo  = df2["low"].rolling(trail_bars,  min_periods=trail_bars).min()
    trail_rng_raw = (roll_hi - roll_lo) / df2["close"].replace(0, np.nan) * 10000

    df2["trail_ret_bps"]   = trail_ret_raw * 10000
    df2["trail_vol"]       = trail_vol_raw
    df2["trail_lin"]       = trail_lin_raw
    df2["trail_range_bps"] = trail_rng_raw

    # Short-term 2-min returns
    df2["r2m_1"] = lr * 10000
    df2["r2m_2"] = lr.shift(1) * 10000

    # ── Rolling daily summaries for vol_regime and atr_ratio ──────────────────
    # Compute per-day avg trail_vol and avg trail_range
    day_avg_vol   = df2.groupby("date")["trail_vol"].mean()
    day_avg_range = df2.groupby("date")["trail_range_bps"].mean()

    days = sorted(day_avg_vol.index)
    vol_regime_map   = {}
    atr_ratio_map    = {}
    hist_vol   = []
    hist_range = []
    for d in days:
        past_vols   = [v for v in hist_vol[-VOL_HIST_DAYS:]   if not np.isnan(v)]
        past_ranges = [r for r in hist_range[-ATR_HIST_DAYS:] if not np.isnan(r)]
        v = day_avg_vol.get(d, np.nan)
        r = day_avg_range.get(d, np.nan)
        vol_regime_map[d] = (float(np.mean(np.array(past_vols) <= v))
                              if len(past_vols) >= 10 else np.nan)
        atr_ratio_map[d]  = (v / np.mean(past_ranges)
                              if len(past_ranges) >= 5 and np.mean(past_ranges) > 0
                              else np.nan)
        hist_vol.append(v)
        hist_range.append(r)

    df2["vol_regime"] = df2["date"].map(vol_regime_map)
    df2["atr_ratio"]  = df2["date"].map(atr_ratio_map)

    # ── Time-of-day encoding ──────────────────────────────────────────────────
    # Map bar_time (9:40–15:45) to [0,1] then encode cyclically
    def time_frac(t: dtime) -> float:
        minutes = t.hour * 60 + t.minute
        start   = 9 * 60 + 40
        end     = 15 * 60 + 45
        return (minutes - start) / max(end - start, 1)

    frac = df2["bar_time"].apply(time_frac).clip(0, 1)
    df2["tod_sin"] = np.sin(2 * np.pi * frac)
    df2["tod_cos"] = np.cos(2 * np.pi * frac)

    return df2


def get_feature_cols() -> list[str]:
    return [
        "trail_ret_bps",   # sign-flipped continuation magnitude
        "trail_vol",
        "trail_lin",       # trend linearity (regime_pl analog)
        "trail_range_bps",
        "atr_ratio",
        "vol_regime",
        "session_ret_bps", # session-wide ret from open (sign-flipped)
        "r2m_1",           # short-term momentum
        "r2m_2",
        "tod_sin",
        "tod_cos",
    ]


# ── Dataset construction ──────────────────────────────────────────────────────

def precompute_eligible(df2: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all RTH bars with valid features (eligible universe).
    Returns a DataFrame keeping the original df2 positional index as 'df2_pos'.
    Call once per trail_bars; then use build_dataset() to slice by config params.
    """
    elig = df2[(df2["bar_time"] >= BLACKOUT_END) &
               (df2["bar_time"] < SESSION_END)].copy()
    elig = elig.dropna(subset=["trail_ret_bps", "trail_vol",
                                "vol_regime", "atr_ratio"])
    elig = elig.reset_index()          # 'index' = positional index in df2
    elig.rename(columns={"index": "df2_pos"}, inplace=True)
    return elig


def build_dataset(elig: pd.DataFrame, df2: pd.DataFrame,
                  move_thr_bps: float, target_bps: float, stop_bps: float,
                  horizon_bars: int, tick_size: float) -> pd.DataFrame:
    """
    Slice pre-computed eligible bars by move gate; label TP/SL using
    vectorized forward-scan on numpy arrays.
    """
    # Apply move gate
    gate = elig[elig["trail_ret_bps"].abs() >= move_thr_bps].copy()
    if gate.empty:
        return pd.DataFrame()

    direction = np.where(gate["trail_ret_bps"].values > 0, 1, -1)
    entries   = gate["close"].values
    dates_arr = gate["date"].values
    pos_arr   = gate["df2_pos"].values   # positional indices into df2

    # Precompute tgt_pts and stp_pts per bar (vectorized)
    raw_tgt  = entries * target_bps / 10000.0
    raw_stp  = entries * stop_bps   / 10000.0
    tgt_pts  = np.ceil(raw_tgt / tick_size) * tick_size
    stp_pts  = np.ceil(raw_stp / tick_size) * tick_size

    # Arrays from df2 for forward scan
    df2_high  = df2["high"].values
    df2_low   = df2["low"].values
    df2_dates = df2["date"].values
    n_df2     = len(df2)

    outcomes = np.full(len(gate), -1, dtype=np.int8)  # -1 = inconclusive

    for k in range(len(gate)):
        i   = pos_arr[k]
        d   = direction[k]
        tgt = tgt_pts[k]
        stp = stp_pts[k]
        en  = entries[k]
        today = dates_arr[k]
        for fwd in range(1, horizon_bars + 1):
            j = i + fwd
            if j >= n_df2 or df2_dates[j] != today:
                break
            h = df2_high[j]
            lv = df2_low[j]
            if d == 1:
                if h >= en + tgt:   outcomes[k] = 1; break
                if lv <= en - stp:  outcomes[k] = 0; break
            else:
                if lv <= en - tgt:  outcomes[k] = 1; break
                if h >= en + stp:   outcomes[k] = 0; break

    # Drop inconclusive
    mask = outcomes >= 0
    if not mask.any():
        return pd.DataFrame()

    gate2 = gate[mask].copy().reset_index(drop=True)
    dir2  = direction[mask]
    out2  = outcomes[mask]

    gate2["direction"]      = dir2
    gate2["target"]         = out2.astype(int)
    # Sign-flip directional features
    gate2["trail_ret_bps"]  = gate2["trail_ret_bps"]  * dir2
    gate2["session_ret_bps"]= gate2["ret_open"] * 10000 * dir2
    gate2["r2m_1"]          = gate2["r2m_1"] * dir2
    gate2["r2m_2"]          = gate2["r2m_2"] * dir2
    gate2["entry"]          = entries[mask]

    return gate2.reset_index(drop=True)


# ── Simulation (re-entry allowed) ─────────────────────────────────────────────

def simulate(test_feat: pd.DataFrame, df2_feat: pd.DataFrame,
             clf, fcols: list[str],
             target_bps: float, stop_bps: float, tick_size: float) -> pd.DataFrame:
    """
    Walk the test feature rows in time order.
    Re-entry is allowed as soon as a trade closes.
    Uses df2_feat (full bar series) to check TP/SL on non-qualifying bars.
    """
    feat_arr = test_feat[fcols].values
    proba    = clf.predict_proba(feat_arr)[:, 1]

    # Build a ts→(high,low) lookup from df2_feat for TP/SL tracking
    df2_lu = df2_feat.set_index("ts")[["high","low","bar_time","date"]]

    # Convert test_feat to list of dicts for fast access
    test_records = test_feat.reset_index(drop=True)

    trades   = []
    in_trade = False
    open_pos: dict = {}

    # We need to walk ALL bars in the test period (not just qualifying bars)
    # so TP/SL can be checked on non-qualifying bars.
    # Get test date range
    if test_feat.empty:
        return pd.DataFrame()
    t_min = test_feat["date"].min()
    t_max = test_feat["date"].max()
    all_bars = df2_feat[(df2_feat["date"] >= t_min) &
                        (df2_feat["date"] <= t_max) &
                        (df2_feat["bar_time"] >= BLACKOUT_END) &
                        (df2_feat["bar_time"] < RTH_END)].reset_index(drop=True)

    # Index test_feat qualifying rows by ts for O(1) lookup
    qual_by_ts = {row["ts"]: (i, proba[i]) for i, row in test_records.iterrows()}

    for _, bar in all_bars.iterrows():
        bar_t = bar["bar_time"]
        today = bar["date"]

        # ── Manage open position ──────────────────────────────────────────────
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
                    # fall through to check new entry on same bar
                else:
                    continue

        # ── Entry gate (only on qualifying bars) ─────────────────────────────
        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue
        ts = bar["ts"]
        if ts not in qual_by_ts:
            continue
        qi, p = qual_by_ts[ts]
        qrow = test_records.iloc[qi]
        if p < PROB_THRESHOLD or qrow.get("vol_regime", 0) < VOL_REGIME_MIN:
            continue

        d     = int(qrow["direction"])
        entry = qrow["entry"]
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
    df_t = pd.DataFrame(trades).dropna(subset=["outcome"])
    return df_t.reset_index(drop=True)


# ── Reporting ─────────────────────────────────────────────────────────────────

def summarise(label: str, trades: pd.DataFrame, test_days: int,
              tick_val: float, point_value: float) -> dict:
    if trades.empty:
        print(f"  {label}: no trades")
        return {}
    n   = len(trades)
    wr  = (trades["outcome"] == "TARGET").mean() * 100
    pnl = trades["pnl_pts"].sum()
    avg = trades["pnl_pts"].mean()
    usd = pnl * point_value
    exp = (trades["outcome"] == "EXPIRED").sum()
    per_day = n / test_days
    print(f"  {label}:  n={n} ({per_day:.2f}/day)  WR={wr:.1f}%  "
          f"avg={avg:+.2f}pt  P&L={pnl:+.1f}pt (~${usd:+,.0f})"
          + (f"  exp={exp}" if exp else ""), flush=True)
    return dict(n=n, per_day=per_day, wr=wr, avg_pnl=avg,
                total_pnl=pnl, usd=usd, expired=exp)


def detail_report(label: str, trades: pd.DataFrame, auc: float,
                  train_days: int, test_days: int,
                  tick_val: float, point_value: float):
    if trades.empty:
        print(f"\n{label}: no trades"); return
    n   = len(trades)
    wr  = (trades["outcome"] == "TARGET").mean() * 100
    pnl = trades["pnl_pts"].sum()
    usd = pnl * point_value
    avg = trades["pnl_pts"].mean()
    print(f"\n{'═'*68}")
    print(f"  DETAIL  ·  {label}")
    print(f"{'═'*68}")
    print(f"  AUC={auc:.4f}   train={train_days}d  test={test_days}d")
    print(f"  Trades: {n} ({n/test_days:.2f}/day)   WR={wr:.1f}%")
    print(f"  P&L: {pnl:+.1f}pt  avg={avg:+.2f}pt  (~${usd:+,.0f})")

    # TOD breakdown
    windows = [("9:40–11:00", dtime(9,40), dtime(11,0)),
               ("11:00–13:00", dtime(11,0), dtime(13,0)),
               ("13:00–15:45", dtime(13,0), dtime(15,45))]
    print("\n  Time-of-day:")
    for lbl, t0, t1 in windows:
        w = trades[(trades["bar_time"] >= t0) & (trades["bar_time"] < t1)]
        if w.empty:
            print(f"    {lbl:<14} —"); continue
        nw  = len(w)
        wrw = (w["outcome"] == "TARGET").mean() * 100
        pw  = w["pnl_pts"].sum()
        uw  = pw * point_value
        print(f"    {lbl:<14} n={nw:4d}  WR={wrw:5.1f}%  "
              f"pnl={pw:+.1f}pt (~${uw:+,.0f})")

    # Monthly P&L
    print("\n  Monthly P&L:")
    t2 = trades.copy()
    t2["month"] = pd.to_datetime(t2["date"]).dt.to_period("M")
    monthly = (t2.groupby("month").agg(
        n=("pnl_pts","count"),
        wr=("outcome", lambda x: (x=="TARGET").mean()*100),
        pnl=("pnl_pts","sum"),
    ).reset_index())
    for _, row in monthly.iterrows():
        bar = "█" * max(0, int(row["pnl"] / 1.5))
        neg = "░" * max(0, int(-row["pnl"] / 1.5))
        print(f"    {str(row['month']):<8}  n={int(row['n']):3d}  "
              f"WR={row['wr']:5.1f}%  pnl={row['pnl']:+6.1f}pt  {bar}{neg}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym  = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "sweep"

    cfg = SYMBOL_CONFIGS.get(sym)
    if cfg is None:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    tick_size   = cfg["tick_size"]
    tick_val    = cfg["tick_val"]
    point_value = cfg["point_value"]

    print(f"\n=== Trailing Momentum Backtest  ·  {sym}  ·  {mode.upper()} ===\n")
    print("Loading 1-min bars …", flush=True)
    df1 = load_1min(sym)

    print("Resampling to 2-min …", end=" ", flush=True)
    df2 = build_2min(df1)

    if mode == "single":
        configs = [SINGLE_CFG]
    else:
        configs = [
            dict(trail_bars=tb, move_thr_bps=mt, target_bps=tg,
                 stop_bps=st, horizon_bars=hz)
            for tb, mt, tg, st, hz
            in product(TRAIL_BARS, MOVE_THR_BPS, TARGET_BPS, STOP_BPS, HORIZON_BARS)
            if st < tg   # stop must be less than target
        ]
        print(f"\nSweep: {len(configs)} configs "
              f"({len(TRAIL_BARS)}×{len(MOVE_THR_BPS)}×{len(TARGET_BPS)}"
              f"×{len(STOP_BPS)}×{len(HORIZON_BARS)} with stp<tgt) …\n")

    fcols = get_feature_cols()
    summary_rows = []
    last_trail_bars = None
    df2_feat = None
    elig_cache = None
    n_trees = 300 if mode == "single" else 100   # faster sweep ranking

    for ci, sc in enumerate(configs):
        tb  = sc["trail_bars"]
        mt  = sc["move_thr_bps"]
        tg  = sc["target_bps"]
        st  = sc["stop_bps"]
        hz  = sc["horizon_bars"]

        if mode != "single" and (ci % 20 == 0):
            print(f"  {ci}/{len(configs)} …", flush=True)

        # Only rebuild features + eligible cache when trail_bars changes
        if tb != last_trail_bars:
            df2_feat = build_features(df2, tb)
            elig_cache = precompute_eligible(df2_feat)
            last_trail_bars = tb

        df_dir = build_dataset(elig_cache, df2_feat, mt, tg, st, hz, tick_size)
        if len(df_dir) < 200:
            continue

        dates  = sorted(df_dir["date"].unique())
        cutoff = dates[int(len(dates) * TRAIN_FRAC)]
        train  = df_dir[df_dir["date"] <  cutoff]
        test   = df_dir[df_dir["date"] >= cutoff]
        test_days = test["date"].nunique()
        train_days = train["date"].nunique()

        if len(train) < 100 or len(test) < 50 or test_days < 50:
            continue

        test2  = test.copy()

        clf = RandomForestClassifier(
            n_estimators=n_trees, max_depth=8, min_samples_leaf=15,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(train[fcols].values, train["target"].astype(int).values)

        proba = clf.predict_proba(test[fcols].values)[:, 1]
        auc   = roc_auc_score(test["target"].astype(int).values, proba)

        trades = simulate(test2, df2_feat, clf, fcols, tg, st, tick_size)


        label = (f"tb={tb}  mt={mt}bp  tg={tg}bp  st={st}bp  hz={hz}bar")

        if mode == "single":
            print(f"\n  Config: trail_bars={tb}  move_thr={mt}bp  "
                  f"target={tg}bp  stop={st}bp  horizon={hz}bars")
            print(f"  Train: {train_days}d  Test: {test_days}d  "
                  f"AUC={auc:.4f}  base={test['target'].mean()*100:.1f}%")
            info = summarise(label, trades, test_days, tick_val, point_value)
            if not trades.empty:
                detail_report(label, trades, auc, train_days, test_days,
                              tick_val, point_value)
                # Feature importances
                print("\n  Feature importances (RF):")
                fi = sorted(zip(fcols, clf.feature_importances_),
                            key=lambda x: -x[1])
                for fname, fv in fi:
                    bar = "█" * int(fv * 300)
                    print(f"    {fname:<20} {fv*100:5.1f}%  {bar}")
        else:
            if trades.empty:
                continue
            n_t     = len(trades)
            wr_t    = (trades["outcome"] == "TARGET").mean() * 100
            pnl_t   = trades["pnl_pts"].sum()
            usd_t   = pnl_t * point_value
            n_months = pd.to_datetime(trades["date"]).dt.to_period("M").nunique()
            monthly_pnl = (trades.assign(
                month=pd.to_datetime(trades["date"]).dt.to_period("M"))
                .groupby("month")["pnl_pts"].sum())
            conc = (monthly_pnl.abs().max() /
                    (trades["pnl_pts"].abs().sum() + 1e-9))
            summary_rows.append({
                "trail_bars": tb, "move_thr": mt, "target": tg,
                "stop": st, "horizon": hz,
                "n": n_t, "per_day": n_t / test_days,
                "wr": wr_t, "avg_pnl": trades["pnl_pts"].mean(),
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
        print(f"  SWEEP RESULTS  ·  {sym}  (sorted by total_pnl,  conc=monthly concentration)")
        print(f"{'═'*100}")
        cols = ["trail_bars","move_thr","target","stop","horizon",
                "n","per_day","wr","avg_pnl","total_pnl","usd","auc","months","conc"]
        df_show = df_sum[cols].sort_values("total_pnl", ascending=False).head(40)
        print(df_show.to_string(index=True, float_format="{:.2f}".format))

        # Also show top-10 excluding concentrated configs (conc < 0.6)
        broad = df_sum[df_sum["conc"] < 0.6].sort_values("total_pnl", ascending=False).head(10)
        if not broad.empty:
            print(f"\n{'─'*100}")
            print(f"  TOP CONFIGS WITH SPREAD ACROSS MONTHS (conc < 0.60)")
            print(f"{'─'*100}")
            print(broad[cols].to_string(index=True, float_format="{:.2f}".format))

        # Detail for top-3 broad configs
        broad_rows = [r for r in summary_rows if r["conc"] < 0.6]
        broad_rows.sort(key=lambda r: r["total_pnl"], reverse=True)
        for r in broad_rows[:3]:
            lbl = (f"tb={r['trail_bars']}  mt={r['move_thr']}bp  "
                   f"tg={r['target']}bp  st={r['stop']}bp  hz={r['horizon']}bar")
            detail_report(lbl, r["_trades"], r["auc"],
                          r["_train_days"], r["_test_days"],
                          tick_val, point_value)
        if not broad_rows:
            # Fall back to overall top-3
            top3 = sorted(summary_rows, key=lambda r: r["total_pnl"], reverse=True)[:3]
            for r in top3:
                lbl = (f"tb={r['trail_bars']}  mt={r['move_thr']}bp  "
                       f"tg={r['target']}bp  st={r['stop']}bp  hz={r['horizon']}bar")
                detail_report(lbl, r["_trades"], r["auc"],
                              r["_train_days"], r["_test_days"],
                              tick_val, point_value)


if __name__ == "__main__":
    main()
