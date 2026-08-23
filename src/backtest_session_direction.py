"""
backtest_session_direction.py — ML session-direction model.

Core idea: given the first N minutes of price action each day, predict whether
price will trend up or down by target_bps before reversing by stop_bps, within a
hold window.  One trade per day, entered at a fixed "signal bar" time.

Entry: signal bar fired once per session at a fixed time (e.g. 9:40 ET).
Direction: sign of move from 9:30 open to signal bar close.
Label: 1 if price hits +target_bps (from signal close) before -stop_bps within
       horizon 2-min bars; 0 if stop hits first; skip if inconclusive.

Features (all computed from bars up to but NOT including the signal bar):
  gap_bps           — log(9:30 open / prior day close) * 10000
  pre_range_bps     — (max_high - min_low) of 8:30–9:29 pre-market bars / 9:30 open * 10000
  pre_ret_bps       — log(9:30 open / 8:30 first close) * 10000
  open_ret_bps      — log(signal_close / 9:30 open) * 10000  [sign-flipped by direction]
  open_range_bps    — (max_high - min_low) of 9:30–signal bars / signal_close * 10000
  open_vol          — realized vol of 1-min log returns from 9:30 to signal bar
  vol_regime        — percentile rank of open_vol vs prior 60-day rolling window
  atr_ratio         — today's open_range vs 20-day rolling avg open_range
  prior_day_range_bps — prior day (high-low) / prior day close * 10000
  prior_day_ret_bps   — prior day log return * 10000
  dow_sin, dow_cos  — cyclic day-of-week encoding (0=Mon … 4=Fri)

All return/momentum features are sign-flipped by direction so the model learns
"continuation" regardless of long/short.

Sweep parameters:
  signal_time  : [9:40, 9:50, 10:00, 10:15, 10:30] ET
  target_bps   : [20, 30, 40, 50]
  stop_bps     : [10, 15, 20]  (always < target → R:R ≥ 1.5:1)
  horizon_bars : [30, 45, 60, 90] 2-min bars (60/90/120/180 min)
  Train/test   : 75/25 walk-forward split by date

Usage (from repo root):
    python src/backtest_session_direction.py [MES|MNQ] [sweep|single]
"""

import math
import sys
import sqlite3
import warnings
import numpy as np
import pandas as pd
from datetime import time as dtime, date
from zoneinfo import ZoneInfo
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Constants ─────────────────────────────────────────────────────────────────

ET = ZoneInfo("America/New_York")
DB_PATH   = "data/bars.db"

# Symbol configs
SYMBOL_CONFIGS = {
    "MES": dict(hist_csv="mes_hist_1min.csv", tick_size=0.25, tick_val=1.25),
    "MNQ": dict(hist_csv="mnq_hist_1min.csv", tick_size=0.25, tick_val=0.50),
}

PRE_START  = dtime(8, 30)   # pre-market start
RTH_START  = dtime(9, 30)   # NYSE open
RTH_END    = dtime(16, 0)   # NYSE close
TRAIN_FRAC = 0.75

# Rolling windows for regime features
VOL_REGIME_DAYS = 60   # days of history for vol_regime percentile
ATR_RATIO_DAYS  = 20   # days for rolling avg open_range

# Sweep parameter space
SIGNAL_TIMES  = [dtime(9, 40), dtime(9, 50), dtime(10, 0), dtime(10, 15), dtime(10, 30)]
TARGET_BPS    = [20, 30, 40, 50]
STOP_BPS      = [10, 15, 20]
HORIZON_BARS  = [30, 45, 60, 90]   # 2-min bars


# ── Helpers ───────────────────────────────────────────────────────────────────

def pts_from_bps(price: float, bps: float, tick_size: float = 0.25) -> float:
    """Convert bps target/stop to the nearest tick above the raw price distance."""
    raw   = price * bps / 10000.0
    ticks = math.ceil(raw / tick_size)
    return round(ticks * tick_size, 4)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_1min(symbol: str) -> pd.DataFrame:
    """
    Load 1-min bars for symbol from hist CSV + bars.db.
    Keeps both pre-market (8:30–9:30) and RTH (9:30–16:00) bars.
    Returns a DataFrame sorted by ts with tz-aware timestamps in ET.
    """
    cfg    = SYMBOL_CONFIGS[symbol]
    frames = []

    # Historical CSV (covers both pre-mkt and RTH)
    hist = Path(cfg["hist_csv"])
    if hist.exists():
        df_h = pd.read_csv(hist, parse_dates=["ts"])
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= PRE_START) & (t < RTH_END)]
        frames.append(df_h)

    # Live bars.db
    db    = sqlite3.connect(DB_PATH)
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

    print(f"  Loaded {len(df):,} bars ({PRE_START}–{RTH_END})  "
          f"{df['ts'].dt.date.min()} → {df['ts'].dt.date.max()}", flush=True)
    return df


# ── Feature construction ──────────────────────────────────────────────────────

def build_daily_rows(df: pd.DataFrame, signal_time: dtime) -> pd.DataFrame:
    """
    Build one feature row per tradeable day for a given signal_time.

    For each day the function requires:
      - At least one 8:30 pre-market bar (to compute pre_ret_bps)
      - The 9:30 bar (RTH open)
      - A bar at exactly signal_time (used as the signal bar)
      - A prior trading day (to compute gap_bps and prior-day features)

    Rows where pre-market bars are missing are returned with NaN features and
    must be dropped before training.

    Returns DataFrame with columns:
        date, signal_ts, signal_close, direction,
        gap_bps, pre_range_bps, pre_ret_bps,
        open_ret_bps, open_range_bps, open_vol,
        vol_regime, atr_ratio,
        prior_day_range_bps, prior_day_ret_bps,
        dow_sin, dow_cos
    All return/momentum features are already sign-flipped by direction.
    """
    df = df.copy()
    df["date"] = df["ts"].dt.date
    df["bar_time"] = df["ts"].dt.time

    # Pre-group all bars by date once — avoids O(N*D) re-scans
    by_date: dict = {d: g for d, g in df.groupby("date")}
    dates = sorted(by_date.keys())

    # ── Pre-compute per-day summaries ─────────────────────────────────────────

    # 9:30 open prices keyed by date
    open_930: dict[date, float] = {}
    for d, grp in df[df["bar_time"] == RTH_START].groupby("date"):
        open_930[d] = grp["open"].iloc[0]

    # Prior day close / high / low (from RTH bars only)
    rth_mask = (df["bar_time"] >= RTH_START) & (df["bar_time"] < RTH_END)
    rth_bars = df[rth_mask]
    day_close: dict[date, float] = {}
    day_high:  dict[date, float] = {}
    day_low:   dict[date, float] = {}
    for d, grp in rth_bars.groupby("date"):
        day_close[d] = grp["close"].iloc[-1]
        day_high[d]  = grp["high"].max()
        day_low[d]   = grp["low"].min()

    # Ordered list of days that have RTH close data (for prior-day lookup)
    rth_days = sorted(day_close.keys())
    rth_day_set = set(rth_days)

    # ── Rolling history accumulators (maintained in date order) ───────────────
    # Used for vol_regime and atr_ratio — always computed from past days only.
    hist_open_vol:   list[tuple] = []   # (date, float)
    hist_open_range: list[tuple] = []   # (date, float)

    rows = []

    for i, d in enumerate(dates):
        grp = by_date[d]

        # Need a prior RTH day for gap and prior-day features
        prior_rth = [dd for dd in rth_days if dd < d]
        if not prior_rth:
            # Still accumulate vol/range history for future days
            # (use 0 placeholders — they'll be NaN since not added)
            continue
        prior_date     = prior_rth[-1]
        prior2_date    = prior_rth[-2] if len(prior_rth) >= 2 else None

        # Must have 9:30 open bar today
        if d not in open_930:
            continue
        open_px = open_930[d]

        # Signal bar — must exist at exactly signal_time
        sig_grp = grp[grp["bar_time"] == signal_time]
        if sig_grp.empty:
            continue
        sig_bar      = sig_grp.iloc[0]
        signal_ts    = sig_bar["ts"]
        signal_close = sig_bar["close"]

        # Direction: sign of move from 9:30 open to signal bar close
        direction = 1 if signal_close >= open_px else -1

        # ── Pre-market features ───────────────────────────────────────────────
        pre_grp = grp[grp["bar_time"] < RTH_START]

        if pre_grp.empty:
            # No pre-market data — will be dropped in training (NaN)
            gap_bps       = np.nan
            pre_range_bps = np.nan
            pre_ret_bps   = np.nan
        else:
            # gap_bps: log return from prior close to 9:30 open
            pc      = day_close.get(prior_date, np.nan)
            gap_bps = np.log(open_px / pc) * 10000 if (pc and pc > 0) else np.nan

            # pre_range_bps: (high - low) of all pre-market bars / open
            pre_range_pts = pre_grp["high"].max() - pre_grp["low"].min()
            pre_range_bps = pre_range_pts / open_px * 10000 if open_px > 0 else np.nan

            # pre_ret_bps: log(9:30 open / first 8:30 bar close)
            first_pre_close = pre_grp.iloc[0]["close"]
            pre_ret_bps = (np.log(open_px / first_pre_close) * 10000
                           if first_pre_close > 0 else np.nan)

        # ── RTH intraday features ─────────────────────────────────────────────
        # "before_sig": bars strictly between 9:30 open and signal bar (not including)
        rth_before = grp[(grp["bar_time"] >= RTH_START) & (grp["bar_time"] < signal_time)]
        # "up_to_sig": bars from 9:30 through signal bar (inclusive)
        rth_up_to  = grp[(grp["bar_time"] >= RTH_START) & (grp["bar_time"] <= signal_time)]

        if rth_before.empty:
            open_ret_bps    = np.nan
            open_range_bps  = np.nan
            open_vol        = np.nan
            range_pts_today = np.nan
        else:
            # open_ret_bps: log(signal_close / 9:30 open), sign-flipped by direction
            open_ret_bps = np.log(signal_close / open_px) * 10000 * direction

            # open_range_bps: range from 9:30 to signal bar (inclusive)
            range_pts_today = rth_up_to["high"].max() - rth_up_to["low"].min()
            open_range_bps  = (range_pts_today / signal_close * 10000
                               if signal_close > 0 else np.nan)

            # open_vol: std of 1-min log returns from 9:30 to signal bar
            cls_seq  = rth_before["close"].values
            if len(cls_seq) >= 2:
                log_rets = np.log(cls_seq[1:] / cls_seq[:-1])
                open_vol = float(np.std(log_rets, ddof=1)) if len(log_rets) >= 2 else np.nan
            else:
                open_vol = np.nan

        # ── Rolling vol_regime ────────────────────────────────────────────────
        # Percentile rank of today's open_vol vs the prior VOL_REGIME_DAYS days
        if open_vol is not None and not np.isnan(open_vol):
            past_vols = [v for (_, v) in hist_open_vol[-VOL_REGIME_DAYS:]
                         if not np.isnan(v)]
            vol_regime = (float(np.mean(np.array(past_vols) <= open_vol))
                          if len(past_vols) >= 10 else np.nan)
        else:
            vol_regime = np.nan

        # ── Rolling atr_ratio ─────────────────────────────────────────────────
        # Today's open_range vs mean of prior ATR_RATIO_DAYS days
        past_ranges = [r for (_, r) in hist_open_range[-ATR_RATIO_DAYS:]
                       if not np.isnan(r)]
        if past_ranges and not np.isnan(range_pts_today):
            atr_ratio = range_pts_today / np.mean(past_ranges)
        else:
            atr_ratio = np.nan

        # ── Prior day features ────────────────────────────────────────────────
        pd_close = day_close.get(prior_date, np.nan)
        pd_high  = day_high.get(prior_date, np.nan)
        pd_low   = day_low.get(prior_date, np.nan)
        prior_day_range_bps = (
            (pd_high - pd_low) / pd_close * 10000
            if (pd_close and pd_close > 0
                and not np.isnan(pd_high) and not np.isnan(pd_low))
            else np.nan
        )

        # Prior day log return (requires two prior RTH days)
        if prior2_date is not None:
            pp_close = day_close.get(prior2_date, np.nan)
            prior_day_ret_bps = (
                np.log(pd_close / pp_close) * 10000
                if (pp_close and pp_close > 0 and pd_close and pd_close > 0)
                else np.nan
            )
        else:
            prior_day_ret_bps = np.nan

        # ── Sign-flip directional features ───────────────────────────────────
        # gap_bps, pre_ret_bps, prior_day_ret_bps are flipped so the model
        # always sees them from the "continuation" perspective.
        gap_bps_dir       = gap_bps * direction       if not np.isnan(gap_bps)           else np.nan
        pre_ret_bps_dir   = pre_ret_bps * direction   if not np.isnan(pre_ret_bps)        else np.nan
        prior_ret_bps_dir = prior_day_ret_bps * direction if not np.isnan(prior_day_ret_bps) else np.nan

        # ── Cyclic day-of-week encoding ───────────────────────────────────────
        dow     = pd.Timestamp(d).dayofweek   # 0=Mon … 4=Fri
        dow_sin = math.sin(2 * math.pi * dow / 5)
        dow_cos = math.cos(2 * math.pi * dow / 5)

        # ── Update rolling history before moving to next day ──────────────────
        hist_open_vol.append((d, open_vol if (open_vol is not None and not np.isnan(open_vol)) else np.nan))
        hist_open_range.append((d, range_pts_today if not np.isnan(range_pts_today) else np.nan))

        rows.append({
            "date":                d,
            "signal_ts":           signal_ts,
            "signal_close":        signal_close,
            "direction":           direction,
            # Features — directional features already sign-flipped by direction
            "gap_bps":             gap_bps_dir,
            "pre_range_bps":       pre_range_bps,
            "pre_ret_bps":         pre_ret_bps_dir,
            "open_ret_bps":        open_ret_bps,
            "open_range_bps":      open_range_bps,
            "open_vol":            open_vol,
            "vol_regime":          vol_regime,
            "atr_ratio":           atr_ratio,
            "prior_day_range_bps": prior_day_range_bps,
            "prior_day_ret_bps":   prior_ret_bps_dir,
            "dow_sin":             dow_sin,
            "dow_cos":             dow_cos,
        })

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "gap_bps", "pre_range_bps", "pre_ret_bps",
    "open_ret_bps", "open_range_bps", "open_vol",
    "vol_regime", "atr_ratio",
    "prior_day_range_bps", "prior_day_ret_bps",
    "dow_sin", "dow_cos",
]


# ── Labelling ─────────────────────────────────────────────────────────────────

def add_labels(df_daily: pd.DataFrame,
               df_2min: pd.DataFrame,
               target_bps: int,
               stop_bps: int,
               horizon_bars: int,
               tick_size: float) -> pd.DataFrame:
    """
    For each daily row, simulate a trade entered at signal_close in `direction`.

    TP = signal_close + direction * pts_from_bps(signal_close, target_bps)
    SL = signal_close - direction * pts_from_bps(signal_close, stop_bps)

    Walk forward through 2-min bars (starting from the bar *after* signal_ts),
    capped at horizon_bars, checking bar highs/lows.

    label = 1  → target hit before stop
    label = 0  → stop hit before target
    skip       → neither within horizon (inconclusive)

    Returns df with a 'label' column (NaN rows dropped).
    """
    # Build a lookup: date → list of 2-min bars after signal_time, sorted by ts
    df_2min = df_2min.copy()
    df_2min["date"] = df_2min["ts"].dt.date

    rows = []
    for _, row in df_daily.iterrows():
        d          = row["date"]
        sig_ts     = row["signal_ts"]
        entry      = row["signal_close"]
        direction  = row["direction"]

        tgt_pts = pts_from_bps(entry, target_bps, tick_size)
        stp_pts = pts_from_bps(entry, stop_bps,   tick_size)

        if direction == 1:
            tp = entry + tgt_pts
            sl = entry - stp_pts
        else:
            tp = entry - tgt_pts
            sl = entry + stp_pts

        # Forward 2-min bars: same day, strictly after signal_ts
        fwd = df_2min[(df_2min["date"] == d) & (df_2min["ts"] > sig_ts)].head(horizon_bars)

        if fwd.empty:
            continue   # no forward bars — skip

        label     = np.nan
        hit_tgt   = False
        hit_stp   = False
        for _, b in fwd.iterrows():
            if direction == 1:
                if b["high"] >= tp:
                    hit_tgt = True; break
                if b["low"]  <= sl:
                    hit_stp = True; break
            else:
                if b["low"]  <= tp:
                    hit_tgt = True; break
                if b["high"] >= sl:
                    hit_stp = True; break

        if not hit_tgt and not hit_stp:
            continue   # inconclusive — skip

        label = 1 if hit_tgt else 0

        new_row = row.to_dict()
        new_row["label"]    = label
        new_row["tgt_pts"]  = tgt_pts
        new_row["stp_pts"]  = stp_pts
        rows.append(new_row)

    return pd.DataFrame(rows)


# ── Train / test split ────────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame):
    """75/25 split by date — no lookahead."""
    dates  = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRAC)]
    train  = df[df["date"] <  cutoff].copy()
    test   = df[df["date"] >= cutoff].copy()
    return train, test, cutoff


# ── RF model ──────────────────────────────────────────────────────────────────

def train_rf(train: pd.DataFrame, fcols: list[str]) -> RandomForestClassifier:
    X = train[fcols].values
    y = train["label"].astype(int).values
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def compute_auc(clf: RandomForestClassifier, test: pd.DataFrame, fcols: list[str]) -> float:
    y   = test["label"].astype(int).values
    prb = clf.predict_proba(test[fcols].values)[:, 1]
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, prb)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_trades(test: pd.DataFrame, clf: RandomForestClassifier,
                    fcols: list[str]) -> pd.DataFrame:
    """
    Walk through test rows in date order.  One trade per day — fire at signal
    bar and hold until TP, SL, or end of session (which we already know from the
    'label' column since we've pre-simulated).

    The label already encodes the outcome.  Here we use the model's probability
    to decide whether to take the trade (threshold = 0.5) and then read the
    known outcome from the label column.

    Returns a DataFrame of trades taken by the model, with pnl_pts.
    """
    proba = clf.predict_proba(test[fcols].values)[:, 1]
    test  = test.copy()
    test["prob"] = proba

    trades = []
    for _, row in test.iterrows():
        if row["prob"] < 0.5:
            continue   # model says no-trade
        outcome_win = (row["label"] == 1)
        pnl = row["tgt_pts"] if outcome_win else -row["stp_pts"]
        trades.append({
            "date":      row["date"],
            "direction": row["direction"],
            "entry":     row["signal_close"],
            "tgt_pts":   row["tgt_pts"],
            "stp_pts":   row["stp_pts"],
            "prob":      row["prob"],
            "outcome":   "TARGET" if outcome_win else "STOPPED",
            "pnl_pts":   pnl,
            "label":     row["label"],
        })

    return pd.DataFrame(trades)


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(df_all: pd.DataFrame, df_2min: pd.DataFrame,
              tick_size: float, symbol: str) -> pd.DataFrame:
    """
    Iterate over all parameter combinations and return a summary DataFrame.
    """
    configs = [
        (st, tgt, stp, hz)
        for st  in SIGNAL_TIMES
        for tgt in TARGET_BPS
        for stp in STOP_BPS
        for hz  in HORIZON_BARS
        if stp < tgt   # enforce R:R ≥ 1.5:1 (stop strictly < target)
    ]

    print(f"\nRunning sweep: {len(configs)} configs …\n", flush=True)

    results = []
    n_cfg   = len(configs)

    for idx, (sig_time, tgt_bps, stp_bps, hz) in enumerate(configs, 1):
        if idx % 20 == 0 or idx == n_cfg:
            print(f"  {idx}/{n_cfg} …", flush=True)

        # Select the pre-built daily rows for this signal_time
        sig_key = sig_time.strftime("%H:%M")
        if sig_key not in _daily_cache:
            continue
        df_daily = _daily_cache[sig_key]

        # Add labels
        labeled = add_labels(df_daily, df_2min, tgt_bps, stp_bps, hz, tick_size)
        if len(labeled) < 60:
            continue   # too few samples

        # Drop rows missing any feature
        labeled = labeled.dropna(subset=FEATURE_COLS + ["label"]).copy()
        if len(labeled) < 60:
            continue

        train, test, _ = walk_forward_split(labeled)
        if len(train) < 30 or len(test) < 15:
            continue

        clf  = train_rf(train, FEATURE_COLS)
        auc  = compute_auc(clf, test, FEATURE_COLS)

        # Simulate trades on test set
        trades = simulate_trades(test, clf, FEATURE_COLS)

        n_trades   = len(trades)
        test_days  = test["date"].nunique()
        trades_day = n_trades / test_days if test_days > 0 else 0.0

        if n_trades == 0:
            wr = np.nan; avg_pnl = np.nan; total_pnl = np.nan
        else:
            wr        = (trades["outcome"] == "TARGET").mean() * 100
            avg_pnl   = trades["pnl_pts"].mean()
            total_pnl = trades["pnl_pts"].sum()

        results.append({
            "signal_time":  sig_time.strftime("%H:%M"),
            "target_bps":   tgt_bps,
            "stop_bps":     stp_bps,
            "horizon":      hz,
            "n_trades":     n_trades,
            "trades_day":   round(trades_day, 2),
            "WR":           round(wr, 1) if not np.isnan(wr) else np.nan,
            "avg_pnl_pts":  round(avg_pnl, 3) if not np.isnan(avg_pnl) else np.nan,
            "total_pnl_pts": round(total_pnl, 2) if not np.isnan(total_pnl) else np.nan,
            "AUC":          round(auc, 4) if not np.isnan(auc) else np.nan,
        })

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res

    # Sort by total_pnl_pts descending
    df_res = df_res.sort_values("total_pnl_pts", ascending=False).reset_index(drop=True)
    return df_res


# ── Detail report ─────────────────────────────────────────────────────────────

def run_detail(labeled: pd.DataFrame, tick_size: float, tick_val: float,
               symbol: str, cfg_label: str):
    """
    Train/test on the best config and print a full breakdown:
      - Monthly P&L bar chart
      - Outcome distribution
      - Day-of-week breakdown
    """
    labeled = labeled.dropna(subset=FEATURE_COLS + ["label"]).copy()
    train, test, cutoff = walk_forward_split(labeled)

    if len(train) < 20 or len(test) < 10:
        print("  Insufficient data for detail report.")
        return

    clf    = train_rf(train, FEATURE_COLS)
    auc    = compute_auc(clf, test, FEATURE_COLS)
    trades = simulate_trades(test, clf, FEATURE_COLS)

    n_test_days = test["date"].nunique()
    n_train_days = train["date"].nunique()

    print(f"\n{'═'*70}")
    print(f"  DETAIL REPORT  ·  {symbol}  ·  {cfg_label}")
    print(f"{'═'*70}")
    print(f"  Train: {n_train_days} days ({train['date'].min()} → {cutoff})")
    print(f"  Test : {n_test_days} days ({cutoff} → {test['date'].max()})")
    print(f"  AUC  : {auc:.4f}   Base rate: {labeled['label'].mean()*100:.1f}%")

    if trades.empty:
        print("  No trades taken by model.")
        return

    total_n   = len(trades)
    total_pnl = trades["pnl_pts"].sum()
    wr        = (trades["outcome"] == "TARGET").mean() * 100
    pnl_usd   = total_pnl * tick_val * (1 / tick_size)

    print(f"\n  Trades : {total_n}  ({total_n/n_test_days:.2f}/day)")
    print(f"  Win rate: {wr:.1f}%")
    print(f"  Total P&L: {total_pnl:+.2f} pts  (~${pnl_usd:+,.0f})")
    print(f"  Avg P&L  : {trades['pnl_pts'].mean():+.3f} pts/trade")

    # ── Outcome distribution ──────────────────────────────────────────────────
    n_target  = (trades["outcome"] == "TARGET").sum()
    n_stopped = (trades["outcome"] == "STOPPED").sum()
    # Count days model chose not to trade
    n_total_labeled = len(test)
    n_no_trade = n_total_labeled - total_n

    print(f"\n  Outcome distribution (test set labeled days={n_total_labeled}):")
    print(f"    Target  hit : {n_target:>4}  ({n_target/n_total_labeled*100:.1f}%)")
    print(f"    Stopped     : {n_stopped:>4}  ({n_stopped/n_total_labeled*100:.1f}%)")
    print(f"    No-trade    : {n_no_trade:>4}  ({n_no_trade/n_total_labeled*100:.1f}%)")

    # Also show full label distribution (including model-skipped)
    n_lbl_target  = (test["label"] == 1).sum()
    n_lbl_stopped = (test["label"] == 0).sum()
    print(f"\n  True label distribution (all labeled test days):")
    print(f"    Would-be target : {n_lbl_target:>4}  ({n_lbl_target/n_total_labeled*100:.1f}%)")
    print(f"    Would-be stopped: {n_lbl_stopped:>4}  ({n_lbl_stopped/n_total_labeled*100:.1f}%)")

    # ── Day-of-week breakdown ─────────────────────────────────────────────────
    print(f"\n  Day-of-week breakdown:")
    print(f"    {'DOW':<5}  {'N':>4}  {'WR':>6}  {'avg_pnl':>9}  {'total_pnl':>10}")
    trades["dow_label"] = pd.to_datetime(trades["date"]).dt.strftime("%a")
    for dow in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
        sub = trades[trades["dow_label"] == dow]
        if sub.empty:
            continue
        n_d   = len(sub)
        wr_d  = (sub["outcome"] == "TARGET").mean() * 100
        avg_d = sub["pnl_pts"].mean()
        tot_d = sub["pnl_pts"].sum()
        print(f"    {dow:<5}  {n_d:>4}  {wr_d:>5.1f}%  {avg_d:>+9.3f}  {tot_d:>+10.2f}")

    # ── Monthly P&L bar chart ─────────────────────────────────────────────────
    print(f"\n  Monthly P&L (test set):")
    print(f"    {'Month':<8}  {'N':>4}  {'WR':>6}  {'pnl_pts':>9}  chart")
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M")
    monthly = (trades.groupby("month")
                     .agg(n=("pnl_pts", "count"),
                          wr=("outcome", lambda x: (x == "TARGET").mean() * 100),
                          pnl=("pnl_pts", "sum"))
                     .reset_index())
    for _, mrow in monthly.iterrows():
        bar_len = int(abs(mrow["pnl"]) / max(1, abs(monthly["pnl"]).max()) * 30)
        bar = ("█" if mrow["pnl"] >= 0 else "░") * bar_len
        print(f"    {str(mrow['month']):<8}  {int(mrow['n']):>4}  "
              f"{mrow['wr']:>5.1f}%  {mrow['pnl']:>+9.2f}  {bar}")

    # ── Feature importances ───────────────────────────────────────────────────
    print(f"\n  Feature importances (RF):")
    imp_pairs = sorted(zip(FEATURE_COLS, clf.feature_importances_), key=lambda x: -x[1])
    for fname, fval in imp_pairs:
        bar = "█" * int(fval * 400)
        print(f"    {fname:<22} {fval*100:5.2f}%  {bar}")

    print()


# ── 2-min resampler ───────────────────────────────────────────────────────────

def resample_2min(df1: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min bars to 2-min for TP/SL simulation (RTH only)."""
    df1_rth = df1[(df1["ts"].dt.time >= RTH_START) & (df1["ts"].dt.time < RTH_END)].copy()
    df1_idx = df1_rth.set_index("ts")
    df2 = df1_idx[["open", "high", "low", "close", "volume"]].resample(
        "2min", label="right", closed="right"
    ).agg(open=("open", "first"), high=("high", "max"),
          low=("low", "min"),   close=("close", "last"),
          volume=("volume", "sum")).dropna(subset=["close"])
    df2.index = df2.index.tz_convert(ET)
    t2 = df2.index.time
    df2 = df2[(t2 >= RTH_START) & (t2 < RTH_END)].reset_index()
    df2["date"] = df2["ts"].dt.date
    return df2


# ── Cache for daily row building ──────────────────────────────────────────────
# We build daily rows once per signal_time (expensive), then reuse for all
# target/stop/horizon combos at that signal_time.
_daily_cache: dict[str, pd.DataFrame] = {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    sym  = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "sweep"

    if sym not in SYMBOL_CONFIGS:
        print(f"Unknown symbol '{sym}'.  Use MES or MNQ.")
        sys.exit(1)
    if mode not in ("sweep", "single"):
        print(f"Unknown mode '{mode}'.  Use sweep or single.")
        sys.exit(1)

    cfg       = SYMBOL_CONFIGS[sym]
    tick_size = cfg["tick_size"]
    tick_val  = cfg["tick_val"]

    print(f"\n=== Session Direction Backtest  ·  {sym}  ·  {mode.upper()} ===\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading 1-min bars …", flush=True)
    df_all = load_all_1min(sym)

    print("Resampling to 2-min (for TP/SL simulation) …", flush=True)
    df_2min = resample_2min(df_all)
    print(f"  {len(df_2min):,} 2-min RTH bars", flush=True)

    # ── Build daily feature rows for each signal_time ─────────────────────────
    print("\nBuilding daily feature rows for each signal_time …", flush=True)
    for sig_time in SIGNAL_TIMES:
        sig_key = sig_time.strftime("%H:%M")
        print(f"  signal_time={sig_key} …", end=" ", flush=True)
        _daily_cache[sig_key] = build_daily_rows(df_all, sig_time)
        n = len(_daily_cache[sig_key])
        print(f"{n} days", flush=True)

    # ── Sweep or single ───────────────────────────────────────────────────────
    if mode == "sweep":
        df_res = run_sweep(df_all, df_2min, tick_size, sym)

        if df_res.empty:
            print("No results — check data coverage.")
            return

        # Print sweep summary table
        print(f"\n{'═'*100}")
        print(f"  SWEEP RESULTS  ·  {sym}  (sorted by total_pnl_pts)")
        print(f"{'═'*100}")
        pd.set_option("display.width", 140)
        pd.set_option("display.max_rows", 200)
        print(df_res.to_string(index=True))

        # Detail on the best config
        best = df_res.iloc[0]
        best_sig_time = dtime(*map(int, best["signal_time"].split(":")))
        best_tgt      = int(best["target_bps"])
        best_stp      = int(best["stop_bps"])
        best_hz       = int(best["horizon"])

        sig_key = best["signal_time"]
        df_daily_best = _daily_cache[sig_key]
        labeled_best  = add_labels(df_daily_best, df_2min, best_tgt, best_stp, best_hz, tick_size)
        labeled_best  = labeled_best.dropna(subset=FEATURE_COLS + ["label"])

        cfg_label = (f"signal={sig_key}  tgt={best_tgt}bps  "
                     f"stp={best_stp}bps  hz={best_hz}bars")
        run_detail(labeled_best, tick_size, tick_val, sym, cfg_label)

    else:
        # Single: use first signal_time and first target/stop/horizon as a quick test
        sig_time = SIGNAL_TIMES[0]   # 9:40
        tgt_bps  = TARGET_BPS[1]     # 30
        stp_bps  = STOP_BPS[0]       # 10
        hz       = HORIZON_BARS[1]   # 45 bars = 90 min

        sig_key   = sig_time.strftime("%H:%M")
        df_daily  = _daily_cache[sig_key]
        labeled   = add_labels(df_daily, df_2min, tgt_bps, stp_bps, hz, tick_size)

        cfg_label = (f"signal={sig_key}  tgt={tgt_bps}bps  "
                     f"stp={stp_bps}bps  hz={hz}bars")
        run_detail(labeled, tick_size, tick_val, sym, cfg_label)


if __name__ == "__main__":
    main()
