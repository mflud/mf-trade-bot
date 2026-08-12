"""
backtest_pl_rev_sensitivity.py — Robustness check for PL_REV parameters.

Sweeps each parameter independently (others held at live values) to test
whether performance is stable or cliff-edge near the chosen thresholds.

Model: no ADX gate, 120s minimum gap between trades (realistic cooldown).

Live params (trading_bot.py):
  PL_THR=0.80, MOVE_BPS=20, TP_BPS=12, STOP_BPS=15, RESUME_PL=0.70

Usage:
  python src/backtest_pl_rev_sensitivity.py          # MES (default)
  python src/backtest_pl_rev_sensitivity.py MNQ
"""

import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

SYMBOL_CONFIGS = {
    "MES": dict(csv="es_hist_5sec.csv", point_value=5.0),
    "MNQ": dict(csv="nq_hist_5sec.csv", point_value=2.0),
}

SETTLEMENT_UTC = (21, 22)
RTH_START      = dtime(9, 30)
RTH_END        = dtime(16, 0)
BLACKOUT_END   = dtime(9, 40)
SESSION_END    = dtime(15, 55)
WINDOW         = 6    # 5s bars = 30s PL window
MAX_HOLD       = 24   # bars = 120s
MIN_HOLD       = 2    # bars = 10s
MIN_GAP_S      = 120  # realistic cooldown between trades

# ── Live values (centre of sweep) ─────────────────────────────────────────────
LIVE = dict(pl_thr=0.80, move_bps=20.0, tp_bps=12.0, stop_bps=15.0, resume_pl=0.70)

# ── Sweep ranges (one param varied at a time, others = LIVE) ──────────────────
SWEEPS = {
    "pl_thr":    [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    "move_bps":  [10, 14, 17, 20, 23, 26, 30],
    "tp_bps":    [7,  9, 11, 12, 13, 15, 18],
    "stop_bps":  [9, 11, 13, 15, 17, 19, 22],
    "resume_pl": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
}

PARAM_LABELS = {
    "pl_thr":    "PL threshold",
    "move_bps":  "Move (bp)",
    "tp_bps":    "Take profit (bp)",
    "stop_bps":  "Stop loss (bp)",
    "resume_pl": "Resume PL exit",
}

EXCLUDE_MONTHS: set[tuple[int, int]] = set()


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_5s(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    h = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC[0]) & (h < SETTLEMENT_UTC[1]))].copy()
    ts_et = df["ts"].dt.tz_convert(ET)
    df["bar_time"] = ts_et.dt.time
    df["date"]     = ts_et.dt.date
    df = df[(df["bar_time"] >= RTH_START) & (df["bar_time"] < RTH_END)].copy()
    if EXCLUDE_MONTHS:
        df = df[df["date"].apply(lambda d: (d.year, d.month) not in EXCLUDE_MONTHS)]
    df["gap"] = (df["ts"].diff() > pd.Timedelta(seconds=10)).values
    df.iloc[0, df.columns.get_loc("gap")] = True
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PL arrays (vectorised, recomputed if window changes — but window is fixed)
# ═══════════════════════════════════════════════════════════════════════════

def compute_pl_arrays(df: pd.DataFrame):
    s    = pd.Series(df["close"].values.astype(float), index=df.index)
    gaps = df["gap"].values.astype(bool)
    lr   = np.log(s / s.shift(1)).copy()
    lr.iloc[np.where(gaps)[0]] = np.nan
    abs_lr  = lr.abs()
    net_ret = lr.rolling(WINDOW, min_periods=WINDOW).sum()
    abs_sum = abs_lr.rolling(WINDOW, min_periods=WINDOW).sum()
    gap_in_window = abs_lr.isna().rolling(WINDOW, min_periods=1).max().astype(bool)
    net_ret[gap_in_window] = np.nan
    abs_sum[gap_in_window] = np.nan
    pl_arr  = (net_ret.abs() / abs_sum.replace(0, np.nan)).values
    dir_arr = np.sign(net_ret.values)
    closes  = df["close"].values.astype(float)
    lb      = np.full(len(closes), np.nan)
    lb[WINDOW:] = closes[:len(closes) - WINDOW]
    move_arr = np.abs(closes - lb) / np.where(lb > 0, lb, 1) * 10000
    move_arr[gap_in_window.values] = np.nan
    return pl_arr, dir_arr, move_arr


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate(df: pd.DataFrame,
             pl_arr: np.ndarray, dir_arr: np.ndarray, move_arr: np.ndarray,
             pl_thr: float, move_bps: float, tp_bps: float,
             stop_bps: float, resume_pl: float,
             point_value: float) -> dict:
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    gaps   = df["gap"].values.astype(bool)
    dates  = df["date"].values
    times  = df["bar_time"].values
    ts5s   = df["ts"].values
    n      = len(df)
    min_gap_ns   = np.timedelta64(MIN_GAP_S, "s")
    last_exit_ts = np.datetime64("NaT")

    trades = []
    i = 0
    while i < n:
        if not (times[i] >= BLACKOUT_END and times[i] < SESSION_END
                and not gaps[i]
                and not np.isnan(pl_arr[i]) and pl_arr[i] >= pl_thr
                and not np.isnan(move_arr[i]) and move_arr[i] >= move_bps):
            i += 1
            continue

        if last_exit_ts is not np.datetime64("NaT"):
            if ts5s[i] - last_exit_ts < min_gap_ns:
                i += 1
                continue

        direction = -dir_arr[i]
        entry     = closes[i]
        day_0     = dates[i]
        tp_pt     = entry * tp_bps   / 10000
        stop_pt   = entry * stop_bps / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != day_0:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -stop_bps); exit_i = j - 1; break
            if direction == 1 and highs[j] >= entry + tp_pt:
                pnl_bps = tp_bps; exit_i = j; break
            if direction == -1 and lows[j] <= entry - tp_pt:
                pnl_bps = tp_bps; exit_i = j; break
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -stop_bps; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -stop_bps; exit_i = j; break
            if k >= MIN_HOLD:
                if not np.isnan(pl_arr[j]) and pl_arr[j] >= resume_pl:
                    raw = (closes[j] - entry) * direction / entry * 10000
                    pnl_bps = max(raw, -stop_bps); exit_i = j; break
        else:
            j = i + MAX_HOLD
            raw = (closes[min(j, n-1)] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -stop_bps); exit_i = min(j, n - 1)

        trades.append(pnl_bps)
        last_exit_ts = ts5s[exit_i]
        i = exit_i + 1

    if not trades:
        return dict(n=0, ev=0.0, wr=0.0, pnl=0.0)

    pnls  = np.array(trades)
    ev    = pnls.mean()
    wr    = (pnls > 0).mean()
    n_t   = len(trades)
    pnl_d = ev * closes.mean() / 10000 * n_t * point_value
    return dict(n=n_t, ev=ev, wr=wr, pnl=pnl_d)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def run(symbol: str):
    cfg = SYMBOL_CONFIGS[symbol]
    pv  = cfg["point_value"]

    print(f"\n=== PL_REV Sensitivity Analysis  ·  {symbol}  "
          f"(no ADX gate, {MIN_GAP_S}s gap) ===")
    print(f"    Live: PL≥{LIVE['pl_thr']}  mv≥{LIVE['move_bps']}bp  "
          f"TP={LIVE['tp_bps']}bp  stop={LIVE['stop_bps']}bp  "
          f"resume≥{LIVE['resume_pl']}\n")

    print("Loading 5s bars …", flush=True)
    df = load_5s(cfg["csv"])
    print(f"  {len(df):,} bars  {df['date'].min()} → {df['date'].max()}")

    print("Computing PL arrays …", flush=True)
    pl_arr, dir_arr, move_arr = compute_pl_arrays(df)

    # ── Sweep each param ──────────────────────────────────────────────────────
    for param, values in SWEEPS.items():
        label = PARAM_LABELS[param]
        live_val = LIVE[param]

        print(f"\n{'─'*62}")
        print(f"  {label}  (others fixed at live values)")
        print(f"  {'Value':>8}  {'n':>5}  {'WR%':>6}  {'EV(bp)':>8}  "
              f"{'P&L $':>10}  {'vs live':>8}")
        print(f"  {'─'*60}")

        live_result = None
        rows = []
        for val in values:
            p = {**LIVE, param: val}
            r = simulate(df, pl_arr, dir_arr, move_arr,
                         p["pl_thr"], p["move_bps"], p["tp_bps"],
                         p["stop_bps"], p["resume_pl"], pv)
            rows.append((val, r))
            if abs(val - live_val) < 1e-9:
                live_result = r

        for val, r in rows:
            is_live = abs(val - live_val) < 1e-9
            marker  = " ◀ live" if is_live else ""
            if r["n"] == 0:
                print(f"  {val:>8}  {'—':>5}{marker}")
                continue
            if live_result and live_result["n"] > 0 and not is_live:
                delta_ev = r["ev"] - live_result["ev"]
                vs = f"{delta_ev:>+.2f}bp"
            else:
                vs = ""
            print(f"  {val:>8}  {r['n']:>5}  {r['wr']*100:>6.1f}  "
                  f"{r['ev']:>+8.2f}  {r['pnl']:>+10,.0f}  {vs:>8}{marker}")

    print(f"\n{'═'*62}")
    print("  Interpretation:")
    print("  Stable EV/WR across values → robust parameter choice.")
    print("  Sharp cliff at live value  → potential overfit, review carefully.")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use: {', '.join(SYMBOL_CONFIGS)}")
        sys.exit(1)
    run(symbol)
