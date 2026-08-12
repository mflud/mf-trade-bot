"""
backtest_pl_rev_adx.py — Sweep ADX gate thresholds for PL_REV (daily vs hourly).

Two gate modes, each swept across ADX thresholds [15, 20, 25, 30, 35, 40, 50, ∞]:
  daily  — ADX(14) on 9:00-10:00 ET 1-min bars, held for entire day
  hourly — ADX(14) on trailing 60 1-min bars, recomputed at each signal

ADX values are pre-computed once per bar; threshold sweep is fast.

Live params used (from trading_bot.py):
  PL ≥ 0.80, move ≥ 20bp, TP = 12bp, stop = 15bp, max_hold = 120s

Usage:
  python src/backtest_pl_rev_adx.py          # MES (default)
  python src/backtest_pl_rev_adx.py MNQ
"""

import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

SYMBOL_CONFIGS = {
    "MES": dict(csv="es_hist_5sec.csv",  tick_size=0.25, point_value=5.0),
    "MNQ": dict(csv="nq_hist_5sec.csv",  tick_size=0.25, point_value=2.0),
}

SETTLEMENT_UTC  = (21, 22)
RTH_START       = dtime(9, 30)
RTH_END         = dtime(16, 0)
BLACKOUT_END    = dtime(9, 40)
SESSION_END     = dtime(15, 55)

# Fixed live params
PL_THR    = 0.80
MOVE_BPS  = 20.0
TP_BPS    = 12.0
STOP_BPS  = 15.0
WINDOW    = 6       # 5s bars = 30s PL window
MAX_HOLD  = 24      # bars = 120s
MIN_HOLD  = 2       # bars = 10s
RESUME_PL = 0.70

# ADX params
ADX_PERIOD      = 14
ADX_DAILY_START = dtime(9, 0)
ADX_DAILY_END   = dtime(10, 0)
ADX_HOURLY_BARS = 60   # trailing 1-min bars

# Thresholds to sweep (plus inf = no gate)
ADX_THRESHOLDS = [15, 20, 25, 30, 35, 40, 50, float("inf")]

# Minimum gap between trade exit and next entry (seconds)
# 0 = original backtest; 120 = realistic (one trade at a time, max hold = 120s)
MIN_GAP_SWEEP = [0, 120]

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


def build_1min(df5s: pd.DataFrame) -> pd.DataFrame:
    df = df5s.copy().set_index("ts")
    ohlc          = df["close"].resample("1min").ohlc()
    ohlc["high"]  = df["high"].resample("1min").max()
    ohlc["low"]   = df["low"].resample("1min").min()
    ohlc["close"] = df["close"].resample("1min").last()
    ohlc = ohlc.dropna(subset=["close"])
    ts_et = ohlc.index.tz_convert(ET)
    ohlc["bar_time"] = ts_et.time
    ohlc["date"]     = ts_et.date
    return ohlc.reset_index()


# ═══════════════════════════════════════════════════════════════════════════
#  ADX computation
# ═══════════════════════════════════════════════════════════════════════════

def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = ADX_PERIOD) -> float:
    n = len(highs)
    if n < period + 1:
        return 0.0
    tr  = np.zeros(n); pdm = np.zeros(n); ndm = np.zeros(n)
    for i in range(1, n):
        hl  = highs[i] - lows[i]
        hpc = abs(highs[i]  - closes[i-1])
        lpc = abs(lows[i]   - closes[i-1])
        tr[i]  = max(hl, hpc, lpc)
        up   = highs[i]  - highs[i-1]
        down = lows[i-1] - lows[i]
        pdm[i] = up   if up > down and up > 0   else 0.0
        ndm[i] = down if down > up and down > 0 else 0.0
    atr = np.zeros(n); pdi = np.zeros(n); ndi = np.zeros(n)
    atr[period] = tr[1:period+1].sum()
    pdi[period] = pdm[1:period+1].sum()
    ndi[period] = ndm[1:period+1].sum()
    for i in range(period + 1, n):
        atr[i] = atr[i-1] - atr[i-1]/period + tr[i]
        pdi[i] = pdi[i-1] - pdi[i-1]/period + pdm[i]
        ndi[i] = ndi[i-1] - ndi[i-1]/period + ndm[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi_pct = np.where(atr > 0, pdi / atr * 100, 0.0)
        ndi_pct = np.where(atr > 0, ndi / atr * 100, 0.0)
        dsum    = pdi_pct + ndi_pct
        dx      = np.where(dsum > 0, np.abs(pdi_pct - ndi_pct) / dsum * 100, 0.0)
    dx_valid = dx[period:]
    if len(dx_valid) < period:
        return 0.0
    adx_val = dx_valid[:period].mean()
    for v in dx_valid[period:]:
        adx_val = (adx_val * (period - 1) + v) / period
    return float(adx_val)


# ═══════════════════════════════════════════════════════════════════════════
#  Pre-compute ADX per bar (both modes) — done once, swept across thresholds
# ═══════════════════════════════════════════════════════════════════════════

def precompute_adx_arrays(df5s: pd.DataFrame, df1m: pd.DataFrame) -> tuple:
    """
    Returns (daily_adx_arr, hourly_adx_arr) — one float per 5s bar.
    daily  : ADX from 9-10 ET on that day (same value all day).
    hourly : ADX on trailing 60 1-min bars at signal time.
    """
    n = len(df5s)
    dates = df5s["date"].values
    ts5s  = df5s["ts"].values

    # Daily ADX: one value per date
    daily_by_date = {}
    for date, grp in df1m.groupby("date"):
        w = grp[(grp["bar_time"] >= ADX_DAILY_START) &
                (grp["bar_time"] <  ADX_DAILY_END)]
        daily_by_date[date] = _adx(w["high"].values, w["low"].values,
                                   w["close"].values) if len(w) >= ADX_PERIOD + 1 else 0.0

    daily_adx_arr = np.array([daily_by_date.get(d, 0.0) for d in dates])

    # Hourly ADX: trailing 60 1-min bars per bar
    df1m_s  = df1m.sort_values("ts").reset_index(drop=True)
    ts1m    = df1m_s["ts"].values
    h1m     = df1m_s["high"].values.astype(float)
    l1m     = df1m_s["low"].values.astype(float)
    c1m     = df1m_s["close"].values.astype(float)

    print("  Pre-computing hourly ADX per bar …", flush=True)
    hourly_adx_arr = np.zeros(n)
    for i in range(n):
        idx   = int(np.searchsorted(ts1m, ts5s[i], side="right"))
        start = max(0, idx - ADX_HOURLY_BARS)
        if idx - start >= ADX_PERIOD + 1:
            hourly_adx_arr[i] = _adx(h1m[start:idx], l1m[start:idx], c1m[start:idx])

    return daily_adx_arr, hourly_adx_arr


# ═══════════════════════════════════════════════════════════════════════════
#  PL arrays
# ═══════════════════════════════════════════════════════════════════════════

def compute_pl_arrays(df: pd.DataFrame, window: int):
    s    = pd.Series(df["close"].values.astype(float), index=df.index)
    gaps = df["gap"].values.astype(bool)
    lr   = np.log(s / s.shift(1)).copy()
    lr.iloc[np.where(gaps)[0]] = np.nan
    abs_lr  = lr.abs()
    net_ret = lr.rolling(window, min_periods=window).sum()
    abs_sum = abs_lr.rolling(window, min_periods=window).sum()
    gap_in_window = abs_lr.isna().rolling(window, min_periods=1).max().astype(bool)
    net_ret[gap_in_window] = np.nan
    abs_sum[gap_in_window] = np.nan
    pl_arr  = (net_ret.abs() / abs_sum.replace(0, np.nan)).values
    dir_arr = np.sign(net_ret.values)
    closes  = df["close"].values.astype(float)
    lb_close = np.full(len(closes), np.nan)
    lb_close[window:] = closes[:len(closes) - window]
    move_arr = np.abs(closes - lb_close) / np.where(lb_close > 0, lb_close, 1) * 10000
    move_arr[gap_in_window.values] = np.nan
    return pl_arr, dir_arr, move_arr


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation (adx_arr already pre-computed; threshold passed in)
# ═══════════════════════════════════════════════════════════════════════════

def simulate(df5s: pd.DataFrame,
             pl_arr: np.ndarray, dir_arr: np.ndarray, move_arr: np.ndarray,
             adx_arr: np.ndarray, gate: float,
             point_value: float, min_gap_s: int = 0) -> dict:
    closes = df5s["close"].values.astype(float)
    highs  = df5s["high"].values.astype(float)
    lows   = df5s["low"].values.astype(float)
    gaps   = df5s["gap"].values.astype(bool)
    dates  = df5s["date"].values
    times  = df5s["bar_time"].values
    ts5s   = df5s["ts"].values  # numpy datetime64[ns] UTC
    n      = len(df5s)
    min_gap_ns = np.timedelta64(min_gap_s, "s")

    trades = []
    last_exit_ts = np.datetime64("NaT")
    i = 0
    while i < n:
        if not (times[i] >= BLACKOUT_END and times[i] < SESSION_END
                and not gaps[i]
                and not np.isnan(pl_arr[i]) and pl_arr[i] >= PL_THR
                and not np.isnan(move_arr[i]) and move_arr[i] >= MOVE_BPS):
            i += 1
            continue

        if adx_arr[i] > gate:
            i += 1
            continue

        if min_gap_s > 0 and last_exit_ts is not np.datetime64("NaT"):
            if ts5s[i] - last_exit_ts < min_gap_ns:
                i += 1
                continue

        direction = -dir_arr[i]
        entry     = closes[i]
        day_0     = dates[i]
        tp_pt     = entry * TP_BPS   / 10000
        stop_pt   = entry * STOP_BPS / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != day_0:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS); exit_i = j - 1; break
            if direction == 1 and highs[j] >= entry + tp_pt:
                pnl_bps = TP_BPS; exit_i = j; break
            if direction == -1 and lows[j] <= entry - tp_pt:
                pnl_bps = TP_BPS; exit_i = j; break
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if k >= MIN_HOLD:
                cur_pl = pl_arr[j]
                if not np.isnan(cur_pl) and cur_pl >= RESUME_PL:
                    raw = (closes[j] - entry) * direction / entry * 10000
                    pnl_bps = max(raw, -STOP_BPS); exit_i = j; break
        else:
            j = i + MAX_HOLD
            raw = (closes[min(j, n-1)] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -STOP_BPS); exit_i = min(j, n - 1)

        trades.append(dict(date=day_0, pnl_bps=pnl_bps))
        last_exit_ts = ts5s[exit_i]
        i = exit_i + 1

    if not trades:
        return dict(n=0, ev=0.0, wr=0.0, pnl=0.0)

    pnls      = np.array([t["pnl_bps"] for t in trades])
    avg_price = closes.mean()
    ev        = pnls.mean()
    wr        = (pnls > 0).mean()
    n_t       = len(trades)
    pnl_d     = ev * avg_price / 10000 * n_t * point_value
    return dict(n=n_t, ev=ev, wr=wr, pnl=pnl_d)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def run(symbol: str):
    cfg = SYMBOL_CONFIGS[symbol]
    print(f"\n=== PL_REV ADX Gate Sweep  ·  {symbol} ===")
    print(f"    Live params: PL≥{PL_THR}  mv≥{MOVE_BPS}bp  "
          f"TP={TP_BPS}bp  stop={STOP_BPS}bp\n")

    print("Loading 5s bars …", flush=True)
    df5s = load_5s(cfg["csv"])
    print(f"  {len(df5s):,} 5s RTH bars  "
          f"{df5s['date'].min()} → {df5s['date'].max()}")

    print("Resampling to 1-min bars …", flush=True)
    df1m = build_1min(df5s)
    print(f"  {len(df1m):,} 1-min bars")

    print("Computing PL arrays …", flush=True)
    pl_arr, dir_arr, move_arr = compute_pl_arrays(df5s, WINDOW)

    print("Pre-computing ADX arrays (this takes ~30s) …", flush=True)
    daily_adx_arr, hourly_adx_arr = precompute_adx_arrays(df5s, df1m)

    pv = cfg["point_value"]

    # ── Hourly sweep: gap=0 vs gap=120s side by side ─────────────────────────
    print(f"\n  Hourly ADX gate sweep — gap=0s (original) vs gap=120s (realistic)\n")
    hdr = (f"  {'Gate':>8}  "
           f"{'n(0s)':>6}  {'WR%(0s)':>8}  {'EV(0s)':>8}  {'P&L(0s)':>10}  "
           f"{'n(120s)':>7}  {'WR%(120s)':>9}  {'EV(120s)':>9}  {'P&L(120s)':>11}")
    print(hdr)
    print(f"  {'─'*105}")

    best_0, best_120 = None, None

    for gate in ADX_THRESHOLDS:
        gate_label = f"{gate:.0f}" if gate != float("inf") else "∞ (none)"
        r0   = simulate(df5s, pl_arr, dir_arr, move_arr, hourly_adx_arr, gate, pv, min_gap_s=0)
        r120 = simulate(df5s, pl_arr, dir_arr, move_arr, hourly_adx_arr, gate, pv, min_gap_s=120)

        def _fmt(r):
            if r["n"] == 0:
                return f"{'—':>6}  {'—':>8}  {'—':>8}  {'—':>10}"
            return (f"{r['n']:>6}  {r['wr']*100:>8.1f}  "
                    f"{r['ev']:>+8.2f}  {r['pnl']:>+10,.0f}")

        flag0   = " ◀" if best_0   is None or r0["ev"]   > best_0["ev"]   else ""
        flag120 = " ◀" if best_120 is None or r120["ev"] > best_120["ev"] else ""
        if r0["n"]   > 0 and (best_0   is None or r0["ev"]   > best_0["ev"]):   best_0   = r0
        if r120["n"] > 0 and (best_120 is None or r120["ev"] > best_120["ev"]): best_120 = r120

        print(f"  {gate_label:>8}  {_fmt(r0)}{flag0}  {_fmt(r120)}{flag120}")

    print(f"\n{'═'*105}")
    print("  Best EV per gap:")
    if best_0:
        print(f"    gap=  0s  n={best_0['n']:>5}  WR={best_0['wr']*100:.1f}%  "
              f"EV={best_0['ev']:>+.2f}bp  P&L=${best_0['pnl']:>+,.0f}")
    if best_120:
        print(f"    gap=120s  n={best_120['n']:>5}  WR={best_120['wr']*100:.1f}%  "
              f"EV={best_120['ev']:>+.2f}bp  P&L=${best_120['pnl']:>+,.0f}")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use: {', '.join(SYMBOL_CONFIGS)}")
        sys.exit(1)
    run(symbol)
