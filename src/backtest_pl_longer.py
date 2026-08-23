"""
backtest_pl_longer.py — PL_MOM at longer timescales (5-min and 15-min bars).

Tests whether price-linearity momentum continuation has edge when the
qualifying window is 15–60 minutes rather than 30 seconds (5s bars).

Strategy:
  - Compute rolling PL over WINDOW bars of bar_size-min bars
  - Entry: PL ≥ pl_thr AND net move ≥ move_thr_bps
    Direction: LONG if net > 0, SHORT otherwise
  - Exit: PL drops to ≤ EXIT_PL, OR stop (STOP_BPS), OR max hold (MAX_HOLD bars)
  - Min hold MIN_HOLD bars before PL exit allowed
  - No new entries in first BLACKOUT_END minutes or last SESSION_END minutes of RTH

Sweeps over:
  bar_size : [5, 15] minutes
  window   : [3, 6] bars  → 15/30/45/90 min lookback
  pl_thr   : [0.60, 0.70, 0.80]
  move_bps : [10, 20, 30, 50]

Usage:
  python src/backtest_pl_longer.py          # MES
  python src/backtest_pl_longer.py MNQ
"""

import sys
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

SYMBOL_CONFIGS = {
    "MES": dict(csv_5s="es_hist_5sec.csv",  csv_1m="mes_hist_1min.csv",
                tick_size=0.25, point_value=5.0),
    "MNQ": dict(csv_5s="nq_hist_5sec.csv",  csv_1m="mnq_hist_1min.csv",
                tick_size=0.25, point_value=2.0),
}

SETTLEMENT_UTC = (21, 22)
RTH_START      = dtime(9, 30)
RTH_END        = dtime(16, 0)
BLACKOUT_END   = dtime(9, 40)   # no entries in first 10 min of RTH
SESSION_END    = dtime(15, 45)  # no new entries within 15 min of close

EXIT_PL   = 0.40
STOP_BPS  = 15.0   # wider stop for longer bars
MIN_TRADES = 30

# Sweep params
BAR_SIZES = [5, 15]          # minutes
WINDOWS   = [3, 6]           # bars
PL_THRS   = [0.60, 0.70, 0.80]
MOVE_BPSS = [10, 20, 30, 50]

# Exclude extreme-regime months so results reflect normal-market edge
EXCLUDE_MONTHS: set[tuple[int,int]] = {(2025, 4)}

MAX_HOLD_BARS = {5: 12, 15: 4}   # bar_size_min → max hold (60 min / bar_size)
MIN_HOLD_BARS = {5: 2,  15: 1}   # bar_size_min → min hold before PL exit


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
    # RTH only
    df = df[(df["bar_time"] >= RTH_START) & (df["bar_time"] < RTH_END)].copy()
    if EXCLUDE_MONTHS:
        df = df[df["date"].apply(lambda d: (d.year, d.month) not in EXCLUDE_MONTHS)]
    return df.reset_index(drop=True)


def resample_to_min(df_5s: pd.DataFrame, bar_size_min: int) -> pd.DataFrame:
    """Resample RTH 5s bars to bar_size_min-minute OHLCV bars."""
    out = (df_5s.set_index("ts")
           .resample(f"{bar_size_min}min", closed="left", label="left")
           .agg(open=("open","first"), high=("high","max"),
                low=("low","min"),   close=("close","last"),
                volume=("volume","sum"))
           .dropna(subset=["open"])
           .reset_index())
    out["ts_et"]    = out["ts"].dt.tz_convert(ET)
    out["bar_time"] = out["ts_et"].dt.time
    out["date"]     = out["ts_et"].dt.date
    # Keep only RTH bars after blackout
    out = out[(out["bar_time"] >= BLACKOUT_END) &
              (out["bar_time"] < RTH_END)].copy()
    # Gap flag: diff > 2× bar size
    out["gap"] = (out["ts"].diff() > pd.Timedelta(minutes=bar_size_min * 2)).values
    out.iloc[0, out.columns.get_loc("gap")] = True
    return out.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PL + simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_config(df: pd.DataFrame, window: int, pl_thr: float,
               move_bps: float, bar_size_min: int,
               point_value: float) -> dict | None:
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    gaps   = df["gap"].values.astype(bool)
    dates  = df["date"].values
    times  = df["bar_time"].values
    n      = len(df)

    max_hold = MAX_HOLD_BARS[bar_size_min]
    min_hold = MIN_HOLD_BARS[bar_size_min]

    # Log returns (NaN at gaps)
    lr = np.empty(n); lr[:] = np.nan
    lr[1:] = np.log(closes[1:] / closes[:-1])
    lr[gaps] = np.nan

    # PL, direction, move
    pl_arr   = np.full(n, np.nan)
    dir_arr  = np.zeros(n)
    move_arr = np.full(n, np.nan)

    for i in range(window, n):
        if gaps[i - window + 1: i + 1].any():
            continue
        rets = lr[i - window + 1: i + 1]
        if np.isnan(rets).any():
            continue
        sa = np.abs(rets).sum()
        if sa == 0:
            continue
        net = rets.sum()
        pl_arr[i]   = abs(net) / sa
        dir_arr[i]  = 1.0 if net > 0 else -1.0
        move_arr[i] = abs(closes[i] - closes[i - window]) / closes[i] * 10000

    # Sequential simulation with re-entry
    trades = []
    i = 0
    while i < n:
        # Qualifying gate
        if not (times[i] < SESSION_END
                and not gaps[i]
                and not np.isnan(pl_arr[i]) and pl_arr[i] >= pl_thr
                and not np.isnan(move_arr[i]) and move_arr[i] >= move_bps):
            i += 1
            continue

        direction = dir_arr[i]
        entry     = closes[i]
        day_0     = dates[i]
        stop_pt   = entry * STOP_BPS / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, max_hold + 1):
            j = i + k
            if j >= n or dates[j] != day_0:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS); exit_i = j - 1; break
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            cur_pl = pl_arr[j]
            if k >= min_hold and not np.isnan(cur_pl) and cur_pl <= EXIT_PL:
                raw = (closes[j] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS); exit_i = j; break
        else:
            j = i + max_hold
            raw = (closes[min(j, n-1)] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -STOP_BPS); exit_i = min(j, n - 1)

        trades.append(dict(date=day_0, pnl_bps=pnl_bps,
                           direction=direction, entry=entry))
        i = exit_i + 1

    if len(trades) < MIN_TRADES:
        return None

    pnls = np.array([t["pnl_bps"] for t in trades])
    avg_price = np.mean([t["entry"] for t in trades])
    ev    = pnls.mean()
    wr    = (pnls > 0).mean()
    n_t   = len(trades)
    pts   = ev * avg_price / 10000 * n_t
    pnl_d = pts * point_value

    # Monthly breakdown
    tr_df = pd.DataFrame(trades)
    tr_df["ym"] = tr_df["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
    monthly = tr_df.groupby("ym")["pnl_bps"].agg(
        n="count", ev="mean", wins=lambda x: (x>0).sum()
    )

    return dict(n=n_t, ev=ev, wr=wr, pts=pts, pnl=pnl_d,
                monthly=monthly, avg_price=avg_price)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def run(symbol: str):
    cfg = SYMBOL_CONFIGS[symbol]
    excl = ", ".join(f"{y}-{m:02d}" for y, m in sorted(EXCLUDE_MONTHS)) if EXCLUDE_MONTHS else "none"
    print(f"\n=== PL_MOM Longer Window  ·  {symbol}  [excl {excl}] ===\n")

    print("Loading 5s bars and resampling …", flush=True)
    df_5s = load_5s(cfg["csv_5s"])
    print(f"  {len(df_5s):,} 5s RTH bars  "
          f"{df_5s['date'].min()} → {df_5s['date'].max()}")

    best_results = []

    for bar_size in BAR_SIZES:
        df = resample_to_min(df_5s, bar_size)
        window_mins = {w: w * bar_size for w in WINDOWS}
        print(f"\n{'═'*70}")
        print(f"  Bar size: {bar_size}-min   "
              f"Windows: {[f'{w}×{bar_size}={w*bar_size}min' for w in WINDOWS]}")
        print(f"  {'window':>8}  {'pl_thr':>7}  {'move':>6}  "
              f"{'n':>6}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L $':>10}")
        print(f"  {'─'*65}")

        for window in WINDOWS:
            for pl_thr in PL_THRS:
                for move_bps in MOVE_BPSS:
                    r = run_config(df, window, pl_thr, move_bps,
                                   bar_size, cfg["point_value"])
                    if r is None:
                        continue
                    label = f"{window}×{bar_size}={window*bar_size}min"
                    flag = " ◀" if r["ev"] > 0 else ""
                    print(f"  {label:>8}  {pl_thr:>7.2f}  {move_bps:>6}  "
                          f"{r['n']:>6}  {r['wr']*100:>6.1f}  "
                          f"{r['ev']:>+8.2f}  {r['pnl']:>+10,.0f}{flag}")
                    if r["ev"] > 0:
                        best_results.append((bar_size, window, pl_thr, move_bps, r))

    # Monthly breakdown for best configs
    if best_results:
        # Sort by EV
        best_results.sort(key=lambda x: x[4]["ev"], reverse=True)
        print(f"\n{'═'*70}")
        print(f"  Top configs with EV > 0  —  monthly P&L breakdown")
        for bar_size, window, pl_thr, move_bps, r in best_results[:5]:
            label = f"{window}×{bar_size}min  PL≥{pl_thr}  mv≥{move_bps}bp"
            print(f"\n  {label}  |  n={r['n']}  WR={r['wr']*100:.1f}%  "
                  f"EV={r['ev']:+.2f}bp  P&L=${r['pnl']:+,.0f}")
            print(f"  {'month':>9}  {'n':>5}  {'WR%':>6}  {'EV(bp)':>8}  chart")
            print(f"  {'─'*50}")
            for ym, row in r["monthly"].iterrows():
                n_m  = int(row["n"])
                ev_m = row["ev"]
                wr_m = row["wins"] / n_m * 100
                pnl_m = ev_m * r["avg_price"] / 10000 * n_m * r["pnl"] / max(abs(r["pnl"]), 1)
                bar = ("█" if ev_m >= 0 else "░") * min(int(abs(ev_m) * 5), 30)
                print(f"  {ym:>9}  {n_m:>5}  {wr_m:>6.1f}  {ev_m:>+8.2f}  {bar}")
    else:
        print("\n  No configs with EV > 0 found (excluding April 2025).")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use MES or MNQ.")
        sys.exit(1)
    run(symbol)
