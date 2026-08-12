"""
backtest_pl_reversion.py — Mean-reversion flip of PL_MOM on 5s bars.

Uses identical qualifying gate as the live PL_MOM strategy (PL ≥ threshold,
move ≥ threshold, 30s window) but enters in the OPPOSITE direction —
betting that the directional move will revert rather than continue.

Rationale: if PL_MOM continuation has WR ≈ 35% (loses 65% of the time),
then fading the signal might have WR ≈ 65%.

Exit logic (mirrored from PL_MOM):
  - Take profit: price reverts TP_BPS from entry (opposite to signal direction)
  - Stop loss: STOP_BPS against entry
  - Max hold: MAX_HOLD bars (120s), same as PL_MOM
  - Min hold: MIN_HOLD bars (10s) before rolling-PL exit

Rolling-PL exit for reversion:
  Instead of exiting when PL drops (continuation weakens), exit when PL rises
  above RESUME_PL — meaning the market IS trending and our reversion bet is wrong.

Sweep params:
  pl_thr   : [0.60, 0.70, 0.80]   entry: PL ≥ pl_thr (same as PL_MOM gate)
  move_bps : [8, 12, 16, 20]      entry: move ≥ move_bps
  tp_bps   : [5, 8, 10, 12]       take profit on reversion
  stop_bps : [8, 12, 15]          stop on continuation

Usage:
  python src/backtest_pl_reversion.py          # MES
  python src/backtest_pl_reversion.py MNQ
"""

import sys
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

SYMBOL_CONFIGS = {
    "MES": dict(csv="es_hist_5sec.csv",  tick_size=0.25, point_value=5.0),
    "MNQ": dict(csv="nq_hist_5sec.csv",  tick_size=0.25, point_value=2.0),
}

SETTLEMENT_UTC = (21, 22)
RTH_START      = dtime(9, 30)
RTH_END        = dtime(16, 0)
BLACKOUT_END   = dtime(9, 40)
SESSION_END    = dtime(15, 55)

WINDOW    = 6    # 5s bars = 30s window (same as live PL_MOM)
MAX_HOLD  = 24   # bars = 120s
MIN_HOLD  = 2    # bars = 10s

RESUME_PL = 0.70  # exit reversion trade if PL surges back above this (trend resuming)

MIN_TRADES = 30

# Sweep params
PL_THRS   = [0.60, 0.70, 0.80]
MOVE_BPSS = [8, 12, 16, 20]
TP_BPSS   = [5, 8, 10, 12]
STOP_BPSS = [8, 12, 15]

# Exclude extreme-regime months (set empty to include all)
EXCLUDE_MONTHS: set[tuple[int,int]] = {(2025, 4)}  # exclude tariff crash


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
#  Simulation
# ═══════════════════════════════════════════════════════════════════════════

def _compute_pl_arrays(df: pd.DataFrame, window: int):
    """Vectorized PL, direction, move arrays using pandas rolling."""
    s = pd.Series(df["close"].values.astype(float), index=df.index)
    gaps = df["gap"].values.astype(bool)

    lr = np.log(s / s.shift(1)).copy()
    lr.iloc[np.where(gaps)[0]] = np.nan   # NaN at day boundaries / gaps

    abs_lr  = lr.abs()
    roll    = lr.rolling(window, min_periods=window)
    net_ret = roll.sum()
    abs_sum = abs_lr.rolling(window, min_periods=window).sum()

    # Zero out windows that span a gap
    gap_in_window = abs_lr.isna().rolling(window, min_periods=1).max().astype(bool)
    net_ret[gap_in_window] = np.nan
    abs_sum[gap_in_window] = np.nan

    pl_arr   = (net_ret.abs() / abs_sum.replace(0, np.nan)).values
    dir_arr  = np.sign(net_ret.values)   # +1 long, -1 short
    closes   = df["close"].values.astype(float)
    lb_close = np.full(len(closes), np.nan)
    lb_close[window:] = closes[:len(closes) - window]
    move_arr = np.abs(closes - lb_close) / np.where(lb_close > 0, lb_close, 1) * 10000
    move_arr[gap_in_window.values] = np.nan

    return pl_arr, dir_arr, move_arr


def run_config(df: pd.DataFrame, pl_arr: np.ndarray, dir_arr: np.ndarray,
               move_arr: np.ndarray, pl_thr: float, move_bps: float,
               tp_bps: float, stop_bps: float,
               point_value: float) -> dict | None:
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    gaps   = df["gap"].values.astype(bool)
    dates  = df["date"].values
    times  = df["bar_time"].values
    n      = len(df)

    trades = []
    i = 0
    while i < n:
        # Qualifying gate (same as PL_MOM)
        if not (times[i] >= BLACKOUT_END and times[i] < SESSION_END
                and not gaps[i]
                and not np.isnan(pl_arr[i]) and pl_arr[i] >= pl_thr
                and not np.isnan(move_arr[i]) and move_arr[i] >= move_bps):
            i += 1
            continue

        # FLIP: enter opposite to momentum direction
        signal_dir = dir_arr[i]
        direction  = -signal_dir   # fade the move
        entry      = closes[i]
        day_0      = dates[i]
        tp_pt      = entry * tp_bps   / 10000
        stop_pt    = entry * stop_bps / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != day_0:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -stop_bps); exit_i = j - 1; break

            # Take profit: price moves in reversion direction
            if direction == 1 and highs[j] >= entry + tp_pt:
                pnl_bps = tp_bps; exit_i = j; break
            if direction == -1 and lows[j] <= entry - tp_pt:
                pnl_bps = tp_bps; exit_i = j; break

            # Stop loss: price continues in original signal direction
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -stop_bps; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -stop_bps; exit_i = j; break

            # Rolling PL exit: if PL surges above RESUME_PL, trend is resuming
            if k >= MIN_HOLD:
                cur_pl = pl_arr[j]
                if not np.isnan(cur_pl) and cur_pl >= RESUME_PL:
                    raw = (closes[j] - entry) * direction / entry * 10000
                    pnl_bps = max(raw, -stop_bps); exit_i = j; break
        else:
            j = i + MAX_HOLD
            raw = (closes[min(j, n-1)] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -stop_bps); exit_i = min(j, n - 1)

        trades.append(dict(date=day_0, pnl_bps=pnl_bps, direction=direction,
                           entry=entry))
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
    print(f"\n=== PL_MOM Reversion Flip  ·  {symbol}  [excl {excl}] ===\n")
    print("Loading 5s bars …", flush=True)
    df = load_5s(cfg["csv"])
    print(f"  {len(df):,} 5s RTH bars  "
          f"{df['date'].min()} → {df['date'].max()}")

    best_results = []
    total_configs = len(PL_THRS) * len(MOVE_BPSS) * len(TP_BPSS) * len(STOP_BPSS)
    print(f"\n  Sweeping {total_configs} configs …\n")
    print(f"  {'pl_thr':>7}  {'move':>6}  {'tp':>5}  {'stop':>5}  "
          f"{'n':>6}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L $':>10}")
    print(f"  {'─'*65}")

    # Precompute PL arrays once (shared across all configs)
    pl_arr, dir_arr, move_arr = _compute_pl_arrays(df, WINDOW)

    for pl_thr in PL_THRS:
        for move_bps in MOVE_BPSS:
            for tp_bps in TP_BPSS:
                for stop_bps in STOP_BPSS:
                    if tp_bps >= stop_bps:
                        continue   # skip configs where TP ≥ stop (no edge by definition)
                    r = run_config(df, pl_arr, dir_arr, move_arr,
                                   pl_thr, move_bps, tp_bps, stop_bps,
                                   cfg["point_value"])
                    if r is None:
                        continue
                    flag = " ◀" if r["ev"] > 0 else ""
                    print(f"  {pl_thr:>7.2f}  {move_bps:>6}  {tp_bps:>5}  "
                          f"{stop_bps:>5}  {r['n']:>6}  {r['wr']*100:>6.1f}  "
                          f"{r['ev']:>+8.2f}  {r['pnl']:>+10,.0f}{flag}")
                    if r["ev"] > 0:
                        best_results.append((pl_thr, move_bps, tp_bps, stop_bps, r))

    # Monthly breakdown for top configs
    if best_results:
        best_results.sort(key=lambda x: x[4]["ev"], reverse=True)
        print(f"\n{'═'*70}")
        print(f"  Top configs EV > 0  —  monthly P&L  (P&L bars scaled by EV)")
        for pl_thr, move_bps, tp_bps, stop_bps, r in best_results[:5]:
            label = f"PL≥{pl_thr}  mv≥{move_bps}bp  TP={tp_bps}bp  stop={stop_bps}bp"
            print(f"\n  {label}  |  n={r['n']}  WR={r['wr']*100:.1f}%  "
                  f"EV={r['ev']:+.2f}bp  P&L=${r['pnl']:+,.0f}")
            print(f"  {'month':>9}  {'n':>5}  {'WR%':>6}  {'EV(bp)':>8}  chart")
            print(f"  {'─'*50}")
            for ym, row in r["monthly"].iterrows():
                n_m  = int(row["n"])
                ev_m = row["ev"]
                wr_m = row["wins"] / n_m * 100
                bar  = ("█" if ev_m >= 0 else "░") * min(int(abs(ev_m) * 4), 30)
                print(f"  {ym:>9}  {n_m:>5}  {wr_m:>6.1f}  {ev_m:>+8.2f}  {bar}")
    else:
        print("\n  No configs with EV > 0 found (excluding April 2025).")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use MES or MNQ.")
        sys.exit(1)
    run(symbol)
