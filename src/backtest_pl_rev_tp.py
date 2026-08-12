"""
backtest_pl_rev_tp.py — Find optimal take-profit for PL_REV.

Sweeps TP from 10 to 80bp. All other params fixed at live values.
Model: no ADX gate, 120s minimum gap between trades.

Shows where WR degrades and EV/P&L peaks.
Monthly breakdown for selected TP values to check consistency.

Usage:
  python src/backtest_pl_rev_tp.py          # MES (default)
  python src/backtest_pl_rev_tp.py MNQ
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
WINDOW         = 6
MAX_HOLD       = 24   # bars = 120s
MIN_HOLD       = 2
MIN_GAP_S      = 120

# Fixed live params
PL_THR    = 0.80
MOVE_BPS  = 20.0
STOP_BPS  = 15.0
RESUME_PL = 0.70

# TP sweep range
TP_VALUES = [10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80]

# Monthly detail for these TP values
TP_DETAIL = [12, 18, 24, 32, 40]

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
#  PL arrays
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
#  Simulation — returns trade list for monthly breakdown
# ═══════════════════════════════════════════════════════════════════════════

def simulate(df: pd.DataFrame,
             pl_arr: np.ndarray, dir_arr: np.ndarray, move_arr: np.ndarray,
             tp_bps: float, point_value: float) -> tuple[dict, list]:
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
                and not np.isnan(pl_arr[i]) and pl_arr[i] >= PL_THR
                and not np.isnan(move_arr[i]) and move_arr[i] >= MOVE_BPS):
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
        stop_pt   = entry * STOP_BPS / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != day_0:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS); exit_i = j - 1; break
            if direction == 1 and highs[j] >= entry + tp_pt:
                pnl_bps = tp_bps; exit_i = j; break
            if direction == -1 and lows[j] <= entry - tp_pt:
                pnl_bps = tp_bps; exit_i = j; break
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if k >= MIN_HOLD:
                if not np.isnan(pl_arr[j]) and pl_arr[j] >= RESUME_PL:
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
        return dict(n=0, ev=0.0, wr=0.0, pnl=0.0), []

    pnls  = np.array([t["pnl_bps"] for t in trades])
    ev    = pnls.mean()
    wr    = (pnls > 0).mean()
    n_t   = len(trades)
    pnl_d = ev * closes.mean() / 10000 * n_t * point_value

    # win rate breakdown: TP hit vs max-hold/resume exit
    tp_hits   = sum(1 for t in trades if abs(t["pnl_bps"] - tp_bps) < 0.01)
    tp_hit_pct = tp_hits / n_t * 100

    return dict(n=n_t, ev=ev, wr=wr, pnl=pnl_d, tp_hit_pct=tp_hit_pct), trades


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def run(symbol: str):
    cfg = SYMBOL_CONFIGS[symbol]
    pv  = cfg["point_value"]

    print(f"\n=== PL_REV Take-Profit Sweep  ·  {symbol}  "
          f"(no ADX gate, {MIN_GAP_S}s gap) ===")
    print(f"    Fixed: PL≥{PL_THR}  mv≥{MOVE_BPS}bp  "
          f"stop={STOP_BPS}bp  resume≥{RESUME_PL}\n")

    print("Loading 5s bars …", flush=True)
    df = load_5s(cfg["csv"])
    print(f"  {len(df):,} bars  {df['date'].min()} → {df['date'].max()}")

    print("Computing PL arrays …", flush=True)
    pl_arr, dir_arr, move_arr = compute_pl_arrays(df)

    # ── TP sweep ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  {'TP(bp)':>7}  {'n':>5}  {'WR%':>6}  {'TP-hit%':>8}  "
          f"{'EV(bp)':>8}  {'P&L $':>10}")
    print(f"  {'─'*70}")

    all_results = {}
    best_pnl = None
    for tp in TP_VALUES:
        r, trades = simulate(df, pl_arr, dir_arr, move_arr, tp, pv)
        all_results[tp] = (r, trades)
        if r["n"] == 0:
            print(f"  {tp:>7}  {'—':>5}")
            continue
        live_mark = " ◀ live" if tp == 12 else ""
        if best_pnl is None or r["pnl"] > best_pnl["pnl"]:
            best_pnl = r | {"tp": tp}
        print(f"  {tp:>7}  {r['n']:>5}  {r['wr']*100:>6.1f}  "
              f"{r['tp_hit_pct']:>8.1f}  {r['ev']:>+8.2f}  "
              f"{r['pnl']:>+10,.0f}{live_mark}")

    # ── Monthly detail for selected TP values ─────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Monthly EV (bp) — selected TP values\n")

    detail_results = {tp: all_results[tp] for tp in TP_DETAIL if tp in all_results}
    all_months = sorted(set(
        f"{t['date'].year}-{t['date'].month:02d}"
        for _, (_, trades) in detail_results.items()
        for t in trades
    ))

    tp_labels = [f"TP={tp:>2}" for tp in TP_DETAIL]
    header = f"  {'Month':<9}" + "".join(f"  {lbl:>9}" for lbl in tp_labels)
    print(header)
    print(f"  {'─'*60}")

    for ym in all_months:
        row = f"  {ym:<9}"
        for tp in TP_DETAIL:
            if tp not in detail_results:
                row += f"  {'—':>9}"; continue
            _, trades = detail_results[tp]
            month_trades = [t["pnl_bps"] for t in trades
                            if f"{t['date'].year}-{t['date'].month:02d}" == ym]
            if month_trades:
                ev_m = np.mean(month_trades)
                row += f"  {ev_m:>+9.2f}"
            else:
                row += f"  {'—':>9}"
        print(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    if best_pnl:
        print(f"  Best total P&L: TP={best_pnl['tp']}bp  "
              f"n={best_pnl['n']}  WR={best_pnl['wr']*100:.1f}%  "
              f"EV={best_pnl['ev']:+.2f}bp  P&L=${best_pnl['pnl']:+,.0f}")
    print(f"\n  Note: TP-hit% = fraction of trades that hit the target.")
    print(f"  Low TP-hit% at high TP means most exits are time/resume exits,")
    print(f"  not actual target hits — EV gains may be less reliable.")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use: {', '.join(SYMBOL_CONFIGS)}")
        sys.exit(1)
    run(symbol)
