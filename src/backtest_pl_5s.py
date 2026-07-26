"""
Backtest: Price Linearity (PL) momentum continuation on 5-second bars.

Two modes:
  RAW    — PL computed on raw log returns; move threshold in basis points
  SCALED — PL computed on volatility-normalised log returns (each return divided
           by a rolling std); move threshold in σ units

PL = |sum(returns)| / sum(|returns|)  in [0, 1]
  -> 1.0 = perfectly trending; 0.0 = perfectly choppy

Scaling rationale: a 6bp drift in a quiet regime is more meaningful than 6bp
during a volatile open.  Dividing each log return by rolling σ makes PL
regime-adaptive — a high PL on scaled returns means the move was directional
*relative to prevailing volatility*.

Strategy:
  - Compute rolling PL over W bars (raw or scaled)
  - Entry: PL >= entry_pl AND net move >= threshold
    Direction: LONG if net > 0, SHORT otherwise
  - Exit: PL drops to <= 0.40, OR stop (10bp), OR max hold (120s)
  - Min hold 10s before PL exit is checked (stop always active)

Sweeps:
  bar_size      : 5, 10, 15, 20 seconds (--bar-size)
  window        : 6, 8 bars
  entry_pl      : 0.60, 0.70, 0.80
  RAW   move    : 0, 4, 8, 12, 16, 20, 24 bp
  SCALED move   : 0, 0.5, 1.0, 1.5, 2.0 σ
  std_lookback  : 20, 40, 80 bars

Usage:
  python src/backtest_pl_5s.py                              # ES, 5s bars, raw mode
  python src/backtest_pl_5s.py --csv mnq_hist_5sec.csv --label MNQ
  python src/backtest_pl_5s.py --bar-size 5 10 15 20       # sweep all bar sizes
  python src/backtest_pl_5s.py --mode raw                  # raw only
"""

import argparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET                   = ZoneInfo("America/New_York")
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

STOP_BPS      = 10.0
EXIT_PL       = 0.40
MAX_HOLD_S    = 120    # seconds (constant regardless of bar size)
MIN_HOLD_S    = 10     # seconds before PL exit allowed
MIN_TRADES    = 10

WINDOWS       = [6, 8]                       # bars
ENTRY_PLS     = [0.60, 0.70, 0.80]
MOVE_BPSS     = [0, 4, 8, 12, 16, 20, 24]   # raw mode: bp
MOVE_SIGMAS   = [0, 0.5, 1.0, 1.5, 2.0]     # scaled mode: σ units
STD_LOOKBACKS = [20, 40, 80]                 # bars for rolling σ


def load_5s_rth(csv_path):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    h  = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    ts_et = df["ts"].dt.tz_convert(ET)
    hm    = ts_et.dt.hour * 60 + ts_et.dt.minute
    df    = df[(hm >= 570) & (hm < 960)].copy()
    df["gap"] = (df["ts"].diff() > pd.Timedelta(seconds=10)).values
    df.iloc[0, df.columns.get_loc("gap")] = True
    return df.reset_index(drop=True)


def resample_to_ns(df_5s, bar_size_s):
    """Resample 5s bars to bar_size_s-second bars via OHLCV aggregation."""
    if bar_size_s == 5:
        return df_5s.copy()
    df = df_5s.set_index("ts")
    agg = df.resample(f"{bar_size_s}s", closed="left", label="left").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"]).reset_index()
    # Gap: time diff > 2× bar size (e.g. missing bar)
    agg["gap"] = (agg["ts"].diff() > pd.Timedelta(seconds=bar_size_s * 2)).values
    agg.iloc[0, agg.columns.get_loc("gap")] = True
    return agg.reset_index(drop=True)


def compute_raw_lr(df):
    """Raw log returns array, NaN at gaps."""
    close = df["close"].values
    gap   = df["gap"].values.astype(bool)
    n     = len(df)
    lr    = np.empty(n); lr[:] = np.nan
    lr[1:] = np.log(close[1:] / close[:-1])
    lr[gap] = np.nan
    return lr


def compute_indicators_raw(df, window):
    """
    PL on raw log returns.
    Returns (pl_arr, dir_arr, move_arr) where move_arr is in basis points.
    """
    close = df["close"].values
    gap   = df["gap"].values.astype(bool)
    n     = len(df)
    lr    = compute_raw_lr(df)

    pl_arr   = np.full(n, np.nan)
    dir_arr  = np.zeros(n)
    move_arr = np.full(n, np.nan)   # bp

    for i in range(window, n):
        if gap[i - window + 1: i + 1].any():
            continue
        rets = lr[i - window + 1: i + 1]
        if np.isnan(rets).any():
            continue
        sum_abs = np.abs(rets).sum()
        if sum_abs == 0:
            continue
        net = rets.sum()
        pl_arr[i]  = abs(net) / sum_abs
        dir_arr[i] = 1.0 if net > 0 else -1.0
        move_arr[i] = abs(close[i] - close[i - window]) / close[i] * 10000

    return pl_arr, dir_arr, move_arr


def compute_indicators_scaled(df, window, std_lookback):
    """
    PL on volatility-scaled log returns: lr_scaled[i] = lr[i] / rolling_std(lr, std_lookback).
    Returns (pl_arr, dir_arr, move_arr) where move_arr is in σ units
    (net scaled return over window).
    """
    gap  = df["gap"].values.astype(bool)
    n    = len(df)
    lr   = compute_raw_lr(df)

    # Rolling std (gap-aware: treat gap bars as NaN so they don't contaminate σ)
    lr_s     = pd.Series(np.where(gap, np.nan, lr))
    roll_std = lr_s.rolling(std_lookback, min_periods=std_lookback).std().values

    pl_arr   = np.full(n, np.nan)
    dir_arr  = np.zeros(n)
    move_arr = np.full(n, np.nan)   # σ units

    min_i = max(window, std_lookback)
    for i in range(min_i, n):
        if gap[i - window + 1: i + 1].any():
            continue
        if np.isnan(roll_std[i]) or roll_std[i] == 0:
            continue
        rets = lr[i - window + 1: i + 1]
        if np.isnan(rets).any():
            continue
        scaled = rets / roll_std[i]
        sum_abs = np.abs(scaled).sum()
        if sum_abs == 0:
            continue
        net_scaled = scaled.sum()
        pl_arr[i]  = abs(net_scaled) / sum_abs
        dir_arr[i] = 1.0 if net_scaled > 0 else -1.0
        move_arr[i] = abs(net_scaled)   # σ units

    return pl_arr, dir_arr, move_arr


def simulate(df, pl_arr, dir_arr, move_arr, entry_pl, min_move, bar_size_s):
    """
    Simulate entries where PL >= entry_pl and move >= min_move.
    move_arr units match min_move (bp for raw, σ for scaled).
    """
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    gap   = df["gap"].values.astype(bool)
    n     = len(df)

    max_hold = MAX_HOLD_S // bar_size_s
    min_hold = max(1, MIN_HOLD_S // bar_size_s)

    hold_until = -1
    records    = []

    for i in range(n - max_hold - 2):
        if i <= hold_until:
            continue
        if np.isnan(pl_arr[i]) or pl_arr[i] < entry_pl:
            continue
        if np.isnan(move_arr[i]) or move_arr[i] < min_move:
            continue
        if gap[i]:
            continue
        direction = dir_arr[i]
        if direction == 0:
            continue

        eb = i + 1
        if eb >= n - max_hold - 1 or gap[eb]:
            continue

        entry  = close[eb]
        sl_pts = entry * STOP_BPS / 10000

        exit_bar = None
        exit_pnl = None

        for k in range(1, max_hold + 1):
            j = eb + k
            if j >= n:
                break
            if direction == 1 and low[j] <= entry - sl_pts:
                exit_bar = j; exit_pnl = -sl_pts; break
            if direction == -1 and high[j] >= entry + sl_pts:
                exit_bar = j; exit_pnl = -sl_pts; break
            if k >= min_hold and not np.isnan(pl_arr[j]) and pl_arr[j] <= EXIT_PL:
                exit_bar = j
                exit_pnl = direction * (close[j] - entry)
                break

        if exit_bar is None:
            exit_bar = eb + max_hold
            if exit_bar < n:
                exit_pnl = direction * (close[exit_bar] - entry)
            else:
                continue

        hold_until = exit_bar
        records.append({
            "direction": direction,
            "pnl_bp":    exit_pnl / entry * 10000,
            "hold_s":    (exit_bar - eb) * bar_size_s,
        })

    return pd.DataFrame(records)


def fmt_cell(t, min_t):
    n = len(t)
    if n < min_t:
        return "--"
    ev   = t["pnl_bp"].mean()
    hold = t["hold_s"].mean()
    winp = (t["pnl_bp"] > 0).mean() * 100
    return f"{ev:+.2f}bp/{n}n/{hold:.0f}s/{winp:.0f}%"


def print_grid(label, w, ep, indicators, thresholds, thresh_labels, mode_tag, bar_size_s):
    col_w = 28
    print(f"\n  {label}  [{mode_tag}]  bar={bar_size_s}s  window={w*bar_size_s}s  PL>={ep:.2f}")
    print(f"  Cell: EV(bp) / n / hold(s) / win%   (-- = <{MIN_TRADES})")
    print(f"  {'threshold':16}", end="")
    for ww in WINDOWS:
        print(f"  {f'window={ww*bar_size_s}s':^{col_w}}", end="")
    print()
    print("  " + "-" * (16 + len(WINDOWS) * (col_w + 2)))

    for thr, tlbl in zip(thresholds, thresh_labels):
        print(f"  {tlbl:16}", end="")
        for ww in WINDOWS:
            pl_arr, dir_arr, move_arr = indicators[ww]
            t = simulate(df_global, pl_arr, dir_arr, move_arr, ep, thr, bar_size_s)
            print(f"  {fmt_cell(t, MIN_TRADES):^{col_w}}", end="")
        print()

    # Direction split at middle threshold, first window
    mid_thr = thresholds[len(thresholds) // 2]
    pl_arr, dir_arr, move_arr = indicators[WINDOWS[0]]
    t_dir = simulate(df_global, pl_arr, dir_arr, move_arr, ep, mid_thr, bar_size_s)
    if not t_dir.empty and len(t_dir) >= MIN_TRADES:
        longs  = t_dir[t_dir["direction"] ==  1]
        shorts = t_dir[t_dir["direction"] == -1]
        mid_lbl = thresh_labels[len(thresh_labels) // 2]
        print(f"  Direction at {mid_lbl} ({WINDOWS[0]*bar_size_s}s window):", end="")
        if len(longs)  >= MIN_TRADES: print(f"  LONG {len(longs)}n EV={longs['pnl_bp'].mean():+.2f}bp win={100*(longs['pnl_bp']>0).mean():.0f}%", end="")
        if len(shorts) >= MIN_TRADES: print(f"  |  SHORT {len(shorts)}n EV={shorts['pnl_bp'].mean():+.2f}bp win={100*(shorts['pnl_bp']>0).mean():.0f}%", end="")
        print()


# Module-level df reference used by print_grid (avoids passing everywhere)
df_global = None


# ── Exit parameter sweep ───────────────────────────────────────────────────────

EXIT_PLS_SWEEP    = [0.20, 0.30, 0.40, 0.50, 0.60]
STOP_BPS_SWEEP    = [6.0, 8.0, 10.0, 12.0, 15.0]
MAX_HOLD_S_SWEEP  = [60, 90, 120, 180]

# Fixed entry params per instrument for exit sweep
EXIT_SWEEP_CONFIGS = {
    "MES": {"entry_pl": 0.80, "move_bps": 8.0},
    "MNQ": {"entry_pl": 0.70, "move_bps": 12.0},
}


def simulate_exit(df, pl_arr, dir_arr, move_arr,
                  entry_pl, min_move,
                  exit_pl, stop_bps, max_hold_s, bar_size_s=5):
    """Like simulate() but with explicit exit params rather than globals."""
    close = df["close"].values
    high  = df["high"].values
    low   = df["low"].values
    gap   = df["gap"].values.astype(bool)
    n     = len(df)

    max_hold = max(1, max_hold_s // bar_size_s)
    min_hold = max(1, MIN_HOLD_S // bar_size_s)

    hold_until = -1
    records    = []

    for i in range(n - max_hold - 2):
        if i <= hold_until:
            continue
        if np.isnan(pl_arr[i]) or pl_arr[i] < entry_pl:
            continue
        if np.isnan(move_arr[i]) or move_arr[i] < min_move:
            continue
        if gap[i]:
            continue
        direction = dir_arr[i]
        if direction == 0:
            continue

        eb = i + 1
        if eb >= n - max_hold - 1 or gap[eb]:
            continue

        entry  = close[eb]
        sl_pts = entry * stop_bps / 10000

        exit_bar = None
        exit_pnl = None

        for k in range(1, max_hold + 1):
            j = eb + k
            if j >= n:
                break
            if direction == 1 and low[j] <= entry - sl_pts:
                exit_bar = j; exit_pnl = -sl_pts; break
            if direction == -1 and high[j] >= entry + sl_pts:
                exit_bar = j; exit_pnl = -sl_pts; break
            if k >= min_hold and not np.isnan(pl_arr[j]) and pl_arr[j] <= exit_pl:
                exit_bar = j
                exit_pnl = direction * (close[j] - entry)
                break

        if exit_bar is None:
            exit_bar = eb + max_hold
            if exit_bar < n:
                exit_pnl = direction * (close[exit_bar] - entry)
            else:
                continue

        hold_until = exit_bar
        records.append({
            "direction": direction,
            "pnl_bp":    exit_pnl / entry * 10000,
            "hold_s":    (exit_bar - eb) * bar_size_s,
        })

    return pd.DataFrame(records)


def run_exit_sweep(csv_path, label):
    global df_global
    cfg = EXIT_SWEEP_CONFIGS.get(label)
    if cfg is None:
        print(f"  No exit sweep config for {label} — add to EXIT_SWEEP_CONFIGS")
        return

    print(f"\nLoading {csv_path} ...")
    df_5s = load_5s_rth(csv_path)
    df    = resample_to_ns(df_5s, 5)
    df_global = df
    print(f"  {len(df):,} 5s RTH bars  ({df['ts'].min().date()} -> {df['ts'].max().date()})")

    entry_pl = cfg["entry_pl"]
    move_bps = cfg["move_bps"]
    pl_arr, dir_arr, move_arr = compute_indicators_raw(df, window=6)

    print(f"\n{'#'*90}")
    print(f"  {label}  EXIT SWEEP  |  entry: PL>={entry_pl:.2f}  move>={move_bps:.0f}bp  window=30s")
    print(f"  Cell: EV(bp) / n / hold(s) / win%   (-- = <{MIN_TRADES})")
    print(f"{'#'*90}")

    # ── Table 1: exit_pl vs stop_bps (max_hold fixed at current 120s) ─────────
    col_w = 26
    print(f"\n  Max hold fixed at {MAX_HOLD_S}s")
    print(f"  {'stop \\ exit_pl':18}", end="")
    for ep in EXIT_PLS_SWEEP:
        print(f"  {f'exit<={ep:.2f}':^{col_w}}", end="")
    print()
    print("  " + "-" * (18 + len(EXIT_PLS_SWEEP) * (col_w + 2)))
    for stop in STOP_BPS_SWEEP:
        marker = " ◄" if stop == STOP_BPS else ""
        print(f"  {f'stop={stop:.0f}bp{marker}':18}", end="")
        for ep in EXIT_PLS_SWEEP:
            marker2 = "◄" if ep == EXIT_PL and stop == STOP_BPS else ""
            t = simulate_exit(df, pl_arr, dir_arr, move_arr,
                              entry_pl, move_bps, ep, stop, MAX_HOLD_S)
            cell = fmt_cell(t, MIN_TRADES)
            print(f"  {cell+marker2:^{col_w}}", end="")
        print()

    # ── Table 2: exit_pl vs max_hold_s (stop fixed at current 10bp) ───────────
    print(f"\n  Stop fixed at {STOP_BPS:.0f}bp")
    print(f"  {'max_hold \\ exit_pl':20}", end="")
    for ep in EXIT_PLS_SWEEP:
        print(f"  {f'exit<={ep:.2f}':^{col_w}}", end="")
    print()
    print("  " + "-" * (20 + len(EXIT_PLS_SWEEP) * (col_w + 2)))
    for mh in MAX_HOLD_S_SWEEP:
        marker = " ◄" if mh == MAX_HOLD_S else ""
        print(f"  {f'max={mh}s{marker}':20}", end="")
        for ep in EXIT_PLS_SWEEP:
            t = simulate_exit(df, pl_arr, dir_arr, move_arr,
                              entry_pl, move_bps, ep, STOP_BPS, mh)
            cell = fmt_cell(t, MIN_TRADES)
            print(f"  {cell:^{col_w}}", end="")
        print()


def run(csv_path, label, mode, bar_sizes):
    global df_global
    print(f"Loading {csv_path} ...")
    df_5s = load_5s_rth(csv_path)
    print(f"  {len(df_5s):,} 5s RTH bars  ({df_5s['ts'].min().date()} -> {df_5s['ts'].max().date()})")

    for bar_size_s in bar_sizes:
        df = resample_to_ns(df_5s, bar_size_s)
        df_global = df
        n_bars = len(df)
        print(f"\n{'#'*90}")
        print(f"  BAR SIZE = {bar_size_s}s  →  {n_bars:,} bars  "
              f"(window 6 bars = {6*bar_size_s}s, max hold {MAX_HOLD_S}s)")
        print(f"{'#'*90}")

        if mode in ("raw", "both"):
            raw_ind = {w: compute_indicators_raw(df, w) for w in WINDOWS}
            for ep in ENTRY_PLS:
                print(f"\n{'='*90}")
                print(f"  {label}  RAW PL  |  bar={bar_size_s}s  PL>={ep:.2f}  "
                      f"Exit<={EXIT_PL}  Stop {STOP_BPS:.0f}bp  "
                      f"Min hold {MIN_HOLD_S}s  Max hold {MAX_HOLD_S}s")
                print(f"  Move threshold in basis points")
                print(f"{'='*90}")
                thresh_labels = [f"move>={m}bp" if m > 0 else "no filter" for m in MOVE_BPSS]
                print_grid(label, WINDOWS[0], ep, raw_ind,
                           MOVE_BPSS, thresh_labels, "RAW", bar_size_s)

        if mode in ("scaled", "both"):
            for std_lb in STD_LOOKBACKS:
                scaled_ind = {w: compute_indicators_scaled(df, w, std_lb) for w in WINDOWS}
                for ep in ENTRY_PLS:
                    print(f"\n{'='*90}")
                    print(f"  {label}  SCALED PL  |  bar={bar_size_s}s  PL>={ep:.2f}  "
                          f"Exit<={EXIT_PL}  Stop {STOP_BPS:.0f}bp  "
                          f"Min hold {MIN_HOLD_S}s  Max hold {MAX_HOLD_S}s")
                    print(f"  σ lookback={std_lb} bars ({std_lb*bar_size_s}s)  "
                          f"|  Move threshold in σ units")
                    print(f"{'='*90}")
                    thresh_labels = [f"move>={m}σ" if m > 0 else "no filter" for m in MOVE_SIGMAS]
                    print_grid(label, WINDOWS[0], ep, scaled_ind,
                               MOVE_SIGMAS, thresh_labels,
                               f"SCALED σlb={std_lb*bar_size_s}s", bar_size_s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",        default="es_hist_5sec.csv")
    parser.add_argument("--label",      default="ES")
    parser.add_argument("--mode",       default="raw", choices=["raw", "scaled", "both"])
    parser.add_argument("--bar-size",   dest="bar_sizes", type=int, nargs="+", default=[5],
                        help="Bar size(s) in seconds to sweep, e.g. --bar-size 5 10 15 20")
    parser.add_argument("--exit-sweep", action="store_true",
                        help="Sweep exit params (exit_pl, stop_bps, max_hold) at fixed entry params")
    args = parser.parse_args()
    if args.exit_sweep:
        run_exit_sweep(args.csv, args.label)
    else:
        run(args.csv, args.label, args.mode, args.bar_sizes)
