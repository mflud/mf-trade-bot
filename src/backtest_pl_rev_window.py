"""
backtest_pl_rev_window.py — PL_REV window-length sweep.

Sweeps window lengths (6/12/18/24 bars = 30s/60s/90s/120s) to find the
optimal lookback for the PL_REV mean-reversion strategy.

PL normalization: raw PL is scaled by sqrt(window/6) before comparing to
the threshold, because longer windows have more opportunity for noise to
reduce linearity — a 90s PL of 0.70 is "as linear" as a 30s PL of 0.85.

Entry gate (same as live PL_REV):
  - PL_scaled ≥ pl_thr  (scaled_pl = raw_pl * sqrt(window/6))
  - |move| ≥ move_bps
  - FADE: enter opposite to the momentum direction
  - COOLDOWN_BARS between trades (no immediate re-entry)

Exit logic:
  - Take profit: price reverts TP_BPS from entry
  - Stop loss: STOP_BPS against entry
  - Max hold: MAX_HOLD_MULT * window bars (keeps hold proportional to window)
  - Rolling PL exit (after MIN_HOLD bars): if scaled PL surges above
    pl_thr + RESUME_MARGIN (trend resuming) — set above entry threshold to
    avoid immediate exit (RESUME_PL must be > pl_thr)

Usage:
  python src/backtest_pl_rev_window.py          # MES
  python src/backtest_pl_rev_window.py MNQ
"""

import math
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

SETTLEMENT_UTC = (21, 22)
RTH_START      = dtime(9, 30)
RTH_END        = dtime(16, 0)
BLACKOUT_END   = dtime(9, 40)
SESSION_END    = dtime(15, 55)

# Hold duration scales with window so all windows get a "fair" exit window
MAX_HOLD_MULT  = 4    # window=6 → 24 bars (120s); window=24 → 96 bars (480s)
MIN_HOLD       = 2    # bars before rolling-PL exit check

# RESUME_PL is set to pl_thr + RESUME_MARGIN so it's always above the entry
# threshold.  Without this, the exit fires immediately because the 6-bar
# window barely shifts between bar i and bar i+2.
RESUME_MARGIN  = 0.10   # resume_pl = pl_thr + 0.10, capped at 0.98

# Post-trade cooldown: skip this many bars after each exit before re-entering.
# Prevents TP=5bp < move=8bp from cycling a new trade on every 5s bar.
COOLDOWN_BARS  = 60    # 5 minutes of 5s bars

MIN_TRADES     = 20   # minimum trades required to report a config

# Sweep params
WINDOWS   = [6, 12, 18, 24]   # bars (×5s = 30s/60s/90s/120s)
PL_THRS   = [0.70, 0.75, 0.80, 0.85, 0.90]
MOVE_BPSS = [8, 12, 16, 20, 25]
TP_BPSS   = [5, 8, 10, 12]
STOP_BPSS = [8, 12, 15]

# Exclude extreme-regime months
EXCLUDE_MONTHS: set[tuple[int, int]] = {(2025, 4)}


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
#  PL array computation
# ═══════════════════════════════════════════════════════════════════════════

def _compute_pl_arrays(df: pd.DataFrame, window: int):
    """
    Returns (pl_scaled, dir_arr, move_arr) where:
      pl_scaled = raw_pl * sqrt(window/6)   — normalized to 6-bar equivalent
    """
    s    = pd.Series(df["close"].values.astype(float), index=df.index)
    gaps = df["gap"].values.astype(bool)

    lr      = np.log(s / s.shift(1)).copy()
    lr.iloc[np.where(gaps)[0]] = np.nan

    abs_lr  = lr.abs()
    net_ret = lr.rolling(window, min_periods=window).sum()
    abs_sum = abs_lr.rolling(window, min_periods=window).sum()

    # Zero out windows that span a gap
    gap_in_window = abs_lr.isna().rolling(window, min_periods=1).max().astype(bool)
    net_ret[gap_in_window] = np.nan
    abs_sum[gap_in_window] = np.nan

    raw_pl    = (net_ret.abs() / abs_sum.replace(0, np.nan)).values
    scale     = math.sqrt(window / 6)
    pl_scaled = raw_pl * scale          # normalize so thresholds are window-agnostic

    dir_arr   = np.sign(net_ret.values)
    closes    = df["close"].values.astype(float)
    lb_close  = np.full(len(closes), np.nan)
    lb_close[window:] = closes[:len(closes) - window]
    move_arr  = np.abs(closes - lb_close) / np.where(lb_close > 0, lb_close, 1) * 10000
    move_arr[gap_in_window.values] = np.nan

    return pl_scaled, dir_arr, move_arr


# ═══════════════════════════════════════════════════════════════════════════
#  Simulation
# ═══════════════════════════════════════════════════════════════════════════

def run_config(df: pd.DataFrame, pl_arr: np.ndarray, dir_arr: np.ndarray,
               move_arr: np.ndarray, window: int,
               pl_thr: float, move_bps: float,
               tp_bps: float, stop_bps: float,
               point_value: float) -> dict | None:
    closes   = df["close"].values.astype(float)
    highs    = df["high"].values.astype(float)
    lows     = df["low"].values.astype(float)
    gaps     = df["gap"].values.astype(bool)
    dates    = df["date"].values
    times    = df["bar_time"].values
    n        = len(df)
    max_hold = MAX_HOLD_MULT * window
    resume_pl = min(pl_thr + RESUME_MARGIN, 0.98)

    trades     = []
    cool_until = 0   # bar index before which we won't enter (post-trade cooldown)
    i = 0
    while i < n:
        if i < cool_until:
            i += 1
            continue
        if not (times[i] >= BLACKOUT_END and times[i] < SESSION_END
                and not gaps[i]
                and not np.isnan(pl_arr[i]) and pl_arr[i] >= pl_thr
                and not np.isnan(move_arr[i]) and move_arr[i] >= move_bps):
            i += 1
            continue

        # FADE: enter opposite to momentum direction
        direction = -dir_arr[i]
        entry     = closes[i]
        day_0     = dates[i]
        tp_pt     = entry * tp_bps   / 10000
        stop_pt   = entry * stop_bps / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, max_hold + 1):
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
                cur_pl = pl_arr[j]
                if not np.isnan(cur_pl) and cur_pl >= resume_pl:
                    raw = (closes[j] - entry) * direction / entry * 10000
                    pnl_bps = max(raw, -stop_bps); exit_i = j; break
        else:
            j = i + max_hold
            raw = (closes[min(j, n-1)] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -stop_bps); exit_i = min(j, n - 1)

        trades.append(dict(date=day_0, pnl_bps=pnl_bps, direction=direction,
                           entry=entry))
        cool_until = exit_i + 1 + COOLDOWN_BARS
        i = exit_i + 1

    if len(trades) < MIN_TRADES:
        return None

    pnls      = np.array([t["pnl_bps"] for t in trades])
    avg_price = np.mean([t["entry"] for t in trades])
    ev        = pnls.mean()
    wr        = (pnls > 0).mean()
    n_t       = len(trades)

    tr_df  = pd.DataFrame(trades)
    n_days = df["date"].nunique()   # total trading days in dataset (incl. zero-trade days)
    tpd    = n_t / n_days if n_days else 0

    pts   = ev * avg_price / 10000
    pnl_d = pts * point_value * n_t

    tr_df["ym"] = tr_df["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
    monthly = tr_df.groupby("ym")["pnl_bps"].agg(
        n="count", ev="mean", wins=lambda x: (x > 0).sum()
    )

    return dict(n=n_t, ev=ev, wr=wr, tpd=tpd, pts=pts, pnl=pnl_d,
                monthly=monthly, avg_price=avg_price)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def run(symbol: str):
    cfg  = SYMBOL_CONFIGS[symbol]
    excl = ", ".join(f"{y}-{m:02d}" for y, m in sorted(EXCLUDE_MONTHS)) if EXCLUDE_MONTHS else "none"
    print(f"\n=== PL_REV Window Sweep  ·  {symbol}  [excl {excl}] ===")
    print(f"    PL normalized by sqrt(window/6)  |  cooldown={COOLDOWN_BARS} bars ({COOLDOWN_BARS*5}s)")
    print(f"    RESUME_MARGIN=+{RESUME_MARGIN:.2f} above pl_thr\n")
    print("Loading 5s bars …", flush=True)
    df = load_5s(cfg["csv"])
    print(f"  {len(df):,} 5s RTH bars  "
          f"{df['date'].min()} → {df['date'].max()}\n")

    all_best: list[tuple] = []

    for window in WINDOWS:
        secs  = window * 5
        scale = math.sqrt(window / 6)
        print(f"\n{'═'*72}")
        print(f"  WINDOW = {window} bars ({secs}s)  |  PL scale = ×{scale:.3f}  "
              f"|  max_hold = {MAX_HOLD_MULT*window} bars ({MAX_HOLD_MULT*window*5}s)")
        print(f"{'═'*72}")
        print(f"  {'pl_thr':>7}  {'move':>6}  {'tp':>5}  {'stop':>5}  "
              f"{'n':>6}  {'t/day':>6}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L $':>10}")
        print(f"  {'─'*68}")

        pl_arr, dir_arr, move_arr = _compute_pl_arrays(df, window)

        found_any = False
        for pl_thr in PL_THRS:
            for move_bps in MOVE_BPSS:
                for tp_bps in TP_BPSS:
                    for stop_bps in STOP_BPSS:
                        if tp_bps >= stop_bps:
                            continue
                        r = run_config(df, pl_arr, dir_arr, move_arr, window,
                                       pl_thr, move_bps, tp_bps, stop_bps,
                                       cfg["point_value"])
                        if r is None:
                            continue
                        flag = " ◀" if r["ev"] > 0 else ""
                        print(f"  {pl_thr:>7.2f}  {move_bps:>6}  {tp_bps:>5}  "
                              f"{stop_bps:>5}  {r['n']:>6}  {r['tpd']:>6.2f}  "
                              f"{r['wr']*100:>6.1f}  {r['ev']:>+8.2f}  "
                              f"{r['pnl']:>+10,.0f}{flag}")
                        if r["ev"] > 0:
                            all_best.append((window, pl_thr, move_bps, tp_bps, stop_bps, r))
                            found_any = True
        if not found_any:
            print(f"  (no configs with n≥{MIN_TRADES} and EV>0)")

    # ── Cross-window summary ───────────────────────────────────────────────
    if all_best:
        all_best.sort(key=lambda x: x[5]["ev"], reverse=True)
        print(f"\n\n{'═'*72}")
        print(f"  TOP CONFIGS BY EV  (all windows, EV > 0)")
        print(f"{'═'*72}")
        print(f"  {'win':>4}  {'pl_thr':>7}  {'move':>6}  {'tp':>5}  {'stop':>5}  "
              f"{'n':>6}  {'t/day':>6}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L $':>10}")
        print(f"  {'─'*75}")
        for window, pl_thr, move_bps, tp_bps, stop_bps, r in all_best[:20]:
            print(f"  {window:>4}  {pl_thr:>7.2f}  {move_bps:>6}  {tp_bps:>5}  "
                  f"{stop_bps:>5}  {r['n']:>6}  {r['tpd']:>6.2f}  "
                  f"{r['wr']*100:>6.1f}  {r['ev']:>+8.2f}  {r['pnl']:>+10,.0f}")

        # ── Monthly breakdown for top 5 ────────────────────────────────────
        print(f"\n{'═'*72}")
        print(f"  MONTHLY BREAKDOWN  (top 5 configs by EV)")
        for window, pl_thr, move_bps, tp_bps, stop_bps, r in all_best[:5]:
            secs  = window * 5
            label = (f"win={window}({secs}s)  PL≥{pl_thr}  mv≥{move_bps}bp  "
                     f"TP={tp_bps}bp  stop={stop_bps}bp")
            print(f"\n  {label}")
            print(f"  n={r['n']}  t/day={r['tpd']:.2f}  WR={r['wr']*100:.1f}%  "
                  f"EV={r['ev']:+.2f}bp  P&L=${r['pnl']:+,.0f}")
            print(f"  {'month':>9}  {'n':>5}  {'WR%':>6}  {'EV(bp)':>8}  chart")
            print(f"  {'─'*52}")
            for ym, row in r["monthly"].iterrows():
                n_m  = int(row["n"])
                ev_m = row["ev"]
                wr_m = row["wins"] / n_m * 100 if n_m else 0
                bar  = ("█" if ev_m >= 0 else "░") * min(int(abs(ev_m) * 4), 30)
                print(f"  {ym:>9}  {n_m:>5}  {wr_m:>6.1f}  {ev_m:>+8.2f}  {bar}")
    else:
        print(f"\n  No configs with EV > 0 and n≥{MIN_TRADES} across any window.")


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use MES or MNQ.")
        sys.exit(1)
    run(symbol)
