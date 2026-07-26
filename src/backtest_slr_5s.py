"""
Backtest: SLR breakout-continuation on 15s and 20s bars, MES RTH only.

Resamples MES 5-second OHLCV bars to 15s and 20s, then runs a parameter
sweep over vol multiplier and move threshold.

Setup:
  - Surge bar: vol ≥ N× rolling median, close > open,
               move = open[prev bar] → close[surge] ≥ M bp  (WO2)
  - Entry: open of the bar immediately after the surge (no pullback wait)
  - Stop: 10bp (live-bot minimum)
  - Target: 15bp
  - Hold: up to HOLD_MAX bars (≈ 2 min)

RTH filter: 9:30–16:00 ET.
CME settlement gap (21:00–22:00 UTC) filtered before resampling.

Usage:
  python src/backtest_slr_5s.py               # both 15s and 20s
  python src/backtest_slr_5s.py --bar 15      # 15s only
  python src/backtest_slr_5s.py --csv mes_hist_5sec.csv
"""

import argparse
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET                   = ZoneInfo("America/New_York")
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

# Fixed strategy parameters
VOL_LOOKBACK = 40     # bars (≈10 min at 15s, ≈13 min at 20s)
TARGET_BPS   = 15.0
STOP_BPS     = 10.0
MIN_TRADES   = 10     # lower threshold — limited data at 5s resolution

# Sweep grid
VOL_MULTS = [5.0, 7.0, 9.0, 11.0]
MOVE_BPSS = [8.0, 10.0, 12.0, 14.0]

# Bar configs: bar_seconds → (hold_max bars, ~max hold minutes)
BAR_CONFIGS = {
    15: {"hold_max": 8,  "desc": "15s bars  hold ≤ 8 bars (2 min)"},
    20: {"hold_max": 6,  "desc": "20s bars  hold ≤ 6 bars (2 min)"},
}

# Enough hi/lo history for the largest hold window
_MAX_HOLD = max(c["hold_max"] for c in BAR_CONFIGS.values())


# ── Data loading ──────────────────────────────────────────────────────────────

def load_5s_bars(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # Remove CME settlement gap
    h  = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    # Filter to RTH: 9:30–16:00 ET
    ts_et = df["ts"].dt.tz_convert(ET)
    hm    = ts_et.dt.hour * 60 + ts_et.dt.minute
    df    = df[(hm >= 570) & (hm < 960)].copy()
    return df.reset_index(drop=True)


def resample_ohlcv(df5: pd.DataFrame, bar_seconds: int) -> pd.DataFrame:
    """Aggregate 5s RTH bars into bar_seconds-wide bars."""
    df = df5.set_index("ts")
    rs = df.resample(f"{bar_seconds}s").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",    "min"),
        close=("close",  "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])
    # Gap flag: consecutive bar timestamp gap > 1.5× expected interval
    expected = pd.Timedelta(seconds=bar_seconds)
    rs["gap"] = rs.index.to_series().diff() > expected * 1.5
    rs.iloc[0, rs.columns.get_loc("gap")] = True
    return rs.reset_index()


# ── Signal detection ──────────────────────────────────────────────────────────

def find_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scan all bars at minimum thresholds (vol≥min(VOL_MULTS)×, move≥min(MOVE_BPSS)).
    Store vol_ratio and move_bps so the sweep can filter without re-scanning.
    Also store hi_arr/lo_arr paths for the largest hold window.
    """
    c   = df["close"].values
    o   = df["open"].values
    hi  = df["high"].values
    lo  = df["low"].values
    vol = df["volume"].values.astype(float)
    gap = df["gap"].values.astype(bool)
    nb  = len(df)

    vol_s = pd.Series(np.where(gap, np.nan, vol))
    med   = vol_s.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().values

    vol_min  = min(VOL_MULTS)
    move_min = min(MOVE_BPSS)
    records  = []

    for i in range(VOL_LOOKBACK + 1, nb - _MAX_HOLD - 2):
        if np.isnan(med[i]) or med[i] == 0:
            continue
        vol_ratio = vol[i] / med[i]
        if vol_ratio < vol_min:
            continue
        if gap[i] or gap[i - 1]:
            continue
        if c[i] <= o[i]:          # must be bullish
            continue

        # WO2 move: open of previous bar → close of surge bar
        move_bps = (c[i] - o[i - 1]) / c[i] * 10000
        if move_bps < move_min:
            continue

        entry_bar = i + 1
        if entry_bar + _MAX_HOLD >= nb:
            continue
        if gap[entry_bar]:        # gap right after surge — skip
            continue

        entry = o[entry_bar]

        # Pre-slice price path for hold window (hi/lo starting one bar after entry)
        hi_path = hi[entry_bar + 1: entry_bar + 1 + _MAX_HOLD].copy()
        lo_path = lo[entry_bar + 1: entry_bar + 1 + _MAX_HOLD].copy()

        if len(hi_path) < _MAX_HOLD:
            continue

        te = c[entry_bar + _MAX_HOLD] - entry   # time-expiry PnL at max hold

        records.append({
            "surge_bar": i,
            "entry_bar": entry_bar,
            "entry":     entry,
            "vol_ratio": vol_ratio,
            "move_bps":  move_bps,
            "hi_path":   hi_path,
            "lo_path":   lo_path,
            "te_pts":    te,
        })

    return pd.DataFrame(records)


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(cands: pd.DataFrame, hold_max: int) -> pd.DataFrame:
    """
    Simulate each candidate with given hold_max.
    Dedup: skip any trade whose entry_bar falls within the hold window of a
    previous trade (same logic as 1-min backtest).
    """
    if cands.empty:
        return pd.DataFrame()

    hold_until = -1
    records    = []

    for _, row in cands.sort_values("entry_bar").iterrows():
        eb = int(row["entry_bar"])
        if eb <= hold_until:
            continue

        entry      = row["entry"]
        target_pts = entry * TARGET_BPS / 10000
        sl_pts     = entry * STOP_BPS   / 10000
        hi_path    = row["hi_path"][:hold_max]
        lo_path    = row["lo_path"][:hold_max]

        hit_tgt = hit_stop = None
        for k in range(hold_max):
            if hit_stop is None and (entry - lo_path[k]) >= sl_pts:
                hit_stop = k + 1
            if hit_tgt  is None and (hi_path[k] - entry) >= target_pts:
                hit_tgt  = k + 1

        if hit_tgt is not None and (hit_stop is None or hit_tgt <= hit_stop):
            pnl_pts   = target_pts
            hold_bars = hit_tgt
        elif hit_stop is not None:
            pnl_pts   = -sl_pts
            hold_bars = hit_stop
        else:
            pnl_pts   = row["te_pts"]
            hold_bars = hold_max

        hold_until = eb + hold_max
        records.append({
            "entry_bar": eb,
            "entry":     entry,
            "vol_ratio": row["vol_ratio"],
            "move_bps":  row["move_bps"],
            "pnl_pts":   pnl_pts,
            "hold_bars": int(hold_bars),
        })

    return pd.DataFrame(records)


# ── Output ────────────────────────────────────────────────────────────────────

def print_sweep(trades: pd.DataFrame, hold_max: int, bar_seconds: int):
    sec_per_bar = bar_seconds / 60.0   # bars → minutes

    print(f"\n  {'vol →':8}", end="")
    for vm in VOL_MULTS:
        print(f"  {f'≥{vm:.0f}×':^22}", end="")
    print()
    print(f"  {'move ↓':8}", end="")
    for _ in VOL_MULTS:
        print(f"  {'EV(bp) / n / hold':^22}", end="")
    print()
    print("  " + "─" * (8 + len(VOL_MULTS) * 24))

    for mb in MOVE_BPSS:
        print(f"  {f'≥{mb:.0f}bp':8}", end="")
        for vm in VOL_MULTS:
            sub = trades[(trades["vol_ratio"] >= vm) & (trades["move_bps"] >= mb)]
            n   = len(sub)
            if n < MIN_TRADES:
                print(f"  {'—':^22}", end="")
            else:
                avg_entry = sub["entry"].mean()
                ev_bp     = sub["pnl_pts"].mean() / avg_entry * 10000
                avg_hold  = sub["hold_bars"].mean() * sec_per_bar
                cell = f"{ev_bp:+.2f}bp / {n}n / {avg_hold:.1f}m"
                print(f"  {cell:^22}", end="")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(csv_path: str, bar_sizes: list[int]):
    print(f"Loading {csv_path} …")
    df5 = load_5s_bars(csv_path)
    print(f"  {len(df5):,} 5s RTH bars  "
          f"({df5['ts'].min().date()} → {df5['ts'].max().date()})")

    for bar_seconds in bar_sizes:
        cfg      = BAR_CONFIGS[bar_seconds]
        hold_max = cfg["hold_max"]

        print(f"\n{'═'*80}")
        print(f"  MES  {cfg['desc']}")
        print(f"  Immediate entry: open of bar after surge  LONG only  RTH only")
        print(f"  Stop={STOP_BPS:.0f}bp  Target={TARGET_BPS:.0f}bp  "
              f"Vol lookback={VOL_LOOKBACK} bars "
              f"({VOL_LOOKBACK * bar_seconds // 60} min {VOL_LOOKBACK * bar_seconds % 60}s)")
        print(f"  Move = open[prev bar] → close[surge bar]  (WO2)")
        print(f"  Cell: EV(bp) / n trades / avg hold(min)   (blank = <{MIN_TRADES} trades)")
        print(f"{'═'*80}")

        df = resample_ohlcv(df5.copy(), bar_seconds)
        print(f"  {len(df):,} {bar_seconds}s bars after resampling")

        cands = find_candidates(df)
        print(f"  {len(cands):,} surge candidates at vol≥{min(VOL_MULTS):.0f}×, "
              f"move≥{min(MOVE_BPSS):.0f}bp")

        if cands.empty:
            print("  No candidates found.")
            continue

        trades = simulate(cands, hold_max)
        print(f"  {len(trades):,} trades after hold-window dedup")

        if trades.empty:
            print("  No trades to report.")
            continue

        avg_p = trades["entry"].mean()
        ev_all = trades["pnl_pts"].mean() / avg_p * 10000
        print(f"  Overall EV (all thresholds): {ev_all:+.3f} bp  "
              f"avg hold: {trades['hold_bars'].mean() * bar_seconds / 60:.2f} min")

        print_sweep(trades, hold_max, bar_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SLR breakout-continuation backtest on 15s/20s MES bars")
    parser.add_argument("--csv", default="mes_hist_5sec.csv",
                        help="Path to 5s OHLCV CSV (default: mes_hist_5sec.csv)")
    parser.add_argument("--bar", type=int, choices=[15, 20],
                        help="Single bar size to test (default: both 15 and 20)")
    args = parser.parse_args()

    bar_sizes = [args.bar] if args.bar else [15, 20]
    run(args.csv, bar_sizes)
