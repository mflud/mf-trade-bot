"""
Backtest: SLR_Scalp — parameter sweep over vol multiplier and move threshold.

Sweeps:
  VOL_MULT  : 5× 6× 7× 8× 9× 10×
  MOVE_BPS  : 8  10  12  14  16  18 bp

Move is measured open[surge-1] → close[surge]  (WO2: 2-bar window).
Stop = 10bp (live-bot minimum).  Target = 15bp.  Hold ≤ 15 bars.

For each cell shows: EV(bp) / n trades / avg hold(min)

Usage:
  python src/backtest_slr_sweep.py --csv mes_hist_1min.csv --label MES
  python src/backtest_slr_sweep.py --all        # all four instruments
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET                   = ZoneInfo("America/New_York")
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

VOL_LOOKBACK     = 20
PULLBACK_MIN_BPS = 1.5
PULLBACK_MAX_BPS = 4.5
PULLBACK_BARS    = 3
TARGET_BPS       = 15.0
HOLD_MAX         = 15
STOP_BPS         = 10.0
MIN_TRADES       = 20     # lower threshold so sparse cells still show

# Sweep grid
VOL_MULTS  = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
MOVE_BPSS  = [8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

# Minimum thresholds for initial candidate scan
VOL_MIN  = min(VOL_MULTS)
MOVE_MIN = min(MOVE_BPSS)

ALL_INSTRUMENTS = [
    ("MES", "mes_hist_1min.csv"),
    ("MNQ", "mnq_hist_1min.csv"),
    ("ES",  "es_hist_1min.csv"),
    ("NQ",  "nq_hist_1min.csv"),
]

POINT_VALUE = {"MES": 5.0, "MNQ": 2.0, "ES": 50.0, "NQ": 20.0}


def load_bars(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")
    df = df.sort_values("ts").reset_index(drop=True)
    h  = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    df = df.reset_index(drop=True)
    ts_et    = df["ts"].dt.tz_convert(ET)
    hm       = ts_et.dt.hour.values * 60 + ts_et.dt.minute.values
    df["is_rth"] = (hm >= 570) & (hm < 960)
    df["year"]   = df["ts"].dt.year.values
    df["gap"]    = (df["ts"].diff() > pd.Timedelta(minutes=2)).values
    df.loc[0, "gap"] = True
    return df


def find_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Find all candidates at minimum thresholds; store vol_ratio and move_bps
    so the sweep can filter without re-scanning bars."""
    c      = df["close"].values
    o      = df["open"].values
    hi     = df["high"].values
    lo     = df["low"].values
    vol    = df["volume"].values.astype(float)
    gap    = df["gap"].values.astype(bool)
    is_rth = df["is_rth"].values.astype(bool)
    year   = df["year"].values
    nb     = len(df)

    vol_s   = pd.Series(np.where(gap, np.nan, vol))
    med20   = vol_s.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().values
    gap_cum = np.cumsum(gap.astype(int))

    records = []
    for i in range(VOL_LOOKBACK + 2, nb - HOLD_MAX - PULLBACK_BARS - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        vol_ratio = vol[i] / med20[i]
        if vol_ratio < VOL_MIN:
            continue
        if gap[i] or gap[i - 1]:
            continue
        if c[i] < o[i]:
            continue

        move     = c[i] - o[i - 1]   # open[prev] → close[surge]  (WO2)
        move_bps = move / c[i] * 10000
        if move_bps < MOVE_MIN:
            continue

        extreme = hi[i]
        pb_min  = extreme * PULLBACK_MIN_BPS / 10000
        pb_max  = extreme * PULLBACK_MAX_BPS / 10000

        pullback_bar = None
        for j in range(i + 1, min(i + 1 + PULLBACK_BARS, nb - HOLD_MAX - 2)):
            if gap[j]:
                break
            retrace = extreme - c[j]
            if pb_min <= retrace <= pb_max:
                pullback_bar = j
                break

        if pullback_bar is None:
            continue

        entry_bar = pullback_bar + 1
        if entry_bar + HOLD_MAX >= nb:
            continue
        if gap_cum[entry_bar + HOLD_MAX] > gap_cum[i]:
            continue
        if c[pullback_bar] <= o[i - 1]:
            continue

        entry   = o[entry_bar]
        path_hi = np.array([hi[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        path_lo = np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])

        records.append({
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    bool(is_rth[entry_bar]),
            "year":      year[entry_bar],
            "vol_ratio": vol_ratio,
            "move_bps":  move_bps,
            "path_hi":   path_hi,
            "path_lo":   path_lo,
            "te_pts":    c[entry_bar + HOLD_MAX] - entry,
        })

    return pd.DataFrame(records)


def simulate(cands: pd.DataFrame) -> pd.DataFrame:
    """Simulate all candidates; dedup by hold window. Returns per-trade results."""
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
        path_hi    = row["path_hi"]
        path_lo    = row["path_lo"]

        hit_tgt = hit_stop = None
        for k in range(HOLD_MAX):
            if hit_stop is None and (entry - path_lo[k]) >= sl_pts:
                hit_stop = k + 1
            if hit_tgt  is None and (path_hi[k] - entry) >= target_pts:
                hit_tgt  = k + 1

        # Hold time in minutes (bars = minutes for 1-min data)
        if hit_tgt is not None and (hit_stop is None or hit_tgt <= hit_stop):
            hold_bars = hit_tgt
            pnl_pts   = target_pts
        elif hit_stop is not None:
            hold_bars = hit_stop
            pnl_pts   = -sl_pts
        else:
            hold_bars = HOLD_MAX
            pnl_pts   = row["te_pts"]

        hold_until = eb + HOLD_MAX

        records.append({
            "entry_bar": eb,
            "entry":     entry,
            "is_rth":    row["is_rth"],
            "year":      row["year"],
            "vol_ratio": row["vol_ratio"],
            "move_bps":  row["move_bps"],
            "pnl_pts":   pnl_pts,
            "hold_min":  float(hold_bars),
            "target_pts": target_pts,
            "sl_pts":    sl_pts,
        })

    return pd.DataFrame(records)


def cell_stats(trades: pd.DataFrame) -> dict | None:
    n = len(trades)
    if n < MIN_TRADES:
        return None
    avg_price = trades["entry"].mean()
    ev_bp     = trades["pnl_pts"].mean() / avg_price * 10000
    avg_hold  = trades["hold_min"].mean()
    return {"ev_bp": ev_bp, "n": n, "avg_hold": avg_hold}


def print_grid(label: str, trades: pd.DataFrame, session_filter):
    if session_filter is not None:
        trades = trades[trades["is_rth"] == session_filter]

    print(f"\n  {'vol →':8}", end="")
    for vm in VOL_MULTS:
        print(f"  {f'≥{vm:.0f}×':^20}", end="")
    print()
    print(f"  {'move ↓':8}", end="")
    for _ in VOL_MULTS:
        print(f"  {'EV(bp) / n / hold':^20}", end="")
    print()
    print("  " + "─" * (8 + len(VOL_MULTS) * 22))

    for mb in MOVE_BPSS:
        print(f"  {f'≥{mb:.0f}bp':8}", end="")
        for vm in VOL_MULTS:
            sub = trades[(trades["vol_ratio"] >= vm) & (trades["move_bps"] >= mb)]
            st  = cell_stats(sub)
            if st is None:
                print(f"  {'—':^20}", end="")
            else:
                cell = f"{st['ev_bp']:+.2f}bp/{st['n']}n/{st['avg_hold']:.1f}m"
                print(f"  {cell:^20}", end="")
        print()


def run_one(label: str, csv_path: str):
    print(f"\n{'═'*80}")
    print(f"  {label}  sweep: vol {VOL_MULTS[0]:.0f}–{VOL_MULTS[-1]:.0f}×  "
          f"move {MOVE_BPSS[0]:.0f}–{MOVE_BPSS[-1]:.0f}bp")
    print(f"  Move = open[prev bar] → close[surge bar]  (WO2 2-bar window)")
    print(f"  Stop={STOP_BPS:.0f}bp  Target={TARGET_BPS:.0f}bp  Hold≤{HOLD_MAX}min  LONG only")
    print(f"  Cell: EV(bp) / n trades / avg hold(min)   (blank = <{MIN_TRADES} trades)")
    print(f"{'═'*80}")

    df = load_bars(csv_path)
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    cands = find_candidates(df)
    print(f"  {len(cands):,} candidates at minimum thresholds "
          f"(vol≥{VOL_MIN:.0f}×, move≥{MOVE_MIN:.0f}bp)")

    if cands.empty:
        print("  No candidates found.")
        return

    trades = simulate(cands)

    for sess_label, sess_filter in [("ALL SESSIONS", None),
                                     ("RTH ONLY",     True),
                                     ("GLOBEX ONLY",  False)]:
        print(f"\n  ── {sess_label} ──")
        print_grid(label, trades, sess_filter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   type=str, help="Path to 1-min OHLCV CSV")
    parser.add_argument("--label", type=str, help="Instrument label")
    parser.add_argument("--all",   action="store_true",
                        help="Run all instruments (MES, MNQ, ES, NQ)")
    args = parser.parse_args()

    if args.all:
        for label, csv_path in ALL_INSTRUMENTS:
            try:
                run_one(label, csv_path)
            except FileNotFoundError:
                print(f"\n  {label}: {csv_path} not found — skipping")
    elif args.csv:
        run_one(args.label or args.csv, args.csv)
    else:
        parser.print_help()
