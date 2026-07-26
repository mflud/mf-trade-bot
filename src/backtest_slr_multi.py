"""
Backtest: SLR_Scalp — generalised, runs against any 1-min OHLCV CSV.

WO2 setup (surge → pullback → entry), LONG only.
Stop ≥ 10bp (live-bot minimum).  Target = 15bp.  Hold ≤ 15 bars.

Stop variants:
  BP 10bp  : entry × 10bp / 10000          (live bot default)
  BP 12bp  : entry × 12bp / 10000
  BP 15bp  : entry × 15bp / 10000          (equal to target — 1:1 R:R)

Usage:
  python src/backtest_slr_multi.py --csv mes_hist_1min.csv --label MES
  python src/backtest_slr_multi.py --csv es_hist_1min.csv  --label ES
  python src/backtest_slr_multi.py --csv nq_hist_1min.csv  --label NQ
  python src/backtest_slr_multi.py --csv mnq_hist_1min.csv --label MNQ

  # Compare all at once (runs each in turn):
  python src/backtest_slr_multi.py --all
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
VOL_MULT         = 7.0
MOVE_BPS         = 12.0
STOP_MIN_BPS     = 10.0    # live-bot minimum — stops tighter than this trigger too often
MIN_TRADES       = 30

STOPS = [
    ("BP 10bp", 10.0),
    ("BP 12bp", 12.0),
    ("BP 15bp", 15.0),
]

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
    df["is_rth"] = (hm >= 570) & (hm < 960)   # 9:30–16:00 ET
    df["year"]   = df["ts"].dt.year.values
    df["gap"]    = (df["ts"].diff() > pd.Timedelta(minutes=2)).values
    df.loc[0, "gap"] = True
    return df


def find_candidates(df: pd.DataFrame) -> pd.DataFrame:
    c      = df["close"].values
    o      = df["open"].values
    hi     = df["high"].values
    lo     = df["low"].values
    vol    = df["volume"].values.astype(float)
    gap    = df["gap"].values.astype(bool)
    is_rth = df["is_rth"].values.astype(bool)
    year   = df["year"].values
    nb     = len(df)

    vol_s  = pd.Series(np.where(gap, np.nan, vol))
    med20  = vol_s.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().values
    gap_cum = np.cumsum(gap.astype(int))

    records = []
    for i in range(VOL_LOOKBACK + 2, nb - HOLD_MAX - PULLBACK_BARS - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        if vol[i] < VOL_MULT * med20[i]:
            continue
        if gap[i] or gap[i - 1]:
            continue
        if c[i] < o[i]:
            continue

        move = c[i] - o[i - 1]
        if move < c[i] * MOVE_BPS / 10000:
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
            "path_hi":   path_hi,
            "path_lo":   np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)]),
            "te_pts":    c[entry_bar + HOLD_MAX] - entry,
        })

    return pd.DataFrame(records)


def simulate(cands: pd.DataFrame, stop_bps: float, point_value: float = 1.0) -> pd.DataFrame:
    if cands.empty:
        return pd.DataFrame()

    hold_until = -1
    records    = []

    for _, row in cands.iterrows():
        eb = row["entry_bar"]
        if eb <= hold_until:
            continue

        entry      = row["entry"]
        target_pts = entry * TARGET_BPS / 10000
        sl_pts     = max(entry * stop_bps / 10000, entry * STOP_MIN_BPS / 10000)
        path_hi    = row["path_hi"]
        path_lo    = row["path_lo"]

        hit_tgt = hit_stop = None
        for k in range(HOLD_MAX):
            if hit_stop is None and (entry - path_lo[k]) >= sl_pts:
                hit_stop = k + 1
            if hit_tgt  is None and (path_hi[k] - entry) >= target_pts:
                hit_tgt  = k + 1

        te = row["te_pts"] if (hit_tgt is None and hit_stop is None) else 0.0
        hold_until = eb + HOLD_MAX

        # Dollar EV for this trade
        if hit_tgt is not None and (hit_stop is None or hit_tgt <= hit_stop):
            trade_pnl = target_pts
        elif hit_stop is not None:
            trade_pnl = -sl_pts
        else:
            trade_pnl = te
        ev_usd = trade_pnl * point_value

        records.append({
            "year":       row["year"],
            "is_rth":     row["is_rth"],
            "entry":      entry,
            "hit_tgt":    hit_tgt,
            "hit_stop":   hit_stop,
            "te_pts":     te,
            "target_pts": target_pts,
            "sl_pts":     sl_pts,
            "ev_usd":     ev_usd,
        })

    return pd.DataFrame(records)


def ev_stats(df: pd.DataFrame, session_filter=None) -> dict:
    if session_filter is not None:
        df = df[df["is_rth"] == session_filter]
    n = len(df)
    if n < MIN_TRADES:
        return {"ev": float("nan"), "ev_bp": float("nan"), "ev_usd": float("nan"),
                "p_tgt": float("nan"), "p_stop": float("nan"), "n": n}
    tgt_bar   = df["hit_tgt"].fillna(999).values
    stop_bar  = df["hit_stop"].fillna(999).values
    ht_first  = df["hit_tgt"].notna().values  & (tgt_bar  <= stop_bar)
    hs_first  = df["hit_stop"].notna().values & (stop_bar <  tgt_bar)
    neither   = ~ht_first & ~hs_first
    te        = df["te_pts"].values
    tgt_pts   = df["target_pts"].values
    sl_pts    = df["sl_pts"].values
    avg_entry = df["entry"].values.mean()
    ev = (float((ht_first * tgt_pts).mean())
          - float((hs_first * sl_pts).mean())
          + (float(te[neither].mean()) * neither.mean() if neither.any() else 0.0))
    ev_bp  = ev / avg_entry * 10000
    ev_usd = df["ev_usd"].mean()
    return {"ev": ev, "ev_bp": ev_bp, "ev_usd": ev_usd,
            "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": n}


def run_one(label: str, csv_path: str):
    pv = POINT_VALUE.get(label, 1.0)
    print(f"\n{'═'*80}")
    print(f"  {label}  —  {csv_path}  (${pv:.0f}/pt)")
    print(f"  vol≥{VOL_MULT}×  move≥{MOVE_BPS}bp  pb {PULLBACK_MIN_BPS}–{PULLBACK_MAX_BPS}bp  "
          f"target={TARGET_BPS}bp  stop≥{STOP_MIN_BPS}bp  hold≤{HOLD_MAX}b  LONG only")
    print(f"{'═'*80}")

    df = load_bars(csv_path)
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    cands = find_candidates(df)
    print(f"  {len(cands):,} candidates")

    if cands.empty:
        print("  No candidates found.")
        return

    results = {lbl: simulate(cands, bps, pv) for lbl, bps in STOPS}

    hdr = (f"  {'Stop':10}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}"
           f"  {'EV(pts)':>9}  {'EV(bp)':>7}  {'EV($)':>8}")
    sep = f"  {'─'*68}"

    for sess_label, sess_filter in [("ALL SESSIONS", None),
                                     ("RTH ONLY",     True),
                                     ("GLOBEX ONLY",  False)]:
        print(f"\n  {sess_label}")
        print(hdr); print(sep)
        for lbl, _ in STOPS:
            st = ev_stats(results[lbl], sess_filter)
            if math.isnan(st["ev"]):
                print(f"  {lbl:10}  {'<min':>5}")
                continue
            flag = " ◄" if st["ev_bp"] > 0 else "  "
            print(f"  {lbl:10}  {st['n']:>5}  {st['p_tgt']:>7.3f}  "
                  f"{st['p_stop']:>8.3f}  {st['ev']:>+9.3f}  "
                  f"{st['ev_bp']:>+7.2f}  {st['ev_usd']:>+8.2f}{flag}")

    # Year-by-year RTH — show EV in bp
    print(f"\n  YEAR-BY-YEAR — RTH only  (EV in bp)")
    print(f"  {'─'*70}")
    years  = sorted(df["year"].unique())
    labels = [lbl for lbl, _ in STOPS]
    print(f"  {'Year':>5}  " + "  ".join(f"{l:>16}" for l in labels))
    print(f"  {'─'*65}")
    for yr in years:
        parts = []
        for lbl in labels:
            sub = results[lbl]
            yr_sub = sub[(sub["year"] == yr) & sub["is_rth"]] if not sub.empty else pd.DataFrame()
            st = ev_stats(yr_sub)
            if math.isnan(st["ev_bp"]):
                parts.append(f"{'—':>16}")
            else:
                s = f"{st['ev_bp']:+.2f}bp({st['n']:3}n)"
                parts.append(f"{s:>16}")
        print(f"  {yr:>5}  " + "  ".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   type=str, help="Path to 1-min OHLCV CSV")
    parser.add_argument("--label", type=str, help="Instrument label for display")
    parser.add_argument("--all",   action="store_true",
                        help="Run against all instruments (MES, MNQ, ES, NQ)")
    args = parser.parse_args()

    if args.all:
        for label, csv_path in ALL_INSTRUMENTS:
            try:
                run_one(label, csv_path)
            except FileNotFoundError:
                print(f"\n  {label}: {csv_path} not found — skipping")
    elif args.csv:
        label = args.label or args.csv
        run_one(label, args.csv)
    else:
        parser.print_help()
