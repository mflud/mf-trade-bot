"""
backtest_350_continuation.py — Does the 3:50 ET bar direction predict continuation
to the close (and beyond)?

For each RTH trading day in the mizu ES/NQ 1s data:
  1. Aggregate 1s bars into 1-min bars
  2. Take the 3:50 ET bar: direction = sign(close - open)
  3. Reference price = close of 3:50 bar
  4. At each minute from 3:51 to 4:05, measure open price vs reference
  5. Record whether price moved in the signal direction (continuation) or against

Output: hit rate and average move (in pts and bp) at each minute offset.

Usage:
    python src/backtest_350_continuation.py          # ES + NQ
    python src/backtest_350_continuation.py --sym ES
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

INSTRUMENTS = {
    "ES": Path("data/mizu/es"),
    "NQ": Path("data/mizu/nq"),
}

MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

SETTLE_UTC_START = 21
SETTLE_UTC_END   = 22

TARGET_MINUTES = list(range(1, 16))   # 3:51 through 4:05


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(weeks=2)


def expiry_date(sym: str) -> date:
    m = re.search(r"([A-Z])(\d)$", sym)
    if not m:
        return date(2099, 1, 1)
    month = MONTH_CODE.get(m.group(1), 1)
    year  = 2020 + int(m.group(2))
    return third_friday(year, month)


def load_1min_bars(ndjson_dir: Path) -> pd.DataFrame:
    """Load mizu 1s ndjson, pick front-month contract per day, resample to 1min."""
    records = []
    for part in sorted(ndjson_dir.glob("*.ndjson")):
        with open(part) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                ts_val  = r.get("hd", {}).get("ts_event") or r.get("ts_event")
                sym_val = r.get("symbol") or str(r.get("hd", {}).get("instrument_id", ""))
                if ts_val is None:
                    continue
                try:
                    ts = pd.Timestamp(ts_val, tz="UTC")
                except Exception:
                    continue
                records.append({
                    "ts":     ts,
                    "sym":    str(sym_val),
                    "open":   float(r.get("open",  0)),
                    "high":   float(r.get("high",  0)),
                    "low":    float(r.get("low",   0)),
                    "close":  float(r.get("close", 0)),
                    "volume": float(r.get("volume", 0)),
                })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df[df["open"] > 0]

    # Remove settlement gap
    df = df[~df["ts"].dt.tz_convert("UTC").dt.hour.between(
        SETTLE_UTC_START, SETTLE_UTC_END - 1)]

    # Front-month selection: per calendar day, use contract with nearest expiry
    # that hasn't expired yet
    df["date_utc"] = df["ts"].dt.date

    # Map symbol → expiry
    syms = df["sym"].unique()
    expiries = {s: expiry_date(s) for s in syms}

    rows = []
    for day, grp in df.groupby("date_utc"):
        valid = {s: e for s, e in expiries.items() if e >= day}
        if not valid:
            continue
        front = min(valid, key=valid.get)
        subset = grp[grp["sym"] == front].copy()
        rows.append(subset)

    if not rows:
        return pd.DataFrame()

    df2 = pd.concat(rows).sort_values("ts")

    # Resample to 1-min — use label="left" so bar ts = bar open time
    df2 = df2.set_index("ts")
    df2 = df2[["open", "high", "low", "close", "volume"]].resample("1min").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["open"])

    return df2


def run(sym: str):
    ndjson_dir = INSTRUMENTS[sym]
    print(f"\n{'='*60}")
    print(f"  {sym}  —  Loading 1s bars from {ndjson_dir} ...")
    df = load_1min_bars(ndjson_dir)
    if df.empty:
        print("  No data loaded.")
        return

    df.index = df.index.tz_convert(ET)
    print(f"  {len(df):,} 1-min bars  |  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    # Index by (date, time) for fast lookup
    df["date_et"] = df.index.date
    df["hm"]      = df.index.strftime("%H:%M")

    # Build per-day dict: date → {hm: row}
    by_day = defaultdict(dict)
    for ts, row in df.iterrows():
        by_day[row["date_et"]][row["hm"]] = row

    # Per-offset accumulators
    moves   = defaultdict(list)   # offset_min → list of signed moves (in direction)
    n_total = 0

    for day, bars in sorted(by_day.items()):
        bar_350 = bars.get("15:50")
        if bar_350 is None:
            continue
        direction = 1 if bar_350["close"] > bar_350["open"] else (
                   -1 if bar_350["close"] < bar_350["open"] else 0)
        if direction == 0:
            continue

        ref = bar_350["close"]
        n_total += 1

        for offset in TARGET_MINUTES:
            # Target time = 15:50 + offset minutes
            from datetime import time as dtime
            total_min = 15 * 60 + 50 + offset
            hh, mm = divmod(total_min, 60)
            hm = f"{hh:02d}:{mm:02d}"
            target_bar = bars.get(hm)
            if target_bar is None:
                continue
            # Use open of target bar as the price at that minute
            price = target_bar["open"]
            move_pts = (price - ref) * direction   # positive = continuation
            moves[offset].append(move_pts)

    if n_total == 0:
        print("  No 3:50 bars found.")
        return

    print(f"\n  Trading days with 3:50 bar: {n_total}")
    print(f"\n  {'Offset':<10} {'N':>4}  {'Hit%':>6}  {'AvgMove':>9}  {'MedMove':>9}  "
          f"{'Avg|pts|':>9}  {'AvgBp':>7}")
    print(f"  {'─'*62}")

    ref_price = df["close"].median()   # rough reference for bp calc

    for offset in TARGET_MINUTES:
        mvs = moves[offset]
        if not mvs:
            continue
        arr     = np.array(mvs)
        n       = len(arr)
        hit_pct = (arr > 0).mean() * 100
        avg     = arr.mean()
        med     = np.median(arr)
        avg_abs = np.abs(arr).mean()
        avg_bp  = avg / ref_price * 10000

        total_min = 15 * 60 + 50 + offset
        hh, mm = divmod(total_min, 60)
        label = f"+{offset}m ({hh:02d}:{mm:02d})"

        flag = " ◀" if hit_pct >= 60 or hit_pct <= 40 else ""
        print(f"  {label:<14} {n:>4}  {hit_pct:>5.1f}%  {avg:>+9.2f}  {med:>+9.2f}  "
              f"{avg_abs:>9.2f}  {avg_bp:>+7.2f}{flag}")

    # Directional breakdown
    print(f"\n  By direction:")
    long_days  = []
    short_days = []
    for day, bars in sorted(by_day.items()):
        bar_350 = bars.get("15:50")
        if bar_350 is None:
            continue
        direction = 1 if bar_350["close"] > bar_350["open"] else (
                   -1 if bar_350["close"] < bar_350["open"] else 0)
        if direction == 1:
            long_days.append(day)
        elif direction == -1:
            short_days.append(day)

    print(f"    Bullish 3:50 bars: {len(long_days)}   "
          f"Bearish 3:50 bars: {len(short_days)}")

    # Show hit rate at 4:00 by direction
    for dirn, days, label in [(1, long_days, "Bullish"), (-1, short_days, "Bearish")]:
        mvs_400 = []
        for day in days:
            bars = by_day[day]
            bar_350 = bars.get("15:50")
            bar_400 = bars.get("16:00")
            if bar_350 is None or bar_400 is None:
                continue
            ref   = bar_350["close"]
            price = bar_400["open"]
            mvs_400.append((price - ref) * dirn)
        if mvs_400:
            arr = np.array(mvs_400)
            print(f"    {label:>8} → 4:00 hit rate: {(arr>0).mean()*100:.1f}%  "
                  f"avg: {arr.mean():+.2f}pts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", choices=["ES", "NQ", "both"], default="both")
    args = parser.parse_args()

    syms = ["ES", "NQ"] if args.sym == "both" else [args.sym]
    for s in syms:
        run(s)


if __name__ == "__main__":
    main()
