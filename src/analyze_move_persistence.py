"""
analyze_move_persistence.py — For each 30-min bucket of the trading day
(08:30-16:00 ET), count how many MES swings of >=10pt magnitude occur that
never retrace more than 3pt for 3+ continuous minutes along the way (i.e.
"clean" swings, not ones that whipsaw and give back the move before truly
reversing).

Methodology (zigzag with time-confirmed reversal):
  - Track a running swing in the current direction (up or down), with
    `extreme` = the furthest price reached so far in that direction.
  - A pullback from `extreme` only ends the swing if it exceeds 3pt AND
    stays beyond 3pt continuously for >=3 minutes. Smaller or shorter
    pullbacks are noise — the swing's extreme is unaffected and keeps
    tracking new highs/lows normally.
  - When a swing is confirmed ended, its magnitude = |extreme - start|.
    If magnitude >= 10pt, it qualifies. A new swing then starts from the
    old extreme, in the opposite direction.
  - Each day (08:30-16:00 ET) is processed independently; a swing still
    in progress at 16:00 is dropped (right-censored, can't confirm it).
  - Bucketed by the swing's START time (when the prior extreme — now this
    swing's origin — was set).

Uses MES 1-min bars from bars.db (Apr-Oct 2026) — intrabar high/low used
for extreme-tracking and retracement checks, not just closes.

Usage:
  python src/analyze_move_persistence.py
"""

import sqlite3
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
RETRACE_PTS   = 3.0
CONFIRM_MIN   = 3
QUALIFY_PTS   = 10.0
DAY_START_HM  = (8, 30)
DAY_END_HM    = (16, 0)
BUCKET_MIN    = 30


def fetch_bars() -> pd.DataFrame:
    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close FROM bars "
        "WHERE symbol='MES' AND minutes=1 ORDER BY ts"
    ).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"])
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)   # CME settlement gap
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df["et"]   = df["ts"].dt.tz_convert(ET)
    df["hm"]   = df["et"].dt.hour * 60 + df["et"].dt.minute
    df["date"] = df["et"].dt.date
    start_hm = DAY_START_HM[0] * 60 + DAY_START_HM[1]
    end_hm   = DAY_END_HM[0] * 60 + DAY_END_HM[1]
    return df[(df["hm"] >= start_hm) & (df["hm"] < end_hm)].reset_index(drop=True)


def find_swings(day_bars: pd.DataFrame) -> list[dict]:
    """Returns ALL confirmed swings for one day (any magnitude) — caller
    filters to >=QUALIFY_PTS. Each swing: start_ts, start_price, end_ts
    (when extreme was reached), extreme_price, direction, magnitude."""
    bars = day_bars.to_dict("records")
    if len(bars) < 2:
        return []

    swings = []
    direction = 0
    swing_start_price = bars[0]["c"]
    swing_start_ts     = bars[0]["ts"]
    extreme_price = bars[0]["c"]
    extreme_ts    = bars[0]["ts"]
    pullback_since_ts  = None
    pullback_extreme   = None   # min low (up-swing) / max high (down-swing) seen during current pullback

    for bar in bars[1:]:
        if direction == 0:
            if bar["h"] > swing_start_price:
                direction, extreme_price, extreme_ts = 1, bar["h"], bar["ts"]
            elif bar["l"] < swing_start_price:
                direction, extreme_price, extreme_ts = -1, bar["l"], bar["ts"]
            continue

        if direction == 1:
            if bar["h"] > extreme_price:
                extreme_price, extreme_ts = bar["h"], bar["ts"]
                pullback_since_ts, pullback_extreme = None, None
                continue
            retrace = extreme_price - bar["l"]
            if retrace > RETRACE_PTS:
                if pullback_since_ts is None:
                    pullback_since_ts, pullback_extreme = bar["ts"], bar["l"]
                else:
                    pullback_extreme = min(pullback_extreme, bar["l"])
                if (bar["ts"] - pullback_since_ts) >= timedelta(minutes=CONFIRM_MIN):
                    swings.append(dict(start_ts=swing_start_ts, start_price=swing_start_price,
                                       end_ts=extreme_ts, extreme=extreme_price,
                                       direction=1, magnitude=extreme_price - swing_start_price))
                    swing_start_price, swing_start_ts = extreme_price, extreme_ts
                    direction = -1
                    extreme_price, extreme_ts = pullback_extreme, bar["ts"]
                    pullback_since_ts, pullback_extreme = None, None
            else:
                pullback_since_ts, pullback_extreme = None, None

        else:  # direction == -1, mirror image
            if bar["l"] < extreme_price:
                extreme_price, extreme_ts = bar["l"], bar["ts"]
                pullback_since_ts, pullback_extreme = None, None
                continue
            retrace = bar["h"] - extreme_price
            if retrace > RETRACE_PTS:
                if pullback_since_ts is None:
                    pullback_since_ts, pullback_extreme = bar["ts"], bar["h"]
                else:
                    pullback_extreme = max(pullback_extreme, bar["h"])
                if (bar["ts"] - pullback_since_ts) >= timedelta(minutes=CONFIRM_MIN):
                    swings.append(dict(start_ts=swing_start_ts, start_price=swing_start_price,
                                       end_ts=extreme_ts, extreme=extreme_price,
                                       direction=-1, magnitude=swing_start_price - extreme_price))
                    swing_start_price, swing_start_ts = extreme_price, extreme_ts
                    direction = 1
                    extreme_price, extreme_ts = pullback_extreme, bar["ts"]
                    pullback_since_ts, pullback_extreme = None, None
            else:
                pullback_since_ts, pullback_extreme = None, None

    return swings


def bucket_label(ts) -> str:
    et = ts.astimezone(ET)
    hm = et.hour * 60 + et.minute
    start_hm = DAY_START_HM[0] * 60 + DAY_START_HM[1]
    idx = (hm - start_hm) // BUCKET_MIN
    bh, bm = divmod(start_hm + idx * BUCKET_MIN, 60)
    eh, em = divmod(start_hm + (idx + 1) * BUCKET_MIN, 60)
    return f"{bh:02d}:{bm:02d}-{eh:02d}:{em:02d}"


def holds_after(day_bars: pd.DataFrame, swing: dict) -> bool:
    """Does price stay within 3pt of the swing's extreme for the 3 minutes
    immediately AFTER it was reached (the user's original "does the move
    hold up afterward" question — distinct from the zigzag definition's
    "was the push to the extreme itself clean" question)."""
    window = day_bars[(day_bars["ts"] > swing["end_ts"]) &
                      (day_bars["ts"] <= swing["end_ts"] + timedelta(minutes=CONFIRM_MIN))]
    if window.empty:
        return None   # right-censored (move happened too close to 16:00 to check)
    if swing["direction"] == 1:
        return bool((swing["extreme"] - window["l"]).max() <= RETRACE_PTS)
    else:
        return bool((window["h"] - swing["extreme"]).max() <= RETRACE_PTS)


def main():
    df = fetch_bars()
    print(f"Loaded {len(df)} MES 1-min bars, {df['date'].nunique()} sessions "
          f"({df['date'].min()} -> {df['date'].max()})", flush=True)

    all_swings = []
    day_bars_by_date = {}
    for date, day_df in df.groupby("date"):
        day_df = day_df.sort_values("ts")
        day_bars_by_date[date] = day_df
        all_swings.extend(find_swings(day_df))
    for s in all_swings:
        s["date"] = s["start_ts"].astimezone(ET).date()

    qualifying = [s for s in all_swings if s["magnitude"] >= QUALIFY_PTS]
    print(f"\nTotal confirmed swings (any size): {len(all_swings)}")
    print(f"Qualifying swings (>={QUALIFY_PTS}pt, held per the 3pt/3min rule): {len(qualifying)}")

    n_days = df["date"].nunique()
    start_hm = DAY_START_HM[0] * 60 + DAY_START_HM[1]
    end_hm   = DAY_END_HM[0] * 60 + DAY_END_HM[1]
    n_buckets = (end_hm - start_hm) // BUCKET_MIN
    labels = []
    for i in range(n_buckets):
        bh, bm = divmod(start_hm + i * BUCKET_MIN, 60)
        eh, em = divmod(start_hm + (i + 1) * BUCKET_MIN, 60)
        labels.append(f"{bh:02d}:{bm:02d}-{eh:02d}:{em:02d}")

    print("\nComputing post-move 3min holding rate for each qualifying swing "
          "(user's original 'does it revert afterward' question)...", flush=True)
    for s in qualifying:
        s["holds"] = holds_after(day_bars_by_date[s["date"]], s)

    rows = []
    for lbl in labels:
        q = [s for s in qualifying if bucket_label(s["start_ts"]) == lbl]
        up = sum(1 for s in q if s["direction"] == 1)
        dn = sum(1 for s in q if s["direction"] == -1)
        mags = [s["magnitude"] for s in q]
        checkable = [s for s in q if s["holds"] is not None]
        held = [s for s in checkable if s["holds"]]
        rows.append(dict(bucket=lbl, count=len(q), up=up, down=dn,
                         per_day=round(len(q) / n_days, 3),
                         avg_magnitude=round(np.mean(mags), 1) if mags else 0.0,
                         holds_3min_pct=round(100 * len(held) / len(checkable), 1) if checkable else None,
                         holds_3min_n=len(held)))

    out = pd.DataFrame(rows)
    print(f"\n{'='*70}\nQUALIFYING 10pt+ SWINGS BY START-TIME BUCKET  ({n_days} sessions)\n{'='*70}")
    print(out.to_string(index=False))

    out.to_csv("mes_move_persistence_by_bucket.csv", index=False)
    print("\nSaved to mes_move_persistence_by_bucket.csv")


if __name__ == "__main__":
    main()
