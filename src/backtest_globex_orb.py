"""
Backtest: Opening Range Breakout at the Globex open (18:00 ET).

Session definition:
  - Globex open: 18:00 ET
  - ORB range:   18:00 – 18:00+N min
  - Fire window: 18:15 – 20:30 ET  (before overnight session thins out)
  - Forward walk: up to 22:00 ET (well before next day's settlement gap at ~17:00 ET)

Sweeps ORB periods: 5, 10, 15, 30 min.
Reports are split by:
  - All sessions combined
  - Sunday night only  (first open of the week, gap from Friday close)
  - Mon–Thu nights

Only LONG breakouts are reported (consistent with RTH ORB finding that
short breakouts are consistently negative).

Usage:
  python src/backtest_globex_orb.py          # MES (default)
  python src/backtest_globex_orb.py --sym M2K
"""

import argparse
import sys
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

GLOBEX_OPEN  = time(18,  0)   # ET: session open
FIRE_END_ET  = time(20, 30)   # ET: last bar allowed to trigger a breakout
SESSION_END  = time(22,  0)   # ET: cap for forward-return walk
ORB_PERIODS  = [5, 10, 15, 30]
HOLD_MINS    = [15, 30, 45, 60]
MIN_SESSIONS = 20
SIGMA_WINDOW = 20
STOP_SIGMA   = 2.0

HIST = {
    "MES": "mes_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_globex(sym: str) -> pd.DataFrame:
    path = Path(HIST[sym])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    print(f"Loading {path} …", flush=True)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    df["ts_et"] = df["ts"].dt.tz_convert(ET)
    df["t_et"]  = df["ts_et"].dt.time
    df["dow"]   = df["ts_et"].dt.dayofweek   # Mon=0, Sun=6

    # Filter to Globex window: 18:00–22:00 ET
    mask = (df["t_et"] >= GLOBEX_OPEN) & (df["t_et"] < SESSION_END)
    df = df[mask].copy().reset_index(drop=True)

    # Precompute rolling sigma across the full (filtered) series
    log_rets       = np.log(df["close"] / df["close"].shift(1))
    sigma          = log_rets.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW).std(ddof=1)
    df["sigma_pts"] = sigma * df["close"]

    # Session key = calendar date at Globex open (the date the bar falls on when t_et >= 18:00)
    df["session_date"] = df["ts_et"].dt.date
    df["mins_from_open"] = (
        df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute - 18 * 60
    )

    # Day type: Sunday session (first open of week), or Mon–Thu
    # The Globex open at 18:00 ET:
    #   Sunday  night → dow == 6
    #   Mon–Thu night → dow in {0,1,2,3}
    #   Friday night doesn't exist (market closes Friday 16:00 ET)
    df["is_sunday"] = df["dow"] == 6

    print(
        f"Globex bars (18:00–22:00 ET): {len(df):,}  "
        f"({df['session_date'].iloc[0]} → {df['session_date'].iloc[-1]})",
        flush=True,
    )
    return df


# ── ORB scanner ───────────────────────────────────────────────────────────────

def scan_orb(df: pd.DataFrame, orb_min: int) -> pd.DataFrame:
    """
    For each Globex session, build the ORB from the first orb_min minutes,
    then record the first LONG close above ORB high within the fire window
    (18:00+orb_min → 20:30 ET).  Forward returns measured from the breakout bar.
    """
    fire_end_mins = (
        FIRE_END_ET.hour * 60 + FIRE_END_ET.minute - 18 * 60
    )   # minutes from Globex open

    records = []

    for session_date, session in df.groupby("session_date"):
        session    = session.reset_index(drop=True)
        is_sunday  = bool(session["is_sunday"].iloc[0])

        # ORB bars: first orb_min minutes (mins_from_open in [0, orb_min))
        orb_bars  = session[session["mins_from_open"] < orb_min]
        post_bars = session[
            (session["mins_from_open"] >= orb_min) &
            (session["mins_from_open"] <= fire_end_mins)
        ]

        if len(orb_bars) < orb_min or len(post_bars) < 2:
            continue

        orb_high  = orb_bars["high"].max()
        orb_low   = orb_bars["low"].min()
        orb_width = orb_high - orb_low
        if orb_width <= 0:
            continue

        post = post_bars.reset_index(drop=True)

        # LONG only: first close above ORB high
        for idx in range(len(post)):
            bar = post.iloc[idx]
            if bar["close"] <= orb_high:
                continue

            entry      = bar["close"]
            entry_min  = bar["mins_from_open"]
            sigma_pts  = bar["sigma_pts"]

            row = {
                "session_date": session_date,
                "is_sunday":    is_sunday,
                "entry":        entry,
                "orb_high":     orb_high,
                "orb_low":      orb_low,
                "orb_width":    orb_width,
                "entry_min":    entry_min,
                "sigma_pts":    sigma_pts,
            }

            # All bars after breakout (up to SESSION_END)
            post_after = session[
                session["mins_from_open"] > entry_min
            ].reset_index(drop=True)

            # Fixed-horizon forwards
            for h in HOLD_MINS:
                target_min = entry_min + h
                future = post_after[post_after["mins_from_open"] >= target_min]
                if len(future) > 0:
                    row[f"fwd_{h}"] = future["close"].iloc[0] - entry
                else:
                    row[f"fwd_{h}"] = post_after["close"].iloc[-1] - entry if len(post_after) > 0 else 0.0

            # Sigma-based stop/target
            if not np.isnan(sigma_pts) and sigma_pts > 0:
                sig_stop = entry - STOP_SIGMA * sigma_pts
                row["risk_sigma"] = STOP_SIGMA * sigma_pts
                for r_mult in [1, 2, 3]:
                    tgt_px  = entry + r_mult * sigma_pts
                    hit_tgt = hit_stp = False
                    mins_to = None
                    for j in range(len(post_after)):
                        b = post_after.iloc[j]
                        if b["low"] <= sig_stop:
                            hit_stp = True
                            mins_to = int(b["mins_from_open"] - entry_min)
                            break
                        if b["high"] >= tgt_px:
                            hit_tgt = True
                            mins_to = int(b["mins_from_open"] - entry_min)
                            break
                    row[f"hit_sig_{r_mult}R"] = hit_tgt
                    row[f"stop_sig_{r_mult}R"] = hit_stp
                    row[f"mins_sig_{r_mult}R"] = mins_to
            else:
                row["risk_sigma"] = np.nan

            records.append(row)
            break   # first breakout only per session

    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _print_sigma_block(sub: pd.DataFrame, label: str, orb_min: int):
    n = len(sub)
    if n < MIN_SESSIONS:
        print(f"      {label}: n={n} (too few)")
        return

    for r_mult in [1, 2, 3]:
        col_hit  = f"hit_sig_{r_mult}R"
        col_stop = f"stop_sig_{r_mult}R"
        if col_hit not in sub.columns:
            continue
        valid = sub[sub["risk_sigma"].notna()]
        if len(valid) < MIN_SESSIONS:
            continue
        n_v      = len(valid)
        p_tgt    = valid[col_hit].mean()
        p_stp    = valid[col_stop].mean()
        p_open   = 1 - p_tgt - p_stp
        ev_r     = p_tgt * r_mult - p_stp * STOP_SIGMA
        total_r  = ev_r * n_v
        print(f"      {label}  {r_mult}R tgt: "
              f"n={n_v:3d}  P(tgt)={p_tgt:.1%}  P(stp)={p_stp:.1%}  "
              f"P(open)={p_open:.1%}  EV={ev_r:+.3f}R  total={total_r:+.1f}R")


def print_results(results: pd.DataFrame, orb_min: int, sym: str):
    if len(results) < MIN_SESSIONS:
        print(f"\n  ORB {orb_min:>2} min: insufficient sessions ({len(results)})")
        return

    print(f"\n{'─'*70}")
    print(f"  {sym}  Globex ORB {orb_min:>2} min  —  n={len(results)} sessions total")
    print(f"{'─'*70}")

    # Width tertiles
    q33, q67 = results["orb_width"].quantile([1/3, 2/3])
    results = results.copy()
    results["width_tier"] = pd.cut(
        results["orb_width"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Narrow", "Medium", "Wide"],
    )
    print(f"  Width tertiles: narrow ≤{q33:.2f}  medium ≤{q67:.2f}  wide >{q67:.2f} pts")

    # --- All sessions ---
    print(f"\n  ALL SESSIONS ({len(results)} total):")
    for tier in ["All", "Narrow", "Medium", "Wide"]:
        sub = results if tier == "All" else results[results["width_tier"] == tier]
        _print_sigma_block(sub, f"  {tier:<7}", orb_min)

    # --- Sunday vs Weekday ---
    sunday  = results[results["is_sunday"]]
    weekday = results[~results["is_sunday"]]
    print(f"\n  SUNDAY night ({len(sunday)} sessions):")
    for tier in ["All", "Narrow", "Medium", "Wide"]:
        sub = sunday if tier == "All" else sunday[sunday["width_tier"] == tier]
        _print_sigma_block(sub, f"  {tier:<7}", orb_min)

    print(f"\n  MON–THU night ({len(weekday)} sessions):")
    for tier in ["All", "Narrow", "Medium", "Wide"]:
        sub = weekday if tier == "All" else weekday[weekday["width_tier"] == tier]
        _print_sigma_block(sub, f"  {tier:<7}", orb_min)

    # --- Forward-return summary (all sessions) ---
    print(f"\n  FIXED-HOLD FORWARD RETURNS (LONG, all sessions, mean pts):")
    for h in HOLD_MINS:
        col = f"fwd_{h}"
        if col in results.columns:
            avg = results[col].mean()
            pct = (results[col] > 0).mean()
            print(f"    {h:3d} min:  mean={avg:+.2f} pts  P(pos)={pct:.1%}  n={len(results)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=list(HIST))
    args = parser.parse_args()
    sym = args.sym

    df = load_globex(sym)

    # Quick session count by type
    sessions = df.groupby("session_date")["is_sunday"].first()
    n_sun  = sessions.sum()
    n_week = (~sessions).sum()
    print(f"Sessions: {len(sessions)} total  ({n_sun} Sunday, {n_week} Mon–Thu)")

    for orb_min in ORB_PERIODS:
        print(f"\nScanning ORB {orb_min} min …", flush=True)
        results = scan_orb(df, orb_min)
        print_results(results, orb_min, sym)

    print()


if __name__ == "__main__":
    main()
