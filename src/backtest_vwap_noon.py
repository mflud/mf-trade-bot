"""
Backtest: Does a VWAP breakout at ~12:00 ET continue for 2-4 hours?

For each RTH session:
  1. Compute cumulative VWAP (+ ±1σ, ±2σ bands) from 9:30 ET using 1-min bars.
  2. At a trigger window (11:50–12:10 ET), measure how far price is from VWAP
     in units of the VWAP standard deviation (z-score).
  3. If |z| >= threshold, enter in the direction of the breakout at the
     first bar that crosses the threshold (or at 12:00 exactly).
  4. Measure directional forward returns at 30, 60, 90, 120, 180, 240 min.
  5. Year-by-year breakdown.

Usage:
  python src/backtest_vwap_noon.py
"""

import math
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

ET          = ZoneInfo("America/New_York")
CSV_PATH    = "mes_hist_1min.csv"

# Trigger window: look for breakout between these times (ET hour, minute)
TRIGGER_START = (11, 50)
TRIGGER_END   = (12, 10)

# VWAP z-score thresholds to sweep
Z_THRESHOLDS  = [0.5, 1.0, 1.5, 2.0]

# Forward hold windows (in minutes)
HOLD_MINS     = [30, 60, 90, 120, 180, 240]

MIN_N         = 15


# ── Data ──────────────────────────────────────────────────────────────────────

def load_rth_1min() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(ET)
    df = df.reset_index().rename(columns={df.index.name or "ts": "ts"})
    df.columns = ["ts"] + list(df.columns[1:])

    h = df["ts"].dt.hour
    m = df["ts"].dt.minute
    mins = h * 60 + m
    rth = df[(mins >= 9*60+30) & (mins < 16*60)].copy()
    return rth.reset_index(drop=True)


# ── VWAP helpers ──────────────────────────────────────────────────────────────

def compute_vwap_series(day_df: pd.DataFrame):
    """Return vwap and vwap_sd series for a single day's RTH bars."""
    tp  = (day_df["high"] + day_df["low"] + day_df["close"]).values / 3
    vol = day_df["volume"].values.astype(float)
    vol = np.where(vol == 0, 1e-9, vol)

    cum_vol  = np.cumsum(vol)
    vwap     = np.cumsum(tp * vol) / cum_vol
    var      = np.maximum(np.cumsum(tp**2 * vol) / cum_vol - vwap**2, 0)
    sd       = np.sqrt(var)
    return vwap, sd


# ── Event detection ───────────────────────────────────────────────────────────

def find_noon_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each session day, find the first bar in the trigger window where
    |z_score| >= min threshold. Return one row per day with z-score and
    forward return lookup index.
    """
    t_start = TRIGGER_START[0] * 60 + TRIGGER_START[1]
    t_end   = TRIGGER_END[0]   * 60 + TRIGGER_END[1]

    df["date"] = df["ts"].dt.date
    df["mins"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute

    records = []
    for date, day in df.groupby("date"):
        day = day.reset_index(drop=True)
        if len(day) < 60:
            continue

        vwap, sd = compute_vwap_series(day)
        day = day.copy()
        day["vwap"] = vwap
        day["sd"]   = sd

        # Find trigger bar: first bar in window where |z| >= min threshold
        win = day[(day["mins"] >= t_start) & (day["mins"] <= t_end)].copy()
        if win.empty:
            continue

        # Use the 12:00 bar specifically (or closest)
        noon_candidates = win[win["mins"] == 12*60]
        if noon_candidates.empty:
            noon_candidates = win.iloc[[len(win)//2]]  # midpoint fallback

        row = noon_candidates.iloc[0]
        price = float(row["close"])
        vw    = float(row["vwap"])
        s     = float(row["sd"])
        if s < 0.01:
            continue

        z = (price - vw) / s
        idx = int(row.name)

        # Forward returns at each hold window
        fwd = {}
        for h in HOLD_MINS:
            target_idx = idx + h
            if target_idx < len(day):
                exit_p = float(day.iloc[target_idx]["close"])
                fwd[h] = math.log(exit_p / price)
            else:
                fwd[h] = float("nan")

        rec = {
            "date":  date,
            "year":  pd.Timestamp(date).year,
            "ts":    row["ts"],
            "price": price,
            "vwap":  vw,
            "sd":    s,
            "z":     z,
        }
        rec.update({f"fwd_{h}": v for h, v in fwd.items()})
        records.append(rec)

    return pd.DataFrame(records)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(data: pd.DataFrame, z_thresh: float, hold_min: int) -> dict | None:
    fwd_col = f"fwd_{hold_min}"
    sub = data[data["z"].abs() >= z_thresh].dropna(subset=[fwd_col]).copy()
    if len(sub) < MIN_N:
        return None

    direction   = np.sign(sub["z"].values)
    fwd         = sub[fwd_col].values
    dir_ret     = direction * fwd   # positive = continuation

    sigma_all   = float(np.std(data[fwd_col].dropna()))
    mean_cont   = float(np.mean(dir_ret))
    mean_sigma  = mean_cont / sigma_all if sigma_all > 0 else 0.0

    return {
        "z_thresh":   z_thresh,
        "hold_min":   hold_min,
        "n":          len(sub),
        "mean_ret%":  mean_cont * 100,
        "mean_sigma": mean_sigma,
        "pct_cont":   float(np.mean(dir_ret > 0)) * 100,
    }


def print_results(data: pd.DataFrame):
    print(f"\n{'='*68}")
    print(f"  VWAP noon breakout continuation  (trigger: 11:50–12:10 ET)")
    print(f"{'='*68}")
    print(f"  {'z≥':>5}  {'hold':>5}  {'n':>5}  {'mean_ret%':>9}  "
          f"{'sigma':>7}  {'pct_cont%':>9}")
    print(f"  {'-'*55}")

    for z in Z_THRESHOLDS:
        for h in HOLD_MINS:
            r = analyse(data, z, h)
            if r is None:
                continue
            marker = " ◀" if r["mean_sigma"] > 0.05 and r["pct_cont"] > 55 else ""
            print(f"  {z:>4.1f}σ  {h:>4}m  {r['n']:>5}  "
                  f"{r['mean_ret%']:>+8.3f}%  "
                  f"{r['mean_sigma']:>+7.3f}σ  "
                  f"{r['pct_cont']:>8.1f}%{marker}")
        print()


def print_yearly(data: pd.DataFrame, z_thresh: float = 1.0, hold_min: int = 120):
    fwd_col = f"fwd_{hold_min}"
    sub = data[data["z"].abs() >= z_thresh].dropna(subset=[fwd_col]).copy()
    if sub.empty:
        return

    sigma_all = float(np.std(data[fwd_col].dropna()))
    sub["dir_ret"] = np.sign(sub["z"]) * sub[fwd_col]

    print(f"\n{'='*55}")
    print(f"  Year-by-year  (z≥{z_thresh}σ, hold={hold_min}m)")
    print(f"{'='*55}")
    print(f"  {'Year':>5}  {'n':>5}  {'mean_sigma':>10}  {'pct_cont%':>9}")
    print(f"  {'-'*40}")
    for yr, grp in sub.groupby("year"):
        ms = grp["dir_ret"].mean() / sigma_all if sigma_all else 0
        pc = (grp["dir_ret"] > 0).mean() * 100
        print(f"  {yr:>5}  {len(grp):>5}  {ms:>+10.3f}σ  {pc:>8.1f}%")


def print_z_distribution(data: pd.DataFrame):
    print(f"\n  Z-score distribution at 12:00 ET ({len(data)} sessions):")
    for thr in Z_THRESHOLDS:
        n = (data["z"].abs() >= thr).sum()
        print(f"    |z| ≥ {thr:.1f}σ:  {n:4d}  ({n/len(data)*100:.0f}%)")
    print(f"    median |z|:   {data['z'].abs().median():.2f}σ")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {CSV_PATH} …")
    df = load_rth_1min()
    print(f"  {len(df):,} RTH bars  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    print("Finding noon VWAP breakouts …")
    data = find_noon_breakouts(df)
    print(f"  {len(data)} sessions with 12:00 ET bars")

    print_z_distribution(data)
    print_results(data)

    # Year-by-year for the most informative combos
    for z, h in [(1.0, 120), (1.5, 120), (1.0, 240)]:
        print_yearly(data, z_thresh=z, hold_min=h)
