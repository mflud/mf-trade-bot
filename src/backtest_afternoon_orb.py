"""
Backtest: Afternoon ORB (14:30–15:00 range, 15:00–16:00 fire window).

Hypothesis: using a fresh 14:30–15:00 range as the reference level for the
"power hour" produces better breakout follow-through than the current approach
of watching for breakouts of the morning 9:30–9:45 range at 13:30–16:00.

For each session:
  - Builds the range from 14:30–15:00 ET (6 × 5-min bars)
  - Looks for the first close above the range high (LONG only) in 15:00–16:00 ET
  - Tests σ-based stop (2σ) with targets at 1σ, 2σ, 3σ
  - Two exit policies: max 25-min hold vs exit-at-close (16:00 ET)
  - Width filter swept to find the wide-tertile cutoff

Also runs the CURRENT power hour setup (morning ORB, 13:30–16:00 fire window)
as a baseline for comparison.

Usage:
  python src/backtest_afternoon_orb.py           # MES
  python src/backtest_afternoon_orb.py --sym M2K
"""

import argparse
import math
import sys
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Parameters ────────────────────────────────────────────────────────────────

SIGMA_WINDOW = 20          # rolling σ window in bars (matches production)
STOP_SIG     = 2.0
TGT_SIGS     = [1.0, 2.0, 3.0]
MAX_HOLD_MIN = 25          # max hold policy A

# Afternoon ORB
AFT_RANGE_START = dtime(14, 30)
AFT_RANGE_END   = dtime(15,  0)   # 6 × 5-min bars
AFT_FIRE_START  = dtime(15,  0)
AFT_FIRE_END    = dtime(16,  0)

# Morning ORB (current production — used as baseline comparison)
MRN_RANGE_START = dtime(9, 30)
MRN_RANGE_END   = dtime(9, 45)    # 3 × 5-min bars
MRN_FIRE_START  = dtime(13, 30)
MRN_FIRE_END    = dtime(16,  0)
MRN_WIDTH_MIN   = {"MES": 15.25, "M2K": 14.30, "MYM": 0.0}

RTH_OPEN  = dtime(9, 30)
RTH_CLOSE = dtime(16, 0)

HIST = {
    "MES": "mes_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

MIN_SESSIONS = 10


# ── Data ──────────────────────────────────────────────────────────────────────

def load_rth(sym: str) -> pd.DataFrame:
    path = Path(HIST[sym])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    print(f"Loading {path} …", flush=True)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    ts_et = df["ts"].dt.tz_convert(ET)
    rth   = (ts_et.dt.time >= RTH_OPEN) & (ts_et.dt.time < RTH_CLOSE)
    df    = df[rth].copy()
    df["ts_et"] = ts_et[rth]
    df["t_et"]  = df["ts_et"].dt.time
    df["date"]  = df["ts_et"].dt.date

    # Rolling σ (in price points) computed within RTH only
    log_rets = np.log(df["close"] / df["close"].shift(1))
    sigma    = log_rets.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW).std(ddof=1)
    df["sigma_pts"] = sigma * df["close"]

    print(f"  RTH bars: {len(df):,}  "
          f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})", flush=True)
    return df


# ── Forward walk (σ-based stop/target, two exit policies) ─────────────────────

def eval_bracket(post: pd.DataFrame, entry: float, sigma_pts: float,
                 tgt_sig: float, close_deadline: dtime) -> dict:
    """
    Walk forward bar-by-bar from post (bars after entry bar).
    Policy A: exit at target, stop, or MAX_HOLD_MIN minutes (whichever first).
    Policy B: exit at target, stop, or market close (16:00 ET).
    Returns dict of outcomes for both policies.
    """
    tgt_px  = entry + tgt_sig  * sigma_pts
    stop_px = entry - STOP_SIG * sigma_pts

    res = {"A": None, "B": None}   # None = not yet resolved

    for _, bar in post.iterrows():
        elapsed = int((bar["ts_et"] - post["ts_et"].iloc[0]).total_seconds() / 60) + 1

        for policy, deadline in [("A", MAX_HOLD_MIN), ("B", None)]:
            if res[policy] is not None:
                continue
            if policy == "A" and elapsed > deadline:
                # Time exit at close of this bar
                pnl = (bar["close"] - entry) / sigma_pts
                res[policy] = {"outcome": "time", "r": pnl, "mins": elapsed}
                continue
            if policy == "B" and bar["t_et"] >= close_deadline:
                pnl = (bar["close"] - entry) / sigma_pts
                res[policy] = {"outcome": "close", "r": pnl, "mins": elapsed}
                continue

            # Target checked first (convention matching existing backtests)
            if bar["high"] >= tgt_px:
                res[policy] = {"outcome": "target", "r": tgt_sig, "mins": elapsed}
            elif bar["low"] <= stop_px:
                res[policy] = {"outcome": "stop", "r": -STOP_SIG, "mins": elapsed}

        if res["A"] is not None and res["B"] is not None:
            break

    # Fallback: last bar
    for policy in ("A", "B"):
        if res[policy] is None:
            last = post.iloc[-1]
            pnl  = (last["close"] - entry) / sigma_pts
            tag  = "time" if policy == "A" else "close"
            res[policy] = {"outcome": tag, "r": pnl, "mins": len(post)}

    return res


# ── Afternoon ORB scanner ─────────────────────────────────────────────────────

def scan_afternoon_orb(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for date, session in df.groupby("date"):
        session = session.reset_index(drop=True)

        range_bars = session[
            (session["t_et"] >= AFT_RANGE_START) &
            (session["t_et"] <  AFT_RANGE_END)
        ]
        fire_bars = session[
            (session["t_et"] >= AFT_FIRE_START) &
            (session["t_et"] <  AFT_FIRE_END)
        ]

        if len(range_bars) < 4 or fire_bars.empty:   # need most of range window
            continue

        aft_high  = range_bars["high"].max()
        aft_low   = range_bars["low"].min()
        aft_width = aft_high - aft_low
        if aft_width <= 0:
            continue

        # First close above range high (LONG only)
        for idx, bar in fire_bars.iterrows():
            if bar["close"] > aft_high:
                entry      = bar["close"]
                sigma_pts  = bar["sigma_pts"]
                if math.isnan(sigma_pts) or sigma_pts <= 0:
                    break

                post = fire_bars[fire_bars.index > idx].reset_index(drop=True)
                if post.empty:
                    break

                row = {
                    "date":      date,
                    "year":      date.year,
                    "aft_high":  aft_high,
                    "aft_low":   aft_low,
                    "aft_width": aft_width,
                    "entry":     entry,
                    "sigma_pts": sigma_pts,
                    "entry_t":   bar["t_et"],
                }

                for tgt in TGT_SIGS:
                    results = eval_bracket(post, entry, sigma_pts, tgt, AFT_FIRE_END)
                    for pol in ("A", "B"):
                        row[f"tgt{tgt:.0f}_{pol}_r"]       = results[pol]["r"]
                        row[f"tgt{tgt:.0f}_{pol}_outcome"]  = results[pol]["outcome"]
                        row[f"tgt{tgt:.0f}_{pol}_mins"]     = results[pol]["mins"]

                records.append(row)
                break   # first breakout only

    return pd.DataFrame(records)


# ── Morning ORB scanner (current power-hour baseline) ─────────────────────────

def scan_morning_orb_powerhr(df: pd.DataFrame, width_min: float) -> pd.DataFrame:
    """Morning ORB (9:30–9:45), fires in 13:30–16:00 window only (power hour)."""
    records = []
    for date, session in df.groupby("date"):
        session = session.reset_index(drop=True)

        range_bars = session[
            (session["t_et"] >= MRN_RANGE_START) &
            (session["t_et"] <  MRN_RANGE_END)
        ]
        fire_bars = session[
            (session["t_et"] >= MRN_FIRE_START) &
            (session["t_et"] <  MRN_FIRE_END)
        ]

        if len(range_bars) < 2 or fire_bars.empty:
            continue

        mrn_high  = range_bars["high"].max()
        mrn_low   = range_bars["low"].min()
        mrn_width = mrn_high - mrn_low
        if mrn_width <= 0 or (width_min > 0 and mrn_width < width_min):
            continue

        for idx, bar in fire_bars.iterrows():
            if bar["close"] > mrn_high:
                entry     = bar["close"]
                sigma_pts = bar["sigma_pts"]
                if math.isnan(sigma_pts) or sigma_pts <= 0:
                    break

                post = fire_bars[fire_bars.index > idx].reset_index(drop=True)
                if post.empty:
                    break

                row = {
                    "date":       date,
                    "year":       date.year,
                    "mrn_width":  mrn_width,
                    "entry":      entry,
                    "sigma_pts":  sigma_pts,
                }
                for tgt in TGT_SIGS:
                    results = eval_bracket(post, entry, sigma_pts, tgt, MRN_FIRE_END)
                    for pol in ("A", "B"):
                        row[f"tgt{tgt:.0f}_{pol}_r"]       = results[pol]["r"]
                        row[f"tgt{tgt:.0f}_{pol}_outcome"]  = results[pol]["outcome"]

                records.append(row)
                break

    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def ev_table(df: pd.DataFrame, tgt: float, policy: str) -> dict:
    col = f"tgt{tgt:.0f}_{policy}_r"
    oc  = f"tgt{tgt:.0f}_{policy}_outcome"
    if col not in df.columns or len(df) < MIN_SESSIONS:
        return {}
    rs      = df[col].values
    labels  = df[oc].values
    n       = len(rs)
    p_tgt   = (labels == "target").mean()
    p_stop  = (labels == "stop").mean()
    p_time  = (~np.isin(labels, ["target", "stop"])).mean()
    ev      = float(np.mean(rs))
    total_r = float(np.sum(rs))
    return {"n": n, "ev": ev, "total_r": total_r,
            "p_tgt": p_tgt, "p_stop": p_stop, "p_time": p_time}


def print_section(title: str, df: pd.DataFrame, note: str = ""):
    if df.empty:
        print(f"\n  {title}: no signals.")
        return

    print(f"\n{'═'*82}")
    print(f"  {title}")
    if note:
        print(f"  {note}")
    print(f"  n={len(df):,}  "
          f"avg range width: {df['aft_width'].mean():.2f} pts"
          if "aft_width" in df.columns else f"  n={len(df):,}")
    print(f"{'─'*82}")

    print(f"\n  {'Target':>8}  {'Policy':>8}  {'P(tgt)':>8} {'P(stop)':>8} "
          f"{'P(time)':>8}  {'EV':>9}  {'TotalR':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8} {'-'*8} {'-'*8}  {'-'*9}  {'-'*8}")

    best_ev  = -999.0
    best_key = None
    for tgt in TGT_SIGS:
        for pol, pol_label in [("A", f"≤{MAX_HOLD_MIN}m"), ("B", "→close")]:
            s = ev_table(df, tgt, pol)
            if not s:
                continue
            flag = " ◄" if s["ev"] > best_ev else ""
            if s["ev"] > best_ev:
                best_ev  = s["ev"]
                best_key = (tgt, pol)
            print(f"  {f'{tgt:.0f}σ':>8}  {pol_label:>8}  "
                  f"{s['p_tgt']:>8.3f} {s['p_stop']:>8.3f} {s['p_time']:>8.3f}  "
                  f"{s['ev']:>+9.4f}  {s['total_r']:>+8.1f}{flag}")

    if best_key is None:
        return

    tgt, pol = best_key
    pol_label = f"≤{MAX_HOLD_MIN}m" if pol == "A" else "→close"
    print(f"\n  Year-by-year  (best: target={tgt:.0f}σ  policy={pol_label}):")
    print(f"  {'Year':>6} {'n':>5}  {'P(tgt)':>8} {'P(stop)':>8}  "
          f"{'EV':>9}  {'TotalR':>8}")
    col = f"tgt{tgt:.0f}_{pol}_r"
    oc  = f"tgt{tgt:.0f}_{pol}_outcome"
    for yr in sorted(df["year"].unique()):
        yr_df = df[df["year"] == yr]
        if len(yr_df) < MIN_SESSIONS:
            continue
        rs     = yr_df[col].values
        labels = yr_df[oc].values
        p_t    = (labels == "target").mean()
        p_s    = (labels == "stop").mean()
        ev_yr  = float(np.mean(rs))
        tot_yr = float(np.sum(rs))
        flag   = " ◄" if ev_yr > 0 else ""
        print(f"  {yr:>6} {len(yr_df):>5}  "
              f"{p_t:>8.3f} {p_s:>8.3f}  "
              f"{ev_yr:>+9.4f}  {tot_yr:>+8.1f}{flag}")


def print_width_tiers(df: pd.DataFrame, width_col: str, tgt: float, pol: str):
    """Show EV by range-width tertile to find the wide-cutoff."""
    if df.empty or width_col not in df.columns:
        return
    q33, q67 = np.percentile(df[width_col], [33, 67])
    print(f"\n  Width tertiles  (target={tgt:.0f}σ  policy="
          f"{'≤25m' if pol == 'A' else '→close'}):")
    print(f"  {'Tier':>22}  {'n':>5}  {'width':>7}  "
          f"{'P(tgt)':>8} {'P(stop)':>8}  {'EV':>9}  {'TotalR':>8}")
    tiers = [
        (f"Narrow (≤{q33:.1f})",  df[df[width_col] <= q33]),
        (f"Medium ({q33:.1f}–{q67:.1f})", df[(df[width_col] > q33) & (df[width_col] <= q67)]),
        (f"Wide   (>{q67:.1f})",  df[df[width_col] > q67]),
    ]
    for label, sub in tiers:
        s = ev_table(sub, tgt, pol)
        if not s:
            continue
        avg_w = sub[width_col].mean()
        print(f"  {label:>22}  {s['n']:>5}  {avg_w:>7.2f}  "
              f"{s['p_tgt']:>8.3f} {s['p_stop']:>8.3f}  "
              f"{s['ev']:>+9.4f}  {s['total_r']:>+8.1f}")
    print(f"  Wide-tertile cutoff: >{q67:.2f} pts")
    return q67


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=sorted(HIST.keys()))
    args = parser.parse_args()

    df = load_rth(args.sym)

    # ── Afternoon ORB ─────────────────────────────────────────────────────────
    print("\nScanning afternoon ORB (14:30–15:00 range, 15:00–16:00 fire) …",
          flush=True)
    aft = scan_afternoon_orb(df)
    print(f"  {len(aft):,} afternoon ORB signals (all widths)", flush=True)

    print_section(
        f"AFTERNOON ORB — {args.sym}  (14:30–15:00 range  |  15:00–16:00 fire  |  LONG only)",
        aft,
    )

    # Width tertile breakdown for best target/policy
    if not aft.empty:
        best_tgt, best_pol = 2.0, "A"   # typical best; shown for reference
        q67 = print_width_tiers(aft, "aft_width", best_tgt, best_pol)

        # Wide only
        if q67 is not None:
            aft_wide = aft[aft["aft_width"] > q67]
            print_section(
                f"AFTERNOON ORB — {args.sym}  WIDE ONLY (>{q67:.1f} pts)",
                aft_wide,
                note=f"top tertile of afternoon range width",
            )

    # ── Morning ORB power-hour baseline ───────────────────────────────────────
    print("\nScanning current power-hour baseline "
          "(morning ORB, 13:30–16:00 fire) …", flush=True)
    mrn = scan_morning_orb_powerhr(df, MRN_WIDTH_MIN.get(args.sym, 0.0))
    print(f"  {len(mrn):,} current power-hour signals", flush=True)

    print_section(
        f"CURRENT POWER HOUR — {args.sym}  "
        f"(morning 9:30–9:45 range  |  fire 13:30–16:00  |  "
        f"wide >{MRN_WIDTH_MIN.get(args.sym, 0):.1f} pts)",
        mrn,
        note="Baseline comparison for the afternoon ORB",
    )
    print()


if __name__ == "__main__":
    main()
