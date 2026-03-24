"""
Backtest: Opening Range Breakout (ORB) on MES or MYM 1-min bars.

For each RTH session (9:30–16:00 ET), defines the opening range as the
high/low of the first N minutes. After the ORB period closes, records the
FIRST bar whose close breaks above the ORB high (long) or below the ORB low
(short), then measures forward returns at multiple horizons including EOD.

Sweeps ORB periods: 1, 5, 10, 15, 30, 60 min.

Usage:
  python src/backtest_orb.py           # MES
  python src/backtest_orb.py --sym MYM
"""

import argparse
import sys
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

RTH_OPEN  = time(9, 30)
RTH_CLOSE = time(16, 0)

ORB_PERIODS  = [1, 5, 10, 15, 30, 60]   # minutes after open
HOLD_MINS    = [15, 30, 45, 60, 120]     # forward horizons in minutes
MIN_SESSIONS = 20                         # suppress results with fewer sessions
SIGMA_WINDOW = 20                         # bars for rolling σ (matches signal_monitor TRAILING_BARS)
STOP_SIGMA   = 2.0                        # stop distance in σ units

HIST = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_rth(sym: str) -> pd.DataFrame:
    path = Path(HIST[sym])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    print(f"Loading {path} …", flush=True)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    # Filter to RTH only
    ts_et    = df["ts"].dt.tz_convert(ET)
    rth_mask = (
        (ts_et.dt.time >= RTH_OPEN) &
        (ts_et.dt.time <  RTH_CLOSE)
    )
    df = df[rth_mask].copy()
    df["ts_et"]          = ts_et[rth_mask]           # keep tz-aware
    df["date"]           = df["ts_et"].dt.date
    df["mins_from_open"] = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute - 570
    df = df.reset_index(drop=True)

    # Precompute rolling sigma at each bar (log-return std over trailing SIGMA_WINDOW bars)
    log_rets = np.log(df["close"] / df["close"].shift(1))
    sigma    = log_rets.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW).std(ddof=1)
    df["sigma_pts"] = sigma * df["close"]   # in price points

    print(f"RTH bars: {len(df):,}  "
          f"({df['date'].iloc[0]} → {df['date'].iloc[-1]})",
          flush=True)
    return df


# ── ORB scanner ───────────────────────────────────────────────────────────────

def scan_orb(df: pd.DataFrame, orb_min: int) -> pd.DataFrame:
    """
    For each session, build the ORB from the first orb_min bars, then
    record the first close that breaks above (long) or below (short) the range.
    Forward returns measured from the breakout bar.
    EOD return = close of last bar of the session.
    """
    records = []

    for date, session in df.groupby("date"):
        session = session.reset_index(drop=True)   # preserves sigma_pts column

        # ORB bars: mins_from_open in [0, orb_min)
        orb_bars = session[session["mins_from_open"] < orb_min]
        post_bars = session[session["mins_from_open"] >= orb_min]

        if len(orb_bars) < orb_min or len(post_bars) < 2:
            continue   # incomplete session

        orb_high = orb_bars["high"].max()
        orb_low  = orb_bars["low"].min()
        orb_width = orb_high - orb_low
        if orb_width <= 0:
            continue

        eod_close = session["close"].iloc[-1]
        post = post_bars.reset_index(drop=True)

        # Find first long and first short breakout separately
        for direction, label in [(1, "LONG"), (-1, "SHORT")]:
            triggered = False
            for idx in range(len(post)):
                bar = post.iloc[idx]
                if direction == 1 and bar["close"] > orb_high:
                    triggered = True
                elif direction == -1 and bar["close"] < orb_low:
                    triggered = True
                if not triggered:
                    continue

                entry      = bar["close"]
                entry_min  = bar["mins_from_open"]   # minutes from open
                sigma_pts  = bar["sigma_pts"]        # rolling σ in price points at entry bar
                row = {
                    "date":       date,
                    "direction":  direction,
                    "entry":      entry,
                    "orb_high":   orb_high,
                    "orb_low":    orb_low,
                    "orb_width":  orb_width,
                    "entry_min":  entry_min,
                    "sigma_pts":  sigma_pts,
                    # How far through the range did the breakout close?
                    "break_ext":  (entry - orb_high) / orb_width if direction == 1
                                  else (orb_low - entry) / orb_width,
                    # EOD return (positive = continuation)
                    "fwd_eod":    (eod_close - entry) * direction,
                }

                # Fixed-horizon forwards
                for h in HOLD_MINS:
                    target_min = entry_min + h
                    future = post[post["mins_from_open"] >= target_min]
                    if len(future) > 0:
                        fwd_close = future["close"].iloc[0]
                        row[f"fwd_{h}"] = (fwd_close - entry) * direction
                    else:
                        row[f"fwd_{h}"] = (eod_close - entry) * direction

                # Stop-placement analysis: stop = ORB low (LONG) / ORB high (SHORT)
                post_after = post.iloc[idx + 1:].reset_index(drop=True)

                # Two stop variants:
                #   wide  stop = ORB low (LONG) / ORB high (SHORT)
                #   tight stop = ORB high (LONG) / ORB low (SHORT) — just the breakout level
                for stop_label, stop_px in [
                    ("wide",  orb_low  if direction == 1 else orb_high),
                    ("tight", orb_high if direction == 1 else orb_low),
                ]:
                    risk_pts = (entry - stop_px) * direction
                    row[f"risk_{stop_label}"] = risk_pts
                    for r_mult in [1, 2, 3]:
                        tgt_px = entry + direction * risk_pts * r_mult
                        hit_tgt = hit_stp = False
                        mins_to = None
                        for j in range(len(post_after)):
                            b = post_after.iloc[j]
                            if direction == 1 and b["low"] <= stop_px:
                                hit_stp = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                            elif direction == -1 and b["high"] >= stop_px:
                                hit_stp = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                            if direction == 1 and b["high"] >= tgt_px:
                                hit_tgt = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                            elif direction == -1 and b["low"] <= tgt_px:
                                hit_tgt = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                        row[f"hit_{stop_label}_{r_mult}R"]  = hit_tgt
                        row[f"stop_{stop_label}_{r_mult}R"] = hit_stp
                        row[f"mins_{stop_label}_{r_mult}R"] = mins_to
                # keep backward-compat alias
                row["risk_pts"] = row["risk_wide"]

                # Sigma-based stop: stop = entry ∓ STOP_SIGMA × sigma_pts
                # Target multiples: 1×, 2×, 3× the same sigma_pts distance
                if not np.isnan(sigma_pts) and sigma_pts > 0:
                    sig_stop_px = entry - direction * STOP_SIGMA * sigma_pts
                    risk_sig    = STOP_SIGMA * sigma_pts
                    row["risk_sigma"] = risk_sig
                    for r_mult in [1, 2, 3]:
                        tgt_px  = entry + direction * r_mult * sigma_pts
                        hit_tgt = hit_stp = False
                        mins_to = None
                        for j in range(len(post_after)):
                            b = post_after.iloc[j]
                            if direction == 1 and b["low"] <= sig_stop_px:
                                hit_stp = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                            elif direction == -1 and b["high"] >= sig_stop_px:
                                hit_stp = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                            if direction == 1 and b["high"] >= tgt_px:
                                hit_tgt = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                            elif direction == -1 and b["low"] <= tgt_px:
                                hit_tgt = True
                                mins_to = int(b["mins_from_open"] - entry_min)
                                break
                        row[f"hit_sig_{r_mult}R"]  = hit_tgt
                        row[f"stop_sig_{r_mult}R"] = hit_stp
                        row[f"mins_sig_{r_mult}R"] = mins_to
                else:
                    row["risk_sigma"] = np.nan

                records.append(row)
                break   # first breakout only

    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_orb_results(results: pd.DataFrame, orb_min: int, sym: str):
    total = len(results)
    up    = (results["direction"] ==  1).sum()
    dn    = (results["direction"] == -1).sum()

    if total < MIN_SESSIONS:
        print(f"\n  ORB {orb_min:>2} min: insufficient sessions ({total})")
        return

    print(f"\n{'═'*68}")
    print(f"  {sym}  ORB {orb_min:>2} min   sessions: {total:,}  "
          f"(long={up:,}  short={dn:,})")
    print(f"  ORB width: avg {results['orb_width'].mean():.2f} pts  "
          f"  entry delay from open: avg {results['entry_min'].mean():.0f} min")
    print(f"{'─'*68}")

    horizons = HOLD_MINS + ["eod"]
    print(f"  {'Horizon':>8}  {'ALL':>24}  {'LONG':>24}  {'SHORT':>24}")
    print(f"  {'':>8}  {'n':>5} {'Hit%':>6} {'Avg pts':>8}  "
          f"{'n':>5} {'Hit%':>6} {'Avg pts':>8}  "
          f"{'n':>5} {'Hit%':>6} {'Avg pts':>8}")
    print(f"  {'-'*8}  {'-'*24}  {'-'*24}  {'-'*24}")

    long_mask  = results["direction"] ==  1
    short_mask = results["direction"] == -1

    for h in horizons:
        col = f"fwd_{h}" if h != "eod" else "fwd_eod"
        label = f"{h} min" if h != "eod" else "EOD"

        for subset, mask in [
            ("ALL",   slice(None)),
            ("LONG",  long_mask),
            ("SHORT", short_mask),
        ]:
            pass   # collect below

        subsets = [
            ("ALL",   results[col]),
            ("LONG",  results.loc[long_mask,  col]),
            ("SHORT", results.loc[short_mask, col]),
        ]
        parts = []
        for label2, fwd in subsets:
            if len(fwd) < MIN_SESSIONS:
                parts.append(f"{'—':>5} {'—':>6} {'—':>8}")
            else:
                hit = (fwd > 0).mean() * 100
                avg = fwd.mean()
                flag = "◄" if avg > 0 else " "
                parts.append(f"{len(fwd):>5,} {hit:>5.1f}% {avg:>+7.2f}{flag}")

        print(f"  {label:>8}  {'  '.join(parts)}")

    # ORB width tertile breakdown (LONG only, all horizons)
    long_res = results[results["direction"] == 1].copy()
    if len(long_res) >= 30:
        q33 = long_res["orb_width"].quantile(1/3)
        q67 = long_res["orb_width"].quantile(2/3)
        def width_tier(w):
            if w <= q33: return "narrow"
            if w <= q67: return "medium"
            return "wide"
        long_res["width_tier"] = long_res["orb_width"].apply(width_tier)

        print(f"\n  ORB width tertiles (LONG only):  "
              f"narrow ≤{q33:.2f}  medium ≤{q67:.2f}  wide >{q67:.2f}")
        tier_order = ["narrow", "medium", "wide"]
        h_labels = [(f"fwd_{h}", f"{h}m") for h in HOLD_MINS] + [("fwd_eod", "EOD")]
        header_parts = "  ".join(f"{'n':>5} {'Hit%':>6} {'Avg':>7}" for _ in h_labels)
        col_headers  = "  ".join(f"{lbl:>20}" for _, lbl in h_labels)
        print(f"  {'Tier':>8}  {'w avg':>6}  " + "  ".join(
            f"{'─── ' + lbl + ' ───':>20}" for _, lbl in h_labels))
        print(f"  {'':>8}  {'':>6}  " + "  ".join(
            f"{'n':>5} {'Hit%':>6} {'Avg':>7}" for _ in h_labels))
        for tier in tier_order:
            grp = long_res[long_res["width_tier"] == tier]
            w_avg = grp["orb_width"].mean()
            parts = []
            for col, _ in h_labels:
                fwd = grp[col].dropna()
                if len(fwd) < MIN_SESSIONS:
                    parts.append(f"{'—':>5} {'—':>6} {'—':>7}")
                else:
                    hit = (fwd > 0).mean() * 100
                    avg = fwd.mean()
                    flag = "◄" if avg > 0 else " "
                    parts.append(f"{len(fwd):>5,} {hit:>5.1f}% {avg:>+6.2f}{flag}")
            print(f"  {tier:>8}  {w_avg:>6.2f}  " + "  ".join(parts))

    # Stop-placement analysis: wide-ORB LONG only, both stop variants
    if len(long_res) >= 30 and "risk_wide" in long_res.columns:
        wide_s = long_res[long_res["width_tier"] == "wide"].copy()
        for stop_label, risk_col, stop_desc in [
            ("wide",  "risk_wide",  "ORB low  (full range + extension)"),
            ("tight", "risk_tight", "ORB high (breakout extension only)"),
        ]:
            avg_risk = wide_s[risk_col].mean()
            print(f"\n  Stop = {stop_desc} — Wide LONG "
                  f"(n={len(wide_s)}, avg risk = {avg_risk:.2f} pts):")
            print(f"  {'Target':>16}  {'Hit tgt%':>9}  {'Hit stp%':>9}  "
                  f"{'Neither%':>9}  {'EV (R)':>8}  {'Avg mins to tgt':>16}")
            for r_mult in [1, 2, 3]:
                hc = wide_s[f"hit_{stop_label}_{r_mult}R"]
                sc = wide_s[f"stop_{stop_label}_{r_mult}R"]
                mc = wide_s[f"mins_{stop_label}_{r_mult}R"]
                p_tgt  = hc.mean()
                p_stp  = sc.mean()
                p_none = 1 - p_tgt - p_stp
                eod_r_neither = (wide_s.loc[~hc & ~sc, "fwd_eod"] /
                                 wide_s.loc[~hc & ~sc, risk_col]).mean() if p_none > 0 else 0
                ev = p_tgt * r_mult + p_stp * (-1) + p_none * eod_r_neither
                avg_mins = mc[hc].mean()
                mins_str = f"{avg_mins:.0f} min" if not np.isnan(avg_mins) else "—"
                tgt_label = f"{r_mult}R ({r_mult * avg_risk:.1f} pts)"
                print(f"  {tgt_label:>16}  "
                      f"{p_tgt*100:>8.1f}%  {p_stp*100:>8.1f}%  {p_none*100:>8.1f}%  "
                      f"{ev:>+8.3f}  {mins_str:>16}")

    # Sigma-based stop analysis: wide-ORB LONG, by time window
    TOD_WINDOWS = [
        ("Morning  9:45–10:30", 15,  60),
        ("Power hr 13:30–16:00", 240, 391),
    ]
    if len(long_res) >= 30 and "risk_sigma" in long_res.columns:
        wide_all = long_res[long_res["width_tier"] == "wide"].dropna(subset=["risk_sigma"])
        wide_all = wide_all.copy()
        wide_all["year"] = pd.to_datetime(wide_all["date"]).dt.year

        for win_label, min_lo, min_hi in TOD_WINDOWS:
            wide_s = wide_all[
                (wide_all["entry_min"] >= min_lo) &
                (wide_all["entry_min"] <  min_hi)
            ]
            if len(wide_s) < 10:
                continue
            avg_sig  = wide_s["sigma_pts"].mean()
            avg_risk = wide_s["risk_sigma"].mean()
            print(f"\n  Stop = {STOP_SIGMA}σ — Wide LONG, {win_label} ET "
                  f"(n={len(wide_s)}, avg σ = {avg_sig:.2f} pts, stop = {avg_risk:.2f} pts):")
            print(f"  {'Target':>16}  {'Hit tgt%':>9}  {'Hit stp%':>9}  "
                  f"{'Neither%':>9}  {'EV (R)':>8}  {'Avg mins to tgt':>16}")
            for r_mult in [1, 2, 3]:
                hc = wide_s[f"hit_sig_{r_mult}R"]
                sc = wide_s[f"stop_sig_{r_mult}R"]
                mc = wide_s[f"mins_sig_{r_mult}R"]
                p_tgt  = hc.mean()
                p_stp  = sc.mean()
                p_none = 1 - p_tgt - p_stp
                neither_r = (wide_s.loc[~hc & ~sc, "fwd_eod"] /
                             wide_s.loc[~hc & ~sc, "risk_sigma"])
                eod_r_n = neither_r.mean() if len(neither_r) > 0 else 0.0
                eod_r_n = 0.0 if np.isnan(eod_r_n) else eod_r_n
                ev = p_tgt * r_mult + p_stp * (-1) + p_none * eod_r_n
                avg_mins = mc[hc].mean()
                mins_str = f"{avg_mins:.0f} min" if not np.isnan(avg_mins) else "—"
                tgt_label = f"{r_mult}σ ({r_mult * avg_sig:.2f} pts)"
                print(f"  {tgt_label:>16}  "
                      f"{p_tgt*100:>8.1f}%  {p_stp*100:>8.1f}%  {p_none*100:>8.1f}%  "
                      f"{ev:>+8.3f}  {mins_str:>16}")

            # Year-by-year P&L using 2σ target (R_MULT=2): +2σ pts if hit, -risk if stopped, EOD otherwise
            R_SHOW = 2
            hc2 = wide_s[f"hit_sig_{R_SHOW}R"]
            sc2 = wide_s[f"stop_sig_{R_SHOW}R"]
            print(f"\n  Year-by-year P&L  [{R_SHOW}σ target / {STOP_SIGMA}σ stop]:")
            print(f"  {'Year':>6}  {'n':>4}  {'Win%':>6}  {'Stp%':>6}  "
                  f"{'EV(R)':>7}  {'Tot R':>7}  {'Avg σ':>7}")
            total_r = 0.0
            for yr, grp in wide_s.groupby("year"):
                if len(grp) < 3:
                    continue
                hc_y  = grp[f"hit_sig_{R_SHOW}R"]
                sc_y  = grp[f"stop_sig_{R_SHOW}R"]
                p_t   = hc_y.mean()
                p_s   = sc_y.mean()
                p_n   = 1 - p_t - p_s
                neither_yr = (grp.loc[~hc_y & ~sc_y, "fwd_eod"] /
                              grp.loc[~hc_y & ~sc_y, "risk_sigma"])
                eod_n = neither_yr.mean() if len(neither_yr) > 0 else 0.0
                eod_n = 0.0 if np.isnan(eod_n) else eod_n
                ev_yr = p_t * R_SHOW + p_s * (-1) + p_n * eod_n
                tot_r = ev_yr * len(grp)
                total_r += tot_r
                avg_s = grp["sigma_pts"].mean()
                flag = " ◄" if ev_yr > 0 else ""
                print(f"  {yr:>6}  {len(grp):>4}  {p_t*100:>5.1f}%  {p_s*100:>5.1f}%  "
                      f"{ev_yr:>+7.3f}  {tot_r:>+7.2f}R{flag}  {avg_s:>6.2f}")
            print(f"  {'TOTAL':>6}  {len(wide_s):>4}  {'':>6}  {'':>6}  "
                  f"{'':>7}  {total_r:>+7.2f}R")

    # TOD breakdown for wide-ORB LONG only
    if len(long_res) >= 30:
        wide = long_res[long_res["width_tier"] == "wide"].copy()
        if len(wide) >= 30:
            # Bucket by entry_min (minutes from 9:30 open) into hourly windows
            tod_bins   = [15, 60, 120, 180, 240, 391]   # 391 = past 16:00 catch-all
            tod_labels = ["9:45-10:30", "10:30-11:30", "11:30-12:30",
                          "12:30-13:30", "13:30-16:00"]
            wide["tod_bucket"] = pd.cut(wide["entry_min"], bins=tod_bins,
                                        labels=tod_labels, right=False)
            key_horizons = [(f"fwd_{h}", f"{h}m") for h in [15, 30, 45, 60]] + \
                           [("fwd_eod", "EOD")]
            print(f"\n  TOD breakdown — Wide ORB LONG (w > {q67:.2f} pts):")
            print(f"  {'Window':>14}  {'n':>5}  " + "  ".join(
                f"{'─ ' + lbl + ' ─':>16}" for _, lbl in key_horizons))
            print(f"  {'':>14}  {'':>5}  " + "  ".join(
                f"{'Hit%':>6} {'Avg':>7}" for _ in key_horizons))
            for lbl in tod_labels:
                grp = wide[wide["tod_bucket"] == lbl]
                if len(grp) < 5:
                    continue
                parts = []
                for col, _ in key_horizons:
                    fwd = grp[col].dropna()
                    if len(fwd) < 5:
                        parts.append(f"{'—':>6} {'—':>7}")
                    else:
                        hit = (fwd > 0).mean() * 100
                        avg = fwd.mean()
                        flag = "◄" if avg > 0 else " "
                        parts.append(f"{hit:>5.1f}% {avg:>+6.2f}{flag}")
                print(f"  {lbl:>14}  {len(grp):>5,}  " + "  ".join(parts))

    # Year-by-year breakdown at EOD for the 'ALL' subset
    print(f"\n  Year breakdown (EOD, ALL):")
    print(f"  {'Year':>6}  {'n':>5}  {'Hit%':>6}  {'Avg pts':>8}  {'ORB width':>10}")
    results["year"] = pd.to_datetime(results["date"]).dt.year
    for yr, grp in results.groupby("year"):
        fwd = grp["fwd_eod"]
        if len(fwd) < 5:
            continue
        hit = (fwd > 0).mean() * 100
        avg = fwd.mean()
        w   = grp["orb_width"].mean()
        flag = " ◄" if avg > 0 else ""
        print(f"  {yr:>6}  {len(fwd):>5,}  {hit:>5.1f}%  {avg:>+7.2f}{flag}  "
              f"{w:>10.2f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=list(HIST.keys()))
    parser.add_argument("--orb", type=int, default=None,
                        help="Single ORB period to test (default: sweep all)")
    args = parser.parse_args()

    df = load_rth(args.sym)
    periods = [args.orb] if args.orb else ORB_PERIODS

    for orb_min in periods:
        results = scan_orb(df, orb_min)
        if results.empty:
            print(f"\nNo signals for ORB {orb_min} min")
            continue
        print_orb_results(results, orb_min, args.sym)

    print()


if __name__ == "__main__":
    main()
