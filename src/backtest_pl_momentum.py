"""
Backtest: Path-Length + Strong-Bar Momentum signal on MES 1-min bars.

For each bar, looks back N 1-min bars and computes two conditions:
  1. M/N  > mn_thresh  — fraction of bars with |log_return| > 0.5σ in signal direction
  2. |PL| > pl_thresh  — signed path length = Σr / Σ|r|, direction matches signal

σ is the trailing 100-bar log-return std (in price points).
Signal direction = sign(PL).

Forward window K = N // 10 bars.
Records forward K-bar cumulative return (scaled by σ) and whether it's same-sign.

Sweeps:
  N          : 10, 20, 30, 50
  mn_thresh  : 0.50, 0.60
  pl_thresh  : 0.50, 0.70, 0.80

Usage:
  python src/backtest_pl_momentum.py          # MES, all sweeps
  python src/backtest_pl_momentum.py --sym MYM
  python src/backtest_pl_momentum.py --n 20   # single N
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

SIGMA_WINDOW  = 100          # bars for trailing σ
STRONG_THRESH = 0.5          # |r| > STRONG_THRESH * σ to count as "strong bar"
MIN_SIGNALS   = 30           # suppress cells with fewer signals

N_VALUES  = [10, 20, 30, 50]
MN_THRESH = [0.50, 0.60]
PL_THRESH = [0.50, 0.70, 0.80]

HIST = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_rth(sym: str) -> pd.DataFrame:
    path = Path(HIST[sym])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    print(f"Loading {path} …", flush=True)
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    ts_et    = df["ts"].dt.tz_convert(ET)
    rth_mask = (
        (ts_et.dt.time >= RTH_OPEN) &
        (ts_et.dt.time <  RTH_CLOSE)
    )
    df = df[rth_mask].copy()
    df["ts_et"] = ts_et[rth_mask]
    df["date"]  = df["ts_et"].dt.date
    df = df.reset_index(drop=True)

    # Trailing σ (price points) over SIGMA_WINDOW bars
    log_rets       = np.log(df["close"] / df["close"].shift(1))
    sigma          = log_rets.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW).std(ddof=1)
    df["log_ret"]  = log_rets
    df["sigma_pts"] = sigma * df["close"]

    print(f"RTH bars: {len(df):,}  ({df['date'].iloc[0]} → {df['date'].iloc[-1]})",
          flush=True)
    return df


# ── Signal scanner ─────────────────────────────────────────────────────────────

def scan(df: pd.DataFrame, N: int) -> pd.DataFrame:
    """
    For every bar i (with enough history), compute:
      - signed PL over [i-N+1 .. i]
      - M/N fraction of bars with |r| > 0.5σ in the signal direction
      - forward K = N//10 bar cumulative return (scaled by σ)

    Only emits one signal per bar (long or short based on PL sign).
    Requires: σ available (i >= SIGMA_WINDOW) and i+K < len(df) and same RTH session.
    """
    K       = max(1, N // 10)
    closes  = df["close"].values
    log_ret = df["log_ret"].values
    sig_pts = df["sigma_pts"].values
    dates   = df["date"].values

    records = []
    # need SIGMA_WINDOW bars of history before we start, plus N lookback
    start = max(SIGMA_WINDOW, N)

    for i in range(start, len(df) - K):
        # forward window must stay in same session
        if dates[i + K] != dates[i]:
            continue

        sp = sig_pts[i]
        if np.isnan(sp) or sp <= 0:
            continue

        window = log_ret[i - N + 1 : i + 1]   # N bars ending at i (inclusive)
        if np.any(np.isnan(window)):
            continue

        sum_r   = window.sum()
        sum_absr = np.abs(window).sum()
        if sum_absr == 0:
            continue

        pl = sum_r / sum_absr           # signed path length ∈ [-1, 1]
        direction = 1 if pl >= 0 else -1

        # Count strong bars in signal direction
        strong = np.sum((window * direction) > STRONG_THRESH * sp / closes[i])
        mn     = strong / N

        # Forward K-bar cumulative return scaled by σ (direction-adjusted)
        fwd_raw = np.log(closes[i + K] / closes[i])
        fwd_sig = fwd_raw * direction / (sp / closes[i])   # in σ units

        records.append({
            "i":         i,
            "direction": direction,
            "pl":        pl,
            "abs_pl":    abs(pl),
            "mn":        mn,
            "sigma_pts": sp,
            "fwd_sig":   fwd_sig,                          # forward return in σ
            "fwd_pts":   fwd_raw * closes[i] * direction,  # forward return in pts
            "year":      pd.Timestamp(df["ts_et"].iloc[i]).year,
        })

    return pd.DataFrame(records)


# ── Reporting ──────────────────────────────────────────────────────────────────

def _stats(fwd: pd.Series) -> dict:
    n    = len(fwd)
    hit  = (fwd > 0).mean() * 100 if n > 0 else float("nan")
    ev   = fwd.mean() if n > 0 else float("nan")
    cond = fwd[fwd > 0].mean() if (fwd > 0).any() else float("nan")
    return {"n": n, "hit": hit, "ev": ev, "cond": cond}


def print_results(all_records: pd.DataFrame, N: int, sym: str):
    K = max(1, N // 10)
    print(f"\n{'═'*72}")
    print(f"  {sym}  N={N} bars lookback  K={K} bars forward  "
          f"(σ window={SIGMA_WINDOW})")
    print(f"{'─'*72}")

    # Baseline: no conditions (all bars with valid data)
    base = all_records["fwd_sig"]
    bs   = _stats(base)
    print(f"\n  Baseline (no filter):  "
          f"n={bs['n']:,}  hit={bs['hit']:.1f}%  "
          f"EV={bs['ev']:+.4f}σ  E[fwd|cont]={bs['cond']:+.4f}σ")

    # Header
    print(f"\n  {'MN>':>6}  {'|PL|>':>6}  "
          f"{'n':>6}  {'Hit%':>6}  {'EV(σ)':>8}  "
          f"{'E[fwd|cont]':>12}  {'Reduce%':>8}")
    print(f"  {'-'*6}  {'-'*6}  "
          f"{'-'*6}  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*8}")

    for mn_t in MN_THRESH:
        for pl_t in PL_THRESH:
            sub = all_records[
                (all_records["mn"]     > mn_t) &
                (all_records["abs_pl"] > pl_t)
            ]["fwd_sig"]
            s = _stats(sub)
            if s["n"] < MIN_SIGNALS:
                print(f"  {mn_t:.2f}   {pl_t:.2f}   "
                      f"{'<30':>6}  {'—':>6}  {'—':>8}  {'—':>12}  {'—':>8}")
                continue
            reduce = (1 - s["n"] / bs["n"]) * 100
            flag   = " ◄" if s["ev"] > 0.02 else ""
            print(f"  {mn_t:.2f}   {pl_t:.2f}   "
                  f"{s['n']:>6,}  {s['hit']:>5.1f}%  "
                  f"{s['ev']:>+8.4f}  "
                  f"{s['cond']:>+12.4f}  "
                  f"{reduce:>7.1f}%{flag}")

    # Year-by-year for the best combo (mn=0.5, pl=0.7 as representative)
    for mn_t, pl_t in [(0.50, 0.70), (0.60, 0.70)]:
        sub = all_records[
            (all_records["mn"]     > mn_t) &
            (all_records["abs_pl"] > pl_t)
        ]
        if len(sub) < MIN_SIGNALS:
            continue
        print(f"\n  Year-by-year  [MN>{mn_t:.2f}, |PL|>{pl_t:.2f}]:")
        print(f"  {'Year':>6}  {'n':>5}  {'Hit%':>6}  {'EV(σ)':>8}  "
              f"{'Tot σ':>8}  {'Avg σ/sig':>10}")
        total = 0.0
        for yr, grp in sub.groupby("year"):
            if len(grp) < 5:
                continue
            s = _stats(grp["fwd_sig"])
            tot = s["ev"] * s["n"]
            total += tot
            flag = " ◄" if s["ev"] > 0 else ""
            print(f"  {yr:>6}  {s['n']:>5,}  {s['hit']:>5.1f}%  "
                  f"{s['ev']:>+8.4f}  {tot:>+8.2f}  "
                  f"{s['cond']:>+10.4f}{flag}")
        print(f"  {'TOTAL':>6}  {len(sub):>5,}  "
              f"{'':>6}  {'':>8}  {total:>+8.2f}")

    # Direction breakdown at best combo
    sub = all_records[
        (all_records["mn"]     > 0.50) &
        (all_records["abs_pl"] > 0.70)
    ]
    if len(sub) >= MIN_SIGNALS:
        long_s  = _stats(sub[sub["direction"] ==  1]["fwd_sig"])
        short_s = _stats(sub[sub["direction"] == -1]["fwd_sig"])
        print(f"\n  Direction split  [MN>0.50, |PL|>0.70]:")
        print(f"  {'':>8}  {'n':>6}  {'Hit%':>6}  {'EV(σ)':>8}  {'E[fwd|cont]':>12}")
        for label, s in [("LONG", long_s), ("SHORT", short_s)]:
            if s["n"] < MIN_SIGNALS:
                continue
            print(f"  {label:>8}  {s['n']:>6,}  {s['hit']:>5.1f}%  "
                  f"{s['ev']:>+8.4f}  {s['cond']:>+12.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=list(HIST.keys()))
    parser.add_argument("--n",   type=int, default=None,
                        help="Single N value to test (default: sweep all)")
    args = parser.parse_args()

    df      = load_rth(args.sym)
    ns      = [args.n] if args.n else N_VALUES

    for N in ns:
        print(f"\nScanning N={N} …", flush=True)
        records = scan(df, N)
        if records.empty:
            print(f"  No signals for N={N}")
            continue
        print_results(records, N, args.sym)

    print()


if __name__ == "__main__":
    main()
