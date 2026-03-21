"""
Backtest: hi/lo range breakout on MES 1-min bars.

For each bar, define the high and low of the previous N bars (N = 20 or 40,
matching the dynamic CSR windows). A breakout occurs when the close exceeds
that range. Measures whether breakouts continue in the breakout direction.

Usage:
  python src/backtest_breakout.py              # continuation (both windows)
  python src/backtest_breakout.py --fade       # fade immediately at breakout bar
  python src/backtest_breakout.py --fade --confirm  # fade only after close re-enters range
  python src/backtest_breakout.py --window 20  # single window
"""

import argparse
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from topstep_client import TopstepClient

CACHE_CSV      = "mes_1min_bars.csv"
HIST_CSV       = "mes_hist_1min.csv"

WINDOWS       = [20, 40]          # lookback in 1-min bars (= 20 min, 40 min)
HOLD_BARS     = [5, 10, 15, 20, 30]  # forward measurement horizons
MIN_SIGNALS   = 30                # suppress results with fewer signals
CSR_THRESHOLD  = 1.5              # minimum CSR to pass filter
SIGMA_WINDOW   = 30               # bars for rolling σ estimate
CONFIRM_MAX    = 10               # max bars to wait for re-entry confirmation

# Hours (ET) where breakouts show continuation across both windows.
# Derived from TOD analysis; all others skipped.
GOOD_HOURS_ET = {4, 8, 10, 13, 15}

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22


# ── Data ─────────────────────────────────────────────────────────────────────

def load_bars() -> pd.DataFrame:
    hist = Path(HIST_CSV)
    cache = Path(CACHE_CSV)
    if hist.exists():
        print(f"Loading {HIST_CSV} …", flush=True)
        df = pd.read_csv(hist, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    elif cache.exists():
        print(f"Loading {CACHE_CSV} …", flush=True)
        df = pd.read_csv(cache, parse_dates=["ts"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    else:
        print("No cache — fetching from API …", flush=True)
        with TopstepClient() as client:
            contracts = client.search_contracts("MES")
            cid = contracts[0]["id"]
            print(f"Contract: {contracts[0]['name']}  id={cid}")
            all_bars, end = [], datetime.now(timezone.utc)
            limit = end - timedelta(days=180)
            while end > limit:
                start = end - timedelta(days=60)
                bars  = client.get_bars(cid, start, end,
                                        unit=TopstepClient.MINUTE,
                                        unit_number=1, limit=20000)
                if not bars:
                    break
                oldest = datetime.fromisoformat(bars[-1]["t"])
                all_bars.extend(bars)
                end = oldest - timedelta(minutes=1)
                print(f"  fetched {len(bars):,} back to {oldest.date()}")
                if len(bars) < 20000:
                    break

        df = pd.DataFrame(all_bars)
        df["ts"] = pd.to_datetime(df["t"], utc=True)
        df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
        df = df.rename(columns={"o": "open", "h": "high",
                                 "l": "low",  "c": "close", "v": "volume"})
        df.to_csv(cache, index=False)
        print(f"Cached → {CACHE_CSV}")

    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))]
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    # Mark session boundaries (gap > 2 min between consecutive bars)
    df["new_session"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    print(f"Bars: {len(df):,}  ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})")
    return df


# ── Breakout scanner ─────────────────────────────────────────────────────────

def scan(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    For each bar i, compute:
      range_high = max high of bars [i-n .. i-1]   (n bars, not including i)
      range_low  = min low  of bars [i-n .. i-1]

    Breakout types:
      UP   — close[i] > range_high
      DOWN — close[i] < range_low

    Also computes CSR = sum(log_returns over n bars) / sigma * direction,
    matching the live signal's momentum-alignment filter.

    Records forward close relative to close[i] at each HOLD_BARS horizon.
    Skips any window that crosses a session boundary.
    """
    highs       = df["high"].values
    lows        = df["low"].values
    closes      = df["close"].values
    new_session = df["new_session"].values
    log_rets    = np.diff(np.log(closes), prepend=np.nan)  # log_ret[i] = log(c[i]/c[i-1])
    max_hold    = max(HOLD_BARS)
    warm        = max(n, SIGMA_WINDOW)

    records = []
    for i in range(warm, len(df) - max_hold):
        # Skip if any session boundary in lookback or forward window
        if new_session[i - n + 1 : i + max_hold + 1].any():
            continue

        rh = highs[i - n : i].max()
        rl = lows [i - n : i].min()
        c  = closes[i]

        if c > rh:
            direction = 1
        elif c < rl:
            direction = -1
        else:
            continue   # inside range

        width = rh - rl
        if width <= 0:
            continue

        # CSR: cumulative momentum over the lookback window, in sigma units
        window_rets = log_rets[i - n + 1 : i + 1]   # n log returns ending at bar i
        sigma = float(np.nanstd(log_rets[i - SIGMA_WINDOW : i], ddof=1))
        if sigma <= 0 or np.isnan(sigma):
            continue
        csr = float(np.nansum(window_rets)) / sigma * direction

        row = {
            "ts":         df["ts"].iloc[i],
            "direction":  direction,
            "entry":      c,
            "range_high": rh,
            "range_low":  rl,
            "width":      width,
            "break_pts":  (c - rh) if direction == 1 else (rl - c),
            "csr":        csr,
        }
        for k in HOLD_BARS:
            fwd = closes[i + k] - c
            row[f"fwd_{k}"] = fwd * direction   # positive = continuation
        records.append(row)

    return pd.DataFrame(records)


def scan_confirmed(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Confirmed-reversion scan: like scan(), but entry is delayed until the
    first bar that closes BACK INSIDE the range after the initial breakout.

      UP breakout at bar i (close > range_high):
        Wait for bar j > i where close[j] < range_high  → entry at close[j]

      DOWN breakout at bar i (close < range_low):
        Wait for bar j > i where close[j] > range_low   → entry at close[j]

    Only one signal per breakout episode (first re-entry bar).
    Skips episodes where re-entry doesn't happen within CONFIRM_MAX bars,
    or where any bar in the window crosses a session boundary.
    Forward returns measured from the confirmation bar j.
    """
    highs       = df["high"].values
    lows        = df["low"].values
    closes      = df["close"].values
    new_session = df["new_session"].values
    log_rets    = np.diff(np.log(closes), prepend=np.nan)
    max_hold    = max(HOLD_BARS)
    warm        = max(n, SIGMA_WINDOW)

    records = []
    i = warm
    while i < len(df) - CONFIRM_MAX - max_hold:
        # ── find initial breakout at bar i ──────────────────────────────────
        if new_session[i - n + 1 : i + 1].any():
            i += 1
            continue

        rh = highs[i - n : i].max()
        rl = lows [i - n : i].min()
        c  = closes[i]

        if c > rh:
            direction = 1     # up breakout → fade = short
        elif c < rl:
            direction = -1    # down breakout → fade = long
        else:
            i += 1
            continue

        width = rh - rl
        if width <= 0:
            i += 1
            continue

        # CSR at breakout bar
        window_rets = log_rets[i - n + 1 : i + 1]
        sigma = float(np.nanstd(log_rets[i - SIGMA_WINDOW : i], ddof=1))
        if sigma <= 0 or np.isnan(sigma):
            i += 1
            continue
        csr = float(np.nansum(window_rets)) / sigma * direction

        # ── scan forward for first close back inside range ───────────────────
        confirmed_at = None
        for j in range(i + 1, min(i + CONFIRM_MAX + 1, len(df) - max_hold)):
            if new_session[j]:
                break                          # session ended, no confirmation
            if direction == 1 and closes[j] < rh:
                confirmed_at = j
                break
            if direction == -1 and closes[j] > rl:
                confirmed_at = j
                break

        if confirmed_at is None:
            i += 1
            continue

        # ── check forward window is clean ────────────────────────────────────
        if new_session[confirmed_at : confirmed_at + max_hold + 1].any():
            i += 1
            continue

        entry = closes[confirmed_at]
        row = {
            "ts":           df["ts"].iloc[confirmed_at],
            "ts_breakout":  df["ts"].iloc[i],
            "confirm_bars": confirmed_at - i,
            "direction":    direction,
            "entry":        entry,
            "range_high":   rh,
            "range_low":    rl,
            "width":        width,
            "csr":          csr,
        }
        for k in HOLD_BARS:
            fwd = closes[confirmed_at + k] - entry
            # positive = reversion (move opposite to original breakout)
            row[f"fwd_{k}"] = fwd * (-direction)
        records.append(row)

        # Skip to after confirmation bar to avoid overlapping signals
        i = confirmed_at + 1

    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _row(label: str, subset: pd.DataFrame, k: int) -> str:
    fwd = subset[f"fwd_{k}"]
    if len(fwd) < MIN_SIGNALS:
        return f"  {label:<22}  [n<{MIN_SIGNALS}]"
    hit = (fwd > 0).mean() * 100
    avg = fwd.mean()
    flag = " ◄" if avg > 0 else ""
    return (f"  {label:<22}  n={len(fwd):>5,}  {hit:>5.1f}%  {avg:>+7.3f}{flag}")


def print_results(results: pd.DataFrame, n: int):
    csr_ok  = results[results["csr"] >= CSR_THRESHOLD]
    csr_bad = results[results["csr"] <  CSR_THRESHOLD]

    total = len(results)
    up    = (results["direction"] ==  1).sum()
    dn    = (results["direction"] == -1).sum()
    pct_ok = len(csr_ok) / total * 100 if total else 0

    print(f"\n{'═'*65}")
    print(f"  Window: {n} bars ({n} min)   all signals: {total:,}  "
          f"(up={up:,}  dn={dn:,})")
    print(f"  CSR≥{CSR_THRESHOLD}: {len(csr_ok):,} ({pct_ok:.0f}%)   "
          f"CSR<{CSR_THRESHOLD}: {len(csr_bad):,} ({100-pct_ok:.0f}%)")
    print(f"{'─'*65}")

    for k in HOLD_BARS:
        print(f"\n  ── {k}-min horizon ──")
        print(_row("ALL breakouts",       results, k))
        print(_row(f"CSR≥{CSR_THRESHOLD} (with momentum)", csr_ok,  k))
        print(_row(f"CSR<{CSR_THRESHOLD} (weak momentum)", csr_bad, k))

    # Direction split with CSR filter
    print(f"\n{'─'*65}")
    print(f"  Direction × CSR split  (15-min horizon):")
    print(f"  {'Subset':<30}  {'n':>5}  {'Hit%':>6}  {'Avg pts':>8}")
    for label, mask in [
        ("UP  + CSR≥1.5",  (results["direction"]==1) & (results["csr"]>=CSR_THRESHOLD)),
        ("UP  + CSR<1.5",  (results["direction"]==1) & (results["csr"]< CSR_THRESHOLD)),
        ("DN  + CSR≥1.5",  (results["direction"]==-1) & (results["csr"]>=CSR_THRESHOLD)),
        ("DN  + CSR<1.5",  (results["direction"]==-1) & (results["csr"]< CSR_THRESHOLD)),
    ]:
        sub = results[mask]
        if len(sub) < MIN_SIGNALS:
            print(f"  {label:<30}  [n<{MIN_SIGNALS}]")
            continue
        fwd  = sub["fwd_15"]
        hit  = (fwd > 0).mean() * 100
        avg  = fwd.mean()
        flag = " ◄" if avg > 0 else ""
        print(f"  {label:<30}  {len(sub):>5,}  {hit:>5.1f}%  {avg:>+7.3f}{flag}")

    # TOD filter: only good hours
    et_all  = results["ts"].dt.tz_convert("America/New_York")
    tod_all = et_all.dt.hour
    tod_ok  = results[tod_all.isin(GOOD_HOURS_ET)]
    tod_bad = results[~tod_all.isin(GOOD_HOURS_ET)]

    print(f"\n{'─'*65}")
    good_str = ", ".join(f"{h:02d}:00" for h in sorted(GOOD_HOURS_ET))
    print(f"  TOD filter  (good hours ET: {good_str})")
    print(f"  {'Subset':<28}  {'n':>5}  {'Hit%':>6}  {'Avg pts':>8}")
    for k in [10, 15, 20]:
        print(f"\n  ── {k}-min horizon ──")
        for label, sub in [("ALL hours",   results),
                            ("Good hours", tod_ok),
                            ("Other hours", tod_bad)]:
            if len(sub) < MIN_SIGNALS:
                print(f"  {label:<28}  [n<{MIN_SIGNALS}]")
                continue
            fwd  = sub[f"fwd_{k}"]
            hit  = (fwd > 0).mean() * 100
            avg  = fwd.mean()
            flag = " ◄" if avg > 0 else ""
            print(f"  {label:<28}  {len(sub):>5,}  {hit:>5.1f}%  {avg:>+7.3f}{flag}")

    # TOD at 15-min with CSR filter
    print(f"\n{'─'*65}")
    print(f"  Time-of-day (ET) at 15-min  |  ALL vs CSR≥{CSR_THRESHOLD}:")
    print(f"  {'Hour':>6}  {'ALL n':>6}  {'ALL Hit%':>8}  {'ALL avg':>8}"
          f"    {'CSR n':>6}  {'CSR Hit%':>9}  {'CSR avg':>8}")
    et  = results["ts"].dt.tz_convert("America/New_York")
    tod = et.dt.hour
    et_csr = csr_ok["ts"].dt.tz_convert("America/New_York")
    tod_csr = et_csr.dt.hour
    for h in sorted(tod.unique()):
        sub_all = results[tod == h]
        sub_csr = csr_ok[tod_csr == h]
        if len(sub_all) < 10:
            continue
        fa  = sub_all["fwd_15"]
        fc  = sub_csr["fwd_15"] if len(sub_csr) >= 5 else None
        hit_a = (fa > 0).mean() * 100
        avg_a = fa.mean()
        flag_a = "◄" if avg_a > 0 else " "
        if fc is not None:
            hit_c = (fc > 0).mean() * 100
            avg_c = fc.mean()
            flag_c = "◄" if avg_c > 0 else " "
            csr_str = f"{len(fc):>6,}  {hit_c:>8.1f}%  {avg_c:>+8.3f} {flag_c}"
        else:
            csr_str = f"{'—':>6}  {'—':>9}  {'—':>8}"
        print(f"  {h:02d}:00 ET  {len(sub_all):>6,}  {hit_a:>7.1f}%  "
              f"{avg_a:>+7.3f} {flag_a}    {csr_str}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=None,
                        help="Lookback window in bars (default: both 20 and 40)")
    parser.add_argument("--fade", action="store_true",
                        help="Analyse the fade/reversion side (short above range_high, long below range_low)")
    parser.add_argument("--confirm", action="store_true",
                        help="With --fade: wait for close back inside range before entering")
    args = parser.parse_args()

    df = load_bars()
    windows = [args.window] if args.window else WINDOWS

    if args.fade and args.confirm:
        print(f"\n  *** CONFIRMED FADE: entry only after close re-enters range "
              f"(max {CONFIRM_MAX} bars to confirm) ***")
    elif args.fade:
        print("\n  *** FADE / REVERSION mode: entry at breakout bar close ***")

    for n in windows:
        if args.fade and args.confirm:
            results = scan_confirmed(df, n)
            if not results.empty:
                print(f"\n  Confirm delay: "
                      f"median {results['confirm_bars'].median():.0f} bar(s), "
                      f"max {results['confirm_bars'].max():.0f}")
        else:
            results = scan(df, n)
            if not results.empty and args.fade:
                for k in HOLD_BARS:
                    results[f"fwd_{k}"] = -results[f"fwd_{k}"]

        if results.empty:
            print(f"\nNo signals found for window={n}")
            continue
        print_results(results, n)

    print()


if __name__ == "__main__":
    main()
