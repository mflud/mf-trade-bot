"""
Backtest: Weekday evening Globex resumption gap momentum for MES.

CME pauses MES trading from 17:00–18:00 ET daily (settlement gap).
At 18:00 ET, Globex resumes. On days with significant post-close news
(earnings, Fed, macro), the price at resumption can differ meaningfully
from the 16:00 close.

Strategy:
  - At the 17:00 ET resumption, measure the gap from the 16:00 ET close.
  - If |gap| >= threshold AND first-candle volume >= vol_mult × rolling
    median of prior N evening opens, enter in the direction of the gap.
  - Hold for HOLD_BARS × 5 min, no stop/target.
  - Measure mean scaled return and P(continuation).

Gap definitions:
  GAP_OPEN  — gap from 16:00 close to 17:00 first-bar OPEN
  GAP_CLOSE — gap from 16:00 close to 17:00 first-bar CLOSE (confirmed)

Data source: mes_hist_1min.csv (Databento continuous, 2019–2026).
Note: ~4 roll weeks/year may show spurious gaps; not filtered.

Usage:
  python src/backtest_evening_globex.py
"""

import math
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────

CSV_PATH     = "mes_hist_1min.csv"
BAR_MINUTES  = 5
ET           = ZoneInfo("America/New_York")

HOLD_BARS    = [6, 12, 24, 48]                          # 30m, 60m, 120m, 240m
GAP_THRESH   = [0.0005, 0.001, 0.002, 0.003, 0.005]    # 0.05% … 0.5%
VOL_MULTS    = [0.0, 0.5, 1.0, 1.5]
VOL_LOOKBACK = 20    # more history than Sunday (daily events, not weekly)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_5min(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load 1-min CSV, resample to 5-min, return with ET timestamps."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts"

    # Keep bars aligned to 5-min boundaries only (matches backtest convention)
    df = df[df.index.minute % BAR_MINUTES == 0]

    df5 = df.resample(f"{BAR_MINUTES}min", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    df5.index = df5.index.tz_convert(ET)
    df5 = df5.reset_index()
    df5 = df5.rename(columns={"ts": "ts"})
    return df5


# ── Evening session identification ───────────────────────────────────────────

def find_evening_opens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per weekday 17:00 ET Globex resumption.

    The resumption bar is the first 5-min bar at or after 17:00 ET that
    follows a gap from a bar at 15:55–16:00 ET (i.e. the last RTH bar).
    We detect this as bars where:
      - ET hour == 18, minute == 0
      - Day of week is Mon–Fri (0–4)
      - Preceded by a gap of at least 55 minutes (the settlement hour)

    Columns:
      ts_open    — timestamp of the 17:00 resumption bar
      open       — open of that bar
      close      — close of that bar
      volume     — volume of that bar
      prev_close — close of the last bar before the gap (16:00 ET)
      idx_open   — integer index in df
    """
    rows = []
    for i in range(1, len(df)):
        bar  = df.iloc[i]
        prev = df.iloc[i - 1]

        ts   = bar["ts"]
        # Must be 17:00 ET on a weekday
        if ts.hour != 18 or ts.minute != 0:
            continue
        if ts.weekday() > 4:   # Sat=5, Sun=6
            continue

        # Gap from previous bar must be ≥ 55 min (the settlement gap)
        gap = ts - prev["ts"]
        if gap < timedelta(minutes=55):
            continue

        rows.append({
            "ts_open":    ts,
            "open":       float(bar["open"]),
            "close":      float(bar["close"]),
            "volume":     float(bar["volume"]),
            "prev_close": float(prev["close"]),
            "idx_open":   i,
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Forward returns ───────────────────────────────────────────────────────────

def compute_forward_returns(df: pd.DataFrame,
                             opens: pd.DataFrame,
                             max_hold: int) -> pd.DataFrame:
    """
    For each evening open, compute forward log-returns at each hold step.
    Entry = close of the first 17:00 bar (gap_close entry).
    Also computes gap_open and gap_close in log-return units.
    """
    records = []
    for _, row in opens.iterrows():
        entry     = row["close"]
        prev_cl   = row["prev_close"]
        gap_open  = math.log(row["open"]  / prev_cl)
        gap_close = math.log(entry        / prev_cl)

        # Time-based exit lookup
        fwd = {}
        for h in HOLD_BARS:
            exit_ts = row["ts_open"] + timedelta(minutes=h * BAR_MINUTES)
            later   = df[df["ts"] >= exit_ts]
            fwd[h]  = math.log(float(later.iloc[0]["close"]) / entry) \
                      if len(later) > 0 else float("nan")

        rec = {
            "ts_open":   row["ts_open"],
            "gap_open":  gap_open,
            "gap_close": gap_close,
            "vol":       row["volume"],
            "prev_close": prev_cl,
            "dow":       row["ts_open"].weekday(),   # 0=Mon … 4=Fri
        }
        rec.update({f"fwd_{h}": v for h, v in fwd.items()})
        records.append(rec)

    return pd.DataFrame(records)


def add_vol_baseline(df: pd.DataFrame,
                     lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    """Rolling median of prior N evening open first-candle volumes."""
    df = df.copy()
    df["vol_med"] = (df["vol"]
                     .shift(1)
                     .rolling(lookback, min_periods=5)
                     .median())
    return df


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(data: pd.DataFrame,
                 gap_col: str,
                 gap_thresh: float,
                 vol_mult: float,
                 hold_bars: int,
                 dow_filter: list | None = None) -> dict | None:
    """
    Filter by gap + volume, compute directional forward returns.
    dow_filter: list of weekday ints (0=Mon…4=Fri) to restrict; None = all.
    """
    fwd_col = f"fwd_{hold_bars}"
    sub = data.dropna(subset=[fwd_col, "vol_med"]).copy()

    if dow_filter is not None:
        sub = sub[sub["dow"].isin(dow_filter)]

    sub = sub[sub[gap_col].abs() >= gap_thresh]
    if len(sub) < 10:
        return None

    if vol_mult > 0:
        sub = sub[sub["vol"] >= vol_mult * sub["vol_med"]]
    if len(sub) < 10:
        return None

    direction     = np.sign(sub[gap_col].values)
    fwd           = sub[fwd_col].values
    dir_ret       = direction * fwd

    sigma_all  = float(np.std(data[fwd_col].dropna()))
    mean_sigma = float(np.mean(dir_ret)) / sigma_all if sigma_all > 0 else 0.0

    return {
        "gap_col":       gap_col,
        "gap_thresh":    gap_thresh,
        "vol_mult":      vol_mult,
        "hold_bars":     hold_bars,
        "hold_min":      hold_bars * BAR_MINUTES,
        "dow":           "all" if dow_filter is None else
                         "".join("MTWRF"[d] for d in sorted(dow_filter)),
        "n":             len(sub),
        "mean_ret_pct":  float(np.mean(dir_ret)) * 100,
        "mean_sigma":    mean_sigma,
        "pct_cont":      float(np.mean(dir_ret > 0)) * 100,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results: list[dict], min_n: int = 20):
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)

    for gap_col in ("gap_open", "gap_close"):
        print(f"\n{'='*82}")
        print(f"  Gap definition: {gap_col.upper()}  (all weekdays, n ≥ {min_n})")
        print(f"{'='*82}")
        sub = df[(df["gap_col"] == gap_col) & (df["dow"] == "all")].copy()
        if sub.empty:
            continue

        for hold_min in sorted(sub["hold_min"].unique()):
            hdf = sub[sub["hold_min"] == hold_min].sort_values(["gap_thresh", "vol_mult"])
            print(f"\n  Hold: {hold_min:3d}min")
            print(f"  {'gap%':>6}  {'vol×':>5}  {'n':>5}  "
                  f"{'mean_ret%':>9}  {'sigma':>7}  {'pct_cont%':>9}")
            print(f"  {'-'*57}")
            for _, r in hdf.iterrows():
                if r["n"] < min_n:
                    continue
                marker = " ◀" if (r["mean_sigma"] > 0.05 and r["pct_cont"] > 55) else ""
                print(f"  {r['gap_thresh']*100:>5.2f}%  "
                      f"{r['vol_mult']:>5.1f}×  "
                      f"{r['n']:>5.0f}  "
                      f"{r['mean_ret_pct']:>+8.3f}%  "
                      f"{r['mean_sigma']:>+7.3f}σ  "
                      f"{r['pct_cont']:>8.1f}%"
                      f"{marker}")


def print_dow_breakdown(results: list[dict], gap_col: str = "gap_open",
                        gap_thresh: float = 0.002, vol_mult: float = 1.0,
                        hold_bars: int = 6):
    """Show per-day-of-week breakdown for a given parameter combo."""
    df = pd.DataFrame(results)
    sub = df[
        (df["gap_col"]    == gap_col) &
        (df["gap_thresh"] == gap_thresh) &
        (df["vol_mult"]   == vol_mult) &
        (df["hold_bars"]  == hold_bars)
    ].copy()

    if sub.empty:
        return

    print(f"\n{'='*60}")
    print(f"  Day-of-week breakdown  "
          f"(gap={gap_thresh*100:.2f}% vol={vol_mult}× hold={hold_bars*BAR_MINUTES}m)")
    print(f"{'='*60}")
    print(f"  {'Day':>5}  {'n':>5}  {'mean_sigma':>10}  {'pct_cont%':>9}")
    print(f"  {'-'*40}")
    day_names = {"M":"Mon","T":"Tue","W":"Wed","R":"Thu","F":"Fri","all":"All"}
    for _, r in sub.sort_values("dow").iterrows():
        name = day_names.get(r["dow"], r["dow"])
        print(f"  {name:>5}  {r['n']:>5.0f}  "
              f"{r['mean_sigma']:>+10.3f}σ  "
              f"{r['pct_cont']:>8.1f}%")


def print_yearly_breakdown(data: pd.DataFrame,
                            gap_col: str = "gap_open",
                            gap_thresh: float = 0.002,
                            vol_mult: float = 1.0,
                            hold_bars: int = 6):
    """Year-by-year stability check for a given parameter combo."""
    fwd_col = f"fwd_{hold_bars}"
    sub = data.dropna(subset=[fwd_col, "vol_med"]).copy()
    sub = sub[sub[gap_col].abs() >= gap_thresh]
    if vol_mult > 0:
        sub = sub[sub["vol"] >= vol_mult * sub["vol_med"]]
    if sub.empty:
        return

    sigma_all = float(np.std(data[fwd_col].dropna()))
    sub = sub.copy()
    sub["dir_ret"] = np.sign(sub[gap_col].values) * sub[fwd_col].values
    sub["year"] = sub["ts_open"].dt.year

    print(f"\n{'='*60}")
    print(f"  Year-by-year  "
          f"(gap={gap_thresh*100:.2f}% vol={vol_mult}× hold={hold_bars*BAR_MINUTES}m)")
    print(f"{'='*60}")
    print(f"  {'Year':>5}  {'n':>5}  {'mean_sigma':>10}  {'pct_cont%':>9}")
    print(f"  {'-'*40}")
    for yr, grp in sub.groupby("year"):
        ms = grp["dir_ret"].mean() / sigma_all if sigma_all else 0
        pc = (grp["dir_ret"] > 0).mean() * 100
        print(f"  {yr:>5}  {len(grp):>5}  {ms:>+10.3f}σ  {pc:>8.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {CSV_PATH} and resampling to {BAR_MINUTES}-min (ET) …")
    df = load_5min()
    print(f"  {len(df):,} bars  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    print("\nFinding 17:00 ET evening resumption bars …")
    opens = find_evening_opens(df)
    print(f"  {len(opens)} evening sessions found")

    data = compute_forward_returns(df, opens, max(HOLD_BARS))
    data = add_vol_baseline(data)

    # Quick distribution of gap sizes
    print(f"\n  Gap (open) distribution:")
    for thr in GAP_THRESH:
        n = (data["gap_open"].abs() >= thr).sum()
        print(f"    ≥ {thr*100:.2f}%:  {n:4d}  ({n/len(data)*100:.0f}%)")

    print("\nRunning parameter sweep …")
    results = []

    # All weekdays combined
    for gap_col in ("gap_open", "gap_close"):
        for gap_thresh in GAP_THRESH:
            for vol_mult in VOL_MULTS:
                for hold_bars in HOLD_BARS:
                    r = run_backtest(data, gap_col, gap_thresh, vol_mult, hold_bars)
                    if r:
                        results.append(r)

    # Per day-of-week
    for dow in range(5):
        for gap_col in ("gap_open", "gap_close"):
            for gap_thresh in GAP_THRESH:
                for vol_mult in VOL_MULTS:
                    for hold_bars in HOLD_BARS:
                        r = run_backtest(data, gap_col, gap_thresh, vol_mult,
                                         hold_bars, dow_filter=[dow])
                        if r:
                            results.append(r)

    print_results(results)

    # Day-of-week breakdown for the most interesting combo
    print_dow_breakdown(results, gap_col="gap_open", gap_thresh=0.002,
                        vol_mult=1.0, hold_bars=6)
    print_dow_breakdown(results, gap_col="gap_open", gap_thresh=0.003,
                        vol_mult=1.5, hold_bars=6)

    # Year-by-year stability
    print_yearly_breakdown(data, gap_col="gap_open", gap_thresh=0.002,
                           vol_mult=1.0, hold_bars=6)

    # Top combos
    rdf = pd.DataFrame(results)
    best = (rdf[(rdf["n"] >= 20) & (rdf["dow"] == "all")]
            .sort_values("mean_sigma", ascending=False)
            .head(12))
    if not best.empty:
        print(f"\n{'='*82}")
        print("  Top 12 combos  (n ≥ 20, all weekdays)")
        print(f"{'='*82}")
        print(best[["gap_col", "gap_thresh", "vol_mult",
                     "hold_min", "n", "mean_sigma", "pct_cont"]].to_string(index=False))
