"""
Backtest: Sunday Globex open gap momentum for MES.

Strategy:
  - At the Sunday Globex open (first 5-min bar ~18:00 ET), measure the gap
    from Friday's last close.
  - If |gap| >= threshold AND first-candle volume >= vol_mult × rolling median
    of prior Sunday open volumes, enter in the direction of the gap.
  - Hold for N bars (hold_bars × 5 min), no stop/target.
  - Measure mean scaled return and P(continuation).

Gap definitions (both swept):
  GAP_OPEN  — gap from Friday close to Sunday first-bar OPEN  (gap before any trade)
  GAP_CLOSE — gap from Friday close to Sunday first-bar CLOSE (first candle confirms)

Data source: mes_hist_1min.csv (Databento continuous front-month, 2019–2026).
Note: ~4 roll weekends/year may show spurious gaps in the continuous series;
these are flagged but not filtered.

Usage:
  python src/backtest_sunday_globex.py
"""

import numpy as np
import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────

CSV_PATH      = "mes_hist_1min.csv"
BAR_MINUTES   = 5
HOLD_BARS     = [6, 12, 24, 48]          # 30m, 60m, 120m, 240m
GAP_THRESH    = [0.0005, 0.001, 0.002, 0.003, 0.005]   # 0.05% … 0.5%
VOL_MULTS     = [0.0, 0.5, 1.0, 1.5]    # 0.0 = no volume filter
VOL_LOOKBACK  = 8                         # prior Sunday opens for median baseline

# CME quarterly roll months (March=3, June=6, Sep=9, Dec=12)
ROLL_MONTHS   = {3, 6, 9, 12}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_5min(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load 1-min CSV and resample to 5-min OHLCV."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts"

    # Drop CME settlement gap: 21:00–22:00 UTC
    df = df[~((df.index.hour == 21) & (df.index.minute < 60))]
    # Keep only bars that align to 5-min boundaries (minute % 5 == 0)
    df = df[df.index.minute % BAR_MINUTES == 0]

    # Resample to 5-min (aggregating 1-min bars within each 5-min window)
    df5 = df.resample(f"{BAR_MINUTES}min", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()

    df5 = df5.reset_index()
    df5 = df5.rename(columns={"ts": "ts"})
    return df5


# ── Sunday-open identification ────────────────────────────────────────────────

def find_sunday_opens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per Sunday Globex open session.

    A Sunday open is defined as the first bar after a gap > 24 hours where
    the prior session ended on a Friday or Saturday (weekday 4 or 5).

    Columns returned:
      ts_open   — timestamp of the first Sunday bar
      open      — open price of that bar
      close     — close price of that bar
      volume    — volume of that bar
      fri_close — close of the last bar of the preceding Friday session
      idx_open  — integer index in df of the Sunday open bar
      is_roll   — True if this weekend coincides with a quarterly roll expiry
    """
    # Detect gaps > 24 hours — specifically targets the Sunday weekend gap
    gap = df["ts"].diff() > pd.Timedelta(hours=24)
    session_starts = df[gap].index.tolist()

    rows = []
    for idx in session_starts:
        bar = df.iloc[idx]
        dow = bar["ts"].weekday()   # Mon=0 … Sun=6
        if dow != 6:
            continue

        prev_idx = idx - 1
        if prev_idx < 0:
            continue
        prev_bar = df.iloc[prev_idx]
        prev_dow = prev_bar["ts"].weekday()
        if prev_dow not in (4, 5):
            continue

        # Flag roll weekends (3rd Friday expiry months)
        is_roll = (prev_bar["ts"].month in ROLL_MONTHS)

        rows.append({
            "ts_open":   bar["ts"],
            "open":      bar["open"],
            "close":     bar["close"],
            "volume":    bar["volume"],
            "fri_close": prev_bar["close"],
            "idx_open":  idx,
            "is_roll":   is_roll,
        })

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Forward return computation ────────────────────────────────────────────────

def compute_forward_returns(df: pd.DataFrame,
                             opens: pd.DataFrame,
                             max_hold: int) -> pd.DataFrame:
    """
    For each Sunday open, add forward log-returns at each hold window.
    Entry price = close of the first candle (i.e. confirmed gap_close entry).
    Also computes gap_open and gap_close in log-return terms.
    """
    closes = df["close"].values
    n      = len(closes)

    records = []
    for _, row in opens.iterrows():
        i        = row["idx_open"]
        entry    = row["close"]           # enter at close of first candle
        fri_cl   = row["fri_close"]

        gap_open  = float(np.log(row["open"]  / fri_cl))  # pre-candle gap
        gap_close = float(np.log(entry        / fri_cl))  # post-candle gap

        fwd = {}
        for h in range(1, max_hold + 1):
            j = i + h
            if j >= n:
                fwd[h] = np.nan
            else:
                fwd[h] = float(np.log(closes[j] / entry))

        rec = {
            "ts_open":   row["ts_open"],
            "gap_open":  gap_open,
            "gap_close": gap_close,
            "vol":       row["volume"],
            "fri_close": fri_cl,
        }
        rec.update({f"fwd_{h}": v for h, v in fwd.items()})
        records.append(rec)

    return pd.DataFrame(records)


# ── Volume baseline ───────────────────────────────────────────────────────────

def add_vol_baseline(df: pd.DataFrame, lookback: int = VOL_LOOKBACK) -> pd.DataFrame:
    """Add rolling median of prior N Sunday-open volumes as vol_med column."""
    df = df.copy()
    df["vol_med"] = (df["vol"]
                     .shift(1)
                     .rolling(lookback, min_periods=3)
                     .median())
    return df


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(data: pd.DataFrame,
                 gap_col: str,
                 gap_thresh: float,
                 vol_mult: float,
                 hold_bars: int) -> dict | None:
    """
    Filter events by gap threshold + volume filter, then score forward return
    directionally (gap direction × forward return = positive means continuation).
    """
    fwd_col = f"fwd_{hold_bars}"
    sub = data.dropna(subset=[fwd_col, "vol_med"]).copy()

    # Gap threshold filter
    sub = sub[sub[gap_col].abs() >= gap_thresh]
    if len(sub) < 4:
        return None

    # Volume filter
    if vol_mult > 0:
        sub = sub[sub["vol"] >= vol_mult * sub["vol_med"]]
    if len(sub) < 4:
        return None

    direction = np.sign(sub[gap_col].values)
    fwd       = sub[fwd_col].values
    directional_ret = direction * fwd      # positive = continuation

    n        = len(sub)
    mean_ret = float(np.mean(directional_ret))
    pct_cont = float(np.mean(directional_ret > 0))
    # Express mean_ret in σ of the unconditional distribution
    sigma_all = float(np.std(data[fwd_col].dropna()))
    mean_sigma = mean_ret / sigma_all if sigma_all > 0 else 0.0

    return {
        "gap_col":    gap_col,
        "gap_thresh": gap_thresh,
        "vol_mult":   vol_mult,
        "hold_bars":  hold_bars,
        "hold_min":   hold_bars * BAR_MINUTES,
        "n":          n,
        "mean_ret_pct":  mean_ret * 100,
        "mean_sigma":    mean_sigma,
        "pct_cont":      pct_cont * 100,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results: list[dict]):
    if not results:
        print("No results.")
        return

    df = pd.DataFrame(results)

    for gap_col in ("gap_open", "gap_close"):
        print(f"\n{'='*80}")
        print(f"  Gap definition: {gap_col.upper()}")
        print(f"{'='*80}")

        sub = df[df["gap_col"] == gap_col].copy()
        if sub.empty:
            continue

        for hold_min in sorted(sub["hold_min"].unique()):
            hdf = sub[sub["hold_min"] == hold_min].copy()
            hdf = hdf.sort_values(["gap_thresh", "vol_mult"])

            print(f"\n  Hold: {hold_min:3d}min")
            print(f"  {'gap%':>6}  {'vol×':>5}  {'n':>4}  "
                  f"{'mean_ret%':>9}  {'sigma':>7}  {'pct_cont%':>9}")
            print(f"  {'-'*55}")
            for _, r in hdf.iterrows():
                marker = " ◀" if (r["mean_sigma"] > 0.05 and r["n"] >= 10
                                   and r["pct_cont"] > 55) else ""
                print(f"  {r['gap_thresh']*100:>5.2f}%  "
                      f"{r['vol_mult']:>5.1f}×  "
                      f"{r['n']:>4.0f}  "
                      f"{r['mean_ret_pct']:>+8.3f}%  "
                      f"{r['mean_sigma']:>+7.3f}σ  "
                      f"{r['pct_cont']:>8.1f}%"
                      f"{marker}")


def print_event_list(data: pd.DataFrame):
    """Print the raw Sunday open events with gap and volume info."""
    print(f"\n{'='*80}")
    print(f"  Sunday Globex Opens ({len(data)} events)")
    print(f"{'='*80}")
    print(f"  {'Date':>16}  {'fri_close':>9}  {'gap_open%':>9}  "
          f"{'gap_close%':>10}  {'volume':>7}  {'vol_med':>7}  roll")
    print(f"  {'-'*72}")
    for _, r in data.iterrows():
        vm = f"{r['vol_med']:.0f}" if pd.notna(r.get("vol_med")) else "  n/a"
        roll = " ★" if r.get("is_roll") else ""
        print(f"  {str(r['ts_open'])[:16]:>16}  "
              f"{r['fri_close']:>9.2f}  "
              f"{r['gap_open']*100:>+8.3f}%  "
              f"{r['gap_close']*100:>+9.3f}%  "
              f"{r['vol']:>7.0f}  "
              f"{vm:>7}"
              f"{roll}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {CSV_PATH} and resampling to {BAR_MINUTES}-min …")
    df = load_5min()
    print(f"  {len(df):,} bars  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    print("\nFinding Sunday opens...")
    opens = find_sunday_opens(df)
    print(f"  {len(opens)} Sunday sessions found")

    max_hold = max(HOLD_BARS)
    data     = compute_forward_returns(df, opens, max_hold)
    data     = add_vol_baseline(data)

    print_event_list(data)

    print("\nRunning parameter sweep...")
    results = []
    for gap_col in ("gap_open", "gap_close"):
        for gap_thresh in GAP_THRESH:
            for vol_mult in VOL_MULTS:
                for hold_bars in HOLD_BARS:
                    r = run_backtest(data, gap_col, gap_thresh, vol_mult, hold_bars)
                    if r:
                        results.append(r)

    print_results(results)

    # Best combos summary
    if results:
        rdf = pd.DataFrame(results)
        best = (rdf[rdf["n"] >= 8]
                .sort_values("mean_sigma", ascending=False)
                .head(10))
        if not best.empty:
            print(f"\n{'='*80}")
            print("  Top 10 combos by mean_sigma  (n ≥ 8)")
            print(f"{'='*80}")
            print(best[["gap_col", "gap_thresh", "vol_mult",
                         "hold_min", "n", "mean_sigma", "pct_cont"]].to_string(index=False))
