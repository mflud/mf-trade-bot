"""
Backtest: Do rapid price spikes (≥ threshold in 1–2 min) mean-revert on MES?

Spike definition:
  - 1-bar spike:  |log(close / open)| >= THRESH in a single 1-min bar
  - 2-bar spike:  |cumulative log-return over 2 consecutive 1-min bars| >= THRESH
                  AND neither bar alone meets the threshold (distinct from above)

After detecting a spike, measure forward log-returns in the OPPOSITE direction
(i.e. directional_ret = -sign(spike) × fwd_ret, positive = reversion).

Hold windows: 1, 2, 3, 5, 10, 15, 30 min after entry at spike close.

RTH only (09:30–16:00 ET) to avoid thin-book Globex noise.

Usage:
  python src/backtest_spike_revert.py
"""

import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Parameters ────────────────────────────────────────────────────────────────

CSV_PATH   = "mes_hist_1min.csv"
ET         = ZoneInfo("America/New_York")
RTH_START  = (9, 30)
RTH_END    = (16, 0)

THRESHOLDS = [0.002, 0.003, 0.004, 0.005]   # 0.2%, 0.3%, 0.4%, 0.5%
HOLD_MINS  = [1, 2, 3, 5, 10, 15, 30]
MIN_N      = 10   # minimum trades for a result to be shown


# ── Data loading ───────────────────────────────────────────────────────────────

def load_1min_rth(csv_path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert(ET)
    df = df.reset_index().rename(columns={"ts": "ts"})
    df.columns = [c if c != df.columns[0] else "ts" for c in df.columns]
    # Rename first column safely
    df = df.rename(columns={df.columns[0]: "ts"})

    # RTH filter
    h = df["ts"].dt.hour
    m = df["ts"].dt.minute
    time_mins = h * 60 + m
    rth_start = RTH_START[0] * 60 + RTH_START[1]
    rth_end   = RTH_END[0]   * 60 + RTH_END[1]
    df = df[(time_mins >= rth_start) & (time_mins < rth_end)].copy()
    df = df.reset_index(drop=True)
    return df


# ── Spike detection ────────────────────────────────────────────────────────────

def find_spikes(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Returns DataFrame of spike events with columns:
      ts, spike_ret, bars (1 or 2), entry_price, idx
    """
    closes = df["close"].values
    opens  = df["open"].values
    ts     = df["ts"].values
    n      = len(df)

    events = []

    i = 1  # start at 1 so we can check bar i-1 for 2-bar case
    while i < n - 1:
        ret1 = math.log(closes[i] / opens[i])

        # 1-bar spike
        if abs(ret1) >= thresh:
            events.append({
                "ts":          ts[i],
                "spike_ret":   ret1,
                "bars":        1,
                "entry_price": closes[i],
                "idx":         i,
            })
            i += 2   # skip next bar to avoid overlapping events
            continue

        # 2-bar spike (two consecutive bars, neither alone ≥ thresh)
        if i + 1 < n:
            ret2 = math.log(closes[i + 1] / opens[i])
            if abs(ret2) >= thresh and abs(ret1) < thresh:
                ret_bar2 = math.log(closes[i + 1] / opens[i + 1])
                if abs(ret_bar2) < thresh:
                    events.append({
                        "ts":          ts[i + 1],
                        "spike_ret":   ret2,
                        "bars":        2,
                        "entry_price": closes[i + 1],
                        "idx":         i + 1,
                    })
                    i += 3
                    continue

        i += 1

    return pd.DataFrame(events)


# ── Forward returns ────────────────────────────────────────────────────────────

def compute_forward_returns(df: pd.DataFrame, spikes: pd.DataFrame) -> pd.DataFrame:
    closes = df["close"].values
    ts_arr = df["ts"].values
    n      = len(closes)

    records = []
    for _, row in spikes.iterrows():
        idx   = int(row["idx"])
        entry = float(row["entry_price"])
        spike = float(row["spike_ret"])

        fwd = {}
        for h in HOLD_MINS:
            j = idx + h
            if j < n:
                fwd[f"fwd_{h}"] = math.log(closes[j] / entry)
            else:
                fwd[f"fwd_{h}"] = float("nan")

        rec = {
            "ts":        row["ts"],
            "spike_ret": spike,
            "bars":      row["bars"],
            "year":      pd.Timestamp(row["ts"]).year,
        }
        rec.update(fwd)
        records.append(rec)

    return pd.DataFrame(records)


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyse(data: pd.DataFrame, hold_min: int, label: str = "all") -> dict | None:
    fwd_col = f"fwd_{hold_min}"
    sub = data.dropna(subset=[fwd_col]).copy()
    if len(sub) < MIN_N:
        return None

    direction  = np.sign(sub["spike_ret"].values)
    fwd        = sub[fwd_col].values
    rev_ret    = -direction * fwd   # positive = reversion

    sigma_all  = float(np.std(fwd))
    mean_rev   = float(np.mean(rev_ret))
    mean_sigma = mean_rev / sigma_all if sigma_all > 0 else 0.0

    return {
        "label":      label,
        "hold_min":   hold_min,
        "n":          len(sub),
        "mean_rev%":  mean_rev * 100,
        "mean_sigma": mean_sigma,
        "pct_rev":    float(np.mean(rev_ret > 0)) * 100,
    }


# ── Printing ───────────────────────────────────────────────────────────────────

def print_summary(data: pd.DataFrame, thresh: float, bars_filter=None):
    tag = f"thresh={thresh*100:.1f}%"
    if bars_filter is not None:
        sub = data[data["bars"].isin(bars_filter)].copy()
        tag += f"  bars={bars_filter}"
    else:
        sub = data.copy()

    if sub.empty:
        return

    print(f"\n{'='*70}")
    print(f"  Spike reversion  ({tag},  n_spikes={len(sub)})")
    print(f"{'='*70}")
    print(f"  {'Hold':>5}  {'n':>5}  {'mean_rev%':>9}  {'sigma':>7}  {'pct_rev%':>9}")
    print(f"  {'-'*48}")

    for h in HOLD_MINS:
        r = analyse(sub, h)
        if r is None:
            continue
        marker = " ◀" if r["mean_sigma"] > 0.05 and r["pct_rev"] > 55 else ""
        print(f"  {h:>5}  {r['n']:>5}  "
              f"{r['mean_rev%']:>+8.3f}%  "
              f"{r['mean_sigma']:>+7.3f}σ  "
              f"{r['pct_rev']:>8.1f}%"
              f"{marker}")


def print_yearly(data: pd.DataFrame, thresh: float, hold_min: int, bars_filter=None):
    tag = f"thresh={thresh*100:.1f}%  hold={hold_min}m"
    if bars_filter is not None:
        sub = data[data["bars"].isin(bars_filter)].copy()
    else:
        sub = data.copy()

    sub = sub.dropna(subset=[f"fwd_{hold_min}"]).copy()
    if sub.empty:
        return

    fwd_col   = f"fwd_{hold_min}"
    sigma_all = float(np.std(sub[fwd_col].dropna()))
    sub["rev_ret"] = -np.sign(sub["spike_ret"]) * sub[fwd_col]

    print(f"\n{'='*60}")
    print(f"  Year-by-year  ({tag})")
    print(f"{'='*60}")
    print(f"  {'Year':>5}  {'n':>5}  {'mean_sigma':>10}  {'pct_rev%':>9}")
    print(f"  {'-'*40}")
    for yr, grp in sub.groupby("year"):
        ms = grp["rev_ret"].mean() / sigma_all if sigma_all else 0
        pc = (grp["rev_ret"] > 0).mean() * 100
        print(f"  {yr:>5}  {len(grp):>5}  {ms:>+10.3f}σ  {pc:>8.1f}%")


def print_spike_distribution(data: pd.DataFrame, thresh: float):
    """How large are the spikes? Distribution of |spike_ret|."""
    print(f"\n  Spike magnitude distribution (thresh={thresh*100:.1f}%):")
    mags = data["spike_ret"].abs()
    for p in [50, 75, 90, 95, 99]:
        print(f"    p{p:2d}: {np.percentile(mags, p)*100:+.3f}%")
    print(f"    max: {mags.max()*100:+.3f}%")
    bar_counts = data["bars"].value_counts().sort_index()
    for b, cnt in bar_counts.items():
        print(f"    {b}-bar spikes: {cnt}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {CSV_PATH} (RTH only) …")
    df = load_1min_rth()
    print(f"  {len(df):,} RTH 1-min bars  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    for thresh in THRESHOLDS:
        spikes = find_spikes(df, thresh)
        if spikes.empty:
            print(f"\n  No spikes at {thresh*100:.1f}%")
            continue

        data = compute_forward_returns(df, spikes)
        print(f"\n{'#'*70}")
        print(f"  THRESHOLD = {thresh*100:.1f}%   total spikes = {len(spikes)}")
        print(f"{'#'*70}")
        print_spike_distribution(data, thresh)

        # All spikes combined (1 and 2-bar)
        print_summary(data, thresh)

        # 1-bar only
        print_summary(data, thresh, bars_filter=[1])

        # Year-by-year for the main hold windows (5 and 10 min)
        for hold in [5, 10]:
            print_yearly(data, thresh, hold)
