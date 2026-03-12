"""
Backtest: compare regime_pl vs forward volatility using polygon 5-min bar data.

For each window (30min, 1h, 2h, 3h) and each bar in the dataset:
  - compute regime_pl over the trailing window
  - compute realised volatility over the same trailing window
  - compute forward volatility over the next window (same length)

Overnight moves are excluded by filtering to regular trading hours (9:30-16:00 ET)
and dropping any window that spans a session boundary.
"""

import math
import sys
import pandas as pd
import numpy as np

BAR_MINUTES = 5  # each row is a 5-min bar
WINDOWS = {
    "30min":  30  // BAR_MINUTES,   # 6 bars
    "1hour":  60  // BAR_MINUTES,   # 12 bars
    "2hour": 120  // BAR_MINUTES,   # 24 bars
    "3hour": 180  // BAR_MINUTES,   # 36 bars
}
RTH_START = "09:30"
RTH_END   = "16:00"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["ts"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
    )
    df = df.sort_values("ts").reset_index(drop=True)

    # Regular trading hours only — eliminates all overnight moves
    time = df["ts"].dt.time
    df = df[
        (df["ts"].dt.strftime("%H:%M") >= RTH_START) &
        (df["ts"].dt.strftime("%H:%M") <  RTH_END)
    ].reset_index(drop=True)

    # Mark session boundaries (gap > 1 bar means a new day started)
    df["new_session"] = df["ts"].diff() > pd.Timedelta(minutes=BAR_MINUTES * 2)
    return df


def log_returns(closes: np.ndarray) -> np.ndarray:
    return np.log(closes[1:] / closes[:-1])


def compute_regime_pl(rets: np.ndarray) -> float:
    total_abs = np.sum(np.abs(rets))
    if total_abs == 0:
        return 0.0
    return abs(float(np.sum(rets))) / float(total_abs)


def compute_realised_vol(rets: np.ndarray) -> float:
    """Annualised realised vol from log returns (equity RTH: 6.5h/day)."""
    if len(rets) < 2:
        return float("nan")
    bars_per_year = 252 * (6.5 * 60 / BAR_MINUTES)
    return float(np.std(rets, ddof=1) * math.sqrt(bars_per_year))


def run_backtest(df: pd.DataFrame, window: int) -> pd.DataFrame:
    closes      = df["c"].values
    new_session = df["new_session"].values
    results = []

    for i in range(window, len(df) - window):
        # Reject if any session boundary falls inside the trailing or forward slice
        if new_session[i - window + 1 : i + window + 1].any():
            continue

        trailing_rets = log_returns(closes[i - window : i + 1])
        forward_rets  = log_returns(closes[i : i + window + 1])

        results.append({
            "ts":           df["ts"].iloc[i],
            "regime_pl":    compute_regime_pl(trailing_rets),
            "trailing_vol": compute_realised_vol(trailing_rets),
            "forward_vol":  compute_realised_vol(forward_rets),
        })

    return pd.DataFrame(results)


def summarise(res: pd.DataFrame, label: str):
    print(f"\n{'='*52}")
    print(f"Window: {label}  ({len(res):,} observations)")
    print(f"{'='*52}")

    print(f"Corr(regime_pl,    forward_vol): {res['regime_pl'].corr(res['forward_vol']):+.4f}")
    print(f"Corr(trailing_vol, forward_vol): {res['trailing_vol'].corr(res['forward_vol']):+.4f}")

    res = res.copy()

    res["pl_quartile"] = pd.qcut(res["regime_pl"], 4,
                                  labels=["Q1 choppy", "Q2", "Q3", "Q4 trending"])
    print("\nForward vol by regime_pl quartile:")
    print(
        res.groupby("pl_quartile", observed=True)["forward_vol"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "fwd_vol_mean", "median": "fwd_vol_med",
                         "std": "fwd_vol_std", "count": "n"})
        .to_string()
    )

    res["tv_quartile"] = pd.qcut(res["trailing_vol"], 4,
                                  labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    print("\nForward vol by trailing_vol quartile (benchmark):")
    print(
        res.groupby("tv_quartile", observed=True)["forward_vol"]
        .agg(["mean", "median", "std", "count"])
        .rename(columns={"mean": "fwd_vol_mean", "median": "fwd_vol_med",
                         "std": "fwd_vol_std", "count": "n"})
        .to_string()
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../polygon_data_QQQ.xlsx"
    df = load_data(path)
    ticker = path.split("_")[-1].replace(".xlsx", "")
    print(f"Ticker: {ticker}")
    print(f"Loaded {len(df):,} RTH bars  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    for label, window in WINDOWS.items():
        res = run_backtest(df, window)
        summarise(res, label)
