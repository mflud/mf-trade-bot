"""
Backtest: regime_pl vs forward volatility on MES 1-min bars from TopstepX.

Fetches all available 1-min history (pages backward), excludes the CME
settlement gap (16:00-17:00 CT = 21:00-22:00 UTC), then runs the same
regime_pl / trailing_vol analysis as the equity backtest.
"""

import math
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from topstep_client import TopstepClient

BAR_MINUTES = 1
WINDOWS = {
    "30min":  30,
    "1hour":  60,
    "2hour": 120,
    "3hour": 180,
}
# CME daily settlement gap: 16:00-17:00 CT = 21:00-22:00 UTC
SETTLEMENT_START_UTC = 21   # hour
SETTLEMENT_END_UTC   = 22


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_all_bars(client: TopstepClient, contract_id: str) -> pd.DataFrame:
    """Page backward through 1-min history and return a clean DataFrame."""
    all_bars = []
    end = datetime.now(timezone.utc)
    limit_date = end - timedelta(days=180)

    while end > limit_date:
        start = end - timedelta(days=60)
        bars = client.get_bars(
            contract_id=contract_id,
            start=start,
            end=end,
            unit=TopstepClient.MINUTE,
            unit_number=BAR_MINUTES,
            limit=20000,
        )
        if not bars:
            break
        oldest_dt = datetime.fromisoformat(bars[-1]["t"])
        all_bars.extend(bars)
        end = oldest_dt - timedelta(minutes=1)
        print(f"  fetched {len(bars):,} bars back to {oldest_dt.date()}", flush=True)
        if len(bars) < 20000:
            break

    df = pd.DataFrame(all_bars)
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume"})
    return df


def filter_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bars during the CME settlement gap (21:00-22:00 UTC)."""
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))]
    df = df.reset_index(drop=True)
    # Mark wherever a gap larger than 1 bar appears (session boundary)
    df["new_session"] = df["ts"].diff() > pd.Timedelta(minutes=BAR_MINUTES * 2)
    return df


# ── Analytics ────────────────────────────────────────────────────────────────

def log_returns(closes: np.ndarray) -> np.ndarray:
    return np.log(closes[1:] / closes[:-1])


def regime_pl(rets: np.ndarray) -> float:
    total_abs = np.sum(np.abs(rets))
    if total_abs == 0:
        return 0.0
    return abs(float(np.sum(rets))) / float(total_abs)


def realised_vol(rets: np.ndarray) -> float:
    """Annualised vol; CME MES trades ~23h/day, 252 days."""
    if len(rets) < 2:
        return float("nan")
    bars_per_year = 252 * 23 * 60 / BAR_MINUTES
    return float(np.std(rets, ddof=1) * math.sqrt(bars_per_year))


# ── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, window: int) -> pd.DataFrame:
    closes      = df["close"].values
    new_session = df["new_session"].values
    results = []

    for i in range(window, len(df) - window):
        if new_session[i - window + 1 : i + window + 1].any():
            continue

        trailing_rets = log_returns(closes[i - window : i + 1])
        forward_rets  = log_returns(closes[i : i + window + 1])

        results.append({
            "ts":           df["ts"].iloc[i],
            "regime_pl":    regime_pl(trailing_rets),
            "trailing_vol": realised_vol(trailing_rets),
            "forward_vol":  realised_vol(forward_rets),
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


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with TopstepClient() as client:
        contracts = client.search_contracts("MES")
        contract  = contracts[0]
        print(f"Contract: {contract['name']}  id={contract['id']}")

        print("Fetching 1-min bars...")
        df = fetch_all_bars(client, contract["id"])

    df = filter_sessions(df)
    print(f"\n{len(df):,} bars after settlement-gap filter  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    for label, window in WINDOWS.items():
        res = run_backtest(df, window)
        summarise(res, label)
