"""
Compare trailing 10-day realised volatility for MES 1-min bars under two session definitions:
  a) NYSE session only: 09:30-16:00 ET
  b) Full 24h session (excluding CME settlement gap 21:00-22:00 UTC)
"""

import math
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from topstep_client import TopstepClient

TRADING_DAYS = 10
BAR_MINUTES = 1

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC = 22

NYSE_OPEN_ET  = "09:30"
NYSE_CLOSE_ET = "16:00"

# Annualisation denominators
BARS_PER_YEAR_NYSE = 252 * 390       # 6.5h * 60 min
BARS_PER_YEAR_FULL = 252 * 23 * 60  # 23h * 60 min (settlement gap excluded)


def fetch_bars(client: TopstepClient, contract_id: str, days_back: int = 16) -> pd.DataFrame:
    """Fetch ~days_back calendar days of 1-min bars (newest-first from API → sorted asc)."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    bars = client.get_bars(
        contract_id=contract_id,
        start=start,
        end=end,
        unit=TopstepClient.MINUTE,
        unit_number=BAR_MINUTES,
        limit=20000,
    )
    df = pd.DataFrame(bars)
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = (df.drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True)
            .rename(columns={"o": "open", "h": "high", "l": "low",
                             "c": "close", "v": "volume"}))
    return df


def realised_vol(closes: np.ndarray, bars_per_year: int) -> float:
    rets = np.log(closes[1:] / closes[:-1])
    if len(rets) < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * math.sqrt(bars_per_year))


def nyse_session_vol(df: pd.DataFrame) -> tuple[float, int, int]:
    """
    Trailing TRADING_DAYS NYSE sessions, 09:30-16:00 ET.
    Returns (annualised_vol, n_bars, n_days).
    """
    df2 = df.copy()
    df2["ts_et"] = df2["ts"].dt.tz_convert("America/New_York")
    time_str = df2["ts_et"].dt.strftime("%H:%M")
    df2 = df2[(time_str >= NYSE_OPEN_ET) & (time_str < NYSE_CLOSE_ET)].copy()
    df2["date"] = df2["ts_et"].dt.date

    dates = sorted(df2["date"].unique())
    if len(dates) < TRADING_DAYS:
        raise ValueError(f"Only {len(dates)} NYSE trading days in fetch window; need {TRADING_DAYS}.")
    window = df2[df2["date"].isin(set(dates[-TRADING_DAYS:]))]

    closes = window["close"].values
    return realised_vol(closes, BARS_PER_YEAR_NYSE), len(closes), TRADING_DAYS


def full_session_vol(df: pd.DataFrame) -> tuple[float, int, int]:
    """
    Trailing TRADING_DAYS CME sessions, full 24h minus settlement gap.

    CME session convention: 17:00 CT (prev day) → 16:00 CT (current day).
    Bars from 17:00-23:59 CT are attributed to the *next* calendar date's session.
    Returns (annualised_vol, n_bars, n_sessions).
    """
    hour_utc = df["ts"].dt.hour
    df2 = df[~((hour_utc >= SETTLEMENT_START_UTC) & (hour_utc < SETTLEMENT_END_UTC))].copy()

    df2["ts_ct"] = df2["ts"].dt.tz_convert("America/Chicago")
    df2["session_date"] = df2["ts_ct"].dt.date
    after_close = df2["ts_ct"].dt.hour >= 17
    df2.loc[after_close, "session_date"] = (
        df2.loc[after_close, "ts_ct"] + pd.Timedelta(days=1)
    ).dt.date

    dates = sorted(df2["session_date"].unique())
    if len(dates) < TRADING_DAYS:
        raise ValueError(f"Only {len(dates)} CME sessions in fetch window; need {TRADING_DAYS}.")
    window = df2[df2["session_date"].isin(set(dates[-TRADING_DAYS:]))]

    closes = window["close"].values
    return realised_vol(closes, BARS_PER_YEAR_FULL), len(closes), TRADING_DAYS


if __name__ == "__main__":
    with TopstepClient() as client:
        contracts = client.search_contracts("MES")
        contract = contracts[0]
        print(f"Contract: {contract['name']}  id={contract['id']}")
        print(f"Fetching 1-min bars (last ~16 calendar days)...")
        df = fetch_bars(client, contract["id"])
        print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    print(f"\nTrailing {TRADING_DAYS}-day realised volatility  (1-min log returns, annualised)")
    print("-" * 52)

    vol_nyse, n_bars_nyse, _ = nyse_session_vol(df)
    print(f"a) NYSE session only  (09:30-16:00 ET)")
    print(f"   {n_bars_nyse:,} bars  |  ann. vol = {vol_nyse * 100:.2f}%")

    vol_full, n_bars_full, _ = full_session_vol(df)
    print(f"\nb) Full 24h session  (excl. 21:00-22:00 UTC settlement gap)")
    print(f"   {n_bars_full:,} bars  |  ann. vol = {vol_full * 100:.2f}%")

    diff = vol_full - vol_nyse
    print(f"\nDifference (full - NYSE): {diff * 100:+.2f}pp")
