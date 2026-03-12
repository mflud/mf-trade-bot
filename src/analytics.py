"""
Bar-based analytics: trailing high/low range and volatility.
"""

import math
from datetime import datetime, timedelta, timezone
from topstep_client import TopstepClient


def trailing_range_and_volatility(
    client: TopstepClient,
    contract_id: str,
    n_minutes: int,
    bar_size: int = 1,
) -> dict:
    """
    Compute the trailing n-minute high/low range and volatility for a contract.

    Parameters
    ----------
    client      : authenticated TopstepClient
    contract_id : contract id string (e.g. 'CON.F.US.MES.H26')
    n_minutes   : lookback window in minutes
    bar_size    : bar width in minutes (default 1)

    Returns
    -------
    dict with keys:
      high          - highest high over the window
      low           - lowest low over the window
      hl_range      - high - low (points)
      hl_range_pct  - hl_range / midpoint * 100
      returns_std   - std dev of bar close-to-close returns
      volatility    - annualised volatility (returns_std * sqrt(bars_per_year))
      n_bars        - number of bars used
    """
    # Fetch a wider window to handle session gaps (e.g. CME 4-5pm ET settlement).
    # We request enough bars to cover n_minutes of actual trading time and take
    # the most recent n_bars from whatever the API returns.
    n_bars_needed = n_minutes // bar_size
    end = datetime.now(timezone.utc)
    # Look back 3x to absorb gaps; cap fetch at 3000 bars
    fetch_limit = min(n_bars_needed * 3, 3000)
    start = end - timedelta(minutes=n_minutes * 3)

    all_bars = client.get_bars(
        contract_id=contract_id,
        start=start,
        end=end,
        unit=TopstepClient.MINUTE,
        unit_number=bar_size,
        limit=fetch_limit,
    )

    # API returns newest-first; take the n most recent bars
    bars = all_bars[:n_bars_needed] if len(all_bars) >= n_bars_needed else all_bars
    # Reverse so bars are oldest-first for return calculations
    bars = list(reversed(bars))

    if len(bars) < 2:
        raise ValueError(f"Not enough bars returned ({len(bars)}); need at least 2.")

    highs  = [b["h"] for b in bars]
    lows   = [b["l"] for b in bars]
    closes = [b["c"] for b in bars]

    high = max(highs)
    low  = min(lows)
    hl_range = high - low
    midpoint = (high + low) / 2
    hl_range_pct = (hl_range / midpoint) * 100

    # Close-to-close log returns
    returns = [
        math.log(closes[i] / closes[i - 1])
        for i in range(1, len(closes))
    ]
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
    returns_std = math.sqrt(variance)

    # Annualise: trading minutes in a year for CME futures (~23h/day, 252 days)
    bars_per_year = int(252 * 23 * 60 / bar_size)
    volatility = returns_std * math.sqrt(bars_per_year)

    return {
        "high": high,
        "low": low,
        "hl_range": hl_range,
        "hl_range_pct": round(hl_range_pct, 4),
        "returns_std": round(returns_std, 6),
        "volatility": round(volatility, 4),
        "n_bars": len(bars),
    }


def regime_pl(
    client: TopstepClient,
    contract_id: str,
    n_minutes: int,
    bar_size: int = 1,
) -> float:
    """
    Price Linearity ratio over a trailing n-minute window.

    regime_pl = |sum(log_returns)| / sum(|log_returns|)

    Range: [0, 1]
      ~1  => strongly trending (returns stack in one direction)
      ~0  => choppy/mean-reverting (returns cancel each other out)
    """
    n_bars_needed = n_minutes // bar_size
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=n_minutes * 3)

    all_bars = client.get_bars(
        contract_id=contract_id,
        start=start,
        end=end,
        unit=TopstepClient.MINUTE,
        unit_number=bar_size,
        limit=min(n_bars_needed * 3, 3000),
    )

    bars = list(reversed(all_bars[:n_bars_needed] if len(all_bars) >= n_bars_needed else all_bars))

    if len(bars) < 2:
        raise ValueError(f"Not enough bars returned ({len(bars)}); need at least 2.")

    closes = [b["c"] for b in bars]
    returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

    sum_returns = sum(returns)
    sum_abs_returns = sum(abs(r) for r in returns)

    if sum_abs_returns == 0:
        return 0.0

    return abs(sum_returns) / sum_abs_returns


if __name__ == "__main__":
    with TopstepClient() as client:
        contracts = client.search_contracts("MES")
        contract = contracts[0]
        print(f"Contract: {contract['name']}  id={contract['id']}\n")

        print("--- regime_pl ---")
        for window in [30, 60, 240]:
            pl = regime_pl(client, contract_id=contract["id"], n_minutes=window)
            print(f"  {window:>4}min  regime_pl={pl:.4f}")
        print()

        print("--- range & volatility ---")
        for window in [30, 60, 240]:
            result = trailing_range_and_volatility(
                client,
                contract_id=contract["id"],
                n_minutes=window,
                bar_size=1,
            )
            print(f"Trailing {window}min ({result['n_bars']} bars):")
            print(f"  High:         {result['high']}")
            print(f"  Low:          {result['low']}")
            print(f"  H/L Range:    {result['hl_range']} pts  ({result['hl_range_pct']}%)")
            print(f"  Returns StdDev: {result['returns_std']}")
            print(f"  Ann. Vol:     {result['volatility'] * 100:.2f}%")
            print()
