"""
One-time backfill of 5-second MES bars from the TopstepX API.

Chains paginated calls backward in time until the API returns fewer than
20,000 bars (true beginning of available data), then merges with the
existing mes_hist_5sec.csv, deduplicates, and saves.

Usage:
  python src/backfill_5sec.py
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from topstep_client import TopstepClient

CSV_PATH    = Path("mes_hist_5sec.csv")
LIMIT       = 20_000
PAUSE_SECS  = 0.5     # be polite between pages


def fetch_all_pages(client: TopstepClient, contract_id: str) -> pd.DataFrame:
    end      = datetime.now(timezone.utc)
    anchor   = datetime(2020, 1, 1, tzinfo=timezone.utc)
    all_bars = []
    page     = 1

    while True:
        print(f"  Page {page}: fetching up to {end.strftime('%Y-%m-%d %H:%M')} UTC …",
              end="", flush=True)
        bars = client.get_bars(
            contract_id, anchor, end,
            unit=TopstepClient.SECOND, unit_number=5,
            limit=LIMIT,
        )
        if not bars:
            print(" empty — done")
            break

        print(f" {len(bars):,} bars  "
              f"({bars[-1]['t'][:16]} → {bars[0]['t'][:16]})")
        all_bars.extend(bars)

        if len(bars) < LIMIT:
            print("  → reached beginning of available data")
            break

        # Move end back to just before the oldest bar
        oldest_ts = datetime.fromisoformat(bars[-1]["t"])
        end = oldest_ts - timedelta(seconds=1)
        page += 1
        time.sleep(PAUSE_SECS)

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df = df.rename(columns={"t": "ts", "o": "open", "h": "high",
                             "l": "low",  "c": "close", "v": "volume"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    if not CSV_PATH.exists():
        print(f"  No existing {CSV_PATH} — using fetched data only")
        return new_df

    existing = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    existing.index = pd.to_datetime(existing.index, utc=True)
    print(f"  Existing: {len(existing):,} bars  "
          f"({existing.index[0].date()} → {existing.index[-1].date()})")

    combined = pd.concat([existing, new_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined


if __name__ == "__main__":
    print("Connecting …")
    with TopstepClient() as client:
        contracts   = client.search_contracts("MES")
        contract_id = contracts[0]["id"]
        print(f"Contract: {contracts[0]['name']}  id={contract_id}\n")

        print("Fetching 5-sec bars (paging backward) …")
        new_df = fetch_all_pages(client, contract_id)

    if new_df.empty:
        print("No data fetched.")
        sys.exit(1)

    print(f"\nFetched: {len(new_df):,} bars  "
          f"({new_df.index[0].date()} → {new_df.index[-1].date()})")

    print("\nMerging with existing CSV …")
    merged = merge_with_existing(new_df)

    merged.to_csv(CSV_PATH)
    print(f"\nSaved {len(merged):,} bars → {CSV_PATH}")
    print(f"  Range: {merged.index[0].date()} → {merged.index[-1].date()}")
