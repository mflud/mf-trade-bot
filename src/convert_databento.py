"""
Convert Databento GLBX.MDP3 ohlcv-1m DBN file to the project's CSV format.

Builds a continuous front-month series for each instrument by rolling to the
next contract once the current front-month expires (third Friday of expiry month).

Output files (same format as TopstepX cache):
  mes_hist_1min.csv    — MES continuous 1-min OHLCV
  mym_hist_1min.csv    — MYM continuous 1-min OHLCV

Usage:
  python src/convert_databento.py
  python src/convert_databento.py --dbn path/to/file.dbn.zst
"""

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import databento as db

# ── Config ─────────────────────────────────────────────────────────────────────

DBN_DEFAULT   = "glbx-mdp3-20100606-20260312.ohlcv-1m.dbn.zst"
INSTRUMENTS   = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

# Month code → month number
MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


# ── Expiry helpers ─────────────────────────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month (CME equity futures expiry)."""
    d = date(year, month, 1)
    # Advance to first Friday
    d += timedelta(days=(4 - d.weekday()) % 7)
    # Advance two more weeks
    return d + timedelta(weeks=2)


def parse_expiry(symbol: str) -> date | None:
    """
    Parse CME futures symbol like MESM9, MESH26, MYMZ25 → expiry date.
    Returns None if the symbol can't be parsed (e.g. spread symbols like MESM9-MESU9).

    CME year encoding in tickers:
      1-digit  9       → 2019
      1-digit  0–8     → 2020–2028
      2-digit  19–29   → 2019–2029   (Databento uses these for newer contracts)
    """
    m = re.match(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})$", symbol)
    if not m:
        return None
    month_code = m.group(2)
    year_raw   = m.group(3)
    year_int   = int(year_raw)
    month      = MONTH_CODE[month_code]

    if len(year_raw) == 1:
        # Single-digit: 9 → 2019, 0 → 2020, 1 → 2021, …
        year = 2019 if year_int == 9 else 2020 + year_int
    else:
        # Two-digit: 19 → 2019, 25 → 2025, 26 → 2026, …
        year = 2000 + year_int

    return third_friday(year, month)


# ── Build continuous series ────────────────────────────────────────────────────

def build_continuous(df: pd.DataFrame, root: str) -> pd.DataFrame:
    """
    Given a DataFrame of all contracts for one root (e.g. MES), return a
    continuous front-month series.

    At each 1-min bar timestamp, the front month is the contract whose expiry
    is the nearest future date relative to that bar's date.  Once a contract
    expires we switch to the next one.
    """
    mask = df["symbol"].str.startswith(root)
    sub  = df[mask].copy()
    if sub.empty:
        raise ValueError(f"No data found for root {root}")

    sub["expiry"] = pd.to_datetime(
        sub["symbol"].map(parse_expiry)
    ).dt.tz_localize("UTC")

    sub = sub.dropna(subset=["expiry"])

    # Drop bars where the contract has already expired
    sub = sub[sub["expiry"] >= sub.index]

    # At each timestamp keep only the front month (nearest expiry)
    sub = sub.sort_values("expiry")
    sub = sub.groupby(sub.index)[["open", "high", "low", "close", "volume", "expiry"]].first()

    out = sub[["open", "high", "low", "close", "volume"]].copy()
    out.index.name = "ts"
    out = out.sort_index()

    print(f"  {root}: {len(out):,} bars  "
          f"{out.index.min().date()} → {out.index.max().date()}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dbn", default=DBN_DEFAULT,
                        help="Path to the .dbn or .dbn.zst file")
    args = parser.parse_args()

    dbn_path = Path(args.dbn)
    if not dbn_path.exists():
        raise FileNotFoundError(f"DBN file not found: {dbn_path}")

    print(f"Loading {dbn_path} …")
    store = db.DBNStore.from_file(str(dbn_path))
    df    = store.to_df()
    print(f"  Loaded {len(df):,} rows  columns: {df.columns.tolist()}")

    # Ensure index is named ts_event and is UTC
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts_event"

    for root, out_path in INSTRUMENTS.items():
        print(f"\nBuilding continuous series for {root} …")
        try:
            out = build_continuous(df, root)
            out.to_csv(out_path)
            print(f"  Saved → {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")
