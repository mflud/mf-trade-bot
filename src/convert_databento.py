"""
Convert Databento GLBX.MDP3 ohlcv-1m DBN file to the project's CSV format.

Builds a continuous front-month series for each instrument by rolling to the
next contract once the current front-month expires.

Expiry conventions:
  - US equity index micro futures (MES, MNQ, MYM, M2K): third Friday of month
  - Nikkei dollar futures (NKD): second Friday of month (CME SQ date)

Output files (same format as TopstepX cache):
  mes_hist_1min.csv    — MES continuous 1-min OHLCV
  mym_hist_1min.csv    — MYM continuous 1-min OHLCV
  m2k_hist_1min.csv    — M2K continuous 1-min OHLCV
  nkd_hist_1min.csv    — NKD continuous 1-min OHLCV

Usage:
  python src/convert_databento.py
  python src/convert_databento.py --dbn path/to/file.dbn.zst
  python src/convert_databento.py --dbn path/to/file.dbn.zst --sym M2K NKD
"""

import argparse
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import databento as db

# ── Config ─────────────────────────────────────────────────────────────────────

DBN_DEFAULT = "glbx-mdp3-20100606-20260312.ohlcv-1m.dbn.zst"

# (output csv, expiry_fn key)
INSTRUMENTS = {
    "MES": ("mes_hist_1min.csv", "third_friday"),
    "MNQ": ("mnq_hist_1min.csv", "third_friday"),
    "MYM": ("mym_hist_1min.csv", "third_friday"),
    "M2K": ("m2k_hist_1min.csv", "third_friday"),
    "NKD": ("nkd_hist_1min.csv", "second_friday"),
}

# Month code → month number
MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


# ── Expiry helpers ─────────────────────────────────────────────────────────────

def third_friday(year: int, month: int) -> date:
    """Third Friday of the month — CME US equity index futures expiry."""
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)   # first Friday
    return d + timedelta(weeks=2)


def second_friday(year: int, month: int) -> date:
    """Second Friday of the month — CME NKD (Nikkei dollar) expiry."""
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)   # first Friday
    return d + timedelta(weeks=1)


EXPIRY_FN = {
    "third_friday":  third_friday,
    "second_friday": second_friday,
}


def parse_expiry(symbol: str, expiry_fn) -> date | None:
    """
    Parse CME futures symbol like MESM9, MESH26, NKDZ25 → expiry date.
    Returns None if the symbol can't be parsed (e.g. spread symbols).

    CME year encoding:
      1-digit  9       → 2019
      1-digit  0–8     → 2020–2028
      2-digit  19–29   → 2019–2029   (Databento uses these for newer contracts)
    """
    m = re.match(r"^([A-Z0-9]+)([FGHJKMNQUVXZ])(\d{1,2})$", symbol)
    if not m:
        return None
    month_code = m.group(2)
    year_raw   = m.group(3)
    year_int   = int(year_raw)
    month      = MONTH_CODE[month_code]

    if len(year_raw) == 1:
        year = 2019 if year_int == 9 else 2020 + year_int
    else:
        year = 2000 + year_int

    return expiry_fn(year, month)


# ── Build continuous series ────────────────────────────────────────────────────

def build_continuous(df: pd.DataFrame, root: str, expiry_fn) -> pd.DataFrame:
    """
    Given a DataFrame of all contracts for one root (e.g. MES), return a
    continuous front-month series.

    At each 1-min bar timestamp the front month is the contract whose expiry
    is the nearest future date.  Once a contract expires we switch to the next.
    """
    mask = df["symbol"].str.match(rf"^{re.escape(root)}[FGHJKMNQUVXZ]\d{{1,2}}$")
    sub  = df[mask].copy()
    if sub.empty:
        raise ValueError(f"No data found for root {root}")

    sub["expiry"] = pd.to_datetime(
        sub["symbol"].map(lambda s: parse_expiry(s, expiry_fn))
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
    parser.add_argument("--sym", nargs="+", choices=sorted(INSTRUMENTS.keys()),
                        help="Instruments to extract (default: all)")
    args = parser.parse_args()

    dbn_path = Path(args.dbn)
    if not dbn_path.exists():
        raise FileNotFoundError(f"DBN file not found: {dbn_path}")

    targets = {k: INSTRUMENTS[k] for k in (args.sym or INSTRUMENTS)}

    print(f"Loading {dbn_path} …")
    store = db.DBNStore.from_file(str(dbn_path))
    df    = store.to_df()
    print(f"  Loaded {len(df):,} rows  columns: {df.columns.tolist()}")

    # Ensure index is UTC datetime
    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "ts_event"

    for root, (out_path, expiry_key) in targets.items():
        print(f"\nBuilding continuous series for {root} …")
        try:
            out = build_continuous(df, root, EXPIRY_FN[expiry_key])
            out.to_csv(out_path)
            print(f"  Saved → {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")
