"""
Convert mizu-trading 1s ndjson OHLCV files (ES, NQ) to 5s CSV and merge with
any existing bar_recorder 5s data.

Reads all part files from data/mizu/es/ and data/mizu/nq/, builds a
continuous front-month series (same expiry logic as convert_databento.py),
resamples to 5s, then prepends the result to any existing *_hist_5sec.csv so
the final file covers the full available history.

Output files (project root):
  es_hist_5sec.csv   — ES continuous 5s OHLCV
  nq_hist_5sec.csv   — NQ continuous 5s OHLCV

CSV columns: ts, open, high, low, close, volume
  ts is UTC ISO-8601 (same format as bar_recorder output)

Usage:
  python src/convert_mizu_ndjson.py
  python src/convert_mizu_ndjson.py --sym ES
  python src/convert_mizu_ndjson.py --sym NQ
  python src/convert_mizu_ndjson.py --bar 1   # keep 1s resolution (no resample)
"""

import argparse
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────

INSTRUMENTS = {
    "ES": {
        "ndjson_dir": Path("data/mizu/es"),
        "out_csv":    Path("es_hist_5sec.csv"),
        "expiry":     "third_friday",
    },
    "NQ": {
        "ndjson_dir": Path("data/mizu/nq"),
        "out_csv":    Path("nq_hist_5sec.csv"),
        "expiry":     "third_friday",
    },
}

MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

BAR_SECONDS = 5    # resample target


# ── Expiry helpers (same as convert_databento.py) ─────────────────────────────

def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(4 - d.weekday()) % 7)
    return d + timedelta(weeks=2)


EXPIRY_FN = {"third_friday": third_friday}


def parse_expiry(symbol: str, expiry_fn) -> date | None:
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


# ── ndjson reader ─────────────────────────────────────────────────────────────

def read_ndjson_dir(ndjson_dir: Path) -> pd.DataFrame:
    """Stream all part files in sorted order; return raw 1s DataFrame."""
    part_files = sorted(ndjson_dir.glob("*.ndjson"))
    if not part_files:
        raise FileNotFoundError(f"No .ndjson files in {ndjson_dir}")

    print(f"  Reading {len(part_files)} part file(s) from {ndjson_dir} …")
    rows = []
    for pf in part_files:
        print(f"    {pf.name}", end="\r", flush=True)
        bad = 0
        with open(pf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                rows.append({
                    "ts":     rec["hd"]["ts_event"],
                    "open":   float(rec["open"]),
                    "high":   float(rec["high"]),
                    "low":    float(rec["low"]),
                    "close":  float(rec["close"]),
                    "volume": int(rec["volume"]),
                    "symbol": rec["symbol"],
                })
    print()   # clear \r line
    if bad:
        print(f"  WARNING: {bad} unparseable lines skipped across all parts")
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"  {len(df):,} raw 1s bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")
    return df


# ── Continuous front-month ────────────────────────────────────────────────────

def build_continuous(df: pd.DataFrame, root: str, expiry_fn) -> pd.DataFrame:
    """
    Keep only front-month bars at each timestamp.
    Same approach as convert_databento.py.
    """
    mask = df["symbol"].str.match(rf"^{re.escape(root)}[FGHJKMNQUVXZ]\d{{1,2}}$")
    sub  = df[mask].copy()

    sub["expiry"] = pd.to_datetime(
        sub["symbol"].map(lambda s: parse_expiry(s, expiry_fn))
    ).dt.tz_localize("UTC")
    sub = sub.dropna(subset=["expiry"])

    # Drop bars where contract has already expired
    sub = sub[sub["expiry"] >= sub["ts"]]

    # At each timestamp keep front month (nearest expiry)
    sub = sub.sort_values("expiry")
    sub = sub.groupby("ts")[["open", "high", "low", "close", "volume"]].first()
    sub = sub.sort_index()

    print(f"  {root}: {len(sub):,} front-month 1s bars")
    return sub


# ── Resample ──────────────────────────────────────────────────────────────────

def resample_to_5s(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1s bars to 5s. Drops settlement gap. Returns indexed by ts."""
    # Filter CME settlement gap (21:00–22:00 UTC)
    h   = df.index.hour
    df  = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()

    rs = df.resample(f"{BAR_SECONDS}s").agg(
        open=("open",   "first"),
        high=("high",   "max"),
        low=("low",    "min"),
        close=("close",  "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"])

    print(f"  → {len(rs):,} 5s bars after resampling + gap filter")
    return rs


# ── Merge with existing CSV ───────────────────────────────────────────────────

def merge_and_save(new_df: pd.DataFrame, out_csv: Path):
    """
    Prepend new_df (mizu history) to any existing out_csv (bar_recorder data).
    Deduplicates on ts, keeps existing data for any overlapping timestamps.
    """
    if out_csv.exists():
        print(f"  Merging with existing {out_csv} …")
        existing = pd.read_csv(out_csv)
        existing["ts"] = pd.to_datetime(existing["ts"], format="ISO8601", utc=True)
        existing = existing.set_index("ts")

        # Combine: existing takes priority for overlapping rows
        combined = pd.concat([new_df, existing])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = new_df.sort_index()

    combined.index.name = "ts"
    combined.to_csv(out_csv)
    print(f"  Saved {len(combined):,} bars → {out_csv}  "
          f"({combined.index.min().date()} → {combined.index.max().date()})")


# ── Main ───────────────────────────────────────────────────────────────────────

def run(symbols: list[str], bar_seconds: int):
    for sym in symbols:
        cfg        = INSTRUMENTS[sym]
        expiry_fn  = EXPIRY_FN[cfg["expiry"]]

        print(f"\n{'═'*70}")
        print(f"  {sym}")
        print(f"{'═'*70}")

        try:
            raw = read_ndjson_dir(cfg["ndjson_dir"])
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        cont = build_continuous(raw, sym, expiry_fn)

        if bar_seconds == 1:
            resampled = cont
            print(f"  Keeping 1s resolution (no resample)")
        else:
            resampled = resample_to_5s(cont)

        merge_and_save(resampled, cfg["out_csv"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", nargs="+", choices=list(INSTRUMENTS.keys()),
                        default=list(INSTRUMENTS.keys()),
                        help="Instrument(s) to convert (default: all)")
    parser.add_argument("--bar", type=int, default=BAR_SECONDS,
                        help=f"Output bar size in seconds (default: {BAR_SECONDS})")
    args = parser.parse_args()
    run([s.upper() for s in args.sym], args.bar)
