"""
5-Second Bar Recorder for MES and MYM.

Polls the TopstepX API every 60 seconds and appends new 5-second OHLCV bars
to CSV files.  On first run it backfills all available history (~28 hours).
On subsequent runs it resumes from the last saved timestamp.

Output files (in project root):
  mes_hist_5sec.csv
  mym_hist_5sec.csv

CSV columns:  ts, open, high, low, close, volume
  ts is UTC ISO-8601, e.g. 2026-03-14T14:30:05+00:00

Usage:
  python src/bar_recorder.py          # record MES + MYM
  python src/bar_recorder.py --sym MES  # single instrument
"""

import argparse
import csv
import logging
import time
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from topstep_client import TopstepClient

# ── Config ───────────────────────────────────────────────────────────────────

POLL_SECONDS  = 60      # how often to fetch new bars
FETCH_OVERLAP = 300     # seconds of overlap on each fetch (avoid missing bars)
BAR_SECONDS   = 5       # bar size

INSTRUMENTS = {
    "MES": {"search": "MES", "file": "mes_hist_5sec.csv"},
    "MYM": {"search": "MYM", "file": "mym_hist_5sec.csv"},
}

CSV_FIELDS = ["ts", "open", "high", "low", "close", "volume"]

log = logging.getLogger("recorder")

_CT = ZoneInfo("America/Chicago")

def is_cme_weekend_closure() -> bool:
    """Return True during CME Globex weekend closure: Fri 16:00 CT – Sun 17:00 CT."""
    now_ct = datetime.now(_CT)
    wd = now_ct.weekday()   # 0=Mon … 4=Fri, 5=Sat, 6=Sun
    t  = now_ct.time()
    if wd == 4 and t >= dtime(16, 0):   # Friday after 4 PM CT
        return True
    if wd == 5:                          # All of Saturday
        return True
    if wd == 6 and t < dtime(17, 0):    # Sunday before 5 PM CT
        return True
    return False

def wait_for_market_open():
    """Block (sleeping 5 min at a time) until CME weekend closure is over."""
    while is_cme_weekend_closure():
        now_ct = datetime.now(_CT)
        log.info(f"CME weekend closure — sleeping until Sunday 17:00 CT "
                 f"(now {now_ct.strftime('%a %H:%M %Z')})")
        time.sleep(300)


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _ensure_csv(path: Path) -> datetime | None:
    """Create CSV with header if missing. Return last saved timestamp or None."""
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        return None

    # Find the last timestamp in the file efficiently
    last_ts = None
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_ts = row["ts"]
    if last_ts:
        return datetime.fromisoformat(last_ts)
    return None


def _append_bars(path: Path, bars: list[dict], after: datetime | None) -> int:
    """Append bars with ts > after. Returns number of rows written."""
    new_bars = []
    for b in bars:
        ts = datetime.fromisoformat(b["t"])
        if after is None or ts > after:
            new_bars.append({
                "ts":     b["t"],
                "open":   b["o"],
                "high":   b["h"],
                "low":    b["l"],
                "close":  b["c"],
                "volume": b["v"],
            })

    if not new_bars:
        return 0

    # Sort ascending before appending
    new_bars.sort(key=lambda r: r["ts"])

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerows(new_bars)

    return len(new_bars)


# ── Recorder ─────────────────────────────────────────────────────────────────

def record(symbols: list[str]):
    wait_for_market_open()

    client = TopstepClient()
    client.login()

    # Resolve contract IDs and initialise CSV files
    states = {}
    for sym in symbols:
        cfg       = INSTRUMENTS[sym]
        contracts = client.search_contracts(cfg["search"])
        if not contracts:
            log.error(f"No contract found for {sym}")
            continue
        cid  = contracts[0]["id"]
        path = Path(cfg["file"])
        last = _ensure_csv(path)
        states[sym] = {"cid": cid, "path": path, "last_ts": last}

        if last:
            log.info(f"{sym}: resuming from {last}  file={path}")
        else:
            log.info(f"{sym}: new file, will backfill all available history  file={path}")

    if not states:
        raise RuntimeError("No instruments initialised.")

    # Initial backfill: fetch as much history as the API holds (~28 hrs)
    log.info("Backfilling available history …")
    for sym, st in states.items():
        try:
            bars = client.get_bars(
                contract_id=st["cid"],
                start=datetime(2020, 1, 1, tzinfo=timezone.utc),
                end=datetime.now(timezone.utc),
                unit=TopstepClient.SECOND,
                unit_number=BAR_SECONDS,
                limit=20000,
                include_partial=False,
            )
            bars = list(reversed(bars))   # ascending
            n = _append_bars(st["path"], bars, st["last_ts"])
            if n:
                st["last_ts"] = datetime.fromisoformat(bars[-1]["t"])
            log.info(f"  {sym}: backfilled {n} bars  last={st['last_ts']}")
        except Exception as e:
            log.error(f"  {sym}: backfill failed — {e}")

    # Continuous polling loop
    log.info(f"Polling every {POLL_SECONDS}s …  Ctrl-C to stop")
    while True:
        if is_cme_weekend_closure():
            wait_for_market_open()
            # Re-login after weekend so the token is fresh
            log.info("Market reopened — re-logging in …")
            client.login()

        time.sleep(POLL_SECONDS)
        now = datetime.now(timezone.utc)

        for sym, st in states.items():
            try:
                # Fetch from (last_ts - overlap) to now
                start = (st["last_ts"] - timedelta(seconds=FETCH_OVERLAP)) \
                        if st["last_ts"] else datetime(2020, 1, 1, tzinfo=timezone.utc)
                bars = client.get_bars(
                    contract_id=st["cid"],
                    start=start,
                    end=now,
                    unit=TopstepClient.SECOND,
                    unit_number=BAR_SECONDS,
                    limit=500,
                    include_partial=False,
                )
                bars = list(reversed(bars))   # ascending
                n = _append_bars(st["path"], bars, st["last_ts"])
                if n:
                    st["last_ts"] = datetime.fromisoformat(bars[-1]["t"])
                    log.info(f"{sym}: +{n} bars  last={st['last_ts']}")
                else:
                    log.debug(f"{sym}: no new bars")
            except Exception as e:
                log.error(f"{sym}: fetch failed — {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/recorder.log"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="5-second bar recorder for MES and MYM")
    parser.add_argument("--sym", nargs="+", default=list(INSTRUMENTS.keys()),
                        choices=list(INSTRUMENTS.keys()),
                        help="Instrument(s) to record (default: all)")
    args = parser.parse_args()

    record([s.upper() for s in args.sym])
