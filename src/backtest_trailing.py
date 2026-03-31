"""
Trailing-stop backtest on all logged trades (VWASLR, ORB, CSR).

For each trade, fetches 1-min bars and simulates a trailing stop with various
trail distances (in σ multiples), then compares aggregate P&L to the fixed-stop
baseline.

Trailing stop mechanics (conservative OHLC simulation):
  LONG  – check if bar.low  <= trail_stop first; if not, update peak with bar.high
  SHORT – check if bar.high >= trail_stop first; if not, update peak with bar.low
Using conservative ordering (adverse side tested before peak update within each bar)
to avoid overstating trailing-stop performance.

Usage:
  python src/backtest_trailing.py           # all strategies
  python src/backtest_trailing.py --vwas    # VWASLR only
  python src/backtest_trailing.py --orb     # ORB only
  python src/backtest_trailing.py --csr     # CSR only
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "src")
from topstep_client import TopstepClient

# ── Config ─────────────────────────────────────────────────────────────────────

VWAS_PATH    = Path("logs/vwaslr_trades.csv")
ORB_PATH     = Path("logs/orb_signals.csv")
CSR_PATH     = Path("logs/signals.csv")
BAR_CACHE    = Path("logs/trailing_bar_cache.json")   # avoids re-fetching on reruns

TRAIL_SIGMAS = [0.5, 1.0, 1.5, 2.0]   # trail distances to sweep (in σ units)
MAX_HOLD_MIN = 25                       # same as live bot
POINT_VALUE  = {"MES": 5.0, "MYM": 0.5, "M2K": 5.0, "MNQ": 2.0}
PAUSE_SECS   = 1.0                      # between API calls to avoid 429
MAX_RETRIES  = 3                        # retries on 429 with backoff


# ── Loading ────────────────────────────────────────────────────────────────────

def _ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_trades(which: set[str]) -> list[dict]:
    trades = []

    if "VWASLR" in which and VWAS_PATH.exists():
        with open(VWAS_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    trades.append({
                        "kind":        "VWASLR",
                        "symbol":      row["symbol"],
                        "direction":   1 if row["direction"] == "LONG" else -1,
                        "entry":       float(row.get("fill_price") or row["est_entry"]),
                        "sigma_pts":   float(row["sigma_pts"]),
                        "fixed_stop":  float(row["stop"]),
                        "fired_at":    _ts(row["fired_at"]),
                    })
                except Exception:
                    continue

    if "ORB" in which and ORB_PATH.exists():
        with open(ORB_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    trades.append({
                        "kind":        "ORB",
                        "symbol":      row["symbol"],
                        "direction":   1 if row["direction"] == "LONG" else -1,
                        "entry":       float(row["entry"]),
                        "sigma_pts":   float(row["sigma_pts"]),
                        "fixed_stop":  float(row["stop"]),
                        "fired_at":    _ts(row["fired_at"]),
                    })
                except Exception:
                    continue

    if "CSR" in which and CSR_PATH.exists():
        with open(CSR_PATH, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    trades.append({
                        "kind":        "CSR",
                        "symbol":      row["symbol"],
                        "direction":   1 if row["direction"] == "LONG" else -1,
                        "entry":       float(row["entry"]),
                        "sigma_pts":   float(row["sigma_pts"]),
                        "fixed_stop":  float(row["stop"]),
                        "fired_at":    _ts(row["fired_at"]),
                    })
                except Exception:
                    continue

    trades.sort(key=lambda t: t["fired_at"])
    return trades


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_fixed(bars: list[dict], trade: dict) -> float:
    """Reproduce the original fixed-stop result from 1-min bar data."""
    entry      = trade["entry"]
    direction  = trade["direction"]
    fixed_stop = trade["fixed_stop"]
    # No fixed target in the simulation — just stop or time exit
    # (targets were set at 3σ for CSR/ORB, 3σ for VWASLR; include them)
    sigma_pts  = trade["sigma_pts"]
    # Use the original stop from the log; target at +3σ (standard across all strategies)
    target     = entry + direction * 3.0 * sigma_pts

    for bar in bars:
        if direction == 1:    # LONG
            if bar["l"] <= fixed_stop:
                return fixed_stop - entry
            if bar["h"] >= target:
                return target - entry
        else:                 # SHORT
            if bar["h"] >= fixed_stop:
                return fixed_stop - entry
            if bar["l"] <= target:
                return target - entry

    # Time exit
    return (bars[-1]["c"] - entry) * direction


def simulate_trailing(bars: list[dict], trade: dict,
                      trail_sigma: float) -> float:
    """
    Trailing stop simulation.
      - Initial stop is trail_sigma * sigma_pts away from entry (same as fixed stop
        if trail_sigma equals the original stop multiple).
      - Stop trails the price: after each favourable bar, stop moves up/down to
        (peak - trail_distance) for long, (peak + trail_distance) for short.
      - Conservative OHLC ordering: adverse side checked before peak update.
    """
    entry       = trade["entry"]
    direction   = trade["direction"]
    sigma_pts   = trade["sigma_pts"]
    trail_dist  = trail_sigma * sigma_pts

    peak        = entry
    trail_stop  = entry - direction * trail_dist

    for bar in bars:
        if direction == 1:    # LONG
            # Conservative: check low first, then update peak
            if bar["l"] <= trail_stop:
                return trail_stop - entry
            if bar["h"] > peak:
                peak       = bar["h"]
                trail_stop = peak - trail_dist
        else:                 # SHORT
            if bar["h"] >= trail_stop:
                return trail_stop - entry    # trail_stop < entry → positive pnl
            if bar["l"] < peak:
                peak       = bar["l"]
                trail_stop = peak + trail_dist

    # Time exit at last close
    return (bars[-1]["c"] - entry) * direction


# ── Fetching ───────────────────────────────────────────────────────────────────

def _cache_key(trade: dict) -> str:
    return f"{trade['symbol']}_{trade['fired_at'].isoformat()}"


def load_cache() -> dict:
    if BAR_CACHE.exists():
        with open(BAR_CACHE) as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    BAR_CACHE.parent.mkdir(exist_ok=True)
    with open(BAR_CACHE, "w") as f:
        json.dump(cache, f)


def fetch_bars(client: TopstepClient, trade: dict,
               cid: dict[str, str], cache: dict) -> list[dict]:
    key = _cache_key(trade)
    if key in cache:
        return cache[key]

    sym = trade["symbol"]
    if sym not in cid:
        return []
    start = trade["fired_at"]
    end   = start + timedelta(minutes=MAX_HOLD_MIN + 2)

    for attempt in range(MAX_RETRIES):
        try:
            raw = client.get_bars(
                contract_id=cid[sym],
                start=start, end=end,
                unit=TopstepClient.MINUTE, unit_number=1,
                limit=MAX_HOLD_MIN + 5,
            )
            bars = list(reversed(raw))   # chronological
            cache[key] = bars
            return bars
        except Exception as e:
            msg = str(e)
            if "429" in msg and attempt < MAX_RETRIES - 1:
                wait = 10 * (attempt + 1)
                print(f"\n    429 rate-limit — waiting {wait}s …", end="", flush=True)
                time.sleep(wait)
            else:
                print(f"\n    fetch error for {sym}: {e}")
                return []
    return []


# ── Reporting ──────────────────────────────────────────────────────────────────

def _dollars(pts: float, sym: str) -> float:
    return pts * POINT_VALUE.get(sym, 1.0)


def print_per_trade(trades: list[dict],
                    all_sims: list[dict],
                    kind_filter: str | None = None):
    filtered = [(t, s) for t, s in zip(trades, all_sims)
                if kind_filter is None or t["kind"] == kind_filter]
    if not filtered:
        return

    kind = kind_filter or "ALL"
    print(f"\n{'='*90}")
    print(f"  PER-TRADE  [{kind}]")
    print(f"{'='*90}")
    trail_hdrs = "  ".join(f"{ts:.1f}σ trail" for ts in TRAIL_SIGMAS)
    print(f"  {'Date':>10}  {'Sym':>4}  {'Dir':>5}  {'Fixed':>8}  {trail_hdrs}")
    print(f"  {'─'*85}")

    for trade, sims in filtered:
        date_str = trade["fired_at"].astimezone().strftime("%m-%d %H:%M")
        dir_str  = "LONG " if trade["direction"] == 1 else "SHORT"
        fixed    = sims["fixed"]
        trail_vals = "  ".join(
            f"{sims[ts]:>+10.1f}" for ts in TRAIL_SIGMAS
        )
        fixed_str = f"{fixed:>+8.1f}"
        print(f"  {date_str}  {trade['symbol']:>4}  {dir_str}  "
              f"{fixed_str}  {trail_vals}")


def print_summary(trades: list[dict], all_sims: list[dict]):
    from collections import defaultdict

    print(f"\n{'='*75}")
    print(f"  SUMMARY  (pts per trade, total pts, total $)")
    print(f"{'='*75}")

    by_kind: dict[str, list] = defaultdict(list)
    for t, s in zip(trades, all_sims):
        if s["fixed"] != s["fixed"]:   # nan = no bar data
            continue
        by_kind[t["kind"]].append((t, s))

    for kind in ("VWASLR", "ORB", "CSR", "ALL"):
        if kind == "ALL":
            rows = [(t, s) for t, s in zip(trades, all_sims)
                    if s["fixed"] == s["fixed"]]
            if len(rows) <= len(by_kind.get("VWASLR", [])):
                continue   # skip ALL if same as one group
        else:
            rows = by_kind.get(kind, [])
        if not rows:
            continue

        print(f"\n  ── {kind} ({len(rows)} trades) ──")

        # Header
        col_w = 14
        print(f"  {'':18}", end="")
        for ts in TRAIL_SIGMAS:
            label = f"{ts:.1f}σ trail"
            print(f"  {label:>{col_w}}", end="")
        print(f"  {'Fixed stop':>{col_w}}")

        # Avg pts per trade
        print(f"  {'Avg pts/trade':18}", end="")
        for ts in TRAIL_SIGMAS:
            vals = [s[ts] for t, s in rows]
            print(f"  {sum(vals)/len(vals):>+{col_w}.2f}", end="")
        vals = [s["fixed"] for t, s in rows]
        print(f"  {sum(vals)/len(vals):>+{col_w}.2f}")

        # Total pts
        print(f"  {'Total pts':18}", end="")
        for ts in TRAIL_SIGMAS:
            total = sum(s[ts] for t, s in rows)
            print(f"  {total:>+{col_w}.1f}", end="")
        total = sum(s["fixed"] for t, s in rows)
        print(f"  {total:>+{col_w}.1f}")

        # Total $ (1 contract each, correct point values)
        print(f"  {'Total $':18}", end="")
        for ts in TRAIL_SIGMAS:
            dol = sum(_dollars(s[ts], t["symbol"]) for t, s in rows)
            print(f"  {dol:>+{col_w}.0f}", end="")
        dol = sum(_dollars(s["fixed"], t["symbol"]) for t, s in rows)
        print(f"  {dol:>+{col_w}.0f}")

        # Win rate
        print(f"  {'Win rate':18}", end="")
        for ts in TRAIL_SIGMAS:
            wins = sum(1 for t, s in rows if s[ts] > 0)
            print(f"  {wins}/{len(rows)} ({wins/len(rows)*100:.0f}%){'':{col_w-9}}", end="")
        wins = sum(1 for t, s in rows if s["fixed"] > 0)
        print(f"  {wins}/{len(rows)} ({wins/len(rows)*100:.0f}%)")

        # Best trailing sigma for this group (by total pts)
        best_ts  = max(TRAIL_SIGMAS, key=lambda ts: sum(s[ts] for t, s in rows))
        best_tot = sum(s[best_ts] for t, s in rows)
        fixed_tot = sum(s["fixed"] for t, s in rows)
        improvement = best_tot - fixed_tot
        print(f"  → Best trail: {best_ts:.1f}σ  "
              f"({best_tot:+.1f} pts vs fixed {fixed_tot:+.1f}  "
              f"Δ = {improvement:+.1f} pts)")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vwas", action="store_true")
    parser.add_argument("--orb",  action="store_true")
    parser.add_argument("--csr",  action="store_true")
    args = parser.parse_args()

    which: set[str]
    if any([args.vwas, args.orb, args.csr]):
        which = set()
        if args.vwas: which.add("VWASLR")
        if args.orb:  which.add("ORB")
        if args.csr:  which.add("CSR")
    else:
        which = {"VWASLR", "ORB", "CSR"}

    trades = load_trades(which)
    print(f"Loaded {len(trades)} trades")

    print("Connecting …")
    with TopstepClient() as client:
        cid: dict[str, str] = {}
        for sym in sorted({t["symbol"] for t in trades}):
            contracts = client.search_contracts(sym)
            if contracts:
                cid[sym] = contracts[0]["id"]
                print(f"  {sym}: {contracts[0]['name']}")
            else:
                print(f"  {sym}: NOT FOUND")

        print(f"\nFetching 1-min bars and simulating "
              f"{TRAIL_SIGMAS} σ trail distances …\n")

        cache = load_cache()
        cached_count = sum(1 for t in trades if _cache_key(t) in cache)
        if cached_count:
            print(f"  ({cached_count}/{len(trades)} trades loaded from cache)\n")

        all_sims: list[dict] = []
        for i, trade in enumerate(trades, 1):
            sym = trade["symbol"]
            from_cache = _cache_key(trade) in cache
            print(f"  [{i:>2}/{len(trades)}] {trade['kind']:7} {sym} "
                  f"{trade['fired_at'].strftime('%m-%d %H:%M')} "
                  f"{'(cached)' if from_cache else '…      '}",
                  end="", flush=True)

            bars = fetch_bars(client, trade, cid, cache)
            if not from_cache:
                save_cache(cache)   # persist after each new fetch
                time.sleep(PAUSE_SECS)
            if not bars:
                all_sims.append({ts: float("nan") for ts in TRAIL_SIGMAS}
                                | {"fixed": float("nan")})
                print("  (no data)")
                continue

            sims: dict = {}
            sims["fixed"] = simulate_fixed(bars, trade)
            for ts in TRAIL_SIGMAS:
                sims[ts] = simulate_trailing(bars, trade, ts)
            all_sims.append(sims)

            parts = [f"fixed={sims['fixed']:+.1f}"] + \
                    [f"{ts:.1f}σ={sims[ts]:+.1f}" for ts in TRAIL_SIGMAS]
            print("  " + "  ".join(parts))
            time.sleep(PAUSE_SECS)

    # ── Results ────────────────────────────────────────────────────────────────
    for kind in (list(which) if len(which) > 1 else list(which)):
        print_per_trade(trades, all_sims, kind_filter=kind)

    print_summary(trades, all_sims)
