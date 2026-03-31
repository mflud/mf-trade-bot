"""
MFE / MAE analysis for all logged trades (VWASLR, ORB, CSR momentum).

For each trade, fetches 1-min bars covering the hold period and computes:
  MFE  Max Favourable Excursion — best price achievable from entry
  MAE  Max Adverse  Excursion  — worst drawdown seen from entry
  Capture %  actual_pnl / MFE  (how much of the peak profit was kept)

Usage:
  python src/analyze_mfe.py           # all strategies
  python src/analyze_mfe.py --vwas    # VWASLR only
  python src/analyze_mfe.py --orb     # ORB only
  python src/analyze_mfe.py --csr     # CSR momentum only
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, "src")
from topstep_client import TopstepClient

# ── Paths ──────────────────────────────────────────────────────────────────────

VWAS_PATH = Path("logs/vwaslr_trades.csv")
ORB_PATH  = Path("logs/orb_signals.csv")
CSR_PATH  = Path("logs/signals.csv")

POINT_VALUE = {"MES": 5.0, "MYM": 0.5, "M2K": 5.0, "MNQ": 2.0}


# ── Data loading ───────────────────────────────────────────────────────────────

def _ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def load_vwaslr() -> list[dict]:
    if not VWAS_PATH.exists():
        return []
    trades = []
    with open(VWAS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                trades.append({
                    "kind":        "VWASLR",
                    "symbol":      row["symbol"],
                    "direction":   1 if row["direction"] == "LONG" else -1,
                    "entry":       float(row.get("fill_price") or row["est_entry"]),
                    "fired_at":    _ts(row["fired_at"]),
                    "resolved_at": _ts(row["resolved_at"]),
                    "outcome":     row["outcome"],
                    "pnl_pts":     float(row["pnl_pts"]),
                })
            except Exception:
                continue
    return trades


def load_orb() -> list[dict]:
    if not ORB_PATH.exists():
        return []
    trades = []
    with open(ORB_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                trades.append({
                    "kind":        "ORB",
                    "symbol":      row["symbol"],
                    "direction":   1 if row["direction"] == "LONG" else -1,
                    "entry":       float(row["entry"]),
                    "fired_at":    _ts(row["fired_at"]),
                    "resolved_at": _ts(row["resolved_at"]),
                    "outcome":     row["outcome"],
                    "pnl_pts":     float(row["pnl_pts"]),
                })
            except Exception:
                continue
    return trades


def load_csr() -> list[dict]:
    if not CSR_PATH.exists():
        return []
    trades = []
    with open(CSR_PATH, newline="") as f:
        for row in csv.DictReader(f):
            try:
                trades.append({
                    "kind":        "CSR",
                    "symbol":      row["symbol"],
                    "direction":   1 if row["direction"] == "LONG" else -1,
                    "entry":       float(row["entry"]),
                    "fired_at":    _ts(row["fired_at"]),
                    "resolved_at": _ts(row["resolved_at"]),
                    "outcome":     row["outcome"],
                    "pnl_pts":     float(row["pnl_pts"]),
                })
            except Exception:
                continue
    return trades


# ── MFE / MAE via 1-min bars ───────────────────────────────────────────────────

def compute_mfe_mae(client: TopstepClient, trade: dict,
                    contract_id: str) -> tuple[float, float]:
    """
    Fetch 1-min bars for the trade hold window and return (MFE, MAE) in points.
    MFE > 0 means there was a favourable peak; MAE < 0 means a drawdown occurred.
    Returns (nan, nan) on fetch failure.
    """
    start = trade["fired_at"]
    end   = trade["resolved_at"] + timedelta(minutes=1)   # inclusive
    try:
        raw = client.get_bars(
            contract_id=contract_id,
            start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=200,
        )
    except Exception as e:
        print(f"    fetch error: {e}")
        return float("nan"), float("nan")

    if not raw:
        return float("nan"), float("nan")

    raw = list(reversed(raw))   # chronological
    entry = trade["entry"]
    d     = trade["direction"]

    if d == 1:   # LONG
        mfe = max(b["h"] - entry for b in raw)
        mae = min(b["l"] - entry for b in raw)
    else:        # SHORT
        mfe = max(entry - b["l"] for b in raw)
        mae = min(entry - b["h"] for b in raw)

    return mfe, mae


# ── Reporting ──────────────────────────────────────────────────────────────────

def _pnl_str(pts: float, sym: str) -> str:
    pv = POINT_VALUE.get(sym, 1.0)
    dollars = pts * pv
    sign = "+" if pts >= 0 else ""
    return f"{sign}{pts:.1f}pt (${sign}{dollars:.0f})"


def _capture(pnl: float, mfe: float) -> str:
    if mfe <= 0 or pnl != pnl:
        return "  —  "
    c = pnl / mfe * 100
    return f"{c:+.0f}%"


def print_table(trades: list[dict], results: list[dict]):
    kind_width = max(len(t["kind"]) for t in trades) + 1

    header = (f"  {'Date':>10}  {'Sym':>4}  {'Kind':<{kind_width}}  "
              f"{'Dir':>5}  {'Entry':>8}  "
              f"{'Actual':>12}  {'MFE':>10}  {'MAE':>10}  "
              f"{'Cap%':>6}  {'Outcome'}")
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    for t, r in zip(trades, results):
        date_str = t["fired_at"].astimezone().strftime("%m-%d %H:%M")
        dir_str  = "LONG " if t["direction"] == 1 else "SHORT"
        pnl      = t["pnl_pts"]
        mfe      = r["mfe"]
        mae      = r["mae"]
        cap      = _capture(pnl, mfe)

        mfe_str  = f"+{mfe:.1f}" if mfe == mfe else "   —"
        mae_str  = f"{mae:.1f}"  if mae == mae else "   —"
        pnl_str  = f"{'+' if pnl >= 0 else ''}{pnl:.1f}"

        print(f"  {date_str}  {t['symbol']:>4}  {t['kind']:<{kind_width}}  "
              f"{dir_str}  {t['entry']:>8.2f}  "
              f"{pnl_str:>12}  {mfe_str:>10}  {mae_str:>10}  "
              f"{cap:>6}  {t['outcome']}")


def print_summary(trades: list[dict], results: list[dict]):
    from collections import defaultdict
    by_kind: dict[str, list] = defaultdict(list)
    for t, r in zip(trades, results):
        if r["mfe"] != r["mfe"]:   # nan
            continue
        by_kind[t["kind"]].append((t, r))

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    for kind in ("VWASLR", "ORB", "CSR"):
        rows = by_kind.get(kind, [])
        if not rows:
            continue
        pnls  = [t["pnl_pts"] for t, _ in rows]
        mfes  = [r["mfe"]     for _, r in rows]
        maes  = [r["mae"]     for _, r in rows]
        caps  = [t["pnl_pts"] / r["mfe"] for t, r in rows if r["mfe"] > 0]

        n_target  = sum(1 for t, _ in rows if t["outcome"] == "TARGET")
        n_stopped = sum(1 for t, _ in rows if t["outcome"] == "STOPPED")
        n_time    = sum(1 for t, _ in rows if t["outcome"] == "TIME EXIT")

        print(f"\n  {kind}  ({len(rows)} trades)")
        print(f"    Actual PnL:    avg {sum(pnls)/len(pnls):+.2f}  "
              f"total {sum(pnls):+.1f}")
        print(f"    MFE:           avg {sum(mfes)/len(mfes):+.2f}  "
              f"total {sum(mfes):+.1f}")
        print(f"    MAE:           avg {sum(maes)/len(maes):+.2f}  "
              f"total {sum(maes):+.1f}")
        if caps:
            print(f"    Capture %:     avg {sum(caps)/len(caps)*100:+.0f}%  "
                  f"(how much of peak profit was kept)")
        wins  = sum(1 for p in pnls if p > 0)
        print(f"    Win rate:      {wins}/{len(rows)}  "
              f"(T={n_target} S={n_stopped} Tx={n_time})")

        # theoretical: exit at MFE
        print(f"    Theoretical (exit at MFE peak):")
        print(f"      total MFE profit: {sum(mfes):+.1f} pts  "
              f"(vs actual {sum(pnls):+.1f} pts)")
        total_by_sym: dict[str, dict] = defaultdict(lambda: {"actual": 0.0, "mfe": 0.0})
        for t, r in rows:
            total_by_sym[t["symbol"]]["actual"] += t["pnl_pts"]
            total_by_sym[t["symbol"]]["mfe"]    += r["mfe"]
        for sym, vals in sorted(total_by_sym.items()):
            pv = POINT_VALUE.get(sym, 1.0)
            print(f"        {sym}: actual {vals['actual']:+.1f} pt "
                  f"(${vals['actual']*pv:+.0f})  "
                  f"vs MFE {vals['mfe']:+.1f} pt "
                  f"(${vals['mfe']*pv:+.0f})")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vwas", action="store_true", help="VWASLR trades only")
    parser.add_argument("--orb",  action="store_true", help="ORB trades only")
    parser.add_argument("--csr",  action="store_true", help="CSR momentum trades only")
    args = parser.parse_args()

    trades: list[dict] = []
    if args.vwas or not any([args.vwas, args.orb, args.csr]):
        trades += load_vwaslr()
    if args.orb  or not any([args.vwas, args.orb, args.csr]):
        trades += load_orb()
    if args.csr  or not any([args.vwas, args.orb, args.csr]):
        trades += load_csr()

    trades.sort(key=lambda t: t["fired_at"])
    print(f"Loaded {len(trades)} trades — fetching bar data …")

    print("Connecting …")
    with TopstepClient() as client:
        # Cache contract IDs per symbol
        cid: dict[str, str] = {}
        for sym in sorted({t["symbol"] for t in trades}):
            contracts = client.search_contracts(sym)
            if contracts:
                cid[sym] = contracts[0]["id"]
                print(f"  {sym}: {contracts[0]['name']}  id={cid[sym]}")
            else:
                print(f"  {sym}: NOT FOUND")

        results: list[dict] = []
        for i, trade in enumerate(trades, 1):
            sym = trade["symbol"]
            if sym not in cid:
                results.append({"mfe": float("nan"), "mae": float("nan")})
                continue
            print(f"  [{i}/{len(trades)}] {sym} {trade['kind']} "
                  f"{trade['fired_at'].strftime('%m-%d %H:%M')} …", end="", flush=True)
            mfe, mae = compute_mfe_mae(client, trade, cid[sym])
            results.append({"mfe": mfe, "mae": mae})
            tag = f"  MFE={mfe:+.1f}  MAE={mae:+.1f}" if mfe == mfe else "  (no data)"
            print(tag)
            time.sleep(0.25)   # polite pacing

    print_table(trades, results)
    print_summary(trades, results)
