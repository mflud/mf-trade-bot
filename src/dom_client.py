"""
TopstepX real-time Depth of Market (DOM) client via SignalR WebSocket.

Connects to the ProjectX Market Hub and subscribes to:
  - SubscribeContractMarketDepth  → GatewayDepth events (bid/ask size at each price)
  - SubscribeContractQuotes       → GatewayQuote events (best bid/ask, last price)
  - SubscribeContractTrades       → GatewayTrade events (individual prints)

Usage:
  python src/dom_client.py             # live snapshot + top-of-book display
  python src/dom_client.py --levels 20 # show 20 levels each side
  python src/dom_client.py --record    # also save DOM snapshots every 5 min to CSV
  python src/dom_client.py --record --record-interval 1  # every 1 min
"""

import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from signalrcore.hub_connection_builder import HubConnectionBuilder

sys.path.insert(0, "src")
from topstep_client import TopstepClient

load_dotenv()

MARKET_HUB_BASE = "https://rtc.topstepx.com/hubs/market"


# ── DOM state ─────────────────────────────────────────────────────────────────

class DOMBook:
    """
    Maintains a live order book from incremental GatewayDepth updates.

    DomType enum (observed from live API, not documented):
      3 = Ask  (sell side — above market)
      4 = Bid  (buy side — below market)
      5 = Last trade
      6 = Reset / clear book
      7 = Session low
      8 = Session high

    Each update arrives as a list of dicts, each with: price, volume, type.
      volume == 0  → remove that level
      volume  > 0  → set/update that level
    """
    BID   = 4
    ASK   = 3
    RESET = 6

    def __init__(self):
        self.bids: dict[float, float] = {}   # price → size
        self.asks: dict[float, float] = {}
        self.last_price: float | None = None
        self.best_bid:   float | None = None
        self.best_ask:   float | None = None
        self.last_update = datetime.now(timezone.utc)
        self._lock = threading.Lock()

    def apply_depth(self, updates: list[dict]):
        with self._lock:
            for u in updates:
                if u is None:
                    continue
                price    = u.get("price",  0)
                volume   = u.get("volume", 0)
                dom_type = u.get("type",   -1)
                if dom_type == self.RESET:
                    self.bids.clear()
                    self.asks.clear()
                elif dom_type == self.BID:
                    if volume == 0:
                        self.bids.pop(price, None)
                    else:
                        self.bids[price] = volume
                elif dom_type == self.ASK:
                    if volume == 0:
                        self.asks.pop(price, None)
                    else:
                        self.asks[price] = volume
            self.last_update = datetime.now(timezone.utc)

    def apply_quote(self, last_price, best_bid, best_ask):
        with self._lock:
            if last_price is not None: self.last_price = last_price
            if best_bid   is not None: self.best_bid   = best_bid
            if best_ask   is not None: self.best_ask   = best_ask
            self.last_update = datetime.now(timezone.utc)

    def snapshot(self, n_levels: int = 10):
        """Return top n bid/ask levels sorted correctly."""
        with self._lock:
            bids = sorted(self.bids.items(), reverse=True)[:n_levels]
            asks = sorted(self.asks.items())[:n_levels]
            return (list(bids), list(asks),
                    self.last_price, self.best_bid, self.best_ask,
                    self.last_update)

    def record_features(self, n_levels: int = 10) -> dict:
        """Return a flat dict of ML-ready features from current book state."""
        with self._lock:
            last  = self.last_price
            bb    = self.best_bid
            ba    = self.best_ask
            mid   = (bb + ba) / 2 if bb and ba else last
            # Filter out crossed levels — stale bids above best_ask or asks below best_bid
            # that weren't removed during fast price moves
            if bb and ba:
                bids = sorted(((p, s) for p, s in self.bids.items() if p < ba), reverse=True)[:n_levels]
                asks = sorted((p, s) for p, s in self.asks.items() if p > bb)[:n_levels]
            else:
                bids = sorted(self.bids.items(), reverse=True)[:n_levels]
                asks = sorted(self.asks.items())[:n_levels]

        row: dict = {
            "ts":        self.last_update.isoformat(),
            "last":      last,
            "best_bid":  bb,
            "best_ask":  ba,
            "spread":    round(ba - bb, 4) if bb and ba else None,
            "mid":       round(mid, 4)     if mid else None,
        }

        # Level-by-level bid/ask price and size
        bid_sizes, ask_sizes = [], []
        for lvl in range(n_levels):
            if lvl < len(bids):
                p, s = bids[lvl]
                row[f"bid_price_{lvl+1}"] = p
                row[f"bid_size_{lvl+1}"]  = s
                bid_sizes.append(s)
            else:
                row[f"bid_price_{lvl+1}"] = None
                row[f"bid_size_{lvl+1}"]  = None
                bid_sizes.append(0.0)

            if lvl < len(asks):
                p, s = asks[lvl]
                row[f"ask_price_{lvl+1}"] = p
                row[f"ask_size_{lvl+1}"]  = s
                ask_sizes.append(s)
            else:
                row[f"ask_price_{lvl+1}"] = None
                row[f"ask_size_{lvl+1}"]  = None
                ask_sizes.append(0.0)

        # Derived imbalance features at L1, L5, L10
        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        denom     = total_bid + total_ask
        row["total_bid_size"] = total_bid
        row["total_ask_size"] = total_ask
        row["bid_ask_imbalance_l10"] = (
            round((total_bid - total_ask) / denom, 4) if denom else None
        )
        for k in [1, 5]:
            b = sum(bid_sizes[:k])
            a = sum(ask_sizes[:k])
            d = b + a
            row[f"bid_ask_imbalance_l{k}"] = round((b - a) / d, 4) if d else None

        # Largest resting order (wall) on each side and its distance from mid
        if bid_sizes and any(bid_sizes):
            max_bid_idx = bid_sizes.index(max(bid_sizes))
            row["max_bid_wall_size"] = bid_sizes[max_bid_idx]
            row["max_bid_wall_dist"] = (
                round(mid - bids[max_bid_idx][0], 4)
                if max_bid_idx < len(bids) and mid else None
            )
        else:
            row["max_bid_wall_size"] = None
            row["max_bid_wall_dist"] = None

        if ask_sizes and any(ask_sizes):
            max_ask_idx = ask_sizes.index(max(ask_sizes))
            row["max_ask_wall_size"] = ask_sizes[max_ask_idx]
            row["max_ask_wall_dist"] = (
                round(asks[max_ask_idx][0] - mid, 4)
                if max_ask_idx < len(asks) and mid else None
            )
        else:
            row["max_ask_wall_size"] = None
            row["max_ask_wall_dist"] = None

        return row


# ── Recorder ─────────────────────────────────────────────────────────────────

DOM_SNAPSHOT_LEVELS = 10   # levels to record per side

class DOMRecorder:
    """
    Captures DOM book snapshots at regular intervals (aligned to clock minutes)
    and appends ML-ready feature rows to a CSV file.

    Columns:
      ts, last, best_bid, best_ask, spread, mid,
      bid_price_1..10, bid_size_1..10,
      ask_price_1..10, ask_size_1..10,
      total_bid_size, total_ask_size,
      bid_ask_imbalance_l1, bid_ask_imbalance_l5, bid_ask_imbalance_l10,
      max_bid_wall_size, max_bid_wall_dist,
      max_ask_wall_size, max_ask_wall_dist,
      contract
    """

    def __init__(self, book: "DOMBook", contract: str,
                 out_path: str = "mes_dom_snapshots.csv",
                 interval_minutes: int = 5):
        self.book              = book
        self.contract          = contract
        self.out_path          = Path(out_path)
        self.interval_minutes  = interval_minutes
        self._thread: threading.Thread | None = None
        self._stop_event       = threading.Event()
        self._wrote_header     = self.out_path.exists() and self.out_path.stat().st_size > 0

    # ── public API ────────────────────────────────────────────────────────────

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[recorder] Saving DOM snapshots every {self.interval_minutes} min → {self.out_path}")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self):
        """Sleep until the next aligned minute boundary, then record."""
        while not self._stop_event.is_set():
            now = datetime.now(timezone.utc)
            mins_to_next = self.interval_minutes - (now.minute % self.interval_minutes)
            secs_to_next = mins_to_next * 60 - now.second - now.microsecond / 1e6
            # Sleep in small chunks so stop_event is checked promptly
            deadline = time.monotonic() + secs_to_next
            while time.monotonic() < deadline and not self._stop_event.is_set():
                time.sleep(min(1.0, deadline - time.monotonic()))
            if not self._stop_event.is_set():
                self._capture()

    def _capture(self):
        row = self.book.record_features(n_levels=DOM_SNAPSHOT_LEVELS)
        row["contract"] = self.contract
        ts_str = row["ts"]

        # Write to CSV (append; write header once)
        file_exists = self._wrote_header
        with open(self.out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)

        print(f"[recorder] {ts_str}  last={row['last']}  "
              f"imb_l5={row.get('bid_ask_imbalance_l5')}  "
              f"saved → {self.out_path}")


# ── Display ───────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
CLEAR  = "\033[2J\033[H"


def bar(size: float, max_size: float, width: int = 20, ch: str = "█") -> str:
    filled = int(round(size / max_size * width)) if max_size else 0
    return ch * filled + "░" * (width - filled)


def render(book: DOMBook, contract: str, n_levels: int = 10):
    bids, asks, last, best_bid, best_ask, updated = book.snapshot(n_levels)

    all_sizes = [s for _, s in bids] + [s for _, s in asks]
    max_size  = max(all_sizes) if all_sizes else 1

    lines = []
    lines.append(CLEAR)
    lines.append(f"{BOLD}{CYAN}  DOM  │  {contract}  │  "
                 f"{updated.astimezone().strftime('%H:%M:%S %Z')}{RESET}")
    lines.append(f"  Last: {BOLD}{last or '—'}{RESET}   "
                 f"Bid: {GREEN}{best_bid or '—'}{RESET}   "
                 f"Ask: {RED}{best_ask or '—'}{RESET}")
    lines.append("")

    col_w = 10
    lines.append(f"  {DIM}{'SIZE':>{col_w}}  {'BAR':<20}  {'PRICE':^10}  "
                 f"{'BAR':<20}  {'SIZE':<{col_w}}{RESET}")
    lines.append(f"  {'─'*76}")

    # Asks (sell side) — show lowest ask at bottom of ask block
    for price, size in reversed(asks):
        b = bar(size, max_size, ch="▓")
        lines.append(f"  {DIM}{size:>{col_w},.0f}  {' '*20}  "
                     f"{RED}{price:^10.2f}{RESET}  "
                     f"{RED}{b:<20}{RESET}  {RED}{size:<{col_w},.0f}{RESET}")

    # Spread
    spread = (best_ask - best_bid) if best_bid and best_ask else None
    spread_str = f"  spread: {spread:.2f} pts" if spread else ""
    lines.append(f"  {'·'*38}{YELLOW}{spread_str}{RESET}")

    # Bids (buy side) — show highest bid at top of bid block
    for price, size in bids:
        b = bar(size, max_size, ch="▓")
        lines.append(f"  {GREEN}{size:>{col_w},.0f}  {b:<20}{RESET}  "
                     f"{GREEN}{price:^10.2f}{RESET}  "
                     f"{DIM}{' '*20}  {' ' * col_w}{RESET}")

    lines.append("")
    lines.append(f"  {DIM}Total bid liquidity: "
                 f"{sum(s for _,s in bids):,.0f}   "
                 f"Total ask liquidity: {sum(s for _,s in asks):,.0f}{RESET}")
    lines.append(f"  {DIM}Bid/Ask ratio: "
                 f"{sum(s for _,s in bids)/sum(s for _,s in asks):.2f}x"
                 if asks and bids else "")

    return "\n".join(lines)


# ── SignalR connection ────────────────────────────────────────────────────────

def build_connection(token: str, contract_id: str,
                     book: DOMBook, verbose: bool = False):
    url = f"{MARKET_HUB_BASE}?access_token={token}"

    hub = (HubConnectionBuilder()
           .with_url(url)
           .with_automatic_reconnect({
               "type": "raw",
               "keep_alive_interval": 10,
               "reconnect_interval": [1, 2, 5, 10],
           })
           .build())

    # ── Event handlers ────────────────────────────────────────────────────────

    def _extract(args):
        """
        SignalR events arrive as [contractId, payload] or just [payload].
        Find the first dict in args.
        """
        for a in args:
            if isinstance(a, dict):
                return a
        return None

    def on_depth(args):
        """GatewayDepth: [contractId, [{"price", "volume", "type"}, ...]]"""
        if verbose:
            print(f"[depth] {args}")
        # Payload is a list of update dicts (second element after contractId)
        updates = next((a for a in args if isinstance(a, list)), None)
        if updates:
            book.apply_depth(updates)

    def on_quote(args):
        """GatewayQuote: [contractId, {lastPrice, bestBid, bestAsk, volume, ...}]"""
        if verbose:
            print(f"[quote] {args}")
        msg = _extract(args)
        if msg is None:
            return
        book.apply_quote(
            msg.get("lastPrice"),
            msg.get("bestBid"),
            msg.get("bestAsk"),
        )

    def on_trade(args):
        if verbose:
            print(f"[trade] {args}")

    def on_error(args):
        print(f"[error] {args}")

    hub.on("GatewayDepth", on_depth)
    hub.on("GatewayQuote", on_quote)
    hub.on("GatewayTrade", on_trade)
    hub.on_error(on_error)

    # ── Subscribe after connect ───────────────────────────────────────────────

    def on_open():
        print(f"Connected to Market Hub — subscribing to {contract_id}")
        hub.send("SubscribeContractQuotes",      [contract_id])
        hub.send("SubscribeContractMarketDepth", [contract_id])
        hub.send("SubscribeContractTrades",      [contract_id])

    hub.on_open(on_open)

    return hub


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels",  type=int,  default=10,
                        help="DOM levels to display each side (default 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw SignalR messages")
    parser.add_argument("--contract", type=str, default=None,
                        help="Contract id (e.g. CON.F.US.MES.H26). "
                             "Defaults to front-month MES.")
    parser.add_argument("--record", action="store_true",
                        help="Save DOM snapshots to CSV for ML training")
    parser.add_argument("--record-interval", type=int, default=5,
                        help="Snapshot interval in minutes (default 5)")
    parser.add_argument("--record-out", type=str, default="mes_dom_snapshots.csv",
                        help="Output CSV path (default mes_dom_snapshots.csv)")
    args = parser.parse_args()

    # Authenticate
    client = TopstepClient()
    token  = client.login()
    print(f"Authenticated.")

    # Resolve contract
    if args.contract:
        contract_id   = args.contract
        contract_name = args.contract
    else:
        contracts     = client.search_contracts("MES")
        contract      = contracts[0]
        contract_id   = contract["id"]
        contract_name = contract["name"]
    print(f"Contract: {contract_name}  id={contract_id}")

    book = DOMBook()
    hub  = build_connection(token, contract_id, book, verbose=args.verbose)

    recorder = None
    if args.record:
        recorder = DOMRecorder(
            book, contract_name,
            out_path=args.record_out,
            interval_minutes=args.record_interval,
        )

    # ── Connection loop — rebuilds hub with fresh token on disconnect ─────────
    _stop = threading.Event()

    def _run_hub():
        nonlocal hub
        while not _stop.is_set():
            try:
                token = client.login()
                book.bids.clear()   # clear stale levels before fresh subscribe
                book.asks.clear()
                hub   = build_connection(token, contract_id, book, verbose=args.verbose)
                hub.start()
                time.sleep(3)   # let first batch of depth events arrive
                # Block until the connection drops (signalrcore calls on_close / raises)
                # We detect a dead connection by watching book.last_update stall
                while not _stop.is_set():
                    time.sleep(30)
                    age = (datetime.now(timezone.utc) - book.last_update).total_seconds()
                    if age > 120:
                        print(f"[dom] No updates for {age:.0f}s — reconnecting with fresh token…")
                        break
                hub.stop()
            except Exception as e:
                print(f"[dom] Hub error: {e} — reconnecting in 10s…")
                time.sleep(10)

    hub_thread = threading.Thread(target=_run_hub, daemon=True)
    hub_thread.start()
    time.sleep(4)   # let first connection establish

    if recorder:
        recorder.start()

    try:
        if args.record:
            # Headless recording mode — no display, just block until killed
            threading.Event().wait()
        else:
            while True:
                print(render(book, contract_name, n_levels=args.levels),
                      end="", flush=True)
                time.sleep(0.5)   # refresh twice per second
    except KeyboardInterrupt:
        print("\nDisconnecting…")
    finally:
        _stop.set()
        if recorder:
            recorder.stop()
        try:
            hub.send("UnsubscribeContractMarketDepth", [contract_id])
            hub.send("UnsubscribeContractQuotes",      [contract_id])
            hub.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
