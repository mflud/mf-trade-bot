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
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone, time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from signalrcore.hub_connection_builder import HubConnectionBuilder

# signalrcore logs every WebSocket disconnect at ERROR level — suppress to WARNING
# so routine reconnects don't flood the log file.
logging.getLogger("SignalRCoreClient").setLevel(logging.CRITICAL)

sys.path.insert(0, "src")
from topstep_client import TopstepClient

load_dotenv()

MARKET_HUB_BASE = "https://rtc.topstepx.com/hubs/market"
DOM_DB_PATH     = Path("data/dom.db")

_CT = ZoneInfo("America/Chicago")


# ── Shared DOM state DB (read by slr_monitor without its own WebSocket) ───────

def _init_dom_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dom_live (
            symbol     TEXT PRIMARY KEY,
            updated    TEXT NOT NULL,
            last_price REAL,
            best_bid   REAL,
            best_ask   REAL,
            bids_json  TEXT NOT NULL,
            asks_json  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _dom_writer_loop(books_by_sym: dict, db_conn: sqlite3.Connection,
                     interval: float = 0.2):
    """Write live DOM state to dom.db every interval seconds."""
    db_lock = threading.Lock()
    while True:
        time.sleep(interval)
        now = datetime.now(timezone.utc).isoformat()
        for sym, book in books_by_sym.items():
            with book._lock:
                bids = {str(k): v for k, v in book.bids.items()}
                asks = {str(k): v for k, v in book.asks.items()}
                last = book.last_price
                bb   = book.best_bid
                ba   = book.best_ask
            try:
                with db_lock:
                    db_conn.execute("""
                        INSERT OR REPLACE INTO dom_live
                        (symbol, updated, last_price, best_bid, best_ask,
                         bids_json, asks_json)
                        VALUES (?,?,?,?,?,?,?)
                    """, (sym, now, last, bb, ba,
                          json.dumps(bids), json.dumps(asks)))
                    db_conn.commit()
            except Exception:
                pass

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

    _ET             = ZoneInfo("America/New_York")
    _SESSION_START  = dtime(9,  0)
    _SESSION_END    = dtime(17, 0)

    def _in_active_session(self) -> bool:
        t = datetime.now(self._ET).time()
        return self._SESSION_START <= t < self._SESSION_END

    def _capture(self):
        if not self._in_active_session():
            return   # outside 09:00–17:00 ET — skip snapshot

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

def build_connection(token: str, books: dict, contract_ids: list,
                     verbose: bool = False):
    """
    books       : {contract_id: DOMBook}
    contract_ids: list of contract IDs to subscribe to
    """
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
        for a in args:
            if isinstance(a, dict):
                return a
        return None

    def _book_for(args):
        """Route to the correct DOMBook by contractId (first arg)."""
        cid = str(args[0]) if args else ""
        return books.get(cid)

    def on_depth(args):
        """GatewayDepth: [contractId, [{"price", "volume", "type"}, ...]]"""
        if verbose:
            print(f"[depth] {args}")
        book = _book_for(args)
        if book:
            updates = next((a for a in args if isinstance(a, list)), None)
            if updates:
                book.apply_depth(updates)

    def on_quote(args):
        """GatewayQuote: [contractId, {lastPrice, bestBid, bestAsk, ...}]"""
        if verbose:
            print(f"[quote] {args}")
        book = _book_for(args)
        if book:
            msg = _extract(args)
            if msg:
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
        syms = ", ".join(contract_ids)
        print(f"Connected to Market Hub — subscribing to {syms}")
        for cid in contract_ids:
            hub.send("SubscribeContractQuotes",      [cid])
            hub.send("SubscribeContractMarketDepth", [cid])
            hub.send("SubscribeContractTrades",      [cid])

    hub.on_open(on_open)

    return hub


# ── Session guard ─────────────────────────────────────────────────────────────

_SESSION_ET_START = dtime(8, 30)
_SESSION_ET_END   = dtime(17, 0)
_SESSION_ET_TZ    = ZoneInfo("America/New_York")


def _wait_for_session():
    """Block until 08:30–17:00 ET on a Mon–Fri weekday, sleeping in 30-min chunks."""
    while True:
        now_et = datetime.now(_SESSION_ET_TZ)
        wd  = now_et.weekday()   # 0=Mon…4=Fri, 5=Sat, 6=Sun
        t   = now_et.time()
        if wd < 5 and _SESSION_ET_START <= t < _SESSION_ET_END:
            return
        target = now_et.replace(hour=8, minute=30, second=0, microsecond=0)
        if t >= _SESSION_ET_START:
            from datetime import timedelta as _td
            target += _td(days=1)
        while target.weekday() >= 5:
            from datetime import timedelta as _td
            target += _td(days=1)
        wait = (target - now_et).total_seconds()
        print(f"[session] Outside 08:30–17:00 ET — sleeping {wait/3600:.1f}h until "
              f"{target.strftime('%a %H:%M ET')}")
        time.sleep(min(wait, 1800))


def _start_session_exit_watcher():
    """Background thread: exit cleanly at 17:00 ET so launchd restarts us fresh tomorrow."""
    def _watch():
        while True:
            time.sleep(30)
            now_et = datetime.now(_SESSION_ET_TZ)
            if now_et.time() >= _SESSION_ET_END:
                print("[session] 17:00 ET — session ended, exiting cleanly")
                import os
                os._exit(0)   # hard kill — SIGTERM can be caught by signalrcore
    threading.Thread(target=_watch, daemon=True, name="session-exit").start()


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_INSTRUMENTS = ["MES", "MNQ", "ES", "NQ"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruments", type=str,
                        default=",".join(DEFAULT_INSTRUMENTS),
                        help="Comma-separated instrument list "
                             f"(default: {','.join(DEFAULT_INSTRUMENTS)})")
    parser.add_argument("--levels",  type=int,  default=10,
                        help="DOM levels to display each side (default 10)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print raw SignalR messages")
    parser.add_argument("--record", action="store_true",
                        help="Save DOM snapshots to CSV for ML training")
    parser.add_argument("--record-interval", type=int, default=5,
                        help="Snapshot interval in minutes (default 5)")
    args = parser.parse_args()

    syms = [s.strip().upper() for s in args.instruments.split(",") if s.strip()]

    _wait_for_session()
    _start_session_exit_watcher()

    # Authenticate
    client = TopstepClient()
    token  = client.login()
    print(f"Authenticated.")

    # Resolve contracts
    contracts_info = {}   # sym → {id, name}
    for sym in syms:
        results = client.search_contracts(sym)
        if not results:
            print(f"WARNING: no contract found for {sym} — skipping")
            continue
        c = results[0]
        contracts_info[sym] = {"id": c["id"], "name": c["name"]}
        print(f"  {sym}: {c['name']}  id={c['id']}")

    if not contracts_info:
        print("ERROR: no contracts resolved — exiting")
        return

    # Per-instrument DOMBook and DOMRecorder
    books     = {}   # contract_id → DOMBook
    sym_by_id = {}   # contract_id → sym
    recorders = []

    for sym, info in contracts_info.items():
        cid        = info["id"]
        book       = DOMBook()
        books[cid] = book
        sym_by_id[cid] = sym
        if args.record:
            out_path = Path("data") / f"{sym.lower()}_dom_snapshots.csv"
            rec = DOMRecorder(book, info["name"],
                              out_path=str(out_path),
                              interval_minutes=args.record_interval)
            recorders.append(rec)

    contract_ids = list(books.keys())
    hub = build_connection(token, books, contract_ids, verbose=args.verbose)

    # ── Connection loop — rebuilds hub with fresh token on disconnect ─────────
    _stop = threading.Event()

    def _run_hub():
        nonlocal hub
        while not _stop.is_set():
            if is_cme_weekend_closure():
                now_ct = datetime.now(_CT)
                print(f"[dom] CME weekend closure — pausing until Sunday 17:00 CT "
                      f"(now {now_ct.strftime('%a %H:%M %Z')})")
                for _ in range(300):
                    if _stop.is_set():
                        return
                    time.sleep(1)
                continue
            try:
                token = client.login()
                for book in books.values():
                    book.bids.clear()
                    book.asks.clear()
                hub = build_connection(token, books, contract_ids,
                                       verbose=args.verbose)
                hub.start()
                time.sleep(3)
                # Detect dead connection by watching for any book update
                while not _stop.is_set():
                    time.sleep(30)
                    most_recent = max(
                        (b.last_update for b in books.values()),
                        default=datetime.now(timezone.utc))
                    age = (datetime.now(timezone.utc) - most_recent).total_seconds()
                    if age > 120:
                        print(f"[dom] No updates for {age:.0f}s — reconnecting…")
                        break
                hub.stop()
            except Exception as e:
                print(f"[dom] Hub error: {e} — reconnecting in 10s…")
                time.sleep(10)

    hub_thread = threading.Thread(target=_run_hub, daemon=True)
    hub_thread.start()
    time.sleep(4)

    # Write live DOM state to data/dom.db so slr_monitor can read it
    # without opening its own WebSocket connection
    books_by_sym = {sym: books[info["id"]] for sym, info in contracts_info.items()}
    dom_db = _init_dom_db(DOM_DB_PATH)
    threading.Thread(target=_dom_writer_loop, args=(books_by_sym, dom_db),
                     daemon=True, name="dom-db-writer").start()

    for rec in recorders:
        rec.start()

    try:
        if args.record:
            # Headless recording mode — block until killed
            threading.Event().wait()
        else:
            # Display first instrument interactively
            first_sym = syms[0]
            first_cid = contracts_info[first_sym]["id"]
            first_name = contracts_info[first_sym]["name"]
            while True:
                print(render(books[first_cid], first_name, n_levels=args.levels),
                      end="", flush=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nDisconnecting…")
    finally:
        _stop.set()
        for rec in recorders:
            rec.stop()
        try:
            for cid in contract_ids:
                hub.send("UnsubscribeContractMarketDepth", [cid])
                hub.send("UnsubscribeContractQuotes",      [cid])
            hub.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
