"""
bar_collector.py — Real-time bar builder + DOM writer via TopstepX SignalR WebSocket.

Subscribes to GatewayTrade ticks for all configured instruments on the
TopstepX market hub (rtc.topstepx.com/hubs/market) and builds 1-min and
5-min OHLCV bars in real-time.  Completed bars are written to a shared
SQLite database (data/bars.db) so that signal_monitor, trading_bot, and
globex_monitor can read them without making their own API calls.

Also subscribes to GatewayDepth and GatewayQuote on the same connection
and writes live DOM state to data/dom.db every 200ms (read by slr_monitor).
This keeps the total number of TopstepX market hub sessions at one, avoiding
the "multiple sessions detected" disconnect.

Uses websocket-client directly (not signalrcore) with a minimal SignalR
JSON protocol implementation for reliability.

Usage:
    python src/bar_collector.py          # live mode
    python src/bar_collector.py --demo   # print ticks, no DB writes
"""

import argparse
import json
import logging
import os
import sqlite3
import ssl
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

import websocket

sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from topstep_client import TopstepClient

# ── Config ────────────────────────────────────────────────────────────────────

MARKET_HUB_URL = "wss://rtc.topstepx.com/hubs/market"
USER_HUB_URL   = "wss://rtc.topstepx.com/hubs/user"

# Instruments to subscribe to.  symbol → search term for contract lookup.
INSTRUMENTS = {
    "MES": "MES",
    "MNQ": "MNQ",
    "ES":  "ES",
    "NQ":  "NQ",
}

# Bar sizes to build (in minutes).
BAR_SIZES = [1, 5]

# Bar sizes to build in seconds (stored in separate bars5s table).
BAR_SIZES_SEC = [5]

# CME settlement gap: 21:00–22:00 UTC daily (16:00–17:00 ET).
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

# Active session: 09:00–17:00 ET (14:00–22:00 UTC).  Ticks outside this window
# are still received via WebSocket but not written to the DB.
# 17:00 ET = 14:00 MST; TopstepX data is available until then.
ACTIVE_ET_START = (9,  0)   # (hour, minute)
ACTIVE_ET_END   = (17, 0)

# SignalR record separator
RS = "\x1e"

# Output paths
DB_DIR   = Path("data")
BARS_DB  = DB_DIR / "bars.db"
FILLS_DB = DB_DIR / "fills.db"
DOM_DB   = DB_DIR / "dom.db"

# DOM writer interval
DOM_WRITE_INTERVAL = 0.2   # seconds

log = logging.getLogger("collector")

# SSL context that skips certificate verification (needed on macOS Python.org builds)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class LiveBar:
    """Partially-built bar accumulating ticks."""
    symbol:    str
    minutes:   int
    bar_ts:    datetime
    open:      float = 0.0
    high:      float = 0.0
    low:       float = 0.0
    close:     float = 0.0
    volume:    float = 0.0
    tick_count: int  = 0

    def apply(self, price: float, size: float):
        if self.tick_count == 0:
            self.open = self.high = self.low = price
        else:
            self.high = max(self.high, price)
            self.low  = min(self.low,  price)
        self.close      = price
        self.volume    += size
        self.tick_count += 1

    def to_row(self) -> tuple:
        return (
            self.symbol, self.minutes, self.bar_ts.isoformat(),
            self.open, self.high, self.low, self.close, self.volume,
        )


@dataclass
class SymbolState:
    symbol:      str
    contract_id: str
    bars:        dict[int, LiveBar] = field(default_factory=dict)
    bars_sec:    dict[int, LiveBar] = field(default_factory=dict)  # second-resolution bars
    dom:         "DOMBook" = field(default_factory=lambda: DOMBook())


# ── DOM book ─────────────────────────────────────────────────────────────────

class DOMBook:
    """
    Maintains a live order book from incremental GatewayDepth updates.

    DomType enum (observed from live API):
      3 = Ask, 4 = Bid, 5 = Last trade, 6 = Reset, 7 = Session low, 8 = Session high
    """
    BID   = 4
    ASK   = 3
    RESET = 6

    def __init__(self):
        self.bids: dict[float, float] = {}
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


def init_dom_db(db_path: Path) -> sqlite3.Connection:
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


def dom_writer_loop(books_by_sym: dict[str, DOMBook], db_conn: sqlite3.Connection,
                    interval: float = DOM_WRITE_INTERVAL):
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
                        (symbol, updated, last_price, best_bid, best_ask, bids_json, asks_json)
                        VALUES (?,?,?,?,?,?,?)
                    """, (sym, now, last, bb, ba,
                          json.dumps(bids), json.dumps(asks)))
                    db_conn.commit()
            except Exception:
                pass


# ── Database ──────────────────────────────────────────────────────────────────

_db_lock = threading.Lock()


def init_bars_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol   TEXT    NOT NULL,
            minutes  INTEGER NOT NULL,
            ts       TEXT    NOT NULL,
            open     REAL    NOT NULL,
            high     REAL    NOT NULL,
            low      REAL    NOT NULL,
            close    REAL    NOT NULL,
            volume   REAL    NOT NULL,
            PRIMARY KEY (symbol, minutes, ts)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bars_sym_min_ts
        ON bars (symbol, minutes, ts DESC)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars5s (
            symbol  TEXT NOT NULL,
            ts      TEXT NOT NULL,
            open    REAL NOT NULL,
            high    REAL NOT NULL,
            low     REAL NOT NULL,
            close   REAL NOT NULL,
            volume  REAL NOT NULL,
            PRIMARY KEY (symbol, ts)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bars5s_sym_ts
        ON bars5s (symbol, ts DESC)
    """)
    # Trim bars5s to last 10,000 rows per symbol on startup to bound growth
    for sym in ["MES", "MNQ", "ES", "NQ"]:
        conn.execute("""
            DELETE FROM bars5s WHERE symbol = ? AND ts NOT IN (
                SELECT ts FROM bars5s WHERE symbol = ?
                ORDER BY ts DESC LIMIT 10000
            )
        """, (sym, sym))
    conn.commit()
    return conn


def init_fills_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            received_at  TEXT    NOT NULL,
            event_type   TEXT    NOT NULL,
            payload      TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def _bar_open_ts_sec(tick_ts: datetime, seconds: int) -> datetime:
    epoch   = tick_ts.timestamp()
    floored = int(epoch // seconds) * seconds
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def write_bar(conn: sqlite3.Connection, bar: LiveBar, demo: bool = False):
    if demo:
        log.info(
            f"[DEMO] BAR {bar.symbol} {bar.minutes}m  {bar.bar_ts.isoformat()}  "
            f"O={bar.open}  H={bar.high}  L={bar.low}  C={bar.close}  V={bar.volume:.0f}"
        )
        return
    with _db_lock:
        conn.execute("""
            INSERT OR REPLACE INTO bars (symbol, minutes, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, bar.to_row())
        conn.commit()
    log.debug(
        f"BAR {bar.symbol} {bar.minutes}m  {bar.bar_ts.strftime('%H:%M')}  "
        f"C={bar.close}  V={bar.volume:.0f}"
    )


def write_bar_5s(conn: sqlite3.Connection, bar: LiveBar, demo: bool = False):
    if demo:
        log.debug(
            f"[DEMO] BAR5S {bar.symbol} 5s  {bar.bar_ts.isoformat()}  "
            f"O={bar.open}  H={bar.high}  L={bar.low}  C={bar.close}  V={bar.volume:.0f}"
        )
        return
    with _db_lock:
        conn.execute("""
            INSERT OR REPLACE INTO bars5s (symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (bar.symbol, bar.bar_ts.isoformat(),
              bar.open, bar.high, bar.low, bar.close, bar.volume))
        conn.commit()


# ── Bar building ──────────────────────────────────────────────────────────────

def _bar_open_ts(tick_ts: datetime, minutes: int) -> datetime:
    epoch   = tick_ts.timestamp()
    floored = int(epoch // (minutes * 60)) * (minutes * 60)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


_ET = ZoneInfo("America/New_York")

def _in_settlement(tick_ts: datetime) -> bool:
    h = tick_ts.astimezone(timezone.utc).hour
    return SETTLEMENT_UTC_START <= h < SETTLEMENT_UTC_END

def _in_active_session(tick_ts: datetime) -> bool:
    """Return True if tick_ts falls within the 09:00–17:00 ET trading window."""
    et = tick_ts.astimezone(_ET)
    hm = (et.hour, et.minute)
    return ACTIVE_ET_START <= hm < ACTIVE_ET_END


def on_trade(symbol: str, state: SymbolState, bars_conn: sqlite3.Connection,
             trade: dict, demo: bool):
    try:
        price  = float(trade.get("price", 0))
        size   = float(trade.get("volume", trade.get("size", 1)))
        ts_raw = trade.get("timestamp") or trade.get("ts") or trade.get("time")
        if not price or not ts_raw:
            return

        tick_ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        if tick_ts.tzinfo is None:
            tick_ts = tick_ts.replace(tzinfo=timezone.utc)

        if _in_settlement(tick_ts):
            return

        if not _in_active_session(tick_ts):
            return

        if demo:
            log.info(f"TICK {symbol}  {price}  sz={size:.0f}  {tick_ts.strftime('%H:%M:%S')}")

        for minutes in BAR_SIZES:
            bar_ts  = _bar_open_ts(tick_ts, minutes)
            current = state.bars.get(minutes)

            if current is None or bar_ts > current.bar_ts:
                if current is not None and current.tick_count > 0:
                    write_bar(bars_conn, current, demo)
                state.bars[minutes] = LiveBar(
                    symbol=symbol, minutes=minutes, bar_ts=bar_ts
                )

            state.bars[minutes].apply(price, size)

        for seconds in BAR_SIZES_SEC:
            bar_ts  = _bar_open_ts_sec(tick_ts, seconds)
            current = state.bars_sec.get(seconds)

            if current is None or bar_ts > current.bar_ts:
                if current is not None and current.tick_count > 0:
                    write_bar_5s(bars_conn, current, demo)
                state.bars_sec[seconds] = LiveBar(
                    symbol=symbol, minutes=0, bar_ts=bar_ts
                )

            state.bars_sec[seconds].apply(price, size)

    except Exception as e:
        log.warning(f"on_trade error ({symbol}): {e}")


# ── SignalR WebSocket connection ───────────────────────────────────────────────

class SignalRConn:
    """
    Minimal SignalR-over-WebSocket client.

    Handles:
      - Handshake (JSON protocol v1)
      - Invocation sending
      - Ping keepalive (type 6) every 15s
      - Auto-reconnect with back-off on disconnect
      - Token refresh callback
    """

    PING_INTERVAL = 15   # seconds between pings
    RECONNECT_DELAYS = [1, 2, 5, 10, 30, 60]  # seconds
    FAST_FAIL_SECS   = 10   # connection that drops within this many seconds is a "fast fail"
    FAST_FAIL_MAX    = 5    # exit process after this many consecutive fast fails (forces fresh loginKey)

    def __init__(self, hub_url: str, token_factory, on_message, on_connected,
                 name: str = "hub"):
        self.hub_url       = hub_url
        self.token_factory = token_factory
        self.on_message    = on_message    # callable(dict)
        self.on_connected  = on_connected  # callable(ws) — called after handshake
        self.name          = name
        self._ws: websocket.WebSocket | None = None
        self._running      = False
        self._send_lock    = threading.Lock()
        self._invocation_id = 0

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run_loop, daemon=True, name=self.name)
        t.start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def send(self, target: str, arguments: list):
        """Send a SignalR invocation."""
        self._invocation_id += 1
        msg = json.dumps({
            "type": 1,
            "invocationId": str(self._invocation_id),
            "target": target,
            "arguments": arguments,
        }) + RS
        with self._send_lock:
            if self._ws:
                try:
                    self._ws.send(msg)
                except Exception as e:
                    log.warning(f"{self.name}: send error: {e}")

    def _url(self) -> str:
        token = self.token_factory()
        return f"{self.hub_url}?access_token={token}"

    def _run_loop(self):
        attempt    = 0
        fast_fails = 0
        while self._running:
            connect_ts = time.monotonic()
            try:
                self._connect()
                attempt = 0   # reset back-off on clean run
                fast_fails = 0
            except Exception as e:
                if not self._running:
                    break
                # Pause reconnects during the overnight no-trading window.
                # Active session is 09:00–17:00 ET = 13:00–21:00 UTC.
                # Outside that range (17:00 ET onward) don't reconnect until 08:30 ET
                # (12:30 UTC) the next morning.
                now_utc = datetime.now(timezone.utc)
                h, m = now_utc.hour, now_utc.minute
                in_session = (13, 0) <= (h, m) < (21, 0)
                if not in_session:
                    # Resume at 12:30 UTC (= 08:30 ET) today or tomorrow
                    resume = now_utc.replace(hour=12, minute=30, second=0, microsecond=0)
                    if (h, m) >= (12, 30):
                        resume += timedelta(days=1)
                    wait = max(60, int((resume - now_utc).total_seconds()) + 60)
                    log.info(
                        f"{self.name}: overnight no-trading window — "
                        f"pausing {wait//60}min until 12:30 UTC (08:30 ET)"
                    )
                    time.sleep(wait)
                    attempt = 0
                    fast_fails = 0
                    continue
                delay = self.RECONNECT_DELAYS[min(attempt, len(self.RECONNECT_DELAYS) - 1)]
                log.warning(f"{self.name}: disconnected ({e}) — reconnecting in {delay}s")
                attempt += 1
                time.sleep(delay)
            else:
                # _connect() returned cleanly — check if it was a fast fail
                elapsed = time.monotonic() - connect_ts
                if elapsed < self.FAST_FAIL_SECS:
                    fast_fails += 1
                    log.warning(
                        f"{self.name}: fast disconnect after {elapsed:.1f}s "
                        f"({fast_fails}/{self.FAST_FAIL_MAX}) — likely session invalidated"
                    )
                    if fast_fails >= self.FAST_FAIL_MAX:
                        log.error(
                            f"{self.name}: {self.FAST_FAIL_MAX} consecutive fast disconnects — "
                            f"exiting so wrapper can restart with fresh token"
                        )
                        import sys; sys.exit(1)
                    time.sleep(self.RECONNECT_DELAYS[min(fast_fails, len(self.RECONNECT_DELAYS) - 1)])
                else:
                    fast_fails = 0
                    attempt = 0

    def _connect(self):
        ws = websocket.create_connection(
            self._url(),
            sslopt={"context": _ssl_ctx},
            timeout=30,
        )
        self._ws = ws
        log.info(f"{self.name}: connected")

        # Send SignalR handshake
        ws.send('{"protocol":"json","version":1}' + RS)

        # Read handshake response (may be batched with first real message)
        raw = ws.recv()
        frames = [f for f in raw.split(RS) if f.strip()]
        handshake = json.loads(frames[0])
        if handshake.get("error"):
            raise RuntimeError(f"Handshake error: {handshake['error']}")
        log.info(f"{self.name}: handshake OK")

        # Process any extra frames bundled with the handshake response
        for frame in frames[1:]:
            self._dispatch(json.loads(frame))

        # Notify caller so it can subscribe
        self.on_connected(self)

        # Start ping thread
        ping_stop = threading.Event()
        ping_thread = threading.Thread(
            target=self._ping_loop, args=(ws, ping_stop), daemon=True
        )
        ping_thread.start()

        try:
            while self._running:
                raw = ws.recv()
                for frame in raw.split(RS):
                    frame = frame.strip()
                    if not frame:
                        continue
                    try:
                        msg = json.loads(frame)
                    except json.JSONDecodeError:
                        log.debug(f"{self.name}: non-JSON frame: {frame!r}")
                        continue
                    if not isinstance(msg, dict):
                        log.debug(f"{self.name}: unexpected frame type {type(msg).__name__}: {frame!r}")
                        continue
                    self._dispatch(msg)
        finally:
            ping_stop.set()
            try:
                ws.close()
            except Exception:
                pass
            self._ws = None
            log.info(f"{self.name}: disconnected")

    def _dispatch(self, msg: dict):
        msg_type = msg.get("type")
        if msg_type == 1:   # Invocation (server→client push)
            self.on_message(msg)
        elif msg_type == 6: # Ping — no response needed (server pings us)
            pass
        elif msg_type == 7: # Close
            error = msg.get("error", "")
            log.warning(f"{self.name}: server close: {error!r}")
        # types 2-5 are streaming/completion, not used here

    def _ping_loop(self, ws: websocket.WebSocket, stop: threading.Event):
        """Send a SignalR ping every PING_INTERVAL seconds to keep the connection alive."""
        while not stop.wait(self.PING_INTERVAL):
            try:
                with self._send_lock:
                    ws.send('{"type":6}' + RS)
            except Exception:
                break


# ── Warmup: backfill historical bars ─────────────────────────────────────────

def backfill(client: TopstepClient, states: dict[str, SymbolState],
             bars_conn: sqlite3.Connection, demo: bool):
    now = datetime.now(timezone.utc)
    log.info("Backfilling historical bars …")
    for sym, state in states.items():
        for minutes, count in [(1, 600), (5, 200)]:
            start = now - timedelta(minutes=minutes * count + 60)
            try:
                raw = client.get_bars(
                    contract_id=state.contract_id,
                    start=start, end=now,
                    unit=TopstepClient.MINUTE,
                    unit_number=minutes,
                    limit=count,
                )
            except Exception as e:
                log.warning(f"Backfill {sym} {minutes}m failed: {e}")
                continue

            rows = []
            for b in reversed(raw):
                ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                rows.append((sym, minutes, ts.isoformat(),
                             b["o"], b["h"], b["l"], b["c"], b["v"]))

            if not demo and rows:
                with _db_lock:
                    bars_conn.executemany("""
                        INSERT OR IGNORE INTO bars
                            (symbol, minutes, ts, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, rows)
                    bars_conn.commit()
            log.info(f"  {sym} {minutes}m: {len(rows)} bars backfilled")
            time.sleep(5)

        # Backfill 5s bars (~30 minutes of history is plenty for PL_mom warm-up)
        try:
            start_5s = now - timedelta(minutes=35)
            raw_5s = client.get_bars(
                contract_id=state.contract_id,
                start=start_5s, end=now,
                unit=TopstepClient.SECOND,
                unit_number=5,
                limit=360,
            )
            rows_5s = []
            for b in reversed(raw_5s):
                ts = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                rows_5s.append((sym, ts.isoformat(),
                                b["o"], b["h"], b["l"], b["c"], b["v"]))
            if not demo and rows_5s:
                with _db_lock:
                    bars_conn.executemany("""
                        INSERT OR IGNORE INTO bars5s
                            (symbol, ts, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, rows_5s)
                    bars_conn.commit()
            log.info(f"  {sym} 5s: {len(rows_5s)} bars backfilled")
        except Exception as e:
            log.warning(f"Backfill {sym} 5s failed: {e}")
        time.sleep(3)

    log.info("Backfill complete")


# ── Partial-bar flush loop ────────────────────────────────────────────────────

def flush_partial_bars_loop(states: dict[str, SymbolState],
                            bars_conn: sqlite3.Connection, demo: bool):
    while True:
        time.sleep(30)
        for sym, state in states.items():
            for minutes, bar in state.bars.items():
                if bar.tick_count > 0:
                    write_bar(bars_conn, bar, demo)
            for seconds, bar in state.bars_sec.items():
                if bar.tick_count > 0:
                    write_bar_5s(bars_conn, bar, demo)


# ── Silent-feed watchdog ──────────────────────────────────────────────────────

TICK_TIMEOUT_SECS = 300   # force reconnect if no ticks for 5 min during market hours

def silent_feed_watchdog(market_hub: "SignalRConn", last_tick: list):
    """Force-closes the market-hub WebSocket if no ticks arrive during market hours.
    last_tick is a one-element list [datetime | None] updated by on_trade."""
    while True:
        time.sleep(60)
        now_utc = datetime.now(timezone.utc)
        # Only check during active session: 13:00–21:00 UTC (09:00–17:00 ET)
        h, m = now_utc.hour, now_utc.minute
        if not ((13, 0) <= (h, m) < (21, 0)):
            continue
        t = last_tick[0]
        if t is None:
            continue
        age = (now_utc - t).total_seconds()
        if age > TICK_TIMEOUT_SECS:
            log.warning(
                f"market-hub: no ticks for {int(age)}s — forcing reconnect"
            )
            last_tick[0] = now_utc   # reset so we don't fire again immediately
            if market_hub._ws:
                try:
                    market_hub._ws.close()
                except Exception:
                    pass


# ── Main ──────────────────────────────────────────────────────────────────────

def _wait_for_session():
    """Block until 08:30–17:00 ET on a Mon–Fri weekday."""
    while True:
        now_et = datetime.now(_ET)
        wd  = now_et.weekday()          # 0=Mon … 4=Fri, 5=Sat, 6=Sun
        hm  = (now_et.hour, now_et.minute)
        if wd < 5 and (8, 30) <= hm < (17, 0):
            return
        # Next 08:30 ET on a weekday
        target = now_et.replace(hour=8, minute=30, second=0, microsecond=0)
        if hm >= (8, 30):
            target += timedelta(days=1)
        while target.weekday() >= 5:
            target += timedelta(days=1)
        wait = (target - now_et).total_seconds()
        # Sleep in short chunks when close to start; long chunks otherwise
        chunk = 30 if wait <= 60 else 300 if wait <= 600 else 1800
        log.info(f"Outside session — sleeping {wait/3600:.1f}h until "
                 f"{target.strftime('%a %H:%M ET')}")
        time.sleep(min(wait, chunk))


def _start_session_exit_watcher():
    """Background thread: exit cleanly at 17:00 ET so launchd restarts us fresh tomorrow."""
    def _watch():
        while True:
            time.sleep(30)
            if (datetime.now(_ET).hour, datetime.now(_ET).minute) >= (17, 0):
                log.info("17:00 ET — session ended, exiting cleanly")
                os._exit(0)   # hard kill — sys.exit() gets caught by websocket threads
    threading.Thread(target=_watch, daemon=True, name="session-exit").start()


def run(demo: bool = False):
    _wait_for_session()
    _start_session_exit_watcher()

    # Clear stale shared token so other processes wait for our fresh login
    from topstep_client import _SHARED_TOKEN_FILE
    try:
        _SHARED_TOKEN_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    client = TopstepClient()
    client.login()
    log.info("Logged in")

    # Resolve contracts
    states: dict[str, SymbolState] = {}
    for sym, search_term in INSTRUMENTS.items():
        contracts = client.search_contracts(search_term)
        if not contracts:
            log.error(f"No contract found for {sym} — skipping")
            continue
        cid = contracts[0]["id"]
        log.info(f"  {sym}: contract_id={cid}")
        states[sym] = SymbolState(symbol=sym, contract_id=cid)

    if not states:
        raise RuntimeError("No instruments resolved — aborting")

    # Open databases
    bars_conn  = init_bars_db(BARS_DB)   if not demo else None
    _bars_conn = bars_conn or sqlite3.connect(":memory:")

    # Backfill
    backfill(client, states, _bars_conn, demo)

    # Token factory
    def token_factory():
        return client.token

    # Build a lookup: contract_id → (sym, SymbolState)
    by_cid: dict[str, tuple[str, SymbolState]] = {
        st.contract_id: (sym, st) for sym, st in states.items()
    }

    # Market hub — handles GatewayTrade, GatewayDepth, GatewayQuote
    def on_market_message(msg: dict):
        target = msg.get("target", "")
        args   = msg.get("arguments", [])
        if not args:
            return
        contract_id = str(args[0])
        entry = by_cid.get(contract_id)
        if not entry:
            return
        sym, state = entry

        if target == "GatewayTrade" and len(args) >= 2:
            trades = args[1]
            if isinstance(trades, dict):
                trades = [trades]
            for trade in trades:
                if isinstance(trade, dict):
                    on_trade(sym, state, _bars_conn, trade, demo)

        elif target == "GatewayDepth" and len(args) >= 2:
            updates = args[1]
            if isinstance(updates, list):
                state.dom.apply_depth(updates)

        elif target == "GatewayQuote" and len(args) >= 2:
            q = args[1]
            if isinstance(q, dict):
                state.dom.apply_quote(
                    q.get("lastPrice"), q.get("bestBid"), q.get("bestAsk")
                )

    def on_market_connected(conn: SignalRConn):
        log.info("Market hub: subscribing to contracts")
        for sym, state in states.items():
            cid = state.contract_id
            conn.send("SubscribeContractTrades",      [cid])
            conn.send("SubscribeContractMarketDepth", [cid])
            conn.send("SubscribeContractQuotes",      [cid])
            log.info(f"  Subscribed {sym} ({cid})")

    market_hub = SignalRConn(
        hub_url=MARKET_HUB_URL,
        token_factory=token_factory,
        on_message=on_market_message,
        on_connected=on_market_connected,
        name="market-hub",
    )

    market_hub.start()
    log.info(f"Bar collector running — instruments={list(states.keys())}  demo={demo}")

    # DOM writer thread — writes live DOM state to dom.db every 200ms
    if not demo:
        dom_conn = init_dom_db(DOM_DB)
        books_by_sym = {sym: st.dom for sym, st in states.items()}
        threading.Thread(
            target=dom_writer_loop,
            args=(books_by_sym, dom_conn),
            daemon=True,
            name="dom-writer",
        ).start()
        log.info("DOM writer started → data/dom.db")

    # Partial-bar flush thread
    if bars_conn:
        threading.Thread(
            target=flush_partial_bars_loop,
            args=(states, bars_conn, demo),
            daemon=True,
        ).start()


    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Shutting down …")
        market_hub.stop()
        if bars_conn:
            bars_conn.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    handlers = [logging.FileHandler("logs/bar_collector.log")]
    if sys.stdout.isatty():
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

    parser = argparse.ArgumentParser(description="Real-time bar collector via TopstepX WebSocket")
    parser.add_argument("--demo", action="store_true",
                        help="Print ticks/bars to console without writing to DB")
    args = parser.parse_args()

    run(demo=args.demo)
