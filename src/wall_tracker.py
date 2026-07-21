"""
WallTracker — real-time DOM support/resistance wall detector.

Tracks large resting orders (walls) in the live order book and counts
how many times price has tested each level. Designed around the observation
that a wall tested 2-3× tends to break on the next approach, generating
momentum in the breakout direction.

Terminology
-----------
  wall        : a resting order whose size ≥ WALL_MULT × rolling median
  test        : price enters the touch zone (<= TOUCH_PTS of the wall price),
                then retreats at least BOUNCE_PTS away from the wall
  breakout    : price crosses to the other side of the wall by >= BREAK_PTS
  wall pulled : wall size drops below PULL_THRESH of its peak without a
                breakout (iceberg refill is not counted as "pulled")

Events logged to logs/wall_events.csv
--------------------------------------
  wall_found  : new significant wall detected
  wall_grown  : wall increased by >= 50% (re-reinforced level)
  test        : price touched and bounced; test_count updated
  breakout    : price crossed the wall (include test_count at breakout)
  wall_pulled : wall disappeared without a breakout
"""

import csv
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np

# ── Parameters ────────────────────────────────────────────────────────────────

WALL_MULT      = 3.0    # size >= WALL_MULT × median to qualify as a wall
WALL_MIN_SIZE  = 50     # absolute minimum (ignores thin-book noise)
TOUCH_PTS      = 1.0    # within this many pts of wall price = "in touch zone"
BOUNCE_PTS     = 1.0    # must retreat this far to confirm a test (not just noise)
BREAK_PTS      = 0.5    # must cross this far through wall price to call breakout
PULL_THRESH    = 0.35   # wall size drops below this fraction of peak → wall pulled
MAX_WALL_AGE_S = 4 * 3600   # forget walls older than 4 hours
MIN_WALL_GAP   = 1.0    # two walls on same side must be >= this pts apart

LOG_PATH = Path("logs/wall_events.csv")
LOG_FIELDS = [
    "ts", "symbol", "side", "wall_price", "wall_size", "peak_size",
    "event", "test_count", "price_at_event",
]

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WallLevel:
    side:          Literal["bid", "ask"]
    price:         float
    size:          float           # current size
    peak_size:     float           # maximum size seen since detection
    first_seen:    datetime
    last_seen:     datetime
    test_count:    int    = 0
    in_touch_zone: bool   = False  # is price currently in the touch zone?
    touch_entry_price: float | None = None  # price when touch zone was entered
    broken:        bool   = False
    pulled:        bool   = False


@dataclass
class WallEvent:
    ts:          datetime
    symbol:      str
    side:        str
    wall_price:  float
    wall_size:   float
    peak_size:   float
    event:       str    # 'wall_found' | 'wall_grown' | 'test' | 'breakout' | 'wall_pulled'
    test_count:  int
    price:       float | None = None


# ── WallTracker ───────────────────────────────────────────────────────────────

class WallTracker:
    """
    Feed this with DOM snapshots via update(); it returns a list of WallEvents
    for any notable state changes.

    Intended update frequency: every 1-5 seconds (from the display/eval loop).
    """

    def __init__(self, symbol: str):
        self.symbol  = symbol
        self._walls: list[WallLevel] = []
        self._lock   = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self,
               bids: dict,    # price → size (raw DOM book bid side)
               asks: dict,    # price → size (raw DOM book ask side)
               best_bid: float | None,
               best_ask: float | None,
               ts: datetime | None = None) -> list[WallEvent]:
        """
        Consume a snapshot of the current DOM book.  Returns new WallEvents.
        Thread-safe — can be called from any thread.
        """
        if ts is None:
            ts = datetime.now(timezone.utc)
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else None

        with self._lock:
            events = []
            events += self._detect_new_walls(bids, asks, best_bid, best_ask, ts)
            events += self._update_existing_walls(bids, asks, best_bid, best_ask, mid, ts)
            self._expire_old_walls(ts)
            return events

    def active_walls(self) -> list[WallLevel]:
        """Return current active (not broken/pulled) walls, sorted price desc."""
        with self._lock:
            return sorted(
                [w for w in self._walls if not w.broken and not w.pulled],
                key=lambda w: w.price, reverse=True
            )

    def wall_at(self, price: float, side: str) -> WallLevel | None:
        """Return active wall at this exact price level, or None."""
        with self._lock:
            for w in self._walls:
                if not w.broken and not w.pulled and w.side == side and w.price == price:
                    return w
            return None

    def test_count_near(self, price: float, side: str, pts: float = 2.0) -> int:
        """Max test count of any active wall within pts of price on given side."""
        with self._lock:
            best = 0
            for w in self._walls:
                if not w.broken and not w.pulled and w.side == side:
                    if abs(w.price - price) <= pts:
                        best = max(best, w.test_count)
            return best

    # ── Internal ─────────────────────────────────────────────────────────────

    def _median_size(self, book: dict) -> float:
        sizes = [s for s in book.values() if s > 0]
        return float(np.median(sizes)) if sizes else 1.0

    def _is_significant(self, size: float, median: float) -> bool:
        return size >= WALL_MULT * median and size >= WALL_MIN_SIZE

    def _detect_new_walls(self, bids, asks, best_bid, best_ask, ts) -> list[WallEvent]:
        events = []
        bid_med = self._median_size(bids)
        ask_med = self._median_size(asks)

        for side, book, med, ref_price in [
            ("bid", bids, bid_med, best_bid),
            ("ask", asks, ask_med, best_ask),
        ]:
            if ref_price is None:
                continue
            for price, size in book.items():
                if not self._is_significant(size, med):
                    continue
                # Only track walls behind the current price
                # (bid walls below best_bid, ask walls above best_ask)
                if side == "bid" and price >= best_ask:
                    continue
                if side == "ask" and price <= best_bid:
                    continue

                # Check if we already track this level
                existing = self._find_wall(price, side)
                if existing:
                    # Update size; check for re-reinforcement
                    if size > existing.peak_size * 1.5:
                        existing.peak_size = size
                        events.append(WallEvent(
                            ts=ts, symbol=self.symbol, side=side,
                            wall_price=price, wall_size=size,
                            peak_size=existing.peak_size,
                            event="wall_grown", test_count=existing.test_count,
                            price=ref_price,
                        ))
                    existing.size = size
                    existing.last_seen = ts
                    continue

                # New wall — check minimum gap from existing walls
                if self._too_close(price, side):
                    continue

                wall = WallLevel(
                    side=side, price=price, size=size,
                    peak_size=size, first_seen=ts, last_seen=ts,
                )
                self._walls.append(wall)
                events.append(WallEvent(
                    ts=ts, symbol=self.symbol, side=side,
                    wall_price=price, wall_size=size, peak_size=size,
                    event="wall_found", test_count=0, price=ref_price,
                ))

        return events

    def _update_existing_walls(self, bids, asks, best_bid, best_ask, mid, ts) -> list[WallEvent]:
        events = []
        if mid is None:
            return events

        for w in self._walls:
            if w.broken or w.pulled:
                continue

            book = bids if w.side == "bid" else asks
            current_size = book.get(w.price, 0.0)

            # Update size
            w.size = current_size
            if current_size > w.peak_size:
                w.peak_size = current_size

            # ── Breakout detection ─────────────────────────────────────────
            if w.side == "bid":
                # bid wall broken when price drops BREAK_PTS below wall price
                if best_bid is not None and best_bid < w.price - BREAK_PTS:
                    w.broken = True
                    events.append(WallEvent(
                        ts=ts, symbol=self.symbol, side=w.side,
                        wall_price=w.price, wall_size=w.size, peak_size=w.peak_size,
                        event="breakout", test_count=w.test_count, price=mid,
                    ))
                    continue
            else:  # ask
                # ask wall broken when price rises BREAK_PTS above wall price
                if best_ask is not None and best_ask > w.price + BREAK_PTS:
                    w.broken = True
                    events.append(WallEvent(
                        ts=ts, symbol=self.symbol, side=w.side,
                        wall_price=w.price, wall_size=w.size, peak_size=w.peak_size,
                        event="breakout", test_count=w.test_count, price=mid,
                    ))
                    continue

            # ── Wall pulled (disappeared without breakout) ─────────────────
            if w.peak_size > 0 and current_size < w.peak_size * PULL_THRESH and current_size < WALL_MIN_SIZE:
                w.pulled = True
                events.append(WallEvent(
                    ts=ts, symbol=self.symbol, side=w.side,
                    wall_price=w.price, wall_size=current_size, peak_size=w.peak_size,
                    event="wall_pulled", test_count=w.test_count, price=mid,
                ))
                continue

            # ── Touch / test detection ────────────────────────────────────
            if w.side == "bid":
                dist = mid - w.price          # positive = price above wall
            else:
                dist = w.price - mid          # positive = price below wall (above market)

            in_zone = dist <= TOUCH_PTS and dist >= -BREAK_PTS

            if in_zone and not w.in_touch_zone:
                # Entering touch zone
                w.in_touch_zone = True
                w.touch_entry_price = mid

            elif not in_zone and w.in_touch_zone:
                # Exiting touch zone — was this a real test?
                if w.touch_entry_price is not None:
                    # Confirm: price is at least BOUNCE_PTS away from the wall
                    if dist >= BOUNCE_PTS:
                        w.test_count += 1
                        w.in_touch_zone = False
                        w.touch_entry_price = None
                        events.append(WallEvent(
                            ts=ts, symbol=self.symbol, side=w.side,
                            wall_price=w.price, wall_size=w.size, peak_size=w.peak_size,
                            event="test", test_count=w.test_count, price=mid,
                        ))
                        continue
                w.in_touch_zone = False
                w.touch_entry_price = None

            w.last_seen = ts

        return events

    def _expire_old_walls(self, ts: datetime):
        cutoff = (ts - __import__("datetime").timedelta(seconds=MAX_WALL_AGE_S)).timestamp()
        self._walls = [
            w for w in self._walls
            if w.last_seen.timestamp() > cutoff or w.in_touch_zone
        ]

    def _find_wall(self, price: float, side: str) -> WallLevel | None:
        for w in self._walls:
            if w.side == side and w.price == price and not w.broken and not w.pulled:
                return w
        return None

    def _too_close(self, price: float, side: str) -> bool:
        for w in self._walls:
            if w.side == side and not w.broken and not w.pulled:
                if abs(w.price - price) < MIN_WALL_GAP:
                    return True
        return False


# ── CSV logging ───────────────────────────────────────────────────────────────

_log_lock   = threading.Lock()
_log_header = False


def ensure_wall_log():
    global _log_header
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    _log_header = True


def log_wall_events(events: list[WallEvent]):
    if not events:
        return
    with _log_lock:
        with open(LOG_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            for e in events:
                w.writerow({
                    "ts":             e.ts.isoformat(),
                    "symbol":         e.symbol,
                    "side":           e.side,
                    "wall_price":     e.wall_price,
                    "wall_size":      f"{e.wall_size:.0f}",
                    "peak_size":      f"{e.peak_size:.0f}",
                    "event":          e.event,
                    "test_count":     e.test_count,
                    "price_at_event": f"{e.price:.2f}" if e.price else "",
                })
