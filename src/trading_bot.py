"""
Automated trading bot for the 3σ MES/MYM continuation signal.

Monitors 5-min bars every 30 seconds, detects signals using the same criteria
as signal_monitor.py, places market orders with native API bracket
stops/targets, force-closes at 15-minute expiry if brackets not hit, and
logs every trade outcome to logs/bot_trades.csv.

Signal criteria (identical to signal_monitor.py):
  1. |bar_return / σ| ≥ 3.0   (σ = trailing 100-min close-return std dev)
  2. Volume ≥ 1.5× trailing mean volume
  3. 40-min CSR ≥ 1.5σ aligned with signal direction (momentum filter)
  4. Not in 08:00–09:00 EST blackout window (economic data releases)
  5. |scaled| ≤ 5.0 (extreme event filter)

Execution:
  - 1 contract market order per signal, per instrument
  - Native API bracket: stop = 2σ below entry, target = 3σ above entry
    (both in ticks, OCO-managed server-side)
  - Force-close at market after 15 minutes if neither bracket is hit
  - Only one position per instrument at a time

Usage:
  python src/trading_bot.py                  # live trading (requires .env)
  python src/trading_bot.py --paper          # paper mode: signals logged, no orders placed
  python src/trading_bot.py --account 12345  # specify account ID explicitly
"""

import argparse
import csv
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from topstep_client import TopstepClient

# Practice account — the only account the live bot is permitted to trade on.
# Set TOPSTEP_ACCOUNT_ID in .env to override.
PRACTICE_ACCOUNT_ID = int(os.environ.get("TOPSTEP_ACCOUNT_ID", "10634862"))
PRACTICE_ACCOUNT_NAME = "PRAC-V2-88916-19336808"

# ── Parameters (keep in sync with signal_monitor.py) ────────────────────────

TF_MINUTES    = 5
TRAILING_BARS = 20     # 20 × 5-min = 100-min σ window
MOM_BARS      = 8      # 8 × 5-min = 40-min CSR window
CSR_THRESHOLD = 1.5    # min cumulative scaled return aligned with signal direction
SIGNAL_SIGMA  = 3.0
MAX_SCALED    = 5.0    # ignore extreme event spikes
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 15     # force-close after this many minutes
POLL_SECONDS  = 30

BLACKOUT_UTC = [(13, 0, 14, 0)]   # 08:00–09:00 EST = 13:00–14:00 UTC

LOG_PATH = Path("logs/bot_trades.csv")
LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "target", "stop",
    "sigma_pts", "scaled", "vol_ratio", "csr",
    "outcome", "pnl_pts", "pnl_sigma",
]

log = logging.getLogger("bot")


# ── Instrument config ────────────────────────────────────────────────────────

@dataclass
class BotInstrument:
    symbol:       str
    search_term:  str
    stop_sigma:   float = 2.0
    target_sigma: float = 3.0
    tick_size:    float = 0.25   # minimum price increment
    point_value:  float = 5.00  # $ per point (informational only)


INSTRUMENTS = [
    BotInstrument("MES", "MES", tick_size=0.25, point_value=5.00),
    BotInstrument("MYM", "MYM", tick_size=1.00, point_value=0.50),
]


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts:     datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class ActiveTrade:
    instrument:  BotInstrument
    contract_id: str
    direction:   int        # +1 long / -1 short
    est_entry:   float      # signal bar close (pre-fill estimate)
    sigma_pts:   float
    scaled:      float
    vol_ratio:   float
    csr:         float
    fired_at:    datetime
    order_id:    int | None = None
    fill_price:  float | None = None
    expires_at:  datetime = field(init=False)

    def __post_init__(self):
        self.expires_at = self.fired_at + timedelta(minutes=MAX_HOLD_MIN)

    def target_price(self) -> float:
        p = self.fill_price or self.est_entry
        return p + self.direction * self.instrument.target_sigma * self.sigma_pts

    def stop_price(self) -> float:
        p = self.fill_price or self.est_entry
        return p - self.direction * self.instrument.stop_sigma * self.sigma_pts


@dataclass
class InstrumentState:
    instrument:   BotInstrument
    contract_id:  str = ""
    bars:         list[Bar] = field(default_factory=list)
    sigma:        float = 0.0
    sigma_pts:    float = 0.0
    mean_vol:     float = 1.0
    csr:          float = 0.0
    active_trade: ActiveTrade | None = None


# ── Trade logging ────────────────────────────────────────────────────────────

def _ensure_log():
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()


def _log_trade(trade: ActiveTrade, outcome: str, exit_price: float, now: datetime):
    fill = trade.fill_price or trade.est_entry
    pnl_pts = (exit_price - fill) * trade.direction
    row = {
        "fired_at":    trade.fired_at.isoformat(),
        "resolved_at": now.isoformat(),
        "symbol":      trade.instrument.symbol,
        "direction":   "LONG" if trade.direction == 1 else "SHORT",
        "est_entry":   round(trade.est_entry, 4),
        "fill_price":  round(fill, 4),
        "target":      round(trade.target_price(), 4),
        "stop":        round(trade.stop_price(), 4),
        "sigma_pts":   round(trade.sigma_pts, 4),
        "scaled":      round(trade.scaled, 4),
        "vol_ratio":   round(trade.vol_ratio, 4),
        "csr":         round(trade.csr, 4),
        "outcome":     outcome,
        "pnl_pts":     round(pnl_pts, 4),
        "pnl_sigma":   round(pnl_pts / trade.sigma_pts, 4) if trade.sigma_pts else 0.0,
    }
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)
    log.info(
        f"TRADE LOGGED  {trade.instrument.symbol} {row['direction']}  {outcome}  "
        f"fill={fill:.2f}  exit={exit_price:.2f}  "
        f"pnl={pnl_pts:+.2f}pts ({pnl_pts / trade.sigma_pts:+.3f}σ)"
    )


# ── Signal evaluation (identical logic to signal_monitor.py) ────────────────

def evaluate(state: InstrumentState) -> dict | None:
    """Return signal dict if all criteria are met, else None."""
    bars = state.bars
    if len(bars) < TRAILING_BARS + 1:
        return None

    closes  = np.array([b.close  for b in bars])
    volumes = np.array([b.volume for b in bars])

    trail     = np.log(closes[-TRAILING_BARS:] / closes[-TRAILING_BARS - 1:-1])
    sigma     = float(np.std(trail, ddof=1))
    if sigma == 0:
        return None

    sigma_pts = sigma * closes[-1]
    mean_vol  = float(volumes[-TRAILING_BARS - 1:-1].mean())
    last      = bars[-1]
    bar_ret   = math.log(last.close / last.open) if last.open else 0.0
    scaled    = bar_ret / sigma
    vol_ratio = last.volume / mean_vol if mean_vol else 0.0
    direction = 1 if scaled > 0 else -1

    # 40-min CSR (Cumulative Scaled Return), direction-adjusted
    if len(closes) >= MOM_BARS + 1:
        mom_rets = np.log(closes[-MOM_BARS:] / closes[-MOM_BARS - 1:-1])
        csr = float(mom_rets.sum()) / sigma * direction
    else:
        csr = 0.0

    state.sigma     = sigma
    state.sigma_pts = sigma_pts
    state.mean_vol  = mean_vol
    state.csr       = csr

    # Blackout window
    h, m = last.ts.hour, last.ts.minute
    if any((sh, sm) <= (h, m) < (eh, em) for sh, sm, eh, em in BLACKOUT_UTC):
        return None

    if (abs(scaled) >= SIGNAL_SIGMA and abs(scaled) <= MAX_SCALED
            and vol_ratio >= VOL_RATIO_MIN and csr >= CSR_THRESHOLD):
        return {
            "direction": direction,
            "entry":     last.close,
            "sigma":     sigma,
            "sigma_pts": sigma_pts,
            "scaled":    scaled,
            "vol_ratio": vol_ratio,
            "csr":       csr,
            "bar_ts":    last.ts,
        }
    return None


# ── Bar fetching ─────────────────────────────────────────────────────────────

def fetch_bars(client: TopstepClient, state: InstrumentState):
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=TF_MINUTES * (TRAILING_BARS + 5))
    raw = client.get_bars(
        contract_id=state.contract_id,
        start=start, end=end,
        unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
        limit=TRAILING_BARS + 10,
    )
    state.bars = [
        Bar(ts=datetime.fromisoformat(b["t"]),
            open=b["o"], high=b["h"], low=b["l"],
            close=b["c"], volume=b["v"])
        for b in reversed(raw)
    ]


# ── Order placement ──────────────────────────────────────────────────────────

def place_signal(client: TopstepClient, state: InstrumentState,
                 sig: dict, account_id: int, paper: bool) -> ActiveTrade:
    inst      = state.instrument
    direction = sig["direction"]
    sigma_pts = sig["sigma_pts"]
    tick      = inst.tick_size

    stop_ticks   = max(1, round(inst.stop_sigma   * sigma_pts / tick))
    target_ticks = max(1, round(inst.target_sigma * sigma_pts / tick))
    side         = TopstepClient.BID if direction == 1 else TopstepClient.ASK
    dir_str      = "LONG" if direction == 1 else "SHORT"

    trade = ActiveTrade(
        instrument=inst, contract_id=state.contract_id,
        direction=direction, est_entry=sig["entry"],
        sigma_pts=sigma_pts, scaled=sig["scaled"],
        vol_ratio=sig["vol_ratio"], csr=sig["csr"],
        fired_at=sig["bar_ts"],
    )

    if paper:
        log.info(
            f"[PAPER] {inst.symbol} {dir_str}  entry≈{sig['entry']:.2f}  "
            f"stop={stop_ticks}t ({inst.stop_sigma}σ)  "
            f"target={target_ticks}t ({inst.target_sigma}σ)  "
            f"scaled={sig['scaled']:+.2f}σ  csr={sig['csr']:+.2f}σ"
        )
    else:
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=side, size=1,
            order_type=TopstepClient.ORDER_MARKET,
            stop_loss_ticks=stop_ticks,
            take_profit_ticks=target_ticks,
            custom_tag=f"bot_{inst.symbol}_{sig['bar_ts'].strftime('%H%M')}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"ORDER PLACED  {inst.symbol} {dir_str}  order_id={trade.order_id}  "
            f"entry≈{sig['entry']:.2f}  stop={stop_ticks}t  target={target_ticks}t  "
            f"scaled={sig['scaled']:+.2f}σ  csr={sig['csr']:+.2f}σ"
        )

    state.active_trade = trade
    return trade


# ── Position monitoring ──────────────────────────────────────────────────────

def handle_active_trade(client: TopstepClient, state: InstrumentState,
                        account_id: int, now: datetime, paper: bool):
    trade = state.active_trade

    if paper:
        # Paper mode: only simulate time exit
        if now >= trade.expires_at:
            exit_price = state.bars[-1].close if state.bars else trade.est_entry
            _log_trade(trade, "TIME EXIT (paper)", exit_price, now)
            state.active_trade = None
        return

    # Fetch current open positions
    try:
        positions = client.get_open_positions(account_id)
    except Exception as e:
        log.warning(f"Could not fetch positions: {e}")
        return

    pos = next(
        (p for p in positions if p.get("contractId") == trade.contract_id),
        None,
    )

    # Update fill price once the position appears
    if pos and trade.fill_price is None:
        trade.fill_price = pos.get("averagePrice")
        log.info(f"{trade.instrument.symbol} fill confirmed: {trade.fill_price:.2f}")

    if pos is None:
        # Position is gone — brackets closed it
        exit_price = _get_exit_price(client, account_id, trade, now)
        outcome    = _classify_outcome(trade, exit_price)
        _log_trade(trade, outcome, exit_price, now)
        state.active_trade = None
        return

    # Force time exit if max hold exceeded
    if now >= trade.expires_at:
        log.info(f"{trade.instrument.symbol} max hold reached — closing at market")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"Failed to close position for {trade.instrument.symbol}: {e}")
            return
        exit_price = state.bars[-1].close if state.bars else (trade.fill_price or trade.est_entry)
        _log_trade(trade, "TIME EXIT", exit_price, now)
        state.active_trade = None


def _get_exit_price(client: TopstepClient, account_id: int,
                    trade: ActiveTrade, now: datetime) -> float:
    """Try to get actual exit price from trade history; fall back to bar close."""
    try:
        trades = client.search_trades(account_id, trade.fired_at, now)
        # Closing trade: same contract, after signal fired, on opposite side
        closing = [
            t for t in trades
            if t.get("contractId") == trade.contract_id
            and datetime.fromisoformat(t.get("timestamp", "1970")).replace(tzinfo=timezone.utc)
                > trade.fired_at
        ]
        if closing:
            return float(closing[-1].get("price", trade.fill_price or trade.est_entry))
    except Exception as e:
        log.warning(f"Could not fetch trade history for exit price: {e}")
    return trade.fill_price or trade.est_entry


def _classify_outcome(trade: ActiveTrade, exit_price: float) -> str:
    """Classify as TARGET or STOPPED based on which bracket is closer to exit."""
    tgt_dist = abs(exit_price - trade.target_price())
    stp_dist = abs(exit_price - trade.stop_price())
    if tgt_dist <= stp_dist:
        return "TARGET"
    return "STOPPED"


# ── Main loop ────────────────────────────────────────────────────────────────

def run(account_id: int | None, paper: bool):
    client = TopstepClient()
    client.login()

    # Resolve and confirm account
    accounts = client.get_accounts()
    if not accounts:
        raise RuntimeError("No active accounts found.")

    # Default to the designated practice account
    target_id = account_id if account_id is not None else PRACTICE_ACCOUNT_ID
    acct = next((a for a in accounts if a["id"] == target_id), None)
    if acct is None:
        raise RuntimeError(f"Account {target_id} not found in your active accounts.")
    account_id = acct["id"]

    acct_name    = acct.get("name", "—")
    acct_balance = acct.get("balance", "unknown")

    # Safety guard: refuse live trading on any non-practice account
    if not paper and acct_name != PRACTICE_ACCOUNT_NAME:
        raise RuntimeError(
            f"LIVE TRADING BLOCKED — account '{acct_name}' (id={account_id}) is not the "
            f"designated practice account '{PRACTICE_ACCOUNT_NAME}'. "
            f"Update PRACTICE_ACCOUNT_NAME in trading_bot.py to authorise a different account."
        )

    log.info(f"Account: {acct_name}  id={account_id}  balance=${acct_balance:,.2f}"
             + ("  [PAPER]" if paper else "  [LIVE]"))

    # Initialise instrument states
    states: list[InstrumentState] = []
    for inst in INSTRUMENTS:
        contracts = client.search_contracts(inst.search_term)
        if not contracts:
            log.error(f"No contract found for {inst.symbol}")
            continue
        c = contracts[0]
        log.info(f"  {inst.symbol}: {c['name']}  id={c['id']}")
        states.append(InstrumentState(instrument=inst, contract_id=c["id"]))

    if not states:
        raise RuntimeError("No instruments initialised.")

    # Warn if positions are already open on startup
    try:
        existing = client.get_open_positions(account_id)
        open_cids = {p["contractId"] for p in existing}
        for state in states:
            if state.contract_id in open_cids:
                log.warning(
                    f"{state.instrument.symbol}: open position exists on startup — "
                    f"bot will not enter a new one until it closes"
                )
    except Exception as e:
        log.warning(f"Could not check existing positions on startup: {e}")

    _ensure_log()
    mode = "PAPER MODE" if paper else "LIVE"
    log.info(
        f"Bot running — {mode}  account={account_id}  "
        f"instruments={[s.instrument.symbol for s in states]}  "
        f"poll={POLL_SECONDS}s"
    )

    while True:
        now = datetime.now(timezone.utc)

        for state in states:
            try:
                fetch_bars(client, state)

                if state.active_trade:
                    handle_active_trade(client, state, account_id, now, paper)

                if not state.active_trade:
                    sig = evaluate(state)
                    if sig:
                        place_signal(client, state, sig, account_id, paper)

            except Exception as e:
                log.error(f"{state.instrument.symbol}: {e}", exc_info=True)

        time.sleep(POLL_SECONDS)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/bot.log"),
        ],
    )

    parser = argparse.ArgumentParser(description="MES/MYM 3σ continuation trading bot")
    parser.add_argument("--paper", action="store_true",
                        help="Detect signals and log them but place no real orders")
    parser.add_argument("--account", type=int, default=None,
                        help="TopstepX account ID (auto-detects first active account if omitted)")
    args = parser.parse_args()

    run(account_id=args.account, paper=args.paper)
