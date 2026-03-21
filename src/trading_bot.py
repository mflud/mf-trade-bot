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
  4. Not in instrument-specific blackout window (DST-aware, per-instrument):
       MES: 08:00–09:00 ET conditional (block only if CSR<1.5; tech earnings pre-mkt)
       MYM: 09:00–09:30 ET unconditional; 15:00–16:00 ET unconditional
  5. |scaled| ≤ 5.0 (extreme event filter)

Execution:
  - 1 contract market order per signal, per instrument
  - Native API bracket: stop = 2σ below entry, target = 3σ above entry
    (both in ticks, OCO-managed server-side)
  - Force-close at market after 25 minutes if neither bracket is hit
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
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

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
MOM_BARS      = 8      # 8 × 5-min = 40-min CSR window (default; overridden dynamically)
CSR_THRESHOLD = 1.5    # min cumulative scaled return aligned with signal direction
SIGNAL_SIGMA  = 3.0
MAX_SCALED    = 5.0    # ignore extreme event spikes
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 25     # force-close after this many minutes
POLL_SECONDS  = 30

PL_N_BARS = 10    # 1-min bars to look back for PL computation
PL_THRESH = 0.50  # PL_aligned ≥ this → 2× sizing

ET = ZoneInfo("America/New_York")

LOG_PATH = Path("logs/bot_trades.csv")
LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "est_entry", "fill_price", "target", "stop",
    "sigma_pts", "scaled", "vol_ratio", "csr",
    "pl_aligned", "contracts",
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
    # Dynamic CSR window: list of (gk_ann_vol_upper_bound, mom_bars)
    csr_vol_windows: list = field(default_factory=lambda: [(1.0, 8)])
    # Per-instrument blackout windows: (start_h, start_m, end_h, end_m, conditional)
    # conditional=True: block only when CSR < threshold; False: always block.
    blackout_windows: list = field(default_factory=list)


INSTRUMENTS = [
    BotInstrument("MES", "MES", tick_size=0.25, point_value=5.00,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (8, 0, 9, 0, True),   # econ releases: block only if CSR<1.5
                  ]),
    BotInstrument("MYM", "MYM", tick_size=1.00, point_value=0.50,
                  csr_vol_windows=[(0.08, 4), (1.0, 8)],
                  blackout_windows=[
                      (9,  0,  9, 30, False),  # pre-open: EV=-0.076σ CSR-filtered
                      (15, 0, 16,  0, False),  # NYSE close: EV=-0.375σ CSR-filtered
                  ]),
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
    pl_aligned:  float | None = None
    contracts:   int = 1
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
    mean_vol:           float | None = None
    gk_ann_vol:         float = 0.0
    csr:                float = 0.0
    active_trade:       ActiveTrade | None = None
    last_evaluated_ts:  datetime | None = None


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
        "pl_aligned":  round(trade.pl_aligned, 4) if trade.pl_aligned is not None else "",
        "contracts":   trade.contracts,
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


# ── Helpers ──────────────────────────────────────────────────────────────────

GK_VOL_BARS   = 20
BARS_PER_YEAR = 252 * 23 * 60

def _gk_annualised_vol(bars: list) -> float:
    sample = bars[-GK_VOL_BARS:] if len(bars) >= GK_VOL_BARS else bars
    if len(sample) < 2:
        return 0.0
    vals = []
    for b in sample:
        if b.open <= 0 or b.high <= 0 or b.low <= 0 or b.close <= 0:
            continue
        hl = math.log(b.high / b.low) ** 2
        co = math.log(b.close / b.open) ** 2
        vals.append(0.5 * hl - (2 * math.log(2) - 1) * co)
    if not vals:
        return 0.0
    return math.sqrt(max(0.0, float(np.mean(vals))) * BARS_PER_YEAR / TF_MINUTES)


def get_mom_bars(gk_ann_vol: float, csr_vol_windows: list) -> int:
    """Return CSR window (bars) for current GK vol regime."""
    for upper, bars in csr_vol_windows:
        if gk_ann_vol < upper:
            return bars
    return csr_vol_windows[-1][1]


# ── PL confidence sizing ─────────────────────────────────────────────────────

def fetch_1min_pl(client: TopstepClient, contract_id: str,
                  signal_bar_ts: datetime, direction: int) -> float | None:
    """
    Fetch PL_N_BARS 1-min bars ending just before the signal 5-min bar and
    return PL_aligned = (signed path length) × direction.
    +1 = 1-min flow perfectly aligned; ≥ PL_THRESH → 2× sizing.
    Returns None on fetch error or insufficient data.
    """
    end   = signal_bar_ts
    start = end - timedelta(minutes=PL_N_BARS + 5)
    try:
        raw = client.get_bars(
            contract_id=contract_id, start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=PL_N_BARS + 5,
        )
        raw = list(reversed(raw))
    except Exception as e:
        log.debug(f"fetch_1min_pl error for {contract_id}: {e}")
        return None
    if len(raw) < PL_N_BARS + 1:
        return None
    closes   = np.array([b["c"] for b in raw[-(PL_N_BARS + 1):]])
    rets     = np.log(closes[1:] / closes[:-1])
    sum_absr = float(np.abs(rets).sum())
    if sum_absr == 0:
        return None
    pl = float(rets.sum()) / sum_absr
    return pl * direction


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
    prior_vols   = volumes[-TRAILING_BARS - 1:-1]
    active_vols  = prior_vols[prior_vols >= 10]
    mean_vol     = float(np.median(active_vols)) if len(active_vols) >= 10 else None
    last      = bars[-1]
    bar_ret   = math.log(last.close / last.open) if last.open else 0.0
    scaled    = bar_ret / sigma
    vol_ratio = (last.volume / mean_vol) if mean_vol is not None else None
    direction = 1 if scaled > 0 else -1

    # Dynamic CSR window based on current GK vol regime
    state.gk_ann_vol = _gk_annualised_vol(bars)
    mom_bars = get_mom_bars(state.gk_ann_vol, state.instrument.csr_vol_windows)
    if len(closes) >= mom_bars + 1:
        mom_rets = np.log(closes[-mom_bars:] / closes[-mom_bars - 1:-1])
        csr = float(mom_rets.sum()) / sigma * direction
    else:
        csr = 0.0

    state.sigma     = sigma
    state.sigma_pts = sigma_pts
    state.mean_vol  = mean_vol
    state.csr       = csr

    # Per-instrument blackout windows.
    bar_et = last.ts.astimezone(ET)
    bar_hm = (bar_et.hour, bar_et.minute)
    for sh, sm, eh, em, conditional in state.instrument.blackout_windows:
        if (sh, sm) <= bar_hm < (eh, em):
            if not conditional or csr < CSR_THRESHOLD:
                return None

    if (abs(scaled) >= SIGNAL_SIGMA and abs(scaled) <= MAX_SCALED
            and vol_ratio is not None and vol_ratio >= VOL_RATIO_MIN
            and csr >= CSR_THRESHOLD):
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
    end      = datetime.now(timezone.utc)
    max_mom  = max(bars for inst in INSTRUMENTS for _, bars in inst.csr_vol_windows)
    lookback = TRAILING_BARS + max_mom + 10
    start    = end - timedelta(minutes=TF_MINUTES * lookback)
    raw = client.get_bars(
        contract_id=state.contract_id,
        start=start, end=end,
        unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
        limit=lookback,
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

    # API ticks are signed relative to fill price: negative = below, positive = above.
    # Long:  stop below entry (negative), target above entry (positive)
    # Short: stop above entry (positive), target below entry (negative)
    stop_mag   = max(1, round(inst.stop_sigma   * sigma_pts / tick))
    target_mag = max(1, round(inst.target_sigma * sigma_pts / tick))
    stop_ticks   = -stop_mag   * direction
    target_ticks =  target_mag * direction
    side         = TopstepClient.BID if direction == 1 else TopstepClient.ASK
    dir_str      = "LONG" if direction == 1 else "SHORT"

    n_contracts = sig.get("contracts", 1)
    pl_aligned  = sig.get("pl_aligned")

    trade = ActiveTrade(
        instrument=inst, contract_id=state.contract_id,
        direction=direction, est_entry=sig["entry"],
        sigma_pts=sigma_pts, scaled=sig["scaled"],
        vol_ratio=sig["vol_ratio"], csr=sig["csr"],
        fired_at=sig["bar_ts"],
        pl_aligned=pl_aligned, contracts=n_contracts,
    )

    pl_note = f"  pl={pl_aligned:+.2f}" if pl_aligned is not None else ""
    size_note = f"  [⚡ 2× SIZE]" if n_contracts == 2 else ""

    if paper:
        log.info(
            f"[PAPER] {inst.symbol} {dir_str}  entry≈{sig['entry']:.2f}  "
            f"stop={stop_ticks}t ({inst.stop_sigma}σ)  "
            f"target={target_ticks}t ({inst.target_sigma}σ)  "
            f"scaled={sig['scaled']:+.2f}σ  csr={sig['csr']:+.2f}σ"
            f"{pl_note}  contracts={n_contracts}{size_note}"
        )
    else:
        resp = client.place_order(
            account_id=account_id,
            contract_id=state.contract_id,
            side=side, size=n_contracts,
            order_type=TopstepClient.ORDER_MARKET,
            stop_loss_ticks=stop_ticks,
            take_profit_ticks=target_ticks,
            custom_tag=f"bot_{inst.symbol}_{sig['bar_ts'].strftime('%Y%m%d%H%M%S')}_{random.randint(100,999)}",
        )
        trade.order_id = resp.get("orderId")
        log.info(
            f"ORDER PLACED  {inst.symbol} {dir_str}  order_id={trade.order_id}  "
            f"entry≈{sig['entry']:.2f}  stop={stop_ticks}t  target={target_ticks}t  "
            f"scaled={sig['scaled']:+.2f}σ  csr={sig['csr']:+.2f}σ"
            f"{pl_note}  contracts={n_contracts}{size_note}"
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
        # Position is gone — brackets closed it; cancel any residual OCO orders
        exit_price = _get_exit_price(client, account_id, trade, now)
        if exit_price is not None:
            # Actual fill from trade history — classify by proximity to brackets
            if abs(exit_price - trade.target_price()) <= abs(exit_price - trade.stop_price()):
                outcome = "TARGET"
            else:
                outcome = "STOPPED"
        else:
            # Lookup failed — infer from bar highs/lows (accurate to ~1 tick)
            outcome, exit_price = _classify_outcome_from_bars(trade, state.bars)
        _log_trade(trade, outcome, exit_price, now)
        state.active_trade = None
        try:
            n_cancelled = client.cancel_all_orders(account_id)
            if n_cancelled:
                log.info(f"{trade.instrument.symbol} {outcome}: cancelled {n_cancelled} residual order(s)")
        except Exception as e:
            log.warning(f"{trade.instrument.symbol}: cancel_all_orders failed: {e}")
        return

    # Force time exit if max hold exceeded
    if now >= trade.expires_at:
        log.info(f"{trade.instrument.symbol} max hold reached — cancelling brackets then closing at market")
        # Cancel brackets BEFORE closing: the API treats close_position() as a new
        # market order and can attach fresh brackets to it, or leave orphan OCO legs
        # that later fill and open an unwanted opposing position.
        try:
            n_cancelled = client.cancel_all_orders(account_id)
            if n_cancelled:
                log.info(f"{trade.instrument.symbol}: cancelled {n_cancelled} bracket order(s) before time exit")
        except Exception as e:
            log.warning(f"{trade.instrument.symbol}: pre-close cancel_all failed: {e}")
        try:
            client.close_position(account_id, trade.contract_id)
        except Exception as e:
            log.error(f"Failed to close position for {trade.instrument.symbol}: {e}")
            return
        exit_price = state.bars[-1].close if state.bars else (trade.fill_price or trade.est_entry)
        _log_trade(trade, "TIME EXIT", exit_price, now)
        state.active_trade = None
        # Cancel again in case the closing order itself spawned new brackets
        try:
            n_cancelled = client.cancel_all_orders(account_id)
            if n_cancelled:
                log.info(f"{trade.instrument.symbol} TIME EXIT: cancelled {n_cancelled} residual order(s) after close")
        except Exception as e:
            log.warning(f"{trade.instrument.symbol}: post-close cancel_all failed: {e}")


def _get_exit_price(client: TopstepClient, account_id: int,
                    trade: ActiveTrade, now: datetime) -> float | None:
    """
    Try to get actual exit price from trade history.
    Returns None if lookup fails or returns nothing useful — caller should
    fall back to bracket prices rather than fill_price to avoid pnl=0.
    """
    try:
        trades = client.search_trades(account_id, trade.fired_at, now)
        closing = [
            t for t in trades
            if t.get("contractId") == trade.contract_id
            and datetime.fromisoformat(t.get("timestamp", "1970")).replace(tzinfo=timezone.utc)
                > trade.fired_at
        ]
        if closing:
            price = float(closing[-1].get("price", 0))
            if price > 0:
                return price
    except Exception as e:
        log.warning(f"Could not fetch trade history for exit price: {e}")
    return None


def _classify_outcome_from_bars(trade: ActiveTrade, bars: list[Bar]) -> tuple[str, float]:
    """
    Infer whether a bracket-closed trade hit target or stop by scanning bar
    data since the signal fired.  Returns (outcome, exit_price) using the
    bracket price as the exit, which is accurate to within one tick for
    exchange-managed OCO orders.
    """
    for bar in bars:
        if bar.ts <= trade.fired_at:
            continue
        if trade.direction == 1:   # long
            if bar.high >= trade.target_price():
                return "TARGET",  trade.target_price()
            if bar.low  <= trade.stop_price():
                return "STOPPED", trade.stop_price()
        else:                      # short
            if bar.low  <= trade.target_price():
                return "TARGET",  trade.target_price()
            if bar.high >= trade.stop_price():
                return "STOPPED", trade.stop_price()
    # No bracket found in available bars — classify by which is closer to last close
    last_close = bars[-1].close if bars else (trade.fill_price or trade.est_entry)
    if abs(last_close - trade.target_price()) <= abs(last_close - trade.stop_price()):
        return "TARGET",  trade.target_price()
    return "STOPPED", trade.stop_price()


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
                    last_bar_ts = state.bars[-1].ts if state.bars else None
                    if sig and last_bar_ts != state.last_evaluated_ts:
                        state.last_evaluated_ts = last_bar_ts
                        pl = fetch_1min_pl(client, state.contract_id,
                                           sig["bar_ts"], sig["direction"])
                        sig["pl_aligned"] = pl
                        sig["contracts"]  = 2 if (pl is not None and pl >= PL_THRESH) else 1
                        place_signal(client, state, sig, account_id, paper)
                    elif last_bar_ts != state.last_evaluated_ts:
                        state.last_evaluated_ts = last_bar_ts

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
