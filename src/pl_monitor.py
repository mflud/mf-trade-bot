"""
PL_MOM Trade Monitor — simple real-time signal display for manual following.

Shows BUY LONG / SELL SHORT / CLOSE TRD / NO TRADE for MES and MNQ
using the same PL_MOM logic as trading_bot.py.

Usage:
    python src/pl_monitor.py
"""

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from zoneinfo import ZoneInfo

sys.path.insert(0, "src")
from topstep_client import TopstepClient, get_5s_bars_from_db, bars_db_available

# ── Constants (keep in sync with signal_monitor.py / trading_bot.py) ──────────
PL_MOM_WINDOW        = 6       # 5s bars = 30s
PL_MOM_ENTRY_PL      = 0.70
PL_MOM_MOVE_BPS      = 12.0
PL_MOM_EXIT_PL       = 0.40
PL_MOM_STOP_BPS      = 10.0
PL_MOM_MIN_HOLD_S    = 10
PL_MOM_MAX_HOLD_S    = 120
PL_MOM_5S_FETCH      = 130   # covers sigma lookback (120) + window (6) + margin
PL_MOM_SIGMA_N       = 3.0   # effective threshold = max(floor, N × σ_30s_bps)
PL_MOM_SIGMA_LB      = 120   # 5s bars for σ computation (10 min)
HISTORY_BARS         = 24    # 24 × 5s = 120s of history shown
CLOSE_TRD_SHOW_S  = 8       # how long to display CLOSE TRD after signal clears
POLL_INTERVAL_S   = 2

ET       = ZoneInfo("America/New_York")
LOCAL_TZ = datetime.now().astimezone().tzinfo
ALERT    = "/System/Library/Sounds/Pop.aiff"

INSTRUMENTS = [
    {"symbol": "MES", "search": "MES", "entry_pl": 0.80, "move_bps": 8.0,  "stop_bps": 7.0, "exit_pl": 0.40},
    {"symbol": "MNQ", "search": "MNQ", "entry_pl": 0.70, "move_bps": 12.0, "stop_bps": 8.0, "exit_pl": 0.20},
]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Bar:
    ts:     datetime
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float


@dataclass
class PLMomSignal:
    direction: int        # +1 long, -1 short
    entry:     float
    stop:      float
    pl:        float
    move_bps:  float
    bar_ts:    datetime

    def stop_pts(self): return abs(self.stop - self.entry)
    def expires_at(self): return self.bar_ts + timedelta(seconds=PL_MOM_MAX_HOLD_S)


@dataclass
class InstrumentState:
    symbol:      str
    entry_pl:    float = PL_MOM_ENTRY_PL
    move_bps:    float = PL_MOM_MOVE_BPS   # floor
    stop_bps:    float = PL_MOM_STOP_BPS
    exit_pl:     float = PL_MOM_EXIT_PL
    sigma_lb:    int   = PL_MOM_SIGMA_LB
    cid:         str = ""
    bars_5s:     list = field(default_factory=list)
    history:     list = field(default_factory=list)   # (ts, pl, dir_sym, move_bps)
    last_bar_ts: datetime | None = None
    active_sig:         PLMomSignal | None = None
    entry_ts:           datetime | None = None
    close_until:        datetime | None = None   # show CLOSE TRD until this time
    last_signal_bar_ts: datetime | None = None   # prevent re-entry on same bar
    sigma_30s_bps:      float = 0.0              # rolling σ of 30s moves in bps


# ── Display helpers ───────────────────────────────────────────────────────────

def _pl_bar(pl: float, dir_sym: str, half: int = 6) -> str:
    """Bidirectional bar: green extends right for longs, red left for shorts."""
    filled = max(0, min(half, round(min(1.0, pl) * half)))
    empty  = half - filled
    if dir_sym == "▲":
        return f"{'░' * half}|[green]{'█' * filled}{'░' * empty}[/]"
    else:
        return f"[red]{'░' * empty}{'█' * filled}[/]|{'░' * half}"


# ── Logic ──────────────────────────────────────────────────────────────────────

def _compute_sigma(bars: list, lookback: int) -> float:
    """σ of rolling 30s (PL_MOM_WINDOW × 5s) net moves in bps over last `lookback` bars."""
    closes = np.array([b.close for b in bars], dtype=float)
    if len(closes) < PL_MOM_WINDOW + 2:
        return 0.0
    rets = np.log(closes[1:] / closes[:-1])
    n = len(rets)
    lb = min(lookback, n - PL_MOM_WINDOW + 1)
    moves = []
    for i in range(n - lb, n - PL_MOM_WINDOW + 1):
        if i < 0:
            continue
        moves.append(abs(rets[i:i + PL_MOM_WINDOW].sum()))
    if len(moves) < 4:
        return 0.0
    return float(np.std(moves)) * 10000


_last_alert_time = 0.0
ALERT_COOLDOWN_S = 15.0   # minimum seconds between any two alerts

def play_alert():
    global _last_alert_time
    t = time.time()
    if t - _last_alert_time < ALERT_COOLDOWN_S:
        return
    _last_alert_time = t
    threading.Thread(
        target=lambda: subprocess.run(["afplay", ALERT], check=False),
        daemon=True,
    ).start()


def evaluate(state: InstrumentState) -> PLMomSignal | None:
    """Same PL_MOM evaluation as signal_monitor / trading_bot."""
    bars = state.bars_5s
    if len(bars) < PL_MOM_WINDOW + 1:
        return None
    window = bars[-PL_MOM_WINDOW:]
    for i in range(1, len(window)):
        if (window[i].ts - window[i - 1].ts).total_seconds() > 8:
            return None
    last = window[-1]
    if state.active_sig and last.ts == state.active_sig.bar_ts:
        return None
    closes  = np.array([b.close for b in window], dtype=float)
    rets    = np.log(closes[1:] / closes[:-1])
    sum_abs = float(np.abs(rets).sum())
    if sum_abs == 0:
        return None
    pl = float(abs(rets.sum()) / sum_abs)
    if pl < state.entry_pl:
        return None
    net_ret  = float(rets.sum())
    entry    = last.close
    move_bps = abs(net_ret) * 10000
    sigma = state.sigma_30s_bps
    effective_thr = max(state.move_bps, PL_MOM_SIGMA_N * sigma) if sigma > 0 else state.move_bps
    if move_bps < effective_thr:
        return None
    direction = 1 if net_ret > 0 else -1
    stop = entry * (1.0 - direction * state.stop_bps / 10000.0)
    return PLMomSignal(direction=direction, entry=entry, stop=stop,
                       pl=pl, move_bps=move_bps, bar_ts=last.ts)


# ── Display ────────────────────────────────────────────────────────────────────

def build_panel(state: InstrumentState, now: datetime) -> Panel:
    sig     = state.active_sig
    closing = state.close_until is not None and now < state.close_until

    # Status
    if closing:
        status, style, border = "CLOSE TRD", "bold yellow on black", "yellow"
    elif sig is None:
        status, style, border = "NO TRADE",  "bold",                    "default"
    elif sig.direction == 1:
        status, style, border = "BUY LONG",  "bold green",           "green"
    else:
        status, style, border = "SELL SHORT", "bold red",            "red"

    root = Table.grid(padding=(0, 0))
    root.add_column(justify="center")

    # ── Status row (large, centred) ──────────────────────────────────────────
    hdr = Table.grid()
    hdr.add_column(justify="center", min_width=28)
    hdr.add_row(f"[{style}]  {status}  [/]")
    root.add_row(hdr)
    root.add_row("")   # blank separator

    # ── Active signal details ────────────────────────────────────────────────
    if sig and not closing:
        det = Table.grid(padding=(0, 1))
        det.add_column(width=8,  justify="right")
        det.add_column()
        rem_s   = max(0, int((sig.expires_at() - now).total_seconds()))
        rem_str = f"{rem_s // 60}m {rem_s % 60:02d}s"
        dir_str = "▲ LONG" if sig.direction == 1 else "▼ SHORT"
        clr     = "green" if sig.direction == 1 else "red"
        det.add_row("", f"[bold {clr}]{dir_str}[/]  "
                        f"PL={sig.pl:.3f}  {sig.move_bps:.1f}bp")
        det.add_row("Entry:",   f"[bold]{sig.entry:,.2f}[/]")
        det.add_row("Stop:",    f"[bold red]{sig.stop:,.2f}[/]"
                                f"  ({sig.stop_pts():.2f} pts)")
        det.add_row("Expires:", rem_str)
        root.add_row(det)
        root.add_row("")

    # ── History table — last HISTORY_BARS entries ────────────────────────────
    hist = state.history[-HISTORY_BARS:]
    if hist:
        ht = Table(box=box.SIMPLE, show_header=True, padding=(0, 1),
                   header_style="bold")
        ht.add_column("time",  justify="right")
        ht.add_column("",      justify="center")
        ht.add_column("",      justify="left", no_wrap=True)
        ht.add_column("PL",    justify="right")
        ht.add_column("bp",    justify="right")
        for ts, pl, dir_sym, move in reversed(hist):
            t_str  = ts.astimezone(LOCAL_TZ).strftime("%H:%M:%S")
            pl_sty = ("bold green" if pl >= state.entry_pl else
                      "yellow"     if pl >= state.entry_pl * 0.85 else "")
            bp_sty = "bold green" if move >= state.move_bps else ""
            ht.add_row(t_str, dir_sym,
                       _pl_bar(pl, dir_sym),
                       f"[{pl_sty}]{pl:.3f}[/]" if pl_sty else f"{pl:.3f}",
                       f"[{bp_sty}]{move:.1f}[/]" if bp_sty else f"{move:.1f}")
        root.add_row(ht)
    elif not state.bars_5s:
        root.add_row("warming up…")

    sigma = state.sigma_30s_bps
    eff_thr = max(state.move_bps, PL_MOM_SIGMA_N * sigma) if sigma > 0 else state.move_bps
    if sigma > 0 and eff_thr > state.move_bps:
        thr_str = f"bp ≥ {eff_thr:.1f} (floor {state.move_bps:.0f}, σ={sigma:.1f})"
    elif sigma > 0:
        thr_str = f"bp ≥ {state.move_bps:.0f} (σ={sigma:.1f})"
    else:
        thr_str = f"bp ≥ {state.move_bps:.0f}"
    thresh = Table.grid()
    thresh.add_column(justify="center", min_width=28)
    thresh.add_row(f"entry: PL ≥ {state.entry_pl:.2f}  {thr_str}")
    thresh.add_row(f"exit:  PL ≤ {state.exit_pl:.2f}  stop {state.stop_bps:.0f}bp")
    root.add_row(thresh)

    title_markup = (f"[bold]PL MOM  {state.symbol}[/]" if border != "default"
                    else f"PL MOM  {state.symbol}")
    return Panel(root, title=title_markup,
                 border_style=border, padding=(0, 1), expand=False)


def build_display(states: list[InstrumentState], now: datetime) -> Table:
    hdr = Table.grid(padding=(0, 2))
    hdr.add_column(justify="center")
    hdr.add_row(f"{now.astimezone(LOCAL_TZ).strftime('%H:%M:%S')}  "
                f"PL_MOM Monitor")

    panels = Table.grid(padding=(0, 2))
    for _ in states:
        panels.add_column()
    panels.add_row(*[build_panel(s, now) for s in states])

    note = Table.grid()
    note.add_column(justify="center")
    note.add_row("PL and bp computed over trailing six 5-second intervals.")

    root = Table.grid()
    root.add_column()
    root.add_row(hdr)
    root.add_row(panels)
    root.add_row(note)
    return root


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    states = [InstrumentState(symbol=cfg["symbol"],
                              entry_pl=cfg["entry_pl"],
                              move_bps=cfg["move_bps"],
                              stop_bps=cfg["stop_bps"],
                              exit_pl=cfg["exit_pl"],
                              sigma_lb=cfg.get("sigma_lb", PL_MOM_SIGMA_LB)) for cfg in INSTRUMENTS]
    console = Console()

    client = TopstepClient()
    client.use_shared_token()  # reuse bar_collector's token — avoids multiple-sessions disconnect

    for state, cfg in zip(states, INSTRUMENTS):
        contracts = client.search_contracts(cfg["search"])
        if contracts:
            state.cid = str(contracts[0]["id"])
            console.print(f"  {state.symbol}: id={state.cid}")
        else:
            console.print(f"[red]No contract for {state.symbol}[/]")

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            now    = datetime.now(timezone.utc)
            now_et = now.astimezone(ET)
            in_rth = (9, 30) <= (now_et.hour, now_et.minute) < (16, 0)

            should_alert = False
            for state in states:
                if not in_rth:
                    continue

                # ── Fetch 5s bars ─────────────────────────────────────────────
                try:
                    if bars_db_available():
                        raw = get_5s_bars_from_db(state.symbol, PL_MOM_5S_FETCH)
                    else:
                        floor = datetime.fromtimestamp(
                            (int(now.timestamp()) // 5) * 5, tz=timezone.utc)
                        start = floor - timedelta(seconds=5 * (PL_MOM_5S_FETCH + 5))
                        raw   = client.get_bars(
                            contract_id=state.cid, start=start, end=floor,
                            unit=TopstepClient.SECOND, unit_number=5,
                            limit=PL_MOM_5S_FETCH,
                        )
                    if raw:
                        state.bars_5s = [
                            Bar(ts=datetime.fromisoformat(b["t"]),
                                open=b["o"], high=b["h"], low=b["l"],
                                close=b["c"], volume=b["v"])
                            for b in raw
                        ]
                        state.sigma_30s_bps = _compute_sigma(state.bars_5s, state.sigma_lb)
                except Exception:
                    pass

                # ── Update history on each new bar ────────────────────────────
                bar_ts = state.bars_5s[-1].ts if state.bars_5s else None
                if bar_ts and bar_ts != state.last_bar_ts:
                    state.last_bar_ts = bar_ts
                    if len(state.bars_5s) >= PL_MOM_WINDOW:
                        window  = state.bars_5s[-PL_MOM_WINDOW:]
                        closes  = np.array([b.close for b in window], dtype=float)
                        rets    = np.log(closes[1:] / closes[:-1])
                        sum_abs = float(np.abs(rets).sum())
                        if sum_abs > 0:
                            net = float(rets.sum())
                            state.history.append((
                                bar_ts,
                                abs(net) / sum_abs,
                                "▲" if net > 0 else "▼",
                                abs(net) * 10000,
                            ))
                            if len(state.history) > 40:
                                state.history = state.history[-40:]

                    # Evaluate for new signal
                    new_sig = evaluate(state)
                    if (new_sig and state.active_sig is None
                            and new_sig.bar_ts != state.last_signal_bar_ts):
                        state.active_sig         = new_sig
                        state.entry_ts           = now
                        state.close_until        = None
                        state.last_signal_bar_ts = new_sig.bar_ts
                        should_alert = True

                # ── Expire / PL exit ──────────────────────────────────────────
                if state.active_sig is not None:
                    sig          = state.active_sig
                    min_hold_ok  = (state.entry_ts is not None and
                                    (now - state.entry_ts).total_seconds() >= PL_MOM_MIN_HOLD_S)
                    expired      = now >= sig.expires_at()
                    pl_exit      = False
                    if min_hold_ok and len(state.bars_5s) >= PL_MOM_WINDOW:
                        window  = state.bars_5s[-PL_MOM_WINDOW:]
                        closes  = np.array([b.close for b in window], dtype=float)
                        rets    = np.log(closes[1:] / closes[:-1])
                        sa      = float(np.abs(rets).sum())
                        cur_pl  = abs(float(rets.sum()) / sa) if sa > 0 else 0.0
                        pl_exit = cur_pl <= state.exit_pl
                    if expired or pl_exit:
                        state.active_sig  = None
                        state.entry_ts    = None
                        state.close_until = now + timedelta(seconds=CLOSE_TRD_SHOW_S)

            if should_alert:
                play_alert()
            live.update(build_display(states, now))
            time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
