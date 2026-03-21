"""
MES-only Signal Monitor — two signal types displayed side by side:

  1. MOMENTUM  (existing)  — 5-min bar, 3σ scaled return + vol + CSR filter
                             stop=2σ  target=3σ  EV≈+0.07σ/signal

  2. ORB LONG  (new)       — 15-min Opening Range Breakout, LONG only
                             Fires when first 5-min close > ORB high,
                             ORB width > 15.25 pts, morning 9:45-10:30 ET
                             or power-hour 13:30-16:00 ET.
                             stop=2σ  target=2σ  EV≈+0.61R/signal

Run modes:
  python src/signal_monitor_mes.py           # live (requires .env credentials)
  python src/signal_monitor_mes.py --demo    # static demo with synthetic data
"""

import argparse
import csv
import math
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, "src")

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Strategy parameters ────────────────────────────────────────────────────────

TF_MINUTES    = 5
TRAILING_BARS = 20
GK_VOL_BARS   = 20
CSR_THRESHOLD = 1.5
SIGNAL_SIGMA  = 3.0
MAX_SCALED    = 5.0
VOL_RATIO_MIN = 1.5
MAX_HOLD_MIN  = 25
BARS_PER_YEAR = 252 * 23 * 60

# ORB parameters (from backtest_orb.py analysis)
ORB_BARS      = 3          # 3 × 5-min = 15-min opening range
ORB_WIDTH_MIN = 15.25      # pts — wide tertile cutoff
ORB_STOP_SIG  = 2.0        # stop distance in σ
ORB_TGT_SIG   = 2.0        # target distance in σ  (2σ:2σ = +0.61R EV)
ORB_WINDOWS   = [          # (start_h, start_m, end_h, end_m, label)
    (9,  45, 10, 30, "Morning"),
    (13, 30, 16,  0, "Power hr"),
]

ET = ZoneInfo("America/New_York")

LOG_PATH     = Path("logs/signals.csv")
ORB_LOG_PATH = Path("logs/orb_signals.csv")

LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "entry", "target", "stop",
    "sigma_pts", "scaled", "vol_ratio", "csr",
    "outcome", "pnl_pts", "pnl_sigma",
]
ORB_LOG_FIELDS = [
    "fired_at", "resolved_at", "symbol", "direction",
    "entry", "target", "stop",
    "orb_high", "orb_low", "orb_width", "sigma_pts",
    "window", "outcome", "pnl_pts", "pnl_r",
]

REGIME_THRESHOLDS = [
    (0.10, "QUIET",    "dim"),
    (0.15, "NORMAL",   "cyan"),
    (0.20, "ELEVATED", "yellow"),
    (0.30, "ACTIVE",   "orange1"),
    (1.00, "HIGH VOL", "red"),
]


ALERT_SOUND = "/System/Library/Sounds/Ping.aiff"


def play_alert():
    """Play alert sound in background thread (non-blocking)."""
    threading.Thread(
        target=lambda: subprocess.run(["afplay", ALERT_SOUND], check=False),
        daemon=True,
    ).start()


# ── Configs ────────────────────────────────────────────────────────────────────

@dataclass
class InstrumentConfig:
    symbol:      str
    search_term: str
    stop_sigma:  float
    target_sigma: float
    point_value: float
    ev_sigma:    float
    csr_vol_windows: list = field(default_factory=lambda: [(1.0, 8)])
    blackout_windows: list = field(default_factory=list)


MES_CFG = InstrumentConfig(
    "MES", "MES", stop_sigma=2.0, target_sigma=3.0,
    point_value=5.00, ev_sigma=0.073,
    csr_vol_windows=[(0.08, 4), (1.0, 8)],
    blackout_windows=[(8, 0, 9, 0, True)],
)

ORB_CFG = InstrumentConfig(
    "MES", "MES", stop_sigma=ORB_STOP_SIG, target_sigma=ORB_TGT_SIG,
    point_value=5.00, ev_sigma=1.22,   # 0.61R × 2σ/R
    csr_vol_windows=[], blackout_windows=[],
)

console = Console()


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
class Signal:
    cfg:        InstrumentConfig
    direction:  int
    entry:      float
    sigma:      float
    sigma_pts:  float
    scaled:     float
    vol_ratio:  float
    csr:        float
    bar_ts:     datetime
    target:     float = field(init=False)
    stop:       float = field(init=False)
    expires_at: datetime = field(init=False)

    def __post_init__(self):
        self.target     = self.entry + self.direction * self.cfg.target_sigma * self.sigma_pts
        self.stop       = self.entry - self.direction * self.cfg.stop_sigma   * self.sigma_pts
        self.expires_at = self.bar_ts + timedelta(minutes=MAX_HOLD_MIN)

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)


@dataclass
class OrbSignal:
    entry:      float
    target:     float
    stop:       float
    orb_high:   float
    orb_low:    float
    sigma_pts:  float
    window:     str
    bar_ts:     datetime

    def target_pts(self): return abs(self.target - self.entry)
    def stop_pts(self):   return abs(self.stop   - self.entry)
    def risk_pts(self):   return self.stop_pts()


@dataclass
class OrbState:
    session_date:    date | None = None
    orb_high:        float = 0.0
    orb_low:         float = 0.0
    orb_bars_seen:   int   = 0          # RTH bars accumulated for ORB
    orb_complete:    bool  = False
    morning_fired:   bool  = False
    power_hr_fired:  bool  = False
    active_signal:   OrbSignal | None = None


@dataclass
class RecentSignal:
    symbol:   str
    kind:     str     # "MOM" or "ORB"
    direction: int
    entry:    float
    target:   float
    stop:     float
    outcome:  str
    pnl_pts:  float
    fired_at: datetime


@dataclass
class InstrumentState:
    cfg:               InstrumentConfig
    cid:               str = ""
    cname:             str = ""
    bars:              list[Bar] = field(default_factory=list)
    sigma:             float = 0.0
    sigma_pts:         float = 0.0
    gk_ann_vol:        float = 0.0
    csr:               float = 0.0
    mean_vol:          float | None = None
    active_signal:     Signal | None = None
    orb:               OrbState = field(default_factory=OrbState)
    history:           list[RecentSignal] = field(default_factory=list)
    error:             str | None = None
    last_evaluated_ts: datetime | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def annualised_vol(sigma: float) -> float:
    return sigma * math.sqrt(BARS_PER_YEAR / TF_MINUTES)


def get_mom_bars(gk_ann_vol: float, csr_vol_windows: list) -> int:
    for upper, bars in csr_vol_windows:
        if gk_ann_vol < upper:
            return bars
    return csr_vol_windows[-1][1]


def gk_annualised_vol(bars: list) -> float:
    sample = bars[-GK_VOL_BARS:] if len(bars) >= GK_VOL_BARS else bars
    if len(sample) < 2:
        return float("nan")
    ln_hl = np.log(np.array([b.high / b.low   for b in sample]))
    ln_co = np.log(np.array([b.close / b.open for b in sample]))
    gk  = 0.5 * ln_hl ** 2 - (2 * math.log(2) - 1) * ln_co ** 2
    var = float(np.mean(gk))
    return math.sqrt(var * BARS_PER_YEAR / TF_MINUTES) if var > 0 else float("nan")


def regime_label(ann_vol: float) -> tuple[str, str]:
    for thresh, label, style in REGIME_THRESHOLDS:
        if ann_vol < thresh:
            return label, style
    return REGIME_THRESHOLDS[-1][1], REGIME_THRESHOLDS[-1][2]


def _next_bar_close(now: datetime) -> datetime:
    epoch_min      = int(now.timestamp() // 60)
    next_close_min = ((epoch_min // TF_MINUTES) + 1) * TF_MINUTES
    return datetime.fromtimestamp(next_close_min * 60, tz=timezone.utc)


def _orb_window(bar_et: datetime) -> str | None:
    """Return the active ORB window name or None."""
    hm = (bar_et.hour, bar_et.minute)
    for sh, sm, eh, em, label in ORB_WINDOWS:
        if (sh, sm) <= hm < (eh, em):
            return label
    return None


# ── ORB evaluation ─────────────────────────────────────────────────────────────

def evaluate_orb(state: InstrumentState) -> OrbSignal | None:
    """
    Incrementally update OrbState with the latest bar.
    Returns a new OrbSignal if a qualifying breakout just occurred.
    """
    if not state.bars:
        return None

    bar    = state.bars[-1]
    bar_et = bar.ts.astimezone(ET)
    today  = bar_et.date()
    orb    = state.orb

    # New session — reset ORB state
    if orb.session_date != today:
        orb.session_date   = today
        orb.orb_high       = 0.0
        orb.orb_low        = 0.0
        orb.orb_bars_seen  = 0
        orb.orb_complete   = False
        orb.morning_fired  = False
        orb.power_hr_fired = False
        orb.active_signal  = None

    hm = (bar_et.hour, bar_et.minute)

    # Accumulate ORB bars (9:30–9:44 ET inclusive, i.e. ts in [9:30, 9:45))
    if (9, 30) <= hm < (9, 30 + ORB_BARS * TF_MINUTES) and not orb.orb_complete:
        if orb.orb_bars_seen == 0:
            orb.orb_high = bar.high
            orb.orb_low  = bar.low
        else:
            orb.orb_high = max(orb.orb_high, bar.high)
            orb.orb_low  = min(orb.orb_low,  bar.low)
        orb.orb_bars_seen += 1
        if orb.orb_bars_seen >= ORB_BARS:
            orb.orb_complete = True
        return None

    if not orb.orb_complete:
        return None

    # Outside RTH
    if hm < (9, 30) or hm >= (16, 0):
        return None

    # Determine active window and whether we've already fired in it
    window = _orb_window(bar_et)
    if window is None:
        return None
    if window == "Morning"  and orb.morning_fired:
        return None
    if window == "Power hr" and orb.power_hr_fired:
        return None

    # Check breakout conditions
    orb_width = orb.orb_high - orb.orb_low
    if orb_width < ORB_WIDTH_MIN:
        return None

    sigma_pts = state.sigma_pts
    if sigma_pts <= 0:
        return None

    if bar.close > orb.orb_high:
        # LONG breakout
        entry  = bar.close
        stop   = entry - ORB_STOP_SIG * sigma_pts
        target = entry + ORB_TGT_SIG  * sigma_pts

        if window == "Morning":
            orb.morning_fired  = True
        else:
            orb.power_hr_fired = True

        sig = OrbSignal(
            entry=entry, target=target, stop=stop,
            orb_high=orb.orb_high, orb_low=orb.orb_low,
            sigma_pts=sigma_pts, window=window, bar_ts=bar.ts,
        )
        orb.active_signal = sig
        return sig

    return None


def _check_orb_resolution(sig: OrbSignal, bars: list[Bar]) -> tuple[str, float] | None:
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if bar.high >= sig.target:
            return "TARGET",  sig.target_pts()
        if bar.low  <= sig.stop:
            return "STOPPED", -sig.stop_pts()
    return None


# ── Momentum evaluation ────────────────────────────────────────────────────────

def evaluate(state: InstrumentState) -> Signal | None:
    bars    = state.bars
    closes  = np.array([b.close  for b in bars])
    volumes = np.array([b.volume for b in bars])

    trail = np.log(closes[1:] / closes[:-1])[-TRAILING_BARS:] \
            if len(closes) >= 2 else np.array([])
    sigma     = float(np.std(trail, ddof=1)) if len(trail) >= 2 else 0.0
    sigma_pts = sigma * closes[-1]
    warmed_up = len(closes) > TRAILING_BARS
    prior_vols  = volumes[-TRAILING_BARS - 1:-1]
    active_vols = prior_vols[prior_vols >= 10]
    mean_vol    = float(np.median(active_vols)) if len(active_vols) >= 10 else None

    state.sigma      = sigma
    state.sigma_pts  = sigma_pts
    state.mean_vol   = mean_vol
    state.gk_ann_vol = gk_annualised_vol(bars)

    last      = bars[-1]
    bar_ret   = math.log(last.close / last.open) if last.open else 0.0
    scaled    = bar_ret / sigma if sigma else 0.0
    vol_ratio = (last.volume / mean_vol) if mean_vol is not None else None

    direction = 1 if scaled > 0 else -1
    mom_bars  = get_mom_bars(state.gk_ann_vol, state.cfg.csr_vol_windows)
    if len(closes) >= mom_bars + 1:
        mom_rets  = np.log(closes[-mom_bars:] / closes[-mom_bars - 1:-1])
        state.csr = float(mom_rets.sum()) / sigma * direction if sigma else 0.0
    else:
        state.csr = 0.0

    bar_et = last.ts.astimezone(ET)
    bar_hm = (bar_et.hour, bar_et.minute)
    for sh, sm, eh, em, conditional in state.cfg.blackout_windows:
        if (sh, sm) <= bar_hm < (eh, em):
            if not conditional or state.csr < CSR_THRESHOLD:
                return None

    if (warmed_up
            and abs(scaled) >= SIGNAL_SIGMA and abs(scaled) <= MAX_SCALED
            and vol_ratio is not None and vol_ratio >= VOL_RATIO_MIN
            and state.csr >= CSR_THRESHOLD):
        return Signal(cfg=state.cfg,
                      direction=direction, entry=last.close,
                      sigma=sigma, sigma_pts=sigma_pts,
                      scaled=scaled, vol_ratio=vol_ratio, csr=state.csr,
                      bar_ts=last.ts)
    return None


# ── Panel builders ─────────────────────────────────────────────────────────────

def build_regime_panel(state: InstrumentState) -> Panel:
    sigma = state.sigma
    ann   = annualised_vol(sigma)
    gk    = state.gk_ann_vol
    gk_label, gk_style = regime_label(gk) if gk > 0 else regime_label(ann)
    cur_vol   = state.bars[-1].volume if state.bars else 0.0
    vol_ratio = (cur_vol / state.mean_vol) if state.mean_vol is not None else None

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=16)
    t.add_column(width=34)

    t.add_row("σ per bar:",
              f"[bold]{sigma * 10000:.2f} bps[/]  │  {state.sigma_pts:.2f} pts")
    t.add_row("Ann. vol (CR):",
              f"{ann*100:.1f}%  [dim](close-return, 100 bars)[/]")
    t.add_row("Ann. vol (GK):",
              f"[bold]{gk*100:.1f}%[/]  [dim](Garman-Klass, 20 bars)[/]")
    t.add_row("Regime:",
              f"[bold {gk_style}]{gk_label}[/]  [dim](GK)[/]")
    t.add_row("Avg volume:",
              f"{state.mean_vol:,.0f}" if state.mean_vol is not None else "[dim]—[/]")
    t.add_row("Cur volume:",
              f"{cur_vol:,.0f}  ({vol_ratio:.1f}× avg)" if vol_ratio is not None
              else f"{cur_vol:,.0f}  [dim](warming up)[/]")

    return Panel(t, title="[bold]VOL REGIME[/]",
                 border_style="blue", padding=(0, 1))


def build_bar_panel(state: InstrumentState) -> Panel:
    bar   = state.bars[-1]
    sigma = state.sigma
    ret       = math.log(bar.close / bar.open) if bar.open else 0.0
    scaled    = ret / sigma if sigma else 0.0
    vol_ratio = (bar.volume / state.mean_vol) if state.mean_vol is not None else None

    sc_style = ("green" if scaled > 0 else "red") if abs(scaled) >= SIGNAL_SIGMA \
               else ("yellow" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "white")
    vr_style = ("green" if vol_ratio >= VOL_RATIO_MIN else
                ("yellow" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "white")) \
               if vol_ratio is not None else "dim"
    sc_check = "✓" if abs(scaled) >= SIGNAL_SIGMA else \
               ("~" if abs(scaled) >= SIGNAL_SIGMA * 0.7 else "✗")
    vr_check = ("✓" if vol_ratio >= VOL_RATIO_MIN else
                ("~" if vol_ratio >= VOL_RATIO_MIN * 0.7 else "✗")) \
               if vol_ratio is not None else "?"

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=10)
    t.add_column(width=30)

    t.add_row("Open:",   f"{bar.open:,.2f}")
    t.add_row("High:",   f"[green]{bar.high:,.2f}[/]")
    t.add_row("Low:",    f"[red]{bar.low:,.2f}[/]")
    t.add_row("Close:",  f"[bold]{bar.close:,.2f}[/]")
    t.add_row("Volume:", f"{bar.volume:,.0f}  [{vr_style}]{vol_ratio:.2f}× "
                         f"[thr {VOL_RATIO_MIN:.1f}×] {vr_check}[/]"
              if vol_ratio is not None else
              f"{bar.volume:,.0f}  [dim](warming up)[/]")
    val_style = "bold green" if scaled > 0 else "bold red"
    t.add_row("Scaled:", f"[{val_style}]{scaled:+.2f}σ[/]  "
                         f"[{sc_style}][thr {SIGNAL_SIGMA:.0f}σ] {sc_check}[/]")

    return Panel(t, title=f"[bold]LAST {TF_MINUTES}-MIN BAR[/]",
                 border_style="blue", padding=(0, 1))


def build_momentum_panel(state: InstrumentState, now: datetime) -> Panel:
    sig = state.active_signal
    cfg = state.cfg

    if sig is None:
        sigma = state.sigma
        bar   = state.bars[-1]
        scaled = math.log(bar.close / bar.open) / sigma if sigma and bar.open else 0.0
        vol_ratio = (bar.volume / state.mean_vol) if state.mean_vol is not None else None
        csr   = state.csr
        mom_bars = get_mom_bars(state.gk_ann_vol, state.cfg.csr_vol_windows)

        sc_pct  = min(abs(scaled) / SIGNAL_SIGMA  * 100, 100)
        vr_pct  = min(vol_ratio   / VOL_RATIO_MIN * 100, 100) if vol_ratio else 0.0
        csr_pct = min(abs(csr)    / CSR_THRESHOLD * 100, 100)
        bar_sc  = "█" * int(sc_pct  / 5) + "░" * (20 - int(sc_pct  / 5))
        bar_vr  = "█" * int(vr_pct  / 5) + "░" * (20 - int(vr_pct  / 5))
        bar_csr = "█" * int(csr_pct / 5) + "░" * (20 - int(csr_pct / 5))
        csr_style = "green" if csr >= CSR_THRESHOLD else ("yellow" if csr > 0 else "red")

        t = Table.grid(padding=(0, 1))
        t.add_column(style="dim", width=16)
        t.add_column()
        t.add_row("Scaled return:", f"[yellow]{bar_sc}[/] {abs(scaled):.2f}σ / {SIGNAL_SIGMA:.0f}σ")
        t.add_row("Volume ratio:",  f"[yellow]{bar_vr}[/] {vol_ratio:.2f}× / {VOL_RATIO_MIN:.1f}×"
                  if vol_ratio else f"[dim]{bar_vr}[/] warming up")
        t.add_row(f"Mom({mom_bars*TF_MINUTES}m):",
                  f"[{csr_style}]{bar_csr}[/] {csr:+.2f}σ / {CSR_THRESHOLD:.1f}σ")

        price = bar.close
        sp    = state.sigma_pts
        nt = Table.grid(padding=(0, 1))
        nt.add_column(style="dim", width=8)
        nt.add_column()
        if sp > 0:
            nt.add_row("LONG:",  f"[dim]tgt {price + cfg.target_sigma*sp:,.2f}  "
                                  f"sl {price - cfg.stop_sigma*sp:,.2f}[/]")
            nt.add_row("SHORT:", f"[dim]tgt {price - cfg.target_sigma*sp:,.2f}  "
                                  f"sl {price + cfg.stop_sigma*sp:,.2f}[/]")

        watching = Panel(t,  title="[bold yellow]⬤  MOMENTUM — WATCHING[/]",
                         border_style="yellow", padding=(0, 1))
        levels   = Panel(nt, title="[dim]INDICATIVE LEVELS[/]",
                         border_style="dim",    padding=(0, 1))
        col = Table.grid()
        col.add_column()
        col.add_row(watching)
        col.add_row(levels)
        return col

    direction_str = "LONG  ▲" if sig.direction == 1 else "SHORT ▼"
    color  = "green" if sig.direction == 1 else "red"
    rem    = sig.expires_at - now
    rem_s  = int(rem.total_seconds())
    rem_str = f"{rem_s // 60}m {rem_s % 60:02d}s" if rem_s > 0 else "[blink]EXPIRED[/]"
    rr = sig.target_pts() / sig.stop_pts() if sig.stop_pts() else 0.0

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=10)
    t.add_column()
    t.add_row("Entry:",   f"[bold]{sig.entry:,.2f}[/]")
    t.add_row("Target:",  f"[bold {color}]{sig.target:,.2f}[/]  "
                           f"([{color}]+{sig.target_pts():.2f} pts[/] │ +{cfg.target_sigma:.1f}σ)")
    t.add_row("Stop:",    f"[bold red]{sig.stop:,.2f}[/]  "
                           f"([red]−{sig.stop_pts():.2f} pts[/] │ −{cfg.stop_sigma:.1f}σ)")
    t.add_row("R:R / EV:", f"{rr:.2f}:1  (EV ≈ +{cfg.ev_sigma:.2f}σ)")
    t.add_row("Expires:", f"{rem_str}")

    return Panel(t, title=f"[bold {color}]⬤  MOMENTUM — {direction_str}[/]",
                 border_style=color, padding=(0, 1))


def build_orb_panel(state: InstrumentState, now: datetime) -> Panel:
    orb = state.orb
    bar_et = state.bars[-1].ts.astimezone(ET) if state.bars else None
    hm     = (bar_et.hour, bar_et.minute) if bar_et else (0, 0)

    t = Table.grid(padding=(0, 1))
    t.add_column(style="dim", width=14)
    t.add_column()

    # ORB range info
    if orb.orb_complete:
        width = orb.orb_high - orb.orb_low
        w_style = "green" if width >= ORB_WIDTH_MIN else "red"
        w_flag  = " ✓ WIDE" if width >= ORB_WIDTH_MIN else f" ✗ need>{ORB_WIDTH_MIN:.1f}"
        t.add_row("ORB high:", f"{orb.orb_high:,.2f}")
        t.add_row("ORB low:",  f"{orb.orb_low:,.2f}")
        t.add_row("ORB width:", f"[{w_style}]{width:.2f} pts{w_flag}[/]")
    elif orb.session_date == (bar_et.date() if bar_et else None):
        t.add_row("ORB:", f"[yellow]Building… {orb.orb_bars_seen}/{ORB_BARS} bars[/]")
    else:
        t.add_row("ORB:", "[dim]Waiting for RTH open (9:30 ET)[/]")

    # Active ORB signal
    if orb.active_signal:
        sig = orb.active_signal
        rem  = (sig.bar_ts + timedelta(minutes=MAX_HOLD_MIN)) - now
        rem_s = int(rem.total_seconds())
        rem_str = f"{rem_s // 60}m {rem_s % 60:02d}s" if rem_s > 0 else "[blink]EXPIRED[/]"
        t.add_row("", "")
        t.add_row("Entry:",   f"[bold]{sig.entry:,.2f}[/]  [dim]({sig.window})[/]")
        t.add_row("Target:",  f"[bold green]{sig.target:,.2f}[/]  "
                               f"([green]+{sig.target_pts():.2f} pts[/] │ +{ORB_TGT_SIG:.1f}σ)")
        t.add_row("Stop:",    f"[bold red]{sig.stop:,.2f}[/]  "
                               f"([red]−{sig.stop_pts():.2f} pts[/] │ −{ORB_STOP_SIG:.1f}σ)")
        t.add_row("EV:",      f"+0.61R ≈ +{0.61 * sig.risk_pts():.1f} pts  [{rem_str}]")
        return Panel(t, title="[bold green]⬤  ORB — LONG ▲  SIGNAL[/]",
                     border_style="green", padding=(0, 1))

    # Window status
    window = _orb_window(bar_et) if bar_et else None
    if orb.orb_complete:
        width = orb.orb_high - orb.orb_low
        if width < ORB_WIDTH_MIN:
            win_note = "[dim]ORB too narrow — no signal today[/]"
        elif window:
            fired = (window == "Morning" and orb.morning_fired) or \
                    (window == "Power hr" and orb.power_hr_fired)
            if fired:
                win_note = f"[dim]{window} signal already fired[/]"
            else:
                win_note = (f"[bold yellow]Watching — break above "
                            f"{orb.orb_high:.2f} fires LONG[/]")
        else:
            # Show next window
            next_wins = [f"{l} {sh:02d}:{sm:02d}–{eh:02d}:{em:02d}"
                         for sh, sm, eh, em, l in ORB_WINDOWS
                         if not ((l == "Morning" and orb.morning_fired) or
                                 (l == "Power hr" and orb.power_hr_fired))]
            if next_wins:
                win_note = f"[dim]Next window: {', '.join(next_wins)} ET[/]"
            else:
                win_note = "[dim]All windows done for today[/]"
        t.add_row("Status:", win_note)

    return Panel(t, title="[bold]ORB — LONG ONLY[/]",
                 border_style="yellow" if window and orb.orb_complete else "dim",
                 padding=(0, 1))


def build_history_table(history: list[RecentSignal]) -> Panel:
    t = Table(box=box.SIMPLE, padding=(0, 1), show_header=True, header_style="bold dim")
    t.add_column("Time",    width=8)
    t.add_column("Type",    width=5)
    t.add_column("Dir",     width=6)
    t.add_column("Entry",   width=10, justify="right")
    t.add_column("Target",  width=10, justify="right")
    t.add_column("Stop",    width=10, justify="right")
    t.add_column("Outcome", width=12)
    t.add_column("P&L",     width=10, justify="right")

    for rs in reversed(history[-10:]):
        local   = rs.fired_at.astimezone(datetime.now().astimezone().tzinfo)
        dir_str = "[green]LONG[/]"  if rs.direction == 1 else "[red]SHORT[/]"
        kind_str = f"[cyan]{rs.kind}[/]"

        if rs.outcome == "TARGET":
            out_str = "[green]HIT TARGET[/]"
            pnl_str = f"[green]+{rs.pnl_pts:.2f}[/]"
        elif rs.outcome == "STOPPED":
            out_str = "[red]STOPPED[/]"
            pnl_str = f"[red]−{abs(rs.pnl_pts):.2f}[/]"
        elif rs.outcome == "TIME EXIT":
            col  = "green" if rs.pnl_pts >= 0 else "red"
            sign = "+" if rs.pnl_pts >= 0 else "−"
            out_str = "[yellow]TIME EXIT[/]"
            pnl_str = f"[{col}]{sign}{abs(rs.pnl_pts):.2f}[/]"
        else:
            out_str = "[bold yellow]OPEN[/]"
            pnl_str = "[dim]—[/]"

        t.add_row(local.strftime("%H:%M"), kind_str, dir_str,
                  f"{rs.entry:,.2f}", f"{rs.target:,.2f}", f"{rs.stop:,.2f}",
                  out_str, pnl_str)

    return Panel(t, title="[bold]RECENT SIGNALS (MOM + ORB)[/]",
                 border_style="dim", padding=(0, 1))


def build_header(now: datetime, cname: str) -> Panel:
    local = now.astimezone(datetime.now().astimezone().tzinfo)
    t = Text(justify="center")
    t.append("  MES SIGNAL MONITOR  ", style="bold white on dark_blue")
    t.append(f"  {cname}  ", style="bold cyan")
    t.append("│  ")
    t.append(local.strftime("%a %Y-%m-%d  %H:%M:%S %Z"), style="dim")
    t.append("  │  next bar: ")
    nb = _next_bar_close(now).astimezone(datetime.now().astimezone().tzinfo)
    t.append(nb.strftime("%H:%M:%S"), style="yellow")
    return Panel(t, border_style="dark_blue", padding=(0, 0))


def render(state: InstrumentState, now: datetime) -> Table:
    root = Table.grid(padding=(0, 0))
    root.add_column()
    root.add_row(build_header(now, state.cname))

    if not state.bars:
        msg = f"[red]{state.error}[/]" if state.error else "[dim]Waiting for bars…[/]"
        root.add_row(Panel(msg, border_style="dim", padding=(0, 1)))
        return root
    if state.error:
        root.add_row(Panel(f"[yellow]⚠ {state.error} — showing last known data[/]",
                           border_style="yellow", padding=(0, 1)))

    info_row = Columns(
        [build_regime_panel(state), build_bar_panel(state)],
        equal=True, padding=(0, 1),
    )
    root.add_row(info_row)

    signals_row = Columns(
        [build_momentum_panel(state, now), build_orb_panel(state, now)],
        equal=True, padding=(0, 1),
    )
    root.add_row(signals_row)

    if state.history:
        root.add_row(build_history_table(state.history))

    return root


# ── Trade logging ──────────────────────────────────────────────────────────────

def _ensure_logs():
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDS).writeheader()
    if not ORB_LOG_PATH.exists():
        with open(ORB_LOG_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writeheader()


def _log_mom(sig: Signal, outcome: str, pnl_pts: float, resolved_at: datetime):
    row = {
        "fired_at": sig.bar_ts.isoformat(), "resolved_at": resolved_at.isoformat(),
        "symbol": "MES", "direction": "LONG" if sig.direction == 1 else "SHORT",
        "entry": round(sig.entry, 4), "target": round(sig.target, 4),
        "stop": round(sig.stop, 4), "sigma_pts": round(sig.sigma_pts, 4),
        "scaled": round(sig.scaled, 4), "vol_ratio": round(sig.vol_ratio, 4),
        "csr": round(sig.csr, 4), "outcome": outcome,
        "pnl_pts": round(pnl_pts, 4),
        "pnl_sigma": round(pnl_pts / sig.sigma_pts, 4) if sig.sigma_pts else 0.0,
    }
    with open(LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=LOG_FIELDS).writerow(row)


def _log_orb(sig: OrbSignal, outcome: str, pnl_pts: float, resolved_at: datetime):
    row = {
        "fired_at": sig.bar_ts.isoformat(), "resolved_at": resolved_at.isoformat(),
        "symbol": "MES", "direction": "LONG",
        "entry": round(sig.entry, 4), "target": round(sig.target, 4),
        "stop": round(sig.stop, 4),
        "orb_high": round(sig.orb_high, 4), "orb_low": round(sig.orb_low, 4),
        "orb_width": round(sig.orb_high - sig.orb_low, 4),
        "sigma_pts": round(sig.sigma_pts, 4), "window": sig.window,
        "outcome": outcome, "pnl_pts": round(pnl_pts, 4),
        "pnl_r": round(pnl_pts / sig.risk_pts(), 4) if sig.risk_pts() else 0.0,
    }
    with open(ORB_LOG_PATH, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=ORB_LOG_FIELDS).writerow(row)


def _check_mom_resolution(sig: Signal, bars: list[Bar]) -> tuple[str, float] | None:
    for bar in bars:
        if bar.ts <= sig.bar_ts:
            continue
        if sig.direction == 1:
            if bar.high >= sig.target:  return "TARGET",  sig.target_pts()
            if bar.low  <= sig.stop:    return "STOPPED", -sig.stop_pts()
        else:
            if bar.low  <= sig.target:  return "TARGET",  sig.target_pts()
            if bar.high >= sig.stop:    return "STOPPED", -sig.stop_pts()
    return None


# ── Live mode ──────────────────────────────────────────────────────────────────

def run_live():
    from topstep_client import TopstepClient

    client = TopstepClient()
    client.login()

    contracts = client.search_contracts(MES_CFG.search_term)
    if not contracts:
        console.print("[red]No MES contract found[/]")
        sys.exit(1)
    c = contracts[0]
    state = InstrumentState(cfg=MES_CFG, cid=c["id"], cname=c["name"])
    console.print(f"  MES: {c['name']}  id={c['id']}")

    _ensure_logs()

    def fetch_bars():
        end      = datetime.now(timezone.utc)
        lookback = TRAILING_BARS + 18 + 10   # extra for ORB bars
        start    = end - timedelta(minutes=TF_MINUTES * lookback)
        try:
            raw = client.get_bars(contract_id=state.cid, start=start, end=end,
                                  unit=TopstepClient.MINUTE, unit_number=TF_MINUTES,
                                  limit=lookback)
            raw = list(reversed(raw))
            state.bars = [Bar(ts=datetime.fromisoformat(b["t"]),
                              open=b["o"], high=b["h"], low=b["l"],
                              close=b["c"], volume=b["v"]) for b in raw]
            state.error = None
        except Exception as e:
            state.error = f"fetch error: {e}"

    def resolve_mom(outcome: str, pnl_pts: float, now: datetime):
        sig = state.active_signal
        state.history.append(RecentSignal(
            "MES", "MOM", sig.direction, sig.entry, sig.target, sig.stop,
            outcome, pnl_pts, sig.bar_ts,
        ))
        _log_mom(sig, outcome, pnl_pts, now)
        state.active_signal = None

    def resolve_orb(outcome: str, pnl_pts: float, now: datetime):
        sig = state.orb.active_signal
        state.history.append(RecentSignal(
            "MES", "ORB", 1, sig.entry, sig.target, sig.stop,
            outcome, pnl_pts, sig.bar_ts,
        ))
        _log_orb(sig, outcome, pnl_pts, now)
        state.orb.active_signal = None

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            now = datetime.now(timezone.utc)
            fetch_bars()

            if state.bars:
                last_bar_ts = state.bars[-1].ts
                new_bar = (last_bar_ts != state.last_evaluated_ts)

                # Always update vol/sigma metrics
                new_mom = evaluate(state)
                new_orb = evaluate_orb(state)

                if new_bar:
                    state.last_evaluated_ts = last_bar_ts

                    # Momentum signal
                    if state.active_signal:
                        hit = _check_mom_resolution(state.active_signal, state.bars)
                        if hit:
                            resolve_mom(hit[0], hit[1], now)
                        elif now >= state.active_signal.expires_at:
                            pnl = (state.bars[-1].close - state.active_signal.entry) \
                                  * state.active_signal.direction
                            resolve_mom("TIME EXIT", pnl, now)

                    if new_mom and state.active_signal is None:
                        state.active_signal = new_mom
                        play_alert()

                    # ORB signal
                    if state.orb.active_signal:
                        hit = _check_orb_resolution(state.orb.active_signal, state.bars)
                        if hit:
                            resolve_orb(hit[0], hit[1], now)
                        elif now >= state.orb.active_signal.bar_ts + timedelta(minutes=MAX_HOLD_MIN):
                            pnl = state.bars[-1].close - state.orb.active_signal.entry
                            resolve_orb("TIME EXIT", pnl, now)

                    if new_orb and state.orb.active_signal is None:
                        state.orb.active_signal = new_orb
                        play_alert()

            live.update(render(state, now))
            time.sleep(30)


# ── Demo mode ──────────────────────────────────────────────────────────────────

def run_demo():
    now       = datetime.now(timezone.utc)
    now_et    = now.astimezone(ET)
    sigma     = 0.000721
    price     = 5_680.0
    sigma_pts = sigma * price   # ≈ 4.1 pts

    state = InstrumentState(cfg=MES_CFG, cname="MESH6")
    state.sigma      = sigma
    state.sigma_pts  = sigma_pts
    state.gk_ann_vol = 0.198
    state.mean_vol   = 8_450.0
    state.bars = [Bar(ts=now - timedelta(minutes=5),
                      open=5_675.0, high=5_682.0, low=5_673.0,
                      close=price, volume=9_200)]

    # Simulate a completed wide ORB with active signal
    state.orb.session_date   = now_et.date()
    state.orb.orb_high       = 5_672.5
    state.orb.orb_low        = 5_648.0   # width = 24.5 pts (> 15.25)
    state.orb.orb_complete   = True
    state.orb.morning_fired  = True
    state.orb.active_signal  = OrbSignal(
        entry=5_680.0, target=5_680.0 + 2 * sigma_pts,
        stop=5_680.0 - 2 * sigma_pts,
        orb_high=5_672.5, orb_low=5_648.0,
        sigma_pts=sigma_pts, window="Morning",
        bar_ts=now - timedelta(minutes=10),
    )

    state.history = [
        RecentSignal("MES", "MOM",  -1, 5_710.0, 5_697.7, 5_716.3, "TARGET",    12.3,
                     now - timedelta(minutes=90)),
        RecentSignal("MES", "ORB",  +1, 5_688.0, 5_696.2, 5_679.8, "STOPPED",  -8.2,
                     now - timedelta(minutes=200)),
        RecentSignal("MES", "MOM",  +1, 5_650.0, 5_662.3, 5_641.8, "TIME EXIT",  4.5,
                     now - timedelta(minutes=310)),
    ]

    console.print()
    console.print("[bold underline]DEMO — ORB signal active, momentum watching[/]",
                  justify="center")
    console.print(render(state, now))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true",
                        help="Show sample output without API calls")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    else:
        run_live()
