"""
ml_monitor.py — Rich live monitor for ml_trading_bot.

Reads logs/ml_state.json (written every poll cycle by ml_trading_bot)
and logs/ml_trades.csv for trade history.

Panels:
  Header      — time, MES price, vol regime gauge, session mode
  Model       — probability bar, signal, gate status, threshold vs actual
  Features    — r2m sparkline, r1m sparkline, ret_open, realized_vol, ATR ratio
  Position    — entry, current P&L estimate, bars held, target/stop levels
  Signals     — last 20 signal evaluations (fired + skipped) with outcomes
  Trades      — today's completed trades with P&L
  Stats       — session performance + model training info

Usage:
    python src/ml_monitor.py
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ET    = ZoneInfo("America/New_York")
MST   = ZoneInfo("America/Phoenix")

LOOKBACK = 15   # 2-min lagged returns shown in sparkline
SHORT_N  = 6    # 1-min lagged returns shown in sparkline

STATE_FILES = {
    "MES": Path("logs/ml_state_MES.json"),
    "MNQ": Path("logs/ml_state_MNQ.json"),
}
TRADE_LOGS = {
    "MES": Path("logs/ml_trades_MES.csv"),
    "MNQ": Path("logs/ml_trades_MNQ.csv"),
}

REFRESH_S  = 2


# ── State / trade loading ──────────────────────────────────────────────────────

def load_state(symbol: str = "MES") -> dict | None:
    try:
        return json.loads(STATE_FILES[symbol].read_text())
    except Exception:
        return None


def load_all_states() -> dict[str, dict]:
    """Return all available states keyed by symbol."""
    out = {}
    for sym, path in STATE_FILES.items():
        try:
            out[sym] = json.loads(path.read_text())
        except Exception:
            pass
    return out


def load_trades(symbol: str = "MES") -> list[dict]:
    path = TRADE_LOGS.get(symbol)
    if not path or not path.exists():
        return []
    try:
        today = datetime.now(ET).date().isoformat()
        rows  = []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("fired_at", "")[:10] == today:
                    rows.append(r)
        return rows
    except Exception:
        return []


def load_all_trades() -> list[dict]:
    """Combined trades from all symbols, sorted newest first."""
    all_rows = []
    for sym in TRADE_LOGS:
        for r in load_trades(sym):
            r["_sym"] = sym
            all_rows.append(r)
    all_rows.sort(key=lambda r: r.get("fired_at", ""), reverse=True)
    return all_rows


# ── Sparkline / visual helpers ─────────────────────────────────────────────────

SPARK_UP   = "▲"
SPARK_DOWN = "▼"
SPARK_FLAT = "─"

def sparkline(values: list[float], width: int | None = None) -> Text:
    """
    Render a list of floats as a text sparkline.
    Positive values are green ▲, negative red ▼, near-zero dim ─.
    Most recent value is leftmost (lag 1 first).
    """
    t = Text()
    vals = values if width is None else values[:width]
    for v in vals:
        if abs(v) < 1e-7:
            t.append(SPARK_FLAT, style="dim")
        elif v > 0:
            t.append(SPARK_UP, style="green")
        else:
            t.append(SPARK_DOWN, style="red")
        t.append(" ")
    return t


def prob_bar(prob: float, threshold: float, width: int = 30) -> Text:
    """Horizontal probability bar with threshold marker."""
    filled = int(prob * width)
    thresh = int(threshold * width)
    t = Text()
    for i in range(width):
        if i == thresh:
            t.append("|", style="yellow")
        elif i < filled:
            color = "green" if prob >= threshold else "red"
            t.append("█", style=color)
        else:
            t.append("░", style="dim")
    t.append(f" {prob:.3f}", style="bold" if prob >= threshold else "")
    return t


def regime_bar(regime: float, width: int = 20) -> Text:
    """Vol regime percentile bar."""
    filled = int(regime * width)
    t = Text()
    for i in range(width):
        if i < filled:
            if regime >= 0.67:
                t.append("█", style="red")
            elif regime >= 0.33:
                t.append("█", style="yellow")
            else:
                t.append("█", style="green")
        else:
            t.append("░", style="dim")
    label = "HIGH" if regime >= 0.67 else ("MED" if regime >= 0.33 else "LOW")
    color = "red" if regime >= 0.67 else ("yellow" if regime >= 0.33 else "green")
    t.append(f" [{label}] {regime:.2f}", style=color)
    return t


def fmt_pts(v: float) -> Text:
    if v >= 0:
        return Text(f"+{v:.2f}", style="green")
    return Text(f"−{abs(v):.2f}", style="red")


def fmt_ret(v: float) -> Text:
    """Format a log return as basis points."""
    bp = v * 10000
    if bp >= 0:
        return Text(f"+{bp:.1f}bp", style="green")
    return Text(f"−{abs(bp):.1f}bp", style="red")


def gate_icon(ok: bool) -> str:
    return "[green]✓[/]" if ok else "[red]✗[/]"


# ── Panel builders ─────────────────────────────────────────────────────────────

def build_header(state: dict) -> Panel:
    feat = state.get("feat") or {}
    now_et  = datetime.now(ET)
    now_mst = datetime.now(MST)

    close = feat.get("close", 0)
    bar_ts_raw = state.get("bar_ts") or ""
    bar_ts = bar_ts_raw[11:16] if len(bar_ts_raw) > 16 else "—"

    paper = state.get("paper", False)
    mode  = Text("[PAPER]", style="yellow bold") if paper else Text("[LIVE]", style="green bold")

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(width=20)
    t.add_column(width=20)
    t.add_column(width=20)
    t.add_column(width=20)
    t.add_row(
        f"[bold]{now_et.strftime('%H:%M:%S')} ET[/]",
        f"MES  [bold]{close:.2f}[/]  (bar {bar_ts})",
        Text.assemble("Vol Regime  ", regime_bar(feat.get("vol_regime", 0))),
        mode,
    )
    return Panel(t, title="ML Trading Bot", border_style="blue", expand=True)


def build_model_panel(state: dict) -> Panel:
    feat   = state.get("feat") or {}
    signal = state.get("signal") or {}
    thr    = state.get("thresholds") or {}

    prob       = signal.get("prob", 0.0)
    prob_thr   = thr.get("prob", 0.65)
    vol_thr    = thr.get("vol_regime", 0.50)
    vol_regime = feat.get("vol_regime", 0.0)
    direction  = signal.get("direction")
    reason     = signal.get("reason", "—")

    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(width=18, no_wrap=True)
    t.add_column()

    t.add_row("Probability",
              Text.assemble(prob_bar(prob, prob_thr), f"  (threshold {prob_thr:.2f})"))
    t.add_row("Vol Regime",
              Text.assemble(regime_bar(vol_regime), f"  (min {vol_thr:.2f})"))

    # Gate status
    gates = Text()
    gates.append(f"prob {gate_icon(signal.get('prob_ok', False))}  ", style="")
    gates.append(f"vol  {gate_icon(signal.get('vol_ok',  False))}  ", style="")
    gates.append(f"dir  {gate_icon(signal.get('dir_ok',  False))}", style="")
    t.add_row("Gates", gates)

    # Direction
    if direction == "LONG":
        dir_txt = Text("▲ LONG", style="green bold")
    elif direction == "SHORT":
        dir_txt = Text("▼ SHORT", style="red bold")
    else:
        dir_txt = Text("— no direction", style="dim")
    t.add_row("Direction", dir_txt)
    t.add_row("Reason", Text(reason, style="dim"))

    # Signal fired indicator
    if signal.get("gates_pass"):
        sig_txt = Text(f"◉ SIGNAL: {direction}", style="bold green" if direction == "LONG" else "bold red")
    else:
        sig_txt = Text("○ no signal", style="dim")
    t.add_row("Signal", sig_txt)

    return Panel(t, title="Model", border_style="blue", expand=False)


def build_features_panel(state: dict) -> Panel:
    feat = state.get("feat") or {}

    r2m = feat.get("r2m_series", [])
    r1m = feat.get("r1m_series", [])

    ret_open    = feat.get("ret_open", 0)
    realized_vol = feat.get("realized_vol", 0)
    atr_ratio   = feat.get("atr_ratio", 0)
    r2m_1       = feat.get("r2m_1", 0)
    r1m_1       = feat.get("r1m_1", 0)

    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(width=16, no_wrap=True)
    t.add_column()

    t.add_row("2-min returns",
              Text.assemble(sparkline(r2m), f" lag1→{LOOKBACK-1}"))
    t.add_row("1-min returns",
              Text.assemble(sparkline(r1m), f" lag1→{SHORT_N}"))
    t.add_row("r2m_1 (last)",  fmt_ret(r2m_1))
    t.add_row("r1m_1 (last)",  fmt_ret(r1m_1))
    t.add_row("ret_open",      fmt_ret(ret_open))

    vol_color = "red" if realized_vol > 0.0005 else ("yellow" if realized_vol > 0.0003 else "green")
    t.add_row("realized_vol",
              Text(f"{realized_vol*100:.4f}%", style=vol_color))
    atr_color = "red" if atr_ratio > 1.5 else ("yellow" if atr_ratio > 1.0 else "green")
    t.add_row("ATR ratio",
              Text(f"{atr_ratio:.2f}×", style=atr_color))

    return Panel(t, title="Features", border_style="blue", expand=False)


def build_position_panel(state: dict) -> Panel:
    pos  = state.get("position")
    feat = state.get("feat") or {}
    thr  = state.get("thresholds") or {}

    if pos is None:
        return Panel(Text("  No position", style="dim"),
                     title="Position", border_style="blue", expand=False)

    direction  = pos["direction"]
    entry      = pos["entry_pts"]
    bars_held  = pos["bars_held"]
    prob       = pos["prob"]
    vol_regime = pos["vol_regime"]
    current    = feat.get("close", entry)
    tgt_pts    = thr.get("target_pts", 5.0)
    stp_pts    = thr.get("stop_pts",   3.0)

    is_long = direction == "LONG"
    pnl     = (current - entry) if is_long else (entry - current)
    target  = entry + (tgt_pts if is_long else -tgt_pts)
    stop    = entry - (stp_pts if is_long else -stp_pts)

    dir_txt = Text("▲ LONG", style="green bold") if is_long else Text("▼ SHORT", style="red bold")

    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column(width=14, no_wrap=True)
    t.add_column()
    t.add_row("Direction",  dir_txt)
    t.add_row("Entry",      f"{entry:.2f}")
    t.add_row("Current",    f"{current:.2f}")
    t.add_row("P&L (est.)", fmt_pts(pnl))
    t.add_row("Target",     f"{target:.2f}  (+{tgt_pts:.1f}pts)")
    t.add_row("Stop",       f"{stop:.2f}  (-{stp_pts:.1f}pts)")
    t.add_row("Bars held",  f"{bars_held}  ({bars_held*2} min)")
    t.add_row("Entry prob", f"{prob:.3f}")
    t.add_row("Entry vReg", f"{vol_regime:.2f}")

    return Panel(t, title="Position", border_style="green" if pnl >= 0 else "red",
                 expand=False)


def build_signals_history_panel() -> Panel:
    trades = load_all_trades()

    t = Table(box=None, show_header=True, padding=(0, 1))
    t.add_column("Sym",     width=4)
    t.add_column("Time",    width=5)
    t.add_column("Dir",     width=7)
    t.add_column("Entry",   width=9, justify="right")
    t.add_column("Prob",    width=5, justify="right")
    t.add_column("vReg",    width=5, justify="right")
    t.add_column("Outcome", width=10)
    t.add_column("P&L",     width=8, justify="right")
    t.add_column("Bars",    width=4, justify="right")

    for r in trades[:20]:
        ts    = r.get("fired_at", "")
        bar_t = ts[11:16] if len(ts) > 16 else "—"
        d     = r.get("direction", "")
        dir_t = Text("▲ Long", style="green") if d == "LONG" else Text("▼ Short", style="red")
        out   = r.get("outcome", "")
        if out == "TARGET":
            out_t = Text("TARGET",  style="green")
        elif out == "STOPPED":
            out_t = Text("STOPPED", style="red")
        else:
            out_t = Text(out[:10], style="dim")
        pnl = float(r.get("pnl_pts", 0) or 0)
        t.add_row(
            r.get("symbol", r.get("_sym", "—")),
            bar_t,
            dir_t,
            r.get("entry", "—"),
            f"{float(r.get('prob',0)):.3f}",
            f"{float(r.get('vol_regime',0)):.2f}",
            out_t,
            fmt_pts(pnl),
            r.get("bars_held", "—"),
        )

    if not trades:
        t.add_row("—", "—", "—", "—", "—", "—", Text("no trades today", style="dim"), "—", "—")

    return Panel(t, title="Trade History (today — all symbols)", border_style="blue", expand=True)


def build_stats_panel(state: dict) -> Panel:
    stats = state.get("session_stats") or {}
    minfo = state.get("model_info")   or {}
    thr   = state.get("thresholds")   or {}

    trades = stats.get("trades", 0)
    wins   = stats.get("wins",   0)
    losses = stats.get("losses", 0)
    pnl    = stats.get("pnl_pts", 0.0)
    sigs   = stats.get("signals", 0)
    bars   = stats.get("bars_seen", 0)

    wr = f"{wins/trades*100:.0f}%" if trades > 0 else "—"

    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(width=22)
    t.add_column(width=22)
    t.add_column(width=30)

    t.add_row(
        f"Signals: {sigs}   Trades: {trades}",
        f"W:{wins} L:{losses}  WR:{wr}",
        Text.assemble("P&L: ", fmt_pts(pnl), " pts"),
    )
    t.add_row(
        f"Bars seen: {bars}  (~{bars*2} min)",
        f"Thr: prob≥{thr.get('prob',0.65):.2f} vReg≥{thr.get('vol_regime',0.50):.2f}",
        f"Tgt:{thr.get('target_pts',5.0):.1f}pt  Stp:{thr.get('stop_pts',3.0):.1f}pt",
    )

    # Model info row
    n_train = minfo.get("n_train", 0)
    auc     = minfo.get("auc", 0)
    d_from  = minfo.get("date_range_from", "")
    d_to    = minfo.get("date_range_to", "")
    cutoff  = minfo.get("train_cutoff", "")
    base    = minfo.get("base_rate", 0)

    sym    = minfo.get("symbol", "?")
    x_bps  = minfo.get("threshold_bps", 9.0)
    h_bars = minfo.get("horizon_bars", 1)
    t.add_row(
        f"{sym}  lb=15  X={x_bps:.0f}bp  H={h_bars}",
        f"Train: {n_train:,} bars  AUC={auc:.4f}",
        f"Data: {d_from} → {d_to}  base={base*100:.1f}%",
    )

    return Panel(t, title="Session Stats  ·  Model Info", border_style="blue", expand=True)


def build_no_state_panel() -> Panel:
    return Panel(
        Text("\n  Waiting for ml_trading_bot to start …\n"
             "  (run: python src/ml_trading_bot.py --paper)\n", style="dim"),
        title="ML Monitor", border_style="dim",
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def build_symbol_block(state: dict) -> Group:
    """Model + features + position panels for one symbol."""
    model_pnl = build_model_panel(state)
    feat_pnl  = build_features_panel(state)
    pos_pnl   = build_position_panel(state)
    stats_pnl = build_stats_panel(state)
    top_row   = Columns([Group(model_pnl, feat_pnl), pos_pnl], expand=True)
    return Group(top_row, stats_pnl)


def run():
    console = Console()

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            try:
                states = load_all_states()

                if not states:
                    live.update(build_no_state_panel())
                    time.sleep(REFRESH_S)
                    continue

                # Use first available state for the shared header
                primary_state = next(iter(states.values()))
                header  = build_header(primary_state)
                hist    = build_signals_history_panel()

                # One block per active symbol
                sym_blocks = []
                for sym in ["MES", "MNQ"]:
                    if sym in states:
                        sym_blocks.append(build_symbol_block(states[sym]))

                if len(sym_blocks) == 1:
                    mid = sym_blocks[0]
                else:
                    mid = Columns(sym_blocks, expand=True)

                live.update(Group(header, mid, hist))

            except Exception as e:
                live.update(Panel(f"Monitor error: {e}", border_style="red"))

            time.sleep(REFRESH_S)


if __name__ == "__main__":
    run()
