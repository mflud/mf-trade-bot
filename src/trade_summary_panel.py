"""
trade_summary_panel.py — Shared "Trade Summary" panel for slr_monitor and signal_monitor.

Loads all trade logs (CSR, ORB, VWASLR, SLR, PL MOM) and renders the last N trades
newest-first.  No session filter — shows all trades.
"""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ET  = ZoneInfo("America/New_York")
MST = ZoneInfo("America/Phoenix")

TRADE_LOGS = {
    "CSR":    Path("logs/bot_trades.csv"),
    "ORB":    Path("logs/orb_trades.csv"),
    "VWASLR": Path("logs/vwaslr_trades.csv"),
    "SLR":    Path("logs/slr_trades.csv"),
    "PL MOM": Path("logs/pl_mom_trades.csv"),
}

STRAT_STYLE = {
    "CSR":    "cyan",
    "ORB":    "yellow",
    "VWASLR": "blue",
    "SLR":    "blue",
    "PL MOM": "magenta",
}

MES_MNQ = {"MES", "MNQ"}


def _load_all() -> list[dict]:
    now_et  = datetime.now(ET)
    today   = now_et.date()
    cutoff  = datetime(today.year, today.month, today.day, 8, 30,
                       tzinfo=ET).astimezone(timezone.utc)
    rows = []

    for strat, path in TRADE_LOGS.items():
        if not path.exists():
            continue
        try:
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        sym = r.get("symbol", "")
                        if sym not in MES_MNQ:
                            continue
                        fired = datetime.fromisoformat(r["fired_at"])
                        if fired.tzinfo is None:
                            fired = fired.replace(tzinfo=timezone.utc)
                        if fired < cutoff:
                            continue

                        pnl = float(r.get("pnl_pts", 0) or 0)

                        # Detail string per strategy
                        if strat == "SLR":
                            detail = f"{float(r.get('vol_ratio', 0)):.1f}× vol"
                        elif strat == "PL MOM":
                            pl  = float(r.get("pl",       0) or 0)
                            mbp = float(r.get("move_bps", 0) or 0)
                            detail = f"PL={pl:.2f} {mbp:.0f}bp"
                        elif strat == "CSR":
                            sc = float(r.get("scaled", 0) or 0)
                            detail = f"{sc:.1f}σ"
                        elif strat == "VWASLR":
                            vw = float(r.get("vwaslr", 0) or 0)
                            detail = f"vw={vw:.2f}"
                        elif strat == "ORB":
                            detail = "ORB"
                        else:
                            detail = ""

                        rows.append({
                            "strat":    strat,
                            "sym":      sym,
                            "fired_at": fired,
                            "direction": r.get("direction", "LONG"),
                            "entry":    r.get("fill_price") or r.get("est_entry", "—"),
                            "detail":   detail,
                            "outcome":  r.get("outcome", ""),
                            "pnl_pts":  pnl,
                        })
                    except Exception:
                        continue
        except Exception:
            continue

    rows.sort(key=lambda r: r["fired_at"], reverse=True)
    return rows


def build_trade_summary_panel(max_rows: int = 20) -> Panel:
    rows = _load_all()

    tbl = Table(box=None, show_header=True, padding=(0, 1))
    tbl.add_column("Sym",        width=4)
    tbl.add_column("Time",       width=5)
    tbl.add_column("Strat",      width=6)
    tbl.add_column("Dir",        width=7)
    tbl.add_column("Entry",      justify="right", width=9)
    tbl.add_column("Detail",     width=14)
    tbl.add_column("Outcome",    width=10)
    tbl.add_column("P&L pts",    justify="right", width=8)

    counts: dict[str, int] = {}
    pnl_by_sym: dict[str, float] = {}
    shown = 0

    for row in rows[:max_rows]:
        try:
            fired_et = row["fired_at"].astimezone(MST)
            pnl      = row["pnl_pts"]
            sym      = row["sym"]
            pnl_by_sym[sym] = pnl_by_sym.get(sym, 0.0) + pnl
            pnl_col  = "green" if pnl >= 0 else "red"
            pnl_sign = "+" if pnl >= 0 else "−"
            outcome  = row["outcome"]
            strat    = row["strat"]
            counts[strat] = counts.get(strat, 0) + 1

            if outcome in ("TARGET", "HIT TARGET"):
                out_txt = Text("TARGET",   style="green")
            elif "PL EXIT" in outcome or "PL_EXIT" in outcome:
                out_txt = Text("PL EXIT",  style="green")
            elif outcome == "STOPPED":
                out_txt = Text("STOPPED",  style="red")
            elif "TRAIL" in outcome:
                out_txt = Text("TRAIL STP", style="yellow")
            elif "TIME" in outcome:
                out_txt = Text("TIME EXIT", style="cyan")
            else:
                out_txt = Text(outcome[:10], style="dim")

            is_long = row["direction"] != "SHORT"
            dir_txt = Text("▲ Long", style="green") if is_long else Text("▼ Short", style="red")

            strat_style = STRAT_STYLE.get(strat, "white")
            tbl.add_row(
                row["sym"],
                fired_et.strftime("%H:%M"),
                f"[{strat_style}]{strat}[/]",
                dir_txt,
                str(row["entry"]),
                f"[dim]{row['detail']}[/]",
                out_txt,
                Text(f"{pnl_sign}{abs(pnl):.2f}", style=pnl_col),
            )
            shown += 1
        except Exception:
            continue

    # Summary footer
    from rich.console import Group as RichGroup
    from rich.text import Text as RichText

    count_str = "  ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
    sym_totals = "  ".join(
        f"{sym}: [{('green' if v >= 0 else 'red')}]{('+' if v >= 0 else '−')}{abs(v):.2f}[/]"
        for sym, v in sorted(pnl_by_sym.items())
    )
    blank = RichText("")
    note  = RichText.from_markup(
        f"  {count_str}    {sym_totals}  [dim](since 08:30 ET)[/]"
    )

    return Panel(
        RichGroup(tbl, blank, note),
        title="Trade Summary  (since 08:30 ET)",
        border_style="blue",
        expand=False,
    )
