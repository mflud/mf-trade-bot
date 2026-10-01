"""
backtest_orb_cls_cross_confirm.py — Does MES/MNQ cross-confirmation help
ORB-cls, the same way it helps the morning ORB (see
backtest_orb_cross_confirm.py and project memory project_orb_cross_confirm)?

Mechanics mirror evaluate_orb()'s live cross-confirmation path in
trading_bot.py exactly:
  - Each symbol's "candidate" = the first 1-min bar in the entry window
    whose CLOSE breaks its own 15:50 ET range (direction='first').
  - If both symbols' candidates agree in direction, BOTH fire — priced off
    the close of the bar at the LATER of the two candidate timestamps (not
    the original breakout price), matching the live re-pricing behavior.
  - If a symbol never gets a candidate, or the two disagree, neither fires.

Compares combined (and per-symbol) WR/PF/total against the already-measured
direction='first' (no cross-confirm) baseline from backtest_orb_cls.py's
sweep, using the parameter region that backtest identified as robust:
  entry_type=close, stop_type=midpoint, target in {3.0, 3.25}, entry_win in
  {3, 5}, hold=15 (uncapped).

Usage:
  python src/backtest_orb_cls_cross_confirm.py
  python src/backtest_orb_cls_cross_confirm.py --walkforward
"""

import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from backtest_orb_cls import (  # noqa: E402
    fetch_bars, build_sessions, evaluate_combo, _sim_exit,
    POINT_VALUE, COMMISSION,
)

CANDIDATES = dict(
    target_mults = [3.0, 3.25],
    entry_wins   = [3, 5],
    hold         = 15,
    stop_type    = "midpoint",
    cap_1600     = False,
)


# ── Candidate detection (mirrors evaluate_orb's bar_direction logic) ───────────

def _find_candidate(sess: dict, entry_win: int):
    orb_h, orb_l = sess["range_high"], sess["range_low"]
    for bar in sess["bars"]:
        elapsed = (bar["ts"] - sess["range_ts"]).total_seconds() / 60.0
        if elapsed > entry_win:
            break
        if bar["c"] > orb_h:
            return 1, bar["ts"]
        if bar["c"] < orb_l:
            return -1, bar["ts"]
    return 0, None


def sim_cross_day(mes_sess: dict, mnq_sess: dict, target_mult: float,
                  entry_win: int, hold, stop_type: str, cap_1600: bool) -> list[dict]:
    mes_dir, mes_ts = _find_candidate(mes_sess, entry_win)
    mnq_dir, mnq_ts = _find_candidate(mnq_sess, entry_win)
    if mes_dir == 0 or mnq_dir == 0 or mes_dir != mnq_dir:
        return []

    confirm_ts = max(mes_ts, mnq_ts)
    trades = []
    for symbol, sess, direction in (("MES", mes_sess, mes_dir), ("MNQ", mnq_sess, mnq_dir)):
        entry_bar = next((b for b in sess["bars"] if b["ts"] >= confirm_ts), None)
        if entry_bar is None:
            continue
        orb_h, orb_l = sess["range_high"], sess["range_low"]
        width = orb_h - orb_l
        mid   = (orb_h + orb_l) / 2
        target_pts = width * target_mult
        if stop_type == "midpoint":
            stop_pts = (orb_h - mid) if direction == 1 else (mid - orb_l)
        else:
            stop_pts = width

        entry = entry_bar["c"]
        tgt   = entry + direction * target_pts
        stp   = entry - direction * stop_pts
        future = [b for b in sess["bars"] if b["ts"] > entry_bar["ts"]]
        exit_p, result = _sim_exit(entry, tgt, stp, direction, entry_bar, future, hold, cap_1600)
        pnl = (exit_p - entry) * direction * POINT_VALUE[symbol] - COMMISSION
        trades.append(dict(symbol=symbol, date=sess["date"], dir=direction,
                           entry=entry, exit=exit_p, result=result, pnl=pnl))
    return trades


def run_cross_confirm(mes_sessions: list[dict], mnq_sessions: list[dict],
                      target_mult: float, entry_win: int, hold, stop_type: str,
                      cap_1600: bool) -> dict:
    mnq_by_date = {s["date"]: s for s in mnq_sessions}
    all_trades = []
    for mes_sess in mes_sessions:
        mnq_sess = mnq_by_date.get(mes_sess["date"])
        if mnq_sess is None:
            continue
        all_trades.extend(sim_cross_day(mes_sess, mnq_sess, target_mult, entry_win,
                                        hold, stop_type, cap_1600))
    return _summarize(all_trades)


def _summarize(trades: list[dict]) -> dict:
    n_days = len(set((t["date"], ) for t in trades)) if trades else 0
    pnls   = [t["pnl"] for t in trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) if pnls else 0.0
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else (999 if wins else 0)
    return dict(trades=len(pnls), confirmed_days=n_days, wr=round(wr, 3), pf=round(pf, 3),
               total=round(sum(pnls), 0))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    walkforward = "--walkforward" in sys.argv

    mes_df = fetch_bars("MES")
    mnq_df = fetch_bars("MNQ")
    mes_sessions = sorted(build_sessions(mes_df), key=lambda s: s["date"])
    mnq_sessions = sorted(build_sessions(mnq_df), key=lambda s: s["date"])
    print(f"MES sessions: {len(mes_sessions)}  MNQ sessions: {len(mnq_sessions)}", flush=True)

    splits = [("full-sample", mes_sessions, mnq_sessions)]
    if walkforward:
        cut_m = int(len(mes_sessions) * 0.7)
        cut_q = int(len(mnq_sessions) * 0.7)
        splits = [
            ("train", mes_sessions[:cut_m], mnq_sessions[:cut_q]),
            ("test",  mes_sessions[cut_m:], mnq_sessions[cut_q:]),
        ]

    combos = list(product(CANDIDATES["target_mults"], CANDIDATES["entry_wins"]))

    for split_name, mes_s, mnq_s in splits:
        print(f"\n{'='*100}\n{split_name.upper()}  "
              f"({len(mes_s)} MES / {len(mnq_s)} MNQ sessions)\n{'='*100}")

        rows = []
        for tm, ew in combos:
            cross = run_cross_confirm(mes_s, mnq_s, tm, ew, CANDIDATES["hold"],
                                      CANDIDATES["stop_type"], CANDIDATES["cap_1600"])

            # Baseline: each symbol's own direction='first' performance, no
            # cross-confirmation gate (what backtest_orb_cls.py already measured).
            mes_base = evaluate_combo(mes_s, "MES", "close", tm, ew, "first",
                                      CANDIDATES["hold"], CANDIDATES["stop_type"],
                                      CANDIDATES["cap_1600"]) or dict(trades=0, wr=0, pf=0, total=0)
            mnq_base = evaluate_combo(mnq_s, "MNQ", "close", tm, ew, "first",
                                      CANDIDATES["hold"], CANDIDATES["stop_type"],
                                      CANDIDATES["cap_1600"]) or dict(trades=0, wr=0, pf=0, total=0)
            base_total  = mes_base["total"] + mnq_base["total"]
            base_trades = mes_base["trades"] + mnq_base["trades"]
            base_wins   = round(mes_base["wr"] * mes_base["trades"]) + round(mnq_base["wr"] * mnq_base["trades"])
            base_wr     = base_wins / base_trades if base_trades else 0.0
            wins_sum   = (mes_base.get("avg_win", 0) * round(mes_base["wr"] * mes_base["trades"])
                         + mnq_base.get("avg_win", 0) * round(mnq_base["wr"] * mnq_base["trades"]))
            losses_sum = (mes_base.get("avg_loss", 0) * round((1 - mes_base["wr"]) * mes_base["trades"])
                         + mnq_base.get("avg_loss", 0) * round((1 - mnq_base["wr"]) * mnq_base["trades"]))
            base_pf = (wins_sum / abs(losses_sum)) if losses_sum else (999 if wins_sum else 0)

            rows.append(dict(
                target=tm, entry_win=ew,
                confirm_days=cross["confirmed_days"], confirm_trades=cross["trades"],
                confirm_wr=cross["wr"], confirm_pf=cross["pf"], confirm_total=cross["total"],
                base_trades=base_trades, base_wr=round(base_wr, 3), base_pf=round(base_pf, 3),
                base_total=base_total,
            ))

        out = pd.DataFrame(rows)
        print(out.to_string(index=False))

    print(f"\nconfirm_* = both legs fire only when MES & MNQ closing-range breakouts "
          f"agree in direction (entry re-priced at confirmation bar).")
    print(f"base_* = each symbol trading its own direction='first' breakout "
          f"independently, no cross-confirmation gate (sum of both symbols).")


if __name__ == "__main__":
    main()
