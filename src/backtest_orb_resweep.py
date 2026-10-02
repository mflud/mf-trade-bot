"""
backtest_orb_resweep.py — Re-sweep ORB (morning) and ORB-cls (closing) for
MES and MNQ with CORRECTED entry pricing (entry = bar.close, matching
trading_bot.py's evaluate_orb/evaluate_orb_cls exactly) — not entry = the
ORB level itself, which is what backtest_mes_1min_orb.py / backtest_mnq_
1min_orb.py / the original backtest_orb_cls.py all assumed. That bug made
every one of those backtests look far better than the live code actually
performs (see project_orb_cls_strategy / project_orb_cross_confirm memory,
2026-10-02 finding).

Sweeps width_min (skip trades on narrow ranges) crossed with a stop-sizing
lever:
  - MES (both strategies): midpoint stop with a minimum-distance FLOOR
    (stop = max(half-width, floor_pts)) — tests whether widening the stop
    on narrow-range days helps, per user hypothesis.
  - MNQ morning ORB: far-side stop with a dollar CAP (current production:
    $500) — tests whether a SMALLER cap (tighter stop) helps, per user's
    opposite hypothesis for MNQ.
  - MNQ ORB-cls: currently plain midpoint (same as MES) — tested the same
    midpoint+floor way as MES for consistency, since that's what's live.

target_mult, entry_win and hold_min are held at their currently-shipped
values throughout (not re-swept here) to keep this a focused 2D sweep per
combo rather than a wide, overfit-prone grid search. A note at the end
flags whether revisiting target_mult on top of the winning width/stop
looks worth a follow-up pass.

Usage:
  python src/backtest_orb_resweep.py
  python src/backtest_orb_resweep.py --walkforward
"""

import sqlite3
import sys
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}
COMMISSION  = 4.0
MAX_FUTURE_MIN = 260


def fetch_bars(symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol=? AND minutes=1 ORDER BY ts", (symbol,)
    ).fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)   # CME settlement gap
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    df["et"] = df["ts"].dt.tz_convert(ET)
    df["hm"] = df["et"].dt.hour * 60 + df["et"].dt.minute
    df["date"] = df["et"].dt.date
    return df


def build_sessions(df: pd.DataFrame, range_hm: int) -> list[dict]:
    sessions = []
    for idx in df.index[df["hm"] == range_hm]:
        row = df.iloc[idx]
        future = df.iloc[idx + 1: idx + 1 + MAX_FUTURE_MIN]
        future = future[(future["ts"] - row["ts"]) <= pd.Timedelta(minutes=MAX_FUTURE_MIN)]
        if future.empty:
            continue
        sessions.append({
            "date": row["date"], "range_ts": row["ts"],
            "range_high": float(row["h"]), "range_low": float(row["l"]),
            "bars": future[["ts", "hm", "o", "h", "l", "c"]].to_dict("records"),
        })
    return sessions


def sim_session(sess: dict, symbol: str, entry_win: int, hold_min: int,
                target_mult: float, stop_mode: str, stop_param: float,
                width_min_pts: float) -> list[dict]:
    """entry = bar.close (matches production exactly), direction='first'
    (one trade/session, matches production's fire-once latch)."""
    point_value = POINT_VALUE[symbol]
    orb_h, orb_l = sess["range_high"], sess["range_low"]
    width = orb_h - orb_l
    if width <= 0 or width < width_min_pts:
        return []
    mid = (orb_h + orb_l) / 2
    target_pts = width * target_mult

    for i, bar in enumerate(sess["bars"]):
        elapsed = (bar["ts"] - sess["range_ts"]).total_seconds() / 60.0
        if elapsed > entry_win:
            break
        direction = 0
        if bar["c"] > orb_h: direction = 1
        elif bar["c"] < orb_l: direction = -1
        if direction == 0:
            continue

        entry = bar["c"]
        if stop_mode == "midpoint_floor":
            raw_stop_pts = (orb_h - mid) if direction == 1 else (mid - orb_l)
            stop_pts = max(raw_stop_pts, stop_param)   # stop_param = floor, pts
        else:  # far_side_cap
            raw_stop_pts = width
            max_pts = (stop_param / point_value) if stop_param > 0 else raw_stop_pts
            stop_pts = min(raw_stop_pts, max_pts)

        tgt = entry + direction * target_pts
        stp = entry - direction * stop_pts

        future = sess["bars"][i + 1:]
        window = [b for b in future if (b["ts"] - bar["ts"]).total_seconds() / 60.0 <= hold_min]
        result, exit_p = None, None
        for b in window:
            if direction == 1:
                if b["l"] <= stp: result, exit_p = "stop", stp; break
                if b["h"] >= tgt: result, exit_p = "target", tgt; break
            else:
                if b["h"] >= stp: result, exit_p = "stop", stp; break
                if b["l"] <= tgt: result, exit_p = "target", tgt; break
        if result is None:
            exit_p = window[-1]["c"] if window else entry
            result = "hold_exit"
        pnl = (exit_p - entry) * direction * point_value - COMMISSION
        return [dict(result=result, pnl=pnl, width=width)]
    return []


def _summarize(trades: list[dict]) -> dict:
    pnls = [t["pnl"] for t in trades]
    if not pnls:
        return dict(trades=0, wr=0, pf=0, total=0.0)
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) else (999 if wins else 0)
    return dict(trades=len(pnls), wr=round(len(wins) / len(pnls), 3), pf=round(pf, 3),
               total=round(sum(pnls), 1))


# ── Per-combo config ────────────────────────────────────────────────────────

COMBOS = {
    "ORB MES": dict(symbol="MES", range_hm=9 * 60 + 30, entry_win=5, hold_min=10,
                    target_mult=1.0, stop_mode="midpoint_floor",
                    stop_params=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
                    width_mins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]),
    "ORB MNQ": dict(symbol="MNQ", range_hm=9 * 60 + 30, entry_win=5, hold_min=10,
                    target_mult=1.0, stop_mode="far_side_cap", baseline_stop_param=500,
                    stop_params=[100, 150, 200, 250, 300, 350, 400, 500, 650, 800, 1000, 0],
                    width_mins=[0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]),
    "ORB-cls MES": dict(symbol="MES", range_hm=15 * 60 + 50, entry_win=5, hold_min=15,
                        target_mult=3.25, stop_mode="midpoint_floor",
                        stop_params=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10],
                        width_mins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]),
    "ORB-cls MNQ": dict(symbol="MNQ", range_hm=15 * 60 + 50, entry_win=3, hold_min=15,
                        target_mult=3.25, stop_mode="midpoint_floor",
                        stop_params=[0, 5, 10, 15, 20, 25, 30, 35, 40, 50],
                        width_mins=[0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80]),
}


def run_combo(label: str, cfg: dict, sessions: list[dict]) -> pd.DataFrame:
    rows = []
    for sp, wm in product(cfg["stop_params"], cfg["width_mins"]):
        trades = []
        for sess in sessions:
            trades.extend(sim_session(sess, cfg["symbol"], cfg["entry_win"], cfg["hold_min"],
                                      cfg["target_mult"], cfg["stop_mode"], sp, wm))
        s = _summarize(trades)
        if s["trades"] < 15:
            continue
        rows.append(dict(stop_param=sp, width_min=wm, **s))
    return pd.DataFrame(rows).sort_values("pf", ascending=False)


def main():
    walkforward = "--walkforward" in sys.argv
    bars_cache: dict = {}

    def get_sessions(symbol: str, range_hm: int) -> list[dict]:
        key = (symbol, range_hm)
        if key not in bars_cache:
            df = fetch_bars(symbol)
            bars_cache[key] = sorted(build_sessions(df, range_hm), key=lambda s: s["date"])
        return bars_cache[key]

    for label, cfg in COMBOS.items():
        sessions = get_sessions(cfg["symbol"], cfg["range_hm"])
        print(f"\n{'#'*90}\n{label}  ({cfg['symbol']}, stop_mode={cfg['stop_mode']}, "
              f"{len(sessions)} sessions, target={cfg['target_mult']}x entry_win={cfg['entry_win']} "
              f"hold={cfg['hold_min']})\n{'#'*90}")

        # Baseline: current shipped config (width_min=0; stop_param=0 means
        # "no floor" for midpoint_floor combos, or the explicit current $
        # cap for far_side_cap combos)
        baseline_sp = cfg.get("baseline_stop_param", 0)
        baseline = sim_combo_single(cfg, sessions, baseline_sp, 0.0)
        print(f"BASELINE (current live config, stop_param={baseline_sp}): {baseline}")

        if not walkforward:
            results = run_combo(label, cfg, sessions)
            print(f"\nTOP 15 by PF:")
            print(results.head(15).to_string(index=False))
            print(f"\nTOP 10 by total P&L:")
            print(results.sort_values("total", ascending=False).head(10).to_string(index=False))
        else:
            cut = int(len(sessions) * 0.7)
            train, test = sessions[:cut], sessions[cut:]
            print(f"Train: {len(train)} sessions ({train[0]['date']}->{train[-1]['date']})  "
                  f"Test: {len(test)} sessions ({test[0]['date']}->{test[-1]['date']})")
            train_results = run_combo(label, cfg, train)
            top5 = train_results.head(5)
            print("\nTop 5 on TRAIN:")
            print(top5.to_string(index=False))
            print("\nSame params on TEST (out-of-sample):")
            out_rows = []
            for _, r in top5.iterrows():
                trades = []
                for sess in test:
                    trades.extend(sim_session(sess, cfg["symbol"], cfg["entry_win"], cfg["hold_min"],
                                              cfg["target_mult"], cfg["stop_mode"],
                                              r.stop_param, r.width_min))
                s = _summarize(trades)
                out_rows.append(dict(stop_param=r.stop_param, width_min=r.width_min,
                                     train_pf=r.pf, train_total=r.total,
                                     test_trades=s["trades"], test_pf=s["pf"], test_total=s["total"]))
            print(pd.DataFrame(out_rows).to_string(index=False))


def sim_combo_single(cfg: dict, sessions: list[dict], stop_param: float, width_min: float) -> dict:
    trades = []
    for sess in sessions:
        trades.extend(sim_session(sess, cfg["symbol"], cfg["entry_win"], cfg["hold_min"],
                                  cfg["target_mult"], cfg["stop_mode"], stop_param, width_min))
    return _summarize(trades)


if __name__ == "__main__":
    main()
