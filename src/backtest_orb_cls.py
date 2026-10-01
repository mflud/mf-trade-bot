"""
backtest_orb_cls.py — Exploratory backtest for a "closing range breakout"
(ORB-cls): range built from the 15:50-15:51 ET bar instead of the 9:30-9:31
opening bar, motivated by the observation that price whips around in the
last minute before the 16:00 ET cash close similarly to how it does in the
first minute after the 9:30 open.

Mirrors the structure of backtest_mes_1min_orb.py / backtest_mnq_1min_orb.py
(same bars.db source, same entry/stop/target mechanics) but:
  - Range bar = the 15:50 ET 1-min bar (vs. 9:30 for the morning ORB).
  - Two exit-horizon families are swept side by side, per user request:
      'rth_close' — forced flat at the 16:00 ET cash close (like the
                    existing morning ORB; ~9 min of live price action).
      numeric N   — hold up to N minutes into the Globex session,
                    stop/target checked every bar as usual.
  - Entry window is short (a few minutes after 15:51) since the whole
    premise is a tight closing range, not a multi-hour accumulation.

This is a first exploratory pass — stop_type is fixed at 'midpoint' and
width filters are off, to keep the combo grid tractable. Revisit with a
wider sweep once a promising region shows up here.

Usage:
  python src/backtest_orb_cls.py            # MES
  python src/backtest_orb_cls.py --sym MNQ
"""

import argparse
import sqlite3
import sys
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

RANGE_HM   = 15 * 60 + 50   # 15:50 ET bar defines the range
RTH_CLOSE_HM = 16 * 60      # 16:00 ET cash close

# TopstepX requires flat by 16:10 ET (TRADING_CUTOFF_MST default 13:10 MST in
# trading_bot.py). That's only 20 min after the 15:50 range bar opens, so most
# of the Globex-carry sweep below is research-only — see the live_deployable
# column / the dedicated section in the report.
TOPSTEP_CUTOFF_MIN_AFTER_RANGE = 20

POINT_VALUE = {"MES": 5.0, "MNQ": 2.0}
COMMISSION  = 2.0 * 2       # $2/side × 2 sides (approximate)

MAX_FUTURE_MIN = 260        # cap future-bar lookup (covers largest hold + buffer)


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_bars(symbol: str) -> pd.DataFrame:
    print(f"Loading {symbol} 1-min bars from bars.db…", flush=True)
    conn = sqlite3.connect("data/bars.db")
    rows = conn.execute(
        "SELECT ts, open, high, low, close, volume FROM bars "
        "WHERE symbol=? AND minutes=1 ORDER BY ts", (symbol,)
    ).fetchall()
    conn.close()
    if not rows:
        raise RuntimeError(f"No {symbol} 1-min bars in bars.db")
    df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"])
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = df[df["ts"].dt.hour != 21].reset_index(drop=True)  # CME settlement gap
    df = df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    print(f"Loaded {len(df)} bars  ({df['ts'].iloc[0].date()} → {df['ts'].iloc[-1].date()})",
          flush=True)
    return df


def build_sessions(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    df["et"]      = df["ts"].dt.tz_convert(ET)
    df["date_et"] = df["et"].dt.date
    df["hm"]      = df["et"].dt.hour * 60 + df["et"].dt.minute

    sessions = []
    range_idx = df.index[df["hm"] == RANGE_HM].tolist()
    for idx in range_idx:
        row = df.iloc[idx]
        future = df.iloc[idx + 1: idx + 1 + MAX_FUTURE_MIN]
        future = future[(future["ts"] - row["ts"]) <= pd.Timedelta(minutes=MAX_FUTURE_MIN)]
        if future.empty:
            continue
        sessions.append({
            "date":       row["date_et"],
            "range_ts":   row["ts"],
            "range_high": float(row["h"]),
            "range_low":  float(row["l"]),
            "bars":       future[["ts", "hm", "o", "h", "l", "c"]].to_dict("records"),
        })
    return sessions


# ── Single-session trade simulator ────────────────────────────────────────────

def sim_session(sess: dict, symbol: str,
                entry_type: str,     # 'touch' | 'close'
                target_mult: float,
                entry_win: int,       # minutes after range bar to look for break
                direction: str,       # 'first' | 'both'
                hold: "int|str",      # 'rth_close' or minutes
                stop_type: str = "midpoint",  # 'midpoint' | 'far_side'
                cap_1600: bool = False,       # also force-exit at 16:00 ET regardless of hold
                ) -> list[dict]:
    point_value = POINT_VALUE[symbol]
    orb_h = sess["range_high"]
    orb_l = sess["range_low"]
    width = orb_h - orb_l
    mid   = (orb_h + orb_l) / 2
    if width <= 0:
        return []

    target_pts = width * target_mult
    if stop_type == "midpoint":
        stop_pts_long  = orb_h - mid
        stop_pts_short = mid - orb_l
    else:  # far_side
        stop_pts_long  = width
        stop_pts_short = width

    trades     = []
    long_done  = False
    short_done = False

    for i, bar in enumerate(sess["bars"]):
        elapsed_min = (bar["ts"] - sess["range_ts"]).total_seconds() / 60.0
        if elapsed_min > entry_win:
            break

        if not long_done and (direction == "both" or not short_done):
            triggered = (bar["h"] >= orb_h) if entry_type == "touch" else (bar["c"] > orb_h)
            if triggered:
                entry = orb_h
                tgt   = entry + target_pts
                stp   = entry - stop_pts_long
                exit_p, result = _sim_exit(entry, tgt, stp, 1, bar, sess["bars"][i + 1:], hold, cap_1600)
                pnl = (exit_p - entry) * point_value - COMMISSION
                trades.append(dict(dir=1, entry=entry, target=tgt, stop=stp,
                                   exit=exit_p, result=result, pnl=pnl,
                                   entry_hm=bar["hm"]))
                long_done = True
                if direction == "first":
                    break

        if not short_done and (direction == "both" or not long_done):
            triggered = (bar["l"] <= orb_l) if entry_type == "touch" else (bar["c"] < orb_l)
            if triggered:
                entry = orb_l
                tgt   = entry - target_pts
                stp   = entry + stop_pts_short
                exit_p, result = _sim_exit(entry, tgt, stp, -1, bar, sess["bars"][i + 1:], hold, cap_1600)
                pnl = (entry - exit_p) * point_value - COMMISSION
                trades.append(dict(dir=-1, entry=entry, target=tgt, stop=stp,
                                   exit=exit_p, result=result, pnl=pnl,
                                   entry_hm=bar["hm"]))
                short_done = True
                if direction == "first":
                    break

    return trades


def _sim_exit(entry, tgt, stp, direction, trigger_bar, future_bars, hold, cap_1600):
    trigger_ts = trigger_bar["ts"]
    force_1600 = cap_1600 or hold == "rth_close"

    def _cutoff_ok(b):
        if force_1600 and b["hm"] >= RTH_CLOSE_HM:
            return False
        if hold == "rth_close":
            return True
        return (b["ts"] - trigger_ts).total_seconds() / 60.0 <= hold

    window = [b for b in future_bars if _cutoff_ok(b)]
    if direction == 1:
        for b in window:
            if b["l"] <= stp: return stp, "stop"
            if b["h"] >= tgt: return tgt, "target"
    else:
        for b in window:
            if b["h"] >= stp: return stp, "stop"
            if b["l"] <= tgt: return tgt, "target"

    label = "eod" if force_1600 else "hold_exit"
    return (window[-1]["c"] if window else entry), label


# ── Sweep ──────────────────────────────────────────────────────────────────────

# Hold specs: (hold, cap_1600). cap_1600=True additionally force-exits at the
# 16:00 ET cash close even if the nominal hold window hasn't elapsed — tests
# the hypothesis that post-close movement dies down and carrying past 16:00
# isn't worth the extra time. 'rth_close' is its own case (always capped).
_NUMERIC_HOLDS = [15, 30, 60, 90, 120, 180, 240]
HOLD_SPECS = (
    [("rth_close", True)]
    + [(h, True) for h in _NUMERIC_HOLDS]    # capped at 16:00 regardless of h
    + [(h, False) for h in _NUMERIC_HOLDS]   # uncapped Globex carry (original behavior)
)

COMBO_GRID = dict(
    entry_types  = ["close"],                              # 'touch' never led in prior sweeps
    stop_types   = ["midpoint", "far_side"],
    target_mults = [2.0, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0],   # widened around the 3.0x winner
    entry_wins   = [3, 4, 5, 6, 7, 9],
    directions   = ["first", "both"],
    hold_specs   = HOLD_SPECS,
)


def _is_live_deployable(entry_win: int, hold, cap_1600: bool) -> bool:
    # Capped-at-1600 (or rth_close) always finishes by 16:00, well inside
    # TopstepX's 16:10 ET flat-by cutoff, regardless of the nominal hold.
    if hold == "rth_close" or cap_1600:
        return True
    return (entry_win + hold) <= TOPSTEP_CUTOFF_MIN_AFTER_RANGE


def evaluate_combo(sessions: list[dict], symbol: str,
                   entry_type: str, target: float, entry_win: int,
                   direction: str, hold, stop_type: str = "midpoint",
                   cap_1600: bool = False) -> dict | None:
    """Run one fixed param combo across `sessions` and summarize. None if no trades."""
    all_trades = []
    for sess in sessions:
        all_trades.extend(sim_session(sess, symbol, entry_type, target, entry_win, direction,
                                      hold, stop_type, cap_1600))
    if not all_trades:
        return None
    pnls   = [t["pnl"] for t in all_trades]
    wins   = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr     = len(wins) / len(pnls)
    pf     = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 999
    return dict(
        entry_type=entry_type, stop_type=stop_type, target=target, entry_win=entry_win,
        direction=direction, hold=hold, cap_1600=cap_1600,
        trades=len(pnls), wr=round(wr, 3), pf=round(pf, 3),
        total=round(sum(pnls), 0),
        avg_win=round(np.mean(wins) if wins else 0, 2),
        avg_loss=round(np.mean(losses) if losses else 0, 2),
        live_deployable=_is_live_deployable(entry_win, hold, cap_1600),
    )


def run_sweep(sessions: list[dict], symbol: str, min_trades: int = 20) -> pd.DataFrame:
    combos = list(product(COMBO_GRID["entry_types"], COMBO_GRID["stop_types"],
                          COMBO_GRID["target_mults"], COMBO_GRID["entry_wins"],
                          COMBO_GRID["directions"], COMBO_GRID["hold_specs"]))
    print(f"\nRunning {len(combos)} parameter combos on {len(sessions)} sessions…", flush=True)

    results = []
    for i, (et, stp_t, tm, ew, dr, (hold, cap)) in enumerate(combos):
        if i % 200 == 0:
            print(f"  {i}/{len(combos)}…", flush=True, end="\r")

        r = evaluate_combo(sessions, symbol, et, tm, ew, dr, hold, stp_t, cap)
        if r is None or r["trades"] < min_trades:
            continue
        results.append(r)

    print(f"\nDone. {len(results)} combos with ≥{min_trades} trades.", flush=True)
    return pd.DataFrame(results).sort_values("pf", ascending=False)


# ── Walk-forward validation ─────────────────────────────────────────────────────

def run_walkforward(symbol: str, train_frac: float = 0.7, top_n: int = 5):
    df = fetch_bars(symbol)
    sessions = sorted(build_sessions(df), key=lambda s: s["date"])
    print(f"Sessions with a valid 15:50 ET range bar: {len(sessions)}", flush=True)
    if len(sessions) < 30:
        sys.exit("Too few sessions for a meaningful walk-forward split.")

    cut = int(len(sessions) * train_frac)
    train, test = sessions[:cut], sessions[cut:]
    print(f"\nTrain: {len(train)} sessions  ({train[0]['date']} → {train[-1]['date']})")
    print(f"Test:  {len(test)} sessions  ({test[0]['date']} → {test[-1]['date']})")

    train_results = run_sweep(train, symbol, min_trades=15)
    deployable = train_results[train_results["live_deployable"]].sort_values("pf", ascending=False)
    if deployable.empty:
        sys.exit("No live-deployable combo cleared the train-set trade threshold.")

    top = deployable.head(top_n)
    print(f"\n{'='*95}")
    print(f"TOP {top_n} LIVE-DEPLOYABLE COMBOS — fit on TRAIN only ({symbol})")
    print("=" * 95)
    print(top.to_string(index=False))

    rows = []
    for _, r in top.iterrows():
        test_r = evaluate_combo(test, symbol, r.entry_type, r.target,
                                int(r.entry_win), r.direction, r.hold,
                                r.stop_type, bool(r.cap_1600))
        rows.append(dict(
            entry_type=r.entry_type, stop_type=r.stop_type, target=r.target,
            entry_win=r.entry_win, direction=r.direction, hold=r.hold, cap_1600=r.cap_1600,
            train_trades=r.trades, train_wr=r.wr, train_pf=r.pf, train_total=r.total,
            test_trades=(test_r["trades"] if test_r else 0),
            test_wr=(test_r["wr"] if test_r else None),
            test_pf=(test_r["pf"] if test_r else None),
            test_total=(test_r["total"] if test_r else None),
        ))

    out = pd.DataFrame(rows)
    print(f"\n{'='*95}")
    print(f"TRAIN vs. OUT-OF-SAMPLE TEST — {symbol}")
    print("=" * 95)
    print(out.to_string(index=False))

    held_up = out[(out["test_pf"].notna()) & (out["test_pf"] >= 1.0) & (out["test_total"] > 0)]
    print(f"\n{len(held_up)}/{len(out)} train-picked combos stayed profitable (PF≥1.0, total>0) out-of-sample.")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES", choices=["MES", "MNQ"])
    parser.add_argument("--walkforward", action="store_true",
                        help="Fit top combos on the first --train-frac of sessions, "
                             "evaluate out-of-sample on the rest.")
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()
    symbol = args.sym

    if args.walkforward:
        run_walkforward(symbol, train_frac=args.train_frac)
        return

    df = fetch_bars(symbol)
    sessions = build_sessions(df)
    print(f"Sessions with a valid 15:50 ET range bar: {len(sessions)}", flush=True)
    if len(sessions) < 20:
        sys.exit("Too few sessions to backtest meaningfully — check bars.db coverage.")

    widths = [s["range_high"] - s["range_low"] for s in sessions]
    print(f"Range width (pts): mean={np.mean(widths):.2f}  median={np.median(widths):.2f}  "
          f"p90={np.percentile(widths, 90):.2f}", flush=True)

    results = run_sweep(sessions, symbol)
    if results.empty:
        sys.exit("No combos reached 20 trades — range likely too tight / entry window too short.")

    print("\n" + "=" * 95)
    print(f"TOP 20 COMBOS BY PROFIT FACTOR — {symbol}")
    print("=" * 95)
    print(results.head(20).to_string(index=False))

    by_total = results.sort_values("total", ascending=False).head(20)
    print("\n" + "=" * 95)
    print(f"TOP 20 COMBOS BY TOTAL P&L — {symbol}")
    print("=" * 95)
    print(by_total.to_string(index=False))

    high_wr = results[(results["wr"] >= 0.55) & (results["pf"] >= 1.3)].sort_values("total", ascending=False)
    print(f"\n{'='*95}")
    print(f"DECENT WIN-RATE COMBOS (WR≥55%, PF≥1.3) — {len(high_wr)} found")
    print("=" * 95)
    print(high_wr.head(20).to_string(index=False) if not high_wr.empty else "  none")

    # Split rth_close / capped-at-1600 / uncapped Globex-carry so the three
    # exit families are easy to compare (tests the "movement dies after
    # 16:00" hypothesis directly: capped vs. uncapped at otherwise-identical
    # params).
    rth_only    = results[results["hold"] == "rth_close"].sort_values("pf", ascending=False)
    capped_only = results[(results["hold"] != "rth_close") & (results["cap_1600"])].sort_values("pf", ascending=False)
    carry_only  = results[(results["hold"] != "rth_close") & (~results["cap_1600"])].sort_values("pf", ascending=False)
    print(f"\n{'='*95}")
    print(f"TOP 10 — RTH-close-only exits")
    print("=" * 95)
    print(rth_only.head(10).to_string(index=False) if not rth_only.empty else "  none")
    print(f"\n{'='*95}")
    print(f"TOP 10 — numeric hold, CAPPED at 16:00 ET")
    print("=" * 95)
    print(capped_only.head(10).to_string(index=False) if not capped_only.empty else "  none")
    print(f"\n{'='*95}")
    print(f"TOP 10 — numeric hold, UNCAPPED (Globex-carry past 16:00)")
    print("=" * 95)
    print(carry_only.head(10).to_string(index=False) if not carry_only.empty else "  none")

    # Direct capped-vs-uncapped comparison at matching params
    merge_keys = ["entry_type", "stop_type", "target", "entry_win", "direction", "hold"]
    cmp_df = capped_only.merge(carry_only, on=merge_keys, suffixes=("_capped", "_uncapped"))
    if not cmp_df.empty:
        cmp_df["pf_delta"] = cmp_df["pf_capped"] - cmp_df["pf_uncapped"]
        print(f"\n{'='*95}")
        print("CAPPED-AT-1600 vs. UNCAPPED, matched params (positive pf_delta = capping helped)")
        print("=" * 95)
        cols = merge_keys + ["pf_capped", "pf_uncapped", "pf_delta",
                             "total_capped", "total_uncapped"]
        print(cmp_df.sort_values("pf_delta", ascending=False)[cols].head(15).to_string(index=False))
        print(f"\nMean pf_delta across {len(cmp_df)} matched combos: {cmp_df['pf_delta'].mean():.3f} "
              f"({(cmp_df['pf_delta'] > 0).sum()}/{len(cmp_df)} combos favor capping at 16:00)")

    deployable = results[results["live_deployable"]].sort_values("pf", ascending=False)
    print(f"\n{'='*95}")
    print(f"TOP 15 — LIVE-DEPLOYABLE under TopstepX's 16:10 ET flat-by cutoff "
          f"({len(deployable)}/{len(results)} combos qualify)")
    print("=" * 95)
    print(deployable.head(15).to_string(index=False) if not deployable.empty else
          "  none — every combo that hit 20+ trades needs more time than the 16:10 ET cutoff allows")

    out_path = f"{symbol.lower()}_orb_cls_sweep.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
