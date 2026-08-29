"""
Backtest: "Wall cascade" signal — multiple ask (or bid) walls broken in rapid
succession, rather than a single isolated wall break.

Motivated by a real trade (2026-08-28 14:45 UTC, MESU26 long, +13pts/$390):
logs/wall_events.csv showed a rapid sequence of ask walls placed and broken
at successive price levels (7762.50 -> 7763.50 -> 7764.50 -> 7765.50 -> ...)
within about 5 minutes — a cascade of offers being taken out, not a single
wall_break event. The live wall_break strategy only acts on one wall at a
time (peak_size 100-300, 2+ tests) and would have missed most of this
sequence (individual walls here were mostly 60-90 contracts). This tests
whether requiring N consecutive breakouts (same side) within a short time
window, rather than one, is a better-conditioned entry signal.

Cascade detection: within logs/wall_events.csv, walk breakout events for one
side in time order. Consecutive breakouts (same side) less than max_gap_sec
apart extend the current cascade; a gap larger than that starts a new one.
A signal fires the moment a cascade first reaches cascade_min breakouts —
entry price = price_at_event of that breakout. One signal per cascade (no
repeat-firing as a long cascade keeps extending).

Usage:
    python src/backtest_wall_cascade.py
"""

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz

WALL_EVENTS_CSV = "logs/wall_events.csv"
BARS_DB         = "data/bars.db"
ET              = pytz.timezone("America/New_York")

CASCADE_MINS   = [2, 3, 4]
MAX_GAP_SECS   = [15, 20, 30, 45, 60]

STOP_PTS   = 4.5
TARGET_PTS = 12.0
HOLD_MIN   = 25

SETTLE_MIN   = 9 * 60 + 40   # 9:40 ET
LIVE_START   = 10 * 60 + 30  # 10:30 ET — live wall_break's actual entry gate
LIVE_END     = 13 * 60       # 13:00 ET


def load_breakouts(symbol: str = "MES", window: str = "settled") -> pd.DataFrame:
    """window: 'rth' (9:30-16:00), 'settled' (9:40-16:00), or 'live' (10:30-13:00,
    matching the current live wall_break entry gate)."""
    df = pd.read_csv(WALL_EVENTS_CSV, header=None,
        names=["ts", "symbol", "side", "wall_price", "wall_size", "peak_size",
               "event", "test_count", "price_at_event"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["price_at_event"] = pd.to_numeric(df["price_at_event"], errors="coerce")
    bo = df[(df["event"] == "breakout") & (df["symbol"] == symbol) &
            df["price_at_event"].notna()].copy()
    bo["ts_et"] = bo["ts"].dt.tz_convert(ET)
    bo["hm"]    = bo["ts_et"].dt.hour * 60 + bo["ts_et"].dt.minute
    if window == "rth":
        bo = bo[(bo["hm"] >= 570) & (bo["hm"] < 960)]
    elif window == "settled":
        bo = bo[(bo["hm"] >= SETTLE_MIN) & (bo["hm"] < 960)]
    elif window == "live":
        bo = bo[(bo["hm"] >= LIVE_START) & (bo["hm"] < LIVE_END)]
    else:
        raise ValueError(window)
    return bo.sort_values("ts").reset_index(drop=True)


def load_bars(symbol: str = "MES") -> pd.DataFrame:
    conn = sqlite3.connect(BARS_DB)
    bars = pd.read_sql(
        f"SELECT ts, high, low, close FROM bars WHERE symbol='{symbol}' AND minutes=1 ORDER BY ts",
        conn)
    conn.close()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    return bars


def detect_cascades(breakouts: pd.DataFrame, side: str,
                     cascade_min: int, max_gap_sec: int) -> pd.DataFrame:
    direction = 1 if side == "ask" else -1
    events = breakouts[breakouts["side"] == side].sort_values("ts")

    signals = []
    cascade_count = 0
    last_ts = None
    signaled_this_cascade = False

    for _, e in events.iterrows():
        if last_ts is not None and (e["ts"] - last_ts).total_seconds() <= max_gap_sec:
            cascade_count += 1
        else:
            cascade_count = 1
            signaled_this_cascade = False
        last_ts = e["ts"]

        if cascade_count >= cascade_min and not signaled_this_cascade:
            signals.append({
                "ts": e["ts"], "side": side, "direction": direction,
                "entry": e["price_at_event"], "cascade_len": cascade_count,
            })
            signaled_this_cascade = True

    return pd.DataFrame(signals)


def simulate_trade(row, bars, stop_pts, target_pts, hold_min):
    future = bars[bars["ts"] > row["ts"]].head(hold_min)
    if future.empty:
        return None, None
    entry = row["entry"]
    d     = row["direction"]
    stop_price   = entry - d * stop_pts
    target_price = entry + d * target_pts
    for _, bar in future.iterrows():
        if d == 1:
            if bar["low"]  <= stop_price:   return "STOPPED", -stop_pts
            if bar["high"] >= target_price: return "TARGET",  +target_pts
        else:
            if bar["high"] >= stop_price:   return "STOPPED", -stop_pts
            if bar["low"]  <= target_price: return "TARGET",  +target_pts
    exit_price = future["close"].iloc[-1]
    return "TIME", (exit_price - entry) * d


def summarize(label, outcomes, pnls):
    n = len(pnls)
    if n == 0:
        return None
    wr  = sum(1 for p in pnls if p > 0) / n
    ev  = np.mean(pnls)
    return {"label": label, "n": n, "wr": wr, "ev": ev, "total": sum(pnls)}


def main():
    print("Loading data…")
    breakouts = load_breakouts()
    bars      = load_bars()
    print(f"  RTH MES breakout events: {len(breakouts)}  "
          f"({breakouts['ts'].min().date()} -> {breakouts['ts'].max().date()})")
    print(f"  MES 1-min bars: {len(bars):,}  "
          f"({bars['ts'].min().date()} -> {bars['ts'].max().date()})")

    # ── Baseline: single wall break (cascade_min=1) for comparison ──────────
    print(f"\n{'='*82}")
    print(f"BASELINE  single breakout (cascade_min=1)  stop={STOP_PTS}pt  "
          f"target={TARGET_PTS:.0f}pt  hold={HOLD_MIN}min")
    print(f"{'='*82}")
    for side in ["ask", "bid"]:
        sigs = detect_cascades(breakouts, side, cascade_min=1, max_gap_sec=1)
        outcomes, pnls = [], []
        for _, row in sigs.iterrows():
            o, p = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
            if o is not None:
                outcomes.append(o); pnls.append(p)
        r = summarize(f"side={side}", outcomes, pnls)
        if r:
            print(f"  {r['label']:<10} n={r['n']:5d}  WR={r['wr']:6.1%}  "
                  f"EV={r['ev']:6.2f}pts  total={r['total']:8.1f}pts")

    # ── Cascade sweep ─────────────────────────────────────────────────────
    print(f"\n{'='*82}")
    print(f"CASCADE sweep  stop={STOP_PTS}pt  target={TARGET_PTS:.0f}pt  hold={HOLD_MIN}min")
    print(f"{'='*82}")
    print(f"  {'cascade_min':>11} {'max_gap':>8} {'side':>5} {'n':>5} {'WR':>7} {'EV':>7} {'total':>8}")

    results = []
    for cmin in CASCADE_MINS:
        for gap in MAX_GAP_SECS:
            for side in ["ask", "bid"]:
                sigs = detect_cascades(breakouts, side, cascade_min=cmin, max_gap_sec=gap)
                outcomes, pnls = [], []
                for _, row in sigs.iterrows():
                    o, p = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
                    if o is not None:
                        outcomes.append(o); pnls.append(p)
                r = summarize(f"{side}", outcomes, pnls)
                if r is None:
                    continue
                results.append({**r, "cascade_min": cmin, "max_gap_sec": gap, "side": side})
                print(f"  {cmin:11d} {gap:8d} {side:>5} {r['n']:5d} {r['wr']:7.1%} "
                      f"{r['ev']:7.2f} {r['total']:8.1f}")

    r = pd.DataFrame(results)
    print(f"\n{'='*82}")
    print("Top 15 by EV (min 20 trades)")
    print(f"{'='*82}")
    top = r[r["n"] >= 20].sort_values("ev", ascending=False).head(15)
    for _, row in top.iterrows():
        print(f"  cascade_min={row['cascade_min']}  max_gap={row['max_gap_sec']}s  "
              f"side={row['side']}  n={row['n']:.0f}  WR={row['wr']:.1%}  "
              f"EV={row['ev']:.2f}pts  total={row['total']:.1f}pts")

    # ── Combined (both sides) at a few representative configs ───────────────
    print(f"\n{'='*82}")
    print("Combined (both sides) at each cascade_min, max_gap=30s")
    print(f"{'='*82}")
    for cmin in CASCADE_MINS:
        outcomes, pnls = [], []
        for side in ["ask", "bid"]:
            sigs = detect_cascades(breakouts, side, cascade_min=cmin, max_gap_sec=30)
            for _, row in sigs.iterrows():
                o, p = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
                if o is not None:
                    outcomes.append(o); pnls.append(p)
        rr = summarize(f"cascade_min={cmin}", outcomes, pnls)
        if rr:
            print(f"  cascade_min={cmin}  n={rr['n']:5d}  WR={rr['wr']:6.1%}  "
                  f"EV={rr['ev']:6.2f}pts  total={rr['total']:8.1f}pts")


if __name__ == "__main__":
    main()
