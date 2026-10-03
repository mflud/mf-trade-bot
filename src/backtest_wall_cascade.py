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

# Matches current live ask-side config (trading_bot.py BA_BRK_* constants,
# checked 2026-10-02 — this script's constants had drifted stale: stop was
# 4.5 here vs the live 3.5, and the live window is 9:40-13:00, not 10:30-13:00).
STOP_PTS   = 3.5
TARGET_PTS = 12.0
HOLD_MIN   = 25
BA_BRK_LIVE_MAX_GAP = 45     # matches live BA_BRK_MAX_GAP_SEC

# Live bid-side config (different stop/target/hold from ask) — trading_bot.py
# BA_BRK_BID_STOP_PTS / BID_TARGET_PTS / BID_HOLD_MIN.
BA_BRK_BID_STOP   = 4.0
BA_BRK_BID_TARGET = 18.0
BA_BRK_BID_HOLD   = 30

SETTLE_MIN   = 9 * 60 + 40   # 9:40 ET
LIVE_START   = 9 * 60 + 40   # 9:40 ET — matches live BA_BRK_TRADE_START
LIVE_END     = 13 * 60       # 13:00 ET — matches live BA_BRK_TRADE_END


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
        f"SELECT ts, open, high, low, close FROM bars WHERE symbol='{symbol}' AND minutes=1 ORDER BY ts",
        conn)
    conn.close()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    return bars


def compute_day_opens(bars: pd.DataFrame) -> dict:
    """date -> first RTH (>=9:30 ET) bar's open, matching trading_bot.py's
    _day_open_price() — used for the ask/bid 'aligned with the day's move
    since open' hard filter that's live but this script previously lacked."""
    b = bars.copy()
    b["ts_et"] = b["ts"].dt.tz_convert(ET)
    b["date"] = b["ts_et"].dt.date
    b["hm"] = b["ts_et"].dt.hour * 60 + b["ts_et"].dt.minute
    rth = b[b["hm"] >= 570].sort_values("ts")
    return rth.groupby("date")["open"].first().to_dict()


def detect_cascades(breakouts: pd.DataFrame, side: str,
                     cascade_min: int, max_gap_sec: int,
                     day_opens: "dict | None" = None) -> pd.DataFrame:
    """day_opens: if given, applies the live 'aligned with the day's move
    since open' hard filter at signal-fire time (ask only fires above the
    day's open, bid only below) — matches trading_bot.py's
    evaluate_ba_brk_signal exactly rather than ignoring that gate."""
    direction = 1 if side == "ask" else -1
    events = breakouts[breakouts["side"] == side].sort_values("ts")

    signals = []
    cascade_count = 0
    last_ts = None
    signaled_this_cascade = False
    cascade_test_counts: list = []

    for _, e in events.iterrows():
        if last_ts is not None and (e["ts"] - last_ts).total_seconds() <= max_gap_sec:
            cascade_count += 1
        else:
            cascade_count = 1
            signaled_this_cascade = False
            cascade_test_counts = []
        last_ts = e["ts"]
        cascade_test_counts.append(e["test_count"])

        if cascade_count >= cascade_min and not signaled_this_cascade:
            if day_opens is not None:
                day_open = day_opens.get(e["ts_et"].date())
                if day_open is None:
                    continue
                if side == "ask" and e["price_at_event"] <= day_open:
                    continue
                if side == "bid" and e["price_at_event"] >= day_open:
                    continue
            signals.append({
                "ts": e["ts"], "side": side, "direction": direction,
                "entry": e["price_at_event"], "cascade_len": cascade_count,
                "trigger_test_count": e["test_count"],
                "max_test_count": max(cascade_test_counts),
            })
            signaled_this_cascade = True

    return pd.DataFrame(signals)


def simulate_trade(row, bars, stop_pts, target_pts, hold_min):
    future = bars[bars["ts"] > row["ts"]].head(hold_min)
    if future.empty:
        return None, None, None
    entry = row["entry"]
    d     = row["direction"]
    stop_price   = entry - d * stop_pts
    target_price = entry + d * target_pts
    for _, bar in future.iterrows():
        if d == 1:
            if bar["low"]  <= stop_price:   return "STOPPED", -stop_pts, bar["ts"]
            if bar["high"] >= target_price: return "TARGET",  +target_pts, bar["ts"]
        else:
            if bar["high"] >= stop_price:   return "STOPPED", -stop_pts, bar["ts"]
            if bar["low"]  <= target_price: return "TARGET",  +target_pts, bar["ts"]
    exit_price = future["close"].iloc[-1]
    return "TIME", (exit_price - entry) * d, future["ts"].iloc[-1]


def simulate_realistic(sigs: pd.DataFrame, bars, stop_pts, target_pts, hold_min):
    """One-position-at-a-time: skip any signal firing while a previous trade
    (from either side) is still open — matches the live bot's no_position
    gate. Without this, overlapping signals get double/triple-counted as if
    the account could hold unlimited concurrent positions."""
    outcomes, pnls = [], []
    busy_until = None
    for _, row in sigs.sort_values("ts").iterrows():
        if busy_until is not None and row["ts"] <= busy_until:
            continue
        o, p, exit_ts = simulate_trade(row, bars, stop_pts, target_pts, hold_min)
        if o is None:
            continue
        outcomes.append(o); pnls.append(p)
        busy_until = exit_ts
    return outcomes, pnls


def summarize(label, outcomes, pnls):
    n = len(pnls)
    if n == 0:
        return None
    wr  = sum(1 for p in pnls if p > 0) / n
    ev  = np.mean(pnls)
    total = sum(pnls)
    top5 = sum(sorted(pnls, reverse=True)[:5])
    return {"label": label, "n": n, "wr": wr, "ev": ev, "total": total,
           "top5_share": (top5 / total) if total != 0 else np.nan}


def main():
    print("Loading data…")
    breakouts = load_breakouts(window="live")   # 9:40-13:00 ET, matches live BA_BRK_TRADE_START/END
    bars      = load_bars()
    day_opens = compute_day_opens(bars)
    print(f"  live-window (9:40-13:00 ET) MES breakout events: {len(breakouts)}  "
          f"({breakouts['ts'].min().date()} -> {breakouts['ts'].max().date()})")
    print(f"  MES 1-min bars: {len(bars):,}  "
          f"({bars['ts'].min().date()} -> {bars['ts'].max().date()})")
    print(f"  stop={STOP_PTS}pt  target={TARGET_PTS:.0f}pt  hold={HOLD_MIN}min  "
          f"(matches live BA_BRK_* config) + day-open alignment filter + "
          f"one-position-at-a-time serialization\n")

    # ── Primary ask: PRIMARY FOCUS — cascade_min robustness (this is the "live"
    # strategy's actual active side; bid uses different stop/target/hold) ──────
    print(f"{'='*82}")
    print(f"ASK-SIDE cascade_min ROBUSTNESS (the question that matters): "
          f"stop={STOP_PTS}pt target={TARGET_PTS:.0f}pt hold={HOLD_MIN}min, "
          f"max_gap={BA_BRK_LIVE_MAX_GAP}s (live value)")
    print(f"{'='*82}")
    for cmin in CASCADE_MINS:
        sigs = detect_cascades(breakouts, "ask", cascade_min=cmin,
                               max_gap_sec=BA_BRK_LIVE_MAX_GAP, day_opens=day_opens)
        outcomes, pnls = simulate_realistic(sigs, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
        r = summarize(f"cascade_min={cmin}", outcomes, pnls)
        if r:
            print(f"  cascade_min={cmin}  n={r['n']:5d}  WR={r['wr']:6.1%}  "
                  f"EV={r['ev']:+6.2f}pt  total={r['total']:+8.1f}pt  "
                  f"top5_share={r['top5_share']*100:6.0f}%")
        else:
            print(f"  cascade_min={cmin}  no trades")

    # ── Baseline: single wall break (cascade_min=1) for comparison ──────────
    print(f"\n{'='*82}")
    print(f"BASELINE  single breakout (cascade_min=1), one-at-a-time, aligned")
    print(f"{'='*82}")
    for side in ["ask", "bid"]:
        sigs = detect_cascades(breakouts, side, cascade_min=1, max_gap_sec=1, day_opens=day_opens)
        outcomes, pnls = simulate_realistic(sigs, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
        r = summarize(f"side={side}", outcomes, pnls)
        if r:
            print(f"  {r['label']:<10} n={r['n']:5d}  WR={r['wr']:6.1%}  "
                  f"EV={r['ev']:6.2f}pts  total={r['total']:8.1f}pts  top5_share={r['top5_share']*100:.0f}%")

    # ── Cascade x gap sweep (ask side, aligned, one-at-a-time) ──────────────
    print(f"\n{'='*82}")
    print(f"CASCADE x GAP sweep (ask side)  stop={STOP_PTS}pt  target={TARGET_PTS:.0f}pt  hold={HOLD_MIN}min")
    print(f"{'='*82}")
    print(f"  {'cascade_min':>11} {'max_gap':>8} {'n':>5} {'WR':>7} {'EV':>7} {'total':>8} {'top5%':>6}")

    results = []
    for cmin in CASCADE_MINS:
        for gap in MAX_GAP_SECS:
            sigs = detect_cascades(breakouts, "ask", cascade_min=cmin, max_gap_sec=gap, day_opens=day_opens)
            outcomes, pnls = simulate_realistic(sigs, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
            r = summarize("ask", outcomes, pnls)
            if r is None:
                continue
            results.append({**r, "cascade_min": cmin, "max_gap_sec": gap})
            print(f"  {cmin:11d} {gap:8d} {r['n']:5d} {r['wr']:7.1%} "
                  f"{r['ev']:7.2f} {r['total']:8.1f} {r['top5_share']*100:6.0f}")

    r = pd.DataFrame(results)
    print(f"\n{'='*82}")
    print("Top 15 by total profit (min 20 trades)")
    print(f"{'='*82}")
    top = r[r["n"] >= 20].sort_values("total", ascending=False).head(15)
    for _, row in top.iterrows():
        print(f"  cascade_min={row['cascade_min']}  max_gap={row['max_gap_sec']}s  "
              f"n={row['n']:.0f}  WR={row['wr']:.1%}  "
              f"EV={row['ev']:.2f}pts  total={row['total']:.1f}pts  top5_share={row['top5_share']*100:.0f}%")

    # ── Combined (both sides, each side's own live stop/target/hold) ────────
    print(f"\n{'='*82}")
    print("Combined (both sides, bid uses its own live stop/target/hold) at each cascade_min, max_gap=30s")
    print(f"{'='*82}")
    for cmin in CASCADE_MINS:
        all_sigs = []
        for side, sp, tp, hm in [("ask", STOP_PTS, TARGET_PTS, HOLD_MIN),
                                 ("bid", BA_BRK_BID_STOP, BA_BRK_BID_TARGET, BA_BRK_BID_HOLD)]:
            sigs = detect_cascades(breakouts, side, cascade_min=cmin, max_gap_sec=30, day_opens=day_opens)
            sigs = sigs.assign(_stop=sp, _target=tp, _hold=hm)
            all_sigs.append(sigs)
        combined = pd.concat(all_sigs, ignore_index=True).sort_values("ts") if all_sigs else pd.DataFrame()
        outcomes, pnls = [], []
        busy_until = None
        for _, row in combined.iterrows():
            if busy_until is not None and row["ts"] <= busy_until:
                continue
            o, p, exit_ts = simulate_trade(row, bars, row["_stop"], row["_target"], int(row["_hold"]))
            if o is None:
                continue
            outcomes.append(o); pnls.append(p)
            busy_until = exit_ts
        rr = summarize(f"cascade_min={cmin}", outcomes, pnls)
        if rr:
            print(f"  cascade_min={cmin}  n={rr['n']:5d}  WR={rr['wr']:6.1%}  "
                  f"EV={rr['ev']:6.2f}pts  total={rr['total']:8.1f}pts  "
                  f"top5_share={rr['top5_share']*100:.0f}%")


if __name__ == "__main__":
    main()
