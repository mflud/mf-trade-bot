"""
analyze_wall_ahead_exit.py — Tests the user's discretionary exit heuristic:
"After a trend up, if there is a big wall above the current price and it
does NOT get tested, it's time to close out. If it tests and then breaks
through, it'll keep going."

Uses the existing BA-BRK ask-side long trades (same cascade_min=3 signal,
live stop/target/hold, alignment filter — see backtest_wall_cascade.py) as
the "after a trend up" positions. For each trade, finds the nearest ask
wall ABOVE the entry price that appears during the hold window and
classifies its fate:
  NO_WALL         — no ask wall ever formed above entry during the hold
  UNTESTED        — wall formed, price never entered its touch zone
                    (no 'test' event) before it got pulled or hold ended
  TESTED_BROKE    — a 'breakout' event occurred (with or without a prior
                    'test' — a clean break-through counts as "broke")
  TESTED_REJECTED — 'test' event(s) occurred but no breakout — wall held
  STILL_OPEN      — wall never resolved before the hold window ended (censored)

Then compares actual trade P&L by category.

Usage:
  python src/analyze_wall_ahead_exit.py
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from backtest_wall_cascade import (  # noqa: E402
    load_breakouts, load_bars, compute_day_opens, detect_cascades,
    simulate_realistic, simulate_trade,
    STOP_PTS, TARGET_PTS, HOLD_MIN, BA_BRK_LIVE_MAX_GAP, WALL_EVENTS_CSV, ET,
)

CASCADE_MIN = 3


def load_all_wall_events() -> pd.DataFrame:
    df = pd.read_csv(WALL_EVENTS_CSV, header=None,
        names=["ts", "symbol", "side", "wall_price", "wall_size", "peak_size",
               "event", "test_count", "price_at_event"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["wall_price"] = pd.to_numeric(df["wall_price"], errors="coerce")
    return df[df["symbol"] == "MES"].sort_values("ts").reset_index(drop=True)


def classify_wall_ahead(events: pd.DataFrame, entry_ts, entry_price: float,
                        window_end) -> tuple[str, float]:
    """Returns (category, wall_price) for the nearest ask wall above
    entry_price that appears between entry_ts and window_end."""
    window = events[(events["ts"] >= entry_ts) & (events["ts"] <= window_end) &
                    (events["side"] == "ask") & (events["wall_price"] > entry_price)]
    if window.empty:
        return "NO_WALL", np.nan

    # nearest wall = smallest wall_price among wall_found events in the window
    founds = window[window["event"] == "wall_found"]
    if founds.empty:
        return "NO_WALL", np.nan
    nearest_price = founds.sort_values("wall_price")["wall_price"].iloc[0]

    lifecycle = window[np.isclose(window["wall_price"], nearest_price, atol=0.01)].sort_values("ts")
    events_seen = set(lifecycle["event"])
    if "breakout" in events_seen:
        return "TESTED_BROKE", nearest_price
    if "test" in events_seen:
        if "wall_pulled" in events_seen:
            return "TESTED_REJECTED", nearest_price
        return "STILL_OPEN", nearest_price   # tested, not yet resolved by window end
    if "wall_pulled" in events_seen:
        return "UNTESTED", nearest_price
    return "STILL_OPEN", nearest_price


def main():
    breakouts = load_breakouts(window="live")
    bars = load_bars()
    day_opens = compute_day_opens(bars)
    wall_events = load_all_wall_events()

    sigs = detect_cascades(breakouts, "ask", cascade_min=CASCADE_MIN,
                           max_gap_sec=BA_BRK_LIVE_MAX_GAP, day_opens=day_opens)
    print(f"BA-BRK ask signals (cascade_min={CASCADE_MIN}): {len(sigs)}")

    # Realistic one-at-a-time filtering first (same trades as the live strategy
    # would actually take), THEN classify each by its wall-ahead fate.
    sigs = sigs.sort_values("ts").reset_index(drop=True)
    rows = []
    busy_until = None
    for _, row in sigs.iterrows():
        if busy_until is not None and row["ts"] <= busy_until:
            continue
        outcome, pnl, exit_ts = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
        if outcome is None:
            continue
        busy_until = exit_ts
        window_end = row["ts"] + pd.Timedelta(minutes=HOLD_MIN)
        category, wall_price = classify_wall_ahead(wall_events, row["ts"], row["entry"], window_end)
        rows.append(dict(ts=row["ts"], entry=row["entry"], outcome=outcome, pnl=pnl,
                         category=category, wall_price=wall_price))

    df = pd.DataFrame(rows)
    print(f"Trades classified: {len(df)}\n")

    print("=" * 80)
    print("P&L BY WALL-AHEAD CATEGORY")
    print("=" * 80)
    summary = df.groupby("category")["pnl"].agg(["count", "mean", "sum"])
    summary["win_rate"] = df.groupby("category")["pnl"].apply(lambda p: (p > 0).mean())
    summary["outcome_target_rate"] = df.groupby("category")["outcome"].apply(lambda o: (o == "TARGET").mean())
    print(summary.to_string())

    print(f"\n{'='*80}\nOUTCOME BREAKDOWN BY CATEGORY\n{'='*80}")
    print(pd.crosstab(df["category"], df["outcome"]))


if __name__ == "__main__":
    main()


# ── NEW standalone entry signal: untested wall gets pulled while price sits
# near it (user's discretionary description, 2026-10-02) ──────────────────

PROXIMITY_PTS = 2.0   # how close price must be to the wall at pull-time


def detect_untested_pull_signals(wall_events: pd.DataFrame, side: str,
                                  day_opens: dict, proximity_pts: float = PROXIMITY_PTS) -> pd.DataFrame:
    direction = 1 if side == "ask" else -1
    events = wall_events[wall_events["side"] == side].sort_values("ts")

    signals = []
    for wall_price, grp in events.groupby("wall_price"):
        grp = grp.sort_values("ts")
        event_types = set(grp["event"])
        if "test" in event_types or "breakout" in event_types:
            continue   # not purely untested
        pulled = grp[grp["event"] == "wall_pulled"]
        if pulled.empty:
            continue
        for _, p in pulled.iterrows():
            if abs(p["price_at_event"] - wall_price) > proximity_pts:
                continue   # price wasn't actually near the wall when it got pulled
            ts_et = p["ts"].tz_convert(ET)
            day_open = day_opens.get(ts_et.date())
            if day_open is None:
                continue
            if side == "ask" and p["price_at_event"] <= day_open:
                continue
            if side == "bid" and p["price_at_event"] >= day_open:
                continue
            signals.append({"ts": p["ts"], "side": side, "direction": direction,
                            "entry": p["price_at_event"], "wall_price": wall_price})
    return pd.DataFrame(signals).sort_values("ts").reset_index(drop=True) if signals else pd.DataFrame()


if __name__ == "__main__" and "--untested-pull" in sys.argv:
    from backtest_wall_cascade import simulate_realistic, STOP_PTS, TARGET_PTS, HOLD_MIN

    bars = load_bars()
    day_opens = compute_day_opens(bars)
    wall_events = load_all_wall_events()

    for side in ["ask", "bid"]:
        sigs = detect_untested_pull_signals(wall_events, side, day_opens)
        print(f"{side}: {len(sigs)} untested-pull signals")
        if sigs.empty:
            continue
        sp, tp, hm = (STOP_PTS, TARGET_PTS, HOLD_MIN) if side == "ask" else (4.0, 18.0, 30)
        outcomes, pnls = simulate_realistic(sigs, bars, sp, tp, hm)
        n = len(pnls)
        if n == 0:
            print("  no trades resolved")
            continue
        wr = sum(1 for p in pnls if p > 0) / n
        total = sum(pnls)
        top5 = sum(sorted(pnls, reverse=True)[:5])
        print(f"  n={n}  WR={wr:.1%}  EV={np.mean(pnls):+.2f}pt  total={total:+.1f}pt  "
              f"top5_share={(top5/total*100) if total else float('nan'):.0f}%")


# ── Management rule backtest: extend hold + move stop to breakeven when an
# untested wall ahead gets pulled WHILE ALREADY IN a BA-BRK cascade trade ──

def find_untested_pull_trigger(events: pd.DataFrame, entry_ts, entry_price: float,
                               window_end) -> "pd.Timestamp | None":
    """Same classification as classify_wall_ahead, but returns the pull
    event's timestamp (not just the category) when the nearest ask wall
    above entry resolves UNTESTED — the trigger moment for the management
    rule below."""
    window = events[(events["ts"] >= entry_ts) & (events["ts"] <= window_end) &
                    (events["side"] == "ask") & (events["wall_price"] > entry_price)]
    if window.empty:
        return None
    founds = window[window["event"] == "wall_found"]
    if founds.empty:
        return None
    nearest_price = founds.sort_values("wall_price")["wall_price"].iloc[0]
    lifecycle = window[np.isclose(window["wall_price"], nearest_price, atol=0.01)].sort_values("ts")
    events_seen = set(lifecycle["event"])
    if "breakout" in events_seen or "test" in events_seen:
        return None   # not the pure-untested case
    pulled = lifecycle[lifecycle["event"] == "wall_pulled"]
    if pulled.empty:
        return None
    return pulled["ts"].iloc[0]


def simulate_managed(row, bars, wall_events, stop_pts, target_pts, hold_min,
                     extend_min: int) -> dict:
    """Baseline trade, but if an untested-wall-ahead pull fires while still
    open: (1) ratchet the stop to breakeven (never worse), (2) extend the
    hold deadline to trigger_ts + extend_min (a fresh window from the
    trigger, not just tacked onto the original deadline)."""
    entry, d = row["entry"], row["direction"]
    stop_price   = entry - d * stop_pts
    target_price = entry + d * target_pts
    deadline = row["ts"] + pd.Timedelta(minutes=hold_min)

    future = bars[bars["ts"] > row["ts"]]
    if future.empty:
        return None

    trigger_ts = find_untested_pull_trigger(wall_events, row["ts"], entry, deadline)
    managed = False

    for _, bar in future.iterrows():
        if trigger_ts is not None and not managed and bar["ts"] >= trigger_ts:
            new_stop = entry   # breakeven
            stop_price = max(stop_price, new_stop) if d == 1 else min(stop_price, new_stop)
            deadline = max(deadline, trigger_ts + pd.Timedelta(minutes=extend_min))
            managed = True
        if bar["ts"] > deadline:
            return {"outcome": "TIME", "pnl": (bar["close"] - entry) * d,
                    "exit_ts": bar["ts"], "managed": managed}
        if d == 1:
            if bar["low"]  <= stop_price:   return {"outcome": "STOPPED", "pnl": stop_price - entry, "exit_ts": bar["ts"], "managed": managed}
            if bar["high"] >= target_price: return {"outcome": "TARGET",  "pnl": target_pts, "exit_ts": bar["ts"], "managed": managed}
        else:
            if bar["high"] >= stop_price:   return {"outcome": "STOPPED", "pnl": entry - stop_price, "exit_ts": bar["ts"], "managed": managed}
            if bar["low"]  <= target_price: return {"outcome": "TARGET",  "pnl": target_pts, "exit_ts": bar["ts"], "managed": managed}
    last = future.iloc[-1]
    return {"outcome": "TIME", "pnl": (last["close"] - entry) * d, "exit_ts": last["ts"], "managed": managed}


if __name__ == "__main__" and "--manage" in sys.argv:
    from backtest_wall_cascade import detect_cascades, STOP_PTS, TARGET_PTS, HOLD_MIN, BA_BRK_LIVE_MAX_GAP

    bars = load_bars()
    day_opens = compute_day_opens(bars)
    wall_events = load_all_wall_events()
    sigs = detect_cascades(load_breakouts(window="live"), "ask", cascade_min=3,
                           max_gap_sec=BA_BRK_LIVE_MAX_GAP, day_opens=day_opens).sort_values("ts").reset_index(drop=True)

    for extend_min in [15, 25, 40]:
        # Serialize ONCE using the managed rule's own exit times (the rule
        # being tested), then compute baseline (unmanaged) pnl on the exact
        # same trade set for a valid like-for-like comparison — never
        # re-serialize separately per variant, or the two runs could end up
        # comparing different trade sets.
        busy_until = None
        rows = []
        for _, row in sigs.iterrows():
            if busy_until is not None and row["ts"] <= busy_until:
                continue
            r = simulate_managed(row, bars, wall_events, STOP_PTS, TARGET_PTS, HOLD_MIN, extend_min)
            if r is None:
                continue
            busy_until = r["exit_ts"]
            base_outcome, base_pnl, _ = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
            rows.append(dict(managed_pnl=r["pnl"], managed=r["managed"],
                             baseline_pnl=base_pnl if base_pnl is not None else 0.0))

        df = pd.DataFrame(rows)
        print(f"\nextend_min={extend_min}  n={len(df)}  managed_trades={df['managed'].sum()}")
        print(f"  ALL trades:     baseline total={df['baseline_pnl'].sum():.1f}pt  "
              f"managed total={df['managed_pnl'].sum():.1f}pt")
        mdf = df[df["managed"]]
        if not mdf.empty:
            print(f"  MANAGED-ONLY (n={len(mdf)}): baseline total={mdf['baseline_pnl'].sum():.1f}pt "
                  f"EV={mdf['baseline_pnl'].mean():+.3f}  WR={(mdf['baseline_pnl']>0).mean():.1%}   |   "
                  f"managed total={mdf['managed_pnl'].sum():.1f}pt "
                  f"EV={mdf['managed_pnl'].mean():+.3f}  WR={(mdf['managed_pnl']>0).mean():.1%}")


# ── Alternative management: widen the target (not breakeven+extend) when
# the untested-pull trigger fires mid-trade — keep the original stop as-is,
# just give the trade more room to run toward a bigger target. ────────────

def simulate_widen_target(row, bars, wall_events, stop_pts, target_pts, hold_min,
                          new_target_pts, extend_min: int) -> dict:
    entry, d = row["entry"], row["direction"]
    stop_price   = entry - d * stop_pts
    target_price = entry + d * target_pts
    deadline = row["ts"] + pd.Timedelta(minutes=hold_min)

    future = bars[bars["ts"] > row["ts"]]
    if future.empty:
        return None

    trigger_ts = find_untested_pull_trigger(wall_events, row["ts"], entry, deadline)
    managed = False

    for _, bar in future.iterrows():
        if trigger_ts is not None and not managed and bar["ts"] >= trigger_ts:
            target_price = entry + d * new_target_pts   # widen only — stop untouched
            deadline = max(deadline, trigger_ts + pd.Timedelta(minutes=extend_min))
            managed = True
        if bar["ts"] > deadline:
            return {"outcome": "TIME", "pnl": (bar["close"] - entry) * d,
                    "exit_ts": bar["ts"], "managed": managed}
        if d == 1:
            if bar["low"]  <= stop_price:   return {"outcome": "STOPPED", "pnl": stop_price - entry, "exit_ts": bar["ts"], "managed": managed}
            if bar["high"] >= target_price: return {"outcome": "TARGET",  "pnl": target_price - entry, "exit_ts": bar["ts"], "managed": managed}
        else:
            if bar["high"] >= stop_price:   return {"outcome": "STOPPED", "pnl": entry - stop_price, "exit_ts": bar["ts"], "managed": managed}
            if bar["low"]  <= target_price: return {"outcome": "TARGET",  "pnl": entry - target_price, "exit_ts": bar["ts"], "managed": managed}
    last = future.iloc[-1]
    return {"outcome": "TIME", "pnl": (last["close"] - entry) * d, "exit_ts": last["ts"], "managed": managed}


if __name__ == "__main__" and "--widen" in sys.argv:
    from backtest_wall_cascade import detect_cascades, STOP_PTS, TARGET_PTS, HOLD_MIN, BA_BRK_LIVE_MAX_GAP

    bars = load_bars()
    day_opens = compute_day_opens(bars)
    wall_events = load_all_wall_events()
    sigs = detect_cascades(load_breakouts(window="live"), "ask", cascade_min=3,
                           max_gap_sec=BA_BRK_LIVE_MAX_GAP, day_opens=day_opens).sort_values("ts").reset_index(drop=True)

    for new_target, extend_min in [(18, 25), (18, 40), (24, 25), (24, 40), (30, 60)]:
        busy_until = None
        rows = []
        for _, row in sigs.iterrows():
            if busy_until is not None and row["ts"] <= busy_until:
                continue
            r = simulate_widen_target(row, bars, wall_events, STOP_PTS, TARGET_PTS, HOLD_MIN, new_target, extend_min)
            if r is None:
                continue
            busy_until = r["exit_ts"]
            base_outcome, base_pnl, _ = simulate_trade(row, bars, STOP_PTS, TARGET_PTS, HOLD_MIN)
            rows.append(dict(managed_pnl=r["pnl"], managed=r["managed"],
                             baseline_pnl=base_pnl if base_pnl is not None else 0.0))

        df = pd.DataFrame(rows)
        mdf = df[df["managed"]]
        print(f"\nnew_target={new_target}  extend_min={extend_min}  "
              f"n={len(df)}  managed_trades={len(mdf)}")
        print(f"  ALL trades:    baseline total={df['baseline_pnl'].sum():.1f}pt   "
              f"widened total={df['managed_pnl'].sum():.1f}pt")
        if not mdf.empty:
            print(f"  MANAGED-ONLY (n={len(mdf)}): baseline total={mdf['baseline_pnl'].sum():.1f}pt "
                  f"EV={mdf['baseline_pnl'].mean():+.3f} WR={(mdf['baseline_pnl']>0).mean():.1%}   |   "
                  f"widened total={mdf['managed_pnl'].sum():.1f}pt "
                  f"EV={mdf['managed_pnl'].mean():+.3f} WR={(mdf['managed_pnl']>0).mean():.1%}")
