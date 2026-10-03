"""
backtest_ba_rev_resweep.py — Re-validate BA-REV with the now-larger
wall_events.csv (Jun-Oct 2026, ~4 months) AND the actual live gating
mechanism.

backtest_wall_reject_trade.py predates the 2026-09-15 redesign — it gates
on an "overnight range" bucket, a concept that was replaced. Live
(trading_bot.py _update_ba_rev_reference/_update_ba_rev_containment)
actually gates like this:
  1. Reference range = 07:00-10:30 ET high/low (price data only).
  2. Trading starts at 10:30 ET. One bar breaking outside [ref_low,
     ref_high] after 10:30 permanently disables BA-REV for the rest of
     that day (cumulative — never "re-contains").
  3. Entry = price_at_event of the Nth test on a wall; direction is
     AWAY from the wall (short on ask-wall rejection, long on bid).
This script replicates that exactly, same mistake-class as BA-BRK's
stale backtest script last time — fix before trusting any numbers.

Runs:
  - Stage 1: stop/target/hold sweep at the live N=3, full sample, with
    the top5_share tail-concentration robustness check.
  - Stage 2: walk-forward train/test (chronological 70/30) on the CURRENT
    live config (stop=2.0/target=6.0/hold=60) and on the stage-1 best
    config, to see whether either actually holds up out-of-sample (BA-BRK
    did NOT, on the stale gate — this checks if BA-REV does, on the
    correct one).
  - Stage 3: N in {2,3,4} sensitivity at the live stop/target/hold, full
    sample AND walk-forward, to answer "would this fall apart at N=2 or
    N=4" the way we checked for BA-BRK's cascade_min.
  - Stage 4: ask vs bid split.

Usage:
  python src/backtest_ba_rev_resweep.py
"""

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz

WALL_EVENTS_CSV = "logs/wall_events.csv"
BARS_DB         = "data/bars.db"
ET              = pytz.timezone("America/New_York")

REF_START_HM   = 7 * 60            # 07:00 ET
LOCK_HM        = 10 * 60 + 30      # 10:30 ET — reference locks, trading starts
TRADE_END_HM   = 16 * 60           # 16:00 ET

LIVE_STOP, LIVE_TARGET, LIVE_HOLD, LIVE_N = 2.0, 6.0, 60, 3

STOP_GRID   = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
TARGET_GRID = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0]
HOLD_GRID   = [20, 30, 45, 60, 90]


def load_bars(symbol: str = "MES") -> pd.DataFrame:
    conn = sqlite3.connect(BARS_DB)
    bars = pd.read_sql(
        f"SELECT ts, high, low, close FROM bars WHERE symbol='{symbol}' AND minutes=1 ORDER BY ts",
        conn)
    conn.close()
    bars["ts"] = pd.to_datetime(bars["ts"], utc=True)
    bars["ts_et"] = bars["ts"].dt.tz_convert(ET)
    bars["date"]  = bars["ts_et"].dt.date
    bars["hm"]    = bars["ts_et"].dt.hour * 60 + bars["ts_et"].dt.minute
    return bars


def load_tests(symbol: str = "MES") -> pd.DataFrame:
    df = pd.read_csv(WALL_EVENTS_CSV, header=None,
        names=["ts", "symbol", "side", "wall_price", "wall_size", "peak_size",
               "event", "test_count", "price_at_event"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["price_at_event"] = pd.to_numeric(df["price_at_event"], errors="coerce")
    df["test_count"] = pd.to_numeric(df["test_count"], errors="coerce")
    t = df[(df["event"] == "test") & (df["symbol"] == symbol) &
           df["price_at_event"].notna()].copy()
    t["ts_et"] = t["ts"].dt.tz_convert(ET)
    t["hm"]    = t["ts_et"].dt.hour * 60 + t["ts_et"].dt.minute
    t["date"]  = t["ts_et"].dt.date
    return t.sort_values("ts").reset_index(drop=True)


def compute_containment(bars: pd.DataFrame) -> dict:
    """Per date: (ref_high, ref_low, first_breach_ts or None) — mirrors
    _update_ba_rev_reference + _update_ba_rev_containment exactly."""
    out = {}
    for date, day in bars.groupby("date"):
        ref = day[(day["hm"] >= REF_START_HM) & (day["hm"] < LOCK_HM)]
        if ref.empty:
            continue
        ref_hi, ref_lo = ref["high"].max(), ref["low"].min()
        after = day[day["hm"] >= LOCK_HM].sort_values("ts")
        breach = after[(after["high"] > ref_hi) | (after["low"] < ref_lo)]
        first_breach_ts = breach["ts"].iloc[0] if not breach.empty else None
        out[date] = (ref_hi, ref_lo, first_breach_ts)
    return out


def build_signals(tests: pd.DataFrame, containment: dict, n: int) -> pd.DataFrame:
    sub = tests[(tests["test_count"] == n) &
               (tests["hm"] >= LOCK_HM) & (tests["hm"] < TRADE_END_HM)].copy()

    def _contained(row):
        c = containment.get(row["date"])
        if c is None:
            return False
        _, _, first_breach_ts = c
        return first_breach_ts is None or row["ts"] < first_breach_ts

    sub = sub[sub.apply(_contained, axis=1)]
    sub["direction"] = sub["side"].map({"ask": -1, "bid": 1})
    sub["entry"] = sub["price_at_event"]
    return sub[["ts", "date", "side", "direction", "entry"]].sort_values("ts").reset_index(drop=True)


def simulate_one(entry_ts, entry_price, direction, bars_ts, bars_high, bars_low, bars_close,
                 stop_pts, target_pts, hold_min):
    idx = np.searchsorted(bars_ts, entry_ts, side="right")
    end_idx = min(idx + hold_min, len(bars_ts))
    if idx >= end_idx:
        return None
    stop_price   = entry_price - direction * stop_pts
    target_price = entry_price + direction * target_pts
    exit_ts = bars_ts[end_idx - 1]
    for i in range(idx, end_idx):
        hi, lo = bars_high[i], bars_low[i]
        if direction == 1:
            if lo <= stop_price:  return {"outcome": "STOPPED", "pnl": -stop_pts, "exit_ts": bars_ts[i]}
            if hi >= target_price: return {"outcome": "TARGET", "pnl": target_pts, "exit_ts": bars_ts[i]}
        else:
            if hi >= stop_price:  return {"outcome": "STOPPED", "pnl": -stop_pts, "exit_ts": bars_ts[i]}
            if lo <= target_price: return {"outcome": "TARGET", "pnl": target_pts, "exit_ts": bars_ts[i]}
    close = bars_close[end_idx - 1]
    return {"outcome": "TIME", "pnl": (close - entry_price) * direction, "exit_ts": exit_ts}


def simulate_realistic(signals: pd.DataFrame, bars: pd.DataFrame,
                       stop_pts, target_pts, hold_min) -> pd.DataFrame:
    bars_ts    = bars["ts"].values
    bars_high  = bars["high"].values
    bars_low   = bars["low"].values
    bars_close = bars["close"].values

    trades = []
    busy_until = None
    for _, s in signals.iterrows():
        s_ts = np.datetime64(s["ts"].tz_convert("UTC").tz_localize(None))
        if busy_until is not None and s_ts < busy_until:
            continue
        r = simulate_one(s_ts, s["entry"], s["direction"],
                         bars_ts, bars_high, bars_low, bars_close,
                         stop_pts, target_pts, hold_min)
        if r is None:
            continue
        trades.append({**r, "ts": s["ts"], "side": s["side"], "direction": s["direction"]})
        busy_until = r["exit_ts"]
    return pd.DataFrame(trades)


def summarize(label, trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"label": label, "n": 0, "wr": np.nan, "ev": np.nan, "total": 0.0, "top5_share": np.nan}
    total = trades["pnl"].sum()
    top5 = trades["pnl"].sort_values(ascending=False).head(5).sum()
    return {"label": label, "n": len(trades), "wr": (trades["pnl"] > 0).mean(),
           "ev": trades["pnl"].mean(), "total": total,
           "top5_share": (top5 / total) if total != 0 else np.nan}


def date_filter(sigs: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    mask = pd.Series(True, index=sigs.index)
    if start is not None: mask &= sigs["date"] >= start
    if end is not None:   mask &= sigs["date"] < end
    return sigs[mask]


def main():
    bars = load_bars()
    tests = load_tests()
    containment = compute_containment(bars)
    dates_with_ref = sorted(containment.keys())
    n_breached = sum(1 for c in containment.values() if c[2] is not None)
    print(f"Days with a locked reference: {len(dates_with_ref)}  "
          f"({dates_with_ref[0]} -> {dates_with_ref[-1]})")
    print(f"Days that breached containment after 10:30: {n_breached}/{len(dates_with_ref)} "
          f"({n_breached/len(dates_with_ref)*100:.0f}%)")

    sig3 = build_signals(tests, containment, LIVE_N)
    print(f"\nN={LIVE_N} contained signals (both sides): {len(sig3)}")

    print(f"\n{'='*100}\nSTAGE 1 — stop/target/hold sweep at N={LIVE_N}, full sample, "
          f"realistic one-at-a-time\n{'='*100}")
    results = []
    for stop in STOP_GRID:
        for target in TARGET_GRID:
            for hold in HOLD_GRID:
                trades = simulate_realistic(sig3, bars, stop, target, hold)
                s = summarize(f"{stop}/{target}/{hold}", trades)
                results.append({**s, "stop": stop, "target": target, "hold": hold})
    rdf = pd.DataFrame(results)
    rdf = rdf[rdf["n"] >= 30]
    print("Top 10 by total (unfiltered):")
    print(rdf.sort_values("total", ascending=False).head(10)
         [["stop", "target", "hold", "n", "wr", "ev", "total", "top5_share"]].to_string(index=False))
    robust = rdf[rdf["top5_share"].abs() <= 0.5]
    print(f"\nTop 10 by total, ROBUST (top5_share<=50%) — {len(robust)} qualify:")
    top_robust = robust.sort_values("total", ascending=False).head(10)
    print(top_robust[["stop", "target", "hold", "n", "wr", "ev", "total", "top5_share"]].to_string(index=False))

    live_cfg = rdf[(rdf["stop"] == LIVE_STOP) & (rdf["target"] == LIVE_TARGET) & (rdf["hold"] == LIVE_HOLD)]
    print(f"\nCurrent LIVE config (stop={LIVE_STOP}/target={LIVE_TARGET}/hold={LIVE_HOLD}):")
    print(live_cfg[["stop", "target", "hold", "n", "wr", "ev", "total", "top5_share"]].to_string(index=False)
         if not live_cfg.empty else "  not in grid")

    best = top_robust.iloc[0] if not top_robust.empty else rdf.sort_values("total", ascending=False).iloc[0]

    print(f"\n{'='*100}\nSTAGE 2 — walk-forward (train 70% / test 30%, chronological)\n{'='*100}")
    cut_idx = int(len(dates_with_ref) * 0.7)
    cut = dates_with_ref[cut_idx]
    print(f"Train: {dates_with_ref[0]} -> {dates_with_ref[cut_idx-1]}   "
          f"Test: {cut} -> {dates_with_ref[-1]}")

    for label, stop, target, hold in [
        ("LIVE config", LIVE_STOP, LIVE_TARGET, LIVE_HOLD),
        ("Stage-1 best", best["stop"], best["target"], best["hold"]),
    ]:
        train_sig = date_filter(sig3, end=cut)
        test_sig  = date_filter(sig3, start=cut)
        tr = summarize("train", simulate_realistic(train_sig, bars, stop, target, hold))
        te = summarize("test",  simulate_realistic(test_sig, bars, stop, target, hold))
        print(f"\n{label} (stop={stop}/target={target}/hold={hold}):")
        print(f"  TRAIN: n={tr['n']} wr={tr['wr']:.3f} ev={tr['ev']:+.3f} total={tr['total']:+.1f} top5={tr['top5_share']*100 if tr['n'] else 0:.0f}%")
        print(f"  TEST:  n={te['n']} wr={te['wr']:.3f} ev={te['ev']:+.3f} total={te['total']:+.1f} top5={te['top5_share']*100 if te['n'] else 0:.0f}%")

    print(f"\n{'='*100}\nSTAGE 3 — N sensitivity (2/3/4) at LIVE stop/target/hold\n{'='*100}")
    for n in [2, 3, 4]:
        sig_n = build_signals(tests, containment, n)
        full = summarize("full", simulate_realistic(sig_n, bars, LIVE_STOP, LIVE_TARGET, LIVE_HOLD))
        train_sig = date_filter(sig_n, end=cut)
        test_sig  = date_filter(sig_n, start=cut)
        tr = summarize("train", simulate_realistic(train_sig, bars, LIVE_STOP, LIVE_TARGET, LIVE_HOLD))
        te = summarize("test",  simulate_realistic(test_sig, bars, LIVE_STOP, LIVE_TARGET, LIVE_HOLD))
        print(f"N={n}  FULL: n={full['n']:4d} wr={full['wr']:.3f} ev={full['ev']:+.3f} "
              f"total={full['total']:+7.1f} top5={full['top5_share']*100 if full['n'] else 0:6.0f}%   |  "
              f"TRAIN: n={tr['n']:4d} ev={tr['ev']:+.3f} total={tr['total']:+7.1f}   |  "
              f"TEST: n={te['n']:4d} ev={te['ev']:+.3f} total={te['total']:+7.1f} top5={te['top5_share']*100 if te['n'] else 0:6.0f}%")

    print(f"\n{'='*100}\nSTAGE 4 — ask vs bid split, N={LIVE_N}, LIVE stop/target/hold\n{'='*100}")
    for side in ["ask", "bid"]:
        sub = sig3[sig3["side"] == side]
        s = summarize(side, simulate_realistic(sub, bars, LIVE_STOP, LIVE_TARGET, LIVE_HOLD))
        print(f"  {side:>4}  n={s['n']:4d}  wr={s['wr']:.3f}  ev={s['ev']:+.3f}  "
              f"total={s['total']:+.1f}  top5_share={s['top5_share']*100 if s['n'] else 0:.0f}%")


if __name__ == "__main__":
    main()
