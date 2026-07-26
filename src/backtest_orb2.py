"""
ORB backtest v2 — clean sweep on MES 1-min bars from bars.db.

Opening range = high/low of first N minutes after 9:30 ET.
Entry = first 1-min bar whose CLOSE breaks outside the range.
Stop  = midpoint of range (half-range risk — confirmed best in sweep).
Target = 1× range width (confirmed best in sweep).
Entry window = 60 min after ORB close (9:45–10:45 for 15-min ORB).
Width filter = 0.15%–0.5% of price.

Trend filters swept:
  none       — no filter (baseline)
  gap        — only trade in direction of overnight gap (open vs prior RTH close)
  gap_fade   — only trade AGAINST the gap (fade it)
  pm         — only trade in direction of pre-market momentum
               (9:30 open vs last 5-min bar before open)
  prev_rth   — only trade in direction of prior RTH session (close vs open)
  orb_pos    — only trade in direction price is relative to yesterday's close
               at the time the ORB forms (open > prev_close → long only)
  above_ema  — only LONG when 9:30 open is above 20-session EMA of prior closes;
               only SHORT when below

Usage:
    python src/backtest_orb2.py
"""

import sqlite3
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz

# ── Config ────────────────────────────────────────────────────────────────────

BARS_DB = "data/bars.db"
SYMBOL  = "MES"
ET      = pytz.timezone("America/New_York")

RTH_OPEN_HM  = 9 * 60 + 30
RTH_CLOSE_HM = 16 * 60

# Fixed best params from sweep
ORB_MIN      = 15
STOP_TYPE    = "half"     # stop at midpoint of range
TARGET_MULT  = 1.0        # target = 1× range width
ENTRY_WIN    = 60         # minutes after ORB close
WIDTH_MIN    = 0.0015     # 0.15% of price
WIDTH_MAX    = 0.005      # 0.50% of price

TREND_FILTERS = ["none", "gap", "gap_fade", "pm", "prev_rth", "orb_pos", "above_ema"]
EMA_N = 20   # sessions for EMA of prior closes


# ── Load data ─────────────────────────────────────────────────────────────────

def load_bars(minutes: int) -> pd.DataFrame:
    conn = sqlite3.connect(BARS_DB)
    df = pd.read_sql(
        f"SELECT ts,open,high,low,close FROM bars "
        f"WHERE symbol='{SYMBOL}' AND minutes={minutes} ORDER BY ts",
        conn)
    conn.close()
    df["ts"]    = pd.to_datetime(df["ts"], utc=True)
    df["ts_et"] = df["ts"].dt.tz_convert(ET)
    df["hm"]    = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df["date"]  = df["ts_et"].dt.date
    return df


# ── Build per-session context ─────────────────────────────────────────────────

def build_session_context(bars1m: pd.DataFrame, bars5m: pd.DataFrame) -> dict:
    """Return dict date → context with trend filter values."""
    rth1 = bars1m[(bars1m["hm"] >= RTH_OPEN_HM) & (bars1m["hm"] < RTH_CLOSE_HM)]
    pre5 = bars5m[bars5m["hm"] < RTH_OPEN_HM]   # pre-market 5-min bars

    dates = sorted(rth1["date"].unique())
    ctx   = {}
    prev_rth_close  = None
    prev_rth_open   = None
    ema_closes: list[float] = []   # rolling list of prior RTH closes

    for d in dates:
        sess    = rth1[rth1["date"] == d]
        pre_d   = pre5[pre5["date"] == d]
        if sess.empty:
            continue

        rth_open  = sess["open"].iloc[0]
        rth_close = sess["close"].iloc[-1]

        # Pre-market: last 5-min bar before 9:30
        pm_last = pre_d["close"].iloc[-1] if len(pre_d) else None

        # EMA of prior RTH closes (computed before adding today's close)
        ema = float(np.mean(ema_closes[-EMA_N:])) if len(ema_closes) >= 5 else None

        ctx[d] = {
            "rth_open":       rth_open,
            "prev_rth_close": prev_rth_close,
            "prev_rth_open":  prev_rth_open,
            "pm_last":        pm_last,
            "prior_ema":      ema,
        }

        # Update for next session
        ema_closes.append(rth_close)
        prev_rth_close = rth_close
        prev_rth_open  = rth_open

    return ctx


# ── Trend filter ──────────────────────────────────────────────────────────────

def allowed_direction(trade_dir: int, ctx: dict, filt: str) -> bool:
    """Return True if trade_dir is allowed under the given trend filter."""
    if filt == "none":
        return True

    prev_close = ctx["prev_rth_close"]
    rth_open   = ctx["rth_open"]
    pm_last    = ctx["pm_last"]
    prev_open  = ctx["prev_rth_open"]
    ema        = ctx["prior_ema"]

    if filt == "gap":
        # Trade in direction of overnight gap
        if prev_close is None: return True
        gap_dir = 1 if rth_open > prev_close else -1
        return trade_dir == gap_dir

    if filt == "gap_fade":
        # Trade against the overnight gap
        if prev_close is None: return True
        gap_dir = 1 if rth_open > prev_close else -1
        return trade_dir == -gap_dir

    if filt == "pm":
        # Trade in direction of pre-market momentum (open vs last pre-mkt bar)
        if pm_last is None: return True
        pm_dir = 1 if rth_open > pm_last else -1
        return trade_dir == pm_dir

    if filt == "prev_rth":
        # Trade in direction of prior RTH session (close vs open)
        if prev_close is None or prev_open is None: return True
        prev_dir = 1 if prev_close > prev_open else -1
        return trade_dir == prev_dir

    if filt == "orb_pos":
        # Trade in direction price is displaced vs prior close
        if prev_close is None: return True
        dir_vs_prev = 1 if rth_open >= prev_close else -1
        return trade_dir == dir_vs_prev

    if filt == "above_ema":
        if ema is None: return True
        if trade_dir == 1:  return rth_open >= ema
        if trade_dir == -1: return rth_open <  ema

    return True


# ── Per-session simulation ────────────────────────────────────────────────────

def simulate_session(session_bars: pd.DataFrame, ctx: dict, filt: str):
    bars     = session_bars.reset_index(drop=True)
    orb_end  = RTH_OPEN_HM + ORB_MIN
    orb_bars = bars[bars["hm"] < orb_end]
    if len(orb_bars) < ORB_MIN - 1:
        return None

    orb_high  = orb_bars["high"].max()
    orb_low   = orb_bars["low"].min()
    orb_width = orb_high - orb_low
    mid_price = (orb_high + orb_low) / 2
    orb_mid   = (orb_high + orb_low) / 2

    width_frac = orb_width / mid_price
    if width_frac < WIDTH_MIN or width_frac > WIDTH_MAX:
        return None

    stop_dist   = orb_width / 2    # half-range stop
    target_dist = orb_width * TARGET_MULT

    entry_bars = bars[
        (bars["hm"] >= orb_end) &
        (bars["hm"] <  orb_end + ENTRY_WIN)
    ]

    trade_dir   = None
    entry_price = None
    entry_ts    = None

    for _, bar in entry_bars.iterrows():
        if bar["close"] > orb_high:
            trade_dir = 1; entry_price = bar["close"]; entry_ts = bar["ts_et"]; break
        if bar["close"] < orb_low:
            trade_dir = -1; entry_price = bar["close"]; entry_ts = bar["ts_et"]; break

    if trade_dir is None:
        return None

    # Apply trend filter
    if not allowed_direction(trade_dir, ctx, filt):
        return None

    stop_price   = entry_price - trade_dir * stop_dist
    target_price = entry_price + trade_dir * target_dist

    post_bars  = bars[bars["ts_et"] > entry_ts]
    outcome    = "TIME"
    exit_price = post_bars["close"].iloc[-1] if len(post_bars) else entry_price

    for _, bar in post_bars.iterrows():
        if trade_dir == 1:
            if bar["low"]  <= stop_price:   outcome = "STOPPED"; exit_price = stop_price; break
            if bar["high"] >= target_price: outcome = "TARGET";  exit_price = target_price; break
        else:
            if bar["high"] >= stop_price:   outcome = "STOPPED"; exit_price = stop_price; break
            if bar["low"]  <= target_price: outcome = "TARGET";  exit_price = target_price; break

    pnl = (exit_price - entry_price) * trade_dir
    return {
        "date":      session_bars["date"].iloc[0],
        "direction": "LONG" if trade_dir == 1 else "SHORT",
        "orb_width": round(orb_width, 2),
        "orb_mid":   round(orb_mid, 2),
        "entry":     round(entry_price, 2),
        "outcome":   outcome,
        "pnl":       round(pnl, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading bars…")
    bars1m = load_bars(1)
    bars5m = load_bars(5)

    rth1 = bars1m[(bars1m["hm"] >= RTH_OPEN_HM) & (bars1m["hm"] < RTH_CLOSE_HM)]
    sessions = sorted(rth1["date"].unique())
    print(f"  Sessions: {len(sessions)}  ({sessions[0]} → {sessions[-1]})")

    ctx      = build_session_context(bars1m, bars5m)
    by_date  = {d: g for d, g in rth1.groupby("date")}

    # ── Width distribution ────────────────────────────────────────────────────
    widths = []
    for d, bars in by_date.items():
        orb = bars[bars["hm"] < RTH_OPEN_HM + ORB_MIN]
        if len(orb) >= ORB_MIN - 1:
            h = orb["high"].max(); l = orb["low"].min()
            widths.append((h - l) / ((h + l) / 2) * 100)
    print(f"\n15-min ORB width (% of price): "
          f"p25={np.percentile(widths,25):.3f}%  p50={np.percentile(widths,50):.3f}%  "
          f"p75={np.percentile(widths,75):.3f}%  p90={np.percentile(widths,90):.3f}%  "
          f"max={max(widths):.3f}%")

    # ── Run all filters ───────────────────────────────────────────────────────
    filter_results = {}
    for filt in TREND_FILTERS:
        trades = []
        for d, bars in by_date.items():
            c = ctx.get(d)
            if c is None:
                continue
            rec = simulate_session(bars, c, filt)
            if rec:
                trades.append(rec)
        filter_results[filt] = trades

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Filter comparison  (ORB={ORB_MIN}min  stop=half  target=1×  entry_win={ENTRY_WIN}min)")
    print(f"{'='*70}")
    print(f"  {'filter':>10}  {'n':>4}  {'WR':>6}  {'EV':>6}  {'total':>7}  "
          f"{'avg_win':>8}  {'avg_los':>8}  {'PF':>5}")

    summary = []
    for filt in TREND_FILTERS:
        trades = filter_results[filt]
        if not trades:
            continue
        pnls = [t["pnl"] for t in trades]
        wr   = sum(1 for t in trades if t["outcome"] == "TARGET") / len(trades)
        ev   = np.mean(pnls)
        tot  = np.sum(pnls)
        wins = [p for p in pnls if p > 0]
        loss = [p for p in pnls if p < 0]
        aw   = np.mean(wins) if wins else 0
        al   = np.mean(loss) if loss else 0
        pf   = -aw / al if al < 0 else float("inf")
        summary.append((filt, len(trades), wr, ev, tot, aw, al, pf))
        print(f"  {filt:>10}  {len(trades):4d}  {wr:6.1%}  {ev:6.2f}  {tot:7.2f}  "
              f"{aw:8.2f}  {al:8.2f}  {pf:5.2f}")

    # ── Monthly breakdown for each filter ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("Monthly EV breakdown by filter")
    print(f"{'='*70}")

    months_all = sorted(set(
        pd.to_datetime(str(t["date"])).to_period("M")
        for trades in filter_results.values()
        for t in trades
    ))

    header = f"  {'month':>8}" + "".join(f"  {f:>9}" for f in TREND_FILTERS)
    print(header)

    month_data = {}
    for filt in TREND_FILTERS:
        trades = filter_results[filt]
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        tdf["month"] = pd.to_datetime(tdf["date"].astype(str)).dt.to_period("M")
        by_m = tdf.groupby("month")["pnl"].sum()
        month_data[filt] = by_m

    for mo in months_all:
        row = f"  {str(mo):>8}"
        for filt in TREND_FILTERS:
            val = month_data.get(filt, pd.Series(dtype=float)).get(mo, None)
            row += f"  {val:9.1f}" if val is not None else f"  {'—':>9}"
        print(row)

    # ── Direction breakdown per filter ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Long vs Short breakdown per filter")
    print(f"{'='*70}")
    print(f"  {'filter':>10}  {'L_n':>4}  {'L_WR':>6}  {'L_EV':>6}  "
          f"{'S_n':>4}  {'S_WR':>6}  {'S_EV':>6}")
    for filt in TREND_FILTERS:
        trades = filter_results[filt]
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        longs  = tdf[tdf["direction"] == "LONG"]
        shorts = tdf[tdf["direction"] == "SHORT"]
        l_wr = (longs["outcome"] == "TARGET").mean() if len(longs) else 0
        s_wr = (shorts["outcome"] == "TARGET").mean() if len(shorts) else 0
        l_ev = longs["pnl"].mean() if len(longs) else 0
        s_ev = shorts["pnl"].mean() if len(shorts) else 0
        print(f"  {filt:>10}  {len(longs):4d}  {l_wr:6.1%}  {l_ev:6.2f}  "
              f"{len(shorts):4d}  {s_wr:6.1%}  {s_ev:6.2f}")

    # ── Best filter: per-trade detail ─────────────────────────────────────────
    best_filt = max(summary, key=lambda x: x[3])[0]
    print(f"\n{'='*70}")
    print(f"Best filter: '{best_filt}' — trade list")
    print(f"{'='*70}")
    tdf = pd.DataFrame(filter_results[best_filt])
    print(tdf[["date","direction","orb_width","outcome","pnl"]].to_string(index=False))

    # ── Combined: best filter + long-only ────────────────────────────────────
    print(f"\n{'='*70}")
    print("Long-only variants (all filters)")
    print(f"{'='*70}")
    print(f"  {'filter':>10}  {'n':>4}  {'WR':>6}  {'EV':>6}  {'total':>7}")
    for filt in TREND_FILTERS:
        trades = [t for t in filter_results[filt] if t["direction"] == "LONG"]
        if len(trades) < 10:
            continue
        pnls = [t["pnl"] for t in trades]
        wr   = sum(1 for t in trades if t["outcome"] == "TARGET") / len(trades)
        print(f"  {filt:>10}  {len(trades):4d}  {wr:6.1%}  {np.mean(pnls):6.2f}  {np.sum(pnls):7.2f}")


if __name__ == "__main__":
    main()
