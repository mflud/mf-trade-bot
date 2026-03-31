"""
Backtest: Heiken-Ashi consecutive-color streak strategy on futures 1-min bars.

Heiken-Ashi construction:
  HA_Close = (O + H + L + C) / 4
  HA_Open  = (prev_HA_Open + prev_HA_Close) / 2
  HA_High  = max(H, HA_Open, HA_Close)
  HA_Low   = min(L, HA_Open, HA_Close)
  Color    = Green if HA_Close >= HA_Open else Red

Strategy A — color-flip exit (--mode flip):
  After N consecutive green HA candles → enter LONG.
  After N consecutive red   HA candles → enter SHORT.
  Exit at first candle whose color changes.

Strategy B — fixed-hold exit (--mode hold, default):
  After N consecutive same-color candles → enter.
  Exit after exactly M candles (M < N).
  Sweeps all valid (N, M) pairs: N=2..8, M=1..N-1.

Entry/exit price: HA_Close (theoretical) or actual bar Close (executable).
HA is reset each session (no overnight carry-over).
1-min source bars are resampled to the requested bar size before HA construction.

Sessions:
  rth    — NYSE regular hours 09:30–16:00 ET  (default)
  globex — CME Globex overnight 18:00–09:29 ET (next calendar day)
           Sessions are delimited by the 18:00 ET Globex open; each runs
           18:00 ET day-D through 09:29 ET day-D+1.

Usage:
  python src/backtest_ha.py                              # MES, 5-min, RTH
  python src/backtest_ha.py --sym all --tf 5             # all three, 5-min
  python src/backtest_ha.py --sym M2K,MYM --tf 5
  python src/backtest_ha.py --tf 5 --show 3:2,8:7       # detail specific combos
  python src/backtest_ha.py --session globex --tf 5
  python src/backtest_ha.py --tf 5 --mode flip
"""

import argparse
import sys
from datetime import time, timedelta, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Session windows ────────────────────────────────────────────────────────────
# RTH: 09:30–16:00 ET
RTH_OPEN  = time(9, 30)
RTH_CLOSE = time(15, 59)
# Globex overnight: 18:00 ET → 09:29 ET next day
GLOBEX_OPEN  = time(18, 0)
GLOBEX_CLOSE = time(9, 29)

HIST = {
    "MES": "mes_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}
N_SWEEP    = list(range(2, 9))   # N=2..8
MIN_TRADES = 30


# ── Data loading ───────────────────────────────────────────────────────────────

def load_rth(sym: str) -> pd.DataFrame:
    """Load 1-min bars filtered to NYSE RTH (09:30–16:00 ET)."""
    path = Path(HIST[sym])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    ts_et    = df["ts"].dt.tz_convert(ET)
    rth_mask = (ts_et.dt.time >= RTH_OPEN) & (ts_et.dt.time <= RTH_CLOSE)
    df = df[rth_mask].copy()
    df["ts_et"] = ts_et[rth_mask]
    df["date"]  = df["ts_et"].dt.date          # trading date = calendar date
    df["session_date"] = df["date"]
    df = df.reset_index(drop=True)
    return df


def load_globex(sym: str) -> pd.DataFrame:
    """Load 1-min bars filtered to Globex overnight (18:00–09:29 ET).
    Each 'session' is keyed by the evening date (start of the overnight),
    so Monday 18:00 → Tuesday 09:29 is labelled as Monday's session.
    """
    path = Path(HIST[sym])
    if not path.exists():
        sys.exit(f"File not found: {path}")
    df = pd.read_csv(path, parse_dates=["ts"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

    ts_et = df["ts"].dt.tz_convert(ET)
    t     = ts_et.dt.time
    # Bars at or after 18:00 ET, or before 09:30 ET
    mask  = (t >= GLOBEX_OPEN) | (t <= GLOBEX_CLOSE)
    df    = df[mask].copy()
    df["ts_et"] = ts_et[mask]

    # Assign session_date = calendar date of the 18:00 open
    # Bars before 09:30 belong to the previous calendar day's session
    cal_date = df["ts_et"].dt.date
    is_early = df["ts_et"].dt.time <= GLOBEX_CLOSE   # before 09:30 → prev day
    session_date = pd.to_datetime(cal_date) - pd.to_timedelta(is_early.astype(int), unit="D")
    df["session_date"] = session_date.dt.date
    df["date"]         = df["session_date"]
    df = df.reset_index(drop=True)
    return df


def load_session(sym: str, session: str) -> pd.DataFrame:
    print(f"Loading {HIST[sym]} [{session}] …", flush=True)
    if session == "rth":
        df = load_rth(sym)
    else:
        df = load_globex(sym)
    n_sessions = df["session_date"].nunique()
    print(f"  1-min bars: {len(df):,}  ({n_sessions} sessions,  "
          f"{df['session_date'].min()} → {df['session_date'].max()})")
    return df


# ── Resampling ─────────────────────────────────────────────────────────────────

def resample_bars(df: pd.DataFrame, minutes: int, session: str) -> pd.DataFrame:
    """Resample 1-min bars to clock-aligned N-minute bars within each session."""
    if minutes == 1:
        return df.copy()

    freq = f"{minutes}min"

    # Resample per session so boundaries are respected
    pieces = []
    for sd, grp in df.groupby("session_date"):
        grp2 = grp.set_index("ts_et")
        rs   = grp2.resample(freq, closed="left", label="left").agg(
            open  =("open",  "first"),
            high  =("high",  "max"),
            low   =("low",   "min"),
            close =("close", "last"),
        ).dropna(subset=["close"])
        rs = rs.reset_index()
        rs["session_date"] = sd
        rs["date"]         = sd
        pieces.append(rs)

    out = pd.concat(pieces, ignore_index=True)

    # For RTH, drop bars that start outside the window
    if session == "rth":
        t    = out["ts_et"].dt.time
        out  = out[(t >= RTH_OPEN) & (t <= RTH_CLOSE)].reset_index(drop=True)

    return out


# ── HA construction ────────────────────────────────────────────────────────────

def compute_ha(df: pd.DataFrame) -> pd.DataFrame:
    ha_open  = np.empty(len(df))
    ha_close = ((df["open"] + df["high"] + df["low"] + df["close"]).values / 4)
    ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum(df["high"].values, np.maximum(ha_open, ha_close))
    ha_low  = np.minimum(df["low"].values,  np.minimum(ha_open, ha_close))
    out = df.copy()
    out["ha_open"]  = ha_open
    out["ha_close"] = ha_close
    out["ha_high"]  = ha_high
    out["ha_low"]   = ha_low
    out["green"]    = ha_close >= ha_open
    return out


# ── Backtests ──────────────────────────────────────────────────────────────────

def _ep(bar, use_real):
    return bar["close"] if use_real else bar["ha_close"]


def run_flip(df: pd.DataFrame, n: int, use_real: bool) -> pd.DataFrame:
    records = []
    for date, session in df.groupby("session_date"):
        session = session.reset_index(drop=True)
        if len(session) < n + 2:
            continue
        ha = compute_ha(session)
        bars = ha.to_dict("records")
        nbars = len(bars)
        streak = 0; streak_color = None
        in_trade = False; entry_price = 0.0; direction = 0; entry_idx = 0
        for i, bar in enumerate(bars):
            color = bar["green"]
            if in_trade:
                if color != (direction == 1):
                    pnl = (_ep(bar, use_real) - entry_price) * direction
                    records.append({"date": date,
                                    "direction": "LONG" if direction == 1 else "SHORT",
                                    "n": n, "m": 0,
                                    "pnl": pnl, "hold_bars": i - entry_idx})
                    in_trade = False; streak = 1; streak_color = color
                continue
            if streak_color is None or color != streak_color:
                streak = 1; streak_color = color
            else:
                streak += 1
            if streak == n and i + 1 < nbars:
                entry_price = _ep(bar, use_real); direction = 1 if color else -1
                in_trade = True; entry_idx = i
        if in_trade:
            pnl = (_ep(bars[-1], use_real) - entry_price) * direction
            records.append({"date": date,
                            "direction": "LONG" if direction == 1 else "SHORT",
                            "n": n, "m": 0,
                            "pnl": pnl, "hold_bars": len(bars) - 1 - entry_idx})
    return pd.DataFrame(records)


def run_hold(df: pd.DataFrame, n: int, m: int, use_real: bool) -> pd.DataFrame:
    records = []
    for date, session in df.groupby("session_date"):
        session = session.reset_index(drop=True)
        if len(session) < n + 1:
            continue
        ha = compute_ha(session)
        bars = ha.to_dict("records")
        nbars = len(bars)
        streak = 0; streak_color = None
        in_trade = False; entry_price = 0.0; direction = 0; entry_idx = 0
        for i, bar in enumerate(bars):
            color = bar["green"]
            if in_trade:
                if i == entry_idx + m or i == nbars - 1:
                    pnl = (_ep(bar, use_real) - entry_price) * direction
                    records.append({"date": date,
                                    "direction": "LONG" if direction == 1 else "SHORT",
                                    "n": n, "m": m,
                                    "pnl": pnl, "hold_bars": i - entry_idx})
                    in_trade = False; streak = 1; streak_color = color
                continue
            if streak_color is None or color != streak_color:
                streak = 1; streak_color = color
            else:
                streak += 1
            if streak == n and i + m < nbars:
                entry_price = _ep(bar, use_real); direction = 1 if color else -1
                in_trade = True; entry_idx = i
    return pd.DataFrame(records)


# ── Reporting ──────────────────────────────────────────────────────────────────

def _stats(grp):
    win  = (grp["pnl"] > 0).mean() * 100
    avg  = grp["pnl"].mean()
    tot  = grp["pnl"].sum()
    hold = grp["hold_bars"].mean()
    days = grp["date"].nunique()
    ev_d = tot / days
    return win, avg, tot, hold, ev_d


def _year_table(sub, label):
    sub = sub.copy()
    sub["year"] = pd.to_datetime(sub["date"]).dt.year
    print(f"\n  Year-by-year  {label}:")
    print(f"  {'Year':>6}  {'Trades':>7}  {'Win%':>6}  {'Avg pts':>8}  {'Tot pts':>9}")
    for yr, grp in sub.groupby("year"):
        win, avg, tot, *_ = _stats(grp)
        flag = " ◄" if avg > 0 else ""
        print(f"  {yr:>6}  {len(grp):>7,}  {win:>5.1f}%  {avg:>+8.3f}  {tot:>+9.1f}{flag}")
    print(f"  {'Dir split':>9}")
    for d in ["LONG", "SHORT"]:
        g = sub[sub["direction"] == d]
        if len(g) < MIN_TRADES: continue
        win, avg, tot, *_ = _stats(g)
        flag = " ◄" if avg > 0 else ""
        print(f"    {d:>6}: n={len(g):,}  win={win:.1f}%  avg={avg:+.3f}  total={tot:+.1f}{flag}")


def print_hold_results(all_trades: pd.DataFrame, sym: str, tf: int,
                       session: str, show_combos: list):
    sess_label = "RTH 09:30–16:00 ET" if session == "rth" else "Globex 18:00–09:29 ET"
    print(f"\n{'═'*80}")
    print(f"  {sym}  {tf}-min  HA fixed-hold  —  {sess_label}")
    print(f"{'═'*80}")
    print(f"  {'N':>3}  {'M':>3}  {'Dir':>6}  {'Trades':>7}  {'Win%':>6}  "
          f"{'Avg pts':>8}  {'Tot pts':>9}  {'EV/day':>8}")
    print(f"  {'-'*78}")

    best_avg = -999; best_nm = (0, 0)
    for n in N_SWEEP:
        printed_n = False
        for m in range(1, n):
            sub = all_trades[(all_trades["n"] == n) & (all_trades["m"] == m)]
            if len(sub) < MIN_TRADES:
                continue
            label_n = f"{n:>3}" if not printed_n else "   "
            printed_n = True
            for dirn in ["LONG", "SHORT", "ALL"]:
                grp = sub if dirn == "ALL" else sub[sub["direction"] == dirn]
                if len(grp) < MIN_TRADES:
                    continue
                win, avg, tot, hold, ev_d = _stats(grp)
                flag = " ◄" if avg > 0 else ""
                if dirn == "ALL" and avg > best_avg:
                    best_avg = avg; best_nm = (n, m)
                print(f"  {label_n}  {m:>3}  {dirn:>6}  {len(grp):>7,}  {win:>5.1f}%  "
                      f"{avg:>+8.3f}  {tot:>+9.1f}  {ev_d:>+7.3f}{flag}")
                label_n = "   "
        if printed_n:
            print()

    # Year-by-year for best combo
    if best_nm[0]:
        bn, bm = best_nm
        sub = all_trades[(all_trades["n"] == bn) & (all_trades["m"] == bm)]
        _year_table(sub, f"(N={bn}, M={bm}, ALL — best combo)")

    # Year-by-year for each requested combo
    for n, m in show_combos:
        if (n, m) == best_nm:
            continue   # already printed
        sub = all_trades[(all_trades["n"] == n) & (all_trades["m"] == m)]
        if len(sub) < MIN_TRADES:
            print(f"\n  [N={n}, M={m}]: too few trades ({len(sub)})")
            continue
        win, avg, tot, *_ = _stats(sub)
        _year_table(sub, f"(N={n}, M={m}, ALL — highlighted)")


def print_flip_results(all_trades: pd.DataFrame, sym: str, tf: int,
                       session: str, show_combos: list):
    sess_label = "RTH 09:30–16:00 ET" if session == "rth" else "Globex 18:00–09:29 ET"
    print(f"\n{'═'*80}")
    print(f"  {sym}  {tf}-min  HA color-flip  —  {sess_label}")
    print(f"{'═'*80}")
    print(f"  {'N':>3}  {'Dir':>6}  {'Trades':>7}  {'Win%':>6}  "
          f"{'Avg pts':>8}  {'Hold':>6}  {'Tot pts':>9}  {'EV/day':>8}")
    print(f"  {'-'*78}")
    best_avg = -999; best_n = 0
    for n in N_SWEEP:
        sub = all_trades[all_trades["n"] == n]
        if len(sub) < MIN_TRADES:
            continue
        for dirn in ["LONG", "SHORT", "ALL"]:
            grp = sub if dirn == "ALL" else sub[sub["direction"] == dirn]
            if len(grp) < MIN_TRADES: continue
            win, avg, tot, hold, ev_d = _stats(grp)
            flag = " ◄" if avg > 0 else ""
            if dirn == "ALL" and avg > best_avg:
                best_avg = avg; best_n = n
            print(f"  {n:>3}  {dirn:>6}  {len(grp):>7,}  {win:>5.1f}%  "
                  f"{avg:>+8.3f}  {hold:>5.1f}b  {tot:>+9.1f}  {ev_d:>+7.3f}{flag}")
        print()
    if best_n:
        sub = all_trades[all_trades["n"] == best_n]
        _year_table(sub, f"(N={best_n}, ALL — best)")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_show(s: str) -> list:
    """Parse '3:2,8:7' into [(3,2),(8,7)]."""
    if not s:
        return []
    pairs = []
    for tok in s.split(","):
        n, m = tok.strip().split(":")
        pairs.append((int(n), int(m)))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym",     default="MES",
                        help="Symbol(s): MES, M2K, MYM, or comma-sep list, or 'all'")
    parser.add_argument("--tf",      default="5",
                        help="Bar size(s) in minutes, comma-separated (default: 5)")
    parser.add_argument("--mode",    default="hold", choices=["hold", "flip"])
    parser.add_argument("--session", default="rth",  choices=["rth", "globex"],
                        help="rth (NYSE) or globex (overnight)")
    parser.add_argument("--show",    default="3:2",
                        help="N:M combos to always detail, e.g. '3:2,8:7'")
    args = parser.parse_args()

    syms = list(HIST) if args.sym.lower() == "all" else [s.strip() for s in args.sym.split(",")]
    timeframes  = [int(x.strip()) for x in args.tf.split(",")]
    show_combos = parse_show(args.show)

    for sym in syms:
        df1 = load_session(sym, args.session)

        for tf in timeframes:
            df = resample_bars(df1, tf, args.session)
            n_sess = df["session_date"].nunique()
            print(f"\n  [{sym}  {tf}-min]  {len(df):,} bars  ({n_sess} sessions)")

            for use_real, label in [(False, "HA prices (theoretical)"),
                                    (True,  "Real close prices (executable)")]:
                print(f"\n    ── {label} ──")
                all_trades = []

                if args.mode == "flip":
                    for n in N_SWEEP:
                        print(f"    Scanning N={n} …", end="\r", flush=True)
                        all_trades.append(run_flip(df, n, use_real))
                    print()
                    all_trades = pd.concat(all_trades, ignore_index=True)
                    print_flip_results(all_trades, sym, tf, args.session, show_combos)

                else:
                    pairs = [(n, m) for n in N_SWEEP for m in range(1, n)]
                    for n, m in pairs:
                        print(f"    Scanning N={n}, M={m} …", end="\r", flush=True)
                        all_trades.append(run_hold(df, n, m, use_real))
                    print()
                    all_trades = pd.concat(all_trades, ignore_index=True)
                    print_hold_results(all_trades, sym, tf, args.session, show_combos)

        print()


if __name__ == "__main__":
    main()
