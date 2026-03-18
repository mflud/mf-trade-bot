"""
Hold-period sweep: tests MAX_BARS_HOLD from 1 to 8 (5–40 minutes) for
each instrument, using the full current signal configuration:
  - Dynamic CSR window (4 bars if GK vol < 8%, else 8 bars)
  - Per-instrument blackout windows (ET-aware)
  - stop=2.0σ / target=3.0σ

Usage:
  python src/backtest_hold_sweep.py
  python src/backtest_hold_sweep.py --sym MES
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Signal parameters ─────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

TF            = 5
TRAILING_BARS = 20
MIN_SCALED    = 3.0
MAX_SCALED    = 5.0
MIN_VOL_RATIO = 1.5
CSR_THRESHOLD = 1.5
GK_VOL_BARS   = 20
BARS_PER_YEAR = 252 * 23 * 60 / TF

PRAC_S = 2.0
PRAC_T = 3.0

HOLD_RANGE    = range(1, 13)  # 1–12 bars (5–60 minutes)
MAX_HOLD      = max(HOLD_RANGE)
MAX_CSR_WIN   = 8

# Per-instrument config: (csv_path, csr_vol_windows, blackout_windows)
# blackout_windows: (sh, sm, eh, em, conditional)
INSTRUMENTS = {
    "MES": (
        "mes_hist_1min.csv",
        [(0.08, 4), (1.0, 8)],
        [(8, 0, 9, 0, True)],
    ),
    "MYM": (
        "mym_hist_1min.csv",
        [(0.08, 4), (1.0, 8)],
        [(9, 0, 9, 30, False), (15, 0, 16, 0, False)],
    ),
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_1min(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    return df.sort_values("ts").reset_index(drop=True)


def make_5min_bars(df1):
    df1 = df1.copy()
    df1["gap"] = df1["ts"].diff() > pd.Timedelta(minutes=2)
    records = []
    i, n = 0, len(df1)
    while i < n and df1["ts"].iloc[i].minute % TF != 0:
        i += 1
    while i + TF <= n:
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            while i < n and df1["ts"].iloc[i].minute % TF != 0:
                i += 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[-1],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += TF
    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


def gk_val(o, h, l, c):
    if o <= 0 or h <= 0 or l <= 0 or c <= 0:
        return 0.0
    return 0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2


def get_mom_bars(gk_ann, csr_vol_windows):
    for upper, bars in csr_vol_windows:
        if gk_ann < upper:
            return bars
    return csr_vol_windows[-1][1]


# ── Scan ──────────────────────────────────────────────────────────────────────

def scan(bars, csr_vol_windows, blackout_windows):
    """
    Returns trigger records. Each record stores forward highs, lows, and
    closes for bars i+1 … i+MAX_HOLD, so any hold <= MAX_HOLD can be
    evaluated without re-scanning.
    """
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    opens   = bars["open"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)

    lookback = max(TRAILING_BARS, MAX_CSR_WIN, GK_VOL_BARS) + 1

    gk_var = np.array([gk_val(opens[j], highs[j], lows[j], closes[j]) for j in range(n)])

    records = []

    for i in range(lookback, n - MAX_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                          / closes[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or abs(scaled) > MAX_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1

        gk_ann   = math.sqrt(max(0.0, float(gk_var[i - GK_VOL_BARS: i].mean())) * BARS_PER_YEAR)
        mom_bars = get_mom_bars(gk_ann, csr_vol_windows)

        prior_rets = np.log(closes[i - mom_bars: i] / closes[i - mom_bars - 1: i - 1])
        csr = float(prior_rets.sum()) / sigma * direction

        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        blocked = False
        for sh, sm, eh, em, conditional in blackout_windows:
            if (sh, sm) <= bar_hm < (eh, em):
                if not conditional or csr < CSR_THRESHOLD:
                    blocked = True
                    break
        if blocked or csr < CSR_THRESHOLD:
            continue

        records.append({
            "entry":       closes[i],
            "sigma":       sigma,
            "direction":   direction,
            "fwd_highs":   highs[i + 1:  i + MAX_HOLD + 1].copy(),
            "fwd_lows":    lows[i + 1:   i + MAX_HOLD + 1].copy(),
            "fwd_closes":  closes[i + 1: i + MAX_HOLD + 1].copy(),
        })

    return records


# ── EV for a given hold period ────────────────────────────────────────────────

def ev_for_hold(records, hold, s=PRAC_S, t=PRAC_T):
    if not records:
        return {"n": 0, "ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan")}

    n = len(records)
    p_tgt_count = p_stop_count = neither_count = 0
    neither_ev_sum = 0.0

    for r in records:
        entry     = r["entry"]
        sigma     = r["sigma"]
        direction = r["direction"]
        tgt_p  = entry * math.exp( direction * t * sigma)
        stop_p = entry * math.exp(-direction * s * sigma)

        hit_tgt = hit_stop = None
        for j in range(hold):
            h = r["fwd_highs"][j]
            l = r["fwd_lows"][j]
            if hit_tgt is None:
                if direction == 1 and h >= tgt_p:    hit_tgt  = j
                elif direction == -1 and l <= tgt_p:  hit_tgt  = j
            if hit_stop is None:
                if direction == 1 and l <= stop_p:   hit_stop = j
                elif direction == -1 and h >= stop_p: hit_stop = j

        tgt_first  = hit_tgt  is not None and (hit_stop is None or hit_tgt  <= hit_stop)
        stop_first = hit_stop is not None and (hit_tgt  is None or hit_stop <  hit_tgt)

        if tgt_first:
            p_tgt_count += 1
        elif stop_first:
            p_stop_count += 1
        else:
            neither_count += 1
            exit_close = r["fwd_closes"][hold - 1]
            time_ret = math.log(exit_close / entry) * direction / sigma
            neither_ev_sum += time_ret

    p_tgt  = p_tgt_count  / n
    p_stop = p_stop_count / n
    ev_nei = neither_ev_sum / neither_count if neither_count else 0.0
    ev     = p_tgt * t - p_stop * s + (neither_count / n) * ev_nei

    return {"n": n, "ev": ev, "p_tgt": p_tgt, "p_stop": p_stop}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None)
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, (path, csr_windows, blackout_wins) in syms.items():
        print(f"\n{'═'*68}")
        print(f"  {sym}  —  Hold-Period Sweep  (stop={PRAC_S}σ / target={PRAC_T}σ)")
        print(f"  Dynamic CSR + per-instrument blackouts")
        print(f"{'═'*68}")

        bars = make_5min_bars(load_1min(path))
        print(f"  {len(bars):,} 5-min bars — scanning …", end=" ", flush=True)
        records = scan(bars, csr_windows, blackout_wins)
        print(f"{len(records):,} triggers (CSR-filtered, blackouts applied)")

        print(f"\n  {'Hold':>10}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>7}  {'EV':>9}")
        print(f"  {'─'*52}")

        evs = {}
        best_ev, best_hold = -999.0, 3
        for h in HOLD_RANGE:
            p = ev_for_hold(records, h)
            evs[h] = p["ev"]
            flag = " ◄ current" if h == 3 else ""
            if not math.isnan(p["ev"]) and p["ev"] > best_ev:
                best_ev   = p["ev"]
                best_hold = h
            print(f"  {h}bar ({h*TF:>2}min)  {p['n']:>5,}  "
                  f"{p['p_tgt']:>7.3f}  {p['p_stop']:>7.3f}  "
                  f"{p['ev']:>+9.4f}σ{flag}")

        print(f"\n  Best: {best_hold} bars ({best_hold * TF} min)  EV={best_ev:+.4f}σ", end="")
        if best_hold != 3:
            print(f"  (vs current 3-bar: {best_ev - evs[3]:+.4f}σ)", end="")
        print()
