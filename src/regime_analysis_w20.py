"""
Full regime analysis at trail=20 bars (100 min) — the optimal unfiltered window.

Compares trail=20 vs trail=100 (current baseline) across:
  1. Year-by-year EV (checks if the +0.040σ edge is stable across time)
  2. Vol regime (QUIET / NORMAL / ELEVATED / ACTIVE / HIGH VOL)
  3. Session (Overnight / NYSE / CME close)
  4. Various vol filters — is the right filter tighter than 10–20%?

Usage:
  python src/regime_analysis_w20.py            # MES
  python src/regime_analysis_w20.py --sym MYM
"""

import argparse
import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

# ── Config ─────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TF                   = 5
MAX_BARS_HOLD        = 3
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5
BARS_PER_YEAR        = 252 * 23 * 60

PRAC_S, PRAC_T = 1.5, 2.5
STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

VOL_REGIME_BINS   = [0, 0.10, 0.15, 0.20, 0.30, 99]
VOL_REGIME_LABELS = ["QUIET (<10%)", "NORMAL (10–15%)", "ELEVATED (15–20%)",
                     "ACTIVE (20–30%)", "HIGH VOL (>30%)"]

SESSION_LABELS = ["Overnight (pre-NYSE)", "NYSE hours (13:30–20 UTC)",
                  "CME close (20–21 UTC)"]

# Vol filters to compare
VOL_FILTERS = [
    (0.00, 99.0, "No filter"),
    (0.10, 0.20, "10–20% (current)"),
    (0.10, 0.15, "10–15% NORMAL only"),
    (0.08, 0.18, "8–18%"),
    (0.12, 0.22, "12–22%"),
]


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
    records, i = [], 0
    while i + TF <= len(df1):
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].values.any():
            i += int(chunk["gap"].iloc[1:].values.argmax()) + 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[0],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += TF
    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Scanner ────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, trailing: int) -> pd.DataFrame:
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)

    VOL_MEAN_BARS = 100
    lookback = max(trailing, VOL_MEAN_BARS)

    records = []
    for i in range(lookback, n - MAX_BARS_HOLD):
        if gaps[i - trailing + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - trailing + 1: i + 1]
                          / closes[i - trailing:     i    ])
        sigma = float(np.std(trail_rets, ddof=1))
        if sigma == 0:
            continue

        mean_vol  = volumes[i - VOL_MEAN_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction  = 1 if scaled > 0 else -1
        entry      = closes[i]
        sigma_pts  = sigma * entry
        ann_vol    = sigma * math.sqrt(BARS_PER_YEAR / TF)

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if (direction == 1 and h >= tgt_prices[t]) or \
                       (direction == -1 and l <= tgt_prices[t]):
                        hit_tgt[t] = j - i
            for s in STOPS:
                if hit_stop[s] is None:
                    if (direction == 1 and l <= stop_prices[s]) or \
                       (direction == -1 and h >= stop_prices[s]):
                        hit_stop[s] = j - i

        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        bar_ts   = ts_pd[i]
        bar_hour = bar_ts.hour + bar_ts.minute / 60
        if bar_hour >= 20:
            session = SESSION_LABELS[2]
        elif bar_hour >= 13.5:
            session = SESSION_LABELS[1]
        else:
            session = SESSION_LABELS[0]

        records.append({
            "year":          bar_ts.year,
            "ann_vol":       ann_vol,
            "sigma_pts":     sigma_pts,
            "session":       session,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV helpers ─────────────────────────────────────────────────────────────────

def ev_stats(sub: pd.DataFrame, s: float, t: float) -> dict:
    if len(sub) < 5:
        return {"ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan"), "n": len(sub)}
    ht = sub[f"hit_tgt_{t}"].notna().values
    hs = sub[f"hit_stop_{s}"].notna().values
    ht_first = ht & ~(hs & (sub[f"hit_stop_{s}"].fillna(999)
                            <= sub[f"hit_tgt_{t}"].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    ev_nei   = sub["time_exit_ret"].values[neither].mean() if neither.any() else 0.0
    return {
        "ev":     ht_first.mean() * t - hs_first.mean() * s + neither.mean() * ev_nei,
        "p_tgt":  ht_first.mean(),
        "p_stop": hs_first.mean(),
        "n":      len(sub),
    }


def best_ev(sub: pd.DataFrame) -> tuple[float, float, float]:
    best, bs, bt = -999.0, 0.0, 0.0
    for s in STOPS:
        for t in TARGETS:
            st = ev_stats(sub, s, t)
            if not math.isnan(st["ev"]) and st["ev"] > best:
                best, bs, bt = st["ev"], s, t
    return best, bs, bt


def fmt(ev: float, n: int, min_n: int = 30, flag: bool = True) -> str:
    if n < min_n or math.isnan(ev):
        return f"{'[n=' + str(n) + ']':>16}"
    mk = "◄" if (flag and ev > 0) else " "
    return f"{ev:>+9.4f}σ n={n:<4} {mk}"


# ── Report ─────────────────────────────────────────────────────────────────────

def report(sym: str, res20: pd.DataFrame, res100: pd.DataFrame):
    hline = "  " + "─" * 86

    def side_by_side(title: str, groups20, groups100, labels):
        print(f"\n  {title}")
        print(hline)
        print(f"  {'Slice':<28}  {'trail=20 (100 min)':>22}   {'trail=100 (500 min)':>22}")
        print(hline)
        for lbl, sub20, sub100 in zip(labels, groups20, groups100):
            p20  = ev_stats(sub20,  PRAC_S, PRAC_T)
            p100 = ev_stats(sub100, PRAC_S, PRAC_T)
            print(f"  {lbl:<28}  {fmt(p20['ev'],  p20['n'])}   {fmt(p100['ev'], p100['n'])}")

    print(f"\n{'═'*90}")
    print(f"  {sym}  —  TRAIL=20 (100 min)  vs  TRAIL=100 (500 min)")
    print(f"  Practical combo: -{PRAC_S:.1f}σ / +{PRAC_T:.1f}σ")
    print(f"{'═'*90}")

    # Overall
    p20  = ev_stats(res20,  PRAC_S, PRAC_T)
    p100 = ev_stats(res100, PRAC_S, PRAC_T)
    b20,  _, _ = best_ev(res20)
    b100, _, _ = best_ev(res100)
    print(f"\n  {'Overall (no filter)':<28}  {fmt(p20['ev'],  p20['n'])}   {fmt(p100['ev'], p100['n'])}")
    print(f"  {'Best combo EV':<28}  {b20:>+9.4f}σ           {b100:>+9.4f}σ")

    # ── 1. Year by year ───────────────────────────────────────────────────────
    all_years = sorted(set(res20["year"]) | set(res100["year"]))
    side_by_side(
        "1. BY YEAR (no filter)",
        [res20[res20["year"] == y]  for y in all_years],
        [res100[res100["year"] == y] for y in all_years],
        [str(y) for y in all_years],
    )

    # ── 2. Vol regime ─────────────────────────────────────────────────────────
    for r in [res20, res100]:
        r["vol_regime"] = pd.cut(r["ann_vol"], bins=VOL_REGIME_BINS,
                                 labels=VOL_REGIME_LABELS)
    side_by_side(
        "2. BY VOL REGIME (no filter)",
        [res20[res20["vol_regime"] == lbl]  for lbl in VOL_REGIME_LABELS],
        [res100[res100["vol_regime"] == lbl] for lbl in VOL_REGIME_LABELS],
        VOL_REGIME_LABELS,
    )

    # ── 3. Session ────────────────────────────────────────────────────────────
    side_by_side(
        "3. BY SESSION (no filter)",
        [res20[res20["session"] == s]  for s in SESSION_LABELS],
        [res100[res100["session"] == s] for s in SESSION_LABELS],
        SESSION_LABELS,
    )

    # ── 4. Vol filter comparison ──────────────────────────────────────────────
    print(f"\n{hline}")
    print(f"  4. VOL FILTER COMPARISON (trail=20 only)")
    print(hline)
    print(f"  {'Filter':<22}  {'n':>6}  {'P(tgt)':>7}  {'EV prac':>10}  "
          f"{'Best EV':>9}  {'Best combo':>14}")
    print(hline)
    for lo, hi, label in VOL_FILTERS:
        if lo == 0 and hi == 99.0:
            sub = res20
        else:
            sub = res20[(res20["ann_vol"] >= lo) & (res20["ann_vol"] < hi)]
        p  = ev_stats(sub, PRAC_S, PRAC_T)
        bv, bs, bt = best_ev(sub)
        flag = "◄" if p["ev"] > 0 else " "
        print(f"  {label:<22}  {p['n']:>6,}  {p['p_tgt']:>7.3f}  "
              f"{p['ev']:>+10.4f}σ{flag}  {bv:>+9.4f}σ  -{bs:.1f}σ/+{bt:.1f}σ")

    # ── 5. 2D: Vol regime × Year (trail=20, no filter) ───────────────────────
    print(f"\n{hline}")
    print(f"  5. VOL REGIME × YEAR  (trail=20, practical EV, n≥30)")
    print(hline)
    years = sorted(res20["year"].unique())
    yr_hdr = "".join(f"  {y:>13}" for y in years)
    print(f"  {'Vol regime':<22}" + yr_hdr)
    print(hline)
    for vr in VOL_REGIME_LABELS:
        row = f"  {vr:<22}"
        for y in years:
            cell = res20[(res20["vol_regime"] == vr) & (res20["year"] == y)]
            p = ev_stats(cell, PRAC_S, PRAC_T)
            if cell.shape[0] < 30 or math.isnan(p["ev"]):
                row += f"  {'—':>13}"
            else:
                mk = "◄" if p["ev"] > 0 else " "
                row += f"  {p['ev']:>+7.4f} n={cell.shape[0]:<3}{mk}"
        print(row)

    # ── 6. 2D: Vol regime × Session (trail=20, no filter) ────────────────────
    print(f"\n{hline}")
    print(f"  6. VOL REGIME × SESSION  (trail=20, practical EV, n≥30)")
    print(hline)
    sess_hdr = "".join(f"  {s[:18]:>22}" for s in SESSION_LABELS)
    print(f"  {'Vol regime':<22}" + sess_hdr)
    print(hline)
    for vr in VOL_REGIME_LABELS:
        row = f"  {vr:<22}"
        for sess in SESSION_LABELS:
            cell = res20[(res20["vol_regime"] == vr) & (res20["session"] == sess)]
            p = ev_stats(cell, PRAC_S, PRAC_T)
            if cell.shape[0] < 30 or math.isnan(p["ev"]):
                row += f"  {'— (n=' + str(len(cell)) + ')':>22}"
            else:
                mk = "◄" if p["ev"] > 0 else " "
                row += f"  {p['ev']:>+8.4f} n={len(cell):<4}{mk}"
        print(row)

    # ── EV grid at trail=20, no filter ───────────────────────────────────────
    print(f"\n{hline}")
    print(f"  EV GRID  (trail=20, no filter)  rows=stop, cols=target:")
    col_hdr = "".join(f"  +{t:.1f}σ" for t in TARGETS)
    print(f"  {'Stop':<8}" + col_hdr)
    print("  " + "─" * 52)
    for s in STOPS:
        line = f"  -{s:.1f}σ  "
        for t in TARGETS:
            st = ev_stats(res20, s, t)
            mk = "◄" if st["ev"] > 0 else " "
            line += f"  {st['ev']:>+5.3f}{mk}"
        print(line)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES")
    args = parser.parse_args()

    sym   = args.sym.upper()
    cache = INSTRUMENTS.get(sym)
    if not cache:
        print(f"Unknown instrument: {sym}")
        sys.exit(1)

    print(f"Loading {cache} …")
    df1  = load_1min(cache)
    bars = make_5min_bars(df1)
    print(f"  {len(bars):,} 5-min bars")

    print("Scanning trail=20 …")
    res20  = scan(bars, 20)
    print(f"  {len(res20):,} triggers")

    print("Scanning trail=100 …")
    res100 = scan(bars, 100)
    print(f"  {len(res100):,} triggers")

    report(sym, res20, res100)
