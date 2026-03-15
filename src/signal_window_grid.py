"""
Grid search: optimal trailing window for signal detection σ.

Tests how strategy EV varies with the trailing close-return std window used to
normalise the trigger bar's return (|ret / σ| ≥ 3) and size stops/targets.

Windows tested (5-min bars → minutes):
  10→50   20→100   30→150   40→200   50→250   60→300
  78→390  100→500  130→650  156→780  200→1000

Reports EV at the practical combo (-1.5σ/+2.5σ) unfiltered and with the
10–20% vol filter, plus the best (stop, target) at each window.

Usage:
  python src/signal_window_grid.py            # MES
  python src/signal_window_grid.py --sym MYM
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
VOL_FILTER_LO  = 0.10
VOL_FILTER_HI  = 0.20

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Trailing windows to test (in 5-min bars)
WINDOWS = [10, 20, 30, 40, 50, 60, 78, 100, 130, 156, 200]


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
    n       = len(bars)

    # Volume mean always uses the same 100-bar window so comparisons are fair
    VOL_MEAN_BARS = 100

    records = []
    for i in range(max(trailing, VOL_MEAN_BARS), n - MAX_BARS_HOLD):
        if gaps[i - trailing + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - trailing + 1: i + 1]
                          / closes[i - trailing:     i    ])
        sigma = float(np.std(trail_rets, ddof=1))
        if sigma == 0:
            continue

        mean_vol  = volumes[i - VOL_MEAN_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")

        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma

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

        records.append({
            "ann_vol":       ann_vol,
            "sigma_pts":     sigma_pts,
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
    time_ret = sub["time_exit_ret"].values
    ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
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
    print(f"  {len(bars):,} 5-min bars\n")

    rows = []
    for tw in WINDOWS:
        mins = tw * TF
        print(f"  Scanning trail={tw} bars ({mins} min) …", end=" ", flush=True)
        res = scan(bars, tw)

        filt = res[(res["ann_vol"] >= VOL_FILTER_LO) & (res["ann_vol"] < VOL_FILTER_HI)]

        prac_all  = ev_stats(res,  PRAC_S, PRAC_T)
        prac_filt = ev_stats(filt, PRAC_S, PRAC_T)
        bev_all,  bs_all,  bt_all  = best_ev(res)
        bev_filt, bs_filt, bt_filt = best_ev(filt)

        rows.append({
            "trail_bars": tw, "trail_min": mins,
            "n_all":        prac_all["n"],
            "ev_all":       prac_all["ev"],
            "ptgt_all":     prac_all["p_tgt"],
            "bev_all":      bev_all,
            "bcombo_all":   f"-{bs_all:.1f}/+{bt_all:.1f}σ",
            "n_filt":       prac_filt["n"],
            "ev_filt":      prac_filt["ev"],
            "ptgt_filt":    prac_filt["p_tgt"],
            "bev_filt":     bev_filt,
            "bcombo_filt":  f"-{bs_filt:.1f}/+{bt_filt:.1f}σ",
        })
        print(f"n={prac_all['n']:,}  ev_all={prac_all['ev']:>+.4f}σ  "
              f"ev_filt={prac_filt['ev']:>+.4f}σ")

    # ── Summary table ──────────────────────────────────────────────────────────

    print(f"\n{'═'*110}")
    print(f"  {sym}  —  SIGNAL WINDOW GRID  (practical combo: -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)")
    print(f"{'═'*110}")
    print(f"\n  {'Trail':>6}  {'Min':>5}  "
          f"{'── UNFILTERED ──────────────────────────':^42}   "
          f"{'── VOL FILTER 10–20% ───────────────────':^42}")
    print(f"  {'bars':>6}  {'':>5}  "
          f"{'n':>7}  {'P(tgt)':>7}  {'EV prac':>9}  {'Best EV':>8}  {'Best combo':>12}   "
          f"{'n':>7}  {'P(tgt)':>7}  {'EV prac':>9}  {'Best EV':>8}  {'Best combo':>12}")
    print("  " + "─" * 106)

    for r in rows:
        fa = "◄" if r["ev_all"]  > 0 else " "
        ff = "◄" if r["ev_filt"] > 0 else " "
        marker = ""
        if r["trail_bars"] == 78:
            marker = " ← NYSE session"
        elif r["trail_bars"] == 100:
            marker = " ← current baseline"
        print(f"  {r['trail_bars']:>6}  {r['trail_min']:>5}  "
              f"{r['n_all']:>7,}  {r['ptgt_all']:>7.3f}  "
              f"{r['ev_all']:>+9.4f}σ{fa}  {r['bev_all']:>+8.4f}σ  {r['bcombo_all']:>12}   "
              f"{r['n_filt']:>7,}  {r['ptgt_filt']:>7.3f}  "
              f"{r['ev_filt']:>+9.4f}σ{ff}  {r['bev_filt']:>+8.4f}σ  "
              f"{r['bcombo_filt']:>12}{marker}")

    # Best window
    best_row_all  = max(rows, key=lambda r: r["ev_all"]  if not math.isnan(r["ev_all"])  else -999)
    best_row_filt = max(rows, key=lambda r: r["ev_filt"] if not math.isnan(r["ev_filt"]) else -999)

    print(f"\n  Best (unfiltered): trail={best_row_all['trail_bars']} bars "
          f"({best_row_all['trail_min']} min)  EV={best_row_all['ev_all']:>+.4f}σ")
    print(f"  Best (vol-filt):   trail={best_row_filt['trail_bars']} bars "
          f"({best_row_filt['trail_min']} min)  EV={best_row_filt['ev_filt']:>+.4f}σ")
