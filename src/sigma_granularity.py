"""
Does bar granularity matter for the trailing σ computation?

Holds the trailing window fixed at 100 minutes but varies the bar size:
  100 × 1-min bars   (most data points, captures intrabar moves)
   50 × 2-min bars
   20 × 5-min bars   (current baseline)
   10 × 10-min bars  (fewest data points)

The signal trigger (5-min bar with |ret/σ| ≥ 3 and vol ≥ 1.5× mean) and the
hold/target/stop logic are identical across all variants.  Only σ changes.

All σ estimates are scaled to per-5-min units via σ_5min = σ_bar × √(5/bar_min)
so that stop/target sizes are directly comparable.

Usage:
  python src/sigma_granularity.py            # MES
  python src/sigma_granularity.py --sym MYM
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
WINDOW_MINUTES       = 100    # fixed trailing window for all variants
TF_SIGNAL            = 5      # signal bar size (minutes) — always 5-min
MAX_BARS_HOLD        = 3      # × TF_SIGNAL = 15-min max hold
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5
VOL_MEAN_BARS        = 20     # 20 × 5-min = 100 min volume baseline
BARS_PER_YEAR        = 252 * 23 * 60

PRAC_S, PRAC_T = 2.0, 3.0
STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Bar sizes to test (minutes)
BAR_SIZES = [1, 2, 5, 10]

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data ───────────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def agg_bars(df1: pd.DataFrame, bar_min: int) -> pd.DataFrame:
    """Aggregate 1-min bars to bar_min-minute bars, discarding gap-spanning chunks."""
    records, i = [], 0
    n = bar_min
    while i + n <= len(df1):
        chunk = df1.iloc[i: i + n]
        if chunk["gap"].iloc[1:].values.any():
            i += int(chunk["gap"].iloc[1:].values.argmax()) + 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[0],
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += n
    out = pd.DataFrame(records)
    if len(out):
        out["gap"] = out["ts"].diff() != pd.Timedelta(minutes=bar_min)
        out.iloc[0, out.columns.get_loc("gap")] = True
    return out


# ── Scanner ────────────────────────────────────────────────────────────────────

def scan(bars5: pd.DataFrame, sigma_bars: pd.DataFrame,
         bar_min: int) -> pd.DataFrame:
    """
    bars5       — 5-min OHLCV bars (trigger bars)
    sigma_bars  — bar_min-minute close+gap bars (used only for σ computation)
    bar_min     — minutes per sigma bar
    """
    trailing_n = WINDOW_MINUTES // bar_min   # number of sigma bars in 100 min
    scale      = math.sqrt(TF_SIGNAL / bar_min)   # scale σ to per-5-min units

    # Build timestamp → position index for sigma_bars
    sb_ts    = sigma_bars["ts"].values
    sb_close = sigma_bars["close"].values
    sb_gaps  = sigma_bars["gap"].values
    sb_idx   = {ts: i for i, ts in enumerate(sb_ts)}

    closes5  = bars5["close"].values
    highs5   = bars5["high"].values
    lows5    = bars5["low"].values
    volumes5 = bars5["volume"].values
    gaps5    = bars5["gap"].values
    ts5      = bars5["ts"].values
    n5       = len(bars5)

    records = []
    for i in range(VOL_MEAN_BARS, n5 - MAX_BARS_HOLD):
        # Gap check on 5-min bars: forward hold window
        if gaps5[i - VOL_MEAN_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        # Find this bar's timestamp in the sigma_bars index
        pos = sb_idx.get(ts5[i])
        if pos is None or pos < trailing_n:
            continue

        # σ computation: trailing_n bars in sigma_bars ending at this timestamp
        sb_slice = slice(pos - trailing_n, pos + 1)
        if sb_gaps[sb_slice].any():
            continue

        trail_rets = np.log(sb_close[pos - trailing_n + 1: pos + 1]
                          / sb_close[pos - trailing_n:     pos    ])
        sigma_bar  = float(np.std(trail_rets, ddof=1))
        if sigma_bar == 0:
            continue
        sigma = sigma_bar * scale   # per-5-min σ

        mean_vol  = volumes5[i - VOL_MEAN_BARS: i].mean()
        vol_ratio = volumes5[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes5[i] / closes5[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction  = 1 if scaled > 0 else -1
        entry      = closes5[i]
        sigma_pts  = sigma * entry
        ann_vol    = sigma * math.sqrt(BARS_PER_YEAR / TF_SIGNAL)

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs5[j], lows5[j]
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

        time_exit_ret = math.log(closes5[i + MAX_BARS_HOLD] / entry) * direction / sigma

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


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="MES")
    args   = parser.parse_args()

    sym   = args.sym.upper()
    cache = INSTRUMENTS.get(sym)
    if not cache:
        print(f"Unknown instrument: {sym}")
        sys.exit(1)

    print(f"Loading {cache} …")
    df1 = load_1min(cache)
    print(f"  {len(df1):,} 1-min bars")

    # Build 5-min signal bars once
    bars5 = agg_bars(df1, 5)
    # Add high/low (need them for target/stop checks — agg from 1-min)
    hl_records, i = [], 0
    while i + 5 <= len(df1):
        chunk = df1.iloc[i: i + 5]
        if chunk["gap"].iloc[1:].values.any():
            i += int(chunk["gap"].iloc[1:].values.argmax()) + 1
            continue
        hl_records.append({"ts": chunk["ts"].iloc[0],
                            "high": chunk["high"].max(),
                            "low":  chunk["low"].min()})
        i += 5
    hl = pd.DataFrame(hl_records)
    bars5 = bars5.merge(hl, on="ts")
    bars5["gap"] = bars5["ts"].diff() != pd.Timedelta(minutes=5)
    bars5.iloc[0, bars5.columns.get_loc("gap")] = True
    print(f"  {len(bars5):,} 5-min signal bars")

    results = []
    for bm in BAR_SIZES:
        trailing_n = WINDOW_MINUTES // bm
        print(f"\nBar size {bm}-min  ({trailing_n} bars × {bm} min = {WINDOW_MINUTES} min) …",
              end=" ", flush=True)
        sb = agg_bars(df1, bm)
        res = scan(bars5, sb, bm)
        p   = ev_stats(res, PRAC_S, PRAC_T)
        bev, bs, bt = best_ev(res)
        results.append({
            "bar_min": bm, "n_sigma_bars": trailing_n,
            "n": p["n"], "p_tgt": p["p_tgt"], "p_stop": p["p_stop"],
            "ev_prac": p["ev"], "best_ev": bev,
            "best_s": bs, "best_t": bt,
            "sigma_pts_med": res["sigma_pts"].median() if len(res) else float("nan"),
        })
        print(f"n={p['n']:,}  EV={p['ev']:>+.4f}σ  best={bev:>+.4f}σ  "
              f"median_1σ={res['sigma_pts'].median():.2f}pts")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  {sym}  —  SIGMA GRANULARITY  (100-min window, signal on 5-min bars)")
    print(f"  Practical combo: -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ")
    print(f"{'═'*80}")
    print(f"\n  {'Bar':>5}  {'n σ-bars':>9}  {'n trig':>7}  "
          f"{'P(tgt)':>7}  {'P(stop)':>7}  {'EV prac':>9}  "
          f"{'Best EV':>8}  {'Best combo':>12}  {'1σ (pts)':>9}")
    print("  " + "─" * 76)
    for r in results:
        flag = "◄" if r["ev_prac"] > 0 else " "
        curr = "  ← current" if r["bar_min"] == 5 else ""
        print(f"  {r['bar_min']:>3}-min  {r['n_sigma_bars']:>9}  {r['n']:>7,}  "
              f"{r['p_tgt']:>7.3f}  {r['p_stop']:>7.3f}  "
              f"{r['ev_prac']:>+9.4f}σ{flag}  {r['best_ev']:>+8.4f}σ  "
              f"-{r['best_s']:.1f}σ/+{r['best_t']:.1f}σ  "
              f"{r['sigma_pts_med']:>9.2f}{curr}")

    # EV grid for each bar size
    print(f"\n  EV GRIDS  (rows=stop, cols=target):")
    for bm in BAR_SIZES:
        trailing_n = WINDOW_MINUTES // bm
        sb  = agg_bars(df1, bm)
        res = scan(bars5, sb, bm)
        if len(res) < 10:
            continue
        col_hdr = "".join(f"  +{t:.1f}σ" for t in TARGETS)
        print(f"\n  {bm}-min bars ({trailing_n} × {bm} = {WINDOW_MINUTES} min):")
        print(f"  {'Stop':<8}" + col_hdr)
        print("  " + "─" * 52)
        for s in STOPS:
            line = f"  -{s:.1f}σ  "
            for t in TARGETS:
                st = ev_stats(res, s, t)
                mk = "◄" if st["ev"] > 0 else " "
                line += f"  {st['ev']:>+5.3f}{mk}"
            print(line)
