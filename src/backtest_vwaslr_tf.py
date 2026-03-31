"""
VWASLR multi-timeframe backtest.

Compares VWASLR signal quality at 1-min, 2-min, 3-min vs the production 5-min
baseline. All timeframes use the same underlying time windows:

  Signal window : ≈ 50 min  (N × TF)
  σ window      : ≈ 500 min (σ_bars × TF)
  Hold window   : ≈ 25 min  (hold × TF)

Timeframe parameters:
  1-min : N=50  σ=500  hold=25
  2-min : N=25  σ=250  hold=12  (24 min hold)
  3-min : N=17  σ=167  hold=8   (24 min hold)
  5-min : N=10  σ=100  hold=5   (production baseline)

RTH only, offset=0 (clock-aligned bars).
Conservative OHLC ordering: stop checked before target within each bar.

Usage:
  python src/backtest_vwaslr_tf.py --sym MES
  python src/backtest_vwaslr_tf.py --sym MES MYM M2K
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config ──────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

RTH_START = (9, 30)
RTH_END   = (16, 0)

STOP_SIGMA   = 2.0
TARGET_SIGMA = 3.0

# Timeframe definitions — each entry preserves ≈50-min signal, ≈500-min σ, ≈25-min hold
TIMEFRAMES = {
    1: {"tf": 1,  "n": 50, "sigma_bars": 500, "hold": 25, "label": "1-min"},
    2: {"tf": 2,  "n": 25, "sigma_bars": 250, "hold": 12, "label": "2-min"},
    3: {"tf": 3,  "n": 17, "sigma_bars": 167, "hold":  8, "label": "3-min"},
    5: {"tf": 5,  "n": 10, "sigma_bars": 100, "hold":  5, "label": "5-min (production)"},
}

THRESHOLDS = {
    "MES": [0.5, 0.7, 1.0, 1.5],
    "MYM": [0.3, 0.5, 0.7, 1.0],
    "M2K": [0.5, 0.7, 1.0, 1.5],
}
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 1.0, 1.5]

PROD_THRESHOLD = {
    "MES": 1.0,
    "MYM": 0.5,
    "M2K": 1.0,
}

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
}

MIN_N = 20


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_tf_bars(df1: pd.DataFrame, tf: int) -> pd.DataFrame:
    """
    Aggregate 1-min bars into tf-min bars (clock-aligned, offset=0).
    Bar timestamp = close time of the tf-min bar.
    Session gaps are respected — chunks spanning a gap are discarded.
    """
    if tf == 1:
        bars = df1[["ts", "open", "high", "low", "close", "volume", "gap"]].copy()
        bars["gap"] = bars["gap"].copy()
        bars.iloc[0, bars.columns.get_loc("gap")] = True
        return bars.reset_index(drop=True)

    records = []
    i = 0
    n = len(df1)

    # Skip to first clock-aligned bar
    while i < n:
        m = df1["ts"].iloc[i].minute
        if m % tf == 0:
            break
        i += 1

    while i + tf <= n:
        chunk = df1.iloc[i: i + tf]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            while i < n:
                m = df1["ts"].iloc[i].minute
                if m % tf == 0:
                    break
                i += 1
            continue

        records.append({
            "ts":     chunk["ts"].iloc[-1],   # close time
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += tf

    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=tf)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Signal scan ───────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, tf: int, n_win: int, sigma_bars: int,
         hold: int, threshold: float) -> pd.DataFrame:
    """
    Scan bars for VWASLR signals.
    Conservative OHLC ordering: stop checked before target within each bar.
    """
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    nb      = len(bars)

    warmup     = max(sigma_bars, n_win) + 1
    hold_until = -1
    records    = []

    for i in range(warmup, nb - hold):
        if gaps[i - sigma_bars + 1: i + hold + 1].any():
            continue

        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        if bar_hm < RTH_START or bar_hm >= RTH_END:
            continue

        # σ from slow window
        trail_rets = np.log(closes[i - sigma_bars + 1: i + 1]
                          / closes[i - sigma_bars:     i    ])
        sigma = float(np.std(trail_rets, ddof=1))
        if sigma == 0:
            continue

        # VWASLR over last n_win bars
        ret_win = np.log(closes[i - n_win + 1: i + 1]
                       / closes[i - n_win:     i    ])
        vol_win = volumes[i - n_win: i]
        sum_vol = vol_win.sum()
        if sum_vol == 0:
            continue

        vwaslr = float((ret_win / sigma * vol_win).sum() / sum_vol)

        if abs(vwaslr) < threshold:
            continue
        if i <= hold_until:
            continue

        direction  = 1 if vwaslr > 0 else -1
        entry      = closes[i]
        tgt_price  = entry * math.exp( direction * TARGET_SIGMA * sigma)
        stop_price = entry * math.exp(-direction * STOP_SIGMA   * sigma)

        # Conservative: stop before target
        hit_tgt = hit_stop = None
        for j in range(i + 1, i + hold + 1):
            h, l = highs[j], lows[j]
            if direction == 1:
                if hit_stop is None and l <= stop_price: hit_stop = j - i
                if hit_tgt  is None and h >= tgt_price:  hit_tgt  = j - i
            else:
                if hit_stop is None and h >= stop_price: hit_stop = j - i
                if hit_tgt  is None and l <= tgt_price:  hit_tgt  = j - i

        time_exit_ret = math.log(closes[i + hold] / entry) * direction / sigma
        hold_until    = i + hold

        records.append({
            "year":          ts_pd[i].year,
            "vwaslr":        vwaslr,
            "direction":     direction,
            "hit_tgt":       hit_tgt,
            "hit_stop":      hit_stop,
            "time_exit_ret": time_exit_ret,
        })

    return pd.DataFrame(records)


# ── EV helpers ────────────────────────────────────────────────────────────────

def ev_stats(df: pd.DataFrame) -> dict:
    if len(df) < MIN_N:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(df)}
    ht = df["hit_tgt"].notna().values
    hs = df["hit_stop"].notna().values
    tgt_bar  = df["hit_tgt"].fillna(999).values
    stop_bar = df["hit_stop"].fillna(999).values
    ht_first = ht & (tgt_bar <= stop_bar)
    hs_first = hs & (stop_bar < tgt_bar)
    neither  = ~ht_first & ~hs_first
    te       = df["time_exit_ret"].values
    ev_nei   = te[neither].mean() if neither.any() else 0.0
    ev       = ht_first.mean() * TARGET_SIGMA - hs_first.mean() * STOP_SIGMA + neither.mean() * ev_nei
    return {"ev": ev, "p_tgt": ht_first.mean(), "p_stop": hs_first.mean(), "n": len(df)}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", nargs="+", default=["MES"],
                        help="Symbols to test (default: MES)")
    args = parser.parse_args()

    for sym in args.sym:
        if sym not in INSTRUMENTS:
            print(f"Unknown symbol {sym}, skipping")
            continue

        thresholds = THRESHOLDS.get(sym, DEFAULT_THRESHOLDS)
        prod_thr   = PROD_THRESHOLD.get(sym, 1.0)

        print(f"\n{'═'*76}")
        print(f"  {sym}  —  VWASLR Multi-Timeframe Backtest")
        print(f"  Signal window ≈50min  σ-window ≈500min  Hold ≈25min")
        print(f"  stop={STOP_SIGMA:.0f}σ  target={TARGET_SIGMA:.0f}σ  RTH only  conservative OHLC")
        print(f"{'═'*76}")

        print(f"\nLoading {INSTRUMENTS[sym]} …")
        df1 = load_1min(INSTRUMENTS[sym])
        print(f"  {len(df1):,} 1-min bars  ({df1['ts'].min().date()} → {df1['ts'].max().date()})")

        # Build bars and scan for each timeframe × threshold
        all_results: dict[int, dict[float, pd.DataFrame]] = {}
        for tf_key, cfg in TIMEFRAMES.items():
            tf    = cfg["tf"]
            n     = cfg["n"]
            sb    = cfg["sigma_bars"]
            hold  = cfg["hold"]
            label = cfg["label"]

            bars = make_tf_bars(df1, tf)
            if bars.empty:
                print(f"  {label}: no bars")
                continue

            all_results[tf_key] = {}
            for thr in thresholds:
                res = scan(bars, tf, n, sb, hold, thr)
                all_results[tf_key][thr] = res

            n_prod = len(all_results[tf_key].get(prod_thr, pd.DataFrame()))
            print(f"  {label:22s}: {len(bars):,} bars  "
                  f"N={n} σ={sb} hold={hold}  "
                  f"prod thr ±{prod_thr:.1f}σ → {n_prod} signals")

        # ── Summary table: rows=TF, cols=threshold ────────────────────────────
        print(f"\n{'─'*76}")
        print(f"  {sym}  —  EV BY TIMEFRAME × THRESHOLD")
        print(f"{'─'*76}")
        thr_header = "  ".join(f"±{t:.1f}σ" for t in thresholds)
        print(f"  {'Timeframe':<22}  {'N':>4}  {'hold':>5}    {thr_header}  {'n (prod)':>10}")
        print(f"  {'─'*70}")

        for tf_key, cfg in TIMEFRAMES.items():
            if tf_key not in all_results:
                continue
            label = cfg["label"]
            n     = cfg["n"]
            hold  = cfg["hold"]
            parts = []
            for thr in thresholds:
                st = ev_stats(all_results[tf_key].get(thr, pd.DataFrame()))
                if math.isnan(st["ev"]):
                    parts.append(f"{'—':>8}")
                else:
                    flag = "◄" if st["ev"] > 0 else " "
                    parts.append(f"{st['ev']:>+7.4f}{flag}")
            n_prod = ev_stats(all_results[tf_key].get(prod_thr, pd.DataFrame()))["n"]
            prod_str = f"{n_prod:>5}" if not math.isnan(float(n_prod)) else "—"
            is_prod  = "  ← production" if tf_key == 5 else ""
            print(f"  {label:<22}  {n:>4}  {hold:>5}    " +
                  "  ".join(parts) + f"  {prod_str:>10}{is_prod}")

        # ── Year-by-year for production threshold ─────────────────────────────
        print(f"\n{'─'*76}")
        print(f"  {sym}  —  YEAR-BY-YEAR  production threshold ±{prod_thr:.1f}σ")
        print(f"{'─'*76}")

        ref_df = all_results.get(5, {}).get(prod_thr, pd.DataFrame())
        years  = sorted(ref_df["year"].unique()) if not ref_df.empty else []

        tf_labels = [cfg["label"] for cfg in TIMEFRAMES.values() if cfg["tf"] in all_results]
        header = f"  {'Year':<6}" + "".join(f"  {lb:>18}" for lb in tf_labels)
        print(header)
        print(f"  {'─'*72}")

        for yr in years:
            row = f"  {yr:<6}"
            for tf_key, cfg in TIMEFRAMES.items():
                if tf_key not in all_results:
                    continue
                sub = all_results[tf_key].get(prod_thr, pd.DataFrame())
                if sub.empty:
                    row += f"  {'—':>18}"
                    continue
                sub_yr = sub[sub["year"] == yr]
                st = ev_stats(sub_yr)
                if math.isnan(st["ev"]):
                    row += f"  {'—':>18}"
                else:
                    row += f"  {st['ev']:>+10.4f} ({st['n']:>3}n)"
            print(row)

        # ── Direction split at production threshold ────────────────────────────
        print(f"\n{'─'*76}")
        print(f"  {sym}  —  DIRECTION SPLIT  production threshold ±{prod_thr:.1f}σ")
        print(f"{'─'*76}")
        dir_header = f"  {'Timeframe':<22}  {'Dir':<6}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV':>10}"
        print(dir_header)
        print(f"  {'─'*68}")
        for tf_key, cfg in TIMEFRAMES.items():
            if tf_key not in all_results:
                continue
            label = cfg["label"]
            res   = all_results[tf_key].get(prod_thr, pd.DataFrame())
            if res.empty:
                continue
            for dir_lbl, d in [("LONG ", 1), ("SHORT", -1)]:
                sub = res[res["direction"] == d]
                st  = ev_stats(sub)
                ev_s = f"{st['ev']:>+10.4f}σ" if not math.isnan(st["ev"]) else f"{'—':>11}"
                flag = " ◄" if not math.isnan(st.get("ev", float("nan"))) and st["ev"] > 0 else ""
                print(f"  {label:<22}  {dir_lbl}  {st['n']:>5}  "
                      f"{st['p_tgt'] if not math.isnan(st['p_tgt']) else 0:.3f}    "
                      f"{st['p_stop'] if not math.isnan(st['p_stop']) else 0:.3f}  {ev_s}{flag}")
            print()
