"""
Bar alignment test for the VWASLR signal.

Tests whether VWASLR edge depends on bars being clock-aligned (:00/:05/:10)
or is robust to offset, using the same offset methodology as backtest_offset.py.

For each offset 0–4:
  offset 0: bars close at :00, :05, :10 ...  (production standard)
  offset 1: bars close at :01, :06, :11 ...
  ...

Production parameters are used (N=10, σ-window=100 bars, RTH only).
Thresholds swept per instrument (same as backtest_vwaslr.py).

Usage:
  python src/backtest_vwaslr_offset.py --sym MES
  python src/backtest_vwaslr_offset.py --sym MES MYM
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config ─────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TF                   = 5

RTH_START = (9, 30)
RTH_END   = (16, 0)

STOP_SIGMA   = 2.0
TARGET_SIGMA = 3.0
HOLD_BARS    = 5        # 25 min — matches MAX_HOLD_MIN in production

SIGMA_BARS   = 100      # 500-min slow σ window — matches production
N_WIN        = 10       # VWASLR window — matches production

THRESHOLDS = {          # production thresholds per instrument
    "MES": [0.5, 0.7, 1.0, 1.5],
    "MYM": [0.3, 0.5, 0.7, 1.0],
    "M2K": [0.5, 0.7, 1.0, 1.5],
}
DEFAULT_THRESHOLDS = [0.3, 0.5, 0.7, 1.0, 1.5]

PROD_THRESHOLD = {      # production trigger level
    "MES": 1.0,
    "MYM": 0.5,
    "M2K": 1.0,
}

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
}

MIN_N = 20   # skip combos with fewer trades


# ── Data helpers ────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_offset_bars(df1: pd.DataFrame, offset: int) -> pd.DataFrame:
    """
    Aggregate 1-min bars into 5-min bars whose close minute ≡ offset (mod 5).
    Session gaps are respected — chunks spanning a gap are discarded.
    Bar timestamp = close time of the 5-min bar.
    """
    records = []
    i = 0
    n = len(df1)

    while i < n:
        m = df1["ts"].iloc[i].minute
        if (m - offset) % TF == 0:
            break
        i += 1

    while i + TF <= n:
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            while i < n:
                m = df1["ts"].iloc[i].minute
                if (m - offset) % TF == 0:
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
        i += TF

    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Signal scan ─────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Scan for VWASLR signals using production parameters (N_WIN, SIGMA_BARS, HOLD_BARS).
    Conservative OHLC ordering: stop checked before target within each bar.
    """
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    nb      = len(bars)

    warmup     = max(SIGMA_BARS, N_WIN) + 1
    hold_until = -1
    records    = []

    for i in range(warmup, nb - HOLD_BARS):
        if gaps[i - SIGMA_BARS + 1: i + HOLD_BARS + 1].any():
            continue

        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        if bar_hm < RTH_START or bar_hm >= RTH_END:
            continue

        # σ from slow 500-min window
        trail_rets = np.log(closes[i - SIGMA_BARS + 1: i + 1]
                          / closes[i - SIGMA_BARS:     i    ])
        sigma = float(np.std(trail_rets, ddof=1))
        if sigma == 0:
            continue

        # VWASLR over last N_WIN bars
        ret_win = np.log(closes[i - N_WIN + 1: i + 1]
                       / closes[i - N_WIN:     i    ])
        vol_win = volumes[i - N_WIN: i]
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

        # Conservative: check stop before target within each bar
        hit_tgt = hit_stop = None
        for j in range(i + 1, i + HOLD_BARS + 1):
            h, l = highs[j], lows[j]
            if hit_stop is None:
                if direction == 1  and l <= stop_price: hit_stop = j - i
                elif direction == -1 and h >= stop_price: hit_stop = j - i
            if hit_tgt is None:
                if direction == 1  and h >= tgt_price:  hit_tgt  = j - i
                elif direction == -1 and l <= tgt_price:  hit_tgt  = j - i

        time_exit_ret = math.log(closes[i + HOLD_BARS] / entry) * direction / sigma
        hold_until    = i + HOLD_BARS

        records.append({
            "year":          ts_pd[i].year,
            "vwaslr":        vwaslr,
            "direction":     direction,
            "hit_tgt":       hit_tgt,
            "hit_stop":      hit_stop,
            "time_exit_ret": time_exit_ret,
        })

    return pd.DataFrame(records)


# ── EV helpers ──────────────────────────────────────────────────────────────────

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


# ── Main ────────────────────────────────────────────────────────────────────────

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

        print(f"\n{'═'*72}")
        print(f"  {sym}  —  VWASLR Bar Alignment Test (offsets 0–4)")
        print(f"  N={N_WIN} bars ({N_WIN*TF}min)  σ-window={SIGMA_BARS} bars ({SIGMA_BARS*TF}min)"
              f"  hold={HOLD_BARS} bars ({HOLD_BARS*TF}min)")
        print(f"  stop={STOP_SIGMA:.0f}σ  target={TARGET_SIGMA:.0f}σ  RTH only")
        print(f"{'═'*72}")

        print(f"\nLoading {INSTRUMENTS[sym]} …")
        df1 = load_1min(INSTRUMENTS[sym])
        print(f"  {len(df1):,} 1-min bars  ({df1['ts'].min().date()} → {df1['ts'].max().date()})")

        # Build offset bars and scan all thresholds
        all_results: dict[int, dict[float, pd.DataFrame]] = {}
        for offset in range(TF):
            bars = make_offset_bars(df1, offset)
            all_results[offset] = {}
            for thr in thresholds:
                res = scan(bars, thr)
                all_results[offset][thr] = res
            n_prod = len(all_results[offset][prod_thr])
            print(f"  offset {offset}: {len(bars):,} bars  "
                  f"prod threshold (±{prod_thr:.1f}σ): {n_prod} signals")

        # ── Summary table: one row per offset, columns = threshold ────────────
        for label, thr_filter in [("ALL THRESHOLDS", thresholds),
                                   (f"PRODUCTION (±{prod_thr:.1f}σ)", [prod_thr])]:
            print(f"\n{'─'*72}")
            print(f"  {sym}  —  {label}")
            print(f"{'─'*72}")
            thr_header = "  ".join(f"±{t:.1f}σ ({ev_stats(all_results[0][t])['n']:>4}n)"
                                    for t in thr_filter)
            print(f"  {'Offset':>7}  {'Closes at':>12}    {thr_header}")
            print(f"  {'─'*68}")
            for offset in range(TF):
                closes_at = f":{offset:02d}/:{offset+5:02d}/..."
                parts = []
                for thr in thr_filter:
                    st = ev_stats(all_results[offset][thr])
                    if math.isnan(st["ev"]):
                        parts.append(f"{'—':>14}")
                    else:
                        flag = "◄" if st["ev"] > 0 else " "
                        parts.append(f"{st['ev']:>+10.4f}σ {flag}")
                print(f"  {offset:>7}  {closes_at:>12}    " + "  ".join(parts))

        # ── Year-by-year for production threshold ─────────────────────────────
        print(f"\n{'─'*72}")
        print(f"  {sym}  —  BY YEAR  production threshold ±{prod_thr:.1f}σ  (EV per offset)")
        print(f"{'─'*72}")
        years = sorted({y for res in all_results[0][prod_thr]["year"] for y in [res]}
                       if False else
                       set(all_results[0][prod_thr]["year"].unique()))
        header = f"  {'Year':<6}" + "".join(f"  {'off='+str(o):>10}" for o in range(TF))
        print(header)
        print(f"  {'─'*60}")
        for yr in sorted(years):
            row = f"  {yr:<6}"
            for offset in range(TF):
                sub = all_results[offset][prod_thr]
                sub_yr = sub[sub["year"] == yr]
                st = ev_stats(sub_yr)
                if math.isnan(st["ev"]):
                    row += f"  {'—':>10}"
                else:
                    row += f"  {st['ev']:>+10.4f}"
            print(row)

        # ── Direction split for production threshold at offset 0 ──────────────
        prod_res = all_results[0][prod_thr]
        if len(prod_res) >= MIN_N:
            print(f"\n{'─'*72}")
            print(f"  {sym}  —  DIRECTION SPLIT  offset=0  ±{prod_thr:.1f}σ")
            print(f"{'─'*72}")
            for lbl, sub in [("LONG ", prod_res[prod_res["direction"] ==  1]),
                              ("SHORT", prod_res[prod_res["direction"] == -1])]:
                st = ev_stats(sub)
                ev_s = f"{st['ev']:+.4f}σ" if not math.isnan(st["ev"]) else "—"
                print(f"  {lbl}  n={st['n']:>4}  P(tgt)={st['p_tgt']:.3f}  "
                      f"P(stop)={st['p_stop']:.3f}  EV={ev_s}")
