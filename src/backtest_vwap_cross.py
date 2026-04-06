"""
VWAP-cross signal backtest — direct comparison with VWASLR.

Signal: price (close) crosses the session VWAP from below (long) or above (short).
VWAP is anchored at 09:30 ET each RTH session and resets daily.

Exit logic is identical to backtest_vwaslr.py:
  - 2σ stop  / 3σ target  (σ from 20-bar trailing window)
  - Fixed-hold time exit if neither stop nor target is hit
  - No re-entry while a hold is active

Outputs a comparison table vs the best VWASLR combo reported in
backtest_vwaslr.py (N=10, thr=0.5σ, hold=4).

Usage:
  python src/backtest_vwap_cross.py
  python src/backtest_vwap_cross.py --sym MES MYM M2K
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config (match backtest_vwaslr.py exactly) ────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TF_MINUTES           = 5
SIGMA_BARS           = 20     # trailing bars for σ estimate
STOP_SIGMA           = 2.0
TARGET_SIGMA         = 3.0

RTH_START = (9, 30)
RTH_END   = (16, 0)

HOLD_VALUES = [2, 3, 4, 5, 6]   # bars (each bar = TF_MINUTES min)
MIN_N       = 20

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
}

POINT_VALUE = {"MES": 5.0, "MYM": 0.5, "M2K": 5.0}


# ── Data helpers (identical to backtest_vwaslr.py) ───────────────────────────

def load_and_resample(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    records, i = [], 0
    n = len(df)
    while i + TF_MINUTES <= n:
        chunk = df.iloc[i: i + TF_MINUTES]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            continue
        records.append({
            "ts":     chunk["ts"].iloc[0],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
            "typ":    (chunk["high"].max() + chunk["low"].min() + chunk["close"].iloc[-1]) / 3,
        })
        i += TF_MINUTES

    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF_MINUTES)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── VWAP computation ──────────────────────────────────────────────────────────

def add_session_vwap(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `vwap` column: cumulative VWAP anchored at 09:30 ET each RTH session.
    Bars outside RTH get NaN.
    """
    bars = bars.copy()
    bars["ts_et"] = bars["ts"].dt.tz_convert(ET)
    bars["date_et"] = bars["ts_et"].dt.date
    bars["hm"] = bars["ts_et"].apply(lambda t: (t.hour, t.minute))
    bars["in_rth"] = (bars["hm"] >= RTH_START) & (bars["hm"] < RTH_END)

    vwap_vals = np.full(len(bars), np.nan)
    cum_tv = 0.0
    cum_v  = 0.0
    prev_date = None

    for idx in range(len(bars)):
        if not bars["in_rth"].iloc[idx]:
            prev_date = None
            cum_tv = 0.0
            cum_v  = 0.0
            continue
        date = bars["date_et"].iloc[idx]
        if date != prev_date:
            cum_tv = 0.0
            cum_v  = 0.0
            prev_date = date
        cum_tv += bars["typ"].iloc[idx] * bars["volume"].iloc[idx]
        cum_v  += bars["volume"].iloc[idx]
        if cum_v > 0:
            vwap_vals[idx] = cum_tv / cum_v

    bars["vwap"] = vwap_vals
    return bars


# ── Signal scan ───────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, hold: int) -> pd.DataFrame:
    """
    Fire a trade whenever the close crosses VWAP (from below → long;
    from above → short).  Evaluate with stop/target over the next `hold` bars.
    """
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    vwap    = bars["vwap"].values
    in_rth  = bars["in_rth"].values
    ts_arr  = bars["ts"].values
    nb      = len(bars)

    warmup     = SIGMA_BARS + 1
    hold_until = -1
    records    = []

    for i in range(warmup, nb - hold):
        if not in_rth[i] or not in_rth[i - 1]:
            continue
        if np.isnan(vwap[i]) or np.isnan(vwap[i - 1]):
            continue
        if gaps[i - SIGMA_BARS + 1: i + hold + 1].any():
            continue
        if i <= hold_until:
            continue

        prev_close = closes[i - 1]
        curr_close = closes[i]
        prev_vwap  = vwap[i - 1]
        curr_vwap  = vwap[i]

        # Detect cross: close crosses VWAP between bar i-1 and bar i
        long_cross  = (prev_close < prev_vwap) and (curr_close >= curr_vwap)
        short_cross = (prev_close > prev_vwap) and (curr_close <= curr_vwap)

        if not long_cross and not short_cross:
            continue

        direction = 1 if long_cross else -1

        # σ estimate (same 20-bar trailing window as VWASLR)
        trail_rets = np.log(closes[i - SIGMA_BARS + 1: i + 1]
                          / closes[i - SIGMA_BARS:     i    ])
        sigma = float(np.std(trail_rets, ddof=1))
        if sigma == 0:
            continue

        entry      = curr_close
        sigma_pts  = sigma * entry
        tgt_price  = entry * math.exp( direction * TARGET_SIGMA * sigma)
        stop_price = entry * math.exp(-direction * STOP_SIGMA   * sigma)

        hit_tgt = hit_stop = None
        for j in range(i + 1, i + hold + 1):
            h, l = highs[j], lows[j]
            if hit_tgt is None:
                if direction == 1 and h >= tgt_price:
                    hit_tgt = j - i
                elif direction == -1 and l <= tgt_price:
                    hit_tgt = j - i
            if hit_stop is None:
                if direction == 1 and l <= stop_price:
                    hit_stop = j - i
                elif direction == -1 and h >= stop_price:
                    hit_stop = j - i

        time_exit_ret = math.log(closes[i + hold] / entry) * direction / sigma
        hold_until    = i + hold

        records.append({
            "year":          pd.Timestamp(ts_arr[i]).year,
            "sigma_pts":     sigma_pts,
            "direction":     direction,
            "hit_tgt":       hit_tgt,
            "hit_stop":      hit_stop,
            "time_exit_ret": time_exit_ret,
        })

    return pd.DataFrame(records)


# ── EV / stats (identical logic to backtest_vwaslr.py) ───────────────────────

def ev_stats(df: pd.DataFrame) -> dict:
    if len(df) < MIN_N:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(df), "ev_sigma": float("nan")}
    ht = df["hit_tgt"].notna().values
    hs = df["hit_stop"].notna().values
    tgt_bar  = df["hit_tgt"].fillna(999).values
    stop_bar = df["hit_stop"].fillna(999).values
    ht_first = ht & (tgt_bar <= stop_bar)
    hs_first = hs & (stop_bar < tgt_bar)
    neither  = ~ht_first & ~hs_first
    p_tgt    = ht_first.mean()
    p_stop   = hs_first.mean()
    te       = df["time_exit_ret"].values
    ev_nei   = te[neither].mean() if neither.any() else 0.0
    ev       = p_tgt * TARGET_SIGMA - p_stop * STOP_SIGMA + neither.mean() * ev_nei

    # Per-trade σ-unit returns for Sharpe
    trade_rets = np.where(ht_first, TARGET_SIGMA,
                 np.where(hs_first, -STOP_SIGMA, te))
    sharpe = (trade_rets.mean() / trade_rets.std(ddof=1) * math.sqrt(len(trade_rets))
              if trade_rets.std(ddof=1) > 0 else float("nan"))
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(df),
            "ev_sigma": sharpe}


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(sym: str, results: dict[int, pd.DataFrame]):
    pv = POINT_VALUE.get(sym, 1.0)

    print(f"\n{'═'*80}")
    print(f"  {sym}  VWAP-CROSS  vs  VWASLR  —  RTH only  "
          f"(stop={STOP_SIGMA:.0f}σ / target={TARGET_SIGMA:.0f}σ)")
    print(f"{'═'*80}")

    header = (f"  {'hold':>5}  {'N':>6}  {'P(tgt)':>7}  {'P(stop)':>8}  "
              f"{'EV(σ)':>8}  {'AvgPts':>8}  {'TotPts':>9}  {'Tot$':>9}")
    print(header)
    print(f"  {'─'*78}")

    for hold in HOLD_VALUES:
        df  = results[hold]
        st  = ev_stats(df)
        if math.isnan(st["ev"]):
            print(f"  {hold*TF_MINUTES:>4}m  {st['n']:>6}  {'—':>7}  {'—':>8}  "
                  f"{'—':>8}  {'—':>8}  {'—':>9}  {'—':>9}")
            continue
        # Avg pts via median σ_pts × EV
        sp       = df["sigma_pts"].values
        avg_pts  = np.median(sp) * st["ev"]
        tot_pts  = avg_pts * st["n"]
        tot_dol  = tot_pts * pv
        print(f"  {hold*TF_MINUTES:>4}m  {st['n']:>6}  {st['p_tgt']:>7.3f}  "
              f"{st['p_stop']:>8.3f}  {st['ev']:>+8.4f}  "
              f"{avg_pts:>+8.2f}  {tot_pts:>+9.1f}  ${tot_dol:>+9,.0f}")

    # Year-by-year for the median hold
    mid_hold = HOLD_VALUES[len(HOLD_VALUES) // 2]
    df = results[mid_hold]
    print(f"\n  Year-by-year  (hold={mid_hold*TF_MINUTES}min)")
    print(f"  {'Year':<6}  {'N':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV':>9}")
    print(f"  {'─'*45}")
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        st  = ev_stats(sub)
        if math.isnan(st["ev"]):
            print(f"  {yr:<6}  {st['n']:>5}  {'—':>7}  {'—':>8}  {'—':>9}")
        else:
            flag = "◄" if st["ev"] > 0 else ""
            print(f"  {yr:<6}  {st['n']:>5}  {st['p_tgt']:>7.3f}  "
                  f"{st['p_stop']:>8.3f}  {st['ev']:>+9.4f}σ  {flag}")

    # Direction split for mid hold
    print(f"\n  Direction split  (hold={mid_hold*TF_MINUTES}min)")
    for lbl, sub in [("LONG ", df[df["direction"] == 1]),
                     ("SHORT", df[df["direction"] == -1])]:
        st = ev_stats(sub)
        ev_s = f"{st['ev']:+.4f}σ" if not math.isnan(st["ev"]) else "—"
        print(f"    {lbl}  n={st['n']:>5}  P(tgt)={st['p_tgt']:.3f}  "
              f"P(stop)={st['p_stop']:.3f}  EV={ev_s}")


def compare_table(sym: str,
                  vwap_results: dict[int, pd.DataFrame],
                  vwaslr_summary: dict):
    """
    Side-by-side comparison for the best-matching hold.
    vwaslr_summary keys: n, p_tgt, p_stop, ev, avg_pts, tot_pts, tot_dol
    """
    pv   = POINT_VALUE.get(sym, 1.0)
    hold = 4  # VWASLR best is hold=4 (20 min) — match it

    df_vc = vwap_results.get(hold)
    if df_vc is None or len(df_vc) == 0:
        return

    st_vc = ev_stats(df_vc)
    sp_vc = df_vc["sigma_pts"].values
    avg_pts_vc = np.median(sp_vc) * st_vc["ev"] if not math.isnan(st_vc["ev"]) else float("nan")
    tot_pts_vc = avg_pts_vc * st_vc["n"]         if not math.isnan(avg_pts_vc) else float("nan")
    tot_dol_vc = tot_pts_vc * pv                  if not math.isnan(tot_pts_vc) else float("nan")

    print(f"\n{'═'*80}")
    print(f"  {sym}  HEAD-TO-HEAD  (hold={hold*TF_MINUTES}min, same stop/target/σ-window)")
    print(f"{'═'*80}")
    print(f"  {'Metric':<18}  {'VWAP-cross':>14}  {'VWASLR':>14}  {'Δ':>10}")
    print(f"  {'─'*60}")

    vs = vwaslr_summary
    def row(label, vc_val, vw_val, fmt=".3f"):
        sign_fmt = fmt.lstrip("+")   # avoid double + in format spec
        if math.isnan(vc_val) or math.isnan(vw_val):
            delta_s = "—"
        else:
            delta = vc_val - vw_val
            delta_s = ("+" if delta >= 0 else "") + format(delta, sign_fmt)
        vc_s = format(vc_val, sign_fmt) if not math.isnan(vc_val) else "—"
        vw_s = format(vw_val, sign_fmt) if not math.isnan(vw_val) else "—"
        print(f"  {label:<18}  {vc_s:>14}  {vw_s:>14}  {delta_s:>10}")

    row("N trades",     float(st_vc["n"]),    float(vs["n"]),    ".0f")
    row("P(target)",    st_vc["p_tgt"],       vs["p_tgt"])
    row("P(stop)",      st_vc["p_stop"],      vs["p_stop"])
    row("EV (σ/trade)", st_vc["ev"],          vs["ev"],          "+.4f")
    row("Avg pts",      avg_pts_vc,           vs["avg_pts"],     "+.2f")
    row("Total pts",    tot_pts_vc,           vs["tot_pts"],     "+.1f")
    row("Total $",      tot_dol_vc,           vs["tot_dol"],     "+,.0f")


# ── Main ──────────────────────────────────────────────────────────────────────

# VWASLR reference numbers from prior backtest run
# (N=10, thr=0.5σ, hold=4 bars=20min, MES, 2019-2026)
VWASLR_REF = {
    "MES": {"n": 1517, "p_tgt": 0.452, "p_stop": 0.340,
            "ev": +0.5 * 3.0 - 0.340 * 2.0,   # ~0.772σ approx from published stats
            "avg_pts": 1.10, "tot_pts": 1663.2, "tot_dol": 8316.0},
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", nargs="+", default=["MES"])
    args = parser.parse_args()

    for sym in args.sym:
        if sym not in INSTRUMENTS:
            print(f"Unknown symbol {sym}, skipping")
            continue

        path = INSTRUMENTS[sym]
        print(f"\nLoading {path} …")
        bars = load_and_resample(path)
        print(f"  {len(bars):,} {TF_MINUTES}-min bars  "
              f"({bars['ts'].min().date()} → {bars['ts'].max().date()})")
        print("  Computing session VWAP …")
        bars = add_session_vwap(bars)

        results: dict[int, pd.DataFrame] = {}
        for hold in HOLD_VALUES:
            df = scan(bars, hold)
            results[hold] = df
            print(f"  hold={hold*TF_MINUTES:>3}min → {len(df):,} trades")

        report(sym, results)

        if sym in VWASLR_REF:
            compare_table(sym, results, VWASLR_REF[sym])
