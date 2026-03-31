"""
Standalone VWASLR signal backtest.

Volume-Weighted Average Scaled Log Return:
  VWASLR_N[i] = Σ(scaled_ret_j × vol_j) / Σ(vol_j)  for j in [i-N+1 .. i]
  scaled_ret_j = log(close_j / close_{j-1}) / σ

Signal fires when VWASLR crosses ±threshold — entering in the direction of
the persistent drift (trend-following, not counter-trend).

Sweeps:
  N         : VWASLR window in bars      (6, 8, 10, 12)
  threshold : trigger level in σ/bar     (0.3, 0.5, 0.7, 1.0)
  hold      : bars to hold after entry   (2, 3, 4, 5, 6)

Outcome measured with fixed hold + 2σ stop / 3σ target (same as primary signal).
No re-entry while a hold is active.

Usage:
  python src/backtest_vwaslr.py --sym MES
  python src/backtest_vwaslr.py --sym MES MYM M2K
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
TF_MINUTES           = 5
TRAILING_BARS        = 20    # default σ window: 20 × 5-min = 100 min

RTH_START    = (9, 30)
RTH_END      = (16, 0)
GLOBEX_START = (18, 0)    # 18:00 ET evening open
GLOBEX_END   = (9, 30)    # through to 09:29 ET next morning

STOP_SIGMA   = 2.0
TARGET_SIGMA = 3.0

N_VALUES             = [6, 8, 10, 12]
THRESHOLD_VALUES     = [0.3, 0.5, 0.7, 1.0]
RAW_THRESHOLD_VALUES = [0.0001, 0.0002, 0.0005, 0.001, 0.002]  # for σ=0 (raw VWALR)
HOLD_VALUES          = [2, 3, 4, 5, 6]

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
}

MIN_N_FOR_STATS = 20   # skip combos with fewer trades than this


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_and_resample(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    # Remove CME settlement gap
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # Resample to TF_MINUTES using gap-aware chunking
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
        })
        i += TF_MINUTES

    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF_MINUTES)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Signal scan ───────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame, n_win: int, threshold: float,
         hold: int, session: str = "rth",
         sigma_bars: int = TRAILING_BARS) -> pd.DataFrame:
    """
    For each bar (after warmup), compute VWASLR_n. Fire a trade when VWASLR
    crosses ±threshold for the first time (direction = sign of VWASLR).
    Evaluate outcome over next `hold` bars with STOP_SIGMA stop / TARGET_SIGMA
    target.  No re-entry during an active hold.

    sigma_bars : number of trailing bars for σ estimate.
                 0  → no scaling (raw volume-weighted log return, VWALR).
                 20 → adaptive 100-min σ (default, original).
                 100 → slow 500-min σ (near-constant baseline test).
    """
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    nb      = len(bars)

    sig_win    = max(sigma_bars, 1)   # bars needed for σ (0 → skip σ calc)
    warmup     = max(sig_win, n_win) + 1
    hold_until = -1
    records    = []

    for i in range(warmup, nb - hold):
        # Skip bars with gaps in any relevant window
        if gaps[i - sig_win + 1: i + hold + 1].any():
            continue

        # Session filter
        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        if session == "rth":
            if bar_hm < RTH_START or bar_hm >= RTH_END:
                continue
        else:  # globex: 18:00–09:29 ET
            if GLOBEX_END <= bar_hm < GLOBEX_START:
                continue

        # σ estimate
        if sigma_bars == 0:
            sigma = 1.0   # no scaling: threshold is in raw return units
        else:
            trail_rets = np.log(closes[i - sigma_bars + 1: i + 1]
                              / closes[i - sigma_bars:     i    ])
            sigma = float(np.std(trail_rets, ddof=1))
            if sigma == 0:
                continue

        # VWASLR (or VWALR if sigma_bars=0) over last n_win bars
        ret_window = np.log(closes[i - n_win + 1: i + 1]
                          / closes[i - n_win:     i    ])
        vol_window = volumes[i - n_win: i]
        sum_vol    = vol_window.sum()
        if sum_vol == 0:
            continue
        scaled_rets = ret_window / sigma
        vwaslr      = float((scaled_rets * vol_window).sum() / sum_vol)

        # Only fire if VWASLR has crossed threshold (not already in a trade)
        if abs(vwaslr) < threshold:
            continue
        if i <= hold_until:
            continue

        direction = 1 if vwaslr > 0 else -1
        entry     = closes[i]
        sigma_pts = sigma * entry

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
            "year":          ts_pd[i].year,
            "vwaslr":        vwaslr,
            "sigma_pts":     sigma_pts,
            "direction":     direction,
            "hit_tgt":       hit_tgt,
            "hit_stop":      hit_stop,
            "time_exit_ret": time_exit_ret,
        })

    return pd.DataFrame(records)


# ── EV helpers ────────────────────────────────────────────────────────────────

def ev_stats(df: pd.DataFrame) -> dict:
    if len(df) < MIN_N_FOR_STATS:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(df)}
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
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(df)}


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary_table(sym: str, results: dict,
                        thr_values: list | None = None):
    """
    Print summary matrix: rows = (n_win, threshold), columns = hold bars.
    Show EV for each combo; highlight positive EVs.
    """
    if thr_values is None:
        thr_values = THRESHOLD_VALUES
    thr_fmt = ".4f" if max(thr_values) < 0.1 else ".1f"

    print(f"\n{'═'*80}")
    print(f"  {sym}  VWASLR  —  EV summary  "
          f"(stop={STOP_SIGMA:.0f}σ / target={TARGET_SIGMA:.0f}σ, NYSE RTH only)")
    print(f"{'═'*80}")

    hold_header = "  ".join(f"hold={h:1d}" for h in HOLD_VALUES)
    print(f"\n  {'N':>3}  {'thr':>7}  {'n_med':>6}    {hold_header}")
    print(f"  {'─'*72}")

    best_ev = -999.0
    best_key = None

    for n_win in N_VALUES:
        for thr in thr_values:
            row_parts = []
            n_vals = []
            for hold in HOLD_VALUES:
                key = (n_win, thr, hold)
                if key not in results:
                    row_parts.append(f"{'—':>7}")
                    continue
                st = ev_stats(results[key])
                n_vals.append(st["n"])
                if math.isnan(st["ev"]):
                    row_parts.append(f"{'—':>7}")
                else:
                    tag = "◄" if st["ev"] > 0 else " "
                    row_parts.append(f"{st['ev']:>+6.3f}{tag}")
                    if st["ev"] > best_ev and st["n"] >= MIN_N_FOR_STATS:
                        best_ev  = st["ev"]
                        best_key = key
            n_med = int(np.median(n_vals)) if n_vals else 0
            thr_str = format(thr, thr_fmt).rjust(7)
            print(f"  {n_win:>3}  {thr_str}  {n_med:>6}  " +
                  "  ".join(row_parts))
        print()

    if best_key:
        print(f"  Best: N={best_key[0]}, thr={best_key[1]:.1f}, "
              f"hold={best_key[2]} → EV={best_ev:+.4f}σ")


def print_best_detail(sym: str, results: dict, top_n: int = 3,
                      raw_units: bool = False):
    """Print year-by-year and direction split for the top combos by EV."""
    ranked = []
    for key, df in results.items():
        st = ev_stats(df)
        if not math.isnan(st["ev"]) and st["n"] >= MIN_N_FOR_STATS:
            ranked.append((st["ev"], key, df, st))
    ranked.sort(reverse=True)

    print(f"\n{'═'*80}")
    print(f"  {sym}  VWASLR  —  Top {top_n} combos detail")
    print(f"{'═'*80}")

    for rank, (ev, key, df, st) in enumerate(ranked[:top_n], 1):
        n_win, thr, hold = key
        thr_lbl = f"±{thr:.4f} raw" if raw_units else f"±{thr:.1f}σ/bar"
        print(f"\n  #{rank}  N={n_win} bars ({n_win*TF_MINUTES}min)  "
              f"thr={thr_lbl}  hold={hold} bars ({hold*TF_MINUTES}min)"
              f"  →  EV={ev:+.4f}σ  n={st['n']}  "
              f"P(tgt)={st['p_tgt']:.3f}  P(stop)={st['p_stop']:.3f}")

        # Year-by-year
        print(f"\n     {'Year':<6}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV':>9}")
        print(f"     {'─'*40}")
        for yr in sorted(df["year"].unique()):
            sub = df[df["year"] == yr]
            s   = ev_stats(sub)
            if math.isnan(s["ev"]):
                print(f"     {yr:<6}  {s['n']:>5}  {'—':>7}  {'—':>8}  {'—':>9}")
            else:
                flag = "◄" if s["ev"] > 0 else ""
                print(f"     {yr:<6}  {s['n']:>5}  {s['p_tgt']:>7.3f}  "
                      f"{s['p_stop']:>8.3f}  {s['ev']:>+9.4f}σ  {flag}")

        # Long vs short split
        longs  = df[df["direction"] ==  1]
        shorts = df[df["direction"] == -1]
        print(f"\n     Direction split:")
        for lbl, sub in [("LONG ", longs), ("SHORT", shorts)]:
            s = ev_stats(sub)
            if math.isnan(s["ev"]):
                ev_s = "—"
            else:
                ev_s = f"{s['ev']:+.4f}σ"
            print(f"       {lbl}  n={s['n']:>4}  P(tgt)={s['p_tgt']:.3f}  "
                  f"P(stop)={s['p_stop']:.3f}  EV={ev_s}")

        # σ_pts distribution
        sp = df["sigma_pts"].values
        print(f"\n     σ_pts  median={np.median(sp):.2f}  "
              f"tgt({TARGET_SIGMA:.0f}σ)={np.median(sp)*TARGET_SIGMA:.2f}  "
              f"stop({STOP_SIGMA:.0f}σ)={np.median(sp)*STOP_SIGMA:.2f} pts")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", nargs="+", default=["MES"],
                        help="Symbols to run (default: MES)")
    parser.add_argument("--session", default="rth", choices=["rth", "globex"],
                        help="rth=09:30–16:00 ET  globex=18:00–09:29 ET (default: rth)")
    parser.add_argument("--sigma-bars", type=int, default=None,
                        help="Trailing bars for σ: 20=adaptive (default), "
                             "100=slow/stable, 0=no scaling (raw VWALR)")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of top combos to detail (default: 3)")
    args = parser.parse_args()

    # Determine σ configurations to run
    if args.sigma_bars is not None:
        sigma_configs = [args.sigma_bars]
    else:
        sigma_configs = [20]   # default: adaptive only

    for sym in args.sym:
        if sym not in INSTRUMENTS:
            print(f"Unknown symbol {sym}, skipping")
            continue

        path = INSTRUMENTS[sym]
        print(f"\nLoading {path} …")
        bars = load_and_resample(path)
        print(f"  {len(bars):,} {TF_MINUTES}-min bars  "
              f"({bars['ts'].min().date()} → {bars['ts'].max().date()})")

        for sb in sigma_configs:
            raw_mode = (sb == 0)
            thr_values = RAW_THRESHOLD_VALUES if raw_mode else THRESHOLD_VALUES
            sb_label = (f"σ-window={sb} bars ({sb*TF_MINUTES}min)"
                        if sb > 0 else "no σ-scaling (raw VWALR, thresholds in raw log-return units)")
            print(f"\n  ── {sb_label} ──")

            results: dict[tuple, pd.DataFrame] = {}
            total = len(N_VALUES) * len(thr_values) * len(HOLD_VALUES)
            done  = 0
            for n_win in N_VALUES:
                for thr in thr_values:
                    for hold in HOLD_VALUES:
                        df = scan(bars, n_win, thr, hold,
                                  session=args.session, sigma_bars=sb)
                        results[(n_win, thr, hold)] = df
                        done += 1
                        thr_disp = f"{thr:.4f}" if raw_mode else f"{thr:.1f}"
                        print(f"  [{done:>3}/{total}] N={n_win} thr={thr_disp} "
                              f"hold={hold}  → {len(df):,} trades", end="\r")

            print(f"  Sweep complete ({total} combos)                    ")
            print_summary_table(sym, results, thr_values=thr_values)
            print_best_detail(sym, results, top_n=args.top, raw_units=raw_mode)
