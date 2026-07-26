"""
Globex VWASLR backtest — MES, 1-min bars, short signal windows.

Tests whether VWASLR with shorter N (10 / 15 / 20 min) has positive EV
during overnight Globex hours, where the 50-min production window is known
to fail due to order-book fragility.

Session windows:
  Evening   18:00–22:00 ET  (post-NYSE-close)
  Overnight 22:00–06:00 ET  (deep overnight)
  Pre-open  06:00–09:30 ET  (early pre-market)
  Full Glob 18:00–09:30 ET  (all non-RTH, gap excluded)

Parameters swept:
  N         : 10, 15, 20  (1-min bars = minutes)
  sigma_bars: 500
  threshold : 0.1 … 0.5
  stop / tgt: (1.0/1.5), (1.0/2.0), (1.5/2.0), (1.5/3.0)
  hold      : N bars

Uses vectorised rolling ops + signal-index iteration (fast).

Usage:
  python src/backtest_vwaslr_globex.py
"""

import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

CSV_PATH        = "mes_hist_1min.csv"
ET              = ZoneInfo("America/New_York")
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

N_VALUES        = [10, 15, 20]
SIGMA_BARS      = 500
THRESHOLDS      = [0.1, 0.2, 0.3, 0.4, 0.5]
STOP_TGT_PAIRS  = [(1.0, 1.5), (1.0, 2.0), (1.5, 2.0), (1.5, 3.0)]
MIN_N           = 20

WINDOWS = {
    "Evening   18–22 ET": (18, 22),
    "Overnight 22–06 ET": (22, 30),   # 00–06 ET stored as 24–30
    "Pre-open  06–09 ET": (6,   9),
    "Full Globex        ": (18, 34),  # 18:00 → 09:30 (09:30 = adj-hour 33.5, use 34)
}


# ── Data ─────────────────────────────────────────────────────────────────────

def load_bars() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")
    df = df.sort_values("ts").reset_index(drop=True)

    h_utc = df["ts"].dt.hour
    df = df[~((h_utc >= SETTLEMENT_UTC_START) & (h_utc < SETTLEMENT_UTC_END))].copy()
    df = df.reset_index(drop=True)

    ts_et = df["ts"].dt.tz_convert(ET)
    et_h  = ts_et.dt.hour.values
    et_m  = ts_et.dt.minute.values

    # hour_adj: 00–09 ET → 24–33 so overnight comparisons don't wrap
    df["hour_adj"] = np.where(et_h < 10, et_h + 24, et_h)
    df["hm_adj"]   = df["hour_adj"] * 60 + et_m

    # Gap flag (>2-min jump)
    df["gap"] = (df["ts"].diff() > pd.Timedelta(minutes=2)).values
    df.loc[0, "gap"] = True

    # Globex: not RTH 09:30–16:00 ET
    hm_raw = et_h * 60 + et_m
    df["is_globex"] = ~((hm_raw >= 570) & (hm_raw < 960))

    df["year"] = df["ts"].dt.year.values
    return df


# ── Rolling VWASLR + sigma ────────────────────────────────────────────────────

def compute_rolling(df: pd.DataFrame, n: int, sigma_bars: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (vwaslr, sigma) as numpy arrays (NaN where insufficient/gap).
    sigma is the rolling close-return std (ddof=1) over sigma_bars bars.
    vwaslr = Σ(ret_j * vol_j) / (Σvol_j * sigma)
    """
    c   = df["close"].values.astype(float)
    v   = df["volume"].values.astype(float)
    gap = df["gap"].values.astype(bool)
    nb  = len(df)

    log_ret = np.empty(nb)
    log_ret[0] = np.nan
    log_ret[1:] = np.log(c[1:] / c[:-1])
    log_ret[gap] = np.nan          # break continuity at gaps

    lr_s  = pd.Series(log_ret)
    vol_s = pd.Series(np.where(gap, np.nan, v))

    # Rolling sigma (slow window)
    sigma_arr = lr_s.rolling(sigma_bars, min_periods=sigma_bars).std(ddof=1).values

    # Rolling volume-weighted return numerator and denominator
    rv_arr  = (lr_s * vol_s).rolling(n, min_periods=n).sum().values
    sv_arr  = vol_s.rolling(n, min_periods=n).sum().values

    # Gap-safety: no gap in the sigma window [i-sigma_bars .. i]
    gap_cum = np.cumsum(gap.astype(int))
    safe    = np.zeros(nb, dtype=bool)
    safe[sigma_bars:] = (gap_cum[sigma_bars:] == gap_cum[:nb - sigma_bars])

    with np.errstate(invalid="ignore", divide="ignore"):
        vwaslr = np.where(
            safe & (sigma_arr > 0) & (sv_arr > 0),
            rv_arr / (sv_arr * sigma_arr),
            np.nan,
        )

    return vwaslr, sigma_arr


# ── Trade simulation ─────────────────────────────────────────────────────────

def run_trades(df: pd.DataFrame, vwaslr: np.ndarray, sigma: np.ndarray,
               threshold: float, stop_s: float, tgt_s: float,
               hold: int) -> pd.DataFrame:
    """
    Detect EMA-less threshold crossings on Globex bars.
    Uses the same sigma as the VWASLR computation for stop/target.
    """
    c      = df["close"].values
    hi     = df["high"].values
    lo     = df["low"].values
    gap    = df["gap"].values.astype(bool)
    gx     = df["is_globex"].values.astype(bool)
    h_adj  = df["hour_adj"].values
    year   = df["year"].values
    nb     = len(df)

    # Vectorised crossing detection: |v[i]| >= thr AND |v[i-1]| < thr
    abs_v = np.abs(vwaslr)
    crossed = (
        (abs_v[1:] >= threshold) &
        (abs_v[:-1] < threshold) &
        gx[1:] &
        ~np.isnan(vwaslr[1:]) &
        ~np.isnan(vwaslr[:-1]) &
        ~np.isnan(sigma[1:])
    )
    signal_idx = np.where(crossed)[0] + 1   # actual bar index

    hold_until = -1
    records    = []

    for i in signal_idx:
        if i <= hold_until:
            continue
        if i + hold >= nb:
            continue
        # No gap in hold window
        if gap[i + 1: i + hold + 1].any():
            continue

        sig = sigma[i]
        if sig == 0 or np.isnan(sig):
            continue

        direction  = 1 if vwaslr[i] > 0 else -1
        entry      = c[i]
        tgt_price  = entry * math.exp( direction * tgt_s  * sig)
        stop_price = entry * math.exp(-direction * stop_s * sig)

        hit_tgt = hit_stop = None
        for j in range(i + 1, i + hold + 1):
            hj, lj = hi[j], lo[j]
            if direction == 1:
                if hit_stop is None and lj <= stop_price: hit_stop = j - i
                if hit_tgt  is None and hj >= tgt_price:  hit_tgt  = j - i
            else:
                if hit_stop is None and hj >= stop_price: hit_stop = j - i
                if hit_tgt  is None and lj <= tgt_price:  hit_tgt  = j - i

        te_ret     = math.log(c[i + hold] / entry) * direction / sig
        hold_until = i + hold

        records.append({
            "year":          year[i],
            "hour_adj":      h_adj[i],
            "direction":     direction,
            "hit_tgt":       hit_tgt,
            "hit_stop":      hit_stop,
            "time_exit_ret": te_ret,
        })

    return pd.DataFrame(records)


# ── EV helper ─────────────────────────────────────────────────────────────────

def ev_stats(df: pd.DataFrame, tgt_s: float, stop_s: float) -> dict:
    if len(df) < MIN_N:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(df)}
    ht       = df["hit_tgt"].notna().values
    hs       = df["hit_stop"].notna().values
    tgt_bar  = df["hit_tgt"].fillna(999).values
    stop_bar = df["hit_stop"].fillna(999).values
    ht_first = ht & (tgt_bar <= stop_bar)
    hs_first = hs & (stop_bar < tgt_bar)
    neither  = ~ht_first & ~hs_first
    te       = df["time_exit_ret"].values
    ev_nei   = float(te[neither].mean()) if neither.any() else 0.0
    ev       = float(ht_first.mean() * tgt_s
                     - hs_first.mean() * stop_s
                     + neither.mean() * ev_nei)
    return {"ev": ev, "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": len(df)}


def wmask(df: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    return (df["hour_adj"] >= lo) & (df["hour_adj"] < hi)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*78}")
    print(f"  MES — Globex VWASLR Backtest  (N = 10 / 15 / 20 min, 1-min bars)")
    print(f"  σ-window={SIGMA_BARS}min  hold=N bars  conservative OHLC")
    print(f"{'═'*78}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} 1-min bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")
    print(f"  Globex bars: {int(df['is_globex'].sum()):,}")

    print("\nComputing rolling VWASLR for each N …", flush=True)
    rolling: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in N_VALUES:
        print(f"  N={n} …", end=" ", flush=True)
        vw, sg = compute_rolling(df, n, SIGMA_BARS)
        n_sig = int((np.abs(vw) >= THRESHOLDS[0]).sum())
        print(f"done  ({n_sig:,} bars ≥ min threshold {THRESHOLDS[0]})")
        rolling[n] = (vw, sg)

    print("\nRunning trade simulations …", flush=True)
    results: dict[tuple, pd.DataFrame] = {}
    for n in N_VALUES:
        vw, sg = rolling[n]
        for thr in THRESHOLDS:
            for stop_s, tgt_s in STOP_TGT_PAIRS:
                key = (n, thr, stop_s, tgt_s)
                results[key] = run_trades(df, vw, sg, thr, stop_s, tgt_s, n)
        total = sum(len(results[(n, thr, s, t)])
                    for thr in THRESHOLDS for s, t in STOP_TGT_PAIRS)
        print(f"  N={n}: {total} trades total across all param combos", flush=True)

    primary = (1.0, 2.0)

    # ── Table 1: EV by N × threshold ─────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  EV BY N × THRESHOLD  (stop={primary[0]}σ / target={primary[1]}σ  |  Full Globex)")
    print(f"{'─'*78}")
    thr_hdr = "   ".join(f"thr±{t:.1f}" for t in THRESHOLDS)
    print(f"  {'N':>4}  hold   {thr_hdr}")
    print(f"  {'─'*74}")
    for n in N_VALUES:
        parts = []
        for thr in THRESHOLDS:
            st = ev_stats(results[(n, thr, *primary)], primary[1], primary[0])
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>12}")
            else:
                flag = "◄" if st["ev"] > 0 else " "
                parts.append(f"{st['ev']:>+8.4f}{flag}({st['n']:3}n)")
        print(f"  {n:>4}  {n:>4}m   " + "   ".join(parts))

    # ── Table 2: EV by stop/target combo ─────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  EV BY STOP/TARGET COMBO  (Full Globex)")
    print(f"{'─'*78}")
    combo_hdr = "      ".join(f"s{s:.1f}/t{t:.1f}" for s, t in STOP_TGT_PAIRS)
    print(f"  {'N':>3}  thr    {combo_hdr}")
    print(f"  {'─'*78}")
    for n in N_VALUES:
        for thr in THRESHOLDS:
            parts = []
            for stop_s, tgt_s in STOP_TGT_PAIRS:
                st = ev_stats(results[(n, thr, stop_s, tgt_s)], tgt_s, stop_s)
                if math.isnan(st["ev"]):
                    parts.append(f"{'—':>14}")
                else:
                    flag = "◄" if st["ev"] > 0 else " "
                    parts.append(f"{st['ev']:>+8.4f}{flag}({st['n']:3}n)")
            print(f"  {n:>3}  ±{thr:.1f}   " + "      ".join(parts))
        print()

    # ── Table 3: Session window breakdown ────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  EV BY SESSION WINDOW  (stop={primary[0]}σ / target={primary[1]}σ)")
    print(f"{'─'*78}")
    for n in N_VALUES:
        for thr in THRESHOLDS:
            df_r = results[(n, thr, *primary)]
            if df_r.empty:
                continue
            row_parts = []
            any_pos = False
            for lbl, (lo, hi) in WINDOWS.items():
                sub = df_r[wmask(df_r, lo, hi)]
                st  = ev_stats(sub, primary[1], primary[0])
                if math.isnan(st["ev"]):
                    row_parts.append(f"  {lbl}: {'—':>8} ({st['n']:3}n)")
                else:
                    flag = "◄" if st["ev"] > 0 else " "
                    if st["ev"] > 0:
                        any_pos = True
                    row_parts.append(f"  {lbl}: {st['ev']:>+8.4f}{flag}({st['n']:3}n)")
            marker = "  ★" if any_pos else ""
            print(f"\n  N={n}  thr=±{thr:.1f}{marker}")
            for rp in row_parts:
                print(f"   {rp}")

    # ── Table 4: Year-by-year for best combos ────────────────────────────────
    print(f"\n\n{'─'*78}")
    print(f"  YEAR-BY-YEAR  (stop={primary[0]}σ / target={primary[1]}σ  |  Full Globex)")
    print(f"{'─'*78}")
    best = []
    for n in N_VALUES:
        for thr in THRESHOLDS:
            df_r = results[(n, thr, *primary)]
            st   = ev_stats(df_r, primary[1], primary[0])
            if not math.isnan(st["ev"]) and st["ev"] > 0:
                best.append((st["ev"], n, thr, df_r))
    best.sort(reverse=True)

    if not best:
        print("  No parameter combo produced positive EV.")
    else:
        for ev_val, n, thr, df_r in best[:5]:
            years = sorted(df_r["year"].unique())
            print(f"\n  N={n}  thr=±{thr:.1f}  overall EV={ev_val:+.4f}σ  (n={len(df_r)})")
            print(f"  {'Year':<6}  {'n':>4}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV':>10}")
            print(f"  {'─'*44}")
            for yr in years:
                sub = df_r[df_r["year"] == yr]
                st  = ev_stats(sub, primary[1], primary[0])
                if math.isnan(st["ev"]):
                    print(f"  {yr}   {st['n']:>4}  {'—':>7}  {'—':>8}  {'—':>10}")
                else:
                    flag = " ◄" if st["ev"] > 0 else "  "
                    print(f"  {yr}   {st['n']:>4}  {st['p_tgt']:.3f}    "
                          f"{st['p_stop']:.3f}  {st['ev']:>+10.4f}{flag}")

    # ── Table 5: Direction split ───────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  DIRECTION SPLIT  (stop={primary[0]}σ / target={primary[1]}σ  |  Full Globex)")
    print(f"{'─'*78}")
    print(f"  {'N':>3}  thr    {'Dir':<6}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV':>10}")
    print(f"  {'─'*62}")
    for n in N_VALUES:
        for thr in THRESHOLDS:
            df_r = results[(n, thr, *primary)]
            if df_r.empty:
                continue
            for dir_lbl, d in [("LONG ", 1), ("SHORT", -1)]:
                sub = df_r[df_r["direction"] == d]
                st  = ev_stats(sub, primary[1], primary[0])
                ev_s = f"{st['ev']:>+10.4f}σ" if not math.isnan(st["ev"]) else f"{'—':>11}"
                flag = " ◄" if not math.isnan(st.get("ev", float("nan"))) and st["ev"] > 0 else ""
                p_t  = f"{st['p_tgt']:.3f}" if not math.isnan(st.get("p_tgt", float("nan"))) else "  — "
                p_s  = f"{st['p_stop']:.3f}" if not math.isnan(st.get("p_stop", float("nan"))) else "  — "
                print(f"  {n:>3}  ±{thr:.1f}   {dir_lbl}  {st['n']:>5}  "
                      f"{p_t:>7}  {p_s:>8}  {ev_s}{flag}")
        print()
