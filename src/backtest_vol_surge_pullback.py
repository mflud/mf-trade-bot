"""
Backtest: Volume Surge + Shallow Pullback — parameter sweep.

Pattern:
  1. Volume surge  : bar volume >= mult × rolling 20-bar median
  2. Directional   : net move >= MOVE_BPS (price-scaled) in trigger bar window
  3. Pullback      : next 1–3 bars retrace 1.5–4.5bp from the extreme
  4. Entry         : open of bar after pullback bar
  5. Exit          : target (bp of entry price), stop (fixed pts), or time cap

Sweeps:
  Vol multiplier : 5×, 6×, 7×, 8×
  Move threshold : 5, 8, 10, 12 bp
  Target         : 15, 20 bp  (price-scaled; stop stays fixed at 3 pts)
  Hold (time cap): 6, 8, 10, 15 bars

Sessions: All / RTH (09:30–16:00 ET) / Globex (17:00–09:30 ET)

Data: mes_hist_1min.csv

Usage:
  python src/backtest_vol_surge_pullback.py
"""

import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

CSV_PATH             = "mes_hist_1min.csv"
ET                   = ZoneInfo("America/New_York")
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

PULLBACK_MIN_BPS = 1.5
PULLBACK_MAX_BPS = 4.5
PULLBACK_BARS    = 3
VOL_LOOKBACK     = 20
STOP_PTS         = 3.0    # fixed invalidation stop (matches pullback depth)

VOL_MULTS    = [5, 6, 7, 8]
MOVE_BPS_LIST = [5, 8, 10, 12]
TARGET_BPS_LIST = [15, 20]
HOLD_BARS    = [6, 8, 10, 15]
MIN_TRADES   = 20

# Reference prices for context footnotes
PRICE_NOW  = 6640
PRICE_2019 = 2900


# ── Data ─────────────────────────────────────────────────────────────────────

def load_bars() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")
    df = df.sort_values("ts").reset_index(drop=True)

    h = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    df = df.reset_index(drop=True)

    ts_et  = df["ts"].dt.tz_convert(ET)
    et_h   = ts_et.dt.hour.values
    et_m   = ts_et.dt.minute.values
    hm_raw = et_h * 60 + et_m

    df["is_rth"] = (hm_raw >= 570) & (hm_raw < 960)
    df["year"]   = df["ts"].dt.year.values
    df["gap"]    = (df["ts"].diff() > pd.Timedelta(minutes=2)).values
    df.loc[0, "gap"] = True
    return df


# ── Phase 1: find signal candidates ──────────────────────────────────────────

def find_candidates(df: pd.DataFrame, vol_mult: float, move_bps: float,
                    hold_max: int) -> pd.DataFrame:
    """
    Find all valid surge→pullback entries (ignoring hold_until overlap).
    Store per-trade: entry metadata + bar-by-bar (high_delta, low_delta)
    relative to entry price for the max hold window (direction-adjusted).
    """
    c      = df["close"].values
    o      = df["open"].values
    hi     = df["high"].values
    lo     = df["low"].values
    vol    = df["volume"].values.astype(float)
    gap    = df["gap"].values.astype(bool)
    is_rth = df["is_rth"].values.astype(bool)
    year   = df["year"].values
    nb     = len(df)

    vol_s = pd.Series(np.where(gap, np.nan, vol))
    med20 = vol_s.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().values
    gap_cum = np.cumsum(gap.astype(int))

    records = []

    for i in range(VOL_LOOKBACK + 1, nb - hold_max - PULLBACK_BARS - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        if vol[i] < vol_mult * med20[i]:
            continue

        move_threshold = c[i] * move_bps / 10000
        pb_min         = c[i] * PULLBACK_MIN_BPS / 10000
        pb_max         = c[i] * PULLBACK_MAX_BPS / 10000

        move_up = hi[i] - min(lo[i-1], lo[i])
        move_dn = max(hi[i-1], hi[i]) - lo[i]

        if move_up >= move_threshold and move_up >= move_dn:
            direction = 1
            extreme   = hi[i]
        elif move_dn >= move_threshold:
            direction = -1
            extreme   = lo[i]
        else:
            continue

        if direction == 1 and c[i] < o[i]:
            continue
        if direction == -1 and c[i] > o[i]:
            continue

        pullback_bar = None
        for j in range(i + 1, min(i + 1 + PULLBACK_BARS, nb - hold_max - 2)):
            if gap[j]:
                break
            retrace = (extreme - c[j]) * direction
            if pb_min <= retrace <= pb_max:
                pullback_bar = j
                break

        if pullback_bar is None:
            continue

        entry_bar = pullback_bar + 1
        if entry_bar + hold_max >= nb:
            continue
        if gap_cum[entry_bar + hold_max] > gap_cum[i]:
            continue

        entry = o[entry_bar]

        # Store direction-adjusted bar deltas for the full hold window
        # fav[k] = max favourable excursion at bar k (positive = in our direction)
        # adv[k] = max adverse excursion at bar k (positive = against us)
        path_fav = np.empty(hold_max)
        path_adv = np.empty(hold_max)
        for k in range(hold_max):
            bar_k = entry_bar + 1 + k
            if direction == 1:
                path_fav[k] = hi[bar_k] - entry
                path_adv[k] = entry - lo[bar_k]
            else:
                path_fav[k] = entry - lo[bar_k]
                path_adv[k] = hi[bar_k] - entry

        te_pts = (c[entry_bar + hold_max] - entry) * direction

        records.append({
            "entry_bar": entry_bar,
            "entry":     entry,
            "direction": direction,
            "is_rth":    bool(is_rth[entry_bar]),
            "year":      year[entry_bar],
            "path_fav":  path_fav,
            "path_adv":  path_adv,
            "te_pts":    te_pts,   # time-exit at hold_max
        })

    return pd.DataFrame(records)


# ── Phase 2: apply hold_until + simulate exit ─────────────────────────────────

def simulate(cands: pd.DataFrame, hold: int,
             target_bps: float) -> pd.DataFrame:
    """
    Apply hold_until (no overlapping trades) and simulate target/stop/time exit.
    target_pts = entry * target_bps / 10000  (price-scaled)
    """
    if cands.empty:
        return pd.DataFrame()

    hold_until = -1
    records    = []

    for _, row in cands.iterrows():
        eb = row["entry_bar"]
        if eb <= hold_until:
            continue

        entry      = row["entry"]
        target_pts = entry * target_bps / 10000
        path_fav   = row["path_fav"][:hold]
        path_adv   = row["path_adv"][:hold]

        # Time exit: price at bar hold (direction-adjusted), re-derive
        # from path_fav/adv not available cleanly; use stored te_pts
        # adjusted for shorter hold if needed — for now use the close
        # at hold bar relative to full-path. Approximate: use path at hold-1.
        # Better: we stored te_pts at hold_max; for shorter holds use
        # the last fav/adv bar's midpoint. Simplification: just check
        # target/stop and call the remainder a time exit.
        hit_tgt = hit_stop = None
        for k in range(hold):
            if hit_stop is None and path_adv[k] >= STOP_PTS:
                hit_stop = k + 1
            if hit_tgt  is None and path_fav[k] >= target_pts:
                hit_tgt  = k + 1

        # For time exit P&L: use stored te_pts (valid when hold==hold_max);
        # for shorter holds approximate as 0 (conservative — unknown close).
        # Since we stored hold_max=max(HOLD_BARS), use te_pts when hold matches,
        # else use last-bar fav minus adv as rough estimate.
        if hit_tgt is None and hit_stop is None:
            te = row["te_pts"] if hold == max(HOLD_BARS) else float(path_fav[hold-1] - path_adv[hold-1]) / 2
        else:
            te = 0.0

        hold_until = eb + hold

        records.append({
            "year":      row["year"],
            "is_rth":    row["is_rth"],
            "direction": row["direction"],
            "hit_tgt":   hit_tgt,
            "hit_stop":  hit_stop,
            "te_pts":    te,
            "target_pts": target_pts,
        })

    return pd.DataFrame(records)


# ── EV helper ─────────────────────────────────────────────────────────────────

def ev_stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n < MIN_TRADES:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": n}
    ht       = df["hit_tgt"].notna().values
    hs       = df["hit_stop"].notna().values
    tgt_bar  = df["hit_tgt"].fillna(999).values
    stop_bar = df["hit_stop"].fillna(999).values
    ht_first = ht & (tgt_bar <= stop_bar)
    hs_first = hs & (stop_bar < tgt_bar)
    neither  = ~ht_first & ~hs_first
    te       = df["te_pts"].values
    tgt_pts  = df["target_pts"].values
    ev_tgt   = float((ht_first * tgt_pts).mean())
    ev_stop  = float(hs_first.mean() * STOP_PTS)
    ev_nei   = float(te[neither].mean()) if neither.any() else 0.0
    ev_pts   = ev_tgt - ev_stop + neither.mean() * ev_nei
    return {"ev": ev_pts, "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": n}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*78}")
    print(f"  MES — Volume Surge + Shallow Pullback  (refined sweep)")
    print(f"  pullback {PULLBACK_MIN_BPS}–{PULLBACK_MAX_BPS}bp  stop={STOP_PTS}pts fixed")
    print(f"  target: 15bp / 20bp (price-scaled)")
    print(f"  At today ~{PRICE_NOW}: 15bp={PRICE_NOW*.0015:.1f}pts  20bp={PRICE_NOW*.002:.1f}pts")
    print(f"  At 2019  ~{PRICE_2019}: 15bp={PRICE_2019*.0015:.1f}pts  20bp={PRICE_2019*.002:.1f}pts")
    print(f"{'═'*78}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} 1-min bars  "
          f"({df['ts'].min().date()} → {df['ts'].max().date()})")

    hold_max = max(HOLD_BARS)

    # Cache candidates for each (vol_mult, move_bps)
    print("\nScanning signal candidates …", flush=True)
    cand_cache: dict[tuple, pd.DataFrame] = {}
    for vm in VOL_MULTS:
        for mb in MOVE_BPS_LIST:
            key = (vm, mb)
            cands = find_candidates(df, vm, mb, hold_max)
            cand_cache[key] = cands
            print(f"  vol≥{vm}×  move≥{mb:2}bp  →  {len(cands):5,} candidates", flush=True)

    # Simulate all combos
    print("\nSimulating exits …", flush=True)
    results: dict[tuple, pd.DataFrame] = {}
    for vm in VOL_MULTS:
        for mb in MOVE_BPS_LIST:
            for hb in HOLD_BARS:
                for tb in TARGET_BPS_LIST:
                    key = (vm, mb, hb, tb)
                    results[key] = simulate(cand_cache[(vm, mb)], hb, tb)
    print("  done", flush=True)

    # ── Tables by target_bps ──────────────────────────────────────────────────
    for tb in TARGET_BPS_LIST:
        tgt_now  = PRICE_NOW  * tb / 10000
        tgt_2019 = PRICE_2019 * tb / 10000

        for session_label, session_filter in [
            ("ALL SESSIONS", None),
            ("RTH ONLY", True),
            ("GLOBEX ONLY", False),
        ]:
            print(f"\n{'─'*78}")
            print(f"  TARGET={tb}bp  ({tgt_now:.1f}pts @ {PRICE_NOW} / "
                  f"{tgt_2019:.1f}pts @ {PRICE_2019})  —  {session_label}")
            print(f"  EV in pts  |  best hold shown per cell  |  ◄ = positive EV")
            print(f"{'─'*78}")

            # Header: move_bps columns
            move_hdr = "   ".join(f"move≥{mb}bp" for mb in MOVE_BPS_LIST)
            print(f"  {'vol':>4}   {move_hdr}")
            print(f"  {'─'*72}")

            for vm in VOL_MULTS:
                parts = []
                for mb in MOVE_BPS_LIST:
                    # Find best hold for this combo
                    best_ev, best_str = float("-inf"), f"{'—':>16}"
                    for hb in HOLD_BARS:
                        key = (vm, mb, hb, tb)
                        df_r = results[key]
                        if session_filter is not None:
                            df_r = df_r[df_r["is_rth"] == session_filter]
                        st = ev_stats(df_r)
                        if not math.isnan(st["ev"]) and st["ev"] > best_ev:
                            best_ev = st["ev"]
                            flag = "◄" if st["ev"] > 0 else " "
                            best_str = (f"{st['ev']:>+7.3f}p{flag}"
                                        f"({st['n']:4}n,{hb}b)")
                    parts.append(best_str)
                print(f"  {vm:>4}×   " + "   ".join(parts))

    # ── Outcome detail: best combos ───────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  OUTCOME DETAIL  —  ALL SESSIONS  (all holds shown)")
    print(f"  P(tgt) + P(stop) + P(time) = 1.0")
    print(f"{'─'*78}")
    for tb in TARGET_BPS_LIST:
        tgt_now = PRICE_NOW * tb / 10000
        print(f"\n  Target = {tb}bp  (~{tgt_now:.1f}pts today)  stop={STOP_PTS}pts")
        print(f"  {'vol':>4}  {'move':>6}  {'hold':>5}  {'n':>5}  "
              f"{'P(tgt)':>7}  {'P(stop)':>8}  {'P(time)':>8}  {'EV(pts)':>10}")
        print(f"  {'─'*66}")
        for vm in VOL_MULTS:
            for mb in MOVE_BPS_LIST:
                for hb in HOLD_BARS:
                    st = ev_stats(results[(vm, mb, hb, tb)])
                    if math.isnan(st["ev"]):
                        continue
                    p_time = 1.0 - st["p_tgt"] - st["p_stop"]
                    flag   = " ◄" if st["ev"] > 0 else "  "
                    print(f"  {vm:>4}×  {mb:>4}bp  {hb:>4}b  {st['n']:>5}  "
                          f"{st['p_tgt']:>7.3f}  {st['p_stop']:>8.3f}  "
                          f"{p_time:>8.3f}  {st['ev']:>+10.3f}{flag}")
            print()

    # ── RTH vs Globex for best combos ─────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  RTH vs GLOBEX  —  top combinations by EV (all sessions)")
    print(f"{'─'*78}")
    # Rank all combos
    ranked = []
    for vm in VOL_MULTS:
        for mb in MOVE_BPS_LIST:
            for hb in HOLD_BARS:
                for tb in TARGET_BPS_LIST:
                    st = ev_stats(results[(vm, mb, hb, tb)])
                    if not math.isnan(st["ev"]):
                        ranked.append((st["ev"], vm, mb, hb, tb))
    ranked.sort(reverse=True)

    print(f"  {'combo':<30}  {'ALL':>12}  {'RTH':>14}  {'GLOBEX':>14}")
    print(f"  {'─'*74}")
    def fmt_st(st):
        if math.isnan(st["ev"]):
            return f"{'—':>14}"
        flag = "◄" if st["ev"] > 0 else " "
        return f"{st['ev']:>+8.3f}pts{flag}({st['n']:4}n)"

    for ev_val, vm, mb, hb, tb in ranked[:12]:
        key   = (vm, mb, hb, tb)
        df_r  = results[key]
        label = f"vol≥{vm}× move≥{mb}bp hold={hb}b tgt={tb}bp"
        st_a  = ev_stats(df_r)
        st_r  = ev_stats(df_r[df_r["is_rth"]])
        st_g  = ev_stats(df_r[~df_r["is_rth"]])
        print(f"  {label:<30}  {fmt_st(st_a)}  {fmt_st(st_r)}  {fmt_st(st_g)}")

    # ── Year-by-year for top 3 ────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  YEAR-BY-YEAR  —  top 3 combos  (all sessions)")
    print(f"{'─'*78}")
    for ev_val, vm, mb, hb, tb in ranked[:3]:
        key   = (vm, mb, hb, tb)
        df_r  = results[key]
        years = sorted(df_r["year"].unique())
        st_all = ev_stats(df_r)
        tgt_now = PRICE_NOW * tb / 10000
        print(f"\n  vol≥{vm}×  move≥{mb}bp  hold={hb}b  tgt={tb}bp (~{tgt_now:.1f}pts)  "
              f"EV={st_all['ev']:+.3f}pts  (n={st_all['n']})")
        print(f"  {'Year':<6}  {'n':>4}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}")
        print(f"  {'─'*44}")
        for yr in years:
            sub = df_r[df_r["year"] == yr]
            st  = ev_stats(sub)
            if math.isnan(st["ev"]):
                print(f"  {yr}   {st['n']:>4}  {'—':>7}  {'—':>8}  {'—':>10}")
            else:
                flag = " ◄" if st["ev"] > 0 else "  "
                print(f"  {yr}   {st['n']:>4}  {st['p_tgt']:.3f}    "
                      f"{st['p_stop']:.3f}  {st['ev']:>+10.3f}{flag}")

    # ── Direction split ───────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  DIRECTION SPLIT  —  top 6 combos  (all sessions)")
    print(f"{'─'*78}")
    print(f"  {'combo':<28}  {'Dir':<6}  {'n':>5}  "
          f"{'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}")
    print(f"  {'─'*72}")
    for ev_val, vm, mb, hb, tb in ranked[:6]:
        key   = (vm, mb, hb, tb)
        df_r  = results[key]
        label = f"vol≥{vm}× move≥{mb}bp h={hb}b t={tb}bp"
        for dir_lbl, d in [("LONG ", 1), ("SHORT", -1)]:
            sub = df_r[df_r["direction"] == d]
            st  = ev_stats(sub)
            ev_s = f"{st['ev']:>+10.3f}" if not math.isnan(st["ev"]) else f"{'—':>10}"
            flag = " ◄" if not math.isnan(st.get("ev", float("nan"))) and st["ev"] > 0 else ""
            p_t  = f"{st['p_tgt']:.3f}" if not math.isnan(st.get("p_tgt", float("nan"))) else "  —  "
            p_s  = f"{st['p_stop']:.3f}" if not math.isnan(st.get("p_stop", float("nan"))) else "  —  "
            print(f"  {label:<28}  {dir_lbl}  {st['n']:>5}  "
                  f"{p_t:>7}  {p_s:>8}  {ev_s}{flag}")
        print()
