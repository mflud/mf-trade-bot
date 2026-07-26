"""
Backtest: SLR_Scalp SHORT side — WO2 move window (open[-1] → close).

Mirrors the LONG backtest exactly but flips direction:
  - Surge bar: vol ≥ 7× median, WO2 move ≤ −threshold (bearish)
  - Surge bar must close bearish (close < open)
  - Pullback: close[j] bounces UP from surge low by 1.5–4.5 bp within next 1–3 bars
  - Entry: open of bar after pullback bar (SHORT)
  - Target: entry − entry×15bp/10000
  - Stop:   entry + 3.0 pts

Also runs LONG at the same thresholds for direct comparison.

Usage:
  python src/backtest_slr_short.py
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

VOL_LOOKBACK     = 20
PULLBACK_MIN_BPS = 1.5
PULLBACK_MAX_BPS = 4.5
PULLBACK_BARS    = 3
STOP_PTS         = 3.0
TARGET_BPS       = 15.0
HOLD_MAX         = 15
VOL_MULT         = 7.0
MIN_TRADES       = 30

MOVE_BPS_LIST = [8, 10, 12, 15]


def load_bars() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")
    df = df.sort_values("ts").reset_index(drop=True)
    h  = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    df = df.reset_index(drop=True)
    ts_et     = df["ts"].dt.tz_convert(ET)
    hm_raw    = ts_et.dt.hour.values * 60 + ts_et.dt.minute.values
    df["is_rth"] = (hm_raw >= 570) & (hm_raw < 960)
    df["year"]   = df["ts"].dt.year.values
    df["gap"]    = (df["ts"].diff() > pd.Timedelta(minutes=2)).values
    df.loc[0, "gap"] = True
    return df


def find_candidates(df: pd.DataFrame, direction: int, move_bps: float) -> pd.DataFrame:
    """direction: +1 = LONG, -1 = SHORT."""
    c      = df["close"].values
    o      = df["open"].values
    hi     = df["high"].values
    lo     = df["low"].values
    vol    = df["volume"].values.astype(float)
    gap    = df["gap"].values.astype(bool)
    is_rth = df["is_rth"].values.astype(bool)
    year   = df["year"].values
    nb     = len(df)

    vol_s   = pd.Series(np.where(gap, np.nan, vol))
    med20   = vol_s.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().values
    gap_cum = np.cumsum(gap.astype(int))

    records = []

    for i in range(VOL_LOOKBACK + 2, nb - HOLD_MAX - PULLBACK_BARS - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        if vol[i] < VOL_MULT * med20[i]:
            continue
        if gap[i] or gap[i - 1]:
            continue

        move_threshold = c[i] * move_bps / 10000

        # WO2: open[i-1] → close[i], signed by direction
        move = (c[i] - o[i - 1]) * direction
        if move < move_threshold:
            continue

        # Surge bar body must align with direction
        if direction == 1 and c[i] < o[i]:
            continue
        if direction == -1 and c[i] > o[i]:
            continue

        # Extreme and pullback reference
        if direction == 1:
            extreme = hi[i]
            pb_ref  = o[i - 1]   # start of move (sanity floor)
        else:
            extreme = lo[i]
            pb_ref  = o[i - 1]   # start of move (sanity ceiling)

        pb_min = abs(extreme) * PULLBACK_MIN_BPS / 10000
        pb_max = abs(extreme) * PULLBACK_MAX_BPS / 10000

        pullback_bar = None
        for j in range(i + 1, min(i + 1 + PULLBACK_BARS, nb - HOLD_MAX - 2)):
            if gap[j]:
                break
            retrace = (c[j] - extreme) * direction * -1   # positive = retracing
            if pb_min <= retrace <= pb_max:
                pullback_bar = j
                break

        if pullback_bar is None:
            continue

        entry_bar = pullback_bar + 1
        if entry_bar + HOLD_MAX >= nb:
            continue
        if gap_cum[entry_bar + HOLD_MAX] > gap_cum[i]:
            continue

        # Sanity: pullback close hasn't reversed past the start of the move
        if direction == 1 and c[pullback_bar] <= o[i - 1]:
            continue
        if direction == -1 and c[pullback_bar] >= o[i - 1]:
            continue

        entry = o[entry_bar]

        if direction == 1:
            path_fav = np.array([hi[entry_bar + 1 + k] - entry for k in range(HOLD_MAX)])
            path_adv = np.array([entry - lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
            te_pts   = c[entry_bar + HOLD_MAX] - entry
        else:
            path_fav = np.array([entry - lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
            path_adv = np.array([hi[entry_bar + 1 + k] - entry for k in range(HOLD_MAX)])
            te_pts   = entry - c[entry_bar + HOLD_MAX]

        records.append({
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    bool(is_rth[entry_bar]),
            "year":      year[entry_bar],
            "path_fav":  path_fav,
            "path_adv":  path_adv,
            "te_pts":    te_pts,
        })

    return pd.DataFrame(records)


def simulate(cands: pd.DataFrame) -> pd.DataFrame:
    if cands.empty:
        return pd.DataFrame()
    hold_until = -1
    records    = []
    for _, row in cands.iterrows():
        eb = row["entry_bar"]
        if eb <= hold_until:
            continue
        entry      = row["entry"]
        target_pts = entry * TARGET_BPS / 10000
        path_fav   = row["path_fav"]
        path_adv   = row["path_adv"]
        hit_tgt = hit_stop = None
        for k in range(HOLD_MAX):
            if hit_stop is None and path_adv[k] >= STOP_PTS:
                hit_stop = k + 1
            if hit_tgt  is None and path_fav[k] >= target_pts:
                hit_tgt  = k + 1
        te = row["te_pts"] if (hit_tgt is None and hit_stop is None) else 0.0
        hold_until = eb + HOLD_MAX
        records.append({
            "year":      row["year"],
            "is_rth":    row["is_rth"],
            "hit_tgt":   hit_tgt,
            "hit_stop":  hit_stop,
            "te_pts":    te,
            "target_pts": target_pts,
        })
    return pd.DataFrame(records)


def ev_stats(df: pd.DataFrame, session_filter=None) -> dict:
    if session_filter is not None:
        df = df[df["is_rth"] == session_filter]
    n = len(df)
    if n < MIN_TRADES:
        return {"ev": float("nan"), "p_tgt": float("nan"), "p_stop": float("nan"), "n": n}
    tgt_bar  = df["hit_tgt"].fillna(999).values
    stop_bar = df["hit_stop"].fillna(999).values
    ht_first = df["hit_tgt"].notna().values & (tgt_bar <= stop_bar)
    hs_first = df["hit_stop"].notna().values & (stop_bar < tgt_bar)
    neither  = ~ht_first & ~hs_first
    te       = df["te_pts"].values
    tgt_pts  = df["target_pts"].values
    ev = (float((ht_first * tgt_pts).mean()) - float(hs_first.mean() * STOP_PTS)
          + (float(te[neither].mean()) * neither.mean() if neither.any() else 0.0))
    return {"ev": ev, "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": n}


if __name__ == "__main__":
    print(f"\n{'═'*78}")
    print(f"  SLR_Scalp — LONG vs SHORT  (WO2: open[-1]→close)")
    print(f"  vol≥{VOL_MULT}×  pullback {PULLBACK_MIN_BPS}–{PULLBACK_MAX_BPS}bp  "
          f"stop={STOP_PTS}pts  target={TARGET_BPS}bp  hold≤{HOLD_MAX}b")
    print(f"{'═'*78}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    results = {}
    for direction, label in [(1, "LONG"), (-1, "SHORT")]:
        for mb in MOVE_BPS_LIST:
            cands = find_candidates(df, direction, mb)
            results[(direction, mb)] = simulate(cands)
            print(f"  {label}  move≥{mb:2}bp  →  {len(cands):5,} candidates", flush=True)

    # ── Summary tables ────────────────────────────────────────────────────────
    for session_label, sess_filter in [("ALL SESSIONS", None),
                                        ("RTH ONLY",     True),
                                        ("GLOBEX ONLY",  False)]:
        print(f"\n{'─'*78}")
        print(f"  EV (pts/trade)  —  {session_label}")
        print(f"{'─'*78}")
        hdr = "   ".join(f"move≥{mb}bp" for mb in MOVE_BPS_LIST)
        print(f"  {'Dir':6}  {hdr}")
        print(f"  {'─'*70}")
        for direction, label in [(1, "LONG"), (-1, "SHORT")]:
            parts = []
            for mb in MOVE_BPS_LIST:
                st = ev_stats(results[(direction, mb)], sess_filter)
                if math.isnan(st["ev"]):
                    parts.append(f"{'—':>18}")
                else:
                    flag = "◄" if st["ev"] > 0 else " "
                    parts.append(f"{st['ev']:>+7.3f}p{flag}({st['n']:4}n)")
            print(f"  {label:6}  {'   '.join(parts)}")

    # ── Outcome detail ────────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  OUTCOME DETAIL — ALL SESSIONS  P(tgt) / P(stop) / EV")
    print(f"{'─'*78}")
    for direction, label in [(1, "LONG"), (-1, "SHORT")]:
        print(f"\n  {label}")
        print(f"  {'move':>6}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}")
        print(f"  {'─'*44}")
        for mb in MOVE_BPS_LIST:
            st = ev_stats(results[(direction, mb)])
            if math.isnan(st["ev"]):
                print(f"  {mb:>4}bp  {'<min':>5}")
                continue
            flag = " ◄" if st["ev"] > 0 else "  "
            print(f"  {mb:>4}bp  {st['n']:>5}  "
                  f"{st['p_tgt']:>7.3f}  {st['p_stop']:>8.3f}  "
                  f"{st['ev']:>+10.3f}{flag}")

    # ── Year-by-year at 12bp ──────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  YEAR-BY-YEAR — move≥12bp, all sessions")
    print(f"{'─'*78}")
    years = sorted(df["year"].unique())
    print(f"  {'Year':>5}  {'LONG':>22}  {'SHORT':>22}  ratio")
    print(f"  {'─'*60}")
    for yr in years:
        parts = []
        evs   = {}
        for direction, label in [(1, "LONG"), (-1, "SHORT")]:
            sub = results[(direction, 12)]
            yr_sub = sub[sub["year"] == yr] if not sub.empty else pd.DataFrame()
            st = ev_stats(yr_sub)
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>22}")
                evs[label] = None
            else:
                flag = "+" if st["ev"] > 0 else "-"
                parts.append(f"{flag}{abs(st['ev']):.3f}  ({st['n']:3}n)      ")
                evs[label] = st["ev"]
        ratio = ""
        if evs["LONG"] and evs["SHORT"] and evs["SHORT"] != 0:
            r = evs["LONG"] / evs["SHORT"]
            ratio = f"{r:.1f}×" if r > 0 else "neg"
        print(f"  {yr:>5}  {'  '.join(parts)}  {ratio}")
