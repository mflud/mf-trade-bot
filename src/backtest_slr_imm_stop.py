"""
Backtest: SLR_Scalp IMMEDIATE entry — stop size comparison.

No pullback. Entry = open[surge+1]. LONG only, vol≥7×, move≥12bp, hold≤15b.

Stop variants (all bp-scaled):
  BP 3.0bp, BP 3.5bp, BP 4.0bp, BP 4.5bp, BP 5.0bp, BP 6.0bp, BP 7.0bp

Usage:
  python src/backtest_slr_imm_stop.py
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

VOL_LOOKBACK = 20
TARGET_BPS   = 15.0
HOLD_MAX     = 15
VOL_MULT     = 7.0
MOVE_BPS     = 12.0
MIN_TRADES   = 30

STOP_BPS_LIST = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]


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
    ts_et    = df["ts"].dt.tz_convert(ET)
    hm_raw   = ts_et.dt.hour.values * 60 + ts_et.dt.minute.values
    df["is_rth"] = (hm_raw >= 570) & (hm_raw < 960)
    df["year"]   = df["ts"].dt.year.values
    df["gap"]    = (df["ts"].diff() > pd.Timedelta(minutes=2)).values
    df.loc[0, "gap"] = True
    return df


def find_candidates(df: pd.DataFrame) -> pd.DataFrame:
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
    for i in range(VOL_LOOKBACK + 2, nb - HOLD_MAX - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        if vol[i] < VOL_MULT * med20[i]:
            continue
        if gap[i] or gap[i - 1]:
            continue
        if c[i] < o[i]:
            continue
        move = c[i] - o[i - 1]
        if move < c[i] * MOVE_BPS / 10000:
            continue

        entry_bar = i + 1
        if entry_bar + HOLD_MAX >= nb:
            continue
        if gap[entry_bar]:
            continue
        if gap_cum[entry_bar + HOLD_MAX] > gap_cum[i]:
            continue

        entry     = o[entry_bar]
        path_hi   = np.array([hi[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        path_lo   = np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        vol_ratio = vol[i] / med20[i]

        records.append({
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    bool(is_rth[entry_bar]),
            "year":      year[entry_bar],
            "path_hi":   path_hi,
            "path_lo":   path_lo,
            "vol_ratio": vol_ratio,
        })
    return pd.DataFrame(records)


def simulate(cands: pd.DataFrame, stop_bps: float) -> pd.DataFrame:
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
        sl_pts     = entry * stop_bps   / 10000
        path_hi    = row["path_hi"]
        path_lo    = row["path_lo"]

        hit_tgt = hit_stop = None
        for k in range(HOLD_MAX):
            if hit_stop is None and (entry - path_lo[k]) >= sl_pts:
                hit_stop = k + 1
            if hit_tgt  is None and (path_hi[k] - entry) >= target_pts:
                hit_tgt  = k + 1

        te = 0.0
        if hit_tgt is None and hit_stop is None:
            te = path_hi[-1] - entry
        hold_until = eb + HOLD_MAX

        records.append({
            "year":       row["year"],
            "is_rth":     row["is_rth"],
            "hit_tgt":    hit_tgt,
            "hit_stop":   hit_stop,
            "te_pts":     te,
            "target_pts": target_pts,
            "sl_pts":     sl_pts,
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
    ht_first = df["hit_tgt"].notna().values  & (tgt_bar  <= stop_bar)
    hs_first = df["hit_stop"].notna().values & (stop_bar <  tgt_bar)
    neither  = ~ht_first & ~hs_first
    te       = df["te_pts"].values
    tgt_pts  = df["target_pts"].values
    sl_pts   = df["sl_pts"].values
    ev = (float((ht_first * tgt_pts).mean())
          - float((hs_first * sl_pts).mean())
          + (float(te[neither].mean()) * neither.mean() if neither.any() else 0.0))
    return {"ev": ev, "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": n}


if __name__ == "__main__":
    print(f"\n{'═'*78}")
    print(f"  SLR_Scalp IMMEDIATE entry — stop size comparison")
    print(f"  vol≥{VOL_MULT}×  move≥{MOVE_BPS}bp  target={TARGET_BPS}bp  hold≤{HOLD_MAX}b  LONG only")
    print(f"{'═'*78}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    cands = find_candidates(df)
    print(f"  {len(cands):,} candidates")

    results = {sb: simulate(cands, sb) for sb in STOP_BPS_LIST}

    for session_label, sess_filter in [("ALL SESSIONS", None),
                                        ("RTH ONLY",     True),
                                        ("GLOBEX ONLY",  False)]:
        print(f"\n{'─'*78}")
        print(f"  {session_label}")
        print(f"  {'Stop':10}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}")
        print(f"  {'─'*50}")
        for sb in STOP_BPS_LIST:
            st = ev_stats(results[sb], sess_filter)
            if math.isnan(st["ev"]):
                print(f"  {sb:.1f}bp      {'<min':>5}")
                continue
            flag = " ◄" if st["ev"] == max(
                ev_stats(results[s], sess_filter)["ev"] for s in STOP_BPS_LIST
                if not math.isnan(ev_stats(results[s], sess_filter)["ev"])
            ) else "  "
            print(f"  {sb:.1f}bp      {st['n']:>5}  {st['p_tgt']:>7.3f}  "
                  f"{st['p_stop']:>8.3f}  {st['ev']:>+10.3f}{flag}")

    print(f"\n{'─'*78}")
    print(f"  YEAR-BY-YEAR — all sessions")
    print(f"{'─'*78}")
    years  = sorted(df["year"].unique())
    labels = [f"{sb:.1f}bp" for sb in STOP_BPS_LIST]
    print(f"  {'Year':>5}  " + "  ".join(f"{l:>14}" for l in labels))
    print(f"  {'─'*74}")
    for yr in years:
        parts = []
        for sb in STOP_BPS_LIST:
            sub = results[sb]
            yr_sub = sub[sub["year"] == yr] if not sub.empty else pd.DataFrame()
            st = ev_stats(yr_sub)
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>14}")
            else:
                s = f"{st['ev']:+.3f}({st['n']:3}n)"
                parts.append(f"{s:>14}")
        print(f"  {yr:>5}  " + "  ".join(parts))
