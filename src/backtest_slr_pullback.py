"""
Backtest: SLR_Scalp — pullback-entry vs immediate-entry (no pullback).

SURGE conditions (identical for both):
  vol ≥ 7× 20-bar median, WO2 move ≥ 12bp (open[-1]→close), bullish bar.

Entry variants:
  PULLBACK  — current bot: close[j] retraces 1.5–4.5bp from surge high
              within next 1–3 bars; entry = open[j+1]
  IMMEDIATE — enter on open[i+1] immediately after surge bar (no pullback)

Stop = 4.5bp, target = 15bp, hold ≤ 15b.  LONG only.

Usage:
  python src/backtest_slr_pullback.py
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
STOP_BPS         = 4.5
TARGET_BPS       = 15.0
HOLD_MAX         = 15
VOL_MULT         = 7.0
MOVE_BPS         = 12.0
MIN_TRADES       = 30


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


def find_surges(df: pd.DataFrame) -> pd.DataFrame:
    """All valid WO2 surge bars (before any entry logic)."""
    c      = df["close"].values
    o      = df["open"].values
    hi     = df["high"].values
    lo     = df["low"].values
    vol    = df["volume"].values.astype(float)
    gap    = df["gap"].values.astype(bool)
    is_rth = df["is_rth"].values.astype(bool)
    year   = df["year"].values
    nb     = len(df)

    vol_s  = pd.Series(np.where(gap, np.nan, vol))
    med20  = vol_s.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).median().values

    records = []
    for i in range(VOL_LOOKBACK + 2, nb - HOLD_MAX - PULLBACK_BARS - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        if vol[i] < VOL_MULT * med20[i]:
            continue
        if gap[i] or gap[i - 1]:
            continue
        if c[i] < o[i]:                          # must close bullish
            continue
        move = c[i] - o[i - 1]
        if move < c[i] * MOVE_BPS / 10000:
            continue
        records.append({
            "i":      i,
            "is_rth": bool(is_rth[i]),
            "year":   year[i],
        })
    return pd.DataFrame(records)


def find_pullback_entries(df: pd.DataFrame, surges: pd.DataFrame) -> pd.DataFrame:
    """PULLBACK variant: entry = open[pullback+1], pullback 1.5–4.5bp."""
    c   = df["close"].values
    hi  = df["high"].values
    lo  = df["low"].values
    o   = df["open"].values
    gap = df["gap"].values.astype(bool)
    nb  = len(df)
    gap_cum = np.cumsum(gap.astype(int))

    records = []
    for _, row in surges.iterrows():
        i = int(row["i"])
        extreme = hi[i]
        pb_min  = extreme * PULLBACK_MIN_BPS / 10000
        pb_max  = extreme * PULLBACK_MAX_BPS / 10000

        pullback_bar = None
        for j in range(i + 1, min(i + 1 + PULLBACK_BARS, nb - HOLD_MAX - 2)):
            if gap[j]:
                break
            retrace = extreme - c[j]
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
        if c[pullback_bar] <= o[i - 1]:          # sanity: hasn't reversed below move start
            continue

        entry     = o[entry_bar]
        path_hi   = np.array([hi[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        path_lo   = np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        records.append({
            "surge_i":   i,
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    row["is_rth"],
            "year":      row["year"],
            "path_hi":   path_hi,
            "path_lo":   path_lo,
        })
    return pd.DataFrame(records)


def find_immediate_entries(df: pd.DataFrame, surges: pd.DataFrame) -> pd.DataFrame:
    """IMMEDIATE variant: entry = open[i+1], no pullback required."""
    hi  = df["high"].values
    lo  = df["low"].values
    o   = df["open"].values
    gap = df["gap"].values.astype(bool)
    nb  = len(df)
    gap_cum = np.cumsum(gap.astype(int))

    records = []
    for _, row in surges.iterrows():
        i         = int(row["i"])
        entry_bar = i + 1
        if entry_bar + HOLD_MAX >= nb:
            continue
        if gap[entry_bar]:
            continue
        if gap_cum[entry_bar + HOLD_MAX] > gap_cum[i]:
            continue

        entry   = o[entry_bar]
        path_hi = np.array([hi[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        path_lo = np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        records.append({
            "surge_i":   i,
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    row["is_rth"],
            "year":      row["year"],
            "path_hi":   path_hi,
            "path_lo":   path_lo,
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
        sl_pts     = entry * STOP_BPS   / 10000
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
            te = path_hi[-1] - entry   # use last bar high as proxy (conservative)
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
    print(f"  SLR_Scalp — Pullback-entry vs Immediate-entry")
    print(f"  vol≥{VOL_MULT}×  move≥{MOVE_BPS}bp  stop={STOP_BPS}bp  target={TARGET_BPS}bp  hold≤{HOLD_MAX}b  LONG only")
    print(f"{'═'*78}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    surges = find_surges(df)
    print(f"  {len(surges):,} surge bars found")

    pb_cands  = find_pullback_entries(df, surges)
    imm_cands = find_immediate_entries(df, surges)
    print(f"  PULLBACK:  {len(pb_cands):,} candidates")
    print(f"  IMMEDIATE: {len(imm_cands):,} candidates")

    pb_res  = simulate(pb_cands)
    imm_res = simulate(imm_cands)

    VARIANTS = [("PULLBACK",  pb_res), ("IMMEDIATE", imm_res)]

    for session_label, sess_filter in [("ALL SESSIONS", None),
                                        ("RTH ONLY",     True),
                                        ("GLOBEX ONLY",  False)]:
        print(f"\n{'─'*78}")
        print(f"  {session_label}")
        print(f"  {'Variant':12}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}")
        print(f"  {'─'*50}")
        for label, res in VARIANTS:
            st = ev_stats(res, sess_filter)
            if math.isnan(st["ev"]):
                print(f"  {label:12}  {'<min':>5}")
                continue
            flag = " ◄" if st["ev"] > 0 else "  "
            print(f"  {label:12}  {st['n']:>5}  {st['p_tgt']:>7.3f}  "
                  f"{st['p_stop']:>8.3f}  {st['ev']:>+10.3f}{flag}")

    print(f"\n{'─'*78}")
    print(f"  YEAR-BY-YEAR — all sessions")
    print(f"{'─'*78}")
    years = sorted(df["year"].unique())
    print(f"  {'Year':>5}  {'PULLBACK':>18}  {'IMMEDIATE':>18}  diff")
    print(f"  {'─'*60}")
    for yr in years:
        parts = []
        evs   = {}
        for label, res in VARIANTS:
            sub = res[res["year"] == yr] if not res.empty else pd.DataFrame()
            st  = ev_stats(sub)
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>18}")
                evs[label] = None
            else:
                s = f"{st['ev']:+.3f}({st['n']:3}n)"
                parts.append(f"{s:>18}")
                evs[label] = st["ev"]
        diff = ""
        pb_ev  = evs.get("PULLBACK")
        imm_ev = evs.get("IMMEDIATE")
        if pb_ev is not None and imm_ev is not None:
            d = pb_ev - imm_ev
            diff = f"{d:+.3f} {'PB better' if d > 0 else 'IMM better'}"
        print(f"  {yr:>5}  {'  '.join(parts)}  {diff}")
