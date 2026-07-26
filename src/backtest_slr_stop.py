"""
Backtest: SLR_Scalp — fixed-pt stop vs bp-scaled stop.

WO2 move window (open[-1]→close), LONG only, vol≥7×, move≥12bp,
pullback 1.5–4.5bp, target=15bp, hold≤15b.

Stop variants:
  FIXED_3:  3.0 pts regardless of price         (current bot)
  FIXED_4:  4.0 pts regardless of price
  BP_4:     entry × 4.0bp / 10000               (≈ pullback max)
  BP_4_5:   entry × 4.5bp / 10000               (matches pullback max exactly)
  BP_5:     entry × 5.0bp / 10000
  BP_6:     entry × 6.0bp / 10000

Usage:
  python src/backtest_slr_stop.py
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
TARGET_BPS       = 15.0
HOLD_MAX         = 15
VOL_MULT         = 7.0
MOVE_BPS         = 12.0
MIN_TRADES       = 30

# Stop variants: (label, fixed_pts_or_None, bp_or_None)
STOPS = [
    ("FIXED 3pt",  3.0,  None),
    ("FIXED 4pt",  4.0,  None),
    ("BP 4.0bp",   None, 4.0),
    ("BP 4.5bp",   None, 4.5),
    ("BP 5.0bp",   None, 5.0),
    ("BP 6.0bp",   None, 6.0),
]


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
    """Find all valid WO2 surge→pullback entries. Stop applied at simulate time."""
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
        if c[i] < o[i]:
            continue

        move = c[i] - o[i - 1]
        if move < c[i] * MOVE_BPS / 10000:
            continue

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
        if c[pullback_bar] <= o[i - 1]:
            continue

        entry = o[entry_bar]

        # Store full path for flexible stop application at simulate time
        path_hi = np.array([hi[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        path_lo = np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        te_pts  = c[entry_bar + HOLD_MAX] - entry

        records.append({
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    bool(is_rth[entry_bar]),
            "year":      year[entry_bar],
            "path_hi":   path_hi,
            "path_lo":   path_lo,
            "te_pts":    te_pts,
        })

    return pd.DataFrame(records)


def simulate(cands: pd.DataFrame, stop_pts: float | None, stop_bps: float | None) -> pd.DataFrame:
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
        sl_pts     = stop_pts if stop_pts is not None else entry * stop_bps / 10000
        path_hi    = row["path_hi"]
        path_lo    = row["path_lo"]

        hit_tgt = hit_stop = None
        for k in range(HOLD_MAX):
            if hit_stop is None and (entry - path_lo[k]) >= sl_pts:
                hit_stop = k + 1
            if hit_tgt  is None and (path_hi[k] - entry) >= target_pts:
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
            "sl_pts":    sl_pts,
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
    sl_pts   = df["sl_pts"].values
    ev = (float((ht_first * tgt_pts).mean())
          - float((hs_first * sl_pts).mean())
          + (float(te[neither].mean()) * neither.mean() if neither.any() else 0.0))
    return {"ev": ev, "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": n}


if __name__ == "__main__":
    print(f"\n{'═'*78}")
    print(f"  SLR_Scalp — Fixed-pt stop vs bp-scaled stop")
    print(f"  WO2  vol≥{VOL_MULT}×  move≥{MOVE_BPS}bp  pullback {PULLBACK_MIN_BPS}–{PULLBACK_MAX_BPS}bp  "
          f"target={TARGET_BPS}bp  hold≤{HOLD_MAX}b  LONG only")
    print(f"{'═'*78}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    print(f"\nFinding candidates …", flush=True)
    cands = find_candidates(df)
    print(f"  {len(cands):,} candidates before dedup")

    results = {}
    for label, sp, sb in STOPS:
        results[label] = simulate(cands, sp, sb)

    # ── Summary table ─────────────────────────────────────────────────────────
    for session_label, sess_filter in [("ALL SESSIONS", None),
                                        ("RTH ONLY",     True),
                                        ("GLOBEX ONLY",  False)]:
        print(f"\n{'─'*78}")
        print(f"  {session_label}")
        print(f"  {'Stop':12}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}  note")
        print(f"  {'─'*65}")
        for label, sp, sb in STOPS:
            st = ev_stats(results[label], sess_filter)
            if math.isnan(st["ev"]):
                print(f"  {label:12}  {'<min':>5}")
                continue
            flag = " ◄" if st["ev"] > 0 else "  "
            # Approximate stop in bp at today's prices (~6640) and 2019 (~2900)
            if sp is not None:
                note = f"≈{sp/6640*10000:.1f}bp today / {sp/2900*10000:.1f}bp @2019"
            else:
                note = f"scales with price"
            print(f"  {label:12}  {st['n']:>5}  {st['p_tgt']:>7.3f}  "
                  f"{st['p_stop']:>8.3f}  {st['ev']:>+10.3f}{flag}  {note}")

    # ── Year-by-year ──────────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print(f"  YEAR-BY-YEAR — all sessions")
    print(f"{'─'*78}")
    years  = sorted(df["year"].unique())
    labels = [l for l, _, _ in STOPS]
    print(f"  {'Year':>5}  " + "  ".join(f"{l:>14}" for l in labels))
    print(f"  {'─'*74}")
    for yr in years:
        parts = []
        for label in labels:
            sub = results[label]
            yr_sub = sub[sub["year"] == yr] if not sub.empty else pd.DataFrame()
            st = ev_stats(yr_sub)
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>14}")
            else:
                s = f"{st['ev']:+.3f}({st['n']:3}n)"
                parts.append(f"{s:>14}")
        print(f"  {yr:>5}  " + "  ".join(parts))
