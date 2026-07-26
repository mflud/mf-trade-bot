"""
Backtest: SLR_Scalp — support-level stop vs fixed/bp stops.

Same WO2 setup as backtest_slr_stop.py (surge → pullback → entry, LONG only),
but adds support-anchored stop variants:

  SUPP_PB_1T   : SL just below the pullback bar low  (low - 0.25 pt, 1 tick)
  SUPP_PB_2T   : SL just below the pullback bar low  (low - 0.50 pt, 2 ticks)
  SUPP_SURGE_1T: SL just below the surge bar low      (low - 0.25 pt)

These are compared against the best fixed/bp stops from the prior backtest.

Rationale: price tends to bounce off support levels until they break, so
placing the SL just below the most recent structural low should reduce
whipsaw stops while still protecting against genuine breakdowns.

Usage:
  python src/backtest_slr_support_sl.py
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
TICK_SIZE        = 0.25    # MES minimum move
STOP_MIN_BPS     = 10.0   # floor used by the live bot (stops < 10bp trigger too often)
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


def find_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Find all valid WO2 surge→pullback entries. Store structural lows."""
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

        path_hi = np.array([hi[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        path_lo = np.array([lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        te_pts  = c[entry_bar + HOLD_MAX] - entry

        # Structural lows for support-anchored stops
        surge_lo   = lo[i]                # low of the volume surge bar
        pullback_lo = lo[pullback_bar]    # low of the pullback bar (nearest support)

        records.append({
            "entry_bar":   entry_bar,
            "entry":       entry,
            "is_rth":      bool(is_rth[entry_bar]),
            "year":        year[entry_bar],
            "path_hi":     path_hi,
            "path_lo":     path_lo,
            "te_pts":      te_pts,
            "surge_lo":    surge_lo,
            "pullback_lo": pullback_lo,
        })

    return pd.DataFrame(records)


# ── Stop variants ─────────────────────────────────────────────────────────────
#
# Each variant is a dict with a "kind" key:
#   fixed  → sl_pts = pts
#   bps    → sl_pts = entry × bps / 10000
#   supp   → sl_price = structural_low - buffer; sl_pts = entry - sl_price
#             structural_low is "pullback_lo" or "surge_lo"

STOPS = [
    # Baseline: 10bp floor (live bot minimum)
    {"label": "BP 10bp",      "kind": "bps",   "bps": 10.0},
    # Support-anchored, floored at 10bp
    {"label": "SUPP_PB  1T",  "kind": "supp",  "src": "pullback_lo", "buf": TICK_SIZE},
    {"label": "SUPP_PB  2T",  "kind": "supp",  "src": "pullback_lo", "buf": 2 * TICK_SIZE},
    {"label": "SUPP_SRG 1T",  "kind": "supp",  "src": "surge_lo",    "buf": TICK_SIZE},
    {"label": "SUPP_SRG 2T",  "kind": "supp",  "src": "surge_lo",    "buf": 2 * TICK_SIZE},
]


def _sl_pts(row, variant: dict) -> float:
    entry   = row["entry"]
    kind    = variant["kind"]
    min_pts = entry * STOP_MIN_BPS / 10000   # 10bp floor (live bot minimum)
    if kind == "fixed":
        return max(variant["pts"], min_pts)
    if kind == "bps":
        return max(entry * variant["bps"] / 10000, min_pts)
    # support-anchored: place SL just below structural low, but never tighter than 10bp
    sl_price = row[variant["src"]] - variant["buf"]
    sl_pts   = entry - sl_price
    return max(sl_pts, min_pts)


def simulate(cands: pd.DataFrame, variant: dict) -> pd.DataFrame:
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
        sl_pts     = _sl_pts(row, variant)
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
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "avg_sl": float("nan"), "n": n}
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
            "p_stop": float(hs_first.mean()),
            "avg_sl": float(sl_pts.mean()), "n": n}


if __name__ == "__main__":
    print(f"\n{'═'*86}")
    print(f"  SLR_Scalp — Support-level SL vs fixed/bp stops")
    print(f"  WO2  vol≥{VOL_MULT}×  move≥{MOVE_BPS}bp  pullback {PULLBACK_MIN_BPS}–{PULLBACK_MAX_BPS}bp  "
          f"target={TARGET_BPS}bp  hold≤{HOLD_MAX}b  LONG only")
    print(f"  SUPP_PB   = SL just below pullback bar low (nearest support)")
    print(f"  SUPP_SRG  = SL just below surge bar low    (breakout level)")
    print(f"{'═'*86}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    print(f"\nFinding candidates …", flush=True)
    cands = find_candidates(df)
    print(f"  {len(cands):,} candidates")

    # Show distribution of support-based SL sizes for context
    if not cands.empty:
        pb_sl  = cands["entry"] - cands["pullback_lo"] + TICK_SIZE
        srg_sl = cands["entry"] - cands["surge_lo"]    + TICK_SIZE
        print(f"\n  Support SL size distribution (pts):")
        print(f"  {'':20}  {'p25':>6}  {'med':>6}  {'p75':>6}  {'mean':>6}  {'max':>6}")
        for label2, series in [("PB low - 1T", pb_sl), ("Surge low - 1T", srg_sl)]:
            print(f"  {label2:20}  "
                  f"{series.quantile(.25):>6.2f}  "
                  f"{series.median():>6.2f}  "
                  f"{series.quantile(.75):>6.2f}  "
                  f"{series.mean():>6.2f}  "
                  f"{series.max():>6.2f}")

    results = {v["label"]: simulate(cands, v) for v in STOPS}

    hdr = (f"  {'Stop':14}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  "
           f"{'EV(pts)':>10}  {'AvgSL':>6}")
    sep = f"  {'─'*72}"

    for session_label, sess_filter in [("ALL SESSIONS", None),
                                        ("RTH ONLY",     True),
                                        ("GLOBEX ONLY",  False)]:
        print(f"\n{'─'*86}")
        print(f"  {session_label}")
        print(hdr)
        print(sep)
        for v in STOPS:
            lbl = v["label"]
            st  = ev_stats(results[lbl], sess_filter)
            if math.isnan(st["ev"]):
                print(f"  {lbl:14}  {'<min':>5}")
                continue
            flag = " ◄" if st["ev"] > 0 else "  "
            print(f"  {lbl:14}  {st['n']:>5}  {st['p_tgt']:>7.3f}  "
                  f"{st['p_stop']:>8.3f}  {st['ev']:>+10.3f}{flag}  "
                  f"{st['avg_sl']:>6.2f}")

    # ── Year-by-year EV ───────────────────────────────────────────────────────
    print(f"\n{'─'*86}")
    print(f"  YEAR-BY-YEAR EV (pts) — RTH only")
    print(f"{'─'*86}")
    years  = sorted(df["year"].unique())
    labels = [v["label"] for v in STOPS]
    print(f"  {'Year':>5}  " + "  ".join(f"{l:>16}" for l in labels))
    print(f"  {'─'*80}")
    for yr in years:
        parts = []
        for lbl in labels:
            sub    = results[lbl]
            yr_sub = sub[(sub["year"] == yr) & sub["is_rth"]] if not sub.empty else pd.DataFrame()
            st     = ev_stats(yr_sub)
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>16}")
            else:
                s = f"{st['ev']:+.3f}({st['n']:3}n)"
                parts.append(f"{s:>16}")
        print(f"  {yr:>5}  " + "  ".join(parts))
