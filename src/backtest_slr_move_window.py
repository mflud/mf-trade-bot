"""
Backtest: SLR_Scalp — move-window sweep.

Compare four definitions of the "surge move":
  W1 : close - open   of the surge bar alone          (current bot)
  W2 : close[i] - close[i-1]                          (2-bar close-to-close)
  W3 : close[i] - close[i-2]                          (3-bar close-to-close)
  W1h: hi[i] - min(lo[i-1], lo[i])                    (original backtest — 2-bar H/L)

Volume surge condition is identical for all: vol[i] >= mult × 20-bar rolling median.
Pullback, entry, and exit rules are also identical:
  - Pullback 1.5–4.5 bp from surge high within next 1–3 bars
  - Entry: open of bar after pullback bar
  - Target: 15 bp (price-scaled); Stop: 3 pts fixed; Max hold: 15 bars

LONG only.

Move thresholds swept per window:
  W1 / W2 / W3  : 5, 8, 10, 12, 15 bp
  W1h           : 5, 8, 10, 12, 15 bp (same grid for comparability)

Usage:
  python src/backtest_slr_move_window.py
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
VOL_MULT         = 7.0   # fixed at backtest-recommended value
MIN_TRADES       = 30

MOVE_BPS_LIST = [5, 8, 10, 12, 15]

WINDOWS = ["W1", "W2", "W3", "WO2", "WO3", "W1h"]
WINDOW_LABELS = {
    "W1":  "1-bar  O→C       (bot today)",
    "W2":  "2-bar  C[-1]→C",
    "W3":  "3-bar  C[-2]→C",
    "WO2": "2-bar  O[-1]→C   (your idea)",
    "WO3": "3-bar  O[-2]→C   (your idea)",
    "W1h": "2-bar  H/L       (orig backtest)",
}


# ── Data ─────────────────────────────────────────────────────────────────────

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


# ── Candidate finder ──────────────────────────────────────────────────────────

def find_candidates(df: pd.DataFrame, window: str, move_bps: float) -> pd.DataFrame:
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
    gap_cum = np.cumsum(gap.astype(int))

    records = []
    start   = max(VOL_LOOKBACK + 3, 3)   # need up to 3 prior bars for W3

    for i in range(start, nb - HOLD_MAX - PULLBACK_BARS - 2):
        if np.isnan(med20[i]) or med20[i] == 0:
            continue
        if vol[i] < VOL_MULT * med20[i]:
            continue

        # Skip if any bar in the window spans a gap
        if gap[i] or (window in ("W2", "W3", "WO2", "WO3", "W1h") and gap[i - 1]):
            continue
        if window in ("W3", "WO3") and gap[i - 2]:
            continue

        move_threshold = c[i] * move_bps / 10000

        # Measure move according to window definition
        if window == "W1":
            move = c[i] - o[i]
        elif window == "W2":
            move = c[i] - c[i - 1]
        elif window == "W3":
            move = c[i] - c[i - 2]
        elif window == "WO2":
            move = c[i] - o[i - 1]   # open of bar 1 ago → current close
        elif window == "WO3":
            move = c[i] - o[i - 2]   # open of bar 2 ago → current close
        elif window == "W1h":
            move = hi[i] - min(lo[i - 1], lo[i])
        else:
            raise ValueError(window)

        # LONG only: move must be bullish and close > open on surge bar
        if move < move_threshold:
            continue
        if c[i] < o[i]:          # surge bar must close bullish
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
        # Pullback close must remain above surge open (move not fully reversed)
        if c[pullback_bar] <= o[i]:
            continue

        entry = o[entry_bar]

        path_fav = np.array([hi[entry_bar + 1 + k] - entry for k in range(HOLD_MAX)])
        path_adv = np.array([entry - lo[entry_bar + 1 + k] for k in range(HOLD_MAX)])
        te_pts   = c[entry_bar + HOLD_MAX] - entry

        records.append({
            "entry_bar": entry_bar,
            "entry":     entry,
            "is_rth":    bool(is_rth[entry_bar]),
            "year":      year[entry_bar],
            "path_fav":  path_fav,
            "path_adv":  path_adv,
            "te_pts":    te_pts,
            "move_bps_actual": move / c[i] * 10000,
        })

    return pd.DataFrame(records)


# ── Simulation ────────────────────────────────────────────────────────────────

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
    ev = float((ht_first * tgt_pts).mean()) - float(hs_first.mean() * STOP_PTS) \
         + (float(te[neither].mean()) * neither.mean() if neither.any() else 0.0)
    return {"ev": ev, "p_tgt": float(ht_first.mean()),
            "p_stop": float(hs_first.mean()), "n": n}


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*80}")
    print(f"  SLR_Scalp — Move Window Sweep")
    print(f"  vol≥{VOL_MULT}×  pullback {PULLBACK_MIN_BPS}–{PULLBACK_MAX_BPS}bp  "
          f"stop={STOP_PTS}pts  target={TARGET_BPS}bp  hold≤{HOLD_MAX}b  LONG only")
    print(f"{'═'*80}")

    print(f"\nLoading {CSV_PATH} …", flush=True)
    df = load_bars()
    print(f"  {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    # Build results table
    results = {}   # (window, move_bps) → simulated DataFrame
    for w in WINDOWS:
        for mb in MOVE_BPS_LIST:
            cands = find_candidates(df, w, mb)
            results[(w, mb)] = simulate(cands)
            print(f"  {w:4s}  move≥{mb:2}bp  →  {len(cands):5,} candidates", flush=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    for session_label, sess_filter in [("ALL SESSIONS", None),
                                        ("RTH ONLY",     True),
                                        ("GLOBEX ONLY",  False)]:
        print(f"\n{'─'*80}")
        print(f"  EV (pts/trade)  —  {session_label}"
              f"  target={TARGET_BPS}bp  stop={STOP_PTS}pts  hold≤{HOLD_MAX}b")
        print(f"  ◄ = positive EV    n = trades after hold_until dedup")
        print(f"{'─'*80}")

        hdr = "  ".join(f"move≥{mb:2}bp" for mb in MOVE_BPS_LIST)
        print(f"  {'Window':22s}  {hdr}")
        print(f"  {'─'*76}")

        for w in WINDOWS:
            parts = []
            for mb in MOVE_BPS_LIST:
                st = ev_stats(results[(w, mb)], sess_filter)
                if math.isnan(st["ev"]):
                    parts.append(f"{'—':>16}")
                else:
                    flag = "◄" if st["ev"] > 0 else " "
                    parts.append(f"{st['ev']:>+7.3f}p{flag}({st['n']:4}n)")
            print(f"  {WINDOW_LABELS[w]:22s}  {'  '.join(parts)}")

    # ── Outcome detail for best candidates ────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"  OUTCOME DETAIL — ALL SESSIONS  (P_tgt / P_stop / EV)")
    print(f"{'─'*80}")
    for w in WINDOWS:
        print(f"\n  {WINDOW_LABELS[w]}")
        print(f"  {'move':>6}  {'n':>5}  {'P(tgt)':>7}  {'P(stop)':>8}  {'EV(pts)':>10}")
        print(f"  {'─'*44}")
        for mb in MOVE_BPS_LIST:
            st = ev_stats(results[(w, mb)])
            if math.isnan(st["ev"]):
                print(f"  {mb:>4}bp  {'<min':>5}")
                continue
            p_time = 1.0 - st["p_tgt"] - st["p_stop"]
            flag   = " ◄" if st["ev"] > 0 else "  "
            print(f"  {mb:>4}bp  {st['n']:>5}  "
                  f"{st['p_tgt']:>7.3f}  {st['p_stop']:>8.3f}  "
                  f"{st['ev']:>+10.3f}{flag}")

    # ── Year-by-year for W1 vs W2 vs W3 at move=12bp ─────────────────────────
    print(f"\n{'─'*80}")
    print(f"  YEAR-BY-YEAR  —  move≥12bp, all sessions")
    print(f"{'─'*80}")
    years = sorted(df["year"].unique())
    print(f"  {'Year':>5}  " + "  ".join(f"{WINDOW_LABELS[w]:>22}" for w in WINDOWS))
    print(f"  {'─'*76}")
    for yr in years:
        parts = []
        for w in WINDOWS:
            sub = results[(w, 12)]
            yr_sub = sub[sub["year"] == yr] if not sub.empty else pd.DataFrame()
            st = ev_stats(yr_sub)
            if math.isnan(st["ev"]):
                parts.append(f"{'—':>22}")
            else:
                flag = "+" if st["ev"] > 0 else "-"
                parts.append(f"{flag}{abs(st['ev']):.3f}  ({st['n']:3}n)            ")
        print(f"  {yr:>5}  " + "  ".join(parts))
