"""
Time-of-day EV breakdown for the 3σ continuation signal (offset-0 5-min bars).

Breaks the trading session into time buckets (ET local time) and shows
P(tgt), P(stop), EV and n for each bucket — with and without the CSR filter.

Usage:
  python src/backtest_tod.py --sym MES
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config (keep in sync with backtest_offset.py) ──────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

TF             = 5
TRAILING_BARS  = 20
MOM_BARS       = 8
MAX_BARS_HOLD  = 3
MIN_SCALED     = 3.0
MAX_SCALED     = 99.0
MIN_VOL_RATIO  = 1.5
CSR_THRESHOLD  = 1.5

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}

# Time-of-day buckets in ET: (label, start_h, start_m, end_h, end_m)
TOD_BUCKETS = [
    ("08:00–09:00",  8,  0,  9,  0),
    ("09:00–09:30",  9,  0,  9, 30),
    ("09:30–10:00",  9, 30, 10,  0),
    ("10:00–11:00", 10,  0, 11,  0),
    ("11:00–12:00", 11,  0, 12,  0),
    ("12:00–13:00", 12,  0, 13,  0),
    ("13:00–14:00", 13,  0, 14,  0),
    ("14:00–15:00", 14,  0, 15,  0),
    ("15:00–16:00", 15,  0, 16,  0),
    ("16:00–17:00", 16,  0, 17,  0),
    ("17:00+",      17,  0, 24,  0),
]


# ── Data helpers ────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_offset_bars(df1: pd.DataFrame, offset: int) -> pd.DataFrame:
    records = []
    i = 0
    n = len(df1)
    while i < n:
        m = df1["ts"].iloc[i].minute
        if (m - offset) % TF == 0:
            break
        i += 1
    while i + TF <= n:
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            while i < n:
                m = df1["ts"].iloc[i].minute
                if (m - offset) % TF == 0:
                    break
                i += 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[-1],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += TF
    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Scan (records tod_bucket in ET, no blackout applied so all hours visible) ──

def scan(bars: pd.DataFrame) -> pd.DataFrame:
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)
    records = []

    for i in range(max(TRAILING_BARS, MOM_BARS), n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                          / closes[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or abs(scaled) > MAX_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        bar_et  = ts_pd[i].astimezone(ET)
        bar_hm  = (bar_et.hour, bar_et.minute)

        # Assign to TOD bucket
        bucket_label = "other"
        for label, sh, sm, eh, em in TOD_BUCKETS:
            if (sh, sm) <= bar_hm < (eh, em):
                bucket_label = label
                break

        direction = 1 if scaled > 0 else -1
        entry     = closes[i]

        if i >= MOM_BARS and not gaps[i - MOM_BARS: i].any():
            mom_rets = np.log(closes[i - MOM_BARS + 1: i]
                            / closes[i - MOM_BARS:     i - 1])
            csr = float(mom_rets.sum()) / sigma * direction
        else:
            csr = float("nan")

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if direction == 1 and h >= tgt_prices[t]: hit_tgt[t] = j - i
                    elif direction == -1 and l <= tgt_prices[t]: hit_tgt[t] = j - i
            for s in STOPS:
                if hit_stop[s] is None:
                    if direction == 1 and l <= stop_prices[s]: hit_stop[s] = j - i
                    elif direction == -1 and h >= stop_prices[s]: hit_stop[s] = j - i

        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
            "bucket":        bucket_label,
            "csr":           csr,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV helpers ──────────────────────────────────────────────────────────────────

def ev_stats(sub: pd.DataFrame, s: float, t: float) -> dict:
    if len(sub) < 5:
        return {"ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "n": len(sub)}
    ht = sub[f"hit_tgt_{t}"].notna().values
    hs = sub[f"hit_stop_{s}"].notna().values
    ht_first = ht & ~(hs & (sub[f"hit_stop_{s}"].fillna(999)
                            <= sub[f"hit_tgt_{t}"].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    time_ret = sub["time_exit_ret"].values
    p_tgt    = ht_first.mean()
    p_stop   = hs_first.mean()
    ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
    ev       = p_tgt * t - p_stop * s + neither.mean() * ev_nei
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "n": len(sub)}


def print_tod_table(res: pd.DataFrame, csr_filter: bool, title: str):
    print(f"\n  {title}")
    print(f"  {'─'*70}")
    print(f"  {'Time (ET)':>13}  {'n':>5}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}  {'Note'}")
    print(f"  {'─'*70}")
    total_n = 0
    for label, *_ in TOD_BUCKETS:
        sub = res[res["bucket"] == label]
        if csr_filter:
            sub = sub.dropna(subset=["csr"])
            sub = sub[sub["csr"] > CSR_THRESHOLD]
        p = ev_stats(sub, PRAC_S, PRAC_T)
        total_n += p["n"]
        if math.isnan(p["ev"]):
            print(f"  {label:>13}  {p['n']:>5}  {'—':>8}  {'—':>8}  {'—':>9}")
        else:
            flag = "◄ CURRENT BLACKOUT" if label == "08:00–09:00" else (
                   "◄" if p["ev"] > 0 else "")
            print(f"  {label:>13}  {p['n']:>5,}  {p['p_tgt']:>8.3f}  "
                  f"{p['p_stop']:>8.3f}  {p['ev']:>+9.4f}σ  {flag}")
    print(f"  {'─'*70}")
    all_sub = res if not csr_filter else res.dropna(subset=["csr"])
    if csr_filter:
        all_sub = all_sub[all_sub["csr"] > CSR_THRESHOLD]
    p_all = ev_stats(all_sub, PRAC_S, PRAC_T)
    print(f"  {'ALL':>13}  {p_all['n']:>5,}  {p_all['p_tgt']:>8.3f}  "
          f"{p_all['p_stop']:>8.3f}  {p_all['ev']:>+9.4f}σ")


# ── Main ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None, help="MES or MYM (default: both)")
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, cache in syms.items():
        print(f"\n{'═'*72}")
        print(f"  {sym}  —  Time-of-Day EV Breakdown (offset-0, :00/:05/:10 bars)")
        print(f"  σ={TRAILING_BARS}bars  CSR={MOM_BARS}bars  hold={MAX_BARS_HOLD}bars  "
              f"stop={PRAC_S}σ  target={PRAC_T}σ")
        print(f"{'═'*72}")
        print(f"  Note: 08:00–09:00 ET is the current blackout window.")
        print(f"        All hours shown so you can evaluate alternatives.")

        print(f"\n  Loading {cache} …")
        df1 = load_1min(cache)
        print(f"  {len(df1):,} 1-min bars  ({df1['ts'].min().date()} → {df1['ts'].max().date()})")

        bars = make_offset_bars(df1, 0)
        print(f"  {len(bars):,} 5-min bars — scanning …", end=" ", flush=True)
        res = scan(bars)
        print(f"{len(res):,} triggers (no blackout applied)")

        print_tod_table(res, csr_filter=False,
                        title=f"No CSR filter  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)")
        print_tod_table(res, csr_filter=True,
                        title=f"CSR≥{CSR_THRESHOLD:.1f} filter  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)")
