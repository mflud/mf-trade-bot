"""
Bar alignment test: does the 5-min signal edge depend on bars being aligned
to the standard clock grid (:00, :05, :10 ...) or just on the 5-min duration?

Tests offsets 0–4 minutes:
  offset 0: bars close at :00, :05, :10 ...  (standard)
  offset 1: bars close at :01, :06, :11 ...
  offset 2: bars close at :02, :07, :12 ...
  offset 3: bars close at :03, :08, :13 ...
  offset 4: bars close at :04, :09, :14 ...

Each 1-min bar is assigned to the 5-min bucket whose close time equals the
next minute ≡ offset (mod 5). Aggregation is gap-aware (session boundaries
are respected). Signal logic is identical to regime_analysis.py.

Usage:
  python src/backtest_offset.py --sym MES
"""

import argparse
import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

ET = ZoneInfo("America/New_York")

# ── Config ─────────────────────────────────────────────────────────────────────

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
BARS_PER_YEAR  = 252 * 23 * 60 / TF

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRAC_S, PRAC_T = 2.0, 3.0

BLACKOUT_ET = [(8, 0, 9, 0)]   # 08:00–09:00 ET (DST-aware)

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data helpers ───────────────────────────────────────────────────────────────

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
    """
    Aggregate 1-min bars into 5-min bars whose close minute ≡ offset (mod 5).
    e.g. offset=0: closes at :00, :05, :10 ...
         offset=3: closes at :03, :08, :13 ...

    Session gaps are respected — a chunk that spans a gap is discarded.
    """
    records = []
    i = 0
    n = len(df1)

    # Find the first bar that is the START of an offset-aligned bucket.
    # A bar at minute m starts a bucket if (m - offset) % TF == 0  (i.e. it's
    # the first minute of a new offset-5 period).
    while i < n:
        m = df1["ts"].iloc[i].minute
        if (m - offset) % TF == 0:
            break
        i += 1

    while i + TF <= n:
        chunk = df1.iloc[i: i + TF]
        # Discard if any gap within the chunk
        if chunk["gap"].iloc[1:].any():
            # Skip to just past the gap, then re-align
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            # Re-align to next offset-boundary
            while i < n:
                m = df1["ts"].iloc[i].minute
                if (m - offset) % TF == 0:
                    break
                i += 1
            continue

        records.append({
            "ts":     chunk["ts"].iloc[-1],   # use close time as bar timestamp
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


# ── Scan ───────────────────────────────────────────────────────────────────────

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

        bar_et = ts_pd[i].astimezone(ET)
        bar_hm = (bar_et.hour, bar_et.minute)
        if any((sh, sm) <= bar_hm < (eh, em) for sh, sm, eh, em in BLACKOUT_ET):
            continue

        direction = 1 if scaled > 0 else -1
        entry     = closes[i]
        sigma_pts = sigma * entry

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
            "year":          ts_pd[i].year,
            "csr":           csr,
            "sigma_pts":     sigma_pts,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV helpers ─────────────────────────────────────────────────────────────────

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


def best_ev(sub: pd.DataFrame) -> tuple[float, float, float]:
    best, bs, bt = -999.0, PRAC_S, PRAC_T
    for s in STOPS:
        for t in TARGETS:
            st = ev_stats(sub, s, t)
            if not math.isnan(st["ev"]) and st["ev"] > best:
                best, bs, bt = st["ev"], s, t
    return best, bs, bt


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None, help="MES or MYM (default: both)")
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, cache in syms.items():
        print(f"\n{'═'*72}")
        print(f"  {sym}  —  5-min Bar Alignment Test (offsets 0–4)")
        print(f"  σ={TRAILING_BARS}bars  CSR={MOM_BARS}bars  hold={MAX_BARS_HOLD}bars")
        print(f"{'═'*72}")

        print(f"\nLoading {cache} …")
        df1 = load_1min(cache)
        print(f"  {len(df1):,} 1-min bars  ({df1['ts'].min().date()} → {df1['ts'].max().date()})")

        results = {}
        for offset in range(TF):
            bars = make_offset_bars(df1, offset)
            print(f"  offset {offset}: {len(bars):,} bars — scanning …", end=" ", flush=True)
            res = scan(bars)
            results[offset] = res
            print(f"{len(res):,} triggers")

        # ── Summary table ─────────────────────────────────────────────────────
        print(f"\n{'═'*72}")
        print(f"  SUMMARY  (no CSR filter,  -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)")
        print(f"{'═'*72}")
        print(f"  {'Offset':>7}  {'Closes at':>12}  {'n':>6}  "
              f"{'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}  {'BestEV':>8}")
        print(f"  {'─'*66}")
        for offset, res in results.items():
            p   = ev_stats(res, PRAC_S, PRAC_T)
            bev, _, _ = best_ev(res)
            closes_at = f":{offset:02d}/:{offset+5:02d}/..."
            flag = "◄" if p["ev"] > 0 else ""
            print(f"  {offset:>7}  {closes_at:>12}  {p['n']:>6,}  "
                  f"{p['p_tgt']:>8.3f}  {p['p_stop']:>8.3f}  "
                  f"{p['ev']:>+9.4f}σ  {bev:>+8.4f}σ  {flag}")

        # ── CSR-filtered summary ───────────────────────────────────────────────
        print(f"\n  SUMMARY  (CSR≥{CSR_THRESHOLD:.1f} filter,  -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)")
        print(f"  {'─'*66}")
        print(f"  {'Offset':>7}  {'Closes at':>12}  {'n':>6}  "
              f"{'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}  {'BestEV':>8}")
        print(f"  {'─'*66}")
        for offset, res in results.items():
            valid    = res.dropna(subset=["csr"])
            with_mom = valid[valid["csr"] > CSR_THRESHOLD]
            p   = ev_stats(with_mom, PRAC_S, PRAC_T)
            bev, _, _ = best_ev(with_mom)
            closes_at = f":{offset:02d}/:{offset+5:02d}/..."
            flag = "◄" if p["ev"] > 0 else ""
            print(f"  {offset:>7}  {closes_at:>12}  {p['n']:>6,}  "
                  f"{p['p_tgt']:>8.3f}  {p['p_stop']:>8.3f}  "
                  f"{p['ev']:>+9.4f}σ  {bev:>+8.4f}σ  {flag}")

        # ── Year-by-year for each offset ──────────────────────────────────────
        print(f"\n  BY YEAR  (CSR≥{CSR_THRESHOLD:.1f},  -{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ)  — EV per offset")
        years = sorted(set(y for res in results.values() for y in res["year"].unique()))
        header = f"  {'Year':<6}" + "".join(f"  {'off='+str(o):>10}" for o in range(TF))
        print(header)
        print(f"  {'─'*60}")
        for y in years:
            row = f"  {y:<6}"
            for offset, res in results.items():
                valid    = res[res["year"] == y].dropna(subset=["csr"])
                with_mom = valid[valid["csr"] > CSR_THRESHOLD]
                p = ev_stats(with_mom, PRAC_S, PRAC_T)
                if math.isnan(p["ev"]):
                    row += f"  {'—':>10}"
                else:
                    flag = "+" if p["ev"] > 0 else " "
                    row += f"  {p['ev']:>+10.4f}"
            print(row)
