"""
Multi-instrument strategy comparison: MES, MNQ, M2K, MYM.

Runs the same 5-min bar signal strategy on all four micro futures:
  Signal  : |scaled return| >= 3σ  AND  volume >= 1.5× 100-bar mean
  Hold    : up to 3 bars (15 min)
  Grid    : stop ∈ {0.5, 1.0, 1.5, 2.0}σ  ×  target ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}σ

Data is fetched fresh for each instrument and cached to {symbol}_hist_1min.csv.
The MES cache (mes_hist_1min.csv) is reused if present.

Usage:
  python src/multi_instrument_sim.py
  python src/multi_instrument_sim.py --fetch   # force re-download even if cache exists
"""

import argparse
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from topstep_client import TopstepClient

# ── Config ────────────────────────────────────────────────────────────────────

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TRAILING_BARS        = 100
TF                   = 5        # 5-min bars
MAX_BARS_HOLD        = 3
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5

STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Contracts in expiry order (oldest active → current front month).
# The fetcher iterates all and stitches them together.
INSTRUMENTS = {
    "MES": {
        "cache":     "mes_hist_1min.csv",
        "contracts": [
            "CON.F.US.MES.H25",
            "CON.F.US.MES.M25",
            "CON.F.US.MES.U25",
            "CON.F.US.MES.Z25",
            "CON.F.US.MES.H26",
        ],
        "tick":  0.25,
        "point_value": 5.0,    # $ per point
    },
    "ES": {
        "cache":     "es_hist_1min.csv",
        "contracts": ["CON.F.US.EP.H26"],
        "tick":  0.25,
        "point_value": 50.0,
    },
    "M2K": {
        "cache":     "m2k_hist_1min.csv",
        "contracts": [
            "CON.F.US.M2K.H25",
            "CON.F.US.M2K.M25",
            "CON.F.US.M2K.U25",
            "CON.F.US.M2K.Z25",
            "CON.F.US.M2K.H26",
        ],
        "tick":  0.10,
        "point_value": 5.0,
    },
    "MYM": {
        "cache":     "mym_hist_1min.csv",
        "contracts": [
            "CON.F.US.MYM.H25",
            "CON.F.US.MYM.M25",
            "CON.F.US.MYM.U25",
            "CON.F.US.MYM.Z25",
            "CON.F.US.MYM.H26",
        ],
        "tick":  1.0,
        "point_value": 0.50,
    },
    "RTY": {
        "cache":     "rty_hist_1min.csv",
        "contracts": ["CON.F.US.RTY.H26"],
        "tick":  0.10,
        "point_value": 50.0,
    },
    "YM": {
        "cache":     "ym_hist_1min.csv",
        "contracts": ["CON.F.US.YM.H26"],
        "tick":  1.0,
        "point_value": 5.0,
    },
    "NKD": {
        "cache":     "nkd_hist_1min.csv",
        "contracts": ["CON.F.US.NKD.M26"],
        "tick":  5.0,
        "point_value": 5.0,
    },
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def fetch_1min(client: TopstepClient, contract_ids: list[str]) -> pd.DataFrame:
    """Fetch 1-min bars across multiple contracts (newest contract wins on overlap)."""
    far_past   = datetime(2020, 1, 1, tzinfo=timezone.utc)
    far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
    merged: dict[str, dict] = {}

    for cid in contract_ids:
        bars = client.get_bars(
            contract_id=cid,
            start=far_past,
            end=far_future,
            unit=TopstepClient.MINUTE,
            unit_number=1,
            limit=20000,
        )
        if not bars:
            continue
        print(f"  {cid}: {len(bars)} bars")
        for b in bars:
            merged[b["t"]] = b

    if not merged:
        return pd.DataFrame()

    rows = sorted(merged.values(), key=lambda b: b["t"])
    df = pd.DataFrame(rows)
    df = df.rename(columns={"t": "ts", "o": "open", "h": "high",
                             "l": "low",  "c": "close", "v": "volume"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    return df


def load_or_fetch(symbol: str, info: dict, client: TopstepClient,
                  force_fetch: bool = False) -> pd.DataFrame:
    cache = info["cache"]
    if not force_fetch:
        try:
            df = pd.read_csv(cache, parse_dates=["ts"])
            if not df["ts"].dt.tz:
                df["ts"] = df["ts"].dt.tz_localize("UTC")
            days = (df["ts"].max() - df["ts"].min()).days
            if days >= 5:
                print(f"  {symbol}: loaded {len(df):,} rows from {cache} ({days} days)")
                return df
            print(f"  {symbol}: cache only {days} days — re-fetching")
        except FileNotFoundError:
            pass

    print(f"  {symbol}: fetching 1-min bars …")
    df = fetch_1min(client, info["contracts"])
    if df.empty:
        print(f"  {symbol}: no data returned!")
        return df
    df.to_csv(cache, index=False)
    days = (df["ts"].max() - df["ts"].min()).days
    print(f"  {symbol}: {len(df):,} rows saved to {cache} ({days} days)")
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter settlement gap, sort, add gap flag."""
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
    records, i = [], 0
    n = TF
    while i + n <= len(df1):
        chunk = df1.iloc[i : i + n]
        internal = chunk["gap"].iloc[1:].values
        if internal.any():
            i += int(internal.argmax()) + 1
            continue
        records.append({
            "ts":     chunk["ts"].iloc[0],
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
        i += n
    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=n)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Scan ──────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame) -> pd.DataFrame:
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    n       = len(bars)
    records = []

    for i in range(TRAILING_BARS, n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1 : i + MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1 : i + 1]
                          / closes[i - TRAILING_BARS     : i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAILING_BARS : i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")

        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or vol_ratio < MIN_VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1
        entry     = closes[i]
        sigma_pts = sigma * entry

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}

        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            bar_idx = j - i
            h, l    = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if direction == 1 and h >= tgt_prices[t]:
                        hit_tgt[t] = bar_idx
                    elif direction == -1 and l <= tgt_prices[t]:
                        hit_tgt[t] = bar_idx
            for s in STOPS:
                if hit_stop[s] is None:
                    if direction == 1 and l <= stop_prices[s]:
                        hit_stop[s] = bar_idx
                    elif direction == -1 and h >= stop_prices[s]:
                        hit_stop[s] = bar_idx

        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
            "sigma":         sigma,
            "sigma_pts":     sigma_pts,
            "scaled":        abs(scaled),
            "vol_ratio":     vol_ratio,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV calculation ────────────────────────────────────────────────────────────

def ev_for_combo(res: pd.DataFrame, s: float, t: float) -> dict:
    ht = res[f"hit_tgt_{t}"].notna().values
    hs = res[f"hit_stop_{s}"].notna().values
    ht_first = ht & ~(hs & (res[f"hit_stop_{s}"].fillna(999) <= res[f"hit_tgt_{t}"].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    time_ret = res["time_exit_ret"].values
    p_tgt    = ht_first.mean()
    p_stop   = hs_first.mean()
    p_nei    = neither.mean()
    ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
    ev       = p_tgt * t - p_stop * s + p_nei * ev_nei
    return {"ev": ev, "p_tgt": p_tgt, "p_stop": p_stop, "p_nei": p_nei, "ev_nei": ev_nei}


def best_combo(res: pd.DataFrame) -> tuple[float, float, dict]:
    best_ev, best_s, best_t, best_stats = -999, None, None, None
    for s in STOPS:
        for t in TARGETS:
            stats = ev_for_combo(res, s, t)
            if stats["ev"] > best_ev:
                best_ev, best_s, best_t, best_stats = stats["ev"], s, t, stats
    return best_s, best_t, best_stats


# ── Full report for one instrument ────────────────────────────────────────────

def instrument_report(symbol: str, res: pd.DataFrame, point_value: float):
    n = len(res)
    sp = res["sigma_pts"].values

    print(f"\n{'═'*72}")
    print(f"  {symbol}   n={n} qualifying triggers  "
          f"(≥{MIN_SCALED:.0f}σ, vol≥{MIN_VOL_RATIO:.1f}×)  hold≤{MAX_BARS_HOLD*TF}min")
    print(f"{'═'*72}")

    print(f"\n  1σ in {symbol} points (p25/median/p75/p90):")
    for pct in [25, 50, 75, 90]:
        v = np.percentile(sp, pct)
        print(f"    p{pct}: {v:.2f} pts")

    # EV grid
    print(f"\n  EV grid (σ units)  — rows=stop, cols=target:")
    hdr = f"  {'Stop \\ Target':>14}" + "".join(f"  +{t:.1f}σ" for t in TARGETS)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for s in STOPS:
        row = f"  -{s:.1f}σ          "
        for t in TARGETS:
            stats = ev_for_combo(res, s, t)
            row += f"  {stats['ev']:>+5.3f}"
        print(row)

    best_s, best_t, stats = best_combo(res)
    print(f"\n  Best:  stop={best_s:.1f}σ  target={best_t:.1f}σ  →  EV={stats['ev']:+.4f}σ")
    print(f"    P(target): {stats['p_tgt']:.3f}  P(stop): {stats['p_stop']:.3f}  "
          f"P(time exit): {stats['p_nei']:.3f}  (mean exit={stats['ev_nei']:+.3f}σ)")

    # Practical combo: fixed at -1.5σ / +2.5σ
    prac_s, prac_t = 1.5, 2.5
    prac = ev_for_combo(res, prac_s, prac_t)
    print(f"\n  Practical (-{prac_s}σ/+{prac_t}σ):")
    print(f"    EV={prac['ev']:+.4f}σ  "
          f"P(tgt)={prac['p_tgt']:.3f}  P(stop)={prac['p_stop']:.3f}  "
          f"P(time)={prac['p_nei']:.3f}")
    for pct, label in [(25,"p25"),(50,"median"),(75,"p75"),(90,"p90")]:
        sigma_p = np.percentile(sp, pct)
        tgt_pts  = prac_t * sigma_p
        stop_pts = prac_s * sigma_p
        ev_pts   = prac["ev"] * sigma_p * point_value
        print(f"    {label}: 1σ={sigma_p:.2f}pts  tgt={tgt_pts:.2f}  "
              f"stop={stop_pts:.2f}  EV={ev_pts:+.2f}$/trade")

    return {
        "symbol":   symbol,
        "n":        n,
        "best_s":   best_s,
        "best_t":   best_t,
        "best_ev":  stats["ev"],
        "prac_ev":  prac["ev"],
        "p_tgt":    prac["p_tgt"],
        "p_stop":   prac["p_stop"],
        "med_sigma_pts": np.percentile(sp, 50),
        "point_value":   point_value,
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def summary_table(results: list[dict]):
    print(f"\n\n{'═'*90}")
    print("  SUMMARY COMPARISON  (-1.5σ stop / +2.5σ target, ≤15min hold)")
    print(f"{'═'*90}")
    hdr = (f"  {'Symbol':<6}  {'n':>5}  {'EV(σ)':>8}  "
           f"{'P(tgt)':>8}  {'P(stop)':>8}  "
           f"{'MedianσPts':>11}  {'EV($/trade)':>12}  "
           f"{'BestStop':>9}  {'BestTgt':>8}  {'BestEV(σ)':>10}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in results:
        med_ev_usd = r["prac_ev"] * r["med_sigma_pts"] * r["point_value"]
        print(f"  {r['symbol']:<6}  {r['n']:>5}  {r['prac_ev']:>+8.4f}  "
              f"{r['p_tgt']:>8.3f}  {r['p_stop']:>8.3f}  "
              f"{r['med_sigma_pts']:>11.2f}  {med_ev_usd:>+12.2f}  "
              f"{r['best_s']:>8.1f}σ  {r['best_t']:>7.1f}σ  {r['best_ev']:>+10.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="Force re-download even if cache exists")
    args = parser.parse_args()

    from topstep_client import TopstepClient
    client = TopstepClient()
    client.login()
    print("Authenticated.\n")

    summary = []
    for symbol, info in INSTRUMENTS.items():
        print(f"── {symbol} ──────────────────────────────────────")
        df1 = load_or_fetch(symbol, info, client, force_fetch=args.fetch)
        if df1.empty:
            print(f"  Skipping {symbol} (no data)")
            continue

        df1   = prepare(df1)
        bars  = make_5min_bars(df1)
        print(f"  {len(bars):,} 5-min bars")

        res = scan(bars)
        if len(res) < 10:
            print(f"  Insufficient triggers ({len(res)}), skipping.")
            continue

        row = instrument_report(symbol, res, info["point_value"])
        summary.append(row)

    if summary:
        summary_table(summary)
