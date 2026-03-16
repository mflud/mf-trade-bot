"""
Intra-bar signal backtest: evaluate the 3σ continuation signal at every
1-minute mark within a developing 5-minute candle, not just at bar close.

Uses the same σ, CSR, and mean_vol as the closed-bar backtest (all computed
from completed 5-min bars only). The partial bar return is measured from the
5-min bar open to the current 1-min close. Volume is the cumulative sum so far.

Once a signal fires from a given 5-min bar, that bar is not re-evaluated
(no double-counting).

Compares intra-bar results against the closed-bar baseline side by side.

Usage:
  python src/backtest_intrabar.py            # MES + MYM
  python src/backtest_intrabar.py --sym MES
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

TF             = 5       # 5-min bars
TRAILING_BARS  = 20
MOM_BARS       = 8
MAX_HOLD_MIN   = 15      # max hold in minutes (checked against 1-min bars)
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


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
    """Build completed 5-min bars from 1-min data."""
    records, i = [], 0
    while i + TF <= len(df1):
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            continue
        records.append({
            "ts":          chunk["ts"].iloc[0],
            "ts_end":      chunk["ts"].iloc[-1],
            "open":        chunk["open"].iloc[0],
            "high":        chunk["high"].max(),
            "low":         chunk["low"].min(),
            "close":       chunk["close"].iloc[-1],
            "volume":      chunk["volume"].sum(),
            "start_1min":  i,                   # index into df1
        })
        i += TF

    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


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
    best = -999.0
    bs = bt = PRAC_S, PRAC_T
    for s in STOPS:
        for t in TARGETS:
            st = ev_stats(sub, s, t)
            if not math.isnan(st["ev"]) and st["ev"] > best:
                best, bs, bt = st["ev"], s, t
    return best, bs, bt


# ── Scan ───────────────────────────────────────────────────────────────────────

def scan_closed(bars5: pd.DataFrame) -> pd.DataFrame:
    """Baseline: signal only on completed 5-min bar close (matches regime_analysis.py)."""
    closes  = bars5["close"].values
    highs   = bars5["high"].values
    lows    = bars5["low"].values
    volumes = bars5["volume"].values
    gaps    = bars5["gap"].values
    ts_pd   = pd.DatetimeIndex(bars5["ts"].values, tz="UTC")
    n       = len(bars5)
    records = []

    for i in range(max(TRAILING_BARS, MOM_BARS), n - (MAX_HOLD_MIN // TF + 1)):
        if gaps[i - TRAILING_BARS + 1: i + MAX_HOLD_MIN // TF + 2].any():
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

        max_hold_bars = MAX_HOLD_MIN // TF
        hit_tgt  = {t: None for t in TARGETS}
        hit_stop = {s: None for s in STOPS}
        for j in range(i + 1, i + max_hold_bars + 1):
            h, l = highs[j], lows[j]
            for t in TARGETS:
                if hit_tgt[t] is None:
                    if direction == 1 and h >= tgt_prices[t]: hit_tgt[t] = j - i
                    elif direction == -1 and l <= tgt_prices[t]: hit_tgt[t] = j - i
            for s in STOPS:
                if hit_stop[s] is None:
                    if direction == 1 and l <= stop_prices[s]: hit_stop[s] = j - i
                    elif direction == -1 and h >= stop_prices[s]: hit_stop[s] = j - i

        time_exit_ret = math.log(closes[i + max_hold_bars] / entry) * direction / sigma

        records.append({
            "year":          ts_pd[i].year,
            "minute_in_bar": TF,   # closed bar = minute 5
            "csr":           csr,
            "sigma_pts":     sigma_pts,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


def scan_intrabar(bars5: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    """
    Intra-bar: evaluate signal at each 1-min mark within the developing bar.
    σ, CSR, mean_vol all come from completed 5-min bars only.
    Partial bar: open = 5-min bar open, close = current 1-min close,
                 volume = cumulative sum so far.
    Exits checked against subsequent 1-min bars (up to MAX_HOLD_MIN minutes).
    One trigger per 5-min bar (first minute that passes all filters).
    """
    closes5  = bars5["close"].values
    volumes5 = bars5["volume"].values
    gaps5    = bars5["gap"].values
    ts5_pd   = pd.DatetimeIndex(bars5["ts"].values, tz="UTC")

    closes1  = df1["close"].values
    highs1   = df1["high"].values
    lows1    = df1["low"].values
    volumes1 = df1["volume"].values
    gaps1    = df1["gap"].values
    n5       = len(bars5)
    n1       = len(df1)
    records  = []

    for i in range(max(TRAILING_BARS, MOM_BARS), n5 - (MAX_HOLD_MIN // TF + 1)):
        # Need clean window of completed bars for σ and gap check
        if gaps5[i - TRAILING_BARS + 1: i + 1].any():
            continue

        trail_rets = np.log(closes5[i - TRAILING_BARS + 1: i + 1]
                          / closes5[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol = volumes5[i - TRAILING_BARS: i].mean()
        if mean_vol == 0:
            continue

        # CSR from prior MOM_BARS completed 5-min bars
        if i >= MOM_BARS and not gaps5[i - MOM_BARS: i].any():
            mom_rets_5 = np.log(closes5[i - MOM_BARS + 1: i]
                              / closes5[i - MOM_BARS:     i - 1])
        else:
            mom_rets_5 = None

        # 5-min bar open and its starting index in df1
        bar_open      = bars5["open"].iloc[i]
        bar_start_1m  = bars5["start_1min"].iloc[i]

        fired = False
        # Evaluate at each 1-min mark within the bar (minutes 1..5)
        for minute in range(1, TF + 1):
            idx1 = bar_start_1m + minute - 1
            if idx1 >= n1:
                break
            # Skip if there's a gap within the partial bar
            if gaps1[bar_start_1m + 1: idx1 + 1].any():
                break

            partial_close  = closes1[idx1]
            partial_volume = float(volumes1[bar_start_1m: idx1 + 1].sum())

            bar_ret   = math.log(partial_close / bar_open) if bar_open > 0 else 0.0
            scaled    = bar_ret / sigma
            vol_ratio = partial_volume / mean_vol

            if abs(scaled) < MIN_SCALED or abs(scaled) > MAX_SCALED:
                continue
            if vol_ratio < MIN_VOL_RATIO:
                continue

            bar_et = ts5_pd[i].astimezone(ET)
            bar_hm = (bar_et.hour, bar_et.minute)
            if any((sh, sm) <= bar_hm < (eh, em) for sh, sm, eh, em in BLACKOUT_ET):
                break   # whole bar is blacked out

            direction = 1 if scaled > 0 else -1

            if mom_rets_5 is not None:
                csr = float(mom_rets_5.sum()) / sigma * direction
            else:
                csr = float("nan")

            entry     = partial_close
            sigma_pts = sigma * entry

            tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
            stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

            # Check exits in subsequent 1-min bars, up to MAX_HOLD_MIN minutes
            hit_tgt  = {t: None for t in TARGETS}
            hit_stop = {s: None for s in STOPS}
            exit_idx = min(idx1 + MAX_HOLD_MIN, n1 - 1)
            for j in range(idx1 + 1, exit_idx + 1):
                if gaps1[j]:
                    break
                h, l = highs1[j], lows1[j]
                for t in TARGETS:
                    if hit_tgt[t] is None:
                        if direction == 1 and h >= tgt_prices[t]: hit_tgt[t] = j - idx1
                        elif direction == -1 and l <= tgt_prices[t]: hit_tgt[t] = j - idx1
                for s in STOPS:
                    if hit_stop[s] is None:
                        if direction == 1 and l <= stop_prices[s]: hit_stop[s] = j - idx1
                        elif direction == -1 and h >= stop_prices[s]: hit_stop[s] = j - idx1

            time_exit_close = closes1[min(idx1 + MAX_HOLD_MIN, n1 - 1)]
            time_exit_ret   = math.log(time_exit_close / entry) * direction / sigma

            records.append({
                "year":          ts5_pd[i].year,
                "minute_in_bar": minute,
                "csr":           csr,
                "sigma_pts":     sigma_pts,
                "time_exit_ret": time_exit_ret,
                **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
                **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
            })
            fired = True
            break   # one trigger per 5-min bar

    return pd.DataFrame(records)


# ── Reporting ──────────────────────────────────────────────────────────────────

def report(label: str, res: pd.DataFrame):
    if res.empty:
        print(f"\n  {label}: no triggers")
        return

    n    = len(res)
    prac = ev_stats(res, PRAC_S, PRAC_T)
    bev, bs, bt = best_ev(res)
    print(f"\n{'─'*70}")
    print(f"  {label}  —  {n:,} triggers")
    print(f"{'─'*70}")
    print(f"  Overall:  EV={prac['ev']:+.4f}σ  P(tgt)={prac['p_tgt']:.3f}  "
          f"P(stop)={prac['p_stop']:.3f}  Best: -{bs:.1f}σ/+{bt:.1f}σ EV={bev:+.4f}σ")

    # By minute-in-bar
    if "minute_in_bar" in res.columns and res["minute_in_bar"].nunique() > 1:
        print(f"\n  By minute within bar  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ):")
        print(f"  {'Minute':>7}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
        print(f"  {'─'*46}")
        for m in sorted(res["minute_in_bar"].unique()):
            sub = res[res["minute_in_bar"] == m]
            p   = ev_stats(sub, PRAC_S, PRAC_T)
            flag = "◄" if p["ev"] > 0 else ""
            print(f"  {m:>7}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
                  f"{p['p_stop']:>8.3f}  {p['ev']:>+9.4f}σ  {flag}")

    # Momentum filter
    valid       = res.dropna(subset=["csr"])
    with_mom    = valid[valid["csr"] >  CSR_THRESHOLD]
    against_mom = valid[valid["csr"] < -CSR_THRESHOLD]
    neutral_mom = valid[valid["csr"].abs() <= CSR_THRESHOLD]

    print(f"\n  Momentum filter (CSR≥{CSR_THRESHOLD:.1f}):")
    print(f"  {'Slice':<22}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
    print(f"  {'─'*54}")
    for lbl, sub in [("WITH momentum",    with_mom),
                     ("AGAINST momentum", against_mom),
                     ("NEUTRAL",          neutral_mom)]:
        p    = ev_stats(sub, PRAC_S, PRAC_T)
        flag = "◄" if p["ev"] > 0 else ""
        ev_s = f"{p['ev']:>+9.4f}σ" if not math.isnan(p["ev"]) else f"{'—':>10}"
        print(f"  {lbl:<22}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
              f"{p['p_stop']:>8.3f}  {ev_s}  {flag}")

    # Year by year
    print(f"\n  By year  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ):")
    print(f"  {'Year':<6}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
    print(f"  {'─'*44}")
    for y in sorted(res["year"].unique()):
        sub  = res[res["year"] == y]
        p    = ev_stats(sub, PRAC_S, PRAC_T)
        flag = "◄" if p["ev"] > 0 else ""
        print(f"  {y:<6}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
              f"{p['p_stop']:>8.3f}  {p['ev']:>+9.4f}σ  {flag}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None, help="MES or MYM (default: both)")
    args = parser.parse_args()

    syms = {args.sym: INSTRUMENTS[args.sym]} if args.sym else INSTRUMENTS

    for sym, cache in syms.items():
        print(f"\n{'═'*70}")
        print(f"  {sym}  —  Intra-bar vs Closed-bar Signal Comparison")
        print(f"{'═'*70}")
        print(f"  σ window: {TRAILING_BARS} bars ({TRAILING_BARS*TF} min)  "
              f"CSR window: {MOM_BARS} bars ({MOM_BARS*TF} min)  "
              f"Hold: {MAX_HOLD_MIN} min")

        print(f"\nLoading {cache} …")
        df1   = load_1min(cache)
        print(f"  {len(df1):,} 1-min bars  ({df1['ts'].min().date()} → {df1['ts'].max().date()})")

        print("  Building 5-min bars …")
        bars5 = make_5min_bars(df1)
        print(f"  {len(bars5):,} 5-min bars")

        print("  Scanning closed bars …")
        res_closed = scan_closed(bars5)
        print(f"  {len(res_closed):,} triggers (closed bar)")

        print("  Scanning intra-bar (1-min resolution) …")
        res_intra = scan_intrabar(bars5, df1)
        print(f"  {len(res_intra):,} triggers (intra-bar, first qualifying minute)")

        # Summary comparison
        print(f"\n{'═'*70}")
        print(f"  SUMMARY  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ, no CSR filter)")
        print(f"{'═'*70}")
        print(f"  {'Mode':<30}  {'n':>6}  {'P(tgt)':>8}  {'P(stop)':>8}  {'EV':>9}")
        print(f"  {'─'*60}")
        for lbl, res in [("Closed bar (bar close only)", res_closed),
                         ("Intra-bar (any 1-min mark)",  res_intra)]:
            p = ev_stats(res, PRAC_S, PRAC_T)
            print(f"  {lbl:<30}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
                  f"{p['p_stop']:>8.3f}  {p['ev']:>+9.4f}σ")

        # CSR-filtered summary
        print(f"\n  SUMMARY  (-{PRAC_S:.1f}σ/+{PRAC_T:.1f}σ, CSR≥{CSR_THRESHOLD:.1f} filter)")
        print(f"  {'─'*60}")
        for lbl, res in [("Closed bar (bar close only)", res_closed),
                         ("Intra-bar (any 1-min mark)",  res_intra)]:
            valid    = res.dropna(subset=["csr"])
            with_mom = valid[valid["csr"] > CSR_THRESHOLD]
            p = ev_stats(with_mom, PRAC_S, PRAC_T)
            print(f"  {lbl:<30}  {p['n']:>6,}  {p['p_tgt']:>8.3f}  "
                  f"{p['p_stop']:>8.3f}  {p['ev']:>+9.4f}σ")

        report(f"{sym} — Closed bar baseline", res_closed)
        report(f"{sym} — Intra-bar (first qualifying 1-min)", res_intra)
