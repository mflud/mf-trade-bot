"""
Combined backtest: does a 5-min HA streak aligned with the primary 3σ signal
improve EV?

At each signal trigger, we record:
  ha_streak   — how many consecutive same-color HA bars end at the signal bar
                (1 = only this bar, 2 = this + previous, etc.)
  ha_aligned  — True if streak color matches signal direction

Then we split EV by streak length and compare to the unfiltered baseline.

Usage:
  python src/backtest_ha_signal.py
  python src/backtest_ha_signal.py --sym MYM
"""

import argparse
import math
import sys
from datetime import timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import backtest_tod as bt

ET = ZoneInfo("America/New_York")


# ── HA helpers ─────────────────────────────────────────────────────────────────

def compute_ha_series(opens, highs, lows, closes):
    """Return (ha_open, ha_close) arrays for a price series."""
    n = len(opens)
    ha_close = (opens + highs + lows + closes) / 4.0
    ha_open  = np.empty(n)
    ha_open[0] = (opens[0] + closes[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    return ha_open, ha_close


def build_ha_streaks(bars: pd.DataFrame) -> np.ndarray:
    """
    For each bar, compute the HA streak length ending at that bar:
      streak[i] > 0  → i consecutive green HA bars ending here (LONG-aligned)
      streak[i] < 0  → i consecutive red HA bars ending here  (SHORT-aligned)
      (sign encodes color, magnitude encodes length)

    HA is reset at each calendar session (ET date) so there's no overnight
    carry-over — matching how signal_monitor would compute it in real time.
    """
    ts_et = bars["ts"].dt.tz_convert(ET)
    dates  = ts_et.dt.date.values
    opens  = bars["open"].values
    highs  = bars["high"].values
    lows   = bars["low"].values
    closes = bars["close"].values
    n      = len(bars)

    streak = np.zeros(n, dtype=int)

    # Process session by session
    unique_dates = pd.unique(dates)
    for d in unique_dates:
        idx = np.where(dates == d)[0]
        if len(idx) < 2:
            continue
        ha_o, ha_c = compute_ha_series(opens[idx], highs[idx],
                                        lows[idx],   closes[idx])
        green = ha_c >= ha_o   # True = green, False = red
        run = 0
        for k, gi in enumerate(green):
            if k == 0:
                run = 1 if gi else -1
            else:
                if gi == (run > 0):   # same color as current streak
                    run += (1 if gi else -1)
                else:
                    run = 1 if gi else -1
            streak[idx[k]] = run

    return streak


# ── Extended scan ──────────────────────────────────────────────────────────────

def scan_with_ha(bars: pd.DataFrame) -> pd.DataFrame:
    """Run bt.scan() then attach ha_streak and ha_aligned columns."""
    # We need the streak array indexed to the same bars DataFrame
    streak_arr = build_ha_streaks(bars)

    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    ts_pd   = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n       = len(bars)

    records = []
    for i in range(max(bt.TRAILING_BARS, bt.MOM_BARS), n - bt.MAX_BARS_HOLD):
        if gaps[i - bt.TRAILING_BARS + 1: i + bt.MAX_BARS_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - bt.TRAILING_BARS + 1: i + 1]
                          / closes[i - bt.TRAILING_BARS:     i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - bt.TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < bt.MIN_SCALED or abs(scaled) > bt.MAX_SCALED \
                or vol_ratio < bt.MIN_VOL_RATIO:
            continue

        bar_et    = ts_pd[i].astimezone(ET)
        bar_hm    = (bar_et.hour, bar_et.minute)
        direction = 1 if scaled > 0 else -1
        entry     = closes[i]

        bucket_label = "other"
        for label, sh, sm, eh, em in bt.TOD_BUCKETS:
            if (sh, sm) <= bar_hm < (eh, em):
                bucket_label = label
                break

        if i >= bt.MOM_BARS and not gaps[i - bt.MOM_BARS: i].any():
            mom_rets = np.log(closes[i - bt.MOM_BARS + 1: i]
                            / closes[i - bt.MOM_BARS:     i - 1])
            csr = float(mom_rets.sum()) / sigma * direction
        else:
            csr = float("nan")

        # ── HA streak ──────────────────────────────────────────────────────────
        s = streak_arr[i]
        ha_streak   = abs(s)                     # length (always >= 1)
        ha_color    = 1 if s > 0 else -1          # +1=green/LONG, -1=red/SHORT
        ha_aligned  = (ha_color == direction)     # does streak match signal?

        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in bt.TARGETS}
        stop_prices = {s_: entry * math.exp(-direction * s_ * sigma) for s_ in bt.STOPS}

        hit_tgt  = {t: None for t in bt.TARGETS}
        hit_stop = {s_: None for s_ in bt.STOPS}
        for j in range(i + 1, i + bt.MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            for t in bt.TARGETS:
                if hit_tgt[t] is None:
                    if direction == 1  and h >= tgt_prices[t]:  hit_tgt[t]  = j - i
                    elif direction == -1 and l <= tgt_prices[t]: hit_tgt[t]  = j - i
            for s_ in bt.STOPS:
                if hit_stop[s_] is None:
                    if direction == 1  and l <= stop_prices[s_]: hit_stop[s_] = j - i
                    elif direction == -1 and h >= stop_prices[s_]: hit_stop[s_] = j - i

        time_exit_ret = math.log(closes[i + bt.MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
            "bucket":        bucket_label,
            "csr":           csr,
            "ha_streak":     ha_streak,
            "ha_aligned":    ha_aligned,
            "time_exit_ret": time_exit_ret,
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in bt.TARGETS},
            **{f"hit_stop_{s_}": hit_stop[s_] for s_ in bt.STOPS},
        })

    return pd.DataFrame(records)


# ── Reporting ──────────────────────────────────────────────────────────────────

def ev_row(sub, s, t, label, n_base):
    p = bt.ev_stats(sub, s, t)
    if math.isnan(p["ev"]):
        return f"  {label:<28}  {'—':>5}  {'—':>6}  {'—':>7}  {'—':>8}  {'—':>5}"
    pct   = f"{100*p['n']/n_base:.0f}%" if n_base else "—"
    flag  = " ◄" if p["ev"] > 0 else ""
    return (f"  {label:<28}  {p['n']:>5,}  {pct:>5}  "
            f"{p['p_tgt']:>6.3f}  {p['p_stop']:>7.3f}  "
            f"{p['ev']:>+8.4f}σ{flag}")


def print_ha_table(res: pd.DataFrame, csr_on: bool, s: float, t: float, sym: str):
    tag = f"CSR≥{bt.CSR_THRESHOLD}" if csr_on else "No CSR filter"
    print(f"\n{'═'*72}")
    print(f"  {sym}  HA streak filter  —  {tag}  "
          f"(stop={s}σ / target={t}σ)")
    print(f"{'═'*72}")
    hdr = (f"  {'Filter':<28}  {'n':>5}  {'pct':>5}  "
           f"{'P(tgt)':>6}  {'P(stop)':>7}  {'EV':>9}")
    print(hdr)
    print(f"  {'─'*70}")

    sub = res.copy()
    if csr_on:
        sub = sub.dropna(subset=["csr"])
        sub = sub[sub["csr"] > bt.CSR_THRESHOLD]

    # RTH only (10:00–15:44) matching our HA backtest window
    rth_buckets = {"10:00–11:00","11:00–12:00","12:00–13:00",
                   "13:00–14:00","14:00–15:00","15:00–16:00"}
    sub_rth = sub[sub["bucket"].isin(rth_buckets)]

    n_base     = len(sub)
    n_base_rth = len(sub_rth)

    print(ev_row(sub,     s, t, "ALL signals",            n_base))
    print(ev_row(sub_rth, s, t, "RTH 10:00–15:44",        n_base_rth))
    print(f"  {'─'*70}")

    for sub_, nb, prefix in [(sub, n_base, ""), (sub_rth, n_base_rth, "RTH ")]:
        aln  = sub_[sub_["ha_aligned"] == True]
        mis  = sub_[sub_["ha_aligned"] == False]
        aln1 = aln[aln["ha_streak"] >= 1]
        aln2 = aln[aln["ha_streak"] >= 2]
        aln3 = aln[aln["ha_streak"] >= 3]
        aln4 = aln[aln["ha_streak"] >= 4]
        print(ev_row(aln1, s, t, f"{prefix}HA aligned  ≥1 bar",  nb))
        print(ev_row(aln2, s, t, f"{prefix}HA aligned  ≥2 bars", nb))
        print(ev_row(aln3, s, t, f"{prefix}HA aligned  ≥3 bars", nb))
        print(ev_row(aln4, s, t, f"{prefix}HA aligned  ≥4 bars", nb))
        print(ev_row(mis,  s, t, f"{prefix}HA opposing",          nb))
        print(f"  {'─'*70}")

    # Year-by-year for best HA filter in RTH
    print(f"\n  Year-by-year (RTH 10:00–15:44, HA aligned ≥3):")
    sub_y = sub_rth[sub_rth["ha_aligned"] == True]
    sub_y = sub_y[sub_y["ha_streak"] >= 3].copy()
    if "bucket" not in res.columns:
        return
    # reconstruct year from bucket — need ts; approximate from res index
    # Instead use the full res with ts if available
    print(f"  (n={len(sub_y)}  across all years — year split needs ts, see note)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default=None)
    args = parser.parse_args()

    syms = {args.sym: bt.INSTRUMENTS[args.sym]} if args.sym \
           else bt.INSTRUMENTS

    for sym, path in syms.items():
        print(f"\nLoading {path} …", flush=True)
        df1  = bt.load_1min(path)
        bars = bt.make_offset_bars(df1, 0)
        print(f"  {len(bars):,} 5-min bars — scanning with HA …", end=" ", flush=True)
        res  = scan_with_ha(bars)
        print(f"{len(res):,} signal triggers")

        # Quick streak distribution
        print(f"\n  HA streak distribution at signal time:")
        for aligned, tag in [(True, "Aligned"), (False, "Opposing")]:
            sub = res[res["ha_aligned"] == aligned]
            counts = sub["ha_streak"].value_counts().sort_index()
            pct    = 100 * len(sub) / len(res)
            print(f"    {tag} ({pct:.0f}%): " +
                  "  ".join(f"streak={k}: {v:,}" for k, v in counts.items()
                            if k <= 6))

        for csr_on in [False, True]:
            print_ha_table(res, csr_on, bt.PRAC_S, bt.PRAC_T, sym)

        # Year breakdown — attach year via ts
        bars2 = bars.copy()
        bars2["ts_et"] = bars2["ts"].dt.tz_convert(ET)
        bars2["year"]  = bars2["ts_et"].dt.year

        # Re-scan attaching bar index so we can merge year
        closes  = bars["close"].values
        highs   = bars["high"].values
        gaps    = bars["gap"].values
        volumes = bars["volume"].values

        # Attach year to res by rebuilding with bar index
        year_records = []
        for i in range(max(bt.TRAILING_BARS, bt.MOM_BARS),
                       len(bars) - bt.MAX_BARS_HOLD):
            if gaps[i - bt.TRAILING_BARS + 1: i + bt.MAX_BARS_HOLD + 1].any():
                continue
            trail_rets = np.log(closes[i - bt.TRAILING_BARS + 1: i + 1]
                              / closes[i - bt.TRAILING_BARS:     i    ])
            sigma = np.std(trail_rets, ddof=1)
            if sigma == 0: continue
            mean_vol  = volumes[i - bt.TRAILING_BARS: i].mean()
            vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
            bar_ret   = math.log(closes[i] / closes[i - 1])
            scaled    = bar_ret / sigma
            if abs(scaled) < bt.MIN_SCALED or abs(scaled) > bt.MAX_SCALED \
                    or vol_ratio < bt.MIN_VOL_RATIO: continue
            year_records.append({"idx": i, "year": bars2["year"].iloc[i]})

        year_map = pd.DataFrame(year_records).set_index("idx")["year"]
        res["year"] = year_map.values if len(year_map) == len(res) else None

        rth_buckets = {"10:00–11:00","11:00–12:00","12:00–13:00",
                       "13:00–14:00","14:00–15:00","15:00–16:00"}

        print(f"\n  Year-by-year  (RTH, no CSR filter):")
        print(f"  {'Year':>6}  {'n_all':>6}  {'EV_all':>9}  "
              f"{'n_ha≥3':>7}  {'pct':>5}  {'EV_ha≥3':>10}  "
              f"{'n_opp':>6}  {'EV_opp':>9}")
        print(f"  {'─'*72}")

        rth = res[res["bucket"].isin(rth_buckets)].copy()
        for yr, grp in rth.groupby("year"):
            p_all  = bt.ev_stats(grp, bt.PRAC_S, bt.PRAC_T)
            ha3    = grp[(grp["ha_aligned"]) & (grp["ha_streak"] >= 3)]
            opp    = grp[~grp["ha_aligned"]]
            p_ha3  = bt.ev_stats(ha3, bt.PRAC_S, bt.PRAC_T)
            p_opp  = bt.ev_stats(opp, bt.PRAC_S, bt.PRAC_T)
            pct    = f"{100*len(ha3)/len(grp):.0f}%" if len(grp) else "—"
            ev_all = f"{p_all['ev']:+.4f}σ" if not math.isnan(p_all["ev"]) else "—"
            ev_ha3 = f"{p_ha3['ev']:+.4f}σ" if not math.isnan(p_ha3["ev"]) else "—"
            ev_opp = f"{p_opp['ev']:+.4f}σ" if not math.isnan(p_opp["ev"]) else "—"
            flag   = " ◄" if (not math.isnan(p_ha3["ev"]) and p_ha3["ev"] > 0) else ""
            print(f"  {int(yr):>6}  {len(grp):>6,}  {ev_all:>9}  "
                  f"{len(ha3):>7,}  {pct:>5}  {ev_ha3:>9}{flag}  "
                  f"{len(opp):>6,}  {ev_opp:>9}")

        print()


if __name__ == "__main__":
    main()
