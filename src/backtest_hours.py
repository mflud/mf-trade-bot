"""
Time-of-day stratified backtest.

Runs the standard 3σ + CSR + PL signal on M2K and NKD, then breaks
results by session window so we can see where the edge lives.

Sessions defined:
  nyse  — 09:30–16:00 ET  (13:30–20:00 UTC)
  asia  — 00:00–06:30 UTC (09:00–15:30 JST = TSE cash session)
  euro  — 07:00–13:30 UTC (London open through NYSE open)
  globex — everything except the CME settlement gap

Usage:
  python src/backtest_hours.py --sym M2K
  python src/backtest_hours.py --sym NKD
"""

import argparse
import math
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Parameters (identical to backtest_pl_sizing.py) ───────────────────────────

TF              = 5
TRAILING_BARS   = 20
MAX_BARS_HOLD   = 5
MIN_SCALED      = 3.0
MIN_VOL_RATIO   = 1.5
STOP_SIG        = 2.0
TGT_SIG         = 3.0
CSR_THRESHOLD   = 1.5
GK_VOL_BARS     = 20
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
CSR_LOW_WIN     = 4
CSR_NORM_WIN    = 8
GK_LOW_THRESH   = 0.08
BLACKOUT_ET     = (8, 0, 9, 0)

N               = 10
STRONG_THRESH   = 0.5
SIGMA_1M_WIN    = 100

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
    "M2K": "m2k_hist_1min.csv",
    "NKD": "nkd_hist_1min.csv",
}

# Session windows: list of (start_utc_hour_inclusive, end_utc_hour_exclusive)
# fractional hours supported (e.g. 13.5 = 13:30)
SESSIONS = {
    "nyse":   [(13.5, 20.0)],
    "asia":   [(0.0, 6.5)],
    "euro":   [(7.0, 13.5)],
    "overnight": [(20.0, 24.0), (0.0, 6.5)],   # US close → TSE close
}


# ── Data loading (same as backtest_pl_sizing.py) ──────────────────────────────

def load_1min(path):
    df = pd.read_csv(path, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["log_ret"]  = np.log(df["close"] / df["close"].shift(1))
    sigma_1m       = df["log_ret"].rolling(SIGMA_1M_WIN, min_periods=SIGMA_1M_WIN).std(ddof=1)
    df["sigma_1m"] = sigma_1m * df["close"]
    return df


def make_5min_bars(df1):
    df1 = df1.copy()
    df1["gap"] = df1["ts"].diff() > pd.Timedelta(minutes=2)
    records, i, n = [], 0, len(df1)
    while i < n and df1["ts"].iloc[i].minute % TF != 0:
        i += 1
    while i + TF <= n:
        chunk = df1.iloc[i: i + TF]
        if chunk["gap"].iloc[1:].any():
            gap_pos = int(chunk["gap"].iloc[1:].values.argmax()) + 1
            i += gap_pos
            while i < n and df1["ts"].iloc[i].minute % TF != 0:
                i += 1
            continue
        records.append({"ts": chunk["ts"].iloc[-1], "open": chunk["open"].iloc[0],
                        "high": chunk["high"].max(), "low": chunk["low"].min(),
                        "close": chunk["close"].iloc[-1], "volume": chunk["volume"].sum()})
        i += TF
    bars = pd.DataFrame(records)
    if bars.empty:
        return bars
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


def gk_vol(opens, highs, lows, closes):
    bpy = 252 * 23 * 60 / TF
    vals = [0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2
            for o, h, l, c in zip(opens, highs, lows, closes) if min(o, h, l, c) > 0]
    return math.sqrt(max(0.0, float(np.mean(vals))) * bpy) if vals else 0.0


def scan_5min(bars):
    MAX_CSR  = max(CSR_LOW_WIN, CSR_NORM_WIN)
    closes   = bars["close"].values;  highs  = bars["high"].values
    lows     = bars["low"].values;    opens  = bars["open"].values
    volumes  = bars["volume"].values; gaps   = bars["gap"].values
    ts_pd    = pd.DatetimeIndex(bars["ts"].values, tz="UTC")
    n        = len(bars);  records = []
    lookback = max(TRAILING_BARS, MAX_CSR, GK_VOL_BARS) + 1

    for i in range(lookback, n - MAX_BARS_HOLD):
        if gaps[i - TRAILING_BARS + 1: i + MAX_BARS_HOLD + 1].any():
            continue
        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1] /
                            closes[i - TRAILING_BARS:     i    ])
        sigma = trail_rets.std(ddof=1)
        if sigma == 0:
            continue
        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or abs(scaled) > 99.0:
            continue
        mean_vol  = volumes[i - TRAILING_BARS: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")
        if vol_ratio < MIN_VOL_RATIO:
            continue
        direction = 1 if scaled > 0 else -1
        gk = gk_vol(opens[i - GK_VOL_BARS: i], highs[i - GK_VOL_BARS: i],
                    lows[i - GK_VOL_BARS: i],  closes[i - GK_VOL_BARS: i])
        csr_win    = CSR_LOW_WIN if gk < GK_LOW_THRESH else CSR_NORM_WIN
        prior_rets = np.log(closes[i - csr_win: i] / closes[i - csr_win - 1: i - 1])
        csr        = float(prior_rets.sum()) / sigma * direction
        if csr < CSR_THRESHOLD:
            continue

        entry   = closes[i]
        tgt_px  = entry * math.exp( direction * TGT_SIG  * sigma)
        stop_px = entry * math.exp(-direction * STOP_SIG * sigma)
        hit_tgt = hit_stop = False
        for j in range(i + 1, i + MAX_BARS_HOLD + 1):
            h, l = highs[j], lows[j]
            if not hit_tgt and not hit_stop:
                if direction == 1:
                    if h >= tgt_px: hit_tgt = True
                    elif l <= stop_px: hit_stop = True
                else:
                    if l <= tgt_px: hit_tgt = True
                    elif h >= stop_px: hit_stop = True
            else:
                break
        time_exit_r = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        if hit_tgt and not hit_stop:
            outcome_r = TGT_SIG
        elif hit_stop and not hit_tgt:
            outcome_r = -STOP_SIG
        else:
            outcome_r = time_exit_r

        utc_frac = ts_pd[i].hour + ts_pd[i].minute / 60.0
        bar_et   = ts_pd[i].astimezone(ET)

        records.append({
            "ts":           ts_pd[i],
            "year":         ts_pd[i].year,
            "utc_frac":     utc_frac,
            "et_hm":        (bar_et.hour, bar_et.minute),
            "direction":    direction,
            "sigma":        sigma,
            "hit_tgt":      hit_tgt,
            "hit_stop":     hit_stop,
            "time_exit_r":  time_exit_r,
            "outcome_r":    outcome_r,
        })
    return records


def attach_pl(records, df1):
    log_ret   = df1["log_ret"].values
    sigma_1m  = df1["sigma_1m"].values
    ts_ns     = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts_map    = {v: i for i, v in enumerate(ts_ns)}

    for rec in records:
        sig_ts_ns = int(rec["ts"].as_unit("ns").value)
        pre_ts_ns = sig_ts_ns - TF * 60 * 10**9
        end_idx   = ts_map.get(pre_ts_ns)
        if end_idx is None:
            rec["pl_aligned"] = float("nan"); continue
        start_idx = end_idx - N + 1
        if start_idx < SIGMA_1M_WIN:
            rec["pl_aligned"] = float("nan"); continue
        rets     = log_ret[start_idx: end_idx + 1]
        if np.any(np.isnan(rets)):
            rec["pl_aligned"] = float("nan"); continue
        sum_absr = float(np.abs(rets).sum())
        if sum_absr == 0:
            rec["pl_aligned"] = 0.0; continue
        rec["pl_aligned"] = float(rets.sum()) / sum_absr * rec["direction"]


# ── Session helpers ────────────────────────────────────────────────────────────

def in_session(utc_frac, windows):
    for start, end in windows:
        if start <= utc_frac < end:
            return True
    return False


def session_label(utc_frac):
    if 0.0 <= utc_frac < 6.5:
        return "asia"
    if 7.0 <= utc_frac < 13.5:
        return "euro"
    if 13.5 <= utc_frac < 20.0:
        return "nyse"
    return "other"


# ── Stats ──────────────────────────────────────────────────────────────────────

def ev_stats(recs):
    n = len(recs)
    if n < 5:
        return {"n": n, "ev": float("nan"), "p_tgt": float("nan"),
                "p_stop": float("nan"), "total_r": float("nan")}
    p_tgt  = sum(1 for r in recs if r["hit_tgt"]  and not r["hit_stop"]) / n
    p_stop = sum(1 for r in recs if r["hit_stop"] and not r["hit_tgt"])  / n
    nei    = [r["time_exit_r"] for r in recs if not r["hit_tgt"] and not r["hit_stop"]]
    ev_nei = float(np.mean(nei)) if nei else 0.0
    ev     = p_tgt * TGT_SIG - p_stop * STOP_SIG + (1 - p_tgt - p_stop) * ev_nei
    return {"n": n, "ev": ev, "p_tgt": p_tgt, "p_stop": p_stop,
            "total_r": sum(r["outcome_r"] for r in recs)}


def print_stats(label, recs, base_ev=None):
    s = ev_stats(recs)
    if math.isnan(s["ev"]):
        print(f"  {label:<22}  n={s['n']:>4}  (too few)")
        return
    flag = ""
    if base_ev is not None and not math.isnan(base_ev):
        d = s["ev"] - base_ev
        flag = f"  Δ{d:+.4f}"
    print(f"  {label:<22}  n={s['n']:>4}  P(tgt)={s['p_tgt']:.3f}  "
          f"P(stp)={s['p_stop']:.3f}  EV={s['ev']:+.4f}σ{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym", default="M2K", choices=sorted(INSTRUMENTS.keys()))
    args = parser.parse_args()

    path = INSTRUMENTS[args.sym]
    print(f"Loading {path} …", flush=True)
    df1   = load_1min(path)
    bars5 = make_5min_bars(df1)
    print(f"  {len(bars5):,} 5-min bars", flush=True)

    print("Scanning …", flush=True)
    records = scan_5min(bars5)
    attach_pl(records, df1)
    recs = [r for r in records if not math.isnan(r.get("pl_aligned", float("nan")))]
    print(f"  {len(recs):,} signals with PL", flush=True)

    base = ev_stats(recs)

    # ── Session breakdown ──────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  {args.sym}  —  SIGNAL COUNT AND EV BY SESSION")
    print(f"  Baseline (all sessions):  n={base['n']}  EV={base['ev']:+.4f}σ")
    print(f"{'─'*72}")

    for sess in ["nyse", "asia", "euro", "other"]:
        sub = [r for r in recs if session_label(r["utc_frac"]) == sess]
        print_stats(sess, sub, base["ev"])

    # ── PL filter within each session ─────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  PL_aligned ≥ 0 filter within each session")
    print(f"{'─'*72}")
    for sess in ["nyse", "asia", "euro"]:
        sub_all = [r for r in recs if session_label(r["utc_frac"]) == sess]
        sub_pl  = [r for r in sub_all if r["pl_aligned"] >= 0.0]
        sub_hi  = [r for r in sub_all if r["pl_aligned"] >= 0.5]
        s0 = ev_stats(sub_all)
        print(f"\n  {sess.upper()}")
        print_stats("  all signals",     sub_all)
        print_stats("  PL_aligned ≥ 0",  sub_pl,  s0["ev"])
        print_stats("  PL_aligned ≥ 0.5",sub_hi,  s0["ev"])

    # ── PL quartile EV by session ──────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  PL_aligned quartile EV — NYSE vs ASIA")
    print(f"{'─'*72}")
    for sess in ["nyse", "asia"]:
        sub = [r for r in recs if session_label(r["utc_frac"]) == sess]
        if len(sub) < 20:
            print(f"  {sess.upper()}: too few signals ({len(sub)})")
            continue
        vals = np.array([r["pl_aligned"] for r in sub])
        q25, q50, q75 = np.percentile(vals, [25, 50, 75])
        s_all = ev_stats(sub)
        print(f"\n  {sess.upper()}  (n={len(sub)}, EV={s_all['ev']:+.4f}σ)")
        buckets = [
            (f"Q1 (≤{q25:+.2f})",           [r for r in sub if r["pl_aligned"] <= q25]),
            (f"Q2 ({q25:+.2f}–{q50:+.2f})", [r for r in sub if q25 < r["pl_aligned"] <= q50]),
            (f"Q3 ({q50:+.2f}–{q75:+.2f})", [r for r in sub if q50 < r["pl_aligned"] <= q75]),
            (f"Q4 (>{q75:+.2f})",            [r for r in sub if r["pl_aligned"] > q75]),
        ]
        for label, grp in buckets:
            print_stats(f"  {label}", grp, s_all["ev"])

    # ── Year-by-year for best session ─────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Year-by-year:  NYSE-hours only  (all PL  vs  PL ≥ 0)")
    print(f"{'─'*72}")
    nyse_recs = [r for r in recs if session_label(r["utc_frac"]) == "nyse"]
    years = sorted(set(r["year"] for r in nyse_recs))
    print(f"  {'Year':>6}  {'n':>5}  {'EV(all)':>9}  {'n(PL≥0)':>8}  {'EV(PL≥0)':>10}  {'ΔEV':>8}")
    for yr in years:
        yr_all = [r for r in nyse_recs if r["year"] == yr]
        yr_pl  = [r for r in yr_all    if r["pl_aligned"] >= 0.0]
        sa = ev_stats(yr_all); sp = ev_stats(yr_pl)
        if math.isnan(sa["ev"]) or sa["n"] < 5:
            continue
        d    = sp["ev"] - sa["ev"] if not math.isnan(sp["ev"]) else float("nan")
        flag = " ◄" if not math.isnan(d) and d > 0.02 else ""
        ev_pl_str = f"{sp['ev']:>+10.4f}" if not math.isnan(sp["ev"]) else "       n/a"
        d_str     = f"{d:>+8.4f}" if not math.isnan(d) else "     n/a"
        print(f"  {yr:>6}  {sa['n']:>5}  {sa['ev']:>+9.4f}  {sp['n']:>8}  {ev_pl_str}  {d_str}{flag}")

    print()


if __name__ == "__main__":
    main()
