"""
Backtest: Time-Decaying Profit Target.

Tests whether linearly decaying the take-profit toward entry (starting at
decay_start minutes, reaching entry/breakeven at decay_end minutes) improves
EV vs the current flat-target + hard time exit.

Evaluated for both:
  1. Momentum (3σ continuation) signal — production parameters
  2. ORB (15-min wide opening range breakout, morning window) — production parameters

Decay logic at each elapsed minute t:
  frac         = clamp((t - decay_start) / (decay_end - decay_start), 0, 1)
  eff_tgt_dist = original_tgt_dist * (1 - frac)     # → 0 at decay_end
  eff_tgt_px   = entry ± eff_tgt_dist

At decay_end the target equals entry (breakeven). Stop is unchanged throughout.

Usage:
  python src/backtest_decay_target.py             # MES
  python src/backtest_decay_target.py --sym M2K
  python src/backtest_decay_target.py --sym MYM
"""

import argparse
import math
import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# ── Shared ────────────────────────────────────────────────────────────────────
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

# ── Momentum signal (must match production) ───────────────────────────────────
TF            = 5
TRAILING_BARS = 20
HOLD_MIN      = 25
MOM_TGT_SIG   = 3.0
MOM_STOP_SIG  = 2.0
MIN_SCALED    = 3.0
MAX_SCALED    = 99.0
MIN_VOL_RATIO = 1.5
CSR_THRESHOLD = 1.5
CSR_LOW_WIN   = 4
CSR_NORM_WIN  = 8
GK_LOW_THRESH = 0.08
GK_VOL_BARS   = 20
BLACKOUT_ET   = (8, 0, 9, 0)

# ── ORB (must match production) ───────────────────────────────────────────────
ORB_MIN       = 15
ORB_TGT_SIG   = 2.0
ORB_STOP_SIG  = 2.0
ORB_HOLD_MIN  = 25
ORB_WIN_START = dtime(9, 45)
ORB_WIN_END   = dtime(10, 30)

# ── Decay configs: (decay_start_min, decay_end_min) ───────────────────────────
DECAY_CONFIGS = [
    (5,  15),
    (5,  20),
    (5,  25),
    (10, 15),
    (10, 20),
    (10, 25),
]

INSTRUMENTS = {
    "MES": {"file": "mes_hist_1min.csv", "orb_width_min": 15.25},
    "M2K": {"file": "m2k_hist_1min.csv", "orb_width_min": 14.30},
    "MYM": {"file": "mym_hist_1min.csv", "orb_width_min":  0.0},
}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    return df


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
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


def gk_vol(opens, highs, lows, closes):
    bpy = 252 * 23 * 60 / TF
    vals = [
        0.5 * math.log(h / l) ** 2 - (2 * math.log(2) - 1) * math.log(c / o) ** 2
        for o, h, l, c in zip(opens, highs, lows, closes)
        if min(o, h, l, c) > 0
    ]
    return math.sqrt(max(0.0, float(np.mean(vals))) * bpy) if vals else 0.0


# ── Forward walk ──────────────────────────────────────────────────────────────

def forward_walk(highs, lows, closes, start_idx: int,
                 entry: float, direction: int, sigma_pts: float,
                 tgt_sig: float, stop_sig: float, hold_min: int) -> dict:
    """
    Walk forward hold_min 1-min bars from start_idx.
    Returns outcomes for flat (production baseline) and all DECAY_CONFIGS.
    Each outcome is {"r": float, "outcome": str}.
    Outcomes: "target", "stop", "decay" (decayed-target exit), "time".
    Convention (matching existing backtests): target checked before stop.
    """
    tgt_dist  = tgt_sig  * sigma_pts
    stop_dist = stop_sig * sigma_pts
    tgt_px    = entry + direction * tgt_dist
    stop_px   = entry - direction * stop_dist
    nm        = len(highs)

    all_keys   = ["flat"] + DECAY_CONFIGS
    exit_r     = {k: None for k in all_keys}
    exit_label = {k: None for k in all_keys}

    for step in range(1, hold_min + 1):
        idx = start_idx + step
        if idx >= nm:
            break
        h = highs[idx]
        l = lows[idx]

        for key in all_keys:
            if exit_r[key] is not None:
                continue

            if key == "flat":
                eff_dist = tgt_dist
            else:
                ds, de = key
                frac     = max(0.0, min(1.0, (step - ds) / max(de - ds, 1)))
                eff_dist = tgt_dist * (1.0 - frac)

            eff_tgt_px = entry + direction * eff_dist

            # Target checked first (matches existing convention)
            if direction == 1:
                if h >= eff_tgt_px:
                    r = eff_dist / sigma_pts   # in σ units
                    exit_r[key]     = r
                    exit_label[key] = "target" if r >= tgt_sig - 0.01 else "decay"
                elif l <= stop_px:
                    exit_r[key]     = -stop_sig
                    exit_label[key] = "stop"
            else:
                if l <= eff_tgt_px:
                    r = eff_dist / sigma_pts
                    exit_r[key]     = r
                    exit_label[key] = "target" if r >= tgt_sig - 0.01 else "decay"
                elif h >= stop_px:
                    exit_r[key]     = -stop_sig
                    exit_label[key] = "stop"

        if all(exit_r[k] is not None for k in all_keys):
            break

    # Time exit for any unresolved
    last_idx = min(start_idx + hold_min, nm - 1)
    time_r   = (closes[last_idx] - entry) * direction / sigma_pts if sigma_pts > 0 else 0.0
    for key in all_keys:
        if exit_r[key] is None:
            exit_r[key]     = time_r
            exit_label[key] = "time"

    return {k: {"r": exit_r[k], "outcome": exit_label[k]} for k in all_keys}


# ── Momentum scanner ──────────────────────────────────────────────────────────

def scan_momentum(bars5: pd.DataFrame, df1: pd.DataFrame) -> list[dict]:
    """Detect 3σ signals on 5-min bars; forward-walk on 1-min bars."""
    closes   = bars5["close"].values
    highs    = bars5["high"].values
    lows     = bars5["low"].values
    opens    = bars5["open"].values
    volumes  = bars5["volume"].values
    gaps     = bars5["gap"].values
    ts_pd    = pd.DatetimeIndex(bars5["ts"].values, tz="UTC")

    # 1-min arrays for forward walk
    h1  = df1["high"].values
    l1  = df1["low"].values
    c1  = df1["close"].values
    ts1 = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts1_map = {v: i for i, v in enumerate(ts1)}

    MAX_CSR  = max(CSR_LOW_WIN, CSR_NORM_WIN)
    lookback = max(TRAILING_BARS, MAX_CSR, GK_VOL_BARS) + 1
    n5       = len(bars5)
    records  = []

    for i in range(lookback, n5 - 1):
        if gaps[i - TRAILING_BARS + 1: i + 2].any():
            continue

        trail_rets = np.log(closes[i - TRAILING_BARS + 1: i + 1] /
                            closes[i - TRAILING_BARS:     i    ])
        sigma = trail_rets.std(ddof=1)
        if sigma == 0:
            continue

        bar_ret = math.log(closes[i] / closes[i - 1])
        scaled  = bar_ret / sigma
        if abs(scaled) < MIN_SCALED or abs(scaled) > MAX_SCALED:
            continue

        vols = volumes[i - TRAILING_BARS: i]
        mean_vol = vols[vols >= 10].mean() if (vols >= 10).sum() >= 10 else float("nan")
        if math.isnan(mean_vol) or volumes[i] / mean_vol < MIN_VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1
        gk = gk_vol(opens[i - GK_VOL_BARS: i], highs[i - GK_VOL_BARS: i],
                    lows[i - GK_VOL_BARS: i],  closes[i - GK_VOL_BARS: i])
        csr_win    = CSR_LOW_WIN if gk < GK_LOW_THRESH else CSR_NORM_WIN
        prior_rets = np.log(closes[i - csr_win: i] / closes[i - csr_win - 1: i - 1])
        csr        = float(prior_rets.sum()) / sigma * direction

        bar_et = ts_pd[i].astimezone(ET)
        hm     = (bar_et.hour, bar_et.minute)
        sh, sm, eh, em = BLACKOUT_ET
        if (sh, sm) <= hm < (eh, em) and csr < CSR_THRESHOLD:
            continue
        if csr < CSR_THRESHOLD:
            continue

        entry      = closes[i]
        sigma_pts  = sigma * entry
        sig_ts_ns  = int(ts_pd[i].as_unit("ns").value)
        start_idx  = ts1_map.get(sig_ts_ns)
        if start_idx is None:
            continue

        outcomes = forward_walk(h1, l1, c1, start_idx, entry, direction,
                                sigma_pts, MOM_TGT_SIG, MOM_STOP_SIG, HOLD_MIN)
        rec = {"year": ts_pd[i].year, "direction": direction,
               "sigma_pts": sigma_pts}
        for key, val in outcomes.items():
            rec[f"{key}_r"]       = val["r"]
            rec[f"{key}_outcome"] = val["outcome"]
        records.append(rec)

    return records


# ── ORB scanner ───────────────────────────────────────────────────────────────

def scan_orb(df1: pd.DataFrame, orb_width_min: float) -> list[dict]:
    """Detect wide-ORB morning breakouts; forward-walk with flat and decay targets."""
    ts_et = df1["ts"].dt.tz_convert(ET)
    df    = df1.copy()
    df["ts_et"] = ts_et
    df["date"]  = ts_et.dt.date
    df["t_et"]  = ts_et.dt.time

    # Precompute rolling sigma on RTH bars only
    rth_mask = (df["t_et"] >= dtime(9, 30)) & (df["t_et"] < dtime(16, 0))
    df_rth   = df[rth_mask].copy()
    log_rets = np.log(df_rth["close"] / df_rth["close"].shift(1))
    sigma    = log_rets.rolling(20, min_periods=20).std(ddof=1)
    df_rth["sigma_pts"] = sigma * df_rth["close"]
    df      = df.join(df_rth[["sigma_pts"]], how="left")

    h1  = df1["high"].values
    l1  = df1["low"].values
    c1  = df1["close"].values
    ts1 = df1["ts"].values.astype("datetime64[ns]").astype(np.int64)
    ts1_map = {v: i for i, v in enumerate(ts1)}

    records = []
    for date, session in df.groupby("date"):
        session = session.reset_index(drop=True)
        rth_bars = session[
            (session["t_et"] >= dtime(9, 30)) &
            (session["t_et"] <  dtime(16, 0))
        ]
        if len(rth_bars) < ORB_MIN + 2:
            continue

        orb_bars  = rth_bars[rth_bars.index < rth_bars.index[ORB_MIN]]
        post_bars = rth_bars.iloc[ORB_MIN:]

        if len(orb_bars) < ORB_MIN or len(post_bars) < 2:
            continue

        orb_high  = orb_bars["high"].max()
        orb_low   = orb_bars["low"].min()
        orb_width = orb_high - orb_low
        if orb_width <= 0 or (orb_width_min > 0 and orb_width < orb_width_min):
            continue

        # Morning window only (matching production)
        morning = post_bars[
            (post_bars["t_et"] >= ORB_WIN_START) &
            (post_bars["t_et"] <  ORB_WIN_END)
        ]
        if morning.empty:
            continue

        # First close above ORB high (LONG only, matching production)
        triggered = False
        for _, bar in morning.iterrows():
            if bar["close"] > orb_high:
                triggered  = True
                entry      = bar["close"]
                sigma_pts  = bar["sigma_pts"]
                if math.isnan(sigma_pts) or sigma_pts <= 0:
                    break
                sig_ts_ns  = int(bar["ts"].as_unit("ns").value)
                start_idx  = ts1_map.get(sig_ts_ns)
                if start_idx is None:
                    break

                outcomes = forward_walk(h1, l1, c1, start_idx, entry, 1,
                                        sigma_pts, ORB_TGT_SIG, ORB_STOP_SIG,
                                        ORB_HOLD_MIN)
                rec = {"year": date.year, "direction": 1,
                       "sigma_pts": sigma_pts, "orb_width": orb_width}
                for key, val in outcomes.items():
                    rec[f"{key}_r"]       = val["r"]
                    rec[f"{key}_outcome"] = val["outcome"]
                records.append(rec)
                break

    return records


# ── Reporting ─────────────────────────────────────────────────────────────────

def outcome_stats(records: list[dict], key: str, tgt_sig: float, stop_sig: float) -> dict:
    """Compute stats for a given config key across all records."""
    r_col  = f"{key}_r"
    o_col  = f"{key}_outcome"
    rs     = [rec[r_col] for rec in records]
    labels = [rec[o_col] for rec in records]
    n      = len(rs)
    if n == 0:
        return {}
    p_tgt   = sum(1 for l in labels if l == "target") / n
    p_stop  = sum(1 for l in labels if l == "stop")   / n
    p_decay = sum(1 for l in labels if l == "decay")  / n
    p_time  = sum(1 for l in labels if l == "time")   / n
    ev      = float(np.mean(rs))
    total_r = float(np.sum(rs))

    decay_rs = [r for r, l in zip(rs, labels) if l == "decay"]
    mean_decay_r = float(np.mean(decay_rs)) if decay_rs else float("nan")

    return {"n": n, "ev": ev, "total_r": total_r,
            "p_tgt": p_tgt, "p_stop": p_stop,
            "p_decay": p_decay, "p_time": p_time,
            "mean_decay_r": mean_decay_r}


def print_comparison(records: list[dict], tgt_sig: float, stop_sig: float,
                     title: str):
    if not records:
        print(f"  No records for {title}.")
        return

    base = outcome_stats(records, "flat", tgt_sig, stop_sig)
    print(f"\n{'═' * 90}")
    print(f"  {title}")
    print(f"  n={base['n']:,}  baseline EV={base['ev']:+.4f}σ  total R={base['total_r']:+.1f}R")
    print(f"{'─' * 90}")
    print(f"  {'Config':<14} {'P(full tgt)':>11} {'P(stop)':>8} "
          f"{'P(decay)':>9} {'P(time)':>8} "
          f"{'EV':>9} {'ΔEV':>8} "
          f"{'TotR':>8} {'ΔR':>7} {'AvgDecayR':>10}")
    print(f"  {'-'*14} {'-'*11} {'-'*8} {'-'*9} {'-'*8} "
          f"{'-'*9} {'-'*8} {'-'*8} {'-'*7} {'-'*10}")

    # Flat baseline
    s = base
    print(f"  {'flat (base)':<14} {s['p_tgt']:>11.3f} {s['p_stop']:>8.3f} "
          f"{'—':>9} {s['p_time']:>8.3f} "
          f"{s['ev']:>+9.4f} {'—':>8} "
          f"{s['total_r']:>+8.1f} {'—':>7} {'—':>10}")

    best_ev  = base["ev"]
    best_key = "flat"
    for cfg in DECAY_CONFIGS:
        key = cfg
        s   = outcome_stats(records, key, tgt_sig, stop_sig)
        if not s:
            continue
        d_ev = s["ev"] - base["ev"]
        d_r  = s["total_r"] - base["total_r"]
        flag = " ◄" if d_ev > 0.005 else ""
        mdr  = f"{s['mean_decay_r']:+.3f}" if not math.isnan(s["mean_decay_r"]) else "—"
        label = f"{cfg[0]}→{cfg[1]}m"
        print(f"  {label:<14} {s['p_tgt']:>11.3f} {s['p_stop']:>8.3f} "
              f"{s['p_decay']:>9.3f} {s['p_time']:>8.3f} "
              f"{s['ev']:>+9.4f} {d_ev:>+8.4f} "
              f"{s['total_r']:>+8.1f} {d_r:>+7.1f} {mdr:>10}{flag}")
        if s["ev"] > best_ev:
            best_ev  = s["ev"]
            best_key = key

    # Year-by-year for best config vs flat
    if best_key != "flat":
        print(f"\n  Year-by-year: flat vs best ({best_key[0]}→{best_key[1]}m):")
        print(f"  {'Year':>6} {'n':>5}  "
              f"{'EV(flat)':>10} {'TotR(flat)':>11}  "
              f"{'EV(decay)':>10} {'TotR(decay)':>12} {'ΔEV':>8}")
        years = sorted(set(rec["year"] for rec in records))
        for yr in years:
            yr_recs = [r for r in records if r["year"] == yr]
            if len(yr_recs) < 5:
                continue
            sf = outcome_stats(yr_recs, "flat",    tgt_sig, stop_sig)
            sd = outcome_stats(yr_recs, best_key,  tgt_sig, stop_sig)
            d  = sd["ev"] - sf["ev"]
            flag = " ◄" if d > 0 else ""
            print(f"  {yr:>6} {sf['n']:>5}  "
                  f"{sf['ev']:>+10.4f} {sf['total_r']:>+11.1f}  "
                  f"{sd['ev']:>+10.4f} {sd['total_r']:>+12.1f} {d:>+8.4f}{flag}")

    # Decay exit R distribution for best decay config
    if best_key != "flat":
        decay_rs = [rec[f"{best_key}_r"] for rec in records
                    if rec[f"{best_key}_outcome"] == "decay"]
        if decay_rs:
            arr = np.array(decay_rs)
            print(f"\n  Decay exit distribution (config {best_key[0]}→{best_key[1]}m, "
                  f"n={len(arr)} decay exits of {len(records)} total):")
            print(f"    mean={arr.mean():+.3f}σ  "
                  f"p25={np.percentile(arr,25):+.3f}σ  "
                  f"median={np.median(arr):+.3f}σ  "
                  f"p75={np.percentile(arr,75):+.3f}σ")
            pct_pos  = (arr > 0.05).mean() * 100
            pct_half = (arr > tgt_sig / 2).mean() * 100
            pct_be   = (np.abs(arr) <= 0.05).mean() * 100
            pct_neg  = (arr < -0.05).mean() * 100
            print(f"    > half target ({tgt_sig/2:.1f}σ): {pct_half:.0f}%  "
                  f"  positive (>0): {pct_pos:.0f}%  "
                  f"  ~breakeven: {pct_be:.0f}%  "
                  f"  negative: {pct_neg:.0f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Decay-target backtest")
    parser.add_argument("--sym", default="MES", choices=sorted(INSTRUMENTS.keys()))
    args = parser.parse_args()

    cfg  = INSTRUMENTS[args.sym]
    path = cfg["file"]

    print(f"Loading {path} …", flush=True)
    df1   = load_1min(path)
    bars5 = make_5min_bars(df1)
    print(f"  1-min bars: {len(df1):,}   5-min bars: {len(bars5):,}", flush=True)

    # ── Momentum ──────────────────────────────────────────────────────────────
    print("\nScanning momentum signals …", flush=True)
    mom_records = scan_momentum(bars5, df1)
    print(f"  {len(mom_records):,} signals", flush=True)

    print_comparison(
        mom_records, MOM_TGT_SIG, MOM_STOP_SIG,
        f"MOMENTUM SIGNAL — {args.sym}  "
        f"(target={MOM_TGT_SIG}σ  stop={MOM_STOP_SIG}σ  hold={HOLD_MIN}min)",
    )

    # ── ORB ───────────────────────────────────────────────────────────────────
    print("\nScanning ORB signals …", flush=True)
    orb_records = scan_orb(df1, cfg["orb_width_min"])
    print(f"  {len(orb_records):,} ORB signals", flush=True)

    orb_width_label = (f"wide >{cfg['orb_width_min']:.1f}pts"
                       if cfg["orb_width_min"] > 0 else "all widths")
    print_comparison(
        orb_records, ORB_TGT_SIG, ORB_STOP_SIG,
        f"ORB SIGNAL — {args.sym}  "
        f"({orb_width_label}  morning 9:45–10:30 ET  "
        f"target={ORB_TGT_SIG}σ  stop={ORB_STOP_SIG}σ  hold={ORB_HOLD_MIN}min)",
    )
    print()


if __name__ == "__main__":
    main()
