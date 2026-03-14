"""
Maximum Favorable Excursion (MFE) analysis for MES continuation moves.

After a triggering bar (n-sigma, optionally with high volume), scans forward
to measure:
  1. How far the move extends (MFE distribution in σ units)
  2. P(reach profit target T before stop S) for various T levels
  3. Expected value at each profit target (assuming -1σ stop)
  4. Optimal bar at which MFE is typically achieved

Analysed for 1-min and 5-min bars across three trigger subsets:
  - All bars
  - ≥ 2σ single-bar move
  - ≥ 3σ + high volume (≥ 1.5× mean)  — our best-edge subset
"""

import math
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

CACHE_PATH           = "mes_hist_1min.csv"
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TRAILING_BARS        = 100
MAX_FORWARD          = 60    # bars (same for every timeframe)
STOP_SIGMA           = 1.0   # stop at -1σ from entry
TARGETS              = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ── Data / aggregation (reused from continuation_multitf) ────────────────────

def load_1min() -> pd.DataFrame:
    df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


def make_bars(df1: pd.DataFrame, n: int) -> pd.DataFrame:
    if n == 1:
        return df1[["ts", "open", "high", "low", "close", "volume", "gap"]].copy()
    records = []
    i = 0
    while i + n <= len(df1):
        chunk = df1.iloc[i : i + n]
        internal_gaps = chunk["gap"].iloc[1:].values
        if internal_gaps.any():
            i += int(internal_gaps.argmax()) + 1
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
    if not records:
        return pd.DataFrame()
    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=n)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── MFE scan ──────────────────────────────────────────────────────────────────

def scan_mfe(bars: pd.DataFrame,
             min_scaled:    float = 0.0,
             min_vol_ratio: float = 0.0) -> pd.DataFrame:
    """
    For each qualifying triggering bar, scan MAX_FORWARD bars forward and record:
      mfe          – max favorable excursion in σ units (best the trade got)
      mfe_bar      – which bar (1-indexed) the MFE was achieved
      mae          – max adverse excursion in σ units (worst the trade got)
      stopped      – True if -STOP_SIGMA was hit before MAX_FORWARD
      stop_bar     – bar at which stop was hit (NaN if not stopped)
      hit_Xσ       – True if +Xσ target was reached before stop
    """
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    n       = len(bars)
    records = []

    for i in range(TRAILING_BARS, n - MAX_FORWARD):
        if gaps[i - TRAILING_BARS + 1 : i + MAX_FORWARD + 1].any():
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
        if abs(scaled) < min_scaled:
            continue
        if vol_ratio < min_vol_ratio:
            continue

        direction = 1 if scaled > 0 else -1
        entry     = closes[i]

        stop_price = entry * math.exp(-direction * STOP_SIGMA * sigma)
        tgt_prices = [entry * math.exp(direction * t * sigma) for t in TARGETS]

        mfe = 0.0
        mfe_bar = 0
        mae = 0.0
        stopped = False
        stop_bar = None
        hit_target = [False] * len(TARGETS)

        for j in range(i + 1, i + MAX_FORWARD + 1):
            bar_idx = j - i   # 1-indexed

            if direction == 1:
                fav_price = highs[j]
                adv_price = lows[j]
            else:
                fav_price = lows[j]
                adv_price = highs[j]

            # Convert to σ units (signed, positive = favourable)
            fav_sigma = math.log(fav_price / entry) * direction / sigma
            adv_sigma = math.log(adv_price / entry) * direction / sigma  # negative

            # Update MFE / MAE
            if fav_sigma > mfe:
                mfe     = fav_sigma
                mfe_bar = bar_idx
            adverse = -adv_sigma   # positive number = how far against us
            if adverse > mae:
                mae = adverse

            # Check stop (conservative: if both stop and target in same bar, stop wins)
            stop_hit_here = adv_sigma <= -STOP_SIGMA

            for k, tp in enumerate(tgt_prices):
                if not hit_target[k]:
                    tgt_hit_here = (highs[j] >= tp if direction == 1 else lows[j] <= tp)
                    if tgt_hit_here and not stop_hit_here:
                        hit_target[k] = True

            if stop_hit_here:
                stopped  = True
                stop_bar = bar_idx
                break

        rec = {
            "scaled":    abs(scaled),
            "vol_ratio": vol_ratio,
            "mfe":       mfe,
            "mfe_bar":   mfe_bar,
            "mae":       mae,
            "stopped":   stopped,
            "stop_bar":  stop_bar,
        }
        for k, t in enumerate(TARGETS):
            rec[f"hit_{t}"] = hit_target[k]

        records.append(rec)

    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def report(res: pd.DataFrame, title: str, tf_min: int):
    n = len(res)
    if n < 10:
        print(f"\n  [{title}] insufficient data (n={n})")
        return

    print(f"\n{'═'*70}")
    print(f"  {title}  (n={n:,})")
    print(f"{'═'*70}")

    # ── MFE distribution ──────────────────────────────────────────────────────
    mfe = res["mfe"].values
    mae = res["mae"].values
    print(f"\n  MFE distribution (σ units, over {MAX_FORWARD}-bar forward window = "
          f"{MAX_FORWARD * tf_min}min):")
    pcts = [10, 25, 50, 75, 90, 95, 99]
    hdr  = f"  {'':6}" + "".join(f"  p{p:<4}" for p in pcts) + "   mean"
    print(hdr)
    mfe_row = f"  {'MFE':<6}" + "".join(f"  {np.percentile(mfe, p):>5.2f}" for p in pcts) \
              + f"   {mfe.mean():.2f}"
    mae_row = f"  {'MAE':<6}" + "".join(f"  {np.percentile(mae, p):>5.2f}" for p in pcts) \
              + f"   {mae.mean():.2f}"
    print(mfe_row)
    print(mae_row)

    stopped_pct = res["stopped"].mean() * 100
    print(f"\n  Stopped out (-{STOP_SIGMA}σ hit before {MAX_FORWARD} bars): "
          f"{stopped_pct:.1f}%")

    # ── MFE timing: when is it typically achieved? ────────────────────────────
    mfe_bars = res.loc[res["mfe"] > 0, "mfe_bar"]
    if len(mfe_bars):
        print(f"\n  Bar at which MFE is achieved (median={np.median(mfe_bars):.0f}, "
              f"mean={mfe_bars.mean():.1f}, p75={np.percentile(mfe_bars,75):.0f}, "
              f"p90={np.percentile(mfe_bars,90):.0f})  [1 bar = {tf_min}min]")

    # ── Profit target table ───────────────────────────────────────────────────
    print(f"\n  P(hit target before -{STOP_SIGMA}σ stop) & expected value:")
    print(f"  {'Target':>8}  {'P(hit)':>8}  {'P(stop)':>8}  {'EV (σ)':>9}  "
          f"{'Best exit?':>10}")
    print(f"  {'-'*56}")
    best_ev   = -999
    best_tgt  = None
    evs = []
    for t in TARGETS:
        col  = f"hit_{t}"
        if col not in res.columns:
            continue
        p_hit  = res[col].mean()
        p_stop = res["stopped"].mean()
        # EV: if hit, gain = t; if stopped, loss = STOP_SIGMA; if neither, ~0
        p_neither = 1 - p_hit - p_stop
        ev = p_hit * t - p_stop * STOP_SIGMA + p_neither * 0
        evs.append((t, p_hit, p_stop, ev))
        flag = ""
        if ev > best_ev:
            best_ev  = ev
            best_tgt = t
            flag = "  ◄ best EV"
        print(f"  {t:>7.1f}σ  {p_hit:>8.4f}  {p_stop:>8.4f}  {ev:>+9.4f}{flag}")

    # ── MFE histogram (rough) ──────────────────────────────────────────────────
    print(f"\n  MFE histogram (% of trades reaching at least Xσ at any point):")
    checkpoints = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    row1 = "  " + "".join(f"  {x:>5.2f}σ" for x in checkpoints)
    row2 = "  " + "".join(f"  {(mfe >= x).mean()*100:>5.1f}%" for x in checkpoints)
    print(row1)
    print(row2)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df1 = load_1min()

    for tf in [1, 5]:
        bars  = make_bars(df1, tf)
        label = f"{tf}-min"

        subsets = [
            ("all bars",                   0.0, 0.0),
            ("≥2σ",                        2.0, 0.0),
            ("≥3σ + high vol (≥1.5×)",     3.0, 1.5),
        ]

        for desc, min_sc, min_vr in subsets:
            res = scan_mfe(bars, min_scaled=min_sc, min_vol_ratio=min_vr)
            report(res, f"{label} | {desc}", tf)
