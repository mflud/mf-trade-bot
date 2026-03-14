"""
Strategy simulation: what's the best stop/target structure for a
10-15 minute hold on MES, given all prior findings?

Signal: 5-min bar with |scaled return| >= 3σ AND volume >= 1.5× 100-bar mean.
Entry:  close of triggering bar.
Hold:   max MAX_BARS_HOLD bars (varies from 1 to 3 = 5–15 min).

For each stop/target combination:
  - P(hit profit target before stop, within hold period)
  - P(hit stop before target, within hold period)
  - P(neither hit → time exit, close-to-close P&L)
  - Expected value in σ units
  - Expected value in MES points (using actual σ from each trade)

Also reports:
  - Distribution of σ (in points) for this setup
  - EV sensitivity: how much does signal quality matter?
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
TF                   = 5      # 5-min bars
MAX_BARS_HOLD        = 3      # 3 × 5 min = 15 min
MIN_SCALED           = 3.0
MIN_VOL_RATIO        = 1.5

# Stop / target grid (in σ units)
STOPS   = [0.5, 1.0, 1.5, 2.0]
TARGETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ── Data helpers (same as other scripts) ─────────────────────────────────────

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
    records, i = [], 0
    while i + n <= len(df1):
        chunk = df1.iloc[i : i + n]
        internal = chunk["gap"].iloc[1:].values
        if internal.any():
            i += int(internal.argmax()) + 1
            continue
        records.append({"ts": chunk["ts"].iloc[0], "open": chunk["open"].iloc[0],
                         "high": chunk["high"].max(), "low": chunk["low"].min(),
                         "close": chunk["close"].iloc[-1], "volume": chunk["volume"].sum()})
        i += n
    bars = pd.DataFrame(records)
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=n)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    return bars


# ── Scan all qualifying triggers ──────────────────────────────────────────────

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
        sigma_pts = sigma * entry          # 1σ in MES points

        # Pre-compute all target and stop prices on the grid
        tgt_prices  = {t: entry * math.exp( direction * t * sigma) for t in TARGETS}
        stop_prices = {s: entry * math.exp(-direction * s * sigma) for s in STOPS}

        # Forward scan
        hit_tgt  = {t: None for t in TARGETS}   # bar index when first hit
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

        # Time-exit P&L: close of last bar vs entry
        time_exit_ret = math.log(closes[i + MAX_BARS_HOLD] / entry) * direction / sigma

        records.append({
            "sigma":          sigma,
            "sigma_pts":      sigma_pts,
            "scaled":         abs(scaled),
            "vol_ratio":      vol_ratio,
            "time_exit_ret":  time_exit_ret,   # in σ units, positive = favourable
            **{f"hit_tgt_{t}":  hit_tgt[t]  for t in TARGETS},
            **{f"hit_stop_{s}": hit_stop[s] for s in STOPS},
        })

    return pd.DataFrame(records)


# ── EV grid ───────────────────────────────────────────────────────────────────

def ev_grid(res: pd.DataFrame):
    n = len(res)
    time_ret = res["time_exit_ret"].values

    print(f"\n{'═'*72}")
    print(f"  EV grid: {TF}-min bars, ≥{MIN_SCALED:.0f}σ + vol≥{MIN_VOL_RATIO:.1f}×, "
          f"hold≤{MAX_BARS_HOLD} bars ({MAX_BARS_HOLD*TF}min)   n={n}")
    print(f"{'═'*72}")
    print(f"\n  Columns: target (in σ)  |  Rows: stop (in σ)")
    print(f"\n  ── Expected value (σ units) ──────────────────────────────────────────")

    best_ev  = -999
    best_cfg = None

    # Header
    hdr = f"  {'Stop \\ Target':>14}" + "".join(f"  +{t:.1f}σ" for t in TARGETS)
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    ev_table = {}
    for s in STOPS:
        row = f"  -{s:.1f}σ          "
        for t in TARGETS:
            hit_t_col  = f"hit_tgt_{t}"
            hit_s_col  = f"hit_stop_{s}"
            ht = res[hit_t_col].notna().values
            hs = res[hit_s_col].notna().values
            ht_first = ht & ~(hs & (res[hit_s_col].fillna(999) <= res[hit_t_col].fillna(999)).values)
            hs_first = hs & ~ht_first
            neither  = ~ht_first & ~hs_first
            p_tgt    = ht_first.mean()
            p_stop   = hs_first.mean()
            p_nei    = neither.mean()
            ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
            ev       = p_tgt * t - p_stop * s + p_nei * ev_nei
            ev_table[(s, t)] = (ev, p_tgt, p_stop, p_nei, ev_nei)
            flag = "◄" if ev == max(ev_table.get((s, tt), (-999,))[0] for tt in TARGETS) else " "
            row += f"  {ev:>+5.3f}"
            if ev > best_ev:
                best_ev  = ev
                best_cfg = (s, t, p_tgt, p_stop, p_nei, ev_nei)
        print(row)

    # Highlight best config
    s, t, p_tgt, p_stop, p_nei, ev_nei = best_cfg
    print(f"\n  Best config:  stop={s:.1f}σ  target={t:.1f}σ  →  EV = {best_ev:+.4f}σ")
    print(f"    P(target hit first): {p_tgt:.3f}")
    print(f"    P(stop hit first):   {p_stop:.3f}")
    print(f"    P(time exit):        {p_nei:.3f}  (mean P&L at exit: {ev_nei:+.3f}σ)")

    return ev_table, best_cfg


def detail_report(res: pd.DataFrame, s: float, t: float):
    """Full breakdown for one stop/target combo."""
    hit_t_col = f"hit_tgt_{t}"
    hit_s_col = f"hit_stop_{s}"
    ht = res[hit_t_col].notna().values
    hs = res[hit_s_col].notna().values
    ht_first = ht & ~(hs & (res[hit_s_col].fillna(999) <= res[hit_t_col].fillna(999)).values)
    hs_first = hs & ~ht_first
    neither  = ~ht_first & ~hs_first
    time_ret = res["time_exit_ret"].values
    p_tgt    = ht_first.mean()
    p_stop   = hs_first.mean()
    p_nei    = neither.mean()
    ev_nei   = time_ret[neither].mean() if neither.any() else 0.0
    ev       = p_tgt * t - p_stop * s + p_nei * ev_nei

    sigma_pts = res["sigma_pts"].values
    print(f"\n  Detail:  stop=-{s:.1f}σ  target=+{t:.1f}σ  hold≤{MAX_BARS_HOLD*TF}min")
    print(f"  {'─'*52}")
    print(f"  P(target hit first) : {p_tgt:.4f}  ({p_tgt*100:.1f}%)")
    print(f"  P(stop hit first)   : {p_stop:.4f}  ({p_stop*100:.1f}%)")
    print(f"  P(time exit)        : {p_nei:.4f}  ({p_nei*100:.1f}%)  "
          f"mean exit P&L={ev_nei:+.3f}σ")
    print(f"  EV per trade        : {ev:+.4f}σ")

    print(f"\n  In MES points (1σ distribution):")
    for pct, label in [(25,"p25"),(50,"median"),(75,"p75"),(90,"p90")]:
        sp = np.percentile(sigma_pts, pct)
        print(f"    {label}: 1σ = {sp:.2f} pts  →  "
              f"target={t*sp:.2f} pts  stop={s*sp:.2f} pts  "
              f"EV={ev*sp:+.2f} pts/trade")

    # Timing: at which bar does target get hit?
    if ht_first.any():
        hit_bars = res.loc[ht_first, hit_t_col].values
        print(f"\n  When target is hit (of winning trades):")
        print(f"    bar 1 ({TF}min): {(hit_bars==1).mean()*100:.1f}%  "
              f"bar 2 ({TF*2}min): {(hit_bars==2).mean()*100:.1f}%  "
              f"bar 3 ({TF*3}min): {(hit_bars==3).mean()*100:.1f}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df1  = load_1min()
    bars = make_bars(df1, TF)
    print(f"{len(bars):,} {TF}-min bars")

    res  = scan(bars)
    print(f"{len(res):,} qualifying triggers  "
          f"(≥{MIN_SCALED:.0f}σ, vol≥{MIN_VOL_RATIO:.1f}×)")

    # σ in points
    sp = res["sigma_pts"].values
    print(f"\n  Triggering bar σ in MES points:")
    for pct in [25, 50, 75, 90]:
        print(f"    p{pct}: {np.percentile(sp, pct):.2f} pts")

    _, best_cfg = ev_grid(res)
    s_best, t_best = best_cfg[0], best_cfg[1]
    detail_report(res, s_best, t_best)

    # Also show the symmetric 1σ/1σ case for reference
    print(f"\n{'─'*52}")
    print(f"  Reference: symmetric 1σ stop / 1σ target")
    detail_report(res, 1.0, 1.0)
