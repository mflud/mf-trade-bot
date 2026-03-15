"""
Vol prediction grid search: trailing window × forward window × price type.

Tests which combination of:
  - trailing window (20, 30, 50, 75, 100, 150, 200 bars)
  - forward window  (5, 10, 15, 20, 30 bars)
  - price type      (close, HL/2, OHLC/4, Parkinson, Garman-Klass)

gives the best Pearson r predicting realised forward vol from trailing vol.

Uses non-overlapping forward windows and skips any window spanning a session gap.

Input:  mes_hist_1min.csv  (output of convert_databento.py)
"""

import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────

CACHE_PATH           = "mes_hist_1min.csv"
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
BARS_PER_YEAR        = 252 * 23 * 60

TRAILING_WINDOWS = [20, 30, 50, 75, 100, 150, 200]
FORWARD_WINDOWS  = [5, 10, 15, 20, 30]
PRICE_TYPES      = ["close", "HL", "OHLC", "Parkinson", "GarmanKlass"]


# ── Data ───────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")

    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    print(f"Loaded {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")
    return df


# ── Vol estimators ──────────────────────────────────────────────────────────────

def _ann(v: float) -> float:
    return v * math.sqrt(BARS_PER_YEAR)


def vol_close(opens, highs, lows, closes) -> float:
    """Close-to-close log returns."""
    r = np.log(closes[1:] / closes[:-1])
    return _ann(float(np.std(r, ddof=1))) if len(r) >= 2 else float("nan")


def vol_hl(opens, highs, lows, closes) -> float:
    """HL/2 midpoint log returns."""
    mid = (highs + lows) / 2.0
    r = np.log(mid[1:] / mid[:-1])
    return _ann(float(np.std(r, ddof=1))) if len(r) >= 2 else float("nan")


def vol_ohlc(opens, highs, lows, closes) -> float:
    """OHLC/4 (typical + open) log returns."""
    avg = (opens + highs + lows + closes) / 4.0
    r = np.log(avg[1:] / avg[:-1])
    return _ann(float(np.std(r, ddof=1))) if len(r) >= 2 else float("nan")


def vol_parkinson(opens, highs, lows, closes) -> float:
    """Parkinson (high-low range) estimator, per-bar then averaged."""
    # σ² = 1/(4·n·ln2) · Σ ln(H/L)²
    ln_hl = np.log(highs / lows)
    var_per_bar = (ln_hl ** 2) / (4.0 * math.log(2))
    return _ann(math.sqrt(float(np.mean(var_per_bar))))


def vol_garman_klass(opens, highs, lows, closes) -> float:
    """Garman-Klass estimator."""
    # σ² = Σ [0.5·(ln H/L)² − (2·ln2−1)·(ln C/O)²]
    ln_hl = np.log(highs / lows)
    ln_co = np.log(closes / opens)
    gk = 0.5 * ln_hl ** 2 - (2 * math.log(2) - 1) * ln_co ** 2
    var = float(np.mean(gk))
    if var <= 0:
        return float("nan")
    return _ann(math.sqrt(var))


VOL_FN = {
    "close":      vol_close,
    "HL":         vol_hl,
    "OHLC":       vol_ohlc,
    "Parkinson":  vol_parkinson,
    "GarmanKlass": vol_garman_klass,
}


# ── Grid runner ────────────────────────────────────────────────────────────────

def run_one(
    opens, highs, lows, closes, gaps,
    trailing: int, forward: int, price_type: str,
) -> tuple[float, float, int]:
    """Return (pearson_r, r_squared, n)."""
    fn = VOL_FN[price_type]
    n  = len(closes)
    tv_list, fv_list = [], []

    i = trailing
    while i + forward < n:
        ts  = slice(i - trailing, i + 1)
        fwd = slice(i + 1, i + forward + 1)

        if gaps[ts].any() or gaps[fwd].any():
            i += 1
            continue

        tv = fn(opens[ts], highs[ts], lows[ts], closes[ts])
        fv = fn(opens[fwd], highs[fwd], lows[fwd], closes[fwd])

        if not (math.isnan(tv) or math.isnan(fv)):
            tv_list.append(tv)
            fv_list.append(fv)

        i += forward  # non-overlapping stride

    if len(tv_list) < 10:
        return float("nan"), float("nan"), len(tv_list)

    r, _ = stats.pearsonr(tv_list, fv_list)
    return r, r ** 2, len(tv_list)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()

    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    gaps   = df["gap"].values

    total = len(TRAILING_WINDOWS) * len(FORWARD_WINDOWS) * len(PRICE_TYPES)
    done  = 0
    results = []

    print(f"\nRunning {total} combinations …\n")

    for fw in FORWARD_WINDOWS:
        for tw in TRAILING_WINDOWS:
            for pt in PRICE_TYPES:
                r, r2, n = run_one(opens, highs, lows, closes, gaps, tw, fw, pt)
                results.append({"fwd": fw, "trail": tw, "price": pt, "r": r, "r2": r2, "n": n})
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  {done}/{total}  last: trail={tw} fwd={fw} {pt}  r={r:.4f}  n={n:,}")

    res = pd.DataFrame(results).sort_values("r", ascending=False)

    # ── Summary tables ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  TOP 15 COMBINATIONS (by Pearson r)")
    print("=" * 70)
    print(f"  {'trail':>6}  {'fwd':>4}  {'price':<12}  {'r':>7}  {'R²':>7}  {'n':>7}")
    print("  " + "-" * 55)
    for _, row in res.head(15).iterrows():
        print(f"  {int(row.trail):>6}  {int(row.fwd):>4}  {row.price:<12}  "
              f"{row.r:>7.4f}  {row.r2*100:>6.1f}%  {int(row.n):>7,}")

    # ── Heatmap: best price type per (trail, fwd) ──────────────────────────────

    print("\n" + "=" * 70)
    print("  BEST PEARSON r BY (trailing, forward)  [price type in parens]")
    print("=" * 70)

    best = (
        res.groupby(["trail", "fwd"])
           .apply(lambda g: g.loc[g["r"].idxmax()])
           .reset_index(drop=True)
    )

    header = "  trail \\ fwd" + "".join(f"  {fw:>5}" for fw in FORWARD_WINDOWS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tw in TRAILING_WINDOWS:
        row_parts = []
        for fw in FORWARD_WINDOWS:
            sub = best[(best.trail == tw) & (best.fwd == fw)]
            if sub.empty:
                row_parts.append("    --")
            else:
                row_parts.append(f"  {sub.iloc[0].r:.3f}")
        print(f"  {tw:>5}" + "".join(row_parts))

    # ── Heatmap by price type (averaged over all trail/fwd) ───────────────────

    print("\n" + "=" * 70)
    print("  MEAN PEARSON r BY PRICE TYPE  (averaged over all trail×fwd)")
    print("=" * 70)
    for pt in PRICE_TYPES:
        sub = res[res.price == pt]
        print(f"  {pt:<14}  mean r = {sub.r.mean():.4f}   "
              f"best r = {sub.r.max():.4f}  "
              f"(trail={int(sub.loc[sub.r.idxmax(), 'trail'])}, "
              f"fwd={int(sub.loc[sub.r.idxmax(), 'fwd'])})")

    # ── Best forward window per trailing window ────────────────────────────────

    print("\n" + "=" * 70)
    print("  BEST r PER TRAILING WINDOW  (maximised over fwd + price)")
    print("=" * 70)
    for tw in TRAILING_WINDOWS:
        sub  = res[res.trail == tw]
        best_row = sub.loc[sub.r.idxmax()]
        print(f"  trail={tw:>3}  best r={best_row.r:.4f}  "
              f"fwd={int(best_row.fwd):>2}  {best_row.price}")

    # ── Overall winner ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    winner = res.iloc[0]
    print(f"  OVERALL BEST:  trail={int(winner.trail)}  fwd={int(winner.fwd)}  "
          f"price={winner.price}  r={winner.r:.4f}  R²={winner.r2*100:.1f}%  n={int(winner.n):,}")
    print("=" * 70)
