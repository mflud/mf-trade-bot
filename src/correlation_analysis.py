"""
Correlation analysis of scaled log returns across MES, MNQ, MYM.

For each 5-min bar, computes:
  scaled_return = log(close/prev_close) / trailing_sigma

Then:
  1. Overall correlation matrix
  2. Rolling 20-bar correlation (how stable is it?)
  3. Spread z-score analysis: when one instrument deviates from the others,
     does price revert? (potential pair-trade / convergence signal)

Usage:
  python src/correlation_analysis.py
"""

import math
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
TF             = 5
TRAILING_BARS  = 20
MIN_VOL_RATIO  = 1.5

INSTRUMENTS = {
    "MES": "mes_hist_1min.csv",
    "MNQ": "mnq_hist_1min.csv",
    "MYM": "mym_hist_1min.csv",
}


# ── Data helpers ────────────────────────────────────────────────────────────────

def load_1min(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    return df.sort_values("ts").reset_index(drop=True)


def make_5min_bars(df1: pd.DataFrame) -> pd.DataFrame:
    """Standard offset-0 5-min bars (:00/:05/:10)."""
    df1 = df1.copy()
    df1["gap"] = df1["ts"].diff() > pd.Timedelta(minutes=2)
    records = []
    i, n = 0, len(df1)
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


def compute_scaled(bars: pd.DataFrame) -> pd.Series:
    """Trailing-sigma-scaled log return for each bar."""
    closes = bars["close"].values
    gaps   = bars["gap"].values
    n      = len(bars)
    scaled = np.full(n, np.nan)

    for i in range(TRAILING_BARS, n):
        if gaps[i - TRAILING_BARS + 1: i + 1].any():
            continue
        trail = np.log(closes[i - TRAILING_BARS + 1: i + 1]
                     / closes[i - TRAILING_BARS:     i    ])
        sigma = np.std(trail, ddof=1)
        if sigma == 0:
            continue
        scaled[i] = math.log(closes[i] / closes[i - 1]) / sigma

    return pd.Series(scaled, index=bars["ts"], name="scaled")


# ── Load and align ──────────────────────────────────────────────────────────────

print("Loading 1-min data …")
all_bars = {}
for sym, path in INSTRUMENTS.items():
    df1  = load_1min(path)
    bars = make_5min_bars(df1)
    sc   = compute_scaled(bars)
    all_bars[sym] = sc
    print(f"  {sym}: {len(bars):,} bars  ({bars['ts'].min().date()} → {bars['ts'].max().date()})")

# Align on common timestamps
df = pd.DataFrame(all_bars).dropna()
print(f"\n  {len(df):,} aligned bars (all three present, no gaps)\n")


# ── 1. Overall correlation matrix ───────────────────────────────────────────────

print("═" * 60)
print("  OVERALL CORRELATION  (scaled log returns, 5-min bars)")
print("═" * 60)
corr = df.corr()
print(f"\n{corr.to_string()}\n")


# ── 2. Rolling correlation stability ───────────────────────────────────────────

print("═" * 60)
print("  ROLLING CORRELATION STABILITY  (20-bar = 100-min window)")
print("═" * 60)
roll = df.rolling(20).corr()

pairs = [("MES","MNQ"), ("MES","MYM"), ("MNQ","MYM")]
for a, b in pairs:
    r = roll.xs(b, level=1)[a].dropna()
    print(f"\n  {a}–{b}:  mean={r.mean():.3f}  std={r.std():.3f}  "
          f"min={r.min():.3f}  p5={r.quantile(0.05):.3f}  "
          f"p95={r.quantile(0.95):.3f}  max={r.max():.3f}")

# By year
print(f"\n  Year-by-year correlation:")
print(f"  {'Year':<6}  {'MES–MNQ':>8}  {'MES–MYM':>8}  {'MNQ–MYM':>8}")
print(f"  {'─'*38}")
for yr, grp in df.groupby(df.index.year):
    if len(grp) < 50:
        continue
    c = grp.corr()
    print(f"  {yr:<6}  {c.loc['MES','MNQ']:>8.3f}  {c.loc['MES','MYM']:>8.3f}  {c.loc['MNQ','MYM']:>8.3f}")


# ── 3. Divergence / spread analysis ────────────────────────────────────────────

print(f"\n{'═'*60}")
print(f"  SPREAD DIVERGENCE ANALYSIS")
print(f"  Question: when instrument A deviates from B+C consensus,")
print(f"  does it revert over the next 1–5 bars?")
print(f"{'═'*60}")

# "Consensus" for each bar = equal-weighted average of all three scaled returns.
# "Deviation" = instrument - consensus.
# We then ask: if |deviation| > threshold, what is the sign of the next bar's
# raw return for that instrument?  Reversion would show: large +deviation →
# next bar negative return.

df2 = df.copy()
df2["consensus"] = df2[["MES","MNQ","MYM"]].mean(axis=1)
for sym in ["MES","MNQ","MYM"]:
    df2[f"dev_{sym}"] = df2[sym] - df2["consensus"]

# z-score the deviations (rolling 100-bar)
for sym in ["MES","MNQ","MYM"]:
    col = f"dev_{sym}"
    roll_mean = df2[col].rolling(100).mean()
    roll_std  = df2[col].rolling(100).std()
    df2[f"zdev_{sym}"] = (df2[col] - roll_mean) / roll_std.replace(0, np.nan)

# For each instrument, load raw returns to measure forward outcome
raw_rets = {}
for sym, path in INSTRUMENTS.items():
    df1  = load_1min(path)
    bars = make_5min_bars(df1)
    bars["ret"] = np.log(bars["close"] / bars["close"].shift(1))
    raw_rets[sym] = bars.set_index("ts")["ret"]

print(f"\n  Reversion test: when z-score of deviation > threshold,")
print(f"  sign of next-bar raw return (negative = reversion)")
print(f"\n  Threshold  {'Sym':<4}  {'n(+dev)':>8}  {'fwd_ret(+)':>10}  "
      f"{'n(-dev)':>8}  {'fwd_ret(-)':>10}  {'reversion?':>10}")
print(f"  {'─'*72}")

for thr in [1.0, 1.5, 2.0, 2.5]:
    for sym in ["MES","MNQ","MYM"]:
        zd = df2[f"zdev_{sym}"].shift(1)   # deviation from PREVIOUS bar
        fwd = raw_rets[sym].reindex(df2.index)

        pos = df2[zd > thr]
        neg = df2[zd < -thr]

        fwd_pos = fwd.reindex(pos.index).mean()
        fwd_neg = fwd.reindex(neg.index).mean()
        n_pos   = len(pos)
        n_neg   = len(neg)

        rev_pos = "YES ✓" if fwd_pos < -1e-5 else ("weak" if fwd_pos < 0 else "NO")
        rev_neg = "YES ✓" if fwd_neg >  1e-5 else ("weak" if fwd_neg > 0 else "NO")
        print(f"  z>{thr:.1f}      {sym:<4}  {n_pos:>8,}  {fwd_pos:>+10.6f}  "
              f"{n_neg:>8,}  {fwd_neg:>+10.6f}  {rev_pos}/{rev_neg}")
    print()


# ── 4. Same-bar lead/lag ────────────────────────────────────────────────────────

print(f"{'═'*60}")
print(f"  LEAD/LAG ANALYSIS  (does one instrument move first?)")
print(f"  Cross-correlation at lags -3 to +3 bars")
print(f"{'═'*60}\n")

for a, b in pairs:
    print(f"  {a} → {b}  (positive lag = {a} leads)")
    print(f"  {'Lag':>5}  {'Corr':>8}")
    for lag in range(-3, 4):
        c = df[a].corr(df[b].shift(lag))
        marker = " ◄" if lag == 0 else ""
        print(f"  {lag:>5}  {c:>8.4f}{marker}")
    print()
