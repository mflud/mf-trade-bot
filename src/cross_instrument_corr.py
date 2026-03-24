"""
Cross-instrument correlation analysis: MES vs M2K
- 5-min returns correlation
- Signal co-occurrence (how often both instruments signal same direction)
- Diversification benefit: combined position vs individual
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA = {
    "MES": Path("mes_hist_1min.csv"),
    "M2K": Path("m2k_hist_1min.csv"),
}

TF = 5  # resample to 5-min bars


def load_csv(sym: str) -> pd.DataFrame:
    df = pd.read_csv(DATA[sym], parse_dates=["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    else:
        df["ts"] = df["ts"].dt.tz_convert("UTC")
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def make_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-min OHLCV to 5-min bars, dropping CME gap."""
    df = df.copy()
    # Drop CME gap: 21:00–22:00 UTC
    hour = df["ts"].dt.hour
    df = df[~((hour >= 21) & (hour < 22))].reset_index(drop=True)

    df["ts5"] = df["ts"].dt.floor(f"{TF}min")
    g = df.groupby("ts5")
    bars = g.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    # Drop bars that don't have all TF minutes (e.g. session open/close edge)
    counts = g.size().rename("cnt")
    bars = bars[counts == TF].reset_index()
    bars = bars.rename(columns={"ts5": "ts"})
    if bars.empty:
        return bars

    # Mark gap bars (session breaks) so we don't compute returns across them
    bars["gap"] = bars["ts"].diff() != pd.Timedelta(minutes=TF)
    bars.iloc[0, bars.columns.get_loc("gap")] = True
    bars["ret"] = np.log(bars["close"] / bars["close"].shift(1))
    bars.loc[bars["gap"], "ret"] = np.nan
    return bars


def session_label(ts: pd.Series) -> pd.Series:
    """Map UTC hour to session string."""
    h = ts.dt.hour + ts.dt.minute / 60
    sess = pd.Series("other", index=ts.index)
    sess[(h >= 7.0) & (h < 13.5)] = "euro"
    sess[(h >= 13.5) & (h < 20.0)] = "nyse"
    sess[(h >= 0.0) & (h < 6.5)] = "asia"
    return sess


def main():
    print("Loading data...")
    mes = load_csv("MES")
    m2k = load_csv("M2K")

    print(f"MES: {len(mes):,} 1-min bars  {mes['ts'].min().date()} – {mes['ts'].max().date()}")
    print(f"M2K: {len(m2k):,} 1-min bars  {m2k['ts'].min().date()} – {m2k['ts'].max().date()}")

    # Common date range
    start = max(mes["ts"].min(), m2k["ts"].min())
    end = min(mes["ts"].max(), m2k["ts"].max())
    mes = mes[(mes["ts"] >= start) & (mes["ts"] <= end)]
    m2k = m2k[(m2k["ts"] >= start) & (m2k["ts"] <= end)]
    print(f"\nCommon window: {start.date()} – {end.date()}")

    print("Resampling to 5-min...")
    mes5 = make_5min(mes)
    m2k5 = make_5min(m2k)
    print(f"MES 5-min bars: {len(mes5):,}   M2K 5-min bars: {len(m2k5):,}")

    # Align on timestamp
    merged = mes5[["ts", "ret"]].merge(
        m2k5[["ts", "ret"]], on="ts", suffixes=("_mes", "_m2k")
    ).dropna()
    print(f"Aligned bars (both non-NaN): {len(merged):,}")

    # ── Overall correlation ──────────────────────────────────────────────────
    r_all = merged["ret_mes"].corr(merged["ret_m2k"])
    print(f"\n{'='*60}")
    print(f"Overall 5-min return correlation  r = {r_all:.4f}")
    print(f"{'='*60}")

    # ── By session ───────────────────────────────────────────────────────────
    merged["session"] = session_label(merged["ts"])
    print("\nCorrelation by session:")
    print(f"  {'Session':<8} {'n':>7}  {'r':>7}")
    for sess in ["nyse", "euro", "asia", "other"]:
        s = merged[merged["session"] == sess]
        if len(s) < 30:
            continue
        r = s["ret_mes"].corr(s["ret_m2k"])
        print(f"  {sess:<8} {len(s):>7,}  {r:>7.4f}")

    # ── Rolling 90-day correlation ────────────────────────────────────────────
    merged_sorted = merged.sort_values("ts").reset_index(drop=True)
    # Use ~78 bars/day * 90 ≈ 7020 bars rolling window
    window = 7020
    if len(merged_sorted) > window:
        roll_r = merged_sorted["ret_mes"].rolling(window).corr(merged_sorted["ret_m2k"])
        print(f"\nRolling 90-day correlation:")
        print(f"  Min: {roll_r.min():.4f}  Max: {roll_r.max():.4f}  Mean: {roll_r.mean():.4f}")
        # By calendar year
        merged_sorted["year"] = merged_sorted["ts"].dt.year
        print("\nMean correlation by year:")
        print(f"  {'Year':<6} {'n':>7}  {'r_mean':>8}")
        for yr, g in merged_sorted.groupby("year"):
            r_yr = g["ret_mes"].corr(g["ret_m2k"])
            print(f"  {yr:<6} {len(g):>7,}  {r_yr:>8.4f}")

    # ── Diversification: combined vs individual ──────────────────────────────
    # Equal-weight combined return
    merged["ret_combo"] = 0.5 * merged["ret_mes"] + 0.5 * merged["ret_m2k"]
    vol_mes = merged["ret_mes"].std()
    vol_m2k = merged["ret_m2k"].std()
    vol_combo = merged["ret_combo"].std()
    vol_avg = 0.5 * (vol_mes + vol_m2k)
    diversification_ratio = vol_combo / vol_avg

    print(f"\n{'='*60}")
    print("Diversification analysis (equal-weight 50/50 combo):")
    print(f"  σ_MES   = {vol_mes:.6f}")
    print(f"  σ_M2K   = {vol_m2k:.6f}")
    print(f"  σ_combo = {vol_combo:.6f}")
    print(f"  Diversification ratio (combo/avg): {diversification_ratio:.4f}")
    print(f"  Variance reduction: {(1 - diversification_ratio**2 / ((0.5)**2 * (1 + 1 + 2*r_all))):.2%}  (theoretical: {1 - (0.5*(1+r_all))**0.5:.2%})")

    # ── Signal co-occurrence ─────────────────────────────────────────────────
    # Load signals from backtest PL sizing results if available
    # Otherwise just show return direction agreement
    merged["dir_mes"] = np.sign(merged["ret_mes"])
    merged["dir_m2k"] = np.sign(merged["ret_m2k"])
    merged_nonzero = merged[(merged["dir_mes"] != 0) & (merged["dir_m2k"] != 0)]
    agreement = (merged_nonzero["dir_mes"] == merged_nonzero["dir_m2k"]).mean()
    print(f"\n{'='*60}")
    print(f"Return direction agreement: {agreement:.2%}  (random = 50%)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Period: {start.date()} – {end.date()}")
    print(f"Overall 5-min correlation: {r_all:.4f}")
    print(f"Implication: MES/M2K are {'highly' if r_all > 0.8 else 'moderately' if r_all > 0.6 else 'weakly'} correlated")
    if r_all > 0.7:
        print("  => Signals that co-occur in both instruments add limited diversification")
        print("  => Recommend NOT double-sizing when both fire simultaneously")
    else:
        print("  => Meaningful diversification benefit from trading both instruments")


if __name__ == "__main__":
    main()
