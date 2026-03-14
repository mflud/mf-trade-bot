"""
Multi-timeframe continuation analysis for MES.

Aggregates 1-min bars into 2, 3, 4, 5-min bars and repeats the
continuation edge analysis for each timeframe.

For each N-min bar that is an n-sigma event:
  - P(next bar closes in same direction)          — simple close-to-close
  - P(price continues ≥1σ before reverting ≥1σ)  — scanning 3 bars forward

Trailing vol: 100 bars of that timeframe.
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
MAX_FORWARD          = 3     # bars of the target timeframe


# ── Data loading ──────────────────────────────────────────────────────────────

def load_1min() -> pd.DataFrame:
    df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    return df


# ── Aggregation ───────────────────────────────────────────────────────────────

def make_bars(df1: pd.DataFrame, n: int) -> pd.DataFrame:
    """Aggregate 1-min bars into complete N-min bars with no internal gaps."""
    if n == 1:
        return df1[["ts", "open", "high", "low", "close", "volume", "gap"]].copy()

    records = []
    i = 0
    while i + n <= len(df1):
        chunk = df1.iloc[i : i + n]
        # If any gap inside the chunk, advance past it and restart
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


# ── Scan ─────────────────────────────────────────────────────────────────────

def scan(bars: pd.DataFrame) -> pd.DataFrame:
    closes  = bars["close"].values
    highs   = bars["high"].values
    lows    = bars["low"].values
    volumes = bars["volume"].values
    gaps    = bars["gap"].values
    n       = len(bars)
    records = []

    for i in range(TRAILING_BARS, n - MAX_FORWARD):
        # Require no gap in trailing or forward window
        if gaps[i - TRAILING_BARS + 1 : i + MAX_FORWARD + 1].any():
            continue

        # Trailing vol
        trail_rets = np.log(closes[i - TRAILING_BARS + 1 : i + 1]
                          / closes[i - TRAILING_BARS     : i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        # Relative volume
        mean_vol  = volumes[i - TRAILING_BARS : i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")

        # Triggering bar
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma
        direction = 1 if scaled > 0 else -1

        # Simple: does next bar close in same direction?
        next_ret      = math.log(closes[i + 1] / closes[i])
        same_dir_next = int((next_ret * direction) > 0)

        # Target-based: ±1σ scanning MAX_FORWARD bars
        cont_target = closes[i] * math.exp( direction * sigma)
        rev_target  = closes[i] * math.exp(-direction * sigma)
        outcome = "timeout"
        for j in range(i + 1, i + MAX_FORWARD + 1):
            cont_hit = highs[j] >= cont_target if direction == 1 else lows[j] <= cont_target
            rev_hit  = lows[j]  <= rev_target  if direction == 1 else highs[j] >= rev_target
            if cont_hit:
                outcome = "continuation"
                break
            if rev_hit:
                outcome = "reversion"
                break

        records.append({
            "sigma":         sigma,
            "scaled":        abs(scaled),
            "vol_ratio":     vol_ratio,
            "same_dir_next": same_dir_next,
            "outcome":       outcome,
        })

    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def p_cont(grp: pd.DataFrame) -> tuple[float, float, int, int]:
    """Returns (P(same_dir_next), P(cont_target), n, n_timeout)."""
    n       = len(grp)
    p_dir   = grp["same_dir_next"].mean() if n else float("nan")
    valid   = grp[grp["outcome"] != "timeout"]
    n_to    = (grp["outcome"] == "timeout").sum()
    p_tgt   = (valid["outcome"] == "continuation").mean() if len(valid) else float("nan")
    return p_dir, p_tgt, n, n_to


def report(res: pd.DataFrame, tf_label: str):
    p_dir, p_tgt, n, n_to = p_cont(res)
    print(f"\n{'═'*70}")
    print(f"  Timeframe: {tf_label}   |   n={n:,}   "
          f"(timeout={n_to:,} = {n_to/n*100:.0f}%)")
    print(f"  Overall:  P(same dir next bar)={p_dir:.4f}   "
          f"P(cont ≥1σ, 3-bar scan)={p_tgt:.4f}")
    print(f"{'═'*70}")

    bins   = [0, 1, 2, 3, 4, 5, np.inf]
    labels = ["<1σ", "1–2σ", "2–3σ", "3–4σ", "4–5σ", "5σ+"]
    res2   = res.copy()
    res2["size"] = pd.cut(res2["scaled"], bins=bins, labels=labels)

    print(f"\n  By move size:")
    print(f"  {'Bucket':<8} {'n':>6}  {'P(same dir)':>12}  {'P(cont≥1σ)':>11}  {'timeout':>8}")
    print(f"  {'-'*52}")
    for lbl, grp in res2.groupby("size", observed=True):
        if len(grp) < 10:
            continue
        pd2, pt2, nn, nt2 = p_cont(grp)
        flag = "  ◄" if pt2 >= 0.55 or pd2 >= 0.55 else ""
        print(f"  {str(lbl):<8} {nn:>6,}  {pd2:>12.4f}  {pt2:>11.4f}  {nt2:>8,}{flag}")

    vbins   = [0, 0.75, 1.5, np.inf]
    vlabels = ["low (<0.75×)", "normal (0.75–1.5×)", "high (1.5×+)"]
    res2["vtier"] = pd.cut(res2["vol_ratio"], bins=vbins, labels=vlabels)

    print(f"\n  By relative volume:")
    print(f"  {'Vol tier':<22} {'n':>6}  {'P(same dir)':>12}  {'P(cont≥1σ)':>11}  {'timeout':>8}")
    print(f"  {'-'*64}")
    for lbl, grp in res2.groupby("vtier", observed=True):
        if len(grp) < 10:
            continue
        pd2, pt2, nn, nt2 = p_cont(grp)
        flag = "  ◄" if pt2 >= 0.55 or pd2 >= 0.55 else ""
        print(f"  {str(lbl):<22} {nn:>6,}  {pd2:>12.4f}  {pt2:>11.4f}  {nt2:>8,}{flag}")

    # 2D: size ≥ 2σ × volume
    res3 = res2[res2["scaled"] >= 2].copy()
    if len(res3) < 20:
        return
    res3["stier"] = pd.cut(res3["scaled"], bins=[2, 3, 4, np.inf],
                           labels=["2–3σ", "3–4σ", "4σ+"])
    print(f"\n  P(cont ≥1σ): move size × volume  [|bar| ≥ 2σ]")
    print(f"  {'Size \\ Vol':<12}  {'low (<0.75×)':>16}  {'normal':>16}  {'high (1.5×+)':>16}")
    print(f"  {'-'*66}")
    for slbl, sg in res3.groupby("stier", observed=True):
        row = f"  {str(slbl):<12}"
        for vlbl in vlabels:
            cell = sg[sg["vtier"] == vlbl]
            _, pt, nn, _ = p_cont(cell)
            row += f"  {pt:.3f} (n={nn:>4})" if nn >= 10 else f"  {'—':>16}"
        print(row)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df1 = load_1min()
    print(f"\n{'─'*70}")
    print(f"  Settings: trailing={TRAILING_BARS} bars, forward={MAX_FORWARD} bars, "
          f"targets=±1σ")
    print(f"{'─'*70}")

    for tf in [1, 2, 3, 4, 5]:
        bars = make_bars(df1, tf)
        if len(bars) < TRAILING_BARS + MAX_FORWARD + 10:
            print(f"\n[{tf}min] insufficient bars ({len(bars)}), skipping.")
            continue
        label = f"{tf}-min  ({len(bars):,} bars, "  \
                f"trailing={TRAILING_BARS*tf}min, forward={MAX_FORWARD*tf}min)"
        res  = scan(bars)
        report(res, label)
