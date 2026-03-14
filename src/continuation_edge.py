"""
Continuation probability analysis for MES 1-min bars.

Tests whether P(price continues ≥1σ before reverting ≥1σ) varies with:
  1. Size of the triggering move  (single-bar scaled return buckets)
  2. Streak length                (consecutive same-direction bars)
  3. Cumulative scaled move       (total σ over the streak)
  4. Relative volume              (bar volume vs trailing 100-bar mean)

Trailing 100-min vol (σ) is used for all scaling.
Continuation target : close[i] * exp(+direction * σ)
Reversion  target   : close[i] * exp(-direction * σ)
Timeout             : neither hit within MAX_FORWARD bars
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
MAX_FORWARD          = 60
MAX_STREAK_LOOKBACK  = 20    # max bars to look back for streak


# ── Data ─────────────────────────────────────────────────────────────────────

def load() -> pd.DataFrame:
    df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
    if not df["ts"].dt.tz:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))].copy()
    df = df.sort_values("ts").reset_index(drop=True)
    df["gap"] = df["ts"].diff() > pd.Timedelta(minutes=2)
    print(f"Loaded {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")
    return df


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan(df: pd.DataFrame) -> pd.DataFrame:
    closes  = df["close"].values
    highs   = df["high"].values
    lows    = df["low"].values
    volumes = df["volume"].values
    gaps    = df["gap"].values
    n       = len(df)

    records = []

    for i in range(TRAILING_BARS, n - MAX_FORWARD):
        # Skip if gap anywhere in the forward window (trailing checked below)
        if gaps[i + 1 : i + MAX_FORWARD + 1].any():
            continue
        if gaps[i - TRAILING_BARS + 1 : i + 1].any():
            continue

        # ── Trailing vol ──────────────────────────────────────────────────────
        trail_rets = np.log(closes[i - TRAILING_BARS + 1 : i + 1]
                            / closes[i - TRAILING_BARS     : i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        # ── Relative volume ───────────────────────────────────────────────────
        mean_vol = volumes[i - TRAILING_BARS : i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else float("nan")

        # ── Current bar scaled return ─────────────────────────────────────────
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma
        direction = 1 if scaled > 0 else -1

        # ── Streak: consecutive same-direction bars ending at i ───────────────
        streak     = 1
        cum_scaled = scaled
        for k in range(i - 1, max(i - MAX_STREAK_LOOKBACK, TRAILING_BARS - 1), -1):
            if gaps[k]:
                break
            ret_k = math.log(closes[k] / closes[k - 1])
            if (ret_k > 0) == (scaled > 0) and ret_k != 0:
                streak     += 1
                cum_scaled += ret_k / sigma
            else:
                break

        # ── Forward outcome: which ±1σ target is hit first ───────────────────
        cont_target = closes[i] * math.exp( direction * sigma)
        rev_target  = closes[i] * math.exp(-direction * sigma)

        outcome = "timeout"
        for j in range(i + 1, i + MAX_FORWARD + 1):
            h, l = highs[j], lows[j]
            cont_hit = h >= cont_target if direction == 1 else l <= cont_target
            rev_hit  = l <= rev_target  if direction == 1 else h >= rev_target
            if cont_hit:
                outcome = "continuation"
                break
            if rev_hit:
                outcome = "reversion"
                break

        records.append({
            "sigma":      sigma,
            "scaled":     abs(scaled),
            "cum_scaled": abs(cum_scaled),
            "streak":     streak,
            "vol_ratio":  vol_ratio,
            "outcome":    outcome,
        })

    return pd.DataFrame(records)


# ── Reporting helpers ─────────────────────────────────────────────────────────

def p_cont(subset: pd.DataFrame) -> tuple[float, int, int]:
    valid   = subset[subset["outcome"] != "timeout"]
    timeout = (subset["outcome"] == "timeout").sum()
    if len(valid) == 0:
        return float("nan"), 0, timeout
    return (valid["outcome"] == "continuation").mean(), len(valid), timeout


def print_table(title: str, groups, col_fmt="{}", min_n: int = 20):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")
    print(f"  {'Group':<28} {'n':>6}  {'timeout':>8}  {'P(cont)':>8}")
    print(f"  {'-'*56}")
    for label, grp in groups:
        pc, n, nt = p_cont(grp)
        if n < min_n:
            continue
        flag = "  ◄" if pc >= 0.55 else ("  ▼" if pc <= 0.45 else "")
        print(f"  {str(label):<28} {n:>6,}  {nt:>8,}  {pc:>8.4f}{flag}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df  = load()
    res = scan(df)
    print(f"\nTotal bars scanned: {len(res):,}")

    overall_pc, overall_n, overall_nt = p_cont(res)
    print(f"Overall P(cont) all bars: {overall_pc:.4f}  (n={overall_n:,}, timeout={overall_nt:,})")

    # ── 1. Single-bar move size ───────────────────────────────────────────────
    bins   = [0, 1, 2, 3, 4, 5, 6, np.inf]
    labels = ["<1σ", "1–2σ", "2–3σ", "3–4σ", "4–5σ", "5–6σ", "6σ+"]
    res["size_bucket"] = pd.cut(res["scaled"], bins=bins, labels=labels)
    print_table("1. P(cont) by single-bar move size",
                res.groupby("size_bucket", observed=True))

    # ── 2. Streak length ──────────────────────────────────────────────────────
    streak_labels = {1: "1 bar (no streak)", 2: "2 bars", 3: "3 bars",
                     4: "4 bars", 5: "5 bars"}
    res["streak_label"] = res["streak"].apply(
        lambda s: streak_labels.get(s, "6+ bars"))
    streak_order = ["1 bar (no streak)", "2 bars", "3 bars",
                    "4 bars", "5 bars", "6+ bars"]
    print_table("2. P(cont) by same-direction streak length (all bars)",
                [(lbl, res[res["streak_label"] == lbl]) for lbl in streak_order])

    # Same but filtered to |scaled| ≥ 1σ (meaningful single-bar move)
    res1 = res[res["scaled"] >= 1]
    print_table("2b. P(cont) by streak length  [|single bar| ≥ 1σ only]",
                [(lbl, res1[res1["streak_label"] == lbl]) for lbl in streak_order])

    # ── 3. Cumulative scaled move ─────────────────────────────────────────────
    cbins   = [0, 1, 2, 3, 4, 5, 6, 8, np.inf]
    clabels = ["<1σ", "1–2σ", "2–3σ", "3–4σ", "4–5σ", "5–6σ", "6–8σ", "8σ+"]
    res["cum_bucket"] = pd.cut(res["cum_scaled"], bins=cbins, labels=clabels)
    print_table("3. P(cont) by cumulative scaled move over streak",
                res.groupby("cum_bucket", observed=True))

    # ── 4. Relative volume ────────────────────────────────────────────────────
    vbins   = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
    vlabels = ["<0.5×", "0.5–0.75×", "0.75–1×", "1–1.5×",
               "1.5–2×", "2–3×", "3×+"]
    res["vol_bucket"] = pd.cut(res["vol_ratio"], bins=vbins, labels=vlabels)
    print_table("4. P(cont) by relative volume (bar vol / 100-bar mean)",
                res.groupby("vol_bucket", observed=True))

    # ── 5. Move size × relative volume (2D, filtered to |scaled| ≥ 2σ) ───────
    res2 = res[res["scaled"] >= 2].copy()
    res2["vol_tier"] = pd.cut(res2["vol_ratio"],
                              bins=[0, 0.75, 1.5, np.inf],
                              labels=["low vol (<0.75×)", "normal (0.75–1.5×)", "high vol (1.5×+)"])
    res2["size_tier"] = pd.cut(res2["scaled"],
                               bins=[2, 3, 4, np.inf],
                               labels=["2–3σ", "3–4σ", "4σ+"])
    print(f"\n{'─'*62}")
    print(f"  5. P(cont): move size × relative volume  [|bar| ≥ 2σ]")
    print(f"{'─'*62}")
    print(f"  {'Size \\ Volume':<16} {'low (<0.75×)':>14}  {'normal (0.75–1.5×)':>20}  {'high (1.5×+)':>14}")
    print(f"  {'-'*68}")
    for size_lbl, sg in res2.groupby("size_tier", observed=True):
        row = f"  {str(size_lbl):<16}"
        for vol_lbl in ["low vol (<0.75×)", "normal (0.75–1.5×)", "high vol (1.5×+)"]:
            cell = sg[sg["vol_tier"] == vol_lbl]
            pc, n, _ = p_cont(cell)
            if n < 10:
                row += f"  {'—':>14}"
            else:
                row += f"  {pc:.3f} (n={n:>4})"
        print(row)
