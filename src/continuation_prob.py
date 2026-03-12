"""
Conditional continuation probability for MES 1-min bars.

For a detected net price move of X pts over a lookback window, what is the
probability that price continues in the same direction by 2.5 pts before
reverting 2.5 pts — conditioned on trailing volatility or regime_pl?

Logic per bar i:
  1. net_move = close[i] - close[i - lookback]
  2. If |net_move| >= move_threshold:
       - direction = sign(net_move)
       - trailing_vol and regime_pl computed over closes[i-lookback : i+1]
       - scan forward bars using high/low to find which target is hit first:
           continuation target : close[i] + direction * target_pts
           reversion  target   : close[i] - direction * target_pts
       - label = "continuation" | "reversion" | "timeout" (neither hit in time)
  3. Sweep vol thresholds to find which (if any) improve continuation probability
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from topstep_client import TopstepClient

# ── Config ───────────────────────────────────────────────────────────────────

CACHE_PATH   = "mes_1min_bars.csv"
BAR_MINUTES  = 1
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22

MOVE_THRESHOLDS  = [5.0, 8.0, 10.0]
TARGET_PTS       = 2.5
LOOKBACK         = 30
MAX_FORWARD      = 60
BARS_PER_YEAR    = 252 * 23 * 60   # ~23h/day for CME futures


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_and_cache(client: TopstepClient, contract_id: str) -> pd.DataFrame:
    all_bars = []
    end = datetime.now(timezone.utc)
    limit_date = end - timedelta(days=180)
    page_days = 10
    page_limit = 10000

    while end > limit_date:
        start = end - timedelta(days=page_days)
        for attempt in range(3):
            try:
                bars = client.get_bars(
                    contract_id=contract_id,
                    start=start, end=end,
                    unit=TopstepClient.MINUTE, unit_number=BAR_MINUTES,
                    limit=page_limit,
                )
                break
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  retrying ({e})", flush=True)
        if not bars:
            break
        oldest_dt = datetime.fromisoformat(bars[-1]["t"])
        all_bars.extend(bars)
        end = oldest_dt - timedelta(minutes=1)
        print(f"  fetched {len(bars):,} bars back to {oldest_dt.date()}", flush=True)
        if len(bars) < page_limit:
            break

    df = pd.DataFrame(all_bars)
    df["ts"] = pd.to_datetime(df["t"], utc=True)
    df = (df.drop_duplicates("ts")
            .sort_values("ts")
            .reset_index(drop=True)
            .rename(columns={"o": "open", "h": "high", "l": "low",
                              "c": "close", "v": "volume"}))
    df.to_csv(CACHE_PATH, index=False)
    print(f"  cached to {CACHE_PATH}")
    return df


def load_data() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached bars from {CACHE_PATH}")
        df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
    else:
        print("Fetching 1-min bars from TopstepX...")
        with TopstepClient() as client:
            contracts = client.search_contracts("MES")
            contract  = contracts[0]
            print(f"Contract: {contract['name']}  id={contract['id']}")
            df = fetch_and_cache(client, contract["id"])

    hour = df["ts"].dt.hour
    df = df[~((hour >= SETTLEMENT_START_UTC) & (hour < SETTLEMENT_END_UTC))]
    df = df.reset_index(drop=True)
    df["new_session"] = df["ts"].diff() > pd.Timedelta(minutes=BAR_MINUTES * 2)
    return df


# ── Analytics ────────────────────────────────────────────────────────────────

def compute_features(closes: np.ndarray) -> tuple[float, float]:
    """Return (trailing_vol_annualised, regime_pl) for a close array."""
    rets = np.log(closes[1:] / closes[:-1])
    # vol
    vol = float(np.std(rets, ddof=1) * math.sqrt(BARS_PER_YEAR)) if len(rets) >= 2 else float("nan")
    # regime_pl
    total_abs = np.sum(np.abs(rets))
    pl = abs(float(np.sum(rets))) / float(total_abs) if total_abs > 0 else 0.0
    return vol, pl


def hit_target(highs: np.ndarray, lows: np.ndarray,
               cont_target: float, rev_target: float,
               direction: int) -> str:
    for h, l in zip(highs, lows):
        cont_hit = h >= cont_target if direction == 1 else l <= cont_target
        rev_hit  = l <= rev_target  if direction == 1 else h >= rev_target
        if cont_hit and rev_hit:
            return "continuation"
        if cont_hit:
            return "continuation"
        if rev_hit:
            return "reversion"
    return "timeout"


# ── Backtest ─────────────────────────────────────────────────────────────────

def run(df: pd.DataFrame, move_threshold: float,
        target_pts: float = TARGET_PTS,
        lookback: int = LOOKBACK,
        max_forward: int = MAX_FORWARD) -> pd.DataFrame:

    closes      = df["close"].values
    highs       = df["high"].values
    lows        = df["low"].values
    new_session = df["new_session"].values
    records     = []

    for i in range(lookback, len(df) - max_forward):
        if new_session[i - lookback + 1 : i + max_forward + 1].any():
            continue

        net_move = closes[i] - closes[i - lookback]
        if abs(net_move) < move_threshold:
            continue

        direction = 1 if net_move > 0 else -1
        vol, pl = compute_features(closes[i - lookback : i + 1])

        outcome = hit_target(
            highs[i + 1 : i + max_forward + 1],
            lows[i + 1  : i + max_forward + 1],
            closes[i] + direction * target_pts,
            closes[i] - direction * target_pts,
            direction,
        )

        records.append({
            "ts":           df["ts"].iloc[i],
            "net_move":     net_move,
            "trailing_vol": vol,
            "regime_pl":    pl,
            "outcome":      outcome,
        })

    return pd.DataFrame(records)


def p_cont(subset: pd.DataFrame) -> tuple[float, int]:
    valid = subset[subset["outcome"] != "timeout"]
    n = len(valid)
    if n == 0:
        return float("nan"), 0
    return (valid["outcome"] == "continuation").mean(), n


def vol_threshold_sweep(res: pd.DataFrame, col: str, n_steps: int = 20) -> pd.DataFrame:
    """
    Sweep percentile thresholds for `col` and report P(cont) above/below each.
    Returns a DataFrame sorted by best P(cont) above threshold.
    """
    valid = res[res["outcome"] != "timeout"].copy()
    percentiles = np.linspace(10, 90, n_steps)
    rows = []
    for pct in percentiles:
        thresh = np.percentile(valid[col], pct)
        above = valid[valid[col] >  thresh]
        below = valid[valid[col] <= thresh]
        p_above, n_above = p_cont(above)
        p_below, n_below = p_cont(below)
        rows.append({
            "percentile":  pct,
            "threshold":   thresh,
            "n_above":     n_above,
            "P_cont_above": p_above,
            "n_below":     n_below,
            "P_cont_below": p_below,
            "spread":      p_above - p_below,
        })
    return pd.DataFrame(rows)


def summarise(res: pd.DataFrame, move_threshold: float):
    print(f"\n{'='*62}")
    print(f"Move >= {move_threshold:.0f} pts  |  target={TARGET_PTS} pts  "
          f"|  lookback={LOOKBACK} bars  |  n={len(res):,}")
    print(f"{'='*62}")

    p_all, n_all = p_cont(res)
    print(f"  Baseline P(continuation) = {p_all:.4f}  (n={n_all:,})")

    for col, label in [("trailing_vol", "Trailing vol"), ("regime_pl", "Regime PL")]:
        print(f"\n  ── {label} threshold sweep ──────────────────────────")
        sweep = vol_threshold_sweep(res, col)

        # Best row by P(cont) above threshold (min 30 obs)
        best_above = sweep[sweep["n_above"] >= 30].sort_values("P_cont_above", ascending=False).iloc[0]
        best_below = sweep[sweep["n_below"] >= 30].sort_values("P_cont_below", ascending=False).iloc[0]

        print(f"  {'Pct':>4}  {'Threshold':>10}  {'n_above':>7}  "
              f"{'P(cont|above)':>14}  {'n_below':>7}  {'P(cont|below)':>14}  {'spread':>8}")
        for _, row in sweep.iterrows():
            marker = ""
            if row["percentile"] == best_above["percentile"]:
                marker = " <-- best above"
            elif row["percentile"] == best_below["percentile"]:
                marker = " <-- best below"
            print(f"  {row['percentile']:>4.0f}%  {row['threshold']:>10.4f}  "
                  f"{row['n_above']:>7.0f}  {row['P_cont_above']:>14.4f}  "
                  f"{row['n_below']:>7.0f}  {row['P_cont_below']:>14.4f}  "
                  f"{row['spread']:>+8.4f}{marker}")

        print(f"\n  Best P(cont) above threshold: "
              f"{best_above['P_cont_above']:.4f}  "
              f"at {col} > {best_above['threshold']:.4f}  "
              f"(p{best_above['percentile']:.0f}, n={best_above['n_above']:.0f})")
        print(f"  Best P(cont) below threshold: "
              f"{best_below['P_cont_below']:.4f}  "
              f"at {col} <= {best_below['threshold']:.4f}  "
              f"(p{best_below['percentile']:.0f}, n={best_below['n_below']:.0f})")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"\n{len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")

    for thresh in MOVE_THRESHOLDS:
        res = run(df, move_threshold=thresh)
        summarise(res, move_threshold=thresh)
