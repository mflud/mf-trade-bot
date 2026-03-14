"""
Historical backtest: does trailing 10-day realised vol predict next-day vol for MES?

Session definition: full 24h CME session (excluding 21:00-22:00 UTC settlement gap).
CME session boundary: 17:00 CT (prev day) → 16:00 CT (current day).

For each session i with at least 10 prior complete sessions:
  trailing_vol  = annualised realised vol over sessions [i-9 … i]  (10 sessions)
  forward_vol   = annualised realised vol of session i+1            (1 session)

Data: fetches all available 1-min bars across known MES quarterly contracts,
      paging backward up to 365 calendar days per contract. Cached to
      mes_hist_1min.csv to avoid re-fetching.
"""

import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, "src")
from topstep_client import TopstepClient

CACHE_PATH = "mes_hist_1min.csv"
SETTLEMENT_START_UTC = 21
SETTLEMENT_END_UTC   = 22
BARS_PER_YEAR_FULL   = 252 * 23 * 60   # annualisation for full 24h session
TRAILING_SESSIONS    = 10
MIN_BARS_PER_SESSION = 60              # skip incomplete sessions (< 1h of data)


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_contract_bars(client: TopstepClient, contract_id: str,
                        days_back: int = 365) -> pd.DataFrame:
    """Page backward through 1-min history for a single contract."""
    all_bars = []
    end = datetime.now(timezone.utc)
    limit_date = end - timedelta(days=days_back)

    while end > limit_date:
        start = max(end - timedelta(days=30), limit_date)
        bars = client.get_bars(
            contract_id=contract_id,
            start=start, end=end,
            unit=TopstepClient.MINUTE, unit_number=1,
            limit=20000,
        )
        if not bars:
            break
        oldest_dt = datetime.fromisoformat(bars[-1]["t"])
        all_bars.extend(bars)
        print(f"    {contract_id}: {len(bars):,} bars back to {oldest_dt.date()}", flush=True)
        end = oldest_dt - timedelta(minutes=1)
        if len(bars) < 20000:
            break

    if not all_bars:
        return pd.DataFrame()

    df = pd.DataFrame(all_bars)
    df["contract"] = contract_id
    return df


def fetch_all(client: TopstepClient) -> pd.DataFrame:
    """Fetch 1-min history for all known MES quarterly contracts."""
    frames = []
    for cid in TopstepClient.MES_CONTRACTS:
        print(f"  Fetching {cid}…", flush=True)
        df = fetch_contract_bars(client, cid)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No data fetched.")

    combined = pd.concat(frames, ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["t"], utc=True)
    combined = (combined
                .drop_duplicates("ts")
                .sort_values("ts")
                .reset_index(drop=True)
                .rename(columns={"o": "open", "h": "high", "l": "low",
                                 "c": "close", "v": "volume"}))
    combined.to_csv(CACHE_PATH, index=False)
    print(f"  Cached {len(combined):,} bars to {CACHE_PATH}")
    return combined


def load_data() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["ts"])
        if "ts" in df.columns and not df["ts"].dt.tz:
            df["ts"] = df["ts"].dt.tz_localize("UTC")
        n_days = (df["ts"].max() - df["ts"].min()).days
        print(f"Cache: {len(df):,} bars  ({df['ts'].min().date()} → {df['ts'].max().date()})")
        if n_days < 30:
            print("  Cache too short — re-fetching.")
        else:
            return df

    print("Fetching 1-min bars from TopstepX…")
    with TopstepClient() as client:
        return fetch_all(client)


# ── Session splitting ─────────────────────────────────────────────────────────

def assign_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out settlement gap and assign each bar a CME session date.
    Session boundary: 17:00 CT (prev day) → 16:00 CT (current day).
    Bars at/after 17:00 CT are labelled as the next calendar date's session.
    """
    hour_utc = df["ts"].dt.hour
    df = df[~((hour_utc >= SETTLEMENT_START_UTC) & (hour_utc < SETTLEMENT_END_UTC))].copy()

    df["ts_ct"] = df["ts"].dt.tz_convert("America/Chicago")
    df["session_date"] = df["ts_ct"].dt.date
    after_close = df["ts_ct"].dt.hour >= 17
    df.loc[after_close, "session_date"] = (
        df.loc[after_close, "ts_ct"] + pd.Timedelta(days=1)
    ).dt.date

    return df.reset_index(drop=True)


def session_closes(df: pd.DataFrame) -> dict[object, np.ndarray]:
    """Return {session_date: array_of_closes} for sessions with enough bars."""
    grouped = {}
    for date, grp in df.groupby("session_date"):
        closes = grp.sort_values("ts")["close"].values
        if len(closes) >= MIN_BARS_PER_SESSION:
            grouped[date] = closes
    return grouped


# ── Volatility ────────────────────────────────────────────────────────────────

def realised_vol(closes: np.ndarray) -> float:
    rets = np.log(closes[1:] / closes[:-1])
    if len(rets) < 2:
        return float("nan")
    return float(np.std(rets, ddof=1) * math.sqrt(BARS_PER_YEAR_FULL))


# ── Backtest ──────────────────────────────────────────────────────────────────

def build_records(sessions: dict) -> pd.DataFrame:
    """
    For each session i, compute trailing 10-session vol and next-session vol.
    Returns one row per valid observation.
    """
    dates = sorted(sessions.keys())
    records = []

    for i in range(TRAILING_SESSIONS, len(dates) - 1):
        trailing_dates = dates[i - TRAILING_SESSIONS : i]
        forward_date   = dates[i + 1]

        # Concatenate closes across the trailing window
        trailing_closes = np.concatenate([sessions[d] for d in trailing_dates])
        forward_closes  = sessions[forward_date]

        tv = realised_vol(trailing_closes)
        fv = realised_vol(forward_closes)

        if not (math.isnan(tv) or math.isnan(fv)):
            records.append({
                "date":         dates[i],
                "trailing_vol": tv,
                "forward_vol":  fv,
            })

    return pd.DataFrame(records)


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse(res: pd.DataFrame):
    tv = res["trailing_vol"].values
    fv = res["forward_vol"].values
    n  = len(res)

    pearson_r, pearson_p  = stats.pearsonr(tv, fv)
    spearman_r, spearman_p = stats.spearmanr(tv, fv)

    slope, intercept, *_ = stats.linregress(tv, fv)
    r2 = pearson_r ** 2

    # Baseline: predict with historical mean of trailing vol
    mean_tv  = tv.mean()
    mae_model   = np.mean(np.abs(tv - fv))
    mae_baseline = np.mean(np.abs(mean_tv - fv))
    skill = 1 - mae_model / mae_baseline   # positive = beats mean

    print(f"\n{'='*60}")
    print(f"Trailing {TRAILING_SESSIONS}-session vol → next-session vol")
    print(f"n = {n} observations  "
          f"({res['date'].min()} → {res['date'].max()})")
    print(f"{'='*60}")

    print(f"\nCorrelation")
    print(f"  Pearson  r  = {pearson_r:+.4f}   p = {pearson_p:.2e}")
    print(f"  Spearman r  = {spearman_r:+.4f}   p = {spearman_p:.2e}")
    print(f"  R²          = {r2:.4f}  ({r2*100:.1f}% of variance explained)")

    print(f"\nLinear fit:  fwd_vol = {slope:.4f} × trailing_vol + {intercept:.4f}")

    print(f"\nMAE comparison (annualised vol)")
    print(f"  Model  (trailing vol as predictor): {mae_model*100:.2f}pp")
    print(f"  Baseline (historical mean = {mean_tv*100:.1f}%): {mae_baseline*100:.2f}pp")
    skill_sign = "better" if skill > 0 else "worse"
    print(f"  Skill score: {skill:+.3f}  ({abs(skill)*100:.1f}% {skill_sign} than mean baseline)")

    # Quartile breakdown
    res2 = res.copy()
    res2["quartile"] = pd.qcut(res2["trailing_vol"], 4,
                               labels=["Q1 low vol", "Q2", "Q3", "Q4 high vol"])
    print(f"\nForward vol by trailing vol quartile:")
    print(f"  {'Quartile':<14} {'n':>5}  {'mean fwd':>10}  {'median fwd':>11}  {'std':>8}")
    for label, grp in res2.groupby("quartile", observed=True):
        fv_grp = grp["forward_vol"]
        q_range = grp["trailing_vol"]
        print(f"  {str(label):<14} {len(grp):>5}  "
              f"{fv_grp.mean()*100:>9.2f}%  "
              f"{fv_grp.median()*100:>10.2f}%  "
              f"{fv_grp.std()*100:>7.2f}%  "
              f"(trailing: {q_range.min()*100:.1f}–{q_range.max()*100:.1f}%)")

    print(f"\nConclusion:")
    if abs(pearson_r) >= 0.5 and pearson_p < 0.05:
        strength = "strong"
    elif abs(pearson_r) >= 0.3 and pearson_p < 0.05:
        strength = "moderate"
    elif abs(pearson_r) >= 0.1 and pearson_p < 0.05:
        strength = "weak"
    else:
        strength = "negligible"
    print(f"  {strength.capitalize()} predictive signal (r={pearson_r:.3f}, R²={r2*100:.1f}%).")
    if skill > 0:
        print(f"  Using trailing vol beats the mean baseline by {skill*100:.1f}% on MAE.")
    else:
        print(f"  Using trailing vol does NOT beat the mean baseline on MAE.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    print(f"\nAssigning CME sessions…")
    df = assign_sessions(df)

    sessions = session_closes(df)
    dates = sorted(sessions.keys())
    print(f"{len(sessions)} complete sessions  ({dates[0]} → {dates[-1]})")

    if len(sessions) < TRAILING_SESSIONS + 2:
        print(f"Not enough sessions — re-fetching fresh data.")
        os.remove(CACHE_PATH)
        print("Fetching 1-min bars from TopstepX…")
        with TopstepClient() as client:
            df = fetch_all(client)
        df = assign_sessions(df)
        sessions = session_closes(df)
        dates = sorted(sessions.keys())
        print(f"{len(sessions)} sessions after re-fetch.")

    res = build_records(sessions)
    analyse(res)
