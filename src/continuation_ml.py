"""
ML continuation-probability model for MES, in basis points.

Extends continuation_prob.py's continuation/reversion framework with:
  - non-overlapping move detection via pivot chaining (a mini zigzag): once a
    move's outcome resolves (continuation, reversion, or timeout), the next
    search starts fresh from the resolution bar, so no two labeled events
    share bars.
  - everything expressed in basis points (bp) instead of points, since a
    fixed point move means a different real move as MES's price level shifts
    (10bp ~= 8pts at MES 7800, but ~= 4pts at MES 4000).
  - four feature groups: return since session open, return since 10 minutes
    after open ("settled" market), return since previous day's close, and
    trailing 5/10/15/20-minute returns.
  - logistic regression + gradient boosting classifiers instead of a
    percentile sweep, with a chronological train/test split.

Move definition (per move_threshold_bp in MOVE_THRESHOLDS_BP):
  1. Within a trading day, starting from an anchor bar, scan forward for the
     first bar j where |close[j] - close[anchor]| / close[anchor] * 1e4 >=
     move_threshold_bp. That's the trigger — features are computed as of bar j.
  2. direction = sign(close[j] - close[anchor]).
  3. continuation_target = close[j] * (1 + direction * move_threshold_bp/1e4)
     reversion_target     = close[j] * (1 - direction * move_threshold_bp/1e4)
     (a further move of the same bp size, in bp terms — symmetric with the
     trigger itself, matching continuation_prob.py's target_pts convention.)
  4. Scan forward (capped at MAX_FORWARD_MIN, and never past the trading
     day) for whichever target is hit first using bar highs/lows ->
     "continuation" | "reversion" | "timeout".
  5. anchor moves to the resolution bar; repeat. This guarantees legs never
     overlap and chains naturally through a long run (a 30bp+ move becomes a
     sequence of chained continuation legs).

Usage:
    python src/continuation_ml.py
"""

import warnings

import numpy as np
import pandas as pd
import pytz
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Config ───────────────────────────────────────────────────────────────────

HIST_CSV = "mes_hist_1min.csv"   # back-adjusted continuous MES series, 2019-present
ET       = pytz.timezone("America/New_York")

RTH_START_MIN = 9 * 60 + 30    # 9:30 ET
RTH_END_MIN   = 16 * 60        # 16:00 ET
SETTLED_MIN   = 9 * 60 + 40    # 9:40 ET — "market has settled"

MOVE_THRESHOLDS_BP = [5.0, 10.0, 15.0, 20.0]
MAX_FORWARD_MIN     = 30   # cap outcome search — matches your stated hold horizon
TRAIL_WINDOWS_MIN   = [5, 10, 15, 20]

FEATURE_COLS = [
    "ret_since_open_bp",
    "ret_since_settle_bp",
    "ret_since_prevclose_bp",
    "ret_trail_5m_bp",
    "ret_trail_10m_bp",
    "ret_trail_15m_bp",
    "ret_trail_20m_bp",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_rth_bars() -> pd.DataFrame:
    df = pd.read_csv(HIST_CSV, usecols=["ts", "open", "high", "low", "close"])
    df["ts"]    = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    df["ts_et"] = df["ts"].dt.tz_convert(ET)
    df["date"]  = df["ts_et"].dt.date
    df["hm"]    = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df = df[(df["hm"] >= RTH_START_MIN) & (df["hm"] < RTH_END_MIN)].copy()
    return df.sort_values("ts").reset_index(drop=True)


def build_daily_state(df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-bar features that only need daily context: return since
    open, return since 9:40 settle price, return since previous day's close,
    and trailing 5/10/15/20-min returns (all in bp, all within-day only)."""
    df = df.copy()
    daily_close = df.groupby("date")["close"].last()
    prev_close  = daily_close.shift(1)

    out_frames = []
    for date, day in df.groupby("date", sort=True):
        day = day.reset_index(drop=True)
        open_px = day["close"].iloc[0]
        settle_rows = day[day["hm"] >= SETTLED_MIN]
        settle_px = settle_rows["close"].iloc[0] if not settle_rows.empty else np.nan
        prev_px = prev_close.get(date, np.nan)

        day["ret_since_open_bp"]     = (day["close"] / open_px - 1) * 1e4
        day["ret_since_settle_bp"]   = (day["close"] / settle_px - 1) * 1e4
        day["ret_since_prevclose_bp"] = (day["close"] / prev_px - 1) * 1e4 if pd.notna(prev_px) else np.nan

        for m in TRAIL_WINDOWS_MIN:
            shifted = day["close"].shift(m)
            day[f"ret_trail_{m}m_bp"] = (day["close"] / shifted - 1) * 1e4

        out_frames.append(day)

    return pd.concat(out_frames, ignore_index=True)


# ── Move detection (pivot chaining) ─────────────────────────────────────────

def detect_moves(df: pd.DataFrame, move_threshold_bp: float,
                  max_forward_min: int = MAX_FORWARD_MIN) -> pd.DataFrame:
    records = []
    for date, day in df.groupby("date", sort=True):
        day = day.reset_index(drop=True)
        closes = day["close"].values
        highs  = day["high"].values
        lows   = day["low"].values
        n = len(day)
        if n < max(TRAIL_WINDOWS_MIN) + 2:
            continue

        anchor = 0
        while anchor < n - 1:
            anchor_px = closes[anchor]
            trigger = None
            for j in range(anchor + 1, n):
                move_bp = (closes[j] - anchor_px) / anchor_px * 1e4
                if abs(move_bp) >= move_threshold_bp:
                    trigger = j
                    direction = 1 if move_bp > 0 else -1
                    break
            if trigger is None:
                break   # no more qualifying moves left today

            trig_px = closes[trigger]
            cont_target = trig_px * (1 + direction * move_threshold_bp / 1e4)
            rev_target  = trig_px * (1 - direction * move_threshold_bp / 1e4)

            eod_censored = trigger + max_forward_min > n - 1   # day ends before horizon is reached
            end = min(trigger + max_forward_min, n - 1)
            outcome, resolve_idx = "timeout", end
            for k in range(trigger + 1, end + 1):
                cont_hit = highs[k] >= cont_target if direction == 1 else lows[k] <= cont_target
                rev_hit  = lows[k]  <= rev_target  if direction == 1 else highs[k] >= rev_target
                if cont_hit or rev_hit:
                    outcome = "continuation" if cont_hit else "reversion"
                    resolve_idx = k
                    eod_censored = False
                    break

            row = day.iloc[trigger]
            records.append({
                "date": date, "trigger_ts": row["ts"], "direction": direction,
                "move_threshold_bp": move_threshold_bp, "outcome": outcome,
                "bars_to_resolve": resolve_idx - trigger, "eod_censored": eod_censored,
                **{c: row[c] for c in FEATURE_COLS},
            })

            anchor = resolve_idx if resolve_idx > anchor else anchor + 1

    return pd.DataFrame(records)


# ── Modeling ─────────────────────────────────────────────────────────────────

def prep_xy(events: pd.DataFrame):
    """Direction-adjust features (positive = in favor of the move's direction)
    and pool longs/shorts into one sample. Drops timeouts and NaN rows."""
    ev = events[events["outcome"] != "timeout"].copy()
    ev = ev.dropna(subset=FEATURE_COLS)
    for c in FEATURE_COLS:
        ev[c + "_adj"] = ev[c] * ev["direction"]
    X = ev[[c + "_adj" for c in FEATURE_COLS]].values
    y = (ev["outcome"] == "continuation").astype(int).values
    return ev, X, y


def evaluate_threshold(events: pd.DataFrame, threshold_bp: float):
    print(f"\n{'='*76}")
    print(f"Move threshold = {threshold_bp:.0f}bp  |  target = {threshold_bp:.0f}bp further  "
          f"|  max_forward = {MAX_FORWARD_MIN}min")
    print(f"{'='*76}")

    n_total = len(events)
    n_timeout = (events["outcome"] == "timeout").sum()
    print(f"  Raw events: {n_total}  (timeouts: {n_timeout}, {n_timeout/n_total:.1%})")

    ev, X, y = prep_xy(events)
    n = len(ev)
    if n < 100:
        print(f"  Only {n} usable events after dropping timeouts/NaNs — too few to model.")
        return

    base_rate = y.mean()
    print(f"  Usable events (non-timeout, complete features): {n}")
    print(f"  Baseline P(continuation) = {base_rate:.3f}")

    # Chronological split — no shuffling, avoids leakage across autocorrelated days.
    ev_sorted_idx = np.argsort(ev["trigger_ts"].values)
    X, y = X[ev_sorted_idx], y[ev_sorted_idx]
    split = int(n * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    if len(np.unique(y_test)) < 2 or len(np.unique(y_train)) < 2:
        print("  Train/test split has only one class present — skipping.")
        return

    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    logit = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
    logit_auc = roc_auc_score(y_test, logit.predict_proba(Xte)[:, 1])

    gbc = HistGradientBoostingClassifier(max_depth=3, random_state=0).fit(X_train, y_train)
    gbc_auc = roc_auc_score(y_test, gbc.predict_proba(X_test)[:, 1])

    print(f"\n  Test set: n={len(y_test)}  P(continuation)={y_test.mean():.3f}")
    print(f"  Logistic regression AUC: {logit_auc:.3f}   Gradient boosting AUC: {gbc_auc:.3f}")
    print(f"  (0.50 = no predictive power, 1.00 = perfect separation)")

    print(f"\n  Logistic regression coefficients (standardized; direction-adjusted features,")
    print(f"  positive = higher value -> more likely to continue in the move's direction):")
    for c, coef in sorted(zip(FEATURE_COLS, logit.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"    {c:<26} {coef:+.3f}")

    perm_imp = _permutation_importance(gbc, X_test, y_test)
    print(f"\n  Gradient boosting permutation importance (AUC drop when shuffled):")
    for c, imp in sorted(zip(FEATURE_COLS, perm_imp), key=lambda t: -t[1]):
        print(f"    {c:<26} {imp:+.4f}")


def _permutation_importance(model, X, y, n_repeats: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    importances = np.zeros(X.shape[1])
    for col in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, col])
            auc = roc_auc_score(y, model.predict_proba(Xp)[:, 1])
            drops.append(base_auc - auc)
        importances[col] = np.mean(drops)
    return importances


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading RTH 1-min MES bars from {HIST_CSV}…")
    df = load_rth_bars()
    print(f"  {len(df):,} RTH bars  ({df['date'].min()} -> {df['date'].max()}, "
          f"{df['date'].nunique()} trading days)")

    df = build_daily_state(df)

    for thresh in MOVE_THRESHOLDS_BP:
        events = detect_moves(df, thresh)
        evaluate_threshold(events, thresh)


if __name__ == "__main__":
    main()
