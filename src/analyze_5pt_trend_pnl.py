"""
analyze_5pt_trend_pnl.py — Same UP signal as analyze_5pt_move_ml.py/
analyze_5pt_move_pnl.py, but instead of a small fixed bp target, let a
trailing stop ride the move to see if it can capture a longer trend.

Mechanics per trade:
  - entry = signal bar's close
  - initial_stop = entry - initial_stop_mult * move_pts (hard stop before
    any favorable movement)
  - as price makes new highs, trailing_stop ratchets up to
    max(trailing_stop, peak_high - trail_mult * move_pts) — never down
  - exit when a bar's low touches trailing_stop, or max_hold_min elapses
    (time exit at last close) — no fixed profit target at all

Usage:
  python src/analyze_5pt_trend_pnl.py
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "src")
from analyze_5pt_move_ml import load_data, fetch_bars  # noqa: E402

POINT_VALUE = 5.0
COMMISSION  = 4.0
INIT_STOP_MULTS = [1.5, 2.0]
TRAIL_MULTS     = [1.0, 2.0]
MAX_HOLDS       = [15, 30, 60]
THRESHOLDS      = [0.5, 0.6]


def simulate_trailing_trades(test: pd.DataFrame, p_up: np.ndarray, raw_by_date: dict,
                             threshold: float, init_stop_mult: float, trail_mult: float,
                             max_hold_min: int) -> list[dict]:
    trades = []
    test = test.assign(p_up=p_up).sort_values("ts")

    for date, day_rows in test.groupby("date"):
        day_bars = raw_by_date.get(date)
        if day_bars is None:
            continue
        bar_ts = day_bars["ts"].values
        highs, lows, closes = day_bars["h"].values, day_bars["l"].values, day_bars["c"].values

        blocked_until = None
        for _, row in day_rows.iterrows():
            if row["p_up"] < threshold:
                continue
            if blocked_until is not None and row["ts"] <= blocked_until:
                continue

            idx = np.searchsorted(bar_ts, row["ts"].to_datetime64())
            if idx >= len(bar_ts) or bar_ts[idx] != row["ts"].to_datetime64():
                continue
            entry = closes[idx]
            move_pts = row["move_pts"]
            trailing_stop = entry - init_stop_mult * move_pts
            peak = entry
            expires_at = row["ts"] + pd.Timedelta(minutes=max_hold_min)

            outcome, exit_price, exit_ts = None, None, None
            j_end = idx
            for j in range(idx + 1, len(bar_ts)):
                if pd.Timestamp(bar_ts[j]).tz_localize("UTC") > expires_at:
                    break
                j_end = j
                if lows[j] <= trailing_stop:
                    outcome, exit_price, exit_ts = "TRAIL_STOP", trailing_stop, bar_ts[j]
                    break
                peak = max(peak, highs[j])
                trailing_stop = max(trailing_stop, peak - trail_mult * move_pts)
            if outcome is None:
                outcome, exit_price, exit_ts = "TIME", closes[j_end], bar_ts[j_end]

            pnl_pts = exit_price - entry
            pnl_usd = pnl_pts * POINT_VALUE - COMMISSION
            trades.append(dict(date=date, ts=row["ts"], entry=entry, outcome=outcome,
                               exit_ts=exit_ts, pnl_pts=pnl_pts, pnl_usd=pnl_usd,
                               peak_gain=peak - entry))
            blocked_until = pd.Timestamp(exit_ts).tz_localize("UTC")

    return trades


def summarize(trades: list[dict], n_days: int) -> dict:
    if not trades:
        return dict(n=0, n_days=n_days)
    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) else (999 if wins else 0)
    return dict(n=len(trades), per_day=round(len(trades) / n_days, 2) if n_days else 0,
               win_rate=round(len(wins) / len(trades), 3),
               avg_win=round(np.mean(wins), 2) if wins else 0,
               avg_loss=round(np.mean(losses), 2) if losses else 0,
               avg_peak_gain=round(np.mean([t["peak_gain"] for t in trades]), 2),
               total_usd=round(sum(pnls), 1), pf=round(pf, 3),
               usd_per_day=round(sum(pnls) / n_days, 1) if n_days else 0, n_days=n_days)


def main():
    data = load_data()
    bars = fetch_bars()
    raw_by_date = {date: g.sort_values("ts") for date, g in bars.groupby("date")}
    feature_cols = [c for c in data.columns if c not in ("ts", "date", "label", "move_pts")]

    dates = sorted(data["date"].unique())
    n_folds = 5
    fold_size = len(dates) // (n_folds + 1)

    all_rows = []
    for k in range(1, n_folds + 1):
        train_dates = set(dates[:fold_size * k])
        test_dates  = set(dates[fold_size * k: fold_size * (k + 1)])
        if not test_dates:
            break
        train = data[data["date"].isin(train_dates)]
        test  = data[data["date"].isin(test_dates)]
        n_test_days = len(test_dates)

        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=50,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(train[feature_cols], train["label"])
        p_up = clf.predict_proba(test[feature_cols])[:, list(clf.classes_).index("UP")]

        print(f"\n{'='*105}\nFold {k}: train {min(train_dates)}->{max(train_dates)} "
              f"({len(train_dates)}d)  test {min(test_dates)}->{max(test_dates)} ({n_test_days}d)")
        print(f"{'='*105}")
        fold_rows = []
        for thr in THRESHOLDS:
            for ism in INIT_STOP_MULTS:
                for tm in TRAIL_MULTS:
                    for mh in MAX_HOLDS:
                        trades = simulate_trailing_trades(test, p_up, raw_by_date, thr, ism, tm, mh)
                        s = summarize(trades, n_test_days)
                        s.update(fold=k, threshold=thr, init_stop_mult=ism,
                                 trail_mult=tm, max_hold=mh)
                        fold_rows.append(s)
        df = pd.DataFrame(fold_rows)
        print(df[["fold", "threshold", "init_stop_mult", "trail_mult", "max_hold", "n", "per_day",
                  "win_rate", "avg_win", "avg_loss", "avg_peak_gain", "total_usd", "pf", "usd_per_day"]]
              .sort_values("pf", ascending=False).head(10).to_string(index=False))
        all_rows.extend(fold_rows)

    print(f"\n{'='*105}\nPOOLED ACROSS FOLDS 2-5\n{'='*105}")
    all_df = pd.DataFrame(all_rows)
    pooled = all_df[all_df["fold"] >= 2].groupby(
        ["threshold", "init_stop_mult", "trail_mult", "max_hold"]
    ).agg(n=("n", "sum"), total_usd=("total_usd", "sum"), n_days=("n_days", "sum")).reset_index()
    pooled["usd_per_day"] = pooled["total_usd"] / pooled["n_days"]
    print(pooled.sort_values("total_usd", ascending=False).to_string(index=False))
    print(f"\n{'='*105}\nWORST 5\n{'='*105}")
    print(pooled.sort_values("total_usd", ascending=True).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
