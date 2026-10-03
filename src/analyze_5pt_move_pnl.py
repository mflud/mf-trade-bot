"""
analyze_5pt_move_pnl.py — P&L simulation for the UP-signal model from
analyze_5pt_move_ml.py (DOWN dropped — walk-forward showed it's not
predictable from trailing features; see that script's docstring).

For each walk-forward fold (same expanding-window folds, same RF), opens
an actual simulated trade whenever P(UP) >= threshold and no trade is
already open that day: entry = signal bar's close, target = entry +
move_pts (the same bp-based distance the label used), stop = entry -
stop_mult*move_pts, time-exit after hold_min minutes if neither hit.
Walks the RAW 1-min bars (not just the 5/10/15-feature snapshots) to
resolve the outcome, checking stop before target within a bar.

This is what actually answers "is this profitable" — the ML script only
measured hit rate, not what the misses cost.

Usage:
  python src/analyze_5pt_move_pnl.py
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "src")
from analyze_5pt_move_ml import load_data, fetch_bars  # noqa: E402

POINT_VALUE = 5.0   # MES $/pt
COMMISSION  = 4.0   # round-trip, matches the rest of this session's backtests
STOP_MULTS  = [1.0, 1.5, 2.0]
HOLD_MINS   = [5, 10]
THRESHOLDS  = [0.5, 0.6]


def simulate_trades(test: pd.DataFrame, p_up: np.ndarray, raw_by_date: dict,
                    threshold: float, stop_mult: float, hold_min: int) -> list[dict]:
    trades = []
    test = test.assign(p_up=p_up).sort_values("ts")
    open_until = {}   # date -> ts when the current open trade expires/resolves

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
                continue   # a trade is already open

            # entry price: derive from the raw bars at this ts (feature frame has no close col)
            idx = np.searchsorted(bar_ts, row["ts"].to_datetime64())
            if idx >= len(bar_ts) or bar_ts[idx] != row["ts"].to_datetime64():
                continue
            entry = closes[idx]
            move_pts = row["move_pts"]
            target = entry + move_pts
            stop   = entry - stop_mult * move_pts
            expires_at = row["ts"] + pd.Timedelta(minutes=hold_min)

            outcome, exit_price, exit_ts = None, None, None
            for j in range(idx + 1, len(bar_ts)):
                if pd.Timestamp(bar_ts[j]).tz_localize("UTC") > expires_at:
                    break
                if lows[j] <= stop:
                    outcome, exit_price, exit_ts = "STOP", stop, bar_ts[j]; break
                if highs[j] >= target:
                    outcome, exit_price, exit_ts = "TARGET", target, bar_ts[j]; break
            if outcome is None:
                # time exit at last bar within the hold window (or last bar of day)
                j_end = idx
                for j in range(idx + 1, len(bar_ts)):
                    if pd.Timestamp(bar_ts[j]).tz_localize("UTC") > expires_at:
                        break
                    j_end = j
                outcome, exit_price, exit_ts = "TIME", closes[j_end], bar_ts[j_end]

            pnl_pts = exit_price - entry
            pnl_usd = pnl_pts * POINT_VALUE - COMMISSION
            trades.append(dict(date=date, ts=row["ts"], entry=entry, target=target, stop=stop,
                               outcome=outcome, exit_ts=exit_ts, pnl_pts=pnl_pts, pnl_usd=pnl_usd,
                               move_pts=move_pts))
            blocked_until = pd.Timestamp(exit_ts).tz_localize("UTC")

    return trades


def summarize(trades: list[dict], n_days: int) -> dict:
    if not trades:
        return dict(n=0)
    pnls = [t["pnl_usd"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) else (999 if wins else 0)
    return dict(n=len(trades), per_day=round(len(trades) / n_days, 2) if n_days else 0,
               win_rate=round(len(wins) / len(trades), 3),
               avg_win=round(np.mean(wins), 2) if wins else 0,
               avg_loss=round(np.mean(losses), 2) if losses else 0,
               total_usd=round(sum(pnls), 1), pf=round(pf, 3),
               usd_per_day=round(sum(pnls) / n_days, 1) if n_days else 0)


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

        print(f"\n{'='*100}\nFold {k}: train {min(train_dates)}->{max(train_dates)} "
              f"({len(train_dates)}d)  test {min(test_dates)}->{max(test_dates)} ({n_test_days}d)")
        print(f"{'='*100}")
        fold_rows = []
        for thr in THRESHOLDS:
            for sm in STOP_MULTS:
                for hm in HOLD_MINS:
                    trades = simulate_trades(test, p_up, raw_by_date, thr, sm, hm)
                    s = summarize(trades, n_test_days)
                    s.update(fold=k, threshold=thr, stop_mult=sm, hold_min=hm, n_days=n_test_days)
                    fold_rows.append(s)
        df = pd.DataFrame(fold_rows)
        print(df[["fold", "threshold", "stop_mult", "hold_min", "n", "per_day",
                  "win_rate", "avg_win", "avg_loss", "total_usd", "pf", "usd_per_day"]]
              .sort_values("pf", ascending=False).to_string(index=False))
        all_rows.extend(fold_rows)

    print(f"\n{'='*100}\nPOOLED ACROSS FOLDS 2-5 (dropping fold1 — too little training data, unstable per the ML script)\n{'='*100}")
    all_df = pd.DataFrame(all_rows)
    pooled = all_df[all_df["fold"] >= 2].groupby(["threshold", "stop_mult", "hold_min"]).agg(
        n=("n", "sum"), total_usd=("total_usd", "sum"), n_days=("n_days", "sum")
    ).reset_index()
    pooled["usd_per_day"] = pooled["total_usd"] / pooled["n_days"]
    print(pooled.sort_values("total_usd", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
