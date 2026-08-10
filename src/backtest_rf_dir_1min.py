"""
backtest_rf_dir_1min.py — Directional RF on 1-min bars.

Same architecture as the 2-min directional model (backtest_ml_dir_tod.py)
but operating at 1-min bar resolution:

  - Qualifying move gate: |close - close[N bars ago]| >= move_threshold_bps
  - Return features: r1m_1..LOOKBACK (sign-flipped by direction)
  - Vol features: realized_vol (10-bar std), atr_ratio, vol_regime (60-bar pct)
  - ret_open: log(close / 9:30 open)
  - move_size_bps: magnitude of qualifying move
  - Label: does price hit +target_bps before -stop_bps within HORIZON bars?
  - Bars after 9:40 ET only (matches live gate)
  - Pre-market bars from 8:30 used for vol warmup, dropped before labelling

Sweep:
  move_threshold_bps ∈ {3, 4, 5, 6, 7}
  move_lookback_bars ∈ {1, 2, 3}  (1–3 minutes)

Run from repo root:
    python src/backtest_rf_dir_1min.py           # MES
    python src/backtest_rf_dir_1min.py MNQ
    python src/backtest_rf_dir_1min.py MES single   # default config only
"""

import sys
import sqlite3
import numpy as np
import pandas as pd
from copy import copy
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ET         = ZoneInfo("America/New_York")
DB_PATH    = Path("data/bars.db")
HIST_CSVS  = {"MES": Path("mes_hist_1min.csv"), "MNQ": Path("mnq_hist_1min.csv")}

PRE_START    = dtime(8, 30)
RTH_START    = dtime(9, 30)
RTH_END      = dtime(16, 0)
BLACKOUT_END = dtime(9, 40)
SESSION_END  = dtime(15, 45)

LOOKBACK     = 20    # 1-min lagged returns (20 min of context)
VOL_WINDOW   = 10    # bars for realized vol / ATR
VOL_PCT_WIN  = 60    # bars for vol_regime percentile
TRAIN_FRAC   = 0.75
TRAIN_HORIZON = 5    # bars (5 min) to resolve TP/SL label

PROB_THRESHOLD = 0.65
VOL_REGIME_MIN = 0.50

WINDOWS = [
    ("9:40–11:00",  dtime(9, 40),  dtime(11,  0)),
    ("11:00–13:00", dtime(11,  0), dtime(13,  0)),
    ("13:00–15:45", dtime(13,  0), dtime(15, 45)),
]

# Default configs (matching live bot bps settings)
DEFAULT_CONFIGS = {
    "MES": dict(target_bps=16.0, stop_bps=9.0,
                move_threshold_bps=4.0, move_lookback_bars=2,
                tick_size=0.25, tick_val=1.25),
    "MNQ": dict(target_bps=17.0, stop_bps=11.0,
                move_threshold_bps=4.0, move_lookback_bars=1,
                tick_size=0.25, tick_val=0.50),
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_1min(sym: str) -> pd.DataFrame:
    frames = []
    hist = HIST_CSVS[sym]
    if hist.exists():
        df_h = pd.read_csv(hist, parse_dates=["ts"])
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= PRE_START) & (t < RTH_END)]
        frames.append(df_h)
        print(f"  Historical CSV: {len(df_h):,} bars  "
              f"({df_h['ts'].dt.date.min()} → {df_h['ts'].dt.date.max()})",
              flush=True)
    db = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql(
        f"SELECT ts, open, high, low, close, volume "
        f"FROM bars WHERE symbol='{sym}' AND minutes=1 ORDER BY ts", db)
    db.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t = df_db["ts"].dt.time
    df_db = df_db[(t >= PRE_START) & (t < RTH_END)]
    frames.append(df_db)
    print(f"  Live DB: {len(df_db):,} bars  "
          f"({df_db['ts'].dt.date.min()} → {df_db['ts'].dt.date.max()})",
          flush=True)
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True))
    print(f"  Combined: {len(df):,} bars", flush=True)
    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df1: pd.DataFrame) -> pd.DataFrame:
    """Build 1-min bar feature matrix with warmup rows intact."""
    df = df1.copy()
    df["date"] = df["ts"].dt.date

    # 1-min log returns + lags
    df["lret"] = np.log(df["close"] / df["close"].shift(1))
    df.loc[df["date"] != df["date"].shift(1), "lret"] = np.nan
    for lag in range(1, LOOKBACK + 1):
        s = df["lret"].shift(lag)
        s[df["date"] != df["date"].shift(lag)] = np.nan
        df[f"r1m_{lag}"] = s

    # 9:30 open per day → ret_open
    open_930 = (df[df["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df = df.join(open_930, on="date")
    df["ret_open"] = np.log(df["close"] / df["open_930"])

    # Volatility features (rolling within day, lag-1 to avoid lookahead)
    df["realized_vol"] = (df.groupby("date")["lret"]
                           .transform(lambda x:
                               x.shift(1).rolling(VOL_WINDOW, min_periods=5).std()))
    df["range_1m"] = df["high"] - df["low"]
    avg_range = (df.groupby("date")["range_1m"]
                   .transform(lambda x:
                       x.shift(1).rolling(VOL_WINDOW, min_periods=5).mean()))
    df["atr_ratio"] = df["range_1m"] / avg_range.clip(lower=0.01)
    df["vol_regime"] = (df.groupby("date")["realized_vol"]
                          .transform(lambda x:
                              x.shift(1).rolling(VOL_PCT_WIN, min_periods=20)
                               .rank(pct=True)))

    return df


def get_feature_cols() -> list[str]:
    return ([f"r1m_{i}" for i in range(1, LOOKBACK + 1)]
            + ["ret_open", "realized_vol", "atr_ratio", "vol_regime",
               "move_size_bps"])


def pts_from_bps(price: float, bps: float, tick_size: float = 0.25) -> float:
    raw = price * bps / 10000.0
    ticks = int(raw / tick_size) + (1 if raw % tick_size > 0 else 0)
    return round(ticks * tick_size, 4)


# ── Directional dataset ────────────────────────────────────────────────────────

def build_directional_dataset(df: pd.DataFrame, cfg: dict,
                               horizon: int = TRAIN_HORIZON) -> pd.DataFrame:
    """
    Only label bars at/after 9:40 where a qualifying momentum move exists.
    Features sign-flipped by direction. Inconclusive bars skipped.
    """
    # Only label RTH (ret_open defined), after blackout
    df_rth = df[(df["ts"].dt.time >= BLACKOUT_END) &
                (df["ts"].dt.time < SESSION_END)].copy()

    closes = df["close"].values        # full df for lookback
    highs  = df["high"].values
    lows   = df["low"].values
    dates  = df["date"].values

    # Map ts → index in full df for lookback access
    full_idx = {ts: i for i, ts in enumerate(df["ts"])}

    r1m_cols = [f"r1m_{i}" for i in range(1, LOOKBACK + 1)]
    lb   = cfg["move_lookback_bars"]
    tgt_bps = cfg["target_bps"]
    stp_bps = cfg["stop_bps"]
    mthr_bps = cfg["move_threshold_bps"]

    rows = []
    for _, row in df_rth.iterrows():
        i = full_idx.get(row["ts"])
        if i is None or i < lb or i + horizon >= len(df):
            continue

        # Same-day lookback check
        if dates[i - lb] != dates[i]:
            continue

        # Qualifying move gate
        move = closes[i] - closes[i - lb]
        thr  = closes[i] * mthr_bps / 10000.0
        if abs(move) < thr:
            continue

        direction = 1 if move > 0 else -1

        # Forward window same day
        fwd = dates[i + 1 : i + 1 + horizon]
        if len(fwd) < horizon or fwd[0] != fwd[-1] or fwd[0] != dates[i]:
            continue

        # Directional TP/SL label
        entry = closes[i]
        tgt   = pts_from_bps(entry, tgt_bps)
        stp   = pts_from_bps(entry, stp_bps)

        hit_target = hit_stop = False
        for j in range(i + 1, i + 1 + horizon):
            if direction == 1:
                if highs[j] >= entry + tgt:  hit_target = True; break
                if lows[j]  <= entry - stp:  hit_stop   = True; break
            else:
                if lows[j]  <= entry - tgt:  hit_target = True; break
                if highs[j] >= entry + stp:  hit_stop   = True; break

        if not hit_target and not hit_stop:
            continue

        move_bps = abs(move) / closes[i] * 10000.0
        r = {}
        for c in r1m_cols:
            r[c] = row[c] * direction if pd.notna(row.get(c)) else np.nan
        r["ret_open"]     = row["ret_open"] * direction if pd.notna(row.get("ret_open")) else np.nan
        r["realized_vol"] = row["realized_vol"]
        r["atr_ratio"]    = row["atr_ratio"]
        r["vol_regime"]   = row["vol_regime"]
        r["move_size_bps"] = move_bps
        r["close"]        = entry
        r["high"]         = highs[i]
        r["low"]          = lows[i]
        r["date"]         = dates[i]
        r["ts"]           = row["ts"]
        r["direction"]    = direction
        r["target"]       = int(hit_target)
        rows.append(r)

    return pd.DataFrame(rows).dropna(subset=get_feature_cols())


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate(test: pd.DataFrame, clf, fcols: list[str], cfg: dict) -> pd.DataFrame:
    feat_arr = test[fcols].values
    proba    = clf.predict_proba(feat_arr)[:, 1]
    rows     = test.reset_index(drop=True)
    trades   = []
    in_trade = False

    for i, row in rows.iterrows():
        bar_t = row["ts"].time()

        if in_trade:
            if row["date"] != trades[-1]["date"] or bar_t >= SESSION_END:
                trades[-1].update(outcome="EXPIRED", pnl_pts=0.0)
                in_trade = False
            else:
                h, l  = row["high"], row["low"]
                entry = trades[-1]["entry"]
                tgt   = trades[-1]["tgt"]
                stp   = trades[-1]["stp"]
                d     = trades[-1]["dir_int"]
                if d == 1:
                    if h >= entry + tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt); in_trade=False; continue
                    if l <= entry - stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp); in_trade=False; continue
                else:
                    if l <= entry - tgt:
                        trades[-1].update(outcome="TARGET", pnl_pts=tgt); in_trade=False; continue
                    if h >= entry + stp:
                        trades[-1].update(outcome="STOPPED", pnl_pts=-stp); in_trade=False; continue
                continue

        if bar_t < BLACKOUT_END or bar_t >= SESSION_END:
            continue
        if proba[i] < PROB_THRESHOLD or row["vol_regime"] < VOL_REGIME_MIN:
            continue

        d     = int(row["direction"])
        entry = row["close"]
        tgt   = pts_from_bps(entry, cfg["target_bps"])
        stp   = pts_from_bps(entry, cfg["stop_bps"])

        trades.append({
            "ts": row["ts"], "date": row["date"], "bar_time": bar_t,
            "direction": "LONG" if d == 1 else "SHORT", "dir_int": d,
            "entry": entry, "tgt": tgt, "stp": stp,
            "high": row["high"], "low": row["low"],
            "prob": proba[i], "outcome": None, "pnl_pts": None,
        })
        in_trade = True

    if not trades:
        return pd.DataFrame()
    df_t = pd.DataFrame(trades)
    if "outcome" in df_t.columns:
        df_t = df_t.dropna(subset=["outcome"])
    return df_t.reset_index(drop=True)


# ── Reporting ──────────────────────────────────────────────────────────────────

def report_window(label: str, trades: pd.DataFrame, tick_val: float):
    n = len(trades)
    if n == 0:
        print(f"  {label:<14}  —")
        return
    wins    = (trades["outcome"] == "TARGET").sum()
    expired = (trades["outcome"] == "EXPIRED").sum()
    wr      = wins / n * 100
    pnl     = trades["pnl_pts"].sum()
    avg     = trades["pnl_pts"].mean()
    usd     = pnl * tick_val * 4
    xs      = f"  exp={expired}" if expired else ""
    print(f"  {label:<14}  n={n:4d}  WR={wr:5.1f}%  "
          f"avg={avg:+.2f}pt  total={pnl:+.1f}pt  (~${usd:+,.0f}){xs}")


def run_config(sym: str, df_feat: pd.DataFrame, cfg: dict,
               cutoff, test_days: int) -> dict | None:
    fcols  = get_feature_cols()
    df_dir = build_directional_dataset(df_feat, cfg)
    if len(df_dir) < 300:
        return None

    dates  = sorted(df_dir["date"].unique())
    cutoff = dates[int(len(dates) * TRAIN_FRAC)]
    train  = df_dir[df_dir["date"] <  cutoff]
    test   = df_dir[df_dir["date"] >= cutoff]

    if len(train) < 200 or len(test) < 50:
        return None

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(train[fcols].values, train["target"].astype(int).values)
    proba = clf.predict_proba(test[fcols].values)[:, 1]
    auc   = roc_auc_score(test["target"].astype(int).values, proba)

    tick_val = cfg["tick_val"]
    trades   = simulate(test, clf, fcols, cfg)
    n        = len(trades)
    if n == 0:
        return {"auc": auc, "n": 0, **cfg}

    test_days_actual = test["date"].nunique()
    pnl  = trades["pnl_pts"].sum()
    wr   = (trades["outcome"] == "TARGET").mean() * 100
    usd  = pnl * tick_val * 4

    print(f"  move≥{cfg['move_threshold_bps']:.0f}bp/{cfg['move_lookback_bars']}bar  "
          f"AUC={auc:.4f}  base={test['target'].mean()*100:.1f}%  "
          f"n_dir={len(df_dir):,}  →  "
          f"trades={n} ({n/test_days_actual:.1f}/day)  WR={wr:.1f}%  "
          f"P&L={pnl:+.1f}pt (~${usd:+,.0f})", flush=True)

    return {
        "move_thr": cfg["move_threshold_bps"],
        "move_lb":  cfg["move_lookback_bars"],
        "auc":      auc,
        "n":        n,
        "per_day":  n / test_days_actual,
        "wr":       wr,
        "pnl":      pnl,
        "usd":      usd,
        "trades":   trades,
        "test_days": test_days_actual,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    sym  = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "sweep"

    if sym not in DEFAULT_CONFIGS:
        print(f"Unknown symbol {sym}"); sys.exit(1)

    base_cfg  = DEFAULT_CONFIGS[sym]
    tick_val  = base_cfg["tick_val"]

    print(f"\n=== 1-Min Directional RF Backtest  ·  {sym} ===\n")
    print("Loading 1-min bars …", flush=True)
    df1 = load_1min(sym)

    print("\nBuilding features …", end=" ", flush=True)
    df_feat = build_features(df1)
    # Keep all bars including pre-market for lookback; labels only assigned post-9:40
    print(f"{len(df_feat):,} bars total", flush=True)

    if mode == "single":
        configs = [base_cfg]
    else:
        configs = [
            {**base_cfg,
             "move_threshold_bps": thr,
             "move_lookback_bars": lb}
            for thr in [3.0, 4.0, 5.0, 6.0, 7.0]
            for lb  in [1, 2, 3]
        ]

    print(f"\n── Sweep ({len(configs)} configs) ───────────────────────────────────────────")
    results = []
    best_trades = None
    best_pnl = -999999

    for cfg in configs:
        r = run_config(sym, df_feat, cfg, None, 0)
        if r and r["n"] > 0:
            results.append(r)
            if r["pnl"] > best_pnl:
                best_pnl    = r["pnl"]
                best_trades = r["trades"]
                best_r      = r

    if not results:
        print("No results."); return

    # Summary table
    print(f"\n── Sweep Summary (sorted by P&L) {'─'*40}")
    df_r = pd.DataFrame([{k: v for k, v in r.items() if k != "trades"}
                          for r in results]).sort_values("pnl", ascending=False)
    pd.set_option("display.width", 130)
    print(df_r[["move_thr","move_lb","n","per_day","wr","pnl","usd","auc"]]
          .to_string(index=False, float_format="{:.2f}".format))

    # TOD breakdown for best config
    print(f"\n── Best Config Detail: "
          f"move≥{best_r['move_thr']:.0f}bp/{best_r['move_lb']:.0f}bar  "
          f"({best_r['test_days']} test days) ──────────────────")
    trades = best_trades
    total_wr  = (trades["outcome"] == "TARGET").mean() * 100
    print(f"  Overall: n={len(trades)}  ({len(trades)/best_r['test_days']:.1f}/day)  "
          f"WR={total_wr:.1f}%  P&L={best_pnl:+.1f}pt (~${best_pnl*tick_val*4:+,.0f})")

    print(f"\n  By Time Window:")
    for label, t0, t1 in WINDOWS:
        mask = (trades["bar_time"] >= t0) & (trades["bar_time"] < t1)
        report_window(label, trades[mask], tick_val)

    print(f"\n  By Day-of-Week:")
    trades = trades.copy()
    trades["dow"] = pd.to_datetime(trades["date"]).dt.strftime("%a")
    for dow in ["Mon","Tue","Wed","Thu","Fri"]:
        report_window(dow, trades[trades["dow"]==dow], tick_val)

    print(f"\n  Monthly P&L:")
    trades["month"] = pd.to_datetime(trades["date"]).dt.to_period("M")
    monthly = trades.groupby("month").agg(
        n=("pnl_pts","count"),
        wr=("outcome", lambda x: (x=="TARGET").mean()*100),
        pnl=("pnl_pts","sum"),
    ).reset_index()
    for _, row in monthly.iterrows():
        bar = "█" * max(0, int(row["pnl"] / 2))
        neg = "░" * max(0, int(-row["pnl"] / 2))
        print(f"  {str(row['month']):<8}  n={int(row['n']):3d}  "
              f"WR={row['wr']:5.1f}%  pnl={row['pnl']:+6.1f}pt  {bar}{neg}")
    print()


if __name__ == "__main__":
    main()
