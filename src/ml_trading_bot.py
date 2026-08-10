"""
ml_trading_bot.py — Random Forest-based trading bot (MES and MNQ).

Strategy
--------
Model predicts probability that price crosses ±X pts in the next 2-min bar.
AUC ~0.855 (MES, high-vol regime) / ~0.850 (MNQ) on 7 years of RTH data.

Entry conditions (all must be met):
  1. vol_regime >= VOL_REGIME_MIN (default 0.50 — medium/high vol only)
  2. model prob >= PROB_THRESHOLD (default 0.65)
  3. Time between 09:40 and 15:45 ET
  4. No existing position

Direction:
  • LONG  — r2m_1 > 0 AND ret_open > 0  (momentum and trend agree)
  • SHORT — r2m_1 < 0 AND ret_open < 0
  • No trade if they disagree

Execution:
  • Market order with native API bracket (stop_loss + take_profit ticks)
  • Bracket is exchange-managed OCO — no manual tracking needed

Per-symbol parameters (TARGET_PTS / STOP_PTS / threshold scaled to volatility):
  MES : threshold=5pt   target=5pt  stop=3pt   tick=0.25  (AUC 0.855)
  MNQ : threshold=25pt  target=20pt stop=12pt  tick=0.25  (AUC 0.850)

Logs trades to logs/ml_trades_{symbol}.csv.
Writes logs/ml_state_{symbol}.json every poll cycle for ml_monitor to read.

Safety: hardcoded practice account — refuses to run against any other account.

Usage:
    python src/ml_trading_bot.py                      # MES live (practice account)
    python src/ml_trading_bot.py --paper              # MES paper mode
    python src/ml_trading_bot.py --symbol MNQ         # MNQ live
    python src/ml_trading_bot.py --symbol MNQ --paper # MNQ paper
"""

import argparse
import csv
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()
from topstep_client import TopstepClient

# ── Constants ──────────────────────────────────────────────────────────────────

PRACTICE_ACCOUNT_ID   = int(os.environ.get("TOPSTEP_ACCOUNT_ID", "10634862"))
PRACTICE_ACCOUNT_NAME = "PRAC-V2-88916-19336808"

BLACKOUT_END = dtime(9,  40)
SESSION_END  = dtime(15, 45)

# Feature params — same for all symbols
LOOKBACK    = 15   # 2-min bars
SHORT_N     = 6    # 1-min bars
VOL_WINDOW  = 10   # bars for realized vol / ATR
VOL_PCT_WIN = 60   # bars for vol_regime percentile

DB_PATH = Path("data/bars.db")

# Shared gate thresholds
PROB_THRESHOLD = 0.65
VOL_REGIME_MIN = 0.50

BID = 1   # TopstepX side constants
ASK = 2


@dataclass
class SymbolConfig:
    symbol:        str
    tick_size:     float
    point_value:   float   # $ per point per contract
    hist_csv:      Path
    threshold_bps: float   # RF model target threshold in basis points (training)
    target_bps:    float   # live profit target in basis points
    stop_bps:      float   # live stop loss in basis points

    def pts_from_bps(self, price: float, bps: float) -> float:
        """Convert bps to points, rounded UP to the nearest tick."""
        raw = price * bps / 10000.0
        ticks = int(raw / self.tick_size) + (1 if raw % self.tick_size > 0 else 0)
        return round(ticks * self.tick_size, 4)

    @property
    def state_file(self) -> Path:
        return Path(f"logs/ml_state_{self.symbol}.json")

    @property
    def trade_log(self) -> Path:
        return Path(f"logs/ml_trades_{self.symbol}.csv")


SYMBOL_CONFIGS: dict[str, SymbolConfig] = {
    "MES": SymbolConfig(
        symbol        = "MES",
        tick_size     = 0.25,
        point_value   = 5.0,
        hist_csv      = Path("mes_hist_1min.csv"),
        threshold_bps = 9.0,    # AUC=0.867 @ ±9bp H=1
        target_bps    = 16.0,   # sweep optimum: E=+0.94bp, WR=39%, R:R=1.8
        stop_bps      = 9.0,    # (≈9pt / 5.25pt at MES ~5600)
    ),
    "MNQ": SymbolConfig(
        symbol        = "MNQ",
        tick_size     = 0.25,
        point_value   = 2.0,
        hist_csv      = Path("mnq_hist_1min.csv"),
        threshold_bps = 13.0,   # AUC=0.865 @ ±13bp H=1
        target_bps    = 17.0,   # sweep optimum: E=+1.12bp, WR=43%, R:R=1.5
        stop_bps      = 11.0,   # (≈33.75pt / 22pt at MNQ ~19800)
    ),
}

ET        = ZoneInfo("America/New_York")
RTH_START = dtime(9, 30)
RTH_END   = dtime(16, 0)

POLL_S    = 3   # seconds between bar polls

BID = 1   # TopstepX side constants
ASK = 2

# ── Logging ────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[
        logging.FileHandler("logs/ml_trading_bot.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ml_bot")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Position:
    direction:  str         # "LONG" or "SHORT"
    entry_pts:  float       # entry price in points
    entry_ts:   str         # ISO timestamp
    prob:       float       # model probability at entry
    vol_regime: float
    order_id:   int | None = None
    bars_held:  int = 0


# ── Feature engineering (matches backtest_rf_feat_v3.py) ──────────────────────

def load_training_data(cfg: SymbolConfig) -> pd.DataFrame:
    """Load RTH 1-min bars from historical CSV + live DB, deduplicated."""
    frames = []
    if cfg.hist_csv.exists():
        log.info(f"Loading historical CSV: {cfg.hist_csv}")
        df_h = pd.read_csv(cfg.hist_csv, parse_dates=["ts"])
        df_h["ts"] = pd.to_datetime(df_h["ts"], utc=True).dt.tz_convert(ET)
        t = df_h["ts"].dt.time
        df_h = df_h[(t >= RTH_START) & (t < RTH_END)]
        frames.append(df_h)
        log.info(f"  Historical: {len(df_h):,} bars  "
                 f"({df_h['ts'].dt.date.min()} → {df_h['ts'].dt.date.max()})")
    db = sqlite3.connect(DB_PATH)
    df_db = pd.read_sql(
        f"SELECT ts, open, high, low, close, volume "
        f"FROM bars WHERE symbol='{cfg.symbol}' AND minutes=1 ORDER BY ts", db)
    db.close()
    df_db["ts"] = pd.to_datetime(df_db["ts"], utc=True).dt.tz_convert(ET)
    t = df_db["ts"].dt.time
    df_db = df_db[(t >= RTH_START) & (t < RTH_END)]
    frames.append(df_db)
    log.info(f"  Live DB: {len(df_db):,} bars  "
             f"({df_db['ts'].dt.date.min()} → {df_db['ts'].dt.date.max()})")
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates(subset="ts")
            .sort_values("ts")
            .reset_index(drop=True))
    log.info(f"  Combined: {len(df):,} bars")
    return df


def build_2min_features(df1: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """
    Build full 2-min feature DataFrame from 1-min RTH bars.
    Matches feature engineering in backtest_rf_feat_v3.py.
    """
    df1 = df1.copy()
    df1["date"] = df1["ts"].dt.date

    # 1-min log returns
    df1["lret1"] = np.log(df1["close"] / df1["close"].shift(1))
    df1.loc[df1["date"] != df1["date"].shift(1), "lret1"] = np.nan
    for lag in range(1, SHORT_N + 1):
        s = df1["lret1"].shift(lag)
        s[df1["date"] != df1["date"].shift(lag)] = np.nan
        df1[f"r1m_{lag}"] = s

    # 9:30 open per day
    open_930 = (df1[df1["ts"].dt.time == RTH_START]
                .groupby("date")["open"].first().rename("open_930"))
    df1 = df1.join(open_930, on="date")
    df1["ret_open"] = np.log(df1["close"] / df1["open_930"])

    # Resample to 2-min
    df1_idx = df1.set_index("ts")
    df2 = df1_idx[["open","high","low","close","volume"]].resample(
        "2min", label="right", closed="right"
    ).agg(open=("open","first"), high=("high","max"),
          low=("low","min"),   close=("close","last"),
          volume=("volume","sum")).dropna(subset=["close"])
    df2.index = df2.index.tz_convert(ET)
    t2 = df2.index.time
    df2 = df2[(t2 >= RTH_START) & (t2 < RTH_END)].reset_index()
    df2["date"] = df2["ts"].dt.date

    # 2-min log returns
    df2["lret2"] = np.log(df2["close"] / df2["close"].shift(1))
    df2.loc[df2["date"] != df2["date"].shift(1), "lret2"] = np.nan
    for lag in range(1, LOOKBACK):
        s = df2["lret2"].shift(lag)
        s[df2["date"] != df2["date"].shift(lag)] = np.nan
        df2[f"r2m_{lag}"] = s

    # Volatility features
    df2["realized_vol"] = (df2.groupby("date")["lret2"]
                             .transform(lambda x: x.shift(1)
                                        .rolling(VOL_WINDOW, min_periods=5).std()))
    df2["range_2m"] = df2["high"] - df2["low"]
    avg_range = (df2.groupby("date")["range_2m"]
                   .transform(lambda x: x.shift(1)
                              .rolling(VOL_WINDOW, min_periods=5).mean()))
    df2["atr_ratio"] = df2["range_2m"] / avg_range.clip(lower=0.01)
    df2["vol_regime"] = (df2.groupby("date")["realized_vol"]
                           .transform(lambda x:
                               x.shift(1).rolling(VOL_PCT_WIN, min_periods=20)
                                .rank(pct=True)))

    # Merge 1-min short-term features
    r1m_cols    = [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
    scalar_cols = ["ret_open"]
    df1_small   = df1[["ts"] + r1m_cols + scalar_cols].sort_values("ts")
    df2 = df2.sort_values("ts")
    df2 = pd.merge_asof(df2, df1_small, on="ts", direction="backward")

    r2m_cols  = [f"r2m_{i}" for i in range(1, LOOKBACK)]
    vola_cols = ["realized_vol", "atr_ratio", "vol_regime"]
    all_need  = r2m_cols + r1m_cols + ["ret_open"] + vola_cols
    df2 = df2.dropna(subset=all_need)
    if drop_warmup:
        df2 = df2[df2.groupby("date").cumcount() >= LOOKBACK * 2].reset_index(drop=True)
    else:
        df2 = df2.reset_index(drop=True)
    return df2


def get_feature_cols() -> list[str]:
    return ([f"r2m_{i}" for i in range(1, LOOKBACK)]
            + [f"r1m_{i}" for i in range(1, SHORT_N + 1)]
            + ["ret_open", "realized_vol", "atr_ratio", "vol_regime"])


def add_targets(df2: pd.DataFrame, threshold_bps: float = 9.0,
                horizon: int = 1) -> pd.DataFrame:
    highs  = df2["high"].values
    lows   = df2["low"].values
    closes = df2["close"].values
    dates  = df2["date"].values
    n      = len(df2)
    targets = np.full(n, np.nan)
    for i in range(n - horizon):
        fd = dates[i + 1 : i + 1 + horizon]
        if len(fd) < horizon or fd[0] != fd[-1] or fd[0] != dates[i]:
            continue
        thr = closes[i] * threshold_bps / 10000.0
        targets[i] = int(highs[i+1:i+1+horizon].max() >= closes[i] + thr or
                         lows[i+1:i+1+horizon].min()  <= closes[i] - thr)
    df2 = df2.copy()
    df2["target"] = targets
    return df2.dropna(subset=["target"])


# ── Model training ─────────────────────────────────────────────────────────────

def train_model(df1: pd.DataFrame, cfg: SymbolConfig) -> tuple[RandomForestClassifier, list[str], dict]:
    """Train RF model. Returns (clf, feature_cols, training_info)."""
    log.info("Building 2-min feature matrix …")
    df2 = build_2min_features(df1)
    df2_tgt = add_targets(df2, threshold_bps=cfg.threshold_bps, horizon=1)

    fcols = get_feature_cols()
    dates = sorted(df2_tgt["date"].unique())
    cutoff = dates[int(len(dates) * 0.75)]

    train = df2_tgt[df2_tgt["date"] < cutoff]
    test  = df2_tgt[df2_tgt["date"] >= cutoff]

    log.info(f"Training RF: {len(train):,} train / {len(test):,} test  "
             f"(split at {cutoff})")

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    clf.fit(train[fcols].values, train["target"].astype(int).values)

    proba = clf.predict_proba(test[fcols].values)[:, 1]
    auc   = roc_auc_score(test["target"].astype(int).values, proba)
    base  = test["target"].mean()

    log.info(f"Model trained: AUC={auc:.4f}  base_rate={base*100:.1f}%  "
             f"features={len(fcols)}")

    info = {
        "n_train":         len(train),
        "n_test":          len(test),
        "train_cutoff":    str(cutoff),
        "date_range_from": str(df2_tgt["date"].min()),
        "date_range_to":   str(df2_tgt["date"].max()),
        "auc":             round(auc, 4),
        "base_rate":       round(float(base), 4),
        "symbol":          cfg.symbol,
        "threshold_bps":   cfg.threshold_bps,
        "target_bps":      cfg.target_bps,
        "stop_bps":        cfg.stop_bps,
        "horizon_bars":    1,
    }
    return clf, fcols, info


# ── Live feature computation ───────────────────────────────────────────────────

def load_live_1min(cfg: SymbolConfig, limit: int = 600) -> pd.DataFrame:
    """Load recent 1-min bars from bars.db for the configured symbol."""
    db = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT ts, open, high, low, close, volume "
        f"FROM bars WHERE symbol='{cfg.symbol}' AND minutes=1 "
        f"ORDER BY ts DESC LIMIT {limit}", db)
    db.close()
    if df.empty:
        return df
    df = df.iloc[::-1].reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    t = df["ts"].dt.time
    return df[(t >= RTH_START) & (t < RTH_END)].reset_index(drop=True)


def compute_live_features(df1_buf: pd.DataFrame) -> dict | None:
    """
    Compute features for the most recent complete 2-min bar.
    Returns feature dict or None if insufficient data.
    """
    if len(df1_buf) < LOOKBACK * 2 + SHORT_N + 5:
        return None

    try:
        df2 = build_2min_features(df1_buf, drop_warmup=False)
        if df2.empty:
            return None

        fcols = get_feature_cols()
        needed = fcols + ["ts", "close", "high", "low", "date",
                          "lret2", "realized_vol", "atr_ratio", "vol_regime",
                          "ret_open", "r2m_1", "r1m_1"]
        row = df2.iloc[-1]

        # Check we have valid features
        for c in fcols:
            if pd.isna(row.get(c)):
                return None

        feat = {c: float(row[c]) for c in fcols}
        feat["ts"]           = row["ts"].isoformat()
        feat["close"]        = float(row["close"])
        feat["high"]         = float(row["high"])
        feat["low"]          = float(row["low"])
        feat["date"]         = str(row["date"])
        feat["lret2"]        = float(row.get("lret2", 0) or 0)
        feat["realized_vol"] = float(row["realized_vol"])
        feat["atr_ratio"]    = float(row["atr_ratio"])
        feat["vol_regime"]   = float(row["vol_regime"])
        feat["ret_open"]     = float(row["ret_open"])
        feat["r2m_1"]        = float(row["r2m_1"])
        feat["r1m_1"]        = float(row.get("r1m_1", 0) or 0)

        # All 14 r2m and 6 r1m as lists for monitor display
        feat["r2m_series"] = [float(row.get(f"r2m_{i}", 0) or 0)
                               for i in range(1, LOOKBACK)]
        feat["r1m_series"] = [float(row.get(f"r1m_{i}", 0) or 0)
                               for i in range(1, SHORT_N + 1)]
        return feat

    except Exception as e:
        log.debug(f"compute_live_features error: {e}")
        return None


# ── Signal logic ───────────────────────────────────────────────────────────────

def get_signal(feat: dict, clf: RandomForestClassifier,
               fcols: list[str], cfg: SymbolConfig) -> dict:
    """
    Compute model probability and determine direction.
    Returns signal dict with prob, direction, and reason.
    """
    X = np.array([[feat[c] for c in fcols]])
    prob = float(clf.predict_proba(X)[0, 1])

    vol_regime = feat["vol_regime"]
    r2m_1      = feat["r2m_1"]
    ret_open   = feat["ret_open"]

    # Direction: r2m_1 and ret_open must agree
    if r2m_1 > 0 and ret_open > 0:
        direction = "LONG"
        reason    = f"r2m_1>0 ret_open>0"
    elif r2m_1 < 0 and ret_open < 0:
        direction = "SHORT"
        reason    = f"r2m_1<0 ret_open<0"
    else:
        direction = None
        reason    = f"r2m_1/ret_open conflict"

    # Gate checks
    gates_pass = (prob >= PROB_THRESHOLD and
                  vol_regime >= VOL_REGIME_MIN and
                  direction is not None)

    return {
        "prob":        prob,
        "direction":   direction,
        "reason":      reason,
        "gates_pass":  gates_pass,
        "prob_ok":     prob >= PROB_THRESHOLD,
        "vol_ok":      vol_regime >= VOL_REGIME_MIN,
        "dir_ok":      direction is not None,
    }


# ── State file ─────────────────────────────────────────────────────────────────

def write_state(feat: dict | None, signal: dict | None,
                position: Position | None,
                model_info: dict, session_stats: dict,
                bar_ts: str | None, paper: bool,
                cfg: SymbolConfig,
                last_close: float | None = None):
    state = {
        "updated_at":  datetime.now(ET).isoformat(),
        "bar_ts":      bar_ts,
        "paper":       paper,
        "symbol":      cfg.symbol,
        "last_close":  feat["close"] if feat else last_close,
        "feat":        feat,
        "signal":     signal,
        "position":   {
            "direction":  position.direction,
            "entry_pts":  position.entry_pts,
            "entry_ts":   position.entry_ts,
            "prob":       position.prob,
            "vol_regime": position.vol_regime,
            "bars_held":  position.bars_held,
        } if position else None,
        "model_info":    model_info,
        "session_stats": session_stats,
        "thresholds": {
            "prob":       PROB_THRESHOLD,
            "vol_regime": VOL_REGIME_MIN,
            "target_bps": cfg.target_bps,
            "stop_bps":   cfg.stop_bps,
        },
    }
    cfg.state_file.parent.mkdir(exist_ok=True)
    cfg.state_file.write_text(json.dumps(state, indent=2))


# ── Trade log ──────────────────────────────────────────────────────────────────

TRADE_FIELDS = [
    "fired_at", "symbol", "direction", "entry", "target", "stop",
    "prob", "vol_regime", "r2m_1", "ret_open", "realized_vol",
    "exit_price", "outcome", "pnl_pts", "bars_held",
]

def init_trade_log(cfg: SymbolConfig):
    cfg.trade_log.parent.mkdir(exist_ok=True)
    if not cfg.trade_log.exists():
        with open(cfg.trade_log, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=TRADE_FIELDS).writeheader()

def append_trade(row: dict, cfg: SymbolConfig):
    with open(cfg.trade_log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_FIELDS, extrasaction="ignore")
        w.writerow(row)
    log.info(f"Trade logged: {row['direction']} {row['outcome']} "
             f"{row['pnl_pts']:+.2f} pts")


# ── Contract lookup ────────────────────────────────────────────────────────────

def get_contract_id(client: TopstepClient, cfg: SymbolConfig) -> str:
    contracts = client.search_contracts(cfg.symbol)
    if not contracts:
        raise RuntimeError(f"No {cfg.symbol} contract found")
    contracts.sort(key=lambda c: c.get("expiry", ""))
    cid = contracts[0]["id"]
    log.info(f"Using contract: {contracts[0].get('name', cid)}  id={cid}")
    return cid


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(cfg: SymbolConfig, paper: bool = False):
    log.info("=" * 60)
    log.info(f"ml_trading_bot starting  symbol={cfg.symbol}  paper={paper}")
    log.info(f"  threshold={cfg.threshold_bps}bp  target={cfg.target_bps}bp  "
             f"stop={cfg.stop_bps}bp")
    log.info("=" * 60)

    # ── Account setup ──────────────────────────────────────────────────────────
    client     = TopstepClient()
    client.use_shared_token()
    accounts   = client.get_accounts()
    account    = next((a for a in accounts if a["id"] == PRACTICE_ACCOUNT_ID), None)
    if account is None:
        raise RuntimeError(f"Practice account {PRACTICE_ACCOUNT_ID} not found")
    acct_name = account.get("name", "")
    if not paper and acct_name != PRACTICE_ACCOUNT_NAME:
        raise RuntimeError(
            f"LIVE TRADING BLOCKED — account '{acct_name}' is not "
            f"'{PRACTICE_ACCOUNT_NAME}'. Refusing to trade.")
    log.info(f"Account: {acct_name}  id={PRACTICE_ACCOUNT_ID}"
             + ("  [PAPER]" if paper else "  [LIVE]"))

    contract_id = get_contract_id(client, cfg)

    # ── Model training ─────────────────────────────────────────────────────────
    log.info("Loading training data …")
    df1_train  = load_training_data(cfg)
    clf, fcols, model_info = train_model(df1_train, cfg)
    del df1_train   # free memory

    init_trade_log(cfg)

    session_stats = {
        "signals": 0, "trades": 0, "wins": 0, "losses": 0,
        "pnl_pts": 0.0, "bars_seen": 0,
    }

    position: Position | None  = None
    last_bar_ts: str | None    = None
    last_signal_ts: str | None = None

    log.info("Entering main loop …")

    while True:
        try:
            now_et = datetime.now(ET)
            now_t  = now_et.time()

            # ── Outside RTH: sleep and reset ──────────────────────────────────
            if now_t >= RTH_END or now_t < RTH_START:
                write_state(None, None, None, model_info, session_stats,
                            last_bar_ts, paper, cfg)
                time.sleep(30)
                continue

            # ── Load latest bars ───────────────────────────────────────────────
            df1_buf = load_live_1min(cfg, limit=400)
            if df1_buf.empty:
                time.sleep(POLL_S)
                continue

            # ── Compute features on latest 2-min bar ───────────────────────────
            feat = compute_live_features(df1_buf)
            if feat is None:
                last_close = float(df1_buf["close"].iloc[-1]) if not df1_buf.empty else None
                write_state(None, None, position, model_info,
                            session_stats, last_bar_ts, paper, cfg,
                            last_close=last_close)
                time.sleep(POLL_S)
                continue

            bar_ts = feat["ts"]
            is_new_bar = (bar_ts != last_bar_ts)
            if is_new_bar:
                last_bar_ts = bar_ts
                session_stats["bars_seen"] += 1
                if position:
                    position.bars_held += 1

            # ── Get signal ─────────────────────────────────────────────────────
            signal = get_signal(feat, clf, fcols, cfg)

            # ── Position monitoring ────────────────────────────────────────────
            if position and is_new_bar:
                try:
                    open_pos  = client.get_open_positions(PRACTICE_ACCOUNT_ID)
                    open_pos_ = next(
                        (p for p in open_pos
                         if (p.get("contractId") == contract_id or
                             cfg.symbol in str(p.get("contractId", "")))),
                        None)
                except Exception:
                    open_pos_ = None  # assume still open on API error

                if open_pos_ is None:
                    # Bracket closed the position — infer outcome from bar extremes
                    entry   = position.entry_pts
                    is_long = position.direction == "LONG"
                    tgt     = cfg.pts_from_bps(entry, cfg.target_bps)
                    stp     = cfg.pts_from_bps(entry, cfg.stop_bps)

                    if is_long:
                        if feat["high"] >= entry + tgt:
                            outcome, exit_px, pnl = "TARGET",  entry + tgt,  tgt
                        else:
                            outcome, exit_px, pnl = "STOPPED", entry - stp, -stp
                    else:
                        if feat["low"] <= entry - tgt:
                            outcome, exit_px, pnl = "TARGET",  entry - tgt,  tgt
                        else:
                            outcome, exit_px, pnl = "STOPPED", entry + stp, -stp

                    session_stats["trades"]  += 1
                    session_stats["pnl_pts"] += pnl
                    if pnl > 0:
                        session_stats["wins"]   += 1
                    else:
                        session_stats["losses"] += 1

                    append_trade({
                        "fired_at":     position.entry_ts,
                        "symbol":       cfg.symbol,
                        "direction":    position.direction,
                        "entry":        entry,
                        "target":       entry + (tgt if is_long else -tgt),
                        "stop":         entry - (stp if is_long else -stp),
                        "prob":         round(position.prob, 4),
                        "vol_regime":   round(position.vol_regime, 4),
                        "r2m_1":        round(feat.get("r2m_1", 0), 6),
                        "ret_open":     round(feat.get("ret_open", 0), 6),
                        "realized_vol": round(feat.get("realized_vol", 0), 6),
                        "exit_price":   exit_px,
                        "outcome":      outcome,
                        "pnl_pts":      round(pnl, 2),
                        "bars_held":    position.bars_held,
                    }, cfg)
                    log.info(f"Position closed: {outcome}  pnl={pnl:+.2f}pts")
                    position = None

            # ── Entry logic ────────────────────────────────────────────────────
            if (position is None
                    and signal["gates_pass"]
                    and is_new_bar
                    and bar_ts != last_signal_ts
                    and BLACKOUT_END <= now_t < SESSION_END):

                direction = signal["direction"]
                close     = feat["close"]
                is_long   = direction == "LONG"
                side      = BID if is_long else ASK
                tgt_ticks = int(cfg.pts_from_bps(close, cfg.target_bps) / cfg.tick_size)
                stp_ticks = int(cfg.pts_from_bps(close, cfg.stop_bps)   / cfg.tick_size)

                session_stats["signals"] += 1
                last_signal_ts = bar_ts
                log.info(
                    f"SIGNAL {cfg.symbol} {direction}  close={close:.2f}  "
                    f"prob={signal['prob']:.3f}  "
                    f"vol_regime={feat['vol_regime']:.2f}  "
                    f"reason={signal['reason']}"
                    + ("  [PAPER]" if paper else ""))

                order_id = None
                if not paper:
                    try:
                        resp = client.place_order(
                            account_id        = PRACTICE_ACCOUNT_ID,
                            contract_id       = contract_id,
                            side              = side,
                            size              = 1,
                            stop_loss_ticks   = stp_ticks,
                            take_profit_ticks = tgt_ticks,
                            custom_tag        = f"ML_{cfg.symbol}_{direction[:1]}_{bar_ts[:16]}",
                        )
                        order_id = resp.get("orderId")
                        log.info(f"Order placed: id={order_id}")
                    except Exception as e:
                        log.error(f"Order failed: {e}")
                        write_state(feat, signal, None, model_info,
                                    session_stats, bar_ts, paper, cfg)
                        time.sleep(POLL_S)
                        continue

                position = Position(
                    direction  = direction,
                    entry_pts  = close,
                    entry_ts   = bar_ts,
                    prob       = signal["prob"],
                    vol_regime = feat["vol_regime"],
                    order_id   = order_id,
                    bars_held  = 0,
                )

            # ── Write state file for monitor ───────────────────────────────────
            write_state(feat, signal, position, model_info,
                        session_stats, bar_ts, paper, cfg)

        except KeyboardInterrupt:
            log.info("Shutting down.")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}", exc_info=True)

        time.sleep(POLL_S)


def main():
    parser = argparse.ArgumentParser(description="ML Trading Bot (MES or MNQ)")
    parser.add_argument("--symbol", choices=["MES", "MNQ"], default="MES",
                        help="Instrument to trade (default: MES)")
    parser.add_argument("--paper", action="store_true",
                        help="Paper mode: signals only, no orders placed")
    args = parser.parse_args()
    cfg  = SYMBOL_CONFIGS[args.symbol]
    run(cfg, paper=args.paper)


if __name__ == "__main__":
    main()
