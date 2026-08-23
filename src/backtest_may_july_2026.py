"""
backtest_may_july_2026.py — Unified strategy backtest for May–July 2026 on MES.

Strategies using live production parameters:
  CSR       — 5-min bars, ≥3σ bar + vol≥1.5× + CSR-40min≥1.5σ, 2σ stop / 3σ target
  ORB       — 1-min bars, 15-min range, LONG-only on gap-down, width 0.15-0.50%
  VWASLR    — 5-min bars, VWASLR(10) crosses ±0.4σ, 2σ stop / 3σ target
  SLR       — 20s bars (from 5sec), vol≥7×, move≥12bp, 10bp stop / 15bp target
  PL_REV    — 5sec bars, PL≥0.80 move≥20bp ADX(9-10ET)≤25, 15bp stop / 12bp target
  Wall Brk  — wall_events.csv, peak 100-300, test≥2, 4pt stop / 12pt target / 15-min hold

Usage:
    python src/backtest_may_july_2026.py
"""

import math
import sqlite3
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

DATE_START = date(2026, 5, 1)
DATE_END   = date(2026, 7, 31)

MES_PV  = 5.0   # $ per point
SETTLE  = (21, 22)  # CME settlement gap UTC hours

BARS_DB       = "data/bars.db"
ES_5SEC_CSV   = "es_hist_5sec.csv"
WALL_EVENTS   = "logs/wall_events.csv"


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_db_bars(minutes: int) -> pd.DataFrame:
    conn = sqlite3.connect(BARS_DB)
    df = pd.read_sql(
        f"SELECT ts,open,high,low,close,volume FROM bars "
        f"WHERE symbol='MES' AND minutes={minutes} ORDER BY ts", conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["ts_et"] = df["ts"].dt.tz_convert(ET)
    df["date"]  = df["ts_et"].dt.date
    df["hm"]    = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    # Remove settlement gap
    h = df["ts"].dt.hour
    df = df[~((h >= SETTLE[0]) & (h < SETTLE[1]))].copy()
    # Filter to date range
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].copy()
    return df.reset_index(drop=True)


def load_5sec() -> pd.DataFrame:
    df = pd.read_csv(ES_5SEC_CSV)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    h = df["ts"].dt.hour
    df = df[~((h >= SETTLE[0]) & (h < SETTLE[1]))].copy()
    df["ts_et"] = df["ts"].dt.tz_convert(ET)
    df["date"]  = df["ts_et"].dt.date
    df["hm"]    = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].copy()
    return df.reset_index(drop=True)


def resample_5sec(df5: pd.DataFrame, bar_s: int) -> pd.DataFrame:
    df = df5.set_index("ts")
    rs = df[["open","high","low","close","volume"]].resample(f"{bar_s}s").agg(
        open=("open","first"), high=("high","max"), low=("low","min"),
        close=("close","last"), volume=("volume","sum")
    ).dropna(subset=["close"])
    expected = pd.Timedelta(seconds=bar_s)
    rs["gap"] = rs.index.to_series().diff() > expected * 1.5
    rs.iloc[0, rs.columns.get_loc("gap")] = True
    rs = rs.reset_index()
    rs["ts_et"] = rs["ts"].dt.tz_convert(ET)
    rs["date"]  = rs["ts_et"].dt.date
    rs["hm"]    = rs["ts_et"].dt.hour * 60 + rs["ts_et"].dt.minute
    return rs


# ─── P&L summary helper ───────────────────────────────────────────────────────

def summarise(trades: list[dict], strat: str, unit: str = "bp") -> None:
    if not trades:
        print(f"\n  {strat}: no trades")
        return
    df = pd.DataFrame(trades)
    df["ym"] = df["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
    pnl_col = "pnl"

    n    = len(df)
    wr   = (df[pnl_col] > 0).mean()
    avg  = df[pnl_col].mean()
    tot  = df[pnl_col].sum()

    if unit == "bp":
        avg_price = df["entry"].mean() if "entry" in df.columns else 5400.0
        pts_per   = avg * avg_price / 10000
        dol_tot   = tot * avg_price / 10000 * MES_PV
        u_label   = f"avg {avg:+.2f}bp  total {tot:+.1f}bp"
        d_label   = f"≈${dol_tot:+,.0f}"
    else:
        pts_per = avg
        dol_tot = tot * MES_PV
        u_label = f"avg {avg:+.3f}pt  total {tot:+.1f}pt"
        d_label = f"≈${dol_tot:+,.0f}"

    print(f"\n  ─── {strat} ───────────────────────────")
    print(f"  Trades: {n}  WR: {wr:.1%}  {u_label}  {d_label}")

    monthly = df.groupby("ym")[pnl_col].agg(n="count", tot="sum",
                                              wins=lambda x: (x>0).sum())
    for ym, row in monthly.iterrows():
        if unit == "bp":
            avg_p2 = df.loc[df["ym"]==ym, "entry"].mean() if "entry" in df.columns else 5400.0
            d = row["tot"] * avg_p2 / 10000 * MES_PV
        else:
            d = row["tot"] * MES_PV
        wr2 = row["wins"] / row["n"] if row["n"] else 0
        bar = "█" * int(max(0, row["tot"]) / (0.5 if unit=="bp" else 1)) if unit=="bp" else ""
        print(f"    {ym}  n={int(row['n']):3d}  WR={wr2:.1%}  "
              f"tot={row['tot']:+7.2f}{unit}  ≈${d:+6,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CSR  —  5-min bars, 3σ momentum + CSR-40min gate
# ═══════════════════════════════════════════════════════════════════════════════

def run_csr(bars5: pd.DataFrame) -> list[dict]:
    MIN_SCALED   = 3.0
    VOL_RATIO    = 1.5
    CSR_WIN      = 8          # bars = 40 min
    CSR_THRESH   = 1.5
    TRAIL        = 20
    STOP_S       = 2.0
    TGT_S        = 3.0
    MAX_HOLD     = 3          # bars = 15 min
    RTH_HM       = (9*60, 16*60)

    # Use RTH 5-min bars only
    rth = bars5[(bars5["hm"] >= RTH_HM[0]) & (bars5["hm"] < RTH_HM[1])].copy()
    rth = rth.reset_index(drop=True)

    closes  = rth["close"].values.astype(float)
    highs   = rth["high"].values.astype(float)
    lows    = rth["low"].values.astype(float)
    volumes = rth["volume"].values.astype(float)
    dates   = rth["date"].values
    n       = len(rth)

    # Gap flag (session boundary)
    gap = (rth["ts"].diff() > pd.Timedelta(minutes=10)).values.copy()
    gap[0] = True

    LOOKBACK = max(TRAIL, CSR_WIN) + 2
    trades = []
    hold_until = -1

    for i in range(LOOKBACK, n - MAX_HOLD - 1):
        if i <= hold_until:
            continue
        # No gaps in trailing window
        if gap[i - TRAIL + 1: i + MAX_HOLD + 1].any():
            continue

        trail_rets = np.log(closes[i - TRAIL + 1: i + 1] /
                            closes[i - TRAIL    : i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        mean_vol  = volumes[i - TRAIL: i].mean()
        vol_ratio = volumes[i] / mean_vol if mean_vol > 0 else 0.0
        bar_ret   = math.log(closes[i] / closes[i - 1])
        scaled    = bar_ret / sigma

        if abs(scaled) < MIN_SCALED or vol_ratio < VOL_RATIO:
            continue

        direction = 1 if scaled > 0 else -1

        # CSR gate: direction-adjusted CSR over prior 8 bars
        prior_rets = np.log(closes[i - CSR_WIN: i] / closes[i - CSR_WIN - 1: i - 1])
        csr = prior_rets.sum() / sigma * direction
        if csr < CSR_THRESH:
            continue

        entry      = closes[i]
        tgt_price  = entry * math.exp( direction * TGT_S  * sigma)
        stop_price = entry * math.exp(-direction * STOP_S * sigma)

        hit_tgt = hit_stop = None
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != dates[i]:
                break
            h, l = highs[j], lows[j]
            if hit_tgt is None:
                if direction == 1 and h >= tgt_price:  hit_tgt  = j
                if direction == -1 and l <= tgt_price:  hit_tgt  = j
            if hit_stop is None:
                if direction == 1 and l <= stop_price:  hit_stop = j
                if direction == -1 and h >= stop_price:  hit_stop = j
            if hit_tgt or hit_stop:
                break

        tv = hit_tgt  if hit_tgt  is not None else 9999
        sv = hit_stop if hit_stop is not None else 9999
        if tv <= sv:
            pnl_bps = TGT_S * 10000 * sigma
        elif sv < tv:
            pnl_bps = -STOP_S * 10000 * sigma
        else:
            raw = math.log(closes[min(i + MAX_HOLD, n-1)] / entry) * direction
            pnl_bps = raw * 10000

        hold_until = min(tv, sv, i + MAX_HOLD)
        trades.append(dict(date=dates[i], pnl=pnl_bps, entry=entry))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  VWASLR  —  5-min bars, VWASLR(10) crosses ±0.4σ
# ═══════════════════════════════════════════════════════════════════════════════

def run_vwaslr(bars5: pd.DataFrame) -> list[dict]:
    N_WIN     = 10      # 50-min window
    THRESH    = 0.4     # σ units
    TRAIL     = 20
    STOP_S    = 2.0
    TGT_S     = 3.0
    MAX_HOLD  = 3       # bars = 15 min
    RTH_HM    = (9*60, 16*60)
    BLACKOUT  = (9*60+30, 9*60+40)  # 9:30-9:40 ET

    rth = bars5[(bars5["hm"] >= RTH_HM[0]) & (bars5["hm"] < RTH_HM[1])].copy()
    rth = rth.reset_index(drop=True)

    closes  = rth["close"].values.astype(float)
    highs   = rth["high"].values.astype(float)
    lows    = rth["low"].values.astype(float)
    volumes = rth["volume"].values.astype(float)
    hms     = rth["hm"].values
    dates   = rth["date"].values
    n       = len(rth)

    gap = (rth["ts"].diff() > pd.Timedelta(minutes=10)).values.copy()
    gap[0] = True

    LOOKBACK = max(TRAIL, N_WIN) + 2
    trades = []
    hold_until = -1

    for i in range(LOOKBACK, n - MAX_HOLD - 1):
        if i <= hold_until:
            continue
        if gap[i - TRAIL + 1: i + MAX_HOLD + 1].any():
            continue

        hm = hms[i]
        if BLACKOUT[0] <= hm < BLACKOUT[1]:
            continue

        trail_rets = np.log(closes[i - TRAIL + 1: i + 1] /
                            closes[i - TRAIL    : i    ])
        sigma = np.std(trail_rets, ddof=1)
        if sigma == 0:
            continue

        ret_win = np.log(closes[i - N_WIN + 1: i + 1] /
                         closes[i - N_WIN    : i    ])
        vol_win = volumes[i - N_WIN: i]
        sv      = vol_win.sum()
        if sv == 0:
            continue
        scaled_rets = ret_win / sigma
        vwaslr = float((scaled_rets * vol_win).sum() / sv)

        if abs(vwaslr) < THRESH:
            continue

        direction = 1 if vwaslr > 0 else -1
        entry     = closes[i]
        tgt_price  = entry * math.exp( direction * TGT_S  * sigma)
        stop_price = entry * math.exp(-direction * STOP_S * sigma)

        hit_tgt = hit_stop = None
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != dates[i]:
                break
            h, l = highs[j], lows[j]
            if hit_tgt is None:
                if direction == 1 and h >= tgt_price:  hit_tgt  = j
                if direction == -1 and l <= tgt_price:  hit_tgt  = j
            if hit_stop is None:
                if direction == 1 and l <= stop_price:  hit_stop = j
                if direction == -1 and h >= stop_price:  hit_stop = j
            if hit_tgt or hit_stop:
                break

        tv = hit_tgt  if hit_tgt  is not None else 9999
        sv2 = hit_stop if hit_stop is not None else 9999
        if tv <= sv2:
            pnl_bps = TGT_S * 10000 * sigma
        elif sv2 < tv:
            pnl_bps = -STOP_S * 10000 * sigma
        else:
            raw = math.log(closes[min(i + MAX_HOLD, n-1)] / entry) * direction
            pnl_bps = raw * 10000

        hold_until = min(tv, sv2, i + MAX_HOLD)
        trades.append(dict(date=dates[i], pnl=pnl_bps, entry=entry))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  ORB  —  1-min bars, 15-min range, LONG-only on gap-down days
# ═══════════════════════════════════════════════════════════════════════════════

def run_orb(bars1: pd.DataFrame) -> list[dict]:
    ORB_MIN    = 15      # opening range minutes
    WIDTH_MIN  = 0.0015
    WIDTH_MAX  = 0.0050
    ENTRY_WIN  = 60      # minutes after ORB closes (9:45 → entry until 10:45)
    ORB_OPEN   = 9*60+30
    ORB_CLOSE  = ORB_OPEN + ORB_MIN

    trades = []
    for day, grp in bars1.groupby("date"):
        rth = grp[(grp["hm"] >= ORB_OPEN) & (grp["hm"] < 16*60)].sort_values("hm")
        if len(rth) < ORB_MIN + 2:
            continue

        # Opening range bars
        orb_bars = rth[rth["hm"] < ORB_CLOSE]
        if len(orb_bars) < ORB_MIN - 1:
            continue

        orb_high = orb_bars["high"].max()
        orb_low  = orb_bars["low"].min()
        orb_mid  = (orb_high + orb_low) / 2
        width    = (orb_high - orb_low) / orb_mid

        if not (WIDTH_MIN <= width <= WIDTH_MAX):
            continue

        # Gap-down filter: day open < previous RTH close
        open_bar = rth.iloc[0]["open"]
        prev_days = bars1[(bars1["date"] < day) & (bars1["hm"] >= 9*60+30) &
                          (bars1["hm"] < 16*60)]
        if prev_days.empty:
            continue
        prev_close = prev_days.sort_values(["date","hm"]).groupby("date")["close"].last()
        if prev_close.empty:
            continue
        prev_c = float(prev_close.iloc[-1])
        if open_bar >= prev_c:   # not a gap-down day → skip
            continue

        # Look for LONG breakout: first bar close > ORB high in entry window
        entry_bars = rth[(rth["hm"] >= ORB_CLOSE) & (rth["hm"] < ORB_CLOSE + ENTRY_WIN)]
        entry_row  = None
        for _, row in entry_bars.iterrows():
            if row["close"] > orb_high:
                entry_row = row
                break
        if entry_row is None:
            continue

        entry    = float(entry_row["close"])
        target   = entry + (orb_high - orb_low)   # 1× width
        stop     = orb_mid                          # half range

        # Simulate remaining bars
        after = rth[rth["hm"] > entry_row["hm"]]
        pnl = None
        for _, row in after.iterrows():
            if row["high"] >= target:
                pnl = target - entry; break
            if row["low"]  <= stop:
                pnl = stop   - entry; break
        if pnl is None:
            pnl = float(after.iloc[-1]["close"]) - entry if len(after) else 0.0

        trades.append(dict(date=day, pnl=pnl, entry=entry))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  SLR Scalp  —  20s bars, vol≥7× median, move≥12bp, 10bp stop / 15bp target
# ═══════════════════════════════════════════════════════════════════════════════

def run_slr(df5: pd.DataFrame) -> list[dict]:
    BAR_S      = 20
    VOL_MULT   = 7.0
    MOVE_BPS   = 12.0
    TARGET_BPS = 15.0
    STOP_BPS   = 10.0
    VOL_LOOK   = 40
    MAX_HOLD   = 6      # bars = 2 min
    RTH_HM     = (9*60+30, 16*60)
    BLACKOUT   = (9*60+30, 9*60+40)

    # Keep only RTH bars
    rth5 = df5[(df5["hm"] >= RTH_HM[0]) & (df5["hm"] < RTH_HM[1])].copy()
    bars = resample_5sec(rth5.reset_index(drop=True), BAR_S)
    # Re-filter hm after resample
    bars = bars[(bars["hm"] >= RTH_HM[0]) & (bars["hm"] < RTH_HM[1])].reset_index(drop=True)

    c   = bars["close"].values.astype(float)
    o   = bars["open"].values.astype(float)
    hi  = bars["high"].values.astype(float)
    lo  = bars["low"].values.astype(float)
    vol = bars["volume"].values.astype(float)
    hms = bars["hm"].values
    dts = bars["date"].values
    gap = bars["gap"].values.astype(bool)
    n   = len(bars)

    vol_s = pd.Series(np.where(gap, np.nan, vol))
    med   = vol_s.rolling(VOL_LOOK, min_periods=VOL_LOOK).median().values

    trades = []
    hold_until = -1

    for i in range(VOL_LOOK + 1, n - MAX_HOLD - 2):
        if i <= hold_until:
            continue
        if np.isnan(med[i]) or med[i] == 0:
            continue
        if gap[i] or gap[i - 1]:
            continue
        if BLACKOUT[0] <= hms[i] < BLACKOUT[1]:
            continue

        vol_ratio = vol[i] / med[i]
        if vol_ratio < VOL_MULT:
            continue

        # Both directions
        is_long = c[i] > o[i]
        if is_long:
            move_bps = (c[i] - o[i - 1]) / c[i] * 10000
            direction = 1
        else:
            move_bps = (o[i - 1] - c[i]) / c[i] * 10000
            direction = -1

        if move_bps < MOVE_BPS:
            continue

        entry_bar = i + 1
        if entry_bar + MAX_HOLD >= n:
            continue
        if gap[entry_bar]:
            continue

        entry      = o[entry_bar]
        target_pts = entry * TARGET_BPS / 10000
        stop_pts   = entry * STOP_BPS   / 10000

        pnl = None
        for k in range(1, MAX_HOLD + 1):
            j = entry_bar + k
            if j >= n or dts[j] != dts[i]:
                break
            if direction == 1:
                if hi[j] >= entry + target_pts: pnl =  TARGET_BPS; break
                if lo[j] <= entry - stop_pts:   pnl = -STOP_BPS;   break
            else:
                if lo[j] <= entry - target_pts: pnl =  TARGET_BPS; break
                if hi[j] >= entry + stop_pts:   pnl = -STOP_BPS;   break

        if pnl is None:
            j = min(entry_bar + MAX_HOLD, n - 1)
            raw = (c[j] - entry) * direction / entry * 10000
            pnl = max(raw, -STOP_BPS)

        hold_until = entry_bar + MAX_HOLD
        trades.append(dict(date=dts[i], pnl=pnl, entry=entry))

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  PL_REV  —  5sec bars, fade PL≥0.80 + move≥20bp, ADX gate
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_adx_per_day(bars1: pd.DataFrame, period: int = 14) -> dict:
    """Wilder ADX(14) on 1-min bars in 9:00-10:00 ET window per day."""
    window_start, window_end = 9*60, 10*60
    adx_by_day = {}

    for day, grp in bars1.groupby("date"):
        win = grp[(grp["hm"] >= window_start) & (grp["hm"] < window_end)].sort_values("hm")
        if len(win) < period * 2:
            continue
        h = win["high"].values.astype(float)
        l = win["low"].values.astype(float)
        c = win["close"].values.astype(float)
        n = len(c)

        tr   = np.maximum(h[1:]-l[1:], np.maximum(abs(h[1:]-c[:-1]), abs(l[1:]-c[:-1])))
        dmp  = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]), np.maximum(h[1:]-h[:-1], 0.0), 0.0)
        dmm  = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]), np.maximum(l[:-1]-l[1:], 0.0), 0.0)

        # Wilder smooth
        def wilder(arr, p):
            out = np.zeros(len(arr))
            out[p-1] = arr[:p].sum()
            for k in range(p, len(arr)):
                out[k] = out[k-1] - out[k-1]/p + arr[k]
            return out

        atr  = wilder(tr, period)
        pdm  = wilder(dmp, period)
        ndm  = wilder(dmm, period)

        with np.errstate(divide="ignore", invalid="ignore"):
            pdi = np.where(atr > 0, pdm / atr * 100, 0.0)
            ndi = np.where(atr > 0, ndm / atr * 100, 0.0)
            dx  = np.where((pdi + ndi) > 0, abs(pdi - ndi) / (pdi + ndi) * 100, 0.0)

        dx_valid = dx[period-1:]
        if len(dx_valid) < period:
            continue
        adx_val = np.zeros(len(dx_valid))
        adx_val[period-1] = dx_valid[:period].mean()
        for k in range(period, len(dx_valid)):
            adx_val[k] = (adx_val[k-1] * (period-1) + dx_valid[k]) / period

        adx_by_day[day] = adx_val[-1]

    return adx_by_day


def run_pl_rev(df5: pd.DataFrame, bars1: pd.DataFrame) -> list[dict]:
    PL_THR    = 0.80
    MOVE_BPS  = 20.0
    TP_BPS    = 12.0
    STOP_BPS  = 15.0
    RESUME_PL = 0.70
    MIN_HOLD  = 2    # bars = 10s
    MAX_HOLD  = 24   # bars = 120s
    WINDOW    = 6    # 5sec bars = 30s
    ADX_GATE  = 25.0
    ENTRY_HM  = (10*60, 16*60)

    print("  Computing per-day ADX(9-10ET) …", flush=True)
    adx_by_day = _compute_adx_per_day(bars1)

    # RTH 5sec bars with entry window
    rth = df5[(df5["hm"] >= 9*60+30) & (df5["hm"] < 16*60)].copy().reset_index(drop=True)

    closes = rth["close"].values.astype(float)
    highs  = rth["high"].values.astype(float)
    lows   = rth["low"].values.astype(float)
    dates  = rth["date"].values
    hms    = rth["hm"].values

    # Gap flag
    gap = (rth["ts"].diff() > pd.Timedelta(seconds=10)).values.copy()
    gap[0] = True

    # Vectorized PL
    s = pd.Series(closes)
    lr = np.log(s / s.shift(1)).copy()
    lr.iloc[np.where(gap)[0]] = np.nan
    abs_lr = lr.abs()
    gap_in_win = abs_lr.isna().rolling(WINDOW, min_periods=1).max().astype(bool)
    net_ret = lr.rolling(WINDOW, min_periods=WINDOW).sum()
    abs_sum = abs_lr.rolling(WINDOW, min_periods=WINDOW).sum()
    net_ret[gap_in_win] = np.nan
    abs_sum[gap_in_win] = np.nan

    pl_arr  = (net_ret.abs() / abs_sum.replace(0, np.nan)).values
    dir_arr = np.sign(net_ret.values)
    lb_close = np.full(len(closes), np.nan)
    lb_close[WINDOW:] = closes[:len(closes) - WINDOW]
    move_arr = np.abs(closes - lb_close) / np.where(lb_close > 0, lb_close, 1) * 10000
    move_arr[gap_in_win.values] = np.nan

    trades = []
    i = 0
    n = len(rth)

    while i < n:
        day = dates[i]
        hm  = hms[i]

        if not (ENTRY_HM[0] <= hm < ENTRY_HM[1]):
            i += 1; continue

        # ADX gate
        adx = adx_by_day.get(day, 0.0)
        if adx > ADX_GATE:
            i += 1; continue

        # Qualifying gate
        if (gap[i] or np.isnan(pl_arr[i]) or pl_arr[i] < PL_THR
                or np.isnan(move_arr[i]) or move_arr[i] < MOVE_BPS):
            i += 1; continue

        direction = -int(dir_arr[i])   # FADE the move
        entry     = closes[i]
        tp_pt     = entry * TP_BPS   / 10000
        stop_pt   = entry * STOP_BPS / 10000

        pnl_bps = None
        exit_i  = i
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != day:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS); exit_i = j - 1; break
            if direction == 1 and highs[j] >= entry + tp_pt:
                pnl_bps = TP_BPS; exit_i = j; break
            if direction == -1 and lows[j] <= entry - tp_pt:
                pnl_bps = TP_BPS; exit_i = j; break
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if k >= MIN_HOLD:
                cp = pl_arr[j]
                if not np.isnan(cp) and cp >= RESUME_PL:
                    raw = (closes[j] - entry) * direction / entry * 10000
                    pnl_bps = max(raw, -STOP_BPS); exit_i = j; break

        if pnl_bps is None:
            j = i + MAX_HOLD
            raw = (closes[min(j, n-1)] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -STOP_BPS); exit_i = min(j, n-1)

        trades.append(dict(date=day, pnl=pnl_bps, entry=entry))
        i = exit_i + 1

    return trades


# ═══════════════════════════════════════════════════════════════════════════════
#  Wall Break  —  wall_events.csv + 1-min bars, 4pt stop / 12pt target / 15-min
# ═══════════════════════════════════════════════════════════════════════════════

def run_wall_break(bars1: pd.DataFrame) -> list[dict]:
    PEAK_MIN  = 100
    PEAK_MAX  = 300
    MIN_TESTS = 2
    STOP_PT   = 4.0
    TGT_PT    = 12.0
    HOLD_MIN  = 15
    RTH_HM    = (9*60+40, 16*60)

    try:
        ev = pd.read_csv(WALL_EVENTS)
    except FileNotFoundError:
        print("  wall_events.csv not found — skipping Wall Break")
        return []

    ev.columns = ["ts","symbol","side","wall_price","wall_size",
                  "peak_size","event","test_count","price_at_event"]
    ev["ts"]          = pd.to_datetime(ev["ts"], utc=True)
    ev["peak_size"]   = pd.to_numeric(ev["peak_size"], errors="coerce")
    ev["test_count"]  = pd.to_numeric(ev["test_count"], errors="coerce")
    ev["price_at_event"] = pd.to_numeric(ev["price_at_event"], errors="coerce")

    bo = ev[
        (ev["event"]     == "breakout") &
        (ev["symbol"]    == "MES") &
        (ev["peak_size"] >= PEAK_MIN) &
        (ev["peak_size"] <  PEAK_MAX) &
        (ev["test_count"] >= MIN_TESTS) &
        ev["price_at_event"].notna()
    ].copy()

    bo["ts_et"] = bo["ts"].dt.tz_convert(ET)
    bo["hm"]    = bo["ts_et"].dt.hour * 60 + bo["ts_et"].dt.minute
    bo["date"]  = bo["ts_et"].dt.date
    bo = bo[(bo["date"] >= DATE_START) & (bo["date"] <= DATE_END)].copy()
    bo = bo[(bo["hm"] >= RTH_HM[0]) & (bo["hm"] < RTH_HM[1])].copy()
    bo["direction"] = bo["side"].map({"ask": 1, "bid": -1})

    trades = []
    bar_by_date = {d: g for d, g in bars1.groupby("date")}

    for _, row in bo.iterrows():
        day  = row["date"]
        bars = bar_by_date.get(day)
        if bars is None:
            continue

        entry    = float(row["price_at_event"])
        ev_hm    = int(row["hm"])
        direction = int(row["direction"])

        after = bars[(bars["hm"] > ev_hm) & (bars["hm"] < 16*60)].sort_values("hm")
        cutoff_hm = ev_hm + HOLD_MIN

        pnl = None
        for _, bar in after.iterrows():
            if bar["hm"] >= cutoff_hm:
                pnl = (float(bar["close"]) - entry) * direction; break
            if direction == 1:
                if bar["high"] >= entry + TGT_PT:  pnl =  TGT_PT; break
                if bar["low"]  <= entry - STOP_PT: pnl = -STOP_PT; break
            else:
                if bar["low"]  <= entry - TGT_PT:  pnl =  TGT_PT; break
                if bar["high"] >= entry + STOP_PT:  pnl = -STOP_PT; break

        if pnl is None and len(after) > 0:
            pnl = (float(after.iloc[-1]["close"]) - entry) * direction

        if pnl is not None:
            trades.append(dict(date=day, pnl=pnl, entry=entry))

    return trades


# ─── Summarise for point-denominated strategies ───────────────────────────────

def summarise_pts(trades: list[dict], strat: str) -> None:
    if not trades:
        print(f"\n  {strat}: no trades")
        return
    df = pd.DataFrame(trades)
    df["ym"] = df["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
    n   = len(df)
    wr  = (df["pnl"] > 0).mean()
    avg = df["pnl"].mean()
    tot = df["pnl"].sum()
    dol = tot * MES_PV

    print(f"\n  ─── {strat} ───────────────────────────")
    print(f"  Trades: {n}  WR: {wr:.1%}  avg {avg:+.3f}pt  total {tot:+.1f}pt  ≈${dol:+,.0f}")

    monthly = df.groupby("ym")["pnl"].agg(n="count", tot="sum",
                                            wins=lambda x: (x>0).sum())
    for ym, row in monthly.iterrows():
        wr2 = row["wins"] / row["n"] if row["n"] else 0
        d   = row["tot"] * MES_PV
        print(f"    {ym}  n={int(row['n']):3d}  WR={wr2:.1%}  "
              f"tot={row['tot']:+7.2f}pt  ≈${d:+6,.0f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'═'*60}")
    print(f"  MES Strategy Backtest  —  {DATE_START} to {DATE_END}")
    print(f"{'═'*60}")

    # Load bars
    print("\nLoading data …")
    bars1 = load_db_bars(1)
    bars5 = load_db_bars(5)
    print(f"  1-min bars: {len(bars1):,}  "
          f"({bars1['date'].min()} → {bars1['date'].max()})")
    print(f"  5-min bars: {len(bars5):,}  "
          f"({bars5['date'].min()} → {bars5['date'].max()})")

    print("  Loading 5sec bars (this may take a moment) …", flush=True)
    df5sec = load_5sec()
    print(f"  5sec bars: {len(df5sec):,}  "
          f"({df5sec['date'].min()} → {df5sec['date'].max()})")

    print(f"\n  Trading days: {bars1['date'].nunique()} (1-min), "
          f"{df5sec['date'].nunique()} (5sec)")

    print(f"\n{'─'*60}")
    print(f"  Running strategies …")
    print(f"{'─'*60}")

    # CSR
    print("\nCSR …", flush=True)
    csr_trades = run_csr(bars5)
    summarise(csr_trades, "CSR  (5-min, 3σ+CSR-40min, 2σ/3σ)", unit="bp")

    # VWASLR
    print("\nVWASLR …", flush=True)
    vwa_trades = run_vwaslr(bars5)
    summarise(vwa_trades, "VWASLR  (5-min, VWASLR(10)±0.4σ, 2σ/3σ)", unit="bp")

    # ORB
    print("\nORB …", flush=True)
    orb_trades = run_orb(bars1)
    summarise_pts(orb_trades, "ORB  (15-min range, gap-down, LONG only)")

    # SLR
    print("\nSLR Scalp …", flush=True)
    slr_trades = run_slr(df5sec)
    summarise(slr_trades, "SLR Scalp  (20s bars, vol≥7×, move≥12bp, 10/15bp)", unit="bp")

    # PL_REV
    print("\nPL_REV …", flush=True)
    plr_trades = run_pl_rev(df5sec, bars1)
    summarise(plr_trades, "PL_REV  (5sec, PL≥0.80, move≥20bp, ADX≤25, 15/12bp)", unit="bp")

    # Wall Break
    print("\nWall Break …", flush=True)
    wb_trades = run_wall_break(bars1)
    summarise_pts(wb_trades, "Wall Break  (peak 100-300, test≥2, 4pt stop/12pt target)")

    # Grand total (in $, converting all bp strategies)
    print(f"\n{'═'*60}")
    all_trades = []
    for tlist in [csr_trades, vwa_trades, slr_trades, plr_trades]:
        for t in tlist:
            pts = t["pnl"] * t.get("entry", 5400) / 10000
            all_trades.append(pts * MES_PV)
    for t in orb_trades + wb_trades:
        all_trades.append(t["pnl"] * MES_PV)
    print(f"  ALL STRATEGIES  {len(all_trades)} total trades  "
          f"≈${sum(all_trades):+,.0f}  (1 lot MES each, no costs)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
