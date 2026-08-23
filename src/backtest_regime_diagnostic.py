"""
Macro-regime diagnostic for PL_MOM.

For each trading day, computes several pre-10:00 ET regime indicators
and compares them against daily PL_MOM win-rate to find which signals
best separate "good" (trending) from "bad" (choppy) days.

Indicators (all computed from data available before 10:00 ET):
  pl_glob_1h    — Price Linearity on 1-min Globex bars, 8:30–9:30 ET
  pl_rth_30m    — Price Linearity on 1-min RTH bars, 9:30–10:00 ET
  adx_rth       — ADX(14) on 1-min bars at 10:00 ET (uses pre-market for warmup)
  or_atr_ratio  — (9:30–10:00 H-L range) / 20-day ATR on 1-min bars
  mid_cross_rt  — Midpoint-cross rate in 9:30–10:00 (crosses / bar)
  range_pos     — Where in the 9:30–10:00 range the 10:00 close sits [0=bottom,1=top]

PL_MOM simulation (fixed params matching live bot):
  PL ≥ 0.70, move ≥ 12bp, 30s window (6 × 5s bars)
  Exit: PL ≤ 0.40 OR stop 10bp OR max hold 120s (24 bars), min hold 10s (2 bars)

Day labels:
  good    WR ≥ 45%  with ≥ MIN_TRADES_PER_DAY trades
  bad     WR ≤ 25%  with ≥ MIN_TRADES_PER_DAY trades
  neutral otherwise

Usage:
  python src/backtest_regime_diagnostic.py          # MES
  python src/backtest_regime_diagnostic.py MNQ
"""

import sys
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

ET = ZoneInfo("America/New_York")

# ── PL_MOM params (must match live bot) ────────────────────────────────────
WINDOW        = 6      # 5s bars = 30s
MAX_HOLD      = 24     # bars = 120s
MIN_HOLD      = 2      # bars = 10s
EXIT_PL       = 0.40
STOP_BPS      = 10.0
PL_THR        = 0.70
MOVE_THR_BPS  = 12.0
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

# ── Day-labelling ───────────────────────────────────────────────────────────
MIN_TRADES_PER_DAY = 5
GOOD_WR = 0.45
BAD_WR  = 0.25

# Months to exclude from ALL analysis (format: set of (year, month) tuples)
# Exclude April 2025 — tariff crash, extreme vol, not representative
EXCLUDE_MONTHS: set[tuple[int,int]] = {(2025, 4)}

SYMBOL_CONFIGS = {
    "MES": dict(csv="es_hist_5sec.csv",  tick_size=0.25, point_value=5.0),
    "MNQ": dict(csv="nq_hist_5sec.csv",  tick_size=0.25, point_value=2.0),
}

# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_all_bars(csv_path):
    """Load all bars (no RTH filter) — needed for Globex pre-market."""
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    h = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    df["ts_et"] = df["ts"].dt.tz_convert(ET)
    df["date"]  = df["ts_et"].dt.date
    df["hm"]    = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
    return df.reset_index(drop=True)


def load_rth_5s(csv_path):
    """Load RTH-only 5s bars (9:30–16:00 ET, CME settlement removed)."""
    df = load_all_bars(csv_path)
    df = df[(df["hm"] >= 570) & (df["hm"] < 960)].copy()
    if EXCLUDE_MONTHS:
        mask = df["date"].apply(lambda d: (d.year, d.month) not in EXCLUDE_MONTHS)
        df = df[mask].copy()
    df["gap"] = (df["ts"].diff() > pd.Timedelta(seconds=10)).values
    df.iloc[0, df.columns.get_loc("gap")] = True
    return df.reset_index(drop=True)


def resample_1min(df_raw):
    """Resample raw 5s bars to 1-min OHLCV using ts as UTC index."""
    out = (df_raw.set_index("ts")
           .resample("1min", closed="left", label="left")
           .agg(open=("open","first"), high=("high","max"),
                low=("low","min"),   close=("close","last"),
                volume=("volume","sum"))
           .dropna(subset=["open"])
           .reset_index())
    out["ts_et"] = out["ts"].dt.tz_convert(ET)
    out["date"]  = out["ts_et"].dt.date
    out["hm"]    = out["ts_et"].dt.hour * 60 + out["ts_et"].dt.minute
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  Macro indicators on 1-min bars
# ═══════════════════════════════════════════════════════════════════════════

def _pl(closes):
    """Price Linearity on a sequence of close prices."""
    if len(closes) < 2:
        return np.nan
    lr = np.diff(np.log(closes))
    sa = np.abs(lr).sum()
    return abs(lr.sum()) / sa if sa > 0 else np.nan


def _adx(highs, lows, closes, period=14):
    """
    Compute standard ADX (0–100) on arrays.  Returns scalar (last value).
    Uses proper Wilder smoothing: first value = sum of first `period` items,
    then EMA-style: S[i] = S[i-1] - S[i-1]/p + x[i].
    +DI / -DI are scaled to 0-100 by dividing by the smoothed ATR.
    ADX = Wilder-smoothed DX, initialised as the *mean* of the first `period`
    DX values (so ADX itself stays in [0, 100]).
    """
    n = len(closes)
    if n < period * 2 + 2:
        return np.nan

    tr  = np.maximum(highs[1:] - lows[1:],
          np.maximum(np.abs(highs[1:] - closes[:-1]),
                     np.abs(lows[1:] - closes[:-1])))
    pdm = np.where((highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
                   np.maximum(highs[1:] - highs[:-1], 0.0), 0.0)
    ndm = np.where((lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
                   np.maximum(lows[:-1] - lows[1:], 0.0), 0.0)

    # Wilder smoothing: init = sum of first period, then rolling subtract+add
    def _wilder(arr):
        s = np.empty(len(arr))
        s[0] = arr[:period].sum()
        for i in range(1, len(arr)):
            s[i] = s[i-1] - s[i-1] / period + arr[i]
        return s

    satr = _wilder(tr)
    spdm = _wilder(pdm)
    sndm = _wilder(ndm)

    # +DI / -DI in 0-100
    pdi = 100.0 * spdm / np.maximum(satr, 1e-12)
    ndi = 100.0 * sndm / np.maximum(satr, 1e-12)

    # DX in 0-100
    dx = 100.0 * np.abs(pdi - ndi) / np.maximum(pdi + ndi, 1e-12)

    # ADX: Wilder-smooth DX but initialise with mean (keeps range 0-100)
    if len(dx) < period:
        return np.nan
    adx = np.empty(len(dx))
    adx[period - 1] = dx[:period].mean()
    for i in range(period, len(dx)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return float(adx[-1])


def compute_daily_macro(df_1min, df_rth_1min):
    """
    Returns DataFrame indexed by date with macro regime indicators.

    df_1min      — full session 1-min bars (Globex + RTH), no day filter
    df_rth_1min  — RTH-only 1-min bars
    """
    dates = sorted(df_rth_1min["date"].unique())

    # 20-day rolling ATR on 1-min RTH bars (prior 20 days)
    rth_atr_by_date = {}
    atr_window = []
    for d in dates:
        dv = df_rth_1min[df_rth_1min["date"] == d]
        if len(dv) > 0:
            day_hl = (dv["high"] - dv["low"]).mean()
            if len(atr_window) >= 1:
                rth_atr_by_date[d] = np.mean(atr_window[-20:])
            atr_window.append(day_hl)
        else:
            rth_atr_by_date[d] = np.nan

    rows = []
    for d in dates:
        day_all  = df_1min[df_1min["date"] == d]
        day_rth  = df_rth_1min[df_rth_1min["date"] == d]

        # ── Globex 1h PL: 8:30–9:30 ET ──────────────────────────────────
        glob = day_all[(day_all["hm"] >= 510) & (day_all["hm"] < 570)]
        pl_glob_1h = _pl(glob["close"].values) if len(glob) >= 5 else np.nan

        # ── RTH 30-min PL: 9:30–10:00 ET ────────────────────────────────
        rth30 = day_rth[(day_rth["hm"] >= 570) & (day_rth["hm"] < 600)]
        pl_rth_30m = _pl(rth30["close"].values) if len(rth30) >= 5 else np.nan

        # ── ADX(14) at 10:00: use 9:00–10:00 (60 bars + warmup) ─────────
        adx_bars = day_all[(day_all["hm"] >= 540) & (day_all["hm"] < 600)]
        if len(adx_bars) >= 29:
            adx_val = _adx(adx_bars["high"].values,
                           adx_bars["low"].values,
                           adx_bars["close"].values, period=14)
        else:
            adx_val = np.nan

        # ── Opening range metrics: 9:30–10:00 ───────────────────────────
        or_bars = rth30
        or_atr_ratio = np.nan
        mid_cross_rt = np.nan
        range_pos    = np.nan
        if len(or_bars) >= 5:
            h_max = or_bars["high"].max()
            h_min = or_bars["low"].min()
            h_rng = h_max - h_min
            atr_ref = rth_atr_by_date.get(d, np.nan)
            if not np.isnan(atr_ref) and atr_ref > 0:
                or_atr_ratio = h_rng / atr_ref

            if h_rng > 0:
                mid = (h_max + h_min) / 2
                closes = or_bars["close"].values
                crosses = np.sum(np.diff(np.sign(closes - mid)) != 0)
                mid_cross_rt = crosses / max(len(closes) - 1, 1)
                range_pos = (closes[-1] - h_min) / h_rng

        rows.append(dict(
            date=d,
            pl_glob_1h=pl_glob_1h,
            pl_rth_30m=pl_rth_30m,
            adx_rth=adx_val,
            or_atr_ratio=or_atr_ratio,
            mid_cross_rt=mid_cross_rt,
            range_pos=range_pos,
        ))

    return pd.DataFrame(rows).set_index("date")

# ═══════════════════════════════════════════════════════════════════════════
#  PL_MOM simulation (per-day WR)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_daily_wr(df_5s):
    """
    Run PL_MOM on 5s bars; return per-day DataFrame with n_trades, wins, WR, ev_bps.
    Qualifying gate: 9:40–15:55 ET, PL≥0.70, move≥12bp.
    Live exit: PL≤0.40 OR stop 10bp OR max hold 120s, min hold 10s.
    """
    closes  = df_5s["close"].values
    highs   = df_5s["high"].values
    lows    = df_5s["low"].values
    gaps    = df_5s["gap"].values.astype(bool)
    dates   = df_5s["date"].values          # numpy array of date objects
    ts_et   = df_5s["ts_et"]
    hm      = df_5s["hm"].values
    n       = len(df_5s)

    # Compute PL, direction, move on 5s bars
    lr = np.empty(n); lr[:] = np.nan
    lr[1:] = np.log(closes[1:] / closes[:-1])
    lr[gaps] = np.nan

    pl_arr   = np.full(n, np.nan)
    dir_arr  = np.zeros(n)
    move_arr = np.full(n, np.nan)

    for i in range(WINDOW, n):
        if gaps[i - WINDOW + 1: i + 1].any():
            continue
        rets = lr[i - WINDOW + 1: i + 1]
        if np.isnan(rets).any():
            continue
        sa = np.abs(rets).sum()
        if sa == 0:
            continue
        net = rets.sum()
        pl_arr[i]   = abs(net) / sa
        dir_arr[i]  = 1.0 if net > 0 else -1.0
        move_arr[i] = abs(closes[i] - closes[i - WINDOW]) / closes[i] * 10000

    # Walk bars, collect per-trade results
    trade_results = []  # list of (date, pnl_bps)
    i = 0
    RTH_START_HM = 580   # 9:40 ET
    RTH_END_HM   = 955   # 15:55 ET

    while i < n:
        # Qualifying gate
        if not (RTH_START_HM <= hm[i] < RTH_END_HM
                and not gaps[i]
                and not np.isnan(pl_arr[i])
                and pl_arr[i] >= PL_THR
                and not np.isnan(move_arr[i])
                and move_arr[i] >= MOVE_THR_BPS):
            i += 1
            continue

        direction = dir_arr[i]
        entry     = closes[i]
        day_0     = dates[i]
        stop_pt   = entry * STOP_BPS / 10000

        # Walk exit
        pnl_bps = None
        exit_i  = i
        for k in range(1, MAX_HOLD + 1):
            j = i + k
            if j >= n or dates[j] != day_0:
                raw = (closes[j-1] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS)
                exit_i  = j - 1
                break
            if direction == 1 and lows[j] <= entry - stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            if direction == -1 and highs[j] >= entry + stop_pt:
                pnl_bps = -STOP_BPS; exit_i = j; break
            cur_pl = pl_arr[j]
            if k >= MIN_HOLD and not np.isnan(cur_pl) and cur_pl <= EXIT_PL:
                raw = (closes[j] - entry) * direction / entry * 10000
                pnl_bps = max(raw, -STOP_BPS); exit_i = j; break
        else:
            j = i + MAX_HOLD
            if j < n:
                raw = (closes[j] - entry) * direction / entry * 10000
            else:
                raw = (closes[n-1] - entry) * direction / entry * 10000
            pnl_bps = max(raw, -STOP_BPS)
            exit_i  = min(j, n - 1)

        trade_results.append((day_0, pnl_bps))
        i = exit_i + 1

    if not trade_results:
        return pd.DataFrame(columns=["date","n","wins","WR","ev_bps"])

    tr_df = pd.DataFrame(trade_results, columns=["date","pnl"])
    grp   = tr_df.groupby("date")["pnl"].agg(
        n="count",
        wins=lambda x: (x > 0).sum(),
        ev_bps="mean"
    ).reset_index()
    grp["WR"] = grp["wins"] / grp["n"]
    return grp

# ═══════════════════════════════════════════════════════════════════════════
#  Analysis & plotting
# ═══════════════════════════════════════════════════════════════════════════

INDICATORS = [
    ("pl_glob_1h",   "PL Globex 1h\n(8:30–9:30)"),
    ("pl_rth_30m",   "PL RTH 30m\n(9:30–10:00)"),
    ("adx_rth",      "ADX(14)\n(9:00–10:00)"),
    ("or_atr_ratio", "OR / ATR\n(9:30–10:00 range ÷ 20d ATR)"),
    ("mid_cross_rt", "Midpoint\ncross rate"),
    ("range_pos",    "Range position\nat 10:00"),
]


def run(symbol):
    cfg      = SYMBOL_CONFIGS[symbol]
    csv_path = Path(cfg["csv"])
    excl_str = (f"  [excluding {', '.join(f'{y}-{m:02d}' for y,m in sorted(EXCLUDE_MONTHS))}]"
                if EXCLUDE_MONTHS else "")
    print(f"\n=== Regime Diagnostic  ·  {symbol}{excl_str} ===\n")

    print("Loading bars …")
    df_all  = load_all_bars(str(csv_path))
    df_rth5 = load_rth_5s(str(csv_path))
    df_rth5["ts_et"] = df_rth5["ts"].dt.tz_convert(ET)
    df_rth5["date"]  = df_rth5["ts_et"].dt.date
    df_rth5["hm"]    = df_rth5["ts_et"].dt.hour * 60 + df_rth5["ts_et"].dt.minute

    df_1min_all = resample_1min(df_all)
    df_1min_rth = df_1min_all[(df_1min_all["hm"] >= 570) & (df_1min_all["hm"] < 960)].copy()

    print(f"  {len(df_rth5):,} 5s RTH bars  "
          f"{df_rth5['date'].min()} → {df_rth5['date'].max()}")

    print("Computing macro indicators …")
    macro = compute_daily_macro(df_1min_all, df_1min_rth)

    print("Running PL_MOM simulation …")
    daily_wr = simulate_daily_wr(df_rth5)
    daily_wr["date"] = pd.to_datetime(daily_wr["date"]).dt.date

    # Merge
    merged = daily_wr.merge(macro.reset_index(), on="date", how="inner")
    merged = merged[merged["n"] >= MIN_TRADES_PER_DAY].copy()

    def label(row):
        if row["WR"] >= GOOD_WR:  return "good"
        if row["WR"] <= BAD_WR:   return "bad"
        return "neutral"
    merged["label"] = merged.apply(label, axis=1)

    counts = merged["label"].value_counts()
    print(f"\n  Days with ≥{MIN_TRADES_PER_DAY} trades: {len(merged)}")
    print(f"  good (WR≥{GOOD_WR:.0%}): {counts.get('good',0)}  "
          f"bad (WR≤{BAD_WR:.0%}): {counts.get('bad',0)}  "
          f"neutral: {counts.get('neutral',0)}")

    # ── Correlation table ────────────────────────────────────────────────
    print(f"\n{'Indicator':<22} {'r':>7} {'p':>8}  {'good':>10} {'bad':>10}  {'sep':>8}")
    print("─" * 75)

    ind_keys = [k for k, _ in INDICATORS]
    corr_rows = []
    for key, label_str in INDICATORS:
        col = merged[key].dropna()
        if len(col) < 10:
            continue
        sub = merged[["WR", key, "label"]].dropna(subset=[key])
        r, p = stats.pearsonr(sub[key], sub["WR"])

        good_m = sub[sub["label"]=="good"][key].mean()
        bad_m  = sub[sub["label"]=="bad"][key].mean()

        # Separability: Cohen's d between good and bad
        good_v = sub[sub["label"]=="good"][key]
        bad_v  = sub[sub["label"]=="bad"][key]
        pooled = np.sqrt((good_v.std()**2 + bad_v.std()**2) / 2) if len(good_v) > 1 and len(bad_v) > 1 else np.nan
        cohens_d = (good_m - bad_m) / pooled if pooled and pooled > 0 else np.nan

        pstr = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
        print(f"  {key:<20} {r:>+7.3f} {pstr:>8}  "
              f"{good_m:>10.3f} {bad_m:>10.3f}  {cohens_d:>+8.2f}")
        corr_rows.append((key, r, p, good_m, bad_m, cohens_d))

    # ── Monthly bad-day calendar ─────────────────────────────────────────
    merged["ym"] = merged["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
    bad_days     = merged[merged["label"] == "bad"]
    good_days    = merged[merged["label"] == "good"]

    print("\n  Bad-day counts by month:")
    for ym, grp in bad_days.groupby("ym"):
        bar = "░" * len(grp)
        print(f"    {ym}  {bar}  ({len(grp)} bad days)")

    # ── Plots ─────────────────────────────────────────────────────────────
    valid_inds = [(k, l) for k, l in INDICATORS if merged[k].notna().sum() > 10]
    n_inds = len(valid_inds)

    fig = plt.figure(figsize=(5 * n_inds, 10))
    fig.suptitle(f"Regime Diagnostic — {symbol}  |  "
                 f"PL_MOM 5s PL≥0.70 mv≥12bp  |  "
                 f"good=WR≥45% bad=WR≤25% (≥{MIN_TRADES_PER_DAY} trades/day)",
                 fontsize=11)

    gs = gridspec.GridSpec(2, n_inds, hspace=0.45, wspace=0.35)

    colors = {"good": "#2ca02c", "neutral": "#7f7f7f", "bad": "#d62728"}

    for col_i, (key, title) in enumerate(valid_inds):
        sub = merged[["WR", key, "label"]].dropna(subset=[key])

        # Row 0: violin / strip plot by label
        ax0 = fig.add_subplot(gs[0, col_i])
        for lbl in ["bad", "neutral", "good"]:
            vals = sub[sub["label"]==lbl][key]
            if len(vals) == 0:
                continue
            parts = ax0.violinplot([vals.values], positions=[{"bad":0,"neutral":1,"good":2}[lbl]],
                                   showmedians=True, widths=0.7)
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[lbl])
                pc.set_alpha(0.6)
            parts["cmedians"].set_color(colors[lbl])
            parts["cbars"].set_color(colors[lbl])
            parts["cmins"].set_color(colors[lbl])
            parts["cmaxes"].set_color(colors[lbl])

        ax0.set_xticks([0,1,2])
        ax0.set_xticklabels(["bad","neutral","good"], fontsize=9)
        ax0.set_title(title, fontsize=9)
        ax0.set_ylabel(key, fontsize=8)

        # Row 1: scatter vs daily WR
        ax1 = fig.add_subplot(gs[1, col_i])
        for lbl in ["bad", "neutral", "good"]:
            sv = sub[sub["label"]==lbl]
            ax1.scatter(sv[key], sv["WR"], c=colors[lbl], s=20, alpha=0.6,
                        label=lbl, zorder=3)

        # regression line
        if len(sub) > 3:
            slope, intercept, *_ = stats.linregress(sub[key], sub["WR"])
            xs = np.linspace(sub[key].min(), sub[key].max(), 50)
            ax1.plot(xs, slope*xs + intercept, "k--", lw=1, alpha=0.5)

        r, p = stats.pearsonr(sub[key], sub["WR"])
        ax1.set_xlabel(key, fontsize=8)
        ax1.set_ylabel("Daily WR", fontsize=8)
        ax1.set_title(f"r={r:+.3f}  p={p:.3f}", fontsize=8)
        ax1.axhline(GOOD_WR, color=colors["good"], lw=0.8, ls="--", alpha=0.5)
        ax1.axhline(BAD_WR,  color=colors["bad"],  lw=0.8, ls="--", alpha=0.5)
        if col_i == 0:
            ax1.legend(fontsize=7, loc="upper left")

    out_png = Path(f"regime_diagnostic_{symbol}.png")
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved → {out_png}")
    plt.show()

    # ── ADX gate simulation sweep ─────────────────────────────────────────
    # Re-run the full simulation but only trade on days where ADX ≤ threshold
    if "adx_rth" in merged.columns and merged["adx_rth"].notna().sum() > 10:
        print(f"\n{'─'*72}")
        print(f"  ADX gate sweep  —  EV and P&L on gated days  (point_value=${cfg['point_value']:.0f})")
        print(f"  {'ADX≤':>8}  {'n_days':>7}  {'n_trades':>9}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L pts':>10}  {'P&L $':>10}")
        print(f"  {'─'*70}")

        # Merge adx_rth into the full per-trade results
        # trade_results were lost; rebuild quickly using simulate_daily_wr return
        # We already have daily_wr and merged (with adx); rebuild per-trade by
        # running simulate with an allowed_dates set
        adx_col = merged[["date","adx_rth"]].dropna(subset=["adx_rth"])
        all_adx_vals = adx_col["adx_rth"].values

        # Also print baseline (no gate)
        baseline_trades = simulate_daily_wr(df_rth5)
        baseline_trades["date"] = pd.to_datetime(baseline_trades["date"]).dt.date
        bt = baseline_trades.merge(adx_col, on="date", how="inner")
        tot_n  = bt["n"].sum()
        tot_w  = bt["wins"].sum()
        tot_ev = (bt["n"] * bt["ev_bps"]).sum() / max(tot_n, 1)
        tot_pt = (bt["n"] * bt["ev_bps"] / 10000 * bt["adx_rth"].apply(
                      lambda _: 1)).sum()  # placeholder
        # recompute P&L in points properly
        def gate_stats(adx_max):
            sub = bt[bt["adx_rth"] <= adx_max] if adx_max < 999 else bt
            n   = sub["n"].sum()
            w   = sub["wins"].sum()
            ev  = (sub["n"] * sub["ev_bps"]).sum() / max(n, 1)
            # pts = sum(n_i * ev_i * entry_price / 10000); approximate with
            # ev_bps * n / 10000 * avg_ES_price
            # We don't store entry prices here; use ev_bps / 10000 * 5500 * n
            AVG_PRICE = 5_500  # rough MES/ES avg over dataset
            pts = ev * AVG_PRICE / 10000 * n
            return len(sub), int(n), w/max(n,1)*100, ev, pts, pts * cfg["point_value"]

        nd, nt, wr, ev, pts, pnl = gate_stats(999)
        print(f"  {'no gate':>8}  {nd:>7}  {nt:>9}  {wr:>6.1f}  {ev:>+8.2f}  {pts:>+10.1f}  {pnl:>+10,.0f}")

        adx_thresholds = [15, 20, 25, 30, 35, 40, 50, 60]
        for thr in adx_thresholds:
            nd, nt, wr, ev, pts, pnl = gate_stats(thr)
            if nt < 5:
                continue
            print(f"  {thr:>8}  {nd:>7}  {nt:>9}  {wr:>6.1f}  {ev:>+8.2f}  {pts:>+10.1f}  {pnl:>+10,.0f}")

        # ── Monthly P&L for ADX ≤ 25 ─────────────────────────────────────
        ADX_GATE = 25
        gated_dates = set(adx_col[adx_col["adx_rth"] <= ADX_GATE]["date"])
        bt_gated = baseline_trades[baseline_trades["date"].isin(gated_dates)].copy()
        bt_gated["ym"] = bt_gated["date"].apply(lambda d: f"{d.year}-{d.month:02d}")
        AVG_PRICE = 5_500
        PV = cfg["point_value"]
        print(f"\n  Monthly P&L  ·  ADX ≤ {ADX_GATE}  ({len(gated_dates)} days, "
              f"{bt_gated['n'].sum()} trades)")
        print(f"  {'month':>9}  {'days':>5}  {'trades':>7}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L $':>10}  chart")
        print(f"  {'─'*75}")
        cum = 0.0
        for ym, grp in bt_gated.groupby("ym"):
            n_t = grp["n"].sum(); n_w = grp["wins"].sum()
            ev  = (grp["n"] * grp["ev_bps"]).sum() / max(n_t, 1)
            pts = ev * AVG_PRICE / 10000 * n_t
            pnl = pts * PV
            cum += pnl
            bar_len = int(abs(pnl) / 100)
            bar = ("█" if pnl >= 0 else "░") * min(bar_len, 40)
            print(f"  {ym:>9}  {len(grp):>5}  {n_t:>7}  {n_w/n_t*100:>6.1f}  "
                  f"{ev:>+8.2f}  {pnl:>+10,.0f}  {bar}")
        print(f"  {'─'*75}")
        print(f"  {'TOTAL':>9}  {len(gated_dates):>5}  {bt_gated['n'].sum():>7}  "
              f"{bt_gated['wins'].sum()/bt_gated['n'].sum()*100:>6.1f}  "
              f"{'':>8}  {cum:>+10,.0f}")

        # ── Combined gate: ADX ≤ thr AND pl_glob_1h ≥ thr2 ─────────────
        if "pl_glob_1h" in merged.columns and merged["pl_glob_1h"].notna().sum() > 10:
            print(f"\n  Combined gate: ADX ≤ 30  AND  Globex-PL ≥ threshold")
            print(f"  {'PL_glob≥':>10}  {'n_days':>7}  {'n_trades':>9}  {'WR%':>6}  {'EV(bp)':>8}  {'P&L $':>10}")
            print(f"  {'─'*60}")
            adx_glob = merged[["date","adx_rth","pl_glob_1h"]].dropna()
            bt2 = baseline_trades.merge(adx_glob, on="date", how="inner")
            for pl_thr in [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
                sub = bt2[(bt2["adx_rth"] <= 30) & (bt2["pl_glob_1h"] >= pl_thr)]
                n  = sub["n"].sum()
                w  = sub["wins"].sum()
                if n < 5:
                    continue
                ev  = (sub["n"] * sub["ev_bps"]).sum() / max(n, 1)
                pts = ev * 5_500 / 10000 * n
                pnl = pts * cfg["point_value"]
                print(f"  {pl_thr:>10.2f}  {len(sub):>7}  {n:>9}  "
                      f"{w/n*100:>6.1f}  {ev:>+8.2f}  {pnl:>+10,.0f}")

    return merged


if __name__ == "__main__":
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "MES"
    if symbol not in SYMBOL_CONFIGS:
        print(f"Unknown symbol {symbol}. Use MES or MNQ.")
        sys.exit(1)
    run(symbol)
