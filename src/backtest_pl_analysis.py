"""
Deep analysis of PL momentum strategy.
Fixed params: window=30s (6 bars), PL>=0.70, move>=12bp, exit_pl=0.40,
              stop=10bp, min_hold=10s, max_hold=120s

Covers:
  1. Month-by-month and year stability
  2. Time-of-day breakdown (30-min buckets)
  3. Stop size sensitivity (6, 8, 10, 12, 15, 20bp)
  4. Exit PL threshold sweep (0.20, 0.30, 0.40, 0.50, 0.60)
  5. Move threshold fine sweep (6, 8, 10, 12, 14, 16bp)
  6. Entry PL fine sweep (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

Usage:
  python src/backtest_pl_analysis.py                       # ES
  python src/backtest_pl_analysis.py --csv nq_hist_5sec.csv --label NQ
"""

import argparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET                   = ZoneInfo("America/New_York")
SETTLEMENT_UTC_START = 21
SETTLEMENT_UTC_END   = 22

# Fixed baseline
WINDOW    = 6       # bars = 30s
ENTRY_PL  = 0.70
MOVE_BPS  = 12.0
EXIT_PL   = 0.40
STOP_BPS  = 10.0
MIN_HOLD  = 2       # bars = 10s
MAX_HOLD  = 24      # bars = 120s


def load_5s_rth(csv_path):
    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    h  = df["ts"].dt.hour
    df = df[~((h >= SETTLEMENT_UTC_START) & (h < SETTLEMENT_UTC_END))].copy()
    ts_et = df["ts"].dt.tz_convert(ET)
    hm    = ts_et.dt.hour * 60 + ts_et.dt.minute
    df    = df[(hm >= 570) & (hm < 960)].copy()
    df = df.reset_index(drop=True)
    df["gap"] = (df["ts"].diff() > pd.Timedelta(seconds=10)).values
    df.iloc[0, df.columns.get_loc("gap")] = True
    ts_et2      = df["ts"].dt.tz_convert(ET)
    df["hm_et"] = (ts_et2.dt.hour * 60 + ts_et2.dt.minute).values
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    df["year"]  = df["ts"].dt.year.values
    return df


def compute_pl_raw(df, window, exit_pl_arr=None):
    """
    Compute PL on raw log returns. Uses a fixed rolling exit_pl array
    (same window) unless exit_pl_arr is overridden.
    Returns (pl_arr, dir_arr, move_bp_arr).
    """
    close = df["close"].values
    gap   = df["gap"].values.astype(bool)
    n     = len(df)
    lr    = np.empty(n); lr[:] = np.nan
    lr[1:] = np.log(close[1:] / close[:-1])
    lr[gap] = np.nan

    pl_arr   = np.full(n, np.nan)
    dir_arr  = np.zeros(n)
    move_arr = np.full(n, np.nan)

    for i in range(window, n):
        if gap[i - window + 1: i + 1].any():
            continue
        rets = lr[i - window + 1: i + 1]
        if np.isnan(rets).any():
            continue
        sa = np.abs(rets).sum()
        if sa == 0:
            continue
        net = rets.sum()
        pl_arr[i]  = abs(net) / sa
        dir_arr[i] = 1.0 if net > 0 else -1.0
        move_arr[i] = abs(close[i] - close[i - window]) / close[i] * 10000

    return pl_arr, dir_arr, move_arr


def compute_rolling_rvol(df, lookback_days=20):
    """
    Per-bar 20-day rolling realised vol (annualised %).
    daily_rvol[d] = std(5s log returns on day d) * sqrt(4680) * 100
    rolling_rvol[d] = mean of last lookback_days daily rvols
    Returns array of length len(df).
    """
    close = df["close"].values
    gap   = df["gap"].values.astype(bool)
    n     = len(df)

    lr = np.empty(n); lr[:] = np.nan
    lr[1:] = np.log(close[1:] / close[:-1])
    lr[gap] = np.nan

    ts_et  = df["ts"].dt.tz_convert(ET)
    dates  = ts_et.dt.date.values

    # Daily rvol
    unique_dates = sorted(set(dates))
    daily_rv = {}
    for d in unique_dates:
        mask   = dates == d
        day_lr = lr[mask]
        day_lr = day_lr[~np.isnan(day_lr)]
        if len(day_lr) >= 10:
            daily_rv[d] = float(np.std(day_lr) * np.sqrt(4680) * 100)

    # Rolling mean over last lookback_days
    date_list   = sorted(daily_rv.keys())
    rolling_rv  = {}
    for i, d in enumerate(date_list):
        window = date_list[max(0, i - lookback_days + 1): i + 1]
        rolling_rv[d] = float(np.mean([daily_rv[dd] for dd in window]))

    rvol_arr = np.full(n, np.nan)
    for i in range(n):
        d = dates[i]
        if d in rolling_rv:
            rvol_arr[i] = rolling_rv[d]
    return rvol_arr


def simulate_full(df, pl_arr, dir_arr, move_arr,
                  entry_pl, min_move, exit_pl_thresh, stop_bps, min_hold, max_hold,
                  rvol_arr=None, min_rvol=0.0):
    """Simulate with full metadata per trade for slicing."""
    close  = df["close"].values
    high   = df["high"].values
    low    = df["low"].values
    gap    = df["gap"].values.astype(bool)
    hm_et  = df["hm_et"].values
    month  = df["month"].values
    year   = df["year"].values
    n      = len(df)

    hold_until = -1
    records    = []

    for i in range(n - max_hold - 2):
        if i <= hold_until:
            continue
        if np.isnan(pl_arr[i]) or pl_arr[i] < entry_pl:
            continue
        if np.isnan(move_arr[i]) or move_arr[i] < min_move:
            continue
        if gap[i]:
            continue
        if rvol_arr is not None and (np.isnan(rvol_arr[i]) or rvol_arr[i] < min_rvol):
            continue
        direction = dir_arr[i]
        if direction == 0:
            continue

        eb = i + 1
        if eb >= n - max_hold - 1 or gap[eb]:
            continue

        entry  = close[eb]
        sl_pts = entry * stop_bps / 10000

        exit_bar = None
        exit_pnl = None

        for k in range(1, max_hold + 1):
            j = eb + k
            if j >= n:
                break
            if direction == 1 and low[j] <= entry - sl_pts:
                exit_bar = j; exit_pnl = -sl_pts; break
            if direction == -1 and high[j] >= entry + sl_pts:
                exit_bar = j; exit_pnl = -sl_pts; break
            if k >= min_hold and not np.isnan(pl_arr[j]) and pl_arr[j] <= exit_pl_thresh:
                exit_bar = j
                exit_pnl = direction * (close[j] - entry)
                break

        if exit_bar is None:
            exit_bar = eb + max_hold
            if exit_bar < n:
                exit_pnl = direction * (close[exit_bar] - entry)
            else:
                continue

        hold_until = exit_bar
        records.append({
            "direction": direction,
            "pnl_bp":    exit_pnl / entry * 10000,
            "hold_s":    (exit_bar - eb) * 5,
            "hm_et":     int(hm_et[eb]),
            "month":     month[eb],
            "year":      int(year[eb]),
        })

    return pd.DataFrame(records)


def stats(t):
    if len(t) < 5:
        return None
    return {
        "n":    len(t),
        "ev":   t["pnl_bp"].mean(),
        "win":  (t["pnl_bp"] > 0).mean() * 100,
        "hold": t["hold_s"].mean(),
    }


def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")


def run(csv_path, label):
    print(f"Loading {csv_path} ...")
    df = load_5s_rth(csv_path)
    print(f"  {len(df):,} 5s RTH bars  ({df['ts'].min().date()} -> {df['ts'].max().date()})")
    print(f"  Baseline: window={WINDOW*5}s  PL>={ENTRY_PL}  move>={MOVE_BPS}bp  "
          f"exit_pl={EXIT_PL}  stop={STOP_BPS}bp  min_hold={MIN_HOLD*5}s  max_hold={MAX_HOLD*5}s")

    pl_arr, dir_arr, move_arr = compute_pl_raw(df, WINDOW)

    # ── Baseline trades ───────────────────────────────────────────────────────
    base = simulate_full(df, pl_arr, dir_arr, move_arr,
                         ENTRY_PL, MOVE_BPS, EXIT_PL, STOP_BPS, MIN_HOLD, MAX_HOLD)
    st = stats(base)
    print(f"\n  BASELINE: {st['n']}n  EV={st['ev']:+.2f}bp  win={st['win']:.0f}%  hold={st['hold']:.0f}s")
    longs  = base[base["direction"] ==  1]
    shorts = base[base["direction"] == -1]
    print(f"    LONG  {len(longs):5d}n  EV={longs['pnl_bp'].mean():+.2f}bp  win={100*(longs['pnl_bp']>0).mean():.0f}%")
    print(f"    SHORT {len(shorts):5d}n  EV={shorts['pnl_bp'].mean():+.2f}bp  win={100*(shorts['pnl_bp']>0).mean():.0f}%")

    # ── 1. Month-by-month ─────────────────────────────────────────────────────
    section(f"{label} — 1. Month-by-month stability")
    print(f"  {'Month':8}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}  bar")
    print(f"  {'-'*50}")
    for mo in sorted(base["month"].unique()):
        sub = base[base["month"] == mo]
        st  = stats(sub)
        if st is None:
            continue
        bar = "+" * int(max(0, st["ev"]) * 2) + ("-" * int(max(0, -st["ev"]) * 2))
        print(f"  {mo:8}  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  {st['hold']:>7.0f}s  {bar}")

    # ── 2. Time-of-day ────────────────────────────────────────────────────────
    section(f"{label} — 2. Time-of-day (30-min buckets, ET)")
    buckets = list(range(570, 960, 30))
    print(f"  {'Time(ET)':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}  bar")
    print(f"  {'-'*55}")
    for b in buckets:
        sub = base[(base["hm_et"] >= b) & (base["hm_et"] < b + 30)]
        st  = stats(sub)
        t   = f"{b//60}:{b%60:02d}-{(b+30)//60}:{(b+30)%60:02d}"
        if st is None:
            print(f"  {t:10}  {'<5':>5}")
            continue
        bar = "+" * int(max(0, st["ev"]) * 2) + ("-" * int(max(0, -st["ev"]) * 2))
        print(f"  {t:10}  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  {st['hold']:>7.0f}s  {bar}")

    # ── 3. Stop size sensitivity ──────────────────────────────────────────────
    section(f"{label} — 3. Stop size sensitivity")
    stops = [6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    print(f"  {'stop(bp)':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'stops%':>7}  {'hold(s)':>8}")
    print(f"  {'-'*55}")
    for s in stops:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ENTRY_PL, MOVE_BPS, EXIT_PL, s, MIN_HOLD, MAX_HOLD)
        st = stats(t)
        if st is None:
            print(f"  {s:>10.0f}bp  <5 trades"); continue
        stop_pct = 100 * (t["pnl_bp"] < -s * 0.9).mean()
        print(f"  {s:>10.1f}bp  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  "
              f"{stop_pct:>6.0f}%  {st['hold']:>7.0f}s")

    # ── 4. Exit PL threshold sweep ────────────────────────────────────────────
    section(f"{label} — 4. Exit PL threshold sweep")
    exit_pls = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    print(f"  {'exit_pl':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}")
    print(f"  {'-'*48}")
    for ep in exit_pls:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ENTRY_PL, MOVE_BPS, ep, STOP_BPS, MIN_HOLD, MAX_HOLD)
        st = stats(t)
        if st is None:
            print(f"  {ep:>10.2f}  <5 trades"); continue
        print(f"  {ep:>10.2f}  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  {st['hold']:>7.0f}s")

    # ── 5. Move threshold fine sweep ──────────────────────────────────────────
    section(f"{label} — 5. Move threshold fine sweep (bp)")
    moves = [0, 4, 6, 8, 10, 12, 14, 16, 20]
    print(f"  {'move(bp)':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}  bar")
    print(f"  {'-'*55}")
    for m in moves:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ENTRY_PL, float(m), EXIT_PL, STOP_BPS, MIN_HOLD, MAX_HOLD)
        st = stats(t)
        if st is None:
            print(f"  {m:>10}bp  <5 trades"); continue
        bar = "+" * int(max(0, st["ev"]) * 1.5)
        print(f"  {m:>10}bp  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  "
              f"{st['hold']:>7.0f}s  {bar}")

    # ── 6. Entry PL fine sweep ────────────────────────────────────────────────
    section(f"{label} — 6. Entry PL fine sweep")
    entry_pls = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    print(f"  {'entry_pl':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}  bar")
    print(f"  {'-'*55}")
    for ep in entry_pls:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ep, MOVE_BPS, EXIT_PL, STOP_BPS, MIN_HOLD, MAX_HOLD)
        st = stats(t)
        if st is None:
            print(f"  {ep:>10.2f}  <5 trades"); continue
        bar = "+" * int(max(0, st["ev"]) * 1.5)
        print(f"  {ep:>10.2f}  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  "
              f"{st['hold']:>7.0f}s  {bar}")

    # ── 7. Max hold sweep ─────────────────────────────────────────────────────
    section(f"{label} — 7. Max hold sweep")
    max_holds = [6, 12, 18, 24, 36, 48, 72]   # bars (30s to 6min)
    print(f"  {'max_hold':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}  time_exit%")
    print(f"  {'-'*60}")
    for mh in max_holds:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ENTRY_PL, MOVE_BPS, EXIT_PL, STOP_BPS, MIN_HOLD, mh)
        st = stats(t)
        if st is None:
            print(f"  {mh*5:>8}s  <5 trades"); continue
        time_exit_pct = 100 * (t["hold_s"] >= mh * 5 - 1).mean()
        print(f"  {mh*5:>8}s  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  "
              f"{st['hold']:>7.0f}s  {time_exit_pct:.0f}%")

    # ── 8. Rolling 20-day RVol gate ───────────────────────────────────────────
    section(f"{label} — 8. Rolling 20-day realised vol gate (annualised %)")
    print(f"  Computing 20-day rolling RVol …")
    rvol_arr = compute_rolling_rvol(df, lookback_days=20)
    valid    = rvol_arr[~np.isnan(rvol_arr)]
    print(f"  RVol stats:  p25={np.percentile(valid,25):.1f}%  "
          f"p50={np.percentile(valid,50):.1f}%  "
          f"p75={np.percentile(valid,75):.1f}%  "
          f"p90={np.percentile(valid,90):.1f}%  "
          f"max={valid.max():.1f}%")

    rvol_thresholds = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    print(f"\n  {'RVol≥':10}  {'n':>5}  {'EV(bp)':>8}  {'win%':>6}  {'hold(s)':>8}  "
          f"{'days_active':>12}  bar")
    print(f"  {'-'*70}")
    total_days = len(df["month"].unique())  # rough
    for rv in rvol_thresholds:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ENTRY_PL, MOVE_BPS, EXIT_PL, STOP_BPS, MIN_HOLD, MAX_HOLD,
                          rvol_arr=rvol_arr, min_rvol=rv)
        # Count distinct trading days that pass the RVol gate
        ts_et2     = df["ts"].dt.tz_convert(ET)
        dates_arr  = ts_et2.dt.date.values
        gate_days  = len(set(dates_arr[~np.isnan(rvol_arr) & (rvol_arr >= rv)]))
        lbl        = f">={rv:.0f}%" if rv > 0 else "no gate"
        st = stats(t)
        if st is None:
            print(f"  {lbl:10}  {'<5':>5}  {'':>8}  {'':>6}  {'':>8}  {gate_days:>12}")
            continue
        bar = "+" * int(max(0, st["ev"]) * 2) + ("-" * int(max(0, -st["ev"]) * 2))
        print(f"  {lbl:10}  {st['n']:>5}  {st['ev']:>+8.2f}  {st['win']:>5.0f}%  "
              f"{st['hold']:>7.0f}s  {gate_days:>12}  {bar}")

    # Direction split at best threshold (first non-trivial gate)
    print(f"\n  Direction split by RVol gate:")
    for rv in [0.0, 5.0, 8.0]:
        t = simulate_full(df, pl_arr, dir_arr, move_arr,
                          ENTRY_PL, MOVE_BPS, EXIT_PL, STOP_BPS, MIN_HOLD, MAX_HOLD,
                          rvol_arr=rvol_arr, min_rvol=rv)
        if t.empty or len(t) < 5:
            continue
        longs  = t[t["direction"] ==  1]
        shorts = t[t["direction"] == -1]
        lbl    = f"RVol>={rv:.0f}%" if rv > 0 else "no gate"
        print(f"  {lbl}:", end="")
        if len(longs)  >= 5: print(f"  LONG  {len(longs):4d}n  EV={longs['pnl_bp'].mean():+.2f}bp  win={100*(longs['pnl_bp']>0).mean():.0f}%", end="")
        if len(shorts) >= 5: print(f"  |  SHORT {len(shorts):4d}n  EV={shorts['pnl_bp'].mean():+.2f}bp  win={100*(shorts['pnl_bp']>0).mean():.0f}%", end="")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   default="es_hist_5sec.csv")
    parser.add_argument("--label", default="ES")
    args = parser.parse_args()
    run(args.csv, args.label)
