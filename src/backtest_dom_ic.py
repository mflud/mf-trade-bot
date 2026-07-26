"""
DOM Imbalance → Forward Return Information Coefficient study.

Loads MES DOM snapshots (5-min intervals, RTH only) and computes:
  - Spearman IC between each DOM feature and forward mid-price return
    at horizons of +1 snap (5 min), +2 (10 min), +3 (15 min)
  - Win rate: does sign(imbalance) match sign(forward return)?
  - Quantile spread: avg return in top vs bottom imbalance quintile
  - Wall distance analysis: does a large resting wall nearby predict reversal?

Usage:
  python src/backtest_dom_ic.py
"""

import csv
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Load & merge DOM snapshots ────────────────────────────────────────────────

SNAP_PATHS = [
    Path("mes_dom_snapshots.csv"),
    Path("data/mes_dom_snapshots.csv"),
]

def load_dom(paths):
    rows = []
    for p in paths:
        if p.exists():
            with open(p) as f:
                rows.extend(list(csv.DictReader(f)))
    rows.sort(key=lambda r: r["ts"])
    # deduplicate by ts
    seen = set()
    deduped = []
    for r in rows:
        if r["ts"] not in seen:
            seen.add(r["ts"])
            deduped.append(r)
    return deduped

def parse_float(v):
    try:
        return float(v) if v not in ("", "None", None) else None
    except (ValueError, TypeError):
        return None

def ts_to_dt(ts_str):
    # handles both '2026-03-20T14:29:59' and '2026-03-20T14:29:59+00:00'
    ts_str = ts_str.replace("Z", "+00:00")
    if "+" not in ts_str[10:] and "-" not in ts_str[11:]:
        ts_str += "+00:00"
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None

def is_rth(dt):
    """13:30–20:00 UTC = 09:30–16:00 ET"""
    if dt is None:
        return False
    t = dt.hour * 60 + dt.minute
    return 13 * 60 + 30 <= t < 20 * 60

# ── IC helpers ────────────────────────────────────────────────────────────────

def spearman_ic(x, y):
    """Spearman rank correlation between x and y (numpy arrays)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan, len(x)
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    corr = np.corrcoef(rx, ry)[0, 1]
    return corr, len(x)

def win_rate(x, y):
    """Fraction of rows where sign(x) == sign(y), excluding zeros."""
    mask = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan, len(x)
    return float(np.mean(np.sign(x) == np.sign(y))), len(x)

def quantile_spread(x, y, q=5):
    """Mean forward return in top vs bottom quintile of x."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < q * 4:
        return np.nan, np.nan
    cutoffs = np.percentile(x, [100 / q, 100 - 100 / q])
    bot = y[x <= cutoffs[0]]
    top = y[x >= cutoffs[1]]
    return float(np.mean(top)) if len(top) else np.nan, float(np.mean(bot)) if len(bot) else np.nan

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading DOM snapshots…")
    all_rows = load_dom(SNAP_PATHS)
    print(f"  Total: {len(all_rows):,} rows after dedup")

    # Parse and filter RTH
    records = []
    for r in all_rows:
        dt = ts_to_dt(r["ts"])
        if not is_rth(dt):
            continue
        mid = parse_float(r.get("mid"))
        if mid is None:
            mid_b = parse_float(r.get("best_bid"))
            mid_a = parse_float(r.get("best_ask"))
            if mid_b and mid_a:
                mid = (mid_b + mid_a) / 2
        if mid is None:
            continue
        records.append({
            "dt":    dt,
            "mid":   mid,
            "imb_l1":  parse_float(r.get("bid_ask_imbalance_l1")),
            "imb_l5":  parse_float(r.get("bid_ask_imbalance_l5")),
            "imb_l10": parse_float(r.get("bid_ask_imbalance_l10")),
            "bid_wall_sz":   parse_float(r.get("max_bid_wall_size")),
            "bid_wall_dist": parse_float(r.get("max_bid_wall_dist")),
            "ask_wall_sz":   parse_float(r.get("max_ask_wall_size")),
            "ask_wall_dist": parse_float(r.get("max_ask_wall_dist")),
            "total_bid": parse_float(r.get("total_bid_size")),
            "total_ask": parse_float(r.get("total_ask_size")),
        })

    print(f"  RTH rows with valid mid: {len(records):,}")

    # Build forward returns at +1/+2/+3 snapshots (5/10/15 min)
    # Match by position — snapshots are ~5 min apart; allow ±90s slop
    HORIZONS = [1, 2, 3]
    fwd_rets = {h: [] for h in HORIZONS}  # parallel to records

    for i, rec in enumerate(records):
        for h in HORIZONS:
            j = i + h
            if j < len(records):
                # Verify timestamps are roughly h×5 min apart
                gap = (records[j]["dt"] - rec["dt"]).total_seconds()
                expected = h * 5 * 60
                if abs(gap - expected) < 90 * h:   # allow 90s slop per hop
                    ret = (records[j]["mid"] - rec["mid"]) / rec["mid"] * 10000  # in bp
                    fwd_rets[h].append(ret)
                else:
                    fwd_rets[h].append(np.nan)
            else:
                fwd_rets[h].append(np.nan)

    features = {
        "imb_l1":          np.array([r["imb_l1"]  if r["imb_l1"]  is not None else np.nan for r in records]),
        "imb_l5":          np.array([r["imb_l5"]  if r["imb_l5"]  is not None else np.nan for r in records]),
        "imb_l10":         np.array([r["imb_l10"] if r["imb_l10"] is not None else np.nan for r in records]),
        "bid_wall_sz":     np.array([r["bid_wall_sz"]   if r["bid_wall_sz"]   is not None else np.nan for r in records]),
        "ask_wall_sz":     np.array([r["ask_wall_sz"]   if r["ask_wall_sz"]   is not None else np.nan for r in records]),
        "bid_wall_dist":   np.array([r["bid_wall_dist"] if r["bid_wall_dist"] is not None else np.nan for r in records]),
        "ask_wall_dist":   np.array([r["ask_wall_dist"] if r["ask_wall_dist"] is not None else np.nan for r in records]),
        "wall_imb":        np.array([                       # bid_wall - ask_wall (signed)
            (r["bid_wall_sz"] - r["ask_wall_sz"])
            if r["bid_wall_sz"] is not None and r["ask_wall_sz"] is not None
            else np.nan
            for r in records
        ]),
        "net_depth":       np.array([                       # (total_bid - total_ask) / (total_bid + total_ask)
            (r["total_bid"] - r["total_ask"]) / (r["total_bid"] + r["total_ask"])
            if r["total_bid"] is not None and r["total_ask"] is not None and (r["total_bid"] + r["total_ask"]) > 0
            else np.nan
            for r in records
        ]),
    }

    fwd_arrays = {h: np.array(fwd_rets[h]) for h in HORIZONS}

    # ── Results ───────────────────────────────────────────────────────────────

    print()
    print("═" * 80)
    print("  DOM IMBALANCE → FORWARD RETURN  (MES, RTH, 5-min snapshots)")
    print("  Forward return in bp from snapshot mid to +N×5min mid")
    print("═" * 80)

    feat_labels = {
        "imb_l1":        "L1 imbalance    (bid-ask)/(bid+ask) at L1",
        "imb_l5":        "L5 imbalance    (bid-ask)/(bid+ask) at L5",
        "imb_l10":       "L10 imbalance   (bid-ask)/(bid+ask) at L10",
        "net_depth":     "Net depth L10   (total_bid-total_ask)/(total)",
        "wall_imb":      "Wall imbalance  max_bid_wall - max_ask_wall",
        "bid_wall_sz":   "Bid wall size   largest resting bid order",
        "ask_wall_sz":   "Ask wall size   largest resting ask order",
        "bid_wall_dist": "Bid wall dist   distance from mid (pts)",
        "ask_wall_dist": "Ask wall dist   distance from mid (pts)",
    }

    for fname, flabel in feat_labels.items():
        fx = features[fname]
        print(f"\n── {flabel}")
        print(f"   {'Horizon':>8}  {'IC':>7}  {'WR':>7}  {'N':>6}  {'Top Q':>8}  {'Bot Q':>8}  {'Spread':>8}")
        print(f"   {'─'*66}")
        for h in HORIZONS:
            fy = fwd_arrays[h]
            ic, n   = spearman_ic(fx, fy)
            wr, n2  = win_rate(fx, fy)
            tq, bq  = quantile_spread(fx, fy)
            spread  = (tq - bq) if (tq is not None and bq is not None and not np.isnan(tq) and not np.isnan(bq)) else np.nan
            ic_str  = f"{ic:+.4f}" if not np.isnan(ic) else "   —   "
            wr_str  = f"{wr*100:.1f}%" if not np.isnan(wr) else "  —  "
            tq_str  = f"{tq:+.3f}bp" if not np.isnan(tq) else "    —   "
            bq_str  = f"{bq:+.3f}bp" if not np.isnan(bq) else "    —   "
            sp_str  = f"{spread:+.3f}bp" if not np.isnan(spread) else "    —   "
            print(f"   +{h*5:>3}min   {ic_str}  {wr_str}  {n:>6,}  {tq_str}  {bq_str}  {sp_str}")

    # ── Wall proximity study ───────────────────────────────────────────────────
    print()
    print("═" * 80)
    print("  WALL PROXIMITY vs FORWARD RETURN")
    print("  When a large bid/ask wall is very close (<2pts from mid),")
    print("  does price tend to reverse at (ask wall) or bounce from (bid wall)?")
    print("═" * 80)

    mid_arr = np.array([r["mid"] for r in records])

    for side, wall_sz_key, wall_dist_key, direction in [
        ("Bid wall CLOSE (<2pts)", "bid_wall_sz", "bid_wall_dist", +1),
        ("Ask wall CLOSE (<2pts)", "ask_wall_sz", "ask_wall_dist", -1),
    ]:
        wsz  = features[wall_sz_key]
        wdst = features[wall_dist_key]
        med_wsz = np.nanmedian(wsz[wsz > 0])

        # Close = wall within 2pts; Large = above median
        close_large = (wdst < 2.0) & (wsz >= med_wsz) & np.isfinite(wsz) & np.isfinite(wdst)
        close_small = (wdst < 2.0) & (wsz < med_wsz)  & np.isfinite(wsz) & np.isfinite(wdst)
        far         = (wdst >= 2.0)                     & np.isfinite(wdst)

        print(f"\n  {side}  (median wall size = {med_wsz:.0f})")
        for label, mask in [("Close+Large", close_large), ("Close+Small", close_small), ("Far (≥2pts)", far)]:
            if mask.sum() < 5:
                continue
            for h in [1, 2]:
                fy = fwd_arrays[h]
                valid = mask & np.isfinite(fy)
                if valid.sum() < 5:
                    continue
                avg_ret = np.mean(fy[valid]) * direction   # sign: +1 = bounce, -1 = reverse
                n_v = valid.sum()
                print(f"    {label:15}  +{h*5}min  n={n_v:4d}  avg_directed_ret={avg_ret:+.4f}bp")

    # ── Decile table for best feature ─────────────────────────────────────────
    print()
    print("═" * 80)
    print("  DECILE BREAKDOWN — L10 imbalance vs +5min forward return")
    print("  (shows whether IC is driven by extremes or is monotonic)")
    print("═" * 80)

    fx = features["imb_l10"]
    fy = fwd_arrays[1]
    mask = np.isfinite(fx) & np.isfinite(fy)
    fx_m, fy_m = fx[mask], fy[mask]
    decile_edges = np.percentile(fx_m, np.arange(0, 110, 10))
    print(f"\n  {'Decile':>8}  {'Imb range':>20}  {'N':>6}  {'Avg fwd ret':>12}  {'WR':>7}")
    print(f"  {'─'*60}")
    for d in range(10):
        lo, hi = decile_edges[d], decile_edges[d + 1]
        if d == 9:
            sel = (fx_m >= lo) & (fx_m <= hi)
        else:
            sel = (fx_m >= lo) & (fx_m < hi)
        n_d = sel.sum()
        if n_d == 0:
            continue
        avg = np.mean(fy_m[sel])
        wr  = np.mean(np.sign(fy_m[sel]) > 0)
        print(f"  D{d+1:>2} (P{d*10:>2}–P{(d+1)*10:>2})  {lo:>+8.3f} → {hi:>+8.3f}  {n_d:>6,}  {avg:>+12.4f}bp  {wr*100:>6.1f}%")

    print()


if __name__ == "__main__":
    main()
