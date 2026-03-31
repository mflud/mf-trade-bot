"""Quick analysis: last-hour fine breakdown for primary signal."""
import sys, math
sys.path.insert(0, "src")
import backtest_tod as bt

# Patch TOD_BUCKETS to split 15:00-16:00 into finer slices
FINE = [b for b in bt.TOD_BUCKETS if b[0] != "15:00\u201316:00"] + [
    ("15:00-15:45", 15,  0, 15, 45),
    ("15:45-16:00", 15, 45, 16,  0),
]
bt.TOD_BUCKETS = FINE

for sym, path in [("MES", "mes_hist_1min.csv"), ("MYM", "mym_hist_1min.csv")]:
    print(f"\n{'='*72}")
    print(f"  {sym}  stop={bt.PRAC_S}s / target={bt.PRAC_T}s")
    print(f"{'='*72}")

    df1  = bt.load_1min(path)
    df5  = bt.make_offset_bars(df1, 0)
    res  = bt.scan(df5)

    print(f"\n  {'Window':>14}  {'n':>5}  {'P(tgt)':>7}  {'P(stp)':>7}  {'EV':>9}  |  "
          f"{'CSR n':>6}  {'P(tgt)':>7}  {'P(stp)':>7}  {'EV':>9}")
    print(f"  {'-'*90}")

    for label in ["14:00-15:00", "15:00-15:45", "15:45-16:00"]:
        sub_all = res[res["bucket"] == label]
        sub_csr = sub_all.dropna(subset=["csr"])
        sub_csr = sub_csr[sub_csr["csr"] > bt.CSR_THRESHOLD]
        a = bt.ev_stats(sub_all, bt.PRAC_S, bt.PRAC_T)
        c = bt.ev_stats(sub_csr, bt.PRAC_S, bt.PRAC_T)
        def fmt(d):
            if math.isnan(d["ev"]):
                return f"{'—':>6}  {'—':>7}  {'—':>7}  {'—':>9}"
            return f"{d['n']:>6}  {d['p_tgt']:>7.3f}  {d['p_stop']:>7.3f}  {d['ev']:>+9.4f}s"
        print(f"  {label:>14}  {fmt(a)}  |  {fmt(c)}")

    print()
    for csr_on, tag in [(False, "No CSR filter"), (True, "CSR >= 1.5  ")]:
        sub = res.copy()
        if csr_on:
            sub = sub.dropna(subset=["csr"])
            sub = sub[sub["csr"] > bt.CSR_THRESHOLD]
        full = bt.ev_stats(sub, bt.PRAC_S, bt.PRAC_T)
        excl = bt.ev_stats(sub[sub["bucket"] != "15:45-16:00"], bt.PRAC_S, bt.PRAC_T)
        delta_n  = full["n"] - excl["n"]
        delta_ev = excl["ev"] - full["ev"]
        print(f"  [{tag}]  full: n={full['n']:4d} EV={full['ev']:+.4f}s  |  "
              f"excl 15:45-16:00: n={excl['n']:4d} EV={excl['ev']:+.4f}s  "
              f"(drop {delta_n} trades, EV {'+' if delta_ev>=0 else ''}{delta_ev:+.4f}s)")
