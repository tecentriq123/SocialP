"""Every number quoted in content/ch07.html is printed by this script.
run: python3 gen/nums_ch07.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.stats as st
from lib_ch07 import (km, surv_at, quantile_ci, reverse_km, rmst, logrank, logrank_table, logrank_k,
                      cox_efron, n_risk, events_between, small_arrays, SMALL, CUTOFF, ym, cohort, scenario, ci_from, Z)

np.set_printoptions(suppress=True)


def f(v, d=3):
    return "NR" if v is None else f"{v:.{d}f}"


print("=" * 70, "\nA. 12 patients (가, 나)")
t, s, g = small_arrays()
for pid, entry, time, stat, reason, drug in SMALL:
    print(f"  #{pid:2d} entry {ym(entry)}  last {ym(entry + time)}  time {time:2d}  status {stat}  {reason}  drug {drug}")
print("  data cutoff", ym(CUTOFF))
n = len(t)
print("  deaths", s.sum(), "censored", n - s.sum(), " lost", sum(1 for r in SMALL if r[4].startswith("전원")),
      " admin", sum(1 for r in SMALL if r[4] == "자료 마감"))
print("  naive: died %", s.sum() / n, " alive %", 1 - s.sum() / n)
print("  naive mean time (all)", t.mean(), " median all", np.median(t), " median of death times", np.median(t[s == 1]),
      " mean of death times", t[s == 1].mean())
# 2-year naive
d24 = ((t < 24) & (s == 1)).sum(); c24 = ((t < 24) & (s == 0)).sum()
print(f"  24-mo: deaths before 24 = {d24}, censored before 24 = {c24}")
print("   censored-as-alive: ", 1 - d24 / n, "  exclude early-censored:", 1 - d24 / (n - c24))
pt = t.sum()
print(f"  person-months {pt}, person-years {pt / 12:.4f}, IR per person-month {s.sum() / pt:.5f}, per 100 PY {s.sum() / (pt / 12) * 100:.2f}")
print("  follow-up: naive median all", np.median(t), " median among censored", np.median(t[s == 0]))
pot = np.array([CUTOFF - r[1] for r in SMALL])
print("  potential follow-up (cutoff-entry)", sorted(pot), "median", np.median(pot))
rk = reverse_km(t, s)
for r in rk:
    print(f"   revKM t={r['t']:4.0f} n={r['n']:2d} 'event'={r['d']} S={r['S']:.4f}")
print("  reverse KM median", quantile_ci(rk)[0], " 25th pct (G<=0.75)", quantile_ci(rk, 0.75)[0], " 75th pct (G<=0.25)", quantile_ci(rk, 0.25)[0])

rows = km(t, s)
print("\n  KM table")
for r in rows:
    lo, hi = ci_from(r["S"], r["gw"], "loglog"); plo, phi = ci_from(r["S"], r["gw"], "plain")
    print(f"   t={r['t']:4.0f} n={r['n']:2d} d={r['d']} c={r['c']}  cond={r['cond']:.4f}  S={r['S']:.4f}  gw={r['gw']:.5f}"
          f"  SE={r['S'] * np.sqrt(r['gw']):.4f}  plain=({plo:.3f},{phi:.3f})  loglog=({lo:.3f},{hi:.3f})")
S24, se24, ci24 = surv_at(rows, 24)
gw24 = [r for r in rows if r["t"] <= 24][-1]["gw"]
print(f"  S(24)={S24:.4f}  greenwood sum={gw24:.5f}  sqrt={np.sqrt(gw24):.4f}  SE={se24:.4f}")
print(f"   plain CI {S24 - Z * se24:.3f}-{S24 + Z * se24:.3f}; loglog CI {ci24[0]:.3f}-{ci24[1]:.3f}")
print(f"   log(-log S)={np.log(-np.log(S24)):.4f}  SE on that scale={np.sqrt(gw24) / abs(np.log(S24)):.4f}")
print("   greenwood terms:", [f"{r['d']}/({r['n']}x{r['n'] - r['d']})={r['d'] / (r['n'] * (r['n'] - r['d'])):.5f}" for r in rows if r['d'] > 0 and r['t'] <= 24])
S38, se38, ci38 = surv_at(rows, 38)
print(f"  S(38)={S38:.4f} SE={se38:.4f} plain CI {S38 - Z * se38:.3f}-{S38 + Z * se38:.3f}  loglog {ci38[0]:.3f}-{ci38[1]:.3f}")
for kind in ("loglog", "log", "plain"):
    print("  median + CI", kind, quantile_ci(rows, 0.5, kind))
print("  quartiles: 25% die by (S<=0.75)", quantile_ci(rows, 0.75)[0], "  S<=0.25 at", quantile_ci(rows, 0.25)[0])
a36, se36, parts = rmst(rows, 36)
print(f"  RMST(36) = {a36:.4f}  SE {se36:.4f}  CI {a36 - Z * se36:.2f}-{a36 + Z * se36:.2f}")
for a, b, sv in parts:
    print(f"    [{a:g},{b:g}) S={sv:.4f} width {b - a:g} area {sv * (b - a):.4f}")
print("  1-S(24) =", 1 - S24)

# informative censoring sensitivity: the two transferred patients die 1 month after transfer
t2 = t.copy(); s2 = s.copy()
for i, r in enumerate(SMALL):
    if r[4].startswith("전원"):
        t2[i] = r[2] + 1; s2[i] = 1
rows2 = km(t2, s2)
print(f"  sensitivity (lost -> died 1 mo later): S(24)={surv_at(rows2, 24)[0]:.4f}  median={quantile_ci(rows2)[0]}")
for r in rows2:
    if r['d']: print(f"    t={r['t']} n={r['n']} d={r['d']} S={r['S']:.4f}")
# all censored patients -> event at their censoring time (worst case, for widget explanation)
print("  if last patient (44) died:", km(t, np.where(t == 44, 1, s))[-1]["S"])

print("\n  competing-risk toy: 100 pts, 40 other-cause deaths first, then 30 cancer deaths among 60")
print("   true cumulative incidence 30/100 =", 30 / 100, " 1-KM (other deaths censored) =", 1 - (1 - 30 / 60))

print("=" * 70, "\nB. log-rank by hand (drug A = 6, drug B = 6)")
for grp, lab in ((1, "A"), (0, "B")):
    m = g == grp
    print(f"  drug {lab}: times {list(t[m].astype(int))} status {list(s[m])}")
    rr = km(t[m], s[m])
    for r in rr:
        print(f"    t={r['t']:.0f} n={r['n']} d={r['d']} c={r['c']} S={r['S']:.4f}")
    print("    median", quantile_ci(rr)[0], " S(24)", round(surv_at(rr, 24)[0], 4))
tab = logrank_table(t, s, g, 1)
for r in tab:
    print(f"   t={r['t']:3.0f}  nA={r['n1']} nB={r['n2']} n={r['n']}  dA={r['d1']} dB={r['d2']} d={r['d']}  EA={r['e1']:.4f} EB={r['e2']:.4f}  OA-EA={r['d1'] - r['e1']:+.4f}  V={r['v']:.4f}")
lr = logrank(t, s, g, 1)
print(f"  OA={lr['O1']:.0f} EA={lr['E1']:.4f} OB={lr['O2']:.0f} EB={lr['E2']:.4f}  U={lr['U']:.4f} V={lr['V']:.4f} chi2={lr['chi2']:.4f} p={lr['p']:.4f}")
approx = (lr['O1'] - lr['E1']) ** 2 / lr['E1'] + (lr['O2'] - lr['E2']) ** 2 / lr['E2']
print(f"  approx sum (O-E)^2/E = {(lr['O1'] - lr['E1']) ** 2 / lr['E1']:.4f} + {(lr['O2'] - lr['E2']) ** 2 / lr['E2']:.4f} = {approx:.4f}  p={st.chi2.sf(approx, 1):.4f}")
print(f"  O/E A={lr['O1'] / lr['E1']:.4f} B={lr['O2'] / lr['E2']:.4f} ratio={lr['O1'] / lr['E1'] / (lr['O2'] / lr['E2']):.4f}  peto one-step exp(U/V)={np.exp(lr['U'] / lr['V']):.4f}")
cx = cox_efron(t, s, g)
print(f"  Cox HR={cx['hr']:.3f} ({cx['lo']:.3f}-{cx['hi']:.3f}) p={cx['p']:.4f}  score chi2={cx['score']:.4f} p={cx['p_score']:.4f}")
for w in ("gehan", "tw", "peto", "fh01"):
    r = logrank(t, s, g, 1, weight=w)
    print(f"  {w}: chi2={r['chi2']:.4f} p={r['p']:.4f}")
# naive chi-square at 24 months
print("  alive at 24 (among not censored before 24): A", ((g == 1) & ~((t < 24) & (s == 1)) & ~((t < 24) & (s == 0))).sum(),
      " B", ((g == 0) & ~((t < 24) & (s == 1)) & ~((t < 24) & (s == 0))).sum())

print("=" * 70, "\nC. cohort (seed 2)")
C = cohort(2)
T, S_, G, IM, R = C["time"], C["status"], C["drug"], C["imdc"], C["reason"]
labs = ["favorable", "intermediate", "poor"]
for arm, lab in ((1, "A"), (0, "B")):
    m = G == arm
    cnt = [int((IM[m] == k).sum()) for k in range(3)]
    print(f"  drug {lab}: n={m.sum()} IMDC {cnt} %={[round(c / m.sum() * 100, 1) for c in cnt]}")
print("  total N", len(T), " deaths", S_.sum(), " lost", (R == "lost").sum(), " admin", (R == "admin").sum())
for arm, lab in ((1, "A"), (0, "B"), (None, "all")):
    m = np.ones(len(T), bool) if arm is None else (G == arm)
    pm = T[m].sum()
    print(f"  {lab}: events {S_[m].sum()}  lost {(R[m] == 'lost').sum()} admin {(R[m] == 'admin').sum()}  person-months {pm:.1f}  PY {pm / 12:.1f}"
          f"  rate/100PY {S_[m].sum() / (pm / 12) * 100:.2f}   % lost {(R[m] == 'lost').mean() * 100:.1f}")
    rk = reverse_km(T[m], S_[m])
    print(f"     reverse-KM median FU {quantile_ci(rk)[0]}  IQR {quantile_ci(rk, 0.75)[0]}-{quantile_ci(rk, 0.25)[0]}"
          f"   naive median all {np.median(T[m]):.1f}  among censored {np.median(T[m][S_[m] == 0]):.1f}")
print("  numbers at risk and events")
grid = [0, 12, 24, 36, 48, 60]
fits = {}
for arm, lab in ((1, "A"), (0, "B")):
    m = G == arm
    rr = km(T[m], S_[m]); fits[lab] = rr
    print(f"   {lab} at risk", [n_risk(T[m], x) for x in grid], " censored in 0-60", int(((T[m] <= 60) & (S_[m] == 0)).sum()))
    med = quantile_ci(rr)
    print(f"   {lab} median {f(med[0], 1)} ({f(med[1], 1)}-{f(med[2], 1)})   log-CI {quantile_ci(rr, 0.5, 'log')}")
    for x in (12, 24, 36):
        Sx, sex, cix = surv_at(rr, x)
        print(f"     t={x}: n.risk={n_risk(T[m], x)} n.event(prev,x]={events_between(T[m], S_[m], x - 12, x)} S={Sx:.4f} se={sex:.4f} CI {cix[0]:.4f}-{cix[1]:.4f}")
    a48, se48, _ = rmst(rr, 48)
    print(f"   RMST48 {a48:.3f} SE {se48:.3f}")
aA, seA, _ = rmst(fits["A"], 48); aB, seB, _ = rmst(fits["B"], 48)
dd = aA - aB; sed = np.sqrt(seA ** 2 + seB ** 2)
print(f"  RMST48 diff {dd:.3f} ({dd - Z * sed:.3f} to {dd + Z * sed:.3f}) p={2 * st.norm.sf(abs(dd / sed)):.4f}")
lr = logrank(T, S_, G, 1)
print(f"  log-rank: OA={lr['O1']:.0f} EA={lr['E1']:.2f} OB={lr['O2']:.0f} EB={lr['E2']:.2f} chi2={lr['chi2']:.3f} p={lr['p']:.4f}")
print(f"   O/E A {lr['O1'] / lr['E1']:.3f} B {lr['O2'] / lr['E2']:.3f} ratio {lr['O1'] / lr['E1'] / (lr['O2'] / lr['E2']):.3f}")
for w in ("gehan", "tw", "peto", "fh01"):
    r = logrank(T, S_, G, 1, weight=w); print(f"   {w} chi2={r['chi2']:.3f} p={r['p']:.4f}")
lrs = logrank(T, S_, G, 1, strata=IM)
print(f"  stratified log-rank chi2={lrs['chi2']:.3f} p={lrs['p']:.4f}  O={lrs['O1']:.0f} E={lrs['E1']:.2f}")
cx = cox_efron(T, S_, G); cxs = cox_efron(T, S_, G, strata=IM)
print(f"  Cox HR {cx['hr']:.3f} ({cx['lo']:.3f}-{cx['hi']:.3f}) p={cx['p']:.4f}; score chi2 {cx['score']:.3f}")
print(f"  stratified Cox HR {cxs['hr']:.3f} ({cxs['lo']:.3f}-{cxs['hi']:.3f}) p={cxs['p']:.4f}")
# per-stratum info
for k in range(3):
    m = IM == k
    r = logrank(T[m], S_[m], G[m], 1)
    print(f"   IMDC {labs[k]}: nA={((G == 1) & m).sum()} nB={((G == 0) & m).sum()} OA={r['O1']:.0f} EA={r['E1']:.2f} OB={r['O2']:.0f} EB={r['E2']:.2f} V={r['V']:.2f}")
# IMDC 3-group
res = logrank_k(T, S_, IM, [0, 1, 2], scores=[0, 1, 2])
print("  IMDC k-group: n", [int((IM == k).sum()) for k in range(3)], "O", res["O"], "E", res["E"].round(2), "O/E", (res["O"] / res["E"]).round(2))
print(f"   global chi2={res['chi_global']:.2f} df={res['df']} p={res['p_global']:.3g}  trend chi2={res['chi_trend']:.2f} p={res['p_trend']:.3g}")
for k in range(3):
    m = IM == k
    rr = km(T[m], S_[m]); med = quantile_ci(rr)
    print(f"   IMDC {labs[k]} median {f(med[0], 1)} ({f(med[1], 1)}-{f(med[2], 1)})  S24 {surv_at(rr, 24)[0]:.3f}")
# point-in-time naive chi-square at 24 months (excluding censored before 24) - misuse illustration
m24 = ~((T < 24) & (S_ == 0))
tabA = [((G == 1) & m24 & (T < 24) & (S_ == 1)).sum(), ((G == 1) & m24 & ~((T < 24) & (S_ == 1))).sum()]
tabB = [((G == 0) & m24 & (T < 24) & (S_ == 1)).sum(), ((G == 0) & m24 & ~((T < 24) & (S_ == 1))).sum()]
print("  naive 24-mo 2x2 (dead, alive) A", tabA, "B", tabB, "excluded censored<24:", (~m24).sum())
chi = st.chi2_contingency(np.array([tabA, tabB]), correction=False)
print(f"   chi2={chi[0]:.3f} p={chi[1]:.4f}; naive survival A {tabA[1] / sum(tabA):.3f} B {tabB[1] / sum(tabB):.3f}")

print("=" * 70, "\nD. weighted tests: early (seed 70) and late/crossing (seed 23)")
for kind, seed in (("early", 70), ("late", 23)):
    tt, ss, gg = scenario(kind, seed)
    print(f"  {kind}: n per group {int((gg == 1).sum())}/{int((gg == 0).sum())} events X {ss[gg == 1].sum()} Y {ss[gg == 0].sum()}")
    for w in ("logrank", "gehan", "tw", "peto", "fh01"):
        r = logrank(tt, ss, gg, 1, weight=w)
        print(f"    {w:8s} chi2={r['chi2']:.3f} p={r['p']:.4f}")
    fx = km(tt[gg == 1], ss[gg == 1]); fy = km(tt[gg == 0], ss[gg == 0])
    for x in (3, 6, 12, 24, 36):
        print(f"    S({x}) X={surv_at(fx, x)[0]:.3f} Y={surv_at(fy, x)[0]:.3f}")
    print("    medians X", quantile_ci(fx)[0], " Y", quantile_ci(fy)[0])
    ax, sx_, _ = rmst(fx, 36); ay, sy_, _ = rmst(fy, 36)
    d_ = ax - ay; s_ = np.sqrt(sx_ ** 2 + sy_ ** 2)
    print(f"    RMST36 X={ax:.2f} Y={ay:.2f} diff={d_:.2f} ({d_ - Z * s_:.2f} to {d_ + Z * s_:.2f}) p={2 * st.norm.sf(abs(d_ / s_)):.4f}")
    cxx = cox_efron(tt, ss, gg)
    print(f"    Cox HR (misleading average) {cxx['hr']:.2f} ({cxx['lo']:.2f}-{cxx['hi']:.2f})")

print("=" * 70, "\nE. claims example: reasons for end of follow-up (arithmetic only)")
claims = {"SGLT2i": (4812, [98, 41, 2166, 297, 2210]), "DPP-4i": (9655, [246, 157, 3862, 584, 4806])}
names = ["HHF", "death", "discontinuation", "switch/add", "end of data"]
for k, (N, v) in claims.items():
    assert sum(v) == N, (k, sum(v))
    print(f"  {k} N={N}: " + ", ".join(f"{nm} {x} ({x / N * 100:.1f}%)" for nm, x in zip(names, v)))

print("=" * 70, "\nF. extras")
C = cohort(2)
T, S_, G, IM = C["time"], C["status"], C["drug"], C["imdc"]
lr = logrank(T, S_, G, 1)
print(f"  unstratified U={lr['U']:.3f} V={lr['V']:.3f}  (O-E)^2/E A={(lr['O1'] - lr['E1']) ** 2 / lr['E1']:.3f} B={(lr['O2'] - lr['E2']) ** 2 / lr['E2']:.3f}  (O-E)^2/V={lr['U'] ** 2 / lr['V']:.3f}")
lrs = logrank(T, S_, G, 1, strata=IM)
print(f"  stratified U={lrs['U']:.3f} V={lrs['V']:.3f}")
fa = km(T[G == 1], S_[G == 1]); fb = km(T[G == 0], S_[G == 0])
for x in (6, 12, 24, 36, 48):
    Sa, sa, _ = surv_at(fa, x); Sb, sb, _ = surv_at(fb, x)
    z = (Sa - Sb) / np.sqrt(sa ** 2 + sb ** 2)
    print(f"  t={x}: SA={Sa:.3f} SB={Sb:.3f} diff={Sa - Sb:.3f} z={z:.2f} p={2 * st.norm.sf(abs(z)):.4f}")
# crude rate ratio
pmA = T[G == 1].sum() / 12; pmB = T[G == 0].sum() / 12
print(f"  crude rate ratio {(S_[G == 1].sum() / pmA) / (S_[G == 0].sum() / pmB):.3f}")
# 1-KM at 24 for small data
t, s, g = small_arrays()
print("  small: 1-S(24)", 1 - surv_at(km(t, s), 24)[0])
# HR is not a risk ratio: under PH, S_A(t) = S_B(t)^HR
cx = cox_efron(T, S_, G)
SB36 = surv_at(fb, 36)[0]
SA_pred = SB36 ** cx["hr"]
print(f"  PH illustration: S_B(36)={SB36:.4f} risk_B={1 - SB36:.4f}; HR={cx['hr']:.4f} -> S_A={SA_pred:.4f} risk_A={1 - SA_pred:.4f} RR={(1 - SA_pred) / (1 - SB36):.3f}")
print(f"   with rounded inputs: 1-0.278**0.73 = {1 - 0.278 ** 0.73:.4f}, RR = {(1 - 0.278 ** 0.73) / 0.722:.3f}")
# same curves, 10x sample: replicate data 10 times
T10, S10, G10 = np.tile(T, 10), np.tile(S_, 10), np.tile(G, 10)
r10 = logrank(T10, S10, G10, 1)
print(f"  10x replicated cohort: chi2={r10['chi2']:.2f} p={r10['p']:.2e}")

print("=" * 70, "\nG. 파이썬 출력 상자 (lifelines; run with: source /home/claude/pylibs/env.sh)")
try:
    import warnings
    import pandas as pd
    from lifelines import KaplanMeierFitter
    from lifelines.utils import median_survival_times
    from lifelines.statistics import logrank_test
    warnings.filterwarnings("ignore")
    C = cohort(2)
    rcc = pd.DataFrame({"time": C["time"], "status": C["status"], "drug": np.where(C["drug"] == 1, "A", "B")})
    tt = [12, 24, 36]
    for d in ("A", "B"):
        sub = rcc[rcc["drug"] == d]
        kmf = KaplanMeierFitter().fit(sub["time"], event_observed=sub["status"], label=d)
        print(kmf)
        print(" median", kmf.median_survival_time_, "\n", median_survival_times(kmf.confidence_interval_))
        print(pd.concat([kmf.survival_function_at_times(tt), kmf.confidence_interval_.asof(tt)], axis=1))
        print(" at risk", [int((sub["time"] >= x).sum()) for x in tt])
    a, b = rcc[rcc["drug"] == "A"], rcc[rcc["drug"] == "B"]
    res = logrank_test(a["time"], b["time"], event_observed_A=a["status"], event_observed_B=b["status"])
    res.print_summary()
    print(" exact", res.test_statistic, res.p_value)
    t, s, g = small_arrays()
    kmf = KaplanMeierFitter().fit(t, s)
    print(kmf.survival_function_.join(kmf.confidence_interval_))
    print(median_survival_times(kmf.confidence_interval_))
    for kind, seed in (("early", 70), ("late", 23)):
        x, e, gg = scenario(kind, seed)
        A, B = gg == 1, gg == 0
        for nm, kw in (("logrank", {}), ("wilcoxon", dict(weightings="wilcoxon")),
                       ("tarone-ware", dict(weightings="tarone-ware")), ("peto", dict(weightings="peto")),
                       ("FH(1,0)", dict(weightings="fleming-harrington", p=1, q=0)),
                       ("FH(0,1)", dict(weightings="fleming-harrington", p=0, q=1))):
            r = logrank_test(x[A], x[B], e[A], e[B], **kw)
            print(f"  {kind} {nm:12s} chi2={r.test_statistic:.4f} p={r.p_value:.4f}")
except ImportError as e:
    print("  (skipped:", e, ")")
