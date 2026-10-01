"""Every number quoted in content/ch12.html is printed by this script.
run: python3 gen/nums_ch12.py
Python outputs in the HTML (statsmodels summaries) and the negative binomial SEs quoted in the text
(observed information, e.g. drug A SE 0.0718, Table 3 female 0.75-1.01) come from gen/pyout_ch12.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.stats as st
from lib_ch12 import (garwood, wald_rate, log_rate, compare_rates, poisson_glm, null_poisson, sandwich_hc0,
                      nb_glm, nb_null_deviance, nb_pmf, cohort, design, binom_test_two_sided, Z)

np.set_printoptions(suppress=True, linewidth=140)
H = lambda s: print("\n" + "=" * 78 + "\n" + s)

# ------------------------------------------------------------------ 가-1 pmf
H("가-1. Poisson pmf: ADR reports, lambda = 4 per month")
lam = 4
for k in range(0, 13):
    print(f"  P(X={k:2d}) = {st.poisson.pmf(k, lam):.4f}   P(X<={k}) = {st.poisson.cdf(k, lam):.4f}   P(X>={k}) = {st.poisson.sf(k - 1, lam):.4f}")
print("  mean", lam, "variance", lam, "SD", np.sqrt(lam))
for l in (1, 4, 10):
    print(f"  lambda={l}: P0={st.poisson.pmf(0, l):.4f}  mode(s)={[k for k in range(30) if abs(st.poisson.pmf(k, l) - st.poisson.pmf(np.arange(30), l).max()) < 1e-12]}"
          f"  SD={np.sqrt(l):.2f}  P(within mean±2SD)={st.poisson.cdf(np.floor(l + 2 * np.sqrt(l)), l) - st.poisson.cdf(np.ceil(l - 2 * np.sqrt(l)) - 1, l):.3f}")

# ------------------------------------------------------------------ 가-2 binomial vs Poisson
H("가-2. Binomial vs Poisson (lambda = n p = 3)")
for k in range(0, 8):
    print(f"  k={k}: Poisson(3)={st.poisson.pmf(k, 3):.4f}  Bin(10000,0.0003)={st.binom.pmf(k, 10000, 0.0003):.4f}  Bin(20,0.15)={st.binom.pmf(k, 20, 0.15):.4f}")
print(f"  k>=8: Poisson={st.poisson.sf(7, 3):.4f}  Bin(10000)={st.binom.sf(7, 10000, 0.0003):.4f}  Bin(20)={st.binom.sf(7, 20, 0.15):.4f}")
print("  variances: Poisson 3, Bin(10000,.0003)", 10000 * 0.0003 * (1 - 0.0003), " Bin(20,.15)", 20 * 0.15 * 0.85)

# ------------------------------------------------------------------ 가-3 rate CIs
H("가-3. CI for a rate")
for d, t in ((3, 1500), (0, 2300), (150, 25000), (1, 1500), (10, 1500)):
    g = garwood(d, t)
    w = wald_rate(d, t)
    s = f"  D={d} T={t}: rate/1000PY={d / t * 1000:.3f}  exact=({g[0] * 1000:.3f}, {g[1] * 1000:.3f})  wald=({w[0] * 1000:.3f}, {w[1] * 1000:.3f})"
    if d > 0:
        lg = log_rate(d, t)
        s += f"  log=({lg[0] * 1000:.3f}, {lg[1] * 1000:.3f})"
    print(s)
print("  chi2 quantiles: chi2(0.025, 6) =", st.chi2.ppf(0.025, 6), " chi2(0.975, 8) =", st.chi2.ppf(0.975, 8),
      " chi2(0.975, 2) =", st.chi2.ppf(0.975, 2), " one-sided chi2(0.95,2)/2 =", st.chi2.ppf(0.95, 2) / 2)
print("  wald half-width D=3:", Z * np.sqrt(3) / 1.5, " sqrt(3) =", np.sqrt(3))
print("  log scale D=3: 1.96/sqrt(3) =", Z / np.sqrt(3), "exp =", np.exp(Z / np.sqrt(3)))


def coverage(mu, kind):
    ks = np.arange(0, int(mu + 12 * np.sqrt(mu) + 30))
    pk = st.poisson.pmf(ks, mu)
    cov = 0.0
    for k, p in zip(ks, pk):
        if kind == "exact":
            lo, hi = garwood(k)
        elif kind == "wald":
            lo, hi = k - Z * np.sqrt(k), k + Z * np.sqrt(k)
        if lo <= mu <= hi:
            cov += p
    return cov


grid = np.round(np.arange(0.5, 30.01, 0.05), 2)
cw = np.array([coverage(m, "wald") for m in grid])
ce = np.array([coverage(m, "exact") for m in grid])
for m in (1, 2, 3, 5, 10, 20, 30):
    i = np.argmin(abs(grid - m))
    print(f"  coverage mu={m}: wald={cw[i]:.3f} exact={ce[i]:.3f}")
sel = (grid >= 1) & (grid <= 10)
print(f"  mu in [1,10]: wald min {cw[sel].min():.3f} at {grid[sel][cw[sel].argmin()]}, exact min {ce[sel].min():.3f}")
sel = (grid >= 10)
print(f"  mu in [10,30]: wald min {cw[sel].min():.3f} max {cw[sel].max():.3f}; exact min {ce[sel].min():.3f} max {ce[sel].max():.3f}")
print(f"  exact overall min {ce.min():.4f}")

# ------------------------------------------------------------------ 가-4 small-count paper box
H("가-4. rare-event paper box (3 events / 1500 PY vs 0 / 2300 PY)")
print("  exact conditional test p (poisson.test(c(3,0), c(1500,2300))):", binom_test_two_sided(3, 3, 1500 / 3800),
      " p0 =", 1500 / 3800, " p0^3 =", (1500 / 3800) ** 3)
print("  rule of three: 3/2300*1000 =", 3 / 2300 * 1000, " one-sided 95% upper:", st.chi2.ppf(0.95, 2) / 2 / 2300 * 1000)

# ------------------------------------------------------------------ 가-5 HHF table (SGLT2 vs DPP-4, ITT-like)
H("가-5. HHF rates, SGLT2i vs DPP-4i (ITT-like), by age")
HHF = {  # (events, person-years)
    ("S", "<65"): (58, 8076), ("S", ">=65"): (52, 2991),
    ("D", "<65"): (162, 16238), ("D", ">=65"): (285, 11764),
}
tot = {g: tuple(np.sum([HHF[(g, a)][i] for a in ("<65", ">=65")]) for i in range(2)) for g in "SD"}
print("  totals", tot, " PY per person: S", tot["S"][1] / 4812, " D", tot["D"][1] / 9655)
print("  share of PY aged >=65: S", HHF[("S", ">=65")][1] / tot["S"][1], " D", HHF[("D", ">=65")][1] / tot["D"][1])
for lab, sd, dd in (("overall", tot["S"], tot["D"]), ("<65", HHF[("S", "<65")], HHF[("D", "<65")]),
                    (">=65", HHF[("S", ">=65")], HHF[("D", ">=65")])):
    gS, gD = garwood(sd[0], sd[1]), garwood(dd[0], dd[1])
    c = compare_rates(sd[0], sd[1], dd[0], dd[1])
    print(f"  {lab:8s} S {sd[0]}/{sd[1]} = {sd[0] / sd[1] * 1000:.2f} ({gS[0] * 1000:.2f}-{gS[1] * 1000:.2f}) | "
          f"D {dd[0]}/{dd[1]} = {dd[0] / dd[1] * 1000:.2f} ({gD[0] * 1000:.2f}-{gD[1] * 1000:.2f}) | IRR {c['irr']:.4f} "
          f"({c['lo']:.3f}-{c['hi']:.3f}) SE {c['se']:.4f} z {c['z']:.3f} p {c['p']:.2e} | exact ({c['ex_lo']:.3f}-{c['ex_hi']:.3f}) p_ex {c['p_exact']:.2e} | "
          f"RD {c['rd'] * 1000:.3f} ({c['rd_lo'] * 1000:.3f} to {c['rd_hi'] * 1000:.3f}) se_rd {c['se_rd'] * 1000:.4f}")
c = compare_rates(tot["S"][0], tot["S"][1], tot["D"][0], tot["D"][1])
print("  overall: 1/110 =", 1 / 110, " 1/447 =", 1 / 447, " sum =", 1 / 110 + 1 / 447, " sqrt =", np.sqrt(1 / 110 + 1 / 447),
      " log IRR =", np.log(c["irr"]))
print("  RD se pieces: 110/11067^2*1e6 =", 110 / 11067 ** 2 * 1e6, " 447/28002^2*1e6 =", 447 / 28002 ** 2 * 1e6)
print("  proportions: S 110/4812 =", 110 / 4812, " D 447/9655 =", 447 / 9655, " ratio =", (110 / 4812) / (447 / 9655))
# Mantel-Haenszel rate ratio
num = den = 0
for a in ("<65", ">=65"):
    d1, t1 = HHF[("S", a)]; d0, t0 = HHF[("D", a)]; T = t1 + t0
    num += d1 * t0 / T; den += d0 * t1 / T
print("  MH IRR =", num / den)
# expected events for SGLT2 if they had DPP-4 age-specific rates (indirect standardization / SIR)
E = sum(HHF[("S", a)][1] * HHF[("D", a)][0] / HHF[("D", a)][1] for a in ("<65", ">=65"))
print("  SGLT2 expected with DPP-4 age-specific rates:", E, " SIR:", 110 / E)

# Poisson regression on grouped table
rows = [("S", "<65"), ("S", ">=65"), ("D", "<65"), ("D", ">=65")]
yv = np.array([HHF[r][0] for r in rows], float)
tv = np.array([HHF[r][1] for r in rows], float)
sg = np.array([1, 1, 0, 0.]); old = np.array([0, 1, 0, 1.])
f1 = poisson_glm(np.column_stack([np.ones(4), sg]), yv, np.log(tv))
f2 = poisson_glm(np.column_stack([np.ones(4), sg, old]), yv, np.log(tv))
print(f"  grouped Poisson, drug only: b1={f1['beta'][1]:.5f} IRR={np.exp(f1['beta'][1]):.4f} SE={f1['se'][1]:.5f}  "
      f"b0={f1['beta'][0]:.5f} exp(b0)*1000={np.exp(f1['beta'][0]) * 1000:.3f}")
b, s = f2["beta"], f2["se"]
print(f"  grouped Poisson, drug + age: IRR_drug={np.exp(b[1]):.4f} ({np.exp(b[1] - Z * s[1]):.3f}-{np.exp(b[1] + Z * s[1]):.3f})"
      f"  IRR_age={np.exp(b[2]):.3f} ({np.exp(b[2] - Z * s[2]):.3f}-{np.exp(b[2] + Z * s[2]):.3f})  resid dev={f2['dev']:.4f} on {f2['df']} df"
      f"  p={st.chi2.sf(f2['dev'], f2['df']):.3f}  b={b}  se={s}  z_drug={b[1] / s[1]:.3f} p_drug={2 * st.norm.sf(abs(b[1] / s[1])):.2e}")
print("  grouped fitted:", f2["mu"], " exp(b0)*1000", np.exp(b[0]) * 1000)

# ------------------------------------------------------------------ 가-6 observed/expected (tip)
H("가-6. observed vs expected (vaccine-safety style)")
O, Ex = 12, 5.3
print("  P(X>=12 | 5.3) =", st.poisson.sf(O - 1, Ex), " O/E =", O / Ex, " exact CI", [v / Ex for v in garwood(O)])

# ------------------------------------------------------------------ 나-1 six patients toy
H("나-1. six patients: offset")
toy = [("#1", 1, 0.5, 1, [0.3]), ("#2", 1, 2.0, 2, [0.6, 1.5]), ("#3", 1, 1.2, 0, []),
       ("#4", 0, 3.0, 4, [0.4, 1.1, 1.9, 2.6]), ("#5", 0, 0.8, 0, []), ("#6", 0, 2.5, 3, [0.7, 1.6, 2.2])]
dr = np.array([r[1] for r in toy], float); tt = np.array([r[2] for r in toy]); yy = np.array([r[3] for r in toy], float)
X6 = np.column_stack([np.ones(6), dr])
fo = poisson_glm(X6, yy, np.log(tt)); fn = poisson_glm(X6, yy)
print("  A: events", yy[dr == 1].sum(), "PY", tt[dr == 1].sum(), "rate", yy[dr == 1].sum() / tt[dr == 1].sum(), " mean count", yy[dr == 1].mean())
print("  B: events", yy[dr == 0].sum(), "PY", tt[dr == 0].sum(), "rate", yy[dr == 0].sum() / tt[dr == 0].sum(), " mean count", yy[dr == 0].mean())
print(f"  with offset: IRR={np.exp(fo['beta'][1]):.4f}  exp(b0)={np.exp(fo['beta'][0]):.4f}   without offset: ratio={np.exp(fn['beta'][1]):.4f}")
print("  per-person rates:", yy / tt, " mean of individual rates A,B:", (yy / tt)[dr == 1].mean(), (yy / tt)[dr == 0].mean())

# ------------------------------------------------------------------ 나-2 COPD cohort
H("나-2. COPD cohort (seed 20260929)")
c = cohort()
y, py, n = c["y"], c["py"], c["n"]
A = c["drug"] == 1
for lab, m in (("A", A), ("B", ~A), ("all", np.ones(n, bool))):
    print(f"  {lab}: n={m.sum()} events={y[m].sum():.0f} PY={py[m].sum():.1f} rate={y[m].sum() / py[m].sum():.4f} "
          f"meanFU={py[m].mean():.3f} medianFU={np.median(py[m]):.3f} age groups %={[round(100 * np.mean(c['age'][m] == k), 1) for k in range(3)]} "
          f"female%={100 * c['female'][m].mean():.1f} CCI mean={c['cci'][m].mean():.2f} sd={c['cci'][m].std(ddof=1):.2f} "
          f"any event%={100 * (y[m] > 0).mean():.1f} >=2 among any%={100 * (y[m] >= 2).sum() / (y[m] > 0).sum():.1f} "
          f"zeros={(y[m] == 0).sum()} max={y[m].max():.0f} mean count={y[m].mean():.3f} var count={y[m].var(ddof=1):.3f}")
print("  PY range", py.min(), py.max(), " days min", c["days"].min())
print("  count distribution all:", np.bincount(y.astype(int)))
cr = compare_rates(y[A].sum(), py[A].sum(), y[~A].sum(), py[~A].sum())
print(f"  crude rate ratio {cr['irr']:.4f} ({cr['lo']:.3f}-{cr['hi']:.3f}) [Poisson Wald]")
print("  top 10% of patients by count hold % of events:",
      100 * np.sort(y)[::-1][: n // 10].sum() / y.sum())

X = design(c)
off = np.log(py)
names = ["(Intercept)", "drugA", "agegrp65-74", "agegrp75+", "female", "cci"]
pf = poisson_glm(X, y, off)
p0 = null_poisson(y, off)
print("\n  --- Poisson (R glm output) ---")
for nm, b_, s_ in zip(names, pf["beta"], pf["se"]):
    zv = b_ / s_
    print(f"  {nm:12s} {b_:9.5f} {s_:9.5f} {zv:8.3f} {2 * st.norm.sf(abs(zv)):.3g}   IRR {np.exp(b_):.4f} ({np.exp(b_ - Z * s_):.3f}-{np.exp(b_ + Z * s_):.3f})")
print(f"  Null deviance {p0['dev']:.1f} on {n - 1} df;  Residual deviance {pf['dev']:.1f} on {pf['df']} df; AIC {pf['aic']:.1f}; iterations {pf['iters']}")
print(f"  LR (null - resid) = {p0['dev'] - pf['dev']:.2f}")
print(f"  loglik {pf['ll']:.3f}  Pearson chi2 {pf['pearson']:.2f}  phi = {pf['pearson'] / pf['df']:.4f}  dev/df = {pf['dev'] / pf['df']:.4f}")
print(f"  P(chi2_df > Pearson) = {st.chi2.sf(pf['pearson'], pf['df']):.2e}   P(chi2_df > dev) = {st.chi2.sf(pf['dev'], pf['df']):.2e}")
print(f"  exp(b0) = {np.exp(pf['beta'][0]):.4f} per PY;  per 100 PY {np.exp(pf['beta'][0]) * 100:.2f}")
b = pf["beta"]
print(f"  cci +1: {np.exp(b[5]):.4f}; +3: {np.exp(3 * b[5]):.4f} = {np.exp(b[5]) ** 3:.4f};  75+ vs 65-74: {np.exp(b[3] - b[2]):.4f}")
# example predicted rate
x_ex = np.array([1, 1, 0, 1, 0, 2.])
print(f"  predicted rate A, 75+, male, CCI 2: {np.exp(x_ex @ b):.4f} per PY; same patient on B: {np.exp(x_ex @ b - b[1]):.4f}")
print("  z for drugA:", b[1] / pf["se"][1])

phi = pf["pearson"] / pf["df"]
rob = np.sqrt(np.diag(sandwich_hc0(pf)))
print("\n  --- SE comparison for drugA ---")
nb = nb_glm(X, y, off)
bn, sn = nb["beta"], nb["se"]
res = {}
for lab, be, se in (("Poisson", b[1], pf["se"][1]), ("quasi-Poisson", b[1], pf["se"][1] * np.sqrt(phi)),
                    ("Poisson robust HC0", b[1], rob[1]), ("NB", bn[1], sn[1])):
    res[lab] = (np.exp(be), np.exp(be - Z * se), np.exp(be + Z * se), se)
    print(f"  {lab:20s} IRR {np.exp(be):.4f} ({np.exp(be - Z * se):.3f}-{np.exp(be + Z * se):.3f}) SE {se:.5f} z {be / se:.3f} p {2 * st.norm.sf(abs(be / se)):.4f} width_ratio {np.exp(2 * Z * se):.4f}")
print("  sqrt(phi) =", np.sqrt(phi), "  robust/model SE ratio (all coefs):", rob / pf["se"])
print("  quasi t-based p for drugA:", 2 * st.t.sf(abs(b[1] / (pf["se"][1] * np.sqrt(phi))), pf["df"]))

print("\n  --- NB (MASS::glm.nb output) ---")
for nm, b_, s_ in zip(names, bn, sn):
    zv = b_ / s_
    print(f"  {nm:12s} {b_:9.5f} {s_:9.5f} {zv:8.3f} {2 * st.norm.sf(abs(zv)):.3g}   IRR {np.exp(b_):.4f} ({np.exp(b_ - Z * s_):.3f}-{np.exp(b_ + Z * s_):.3f})")
nb_null = nb_null_deviance(y, off, nb["theta"])
print(f"  theta {nb['theta']:.4f} SE {nb['theta_se']:.4f} alpha {nb['alpha']:.4f}; alpha CI via theta: "
      f"{1 / (nb['theta'] + Z * nb['theta_se']):.3f}-{1 / (nb['theta'] - Z * nb['theta_se']):.3f}")
print(f"  2 x loglik {2 * nb['ll']:.4f}  AIC {nb['aic']:.1f}  resid dev {nb['dev']:.1f} on {nb['df']} df  null dev {nb_null:.1f} on {n - 1}")
LR = 2 * (nb["ll"] - pf["ll"])
print(f"  LR Poisson vs NB = {LR:.2f}; p (boundary, half chi2_1) = {0.5 * st.chi2.sf(LR, 1):.2e}; AIC Poisson {pf['aic']:.1f} vs NB {nb['aic']:.1f}")
mu_nb = nb["mu"]
pear_nb = np.sum((y - mu_nb) ** 2 / (mu_nb + nb["alpha"] * mu_nb ** 2))
print(f"  NB Pearson/df = {pear_nb / nb['df']:.3f}")

# crude NB and crude Poisson
Xc = design(c, ("drugA",))
pc = poisson_glm(Xc, y, off); nc = nb_glm(Xc, y, off)
for lab, f in (("crude Poisson", pc), ("crude NB", nc)):
    be, se = f["beta"][1], f["se"][1]
    print(f"  {lab}: IRR {np.exp(be):.4f} ({np.exp(be - Z * se):.3f}-{np.exp(be + Z * se):.3f}) p {2 * st.norm.sf(abs(be / se)):.3f}")
print("  crude NB alpha", nc["alpha"], " crude NB fitted rates A,B:", np.exp(nc["beta"][0] + nc["beta"][1]), np.exp(nc["beta"][0]))

# age group LR test (NB, theta re-estimated)
nb_noage = nb_glm(design(c, ("drugA", "female", "cci")), y, off)
LRa = 2 * (nb["ll"] - nb_noage["ll"])
print(f"  NB LR test for age group (2 df): {LRa:.2f}, p = {st.chi2.sf(LRa, 2):.2e}")

# observed vs expected frequencies
H("나-3. observed vs expected count frequencies")
K = 7
obs = [np.sum(y == k) for k in range(K)] + [np.sum(y >= K)]
ep = [st.poisson.pmf(k, pf["mu"]).sum() for k in range(K)] + [st.poisson.sf(K - 1, pf["mu"]).sum()]
en = [nb_pmf(k, mu_nb, nb["alpha"]).sum() for k in range(K)] + [n - sum(nb_pmf(k, mu_nb, nb["alpha"]).sum() for k in range(K))]
for k in range(K + 1):
    lab = f"{k}" if k < K else f">={K}"
    print(f"  {lab:>4s}: observed {obs[k]:5d} ({100 * obs[k] / n:.1f}%)  Poisson {ep[k]:8.1f}  NB {en[k]:8.1f}")
print("  Poisson expected zeros %:", 100 * ep[0] / n, " NB:", 100 * en[0] / n, " observed:", 100 * obs[0] / n)

# common errors
H("나-4. common errors")
fno = poisson_glm(X, y)
print(f"  no offset (Poisson): drug IRR {np.exp(fno['beta'][1]):.4f}")
Xl = np.column_stack([X, off])
fcov = poisson_glm(Xl, y)
print(f"  log(PY) as covariate: coef {fcov['beta'][-1]:.4f} (SE {fcov['se'][-1]:.4f}), drug IRR {np.exp(fcov['beta'][1]):.4f}")
fnbno = nb_glm(X, y)
print(f"  no offset (NB): drug IRR {np.exp(fnbno['beta'][1]):.4f}")
print("  mean follow-up A vs B (years):", py[A].mean(), py[~A].mean(), " ratio", py[A].mean() / py[~A].mean())
# IRR vs risk of >=1 event over 1 year (analytic, using NB fit)
muB = np.exp(bn[0])  # reference profile, 1 year
for mB in (0.6, 1.0):
    for al in (0, nb["alpha"]):
        irr = 0.80
        pB = 1 - (np.exp(-mB) if al == 0 else (1 + al * mB) ** (-1 / al))
        pA = 1 - (np.exp(-irr * mB) if al == 0 else (1 + al * irr * mB) ** (-1 / al))
        print(f"  1-yr, rate_B={mB}, alpha={al:.3f}, IRR 0.80: P(>=1) B={pB:.4f} A={pA:.4f} risk ratio={pA / pB:.4f}")
# logistic on 'any event' ignoring counts, for comparison (crude OR and proportions)
pa, pb = (y[A] > 0).mean(), (y[~A] > 0).mean()
print(f"  any event: A {pa:.4f}, B {pb:.4f}, ratio {pa / pb:.4f}, OR {(pa / (1 - pa)) / (pb / (1 - pb)):.4f}")

# ------------------------------------------------------------------ 나-5 modified Poisson (5장 data)
H("나-5. modified Poisson on 5장 2x2 data (A 48/400, B 102/600)")
yb = np.r_[np.ones(48), np.zeros(352), np.ones(102), np.zeros(498)]
xb = np.r_[np.ones(400), np.zeros(600)]
Xb = np.column_stack([np.ones(1000), xb])
fb = poisson_glm(Xb, yb)
rb = np.sqrt(np.diag(sandwich_hc0(fb)))
be = fb["beta"][1]
print(f"  RR = exp(b) = {np.exp(be):.4f}  (0.12/0.17 = {0.12 / 0.17:.4f})")
print(f"  model SE {fb['se'][1]:.5f} (sqrt(1/48+1/102) = {np.sqrt(1 / 48 + 1 / 102):.5f}) -> CI {np.exp(be - Z * fb['se'][1]):.3f}-{np.exp(be + Z * fb['se'][1]):.3f}")
print(f"  robust SE {rb[1]:.5f} (sqrt(1/48-1/400+1/102-1/600) = {np.sqrt(1 / 48 - 1 / 400 + 1 / 102 - 1 / 600):.5f}) -> CI {np.exp(be - Z * rb[1]):.3f}-{np.exp(be + Z * rb[1]):.3f}")
print(f"  p robust {2 * st.norm.sf(abs(be / rb[1])):.4f}, p model {2 * st.norm.sf(abs(be / fb['se'][1])):.4f}")
print(f"  Pearson/df for binary Poisson fit: {fb['pearson'] / fb['df']:.3f}")

# ------------------------------------------------------------------ 나-6 piecewise exponential = Cox (Breslow)
H("나-6. check: Poisson with one intercept per event-time interval reproduces Cox (Breslow ties)")
rng = np.random.default_rng(5)
m = 80
xs = rng.integers(0, 2, m).astype(float)
tt_ = np.ceil(rng.exponential(10 * np.exp(-0.6 * xs)))
cens = np.ceil(rng.uniform(2, 25, m))
time = np.minimum(tt_, cens); ev = (tt_ <= cens).astype(float)


def cox_breslow(time, ev, x):
    bb = 0.0
    for _ in range(50):
        U = I = 0.0
        for t in np.unique(time[ev == 1]):
            R = time >= t
            d = ((time == t) & (ev == 1))
            w = np.exp(bb * x[R])
            s0, s1, s2 = w.sum(), (w * x[R]).sum(), (w * x[R] ** 2).sum()
            U += x[d].sum() - d.sum() * s1 / s0
            I += d.sum() * (s2 / s0 - (s1 / s0) ** 2)
        bb += U / I
    return bb, 1 / np.sqrt(I)


bc, sc = cox_breslow(time, ev, xs)
ets = np.unique(time[ev == 1])
rows_ = []
for i in range(m):
    prev = 0.0
    for j, t in enumerate(ets):
        if time[i] <= prev:
            break
        stop = min(time[i], t)
        rows_.append((j, xs[i], stop - prev, 1.0 if (ev[i] == 1 and time[i] == t) else 0.0))
        prev = t
    # time after last event time contributes no information (no events); skip
rows_ = np.array(rows_)
J = len(ets)
Xp = np.column_stack([np.eye(J)[rows_[:, 0].astype(int)], rows_[:, 1]])
fp = poisson_glm(Xp, rows_[:, 3], np.log(rows_[:, 2]))
print(f"  Cox Breslow beta {bc:.6f} (SE {sc:.6f});  piecewise Poisson beta {fp['beta'][-1]:.6f} (SE {fp['se'][-1]:.6f});  events {ev.sum():.0f}, distinct times {J}")
# exact version: one record per subject per risk set (exposure 1), time-specific intercepts -> identical to Breslow
rr = []
for j, t in enumerate(ets):
    for i in range(m):
        if time[i] >= t:
            rr.append((j, xs[i], 1.0 if (ev[i] == 1 and time[i] == t) else 0.0))
rr = np.array(rr)
Xr = np.column_stack([np.eye(J)[rr[:, 0].astype(int)], rr[:, 1]])
fr = poisson_glm(Xr, rr[:, 2])
print(f"  risk-set Poisson (exposure 1 per risk set) beta {fr['beta'][-1]:.6f} (SE {fr['se'][-1]:.6f})  -> equals Cox Breslow")


# ------------------------------------------------------------------ R-style printouts (used in the HTML until 2026-09-30; kept for reference)
def fmt_p_group(ps, digits=3, eps=np.finfo(float).eps):
    out = [None] * len(ps)
    fix = [i for i, p in enumerate(ps) if p >= eps and np.floor(np.log10(p)) >= -3]
    exp_ = [i for i, p in enumerate(ps) if p >= eps and np.floor(np.log10(p)) < -3]
    if fix:
        dec = max(digits - 1 - int(np.floor(np.log10(ps[i]))) for i in fix)
        for i in fix:
            out[i] = f"{ps[i]:.{dec}f}"
    for i in exp_:
        out[i] = f"{ps[i]:.{digits - 1}e}"
    for i, p in enumerate(ps):
        if p < eps:
            out[i] = "< 2e-16"
    return out


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.1 else ""


def coef_table(names, beta, se):
    zs = beta / se
    ps = [2 * st.norm.sf(abs(z)) for z in zs]
    pf_ = fmt_p_group(ps)
    w = max(len(s) for s in names)
    lines = [" " * w + " Estimate Std. Error z value Pr(>|z|)    "]
    for nm, b_, s_, z_, p_, ps_ in zip(names, beta, se, zs, pf_, ps):
        lines.append(f"{nm:<{w}} {b_:8.5f} {s_:10.5f} {z_:7.3f} {p_:>8s} {stars(ps_):<3s}")
    return "\n".join(lines)


H("R printouts")
print("glm poisson:")
print(coef_table(names, pf["beta"], pf["se"]))
print(f"\n    Null deviance: {p0['dev']:.1f}  on {n - 1}  degrees of freedom\nResidual deviance: {pf['dev']:.1f}  on {pf['df']}  degrees of freedom\nAIC: {pf['aic']:.1f}\n\nNumber of Fisher Scoring iterations: {pf['iters']}")
print("\nexp(cbind(IRR = coef(fit), confint.default(fit))):")
for nm, b_, s_ in zip(names, pf["beta"], pf["se"]):
    print(f"{nm:<12s} {np.exp(b_):.7f} {np.exp(b_ - Z * s_):.7f} {np.exp(b_ + Z * s_):.7f}")
print("\nglm.nb:")
print(coef_table(names, nb["beta"], nb["se"]))
dp = max(2 - int(np.floor(np.log10(nb["theta_se"]))), 0)
print(f"\n(Dispersion parameter for Negative Binomial({nb['theta']:.4f}) family taken to be 1)")
print(f"\n    Null deviance: {nb_null:.1f}  on {n - 1}  degrees of freedom\nResidual deviance: {nb['dev']:.1f}  on {nb['df']}  degrees of freedom\nAIC: {nb['aic']:.1f}")
print(f"\n              Theta:  {nb['theta']:.{dp}f} \n          Std. Err.:  {nb['theta_se']:.{dp}f} \n\n 2 x log-likelihood:  {2 * nb['ll']:.{dp}f} ")
print("theta, se raw:", nb["theta"], nb["theta_se"], " dp", dp)
print("\nPearson dispersion: sum(residuals(fit, type = 'pearson')^2) / df.residual(fit) =", pf["pearson"], "/", pf["df"], "=", pf["pearson"] / pf["df"])
print("robust SEs (HC0):", dict(zip(names, np.round(rob, 5))))
print("quasi SEs:", dict(zip(names, np.round(pf["se"] * np.sqrt(phi), 5))))

H("IRR vs risk of >=1 event (fitted IRR, crude B rate, fitted alpha, 1 year)")
irr_nb = np.exp(nb["beta"][1]); rB = y[~A].sum() / py[~A].sum(); al = nb["alpha"]
pB = 1 - (1 + al * rB) ** (-1 / al); pA = 1 - (1 + al * irr_nb * rB) ** (-1 / al)
print(f"  IRR {irr_nb:.4f} rate_B {rB:.4f}/PY -> A {irr_nb * rB:.4f}; P(>=1) B {pB:.4f} A {pA:.4f} ratio {pA / pB:.4f}; Poisson-only ratio {(1 - np.exp(-irr_nb * rB)) / (1 - np.exp(-rB)):.4f}")
print(f"  events per 100 patient-years: B {100 * rB:.1f}, A {100 * irr_nb * rB:.1f}, difference {100 * (rB - irr_nb * rB):.1f}")


# ================================================================== additions (second pass)
H("추가-1. HHF numbers with more decimals (rounding check)")
for lab, sd, dd in (("overall", tot["S"], tot["D"]), ("<65", HHF[("S", "<65")], HHF[("D", "<65")]),
                    (">=65", HHF[("S", ">=65")], HHF[("D", ">=65")])):
    gS, gD = garwood(sd[0], sd[1]), garwood(dd[0], dd[1])
    c_ = compare_rates(sd[0], sd[1], dd[0], dd[1])
    print(f"  {lab}: S rate {sd[0] / sd[1] * 1000:.4f} CI {gS[0] * 1000:.4f}-{gS[1] * 1000:.4f}; D rate {dd[0] / dd[1] * 1000:.4f} CI {gD[0] * 1000:.4f}-{gD[1] * 1000:.4f}; "
          f"IRR {c_['irr']:.5f} ({c_['lo']:.5f}-{c_['hi']:.5f}); RD {c_['rd'] * 1000:.4f} ({c_['rd_lo'] * 1000:.4f} to {c_['rd_hi'] * 1000:.4f})")
print("  Garwood pieces overall S: chi2(.025, 220)/2 =", st.chi2.ppf(0.025, 220) / 2, " chi2(.975, 222)/2 =", st.chi2.ppf(0.975, 222) / 2)
print("  log IRR +- 1.96 SE:", np.log(0.6227) , -0.47377 - Z * 0.10643, -0.47377 + Z * 0.10643, np.exp(-0.47377 - Z * 0.10643), np.exp(-0.47377 + Z * 0.10643))
print("  NNT-like 1000/6.024 =", 1000 / 6.024)
print("  grouped Poisson drug+age: beta/se", f2["beta"], f2["se"], " exp CI", np.exp(f2["beta"][1] - Z * f2["se"][1]), np.exp(f2["beta"][1] + Z * f2["se"][1]))

H("추가-2. rare event: exact CI for rate ratio when one group has 0 events")
pl = st.beta.ppf(0.025, 3, 1)  # Clopper-Pearson lower for x = n = 3
print("  CP lower for 3/3:", pl, " = 0.025^(1/3) =", 0.025 ** (1 / 3))
print("  rate ratio lower bound:", pl / (1 - pl) * 2300 / 1500)
print("  exact CI rate 3/1500 per 1000:", [v * 1000 for v in garwood(3, 1500)], " 0/2300:", [v * 1000 for v in garwood(0, 2300)])
print("  chi2(.975, 2)/2 =", st.chi2.ppf(0.975, 2) / 2, " -> /2300*1000 =", st.chi2.ppf(0.975, 2) / 2 / 2300 * 1000)

H("추가-3. relative precision of a rate: 1/sqrt(D)")
for d in (3, 10, 110, 447, 1000):
    print(f"  D={d}: SE/rate = {1 / np.sqrt(d):.3f}; exact CI ratio hi/lo = {garwood(d)[1] / garwood(d)[0] if d > 0 else np.inf:.2f}")

H("추가-4. COPD: crude exact CIs, crude covariate IRRs (NB), gamma frailty interpretation")
for lab, m in (("A", A), ("B", ~A)):
    g = garwood(y[m].sum(), py[m].sum())
    print(f"  {lab}: {y[m].sum():.0f}/{py[m].sum():.4f} = {100 * y[m].sum() / py[m].sum():.3f} per 100 PY ({100 * g[0]:.3f}-{100 * g[1]:.3f}); mean FU days {c['days'][m].mean():.1f}")
for cols, lab in ((("agegrp",), "age"), (("female",), "female"), (("cci",), "cci")):
    if lab == "age":
        Xu = design(c, ("age2", "age3"))
    else:
        Xu = design(c, cols)
    fu_ = nb_glm(Xu, y, off)
    for j in range(1, Xu.shape[1]):
        be, se = fu_["beta"][j], fu_["se"][j]
        print(f"  crude NB {lab}[{j}]: IRR {np.exp(be):.4f} ({np.exp(be - Z * se):.4f}-{np.exp(be + Z * se):.4f})")
be, se = nc["beta"][1], nc["se"][1]
print(f"  crude NB drug: IRR {np.exp(be):.4f} ({np.exp(be - Z * se):.4f}-{np.exp(be + Z * se):.4f})")
for nm, b_, s_ in zip(names, bn, sn):
    print(f"  adj NB {nm}: IRR {np.exp(b_):.4f} ({np.exp(b_ - Z * s_):.4f}-{np.exp(b_ + Z * s_):.4f}) p {2 * st.norm.sf(abs(b_ / s_)):.4g}")
al = nb["alpha"]
gam = st.gamma(a=1 / al, scale=al)
print(f"  alpha {al:.4f}: SD of multipliers {np.sqrt(al):.3f}; P(u>2) {gam.sf(2):.4f}; P(u<0.5) {gam.cdf(0.5):.4f}; median {gam.median():.3f}")
print(f"  var at mu=1: {1 + al:.3f}; at mu=0.64: {0.64 + al * 0.64 ** 2:.3f} (ratio {1 + al * 0.64:.3f})")
print("  75+ vs 65-74 (NB):", np.exp(bn[3] - bn[2]), " cci x3 (NB):", np.exp(3 * bn[5]))

H("추가-5. standardized rates from NB (HEOR)")
eta0 = X @ bn - X[:, 1] * bn[1]            # everyone on drug B, 1 PY
rB_std = np.mean(np.exp(eta0)); rA_std = np.mean(np.exp(eta0 + bn[1]))
print(f"  standardized annual rate B {rB_std:.4f}, A {rA_std:.4f}, ratio {rA_std / rB_std:.4f}, diff per 100 PY {100 * (rB_std - rA_std):.2f}")
print(f"  per 1,000 patient-years avoided: {1000 * (rB_std - rA_std):.1f}")
# delta-method-free bootstrap is heavy; report only point estimates
print("  rate->prob 1yr: B", 1 - np.exp(-rB_std), " A", 1 - np.exp(-rA_std))

H("추가-6. Poisson vs Cox on the 7장 RCC cohort (seed 2)")
from lib_ch07 import cohort as rcc_cohort, cox_efron
C7 = rcc_cohort(2)
T7, S7, G7 = C7["time"], C7["status"], C7["drug"]
ce = cox_efron(T7, S7, G7)
print(f"  Cox (Efron) HR {ce['hr']:.4f} ({ce['lo']:.4f}-{ce['hi']:.4f})")


def split_fit(cuts):
    rows_ = []
    bnd = [0.0] + list(cuts) + [np.inf]
    for i in range(len(T7)):
        for j in range(len(bnd) - 1):
            a0, a1 = bnd[j], bnd[j + 1]
            if T7[i] <= a0:
                break
            ex = min(T7[i], a1) - a0
            ev_ = 1.0 if (S7[i] == 1 and a0 < T7[i] <= a1) else 0.0
            rows_.append((j, G7[i], ex, ev_))
    r_ = np.array(rows_)
    used = np.unique(r_[:, 0].astype(int))
    Xs = np.column_stack([np.eye(len(bnd) - 1)[r_[:, 0].astype(int)][:, used], r_[:, 1]])
    f_ = poisson_glm(Xs, r_[:, 3], np.log(r_[:, 2] / 12))
    b_, s_ = f_["beta"][-1], f_["se"][-1]
    return np.exp(b_), np.exp(b_ - Z * s_), np.exp(b_ + Z * s_), len(r_), [np.exp(v) * 100 for v in f_["beta"][:-1]]


for lab, cuts in (("one interval", []), ("yearly", [12, 24, 36, 48, 60, 72])):
    e_, l_, h_, nr, base = split_fit(cuts)
    print(f"  {lab}: IRR {e_:.4f} ({l_:.4f}-{h_:.4f}), rows {nr}, drug-B rates per 100 PY by interval {np.round(base, 1)}")
print("  deaths/PY by drug:", [(S7[G7 == g].sum(), T7[G7 == g].sum() / 12) for g in (1, 0)])

H("추가-7. verification of the hand-written fitters")
from scipy.optimize import minimize
from scipy.special import gammaln
negll = lambda b: -np.sum(y * (X @ b + off) - np.exp(X @ b + off))
o = minimize(negll, np.zeros(X.shape[1]), method="BFGS", options=dict(gtol=1e-9))
print("  Poisson IRLS vs BFGS max |diff|:", np.max(np.abs(o.x - pf["beta"])))


def nb_negll(par):
    b_, lth = par[:-1], par[-1]
    th = np.exp(lth); mu = np.exp(X @ b_ + off)
    return -np.sum(gammaln(th + y) - gammaln(th) - gammaln(y + 1) + th * np.log(th) + y * np.log(mu) - (th + y) * np.log(th + mu))


o2 = minimize(nb_negll, np.r_[pf["beta"], 0.0], method="BFGS", options=dict(gtol=1e-8))
print("  NB joint ML vs BFGS: beta diff", np.max(np.abs(o2.x[:-1] - bn)), " theta", np.exp(o2.x[-1]), "vs", nb["theta"])
# numerical robust SE check via score contributions
U = X * (y - pf["mu"])[:, None]
Bi = np.linalg.inv(X.T @ (X * pf["mu"][:, None]))
print("  sandwich check:", np.sqrt(np.diag(Bi @ (U.T @ U) @ Bi))[1], "vs", rob[1])
# large simulated cohort: do the fitters recover the true values?
from lib_ch12 import TRUE
cbig = cohort(seed=7, nA=40000, nB=56000)
Xb_ = design(cbig); ob_ = np.log(cbig["py"])
nbb = nb_glm(Xb_, cbig["y"], ob_)
print("  big sim NB: IRRs", np.round(np.exp(nbb["beta"]), 3), " alpha", round(nbb["alpha"], 3),
      " true:", np.round(np.exp([TRUE[k] for k in ("b0", "drugA", "age2", "age3", "female", "cci")]), 3), TRUE["alpha"])
pbb = poisson_glm(Xb_, cbig["y"], ob_)
print("  big sim Poisson IRR drug", np.exp(pbb["beta"][1]), " Pearson/df", pbb["pearson"] / pbb["df"])

# ================================================================== review of the 2026-09-30 beginner rewrite
H("추가-8. numbers introduced by the 2026-09-30 rewrite (easy boxes, 숫자로 따라가기, gamma-frailty reading)")
print("  easy box: 20/100 at risk in 1 yr = 20% risk; 30 admissions / 100 PY = 30 per 100 PY; hospital A 6/3 months =",
      6 / 3, "per month, hospital B 12/12 =", 12 / 12)
DS, TS, DD, TD = 110, 11067, 447, 28002
v1, v0 = DS / TS ** 2 * 1e6, DD / TD ** 2 * 1e6
rd = (DS / TS - DD / TD) * 1000
print(f"  rate difference per 1,000 PY {rd:.4f}; var pieces {v1:.3f} + {v0:.3f}; SE {np.sqrt(v1 + v0):.4f}; "
      f"95% CI {rd - Z * np.sqrt(v1 + v0):.4f} to {rd + Z * np.sqrt(v1 + v0):.4f}")
print(f"  Poisson crude COPD IRR {(554 / 940.0465) / (998 / 1563.6112):.4f}, SE sqrt(1/554+1/998) = {np.sqrt(1 / 554 + 1 / 998):.5f}, "
      f"CI {np.exp(np.log((554 / 940.0465) / (998 / 1563.6112)) - Z * np.sqrt(1 / 554 + 1 / 998)):.4f}-"
      f"{np.exp(np.log((554 / 940.0465) / (998 / 1563.6112)) + Z * np.sqrt(1 / 554 + 1 / 998)):.4f}")
print(f"  Poisson adj: exp(0.5270) = {np.exp(0.5270):.3f} (75+), exp(0.1281) = {np.exp(0.1281):.4f}, ^3 = {np.exp(3 * 0.1281):.3f}; "
      f"NB CCI 1.12^3 -> {np.exp(3 * bn[5]):.3f}")
print(f"  quasi-Poisson SE for drug: {pf['se'][1]:.5f} x sqrt({pf['pearson'] / pf['df']:.4f}) = {pf['se'][1] * np.sqrt(pf['pearson'] / pf['df']):.5f}")
gam8 = st.gamma(a=1 / nb["alpha"], scale=nb["alpha"])
print(f"  gamma frailty (mean 1, var alpha={nb['alpha']:.4f}): P(multiplier >= 2) = {gam8.sf(2):.4f} (~13%), "
      f"P(multiplier <= 0.5) = {gam8.cdf(0.5):.4f} (~36%), SD {gam8.std():.3f}; theta = 1/alpha = {1 / nb['alpha']:.3f}")
