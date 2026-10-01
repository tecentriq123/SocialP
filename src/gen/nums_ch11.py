"""Every number quoted in content/ch11.html (Cox proportional-hazards model) is printed by this script.
run: python3 gen/nums_ch11.py            (about a minute; the bootstrap is the slow part)

Validation of the Cox implementation (lib_ch11) is in section Z at the end:
  - one binary covariate: estimates equal lib_ch07.cox_efron; score test vs log-rank (equal without ties)
  - analytic score / information vs numerical derivatives
  - Breslow vs Efron vs exact ties
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.stats as st
from lib_ch07 import (km, surv_at, quantile_ci, logrank, cox_efron, small_arrays, scenario, rmst, n_risk, Z)
from lib_ch11 import (coxph, cox_loglik, cox_exact_1d, concordance, basehaz, predict_surv, adjusted_curves,
                      schoenfeld, zph, martingale, survsplit, logistic, smd, weighted_km, step_at, lowess,
                      num_grad_check, rcc_cohort, design, CUT_NCC)

np.set_printoptions(suppress=True, linewidth=140)


def hrs(f, i=0, nd=2):
    return f"{f['hr'][i]:.{nd}f} ({f['lo'][i]:.{nd}f}-{f['hi'][i]:.{nd}f}) p={f['p'][i]:.4f}"


def ptxt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


print("=" * 72, "\nA. hazard basics")
lam_B = 110 / 284.9   # drug B deaths / person-years (chapter 7)
print(f"  drug B crude rate {lam_B:.4f}/PY = {lam_B / 12:.5f}/month; exp(-rate*12) = {np.exp(-lam_B):.3f} vs KM S_B(12)=0.656")
lam = 0.03
print(f"  constant hazard 0.03/month: S(12)={np.exp(-0.36):.3f}  S(24)={np.exp(-0.72):.3f}  median={np.log(2) / lam:.1f}")
print(f"  toy: 100 alive at start of month 12, 3 die -> 3/100 = 0.03 per month; per year ~ 0.36 (hazard can exceed 1: 0.03/day = {0.03 * 365:.1f}/year)")
MED = 20.0
shapes = {"decreasing k=0.6": 0.6, "constant k=1": 1.0, "increasing k=1.6": 1.6}
for nm, k in shapes.items():
    b = MED / np.log(2) ** (1 / k)
    h = lambda t: (k / b) * (t / b) ** (k - 1)
    S = lambda t: np.exp(-(t / b) ** k)
    print(f"  {nm:18s} scale={b:.2f}  h(1)={h(1):.4f} h(6)={h(6):.4f} h(12)={h(12):.4f} h(24)={h(24):.4f} h(48)={h(48):.4f}"
          f"  S(6)={S(6):.3f} S(12)={S(12):.3f} S(20)={S(20):.3f} S(48)={S(48):.3f}")

print("=" * 72, "\nB. partial likelihood by hand (chapter-7 12 patients; x = 1 drug A)")
t, s, g = small_arrays()
evt = np.unique(t[s == 1])
fe = coxph(t, s, g[:, None], ties="efron")
fb = coxph(t, s, g[:, None], ties="breslow")
fx = cox_exact_1d(t, s, g)
be = fe["coef"][0]
print(f"  Efron beta={be:.4f} HR={np.exp(be):.4f} se={fe['se'][0]:.4f} CI {fe['lo'][0]:.3f}-{fe['hi'][0]:.3f} p={fe['p'][0]:.4f}")
print(f"  Breslow beta={fb['coef'][0]:.4f} HR={fb['hr'][0]:.4f} se={fb['se'][0]:.4f}  CI {fb['lo'][0]:.3f}-{fb['hi'][0]:.3f}")
print(f"  exact   beta={fx['coef']:.4f} HR={fx['hr']:.4f} se={fx['se']:.4f}  CI {fx['lo']:.3f}-{fx['hi']:.3f}")
print(f"  tests (Efron): LR={fe['lr']:.3f} p={fe['p_lr']:.4f}  Wald={fe['wald']:.3f} p={fe['p_wald']:.4f}  score={fe['score']:.3f} p={fe['p_score']:.4f}")
print(f"  tests (Breslow): score={fb['score']:.3f}  ; log-rank (ch07) = {logrank(t, s, g, 1)['chi2']:.3f}")
print(f"  loglik(0)={fe['loglik0']:.4f}  loglik(bhat)={fe['loglik']:.4f}")
e = np.exp(be)
rows = []
for tk in evt:
    R = t >= tk
    D = (t == tk) & (s == 1)
    nA, nB = int((R & (g == 1)).sum()), int((R & (g == 0)).sum())
    dA, dB = int((D & (g == 1)).sum()), int((D & (g == 0)).sum())
    rows.append((tk, nA, nB, dA, dB))
print("  t   nA nB  dA dB   term(beta=0)          term(beta_hat, Efron)")
logL0 = logLh = 0.0
for tk, nA, nB, dA, dB in rows:
    d = dA + dB
    if d == 1:
        num0, den0 = 1.0, nA + nB
        term0 = 1 / (nA + nB)
        termh = (e if dA else 1.0) / (nA * e + nB)
    else:
        # Efron: product over l of 1/(a0 - l/d * s0D)
        s0 = nA * e + nB; s0D = dA * e + dB
        termh = (e ** dA) / np.prod([s0 - l / d * s0D for l in range(d)])
        term0 = 1 / np.prod([(nA + nB) - l / d * d for l in range(d)])
    logL0 += np.log(term0); logLh += np.log(termh)
    print(f"  {tk:3.0f}  {nA} {nB}   {dA}  {dB}   {term0:.4f} (1/{1 / term0:.0f})      {termh:.4f}")
print(f"  sum log terms beta=0: {logL0:.4f}  beta_hat: {logLh:.4f}  (compare loglik0/loglik above)")
# tie at 12 by the three methods, beta = 0 and beta_hat
nA, nB = 6, 3
for bb, lab in ((0.0, "beta=0"), (be, "beta_hat")):
    E = np.exp(bb)
    bres = E / (nA * E + nB) ** 2
    efr = E / ((nA * E + nB) * ((nA * E + nB) - 0.5 * (E + 1)))
    exa = E / (15 * E ** 2 + 18 * E + 3)
    print(f"  tie at t=12 ({lab}): Breslow {bres:.5f} (1/{1 / bres:.1f})  Efron {efr:.5f} (1/{1 / efr:.1f})  exact {exa:.5f} (1/{1 / exa:.1f})")
# first term at bhat
print(f"  t=3 term at bhat: 1/(6*{e:.4f}+6) = {1 / (6 * e + 6):.4f}; at t=28: e/(3e+1) = {e / (3 * e + 1):.4f}")
# the curve for the figure
bgrid = np.linspace(-3.5, 1.0, 10)
print("  loglik over beta:", [(round(b, 2), round(cox_loglik([b], t, s, (g - g.mean())[:, None])[0], 3)) for b in bgrid])
print(f"  LR = 2*({fe['loglik']:.4f} - ({fe['loglik0']:.4f})) = {fe['lr']:.3f}")
# monotone transformation invariance
f_log = coxph(np.log(t), s, g[:, None])
print(f"  time -> log(time): HR {f_log['hr'][0]:.4f} (same)")

print("=" * 72, "\nC. RCC cohort (300 patients, chapter 7) with extra covariates")
D = rcc_cohort()
T, S_, G, IM = D["time"], D["status"], D["drug"], D["imdc"]
X, nm = design(D)
labs = ["favorable", "intermediate", "poor"]
for arm, lab in ((1, "A"), (0, "B")):
    m = G == arm
    print(f"  drug {lab}: n={m.sum()} age {D['age'][m].mean():.1f} +- {D['age'][m].std(ddof=1):.1f}  male {D['male'][m].sum()} ({D['male'][m].mean() * 100:.1f}%)"
          f"  neph {D['neph'][m].sum()} ({D['neph'][m].mean() * 100:.1f}%)  ncc {D['ncc'][m].sum()} ({D['ncc'][m].mean() * 100:.1f}%)"
          f"  IMDC {[int((IM[m] == k).sum()) for k in range(3)]}  deaths {S_[m].sum()}")
for k in ("age", "male", "neph", "ncc"):
    print(f"   SMD {k}: {smd(D[k], G):+.3f}")
for k in range(3):
    print(f"   SMD imdc={labs[k]}: {smd((IM == k).astype(float), G):+.3f}")
print("  overall: age", D["age"].mean().round(1), "median", np.median(D["age"]), " n age>=65", (D["age"] >= 65).sum(),
      " ncc", D["ncc"].sum(), "deaths in ncc", S_[D["ncc"] == 1].sum())
ut, cnt = np.unique(T[S_ == 1], return_counts=True)
print(f"  distinct death times {len(ut)} for {S_.sum()} deaths; times with ties {int((cnt > 1).sum())}, max tie {cnt.max()}")

print("\n  C1. univariable drug model (R summary.coxph layout)")
f1 = coxph(T, S_, X[:, :1], ties="efron")
c1 = concordance(T, S_, X[:, 0] * f1["coef"][0])
print(f"  n={f1['n']} events={f1['events']}")
print(f"  coef={f1['coef'][0]:.4f} exp(coef)={f1['hr'][0]:.4f} se={f1['se'][0]:.4f} z={f1['z'][0]:.3f} p={f1['p'][0]:.5f}")
print(f"  exp(-coef)={1 / f1['hr'][0]:.4f}  lower={f1['lo'][0]:.4f} upper={f1['hi'][0]:.4f}")
print(f"  concordance={c1['C']:.4f} se={c1['se']:.4f} pairs={c1['pairs']} conc={c1['concordant']} disc={c1['discordant']} tied={c1['tied']}")
print(f"  LR={f1['lr']:.3f} p={f1['p_lr']:.4f}  Wald={f1['wald']:.3f} p={f1['p_wald']:.4f}  score={f1['score']:.3f} p={f1['p_score']:.4f}")
print(f"  z^2 = {f1['z'][0] ** 2:.3f};  log-rank (survdiff) = {logrank(T, S_, G, 1)['chi2']:.3f}")
print(f"  loglik0={f1['loglik0']:.3f} loglik={f1['loglik']:.3f}")
print(f"  CI by hand: exp({f1['coef'][0]:.4f} +- 1.96*{f1['se'][0]:.4f}) = {np.exp(f1['coef'][0] - 1.96 * f1['se'][0]):.4f}-{np.exp(f1['coef'][0] + 1.96 * f1['se'][0]):.4f}")
f1b = coxph(T, S_, X[:, :1], ties="breslow")
print(f"  Breslow: coef={f1b['coef'][0]:.5f} HR={f1b['hr'][0]:.5f} se={f1b['se'][0]:.5f} score={f1b['score']:.3f}")
fx1 = cox_exact_1d(T, S_, X[:, 0])
print(f"  exact:   coef={fx1['coef']:.5f} HR={fx1['hr']:.5f} se={fx1['se']:.5f}")
# heavy ties: whole months
Tm = np.ceil(T)
ut2, cnt2 = np.unique(Tm[S_ == 1], return_counts=True)
print(f"  times rounded up to whole months: {len(ut2)} distinct death times, max tie {cnt2.max()}")
for ties in ("breslow", "efron"):
    ff = coxph(Tm, S_, X[:, :1], ties=ties)
    print(f"   monthly {ties}: coef={ff['coef'][0]:.4f} HR={ff['hr'][0]:.4f} se={ff['se'][0]:.4f} CI {ff['lo'][0]:.3f}-{ff['hi'][0]:.3f}")
ffx = cox_exact_1d(Tm, S_, X[:, 0])
print(f"   monthly exact:  coef={ffx['coef']:.4f} HR={ffx['hr']:.4f} se={ffx['se']:.4f} CI {ffx['lo']:.3f}-{ffx['hi']:.3f}")
# yearly (very heavy ties) to show the Breslow attenuation
Ty = np.ceil(T / 6) * 6
for ties in ("breslow", "efron"):
    ff = coxph(Ty, S_, X[:, :1], ties=ties)
    print(f"   6-monthly {ties}: HR={ff['hr'][0]:.5f} se={ff['se'][0]:.4f}")
ffx = cox_exact_1d(Ty, S_, X[:, 0])
print(f"   6-monthly exact: HR={ffx['hr']:.5f} se={ffx['se']:.4f}")

print("\n  C2. crude (univariable) and adjusted (multivariable) models")
fm = coxph(T, S_, X, names=nm)
uni = {}
for i, k in enumerate(nm):
    cols = [3, 4] if k in ("imdc1", "imdc2") else [i]
    fu = coxph(T, S_, X[:, cols])
    j = cols.index(i)
    uni[k] = (fu["hr"][j], fu["lo"][j], fu["hi"][j], fu["p"][j])
    print(f"  {k:6s} crude {fu['hr'][j]:.2f} ({fu['lo'][j]:.2f}-{fu['hi'][j]:.2f}) p={ptxt(fu['p'][j])}   "
          f"adjusted {fm['hr'][i]:.2f} ({fm['lo'][i]:.2f}-{fm['hi'][i]:.2f}) p={ptxt(fm['p'][i])}  coef={fm['coef'][i]:.4f} se={fm['se'][i]:.4f}")
# IMDC overall tests
fnoI = coxph(T, S_, X[:, [0, 1, 2, 5, 6]])
lrI = 2 * (fm["loglik"] - fnoI["loglik"])
bI = fm["coef"][3:5]; VI = fm["var"][3:5, 3:5]
wI = float(bI @ np.linalg.solve(VI, bI))
print(f"  IMDC 2-df LR={lrI:.2f} p={st.chi2.sf(lrI, 2):.2e}; Wald={wI:.2f} p={st.chi2.sf(wI, 2):.2e}")
fuI = coxph(T, S_, X[:, [3, 4]])
print(f"  IMDC crude 2-df LR={fuI['lr']:.2f} p={fuI['p_lr']:.2e}")
print(f"  poor vs intermediate (adjusted): exp({fm['coef'][4]:.4f}-{fm['coef'][3]:.4f}) = {np.exp(fm['coef'][4] - fm['coef'][3]):.3f}")
v = fm["var"]
dpi = fm["coef"][4] - fm["coef"][3]; sed = np.sqrt(v[3, 3] + v[4, 4] - 2 * v[3, 4])
print(f"     CI {np.exp(dpi - Z * sed):.2f}-{np.exp(dpi + Z * sed):.2f}")
ba = fm["coef"][1] / 10
print(f"  age per year: coef={ba:.5f} HR={np.exp(ba):.4f}; per 10 years HR={np.exp(10 * ba):.4f} = {np.exp(ba):.4f}^10 = {np.exp(ba) ** 10:.4f}; per 5 years {np.exp(5 * ba):.3f}")
print(f"  age per year CI {np.exp(ba - Z * fm['se'][1] / 10):.4f}-{np.exp(ba + Z * fm['se'][1] / 10):.4f}")
print(f"  75 vs 55 years (20 y): {np.exp(20 * ba):.3f}")
print(f"  drug exp(-coef) = {1 / fm['hr'][0]:.3f} (B vs A), CI {1 / fm['hi'][0]:.2f}-{1 / fm['lo'][0]:.2f}")
print(f"  model: LR={fm['lr']:.2f} df={fm['df']} p={fm['p_lr']:.2e}; Wald={fm['wald']:.2f}; score={fm['score']:.2f}")
cm = concordance(T, S_, (X - fm["means"]) @ fm["coef"])
print(f"  concordance multivariable {cm['C']:.4f} (se {cm['se']:.4f})")
print(f"  EPV: {S_.sum()} events / {X.shape[1]} parameters = {S_.sum() / X.shape[1]:.1f}")
fI = coxph(T, S_, X[:, [0, 3, 4]])
print(f"  drug adjusted for IMDC only: {hrs(fI)}")
fsI = coxph(T, S_, X[:, :1], strata=IM)
print(f"  drug, stratified by IMDC (ch07): {hrs(fsI)}")
print(f"  multivariable drug: {hrs(fm)}  coef {fm['coef'][0]:.4f} se {fm['se'][0]:.4f}")
print(f"  % change crude -> adjusted on log scale: {(fm['coef'][0] - f1['coef'][0]) / f1['coef'][0] * 100:.1f}%")
fmB = coxph(T, S_, X, ties="breslow")
print(f"  multivariable Breslow drug HR {fmB['hr'][0]:.4f} vs Efron {fm['hr'][0]:.4f}")
print("  gradient check (max abs error score, info):", num_grad_check(fm))
# models dropping one covariate at a time (change in drug HR)
for i in range(1, len(nm)):
    cols = [c for c in range(len(nm)) if c != i and not (nm[i] in ('imdc1', 'imdc2') and nm[c] in ('imdc1', 'imdc2'))]
    ff = coxph(T, S_, X[:, cols])
    print(f"   without {nm[i]:6s}: drug HR {ff['hr'][0]:.3f}")

print("\n  C3. baseline hazard, predicted and adjusted survival")
grid = [6, 12, 24, 36, 48]
fa = km(T[G == 1], S_[G == 1]); fb_ = km(T[G == 0], S_[G == 0])
for x in grid:
    print(f"   KM t={x}: A {surv_at(fa, x)[0]:.3f} B {surv_at(fb_, x)[0]:.3f} diff {surv_at(fa, x)[0] - surv_at(fb_, x)[0]:+.3f}")
adj = adjusted_curves(fm, X, 0, [1, 0], grid)
for i, x in enumerate(grid):
    print(f"   adjusted t={x}: A {adj[1][i]:.3f} B {adj[0][i]:.3f} diff {adj[1][i] - adj[0][i]:+.3f}")
# the mean-covariate curve
xbar = X.mean(axis=0)
for dv in (1, 0):
    xm = xbar.copy(); xm[0] = dv
    print(f"   'mean covariate' curve drug={dv}: S(24)={predict_surv(fm, xm, [24])[0, 0]:.3f}")
# example patient predictions
pat = np.array([[1, 7.0, 1, 1, 0, 1, 0], [0, 7.0, 1, 1, 0, 1, 0]])
pr = predict_surv(fm, pat, [12, 24])
print(f"   patient (70y male intermediate nephrectomy clear-cell) A: S12 {pr[0, 0]:.3f} S24 {pr[0, 1]:.3f}; B: S12 {pr[1, 0]:.3f} S24 {pr[1, 1]:.3f}"
      f"  check S_A = S_B^HR: {pr[1, 1] ** fm['hr'][0]:.3f}  (unrounded S_B24 {pr[1, 1]:.4f}, HR {fm['hr'][0]:.4f})")
bh = basehaz(fm)[0]
print(f"   H0 at 24 (centred covariates) {np.interp(24, bh[0], bh[1]):.4f}")
# RMST from adjusted curves (fine grid)
tg = np.linspace(0, 48, 961)
adjg = adjusted_curves(fm, X, 0, [1, 0], tg)
dt = tg[1] - tg[0]
rA = np.sum(adjg[1][:-1]) * dt; rB = np.sum(adjg[0][:-1]) * dt
print(f"   adjusted RMST48: A {rA:.2f} B {rB:.2f} diff {rA - rB:.2f}  (unadjusted diff 5.4 in ch07)")

print("\n  C4. bootstrap CI for the adjusted 24-month survival difference and RMST48 difference (200 resamples)")
rng = np.random.default_rng(11)
bd24, bdr = [], []
for bI_ in range(200):
    ii = rng.integers(0, len(T), len(T))
    fbb = coxph(T[ii], S_[ii], X[ii])
    a2 = adjusted_curves(fbb, X[ii], 0, [1, 0], np.concatenate([[24.0], tg]))
    bd24.append(a2[1][0] - a2[0][0])
    bdr.append((np.sum(a2[1][1:-1]) - np.sum(a2[0][1:-1])) * dt)
bd24 = np.array(bd24); bdr = np.array(bdr)
d24 = adj[1][2] - adj[0][2]
print(f"   adjusted diff at 24 mo {d24:.3f}  percentile CI {np.percentile(bd24, 2.5):.3f} to {np.percentile(bd24, 97.5):.3f}")
print(f"   adjusted RMST48 diff {rA - rB:.2f}  percentile CI {np.percentile(bdr, 2.5):.2f} to {np.percentile(bdr, 97.5):.2f}")

print("=" * 72, "\nD. HR is not a risk ratio / median ratio / frailty")
HRc = f1["hr"][0]
print(f"  crude HR = {HRc:.4f}")
for x in (6, 12, 24, 36, 48, 60):
    SB = surv_at(fb_, x)[0]
    SA = SB ** HRc
    rr = (1 - SA) / (1 - SB)
    print(f"   t={x}: S_B={SB:.3f} S_A=S_B^HR={SA:.3f}  risk B {1 - SB:.3f} A {1 - SA:.3f}  RR={rr:.3f}  RD={(1 - SA) - (1 - SB):+.3f}  risk reduction {(1 - rr) * 100:.1f}%")
# median under PH with drug-B KM baseline
target = 0.5 ** (1 / HRc)
evB = [r for r in fb_ if r["d"] > 0]
medA_ph = next(r["t"] for r in evB if r["S"] <= target)
print(f"  under PH: S_A(t)=0.5 <=> S_B(t)={target:.4f}; first t with S_B <= that: {medA_ph} ; median B 19.3 -> ratio {medA_ph / 19.3:.2f}; 1/HR={1 / HRc:.2f}; observed medians 32.4/19.3={32.4 / 19.3:.2f}")
for HRx in (0.5, 0.73):
    for k in (0.6, 1.0, 1.6):
        print(f"   Weibull k={k}: HR={HRx} -> median ratio = HR^(-1/k) = {HRx ** (-1 / k):.2f}")
# rare vs common: RR ~ HR when risk small
for SB in (0.95, 0.90, 0.50, 0.20):
    SA = SB ** 0.5
    print(f"   HR 0.5, baseline risk {1 - SB:.2f}: risk A {1 - SA:.3f} RR {(1 - SA) / (1 - SB):.3f}")
# gamma frailty: marginal HR drifts toward 1
th = 1.0
for H0 in (0.0, 0.5, 1.0, 2.0, 3.0):
    hr_m = 0.6 * (1 + th * H0) / (1 + th * 0.6 * H0)
    S0m = (1 + th * H0) ** (-1 / th)
    print(f"   frailty theta=1, individual HR 0.6: control marginal S={S0m:.3f} (H0={H0}) -> population HR {hr_m:.3f}")
# misread sentence numbers (adjusted)
print(f"  adjusted HR {fm['hr'][0]:.2f}: '1-HR' = {(1 - fm['hr'][0]) * 100:.0f}%")

print("=" * 72, "\nE. proportional hazards checks (multivariable model)")
terms = {"drug": [0], "age": [1], "sex": [2], "imdc": [3, 4], "nephrectomy": [5], "histology": [6]}
zr, gt, sc = zph(fm, terms, "km")
for k, (chi, df, p) in zr.items():
    print(f"   zph(km) {k:12s} chisq={chi:.3f} df={df} p={p:.4f}")
zri, _, _ = zph(fm, terms, "identity")
print("   zph(identity) histology", np.round(zri["histology"], 4), "GLOBAL", np.round(zri["GLOBAL"], 4), "drug", np.round(zri["drug"], 4))
zrl, _, _ = zph(fm, terms, "log")
print("   zph(log) histology", np.round(zrl["histology"], 4), "GLOBAL", np.round(zrl["GLOBAL"], 4))
# univariable drug zph
z1, _, _ = zph(f1, {"drug": [0]}, "km")
print("   univariable drug zph", np.round(z1["drug"], 4))
# scaled Schoenfeld: slope check / correlation with time
scl = sc["scaled"]; tm = sc["time"]
for j, k in ((0, "drug"), (6, "histology")):
    r = np.corrcoef(gt, scl[:, j])[0, 1]
    early = scl[tm < CUT_NCC, j].mean(); late = scl[tm >= CUT_NCC, j].mean()
    print(f"   scaled Schoenfeld {k}: corr with 1-KM {r:.3f}; mean early(<8) {early:.3f} late {late:.3f}  (exp: {np.exp(early):.2f}, {np.exp(late):.2f})  coef {fm['coef'][j]:.3f}")
# log-minus-log by drug at some times
for x in (3, 6, 12, 24, 36, 48):
    SA_ = surv_at(fa, x)[0]; SB_ = surv_at(fb_, x)[0]
    print(f"   LML t={x}: A {np.log(-np.log(SA_)):.3f} B {np.log(-np.log(SB_)):.3f} diff {np.log(-np.log(SA_)) - np.log(-np.log(SB_)):+.3f}")
# LML by histology
fn1 = km(T[D["ncc"] == 1], S_[D["ncc"] == 1]); fn0 = km(T[D["ncc"] == 0], S_[D["ncc"] == 0])
for x in (3, 6, 12, 24, 36):
    a_ = surv_at(fn1, x)[0]; b_ = surv_at(fn0, x)[0]
    print(f"   LML histology t={x}: ncc {np.log(-np.log(a_)):.3f} ccRCC {np.log(-np.log(b_)):.3f} diff {np.log(-np.log(a_)) - np.log(-np.log(b_)):+.3f}  S {a_:.3f} {b_:.3f}")
print("   ncc n", int(D['ncc'].sum()), "deaths", int(S_[D['ncc'] == 1].sum()), " deaths <12 mo in ncc", int(((T < 12) & (S_ == 1) & (D['ncc'] == 1)).sum()))

print("\n  E2. remedies for the histology PH violation")
# (a) stratify by histology
fsH = coxph(T, S_, X[:, :6], strata=D["ncc"])
for i, k in enumerate(nm[:6]):
    print(f"   strata(histology): {k:6s} {fsH['hr'][i]:.2f} ({fsH['lo'][i]:.2f}-{fsH['hi'][i]:.2f}) p={ptxt(fsH['p'][i])}")
zs, _, _ = zph(fsH, {"drug": [0], "age": [1], "sex": [2], "imdc": [3, 4], "nephrectomy": [5]}, "km")
print("   zph after stratification GLOBAL", np.round(zs["GLOBAL"], 4))
# (b) step function: split at 12 months (prespecified: first year vs later)
for cut in (6, 8, 12):
    ids, a, b, ev, ep = survsplit(T, S_, [cut])
    Xs = X[ids]
    Xd = np.column_stack([Xs[:, :6], Xs[:, 6] * (ep == 0), Xs[:, 6] * (ep == 1)])
    fs = coxph(b, ev, Xd, entry=a)
    # test of equal effects: LR vs model with single ncc
    fs0 = coxph(b, ev, Xs, entry=a)
    lrt = 2 * (fs["loglik"] - fs0["loglik"])
    dd = fs["coef"][6] - fs["coef"][7]; sd = np.sqrt(fs["var"][6, 6] + fs["var"][7, 7] - 2 * fs["var"][6, 7])
    print(f"   split at {cut}: rows {len(ids)}; ncc 0-{cut}: {fs['hr'][6]:.2f} ({fs['lo'][6]:.2f}-{fs['hi'][6]:.2f}) p={ptxt(fs['p'][6])};"
          f" >{cut}: {fs['hr'][7]:.2f} ({fs['lo'][7]:.2f}-{fs['hi'][7]:.2f}) p={ptxt(fs['p'][7])}; LR diff {lrt:.2f} p={st.chi2.sf(lrt, 1):.4f}; Wald diff z={dd / sd:.2f} p={2 * st.norm.sf(abs(dd / sd)):.4f}; drug {fs['hr'][0]:.3f}; split-no-tvc check drug {fs0['hr'][0]:.4f} ncc {fs0['hr'][6]:.4f}")
    if cut == 12:
        nd12 = int(((T <= 12) & (S_ == 1) & (D["ncc"] == 1)).sum()); ndl = int(((T > 12) & (S_ == 1) & (D["ncc"] == 1)).sum())
        print(f"     ncc deaths <=12: {nd12}, >12: {ndl}; ncc at risk at 12: {int(((T > 12) & (D['ncc'] == 1)).sum())}")
        ex = [(i_, T[i_], S_[i_]) for i_ in range(len(T)) if D["ncc"][i_] == 1][:3]
        print("     example rows:", ex)
# (c) tt(): ncc * log(t)
ftt = coxph(T, S_, X, tt=[(6, np.log)])
bt, st_ = ftt["coef"][7], ftt["se"][7]
print(f"   tt log(t): ncc main {ftt['coef'][6]:.4f}, ncc*log(t) {bt:.4f} (se {st_:.4f}) z={bt / st_:.2f} p={2 * st.norm.sf(abs(bt / st_)):.4f}")
for x in (1, 3, 6, 12, 24, 36):
    print(f"     HR(ncc) at t={x}: exp({ftt['coef'][6]:.3f} + {bt:.3f}*log({x})) = {np.exp(ftt['coef'][6] + bt * np.log(x)):.2f}")
print(f"     drug HR in tt model {ftt['hr'][0]:.3f}")

print("\n  E3. crossing-curves trial (chapter 7 figure 7-9 (b), seed 23)")
tt_, ss_, gg_ = scenario("late", 23)
fc = coxph(tt_, ss_, gg_[:, None])
print(f"   Cox HR {hrs(fc)}  events {ss_.sum()} (X {ss_[gg_ == 1].sum()}, Y {ss_[gg_ == 0].sum()})")
zc, gtc, scc = zph(fc, {"trt": [0]}, "km")
print("   zph(km) trt", np.round(zc["trt"], 4))
zci, _, _ = zph(fc, {"trt": [0]}, "identity")
print("   zph(identity) trt", np.round(zci["trt"], 4))
for cut in (4, 6):
    ids, a, b, ev, ep = survsplit(tt_, ss_, [cut])
    Xd = np.column_stack([gg_[ids] * (ep == 0), gg_[ids] * (ep == 1)])
    fs = coxph(b, ev, Xd, entry=a)
    fs0 = coxph(b, ev, gg_[ids][:, None], entry=a)
    lrt = 2 * (fs["loglik"] - fs0["loglik"])
    print(f"   split {cut}: 0-{cut} HR {fs['hr'][0]:.2f} ({fs['lo'][0]:.2f}-{fs['hi'][0]:.2f}) p={ptxt(fs['p'][0])}; >{cut} HR {fs['hr'][1]:.2f} ({fs['lo'][1]:.2f}-{fs['hi'][1]:.2f}) p={ptxt(fs['p'][1])}; "
          f"LR interaction {lrt:.2f} p={st.chi2.sf(lrt, 1):.4f}; events early X {int(((tt_ <= cut) & (ss_ == 1) & (gg_ == 1)).sum())} Y {int(((tt_ <= cut) & (ss_ == 1) & (gg_ == 0)).sum())}")
ftc = coxph(tt_, ss_, gg_[:, None], tt=[(0, np.log)])
print(f"   tt log(t): main {ftc['coef'][0]:.4f} slope {ftc['coef'][1]:.4f} se {ftc['se'][1]:.4f} p={ftc['p'][1]:.4f}")
for x in (1, 2, 4, 6, 12, 24, 36):
    print(f"     HR(t={x}) = {np.exp(ftc['coef'][0] + ftc['coef'][1] * np.log(x)):.2f}")
fx_ = km(tt_[gg_ == 1], ss_[gg_ == 1]); fy_ = km(tt_[gg_ == 0], ss_[gg_ == 0])
ax, sx_, _ = rmst(fx_, 36); ay, sy_, _ = rmst(fy_, 36)
print(f"   RMST36 X {ax:.2f} Y {ay:.2f} diff {ax - ay:.2f} ({ax - ay - Z * np.hypot(sx_, sy_):.2f} to {ax - ay + Z * np.hypot(sx_, sy_):.2f})")
for x in (4, 12, 24, 36):
    print(f"   S({x}) X {surv_at(fx_, x)[0]:.3f} Y {surv_at(fy_, x)[0]:.3f}")
for x in (1, 2, 4, 8, 12, 24, 36):
    a_ = surv_at(fx_, x)[0]; b_ = surv_at(fy_, x)[0]
    print(f"   LML t={x}: X {np.log(-np.log(a_)):.3f} Y {np.log(-np.log(b_)):.3f}")

print("=" * 72, "\nF. subgroups (adjusted for the other covariates) and P for interaction")
age65 = (D["age"] >= 65).astype(int)
sub_defs = [
    ("Age", [("< 65 years", age65 == 0), ("≥ 65 years", age65 == 1)], age65, None),
    ("Sex", [("Female", D["male"] == 0), ("Male", D["male"] == 1)], D["male"], None),
    ("IMDC risk group", [("Favorable", IM == 0), ("Intermediate", IM == 1), ("Poor", IM == 2)], None, [3, 4]),
    ("Prior nephrectomy", [("No", D["neph"] == 0), ("Yes", D["neph"] == 1)], D["neph"], None),
]
covcols = {"Age": [1, 2, 3, 4, 5, 6], "Sex": [1, 3, 4, 5, 6], "IMDC risk group": [1, 2, 5, 6], "Prior nephrectomy": [1, 2, 3, 4, 6]}
SUB = []
for name, levels, var, dcols in sub_defs:
    base_cols = [0, 1, 2, 3, 4, 5, 6]
    # interaction model
    if dcols is None:
        inter = (X[:, 0] * var)[:, None]
    else:
        inter = X[:, [0]] * X[:, dcols]
    f_int = coxph(T, S_, np.column_stack([X, inter]))
    lrt = 2 * (f_int["loglik"] - fm["loglik"])
    dfi = inter.shape[1]
    p_int = st.chi2.sf(lrt, dfi)
    print(f"  {name}: LR interaction {lrt:.2f} df {dfi} p={p_int:.3f}")
    rows_ = []
    for lab, m in levels:
        cc = [0] + covcols[name]
        # drop columns that are constant within the subgroup
        cc = [c for c in cc if np.ptp(X[m][:, c]) > 0 or c == 0]
        fsg = coxph(T[m], S_[m], X[m][:, cc])
        nAm, nBm = int((m & (G == 1)).sum()), int((m & (G == 0)).sum())
        ev_ = int(S_[m].sum())
        print(f"     {lab:13s} n={m.sum()} (A {nAm}/B {nBm}) deaths {ev_}: HR {fsg['hr'][0]:.2f} ({fsg['lo'][0]:.2f}-{fsg['hi'][0]:.2f})")
        rows_.append((lab, int(m.sum()), ev_, fsg["hr"][0], fsg["lo"][0], fsg["hi"][0]))
    SUB.append((name, p_int, rows_))

print("=" * 72, "\nG. sample size (Schoenfeld 1983) and power")
za, zb = st.norm.ppf(0.975), st.norm.ppf(0.80)
for HRx, pA in ((0.75, 0.5), (0.80, 0.5), (0.70, 0.5), (0.75, 1 / 3)):
    Dn = (za + zb) ** 2 / (pA * (1 - pA) * np.log(HRx) ** 2)
    print(f"   HR {HRx}, allocation {pA:.2f}: events = ({za:.3f}+{zb:.4f})^2 / ({pA * (1 - pA):.4f} x {np.log(HRx) ** 2:.5f}) = {Dn:.1f}")
Dn = (za + zb) ** 2 / (0.25 * np.log(0.75) ** 2)
for pe in (0.6, 0.3):
    print(f"   if {pe:.0%} die during follow-up: patients = {Dn:.1f}/{pe} = {Dn / pe:.0f}")
for HRx in (0.80, 0.86, 0.70):
    pw = st.norm.cdf(np.sqrt(210 * (152 / 300) * (148 / 300)) * abs(np.log(HRx)) - za)
    print(f"   power of this cohort (210 deaths, 152:148) for HR {HRx}: {pw:.3f}")
print(f"   z95 {za:.3f} z80 {zb:.4f} (za+zb)^2 = {(za + zb) ** 2:.3f}")

print("=" * 72, "\nH. IPTW (propensity score weighting) Cox")
psb, ps = logistic(G, X[:, 1:])
pA = G.mean()
w = np.where(G == 1, pA / ps, (1 - pA) / (1 - ps))
wu = np.where(G == 1, 1 / ps, 1 / (1 - ps))
print(f"   PS coef (intercept, age10, male, imdc1, imdc2, neph, ncc): {np.round(psb, 3)}")
print(f"   P(A)={pA:.4f}; PS range A {ps[G == 1].min():.3f}-{ps[G == 1].max():.3f}, B {ps[G == 0].min():.3f}-{ps[G == 0].max():.3f}")
print(f"   stabilized weights: mean {w.mean():.3f} (A {w[G == 1].mean():.3f}, B {w[G == 0].mean():.3f}) range {w.min():.2f}-{w.max():.2f}; sum {w.sum():.1f}")
print(f"   unstabilized: mean {wu.mean():.3f} range {wu.min():.2f}-{wu.max():.2f} sum {wu.sum():.1f}")
ess = lambda ww: ww.sum() ** 2 / (ww ** 2).sum()
print(f"   effective sample size A {ess(w[G == 1]):.1f} B {ess(w[G == 0]):.1f}")
for k, col in (("age", D["age"]), ("male", D["male"]), ("neph", D["neph"]), ("ncc", D["ncc"]),
               ("fav", (IM == 0).astype(float)), ("int", (IM == 1).astype(float)), ("poor", (IM == 2).astype(float))):
    print(f"   SMD {k:5s}: before {smd(col, G):+.3f} after {smd(col, G, w):+.4f}")
fw = coxph(T, S_, X[:, :1], weights=w, robust=True)
print(f"   IPTW Cox HR {fw['hr'][0]:.3f}; model-based SE {fw['se_model'][0]:.4f} CI {np.exp(fw['coef'][0] - Z * fw['se_model'][0]):.3f}-{np.exp(fw['coef'][0] + Z * fw['se_model'][0]):.3f};"
      f" robust SE {fw['se'][0]:.4f} CI {fw['lo'][0]:.3f}-{fw['hi'][0]:.3f} p={fw['p'][0]:.4f}")
fwb = coxph(T, S_, X[:, :1], weights=w, robust=True, ties="breslow")
print(f"   IPTW Breslow HR {fwb['hr'][0]:.3f} robust se {fwb['se'][0]:.4f}")
fwu = coxph(T, S_, X[:, :1], weights=wu, robust=True)
print(f"   unstabilized weights: HR {fwu['hr'][0]:.3f} model SE {fwu['se_model'][0]:.4f} robust SE {fwu['se'][0]:.4f}")
wkA = weighted_km(T[G == 1], S_[G == 1], w[G == 1]); wkB = weighted_km(T[G == 0], S_[G == 0], w[G == 0])
for x in (12, 24, 36, 48):
    print(f"   IPTW KM t={x}: A {step_at(wkA, x):.3f} B {step_at(wkB, x):.3f} diff {step_at(wkA, x) - step_at(wkB, x):+.3f}")
# robust SE for unweighted multivariable (sanity)
fmr = coxph(T, S_, X, robust=True)
print(f"   unweighted multivariable: model SE drug {fm['se'][0]:.4f}, robust SE {fmr['se'][0]:.4f}")

print("=" * 72, "\nI. non-collapsibility: randomized trial, no confounding (simulation, n = 20,000)")
rng = np.random.default_rng(5)
nn = 20000
trt = rng.integers(0, 2, nn)
risk = rng.integers(0, 2, nn)          # prognostic factor, 50%, independent of treatment
lam_ = 0.02 * np.where(risk == 1, 4.0, 1.0) * np.where(trt == 1, 0.5, 1.0)
Tt = rng.exponential(1 / lam_)
cens = 36.0
tobs = np.minimum(Tt, cens); dobs = (Tt <= cens).astype(int)


def fast_cox_1(time, status, x, ties="breslow"):
    """fast Breslow Cox for one binary covariate without tied times (sorted cumulative sums)."""
    o = np.argsort(-time)
    t_, s_, x_ = time[o], status[o], x[o].astype(float)
    b = 0.0
    for _ in range(30):
        r = np.exp(b * x_)
        s0 = np.cumsum(r); s1 = np.cumsum(r * x_)
        xb = s1 / s0
        U = np.sum(s_ * (x_ - xb))
        I = np.sum(s_ * (xb - xb ** 2))
        b += U / I
        if abs(U / I) < 1e-12:
            break
    return b, np.sqrt(1 / I)


def fast_cox_2(time, status, X2):
    o = np.argsort(-time)
    t_, s_, X_ = time[o], status[o], X2[o].astype(float)
    b = np.zeros(2)
    for _ in range(30):
        r = np.exp(X_ @ b)
        s0 = np.cumsum(r); s1 = np.cumsum(r[:, None] * X_, axis=0)
        s2 = np.cumsum(r[:, None, None] * X_[:, :, None] * X_[:, None, :], axis=0)
        xb = s1 / s0[:, None]
        U = (s_[:, None] * (X_ - xb)).sum(0)
        I = (s_[:, None, None] * (s2 / s0[:, None, None] - xb[:, :, None] * xb[:, None, :])).sum(0)
        step = np.linalg.solve(I, U)
        b += step
        if np.max(np.abs(step)) < 1e-12:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(I)))


bm, sm = fast_cox_1(tobs, dobs, trt)
bc, sc2 = fast_cox_2(tobs, dobs, np.column_stack([trt, risk]))
print(f"   events {dobs.sum()}; marginal (trt only) HR {np.exp(bm):.3f} ({np.exp(bm - Z * sm):.3f}-{np.exp(bm + Z * sm):.3f}); conditional (trt + factor) HR {np.exp(bc[0]):.3f} ({np.exp(bc[0] - Z * sc2[0]):.3f}-{np.exp(bc[0] + Z * sc2[0]):.3f}); factor HR {np.exp(bc[1]):.2f}")
# check fast vs slow engine on a subsample
sub = rng.choice(nn, 800, replace=False)
fsl = coxph(tobs[sub], dobs[sub], trt[sub][:, None], ties="breslow")
print(f"   engine check on 800: slow {fsl['coef'][0]:.6f} fast {fast_cox_1(tobs[sub], dobs[sub], trt[sub])[0]:.6f}")
# analytic marginal HR over time
for tq in (0, 6, 12, 24, 36):
    num = 0.5 * 0.5 * 0.02 * np.exp(-0.5 * 0.02 * tq) + 0.5 * 0.5 * 0.08 * np.exp(-0.5 * 0.08 * tq)
    den1 = 0.5 * np.exp(-0.5 * 0.02 * tq) + 0.5 * np.exp(-0.5 * 0.08 * tq)
    num0 = 0.5 * 0.02 * np.exp(-0.02 * tq) + 0.5 * 0.08 * np.exp(-0.08 * tq)
    den0 = 0.5 * np.exp(-0.02 * tq) + 0.5 * np.exp(-0.08 * tq)
    print(f"   population HR at t={tq}: {(num / den1 / 0.5) / (num0 / den0):.3f}")

print("=" * 72, "\nJ. functional form of age: martingale residuals and a quadratic term")
f_noage = coxph(T, S_, X[:, [0, 2, 3, 4, 5, 6]])
M = martingale(f_noage)
print(f"   martingale residuals: sum {M.sum():.2e}, range {M.min():.2f} to {M.max():.2f}")
xs_, ys_ = lowess(D["age"].astype(float), M, frac=0.6)
for a_ in (45, 55, 65, 75, 85):
    i_ = np.argmin(np.abs(xs_ - a_))
    print(f"     lowess at age~{xs_[i_]:.0f}: {ys_[i_]:+.3f}")
Xq = np.column_stack([X, ((D["age"] - 65) / 10.0) ** 2])
fq = coxph(T, S_, Xq)
lrq = 2 * (fq["loglik"] - fm["loglik"])
print(f"   add (age-65)^2/100: coef {fq['coef'][7]:.4f} se {fq['se'][7]:.4f}; LR {lrq:.3f} p={st.chi2.sf(lrq, 1):.3f}")
ag_c = ((D["age"] >= 65)).astype(float)
Xd2 = np.column_stack([X[:, [0, 2, 3, 4, 5, 6]], ag_c])
fd2 = coxph(T, S_, Xd2)
print(f"   age dichotomized at 65: HR {fd2['hr'][6]:.2f} ({fd2['lo'][6]:.2f}-{fd2['hi'][6]:.2f}); model loglik {fd2['loglik']:.2f} vs continuous {fm['loglik']:.2f}")

print("=" * 72, "\nK. Nelson-Aalen (12 patients) and hazard <-> survival")
rows12 = km(t, s)
H = 0.0
for r in rows12:
    if r["d"] > 0:
        H += r["d"] / r["n"]
        print(f"   t={r['t']:.0f}: d/n={r['d']}/{r['n']}  H={H:.4f}  exp(-H)={np.exp(-H):.3f}  KM={r['S']:.3f}")
print("   (H at 24 months = value after t=20)")

print("=" * 72, "\nL. HR vs RR: fresh examples; P(T_A < T_B)")
for HRx, S0x, lab in ((3.59, 0.65, "poor vs favorable, favorable 2-y death risk 35%"),
                      (1.30, 0.98, "claims: 1-y risk 2%"), (1.30, 0.60, "claims: 5-y risk 40%"),
                      (0.73, 0.95, "rare: risk 5%")):
    S1x = S0x ** HRx
    print(f"   HR {HRx} ({lab}): risk0={1 - S0x:.3f} risk1={1 - S1x:.4f} RR={(1 - S1x) / (1 - S0x):.3f} RD={(1 - S1x) - (1 - S0x):+.3f}")
for HRx in (0.7289, 0.8557, 3.59):
    print(f"   P(T_A < T_B) under PH = HR/(1+HR) = {HRx / (1 + HRx):.3f} (HR {HRx})")
# simulation check of HR/(1+HR)
rng_ = np.random.default_rng(1)
ta = rng_.weibull(1.4, 200000) / 0.7289 ** (1 / 1.4); tb = rng_.weibull(1.4, 200000)
print(f"   simulation (Weibull k=1.4, HR 0.7289): P(T_A<T_B) = {(ta < tb).mean():.3f}")
# RR curve data for the figure (drug B KM baseline, crude HR)
for x in (3, 6, 12, 24, 36, 48, 60):
    SB = surv_at(fb_, x)[0]
    print(f"   RR curve t={x}: {(1 - SB ** HRc) / (1 - SB):.3f}")

print("=" * 72, "\nM. exact (discrete) ties: score test at beta=0 equals the log-rank test")


def exact_ll(b, time, status, x):
    evt_ = np.unique(time[status == 1]); tot = 0.0
    for tk in evt_:
        R = time >= tk; Dm = (time == tk) & (status == 1); d = int(Dm.sum())
        w = np.exp(b * x[R]); e_ = np.zeros(d + 1); e_[0] = 1.0
        for wi in w:
            e_[1:] = e_[1:] + wi * e_[:-1]
        tot += b * x[Dm].sum() - np.log(e_[d])
    return tot


for lab, (tt0, ss0, xx0) in (("12 patients", (t, s, g.astype(float))), ("cohort", (T, S_, G.astype(float)))):
    h_ = 1e-4
    U0 = (exact_ll(h_, tt0, ss0, xx0) - exact_ll(-h_, tt0, ss0, xx0)) / (2 * h_)
    I0 = -(exact_ll(h_, tt0, ss0, xx0) - 2 * exact_ll(0, tt0, ss0, xx0) + exact_ll(-h_, tt0, ss0, xx0)) / h_ ** 2
    lr_ = logrank(tt0, ss0, xx0.astype(int), 1)
    print(f"   {lab}: exact score U^2/I = {U0 ** 2 / I0:.4f}; log-rank = {lr_['chi2']:.4f}; U = {U0:.4f} vs O-E = {lr_['U']:.4f}")

print("=" * 72, "\nN. influence of single patients on the adjusted drug HR (leave-one-out)")
loo = np.zeros(len(T))
for i in range(len(T)):
    keep = np.arange(len(T)) != i
    loo[i] = coxph(T[keep], S_[keep], X[keep])["coef"][0]
dfb = fm["coef"][0] - loo            # change in coefficient when patient i is removed (R: dfbeta sign = b - b(-i))
from lib_ch11 import score_residuals
L_ = score_residuals(fm)
dfb_approx = (L_ @ np.linalg.inv(fm["info"]))[:, 0]
o_ = np.argsort(-np.abs(dfb))
for i in o_[:3]:
    print(f"   patient idx {i}: time {T[i]} status {S_[i]} drug {G[i]} imdc {IM[i]} age {D['age'][i]} ncc {D['ncc'][i]} -> HR without = {np.exp(loo[i]):.3f} "
          f"(dfbeta {dfb[i]:+.4f}, approx {dfb_approx[i]:+.4f}; dfbeta/se = {dfb[i] / fm['se'][0]:+.3f})")
print(f"   range of HR when one patient is removed: {np.exp(loo.min()):.3f} to {np.exp(loo.max()):.3f}")
print(f"   max |dfbeta|/se = {np.max(np.abs(dfb)) / fm['se'][0]:.3f}; corr(exact, approx) = {np.corrcoef(dfb, dfb_approx)[0, 1]:.4f}")

print("=" * 72, "\nO. simulated data with known beta (validation of the multivariable Cox engine)")
rng_ = np.random.default_rng(2025)
nS = 3000
trtS = rng_.integers(0, 2, nS); ageS = rng_.normal(0, 1, nS); grpS = rng_.choice(3, nS, p=[0.3, 0.5, 0.2])
XS = np.column_stack([trtS, ageS, grpS == 1, grpS == 2]).astype(float)
bS = np.array([-0.4, 0.3, 0.5, 1.1])
kS, lamS = 1.3, 0.02
TS = (rng_.exponential(1, nS) / (lamS * np.exp(XS @ bS))) ** (1 / kS)
CS = rng_.uniform(10, 60, nS)
tS = np.round(np.minimum(TS, CS), 1); dS = (TS <= CS).astype(int)
fS = coxph(tS, dS, XS)
print(f"   n={nS} events={dS.sum()} true beta {bS}")
print(f"   estimated {np.round(fS['coef'], 4)}  se {np.round(fS['se'], 4)}  z vs truth {np.round((fS['coef'] - bS) / fS['se'], 2)}")
# coverage of the 95% CI for the treatment coefficient (n = 300 per replicate)
cov = 0; nrep = 200; ests = []
for rep in range(nrep):
    ii = rng_.integers(0, 2, 300); aa = rng_.normal(0, 1, 300)
    Xr = np.column_stack([ii, aa]).astype(float)
    Tr = (rng_.exponential(1, 300) / (lamS * np.exp(Xr @ bS[:2]))) ** (1 / kS)
    Cr = rng_.uniform(10, 60, 300)
    tr_ = np.round(np.minimum(Tr, Cr), 1); dr_ = (Tr <= Cr).astype(int)
    fr = coxph(tr_, dr_, Xr)
    ests.append(fr["coef"][0])
    cov += (fr["coef"][0] - Z * fr["se"][0] <= bS[0] <= fr["coef"][0] + Z * fr["se"][0])
print(f"   {nrep} replicates (n=300): mean estimate {np.mean(ests):.4f} (true -0.4); 95% CI coverage {cov / nrep:.3f}")

print("=" * 72, "\nP. Schoenfeld residuals by hand (12 patients, Efron fit) and small extras")
sc12 = schoenfeld(fe)
for tk, r_ in zip(sc12["time"], sc12["resid"][:, 0]):
    print(f"   t={tk:.0f}: residual {r_:+.4f}")
print(f"   sum of residuals = {sc12['resid'].sum():.2e} (score equation at beta_hat)")
e_ = np.exp(fe["coef"][0])
print(f"   t=28 by hand: 1 - 3e/(3e+1) = 1 - {3 * e_:.4f}/{3 * e_ + 1:.4f} = {1 - 3 * e_ / (3 * e_ + 1):.4f};  t=3: 0 - 6e/(6e+6) = {-e_ / (e_ + 1):.4f}")
print(f"   drug B monthly rate {110 / 284.9 / 12:.4f}: per 1000 B alive ~ {1000 * 110 / 284.9 / 12:.1f} deaths/month; A ~ {1000 * 110 / 284.9 / 12 * HRc:.1f}")
for name, p_ in [(n_, p__) for n_, p__, _ in SUB]:
    print(f"   P interaction {name}: {p_:.4f}")
print(f"   concordance univariable by hand: ({c1['concordant']} + 0.5 x {c1['tied']}) / {c1['pairs']} = {(c1['concordant'] + 0.5 * c1['tied']) / c1['pairs']:.4f}")
print(f"   mean covariates (centering, R 'means'): {np.round(fm['means'], 4)}")
print(f"   crude B vs A: HR {1 / f1['hr'][0]:.3f} CI {1 / f1['hi'][0]:.3f}-{1 / f1['lo'][0]:.3f}")
print(f"   CI asymmetry around crude HR: down {f1['hr'][0] - f1['lo'][0]:.3f}, up {f1['hi'][0] - f1['hr'][0]:.3f}")
for x in (3, 6, 12, 24, 36, 48):
    a_ = np.log(-np.log(surv_at(fa, x)[0])) - np.log(-np.log(surv_at(fb_, x)[0]))
    b_ = np.log(-np.log(surv_at(fn1, x)[0])) - np.log(-np.log(surv_at(fn0, x)[0])) if x <= 36 else np.nan
    print(f"   LML gap t={x}: drug {a_:.2f}  histology {b_:.2f}")
print(f"   scaled Schoenfeld histology means: early {scl[tm < CUT_NCC, 6].mean():.2f} (HR {np.exp(scl[tm < CUT_NCC, 6].mean()):.1f}), late {scl[tm >= CUT_NCC, 6].mean():.2f} (HR {np.exp(scl[tm >= CUT_NCC, 6].mean()):.2f})")
ids12, a12, b12, ev12, ep12 = survsplit(T, S_, [12])
Xd12 = np.column_stack([X[ids12][:, :6], X[ids12][:, 6] * (ep12 == 0), X[ids12][:, 6] * (ep12 == 1)])
fs12 = coxph(b12, ev12, Xd12, entry=a12)
print(f"   split-12 model drug HR {fs12['hr'][0]:.4f} ({fs12['lo'][0]:.3f}-{fs12['hi'][0]:.3f}); tt model drug HR {ftt['hr'][0]:.4f} ({ftt['lo'][0]:.3f}-{ftt['hi'][0]:.3f}); tt slope p {2 * st.norm.sf(abs(bt / st_)):.5f}")
print(f"   strata(histology) drug {fsH['hr'][0]:.4f}")
for tq in (0, 6, 12, 24, 36):
    num = 0.5 * 0.5 * 0.02 * np.exp(-0.5 * 0.02 * tq) + 0.5 * 0.5 * 0.08 * np.exp(-0.5 * 0.08 * tq)
    den1 = 0.5 * np.exp(-0.5 * 0.02 * tq) + 0.5 * np.exp(-0.5 * 0.08 * tq)
    num0 = 0.5 * 0.02 * np.exp(-0.02 * tq) + 0.5 * 0.08 * np.exp(-0.08 * tq)
    den0 = 0.5 * np.exp(-0.02 * tq) + 0.5 * np.exp(-0.08 * tq)
    print(f"   marginal (population) HR at t={tq}: {(num / den1) / (num0 / den0):.3f}")
print(f"   standardized curves at 24: A {adj[1][2]:.4f} B {adj[0][2]:.4f}; B^HR_adj = {adj[0][2] ** fm['hr'][0]:.4f}")
print(f"   univariable Wald z^2 = {f1['z'][0] ** 2:.3f}; LR by hand 2*({f1['loglik']:.3f} - ({f1['loglik0']:.3f})) = {2 * (f1['loglik'] - f1['loglik0']):.3f}")
print(f"   12 patients Wald z = {fe['coef'][0] / fe['se'][0]:.3f}")

print("=" * 72, "\nQ. Python (lifelines) output boxes")
from lib_ch11 import ph_test_approx
# CoxPHFitter.print_summary(decimals=3) for the univariable drug model
b1, s1 = f1["coef"][0], f1["se"][0]
print(f"   n={len(T)} events={int(S_.sum())} right-censored={len(T) - int(S_.sum())}; partial log-likelihood {f1['loglik']:.3f}")
print(f"   coef {b1:.3f} exp {np.exp(b1):.3f} se {s1:.3f} coef CI {b1 - Z * s1:.3f} {b1 + Z * s1:.3f} exp CI {np.exp(b1 - Z * s1):.3f} {np.exp(b1 + Z * s1):.3f}")
print(f"   z {b1 / s1:.3f} p {f1['p'][0]:.3f} -log2(p) {-np.log2(f1['p'][0]):.3f}")
print(f"   Concordance {c1['C']:.3f}; Partial AIC = -2 loglik + 2k = {-2 * f1['loglik'] + 2:.3f}; LR test {f1['lr']:.3f} on 1 df, -log2(p) {-np.log2(f1['p_lr']):.3f}")
print(f"   Wald z^2 {f1['z'][0] ** 2:.3f}")
# proportional_hazard_test(cph, df, time_transform=["km", "rank"]) for the multivariable model
for tr in ("km", "rank", "identity", "log"):
    Tt, pp, lg = ph_test_approx(fm, tr)
    for k_, a_, b_, c_ in zip(nm, Tt, pp, lg):
        print(f"   PH test {tr:8s} {k_:6s} stat {a_:.3f} p {b_:.4f} -log2(p) {c_:.3f}")
Ta, pa, _ = ph_test_approx(fm, "rank_avg")
print(f"   PH test with average ranks (other programs): drug stat {Ta[0]:.3f} p {pa[0]:.3f}")
Tt1, pp1, _ = ph_test_approx(f1, "km")
print(f"   univariable drug PH test km: stat {Tt1[0]:.3f} p {pp1[0]:.3f}")
for tr in ("km", "rank"):
    print(f"   stratified-by-histology model PH test {tr}: p = {np.round(ph_test_approx(fsH, tr)[1], 3)} (min {ph_test_approx(fsH, tr)[1].min():.3f})")
for tr in ("km", "rank"):
    Tc, pc, _ = ph_test_approx(fc, tr)
    print(f"   crossing trial PH test {tr}: stat {Tc[0]:.2f} p {pc[0]:.3f}")
print(f"   global (score) test of all 7 coefficients, km: chisq {zr['GLOBAL'][0]:.2f} df 7 p {zr['GLOBAL'][2]:.3f}")
print(f"   nephrectomy by IMDC: {[round(D['neph'][IM == k].mean(), 3) for k in range(3)]}; ncc by IMDC: {[round(D['ncc'][IM == k].mean(), 3) for k in range(3)]}")
print(f"   histology HR crude {uni['ncc'][0]:.3f}; adjusted for IMDC only {coxph(T, S_, X[:, [6, 3, 4]])['hr'][0]:.3f}")

# real lifelines 0.30 output for the two boxes (source /home/claude/pylibs/env.sh first)
try:
    import pandas as pd
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test
except ImportError:
    CoxPHFitter = None
    print("   (lifelines not available: skip the real-library check)")
if CoxPHFitter is not None:
    df_ll = pd.DataFrame({"time": T, "status": S_, "drug_A": X[:, 0], "age10": X[:, 1], "male": X[:, 2],
                          "imdc_int": X[:, 3], "imdc_poor": X[:, 4], "nephrectomy": X[:, 5], "histology": X[:, 6]})
    cph1 = CoxPHFitter().fit(df_ll[["time", "status", "drug_A"]], duration_col="time", event_col="status")
    cph1.print_summary(decimals=3)
    assert abs(cph1.log_likelihood_ - f1["loglik"]) < 1e-6 and abs(cph1.params_.iloc[0] - f1["coef"][0]) < 1e-6
    assert abs(cph1.standard_errors_.iloc[0] - f1["se"][0]) < 1e-6 and abs(cph1.concordance_index_ - c1["C"]) < 1e-9
    cols_ll = ["drug_A", "age10", "male", "imdc_int", "imdc_poor", "nephrectomy", "histology"]
    cph = CoxPHFitter().fit(df_ll[["time", "status"] + cols_ll], duration_col="time", event_col="status")
    assert np.allclose(cph.params_.values, fm["coef"], atol=1e-6) and np.allclose(cph.standard_errors_.values, fm["se"], atol=1e-6)
    res_ll = proportional_hazard_test(cph, df_ll, time_transform=["km", "rank"])
    res_ll.print_summary(decimals=3)
    for tr in ("km", "rank"):
        T_in = dict(zip(cols_ll, ph_test_approx(fm, tr)[0]))
        assert all(abs(res_ll.summary.loc[(c_, tr), "test_statistic"] - T_in[c_]) < 1e-6 for c_ in cols_ll), tr
    for tr in ("identity", "log"):
        r_ = proportional_hazard_test(cph, df_ll, time_transform=tr).summary
        print(f"   lifelines {tr}: male p {r_.loc['male', 'p']:.4f}  histology p {r_.loc['histology', 'p']:.2e}")
    print("   lifelines check passed: Efron ties, same coefficients/SE/loglik/concordance and PH statistics (km, rank)")

print("=" * 72, "\nZ. validation summary")
print("  12 patients: coef equal to lib_ch07.cox_efron:", np.isclose(cox_efron(t, s, g)["beta"], fe["coef"][0]))
# no ties -> score test equals log-rank exactly
t_nt = t + np.arange(len(t)) * 0.01
print(f"  12 patients with ties broken: Cox score {coxph(t_nt, s, g[:, None])['score']:.5f} vs log-rank {logrank(t_nt, s, g, 1)['chi2']:.5f}")
print(f"  cohort: Cox score (Efron) {f1['score']:.4f}, (Breslow) {f1b['score']:.4f}, log-rank {logrank(T, S_, G, 1)['chi2']:.4f}")
Tj = T + np.random.default_rng(3).uniform(0, 0.01, len(T))
print(f"  cohort, ties broken by jitter: Cox score {coxph(Tj, S_, X[:, :1])['score']:.5f} vs log-rank {logrank(Tj, S_, G, 1)['chi2']:.5f}")
print("  gradient checks: univariable", num_grad_check(f1), " stratified", num_grad_check(fsH), " weighted", num_grad_check(fw))
ids, a, b, ev, ep = survsplit(T, S_, [12])
fsplit = coxph(b, ev, X[ids], entry=a)
print(f"  split data without time interaction reproduces the fit: {np.allclose(fsplit['coef'], fm['coef'], atol=1e-8)}")
print("  counting-process gradient check", num_grad_check(fsplit))
