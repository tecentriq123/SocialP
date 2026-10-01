"""Chapter 8: every number in content/ch08.html is computed here (run: python3 gen/nums_ch08.py).
fig_ch08.py imports this module (R dict) so figures, text and tables share the same values."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.stats as st
from scipy.optimize import brentq
from lib_ch08 import glm, loglik, robust_se, wald_rows, make_readmit, design, logistic, Z

R = {}
LOG = []


def out(*a):
    LOG.append(" ".join(str(x) for x in a))


def rr_ci(a, n1, c, n0):
    r1, r0 = a / n1, c / n0
    rr = r1 / r0
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    return rr, rr * np.exp(-Z * se), rr * np.exp(Z * se), se


def or_ci(a, n1, c, n0):
    b, d = n1 - a, n0 - c
    o = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return o, o * np.exp(-Z * se), o * np.exp(Z * se), se


def rd_ci(a, n1, c, n0):
    r1, r0 = a / n1, c / n0
    se = np.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
    return r1 - r0, r1 - r0 - Z * se, r1 - r0 + Z * se, se


def wilson(x, n, z=Z):
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def clopper(x, n):
    return st.beta.ppf(0.025, x, n - x + 1), st.beta.ppf(0.975, x + 1, n - x)


# =================================================================== 가. 95% CI
# (1) mean: HbA1c change after pharmacist counselling, n = 25, mean -0.62, SD 0.80
m, s, n = -0.62, 0.80, 25
se = s / np.sqrt(n); tc = st.t.ppf(0.975, n - 1)
R["mean"] = dict(m=m, s=s, n=n, se=se, tc=tc, half=tc * se, lo=m - tc * se, hi=m + tc * se)
out("mean CI", R["mean"])
R["width"] = []
for nn in (25, 100, 400):
    se_ = s / np.sqrt(nn); t_ = st.t.ppf(0.975, nn - 1)
    R["width"].append(dict(n=nn, se=se_, t=t_, half=t_ * se_, lo=m - t_ * se_, hi=m + t_ * se_))
    out("width", nn, round(se_, 3), round(t_, 3), round(t_ * se_, 3), round(m - t_ * se_, 2), round(m + t_ * se_, 2))

# (2) proportion: 7 ADRs among 50 patients
x, nn = 7, 50
p = x / nn; sep = np.sqrt(p * (1 - p) / nn)
R["prop"] = dict(x=x, n=nn, p=p, se=sep, wald=(p - Z * sep, p + Z * sep), wilson=wilson(x, nn), exact=clopper(x, nn))
out("prop", R["prop"])

# (3) difference of proportions (chapter 5 data 48/400 vs 102/600)
R["rd5"] = rd_ci(48, 400, 102, 600)
out("RD ch05", R["rd5"])
# (4) ratio: RR 48/400 vs 102/600 on log scale
rr, lo, hi, se_l = rr_ci(48, 400, 102, 600)
R["rr5"] = dict(rr=rr, lo=lo, hi=hi, se=se_l, ln=np.log(rr), lnlo=np.log(rr) - Z * se_l, lnhi=np.log(rr) + Z * se_l,
                geo=np.sqrt(lo * hi), dlo=rr - lo, dhi=hi - rr)
out("RR ch05", R["rr5"])
# p from CI (Altman & Bland 2011): SE = (ln hi - ln lo)/(2*1.96) using the rounded published CI 0.51-0.97
se_pub = (np.log(0.97) - np.log(0.51)) / (2 * 1.96)
z_pub = np.log(0.71) / se_pub
R["p_from_ci"] = dict(se=se_pub, z=z_pub, p=2 * st.norm.sf(abs(z_pub)), lnrr=np.log(0.71), lnlo=np.log(0.51), lnhi=np.log(0.97))
out("p from CI", R["p_from_ci"])

# overlap of two CIs: means 10 and 13, SE 1 each
d_, se_d = 3.0, np.sqrt(2)
R["overlap"] = dict(ci1=(10 - Z, 10 + Z), ci2=(13 - Z, 13 + Z), z=d_ / se_d, p=2 * st.norm.sf(d_ / se_d),
                    dlo=d_ - Z * se_d, dhi=d_ + Z * se_d)
out("overlap", R["overlap"])

# four/five scenarios vs MCID 5 mmHg (difference in SBP reduction, positive = benefit)
SCEN = [("A", 9.0, 6.5, 11.5), ("B", 4.3, 0.6, 8.0), ("C", 1.5, 0.5, 2.5), ("D", 3.0, -2.0, 8.0), ("E", 0.2, -1.5, 1.9)]
R["scen"] = []
for lab, e, l, h in SCEN:
    se_ = (h - l) / (2 * Z); zz = e / se_
    R["scen"].append(dict(lab=lab, est=e, lo=l, hi=h, se=se_, z=zz, p=2 * st.norm.sf(abs(zz))))
    out("scen", lab, e, l, h, round(se_, 3), round(zz, 2), round(2 * st.norm.sf(abs(zz)), 4))

# simulation of 20 CIs: true mean change -0.5, SD 0.8, n = 25; choose a seed with exactly one miss
TRUE_MU, SIG, NS = -0.5, 0.8, 25
for seed in range(1, 500):
    rng = np.random.default_rng(seed)
    ci = []
    for i in range(20):
        xs = rng.normal(TRUE_MU, SIG, NS)
        mm, ss = xs.mean(), xs.std(ddof=1)
        h = st.t.ppf(0.975, NS - 1) * ss / np.sqrt(NS)
        ci.append((mm, mm - h, mm + h))
    miss = [i for i, (mm, l, h) in enumerate(ci) if not (l <= TRUE_MU <= h)]
    if len(miss) == 1 and 6 <= miss[0] <= 14:
        break
R["sim20"] = dict(seed=seed, ci=ci, miss=miss, mu=TRUE_MU)
out("sim20 seed", seed, "miss", miss, [tuple(round(v, 2) for v in c) for c in ci])
# long-run coverage check of the procedure (and of z = 1.96 with n = 10)
rng = np.random.default_rng(1)
xs = rng.normal(TRUE_MU, SIG, (100000, 10))
mm, ss = xs.mean(1), xs.std(1, ddof=1)
cov_t = np.mean(np.abs(mm - TRUE_MU) <= st.t.ppf(0.975, 9) * ss / np.sqrt(10))
cov_z = np.mean(np.abs(mm - TRUE_MU) <= 1.96 * ss / np.sqrt(10))
R["cov10"] = dict(t=cov_t, z=cov_z)
out("coverage n=10 t", cov_t, "z", cov_z)
R["tq"] = {df: [st.t.ppf(q, df) for q in (0.95, 0.975, 0.995)] for df in (9, 29, 99)}
R["zq"] = [st.norm.ppf(q) for q in (0.95, 0.975, 0.995)]
out("t quantiles", R["tq"], R["zq"])

# paper box 가-1: pharmacist RCT in T2DM, 120 per group
n1 = n0 = 120
m1, s1, m0, s0 = -0.95, 1.10, -0.53, 1.15
dif = m1 - m0
se_ = np.sqrt(s1 ** 2 / n1 + s0 ** 2 / n0)
sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n0 - 1) * s0 ** 2) / (n1 + n0 - 2)); se_p = sp * np.sqrt(1 / n1 + 1 / n0)
tcr = st.t.ppf(0.975, n1 + n0 - 2)
tt = dif / se_p
R["rct_a1c"] = dict(dif=dif, se=se_p, t=tt, df=n1 + n0 - 2, p=2 * st.t.sf(abs(tt), n1 + n0 - 2), lo=dif - tcr * se_p, hi=dif + tcr * se_p)
out("rct a1c", R["rct_a1c"])
goal = (46, 120, 33, 120)
R["rct_goal"] = dict(rr=rr_ci(*goal), rd=rd_ci(*goal), p1=46 / 120, p0=33 / 120,
                     chi=st.chi2_contingency(np.array([[46, 74], [33, 87]]), correction=False)[1])
out("rct goal", R["rct_goal"])
hypo = (9, 120, 11, 120)
R["rct_hypo"] = dict(rr=rr_ci(*hypo), p1=9 / 120, p0=11 / 120)
out("rct hypo", R["rct_hypo"])

# paper box 가-2: observational HR reading (HR 1.18, 0.89-1.56) -> implied SE and P
lo_, hi_, hr_ = 0.89, 1.56, 1.18
se_h = (np.log(hi_) - np.log(lo_)) / (2 * 1.96)
R["hr_obs"] = dict(se=se_h, z=np.log(hr_) / se_h, p=2 * st.norm.sf(abs(np.log(hr_) / se_h)))
out("hr obs", R["hr_obs"])

# =================================================================== 나. RR and OR
# RCT: pharmacist-led medication review vs usual care, 300 per group, 90-day outcomes
RCT = {"readm": (45, 300, 72, 300), "ed": (60, 300, 75, 300), "ade": (18, 300, 30, 300)}
R["rct"] = {}
for k, v in RCT.items():
    a, n1_, c, n0_ = v
    rr_ = rr_ci(*v); or_ = or_ci(*v); rd_ = rd_ci(*v)
    nnt = -1 / rd_[0]
    # NNT CI by inverting RD limits (Altman 1998)
    lims = sorted([-1 / rd_[1], -1 / rd_[2]])
    chi = st.chi2_contingency(np.array([[a, n1_ - a], [c, n0_ - c]]), correction=False)
    R["rct"][k] = dict(p1=a / n1_, p0=c / n0_, rr=rr_, orr=or_, rd=rd_, rrr=1 - rr_[0], nnt=nnt,
                       nnt_from_rdlo=-1 / rd_[1], nnt_from_rdhi=-1 / rd_[2], p=chi[1], chi2=chi[0])
    out("rct", k, "risk", a / n1_, c / n0_, "RR", [round(x, 3) for x in rr_[:3]], "OR", [round(x, 3) for x in or_[:3]],
        "RD", [round(x * 100, 2) for x in rd_[:3]], "RRR", round((1 - rr_[0]) * 100, 1), "NNT", round(nnt, 2),
        "NNT lims from RD lo/hi", round(-1 / rd_[1], 2), round(-1 / rd_[2], 2), "P", round(chi[1], 4))
# OR = RR * (1-p0)/(1-p1)
p1_, p0_ = 0.15, 0.24
R["or_rr_identity"] = dict(factor=(1 - p0_) / (1 - p1_), rr=p1_ / p0_, orv=(p1_ / p0_) * (1 - p0_) / (1 - p1_))
out("OR identity", R["or_rr_identity"])
# same RR at low baseline risk
for p0b in (0.24, 0.10, 0.024):
    p1b = 0.625 * p0b
    o_ = (p1b / (1 - p1b)) / (p0b / (1 - p0b))
    out("baseline", p0b, "p1", p1b, "ARR", round((p0b - p1b) * 100, 2), "NNT", round(1 / (p0b - p1b), 1), "OR", round(o_, 3))
R["lowrisk"] = dict(p0=0.024, p1=0.625 * 0.024, arr=0.024 * 0.375, nnt=1 / (0.024 * 0.375))
# case-control from a source population
pop = dict(e_case=100, e_n=10000, u_case=200, u_n=40000)
rr_pop = (100 / 10000) / (200 / 40000)
or_pop = (100 / 9900) / (200 / 39800)
p_exp_noncase = 9900 / (9900 + 39800)
cc = {}
for k_ in (1, 2, 4):
    ctrl = 300 * k_
    ec = round(ctrl * p_exp_noncase)
    uc = ctrl - ec
    fake_rr = (100 / (100 + ec)) / (200 / (200 + uc))
    or_cc = (100 * uc) / (200 * ec)
    cc[k_] = dict(ctrl=ctrl, ec=ec, uc=uc, fake_rr=fake_rr, orv=or_cc, r1=100 / (100 + ec), r0=200 / (200 + uc))
    out("case-control", k_, cc[k_])
R["cc"] = dict(rr_pop=rr_pop, or_pop=or_pop, pexp=p_exp_noncase, tab=cc)
out("pop RR", rr_pop, "pop OR", or_pop, "p exp among noncases", p_exp_noncase)

# case-control paper box: PPI and hip fracture? -> use 'benzodiazepine & hip fracture' nested case-control
# 1,200 cases, 4,800 matched controls; current use: 180 cases, 520 controls
a, b_, c_, d_ = 180, 1200 - 180, 520, 4800 - 520
o_, l_, h_, s_ = or_ci(a, 1200, c_, 4800)
R["ccbox"] = dict(a=a, b=b_, c=c_, d=d_, orv=o_, lo=l_, hi=h_, pcase=a / 1200, pctrl=c_ / 4800)
out("cc box crude OR", R["ccbox"])

# =================================================================== 다-마: discharge cohort (n = 1,500)
D = make_readmit()
n = D["n"]
R["D"] = D
tab = dict(np_=int(D["poly"].sum()), n0=int(n - D["poly"].sum()), e1=int(D["readm"][D["poly"] == 1].sum()),
           e0=int(D["readm"][D["poly"] == 0].sum()), ev=int(D["readm"].sum()))
R["tab"] = tab
cr = dict(rr=rr_ci(tab["e1"], tab["np_"], tab["e0"], tab["n0"]), orr=or_ci(tab["e1"], tab["np_"], tab["e0"], tab["n0"]),
          rd=rd_ci(tab["e1"], tab["np_"], tab["e0"], tab["n0"]), r1=tab["e1"] / tab["np_"], r0=tab["e0"] / tab["n0"])
R["crude"] = cr
out("cohort n", n, tab, "risk poly", round(cr["r1"], 4), "nonpoly", round(cr["r0"], 4), "overall", round(tab["ev"] / n, 4))
out("crude RR", [round(v, 3) for v in cr["rr"][:3]], "OR", [round(v, 3) for v in cr["orr"][:3]], "RD", [round(v * 100, 2) for v in cr["rd"][:3]])
# descriptives
out("age mean", D["age"].mean().round(2), "sd", D["age"].std(ddof=1).round(2), "min/max", D["age"].min(), D["age"].max(),
    "female", D["female"].mean().round(3), "medaid", D["medaid"].mean().round(3), "cci dist", np.bincount(D["cci"]),
    "poly", D["poly"].mean().round(3))
out("age by medaid", D["age"][D["medaid"] == 1].mean().round(1), D["age"][D["medaid"] == 0].mean().round(1),
    "poly by medaid", D["poly"][D["medaid"] == 1].mean().round(3), D["poly"][D["medaid"] == 0].mean().round(3),
    "readm by medaid", D["readm"][D["medaid"] == 1].mean().round(3), D["readm"][D["medaid"] == 0].mean().round(3))
for k in range(4):
    msk = D["cci"] == k
    out("cci", k, "n", msk.sum(), "events", D["readm"][msk].sum(), "risk", D["readm"][msk].mean().round(4),
        "poly%", D["poly"][msk].mean().round(3))
out("poly by cci", [D["poly"][D["cci"] == k].mean().round(3) for k in range(4)])
out("age by poly", D["age"][D["poly"] == 1].mean().round(1), D["age"][D["poly"] == 0].mean().round(1))

FULL = ["poly", "age", "female", "cci_d", "medaid"]
full = logistic(D, FULL)
R["full"] = full
R["full_rows"] = wald_rows(full, full["names"])
out("FULL model -2LL", -2 * full["ll"], "dev", full["dev"], "AIC", -2 * full["ll"] + 2 * len(full["beta"]))
for r in R["full_rows"]:
    out("  ", r["name"], round(r["b"], 5), round(r["se"], 5), round(r["z"], 3), "%.3g" % r["p"], "OR", round(np.exp(r["b"]), 3),
        round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3))
null = logistic(D, [])
R["null"] = null
out("null -2LL", -2 * null["ll"])


b = dict(zip(full["names"], full["beta"]))
se = dict(zip(full["names"], full["se"]))
R["age10"] = dict(b=b["age"], or1=np.exp(b["age"]), or10=np.exp(10 * b["age"]), naive=10 * (np.exp(b["age"]) - 1),
                  lo10=np.exp(10 * (b["age"] - Z * se["age"])), hi10=np.exp(10 * (b["age"] + Z * se["age"])), or5=np.exp(5 * b["age"]))
out("age per 10", R["age10"])
R["joint"] = dict(poly=np.exp(b["poly"]), cci5=np.exp(b["cci5+"]), both=np.exp(b["poly"] + b["cci5+"]),
                  add=np.exp(b["poly"]) + np.exp(b["cci5+"]) - 1)
out("joint", R["joint"])
# model-predicted risk for a reference person: 80-year-old man, CCI 3-4, no medical aid, poly 0/1
xr = np.array([1, 0, 80, 0, 0, 1, 0, 0.0])
for pv in (0, 1):
    xr[1] = pv
    eta = xr @ full["beta"]
    out("pred 80M cci3-4 poly", pv, "logit", round(eta, 4), "odds", round(np.exp(eta), 4), "risk", round(1 / (1 + np.exp(-eta)), 4))
R["pred"] = []
for pv in (0, 1):
    xr[1] = pv
    eta = xr @ full["beta"]
    R["pred"].append(dict(eta=eta, odds=np.exp(eta), risk=1 / (1 + np.exp(-eta))))

# ------------------------------------------------ 라. same covariates, different GLMs
X, names = design(D, FULL)
y = D["readm"].astype(float)
lb = glm(X, y, "binomial", "log", start=np.r_[np.log(y.mean()), np.zeros(X.shape[1] - 1)])
out("log-binomial iters", lb["iters"], "max fitted", lb["mu"].max().round(4))
mp = glm(X, y, "poisson", "log")
mp_se, _ = robust_se(mp)
out("mod poisson max fitted", mp["mu"].max().round(4), "n fitted>1", int((mp["mu"] > 1).sum()))
try:
    idb = glm(X, y, "binomial", "identity", start=np.linalg.lstsq(X, y, rcond=None)[0])
    idb_ok = True
except Exception as e:
    idb_ok = False
    out("identity-binomial failed", e)
ols = np.linalg.lstsq(X, y, rcond=None)[0]
res = y - X @ ols
bread = np.linalg.inv(X.T @ X)
ols_se = np.sqrt(np.diag(bread @ (X.T * res ** 2) @ X @ bread))
out("LPM fitted range", (X @ ols).min().round(3), (X @ ols).max().round(3))
R["glmcmp"] = dict(
    logit=(np.exp(full["beta"][1]), np.exp(full["beta"][1] - Z * full["se"][1]), np.exp(full["beta"][1] + Z * full["se"][1]), full["beta"][1], full["se"][1]),
    logbin=(np.exp(lb["beta"][1]), np.exp(lb["beta"][1] - Z * lb["se"][1]), np.exp(lb["beta"][1] + Z * lb["se"][1]), lb["beta"][1], lb["se"][1]),
    mpois=(np.exp(mp["beta"][1]), np.exp(mp["beta"][1] - Z * mp_se[1]), np.exp(mp["beta"][1] + Z * mp_se[1]), mp["beta"][1], mp_se[1]),
    mpois_naive_se=mp["se"][1],
    lpm=(ols[1], ols[1] - Z * ols_se[1], ols[1] + Z * ols_se[1]),
)
if idb_ok:
    R["glmcmp"]["idbin"] = (idb["beta"][1], idb["beta"][1] - Z * idb["se"][1], idb["beta"][1] + Z * idb["se"][1])
for k, v in R["glmcmp"].items():
    out("GLM", k, np.round(v, 4) if not np.isscalar(v) else round(v, 4))
# crude versions from GLMs equal the 2x2 table
Xc = np.column_stack([np.ones(n), D["poly"]])
cl = glm(Xc, y, "binomial", "logit"); clb = glm(Xc, y, "binomial", "log", start=[np.log(y.mean()), 0]); cid = glm(Xc, y, "binomial", "identity", start=[y.mean(), 0])
out("crude via GLM: OR", np.exp(cl["beta"][1]).round(4), "RR", np.exp(clb["beta"][1]).round(4), "RD", cid["beta"][1].round(4))
# gamma GLM vs log-OLS with ch06 numbers (group means only)
R["gamma6"] = dict(arith=251.1 / 327.1, geo=165.5 / 211.9)
out("ch06 means ratio arith", round(251.1 / 327.1, 3), "geo", round(165.5 / 211.9, 3))
# Poisson rate example: ED visits 120 / 400 PY vs 90 / 450 PY
irr = (120 / 400) / (90 / 450); se_irr = np.sqrt(1 / 120 + 1 / 90)
R["irr"] = dict(r1=120 / 400, r0=90 / 450, irr=irr, se=se_irr, lo=irr * np.exp(-Z * se_irr), hi=irr * np.exp(Z * se_irr))
out("IRR", R["irr"])

# ------------------------------------------------ 마. likelihood
x, nn = 7, 50
pg = np.linspace(0.001, 0.45, 2000)
ll = x * np.log(pg) + (nn - x) * np.log(1 - pg)
llmax = x * np.log(0.14) + (nn - x) * np.log(0.86)
R["lik"] = dict(llmax=llmax, Lmax=np.exp(llmax))
for pv in (0.05, 0.10, 0.14, 0.20, 0.30):
    l_ = x * np.log(pv) + (nn - x) * np.log(1 - pv)
    out("lik p", pv, "L", "%.4e" % np.exp(l_), "lnL", round(l_, 3), "rel", round(np.exp(l_ - llmax), 4))
f = lambda pv: x * np.log(pv) + (nn - x) * np.log(1 - pv) - llmax + st.chi2.ppf(0.95, 1) / 2
lr_lo, lr_hi = brentq(f, 0.01, 0.14), brentq(f, 0.14, 0.5)
R["lik"].update(lr_lo=lr_lo, lr_hi=lr_hi, drop=st.chi2.ppf(0.95, 1) / 2)
out("LR interval", lr_lo, lr_hi, "drop", st.chi2.ppf(0.95, 1) / 2)
p0 = 0.05
ll0 = x * np.log(p0) + (nn - x) * np.log(1 - p0)
lr = 2 * (llmax - ll0)
wz = (0.14 - p0) / np.sqrt(0.14 * 0.86 / nn)
sz = (0.14 - p0) / np.sqrt(p0 * (1 - p0) / nn)
exact2 = st.binomtest(x, nn, p0).pvalue
exact1 = st.binom.sf(x - 1, nn, p0)
R["tests"] = dict(ll0=ll0, lr=lr, p_lr=st.chi2.sf(lr, 1), wz=wz, p_w=2 * st.norm.sf(wz), sz=sz, p_s=2 * st.norm.sf(sz),
                  exact2=exact2, exact1=exact1, score_slope=x / p0 - (nn - x) / (1 - p0), info0=nn / (p0 * (1 - p0)),
                  info_hat=nn / (0.14 * 0.86))
out("tests at p0=0.05", R["tests"])
# with n = 500 (70 events) the three agree
x2, n2 = 70, 500
ll_h = x2 * np.log(.14) + (n2 - x2) * np.log(.86); ll_0 = x2 * np.log(p0) + (n2 - x2) * np.log(1 - p0)
R["tests500"] = dict(lr=2 * (ll_h - ll_0), w=((0.14 - p0) / np.sqrt(.14 * .86 / n2)) ** 2, s=((0.14 - p0) / np.sqrt(p0 * (1 - p0) / n2)) ** 2)
out("n=500 chi2 LR, Wald, score", R["tests500"])
# p0 = 0.10 as well (closer)
for p0b in (0.10,):
    ll0b = x * np.log(p0b) + (nn - x) * np.log(1 - p0b)
    out("p0", p0b, "LR", 2 * (llmax - ll0b), "wald z2", ((0.14 - p0b) / np.sqrt(.14 * .86 / nn)) ** 2, "score z2", ((0.14 - p0b) / np.sqrt(p0b * (1 - p0b) / nn)) ** 2)

# nested models in the cohort
m_nopoly = logistic(D, ["age", "female", "cci_d", "medaid"])
lrt = 2 * (full["ll"] - m_nopoly["ll"])
R["lrt_poly"] = dict(m2ll_0=-2 * m_nopoly["ll"], m2ll_1=-2 * full["ll"], chi=lrt, p=st.chi2.sf(lrt, 1),
                     wald=R["full_rows"][1]["z"] ** 2, p_w=R["full_rows"][1]["p"],
                     aic0=-2 * m_nopoly["ll"] + 2 * len(m_nopoly["beta"]), aic1=-2 * full["ll"] + 2 * len(full["beta"]),
                     bic0=-2 * m_nopoly["ll"] + np.log(n) * len(m_nopoly["beta"]), bic1=-2 * full["ll"] + np.log(n) * len(full["beta"]))
out("LRT poly", R["lrt_poly"])


# profile-likelihood CI for poly
def prof(bp):
    Xo, _ = design(D, ["age", "female", "cci_d", "medaid"])
    f_ = glm(Xo, y, "binomial", "logit", offset=bp * D["poly"])
    return -2 * loglik(f_)


m2 = -2 * full["ll"]
crit = st.chi2.ppf(0.95, 1)
bp_hat = full["beta"][1]
pl_lo = brentq(lambda v: prof(v) - m2 - crit, bp_hat - 1, bp_hat)
pl_hi = brentq(lambda v: prof(v) - m2 - crit, bp_hat, bp_hat + 1)
R["profile"] = dict(lo=np.exp(pl_lo), hi=np.exp(pl_hi), wlo=np.exp(bp_hat - Z * full["se"][1]), whi=np.exp(bp_hat + Z * full["se"][1]))
out("profile CI poly", R["profile"])

# global tests (SAS 'Testing Global Null Hypothesis: BETA=0')
Xf = full["X"]
mu0 = np.full(n, y.mean())
U = Xf.T @ (y - mu0)
I0 = (Xf.T * (mu0 * (1 - mu0))) @ Xf
score_g = U @ np.linalg.solve(I0, U)
bb = full["beta"][1:]
Vb = full["cov"][1:, 1:]
wald_g = bb @ np.linalg.solve(Vb, bb)
lr_g = 2 * (full["ll"] - null["ll"])
kf = len(full["beta"])
R["global"] = dict(lr=lr_g, score=score_g, wald=wald_g, df=kf - 1, p_lr=st.chi2.sf(lr_g, kf - 1),
                   p_s=st.chi2.sf(score_g, kf - 1), p_w=st.chi2.sf(wald_g, kf - 1),
                   aic_i=-2 * null["ll"] + 2, aic_f=-2 * full["ll"] + 2 * kf, sc_i=-2 * null["ll"] + np.log(n),
                   sc_f=-2 * full["ll"] + np.log(n) * kf, m2_i=-2 * null["ll"], m2_f=-2 * full["ll"])
out("global", R["global"])
# CCI: dummies vs linear score (nested) and age linear vs groups (non-nested)
m_lin = logistic(D, ["poly", "age", "female", "cci_lin", "medaid"])
lr_lin = 2 * (full["ll"] - m_lin["ll"])
R["cci_lin"] = dict(m=m_lin, rows=wald_rows(m_lin, m_lin["names"]), lr=lr_lin, p=st.chi2.sf(lr_lin, 2),
                    aic_lin=-2 * m_lin["ll"] + 2 * len(m_lin["beta"]), aic_d=-2 * full["ll"] + 2 * kf,
                    bic_lin=-2 * m_lin["ll"] + np.log(n) * len(m_lin["beta"]), bic_d=-2 * full["ll"] + np.log(n) * kf)
out("CCI linear vs dummies", {k: v for k, v in R["cci_lin"].items() if k not in ("m", "rows")})
for r in R["cci_lin"]["rows"]:
    out("   lin", r["name"], round(r["b"], 4), round(r["se"], 4), "%.3g" % r["p"], round(np.exp(r["b"]), 3), round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3))
m_ag = logistic(D, ["poly", "agegrp", "female", "cci_d", "medaid"])
R["agegrp"] = dict(m=m_ag, rows=wald_rows(m_ag, m_ag["names"]), aic=-2 * m_ag["ll"] + 2 * len(m_ag["beta"]),
                   bic=-2 * m_ag["ll"] + np.log(n) * len(m_ag["beta"]), m2=-2 * m_ag["ll"])
out("age groups model -2LL", -2 * m_ag["ll"], "AIC", R["agegrp"]["aic"], "BIC", R["agegrp"]["bic"], "vs linear AIC", -2 * full["ll"] + 2 * kf, "BIC", -2 * full["ll"] + np.log(n) * kf)
out("age group sizes", [int(((D['age'] >= a) & (D['age'] < b_)).sum()) for a, b_ in ((65, 75), (75, 85), (85, 120))])

# identity-binomial: converges to the boundary (a fitted risk hits 0)
idb = glm(X, y, "binomial", "identity", start=np.r_[y.mean(), np.zeros(X.shape[1] - 1)])
R["idbin"] = dict(b=idb["beta"][1], mumin=idb["mu"].min(), iters=idb["iters"])
out("identity-binomial at boundary", R["idbin"])

# =================================================================== 바. confounding & interaction
# confounding by indication: drug A (new) vs drug B, severe vs mild
S = {"severe": dict(a=288, n1=900, c=120, n0=300), "mild": dict(a=24, n1=300, c=90, n0=900)}
R["conf"] = {}
for k, v in S.items():
    a, n1_, c, n0_ = v["a"], v["n1"], v["c"], v["n0"]
    R["conf"][k] = dict(r1=a / n1_, r0=c / n0_, rr=rr_ci(a, n1_, c, n0_), orr=or_ci(a, n1_, c, n0_), rd=rd_ci(a, n1_, c, n0_), **v)
A1 = sum(v["a"] for v in S.values()); N1 = sum(v["n1"] for v in S.values())
C0 = sum(v["c"] for v in S.values()); N0 = sum(v["n0"] for v in S.values())
R["conf"]["crude"] = dict(a=A1, n1=N1, c=C0, n0=N0, r1=A1 / N1, r0=C0 / N0, rr=rr_ci(A1, N1, C0, N0), orr=or_ci(A1, N1, C0, N0), rd=rd_ci(A1, N1, C0, N0))
# Mantel-Haenszel RR with Greenland-Robins variance, MH OR, MH RD
num = den = 0; vnum = 0; orn = ord_ = 0; rdn = 0; wsum = 0
for v in S.values():
    a, n1_, c, n0_ = v["a"], v["n1"], v["c"], v["n0"]
    N = n1_ + n0_; b, d_ = n1_ - a, n0_ - c; M1 = a + c
    num += a * n0_ / N; den += c * n1_ / N
    vnum += (M1 * n1_ * n0_ - a * c * N) / N ** 2
    orn += a * d_ / N; ord_ += b * c / N
    w = n1_ * n0_ / N; rdn += w * (a / n1_ - c / n0_); wsum += w
rr_mh = num / den; se_mh = np.sqrt(vnum / (num * den))
R["mh"] = dict(rr=rr_mh, lo=rr_mh * np.exp(-Z * se_mh), hi=rr_mh * np.exp(Z * se_mh), se=se_mh, num=num, den=den,
               orr=orn / ord_, rd=rdn / wsum)
out("confounding strata", {k: (round(v["r1"], 3), round(v["r0"], 3), [round(x, 3) for x in v["rr"][:3]], [round(x, 3) for x in v["orr"][:3]], [round(x * 100, 2) for x in v["rd"][:3]]) for k, v in R["conf"].items()})
out("MH", R["mh"])
# distribution of severity by drug
out("severe share A", 900 / 1200, "B", 300 / 1200, "risk severe/mild overall", (288 + 120) / 1200, (24 + 90) / 1200)
# additive-scale difference test
v1 = R["conf"]["severe"]["rd"]; v2 = R["conf"]["mild"]["rd"]
zrd = (v1[0] - v2[0]) / np.sqrt(v1[3] ** 2 + v2[3] ** 2)
R["rd_int"] = dict(diff=v1[0] - v2[0], z=zrd, p=2 * st.norm.sf(abs(zrd)))
out("RD difference", R["rd_int"])
# joint effects with common reference (mild, drug B) as in Knol & VanderWeele
r00 = 90 / 900; r10 = 24 / 300; r01 = 120 / 300; r11 = 288 / 900
R["joint_rr"] = dict(A_mild=r10 / r00, B_sev=r01 / r00, A_sev=r11 / r00, reri=r11 / r00 - r10 / r00 - r01 / r00 + 1,
                     mult=(r11 / r00) / ((r10 / r00) * (r01 / r00)))
out("joint RR", R["joint_rr"])
# Berkson / collider example
Ep, Yp, N = 0.20, 0.10, 10000
cells = {}
for e in (0, 1):
    for yy in (0, 1):
        cnt = N * (Ep if e else 1 - Ep) * (Yp if yy else 1 - Yp)
        ph = 0.05 + 0.30 * e + 0.40 * yy
        cells[(e, yy)] = (cnt, cnt * ph)
or_pop = (cells[(1, 1)][0] * cells[(0, 0)][0]) / (cells[(1, 0)][0] * cells[(0, 1)][0])
or_h = (cells[(1, 1)][1] * cells[(0, 0)][1]) / (cells[(1, 0)][1] * cells[(0, 1)][1])
R["berkson"] = dict(cells=cells, or_pop=or_pop, or_h=or_h,
                    y_given_e1=cells[(1, 1)][1] / (cells[(1, 1)][1] + cells[(1, 0)][1]),
                    y_given_e0=cells[(0, 1)][1] / (cells[(0, 1)][1] + cells[(0, 0)][1]))
out("Berkson", R["berkson"])

# subgroup forest (RCT drug X vs placebo, 1-year hospitalization)
SG = [("Age", "&lt;65 years", (62, 480, 88, 470)), ("Age", "≥65 years", (88, 520, 112, 530)),
      ("Sex", "Male", (84, 560, 122, 570)), ("Sex", "Female", (66, 440, 78, 430)),
      ("eGFR", "≥60 mL/min/1.73 m²", (80, 690, 138, 700)), ("eGFR", "&lt;60 mL/min/1.73 m²", (70, 310, 62, 300)),
      ("Diabetes", "No", (92, 610, 121, 600)), ("Diabetes", "Yes", (58, 390, 79, 400))]
tot = (150, 1000, 200, 1000)
R["sg_overall"] = dict(rr=rr_ci(*tot), p=st.chi2_contingency(np.array([[150, 850], [200, 800]]), correction=False)[1])
R["sg"] = []
for i in range(0, len(SG), 2):
    (g, l1, t1), (_, l2, t2) = SG[i], SG[i + 1]
    r1_ = rr_ci(*t1); r2_ = rr_ci(*t2)
    zi = (np.log(r1_[0]) - np.log(r2_[0])) / np.sqrt(r1_[3] ** 2 + r2_[3] ** 2)
    p1s = st.chi2_contingency(np.array([[t1[0], t1[1] - t1[0]], [t1[2], t1[3] - t1[2]]]), correction=False)[1]
    p2s = st.chi2_contingency(np.array([[t2[0], t2[1] - t2[0]], [t2[2], t2[3] - t2[2]]]), correction=False)[1]
    R["sg"].append(dict(g=g, l1=l1, t1=t1, rr1=r1_, p1=p1s, l2=l2, t2=t2, rr2=r2_, p2=p2s, pint=2 * st.norm.sf(abs(zi)),
                        ratio=r1_[0] / r2_[0], zint=zi))
    # check sums
    assert t1[0] + t2[0] == 150 and t1[1] + t2[1] == 1000 and t1[2] + t2[2] == 200 and t1[3] + t2[3] == 1000, g
out("overall RR", [round(v, 3) for v in R["sg_overall"]["rr"][:3]], "P", R["sg_overall"]["p"])
for sgr in R["sg"]:
    out("SG", sgr["g"], sgr["l1"], [round(v, 2) for v in sgr["rr1"][:3]], round(sgr["p1"], 4), "|", sgr["l2"], [round(v, 2) for v in sgr["rr2"][:3]], round(sgr["p2"], 4), "Pint", round(sgr["pint"], 3), "ratio", round(sgr["ratio"], 3))

# =================================================================== 사. dummy variables
R["dum"] = {}
for key, terms in (("ref0", FULL), ("ref1", ["poly", "age", "female", "cci_d_ref1", "medaid"]), ("ref3", ["poly", "age", "female", "cci_d_ref3", "medaid"])):
    m_ = logistic(D, terms)
    R["dum"][key] = dict(m=m_, rows=wald_rows(m_, m_["names"]), m2=-2 * m_["ll"])
    out("dummy", key, "-2LL", round(-2 * m_["ll"], 4), [(r["name"], round(r["b"], 4), round(r["se"], 4), round(np.exp(r["b"]), 3), round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3), "%.3g" % r["p"]) for r in R["dum"][key]["rows"] if r["name"].startswith("cci")])
    out("   poly row", [(round(np.exp(r["b"]), 4), round(r["se"], 5)) for r in R["dum"][key]["rows"] if r["name"] == "poly"])
# 3 df test for CCI (LR)
m_nocci = logistic(D, ["poly", "age", "female", "medaid"])
lr_cci = 2 * (full["ll"] - m_nocci["ll"])
R["cci_overall"] = dict(lr=lr_cci, p=st.chi2.sf(lr_cci, 3))
out("CCI overall LR df3", R["cci_overall"])
# the "wrong" coding: medaid coded 1/2 (1 = no, 2 = yes)? -> show nominal codes: institution not in data; use CCI codes 1-4
# crude ORs per CCI group
cc_ = logistic(D, ["cci_d"])
R["cci_crude"] = wald_rows(cc_, cc_["names"])
out("crude CCI", [(r["name"], round(np.exp(r["b"]), 3), round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3)) for r in R["cci_crude"]])

# =================================================================== 아. univariable vs multivariable
R["uni"] = {}
for t in ("poly", "age", "female", "cci_d", "medaid"):
    m_ = logistic(D, [t])
    R["uni"][t] = wald_rows(m_, m_["names"])[1:]
    if t == "cci_d":
        m0_ = logistic(D, [])
        R["uni"]["cci_overall_p"] = st.chi2.sf(2 * (m_["ll"] - m0_["ll"]), 3)
    out("uni", t, [(r["name"], round(np.exp(r["b"]), 3), round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3), "%.3g" % r["p"]) for r in R["uni"][t]])
out("uni cci overall p", R["uni"]["cci_overall_p"])
# screening at P < 0.2 -> drops medaid; model without medaid
scr = logistic(D, ["poly", "age", "female", "cci_d"])
R["screen"] = dict(m=scr, rows=wald_rows(scr, scr["names"]))
out("screened model", [(r["name"], round(np.exp(r["b"]), 3), round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3), "%.3g" % r["p"]) for r in R["screen"]["rows"]])
# backward elimination at 0.05 from screened model
cur = ["poly", "age", "female", "cci_d"]
while True:
    m_ = logistic(D, cur)
    rows = wald_rows(m_, m_["names"])
    # p per term (cci_d by LR)
    pv = {}
    for t in cur:
        if t == "poly":
            continue  # exposure kept
        if t == "cci_d":
            m_red = logistic(D, [u for u in cur if u != t]); pv[t] = st.chi2.sf(2 * (m_["ll"] - m_red["ll"]), 3)
        else:
            pv[t] = [r["p"] for r in rows if r["name"] == t][0]
    worst = max(pv, key=pv.get)
    if pv[worst] < 0.05:
        break
    out("backward: drop", worst, round(pv[worst], 4))
    cur.remove(worst)
bw = logistic(D, cur)
R["backward"] = dict(terms=cur, m=bw, rows=wald_rows(bw, bw["names"]))
out("backward final", cur, [(r["name"], round(np.exp(r["b"]), 3), round(np.exp(r["lo"]), 3), round(np.exp(r["hi"]), 3), "%.3g" % r["p"]) for r in R["backward"]["rows"]])
# change in poly OR when each covariate is dropped from the full model
for t in ("age", "female", "cci_d", "medaid"):
    m_ = logistic(D, [u for u in FULL if u != t])
    out("drop", t, "poly OR", round(np.exp(m_["beta"][1]), 3), "change %", round((np.exp(m_["beta"][1]) / np.exp(full["beta"][1]) - 1) * 100, 1))
R["epv"] = dict(events=tab["ev"], params=len(full["beta"]) - 1, epv=tab["ev"] / (len(full["beta"]) - 1))
out("EPV", R["epv"])
# medaid conditional associations
out("medaid & poly crude OR", or_ci(int(((D['medaid'] == 1) & (D['poly'] == 1)).sum()), int(D['medaid'].sum()), int(((D['medaid'] == 0) & (D['poly'] == 1)).sum()), int((D['medaid'] == 0).sum()))[:3])


# =================================================================== additions (second pass)
# --- 가: quantiles used by the widget (df = n - 1 for n = 10, 25, 100)
R["wq"] = {n_: [float(st.t.ppf(q, n_ - 1)) for q in (0.95, 0.975, 0.995)] for n_ in (10, 25, 100)}
R["wq"]["z"] = [float(st.norm.ppf(q)) for q in (0.95, 0.975, 0.995)]
out("widget quantiles", {k: [round(v, 4) for v in vv] for k, vv in R["wq"].items()})
# coverage of z-interval (1.96) with n = 10, 25 for the text
for nn_ in (10, 25):
    rng = np.random.default_rng(7)
    xs_ = rng.normal(TRUE_MU, SIG, (200000, nn_))
    mm_, ss_ = xs_.mean(1), xs_.std(1, ddof=1)
    out("coverage n", nn_, "t", np.mean(np.abs(mm_ - TRUE_MU) <= st.t.ppf(0.975, nn_ - 1) * ss_ / np.sqrt(nn_)),
        "z", np.mean(np.abs(mm_ - TRUE_MU) <= Z * ss_ / np.sqrt(nn_)))
# exact Wald p for RR of chapter 5 (unrounded)
R["rr5"]["z"] = R["rr5"]["ln"] / R["rr5"]["se"]
R["rr5"]["p"] = 2 * st.norm.sf(abs(R["rr5"]["z"]))
out("RR ch05 Wald z", R["rr5"]["z"], "p", R["rr5"]["p"])
# rct_goal: RR and RD CIs (percent)
out("rct goal RR", [round(v, 3) for v in R["rct_goal"]["rr"][:3]], "RD %", [round(v * 100, 1) for v in R["rct_goal"]["rd"][:3]])
# CI half width in the mean example, t crit
out("mean example t(24)", R["mean"]["tc"], "half", R["mean"]["half"])

# --- 나: relative vs absolute (RRR 50%)
R["abs"] = []
for p0_, p1_ in ((0.02, 0.01), (0.40, 0.20)):
    R["abs"].append(dict(p0=p0_, p1=p1_, rr=p1_ / p0_, rrr=1 - p1_ / p0_, arr=p0_ - p1_, nnt=1 / (p0_ - p1_),
                         orv=(p1_ / (1 - p1_)) / (p0_ / (1 - p0_))))
out("abs vs rel", R["abs"])
# RCT: 2x2 cells and odds for the readmission row
a_, n1_, c_, n0_ = RCT["readm"]
R["rct_cells"] = dict(a=a_, b=n1_ - a_, c=c_, d=n0_ - c_, odds1=a_ / (n1_ - a_), odds0=c_ / (n0_ - c_))
out("rct cells", R["rct_cells"])
out("rct readm RR se", R["rct"]["readm"]["rr"][3], "OR se", R["rct"]["readm"]["orr"][3], "RD se", R["rct"]["readm"]["rd"][3])

# case-control box: benzodiazepine exposure categories (nested case-control, 1,200 cases, 4,800 controls)
CCAT = [("Current use (≤30 days)", 180, 520), ("Recent use (31–180 days)", 60, 260), ("Past use (&gt;180 days)", 90, 400),
        ("Nonuse", 870, 3620)]
assert sum(x[1] for x in CCAT) == 1200 and sum(x[2] for x in CCAT) == 4800
R["ccat"] = []
ca_ref, co_ref = CCAT[-1][1], CCAT[-1][2]
for lab, ca, co in CCAT:
    if ca == ca_ref:
        R["ccat"].append(dict(lab=lab, ca=ca, co=co, pca=ca / 1200, pco=co / 4800, ref=True))
        continue
    o = (ca * co_ref) / (co * ca_ref)
    s = np.sqrt(1 / ca + 1 / co + 1 / ca_ref + 1 / co_ref)
    R["ccat"].append(dict(lab=lab, ca=ca, co=co, pca=ca / 1200, pco=co / 4800, orv=o, lo=o * np.exp(-Z * s), hi=o * np.exp(Z * s), se=s))
for r_ in R["ccat"]:
    out("ccat", r_["lab"], r_["ca"], round(r_["pca"] * 100, 1), r_["co"], round(r_["pco"] * 100, 1),
        "" if r_.get("ref") else [round(r_["orv"], 2), round(r_["lo"], 2), round(r_["hi"], 2)])
# the 'naive risk' one would (wrongly) compute from the case-control table
R["cc_naive"] = dict(r_cur=180 / (180 + 520), r_non=870 / (870 + 3620))
out("case-control naive 'risk'", R["cc_naive"])

# common outcome in the discharge cohort: Zhang-Yu on the adjusted OR, marginal standardisation
p0u = R["crude"]["r0"]
zy = lambda o: o / (1 - p0u + p0u * o)
fr = R["full_rows"][1]
R["zy"] = dict(p0=p0u, rr=zy(np.exp(fr["b"])), lo=zy(np.exp(fr["lo"])), hi=zy(np.exp(fr["hi"])),
               crude=zy(R["crude"]["orr"][0]))
out("Zhang-Yu on adjusted OR", R["zy"])


def standardised(Xm, beta):
    X1 = Xm.copy(); X1[:, 1] = 1
    X0 = Xm.copy(); X0[:, 1] = 0
    r1 = np.mean(1 / (1 + np.exp(-(X1 @ beta))))
    r0 = np.mean(1 / (1 + np.exp(-(X0 @ beta))))
    return r1, r0


Xf_ = full["X"]
r1s, r0s = standardised(Xf_, full["beta"])
rng = np.random.default_rng(2024)
bs = []
for _ in range(1000):
    idx = rng.integers(0, n, n)
    fb = glm(Xf_[idx], y[idx], "binomial", "logit", start=full["beta"])
    a1, a0 = standardised(Xf_[idx], fb["beta"])
    bs.append((a1, a0, a1 / a0, a1 - a0))
bs = np.array(bs)
q = lambda j: np.percentile(bs[:, j], [2.5, 97.5])
R["std"] = dict(r1=r1s, r0=r0s, rr=r1s / r0s, rd=r1s - r0s, rr_ci=q(2), rd_ci=q(3), r1_ci=q(0), r0_ci=q(1))
out("standardised risks", {k: np.round(v, 4) for k, v in R["std"].items()})

# --- 다: exp examples
R["ch01ex"] = dict(orv=np.exp(0.642), lo=np.exp(0.642 - 1.96 * 0.211), hi=np.exp(0.642 + 1.96 * 0.211))
out("ch01 example", R["ch01ex"])
R["pct_poly"] = (np.exp(full["beta"][1]) - 1) * 100
out("poly odds % higher", R["pct_poly"])
# a log-transformed exposure: ln(CRP) in a logistic model, beta 0.38, SE 0.09 (hypothetical)
bC, sC = 0.38, 0.09
R["crp"] = {}
for lab, k in (("e", 1.0), ("x2", np.log(2)), ("x10", np.log(10))):
    R["crp"][lab] = (np.exp(bC * k), np.exp((bC - Z * sC) * k), np.exp((bC + Z * sC) * k))
R["crp"]["b"] = bC; R["crp"]["se"] = sC
out("CRP scaling", R["crp"])
# per-SD scaling for age (SD from the cohort)
BF = dict(zip(full["names"], full["beta"])); SF = dict(zip(full["names"], full["se"]))
sd_age = D["age"].std(ddof=1)
R["age_sd"] = dict(sd=sd_age, orv=np.exp(BF["age"] * sd_age), lo=np.exp((BF["age"] - Z * SF["age"]) * sd_age), hi=np.exp((BF["age"] + Z * SF["age"]) * sd_age))
out("age per SD", R["age_sd"])
# log10 vs ln of the same coefficient
R["log10"] = dict(ln2=np.log(2), log10_2=np.log10(2), ln10=np.log(10))
out("logs", R["log10"])
# multiplicative accumulation: age + 20 years
R["age20"] = np.exp(20 * BF["age"])
out("age 20 years OR", R["age20"], "vs 1.062^20", np.exp(BF["age"]) ** 20)
out("pred risk ratio from OR (80M cci3-4)", R["pred"][1]["risk"] / R["pred"][0]["risk"], "odds ratio", R["pred"][1]["odds"] / R["pred"][0]["odds"])

# --- 라: healthcare cost GLM (the 400-patient data set of chapter 6)
import nums_ch06 as N6
C6 = N6.cost400()
Xc6 = np.column_stack([np.ones(C6["n"]), C6["adh"], C6["age"] / 10, C6["cci"]])
yc6 = C6["cost"]
gfit = glm(Xc6, yc6, "gamma", "log", start=np.r_[np.log(yc6.mean()), 0, 0, 0])
gr = wald_rows(gfit, ["(Intercept)", "adh", "age10", "cci"])


def recycle(Xm, beta):
    X1 = Xm.copy(); X1[:, 1] = 1
    X0 = Xm.copy(); X0[:, 1] = 0
    return np.mean(np.exp(X1 @ beta)), np.mean(np.exp(X0 @ beta))


m1g, m0g = recycle(Xc6, gfit["beta"])
# OLS on raw cost (robust SE), and log-OLS with smearing
bo = np.linalg.lstsq(Xc6, yc6, rcond=None)[0]
ro = yc6 - Xc6 @ bo
Bo = np.linalg.inv(Xc6.T @ Xc6)
so = np.sqrt(np.diag(Bo @ (Xc6.T * ro ** 2) @ Xc6 @ Bo))
bl = np.linalg.lstsq(Xc6, np.log(yc6), rcond=None)[0]
rl = np.log(yc6) - Xc6 @ bl
smear = np.mean(np.exp(rl))
m1l, m0l = recycle(Xc6, bl)
rng = np.random.default_rng(606)
bsg = []
for _ in range(1000):
    idx = rng.integers(0, C6["n"], C6["n"])
    fb = glm(Xc6[idx], yc6[idx], "gamma", "log", start=gfit["beta"])
    a1, a0 = recycle(Xc6[idx], fb["beta"])
    bsg.append((a1, a0, a1 - a0))
bsg = np.array(bsg)
R["cost"] = dict(n=C6["n"], n_adh=C6["n_adh"], am=C6["am"], gm=C6["gm"], smear=smear, rows=gr, disp=gfit["disp"],
                 m1=m1g, m0=m0g, diff=m1g - m0g, diff_ci=np.percentile(bsg[:, 2], [2.5, 97.5]),
                 m1_ci=np.percentile(bsg[:, 0], [2.5, 97.5]), m0_ci=np.percentile(bsg[:, 1], [2.5, 97.5]),
                 ols=(bo[1], bo[1] - Z * so[1], bo[1] + Z * so[1]),
                 logols=(np.exp(bl[1]), m1l, m0l, m1l * smear, m0l * smear),
                 crude_ratio=C6["am"][0] / C6["am"][1], crude_diff=C6["am"][0] - C6["am"][1],
                 mean_all=yc6.mean(), iters=gfit["iters"])
# unadjusted gamma GLM reproduces the ratio of arithmetic means exactly
g1 = glm(Xc6[:, :2], yc6, "gamma", "log", start=[np.log(yc6.mean()), 0])
R["cost"]["unadj_exp"] = np.exp(g1["beta"][1])
out("cost: arith", np.round(C6["am"], 2), "geo", np.round(C6["gm"], 2), "crude ratio", round(C6["am"][0] / C6["am"][1], 4),
    "unadj gamma exp", round(R["cost"]["unadj_exp"], 4))
for r_ in gr:
    out("  gamma", r_["name"], round(r_["b"], 4), round(r_["se"], 4), "exp", round(np.exp(r_["b"]), 3), round(np.exp(r_["lo"]), 3), round(np.exp(r_["hi"]), 3), "%.3g" % r_["p"])
out("  dispersion", gfit["disp"], "shape", 1 / gfit["disp"], "iters", gfit["iters"])
out("  recycled means", round(m1g, 1), round(m0g, 1), "diff", round(m1g - m0g, 1), "CI", np.round(R["cost"]["diff_ci"], 1),
    "m1 CI", np.round(R["cost"]["m1_ci"], 1), "m0 CI", np.round(R["cost"]["m0_ci"], 1))
out("  OLS raw adh", np.round(R["cost"]["ols"], 2))
out("  log-OLS GMR", round(np.exp(bl[1]), 4), "retransformed means (no smear)", round(m1l, 1), round(m0l, 1), "smear", round(smear, 4),
    "smeared", round(m1l * smear, 1), round(m0l * smear, 1), "overall mean cost", round(yc6.mean(), 1))

# --- 라: Poisson regression with offset reproduces the crude IRR
Xp = np.array([[1, 1], [1, 0]], float); yp = np.array([120, 90], float); off = np.log([400, 450])
pf = glm(Xp, yp, "poisson", "log", offset=off)
out("Poisson offset IRR", np.exp(pf["beta"][1]), "se", pf["se"][1], "intercept rate", np.exp(pf["beta"][0]))

# --- 마: extra tests
R["tests10"] = {}
p0b = 0.10
ll0b = x * np.log(p0b) + (nn - x) * np.log(1 - p0b)
lrb = 2 * (llmax - ll0b)
wb = ((0.14 - p0b) / np.sqrt(.14 * .86 / nn)) ** 2
sb = ((0.14 - p0b) / np.sqrt(p0b * (1 - p0b) / nn)) ** 2
R["tests10"] = dict(lr=lrb, w=wb, s=sb, p_lr=st.chi2.sf(lrb, 1), p_w=st.chi2.sf(wb, 1), p_s=st.chi2.sf(sb, 1),
                    exact=st.binomtest(x, nn, p0b).pvalue)
out("tests p0=0.10", R["tests10"])
# LR test poly with deviances
out("dev null", null["dev"], "dev full", full["dev"], "dev no-poly", m_nopoly["dev"])

# --- 바: OR homogeneity in severe/mild, MH OR by CCI in the cohort
o1, o2 = R["conf"]["severe"]["orr"], R["conf"]["mild"]["orr"]
zo = (np.log(o1[0]) - np.log(o2[0])) / np.sqrt(o1[3] ** 2 + o2[3] ** 2)
R["or_hom"] = dict(z=zo, p=2 * st.norm.sf(abs(zo)))
out("OR homogeneity severe vs mild", R["or_hom"])
# MH OR (with RGB variance) for poly, stratified by CCI (4 strata)


def mh_or(strata):
    R_ = S_ = 0.0; PR = PS_QR = QS = 0.0
    for a, b_, c, d in strata:
        N_ = a + b_ + c + d
        Ri, Si = a * d / N_, b_ * c / N_
        Pi, Qi = (a + d) / N_, (b_ + c) / N_
        R_ += Ri; S_ += Si
        PR += Pi * Ri; PS_QR += Pi * Si + Qi * Ri; QS += Qi * Si
    o = R_ / S_
    v = PR / (2 * R_ ** 2) + PS_QR / (2 * R_ * S_) + QS / (2 * S_ ** 2)
    return o, o * np.exp(-Z * np.sqrt(v)), o * np.exp(Z * np.sqrt(v))


str_cci = []
for k in range(4):
    msk = D["cci"] == k
    pe, ye = D["poly"][msk], D["readm"][msk]
    a = int(((pe == 1) & (ye == 1)).sum()); b_ = int(((pe == 1) & (ye == 0)).sum())
    c = int(((pe == 0) & (ye == 1)).sum()); d_ = int(((pe == 0) & (ye == 0)).sum())
    str_cci.append((a, b_, c, d_))
    out("cci stratum", k, (a, b_, c, d_), "OR", round(a * d_ / (b_ * c), 3))
R["mh_cci"] = mh_or(str_cci)
R["str_cci"] = str_cci
m_pc = logistic(D, ["poly", "cci_d"])
R["logit_pc"] = wald_rows(m_pc, m_pc["names"])[1]
out("MH OR by CCI", np.round(R["mh_cci"], 3), "logistic poly+cci", round(np.exp(R["logit_pc"]["b"]), 3), round(np.exp(R["logit_pc"]["lo"]), 3), round(np.exp(R["logit_pc"]["hi"]), 3))
# stratum-specific ORs by CCI with CIs
R["str_cci_or"] = []
for a, b_, c, d_ in str_cci:
    o = a * d_ / (b_ * c); s = np.sqrt(1 / a + 1 / b_ + 1 / c + 1 / d_)
    R["str_cci_or"].append((o, o * np.exp(-Z * s), o * np.exp(Z * s)))
out("stratum ORs", [tuple(round(v, 2) for v in t) for t in R["str_cci_or"]])
# Woolf test of homogeneity across CCI strata, and LR test for poly x CCI product terms
lw = np.array([np.log(a * d_ / (b_ * c)) for a, b_, c, d_ in str_cci])
ww = np.array([1 / (1 / a + 1 / b_ + 1 / c + 1 / d_) for a, b_, c, d_ in str_cci])
lbar = np.sum(ww * lw) / ww.sum()
woolf = float(np.sum(ww * (lw - lbar) ** 2))
Xi, _ = design(D, FULL)
cc3 = Xi[:, [full["names"].index(k) for k in ("cci1-2", "cci3-4", "cci5+")]]
Xint = np.column_stack([Xi, cc3 * D["poly"][:, None]])
fint = glm(Xint, y, "binomial", "logit")
lr_int = 2 * (loglik(fint) - full["ll"])
R["hom_cci"] = dict(woolf=woolf, p_woolf=st.chi2.sf(woolf, 3), lr_int=lr_int, p_int=st.chi2.sf(lr_int, 3))
out("homogeneity CCI strata", R["hom_cci"])
# Berkson in words
out("Berkson hospital: P(Y|E=1), P(Y|E=0)", R["berkson"]["y_given_e1"], R["berkson"]["y_given_e0"])

# CMH chi-square (no continuity correction) and MH OR with RGB CI for the severity example and the CCI strata


def cmh(strata):
    num = 0.0; var = 0.0
    for a, b_, c, d_ in strata:
        N_ = a + b_ + c + d_; n1s = a + b_; n0s = c + d_; m1 = a + c; m0 = b_ + d_
        num += a - n1s * m1 / N_
        var += n1s * n0s * m1 * m0 / (N_ ** 2 * (N_ - 1))
    x2 = num ** 2 / var
    x2c = (abs(num) - 0.5) ** 2 / var
    return x2, st.chi2.sf(x2, 1), x2c, st.chi2.sf(x2c, 1)


sev_str = [(288, 612, 120, 180), (24, 276, 90, 810)]
R["mh_sev_or"] = mh_or(sev_str)
R["cmh_sev"] = cmh(sev_str)
R["cmh_cci"] = cmh(str_cci)
out("MH OR severity (RGB CI)", np.round(R["mh_sev_or"], 4), "CMH chi2 severity", np.round(R["cmh_sev"], 5), "CMH CCI", np.round(R["cmh_cci"], 5))
out("SG exact", [(s_["g"], s_["pint"], s_["p1"], s_["p2"]) for s_ in R["sg"]])

# --- 사: Wald 3-df test for CCI, effect coding, age groups
ic = [full["names"].index(k) for k in ("cci1-2", "cci3-4", "cci5+")]
bc = full["beta"][ic]; Vc = full["cov"][np.ix_(ic, ic)]
wc = float(bc @ np.linalg.solve(Vc, bc))
R["cci_wald"] = dict(w=wc, p=st.chi2.sf(wc, 3))
out("CCI Wald 3df", R["cci_wald"])
lo_ = np.r_[0.0, bc]
R["effcode"] = dict(mean=lo_.mean(), eff=lo_ - lo_.mean())
out("effect coding", R["effcode"])
for r_ in R["agegrp"]["rows"]:
    if r_["name"].startswith("age"):
        out("agegrp", r_["name"], round(np.exp(r_["b"]), 3), round(np.exp(r_["lo"]), 3), round(np.exp(r_["hi"]), 3), "%.3g" % r_["p"])
m_ag0 = logistic(D, ["poly", "female", "cci_d", "medaid"])
lr_ag = 2 * (R["agegrp"]["m"]["ll"] - m_ag0["ll"])
R["agegrp"]["lr"] = lr_ag; R["agegrp"]["p_overall"] = st.chi2.sf(lr_ag, 2)
out("agegrp overall LR", lr_ag, st.chi2.sf(lr_ag, 2))
# trend P for CCI (score 0-3) in the full-adjusted model = cci_lin row
out("P for trend (cci score)", [r_["p"] for r_ in R["cci_lin"]["rows"] if r_["name"] == "cci_score"])
# group sizes of CCI with events
R["cci_n"] = [(int((D["cci"] == k).sum()), int(D["readm"][D["cci"] == k].sum())) for k in range(4)]

# --- 아: per-10-year age ORs, model sequence
R["uni_age10"] = {}
m_ = logistic(D, ["age10"]); r_ = wald_rows(m_, m_["names"])[1]
R["uni_age10"] = (np.exp(r_["b"]), np.exp(r_["lo"]), np.exp(r_["hi"]), r_["p"])
m_ = logistic(D, ["poly", "age10", "female", "cci_d", "medaid"]); rows10 = wald_rows(m_, m_["names"])
R["multi_age10"] = [(np.exp(r_["b"]), np.exp(r_["lo"]), np.exp(r_["hi"]), r_["p"]) for r_ in rows10 if r_["name"] == "age10"][0]
out("age per 10: uni", np.round(R["uni_age10"], 4), "multi", np.round(R["multi_age10"], 4))
SEQ = [("crude", ["poly"]), ("+age", ["poly", "age"]), ("+age+female", ["poly", "age", "female"]),
       ("+age+female+cci", ["poly", "age", "female", "cci_d"]), ("full", FULL), ("screen", ["poly", "age", "female", "cci_d"]),
       ("backward", ["poly", "age", "cci_d"]), ("+medaid only", ["poly", "medaid"]), ("+cci only", ["poly", "cci_d"])]
R["seq"] = {}
for lab, terms in SEQ:
    m_ = logistic(D, terms); r_ = wald_rows(m_, m_["names"])[1]
    R["seq"][lab] = (np.exp(r_["b"]), np.exp(r_["lo"]), np.exp(r_["hi"]), r_["p"])
    out("seq", lab, np.round(R["seq"][lab], 3))
# P values in univariable models (for the screening story)
out("uni P", {t: [round(r_["p"], 4) for r_ in R["uni"][t]] for t in ("poly", "age", "female", "medaid")})
# medaid vs age / cci
out("medaid: mean age", D["age"][D["medaid"] == 1].mean(), D["age"][D["medaid"] == 0].mean(),
    "cci>=3-4 share", np.mean(D["cci"][D["medaid"] == 1] >= 2), np.mean(D["cci"][D["medaid"] == 0] >= 2),
    "readm", D["readm"][D["medaid"] == 1].mean(), D["readm"][D["medaid"] == 0].mean(), "n medaid", D["medaid"].sum())


# --- third pass: values for tables in paper boxes
# hypoglycaemia row of the RCT box (9/120 vs 11/120): chi-square and Fisher
tb = np.array([[9, 111], [11, 109]])
R["rct_hypo"]["chi_p"] = st.chi2_contingency(tb, correction=False)[1]
R["rct_hypo"]["fisher_p"] = st.fisher_exact(tb)[1]
R["rct_hypo"]["wald_p"] = 2 * st.norm.sf(abs(np.log(R["rct_hypo"]["rr"][0]) / R["rct_hypo"]["rr"][3]))
out("hypo P chi", R["rct_hypo"]["chi_p"], "fisher", R["rct_hypo"]["fisher_p"], "wald", R["rct_hypo"]["wald_p"])
out("goal RR wald p", 2 * st.norm.sf(abs(np.log(R["rct_goal"]["rr"][0]) / R["rct_goal"]["rr"][3])))
# cost: observed SD, median, IQR by group
cst = C6["cost"]; ad = C6["adh"]
R["cost"]["obs"] = {}
for g_, lab in ((1, "adh"), (0, "non")):
    v_ = cst[ad == g_]
    R["cost"]["obs"][lab] = dict(n=len(v_), mean=v_.mean(), sd=v_.std(ddof=1), med=np.median(v_), q1=np.percentile(v_, 25), q3=np.percentile(v_, 75))
    out("cost obs", lab, {k: round(float(vv), 1) for k, vv in R["cost"]["obs"][lab].items()})
# age-group model: all rows, CCI overall LR and P for trend within that model
ag = R["agegrp"]["m"]
for r_ in R["agegrp"]["rows"]:
    out("agegrp model", r_["name"], round(np.exp(r_["b"]), 3), round(np.exp(r_["lo"]), 3), round(np.exp(r_["hi"]), 3), "%.3g" % r_["p"])
m_ag_nocci = logistic(D, ["poly", "agegrp", "female", "medaid"])
m_ag_lin = logistic(D, ["poly", "agegrp", "female", "cci_lin", "medaid"])
R["agegrp"]["cci_overall"] = st.chi2.sf(2 * (ag["ll"] - m_ag_nocci["ll"]), 3)
R["agegrp"]["cci_trend"] = [r_ for r_ in wald_rows(m_ag_lin, m_ag_lin["names"]) if r_["name"] == "cci_score"][0]
out("agegrp model: CCI overall P", R["agegrp"]["cci_overall"], "trend OR", np.exp(R["agegrp"]["cci_trend"]["b"]), "P", R["agegrp"]["cci_trend"]["p"])
out("agegrp model group sizes and events", [(int(((D['age'] >= a) & (D['age'] < b_)).sum()), int(D['readm'][(D['age'] >= a) & (D['age'] < b_)].sum())) for a, b_ in ((65, 75), (75, 85), (85, 120))])
# univariable table rows (per 10 years for age)
out("uni rows", {t: [(round(np.exp(r_["b"]), 2), round(np.exp(r_["lo"]), 2), round(np.exp(r_["hi"]), 2), "%.3g" % r_["p"]) for r_ in R["uni"][t]] for t in ("poly", "female", "cci_d", "medaid")})
m10 = logistic(D, ["poly", "age10", "female", "cci_d", "medaid"])
R["multi10_rows"] = wald_rows(m10, m10["names"])
out("multi rows (age per 10)", [(r_["name"], round(np.exp(r_["b"]), 2), round(np.exp(r_["lo"]), 2), round(np.exp(r_["hi"]), 2), "%.3g" % r_["p"]) for r_ in R["multi10_rows"]])
# counts by variable for the table (n, events)
for t in ("poly", "female", "medaid"):
    for g_ in (1, 0):
        msk = D[t] == g_
        out("count", t, g_, int(msk.sum()), int(D["readm"][msk].sum()), round(D["readm"][msk].mean() * 100, 1))
out("CCI counts", R["cci_n"])
# how the age OR (per 10 years) moves with medaid / CCI
R["age_moves"] = {}
for lab, terms in (("age", ["age10"]), ("age+medaid", ["age10", "medaid"]), ("age+cci", ["age10", "cci_d"])):
    m_ = logistic(D, terms); r_ = wald_rows(m_, m_["names"])[1]
    R["age_moves"][lab] = np.exp(r_["b"])
out("age OR moves", {k: round(v, 3) for k, v in R["age_moves"].items()})
# mean age by medical aid within the 65-74 group (residual confounding after categorising age)
s_ = (D["age"] >= 65) & (D["age"] < 75)
R["age6574"] = (D["age"][s_ & (D["medaid"] == 1)].mean(), D["age"][s_ & (D["medaid"] == 0)].mean())
out("mean age within 65-74 by medaid", np.round(R["age6574"], 2))
# poly x medaid share
out("poly share by medaid", D["poly"][D["medaid"] == 1].mean(), D["poly"][D["medaid"] == 0].mean())


# =================================================================== python outputs (real statsmodels)
# The boxes in 다 and 마 절 print the real statsmodels 0.15.0 output. Run with
#   source /home/claude/pylibs/env.sh && python3 gen/nums_ch08.py
# The block below refits the model with statsmodels and checks it against the in-house IRLS fit above.
llf, lln = full["ll"], null["ll"]
try:
    import pandas as pd
    import statsmodels.formula.api as smf
except ImportError:  # statsmodels not on the path: skip the cross-check
    smf = None
if smf is not None:
    dc = pd.DataFrame({k: D[k] for k in ("age", "female", "cci", "medaid", "poly", "readm")})
    dc["cci"] = np.array(["0", "1-2", "3-4", "5+"])[dc["cci"]]
    sm_m = smf.logit("readm ~ poly + age + female + C(cci) + medaid", data=dc).fit(disp=0)
    sm_m0 = smf.logit("readm ~ age + female + C(cci) + medaid", data=dc).fit(disp=0)
    # patsy puts the categorical term first: Intercept, C(cci)[T.*], then the numeric terms
    SM_ORDER = [("Intercept", "(Intercept)"), ("C(cci)[T.1-2]", "cci1-2"), ("C(cci)[T.3-4]", "cci3-4"),
                ("C(cci)[T.5+]", "cci5+"), ("poly", "poly"), ("age", "age"), ("female", "female"), ("medaid", "medaid")]
    rows_by = {r_["name"]: r_ for r_ in R["full_rows"]}
    for smn, nm in SM_ORDER:
        assert abs(sm_m.params[smn] - rows_by[nm]["b"]) < 1e-6 and abs(sm_m.bse[smn] - rows_by[nm]["se"]) < 1e-6, smn
    assert abs(sm_m.llf - llf) < 1e-6 and abs(sm_m.llnull - lln) < 1e-6 and abs(sm_m0.llf - m_nopoly["ll"]) < 1e-6
    R["sm_iters"] = sm_m.mle_retvals["iterations"]
    out(str(sm_m.summary()))
    ci_ = np.exp(sm_m.conf_int()); ci_.columns = ["2.5%", "97.5%"]
    ci_.insert(0, "OR", np.exp(sm_m.params))
    out(str(ci_.round(3)))
    out("statsmodels: iters", R["sm_iters"], "llf", round(sm_m.llf, 3), "llnull", round(sm_m.llnull, 3), "llr", round(sm_m.llr, 2),
        "p", "%.2e" % sm_m.llr_pvalue, "aic", round(sm_m.aic, 3), "bic", round(sm_m.bic, 3),
        "| without poly aic", round(sm_m0.aic, 3), "bic", round(sm_m0.bic, 3), "| exact p poly", round(sm_m.pvalues["poly"], 4),
        "cci5+", round(sm_m.pvalues["C(cci)[T.5+]"], 4))
out("LR poly", 2 * (full["ll"] - m_nopoly["ll"]), st.chi2.sf(2 * (full["ll"] - m_nopoly["ll"]), 1), "m0 llf", m_nopoly["ll"])
out("CMH exact", R["cmh_sev"], "MH OR sev", R["mh_sev_or"])


if __name__ == "__main__":
    print("\n".join(LOG))
