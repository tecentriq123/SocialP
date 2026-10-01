"""Every number quoted in content/ch13.html is printed (and checked) by this script.
run: python3 gen/nums_ch13.py
Other scripts import the dicts defined here (fig_ch13.py).
The scipy ttest_ind output in 나 절 comes from gen/pyout_ch13.py (individual-level data with the
same means/SDs as the ITT summary below).
"""
import math
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar

Z975 = st.norm.ppf(0.975)
Z90 = st.norm.ppf(0.90)
Z80 = st.norm.ppf(0.80)


def welch(m1, s1, n1, m2, s2, n2):
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    return m1 - m2, se, df


def wilson(x, n, z=Z975):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def newcombe(x1, n1, x2, n2, z=Z975):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def wald(x1, n1, x2, n2, z=Z975):
    p1, p2 = x1 / n1, x2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    d = p1 - p2
    return d, se, d - z * se, d + z * se


def fm_restricted(x1, n1, x2, n2, delta):
    """Farrington-Manning restricted MLE under H0: p1 - p2 = delta (closed form), checked numerically."""
    p1h, p2h = x1 / n1, x2 / n2
    th = n2 / n1
    a = 1 + th
    b = -(1 + th + p1h + th * p2h + delta * (th + 2))
    c = delta ** 2 + delta * (2 * p1h + th + 1) + p1h + th * p2h
    d = -p1h * delta * (1 + delta)
    v = b ** 3 / (3 * a) ** 3 - b * c / (6 * a ** 2) + d / (2 * a)
    u = math.copysign(math.sqrt(b ** 2 / (3 * a) ** 2 - c / (3 * a)), v)
    w = (math.pi + math.acos(max(-1, min(1, v / u ** 3)))) / 3
    p1t = 2 * u * math.cos(w) - b / (3 * a)
    p2t = p1t - delta

    # numerical check: maximise the binomial log-likelihood with p1 = p2 + delta
    def nll(p2):
        p1 = p2 + delta
        if not (0 < p1 < 1 and 0 < p2 < 1):
            return 1e18
        return -(x1 * math.log(p1) + (n1 - x1) * math.log(1 - p1) + x2 * math.log(p2) + (n2 - x2) * math.log(1 - p2))
    lo, hi = max(1e-9, -delta + 1e-9), min(1 - 1e-9, 1 - delta - 1e-9)
    r = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    assert abs(r.x - p2t) < 1e-5, (r.x, p2t)
    return p1t, p2t


def fm_test(x1, n1, x2, n2, delta):
    p1t, p2t = fm_restricted(x1, n1, x2, n2, delta)
    se0 = math.sqrt(p1t * (1 - p1t) / n1 + p2t * (1 - p2t) / n2)
    z = (x1 / n1 - x2 / n2 - delta) / se0
    return z, 1 - st.norm.cdf(z), p1t, p2t, se0


def pr(*a):
    print(*a)


# =====================================================================================
pr("=" * 78, "\n가-1. 'no significant difference' example: cure 40/50 vs 43/50")
x1, n1, x2, n2 = 40, 50, 43, 50
d, se, lo, hi = wald(x1, n1, x2, n2)
pbar = (x1 + x2) / (n1 + n2)
se0 = math.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n2))
z0 = d / se0
pr(f"  RD {d*100:.1f}%p  Wald SE {se:.4f}  95% CI {lo*100:.1f} to {hi*100:.1f}")
pr(f"  pooled p {pbar:.2f} SE0 {se0:.4f} z {z0:.3f}  two-sided p {2*(1-st.norm.cdf(abs(z0))):.3f}")
chi = st.chi2_contingency(np.array([[x1, n1 - x1], [x2, n2 - x2]]), correction=False)
pr(f"  chi-square (no correction) {chi[0]:.3f} p {chi[1]:.3f}")
A1 = dict(d=d, lo=lo, hi=hi, p=2 * (1 - st.norm.cdf(abs(z0))))

# =====================================================================================
pr("=" * 78, "\n가-2. Figure 13-1 scenarios (schematic, cure-rate difference, margin -10)")
SCEN = [
    ("A", 8.0, 2.5, 13.5, "우월성 입증"),
    ("B", 1.0, -5.5, 7.5, "비열등성 입증"),
    ("C", -4.5, -8.5, -0.5, "비열등성 입증, 단 통계적으로는 열세"),
    ("D", -2.5, -13.0, 8.0, "결론 불가"),
    ("E", -8.5, -14.5, -2.5, "결론 불가, 통계적으로 열세"),
    ("F", -17.0, -23.0, -11.0, "열등"),
]
for s in SCEN:
    pr("  ", s)

# =====================================================================================
pr("=" * 78, "\n가-3. one-sided 0.025 <-> two-sided 95%; TOST 0.05 <-> 90%")
pr(f"  z 0.975 = {Z975:.3f}; z 0.95 = {st.norm.ppf(0.95):.3f}")
pr(f"  ln 0.8 = {math.log(0.8):.4f}, ln 1.25 = {math.log(1.25):.4f}, 0.8*1.25 = {0.8*1.25}")

# =====================================================================================
pr("=" * 78, "\n가-4. BE worked example, 12 volunteers (paired simplification)")
AUC_R = np.array([412, 538, 365, 620, 488, 297, 554, 431, 702, 389, 515, 460], float)
rng = np.random.default_rng(1307)
BEST = None
for seed in range(4000):
    r = np.random.default_rng(seed)
    lr = r.normal(-0.04, 0.12, 12)
    auc_t = np.round(AUC_R * np.exp(lr))
    l = np.log(auc_t / AUC_R)
    m, s = l.mean(), l.std(ddof=1)
    se_ = s / math.sqrt(12)
    t_ = st.t.ppf(0.95, 11)
    g, glo, ghi = math.exp(m), math.exp(m - t_ * se_), math.exp(m + t_ * se_)
    pconv = 2 * (1 - st.t.cdf(abs(m / se_), 11))
    score = abs(g - 0.955) + abs(glo - 0.89) + abs(ghi - 1.03) + abs(pconv - 0.30) * 0.2
    if BEST is None or score < BEST[0]:
        BEST = (score, seed, auc_t)
AUC_T = BEST[2]
pr("  seed", BEST[1])
LR = np.log(AUC_T / AUC_R)
m, s = LR.mean(), LR.std(ddof=1)
se_be = s / math.sqrt(12)
t95_11 = st.t.ppf(0.95, 11)
t975_11 = st.t.ppf(0.975, 11)
BE12 = dict(R=AUC_R, T=AUC_T, lr=LR, m=m, sd=s, se=se_be, t=t95_11,
            lo=m - t95_11 * se_be, hi=m + t95_11 * se_be)
pr("  subj  R     T     ratio   ln(ratio)")
for i in range(12):
    pr(f"  {i+1:2d}  {AUC_R[i]:4.0f}  {AUC_T[i]:4.0f}  {AUC_T[i]/AUC_R[i]:.3f}  {LR[i]:+.3f}")
pr(f"  sum ln ratio {LR.sum():.4f}")
pr(f"  mean ln ratio {m:.4f}  SD {s:.4f}  SE {se_be:.4f}  t(0.95,11) {t95_11:.3f}  t(0.975,11) {t975_11:.3f}")
pr(f"  geometric means: R {math.exp(np.log(AUC_R).mean()):.1f}  T {math.exp(np.log(AUC_T).mean()):.1f}")
pr(f"  arithmetic means: R {AUC_R.mean():.1f}  T {AUC_T.mean():.1f}  ratio of arithmetic means {AUC_T.mean()/AUC_R.mean():.3f}")
pr(f"  GMR {math.exp(m):.4f}  -> {math.exp(m)*100:.2f}%")
pr(f"  90% CI log {m - t95_11*se_be:.4f} to {m + t95_11*se_be:.4f}; ratio {math.exp(m - t95_11*se_be)*100:.2f}% to {math.exp(m + t95_11*se_be)*100:.2f}%")
pr(f"  95% CI ratio {math.exp(m - t975_11*se_be)*100:.2f}% to {math.exp(m + t975_11*se_be)*100:.2f}%")
tconv = m / se_be
pr(f"  conventional paired t {tconv:.3f}  p {2*(1-st.t.cdf(abs(tconv),11)):.3f}")
tL = (m - math.log(0.8)) / se_be
tU = (math.log(1.25) - m) / se_be
pr(f"  TOST: t_L = ({m:.4f} - ({math.log(0.8):.4f}))/{se_be:.4f} = {tL:.3f} p {1-st.t.cdf(tL,11):.2e}")
pr(f"        t_U = ({math.log(1.25):.4f} - ({m:.4f}))/{se_be:.4f} = {tU:.3f} p {1-st.t.cdf(tU,11):.2e}")
BE12.update(tL=tL, tU=tU, pL=1 - st.t.cdf(tL, 11), pU=1 - st.t.cdf(tU, 11), pconv=2 * (1 - st.t.cdf(abs(tconv), 11)))
cvw = math.sqrt(math.exp(s ** 2 / 2) - 1)
pr(f"  implied within-subject CV (sigma_W^2 = SD^2/2): {cvw*100:.1f}%")
# check with R-like paired t.test mu = log(0.8)
tt = st.ttest_1samp(LR, math.log(0.8), alternative="greater")
tt2 = st.ttest_1samp(LR, math.log(1.25), alternative="less")
pr(f"  scipy check TOST: {tt.statistic:.3f} {tt.pvalue:.2e} | {tt2.statistic:.3f} {tt2.pvalue:.2e}")

# =====================================================================================
pr("=" * 78, "\n가-5. Figure 13-2 BE scenarios (GMR, CV, n, 2x2 crossover)")


def be_case(gmr, cv, n):
    s2 = math.log(1 + cv ** 2)
    se = math.sqrt(2 * s2 / n)
    df = n - 2
    t = st.t.ppf(0.95, df)
    m = math.log(gmr)
    p = 2 * (1 - st.t.cdf(abs(m / se), df))
    return dict(gmr=gmr, lo=math.exp(m - t * se), hi=math.exp(m + t * se), p=p, cv=cv, n=n, df=df, se=se)


BECASES = [
    ("①", dict(gmr=math.exp(BE12["m"]), lo=math.exp(BE12["lo"]), hi=math.exp(BE12["hi"]), p=BE12["pconv"], n=12, cv=cvw),
     "위 예제 (12명)"),
    ("②", be_case(0.95, 0.35, 12), "변동이 큰 약 (12명)"),
    ("③", be_case(0.92, 0.12, 48), "변동이 작고 대상자가 많음 (48명)"),
    ("④", be_case(1.15, 0.20, 24), "흡수가 더 많은 제제 (24명)"),
]
for k, c, lab in BECASES:
    be = c["lo"] >= 0.8 and c["hi"] <= 1.25
    pr(f"  {k} {lab}: GMR {c['gmr']*100:.1f}%  90% CI {c['lo']*100:.1f}-{c['hi']*100:.1f}  p(diff=0) {c['p']:.3f}  CV {c['cv']*100:.0f}% n {c['n']}  BE={be}")

# =====================================================================================
pr("=" * 78, "\n가-6. BE paper table: 2x2 crossover, n = 24 (difference method = ANOVA)")


def crossover(seed, n=24, gmr=0.97, cvw=0.15, sb=0.30, mu=math.log(5200), per=0.02):
    r = np.random.default_rng(seed)
    sw = math.sqrt(math.log(1 + cvw ** 2))
    seq = np.array([0] * (n // 2) + [1] * (n // 2))  # 0 = TR, 1 = RT
    subj = r.normal(0, sb, n)
    y = np.zeros((n, 2))
    trt = np.zeros((n, 2), int)
    for i in range(n):
        for p in range(2):
            t = 1 if (seq[i] == 0 and p == 0) or (seq[i] == 1 and p == 1) else 0
            trt[i, p] = t
            y[i, p] = mu + subj[i] + (math.log(gmr) if t else 0) + (per if p == 1 else 0) + r.normal(0, sw)
    return seq, y, trt


def analyse(seq, y, trt):
    d = (y[:, 0] - y[:, 1]) / 2
    d1, d2 = d[seq == 0], d[seq == 1]
    n1, n2 = len(d1), len(d2)
    est = d1.mean() - d2.mean()
    sp2 = ((n1 - 1) * d1.var(ddof=1) + (n2 - 1) * d2.var(ddof=1)) / (n1 + n2 - 2)
    se = math.sqrt(sp2 * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    t = st.t.ppf(0.95, df)
    mse = 2 * sp2
    cv = math.sqrt(math.exp(mse) - 1)
    gT = math.exp(y[trt == 1].mean())
    gR = math.exp(y[trt == 0].mean())
    return dict(est=est, se=se, df=df, lo=est - t * se, hi=est + t * se, cv=cv, gT=gT, gR=gR, mse=mse,
                gmr=math.exp(est), glo=math.exp(est - t * se), ghi=math.exp(est + t * se))


def ols_check(seq, y, trt):
    """verify difference method against OLS with subject + period + treatment."""
    n = len(seq)
    rows, yy = [], []
    for i in range(n):
        for p in range(2):
            x = np.zeros(n + 2)
            x[i] = 1
            x[n] = 1 if p == 1 else 0
            x[n + 1] = trt[i, p]
            rows.append(x)
            yy.append(y[i, p])
    X, Y = np.array(rows), np.array(yy)
    b, res, rk, _ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ b
    dfr = len(Y) - rk
    mse = resid @ resid / dfr
    cov = mse * np.linalg.pinv(X.T @ X)
    return b[-1], math.sqrt(cov[-1, -1]), dfr, mse


BEP = {}
spec = {"AUC0-t": (0.97, 0.15, math.log(5200), 10), "AUC0-inf": (0.97, 0.15, math.log(5480), 10), "Cmax": (0.94, 0.21, math.log(812), 62)}
# AUC0-t and AUC0-inf share the subject/period structure: use same seed but slightly different means + small extra noise
for k, (g, cv, mu, seed) in spec.items():
    seq, y, trt = crossover(seed, gmr=g, cvw=cv, mu=mu)
    if k == "AUC0-inf":
        seq, y0, trt = crossover(seed, gmr=g, cvw=cv, mu=math.log(5200))
        r = np.random.default_rng(99)
        y = y0 + math.log(5480 / 5200) + r.normal(0, 0.012, y0.shape)
    a = analyse(seq, y, trt)
    b = ols_check(seq, y, trt)
    assert abs(a["est"] - b[0]) < 1e-10 and abs(a["se"] - b[1]) < 1e-10 and abs(a["mse"] - b[3]) < 1e-10
    BEP[k] = a
    pr(f"  {k}: geoLS T {a['gT']:.1f}  R {a['gR']:.1f}  GMR {a['gmr']*100:.2f}%  90% CI {a['glo']*100:.2f}-{a['ghi']*100:.2f}  "
       f"CVintra {a['cv']*100:.1f}%  df {a['df']}  (OLS check ok)  ratio of geo means {a['gT']/a['gR']*100:.2f}")
    ltest = (a["est"] - math.log(0.8)) / a["se"]
    utest = (math.log(1.25) - a["est"]) / a["se"]
    pr(f"      TOST p_L {1-st.t.cdf(ltest, a['df']):.2e}  p_U {1-st.t.cdf(utest, a['df']):.2e}   p(diff=0) {2*(1-st.t.cdf(abs(a['est']/a['se']), a['df'])):.3f}")

# =====================================================================================
pr("=" * 78, "\n가-7. margin derivation (hypothetical meta-analysis, HbA1c)")
HIST = dict(est=-0.95, lo=-1.10, hi=-0.80)
M1 = 0.80
M2 = 0.5 * M1
pr(f"  C vs placebo {HIST}  SE {(HIST['hi']-HIST['lo'])/(2*Z975):.4f}; M1 = {M1}; M2 (50%) = {M2}")
pr(f"  if T - C = +{M2}: T - P <= {M2} - {M1} = {M2 - M1:.2f}")

# =====================================================================================
pr("=" * 78, "\n가-8. ITT dilution example")
pT, pC, nonadh, pnon = 0.70, 0.82, 0.20, 0.30
iT = (1 - nonadh) * pT + nonadh * pnon
iC = (1 - nonadh) * pC + nonadh * pnon
pr(f"  adherent: T {pT} C {pC} diff {(pT-pC)*100:.1f}; ITT: T {iT*100:.1f} C {iC*100:.1f} diff {(iT-iC)*100:.1f}")

# =====================================================================================
pr("=" * 78, "\n다-0 / 가-9. HbA1c NI trial design (Methods box in 가)")
sd_a, Dm = 1.1, 0.4
zsum = Z975 + Z90
n_h = 2 * sd_a ** 2 * zsum ** 2 / Dm ** 2
pr(f"  (z.975+z.90)^2 = {zsum**2:.3f}; n/group = 2*{sd_a}^2*{zsum**2:.3f}/{Dm}^2 = {n_h:.1f} -> {math.ceil(n_h)}")
n_h_c = math.ceil(n_h)
pr(f"  15% dropout: {n_h_c}/0.85 = {n_h_c/0.85:.1f} -> {math.ceil(n_h_c/0.85)} ; total {2*math.ceil(n_h_c/0.85)}")
for dm in (0.3, 0.4, 0.5):
    pr(f"    margin {dm}: n = {2*sd_a**2*zsum**2/dm**2:.1f}")
pr(f"    margin 0.4, true diff +0.1 (T slightly worse): n = {2*sd_a**2*zsum**2/(0.4-0.1)**2:.1f}")

# =====================================================================================
pr("=" * 78, "\n나. PDC telepharmacy NI trial (higher is better), margin 5%p")
DP = 5.0
sd_p = 18.0
n_p = 2 * sd_p ** 2 * zsum ** 2 / DP ** 2
pr(f"  design: n/group = 2*18^2*{zsum**2:.3f}/5^2 = {n_p:.1f} -> {math.ceil(n_p)}; 10% dropout -> {math.ceil(n_p)/0.9:.1f} -> {math.ceil(math.ceil(n_p)/0.9)}")
pr(f"  if true diff = -1: n = {2*sd_p**2*zsum**2/(DP-1)**2:.1f};  margin 4, diff 0: n = {2*sd_p**2*zsum**2/4**2:.1f}; margin 10: {2*sd_p**2*zsum**2/10**2:.1f}")
pr(f"  superiority design for 5%p difference, two-sided 0.05, 90% power: {2*sd_p**2*zsum**2/5**2:.1f} (same formula)")
PDC = {}
for lab, (mT, sT, nT, mC, sC, nC) in {"ITT": (83.9, 16.8, 304, 85.2, 17.5, 304), "PP": (85.0, 15.9, 262, 85.6, 17.1, 275)}.items():
    d, se, df = welch(mT, sT, nT, mC, sC, nC)
    t975 = st.t.ppf(0.975, df)
    lo, hi = d - t975 * se, d + t975 * se
    tni = (d + DP) / se
    pni = 1 - st.t.cdf(tni, df)
    tsup = d / se
    psup = 2 * (1 - st.t.cdf(abs(tsup), df))
    crit = -DP + t975 * se
    PDC[lab] = dict(mT=mT, sT=sT, nT=nT, mC=mC, sC=sC, nC=nC, d=d, se=se, df=df, t975=t975, lo=lo, hi=hi,
                    tni=tni, pni=pni, tsup=tsup, psup=psup, crit=crit)
    pr(f"  {lab}: T {mT}±{sT} (n={nT})  C {mC}±{sC} (n={nC})  d {d:.2f}  SE {se:.4f}  df {df:.2f}  t.975 {t975:.4f}")
    pr(f"       SE^2 parts {sT**2/nT:.4f} + {sC**2/nC:.4f} = {se**2:.4f}")
    pr(f"       95% CI {lo:.2f} to {hi:.2f} (exact {lo:.4f}, {hi:.4f}); t_NI = ({d:.1f}+5)/{se:.3f} = {tni:.4f} one-sided p {pni:.5f}")
    pr(f"       superiority t {tsup:.4f} two-sided p {psup:.4f}; critical d for NI rejection: {crit:.3f}")
    pr(f"       z-approx p_NI {1-st.norm.cdf(tni):.5f}")

# =====================================================================================
pr("=" * 78, "\n다. HbA1c NI trial (lower is better), margin +0.4")
HB = {}
for lab, (mT, sT, nT, mC, sC, nC) in {"FAS": (-0.74, 1.02, 185, -0.83, 1.06, 183), "PP": (-0.79, 1.00, 162, -0.85, 1.04, 165)}.items():
    d, se, df = welch(mT, sT, nT, mC, sC, nC)
    t975 = st.t.ppf(0.975, df)
    lo, hi = d - t975 * se, d + t975 * se
    tni = (Dm - d) / se
    pni = 1 - st.t.cdf(tni, df)
    tsup = d / se
    psup = 2 * (1 - st.t.cdf(abs(tsup), df))
    HB[lab] = dict(mT=mT, sT=sT, nT=nT, mC=mC, sC=sC, nC=nC, d=d, se=se, df=df, t975=t975, lo=lo, hi=hi, tni=tni, pni=pni, psup=psup)
    pr(f"  {lab}: T {mT}±{sT} (n={nT})  C {mC}±{sC} (n={nC})  d(T-C) {d:.2f}  SE {se:.4f} df {df:.2f} t.975 {t975:.4f}")
    pr(f"       SE^2 parts {sT**2/nT:.5f} + {sC**2/nC:.5f} = {se**2:.5f}")
    pr(f"       95% CI {lo:.3f} to {hi:.3f}; t_NI = (0.4 - {d:.2f})/{se:.4f} = {tni:.3f} one-sided p {pni:.5f}; sup p {psup:.3f}")
    pr(f"       C - T: {-d:.2f} (95% CI {-hi:.3f} to {-lo:.3f})")
    pr(f"       T SE {sT/math.sqrt(nT):.3f}  C SE {sC/math.sqrt(nC):.3f}")
pr(f"  enrolled 188 + 188 = 376; FAS 185 + 183; PP 162 + 165")

# =====================================================================================
pr("=" * 78, "\n라. H. pylori eradication NI trial, margin -10%p")
DR = -0.10
p0 = 0.85
n_r = (Z975 + Z80) ** 2 * 2 * p0 * (1 - p0) / 0.10 ** 2
pr(f"  design: (z.975+z.80)^2 = {(Z975+Z80)**2:.3f}; n = {(Z975+Z80)**2:.3f} * 2*0.85*0.15 / 0.1^2 = {n_r:.1f} -> {math.ceil(n_r)}; /0.9 = {math.ceil(n_r)/0.9:.1f} -> {math.ceil(math.ceil(n_r)/0.9)}")
for pp in (0.70, 0.85, 0.95):
    sep = math.sqrt(2 * pp * (1 - pp) / math.ceil(n_r))
    pw = st.norm.cdf(0.10 / sep - Z975)
    pr(f"    power at true rates {pp} both, n={math.ceil(n_r)}: SE {sep:.4f}  power {pw:.3f}")
HP = {}
for lab, (x1, n1, x2, n2) in {"ITT": (186, 224, 190, 224), "PP": (180, 205, 185, 208)}.items():
    d, se, lo, hi = wald(x1, n1, x2, n2)
    nlo, nhi = newcombe(x1, n1, x2, n2)
    zw = (d - DR) / se
    pw = 1 - st.norm.cdf(zw)
    zf, pf, p1t, p2t, se0 = fm_test(x1, n1, x2, n2, DR)
    HP[lab] = dict(x1=x1, n1=n1, x2=x2, n2=n2, p1=x1 / n1, p2=x2 / n2, d=d, se=se, lo=lo, hi=hi, nlo=nlo, nhi=nhi,
                   zw=zw, pw=pw, zf=zf, pf=pf, p1t=p1t, p2t=p2t, se0=se0)
    pr(f"  {lab}: T {x1}/{n1} = {x1/n1*100:.2f}%  C {x2}/{n2} = {x2/n2*100:.2f}%  RD {d*100:.2f}%p")
    pr(f"       Wald SE {se:.5f} parts {x1/n1*(1-x1/n1)/n1:.6f} + {x2/n2*(1-x2/n2)/n2:.6f};  95% CI {lo*100:.2f} to {hi*100:.2f}")
    pr(f"       Newcombe 95% CI {nlo*100:.2f} to {nhi*100:.2f};  Wilson T {tuple(round(v*100,2) for v in wilson(x1,n1))} C {tuple(round(v*100,2) for v in wilson(x2,n2))}")
    pr(f"       Wald NI z = ({d:.4f} + 0.10)/{se:.4f} = {zw:.3f} p {pw:.5f}")
    pr(f"       FM: restricted p~T {p1t:.4f} p~C {p2t:.4f} SE0 {se0:.5f} z {zf:.3f} p {pf:.5f}")
    # superiority chi-square
    tab = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    c2 = st.chi2_contingency(tab, correction=False)
    pr(f"       chi-square p (difference = 0) {c2[1]:.3f}")
    # ratio scales
    rr = (x1 / n1) / (x2 / n2)
    selr = math.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)
    pr(f"       RR success {rr:.3f} (95% CI {rr*math.exp(-Z975*selr):.3f}-{rr*math.exp(Z975*selr):.3f})")
    f1, f2 = n1 - x1, n2 - x2
    rrf = (f1 / n1) / (f2 / n2)
    self_ = math.sqrt(1 / f1 - 1 / n1 + 1 / f2 - 1 / n2)
    pr(f"       failures T {f1}/{n1} = {f1/n1*100:.1f}%  C {f2}/{n2} = {f2/n2*100:.1f}%;  RR failure {rrf:.3f} (95% CI {rrf*math.exp(-Z975*self_):.3f}-{rrf*math.exp(Z975*self_):.3f})")
    orr = (x1 / (n1 - x1)) / (x2 / (n2 - x2))
    seor = math.sqrt(1 / x1 + 1 / (n1 - x1) + 1 / x2 + 1 / (n2 - x2))
    pr(f"       OR success {orr:.3f} (95% CI {orr*math.exp(-Z975*seor):.3f}-{orr*math.exp(Z975*seor):.3f})")
pr(f"  randomized 224+224 = 448; excluded from PP T {224-205}, C {224-208}")

pr("=" * 78, "\n라-2. small-n near-100% example: 48/50 vs 50/50")
SM = {}
for lab, (x1, n1, x2, n2) in {"small": (48, 50, 50, 50), "small2": (46, 50, 48, 50)}.items():
    d, se, lo, hi = wald(x1, n1, x2, n2)
    nlo, nhi = newcombe(x1, n1, x2, n2)
    zf, pf, p1t, p2t, se0 = fm_test(x1, n1, x2, n2, DR)
    zw = (d - DR) / se if se > 0 else float("inf")
    SM[lab] = dict(x1=x1, n1=n1, x2=x2, n2=n2, d=d, se=se, lo=lo, hi=hi, nlo=nlo, nhi=nhi, zf=zf, pf=pf, zw=zw,
                   pw=1 - st.norm.cdf(zw), w1=wilson(x1, n1), w2=wilson(x2, n2), p1t=p1t, p2t=p2t, se0=se0)
    pr(f"  {lab}: {x1}/{n1} vs {x2}/{n2}: RD {d*100:.1f}  Wald SE {se:.4f} CI {lo*100:.2f} to {hi*100:.2f}  Wald z {zw:.3f} p {1-st.norm.cdf(zw):.4f}")
    pr(f"        Wilson T {tuple(round(v*100,2) for v in wilson(x1,n1))}  C {tuple(round(v*100,2) for v in wilson(x2,n2))}")
    pr(f"        Newcombe {nlo*100:.2f} to {nhi*100:.2f};  FM z {zf:.3f} p {pf:.4f}  (p~T {p1t:.4f}, p~C {p2t:.4f}, SE0 {se0:.4f})")

# simulation of coverage of the one-sided lower bound at the margin boundary (type I error of NI claim)
pr("=" * 78, "\n라-3. type I error of NI claim at the boundary (true pT - pC = -0.10), n = 50/group, exact enumeration")


def type1(pC, n, method):
    pT = pC - 0.10
    k = np.arange(n + 1)
    fT = st.binom.pmf(k, n, pT)
    fC = st.binom.pmf(k, n, pC)
    tot = 0.0
    for a in range(n + 1):
        for b in range(n + 1):
            if method == "wald":
                d, se, lo, hi = wald(a, n, b, n)
                rej = lo > -0.10 if se > 0 else (a / n - b / n) > -0.10
            else:
                lo, hi = newcombe(a, n, b, n)
                rej = lo > -0.10
            if rej:
                tot += fT[a] * fC[b]
    return tot


T1 = {}
for pC in (0.85, 0.95, 0.99):
    T1[pC] = (type1(pC, 50, "wald"), type1(pC, 50, "newcombe"))
    pr(f"  control {pC}, new {pC-0.10:.2f}, n=50: P(claim NI) Wald {T1[pC][0]:.4f}  Newcombe {T1[pC][1]:.4f}  (nominal 0.025)")

pr("=" * 78, "\n라-4. meaning of a fixed -10%p margin by control success rate")
for pc in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
    fc = 1 - pc
    pr(f"  control success {pc:.2f}: allowed new success {pc-0.10:.2f}; failure {fc:.2f} -> {fc+0.10:.2f}: failure RR {(fc+0.10)/fc:.2f}; success RR {(pc-0.10)/pc:.3f}")

pr("=" * 78, "\nwidget defaults check")
for est, hw, D in ((1.0, 6.5, 10), (-3.0, 5.0, 10)):
    se = hw / Z975
    pr(f"  est {est} hw95 {hw} -> SE {se:.3f}; 90% hw {Z90*0+st.norm.ppf(0.95)*se:.3f}; p0 {2*(1-st.norm.cdf(abs(est/se))):.3f}; pNI {1-st.norm.cdf((est+D)/se):.4f}")


# =====================================================================================
# Score-based intervals (Mee / Miettinen-Nurminen) by inverting the restricted-MLE score statistic.
# Validated against Newcombe (1998) Table II, example 56/70 vs 48/80:
#   Newcombe hybrid 0.0524-0.3339, Miettinen-Nurminen 0.0528-0.3382, Mee 0.0533-0.3377.
from scipy.optimize import brentq


def score_z(x1, n1, x2, n2, delta, mn=True):
    p1t, p2t = fm_restricted(x1, n1, x2, n2, delta)
    v = p1t * (1 - p1t) / n1 + p2t * (1 - p2t) / n2
    if mn:
        v *= (n1 + n2) / (n1 + n2 - 1)
    return (x1 / n1 - x2 / n2 - delta) / math.sqrt(v)


def score_ci(x1, n1, x2, n2, mn=True, z=Z975):
    d = x1 / n1 - x2 / n2
    eps = 1e-7
    lo = brentq(lambda dl: score_z(x1, n1, x2, n2, dl, mn) - z, -1 + eps, d) if d > -1 + eps else -1.0
    hi = brentq(lambda dl: score_z(x1, n1, x2, n2, dl, mn) + z, d, 1 - eps) if d < 1 - eps else 1.0
    return lo, hi


pr("=" * 78, "\n라-5. validation of Newcombe / MN / Mee against Newcombe 1998 example (56/70 vs 48/80)")
_nc = newcombe(56, 70, 48, 80)
_mn = score_ci(56, 70, 48, 80, True)
_mee = score_ci(56, 70, 48, 80, False)
pr(f"  Newcombe {_nc[0]:.4f} {_nc[1]:.4f} (published 0.0524 0.3339)")
pr(f"  MN       {_mn[0]:.4f} {_mn[1]:.4f} (published 0.0528 0.3382)")
pr(f"  Mee      {_mee[0]:.4f} {_mee[1]:.4f} (published 0.0533 0.3377)")
assert abs(_nc[0] - 0.0524) < 6e-5 and abs(_nc[1] - 0.3339) < 6e-5
assert abs(_mn[0] - 0.0528) < 6e-5 and abs(_mn[1] - 0.3382) < 6e-5
assert abs(_mee[0] - 0.0533) < 6e-5 and abs(_mee[1] - 0.3377) < 6e-5

pr("=" * 78, "\n라-6. three CI methods for the chapter examples")
for key, dct in (("ITT", HP["ITT"]), ("PP", HP["PP"]), ("small", SM["small"]), ("small2", SM["small2"])):
    x1, n1, x2, n2 = dct["x1"], dct["n1"], dct["x2"], dct["n2"]
    mlo, mhi = score_ci(x1, n1, x2, n2, True)
    zmn = score_z(x1, n1, x2, n2, DR, True)
    dct.update(mlo=mlo, mhi=mhi, zmn=zmn, pmn=1 - st.norm.cdf(zmn))
    pr(f"  {key}: Wald {dct['lo']*100:.2f} to {dct['hi']*100:.2f} | Newcombe {dct['nlo']*100:.2f} to {dct['nhi']*100:.2f} | "
       f"MN {mlo*100:.2f} to {mhi*100:.2f} | MN z(-0.10) {zmn:.3f} p {1-st.norm.cdf(zmn):.5f}")

pr("=" * 78, "\n라-7. Newcombe step by step (ITT)")
q = HP["ITT"]
wT, wC = wilson(q["x1"], q["n1"]), wilson(q["x2"], q["n2"])
aT, bT = q["p1"] - wT[0], wT[1] - q["p1"]
aC, bC = q["p2"] - wC[0], wC[1] - q["p2"]
pr(f"  pT {q['p1']*100:.3f}  Wilson {wT[0]*100:.3f}-{wT[1]*100:.3f}  (below {aT*100:.3f}, above {bT*100:.3f})")
pr(f"  pC {q['p2']*100:.3f}  Wilson {wC[0]*100:.3f}-{wC[1]*100:.3f}  (below {aC*100:.3f}, above {bC*100:.3f})")
pr(f"  lower: d - sqrt({aT*100:.2f}^2 + {bC*100:.2f}^2) = {q['d']*100:.3f} - {math.sqrt(aT**2+bC**2)*100:.3f} = {q['nlo']*100:.3f}")
pr(f"  upper: d + sqrt({bT*100:.2f}^2 + {aC*100:.2f}^2) = {q['d']*100:.3f} + {math.sqrt(bT**2+aC**2)*100:.3f} = {q['nhi']*100:.3f}")
NCSTEP = dict(wT=wT, wC=wC, aT=aT, bT=bT, aC=aC, bC=bC)
q = SM["small"]
pr(f"  small: Wilson T {q['w1'][0]*100:.2f}-{q['w1'][1]*100:.2f}  C {q['w2'][0]*100:.2f}-{q['w2'][1]*100:.2f}")
pr(f"  small Wald SE parts: T {0.96*0.04/50:.6f} C {1.0*0/50:.6f}; SE {q['se']:.4f}")

pr("=" * 78, "\n나-2. R t.test(..., mu = -5, alternative = 'greater', conf.level = 0.975) numbers (ITT)")
P_ = PDC["ITT"]
pr(f"  t = {P_['tni']:.4f}, df = {P_['df']:.2f}, p-value = {P_['pni']:.6f}")
pr(f"  one-sided 97.5% lower bound = d - t.975*SE = {P_['d'] - P_['t975']*P_['se']:.6f}")
pr(f"  default conf.level 0.95 one-sided lower bound = {P_['d'] - st.t.ppf(0.95, P_['df'])*P_['se']:.4f}")

pr("=" * 78, "\n가-10. rounding helpers for text")
pr(f"  BE12: GMR {math.exp(BE12['m'])*100:.2f}  CI {math.exp(BE12['lo'])*100:.2f}-{math.exp(BE12['hi'])*100:.2f}; "
   f"exp(-0.0435) {math.exp(-0.0435):.4f}; m - t*se: {BE12['m']:.4f} - {BE12['t']:.3f}*{BE12['se']:.4f} = {BE12['m'] - BE12['t']*BE12['se']:.4f}")
pr(f"  BE12 half-width log {BE12['t']*BE12['se']:.4f}")
pr(f"  PDC ITT half-width {PDC['ITT']['t975']*PDC['ITT']['se']:.3f}; HbA1c FAS half-width {HB['FAS']['t975']*HB['FAS']['se']:.3f}")

pr("=" * 78, "\n가-11. review of the 2026-09-30 rewrite: point estimates that still pass 80.00-125.00% (Table 2 AUC precision)")
a = BEP["AUC0-t"]
hw = math.log(a["ghi"]) - math.log(a["gmr"])          # half-width of the 90% CI on the log scale
pr(f"  AUC0-t: GMR {a['gmr']*100:.2f}%  90% CI {a['glo']*100:.2f}-{a['ghi']*100:.2f}; log half-width {hw:.4f} -> CI limits = GMR x/÷ {math.exp(hw):.4f}")
pr(f"  relative spread: lower {a['glo']/a['gmr']*100 - 100:.2f}%, upper +{a['ghi']/a['gmr']*100 - 100:.2f}% (text: 'about ±6.5%'); "
   f"points: down {(a['gmr']-a['glo'])*100:.2f}, up {(a['ghi']-a['gmr'])*100:.2f}")
pr(f"  passing GMR range with this precision: {0.8*math.exp(hw)*100:.1f}% to {1.25/math.exp(hw)*100:.1f}% (text: 'roughly 85-117%')")
