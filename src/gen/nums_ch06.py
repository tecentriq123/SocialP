"""Chapter 6 numbers: every statistic quoted in content/ch06.html is computed here.

Run:  python3 gen/nums_ch06.py      (prints all numbers)
fig_ch06.py imports the datasets from this module so figures and text agree.
statsmodels is not available, so OLS is implemented with numpy.
"""
import itertools
import numpy as np
import scipy.stats as st

T975 = st.t.ppf


# ------------------------------------------------------------------ helpers
def ols(X, y, names=None, intercept=True):
    """Ordinary least squares. X: (n,p) array (no intercept column)."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    n = len(y)
    Xd = np.column_stack([np.ones(n), X]) if intercept else X
    k = Xd.shape[1]
    XtX_inv = np.linalg.inv(Xd.T @ Xd)
    b = XtX_inv @ Xd.T @ y
    fit = Xd @ b
    e = y - fit
    df = n - k
    rss = float(e @ e)
    tss = float(((y - y.mean()) ** 2).sum())
    s2 = rss / df
    se = np.sqrt(np.diag(s2 * XtX_inv))
    t = b / se
    p = 2 * st.t.sf(np.abs(t), df)
    tc = st.t.ppf(0.975, df)
    p_ = k - 1
    R2 = 1 - rss / tss
    adjR2 = 1 - (1 - R2) * (n - 1) / df
    F = ((tss - rss) / p_) / s2 if p_ > 0 else np.nan
    pF = st.f.sf(F, p_, df) if p_ > 0 else np.nan
    return dict(b=b, se=se, t=t, p=p, lo=b - tc * se, hi=b + tc * se, df=df, n=n, k=k, rss=rss, tss=tss,
                sigma=np.sqrt(s2), R2=R2, adjR2=adjR2, F=F, pF=pF, fit=fit, resid=e, XtX_inv=XtX_inv, Xd=Xd,
                names=names, tcrit=tc)


def vif(X):
    X = np.asarray(X, float)
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = ols(others, X[:, j])["R2"]
        out.append(1 / (1 - r2))
    return np.array(out)


def fisher_ci(r, n, level=0.95):
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    q = st.norm.ppf(0.5 + level / 2)
    return np.tanh(z - q * se), np.tanh(z + q * se), z, se


def r_test(r, n):
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
    return t, 2 * st.t.sf(abs(t), n - 2)


def exact_r(x, r, rng, scale=1.0, loc=0.0):
    """Return y whose sample Pearson correlation with x is exactly r."""
    zx = (x - x.mean()) / x.std()
    e = rng.normal(size=len(x))
    e = e - e.mean()
    e = e - (e @ zx) / (zx @ zx) * zx
    ze = e / e.std()
    return loc + scale * (r * zx + np.sqrt(1 - r ** 2) * ze)


def fmtp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ================================================================== 가. Pearson
# 8 patients: age (years) and systolic blood pressure (mmHg)
AGE8 = np.array([42, 48, 53, 58, 62, 66, 71, 80], float)
SBP8 = np.array([125, 127, 133, 141, 129, 139, 137, 149], float)


def pearson8():
    x, y = AGE8, SBP8
    n = len(x)
    dx, dy = x - x.mean(), y - y.mean()
    Sxy, Sxx, Syy = (dx * dy).sum(), (dx ** 2).sum(), (dy ** 2).sum()
    cov = Sxy / (n - 1)
    sx, sy = np.sqrt(Sxx / (n - 1)), np.sqrt(Syy / (n - 1))
    r = Sxy / np.sqrt(Sxx * Syy)
    t, p = r_test(r, n)
    lo, hi, z, zse = fisher_ci(r, n)
    return dict(n=n, mx=x.mean(), my=y.mean(), dx=dx, dy=dy, prod=dx * dy, Sxy=Sxy, Sxx=Sxx, Syy=Syy, cov=cov,
                sx=sx, sy=sy, r=r, r2=r ** 2, t=t, p=p, lo=lo, hi=hi, z=z, zse=zse,
                zlo=z - 1.96 * zse, zhi=z + 1.96 * zse, scipy=st.pearsonr(x, y))


# gallery of scatterplots (n = 40 each)
def gallery():
    rng = np.random.default_rng(606)
    n = 40
    out = {}
    x = rng.normal(0, 1, n)
    out["a"] = (x, exact_r(x, 0.90, rng))
    x = rng.normal(0, 1, n)
    out["b"] = (x, exact_r(x, 0.50, rng))
    x = rng.normal(0, 1, n)
    out["c"] = (x, exact_r(x, 0.0, rng))
    # U-shape: symmetric x, y = x^2 + noise orthogonal to x -> r = 0
    h = rng.uniform(0.05, 2.0, n // 2)
    x = np.concatenate([h, -h])
    noise = rng.normal(0, 0.35, n)
    noise -= noise.mean()
    noise -= (noise @ x) / (x @ x) * x
    out["d"] = (x, x ** 2 + noise)
    # one outlier: 39 points with r = 0 + one extreme point
    x = rng.normal(0, 1, n - 1)
    y = exact_r(x, 0.0, rng)
    out["e"] = (np.append(x, 8.0), np.append(y, 8.0))
    # restricted range: full r = 0.80, subset -0.6 < x < 0.6
    x = rng.normal(0, 1, 120)
    y = exact_r(x, 0.80, rng)
    out["f"] = (x, y)
    sub = (x > -0.6) & (x < 0.6)
    res = {k: np.corrcoef(*v)[0, 1] for k, v in out.items()}
    res["e_without"] = np.corrcoef(out["e"][0][:-1], out["e"][1][:-1])[0, 1]
    res["f_sub"] = np.corrcoef(x[sub], y[sub])[0, 1]
    res["f_nsub"] = int(sub.sum())
    return out, sub, res


# polypharmacy vs cost, confounded by comorbidity (for the warn box)
def poly_cost():
    rng = np.random.default_rng(61)
    n = 500
    cci = np.clip(rng.poisson(1.5, n), 0, 4)
    meds = np.clip(np.round(2 + 1.6 * cci + rng.normal(0, 1.3, n)), 0, None)
    cost = 80 + 60 * cci + rng.normal(0, 30, n)  # no direct effect of meds
    r_all = np.corrcoef(meds, cost)[0, 1]
    within = {int(c): (int((cci == c).sum()), np.corrcoef(meds[cci == c], cost[cci == c])[0, 1]) for c in range(5)}
    # partial correlation controlling for cci (residual method)
    rm = ols(cci, meds)["resid"]
    rc = ols(cci, cost)["resid"]
    r_part = np.corrcoef(rm, rc)[0, 1]
    return dict(r_all=r_all, within=within, r_part=r_part)


def anscombe():
    x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], float)
    ys = [np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
          np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
          np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
          np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])]
    x4 = np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], float)
    xs = [x, x, x, x4]
    return [(np.corrcoef(a, b)[0, 1], ols(a, b)["b"]) for a, b in zip(xs, ys)]


# ================================================================== 나. Spearman
def spearman8():
    rx = st.rankdata(AGE8)
    ry = st.rankdata(SBP8)
    d = rx - ry
    S = float((d ** 2).sum())
    n = 8
    rho = 1 - 6 * S / (n * (n ** 2 - 1))
    rho_p = np.corrcoef(rx, ry)[0, 1]
    # exact null distribution of S (all 8! permutations), as R does for n <= 9
    perm_S = np.array([((rx - np.array(pm)) ** 2).sum() for pm in itertools.permutations(range(1, 9))])
    p_one = (perm_S <= S).mean()
    p_exact = min(1.0, 2 * p_one)
    t, p_t = r_test(rho, n)
    # Kendall
    C = D = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(AGE8[j] - AGE8[i]) * np.sign(SBP8[j] - SBP8[i])
            C += s > 0
            D += s < 0
    tau = (C - D) / (n * (n - 1) / 2)
    kt = st.kendalltau(AGE8, SBP8)
    return dict(rx=rx, ry=ry, d=d, S=S, rho=rho, rho_p=rho_p, p_exact=p_exact, t=t, p_t=p_t, C=C, D=D, tau=tau,
                tau_p=kt.pvalue)


def emax_data():
    rng = np.random.default_rng(5)
    n = 20
    conc = np.sort(np.exp(rng.uniform(np.log(0.1), np.log(100), n)))
    eff = 100 * conc / (2 + conc) + rng.normal(0, 3, n)
    return conc, eff


def outlier_data():
    rng = np.random.default_rng(7)
    n = 15
    drinks = np.round(rng.uniform(0, 14, n))
    ast = np.round(exact_r(drinks, -0.05, rng, scale=5.5, loc=26))
    return np.append(drinks, 40.0), np.append(ast, 168.0)


def pair_stats(x, y):
    pr = st.pearsonr(x, y)
    sr = st.spearmanr(x, y)
    return dict(r=pr.statistic, pr=pr.pvalue, rho=sr.statistic, prho=sr.pvalue)


# Spearman correlation matrix for the paper box (n = 186)
MAT_NAMES = ["MARS-5", "BMQ necessity", "BMQ concerns", "No. of medications", "Age"]


def corr_matrix_data():
    R = np.array([
        [1.00, 0.26, -0.38, -0.08, 0.17],
        [0.26, 1.00, 0.12, 0.30, 0.20],
        [-0.38, 0.12, 1.00, 0.14, -0.06],
        [-0.08, 0.30, 0.14, 1.00, 0.40],
        [0.17, 0.20, -0.06, 0.40, 1.00]])
    assert np.all(np.linalg.eigvalsh(R) > 0)
    rng = np.random.default_rng(186)
    n = 186
    Z = rng.multivariate_normal(np.zeros(5), R, n)
    U = st.norm.cdf(Z)
    # MARS-5: ceiling effect (most patients 20-25)
    mars = np.clip(np.round(25 - st.gamma.ppf(1 - U[:, 0], 1.3, scale=2.2)), 5, 25)
    nec = np.clip(np.round(18 + 3.6 * Z[:, 1]), 5, 25)
    con = np.clip(np.round(14 + 3.8 * Z[:, 2]), 5, 25)
    meds = np.clip(np.round(np.exp(1.55 + 0.42 * Z[:, 3])), 1, None)
    age = np.round(66 + 9.5 * Z[:, 4])
    D = np.column_stack([mars, nec, con, meds, age])
    k = D.shape[1]
    rho = np.ones((k, k))
    pv = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                s = st.spearmanr(D[:, i], D[:, j])
                rho[i, j] = s.statistic
                pv[i, j] = s.pvalue
    return D, rho, pv


# ================================================================== 다. simple regression
def simple8():
    x, y = AGE8, SBP8
    fit = ols(x, y)
    b0, b1 = fit["b"]
    P = pearson8()
    # an arbitrary 'eyeballed' line for comparison
    alt = (95.0, 0.65)
    alt_res = y - (alt[0] + alt[1] * x)
    ssr = float(((fit["fit"] - y.mean()) ** 2).sum())
    # reverse regression (age on sbp)
    rev = ols(y, x)
    cen = ols(x - 60, y)
    return dict(fit=fit, b0=b0, b1=b1, P=P, alt=alt, alt_sse=float((alt_res ** 2).sum()), ssr=ssr,
                b1_from_r=P["r"] * P["sy"] / P["sx"], rev=rev, cen=cen)


def bp100():
    rng = np.random.default_rng(1006)
    n = 100
    age = np.round(rng.uniform(35, 80, n))
    sbp = np.round(97 + 0.55 * age + rng.normal(0, 13, n))
    return age, sbp


def simple100():
    age, sbp = bp100()
    fit = ols(age, sbp)
    fit10 = ols(age / 10, sbp)
    cen = ols(age - 60, sbp)
    r = np.corrcoef(age, sbp)[0, 1]
    t_r, p_r = r_test(r, len(age))
    q = np.percentile(fit["resid"], [0, 25, 50, 75, 100])
    # CI and PI at chosen ages
    Xi = fit["XtX_inv"]
    pts = {}
    for a in (40, 60, 70, 90):
        x0 = np.array([1, a])
        yhat = x0 @ fit["b"]
        se_m = fit["sigma"] * np.sqrt(x0 @ Xi @ x0)
        se_p = fit["sigma"] * np.sqrt(1 + x0 @ Xi @ x0)
        tc = fit["tcrit"]
        pts[a] = dict(yhat=yhat, se_m=se_m, se_p=se_p, ci=(yhat - tc * se_m, yhat + tc * se_m),
                      pi=(yhat - tc * se_p, yhat + tc * se_p))
    return dict(age=age, sbp=sbp, fit=fit, fit10=fit10, cen=cen, r=r, t_r=t_r, p_r=p_r, rq=q, pts=pts,
                sw=st.shapiro(fit["resid"]))


def resid_patterns():
    rng = np.random.default_rng(33)
    n = 100
    x1 = rng.uniform(30, 85, n)
    y1 = np.exp(3.2 + 0.035 * x1 + rng.normal(0, 0.55, n))  # cost-like, variance grows with mean
    x2 = np.exp(rng.uniform(np.log(0.2), np.log(40), n))
    y2 = 100 * x2 / (2 + x2) + rng.normal(0, 4, n)  # Emax curve fitted by a straight line
    return (x1, y1), (x2, y2)


# ================================================================== 라. multiple regression
INST = ["clinic", "hospital", "tertiary"]


def htn240(seed=26):
    rng = np.random.default_rng(seed)
    n = 240
    age = np.round(np.clip(rng.normal(64, 10, n), 40, 90))
    fem = (rng.random(n) < 0.5).astype(float)
    dm = (rng.random(n) < 0.3).astype(float)
    height = np.where(fem == 1, rng.normal(157, 5.5, n), rng.normal(170, 6, n))
    bmi = rng.normal(25, 3.2, n)
    weight = np.round(bmi * (height / 100) ** 2, 1)
    bmi = weight / (height / 100) ** 2
    base = np.round(rng.normal(152, 14, n) + 0.2 * (age - 64))
    cls = np.clip(np.round(1 + 0.055 * (base - 140) + 0.5 * dm + rng.normal(0, 0.55, n)), 1, 4)
    adh = (rng.random(n) < 0.65).astype(float)
    inst = rng.choice(3, n, p=[0.55, 0.25, 0.20])
    fu = np.round(58 + 0.62 * base - 4.0 * cls - 5 * adh + 0.12 * (age - 64) + 2.5 * dm - 1.0 * fem
                  + 0.3 * (bmi - 25) + 1.5 * (inst == 1) + 2.5 * (inst == 2) + rng.normal(0, 10, n))
    return dict(n=n, age=age, fem=fem, dm=dm, height=height, weight=weight, bmi=bmi, base=base, cls=cls,
                adh=adh, inst=inst, fu=fu)


def multi240():
    d = htn240()
    y = d["fu"]
    h1 = (d["inst"] == 1).astype(float)
    h2 = (d["inst"] == 2).astype(float)
    cols = dict(cls=d["cls"], base10=d["base"] / 10, age10=d["age"] / 10, fem=d["fem"], dm=d["dm"],
                bmi=d["bmi"], adh=d["adh"], hosp=h1, tert=h2)
    names = list(cols)
    X = np.column_stack([cols[k] for k in names])
    full = ols(X, y, names)
    crude = {}
    for k in names:
        if k in ("hosp", "tert"):
            crude[k] = ols(np.column_stack([h1, h2]), y)
        else:
            crude[k] = ols(cols[k], y)
    m1 = ols(d["cls"], y)
    m2 = ols(np.column_stack([d["cls"], d["base"]]), y)
    r_cls_base = np.corrcoef(d["cls"], d["base"])[0, 1]
    vifs = vif(X)
    # multicollinearity demo: add body weight next to BMI
    Xw = np.column_stack([X, d["weight"]])
    fullw = ols(Xw, y, names + ["weight"])
    vifw = vif(Xw)
    r_bmi_wt = np.corrcoef(d["bmi"], d["weight"])[0, 1]
    # overfitting demo: add 10 pure-noise variables
    rng = np.random.default_rng(99)
    noise = rng.normal(size=(len(y), 10))
    full_noise = ols(np.column_stack([X, noise]), y)
    # joint F test for institution type (2 dummies)
    red = ols(np.delete(X, [names.index("hosp"), names.index("tert")], axis=1), y)
    F_inst = ((red["rss"] - full["rss"]) / 2) / (full["rss"] / full["df"])
    p_inst = st.f.sf(F_inst, 2, full["df"])
    # standardized coefficients
    sy = y.std(ddof=1)
    std_b = {k: full["b"][i + 1] * X[:, i].std(ddof=1) / sy for i, k in enumerate(names)}
    # group means of fu by number of classes (for text)
    grp = {int(c): (int((d["cls"] == c).sum()), y[d["cls"] == c].mean(), d["base"][d["cls"] == c].mean())
           for c in (1, 2, 3, 4)}
    # baseline tertiles for figure
    return dict(d=d, X=X, y=y, names=names, full=full, crude=crude, m1=m1, m2=m2, r_cls_base=r_cls_base,
                vifs=vifs, fullw=fullw, vifw=vifw, r_bmi_wt=r_bmi_wt, full_noise=full_noise, F_inst=F_inst,
                p_inst=p_inst, std_b=std_b, sy=sy, grp=grp,
                inst_n=[int((d["inst"] == i).sum()) for i in range(3)])


def cost400():
    rng = np.random.default_rng(404)
    n = 400
    age = np.round(np.clip(rng.normal(63, 11, n), 30, 90))
    cci = np.clip(rng.poisson(1.6, n), 0, 6)
    adh = (rng.random(n) < 0.6).astype(float)
    lc = 5.0 + 0.012 * (age - 63) + 0.19 * cci - 0.14 * adh + rng.normal(0, 0.9, n)
    cost = np.round(np.exp(lc), 1)  # 10,000 KRW (만원)
    X = np.column_stack([adh, age / 10, cci])
    fit = ols(X, np.log(cost), ["adh", "age10", "cci"])
    pct = {nm: (np.exp(fit["b"][i + 1]) - 1, np.exp(fit["lo"][i + 1]) - 1, np.exp(fit["hi"][i + 1]) - 1)
           for i, nm in enumerate(["adh", "age10", "cci"])}
    am = (cost[adh == 1].mean(), cost[adh == 0].mean())
    gm = (np.exp(np.log(cost[adh == 1]).mean()), np.exp(np.log(cost[adh == 0]).mean()))
    med = (np.median(cost[adh == 1]), np.median(cost[adh == 0]))
    # smearing factor (Duan)
    smear = np.exp(fit["resid"]).mean()
    return dict(n=n, age=age, cci=cci, adh=adh, cost=cost, fit=fit, pct=pct, am=am, gm=gm, med=med,
                n_adh=int(adh.sum()), smear=smear, skew=st.skew(cost), skew_log=st.skew(np.log(cost)))


def ancova_rct():
    rng = np.random.default_rng(2030)
    n = 60
    trt = np.repeat([1.0, 0.0], n)
    base = np.round(rng.normal(8.3, 0.9, 2 * n), 1)
    fu = np.round(1.7 + 0.78 * base - 0.45 * trt + rng.normal(0, 0.55, 2 * n), 1)
    chg = fu - base
    a_fu = ols(trt, fu)
    a_chg = ols(trt, chg)
    a_anc = ols(np.column_stack([trt, base]), fu)
    r_bf = np.corrcoef(base, fu)[0, 1]
    means = dict(base=(base[trt == 1].mean(), base[trt == 0].mean()), fu=(fu[trt == 1].mean(), fu[trt == 0].mean()),
                 sd_base=(base[trt == 1].std(ddof=1), base[trt == 0].std(ddof=1)),
                 sd_fu=(fu[trt == 1].std(ddof=1), fu[trt == 0].std(ddof=1)),
                 chg=(chg[trt == 1].mean(), chg[trt == 0].mean()))
    return dict(trt=trt, base=base, fu=fu, a_fu=a_fu, a_chg=a_chg, a_anc=a_anc, r_bf=r_bf, means=means)


# ================================================================== printing
def main():
    np.set_printoptions(precision=4, suppress=True)
    P = pearson8()
    print("== 가. Pearson, 8 patients")
    print("mean age", P["mx"], "mean sbp", P["my"])
    print("dx", P["dx"], "\ndy", P["dy"], "\nprod", P["prod"])
    print(f"Sxy {P['Sxy']:.0f} Sxx {P['Sxx']:.0f} Syy {P['Syy']:.0f} cov {P['cov']:.2f} sx {P['sx']:.2f} sy {P['sy']:.3f}")
    print(f"r {P['r']:.4f} r2 {P['r2']:.4f} t {P['t']:.4f} p {P['p']:.6f} | Fisher z {P['z']:.4f} se {P['zse']:.4f} "
          f"z-CI {P['zlo']:.4f} {P['zhi']:.4f} -> r CI {P['lo']:.7f} {P['hi']:.7f}; scipy {P['scipy']}")
    print("cov check", np.cov(AGE8, SBP8)[0, 1], "product of SDs", P["sx"] * P["sy"], "cov/(sx sy)", P["cov"] / (P["sx"] * P["sy"]))
    g, sub, gr = gallery()
    print("gallery r:", {k: round(v, 3) for k, v in gr.items()})
    for label, r, n in (("warfarin weight", 0.42, 112), ("warfarin age", -0.35, 112), ("ecological", 0.71, 17)):
        lo, hi, z, se = fisher_ci(r, n)
        t, p = r_test(r, n)
        print(f"{label}: r={r} n={n} z={z:.4f} se={se:.4f} CI {lo:.3f} {hi:.3f}  t={t:.3f} df={n - 2} p={p:.2e}  r2={r * r:.3f}")
    t, p = r_test(0.07, 1000)
    print(f"n=1000 r=0.07: t={t:.3f} p={p:.4f}")
    t, p = r_test(0.30, 30)
    print(f"n=30 r=0.30: t={t:.3f} p={p:.4f}")
    lo, hi, *_ = fisher_ci(0.30, 30)
    print(f"   CI {lo:.3f} {hi:.3f}")
    n_req = ((st.norm.ppf(0.975) + st.norm.ppf(0.8)) / np.arctanh(0.3)) ** 2 + 3
    print("n for r=0.3 80% power", n_req)
    pc = poly_cost()
    print("poly-cost r_all", round(pc["r_all"], 3), "within", {k: (v[0], round(v[1], 3)) for k, v in pc["within"].items()},
          "partial", round(pc["r_part"], 3))
    print("anscombe", [(round(a, 3), np.round(b, 3)) for a, b in anscombe()])

    print("\n== 나. Spearman")
    S8 = spearman8()
    print("ranks x", S8["rx"], "ranks y", S8["ry"], "d", S8["d"], "S", S8["S"])
    print(f"rho {S8['rho']:.7f} (pearson on ranks {S8['rho_p']:.7f}) exact p {S8['p_exact']:.5f} t-approx t {S8['t']:.3f} p {S8['p_t']:.4f}")
    print(f"Kendall C {S8['C']} D {S8['D']} tau {S8['tau']:.4f} p {S8['tau_p']:.5f}")
    c, e = emax_data()
    print("emax conc", np.round(c, 2), "\n eff", np.round(e, 1), "\n", pair_stats(c, e))
    x, y = outlier_data()
    print("outlier drinks", x, "\n ast", y, "\n with:", pair_stats(x, y), "\n without:", pair_stats(x[:-1], y[:-1]))
    print("ranks of outlier data x", st.rankdata(x), "\n y", st.rankdata(y))
    D, rho, pv = corr_matrix_data()
    print("matrix means", D.mean(0), "medians", np.median(D, 0), "min", D.min(0), "max", D.max(0))
    print("rho\n", np.round(rho, 3), "\np\n", np.round(pv, 4))
    print("pearson matrix\n", np.round(np.corrcoef(D.T), 3))

    print("\n== 다. simple regression (8 patients)")
    s8 = simple8()
    f = s8["fit"]
    print(f"b0 {s8['b0']:.4f} b1 {s8['b1']:.5f} b1 from r {s8['b1_from_r']:.5f}")
    print("fitted", np.round(f["fit"], 2), "\nresid", np.round(f["resid"], 2))
    print(f"SSE {f['rss']:.3f} SSR {s8['ssr']:.3f} SST {f['tss']:.3f} R2 {f['R2']:.4f} sigma {f['sigma']:.4f}")
    print(f"se b1 {f['se'][1]:.5f} t {f['t'][1]:.4f} p {f['p'][1]:.6f} CI {f['lo'][1]:.4f} {f['hi'][1]:.4f} F {f['F']:.3f} pF {f['pF']:.6f} tcrit {f['tcrit']:.4f}")
    print(f"se b0 {f['se'][0]:.4f} t {f['t'][0]:.3f}")
    print(f"alt line {s8['alt']} SSE {s8['alt_sse']:.2f}")
    print(f"reverse slope (age on sbp) {s8['rev']['b'][1]:.4f}, 1/rev {1 / s8['rev']['b'][1]:.4f}, product {s8['rev']['b'][1] * s8['b1']:.4f}")
    print(f"centered intercept {s8['cen']['b'][0]:.4f} slope {s8['cen']['b'][1]:.5f}")

    print("\n== 다. simple regression (n = 100)")
    S = simple100()
    f = S["fit"]
    print("age range", S["age"].min(), S["age"].max(), "mean age", S["age"].mean(), "sd", S["age"].std(ddof=1),
          "sbp mean", S["sbp"].mean(), "sd", S["sbp"].std(ddof=1))
    print("resid quantiles", np.round(S["rq"], 3))
    for i, nm in enumerate(["(Intercept)", "age"]):
        print(f"{nm}: est {f['b'][i]:.5f} se {f['se'][i]:.5f} t {f['t'][i]:.3f} p {f['p'][i]:.3e} CI {f['lo'][i]:.4f} {f['hi'][i]:.4f}")
    print(f"sigma {f['sigma']:.4f} df {f['df']} R2 {f['R2']:.4f} adj {f['adjR2']:.4f} F {f['F']:.3f} pF {f['pF']:.4e}")
    print(f"r {S['r']:.4f} r^2 {S['r'] ** 2:.4f} t from r {S['t_r']:.3f} p {S['p_r']:.3e}; t^2 {f['t'][1] ** 2:.3f}")
    f10 = S["fit10"]
    print(f"per 10y: b {f10['b'][1]:.4f} se {f10['se'][1]:.4f} CI {f10['lo'][1]:.4f} {f10['hi'][1]:.4f}")
    fc = S["cen"]
    print(f"centered at 60: intercept {fc['b'][0]:.4f} se {fc['se'][0]:.4f} CI {fc['lo'][0]:.3f} {fc['hi'][0]:.3f}")
    for a, v in S["pts"].items():
        print(f"age {a}: yhat {v['yhat']:.2f} se_m {v['se_m']:.3f} CI {v['ci'][0]:.2f} {v['ci'][1]:.2f} se_p {v['se_p']:.3f} PI {v['pi'][0]:.2f} {v['pi'][1]:.2f}")
    print("shapiro resid", S["sw"])

    print("\n== 라. multiple regression (n = 240)")
    M = multi240()
    d = M["d"]
    print("n", d["n"], "cls counts", np.bincount(d["cls"].astype(int)), "inst n", M["inst_n"], "fem", d["fem"].sum(), "dm", d["dm"].sum(), "adh", d["adh"].sum())
    print("means: age", d["age"].mean(), "base", d["base"].mean(), "sd", d["base"].std(ddof=1), "fu", d["fu"].mean(), "sd", d["fu"].std(ddof=1), "bmi", d["bmi"].mean(), "cls", d["cls"].mean())
    print("group means by cls (n, fu mean, base mean):", {k: (v[0], round(v[1], 1), round(v[2], 1)) for k, v in M["grp"].items()})
    m1, m2 = M["m1"], M["m2"]
    print(f"model1 cls: b {m1['b'][1]:.3f} se {m1['se'][1]:.3f} CI {m1['lo'][1]:.2f} {m1['hi'][1]:.2f} p {m1['p'][1]:.4g} R2 {m1['R2']:.4f}")
    print(f"model2 cls: b {m2['b'][1]:.3f} se {m2['se'][1]:.3f} CI {m2['lo'][1]:.2f} {m2['hi'][1]:.2f} p {m2['p'][1]:.4g}; base b {m2['b'][2]:.4f} se {m2['se'][2]:.4f}; R2 {m2['R2']:.4f}; b0 {m2['b'][0]:.3f}")
    print(f"r(cls, base) {M['r_cls_base']:.3f}; VIF 2-var = {1 / (1 - M['r_cls_base'] ** 2):.3f}")
    base_on_cls = ols(d["cls"], d["base"])
    print(f"base on cls slope {base_on_cls['b'][1]:.3f}; decomposition: m2 cls + m2 base*that = {m2['b'][1] + m2['b'][2] * base_on_cls['b'][1]:.3f}")
    F = M["full"]
    print("FULL model: n", F["n"], "df", F["df"], f"R2 {F['R2']:.4f} adj {F['adjR2']:.4f} F {F['F']:.3f} pF {F['pF']:.3e} sigma {F['sigma']:.3f}")
    for i, nm in enumerate(["(Intercept)"] + M["names"]):
        c = M["crude"].get(nm)
        cs = ""
        if c is not None:
            j = 1 if nm != "tert" else 2
            cs = f" | crude {c['b'][j]:.2f} ({c['lo'][j]:.2f}, {c['hi'][j]:.2f}) p {c['p'][j]:.3g}"
        print(f"  {nm:12s} {F['b'][i]:8.3f} se {F['se'][i]:.3f} t {F['t'][i]:7.3f} p {F['p'][i]:.4g}  CI {F['lo'][i]:.2f} {F['hi'][i]:.2f}{cs}")
    print("  VIF", dict(zip(M["names"], np.round(M["vifs"], 2))))
    print(f"  joint F inst {M['F_inst']:.3f} p {M['p_inst']:.4f}")
    print("  std b", {k: round(v, 3) for k, v in M["std_b"].items()}, "sd fu", round(M["sy"], 2), "sd cls", round(d["cls"].std(ddof=1), 3), "sd base10", round((d["base"] / 10).std(ddof=1), 3))
    FW = M["fullw"]
    ib = M["names"].index("bmi") + 1
    print(f"  weight demo: r(bmi,wt) {M['r_bmi_wt']:.3f}; VIF bmi {M['vifw'][ib - 1]:.2f} weight {M['vifw'][-1]:.2f}; "
          f"bmi b {FW['b'][ib]:.3f} se {FW['se'][ib]:.3f} p {FW['p'][ib]:.3f} (vs {F['b'][ib]:.3f} se {F['se'][ib]:.3f} p {F['p'][ib]:.3f}); weight b {FW['b'][-1]:.3f} se {FW['se'][-1]:.3f} p {FW['p'][-1]:.3f}")
    FN = M["full_noise"]
    print(f"  noise demo: R2 {F['R2']:.4f}->{FN['R2']:.4f}; adj {F['adjR2']:.4f}->{FN['adjR2']:.4f}; df {FN['df']}; noise p's {np.round(FN['p'][-10:], 3)}")

    print("\n== 라. log cost (n = 400)")
    C = cost400()
    f = C["fit"]
    print("n adherent", C["n_adh"], "arith means", np.round(C["am"], 1), "geo means", np.round(C["gm"], 1), "medians", C["med"],
          "skew", round(C["skew"], 2), "skew log", round(C["skew_log"], 2), "overall mean", C["cost"].mean(), "median", np.median(C["cost"]))
    for i, nm in enumerate(["(Intercept)", "adh", "age10", "cci"]):
        print(f"  {nm}: b {f['b'][i]:.4f} se {f['se'][i]:.4f} CI {f['lo'][i]:.4f} {f['hi'][i]:.4f} p {f['p'][i]:.3g}  exp {np.exp(f['b'][i]):.4f} ({np.exp(f['lo'][i]):.4f}-{np.exp(f['hi'][i]):.4f})")
    print(f"  R2 {f['R2']:.4f} adj {f['adjR2']:.4f} sigma {f['sigma']:.4f} df {f['df']} smear {C['smear']:.4f}")
    for b in (0.05, 0.10, 0.14, 0.20, 0.50, -0.50, 0.693):
        print(f"   beta {b}: exp-1 = {np.exp(b) - 1:.4f}")

    print("\n== 라. ANCOVA RCT (60 vs 60)")
    A = ancova_rct()
    m = A["means"]
    print("means", {k: np.round(v, 3) for k, v in m.items()}, "r(base,fu)", round(A["r_bf"], 3))
    for nm in ("a_fu", "a_chg", "a_anc"):
        g = A[nm]
        print(f"  {nm}: b {g['b'][1]:.4f} se {g['se'][1]:.4f} CI {g['lo'][1]:.3f} {g['hi'][1]:.3f} p {g['p'][1]:.4g} df {g['df']}")
    print(f"  ancova baseline coef {A['a_anc']['b'][2]:.4f}")


def python_boxes():
    """Numbers shown in the '파이썬 출력' boxes (needs: source /home/claude/pylibs/env.sh)."""
    import pandas as pd
    import statsmodels.formula.api as smf
    res = st.pearsonr(AGE8, SBP8)
    ci = res.confidence_interval(confidence_level=0.95)
    print("pearsonr", res.statistic, res.pvalue, "CI", ci.low, ci.high)
    rs = st.spearmanr(AGE8, SBP8)
    print("spearmanr", rs.statistic, rs.pvalue)
    perm = st.permutation_test((SBP8,), lambda y: st.spearmanr(AGE8, y).statistic,
                               n_resamples=np.inf, permutation_type="pairings")
    print("exact permutation p", perm.pvalue)
    print("kendalltau", st.kendalltau(AGE8, SBP8))
    age, sbp = bp100()
    fit = smf.ols("sbp ~ age", data=pd.DataFrame({"age": age, "sbp": sbp})).fit()
    print(fit.summary())
    print("p(age)", fit.pvalues["age"], "resid SE", np.sqrt(fit.scale))


if __name__ == "__main__":
    main()
    try:
        print("\n== 파이썬 출력 상자")
        python_boxes()
    except ImportError as e:
        print("  (skipped:", e, ")")


# ================================================================== 라. 추가 (2026-09-30 rewrite review)
# Table of 6-month SBP by baseline-SBP tertile x number of classes (1, 2, 3+) shown in 라 절
# '기저 혈압이 비슷한 환자끼리 나눠 보기', plus model-2 predictions at baseline 150 / 170 mmHg.
def tertile_block():
    M = multi240()
    d = M["d"]
    y, base, c3 = d["fu"], d["base"], np.minimum(d["cls"], 3)
    q = np.quantile(base, [1 / 3, 2 / 3])  # 146, 159 -> tertiles 110-145, 146-158, 159-192
    strata = [("low", base < q[0]), ("mid", (base >= q[0]) & (base < q[1])), ("high", base >= q[1])]
    print("\n== 라. baseline-SBP tertile table (6-month SBP mean, n) and model-2 predictions")
    print("tertile cut points", q)
    for nm, m in strata + [("all", np.ones(len(y), bool))]:
        cells = []
        for c in (1, 2, 3):
            mm = m & (c3 == c)
            cells.append(f"cls{c}: {y[mm].mean():.1f} ({mm.sum()}) base {base[mm].mean():.1f}" if mm.sum() else f"cls{c}: -")
        print(f"  {nm:4s} n={m.sum():3d} range {base[m].min():.0f}-{base[m].max():.0f} | " + " | ".join(cells))
    b = M["m2"]["b"]
    for bs in (150, 170):
        print(f"  model 2 prediction at baseline {bs}:", [round(b[0] + b[1] * k + b[2] * bs, 2) for k in (1, 2, 3)])
    print(f"  shift 150 -> 170: 20 x {b[2]:.4f} = {20 * b[2]:.2f}")
    pc = poly_cost()
    print(f"  (가 절 use box) partial r(polypharmacy, cost | CCI) = {pc['r_part']:.3f}")


if __name__ == "__main__":
    tertile_block()
