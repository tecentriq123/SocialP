"""Logistic-regression helpers and data sets for chapter 9 (numpy/scipy only).

statsmodels is not available, so everything is implemented here and cross-checked
against sklearn (penalty=None) in nums_ch09.py:
  fit_logit()      unpenalized logistic MLE by Newton-Raphson (optional offset)
  glm_R()          emulation of R's glm.fit IRLS for binomial/logit (iterations, SE from
                   the last iteration's weights, deviance-based stopping rule) -> R-style output
  profile_ci()     profile-likelihood CI of one coefficient (what R's confint() gives)
  lr_test()        likelihood-ratio test of nested models
  score_global()   score test of all slopes = 0 (SAS 'Testing Global Null Hypothesis')
  firth()          Firth penalized likelihood with profile penalized-likelihood CI (logistf default)
  fit_poisson()    Poisson log-link MLE with robust (HC0 sandwich) SE -> modified Poisson
  fit_logbin()     log-binomial MLE (Fisher scoring with step-halving)
  auc_delong()     C statistic (Mann-Whitney) with DeLong SE
  hosmer_lemeshow()  deciles-of-risk test (quantile breaks like ResourceSelection::hoslem.test)
  cal_int_slope()  calibration-in-the-large (offset) and calibration slope
  rcs_basis()      Harrell restricted cubic spline basis
Data sets:
  vanco()          150 patients: vancomycin trough and AKI (basic examples)
  cohort()         2,400 older new users of SGLT2i or DPP-4i, 1-year all-cause hospitalization
  external()       1,600 patients from another setting (external validation)
  irae()           80 RCC patients on immune checkpoint inhibitors (separation example)
"""
import numpy as np
import scipy.stats as st
from scipy.optimize import brentq

Z = st.norm.ppf(0.975)
Q95 = st.chi2.ppf(0.95, 1)


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ core MLE
def loglik(beta, X, y, offset=None):
    eta = X @ beta + (0 if offset is None else offset)
    # numerically stable: sum y*eta - log(1+exp(eta))
    return float(np.sum(y * eta - np.logaddexp(0, eta)))


def fit_logit(X, y, offset=None, maxit=100, tol=1e-12, beta0=None):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    b = np.zeros(k) if beta0 is None else np.array(beta0, float)
    ll_old = loglik(b, X, y, off)
    for it in range(1, maxit + 1):
        p = expit(X @ b + off)
        W = p * (1 - p)
        g = X.T @ (y - p)
        I = X.T @ (X * W[:, None])
        step = np.linalg.solve(I, g)
        b_new = b + step
        ll = loglik(b_new, X, y, off)
        h = 1.0
        while ll < ll_old - 1e-10 and h > 1e-6:
            h /= 2
            b_new = b + h * step
            ll = loglik(b_new, X, y, off)
        b = b_new
        if abs(ll - ll_old) < tol and np.max(np.abs(step)) < 1e-9:
            break
        ll_old = ll
    p = expit(X @ b + off)
    I = X.T @ (X * (p * (1 - p))[:, None])
    cov = np.linalg.inv(I)
    ll = loglik(b, X, y, off)
    return dict(beta=b, cov=cov, se=np.sqrt(np.diag(cov)), ll=ll, dev=-2 * ll, p=p, iters=it,
                z=b / np.sqrt(np.diag(cov)), pval=2 * st.norm.sf(np.abs(b / np.sqrt(np.diag(cov)))))


def or_ci(b, se):
    return np.exp(b), np.exp(b - Z * se), np.exp(b + Z * se)


def lr_test(full, reduced, df):
    stat = 2 * (full["ll"] - reduced["ll"])
    return stat, st.chi2.sf(stat, df)


def profile_ci(X, y, j, fit=None, level=0.95, offset=None):
    """profile-likelihood CI for coefficient j (natural log scale)."""
    if fit is None:
        fit = fit_logit(X, y, offset)
    crit = st.chi2.ppf(level, 1)
    llmax = fit["ll"]
    others = [k for k in range(X.shape[1]) if k != j]
    Xo = X[:, others]
    off0 = np.zeros(len(y)) if offset is None else offset

    def f(b):
        r = fit_logit(Xo, y, offset=off0 + b * X[:, j], beta0=fit["beta"][others])
        return 2 * (llmax - r["ll"]) - crit

    bj, se = fit["beta"][j], fit["se"][j]
    m = 3.0
    while f(bj - m * se) < 0 or f(bj + m * se) < 0:
        m += 1.0
    lo = brentq(f, bj - m * se, bj, xtol=1e-10)
    hi = brentq(f, bj, bj + m * se, xtol=1e-10)
    return lo, hi


def score_global(X, y):
    """score test of all non-intercept coefficients = 0 (X[:,0] is the intercept)."""
    p0 = y.mean()
    U = X.T @ (y - p0)
    I = X.T @ X * p0 * (1 - p0)
    s = float(U @ np.linalg.solve(I, U))
    return s, X.shape[1] - 1, st.chi2.sf(s, X.shape[1] - 1)


# ------------------------------------------------------------------ R glm.fit emulation
def glm_R(X, y, maxit=25, eps=1e-8):
    """binomial/logit IRLS exactly as stats::glm.fit (weights = 1)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)

    def devres(y, mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(y > 0, y * np.log(y / mu), 0.0)
            b = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2 * (a + b)

    mu = (y + 0.5) / 2
    eta = logit(mu)
    devold = devres(y, mu).sum()
    conv = False
    for it in range(1, maxit + 1):
        mu_eta = mu * (1 - mu)
        z = eta + (y - mu) / mu_eta
        w = np.sqrt(mu_eta)          # sqrt(mu.eta^2 / variance)
        Q, R = np.linalg.qr(X * w[:, None])
        coef = np.linalg.solve(R, Q.T @ (z * w))
        eta = X @ coef
        mu = expit(eta)
        dev = devres(y, mu).sum()
        if abs(dev - devold) / (abs(dev) + 0.1) < eps:
            conv = True
            break
        devold = dev
    Rinv = np.linalg.inv(R)
    cov = Rinv @ Rinv.T
    se = np.sqrt(np.diag(cov))
    ybar = y.mean()
    nulldev = devres(y, np.full_like(y, ybar)).sum()
    epsm = 10 * np.finfo(float).eps
    warn01 = bool(np.any(mu > 1 - epsm) or np.any(mu < epsm))
    return dict(beta=coef, se=se, cov=cov, dev=dev, nulldev=nulldev, iters=it, conv=conv,
                aic=dev + 2 * X.shape[1], mu=mu, warn01=warn01,
                z=coef / se, pval=2 * st.norm.sf(np.abs(coef / se)))


# ------------------------------------------------------------------ Firth
def firth(X, y, fixed=None, beta0=None, maxit=200, tol=1e-10):
    """Firth-penalized MLE. fixed=(j, value) holds coefficient j at value (for profiling).
    returns beta and penalized log-likelihood l + 0.5 log|I|."""
    X = np.asarray(X, float)
    n, k = X.shape
    b = np.zeros(k) if beta0 is None else np.array(beta0, float)
    free = list(range(k))
    if fixed is not None:
        b[fixed[0]] = fixed[1]
        free.remove(fixed[0])

    def pll(b):
        p = expit(X @ b)
        I = X.T @ (X * (p * (1 - p))[:, None])
        return loglik(b, X, y) + 0.5 * np.linalg.slogdet(I)[1]

    cur = pll(b)
    for it in range(maxit):
        p = expit(X @ b)
        W = p * (1 - p)
        I = X.T @ (X * W[:, None])
        Iinv = np.linalg.inv(I)
        h = np.einsum("ij,jk,ik->i", X * np.sqrt(W)[:, None], Iinv, X * np.sqrt(W)[:, None])
        U = X.T @ (y - p + h * (0.5 - p))
        Uf = U[free]
        If = I[np.ix_(free, free)]
        step = np.zeros(k)
        step[free] = np.linalg.solve(If, Uf)
        mx = np.max(np.abs(step))
        if mx > 5:
            step *= 5 / mx
        s = 1.0
        while True:
            nb = b + s * step
            new = pll(nb)
            if new >= cur - 1e-12 or s < 1e-8:
                break
            s /= 2
        b, old, cur = nb, cur, new
        if np.max(np.abs(s * step)) < tol:
            break
    p = expit(X @ b)
    I = X.T @ (X * (p * (1 - p))[:, None])
    return dict(beta=b, pll=cur, se=np.sqrt(np.diag(np.linalg.inv(I))))


def firth_profile_ci(X, y, j, fit=None, level=0.95):
    if fit is None:
        fit = firth(X, y)
    crit = st.chi2.ppf(level, 1)

    def f(v):
        r = firth(X, y, fixed=(j, v), beta0=fit["beta"])
        return 2 * (fit["pll"] - r["pll"]) - crit

    bj, se = fit["beta"][j], fit["se"][j]
    lo = brentq(f, bj - 8 * se, bj, xtol=1e-9)
    hi = brentq(f, bj, bj + 8 * se, xtol=1e-9)
    return lo, hi


def firth_lr(X, y, j, fit=None):
    if fit is None:
        fit = firth(X, y)
    r = firth(X, y, fixed=(j, 0.0), beta0=fit["beta"])
    s = 2 * (fit["pll"] - r["pll"])
    return s, st.chi2.sf(s, 1)


# ------------------------------------------------------------------ RR models
def fit_poisson(X, y, maxit=100):
    """Poisson log-link MLE + robust HC0 sandwich covariance (modified Poisson)."""
    X = np.asarray(X, float)
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean())
    for _ in range(maxit):
        mu = np.exp(X @ b)
        I = X.T @ (X * mu[:, None])
        step = np.linalg.solve(I, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = np.exp(X @ b)
    I = X.T @ (X * mu[:, None])
    Iinv = np.linalg.inv(I)
    meat = X.T @ (X * ((y - mu) ** 2)[:, None])
    rob = Iinv @ meat @ Iinv
    return dict(beta=b, se_model=np.sqrt(np.diag(Iinv)), se=np.sqrt(np.diag(rob)), mu=mu)


def fit_logbin(X, y, beta0=None, maxit=500):
    """log-binomial MLE by Fisher scoring, step-halving to keep fitted p < 1."""
    X = np.asarray(X, float)
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean())
    if beta0 is not None and np.all(np.exp(X @ np.asarray(beta0, float)) < 1):
        b = np.array(beta0, float)

    def ll(b):
        p = np.exp(X @ b)
        if np.any(p >= 1):
            return -np.inf
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

    cur = ll(b)
    for it in range(maxit):
        p = np.exp(X @ b)
        g = X.T @ ((y - p) / (1 - p))
        I = X.T @ (X * (p / (1 - p))[:, None])
        step = np.linalg.solve(I, g)
        s = 1.0
        while True:
            nb = b + s * step
            new = ll(nb)
            if new >= cur - 1e-12 or s < 1e-10:
                break
            s /= 2
        b, cur = nb, new
        if np.max(np.abs(s * step)) < 1e-11:
            break
    p = np.exp(X @ b)
    I = X.T @ (X * (p / (1 - p))[:, None])
    return dict(beta=b, se=np.sqrt(np.diag(np.linalg.inv(I))), ll=cur, maxp=float(p.max()), iters=it)


# ------------------------------------------------------------------ performance
def auc_delong(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    # placement values
    allv = np.concatenate([pos, neg])
    r_all = st.rankdata(allv)
    r_pos = st.rankdata(pos)
    r_neg = st.rankdata(neg)
    auc = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    V10 = (r_all[:m] - r_pos) / n          # for each positive: P(neg < pos) (+0.5 ties)
    V01 = 1 - (r_all[m:] - r_neg) / m      # for each negative: P(pos > neg)
    var = np.var(V10, ddof=1) / m + np.var(V01, ddof=1) / n
    se = np.sqrt(var)
    return auc, se, auc - Z * se, auc + Z * se


def roc_points(y, s):
    y = np.asarray(y).astype(int)
    thr = np.unique(s)[::-1]
    P, N = y.sum(), (1 - y).sum()
    fpr, tpr = [0.0], [0.0]
    for t in thr:
        pred = s >= t
        tpr.append((pred & (y == 1)).sum() / P)
        fpr.append((pred & (y == 0)).sum() / N)
    return np.array(fpr), np.array(tpr), thr


def confusion(y, p, t):
    pred = p >= t
    TP = int((pred & (y == 1)).sum()); FP = int((pred & (y == 0)).sum())
    FN = int((~pred & (y == 1)).sum()); TN = int((~pred & (y == 0)).sum())
    return dict(TP=TP, FP=FP, FN=FN, TN=TN, sens=TP / (TP + FN), spec=TN / (TN + FP),
                ppv=TP / (TP + FP) if TP + FP else np.nan, npv=TN / (TN + FN) if TN + FN else np.nan,
                acc=(TP + TN) / len(y), flagged=(TP + FP) / len(y))


def r_quantile(x, q):
    """R quantile type 7 (numpy default 'linear')."""
    return np.quantile(x, q)


def hosmer_lemeshow(y, p, g=10):
    br = r_quantile(p, np.linspace(0, 1, g + 1))
    # cut(..., include.lowest=TRUE): intervals (a,b], first [a,b]
    idx = np.searchsorted(br[1:-1], p, side="left")
    rows = []
    stat = 0.0
    for k in range(g):
        m = idx == k
        n = m.sum()
        o1, e1 = y[m].sum(), p[m].sum()
        o0, e0 = n - o1, n - e1
        stat += (o1 - e1) ** 2 / e1 + (o0 - e0) ** 2 / e0
        rows.append(dict(n=int(n), obs=float(o1), exp=float(e1), mean_p=float(p[m].mean()), obs_rate=float(o1 / n)))
    return stat, g - 2, st.chi2.sf(stat, g - 2), rows


def cal_int_slope(y, p):
    lp = logit(p)
    one = np.ones((len(y), 1))
    citl = fit_logit(one, y, offset=lp)
    sl = fit_logit(np.column_stack([one, lp]), y)
    return dict(citl=citl["beta"][0], citl_se=citl["se"][0], slope=sl["beta"][1], slope_se=sl["se"][1],
                slope_int=sl["beta"][0])


def rcs_knots(x, k):
    q = {3: [0.10, 0.5, 0.90], 4: [0.05, 0.35, 0.65, 0.95], 5: [0.05, 0.275, 0.5, 0.725, 0.95]}[k]
    return np.quantile(x, q)


def rcs_basis(x, knots):
    """Harrell's restricted cubic spline (rcspline.eval, norm=2): returns x and k-2 nonlinear terms."""
    t = np.asarray(knots, float)
    k = len(t)
    norm = (t[-1] - t[0]) ** 2
    cols = [np.asarray(x, float)]
    pp = lambda u: np.maximum(u, 0) ** 3
    for j in range(k - 2):
        c = (pp(x - t[j]) - pp(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
             + pp(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(c / norm)
    return np.column_stack(cols)


def brier(y, p):
    return float(np.mean((y - p) ** 2))


# ------------------------------------------------------------------ data sets
def vanco(seed=297, n=150):
    rng = np.random.default_rng(seed)
    trough = np.round(np.exp(rng.normal(np.log(14.5), 0.42, n)), 1)
    trough = np.clip(trough, 4.0, 45.0)
    p = expit(-4.6 + 0.21 * trough)
    aki = (rng.random(n) < p).astype(int)
    return trough, aki


def _cohort_draw(rng, n, age_mu=75.0, prior_p=0.20, hf_p=0.12, b0=-2.05, shift_int=0.0, prior_b=1.05,
                 hosp_scale=1.0, sg_bias=0.0):
    age = np.clip(np.round(rng.normal(age_mu, 6.2, n)), 65, 95)
    female = (rng.random(n) < 0.50).astype(int)
    u = rng.random(n)
    cci_cat = np.where(u < 0.35, 0, np.where(u < 0.75, 1, 2))        # 0 | 1-2 | >=3
    prior = (rng.random(n) < prior_p + 0.06 * (cci_cat == 2)).astype(int)
    hf = (rng.random(n) < hf_p + 0.05 * (cci_cat == 2)).astype(int)
    egfr = np.clip(np.round(rng.normal(72 - 0.6 * (age - 75) - 6 * (cci_cat == 2), 18, n)), 15, 120)
    lp_t = (-0.45 + sg_bias - 0.07 * (age - 75) + 0.035 * (egfr - 70) - 0.25 * (cci_cat == 1)
            - 0.55 * (cci_cat == 2) + 0.55 * hf - 0.25 * prior)
    sglt2 = (rng.random(n) < expit(lp_t)).astype(int)
    lp = (b0 + shift_int + hosp_scale * (0.036 * (age - 75) - 0.15 * female + 0.36 * (cci_cat == 1)
          + 0.80 * (cci_cat == 2) + prior_b * prior + 0.55 * hf + 0.035 * np.maximum(60 - egfr, 0))
          - 0.20 * sglt2 - 0.45 * sglt2 * hf)
    hosp = (rng.random(n) < expit(lp)).astype(int)
    return dict(age=age, female=female, cci=cci_cat, prior=prior, hf=hf, egfr=egfr, sglt2=sglt2, hosp=hosp)


def cohort(seed=111, n=2400):
    return _cohort_draw(np.random.default_rng(seed), n)


def external(seed=164, n=1600):
    # another setting: younger, fewer prior admissions, lower baseline risk, weaker predictor effects
    return _cohort_draw(np.random.default_rng(seed), n, age_mu=73.0, prior_p=0.15, hf_p=0.10, b0=-2.05,
                        shift_int=-0.30, prior_b=0.85, hosp_scale=0.80)


def design(d, spec="etio", age_center=75):
    """build design matrix. spec: 'etio' (with drug), 'pred' (no drug), 'crude_x' etc."""
    n = len(d["age"])
    one = np.ones(n)
    age10 = (d["age"] - age_center) / 10
    c12 = (d["cci"] == 1).astype(float); c3 = (d["cci"] == 2).astype(float)
    e1 = ((d["egfr"] >= 45) & (d["egfr"] < 60)).astype(float); e2 = (d["egfr"] < 45).astype(float)
    cols = {"(Intercept)": one, "sglt2": d["sglt2"].astype(float), "age10": age10, "female": d["female"].astype(float),
            "cci12": c12, "cci3": c3, "prior": d["prior"].astype(float), "hf": d["hf"].astype(float),
            "egfr4559": e1, "egfr45": e2}
    if spec == "etio":
        names = ["(Intercept)", "sglt2", "age10", "female", "cci12", "cci3", "prior", "hf", "egfr4559", "egfr45"]
    elif spec == "pred":
        names = ["(Intercept)", "age10", "female", "cci12", "cci3", "prior", "hf", "egfr4559", "egfr45"]
    else:
        names = spec
    return np.column_stack([cols[k] for k in names]), names


def irae():
    """80 RCC patients on ICI: autoimmune history (6, all with grade>=3 irAE), combination therapy."""
    rows = []
    # (autoimmune, combo, n, events)
    for a, c, n, e in [(1, 1, 4, 4), (1, 0, 2, 2), (0, 1, 30, 9), (0, 0, 44, 6)]:
        rows += [(a, c, 1)] * e + [(a, c, 0)] * (n - e)
    arr = np.array(rows, float)
    return arr[:, 0], arr[:, 1], arr[:, 2]
