"""Helpers and data sets for chapter 8 (numpy/scipy; the statsmodels boxes are cross-checked in nums_ch08.py).

  glm()        IRLS / Fisher scoring for GLMs: binomial (logit, log, identity), poisson (log),
               gamma (log), gaussian (identity); step-halving keeps the mean valid
  loglik()     log-likelihood of a fitted GLM (binomial / poisson)
  robust_se()  HC0 sandwich standard errors (modified Poisson regression)
  make_readmit()  the simulated discharge cohort (n = 1,500) used in sections 다-아
All results are printed and cross-checked in nums_ch08.py.
"""
import numpy as np
import scipy.stats as st

Z = st.norm.ppf(0.975)


def _link(link):
    if link == "logit":
        return (lambda m: np.log(m / (1 - m)), lambda e: 1 / (1 + np.exp(-e)), lambda m: 1 / (m * (1 - m)))
    if link == "log":
        return (np.log, np.exp, lambda m: 1 / m)
    if link == "identity":
        return (lambda m: m, lambda e: e, lambda m: np.ones_like(m))
    raise ValueError(link)


def _var(family, m):
    return {"binomial": m * (1 - m), "poisson": m, "gamma": m ** 2, "gaussian": np.ones_like(m)}[family]


def _valid(family, m):
    if family == "binomial":
        return np.all((m > 1e-10) & (m < 1 - 1e-10))
    if family in ("poisson", "gamma"):
        return np.all(m > 0)
    return True


def glm(X, y, family="binomial", link="logit", offset=None, start=None, maxit=200, tol=1e-12):
    """returns dict(beta, se, cov (model-based, dispersion-scaled for gamma/gaussian), mu, iters)"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    g, ginv, gprime = _link(link)
    if start is None:
        m0 = np.full(n, y.mean()) if family != "binomial" else np.full(n, np.clip(y.mean(), .05, .95))
        # least-squares start on link scale
        beta = np.linalg.lstsq(X, g(m0) - off, rcond=None)[0]
    else:
        beta = np.asarray(start, float)
    dev_old = np.inf
    for it in range(maxit):
        eta = X @ beta + off
        mu = ginv(eta)
        if not _valid(family, mu):
            raise RuntimeError("invalid start")
        gp = gprime(mu)
        w = 1 / (_var(family, mu) * gp ** 2)
        zz = eta - off + (y - mu) * gp
        XtW = X.T * w
        new = np.linalg.solve(XtW @ X, XtW @ zz)
        # step-halving to keep mean valid and deviance decreasing
        step = new - beta
        for _ in range(60):
            cand = beta + step
            mu_c = ginv(X @ cand + off)
            if _valid(family, mu_c):
                d_c = deviance(family, y, mu_c)
                if d_c <= dev_old + 1e-9 or not np.isfinite(dev_old):
                    break
            step = step / 2
        beta = cand
        dev = deviance(family, y, mu_c)
        if abs(dev_old - dev) < tol * (abs(dev) + 1):
            dev_old = dev
            break
        dev_old = dev
    eta = X @ beta + off
    mu = ginv(eta)
    gp = gprime(mu)
    w = 1 / (_var(family, mu) * gp ** 2)
    info = (X.T * w) @ X
    cov = np.linalg.inv(info)
    disp = 1.0
    if family in ("gamma", "gaussian"):
        disp = np.sum((y - mu) ** 2 / _var(family, mu)) / (n - k)  # Pearson dispersion (R default)
        cov = cov * disp
    return dict(beta=beta, se=np.sqrt(np.diag(cov)), cov=cov, mu=mu, iters=it + 1, dev=deviance(family, y, mu),
                disp=disp, X=X, y=y, family=family, link=link)


def deviance(family, y, mu):
    if family == "binomial":
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(y > 0, y * np.log(y / mu), 0.0)
            b = np.where(y < 1, (1 - y) * np.log((1 - y) / (1 - mu)), 0.0)
        return 2 * np.sum(a + b)
    if family == "poisson":
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2 * np.sum(a - (y - mu))
    if family == "gamma":
        return 2 * np.sum(-np.log(y / mu) + (y - mu) / mu)
    return np.sum((y - mu) ** 2)


def loglik(fit):
    y, mu = fit["y"], fit["mu"]
    if fit["family"] == "binomial":
        return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
    if fit["family"] == "poisson":
        from scipy.special import gammaln
        return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))
    raise ValueError


def robust_se(fit):
    """HC0 sandwich for a log-link Poisson/binomial working model"""
    X, y, mu = fit["X"], fit["y"], fit["mu"]
    g, ginv, gprime = _link(fit["link"])
    gp = gprime(mu)
    w = 1 / (_var(fit["family"], mu) * gp ** 2)
    bread = np.linalg.inv((X.T * w) @ X)
    u = X * ((y - mu) / (_var(fit["family"], mu) * gp))[:, None]
    meat = u.T @ u
    cov = bread @ meat @ bread
    return np.sqrt(np.diag(cov)), cov


def wald_rows(fit, names, se=None):
    se = fit["se"] if se is None else se
    out = []
    for nm, b, s in zip(names, fit["beta"], se):
        z = b / s
        out.append(dict(name=nm, b=b, se=s, z=z, p=2 * st.norm.sf(abs(z)), lo=b - Z * s, hi=b + Z * s))
    return out


# ------------------------------------------------------------------ simulated discharge cohort
AGE_C = 76


PAR = dict(ma_age=11.0, ma_fem=0.16, ma_cci=0.3, b0=-2.30, bage=0.05, bma=0.75)


def make_readmit(seed=581, n=1500, **kw):
    P = dict(PAR); P.update(kw)
    """65+ patients discharged from hospital (hypothetical claims cohort).
    poly   : >= 10 drugs prescribed at discharge (exposure)
    readm  : 30-day readmission (outcome)
    age, female, cci (0: CCI 0, 1: 1-2, 2: 3-4, 3: >=5), medaid (Medical Aid beneficiary)"""
    rng = np.random.default_rng(seed)
    medaid = (rng.random(n) < 0.16).astype(int)
    age = np.clip(np.round(rng.normal(78.0, 6.3, n) - P["ma_age"] * medaid), 65, 98).astype(int)
    female = (rng.random(n) < 0.54 + P["ma_fem"] * medaid).astype(int)
    lat = 0.055 * (age - AGE_C) + rng.normal(0, 1, n) - P["ma_cci"] * medaid
    cci = np.digitize(lat, [-0.65, 0.55, 1.45])
    lp_poly = -1.35 + 0.65 * cci + 0.025 * (age - AGE_C) + 1.25 * medaid - 0.10 * female
    poly = (rng.random(n) < 1 / (1 + np.exp(-lp_poly))).astype(int)
    lp = P["b0"] + 0.30 * poly + P["bage"] * (age - AGE_C) - 0.25 * female + np.array([0, .38, .78, 1.15])[cci] + P["bma"] * medaid
    readm = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    return dict(age=age, female=female, cci=cci, medaid=medaid, poly=poly, readm=readm, n=n)


def design(d, terms):
    """build a design matrix. terms: list of 'poly','age','age10','female','medaid','cci_d' (3 dummies, ref 0),
    'cci_lin' (0-3 score), 'cci_d_ref1' (ref = 1-2), 'cci_d_ref3' (ref = >=5), 'agegrp' (2 dummies)"""
    n = d["n"]
    cols, names = [np.ones(n)], ["(Intercept)"]
    for t in terms:
        if t == "cci_d":
            for k, lab in ((1, "cci1-2"), (2, "cci3-4"), (3, "cci5+")):
                cols.append((d["cci"] == k).astype(float)); names.append(lab)
        elif t == "cci_d_ref1":
            for k, lab in ((0, "cci0"), (2, "cci3-4"), (3, "cci5+")):
                cols.append((d["cci"] == k).astype(float)); names.append(lab)
        elif t == "cci_d_ref3":
            for k, lab in ((0, "cci0"), (1, "cci1-2"), (2, "cci3-4")):
                cols.append((d["cci"] == k).astype(float)); names.append(lab)
        elif t == "cci_lin":
            cols.append(d["cci"].astype(float)); names.append("cci_score")
        elif t == "agegrp":
            a = d["age"]
            cols.append(((a >= 75) & (a < 85)).astype(float)); names.append("age75-84")
            cols.append((a >= 85).astype(float)); names.append("age85+")
        elif t == "age10":
            cols.append(d["age"] / 10.0); names.append("age10")
        else:
            cols.append(d[t].astype(float)); names.append(t)
    return np.column_stack(cols), names


def logistic(d, terms):
    X, names = design(d, terms)
    f = glm(X, d["readm"], "binomial", "logit")
    f["names"] = names
    f["ll"] = loglik(f)
    return f
