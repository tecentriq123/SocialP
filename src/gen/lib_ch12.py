"""Helpers for chapter 12 (Poisson distribution, Poisson / negative binomial regression).

statsmodels is not available, so the GLMs are written out here:
  * poisson_glm  : IRLS that follows R's glm.fit (start mu = y + 0.1, same convergence rule),
                   so the iteration count printed by summary.glm can be reproduced.
  * nb_glm       : NB2 regression (MASS::glm.nb): joint ML of beta and theta = 1/alpha,
                   theta SE from the observed information used by MASS::theta.ml.
  * sandwich_hc0 : robust (HC0) covariance for a Poisson GLM (sandwich::vcovHC(type = "HC0")).
  * garwood      : exact (chi-square based) CI for a Poisson count / rate.
"""
import numpy as np
from scipy import stats as st
from scipy.special import gammaln, digamma, polygamma

Z = st.norm.ppf(0.975)


# ---------------------------------------------------------------- rates
def garwood(d, t=1.0, level=0.95):
    a = 1 - level
    lo = 0.0 if d == 0 else st.chi2.ppf(a / 2, 2 * d) / 2
    hi = st.chi2.ppf(1 - a / 2, 2 * (d + 1)) / 2
    return lo / t, hi / t


def wald_rate(d, t):
    r = d / t
    se = np.sqrt(d) / t
    return r - Z * se, r + Z * se


def log_rate(d, t):
    r = d / t
    return r * np.exp(-Z / np.sqrt(d)), r * np.exp(Z / np.sqrt(d))


def binom_test_two_sided(x, n, p):
    """R's binom.test two-sided p-value (sum of probabilities <= observed, relErr 1 + 1e-7)."""
    d = st.binom.pmf(x, n, p)
    m = n * p
    relErr = 1 + 1e-7
    if x == m:
        return 1.0
    ks = np.arange(0, n + 1)
    pk = st.binom.pmf(ks, n, p)
    return min(1.0, pk[pk <= d * relErr].sum())


def clopper_pearson(x, n, level=0.95):
    a = 1 - level
    lo = 0.0 if x == 0 else st.beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else st.beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


def compare_rates(d1, t1, d0, t0):
    """rate ratio (group 1 / group 0) and rate difference with Wald CIs; exact conditional test (poisson.test)."""
    r1, r0 = d1 / t1, d0 / t0
    irr = r1 / r0
    se = np.sqrt(1 / d1 + 1 / d0)
    rd = r1 - r0
    se_rd = np.sqrt(d1 / t1 ** 2 + d0 / t0 ** 2)
    zval = np.log(irr) / se
    p_w = 2 * st.norm.sf(abs(zval))
    # exact (poisson.test(c(d1, d0), c(t1, t0)))
    n = d1 + d0
    p0 = t1 / (t1 + t0)
    pl, ph = clopper_pearson(d1, n)
    ex_lo, ex_hi = pl / (1 - pl) * t0 / t1, ph / (1 - ph) * t0 / t1
    p_ex = binom_test_two_sided(d1, n, p0)
    return dict(r1=r1, r0=r0, irr=irr, se=se, lo=irr * np.exp(-Z * se), hi=irr * np.exp(Z * se), z=zval, p=p_w,
                rd=rd, se_rd=se_rd, rd_lo=rd - Z * se_rd, rd_hi=rd + Z * se_rd,
                ex_lo=ex_lo, ex_hi=ex_hi, p_exact=p_ex, p0=p0)


# ---------------------------------------------------------------- Poisson GLM (R glm.fit replica)
def dev_pois(y, mu):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.sum(t - (y - mu))


def loglik_pois(y, mu):
    return np.sum(y * np.log(mu) - mu - gammaln(y + 1))


def _wls(X, z, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)
    return beta


def poisson_glm(X, y, offset=None, maxit=25, eps=1e-8):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    mu = y + 0.1
    eta = np.log(mu)
    devold = dev_pois(y, mu)
    it = 0
    for it in range(1, maxit + 1):
        z = eta - off + (y - mu) / mu
        w = mu.copy()
        beta = _wls(X, z, w)
        eta = X @ beta + off
        mu = np.exp(eta)
        dev = dev_pois(y, mu)
        if abs(dev - devold) / (abs(dev) + 0.1) < eps:
            break
        devold = dev
    info = X.T @ (X * mu[:, None])
    cov = np.linalg.inv(info)
    se = np.sqrt(np.diag(cov))
    ll = loglik_pois(y, mu)
    pearson = np.sum((y - mu) ** 2 / mu)
    return dict(beta=beta, se=se, cov=cov, mu=mu, dev=dev, iters=it, ll=ll, aic=-2 * ll + 2 * p,
                pearson=pearson, df=n - p, n=n, p=p, X=X, y=y, off=off)


def null_poisson(y, offset):
    """intercept + offset model (R refits this for the null deviance when an offset is present)."""
    return poisson_glm(np.ones((len(y), 1)), y, offset)


def sandwich_hc0(fit):
    X, y, mu = fit["X"], fit["y"], fit["mu"]
    bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    meat = X.T @ (X * ((y - mu) ** 2)[:, None])
    return bread @ meat @ bread


# ---------------------------------------------------------------- NB2 (MASS::glm.nb)
def loglik_nb(y, mu, th):
    return np.sum(gammaln(th + y) - gammaln(th) - gammaln(y + 1) + th * np.log(th) + y * np.log(mu)
                  - (th + y) * np.log(th + mu))


def dev_nb(y, mu, th):
    return 2 * np.sum(y * np.log(np.maximum(1, y) / mu) - (y + th) * np.log((y + th) / (mu + th)))


def theta_ml(y, mu, limit=25, eps=np.finfo(float).eps ** 0.25):
    """MASS::theta.ml: Newton on the score for theta, mu fixed. Returns (theta, SE)."""
    n = len(y)
    score = lambda th: np.sum(digamma(th + y) - digamma(th) + np.log(th) + 1 - np.log(th + mu) - (y + th) / (mu + th))
    info = lambda th: np.sum(-polygamma(1, th + y) + polygamma(1, th) - 1 / th + 2 / (mu + th) - (y + th) / (mu + th) ** 2)
    t0 = n / np.sum((y / mu - 1) ** 2)
    it, dl = 0, 1.0
    while it < limit and abs(dl) > eps:
        it += 1
        t0 = abs(t0)
        i = info(t0)
        dl = score(t0) / i
        t0 = t0 + dl
    t0 = max(t0, 0)
    return t0, np.sqrt(1 / info(t0))


def nb_irls(X, y, off, th, beta0, maxit=50, eps=1e-10):
    beta = beta0.copy()
    for _ in range(maxit):
        eta = X @ beta + off
        mu = np.exp(eta)
        w = mu / (1 + mu / th)
        z = eta - off + (y - mu) / mu
        nb = _wls(X, z, w)
        if np.max(np.abs(nb - beta)) < eps:
            beta = nb
            break
        beta = nb
    mu = np.exp(X @ beta + off)
    return beta, mu


def nb_glm(X, y, offset=None):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    off = np.zeros(n) if offset is None else np.asarray(offset, float)
    pf = poisson_glm(X, y, off)
    beta, mu = pf["beta"], pf["mu"]
    th, _ = theta_ml(y, mu)
    for _ in range(200):
        beta_new, mu = nb_irls(X, y, off, th, beta)
        th_new, _ = theta_ml(y, mu)
        done = abs(th_new - th) < 1e-10 and np.max(np.abs(beta_new - beta)) < 1e-10
        beta, th = beta_new, th_new
        if done:
            break
    beta, mu = nb_irls(X, y, off, th, beta)
    th, th_se = theta_ml(y, mu)
    w = mu / (1 + mu / th)
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    se = np.sqrt(np.diag(cov))
    ll = loglik_nb(y, mu, th)
    return dict(beta=beta, se=se, cov=cov, mu=mu, theta=th, theta_se=th_se, alpha=1 / th, ll=ll,
                aic=-2 * ll + 2 * (p + 1), dev=dev_nb(y, mu, th), df=n - p, n=n, p=p, X=X, y=y, off=off)


def nb_null_deviance(y, off, th):
    X1 = np.ones((len(y), 1))
    b0 = np.array([np.log(y.sum() / np.exp(off).sum())])
    b, mu = nb_irls(X1, y, off, th, b0)
    return dev_nb(y, mu, th)


def nb_pmf(k, mu, alpha):
    if alpha == 0:
        return st.poisson.pmf(k, mu)
    r = 1 / alpha
    return st.nbinom.pmf(k, r, r / (r + mu))


# ---------------------------------------------------------------- simulated COPD cohort (나 절)
AGE_LABELS = ["40-64", "65-74", "75+"]
TRUE = dict(b0=np.log(0.42), drugA=np.log(0.80), age2=np.log(1.30), age3=np.log(1.70), female=np.log(0.90),
            cci=np.log(1.12), alpha=1.0)


def cohort(seed=20260929, nA=1000, nB=1400):
    rng = np.random.default_rng(seed)
    drug = np.r_[np.ones(nA, int), np.zeros(nB, int)]
    n = nA + nB
    age = np.where(drug == 1, rng.choice(3, n, p=[0.30, 0.38, 0.32]), rng.choice(3, n, p=[0.44, 0.35, 0.21]))
    female = (rng.random(n) < np.where(drug == 1, 0.27, 0.33)).astype(int)
    cci = np.minimum(rng.poisson(np.where(drug == 1, 2.3, 1.7) + 0.3 * age), 9)
    # as-treated follow-up in days: administrative end (drug A newer -> shorter) or discontinuation
    admin = np.where(drug == 1, rng.uniform(120, 760, n), rng.uniform(120, 1095, n))
    disc = rng.exponential(900, n)
    days = np.maximum(np.round(np.minimum(admin, disc)), 14)
    py = days / 365.25
    lin = (TRUE["b0"] + TRUE["drugA"] * drug + TRUE["age2"] * (age == 1) + TRUE["age3"] * (age == 2)
           + TRUE["female"] * female + TRUE["cci"] * cci)
    a = TRUE["alpha"]
    u = rng.gamma(1 / a, a, n)
    y = rng.poisson(u * np.exp(lin) * py)
    return dict(drug=drug, age=age, female=female, cci=cci, days=days, py=py, y=y.astype(float), n=n)


def design(c, cols=("drugA", "age2", "age3", "female", "cci")):
    parts = [np.ones(c["n"])]
    m = {"drugA": c["drug"], "age2": (c["age"] == 1) * 1, "age3": (c["age"] == 2) * 1,
         "female": c["female"], "cci": c["cci"]}
    for k in cols:
        parts.append(m[k])
    return np.column_stack(parts).astype(float)
