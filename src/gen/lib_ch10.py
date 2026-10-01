"""Chapter 10 library: repeated-measures methods implemented with numpy/scipy only
(the statsmodels MixedLM / GEE boxes are cross-checked in nums_ch10.py --out).

Contents
  simulate()       hypothetical pharmacist-led intervention trial in type 2 diabetes (가상의 예시):
                   HbA1c at 0, 3, 6, 12 months with MAR dropout, binary adherence (PDC >= 80%) at 3, 6, 12 months
  rm_anova()       split-plot (mixed-design) repeated-measures ANOVA with Mauchly, GG, HF, lower bound
  LMM              fast (missing-pattern grouped) Gaussian linear model with structured covariance, REML or ML:
                   'cs' (= random intercept), 'ar1', 'rirs' (random intercept + slope), 'un' (unstructured);
                   Satterthwaite df for contrasts, BLUPs for 'rirs' / 'cs'
  gee()            Liang-Zeger GEE (gaussian identity / binomial logit), independence / exchangeable / ar1 /
                   unstructured working correlation, model-based and sandwich SEs, QIC
  glmm_logit_ri()  random-intercept logistic model by adaptive-free Gauss-Hermite quadrature (ML)
Every function is checked against independent calculations in nums_ch10.py.
"""
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from numpy.polynomial.hermite import hermgauss

MONTHS = np.array([0, 3, 6, 12])


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- data
PAR = dict(n_per=50, mu0=8.5, slope=(-0.015, -0.055), b0_sd=0.8, b1_sd=0.045, e_sd=0.3,
           drop_a=(-2.8, -3.0), drop_b=(2.0, 0.6), p_inter=0.03,
           adh=(0.75, 1.1, -0.45, -0.9), u_sd=1.8)


def simulate(seed, **kw):
    """HbA1c_ij = mu0 + slope[g] * month_j + b0_i + b1_i * month_j + e_ij  (rounded to 0.1)
    g = 0 usual care, 1 pharmacist intervention.
    Dropout before visit j (3, 6, 12 months): logit p = drop_a[g] + drop_b[g] * (last observed HbA1c - mu0)  -> MAR
    (depends only on observed values and randomized group; usual-care patients with poor control leave more often)
    Intermittent missed visit at 3 or 6 months with probability p_inter (MCAR).
    Adherence (PDC >= 80% over the preceding interval) at 3, 6, 12 months:
        logit P = a0 + a_g * g + a6 [6 mo] + a12 [12 mo] + u_i,   u_i ~ N(0, u_sd^2)
      observed whenever the patient is still in the study at that visit (dropouts: missing)."""
    P = dict(PAR)
    P.update(kw)
    rng = np.random.default_rng(seed)
    N = 2 * P["n_per"]
    g = rng.permutation(np.repeat([0, 1], P["n_per"]))
    b0 = rng.normal(0, P["b0_sd"], N)
    b1 = rng.normal(0, P["b1_sd"], N)
    e = rng.normal(0, P["e_sd"], (N, 4))
    sl = np.where(g == 1, P["slope"][1], P["slope"][0])
    full = np.round(P["mu0"] + sl[:, None] * MONTHS[None, :] + b0[:, None] + b1[:, None] * MONTHS[None, :] + e, 1)
    obs = np.ones((N, 4), bool)
    inst = np.ones((N, 4), bool)          # still in study (not dropped out)
    dropped_at = np.full(N, 99)
    u_drop = rng.random((N, 4))
    u_int = rng.random((N, 4))
    for i in range(N):
        last = full[i, 0]
        for j in (1, 2, 3):
            p = expit(P["drop_a"][g[i]] + P["drop_b"][g[i]] * (last - P["mu0"]))
            if u_drop[i, j] < p:
                obs[i, j:] = False
                inst[i, j:] = False
                dropped_at[i] = j
                break
            if j < 3 and u_int[i, j] < P["p_inter"]:
                obs[i, j] = False
                continue
            last = full[i, j]
    y = np.where(obs, full, np.nan)
    a0, ag, a6, a12 = P["adh"]
    u = rng.normal(0, P["u_sd"], N)
    eta = a0 + ag * g[:, None] + np.array([0, a6, a12])[None, :] + u[:, None]
    adh_full = (rng.random((N, 3)) < expit(eta)).astype(float)
    adh = np.where(inst[:, 1:], adh_full, np.nan)
    return dict(N=N, g=g, full=full, y=y, obs=obs, inst=inst, dropped_at=dropped_at, b0=b0, b1=b1,
                adh_full=adh_full, adh=adh, u=u, P=P)


# ---------------------------------------------------------------- OLS / logistic
def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, p = X.shape
    s2 = r @ r / (n - p)
    cov = s2 * np.linalg.inv(X.T @ X)
    return b, cov, s2


def glm_logit(y, X, it=100):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        mu = expit(X @ b)
        W = mu * (1 - mu)
        H = X.T @ (W[:, None] * X)
        step = np.linalg.solve(H, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = expit(X @ b)
    H = X.T @ ((mu * (1 - mu))[:, None] * X)
    return b, np.linalg.inv(H)


# ---------------------------------------------------------------- RM-ANOVA
def orth_contrasts(k):
    """k x (k-1) orthonormal contrasts (normalized Helmert)."""
    C = np.zeros((k, k - 1))
    for j in range(1, k):
        C[:j, j - 1] = 1.0
        C[j, j - 1] = -j
        C[:, j - 1] /= np.linalg.norm(C[:, j - 1])
    return C


def rm_anova(Y, grp=None):
    """Split-plot ANOVA. Y: N x k complete data, grp: group labels (N,) or None (one group).
    Returns SS/df/MS/F/p for group, error(between), time, time x group, error(within),
    Mauchly W / chi2 / df / p, Greenhouse-Geisser, Huynh-Feldt (original 1976 and Lecoutre 1991) and
    lower-bound epsilons, corrected df and p, partial eta^2, pooled within-group covariance S."""
    Y = np.asarray(Y, float)
    N, k = Y.shape
    if grp is None:
        grp = np.zeros(N, int)
    levels = np.unique(grp)
    G = len(levels)
    gm = Y.mean()
    subj = Y.mean(1)
    tm = Y.mean(0)
    SS_total = ((Y - gm) ** 2).sum()
    SS_subj = k * ((subj - gm) ** 2).sum()
    SS_group = sum(k * (grp == l).sum() * (Y[grp == l].mean() - gm) ** 2 for l in levels)
    SS_eb = SS_subj - SS_group
    SS_time = N * ((tm - gm) ** 2).sum()
    cell = np.array([Y[grp == l].mean(0) for l in levels])
    nl = np.array([(grp == l).sum() for l in levels])
    SS_cells = (nl[:, None] * (cell - gm) ** 2).sum()
    SS_tg = SS_cells - SS_time - SS_group
    SS_ew = SS_total - SS_subj - SS_time - SS_tg
    df = dict(group=G - 1, eb=N - G, time=k - 1, tg=(k - 1) * (G - 1), ew=(k - 1) * (N - G))
    SS = dict(group=SS_group, eb=SS_eb, time=SS_time, tg=SS_tg, ew=SS_ew, total=SS_total, subj=SS_subj)
    MS = {key: (SS[key] / df[key] if df[key] > 0 else np.nan) for key in df}
    F, Pv = {}, {}
    if G > 1:
        F["group"] = MS["group"] / MS["eb"]
        Pv["group"] = st.f.sf(F["group"], df["group"], df["eb"])
        F["tg"] = MS["tg"] / MS["ew"]
        Pv["tg"] = st.f.sf(F["tg"], df["tg"], df["ew"])
    F["time"] = MS["time"] / MS["ew"]
    Pv["time"] = st.f.sf(F["time"], df["time"], df["ew"])
    R = np.vstack([Y[grp == l] - Y[grp == l].mean(0) for l in levels])
    S = R.T @ R / (N - G)
    C = orth_contrasts(k)
    Sc = C.T @ S @ C
    p = k - 1
    W = np.linalg.det(Sc) / (np.trace(Sc) / p) ** p
    nu = N - G
    chi2 = -(nu - (2 * p ** 2 + p + 2) / (6 * p)) * np.log(W)
    mdf = p * (p + 1) // 2 - 1
    mp = st.chi2.sf(chi2, mdf)
    gg = np.trace(Sc) ** 2 / (p * np.trace(Sc @ Sc))
    hf_lec = min(1.0, ((nu + 1) * p * gg - 2) / (p * (nu - p * gg)))      # Lecoutre (1991)
    hf_orig = min(1.0, (N * p * gg - 2) / (p * (N - G - p * gg)))          # Huynh & Feldt (1976)
    lb = 1.0 / p
    corr = {}
    for name, eps in (("none", 1.0), ("gg", gg), ("hf", hf_orig), ("hf_lec", hf_lec), ("lb", lb)):
        d = {"eps": eps}
        for eff in ("time", "tg"):
            if eff in F:
                d[eff] = (eps * df[eff], eps * df["ew"], st.f.sf(F[eff], eps * df[eff], eps * df["ew"]))
        corr[name] = d
    peta = {"time": SS_time / (SS_time + SS_ew)}
    if G > 1:
        peta.update(group=SS_group / (SS_group + SS_eb), tg=SS_tg / (SS_tg + SS_ew))
    # Type III (unweighted-means) SS for the time main effect; identical to SS_time when groups are balanced.
    Zc = Y @ C
    zbar = np.array([Zc[grp == l].mean(0) for l in levels])        # G x (k-1)
    u_mean = zbar.mean(0)
    SS_time3 = float((u_mean ** 2).sum() / ((1.0 / nl).sum() / G ** 2))
    SS["time3"] = SS_time3
    MS["time3"] = SS_time3 / df["time"]
    F["time3"] = MS["time3"] / MS["ew"]
    Pv["time3"] = st.f.sf(F["time3"], df["time"], df["ew"])
    peta["time3"] = SS_time3 / (SS_time3 + SS_ew)
    for name, d in corr.items():
        d["time3"] = (d["eps"] * df["time"], d["eps"] * df["ew"], st.f.sf(F["time3"], d["eps"] * df["time"], d["eps"] * df["ew"]))
    return dict(N=N, k=k, G=G, SS=SS, df=df, MS=MS, F=F, P=Pv, S=S, Sc=Sc, W=W, chi2=chi2, mdf=mdf, mp=mp,
                gg=gg, hf=hf_orig, hf_lec=hf_lec, lb=lb, corr=corr, peta=peta, cell=cell, nl=nl)


def gg_box(S):
    """Box (1954) / Greenhouse-Geisser epsilon from the k x k covariance matrix (independent check)."""
    k = S.shape[0]
    sbar = S.mean()
    sd = np.diag(S).mean()
    si = S.mean(1)
    num = k ** 2 * (sd - sbar) ** 2
    den = (k - 1) * ((S ** 2).sum() - 2 * k * (si ** 2).sum() + k ** 2 * sbar ** 2)
    return num / den


# ---------------------------------------------------------------- linear mixed model / GLS (pattern grouped)
def _chol(th, q):
    L = np.zeros((q, q))
    idx = 0
    for i in range(q):
        for j in range(i + 1):
            L[i, j] = np.exp(th[idx]) if i == j else th[idx]
            idx += 1
    return L


def cov_builder(kind, t):
    """Return (n_theta, Sigma(theta) -> k x k, describe(theta) -> dict, theta0)."""
    t = np.asarray(t, float)
    k = len(t)
    if kind == "cs":
        def S(th):
            return np.exp(2 * th[0]) * np.ones((k, k)) + np.exp(2 * th[1]) * np.eye(k)

        def desc(th):
            v0, ve = np.exp(2 * th[0]), np.exp(2 * th[1])
            return dict(var_b0=v0, var_e=ve, icc=v0 / (v0 + ve))
        return 2, S, desc, np.log([0.8, 0.4])
    if kind == "ar1":
        lag = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])

        def S(th):
            return np.exp(2 * th[0]) * np.tanh(th[1]) ** lag

        def desc(th):
            return dict(var=np.exp(2 * th[0]), rho=np.tanh(th[1]))
        return 2, S, desc, np.array([np.log(0.9), 0.9])
    if kind == "rirs":
        Z = np.column_stack([np.ones(k), t])

        def S(th):
            L = _chol(th[:3], 2)
            return Z @ (L @ L.T) @ Z.T + np.exp(2 * th[3]) * np.eye(k)

        def desc(th):
            L = _chol(th[:3], 2)
            Gm = L @ L.T
            return dict(G=Gm, var_b0=Gm[0, 0], var_b1=Gm[1, 1],
                        corr=Gm[0, 1] / np.sqrt(Gm[0, 0] * Gm[1, 1]), var_e=np.exp(2 * th[3]))
        return 4, S, desc, np.array([np.log(0.8), 0.0, np.log(0.03), np.log(0.35)])
    if kind == "un":
        m = k * (k + 1) // 2

        def S(th):
            L = _chol(th, k)
            return L @ L.T

        def desc(th):
            Sg = S(th)
            sd = np.sqrt(np.diag(Sg))
            return dict(Sigma=Sg, sd=sd, R=Sg / np.outer(sd, sd))
        th0 = []
        for i in range(k):
            for j in range(i + 1):
                th0.append(np.log(0.6) if i == j else 0.3)
        return m, S, desc, np.array(th0)
    raise ValueError(kind)


class LMM:
    """y: N x k (NaN = missing), X: N x k x p design array, t: visit times (k,)."""

    def __init__(self, y, X, t, kind, reml=True, names=None):
        self.y = np.asarray(y, float)
        self.X = np.asarray(X, float)
        self.t = np.asarray(t, float)
        self.kind, self.reml = kind, reml
        self.names = names
        N, k, p = self.X.shape
        self.N, self.k, self.p = N, k, p
        obs = ~np.isnan(self.y)
        keys = {}
        for i in range(N):
            if obs[i].any():
                keys.setdefault(tuple(np.where(obs[i])[0]), []).append(i)
        self.pats = []
        for key, ids in keys.items():
            ix = np.array(key)
            ids = np.array(ids)
            self.pats.append((ix, ids, self.X[np.ix_(ids, ix)], self.y[np.ix_(ids, ix)]))
        self.n = int(obs.sum())
        self.nsub = sum(len(p_[1]) for p_ in self.pats)
        self.ntheta, self.Sfun, self.desc_fun, self.theta0 = cov_builder(kind, t)

    def parts(self, th):
        Sg = self.Sfun(th)
        W = np.zeros((self.p, self.p))
        u = np.zeros(self.p)
        logdet = 0.0
        cache = []
        for ix, ids, Xp, yp in self.pats:
            V = Sg[np.ix_(ix, ix)]
            c = np.linalg.cholesky(V)
            logdet += len(ids) * 2 * np.log(np.diag(c)).sum()
            Vi = np.linalg.inv(V)
            XV = np.einsum("nmp,mq->npq", Xp, Vi)
            W += np.einsum("npq,nqr->pr", XV, Xp)
            u += np.einsum("npq,nq->p", XV, yp)
            cache.append(Vi)
        return W, u, logdet, cache

    def dev(self, th):
        try:
            W, u, logdet, cache = self.parts(th)
            beta = np.linalg.solve(W, u)
        except np.linalg.LinAlgError:
            return 1e10
        q = 0.0
        for (ix, ids, Xp, yp), Vi in zip(self.pats, cache):
            r = yp - Xp @ beta
            q += np.einsum("nm,mq,nq->", r, Vi, r)
        if not self.reml:
            return logdet + q + self.n * np.log(2 * np.pi)
        return logdet + np.linalg.slogdet(W)[1] + q + (self.n - self.p) * np.log(2 * np.pi)

    def fit(self, theta0=None):
        th = self.theta0 if theta0 is None else np.asarray(theta0, float)
        best = None
        for method in ("BFGS", "Nelder-Mead", "BFGS"):
            opts = dict(maxiter=40000, xatol=1e-10, fatol=1e-12) if method == "Nelder-Mead" else dict(gtol=1e-8, maxiter=10000)
            r = minimize(self.dev, th if best is None else best.x, method=method, options=opts)
            if best is None or r.fun < best.fun:
                best = r
        self.theta = best.x
        self.deviance = best.fun
        W, u, logdet, cache = self.parts(self.theta)
        self.cov = np.linalg.inv(W)
        self.beta = self.cov @ u
        self.se = np.sqrt(np.diag(self.cov))
        npar = self.ntheta + (0 if self.reml else self.p)
        self.aic = self.deviance + 2 * npar
        self.desc = self.desc_fun(self.theta)
        self.Sigma = self.Sfun(self.theta)
        self.H = None
        return self

    def _phi(self, th):
        W, u, logdet, cache = self.parts(th)
        return np.linalg.inv(W)

    def satterthwaite(self, L, h=1e-4):
        L = np.asarray(L, float)
        th = self.theta
        if self.H is None:
            self.H = num_hess(self.dev, th, h)
        A = 2 * np.linalg.inv(self.H)
        v = lambda tt: L @ self._phi(tt) @ L
        g = np.zeros(len(th))
        for i in range(len(th)):
            e = np.zeros(len(th))
            e[i] = h
            g[i] = (v(th + e) - v(th - e)) / (2 * h)
        return 2 * v(th) ** 2 / (g @ A @ g)

    def contrast(self, L, df=None):
        L = np.asarray(L, float)
        est = L @ self.beta
        se = np.sqrt(L @ self.cov @ L)
        if df is None:
            df = self.satterthwaite(L)
        tv = est / se
        q = st.t.ppf(0.975, df)
        return dict(est=est, se=se, df=df, t=tv, p=2 * st.t.sf(abs(tv), df), lo=est - q * se, hi=est + q * se)

    def blup(self, i):
        """BLUP of random effects for subject i ('rirs' -> (b0, b1), 'cs' -> (b0,))."""
        obs = ~np.isnan(self.y[i])
        ix = np.where(obs)[0]
        V = self.Sigma[np.ix_(ix, ix)]
        r = self.y[i, ix] - self.X[i, ix] @ self.beta
        if self.kind == "rirs":
            Z = np.column_stack([np.ones(len(ix)), self.t[ix]])
            return self.desc["G"] @ Z.T @ np.linalg.solve(V, r)
        if self.kind == "cs":
            return np.array([self.desc["var_b0"] * np.ones(len(ix)) @ np.linalg.solve(V, r)])
        raise ValueError(self.kind)


def num_hess(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = h
            ej[j] = h
            if i == j:
                H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h ** 2
            else:
                H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    return H


# ---------------------------------------------------------------- GEE
def gee(y, X, ids, family="binomial", corstr="exchangeable", pos=None, k=None, it=200, tol=1e-10):
    """Liang-Zeger GEE. ids: cluster labels; pos: within-cluster visit index (0..k-1), needed for ar1 / unstructured.
    Moment estimators (Liang & Zeger 1986) with a p-adjusted denominator.
    Returns beta, naive (model-based) cov, robust (sandwich) cov, alpha / R, phi, QIC."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ids = np.asarray(ids)
    uid = list(dict.fromkeys(ids.tolist()))
    idx = [np.where(ids == c)[0] for c in uid]
    if pos is not None:
        pos = np.asarray(pos, int)
        idx = [ix[np.argsort(pos[ix])] for ix in idx]
    n, p = X.shape
    if family == "binomial":
        beta, _ = glm_logit(y, X)
    else:
        beta, _, _ = ols(y, X)

    def mv(b):
        eta = X @ b
        if family == "binomial":
            mu = expit(eta)
            return mu, mu * (1 - mu), mu * (1 - mu)
        return eta, np.ones(n), np.ones(n)

    state = dict(alpha=0.0, R=None)

    def Rmat(ix):
        m = len(ix)
        a = state["alpha"]
        if corstr == "independence":
            return np.eye(m)
        if corstr == "exchangeable":
            return (1 - a) * np.eye(m) + a * np.ones((m, m))
        if corstr == "ar1":
            pp = pos[ix]
            return a ** np.abs(pp[:, None] - pp[None, :])
        if corstr == "unstructured":
            pp = pos[ix]
            return state["R"][np.ix_(pp, pp)]

    def update_corr(r, phi):
        if corstr == "exchangeable":
            num = sum(((r[ix].sum()) ** 2 - (r[ix] ** 2).sum()) / 2 for ix in idx)
            cnt = sum(len(ix) * (len(ix) - 1) / 2 for ix in idx)
            state["alpha"] = num / ((cnt - p) * phi)
        elif corstr == "ar1":
            num, cnt = 0.0, 0
            for ix in idx:
                pp = pos[ix]
                for a_ in range(len(ix) - 1):
                    if pp[a_ + 1] - pp[a_] == 1:
                        num += r[ix[a_]] * r[ix[a_ + 1]]
                        cnt += 1
            state["alpha"] = num / ((cnt - p) * phi)
        elif corstr == "unstructured":
            Rs = np.zeros((k, k))
            Cn = np.zeros((k, k))
            for ix in idx:
                pp = pos[ix]
                Rs[np.ix_(pp, pp)] += np.outer(r[ix], r[ix])
                Cn[np.ix_(pp, pp)] += 1
            Rf = Rs / (Cn * phi)
            dsd = np.sqrt(np.diag(Rf))
            state["R"] = Rf / np.outer(dsd, dsd)

    for _ in range(it):
        mu, v, dm = mv(beta)
        r = (y - mu) / np.sqrt(v)
        phi = (r @ r) / (n - p)
        update_corr(r, phi)
        I = np.zeros((p, p))
        U = np.zeros(p)
        for ix in idx:
            A = np.sqrt(v[ix])
            Vi = phi * (A[:, None] * Rmat(ix) * A[None, :])
            D = dm[ix][:, None] * X[ix]
            Vinv = np.linalg.inv(Vi)
            I += D.T @ Vinv @ D
            U += D.T @ Vinv @ (y[ix] - mu[ix])
        step = np.linalg.solve(I, U)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    mu, v, dm = mv(beta)
    r = (y - mu) / np.sqrt(v)
    phi = (r @ r) / (n - p)
    # model-based (naive) covariance: binomial scale fixed at 1 as in statsmodels GEE (Scale: 1.000);
    # the Pearson phi is still used for the working-correlation estimate. beta and the robust
    # covariance do not depend on this choice.
    phi_v = 1.0 if family == "binomial" else phi
    I = np.zeros((p, p))
    M = np.zeros((p, p))
    for ix in idx:
        A = np.sqrt(v[ix])
        Vi = phi_v * (A[:, None] * Rmat(ix) * A[None, :])
        D = dm[ix][:, None] * X[ix]
        Vinv = np.linalg.inv(Vi)
        I += D.T @ Vinv @ D
        uu = D.T @ Vinv @ (y[ix] - mu[ix])
        M += np.outer(uu, uu)
    naive = np.linalg.inv(I)
    robust = naive @ M @ naive
    # QIC (Pan 2001): -2 Q(beta; I) + 2 trace(Omega_I V_R), quasi-likelihood under independence, phi = 1 for binomial
    if family == "binomial":
        Q = np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))
        OmegaI = X.T @ ((mu * (1 - mu))[:, None] * X)
    else:
        Q = -0.5 * np.sum((y - mu) ** 2) / phi
        OmegaI = X.T @ X / phi
    qic = -2 * Q + 2 * np.trace(OmegaI @ robust)
    return dict(beta=beta, naive=naive, robust=robust, se_naive=np.sqrt(np.diag(naive)),
                se_robust=np.sqrt(np.diag(robust)), alpha=state["alpha"], R=state["R"], phi=phi,
                nclus=len(idx), maxsize=max(len(ix) for ix in idx), qic=qic, Q=Q)


# ---------------------------------------------------------------- GLMM (random-intercept logistic, ML by Gauss-Hermite)
def glmm_logit_ri(y, X, ids, Q=40, theta0=None):
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    ids = np.asarray(ids)
    uid = list(dict.fromkeys(ids.tolist()))
    idx = [np.where(ids == c)[0] for c in uid]
    xq, wq = hermgauss(Q)
    lw = np.log(wq / np.sqrt(np.pi))
    p = X.shape[1]

    def nll(par):
        b, ls = par[:p], par[p]
        sig = np.exp(ls)
        eta = X @ b
        tot = 0.0
        for ix in idx:
            e = eta[ix][:, None] + np.sqrt(2) * sig * xq[None, :]
            ll = (y[ix][:, None] * e - np.logaddexp(0, e)).sum(0) + lw
            mx = ll.max()
            tot += mx + np.log(np.exp(ll - mx).sum())
        return -tot

    if theta0 is None:
        b0, _ = glm_logit(y, X)
        theta0 = np.r_[b0 * 1.5, np.log(1.5)]
    r = minimize(nll, theta0, method="BFGS", options=dict(gtol=1e-8, maxiter=5000))
    r = minimize(nll, r.x, method="Nelder-Mead", options=dict(maxiter=40000, xatol=1e-9, fatol=1e-11))
    r = minimize(nll, r.x, method="BFGS", options=dict(gtol=1e-9, maxiter=5000))
    H = num_hess(nll, r.x, 1e-4)
    cov = np.linalg.inv(H)
    return dict(beta=r.x[:p], sigma=np.exp(r.x[p]), cov=cov[:p, :p], se=np.sqrt(np.diag(cov))[:p],
                se_logsig=np.sqrt(cov[p, p]), nll=r.fun, par=r.x)


def marginal_from_conditional(eta, sigma, Q=80):
    """P(Y=1) averaged over b ~ N(0, sigma^2) for conditional linear predictor eta (vectorized)."""
    xq, wq = hermgauss(Q)
    eta = np.atleast_1d(eta)
    return (wq[None, :] / np.sqrt(np.pi) * expit(eta[:, None] + np.sqrt(2) * sigma * xq[None, :])).sum(1)
