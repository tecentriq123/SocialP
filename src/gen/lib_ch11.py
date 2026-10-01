"""Cox proportional-hazards tools and data for chapter 11 (numpy/scipy only).

Implemented here with numpy/scipy (the lifelines boxes are cross-checked against lifelines 0.30 in nums_ch11.py):
  cox_loglik()   partial log-likelihood, score U and information I
                 - ties: 'efron' (R coxph default) or 'breslow' (SAS PHREG / Stata default)
                 - counting-process (start, stop] data via `entry`, strata, case weights,
                   time-varying coefficients via tt=[(column, g)] (adds X[:, column] * g(t))
  coxph()        Newton-Raphson with step halving; model-based and robust (sandwich) variance;
                 Wald, likelihood-ratio and score (log-rank) tests like summary.coxph in R
  cox_exact_1d() exact ('discrete') partial likelihood for one covariate (elementary symmetric
                 polynomials), for comparing tie methods
  concordance()  Harrell's C (ties in the predictor count 1/2) with a jackknife SE
  basehaz(), predict_surv(), adjusted_curves()  Breslow/Efron cumulative baseline hazard,
                 predicted S(t|x) and directly standardised ('adjusted') survival curves
  schoenfeld(), zph()  Schoenfeld / scaled Schoenfeld residuals; score test for beta(t) = beta + theta*g(t)
                 per term and GLOBAL, g = 1 - KM(t-) by default -- the test R's survival >= 3.0 cox.zph uses
  martingale()   martingale residuals (functional form)
  survsplit()    episode splitting at cut points (like survival::survSplit)
  logistic(), smd(), weighted_km(), lowess()

Data:
  rcc_cohort()   the 300-patient RCC cohort of chapter 7 (lib_ch07.cohort(2): time, status, drug, imdc)
                 with extra baseline covariates (age, sex, prior nephrectomy, non-clear-cell histology).
                 Because time/status are fixed by chapter 7, the extra covariates are drawn from their
                 conditional distribution given the observed (time, status, drug, imdc) under an assumed
                 full model (sampling-importance-resampling):  prior p(z | drug, imdc) x
                 hazard(T|z)^status x exp(-cumulative hazard(T|z)).  Non-clear-cell histology is given an
                 effect that is strong in the first months and absent later (a deliberate PH violation).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize_scalar
from lib_ch07 import cohort as _cohort07, km, surv_at, quantile_ci, logrank, scenario, Z  # noqa: F401


# ====================================================================== core
def _arr(time, status, X, entry, strata, weights):
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = len(time)
    entry = np.full(n, -np.inf) if entry is None else np.asarray(entry, float)
    strata = np.zeros(n, int) if strata is None else np.asarray(strata)
    weights = np.ones(n) if weights is None else np.asarray(weights, float)
    return time, status, X, entry, strata, weights


def _design_at(X, t, tt):
    if not tt:
        return X
    extra = [X[:, j] * g(t) for j, g in tt]
    return np.column_stack([X] + extra)


def cox_loglik(beta, time, status, X, entry=None, strata=None, weights=None, ties="efron", tt=None,
               keep=False):
    """log partial likelihood, score vector and information matrix at beta.
    Risk set at event time t: entry < t <= time (same stratum).
    Efron with case weights follows R: the d tied deaths share their mean weight in the denominators.
    keep=True also returns per-event-time pieces (for Schoenfeld residuals / zph)."""
    time, status, X, entry, strata, weights = _arr(time, status, X, entry, strata, weights)
    beta = np.atleast_1d(np.asarray(beta, float))
    p = len(beta)
    ll, U, I = 0.0, np.zeros(p), np.zeros((p, p))
    pieces = []
    for s in np.unique(strata):
        m = strata == s
        ts, ss, Xs, es, ws = time[m], status[m], X[m], entry[m], weights[m]
        idx = np.where(m)[0]
        for t in np.unique(ts[ss == 1]):
            R = (es < t) & (ts >= t)
            D = (ts == t) & (ss == 1)
            Zt = _design_at(Xs, t, tt)
            ZR, wR = Zt[R], ws[R]
            r = wR * np.exp(ZR @ beta)
            s0, s1, s2 = r.sum(), r @ ZR, (ZR * r[:, None]).T @ ZR
            ZD, wD = Zt[D], ws[D]
            rD = wD * np.exp(ZD @ beta)
            d = int(D.sum())
            ll += wD @ (ZD @ beta)
            U += wD @ ZD
            xbar_mean = np.zeros(p)
            Vsum = np.zeros((p, p))
            if ties == "breslow" or d == 1:
                sw = wD.sum()
                xb = s1 / s0
                V = s2 / s0 - np.outer(xb, xb)
                ll -= sw * np.log(s0)
                U -= sw * xb
                I += sw * V
                xbar_mean = xb
                Vsum = sw * V
            else:
                s0D, s1D, s2D = rD.sum(), rD @ ZD, (ZD * rD[:, None]).T @ ZD
                mw = wD.mean()
                for l in range(d):
                    f = l / d
                    a0, a1, a2 = s0 - f * s0D, s1 - f * s1D, s2 - f * s2D
                    xb = a1 / a0
                    V = a2 / a0 - np.outer(xb, xb)
                    ll -= mw * np.log(a0)
                    U -= mw * xb
                    I += mw * V
                    xbar_mean += xb / d
                    Vsum += mw * V
            if keep:
                pieces.append(dict(t=t, stratum=s, d=d, deaths=idx[D], xbar=xbar_mean, V=Vsum,
                                   ZD=ZD, s0=s0, s0D=rD.sum(), wD=wD))
    if keep:
        return ll, U, I, pieces
    return ll, U, I


def coxph(time, status, X, names=None, entry=None, strata=None, weights=None, ties="efron", tt=None,
          robust=False, cluster=None, center=True, maxit=60):
    """fit a Cox model. Columns of X are centred internally (as R does); this changes nothing in
    the estimates or tests. Returns a dict with coef, se, HR, CI, z, p, tests, variance matrices."""
    time, status, X, entry, strata, weights = _arr(time, status, X, entry, strata, weights)
    n, p0 = X.shape
    means = (np.average(X, axis=0, weights=weights) if center else np.zeros(p0))
    Xc = X - means
    p = p0 + (len(tt) if tt else 0)
    args = dict(time=time, status=status, X=Xc, entry=entry, strata=strata, weights=weights, ties=ties, tt=tt)
    b = np.zeros(p)
    ll0, U0, I0 = cox_loglik(b, **args)
    score = float(U0 @ np.linalg.solve(I0, U0))
    ll_old = ll0
    for it in range(maxit):
        ll, U, I = cox_loglik(b, **args)
        step = np.linalg.solve(I, U)
        nb = b + step
        ll_new = cox_loglik(nb, **args)[0]
        k = 0
        while ll_new < ll - 1e-10 and k < 30:
            step /= 2
            nb = b + step
            ll_new = cox_loglik(nb, **args)[0]
            k += 1
        b = nb
        if np.max(np.abs(step)) < 1e-10 or abs(ll_new - ll) < 1e-13:
            break
    ll, U, I = cox_loglik(b, **args)
    V = np.linalg.inv(I)
    se = np.sqrt(np.diag(V))
    out = dict(coef=b, var=V, se=se, loglik0=ll0, loglik=ll, score=score, lr=2 * (ll - ll0),
               wald=float(b @ I @ b), df=p, n=n, events=int(status.sum()), means=means,
               names=names, ties=ties, info=I, iters=it + 1, args=args, U=U)
    if robust or cluster is not None:
        Vr = robust_var(out, cluster=cluster)
        out["var_model"] = V
        out["se_model"] = se
        out["var"] = Vr
        out["se"] = np.sqrt(np.diag(Vr))
        out["wald"] = float(b @ np.linalg.solve(Vr, b))
    out["z"] = b / out["se"]
    out["p"] = 2 * st.norm.sf(np.abs(out["z"]))
    out["hr"] = np.exp(b)
    out["lo"] = np.exp(b - Z * out["se"])
    out["hi"] = np.exp(b + Z * out["se"])
    for k in ("lr", "wald", "score"):
        out["p_" + k] = st.chi2.sf(out[k], p)
    return out


def score_residuals(fit):
    """Breslow-form score residuals L_i (n x p), for the sandwich variance (case weights allowed)."""
    a = fit["args"]
    time, status, X, entry, strata, weights = _arr(a["time"], a["status"], a["X"], a["entry"], a["strata"], a["weights"])
    if a["tt"]:
        raise NotImplementedError("robust variance with tt terms")
    b = fit["coef"]
    n, p = X.shape
    L = np.zeros((n, p))
    eta = X @ b
    risk = np.exp(eta)
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        ts, ss, es = time[m], status[m], entry[m]
        for t in np.unique(ts[ss == 1]):
            R = m[(es < t) & (ts >= t)]
            D = m[(ts == t) & (ss == 1)]
            r = weights[R] * risk[R]
            s0 = r.sum()
            xb = (r @ X[R]) / s0
            dL = weights[D].sum() / s0          # Breslow hazard increment
            L[D] += X[D] - xb
            L[R] -= risk[R][:, None] * (X[R] - xb) * dL
    return L


def robust_var(fit, cluster=None):
    L = score_residuals(fit)
    w = fit["args"]["weights"]
    WL = L * w[:, None]
    if cluster is not None:
        cluster = np.asarray(cluster)
        g = np.unique(cluster)
        WL = np.array([WL[cluster == c].sum(axis=0) for c in g])
    Vm = np.linalg.inv(fit["info"])
    D = WL @ Vm            # dfbeta
    return D.T @ D


def num_grad_check(fit, eps=1e-5):
    """compare the analytic score/information with numerical derivatives of the log-likelihood."""
    a = fit["args"]
    b = fit["coef"] * 0.5 + 0.05
    ll, U, I = cox_loglik(b, **a)
    p = len(b)
    Ug = np.zeros(p)
    Hg = np.zeros((p, p))
    for j in range(p):
        e = np.zeros(p); e[j] = eps
        lp, Up, _ = cox_loglik(b + e, **a)
        lm, Um, _ = cox_loglik(b - e, **a)
        Ug[j] = (lp - lm) / (2 * eps)
        Hg[:, j] = -(Up - Um) / (2 * eps)
    return np.max(np.abs(Ug - U)), np.max(np.abs(Hg - I))


# ====================================================================== exact partial likelihood (1 covariate)
def cox_exact_1d(time, status, x):
    """'exact' (discrete / conditional-logistic) partial likelihood for one covariate:
    at a time with d tied deaths the denominator sums prod(exp(b x)) over all size-d subsets
    of the risk set = elementary symmetric polynomial e_d."""
    time = np.asarray(time, float); status = np.asarray(status, int); x = np.asarray(x, float)
    x = x - x.mean()
    evt = np.unique(time[status == 1])
    Rs = [(time >= t) for t in evt]
    Ds = [((time == t) & (status == 1)) for t in evt]

    def ll(b):
        tot = 0.0
        for R, D in zip(Rs, Ds):
            d = int(D.sum())
            w = np.exp(b * x[R])
            e = np.zeros(d + 1); e[0] = 1.0
            for wi in w:
                e[1:] = e[1:] + wi * e[:-1]
            tot += b * x[D].sum() - np.log(e[d])
        return tot

    res = minimize_scalar(lambda b: -ll(b), bounds=(-5, 5), method="bounded", options=dict(xatol=1e-10))
    b = res.x
    h = 1e-4
    d2 = (ll(b + h) - 2 * ll(b) + ll(b - h)) / h ** 2
    se = np.sqrt(-1 / d2)
    return dict(coef=b, se=se, hr=np.exp(b), lo=np.exp(b - Z * se), hi=np.exp(b + Z * se), loglik=ll(b), loglik0=ll(0.0))


# ====================================================================== concordance
def concordance(time, status, lp, jackknife=True):
    """Harrell's C: among comparable pairs (the earlier time is an event; a censoring at the same time
    counts as surviving longer), the share where the earlier failure has the higher risk score.
    Ties in the risk score count 1/2. SE by the (leave-one-out) jackknife."""
    time = np.asarray(time, float); status = np.asarray(status, int); lp = np.asarray(lp, float)
    Ti, Tj = time[:, None], time[None, :]
    comp = (status[:, None] == 1) & ((Ti < Tj) | ((Ti == Tj) & (status[None, :] == 0)))
    gt = lp[:, None] > lp[None, :] + 1e-12
    eq = np.abs(lp[:, None] - lp[None, :]) <= 1e-12
    score = comp * (gt + 0.5 * eq)
    Nc, Ns = comp.sum(), score.sum()
    C = Ns / Nc
    out = dict(C=C, pairs=int(Nc), concordant=int((comp & gt).sum()), tied=int((comp & eq).sum()),
               discordant=int((comp & ~gt & ~eq).sum()))
    if jackknife:
        n = len(time)
        rc, cc = comp.sum(1) + comp.sum(0), score.sum(1) + score.sum(0)
        Cj = (Ns - cc) / (Nc - rc)
        out["se"] = np.sqrt((n - 1) / n * ((Cj - Cj.mean()) ** 2).sum())
    return out


# ====================================================================== baseline hazard, prediction
def basehaz(fit):
    """cumulative baseline hazard (at the centred covariates) per stratum: dict stratum -> (times, H).
    Efron fits use the Efron increment sum_l 1/(s0 - l/d s0D), Breslow fits d/s0 (weighted if weights)."""
    a = fit["args"]
    time, status, X, entry, strata, weights = _arr(a["time"], a["status"], a["X"], a["entry"], a["strata"], a["weights"])
    b = fit["coef"]
    out = {}
    for s in np.unique(strata):
        m = strata == s
        ts, ss, Xs, es, ws = time[m], status[m], X[m], entry[m], weights[m]
        tt_, HH = [], []
        H = 0.0
        for t in np.unique(ts[ss == 1]):
            R = (es < t) & (ts >= t)
            D = (ts == t) & (ss == 1)
            Zt = _design_at(Xs, t, a["tt"])
            r = ws[R] * np.exp(Zt[R] @ b)
            rD = ws[D] * np.exp(Zt[D] @ b)
            d = int(D.sum())
            if fit["ties"] == "breslow" or d == 1:
                H += ws[D].sum() / r.sum()
            else:
                mw = ws[D].mean()
                H += sum(mw / (r.sum() - l / d * rD.sum()) for l in range(d))
            tt_.append(t); HH.append(H)
        out[s] = (np.array(tt_), np.array(HH))
    return out


def _H_at(bh, t):
    ts, H = bh
    i = np.searchsorted(ts, t, side="right") - 1
    return np.where(i >= 0, H[np.clip(i, 0, None)], 0.0)


def predict_surv(fit, Xnew, times, stratum=0):
    """S(t | x) = exp(-H0(t) exp(b'(x - means))) for rows of Xnew (uncentred) at the given times."""
    bh = basehaz(fit)[stratum]
    Xnew = np.atleast_2d(np.asarray(Xnew, float))
    lp = (Xnew - fit["means"]) @ fit["coef"][:Xnew.shape[1]]
    H0 = _H_at(bh, np.asarray(times, float))
    return np.exp(-np.outer(np.exp(lp), H0))   # rows = patients, cols = times


def adjusted_curves(fit, X, col, values, times):
    """direct standardisation: set column `col` to each value for everyone, average predicted S(t|x)."""
    res = {}
    for v in values:
        Xv = np.array(X, float).copy()
        Xv[:, col] = v
        res[v] = predict_surv(fit, Xv, times).mean(axis=0)
    return res


# ====================================================================== residuals, PH test
def schoenfeld(fit):
    """Schoenfeld residuals (one row per death; tied deaths share the Efron-averaged mean),
    scaled residuals r* = b + d V r (as cox.zph), event times, and per-time information pieces."""
    a = dict(fit["args"])
    if a["tt"]:
        raise NotImplementedError
    ll, U, I, pieces = cox_loglik(fit["coef"], keep=True, **a)
    rows, times, strat, gpieces = [], [], [], []
    for pc in pieces:
        for zd in pc["ZD"]:
            rows.append(zd - pc["xbar"])
            times.append(pc["t"])
            strat.append(pc["stratum"])
    r = np.array(rows)
    d = len(r)
    V = np.linalg.inv(I)
    scaled = fit["coef"] + d * r @ V
    return dict(resid=r, scaled=scaled, time=np.array(times), stratum=np.array(strat), pieces=pieces, V=V, I=I)


def zph(fit, terms, transform="km"):
    """score test of H0: theta = 0 in beta_j(t) = beta_j + theta_j g(t), evaluated at the fitted beta
    (as survival::cox.zph >= 3.0). terms: dict name -> list of column indices (a factor = several columns).
    g(t): 'km' = 1 - KM(t-) of all subjects, 'identity' = t, 'log' = log t, 'rank' = rank of event time."""
    a = fit["args"]
    sc = schoenfeld(fit)
    t_ev = sc["time"]
    if transform == "km":
        time, status = a["time"], a["status"]
        rows = km(time, status)
        kt = np.array([r_["t"] for r_ in rows]); kS = np.array([r_["S"] for r_ in rows])

        def g(t):
            i = np.searchsorted(kt, t, side="left") - 1      # left-continuous: S(t-)
            return 1 - (kS[i] if i >= 0 else 1.0)
    elif transform == "identity":
        g = lambda t: t
    elif transform == "log":
        g = np.log
    elif transform == "rank":
        rk = st.rankdata(t_ev)
        mp = {}
        for t, rr in zip(t_ev, rk):
            mp.setdefault(t, rr)
        g = lambda t: mp[t]
    else:
        raise ValueError(transform)
    p = len(fit["coef"])
    Uth = np.zeros(p)
    Ibb = np.zeros((p, p)); Itb = np.zeros((p, p)); Itt = np.zeros((p, p))
    for pc in sc["pieces"]:
        gv = g(pc["t"])
        rsum = (pc["ZD"] - pc["xbar"]).sum(axis=0)
        Uth += gv * rsum
        Ibb += pc["V"]; Itb += gv * pc["V"]; Itt += gv * gv * pc["V"]
    out = {}

    def test(J):
        J = list(J)
        UJ = Uth[J]
        M = Itt[np.ix_(J, J)] - Itb[np.ix_(J, range(p))] @ np.linalg.solve(Ibb, Itb[np.ix_(range(p), J)])
        chi = float(UJ @ np.linalg.solve(M, UJ))
        return chi, len(J), st.chi2.sf(chi, len(J))

    for nm, J in terms.items():
        out[nm] = test(J)
    out["GLOBAL"] = test(range(p))
    gt = np.array([g(t) for t in t_ev])
    return out, gt, sc


def ph_test_approx(fit, transform="rank"):
    """per-covariate proportional-hazards test in the form used by lifelines.statistics.proportional_hazard_test
    (the Grambsch-Therneau approximation, one column at a time, no global test):
        T_j = [sum_k (g_k - gbar) r*_kj]^2 / (d * Var(b_j) * sum_k (g_k - gbar)^2),  r* = d * r V (+ b)
    g = transform of the death times: 'rank' as lifelines 0.30 defines it (np.cumsum(events) over the data sorted by
    (stratum,) time, status: tied deaths get consecutive numbers in data order; the residual rows of schoenfeld() are
    in the same order), 'rank_avg' (average ranks of the death times), 'km' (1 - KM(t), right-continuous,
    KM of all subjects), 'identity', 'log'.  Returns arrays T, p, -log2(p) (one per column of X).
    Checked against lifelines.statistics.proportional_hazard_test in nums_ch11.py (section Q)."""
    a = fit["args"]
    sc = schoenfeld(fit)
    tk = sc["time"]
    d = len(tk)
    V = sc["V"]
    rs = d * sc["resid"] @ V
    if transform == "rank":
        g = np.arange(1, d + 1, dtype=float)
    elif transform == "rank_avg":
        g = st.rankdata(tk)
    elif transform == "km":
        rows = km(a["time"], a["status"])
        kt = np.array([r_["t"] for r_ in rows]); kS = np.array([r_["S"] for r_ in rows])
        g = np.array([1 - kS[np.searchsorted(kt, t, side="right") - 1] for t in tk])
    elif transform == "identity":
        g = tk.astype(float)
    elif transform == "log":
        g = np.log(tk)
    else:
        raise ValueError(transform)
    gd = g - g.mean()
    T = (gd @ rs) ** 2 / (d * np.diag(V) * (gd ** 2).sum())
    p = st.chi2.sf(T, 1)
    return T, p, -np.log2(p)


def martingale(fit):
    """martingale residuals status_i - H0(T_i) exp(eta_i) (right-censored, unstratified or stratified)."""
    a = fit["args"]
    time, status, X, entry, strata, weights = _arr(a["time"], a["status"], a["X"], a["entry"], a["strata"], a["weights"])
    bhs = basehaz(fit)
    eta = X @ fit["coef"][:X.shape[1]]
    M = np.zeros(len(time))
    for s, bh in bhs.items():
        m = strata == s
        M[m] = status[m] - _H_at(bh, time[m]) * np.exp(eta[m])
    return M


# ====================================================================== helpers
def survsplit(time, status, cuts):
    """split follow-up at the cut points. returns arrays id, start, stop, event, episode (0,1,...)"""
    ids, a, b, ev, ep = [], [], [], [], []
    bounds = [0.0] + list(cuts) + [np.inf]
    for i, (t, s) in enumerate(zip(time, status)):
        for k in range(len(bounds) - 1):
            lo, hi = bounds[k], bounds[k + 1]
            if t <= lo:
                break
            ids.append(i); a.append(lo); b.append(min(t, hi)); ep.append(k)
            ev.append(int(s == 1 and t <= hi))
    return tuple(np.array(v) for v in (ids, a, b, ev, ep))


def logistic(y, X, maxit=50):
    """logistic regression by Newton-Raphson; X without intercept. returns coef (with intercept), fitted p."""
    y = np.asarray(y, float)
    A = np.column_stack([np.ones(len(y)), X])
    b = np.zeros(A.shape[1])
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-A @ b))
        W = p * (1 - p)
        step = np.linalg.solve((A * W[:, None]).T @ A, A.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-11:
            break
    p = 1 / (1 + np.exp(-A @ b))
    return b, p


def smd(x, g, w=None):
    """standardised mean difference (group 1 - group 0), pooled SD of the unweighted groups."""
    x = np.asarray(x, float); g = np.asarray(g)
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    m1 = np.average(x[g == 1], weights=w[g == 1]); m0 = np.average(x[g == 0], weights=w[g == 0])
    v1 = np.var(x[g == 1], ddof=1); v0 = np.var(x[g == 0], ddof=1)
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)


def weighted_km(time, status, w):
    time = np.asarray(time, float); status = np.asarray(status, int); w = np.asarray(w, float)
    S = 1.0
    out = []
    for t in np.unique(time):
        n = w[time >= t].sum()
        d = w[(time == t) & (status == 1)].sum()
        if d > 0:
            S *= 1 - d / n
        out.append(dict(t=t, S=S, d=d, n=n))
    return out


def step_at(rows, t):
    S = 1.0
    for r in rows:
        if r["t"] <= t:
            S = r["S"]
    return S


def lowess(x, y, frac=0.6, it=0):
    """simple LOWESS (tricube weights, local linear); it = robustness iterations."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    r = int(np.ceil(frac * n))
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    fit = np.zeros(n)
    rw = np.ones(n)
    for _ in range(it + 1):
        for i in range(n):
            dist = np.abs(xs - xs[i])
            h = np.sort(dist)[r - 1]
            h = max(h, 1e-12)
            w = np.clip(1 - (dist / h) ** 3, 0, None) ** 3 * rw
            A = np.column_stack([np.ones(n), xs - xs[i]])
            WA = A * w[:, None]
            beta = np.linalg.lstsq(WA.T @ A, WA.T @ ys, rcond=None)[0]
            fit[i] = beta[0]
        res = ys - fit
        s = np.median(np.abs(res))
        rw = np.clip(1 - (res / (6 * s + 1e-12)) ** 2, 0, None) ** 2
    return xs, fit


# ====================================================================== data
EXPIT = lambda v: 1 / (1 + np.exp(-v))


def rcc_cohort(seed=6, K=3000, gam=None):
    """chapter-7 cohort (300 patients, same time/status/drug/imdc) + age, male, nephrectomy, nonclear.
    Assumed full model for the covariate draw (per month):
        h(t | x, z) = lambda_imdc * 0.8^drug * exp(g_age (age-63)/10 + g_male male + g_neph neph
                      + g_ncc nonclear * 1{t < CUT_NCC} - centering)
    """
    C = _cohort07(2)
    T, S, G, IM = C["time"], C["status"], C["drug"], C["imdc"]
    n = len(T)
    rng = np.random.default_rng(seed)
    g = dict(age=np.log(1.22), male=np.log(1.08), neph=np.log(0.65), ncc=np.log(3.0)) if gam is None else gam
    med = np.array([44.0, 24.0, 9.0])
    lam = np.log(2) / med[IM] * np.where(G == 1, 0.80, 1.0)
    age = np.zeros(n, int); male = np.zeros(n, int); neph = np.zeros(n, int); ncc = np.zeros(n, int)
    for i in range(n):
        a = np.clip(np.round(rng.normal(63 + 2.5 * IM[i] - 1.5 * G[i], 9.0, K)), 38, 86)
        mo = (rng.random(K) < 0.72).astype(int)
        ne = (rng.random(K) < EXPIT(1.1 - 1.0 * IM[i] + 0.15 * G[i])).astype(int)
        nc = (rng.random(K) < (0.10 + 0.08 * (1 - G[i]))).astype(int)
        base = g["age"] * (a - 63) / 10 + g["male"] * mo + g["neph"] * ne
        cen = np.log(np.mean(np.exp(base + g["ncc"] * nc)))   # keep the average hazard roughly unchanged
        early = min(T[i], CUT_NCC)
        late = max(T[i] - CUT_NCC, 0.0)
        cumh = lam[i] * np.exp(base - cen) * (early * np.exp(g["ncc"] * nc) + late)
        logh = base - cen + g["ncc"] * nc * (T[i] < CUT_NCC)
        lw = S[i] * logh - cumh
        w = np.exp(lw - lw.max()); w /= w.sum()
        k = rng.choice(K, p=w)
        age[i], male[i], neph[i], ncc[i] = int(a[k]), mo[k], ne[k], nc[k]
    return dict(time=T, status=S, drug=G, imdc=IM, age=age, male=male, neph=neph, ncc=ncc,
                reason=C["reason"], entry=C["entry"])


CUT_NCC = 8.0


def design(D, cols=("drug", "age10", "male", "imdc1", "imdc2", "neph", "ncc")):
    """design matrix; age10 = age / 10 (HR per 10 years), imdc1/imdc2 dummies vs favorable."""
    m = dict(drug=D["drug"], age=D["age"], age10=D["age"] / 10.0, male=D["male"],
             imdc1=(D["imdc"] == 1).astype(int), imdc2=(D["imdc"] == 2).astype(int),
             neph=D["neph"], ncc=D["ncc"])
    return np.column_stack([m[c] for c in cols]).astype(float), list(cols)
