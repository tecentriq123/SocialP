"""Survival-analysis helpers and data sets for chapter 7 (numpy/scipy only).

Implemented here (lifelines/statsmodels are not available):
  km()            Kaplan-Meier with Greenwood variance and plain / log / log-log CIs
  surv_at()       S(t) and its CI at a time point (right-continuous step function)
  quantile_ci()   median (or other quantile) with Brookmeyer-Crowley type CI from the CI bands
  reverse_km()    reverse Kaplan-Meier (censoring = 'event') for median follow-up
  rmst()          restricted mean survival time with Greenwood-type SE
  logrank()       two- or k-group (weighted, stratified) log-rank test, trend test
  cox_efron()     Cox model with one covariate, Efron ties, optional strata
All functions were checked against the by-hand tables in nums_ch07.py.
"""
import numpy as np
import scipy.stats as st

Z = st.norm.ppf(0.975)


# ---------------------------------------------------------------- Kaplan-Meier
def km(time, status):
    time = np.asarray(time, float)
    status = np.asarray(status, int)
    ut = np.unique(time)
    rows = []
    S, gw = 1.0, 0.0
    for t in ut:
        n = int((time >= t).sum())
        d = int(((time == t) & (status == 1)).sum())
        c = int(((time == t) & (status == 0)).sum())
        cond = 1 - d / n
        S *= cond
        if d > 0:
            gw += d / (n * (n - d)) if n > d else np.inf
        rows.append(dict(t=t, n=n, d=d, c=c, cond=cond, S=S, gw=gw))
    return rows


def ci_from(S, gw, kind="loglog"):
    if S <= 0:
        return (0.0, 0.0)
    if S >= 1 or gw == 0:
        return (1.0, 1.0)
    if kind == "plain":
        se = S * np.sqrt(gw)
        return (S - Z * se, S + Z * se)
    if kind == "log":
        se = np.sqrt(gw)
        return (S * np.exp(-Z * se), min(1.0, S * np.exp(Z * se)))
    # log-log
    se = np.sqrt(gw) / abs(np.log(S))
    lo = np.exp(-np.exp(np.log(-np.log(S)) + Z * se))
    hi = np.exp(-np.exp(np.log(-np.log(S)) - Z * se))
    return (lo, hi)


def surv_at(rows, t, kind="loglog"):
    """S(t), SE(S), CI at time t (value of the right-continuous step function)."""
    S, gw = 1.0, 0.0
    for r in rows:
        if r["t"] <= t:
            S, gw = r["S"], r["gw"]
    se = S * np.sqrt(gw) if np.isfinite(gw) else np.nan
    return S, se, ci_from(S, gw, kind)


def n_risk(time, t):
    return int((np.asarray(time) >= t).sum())


def events_between(time, status, a, b):
    time = np.asarray(time); status = np.asarray(status)
    return int(((time > a) & (time <= b) & (status == 1)).sum())


def _first_below(ts, vals, p):
    """smallest t with value <= p (R convention: exact equality over a flat stretch -> midpoint)."""
    for i, (t, v) in enumerate(zip(ts, vals)):
        if v <= p + 1e-12:
            if abs(v - p) < 1e-12 and i + 1 < len(ts):
                # flat at exactly p until next drop -> midpoint (R quantile.survfit)
                for j in range(i + 1, len(ts)):
                    if vals[j] < p - 1e-12:
                        return (t + ts[j]) / 2
                return t
            return t
    return None


def quantile_ci(rows, p=0.5, kind="loglog"):
    """time at which the curve first reaches S <= p (p = 0.5 -> median survival).
    95% CI read off the pointwise bands (Brookmeyer-Crowley type, as R's survfit does):
      lower limit = first t where the LOWER band <= p (the lowest curve crosses first)
      upper limit = first t where the UPPER band <= p (None = not reached / not estimable)
    returns (estimate, lower, upper)."""
    ev = [r for r in rows if r["d"] > 0]
    ts = [r["t"] for r in ev]
    S = [r["S"] for r in ev]
    lo = [ci_from(r["S"], r["gw"], kind)[0] for r in ev]
    hi = [ci_from(r["S"], r["gw"], kind)[1] for r in ev]
    return _first_below(ts, S, p), _first_below(ts, lo, p), _first_below(ts, hi, p)


def reverse_km(time, status):
    return km(time, 1 - np.asarray(status))


def rmst(rows, tau):
    """area under the KM step curve from 0 to tau, and Greenwood-type SE."""
    knots = [0.0] + [r["t"] for r in rows if r["d"] > 0 and r["t"] < tau] + [tau]
    Svals = [1.0]
    for r in rows:
        if r["d"] > 0 and r["t"] < tau:
            Svals.append(r["S"])
    area = sum(Svals[i] * (knots[i + 1] - knots[i]) for i in range(len(Svals)))
    # variance: sum over event times t_j<=tau of A_j^2 d_j / (n_j (n_j - d_j)), A_j = area from t_j to tau
    var = 0.0
    evs = [r for r in rows if r["d"] > 0 and r["t"] < tau]
    for k, r in enumerate(evs):
        # area from t_j to tau
        A = 0.0
        for i in range(k + 1, len(Svals)):
            A += Svals[i] * (knots[i + 1] - knots[i])
        if r["n"] > r["d"]:
            var += A ** 2 * r["d"] / (r["n"] * (r["n"] - r["d"]))
    return area, np.sqrt(var), list(zip(knots[:-1], knots[1:], Svals))


# ---------------------------------------------------------------- log-rank
def _pooled_km_left(times_ev, n_all, d_all):
    """pooled KM just before each event time (S(t-)), for Peto/FH weights."""
    s, out = 1.0, []
    for n, d in zip(n_all, d_all):
        out.append(s)
        s *= 1 - d / n
    return np.array(out)


def logrank_table(time, status, group, g1=1):
    """per-event-time table for two groups (group==g1 is 'group 1')."""
    time = np.asarray(time, float); status = np.asarray(status, int); group = np.asarray(group)
    et = np.unique(time[status == 1])
    out = []
    for t in et:
        at = time >= t
        n1 = int((at & (group == g1)).sum()); n = int(at.sum()); n2 = n - n1
        d = int(((time == t) & (status == 1)).sum())
        d1 = int(((time == t) & (status == 1) & (group == g1)).sum())
        e1 = d * n1 / n
        v = d * (n1 / n) * (n2 / n) * (n - d) / (n - 1) if n > 1 else 0.0
        out.append(dict(t=t, n1=n1, n2=n2, n=n, d1=d1, d2=d - d1, d=d, e1=e1, e2=d - e1, v=v))
    return out


def logrank(time, status, group, g1=1, weight="logrank", strata=None):
    """two-group weighted log-rank; weight in logrank|gehan|tw|peto|fh01. strata: array -> stratified."""
    time = np.asarray(time, float); status = np.asarray(status, int); group = np.asarray(group)
    if strata is None:
        strata = np.zeros(len(time), int)
    strata = np.asarray(strata)
    U = V = O1 = E1 = 0.0
    O2 = E2 = 0.0
    for s in np.unique(strata):
        m = strata == s
        tab = logrank_table(time[m], status[m], group[m], g1)
        if not tab:
            continue
        n_all = np.array([r["n"] for r in tab]); d_all = np.array([r["d"] for r in tab])
        Sl = _pooled_km_left(None, n_all, d_all)
        if weight == "logrank":
            w = np.ones(len(tab))
        elif weight == "gehan":
            w = n_all.astype(float)
        elif weight == "tw":
            w = np.sqrt(n_all)
        elif weight == "peto":
            w = Sl
        elif weight == "fh01":
            w = 1 - Sl
        else:
            raise ValueError(weight)
        for wi, r in zip(w, tab):
            U += wi * (r["d1"] - r["e1"])
            V += wi ** 2 * r["v"]
            O1 += r["d1"]; E1 += r["e1"]; O2 += r["d2"]; E2 += r["e2"]
    chi2 = U ** 2 / V
    return dict(O1=O1, E1=E1, O2=O2, E2=E2, U=U, V=V, chi2=chi2, p=st.chi2.sf(chi2, 1))


def logrank_k(time, status, group, levels, scores=None):
    """k-group log-rank (df k-1) and trend test with given scores (df 1)."""
    time = np.asarray(time, float); status = np.asarray(status, int); group = np.asarray(group)
    k = len(levels)
    O = np.zeros(k); E = np.zeros(k); Vm = np.zeros((k, k))
    for t in np.unique(time[status == 1]):
        at = time >= t
        n = at.sum(); d = ((time == t) & (status == 1)).sum()
        ng = np.array([(at & (group == g)).sum() for g in levels], float)
        dg = np.array([((time == t) & (status == 1) & (group == g)).sum() for g in levels], float)
        O += dg; E += d * ng / n
        if n > 1:
            f = d * (n - d) / (n - 1)
            p_ = ng / n
            Vm += f * (np.diag(p_) - np.outer(p_, p_))
    diff = O - E
    chi_global = diff[:-1] @ np.linalg.solve(Vm[:-1, :-1], diff[:-1])
    res = dict(O=O, E=E, chi_global=chi_global, df=k - 1, p_global=st.chi2.sf(chi_global, k - 1))
    if scores is not None:
        w = np.asarray(scores, float)
        U = w @ diff; Vt = w @ Vm @ w
        res.update(chi_trend=U ** 2 / Vt, p_trend=st.chi2.sf(U ** 2 / Vt, 1))
    return res


# ---------------------------------------------------------------- Cox (1 covariate, Efron)
def _cox_ll(beta, time, status, x, strata):
    ll = g = h = 0.0
    for s in np.unique(strata):
        m = strata == s
        t_, s_, x_ = time[m], status[m], x[m]
        for t in np.unique(t_[s_ == 1]):
            R = t_ >= t
            D = (t_ == t) & (s_ == 1)
            d = D.sum()
            eR = np.exp(beta * x_[R]); eD = np.exp(beta * x_[D])
            s0R, s1R, s2R = eR.sum(), (eR * x_[R]).sum(), (eR * x_[R] ** 2).sum()
            s0D, s1D, s2D = eD.sum(), (eD * x_[D]).sum(), (eD * x_[D] ** 2).sum()
            ll += beta * x_[D].sum(); g += x_[D].sum()
            for l in range(d):
                f = l / d
                a0 = s0R - f * s0D; a1 = s1R - f * s1D; a2 = s2R - f * s2D
                ll -= np.log(a0); g -= a1 / a0; h -= a2 / a0 - (a1 / a0) ** 2
    return ll, g, h


def cox_efron(time, status, x, strata=None):
    time = np.asarray(time, float); status = np.asarray(status, int); x = np.asarray(x, float)
    strata = np.zeros(len(time), int) if strata is None else np.asarray(strata)
    b = 0.0
    ll0, g0, h0 = _cox_ll(0.0, time, status, x, strata)
    score = g0 ** 2 / (-h0)
    for _ in range(50):
        ll, g, h = _cox_ll(b, time, status, x, strata)
        step = -g / h
        b += step
        if abs(step) < 1e-10:
            break
    ll, g, h = _cox_ll(b, time, status, x, strata)
    se = np.sqrt(-1 / h)
    zval = b / se
    return dict(beta=b, se=se, hr=np.exp(b), lo=np.exp(b - Z * se), hi=np.exp(b + Z * se),
                p=2 * st.norm.sf(abs(zval)), score=score, p_score=st.chi2.sf(score, 1), lr=2 * (ll - ll0))


# ---------------------------------------------------------------- data sets
MONTH0 = (2019, 1)


def ym(m):
    y, mo = MONTH0
    k = (mo - 1) + int(m)
    return f"{y + k // 12}-{k % 12 + 1:02d}"


# 12 RCC patients (가상의 예시). entry = months since 2019-01, data cutoff at month 48 (2023-01)
SMALL = [
    # id, entry, time, status, reason, drug
    (1, 10, 3, 1, "사망", "B"),
    (2, 2, 5, 0, "전원(추적 소실)", "B"),
    (3, 14, 8, 1, "사망", "B"),
    (4, 6, 12, 1, "사망", "A"),
    (5, 19, 12, 1, "사망", "B"),
    (6, 8, 16, 0, "전원(추적 소실)", "A"),
    (7, 21, 20, 1, "사망", "B"),
    (8, 23, 25, 0, "자료 마감", "A"),
    (9, 1, 28, 1, "사망", "A"),
    (10, 16, 32, 0, "자료 마감", "A"),
    (11, 7, 38, 1, "사망", "B"),
    (12, 4, 44, 0, "자료 마감", "A"),
]
CUTOFF = 48


def small_arrays():
    t = np.array([r[2] for r in SMALL], float)
    s = np.array([r[3] for r in SMALL], int)
    g = np.array([1 if r[5] == "A" else 0 for r in SMALL])
    return t, s, g


def cohort(seed=2024):
    """observational RCC cohort: first-line drug A vs drug B, IMDC risk as confounder."""
    rng = np.random.default_rng(seed)
    nA, nB = 152, 148
    pA = [0.32, 0.53, 0.15]; pB = [0.18, 0.54, 0.28]
    imdc = np.concatenate([rng.choice(3, nA, p=pA), rng.choice(3, nB, p=pB)])
    drug = np.concatenate([np.ones(nA, int), np.zeros(nB, int)])
    med = np.array([44.0, 24.0, 9.0])  # drug-B medians by IMDC (favorable, intermediate, poor)
    lam = np.log(2) / med[imdc] * np.where(drug == 1, 0.80, 1.0)
    T = rng.exponential(1 / lam)
    entry = rng.uniform(0, 60, nA + nB)   # 2016-01 .. 2020-12
    admin = 84 - entry                     # data cutoff 2022-12-31
    loss = rng.exponential(1 / 0.0023, nA + nB)
    obs = np.minimum(np.minimum(T, admin), loss)
    status = (T <= np.minimum(admin, loss)).astype(int)
    reason = np.where(status == 1, "death", np.where(loss < admin, "lost", "admin"))
    time = np.round(obs, 1)
    time = np.maximum(time, 0.1)
    return dict(time=time, status=status, drug=drug, imdc=imdc, reason=reason, entry=entry)


def piecewise_sim(rng, n, cuts, rates):
    """exponential piecewise-constant hazard; cuts=[c1,c2,...], rates len = len(cuts)+1"""
    out = np.empty(n)
    for i in range(n):
        t0 = 0.0
        bounds = [0.0] + list(cuts) + [np.inf]
        for k, lam in enumerate(rates):
            e = rng.exponential(1 / lam)
            if t0 + e < bounds[k + 1]:
                out[i] = t0 + e
                break
            t0 = bounds[k + 1]
    return out


def scenario(kind, seed):
    rng = np.random.default_rng(seed)
    n = 200
    if kind == "early":
        # X: extra early deaths in first 3 months, afterwards same hazard as Y
        tx = piecewise_sim(rng, n, [3], [0.060, 0.025])
        ty = piecewise_sim(rng, n, [3], [0.015, 0.025])
    else:
        # X: worse in the first 4 months, better afterwards (crossing curves / delayed benefit)
        tx = piecewise_sim(rng, n, [4], [0.050, 0.022])
        ty = piecewise_sim(rng, n, [4], [0.035, 0.035])
    cx = rng.uniform(24, 48, n); cy = rng.uniform(24, 48, n)
    time = np.round(np.concatenate([np.minimum(tx, cx), np.minimum(ty, cy)]), 1)
    status = np.concatenate([(tx <= cx), (ty <= cy)]).astype(int)
    group = np.concatenate([np.ones(n, int), np.zeros(n, int)])
    return time, status, group
