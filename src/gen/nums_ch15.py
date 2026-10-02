"""Numbers for chapter 15 (성향점수).

Run:  source /home/claude/pylibs/env.sh && python3 gen/nums_ch15.py
Every number quoted in content/ch15.html, figs/ch15_*.html comes from compute() below.

Running example: 10,000 new users of drug A (4,053) or drug B (5,947) in a simulated claims cohort,
outcome = hospitalization within 1 year.  The cohort, the treatment and the 1-year outcome are drawn
exactly as in the old gen/nums_ch14a.py (same seed, same order of random draws), so the old numbers
(crude RR 1.33, matched 0.91, IPTW 0.95) are unchanged; extra variables (event day, negative-control
outcome) are drawn afterwards.
"""
import math
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st

warnings.filterwarnings("ignore")
Z = st.norm.ppf(0.975)


# ----------------------------------------------------------------------------- toy examples
def ps_toy():
    """One binary confounder (heart failure). True RR = 0.8 in both strata."""
    S = {
        "HF+": dict(nA=150, eA=36, nB=50, eB=15),
        "HF-": dict(nA=100, eA=8, nB=400, eB=40),
    }
    out = {}
    tot = dict(nA=0, eA=0, nB=0, eB=0)
    wA = wB = eAw = eBw = 0.0
    wB_att = eB_att = 0.0
    ow = dict(nA=0.0, nB=0.0, eA=0.0, eB=0.0)
    m = dict(n=0, eA=0.0, eB=0.0, hf=0)
    for k, s in S.items():
        ps = s["nA"] / (s["nA"] + s["nB"])
        w_a, w_b = 1 / ps, 1 / (1 - ps)
        w_b_att = ps / (1 - ps)
        rA, rB = s["eA"] / s["nA"], s["eB"] / s["nB"]
        npair = min(s["nA"], s["nB"])          # 1:1 matching without replacement on the exact PS
        out[k] = dict(s, ps=ps, w_a=w_a, w_b=w_b, w_b_att=w_b_att, rA=rA, rB=rB, rr=rA / rB,
                      nA_w=s["nA"] * w_a, nB_w=s["nB"] * w_b, eA_w=s["eA"] * w_a, eB_w=s["eB"] * w_b,
                      nB_att=s["nB"] * w_b_att, eB_att=s["eB"] * w_b_att,
                      npair=npair, eA_m=npair * rA, eB_m=npair * rB, unmatchedA=s["nA"] - npair,
                      ow_a=1 - ps, ow_b=ps)
        for kk in tot:
            tot[kk] += s[kk]
        wA += s["nA"] * w_a; wB += s["nB"] * w_b
        eAw += s["eA"] * w_a; eBw += s["eB"] * w_b
        wB_att += s["nB"] * w_b_att; eB_att += s["eB"] * w_b_att
        ow["nA"] += s["nA"] * (1 - ps); ow["nB"] += s["nB"] * ps
        ow["eA"] += s["eA"] * (1 - ps); ow["eB"] += s["eB"] * ps
        m["n"] += npair; m["eA"] += npair * rA; m["eB"] += npair * rB
        if k == "HF+":
            m["hf"] = npair
    crude_rA = tot["eA"] / tot["nA"]; crude_rB = tot["eB"] / tot["nB"]
    ate_rA = eAw / wA; ate_rB = eBw / wB
    att_rA = crude_rA; att_rB = eB_att / wB_att
    pA = tot["nA"] / (tot["nA"] + tot["nB"])
    return dict(strata=out, tot=tot, crude_rA=crude_rA, crude_rB=crude_rB, crude_rr=crude_rA / crude_rB,
                wA=wA, wB=wB, eAw=eAw, eBw=eBw, ate_rA=ate_rA, ate_rB=ate_rB, ate_rr=ate_rA / ate_rB,
                ate_rd=ate_rA - ate_rB, wB_att=wB_att, eB_att=eB_att, att_rA=att_rA, att_rB=att_rB,
                att_rr=att_rA / att_rB, att_rd=att_rA - att_rB,
                m=m, m_rA=m["eA"] / m["n"], m_rB=m["eB"] / m["n"], m_rr=m["eA"] / m["eB"],
                m_rd=(m["eA"] - m["eB"]) / m["n"],
                ow=ow, ow_rA=ow["eA"] / ow["nA"], ow_rB=ow["eB"] / ow["nB"],
                ow_rr=(ow["eA"] / ow["nA"]) / (ow["eB"] / ow["nB"]), pA=pA,
                hfA=S["HF+"]["nA"] / tot["nA"], hfB=S["HF+"]["nB"] / tot["nB"],
                hf_all=(S["HF+"]["nA"] + S["HF+"]["nB"]) / (tot["nA"] + tot["nB"]))


def mediator_toy():
    """No confounding; drug A works only by controlling a marker measured 3 months after the index date."""
    n = 1000
    pM = dict(A=0.70, B=0.40)       # proportion controlled at 3 months
    risk = {1: 0.10, 0: 0.20}       # 1-year hospitalization risk by control status (same for both drugs)
    out = {}
    for g in "AB":
        n1 = n * pM[g]; n0 = n - n1
        out[g] = dict(n1=n1, n0=n0, e1=n1 * risk[1], e0=n0 * risk[0], e=n1 * risk[1] + n0 * risk[0])
        out[g]["risk"] = out[g]["e"] / n
    rr = out["A"]["risk"] / out["B"]["risk"]
    return dict(n=n, pM=pM, risk=risk, g=out, rr=rr, rr_m1=1.0, rr_m0=1.0)


# ----------------------------------------------------------------------------- cohort
COVS = [  # (key, Korean label, English label, kind)
    ("age", "나이", "Age, y, mean (SD)", "cont"),
    ("female", "여성", "Female", "bin"),
    ("hf", "심부전", "Heart failure", "bin"),
    ("ckd", "만성콩팥병", "Chronic kidney disease", "bin"),
    ("prior", "지난 1년 입원", "Hospitalization in prior year", "bin"),
    ("ndrug", "복용 약물 계열 수", "No. of drug classes, mean (SD)", "cont"),
    ("tert", "상급종합병원 처방", "Prescribed at tertiary hospital", "bin"),
]
KEYS = [k for k, *_ in COVS]


def simulate_cohort(seed=20260930, N=10000, sel=1.0):
    """sel = 1: the running example.  sel > 1 multiplies every coefficient of the treatment model
    (stronger channelling) and is used only for the 'poor overlap' illustration with another seed."""
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(66, 10, N), 40, 95)
    female = rng.binomial(1, 0.48, N)
    # unmeasured frailty (not in claims): related to age
    frail = rng.binomial(1, 1 / (1 + np.exp(-(-1.6 + 0.06 * (age - 66)))), N)
    hf = rng.binomial(1, 1 / (1 + np.exp(-(-2.4 + 0.04 * (age - 66) + 0.5 * frail))), N)
    ckd = rng.binomial(1, 1 / (1 + np.exp(-(-1.9 + 0.04 * (age - 66)))), N)
    prior = rng.binomial(1, 1 / (1 + np.exp(-(-1.6 + 0.8 * hf + 0.3 * ckd))), N)
    ndrug = rng.poisson(np.exp(np.log(5.5) + 0.25 * hf + 0.2 * ckd + 0.01 * (age - 66)))
    tert = rng.binomial(1, 0.25, N)
    lp = (-0.9 + sel * (0.025 * (age - 66) - 0.1 * female + 0.7 * hf + 0.5 * ckd + 0.5 * prior
                        + 0.07 * (ndrug - 6) + 0.6 * tert + 0.9 * frail))
    A = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    # outcome: 1-year hospitalization, log-linear risk -> conditional RR of drug A = 0.80 everywhere
    lr = (np.log(0.065) + np.log(0.80) * A + 0.02 * (age - 66) + 0.6 * hf + 0.35 * ckd + 0.55 * prior
          + 0.04 * (ndrug - 6) + 0.25 * tert + 1.0 * frail)
    risk = np.minimum(np.exp(lr), 0.95)
    Y = rng.binomial(1, risk)
    # ---- extra draws (after everything above, so A and Y are identical to the old chapter 14 example)
    # day of first hospitalization for those with Y = 1 (somewhat more events early); others censored at day 365
    day = np.where(Y == 1, np.ceil(365 * rng.uniform(0, 1, N) ** 1.25), 365.0)
    # negative-control outcome: injury-related emergency visit within 1 year.
    # Neither drug affects it; age, sex and the unmeasured frailty do.
    lr_nc = np.log(0.045) + 0.02 * (age - 66) + 0.15 * female + 1.2 * frail
    NC = rng.binomial(1, np.minimum(np.exp(lr_nc), 0.95))
    X = dict(age=age, female=female, hf=hf, ckd=ckd, prior=prior, ndrug=ndrug, tert=tert, frail=frail)
    return X, A, Y, risk, day, NC


def smd(x, g, w=None):
    x = np.asarray(x, float)
    if w is None:
        w = np.ones_like(x)
    m1 = np.average(x[g == 1], weights=w[g == 1]); m0 = np.average(x[g == 0], weights=w[g == 0])
    v1 = np.average((x[g == 1] - m1) ** 2, weights=w[g == 1]); v0 = np.average((x[g == 0] - m0) ** 2, weights=w[g == 0])
    return (m1 - m0) / math.sqrt((v1 + v0) / 2)


def fit_ps(X, A, keys):
    import statsmodels.api as sm
    M = pd.DataFrame({k: np.asarray(X[k], float) for k in keys})
    M = sm.add_constant(M)
    res = sm.Logit(A, M).fit(disp=0)
    ps = np.asarray(res.predict(M))
    return ps, res


def greedy_match(ps, A, caliper_sd=0.2, seed=7, replace=False):
    """1:1 nearest-neighbour matching on the logit of the PS, caliper 0.2 SD of the logit."""
    lg = np.log(ps / (1 - ps))
    cal = caliper_sd * lg.std(ddof=1)
    rng = np.random.default_rng(seed)
    t_idx = np.where(A == 1)[0]
    c_idx = np.where(A == 0)[0]
    avail = np.ones(len(c_idx), bool)
    order = rng.permutation(t_idx)
    pairs = []
    lc = lg[c_idx]
    for t in order:
        dist = np.abs(lc - lg[t])
        if not replace:
            dist[~avail] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= cal:
            pairs.append((t, c_idx[j]))
            if not replace:
                avail[j] = False
    return np.array(pairs), cal


def rr_gee(Y, A, groups, w=None, extra=None):
    """Modified Poisson (log link) with robust sandwich SE; groups = pair id or subject id."""
    import statsmodels.api as sm
    cols = [np.ones(len(A)), A.astype(float)]
    if extra is not None:
        cols += [np.asarray(e, float) for e in extra]
    Xd = np.column_stack(cols)
    mod = sm.GEE(Y, Xd, groups=groups, family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence(),
                 weights=w)
    res = mod.fit()
    b, se = res.params[1], res.bse[1]
    return math.exp(b), (math.exp(b - Z * se), math.exp(b + Z * se)), se


def rd_robust(Y, A, groups, w=None):
    """Risk difference from a weighted linear (identity-link) GEE with robust SE."""
    import statsmodels.api as sm
    Xd = sm.add_constant(A.astype(float))
    res = sm.GEE(Y, Xd, groups=groups, family=sm.families.Gaussian(), cov_struct=sm.cov_struct.Independence(),
                 weights=w).fit()
    b, se = res.params[1], res.bse[1]
    return b, (b - Z * se, b + Z * se)


def evalue(rr):
    rr = 1 / rr if rr < 1 else rr
    return rr + math.sqrt(rr * (rr - 1))


def evalue_ci(rr, lo, hi):
    """E-value for the confidence limit closest to the null (1 if the CI includes 1)."""
    if lo <= 1 <= hi:
        return 1.0
    return evalue(hi) if rr < 1 else evalue(lo)


def cox_hr(day, Y, A, w=None, cluster=None, strata=None):
    from lifelines import CoxPHFitter
    df = pd.DataFrame(dict(day=day, Y=Y, A=A))
    kw = {}
    if w is not None:
        df["w"] = w; kw["weights_col"] = "w"; kw["robust"] = True
    if cluster is not None:
        df["cl"] = cluster; kw["cluster_col"] = "cl"
    if strata is not None:
        df["st"] = strata; kw["strata"] = ["st"]
    cph = CoxPHFitter().fit(df, "day", "Y", **kw)
    s = cph.summary.loc["A"]
    return float(s["exp(coef)"]), (float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"])), float(s["se(coef)"]), cph


def km_curve(day, Y, w=None):
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter().fit(day, Y, weights=w)
    sf = kmf.survival_function_
    return np.asarray(sf.index, float), 1 - np.asarray(sf.iloc[:, 0], float)


def ess(w):
    return w.sum() ** 2 / (w ** 2).sum()


def ps_sim():
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    X, A, Y, risk, day, NC = simulate_cohort()
    N = len(A)
    ids = np.arange(N)
    ps, res = fit_ps(X, A, KEYS)
    c_ps = roc_auc_score(A, ps)
    # scikit-learn check (no penalty) -> same PS
    Msk = np.column_stack([X[k] for k in KEYS]).astype(float)
    sk = LogisticRegression(C=np.inf, max_iter=5000).fit(Msk, A)   # C=np.inf: no penalty (scikit-learn 1.8)
    ps_sk = sk.predict_proba(Msk)[:, 1]
    sk_maxdiff = float(np.abs(ps_sk - ps).max())
    sk_def = LogisticRegression(max_iter=5000).fit(Msk, A)  # default L2 penalty
    sk_def_maxdiff = float(np.abs(sk_def.predict_proba(Msk)[:, 1] - ps).max())

    # ---- two example patients (hand calculation of the PS)
    b = res.params
    ex = []
    for label, v in (("78세 여성, 심부전·만성콩팥병, 지난 1년 입원, 약물 9계열, 상급종합병원",
                      dict(age=78, female=1, hf=1, ckd=1, prior=1, ndrug=9, tert=1)),
                     ("58세 남성, 동반질환 없음, 입원 없음, 약물 4계열, 의원",
                      dict(age=58, female=0, hf=0, ckd=0, prior=0, ndrug=4, tert=0))):
        lp_ = b["const"] + sum(b[k] * v[k] for k in KEYS)
        ex.append(dict(label=label, v=v, lp=lp_, ps=1 / (1 + math.exp(-lp_)),
                       terms={k: b[k] * v[k] for k in KEYS}))

    # ---- overlap
    ps_rng = dict(A=(ps[A == 1].min(), ps[A == 1].max()), B=(ps[A == 0].min(), ps[A == 0].max()))
    lo_c, hi_c = max(ps_rng["A"][0], ps_rng["B"][0]), min(ps_rng["A"][1], ps_rng["B"][1])
    out_common = (ps < lo_c) | (ps > hi_c)
    out_0109 = (ps < 0.1) | (ps > 0.9)
    overlap = dict(lo=lo_c, hi=hi_c, n_out=int(out_common.sum()), nA_out=int((out_common & (A == 1)).sum()),
                   nB_out=int((out_common & (A == 0)).sum()), n_0109=int(out_0109.sum()),
                   q=np.quantile(ps, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
                   meanA=ps[A == 1].mean(), meanB=ps[A == 0].mean(),
                   medA=np.median(ps[A == 1]), medB=np.median(ps[A == 0]))

    # ---- matching (primary: 1:1, no replacement, caliper 0.2 SD of logit PS)
    pairs, cal = greedy_match(ps, A)
    ti, ci_ = pairs[:, 0], pairs[:, 1]
    idx = np.r_[ti, ci_]
    Am = A[idx]
    npair = len(pairs)
    pair_id = np.r_[np.arange(npair), np.arange(npair)]
    matched_t = np.zeros(N, bool); matched_t[ti] = True
    un = (A == 1) & ~matched_t
    lg = np.log(ps / (1 - ps))
    unmatched = dict(n=int(un.sum()), ps_un=ps[un].mean(), ps_m=ps[ti].mean(),
                     age_un=X["age"][un].mean(), age_m=X["age"][ti].mean(),
                     hf_un=X["hf"][un].mean(), hf_m=X["hf"][ti].mean(),
                     ckd_un=X["ckd"][un].mean(), ckd_m=X["ckd"][ti].mean(),
                     risk_un=Y[un].mean(), risk_m=Y[ti].mean(),
                     ps_un_min=ps[un].min(), ps_un_med=np.median(ps[un]),
                     maxdist=float(np.abs(lg[ti] - lg[ci_]).max()), meddist=float(np.median(np.abs(lg[ti] - lg[ci_]))),
                     sd_logit=float(lg.std(ddof=1)))
    # matching with replacement (every A user gets the closest B user within the caliper)
    pairs_r, _ = greedy_match(ps, A, replace=True)
    tr, cr = pairs_r[:, 0], pairs_r[:, 1]
    n_distinct = len(np.unique(cr))
    cnt = np.bincount(cr, minlength=N)
    w_r = np.where(A == 1, np.isin(ids, tr).astype(float), cnt.astype(float))   # weight = times used
    keep_r = w_r > 0
    rr_repl = rr_gee(Y[keep_r], A[keep_r], ids[keep_r], w=w_r[keep_r])
    in_nr = np.isin(tr, ti)          # A users who were also matched in the without-replacement analysis
    sub = dict(n_same=int(in_nr.sum()), n_new=int((~in_nr).sum()),
               rr_same=float(Y[tr[in_nr]].mean() / Y[cr[in_nr]].mean()), rr_new=float(Y[tr[~in_nr]].mean() / Y[cr[~in_nr]].mean()),
               rA_new=float(Y[tr[~in_nr]].mean()), rB_new=float(Y[cr[~in_nr]].mean()),
               frail_new=(float(X["frail"][tr[~in_nr]].mean()), float(X["frail"][cr[~in_nr]].mean())),
               frail_same=(float(X["frail"][tr[in_nr]].mean()), float(X["frail"][cr[in_nr]].mean())))
    repl = dict(sub=sub, nA=len(tr), n_distinct=n_distinct, max_used=int(cnt.max()), rr=rr_repl,
                rB=np.average(Y[keep_r & (A == 0)], weights=w_r[keep_r & (A == 0)]), rA=Y[tr].mean(),
                smd_max=max(abs(smd(X[k][keep_r], A[keep_r], w_r[keep_r])) for k in KEYS))

    # ---- weights
    pA = A.mean()
    w_ate = np.where(A == 1, 1 / ps, 1 / (1 - ps))
    w_st = np.where(A == 1, pA / ps, (1 - pA) / (1 - ps))
    w_att = np.where(A == 1, 1.0, ps / (1 - ps))
    w_ow = np.where(A == 1, 1 - ps, ps)
    lo_t, hi_t = np.quantile(w_st, [0.01, 0.99])
    w_tr = np.clip(w_st, lo_t, hi_t)
    wsum = dict(ate_rng=(w_ate.min(), w_ate.max()), st_rng=(w_st.min(), w_st.max()), st_mean=w_st.mean(),
                att_rng=(w_att[A == 0].min(), w_att[A == 0].max()),
                sumA=w_ate[A == 1].sum(), sumB=w_ate[A == 0].sum(),
                sum_stA=w_st[A == 1].sum(), sum_stB=w_st[A == 0].sum(),
                att_sumB=w_att[A == 0].sum(),
                essA=ess(w_ate[A == 1]), essB=ess(w_ate[A == 0]), ess_attB=ess(w_att[A == 0]),
                trunc=(lo_t, hi_t), n_gt10=int((w_ate > 10).sum()),
                top=(w_ate.max(), int(A[np.argmax(w_ate)]), ps[np.argmax(w_ate)]))

    # ---- balance table
    tab = []
    for k, ko, en, kind in COVS + [("frail", "허약(청구자료에 없음)", "Frailty (unmeasured)", "bin")]:
        x = X[k]
        row = dict(key=k, ko=ko, en=en, kind=kind,
                   mA=x[A == 1].mean(), mB=x[A == 0].mean(), sA=x[A == 1].std(ddof=1), sB=x[A == 0].std(ddof=1),
                   smd_pre=smd(x, A),
                   mA_m=x[ti].mean(), mB_m=x[ci_].mean(), sA_m=x[ti].std(ddof=1), sB_m=x[ci_].std(ddof=1),
                   smd_m=smd(x[idx], Am), smd_w=smd(x, A, w_ate), smd_att=smd(x, A, w_att), smd_ow=smd(x, A, w_ow),
                   mA_w=np.average(x[A == 1], weights=w_ate[A == 1]), mB_w=np.average(x[A == 0], weights=w_ate[A == 0]))
        # P values (t test / chi-square) before and after matching, to show why they are not used
        if kind == "cont":
            row["p_pre"] = st.ttest_ind(x[A == 1], x[A == 0], equal_var=False).pvalue
            row["p_m"] = st.ttest_ind(x[ti], x[ci_], equal_var=False).pvalue
        else:
            def chi(a, b_):
                t = np.array([[a.sum(), len(a) - a.sum()], [b_.sum(), len(b_) - b_.sum()]])
                return st.chi2_contingency(t, correction=False)[1]
            row["p_pre"] = chi(x[A == 1], x[A == 0]); row["p_m"] = chi(x[ti], x[ci_])
        tab.append(row)

    # ---- outcome: risks, RR, RD
    rA, rB = Y[A == 1].mean(), Y[A == 0].mean()
    crude = rr_gee(Y, A, ids)
    crude_rd = rd_robust(Y, A, ids)
    rA_m, rB_m = Y[ti].mean(), Y[ci_].mean()
    matched = rr_gee(Y[idx], Am, pair_id)
    matched_rd = rd_robust(Y[idx], Am, pair_id)
    matched_naive = rr_gee(Y[idx], Am, np.arange(2 * npair))   # ignoring the pairing
    # discordant pairs (McNemar view)
    ya, yb = Y[ti], Y[ci_]
    disc = dict(b=int(((ya == 1) & (yb == 0)).sum()), c=int(((ya == 0) & (yb == 1)).sum()),
                both=int(((ya == 1) & (yb == 1)).sum()), none=int(((ya == 0) & (yb == 0)).sum()))
    rA_w = np.average(Y[A == 1], weights=w_ate[A == 1]); rB_w = np.average(Y[A == 0], weights=w_ate[A == 0])
    iptw = rr_gee(Y, A, ids, w=w_st)
    iptw_rd = rd_robust(Y, A, ids, w=w_st)
    iptw_tr = rr_gee(Y, A, ids, w=w_tr)
    # naive (model-based) SE that treats the weights as real frequencies -> wrong
    glm = sm.GLM(Y, sm.add_constant(A.astype(float)), family=sm.families.Poisson(), freq_weights=w_ate).fit()
    b_, se_ = glm.params[1], glm.bse[1]
    iptw_naive = (math.exp(b_), (math.exp(b_ - Z * se_), math.exp(b_ + Z * se_)), se_)
    rB_att = np.average(Y[A == 0], weights=w_att[A == 0])
    att = rr_gee(Y, A, ids, w=w_att)
    att_rd = rd_robust(Y, A, ids, w=w_att)
    rA_ow = np.average(Y[A == 1], weights=w_ow[A == 1]); rB_ow = np.average(Y[A == 0], weights=w_ow[A == 0])
    owr = rr_gee(Y, A, ids, w=w_ow)

    # ---- stratification on PS quintiles (Mantel-Haenszel RR) and PS as covariate
    q = np.quantile(ps, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(ps, q)
    srr = []
    num = den = 0.0
    var_num = 0.0
    for s in range(5):
        m_ = strata == s
        n1 = int((m_ & (A == 1)).sum()); n0 = int((m_ & (A == 0)).sum()); n = n1 + n0
        a = int(Y[m_ & (A == 1)].sum()); c = int(Y[m_ & (A == 0)].sum())
        srr.append(dict(n=n, nA=n1, nB=n0, eA=a, eB=c, rA=a / n1, rB=c / n0, rr=(a / n1) / (c / n0),
                        ps_lo=ps[m_].min(), ps_hi=ps[m_].max(),
                        smd_age=smd(X["age"][m_], A[m_]), smd_hf=smd(X["hf"][m_], A[m_])))
        num += a * n0 / n; den += c * n1 / n
        var_num += ((a + c) * n1 * n0 - a * c * n) / n ** 2
    mh = num / den
    se_mh = math.sqrt(var_num / (num * den))
    strat = dict(rows=srr, rr=mh, ci=(math.exp(math.log(mh) - Z * se_mh), math.exp(math.log(mh) + Z * se_mh)))
    ps_cov = rr_gee(Y, A, ids, extra=[lg])
    ps_cov_q = rr_gee(Y, A, ids, extra=[(strata == s).astype(float) for s in range(1, 5)])

    # ---- multivariable regression (modified Poisson) with different adjustment sets (section 가, 마)
    def adj(keys):
        return rr_gee(Y, A, ids, extra=[X[k] for k in keys])
    sets = [
        ("없음 (보정 전)", []),
        ("나이, 성별", ["age", "female"]),
        ("나이, 성별, 심부전, 만성콩팥병", ["age", "female", "hf", "ckd"]),
        ("측정한 7개 변수 모두", KEYS),
        ("7개 변수 + 허약 (실제 연구에서는 불가능)", KEYS + ["frail"]),
    ]
    adjsets = [dict(label=l, keys=k, rr=(crude if not k else adj(k))) for l, k in sets]
    reg = adjsets[3]["rr"]
    reg_frail = adjsets[4]["rr"]
    # logistic regression OR for the same model (to remind that OR != RR)
    Mx = sm.add_constant(np.column_stack([A] + [X[k] for k in KEYS]).astype(float))
    lr_ = sm.Logit(Y, Mx).fit(disp=0)
    ci_or = lr_.conf_int()
    reg_or = (math.exp(lr_.params[1]), (math.exp(ci_or[1][0]), math.exp(ci_or[1][1])))

    # ---- oracle analyses (frailty in the PS model; only possible in a simulation)
    ps_o, _ = fit_ps(X, A, KEYS + ["frail"])
    w_o = np.where(A == 1, pA / ps_o, (1 - pA) / (1 - ps_o))
    oracle = rr_gee(Y, A, ids, w=w_o)
    pairs_o, _ = greedy_match(ps_o, A)
    idx_o = np.r_[pairs_o[:, 0], pairs_o[:, 1]]
    matched_o = rr_gee(Y[idx_o], A[idx_o], np.r_[np.arange(len(pairs_o)), np.arange(len(pairs_o))])
    c_ps_o = roc_auc_score(A, ps_o)
    frail_smd_o = smd(X["frail"], A, np.where(A == 1, 1 / ps_o, 1 / (1 - ps_o)))

    # ---- truth
    lr0 = np.log(risk) - np.log(0.80) * A
    r0 = np.minimum(np.exp(lr0), 0.95); r1 = np.minimum(np.exp(lr0 + np.log(0.80)), 0.95)
    true_ate = r1.mean() / r0.mean(); true_att = r1[A == 1].mean() / r0[A == 1].mean()
    true = dict(ate=true_ate, att=true_att, r1=r1.mean(), r0=r0.mean(), r1_t=r1[A == 1].mean(), r0_t=r0[A == 1].mean(),
                ate_rd=r1.mean() - r0.mean(), att_rd=r1[A == 1].mean() - r0[A == 1].mean())

    # ---- time-to-event analyses (lifelines)
    hr_crude = cox_hr(day, Y, A)
    hr_iptw = cox_hr(day, Y, A, w=w_st)
    hr_match = cox_hr(day[idx], Y[idx], Am, cluster=pair_id)
    # Cox model stratified on the matched pair: with 1:1 pairs the partial likelihood reduces to
    # "in how many pairs did the A user / the B user have the event first" (a lifelines fit with 3,516 strata
    # gives the same answer but takes minutes: CoxPHFitter().fit(..., strata=["pair"]) -> 0.897, 0.778-1.035).
    da, db = day[ti] + 1000 * (1 - Y[ti]), day[ci_] + 1000 * (1 - Y[ci_])   # censored -> later than any event
    n_a_first = int((da < db).sum()); n_b_first = int((db < da).sum())
    hs = n_a_first / n_b_first
    se_hs = math.sqrt(1 / n_a_first + 1 / n_b_first)
    hr_match_strat = (hs, (math.exp(math.log(hs) - Z * se_hs), math.exp(math.log(hs) + Z * se_hs)), se_hs,
                      dict(a_first=n_a_first, b_first=n_b_first, tie=int(((da == db) & (Y[ti] == 1)).sum())))
    hr_iptw_naive = None
    try:
        from lifelines import CoxPHFitter
        df = pd.DataFrame(dict(day=day, Y=Y, A=A, w=w_ate))
        cph_n = CoxPHFitter().fit(df, "day", "Y", weights_col="w")   # robust=False: weights treated as counts
        s_ = cph_n.summary.loc["A"]
        hr_iptw_naive = (float(s_["exp(coef)"]), (float(s_["exp(coef) lower 95%"]), float(s_["exp(coef) upper 95%"])))
    except Exception as e:  # pragma: no cover
        hr_iptw_naive = str(e)
    # weighted Kaplan-Meier (cumulative incidence at day 365)
    km = dict(cA=km_curve(day[A == 1], Y[A == 1]), cB=km_curve(day[A == 0], Y[A == 0]),
              wA=km_curve(day[A == 1], Y[A == 1], w_st[A == 1]), wB=km_curve(day[A == 0], Y[A == 0], w_st[A == 0]))
    km_end = {k: float(v[1][-1]) for k, v in km.items()}
    km_180 = {k: float(v[1][np.searchsorted(v[0], 180, side="right") - 1]) for k, v in km.items()}

    # ---- bootstrap CI for the IPTW RR (PS re-estimated in every resample)
    rngb = np.random.default_rng(15)
    boots = []
    Mfull = sm.add_constant(pd.DataFrame({k: np.asarray(X[k], float) for k in KEYS}))
    for _ in range(500):
        ii = rngb.integers(0, N, N)
        Ab, Yb, Mb = A[ii], Y[ii], Mfull.iloc[ii]
        pb = np.asarray(sm.Logit(Ab, Mb).fit(disp=0).predict(Mb))
        wb = np.where(Ab == 1, 1 / pb, 1 / (1 - pb))
        boots.append(math.log(np.average(Yb[Ab == 1], weights=wb[Ab == 1]) / np.average(Yb[Ab == 0], weights=wb[Ab == 0])))
    boot_ci = (math.exp(np.quantile(boots, 0.025)), math.exp(np.quantile(boots, 0.975)))
    boot_se = float(np.std(boots, ddof=1))

    # ---- negative-control outcome (injury-related emergency visit; no true effect of either drug)
    nc = dict(nA=int(NC[A == 1].sum()), nB=int(NC[A == 0].sum()), rA=NC[A == 1].mean(), rB=NC[A == 0].mean(),
              crude=rr_gee(NC, A, ids),
              mA=int(NC[ti].sum()), mB=int(NC[ci_].sum()), rA_m=NC[ti].mean(), rB_m=NC[ci_].mean(),
              matched=rr_gee(NC[idx], Am, pair_id), iptw=rr_gee(NC, A, ids, w=w_st),
              oracle=rr_gee(NC, A, ids, w=w_o))

    # ---- E-values and the bias produced by frailty
    fr = X["frail"]
    p1m, p0m = fr[ti].mean(), fr[ci_].mean()
    p1w = np.average(fr[A == 1], weights=w_ate[A == 1]); p0w = np.average(fr[A == 0], weights=w_ate[A == 0])
    RRud = math.exp(1.0)
    bias_m = (1 + p1m * (RRud - 1)) / (1 + p0m * (RRud - 1))
    bias_w = (1 + p1w * (RRud - 1)) / (1 + p0w * (RRud - 1))
    ev = dict(matched=evalue(matched[0]), matched_ci=evalue_ci(matched[0], *matched[1]),
              iptw=evalue(iptw[0]), iptw_ci=evalue_ci(iptw[0], *iptw[1]),
              crude=evalue(crude[0]), crude_ci=evalue_ci(crude[0], *crude[1]),
              to_truth_m=evalue(matched[0] / 0.80), to_truth_w=evalue(iptw[0] / 0.80),
              paper_060=evalue(0.60), paper_064=evalue(0.64),
              p1m=p1m, p0m=p0m, p1w=p1w, p0w=p0w, RRud=RRud, RReu_m=p1m / p0m, RReu_w=p1w / p0w,
              bias_m=bias_m, bias_w=bias_w, pred_m=0.80 * bias_m, pred_w=0.80 * bias_w,
              # joint bounding factor for the E-value pair (E, E)
              )
    # ---- balancing property: within a narrow band of the PS the covariates look alike in the two groups
    band = (ps >= 0.40) & (ps < 0.45)
    bA, bB = band & (A == 1), band & (A == 0)
    bal = dict(n=int(band.sum()), nA=int(bA.sum()), nB=int(bB.sum()),
               age=(X["age"][bA].mean(), X["age"][bB].mean()), hf=(X["hf"][bA].mean(), X["hf"][bB].mean()),
               ckd=(X["ckd"][bA].mean(), X["ckd"][bB].mean()), prior=(X["prior"][bA].mean(), X["prior"][bB].mean()),
               ndrug=(X["ndrug"][bA].mean(), X["ndrug"][bB].mean()),
               smd_age=smd(X["age"][band], A[band]), smd_hf=smd(X["hf"][band], A[band]),
               rA=Y[bA].mean(), rB=Y[bB].mean())
    # ---- scikit-learn NearestNeighbors gives the same nearest B user as the with-replacement search above
    from sklearn.neighbors import NearestNeighbors
    cB = np.where(A == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(lg[cB].reshape(-1, 1))
    dist_nn, j_nn = nn.kneighbors(lg[tr].reshape(-1, 1))
    nn_same = float(np.mean(np.isclose(lg[cB[j_nn[:, 0]]], lg[cr])))
    # ---- practice question: PS of one more patient
    v3 = dict(age=70, female=0, hf=1, ckd=0, prior=0, ndrug=6, tert=0)
    lp3 = b["const"] + sum(b[k] * v3[k] for k in KEYS)
    ex3 = dict(lp=lp3, ps=1 / (1 + math.exp(-lp3)))
    events = dict(total=int(Y.sum()), epv=Y.sum() / len(KEYS))
    return dict(bal=bal, nn_same=nn_same, ex3=ex3, events=events,
                N=N, nA=int(A.sum()), nB=int(N - A.sum()), c_ps=c_ps, cal=cal, npairs=npair, tab=tab,
                rA=rA, rB=rB, eA=int(Y[A == 1].sum()), eB=int(Y[A == 0].sum()), crude=crude, crude_rd=crude_rd,
                rA_m=rA_m, rB_m=rB_m, eA_m=int(Y[ti].sum()), eB_m=int(Y[ci_].sum()), matched=matched,
                matched_rd=matched_rd, matched_naive=matched_naive, disc=disc,
                rA_w=rA_w, rB_w=rB_w, iptw=iptw, iptw_rd=iptw_rd, iptw_tr=iptw_tr, iptw_naive=iptw_naive,
                rB_att=rB_att, att=att, att_rd=att_rd, rA_ow=rA_ow, rB_ow=rB_ow, owr=owr,
                oracle=oracle, matched_o=matched_o, npairs_o=len(pairs_o), c_ps_o=c_ps_o, frail_smd_o=frail_smd_o,
                true=true, ps=ps, A=A, Y=Y, X=X, w_ate=w_ate, w_st=w_st, w_att=w_att, wsum=wsum,
                ps_rng=ps_rng, overlap=overlap, unmatched=unmatched, repl=repl, strat=strat, ps_cov=ps_cov,
                ps_cov_q=ps_cov_q, adjsets=adjsets, reg=reg, reg_frail=reg_frail, reg_or=reg_or,
                res=res, ex=ex, sk_maxdiff=sk_maxdiff, sk_def_maxdiff=sk_def_maxdiff,
                hr_crude=hr_crude, hr_iptw=hr_iptw, hr_match=hr_match, hr_match_strat=hr_match_strat,
                hr_iptw_naive=hr_iptw_naive, km=km, km_end=km_end, km_180=km_180,
                boot_ci=boot_ci, boot_se=boot_se, nc=nc, ev=ev, idx=idx, ti=ti, ci=ci_, pair_id=pair_id)


def poor_overlap():
    """Same covariates, but prescribers channel much more strongly (all treatment-model slopes x 3)."""
    from sklearn.metrics import roc_auc_score
    X, A, Y, risk, day, NC = simulate_cohort(seed=20261002, sel=3.0)
    ps, res = fit_ps(X, A, KEYS)
    c = roc_auc_score(A, ps)
    pairs, cal = greedy_match(ps, A)
    w = np.where(A == 1, 1 / ps, 1 / (1 - ps))
    pA = A.mean()
    w_st = np.where(A == 1, pA / ps, (1 - pA) / (1 - ps))
    rng_ = dict(A=(ps[A == 1].min(), ps[A == 1].max()), B=(ps[A == 0].min(), ps[A == 0].max()))
    out = (ps < 0.1) | (ps > 0.9)
    return dict(N=len(A), nA=int(A.sum()), nB=int(len(A) - A.sum()), ps=ps, A=A, c=c, npairs=len(pairs),
                share=len(pairs) / A.sum(), wmax=w.max(), wst_max=w_st.max(), wst_min=w_st.min(), rng=rng_,
                n_out=int(out.sum()), n_gt10=int((w > 10).sum()), meanA=ps[A == 1].mean(), meanB=ps[A == 0].mean(),
                essA=ess(w[A == 1]), essB=ess(w[A == 0]),
                lt05_B=float((ps[A == 0] < 0.05).mean()), gt95_A=float((ps[A == 1] > 0.95).mean()))


def p_vs_smd():
    """Same SMD, different sample sizes -> different P (two-sample z approximation)."""
    rows = []
    for d, n1, n0 in ((0.35, 41, 59), (0.35, 4053, 5947), (0.02, 3516, 3516), (0.02, 200000, 200000)):
        z = d * math.sqrt(n1 * n0 / (n1 + n0))
        rows.append(dict(d=d, n1=n1, n0=n0, z=z, p=2 * st.norm.sf(z)))
    return rows


def practice_toy():
    """Practice question (section 다): one binary confounder (CKD), RR = 0.75 in both strata."""
    S = {"CKD+": dict(nA=80, eA=24, nB=20, eB=8), "CKD-": dict(nA=120, eA=9, nB=280, eB=28)}
    nA = sum(v["nA"] for v in S.values()); nB = sum(v["nB"] for v in S.values())
    eA = sum(v["eA"] for v in S.values()); eB = sum(v["eB"] for v in S.values())
    out = dict(crude=(eA / nA, eB / nB, (eA / nA) / (eB / nB)), strata={})
    wA = wB = wa_e = wb_e = attB = attB_e = 0.0
    for k, v in S.items():
        ps = v["nA"] / (v["nA"] + v["nB"])
        out["strata"][k] = dict(ps=ps, wA=1 / ps, wB=1 / (1 - ps), rr=(v["eA"] / v["nA"]) / (v["eB"] / v["nB"]),
                                att=ps / (1 - ps))
        wA += v["nA"] / ps; wB += v["nB"] / (1 - ps); wa_e += v["eA"] / ps; wb_e += v["eB"] / (1 - ps)
        attB += v["nB"] * ps / (1 - ps); attB_e += v["eB"] * ps / (1 - ps)
    out["iptw"] = dict(nA=wA, nB=wB, eA=wa_e, eB=wb_e, rA=wa_e / wA, rB=wb_e / wB, rr=(wa_e / wA) / (wb_e / wB),
                       rd=wa_e / wA - wb_e / wB)
    out["att"] = dict(nB=attB, eB=attB_e, rB=attB_e / attB, rr=(eA / nA) / (attB_e / attB), rd=eA / nA - attB_e / attB)
    return out


def py_output():
    """The statsmodels output shown in section 나 (formula interface, readable variable names)."""
    import statsmodels.formula.api as smf
    X, A, Y, risk, day, NC = simulate_cohort()
    df = pd.DataFrame(dict(drugA=A, age=X["age"], female=X["female"], hf=X["hf"], ckd=X["ckd"], prior_hosp=X["prior"],
                           n_drug=X["ndrug"], tertiary=X["tert"]))
    fit = smf.logit("drugA ~ age + female + hf + ckd + prior_hosp + n_drug + tertiary", data=df).fit(disp=0)
    df["ps"] = fit.predict(df)
    return str(fit.summary()) + "\n" + str(df.groupby("drugA")["ps"].agg(["mean", "min", "max"]).round(3))


def compute():
    return dict(toy=ps_toy(), med=mediator_toy(), sim=ps_sim(), poor=poor_overlap(), pv=p_vs_smd(), prac=practice_toy())


def f3(t):
    return "%.3f (%.3f-%.3f)" % (t[0], t[1][0], t[1][1])


if __name__ == "__main__":
    R = compute()
    t = R["toy"]
    print("== toy (heart failure only)")
    for k, s in t["strata"].items():
        print("  ", k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items()})
    print("  crude rA %.4f rB %.4f RR %.4f ; HF share A %.3f B %.3f all %.4f ; P(A) %.4f" % (
        t["crude_rA"], t["crude_rB"], t["crude_rr"], t["hfA"], t["hfB"], t["hf_all"], t["pA"]))
    print("  ATE: wA %.1f wB %.1f eAw %.1f eBw %.1f rA %.4f rB %.4f RR %.4f RD %.4f" % (
        t["wA"], t["wB"], t["eAw"], t["eBw"], t["ate_rA"], t["ate_rB"], t["ate_rr"], t["ate_rd"]))
    print("  ATT: wB %.1f eB %.1f rA %.4f rB %.4f RR %.4f RD %.4f" % (
        t["wB_att"], t["eB_att"], t["att_rA"], t["att_rB"], t["att_rr"], t["att_rd"]))
    print("  matched: n %d per arm (HF %d) eA %.1f eB %.1f rA %.4f rB %.4f RR %.4f RD %.4f" % (
        t["m"]["n"], t["m"]["hf"], t["m"]["eA"], t["m"]["eB"], t["m_rA"], t["m_rB"], t["m_rr"], t["m_rd"]))
    print("  overlap weights: nA %.1f nB %.1f eA %.2f eB %.2f rA %.4f rB %.4f RR %.4f" % (
        t["ow"]["nA"], t["ow"]["nB"], t["ow"]["eA"], t["ow"]["eB"], t["ow_rA"], t["ow_rB"], t["ow_rr"]))
    m = R["med"]
    print("== mediator toy:", m)
    s = R["sim"]
    print("== cohort N=%d A=%d B=%d  PS c=%.4f (with frailty %.4f) caliper=%.4f pairs=%d (oracle pairs %d)" % (
        s["N"], s["nA"], s["nB"], s["c_ps"], s["c_ps_o"], s["cal"], s["npairs"], s["npairs_o"]))
    print(s["res"].summary())
    print("  OR:", np.round(np.exp(s["res"].params), 3).to_dict())
    print("  sklearn(no penalty) max |diff| %.2e ; sklearn default(L2) max |diff| %.4f" % (s["sk_maxdiff"], s["sk_def_maxdiff"]))
    for e in s["ex"]:
        print("  example:", e["label"], "lp %.4f ps %.4f" % (e["lp"], e["ps"]), {k: round(v, 3) for k, v in e["terms"].items()})
    o = s["overlap"]
    print("  PS range A %.3f-%.3f  B %.3f-%.3f ; common %.3f-%.3f ; outside common: %d (A %d, B %d) ; outside 0.1-0.9: %d" % (
        *s["ps_rng"]["A"], *s["ps_rng"]["B"], o["lo"], o["hi"], o["n_out"], o["nA_out"], o["nB_out"], o["n_0109"]))
    print("  PS mean A %.3f B %.3f median A %.3f B %.3f ; quantiles" % (o["meanA"], o["meanB"], o["medA"], o["medB"]), np.round(o["q"], 3))
    u = s["unmatched"]
    print("  unmatched A:", {k: round(v, 4) for k, v in u.items()})
    print("  matched share %.4f" % (s["npairs"] / s["nA"]))
    r = s["repl"]
    print("  with replacement: A matched %d, distinct B %d, max used %d, rA %.4f rB %.4f RR %s max|SMD| %.3f" % (
        r["nA"], r["n_distinct"], r["max_used"], r["rA"], r["rB"], f3(r["rr"]), r["smd_max"]))
    w = s["wsum"]
    print("  weights:", {k: (tuple(round(float(x), 3) for x in v) if isinstance(v, tuple) else round(float(v), 3)) for k, v in w.items()})
    print("== balance")
    for r in s["tab"]:
        print("  %-32s A %.3f (%.2f) B %.3f (%.2f) SMD %.3f p %.3g | matched A %.3f (%.2f) B %.3f (%.2f) SMD %.3f p %.3f | IPTW A %.3f B %.3f SMD %.3f | ATT SMD %.3f | OW SMD %.4f" % (
            r["en"], r["mA"], r["sA"], r["mB"], r["sB"], r["smd_pre"], r["p_pre"], r["mA_m"], r["sA_m"], r["mB_m"], r["sB_m"],
            r["smd_m"], r["p_m"], r["mA_w"], r["mB_w"], r["smd_w"], r["smd_att"], r["smd_ow"]))
    print("== Table 1 rows (before: A, B, SMD | after: A, B, SMD)")
    for r in s["tab"]:
        if r["kind"] == "cont":
            f = lambda m_, sd: f"{m_:.1f} ({sd:.1f})"
            print(f"  {r['en']:<34} {f(r['mA'], r['sA'])} | {f(r['mB'], r['sB'])} | {abs(r['smd_pre']):.2f} || "
                  f"{f(r['mA_m'], r['sA_m'])} | {f(r['mB_m'], r['sB_m'])} | {abs(r['smd_m']):.2f}   (IPTW {abs(r['smd_w']):.3f})")
        else:
            nA, nB = s["nA"], s["nB"]
            print(f"  {r['en']:<34} {round(r['mA']*nA):,} ({100*r['mA']:.1f}) | {round(r['mB']*nB):,} ({100*r['mB']:.1f}) | "
                  f"{abs(r['smd_pre']):.2f} || {round(r['mA_m']*s['npairs']):,} ({100*r['mA_m']:.1f}) | "
                  f"{round(r['mB_m']*s['npairs']):,} ({100*r['mB_m']:.1f}) | {abs(r['smd_m']):.2f}   (IPTW {abs(r['smd_w']):.3f})")
    print("== outcome")
    print("  crude: A %d/%d=%.4f B %d/%d=%.4f RR %s RD %.4f (%.4f, %.4f)" % (
        s["eA"], s["nA"], s["rA"], s["eB"], s["nB"], s["rB"], f3(s["crude"]), s["crude_rd"][0], *s["crude_rd"][1]))
    print("  matched: A %d/%d=%.4f B %d/%d=%.4f RR %s RD %.4f (%.4f, %.4f) ; naive(unpaired) RR %s ; discordant %s" % (
        s["eA_m"], s["npairs"], s["rA_m"], s["eB_m"], s["npairs"], s["rB_m"], f3(s["matched"]),
        s["matched_rd"][0], *s["matched_rd"][1], f3(s["matched_naive"]), s["disc"]))
    print("  IPTW(ATE): rA %.4f rB %.4f RR %s RD %.4f (%.4f, %.4f) ; truncated %s ; naive freq-weight %s ; bootstrap CI %.3f-%.3f (SE log %.4f vs robust %.4f)" % (
        s["rA_w"], s["rB_w"], f3(s["iptw"]), s["iptw_rd"][0], *s["iptw_rd"][1], f3(s["iptw_tr"]), f3(s["iptw_naive"]),
        *s["boot_ci"], s["boot_se"], s["iptw"][2]))
    print("  ATT weights: rA %.4f rB %.4f RR %s RD %.4f (%.4f, %.4f)" % (s["rA"], s["rB_att"], f3(s["att"]), s["att_rd"][0], *s["att_rd"][1]))
    print("  overlap weights: rA %.4f rB %.4f RR %s" % (s["rA_ow"], s["rB_ow"], f3(s["owr"])))
    print("  PS quintiles:")
    for q_ in s["strat"]["rows"]:
        print("    ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q_.items()})
    print("  MH RR over quintiles %.3f (%.3f-%.3f) ; PS (logit) as covariate %s ; PS quintile dummies %s" % (
        s["strat"]["rr"], *s["strat"]["ci"], f3(s["ps_cov"]), f3(s["ps_cov_q"])))
    print("  adjustment sets (modified Poisson):")
    for a in s["adjsets"]:
        print("    %-40s RR %s" % (a["label"], f3(a["rr"])))
    print("  logistic regression OR (7 covariates) %.3f (%.3f-%.3f)" % (s["reg_or"][0], *s["reg_or"][1]))
    print("  oracle IPTW RR %s ; oracle matched RR %s ; frailty SMD after oracle IPTW %.3f" % (
        f3(s["oracle"]), f3(s["matched_o"]), s["frail_smd_o"]))
    print("  truth:", {k: round(float(v), 4) for k, v in s["true"].items()})
    print("== time to event")
    print("  Cox crude HR %s ; IPTW HR %s ; matched (cluster) HR %s ; matched (stratified) HR %s ; naive weighted %s" % (
        f3(s["hr_crude"]), f3(s["hr_iptw"]), f3(s["hr_match"]), f3(s["hr_match_strat"]), s["hr_iptw_naive"]))
    print("  KM 1 - S(365):", {k: round(v, 4) for k, v in s["km_end"].items()}, " at 180:", {k: round(v, 4) for k, v in s["km_180"].items()})
    n = s["nc"]
    print("== negative control: crude A %d (%.4f) B %d (%.4f) RR %s ; matched A %d (%.4f) B %d (%.4f) RR %s ; IPTW %s ; oracle IPTW %s" % (
        n["nA"], n["rA"], n["nB"], n["rB"], f3(n["crude"]), n["mA"], n["rA_m"], n["mB"], n["rB_m"], f3(n["matched"]),
        f3(n["iptw"]), f3(n["oracle"])))
    print("== E-values:", {k: round(float(v), 4) for k, v in s["ev"].items()})
    p = R["poor"]
    print("== poor overlap variant: nA %d nB %d C %.4f pairs %d (%.3f of A) max w %.1f stabilized %.2f-%.2f range A %.4f-%.4f B %.4f-%.4f outside 0.1-0.9 %d ; w>10: %d ; mean PS A %.3f B %.3f ; ESS A %.0f B %.0f ; B<0.05 %.3f A>0.95 %.3f" % (
        p["nA"], p["nB"], p["c"], p["npairs"], p["share"], p["wmax"], p["wst_min"], p["wst_max"], *p["rng"]["A"], *p["rng"]["B"],
        p["n_out"], p["n_gt10"], p["meanA"], p["meanB"], p["essA"], p["essB"], p["lt05_B"], p["gt95_A"]))
    print("== P vs SMD:", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in R["pv"]])
    print("== PS band 0.40-0.45:", {k: (tuple(round(float(x), 3) for x in v) if isinstance(v, tuple) else round(float(v), 3)) for k, v in s["bal"].items()})
    print("== NearestNeighbors agrees with the with-replacement search in %.4f of A users" % s["nn_same"])
    print("== practice patient: lp %.4f ps %.4f ; events %d, events per covariate %.1f" % (s["ex3"]["lp"], s["ex3"]["ps"], s["events"]["total"], s["events"]["epv"]))
    print("== practice toy:", R["prac"])
    # ---- two-decimal strings exactly as quoted in the text
    f2 = lambda t_: "%.2f (%.2f-%.2f)" % (t_[0], t_[1][0], t_[1][1])
    pc = lambda t_: "%.1f%%p (%.1f, %.1f)" % (100 * t_[0], 100 * t_[1][0], 100 * t_[1][1])
    print("== QUOTED")
    for a in s["adjsets"]:
        print("   adj %-38s %s" % (a["label"], f2(a["rr"])))
    for k in ("crude", "matched", "matched_naive", "iptw", "iptw_tr", "iptw_naive", "att", "owr", "ps_cov", "ps_cov_q", "reg", "reg_frail",
              "oracle", "matched_o", "hr_crude", "hr_iptw", "hr_match", "hr_match_strat"):
        print("   %-16s %s" % (k, f2(s[k])))
    print("   strat MH         %.2f (%.2f-%.2f)" % (s["strat"]["rr"], *s["strat"]["ci"]))
    print("   repl             %s" % f2(s["repl"]["rr"]))
    print("   boot CI          %.2f-%.2f" % s["boot_ci"])
    print("   hr naive         %.2f (%.2f-%.2f)" % (s["hr_iptw_naive"][0], *s["hr_iptw_naive"][1]))
    print("   reg OR           %.2f (%.2f-%.2f)" % (s["reg_or"][0], *s["reg_or"][1]))
    print("   RD crude %s ; matched %s ; iptw %s ; att %s" % (pc(s["crude_rd"]), pc(s["matched_rd"]), pc(s["iptw_rd"]), pc(s["att_rd"])))
    for k in ("crude", "matched", "iptw", "oracle"):
        print("   NC %-8s %s" % (k, f2(s["nc"][k])))
    print("   risks: crude %.1f / %.1f ; matched %.1f / %.1f ; iptw %.1f / %.1f ; att B %.1f ; ow %.1f / %.1f" % (
        100 * s["rA"], 100 * s["rB"], 100 * s["rA_m"], 100 * s["rB_m"], 100 * s["rA_w"], 100 * s["rB_w"], 100 * s["rB_att"],
        100 * s["rA_ow"], 100 * s["rB_ow"]))
    print("   stratified-Cox counts:", s["hr_match_strat"][3])
    # P values (Wald, from the robust SE on the log scale) for the estimates quoted with a CI
    for k in ("crude", "matched", "iptw", "hr_crude", "hr_iptw", "hr_match"):
        z_ = math.log(s[k][0]) / s[k][2]
        print("   P %-10s z %.3f P %.4f" % (k, z_, 2 * st.norm.sf(abs(z_))))
    # with-replacement matching restricted to the A users who were also matched without replacement
    print("   with replacement, by subgroup of A users:", s["repl"]["sub"])
    print("   E-value computed from the rounded estimates quoted in the text: RR 0.91 -> %.4f ; HR 0.60 -> %.4f ; limit 0.64 -> %.4f" % (
        evalue(0.91), evalue(0.60), evalue(0.64)))
    print(py_output())
