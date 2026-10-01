"""Numbers for chapter 14, sections 가 (McNemar), 나 (diagnostic accuracy / ROC), 다 (propensity scores).

Run:  source /home/claude/pylibs/env.sh && python3 gen/nums_ch14a.py
Every number quoted in content/_ch14/sa.html comes from compute() below.
"""
import math
import numpy as np
import scipy.stats as st

Z = st.norm.ppf(0.975)


# ----------------------------------------------------------------------------- 가. McNemar
def mcnemar_part():
    # 120 patients, adherence (PDC >= 80%) in the 6 months before vs after pharmacist counseling
    # rows = before (adherent, non-adherent), cols = after (adherent, non-adherent)
    a, c = 47, 7    # before adherent: after adherent (a), after non-adherent (c: adherent -> non-adherent)
    b, d = 25, 41   # before non-adherent: after adherent (b: non-adherent -> adherent), after non-adherent (d)
    n = a + b + c + d
    before = a + c
    after = a + b
    p_before, p_after = before / n, after / n
    diff = (b - c) / n
    se_diff = math.sqrt(b + c - (b - c) ** 2 / n) / n
    ci = (diff - Z * se_diff, diff + Z * se_diff)
    chi2 = (b - c) ** 2 / (b + c)
    p_chi2 = st.chi2.sf(chi2, 1)
    chi2_cc = (abs(b - c) - 1) ** 2 / (b + c)
    p_cc = st.chi2.sf(chi2_cc, 1)
    p_exact = st.binomtest(min(b, c), b + c, 0.5).pvalue
    p_one_tail = st.binom.cdf(min(b, c), b + c, 0.5)
    # wrong: ordinary chi-square treating before and after as two independent groups of 120
    tab_wrong = np.array([[before, n - before], [after, n - after]])
    chi_w, p_w, _, _ = st.chi2_contingency(tab_wrong, correction=False)
    chi_wc, p_wc, _, _ = st.chi2_contingency(tab_wrong, correction=True)
    # matched case-control pairs from chapter 9: case exposed/control not = 50, case not/control exposed = 25
    mb, mc = 50, 25
    or_m = mb / mc
    se_log = math.sqrt(1 / mb + 1 / mc)
    or_ci = (math.exp(math.log(or_m) - Z * se_log), math.exp(math.log(or_m) + Z * se_log))
    chi_m = (mb - mc) ** 2 / (mb + mc)
    p_m = st.chi2.sf(chi_m, 1)
    p_m_exact = st.binomtest(mc, mb + mc, 0.5).pvalue
    # statsmodels check
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        sm_exact = mcnemar(np.array([[a, c], [b, d]]), exact=True)
        sm_asym = mcnemar(np.array([[a, c], [b, d]]), exact=False, correction=False)
        sm = (float(sm_exact.statistic), float(sm_exact.pvalue), float(sm_asym.statistic), float(sm_asym.pvalue))
    except Exception:  # pragma: no cover
        sm = None
    return dict(a=a, b=b, c=c, d=d, n=n, before=before, after=after, p_before=p_before, p_after=p_after,
                diff=diff, se_diff=se_diff, ci=ci, chi2=chi2, p_chi2=p_chi2, chi2_cc=chi2_cc, p_cc=p_cc,
                p_exact=p_exact, p_one_tail=p_one_tail, chi_w=chi_w, p_w=p_w, chi_wc=chi_wc, p_wc=p_wc,
                or_m=or_m, or_ci=or_ci, se_log=se_log, chi_m=chi_m, p_m=p_m, p_m_exact=p_m_exact, sm=sm)


# ----------------------------------------------------------------------------- 나. ROC
SCORES = np.arange(0, 9)
CASES = np.array([1, 2, 4, 6, 9, 11, 12, 9, 6])        # non-adherent (PDC < 80%), n = 60
NONCASES = np.array([14, 22, 26, 22, 22, 15, 10, 6, 3])  # adherent, n = 140


def auc_delong(pos, neg):
    """AUC (ties count 1/2) with DeLong structural-component variance."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    m, n = len(pos), len(neg)
    psi = (pos[:, None] > neg[None, :]).astype(float) + 0.5 * (pos[:, None] == neg[None, :])
    auc = psi.mean()
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    var = v10.var(ddof=1) / m + v01.var(ddof=1) / n
    return auc, math.sqrt(var)


def roc_part():
    pos = np.repeat(SCORES, CASES)
    neg = np.repeat(SCORES, NONCASES)
    m, n = len(pos), len(neg)
    rows = []
    for k in range(0, 10):  # rule: positive if score >= k (k = 9 -> nobody positive)
        tp = int((pos >= k).sum()); fn = m - tp
        fp = int((neg >= k).sum()); tn = n - fp
        sens = tp / m; spec = tn / n
        ppv = tp / (tp + fp) if tp + fp else float("nan")
        npv = tn / (tn + fn) if tn + fn else float("nan")
        acc = (tp + tn) / (m + n)
        rows.append(dict(k=k, tp=tp, fn=fn, fp=fp, tn=tn, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
                         youden=sens + spec - 1))
    auc, se = auc_delong(pos, neg)
    # CI on the logit scale is also common; here the plain Wald interval (as many programs report)
    ci = (auc - Z * se, auc + Z * se)
    # trapezoid check
    fpr = [1 - r["spec"] for r in rows][::-1]
    tpr = [r["sens"] for r in rows][::-1]
    auc_trap = float(np.trapezoid(tpr, fpr))
    # Mann-Whitney U
    U = st.mannwhitneyu(pos, neg, alternative="two-sided")
    # pairs
    gt = int((pos[:, None] > neg[None, :]).sum()); eq = int((pos[:, None] == neg[None, :]).sum())
    try:
        from sklearn.metrics import roc_auc_score
        auc_sk = roc_auc_score(np.r_[np.ones(m), np.zeros(n)], np.r_[pos, neg])
    except Exception:  # pragma: no cover
        auc_sk = None
    # PPV / NPV at cutoff >=5 for different prevalences (same sens/spec), per 1,000 people
    r5 = rows[5]
    prev_tab = []
    for prev in (0.30, 0.10, 0.02):
        N = 1000
        dis = N * prev; nodis = N - dis
        tp = r5["sens"] * dis; fp = (1 - r5["spec"]) * nodis
        tn = r5["spec"] * nodis; fn = dis - tp
        prev_tab.append(dict(prev=prev, dis=dis, tp=tp, fp=fp, tn=tn, fn=fn, ppv=tp / (tp + fp), npv=tn / (tn + fn),
                             acc=(tp + tn) / N))
    # "always negative" rule: accuracy = 1 - prevalence
    acc_all_neg = n / (m + n)
    best = max(rows, key=lambda r: r["youden"])
    return dict(m=m, n=n, rows=rows, auc=auc, se=se, ci=ci, auc_trap=auc_trap, U=float(U.statistic),
                p_mw=float(U.pvalue), gt=gt, eq=eq, pairs=m * n, auc_sk=auc_sk, prev_tab=prev_tab,
                acc_all_neg=acc_all_neg, best=best)


# ----------------------------------------------------------------------------- 다. Propensity scores
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
    for k, s in S.items():
        ps = s["nA"] / (s["nA"] + s["nB"])
        w_a, w_b = 1 / ps, 1 / (1 - ps)
        w_b_att = ps / (1 - ps)
        out[k] = dict(s, ps=ps, w_a=w_a, w_b=w_b, w_b_att=w_b_att, rA=s["eA"] / s["nA"], rB=s["eB"] / s["nB"],
                      rr=(s["eA"] / s["nA"]) / (s["eB"] / s["nB"]),
                      nA_w=s["nA"] * w_a, nB_w=s["nB"] * w_b, eA_w=s["eA"] * w_a, eB_w=s["eB"] * w_b,
                      nB_att=s["nB"] * w_b_att, eB_att=s["eB"] * w_b_att)
        for kk in tot:
            tot[kk] += s[kk]
        wA += s["nA"] * w_a; wB += s["nB"] * w_b
        eAw += s["eA"] * w_a; eBw += s["eB"] * w_b
        wB_att += s["nB"] * w_b_att; eB_att += s["eB"] * w_b_att
    crude_rA = tot["eA"] / tot["nA"]; crude_rB = tot["eB"] / tot["nB"]
    ate_rA = eAw / wA; ate_rB = eBw / wB
    att_rA = crude_rA; att_rB = eB_att / wB_att
    return dict(strata=out, tot=tot, crude_rA=crude_rA, crude_rB=crude_rB, crude_rr=crude_rA / crude_rB,
                wA=wA, wB=wB, eAw=eAw, eBw=eBw, ate_rA=ate_rA, ate_rB=ate_rB, ate_rr=ate_rA / ate_rB,
                ate_rd=ate_rA - ate_rB, wB_att=wB_att, eB_att=eB_att, att_rA=att_rA, att_rB=att_rB,
                att_rr=att_rA / att_rB, att_rd=att_rA - att_rB)


COVS = [  # (key, Korean label, English label, kind)
    ("age", "나이", "Age, y, mean (SD)", "cont"),
    ("female", "여성", "Female", "bin"),
    ("hf", "심부전", "Heart failure", "bin"),
    ("ckd", "만성콩팥병", "Chronic kidney disease", "bin"),
    ("prior", "지난 1년 입원", "Hospitalization in prior year", "bin"),
    ("ndrug", "복용 약물 계열 수", "No. of drug classes, mean (SD)", "cont"),
    ("tert", "상급종합병원 처방", "Prescribed at tertiary hospital", "bin"),
]


def simulate_cohort(seed=20260930, N=10000):
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
    lp = (-0.9 + 0.025 * (age - 66) - 0.1 * female + 0.7 * hf + 0.5 * ckd + 0.5 * prior
          + 0.07 * (ndrug - 6) + 0.6 * tert + 0.9 * frail)
    A = rng.binomial(1, 1 / (1 + np.exp(-lp)))
    # outcome: 1-year hospitalization, log-linear risk -> conditional RR of drug A = 0.80 everywhere
    lr = (np.log(0.065) + np.log(0.80) * A + 0.02 * (age - 66) + 0.6 * hf + 0.35 * ckd + 0.55 * prior
          + 0.04 * (ndrug - 6) + 0.25 * tert + 1.0 * frail)
    risk = np.minimum(np.exp(lr), 0.95)
    Y = rng.binomial(1, risk)
    X = dict(age=age, female=female, hf=hf, ckd=ckd, prior=prior, ndrug=ndrug, tert=tert, frail=frail)
    return X, A, Y, risk


def smd(x, g, w=None):
    x = np.asarray(x, float)
    if w is None:
        w = np.ones_like(x)
    m1 = np.average(x[g == 1], weights=w[g == 1]); m0 = np.average(x[g == 0], weights=w[g == 0])
    v1 = np.average((x[g == 1] - m1) ** 2, weights=w[g == 1]); v0 = np.average((x[g == 0] - m0) ** 2, weights=w[g == 0])
    return (m1 - m0) / math.sqrt((v1 + v0) / 2)


def fit_ps(X, A, keys):
    import statsmodels.api as sm
    M = np.column_stack([X[k] for k in keys]).astype(float)
    M = sm.add_constant(M)
    res = sm.Logit(A, M).fit(disp=0)
    ps = res.predict(M)
    return ps, res


def greedy_match(ps, A, caliper_sd=0.2, seed=7):
    """1:1 nearest-neighbour matching on the logit of the PS, without replacement, caliper 0.2 SD."""
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
        dist[~avail] = np.inf
        j = int(np.argmin(dist))
        if dist[j] <= cal:
            pairs.append((t, c_idx[j]))
            avail[j] = False
    return np.array(pairs), cal


def rr_gee(Y, A, groups, w=None):
    """Modified Poisson (log link) with robust sandwich SE; groups = pair id or subject id."""
    import statsmodels.api as sm
    Xd = sm.add_constant(A.astype(float))
    mod = sm.GEE(Y, Xd, groups=groups, family=sm.families.Poisson(), cov_struct=sm.cov_struct.Independence(),
                 weights=w)
    res = mod.fit()
    b, se = res.params[1], res.bse[1]
    return math.exp(b), (math.exp(b - Z * se), math.exp(b + Z * se)), se


def ps_sim():
    X, A, Y, risk = simulate_cohort()
    N = len(A)
    keys = [k for k, *_ in COVS]
    ps, res = fit_ps(X, A, keys)
    # c statistic of the PS model
    from sklearn.metrics import roc_auc_score
    c_ps = roc_auc_score(A, ps)
    pairs, cal = greedy_match(ps, A)
    ti, ci_ = pairs[:, 0], pairs[:, 1]
    idx = np.r_[ti, ci_]
    Am = A[idx]
    # SMDs before and after matching, and after IPTW
    w_ate = np.where(A == 1, 1 / ps, 1 / (1 - ps))
    pA = A.mean()
    w_st = np.where(A == 1, pA / ps, (1 - pA) / (1 - ps))  # stabilized
    tab = []
    for k, ko, en, kind in COVS + [("frail", "허약(청구자료에 없음)", "Frailty (unmeasured)", "bin")]:
        x = X[k]
        row = dict(key=k, ko=ko, en=en, kind=kind,
                   mA=x[A == 1].mean(), mB=x[A == 0].mean(), sA=x[A == 1].std(ddof=1), sB=x[A == 0].std(ddof=1),
                   smd_pre=smd(x, A),
                   mA_m=x[ti].mean(), mB_m=x[ci_].mean(), sA_m=x[ti].std(ddof=1), sB_m=x[ci_].std(ddof=1),
                   smd_m=smd(x[idx], Am), smd_w=smd(x, A, w_ate))
        tab.append(row)
    # outcomes
    rA, rB = Y[A == 1].mean(), Y[A == 0].mean()
    crude = rr_gee(Y, A, np.arange(N))
    rA_m, rB_m = Y[ti].mean(), Y[ci_].mean()
    pair_id = np.r_[np.arange(len(pairs)), np.arange(len(pairs))]
    matched = rr_gee(Y[idx], Am, pair_id)
    rA_w = np.average(Y[A == 1], weights=w_ate[A == 1]); rB_w = np.average(Y[A == 0], weights=w_ate[A == 0])
    iptw = rr_gee(Y, A, np.arange(N), w=w_st)
    # "oracle" analysis with the unmeasured frailty included in the PS (only possible in a simulation)
    ps_o, _ = fit_ps(X, A, keys + ["frail"])
    w_o = np.where(A == 1, pA / ps_o, (1 - pA) / (1 - ps_o))
    oracle = rr_gee(Y, A, np.arange(N), w=w_o)
    pairs_o, _ = greedy_match(ps_o, A)
    idx_o = np.r_[pairs_o[:, 0], pairs_o[:, 1]]
    matched_o = rr_gee(Y[idx_o], A[idx_o], np.r_[np.arange(len(pairs_o)), np.arange(len(pairs_o))])
    # true marginal RR in the whole cohort and among the treated (from the known risk function)
    # (risk under A=1 vs A=0 for every person; cap rarely binds)
    lr0 = np.log(risk) - np.log(0.80) * A  # risk if untreated (ignoring cap)
    r0 = np.minimum(np.exp(lr0), 0.95); r1 = np.minimum(np.exp(lr0 + np.log(0.80)), 0.95)
    true_ate = r1.mean() / r0.mean(); true_att = r1[A == 1].mean() / r0[A == 1].mean()
    # overlap: PS ranges
    ps_rng = dict(A=(ps[A == 1].min(), ps[A == 1].max()), B=(ps[A == 0].min(), ps[A == 0].max()))
    # PS stratification by quintile (for text)
    q = np.quantile(ps, [0.2, 0.4, 0.6, 0.8])
    strata = np.digitize(ps, q)
    srr = []
    for s in range(5):
        m_ = strata == s
        srr.append((int(m_.sum()), int(A[m_].sum()), Y[m_ & (A == 1)].mean(), Y[m_ & (A == 0)].mean()))
    return dict(N=N, nA=int(A.sum()), nB=int(N - A.sum()), c_ps=c_ps, cal=cal, npairs=len(pairs), tab=tab,
                rA=rA, rB=rB, eA=int(Y[A == 1].sum()), eB=int(Y[A == 0].sum()), crude=crude,
                rA_m=rA_m, rB_m=rB_m, eA_m=int(Y[ti].sum()), eB_m=int(Y[ci_].sum()), matched=matched,
                rA_w=rA_w, rB_w=rB_w, iptw=iptw, oracle=oracle, matched_o=matched_o, npairs_o=len(pairs_o),
                true_ate=true_ate, true_att=true_att, ps=ps, A=A, w_ate=w_ate, w_st=w_st,
                wmin=w_st.min(), wmax=w_st.max(), ps_rng=ps_rng, srr=srr, coefs=res.params)


def compute():
    return dict(mc=mcnemar_part(), roc=roc_part(), toy=ps_toy(), sim=ps_sim())


if __name__ == "__main__":
    R = compute()
    mc = R["mc"]
    print("== 가 McNemar")
    for k, v in mc.items():
        print(f"  {k}: {v}")
    ro = R["roc"]
    print("== 나 ROC  m=%d n=%d" % (ro["m"], ro["n"]))
    for r in ro["rows"]:
        print("  >=%d  TP %3d FN %3d FP %3d TN %3d  sens %.4f spec %.4f ppv %.4f npv %.4f acc %.4f J %.4f" % (
            r["k"], r["tp"], r["fn"], r["fp"], r["tn"], r["sens"], r["spec"], r["ppv"], r["npv"], r["acc"], r["youden"]))
    print("  AUC %.4f SE %.4f CI %.3f-%.3f trap %.4f sk %s U %.1f p_mw %.3g gt %d eq %d pairs %d" % (
        ro["auc"], ro["se"], ro["ci"][0], ro["ci"][1], ro["auc_trap"], ro["auc_sk"], ro["U"], ro["p_mw"],
        ro["gt"], ro["eq"], ro["pairs"]))
    for p in ro["prev_tab"]:
        print("  prev %.2f: dis %.1f TP %.1f FP %.1f TN %.1f FN %.1f PPV %.4f NPV %.4f acc %.4f" % (
            p["prev"], p["dis"], p["tp"], p["fp"], p["tn"], p["fn"], p["ppv"], p["npv"], p["acc"]))
    print("  always-negative accuracy %.3f ; best Youden cutoff >=%d" % (ro["acc_all_neg"], ro["best"]["k"]))
    t = R["toy"]
    print("== 다 toy")
    for k, s in t["strata"].items():
        print("  ", k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items()})
    print("  crude rA %.4f rB %.4f RR %.4f" % (t["crude_rA"], t["crude_rB"], t["crude_rr"]))
    print("  ATE: wA %.1f wB %.1f eAw %.1f eBw %.1f rA %.4f rB %.4f RR %.4f RD %.4f" % (
        t["wA"], t["wB"], t["eAw"], t["eBw"], t["ate_rA"], t["ate_rB"], t["ate_rr"], t["ate_rd"]))
    print("  ATT: wB %.1f eB %.1f rA %.4f rB %.4f RR %.4f RD %.4f" % (
        t["wB_att"], t["eB_att"], t["att_rA"], t["att_rB"], t["att_rr"], t["att_rd"]))
    s = R["sim"]
    print("== 다 simulation N=%d A=%d B=%d  PS c=%.3f caliper=%.4f pairs=%d (oracle pairs %d)" % (
        s["N"], s["nA"], s["nB"], s["c_ps"], s["cal"], s["npairs"], s["npairs_o"]))
    for r in s["tab"]:
        print("  %-26s A %.3f (%.2f) B %.3f (%.2f) SMD %.3f | matched A %.3f (%.2f) B %.3f (%.2f) SMD %.3f | IPTW SMD %.3f" % (
            r["en"], r["mA"], r["sA"], r["mB"], r["sB"], r["smd_pre"], r["mA_m"], r["sA_m"], r["mB_m"], r["sB_m"],
            r["smd_m"], r["smd_w"]))
    print("  crude: A %d/%d=%.4f B %d/%d=%.4f RR %.3f CI %.3f-%.3f" % (
        s["eA"], s["nA"], s["rA"], s["eB"], s["nB"], s["rB"], s["crude"][0], *s["crude"][1]))
    print("  matched: A %d/%d=%.4f B %d/%d=%.4f RR %.3f CI %.3f-%.3f" % (
        s["eA_m"], s["npairs"], s["rA_m"], s["eB_m"], s["npairs"], s["rB_m"], s["matched"][0], *s["matched"][1]))
    print("  IPTW(ATE): rA %.4f rB %.4f RR %.3f CI %.3f-%.3f ; stabilized w range %.2f-%.2f" % (
        s["rA_w"], s["rB_w"], s["iptw"][0], *s["iptw"][1], s["wmin"], s["wmax"]))
    print("  oracle IPTW RR %.4f CI %.3f-%.3f ; oracle matched RR %.4f CI %.3f-%.3f" % (
        s["oracle"][0], *s["oracle"][1], s["matched_o"][0], *s["matched_o"][1]))
    print("  true marginal RR: ATE %.4f ATT %.4f" % (s["true_ate"], s["true_att"]))
    print("  PS range A %.3f-%.3f  B %.3f-%.3f" % (*s["ps_rng"]["A"], *s["ps_rng"]["B"]))
    print("  PS quintile strata (n, nA, rA, rB):", [(a, b, round(c, 3), round(d, 3)) for a, b, c, d in s["srr"]])
    print("  PS coefs:", np.round(s["coefs"], 3))
    # formatted rows for the Table 1 in the paper box (section 다)
    print("== Table 1 rows (before: A, B, SMD | after: A, B, SMD)")
    X, A, Y, _ = simulate_cohort()
    for r in s["tab"]:
        if r["kind"] == "cont":
            f = lambda m, sd: f"{m:.1f} ({sd:.1f})"
            print(f"  {r['en']:<34} {f(r['mA'], r['sA'])} | {f(r['mB'], r['sB'])} | {abs(r['smd_pre']):.2f} || "
                  f"{f(r['mA_m'], r['sA_m'])} | {f(r['mB_m'], r['sB_m'])} | {abs(r['smd_m']):.2f}   (IPTW {abs(r['smd_w']):.3f})")
        else:
            nA, nB = s["nA"], s["nB"]
            print(f"  {r['en']:<34} {round(r['mA']*nA):,} ({100*r['mA']:.1f}) | {round(r['mB']*nB):,} ({100*r['mB']:.1f}) | "
                  f"{abs(r['smd_pre']):.2f} || {round(r['mA_m']*s['npairs']):,} ({100*r['mA_m']:.1f}) | "
                  f"{round(r['mB_m']*s['npairs']):,} ({100*r['mB_m']:.1f}) | {abs(r['smd_m']):.2f}   (IPTW {abs(r['smd_w']):.3f})")
    # extra checks quoted in the text
    se_ind = math.sqrt(0.45 * 0.55 / 120 + 0.60 * 0.40 / 120)
    print("== extra: independent-proportions SE %.4f, CI %.3f to %.3f" % (se_ind, 0.15 - Z * se_ind, 0.15 + Z * se_ind))
    for k_, n_ in ((38, 60), (106, 140), (38, 72), (106, 128)):
        ci_cp = st.binomtest(k_, n_).proportion_ci(method="exact")
        print("  %d/%d = %.4f  Clopper-Pearson %.3f-%.3f" % (k_, n_, k_ / n_, ci_cp.low, ci_cp.high))
    print("  matched share of A users %.4f ; unmatched A %d" % (s["npairs"] / s["nA"], s["nA"] - s["npairs"]))
    print("  PS mean A %.3f B %.3f" % (s["ps"][s["A"] == 1].mean(), s["ps"][s["A"] == 0].mean()))
