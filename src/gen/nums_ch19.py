"""Numbers for chapter 19 (메타분석).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch19.py
Every number quoted in content/ch19.html comes from this script (fig_ch19.py imports compute()).

Running example (hypothetical): a systematic review of 10 randomized placebo-controlled trials that added
an oral anticoagulant ("drug X") to standard antiplatelet therapy after acute coronary syndrome.
Efficacy = cardiovascular death, MI or stroke (MACE); safety = major bleeding; six trials (A, B, D, F, H, I)
also report MACE by index event (STEMI vs NSTE-ACS). Trial-level counts are simulated with a fixed seed
(the seed was chosen among many so that the example shows moderate heterogeneity, one zero cell and a
subgroup contrast with P for interaction near 0.08)."""
import os, sys, warnings
import numpy as np
import scipy.stats as st

VERBOSE = __name__ == "__main__"
Z = st.norm.ppf(0.975)


def pr(*a):
    if VERBOSE:
        print(*a)


SEED = 16730
SPEC = [
    # name, year, phase, nT, nC, follow-up months, STEMI %, DAPT %, reports MACE by index event
    ("A", 2008, 2, 620, 310, 6, 55, 76, True),
    ("B", 2009, 2, 1160, 580, 6, 52, 78, True),
    ("C", 2010, 2, 240, 120, 6, 68, 97, False),
    ("D", 2011, 3, 3650, 3640, 12, 41, 82, True),
    ("E", 2011, 2, 900, 450, 6, 60, 99, False),
    ("F", 2012, 3, 5100, 5080, 18, 50, 93, True),
    ("G", 2013, 2, 410, 205, 6, 47, 88, False),
    ("H", 2014, 3, 2450, 2440, 12, 36, 90, True),
    ("I", 2016, 3, 1800, 1790, 12, 63, 95, True),
    ("J", 2018, 2, 520, 260, 9, 58, 96, False),
]


def base_mace(m):
    return 1 - np.exp(-0.0072 * m)


def base_bleed(m):
    return 1 - np.exp(-0.00085 * m)


def or2p(p0, lor_):
    o = p0 / (1 - p0) * np.exp(lor_)
    return o / (1 + o)


def simulate(seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for (nm, yr, ph, nT, nC, mo, ps, dp, rep) in SPEC:
        u = rng.normal(0, 0.08)
        nTs = int(round(nT * ps / 100)); nCs = int(round(nC * ps / 100))
        p0s = base_mace(mo) * 1.05; p0n = base_mace(mo) * 0.95
        eTs = rng.binomial(nTs, or2p(p0s, np.log(0.76) + u)); eCs = rng.binomial(nCs, p0s)
        eTn = rng.binomial(nT - nTs, or2p(p0n, np.log(0.93) + u)); eCn = rng.binomial(nC - nCs, p0n)
        ub = rng.normal(0, 0.15)
        pb = base_bleed(mo)
        bT = rng.binomial(nT, or2p(pb, np.log(2.5) + ub)); bC = rng.binomial(nC, pb)
        out.append(dict(nm=nm, yr=yr, ph=ph, nT=nT, nC=nC, mo=mo, ps=ps, dp=dp, rep=rep,
                        nTs=nTs, nCs=nCs, eTs=int(eTs), eCs=int(eCs), eTn=int(eTn), eCn=int(eCn),
                        nTn=nT - nTs, nCn=nC - nCs, eT=int(eTs + eTn), eC=int(eCs + eCn), bT=int(bT), bC=int(bC)))
    return out


def lor(a, n1, c, n0, cc=0.5):
    """log odds ratio and its variance; cc is added to the four cells of a study only if it has a zero cell"""
    a = np.asarray(a, float); c = np.asarray(c, float)
    b = np.asarray(n1, float) - a; d = np.asarray(n0, float) - c
    zc = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = [np.where(zc, x + cc, x) for x in (a, b, c, d)]
    return np.log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


def lrr(a, n1, c, n0):
    a = np.asarray(a, float); c = np.asarray(c, float)
    return np.log((a / n1) / (c / n0)), 1 / a - 1 / n1 + 1 / c - 1 / n0


def i2_ci(Q, k):
    """test-based 95% CI for I² (Higgins & Thompson 2002, via ln H)"""
    if k < 3:
        return (np.nan, np.nan)
    H = np.sqrt(max(Q, 1e-12) / (k - 1))
    if Q > k:
        se = 0.5 * (np.log(Q) - np.log(k - 1)) / (np.sqrt(2 * Q) - np.sqrt(2 * k - 3))
    else:
        se = np.sqrt(1 / (2 * (k - 2)) * (1 - 1 / (3 * (k - 2) ** 2)))
    lo, hi = np.exp(np.log(H) - Z * se), np.exp(np.log(H) + Z * se)
    f = lambda h: max(0.0, (h * h - 1) / (h * h))
    return (f(lo), f(hi))


def dl(y, v):
    """inverse-variance fixed effect and DerSimonian–Laird random effects"""
    y = np.asarray(y, float); v = np.asarray(v, float)
    w = 1 / v; k = len(y)
    fe = np.sum(w * y) / w.sum(); fe_se = 1 / np.sqrt(w.sum())
    Q = np.sum(w * (y - fe) ** 2)
    C = w.sum() - np.sum(w ** 2) / w.sum()
    t2 = max(0.0, (Q - (k - 1)) / C) if k > 1 else 0.0
    wr = 1 / (v + t2)
    re = np.sum(wr * y) / wr.sum(); re_se = 1 / np.sqrt(wr.sum())
    I2 = max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0
    out = dict(k=k, y=y, v=v, se=np.sqrt(v), w=w, wr=wr, wfe=w / w.sum(), wre=wr / wr.sum(), sw=w.sum(), swy=np.sum(w * y),
               sw2=np.sum(w ** 2), C=C, fe=fe, fe_se=fe_se, Q=Q, df=k - 1, pQ=st.chi2.sf(Q, k - 1) if k > 1 else np.nan,
               I2=I2, I2ci=i2_ci(Q, k), t2=t2, tau=np.sqrt(t2), re=re, re_se=re_se, swr=wr.sum(), swry=np.sum(wr * y))
    for key in ("fe", "re"):
        e, s = out[key], out[key + "_se"]
        out[key + "_ci"] = (e - Z * s, e + Z * s)
        out[key + "_p"] = 2 * st.norm.sf(abs(e / s))
        out[key + "_or"] = (np.exp(e), np.exp(e - Z * s), np.exp(e + Z * s))
    if k > 2:
        tc = st.t.ppf(0.975, k - 2)
        half = tc * np.sqrt(t2 + re_se ** 2)
        out["tcrit"] = tc
        out["pi"] = (re - half, re + half)
        out["pi_or"] = (np.exp(re - half), np.exp(re + half))
    return out


def wls_fixed(y, X, w):
    """weighted least squares with known variances (scale fixed at 1): coefficients, SE"""
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    b = cov @ (XtW @ y)
    return b, np.sqrt(np.diag(cov)), cov


def metareg(y, v, x):
    """random-effects meta-regression, method-of-moments residual tau²"""
    y = np.asarray(y, float); v = np.asarray(v, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    k, p = X.shape
    w = 1 / v
    b0, _, cov0 = wls_fixed(y, X, w)
    QE = np.sum(w * (y - X @ b0) ** 2)
    tr = w.sum() - np.trace(cov0 @ ((X.T * w ** 2) @ X))
    t2 = max(0.0, (QE - (k - p)) / tr)
    ws = 1 / (v + t2)
    b, se, cov = wls_fixed(y, X, ws)
    zv = b / se
    # Knapp–Hartung: scale the covariance by the weighted residual mean square, t with k - p df
    s2 = np.sum(ws * (y - X @ b) ** 2) / (k - p)
    se_kh = se * np.sqrt(s2)
    return dict(b=b, se=se, z=zv, p=2 * st.norm.sf(np.abs(zv)), QE=QE, pQE=st.chi2.sf(QE, k - p), t2=t2, k=k,
                se_kh=se_kh, p_kh=2 * st.t.sf(np.abs(b / se_kh), k - p), ws=ws)


def egger(y, v):
    """Egger regression: standard normal deviate (y/se) on precision (1/se); the intercept measures asymmetry"""
    y = np.asarray(y, float); se = np.sqrt(np.asarray(v, float))
    X = np.column_stack([np.ones(len(y)), 1 / se]); zz = y / se
    b = np.linalg.lstsq(X, zz, rcond=None)[0]
    r = zz - X @ b
    s2 = r @ r / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    t = b[0] / np.sqrt(cov[0, 0])
    return dict(b0=b[0], se0=np.sqrt(cov[0, 0]), t=t, df=len(y) - 2, p=2 * st.t.sf(abs(t), len(y) - 2), slope=b[1])


def risk_from_or(p0, OR):
    o = p0 / (1 - p0) * OR
    return o / (1 + o)


def compute():
    R = {}
    T = simulate()
    R["T"] = T
    nm = [t["nm"] for t in T]
    g = lambda key: np.array([t[key] for t in T])
    nT, nC, eT, eC, bT, bC = g("nT"), g("nC"), g("eT"), g("eC"), g("bT"), g("bC")
    R["N"] = dict(nT=int(nT.sum()), nC=int(nC.sum()), N=int(nT.sum() + nC.sum()), eT=int(eT.sum()), eC=int(eC.sum()),
                  bT=int(bT.sum()), bC=int(bC.sum()))
    pr("=" * 78, "\ntrial-level data (seed %d)" % SEED)
    for t in T:
        pr("%s %d ph%d  mo %2d STEMI %d%% DAPT %d%%  MACE %d/%d (%.1f%%) vs %d/%d (%.1f%%)  bleed %d/%d (%.2f%%) vs %d/%d (%.2f%%)"
           % (t["nm"], t["yr"], t["ph"], t["mo"], t["ps"], t["dp"], t["eT"], t["nT"], t["eT"] / t["nT"] * 100, t["eC"], t["nC"],
              t["eC"] / t["nC"] * 100, t["bT"], t["nT"], t["bT"] / t["nT"] * 100, t["bC"], t["nC"], t["bC"] / t["nC"] * 100))
    pr("totals", R["N"])

    # ------------------------------------------------------------------ 나. pooling (efficacy)
    pr("=" * 78, "\n나. pooled OR, efficacy (MACE)")
    y, v = lor(eT, nT, eC, nC)
    E = dl(y, v)
    R["eff"] = E
    for i, t in enumerate(T):
        pr("%s logOR %.4f SE %.4f var %.5f OR %.2f (%.2f-%.2f) w %.2f wFE %.1f%% w* %.2f wRE %.1f%%"
           % (t["nm"], y[i], E["se"][i], v[i], np.exp(y[i]), np.exp(y[i] - Z * E["se"][i]), np.exp(y[i] + Z * E["se"][i]),
              E["w"][i], E["wfe"][i] * 100, E["wr"][i], E["wre"][i] * 100))
    pr("sum w %.2f  sum wy %.3f  FE %.4f SE %.4f  OR %.3f (%.3f-%.3f) p=%.2g" % (E["sw"], E["swy"], E["fe"], E["fe_se"], *E["fe_or"], E["fe_p"]))
    pr("Q %.3f df %d p %.4f  I2 %.4f (%.3f-%.3f)  C %.2f tau2 %.5f tau %.4f" % (E["Q"], E["df"], E["pQ"], E["I2"], *E["I2ci"], E["C"], E["t2"], E["tau"]))
    pr("sum w* %.2f sum w*y %.3f  RE %.4f SE %.4f OR %.3f (%.3f-%.3f) p=%.2g z=%.2f" % (E["swr"], E["swry"], E["re"], E["re_se"], *E["re_or"], E["re_p"], E["re"] / E["re_se"]))
    pr("prediction interval t=%.3f: OR %.3f-%.3f ; exp(mu ± 1.96 tau) = %.3f-%.3f" % (E["tcrit"], *E["pi_or"], np.exp(E["re"] - Z * E["tau"]), np.exp(E["re"] + Z * E["tau"])))
    # worked detail: trial I
    i = nm.index("I")
    a, b_, c, d = eT[i], nT[i] - eT[i], eC[i], nC[i] - eC[i]
    pr("Trial I 2x2: a=%d b=%d c=%d d=%d; odds %.4f vs %.4f; OR %.4f; ln %.4f; var=1/a+1/b+1/c+1/d=%.5f SE %.4f; CI %.3f-%.3f; RR %.3f; risks %.2f%% vs %.2f%%"
       % (a, b_, c, d, a / b_, c / d, a * d / (b_ * c), y[i], v[i], E["se"][i], np.exp(y[i] - Z * E["se"][i]), np.exp(y[i] + Z * E["se"][i]),
          (a / nT[i]) / (c / nC[i]), a / nT[i] * 100, c / nC[i] * 100))
    R["trialI"] = dict(a=int(a), b=int(b_), c=int(c), d=int(d), OR=a * d / (b_ * c), y=y[i], v=v[i], se=E["se"][i], RR=(a / nT[i]) / (c / nC[i]))
    # statsmodels cross-check
    from statsmodels.stats.meta_analysis import combine_effects, effectsize_2proportions, effectsize_smd
    ys, vs = effectsize_2proportions(eT, nT, eC, nC, statistic="odds-ratio")
    assert np.allclose(ys, y) and np.allclose(vs, v)
    res = combine_effects(ys, vs, method_re="chi2", row_names=["Trial " + n for n in nm])
    sf = res.summary_frame()
    assert abs(res.tau2 - E["t2"]) < 1e-10 and abs(res.i2 - E["I2"]) < 1e-10 and abs(res.q - E["Q"]) < 1e-9
    assert abs(sf.loc["random effect", "eff"] - E["re"]) < 1e-10 and abs(sf.loc["fixed effect", "sd_eff"] - E["fe_se"]) < 1e-10
    import pandas as pd
    with pd.option_context("display.width", 200):
        pr(sf.round(4))
    pr("statsmodels tau2 %.6f i2 %.4f q %.4f; test_homogeneity" % (res.tau2, res.i2, res.q), res.test_homogeneity())
    R["sf"] = sf
    # Mantel–Haenszel pooled OR (statsmodels StratifiedTable)
    import statsmodels.api as sm
    tabs = [np.array([[eT[i], nT[i] - eT[i]], [eC[i], nC[i] - eC[i]]]) for i in range(len(T))]
    stt = sm.stats.StratifiedTable(tabs)
    mh = (stt.oddsratio_pooled, *stt.oddsratio_pooled_confint())
    pr("Mantel–Haenszel OR %.3f (%.3f-%.3f)" % mh)
    R["mh_eff"] = mh
    # pooled RR instead of OR
    yr_, vr_ = lrr(eT, nT, eC, nC)
    ER = dl(yr_, vr_)
    pr("pooled RR (RE) %.3f (%.3f-%.3f) I2 %.3f ; FE %.3f" % (*ER["re_or"], ER["I2"], ER["fe_or"][0]))
    R["rr"] = ER
    # HR from a reported CI
    hr, lo, hi = 0.82, 0.70, 0.96
    se_lhr = (np.log(hi) - np.log(lo)) / (2 * Z)
    pr("HR 0.82 (0.70-0.96): ln HR %.4f SE %.4f" % (np.log(hr), se_lhr))
    R["lhr"] = (np.log(hr), se_lhr)

    # ------------------------------------------------------------------ bleeding, zero cells
    pr("=" * 78, "\n나/마. bleeding (zero cell in trial C)")
    yb, vb = lor(bT, nT, bC, nC)
    B = dl(yb, vb)
    R["bleed"] = B
    for i, t in enumerate(T):
        pr("%s bleed OR %.2f (%.2f-%.2f) SE %.3f wRE %.1f%%" % (t["nm"], np.exp(yb[i]), np.exp(yb[i] - Z * B["se"][i]), np.exp(yb[i] + Z * B["se"][i]), B["se"][i], B["wre"][i] * 100))
    pr("bleeding RE OR %.3f (%.3f-%.3f) p=%.2g; FE %.3f (%.3f-%.3f); Q %.2f p %.3f I2 %.3f tau2 %.4f" % (*B["re_or"], B["re_p"], *B["fe_or"], B["Q"], B["pQ"], B["I2"], B["t2"]))
    ic = nm.index("C")
    pr("trial C with 0.5: a=%.1f b=%.1f c=%.1f d=%.1f OR %.3f SE %.3f" % (bT[ic] + .5, nT[ic] - bT[ic] + .5, bC[ic] + .5, nC[ic] - bC[ic] + .5, np.exp(yb[ic]), B["se"][ic]))
    keep = np.arange(len(T)) != ic
    Bx = dl(yb[keep], vb[keep])
    pr("excluding C: RE OR %.3f (%.3f-%.3f)" % Bx["re_or"])
    R["bleed_exC"] = Bx
    tabsb = [np.array([[bT[i], nT[i] - bT[i]], [bC[i], nC[i] - bC[i]]]) for i in range(len(T))]
    sb = sm.stats.StratifiedTable(tabsb)
    mhb = (sb.oddsratio_pooled, *sb.oddsratio_pooled_confint())
    pr("bleeding Mantel–Haenszel OR %.3f (%.3f-%.3f)" % mhb)
    R["mh_bleed"] = mhb
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_sm0, _ = effectsize_2proportions(bT, nT, bC, nC, statistic="odds-ratio")
        y_sm5, v_sm5 = effectsize_2proportions(bT, nT, bC, nC, statistic="odds-ratio", zero_correction=0.5)
    pr("statsmodels without correction: trial C logOR =", y_sm0[ic], "; zero_correction=0.5 changes ALL studies:",
       not np.allclose(np.delete(y_sm5, ic), np.delete(yb, ic)), " trial C equal:", np.isclose(y_sm5[ic], yb[ic]))
    B5 = dl(y_sm5, v_sm5)
    pr("  (0.5 added to every study: RE OR %.3f (%.3f-%.3f))" % B5["re_or"])
    R["bleed_all05"] = B5

    # multi-arm trial: trial B had two dose arms sharing one placebo group
    ib = nm.index("B")
    lowE, lowN, highE, highN = 25, 580, 22, 580
    assert lowE + highE == eT[ib] and lowN + highN == nT[ib]
    y2, v2 = lor(np.array([lowE, highE]), np.array([lowN, highN]), np.array([eC[ib], eC[ib]]), np.array([nC[ib], nC[ib]]))
    yd = np.concatenate([np.delete(y, ib), y2]); vd = np.concatenate([np.delete(v, ib), v2])
    Dd = dl(yd, vd)
    wB_ok = E["wfe"][ib]; wB_dbl = Dd["wfe"][-2:].sum()
    pr("multi-arm: low %d/%d OR %.2f, high %d/%d OR %.2f vs placebo %d/%d; combined arms weight FE %.2f%% ; double-counted weight %.2f%% (x%.2f); 1/v combined %.2f vs %.2f"
       % (lowE, lowN, np.exp(y2[0]), highE, highN, np.exp(y2[1]), eC[ib], nC[ib], wB_ok * 100, wB_dbl * 100, wB_dbl / wB_ok, 1 / v[ib], (1 / v2).sum()))
    R["multi"] = dict(lowE=lowE, highE=highE, or_low=np.exp(y2[0]), or_high=np.exp(y2[1]), w_ok=wB_ok, w_dbl=wB_dbl, iv_ok=1 / v[ib], iv_dbl=(1 / v2).sum())

    # ------------------------------------------------------------------ 다. heterogeneity, subgroups, meta-regression
    pr("=" * 78, "\n다. subgroups (six trials reporting MACE by index event)")
    rep = g("rep")
    names_rep = [n for n, r_ in zip(nm, rep) if r_]
    ySt, vSt = lor(g("eTs")[rep], g("nTs")[rep], g("eCs")[rep], g("nCs")[rep])
    yNs, vNs = lor(g("eTn")[rep], g("nTn")[rep], g("eCn")[rep], g("nCn")[rep])
    S, Nn = dl(ySt, vSt), dl(yNs, vNs)
    for lab, M, e1, n1, e0, n0 in (("STEMI", S, "eTs", "nTs", "eCs", "nCs"), ("NSTE-ACS", Nn, "eTn", "nTn", "eCn", "nCn")):
        for j, n in enumerate(names_rep):
            t = T[nm.index(n)]
            pr("  %-8s %s %d/%d vs %d/%d OR %.2f (%.2f-%.2f) w %.1f%%" % (lab, n, t[e1], t[n1], t[e0], t[n0], np.exp(M["y"][j]),
                                                                    np.exp(M["y"][j] - Z * M["se"][j]), np.exp(M["y"][j] + Z * M["se"][j]), M["wre"][j] * 100))
        pr("  %-8s RE OR %.3f (%.3f-%.3f) p=%.4f  FE %.3f (%.3f-%.3f)  Q %.2f df %d p %.3f I2 %.3f (%.2f-%.2f) tau2 %.4f; events %d/%d vs %d/%d"
           % (lab, *M["re_or"], M["re_p"], *M["fe_or"], M["Q"], M["df"], M["pQ"], M["I2"], *M["I2ci"], M["t2"],
              g(e1)[rep].sum(), g(n1)[rep].sum(), g(e0)[rep].sum(), g(n0)[rep].sum()))
    # test for subgroup differences (between-subgroup Q on the two pooled estimates) = z test on the difference
    th = np.array([S["re"], Nn["re"]]); vv = np.array([S["re_se"] ** 2, Nn["re_se"] ** 2])
    wq = 1 / vv; mid = np.sum(wq * th) / wq.sum()
    Qb = np.sum(wq * (th - mid) ** 2)
    diff = S["re"] - Nn["re"]; se_d = np.sqrt(vv.sum())
    pint = st.chi2.sf(Qb, 1)
    pr("test for subgroup differences: Qb %.3f df 1 p %.4f (I2 %.1f%%) ; z %.3f p %.4f ; ratio of ORs %.3f (%.3f-%.3f)"
       % (Qb, pint, max(0, (Qb - 1) / Qb) * 100, diff / se_d, 2 * st.norm.sf(abs(diff / se_d)), np.exp(diff), np.exp(diff - Z * se_d), np.exp(diff + Z * se_d)))
    # same test with fixed-effect subgroup estimates
    dfe = S["fe"] - Nn["fe"]; sfe = np.sqrt(S["fe_se"] ** 2 + Nn["fe_se"] ** 2)
    pr("  (fixed-effect subgroup estimates: z %.3f p %.4f)" % (dfe / sfe, 2 * st.norm.sf(abs(dfe / sfe))))
    # all 12 strata pooled (the 'overall' diamond of a subgroup forest plot) and the six trials unstratified
    All12 = dl(np.concatenate([ySt, yNs]), np.concatenate([vSt, vNs]))
    Six = dl(y[rep], v[rep])
    pr("overall of 12 strata RE OR %.3f (%.3f-%.3f) I2 %.3f pQ %.3f ; six trials unstratified RE %.3f (%.3f-%.3f) I2 %.3f"
       % (*All12["re_or"], All12["I2"], All12["pQ"], *Six["re_or"], Six["I2"]))
    # within-trial interaction pooled (difference of log ORs inside each trial)
    W = dl(ySt - yNs, vSt + vNs)
    pr("within-trial interactions pooled: ratio of ORs RE %.3f (%.3f-%.3f) p=%.4f I2 %.3f" % (*W["re_or"], W["re_p"], W["I2"]))
    R["sub"] = dict(names=names_rep, S=S, N=Nn, Qb=Qb, pint=pint, I2b=max(0, (Qb - 1) / Qb), zdiff=diff / se_d, ror=(np.exp(diff), np.exp(diff - Z * se_d), np.exp(diff + Z * se_d)),
                    all12=All12, six=Six, within=W,
                    tot=dict(S=(int(g("eTs")[rep].sum()), int(g("nTs")[rep].sum()), int(g("eCs")[rep].sum()), int(g("nCs")[rep].sum())),
                             N=(int(g("eTn")[rep].sum()), int(g("nTn")[rep].sum()), int(g("eCn")[rep].sum()), int(g("nCn")[rep].sum()))))
    # individual P values of the two subgroups (the wrong comparison)
    pr("subgroup P values: STEMI %.4f, NSTE-ACS %.4f" % (S["re_p"], Nn["re_p"]))
    # study-level subgroup: phase II vs phase III
    ph = g("ph")
    P2, P3 = dl(y[ph == 2], v[ph == 2]), dl(y[ph == 3], v[ph == 3])
    dph = P2["re"] - P3["re"]; sph = np.sqrt(P2["re_se"] ** 2 + P3["re_se"] ** 2)
    pr("phase II (k=%d) RE OR %.3f (%.3f-%.3f) I2 %.2f ; phase III (k=%d) %.3f (%.3f-%.3f) I2 %.2f ; p diff %.3f"
       % (P2["k"], *P2["re_or"], P2["I2"], P3["k"], *P3["re_or"], P3["I2"], 2 * st.norm.sf(abs(dph / sph))))
    R["phase"] = dict(P2=P2, P3=P3, p=2 * st.norm.sf(abs(dph / sph)))

    pr("-" * 40, "\nmeta-regression: log OR on % STEMI (10 trials)")
    ps = g("ps").astype(float)
    MR = metareg(y, v, (ps - 50) / 10)
    pr("intercept (at 50%% STEMI) %.4f (OR %.3f); slope per 10%%p %.4f SE %.4f z %.2f p %.3f ; exp(slope) %.3f (%.3f-%.3f) ; residual tau2 %.5f QE %.2f p %.3f ; KH p %.3f"
       % (MR["b"][0], np.exp(MR["b"][0]), MR["b"][1], MR["se"][1], MR["z"][1], MR["p"][1], np.exp(MR["b"][1]),
          np.exp(MR["b"][1] - Z * MR["se"][1]), np.exp(MR["b"][1] + Z * MR["se"][1]), MR["t2"], MR["QE"], MR["pQE"], MR["p_kh"][1]))
    Xs = sm.add_constant((ps - 50) / 10)
    f = sm.WLS(y, Xs, weights=MR["ws"]).fit(cov_type="fixed scale")
    assert np.allclose(f.params, MR["b"]) and np.allclose(f.bse, MR["se"])
    pr("  statsmodels WLS(cov_type='fixed scale') agrees: slope %.4f se %.4f p %.3f" % (f.params[1], f.bse[1], f.pvalues[1]))
    pr("  fitted OR at 40%% STEMI %.3f, at 65%% %.3f" % (np.exp(MR["b"][0] - MR["b"][1]), np.exp(MR["b"][0] + 1.5 * MR["b"][1])))
    R["mr"] = MR; R["mr_x"] = ps
    # DAPT % as a second covariate for bleeding (not used in text unless stated)
    MRb = metareg(yb, vb, (g("dp") - 90) / 10)
    pr("  (bleeding log OR on DAPT%% per 10%%p: slope %.3f p %.3f)" % (MRb["b"][1], MRb["p"][1]))

    # ------------------------------------------------------------------ 라. funnel, Egger, sensitivity
    pr("=" * 78, "\n라. funnel / Egger / sensitivity")
    EG = egger(y, v)
    pr("Egger (efficacy, k=10): intercept %.3f SE %.3f t %.2f df %d p %.3f" % (EG["b0"], EG["se0"], EG["t"], EG["df"], EG["p"]))
    fw = sm.WLS(y, sm.add_constant(E["se"]), weights=1 / v).fit()
    pr("  WLS of y on SE (weights 1/SE²): slope %.3f p %.3f (same test)" % (fw.params[1], fw.pvalues[1]))
    assert abs(fw.pvalues[1] - EG["p"]) < 1e-8
    EGb = egger(yb, vb)
    pr("Egger (bleeding): intercept %.3f p %.3f" % (EGb["b0"], EGb["p"]))
    R["egger"] = EG; R["egger_b"] = EGb
    loo = []
    for i, t in enumerate(T):
        m = dl(np.delete(y, i), np.delete(v, i))
        loo.append(dict(nm=t["nm"], or_=m["re_or"], I2=m["I2"], t2=m["t2"]))
        pr("  leave out %s: RE OR %.3f (%.3f-%.3f) I2 %.1f%%" % (t["nm"], *m["re_or"], m["I2"] * 100))
    R["loo"] = loo
    pr("fixed-effect %.3f (%.3f-%.3f) vs random-effects %.3f (%.3f-%.3f)" % (*E["fe_or"], *E["re_or"]))
    pr("phase III only: RE %.3f (%.3f-%.3f) I2 %.2f" % (*P3["re_or"], P3["I2"]))
    lowrob = np.array([n not in ("C", "G") for n in nm])     # trials C and G: some concerns/high risk of bias (hypothetical)
    LR = dl(y[lowrob], v[lowrob])
    pr("excluding C and G (risk of bias): RE %.3f (%.3f-%.3f) I2 %.2f" % (*LR["re_or"], LR["I2"]))
    R["lowrob"] = LR
    # Hartung–Knapp interval for the random-effects mean (few studies)
    q_hk = np.sum(E["wr"] * (y - E["re"]) ** 2) / (E["k"] - 1)
    se_hk = E["re_se"] * np.sqrt(q_hk)
    tc9 = st.t.ppf(0.975, E["k"] - 1)
    pr("Hartung–Knapp: SE %.4f, t9 %.3f, OR CI %.3f-%.3f (statsmodels sd_eff_w_re_hksj %.4f)" % (se_hk, tc9, np.exp(E["re"] - tc9 * se_hk), np.exp(E["re"] + tc9 * se_hk), res.sd_eff_w_re_hksj))
    R["hk"] = (np.exp(E["re"] - tc9 * se_hk), np.exp(E["re"] + tc9 * se_hk))

    # publication-bias illustration: 30 hypothetical small-to-medium trials of a drug with NO true effect (OR 1.0);
    # small trials (SE > 0.18) without a clearly favourable result (z > -0.6) stay unpublished.
    # (seed 1598 was chosen so that the complete set is symmetric and pools to about 1.0)
    rng = np.random.default_rng(1598)
    sef = np.exp(rng.uniform(np.log(0.12), np.log(0.55), 30))
    yf = rng.normal(0, 1, 30) * sef
    pub = ~((sef > 0.18) & (yf / sef > -0.6))
    ea, ep = egger(yf, sef ** 2), egger(yf[pub], sef[pub] ** 2)
    da, dp_ = dl(yf, sef ** 2), dl(yf[pub], sef[pub] ** 2)
    pr("funnel illustration: unpublished %d of 30 (published %d); all RE OR %.3f (%.3f-%.3f) Egger p %.3f; published RE OR %.3f (%.3f-%.3f) p %.3f I2 %.2f Egger intercept %.2f p %.4f"
       % ((~pub).sum(), pub.sum(), *da["re_or"], ea["p"], *dp_["re_or"], dp_["re_p"], dp_["I2"], ep["b0"], ep["p"]))
    R["fun"] = dict(se=sef, y=yf, pub=pub, all=da, publ=dp_, eg_all=ea, eg_pub=ep)

    # ------------------------------------------------------------------ 마. absolute effects, NNT / NNH
    pr("=" * 78, "\n마. NNT / NNH")
    m12 = np.array([t["mo"] == 12 for t in T])
    p0 = eC[m12].sum() / nC[m12].sum(); pb0 = bC[m12].sum() / nC[m12].sum()
    pr("12-month trials (D, H, I) control risk MACE %d/%d = %.4f ; bleeding %d/%d = %.4f" % (eC[m12].sum(), nC[m12].sum(), p0, bC[m12].sum(), nC[m12].sum(), pb0))
    p0r, pb0r = round(p0, 3), round(pb0, 3)

    def absol(p0_, ors):
        est, lo_, hi_ = ors
        p1 = [risk_from_or(p0_, o) for o in (est, lo_, hi_)]
        rd = [p - p0_ for p in p1]
        return dict(p0=p0_, p1=p1, rd=rd, n=[1 / abs(x) for x in rd], per1000=[x * 1000 for x in rd])

    A_e = absol(p0r, E["re_or"]); A_b = absol(pb0r, B["re_or"])
    pr("MACE: p0 %.3f -> p1 %.4f (%.4f, %.4f); RD %.2f%%p (%.2f, %.2f); NNT %.1f (%.1f to %.1f); per 1000: %.1f (%.1f, %.1f)"
       % (p0r, *A_e["p1"], *[x * 100 for x in A_e["rd"]], *A_e["n"], *A_e["per1000"]))
    pr("  odds0 %.4f odds1 %.4f ; RR implied %.3f" % (p0r / (1 - p0r), p0r / (1 - p0r) * E["re_or"][0], A_e["p1"][0] / p0r))
    pr("bleed: p0 %.3f -> p1 %.4f (%.4f, %.4f); RD %.2f%%p (%.2f, %.2f); NNH %.1f (%.1f to %.1f); per 1000: %.1f (%.1f, %.1f)"
       % (pb0r, *A_b["p1"], *[x * 100 for x in A_b["rd"]], *A_b["n"], *A_b["per1000"]))
    R["abs_e"] = A_e; R["abs_b"] = A_b
    lv = []
    for lab, pp in (("low", 0.04), ("mid", p0r), ("high", 0.15)):
        a_ = absol(pp, E["re_or"])
        lv.append(dict(lab=lab, p0=pp, p1=a_["p1"][0], per1000=-a_["per1000"][0], nnt=a_["n"][0], lo=-a_["per1000"][2], hi=-a_["per1000"][1], rr=a_["p1"][0] / pp))
        pr("  baseline %.1f%%: p1 %.2f%% prevented per 1000 %.1f (%.1f to %.1f) NNT %.0f ; RR %.3f" % (pp * 100, a_["p1"][0] * 100, -a_["per1000"][0], -a_["per1000"][2], -a_["per1000"][1], a_["n"][0], a_["p1"][0] / pp))
    R["levels"] = lv
    # OR vs RR at a common outcome
    for pp in (0.084, 0.40):
        pr("  OR %.3f at baseline risk %.0f%% -> risk %.2f%% RR %.3f" % (E["re_or"][0], pp * 100, risk_from_or(pp, E["re_or"][0]) * 100, risk_from_or(pp, E["re_or"][0]) / pp))
    R["or_rr40"] = risk_from_or(0.40, E["re_or"][0]) / 0.40
    # subgroup absolute effects at the same baseline risk
    A_s = absol(p0r, S["re_or"]); A_n = absol(p0r, Nn["re_or"])
    pr("STEMI: RD %.2f%%p (%.2f, %.2f) NNT %.0f (%.0f to %.0f)" % (*[x * 100 for x in A_s["rd"]], *A_s["n"]))
    pr("NSTE-ACS: RD %.2f%%p (%.2f, %.2f) NNT %.0f (NNTB %.0f to inf to NNTH %.0f)" % (*[x * 100 for x in A_n["rd"]], *A_n["n"]))
    R["abs_s"] = A_s; R["abs_n"] = A_n
    # naive NNT from summed counts (unequal allocation + different follow-up -> misleading)
    crT, crC = eT.sum() / nT.sum(), eC.sum() / nC.sum()
    pr("summed counts: %d/%d = %.2f%% vs %d/%d = %.2f%% ; RD %.2f%%p ; naive NNT %.1f ; crude OR %.3f"
       % (eT.sum(), nT.sum(), crT * 100, eC.sum(), nC.sum(), crC * 100, (crC - crT) * 100, 1 / (crC - crT), (crT / (1 - crT)) / (crC / (1 - crC))))
    R["naive"] = dict(pT=crT, pC=crC, nnt=1 / (crC - crT))
    short = np.array([t["mo"] < 12 for t in T])
    pr("  share of patients from short (<12 mo) trials: treatment %.1f%% control %.1f%%" % (nT[short].sum() / nT.sum() * 100, nC[short].sum() / nC.sum() * 100))
    R["naive"]["shT"] = nT[short].sum() / nT.sum(); R["naive"]["shC"] = nC[short].sum() / nC.sum()
    crbT, crbC = bT.sum() / nT.sum(), bC.sum() / nC.sum()
    pr("  bleeding summed: %.2f%% vs %.2f%% naive NNH %.1f" % (crbT * 100, crbC * 100, 1 / (crbT - crbC)))
    # pooled risk difference directly
    yd_, vd_ = effectsize_2proportions(eT, nT, eC, nC, statistic="diff")
    RDm = dl(yd_, vd_)
    pr("pooled risk difference (RE): %.2f%%p (%.2f to %.2f) I2 %.1f%% pQ %.3f -> NNT %.0f" % (RDm["re"] * 100, RDm["re_ci"][0] * 100, RDm["re_ci"][1] * 100, RDm["I2"] * 100, RDm["pQ"], -1 / RDm["re"]))
    R["rdm"] = RDm
    pr("NNT - NNH = %.0f - %.0f = %.0f ; per 1000: prevented %.1f, caused %.1f" % (A_e["n"][0], A_b["n"][0], A_e["n"][0] - A_b["n"][0], -A_e["per1000"][0], A_b["per1000"][0]))

    # ------------------------------------------------------------------ continuous outcome example (from the old section)
    pr("=" * 78, "\n나 (fold). continuous outcome: pharmacist-led intervention, HbA1c change")
    trials = [("Study A", 45, -0.9, 1.1, 44, -0.3, 1.2), ("Study B", 120, -0.7, 1.0, 118, -0.4, 1.0), ("Study C", 60, -1.3, 1.3, 58, -0.2, 1.2),
              ("Study D", 210, -0.5, 0.9, 205, -0.3, 0.9), ("Study E", 80, -0.8, 1.2, 82, -0.1, 1.1), ("Study F", 150, -0.6, 1.0, 148, -0.5, 1.1)]
    ym = np.array([t[2] - t[5] for t in trials]); sem = np.array([np.sqrt(t[3] ** 2 / t[1] + t[6] ** 2 / t[4]) for t in trials])
    Hm = dl(ym, sem ** 2)
    for i, t in enumerate(trials):
        pr("  %s MD %.2f SE %.3f wFE %.1f%% wRE %.1f%%" % (t[0], ym[i], sem[i], Hm["wfe"][i] * 100, Hm["wre"][i] * 100))
    pr("  FE %.3f (%.3f to %.3f); Q %.2f p %.4f I2 %.3f tau2 %.4f tau %.3f; RE %.3f (%.3f to %.3f); PI %.2f to %.2f; N %d"
       % (Hm["fe"], *Hm["fe_ci"], Hm["Q"], Hm["pQ"], Hm["I2"], Hm["t2"], Hm["tau"], Hm["re"], *Hm["re_ci"], *Hm["pi"], sum(t[1] + t[4] for t in trials)))
    smd, vsmd = effectsize_smd(np.array([t[2] for t in trials]), np.array([t[3] for t in trials]), np.array([t[1] for t in trials]),
                               np.array([t[5] for t in trials]), np.array([t[6] for t in trials]), np.array([t[4] for t in trials]))
    Hs = dl(smd, vsmd)
    pr("  SMD (Hedges g) per study", np.round(smd, 3), " pooled RE SMD %.3f (%.3f to %.3f) I2 %.2f" % (Hs["re"], *Hs["re_ci"], Hs["I2"]))
    sd_ch = np.sqrt(1.3 ** 2 + 1.4 ** 2 - 2 * 0.5 * 1.3 * 1.4)
    pr("  imputed SD of change (SD 1.3, 1.4, r 0.5): %.3f" % sd_ch)
    R["hba"] = dict(trials=trials, y=ym, se=sem, M=Hm, smd=Hs, sd_ch=sd_ch)

    # ------------------------------------------------------------------ practice numbers
    pr("=" * 78, "\npractice")
    # 나-1: trial H by hand
    ih = nm.index("H")
    a, b_, c, d = eT[ih], nT[ih] - eT[ih], eC[ih], nC[ih] - eC[ih]
    pr("나 Q1 trial H: a %d b %d c %d d %d OR %.4f lnOR %.4f var %.5f SE %.4f CI %.3f-%.3f w %.1f" % (a, b_, c, d, a * d / (b_ * c), y[ih], v[ih], E["se"][ih],
                                                                                      np.exp(y[ih] - Z * E["se"][ih]), np.exp(y[ih] + Z * E["se"][ih]), E["w"][ih]))
    # 나-2: SE from a reported RR CI
    rr, lo, hi = 0.88, 0.79, 0.98
    pr("나 Q: RR 0.88 (0.79-0.98) -> ln %.4f SE %.4f w %.1f" % (np.log(rr), (np.log(hi) - np.log(lo)) / (2 * Z), 1 / ((np.log(hi) - np.log(lo)) / (2 * Z)) ** 2))
    # 다: I² from Q
    for Q, k in ((13.28, 10), (4.2, 5), (30.0, 6), (6.0, 4)):
        pr("다 Q: Q %.2f k %d -> I2 %.1f%% p %.3f CI %.2f-%.2f" % (Q, k, max(0, (Q - (k - 1)) / Q) * 100, st.chi2.sf(Q, k - 1), *i2_ci(Q, k)))
    # 다: interaction test from two reported ORs with CI
    def from_ci(o, lo, hi):
        return np.log(o), (np.log(hi) - np.log(lo)) / (2 * Z)
    (l1, s1), (l2, s2) = from_ci(0.70, 0.52, 0.94), from_ci(0.90, 0.74, 1.09)
    zq = (l1 - l2) / np.sqrt(s1 ** 2 + s2 ** 2)
    pr("다 Q: 0.70 (0.52-0.94) vs 0.90 (0.74-1.09): SE %.4f %.4f diff %.4f se %.4f z %.3f p %.3f ratio %.3f (%.3f-%.3f)"
       % (s1, s2, l1 - l2, np.sqrt(s1 ** 2 + s2 ** 2), zq, 2 * st.norm.sf(abs(zq)), np.exp(l1 - l2), np.exp(l1 - l2 - Z * np.sqrt(s1 ** 2 + s2 ** 2)), np.exp(l1 - l2 + Z * np.sqrt(s1 ** 2 + s2 ** 2))))
    # 마: NNT at 15% and 4% baseline with OR 0.825 ; NNH
    for pp in (0.04, 0.15, 0.20):
        a_ = absol(pp, E["re_or"])
        pr("마 Q: baseline %.0f%% -> p1 %.2f%% ARR %.2f%%p NNT %.1f (%.1f to %.1f)" % (pp * 100, a_["p1"][0] * 100, -a_["rd"][0] * 100, *a_["n"]))
    for pp in (0.005, 0.03):
        a_ = absol(pp, B["re_or"])
        pr("마 Q: bleeding baseline %.1f%% -> p1 %.2f%% ARI %.2f%%p NNH %.1f" % (pp * 100, a_["p1"][0] * 100, a_["rd"][0] * 100, a_["n"][0]))
    # reciprocal arithmetic used in the real-paper guide (computed from numbers printed in the paper)
    for n_ in (84, 105, 63, 96, 130, 137):
        pr("  1000/%d = %.1f per 1000" % (n_, 1000 / n_))
    return R


if __name__ == "__main__":
    compute()


def real_paper_checks():
    """Arithmetic on numbers printed in Chiarito et al., JAMA Cardiol 2018 (used only in realpapers/ch19-s5.html,
    where each derived value is labelled as this site's own calculation)."""
    f = lambda o, lo, hi: (np.log(o), (np.log(hi) - np.log(lo)) / (2 * Z))
    (l1, s1), (l2, s2) = f(0.76, 0.66, 0.88), f(0.92, 0.78, 1.09)
    d, sd = l1 - l2, np.sqrt(s1 ** 2 + s2 ** 2)
    print("Chiarito efficacy: STEMI 0.76 (0.66-0.88) vs NSTE-ACS 0.92 (0.78-1.09): ratio of ORs %.2f (%.2f-%.2f), z %.2f, P %.3f"
          % (np.exp(d), np.exp(d - Z * sd), np.exp(d + Z * sd), d / sd, 2 * st.norm.sf(abs(d / sd))))
    (l1, s1), (l2, s2) = f(3.45, 1.95, 6.09), f(2.19, 1.38, 3.48)
    d, sd = l1 - l2, np.sqrt(s1 ** 2 + s2 ** 2)
    print("Chiarito bleeding: 3.45 (1.95-6.09) vs 2.19 (1.38-3.48): ratio %.2f (%.2f-%.2f), P %.3f" % (np.exp(d), np.exp(d - Z * sd), np.exp(d + Z * sd), 2 * st.norm.sf(abs(d / sd))))
    print("flow: 473 - 454 = %d ; 19 - 13 = %d ; 10 + 2 + 1 = %d" % (473 - 454, 19 - 13, 10 + 2 + 1))
    print("patients: 1715 + 7392 + 150 + 3491 + 15526 + 1861 = %d ; STEMI+NSTE = %d (51 missing -> %d)" % (1715 + 7392 + 150 + 3491 + 15526 + 1861, 14580 + 15036, 14580 + 15036 + 51))
    print("percent: 14580/29667 = %.1f%% ; 15036/29667 = %.1f%%" % (14580 / 29667 * 100, 15036 / 29667 * 100))
    for n_ in (84, 105, 63, 96, 130, 137):
        print("  1000/%d = %.1f per 1000" % (n_, 1000 / n_))
    print("NNT - NNH: overall %d, STEMI %d, NSTE-ACS %d" % (84 - 105, 63 - 96, 130 - 137))


if __name__ == "__main__":
    print("=" * 78, "\nreal-paper arithmetic")
    real_paper_checks()
