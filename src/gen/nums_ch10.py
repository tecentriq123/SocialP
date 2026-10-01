"""Chapter 10: every number in the text, tables, figures and paper boxes is computed here.

Run:  python3 gen/nums_ch10.py            (main example + checks, about 1-2 minutes)
      python3 gen/nums_ch10.py --rep      (also the 500-replicate missing-data simulation, several minutes)
fig_ch10.py imports example() so that figures use exactly the same numbers.

The running example (가상의 예시): 100 patients with type 2 diabetes randomized 1:1 to a pharmacist-led
medication-management intervention or usual care; HbA1c at 0, 3, 6, 12 months; PDC >= 80% in each
interval (0-3, 3-6, 6-12 months). Data are simulated by lib_ch10.simulate(SEED) with known parameters:
true between-group difference in HbA1c change at month 12 = (-0.055 - (-0.015)) * 12 = -0.48 %p.
"""
import os, sys
import numpy as np
import scipy.stats as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_ch10 import (simulate, rm_anova, gg_box, LMM, gee, glm_logit, glmm_logit_ri, marginal_from_conditional,
                      ols, expit, MONTHS, orth_contrasts)

SEED = 1436
TRUE_DIFF12 = (-0.055 - (-0.015)) * 12
Z = st.norm.ppf(0.975)


# ------------------------------------------------------------------ helpers
def ancova12(D, y12, keep):
    """ANCOVA at month 12: change ~ group + baseline (centered), OLS; returns (est, se, lo, hi, n)."""
    g = D["g"][keep]
    b = D["full"][keep, 0]
    yy = y12[keep]
    X = np.column_stack([np.ones(keep.sum()), g, b - b.mean()])
    bb, cov, s2 = ols(yy - b, X)
    se = np.sqrt(cov[1, 1])
    q = st.t.ppf(0.975, keep.sum() - 3)
    return dict(est=bb[1], se=se, lo=bb[1] - q * se, hi=bb[1] + q * se, n=int(keep.sum()),
                p=2 * st.t.sf(abs(bb[1] / se), keep.sum() - 3))


def locf(y):
    out = y.copy()
    for j in range(1, out.shape[1]):
        m = np.isnan(out[:, j])
        out[m, j] = out[m, j - 1]
    return out


def mmrm(y, g, kind="un", reml=True):
    """change at 3, 6, 12 ~ visit + group:visit + baseline(centered):visit, structured covariance.
    beta order: [v3, v6, v12, g*v3, g*v6, g*v12, cb*v3, cb*v6, cb*v12]"""
    base = y[:, 0]
    chg = y[:, 1:] - base[:, None]
    anal = ~np.isnan(chg).all(1)
    cb = base - base[anal].mean()
    N = len(g)
    X = np.zeros((N, 3, 9))
    for j in range(3):
        X[:, j, j] = 1
        X[:, j, 3 + j] = g
        X[:, j, 6 + j] = cb
    m = LMM(chg, X, [3, 6, 12], kind, reml=reml).fit()
    m.anal = anal
    m.base_mean = base[anal].mean()
    return m


def long_hba1c(D, y=None):
    """observed rows: id (1..N), g, month, visit index, value"""
    y = D["y"] if y is None else y
    rows = []
    for i in range(D["N"]):
        for j in range(4):
            if not np.isnan(y[i, j]):
                rows.append((i + 1, D["g"][i], MONTHS[j], j, y[i, j]))
    return np.array(rows)


def long_adh(D, a=None):
    a = D["adh"] if a is None else a
    rows = []
    for i in range(D["N"]):
        for j in range(3):
            if not np.isnan(a[i, j]):
                rows.append((i + 1, D["g"][i], j, a[i, j]))
    return np.array(rows)


def fmt(x, nd=2):
    return f"{x:.{nd}f}".replace("-", "−")


# ------------------------------------------------------------------ toy example (가 절)
TOY = np.array([[9.2, 9.0, 8.5],
                [7.4, 7.3, 6.9],
                [8.4, 7.8, 7.8],
                [10.2, 9.9, 9.6]])


def toy():
    R = rm_anova(TOY)
    y = TOY
    gm = y.mean()
    ss_tot = ((y - gm) ** 2).sum()
    ss_time = R["SS"]["time"]
    ss_w = ss_tot - ss_time            # one-way ANOVA ignoring patients
    df_w = y.size - 3
    F1 = (ss_time / 2) / (ss_w / df_w)
    p1 = st.f.sf(F1, 2, df_w)
    return dict(R=R, gm=gm, vis_mean=y.mean(0), pat_mean=y.mean(1), ss_tot=ss_tot, ss_w=ss_w, df_w=df_w,
                ms_w=ss_w / df_w, F1=F1, p1=p1)


# ------------------------------------------------------------------ main example
def example(rep=False, verbose=False):
    D = simulate(SEED)
    g, y, full = D["g"], D["y"], D["full"]
    N = D["N"]
    E = dict(D=D)
    # ---- descriptive: n, mean, SD, 95% CI of mean per visit (all available data)
    desc = {}
    for k in (0, 1):
        yy = y[g == k]
        n = (~np.isnan(yy)).sum(0)
        m = np.nanmean(yy, 0)
        s = np.nanstd(yy, 0, ddof=1)
        q = st.t.ppf(0.975, n - 1)
        desc[k] = dict(n=n, mean=m, sd=s, lo=m - q * s / np.sqrt(n), hi=m + q * s / np.sqrt(n))
    E["desc"] = desc
    E["full_mean"] = {k: full[g == k].mean(0) for k in (0, 1)}
    # ---- missing-data patterns
    obs = D["obs"]
    pats = {}
    for i in range(N):
        key = "".join("O" if o else "-" for o in obs[i])
        pats.setdefault(key, [0, 0])[g[i]] += 1
    E["patterns"] = pats
    E["n_obs"] = int(obs.sum())
    E["cc"] = obs.all(1)
    E["n_cc"] = {k: int((E["cc"] & (g == k)).sum()) for k in (0, 1)}
    E["dropped"] = {k: {j: int(((D["dropped_at"] == j) & (g == k)).sum()) for j in (1, 2, 3)} for k in (0, 1)}
    # ---- naive vs patient-clustered SE for the pooled group difference (all observations)
    L = long_hba1c(D)
    Xp = np.column_stack([np.ones(len(L)), L[:, 1]])
    b, cov, s2 = ols(L[:, 4], Xp)
    Gp = gee(L[:, 4], Xp, L[:, 0], "gaussian", "independence", pos=L[:, 3], k=4)
    E["pooled"] = dict(n=len(L), est=b[1], se_naive=np.sqrt(cov[1, 1]), se_rob=Gp["se_robust"][1])
    # ---- RM-ANOVA (complete cases)
    cc = E["cc"]
    R = rm_anova(y[cc], g[cc])
    E["rm"] = R
    S = R["S"]
    pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (0, 3)]
    E["vardiff"] = [(MONTHS[a], MONTHS[b_], S[a, a] + S[b_, b_] - 2 * S[a, b_]) for a, b_ in pairs]
    E["cc_means"] = R["cell"]
    # simple difference in change at 12 months among completers (unadjusted)
    ch = y[cc, 3] - y[cc, 0]
    E["cc_change"] = {k: ch[g[cc] == k].mean() for k in (0, 1)}
    # ---- LMM, linear time, all observations incl. baseline (random intercept / + slope)
    t = MONTHS.astype(float)
    X = np.zeros((N, 4, 4))
    X[:, :, 0] = 1
    X[:, :, 1] = g[:, None]
    X[:, :, 2] = t
    X[:, :, 3] = g[:, None] * t
    ri = LMM(y, X, t, "cs").fit()
    rs = LMM(y, X, t, "rirs").fit()
    ri_ml = LMM(y, X, t, "cs", reml=False).fit()
    rs_ml = LMM(y, X, t, "rirs", reml=False).fit()
    E["ri"], E["rs"], E["ri_ml"], E["rs_ml"] = ri, rs, ri_ml, rs_ml
    E["ri_ct"] = [ri.contrast(np.eye(4)[j]) for j in range(4)]
    E["rs_ct"] = [rs.contrast(np.eye(4)[j]) for j in range(4)]
    E["rs_d12"] = rs.contrast(np.array([0, 0, 0, 12.0]))
    lrt = ri.deviance - rs.deviance             # REML LRT for the random slope (same fixed effects)
    E["lrt_slope"] = dict(stat=lrt, df=2, p_naive=st.chi2.sf(lrt, 2),
                          p_mix=0.5 * st.chi2.sf(lrt, 1) + 0.5 * st.chi2.sf(lrt, 2))
    # BLUPs
    E["blup_rs"] = np.array([rs.blup(i) if (~np.isnan(y[i])).any() else [np.nan, np.nan] for i in range(N)])
    E["blup_ri"] = np.array([ri.blup(i)[0] for i in range(N)])
    # ---- covariance structures with saturated mean (visit x group), all 4 visits as outcome
    Xs = np.zeros((N, 4, 8))
    for j in range(4):
        Xs[:, j, j] = 1
        Xs[:, j, 4 + j] = g
    cs_fits = {}
    for kind in ("cs", "ar1", "rirs", "un"):
        cs_fits[kind] = LMM(y, Xs, t, kind).fit()
    E["covfits"] = cs_fits
    # ---- MMRM primary analysis (change from baseline, baseline covariate, UN, REML, Satterthwaite)
    M = mmrm(y, g)
    E["mmrm"] = M
    lsm = {}
    for j in range(3):
        for k in (0, 1):
            Lv = np.zeros(9)
            Lv[j] = 1
            Lv[3 + j] = k
            lsm[(k, j)] = M.contrast(Lv)
        Ld = np.zeros(9)
        Ld[3 + j] = 1
        lsm[("d", j)] = M.contrast(Ld)
    E["lsm"] = lsm
    E["mmrm_n"] = int(M.anal.sum())
    E["mmrm_nobs"] = int(M.n)
    # ---- missing-data comparison at month 12
    comp = {}
    comp["full"] = ancova12(D, full[:, 3], np.ones(N, bool))
    comp["mmrm"] = dict(est=lsm[("d", 2)]["est"], se=lsm[("d", 2)]["se"], lo=lsm[("d", 2)]["lo"],
                        hi=lsm[("d", 2)]["hi"], n=E["mmrm_n"], p=lsm[("d", 2)]["p"])
    comp["cc12"] = ancova12(D, y[:, 3], obs[:, 3])
    comp["ccall"] = ancova12(D, y[:, 3], cc)
    yl = locf(y)
    comp["locf"] = ancova12(D, yl[:, 3], obs[:, 1:].any(1))
    E["comp"] = comp
    # baseline HbA1c by availability of the 12-month value (evidence against MCAR)
    has12 = obs[:, 3]
    mt = {}
    for k in (1, 0):
        a = full[(g == k) & has12, 0]
        b_ = full[(g == k) & ~has12, 0]
        mt[k] = dict(n_obs=len(a), m_obs=a.mean(), sd_obs=a.std(ddof=1), n_mis=len(b_), m_mis=b_.mean(), sd_mis=b_.std(ddof=1),
                     chg_obs=(y[(g == k) & has12, 3] - y[(g == k) & has12, 0]).mean())
    Xm = np.column_stack([np.ones(N), g, full[:, 0] - full[:, 0].mean()]).astype(float)
    bm, covm = glm_logit((~has12).astype(float), Xm)
    sem = np.sqrt(np.diag(covm))
    mt["logit"] = dict(or_base=np.exp(bm[2]), lo=np.exp(bm[2] - Z * sem[2]), hi=np.exp(bm[2] + Z * sem[2]),
                       p=2 * st.norm.sf(abs(bm[2] / sem[2])), or_g=np.exp(bm[1]), lo_g=np.exp(bm[1] - Z * sem[1]),
                       hi_g=np.exp(bm[1] + Z * sem[1]), p_g=2 * st.norm.sf(abs(bm[1] / sem[1])))
    E["miss_tab"] = mt
    # full-data MMRM (no dropout) should equal full-data ANCOVA at 12 months
    E["mmrm_fulldata"] = mmrm(full, g)
    # ---- adherence: GEE / GLMM
    A = long_adh(D)
    E["adh_long"] = A
    prop = {}
    for k in (0, 1):
        a = D["adh"][g == k]
        prop[k] = dict(n=(~np.isnan(a)).sum(0), x=np.nansum(a, 0).astype(int), p=np.nanmean(a, 0))
    E["adh_prop"] = prop
    Xa = np.column_stack([np.ones(len(A)), A[:, 1], A[:, 2] == 1, A[:, 2] == 2]).astype(float)
    ge = {}
    for cs in ("independence", "exchangeable", "ar1", "unstructured"):
        ge[cs] = gee(A[:, 3], Xa, A[:, 0], "binomial", cs, pos=A[:, 2], k=3)
    E["gee"] = ge
    bn, covn = glm_logit(A[:, 3], Xa)
    E["naive_logit"] = dict(beta=bn, se=np.sqrt(np.diag(covn)))
    # interaction model: group x visit (visit-specific ORs), exchangeable
    Xi = np.column_stack([Xa, A[:, 1] * (A[:, 2] == 1), A[:, 1] * (A[:, 2] == 2)])
    gi = gee(A[:, 3], Xi, A[:, 0], "binomial", "exchangeable", pos=A[:, 2], k=3)
    E["gee_int"] = gi
    Lmat = [np.array([0, 1, 0, 0, 0, 0.]), np.array([0, 1, 0, 0, 1, 0.]), np.array([0, 1, 0, 0, 0, 1.])]
    E["or_visit"] = []
    for Lv in Lmat:
        est = Lv @ gi["beta"]
        se = np.sqrt(Lv @ gi["robust"] @ Lv)
        E["or_visit"].append(dict(est=est, se=se, OR=np.exp(est), lo=np.exp(est - Z * se), hi=np.exp(est + Z * se),
                                  p=2 * st.norm.sf(abs(est / se))))
    bint = gi["beta"][4:]
    Vint = gi["robust"][4:, 4:]
    W = float(bint @ np.linalg.solve(Vint, bint))
    E["int_wald"] = dict(chi2=W, df=2, p=st.chi2.sf(W, 2))
    # crude visit-specific ORs (for checking the GEE interaction model: saturated -> equals crude)
    E["or_crude"] = [(prop[1]["p"][j] / (1 - prop[1]["p"][j])) / (prop[0]["p"][j] / (1 - prop[0]["p"][j])) for j in range(3)]
    # GLMM random-intercept logistic (conditional)
    gm = glmm_logit_ri(A[:, 3], Xa, A[:, 0], Q=40)
    E["glmm"] = gm
    # population-averaged OR implied by the GLMM (average over u ~ N(0, sigma^2)) at each visit
    impl = []
    for j in range(3):
        eta0 = gm["beta"][0] + [0, gm["beta"][2], gm["beta"][3]][j]
        p0 = marginal_from_conditional(np.array([eta0]), gm["sigma"])[0]
        p1 = marginal_from_conditional(np.array([eta0 + gm["beta"][1]]), gm["sigma"])[0]
        impl.append((p0, p1, (p1 / (1 - p1)) / (p0 / (1 - p0))))
    E["glmm_implied"] = impl
    E["atten"] = dict(c2=(16 * np.sqrt(3) / (15 * np.pi)) ** 2,
                      approx_logor=gm["beta"][1] / np.sqrt(1 + (16 * np.sqrt(3) / (15 * np.pi)) ** 2 * gm["sigma"] ** 2))
    # full adherence data (no dropout) GEE for reference
    Af = long_adh(D, D["adh_full"])
    Xf = np.column_stack([np.ones(len(Af)), Af[:, 1], Af[:, 2] == 1, Af[:, 2] == 2]).astype(float)
    E["gee_fulladh"] = gee(Af[:, 3], Xf, Af[:, 0], "binomial", "exchangeable", pos=Af[:, 2], k=3)
    if rep:
        E["rep"] = replicate()
    return E


def replicate(nrep=500, start=5000):
    """Repeat the whole trial nrep times (same design, new random data): bias and SD of each estimator."""
    out = []
    for s in range(start, start + nrep):
        D = simulate(s)
        g, y, full = D["g"], D["y"], D["full"]
        N = D["N"]
        f = ancova12(D, full[:, 3], np.ones(N, bool))["est"]
        M = mmrm(y, g)
        c = ancova12(D, y[:, 3], D["obs"][:, 3])["est"]
        lo = ancova12(D, locf(y)[:, 3], D["obs"][:, 1:].any(1))["est"]
        out.append((f, M.beta[5], c, lo))
    out = np.array(out)
    return dict(n=nrep, mean=out.mean(0), sd=out.std(0, ddof=1), bias=out.mean(0) - TRUE_DIFF12,
                mcse=out.std(0, ddof=1) / np.sqrt(nrep))


# ------------------------------------------------------------------ marginal vs conditional toy (라 절)
def or_toy(logits=(-2.0, 0.0, 2.0), cond_or=3.0):
    p0 = expit(np.array(logits))
    p1 = expit(np.array(logits) + np.log(cond_or))
    P0, P1 = p0.mean(), p1.mean()
    return dict(p0=p0, p1=p1, P0=P0, P1=P1, OR=(P1 / (1 - P1)) / (P0 / (1 - P0)),
                ind_or=(p1 / (1 - p1)) / (p0 / (1 - p0)))


# ------------------------------------------------------------------ design effect example
def deff(m=12.5, icc=0.05, n_clusters=24):
    de = 1 + (m - 1) * icc
    n = m * n_clusters
    return dict(de=de, n=n, ess=n / de)


# ------------------------------------------------------------------ printing
def main(rep=False):
    np.set_printoptions(suppress=True, linewidth=150)
    T = toy()
    R = T["R"]
    print("=== 가. toy RM-ANOVA (4 patients x 3 visits)")
    print(TOY)
    print("visit means", T["vis_mean"], "patient means", T["pat_mean"], "grand", T["gm"])
    print("SS total %.4f subj %.4f time %.4f error %.4f" % (T["ss_tot"], R["SS"]["subj"], R["SS"]["time"], R["SS"]["ew"]))
    print("RM: F(%d,%d) = %.3f  p = %.5f   MS time %.4f MS err %.5f" % (R["df"]["time"], R["df"]["ew"], R["F"]["time"], R["P"]["time"], R["MS"]["time"], R["MS"]["ew"]))
    print("one-way ignoring patients: SSw %.4f df %d MSw %.5f F(2,%d) = %.4f p = %.4f" % (T["ss_w"], T["df_w"], T["ms_w"], T["df_w"], T["F1"], T["p1"]))

    E = example(rep=rep)
    D = E["D"]
    g = D["g"]
    print("\n=== example seed", SEED, "N", D["N"], "true diff12", TRUE_DIFF12)
    for k, lab in ((1, "intervention"), (0, "usual care")):
        d = E["desc"][k]
        print(lab, "n", d["n"], "mean", d["mean"].round(3), "sd", d["sd"].round(3))
        print("   95% CI lo", d["lo"].round(3), "hi", d["hi"].round(3))
        print("   full-data means (unobservable)", E["full_mean"][k].round(3))
    print("observations", E["n_obs"], "patterns", E["patterns"])
    print("complete cases", E["n_cc"], "dropped (before 3/6/12)", E["dropped"])
    P = E["pooled"]
    print("pooled all-observation group difference %.3f  naive SE %.4f  patient-robust SE %.4f  ratio %.2f  n obs %d" %
          (P["est"], P["se_naive"], P["se_rob"], P["se_rob"] / P["se_naive"], P["n"]))

    R = E["rm"]
    print("\n=== RM-ANOVA complete cases N =", R["N"], "per group", R["nl"])
    print("cell means (rows: g=0 UC, g=1 intervention)\n", R["cell"].round(3))
    for key in ("group", "eb", "time", "tg", "ew"):
        print("  %-5s SS %.4f df %d MS %.4f" % (key, R["SS"][key], R["df"][key], R["MS"][key]))
    for key in ("group", "time", "time3", "tg"):
        print("  F %-5s %.3f p %.3g partial eta2 %.3f" % (key, R["F"][key], R["P"][key], R["peta"][key]))
    print("  Type III SS time %.4f MS %.4f" % (R["SS"]["time3"], R["MS"]["time3"]))
    print("  Mauchly W %.4f chi2 %.3f df %d p %.6f" % (R["W"], R["chi2"], R["mdf"], R["mp"]))
    print("  eps GG %.4f HF(orig) %.4f HF(Lecoutre) %.4f LB %.4f  Box check %.4f" % (R["gg"], R["hf"], R["hf_lec"], R["lb"], gg_box(R["S"])))
    for c in ("none", "gg", "hf", "hf_lec", "lb"):
        d = R["corr"][c]
        print("   %-6s eps %.4f  time df1 %.3f df2 %.3f p %.3g (III p %.3g) | tg df1 %.3f df2 %.3f p %.4f | MS: time3 %.4f tg %.4f err %.4f" %
              (c, d["eps"], d["time"][0], d["time"][1], d["time"][2], d["time3"][2], d["tg"][0], d["tg"][1], d["tg"][2],
               R["SS"]["time3"] / d["time"][0], R["SS"]["tg"] / d["tg"][0], R["SS"]["ew"] / d["tg"][1]))
    print("  within-group covariance S\n", R["S"].round(4))
    sd = np.sqrt(np.diag(R["S"]))
    print("  SD", sd.round(3), "corr\n", (R["S"] / np.outer(sd, sd)).round(3))
    print("  variance of differences:", [(a, b, round(v, 4)) for a, b, v in E["vardiff"]])
    print("  completer change at 12m UC %.3f INT %.3f diff %.3f" % (E["cc_change"][0], E["cc_change"][1], E["cc_change"][1] - E["cc_change"][0]))
    # MS check: MS error within = mean of variance of orthonormal contrasts
    print("  check: MS_ew = trace(Sc)/p ->", R["MS"]["ew"], np.trace(R["Sc"]) / 3)

    print("\n=== data structure: first rows (wide)")
    for i in range(12):
        print(i + 1, g[i], D["y"][i])

    print("\n=== LMM random intercept (REML): beta", E["ri"].beta.round(4), "se", E["ri"].se.round(4))
    d = E["ri"].desc
    print("  var_b0 %.4f (SD %.3f) var_e %.4f (SD %.3f) ICC %.3f  REML dev %.3f AIC %.3f" % (d["var_b0"], np.sqrt(d["var_b0"]), d["var_e"], np.sqrt(d["var_e"]), d["icc"], E["ri"].deviance, E["ri"].aic))
    for j, c in enumerate(E["ri_ct"]):
        print("   coef %d est %.4f se %.4f df %.1f t %.2f p %.4g" % (j, c["est"], c["se"], c["df"], c["t"], c["p"]))
    print("=== LMM random intercept + slope (REML): beta", E["rs"].beta.round(5), "se", E["rs"].se.round(5))
    d = E["rs"].desc
    print("  var_b0 %.4f (SD %.4f) var_b1 %.6f (SD %.4f) corr %.3f var_e %.4f (SD %.4f)  REML dev %.3f AIC %.3f" %
          (d["var_b0"], np.sqrt(d["var_b0"]), d["var_b1"], np.sqrt(d["var_b1"]), d["corr"], d["var_e"], np.sqrt(d["var_e"]), E["rs"].deviance, E["rs"].aic))
    for j, c in enumerate(E["rs_ct"]):
        print("   coef %d est %.5f se %.5f df %.1f t %.2f p %.4g" % (j, c["est"], c["se"], c["df"], c["t"], c["p"]))
    c = E["rs_d12"]
    print("   12 x group:month  est %.3f se %.3f 95%% CI %.3f to %.3f p %.4g df %.1f" % (c["est"], c["se"], c["lo"], c["hi"], c["p"], c["df"]))
    print("   LRT random slope (REML) stat %.2f p(chi2_2) %.3g p(mixture) %.3g" % (E["lrt_slope"]["stat"], E["lrt_slope"]["p_naive"], E["lrt_slope"]["p_mix"]))
    print("   ML fits: RI dev %.3f AIC %.3f | RS dev %.3f AIC %.3f" % (E["ri_ml"].deviance, E["ri_ml"].aic, E["rs_ml"].deviance, E["rs_ml"].aic))
    print("   implied SD by month (RS):", np.sqrt(np.diag(E["rs"].Sigma)).round(3))
    print("   n obs", E["rs"].n, "subjects", E["rs"].nsub)

    print("\n=== covariance structures (saturated visit x group means, REML)")
    for kind, m in E["covfits"].items():
        Sg = m.Sigma
        sd = np.sqrt(np.diag(Sg))
        Rm = Sg / np.outer(sd, sd)
        print(" %-5s ntheta %d  -2REML %.3f AIC %.3f  desc %s" % (kind, m.ntheta, m.deviance, m.aic,
              {k_: (np.round(v, 4) if np.ndim(v) == 0 else None) for k_, v in m.desc.items()}))
        print("     SD", sd.round(3), "corr pairs 0-3,3-6,0-6,6-12,3-12,0-12:",
              [round(Rm[a, b], 3) for a, b in [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (0, 3)]])
        print("     group diff at 12m (beta[7]-beta[4]) =", round(m.beta[7] - m.beta[4], 4))

    M = E["mmrm"]
    print("\n=== MMRM (UN, REML, Satterthwaite) patients", E["mmrm_n"], "obs", E["mmrm_nobs"], "baseline mean %.3f" % M.base_mean)
    print("  beta", M.beta.round(4))
    print("  Sigma\n", M.Sigma.round(4))
    for j, mo in enumerate((3, 6, 12)):
        a = E["lsm"][(1, j)]
        b = E["lsm"][(0, j)]
        d = E["lsm"][("d", j)]
        print("  month %2d  INT %.3f (SE %.3f, %.2f to %.2f)  UC %.3f (SE %.3f, %.2f to %.2f)  diff %.3f (SE %.3f) %.3f to %.3f p %.4f df %.1f" %
              (mo, a["est"], a["se"], a["lo"], a["hi"], b["est"], b["se"], b["lo"], b["hi"], d["est"], d["se"], d["lo"], d["hi"], d["p"], d["df"]))

    print("\n=== month-12 difference by method")
    for k_, v in E["comp"].items():
        print("  %-6s est %.3f se %.3f CI %.3f to %.3f n %d p %.4f" % (k_, v["est"], v["se"], v["lo"], v["hi"], v["n"], v["p"]))
    for k in (1, 0):
        t_ = E["miss_tab"][k]
        print("  g %d: 12m available n %d baseline %.3f (SD %.3f) mean change %.3f | missing n %d baseline %.3f (SD %.3f)" %
              (k, t_["n_obs"], t_["m_obs"], t_["sd_obs"], t_["chg_obs"], t_["n_mis"], t_["m_mis"], t_["sd_mis"]))
    lg = E["miss_tab"]["logit"]
    print("  logistic for missing 12m: OR per 1%%p baseline %.2f (%.2f-%.2f) p %.5f; OR intervention %.2f (%.2f-%.2f) p %.4f" %
          (lg["or_base"], lg["lo"], lg["hi"], lg["p"], lg["or_g"], lg["lo_g"], lg["hi_g"], lg["p_g"]))
    mf = E["mmrm_fulldata"]
    print("  check: MMRM on full data diff12 %.4f vs full ANCOVA %.4f" % (mf.beta[5], E["comp"]["full"]["est"]))

    print("\n=== adherence")
    for k in (1, 0):
        p_ = E["adh_prop"][k]
        print(" g", k, "n", p_["n"], "x", p_["x"], "prop", p_["p"].round(3))
    nb = E["naive_logit"]
    print(" naive logistic: OR %.3f se %.4f CI %.2f-%.2f" % (np.exp(nb["beta"][1]), nb["se"][1], np.exp(nb["beta"][1] - Z * nb["se"][1]), np.exp(nb["beta"][1] + Z * nb["se"][1])))
    print(" naive logistic visit12: est %.4f se %.4f" % (nb["beta"][3], nb["se"][3]))
    for cs, G in E["gee"].items():
        b1 = G["beta"][1]
        se = G["se_robust"][1]
        print(" GEE %-13s beta %s" % (cs, G["beta"].round(4)))
        print("     OR %.3f robust se %.4f (CI %.2f-%.2f, p %.4f) naive se %.4f alpha %s phi %.3f QIC %.2f | visit12 rob se %.4f naive %.4f" %
              (np.exp(b1), se, np.exp(b1 - Z * se), np.exp(b1 + Z * se), 2 * st.norm.sf(abs(b1 / se)), G["se_naive"][1],
               np.round(G["alpha"], 4) if G["R"] is None else G["R"].round(3).tolist(), G["phi"], G["qic"], G["se_robust"][3], G["se_naive"][3]))
    ge = E["gee"]["exchangeable"]
    print(" exch full output: se_rob", ge["se_robust"].round(4), "wald", ((ge["beta"] / ge["se_robust"]) ** 2).round(3),
          "p", (2 * st.norm.sf(np.abs(ge["beta"] / ge["se_robust"]))).round(5), "nclus", ge["nclus"], "max", ge["maxsize"], "nobs", len(E["adh_long"]))
    print(" interaction model beta", E["gee_int"]["beta"].round(4), "alpha", round(E["gee_int"]["alpha"], 4))
    for j, o in enumerate(E["or_visit"]):
        print("   visit %d OR %.3f (%.2f-%.2f) p %.4f   crude OR %.3f" % (j, o["OR"], o["lo"], o["hi"], o["p"], E["or_crude"][j]))
    print("   interaction Wald chi2 %.3f df 2 p %.3f" % (E["int_wald"]["chi2"], E["int_wald"]["p"]))
    gm = E["glmm"]
    print(" GLMM beta", gm["beta"].round(4), "se", gm["se"].round(4), "sigma %.3f" % gm["sigma"])
    print("   conditional OR %.3f CI %.2f-%.2f" % (np.exp(gm["beta"][1]), np.exp(gm["beta"][1] - Z * gm["se"][1]), np.exp(gm["beta"][1] + Z * gm["se"][1])))
    print("   latent ICC sigma^2/(sigma^2 + pi^2/3) = %.3f" % (gm["sigma"] ** 2 / (gm["sigma"] ** 2 + np.pi ** 2 / 3)))
    for j, (p0, p1, orr) in enumerate(E["glmm_implied"]):
        print("   implied marginal visit %d: P0 %.3f P1 %.3f OR %.3f" % (j, p0, p1, orr))
    at = E["atten"]
    print("   attenuation c^2 %.4f approx marginal logOR %.4f OR %.3f" % (at["c2"], at["approx_logor"], np.exp(at["approx_logor"])))
    gf = E["gee_fulladh"]
    print(" GEE exch on full adherence (no dropout) OR %.3f" % np.exp(gf["beta"][1]))

    print("\n=== marginal vs conditional toy")
    o = or_toy()
    print(" p0", o["p0"].round(4), "p1", o["p1"].round(4), "P0 %.4f P1 %.4f marginal OR %.3f" % (o["P0"], o["P1"], o["OR"]), "individual ORs", o["ind_or"].round(3))
    print("\n=== design effect", deff())
    rsd = E["ri"].desc
    m_ = E["n_obs"] / D["N"]
    print(" example: m %.2f ICC %.3f DE %.2f sqrt %.2f" % (m_, rsd["icc"], 1 + (m_ - 1) * rsd["icc"], np.sqrt(1 + (m_ - 1) * rsd["icc"])))
    if rep:
        r = E["rep"]
        print("\n=== replication (n = %d): mean, SD, bias (vs %.2f), MC SE  [full, MMRM, CC12, LOCF]" % (r["n"], TRUE_DIFF12))
        print(" mean", r["mean"].round(3), "sd", r["sd"].round(3), "bias", r["bias"].round(3), "mcse", r["mcse"].round(4))
    return E


if __name__ == "__main__":
    main(rep="--rep" in sys.argv)


# ------------------------------------------------------------------ values behind the Python output boxes in content/ch10.html
# run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch10.py --out   (real statsmodels 0.15 output via sm_outputs)
def outputs(E=None):
    """Numbers for the scipy / statsmodels boxes; sm_outputs() prints the real statsmodels output and checks it."""
    E = example() if E is None else E
    R = E["rm"]
    c = R["corr"]
    print("\n--- RM-ANOVA table (journal style)")
    print("Mauchly W %.3f chi2 %.2f df %d p %.6f GG %.3f HF %.3f HF(Lecoutre) %.3f LB %.3f" % (R["W"], R["chi2"], R["mdf"], R["mp"], R["gg"], R["hf"], R["hf_lec"], R["lb"]))
    print("group SS %.3f df 1 MS %.3f F %.2f p %.4f | error(between) SS %.3f df %d MS %.3f" % (R["SS"]["group"], R["MS"]["group"], R["F"]["group"], R["P"]["group"], R["SS"]["eb"], R["df"]["eb"], R["MS"]["eb"]))
    for lab, key in (("SA", "none"), ("GG", "gg"), ("HF", "hf"), ("LB", "lb")):
        d = c[key]
        print("%s time(III) SS %.3f df %.2f MS %.3f F %.2f p %.3g | tg SS %.3f df %.2f MS %.3f F %.2f p %.4f | err df %.2f MS %.3f" %
              (lab, R["SS"]["time3"], d["time"][0], R["SS"]["time3"] / d["time"][0], R["F"]["time3"], d["time3"][2],
               R["SS"]["tg"], d["tg"][0], R["SS"]["tg"] / d["tg"][0], R["F"]["tg"], d["tg"][2], d["tg"][1], R["SS"]["ew"] / d["tg"][1]))
    print("partial eta2: time(III) %.3f tg %.3f group %.3f" % (R["peta"]["time3"], R["peta"]["tg"], R["peta"]["group"]))
    print("\n--- scipy f.sf with the rounded values shown in the text")
    F_, eps = 3.946, 0.809
    for a, b in ((3, 219), (3 * eps, 219 * eps), (1, 73)):
        print("f.sf(%.3f, %.3f, %.3f) = %.6f" % (F_, a, b, st.f.sf(F_, a, b)))
    rs = E["rs"]
    d = rs.desc
    G = d["G"]
    print("\n--- in-house random intercept + slope (REML), SE from (X'V^-1X)^-1")
    print("Scale (residual var) %.4f | Group Var %.4f  Group x month Cov %.4f  month Var %.4f  REML llf %.4f" % (d["var_e"], G[0, 0], G[0, 1], G[1, 1], -rs.deviance / 2))
    for nm, j in zip(("Intercept", "group", "month", "group:month"), range(4)):
        b, se = rs.beta[j], rs.se[j]
        z = b / se
        print("%-12s %.6f se %.6f z %.4f p %.6f" % (nm, b, se, z, 2 * st.norm.sf(abs(z))))
    b3, se3 = rs.beta[3], rs.se[3]
    print("12 x group:month z-based: %.4f (%.4f to %.4f)" % (12 * b3, 12 * (b3 - Z * se3), 12 * (b3 + Z * se3)))
    ri = E["ri"]
    print("RI: Group Var %.4f Scale %.4f icc %.4f  gxm %.5f se %.5f" % (ri.desc["var_b0"], ri.desc["var_e"], ri.desc["icc"], ri.beta[3], ri.se[3]))
    ge = E["gee"]["exchangeable"]
    sizes = np.bincount(E["adh_long"][:, 0].astype(int))
    sizes = sizes[sizes > 0]
    print("GEE exchangeable: alpha %.4f  nclus %d  min %d max %d mean %.1f nobs %d  QIC %.2f" % (ge["alpha"], ge["nclus"], sizes.min(), sizes.max(), sizes.mean(), len(E["adh_long"]), ge["qic"]))
    print("OR table:", [(round(np.exp(ge["beta"][j]), 2), round(np.exp(ge["beta"][j] - Z * ge["se_robust"][j]), 2), round(np.exp(ge["beta"][j] + Z * ge["se_robust"][j]), 2)) for j in range(4)])
    sm_outputs(E)
    pr = E["adh_prop"]
    for j in range(3):
        p1, p0 = pr[1]["p"][j], pr[0]["p"][j]
        print("visit %d: INT %.4f UC %.4f RR %.3f RD %.4f OR %.3f" % (j, p1, p0, p1 / p0, p1 - p0, (p1 / (1 - p1)) / (p0 / (1 - p0))))
    print("\n--- MMRM table")
    for j, mo in enumerate((3, 6, 12)):
        for k in (1, 0):
            a = E["lsm"][(k, j)]
            print("month %d g %d LSM %.4f SE %.4f CI %.4f %.4f n %d" % (mo, k, a["est"], a["se"], a["lo"], a["hi"], E["desc"][k]["n"][j + 1]))
        dd = E["lsm"][("d", j)]
        print("   diff %.4f SE %.4f CI %.4f %.4f p %.5f df %.2f" % (dd["est"], dd["se"], dd["lo"], dd["hi"], dd["p"], dd["df"]))
    M = E["mmrm"]
    sd = np.sqrt(np.diag(M.Sigma))
    print("MMRM Sigma SD", sd.round(4), "corr", (M.Sigma / np.outer(sd, sd)).round(3).tolist())
    print("MMRM baseline coefs", M.beta[6:].round(4))
    for kind in ("cs", "ar1", "rirs"):
        m2 = mmrm(E["D"]["y"], E["D"]["g"], kind=kind)
        print("MMRM with %s: diff12 %.4f se %.4f AIC %.2f" % (kind, m2.beta[5], m2.se[5], m2.aic))
    print("MMRM UN AIC %.2f" % M.aic)
    return E


def sm_outputs(E):
    """Real statsmodels 0.15 output for the MixedLM and GEE boxes (source /home/claude/pylibs/env.sh first).
    Checks that statsmodels reproduces the in-house estimates. Differences that remain:
      * MixedLM: statsmodels' fixed-effect SEs come from inverting the Hessian of ALL parameters jointly
        (fixed effects + variance components, including the cross-derivatives); the chapter (like
        SAS PROC MIXED) uses (X'V^-1X)^-1 with the variance components held at their estimates.
      * MixedLM: the default optimizer sequence (bfgs, lbfgs, cg) does not converge on these data;
        method="cg" reaches the REML optimum (same llf as the in-house fit).
      * GEE: identical beta, alpha and robust SEs; model-based SEs use scale 1 (binomial) in both."""
    try:
        import warnings
        import pandas as pd
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n(statsmodels not available: skip the real-library check)")
        return
    D = E["D"]
    L = long_hba1c(D)
    data = pd.DataFrame({"id": L[:, 0].astype(int), "group": L[:, 1].astype(int), "month": L[:, 2], "hba1c": L[:, 4]})
    rs = E["rs"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = smf.mixedlm("hba1c ~ group * month", data, groups=data["id"], re_formula="~month").fit(reml=True, method="cg")
    print("\n--- statsmodels MixedLM (REML, method='cg'); warnings:", [str(x.message) for x in w])
    print(m.summary())
    assert m.converged and abs(m.llf + rs.deviance / 2) < 1e-4
    assert np.allclose(m.fe_params.values, rs.beta, atol=1e-5) and np.allclose(m.cov_re.values, rs.desc["G"], atol=1e-4)
    b, se = m.fe_params["group:month"], m.bse_fe["group:month"]
    print("group:month %.6f se %.6f (in-house %.6f) z %.4f p %.6f | 12x CI %.4f (%.4f to %.4f)" %
          (b, se, rs.se[3], b / se, m.pvalues["group:month"], 12 * b, 12 * (b - Z * se), 12 * (b + Z * se)))
    print("month se %.6f (in-house %.6f) | cov_re" % (m.bse_fe["month"], rs.se[2]), m.cov_re.values.round(6).tolist(), "scale %.6f" % m.scale)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m0 = smf.mixedlm("hba1c ~ group * month", data, groups=data["id"], re_formula="~month").fit(reml=True)
    print("default optimizer: converged", m0.converged, "llf %.4f" % m0.llf)
    ri = E["ri"]
    mri = smf.mixedlm("hba1c ~ group * month", data, groups=data["id"]).fit(reml=True)
    print("random intercept: llf %.4f (in-house %.4f) group:month se %.6f (in-house %.6f)" % (mri.llf, -ri.deviance / 2, mri.bse_fe["group:month"], ri.se[3]))
    # GEE
    A = E["adh_long"]
    adh = pd.DataFrame({"id": A[:, 0].astype(int), "group": A[:, 1].astype(int),
                        "visit": np.array([3, 6, 12])[A[:, 2].astype(int)], "adh": A[:, 3].astype(int)})
    res = smf.gee("adh ~ group + C(visit)", groups="id", data=adh, family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    print("\n--- statsmodels GEE (exchangeable)")
    print(res.summary())
    print(res.model.cov_struct.summary())
    ge = E["gee"]["exchangeable"]
    order = [0, 2, 3, 1]      # statsmodels: Intercept, C(visit)[T.6], C(visit)[T.12], group
    assert np.allclose(res.params.values, ge["beta"][order], atol=1e-6)
    assert np.allclose(res.bse.values, ge["se_robust"][order], atol=1e-6)
    assert np.allclose(res.standard_errors(cov_type="naive"), ge["se_naive"][order], atol=1e-6)
    assert abs(res.model.cov_struct.dep_params - ge["alpha"]) < 1e-6
    print("naive SE (scale 1):", np.round(res.standard_errors(cov_type="naive"), 4), "| Pearson phi %.4f" % ge["phi"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("statsmodels qic(scale=1) (QIC, QICu):", np.round(res.qic(scale=1.0), 2),
              "| in-house Pan QIC %.2f (statsmodels' penalty uses sum D'D without the variance weights)" % ge["qic"])
    gi = E["gee"]["independence"]
    print("independence: robust se group %.4f naive se group %.4f" % (gi["se_robust"][1], gi["se_naive"][1]))


if __name__ == "__main__" and "--out" in sys.argv:
    outputs()


# ------------------------------------------------------------------ REVIEW2 (2026-09-30): numbers introduced by the beginner rewrite
# (가 절 correlations quoted in the text, GG epsilon via eigenvalues in '수식으로 보기', 라 절 'average patient' probabilities)
if __name__ == "__main__":
    from lib_ch10 import orth_contrasts
    E2 = example()
    R2 = E2["rm"]
    S2 = R2["S"]
    sd2 = np.sqrt(np.diag(S2))
    r2 = S2 / np.outer(sd2, sd2)
    print("\n=== REVIEW2 additions")
    print(" pooled within-group corr (complete cases): baseline-3m %.4f baseline-12m %.4f" % (r2[0, 1], r2[0, 3]))
    k2 = S2.shape[0]
    C2 = orth_contrasts(k2).T                     # (k-1) x k orthonormal contrasts
    lam2 = np.sort(np.linalg.eigvalsh(C2 @ S2 @ C2.T))
    print(" GG eigenvalues", lam2.round(4), "sum %.4f sum sq %.5f eps %.4f (rm_anova %.4f)" %
          (lam2.sum(), (lam2 ** 2).sum(), lam2.sum() ** 2 / ((k2 - 1) * (lam2 ** 2).sum()), R2["gg"]))
    gm2 = E2["glmm"]["beta"]
    p0c, p1c = expit(gm2[0]), expit(gm2[0] + gm2[1])
    print(" GLMM 'average patient' (u = 0), months 0-3: usual care %.3f intervention %.3f OR %.3f" %
          (p0c, p1c, (p1c / (1 - p1c)) / (p0c / (1 - p0c))))
