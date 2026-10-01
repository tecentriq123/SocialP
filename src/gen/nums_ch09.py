"""Every number quoted in content/ch09.html is computed here.
run: python3 gen/nums_ch09.py        (prints a report; takes ~1 min because of bootstraps)
fig_ch09.py imports compute() from this file.
Python outputs in the HTML (statsmodels summaries, separation example) come from gen/pyout_ch09.py,
which re-fits the same data with statsmodels and checks them against compute().
"""
import sys, os, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.stats as st
from scipy import integrate
from lib_ch09 import (fit_logit, glm_R, profile_ci, lr_test, score_global, firth, firth_profile_ci, firth_lr,
                      fit_poisson, fit_logbin, auc_delong, roc_points, confusion, hosmer_lemeshow, cal_int_slope,
                      rcs_knots, rcs_basis, brier, vanco, cohort, external, design, irae, expit, logit, or_ci, Z,
                      loglik, _cohort_draw)


def lincomb(fit, c):
    c = np.asarray(c, float)
    b = float(c @ fit["beta"])
    se = float(np.sqrt(c @ fit["cov"] @ c))
    return b, se


@functools.lru_cache(maxsize=None)
def compute():
    R = {}
    # ================================================================ A. vancomycin
    t, a = vanco()
    n = len(t)
    X = np.column_stack([np.ones(n), t])
    fv = fit_logit(X, a)
    f0 = fit_logit(np.ones((n, 1)), a)
    lpm = np.linalg.lstsq(X, a, rcond=None)[0]
    pl = X @ lpm
    b0, b1 = fv["beta"]
    se1 = fv["se"][1]
    grid = [10, 15, 20, 25, 30]
    pv = {x: float(expit(b0 + b1 * x)) for x in grid}
    p18 = float(expit(b0 + b1 * 18))
    lr, lrp = lr_test(fv, f0, 1)
    pci = profile_ci(X, a, 1, fv)
    gv = glm_R(X, a)
    R["vanco"] = dict(n=n, events=int(a.sum()), rate=a.mean(), tmin=t.min(), tmax=t.max(), tmed=float(np.median(t)),
                      q1=float(np.quantile(t, .25)), q3=float(np.quantile(t, .75)),
                      lpm=lpm, lpm_neg=int((pl < 0).sum()), lpm_gt1=int((pl > 1).sum()),
                      lpm_x0=-lpm[0] / lpm[1], lpm_x1=(1 - lpm[0]) / lpm[1], lpm_at_max=float(lpm[0] + lpm[1] * t.max()),
                      lpm_at_min=float(lpm[0] + lpm[1] * t.min()),
                      b0=b0, b1=b1, se0=fv["se"][0], se1=se1, z1=b1 / se1, p1=fv["pval"][1],
                      or1=np.exp(b1), or1_ci=(np.exp(b1 - Z * se1), np.exp(b1 + Z * se1)),
                      or5=np.exp(5 * b1), or5_ci=(np.exp(5 * (b1 - Z * se1)), np.exp(5 * (b1 + Z * se1))),
                      p=pv, odds={x: pv[x] / (1 - pv[x]) for x in grid},
                      ll=fv["ll"], ll0=f0["ll"], dev=fv["dev"], dev0=f0["dev"], lr=lr, lrp=lrp,
                      prof_ci=pci, prof_or5=(np.exp(5 * pci[0]), np.exp(5 * pci[1])),
                      wald_ci=(b1 - Z * se1, b1 + Z * se1), glm=gv, x50=-b0 / b1, t=t, a=a, p18=p18)

    # ================================================================ B. 2x2 of chapter 5
    y = np.r_[np.ones(48), np.zeros(352), np.ones(102), np.zeros(498)]
    xA = np.r_[np.ones(400), np.zeros(600)]
    f22 = fit_logit(np.column_stack([np.ones(1000), xA]), y)
    woolf = np.sqrt(1 / 48 + 1 / 352 + 1 / 102 + 1 / 498)
    R["t22"] = dict(b0=f22["beta"][0], b1=f22["beta"][1], se1=f22["se"][1], woolf=woolf,
                    or_=np.exp(f22["beta"][1]), ci=(np.exp(f22["beta"][1] - Z * woolf), np.exp(f22["beta"][1] + Z * woolf)),
                    or_hand=(48 * 498) / (352 * 102), lnodds_B=np.log(102 / 498), lnodds_A=np.log(48 / 352),
                    pA=expit(f22["beta"][0] + f22["beta"][1]), pB=expit(f22["beta"][0]), z=f22["z"][1], p=f22["pval"][1])

    # ================================================================ C. cohort
    d = cohort()
    y = d["hosp"].astype(float)
    N = len(y)
    s = d["sglt2"] == 1
    desc = dict(n=N, n_s=int(s.sum()), n_d=int((~s).sum()), ev=int(y.sum()), ev_s=int(y[s].sum()), ev_d=int(y[~s].sum()))
    base = {}
    for k in ["age", "female", "prior", "hf", "egfr"]:
        base[k] = (float(d[k][s].mean()), float(d[k][~s].mean()), float(d[k][s].std(ddof=1)), float(d[k][~s].std(ddof=1)))
    for lvl in range(3):
        base[f"cci{lvl}"] = (float((d["cci"][s] == lvl).mean()), float((d["cci"][~s] == lvl).mean()))
    e1 = (d["egfr"] >= 45) & (d["egfr"] < 60); e2 = d["egfr"] < 45
    base["egfr4559"] = (float(e1[s].mean()), float(e1[~s].mean())); base["egfr45"] = (float(e2[s].mean()), float(e2[~s].mean()))
    desc["base"] = base
    R["desc"] = desc

    Xe, names = design(d, "etio")
    fe = fit_logit(Xe, y)
    fnull = fit_logit(np.ones((N, 1)), y)
    ge = glm_R(Xe, y)
    idx = {nm: i for i, nm in enumerate(names)}
    # crude (univariable) models
    crude = {}
    groups = {"sglt2": ["sglt2"], "age10": ["age10"], "female": ["female"], "cci": ["cci12", "cci3"], "prior": ["prior"],
              "hf": ["hf"], "egfr": ["egfr4559", "egfr45"]}
    for g, cols in groups.items():
        Xg = np.column_stack([np.ones(N)] + [Xe[:, idx[c]] for c in cols])
        fg = fit_logit(Xg, y)
        lr_g, p_g = lr_test(fg, fnull, len(cols))
        for i, c in enumerate(cols):
            crude[c] = dict(b=fg["beta"][i + 1], se=fg["se"][i + 1], p=fg["pval"][i + 1], orci=or_ci(fg["beta"][i + 1], fg["se"][i + 1]))
        crude[g + "_lr"] = (lr_g, p_g)
    adj = {c: dict(b=fe["beta"][i], se=fe["se"][i], z=fe["z"][i], p=fe["pval"][i], orci=or_ci(fe["beta"][i], fe["se"][i]))
           for c, i in idx.items()}
    # LR tests for multi-level variables in the adjusted model
    lrt = {}
    for g in ["cci", "egfr", "sglt2", "prior", "hf", "female", "age10"]:
        keep = [c for c in names if c not in groups[g]]
        fr = fit_logit(Xe[:, [idx[c] for c in keep]], y)
        lrt[g] = lr_test(fe, fr, len(groups[g]))
    # Wald chi-square (joint) for multi-level
    wald = {}
    for g in ["cci", "egfr"]:
        ii = [idx[c] for c in groups[g]]
        bb = fe["beta"][ii]; V = fe["cov"][np.ix_(ii, ii)]
        w = float(bb @ np.linalg.solve(V, bb)); wald[g] = (w, st.chi2.sf(w, len(ii)))
    # P for trend: CCI score 0,1,2 instead of dummies
    Xt = np.column_stack([Xe[:, [idx[c] for c in names if c not in ("cci12", "cci3")]], d["cci"].astype(float)])
    ft = fit_logit(Xt, y)
    trend = dict(b=ft["beta"][-1], se=ft["se"][-1], p=ft["pval"][-1], or_=np.exp(ft["beta"][-1]))
    # sequential models for the exposure
    X1 = np.column_stack([np.ones(N), Xe[:, idx["sglt2"]], Xe[:, idx["age10"]], Xe[:, idx["female"]]])
    fm1 = fit_logit(X1, y)
    seq = dict(crude=crude["sglt2"], m1=dict(b=fm1["beta"][1], se=fm1["se"][1], orci=or_ci(fm1["beta"][1], fm1["se"][1]), p=fm1["pval"][1]),
               m2=adj["sglt2"])
    # profile CI for sglt2
    pci = profile_ci(Xe, y, idx["sglt2"], fe)
    # global tests
    lr_glob = lr_test(fe, fnull, len(names) - 1)
    sc_glob = score_global(Xe, y)
    Vb = fe["cov"][1:, 1:]; bb = fe["beta"][1:]
    wald_glob = float(bb @ np.linalg.solve(Vb, bb))
    k = len(names)
    fitstats = dict(m2ll0=-2 * fnull["ll"], m2ll=-2 * fe["ll"], aic0=-2 * fnull["ll"] + 2, aic=-2 * fe["ll"] + 2 * k,
                    sc0=-2 * fnull["ll"] + np.log(N), sc=-2 * fe["ll"] + k * np.log(N))
    # pseudo R2
    mcf = 1 - fe["ll"] / fnull["ll"]
    cs = 1 - np.exp(2 * (fnull["ll"] - fe["ll"]) / N)
    nag = cs / (1 - np.exp(2 * fnull["ll"] / N))
    # VIF (linear regression of each predictor on the others)
    vif = {}
    for c in names[1:]:
        j = idx[c]
        others = [i for i in range(k) if i != j]
        bj = np.linalg.lstsq(Xe[:, others], Xe[:, j], rcond=None)[0]
        res = Xe[:, j] - Xe[:, others] @ bj
        r2 = 1 - res.var() / Xe[:, j].var()
        vif[c] = 1 / (1 - r2)
    # predicted probabilities for two patients (etiologic model)
    def pat(sg, age, fem, cci, prior, hf, egfr):
        x = np.array([1, sg, (age - 75) / 10, fem, cci == 1, cci == 2, prior, hf, 45 <= egfr < 60, egfr < 45], float)
        lp = float(x @ fe["beta"])
        return dict(lp=lp, p=expit(lp), x=x)
    pats = dict(A=pat(0, 82, 1, 2, 1, 0, 40), A_s=pat(1, 82, 1, 2, 1, 0, 40), B=pat(0, 68, 0, 0, 0, 0, 85), B_s=pat(1, 68, 0, 0, 0, 0, 85))
    R["etio"] = dict(names=names, fit=fe, glm=ge, crude=crude, adj=adj, lrt=lrt, wald=wald, trend=trend, seq=seq, prof_sglt2=pci,
                     lr_glob=lr_glob, sc_glob=sc_glob, wald_glob=(wald_glob, k - 1, st.chi2.sf(wald_glob, k - 1)),
                     fitstats=fitstats, mcfadden=mcf, coxsnell=cs, nagelkerke=nag, vif=vif, pats=pats, k=k,
                     epv=y.sum() / (k - 1), ll=fe["ll"], ll0=fnull["ll"])

    # ---------------------------------------------------------------- counts by category (Table 2 first column)
    cnt = {}
    for c, m in [("sglt2_1", s), ("sglt2_0", ~s), ("female_1", d["female"] == 1), ("female_0", d["female"] == 0),
                 ("cci_0", d["cci"] == 0), ("cci_1", d["cci"] == 1), ("cci_2", d["cci"] == 2), ("prior_1", d["prior"] == 1),
                 ("prior_0", d["prior"] == 0), ("hf_1", d["hf"] == 1), ("hf_0", d["hf"] == 0), ("egfr_0", d["egfr"] >= 60),
                 ("egfr_1", e1), ("egfr_2", e2)]:
        cnt[c] = (int(y[m].sum()), int(m.sum()), float(y[m].mean()))
    R["cnt"] = cnt

    # ================================================================ D. linearity (eGFR, age)
    base_cols = ["(Intercept)", "sglt2", "age10", "female", "cci12", "cci3", "prior", "hf"]
    Xb = Xe[:, [idx[c] for c in base_cols]]
    eg = d["egfr"].astype(float)
    kn = rcs_knots(eg, 4)
    fl = fit_logit(np.column_stack([Xb, eg]), y)
    Bs = rcs_basis(eg, kn)
    fs = fit_logit(np.column_stack([Xb, Bs]), y)
    fc = fe
    lin_lr = lr_test(fs, fl, 2)
    # age
    ag = d["age"].astype(float)
    kna = rcs_knots(ag, 4)
    Xa_lin = np.column_stack([Xe[:, [idx[c] for c in names if c != "age10"]], ag])
    Xa_rcs = np.column_stack([Xe[:, [idx[c] for c in names if c != "age10"]], rcs_basis(ag, kna)])
    fal, far = fit_logit(Xa_lin, y), fit_logit(Xa_rcs, y)
    age_lr = lr_test(far, fal, 2)
    # curve data: log OR relative to eGFR 90
    xs = np.linspace(15, 120, 211)
    ref = rcs_basis(np.array([90.0]), kn)[0]
    js = list(range(Xb.shape[1], Xb.shape[1] + 3))
    curve = []
    for xv in xs:
        c = np.zeros(fs["beta"].shape); c[js] = rcs_basis(np.array([xv]), kn)[0] - ref
        b, se = lincomb(fs, c)
        curve.append((xv, b, b - Z * se, b + Z * se))
    linb = fl["beta"][-1]
    R["lin"] = dict(knots=kn, lin_b=linb, lin_se=fl["se"][-1], lin_or10=np.exp(-10 * linb), lr=lin_lr,
                    aic_lin=-2 * fl["ll"] + 2 * len(fl["beta"]), aic_rcs=-2 * fs["ll"] + 2 * len(fs["beta"]),
                    aic_cat=-2 * fc["ll"] + 2 * len(fc["beta"]), age_knots=kna, age_lr=age_lr, curve=np.array(curve),
                    cat=[(0, 0, 0)], rcs_at={v: float(np.interp(v, xs, np.array(curve)[:, 1])) for v in (30, 40, 45, 50, 60, 75, 105, 120)})

    # ================================================================ E. interaction drug x HF
    Xi = np.column_stack([Xe, Xe[:, idx["sglt2"]] * Xe[:, idx["hf"]]])
    fi = fit_logit(Xi, y)
    ji = Xi.shape[1] - 1
    c0 = np.zeros(ji + 1); c0[idx["sglt2"]] = 1
    c1 = c0.copy(); c1[ji] = 1
    b_no, se_no = lincomb(fi, c0); b_hf, se_hf = lincomb(fi, c1)
    lr_i = lr_test(fi, fe, 1)
    strat = {}
    for h in (0, 1):
        m = d["hf"] == h
        strat[h] = dict(n_s=int((m & s).sum()), n_d=int((m & ~s).sum()), ev_s=int(y[m & s].sum()), ev_d=int(y[m & ~s].sum()))
    R["int"] = dict(fit=fi, b_int=fi["beta"][ji], se_int=fi["se"][ji], p_int=fi["pval"][ji], lr=lr_i,
                    or_no=or_ci(b_no, se_no), or_hf=or_ci(b_hf, se_hf), p_no=2 * st.norm.sf(abs(b_no / se_no)),
                    p_hf=2 * st.norm.sf(abs(b_hf / se_hf)), ror=or_ci(fi["beta"][ji], fi["se"][ji]),
                    b_hf_main=fi["beta"][idx["hf"]], strat=strat,
                    or_hf_in_s=np.exp(fi["beta"][idx["hf"]] + fi["beta"][ji]), or_hf_in_d=np.exp(fi["beta"][idx["hf"]]))

    # ================================================================ F. common outcome: OR vs RR
    pr = d["prior"] == 1
    r1, r0 = y[pr].mean(), y[~pr].mean()
    crude_rr_prior = r1 / r0
    crude_or_prior = (r1 / (1 - r1)) / (r0 / (1 - r0))
    fp_ = fit_poisson(Xe, y)
    rr = {c: dict(b=fp_["beta"][i], se=fp_["se"][i], se_model=fp_["se_model"][i], rrci=or_ci(fp_["beta"][i], fp_["se"][i]))
          for c, i in idx.items()}
    lb = fit_logbin(Xe, y, beta0=fp_["beta"])
    rr_lb = {c: or_ci(lb["beta"][i], lb["se"][i]) for c, i in idx.items()}
    # Zhang-Yu conversion of adjusted OR using the unexposed risk
    def zy(or_, p0):
        return or_ / (1 - p0 + p0 * or_)
    zy_prior = zy(adj["prior"]["orci"][0], r0)
    # marginal standardization for sglt2 (and prior)
    def std_risks(beta, Xm, col):
        X1_, X0_ = Xm.copy(), Xm.copy()
        X1_[:, col] = 1; X0_[:, col] = 0
        return float(expit(X1_ @ beta).mean()), float(expit(X0_ @ beta).mean())
    r_s1, r_s0 = std_risks(fe["beta"], Xe, idx["sglt2"])
    r_p1, r_p0 = std_risks(fe["beta"], Xe, idx["prior"])
    rng = np.random.default_rng(7)
    bs = []
    for _ in range(1000):
        ii = rng.integers(0, N, N)
        fb = fit_logit(Xe[ii], y[ii], beta0=fe["beta"])
        a1, a0 = std_risks(fb["beta"], Xe[ii], idx["sglt2"])
        c1_, c0_ = std_risks(fb["beta"], Xe[ii], idx["prior"])
        bs.append((a1 - a0, a1 / a0, c1_ - c0_, c1_ / c0_))
    bs = np.array(bs)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    # 'prediction at the means' (Muller & MacLehose): plug in covariate means
    xbar = Xe.mean(axis=0)
    xm1, xm0 = xbar.copy(), xbar.copy(); xm1[idx["sglt2"]] = 1; xm0[idx["sglt2"]] = 0
    R["rr"] = dict(prior_r1=r1, prior_r0=r0, prior_n1=int(pr.sum()), prior_e1=int(y[pr].sum()), prior_n0=int((~pr).sum()),
                   prior_e0=int(y[~pr].sum()), crude_rr_prior=crude_rr_prior, crude_or_prior=crude_or_prior,
                   mp=rr, lb=rr_lb, lb_maxp=lb["maxp"], zy_prior=zy_prior,
                   std_s=(r_s1, r_s0, r_s1 - r_s0, r_s1 / r_s0, q(bs[:, 0]), q(bs[:, 1])),
                   std_p=(r_p1, r_p0, r_p1 - r_p0, r_p1 / r_p0, q(bs[:, 2]), q(bs[:, 3])),
                   atmeans=(float(expit(xm1 @ fe["beta"])), float(expit(xm0 @ fe["beta"]))),
                   s_crude_r=(y[s].mean(), y[~s].mean()))

    # ================================================================ G. prediction model
    Xp, pn = design(d, "pred")
    fpm = fit_logit(Xp, y)
    ph = fpm["p"]
    auc = auc_delong(y, ph)
    hl = hosmer_lemeshow(y, ph)
    cal = cal_int_slope(y, ph)
    thr = {tt: confusion(y.astype(int), ph, tt) for tt in (0.1, 0.2, 0.3, 0.5)}
    fpr, tpr, th = roc_points(y.astype(int), ph)
    # bootstrap optimism (Harrell)
    rng = np.random.default_rng(2025)
    opt_c, test_sl, app_c = [], [], []
    for _ in range(500):
        ii = rng.integers(0, N, N)
        fb = fit_logit(Xp[ii], y[ii], beta0=fpm["beta"])
        ca = auc_delong(y[ii], fb["p"])[0]
        pt_ = expit(Xp @ fb["beta"])
        ct = auc_delong(y, pt_)[0]
        opt_c.append(ca - ct)
        test_sl.append(cal_int_slope(y, pt_)["slope"])
    opt = float(np.mean(opt_c))
    # flexible calibration curve: logistic of y on rcs(logit(p), 3 knots)
    lpd = logit(ph)
    knc = rcs_knots(lpd, 3)
    fcal = fit_logit(np.column_stack([np.ones(N), rcs_basis(lpd, knc)]), y)
    pgrid = np.linspace(np.quantile(ph, 0.01), np.quantile(ph, 0.99), 120)
    calc = expit(np.column_stack([np.ones(len(pgrid)), rcs_basis(logit(pgrid), knc)]) @ fcal["beta"])
    fnull_p = fit_logit(np.ones((N, 1)), y)
    R["pred"] = dict(names=pn, fit=fpm, auc=auc, hl=hl, cal=cal, thr=thr, roc=(fpr, tpr), opt_c=opt,
                     auc_corr=auc[0] - opt, slope_corr=float(np.mean(test_sl)), brier=brier(y, ph),
                     brier0=brier(y, np.full(N, y.mean())), prev=y.mean(), p=ph, y=y,
                     mcfadden=1 - fpm["ll"] / fnull_p["ll"],
                     nagelkerke=(1 - np.exp(2 * (fnull_p["ll"] - fpm["ll"]) / N)) / (1 - np.exp(2 * fnull_p["ll"] / N)),
                     calcurve=(pgrid, calc), p_range=(float(ph.min()), float(ph.max())),
                     p_q=(float(np.quantile(ph, .1)), float(np.median(ph)), float(np.quantile(ph, .9))),
                     p_by=(float(ph[y == 1].mean()), float(ph[y == 0].mean())),
                     auc_prior=auc_delong(y, d["prior"].astype(float)), pairs=int(y.sum() * (N - y.sum())),
                     sens_prior=float(d["prior"][y == 1].mean()), spec_prior=float(1 - d["prior"][y == 0].mean()))

    # ---------------------------------------------------------------- external validation
    ex = external()
    ye = ex["hosp"].astype(float)
    Xx, _ = design(ex, "pred")
    pe = expit(Xx @ fpm["beta"])
    auc_e = auc_delong(ye, pe)
    cal_e = cal_int_slope(ye, pe)
    hl_e = hosmer_lemeshow(ye, pe)
    thr_e = {tt: confusion(ye.astype(int), pe, tt) for tt in (0.2, 0.3)}
    # recalibration: update intercept only
    upd = fit_logit(np.ones((len(ye), 1)), ye, offset=logit(pe))
    lpe = logit(pe)
    fcal_e = fit_logit(np.column_stack([np.ones(len(ye)), rcs_basis(lpe, rcs_knots(lpe, 3))]), ye)
    pgrid_e = np.linspace(np.quantile(pe, 0.01), np.quantile(pe, 0.99), 120)
    calc_e = expit(np.column_stack([np.ones(len(pgrid_e)), rcs_basis(logit(pgrid_e), rcs_knots(lpe, 3))]) @ fcal_e["beta"])
    R["ext"] = dict(n=len(ye), ev=int(ye.sum()), prev=ye.mean(), mean_p=float(pe.mean()), auc=auc_e, cal=cal_e, hl=hl_e,
                    thr=thr_e, roc=roc_points(ye.astype(int), pe), upd=upd["beta"][0], p=pe, y=ye,
                    calcurve=(pgrid_e, calc_e), age=float(ex["age"].mean()), prior=float(ex["prior"].mean()),
                    brier=brier(ye, pe))

    # ---------------------------------------------------------------- H-L and sample size
    # prediction model with eGFR entered as a straight line (slightly misspecified, see linearity section)
    def xlin(dd):
        Xq, _ = design(dd, "pred")
        return np.column_stack([Xq[:, :7], dd["egfr"] / 10.0])
    fl_d = fit_logit(xlin(d), y)
    hl_d = hosmer_lemeshow(y, fl_d["p"])
    big = _cohort_draw(np.random.default_rng(34), 100000)
    yb = big["hosp"].astype(float)
    fl_b = fit_logit(xlin(big), yb)
    hl_b = hosmer_lemeshow(yb, fl_b["p"])
    md = lambda h: max(abs(r["obs_rate"] - r["mean_p"]) for r in h[3])
    R["hlss"] = dict(dev=(hl_d[0], hl_d[2], md(hl_d)), big=(hl_b[0], hl_b[2], md(hl_b)), nbig=len(yb), evbig=int(yb.sum()),
                     auc_big=auc_delong(yb, fl_b["p"])[0], auc_dev=auc_delong(y, fl_d["p"])[0], rows_d=hl_d[3], rows_b=hl_b[3])

    # ---------------------------------------------------------------- overfitting demo
    rng = np.random.default_rng(12)
    sub = rng.choice(N, 300, replace=False)
    rest = np.setdiff1d(np.arange(N), sub)
    noise = rng.normal(size=(N, 10))
    Xo = np.column_stack([Xp, noise])
    fo = fit_logit(Xo[sub], y[sub])
    app = auc_delong(y[sub], fo["p"])[0]
    po = expit(Xo[rest] @ fo["beta"])
    val = auc_delong(y[rest], po)[0]
    calo = cal_int_slope(y[rest], po)
    # smaller model in the same 300
    fs_ = fit_logit(Xp[sub], y[sub])
    app_s = auc_delong(y[sub], fs_["p"])[0]
    pso = expit(Xp[rest] @ fs_["beta"])
    val_s = auc_delong(y[rest], pso)[0]
    cal_s = cal_int_slope(y[rest], pso)
    # bootstrap-corrected C for the overfit model
    oc = []
    for _ in range(300):
        ii = sub[rng.integers(0, 300, 300)]
        try:
            fb = fit_logit(Xo[ii], y[ii])
        except np.linalg.LinAlgError:
            continue
        oc.append(auc_delong(y[ii], fb["p"])[0] - auc_delong(y[sub], expit(Xo[sub] @ fb["beta"]))[0])
    R["overfit"] = dict(n=300, ev=int(y[sub].sum()), k_big=Xo.shape[1] - 1, k_small=Xp.shape[1] - 1,
                        app=app, val=val, slope=calo["slope"], app_s=app_s, val_s=val_s, slope_s=cal_s["slope"],
                        corr=app - float(np.mean(oc)), n_rest=len(rest))

    # ================================================================ H. separation
    au, co, yy = irae()
    Xs = np.column_stack([np.ones(len(yy)), au, co])
    gs = glm_R(Xs, yy)
    ff = firth(Xs, yy)
    fci = [firth_profile_ci(Xs, yy, j, ff) for j in (1, 2)]
    flr = [firth_lr(Xs, yy, j, ff) for j in (1, 2)]
    # ML without the separating variable
    R["sep"] = dict(n=len(yy), ev=int(yy.sum()), glm=gs, firth=ff, firth_ci=fci, firth_lr=flr,
                    n_au=int(au.sum()), ev_au=int(yy[au == 1].sum()), n_co=int(co.sum()), ev_co=int(yy[co == 1].sum()))

    # ================================================================ I. matched pairs (conditional logistic)
    pairs = dict(both=30, case_only=50, ctrl_only=25, neither=145)
    npair = sum(pairs.values())
    rows = []
    pid = 0
    for kind, m in pairs.items():
        for _ in range(m):
            ce = kind in ("both", "case_only"); cte = kind in ("both", "ctrl_only")
            rows.append((pid, 1, int(ce))); rows.append((pid, 0, int(cte))); pid += 1
    rows = np.array(rows)
    Xpair = np.zeros((len(rows), npair + 1))
    Xpair[np.arange(len(rows)), rows[:, 0]] = 1
    Xpair[:, -1] = rows[:, 2]
    fu = fit_logit(Xpair, rows[:, 1].astype(float), maxit=200)
    cases_exp = pairs["both"] + pairs["case_only"]; ctrl_exp = pairs["both"] + pairs["ctrl_only"]
    or_ignore = (cases_exp / (npair - cases_exp)) / (ctrl_exp / (npair - ctrl_exp))
    cond = pairs["case_only"] / pairs["ctrl_only"]
    se_cond = np.sqrt(1 / pairs["case_only"] + 1 / pairs["ctrl_only"])
    R["match"] = dict(pairs=pairs, npair=npair, cond=cond, cond_ci=(np.exp(np.log(cond) - Z * se_cond), np.exp(np.log(cond) + Z * se_cond)),
                      uncond=np.exp(fu["beta"][-1]), ignore=or_ignore, cases_exp=cases_exp, ctrl_exp=ctrl_exp)

    # ================================================================ J. marginal vs conditional OR (random intercept)
    sig = 1.0
    b_c = np.log(2.0)
    def marg(b0_, x):
        f = lambda u: expit(b0_ + b_c * x + sig * u) * st.norm.pdf(u)
        return integrate.quad(f, -10, 10)[0]
    # choose b0 so that marginal risk in unexposed = 0.20
    from scipy.optimize import brentq
    b0m = brentq(lambda b: marg(b, 0) - 0.20, -5, 2)
    m0, m1 = marg(b0m, 0), marg(b0m, 1)
    R["mixed"] = dict(sigma=sig, or_cond=2.0, m0=m0, m1=m1, or_marg=(m1 / (1 - m1)) / (m0 / (1 - m0)))

    # ================================================================ K. non-collapsibility (balanced prognostic factor)
    p0s = np.array([0.2, 0.6])
    o1s = p0s / (1 - p0s) * 2.0
    p1s = o1s / (1 + o1s)
    P0, P1 = p0s.mean(), p1s.mean()
    R["noncoll"] = dict(p0=p0s, p1=p1s, P0=P0, P1=P1, or_marg=(P1 / (1 - P1)) / (P0 / (1 - P0)),
                        rr_strata=p1s / p0s, rr_marg=P1 / P0)

    # ================================================================ L. p / odds / logit table
    R["plo"] = [(pp, pp / (1 - pp), np.log(pp / (1 - pp))) for pp in (0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99)]

    # ================================================================ M. extra checks used in the text
    # three tests for the 2x2 table (Wald, LR, score = Pearson chi-square)
    y22 = np.r_[np.ones(48), np.zeros(352), np.ones(102), np.zeros(498)]
    X22 = np.column_stack([np.ones(1000), np.r_[np.ones(400), np.zeros(600)]])
    f22b, f22n = fit_logit(X22, y22), fit_logit(X22[:, :1], y22)
    lr22 = lr_test(f22b, f22n, 1)
    sc22 = score_global(X22, y22)
    pear = st.chi2_contingency(np.array([[48, 352], [102, 498]]), correction=False)
    pc22 = profile_ci(X22, y22, 1, f22b)
    R["t22"].update(prof=(float(np.exp(pc22[0])), float(np.exp(pc22[1]))), lr=lr22, score=(sc22[0], sc22[2]), wald=(f22b["z"][1] ** 2, f22b["pval"][1]), pearson=(pear[0], pear[1]))
    # vancomycin: score test and helpful quantities
    Xv = np.column_stack([np.ones(len(t)), t])
    scv = score_global(Xv, a.astype(float))
    R["vanco"].update(score=(scv[0], scv[2]), p0=float(expit(b0)), slope50=b1 / 4, wald2=(b1 / se1) ** 2)
    # profile-likelihood CIs for every coefficient of the etiologic model
    E_ = R["etio"]
    Xe_, names_ = design(cohort(), "etio")
    prof_all = {}
    for j, nm in enumerate(names_):
        lo_, hi_ = profile_ci(Xe_, y, j, E_["fit"])
        prof_all[nm] = (float(np.exp(lo_)), float(np.exp(hi_)))
    E_["prof_all"] = prof_all
    E_["p_ref"] = float(expit(E_["adj"]["(Intercept)"]["b"]))
    # separation: ML likelihood-ratio tests (the likelihood itself converges even though beta diverges)
    au, co, yy = irae()
    Xs = np.column_stack([np.ones(len(yy)), au, co])
    gs = glm_R(Xs, yy)
    mlr = {}
    for j, nm in ((1, "auto"), (2, "combo")):
        keep = [k for k in range(3) if k != j]
        gr = glm_R(Xs[:, keep], yy)
        s_ = gr["dev"] - gs["dev"]
        mlr[nm] = (s_, st.chi2.sf(s_, 1))
    # Firth: profile CI for the intercept as well (for the logistf-style output)
    ff = R["sep"]["firth"]
    R["sep"].update(ml_lr=mlr, firth_ci0=firth_profile_ci(Xs, yy, 0, ff), maxmu=float(gs["mu"].max()))
    # score test for the SGLT2 coefficient given the other covariates (reduced-model fit, full information)
    js = names_.index("sglt2")
    keep = [k for k in range(len(names_)) if k != js]
    fr = fit_logit(Xe_[:, keep], y)
    bfull = np.zeros(len(names_)); bfull[keep] = fr["beta"]
    pr_ = expit(Xe_ @ bfull)
    U = Xe_.T @ (y - pr_)
    Iinf = Xe_.T @ (Xe_ * (pr_ * (1 - pr_))[:, None])
    sc_s = float(U @ np.linalg.solve(Iinf, U))
    E_["score_sglt2"] = (sc_s, st.chi2.sf(sc_s, 1))
    E_["wald_sglt2"] = (E_["adj"]["sglt2"]["z"] ** 2, E_["adj"]["sglt2"]["p"])
    # NNT from the standardized risk difference and its bootstrap limits
    rd_, (rlo, rhi) = R["rr"]["std_s"][2], R["rr"]["std_s"][4]
    R["rr"]["nnt"] = (-1 / rd_, -1 / rlo, -1 / rhi)
    # classification: 'predict nobody is hospitalized'
    R["pred"]["acc_none"] = 1 - float(R["pred"]["prev"])
    # two illustrative patients: OR constant, risk difference and risk ratio vary
    pa_ = E_["pats"]
    R["pats_eff"] = {k: dict(p0=pa_[k]["p"], p1=pa_[k + "_s"]["p"], rd=pa_[k + "_s"]["p"] - pa_[k]["p"],
                             rr=pa_[k + "_s"]["p"] / pa_[k]["p"],
                             orr=(pa_[k + "_s"]["p"] / (1 - pa_[k + "_s"]["p"])) / (pa_[k]["p"] / (1 - pa_[k]["p"])))
                     for k in ("A", "B")}
    return R


def report():
    R = compute()
    np.set_printoptions(suppress=True, linewidth=150)
    V = R["vanco"]
    print("=" * 70, "\nA. vancomycin")
    print(f" n {V['n']} AKI {V['events']} ({V['rate']*100:.1f}%) trough median {V['tmed']} IQR {V['q1']:.1f}-{V['q3']:.1f} range {V['tmin']}-{V['tmax']}")
    print(f" LPM: b0 {V['lpm'][0]:.4f} b1 {V['lpm'][1]:.5f}; pred<0: {V['lpm_neg']} pred>1: {V['lpm_gt1']}; hits 0 at {V['lpm_x0']:.2f}, 1 at {V['lpm_x1']:.2f}; at min {V['lpm_at_min']:.3f} at max {V['lpm_at_max']:.3f}")
    print(f" logistic: b0 {V['b0']:.4f} (SE {V['se0']:.4f})  b1 {V['b1']:.5f} (SE {V['se1']:.5f}) z {V['z1']:.3f} p {V['p1']:.2e}")
    print(f"  OR/1 {V['or1']:.4f} CI {V['or1_ci'][0]:.4f}-{V['or1_ci'][1]:.4f};  OR/5 {V['or5']:.3f} CI {V['or5_ci'][0]:.3f}-{V['or5_ci'][1]:.3f}")
    print(f"  profile CI b1 {V['prof_ci'][0]:.5f}-{V['prof_ci'][1]:.5f} -> OR/5 {V['prof_or5'][0]:.3f}-{V['prof_or5'][1]:.3f}; wald b1 {V['wald_ci'][0]:.5f}-{V['wald_ci'][1]:.5f}")
    print(f"  ll {V['ll']:.4f} ll0 {V['ll0']:.4f} dev {V['dev']:.3f} dev0 {V['dev0']:.3f} LR {V['lr']:.3f} p {V['lrp']:.2e}  p=0.5 at {V['x50']:.2f}")
    for x in V["p"]:
        print(f"   trough {x}: p {V['p'][x]:.4f} odds {V['odds'][x]:.4f} logit {np.log(V['odds'][x]):.4f}")
    xs = sorted(V["p"])
    for a_, b_ in zip(xs[:-1], xs[1:]):
        print(f"   {a_}->{b_}: RD {(V['p'][b_]-V['p'][a_])*100:.1f}%p  RR {V['p'][b_]/V['p'][a_]:.3f}  OR {V['odds'][b_]/V['odds'][a_]:.4f}")
    g = V["glm"]
    print("  R glm:", g["beta"], g["se"], g["z"], g["pval"], "dev", g["dev"], "null", g["nulldev"], "aic", g["aic"], "iter", g["iters"])
    T = R["t22"]
    print("=" * 70, "\nB. 2x2 of ch05")
    print(f" b0 {T['b0']:.5f} (ln(102/498)={T['lnodds_B']:.5f}) b1 {T['b1']:.5f} OR {T['or_']:.5f} hand {T['or_hand']:.5f}")
    print(f" SE {T['se1']:.5f} woolf {T['woolf']:.5f} CI {T['ci'][0]:.4f}-{T['ci'][1]:.4f} lnoddsA {T['lnodds_A']:.5f} pA {T['pA']:.4f} pB {T['pB']:.4f} z {T['z']:.3f} p {T['p']:.4f}")
    print("=" * 70, "\nC. p/odds/logit")
    for r in R["plo"]:
        print("  p %.2f odds %.4f logit %.3f" % r)
    D = R["desc"]; E = R["etio"]
    print("=" * 70, "\nD. cohort")
    print(" ", {k: v for k, v in D.items() if k != "base"})
    for k, v in D["base"].items():
        print("   base", k, np.round(v, 3))
    for k, v in R["cnt"].items():
        print(f"   cnt {k}: {v[0]}/{v[1]} ({v[2]*100:.1f}%)")
    print(" crude / adjusted OR")
    for c in E["names"][1:]:
        cr, ad = E["crude"][c], E["adj"][c]
        print(f"  {c:9s} crude {cr['orci'][0]:.2f} ({cr['orci'][1]:.2f}-{cr['orci'][2]:.2f}) p {cr['p']:.2e} | adj b {ad['b']:.4f} se {ad['se']:.4f} z {ad['z']:.3f} OR {ad['orci'][0]:.2f} ({ad['orci'][1]:.2f}-{ad['orci'][2]:.2f}) p {ad['p']:.2e}")
    print(f"  intercept b {E['adj']['(Intercept)']['b']:.4f} se {E['adj']['(Intercept)']['se']:.4f} p0={expit(E['adj']['(Intercept)']['b']):.4f}")
    for g_ in ["sglt2", "age10", "female", "cci", "prior", "hf", "egfr"]:
        print(f"  crude LR {g_}: {E['crude'][g_+'_lr'][0]:.2f} p {E['crude'][g_+'_lr'][1]:.2e}")
    for g_, v in E["lrt"].items():
        print(f"  adjusted LR drop {g_}: {v[0]:.3f} p {v[1]:.2e}")
    for g_, v in E["wald"].items():
        print(f"  joint Wald {g_}: {v[0]:.3f} p {v[1]:.2e}")
    print(f"  trend CCI: b {E['trend']['b']:.4f} OR {E['trend']['or_']:.3f} p {E['trend']['p']:.2e}")
    sq = E["seq"]
    print(f"  seq: crude {sq['crude']['orci']}, m1 {sq['m1']['orci']}, full {sq['m2']['orci']}")
    print(f"  profile CI sglt2: OR {np.exp(E['prof_sglt2'][0]):.4f}-{np.exp(E['prof_sglt2'][1]):.4f}  wald {E['adj']['sglt2']['orci'][1]:.4f}-{E['adj']['sglt2']['orci'][2]:.4f}")
    print(f"  global LR {E['lr_glob'][0]:.3f} p {E['lr_glob'][1]:.2e}; score {E['sc_glob'][0]:.3f} df {E['sc_glob'][1]} p {E['sc_glob'][2]:.2e}; wald {E['wald_glob'][0]:.3f} p {E['wald_glob'][2]:.2e}")
    print("  fitstats", {k: round(v, 3) for k, v in E["fitstats"].items()})
    print(f"  McFadden {E['mcfadden']:.4f} CoxSnell {E['coxsnell']:.4f} Nagelkerke {E['nagelkerke']:.4f}  EPV {E['epv']:.1f} (k-1={E['k']-1})")
    print("  VIF", {k: round(v, 3) for k, v in E["vif"].items()})
    for k, v in E["pats"].items():
        print(f"  patient {k}: lp {v['lp']:.4f} p {v['p']:.4f}  x={v['x']}")
    g = E["glm"]
    print("  R glm iter", g["iters"], "dev", round(g["dev"], 3), "null", round(g["nulldev"], 3), "aic", round(g["aic"], 3), "warn01", g["warn01"])
    for nm, b, se_, z, p in zip(E["names"], g["beta"], g["se"], g["z"], g["pval"]):
        print(f"   {nm:12s} {b: .5f} {se_: .5f} {z: .3f} {p:.3g}")
    L = R["lin"]
    print("=" * 70, "\nE. linearity")
    print(f" eGFR knots {np.round(L['knots'],1)}; linear b {L['lin_b']:.5f} (OR per 10 lower {L['lin_or10']:.3f}); LR nonlin {L['lr'][0]:.3f} p {L['lr'][1]:.2e}")
    print(f" AIC lin {L['aic_lin']:.2f} rcs {L['aic_rcs']:.2f} cat {L['aic_cat']:.2f};  age knots {L['age_knots']}, age nonlin LR {L['age_lr'][0]:.3f} p {L['age_lr'][1]:.3f}")
    print("  rcs logOR vs 90 at", {k: (round(v, 3), round(np.exp(v), 2)) for k, v in L["rcs_at"].items()})
    I_ = R["int"]
    print("=" * 70, "\nF. interaction")
    print(f" b_int {I_['b_int']:.4f} se {I_['se_int']:.4f} p {I_['p_int']:.4f} LR {I_['lr'][0]:.3f} p {I_['lr'][1]:.4f}")
    print(f" OR no HF {np.round(I_['or_no'],3)} p {I_['p_no']:.4f}; OR HF {np.round(I_['or_hf'],3)} p {I_['p_hf']:.4f}; ratio {np.round(I_['ror'],3)}")
    print(f" strat {I_['strat']}; OR of HF among DPP4 {I_['or_hf_in_d']:.3f}, among SGLT2 {I_['or_hf_in_s']:.3f}")
    RR = R["rr"]
    print("=" * 70, "\nG. OR vs RR")
    print(f" prior: {RR['prior_e1']}/{RR['prior_n1']} = {RR['prior_r1']:.4f} vs {RR['prior_e0']}/{RR['prior_n0']} = {RR['prior_r0']:.4f}; crude RR {RR['crude_rr_prior']:.3f} OR {RR['crude_or_prior']:.3f}; ZY of adj OR {RR['zy_prior']:.3f}")
    for c in R["etio"]["names"][1:]:
        a_ = R["etio"]["adj"][c]["orci"]; m_ = RR["mp"][c]["rrci"]; l_ = RR["lb"][c]
        print(f"  {c:9s} OR {a_[0]:.2f} ({a_[1]:.2f}-{a_[2]:.2f}) | mPoisson RR {m_[0]:.2f} ({m_[1]:.2f}-{m_[2]:.2f}) robustSE {RR['mp'][c]['se']:.4f} modelSE {RR['mp'][c]['se_model']:.4f} | logbin {l_[0]:.2f} ({l_[1]:.2f}-{l_[2]:.2f})")
    print(f"  logbin max p {RR['lb_maxp']:.3f}")
    ss, sp = RR["std_s"], RR["std_p"]
    print(f"  std sglt2: {ss[0]:.4f} vs {ss[1]:.4f}, RD {ss[2]*100:.2f}%p CI {ss[4][0]*100:.2f}-{ss[4][1]*100:.2f}; RR {ss[3]:.3f} CI {ss[5][0]:.3f}-{ss[5][1]:.3f}")
    print(f"  std prior: {sp[0]:.4f} vs {sp[1]:.4f}, RD {sp[2]*100:.2f}%p CI {sp[4][0]*100:.2f}-{sp[4][1]*100:.2f}; RR {sp[3]:.3f} CI {sp[5][0]:.3f}-{sp[5][1]:.3f}")
    print(f"  at means sglt2: {RR['atmeans']}; crude risks sglt2 {RR['s_crude_r']}")
    P = R["pred"]
    print("=" * 70, "\nH. prediction model")
    for nm, b, se_ in zip(P["names"], P["fit"]["beta"], P["fit"]["se"]):
        print(f"  {nm:12s} {b: .4f} ({se_:.4f}) OR {np.exp(b):.2f}")
    print(f" prior-only AUC {P['auc_prior']} sens {P['sens_prior']:.4f} spec {P['spec_prior']:.4f} pairs {P['pairs']}")
    print(f" vanco p18 {R['vanco']['p18']:.4f}")
    print(f" AUC {P['auc'][0]:.4f} ({P['auc'][2]:.4f}-{P['auc'][3]:.4f}) optimism {P['opt_c']:.4f} corrected {P['auc_corr']:.4f} slope corr {P['slope_corr']:.4f}")
    print(f" HL {P['hl'][0]:.3f} df {P['hl'][1]} p {P['hl'][2]:.3f}; cal {P['cal']}; brier {P['brier']:.4f} brier0 {P['brier0']:.4f} scaled {1-P['brier']/P['brier0']:.3f}")
    print(f" McFadden {P['mcfadden']:.4f} Nagelkerke {P['nagelkerke']:.4f}; p range {P['p_range']}, p10/50/90 {P['p_q']} mean p by y {P['p_by']}")
    for r in P["hl"][3]:
        print("   decile", r)
    for tt, c in P["thr"].items():
        print(f"  thr {tt}: {c}")
    X_ = R["ext"]
    print(f" EXT n {X_['n']} ev {X_['ev']} prev {X_['prev']:.4f} mean p {X_['mean_p']:.4f}; age {X_['age']:.1f} prior {X_['prior']:.3f}")
    print(f"  AUC {np.round(X_['auc'],4)} cal {X_['cal']} HL {X_['hl'][0]:.2f} p {X_['hl'][2]:.2e} upd int {X_['upd']:.4f} brier {X_['brier']:.4f}")
    for tt, c in X_["thr"].items():
        print(f"  ext thr {tt}: {c}")
    for r in X_["hl"][3]:
        print("   ext decile", r)
    H = R["hlss"]
    print(f" HL(linear eGFR) dev n=2400: stat {H['dev'][0]:.2f} p {H['dev'][1]:.3f} maxdev {H['dev'][2]*100:.2f}%p auc {H['auc_dev']:.3f};  n={H['nbig']}: stat {H['big'][0]:.2f} p {H['big'][1]:.2e} maxdev {H['big'][2]*100:.2f}%p ev {H['evbig']} auc {H['auc_big']:.3f}")
    O = R["overfit"]
    print(f" overfit: {O}")
    S_ = R["sep"]
    print("=" * 70, "\nI. separation")
    g = S_["glm"]
    print(f" n {S_['n']} ev {S_['ev']} au {S_['ev_au']}/{S_['n_au']} co {S_['ev_co']}/{S_['n_co']}")
    print(f" R glm iters {g['iters']} conv {g['conv']} warn01 {g['warn01']} dev {g['dev']:.4f} null {g['nulldev']:.4f} aic {g['aic']:.3f}")
    for b, se_, z, p in zip(g["beta"], g["se"], g["z"], g["pval"]):
        print(f"   {b: .5f} {se_: .3f} {z: .3f} {p:.4f}")
    f_ = S_["firth"]
    print(f" firth beta {f_['beta']} se {f_['se']}  OR {np.exp(f_['beta'][1:])}")
    print(f"  profile CI (OR) auto {np.exp(S_['firth_ci'][0])}  combo {np.exp(S_['firth_ci'][1])}; LR {S_['firth_lr']}")
    print(f" ML LR auto {S_['ml_lr']['auto'][0]:.3f} p {S_['ml_lr']['auto'][1]:.2e}; combo {S_['ml_lr']['combo'][0]:.3f} p {S_['ml_lr']['combo'][1]:.4f}; max mu {S_['maxmu']:.10f}")
    print(f" firth intercept profile CI {S_['firth_ci0']}; log CIs auto {np.round(S_['firth_ci'][0],4)} combo {np.round(S_['firth_ci'][1],4)}")
    print("=" * 70, "\nM. extras")
    T = R["t22"]
    print(f" 2x2 profile CI {T['prof'][0]:.4f}-{T['prof'][1]:.4f} vs Wald {T['ci'][0]:.4f}-{T['ci'][1]:.4f}")
    print(f" 2x2 Wald {T['wald'][0]:.3f} p {T['wald'][1]:.4f}; LR {T['lr'][0]:.3f} p {T['lr'][1]:.4f}; score {T['score'][0]:.3f} p {T['score'][1]:.4f}; Pearson {T['pearson'][0]:.3f} p {T['pearson'][1]:.4f}")
    V = R["vanco"]
    print(f" vanco score {V['score'][0]:.2f} p {V['score'][1]:.2e}; Wald {V['wald2']:.2f}; LR {V['lr']:.2f}; p at x=0 {V['p0']:.4f}; slope at p=.5 {V['slope50']:.4f}")
    E = R["etio"]
    print(f" reference-person risk {E['p_ref']:.4f}")
    for k, v in E["prof_all"].items():
        w = E["adj"][k]["orci"]
        print(f"  profile {k:12s} {v[0]:.4f}-{v[1]:.4f}   wald {w[1]:.4f}-{w[2]:.4f}")
    print(f" sglt2 adjusted: Wald {E['wald_sglt2'][0]:.3f} p {E['wald_sglt2'][1]:.4f}; score {E['score_sglt2'][0]:.3f} p {E['score_sglt2'][1]:.4f}; LR {E['lrt']['sglt2'][0]:.3f} p {E['lrt']['sglt2'][1]:.4f}")
    print(f" NNT {R['rr']['nnt']}")
    print(" acc if nobody predicted:", round(R["pred"]["acc_none"], 4))
    for k, v in R["pats_eff"].items():
        print(f"  patient {k}: p0 {v['p0']:.4f} p1 {v['p1']:.4f} RD {v['rd']*100:.2f}%p RR {v['rr']:.3f} OR {v['orr']:.4f}")
    M = R["match"]
    print("=" * 70, "\nJ. matched pairs", M)
    print(" mixed", R["mixed"])
    print(" noncoll", R["noncoll"])


if __name__ == "__main__":
    report()
