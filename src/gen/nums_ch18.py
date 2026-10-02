"""Numbers for chapter 18 (정책 효과의 평가: 전후 비교, ITS, 대조군 ITS, DID).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch18.py
All numbers quoted in content/ch18.html come from this script (printed and saved to gen/_ch18_nums.json).

Running example (hypothetical): a pilot programme that restricts long-term sedative-hypnotic prescribing
for outpatients aged >=65, started in January 2018 in 20 pilot districts (시군구); 40 comparison districts.
Monthly claims aggregates January 2015 - December 2019 (36 months before, 24 after).
Outcome: patients with a sedative-hypnotic prescription per 1,000 outpatients aged >=65, per month.
True values used in the simulation: common pre-trend -0.25/month, a nationwide safety letter in the same
month (-2.0 in every district), pilot effect: level -5.0 and slope -0.20/month."""
import json, os, warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.arima.model import ARIMA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
SEED = int(os.environ.get("CH18_SEED", "91"))


def pr(*a):
    print(*a)


def hdr(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ------------------------------------------------------------------ simulation
T, T0 = 60, 36
t = np.arange(1, T + 1)
post = (t > T0).astype(int)
taft = np.where(t > T0, t - T0, 0)
mon = (t - 1) % 12            # 0 = January
year = 2015 + (t - 1) // 12
NP_, NC_ = 20, 40


def simulate(seed):
    rng = np.random.default_rng(seed)
    treat = np.r_[np.ones(NP_, int), np.zeros(NC_, int)]
    a = np.where(treat == 1, rng.normal(96, 6, NP_ + NC_), rng.normal(84, 6, NP_ + NC_))
    nbase = np.round(np.exp(rng.normal(np.log(9000), 0.45, NP_ + NC_)) / 100) * 100
    g = 1 + 0.002 * (t - 1)
    season = 0.9 * np.cos(2 * np.pi * mon / 12)
    u = np.zeros(T)
    eps = rng.normal(0, 0.9, T)
    for i in range(T):
        u[i] = (0.6 * u[i - 1] if i else 0) + eps[i]
    rows = []
    for i in range(NP_ + NC_):
        v = np.zeros(T)
        e = rng.normal(0, 1.5, T)
        for k in range(T):
            v[k] = (0.4 * v[k - 1] if k else 0) + e[k]
        mu = a[i] - 0.25 * t + season + u - 2.0 * post + treat[i] * (-5.0 * post - 0.20 * taft) + v
        n = np.round(nbase[i] * g).astype(int)
        c = rng.poisson(n * mu / 1000)
        rows.append(pd.DataFrame(dict(unit=i + 1, treat=treat[i], t=t, post=post, taft=taft, n=n, cnt=c)))
    return pd.concat(rows, ignore_index=True)


d = simulate(SEED)
d["rate"] = d.cnt / d.n * 1000
ag = d.groupby(["treat", "t"]).agg(cnt=("cnt", "sum"), n=("n", "sum")).reset_index()
ag["rate"] = ag.cnt / ag.n * 1000
PIL = ag[ag.treat == 1].reset_index(drop=True)
COM = ag[ag.treat == 0].reset_index(drop=True)
yp, yc = PIL.rate.values, COM.rate.values
yd = yp - yc
X = sm.add_constant(np.column_stack([t, post, taft]))
XN = ["const", "time", "post", "time_after"]
OUT["series"] = dict(yp=yp.tolist(), yc=yc.tolist(), yd=yd.tolist(), np=PIL.n.tolist(), nc=COM.n.tolist(),
                     cp=PIL.cnt.tolist(), cc=COM.cnt.tolist())

hdr("data")
pr("seed", SEED, "| pilot districts", NP_, "comparison", NC_)
pr("pilot denominators: month 1 %d, month 60 %d ; comparison %d, %d" % (PIL.n[0], PIL.n[59], COM.n[0], COM.n[59]))
pr("district size (mean n over period) pilot min/median/max:",
   d[d.treat == 1].groupby("unit").n.mean().describe()[["min", "50%", "max"]].round(0).tolist(),
   " comparison:", d[d.treat == 0].groupby("unit").n.mean().describe()[["min", "50%", "max"]].round(0).tolist())
for k in (0, 1, 34, 35, 36, 37, 47, 59):
    pr("t=%2d %d-%02d post=%d taft=%2d | pilot cnt %6d n %7d rate %.2f | comp cnt %6d n %7d rate %.2f | diff %.2f" %
       (t[k], year[k], mon[k] + 1, post[k], taft[k], PIL.cnt[k], PIL.n[k], yp[k], COM.cnt[k], COM.n[k], yc[k], yd[k]))

# ------------------------------------------------------------------ 가. naive before-after
hdr("가. before-after")
nv = {}
for nm, y in (("pilot", yp), ("comp", yc)):
    pre, po = y[:T0], y[T0:]
    tt = st.ttest_ind(po, pre, equal_var=False)
    ci = tt.confidence_interval()
    nv[nm] = dict(pre=pre.mean(), post=po.mean(), diff=po.mean() - pre.mean(), t=tt.statistic, p=tt.pvalue,
                  df=tt.df, lo=ci.low, hi=ci.high, sd_pre=pre.std(ddof=1), sd_post=po.std(ddof=1))
    pr("%s: pre mean %.2f (SD %.2f) post mean %.2f (SD %.2f) diff %.2f (95%% CI %.2f to %.2f) Welch t=%.2f df=%.1f p=%.2e" %
       (nm, pre.mean(), pre.std(ddof=1), po.mean(), po.std(ddof=1), po.mean() - pre.mean(), ci.low, ci.high,
        tt.statistic, tt.df, tt.pvalue))
    pr("   relative change %.1f%%" % (100 * (po.mean() - pre.mean()) / pre.mean()))
OUT["naive"] = nv
# what the pre-existing trend alone would give: distance between period midpoints = 30 months
pre_fit = sm.OLS(yp[:T0], sm.add_constant(t[:T0])).fit()
pr("pilot pre-period slope (first 36 months only): %.4f per month; x30 months = %.2f" %
   (pre_fit.params[1], pre_fit.params[1] * 30))
pr("period midpoints: pre %.1f, post %.1f, distance %.1f months" % (t[:T0].mean(), t[T0:].mean(), t[T0:].mean() - t[:T0].mean()))
pr("first-year mean 2015 %.2f, 2016 %.2f, 2017 %.2f, 2018 %.2f, 2019 %.2f (pilot)" %
   tuple(yp[i * 12:(i + 1) * 12].mean() for i in range(5)))
pr("comparison yearly means:", [round(yc[i * 12:(i + 1) * 12].mean(), 2) for i in range(5)])
OUT["yearly"] = dict(p=[yp[i * 12:(i + 1) * 12].mean() for i in range(5)], c=[yc[i * 12:(i + 1) * 12].mean() for i in range(5)])

# regression to the mean when districts are selected for a high value in one year (analytic, normal model)
sd_b, sd_w, frac = 6.0, 3.0, 20 / 250
sd_tot = np.sqrt(sd_b ** 2 + sd_w ** 2)
z = st.norm.isf(frac)
excess = st.norm.pdf(z) / frac * sd_tot
rel = sd_b ** 2 / sd_tot ** 2
pr("RTM: total SD %.3f, z cut %.3f, selected mean above overall mean %.2f, reliability %.2f, expected fall %.2f, remaining %.2f" %
   (sd_tot, z, excess, rel, (1 - rel) * excess, rel * excess))
OUT["rtm"] = dict(sd_tot=sd_tot, z=z, excess=excess, rel=rel, fall=(1 - rel) * excess)
# check by simulation
rng = np.random.default_rng(1)
falls = []
for _ in range(4000):
    tr = rng.normal(88, sd_b, 250)
    y1 = tr + rng.normal(0, sd_w, 250)
    y2 = tr + rng.normal(0, sd_w, 250)
    idx = np.argsort(y1)[-20:]
    falls.append([y1[idx].mean() - 88, y2[idx].mean() - y1[idx].mean()])
falls = np.array(falls)
pr("RTM simulation (top 20 of 250): mean excess %.2f, mean change next year %.2f" % (falls[:, 0].mean(), falls[:, 1].mean()))

# ------------------------------------------------------------------ 나. ITS (pilot series)
hdr("나. ITS on the pilot series")
ols = sm.OLS(yp, X).fit()
nw = sm.OLS(yp, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
pr(nw.summary(xname=XN))
pr("OLS (nonrobust) SE:", ols.bse.round(4), "CI:", ols.conf_int().round(3).tolist(), "p:", ols.pvalues)
pr("HAC SE:", nw.bse.round(4), "p:", nw.pvalues)
pr("params full:", nw.params)
pr("HAC CI:", nw.conf_int().round(3).tolist())
dw = durbin_watson(ols.resid)
ac = acf(ols.resid, nlags=12, fft=False)
pr("DW %.3f | resid SD %.3f | ACF lags1-12:" % (dw, np.sqrt(ols.scale)), ac[1:].round(3), "| bound 1.96/sqrt(60)=%.3f" % (1.96 / np.sqrt(T)))
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, acorr_ljungbox
bg = acorr_breusch_godfrey(ols, nlags=3, result_object=False)
pr("Breusch-Godfrey (3 lags): LM %.2f p %.4f" % (bg[0], bg[1]))
pr("Ljung-Box lag 12:", acorr_ljungbox(ols.resid, lags=[12]).round(4).values.tolist())
b = nw.params
OUT["its"] = dict(b=b.tolist(), se=nw.bse.tolist(), ci=nw.conf_int().tolist(), p=nw.pvalues.tolist(),
                  se_ols=ols.bse.tolist(), ci_ols=ols.conf_int().tolist(), p_ols=ols.pvalues.tolist(),
                  dw=float(dw), acf=ac.tolist(), resid=ols.resid.tolist(), sigma=float(np.sqrt(ols.scale)),
                  r2=float(ols.rsquared))
pr("post slope = %.4f" % (b[1] + b[3]))
tts = nw.t_test(np.array([[0, 1, 0, 1]]))
pr("post slope CI:", float(tts.effect[0]), tts.conf_int().round(3).tolist())
OUT["its"]["post_slope"] = [float(tts.effect[0]), *[float(v) for v in tts.conf_int()[0]]]
eff = {}
for k in (1, 6, 12, 24):
    tk = T0 + k
    pred = b[0] + b[1] * tk + b[2] + b[3] * k
    cf = b[0] + b[1] * tk
    tt = nw.t_test(np.array([[0, 0, 1, k]]))
    lo, hi = tt.conf_int()[0]
    eff[k] = dict(pred=pred, cf=cf, diff=pred - cf, rel=(pred - cf) / cf, lo=lo, hi=hi, se=float(tt.sd[0][0]), p=float(tt.pvalue))
    pr("k=%2d (t=%d): fitted %.2f counterfactual %.2f effect %.3f (95%% CI %.2f to %.2f; SE %.3f) relative %.1f%% | observed %.2f" %
       (k, tk, pred, cf, pred - cf, lo, hi, tt.sd[0][0], 100 * (pred - cf) / cf, yp[tk - 1]))
OUT["its_eff"] = eff
# average effect over the 24 post months = b2 + b3*12.5
tt = nw.t_test(np.array([[0, 0, 1, 12.5]]))
pr("average effect over 24 post months: %.3f (%.2f to %.2f)" % (float(tt.effect[0]), *tt.conf_int()[0]))
OUT["its_avg"] = [float(tt.effect[0]), *[float(v) for v in tt.conf_int()[0]]]
# cumulative patients: sum over post months of effect x denominator/1000
cum = sum((b[2] + b[3] * k) * PIL.n[T0 + k - 1] / 1000 for k in range(1, 25))
pr("cumulative patient-months averted over 24 months: %.0f" % cum)
OUT["its_cum"] = float(cum)
# fitted lines at ends
pr("fitted at t=1 %.2f, t=36 %.2f ; post fitted at t=37 %.2f, t=60 %.2f ; cf at t=37 %.2f t=48 %.2f t=60 %.2f" %
   (b[0] + b[1], b[0] + b[1] * 36, b[0] + b[1] * 37 + b[2] + b[3], b[0] + b[1] * 60 + b[2] + b[3] * 24,
    b[0] + b[1] * 37, b[0] + b[1] * 48, b[0] + b[1] * 60))

# sensitivity to the method
hdr("나. ITS: alternative ways to handle autocorrelation / seasonality")
alt = {}


def keep(name, params, bse, ci, extra=None):
    alt[name] = dict(level=float(params[0]), slope=float(params[1]), se_level=float(bse[0]), se_slope=float(bse[1]),
                     ci_level=[float(v) for v in ci[0]], ci_slope=[float(v) for v in ci[1]], **(extra or {}))
    pr("%-28s level %.3f (SE %.3f; %.2f to %.2f) slope %.4f (SE %.4f; %.3f to %.3f) %s" %
       (name, params[0], bse[0], ci[0][0], ci[0][1], params[1], bse[1], ci[1][0], ci[1][1], extra or ""))


keep("OLS", ols.params[2:], ols.bse[2:], ols.conf_int()[2:])
for L in (1, 3, 6, 12):
    h = sm.OLS(yp, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    keep("HAC maxlags=%d" % L, h.params[2:], h.bse[2:], h.conf_int()[2:])
hs = sm.OLS(yp, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True})
keep("HAC 3 small-sample corr", hs.params[2:], hs.bse[2:], hs.conf_int()[2:])
gls = sm.GLSAR(yp, X, rho=1).iterative_fit(maxiter=50)
rho = float(np.atleast_1d(gls.model.rho)[0])
keep("GLSAR AR(1)", gls.params[2:], gls.bse[2:], gls.conf_int()[2:], dict(rho=rho, nobs=int(gls.nobs), dw=float(durbin_watson(gls.wresid))))
pr(gls.summary(xname=XN))
pr("GLSAR rho %.4f, iterations %s, converged %s" % (rho, gls.iter, gls.converged))
OUT["glsar"] = dict(b=gls.params.tolist(), se=gls.bse.tolist(), ci=gls.conf_int().tolist(), p=gls.pvalues.tolist(), rho=rho,
                    dw=float(durbin_watson(gls.wresid)))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ar = ARIMA(yp, exog=X[:, 1:], order=(1, 0, 0), trend="c").fit()
pr(ar.summary())
keep("ARIMA(1,0,0) errors, ML", ar.params[2:4], ar.bse[2:4], ar.conf_int()[2:4], dict(rho=float(ar.params[4])))
# seasonality
s1, c1 = np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)
XF = np.column_stack([X, s1, c1])
fo = sm.OLS(yp, XF).fit()
fh = sm.OLS(yp, XF).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
keep("Fourier(1 pair)+HAC3", fh.params[2:4], fh.bse[2:4], fh.conf_int()[2:4],
     dict(dw=float(durbin_watson(fo.resid)), sigma=float(np.sqrt(fo.scale)), amp=float(np.hypot(fh.params[4], fh.params[5]))))
pr("   Fourier coefs sin %.3f cos %.3f ; F-test seasonal terms p=%.4f" % (fh.params[4], fh.params[5], float(fo.f_test(np.eye(6)[4:]).pvalue)))
MD = np.column_stack([X] + [(mon == m).astype(float) for m in range(1, 12)])
mo = sm.OLS(yp, MD).fit()
mh = sm.OLS(yp, MD).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
keep("month dummies+HAC3", mh.params[2:4], mh.bse[2:4], mh.conf_int()[2:4], dict(dw=float(durbin_watson(mo.resid)), sigma=float(np.sqrt(mo.scale))))
gf = sm.GLSAR(yp, XF, rho=1).iterative_fit(maxiter=50)
keep("Fourier + GLSAR AR(1)", gf.params[2:4], gf.bse[2:4], gf.conf_int()[2:4], dict(rho=float(np.atleast_1d(gf.model.rho)[0])))
# 12-month effect under the Fourier model
ttf = fh.t_test(np.array([[0, 0, 1, 12, 0, 0]]))
pr("Fourier model 12-month effect %.3f (%.2f to %.2f)" % (float(ttf.effect[0]), *ttf.conf_int()[0]))
alt["Fourier(1 pair)+HAC3"]["eff12"] = [float(ttf.effect[0]), *[float(v) for v in ttf.conf_int()[0]]]
ttg = gls.t_test(np.array([[0, 0, 1, 12]]))
pr("GLSAR 12-month effect %.3f (%.2f to %.2f)" % (float(ttg.effect[0]), *ttg.conf_int()[0]))
alt["GLSAR AR(1)"]["eff12"] = [float(ttg.effect[0]), *[float(v) for v in ttg.conf_int()[0]]]
tto = ols.t_test(np.array([[0, 0, 1, 12]]))
alt["OLS"]["eff12"] = [float(tto.effect[0]), *[float(v) for v in tto.conf_int()[0]]]
pr("OLS 12-month effect %.3f (%.2f to %.2f)" % tuple(alt["OLS"]["eff12"]))
# phase-in: drop the first 3 post months (t = 37..39); time_after keeps counting
keepm = ~((t > T0) & (t <= T0 + 3))
ph = sm.OLS(yp[keepm], X[keepm]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
keep("drop first 3 post months", ph.params[2:], ph.bse[2:], ph.conf_int()[2:], dict(nobs=int(ph.nobs)))
# anticipation: drop the 3 months before (t = 34..36)
keepa = ~((t > T0 - 3) & (t <= T0))
pa = sm.OLS(yp[keepa], X[keepa]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
keep("drop last 3 pre months", pa.params[2:], pa.bse[2:], pa.conf_int()[2:], dict(nobs=int(pa.nobs)))
# shorter series: only 12 before and 12 after
k12 = (t > T0 - 12) & (t <= T0 + 12)
sh = sm.OLS(yp[k12], X[k12]).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
keep("12 pre + 12 post only", sh.params[2:], sh.bse[2:], sh.conf_int()[2:], dict(nobs=int(sh.nobs)))
k6 = (t > T0 - 6) & (t <= T0 + 6)
s6 = sm.OLS(yp[k6], X[k6]).fit()
keep("6 pre + 6 post only (OLS)", s6.params[2:], s6.bse[2:], s6.conf_int()[2:], dict(nobs=int(s6.nobs)))
OUT["alt"] = alt

# counts: Poisson segmented regression with offset
hdr("나. Poisson segmented regression (counts with offset)")
po_ = sm.GLM(PIL.cnt.values, X, family=sm.families.Poisson(), offset=np.log(PIL.n.values)).fit()
poh = sm.GLM(PIL.cnt.values, X, family=sm.families.Poisson(), offset=np.log(PIL.n.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
pr(poh.summary(xname=XN))
pr("Pearson chi2/df (nonrobust fit): %.2f" % (po_.pearson_chi2 / po_.df_resid))
pr("nonrobust SE:", po_.bse.round(5), " HAC SE:", poh.bse.round(5))
pr("rate ratios: exp(b) =", np.exp(poh.params).round(4), " CI:", np.exp(poh.conf_int()).round(4).tolist())
pr("baseline rate per 1,000: %.2f ; monthly trend %.2f%% ; level RR %.3f ; slope-change RR per month %.4f" %
   (1000 * np.exp(poh.params[0]), 100 * (np.exp(poh.params[1]) - 1), np.exp(poh.params[2]), np.exp(poh.params[3])))
ttp = poh.t_test(np.array([[0, 0, 1, 12]]))
pr("12-month rate ratio %.3f (%.3f to %.3f)" % (np.exp(float(ttp.effect[0])), *np.exp(ttp.conf_int()[0])))
OUT["pois"] = dict(b=poh.params.tolist(), se=poh.bse.tolist(), rr=np.exp(poh.params).tolist(), rr_ci=np.exp(poh.conf_int()).tolist(),
                   disp=float(po_.pearson_chi2 / po_.df_resid),
                   rr12=[float(np.exp(ttp.effect[0])), *[float(v) for v in np.exp(ttp.conf_int()[0])]])

# ------------------------------------------------------------------ 다. controlled ITS
hdr("다. controlled ITS")
cres = {}
for nm, y in (("pilot", yp), ("comp", yc), ("diff", yd)):
    o = sm.OLS(y, X).fit()
    h = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    e12 = h.t_test(np.array([[0, 0, 1, 12]]))
    e24 = h.t_test(np.array([[0, 0, 1, 24]]))
    eav = h.t_test(np.array([[0, 0, 1, 12.5]]))
    cres[nm] = dict(b=h.params.tolist(), se=h.bse.tolist(), ci=h.conf_int().tolist(), p=h.pvalues.tolist(), dw=float(durbin_watson(o.resid)),
                    se_ols=o.bse.tolist(), sigma=float(np.sqrt(o.scale)),
                    e12=[float(e12.effect[0]), *[float(v) for v in e12.conf_int()[0]]],
                    e24=[float(e24.effect[0]), *[float(v) for v in e24.conf_int()[0]]],
                    eav=[float(eav.effect[0]), *[float(v) for v in eav.conf_int()[0]]],
                    acf=acf(o.resid, nlags=12, fft=False).tolist())
    pr("%-6s b=%s" % (nm, h.params.round(4)))
    pr("       HAC SE=%s  OLS SE=%s" % (h.bse.round(4), o.bse.round(4)))
    pr("       CI=%s p=%s" % (h.conf_int().round(3).tolist(), h.pvalues.round(5)))
    pr("       DW %.3f resid SD %.3f | 12-month effect %.3f (%.2f to %.2f) | 24-month %.3f (%.2f to %.2f) | avg %.3f (%.2f to %.2f)" %
       (durbin_watson(o.resid), np.sqrt(o.scale), *cres[nm]["e12"], *cres[nm]["e24"], *cres[nm]["eav"]))
    pr("       ACF1-3:", acf(o.resid, nlags=3, fft=False)[1:].round(3))
OUT["cits"] = cres
bd = np.array(cres["diff"]["b"])
pr("difference series: counterfactual gap at t=48: %.2f ; fitted gap %.2f" % (bd[0] + bd[1] * 48, bd[0] + bd[1] * 48 + bd[2] + bd[3] * 12))
pr("correlation of pilot and comparison OLS residuals: %.3f" %
   np.corrcoef(sm.OLS(yp, X).fit().resid, sm.OLS(yc, X).fit().resid)[0, 1])
# stacked model with interactions (same point estimates as the difference series)
stk = pd.DataFrame(dict(rate=np.r_[yp, yc], grp=np.r_[np.ones(T), np.zeros(T)], time=np.r_[t, t], post=np.r_[post, post],
                        taft=np.r_[taft, taft], month=np.r_[t, t]))
sm_ = smf.ols("rate ~ time + post + taft + grp + grp:time + grp:post + grp:taft", data=stk).fit(
    cov_type="cluster", cov_kwds={"groups": stk["month"]})
pr(sm_.summary())
sm_o = smf.ols("rate ~ time + post + taft + grp + grp:time + grp:post + grp:taft", data=stk).fit()
pr("stacked plain OLS SE of grp:post %.4f grp:taft %.4f" % (sm_o.bse["grp:post"], sm_o.bse["grp:taft"]))
OUT["stack"] = dict(params=sm_.params.to_dict(), se_cluster_month=sm_.bse.to_dict(), se_ols=sm_o.bse.to_dict())
# pre-period: are the two series parallel? (coefficient of time in the difference series)
pr("pre-trend difference (diff series 'time' coef): %.4f (%.3f to %.3f) p=%.3f" %
   (cres["diff"]["b"][1], *cres["diff"]["ci"][1], cres["diff"]["p"][1]))

# ------------------------------------------------------------------ 라. DID
hdr("라. difference-in-differences")
m = dict(tp=yp[:T0].mean(), tpost=yp[T0:].mean(), cp=yc[:T0].mean(), cpost=yc[T0:].mean())
did = (m["tpost"] - m["tp"]) - (m["cpost"] - m["cp"])
pr("2x2 means (monthly pooled rates): pilot pre %.3f post %.3f change %.3f | comp pre %.3f post %.3f change %.3f | DID %.3f" %
   (m["tp"], m["tpost"], m["tpost"] - m["tp"], m["cp"], m["cpost"], m["cpost"] - m["cp"], did))
pr("pre difference %.3f, post difference %.3f" % (m["tp"] - m["cp"], m["tpost"] - m["cpost"]))
OUT["did"] = dict(m, did=did)
# regression on the two aggregate series (120 rows)
ar2 = smf.ols("rate ~ grp * post", data=stk).fit()
pr("aggregate regression (120 rows):", ar2.params.round(3).to_dict(), "SE(interaction) %.3f" % ar2.bse["grp:post"])
# district-month panel, weighted by district size, SE clustered by district
w = d.groupby("unit").n.transform("mean")
d["w"] = w
pan_o = smf.wls("rate ~ treat * post", data=d, weights=d["w"]).fit()
pan_c = smf.wls("rate ~ treat * post", data=d, weights=d["w"]).fit(cov_type="cluster", cov_kwds={"groups": d["unit"]})
pr(pan_c.summary())
pr("panel WLS params:", pan_c.params.round(4).to_dict())
pr("naive SE %.4f (CI %.2f to %.2f) | cluster SE %.4f (CI %.2f to %.2f) ratio %.2f | p cluster %.2e" %
   (pan_o.bse["treat:post"], *pan_o.conf_int().loc["treat:post"], pan_c.bse["treat:post"], *pan_c.conf_int().loc["treat:post"],
    pan_c.bse["treat:post"] / pan_o.bse["treat:post"], pan_c.pvalues["treat:post"]))
OUT["did_reg"] = dict(params=pan_c.params.to_dict(), se_cluster=pan_c.bse.to_dict(), ci_cluster=pan_c.conf_int().values.tolist(),
                      se_naive=pan_o.bse.to_dict(), ci_naive=pan_o.conf_int().values.tolist(), p=pan_c.pvalues.to_dict(),
                      nobs=int(pan_c.nobs), nclu=int(d.unit.nunique()))
# unweighted version
un = smf.ols("rate ~ treat * post", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["unit"]})
pr("unweighted panel DID %.3f (cluster SE %.3f; %.2f to %.2f)" % (un.params["treat:post"], un.bse["treat:post"], *un.conf_int().loc["treat:post"]))
OUT["did_unw"] = [float(un.params["treat:post"]), *[float(v) for v in un.conf_int().loc["treat:post"]]]
# two-way fixed effects (district and month dummies)
fe = smf.wls("rate ~ treat:post + C(unit) + C(t)", data=d, weights=d["w"]).fit(cov_type="cluster", cov_kwds={"groups": d["unit"]})
pr("TWFE DID %.4f (cluster SE %.4f; %.2f to %.2f)" % (fe.params["treat:post"], fe.bse["treat:post"], *fe.conf_int().loc["treat:post"]))
fe_n = smf.wls("rate ~ treat:post + C(unit) + C(t)", data=d, weights=d["w"]).fit()
pr("TWFE with ordinary (non-clustered) SE %.4f (%.2f to %.2f); ratio cluster/ordinary %.2f" %
   (fe_n.bse["treat:post"], *fe_n.conf_int().loc["treat:post"], fe.bse["treat:post"] / fe_n.bse["treat:post"]))
OUT["did_twfe_naive"] = [float(fe_n.bse["treat:post"]), *[float(v) for v in fe_n.conf_int().loc["treat:post"]]]
OUT["did_twfe"] = [float(fe.params["treat:post"]), float(fe.bse["treat:post"]), *[float(v) for v in fe.conf_int().loc["treat:post"]]]
# collapse to one change per district, then compare changes (weighted)
chg = d.groupby(["unit", "treat", "post"]).rate.mean().unstack("post").reset_index()
chg["chg"] = chg[1] - chg[0]
chg["w"] = d.groupby("unit").w.first().values
cw = smf.wls("chg ~ treat", data=chg, weights=chg["w"]).fit(cov_type="HC1")
pr("district-level changes: pilot mean (unweighted) %.2f SD %.2f, comparison %.2f SD %.2f ; weighted difference %.3f (HC1 SE %.3f; %.2f to %.2f)" %
   (chg[chg.treat == 1].chg.mean(), chg[chg.treat == 1].chg.std(), chg[chg.treat == 0].chg.mean(), chg[chg.treat == 0].chg.std(),
    cw.params["treat"], cw.bse["treat"], *cw.conf_int().loc["treat"]))
pr("   range of district changes pilot %.1f to %.1f ; comparison %.1f to %.1f" %
   (chg[chg.treat == 1].chg.min(), chg[chg.treat == 1].chg.max(), chg[chg.treat == 0].chg.min(), chg[chg.treat == 0].chg.max()))
OUT["did_chg"] = dict(pil=chg[chg.treat == 1].chg.tolist(), com=chg[chg.treat == 0].chg.tolist(),
                      est=[float(cw.params["treat"]), *[float(v) for v in cw.conf_int().loc["treat"]]])
# pre-trend check: differential linear trend in the 36 pre months
dpre = d[d.t <= T0]
ptr = smf.wls("rate ~ treat * t", data=dpre, weights=dpre["w"]).fit(cov_type="cluster", cov_kwds={"groups": dpre["unit"]})
pr("pre-period differential trend treat:t = %.4f (SE %.4f; %.3f to %.3f) p=%.3f" %
   (ptr.params["treat:t"], ptr.bse["treat:t"], *ptr.conf_int().loc["treat:t"], ptr.pvalues["treat:t"]))
lo, hi = ptr.conf_int().loc["treat:t"]
pr("   x30 months (distance between period midpoints): %.2f (%.2f to %.2f)" % (ptr.params["treat:t"] * 30, lo * 30, hi * 30))
OUT["pretrend"] = dict(b=float(ptr.params["treat:t"]), se=float(ptr.bse["treat:t"]), lo=float(lo), hi=float(hi), p=float(ptr.pvalues["treat:t"]))
# event study in half-year bins (reference: the last half-year before the policy)
d["half"] = (d.t - 1) // 6 - 6     # -6 .. -1 pre, 0 .. 3 post
names = []
for h in range(-6, 4):
    if h == -1:
        continue
    nm = ("lead%d" % (-h)) if h < 0 else ("lag%d" % (h + 1))
    d[nm] = ((d.half == h) & (d.treat == 1)).astype(float)
    names.append(nm)
es = smf.wls("rate ~ " + " + ".join(names) + " + C(unit) + C(t)", data=d, weights=d["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": d["unit"]})
ev = []
for h in range(-6, 4):
    nm = ("lead%d" % (-h)) if h < 0 else ("lag%d" % (h + 1))
    if h == -1:
        ev.append(dict(h=h, b=0.0, lo=0.0, hi=0.0, ref=True))
        pr("half %2d  reference" % h)
        continue
    lo, hi = es.conf_int().loc[nm]
    ev.append(dict(h=h, b=float(es.params[nm]), lo=float(lo), hi=float(hi), p=float(es.pvalues[nm])))
    pr("half %2d %-6s %.3f (%.2f to %.2f) p=%.3f" % (h, nm, es.params[nm], lo, hi, es.pvalues[nm]))
OUT["event"] = ev
leads = [n for n in names if n.startswith("lead")]
jt = es.wald_test(" = ".join(leads) + " = 0", scalar=True)
pr("joint test of the 5 leads: chi2 %.2f p=%.3f" % (float(jt.statistic), float(jt.pvalue)))
OUT["event_joint"] = [float(jt.statistic), float(jt.pvalue)]
pr("mean of the 4 lag coefficients %.3f" % np.mean([e["b"] for e in ev if e["h"] >= 0]))
# half-year means by group (for the figure)
hm = d.groupby(["treat", "half"]).apply(lambda g: np.average(g.rate, weights=g.w), include_groups=False).unstack("half")
pr("half-year means:\n", hm.round(2))
OUT["half_means"] = dict(p=hm.loc[1].tolist(), c=hm.loc[0].tolist())
# CITS average over post period vs DID
pr("CITS average effect over 24 post months %.3f vs DID %.3f" % (cres["diff"]["eav"][0], did))

# ------------------------------------------------------------------ practice-question numbers
hdr("practice")
# Q (나): level -3.2, slope change -0.15 per month, pre-trend +0.10, baseline 40 at time 0, policy after month 24
b0, b1, b2, b3 = 40.0, 0.10, -3.2, -0.15
for k in (6, 12, 18):
    cf = b0 + b1 * (24 + k)
    e = b2 + b3 * k
    pr("practice ITS k=%d: counterfactual %.2f, effect %.2f, fitted %.2f, relative %.1f%%; post slope %.2f" % (k, cf, e, cf + e, 100 * e / cf, b1 + b3))
# Q (다): pilot level -6.0 slope -0.30 ; control level -2.5 slope -0.10
pr("practice CITS: level diff %.1f, slope diff %.2f, 12-month %.1f" % (-6.0 + 2.5, -0.30 + 0.10, (-6.0 + 2.5) + 12 * (-0.30 + 0.10)))
# Q (라): 2x2
a_, b_, c_, d_ = 32.5, 26.1, 30.2, 28.6
pr("practice DID: treated change %.1f, control change %.1f, DID %.1f, pre diff %.1f, post diff %.1f" %
   (b_ - a_, d_ - c_, (b_ - a_) - (d_ - c_), a_ - c_, b_ - d_))
# Q (가): monthly decline 0.4, 24 pre and 24 post months, no policy effect -> before-after difference
pr("practice before-after: trend -0.4/month over a 24-month midpoint distance = %.1f" % (-0.4 * 24))
# Q (라): pre-trend CI x months
pr("practice pretrend: 0.05 x 18 = %.2f, upper 0.12 x 18 = %.2f" % (0.05 * 18, 0.12 * 18))

with open(os.path.join(HERE, "_ch18_nums.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
pr("\nsaved _ch18_nums.json")

# ------------------------------------------------------------------ python output boxes (text shown in the chapter)
hdr("python output boxes")
its = pd.DataFrame(dict(rate=yp, time=t, post=post, time_after=taft))
pr(">>> its.iloc[[0, 1, 35, 36, 37, 59]]")
pr(its.iloc[[0, 1, 35, 36, 37, 59]].round(2))
fit0 = smf.ols("rate ~ time + post + time_after", data=its).fit()
pr(">>> durbin_watson(fit0.resid):", durbin_watson(fit0.resid))
fit = smf.ols("rate ~ time + post + time_after", data=its).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
pr(fit.summary())
tt = fit.t_test("post + 12 * time_after = 0")
pr(tt)
pr(">>> fit0.bse"); pr(fit0.bse.round(4))
Xg = sm.add_constant(its[["time", "post", "time_after"]])
g = sm.GLSAR(its["rate"], Xg, rho=1).iterative_fit(maxiter=50)
pr(">>> g.model.rho, g.params, g.bse")
pr(g.model.rho.round(4)); pr(g.params.round(4)); pr(g.bse.round(4))
pan = d[["unit", "treat", "t", "post", "rate", "w"]]
didfit = smf.wls("rate ~ treat * post", data=pan, weights=pan["w"]).fit(cov_type="cluster", cov_kwds={"groups": pan["unit"]})
pr(didfit.summary().tables[1])
pr("nobs", int(didfit.nobs), "clusters", pan.unit.nunique())
pr(">>> print(g.model.rho.round(4), g.params.round(4).values, g.bse.round(4).values)")
pr(g.model.rho.round(4), g.params.round(4).values, g.bse.round(4).values)
# checks for statements in the text
ttph = ph.t_test(np.array([[0, 0, 1, 12]]))
pr("drop-first-3 model: 12-month effect %.3f (%.2f to %.2f)" % (float(ttph.effect[0]), *ttph.conf_int()[0]))
both = stk.rename(columns={"grp": "group", "taft": "time_after"})
chk = smf.ols("rate ~ (time + post + time_after) * group", data=both).fit()
pr("formula '(time + post + time_after) * group':", chk.params.round(4).to_dict())
pr("CI width ratio HAC/OLS for level change: %.3f" % ((nw.conf_int()[2][1] - nw.conf_int()[2][0]) / (ols.conf_int()[2][1] - ols.conf_int()[2][0])))
pr("Cov(b2,b3) HAC: %.5f" % nw.cov_params()[2, 3])
