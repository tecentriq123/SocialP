"""Numbers for chapter 16 (시간과 관련된 편향).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch16.py
Every number quoted in content/ch16.html (text, tables, paper boxes, practice answers) and in
figs/ch16_*.html comes from this script. Results are also written to gen/_ch16_nums.json.

Running example (hypothetical): 6,000 patients aged >= 65 discharged after acute myocardial
infarction (NHIS claims), followed for 1 year for all-cause death. Drug X is started at some time
after discharge by some patients and has NO effect on death (same simulation as the old
chapter 14 section 마: gen/nums_ch14b.py, seed 2)."""
import io, json, os, contextlib
import numpy as np
import pandas as pd
import scipy.stats as st
from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.utils import to_long_format, add_covariate_to_timeline

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
DAYS = 365.0


def pr(*a):
    print(*a)


def hrci(fit, name):
    """HR, lower, upper, p from a fitted lifelines model"""
    s = fit.summary.loc[name]
    return [float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"])]


def f3(v):
    return "%.2f (%.2f-%.2f) p=%.4f" % tuple(v)


# =====================================================================================
# 가-1. deterministic illustration (1,000 patients, rate 20/100 PY, 400 start at 6 months)
# =====================================================================================
pr("=" * 70, "\n가. deterministic illustration")
P1_py, P1_d = 500, 100       # 0-6 months: everybody unexposed, 500 PY, 100 deaths
fut_py1 = 200                # 0-6 months of the 400 future users (no deaths possible)
U_py2, U_d2 = 200, 40        # 6-12 months, users
N_py2, N_d2 = 250, 50        # 6-12 months, never users
det = {}
det["wrong"] = [U_d2, fut_py1 + U_py2, U_d2 / (fut_py1 + U_py2) * 100,
                P1_d + N_d2, P1_py - fut_py1 + N_py2, (P1_d + N_d2) / (P1_py - fut_py1 + N_py2) * 100]
det["excl"] = [U_d2, U_py2, U_d2 / U_py2 * 100,
               P1_d + N_d2, P1_py - fut_py1 + N_py2, (P1_d + N_d2) / (P1_py - fut_py1 + N_py2) * 100]
det["tv"] = [U_d2, U_py2, U_d2 / U_py2 * 100, P1_d + N_d2, P1_py + N_py2, (P1_d + N_d2) / (P1_py + N_py2) * 100]
det["lm"] = [U_d2, U_py2, U_d2 / U_py2 * 100, N_d2, N_py2, N_d2 / N_py2 * 100]
for k, v in det.items():
    v.append(v[2] / v[5])
    pr("%-6s exposed %d/%d = %.1f ; unexposed %d/%d = %.1f ; IRR %.3f" % (k, *v))
OUT["det"] = det
# crude proportions the naive reader would quote
pr("naive proportions: users 40/400 = %.1f%%, non-users 150/600 = %.1f%%" % (40 / 400 * 100, 150 / 600 * 100))

# practice (가): another small data set (600 patients, mortality 20/100 PY, 300 start the drug at 3 months)
q = dict(py1=150, d1=30, fut=75, U_py=200, U_d=40, N_py=180, N_d=36)
qw_u = q["U_d"] / (q["fut"] + q["U_py"]) * 100
qw_n = (q["d1"] + q["N_d"]) / (q["py1"] - q["fut"] + q["N_py"]) * 100
qt_u = q["U_d"] / q["U_py"] * 100
qt_n = (q["d1"] + q["N_d"]) / (q["py1"] + q["N_py"]) * 100
OUT["prac1"] = dict(q, wrong=[qw_u, qw_n, qw_u / qw_n], tv=[qt_u, qt_n, qt_u / qt_n])
pr("practice 가: wrong %d/%d = %.1f vs %d/%d = %.1f IRR %.2f ; time-varying %.1f vs %d/%d = %.1f IRR %.2f" %
   (q["U_d"], q["fut"] + q["U_py"], qw_u, q["d1"] + q["N_d"], q["py1"] - q["fut"] + q["N_py"], qw_n, qw_u / qw_n,
    qt_u, q["d1"] + q["N_d"], q["py1"] + q["N_py"], qt_n, qt_u / qt_n))


# =====================================================================================
# running cohort (null drug): identical to nums_ch14b.sim_it(seed=2)
# =====================================================================================
LAM, KSH = 1 / 0.16, 0.8      # Weibull scale (years) and shape (decreasing hazard)


def sim_it(seed=2, n=6000, theta=1.0):
    """theta = true hazard ratio of drug X after initiation (1.0 = no effect).
    The same random numbers are used for every theta."""
    rng = np.random.default_rng(seed)
    t0 = rng.weibull(KSH, n) * LAM                 # death time without the drug (years)
    tinit = rng.exponential(1 / 0.7, n)            # time of first prescription of drug X (years)
    if theta == 1.0:
        tdeath = t0
    else:
        E = (t0 / LAM) ** KSH                      # unit-exponential variate behind t0
        H_init = (tinit / LAM) ** KSH              # cumulative hazard at initiation
        tdeath = np.where(E <= H_init, t0, LAM * (H_init + (E - H_init) / theta) ** (1 / KSH))
    end = np.minimum(tdeath, 1.0)
    died = (tdeath <= 1.0).astype(int)
    user = (tinit < end).astype(int)
    return pd.DataFrame({"id": np.arange(n), "end": end, "died": died,
                         "tinit": np.where(user == 1, tinit, np.nan), "user": user})


def long_format(dd):
    """(start, stop] rows: before the first prescription x = 0, after it x = 1"""
    u, nu = dd[dd.user == 1], dd[dd.user == 0]
    tv = pd.concat([pd.DataFrame({"id": u.id, "start": 0.0, "stop": u.tinit, "died": 0, "x": 0}),
                    pd.DataFrame({"id": u.id, "start": u.tinit, "stop": u.end, "died": u.died, "x": 1}),
                    pd.DataFrame({"id": nu.id, "start": 0.0, "stop": nu.end, "died": nu.died, "x": 0})])
    return tv[tv.stop > tv.start].sort_values(["id", "start"]).reset_index(drop=True)


def fit_tv(tv, col="x"):
    return CoxTimeVaryingFitter().fit(tv[["id", "start", "stop", "died", col]], id_col="id", event_col="died",
                                      start_col="start", stop_col="stop")


def landmark(dd, L):
    lm = dd[dd.end > L].copy()
    lm["xL"] = ((lm.user == 1) & (lm.tinit <= L)).astype(int)
    lm["tL"] = lm.end - L
    c = CoxPHFitter().fit(lm[["tL", "died", "xL"]], "tL", "died")
    late = int(((lm.user == 1) & (lm.tinit > L)).sum())
    res = dict(L_months=L * 12, n=int(len(lm)), excluded=int(len(dd) - len(lm)),
               excluded_deaths=int(((dd.end <= L) & (dd.died == 1)).sum()),
               exposed=int(lm.xL.sum()), unexposed=int((lm.xL == 0).sum()), late_starters=late,
               late_pct=late / int((lm.xL == 0).sum()) * 100,
               d_exp=int(lm[lm.xL == 1].died.sum()), d_unexp=int(lm[lm.xL == 0].died.sum()),
               hr=hrci(c, "xL"))
    for g in (1, 0):
        k = KaplanMeierFitter().fit(lm[lm.xL == g].tL, lm[lm.xL == g].died)
        res["risk_%d" % g] = float(1 - k.survival_function_at_times(1.0 - L - 1e-9).values[0])
    return res, lm


def analyse(dd, label):
    R = {}
    n = len(dd)
    u, nu = dd[dd.user == 1], dd[dd.user == 0]
    R.update(n=n, users=int(len(u)), nonusers=int(len(nu)), deaths=int(dd.died.sum()),
             d_users=int(u.died.sum()), d_non=int(nu.died.sum()),
             mort_users=u.died.mean() * 100, mort_non=nu.died.mean() * 100, mort_all=dd.died.mean() * 100)
    # naive: ever-user from discharge
    c = CoxPHFitter().fit(dd[["end", "died", "user"]], "end", "died")
    R["hr_naive"] = hrci(c, "user")
    # person-time
    py_user_total = float(u.end.sum()); imm = float(u.tinit.sum())
    py_user_after = py_user_total - imm; py_non = float(nu.end.sum())
    R.update(py_user_total=py_user_total, immortal=imm, immortal_pct=imm / py_user_total * 100,
             py_user_after=py_user_after, py_non=py_non, py_total=float(dd.end.sum()),
             median_init_months=float(np.median(u.tinit) * 12),
             init_q1q3_months=[float(np.percentile(u.tinit, 25) * 12), float(np.percentile(u.tinit, 75) * 12)])
    rate = lambda d, py: d / py * 100
    R["rate_naive"] = [rate(R["d_users"], py_user_total), rate(R["d_non"], py_non)]
    R["irr_naive"] = R["rate_naive"][0] / R["rate_naive"][1]
    # excluded immortal time: users followed from first prescription, non-users from discharge
    R["rate_excl"] = [rate(R["d_users"], py_user_after), rate(R["d_non"], py_non)]
    R["irr_excl"] = R["rate_excl"][0] / R["rate_excl"][1]
    ex = pd.concat([pd.DataFrame({"t": u.end - u.tinit, "died": u.died, "user": 1}),
                    pd.DataFrame({"t": nu.end, "died": nu.died, "user": 0})])
    R["hr_excl"] = hrci(CoxPHFitter().fit(ex, "t", "died"), "user")
    # correct person-time: pre-prescription time of users counted as unexposed
    R["rate_tv"] = [rate(R["d_users"], py_user_after), rate(R["d_non"], py_non + imm)]
    R["irr_tv"] = R["rate_tv"][0] / R["rate_tv"][1]
    R["py_unexposed"] = py_non + imm
    se = np.sqrt(1 / R["d_users"] + 1 / R["d_non"])
    R["irr_tv_ci"] = [R["irr_tv"] * np.exp(-1.96 * se), R["irr_tv"] * np.exp(1.96 * se)]
    # time-varying Cox
    tv = long_format(dd)
    ct = fit_tv(tv)
    R["hr_tv"] = hrci(ct, "x")
    R["tv_rows"] = int(len(tv))
    # landmark analyses
    R["lm"] = {}
    for L, key in ((1 / 12, "1"), (0.25, "3"), (0.5, "6")):
        R["lm"][key], _ = landmark(dd, L)
    # early deaths among non-users
    R["non_deaths_3mo"] = int(((nu.died == 1) & (nu.end <= 0.25)).sum())
    R["deaths_3mo"] = int(((dd.died == 1) & (dd.end <= 0.25)).sum())
    pr("-" * 60, "\n", label)
    pr("n", n, "users", R["users"], "non-users", R["nonusers"], "deaths", R["deaths"], "(users", R["d_users"], "non", R["d_non"], ")")
    pr("1-y mortality: users %.1f%% non-users %.1f%% all %.1f%%" % (R["mort_users"], R["mort_non"], R["mort_all"]))
    pr("naive Cox HR", f3(R["hr_naive"]))
    pr("PY users total %.0f (immortal %.0f = %.1f%%, after Rx %.0f); non-users %.0f; all %.0f" %
       (py_user_total, imm, R["immortal_pct"], py_user_after, py_non, R["py_total"]))
    pr("median time to first Rx %.1f months (IQR %.1f-%.1f)" % (R["median_init_months"], *R["init_q1q3_months"]))
    pr("rates/100PY naive: %.1f vs %.1f IRR %.2f" % (*R["rate_naive"], R["irr_naive"]))
    pr("rates/100PY excluded immortal: %.1f vs %.1f IRR %.2f ; Cox (own time origin) HR %s" %
       (*R["rate_excl"], R["irr_excl"], f3(R["hr_excl"])))
    pr("rates/100PY correct person-time: %.1f vs %.1f (unexposed PY %.0f) IRR %.2f (%.2f-%.2f)" %
       (*R["rate_tv"], R["py_unexposed"], R["irr_tv"], *R["irr_tv_ci"]))
    pr("time-varying Cox HR", f3(R["hr_tv"]), "rows", R["tv_rows"])
    for key, v in R["lm"].items():
        pr("landmark %s mo: n=%d (excluded %d, deaths %d) exposed %d unexposed %d (late starters %d = %.1f%%) deaths %d/%d "
           "risk to 12 mo %.1f%% vs %.1f%% HR %s" %
           (key, v["n"], v["excluded"], v["excluded_deaths"], v["exposed"], v["unexposed"], v["late_starters"], v["late_pct"],
            v["d_exp"], v["d_unexp"], v["risk_1"] * 100, v["risk_0"] * 100, f3(v["hr"])))
    pr("deaths within 3 months: %d (non-users %d = %.1f%% of non-user deaths)" %
       (R["deaths_3mo"], R["non_deaths_3mo"], R["non_deaths_3mo"] / R["d_non"] * 100))
    return R, tv


dd = sim_it(2)
OUT["null"], tv = analyse(dd, "running cohort, drug X has no effect (seed 2)")
dd.to_csv(os.path.join(HERE, "_ch16_cohort.csv"), index=False)

# first 30/90 days: how many of the eventual users had not started yet
OUT["null"]["users_started_by_3mo"] = int((dd.tinit <= 0.25).sum())

# ---- scenario with a real effect (true HR 0.70 after initiation), same random numbers
de = sim_it(2, theta=0.70)
OUT["eff"], tv_e = analyse(de, "same cohort, drug X truly lowers the hazard (HR 0.70)")

# =====================================================================================
# 다. long format with lifelines utilities, output box, two example patients
# =====================================================================================
pr("=" * 70, "\n다. long format / CoxTimeVaryingFitter (days)")
base = pd.DataFrame({"id": dd.id, "duration": dd.end * DAYS, "died": dd.died})
long0 = to_long_format(base, duration_col="duration")
rx = pd.DataFrame({"id": dd.id[dd.user == 1], "time": dd.tinit[dd.user == 1] * DAYS, "drugx": 1})   # first prescription
long1 = add_covariate_to_timeline(long0, rx, id_col="id", duration_col="time", event_col="died")
long1["drugx"] = long1["drugx"].fillna(0).astype(int)     # rows before the first prescription and never-users
pr(long0.head(3).to_string()); pr(long1.head(4).to_string())
ctv = CoxTimeVaryingFitter().fit(long1, id_col="id", event_col="died", start_col="start", stop_col="stop")
OUT["tv_util"] = hrci(ctv, "drugx")
OUT["tv_util_rows"] = int(len(long1))
OUT["tv_util_periods"] = int(len(long1))
pr("rows", len(long1), "HR", f3(OUT["tv_util"]), " (must equal the hand-made long format)")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ctv.print_summary(decimals=3)
OUT["tv_summary_text"] = buf.getvalue()
pr(buf.getvalue())
s = ctv.summary.loc["drugx"]
OUT["tv_summary"] = {k: float(s[k]) for k in ["coef", "exp(coef)", "se(coef)", "coef lower 95%", "coef upper 95%",
                                               "exp(coef) lower 95%", "exp(coef) upper 95%", "z", "p", "-log2(p)"]}
OUT["tv_loglik"] = float(ctv.log_likelihood_)
OUT["tv_events"] = int(long1.died.sum())
OUT["tv_subjects"] = int(long1.id.nunique())
pr("log-likelihood %.2f, events %d, subjects %d, periods %d" % (OUT["tv_loglik"], OUT["tv_events"], OUT["tv_subjects"], len(long1)))

# example patients for the long-format table (days): a user who died, a user alive at 1 y, a never-user who died early
dday = dd.assign(end_d=np.round(dd.end * DAYS).astype(int), init_d=np.round(dd.tinit * DAYS))
cand_a = dday[(dday.user == 1) & (dday.died == 1) & dday.init_d.between(100, 180) & dday.end_d.between(230, 320)].head(1)
cand_b = dday[(dday.user == 0) & (dday.died == 1) & dday.end_d.between(30, 80)].head(1)
cand_c = dday[(dday.user == 1) & (dday.died == 0) & dday.init_d.between(40, 90)].head(1)
OUT["ex_patients"] = []
for lab, cnd in (("A", cand_a), ("B", cand_b), ("C", cand_c)):
    r = cnd.iloc[0]
    OUT["ex_patients"].append(dict(label=lab, id=int(r.id), init=None if np.isnan(r.init_d) else int(r.init_d),
                                   end=int(r.end_d), died=int(r.died)))
    pr("example patient", lab, OUT["ex_patients"][-1])
    pr(long1[long1.id == r.id].round(1).to_string(index=False))

# events and person-time by exposure state (table in the 다 paper box), in person-years
N0 = OUT["null"]
pr("exposure-state table: unexposed %d deaths / %.0f PY = %.1f ; exposed %d / %.0f PY = %.1f per 100 PY; crude IRR %.2f (%.2f-%.2f)" %
   (N0["d_non"], N0["py_unexposed"], N0["rate_tv"][1], N0["d_users"], N0["py_user_after"], N0["rate_tv"][0],
    N0["irr_tv"], *N0["irr_tv_ci"]))

# why the crude IRR (0.8x) differs from the time-varying Cox HR (1.03): the hazard falls with time since discharge
# and exposed person-time lies later. Rates by period of follow-up:
per = []
edges = [0, 0.25, 0.5, 1.0]
for a, b in zip(edges[:-1], edges[1:]):
    seg = tv.assign(s=np.maximum(tv.start, a), e=np.minimum(tv.stop, b))
    seg = seg[seg.e > seg.s]
    seg = seg.assign(py=seg.e - seg.s, ev=((seg.died == 1) & (seg.stop <= b) & (seg.stop > a)).astype(int))
    g = seg.groupby("x").agg(py=("py", "sum"), ev=("ev", "sum"))
    row = dict(a=a * 12, b=b * 12, py0=float(g.py[0]), ev0=int(g.ev[0]), py1=float(g.py[1]), ev1=int(g.ev[1]))
    row["r0"] = row["ev0"] / row["py0"] * 100; row["r1"] = row["ev1"] / row["py1"] * 100
    row["irr"] = row["r1"] / row["r0"]
    per.append(row)
    pr("months %2.0f-%2.0f: unexposed %d/%.0f = %.1f ; exposed %d/%.0f = %.1f ; IRR %.2f" %
       (row["a"], row["b"], row["ev0"], row["py0"], row["r0"], row["ev1"], row["py1"], row["r1"], row["irr"]))
OUT["period"] = per
# Mantel-Haenszel rate ratio across the three periods
num = sum(r["ev1"] * r["py0"] / (r["py0"] + r["py1"]) for r in per)
den = sum(r["ev0"] * r["py1"] / (r["py0"] + r["py1"]) for r in per)
OUT["irr_mh"] = num / den
pr("Mantel-Haenszel IRR across periods %.2f" % OUT["irr_mh"])

# small hand calculation for 다 practice: three patients' person-days by exposure
prac = [dict(pid="가", init=60, end=365, died=0), dict(pid="나", init=None, end=120, died=1),
        dict(pid="다", init=200, end=290, died=1)]
un = sum((p["init"] if p["init"] is not None else p["end"]) for p in prac)
ex = sum((p["end"] - p["init"]) for p in prac if p["init"] is not None)
OUT["prac3"] = dict(pat=prac, unexp_days=un, exp_days=ex,
                    naive_user_days=sum(p["end"] for p in prac if p["init"] is not None),
                    naive_non_days=sum(p["end"] for p in prac if p["init"] is None))
pr("practice 다: unexposed %d days (1 death), exposed %d days (1 death); naive users %d days, non-users %d days" %
   (un, ex, OUT["prac3"]["naive_user_days"], OUT["prac3"]["naive_non_days"]))

# =====================================================================================
# 라-1. protopathic bias: PPI started for early symptoms of a not-yet-diagnosed GI bleed (same cohort)
# =====================================================================================
pr("=" * 70, "\n라. protopathic bias, lag time")


def sim_proto(dd, seed=21, rate_b=0.06, rate_ppi=0.5, p_prod=0.35, prod_max=30):
    rng = np.random.default_rng(seed)
    n = len(dd)
    tb = rng.exponential(1 / rate_b, n) * DAYS            # GI bleeding admission (days), PPI has no effect
    end = np.minimum(dd.end.values * DAYS, tb)            # follow-up ends at bleed, death or 1 year
    bleed = (tb <= dd.end.values * DAYS).astype(int)
    t_ppi = rng.exponential(1 / rate_ppi, n) * DAYS       # ordinary initiation
    prod_len = rng.uniform(0, prod_max, n)                # days between first symptoms and admission
    gets = rng.uniform(0, 1, n) < p_prod                  # PPI prescribed for the symptoms
    when = tb - prod_len * rng.uniform(0, 1, n)           # prescription date inside the prodrome
    t_prod = np.where((bleed == 1) & gets & (when > 0), when, np.inf)
    t_ppi = np.minimum(t_ppi, t_prod)
    user = t_ppi < end
    return pd.DataFrame({"id": dd.id.values, "end": end, "bleed": bleed, "t_ppi": np.where(user, t_ppi, np.nan),
                         "user": user.astype(int), "prodromal": np.isfinite(t_prod) & (t_prod <= t_ppi) & user})


def proto_long(dp, lag):
    """exposed from (first prescription + lag days); earlier person-time and events are unexposed"""
    s0 = np.where(dp.user == 1, dp.t_ppi + lag, np.inf)
    sw = s0 < dp.end.values
    u, su, nu = dp[sw], s0[sw], dp[~sw]
    x = pd.concat([pd.DataFrame({"id": u.id, "start": 0.0, "stop": su, "bleed": 0, "ppi": 0}),
                   pd.DataFrame({"id": u.id, "start": su, "stop": u.end, "bleed": u.bleed, "ppi": 1}),
                   pd.DataFrame({"id": nu.id, "start": 0.0, "stop": nu.end, "bleed": nu.bleed, "ppi": 0})])
    return x[x.stop > x.start].sort_values(["id", "start"]).reset_index(drop=True)


dp = sim_proto(dd)
PR = dict(n=int(len(dp)), bleeds=int(dp.bleed.sum()), users=int(dp.user.sum()), prodromal=int(dp.prodromal.sum()),
          py=float(dp.end.sum() / DAYS))
u = dp[(dp.user == 1) & (dp.bleed == 1)]
gap = u.end - u.t_ppi
PR["bleeds_users"] = int(len(u))
PR["bleeds_within30"] = int((gap <= 30).sum())
PR["bleeds_within30_pct"] = float((gap <= 30).mean() * 100)
pr("n %d, bleeds %d (%.1f/100PY), PPI starters %d, prodromal prescriptions %d; bleeds among starters %d, within 30 d of start %d (%.0f%%)" %
   (PR["n"], PR["bleeds"], PR["bleeds"] / PR["py"] * 100, PR["users"], PR["prodromal"], PR["bleeds_users"],
    PR["bleeds_within30"], PR["bleeds_within30_pct"]))
PR["lag"] = {}
for lag in (0, 7, 14, 30, 60, 90):
    x = proto_long(dp, lag)
    c = CoxTimeVaryingFitter().fit(x, id_col="id", event_col="bleed", start_col="start", stop_col="stop")
    ev1 = int(x[(x.ppi == 1)].bleed.sum()); py1 = float((x[x.ppi == 1].stop - x[x.ppi == 1].start).sum() / DAYS)
    ev0 = int(x[(x.ppi == 0)].bleed.sum()); py0 = float((x[x.ppi == 0].stop - x[x.ppi == 0].start).sum() / DAYS)
    PR["lag"][str(lag)] = dict(hr=hrci(c, "ppi"), ev1=ev1, py1=py1, ev0=ev0, py0=py0)
    pr("lag %2d d: exposed %d/%.0f PY, unexposed %d/%.0f PY, HR %s" % (lag, ev1, py1, ev0, py0, f3(PR["lag"][str(lag)]["hr"])))
# the same lag through lifelines' add_covariate_to_timeline(delay=...)
b2 = pd.DataFrame({"id": dp.id, "duration": dp.end, "bleed": dp.bleed})
l0 = to_long_format(b2, duration_col="duration")
rxp = pd.DataFrame({"id": dp.id[dp.user == 1], "time": dp.t_ppi[dp.user == 1], "ppi": 1})
l30 = add_covariate_to_timeline(l0, rxp, id_col="id", duration_col="time", event_col="bleed", delay=30).fillna({"ppi": 0})
c30 = CoxTimeVaryingFitter().fit(l30, id_col="id", event_col="bleed", start_col="start", stop_col="stop")
PR["lag30_util"] = hrci(c30, "ppi")
pr("lag 30 d via add_covariate_to_timeline(delay=30): HR", f3(PR["lag30_util"]))
OUT["proto"] = PR

# =====================================================================================
# 라-2. prevalent-user bias: depletion of susceptibles (deterministic)
# =====================================================================================
pr("=" * 70, "\n라. prevalent users / depletion of susceptibles")
N_new, p_s = 1000, 0.20
r_s, r_n, r_c = 0.60, 0.05, 0.05     # event rates per person-year: susceptible on drug, others on drug, comparator
PU = dict(N=N_new, p_s=p_s, r_s=r_s * 100, r_n=r_n * 100, r_c=r_c * 100, years=[])
ns, nn = N_new * p_s, N_new * (1 - p_s)
for y in (1, 2, 3):
    ev_s = ns * (1 - np.exp(-r_s)); ev_n = nn * (1 - np.exp(-r_n))
    py_s = ev_s / r_s; py_n = ev_n / r_n
    rate = (ev_s + ev_n) / (py_s + py_n) * 100
    PU["years"].append(dict(year=y, at_start=ns + nn, susc_start=ns, susc_pct=ns / (ns + nn) * 100,
                            events=ev_s + ev_n, py=py_s + py_n, rate=rate, irr=rate / (r_c * 100)))
    pr("year %d: on drug at start %.0f (susceptible %.0f = %.1f%%), events %.1f, PY %.1f, rate %.1f/100PY, IRR vs comparator %.2f" %
       (y, ns + nn, ns, ns / (ns + nn) * 100, ev_s + ev_n, py_s + py_n, rate, rate / (r_c * 100)))
    ns -= ev_s; nn -= ev_n
OUT["prev"] = PU
# a 'prevalent-user' cohort: people who have been on the drug for >= 1 year at cohort entry, followed for 1 year (= year 2)
# and a mixed cohort: 30% new users + 70% prevalent (year-2) users, by person-time of 1-year follow-up
y1, y2 = PU["years"][0], PU["years"][1]
mix_rate = (0.3 * y1["events"] / y1["at_start"] + 0.7 * y2["events"] / y2["at_start"]) / \
           (0.3 * y1["py"] / y1["at_start"] + 0.7 * y2["py"] / y2["at_start"]) * 100
PU["mix_irr"] = mix_rate / (r_c * 100)
pr("mixed cohort (30%% new, 70%% prevalent >= 1 y): rate %.1f, IRR %.2f" % (mix_rate, PU["mix_irr"]))

# =====================================================================================
# large-sample check (n = 60,000, seed 7): what each analysis converges to. Slow (about 8 minutes),
# so it runs only with BIG=1 and the result is cached in gen/_ch16_big.json.
# =====================================================================================
BIGF = os.path.join(HERE, "_ch16_big.json")
if os.environ.get("BIG"):
    big = {}
    for th in (1.0, 0.7):
        b = sim_it(7, n=60000, theta=th)
        r = dict(naive=hrci(CoxPHFitter().fit(b[["end", "died", "user"]], "end", "died"), "user"),
                 tv=hrci(fit_tv(long_format(b)), "x"))
        for L, key in ((1 / 12, "1"), (0.25, "3"), (0.5, "6")):
            v, _ = landmark(b, L)
            r["lm" + key] = v["hr"]; r["late" + key] = v["late_pct"]
        big[str(th)] = r
    json.dump(big, open(BIGF, "w"), indent=1)
if os.path.exists(BIGF):
    OUT["big"] = json.load(open(BIGF))
    pr("=" * 70, "\nlarge-sample check (n = 60,000, seed 7)")
    for th, r in OUT["big"].items():
        pr("true HR", th, "| naive", f3(r["naive"]), "| tv", f3(r["tv"]))
        for key in ("1", "3", "6"):
            pr("   landmark %s mo: HR %s, late starters %.1f%%" % (key, f3(r["lm" + key]), r["late" + key]))

# =====================================================================================
# checks used in the text
# =====================================================================================
pr("=" * 70, "\nchecks")
# "HR is not a risk ratio": naive analysis, 1-year risks 11.7% vs 28.0%
N0 = OUT["null"]
pr("naive risk ratio %.2f vs naive HR %.2f ; 1 - HR = %.0f%%" %
   (N0["mort_users"] / N0["mort_non"], N0["hr_naive"][0], (1 - N0["hr_naive"][0]) * 100))
# z and p from HR and CI (paper-box consistency)
for lab, v in (("naive", N0["hr_naive"]), ("tv", N0["hr_tv"]), ("lm3", N0["lm"]["3"]["hr"])):
    se = (np.log(v[2]) - np.log(v[1])) / (2 * 1.96)
    pr("%s: z = %.2f, p = %.3g" % (lab, np.log(v[0]) / se, 2 * st.norm.sf(abs(np.log(v[0]) / se))))

with open(os.path.join(HERE, "_ch16_nums.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
