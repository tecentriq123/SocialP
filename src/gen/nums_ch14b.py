"""Numbers for chapter 14 sections 라–사 (part b).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch14b.py
All numbers quoted in content/_ch14/sb.html come from this script."""
import json, os, sys
import numpy as np
import pandas as pd
import scipy.stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}


def pr(*a):
    print(*a)


# =====================================================================================
# 라. 경쟁위험 분석
# =====================================================================================
pr("=" * 70, "\n라. competing risks")
# ---- tiny 10-patient example (years; 1 = dialysis, 2 = death before dialysis, 0 = censored)
tiny = [(0.8, 1), (1.2, 2), (1.5, 2), (2.0, 1), (2.6, 2), (3.1, 1), (3.5, 2), (4.0, 0), (4.4, 1), (5.0, 0)]


def km_cause(data, cause):
    """1 - KM treating other events as censoring"""
    S, n, rows = 1.0, len(data), []
    for t, e in sorted(data):
        if e == cause:
            S *= (1 - 1 / n)
            rows.append((t, n, S))
        n -= 1
    return rows


def aj(data):
    """Aalen-Johansen with distinct times: returns rows (t, n, event, S_before, cif1, cif2, S_after)"""
    S, c1, c2, n, rows = 1.0, 0.0, 0.0, len(data), []
    for t, e in sorted(data):
        Sb = S
        if e == 1:
            c1 += Sb / n
        if e == 2:
            c2 += Sb / n
        if e in (1, 2):
            S = Sb * (1 - 1 / n)
        rows.append((t, n, e, Sb, c1, c2, S))
        n -= 1
    return rows


kmr = km_cause(tiny, 1)
pr("KM (death censored) steps:", [(t, n, round(S, 4)) for t, n, S in kmr])
ajr = aj(tiny)
for r in ajr:
    pr("t=%.1f n=%d e=%d S-=%.3f  CIF_dial=%.3f CIF_death=%.3f S=%.3f" % r)
OUT["tiny_1mkm"] = 1 - kmr[-1][2]
OUT["tiny_cif_d"] = ajr[-1][4]
OUT["tiny_cif_m"] = ajr[-1][5]
OUT["tiny_S"] = ajr[-1][6]
pr("5y: 1-KM=%.4f  CIF dialysis=%.4f  CIF death=%.4f  event-free=%.4f" %
   (OUT["tiny_1mkm"], OUT["tiny_cif_d"], OUT["tiny_cif_m"], OUT["tiny_S"]))

# ---- simulated cohort: older adults with CKD stage 4, drug A vs drug B
# dialysis hazard 0.10/y in both groups (no effect), death hazard 0.20/y (B) vs 0.11/y (A)
# only administrative censoring (staggered entry): potential follow-up min(5, U(3, 8)) years
from lifelines import CoxPHFitter, KaplanMeierFitter, AalenJohansenFitter, CoxTimeVaryingFitter


def sim_cr(seed, nA=1380, nB=2120):
    rng = np.random.default_rng(seed)
    rows = []
    for g, n, lm in (("A", nA, 0.11), ("B", nB, 0.20)):
        td = rng.exponential(1 / 0.10, n)
        tm = rng.exponential(1 / lm, n)
        C = np.minimum(5.0, rng.uniform(3, 8, n))
        t = np.minimum(np.minimum(td, tm), C)
        ev = np.where(t == C, 0, np.where(td < tm, 1, 2))
        for i in range(n):
            rows.append((g, t[i], ev[i], C[i]))
    d = pd.DataFrame(rows, columns=["grp", "t", "ev", "C"])
    d["A"] = (d.grp == "A").astype(int)
    return d


def fit_cr(d):
    res = {}
    # cause-specific Cox for dialysis (death censored)
    x = d.assign(e=(d.ev == 1).astype(int))[["t", "e", "A"]]
    c = CoxPHFitter().fit(x, "t", "e")
    res["cs_d"] = (c.hazard_ratios_["A"], *np.exp(c.confidence_intervals_.loc["A"].values), c.summary.loc["A", "p"])
    # cause-specific Cox for death
    x = d.assign(e=(d.ev == 2).astype(int))[["t", "e", "A"]]
    c = CoxPHFitter().fit(x, "t", "e")
    res["cs_m"] = (c.hazard_ratios_["A"], *np.exp(c.confidence_intervals_.loc["A"].values), c.summary.loc["A", "p"])
    # Fine-Gray with censoring-complete data: people with a competing event stay in the risk set
    # until their (known) administrative censoring time C
    x = d.assign(t2=np.where(d.ev == 2, d.C, d.t), e=(d.ev == 1).astype(int))[["t2", "e", "A"]]
    c = CoxPHFitter().fit(x, "t2", "e")
    res["fg"] = (c.hazard_ratios_["A"], *np.exp(c.confidence_intervals_.loc["A"].values), c.summary.loc["A", "p"])
    return res


CR_SEED = int(os.environ.get("CR_SEED", "13"))  # 0 = search seeds 1-24 again (slow)
if CR_SEED == 0:
    best = None
    for seed in range(1, 25):
        d = sim_cr(seed)
        r = fit_cr(d)
        score = abs(np.log(r["cs_d"][0])) + abs(np.log(r["cs_m"][0]) - np.log(0.55)) * 0.5
        print("  seed", seed, round(r["cs_d"][0], 3), round(r["cs_m"][0], 3), [round(x, 3) for x in r["fg"][:3]])
        if r["fg"][2] > 1.0 and (best is None or score < best[0]):
            best = (score, seed)
    CR_SEED = best[1]
seed = CR_SEED
d = sim_cr(seed)
r = fit_cr(d)
pr("seed", seed)
for k, v in r.items():
    pr(k, "HR %.2f (%.2f-%.2f) p=%.4f" % v)
OUT["cr_seed"] = seed
OUT["cr"] = {k: [round(x, 4) for x in v] for k, v in r.items()}

# IPCW Fine-Gray check (general method; censoring distribution from KM, here administrative only)
def fg_ipcw(d):
    kmc = KaplanMeierFitter().fit(d.t, (d.ev == 0).astype(int))
    G = lambda t: float(kmc.survival_function_at_times(np.maximum(t - 1e-9, 0)).values[0])
    evt = np.sort(d.t[d.ev == 1].values)
    rows = []
    for i, rr in d.iterrows():
        if rr.ev != 2:
            rows.append((i, 0.0, rr.t, int(rr.ev == 1), rr.A, 1.0))
        else:
            gi = G(rr.t)
            rows.append((i, 0.0, rr.t, 0, rr.A, 1.0))
            later = evt[evt > rr.t]
            prev = rr.t
            for tt in later:
                w = G(tt) / gi
                if w <= 0:
                    break
                rows.append((i, prev, tt, 0, rr.A, w))
                prev = tt
    x = pd.DataFrame(rows, columns=["id", "start", "stop", "e", "A", "w"])
    x = x[x.stop > x.start]
    c = CoxTimeVaryingFitter().fit(x, event_col="e", start_col="start", stop_col="stop", id_col="id",
                                   weights_col="w")
    return np.exp(c.params_["A"]), np.exp(c.confidence_intervals_.loc["A"].values)


if os.environ.get("CHECK_FG"):  # slow; gave HR 1.257 (1.10-1.43) for seed 13
    fgw = fg_ipcw(d)
    pr("IPCW Fine-Gray check: HR %.3f" % fgw[0], fgw[1])

# counts
for g in ("A", "B"):
    s = d[d.grp == g]
    OUT[f"cr_n_{g}"] = int(len(s))
    OUT[f"cr_dial_{g}"] = int((s.ev == 1).sum())
    OUT[f"cr_death_{g}"] = int((s.ev == 2).sum())
    OUT[f"cr_cens_{g}"] = int((s.ev == 0).sum())
    ajf = AalenJohansenFitter(calculate_variance=True, jitter_level=0).fit(s.t, s.ev, event_of_interest=1)
    cif = ajf.cumulative_density_
    ci = ajf.confidence_interval_
    v5 = float(cif[cif.index <= 5].iloc[-1, 0])
    lo5 = float(ci[ci.index <= 5].iloc[-1, 0]); hi5 = float(ci[ci.index <= 5].iloc[-1, 1])
    ajm = AalenJohansenFitter(calculate_variance=False, jitter_level=0).fit(s.t, s.ev, event_of_interest=2)
    m5 = float(ajm.cumulative_density_[ajm.cumulative_density_.index <= 5].iloc[-1, 0])
    kmf = KaplanMeierFitter().fit(s.t, (s.ev == 1).astype(int))
    km5 = 1 - float(kmf.survival_function_at_times(5.0).values[0])
    # own AJ check
    own = aj(list(zip(s.t.values, s.ev.values)))[-1]
    OUT[f"cr_cif5_{g}"] = [v5, lo5, hi5]
    OUT[f"cr_cifm5_{g}"] = m5
    OUT[f"cr_1mkm5_{g}"] = km5
    pr(f"group {g}: n={len(s)} dialysis={OUT[f'cr_dial_{g}']} death={OUT[f'cr_death_{g}']} cens={OUT[f'cr_cens_{g}']}")
    pr(f"   5y CIF dialysis {v5:.4f} ({lo5:.4f}-{hi5:.4f}) [own AJ {own[4]:.4f}]; CIF death {m5:.4f}; 1-KM dialysis {km5:.4f}")
d.to_csv(os.path.join(HERE, "_ch14b_cr.csv"), index=False)

if os.environ.get("ONLY_CR"): sys.exit()
# =====================================================================================
# 마. 불멸시간 편향 — deterministic illustration
# =====================================================================================
pr("=" * 70, "\n마. immortal time")
# period 1: 0-6 mo, all 1000 unexposed, 500 PY, 100 deaths (rate 20/100PY)
# at 6 mo, 900 alive: 400 start drug X, 500 never
# period 2: users 200 PY 40 deaths; non-users 250 PY 50 deaths
P1_py, P1_d = 500, 100
fut_users_py1 = 200          # the 400 future users' time in period 1 (no deaths possible)
U_py2, U_d2 = 200, 40
N_py2, N_d2 = 250, 50
wrong_u = U_d2 / (fut_users_py1 + U_py2) * 100
wrong_n = (P1_d + N_d2) / ((P1_py - fut_users_py1) + N_py2) * 100
OUT["it_wrong"] = [U_d2, fut_users_py1 + U_py2, wrong_u, P1_d + N_d2, (P1_py - fut_users_py1) + N_py2, wrong_n, wrong_u / wrong_n]
pr("wrong: users %d/%d = %.2f ; non-users %d/%d = %.2f ; RR %.3f" % tuple(OUT["it_wrong"]))
tv_u = U_d2 / U_py2 * 100
tv_n = (P1_d + N_d2) / (P1_py + N_py2) * 100
OUT["it_tv"] = [U_d2, U_py2, tv_u, P1_d + N_d2, P1_py + N_py2, tv_n, tv_u / tv_n]
pr("time-varying: exposed %d/%d = %.2f ; unexposed %d/%d = %.2f ; RR %.3f" % tuple(OUT["it_tv"]))
lm_u = U_d2 / U_py2 * 100
lm_n = N_d2 / N_py2 * 100
OUT["it_lm"] = [lm_u, lm_n, lm_u / lm_n]
pr("landmark 6 mo: %.1f vs %.1f RR %.2f" % tuple(OUT["it_lm"]))
# excluding immortal time from users but not reassigning it
ex_u = U_d2 / U_py2 * 100
ex_n = (P1_d + N_d2) / ((P1_py - fut_users_py1) + N_py2) * 100
OUT["it_excl"] = [ex_u, ex_n, ex_u / ex_n]
pr("exclude immortal time (not reassigned): %.1f vs %.2f RR %.3f" % tuple(OUT["it_excl"]))

# ---- simulated cohort for the paper box (null drug)
pr("-- simulated post-MI cohort")


def sim_it(seed, n=6000):
    rng = np.random.default_rng(seed)
    tdeath = rng.weibull(0.8, n) * (1 / 0.16)  # decreasing hazard, ~13% 1-y mortality
    tinit = rng.exponential(1 / 0.7, n)          # initiation of drug X (years)
    end = np.minimum(tdeath, 1.0)
    died = (tdeath <= 1.0).astype(int)
    user = (tinit < end).astype(int)            # started during follow-up while alive
    return pd.DataFrame({"id": np.arange(n), "end": end, "died": died, "tinit": np.where(user == 1, tinit, np.nan),
                         "user": user})


IT_SEED = int(os.environ.get("IT_SEED", "2"))  # 0 = search seeds 1-10 again
best = (0, IT_SEED) if IT_SEED else None
for seed in ([] if IT_SEED else range(1, 11)):
    dd = sim_it(seed)
    c = CoxPHFitter().fit(dd[["end", "died", "user"]], "end", "died")
    rows = []
    for rr in dd.itertuples():
        if rr.user == 1:
            rows.append((rr.id, 0.0, rr.tinit, 0, 0))
            rows.append((rr.id, rr.tinit, rr.end, rr.died, 1))
        else:
            rows.append((rr.id, 0.0, rr.end, rr.died, 0))
    tv = pd.DataFrame(rows, columns=["id", "start", "stop", "died", "x"])
    tv = tv[tv.stop > tv.start]
    ct = CoxTimeVaryingFitter().fit(tv, event_col="died", start_col="start", stop_col="stop", id_col="id")
    hr_tv = float(np.exp(ct.params_["x"]))
    print("  it seed", seed, round(float(c.hazard_ratios_["user"]), 3), round(hr_tv, 3))
    if best is None or abs(np.log(hr_tv)) < best[0]:
        best = (abs(np.log(hr_tv)), seed)
seed = best[1]
dd = sim_it(seed)
c = CoxPHFitter().fit(dd[["end", "died", "user"]], "end", "died")
OUT["it_sim_seed"] = seed
OUT["it_n"] = len(dd)
OUT["it_users"] = int(dd.user.sum())
OUT["it_deaths"] = int(dd.died.sum())
OUT["it_deaths_users"] = int(dd[dd.user == 1].died.sum())
OUT["it_deaths_non"] = int(dd[dd.user == 0].died.sum())
OUT["it_hr_naive"] = [float(c.hazard_ratios_["user"]), *np.exp(c.confidence_intervals_.loc["user"].values)]
rows = []
for rr in dd.itertuples():
    if rr.user == 1:
        rows.append((rr.id, 0.0, rr.tinit, 0, 0))
        rows.append((rr.id, rr.tinit, rr.end, rr.died, 1))
    else:
        rows.append((rr.id, 0.0, rr.end, rr.died, 0))
tv = pd.DataFrame(rows, columns=["id", "start", "stop", "died", "x"])
tv = tv[tv.stop > tv.start]
ct = CoxTimeVaryingFitter().fit(tv, event_col="died", start_col="start", stop_col="stop", id_col="id")
OUT["it_hr_tv"] = [float(np.exp(ct.params_["x"])), *np.exp(ct.confidence_intervals_.loc["x"].values)]
# landmark at 3 months (0.25 y)
L = 0.25
lm = dd[dd.end > L].copy()
lm["xL"] = ((lm.user == 1) & (lm.tinit <= L)).astype(int)
lm["tL"] = lm.end - L
cl = CoxPHFitter().fit(lm[["tL", "died", "xL"]], "tL", "died")
OUT["it_hr_lm"] = [float(cl.hazard_ratios_["xL"]), *np.exp(cl.confidence_intervals_.loc["xL"].values)]
OUT["it_lm_n"] = [int(len(lm)), int(lm.xL.sum())]
# person-time and immortal time
pt_user = float(dd[dd.user == 1].end.sum())
imm = float(dd[dd.user == 1].tinit.sum())
OUT["it_pt_user"] = pt_user
OUT["it_immortal"] = imm
OUT["it_median_init_months"] = float(np.median(dd[dd.user == 1].tinit) * 12)
pr("seed", seed, "n", len(dd), "users", OUT["it_users"], "deaths", OUT["it_deaths"],
   "(users", OUT["it_deaths_users"], "non", OUT["it_deaths_non"], ")")
pr("naive HR %.2f (%.2f-%.2f)" % tuple(OUT["it_hr_naive"]))
pr("time-varying HR %.2f (%.2f-%.2f)" % tuple(OUT["it_hr_tv"]))
pr("landmark 3mo HR %.2f (%.2f-%.2f), n=%d exposed=%d" % (*OUT["it_hr_lm"], *OUT["it_lm_n"]))
pr("user PY %.1f, immortal PY %.1f (%.1f%%), median init %.1f months" %
   (pt_user, imm, imm / pt_user * 100, OUT["it_median_init_months"]))

# =====================================================================================
# 바. 메타분석
# =====================================================================================
pr("=" * 70, "\n바. meta-analysis")
# pharmacist-led intervention vs usual care, change in HbA1c at 6 months (%)
trials = [
    # name, nI, mI, sdI, nC, mC, sdC, imputed_sd
    ("Study A", 45, -0.9, 1.1, 44, -0.3, 1.2, False),
    ("Study B", 120, -0.7, 1.0, 118, -0.4, 1.0, False),
    ("Study C", 60, -1.3, 1.3, 58, -0.2, 1.2, True),
    ("Study D", 210, -0.5, 0.9, 205, -0.3, 0.9, False),
    ("Study E", 80, -0.8, 1.2, 82, -0.1, 1.1, True),
    ("Study F", 150, -0.6, 1.0, 148, -0.5, 1.1, False),
]
y = np.array([t[2] - t[5] for t in trials])
se = np.array([np.sqrt(t[3] ** 2 / t[1] + t[6] ** 2 / t[4]) for t in trials])
v = se ** 2
w = 1 / v
fe = np.sum(w * y) / np.sum(w)
fe_se = 1 / np.sqrt(np.sum(w))
Q = np.sum(w * (y - fe) ** 2)
k = len(y)
dfq = k - 1
pQ = st.chi2.sf(Q, dfq)
I2 = max(0, (Q - dfq) / Q)
tau2 = max(0, (Q - dfq) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
wr = 1 / (v + tau2)
re = np.sum(wr * y) / np.sum(wr)
re_se = 1 / np.sqrt(np.sum(wr))
z = st.norm.ppf(0.975)
tcrit = st.t.ppf(0.975, k - 2)
pi = (re - tcrit * np.sqrt(tau2 + re_se ** 2), re + tcrit * np.sqrt(tau2 + re_se ** 2))
N = sum(t[1] + t[4] for t in trials)
for i, t in enumerate(trials):
    pr("%-10s MD %.2f SE %.4f (%.2f to %.2f) wFE %.1f%% wRE %.1f%%" %
       (t[0], y[i], se[i], y[i] - z * se[i], y[i] + z * se[i], w[i] / w.sum() * 100, wr[i] / wr.sum() * 100))
pr("FE %.4f SE %.4f (%.3f to %.3f) p=%.2g" % (fe, fe_se, fe - z * fe_se, fe + z * fe_se, 2 * st.norm.sf(abs(fe / fe_se))))
pr("Q %.3f df %d p %.4f I2 %.4f tau2 %.4f tau %.4f" % (Q, dfq, pQ, I2, tau2, np.sqrt(tau2)))
pr("RE %.4f SE %.4f (%.3f to %.3f) p=%.2g" % (re, re_se, re - z * re_se, re + z * re_se, 2 * st.norm.sf(abs(re / re_se))))
pr("95%% prediction interval (t_{k-2}=%.3f): %.3f to %.3f" % (tcrit, *pi))
pr("N participants", N)
OUT["ma"] = dict(y=y.tolist(), se=se.tolist(), wfe=(w / w.sum()).tolist(), wre=(wr / wr.sum()).tolist(),
                 fe=[fe, fe_se, fe - z * fe_se, fe + z * fe_se], Q=Q, pQ=pQ, I2=I2, tau2=tau2,
                 re=[re, re_se, re - z * re_se, re + z * re_se], pi=list(pi), N=N,
                 names=[t[0] for t in trials])
# statsmodels cross-check
from statsmodels.stats.meta_analysis import combine_effects
res = combine_effects(y, v, method_re="chi2", row_names=[t[0] for t in trials])
sf = res.summary_frame()
pr(sf.round(4))
pr("statsmodels tau2 (DL):", res.tau2, " i2:", res.i2, " q:", res.q)
# worked detail for the first trial
t0 = trials[0]
pr("Study A SE = sqrt(%.2f/%d + %.2f/%d) = %.4f -> w = %.2f" % (t0[3] ** 2, t0[1], t0[6] ** 2, t0[4], se[0], w[0]))
# SD imputation illustration (link to ch03): SD_change from baseline/follow-up SD with r=0.5
sd_b, sd_f, rr_ = 1.3, 1.4, 0.5
sd_ch = np.sqrt(sd_b ** 2 + sd_f ** 2 - 2 * rr_ * sd_b * sd_f)
pr("imputed SD change with r=0.5 from SDs 1.3 and 1.4: %.3f" % sd_ch)
OUT["ma_sdimp"] = sd_ch
# log HR pooling tip: SE from CI
hr, lo, hi = 0.82, 0.70, 0.96
se_lhr = (np.log(hi) - np.log(lo)) / (2 * z)
pr("HR 0.82 (0.70-0.96): log HR %.4f SE %.4f" % (np.log(hr), se_lhr))
OUT["ma_lhr"] = [np.log(hr), se_lhr]

# funnel illustration: 20 trials with true MD -0.25 (seed chosen so that the full set is symmetric)
rng = np.random.default_rng(6)
nper = np.round(np.exp(rng.uniform(np.log(20), np.log(250), 20))).astype(int)
sef = np.sqrt(2 * 1.05 ** 2 / nper)
yf = -0.25 + rng.normal(0, 1, 20) * sef
pub = ~((yf / sef > -1.96) & (sef > 0.17))       # small trials without a significant benefit stay unpublished
wf = 1 / sef ** 2
fe_all = np.sum(wf * yf) / wf.sum()
fe_pub = np.sum(wf[pub] * yf[pub]) / wf[pub].sum()
pr("funnel: all 20 FE %.3f ; published %d FE %.3f" % (fe_all, pub.sum(), fe_pub))
# Egger regression (standard normal deviate on precision) - just for the record
for lab, m in (("all", np.ones(20, bool)), ("pub", pub)):
    X = np.column_stack([np.ones(m.sum()), 1 / sef[m]])
    b, *_ = np.linalg.lstsq(X, yf[m] / sef[m], rcond=None)
    resid = yf[m] / sef[m] - X @ b
    s2 = resid @ resid / (m.sum() - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    tval = b[0] / np.sqrt(cov[0, 0])
    pr("Egger %s: intercept %.3f p=%.3f" % (lab, b[0], 2 * st.t.sf(abs(tval), m.sum() - 2)))
OUT["funnel"] = dict(y=yf.tolist(), se=sef.tolist(), pub=pub.tolist(), fe_all=fe_all, fe_pub=fe_pub)

# =====================================================================================
# 사. 중단시계열분석과 이중차분법
# =====================================================================================
pr("=" * 70, "\n사. ITS / DID")
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

ITS_SEED = int(os.environ.get("ITS_SEED", "60"))  # 0 = search seeds again
T = 48
t = np.arange(1, T + 1)
post = (t >= 25).astype(int)
tafter = np.where(t >= 25, t - 24, 0)


def its_series(seed):
    rng = np.random.default_rng(seed)
    e = np.zeros(T)
    eps = rng.normal(0, 1.0, T)
    for i in range(T):
        e[i] = (0.3 * e[i - 1] if i else 0) + eps[i]
    return 52.0 - 0.10 * t - 4.0 * post - 0.20 * tafter + e


if ITS_SEED == 0:
    best = None
    Xs = sm.add_constant(np.column_stack([t, post, tafter]))
    for sd in range(1, 200):
        bb = sm.OLS(its_series(sd), Xs).fit().params
        sc = abs(bb[1] + 0.10) * 10 + abs(bb[2] + 4.0) + abs(bb[3] + 0.20) * 10
        if best is None or sc < best[0]:
            best = (sc, sd)
    ITS_SEED = best[1]
pr("ITS seed", ITS_SEED)
yts = its_series(ITS_SEED)
X = sm.add_constant(np.column_stack([t, post, tafter]))
ols = sm.OLS(yts, X).fit()
nw = sm.OLS(yts, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
pr(nw.summary(xname=["const", "time", "post", "time_after"]))
pr("OLS SE:", ols.bse.round(4), " DW:", round(durbin_watson(ols.resid), 3))
b = nw.params
ci = nw.conf_int()
OUT["its"] = dict(b=b.tolist(), ci=ci.tolist(), p=nw.pvalues.tolist(), se=nw.bse.tolist(), y=yts.tolist(),
                  dw=float(durbin_watson(ols.resid)))
# effect 12 months after the policy (time_after = 12, t = 36)
k12 = 12
t12 = 24 + k12
pred = b[0] + b[1] * t12 + b[2] + b[3] * k12
cf = b[0] + b[1] * t12
pr("t=36 predicted %.3f counterfactual %.3f diff %.3f rel %.4f" % (pred, cf, pred - cf, (pred - cf) / cf))
OUT["its_12"] = [pred, cf, pred - cf, (pred - cf) / cf]
tt = nw.t_test(np.array([[0, 0, 1, k12]]))
OUT["its_12ci"] = [float(tt.effect[0]), *[float(v) for v in tt.conf_int()[0]]]
pr("12-month effect with HAC CI: %.3f (%.3f to %.3f)" % tuple(OUT["its_12ci"]))
# pre-policy mean (for text)
pr("mean first 24 months %.2f, last 24 %.2f, naive diff %.2f" % (yts[:24].mean(), yts[24:].mean(), yts[24:].mean() - yts[:24].mean()))
pr("expected secular change between the two period means from pre-trend: %.2f" % (b[1] * 24))
# naive before-after comparison
OUT["its_naive"] = [yts[:24].mean(), yts[24:].mean()]

# DID: quarterly % of older outpatients with long-term benzodiazepine prescriptions
pre_slope = -0.3
tp = np.array([18.85, 18.55, 18.25, 17.95]) + np.array([0.10, -0.10, -0.05, 0.05])
cp = np.array([16.65, 16.35, 16.05, 15.75]) + np.array([-0.05, 0.10, -0.10, 0.05])
cpost = np.array([15.45, 15.15, 14.85, 14.55]) + np.array([0.05, -0.05, 0.10, -0.10])
tcf = np.array([17.65, 17.35, 17.05, 16.75])
tpost = tcf - 4.0 + np.array([-0.10, 0.05, 0.10, -0.05])
did = (tpost.mean() - tp.mean()) - (cpost.mean() - cp.mean())
pr("DID means: T pre %.2f post %.2f | C pre %.2f post %.2f | DID %.2f" %
   (tp.mean(), tpost.mean(), cp.mean(), cpost.mean(), did))
OUT["did"] = dict(tp=tp.tolist(), tpost=tpost.tolist(), cp=cp.tolist(), cpost=cpost.tolist(), tcf=tcf.tolist(),
                  means=[tp.mean(), tpost.mean(), cp.mean(), cpost.mean()], did=did)
# regression form on the 16 quarter-group means
Y = np.concatenate([tp, tpost, cp, cpost])
tr = np.array([1] * 8 + [0] * 8)
po = np.array([0] * 4 + [1] * 4 + [0] * 4 + [1] * 4)
Xd = sm.add_constant(np.column_stack([tr, po, tr * po]))
md = sm.OLS(Y, Xd).fit()
pr("DID regression coefs:", md.params.round(3))

with open(os.path.join(HERE, "_ch14b_nums.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
