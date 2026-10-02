"""Numbers for chapter 17 (경쟁위험 분석).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch17.py
All numbers quoted in content/ch17.html come from this script.
Writes gen/_ch17_cr.csv (running example), gen/_ch17_fx.csv (practice example), gen/_ch17_nums.json.
The running example is the same simulated cohort as the old chapter 14 section 라 (seed 13)."""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from lifelines import CoxPHFitter, KaplanMeierFitter, AalenJohansenFitter, CoxTimeVaryingFitter
from lifelines.statistics import logrank_test

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
Z = st.norm.ppf(0.975)


def pr(*a):
    print(*a)


# =====================================================================================
# 1. tiny 10-patient example (years; 1 = dialysis, 2 = death before dialysis, 0 = censored)
# =====================================================================================
pr("=" * 70, "\n1. tiny example")
tiny = [(0.8, 1), (1.2, 2), (1.5, 2), (2.0, 1), (2.6, 2), (3.1, 1), (3.5, 2), (4.0, 0), (4.4, 1), (5.0, 0)]


def km_cause(data, cause):
    """1 - KM treating other events as censoring (distinct times)"""
    S, n, rows = 1.0, len(data), []
    for t, e in sorted(data):
        if e == cause:
            S *= (1 - 1 / n)
            rows.append((t, n, S))
        n -= 1
    return rows


def aj(data):
    """Aalen-Johansen with distinct times: rows (t, n, event, S_before, cif1, cif2, S_after)"""
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
kmm = km_cause(tiny, 2)
OUT["tiny_1mkm_death"] = 1 - kmm[-1][2]
pr("5y: 1-KM dialysis=%.4f  CIF dialysis=%.4f  CIF death=%.4f  event-free=%.4f ; 1-KM death (dialysis censored)=%.4f" %
   (OUT["tiny_1mkm"], OUT["tiny_cif_d"], OUT["tiny_cif_m"], OUT["tiny_S"], OUT["tiny_1mkm_death"]))
pr("1-KM dialysis + 1-KM death = %.4f" % (OUT["tiny_1mkm"] + OUT["tiny_1mkm_death"]))
# lifelines check of the tiny example
tt = pd.DataFrame(tiny, columns=["t", "ev"])
a_ = AalenJohansenFitter(calculate_variance=False).fit(tt.t, tt.ev, event_of_interest=1)
pr("lifelines AJ tiny:", float(a_.cumulative_density_.iloc[-1, 0]))


# =====================================================================================
# 2. running example: adults aged >=75 with CKD stage 4, new users of drug A vs drug B (same as old ch14 라)
# dialysis hazard 0.10/y in both groups (no effect), death hazard 0.20/y (B) vs 0.11/y (A)
# only administrative censoring (staggered entry): potential follow-up min(5, U(3, 8)) years
# =====================================================================================
pr("=" * 70, "\n2. running example")


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


def cox1(x, tcol, ecol, var="A"):
    c = CoxPHFitter().fit(x[[tcol, ecol, var]], tcol, ecol)
    lo, hi = np.exp(c.confidence_intervals_.loc[var].values)
    return [float(c.hazard_ratios_[var]), float(lo), float(hi), float(c.summary.loc[var, "p"]),
            float(c.params_[var]), float(c.standard_errors_[var])], c


def fit_cr(d, var="A"):
    res = {}
    x = d.assign(e=(d.ev == 1).astype(int))
    res["cs_d"], m1 = cox1(x, "t", "e", var)                 # cause-specific Cox, event of interest
    x = d.assign(e=(d.ev == 2).astype(int))
    res["cs_m"], m2 = cox1(x, "t", "e", var)                 # cause-specific Cox, competing event
    # Fine-Gray with censoring-complete data: people with a competing event stay in the risk set
    # until their (known) administrative censoring time C
    x = d.assign(t2=np.where(d.ev == 2, d.C, d.t), e=(d.ev == 1).astype(int))
    res["fg"], m3 = cox1(x, "t2", "e", var)
    x = d.assign(e=(d.ev > 0).astype(int))
    res["comp"], m4 = cox1(x, "t", "e", var)                 # composite: first of either event
    return res, (m1, m2, m3, m4)


d = sim_cr(13)
d.to_csv(os.path.join(HERE, "_ch17_cr.csv"), index=False)
r, models = fit_cr(d)
names = {"cs_d": "cause-specific HR dialysis", "cs_m": "cause-specific HR death", "fg": "subdistribution HR dialysis (Fine-Gray)",
         "comp": "composite (dialysis or death) HR"}
for k, v in r.items():
    pr("%-42s HR %.3f (%.3f-%.3f) p=%.5f  beta %.4f se %.4f" % (names[k], *v))
OUT["hr"] = r

# Fine-Gray for the competing event (death before dialysis) as outcome
x = d.assign(t2=np.where(d.ev == 1, d.C, d.t), e=(d.ev == 2).astype(int))
OUT["fg_death"], _ = cox1(x, "t2", "e")
pr("subdistribution HR death: %.3f (%.3f-%.3f) p=%.2g" % tuple(OUT["fg_death"][:4]))


# IPCW Fine-Gray check (general method; censoring distribution from KM)
def fg_ipcw(d):
    kmc = KaplanMeierFitter().fit(d.t, (d.ev == 0).astype(int))
    G = lambda t: float(kmc.survival_function_at_times(np.maximum(t - 1e-9, 0)).values[0])
    evt = np.sort(d.t[d.ev == 1].values)
    Gev = np.array([G(t) for t in evt])
    rows = []
    for i, rr in enumerate(d.itertuples()):
        if rr.ev != 2:
            rows.append((i, 0.0, rr.t, int(rr.ev == 1), rr.A, 1.0))
        else:
            gi = G(rr.t)
            rows.append((i, 0.0, rr.t, 0, rr.A, 1.0))
            prev = rr.t
            for tt_, g_ in zip(evt[evt > rr.t], Gev[evt > rr.t]):
                w = g_ / gi
                if w <= 0:
                    break
                rows.append((i, prev, tt_, 0, rr.A, w))
                prev = tt_
    x = pd.DataFrame(rows, columns=["id", "start", "stop", "e", "A", "w"])
    x = x[x.stop > x.start]
    c = CoxTimeVaryingFitter().fit(x, event_col="e", start_col="start", stop_col="stop", id_col="id", weights_col="w")
    return float(np.exp(c.params_["A"])), [float(v) for v in np.exp(c.confidence_intervals_.loc["A"].values)]


if os.environ.get("CHECK_FG"):  # slow (about a minute); gave HR 1.257 (1.10-1.43)
    fgw = fg_ipcw(d)
    pr("IPCW Fine-Gray check: HR %.3f" % fgw[0], fgw[1])
    OUT["fg_ipcw"] = fgw
else:
    OUT["fg_ipcw"] = [1.257, [1.102, 1.435]]  # stored result of the CHECK_FG run (2026-10-02)

# ---- Gray's test (two groups). With censoring-complete data (every patient's potential censoring time known,
# as with purely administrative censoring) it is the log-rank test applied after keeping patients with a competing
# event in the risk set until their censoring time (Gray 1988; Fine & Gray 1999, censoring-complete data).
x = d.assign(t2=np.where(d.ev == 2, d.C, d.t), e=(d.ev == 1).astype(int))
g = logrank_test(x.t2[x.A == 1], x.t2[x.A == 0], x.e[x.A == 1], x.e[x.A == 0])
OUT["gray"] = [float(g.test_statistic), float(g.p_value)]
pr("Gray test (censoring-complete log-rank): chi2 %.3f p=%.5f" % tuple(OUT["gray"]))
x = d.assign(e=(d.ev == 1).astype(int))
g = logrank_test(x.t[x.A == 1], x.t[x.A == 0], x.e[x.A == 1], x.e[x.A == 0])
OUT["logrank_cs"] = [float(g.test_statistic), float(g.p_value)]
pr("ordinary log-rank, death censored (tests cause-specific hazards): chi2 %.3f p=%.4f" % tuple(OUT["logrank_cs"]))
x = d.assign(e=(d.ev == 2).astype(int))
g = logrank_test(x.t[x.A == 1], x.t[x.A == 0], x.e[x.A == 1], x.e[x.A == 0])
pr("log-rank for death (dialysis censored): chi2 %.2f p=%.2g" % (g.test_statistic, g.p_value))
x = d.assign(t2=np.where(d.ev == 1, d.C, d.t), e=(d.ev == 2).astype(int))
g = logrank_test(x.t2[x.A == 1], x.t2[x.A == 0], x.e[x.A == 1], x.e[x.A == 0])
OUT["gray_death"] = [float(g.test_statistic), float(g.p_value)]
pr("Gray test for death: chi2 %.2f p=%.2g" % tuple(OUT["gray_death"]))

# cross-check: Gray's score with estimated risk sets R_k(t) = Y_k(t) (1 - F1_k(t-)) / S_k(t-) (Gray 1988),
# variance from 4,000 permutations of the group labels (censoring has the same distribution in both groups)
def gray_num(t, ev, A):
    R = []
    for k in (1, 0):
        m = (A == k)
        Y = np.cumsum(m[::-1])[::-1].astype(float)
        e1 = ((ev == 1) & m).astype(float); e_any = ((ev > 0) & m).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            h = np.where(Y > 0, e_any / Y, 0.0); h1 = np.where(Y > 0, e1 / Y, 0.0)
        S = np.cumprod(1 - h); Sm = np.concatenate([[1.0], S[:-1]])
        F1m = np.concatenate([[0.0], np.cumsum(Sm * h1)[:-1]])
        R.append(np.where(Sm > 0, Y * (1 - F1m) / Sm, 0.0))
    is1 = (ev == 1)
    return float(np.sum((is1 & (A == 1)).astype(float) - np.where(is1, R[0] / (R[0] + R[1]), 0.0)))


ds = d.sort_values("t")
zg = gray_num(ds.t.values, ds.ev.values, ds.A.values)
rngp = np.random.default_rng(1)
perm = np.array([gray_num(ds.t.values, ds.ev.values, rngp.permutation(ds.A.values)) for _ in range(4000)])
chi_g = (zg / perm.std(ddof=1)) ** 2
OUT["gray_perm"] = [float(chi_g), float(st.chi2.sf(chi_g, 1)), float((np.abs(perm) >= abs(zg)).mean())]
pr("Gray score with estimated risk sets, permutation variance: chi2 %.2f p=%.5f (permutation p %.4f)" % tuple(OUT["gray_perm"]))

# ---- counts, CIFs, 1-KM by year
YEARS = [1, 2, 3, 4, 5]


def at(df, t, col=0):
    return float(df[df.index <= t].iloc[-1, col])


for gname in ("A", "B", "all"):
    s = d if gname == "all" else d[d.grp == gname]
    o = {"n": int(len(s)), "dial": int((s.ev == 1).sum()), "death": int((s.ev == 2).sum()), "cens": int((s.ev == 0).sum()),
         "py": float(s.t.sum()), "median_fu_potential": float(np.median(s.C))}
    aj1 = AalenJohansenFitter(calculate_variance=True, jitter_level=0).fit(s.t, s.ev, event_of_interest=1)
    aj2 = AalenJohansenFitter(calculate_variance=True, jitter_level=0).fit(s.t, s.ev, event_of_interest=2)
    kmd = KaplanMeierFitter().fit(s.t, (s.ev == 1).astype(int))      # death censored
    kmm_ = KaplanMeierFitter().fit(s.t, (s.ev == 2).astype(int))     # dialysis censored
    kma = KaplanMeierFitter().fit(s.t, (s.ev > 0).astype(int))       # event-free
    own = aj(list(zip(s.t.values, s.ev.values)))
    for y in YEARS:
        ci1 = sorted([at(aj1.confidence_interval_, y, 0), at(aj1.confidence_interval_, y, 1)])
        ci2 = sorted([at(aj2.confidence_interval_, y, 0), at(aj2.confidence_interval_, y, 1)])
        cs = s[(s.t >= y)]
        o[f"y{y}"] = dict(cif_d=at(aj1.cumulative_density_, y), cif_d_ci=ci1, var_d=float(aj1.variance_[aj1.variance_.index <= y].iloc[-1]),
                          cif_m=at(aj2.cumulative_density_, y), cif_m_ci=ci2, var_m=float(aj2.variance_[aj2.variance_.index <= y].iloc[-1]),
                          km_d=1 - float(kmd.survival_function_at_times(y).values[0]),
                          km_m=1 - float(kmm_.survival_function_at_times(y).values[0]),
                          free=float(kma.survival_function_at_times(y).values[0]),
                          # risk sets just before year y: cause-specific (event-free and under follow-up) and
                          # subdistribution (plus those who died before dialysis and whose censoring time is >= y)
                          rs_cs=int((s.t >= y).sum()),
                          rs_sd=int(((s.t >= y) | ((s.ev == 2) & (s.C >= y))).sum()),
                          n_dial=int(((s.ev == 1) & (s.t <= y)).sum()), n_death=int(((s.ev == 2) & (s.t <= y)).sum()),
                          n_cens=int(((s.ev == 0) & (s.t <= y)).sum()))
    o["own_aj5"] = [own[-1][4], own[-1][5], own[-1][6]]
    OUT[gname] = o
    pr(f"\ngroup {gname}: n={o['n']} dialysis={o['dial']} death={o['death']} censored={o['cens']} PY={o['py']:.0f}")
    pr("  crude proportions: dialysis %.4f death %.4f" % (o["dial"] / o["n"], o["death"] / o["n"]))
    pr("  rates per 100 PY: dialysis %.2f death %.2f" % (o["dial"] / o["py"] * 100, o["death"] / o["py"] * 100))
    for y in YEARS:
        q = o[f"y{y}"]
        pr("  y%d CIF dial %.4f (%.4f-%.4f) | CIF death %.4f (%.4f-%.4f) | event-free %.4f | sum %.4f | 1-KM dial %.4f | 1-KM death %.4f | "
           "risk set cs %d sd %d | cum n dial %d death %d cens %d" %
           (y, q["cif_d"], *q["cif_d_ci"], q["cif_m"], *q["cif_m_ci"], q["free"], q["cif_d"] + q["cif_m"] + q["free"],
            q["km_d"], q["km_m"], q["rs_cs"], q["rs_sd"], q["n_dial"], q["n_death"], q["n_cens"]))
    pr("  own AJ at 5y (dial, death, free): %.4f %.4f %.4f" % tuple(o["own_aj5"]))
    q = o["y5"]
    pr("  5y overestimation by 1-KM: %.4f / %.4f = %.3f ; 1-KM dial + 1-KM death = %.4f" %
       (q["km_d"], q["cif_d"], q["km_d"] / q["cif_d"], q["km_d"] + q["km_m"]))

# ---- 5-year risk difference and risk ratio (CIFs), delta-method CIs from the Aalen-Johansen variances
a5, b5 = OUT["A"]["y5"], OUT["B"]["y5"]
rd = a5["cif_d"] - b5["cif_d"]
se_rd = np.sqrt(a5["var_d"] + b5["var_d"])
rr = a5["cif_d"] / b5["cif_d"]
se_lrr = np.sqrt(a5["var_d"] / a5["cif_d"] ** 2 + b5["var_d"] / b5["cif_d"] ** 2)
OUT["rd5"] = [rd, rd - Z * se_rd, rd + Z * se_rd]
OUT["rr5"] = [rr, float(np.exp(np.log(rr) - Z * se_lrr)), float(np.exp(np.log(rr) + Z * se_lrr))]
pr("\n5y dialysis risk difference %.4f (%.4f to %.4f); risk ratio %.3f (%.3f-%.3f); NNH-like 1/RD = %.1f" % (*OUT["rd5"], *OUT["rr5"], 1 / rd))
rdm = a5["cif_m"] - b5["cif_m"]
se_rdm = np.sqrt(a5["var_m"] + b5["var_m"])
OUT["rd5_death"] = [rdm, rdm - Z * se_rdm, rdm + Z * se_rdm]
pr("5y death-before-dialysis risk difference %.4f (%.4f to %.4f); ratio %.3f" % (*OUT["rd5_death"], a5["cif_m"] / b5["cif_m"]))
for y in YEARS:
    pr("  y%d CIF ratio A/B dialysis %.3f" % (y, OUT["A"][f"y{y}"]["cif_d"] / OUT["B"][f"y{y}"]["cif_d"]))
# what the Fine-Gray model implies: 1 - F_A = (1 - F_B)^sHR
shr = r["fg"][0]
pred = 1 - (1 - b5["cif_d"]) ** shr
OUT["fg_pred5"] = pred
pr("Fine-Gray implied 5y CIF in A: 1-(1-%.4f)^%.3f = %.4f (observed %.4f); implied ratio %.3f" %
   (b5["cif_d"], shr, pred, a5["cif_d"], pred / b5["cif_d"]))
pr("naive: 1.25 x 24.5 = %.1f" % (1.25 * 24.5))
# the subdistribution HR is not a risk ratio: implied risk for other baseline risks
for f0, h in ((0.60, 1.25), (0.40, 1.50)):
    f1 = 1 - (1 - f0) ** h
    pr("baseline CIF %.2f, sHR %.2f -> implied CIF %.4f (ratio %.3f), naive %.3f" % (f0, h, f1, f1 / f0, f0 * h))
OUT["shr_illus"] = [1 - 0.4 ** 1.25, 1 - 0.6 ** 1.5]
# what a cause-specific HR of 1.02 would 'imply' if misread through 1-KM
pr("1-KM ratio at 5y: %.3f" % (a5["km_d"] / b5["km_d"]))

# =====================================================================================
# 3. expected values with constant hazards (no sampling error): scenarios
#    CIF_1(t) = l1/(l1+l2) * (1 - exp(-(l1+l2) t));  1-KM target = 1 - exp(-l1 t)
# =====================================================================================
pr("=" * 70, "\n3. scenarios (true values, constant hazards)")


def cif(l1, l2, t=5.0):
    return l1 / (l1 + l2) * (1 - np.exp(-(l1 + l2) * t))


def big_fg(l1a, l2a, l1b, l2b, n=150000, seed=1):
    """large-sample Fine-Gray HR (censoring-complete, everyone censored at 5 y)"""
    rng = np.random.default_rng(seed)
    rows = []
    for A, l1, l2 in ((1, l1a, l2a), (0, l1b, l2b)):
        t1 = rng.exponential(1 / l1, n); t2 = rng.exponential(1 / l2, n)
        t = np.minimum(np.minimum(t1, t2), 5.0)
        ev = np.where(t == 5.0, 0, np.where(t1 < t2, 1, 2))
        rows.append(pd.DataFrame({"t2": np.where(ev == 2, 5.0, t), "e": (ev == 1).astype(int), "A": A}))
    x = pd.concat(rows)
    c = CoxPHFitter().fit(x, "t2", "e")
    return float(c.hazard_ratios_["A"])


SCEN = [  # label, dialysis hazard A, death hazard A (B: 0.10, 0.20)
    ("S1 death only (running example)", 0.10, 0.11),
    ("S2 no effect at all", 0.10, 0.20),
    ("S3 lowers dialysis rate only (csHR 0.80)", 0.08, 0.20),
    ("S4 lowers both (0.80 and 0.55)", 0.08, 0.11),
    ("S5 raises death only (csHR 1.50)", 0.10, 0.30),
]
OUT["scen"] = []
for lab, l1a, l2a in SCEN:
    fa, fb = cif(l1a, l2a), cif(0.10, 0.20)
    ma, mb = cif(l2a, l1a), cif(0.20, 0.10)
    sh = big_fg(l1a, l2a, 0.10, 0.20) if not os.environ.get("FAST") else float("nan")
    OUT["scen"].append(dict(label=lab, cs_d=l1a / 0.10, cs_m=l2a / 0.20, cifA=fa, cifB=fb, ratio=fa / fb, deathA=ma, deathB=mb,
                            kmA=1 - np.exp(-l1a * 5), kmB=1 - np.exp(-0.5), shr=sh))
    pr("%-42s csHR dial %.2f death %.2f | 5y CIF dial A %.4f B %.4f ratio %.3f diff %.4f | death A %.4f B %.4f | 1-KM A %.4f B %.4f | large-sample sHR %.5f" %
       (lab, l1a / 0.10, l2a / 0.20, fa, fb, fa / fb, fa - fb, ma, mb, 1 - np.exp(-l1a * 5), 1 - np.exp(-0.5), sh))

# =====================================================================================
# 4. practice example (section 나): hip fracture in older adults with dementia, drug C vs drug D
#    fracture hazard 0.03/y in both groups (no effect); death hazard 0.24/y (C) vs 0.15/y (D); admin censoring as above
# =====================================================================================
pr("=" * 70, "\n4. practice example: hip fracture, drug C vs D")


def sim_fx(seed, n1=2500, n0=2500):
    rng = np.random.default_rng(seed)
    rows = []
    for g, n, lm in (("C", n1, 0.24), ("D", n0, 0.15)):
        td = rng.exponential(1 / 0.03, n)
        tm = rng.exponential(1 / lm, n)
        C = np.minimum(5.0, rng.uniform(3, 8, n))
        t = np.minimum(np.minimum(td, tm), C)
        ev = np.where(t == C, 0, np.where(td < tm, 1, 2))
        rows.append(pd.DataFrame({"grp": g, "t": t, "ev": ev, "C": C}))
    x = pd.concat(rows, ignore_index=True)
    x["A"] = (x.grp == "C").astype(int)
    return x


FX_SEED = int(os.environ.get("FX_SEED", "25"))
if FX_SEED == 0:  # search: cause-specific HR for fracture closest to 1
    best = None
    for sd in range(1, 31):
        rr_, _ = fit_cr(sim_fx(sd))
        print("  fx seed", sd, [round(rr_[k][0], 3) for k in ("cs_d", "cs_m", "fg")], round(rr_["fg"][2], 3))
        sc = abs(np.log(rr_["cs_d"][0]))
        if rr_["fg"][2] < 1 and (best is None or sc < best[0]):
            best = (sc, sd)
    FX_SEED = best[1]
fx = sim_fx(FX_SEED)
fx.to_csv(os.path.join(HERE, "_ch17_fx.csv"), index=False)
rfx, _ = fit_cr(fx)
OUT["fx_seed"] = FX_SEED
OUT["fx_hr"] = rfx
for k, v in rfx.items():
    pr("%-42s HR %.3f (%.3f-%.3f) p=%.5f" % (names[k].replace("dialysis", "fracture"), *v[:4]))
OUT["fx"] = {}
for gname in ("C", "D"):
    s = fx[fx.grp == gname]
    aj1 = AalenJohansenFitter(calculate_variance=True, jitter_level=0).fit(s.t, s.ev, event_of_interest=1)
    aj2 = AalenJohansenFitter(calculate_variance=False, jitter_level=0).fit(s.t, s.ev, event_of_interest=2)
    kmd = KaplanMeierFitter().fit(s.t, (s.ev == 1).astype(int))
    ci = sorted([at(aj1.confidence_interval_, 5, 0), at(aj1.confidence_interval_, 5, 1)])
    OUT["fx"][gname] = dict(n=int(len(s)), fx=int((s.ev == 1).sum()), death=int((s.ev == 2).sum()), cens=int((s.ev == 0).sum()),
                            cif5=at(aj1.cumulative_density_, 5), ci5=ci, death5=at(aj2.cumulative_density_, 5),
                            km5=1 - float(kmd.survival_function_at_times(5.0).values[0]))
    q = OUT["fx"][gname]
    pr(f"  {gname}: n={q['n']} fractures={q['fx']} deaths={q['death']} cens={q['cens']} | 5y CIF fracture {q['cif5']:.4f} "
       f"({ci[0]:.4f}-{ci[1]:.4f}) death {q['death5']:.4f} | 1-KM fracture {q['km5']:.4f}")

# =====================================================================================
# 5. practice numbers for section 가: 8 patients (months; 1 = hip fracture, 2 = death before fracture, 0 = censored)
# =====================================================================================
pr("=" * 70, "\n5. practice: 8 patients")
tiny2 = [(1, 1), (2, 2), (3, 2), (4, 1), (5, 2), (6, 1), (7, 0), (8, 0)]
k2 = km_cause(tiny2, 1)
a2 = aj(tiny2)
for r_ in a2:
    pr("t=%d n=%d e=%d S-=%.4f  CIF_fx=%.4f CIF_death=%.4f S=%.4f" % r_)
pr("KM steps (death censored):", [(t, n, round(S, 4)) for t, n, S in k2])
OUT["prac"] = dict(km=1 - k2[-1][2], cif=a2[-1][4], cifm=a2[-1][5], free=a2[-1][6])
pr("1-KM %.4f  CIF fracture %.4f  CIF death %.4f  event-free %.4f" % (OUT["prac"]["km"], OUT["prac"]["cif"], OUT["prac"]["cifm"], OUT["prac"]["free"]))

# =====================================================================================
# 6. Python output shown in the folded boxes (run exactly as printed in the chapter)
# =====================================================================================
pr("=" * 70, "\n6. python output boxes")
pd.set_option("display.width", 120)
ckd = pd.read_csv(os.path.join(HERE, "_ch17_cr.csv"))
b = ckd[ckd["grp"] == "B"]
pr(">>> print(b['ev'].value_counts().sort_index())")
pr(b["ev"].value_counts().sort_index())
ajf = AalenJohansenFitter()
ajf.fit(b["t"], b["ev"], event_of_interest=1)
pr(">>> print(ajf.predict([1, 3, 5]))")
pr(ajf.predict([1, 3, 5]))
pr(">>> print(ajf.cumulative_density_.loc[:5].tail(1))")
pr(ajf.cumulative_density_.loc[:5].tail(1))
pr(">>> print(ajf.confidence_interval_.loc[:5].tail(1))")
pr(ajf.confidence_interval_.loc[:5].tail(1))
kmf = KaplanMeierFitter().fit(b["t"], b["ev"] == 1)
pr(">>> print(1 - kmf.predict(5))")
pr(1 - kmf.predict(5))
aj2_ = AalenJohansenFitter().fit(b["t"], b["ev"], event_of_interest=2)
pr(">>> death", aj2_.predict(5))
# statsmodels alternative that handles tied times without jittering (mentioned in the folded output box)
from statsmodels.duration.survfunc import CumIncidenceRight
cir = CumIncidenceRight(b["t"].values, b["ev"].values)
i5 = int(np.searchsorted(cir.times, 5, side="right") - 1)
OUT["sm_cif5_B"] = [float(cir.cinc[0][i5]), float(cir.cinc[1][i5])]
pr("statsmodels CumIncidenceRight, group B at 5 y: dialysis %.4f death %.4f" % tuple(OUT["sm_cif5_B"]))
pr("-- section 나")
ckd["dial"] = (ckd["ev"] == 1).astype(int)
cols = ["coef", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
cs = CoxPHFitter().fit(ckd[["t", "dial", "A"]], "t", "dial")
pr(">>> print(cs.summary[cols].round(3))")
pr(cs.summary[cols].round(3))
ckd["death"] = (ckd["ev"] == 2).astype(int)
csm = CoxPHFitter().fit(ckd[["t", "death", "A"]], "t", "death")
pr(csm.summary[cols].round(3))
ckd["t_sd"] = ckd["t"].where(ckd["ev"] != 2, ckd["C"])
fg = CoxPHFitter().fit(ckd[["t_sd", "dial", "A"]], "t_sd", "dial")
pr(">>> print(fg.summary[cols].round(3))")
pr(fg.summary[cols].round(3))
pr("p values:", cs.summary.loc["A", "p"], fg.summary.loc["A", "p"])

with open(os.path.join(HERE, "_ch17_nums.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
pr("\nwritten", os.path.join(HERE, "_ch17_nums.json"))
