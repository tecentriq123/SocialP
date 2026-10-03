"""Numbers for chapter 14 (청구자료 코호트 연구의 설계).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch14.py
Every number quoted in content/ch14.html comes from this script (results also saved to gen/_ch14_nums.json
for gen/fig_ch14.py).

Running example (hypothetical): HIRA-type claims data 2015-01-01 .. 2022-12-31; new users of SGLT2 inhibitors vs
DPP-4 inhibitors (first prescription 2016-01-01 .. 2021-12-31); outcome = first hospitalization for heart failure (HHF).
This is a NEW simulated cohort made for chapter 14; it is not the 4,812 / 9,655 cohort of chapters 1, 7 and 12.
"""
import json, os, warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from statsmodels.stats.rates import confint_poisson, test_poisson_2indep
from statsmodels.stats.proportion import proportion_confint
from lifelines import CoxPHFitter, KaplanMeierFitter

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = {}
rng = np.random.default_rng(20261002)
Z = st.norm.ppf(0.975)


def pr(*a):
    print(*a)


def expit(x):
    return 1 / (1 + np.exp(-x))


def smd_bin(p1, p0):
    return (p1 - p0) / np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)


def smd_cont(x1, x0):
    return (x1.mean() - x0.mean()) / np.sqrt((x1.var(ddof=1) + x0.var(ddof=1)) / 2)


# =====================================================================================
# 가. 청구자료의 구조와 한계 — 진단 코드의 조작적 정의 (deterministic 2x2 example)
# =====================================================================================
pr("=" * 78, "\n가. operational definition of a diagnosis (hypothetical chart-review validation)")
# 2,000 people with at least one E11-E14 code in a year, charts reviewed.
# definition A: any E11-E14 code once (everyone here).  definition B: >= 2 outpatient visits or 1 admission with the
# code AND an antidiabetic prescription.
true_dm, not_dm = 1480, 520            # chart-confirmed type 2 diabetes among the 2,000
B_true, B_false = 1391, 47             # definition B positive among true / among not
ppvA = true_dm / (true_dm + not_dm)
ppvB = B_true / (B_true + B_false)
sensB = B_true / true_dm               # relative to definition A (the widest net)
ciA = proportion_confint(true_dm, true_dm + not_dm, method="wilson")
ciB = proportion_confint(B_true, B_true + B_false, method="wilson")
pr(f"  def A: n={true_dm + not_dm}, true={true_dm}, PPV={ppvA:.3f} (Wilson {ciA[0]:.3f}-{ciA[1]:.3f})")
pr(f"  def B: n={B_true + B_false}, true={B_true}, false={B_false}, PPV={ppvB:.3f} (Wilson {ciB[0]:.3f}-{ciB[1]:.3f}); "
   f"keeps {sensB:.3f} of true cases, drops {true_dm - B_true} true and {not_dm - B_false} false")
OUT["dx"] = dict(nA=true_dm + not_dm, trueA=true_dm, ppvA=ppvA, nB=B_true + B_false, trueB=B_true, falseB=B_false,
                 ppvB=ppvB, sensB=sensB, lostTrue=true_dm - B_true, lostFalse=not_dm - B_false)

# =====================================================================================
# 나. source population -> new users -> exclusions -> analysis cohort (simulated flags)
# =====================================================================================
pr("=" * 78, "\n나. attrition flow")
N0 = 300_000
cls = rng.choice(np.array(["S", "D", "B"]), size=N0, p=[0.125, 0.862, 0.013])   # first drug class in 2016-2021
u = rng.random((N0, 5))
prev = np.where(cls == "S", u[:, 0] < 0.15, np.where(cls == "D", u[:, 0] < 0.45, u[:, 0] < 0.30))
both = cls == "B"
minor = u[:, 1] < 0.0015
esrd = np.where(cls == "S", u[:, 2] < 0.004, u[:, 2] < 0.022)
phhf = np.where(cls == "S", u[:, 3] < 0.018, u[:, 3] < 0.026)

flow = [("source", N0, int((cls == "S").sum()), int((cls == "D").sum()), int(both.sum()))]
keep = np.ones(N0, bool)
steps = [("prevalent (either class in prior 365 d)", prev), ("both classes on index date", both),
         ("age < 18", minor), ("ESRD or dialysis", esrd), ("HHF in prior 365 d", phhf)]
for name, flag in steps:
    ex = keep & flag
    keep &= ~flag
    flow.append((name, int(ex.sum()), int((ex & (cls == "S")).sum()), int((ex & (cls == "D")).sum()),
                 int((ex & both).sum()), int(keep.sum())))
    pr(f"  exclude {name:42s}: {int(ex.sum()):7d}  (S {int((ex & (cls == 'S')).sum())}, D {int((ex & (cls == 'D')).sum())}, "
       f"B {int((ex & both).sum())}) -> remaining {int(keep.sum())}")
nS, nD = int((keep & (cls == "S")).sum()), int((keep & (cls == "D")).sum())
pr(f"  source {N0}: first class S {flow[0][2]}, D {flow[0][3]}, both {flow[0][4]}")
pr(f"  new users (after washout) {flow[1][5]}  = {flow[1][5] / N0:.3f} of source; prevalent {flow[1][1]} = {flow[1][1] / N0:.3f}")
pr(f"  analysis cohort {nS + nD}: SGLT2i {nS}, DPP-4i {nD}; total excluded {N0 - nS - nD}")
OUT["flow"] = dict(N0=N0, srcS=flow[0][2], srcD=flow[0][3], srcB=flow[0][4],
                   steps=[dict(name=f[0], n=f[1], S=f[2], D=f[3], B=f[4], left=f[5]) for f in flow[1:]],
                   nS=nS, nD=nD, n=nS + nD)

# ---- prevalent-user idea in a few numbers (deterministic)
early, late, comp = 40.0, 10.0, 10.0          # events per 1,000 PY: first 6 months / afterwards / comparator
new2y = (early * 0.5 + late * 1.5) / 2.0
pr(f"  prevalent-user idea: new users over 2 y -> {new2y:.1f}/1000 PY, IRR {new2y / comp:.2f}; "
   f"prevalent users (past 6 mo) -> {late:.1f}, IRR {late / comp:.2f}")
OUT["prev"] = dict(early=early, late=late, comp=comp, new2y=new2y, irr_new=new2y / comp, irr_prev=late / comp)

# =====================================================================================
# cohort simulation (analysis cohort + a non-user group for the comparator-choice illustration)
# =====================================================================================
END = 2556            # 2022-12-31 as days since 2016-01-01 (day 0)
ENROLL = 2192         # index dates 2016-01-01 .. 2021-12-31


def make_group(n, kind):
    """kind: 'S' SGLT2i new users, 'D' DPP-4i new users, 'N' non-users (metformin only / no drug)"""
    age_mu, age_sd = {"S": (57.0, 11.0), "D": (61.0, 12.5), "N": (58.0, 12.5)}[kind]
    age = np.clip(rng.normal(age_mu, age_sd, n), 18, 95)
    a = (age - 60) / 10
    female = rng.random(n) < {"S": 0.41, "D": 0.45, "N": 0.46}[kind]
    hf = rng.random(n) < expit({"S": -3.05, "D": -2.95, "N": -3.35}[kind] + 0.45 * a)   # HF diagnosis code, no admission
    ihd = rng.random(n) < expit({"S": -1.85, "D": -2.15, "N": -2.45}[kind] + 0.40 * a)
    ckd = rng.random(n) < expit({"S": -3.45, "D": -2.85, "N": -3.30}[kind] + 0.50 * a)
    htn = rng.random(n) < expit({"S": 0.55, "D": 0.55, "N": 0.20}[kind] + 0.45 * a)
    ins = rng.random(n) < {"S": 0.062, "D": 0.088, "N": 0.004}[kind]
    met = rng.random(n) < {"S": 0.86, "D": 0.79, "N": 0.62}[kind]
    # U: long-standing, poorly controlled diabetes (HbA1c, duration) - NOT recorded in claims
    sev = rng.random(n) < {"S": 0.55, "D": 0.55, "N": 0.15}[kind]
    hba1c = np.where(sev, rng.normal(8.9, 1.0, n), rng.normal(7.1, 0.6, n))
    if kind == "S":
        idx = np.floor(ENROLL * rng.beta(1.9, 1.0, n)).astype(int)
    else:
        idx = np.floor(ENROLL * rng.beta(1.1, 1.0, n)).astype(int)
    return pd.DataFrame(dict(grp=kind, age=age, female=female, hf=hf, ihd=ihd, ckd=ckd, htn=htn, ins=ins, met=met,
                             sev=sev, hba1c=hba1c, idx=idx))


def lin_hhf(d):
    return (0.0036 * np.exp(0.060 * (d.age - 60)) * np.where(d.hf, 3.5, 1) * np.where(d.ckd, 1.9, 1)
            * np.where(d.ihd, 1.5, 1) * np.where(d.ins, 1.4, 1) * np.where(d.htn, 1.25, 1)
            * np.where(d.female, 0.85, 1) * np.where(d.sev, 2.2, 1))


def lin_death(d):          # in-hospital death (the only death visible in HIRA-type claims)
    return (0.0045 * np.exp(0.085 * (d.age - 60)) * np.where(d.hf, 1.8, 1) * np.where(d.ckd, 1.8, 1)
            * np.where(d.ihd, 1.3, 1) * np.where(d.ins, 1.5, 1) * np.where(d.female, 0.75, 1) * np.where(d.sev, 1.3, 1))


HR_ON = 0.65     # true effect of being on an SGLT2 inhibitor on the HHF hazard
HR_ON_D = 0.80   # ... on in-hospital death


def piecewise_time(E, lam1, lam2, c):
    """event time (years) with hazard lam1 before c (years) and lam2 afterwards"""
    t1 = E / lam1
    return np.where(t1 < c, t1, c + (E - lam1 * c) / lam2)


def fills(n, kind):
    """prescription fills of the index drug: returns supply (days), start/run-out matrices, permanent-stop fill L"""
    supply = rng.choice(np.array([30, 60, 90]), size=n, p=[0.30, 0.25, 0.45])
    e1 = {"S": 0.12, "D": 0.09}[kind]          # stop after the first fill
    h = {"S": 0.17, "D": 0.14}[kind]           # later permanent discontinuation, per year
    early = rng.random(n) < e1
    t_stop = rng.exponential(1 / h, n) * 365.25
    L = np.where(early, 1, 2 + np.floor(t_stop / supply)).astype(int)
    K = int(np.ceil((END + 400) / 30)) + 2
    gtype = rng.random((n, K))
    g = np.zeros((n, K), dtype=np.int32)
    r = rng.random((n, K))
    g = np.where(gtype < 0.86, np.floor(r * 8),                       # 0-7
        np.where(gtype < 0.965, 8 + np.floor(r * 23),                 # 8-30
        np.where(gtype < 0.985, 31 + np.floor(r * 30),                # 31-60
        np.where(gtype < 0.995, 61 + np.floor(r * 30),                # 61-90
                 91 + np.floor(r * 60))))).astype(np.int32)           # 91-150
    start = np.zeros((n, K), dtype=np.int32)
    start[:, 1:] = np.cumsum(supply[:, None] + g[:, :-1], axis=1)
    runout = start + supply[:, None]
    return supply, start, runout, g, L


def disc_day(runout, g, L, G):
    """day (since index) of as-treated discontinuation with grace period G: first run-out followed by a gap > G
    (or by no refill at all), plus G."""
    n, K = runout.shape
    k = np.arange(1, K + 1)[None, :]                 # fill number 1..K
    stop_here = (g > G) | (k >= L[:, None])
    # A patient with no such fill among the K simulated fills is still on treatment when the simulated record ends
    # (K fills always run past the end of the data, day END). Without the next line argmax() of an all-False row is 0,
    # which wrongly stopped these patients after their FIRST fill (fixed 2026-10-04).
    stop_here[:, -1] = True
    first = stop_here.argmax(axis=1)                 # index of first such fill
    return runout[np.arange(n), first] + G, first + 1


def pdc365(start, supply, L):
    """proportion of days covered in days 0..364 (no carry-over of overlapping supply; gaps are >= 0 here)"""
    n, K = start.shape
    k = np.arange(1, K + 1)[None, :]
    s = np.clip(start, 0, 365)
    e = np.clip(start + supply[:, None], 0, 365)
    cov = np.where(k <= L[:, None], e - s, 0).sum(axis=1)
    return cov / 365.0


def simulate(nS, nD, nN):
    parts = []
    for kind, n in (("S", nS), ("D", nD), ("N", nN)):
        d = make_group(n, kind)
        admin = END - d.idx.values
        lam_h, lam_d = lin_hhf(d), lin_death(d)
        Eh, Ed = rng.exponential(1, n), rng.exponential(1, n)
        if kind in ("S", "D"):
            supply, start, runout, g, L = fills(n, kind)
            true_stop = runout[np.arange(n), np.minimum(L, runout.shape[1]) - 1]      # run-out of the last fill
            sw = np.ceil(rng.exponential(1 / {"S": 0.03, "D": 0.045}[kind], n) * 365.25).astype(int)
            for G in (30, 60, 90):
                d[f"dc{G}"], d[f"dcfill{G}"] = disc_day(runout, g, L, G)
            d["pdc"] = pdc365(start, supply, L)
            d["supply"] = supply
            d["switch"] = sw
            d["true_stop"] = true_stop
            if kind == "S":      # protected only while on the drug
                c = true_stop / 365.25
                th = piecewise_time(Eh, lam_h * HR_ON, lam_h, c)
                td = piecewise_time(Ed, lam_d * HR_ON_D, lam_d, c)
            else:                # DPP-4i users who add/switch to an SGLT2i are protected from then on
                c = sw / 365.25
                th = piecewise_time(Eh, lam_h, lam_h * HR_ON, c)
                td = piecewise_time(Ed, lam_d, lam_d * HR_ON_D, c)
        else:
            start2 = np.ceil(rng.exponential(1 / 0.10, n) * 365.25).astype(int)   # non-users: start of either class
            d["start2"] = start2
            th, td = Eh / lam_h, Ed / lam_d
        d["hhf_day"] = np.maximum(1, np.ceil(th * 365.25)).astype(np.int64)
        d["death_day"] = np.maximum(1, np.ceil(td * 365.25)).astype(np.int64)
        d["admin"] = admin
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


NN = 60_000
C = simulate(nS, nD, NN)
for b in ("female", "hf", "ihd", "ckd", "htn", "ins", "met", "sev"):
    C[b] = C[b].astype(int)
co = C[C.grp != "N"].copy().reset_index(drop=True)
co["sglt2"] = (co.grp == "S").astype(int)

# ---------------------------------------------------------------- baseline table (Table 1-like) + non-users
pr("=" * 78, "\n나. baseline characteristics: SGLT2i vs DPP-4i vs non-users (SMD vs SGLT2i)")
S_, D_, N_ = C[C.grp == "S"], C[C.grp == "D"], C[C.grp == "N"]
base = {}
rows = [("age", "cont"), ("female", "bin"), ("htn", "bin"), ("ihd", "bin"), ("hf", "bin"), ("ckd", "bin"),
        ("met", "bin"), ("ins", "bin"), ("hba1c", "cont"), ("sev", "bin")]
for v, t in rows:
    if t == "cont":
        r = dict(S=(S_[v].mean(), S_[v].std(ddof=1)), D=(D_[v].mean(), D_[v].std(ddof=1)), N=(N_[v].mean(), N_[v].std(ddof=1)),
                 smdD=smd_cont(S_[v], D_[v]), smdN=smd_cont(S_[v], N_[v]))
        pr(f"  {v:7s} S {r['S'][0]:.1f}±{r['S'][1]:.1f}  D {r['D'][0]:.1f}±{r['D'][1]:.1f}  N {r['N'][0]:.1f}±{r['N'][1]:.1f}"
           f"   SMD S-D {r['smdD']:+.3f}  S-N {r['smdN']:+.3f}")
    else:
        r = dict(S=(int(S_[v].sum()), S_[v].mean()), D=(int(D_[v].sum()), D_[v].mean()), N=(int(N_[v].sum()), N_[v].mean()),
                 smdD=smd_bin(S_[v].mean(), D_[v].mean()), smdN=smd_bin(S_[v].mean(), N_[v].mean()))
        pr(f"  {v:7s} S {r['S'][0]} ({100 * r['S'][1]:.1f})  D {r['D'][0]} ({100 * r['D'][1]:.1f})  N {r['N'][0]} ({100 * r['N'][1]:.1f})"
           f"   SMD S-D {r['smdD']:+.3f}  S-N {r['smdN']:+.3f}")
    base[v] = r
OUT["base"] = base
OUT["nN"] = NN


# ---------------------------------------------------------------- follow-up definitions
def follow(d, mode, G=60):
    """returns time (days), event (1 = HHF), reason code"""
    cand = {"hhf": d.hhf_day.values, "death": d.death_day.values, "admin": d.admin.values}
    if mode == "at":
        cand["disc"] = d[f"dc{G}"].values
        cand["switch"] = d.switch.values
    order = ["hhf", "death", "switch", "disc", "admin"]       # tie priority
    M = np.column_stack([cand[k] for k in order if k in cand])
    names = [k for k in order if k in cand]
    j = M.argmin(axis=1)
    t = M[np.arange(len(d)), j]
    reason = np.array(names)[j]
    return t, (reason == "hhf").astype(int), reason


def rate_block(t, e, g):
    out = {}
    for k, lab in ((1, "S"), (0, "D")):
        ev, py = int(e[g == k].sum()), t[g == k].sum() / 365.25
        lo, hi = confint_poisson(ev, py, method="exact-c")
        out[lab] = dict(n=int((g == k).sum()), ev=ev, py=py, rate=1000 * ev / py, lo=1000 * lo, hi=1000 * hi,
                        mean_fu=t[g == k].mean() / 365.25, med_fu=np.median(t[g == k]) / 365.25)
    a, b = out["S"], out["D"]
    irr = a["rate"] / b["rate"]
    se = np.sqrt(1 / a["ev"] + 1 / b["ev"])
    rd = a["rate"] - b["rate"]
    se_rd = 1000 * np.sqrt(a["ev"] / a["py"] ** 2 + b["ev"] / b["py"] ** 2)
    out.update(irr=irr, irr_lo=np.exp(np.log(irr) - Z * se), irr_hi=np.exp(np.log(irr) + Z * se), se_ln=se,
               rd=rd, rd_lo=rd - Z * se_rd, rd_hi=rd + Z * se_rd)
    return out


COVS = ["age", "female", "htn", "ihd", "hf", "ckd", "met", "ins"]


def cox(d, t, e, covs):
    df = d[covs].copy()
    df["t"], df["e"] = t, e
    m = CoxPHFitter().fit(df, "t", "e")
    s = m.summary.loc["sglt2"]
    return dict(hr=float(s["exp(coef)"]), lo=float(s["exp(coef) lower 95%"]), hi=float(s["exp(coef) upper 95%"]), p=float(s["p"]))


pr("=" * 78, "\n라. follow-up: as-treated (grace 30/60/90) and ITT-like")
res = {}
for key, mode, G in (("at60", "at", 60), ("at30", "at", 30), ("at90", "at", 90), ("itt", "itt", None)):
    t, e, reason = follow(co, mode, G or 60)
    r = rate_block(t, e, co.sglt2.values)
    r["crude"] = cox(co, t, e, ["sglt2"])
    r["adj"] = cox(co, t, e, ["sglt2"] + COVS)
    r["reasons"] = {lab: {k: int(((reason == k) & (co.sglt2.values == v)).sum()) for k in ("hhf", "death", "disc", "switch", "admin")}
                    for v, lab in ((1, "S"), (0, "D"))}
    res[key] = r
    pr(f"  [{key}]")
    for lab in ("S", "D"):
        x = r[lab]
        pr(f"    {lab}: n {x['n']}, events {x['ev']}, PY {x['py']:.0f}, rate {x['rate']:.2f} ({x['lo']:.2f}-{x['hi']:.2f}) /1000PY, "
           f"mean FU {x['mean_fu']:.2f} y, median {x['med_fu']:.2f} y; reasons {r['reasons'][lab]}")
    pr(f"    crude IRR {r['irr']:.3f} ({r['irr_lo']:.3f}-{r['irr_hi']:.3f}), SE ln {r['se_ln']:.4f}; rate diff {r['rd']:.2f} ({r['rd_lo']:.2f} to {r['rd_hi']:.2f})")
    pr(f"    Cox crude HR {r['crude']['hr']:.3f} ({r['crude']['lo']:.3f}-{r['crude']['hi']:.3f}); "
       f"adjusted HR {r['adj']['hr']:.3f} ({r['adj']['lo']:.3f}-{r['adj']['hi']:.3f}), p={r['adj']['p']:.2e}")
OUT["fu"] = res

# checks used in the text: events/N (wrong comparison) and reasons percentages
t, e, reason = follow(co, "at", 60)
g = co.sglt2.values
for lab, k in (("S", 1), ("D", 0)):
    n = (g == k).sum()
    pr(f"  AT60 reasons % {lab}: " + ", ".join(f"{r_} {100 * ((reason == r_) & (g == k)).sum() / n:.1f}" for r_ in ("hhf", "death", "disc", "switch", "admin")))
ti, ei, _ = follow(co, "itt")
pr(f"  ITT events/N: S {ei[g == 1].sum()}/{(g == 1).sum()} = {100 * ei[g == 1].mean():.2f}%, D {100 * ei[g == 0].mean():.2f}%, ratio {ei[g == 1].mean() / ei[g == 0].mean():.3f}")
pr(f"  events after discontinuation that ITT adds: S {res['itt']['S']['ev'] - res['at60']['S']['ev']}, D {res['itt']['D']['ev'] - res['at60']['D']['ev']};"
   f" PY added: S {res['itt']['S']['py'] - res['at60']['S']['py']:.0f}, D {res['itt']['D']['py'] - res['at60']['D']['py']:.0f}")
for lab in ("S", "D"):
    de, dp = res['itt'][lab]['ev'] - res['at60'][lab]['ev'], res['itt'][lab]['py'] - res['at60'][lab]['py']
    pr(f"    off-treatment rate {lab}: {1000 * de / dp:.2f} /1000 PY")
# exact conditional test for the as-treated rate ratio (statsmodels)
tt = test_poisson_2indep(res["at60"]["S"]["ev"], res["at60"]["S"]["py"], res["at60"]["D"]["ev"], res["at60"]["D"]["py"], method="exact-cond")
pr(f"  test_poisson_2indep exact-cond p = {tt.pvalue:.3g}")

# reverse-KM median follow-up (as-treated, whole cohort) with lifelines
km = KaplanMeierFitter().fit(t / 365.25, 1 - e)
pr(f"  reverse-KM median follow-up (AT60): {km.median_survival_time_:.2f} y; simple median {np.median(t) / 365.25:.2f} y")
kmi = KaplanMeierFitter().fit(ti / 365.25, 1 - ei)
pr(f"  reverse-KM median follow-up (ITT): {kmi.median_survival_time_:.2f} y; simple median {np.median(ti) / 365.25:.2f} y")
OUT["medfu"] = dict(at_rkm=float(km.median_survival_time_), at_simple=float(np.median(t) / 365.25),
                    itt_rkm=float(kmi.median_survival_time_), itt_simple=float(np.median(ti) / 365.25))

# ---------------------------------------------------------------- 다. adherence / persistence in the cohort
pr("=" * 78, "\n다. PDC (first 365 days) and persistence")
pdc = {}
for lab, k in (("S", 1), ("D", 0)):
    x = co[co.sglt2 == k]
    pers = {G: float((x[f"dc{G}"] > 365).mean()) for G in (30, 60, 90)}
    pdc[lab] = dict(mean=float(x.pdc.mean()), med=float(x.pdc.median()), ge80=float((x.pdc >= 0.8).mean()),
                    n_ge80=int((x.pdc >= 0.8).sum()), pers=pers)
    pr(f"  {lab}: mean PDC {x.pdc.mean():.3f}, median {x.pdc.median():.3f}, PDC>=0.8: {int((x.pdc >= 0.8).sum())} ({100 * (x.pdc >= 0.8).mean():.1f}%)"
       f"; still on treatment at 1 y (grace 30/60/90): " + ", ".join(f"{100 * pers[G]:.1f}%" for G in (30, 60, 90)))
OUT["pdc"] = pdc
# PDC>=0.8 but 'discontinued' within a year under a 30-day grace period
x = co
both_ = ((x.pdc >= 0.8) & (x.dc30 <= 365)).sum()
pr(f"  PDC>=0.8 yet discontinued within 365 d under grace 30: {int(both_)} ({100 * both_ / len(x):.1f}% of cohort)")
OUT["pdc_vs_pers"] = dict(n=int(both_), pct=float(both_ / len(x)))

# ---- example patient: fills -> exposure episodes under three grace periods, PDC
pr("  example patient (prescription records -> as-treated follow-up)")
rx = pd.DataFrame({"date": pd.to_datetime(["2021-03-02", "2021-04-01", "2021-05-06", "2021-06-05", "2021-10-18", "2022-01-16"]),
                   "supply": [30, 30, 30, 90, 90, 90]})
rx["runout"] = rx["date"] + pd.to_timedelta(rx["supply"], unit="D")
rx["gap"] = (rx["date"].shift(-1) - rx["runout"]).dt.days
index_date, hhf_date, data_end = rx.date.iloc[0], pd.Timestamp("2022-06-20"), pd.Timestamp("2022-12-31")
pr(rx.to_string())
ex = {"rx": [(str(a.date()), int(s), str(b.date()), (None if pd.isna(gp) else int(gp))) for a, s, b, gp in zip(rx.date, rx.supply, rx.runout, rx.gap)]}
for G in (30, 60, 90):
    gaps = rx["gap"].fillna(10 ** 6)
    k = int((gaps > G).idxmax())
    stop = rx.runout.iloc[k] + pd.Timedelta(days=G)
    end = min(stop, hhf_date, data_end)
    status = int(end == hhf_date)
    days = (end - index_date).days
    pr(f"    grace {G}: discontinuation {stop.date()} (after fill {k + 1}), follow-up ends {end.date()}, time {days} d, status {status}")
    ex[f"g{G}"] = dict(stop=str(stop.date()), fill=k + 1, end=str(end.date()), days=days, status=status,
                       stop_day=(stop - index_date).days)
itt_days = (min(hhf_date, data_end) - index_date).days
pr(f"    ITT-like: ends {min(hhf_date, data_end).date()}, time {itt_days} d, status 1")
ex["itt_days"] = itt_days
ex["hhf_day"] = (hhf_date - index_date).days
win_end = index_date + pd.Timedelta(days=365)
cov = sum(max(0, (min(b, win_end) - max(a, index_date)).days) for a, b in zip(rx.date, rx.runout))
pr(f"    PDC over 365 d from index: {cov}/365 = {cov / 365:.3f}; gap days in window {365 - cov}")
ex["pdc_days"], ex["pdc"] = int(cov), cov / 365
ex["fills_day"] = [((a - index_date).days, int(s)) for a, s in zip(rx.date, rx.supply)]
OUT["ex"] = ex

# ---- practice patient (다 절 스스로 확인하기)
rx2 = pd.DataFrame({"date": pd.to_datetime(["2021-07-01", "2021-09-02", "2021-12-20"]), "supply": [60, 60, 60]})
rx2["runout"] = rx2["date"] + pd.to_timedelta(rx2["supply"], unit="D")
rx2["gap"] = (rx2["date"].shift(-1) - rx2["runout"]).dt.days
ev2, i2 = pd.Timestamp("2022-03-25"), rx2.date.iloc[0]
pr("  practice patient")
pr(rx2.to_string())
pq = {}
for G in (30, 60):
    k = int((rx2["gap"].fillna(10 ** 6) > G).idxmax())
    stop = rx2.runout.iloc[k] + pd.Timedelta(days=G)
    end = min(stop, ev2, data_end)
    pr(f"    grace {G}: stop {stop.date()} -> end {end.date()}, time {(end - i2).days} d, status {int(end == ev2)}")
    pq[f"g{G}"] = dict(stop=str(stop.date()), end=str(end.date()), days=(end - i2).days, status=int(end == ev2))
w2 = i2 + pd.Timedelta(days=365)
cov2 = sum(max(0, (min(b, w2) - max(a, i2)).days) for a, b in zip(rx2.date, rx2.runout))
pr(f"    PDC 365: {cov2}/365 = {cov2 / 365:.3f}")
pq["pdc_days"], pq["pdc"] = int(cov2), cov2 / 365
OUT["pq"] = pq

# ---- outcome misclassification: what it does to RR and RD (deterministic)
pr("  outcome misclassification (10,000 per group; true 3-year risk 3.0% vs 5.0%)")
n = 10_000
tS, tD = 300, 500
mis = {}
for lab, se_, sp_ in (("truth", 1.0, 1.0), ("sens80", 0.80, 1.0), ("sens80_spec99", 0.80, 0.99)):
    oS = se_ * tS + (1 - sp_) * (n - tS)
    oD = se_ * tD + (1 - sp_) * (n - tD)
    mis[lab] = dict(S=oS, D=oD, rr=oS / oD, rd=100 * (oS - oD) / n,
                    ppvS=se_ * tS / oS, ppvD=se_ * tD / oD)
    pr(f"    {lab:14s}: observed {oS:.0f} vs {oD:.0f}; RR {oS / oD:.3f}; RD {100 * (oS - oD) / n:.2f} %p; PPV {se_ * tS / oS:.3f} / {se_ * tD / oD:.3f}")
OUT["mis"] = mis
# validation sample PPV for the outcome definition (hypothetical chart review)
vn, vk = 200, 171
lo, hi = proportion_confint(vk, vn, method="wilson")
pr(f"  HHF definition validation: {vk}/{vn} confirmed, PPV {vk / vn:.3f} (Wilson 95% CI {lo:.3f}-{hi:.3f})")
OUT["val"] = dict(n=vn, k=vk, ppv=vk / vn, lo=lo, hi=hi)

# ---------------------------------------------------------------- 나. comparator choice: SGLT2i vs DPP-4i vs non-users
pr("=" * 78, "\n나. active comparator vs non-user comparator (on-treatment follow-up, measured covariates only)")
# on-treatment follow-up for S and D (grace 60); non-users are followed until they start either class
t, e, _ = follow(co, "at", 60)
sd = co[["age", "female", "htn", "ihd", "hf", "ckd", "met", "ins", "sev", "sglt2"]].copy()
sd["t"], sd["e"] = t, e
nu = C[C.grp == "N"].copy()
Mn = np.column_stack([nu.hhf_day.values, nu.death_day.values, nu.start2.values, nu.admin.values])
jn = Mn.argmin(axis=1)
nu["t"], nu["e"], nu["sglt2"] = Mn[np.arange(len(nu)), jn], (jn == 0).astype(int), 0
sn = pd.concat([sd[sd.sglt2 == 1], nu[sd.columns]], ignore_index=True)
cmp_ = {}
for lab, dat in (("vsD", sd), ("vsN", sn)):
    r = rate_block(dat.t.values, dat.e.values, dat.sglt2.values)
    m1 = CoxPHFitter().fit(dat[["sglt2", "t", "e"] + COVS], "t", "e").summary.loc["sglt2"]
    m2 = CoxPHFitter().fit(dat[["sglt2", "t", "e", "sev"] + COVS], "t", "e").summary.loc["sglt2"]
    cmp_[lab] = dict(rateS=r["S"]["rate"], rate0=r["D"]["rate"], ev0=r["D"]["ev"], py0=r["D"]["py"], irr=r["irr"],
                     hr=float(m1["exp(coef)"]), lo=float(m1["exp(coef) lower 95%"]), hi=float(m1["exp(coef) upper 95%"]),
                     hr_u=float(m2["exp(coef)"]), lo_u=float(m2["exp(coef) lower 95%"]), hi_u=float(m2["exp(coef) upper 95%"]))
    pr(f"  {lab}: comparator events {r['D']['ev']}, PY {r['D']['py']:.0f}, rate {r['D']['rate']:.2f}; crude IRR {r['irr']:.3f}; "
       f"HR adjusted for claims covariates {cmp_[lab]['hr']:.3f} ({cmp_[lab]['lo']:.3f}-{cmp_[lab]['hi']:.3f}); "
       f"also adjusted for unmeasured severity {cmp_[lab]['hr_u']:.3f} ({cmp_[lab]['lo_u']:.3f}-{cmp_[lab]['hi_u']:.3f})")
OUT["cmp"] = cmp_
OUT["truth"] = dict(hr_on=HR_ON)

# ---------------------------------------------------------------- 가 절 스스로 확인하기 (asthma definition, hypothetical chart review)
a_n, a_k, b_n, b_k = 150, 63, 71, 58
la, ha = proportion_confint(a_k, a_n, method="wilson")
lb, hb = proportion_confint(b_k, b_n, method="wilson")
pr("=" * 78, "\n가 practice: asthma definitions")
pr(f"  any code once: {a_k}/{a_n} = {a_k / a_n:.3f} ({la:.3f}-{ha:.3f}); code x2 + inhaler: {b_k}/{b_n} = {b_k / b_n:.3f} ({lb:.3f}-{hb:.3f}); keeps {b_k}/{a_k} = {b_k / a_k:.3f}")
OUT["asthma"] = dict(a=a_k / a_n, b=b_k / b_n, keep=b_k / a_k)
for k_ in ("at30", "at90", "at60", "itt"):
    pr(f"  exact PY {k_}: S {res[k_]['S']['py']:.3f}  D {res[k_]['D']['py']:.3f}")

# ---------------------------------------------------------------- 다 절 code cell (run exactly as printed in the chapter)
pr("=" * 78, "\n다 code cell output")
CELL = """
import pandas as pd

# 가상의 환자 한 명의 원외처방 기록 (한 행 = 처방 한 건, supply = 공급일수)
rx = pd.DataFrame({
    "pt": ["P1"] * 6,
    "date": ["2021-03-02", "2021-04-01", "2021-05-06",
             "2021-06-05", "2021-10-18", "2022-01-16"],
    "supply": [30, 30, 30, 90, 90, 90],
})
rx["date"] = pd.to_datetime(rx["date"])       # 글자 → 날짜
rx = rx.sort_values(["pt", "date"])           # 환자별, 날짜순으로 정렬

# 약이 떨어지는 날 = 처방일 + 공급일수
rx["runout"] = rx["date"] + pd.to_timedelta(rx["supply"], unit="D")
# 같은 환자의 다음 처방일 (마지막 처방은 빈칸 NaT)
rx["next"] = rx.groupby("pt")["date"].shift(-1)
# 공백 일수 = 다음 처방일 - 약이 떨어지는 날
rx["gap"] = (rx["next"] - rx["runout"]).dt.days
print(rx[["date", "supply", "runout", "gap"]])

def stop_date(g, grace):
    # 공백이 유예기간보다 길거나 다음 처방이 없는 첫 처방을 찾아
    # 그 처방의 '약이 떨어지는 날 + 유예기간'을 중단일로 돌려줌
    over = g["gap"].isna() | (g["gap"] > grace)
    return g.loc[over, "runout"].iloc[0] + pd.Timedelta(days=grace)

for grace in [30, 60, 90]:
    stop = rx.groupby("pt").apply(stop_date, grace=grace)   # 환자마다 중단일 하나
    print("유예", grace, "일 → 중단일", stop.iloc[0].date())

# 첫 365일의 PDC = 구간 안에서 약이 있던 날 수 ÷ 365
index_date = rx["date"].min()
win_end = index_date + pd.Timedelta(days=365)
days = (rx["runout"].clip(upper=win_end) - rx["date"]).dt.days
covered = days.clip(lower=0).sum()
print("PDC =", covered, "/ 365 =", round(covered / 365, 3))
"""
exec(CELL)
OUT["cell"] = CELL

# ---------------------------------------------------------------- pandas merge / groupby demo (functions named in the '내 연구에 쓸 때' boxes)
claims = pd.DataFrame({"claim": [1, 2, 3, 4, 5, 6, 7], "pt": ["P1"] * 7,
                       "date": pd.to_datetime(["2021-03-02", "2021-04-01", "2021-05-06", "2021-06-05", "2021-10-18", "2022-01-16", "2022-06-20"]),
                       "inpatient": [0, 0, 0, 0, 0, 0, 1]})
dxs = pd.DataFrame({"claim": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7],
                    "code": ["E11", "I10"] * 6 + ["I50", "E11", "I10"], "primary": [1, 0] * 6 + [1, 0, 0]})
m = claims.merge(dxs, on="claim")
pt = m.groupby("pt").agg(first_date=("date", "min"), n_claims=("claim", "nunique"),
                         n_e11_outpt=("code", lambda c: int(((c == "E11") & (m.loc[c.index, "inpatient"] == 0)).sum())))
hhf_ = m[(m.inpatient == 1) & (m.primary == 1) & (m.code == "I50")].groupby("pt")["date"].min()
pr("=" * 78, "\npandas merge/groupby demo (그림 14-1의 환자)")
pr(pt.to_string()); pr("  first HHF admission:", hhf_.iloc[0].date(), "; first rx via groupby().min():", claims.groupby("pt")["date"].min().iloc[0].date())

with open(os.path.join(HERE, "_ch14_nums.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=float)
pr("\nsaved gen/_ch14_nums.json")
