"""Numbers for the 17장 quiz (qz17). Hypothetical examples. Run: python3 gen/quiz_ch17.py"""
import math, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from lifelines import AalenJohansenFitter, KaplanMeierFitter, CoxPHFitter
# q02: 8 patients, outcome fracture (1), death (2), censored (0); 3-year follow-up
t = [0.5, 0.9, 1.4, 1.8, 2.2, 2.5, 3.0, 3.0]; e = [1, 2, 1, 2, 0, 1, 0, 0]
aj = AalenJohansenFitter(calculate_variance=False).fit(t, e, event_of_interest=1)
km = KaplanMeierFitter().fit(t, [1 if x == 1 else 0 for x in e])
print("q02 CIF 3y %.3f  1-KM %.3f  simple %.3f  excl deaths %.3f" % (aj.cumulative_density_.iloc[-1, 0], 1 - km.survival_function_.iloc[-1, 0], 3/8, 3/6))
# q09: constant rates; fracture 0.05/PY both groups, death 0.10 (A) vs 0.20 (B); 5 years
cif = lambda l1, l2, T: l1/(l1 + l2)*(1 - math.exp(-(l1 + l2)*T))
print("q09 CIF A %.3f B %.3f  1-KM both %.3f" % (cif(.05, .10, 5), cif(.05, .20, 5), 1 - math.exp(-.05*5)))
# case: AF patients, anticoagulant X vs Y, stroke (1) vs death before stroke (2), 5-year admin follow-up
import sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 17)  # seed 17 is the one used in qz17
def sim(n, ls, ld, g):
    ts, td = rng.exponential(1/ls, n), rng.exponential(1/ld, n)
    tt = np.minimum(np.minimum(ts, td), 5.0); ev = np.where(tt == 5.0, 0, np.where(ts < td, 1, 2))
    return pd.DataFrame({"t": tt, "ev": ev, "x": g})
d = pd.concat([sim(4000, .04, .08, 1), sim(4000, .04, .16, 0)], ignore_index=True)
for g, name in ((1, "X"), (0, "Y")):
    s = d[d.x == g]
    a1 = AalenJohansenFitter(calculate_variance=False).fit(s.t, s.ev, event_of_interest=1)
    a2 = AalenJohansenFitter(calculate_variance=False).fit(s.t, s.ev, event_of_interest=2)
    k = KaplanMeierFitter().fit(s.t, (s.ev == 1).astype(int))
    c1, c2 = a1.cumulative_density_.iloc[-1, 0], a2.cumulative_density_.iloc[-1, 0]
    print(f"case {name}: n {len(s)} strokes {(s.ev==1).sum()} deaths {(s.ev==2).sum()} CIF stroke {c1:.3f} death {c2:.3f} free {1-c1-c2:.3f} 1-KM {1-k.survival_function_.iloc[-1,0]:.3f}")
def hr(df, col):
    m = CoxPHFitter().fit(df[["t", col, "x"]], "t", col); r = m.summary.loc["x"]
    return "%.2f (%.2f-%.2f)" % (r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"])
d["s"] = (d.ev == 1).astype(int); d["dth"] = (d.ev == 2).astype(int); d["comp"] = (d.ev > 0).astype(int)
print("case CSHR stroke", hr(d, "s"), "CSHR death", hr(d, "dth"), "composite", hr(d, "comp"))
fg = d.copy(); fg.loc[fg.ev == 2, "t"] = 5.0   # subdistribution risk set: deaths kept to planned end (admin censoring only)
print("case SHR stroke", hr(fg, "s"))
