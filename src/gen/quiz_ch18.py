"""Numbers for the 18장 quiz (qz18). Hypothetical examples. Run: python3 gen/quiz_ch18.py"""
import numpy as np, pandas as pd, statsmodels.formula.api as smf
# q02: pre slope -0.30/month, 24 months before, 12 after; observed pre-post difference -9.0
mid_pre, mid_post = (1 + 24)/2, (25 + 36)/2
print("q02 distance", mid_post - mid_pre, "trend part", -0.30*(mid_post - mid_pre), "rest", -9.0 - (-0.30*(mid_post - mid_pre)))
# q04: baseline 50, trend -0.10, level -2.0, slope change -0.15; 36 pre months; 12th month after start = time 48
cf = 50 - 0.10*48; eff = -2.0 + 12*(-0.15); print("q04 counterfactual", cf, "effect", eff, "relative %.1f%%" % (100*eff/cf), "post slope", -0.10 - 0.15)
# q10: DID 2x2
print("q10 DID", (31.0 - 40.0) - (27.0 - 30.0))
# q13: spillover; true comparison level change -2.0 became -0.5; policy level change -6.0
print("q13 true diff", -6.0 - (-2.0), "with spillover", -6.0 - (-0.5))
# case: monthly benzodiazepine rate in older adults (per 1,000), policy region vs comparison region; 36 pre + 24 post months
rng = np.random.default_rng(18)
T = np.arange(1, 61); post = (T > 36).astype(int); ta = np.where(T > 36, T - 36, 0)
season = 1.0*np.cos(2*np.pi*(T - 1)/12)                       # winter high (time 1 = January)
common = np.zeros(60); e = rng.normal(0, 0.6, 60)
for i in range(60): common[i] = (0.6*common[i-1] if i else 0) + e[i]   # shared AR(1) noise
pol = 48 - 0.10*T + season + common - 1.5*post - 3.0*post - 0.08*ta + rng.normal(0, 0.35, 60)
cmp_ = 40 - 0.10*T + season + common - 1.5*post + rng.normal(0, 0.35, 60)
d = pd.DataFrame({"time": T, "post": post, "ta": ta, "pol": pol, "cmp": cmp_, "diff": pol - cmp_})
f = " ~ time + post + ta + np.sin(2*np.pi*time/12) + np.cos(2*np.pi*time/12)"
for y in ("pol", "cmp", "diff"):
    m = smf.ols(y + f, d).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    ci = m.conf_int(); t12 = m.t_test("post + 12*ta = 0")
    lo, hi = t12.conf_int()[0]
    print(f"case {y}: trend {m.params['time']:.3f} ({ci.loc['time',0]:.3f} to {ci.loc['time',1]:.3f}) level {m.params['post']:.2f} ({ci.loc['post',0]:.2f} to {ci.loc['post',1]:.2f}) slope {m.params['ta']:.3f} ({ci.loc['ta',0]:.3f} to {ci.loc['ta',1]:.3f}) 12m {float(t12.effect[0]):.2f} ({lo:.2f} to {hi:.2f})")
pre, po = d[d.post == 0], d[d.post == 1]
a, b, c, dd = pre.pol.mean(), po.pol.mean(), pre.cmp.mean(), po.cmp.mean()
print(f"case 2x2 policy {a:.2f} -> {b:.2f} ({b-a:.2f}); comparison {c:.2f} -> {dd:.2f} ({dd-c:.2f}); DID {(b-a)-(dd-c):.2f}")
print("case true 12m effect", -3.0 - 0.08*12, "true 24m avg", -3.0 - 0.08*12.5)
