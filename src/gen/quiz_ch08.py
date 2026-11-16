"""Numbers for the 8장 quiz (quiz course, qz08). All examples are hypothetical. Run: python3 gen/quiz_ch08.py"""
import math
from scipy import stats
Z = 1.959964

def rr_or(a, n1, c, n0):
    b, d = n1 - a, n0 - c
    rr = (a / n1) / (c / n0); se_rr = math.sqrt(1/a - 1/n1 + 1/c - 1/n0)
    orr = (a * d) / (b * c); se_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
    rd = a / n1 - c / n0; se_rd = math.sqrt((a/n1)*(1-a/n1)/n1 + (c/n0)*(1-c/n0)/n0)
    ci = lambda e, s: (e * math.exp(-Z * s), e * math.exp(Z * s))
    return dict(rr=rr, rr_ci=ci(rr, se_rr), or_=orr, or_ci=ci(orr, se_or), rd=rd, rd_ci=(rd - Z*se_rd, rd + Z*se_rd), nnt=(1/rd if rd else float("inf")))

print("Q2 (60/200 vs 30/200):", {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else round(v, 3)) for k, v in rr_or(60, 200, 30, 200).items()})
b, se = 0.03, 0.01
print("Q4: OR per 10y", round(math.exp(10*b), 3), "CI", round(math.exp(10*(b - Z*se)), 3), round(math.exp(10*(b + Z*se)), 3),
      " wrong: 1+10b", 1 + 10*b, " e^b", round(math.exp(b), 4))
for p0 in (0.01, 0.40):
    o1 = p0/(1-p0)*2; p1 = o1/(1+o1)
    print(f"Q5: base {p0:.0%} -> risk {p1:.4f}, RD {p1-p0:.4f}, RR {p1/p0:.3f}")
chi = 1250.4 - 1241.2
print("Q9: chi2", round(chi, 1), "df 4 P", round(stats.chi2.sf(chi, 4), 4))
# Q7 interaction from two subgroup HRs
def se_from_ci(lo, hi): return (math.log(hi) - math.log(lo)) / (2*Z)
s1, s2 = se_from_ci(0.55, 0.89), se_from_ci(0.62, 1.17)
d = math.log(0.85) - math.log(0.70); sd = math.sqrt(s1**2 + s2**2)
print("Q7: ratio of HRs", round(0.85/0.70, 3), "z", round(d/sd, 3), "P", round(2*stats.norm.sf(abs(d/sd)), 3))
# Case: stratified by age
strata = [(8, 400, 24, 1200), (48, 600, 32, 400)]   # (A events, A n, B events, B n): <75, >=75
a = sum(s[0] for s in strata); n1 = sum(s[1] for s in strata); c = sum(s[2] for s in strata); n0 = sum(s[3] for s in strata)
crude = rr_or(a, n1, c, n0)
print("Case crude:", a, n1, c, n0, {k: (tuple(round(x, 3) for x in v) if isinstance(v, tuple) else round(v, 4)) for k, v in crude.items()})
for s in strata:
    r = rr_or(*s); print("  stratum", s, "risks", round(s[0]/s[1], 3), round(s[2]/s[3], 3), "RR", round(r["rr"], 3), tuple(round(x, 2) for x in r["rr_ci"]))
# Mantel-Haenszel RR with Greenland-Robins variance
num = sum(s[0]*s[3]/(s[1]+s[3]) for s in strata); den = sum(s[2]*s[1]/(s[1]+s[3]) for s in strata)
rr_mh = num / den
P = sum((s[1]*s[3]*(s[0]+s[2]) - s[0]*s[2]*(s[1]+s[3])) / (s[1]+s[3])**2 for s in strata)
se_mh = math.sqrt(P / (num*den))
print("Case MH RR", round(rr_mh, 3), "CI", round(rr_mh*math.exp(-Z*se_mh), 2), round(rr_mh*math.exp(Z*se_mh), 2))
print("Case age >=75 share: A", 600/1000, "B", 400/1600)
