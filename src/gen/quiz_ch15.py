"""Numbers for the 15장 quiz (qz15). Hypothetical examples. Run: python3 gen/quiz_ch15.py"""
import math
ps = lambda l: 1/(1+math.exp(-l))
# q03 propensity score from logit = -1.0 + 0.8*HF + 0.02*(age-70)
for age, hf in ((80, 1), (60, 0)):
    l = -1.0 + 0.8*hf + 0.02*(age-70); print("q03 age", age, "HF", hf, "logit", round(l, 2), "PS", round(ps(l), 3))
# q06 IPTW toy: HF stratum 100 (A 80, B 20), no-HF 200 (A 30, B 170)
strata = {"HF": (80, 20), "noHF": (30, 170)}
wA = {k: (a+b)/a for k, (a, b) in strata.items()}; wB = {k: (a+b)/b for k, (a, b) in strata.items()}
pseudoA = {k: strata[k][0]*wA[k] for k in strata}; pseudoB = {k: strata[k][1]*wB[k] for k in strata}
print("q06 weights A", {k: round(v, 2) for k, v in wA.items()}, "B", {k: round(v, 2) for k, v in wB.items()})
print("q06 pseudo A", pseudoA, "pseudo B", pseudoB, "HF share A", round(pseudoA['HF']/sum(pseudoA.values()), 3), "| before weighting A HF share", round(80/110, 3))
# E-values
def evalue(rr):
    r = 1/rr if rr < 1 else rr
    return r + math.sqrt(r*(r-1))
for rr in (0.70, 0.80, 0.97): print("E-value", rr, round(evalue(rr), 2))
