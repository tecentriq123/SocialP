"""Numbers for the 24장 quiz (qz24). Hypothetical examples (만원, QALY). Run: python3 gen/quiz_ch24.py"""
import math
# q03: dQALY 0.40, dCost 2,400 (ICER 6,000), threshold 5,000; drug used 20 discounted months; monthly price 200
gap = 2400 - 5000*0.40; cut = gap/20
print("q03 gap", gap, "monthly cut", cut, "threshold price", 200 - cut, "cut %", 100*cut/200)
# q06: HR 0.80 (0.64-1.00) -> lognormal
print("q06 log mean %.3f log sd %.3f | wrong (natural scale) %.3f" % (math.log(0.80), (math.log(1.00) - math.log(0.64))/(2*1.96), (1.00 - 0.64)/(2*1.96)))
# q11: five simulations at threshold 4,000
sims = [(0.30, 1000), (0.20, 1100), (0.45, 1500), (0.25, 900), (0.10, 600)]
nmb = [4000*q - c for q, c in sims]; print("q11 NMB", nmb, "prob", sum(x > 0 for x in nmb)/5, "mean", sum(nmb)/5)
# case: PSA mean dCost 2,300, dQALY 0.48
for t in (3000, 4000, 5000, 6000): print("case NMB at", t, 0.48*t - 2300)
print("case PSA ICER %.0f" % (2300/0.48))
