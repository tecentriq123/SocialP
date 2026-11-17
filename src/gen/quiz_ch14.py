"""Numbers for the 14장 quiz (qz14). Hypothetical examples. Run: python3 gen/quiz_ch14.py"""
import math
# q01 PPV of two definitions
print("q01 PPV A", 600/800, "PPV B", round(500/520, 3), "| B captures", round(500/600, 3), "of A's true cases")
# q06 supply days, gaps, grace 30 days; PDC over first 180 days
rx = [(0, 30), (30, 30), (70, 30), (160, 90)]
end = [d+s for d, s in rx]; gaps = [rx[i+1][0]-end[i] for i in range(len(rx)-1)]
print("q06 run-out", end, "gaps", gaps)
grace = 30; disc = None
for i, g in enumerate(gaps):
    if g > grace: disc = end[i] + grace; break
print("q06 discontinuation day (run-out + grace)", disc)
cov = set()
for d, s in rx:
    cov |= set(range(d, d+s))
pdc = len([x for x in cov if x < 180])/180; print("q06 PDC first 180 days", round(pdc, 2), len([x for x in cov if x < 180]))
# q07 outcome misclassification: true 2% vs 4%, sensitivity 75%, PPV 100%
for se, lab in ((1.0, "perfect"), (0.75, "sens 75%")):
    a, b = 0.02*se, 0.04*se; print("q07", lab, "risks", a, b, "RR", round(a/b, 2), "RD %p", round(100*(a-b), 2))
# false positives 1% added to both groups
a, b = 0.02+0.01, 0.04+0.01; print("q07 + 1% false positives RR", round(a/b, 2))
# q11 ROR
a, b, c, d = 30, 970, 600, 98400
ror = (a/b)/(c/d); se = math.sqrt(1/a+1/b+1/c+1/d)
print("q11 ROR", round(ror, 2), "CI", round(math.exp(math.log(ror)-1.96*se), 2), round(math.exp(math.log(ror)+1.96*se), 2), "| PRR", round((a/(a+b))/(c/(c+d)), 2))
