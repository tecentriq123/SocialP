"""Numbers for the 1장 quiz (qz01). Hypothetical examples. Run: python3 gen/quiz_ch01.py"""
import math
from scipy import stats
N = stats.norm
# q03 SMD vs P with 20,000 per group
for name, m1, m0, sd in [("BMI", 27.1, 26.9, 4.5), ("age", 64.0, 61.0, 12.0)]:
    n = 20000; se = sd * math.sqrt(2 / n); t = (m1 - m0) / se
    print("q03", name, "SMD", round((m1 - m0) / sd, 3), "t", round(t, 2), "P", f"{2*stats.t.sf(t, 2*n-2):.1e}")
# q07 power, normal approximation, SD 10, 50 per group
def power(d, sd, n):
    c = d / math.sqrt(sd**2 / n * 2); return N.cdf(c - 1.96) + N.cdf(-c - 1.96), c
print("q07 effect 5:", [round(v, 3) for v in power(5, 10, 50)], " effect 2.5:", [round(v, 3) for v in power(2.5, 10, 50)], " 2.5 with 200/grp:", [round(v, 3) for v in power(2.5, 10, 200)])
# q08 FWER for 8 independent tests
print("q08 FWER 8 tests", round(1 - 0.95**8, 3), " Bonferroni-ish 8*0.05", 8 * 0.05)
# q10 t^2 = F
t = 2.50; df = 58
print("q10 F", t**2, "P t", round(2*stats.t.sf(t, df), 4), "P F", round(stats.f.sf(t**2, 1, df), 4))
# q14 log scale difference
print("q14 exp(-0.357)", round(math.exp(-0.357), 3))
# case: weight change, 40 per group, values given as mean ± SE
n = 40; se1, se0 = 0.63, 0.70; m1, m0 = -6.2, -1.8
sd1, sd0 = se1 * math.sqrt(n), se0 * math.sqrt(n)
print("case SD", round(sd1, 2), round(sd0, 2), " SE*n", se1 * n, " SE/sqrt n", round(se1 / math.sqrt(n), 3))
d = m1 - m0; sed = math.sqrt(se1**2 + se0**2); tc = stats.t.ppf(0.975, 2*n - 2)
print("case diff", round(d, 1), "SE", round(sed, 3), "t", round(d / sed, 2), "df", 2*n-2, "tcrit", round(tc, 3),
      "CI", round(d - tc*sed, 2), round(d + tc*sed, 2), "P", f"{2*stats.t.sf(abs(d/sed), 2*n-2):.1e}")
sdd = math.sqrt(sd1**2 + sd0**2); print("  wrong SD-based CI", round(d - tc*sdd, 1), round(d + tc*sdd, 1),
      " SE-sum CI", round(d - tc*(se1+se0), 1), round(d + tc*(se1+se0), 1))
# case: six secondary outcomes, Holm and Bonferroni
ps = {"허리둘레": 0.0004, "수축기혈압": 0.009, "HbA1c": 0.034, "삶의 질": 0.048, "LDL": 0.21, "중성지방": 0.47}
items = sorted(ps.items(), key=lambda x: x[1]); m = len(items); run = 0; stop = False
for i, (k, p) in enumerate(items):
    h = min(1, p * (m - i)); run = max(run, h)
    sig = (not stop) and h < 0.05; stop = stop or not sig
    print("  Holm", k, p, "x", m - i, "=", round(p*(m-i), 4), "adj", round(run, 4), "sig" if sig else "-", "| Bonf", round(min(1, p*m), 4))
print("  unadjusted <0.05:", sum(p < 0.05 for p in ps.values()), " Bonferroni threshold", round(0.05/m, 4))
# case q02: Welch df with the same SEs (equal n)
a, b = se1**2, se0**2
print("case Welch df", round((a + b)**2 / (a**2 / (n - 1) + b**2 / (n - 1)), 1))
