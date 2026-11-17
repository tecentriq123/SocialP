"""Numbers for the 4장 quiz (qz04). Hypothetical examples. Run: python3 gen/quiz_ch04.py"""
import math
from scipy import stats
from scipy.stats import studentized_range as SR
# q01 FWER of all pairwise t-tests (equal n, large df) via the studentized range
for k in (3, 4, 5):
    df = 1000; tc = stats.t.ppf(.975, df); fwer = SR.sf(math.sqrt(2)*tc, k, df)
    print("q01 groups", k, "pairs", k*(k-1)//2, "FWER(true)", round(fwer, 3), "1-0.95^pairs", round(1-0.95**(k*(k-1)//2), 3))
# q04 ANOVA table: 4 groups, 100 people, SSb 300, SSw 4800
k, N, SSb, SSw = 4, 100, 300, 4800
MSb, MSw = SSb/(k-1), SSw/(N-k); F = MSb/MSw
print("q04 df", k-1, N-k, "MSb", MSb, "MSw", MSw, "F", F, "P", round(stats.f.sf(F, k-1, N-k), 3), "eta2", round(SSb/(SSb+SSw), 3))
# q06 Welch-ANOVA situation: sizes 12, 25, 50; SDs 20, 12, 6
ns, sds = (12, 25, 50), (20, 12, 6)
msw = sum((n-1)*s**2 for n, s in zip(ns, sds))/sum(n-1 for n in ns)
print("q06 pooled MSw", round(msw, 1), "pooled SD", round(math.sqrt(msw), 1), "| true SE of 12-group mean", round(20/math.sqrt(12), 2), "pooled-based", round(math.sqrt(msw/12), 2))
# case: three arms, 20 each, SBP change -3, -8, -10, SD 9 each
means = {"usual": -3.0, "A": -8.0, "B": -10.0}; n = 20; sd = 9.0; k = 3; N = n*k
g = sum(means.values())/k; SSb = n*sum((m-g)**2 for m in means.values()); MSw = sd**2; F = (SSb/(k-1))/MSw
print("case SSb", SSb, "MSb", SSb/(k-1), "MSw", MSw, "F", round(F, 2), "df", (k-1, N-k), "P", round(stats.f.sf(F, k-1, N-k), 3), "eta2", round(SSb/(SSb+MSw*(N-k)), 3))
se = math.sqrt(MSw*2/n); q = SR.ppf(.95, k, N-k); thr = q/math.sqrt(2)*se
print("  Tukey SE", round(se, 3), "q crit", round(q, 3), "threshold", round(thr, 2))
for a, b in (("A", "usual"), ("B", "usual"), ("B", "A")):
    d = means[a]-means[b]; p = SR.sf(abs(d)/se*math.sqrt(2), k, N-k)
    print("  ", a, "-", b, d, "CI", round(d-thr, 1), round(d+thr, 1), "Tukey P", round(p, 3), "| unadjusted P", round(2*stats.t.sf(abs(d)/se, N-k), 3))
