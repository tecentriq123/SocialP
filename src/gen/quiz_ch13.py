"""Numbers for the 13장 quiz (qz13). Hypothetical examples. Run: python3 gen/quiz_ch13.py"""
import math
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep
# q04 margin: M1 = conservative lower bound of control-vs-placebo effect 16%p, preserve 50%
print("q04 M2", 16*0.5)
# q07 NI p vs superiority p: diff -1.0, SE 1.5, margin 4
d, se, D = -1.0, 1.5, 4.0
z = (d + D)/se; print("q07 NI z", round(z, 2), "one-sided p", round(stats.norm.sf(z), 3), "| superiority two-sided p", round(2*stats.norm.sf(abs(d/se)), 2), "| 95% CI", round(d-1.96*se, 2), round(d+1.96*se, 2))
# q09 relative loss allowed by -10%p
for c in (0.90, 0.60): print("q09 control success", c, "failure", round(1-c, 2), "->", round(1-c+0.10, 2), "ratio", round((1-c+0.10)/(1-c), 2))
# case: cure 270/300 vs 276/300 (ITT), 240/260 vs 250/265 (PP)
for lab, a, n1, b, n0 in (("ITT", 270, 300, 276, 300), ("PP", 240, 260, 250, 265)):
    lo, hi = confint_proportions_2indep(a, n1, b, n0, method="newcomb", compare="diff")
    print("case", lab, "rates", round(100*a/n1, 1), round(100*b/n0, 1), "diff", round(100*(a/n1-b/n0), 1), "Newcombe CI", round(100*lo, 1), round(100*hi, 1))
print("case design: failure 15% -> 25%", round(0.25/0.15, 2), "| observed: 8% -> 18%", round(0.18/0.08, 2))
# case q02 condition and q04 distractor P
print("case stricter margin (1.67x of 8% failure)", round(-(0.08*1.67-0.08)*100, 1))
from scipy.stats import chi2_contingency
print("case superiority chi2 P", round(chi2_contingency([[270, 30], [276, 24]], correction=False)[1], 2))
