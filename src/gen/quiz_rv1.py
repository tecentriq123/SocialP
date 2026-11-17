"""Numbers for 종합 1 (qz91, RCT). Hypothetical asthma inhaler trial, N vs M, 500 per arm. Run: python3 gen/quiz_rv1.py"""
import math
from scipy import stats
from statsmodels.stats.proportion import confint_proportions_2indep
# primary: 52-week trough FEV1 change, difference N - M = +12 mL, SE 13.3; NI margin -50 mL
d, se = 12, 13.3
print("b1 95%% CI %.1f to %.1f | superiority P %.2f" % (d - 1.96*se, d + 1.96*se, 2*(1 - stats.norm.cdf(abs(d)/se))))
# exacerbations: N 300 events / 480 PY, M 380 / 470 PY
rn, rm = 300/480, 380/470; print("c1 rates %.3f %.3f ratio %.2f" % (rn, rm, rn/rm))
# time to first exacerbation: KM 1-year proportions 38% vs 44%, HR 0.82
print("c2 1-year risk ratio %.2f" % (0.38/0.44))
# pneumonia 6/500 vs 1/500
print("c3 Fisher P %.3f" % stats.fisher_exact([[6, 494], [1, 499]])[1])
lo, hi = confint_proportions_2indep(6, 500, 1, 500, method="newcomb", compare="diff")
print("c3 RD %.1f%%p (Newcombe %.1f to %.1f) NNH %.0f" % (100*(6/500 - 1/500), 100*lo, 100*hi, 1/(6/500 - 1/500)))
# subgroup FEV1 +40 mL (5-75) vs MCID 100
