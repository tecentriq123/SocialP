"""Numbers for 종합 3 (qz93, prediction model). Hypothetical: solid-tumour patients starting first chemotherapy cycle,
febrile neutropenia (FN) in cycle 1. Development hospital A (n 1,200, FN 96), external validation hospital B (n 1,000, FN 50).
Run: python3 gen/quiz_rv3.py"""
import math
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

expit = lambda x: 1 / (1 + np.exp(-x))

# a2: candidate coefficients: age spline 2, albumin spline 2, cancer type 4 levels -> 3 dummies, 9 others x 1
coefs = 2 + 2 + 3 + 9; n, ev = 1200, 96
print("a2 coefficients %d | EPV %.1f | per variable (12) %.1f | after selection 8 coefs %.1f | non-events/coefs %.1f" %
      (coefs, ev / coefs, ev / 12, ev / 8, (n - ev) / coefs))

# b2: bootstrap optimism (given): apparent 0.780, bootstrap-sample mean 0.801, applied-to-original mean 0.751
app, bs, orig = 0.780, 0.801, 0.751
print("b2 optimism %.3f corrected C %.3f | wrong: app+opt %.3f" % (bs - orig, app - (bs - orig), app + (bs - orig)))

# c: 2x2 at threshold 20% in development data
TP, FN_, FP, TN = 42, 54, 108, 996
sens, spec = TP / (TP + FN_), TN / (FP + TN); ppv, npv = TP / (TP + FP), TN / (TN + FN_)
acc = (TP + TN) / n; acc_none = (n - ev) / n
print("c2 sens %.1f%% spec %.1f%% PPV %.1f%% NPV %.1f%% acc %.1f%% | all-negative acc %.1f%%" %
      (100 * sens, 100 * spec, 100 * ppv, 100 * npv, 100 * acc, 100 * acc_none))
d = norm.ppf(sens) - norm.ppf(1 - spec)
print("   binormal AUC implied by this point %.3f (apparent C 0.78)" % norm.cdf(d / math.sqrt(2)))
for prev in (0.04, 0.02):
    tp, fp = 10000 * prev * sens, 10000 * (1 - prev) * (1 - spec)
    print("   PPV at prevalence %.0f%%: %.1f%%" % (100 * prev, 100 * tp / (tp + fp)))

# b4: adding lymphocyte count
print("b4 delta C %.3f" % (0.740 - 0.730))

# d1: external validation simulation (population values): predicted lp ~ N(mu, s) with mean predicted 8%;
# truth in B: logit p = a + 0.93 * lp, mean true 5%
rng = np.random.default_rng(93)
N = 400_000
s = 0.85
mu = -2.95
for _ in range(60):  # tune mu so mean predicted = 8.0%
    lp = rng.normal(mu, s, N); m = expit(lp).mean(); mu += (0.08 - m) * 8
lp = rng.normal(mu, s, N)
a = -1.0
for _ in range(60):  # tune a so mean true = 5.0%
    a += (0.05 - expit(a + 0.93 * lp).mean()) * 10
p_true = expit(a + 0.93 * lp)
y = rng.binomial(1, p_true)
print("d1 mean predicted %.3f mean true %.3f" % (expit(lp).mean(), p_true.mean()))
X = sm.add_constant(lp)
slope = sm.GLM(y, X, family=sm.families.Binomial()).fit().params[1]
citl = sm.GLM(y, np.ones((N, 1)), family=sm.families.Binomial(), offset=lp).fit().params[0]
# C statistic in B
from scipy.stats import rankdata
r = rankdata(lp); n1 = y.sum(); n0 = N - n1
C = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
print("   calibration slope %.2f intercept (CITL) %.2f | C %.3f | logit difference of means %.2f" %
      (slope, citl, C, math.log(0.05 / 0.95) - math.log(0.08 / 0.92)))
# approximate CI half-width for C with 50 events / 950 non-events (Hanley-McNeil)
Cv, n1s, n0s = 0.71, 50, 950
q1, q2 = Cv / (2 - Cv), 2 * Cv ** 2 / (1 + Cv)
se = math.sqrt((Cv * (1 - Cv) + (n1s - 1) * (q1 - Cv ** 2) + (n0s - 1) * (q2 - Cv ** 2)) / (n1s * n0s))
print("   Hanley-McNeil 95%% CI for C 0.71: %.2f-%.2f" % (Cv - 1.96 * se, Cv + 1.96 * se))
