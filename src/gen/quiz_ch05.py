"""Numbers for the 5장 quiz (qz05). Hypothetical examples. Run: python3 gen/quiz_ch05.py"""
import math
from scipy import stats
from scipy.stats import fisher_exact, chi2_contingency, binomtest
from scipy.stats.contingency import odds_ratio
def measures(a, n1, c, n0):
    r1, r0 = a/n1, c/n0; o1, o0 = a/(n1-a), c/(n0-c)
    return dict(r1=r1, r0=r0, RD=r1-r0, RR=r1/r0, OR=o1/o0)
r = lambda d: {k: round(v, 4) for k, v in d.items()}
# q01 expected frequency and chi-square: A 30/200, B 20/300
t = [[30, 170], [20, 280]]; chi, p, df, E = chi2_contingency(t, correction=False)
print("q01 expected", E.round(1).tolist(), "chi2", round(chi, 2), "df", df, "P", round(p, 4))
# q02 measures: A 24/120, B 30/300
print("q02", r(measures(24, 120, 30, 300)))
# q03 common outcome: 40% vs 25%
print("q03 RR", 0.40/0.25, "OR", round((0.4/0.6)/(0.25/0.75), 3))
# q05 rare AE: 1/40 vs 6/38
t = [[1, 39], [6, 32]]; chi, p, df, E = chi2_contingency(t, correction=False)
print("q05 expected", E.round(2).tolist(), "chi2 P", round(p, 4), "Fisher P", round(fisher_exact(t).pvalue, 4))
# q08 Fisher P < 0.05 but exact CI includes 1: 1/20 vs 8/25
t = [[1, 19], [8, 17]]; ci = odds_ratio(t, kind="conditional").confidence_interval()
print("q08 Fisher P", round(fisher_exact(t).pvalue, 4), "cond OR", round(odds_ratio(t, kind="conditional").statistic, 3), "exact CI", round(ci.low, 3), round(ci.high, 2))
# q09 small trial 0/15 vs 3/15
t = [[0, 15], [3, 12]]; print("q09 Fisher P", round(fisher_exact(t).pvalue, 3))
# q13 McNemar: smoking before/after, 18 quit, 4 started (illustration)
print("q13 exact McNemar P", round(binomtest(4, 22, 0.5).pvalue, 4))
# case: complete response 42/60 vs 30/60
m = measures(42, 60, 30, 60); print("case", r(m))
se = math.sqrt(0.7*0.3/60 + 0.5*0.5/60); print("  RD CI", round(100*(m['RD']-1.96*se), 1), round(100*(m['RD']+1.96*se), 1))
sl = math.sqrt(1/42-1/60+1/30-1/60); print("  RR CI", round(math.exp(math.log(1.4)-1.96*sl), 2), round(math.exp(math.log(1.4)+1.96*sl), 2))
so = math.sqrt(1/42+1/18+1/30+1/30); print("  OR CI", round(math.exp(math.log(m['OR'])-1.96*so), 2), round(math.exp(math.log(m['OR'])+1.96*so), 2))
t = [[42, 18], [30, 30]]
for corr in (False, True):
    chi, p, df, E = chi2_contingency(t, correction=corr); print("  chi2 corr", corr, round(chi, 2), "P", round(p, 4), "E", E.tolist())
print("  Fisher P", round(fisher_exact(t).pvalue, 4))
# q07: the q05 table analysed three ways
t = [[1, 39], [6, 32]]
print("q07 chi2", round(chi2_contingency(t, correction=False)[1], 3), "Fisher", round(fisher_exact(t).pvalue, 3), "Yates", round(chi2_contingency(t, correction=True)[1], 3))
# q03 / case q03: rare-outcome versions
print("q03 rare OR", round((0.04/0.96)/(0.025/0.975), 3), "| case rare OR", round((0.07/0.93)/(0.05/0.95), 3))
