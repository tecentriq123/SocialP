"""Numbers for the 19장 quiz (qz19). Hypothetical examples. Run: python3 gen/quiz_ch19.py"""
import math
from scipy import stats
# q03: fixed-effect inverse-variance pooling of two trials on the log-OR scale
la, sa, lb, sb = -0.20, 0.10, -0.50, 0.20
wa, wb = 1/sa**2, 1/sb**2; pooled = (wa*la + wb*lb)/(wa + wb); se = 1/math.sqrt(wa + wb)
print("q03 weights", wa, wb, "pooled log", round(pooled, 3), "OR %.2f (%.2f-%.2f)" % (math.exp(pooled), math.exp(pooled - 1.96*se), math.exp(pooled + 1.96*se)))
print("q03 distractors: mean of ORs %.2f, unweighted log mean %.2f" % ((math.exp(la) + math.exp(lb))/2, math.exp((la + lb)/2)))
# q04: summing events across trials (Simpson)
odds = lambda e, n: e/(n - e)
print("q04 trial ORs", odds(200, 1000)/odds(100, 500), odds(10, 500)/odds(20, 1000), "summed OR %.2f" % (odds(210, 1500)/odds(120, 1500)), "risks %.1f%% vs %.1f%%" % (100*210/1500, 100*120/1500))
# q07: prediction interval from OR 0.80 (0.70-0.91), tau 0.20, k = 12
se7 = (math.log(0.91) - math.log(0.70))/(2*1.96); h = stats.t.ppf(.975, 10)*math.sqrt(se7**2 + 0.20**2)
print("q07 PI %.2f-%.2f" % (math.exp(math.log(0.80) - h), math.exp(math.log(0.80) + h)))
# q12: OR 0.75 applied to baseline 12%
def apply(orr, p):
    o = p/(1 - p)*orr; return o/(1 + o)
r = apply(0.75, 0.12); print("q12 treated risk %.2f%%, RD %.2f%%p, NNT %.0f | naive 0.12*0.75 = %.1f%% NNT %.0f" % (100*r, 100*(r - .12), 1/(.12 - r), 100*.12*.75, 1/(.12 - .09)))
# q13: NNT 40 / NNH 60 per 1,000
print("q13 per 1000:", 1000/40, 1000/60)
# case: 8 RCTs, RE OR 0.78 (0.66-0.92) HKSJ, tau2 0.03; baseline risk 15% (1 year)
k = 8; sec = (math.log(0.92) - math.log(0.66))/(2*stats.t.ppf(.975, k - 1)); hc = stats.t.ppf(.975, k - 2)*math.sqrt(sec**2 + 0.03)
print("case PI %.2f-%.2f" % (math.exp(math.log(0.78) - hc), math.exp(math.log(0.78) + hc)))
rc = apply(0.78, 0.15); lo, hi = apply(0.66, 0.15), apply(0.92, 0.15)
print("case treated %.2f%% RD %.2f%%p per1000 %.0f (%.0f-%.0f) NNT %.0f (%.0f-%.0f)" % (100*rc, 100*(rc - .15), 1000*(.15 - rc), 1000*(.15 - hi), 1000*(.15 - lo), 1/(.15 - rc), 1/(.15 - lo), 1/(.15 - hi)))
