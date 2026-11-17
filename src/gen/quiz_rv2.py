"""Numbers for 종합 2 (qz92, claims cohort). Hypothetical RA cohort, biologic J vs T, venous thromboembolism. Run: python3 gen/quiz_rv2.py"""
import math
# c1: J 42 events / 6,000 PY ; T 60 / 12,000 PY
rj, rt = 42/6000*1000, 60/12000*1000; print("c1 rates /1000PY J %.1f T %.1f IRR %.2f IRD %.1f | event proportion J %.2f%% T %.2f%%" % (rj, rt, rj/rt, rj - rt, 100*42/3000, 100*60/5000))
# b3: ATT weights: T user with PS 0.25 -> 0.25/0.75 ; J user -> 1 ; ATE weight for that T user 1/(1-0.25)
print("b3 ATT weight T(PS .25) %.2f | ATE weight %.2f | J(PS .25) ATE %.1f" % (0.25/0.75, 1/0.75, 1/0.25))
# d1: E-values for HR 1.35 (1.02-1.79), rare outcome
ev = lambda r: r + math.sqrt(r*(r - 1)); print("d1 E-value point %.2f CI-lower %.2f" % (ev(1.35), ev(1.02)))
# a3: outcome misclassification: true RR 1.40, true risks 1.0% vs 0.714%; definition with PPV-ish false positives 0.4% in both groups
t1, t0 = 0.010, 0.010/1.40; fp = 0.004
print("a3 observed RR with nondifferential false positives %.2f (true 1.40) | RD unchanged %.2f%%p" % ((t1 + fp)/(t0 + fp), 100*(t1 - t0)))
