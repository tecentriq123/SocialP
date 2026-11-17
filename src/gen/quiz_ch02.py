"""Numbers for the 2장 quiz (qz02). Hypothetical examples. Run: python3 gen/quiz_ch02.py"""
import math
from scipy import stats
def welch(m1, s1, n1, m0, s0, n0):
    a, b = s1**2/n1, s0**2/n0; se = math.sqrt(a+b); df = (a+b)**2/(a**2/(n1-1)+b**2/(n0-1))
    t = (m1-m0)/se; tc = stats.t.ppf(.975, df)
    return dict(se=se, t=t, df=df, p=2*stats.t.sf(abs(t), df), lo=(m1-m0)-tc*se, hi=(m1-m0)+tc*se, tc=tc)
def student(m1, s1, n1, m0, s0, n0):
    sp2 = ((n1-1)*s1**2+(n0-1)*s0**2)/(n1+n0-2); se = math.sqrt(sp2*(1/n1+1/n0)); df = n1+n0-2
    t = (m1-m0)/se; tc = stats.t.ppf(.975, df)
    return dict(sp2=sp2, se=se, t=t, df=df, p=2*stats.t.sf(abs(t), df), lo=(m1-m0)-tc*se, hi=(m1-m0)+tc*se)
r = lambda d: {k: round(v, 3) for k, v in d.items()}
print("q02 t", r(welch(12.0, 5.0, 25, 9.0, 4.0, 25)), " summed-SE t", round(3/(5/5+4/5), 2), " SD t", round(3/4.5, 2))
print("q04 Student", r(student(0, 5, 60, 0, 15, 20)), " Welch", r(welch(0, 5, 60, 0, 15, 20)))
# q07 CI bars overlap: means 10 vs 13, each 95% CI half-width 2.0, n 50
tc49 = stats.t.ppf(.975, 49); se = 2.0/tc49
print("q07 SE each", round(se, 3), r(welch(13, se*math.sqrt(50), 50, 10, se*math.sqrt(50), 50)))
# q08 large observational: 5000 per group, HbA1c diff 0.05, SD 1.0
print("q08", r(welch(7.25, 1.0, 5000, 7.20, 1.0, 5000)))
# q11 U by hand
A = [3, 5, 9, 12]; B = [7, 10, 14]
UA = sum((a > b) + 0.5*(a == b) for a in A for b in B); print("q11 U_A", UA, "pairs", len(A)*len(B), "prob", UA/(len(A)*len(B)))
print("   scipy", stats.mannwhitneyu(A, B, alternative="two-sided", method="exact"))
ranks = stats.rankdata(A+B); print("   rank sum A", ranks[:4].sum())
# case: 6-minute walk distance change, app 20 vs usual 40
w = welch(45, 60, 20, 20, 25, 40); s = student(45, 60, 20, 20, 25, 40)
print("case Welch", r(w)); print("case Student", r(s))
print("case summed SE", round(60/math.sqrt(20)+25/math.sqrt(40), 2))
