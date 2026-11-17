"""Numbers for the 0장 quiz (qz00). Hypothetical examples. Run: python3 gen/quiz_ch00.py"""
import math, statistics as st
from scipy import stats
x = [4, 6, 8, 10, 12]
print("Q5 mean", st.mean(x), "SS", sum((v - st.mean(x))**2 for v in x), "var(n-1)", st.variance(x), "SD", round(st.stdev(x), 2), "SD(n)", round(st.pstdev(x), 2))
los = [2, 3, 3, 4, 5, 6, 40]
print("Q6 mean", round(st.mean(los), 2), "median", st.median(los), "SD", round(st.stdev(los), 1), "4 of 7 <= mean?", sum(v <= st.mean(los) for v in los))
z = (180 - 130) / 25
print("Q7 z", z, "P(>)", round(stats.norm.sf(z), 4))
print("Q8 120 - 1.96*200 =", 120 - 1.96 * 200)
print("Q9 SE", 12 / math.sqrt(36))
print("Q10 SE diff", math.sqrt(1.5**2 + 2.0**2))
se = (math.log(1.60) - math.log(0.98)) / (2 * 1.959964)
print("Q13 RR 1.25 (0.98-1.60): z", round(math.log(1.25) / se, 3), "P", round(2 * stats.norm.sf(math.log(1.25) / se), 3))
# Case: LDL change (8wk - baseline), 25 vs 25
m1, s1, m0, s0, n = -32, 15, -20, 16, 25
se1, se0 = s1 / math.sqrt(n), s0 / math.sqrt(n); sed = math.sqrt(se1**2 + se0**2)
sp = math.sqrt(((n-1)*s1**2 + (n-1)*s0**2) / (2*n-2)); se_pool = sp * math.sqrt(2/n)
t48 = stats.t.ppf(0.975, 48); d = m1 - m0
print("Case SE", se1, se0, "SE diff", round(sed, 3), "pooled SE", round(se_pool, 3), "t48", round(t48, 3))
print("  CI t", round(d - t48*sed, 1), round(d + t48*sed, 1), " CI z", round(d - 1.96*sed, 1), round(d + 1.96*sed, 1),
      " P", round(2*stats.t.sf(abs(d/sed), 48), 4), " wrong SD-based", round(d - t48*15.5, 1), round(d + t48*15.5, 1), " SE-sum", round(d - t48*(se1+se0), 1), round(d + t48*(se1+se0), 1))
print("t df9", round(stats.t.ppf(0.975, 9), 3))
