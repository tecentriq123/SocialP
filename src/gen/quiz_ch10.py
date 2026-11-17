"""Numbers for the 10장 quiz (qz10). Hypothetical examples. Run: python3 gen/quiz_ch10.py"""
# q03 Greenhouse-Geisser: 60 patients, 2 groups, 4 visits
k, N, g, eps = 4, 60, 2, 0.70
df1, df2 = k-1, (k-1)*(N-g); print("q03 unadjusted df", df1, df2, "GG df", round(eps*df1, 2), round(eps*df2, 1))
# q08 ICC and design effect
sb, sw, m = 0.6, 0.8, 5
icc = sb**2/(sb**2+sw**2); de = 1+(m-1)*icc
print("q08 ICC", round(icc, 2), "design effect", round(de, 2), "effective measures from 100x5", round(100*m/de))
# case: RM ANOVA df with 100 completers, 4 visits, eps 0.72
k, N, g, eps = 4, 100, 2, 0.72
print("case GG df", round(eps*(k-1), 2), round(eps*(k-1)*(N-g), 1))
