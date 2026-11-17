"""Numbers for the 3장 quiz (qz03). Hypothetical examples. Run: python3 gen/quiz_ch03.py"""
import math
import numpy as np
from scipy import stats
# q02 paired t: 16 patients, mean change -4.0, SD of change 6.0
se = 6.0/math.sqrt(16); t = -4.0/se; print("q02 SE", se, "t", round(t, 2), "df 15 P", round(2*stats.t.sf(abs(t), 15), 4))
# q03 SD of change from time-point SDs and correlation
for r in (0, 0.5, 0.75, 0.9): print("q03 r", r, "SD_d", round(math.sqrt(12**2+12**2-2*r*12*12), 2))
# q04 paired vs independent analysis of pre/post: n 20, 140±15 -> 134±15, SD of change 8
n = 20; d = -6.0; sp = 8.0/math.sqrt(n); si = math.sqrt(15**2/n+15**2/n)
print("q04 paired SE", round(sp, 2), "t", round(d/sp, 2), "P", round(2*stats.t.sf(abs(d/sp), n-1), 4),
      "| independent SE", round(si, 2), "t", round(d/si, 2), "P", round(2*stats.t.sf(abs(d/si), 2*n-2), 3),
      "| implied r", round((15**2+15**2-8**2)/(2*15*15), 3))
# q08 ANCOVA arithmetic: follow-up diff -2.0, baseline diff +1.0, within-group slope 0.6
for b, lab in ((0, "follow-up only"), (1, "change score"), (0.6, "ANCOVA")): print("q08", lab, -2.0 - b*1.0)
# q10 signed-rank by hand
dd = np.array([-5, -2, 1, -3, 0, -4, 2]); nz = dd[dd != 0]; rk = stats.rankdata(np.abs(nz))
print("q10 nonzero", nz, "ranks", rk, "W+", rk[nz > 0].sum(), "W-", rk[nz < 0].sum(), "expected", len(nz)*(len(nz)+1)/4)
print("    scipy", stats.wilcoxon(nz))
# q11 median of change vs change of medians
pre = np.array([2, 5, 8]); post = np.array([6, 1, 7])
print("q11 changes", post-pre, "median change", np.median(post-pre), "medians", np.median(pre), np.median(post), "change of medians", np.median(post)-np.median(pre))
# case: single-arm FPG, n 30, 152±15 -> 141±18, change -11 ± 14 (SD)
n = 30; d = -11; se = 14/math.sqrt(n); tc = stats.t.ppf(.975, n-1)
print("case paired SE", round(se, 3), "t", round(d/se, 2), "tcrit", round(tc, 3), "CI", round(d-tc*se, 1), round(d+tc*se, 1), "P", f"{2*stats.t.sf(abs(d/se), n-1):.1e}")
si = math.sqrt(15**2/n+18**2/n); tc2 = stats.t.ppf(.975, 2*n-2)
print("case independent SE", round(si, 2), "CI", round(d-tc2*si, 1), round(d+tc2*si, 1), "P", round(2*stats.t.sf(abs(d/si), 2*n-2), 3))
print("case implied r", round((15**2+18**2-14**2)/(2*15*18), 3))
