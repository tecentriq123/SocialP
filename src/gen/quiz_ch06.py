"""Numbers for the 6장 quiz (qz06). Hypothetical examples. Run: python3 gen/quiz_ch06.py"""
import math
import numpy as np
from scipy import stats
# q01 U-shaped relation gives r near 0 (illustration)
x = np.linspace(16, 38, 45); y = (x - 27)**2
print("q01 U-shape r", round(np.corrcoef(x, y)[0, 1], 3))
# q07 slope from r and SDs: BMI (SD 4) vs fasting insulin (SD 6), r = 0.5
r, sx, sy = 0.5, 4.0, 6.0; print("q07 slope", r*sy/sx, "R2", r**2, "| reversed slope", r*sx/sy)
# q09 mean CI vs prediction interval with residual SD 12 and n = 10,000
n, s = 10000, 12.0; print("q09 mean CI half-width ~", round(1.96*s/math.sqrt(n), 2), "PI half-width ~", round(1.96*s*math.sqrt(1+1/n), 1))
# q12 current vs former smokers: -5.4 - (-3.1)
print("q12 current - former", round(-5.4 - (-3.1), 1))
# case: r = -0.30 between steps and HbA1c, n = 300
r, n = -0.30, 300; t = r*math.sqrt(n-2)/math.sqrt(1-r**2)
print("case R2", round(r**2, 3), "t", round(t, 2), "P", f"{2*stats.t.sf(abs(t), n-2):.1e}")
# slope consistent with r: SD steps 2,500 (2.5 per 1,000), SD HbA1c 1.0 -> b = r*1.0/2.5 per 1,000 steps
print("case slope per 1,000 steps", round(r*1.0/2.5, 3))
# case: slope SE from t, and its 95% CI
b = r*1.0/2.5; se = abs(b/t); tc = stats.t.ppf(.975, n-2)
print("case slope CI", round(b - tc*se, 3), round(b + tc*se, 3))
