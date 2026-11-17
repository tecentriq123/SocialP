"""Numbers for the 12장 quiz (qz12). Hypothetical examples. Run: python3 gen/quiz_ch12.py"""
import math
from scipy import stats
def exact_ci(d, py, per=1000):
    lo = stats.chi2.ppf(.025, 2*d)/2 if d > 0 else 0.0; hi = stats.chi2.ppf(.975, 2*(d+1))/2
    return round(lo/py*per, 2), round(hi/py*per, 2)
# q01 rates and IRR
print("q01 rates", 18/4500*1000, 30/3000*1000, "IRR", round((18/4500)/(30/3000), 2), "IRD", 18/4500*1000-30/3000*1000)
# q02 zero events in 2,000 PY
print("q02 exact 95% CI for 0/2000 PY", exact_ci(0, 2000), "| rule of three upper", 3/2000*1000)
# q03 age-stratified IRRs (illustration)
young = (12, 4000, 10, 2800); old = (30, 2000, 90, 5300)
for lab, (a, pa, b, pb) in (("young", young), ("old", old)):
    print("q03", lab, "rates", round(a/pa*1000, 2), round(b/pb*1000, 2), "IRR", round((a/pa)/(b/pb), 2))
A = young[0]+old[0]; PA = young[1]+old[1]; B = young[2]+old[2]; PB = young[3]+old[3]
print("q03 crude IRR", round((A/PA)/(B/PB), 2))
num = sum(a*pb/(pa+pb) for a, pa, b, pb in (young, old)); den = sum(b*pa/(pa+pb) for a, pa, b, pb in (young, old))
print("q03 MH IRR", round(num/den, 2))
# q08 quasi-Poisson widening: IRR 0.80 (0.70-0.91), dispersion 2.25
se = math.log(0.91/0.70)/(2*1.96); se2 = se*math.sqrt(2.25)
print("q08 se", round(se, 4), "->", round(se2, 4), "CI", round(math.exp(math.log(0.8)-1.96*se2), 2), round(math.exp(math.log(0.8)+1.96*se2), 2))
# q10 absolute differences for IRR 0.80
for base in (50, 2): print("q10 base per 100 PY", base, "reduction", round(base*0.2, 1))
# case: falls
print("case rates", round(260/520*1000), round(450/1500*1000), "crude IRR", round((260/520)/(450/1500), 2))
