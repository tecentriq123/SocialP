"""Numbers for the 연구 설계 정리 quiz (qz83). Hypothetical examples. Run: python3 gen/quiz_design.py"""
import math
# q07: closed cohort, 1 year. exposed 1,000 (risk 30%), unexposed 4,000 (risk 15%)
n1, n0, r1, r0 = 1000, 4000, 0.30, 0.15
c1, c0 = n1*r1, n0*r0
print("cases", c1, c0, "total", c1 + c0)
rr = r1/r0; risk_or = (r1/(1-r1))/(r0/(1-r0))
irr = (-math.log(1-r1))/(-math.log(1-r0))          # constant hazard within the year
print(f"true RR {rr:.2f}  risk OR {risk_or:.2f}  IRR(constant hazard) {irr:.2f}")
# case-base sampling: 2,000 controls from the baseline cohort of 5,000
k = 2000; e_ctrl = k*n1/(n1+n0)
case_odds = c1/c0; ctrl_odds = e_ctrl/(k-e_ctrl)
print(f"controls exposed {e_ctrl:.0f}/{k}; case odds {case_odds:.3f} control odds {ctrl_odds:.3f} OR {case_odds/ctrl_odds:.2f}")
print(f"distractor proportion ratio {(c1/(c1+c0))/(e_ctrl/k):.2f}; inverted {ctrl_odds/case_odds:.2f}")
# cumulative sampling for comparison: non-cases at end
print(f"cumulative-sampling OR {case_odds/((n1-c1)/(n0-c0)):.2f}")
# q08: cross-sectional prevalence 45% vs 30%
p1, p0 = 0.45, 0.30
print(f"q08 PR {p1/p0:.2f} POR {(p1/(1-p1))/(p0/(1-p0)):.2f}")
# q11 (collider) is conceptual; case study numbers are hypothetical HRs (no calculation)
