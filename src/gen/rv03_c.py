"""Numbers for 연구 설계 정리 c 절: what a case-control odds ratio estimates under three control-sampling schemes,
and prevalence vs incidence. Closed cohort, constant hazards over 1 year. Run: python3 gen/rv03_c.py"""
import math

def schemes(n1, n0, r1, r0):
    l1, l0 = -math.log(1 - r1), -math.log(1 - r0)          # constant hazards giving 1-year risks r1, r0
    c1, c0 = n1 * r1, n0 * r0                              # cases
    pt1, pt0 = n1 * r1 / l1, n0 * r0 / l0                  # person-time (expected) under constant hazard
    case_odds = c1 / c0
    out = {
        "RR": r1 / r0, "risk OR": (r1 / (1 - r1)) / (r0 / (1 - r0)), "IRR": l1 / l0,
        "cumulative": case_odds / ((n1 - c1) / (n0 - c0)),
        "case-cohort": case_odds / (n1 / n0),
        "risk-set": case_odds / (pt1 / pt0),
    }
    print(f"cohort {n1}/{n0}, risks {r1:.0%}/{r0:.0%}: cases {c1:.0f}/{c0:.0f}, case exposure odds {case_odds:.3f}")
    print(f"  control exposure odds: cumulative {(n1-c1)/(n0-c0):.4f}  case-cohort {n1/n0:.4f}  risk-set PT {pt1:.1f}/{pt0:.1f} = {pt1/pt0:.4f}")
    print("  " + "  ".join(f"{k} {v:.3f}" for k, v in out.items()))

schemes(2000, 8000, 0.20, 0.10)   # common outcome: three ORs differ
schemes(5000, 5000, 0.02, 0.01)   # rare outcome: all close to 2

# prevalence ~ incidence x duration (steady state): prevalence odds = I x D
for name, I, D in (("A", 0.02, 1.0), ("B", 0.01, 3.0)):
    o = I * D; print(f"{name}: incidence {I}/PY duration {D}y -> prevalence odds {o:.3f} prevalence {o/(1+o):.4f}")
pa, pb = 0.02 / 1.02, 0.03 / 1.03
print(f"prevalence ratio A/B {pa/pb:.3f}, prevalence OR {0.02/0.03:.3f}, incidence ratio {0.02/0.01:.1f}")
# common outcome: PR vs POR
p1, p0 = 0.40, 0.25
print(f"PR {p1/p0:.2f}  POR {(p1/(1-p1))/(p0/(1-p0)):.2f}")
