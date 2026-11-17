"""Numbers for the 16장 quiz (qz16). Hypothetical examples. Run: python3 gen/quiz_ch16.py"""
import math
# q02/q03: 2,000 newly diagnosed, death rate 40/100 PY for everyone, drug no effect; 500 start at 3 months.
# person-years are rounded approximations (as in the chapter's own toy table)
fut_pre_py, fut_pre_d = 500*0.25, 0          # future users, 0-3 months (immortal time)
oth_pre_py = 1500*0.25                       # others, 0-3 months
oth_pre_d = round((fut_pre_py + oth_pre_py)*0.40)   # all early deaths fall among others
use_py, use_d = 500*0.75, round(500*0.75*0.40)
non_alive = 1500 - oth_pre_d; non_py = non_alive*0.75; non_d = round(non_py*0.40)
print("table:", fut_pre_py, fut_pre_d, oth_pre_py, oth_pre_d, use_py, use_d, non_alive, non_py, non_d)
r = lambda d, py: 100*d/py
naive = r(use_d, use_py + fut_pre_py) / r(oth_pre_d + non_d, oth_pre_py + non_py)
drop = r(use_d, use_py) / r(oth_pre_d + non_d, oth_pre_py + non_py)
move = r(use_d, use_py) / r(oth_pre_d + non_d, oth_pre_py + non_py + fut_pre_py)
print("rates naive users %.1f non %.1f" % (r(use_d, use_py + fut_pre_py), r(oth_pre_d + non_d, oth_pre_py + non_py)))
print("IRR naive %.2f drop-only %.2f move-to-unexposed %.2f" % (naive, drop, move))
print("distractor: users naive rate / true rate = %.2f" % (r(use_d, use_py + fut_pre_py) / 40))
# q07: long format check: start drug day 90, die day 250
rows = [(0, 90, 0, 0), (90, 250, 1, 1)]; print("q07 rows", rows, "days", sum(b - a for a, b, _, _ in rows))
# q11: depletion of susceptibles. Drug V: 10% susceptible with 50/100 PY, others 4/100 PY (= comparator); stop at event
def year(nv, no):
    ev_v, ev_o = nv*(1 - math.exp(-0.50)), no*(1 - math.exp(-0.04))
    py = ev_v/0.50 + ev_o/0.04; rate = 100*(ev_v + ev_o)/py
    return rate, rate/4, nv - ev_v, no - ev_o
nv, no = 100.0, 900.0
for y in (1, 2, 3):
    share = nv/(nv + no); rate, irr, nv, no = year(nv, no)
    print(f"q11 year {y}: susceptible share {share:.1%} rate {rate:.1f}/100PY IRR {irr:.2f}")
# q12 protopathic (hypothetical): exposed cancers 90, 54 within 12 months of first laxative prescription
print("q12 share in first year", 54/90)
# case: hip-fracture surgery cohort 4,000; 360 died before 6-month landmark
print("case landmark n", 4000 - 360)
