"""Numbers for the 7장 quiz (qz07). Hypothetical examples. Run: python3 gen/quiz_ch07.py"""
# q02 incidence rate
print("q02 rate per 100 PY", 8/160*100, "| naive proportion", 8/40)
# q06 Kaplan-Meier by hand: (time, event) for 6 patients
data = [(2, 1), (4, 0), (5, 1), (7, 1), (9, 0), (12, 1)]
S = 1.0; at_risk = len(data)
for t, e in sorted(data):
    if e: S *= (at_risk - 1)/at_risk; print(f"q06 death at {t}: at risk {at_risk}, S = {S:.3f}")
    else: print(f"q06 censored at {t}")
    at_risk -= 1
print("q06 naive (censored alive) S(6)", round(4/6, 3), "| censored dropped", round(3/5, 3))
# q12 hazard ratio vs cumulative risk ratio under proportional hazards
SB = 0.40; HR = 0.70; SA = SB**HR
print("q12 S_A(3)", round(SA, 3), "risk A", round(1-SA, 3), "risk B", 1-SB, "RR", round((1-SA)/(1-SB), 3))
# case: rates and the 2-year risk ratio implied by HR 0.70 with S_B(2) = 0.33
print("case rates per 100 PY", round(60/180*100, 1), round(70/150*100, 1), "rate ratio", round((60/180)/(70/150), 3))
SA2 = 0.33**0.70; print("case S_A(2) implied", round(SA2, 3), "2-yr death risk A", round(1-SA2, 3), "B", 0.67, "RR", round((1-SA2)/0.67, 3))
print("case observed 2-yr death 53% vs 67% RR", round(0.53/0.67, 3))
# q12 condition: rare outcome version
SA = 0.97**0.70; print("q12 rare: death A", round(1-SA, 4), "B", 0.03, "ratio", round((1-SA)/0.03, 3))
print("case 2-yr survival ratio", round(0.47/0.33, 3))
