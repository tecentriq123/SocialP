"""Numbers for the 25장 quiz (qz25). Hypothetical examples (만원). Run: python3 gen/quiz_ch25.py"""
# q04: results table check
print("q04 ICER", (8450 - 6150)/(3.20 - 2.80))
# q11: budget impact, 1,000 new patients/yr; share 20% then 40%; extra 1,000 in start year, 600 in 2nd year (all continue)
y1 = 1000*0.20*1000; y2 = 1000*0.40*1000 + 1000*0.20*600
print("q11 year1 %.0f만원 (%.0f억) year2 %.0f만원 (%.0f억) | new only %.0f억 | cumulative %.0f억" % (y1, y1/1e4, y2, y2/1e4, 1000*0.40*1000/1e4, (y1 + y2)/1e4))
# case: Z (9,800; QALY 2.90; drug 5,200) vs Y (7,000; 2.45; drug 2,700)
dc, dq = 9800 - 7000, 2.90 - 2.45; icer = dc/dq
print("case ICER %.0f drug share %.1f%%" % (icer, 100*(5200 - 2700)/dc))
need = dc - 5000*dq; print("case cut needed", need, "refund on Z drug %.1f%%" % (100*need/5200), "| wrong 1-5000/ICER %.1f%%" % (100*(1 - 5000/icer)), "| on incr drug %.1f%%" % (100*need/2500))
