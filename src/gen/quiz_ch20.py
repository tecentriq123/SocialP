"""Numbers for the 20장 quiz (qz20). Hypothetical examples (만원, QALY). Run: python3 gen/quiz_ch20.py"""
# q03: P (6,300; 3.10) vs Q (4,500; 2.70)
dc, de = 6300 - 4500, 3.10 - 2.70
print("q03 ICER", dc/de, "avg P", round(6300/3.10), "avg Q", round(4500/2.70), "diff of avgs", round(6300/3.10 - 4500/2.70), "cost/QALY_P", round(dc/3.10))
# q04: incremental cost 1,500; LY 0.50; QALY 0.30
print("q04 per LY", 1500/0.50, "per QALY", 1500/0.30)
# q09: 0.20 QALY 8 years later at 4.5%
f = 1/1.045**8; print("q09 factor %.3f PV %.3f | simple %.3f | 1-year %.3f" % (f, 0.20*f, 0.20*(1 - 0.045*8), 0.20/1.045))
# q11: threshold 3,000; dQALY 0.40; dCost 1,500
print("q11 NMB", 3000*0.40 - 1500, "NHB", 0.40 - 1500/3000, "ICER", 1500/0.40)
# q12: L (2,000; 1.00), M (3,200; 1.10), N (4,000; 1.60)
print("q12 M vs L", (3200 - 2000)/(1.10 - 1.00), "N vs M", (4000 - 3200)/(1.60 - 1.10), "N vs L", round((4000 - 2000)/(1.60 - 1.00)))
# q13: comparator 0; F (+100; +0.05), G (+2,000; +0.50); threshold 5,000
print("q13 F vs comp", 100/0.05, "G vs comp", 2000/0.50, "G vs F", round((2000 - 100)/(0.50 - 0.05)), "NMB F", 5000*0.05 - 100, "NMB G", 5000*0.50 - 2000)
# case: R (4,860; QALY 6.42; LY 8.10) vs S (3,960; 6.12; 7.70); old drug T (4,300; 5.90); threshold 4,000 assumed
print("case ICER/QALY", round((4860 - 3960)/(6.42 - 6.12)), "ICER/LY", round((4860 - 3960)/(8.10 - 7.70)), "R vs T", round((4860 - 4300)/(6.42 - 5.90)), "NMB@4000", 4000*0.30 - 900, "NMB@2500", 2500*0.30 - 900)
