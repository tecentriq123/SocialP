"""Numbers for the 21장 quiz (qz21). Hypothetical examples (만원). Run: python3 gen/quiz_ch21.py"""
# q02: 4 patients, 1-year window; P1 claims 150, 200 within window and 80 on day 380; P2 no claims; P3 500; P4 120, 30
pts = {"P1": [150, 200], "P2": [], "P3": [500], "P4": [120, 30]}
tot = {k: sum(v) for k, v in pts.items()}
print("q02 per patient", tot, "mean", sum(tot.values())/4, "| excl. no-claim", round(sum(tot.values())/3, 1), "| incl. day-380", (sum(tot.values()) + 80)/4)
# q03: progressed state: A 2 months 600, B 10 months 1,000, C 8 months 800
m = [(2, 600), (10, 1000), (8, 800)]
ratio_of_sums = sum(c for _, c in m)/sum(t for t, _ in m); mean_of_ratios = sum(c/t for t, c in m)/3
print("q03 ratio of sums", ratio_of_sums, "mean of ratios %.1f" % mean_of_ratios, "model total with mean-of-ratios %.0f vs actual %d" % (mean_of_ratios*20, 2400))
# q04: price year: index 2020 = 100, 2025 = 108 (assumed) ; wrong: discount 4.5% for 5 years
print("q04 index-adjusted", 1000*108/100, "wrong discount-based", round(1000/1.045**5), "or inflate", round(1000*1.045**5))
# q05: listed price 1,000/year, 20% refund -> 800; incremental cost 600 at listed price
print("q05 incremental at net price", 600 - 200)
# q06: 500 patients, mean 300, median 180
print("q06 budget mean", 500*300, "median", 500*180)
# q12: GLM e^b 1.20; observed D-group mean 2,500; standardized all-B 2,100, all-A 2,520
print("q12 standardized diff", 2520 - 2100, "ratio", 2520/2100, "wrong 0.20*2500", 0.20*2500)
# case: C (900) mean 1,850 median 1,320; D (1,500) mean 1,620 median 1,180; standardized C 1,903, D 1,560
print("case unadjusted", 1850 - 1620, "median diff", 1320 - 1180, "ratio of means %.2f" % (1850/1620), "standardized", 1903 - 1560, "ratio %.2f" % (1903/1560), "0.22*1620", round(0.22*1620))
