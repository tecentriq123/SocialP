"""Numbers for the 9장 quiz (qz09). Hypothetical examples. Run: python3 gen/quiz_ch09.py"""
import math
# q01 CI built on the logit scale: OR 1.50, SE(beta) 0.20
b, se = math.log(1.5), 0.20
print("q01 CI", round(math.exp(b-1.96*se), 2), round(math.exp(b+1.96*se), 2), "| wrong symmetric", round(1.5-1.96*se, 2), round(1.5+1.96*se, 2))
# q07 events per coefficient: 45 events, coefficients 1+1+1+2+1+2+4+1
k = 1+1+1+2+1+2+4+1; print("q07 coefficients", k, "EPV", round(45/k, 1))
# case: development data
print("case EPV", 180/10, "| optimism", round(0.79-0.76, 2))
# crude OR/RR for the delirium cohort (illustration in the case intro)
a, n1, c, n0 = 72, 300, 108, 900
print("case crude RR", round((a/n1)/(c/n0), 2), "OR", round((a/(n1-a))/(c/(n0-c)), 2))
