"""Numbers for the 11장 quiz (qz11). Hypothetical examples. Run: python3 gen/quiz_ch11.py"""
import math
# q02 combining units: coef 0.0262 per year
b = 0.0262; print("q02 HR/10y", round(math.exp(10*b), 3), "80 vs 60", round(math.exp(20*b), 3), "| additive wrong", round(2*math.exp(10*b), 2))
# q03 wrong shortcut vs explanation: control KM 3-yr survival 0.50 raised to adjusted HR 0.80
print("q03 shortcut", round(0.5**0.8, 3))
# q08 SE(log HR) ~ sqrt(1/d1 + 1/d0), HR 0.80
for d in (25, 250):
    se = math.sqrt(2/d); print("q08 events", 2*d, "SE", round(se, 3), "CI", round(math.exp(math.log(0.8)-1.96*se), 2), round(math.exp(math.log(0.8)+1.96*se), 2))
# q10 CI from coef and se
c, se = -0.223, 0.08; print("q10 HR", round(math.exp(c), 3), "CI", round(math.exp(c-1.96*se), 2), round(math.exp(c+1.96*se), 2), "| wrong symmetric", round(math.exp(c)-1.96*se, 2), round(math.exp(c)+1.96*se, 2), "z", round(c/se, 2))
# case: crude HR from rates (illustration) 240/8,000 vs 300/6,000 per person (equal follow-up assumed)
print("case crude risk ratio", round((240/8000)/(300/6000), 2))
# q10 reversed coding
print("q10 reversed HR", round(math.exp(0.223), 2), "CI", round(math.exp(0.223-1.96*0.08), 2), round(math.exp(0.223+1.96*0.08), 2))
