"""Numbers for the 22장 quiz (qz22). Hypothetical examples. Run: python3 gen/quiz_ch22.py"""
def trap(times, utils):  # times in years
    return sum((times[i+1] - times[i])*(utils[i] + utils[i+1])/2 for i in range(len(times) - 1))
print("q03 TTO", 13/20)
print("q07 trapezoid %.4f | mean of three %.3f" % (trap([0, .25, 1], [.80, .60, .70]), (.80 + .60 + .70)/3))
print("q10 model QALY", 1.5*0.80 + 1.0*0.60 - 0.02, "| avg utility", (1.5 + 1.0)*0.70, "| no AE", 1.5*0.80 + 1.0*0.60)
qm, qn = trap([0, .5, 1], [.75, .78, .74]), trap([0, .5, 1], [.73, .72, .66])
print("case QALY M %.4f N %.4f diff %.4f | adjusted diff (beta 0.6 x baseline diff 0.02) %.3f" % (qm, qn, qm - qn, qm - qn - 0.6*0.02))
print("case wrong: 12-month only", .74 - .66, "| mean of 3 visits", round((.75+.78+.74)/3 - (.73+.72+.66)/3, 4))
