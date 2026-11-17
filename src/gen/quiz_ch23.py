"""Numbers for the 23장 quiz (qz23). Hypothetical examples. Run: python3 gen/quiz_ch23.py"""
import math
# q02: infection 0.20; among infected, hospitalized 0.25; costs (만원): none 10, outpatient 50, hospital 400
paths = [(0.80, 10), (0.20*0.75, 50), (0.20*0.25, 400)]
print("q02 path probs", [round(p, 3) for p, _ in paths], "sum", sum(p for p, _ in paths), "expected cost", sum(p*c for p, c in paths))
print("q02 wrong (0.25 as overall prob):", 0.80*10 + 0.20*50 + 0.25*400, "| simple mean of path costs", (10 + 50 + 400)/3)
# q07: 1-year progression probability 40% -> monthly
r = -math.log(1 - 0.40); pm = 1 - math.exp(-r/12); print("q07 annual rate %.4f monthly p %.4f | naive %.4f" % (r, pm, 0.40/12))
# q08: annual probability 0.50, HR 0.50
r8 = -math.log(0.5); p8 = 1 - math.exp(-r8*0.5); print("q08 treated annual p %.3f | naive %.3f" % (p8, 0.25))
# q09: 1-year cycles, alive at start of each year 100, 70, 40, 20, 0
alive = [100, 70, 40, 20, 0]
start, end = sum(alive[:-1]), sum(alive[1:]); print("q09 start-sum", start, "end-sum", end, "half-cycle", (start + end)/2)
