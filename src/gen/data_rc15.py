"""Data file for R과 파이썬으로 하는 통계분석의 '반복측정 자료: 혼합모형과 GEE' 장: the example of 보건통계학 기초 10장.

run: python3 gen/data_rc15.py → pub/data/ch10_hba1c.csv (lib_ch10.simulate(1436), 100명 넓은 형식:
id, group(중재군 = 1), hba1c_0, hba1c_3, hba1c_6, hba1c_12, adh_3, adh_6, adh_12; 측정하지 못한 칸은 빈칸)
adh_3·adh_6·adh_12는 0–3, 3–6, 6–12개월의 PDC ≥ 80% 여부. Same random draw as gen/nums_ch10.py (SEED 1436).
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from lib_ch10 import simulate
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pub", "data")
D = simulate(1436)
f = lambda v, fmt: "" if np.isnan(v) else fmt % v
with open(os.path.join(OUT, "ch10_hba1c.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "group", "hba1c_0", "hba1c_3", "hba1c_6", "hba1c_12", "adh_3", "adh_6", "adh_12"])
    for i in range(D["N"]):
        w.writerow([i + 1, int(D["g"][i])] + [f(v, "%.1f") for v in D["y"][i]] + [f(v, "%d") for v in D["adh"][i]])
print("ch10_hba1c.csv", D["N"])
