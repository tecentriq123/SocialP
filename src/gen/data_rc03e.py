"""Data files for R과 파이썬으로 하는 통계분석 'Cox 모형과 비례위험 점검' 장 e 절: the examples of 보건통계학 기초 11장.

run: python3 gen/data_rc03e.py → pub/data/ch11_cohort.csv (lib_ch11.rcc_cohort, 7장 코호트 300명 + 나이·성별·신장절제술·조직형),
     pub/data/ch11_bp.csv (lib_ch11.bp_demo, 가상의 혈압 코호트 3,000명, 그림 11-2)
Same random draws as gen/nums_ch11.py and gen/fig_ch11.py.
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import lib_ch11 as L
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pub", "data")


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(name, len(rows))


D = L.rcc_cohort()
IM = np.array(["favorable", "intermediate", "poor"])
write("ch11_cohort.csv", ["id", "time", "status", "drug", "age", "male", "imdc", "nephrectomy", "histology"],
      [[i + 1, round(float(D["time"][i]), 1), int(D["status"][i]), "A" if D["drug"][i] == 1 else "B", int(D["age"][i]),
        int(D["male"][i]), IM[D["imdc"][i]], int(D["neph"][i]), "nonclear" if D["ncc"][i] == 1 else "clear"]
       for i in range(len(D["time"]))])
B = L.bp_demo()
write("ch11_bp.csv", ["sbp", "time", "status"],
      [[f"{B['sbp'][i]:.4f}", f"{B['time'][i]:.5f}", int(B["status"][i])] for i in range(len(B["sbp"]))])
