"""Data file for R과 파이썬으로 하는 통계분석 5장 e 절: the 300-patient RCC cohort of 보건통계학 기초 7장.

run: python3 gen/data_rc02e.py  → pub/data/ch07_cohort.csv (gen/lib_ch07.py cohort(2), as used by gen/nums_ch07.py)
columns: time (months, 0.1), status (1 = death), drug (A/B), imdc (favorable/intermediate/poor), reason (death/lost/admin)
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_ch07 import cohort
C = cohort(2)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pub", "data", "ch07_cohort.csv")
lv = ["favorable", "intermediate", "poor"]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["time", "status", "drug", "imdc", "reason"])
    for i in range(len(C["time"])):
        w.writerow([f"{C['time'][i]:.1f}", int(C["status"][i]), "A" if C["drug"][i] == 1 else "B", lv[C["imdc"][i]], C["reason"][i]])
print("ch07_cohort.csv", len(C["time"]))
