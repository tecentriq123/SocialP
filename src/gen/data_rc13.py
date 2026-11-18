"""Data files for R과 파이썬으로 하는 통계분석 4장 (상관과 회귀): the examples of 보건통계학 기초 6장.

run: python3 gen/data_rc13.py   → pub/data/ch06_bp100.csv, ch06_htn240.csv, ch06_cost400.csv, ch06_hba1c72.csv
The values come from gen/nums_ch06.py (bp100, htn240, cost400) and gen/nums_ch06e.py (72-patient HbA1c trial),
so the numbers in the R chapter equal those of 6장. BMI is written with 10 decimals (it is weight / height²).
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import nums_ch06 as N
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pub", "data")


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(name, len(rows))


age, sbp = N.bp100()
write("ch06_bp100.csv", ["age", "sbp"], [[int(a), int(s)] for a, s in zip(age, sbp)])

d = N.htn240()
inst = np.array(N.INST)[d["inst"]]
write("ch06_htn240.csv", ["sbp6", "classes", "sbp0", "age", "female", "diabetes", "weight", "bmi", "adherent", "inst"],
      [[int(d["fu"][i]), int(d["cls"][i]), int(d["base"][i]), int(d["age"][i]), int(d["fem"][i]), int(d["dm"][i]),
        f"{d['weight'][i]:.1f}", f"{d['bmi'][i]:.10f}", int(d["adh"][i]), inst[i]] for i in range(d["n"])])

c = N.cost400()
write("ch06_cost400.csv", ["cost", "adherent", "age", "cci"],
      [[f"{c['cost'][i]:.1f}", int(c["adh"][i]), int(c["age"][i]), int(c["cci"][i])] for i in range(c["n"])])

import importlib.util
spec = importlib.util.spec_from_file_location("e", os.path.join(os.path.dirname(os.path.abspath(__file__)), "nums_ch06e.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
write("ch06_hba1c72.csv", ["arm", "hba1c_0", "hba1c_6m"],
      [[int(E.arm[i]), f"{E.base[i]:.2f}", f"{E.post[i]:.2f}"] for i in range(len(E.arm))])
