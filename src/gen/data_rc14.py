"""Data files for R과 파이썬으로 하는 통계분석의 '로지스틱 회귀와 포아송 회귀' 장: the examples of 보건통계학 기초 8·9·12장.

run: python3 gen/data_rc14.py → pub/data/ch08_readmit.csv (lib_ch08.make_readmit, 1,500명 퇴원 코호트),
     ch09_cohort.csv (lib_ch09.cohort, 2,400명), ch09_external.csv (lib_ch09.external, 1,600명),
     ch09_vanco.csv (lib_ch09.vanco, 150명), ch12_copd.csv (lib_ch12.cohort, 2,400명)
Same random draws as the chapter scripts (gen/nums_ch08.py, nums_ch09.py, pyout_ch09.py, nums_ch12.py, pyout_ch12.py).
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pub", "data")


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(name, len(rows))


import lib_ch08 as L8
d = L8.make_readmit()
CCI8 = np.array(["0", "1-2", "3-4", "5+"])
write("ch08_readmit.csv", ["readm", "poly", "age", "female", "cci", "medaid"],
      [[int(d["readm"][i]), int(d["poly"][i]), int(d["age"][i]), int(d["female"][i]), CCI8[d["cci"][i]], int(d["medaid"][i])]
       for i in range(d["n"])])

import lib_ch09 as L9
CCI9 = np.array(["0", "1-2", ">=3"])
for name, c in (("ch09_cohort.csv", L9.cohort()), ("ch09_external.csv", L9.external())):
    write(name, ["hosp", "sglt2", "age", "female", "cci", "prior", "hf", "egfr"],
          [[int(c["hosp"][i]), int(c["sglt2"][i]), int(c["age"][i]), int(c["female"][i]), CCI9[c["cci"][i]],
            int(c["prior"][i]), int(c["hf"][i]), int(c["egfr"][i])] for i in range(len(c["age"]))])
tr, aki = L9.vanco()
write("ch09_vanco.csv", ["trough", "aki"], [[f"{tr[i]:.1f}", int(aki[i])] for i in range(len(tr))])

import lib_ch12 as L12
c = L12.cohort()
write("ch12_copd.csv", ["exac", "drug", "agegrp", "female", "cci", "days", "py"],
      [[int(c["y"][i]), "A" if c["drug"][i] == 1 else "B", L12.AGE_LABELS[c["age"][i]], int(c["female"][i]), int(c["cci"][i]),
        int(c["days"][i]), repr(float(c["py"][i]))] for i in range(c["n"])])
