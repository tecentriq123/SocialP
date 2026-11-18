"""Data files for R과 파이썬으로 하는 통계분석 '동등성과 비열등성 검정' 장: the examples of 보건통계학 기초 13장.

run: python3 gen/data_rc16.py → pub/data/ch13_pdc.csv (원격 상담 비열등성 시험의 환자별 PDC, ITT 304 + 304명;
     gen/pyout_ch13.py와 같은 seed 1302, 평균·SD가 13장 요약값과 정확히 같음),
     pub/data/ch13_be24.csv (13장 e 절 '논문에서 읽기' Table 2의 2×2 교차설계 24명, AUC0-t와 Cmax; gen/nums_ch13.py의 crossover())
"""
import os, sys, csv, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pub", "data")


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(name, len(rows))


def pdc_sample(rng, n, mean, sd, q=0.35, a=1.0, b=2.5):          # same as gen/pyout_ch13.py
    for _ in range(2000):
        gap = np.where(rng.random(n) < q, 0.0, 100 * rng.beta(a, b, n))
        x = 100 - gap
        z = (x - x.mean()) / x.std(ddof=1)
        y = mean + sd * z
        if y.min() >= 0 and y.max() <= 100:
            return y
    raise RuntimeError("no admissible sample")


rng = np.random.default_rng(1302)
remote = pdc_sample(rng, 304, 83.9, 16.8)
inperson = pdc_sample(rng, 304, 85.2, 17.5)
write("ch13_pdc.csv", ["arm", "pdc"],
      [["remote", f"{v:.6f}"] for v in remote] + [["inperson", f"{v:.6f}"] for v in inperson])


def crossover(seed, n=24, gmr=0.97, cvw=0.15, sb=0.30, mu=math.log(5200), per=0.02):   # same as gen/nums_ch13.py
    r = np.random.default_rng(seed)
    sw = math.sqrt(math.log(1 + cvw ** 2))
    seq = np.array([0] * (n // 2) + [1] * (n // 2))
    subj = r.normal(0, sb, n)
    y = np.zeros((n, 2)); trt = np.zeros((n, 2), int)
    for i in range(n):
        for p in range(2):
            t = 1 if (seq[i] == 0 and p == 0) or (seq[i] == 1 and p == 1) else 0
            trt[i, p] = t
            y[i, p] = mu + subj[i] + (math.log(gmr) if t else 0) + (per if p == 1 else 0) + r.normal(0, sw)
    return seq, y, trt


seq, ya, trt = crossover(10, gmr=0.97, cvw=0.15, mu=math.log(5200))
_, yc, trt_c = crossover(62, gmr=0.94, cvw=0.21, mu=math.log(812))
assert (trt == trt_c).all()
rows = []
for i in range(24):
    for p in range(2):
        rows.append([i + 1, "TR" if seq[i] == 0 else "RT", p + 1, "T" if trt[i, p] else "R",
                     f"{math.exp(ya[i, p]):.4f}", f"{math.exp(yc[i, p]):.4f}"])
write("ch13_be24.csv", ["subject", "sequence", "period", "formulation", "auc", "cmax"], rows)
