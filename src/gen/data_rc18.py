"""Data file for R과 파이썬으로 하는 통계분석 '성향점수' 장: the matching order of 보건통계학 기초 15장.

run: python3 gen/data_rc18.py → pub/data/ch15_match_order.csv
15장의 1:1 매칭(gen/nums_ch15.py greedy_match)은 약물 A 사용자를 numpy의 무작위 순서(seed 7)로 차례로 짝짓는다.
R에서 같은 짝을 얻도록 그 순서(ps_cohort.csv의 id)를 파일로 남긴다. 순서는 성향점수와 상관없이 A 사용자의 행 번호만으로 정해진다.
"""
import os, csv
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = list(csv.DictReader(open(os.path.join(ROOT, "pub", "data", "ps_cohort.csv"))))
A = np.array([int(r["drugA"]) for r in rows]); ids = np.array([int(r["id"]) for r in rows])
t_idx = np.where(A == 1)[0]
order = np.random.default_rng(7).permutation(t_idx)          # same call as greedy_match(seed=7)
with open(os.path.join(ROOT, "pub", "data", "ch15_match_order.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id"]); w.writerows([[int(ids[i])] for i in order])
print("ch15_match_order.csv", len(order))
