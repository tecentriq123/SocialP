"""실습 21 자료: pub/data/rcc_cost.csv (환자 한 명이 한 줄인 1년 의료비 자료, 가상).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab21.py

21장의 예제 자료(gen/nums_ch21.py가 저장한 gen/_ch21_cost.csv)를 그대로 쓰고 열만 정리한다.
난수를 쓰지 않으므로 다시 돌리면 같은 파일이 나온다.
  - 남기는 열: 환자 번호, 치료군, 공변량 4개, 항목별 비용 3개, 총비용
  - 치료군 A(1/0) -> arm("A"/"B"), 비용 항목 drug/op/inp -> cost_drug/cost_op/cost_inp
  - 버리는 열: 모의 자료에서만 알 수 있는 생존·진행 시점과 중도절단 관련 열(td, pfs, cdate, cost_c 등)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "_ch21_cost.csv")
OUT = os.path.join(ROOT, "pub", "data", "rcc_cost.csv")


def build():
    s = pd.read_csv(SRC)
    d = pd.DataFrame({
        "pid": s["pid"],
        "arm": np.where(s["A"] == 1, "A", "B"),
        "age": s["age"].round().astype(int),
        "male": s["male"].astype(int),
        "cci": s["cci"].astype(int),
        "stage4": s["stage4"].astype(int),
        "cost_drug": s["drug"],
        "cost_op": s["op"],
        "cost_inp": s["inp"],
        "cost": s["cost"],
    })
    # 본문(21장 나 절)의 숫자와 같은 자료인지 확인
    assert len(d) == 3000 and d["pid"].is_unique
    assert (d["arm"] == "A").sum() == 1203 and (d["arm"] == "B").sum() == 1797
    assert (s["age"] == d["age"]).all()
    assert abs(d.loc[d["arm"] == "A", "cost"].mean() - 2668.862) < 0.001
    assert abs(d.loc[d["arm"] == "B", "cost"].mean() - 2272.788) < 0.001
    assert ((d["cost_drug"] + d["cost_op"] + d["cost_inp"]).round(1) == d["cost"]).all()
    assert d.isna().sum().sum() == 0
    return d


if __name__ == "__main__":
    d = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    print(OUT, d.shape, f"{os.path.getsize(OUT) / 1024:.0f} KB")
    print(d.head())
