"""실습 23 자료: pub/data/rcc_trial.csv (환자 한 명이 한 줄인 가상 임상시험의 전체생존 자료).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab23.py

23장 라 절의 가상 시험 자료(gen/nums_ch23.py가 저장한 gen/_ch23_trial.csv, 표준요법 B군 300명)를 그대로 쓰고
환자 번호 열만 붙인다. 난수를 쓰지 않으므로 다시 돌리면 같은 파일이 나온다.
  - pid    : 환자 번호 (T001–T300, 원자료의 줄 순서)
  - months : 무작위배정부터 사망 또는 자료 마감까지의 기간(개월, 소수 여섯째 자리까지. 원자료 그대로)
  - event  : 1 = 사망, 0 = 자료 마감 때 생존(중도절단)
"""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "_ch23_trial.csv")
OUT = os.path.join(ROOT, "pub", "data", "rcc_trial.csv")


def build():
    s = pd.read_csv(SRC)
    d = pd.DataFrame({
        "pid": [f"T{i:03d}" for i in range(1, len(s) + 1)],
        "months": s["months"],
        "event": s["event"].astype(int),
    })
    # 본문(23장 라 절)의 숫자와 같은 자료인지 확인
    assert len(d) == 300 and d["pid"].is_unique
    assert d["event"].sum() == 145 and (d["event"] == 0).sum() == 155
    assert set(d["event"].unique()) == {0, 1}
    assert d["months"].min() > 0 and 29.9 < d["months"].max() < 30.0
    assert d.loc[d["event"] == 0, "months"].min() >= 24.0          # 중도절단은 모두 24–30개월의 자료 마감
    assert d.isna().sum().sum() == 0
    return d


if __name__ == "__main__":
    d = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")
    back = pd.read_csv(OUT)
    assert (back["months"] == pd.read_csv(SRC)["months"]).all()   # 원자료와 같은 값으로 저장되었는지
    print(OUT, d.shape, f"{os.path.getsize(OUT) / 1024:.1f} KB")
    print(d.head())
