"""실습 15 자료 · pub/data/ps_cohort.csv

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab15.py

15장의 예제 코호트(gen/nums_ch15.py의 simulate_cohort(), seed 20260930)를 그대로 한 사람 한 줄의 CSV로
내보낸다. 난수를 새로 뽑지 않으므로 다시 돌려도 같은 파일이 나온다. 나이는 연속값이고 성향점수와 매칭
결과가 본문과 끝자리까지 같아야 하므로 반올림하지 않고 그대로 쓴다(pandas가 왕복 가능한 자릿수로 적는다).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nums_ch15 import simulate_cohort  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "pub", "data", "ps_cohort.csv")


def build():
    X, A, Y, risk, day, NC = simulate_cohort()
    N = len(A)
    return pd.DataFrame({
        "id": np.arange(1, N + 1),                 # 환자 번호
        "drugA": A.astype(int),                    # 약물 A = 1, 약물 B = 0
        "age": X["age"],                           # 첫 처방일의 나이(세, 연속값)
        "female": X["female"].astype(int),
        "hf": X["hf"].astype(int),                 # 심부전
        "ckd": X["ckd"].astype(int),               # 만성콩팥병
        "prior_hosp": X["prior"].astype(int),      # 지난 1년 입원
        "n_drug": X["ndrug"].astype(int),          # 복용 약물 계열 수
        "tertiary": X["tert"].astype(int),         # 상급종합병원 처방
        "hosp_1y": Y.astype(int),                  # 1년 안 입원 (결과)
        "hosp_day": day.astype(int),               # 입원일(첫 처방일부터의 일수), 입원 없으면 365
        "injury_ed": NC.astype(int),               # 1년 안 외상 응급실 방문 (음성 대조 결과)
        "frail": X["frail"].astype(int),           # 허약 (실제 청구자료에는 없는 변수)
    })


if __name__ == "__main__":
    df = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    back = pd.read_csv(OUT)
    # pandas의 기본 실수 파서는 마지막 자리(1 ulp)까지 되돌리지는 못한다. 그 차이(1e-14 미만)는 성향점수와
    # 매칭 결과에 영향을 주지 않는다(gen/lab_lab15.py 끝의 대조 블록에서 본문 숫자와 맞춰 확인).
    assert np.abs(back["age"].to_numpy() - df["age"].to_numpy()).max() < 1e-12, "age changed in the CSV"
    exact = pd.read_csv(OUT, float_precision="round_trip")
    assert np.array_equal(exact["age"].to_numpy(), df["age"].to_numpy())
    assert len(back) == 10000 and int(back["drugA"].sum()) == 4053
    print(OUT, f"{os.path.getsize(OUT) / 1024:.0f} KB", back.shape)
    print(back.head(3).to_string())
