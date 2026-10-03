"""실습 17 (경쟁위험 분석) 자료 만들기.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab17.py

17장의 예제 자료를 그대로 내보낸다(열 이름만 정리). 새로 난수를 뽑지 않으므로 다시 돌려도 같은 파일이 나온다.
  gen/_ch17_cr.csv  ->  pub/data/ckd_competing.csv           (본문 예제: CKD 4기, 약물 A 대 B, 투석 / 투석 전 사망)
  gen/_ch17_fx.csv  ->  pub/data/dementia_fracture.csv  (나 절 확인 문제 2번: 치매 환자, 약물 C 대 D, 고관절 골절 / 골절 전 사망)
두 원본은 gen/nums_ch17.py가 만든다(sim_cr(13), sim_fx(25)). 원본이 없으면 그 스크립트를 먼저 돌린다.

열
  id         환자 번호 (1부터)
  drug       약물 (A/B 또는 C/D)
  years      약물 시작일부터 투석(골절), 사망, 자료 마감 중 가장 이른 날까지의 연수
  status     0 = 중도절단(자료 마감), 1 = 관심 사건(투석 시작 / 고관절 골절), 2 = 경쟁 사건(그 전의 사망)
  max_years  약물 시작일부터 자료 마감일까지의 연수(최대 5). 모든 환자에서 알 수 있는 값
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "pub", "data")


def export(src, dst):
    d = pd.read_csv(os.path.join(HERE, src), float_precision="round_trip")
    out = pd.DataFrame({"id": range(1, len(d) + 1), "drug": d["grp"], "years": d["t"],
                        "status": d["ev"].astype(int), "max_years": d["C"]})
    # 내보내기 전 점검: 중도절단은 자료 마감뿐이고, 사건은 마감 전에만 일어난다
    cens = out["status"] == 0
    assert (out.loc[cens, "years"] == out.loc[cens, "max_years"]).all()
    assert (out.loc[~cens, "years"] < out.loc[~cens, "max_years"]).all()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, dst)
    out.to_csv(path, index=False)            # 실수는 왕복해도 값이 바뀌지 않는 가장 짧은 표기로 저장된다
    back = pd.read_csv(path, float_precision="round_trip")
    assert (back["years"].values == d["t"].values).all() and (back["max_years"].values == d["C"].values).all()
    # 학생이 쓰는 기본 read_csv는 마지막 자리(1e-16)가 다르게 읽힐 수 있다. 순서와 중도절단 판정은 그대로여야 한다
    stu = pd.read_csv(path)
    assert abs(stu["years"] - back["years"]).max() < 1e-12
    assert (stu["years"].rank().values == back["years"].rank().values).all()
    c2 = stu["status"] == 0
    assert (stu.loc[c2, "years"] == stu.loc[c2, "max_years"]).all()
    assert (stu.loc[~c2, "years"] < stu.loc[~c2, "max_years"]).all()
    print(f"{dst}: {len(out)} rows, {os.path.getsize(path) / 1024:.0f} KB")
    print(pd.crosstab(out["status"], out["drug"], margins=True).to_string())


if __name__ == "__main__":
    export("_ch17_cr.csv", "ckd_competing.csv")
    export("_ch17_fx.csv", "dementia_fracture.csv")
