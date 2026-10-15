"""실습 22 자료: pub/data/lab22_eq5d.csv (설문 응답 한 건이 한 줄), pub/data/lab22_patients.csv (환자 한 명이 한 줄).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab22.py

22장의 가상 임상시험 자료(gen/nums_ch22.py가 저장한 gen/_ch22_trial.csv, gen/_ch22_pts.csv)를 읽어 공개용 사본을 만든다.
원본 두 파일은 읽기만 하고 바꾸지 않는다. 난수를 쓰지 않으므로 다시 돌리면 같은 파일이 나온다.
  lab22_eq5d.csv (1,593행)
  - id, arm      : 환자 번호(1–320), 치료군(A = 신약 A, B = 표준요법 B)
  - month        : 설문 시점(0, 3, 6, 12, 18, 24개월). 원자료의 0.0 같은 소수를 정수로
  - state        : 응답한 때의 상태(PF = 무진행, PD = 진행)
  - eq5d         : EQ-5D-5L 다섯 자리 응답 코드(실습에서는 글자로 읽는다)
  - utility      : 한국 EQ-5D-5L 가치 세트로 바꾼 효용. 소수 셋째 자리로 반올림(0.6940000000000001 -> 0.694)
  - vas          : EQ VAS(0–100). 원자료의 41.0 같은 값을 정수로
  lab22_patients.csv (320행)
  - id, arm      : 위와 같음
  - t_prog       : 무작위배정부터 진행까지(개월). 진행 없이 사망한 47명은 inf
  - t_death      : 무작위배정부터 사망까지(개월). 가상 자료라서 24개월 뒤의 시점도 들어 있다
  - 원자료의 qaly, qaly_d, ly, base, A 열은 실습에서 계산하므로 뺀다
  시점은 소수 여섯째 자리까지 저장한다(실습 23의 rcc_trial.csv와 같은 방식). 아래에서 22장의 숫자가 그대로 나오는지 확인한다.
가치 세트의 계수는 P4_FACTS.md 5.4(한국 EQ-5D-5L, kim2016eq5d5l)의 값이다.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_OBS = os.path.join(HERE, "_ch22_trial.csv")
SRC_PTS = os.path.join(HERE, "_ch22_pts.csv")
OUT_OBS = os.path.join(ROOT, "pub", "data", "lab22_eq5d.csv")
OUT_PTS = os.path.join(ROOT, "pub", "data", "lab22_patients.csv")

# 한국 EQ-5D-5L 가치 세트(P4_FACTS.md 5.4): 1 - 0.096(11111이 아니면) - 영역별 감점 - 0.078(수준 4나 5가 하나라도 있으면)
KR5 = [[0, 0.046, 0.058, 0.133, 0.251], [0, 0.032, 0.050, 0.078, 0.122], [0, 0.021, 0.051, 0.100, 0.175],
       [0, 0.042, 0.053, 0.166, 0.207], [0, 0.033, 0.046, 0.102, 0.137]]


def kr5l(code):
    lv = [int(x) for x in code]
    if lv == [1] * 5:
        return 1.0
    u = 1 - 0.096 - sum(dec[v - 1] for dec, v in zip(KR5, lv)) - (0.078 if max(lv) >= 4 else 0.0)
    return round(u, 3)


def qaly24(obs, pts):
    """22장과 같은 계산(사다리꼴 공식, 사망 시점에 0, 그 뒤 24개월까지 0)으로 환자별 24개월 QALY"""
    out = []
    for r in pts.itertuples():
        o = obs[obs["id"] == r.id]
        t, u = list(o["month"].astype(float)), list(o["utility"].astype(float))
        if r.t_death <= 24:
            t, u = t + [r.t_death], u + [0.0]
            if r.t_death < 24:
                t, u = t + [24.0], u + [0.0]
        out.append(np.trapezoid(u, t) / 12)
    return np.array(out)


def build():
    so = pd.read_csv(SRC_OBS, dtype={"eq5d": str})
    sp = pd.read_csv(SRC_PTS)
    obs = pd.DataFrame({
        "id": so["id"].astype(int),
        "arm": so["arm"],
        "month": so["month"].round().astype(int),
        "state": so["state"],
        "eq5d": so["eq5d"],
        "utility": so["utility"].round(3),
        "vas": so["vas"].round().astype(int),
    })
    pts = pd.DataFrame({"id": sp["id"].astype(int), "arm": sp["arm"], "t_prog": sp["t_prog"], "t_death": sp["t_death"]})

    # 원자료와 같은 값인지
    assert (obs["month"] == so["month"]).all() and (obs["vas"] == so["vas"]).all()
    assert (obs["utility"] - so["utility"]).abs().max() < 1e-12
    assert (obs["eq5d"].str.len() == 5).all() and obs["eq5d"].str.fullmatch("[1-5]{5}").all()
    assert (obs["eq5d"].map(kr5l) == obs["utility"]).all()            # 가치 세트로 다시 계산한 값과 소수 셋째 자리까지 같다
    # 22장 본문의 숫자
    assert len(obs) == 1593 and len(pts) == 320 and pts["id"].is_unique
    assert (pts["arm"].value_counts() == 160).all()
    assert obs["state"].value_counts().to_dict() == {"PF": 1184, "PD": 409}
    assert obs.loc[obs["state"] == "PD", "id"].nunique() == 174
    assert int(np.isinf(pts["t_prog"]).sum()) == 47
    assert pts.loc[pts["t_death"] <= 24].groupby("arm").size().to_dict() == {"A": 55, "B": 68}
    b0 = obs[obs["month"] == 0]
    assert len(b0) == 320
    assert [round(b0.loc[b0["arm"] == a, "utility"].mean(), 3) for a in "AB"] == [0.786, 0.766]
    assert [round(b0.loc[b0["arm"] == a, "vas"].mean(), 1) for a in "AB"] == [73.5, 71.9]
    m = obs.merge(pts, on=["id", "arm"])
    assert ((m["month"] >= m["t_prog"]) == (m["state"] == "PD")).all()   # 상태 열이 진행 시점과 맞다
    assert (m["month"] < m["t_death"]).all()                             # 사망 뒤의 응답은 없다
    assert obs.isna().sum().sum() == 0 and pts.isna().sum().sum() == 0
    return obs, pts, sp


if __name__ == "__main__":
    obs, pts, sp = build()
    os.makedirs(os.path.dirname(OUT_OBS), exist_ok=True)
    obs.to_csv(OUT_OBS, index=False, encoding="utf-8", lineterminator="\n")
    pts.to_csv(OUT_PTS, index=False, encoding="utf-8", lineterminator="\n", float_format="%.6f")
    # 다시 읽어 확인: 코드가 글자로, inf가 inf로 돌아오는지, 시점을 여섯째 자리로 줄여도 22장의 QALY와 같은지
    ob = pd.read_csv(OUT_OBS, dtype={"eq5d": str})
    pb = pd.read_csv(OUT_PTS)
    assert ob.equals(obs)
    assert int(np.isinf(pb["t_prog"]).sum()) == 47
    assert np.allclose(pb["t_death"], sp["t_death"], atol=5e-7, rtol=0)
    q_new, q_old = qaly24(ob, pb), sp["qaly"].to_numpy()
    assert np.abs(q_new - q_old).max() < 1e-6, np.abs(q_new - q_old).max()
    print(OUT_OBS, obs.shape, f"{os.path.getsize(OUT_OBS) / 1024:.1f} KB")
    print(OUT_PTS, pts.shape, f"{os.path.getsize(OUT_PTS) / 1024:.1f} KB")
    print("시점을 여섯째 자리로 저장한 뒤의 24개월 QALY 차이(가장 큰 값):", np.abs(q_new - q_old).max())
    print(ob.head())
    print(pb.head())
