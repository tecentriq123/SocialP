"""실습 19 (메타분석) 자료 만들기.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab19.py

19장의 예제(가상의 체계적 문헌고찰: ACS 뒤 표준 항혈소판요법 + 경구 항응고제 '약물 X' 대 위약, 무작위배정 시험 10개)를
gen/nums_ch19.py의 simulate()에서 그대로 가져와 CSV 두 개로 내보낸다. 난수는 nums_ch19.SEED로 고정되어 있어
다시 돌려도 같은 파일이 나온다.

  pub/data/acs_trials.csv            시험 한 개가 한 줄 (10행)
  pub/data/acs_trials_subgroup.csv   시험 x 기저 진단 하위군이 한 줄 (하위군 결과를 보고한 6개 시험 x 2 = 12행)
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pandas as pd  # noqa: E402
import nums_ch19 as N  # noqa: E402  (import만으로는 아무것도 출력하지 않는다)

OUT = os.path.join(os.path.dirname(HERE), "pub", "data")
ROB = {"C": "some concerns", "G": "high"}   # 19장 가 절의 RoB 2 종합 판정 (나머지는 low)


def build():
    T = N.simulate()
    trials = pd.DataFrame([dict(
        trial="Trial " + t["nm"], year=t["yr"], phase=t["ph"], followup_mo=t["mo"], stemi_pct=t["ps"],
        rob=ROB.get(t["nm"], "low"),
        n_x=t["nT"], mace_x=t["eT"], bleed_x=t["bT"],
        n_pbo=t["nC"], mace_pbo=t["eC"], bleed_pbo=t["bC"]) for t in T])
    rows = []
    for t in T:
        if not t["rep"]:
            continue
        rows.append(dict(trial="Trial " + t["nm"], subgroup="STEMI",
                         n_x=t["nTs"], mace_x=t["eTs"], n_pbo=t["nCs"], mace_pbo=t["eCs"]))
        rows.append(dict(trial="Trial " + t["nm"], subgroup="NSTE-ACS",
                         n_x=t["nTn"], mace_x=t["eTn"], n_pbo=t["nCn"], mace_pbo=t["eCn"]))
    sub = pd.DataFrame(rows)
    return trials, sub


if __name__ == "__main__":
    trials, sub = build()
    # 본문(19장 가 절 표, 나 절, 다 절)의 합계와 대조
    assert trials["n_x"].sum() == 16850 and trials["n_pbo"].sum() == 14875
    assert trials["mace_x"].sum() == 1271 and trials["mace_pbo"].sum() == 1386
    assert trials["bleed_x"].sum() == 410 and trials["bleed_pbo"].sum() == 170
    s = sub.groupby("subgroup")[["mace_x", "n_x", "mace_pbo", "n_pbo"]].sum()
    assert tuple(s.loc["STEMI"]) == (528, 7006, 636, 6510) and tuple(s.loc["NSTE-ACS"]) == (663, 7774, 688, 7330)
    # 하위군 두 줄을 더하면 그 시험의 전체 결과와 같다
    chk = sub.groupby("trial")[["n_x", "mace_x", "n_pbo", "mace_pbo"]].sum()
    ref = trials.set_index("trial").loc[chk.index, ["n_x", "mace_x", "n_pbo", "mace_pbo"]]
    assert (chk == ref).all().all()
    os.makedirs(OUT, exist_ok=True)
    for name, df in (("acs_trials.csv", trials), ("acs_trials_subgroup.csv", sub)):
        path = os.path.join(OUT, name)
        df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        print(f"{name}: {len(df)} rows x {df.shape[1]} cols, {os.path.getsize(path)} bytes")
    print(trials.to_string(index=False))
    print(sub.to_string(index=False))
