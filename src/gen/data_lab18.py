"""실습 18 자료: pub/data/sedative_district_month.csv (시군구 × 월, 3,600줄).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab18.py

18장의 모의자료(gen/nums_ch18.py의 simulate(), seed 91)를 그대로 내보낸다. nums_ch18.py는 import하면
장 전체의 계산이 실행되고 _ch18_nums.json을 다시 쓰므로, 그 파일에서 simulate() 정의 부분의 소스만 읽어
실행한다(생성 코드는 한 곳에만 있다). 내보낸 뒤 _ch18_nums.json의 월별 분자·분모와 맞는지 확인한다.

열: unit(시군구 번호 1–60), treat(시범 지역 1, 비교 지역 0), month(그 달의 첫날, YYYY-MM-DD),
    n_pat(그 달 외래를 이용한 65세 이상 환자 수 = 분모), n_sed(그중 수면진정제를 원외처방받은 환자 수 = 분자)
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SEED = 91

src = open(os.path.join(HERE, "nums_ch18.py"), encoding="utf-8").read()
a = src.index("# ------------------------------------------------------------------ simulation")
b = src.index("d = simulate(SEED)")
ns = {"np": np, "pd": pd}
exec(compile(src[a:b], "nums_ch18.py[simulate]", "exec"), ns)
d = ns["simulate"](SEED)

out = pd.DataFrame({
    "unit": d["unit"].astype(int),
    "treat": d["treat"].astype(int),
    "month": [f"{2015 + (t - 1) // 12}-{(t - 1) % 12 + 1:02d}-01" for t in d["t"]],
    "n_pat": d["n"].astype(int),
    "n_sed": d["cnt"].astype(int),
})
os.makedirs(os.path.join(ROOT, "pub", "data"), exist_ok=True)
path = os.path.join(ROOT, "pub", "data", "sedative_district_month.csv")
out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

# ---- check against the chapter's numbers
J = json.load(open(os.path.join(HERE, "_ch18_nums.json")))["series"]
chk = pd.read_csv(path)
ag = chk.groupby(["treat", "month"])[["n_sed", "n_pat"]].sum()
assert ag.loc[1, "n_sed"].tolist() == J["cp"] and ag.loc[1, "n_pat"].tolist() == J["np"]
assert ag.loc[0, "n_sed"].tolist() == J["cc"] and ag.loc[0, "n_pat"].tolist() == J["nc"]
assert len(chk) == 3600 and chk["unit"].nunique() == 60 and chk.groupby("unit")["treat"].first().sum() == 20
print("saved", path, os.path.getsize(path), "bytes;", len(chk), "rows; matches _ch18_nums.json series")
