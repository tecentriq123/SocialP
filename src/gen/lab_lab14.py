"""실습 14 · 청구자료에서 코호트 만들기 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab14.py && python3 gen/lab_lab14.py
      then  python3 tools/nbcheck.py lab14   (runs the notebook alone in a fresh process and compares outputs)
      (NOMARK=1 python3 gen/lab_lab14.py  prints every cell output without marks)

자료: pub/data/claims_person.csv, claims_visit.csv, claims_rx.csv, sglt2_cohort.csv (gen/data_lab14.py가 만든다).
14장 예제 코호트(gen/nums_ch14.py)의 30분의 1 표본이다. 왜 표본인지와 고른 방법은 data_lab14.py의 설명에 있다.

끝의 대조 블록이 gen/_ch14_nums.json과 맞추는 것
  - 선정 흐름: 단계별 남은 인원 6개, 제외 인원 5개, 군별 인원 2개가 본문 인원의 1/30(반올림)과 정확히 같다.
  - 사람별: 파이프라인이 만든 5,552명의 진입일, 군, 공변량, 추적 일수, 사건, 종료 사유가 nums_ch14의 값과 같다.
  - 표본의 발생률(8개), 발생률비(4개), 조·보정 위험비(8개)는 소수 둘째 자리까지 같다(유예 60일, 30일, 90일, ITT).
  - 기저 특성(군별 비율 7개, 나이 평균과 SD)은 소수 첫째 자리까지 같다.
  - 전체 코호트 파일(sglt2_cohort_full.csv)로는 표 14-5의 사건, 인년, 발생률과 정확 신뢰구간, 발생률비, 조·보정
    위험비와 신뢰구간이 as-treated와 ITT 모두 본문과 같다. 사건 비율의 비(0.37), 1인당 평균 추적(1.87, 2.18년),
    역 Kaplan-Meier 중앙 추적기간(1.80, 3.50년)도 본문과 같다.
  - 양성예측도의 Wilson 신뢰구간(85.5%, 96.7%)과 c 절 예제 환자의 PDC(315/365)는 본문과 같다.
  맞추지 않는 것: 표본의 신뢰구간(사건 수가 1/30이라 넓다), 표본 인년의 절대값(본문/30과 0.2-3.5% 다르다),
  표본의 PDC 0.8 이상 비율과 1년 지속 비율(표본을 발생률과 위험비에 맞춰 골라서 본문과 2%p 안쪽으로 다르다).

(2026-10-14: 14장을 PART 3의 새 틀로 고친 뒤 '실습 14에 더할 것'을 반영했다. 버전 셀 2, 세척 기간 180일 셀 13
 (옛 과제 2), 첫 365일 PDC와 1년 지속 셀 16, 양성예측도의 신뢰구간 셀 19, 사건 비율과 발생률 셀 23, 전체 코호트의
 사건 비율과 중앙 추적기간 셀 29를 더했다. 과제는 ① ITT 결과 해석(옛 과제 3) ② 유예기간 30·60·90일(옛 과제 1에
 90일을 더함) ③ 첫 처방일 뒤의 기록으로 만든 공변량(새로)으로 바꿨다. 셀 번호가 모두 바뀌었다.)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("lab14")
NOMARK = bool(os.environ.get("NOMARK"))


def render(c, marks=None, dfmarks=None):
    """text marks -> first text <pre> only; table marks -> DataFrame only."""
    h = nb.html(c)
    if marks:
        i = h.index('<div class="cell-out">')
        j = h.index("<pre>", i)
        k = h.index("</pre>", j)
        h = h[:j] + _mark(h[j:k], marks) + h[k:]
    if dfmarks:
        j = h.index('<div class="df-wrap">')
        k = h.index("</table>", j)
        h = h[:j] + _mark(h[j:k], dfmarks) + h[k:]
    return h


def save(name, c, marks=None, dfmarks=None):
    if NOMARK:
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


# ---------------------------------------------------------------- 예시로 따라갈 사람 고르기 (자료 파일에서 규칙으로 고른다)
PUBDATA = os.path.join(os.path.dirname(HERE), "pub", "data")
_ref = pd.read_csv(os.path.join(PUBDATA, "sglt2_cohort.csv"))
_rx = pd.read_csv(os.path.join(PUBDATA, "claims_rx.csv"), parse_dates=["rx_date"])
_vis = pd.read_csv(os.path.join(PUBDATA, "claims_visit.csv"))
_person = pd.read_csv(os.path.join(PUBDATA, "claims_person.csv"))


def pick_example():
    """유예 60일에서는 약을 쓰는 중에 심부전으로 입원하고(event 1), 유예 30일이면 그 전에 중단으로 끝나는 사람.
    본문 c 절의 예시 환자와 같은 상황이다. 처방과 진료 건수가 적은 사람을 고른다."""
    st = _rx[_rx["drug"].isin(["SGLT2i", "DPP4i"])].merge(_ref[["pid", "index_date", "arm"]], on="pid")
    st = st[st["drug"] == st["arm"]].sort_values(["pid", "rx_date"])
    st["runout"] = st["rx_date"] + pd.to_timedelta(st["days"], unit="D")
    st["gap"] = (st.groupby("pid")["rx_date"].shift(-1) - st["runout"]).dt.days
    ev = _ref[(_ref["event"] == 1) & (_ref["htn"] == 1) & (_ref["met"] == 1)]
    best = None
    for pid in ev["pid"]:
        g = st[st["pid"] == pid]
        first_long = g[(g["gap"] > 30) | g["gap"].isna()].iloc[0]
        nvis = int((_vis["pid"] == pid).sum())
        nrx = int((_rx["pid"] == pid).sum())
        t = int(ev.loc[ev["pid"] == pid, "time"].iloc[0])
        stop30 = (first_long["runout"] + pd.Timedelta(days=30) - pd.Timestamp(g["index_date"].iloc[0])).days
        dead = bool(_person.loc[_person["pid"] == pid, "death_date"].notna().iloc[0])
        if 30 < (first_long["gap"] if pd.notna(first_long["gap"]) else 999) <= 60 and stop30 < t and 5 <= len(g) <= 18:
            key = (dead, g["arm"].iloc[0] != "SGLT2i", nvis + nrx, pid)
            if nvis <= 11 and (best is None or key < best):
                best = key
    return best[3]


def pick_prevalent():
    """2015년에 DPP-4 억제제를 쓰다가 2016년의 첫 처방이 SGLT2 억제제인 사람(기존 사용자, 계열을 바꾼 경우)"""
    st = _rx[_rx["drug"].isin(["SGLT2i", "DPP4i"])]
    for pid, g in st.sort_values(["pid", "rx_date"]).groupby("pid"):
        if len(g) == 2 and g["rx_date"].iloc[0].year == 2015 and g["rx_date"].iloc[1].year == 2016 \
                and g["drug"].tolist() == ["DPP4i", "SGLT2i"]:
            return int(pid)


P_EX = int(pick_example())
P_PREV = pick_prevalent()
print("example person:", P_EX, " prevalent example:", P_PREV)

PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

# 출력에 붙이는 번호 표식 (자료를 다시 만들어 숫자가 달라지면 여기서 ValueError가 난다)
M = {
    "versions": {"pandas      3.0.5": 1, "statsmodels 0.15.0": 2, "lifelines   0.30.3": 3},
    "load": {"(10306, 6)": 1, "datetime64[us]": 2},
    "head": {"NaT": 1, "setting": 2, "I209": 3, "NaN": 4, "days": 5},
    "units": {"person: 10306 rows, 10306 people": 1, "rx    : 111882 rows": 2, "2015-01-01 2022-12-31": 3,
              "50%          4.0": 4, "DPP4i           86956": 5, "I     1277": 6},
    "one": {"2018-02-17  metformin    60": 1, "2018-05-18      DPP4i    90": 2, "2019-07-02      DPP4i    90": 3,
            "I200": 4, "2021-08-22       I    I501": 5},
    "index": {"10306 -> 10156 people": 1},
    "arm": {"77636 -> 10287 rows": 1, "SGLT2i    1310": 2, "both classes on index date: 131": 3},
    "washout": {"prevalent users: 4103": 1},
    "washout_df": {"38": 2, "SGLT2i": 3},
    "age": {"min         10.0": 1, "age < 18: 14": 2},
    "dxflags": {"esrd         194": 1, "prior_hhf    234": 2},
    "t2dm": {"no E11 code by index date: 156": 1, "SGLT2i    65": 2},
    "flow": {"               10156": 1, "156      10000": 2, "4103       5897": 3, "136       5552": 4,
             "SGLT2i    1028": 5},
    "washout180": {"3785       6215": 1, "141       5856": 2, "DPP4i     4812": 3, "min      190.0": 4,
                   "max      360.0": 5},
    "gap_df": {"2018-08-16": 1, "12.0": 2, "36.0": 3, "NaN": 4},
    "stop": {"grace 30 -> 2019-06-26": 1, "grace 60 -> 2023-04-07": 2},
    "pdc": {"PDC 0.863 = 315 / 365": 1, "smallest gap: 0 days": 2},
    "pdc_df": {"70.6": 3, "73.2": 4, "66.0": 5, "77.0": 6},
    "switch": {"switched or added: 817": 1, "DPP4i     0.160": 2},
    "outcome": {"287 claims, 221 people": 1, "any position, inpt : 321 people": 2,
                "any claim with I50 : 653 people": 3},
    "ppv": {"85.5% (95% CI 80.0-89.7)": 1, "96.7% (95% CI 95.7-97.5)": 2},
    "follow": {"1192      1": 1},
    "follow_df": {"139": 2, "2693": 3, "2104": 4},
    "rate": {"1986.20   5.54": 1, "9882.12  12.95": 2, "2.76   9.91": 3, "IRR 0.43 (0.23-0.79)": 4},
    "share": {"ratio of event shares: 0.38": 1, "rate ratio           : 0.43": 2},
    "share_df": {"1.07": 3, "1.93": 4},
    "table1_df": {"57.0": 1, "6.8": 2, "86.3": 3},
    "cox": {"crude HR    0.42 (0.23-0.77)": 1, "adjusted HR 0.61 (0.32-1.13)": 2},
    "check": {"True": 1},
    "full_df": {"11 / 1986": 1, "320 / 57781": 2, "5.54 (4.95-6.18)": 3, "0.61 (0.32-1.13)": 4, "0.61 (0.54-0.68)": 5},
    "fullfu": {"ratio of event shares: 0.37": 1, "1.87": 2, "as-treated : 1.80": 3, "ITT        : 3.50": 4,
               "sample, as-treated : 1.81": 5},
    "hw1": {"crude HR    0.51 (0.32-0.81)": 1, "adjusted HR 0.73 (0.45-1.17)": 2, "SGLT2i     7.76": 3},
    "hw1_df": {"582 / 92403": 4, "0.73 (0.67-0.79)": 5},
    "hw2": {"grace 30: 2019-06-26, day 404, event 0": 1},
    "hw2_df": {"0.60 (0.30-1.20)": 2, "0.61 (0.32-1.13)": 3, "0.61 (0.33-1.11)": 4},
    "hw3": {"hf = 1, whole period    : 823": 1, "events with hf = 1 in c_any: 1.0": 2,
            "365 days         HR 0.61 (0.32-1.13)": 3, "all before index HR 0.61 (0.32-1.14)": 4,
            "whole period     HR 0.71 (0.38-1.33)": 5},
}

# ================================================================ a. 청구자료의 표 구조
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab14_install", c)

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy, matplotlib, statsmodels, lifelines

print("pandas     ", pd.__version__)
print("numpy      ", np.__version__)
print("scipy      ", scipy.__version__)
print("matplotlib ", matplotlib.__version__)
print("statsmodels", statsmodels.__version__)
print("lifelines  ", lifelines.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab14_versions", c, marks=M.get("versions"))

c = nb.cell('''
BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
person = pd.read_csv(BASE + "claims_person.csv",
    parse_dates=["elig_start", "elig_end", "death_date"],
    storage_options=UA)
visit = pd.read_csv(BASE + "claims_visit.csv",
                    parse_dates=["visit_date"], storage_options=UA)
rx = pd.read_csv(BASE + "claims_rx.csv", parse_dates=["rx_date"],
                 storage_options=UA)
print(person.shape, visit.shape, rx.shape)
print(rx.dtypes)
''', title="세 표 불러오기")
save("lab14_load", c, marks=M.get("load"))

c = nb.cell('''
print(person.head(3))
print(visit.head(3))
print(rx.head(3))
''', title="표마다 앞의 세 줄 보기")
save("lab14_head", c, marks=M.get("head"))

c = nb.cell('''
print("person:", len(person), "rows,",
      person["pid"].nunique(), "people")
print("visit :", len(visit), "rows,",
      visit["pid"].nunique(), "people")
print("rx    :", len(rx), "rows,", rx["pid"].nunique(), "people")
print(person["elig_start"].min().date(),
      person["elig_end"].max().date())
print(rx.groupby("pid").size().describe().round(1))
print(rx["drug"].value_counts())
print(visit["setting"].value_counts())
''', title="행의 단위 확인하기")
save("lab14_units", c, marks=M.get("units"))

c = nb.cell(f'''
p = {P_EX}
print(person[person["pid"] == p])
print(rx[rx["pid"] == p].sort_values("rx_date"))
print(visit[visit["pid"] == p].sort_values("visit_date"))
''', title="한 사람의 기록을 세 표에서 따라가기")
save("lab14_one", c, marks=M.get("one"))

# ================================================================ b. 신규 사용자 코호트 만들기
c = nb.cell('''
study = rx[rx["drug"].isin(["SGLT2i", "DPP4i"])]
enroll = study[(study["rx_date"] >= "2016-01-01")
               & (study["rx_date"] <= "2021-12-31")]
index = enroll.groupby("pid")["rx_date"].min()
index = index.rename("index_date").reset_index()
print(study["pid"].nunique(), "->", len(index), "people")
index.head(3)
''', title="등록 기간의 첫 처방일(index date) 찾기")
save("lab14_index", c, marks=M.get("index"))

c = nb.cell('''
first = enroll.merge(index, on="pid")
first = first[first["rx_date"] == first["index_date"]]
print(len(enroll), "->", len(first), "rows")

coh = index.copy()
coh["arm"] = coh["pid"].map(first.groupby("pid")["drug"].first())
n_class = first.groupby("pid")["drug"].nunique()
coh["both"] = coh["pid"].map(n_class) > 1
print(coh["arm"].value_counts())
print("both classes on index date:", coh["both"].sum())
''', title="index date에 처방받은 계열과 동시 시작 여부")
save("lab14_arm", c, marks=M.get("arm"))

c = nb.cell(f'''
m = study.merge(index, on="pid")
m["before"] = (m["index_date"] - m["rx_date"]).dt.days
prior = m[(m["before"] >= 1) & (m["before"] <= 365)]
coh["prevalent"] = coh["pid"].isin(prior["pid"])
print("prevalent users:", coh["prevalent"].sum())
m[m["pid"] == {P_PREV}]
''', title="세척 기간 365일의 처방으로 기존 사용자 찾기")
save("lab14_washout", c, marks=M.get("washout"), dfmarks=M.get("washout_df"))

c = nb.cell('''
info = person[["pid", "sex", "birth_year", "death_date"]]
coh = coh.merge(info, on="pid", how="left")
coh["age"] = coh["index_date"].dt.year - coh["birth_year"]
coh["minor"] = coh["age"] < 18
print(coh["age"].describe().round(1))
print("age < 18:", coh["minor"].sum())
''', title="자격 표를 붙여 index date의 나이 구하기")
save("lab14_age", c, marks=M.get("age"))

c = nb.cell('''
v = visit.merge(index, on="pid")
v["before"] = (v["index_date"] - v["visit_date"]).dt.days
base = v[(v["before"] >= 0) & (v["before"] <= 365)]

def has_dx(df, codes, cols=("dx_main", "dx_sub1", "dx_sub2")):
    hit = pd.Series(False, index=df.index)
    for col in cols:
        hit = hit | df[col].str.startswith(codes, na=False)
    return hit

esrd = base[has_dx(base, ("N185", "Z49", "Z992"))]
hhf0 = base[(base["setting"] == "I")
            & has_dx(base, "I50", cols=["dx_main"])]
coh["esrd"] = coh["pid"].isin(esrd["pid"])
coh["prior_hhf"] = coh["pid"].isin(hhf0["pid"])
print(coh[["esrd", "prior_hhf"]].sum())
''', title="기저 기간의 상병으로 말기신부전과 심부전 입원력 찾기")
save("lab14_dxflags", c, marks=M.get("dxflags"))

c = nb.cell('''
dm = v[(v["before"] >= 0) & has_dx(v, "E11")]
coh["no_t2dm"] = ~coh["pid"].isin(dm["pid"])
print("no E11 code by index date:", coh["no_t2dm"].sum())
print(coh.loc[coh["no_t2dm"], "arm"].value_counts())
''', title="index date까지의 명세서로 제2형 당뇨병 진단(E11) 확인하기")
save("lab14_t2dm", c, marks=M.get("t2dm"))

c = nb.cell('''
steps = [("no_t2dm", "No type 2 diabetes code (E11)"),
         ("prevalent", "Prior use of either class (365 d)"),
         ("both", "Both classes on index date"),
         ("minor", "Age < 18 years"),
         ("esrd", "ESRD or dialysis (365 d)"),
         ("prior_hhf", "HF hospitalization (365 d)")]

def flow_table(coh):
    keep = pd.Series(True, index=coh.index)
    rows = [("Prescribed in 2016-2021", "", len(coh))]
    for col, label in steps:
        out = keep & coh[col]
        keep = keep & ~coh[col]
        rows.append((label, out.sum(), keep.sum()))
    flow = pd.DataFrame(rows,
                        columns=["step", "excluded", "remaining"])
    return flow, keep

flow, keep = flow_table(coh)
print(flow.to_string(index=False))
cohort = coh[keep].copy()
print(cohort["arm"].value_counts())
''', title="선정 기준을 차례로 적용하며 인원 세기")
save("lab14_flow", c, marks=M.get("flow"))

c = nb.cell('''
coh180 = coh.copy()
prior180 = m[(m["before"] >= 1) & (m["before"] <= 180)]
coh180["prevalent"] = coh180["pid"].isin(prior180["pid"])
flow180, keep180 = flow_table(coh180)
print(flow180.to_string(index=False))
print(coh180[keep180]["arm"].value_counts())
late = prior[~prior["pid"].isin(prior180["pid"])]
print(late.groupby("pid")["before"].min().describe().round(0))
''', title="세척 기간을 180일로 줄이면")
save("lab14_washout180", c, marks=M.get("washout180"))

# ================================================================ c. 노출·결과·공변량 만들기
c = nb.cell('''
f = study.merge(cohort[["pid", "index_date", "arm"]], on="pid")
own = f[(f["drug"] == f["arm"])
        & (f["rx_date"] >= f["index_date"])].copy()
own = own.sort_values(["pid", "rx_date"])
own["runout"] = own["rx_date"] + pd.to_timedelta(own["days"],
                                                 unit="D")
own["next"] = own.groupby("pid")["rx_date"].shift(-1)
own["gap"] = (own["next"] - own["runout"]).dt.days
own.loc[own["pid"] == p, ["rx_date", "days", "runout", "gap"]]
''', title="처방 기록에서 약이 떨어지는 날과 공백 구하기", max_rows=20)
save("lab14_gap", c, dfmarks=M.get("gap_df"))

c = nb.cell('''
def stop_date(own, grace):
    over = own["gap"].isna() | (own["gap"] > grace)
    runout = own[over].groupby("pid")["runout"].first()
    return runout + pd.Timedelta(days=grace)

cohort["stop_date"] = cohort["pid"].map(stop_date(own, 60))
for grace in [30, 60, 90]:
    print("grace", grace, "->", stop_date(own, grace)[p].date())
''', title="유예기간으로 중단일 정하기")
save("lab14_stop", c, marks=M.get("stop"))

c = nb.cell('''
def pdc_365(fills):
    """첫 365일 가운데 약이 있던 날의 비율 (처방이 겹치지 않을 때)"""
    win_end = fills["index_date"] + pd.Timedelta(days=365)
    days = (fills["runout"].clip(upper=win_end)
            - fills["rx_date"]).dt.days.clip(lower=0)
    return days.groupby(fills["pid"]).sum() / 365

ex = pd.DataFrame({"pid": 0, "days": [30, 30, 30, 90, 90, 90],
                   "rx_date": pd.to_datetime(
                       ["2021-03-02", "2021-04-01", "2021-05-06",
                        "2021-06-05", "2021-10-18", "2022-01-16"])})
ex["index_date"] = ex["rx_date"].min()
ex["runout"] = ex["rx_date"] + pd.to_timedelta(ex["days"], unit="D")
pdc_ex = pdc_365(ex).iloc[0]
print(f"chapter example: PDC {pdc_ex:.3f} = {pdc_ex * 365:.0f} / 365")
print("smallest gap:", int(own["gap"].min()), "days")

cohort["pdc"] = cohort["pid"].map(pdc_365(own))
adh = pd.DataFrame({"PDC >= 0.8":
                    (cohort["pdc"] >= 0.8).groupby(cohort["arm"]).mean()})
for grace in [30, 60]:
    stop = cohort["pid"].map(stop_date(own, grace))
    on_drug = (stop - cohort["index_date"]).dt.days > 365
    adh[f"1 y, grace {grace}"] = on_drug.groupby(cohort["arm"]).mean()
(100 * adh).T.round(1)
''', title="첫 365일의 PDC와 1년 지속 비율")
save("lab14_pdc", c, marks=M.get("pdc"), dfmarks=M.get("pdc_df"))

c = nb.cell('''
other = f[(f["drug"] != f["arm"])
          & (f["rx_date"] > f["index_date"])]
switch = other.groupby("pid")["rx_date"].min()
cohort["switch_date"] = cohort["pid"].map(switch)
print("switched or added:", cohort["switch_date"].notna().sum())
print(cohort["switch_date"].notna().groupby(cohort["arm"]).mean()
      .round(3))
''', title="비교약으로 바꾸거나 더한 날")
save("lab14_switch", c, marks=M.get("switch"))

c = nb.cell('''
after = visit.merge(cohort[["pid", "index_date"]], on="pid")
after = after[after["visit_date"] > after["index_date"]]
adm = after[(after["setting"] == "I")
            & has_dx(after, "I50", cols=["dx_main"])]
cohort["hhf_date"] = cohort["pid"].map(
    adm.groupby("pid")["visit_date"].min())
print("primary, inpatient :", len(adm), "claims,",
      adm["pid"].nunique(), "people")
wide = after[(after["setting"] == "I") & has_dx(after, "I50")]
print("any position, inpt :", wide["pid"].nunique(), "people")
print("any claim with I50 :",
      after[has_dx(after, "I50")]["pid"].nunique(), "people")
''', title="결과: 주상병이 심부전인 입원의 첫 날짜")
save("lab14_outcome", c, marks=M.get("outcome"))

c = nb.cell('''
from statsmodels.stats.proportion import proportion_confint

for name, k, n in [("HHF definition", 171, 200),
                   ("diabetes definition B", 1391, 1438)]:
    lo, hi = proportion_confint(k, n, method="wilson")
    print(f"{name}: {k}/{n} = {100 * k / n:.1f}%"
          f" (95% CI {100 * lo:.1f}-{100 * hi:.1f})")
''', title="정의의 양성예측도와 신뢰구간")
save("lab14_ppv", c, marks=M.get("ppv"))

c = nb.cell('''
b = base[base["pid"].isin(cohort["pid"])]
dx_def = {"htn": ("I10", "I11", "I12", "I13", "I15"),
          "ihd": ("I20", "I21", "I22", "I23", "I24", "I25"),
          "hf": "I50", "ckd": "N18"}
for name, codes in dx_def.items():
    pids = b[has_dx(b, codes)]["pid"]
    cohort[name] = cohort["pid"].isin(pids).astype(int)

r = rx.merge(index, on="pid")
r["before"] = (r["index_date"] - r["rx_date"]).dt.days
rb = r[(r["before"] >= 0) & (r["before"] <= 365)]
for name, drug in [("met", "metformin"), ("ins", "insulin")]:
    pids = rb[rb["drug"] == drug]["pid"]
    cohort[name] = cohort["pid"].isin(pids).astype(int)

cohort["female"] = (cohort["sex"] == "F").astype(int)
cohort["sglt2"] = (cohort["arm"] == "SGLT2i").astype(int)
covs = ["age", "female", "htn", "ihd", "hf", "ckd", "met", "ins"]
cohort[["pid", "arm", "index_date"] + covs].head(3)
''', title="기저 기간의 상병과 처방으로 공변량 만들기")
save("lab14_covs", c, dfmarks=M.get("covs_df"))

# ================================================================ d. 추적 기간과 발생률
c = nb.cell('''
cohort["study_end"] = pd.Timestamp("2022-12-31")

def follow_up(c, ends):
    c = c.copy()
    c["end_date"] = c[ends].min(axis=1)
    c["time"] = (c["end_date"] - c["index_date"]).dt.days
    c["event"] = (c["hhf_date"] == c["end_date"]).astype(int)
    hit = [c[col] == c["end_date"] for col in ends]
    c["reason"] = np.select(hit, ends, default="")
    return c

AT = ["hhf_date", "death_date", "switch_date", "stop_date",
      "study_end"]
cohort = follow_up(cohort, AT)
print(cohort.loc[cohort["pid"] == p,
                 ["index_date", "end_date", "time", "event"]])
pd.crosstab(cohort["reason"], cohort["arm"], margins=True)
''', title="추적 종료일, 추적 일수, 사건 여부")
save("lab14_follow", c, marks=M.get("follow"), dfmarks=M.get("follow_df"))

c = nb.cell('''
from scipy import stats

def rate_table(c):
    t = c.groupby("arm").agg(n=("event", "size"),
                             events=("event", "sum"),
                             py=("time", "sum"))
    t["py"] = t["py"] / 365.25
    d = t["events"]
    t["rate"] = 1000 * d / t["py"]
    t["lo"] = 1000 * stats.chi2.ppf(0.025, 2 * d) / 2 / t["py"]
    t["hi"] = 1000 * stats.chi2.ppf(0.975, 2 * d + 2) / 2 / t["py"]
    return t

def irr(t):
    ratio = t.loc["SGLT2i", "rate"] / t.loc["DPP4i", "rate"]
    se = np.sqrt((1 / t["events"]).sum())
    lo, hi = np.exp(-1.96 * se), np.exp(1.96 * se)
    return ratio, ratio * lo, ratio * hi

tab = rate_table(cohort)
print(tab.round(2))
print("IRR %.2f (%.2f-%.2f)" % irr(tab))
''', title="사건 수, 인년, 1,000인년당 발생률과 발생률비")
save("lab14_rate", c, marks=M.get("rate"))

c = nb.cell('''
def share_vs_rate(t):
    """사건 비율(사건 ÷ 인원)과 발생률(사건 ÷ 인년)을 나란히"""
    s = pd.DataFrame({"events %": 100 * t["events"] / t["n"],
                      "mean FU (y)": t["py"] / t["n"],
                      "rate": t["rate"]})
    ratio = s.loc["SGLT2i"] / s.loc["DPP4i"]
    print("ratio of event shares: %.2f" % ratio["events %"])
    print("rate ratio           : %.2f" % ratio["rate"])
    return s.round(2)

share_vs_rate(tab)
''', title="사건 비율로 비교하면")
save("lab14_share", c, marks=M.get("share"), dfmarks=M.get("share_df"))

c = nb.cell('''
g = cohort.groupby("arm")
t1 = g[covs].mean().T
t1.loc[covs[1:]] = t1.loc[covs[1:]] * 100      # 0/1 변수는 %
t1.loc["age SD"] = g["age"].std()
t1.loc["n"] = g.size()
t1.round(1)
''', title="기저 특성 표")
save("lab14_table1", c, dfmarks=M.get("table1_df"))

c = nb.cell('''
from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(6.5, 4))
for arm in ["DPP4i", "SGLT2i"]:
    s = cohort[cohort["arm"] == arm]
    km = KaplanMeierFitter()
    km.fit(s["time"] / 365.25, s["event"], label=arm)
    km.plot_cumulative_density(ax=ax)
ax.set_xlabel("Years since index date (as-treated)")
ax.set_ylabel("Cumulative incidence of HF hospitalization")
plt.tight_layout()
''', title="Kaplan-Meier 누적발생 곡선")
save("lab14_km", c)

c = nb.cell('''
from lifelines import CoxPHFitter

def hr(c, cols):
    m = CoxPHFitter().fit(c[["time", "event"] + cols],
                          "time", "event")
    s = m.summary.loc["sglt2"]
    return (s["exp(coef)"], s["exp(coef) lower 95%"],
            s["exp(coef) upper 95%"])

print("crude HR    %.2f (%.2f-%.2f)" % hr(cohort, ["sglt2"]))
print("adjusted HR %.2f (%.2f-%.2f)" % hr(cohort, ["sglt2"] + covs))
''', title="보정 전과 보정 후의 Cox 위험비")
save("lab14_cox", c, marks=M.get("cox"))

c = nb.cell('''
ref = pd.read_csv(BASE + "sglt2_cohort.csv", storage_options=UA)
mine = cohort.sort_values("pid").reset_index(drop=True)
cols = ["pid", "sglt2", "time", "event"] + covs
print(len(mine), len(ref))
print((mine[cols] == ref[cols]).all().all())
''', title="사이트의 분석용 코호트 파일과 대조하기")
save("lab14_check", c, marks=M.get("check"))

c = nb.cell('''
full = pd.read_csv(BASE + "sglt2_cohort_full.csv", storage_options=UA)
full["arm"] = full["sglt2"].map({1: "SGLT2i", 0: "DPP4i"})

def summary(c):
    t = rate_table(c)
    out = {}
    for arm in ["SGLT2i", "DPP4i"]:
        r = t.loc[arm]
        out[arm + " events / PY"] = "%d / %.0f" % (r["events"],
                                                   r["py"])
        out[arm + " rate (95% CI)"] = "%.2f (%.2f-%.2f)" % (
            r["rate"], r["lo"], r["hi"])
    out["IRR (95% CI)"] = "%.2f (%.2f-%.2f)" % irr(t)
    out["crude HR (95% CI)"] = "%.2f (%.2f-%.2f)" % hr(c, ["sglt2"])
    out["adjusted HR (95% CI)"] = "%.2f (%.2f-%.2f)" % hr(
        c, ["sglt2"] + covs)
    return out

pd.DataFrame({"sample (n = 5,552)": summary(cohort),
              "full (n = 166,548)": summary(full)})
''', title="전체 코호트 166,548명으로 같은 분석 하기")
save("lab14_full", c, dfmarks=M.get("full_df"))

c = nb.cell('''
print(share_vs_rate(rate_table(full)))

def median_fu(time, event):
    """역 Kaplan-Meier 중앙 추적기간(년)"""
    km = KaplanMeierFitter().fit(time / 365.25, 1 - event)
    return km.median_survival_time_

t_itt = full["time_itt"].fillna(full["time"])
e_itt = full["event_itt"].fillna(full["event"])
print("median follow-up (reverse KM, years)")
print("  as-treated : %.2f" % median_fu(full["time"], full["event"]),
      "  simple median %.2f" % (full["time"].median() / 365.25))
print("  ITT        : %.2f" % median_fu(t_itt, e_itt),
      "  simple median %.2f" % (t_itt.median() / 365.25))
print("  sample, as-treated : %.2f" %
      median_fu(cohort["time"], cohort["event"]))
''', title="전체 코호트의 사건 비율과 중앙 추적기간")
save("lab14_fullfu", c, marks=M.get("fullfu"))

# ================================================================ e. 과제 (정답 셀)
c = nb.cell('''
itt = follow_up(cohort, ["hhf_date", "death_date", "study_end"])
ti = rate_table(itt)
print(ti.round(2))
print("IRR %.2f (%.2f-%.2f)" % irr(ti))
print("crude HR    %.2f (%.2f-%.2f)" % hr(itt, ["sglt2"]))
print("adjusted HR %.2f (%.2f-%.2f)" % hr(itt, ["sglt2"] + covs))
off = ti[["events", "py"]] - tab[["events", "py"]]
print((1000 * off["events"] / off["py"]).round(2))

full_itt = full.copy()         # 전체 코호트의 ITT 추적
full_itt["time"] = full["time_itt"].fillna(full["time"])
full_itt["event"] = full["event_itt"].fillna(full["event"])
pd.DataFrame({"sample (n = 5,552)": summary(itt),
              "full (n = 166,548)": summary(full_itt)})
''', title="과제 1 정답. ITT 방식의 추적과 해석")
save("lab14_hw1", c, marks=M.get("hw1"), dfmarks=M.get("hw1_df"))

c = nb.cell('''
res = []
for grace in [30, 60, 90]:
    cg = cohort.copy()
    cg["stop_date"] = cg["pid"].map(stop_date(own, grace))
    cg = follow_up(cg, AT)
    t = rate_table(cg)
    res.append([grace,
                "%d / %d" % tuple(t.loc[["SGLT2i", "DPP4i"], "events"]),
                "%.2f / %.2f" % tuple(t.loc[["SGLT2i", "DPP4i"], "rate"]),
                "%.2f" % irr(t)[0],
                "%.2f (%.2f-%.2f)" % hr(cg, ["sglt2"] + covs)])
    x = cg.loc[cg["pid"] == p].iloc[0]
    print(f"grace {grace}: {x['end_date'].date()},",
          f"day {x['time']}, event {x['event']}")
pd.DataFrame(res, columns=["grace", "events S / D", "rate S / D",
                           "IRR", "adjusted HR"]).set_index("grace")
''', title="과제 2 정답. 유예기간 30일, 60일, 90일")
save("lab14_hw2", c, marks=M.get("hw2"), dfmarks=M.get("hw2_df"))

c = nb.cell('''
va = v[v["pid"].isin(cohort["pid"])]      # 코호트의 모든 명세서
pre = va[va["before"] >= 0]               # 첫 처방일 당일까지
c_any, c_pre = cohort.copy(), cohort.copy()
for name, codes in dx_def.items():
    hit_any = va[has_dx(va, codes)]["pid"]
    hit_pre = pre[has_dx(pre, codes)]["pid"]
    c_any[name] = c_any["pid"].isin(hit_any).astype(int)
    c_pre[name] = c_pre["pid"].isin(hit_pre).astype(int)

for name, d in [("365 days", cohort), ("all before index", c_pre),
                ("whole period", c_any)]:
    print(f"hf = 1, {name:16s}:", d["hf"].sum())
print("events with hf = 1 in c_any:",
      c_any.loc[c_any["event"] == 1, "hf"].mean())
for name, d in [("365 days", cohort), ("all before index", c_pre),
                ("whole period", c_any)]:
    print(f"{name:16s} HR %.2f (%.2f-%.2f)" % hr(d, ["sglt2"] + covs))
''', title="과제 3 정답. 자료 기간 전체로 만든 동반질환")
save("lab14_hw3", c, marks=M.get("hw3"))

print("lab14: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- 노트북 (pub/notebooks/lab14.ipynb)
HW = {
    31: "## 과제 정답\n\n**과제 1.** 약 중단과 비교약으로의 변경을 무시하는 ITT 방식으로 추적을 다시 만들고, 발생률, 발생률비, "
        "보정 전과 보정 후의 위험비를 구합니다. 보정 위험비 행을 Results 문장으로 옮기고, as-treated 분석의 0.61과 "
        "어떤 질문의 답이 다른지, 이 값이 말해 주지 않는 것은 무엇인지 씁니다. 전체 코호트 파일로 본문 표 14-5와도 맞춰 봅니다.",
    32: "**과제 2.** 유예기간을 30일, 60일, 90일로 바꿔 as-treated 추적을 다시 만들고, 군별 사건 수, 발생률, 발생률비, "
        "보정 위험비를 비교합니다. a 절에서 따라가 본 환자의 추적이 어떻게 달라지는지도 확인합니다.",
    33: "**과제 3.** 동료가 자료 기간 전체(2015–2022년)의 명세서로 동반질환을 다시 정의한 분석을 재현하고, 틀린 곳 두 가지를 "
        "찾습니다. 첫 처방일 당일까지의 모든 기록으로 정의한 결과와도 비교합니다.",
}
NOTES = {
    1: "## a. 청구자료의 표 구조\n\n자격, 명세서·상병, 처방 세 표를 불러와 행의 단위를 확인하고 한 사람의 기록을 따라가 봅니다. "
       "Colab에서는 런타임이 새로 배정될 때마다 lifelines를 설치합니다.",
    2: "출력을 만든 패키지의 버전을 확인합니다. 내 화면의 출력이 사이트와 다를 때 먼저 비교합니다.",
    7: "## b. 신규 사용자 코호트 만들기\n\nindex date를 찾고, 선정 기준을 하나씩 표지로 만든 뒤 차례로 적용하며 인원을 셉니다.",
    14: "세척 기간을 180일로 줄인 민감도 분석입니다. 선정 흐름만 다시 셉니다.",
    15: "## c. 노출·결과·공변량 만들기\n\n처방 기록에서 중단일과 변경일을, 명세서에서 심부전 입원일과 기저 동반질환을 만듭니다.",
    17: "14장 c 절의 예제 환자(PDC 315/365)로 함수를 먼저 확인한 뒤 코호트 전체에 적용합니다.",
    22: "## d. 추적 기간과 발생률\n\n추적 종료일을 정하고 인년, 발생률, 기저 특성 표, 누적발생 곡선, 위험비를 구합니다.",
    29: "전체 코호트 파일(5MB)을 읽고 Cox 모형 네 개를 적합하므로 Colab에서는 수십 초가 걸릴 수 있습니다.",
}
NOTES.update(HW)
path = nb.save_ipynb(
    "실습 14. 청구자료에서 코호트 만들기",
    intro=("사회약학 연구방법 노트의 '실습 14 청구자료에서 코호트 만들기'에 딸린 노트북입니다. 청구자료 모양의 가상 자료 "
           "세 표(자격, 명세서·상병, 처방)에서 SGLT2 억제제와 DPP-4 억제제 신규 사용자 코호트를 만들고, 추적 기간과 심부전 "
           "입원의 발생률, 위험비를 구합니다. 자료는 14장 예제 연구의 30분의 1 표본이며 실제 환자 자료가 아닙니다. "
           "셀을 위에서부터 차례로 실행하세요. 각 셀의 설명과 출력 읽는 법은 사이트의 실습 14 쪽에 있습니다."),
    notes=NOTES)
print("notebook:", path)

# ================================================================ 본문(14장) 숫자와 대조
if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    ns = nb.ns
    J = json.load(open(os.path.join(HERE, "_ch14_nums.json")))
    F, FRAC = J["flow"], 30
    n_exact = 0

    # (1) 선정 흐름: 단계마다 본문 인원의 1/30(반올림)과 정확히 같다
    flow = ns["flow"]
    want_left = ([round(F["nodm"]["N_all"] / FRAC), round(F["N0"] / FRAC)]
                 + [round(s["left"] / FRAC) for s in F["steps"]])
    want_out = [round(F["nodm"]["n"] / FRAC)] + [round(s["n"] / FRAC) for s in F["steps"]]
    assert flow["remaining"].tolist() == want_left, (flow["remaining"].tolist(), want_left)
    assert [int(x) for x in flow["excluded"].tolist()[1:]] == want_out
    n_exact += len(want_left) + len(want_out)
    cohort = ns["cohort"]
    assert (cohort["arm"] == "SGLT2i").sum() == round(F["nS"] / FRAC) and len(cohort) == round(F["n"] / FRAC)
    n_exact += 2
    pct = flow["excluded"].iloc[2] / flow["remaining"].iloc[1]
    assert abs(pct - F["steps"][0]["n"] / F["N0"]) < 0.0005          # 41.0%
    print(f"CHECK flow: remaining {want_left}, excluded {want_out} = chapter / 30 (rounded); prevalent {100 * pct:.1f}%")

    # (2) 사람별 대조: 파이프라인으로 만든 코호트 = data_lab14.py가 nums_ch14의 값에서 직접 쓴 sglt2_cohort.csv
    ref = pd.read_csv(os.path.join(PUBDATA, "sglt2_cohort.csv"))
    mine = cohort.sort_values("pid").reset_index(drop=True)
    itt = ns["itt"].sort_values("pid").reset_index(drop=True)
    for col in ["pid", "arm", "sglt2", "age", "female", "htn", "ihd", "hf", "ckd", "met", "ins", "time", "event"]:
        assert (mine[col].values == ref[col].values).all(), col
    assert (mine["index_date"].dt.strftime("%Y-%m-%d").values == ref["index_date"].values).all()
    assert (mine["end_date"].dt.strftime("%Y-%m-%d").values == ref["end_date"].values).all()
    lab = {"hhf_date": "hhf", "death_date": "death", "switch_date": "switch", "stop_date": "stop", "study_end": "study_end"}
    assert (mine["reason"].map(lab).values == ref["reason"].values).all()
    assert (itt["time"].values == ref["time_itt"].values).all() and (itt["event"].values == ref["event_itt"].values).all()
    print(f"CHECK person-level: {len(mine)} people identical to the chapter cohort in 17 variables")

    # (3) 표본의 발생률, 발생률비, 위험비: 본문과 소수 둘째 자리까지 (유예 60일, 30일, 90일, ITT)
    def same2(a, b, what):
        assert round(a, 2) == round(b, 2), (what, a, b)
        return 1

    hrf, covs = ns["hr"], ns["covs"]
    by_grace = {}
    for G in (30, 90):
        cg = cohort.copy()
        cg["stop_date"] = cg["pid"].map(ns["stop_date"](ns["own"], G))
        by_grace[G] = ns["follow_up"](cg, ns["AT"])
    for key, data in (("at60", cohort), ("at30", by_grace[30]), ("at90", by_grace[90]), ("itt", ns["itt"])):
        fu = J["fu"][key]
        tb = ns["rate_table"](data)
        for arm, lab_ in (("SGLT2i", "S"), ("DPP4i", "D")):
            n_exact += same2(tb.loc[arm, "rate"], fu[lab_]["rate"], (key, arm, "rate"))
            assert tb.loc[arm, "events"] == round(fu[lab_]["ev"] / FRAC)
            assert abs(tb.loc[arm, "py"] / (fu[lab_]["py"] / FRAC) - 1) < 0.035, (key, arm, "py")
        r = ns["irr"](tb)[0]
        n_exact += same2(r, fu["irr"], (key, "irr"))
        cr, ad = hrf(data, ["sglt2"])[0], hrf(data, ["sglt2"] + covs)[0]
        n_exact += same2(cr, fu["crude"]["hr"], (key, "crude")) + same2(ad, fu["adj"]["hr"], (key, "adj"))
        print(f"CHECK sample {key}: rate {tb.loc['SGLT2i', 'rate']:.3f} / {tb.loc['DPP4i', 'rate']:.3f} (chapter {fu['S']['rate']:.3f} / "
              f"{fu['D']['rate']:.3f}), IRR {r:.4f} ({fu['irr']:.4f}), crude HR {cr:.4f} ({fu['crude']['hr']:.4f}), "
              f"adjusted HR {ad:.4f} ({fu['adj']['hr']:.4f});  PY {tb.loc['SGLT2i', 'py']:.0f} / {tb.loc['DPP4i', 'py']:.0f} "
              f"(chapter/30 {fu['S']['py'] / FRAC:.0f} / {fu['D']['py'] / FRAC:.0f})")

    # (4) 전체 코호트 파일: 표 14-5의 모든 값이 신뢰구간까지 같다 (as-treated, ITT)
    n_full = 0
    for key, data in (("at60", ns["full"]), ("itt", ns["full_itt"])):
        fu = J["fu"][key]
        tb = ns["rate_table"](data)
        for arm, lab_ in (("SGLT2i", "S"), ("DPP4i", "D")):
            assert tb.loc[arm, "n"] == fu[lab_]["n"] and tb.loc[arm, "events"] == fu[lab_]["ev"]
            assert abs(tb.loc[arm, "py"] - fu[lab_]["py"]) < 1e-6
            for a, b in ((tb.loc[arm, "rate"], fu[lab_]["rate"]), (tb.loc[arm, "lo"], fu[lab_]["lo"]), (tb.loc[arm, "hi"], fu[lab_]["hi"])):
                assert abs(a - b) < 1e-6, (key, arm, a, b)
            n_full += 5
            # 1인당 평균 추적(본문 d 절 2단계 3번: 1.87, 2.18, 3.00, 3.66년)
            assert abs(tb.loc[arm, "py"] / tb.loc[arm, "n"] - fu[lab_]["mean_fu"]) < 1e-9
            n_full += 1
        for a, b in zip(ns["irr"](tb), (fu["irr"], fu["irr_lo"], fu["irr_hi"])):
            n_full += same2(a, b, (key, "full irr"))        # 1.96 대신 정확한 분위수를 쓴 본문과 둘째 자리까지
        for nm, cols in (("crude", ["sglt2"]), ("adj", ["sglt2"] + covs)):
            for a, b in zip(hrf(data, cols), (fu[nm]["hr"], fu[nm]["lo"], fu[nm]["hi"])):
                n_full += same2(a, b, (key, nm, "full"))
    # 셀 29: 역 Kaplan-Meier 중앙 추적기간 (본문 d 절 '더 알아보기' 1.80, 3.50년)
    mf = ns["median_fu"]
    assert abs(mf(ns["full"]["time"], ns["full"]["event"]) - J["medfu"]["at_rkm"]) < 1e-9
    assert abs(mf(ns["t_itt"], ns["e_itt"]) - J["medfu"]["itt_rkm"]) < 1e-9
    assert abs(ns["full"]["time"].median() / 365.25 - J["medfu"]["at_simple"]) < 1e-9
    assert abs(ns["t_itt"].median() / 365.25 - J["medfu"]["itt_simple"]) < 1e-9
    n_full += 4
    print(f"CHECK full cohort file: {n_full} numbers of table 14-5 (events, PY, rates and exact CIs, IRR, crude and adjusted HR "
          f"with CIs; as-treated and ITT), mean and reverse-KM median follow-up equal the chapter")

    # (5) 기저 특성: 본문 표 14-4와 소수 첫째 자리까지
    g = cohort.groupby("arm")
    for arm, lab_ in (("SGLT2i", "S"), ("DPP4i", "D")):
        d = g.get_group(arm)
        for v in covs[1:]:
            assert round(100 * d[v].mean(), 1) == round(100 * J["base"][v][lab_][1], 1), (arm, v)
            n_exact += 1
        assert round(d["age"].mean(), 1) == round(J["base"]["age"][lab_][0], 1)
        assert round(d["age"].std(), 1) == round(J["base"]["age"][lab_][1], 1)
        n_exact += 2
    print("CHECK baseline: 7 percentages, mean age and SD per arm equal the chapter to one decimal")

    # (6) 추적 종료 사유: 본문 인원의 1/30과 2명 이내 (층별 인원을 정수로 나누면서 생기는 차이)
    rs = pd.crosstab(cohort["reason"], cohort["arm"])
    for arm, lab_ in (("SGLT2i", "S"), ("DPP4i", "D")):
        for col, k in (("hhf_date", "hhf"), ("death_date", "death"), ("stop_date", "disc"), ("switch_date", "switch"),
                       ("study_end", "admin")):
            assert abs(rs.loc[col, arm] - J["fu"]["at60"]["reasons"][lab_][k] / FRAC) < 2.0, (arm, col)
    print("CHECK reasons: within 2 people of chapter / 30")

    # (7) 셀 16, 19: 예제 환자의 PDC, 양성예측도의 신뢰구간 (본문과 같음)
    assert round(ns["pdc_ex"] * 365) == J["ex"]["pdc_days"] == 315 and abs(ns["pdc_ex"] - J["ex"]["pdc"]) < 1e-12
    from statsmodels.stats.proportion import proportion_confint as _pc
    lo_, hi_ = _pc(171, 200, method="wilson")
    assert abs(lo_ - J["val"]["lo"]) < 1e-12 and abs(hi_ - J["val"]["hi"]) < 1e-12
    n_exact += 3
    print(f"CHECK total: sample {n_exact} numbers + full cohort {n_full} numbers agree with the chapter")
    lo, hi = hrf(cohort, ["sglt2"] + covs)[1:]
    print(f"NOTE sample adjusted HR 95% CI {lo:.2f}-{hi:.2f}; chapter {J['fu']['at60']['adj']['lo']:.2f}-{J['fu']['at60']['adj']['hi']:.2f}")
    adh = ns["adh"]
    for arm, lab_ in (("SGLT2i", "S"), ("DPP4i", "D")):
        P = J["pdc"][lab_]
        got = [adh.loc[arm, "PDC >= 0.8"], adh.loc[arm, "1 y, grace 30"], adh.loc[arm, "1 y, grace 60"]]
        want = [P["ge80"], P["pers"]["30"], P["pers"]["60"]]
        print(f"NOTE PDC/persistence {arm}: sample " + ", ".join(f"{100 * x:.1f}" for x in got)
              + "  chapter " + ", ".join(f"{100 * x:.1f}" for x in want))
        assert max(abs(a - b) for a, b in zip(got, want)) < 0.021        # 글: 본문과 2%p 안쪽으로 다르다

    # ---------------------------------------------------------------- 본문 글에 인용한 숫자 (셀로 보여 주지 않는 것)
    m, coh, index, study = ns["m"], ns["coh"], ns["index"], ns["study"]
    arm_of = coh.set_index("pid")["arm"]
    same = m[(m["before"] >= 1) & (m["before"] <= 365) & (m["drug"] == m["pid"].map(arm_of))]
    print("VERIFY washout checking own class only:", coh["pid"].isin(same["pid"]).sum(), "(either class 4103)")
    assert coh["pid"].isin(same["pid"]).sum() == 3862
    tab = ns["tab"]
    print("VERIFY mean follow-up (y):", (tab["py"] / tab["n"]).round(2).to_dict(), " events/n %:",
          (100 * tab["events"] / tab["n"]).round(1).to_dict())
    ti = ns["ti"]
    print("VERIFY ITT mean follow-up:", (ti["py"] / ti["n"]).round(2).to_dict())
    print("VERIFY reasons %:", (100 * rs / rs.sum()).round(1).to_dict())
    person = ns["person"]
    print("VERIFY deaths in person table:", int(person["death_date"].notna().sum()),
          " elig_end != 2022-12-31:", int((person["elig_end"] != pd.Timestamp("2022-12-31")).sum()))
    rx, visit = ns["rx"], ns["visit"]
    print("VERIFY rx per person median/max:", rx.groupby("pid").size().median(), rx.groupby("pid").size().max())
    own = ns["own"]
    print("VERIFY own-class fills:", len(own), " people:", own["pid"].nunique(), " gaps > 30:", int((own["gap"] > 30).sum()),
          " > 60:", int((own["gap"] > 60).sum()), " last fill (NaN gap):", int(own["gap"].isna().sum()))
    print("VERIFY HHF: people with >1 primary inpatient I50 claim:",
          int((ns["adm"].groupby("pid").size() > 1).sum()), " cohort hhf_date not null:", int(cohort["hhf_date"].notna().sum()))
    b = ns["base"]
    v = ns["v"]
    print("VERIFY visits in baseline window:", len(b), "of", len(v))
    print("VERIFY sex/age of example:", person.loc[person["pid"] == P_EX].to_dict("records"))
    print("VERIFY example covs:", cohort.loc[cohort["pid"] == P_EX, ["age"] + covs[1:] + ["switch_date", "stop_date", "hhf_date"]].to_dict("records"))
    print("VERIFY washout 180:", ns["flow180"]["remaining"].tolist(), ns["flow180"]["excluded"].tolist(),
          ns["coh180"][ns["keep180"]]["arm"].value_counts().to_dict())
    # 셀 16: PDC 0.8 이상이면서 유예 30일 기준 1년 안에 중단한 사람 (본문 c 절의 예제 환자와 같은 경우)
    st30 = (cohort["pid"].map(ns["stop_date"](own, 30)) - cohort["index_date"]).dt.days
    both = (cohort["pdc"] >= 0.8) & (st30 <= 365)
    print(f"VERIFY PDC>=0.8 yet stopped within 1 y (grace 30): {int(both.sum())} ({100 * both.mean():.1f}%)",
          f" chapter {J['pdc_vs_pers']['n']} ({100 * J['pdc_vs_pers']['pct']:.1f}%)")
    print("VERIFY PDC mean/median by arm:", cohort.groupby("arm")["pdc"].agg(["mean", "median"]).round(3).to_dict())
    print("VERIFY deaths within first year:", int(((cohort["death_date"] - cohort["index_date"]).dt.days <= 365).sum()))
    # 셀 23, 29: 사건 비율의 비와 발생률비
    full = ns["full"]
    tf = ns["rate_table"](full)
    sh = tf["events"] / tf["n"]
    print("VERIFY full: share ratio %.4f, IRR %.4f, mean FU %s" % (sh["SGLT2i"] / sh["DPP4i"], ns["irr"](tf)[0],
                                                                 (tf["py"] / tf["n"]).round(3).to_dict()))
    assert "%.2f" % (sh["SGLT2i"] / sh["DPP4i"]) == "0.37"
    # 과제 1: ITT에만 있는 기간의 발생률 (SGLT2 억제제군 8건, 1,031인년)
    off = ti[["events", "py"]] - tab[["events", "py"]]
    print("VERIFY ITT-only period:", off.round(1).to_dict())
    assert int(off.loc["SGLT2i", "events"]) == 8 and round(ti.loc["SGLT2i", "py"]) - round(tab.loc["SGLT2i", "py"]) == 1031      # 3,017 − 1,986
    # 과제 3: 자료 기간 전체로 정의한 공변량 (글에 인용한 숫자)
    c_any, c_pre = ns["c_any"], ns["c_pre"]
    print("VERIFY hw3 sums:", {k: int(x[k].sum()) for k in ("htn", "ihd", "hf", "ckd") for x in [c_any]},
          {k: int(c_pre[k].sum()) for k in ("htn", "ihd", "hf", "ckd")}, {k: int(cohort[k].sum()) for k in ("htn", "ihd", "hf", "ckd")})
    print("VERIFY hw3 hf share by arm (365 d / whole period):",
          cohort.groupby("arm")["hf"].mean().round(3).to_dict(), c_any.groupby("arm")["hf"].mean().round(3).to_dict())
    print("VERIFY hw3 ckd share by arm (365 d / whole period):",
          cohort.groupby("arm")["ckd"].mean().round(3).to_dict(), c_any.groupby("arm")["ckd"].mean().round(3).to_dict())
    n_out_only = int((c_any["hf"].astype(bool) & ~c_pre["hf"].astype(bool)).sum())
    print("VERIFY hw3 hf=1 only from records after the index date:", n_out_only,
          " of whom had the outcome (as-treated event):", int((c_any["hf"].astype(bool) & ~c_pre["hf"].astype(bool) & (cohort["event"] == 1)).sum()),
          " had any primary inpatient I50 after index:", int(cohort.loc[c_any["hf"].astype(bool) & ~c_pre["hf"].astype(bool), "hhf_date"].notna().sum()))
    # 과제 3: 동료의 모형에서 lifelines가 내는 경고(글에서 설명한다). 다른 셀에는 경고가 없어야 한다.
    warned = {cc.n for cc in nb.cells if cc.warns}
    assert warned == {33} and all("ConvergenceWarning" in w for w in nb.cells[32].warns), warned
    print("VERIFY pandas", pd.__version__, "numpy", np.__version__)
