"""실습 16 · 불멸시간 편향과 시간의존 Cox 모형 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab16.py && python3 gen/lab_lab16.py
      (NOMARK=1 python3 gen/lab_lab16.py prints every cell's raw output without marks)

Data: pub/data/ami_cohort.csv (gen/data_lab16.py; the chapter 16 cohort of gen/nums_ch16.py).
The check block at the end compares the lab's numbers with gen/_ch16_nums.json.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(0)
nb = Notebook("lab16")


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


def freeze_clock(h):
    """lifelines의 print_summary()가 찍는 적합 시각(time fit was run 줄)을 고정값으로 바꾼다. 다시 돌려도
    같은 파일이 나오게 하려는 것이고 통계 출력은 건드리지 않는다."""
    return re.sub(r"(time fit was run = )\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC",
                  r"\g<1>2026-10-03 16:32:00 UTC", h)


def save(name, c, marks=None, dfmarks=None):
    if os.environ.get("NOMARK"):
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        marks = dfmarks = None
    nb.save_fragment(name, freeze_clock(render(c, marks, dfmarks)))


PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

NOTES = {}      # {cell number: markdown shown before that cell in the .ipynb}

# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab16_install", c)
NOTES[c.n] = "## 가. 실습 데이터 준비"

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
d = pd.read_csv(BASE + "ami_cohort.csv")
print(d.shape)
d.head(5)
''', title="자료 불러오기")
save("lab16_load", c, marks={"(6000, 4)": 1}, dfmarks={"177.836900": 2, "NaN": 3, "249.905124": 4})

c = nb.cell('''
print("환자", len(d), "명, id 중복", d["id"].duplicated().sum(), "건")
print("사망", d["died"].sum(), "명",
      f"({d['died'].mean() * 100:.1f}%)")
print("약물 X를 시작한 사람", d["rx_day"].notna().sum(), "명")
alive = d[d["died"] == 0]
print("생존자의 추적 일수:", alive["fu_days"].unique())
print("사망자의 추적 일수 최대:", d.loc[d["died"] == 1, "fu_days"].max())
bad = (d["rx_day"] >= d["fu_days"]).sum()
print("추적이 끝난 뒤의 처방", bad, "건")
''', title="자료 확인")
save("lab16_check", c, marks={"사망 1234 명 (20.6%)": 1, "약물 X를 시작한 사람 2734 명": 2,
                              "생존자의 추적 일수: [365.]": 3, "추적이 끝난 뒤의 처방 0 건": 4})

c = nb.cell('''
rx_month = d["rx_day"] / 365 * 12          # 일 -> 개월
print(rx_month.describe().round(1))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.hist(d["rx_day"].dropna(), bins=24)
ax.set_xlabel("Days from discharge to first prescription")
ax.set_ylabel("Number of patients")
plt.tight_layout()
''', title="약물 X를 시작한 시점의 분포")
save("lab16_rxdist", c, marks={"count    2734.0": 1, "25%         2.1": 2, "50%         4.5": 3, "75%         7.8": 4})

# ---------------------------------------------------------------- 나. 불멸시간 편향 재현하기
c = nb.cell('''
d["ever"] = d["rx_day"].notna().astype(int)   # 1년 안에 시작 = 1

t1 = d.groupby("ever")["died"].agg(["count", "sum", "mean"])
t1.columns = ["n", "deaths", "risk"]
print("상대위험도:", round(t1.loc[1, "risk"] / t1.loc[0, "risk"], 2))
t1["risk"] = (t1["risk"] * 100).round(1)       # 1년 사망 위험(%)
t1
''', title="추적 중 사용 여부로 군 나누기 (잘못된 분류)")
save("lab16_ever", c, marks={"상대위험도: 0.42": 1}, dfmarks={"3266": 2, "28.0": 3, "11.7": 4})
NOTES[c.n] = ("## 나. 불멸시간 편향 재현하기\n\n일부러 잘못된 분석을 합니다. 추적 중 한 번이라도 약을 시작한 사람을 "
              "처음부터 사용군으로 두고 퇴원일부터 추적합니다.")

c = nb.cell('''
from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(6.5, 4))
for g, name in [(1, "Ever users"), (0, "Never users")]:
    s = d[d["ever"] == g]
    km = KaplanMeierFitter().fit(s["fu_days"], s["died"], label=name)
    km.plot_survival_function(ax=ax, ci_show=False)
    print(name, "90일 생존율", round(km.predict(90), 3),
          " 365일 생존율", round(km.predict(365), 3))
ax.set_xlabel("Days since discharge")
ax.set_ylabel("Survival probability")
ax.set_ylim(0.6, 1)
plt.tight_layout()
''', title="잘못된 분류로 그린 Kaplan-Meier 곡선")
save("lab16_naivekm", c, marks={"Ever users 90일 생존율 0.987": 1, "Never users 90일 생존율 0.884": 2})

c = nb.cell('''
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(d, duration_col="fu_days", event_col="died", formula="ever")
cph.print_summary(decimals=3)

def hr_ci(model, name):
    r = model.summary.loc[name]
    lo, hi = r["exp(coef) lower 95%"], r["exp(coef) upper 95%"]
    return f"{r['exp(coef)']:.2f} ({lo:.2f}-{hi:.2f})"

print("HR (95% CI):", hr_ci(cph, "ever"))
''', title="잘못된 분류의 Cox 모형")
save("lab16_naivecox", c, marks={"number of events observed = 1234": 1, "-1.004": 2, "0.366": 3, "0.323": 4,
                                 "0.416": 5, "<0.0005": 6, "HR (95% CI): 0.37 (0.32-0.42)": 7})

c = nb.cell('''
CoxPHFitter().fit(d, duration_col="fu_days", event_col="died")
''', title="formula를 빼면 (결측값이 있는 열까지 모형에 들어감)", expect_error=True)
save("lab16_naerror", c)
ERR_CELL = c.n

c = nb.cell('''
u = d[d["ever"] == 1]                  # 사용군 2,734명
nu = d[d["ever"] == 0]                 # 비사용군 3,266명
py_u = u["fu_days"].sum() / 365        # 사용군 인년 (퇴원일부터)
py_nu = nu["fu_days"].sum() / 365
immortal = u["rx_day"].sum() / 365     # 퇴원일부터 첫 처방일까지
print(f"사용군 {py_u:.0f}인년, 그중 첫 처방 전 {immortal:.0f}인년"
      f" ({immortal / py_u * 100:.0f}%)")
print(f"비사용군 {py_nu:.0f}인년")
rate_u = u["died"].sum() / py_u * 100       # 100인년당 사망률
rate_nu = nu["died"].sum() / py_nu * 100
print(f"사망률 {rate_u:.1f} 대 {rate_nu:.1f},"
      f" 발생률비 {rate_u / rate_nu:.2f}")
''', title="사용군의 불멸 인년 계산")
save("lab16_immortal", c, marks={"사용군 2613인년": 1, "첫 처방 전 1142인년 (44%)": 2, "비사용군 2699인년": 3,
                                 "사망률 12.3 대 33.8": 4, "발생률비 0.36": 5})
NOTES[c.n] = ("셀 8은 일부러 오류를 내 보는 셀이어서 이 노트북에서는 뺐습니다(사이트의 실습 16 나 절에 있습니다). "
              "셀 번호는 사이트와 같게 두었으므로 셀 7 다음이 셀 9입니다.")

c = nb.cell('''
print("사용군에서 첫 처방 전에 사망한 사람:",
      (u["fu_days"] < u["rx_day"]).sum(), "명")
early = d[(d["died"] == 1) & (d["fu_days"] <= 365 * 3 / 12)]
print("퇴원 후 3개월 안의 사망:", len(early), "건")
print(early["ever"].value_counts())
''', title="일찍 사망한 사람은 어느 군에 있는가")
save("lab16_early", c, marks={"첫 처방 전에 사망한 사람: 0 명": 1, "3개월 안의 사망: 417 건": 2, "0    380": 3, "1     37": 4})

# ---------------------------------------------------------------- 다. 랜드마크 분석
c = nb.cell('''
L = 365 * 3 / 12                           # 3개월 = 91.25일
lm = d[d["fu_days"] > L].copy()            # 그때 살아 있는 사람만
lm["exposed"] = (lm["rx_day"] <= L).astype(int)   # 그때까지 시작 = 1
lm["t_lm"] = lm["fu_days"] - L             # 랜드마크부터의 일수
print("대상자", len(lm), "명, 제외", len(d) - len(lm), "명")
late = ((lm["exposed"] == 0) & (lm["ever"] == 1)).sum()
print("비노출군 가운데 3개월 뒤에 시작한 사람", late, "명")

t2 = lm.groupby("exposed")["died"].agg(["count", "sum", "mean"])
t2.columns = ["n", "deaths", "risk"]
t2["risk"] = (t2["risk"] * 100).round(1)
t2
''', title="3개월 랜드마크 자료 만들기")
save("lab16_lmdata", c, marks={"대상자 5583 명": 1, "제외 417 명": 2, "시작한 사람 1763 명": 3},
     dfmarks={"4649": 4, "934": 5, "14.8": 6, "14.0": 7})
NOTES[c.n] = "## 다. 랜드마크 분석"

c = nb.cell('''
fig, ax = plt.subplots(figsize=(6.5, 4))
for g, name in [(1, "Exposed by 3 months"), (0, "Unexposed")]:
    s = lm[lm["exposed"] == g]
    km = KaplanMeierFitter().fit(s["t_lm"], s["died"], label=name)
    km.plot_survival_function(ax=ax, ci_show=False)
ax.set_xlabel("Days since the 3-month landmark")
ax.set_ylabel("Survival probability")
ax.set_ylim(0.6, 1)
plt.tight_layout()

cph_lm = CoxPHFitter().fit(lm, "t_lm", "died", formula="exposed")
print("3개월 랜드마크 HR (95% CI):", hr_ci(cph_lm, "exposed"))
''', title="랜드마크부터의 Kaplan-Meier 곡선과 Cox 모형")
save("lab16_lmfit", c, marks={"0.94 (0.78-1.14)": 1})

c = nb.cell('''
def landmark(d, months):
    L = 365 * months / 12
    lm = d[d["fu_days"] > L].copy()
    lm["exposed"] = (lm["rx_day"] <= L).astype(int)
    lm["t_lm"] = lm["fu_days"] - L
    m = CoxPHFitter().fit(lm, "t_lm", "died", formula="exposed")
    e, ne = lm[lm["exposed"] == 1], lm[lm["exposed"] == 0]
    return {"month": months, "n": len(lm),
            "excluded": len(d) - len(lm), "exposed": len(e),
            "late start %": round(ne["ever"].mean() * 100, 1),
            "deaths": f"{e['died'].sum()} / {ne['died'].sum()}",
            "HR (95% CI)": hr_ci(m, "exposed")}

pd.DataFrame([landmark(d, m) for m in [1, 3, 6]])
''', title="랜드마크 시점을 1, 3, 6개월로 바꿔 보기")
save("lab16_lmtable", c, dfmarks={"5820": 1, "349": 2, "43.5": 3, "131 / 686": 4, "1.06 (0.88-1.27)": 5})

# ---------------------------------------------------------------- 라. 시간의존 Cox 모형
c = nb.cell('''
u = d[d["ever"] == 1]        # 약을 시작한 사람: 두 줄
nu = d[d["ever"] == 0]       # 시작하지 않은 사람: 한 줄
before = pd.DataFrame({"id": u["id"], "start": 0.0,
                       "stop": u["rx_day"], "drugx": 0, "died": 0})
after = pd.DataFrame({"id": u["id"], "start": u["rx_day"],
                      "stop": u["fu_days"], "drugx": 1,
                      "died": u["died"]})
never = pd.DataFrame({"id": nu["id"], "start": 0.0,
                      "stop": nu["fu_days"], "drugx": 0,
                      "died": nu["died"]})
long = pd.concat([before, after, never])
long = long.sort_values(["id", "start"]).reset_index(drop=True)
print(len(d), "명 ->", len(long), "줄")
long.head(6)
''', title="한 사람 한 줄 자료를 긴 형식으로 바꾸기")
save("lab16_long", c, marks={"6000 명 -> 8734 줄": 1}, dfmarks={"177.836900": 2, "249.905124": 3, "365.000000": 4})
NOTES[c.n] = "## 라. 시간의존 Cox 모형"

c = nb.cell('''
ex = {298: "A", 64: "B", 3: "C"}       # 16장 표 16-4의 환자
t = long[long["id"].isin(ex)].copy()
t["patient"] = t["id"].map(ex)
t[["start", "stop"]] = t[["start", "stop"]].round().astype(int)
t = t.sort_values(["patient", "start"])
t[["patient", "id", "start", "stop", "drugx", "died"]]
''', title="표 16-4의 환자 세 명 확인")
save("lab16_three", c, dfmarks={"126": 1, "301": 2, "44": 3, "365": 4})

c = nb.cell('''
g = long.groupby("id")
prev_stop = g["stop"].shift()           # 같은 환자의 앞 줄 stop
second = long[prev_stop.notna()]        # 둘째 줄만
print("둘째 줄", len(second), "줄, 앞 줄의 끝에서 시작하는가:",
      (second["start"] == prev_stop.dropna()).all())
print("길이가 0 이하인 구간", (long["stop"] <= long["start"]).sum(), "줄")
last = g.tail(1)                        # 환자마다 마지막 줄
print("사망", long["died"].sum(), "건, 마지막 줄의 사망",
      last["died"].sum(), "건, 원자료", d["died"].sum(), "건")
py_long = (long["stop"] - long["start"]).sum() / 365
print(f"인년 합계 {py_long:.0f}, 원자료 {d['fu_days'].sum() / 365:.0f}")
''', title="긴 형식 자료 점검")
save("lab16_longcheck", c, marks={"둘째 줄 2734 줄": 1, "True": 2, "길이가 0 이하인 구간 0 줄": 3,
                                  "사망 1234 건, 마지막 줄의 사망 1234 건, 원자료 1234 건": 4, "인년 합계 5312, 원자료 5312": 5})

c = nb.cell('''
long["py"] = (long["stop"] - long["start"]) / 365
t3 = long.groupby("drugx").agg(deaths=("died", "sum"),
                               py=("py", "sum"))
t3["rate"] = t3["deaths"] / t3["py"] * 100      # 100인년당 사망률
print("발생률비:", round(t3.loc[1, "rate"] / t3.loc[0, "rate"], 2))
t3.round({"py": 0, "rate": 1})
''', title="노출 상태별 사망, 인년, 사망률")
save("lab16_pytable", c, marks={"발생률비: 0.92": 1}, dfmarks={"913": 2, "3841.0": 3, "23.8": 4, "1471.0": 5, "21.8": 6})

c = nb.cell('''
from lifelines import CoxTimeVaryingFitter

ctv = CoxTimeVaryingFitter()
ctv.fit(long, id_col="id", event_col="died",
        start_col="start", stop_col="stop", formula="drugx")
ctv.print_summary(decimals=3)
''', title="시간의존 Cox 모형")
save("lab16_ctv", c, marks={"number of periods = 8734": 1, "number of events = 1234": 2, "0.031": 3, "1.031": 4,
                            "0.900": 5, "1.182": 6, "0.656": 7})

c = nb.cell('''
from lifelines.utils import to_long_format, add_covariate_to_timeline

base = to_long_format(d[["id", "fu_days", "died"]],
                      duration_col="fu_days")
rx = d.loc[d["ever"] == 1, ["id", "rx_day"]].assign(drugx=1)
long2 = add_covariate_to_timeline(base, rx, id_col="id",
                                  duration_col="rx_day",
                                  event_col="died")
long2["drugx"] = long2["drugx"].fillna(0).astype(int)
ctv2 = CoxTimeVaryingFitter().fit(long2, id_col="id",
    event_col="died", start_col="start", stop_col="stop")
print(len(long2), "줄, HR (95% CI):", hr_ci(ctv2, "drugx"))
long2.head(4)
''', title="lifelines 함수로 같은 자료 만들기")
save("lab16_utils", c, marks={"8734 줄, HR (95% CI): 1.03 (0.90-1.18)": 1}, dfmarks={"249.905124": 2})

c = nb.cell('''
pd.DataFrame({"HR (95% CI)": {
    "ever vs never use, from discharge": hr_ci(cph, "ever"),
    "landmark at 3 months": hr_ci(cph_lm, "exposed"),
    "time-varying exposure": hr_ci(ctv, "drugx")}})
''', title="세 분석의 위험비 비교")
save("lab16_compare", c, dfmarks={"0.37 (0.32-0.42)": 1, "0.94 (0.78-1.14)": 2, "1.03 (0.90-1.18)": 3})
N_MAIN = len(nb.cells)

# ---------------------------------------------------------------- 마. 과제 (정답 셀)
c = nb.cell('''
L = 365 * 3 / 12
lm_w = d[d["fu_days"] > L].copy()          # 3개월 생존자
lm_w["t_lm"] = lm_w["fu_days"] - L
print(lm_w.groupby("ever")["died"].agg(["count", "sum"]))
m1 = CoxPHFitter().fit(lm_w, "t_lm", "died", formula="ever")
print("HR (95% CI):", hr_ci(m1, "ever"))

late = lm_w[lm_w["rx_day"] > L]            # 3개월 뒤에 시작한 사람
print(len(late), "명의 3개월부터 첫 처방까지:",
      round((late["rx_day"] - L).sum() / 365), "인년")
''', title="과제 1 정답. 랜드마크 뒤의 처방까지 넣어 군을 나누면")
save("lab16_hw1", c, marks={"HR (95% CI): 0.53 (0.46-0.62)": 1, "1763 명": 2})
NOTES[c.n] = ("## 과제 정답\n\n**과제 1.** 3개월 랜드마크 분석에서 군을 '3개월까지 시작했는가'(exposed)가 아니라 "
              "'1년 안에 한 번이라도 시작했는가'(ever)로 나누면 위험비가 얼마로 나옵니까?")

c = nb.cell('''
ex2 = pd.concat([
    pd.DataFrame({"t": u["fu_days"] - u["rx_day"],   # 첫 처방일부터
                  "died": u["died"], "ever": 1}),
    pd.DataFrame({"t": nu["fu_days"],                # 퇴원일부터
                  "died": nu["died"], "ever": 0})])
py2 = ex2.groupby("ever")["t"].sum() / 365
rate2 = ex2.groupby("ever")["died"].sum() / py2 * 100
print("인년:", py2.round(0).to_dict())
print("100인년당 사망률:", rate2.round(1).to_dict(),
      " 발생률비:", round(rate2[1] / rate2[0], 2))
m2 = CoxPHFitter().fit(ex2, "t", "died", formula="ever")
print("HR (95% CI):", hr_ci(m2, "ever"))
''', title="과제 2 정답. 첫 처방 전 시간을 버리기만 하면")
save("lab16_hw2", c, marks={"1: 1471.0": 1, "발생률비: 0.65": 2, "HR (95% CI): 0.58 (0.51-0.66)": 3})
NOTES[c.n] = ("**과제 2.** 사용군은 첫 처방일부터, 비사용군은 퇴원일부터 추적하면(첫 처방 전 시간을 버리면) "
              "사망률, 발생률비, 위험비가 어떻게 됩니까?")

c = nb.cell('''
def make_long(d, lag=0):
    sw = d["rx_day"] + lag              # 노출로 바뀌는 날
    s = sw < d["fu_days"]               # 추적 중에 바뀌는 사람
    a, b = d[s], d[~s]
    before = pd.DataFrame({"id": a["id"], "start": 0.0,
                           "stop": sw[s], "drugx": 0, "died": 0})
    after = pd.DataFrame({"id": a["id"], "start": sw[s],
                          "stop": a["fu_days"], "drugx": 1,
                          "died": a["died"]})
    never = pd.DataFrame({"id": b["id"], "start": 0.0,
                          "stop": b["fu_days"], "drugx": 0,
                          "died": b["died"]})
    out = pd.concat([before, after, never])
    return out.sort_values(["id", "start"]).reset_index(drop=True)

long30 = make_long(d, lag=30)
long30["py"] = (long30["stop"] - long30["start"]) / 365
print(len(long30), "줄")
print(long30.groupby("drugx").agg(deaths=("died", "sum"),
                                  py=("py", "sum")).round(0))
m3 = CoxTimeVaryingFitter().fit(long30, id_col="id",
    event_col="died", start_col="start", stop_col="stop",
    formula="drugx")
print("시차 30일 HR (95% CI):", hr_ci(m3, "drugx"))

chk = add_covariate_to_timeline(base, rx, id_col="id",
    duration_col="rx_day", event_col="died", delay=30)
print("lifelines delay=30:", len(chk), "줄")
''', title="과제 3 정답. 시차 30일을 둔 노출")
save("lab16_hw3", c, marks={"8555 줄": 1, "1254.0": 2, "시차 30일 HR (95% CI): 1.03 (0.89-1.19)": 3,
                            "lifelines delay=30: 8555 줄": 4})
NOTES[c.n] = ("**과제 3.** 첫 처방 뒤 30일까지를 비노출로 세는 시차 30일의 긴 형식 자료를 만들고, "
              "노출 상태별 사망 수와 인년, 시간의존 Cox 모형의 위험비를 구하세요.")

print("lab16: cells", len(nb.cells), "(main", N_MAIN, ")")
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- notebook
nb.save_ipynb(
    "실습 16. 불멸시간 편향과 시간의존 Cox 모형",
    intro=("사회약학 연구방법 노트의 '실습 16. 불멸시간 편향과 시간의존 Cox 모형'에 딸린 노트북입니다. "
           "급성 심근경색 퇴원 환자 6,000명의 가상 자료에서, 효과가 없는 약물 X가 잘못된 분석에서는 위험비 0.37로 "
           "나오는 것(불멸시간 편향)을 직접 만들어 보고, 랜드마크 분석과 시간의존 Cox 모형으로 바로잡습니다. "
           "셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 16에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ---------------------------------------------------------------- 대조 블록: 본문(16장) 숫자와 맞추기
ns = nb.ns
J = json.load(open(os.path.join(HERE, "_ch16_nums.json")))["null"]
JJ = json.load(open(os.path.join(HERE, "_ch16_nums.json")))


def hr3(model, name):
    r = model.summary.loc[name]
    return [float(r["exp(coef)"]), float(r["exp(coef) lower 95%"]), float(r["exp(coef) upper 95%"])]


def close(a, b, tol, what):
    a, b = np.atleast_1d(a).astype(float), np.atleast_1d(b).astype(float)
    assert np.all(np.abs(a - b) < tol), f"MISMATCH {what}: lab {a} vs chapter {b}"
    CHK.append(what)


CHK = []
d_, long_, lm_ = ns["d"], ns["long"], ns["lm"]
close(len(d_), J["n"], 0.5, "n 6000")
close(d_.died.sum(), J["deaths"], 0.5, "deaths 1234")
close(d_.ever.sum(), J["users"], 0.5, "users 2734")
close([d_[d_.ever == 1].died.mean() * 100, d_[d_.ever == 0].died.mean() * 100], [J["mort_users"], J["mort_non"]], 1e-9,
      "1-year risk 11.7 / 28.0")
close(ns["rx_month"].median(), J["median_init_months"], 1e-6, "median months to first Rx 4.5")
close([ns["rx_month"].quantile(.25), ns["rx_month"].quantile(.75)], J["init_q1q3_months"], 1e-6, "IQR 2.1-7.8")
close(hr3(ns["cph"], "ever"), J["hr_naive"][:3], 1e-6, "naive HR 0.37 (0.32-0.42)")
close([ns["py_u"], ns["immortal"], ns["py_nu"]], [J["py_user_total"], J["immortal"], J["py_non"]], 1e-4,
      "person-years 2613 / 1142 / 2699")
close([ns["rate_u"], ns["rate_nu"]], J["rate_naive"], 1e-5, "naive rates 12.3 / 33.8")
close(len(ns["early"]), J["deaths_3mo"], 0.5, "deaths within 3 months 417")
close((ns["early"].ever == 0).sum(), J["non_deaths_3mo"], 0.5, "of which never users 380")
L3 = J["lm"]["3"]
close([len(lm_), lm_.exposed.sum(), ns["late"].shape[0]], [L3["n"], L3["exposed"], L3["late_starters"]], 0.5,
      "3-month landmark n 5583 / exposed 934 / late starters 1763")
close([lm_[lm_.exposed == 1].died.sum(), lm_[lm_.exposed == 0].died.sum()], [L3["d_exp"], L3["d_unexp"]], 0.5,
      "3-month landmark deaths 131 / 686")
close([lm_[lm_.exposed == 1].died.mean(), lm_[lm_.exposed == 0].died.mean()], [L3["risk_1"], L3["risk_0"]], 1e-9,
      "3-month landmark risks 14.0 / 14.8")
close(hr3(ns["cph_lm"], "exposed"), L3["hr"][:3], 1e-6, "3-month landmark HR 0.94 (0.78-1.14)")
for mo in (1, 3, 6):
    r, ref = ns["landmark"](d_, mo), J["lm"][str(mo)]
    close([r["n"], r["excluded"], r["exposed"]], [ref["n"], ref["excluded"], ref["exposed"]], 0.5, f"landmark {mo} mo counts")
    close(r["late start %"], ref["late_pct"], 0.051, f"landmark {mo} mo late starters %")
    assert r["HR (95% CI)"] == "%.2f (%.2f-%.2f)" % tuple(ref["hr"][:3]), (r, ref["hr"])
    assert r["deaths"] == "%d / %d" % (ref["d_exp"], ref["d_unexp"])
    CHK.append(f"landmark {mo} mo HR and deaths")
close(len(long_), J["tv_rows"], 0.5, "long format rows 8734")
t3 = ns["t3"]
close([t3.loc[0, "deaths"], t3.loc[1, "deaths"]], [J["d_non"], J["d_users"]], 0.5, "deaths by exposure state 913 / 321")
close([t3.loc[0, "py"], t3.loc[1, "py"]], [J["py_unexposed"], J["py_user_after"]], 1e-4, "person-years 3841 / 1471")
close([t3.loc[1, "rate"], t3.loc[0, "rate"]], J["rate_tv"], 1e-5, "rates 21.8 / 23.8")
close(t3.loc[1, "rate"] / t3.loc[0, "rate"], J["irr_tv"], 1e-6, "crude IRR 0.92")
close(hr3(ns["ctv"], "drugx"), J["hr_tv"][:3], 1e-6, "time-varying HR 1.03 (0.90-1.18)")
close(ns["ctv"].summary.loc["drugx", ["coef", "se(coef)", "z", "p"]].values,
      [JJ["tv_summary"][k] for k in ("coef", "se(coef)", "z", "p")], 1e-6, "coef 0.031, se 0.070, z 0.445, p 0.656")
close(ns["ctv"].log_likelihood_, JJ["tv_loglik"], 1e-3, "partial log-likelihood -10598.599")
close(hr3(ns["ctv2"], "drugx"), JJ["tv_util"][:3], 1e-6, "lifelines utilities HR")
close(len(ns["long2"]), JJ["tv_util_rows"], 0.5, "lifelines utilities rows 8734")
tt = ns["t"]
for p in JJ["ex_patients"]:
    rows = tt[tt.id == p["id"]]
    assert rows["stop"].iloc[-1] == p["end"] and rows["died"].iloc[-1] == p["died"]
    assert (len(rows) == 2) == (p["init"] is not None) and (p["init"] is None or rows["stop"].iloc[0] == p["init"])
CHK.append("table 16-4 patients A, B, C")
close(hr3(ns["m2"], "ever"), J["hr_excl"][:3], 1e-6, "HW2 excluded immortal time HR 0.58 (0.51-0.66)")
close([ns["rate2"][1], ns["rate2"][0], ns["rate2"][1] / ns["rate2"][0]], J["rate_excl"] + [J["irr_excl"]], 1e-5,
      "HW2 rates 21.8 / 33.8, IRR 0.65")
print("CHECK: %d comparisons with gen/_ch16_nums.json all passed" % len(CHK))

# numbers quoted in the text that are not printed by a cell
import warnings  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pd_, CTV = ns["pd"], ns["CoxTimeVaryingFitter"]
    u_, nu_ = ns["u"], ns["nu"]
    badl = pd_.concat([pd_.DataFrame({"id": u_.id, "start": 0.0, "stop": u_.rx_day, "drugx": 0, "died": u_.died}),
                       pd_.DataFrame({"id": u_.id, "start": u_.rx_day, "stop": u_.fu_days, "drugx": 1, "died": u_.died}),
                       pd_.DataFrame({"id": nu_.id, "start": 0.0, "stop": nu_.fu_days, "drugx": 0, "died": nu_.died})])
    mb = CTV().fit(badl, id_col="id", event_col="died", start_col="start", stop_col="stop")
    print("VERIFY died copied to both rows: events", int(badl.died.sum()), "HR", ns["hr_ci"](mb, "drugx"))
    for lag in (0, 14, 60, 90):
        lg = ns["make_long"](d_, lag)
        m = CTV().fit(lg, id_col="id", event_col="died", start_col="start", stop_col="stop")
        print("VERIFY lag", lag, "rows", len(lg), "HR", ns["hr_ci"](m, "drugx"),
              "exposed deaths", int(lg[lg.drugx == 1].died.sum()))
    chk = ns["chk"].fillna({"drugx": 0})
    l30 = ns["long30"]
    print("VERIFY delay=30 identical to make_long:", len(chk) == len(l30),
          int(chk[chk.drugx == 1].died.sum()) == int(l30[l30.drugx == 1].died.sum()),
          ns["hr_ci"](CTV().fit(chk, id_col="id", event_col="died", start_col="start", stop_col="stop"), "drugx"))
    print("VERIFY lag 30 table:\n", l30.groupby("drugx").agg(deaths=("died", "sum"), py=("py", "sum")))
    print("VERIFY switched with lag 30:", int((d_.rx_day + 30 < d_.fu_days).sum()), "never switched users:",
          int(((d_.rx_day + 30 >= d_.fu_days) & (d_.ever == 1)).sum()))
    lm_w = ns["lm_w"]
    print("VERIFY HW1 risks:", lm_w.groupby("ever").died.mean().round(4).to_dict(),
          "late immortal PY", float((ns["late"].rx_day - 365 * 3 / 12).sum() / 365))
    print("VERIFY users started by 3 mo:", int((d_.rx_day <= 91.25).sum()), " min rx_day", float(d_.rx_day.min()),
          " users dying:", int(u_.died.sum()), " share of deaths excluded at 3/6 mo:",
          round(417 / 1234 * 100, 1), round(712 / 1234 * 100, 1))
    hh = np.histogram(d_.rx_day.dropna(), bins=24)
    print("VERIFY rx histogram (24 bins): first 3", hh[0][:3], "last 3", hh[0][-3:], "width", round(float(hh[1][1] - hh[1][0]), 1))
    print("VERIFY naive: coef", float(ns["cph"].params_["ever"]), "KM 365:",
          {g: round(float(1 - d_[d_.ever == g].died.mean()), 3) for g in (0, 1)})
    for mo in (2, 4, 9):
        print("VERIFY landmark", mo, ns["landmark"](d_, mo))
    print("VERIFY exposed-state table:", ns["t3"].round(3).to_dict())
