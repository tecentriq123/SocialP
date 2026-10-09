"""실습 17 · 경쟁위험 분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab17.py
      (NOMARK=1 로 돌리면 번호 표식 없이 셀의 출력만 찍는다)
자료: pub/data/ckd_competing.csv, pub/data/dementia_fracture.csv (gen/data_lab17.py)
끝의 대조 블록이 실습에서 얻은 숫자를 gen/_ch17_nums.json(17장 본문의 숫자)과 맞춘다.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab17")


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
    if os.environ.get("NOMARK"):
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        for w in c.warns:
            print("WARN:", w)
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab17_install", c)

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
ckd = pd.read_csv(BASE + "ckd_competing.csv", storage_options=UA)
print(ckd.shape)
ckd.head()
''', title="자료 불러오기")
save("lab17_load", c, marks={"(3500, 5)": 1}, dfmarks={"0.960913": 2, "4.717419": 3, "5.000000": 4})

c = nb.cell('''
ckd["drug_a"] = (ckd["drug"] == "A").astype(int)   # A = 1, B = 0
ckd["dial"] = (ckd["status"] == 1).astype(int)     # 투석 시작
ckd["death"] = (ckd["status"] == 2).astype(int)    # 투석 전 사망

tab = pd.crosstab(ckd["status"], ckd["drug"], margins=True)
tab.index = ["0 censored", "1 dialysis", "2 death", "All"]
tab
''', title="사건 코딩 확인과 사건 열 만들기")
save("lab17_events", c, dfmarks={"521": 1, "396": 2, "463": 3, "1380": 4})

c = nb.cell('''
cens = ckd["status"] == 0
same = ckd.loc[cens, "years"] == ckd.loc[cens, "max_years"]
early = ckd.loc[~cens, "years"] < ckd.loc[~cens, "max_years"]
print("중도절단 시점 = 자료 마감:", same.all())
print("사건은 자료 마감 전     :", early.all())
ckd.groupby("drug")["max_years"].describe().round(2)
''', title="중도절단이 자료 마감뿐인지 확인")
save("lab17_admin", c, marks={"자료 마감: True": 1, "마감 전     : True": 2}, dfmarks={"4.63": 3})

c = nb.cell('''
rate = ckd.groupby("drug").agg(
    n=("id", "size"), py=("years", "sum"),
    dial=("dial", "sum"), death=("death", "sum"))
rate["mean_fu"] = rate["py"] / rate["n"]
rate["dial_100py"] = rate["dial"] / rate["py"] * 100
rate["death_100py"] = rate["death"] / rate["py"] * 100
print(rate.round(2).T)

ratio = rate.loc["A"] / rate.loc["B"]
print("A / B:",
      ratio[["dial_100py", "death_100py"]].round(2).tolist())
''', title="군별 사건 수, 인년, 발생률")
save("lab17_rates", c, marks={"4049.16": 1, "2.93": 2, "9.78": 3, "11.43": 4, "[1.02, 0.55]": 5})

# ---------------------------------------------------------------- 나. 누적발생함수
c = nb.cell('''
from lifelines import AalenJohansenFitter, KaplanMeierFitter

b = ckd[ckd["drug"] == "B"]
ajf = AalenJohansenFitter()
ajf.fit(b["years"], b["status"], event_of_interest=1)
print(ajf.predict([1, 3, 5]))
print(ajf.confidence_interval_.loc[:5].tail(1))

kmf = KaplanMeierFitter()
kmf.fit(b["years"], b["status"] == 1)      # 사망도 중도절단이 됨
print("1 - KM:", round(1 - kmf.predict(5), 4))
''', title="약물 B군의 누적발생률과 1 − Kaplan-Meier")
save("lab17_aj", c, marks={"0.078774": 1, "0.245497": 2, "AJ_estimate_upper_0.95": 3, "0.226731": 4,
                           "1 - KM: 0.3924": 5})

c = nb.cell('''
def cif(d, event, t):
    """t년 누적발생률, 95% 신뢰구간의 하한과 상한 (0-1)"""
    aj = AalenJohansenFitter()
    aj.fit(d["years"], d["status"], event_of_interest=event)
    lo, hi = sorted(aj.confidence_interval_.loc[:t].iloc[-1])
    return aj.predict(t), lo, hi

rows = {}
for g in ["A", "B"]:
    d = ckd[ckd["drug"] == g]
    for t in [1, 3, 5]:
        est, lo, hi = cif(d, 1, t)
        rows[(f"{t}-year", g)] = (f"{est*100:.1f} "
                                  f"({lo*100:.1f}-{hi*100:.1f})")
pd.Series(rows).unstack()
''', title="두 군의 1·3·5년 투석 누적발생률 (95% 신뢰구간)")
save("lab17_ciftab", c, dfmarks={"8.5 (7.1-10.0)": 1, "30.2 (27.7-32.7)": 2, "24.5 (22.7-26.5)": 3})

c = nb.cell('''
rows = []
for g in ["A", "B"]:
    d = ckd[ckd["drug"] == g]
    km_any = KaplanMeierFitter().fit(d["years"], d["status"] > 0)
    km_dial = KaplanMeierFitter().fit(d["years"], d["status"] == 1)
    rows.append({"drug": g,
                 "dialysis": cif(d, 1, 5)[0],
                 "death": cif(d, 2, 5)[0],
                 "event_free": km_any.predict(5),
                 "km_dialysis": 1 - km_dial.predict(5)})
five = pd.DataFrame(rows).set_index("drug") * 100
five["sum"] = five["dialysis"] + five["death"] + five["event_free"]
five.round(1)
''', title="5년 시점의 세 상태와 합계, 1 − Kaplan-Meier")
save("lab17_sum", c, dfmarks={"30.2": 1, "35.2": 2, "34.6": 3, "39.0": 4, "100.0": 5, "39.2": 6})

c = nb.cell('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, g in zip(axes, ["A", "B"]):
    d = ckd[ckd["drug"] == g]
    aj = AalenJohansenFitter()
    aj.fit(d["years"], d["status"], event_of_interest=1)
    km = KaplanMeierFitter().fit(d["years"], d["status"] == 1)
    one_minus_km = 1 - km.survival_function_["KM_estimate"]
    ax.step(one_minus_km.index, one_minus_km, where="post",
            ls="--", label="1 - KM (death censored)")
    ax.step(aj.cumulative_density_.index,
            aj.cumulative_density_["CIF_1"], where="post",
            label="Cumulative incidence (AJ)")
    ax.set_title(f"Drug {g}")
    ax.set_xlabel("Years since first prescription")
axes[0].set_ylabel("Proportion starting dialysis")
axes[0].legend()
plt.tight_layout()
''', title="누적발생함수와 1 − Kaplan-Meier 겹쳐 그리기")
save("lab17_overlay", c)

c = nb.cell('''
grid = np.linspace(0, 5, 251)                # 0, 0.02, ..., 5년
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, g in zip(axes, ["A", "B"]):
    d = ckd[ckd["drug"] == g]
    f1 = AalenJohansenFitter().fit(
        d["years"], d["status"], event_of_interest=1).predict(grid)
    f2 = AalenJohansenFitter().fit(
        d["years"], d["status"], event_of_interest=2).predict(grid)
    ax.stackplot(grid, f1, f2, 1 - f1 - f2,
                 labels=["Dialysis", "Death before dialysis",
                         "Alive without dialysis"])
    ax.set_title(f"Drug {g}")
    ax.set_xlabel("Years since first prescription")
    ax.set_xlim(0, 5)
axes[0].set_ylabel("Proportion of patients")
axes[0].set_ylim(0, 1)
axes[0].legend(loc="upper right")
plt.tight_layout()
''', title="세 상태를 쌓아 그리기")
save("lab17_stack", c)

c = nb.cell('''
est = {}
for g in ["A", "B"]:
    d = ckd[ckd["drug"] == g]
    aj = AalenJohansenFitter()
    aj.fit(d["years"], d["status"], event_of_interest=1)
    est[g] = (aj.predict(5), aj.variance_.loc[:5].iloc[-1])
(fa, va), (fb, vb) = est["A"], est["B"]

rd = fa - fb                                 # 위험차
se = np.sqrt(va + vb)
print(f"difference {rd*100:.1f}%p (95% CI {(rd - 1.96*se)*100:.1f}"
      f" to {(rd + 1.96*se)*100:.1f})")
print("ratio", round(fa / fb, 2))
''', title="5년 투석 누적발생률의 차이와 95% 신뢰구간")
save("lab17_rd", c, marks={"difference 5.7%p": 1, "2.5 to 8.8": 2, "ratio 1.23": 3})

c = nb.cell('''
from statsmodels.duration.survfunc import CumIncidenceRight

days = np.ceil(b["years"] * 365.25)          # 일 단위로 바꿈
aj_d = AalenJohansenFitter(seed=1)
aj_d.fit(days, b["status"], event_of_interest=1)
print("lifelines (jitter) :", round(aj_d.predict(1827), 4))

cir = CumIncidenceRight(days, b["status"])
print("statsmodels        :", round(cir.cinc[0][-1], 4))
''', title="일 단위 자료에서 나오는 동점 경고")
save("lab17_ties", c, marks={"lifelines (jitter) : 0.2455": 1, "statsmodels        : 0.2455": 2})

# ---------------------------------------------------------------- 다. 원인별 위험비와 Fine–Gray 위험비
c = nb.cell('''
from lifelines import CoxPHFitter

cols = ["exp(coef)", "exp(coef) lower 95%",
        "exp(coef) upper 95%", "p"]
cs_dial = CoxPHFitter().fit(ckd, "years", "dial", formula="drug_a")
cs_death = CoxPHFitter().fit(ckd, "years", "death", formula="drug_a")
print(cs_dial.summary[cols].round(3))
print(cs_death.summary[cols].round(3))
''', title="원인별 Cox 모형 (투석, 투석 전 사망)")
save("lab17_cs", c, marks={"1.018": 1, "0.787": 2, "0.551": 3})

c = nb.cell('''
t0 = 3
at_risk = b["years"] >= t0                 # 3년까지 사건 없이 추적 중
died_before = ((b["status"] == 2) & (b["years"] < t0)
               & (b["max_years"] >= t0))   # 3년 전에 사망
print("cause-specific risk set :", at_risk.sum())
print("died before, kept       :", died_before.sum())
print("subdistribution risk set:", (at_risk | died_before).sum())
''', title="3년 시점 약물 B군의 두 가지 위험집합")
save("lab17_riskset", c, marks={"cause-specific risk set : 851": 1, "kept       : 884": 2, "risk set: 1735": 3})

c = nb.cell('''
died = ckd["status"] == 2
# 투석 전 사망자의 추적 시간을 자료 마감까지로 늘림
ckd["years_sd"] = np.where(died, ckd["max_years"], ckd["years"])

fg = CoxPHFitter().fit(ckd, "years_sd", "dial", formula="drug_a")
print(fg.summary[cols].round(3))
print("p =", round(fg.summary.loc["drug_a", "p"], 4))
''', title="Fine–Gray 모형 (모든 환자의 자료 마감 시점을 알 때)")
save("lab17_fg", c, marks={"1.254": 1, "1.099": 2, "p = 0.0008": 3})

c = nb.cell('''
# 중도절단을 '사건'으로 놓은 Kaplan-Meier = 추적이 이어질 확률 G(t)
kmc = KaplanMeierFitter().fit(ckd["years"], ckd["status"] == 0)
print(kmc.predict([1, 3, 3.5, 4, 4.5, 4.99]).round(3))
''', title="중도절단되지 않고 남을 확률 G(t)")
save("lab17_g", c, marks={"3.00    1.000": 1, "4.00    0.820": 2, "4.99    0.618": 3})

c = nb.cell('''
ev_t = np.sort(ckd.loc[ckd["dial"] == 1, "years"].unique())
dead = ckd.loc[died, ["id", "drug_a", "years"]]

# 사망자마다, 사망 뒤에 투석이 일어난 시점들을 한 줄씩 붙임
ext = dead.merge(pd.DataFrame({"stop": ev_t}), how="cross")
ext = ext[ext["stop"] > ext["years"]].copy()
ext["w"] = (kmc.predict(ext["stop"]).values
            / kmc.predict(ext["years"]).values)
# 가중치가 같은 이웃 줄은 마지막 줄만 남겨 한 줄로 합침
nxt = ext.groupby("id")["w"].shift(-1)     # 같은 환자의 다음 줄 w
ext = ext[ext["w"] != nxt].copy()
prev = ext.groupby("id")["stop"].shift()   # 같은 환자의 앞 줄 stop
ext["start"] = prev.fillna(ext["years"])   # 첫 줄은 사망 시점부터
ext["dial"] = 0

own = ckd.assign(start=0.0, stop=ckd["years"], w=1.0)
keep = ["id", "drug_a", "start", "stop", "dial", "w"]
long = pd.concat([own[keep], ext[keep]], ignore_index=True)
print(len(ckd), "명 ->", len(long), "줄")
long[long["id"] == 1].round(4)
''', title="사망자를 위험집합에 남긴 긴 자료와 가중치")
save("lab17_long", c, marks={"3500 명 -> 224833 줄": 1}, dfmarks={"2.9957": 2, "0.6171": 3})

c = nb.cell('''
from lifelines import CoxTimeVaryingFitter

fgw = CoxTimeVaryingFitter()
fgw.fit(long, id_col="id", event_col="dial", start_col="start",
        stop_col="stop", weights_col="w", formula="drug_a")
print(fgw.summary[cols].round(3))
''', title="Fine–Gray 모형 (가중치를 쓰는 일반적인 방법)")
save("lab17_fgw", c, marks={"1.257": 1})

c = nb.cell('''
from lifelines.statistics import logrank_test

a = ckd["drug_a"] == 1
gray = logrank_test(ckd.loc[a, "years_sd"], ckd.loc[~a, "years_sd"],
                    ckd.loc[a, "dial"], ckd.loc[~a, "dial"])
lr = logrank_test(ckd.loc[a, "years"], ckd.loc[~a, "years"],
                  ckd.loc[a, "dial"], ckd.loc[~a, "dial"])
print(f"deaths kept in risk set: chi2 {gray.test_statistic:.2f},"
      f" P {gray.p_value:.4f}")
print(f"deaths censored        : chi2 {lr.test_statistic:.2f},"
      f" P {lr.p_value:.4f}")
''', title="누적발생함수의 비교 (Gray 검정의 원리)와 보통의 로그순위 검정")
save("lab17_gray", c, marks={"chi2 11.35, P 0.0008": 1, "chi2 0.07, P 0.7873": 2})

c = nb.cell('''
def hr_ci(m, var="drug_a"):
    s = m.summary.loc[var]
    return (f"{s['exp(coef)']:.2f} ({s['exp(coef) lower 95%']:.2f}"
            f"-{s['exp(coef) upper 95%']:.2f})")

pd.DataFrame({"HR (95% CI), A vs B": {
    "dialysis: cause-specific": hr_ci(cs_dial),
    "dialysis: subdistribution": hr_ci(fg),
    "dialysis: subdistribution (weighted)": hr_ci(fgw),
    "death: cause-specific": hr_ci(cs_death)}})
''', title="위험비를 한 표로 정리")
save("lab17_hrtab", c, dfmarks={"1.02 (0.89-1.16)": 1, "1.25 (1.10-1.43)": 2, "1.26 (1.10-1.43)": 3,
                                "0.55 (0.49-0.61)": 4})

# ---------------------------------------------------------------- 라. 과제 (정답 셀)
c = nb.cell('''
for g in ["A", "B"]:
    d = ckd[ckd["drug"] == g]
    est, lo, hi = cif(d, 2, 5)             # 관심 사건 = 사망(2)
    km = KaplanMeierFitter().fit(d["years"], d["status"] == 2)
    print(f"{g}: CIF {est*100:.1f} ({lo*100:.1f}-{hi*100:.1f}),"
          f" 1 - KM {(1 - km.predict(5))*100:.1f}")

# 이번에는 투석을 시작한 사람을 자료 마감까지 위험집합에 남김
dialyzed = ckd["status"] == 1
ckd["years_sd2"] = np.where(dialyzed, ckd["max_years"], ckd["years"])
fg_death = CoxPHFitter().fit(ckd, "years_sd2", "death",
                             formula="drug_a")
print("subdistribution HR:", hr_ci(fg_death))
print("cause-specific HR :", hr_ci(cs_death))
''', title="과제 1 정답. 투석 전 사망을 관심 사건으로")
save("lab17_hw1", c, marks={"A: CIF 35.2 (32.6-37.8)": 1, "1 - KM 43.2": 2, "subdistribution HR: 0.56 (0.50-0.62)": 3})

c = nb.cell('''
ckd["any"] = (ckd["status"] > 0).astype(int)   # 투석 또는 사망
comp = CoxPHFitter().fit(ckd, "years", "any", formula="drug_a")
print("composite HR:", hr_ci(comp))

for g in ["A", "B"]:
    d = ckd[ckd["drug"] == g]
    km = KaplanMeierFitter().fit(d["years"], d["any"])
    both = cif(d, 1, 5)[0] + cif(d, 2, 5)[0]
    print(f"{g}: 1 - KM {(1 - km.predict(5))*100:.1f},"
          f" CIF dialysis + CIF death {both*100:.1f}")
''', title="과제 2 정답. 복합 결과(투석 또는 사망)")
save("lab17_hw2", c, marks={"composite HR: 0.70 (0.64-0.76)": 1, "A: 1 - KM 65.4": 2})

c = nb.cell('''
fx = pd.read_csv(BASE + "dementia_fracture.csv", storage_options=UA)
fx["drug_c"] = (fx["drug"] == "C").astype(int)     # C = 1, D = 0
fx["frac"] = (fx["status"] == 1).astype(int)
fx["death"] = (fx["status"] == 2).astype(int)
fx["years_sd"] = np.where(fx["status"] == 2,
                          fx["max_years"], fx["years"])

for g in ["C", "D"]:
    d = fx[fx["drug"] == g]
    est, lo, hi = cif(d, 1, 5)
    km = KaplanMeierFitter().fit(d["years"], d["status"] == 1)
    print(f"{g}: fracture {est*100:.1f} ({lo*100:.1f}-{hi*100:.1f}),"
          f" death {cif(d, 2, 5)[0]*100:.1f},"
          f" 1 - KM {(1 - km.predict(5))*100:.1f}")

m1 = CoxPHFitter().fit(fx, "years", "frac", formula="drug_c")
m2 = CoxPHFitter().fit(fx, "years", "death", formula="drug_c")
m3 = CoxPHFitter().fit(fx, "years_sd", "frac", formula="drug_c")
print("fracture, cause-specific HR :", hr_ci(m1, "drug_c"))
print("death, cause-specific HR    :", hr_ci(m2, "drug_c"))
print("fracture, subdistribution HR:", hr_ci(m3, "drug_c"))
''', title="과제 3 정답. 고관절 골절 자료 (17장 나 절 확인 문제)")
save("lab17_hw3", c, marks={"C: fracture 8.3 (7.2-9.5)": 1, "1 - KM 14.1": 2,
                            "cause-specific HR : 0.97 (0.80-1.17)": 3,
                            "subdistribution HR: 0.81 (0.67-0.98)": 4})

print("lab17: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- 노트북
HW = {
    21: ("## 과제 정답\n\n**과제 1. 투석 전 사망을 관심 사건으로.** 두 군의 5년 사망 누적발생률(95% 신뢰구간)과, "
         "투석을 중도절단으로 둔 1 − Kaplan-Meier를 구합니다. 이어서 사망의 하위분포 위험비를 구해 "
         "셀 13의 원인별 위험비와 비교합니다."),
    22: ("**과제 2. 복합 결과.** '투석 또는 사망 중 먼저 일어난 것'을 사건으로 놓고 Cox 모형의 위험비와 "
         "두 군의 5년 1 − Kaplan-Meier를 구합니다. 이 값이 투석과 사망의 누적발생률의 합과 같은지 확인합니다."),
    23: ("**과제 3. 고관절 골절 자료.** dementia_fracture.csv(치매 고령 환자, 약물 C 대 D, 1 = 고관절 골절, "
         "2 = 골절 전 사망)로 5년 누적발생률, 1 − Kaplan-Meier, 골절과 사망의 원인별 위험비, 골절의 하위분포 "
         "위험비를 구합니다."),
}
NOTES = {1: "## 가. 실습 데이터 준비", 6: "## 나. 누적발생함수", 13: "## 다. 원인별 위험비와 Fine–Gray 위험비"}
NOTES.update(HW)
path = nb.save_ipynb(
    "실습 17. 경쟁위험 분석",
    intro=("사회약학 연구방법 노트의 '실습 17 경쟁위험 분석'에 나오는 셀을 순서대로 담은 노트북입니다. "
           "17장의 예제 자료(가상의 만성콩팥병 코호트, 약물 A 대 B)로 누적발생함수, 원인별 위험비, "
           "Fine–Gray 하위분포 위험비를 구합니다. 셀을 위에서부터 차례로 실행하세요. "
           "셀 18은 줄 수가 많아 10초 넘게 걸릴 수 있습니다. 설명은 사이트의 실습 17 쪽에 있습니다."),
    notes=NOTES)
print("notebook:", path)

# ---------------------------------------------------------------- 대조 블록: 실습 결과와 17장 본문의 숫자(gen/_ch17_nums.json)
if __name__ == "__main__":
    ns = nb.ns
    J = json.load(open(os.path.join(HERE, "_ch17_nums.json")))
    checks = []

    def chk(name, a, b, tol):
        ok = abs(a - b) < tol
        checks.append(ok)
        print(f"  {'ok ' if ok else 'BAD'} {name}: lab {a:.6g}  chapter {b:.6g}")
        assert ok, name

    ckd, fx, cif = ns["ckd"], ns["fx"], ns["cif"]
    KM = ns["KaplanMeierFitter"]
    print("대조: 실습 결과와 17장 본문 숫자")
    rate = ns["rate"]
    for g in ("A", "B"):
        d = ckd[ckd["drug"] == g]
        chk(f"{g} n", len(d), J[g]["n"], 0.5)
        chk(f"{g} dialysis n", int(d["dial"].sum()), J[g]["dial"], 0.5)
        chk(f"{g} death n", int(d["death"].sum()), J[g]["death"], 0.5)
        chk(f"{g} censored n", int((d["status"] == 0).sum()), J[g]["cens"], 0.5)
        chk(f"{g} person-years", float(rate.loc[g, "py"]), J[g]["py"], 1e-6)
        for y in (1, 3, 5):
            q = J[g][f"y{y}"]
            e, lo, hi = cif(d, 1, y)
            chk(f"{g} CIF dialysis {y}y", e, q["cif_d"], 1e-9)
            chk(f"{g} CIF dialysis {y}y lower", lo, q["cif_d_ci"][0], 1e-9)
            chk(f"{g} CIF dialysis {y}y upper", hi, q["cif_d_ci"][1], 1e-9)
        q = J[g]["y5"]
        e, lo, hi = cif(d, 2, 5)
        chk(f"{g} CIF death 5y", e, q["cif_m"], 1e-9)
        chk(f"{g} CIF death 5y lower", lo, q["cif_m_ci"][0], 1e-9)
        chk(f"{g} CIF death 5y upper", hi, q["cif_m_ci"][1], 1e-9)
        chk(f"{g} event-free 5y (table)", ns["five"].loc[g, "event_free"] / 100, q["free"], 1e-9)
        chk(f"{g} 1-KM dialysis 5y (table)", ns["five"].loc[g, "km_dialysis"] / 100, q["km_d"], 1e-9)
        km = KM().fit(d["years"], d["status"] == 2)
        chk(f"{g} 1-KM death 5y", 1 - km.predict(5), q["km_m"], 1e-9)
    chk("B 3y cause-specific risk set", int(ns["at_risk"].sum()), J["B"]["y3"]["rs_cs"], 0.5)
    chk("B 3y subdistribution risk set", int((ns["at_risk"] | ns["died_before"]).sum()), J["B"]["y3"]["rs_sd"], 0.5)
    chk("5y risk difference", ns["rd"], J["rd5"][0], 1e-9)
    chk("5y risk difference lower (1.96 vs exact z)", ns["rd"] - 1.96 * ns["se"], J["rd5"][1], 1e-5)
    chk("5y risk difference upper (1.96 vs exact z)", ns["rd"] + 1.96 * ns["se"], J["rd5"][2], 1e-5)
    chk("5y risk ratio", ns["fa"] / ns["fb"], J["rr5"][0], 1e-9)

    def hr3(m):
        s = m.summary.iloc[0]
        return [float(s["exp(coef)"]), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"])]

    for name, m, ref in (("csHR dialysis", ns["cs_dial"], J["hr"]["cs_d"]), ("csHR death", ns["cs_death"], J["hr"]["cs_m"]),
                         ("sHR dialysis (known end)", ns["fg"], J["hr"]["fg"]), ("sHR death", ns["fg_death"], J["fg_death"]),
                         ("composite HR", ns["comp"], J["hr"]["comp"]),
                         ("fx csHR fracture", ns["m1"], J["fx_hr"]["cs_d"]), ("fx csHR death", ns["m2"], J["fx_hr"]["cs_m"]),
                         ("fx sHR fracture", ns["m3"], J["fx_hr"]["fg"])):
        got = hr3(m)
        for lab, a_, b_ in zip(("", " lower", " upper"), got[:3], ref[:3]):
            chk(name + lab, a_, b_, 1e-6)
        chk(name + " p", got[3], ref[3], 1e-6)
    got = hr3(ns["fgw"])   # chapter stores this one to 3 decimals: 1.257 (1.102-1.435)
    chk("sHR dialysis (weighted)", got[0], J["fg_ipcw"][0], 6e-4)
    chk("sHR dialysis (weighted) lower", got[1], J["fg_ipcw"][1][0], 6e-4)
    chk("sHR dialysis (weighted) upper", got[2], J["fg_ipcw"][1][1], 6e-4)
    chk("Gray-type chi2", ns["gray"].test_statistic, J["gray"][0], 1e-6)
    chk("Gray-type p", ns["gray"].p_value, J["gray"][1], 1e-9)
    chk("log-rank chi2", ns["lr"].test_statistic, J["logrank_cs"][0], 1e-6)
    chk("log-rank p", ns["lr"].p_value, J["logrank_cs"][1], 1e-9)
    for g in ("C", "D"):
        d = fx[fx["drug"] == g]
        q = J["fx"][g]
        e, lo, hi = cif(d, 1, 5)
        chk(f"fx {g} n fracture", int(d["frac"].sum()), q["fx"], 0.5)
        chk(f"fx {g} n death", int(d["death"].sum()), q["death"], 0.5)
        chk(f"fx {g} CIF fracture 5y", e, q["cif5"], 1e-9)
        chk(f"fx {g} CIF lower", lo, q["ci5"][0], 1e-9)
        chk(f"fx {g} CIF upper", hi, q["ci5"][1], 1e-9)
        chk(f"fx {g} CIF death 5y", cif(d, 2, 5)[0], q["death5"], 1e-9)
        km = KM().fit(d["years"], d["status"] == 1)
        chk(f"fx {g} 1-KM fracture 5y", 1 - km.predict(5), q["km5"], 1e-9)
    print(f"대조 {len(checks)}개 모두 일치")

    # ---- 본문 글에 인용한 그 밖의 숫자 (셀로 보여 주지 않는 것)
    fa, fb = ns["fa"], ns["fb"]
    shr = float(np.exp(ns["fg"].params_["drug_a"]))
    print("VERIFY implied CIF A = 1-(1-fb)^sHR:", round(1 - (1 - fb) ** shr, 4), " observed", round(fa, 4),
          " (chapter", round(J["fg_pred5"], 4), ")")
    five = ns["five"]
    print("VERIFY 1-KM / CIF:", (five["km_dialysis"] / five["dialysis"]).round(2).to_dict())
    print("VERIFY crude proportions:", (ckd.groupby("drug")["dial"].mean() * 100).round(1).to_dict())
    long = ns["long"]
    print("VERIFY long rows per death (mean, max):", round((len(long) - len(ckd)) / int(ckd["death"].sum()), 1),
          int(long.groupby("id").size().max()) - 1, " dialysis times:", len(ns["ev_t"]))
    x = long[long["id"] == 1]
    print("VERIFY id 1:", ckd.loc[ckd["id"] == 1].to_dict("records"), "rows", len(x), "last w", float(x["w"].iloc[-1]),
          "last stop", float(x["stop"].iloc[-1]))
    kmc = ns["kmc"]
    print("VERIFY first censoring time:", float(ckd.loc[ckd["status"] == 0, "years"].min()),
          " G(4.99):", float(kmc.predict(4.99)), " n censored before 5:", int(((ckd["status"] == 0) & (ckd["years"] < 5)).sum()),
          " at 5:", int(((ckd["status"] == 0) & (ckd["years"] == 5)).sum()))
    print("VERIFY weighted fit coef/se:", ns["fgw"].summary.loc["drug_a", ["coef", "se(coef)", "p"]].to_dict())
    print("VERIFY day ties: n distinct days", int(ns["days"].nunique()), " lifelines", float(ns["aj_d"].predict(1827)),
          " statsmodels", float(ns["cir"].cinc[0][-1]), " continuous", float(ns["ajf"].predict(5)))
    for sd in (2, 3, 4):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_ = ns["AalenJohansenFitter"](seed=sd).fit(ns["days"], ns["b"]["status"], event_of_interest=1)
        print("VERIFY jitter seed", sd, round(float(a_.predict(1827)), 5))
    print("VERIFY death HR p:", ns["cs_death"].summary.loc["drug_a", "p"])
    bb = ckd[ckd["drug"] == "B"]
    k_ = KM().fit(bb["years_sd"], bb["dial"])
    print("VERIFY 1-KM on years_sd (B) at 3, 5:", round(1 - k_.predict(3), 4), round(1 - k_.predict(5), 4),
          " AJ:", round(cif(bb, 1, 3)[0], 4), round(cif(bb, 1, 5)[0], 4))
    print("VERIFY all-patient CIF 5y:", round(cif(ckd, 1, 5)[0], 4), " chapter", round(J["all"]["y5"]["cif_d"], 4))
    print("VERIFY sums with 1-KM:", (five["km_dialysis"] + five["death"] + five["event_free"]).round(1).to_dict())
    print("VERIFY total PY:", round(float(ckd["years"].sum()), 1), " dialysis %:", round(892 / 3500 * 100, 1),
          " death %:", round(1551 / 3500 * 100, 1))
    print("VERIFY potential fu: share with max_years == 5:", round(float((ckd["max_years"] == 5).mean()), 3))
