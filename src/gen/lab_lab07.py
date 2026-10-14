"""실습 7 · Kaplan-Meier와 로그순위법 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab07.py
      then  python3 tools/nbcheck.py lab07   (runs the notebook alone in a fresh process and compares outputs)
Writes figs/lab07_*.html (insert in content/lab07.html with <!--FIG:lab07_xxx-->),
pub/notebooks/lab07.ipynb (download link at the top of content/lab07.html) and checks/lab07.expected.json.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab07.html that are not
marked in a cell output (대응표, figure readings, Cox HR, plain CI, RMST at other tau, the 과제 answers).
(2026-10-14: version cell 2, 과제 1–3 (cells 24–26) and the notebook added; old cells 2–22 became 3–23.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("lab07")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


def render(c, marks=None, dfmarks=None):
    """labkit.html() applies one marks dict to both text and table output.
    Here text marks go only into the first text <pre>, table marks only into the DataFrame."""
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
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}")
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

# ---------------------------------------------------------------- a. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치 (Colab은 세션마다, 내 컴퓨터는 한 번)", shell_output=PIP_OUT)
save("lab07_install", c, marks={"Successfully installed": 1})
NOTES[c.n] = ("## a. 실습 데이터 준비\n\nColab에서는 런타임이 새로 배정될 때마다 lifelines를 다시 설치합니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
import numpy as np
import pandas as pd
import scipy, matplotlib, lifelines

print("pandas    ", pd.__version__)
print("numpy     ", np.__version__)
print("scipy     ", scipy.__version__)
print("matplotlib", matplotlib.__version__)
print("lifelines ", lifelines.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab07_versions", c, marks={"lifelines  0.30.3": 1})
VER_CELL = c.n

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/survival/cancer.csv")
lung = pd.read_csv(url)
print(lung.shape)
lung.head()
''', title="NCCTG 폐암 자료 불러오기")
save("lab07_load", c, marks={"(228, 11)": 1}, dfmarks={"rownames": 2, "status": 3, "ph.ecog": 4})

c = nb.cell('''
print(lung["status"].value_counts())

# 사건(사망) = 1, 중도절단 = 0 으로 새 열을 만듦
lung["event"] = (lung["status"] == 2).astype(int)
pd.crosstab(lung["status"], lung["event"])
''', title="status 확인과 사건 변수 만들기")
save("lab07_event", c, marks={"2    165": 1, "1     63": 2}, dfmarks={"165": 3})

c = nb.cell('''
from lifelines import KaplanMeierFitter

wrong = KaplanMeierFitter().fit(lung["time"], lung["status"])
right = KaplanMeierFitter().fit(lung["time"], lung["event"])
print("status 그대로:", wrong.event_table["observed"].sum(),
      "건, 중앙값", wrong.median_survival_time_, "일")
print("event 사용  :", right.event_table["observed"].sum(),
      "건, 중앙값", right.median_survival_time_, "일")
''', title="사건 부호화를 잘못하면 (오류 없이 틀린 결과)")
save("lab07_wrong", c, marks={"228 건": 1, "252.0": 2, "165 건": 3, "310.0": 4})

c = nb.cell('''
lung["months"] = lung["time"] / 30.4375   # 365.25일 / 12

n, d = len(lung), lung["event"].sum()
print("환자", n, "명, 사망", d, "명, 중도절단", n - d, "명")
print("사망 비율(추적 기간 무시):", round(d / n * 100, 1), "%")

py = lung["months"].sum() / 12            # 총 추적 인년
print("총 추적:", round(py, 1), "인년")
print("사망률:", round(d / py * 100, 1), "/ 100인년")
''', title="시간 단위 바꾸기와 사건 수 세기")
save("lab07_counts", c, marks={"중도절단 63 명": 1, "72.4 %": 2, "190.5 인년": 3, "86.6 / 100인년": 4})
COUNTS_CELL = c.n

c = nb.cell('''
from lifelines.utils import median_survival_times, qth_survival_times

print("전체 time의 중앙값:", round(lung["months"].median(), 1))
alive = lung["event"] == 0
print("생존자 time의 중앙값:", round(lung.loc[alive, "months"].median(), 1))

# 역 Kaplan-Meier: 중도절단(1 - event)을 '사건'으로 놓는다
fu = KaplanMeierFitter().fit(lung["months"], 1 - lung["event"])
print("역 KM 중앙 추적기간:", round(fu.median_survival_time_, 1))
print(qth_survival_times([0.75, 0.25], fu.survival_function_))
''', title="중앙 추적기간 (역 Kaplan-Meier)")
save("lab07_followup", c, marks={"전체 time의 중앙값: 8.4": 1, "생존자 time의 중앙값: 9.3": 2,
                                 "역 KM 중앙 추적기간: 19.3": 3, "9.889117": 4, "31.704312": 5})

# ---------------------------------------------------------------- b. Kaplan-Meier 생존곡선
c = nb.cell('''
kmf = KaplanMeierFitter()
kmf.fit(lung["months"], event_observed=lung["event"],
        label="All patients")
print(kmf)
kmf.survival_function_.head()
''', title="Kaplan-Meier 추정")
save("lab07_kmfit", c, marks={"228 total observations": 1, "63 right-censored": 2},
     dfmarks={"0.995614": 3})
NOTES[c.n] = "## b. Kaplan-Meier 생존곡선"

c = nb.cell('''
kmf.event_table.head(6)
''', title="사건 표 (event_table)")
save("lab07_eventtable", c, dfmarks={"removed": 1, "entrance": 2, "at_risk": 3, "227": 4})

c = nb.cell('''
s1 = (1 - 1/228)                  # 0.16개월: 228명 중 1명 사망
s2 = s1 * (1 - 3/227)             # 0.36개월: 227명 중 3명 사망
print(round(s1, 6), round(s2, 6))
''', title="처음 두 계단을 손으로 계산")
save("lab07_byhand", c, marks={"0.995614": 1, "0.982456": 2})

c = nb.cell('''
print("중앙생존기간:", round(kmf.median_survival_time_, 2), "개월")
median_survival_times(kmf.confidence_interval_)
''', title="중앙생존기간과 95% 신뢰구간")
save("lab07_median", c, marks={"10.18": 1}, dfmarks={"9.330595": 2, "11.86037": 3})

c = nb.cell('''
ci = kmf.confidence_interval_
for t in [6, 12, 24]:
    s = kmf.survival_function_at_times(t).iloc[0]
    lo, hi = ci.loc[:t].iloc[-1]          # t 이전 마지막 계단
    print(f"{t:>2}개월 생존율 {s:.3f} (95% CI {lo:.3f}-{hi:.3f})")
''', title="시점별 생존율과 95% 신뢰구간")
save("lab07_rates", c, marks={"12개월 생존율 0.409 (95% CI 0.339-0.478)": 1, "24개월 생존율 0.116 (95% CI 0.068-0.178)": 2})

c = nb.cell('''
et = kmf.event_table.loc[:12]             # 12개월까지의 사건 표
d, n = et["observed"], et["at_risk"]
gw = (d / (n * (n - d))).sum()            # Greenwood 합
s = kmf.survival_function_at_times(12).iloc[0]
print("표준오차:", round(s * np.sqrt(gw), 4))

sigma = np.sqrt(gw) / abs(np.log(s))      # log-log 척도의 표준오차
print("log-log CI:", round(s ** np.exp(1.96 * sigma), 3),
      round(s ** np.exp(-1.96 * sigma), 3))
''', title="Greenwood 공식과 log-log 신뢰구간 직접 확인")
save("lab07_greenwood", c, marks={"표준오차: 0.0358": 1, "log-log CI: 0.339 0.478": 2})

c = nb.cell('''
import matplotlib.pyplot as plt
from lifelines.plotting import add_at_risk_counts

fig, ax = plt.subplots(figsize=(7, 4))
kmf.plot_survival_function(ax=ax, show_censors=True)
ax.set_xlabel("Months")
ax.set_ylabel("Survival probability")
ax.set_ylim(0, 1)
ax.set_xticks(range(0, 37, 6))
add_at_risk_counts(kmf, ax=ax)
plt.tight_layout()
''', title="Kaplan-Meier 곡선과 위험집합 표")
save("lab07_kmplot", c)

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7, 4))
km_sex = {}
for code, name in [(1, "Male"), (2, "Female")]:
    g = lung[lung["sex"] == code]
    km_sex[name] = KaplanMeierFitter().fit(
        g["months"], g["event"], label=name)
    km_sex[name].plot_survival_function(ax=ax, show_censors=True)
ax.set_xlabel("Months")
ax.set_ylabel("Survival probability")
ax.set_ylim(0, 1)
ax.set_xticks(range(0, 37, 6))
add_at_risk_counts(*km_sex.values(), ax=ax, rows_to_show=["At risk"])
plt.tight_layout()
''', title="성별 Kaplan-Meier 곡선")
save("lab07_kmsex", c)

c = nb.cell('''
rows = []
for name, k in km_sex.items():
    med = k.median_survival_time_
    lo, hi = median_survival_times(k.confidence_interval_).iloc[0]
    s12 = k.survival_function_at_times(12).iloc[0] * 100
    l12, h12 = k.confidence_interval_.loc[:12].iloc[-1] * 100
    rows.append({"group": name,
                 "n": k.event_table["at_risk"].iloc[0],
                 "deaths": k.event_table["observed"].sum(),
                 "median (95% CI)": f"{med:.1f} ({lo:.1f}-{hi:.1f})",
                 "1-yr % (95% CI)": f"{s12:.1f} ({l12:.1f}-{h12:.1f})"})
pd.DataFrame(rows)
''', title="성별 요약표 (논문 표 형식)")
save("lab07_sextable", c, dfmarks={"8.9 (6.9-10.1)": 1, "14.0 (11.3-17.2)": 2, "33.6 (25.3-42.1)": 3})
SEXTAB_CELL = c.n

c = nb.cell('''
from lifelines.utils import restricted_mean_survival_time as rmst

def rmst_se(k, tau):
    """RMST(tau)와 그 표준오차 (Greenwood형 분산)"""
    et = k.event_table.loc[:tau]
    et = et[et["observed"] > 0]           # 사망이 있는 시점만
    area = np.array([rmst(k, t=tau) - rmst(k, t=t) for t in et.index])
    n, d = et["at_risk"], et["observed"]
    var = (area**2 * d / (n * (n - d))).sum()
    return rmst(k, t=tau), np.sqrt(var)

(r_m, se_m), (r_f, se_f) = [rmst_se(k, 24) for k in km_sex.values()]
diff, se = r_f - r_m, np.sqrt(se_m**2 + se_f**2)
print(f"24개월 RMST: 남 {r_m:.2f}, 여 {r_f:.2f}개월")
print(f"차이 {diff:.2f}개월 (95% CI {diff - 1.96*se:.2f}"
      f" to {diff + 1.96*se:.2f})")
''', title="24개월 제한평균생존시간(RMST)과 차이")
save("lab07_rmst", c, marks={"남 10.22": 1, "여 14.28": 2, "차이 4.06개월": 3, "(95% CI 1.91 to 6.21)": 4})

# ---------------------------------------------------------------- c. 로그순위법
c = nb.cell('''
from lifelines.statistics import logrank_test

m = lung["sex"] == 1                       # 남성이면 True
res = logrank_test(lung.loc[m, "months"], lung.loc[~m, "months"],
                   event_observed_A=lung.loc[m, "event"],
                   event_observed_B=lung.loc[~m, "event"])
res.print_summary()
print(res.test_statistic, res.p_value)
''', title="로그순위 검정 (남 대 여)")
save("lab07_logrank", c, marks={"degrees_of_freedom = 1": 1, "10.33": 2, "<0.005": 3, "9.57": 4,
                                "0.0013111645203554858": 5})
NOTES[c.n] = "## c. 로그순위법"
LOGRANK_CELL = c.n

c = nb.cell('''
rows = []
for t in np.sort(lung.loc[lung["event"] == 1, "months"].unique()):
    r = lung[lung["months"] >= t]             # t 직전 위험집합
    dead = (r["months"] == t) & (r["event"] == 1)
    male = r["sex"] == 1
    rows.append({"n": len(r), "nA": male.sum(),
                 "d": dead.sum(), "dA": (dead & male).sum()})
tab = pd.DataFrame(rows)
E = (tab["d"] * tab["nA"] / tab["n"]).sum()
V = (tab["d"] * (tab["nA"] / tab["n"]) * (1 - tab["nA"] / tab["n"])
     * (tab["n"] - tab["d"]) / (tab["n"] - 1).clip(lower=1)).sum()
O = tab["dA"].sum()
print(f"남성 O = {O}, E = {E:.2f}, O-E = {O - E:.2f}, V = {V:.2f}")
print(f"chi2 = {(O - E)**2 / V:.2f}")
''', title="O, E, V를 직접 계산해 확인")
save("lab07_oev", c, marks={"O = 112": 1, "E = 91.58": 2, "V = 40.37": 3, "chi2 = 10.33": 4})
OEV_CELL = c.n

c = nb.cell('''
from lifelines.statistics import multivariate_logrank_test

e = lung.dropna(subset=["ph.ecog"]).copy()      # 결측 1명 제외
e["ecog"] = e["ph.ecog"].clip(upper=2).astype(int)   # 3 -> 2 ("2+")
print(e["ecog"].value_counts().sort_index())

res3 = multivariate_logrank_test(e["months"], e["ecog"], e["event"])
res3.print_summary()
''', title="세 군 비교 (ECOG 0, 1, 2+)")
save("lab07_ecog", c, marks={"2     51": 1, "degrees_of_freedom = 2": 2, "18.98": 3})
ECOG_CELL = c.n

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7, 4))
km_ecog = []
for g in [0, 1, 2]:
    s = e[e["ecog"] == g]
    lab = f"ECOG {g}" + ("+" if g == 2 else "")
    k = KaplanMeierFitter().fit(s["months"], s["event"], label=lab)
    k.plot_survival_function(ax=ax, ci_show=False, show_censors=True)
    km_ecog.append(k)
    print(lab, "median", round(k.median_survival_time_, 1))
ax.set_xlabel("Months")
ax.set_ylabel("Survival probability")
ax.set_xticks(range(0, 37, 6))
add_at_risk_counts(*km_ecog, ax=ax, rows_to_show=["At risk"])
plt.tight_layout()
''', title="ECOG 군별 곡선")
save("lab07_ecogplot", c, marks={"ECOG 0 median 12.9": 1, "ECOG 2+ median 6.0": 2})

c = nb.cell('''
from lifelines.statistics import pairwise_logrank_test

pw = pairwise_logrank_test(e["months"], e["ecog"], e["event"])
out = pw.summary[["test_statistic", "p"]].copy()
out["p_bonferroni"] = (out["p"] * 3).clip(upper=1)   # 비교 3번
out.round(4)
''', title="두 군씩 비교와 다중비교 보정")
save("lab07_pairwise", c, dfmarks={"0.0630": 1, "0.1890": 2})
PAIR_CELL = c.n

c = nb.cell('''
tests = {"log-rank": {},
         "Gehan-Breslow": {"weightings": "wilcoxon"},
         "Tarone-Ware": {"weightings": "tarone-ware"},
         "Peto-Peto": {"weightings": "peto"},
         "Fleming-Harrington(0,1)":
             {"weightings": "fleming-harrington", "p": 0, "q": 1}}
for name, kw in tests.items():
    r = logrank_test(lung.loc[m, "months"], lung.loc[~m, "months"],
                     lung.loc[m, "event"], lung.loc[~m, "event"],
                     **kw)
    print(f"{name:<24} chi2 = {r.test_statistic:6.2f}"
          f"  P = {r.p_value:.4f}")
''', title="가중 로그순위 검정 비교")
save("lab07_weighted", c, marks={"log-rank                 chi2 =  10.33  P = 0.0013": 1,
                                 "Peto-Peto                chi2 =  12.71  P = 0.0004": 2,
                                 "Fleming-Harrington(0,1)  chi2 =   3.46  P = 0.0629": 3})

# ---------------------------------------------------------------- d. 과제 (정답 셀)
c = nb.cell('''
g = lung.groupby("sex")                    # 1 = 남성, 2 = 여성
hw1 = pd.DataFrame({"n": g.size(),
                    "deaths": g["event"].sum(),
                    "person_years": g["months"].sum() / 12})
hw1["deaths_%"] = hw1["deaths"] / hw1["n"] * 100
hw1["rate_per_100py"] = hw1["deaths"] / hw1["person_years"] * 100
hw1["mean_age"] = g["age"].mean()
hw1.round(1)
''', title="과제 1 정답. 성별 요약표의 한 행 읽기")
save("lab07_hw1", c, dfmarks={"81.2": 1, "104.7": 2, "63.5": 3, "63.3": 4})
HW1_CELL = c.n
NOTES[c.n] = (f"## d. 과제 정답\n\n**과제 1.** 셀 {SEXTAB_CELL}의 성별 요약표에서 남성 행을 논문의 Results처럼 한 문장으로 옮기고, "
              "이 행이 말해 주지 않는 것을 두 가지 이상 쓰세요. 성별 사망 비율, 100인년당 사망률, 평균 나이를 구해 비교하세요.")

c = nb.cell('''
from scipy import stats

def oev(df):
    """남성(sex == 1)의 관찰 사망 O, 기대 사망 E, 분산 V"""
    O = E = V = 0
    for t in np.sort(df.loc[df["event"] == 1, "months"].unique()):
        r = df[df["months"] >= t]                    # t 직전 위험집합
        dead = (r["months"] == t) & (r["event"] == 1)
        male = r["sex"] == 1
        n, nA, d = len(r), male.sum(), dead.sum()
        O += (dead & male).sum()
        E += d * nA / n
        V += d * (nA / n) * (1 - nA / n) * (n - d) / max(n - 1, 1)
    return O, E, V

print(pd.crosstab(e["sex"], e["ecog"], normalize="index").round(3))
strata = pd.DataFrame({g: oev(s) for g, s in e.groupby("ecog")},
                      index=["O", "E", "V"]).T      # ECOG 층마다 한 행
print(strata.assign(O_minus_E=strata["O"] - strata["E"]).round(2))
for name, (O, E, V) in [("stratified  ", strata.sum()),
                        ("unstratified", oev(e))]:
    chi2 = (O - E) ** 2 / V
    print(f"{name} O-E = {O - E:5.2f}, V = {V:5.2f}, "
          f"chi2 = {chi2:5.2f}, P = {stats.chi2.sf(chi2, 1):.4f}")
''', title="과제 2 정답. ECOG로 층화한 로그순위 검정")
save("lab07_hw2", c, marks={"0.263  0.518  0.219": 1, "0.300  0.467  0.233": 2, "O_minus_E": 3,
                            "stratified   O-E = 20.67, V = 38.63, chi2 = 11.05, P = 0.0009": 4,
                            "unstratified O-E = 20.04, V = 40.14, chi2 = 10.01, P = 0.0016": 5})
HW2_CELL = c.n
NOTES[c.n] = (f"**과제 2.** 셀 {LOGRANK_CELL}의 성별 로그순위 검정을 ECOG 수행능력(0, 1, 2+, 셀 {ECOG_CELL}의 e)으로 "
              f"층화해 다시 하세요. 층마다 셀 {OEV_CELL}처럼 O, E, V를 구해 더합니다. 층화하지 않은 결과와 비교해 "
              "χ²가 왜 이만큼 달라지는지(또는 왜 거의 그대로인지) 설명하세요.")

c = nb.cell('''
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test

known = (lung["months"] >= 12) | (lung["event"] == 1)  # 1년 생존 여부를 아는 환자
print("excluded:", (~known).sum(), "(male, female) =",
      lung.loc[~known, "sex"].value_counts().sort_index().tolist())
d12 = lung[known].assign(dead12=lambda x: (x["months"] < 12).astype(int))
tab12 = pd.crosstab(d12["sex"], d12["dead12"])        # 열 1 = 12개월 안 사망
print(tab12)
print("colleague 1-yr survival:", (tab12[0] / tab12.sum(axis=1)).round(3).tolist(),
      " chi-square P =", round(stats.chi2_contingency(tab12)[1], 3))
fp = survival_difference_at_fixed_point_in_time_test(
    12, km_sex["Male"], km_sex["Female"])
print("Kaplan-Meier 1-yr survival:",
      [round(float(k.predict(12)), 3) for k in km_sex.values()],
      " P =", round(fp.p_value, 3))
''', title="과제 3 정답. 12개월 생존 여부를 카이제곱으로 비교한 분석")
save("lab07_hw3", c, marks={"excluded: 42 (male, female) = [18, 24]": 1, "[0.292, 0.455]": 2,
                            "chi-square P = 0.039": 3, "[0.336, 0.526]": 4, "P = 0.013": 5})
HW3_CELL = c.n
NOTES[c.n] = ("**과제 3.** 동료가 1년 생존율을 성별로 비교하려고 12개월 전에 중도절단된 환자를 빼고, 남은 환자를 "
              "'12개월 안에 사망 / 12개월 넘게 생존'으로 나눠 카이제곱 검정을 한 뒤 \"1년 생존율은 남성 29.2%, "
              "여성 45.5%였다(P = 0.039)\"라고 썼습니다. 동료의 분석을 재현하고, 틀린 곳 두 가지를 찾아 왜 틀렸는지 설명하세요.")

print("lab07: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- notebook
nb.save_ipynb(
    "실습 7. Kaplan-Meier와 로그순위법",
    intro=("사회약학 연구방법 노트의 '실습 7. Kaplan-Meier와 로그순위법'에 딸린 노트북입니다. NCCTG 폐암 자료"
           "(R survival 패키지의 cancer, 진행성 폐암 환자 228명)로 사건 변수와 중앙 추적기간을 만들고, Kaplan-Meier "
           "생존곡선과 중앙생존기간, 제한평균생존시간을 구한 뒤 로그순위 검정으로 성별과 ECOG 수행능력에 따른 생존을 "
           "비교합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 7에 있습니다."),
    notes=NOTES)

# ---------------------------------------------------------------- VERIFY: numbers quoted in the text (not marked in an output)
if __name__ == "__main__":
    import warnings
    import numpy as np
    import pandas as pd
    from scipy import stats

    ns = nb.ns
    lung, KM = ns["lung"], ns["KaplanMeierFitter"]
    kmf, km_sex, e = ns["kmf"], ns["km_sex"], ns["e"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r1 = lambda x: f"{x:.1f}"
    r2 = lambda x: f"{x:.2f}"
    r3 = lambda x: f"{x:.3f}"
    # 글에서 번호로 가리키는 셀
    cited = {1: "lifelines 설치 (Colab은 세션마다, 내 컴퓨터는 한 번)", 2: "패키지 불러오기와 버전 확인", 6: "시간 단위 바꾸기와 사건 수 세기", 7: "중앙 추적기간 (역 Kaplan-Meier)",
             11: "중앙생존기간과 95% 신뢰구간", 12: "시점별 생존율과 95% 신뢰구간",
             13: "Greenwood 공식과 log-log 신뢰구간 직접 확인", 15: "성별 Kaplan-Meier 곡선",
             16: "성별 요약표 (논문 표 형식)", 17: "24개월 제한평균생존시간(RMST)과 차이",
             18: "로그순위 검정 (남 대 여)", 19: "O, E, V를 직접 계산해 확인", 20: "세 군 비교 (ECOG 0, 1, 2+)",
             22: "두 군씩 비교와 다중비교 보정"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    # 대응표와 a 절
    same("N, deaths, censored", (len(lung), int(lung["event"].sum()), int((lung["event"] == 0).sum())), (228, 165, 63))
    same("sex n (M, F)", tuple(int(x) for x in lung["sex"].value_counts().sort_index()), (138, 90))
    same("ecog n", tuple(int(x) for x in e["ecog"].value_counts().sort_index()), (63, 113, 51))
    same("ecog 2 and 3", (int((e["ph.ecog"] == 2).sum()), int((e["ph.ecog"] == 3).sum())), (50, 1))
    same("meal.cal missing", int(lung["meal.cal"].isna().sum()), 47)
    same("306 days in months", f"{306 / 30.4375:.2f}", "10.05")
    same("1-exp(-rate) and KM 1-yr death", (f"{1 - np.exp(-0.866):.0%}", f"{1 - kmf.predict(12):.0%}"), ("58%", "59%"))
    same("median days 252 vs 310: diff", 310 - 252, 58)
    same("mean follow-up about 10 months", r1(lung["months"].mean()), "10.0")
    # b 절: 위험집합 표(그림)와 그림 읽기
    grid = list(range(0, 37, 6))
    at_risk = [int((lung["months"] >= t).sum()) for t in grid]
    same("at risk 0..36", at_risk, [228, 156, 65, 32, 13, 3, 0])
    cens = [int(((lung["months"] <= t) & (lung["event"] == 0)).sum()) for t in (6, 12)]
    same("cumulative censored at 6, 12", cens, [6, 42])
    ev12 = int(((lung["months"] <= 12) & (lung["event"] == 1)).sum())
    same("12 months: at risk + censored + events", (65, 42, ev12, 65 + 42 + ev12), (65, 42, 121, 228))
    same("censored between 6 and 12 months", 42 - 6, 36)
    same("65/228", f"{65 / 228:.1%}", "28.5%")
    same("unique death times", int(lung.loc[lung["event"] == 1, "months"].nunique()), 139)
    same("24-month CI asymmetry (down, up)", (r1(11.6 - 6.8), r1(17.8 - 11.6)), ("4.8", "6.2"))
    et = kmf.event_table.loc[:24]
    d_, n_ = et["observed"], et["at_risk"]
    s24 = kmf.predict(24)
    se24 = s24 * np.sqrt((d_ / (n_ * (n_ - d_))).sum())
    same("plain CI 24 months", (r3(s24 - 1.96 * se24), r3(s24 + 1.96 * se24)), ("0.060", "0.171"))
    same("plain CI 12 months", (r3(0.409 - 1.96 * 0.0358), r3(0.409 + 1.96 * 0.0358)), ("0.339", "0.479"))
    same("at risk 24m (M, F)", [int(((lung["sex"] == s_) & (lung["months"] >= 24)).sum()) for s_ in (1, 2)], [7, 6])
    same("sex rows", [len(lung[lung["sex"] == s_]) for s_ in (1, 2)], [138, 90])
    tg = np.arange(0, 31)
    dif = km_sex["Female"].survival_function_at_times(tg).values - km_sex["Male"].survival_function_at_times(tg).values
    assert (dif >= 0).all() and all(0.15 <= x <= 0.27 for x in dif[6:24]) and all(x < 0.05 for x in dif[25:31]), dif
    n_ok += 1
    same("female 1-yr CI", f"{km_sex['Female'].predict(12) * 100:.1f}", "52.6")
    # RMST at other tau
    for tau, want in [(12, ("1.84", "0.88", "2.80")), (30, ("4.25", "1.79", "6.71"))]:
        (a, sa), (b, sb) = [ns["rmst_se"](kk, tau) for kk in km_sex.values()]
        se2 = np.sqrt(sa ** 2 + sb ** 2)
        same(f"RMST tau={tau}", (r2(b - a), r2(b - a - 1.96 * se2), r2(b - a + 1.96 * se2)), want)
    same("at risk 30m (M, F)", [int(((lung["sex"] == s_) & (lung["months"] >= 30)).sum()) for s_ in (1, 2)], [2, 1])
    # c 절
    same("2^-9.57", f"{2 ** -9.57:.4f}", "0.0013")
    tb = ns["tab"]                         # 셀 19의 시점별 표 (과제 2 셀이 O, E, V 이름을 다시 쓰므로 표에서 구함)
    O, E = int(tb["dA"].sum()), (tb["d"] * tb["nA"] / tb["n"]).sum()
    same("O-E, female O and E", (r2(O - E), 165 - int(O), r2(165 - E)), ("20.42", 53, "73.42"))
    same("O/E ratio", (r2(O / E), r2(53 / (165 - E)), r2((O / E) / (53 / (165 - E)))), ("1.22", "0.72", "1.69"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from lifelines import CoxPHFitter
        x = lung.assign(male=(lung["sex"] == 1).astype(int))[["months", "event", "male"]]
        cph = CoxPHFitter().fit(x, "months", "event")
    same("Cox HR male", r2(float(np.exp(cph.params_["male"]))), "1.70")
    same("ECOG 3-group P", (r2(ns["res3"].p_value * 1e5), f"{2 ** -13.69:.5f}"), ("7.56", "0.00008"))
    same("ECOG 1 median", r1(ns["km_ecog"][1].median_survival_time_), "10.1")
    k0, k1 = ns["km_ecog"][0], ns["km_ecog"][1]
    dd = k0.survival_function_at_times(tg).values - k1.survival_function_at_times(tg).values
    assert all(x >= 0.18 for x in dd[8:12]) and all(x <= 0.135 for x in dd[12:30]), dd
    n_ok += 1
    pwo = ns["out"]
    same("pairwise Bonferroni 0-2, 1-2", (f"{pwo['p_bonferroni'].iloc[1]:.4f}", f"{pwo['p_bonferroni'].iloc[2]:.4f}"),
         ("0.0001", "0.0077"))
    # 과제 1
    h1 = ns["hw1"]
    same("hw1 ratios", (r2(h1["deaths_%"][1] / h1["deaths_%"][2]), r2(h1["rate_per_100py"][1] / h1["rate_per_100py"][2])),
         ("1.38", "1.65"))
    same("mean age", (r1(h1["mean_age"][1]), r1(h1["mean_age"][2])), ("63.3", "61.1"))
    same("1-yr survival difference", r1(52.6 - 33.6), "19.0")
    # 과제 2
    ct = pd.crosstab(e["sex"], e["ecog"])
    same("ecog by sex counts", ct.values.tolist(), [[36, 71, 30], [27, 42, 21]])
    st = ns["strata"]
    same("stratum O-E", [r1(x) for x in st["O"] - st["E"]], ["4.6", "12.0", "4.0"])
    # 과제 3
    t12 = ns["tab12"]
    same("colleague table", t12.values.tolist(), [[35, 85], [30, 36]])
    same("excluded share", (f"{18 / 138:.0%}", f"{24 / 90:.0%}"), ("13%", "27%"))
    same("survival drop", (r1(33.6 - 29.2), r1(52.6 - 45.5)), ("4.4", "7.1"))
    same("difference colleague vs KM", (r1(45.5 - 29.2), r1(52.6 - 33.6)), ("16.3", "19.0"))
    same("log-rank P", f"{ns['res'].p_value:.3f}", "0.001")
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab07.html passed")
