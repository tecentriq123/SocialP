"""실습 7 · Kaplan-Meier와 로그순위법 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab07.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("lab07")


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

# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치 (Colab은 세션마다, 내 컴퓨터는 한 번)", shell_output=PIP_OUT)
save("lab07_install", c, marks={"Successfully installed": 1})

c = nb.cell('''
import numpy as np
import pandas as pd

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

# ---------------------------------------------------------------- 나. Kaplan-Meier 생존곡선
c = nb.cell('''
kmf = KaplanMeierFitter()
kmf.fit(lung["months"], event_observed=lung["event"],
        label="All patients")
print(kmf)
kmf.survival_function_.head()
''', title="Kaplan-Meier 추정")
save("lab07_kmfit", c, marks={"228 total observations": 1, "63 right-censored": 2},
     dfmarks={"0.995614": 3})

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

# ---------------------------------------------------------------- 다. 로그순위법
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

c = nb.cell('''
from lifelines.statistics import multivariate_logrank_test

e = lung.dropna(subset=["ph.ecog"]).copy()      # 결측 1명 제외
e["ecog"] = e["ph.ecog"].clip(upper=2).astype(int)   # 3 -> 2 ("2+")
print(e["ecog"].value_counts().sort_index())

res3 = multivariate_logrank_test(e["months"], e["ecog"], e["event"])
res3.print_summary()
''', title="세 군 비교 (ECOG 0, 1, 2+)")
save("lab07_ecog", c, marks={"2     51": 1, "degrees_of_freedom = 2": 2, "18.98": 3})

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

print("lab07: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- verification (numbers quoted in the text; not shown as cells)
if __name__ == "__main__":
    import warnings
    import numpy as np
    ns = nb.ns
    lung, KM = ns["lung"], ns["KaplanMeierFitter"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from lifelines import CoxPHFitter
        from lifelines.statistics import survival_difference_at_fixed_point_in_time_test as fx
        x = lung.assign(male=(lung["sex"] == 1).astype(int))[["months", "event", "male"]]
        c = CoxPHFitter().fit(x, "months", "event")
        print("VERIFY Cox HR male:", round(float(np.exp(c.params_["male"])), 2))
        k = ns["kmf"]; et = k.event_table.loc[:24]; d, n = et["observed"], et["at_risk"]
        s = k.predict(24); se = s * np.sqrt((d / (n * (n - d))).sum())
        print("VERIFY plain CI 24m:", round(s - 1.96 * se, 3), round(s + 1.96 * se, 3))
        for tau in [12, 30]:
            (a, sa), (b, sb) = [ns["rmst_se"](kk, tau) for kk in ns["km_sex"].values()]
            se2 = np.sqrt(sa**2 + sb**2)
            print(f"VERIFY RMST tau={tau}: diff {b - a:.2f} ({b - a - 1.96*se2:.2f} to {b - a + 1.96*se2:.2f})")
        r = fx(12, ns["km_sex"]["Male"], ns["km_sex"]["Female"])
        print("VERIFY fixed-point 12m P:", round(r.p_value, 3))
        print("VERIFY at risk 24m (M, F):", [int(((lung["sex"] == s_) & (lung["months"] >= 24)).sum()) for s_ in (1, 2)])
        print("VERIFY unique death times:", lung.loc[lung["event"] == 1, "months"].nunique())
        print("VERIFY at risk at 12m:", int((lung["months"] >= 12).sum()))
