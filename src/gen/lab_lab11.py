"""실습 11 · Cox 비례위험모형 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab11.py
      then  python3 tools/nbcheck.py lab11   (runs the notebook alone in a fresh process and compares outputs)
      (NOMARK=1 prints every cell's text output without marks, for writing marks)
Writes figs/lab11_*.html (insert in content/lab11.html with <!--FIG:lab11_xxx-->),
pub/notebooks/lab11.ipynb (download link at the top of content/lab11.html) and checks/lab11.expected.json.
Data: survival pbc (Rdatasets URL -> offline copy via labkit).
The VERIFY block at the end asserts the numbers quoted in the text of content/lab11.html that are not
marked in a cell output (대응표, figure readings, 직접 바꿔 보기, error callouts, 과제 problems and answers).
(2026-10-14: version cell 2, 과제 1–3 and the notebook added; old cells 2–25 are now 3–26.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

np.random.seed(0)
nb = Notebook("lab11")


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
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)

# ---------------------------------------------------------------- a. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab11_install", c)
NOTES[c.n] = ("## a. 실습 데이터 준비\n\nColab에서는 런타임이 새로 배정될 때마다 lifelines를 설치합니다. 내 컴퓨터"
              "(Anaconda)에서는 한 번만 실행합니다.")

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib, statsmodels, lifelines

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("matplotlib  ", matplotlib.__version__)
print("statsmodels ", statsmodels.__version__)
print("lifelines   ", lifelines.__version__)
''', title="패키지 불러오기와 버전 확인")
_ns = nb.ns
save("lab11_versions", c, marks={f"pandas       {_ns['pd'].__version__}": 1,
                                 f"lifelines    {_ns['lifelines'].__version__}": 2})
NOTES[c.n] = "출력을 만든 패키지의 버전을 확인합니다. 내 화면의 출력이 사이트와 다를 때 먼저 비교합니다."

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/survival/pbc.csv")
pbc = pd.read_csv(url)
print(pbc.shape)
pbc.head(3)
''', title="PBC 자료 불러오기")
save("lab11_load", c, marks={"(418, 21)": 1}, dfmarks={"status": 2, "trt": 3, "edema": 4})

c = nb.cell('''
print(pbc["trt"].value_counts(dropna=False))

d = pbc[pbc["trt"].notna()].copy()        # 무작위배정된 312명만
pd.crosstab(d["status"], d["trt"], margins=True)
''', title="무작위배정 환자만 남기기")
save("lab11_trt", c, marks={"1.0    158": 1, "NaN    106": 2}, dfmarks={"125": 3, "312": 4})

c = nb.cell('''
d["death"] = (d["status"] == 2).astype(int)   # 사망 = 1, 이식·생존 = 0
d["years"] = d["time"] / 365.25
d["dpca"] = (d["trt"] == 1).astype(int)       # D-penicillamine = 1
d["female"] = (d["sex"] == "f").astype(int)
d["age10"] = d["age"] / 10                    # 10세 단위

print("사망", d["death"].sum(), "명, 이식(중도절단)",
      (d["status"] == 1).sum(), "명")
d.groupby("dpca")["death"].agg(["count", "sum"])
''', title="사건, 시간, 치료 변수 만들기")
save("lab11_vars", c, marks={"사망 125 명": 1, "이식(중도절단) 19 명": 2})

c = nb.cell('''
na = d.isna().sum()
print(na[na > 0])
print("dropna() 전체 적용:", len(d.dropna()), "명")

cols = ["years", "death", "dpca", "age10", "female",
        "bili", "albumin", "edema", "protime"]
df = d[cols].dropna().reset_index(drop=True)
print("분석에 쓸 열만 dropna:", len(df), "명")
''', title="결측값 확인과 분석용 자료 만들기")
save("lab11_missing", c, marks={"chol        28": 1, "dropna() 전체 적용: 276": 2, "분석에 쓸 열만 dropna: 312": 3})

c = nb.cell('''
df["log2_bili"] = np.log2(df["bili"])      # 빌리루빈이 2배 -> 1 증가

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].hist(df["bili"], bins=30)
axes[0].set_xlabel("Bilirubin (mg/dL)")
axes[1].hist(df["log2_bili"], bins=30)
axes[1].set_xlabel("log2(bilirubin)")
axes[0].set_ylabel("Number of patients")
plt.tight_layout()
df[["bili", "log2_bili"]].describe().round(2)
''', title="빌리루빈 로그 변환")
save("lab11_logbili", c, dfmarks={"1.35": 1, "0.43": 2, "28.00": 3})

# ---------------------------------------------------------------- b. Cox 모형과 위험비
c = nb.cell('''
from lifelines import CoxPHFitter

cph1 = CoxPHFitter()
cph1.fit(df, duration_col="years", event_col="death", formula="dpca")
cph1.print_summary(decimals=3)
''', title="치료만 넣은 Cox 모형 (단변수)")
NOTES[c.n] = "## b. Cox 모형과 위험비"
UNI_CELL = c.n
save("lab11_uni", c, marks={"187 right-censored observations": 1, "number of events observed = 125": 2,
                            "baseline estimation = breslow": 3, "partial log-likelihood = -639.915": 4,
                            "0.057": 5, "1.059": 6, "0.179": 7, "1.504": 8, "0.319": 9, "0.749": 10,
                            "Concordance = 0.499": 11, "Partial AIC = 1281.831": 12,
                            "log-likelihood ratio test = 0.102 on 1 df": 13})

c = nb.cell('''
from lifelines.statistics import logrank_test

a = df["dpca"] == 1
lr = logrank_test(df.loc[a, "years"], df.loc[~a, "years"],
                  df.loc[a, "death"], df.loc[~a, "death"])
z = cph1.summary.loc["dpca", "z"]
print("log-rank chi2 :", round(lr.test_statistic, 4))
print("Wald z^2      :", round(z**2, 4))
print("LR chi2       :",
      round(cph1.log_likelihood_ratio_test().test_statistic, 4))
''', title="단변수 Cox 모형과 로그순위 검정 비교")
save("lab11_vslogrank", c, marks={"log-rank chi2 : 0.1017": 1, "Wald z^2      : 0.102": 2,
                                  "LR chi2       : 0.1021": 3})

FORMULA = "dpca + age10 + female + log2_bili + albumin + C(edema) + protime"
c = nb.cell('''
cph = CoxPHFitter()
cph.fit(df, duration_col="years", event_col="death",
        formula="dpca + age10 + female + log2_bili + albumin"
                " + C(edema) + protime")
cph.print_summary(decimals=3)
''', title="다변수 Cox 모형 (formula 사용)")
MULTI_CELL = c.n
save("lab11_multi", c, marks={"0.938": 1, "1.369": 2, "1.842": 3, "0.372": 4,
                              "C(edema)[T.0.5]": 5, "2.582": 6, "<0.0005": 7,
                              "Concordance = 0.846": 8, "Partial AIC = 1094.350": 9,
                              "log-likelihood ratio test = 201.583 on 8 df": 10})

c = nb.cell('''
CoxPHFitter().fit(d, "years", "death", formula="dpca + chol")
''', title="결측값이 있는 변수를 formula에 넣으면", expect_error=True)
save("lab11_naerror", c)
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

c = nb.cell('''
b = cph.params_                         # 계수(로그 위험비)
print("나이 1세당 HR   :", round(np.exp(b["age10"] / 10), 3))
print("나이 10세당 HR  :", round(np.exp(b["age10"]), 3))
print("나이 20세당 HR  :", round(np.exp(b["age10"] * 2), 3))
print("빌리루빈 2배 HR :", round(np.exp(b["log2_bili"]), 3))
print("빌리루빈 4배 HR :", round(np.exp(b["log2_bili"] * 2), 3))
print("알부민 0.5 g/dL 감소 HR:", round(np.exp(-0.5 * b["albumin"]), 3))
''', title="위험비의 단위 바꾸기")
NOTES[c.n] = (f"결측값이 있는 변수를 formula에 넣어 일부러 오류를 내 보는 셀(셀 {ERR_CELL})은 이 노트북에서 뺐습니다"
              "(사이트의 실습 11 b 절에 있습니다).")
save("lab11_scale", c, marks={"나이 1세당 HR   : 1.032": 1, "나이 20세당 HR  : 1.874": 2,
                              "빌리루빈 4배 HR : 3.393": 3, "알부민 0.5 g/dL 감소 HR: 1.639": 4})

c = nb.cell('''
pts = pd.DataFrame([df.median()] * 2)       # 중앙값 환자 두 명
pts["dpca"] = 0
pts["edema"] = [0.0, 1.0]                   # 부종 없음 / 이뇨제에도 부종
S = cph.predict_survival_function(pts, times=[1, 5, 10])
risk = 1 - S                                # 누적 사망위험
risk.columns = ["no edema", "edema 1.0"]
risk["risk ratio"] = risk["edema 1.0"] / risk["no edema"]
print("HR (edema 1.0):", round(np.exp(b["C(edema)[T.1.0]"]), 2))
risk.round(3)
''', title="위험비와 누적위험의 비(상대위험도)는 다르다")
HRRR_CELL = c.n
save("lab11_hrrr", c, marks={"HR (edema 1.0): 2.58": 1}, dfmarks={"0.150": 2, "0.342": 3, "2.285": 4, "1.731": 5})

c = nb.cell('''
tab = cph.summary[["exp(coef)", "exp(coef) lower 95%",
                   "exp(coef) upper 95%", "p"]].copy()
tab.columns = ["HR", "lower", "upper", "p"]
tab["HR (95% CI)"] = [f"{h:.2f} ({l:.2f}-{u:.2f})"
                      for h, l, u in zip(tab.HR, tab.lower, tab.upper)]
tab["P"] = [f"{p:.3f}" if p >= 0.001 else "<0.001" for p in tab.p]
tab[["HR (95% CI)", "P"]]
''', title="논문용 위험비 표 만들기")
save("lab11_table", c, dfmarks={"0.94 (0.64-1.37)": 1, "<0.001": 2})

c = nb.cell('''
labels = {"dpca": "D-penicillamine vs placebo",
          "age10": "Age (per 10 years)", "female": "Female vs male",
          "log2_bili": "Bilirubin (per doubling)",
          "albumin": "Albumin (per 1 g/dL)",
          "C(edema)[T.0.5]": "Edema 0.5 vs none",
          "C(edema)[T.1.0]": "Edema 1.0 vs none",
          "protime": "Prothrombin time (per 1 s)"}
fig, ax = plt.subplots(figsize=(7, 4))
cph.plot(hazard_ratios=True, ax=ax)
ax.set_xscale("log")
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.minorticks_off()
names = [labels[t.get_text()] for t in ax.get_yticklabels()]
ax.set_yticklabels(names)
ax.set_xlabel("Hazard ratio (95% CI, log scale)")
plt.tight_layout()
''', title="포레스트 그림")
save("lab11_forest", c)

c = nb.cell('''
for name, f in [("bili", "bili"), ("log2_bili", "log2_bili")]:
    m = CoxPHFitter().fit(df, "years", "death",
        formula=f"dpca + age10 + female + {f} + albumin"
                " + C(edema) + protime")
    print(f"{name:<10} partial AIC = {m.AIC_partial_:.1f}"
          f"  C = {m.concordance_index_:.3f}")
''', title="빌리루빈을 로그로 넣은 이유 확인")
save("lab11_aic", c, marks={"bili       partial AIC = 1124.7": 1, "log2_bili  partial AIC = 1094.3": 2})

# ---------------------------------------------------------------- c. 비례위험 가정 점검
c = nb.cell('''
from lifelines.statistics import proportional_hazard_test

ph = proportional_hazard_test(cph, df, time_transform="rank")
ph.print_summary(decimals=3)
''', title="비례위험 검정 (척도화 Schoenfeld 잔차)")
NOTES[c.n] = "## c. 비례위험 가정 점검"
PH_CELL = c.n
save("lab11_phtest", c, marks={"time_transform = rank": 1, "degrees_of_freedom = 1": 2,
                               "C(edema)[T.0.5]           4.500 0.034": 3,
                               "dpca                      2.136 0.144": 4,
                               "protime                   4.641 0.031": 5})

c = nb.cell('''
cph.check_assumptions(df, p_value_threshold=0.05)
''', title="check_assumptions의 요약과 조언")
save("lab11_check", c, marks={
    "The ``p_value_threshold`` is set at 0.05.": 1,
    "rank            4.50 0.03": 2,
    "p-value is 0.0339.": 3,
    "`strata=['C(edema)[T.0.5]', ...]`": 4,
    "p-value is 0.0312.": 5,
    "Advice 3: try adding an interaction term with your time variable.": 6})

c = nb.cell('''
from statsmodels.nonparametric.smoothers_lowess import lowess

r = cph.compute_residuals(df, kind="scaled_schoenfeld")
t = df.loc[r.index, "years"]                 # 사망 시점
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, v in zip(axes, ["dpca", "protime"]):
    beta_t = r[v] + cph.params_[v]           # 세로축 = beta(t)
    ax.scatter(t, beta_t, s=8, alpha=0.5)
    sm = lowess(beta_t, t, frac=0.5)         # 평활선
    ax.plot(sm[:, 0], sm[:, 1], color="black")
    ax.axhline(cph.params_[v], ls="--", color="gray")
    ax.set_xlabel("Years since randomization")
    ax.set_title(f"beta(t) for {v}")
axes[0].set_ylabel("Scaled Schoenfeld residual + coef")
plt.tight_layout()
''', title="척도화 Schoenfeld 잔차 그림")
save("lab11_schoen", c)

c = nb.cell('''
from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(6, 4))
for v, lab in [(0, "No edema"), (0.5, "Edema 0.5"), (1, "Edema 1.0")]:
    s = df[df["edema"] == v]
    k = KaplanMeierFitter().fit(s["years"], s["death"], label=lab)
    k.plot_loglogs(ax=ax)
ax.set_xlabel("log(years)")
ax.set_ylabel("log(-log S(t))")
plt.tight_layout()
''', title="log(-log) 생존 그림 (부종 범주별)")
save("lab11_loglog", c)

c = nb.cell('''
cut = 3                                     # 3년에서 나눔
x = df.assign(id=df.index)                  # 환자 번호 열
early = x.assign(start=0.0, stop=x["years"].clip(upper=cut), late=0)
early["event"] = np.where(x["years"] <= cut, x["death"], 0)
late = x[x["years"] > cut].assign(start=float(cut), late=1)
late["stop"] = late["years"]
late["event"] = late["death"]
long = pd.concat([early, late]).drop(columns=["years", "death"])
long["protime_late"] = long["protime"] * long["late"]
print(len(df), "명 ->", len(long), "줄")
show = ["id", "start", "stop", "event", "protime", "protime_late"]
long[long["id"].isin([0, 3])].sort_values(["id", "start"])[show]
''', title="추적 기간을 3년에서 나눈 긴 자료 만들기")
save("lab11_split", c, marks={"312 명 -> 552 줄": 1}, dfmarks={"1.095140": 2, "3.000000": 3, "5.270363": 4})

c = nb.cell('''
from lifelines import CoxTimeVaryingFitter

ctv = CoxTimeVaryingFitter()
ctv.fit(long, id_col="id", event_col="event",
        start_col="start", stop_col="stop",
        formula="dpca + age10 + female + log2_bili + albumin"
                " + C(edema) + protime + protime_late")
b2 = ctv.params_
print("protime HR, 0-3년 :", round(np.exp(b2["protime"]), 2))
print("protime HR, 3년 후:",
      round(np.exp(b2["protime"] + b2["protime_late"]), 2))
p_int = ctv.summary.loc["protime_late", "p"]
print("교호작용 P        :", round(p_int, 4))
print("dpca HR           :", round(np.exp(b2["dpca"]), 3))
''', title="기간별 위험비 (시간과의 교호작용)")
save("lab11_timesplit", c, marks={"0-3년 : 1.57": 1, "3년 후: 1.06": 2, "0.0234": 3, "dpca HR           : 0.913": 4})

# ---------------------------------------------------------------- d. 보정 생존곡선과 층화 Cox
c = nb.cell('''
fig, ax = plt.subplots(figsize=(6.5, 4))
cph.plot_partial_effects_on_outcome("edema", [0, 0.5, 1], ax=ax)
ax.set_xlabel("Years since randomization")
ax.set_ylabel("Predicted survival probability")
plt.tight_layout()
''', title="부종 범주별 예측 생존곡선 (나머지 변수는 중앙값)")
NOTES[c.n] = "## d. 보정 생존곡선과 층화 Cox"
save("lab11_partial", c)

c = nb.cell('''
grid = np.round(np.arange(0, 12.05, 0.1), 1)   # 0, 0.1, ..., 12년
surv = {}
for v, lab in [(1, "D-penicillamine"), (0, "Placebo")]:
    Si = cph.predict_survival_function(df.assign(dpca=v), times=grid)
    surv[lab] = Si.mean(axis=1)          # 312명 곡선의 평균
std = pd.DataFrame(surv)
print(std.loc[[1.0, 5.0, 10.0]].round(3).to_string())

fig, ax = plt.subplots(figsize=(6.5, 4))
std.plot(ax=ax, drawstyle="steps-post")
ax.set_xlabel("Years since randomization")
ax.set_ylabel("Adjusted survival probability")
ax.set_ylim(0, 1)
plt.tight_layout()
''', title="직접 표준화한 보정 생존곡선")
save("lab11_standardized", c, marks={"5.0             0.709    0.699": 1})

c = nb.cell('''
cph_s = CoxPHFitter()
cph_s.fit(df, duration_col="years", event_col="death",
          formula="dpca + age10 + female + log2_bili + albumin"
                  " + protime",
          strata=["edema"])
cph_s.print_summary(decimals=3)
''', title="부종으로 층화한 Cox 모형")
save("lab11_strata", c, marks={"strata = edema": 1, "0.922": 2, "Concordance = 0.806": 3, "log-likelihood ratio test = 132.353 on 6 df": 4})

c = nb.cell('''
comp = pd.DataFrame({
    "unadjusted": cph1.summary.loc["dpca"],
    "adjusted": cph.summary.loc["dpca"],
    "stratified by edema": cph_s.summary.loc["dpca"]}).T
comp[["exp(coef)", "exp(coef) lower 95%",
      "exp(coef) upper 95%", "p"]].round(3)
''', title="세 모형의 치료 효과 비교")
save("lab11_compare", c, dfmarks={"1.059": 1, "0.938": 2, "0.922": 3})

# ---------------------------------------------------------------- e. 과제 (정답 셀)
c = nb.cell('''
b_bili = cph.params_["log2_bili"]                # 두 배당 로그 위험비
for lo, hi in [(1, 2), (5, 10), (10, 11)]:
    hr = np.exp(b_bili * np.log2(hi / lo))       # lo에서 hi로 오를 때
    print(f"bilirubin {lo:>2} -> {hi:>2} mg/dL: HR {hr:.3f}")

pb = pd.DataFrame([df.median()] * 2)             # 중앙값 환자 두 명
pb["dpca"] = 0
pb["bili"] = [1.35, 2.7]                         # 빌리루빈 1.35와 그 두 배
pb["log2_bili"] = np.log2(pb["bili"])
rb = 1 - cph.predict_survival_function(pb, times=[5, 10])
rb.columns = ["bili 1.35", "bili 2.7"]
rb["risk ratio"] = rb["bili 2.7"] / rb["bili 1.35"]
rb.round(3)
''', title="과제 1 정답. 빌리루빈의 위험비 읽기")
save("lab11_hw1", c, marks={"bilirubin  1 ->  2 mg/dL: HR 1.842": 1, "bilirubin  5 -> 10 mg/dL: HR 1.842": 2,
                            "bilirubin 10 -> 11 mg/dL: HR 1.088": 3}, dfmarks={"0.259": 4, "1.725": 5})
NOTES[c.n] = (f"## e. 과제 정답\n\n**과제 1.** 셀 {MULTI_CELL}의 다변수 모형에서 log2_bili 행을 Results 문장으로 옮기고, 이 숫자가 "
              f"말해 주지 않는 것을 두 가지 이상 쓰세요. 빌리루빈 10에서 11 mg/dL로 오를 때의 위험비와, 셀 {HRRR_CELL}의 방법으로 "
              "중앙값 환자의 빌리루빈이 두 배일 때 5년·10년 사망위험도 구해 보세요.")

c = nb.cell('''
df2 = d[cols + ["status"]].dropna().reset_index(drop=True)
df2["log2_bili"] = np.log2(df2["bili"])
df2["event2"] = (df2["status"] > 0).astype(int)   # 이식(1) 또는 사망(2)
print("events: death", df2["death"].sum(),
      " death or transplant", df2["event2"].sum())
cph_tx = CoxPHFitter()
cph_tx.fit(df2, duration_col="years", event_col="event2",
           formula="dpca + age10 + female + log2_bili + albumin"
                   " + C(edema) + protime")
s = cph_tx.summary.loc["dpca"]
print(f"dpca HR {s['exp(coef)']:.3f} ({s['exp(coef) lower 95%']:.3f}"
      f"-{s['exp(coef) upper 95%']:.3f}), p {s['p']:.3f}")
print(d.groupby(d["status"] == 1)[["age", "bili"]].median())  # 이식받은 19명(True)
pd.DataFrame({"death": cph.hazard_ratios_,
              "death or transplant": cph_tx.hazard_ratios_}).round(3)
''', title="과제 2 정답. 사망 또는 간이식을 사건으로 보면")
save("lab11_hw2", c, marks={"events: death 125": 1, "death or transplant 144": 2,
                            "dpca HR 0.986 (0.695-1.398), p 0.937": 3, "True    40.900753   3.2": 4},
     dfmarks={"1.180": 5, "1.205": 6})
NOTES[c.n] = ("**과제 2.** 간이식을 중도절단하지 않고 사건으로 세어(사망 또는 간이식) 셀 "
              f"{MULTI_CELL}의 다변수 모형을 다시 적합하세요. 사건 수, 치료의 위험비, 다른 변수의 위험비가 어떻게 바뀌는지 "
              "비교하고, 이식받은 환자가 어떤 환자였는지와 함께 설명하세요.")

c = nb.cell('''
from lifelines.utils import to_episodic_format

f_t = ("dpca + age10 + female + log2_bili + albumin"
       " + C(edema) + protime + dpca_t")
wrong = df.assign(dpca_t=df["dpca"] * df["years"])   # 동료가 만든 열
cph_w = CoxPHFitter().fit(wrong, "years", "death", formula=f_t)

ep = to_episodic_format(df.assign(id=df.index), duration_col="years",
                        event_col="death", id_col="id", time_gaps=0.1)
ep["dpca_t"] = ep["dpca"] * ep["stop"]             # 그 구간 끝의 시간
ctv_t = CoxTimeVaryingFitter().fit(ep, id_col="id", event_col="death",
                                   start_col="start", stop_col="stop",
                                   formula=f_t)
print(len(df), "명 ->", len(ep), "줄")
print(f"P for dpca_t: wrong {cph_w.summary.loc['dpca_t', 'p']:.1e},"
      f" split {ctv_t.summary.loc['dpca_t', 'p']:.3f}")
pd.DataFrame({"wrong (years)": np.exp(cph_w.params_),
              "split (0.1 y)": np.exp(ctv_t.params_)}).round(3)
''', title="과제 3 정답. '치료 × 추적 시간' 열을 넣은 분석")
save("lab11_hw3", c, marks={"312 명 -> 17289 줄": 1, "P for dpca_t: wrong 4.5e-09": 2, "split 0.458": 3},
     dfmarks={"8.165": 4, "0.585": 5, "0.679": 6, "1.047": 7})
NOTES[c.n] = ("**과제 3.** 동료가 치료 효과가 추적 기간에 따라 변하는지 보려고 `df[\"dpca_t\"] = df[\"dpca\"] * df[\"years\"]` "
              f"열을 만들어 셀 {MULTI_CELL}의 다변수 모형에 더했고, \"치료와 시간의 교호작용이 뚜렷했다(위험비 연 0.68배, "
              "P < 0.001). 치료의 비례위험 가정이 깨졌다\"라고 썼습니다. 무엇이 왜 틀렸는지 찾고, 추적 기간을 나눈 자료로 "
              "바르게 다시 적합해 보세요.")

print("lab11: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 11. Cox 비례위험모형",
    intro=("사회약학 연구방법 노트의 '실습 11. Cox 비례위험모형'에 딸린 노트북입니다. R survival 패키지의 pbc 자료"
           "(Mayo Clinic 원발성 담즙성 담관염 무작위배정 시험 312명)로 사건과 결측을 정리하고, 단변수·다변수 Cox 모형을 "
           "적합하고, 비례위험 가정을 점검한 뒤 보정 생존곡선과 층화 Cox 모형을 만듭니다. 셀을 위에서부터 차례로 "
           "실행하세요. 설명과 출력 읽는 법은 사이트의 실습 11에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import warnings
    import pandas as pd
    from lifelines import CoxPHFitter, CoxTimeVaryingFitter, KaplanMeierFitter
    from lifelines.statistics import proportional_hazard_test
    from statsmodels.nonparametric.smoothers_lowess import lowess

    ns = nb.ns
    pbc_, d_, df_, cph_, cph1_, cphs_ = ns["pbc"], ns["d"], ns["df"], ns["cph"], ns["cph1"], ns["cph_s"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r1 = lambda x: f"{x:.1f}"
    r2 = lambda x: f"{x:.2f}"
    r3 = lambda x: f"{x:.3f}"
    hr = lambda m, v: float(np.exp(m.params_[v]))
    ci = lambda m, v: (r2(m.summary.loc[v, "exp(coef) lower 95%"]), r2(m.summary.loc[v, "exp(coef) upper 95%"]))
    # 글(content/lab11.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "lifelines 설치", 2: "패키지 불러오기와 버전 확인", 4: "무작위배정 환자만 남기기",
             5: "사건, 시간, 치료 변수 만들기", 6: "결측값 확인과 분석용 자료 만들기", 8: "치료만 넣은 Cox 모형 (단변수)",
             9: "단변수 Cox 모형과 로그순위 검정 비교", 10: "다변수 Cox 모형 (formula 사용)",
             11: "결측값이 있는 변수를 formula에 넣으면", 13: "위험비와 누적위험의 비(상대위험도)는 다르다",
             14: "논문용 위험비 표 만들기", 16: "빌리루빈을 로그로 넣은 이유 확인", 17: "비례위험 검정 (척도화 Schoenfeld 잔차)",
             19: "척도화 Schoenfeld 잔차 그림", 21: "추적 기간을 3년에서 나눈 긴 자료 만들기",
             22: "기간별 위험비 (시간과의 교호작용)", 23: "부종 범주별 예측 생존곡선 (나머지 변수는 중앙값)",
             24: "직접 표준화한 보정 생존곡선", 25: "부종으로 층화한 Cox 모형", 26: "세 모형의 치료 효과 비교"}
    for k_, t_ in cited.items():
        same(f"cell {k_} title", nb.cells[k_ - 1].title, t_)
    same("number of cells", len(nb.cells), 29)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 대응표 (이 실습의 자료 열)
        same("randomized, arms", (len(df_), int(df_["dpca"].sum()), int((df_["dpca"] == 0).sum())), (312, 158, 154))
        ct = pd.crosstab(d_["status"], d_["trt"])
        same("deaths, transplants, censored", (int(ct.loc[2].sum()), int(ct.loc[1].sum()), int(ct.loc[0].sum())),
             (125, 19, 168))
        same("transplants by arm", (int(ct.loc[1, 1.0]), int(ct.loc[1, 2.0])), (10, 9))
        same("deaths by arm", (int(ct.loc[2, 1.0]), int(ct.loc[2, 2.0])), (65, 60))
        same("death proportion by arm", (f"{65 / 158:.1%}", f"{60 / 154:.1%}"), ("41.1%", "39.0%"))
        same("crude and adjusted HR", (r2(hr(cph1_, "dpca")), ci(cph1_, "dpca"), r2(hr(cph_, "dpca")), ci(cph_, "dpca")),
             ("1.06", ("0.75", "1.50"), "0.94", ("0.64", "1.37")))
        same("stratified HR", (r2(hr(cphs_, "dpca")), ci(cphs_, "dpca")), ("0.92", ("0.63", "1.35")))
        same("EPV", r1(125 / len(cph_.params_)), "15.6")
        # a 절
        na_ = d_.isna().sum()
        same("missing counts", {k: int(v) for k, v in na_[na_ > 0].items()},
             {"chol": 28, "copper": 2, "trig": 30, "platelet": 4})
        same("dropna all", (len(d_.dropna()), 312 - len(d_.dropna()), f"{(312 - len(d_.dropna())) / 312:.0%}"),
             (276, 36, "12%"))
        same("bili median, mean", (r2(df_["bili"].median()), r2(df_["bili"].mean())), ("1.35", "3.26"))
        # b 절
        same("-log2(p) uni", r3(-np.log2(cph1_.summary.loc["dpca", "p"])), "0.416")
        same("mean age by arm", (r1(d_.loc[d_["trt"] == 1, "age"].mean()), r1(d_.loc[d_["trt"] == 2, "age"].mean())),
             ("51.4", "48.6"))
        ma = CoxPHFitter().fit(df_, "years", "death", formula="dpca + age10")
        same("dpca HR adjusted for age only", r3(hr(ma, "dpca")), "0.929")
        same("bili z", r3(cph_.summary.loc["log2_bili", "z"]), "8.861")
        same("albumin SD", r2(df_["albumin"].std()), "0.42")
        same("edema 0.5 HR (CI)", (r3(hr(cph_, "C(edema)[T.0.5]")),
                                   r3(cph_.summary.loc["C(edema)[T.0.5]", "exp(coef) lower 95%"]),
                                   r3(cph_.summary.loc["C(edema)[T.0.5]", "exp(coef) upper 95%"])), ("1.218", "0.697", "2.130"))
        e1 = df_[df_["edema"] == 1]
        same("edema 1.0 patients, deaths", (len(e1), int(e1["death"].sum())), (20, 19))
        same("edema 1.0 by arm", (int(e1["dpca"].sum()), int((e1["dpca"] == 0).sum())), (10, 10))
        # ConvergenceError callout
        f5 = "dpca + age10 + log2_bili + albumin + C(edema)"
        try:
            CoxPHFitter().fit(df_, "years", "death", formula=f5)
            conv = "converged"
        except Exception as e:  # noqa: BLE001
            conv = type(e).__name__
        same("default fit of f5", conv, "ConvergenceError")
        m5 = CoxPHFitter().fit(df_, "years", "death", formula=f5, fit_options={"step_size": 0.5})
        same("step_size 0.5 dpca HR", r3(hr(m5, "dpca")), "0.916")
        # 결측 변수를 formula 없이 넣었을 때의 오류 (이런 오류가 나면 문단)
        try:
            CoxPHFitter().fit(d_[["years", "death", "dpca", "chol"]], "years", "death")
            nan_err = "no error"
        except TypeError as e:
            nan_err = str(e).split(".")[0]
        same("NaN error without formula", nan_err, "NaNs were detected in the dataset")
        same("age10 coef", r3(cph_.params_["age10"]), "0.314")
        same("1 yr HR ^10", r3(np.exp(cph_.params_["age10"] / 10) ** 10), r3(hr(cph_, "age10")))
        same("albumin per -0.5", r2(0.372 ** -0.5), "1.64")
        # 위험비와 누적위험(셀 13): 중앙값 환자와 1·10년 위험
        med = df_.median()
        same("median patient", (int(med["female"]), r1(med["age10"] * 10), r2(med["bili"]), r2(med["albumin"]),
                                 r1(med["protime"]), int(med["dpca"]), r1(med["edema"])),
             (1, "49.8", "1.35", "3.55", "10.6", 1, "0.0"))
        rk = ns["risk"]
        same("1-year risks, RR", (f"{rk.loc[1, 'no edema']:.1%}", f"{rk.loc[1, 'edema 1.0']:.1%}", r2(rk.loc[1, "risk ratio"])),
             ("1.7%", "4.3%", "2.55"))
        same("5-year risks", (f"{rk.loc[5, 'no edema']:.1%}", f"{rk.loc[5, 'edema 1.0']:.1%}", r2(rk.loc[5, "risk ratio"])),
             ("15.0%", "34.2%", "2.29"))
        same("10-year risks", (f"{rk.loc[10, 'no edema']:.1%}", f"{rk.loc[10, 'edema 1.0']:.1%}"), ("46.0%", "79.6%"))
        # 포레스트 그림: P > 0.05인 변수
        same("P > 0.05 rows", sorted(cph_.summary.index[cph_.summary["p"] > 0.05]), ["C(edema)[T.0.5]", "dpca", "female"])
        same("AIC C", (r3(CoxPHFitter().fit(df_, "years", "death",
                                            formula="dpca + age10 + female + bili + albumin + C(edema) + protime").concordance_index_),
                       r3(cph_.concordance_index_)), ("0.832", "0.846"))
        # c 절: km 변환, 잔차 그림, log(-log) 그림
        pk = proportional_hazard_test(cph_, df_, time_transform="km").summary["p"]
        same("PH km p", (r3(pk["C(edema)[T.0.5]"]), r3(pk["protime"])), ("0.045", "0.077"))
        same("dpca coef", r3(cph_.params_["dpca"]), "-0.065")
        rr_ = cph_.compute_residuals(df_, kind="scaled_schoenfeld")
        tt = df_.loc[rr_.index, "years"]
        sm_d = lowess(rr_["dpca"] + cph_.params_["dpca"], tt, frac=0.5)
        same("dpca smooth range", (r1(sm_d[:, 1].min()), r1(sm_d[:, 1].max())), ("-0.6", "0.7"))
        same("deaths after 8 y", int(((df_["death"] == 1) & (df_["years"] > 8)).sum()), 17)
        sm_p = lowess(rr_["protime"] + cph_.params_["protime"], tt, frac=0.5)
        early_p = sm_p[(sm_p[:, 0] >= 1) & (sm_p[:, 0] <= 2), 1]
        mid_p = sm_p[(sm_p[:, 0] >= 4) & (sm_p[:, 0] <= 6), 1]
        same("protime smooth 1-2 y about 0.4", r1(early_p.mean()), "0.4")
        same("protime smooth 4-6 y near 0", abs(mid_p).max() < 0.15, True)
        ll = {}
        for v in [0, 0.5, 1.0]:
            s_ = df_[df_["edema"] == v]
            k_ = KaplanMeierFitter().fit(s_["years"], s_["death"])
            ll[v] = np.array([np.log(-np.log(k_.predict(t_))) for t_ in [1, 3, 7]])
        same("loglog gaps 1.0 - 0", [r1(x) for x in ll[1.0] - ll[0]], ["3.5", "2.5", "1.9"])
        same("loglog gaps 0.5 - 0", [r1(x) for x in ll[0.5] - ll[0]], ["2.1", "1.3", "0.8"])
        same("bili median by edema", (r1(df_.loc[df_["edema"] == 1, "bili"].median()),
                                      r1(df_.loc[df_["edema"] == 0, "bili"].median())), ("7.8", "1.2"))
        # 기간을 나눈 자료와 기간별 위험비, 직접 바꿔 보기(cut 2, 4, 5)
        same("followed > 3 y", int((df_["years"] > 3).sum()), 240)
        same("median death time", r1(df_.loc[df_["death"] == 1, "years"].median()), "3.3")
        cuts = {}
        for cut in [2, 3, 4, 5]:
            x = df_.assign(id=df_.index)
            early = x.assign(start=0.0, stop=x["years"].clip(upper=cut), late=0)
            early["event"] = np.where(x["years"] <= cut, x["death"], 0)
            late = x[x["years"] > cut].assign(start=float(cut), late=1)
            late["stop"] = late["years"]
            late["event"] = late["death"]
            lg = pd.concat([early, late]).drop(columns=["years", "death"])
            lg["protime_late"] = lg["protime"] * lg["late"]
            cc_ = CoxTimeVaryingFitter().fit(lg, id_col="id", event_col="event", start_col="start", stop_col="stop",
                                             formula=FORMULA + " + protime_late")
            bb = cc_.params_
            cuts[cut] = (r2(np.exp(bb["protime"])), r2(np.exp(bb["protime"] + bb["protime_late"])),
                         f"{cc_.summary.loc['protime_late', 'p']:.4f}")
        same("cut 2", cuts[2], ("1.89", "1.06", "0.0009"))
        same("cut 3", cuts[3], ("1.57", "1.06", "0.0234"))
        same("cut 4 p", cuts[4][2], "0.0295")
        same("cut 5 p", cuts[5][2], "0.1026")
        # d 절: 예측 곡선, 표준화, 층화
        cv = cph_._central_values
        same("partial-effects central values", (int(cv["dpca"].iloc[0]), int(cv["female"].iloc[0])), (1, 1))
        pe = {}
        for v in [0.0, 1.0]:
            xx = cv.copy()
            xx["edema"] = v
            pe[v] = [r2(y) for y in cph_.predict_survival_function(xx, times=[5, 10]).values.ravel()]
        same("partial effect curves", (pe[0.0], pe[1.0]), (["0.86", "0.56"], ["0.68", "0.23"]))
        st_ = ns["std"]
        same("standardized 1, 5, 10 y", [(f"{st_.loc[t_, 'D-penicillamine']:.1%}", f"{st_.loc[t_, 'Placebo']:.1%}")
                                         for t_ in [1.0, 5.0, 10.0]],
             [("92.7%", "92.4%"), ("70.9%", "69.9%"), ("45.9%", "44.5%")])
        ps = proportional_hazard_test(cphs_, df_, time_transform="rank").summary["p"]
        same("PH after stratification", (ps.idxmin(), r3(ps.min()), r2(ps["protime"])), ("female", "0.076", "0.37"))
        same("stratified partial LL", r3(cphs_.log_likelihood_), "-458.464")
        # 과제 1
        hw1 = ns["rb"]
        same("hw1 risks", [r3(hw1.loc[t_, c_]) for t_ in (5, 10) for c_ in ("bili 1.35", "bili 2.7", "risk ratio")],
             ["0.150", "0.259", "1.725", "0.460", "0.679", "1.475"])
        same("bili 1 -> 4", r3(np.exp(cph_.params_["log2_bili"] * 2)), "3.393")
        same("bili CI", (r3(cph_.summary.loc["log2_bili", "exp(coef) lower 95%"]),
                         r3(cph_.summary.loc["log2_bili", "exp(coef) upper 95%"])), ("1.609", "2.108"))
        # 과제 2
        ctx = ns["cph_tx"]
        same("composite events", int(ns["df2"]["event2"].sum()), 144)
        c_uni = CoxPHFitter().fit(ns["df2"], "years", "event2", formula="dpca")
        same("composite unadjusted HR", (r2(hr(c_uni, "dpca")), ci(c_uni, "dpca")), ("1.06", ("0.76", "1.47")))
        same("composite age, protime, bili", (r3(hr(ctx, "age10")), r3(hr(ctx, "protime")), r3(hr(ctx, "log2_bili"))),
             ("1.180", "1.205", "1.824"))
        same("composite age P", r3(ctx.summary.loc["age10", "p"]), "0.053")
        tx = d_[d_["status"] == 1]
        same("transplant median age, bili", (r1(tx["age"].median()), r1(tx["bili"].median()),
                                             r1(d_.loc[d_["status"] != 1, "age"].median()),
                                             r1(d_.loc[d_["status"] != 1, "bili"].median())), ("40.9", "3.2", "50.5", "1.3"))
        # 과제 3
        cw, cs = ns["cph_w"], ns["ctv_t"]
        same("wrong model", (r3(hr(cw, "dpca")), r3(hr(cw, "dpca_t")), r2(np.exp(cw.params_["dpca"] + 5 * cw.params_["dpca_t"]))),
             ("8.165", "0.679", "1.18"))
        same("wrong albumin, edema 1.0, bili", (r3(hr(cw, "albumin")), r2(hr(cw, "C(edema)[T.1.0]")), r2(hr(cw, "log2_bili"))),
             ("0.585", "2.10", "1.79"))
        same("split model dpca_t (CI)", (r3(float(np.exp(cs.params_["dpca_t"]))),
                                         r2(cs.summary.loc["dpca_t", "exp(coef) lower 95%"]),
                                         r2(cs.summary.loc["dpca_t", "exp(coef) upper 95%"])), ("1.047", "0.93", "1.18"))
        same("split model dpca, albumin", (r3(float(np.exp(cs.params_["dpca"]))), r3(float(np.exp(cs.params_["albumin"])))),
             ("0.781", "0.363"))
        same("log2(1.1)", r2(np.log2(1.1)), "0.14")
        # 사망 시점마다 나눈 자료(가장 정확한 분할)로도 같은 결론
        dt = np.sort(df_.loc[df_["death"] == 1, "years"].unique())
        rows = []
        for i_, rw in df_.assign(id=df_.index).iterrows():
            cutsi = [t_ for t_ in dt if t_ < rw["years"]]
            for s0, s1 in zip([0.0] + cutsi, cutsi + [rw["years"]]):
                q = rw.to_dict()
                q.update(start=s0, stop=s1, event=int(rw["death"] == 1 and s1 == rw["years"]))
                rows.append(q)
        ex = pd.DataFrame(rows).drop(columns=["years", "death"])
        ex["dpca_t"] = ex["dpca"] * ex["stop"]
        ce = CoxTimeVaryingFitter().fit(ex, id_col="id", event_col="event", start_col="start", stop_col="stop",
                                        formula=ns["f_t"])
        same("exact split dpca_t HR, P", (r2(float(np.exp(ce.params_["dpca_t"]))), r2(ce.summary.loc["dpca_t", "p"])),
             ("1.06", "0.35"))
        same("death times", len(dt), 122)
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab11.html passed")
