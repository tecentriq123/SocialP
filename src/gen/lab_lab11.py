"""실습 11 · Cox 비례위험모형 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab11.py
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

# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab11_install", c)

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ---------------------------------------------------------------- 나. Cox 모형과 위험비
c = nb.cell('''
from lifelines import CoxPHFitter

cph1 = CoxPHFitter()
cph1.fit(df, duration_col="years", event_col="death", formula="dpca")
cph1.print_summary(decimals=3)
''', title="치료만 넣은 Cox 모형 (단변수)")
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
save("lab11_multi", c, marks={"0.938": 1, "1.369": 2, "1.842": 3, "0.372": 4,
                              "C(edema)[T.0.5]": 5, "2.582": 6, "<0.0005": 7,
                              "Concordance = 0.846": 8, "Partial AIC = 1094.350": 9,
                              "log-likelihood ratio test = 201.583 on 8 df": 10})

c = nb.cell('''
CoxPHFitter().fit(d, "years", "death", formula="dpca + chol")
''', title="결측값이 있는 변수를 formula에 넣으면", expect_error=True)
save("lab11_naerror", c)

c = nb.cell('''
b = cph.params_                         # 계수(로그 위험비)
print("나이 1세당 HR   :", round(np.exp(b["age10"] / 10), 3))
print("나이 10세당 HR  :", round(np.exp(b["age10"]), 3))
print("나이 20세당 HR  :", round(np.exp(b["age10"] * 2), 3))
print("빌리루빈 2배 HR :", round(np.exp(b["log2_bili"]), 3))
print("빌리루빈 4배 HR :", round(np.exp(b["log2_bili"] * 2), 3))
print("알부민 0.5 g/dL 감소 HR:", round(np.exp(-0.5 * b["albumin"]), 3))
''', title="위험비의 단위 바꾸기")
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

# ---------------------------------------------------------------- 다. 비례위험 가정 점검
c = nb.cell('''
from lifelines.statistics import proportional_hazard_test

ph = proportional_hazard_test(cph, df, time_transform="rank")
ph.print_summary(decimals=3)
''', title="비례위험 검정 (척도화 Schoenfeld 잔차)")
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

# ---------------------------------------------------------------- 라. 보정 생존곡선과 층화 Cox
c = nb.cell('''
fig, ax = plt.subplots(figsize=(6.5, 4))
cph.plot_partial_effects_on_outcome("edema", [0, 0.5, 1], ax=ax)
ax.set_xlabel("Years since randomization")
ax.set_ylabel("Predicted survival probability")
plt.tight_layout()
''', title="부종 범주별 예측 생존곡선 (나머지 변수는 중앙값)")
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

print("lab11: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- verification (numbers quoted in the text; not shown as cells)
if __name__ == "__main__":
    import warnings
    ns = nb.ns
    df_, CoxPHFitter_ = ns["df"], ns["CoxPHFitter"]
    f5 = "dpca + age10 + log2_bili + albumin + C(edema)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            CoxPHFitter_().fit(df_, "years", "death", formula=f5)
            print("VERIFY default fit of", f5, ": converged")
        except Exception as e:
            print("VERIFY default fit of", f5, ":", type(e).__name__)
        m5 = CoxPHFitter_().fit(df_, "years", "death", formula=f5, fit_options={"step_size": 0.5})
        print("VERIFY step_size=0.5 HR dpca", round(float(np.exp(m5.params_["dpca"])), 3))
        try:
            CoxPHFitter_().fit(df_, "years", "death", formula=ns["cph"].formula if hasattr(ns["cph"], "formula") else None,
                               strata=["C(edema)[T.0.5]"])
            print("VERIFY strata advice: ran")
        except Exception as e:
            print("VERIFY strata advice:", type(e).__name__, str(e)[:120])
        # time split with other cuts
        long_ = ns["long"]
        for cut in [2, 3, 4, 5]:
            x = df_.assign(id=df_.index)
            early = x.assign(start=0.0, stop=x["years"].clip(upper=cut), late=0)
            early["event"] = np.where(x["years"] <= cut, x["death"], 0)
            late = x[x["years"] > cut].assign(start=float(cut), late=1)
            late["stop"] = late["years"]; late["event"] = late["death"]
            lg = ns["pd"].concat([early, late]).drop(columns=["years", "death"])
            lg["protime_late"] = lg["protime"] * lg["late"]
            c = ns["CoxTimeVaryingFitter"]().fit(lg, id_col="id", event_col="event", start_col="start", stop_col="stop",
                formula="dpca + age10 + female + log2_bili + albumin + C(edema) + protime + protime_late")
            b = c.params_
            print(f"VERIFY cut {cut}: early {np.exp(b['protime']):.2f} late {np.exp(b['protime'] + b['protime_late']):.2f}"
                  f" p {c.summary.loc['protime_late', 'p']:.4f}")
        # stratified PH test
        r = ns["proportional_hazard_test"](ns["cph_s"], df_, time_transform="rank")
        print("VERIFY PH test after stratification (rank):", r.summary["p"].round(3).to_dict())
        # km transform
        r = ns["proportional_hazard_test"](ns["cph"], df_, time_transform="km")
        print("VERIFY PH km:", r.summary["p"].round(3).to_dict())
        # median values used in cell 12
        print("VERIFY medians:", df_.median().round(3).to_dict())
        # EPV
        print("VERIFY params", len(ns["cph"].params_), "events", int(df_["death"].sum()))
        # chance imbalance in age (text of cell 9, mark 1)
        dd = ns["d"]
        print("VERIFY mean age by trt:", dd.groupby("trt")["age"].mean().round(1).to_dict())
        ma = CoxPHFitter_().fit(df_, "years", "death", formula="dpca + age10")
        print("VERIFY dpca+age10 HR:", round(float(np.exp(ma.params_["dpca"])), 3))
        # partial-effects curve values quoted in the text
        base = ns["cph"]._central_values
        for v in [0, 1.0]:
            x = base.copy(); x["edema"] = v
            print("VERIFY partial effect edema", v, ns["cph"].predict_survival_function(x, times=[5, 10]).values.ravel().round(2))
        print("VERIFY albumin SD:", round(float(df_["albumin"].std()), 2), " died>3y split rows:", int((df_["years"] > 3).sum()))
        print("VERIFY median death time:", round(float(df_.loc[df_["death"] == 1, "years"].median()), 2))
        # log(-log S) gaps by edema (text of cell 19) and edema balance by arm
        KM_ = ns["KaplanMeierFitter"]
        ll = {}
        for v in [0, 0.5, 1.0]:
            s_ = df_[df_["edema"] == v]
            k_ = KM_().fit(s_["years"], s_["death"])
            ll[v] = np.array([np.log(-np.log(k_.predict(t_))) for t_ in [1, 3, 7]])
        print("VERIFY loglog gaps 1.0-0:", (ll[1.0] - ll[0]).round(2), " 0.5-0:", (ll[0.5] - ll[0]).round(2))
        print("VERIFY bili median by edema:", df_.groupby("edema")["bili"].median().to_dict())
        print("VERIFY edema x arm:", ns["pd"].crosstab(df_["edema"], df_["dpca"]).to_dict())
