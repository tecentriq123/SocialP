"""실습 10 · 반복측정 자료 분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab10.py
      then  python3 tools/nbcheck.py lab10   (runs the notebook alone in a fresh process and compares outputs)
      (LABDEBUG=1 prints every cell's text output, for writing marks)
Writes figs/lab10_*.html (insert in content/lab10.html with <!--FIG:lab10_xxx-->),
pub/notebooks/lab10.ipynb (download link at the top of content/lab10.html) and checks/lab10.expected.json.
Data: lme4 sleepstudy and MASS epil (Rdatasets URLs -> offline copies via labkit).
The VERIFY block at the end asserts the numbers quoted in the text of content/lab10.html that are not
marked in a cell output (대응표, 스파게티 그림 문단, 데이터 버전 참고, 직접 바꿔 보기, 과제 problems and answers).
(2026-10-14: version cell 1, Mauchly/Huynh–Feldt cell, 과제 1–3 and the notebook added; old cells 1–8 are now
2–9 and old cells 9–19 are now 11–21.)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab10")
DEBUG = os.environ.get("LABDEBUG")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


def freeze_clock(h):
    """GEE summary()가 찍는 실행 날짜와 시각(Date, Time 줄)을 고정값으로 바꾼다(gen/lab_lab09.py와 같음). 다시 돌려도
    같은 파일이 나오게 하려는 것이고 통계 출력은 건드리지 않는다. 같은 길이의 글자로 바꾸므로 줄 맞춤도 그대로다."""
    h = re.sub(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}", r"\g<1>Wed, 14 Oct 2026", h)
    return re.sub(r"(Time:\s+)\d{2}:\d{2}:\d{2}", r"\g<1>09:10:59", h)


def save(name, c, marks=None, **kw):
    if DEBUG:
        print(f"\n===== {name} (셀 {c.n}) =====\n{c.stdout}{c.value_repr or ''}")
        if c.warns:
            print("WARN:", c.warns)
        if c.error:
            print("ERR:", c.error)
    h = nb.html(c, **kw)
    if marks:  # labkit's own marks need the text in BOTH print output and table; mark the output area here
        i = h.index('<div class="cell-out">')
        head, out = h[:i], h[i:]
        for sub, n in marks.items():
            if sub.startswith("~"):  # regex on the escaped output
                m = re.search(sub[1:], out)
                j = m.end() if m else -1
            else:
                e = _html.escape(sub)
                j = out.find(e)
                j = j + len(e) if j >= 0 else -1
            if j < 0:
                raise ValueError(f"{name}: mark text not found: {sub!r}")
            out = out[:j] + f'<span class="mk">{n}</span>' + out[j:]
        h = head + out
    nb.save_fragment(name, freeze_clock(h))


# ------------------------------------------------------------------ a. 실습 데이터와 자료구조 변환
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib, scipy, statsmodels

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("matplotlib  ", matplotlib.__version__)
print("scipy       ", scipy.__version__)
print("statsmodels ", statsmodels.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab10_versions", c, marks={"~pandas +[0-9.]+": 1, "~statsmodels +[0-9.]+": 2})
NOTES[c.n] = ("## a. 실습 데이터와 자료구조 변환\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤, sleepstudy와 epil 자료를 "
              "인터넷에서 읽어 옵니다. 인터넷 연결이 필요합니다.")

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/csv/"
       "lme4/sleepstudy.csv")
sleep = pd.read_csv(url).drop(columns="rownames")
print(sleep.shape)
sleep.head(12)
''', title="sleepstudy 불러오기 (긴 형식)", max_rows=12)
save("lab10_load", c, marks={"(180, 3)": 1})

c = nb.cell('''
print(sleep["Subject"].nunique())            # 참가자 수
print(sleep.groupby("Subject").size().value_counts())
print(sleep["Days"].unique())                # 측정 시점
''', title="몇 명을 몇 번 측정했나")
save("lab10_check", c, marks={"~<pre>18": 1, "10    18": 2})

c = nb.cell('''
wide = sleep.pivot(index="Subject", columns="Days",
                   values="Reaction")
print(wide.shape)
wide.round(0).head()
''', title="긴 형식 → 넓은 형식 (pivot)")
save("lab10_pivot", c, marks={"(18, 10)": 1})

c = nb.cell('''
long = wide.reset_index().melt(id_vars="Subject",
                               var_name="Days",
                               value_name="Reaction")
long = long.sort_values(["Subject", "Days"])
print(long.shape)
long.head()
''', title="넓은 형식 → 긴 형식 (melt)")
save("lab10_melt", c, marks={"(180, 3)": 1})

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7, 4.5))
for sid, g in sleep.groupby("Subject"):      # 참가자마다 선 하나
    ax.plot(g["Days"], g["Reaction"], color="grey", alpha=0.5)
mean = sleep.groupby("Days")["Reaction"].mean()
ax.plot(mean.index, mean.values, color="C0", lw=3,
        label="Mean of 18 subjects")
ax.set_xlabel("Days of sleep deprivation")
ax.set_ylabel("Average reaction time (ms)")
ax.legend()
plt.show()
''', title="스파게티 그림")
save("lab10_spaghetti", c)

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/csv/"
       "MASS/epil.csv")
epil = pd.read_csv(url).drop(columns="rownames")
epil["trt"] = pd.Categorical(epil["trt"],
                             categories=["placebo", "progabide"])
print(epil.shape, epil["subject"].nunique())
epil.head(8)
''', title="epil 불러오기", max_rows=8)
save("lab10_epil", c, marks={"(236, 9) 59": 1})

# ------------------------------------------------------------------ b. 반복측정 분산분석
c = nb.cell('''
from statsmodels.stats.anova import AnovaRM

aov = AnovaRM(sleep, depvar="Reaction", subject="Subject",
              within=["Days"]).fit()
print(aov)
''', title="반복측정 분산분석 (AnovaRM)")
save("lab10_anovarm", c, marks={"18.7027": 1, "9.0000": 2, "153.0000": 3, r"~0\.0000(?=\n)": 4})
NOTES[c.n] = "## b. 반복측정 분산분석"

c = nb.cell('''
from scipy import stats

S = np.cov(wide.values, rowvar=False)   # 10×10 공분산 행렬
k, n = S.shape[0], wide.shape[0]        # 시점 10, 참가자 18
# 이중 중심화: 행 평균·열 평균을 빼고 전체 평균을 더함
Sc = S - S.mean(axis=0) - S.mean(axis=1)[:, None] + S.mean()
eps = np.trace(Sc)**2 / ((k - 1) * np.sum(Sc**2))
print("GG epsilon:", round(eps, 3))

F = aov.anova_table.loc["Days", "F Value"]
df1, df2 = k - 1, (k - 1) * (n - 1)
print("보정 자유도:", round(eps * df1, 2), round(eps * df2, 2))
print("p, 보정 없음:", stats.f.sf(F, df1, df2))
print("p, GG 보정  :", stats.f.sf(F, eps * df1, eps * df2))
print("p, 하한 보정:", stats.f.sf(F, 1, n - 1))
''', title="Greenhouse–Geisser ε 직접 계산")
save("lab10_gg", c, marks={"0.369": 1, "3.32 56.46": 2})
GG_CELL = c.n

c = nb.cell('''
lam = np.linalg.eigvalsh(Sc)[1:]         # 고윳값 k - 1개 (맨 앞의 0 하나는 뺌)
W = np.prod(lam) / np.mean(lam)**(k - 1)  # Mauchly의 W (1이면 구형성)
d = 1 - (2*(k - 1)**2 + (k - 1) + 2) / (6*(k - 1)*(n - 1))
chi2 = -(n - 1) * d * np.log(W)          # 카이제곱 근사
df_m = k * (k - 1) // 2 - 1
print("Mauchly W:", round(W, 6), " chi2:", round(chi2, 2),
      " df:", df_m, " p:", stats.chi2.sf(chi2, df_m))

hf = (n*(k - 1)*eps - 2) / ((k - 1)*(n - 1 - (k - 1)*eps))
hf = min(hf, 1)                          # 1을 넘으면 1로 둠
print("HF epsilon:", round(hf, 3))
print("p, HF 보정  :", stats.f.sf(F, hf * df1, hf * df2))
''', title="Mauchly 검정과 Huynh–Feldt ε")
save("lab10_mauchly", c, marks={"Mauchly W: 0.000219": 1, "~ p: [0-9.e-]+": 2,
                                "HF epsilon: 0.469": 3, "~p, HF 보정  : [0-9.e-]+": 4})
MAUCHLY_CELL = c.n

c = nb.cell('''
short = sleep.drop(index=0)      # 첫 행(308번 참가자 0일) 삭제
AnovaRM(short, depvar="Reaction", subject="Subject",
        within=["Days"]).fit()
''', title="한 칸이 빠지면", expect_error=True)
save("lab10_unbal", c)
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

# ------------------------------------------------------------------ c. 선형 혼합모형
c = nb.cell('''
import statsmodels.formula.api as smf

m1 = smf.mixedlm("Reaction ~ Days", data=sleep,
                 groups=sleep["Subject"]).fit()
print(m1.summary())
''', title="무작위 절편 모형")
save("lab10_ri", c, marks={"REML": 1, "No. Groups:        18": 2, "960.4568": 3,
                           "-893.2325": 4, "Yes": 5, r"~Days\s+10\.467\s+0\.804[^\n]*": 6, r"~Group Var\s+1378\.176[^\n]*": 7})
RI_CELL = c.n
NOTES[c.n] = (f"한 칸이 빠지면 오류가 나는 것을 보여 주는 셀(셀 {ERR_CELL})은 이 노트북에서 뺐습니다"
              "(사이트의 실습 10 b 절에 있습니다).\n\n## c. 선형 혼합모형")

c = nb.cell('''
tau2 = m1.cov_re.iloc[0, 0]     # 참가자 간 분산 (Group Var)
sigma2 = m1.scale               # 참가자 내 분산 (Scale)
print("ICC:", round(tau2 / (tau2 + sigma2), 3))

ols = smf.ols("Reaction ~ Days", data=sleep).fit()
print("OLS의 Days SE      :", round(ols.bse["Days"], 3))
print("무작위 절편 Days SE:", round(m1.bse_fe["Days"], 3))
''', title="ICC와 표준오차 비교")
save("lab10_icc", c, marks={"0.589": 1, "1.238": 2, "Days SE: 0.804": 3})

c = nb.cell('''
m2 = smf.mixedlm("Reaction ~ Days", data=sleep,
                 groups=sleep["Subject"],
                 re_formula="~Days").fit()
print(m2.summary())
''', title="무작위 절편 + 기울기 모형")
save("lab10_rs", c, marks={"654.9405": 1, r"~Days\s+10\.467\s+1\.546[^\n]*": 3, r"~Group Var\s+612\.096[^\n]*": 4,
                           r"~Group x Days Cov\s+9\.605[^\n]*": 5, r"~Days Var\s+35\.072[^\n]*": 6, "-871.8141": 2})
RS_CELL = c.n

c = nb.cell('''
lr = 2 * (m2.llf - m1.llf)        # 두 모형 모두 REML로 적합
print("LR 통계량:", round(lr, 2))
print("p (자유도 2):", stats.chi2.sf(lr, df=2))

m2_ml = smf.mixedlm("Reaction ~ Days", data=sleep,
                    groups=sleep["Subject"],
                    re_formula="~Days").fit(reml=False)
vc = pd.DataFrame(
    {"REML": [m2.cov_re.iloc[0, 0], m2.cov_re.iloc[1, 1], m2.scale],
     "ML": [m2_ml.cov_re.iloc[0, 0], m2_ml.cov_re.iloc[1, 1],
            m2_ml.scale]},
    index=["intercept var", "slope var", "residual var"])
vc.round(1)
''', title="우도비 검정, REML과 ML")
save("lab10_lr", c, marks={"42.84": 1})
LR_CELL = c.n

c = nb.cell('''
ols_cl = smf.ols("Reaction ~ Days", data=sleep).fit(
    cov_type="cluster", cov_kwds={"groups": sleep["Subject"]})
se = pd.Series({"OLS (independent)": ols.bse["Days"],
                "OLS + cluster-robust": ols_cl.bse["Days"],
                "random intercept": m1.bse_fe["Days"],
                "random intercept + slope": m2.bse_fe["Days"]})
se.round(3).to_frame("SE of Days")
''', title="Days 계수의 표준오차 네 가지")
save("lab10_se", c)

c = nb.cell('''
fe = m2.fe_params                        # 고정효과(절편, 기울기)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
for ax, sid in zip(axes, [308, 335, 350]):
    g = sleep[sleep["Subject"] == sid]
    x = g["Days"]
    re = m2.random_effects[sid]          # 이 사람의 무작위효과
    b0 = fe["Intercept"] + re["Group"]
    b1 = fe["Days"] + re["Days"]
    own_b1, own_b0 = np.polyfit(x, g["Reaction"], 1)
    ax.scatter(x, g["Reaction"], color="grey", s=15)
    ax.plot(x, own_b0 + own_b1 * x, "--", color="C1",
            label="Own data only")
    ax.plot(x, b0 + b1 * x, color="C0", label="Mixed model")
    ax.plot(x, fe["Intercept"] + fe["Days"] * x, ":",
            color="black", label="Average")
    ax.set_title(f"Subject {sid}: slope {own_b1:.1f} -> {b1:.1f}")
    ax.set_xlabel("Days")
axes[0].set_ylabel("Reaction time (ms)")
axes[0].legend(fontsize=8)
plt.show()
''', title="참가자별 직선과 수축")
save("lab10_blup", c)
BLUP_CELL = c.n

# ------------------------------------------------------------------ d. 일반화 추정 방정식
c = nb.cell('''
print(epil.groupby("trt", observed=True)["subject"].nunique())
tab = epil.groupby(["trt", "period"], observed=True)["y"].mean()
tab.unstack().round(2)
''', title="군별·기간별 평균 발작 건수")
save("lab10_epil_desc", c, marks={"placebo      28": 1})
NOTES[c.n] = "## d. 일반화 추정 방정식"

c = nb.cell('''
import statsmodels.api as sm

form = "y ~ trt + np.log(base / 4) + np.log(age) + C(period)"
gee_ex = smf.gee(form, groups="subject", data=epil,
                 family=sm.families.Poisson(),
                 cov_struct=sm.cov_struct.Exchangeable()).fit()
print(gee_ex.summary())
print(gee_ex.model.cov_struct.summary())
''', title="GEE (포아송, 교환가능 작업상관)")
save("lab10_gee", c, marks={r"~No\. clusters:\s+59": 1, "Exchangeable": 2,
                            r"~Scale:\s+1\.000": 3, "robust": 4,
                            r"~trt\[T\.progabide\][^\n]*": 5, r"~np\.log\(base / 4\)\s+1\.2311[^\n]*": 7, r"~C\(period\)\[T\.4\][^\n]*": 6, "0.400": 8})
GEE_CELL = c.n

c = nb.cell('''
ci = gee_ex.conf_int()                 # 로그 척도의 95% CI
rr = pd.DataFrame({"RR": np.exp(gee_ex.params),
                   "2.5%": np.exp(ci[0]),
                   "97.5%": np.exp(ci[1])})
rr.round(3)
''', title="발생률비로 바꾸기")
save("lab10_gee_rr", c)

c = nb.cell('''
gee_in = smf.gee(form, groups="subject", data=epil,
                 family=sm.families.Poisson(),
                 cov_struct=sm.cov_struct.Independence()).fit()
glm = smf.glm(form, data=epil, family=sm.families.Poisson()).fit()
print("GLM Pearson chi2/df:", round(glm.pearson_chi2 / glm.df_resid, 2))

t = "trt[T.progabide]"
fits = {"Poisson GLM": glm, "GEE independence": gee_in,
        "GEE exchangeable": gee_ex}
pd.DataFrame({"coef": [f.params[t] for f in fits.values()],
              "SE": [f.bse[t] for f in fits.values()]},
             index=fits.keys()).round(4)
''', title="작업상관과 표준오차 비교")
save("lab10_gee_cmp", c, marks={"4.76": 1})

# ------------------------------------------------------------------ e. 과제 (정답 셀)
c = nb.cell('''
slopes = pd.Series({sid: m2.fe_params["Days"] + re["Days"]
                    for sid, re in m2.random_effects.items()})
print(slopes.sort_values().round(1).to_string())   # 참가자별 기울기
print("5 ms/일보다 작은 참가자:", (slopes < 5).sum(), "명")
''', title="과제 1 정답. 수면 제한 하루당 기울기 읽기")
save("lab10_hw1", c, marks={"335    -0.3": 1, "308    19.7": 2, "5 ms/일보다 작은 참가자: 2": 3})
HW1_CELL = c.n
NOTES[c.n] = (f"## e. 과제 정답\n\n**과제 1.** 셀 {RS_CELL}의 무작위 절편 + 기울기 모형에서 Days 행을 Results 문장으로 옮기고, "
              "이 숫자가 말해 주지 않는 것을 두 가지 이상 쓰세요. 혼합모형이 예측한 참가자별 기울기도 모두 구해 보세요.")

c = nb.cell('''
form0 = "y ~ trt + np.log(age) + C(period)"     # 기저 발작 건수를 뺀 모형
gee0 = smf.gee(form0, groups="subject", data=epil,
               family=sm.families.Poisson(),
               cov_struct=sm.cov_struct.Exchangeable()).fit()
for name, g in [("with baseline", gee_ex), ("without baseline", gee0)]:
    lo, hi = np.exp(g.conf_int().loc[t])
    print(f"{name:16}: RR {np.exp(g.params[t]):.3f} ({lo:.3f}-{hi:.3f})",
          f" SE {g.bse[t]:.3f}  alpha {g.model.cov_struct.dep_params:.3f}")
''', title="과제 2 정답. 기저 발작 건수를 보정하지 않으면")
save("lab10_hw2", c, marks={"RR 0.989 (0.679-1.442)": 1, "~with baseline[^\n]*SE 0\\.192": 2,
                            "RR 0.930 (0.480-1.804)": 3, "~without baseline[^\n]*SE 0\\.338": 4,
                            "alpha 0.791": 5})
HW2_CELL = c.n
NOTES[c.n] = (f"**과제 2.** 셀 {GEE_CELL}의 GEE에서 기저 발작 건수(np.log(base / 4))를 빼고 다시 적합하세요. progabide의 "
              "발생률비, 표준오차, 작업상관이 어떻게 바뀌는지 원래 모형과 비교하고 왜 그런지 설명하세요.")

c = nb.cell('''
m_cat = smf.mixedlm("Reaction ~ C(Days)", data=sleep,
                    groups=sleep["Subject"]).fit()       # 동료의 모형(REML)
print("REML로 비교한 LR:", round(2 * (m_cat.llf - m1.llf), 2))

m_cat_ml = smf.mixedlm("Reaction ~ C(Days)", data=sleep,
                       groups=sleep["Subject"]).fit(reml=False)
m1_ml = smf.mixedlm("Reaction ~ Days", data=sleep,
                    groups=sleep["Subject"]).fit(reml=False)
lr_ml = 2 * (m_cat_ml.llf - m1_ml.llf)
print("ML로 비교한 LR  :", round(lr_ml, 2),
      " p (자유도 8):", round(stats.chi2.sf(lr_ml, 8), 3))
print("AIC (ML): linear", round(m1_ml.aic, 1),
      " categorical", round(m_cat_ml.aic, 1))
wt = m_cat.wald_test_terms(scalar=True).table   # 계수 9개를 함께 본 Wald 검정
print(f"Wald chi2 / 9 = {wt.loc['C(Days)', 'statistic'] / 9:.2f}")
''', title="과제 3 정답. 시간을 범주형으로 넣은 모형과 비교하기")
save("lab10_hw3", c, marks={"REML로 비교한 LR: 56.97": 1, "ML로 비교한 LR  : 3.74": 2, "p (자유도 8): 0.879": 3,
                            "AIC (ML): linear 1802.1": 4, "categorical 1814.3": 5, "Wald chi2 / 9 = 18.70": 6})
HW3_CELL = c.n
NOTES[c.n] = (f"**과제 3.** 동료가 Days를 범주형으로 넣은 무작위 절편 모형을 기본 설정(REML)으로 적합하고, 셀 {RI_CELL}의 "
              "직선 모형과 로그우도를 비교해 \"LR = 56.97, 자유도 8, P < 0.001이므로 반응시간은 직선으로 늘지 않는다\"라고 "
              "썼습니다. 동료의 계산을 재현하고, 무엇이 왜 틀렸는지 찾아 바른 비교를 해 보세요.")

if DEBUG:
    print("\ncells:", len(nb.cells))
print("lab10: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 10. 반복측정 자료 분석",
    intro=("사회약학 연구방법 노트의 '실습 10. 반복측정 자료 분석'에 딸린 노트북입니다. 앞부분(a–c)은 lme4 패키지의 "
           "sleepstudy 자료(수면 제한 연구 참가자 18명의 10일 반응시간)로 자료구조를 바꾸고 반복측정 분산분석과 선형 "
           "혼합모형을 적합합니다. 뒷부분(d)은 MASS 패키지의 epil 자료(뇌전증 임상시험 59명의 2주 발작 건수 네 번)로 "
           "포아송 GEE를 적합합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 10에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats as st

    ns = nb.ns
    sleep_, epil_, m1_, m2_, gee_ = ns["sleep"], ns["epil"], ns["m1"], ns["m2"], ns["gee_ex"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r1 = lambda x: f"{x:.1f}"
    r2 = lambda x: f"{x:.2f}"
    r3 = lambda x: f"{x:.3f}"
    # 글(content/lab10.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 6: "스파게티 그림", 8: "반복측정 분산분석 (AnovaRM)",
             9: "Greenhouse–Geisser ε 직접 계산", 10: "Mauchly 검정과 Huynh–Feldt ε", 11: "한 칸이 빠지면",
             12: "무작위 절편 모형", 14: "무작위 절편 + 기울기 모형", 15: "우도비 검정, REML과 ML",
             17: "참가자별 직선과 수축", 19: "GEE (포아송, 교환가능 작업상관)", 20: "발생률비로 바꾸기",
             21: "작업상관과 표준오차 비교"}
    for k_, t_ in cited.items():
        same(f"cell {k_} title", nb.cells[k_ - 1].title, t_)
    same("number of cells", len(nb.cells), 24)
    # 대응표 (이 실습의 자료 열)
    same("sleep rows, subjects", (len(sleep_), sleep_["Subject"].nunique()), (180, 18))
    same("epil rows, patients", (len(epil_), epil_["subject"].nunique()), (236, 59))
    arm = epil_.drop_duplicates("subject")["trt"].value_counts()
    same("progabide, placebo", (int(arm["progabide"]), int(arm["placebo"])), (31, 28))
    same("F, GG df", (r2(ns["F"]), r2(ns["eps"] * 9), r2(ns["eps"] * 153)), ("18.70", "3.32", "56.46"))
    ci2 = m2_.conf_int().loc["Days"]
    same("slope (CI)", (r1(m2_.fe_params["Days"]), r1(ci2[0]), r1(ci2[1])), ("10.5", "7.4", "13.5"))
    rr_ = ns["rr"].loc["trt[T.progabide]"]
    same("rate ratio (CI)", (r2(rr_["RR"]), r2(rr_["2.5%"]), r2(rr_["97.5%"])), ("0.99", "0.68", "1.44"))
    # a 절 스파게티 그림 문단
    dm = sleep_.groupby("Days")["Reaction"].mean()
    same("day 0, 9 means", (r1(dm[0]), r1(dm[9])), ("256.7", "350.9"))
    w_ = ns["wide"]
    same("day 0 range", (f"{w_[0].min():.0f}", f"{w_[0].max():.0f}"), ("199", "322"))
    own = sleep_.groupby("Subject").apply(lambda g: np.polyfit(g["Days"], g["Reaction"], 1)[0])
    same("own slope range", (r1(own.min()), r1(own.max())), ("-2.9", "21.8"))
    # b 절: p 세 개, Mauchly·HF, 직접 바꿔 보기
    same("p unadjusted, GG, lower", (f"{st.f.sf(ns['F'], 9, 153):.1e}", f"{st.f.sf(ns['F'], ns['eps'] * 9, ns['eps'] * 153):.1e}",
                                      f"{st.f.sf(ns['F'], 1, 17):.5f}"), ("9.0e-21", "5.5e-09", "0.00046"))
    same("Mauchly p, HF p", (f"{st.chi2.sf(ns['chi2'], 44):.1e}", f"{st.f.sf(ns['F'], ns['hf'] * 9, ns['hf'] * 153):.1e}"),
         ("1.9e-08", "7.1e-11"))
    same("HF df", (r2(ns["hf"] * 9), r2(ns["hf"] * 153)), ("4.22", "71.81"))
    # 군이 하나이면 Lecoutre의 수정식 ((N - g + 1) 사용)이 원래 식과 같음
    N_, g_, k_, e_ = 18, 1, 10, ns["eps"]
    hf_lec = ((N_ - g_ + 1) * (k_ - 1) * e_ - 2) / ((k_ - 1) * (N_ - g_ - (k_ - 1) * e_))
    same("HF = Lecoutre when one group", r3(hf_lec), r3(ns["hf"]))
    from statsmodels.stats.anova import AnovaRM
    s2 = sleep_[sleep_["Days"] >= 2]
    a2 = AnovaRM(s2, depvar="Reaction", subject="Subject", within=["Days"]).fit().anova_table.loc["Days"]
    same("AnovaRM days 2-9", (r2(a2["F Value"]), int(a2["Num DF"]), int(a2["Den DF"])), ("15.31", 7, 119))
    # c 절
    same("ICC parts", (r1(m1_.cov_re.iloc[0, 0]), r1(m1_.scale)), ("1378.2", "960.5"))
    same("residual SD (RI, RS)", (r1(np.sqrt(m1_.scale)), r1(np.sqrt(m2_.scale))), ("31.0", "25.6"))
    same("tau SD", r1(np.sqrt(m1_.cov_re.iloc[0, 0])), "37.1")
    same("RI slope z, CI", (r1(m1_.fe_params["Days"] / m1_.bse_fe["Days"]),
                            r2(m1_.conf_int().loc["Days", 0]), r2(m1_.conf_int().loc["Days", 1])), ("13.0", "8.89", "12.04"))
    sd0, sd1 = np.sqrt(m2_.cov_re.iloc[0, 0]), np.sqrt(m2_.cov_re.iloc[1, 1])
    same("RS SDs, corr", (r1(sd0), r2(sd0), r3(sd1), r2(m2_.cov_re.iloc[0, 1] / (sd0 * sd1))), ("24.7", "24.74", "5.922", "0.07"))
    b_ = m2_.fe_params["Days"]
    same("95% of slopes", (r1(b_ - 1.96 * sd1), r1(b_ + 1.96 * sd1)), ("-1.1", "22.1"))
    tq = st.t.ppf(0.975, 17)
    same("Satterthwaite-like t(17) CI", (r1(b_ - tq * m2_.bse_fe["Days"]), r1(b_ + tq * m2_.bse_fe["Days"])), ("7.2", "13.7"))
    same("LR p", f"{st.chi2.sf(ns['lr'], 2):.1e}", "5.0e-10")
    m2ml = ns["m2_ml"]
    same("REML vs ML var", (r1(m2_.cov_re.iloc[0, 0]), r1(m2_.cov_re.iloc[1, 1]), r1(m2ml.cov_re.iloc[0, 0]),
                            r1(m2ml.cov_re.iloc[1, 1])), ("612.1", "35.1", "565.5", "32.7"))
    se_ = ns["se"]
    same("four SEs", [r3(x) for x in se_.values], ["1.238", "1.550", "0.804", "1.546"])
    fe_, reff = m2_.fe_params, m2_.random_effects
    blup = {s: fe_["Days"] + reff[s]["Days"] for s in (308, 335, 350)}
    same("own -> mixed slopes", [(r1(own[s]), r1(blup[s])) for s in (308, 335, 350)],
         [("21.8", "19.7"), ("-2.9", "-0.3"), ("19.5", "17.1")])
    same("fixed effects", (r1(fe_["Intercept"]), r2(fe_["Days"])), ("251.4", "10.47"))
    # d 절
    same("base range", (int(epil_["base"].min()), int(epil_["base"].max())), (6, 151))
    same("age doubling", r2(2 ** gee_.params["np.log(age)"]), "1.53")
    same("base doubling", r2(2 ** gee_.params["np.log(base / 4)"]), "2.35")
    same("period 4 RR", r2(np.exp(gee_.params["C(period)[T.4]"])), "0.82")
    same("intercept RR", r3(np.exp(gee_.params["Intercept"])), "0.089")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e21 = epil_.copy()
        same("row 31 y", int(e21.iloc[30]["y"]), 23)
        e21.iloc[30, e21.columns.get_loc("y")] = 21
        g21 = smf.gee(ns["form"], groups="subject", data=e21, family=sm.families.Poisson(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        same("progabide coef with y = 21", f"{g21.params['trt[T.progabide]']:.4f}", "-0.0099")
        e49 = epil_[epil_["subject"] != 49]
        g49 = smf.gee(ns["form"], groups="subject", data=e49, family=sm.families.Poisson(),
                      cov_struct=sm.cov_struct.Exchangeable()).fit()
        lo49, hi49 = np.exp(g49.conf_int().loc["trt[T.progabide]"])
        same("without patient 49", (r2(np.exp(g49.params["trt[T.progabide]"])), r2(lo49), r2(hi49),
                                    r2(g49.model.cov_struct.dep_params)), ("0.80", "0.58", "1.09", "0.34"))
    p49 = epil_[epil_["subject"] == 49]
    same("patient 49", (int(p49["base"].iloc[0]), int(p49["y"].min()), int(p49["y"].max())), (151, 63, 102))
    cmp_ = ns["gee_in"]
    same("GLM vs GEE SE ratio", f"{cmp_.bse['trt[T.progabide]'] / ns['glm'].bse['trt[T.progabide]']:.1f}", "4.0")
    # 과제 1
    sl = ns["slopes"]
    same("model slopes range", (r1(sl.min()), r1(sl.max())), ("-0.3", "19.7"))
    # 과제 2
    g0 = ns["gee0"]
    tt = "trt[T.progabide]"
    same("SE ratio without/with baseline", r2(g0.bse[tt] / gee_.bse[tt]), "1.76")
    bm = epil_.drop_duplicates("subject").groupby("trt", observed=True)["base"].agg(["mean", "median"]) / 4
    same("baseline 2-week mean, median by arm", [r2(x) for x in bm.values.ravel()], ["7.70", "4.75", "7.90", "6.00"])
    # 과제 3
    same("REML LR", r2(2 * (ns["m_cat"].llf - m1_.llf)), "56.97")
    same("ML LR, p", (r2(ns["lr_ml"]), r3(st.chi2.sf(ns["lr_ml"], 8))), ("3.74", "0.879"))
    same("Wald chi2", r2(ns["wt"].loc["C(Days)", "statistic"]), "168.32")
    same("REML llf", (r2(ns["m_cat"].llf), r2(m1_.llf)), ("-864.75", "-893.23"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mc_rs = smf.mixedlm("Reaction ~ C(Days)", data=sleep_, groups=sleep_["Subject"], re_formula="~Days").fit(reml=False)
        ml_rs = smf.mixedlm("Reaction ~ Days", data=sleep_, groups=sleep_["Subject"], re_formula="~Days").fit(reml=False)
    lr_rs = 2 * (mc_rs.llf - ml_rs.llf)
    same("ML LR with random slopes", (r2(lr_rs), r2(st.chi2.sf(lr_rs, 8))), ("5.50", "0.70"))
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab10.html passed")
