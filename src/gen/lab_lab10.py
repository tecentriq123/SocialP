"""실습 10 · 반복측정 자료 분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab10.py
      (LABDEBUG=1 prints every cell's text output, for writing marks)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab10")
DEBUG = os.environ.get("LABDEBUG")


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
    nb.save_fragment(name, h)


# ------------------------------------------------------------------ 가
c = nb.cell('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ------------------------------------------------------------------ 나
c = nb.cell('''
from statsmodels.stats.anova import AnovaRM

aov = AnovaRM(sleep, depvar="Reaction", subject="Subject",
              within=["Days"]).fit()
print(aov)
''', title="반복측정 분산분석 (AnovaRM)")
save("lab10_anovarm", c, marks={"18.7027": 1, "9.0000": 2, "153.0000": 3, r"~0\.0000(?=\n)": 4})

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

c = nb.cell('''
short = sleep.drop(index=0)      # 첫 행(308번 참가자 0일) 삭제
AnovaRM(short, depvar="Reaction", subject="Subject",
        within=["Days"]).fit()
''', title="한 칸이 빠지면", expect_error=True)
save("lab10_unbal", c)

# ------------------------------------------------------------------ 다
c = nb.cell('''
import statsmodels.formula.api as smf

m1 = smf.mixedlm("Reaction ~ Days", data=sleep,
                 groups=sleep["Subject"]).fit()
print(m1.summary())
''', title="무작위 절편 모형")
save("lab10_ri", c, marks={"REML": 1, "No. Groups:        18": 2, "960.4568": 3,
                           "-893.2325": 4, "Yes": 5, r"~Days\s+10\.467\s+0\.804[^\n]*": 6, r"~Group Var\s+1378\.176[^\n]*": 7})

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

# ------------------------------------------------------------------ 라
c = nb.cell('''
print(epil.groupby("trt", observed=True)["subject"].nunique())
tab = epil.groupby(["trt", "period"], observed=True)["y"].mean()
tab.unstack().round(2)
''', title="군별·기간별 평균 발작 건수")
save("lab10_epil_desc", c, marks={"placebo      28": 1})

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

if DEBUG:
    print("\ncells:", len(nb.cells))
