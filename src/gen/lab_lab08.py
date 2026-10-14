"""실습 8 (lab08) — 신뢰구간, 효과크기, 교란: birthwt (흡연 → 저체중아, 인종 = 교란변수).

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab08.py
      then  python3 tools/nbcheck.py lab08   (runs the notebook alone in a fresh process and compares outputs)
All outputs are produced by actually running the cells (labkit).
Set LABDEBUG=1 to print every cell's text output to the console.
Writes figs/lab08_*.html, pub/notebooks/lab08.ipynb and checks/lab08.expected.json.
The VERIFY block at the end asserts the numbers quoted in content/lab08.html that are not marked in a cell output.
(2026-10-14: version cell 1, 과제 1–3 (cells 29–31) and the notebook added; old cells 1–27 became 2–28.)
"""
import html as _h
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab08")
DEBUG = os.environ.get("LABDEBUG")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


def H(c, marks=None):
    """nb.html with each mark placed after the first occurrence anywhere in the
    output block (printed text, warnings or DataFrame table), skipping images.
    Keys starting with "<" match raw table HTML (e.g. "<td>0.42</td>") and the
    mark goes inside the cell."""
    if DEBUG:
        print(f"----- cell {c.n}: {c.title}")
        print(c.stdout, "" if c.value_repr is None else c.value_repr)
        for w in c.warns:
            print("WARN:", w)
    s = nb.html(c)
    if not marks or os.environ.get("LABNOMARK"):
        return s
    i = s.index('<div class="cell-out">')
    j = s.find("<img", i)
    j = len(s) if j < 0 else j
    head, body, tail = s[:i], s[i:j], s[j:]
    for sub, n in marks.items():
        raw = sub.startswith("<")
        e = sub if raw else _h.escape(sub)
        k = body.find(e)
        if k < 0:
            raise ValueError(f"mark text not found in output of cell {c.n}: {sub!r}")
        k += e.rfind("</") if raw else len(e)
        body = body[:k] + f'<span class="mk">{n}</span>' + body[k:]
    return head + body + tail


def S(name, c, marks=None):
    nb.save_fragment(name, H(c, marks))


# ================================================================== a. 신뢰구간과 효과크기
c = nb.cell('''
import numpy as np
import pandas as pd
from scipy import stats
import scipy, statsmodels, matplotlib

print("pandas     ", pd.__version__)
print("numpy      ", np.__version__)
print("scipy      ", scipy.__version__)
print("statsmodels", statsmodels.__version__)
print("matplotlib ", matplotlib.__version__)
''', title="패키지 불러오기와 버전 확인")
S("lab08_versions", c, {"statsmodels 0.15.0": 1})
NOTES[c.n] = ("## a. 신뢰구간과 효과크기\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 birthwt 자료를 읽습니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/MASS/birthwt.csv")
df = pd.read_csv(url)
df["ptl_any"] = (df["ptl"] > 0).astype(int)  # 이전 조산 1회 이상
df["lwt_kg"] = df["lwt"] * 0.4536            # 파운드 → kg
df["race_f"] = pd.Categorical(
    df["race"].map({1: "White", 2: "Black", 3: "Other"}),
    categories=["White", "Black", "Other"])
print(df.shape)
df[["low", "smoke", "race", "age", "lwt_kg", "ptl_any", "ht"]].head(3)
''', title="데이터 불러오기와 변수 만들기")
S("lab08_load", c, {"(189, 14)": 1, "<td>82.5552</td>": 2})

c = nb.cell('''
from statsmodels.stats.proportion import proportion_confint

rows = []
for count, n in [(59, 189), (7, 12)]:
    for method in ["normal", "wilson", "beta"]:
        lo, hi = proportion_confint(count, n, method=method)
        rows.append([f"{count}/{n}", method, count / n, lo, hi])
cols = ["data", "method", "p", "lower", "upper"]
pd.DataFrame(rows, columns=cols).round(3)
''', title="비율의 95% 신뢰구간")
S("lab08_propci", c, {"<td>0.246</td>": 1, "<td>0.250</td>": 2, "<td>0.304</td>": 3,
                      "<td>0.862</td>": 4, "<td>0.807</td>": 5, "<td>0.277</td>": 6})

c = nb.cell('''
bwt = df["bwt"]
n = len(bwt)
mean, sd = bwt.mean(), bwt.std()   # std()는 n - 1로 나눈 표준편차
se = sd / np.sqrt(n)
t_crit = stats.t.ppf(0.975, n - 1)  # 자유도 188인 t 분포의 97.5% 점
print(f"mean = {mean:.1f}, SD = {sd:.1f}, SE = {se:.2f}")
print(f"t = {t_crit:.3f}, half-width = {t_crit * se:.1f}")
print(f"95% CI = {mean - t_crit * se:.1f} to {mean + t_crit * se:.1f}")
''', title="평균의 95% 신뢰구간")
S("lab08_meanci", c, {"mean = 2944.6": 1, "SE = 53.04": 2, "t = 1.973": 3,
                      "95% CI = 2840.0 to 3049.2": 4})

c = nb.cell('''
tab = pd.crosstab(df["smoke"], df["low"]).loc[[1, 0], [1, 0]]
print(tab)
(a, b), (c, d) = tab.to_numpy()
p1, p0 = a / (a + b), c / (c + d)
print(f"risk: smoker = {p1:.3f}, nonsmoker = {p0:.3f}")
''', title="흡연과 저체중아의 2×2 표")
S("lab08_tab", c, {"risk: smoker = 0.405": 1, "nonsmoker = 0.252": 2})

c = nb.cell('''
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.proportion import confint_proportions_2indep

def show(name, est, ci):
    print(f"{name:3s} = {est:6.3f} (95% CI {ci[0]:.3f} to {ci[1]:.3f})")

t22 = Table2x2(tab.to_numpy())
rd_ci = confint_proportions_2indep(a, a + b, c, c + d,
                                   compare="diff", method="wald")
show("RD", p1 - p0, rd_ci)
show("RR", t22.riskratio, t22.riskratio_confint())
show("OR", t22.oddsratio, t22.oddsratio_confint())
show("NNH", 1 / (p1 - p0), (1 / rd_ci[1], 1 / rd_ci[0]))
''', title="위험차, 상대위험도, 오즈비, NNH")
S("lab08_effects", c, {"RD  =  0.153": 1, "RR  =  1.608 (95% CI 1.058 to 2.443)": 2,
                       "OR  =  2.022 (95% CI 1.081 to 3.783)": 3, "NNH =  6.526": 4,
                       "3.444 to 62.221": 5})
EFFECTS_CELL = c.n

c = nb.cell('''
rng = np.random.default_rng(2026)   # 시드를 고정해 같은 결과 재현
y = df["low"].to_numpy()
x = df["smoke"].to_numpy()
boot_rr = []
for _ in range(2000):
    i = rng.integers(0, len(y), size=len(y))   # 189명 복원추출
    yb, xb = y[i], x[i]
    boot_rr.append(yb[xb == 1].mean() / yb[xb == 0].mean())
boot_rr = np.array(boot_rr)
lo, hi = np.percentile(boot_rr, [2.5, 97.5])
print(f"bootstrap RR: median {np.median(boot_rr):.3f}, "
      f"95% CI {lo:.3f} to {hi:.3f}")
''', title="부트스트랩 신뢰구간")
S("lab08_boot", c, {"median 1.612": 1, "95% CI 1.059 to 2.500": 2})

c = nb.cell('''
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 3))
plt.hist(boot_rr, bins=40, color="lightsteelblue", edgecolor="white")
for v in (lo, hi):
    plt.axvline(v, ls="--", color="k")          # 백분위수 구간
plt.axvline(t22.riskratio, color="tab:orange")  # 표본의 RR
plt.xlabel("Risk ratio in bootstrap samples")
plt.ylabel("Count")
plt.show()
''', title="부트스트랩 분포 그래프")
S("lab08_bootplot", c)

# ================================================================== b. 일반화 선형모형과 우도비 검정
c = nb.cell('''
import statsmodels.api as sm
import statsmodels.formula.api as smf

fam = sm.families
m_logit = smf.glm("low ~ smoke", data=df,
                  family=fam.Binomial()).fit()
print(m_logit.summary())
''', title="로지스틱 회귀를 GLM으로 적합하기")
NOTES[c.n] = "## b. 일반화 선형모형과 우도비 검정"
S("lab08_glm", c, {"Binomial": 1, "Logit": 2, "-114.90": 3, "229.80": 4,
                   "0.7041      0.320      2.203      0.028": 5})

c = nb.cell('''
or_tab = np.exp(m_logit.conf_int())
or_tab.columns = ["2.5%", "97.5%"]
or_tab.insert(0, "OR", np.exp(m_logit.params))
or_tab.round(3)
''', title="계수를 오즈비로 바꾸기")
S("lab08_glmor", c, {"<td>2.022</td>": 1, "<td>0.337</td>": 2})

c = nb.cell('''
f = "low ~ smoke"
fits = {
    "logit": smf.glm(f, df, family=fam.Binomial()).fit(),
    "log-binomial": smf.glm(
        f, df, family=fam.Binomial(fam.links.Log())).fit(),
    "Poisson": smf.glm(f, df, family=fam.Poisson()).fit(),
    "Poisson robust": smf.glm(
        f, df, family=fam.Poisson()).fit(cov_type="HC0"),
}
''', title="연결함수와 분포가 다른 네 모형")
S("lab08_fits", c, {"DomainWarning": 1})

c = nb.cell('''
rows = []
for name, r in fits.items():
    lo, hi = np.exp(r.conf_int().loc["smoke"])
    rows.append([name, np.exp(r.params["smoke"]), r.bse["smoke"],
                 lo, hi])
cols = ["model", "exp(b)", "SE(b)", "2.5%", "97.5%"]
pd.DataFrame(rows, columns=cols).round(3)
''', title="흡연의 효과를 네 모형으로 비교")
S("lab08_linkcomp", c, {"<td>2.022</td>": 1, "<td>log-binomial</td>": 2, "<td>0.260</td>": 3,
                        "<td>Poisson robust</td>": 4})

c = nb.cell('''
m0 = smf.glm("low ~ 1", df, family=fam.Binomial()).fit()  # 절편만
for name, r in [("null", m0), ("smoke", m_logit)]:
    print(f"{name:5s}: llf = {r.llf:.3f}, deviance = {r.deviance:.3f},"
          f" df_model = {r.df_model:.0f}, AIC = {r.aic:.3f}")
''', title="로그우도, 이탈도, AIC")
S("lab08_llf", c, {"llf = -117.336": 1, "deviance = 234.672": 2, "llf = -114.902": 3,
                   "deviance = 229.805": 4, "AIC = 233.805": 5})

c = nb.cell('''
lr = 2 * (m_logit.llf - m0.llf)
print(f"LR    chi2 = {lr:.3f}, P = {stats.chi2.sf(lr, 1):.4f}")
z = m_logit.tvalues["smoke"]
print(f"Wald  chi2 = {z**2:.3f}, P = {m_logit.pvalues['smoke']:.4f}")
sc = stats.chi2_contingency(tab, correction=False)
print(f"Score chi2 = {sc.statistic:.3f}, P = {sc.pvalue:.4f}")
''', title="우도비 검정을 직접 계산하기")
S("lab08_lrt", c, {"LR    chi2 = 4.867": 1, "Wald  chi2 = 4.852": 2, "Score chi2 = 4.924": 3})

c = nb.cell('''
f_adj = "low ~ smoke + C(race) + age + lwt_kg + ptl_any + ht"
lb = smf.glm(f_adj, df, family=fam.Binomial(fam.links.Log())).fit()
print("converged     :", lb.converged)
print("log-likelihood:", lb.llf)
print("max fitted p  :", lb.fittedvalues.max())
''', title="보정 모형에서 로그-이항 모형이 실패할 때")
S("lab08_lbfail", c, {"converged     : True": 1, "log-likelihood: nan": 2, "e+74": 3, "SingularMatrixWarning": 4})

c = nb.cell('''
mp = smf.glm(f_adj, df, family=fam.Poisson()).fit(cov_type="HC0")
ml = smf.glm(f_adj, df, family=fam.Binomial()).fit()
for name, r in [("aRR, modified Poisson", mp), ("aOR, logistic", ml)]:
    lo, hi = np.exp(r.conf_int().loc["smoke"])
    est = np.exp(r.params["smoke"])
    print(f"{name:22s}: {est:.2f} (95% CI {lo:.2f} to {hi:.2f})")
print("max fitted p (Poisson):", round(mp.fittedvalues.max(), 3))
''', title="수정 포아송 회귀로 보정 상대위험도 구하기")
S("lab08_modpois", c, {"aRR, modified Poisson : 1.73 (95% CI 1.14 to 2.64)": 1,
                       "aOR, logistic         : 2.36": 2, "1.297": 3})

# ================================================================== c. 교란과 층화 분석
c = nb.cell('''
g = df.groupby("race_f", observed=True)
g_non = df[df["smoke"] == 0].groupby("race_f", observed=True)
pd.DataFrame({
    "n": g.size(),
    "smoker %": g["smoke"].mean() * 100,
    "LBW % (all)": g["low"].mean() * 100,
    "LBW % (nonsmokers)": g_non["low"].mean() * 100,
}).round(1)
''', title="인종은 교란변수의 조건을 갖추었나")
NOTES[c.n] = "## c. 교란과 층화 분석"
S("lab08_confcheck", c, {"<td>54.2</td>": 1, "<td>17.9</td>": 2, "<td>24.0</td>": 3,
                         "<td>9.1</td>": 4, "<td>36.4</td>": 5})

c = nb.cell('''
tables = []
for race, sub in df.groupby("race_f", observed=True):
    t = pd.crosstab(sub["smoke"], sub["low"]).loc[[1, 0], [1, 0]]
    tables.append(t.to_numpy())
    t2 = Table2x2(t.to_numpy())
    lo, hi = t2.oddsratio_confint()
    print(f"{race:5} {t.to_numpy().tolist()}  "
          f"OR = {t2.oddsratio:.2f} ({lo:.2f} to {hi:.2f})")
''', title="인종별 2×2 표와 층별 오즈비")
S("lab08_strata", c, {"[[19, 33], [4, 40]]": 1, "OR = 5.76 (1.78 to 18.60)": 2,
                      "OR = 3.30 (0.63 to 17.16)": 3, "OR = 1.25 (0.35 to 4.46)": 4})
STRATA_CELL = c.n

c = nb.cell('''
from statsmodels.stats.contingency_tables import StratifiedTable

st = StratifiedTable(tables)   # 2×2 표 3개의 목록
print(st.summary())
''', title="Mantel–Haenszel 통합 오즈비와 두 가지 검정")
S("lab08_mh", c, {"3.086   1.491 6.390": 1, "2.151": 2, "9.413   0.002": 3, "3.126   0.210": 4})

c = nb.cell('''
m_race = smf.logit("low ~ smoke + C(race)", df).fit(disp=0)
b = m_race.params["smoke"]
b_lo, b_hi = m_race.conf_int().loc["smoke"]
rows = [["crude", t22.oddsratio, *t22.oddsratio_confint()],
        ["Mantel-Haenszel", st.oddsratio_pooled,
         *st.oddsratio_pooled_confint()],
        ["logistic + C(race)", np.exp(b), np.exp(b_lo), np.exp(b_hi)]]
pd.DataFrame(rows, columns=["method", "OR", "2.5%", "97.5%"]).round(2)
''', title="보정 전과 보정 후 비교")
S("lab08_crudeadj", c, {"<td>2.02</td>": 1, "<td>3.09</td>": 2, "<td>3.05</td>": 3})
RACE_CELL = c.n      # m_race (smoke + C(race))를 만드는 셀

c = nb.cell('''
labels = ["White", "Black", "Other", "Crude", "Mantel-Haenszel"]
t_all = [Table2x2(t) for t in tables] + [t22]
est = [t.oddsratio for t in t_all] + [st.oddsratio_pooled]
ci = [t.oddsratio_confint() for t in t_all]
ci += [st.oddsratio_pooled_confint()]
err = np.array([[e - l, h - e] for e, (l, h) in zip(est, ci)]).T
ypos = np.arange(len(labels))[::-1]
plt.figure(figsize=(6, 2.8))
plt.errorbar(est, ypos, xerr=err, fmt="s", color="k", capsize=3)
plt.axvline(1, ls="--", color="gray")
plt.xscale("log")
plt.xticks([0.5, 1, 2, 5, 10, 20], [0.5, 1, 2, 5, 10, 20])
plt.minorticks_off()
plt.yticks(ypos, labels)
plt.xlabel("Odds ratio for smoking (log scale)")
plt.show()
''', title="층별 오즈비와 통합 오즈비 그래프")
S("lab08_forest", c)

# ================================================================== d. 가변수와 다변수 분석
c = nb.cell('''
m_num = smf.logit("low ~ smoke + race", df).fit(disp=0)
m_cat = smf.logit("low ~ smoke + C(race, Treatment(reference=1))",
                  df).fit(disp=0)
print(m_num.params.round(3), "\\n")
print(m_cat.params.round(3), "\\n")
print(f"llf: numeric race {m_num.llf:.3f}, dummies {m_cat.llf:.3f}")
''', title="인종을 숫자 그대로 넣으면")
NOTES[c.n] = "## d. 가변수와 다변수 분석"
S("lab08_numrace", c, {"race         0.559": 1, "[T.2]    1.084": 2, "[T.3]    1.109": 3,
                       "numeric race -110.670": 4, "dummies -109.987": 5})

c = nb.cell('''
m_ref3 = smf.logit("low ~ smoke + C(race, Treatment(reference=3))",
                   df).fit(disp=0)
print(np.exp(m_ref3.params).round(3), "\\n")
print(f"llf = {m_ref3.llf:.3f}")
''', title="기준범주 바꾸기")
S("lab08_ref3", c, {"[T.1]    0.330": 1, "[T.2]    0.976": 2, "3.053": 3,
                    "llf = -109.987": 4})

c = nb.cell('''
print(m_cat.wald_test_terms())
m_smoke = smf.logit("low ~ smoke", df).fit(disp=0)
lr = 2 * (m_cat.llf - m_smoke.llf)
print(f"\\nLR test for race: chi2 = {lr:.3f}, df = 2, "
      f"P = {stats.chi2.sf(lr, 2):.4f}")
''', title="인종 전체의 P: 결합 검정")
S("lab08_joint", c, {"9.112888": 1, "chi2 = 9.830": 2})

c = nb.cell('''
def or_ci(model):
    t = np.exp(model.conf_int())
    t.columns = ["lo", "hi"]
    t.insert(0, "OR", np.exp(model.params))
    return t.drop(index="Intercept")

terms = ["smoke", "C(race)", "age", "lwt_kg", "ptl_any", "ht"]
uni = pd.concat([or_ci(smf.logit(f"low ~ {t}", df).fit(disp=0))
                 for t in terms])
uni.round(2)
''', title="단변수 오즈비: 변수마다 따로 적합")
S("lab08_uni", c, {"<td>2.02</td>": 1, "<td>0.97</td>": 2})

c = nb.cell('''
full = smf.logit("low ~ " + " + ".join(terms), df).fit(disp=0)
multi = or_ci(full).loc[uni.index]      # 단변수 표와 같은 행 순서
fmt = lambda t: [f"{r.OR:.2f} ({r.lo:.2f}-{r.hi:.2f})"
                 for r in t.itertuples()]
table2 = pd.DataFrame({"Univariable OR (95% CI)": fmt(uni),
                       "Multivariable OR (95% CI)": fmt(multi)},
                      index=uni.index)
print("n =", int(full.nobs), " events =", int(df["low"].sum()),
      " coefficients =", len(full.params) - 1)
table2
''', title="Table 2 만들기")
S("lab08_table2", c, {"events = 59": 1, "coefficients = 7": 2, "2.02 (1.08-3.78)": 3,
                      "2.36 (1.07-5.22)": 4})
FULL_CELL = c.n      # full (다변수 모형)을 만드는 셀

c = nb.cell('''
print(full.wald_test_terms())
''', title="다변수 모형의 변수별 결합 검정")
S("lab08_fullwald", c, {"5.991023": 1, "0.050011": 2, "0.033972": 3})

c = nb.cell('''
table2.index = ["Smoking during pregnancy",
                "Race: Black (vs White)", "Race: Other (vs White)",
                "Age, per year", "Weight before pregnancy, per kg",
                "Previous preterm labor", "History of hypertension"]
table2
''', title="행 이름을 논문용으로 바꾸기")
S("lab08_table2b", c)

# ================================================================== e. 과제 (정답 셀)
c = nb.cell('''
print(f"crude          : risk {p1:.3f} vs {p0:.3f}, "
      f"RD = {p1 - p0:.3f}, NNH = {1 / (p1 - p0):.1f}")
for name, model in [("race only", m_race), ("all covariates", full)]:
    r1 = model.predict(df.assign(smoke=1)).mean()   # 모두 흡연했다면
    r0 = model.predict(df.assign(smoke=0)).mean()   # 모두 비흡연이라면
    print(f"{name:15s}: risk {r1:.3f} vs {r0:.3f}, "
          f"RD = {r1 - r0:.3f}, NNH = {1 / (r1 - r0):.1f}")
''', title="과제 1 정답. NNH를 보정 모형으로 다시 구하기")
S("lab08_hw1", c, {"NNH = 6.5": 1, "race only      : risk 0.458 vs 0.229": 2, "NNH = 4.4": 3,
                   "NNH = 6.4": 4})
HW1_CELL = c.n
NOTES[c.n] = (f"## e. 과제 정답\n\n**과제 1.** 셀 {EFFECTS_CELL}의 NNH 행(6.526, 95% CI 3.444 to 62.221)을 한 문장으로 옮기고, "
              f"이 숫자가 말해 주지 않는 것을 두 가지 이상 쓰세요. 셀 {RACE_CELL}의 인종 보정 모형과 셀 {FULL_CELL}의 다변수 "
              "모형으로 표준화한 위험차와 NNH를 구해 비교하세요.")

c = nb.cell('''
rows = []
for cut in [2000, 2500, 3000]:
    y_cut = (df["bwt"] < cut).astype(int)            # 결과를 다시 정의
    t = pd.crosstab(df["smoke"], y_cut).loc[[1, 0], [1, 0]].to_numpy()
    t2 = Table2x2(t)
    r_s, r_n = t[0, 0] / t[0].sum(), t[1, 0] / t[1].sum()  # 흡연, 비흡연 위험
    rows.append([cut, y_cut.sum(), y_cut.mean(), r_s, r_n,
                 t2.riskratio, t2.oddsratio, t2.oddsratio / t2.riskratio])
cols = ["cut", "events", "prop", "risk_s", "risk_n", "RR", "OR", "OR/RR"]
pd.DataFrame(rows, columns=cols).round(3)
''', title="과제 2 정답. 저체중아의 기준을 바꾸면")
S("lab08_hw2", c, {"<td>0.101</td>": 1, "<td>1.130</td>": 2, "<td>1.014</td>": 3, "<td>0.513</td>": 4,
                   "<td>1.402</td>": 5, "<td>2.062</td>": 6, "<td>1.471</td>": 7})
HW2_CELL = c.n
NOTES[c.n] = (f"**과제 2.** 저체중아의 기준을 2,500 g에서 2,000 g과 3,000 g으로 바꿔 셀 {EFFECTS_CELL}처럼 흡연의 상대위험도와 "
              "오즈비를 다시 구하세요. 기준에 따라 두 지표의 차이가 왜 달라지는지 설명하세요.")

c = nb.cell('''
m_int = smf.logit("low ~ smoke * C(race)", df).fit(disp=0)  # 교호작용 항 포함
lr_int = 2 * (m_int.llf - m_race.llf)               # 교호작용 항이 없는 모형과 비교
print(f"P for interaction (LR, df = 2): chi2 = {lr_int:.2f}, "
      f"P = {stats.chi2.sf(lr_int, 2):.3f}")
print(np.exp(m_int.params).round(2), "\\n")
risk = df.pivot_table(index="race_f", columns="smoke", values="low",
                      aggfunc="mean", observed=True)
print((risk[1] - risk[0]).round(3))                  # 인종별 위험차
''', title="과제 3 정답. 인종에 따라 흡연의 효과가 다른가")
S("lab08_hw3", c, {"chi2 = 3.16, P = 0.206": 1, "smoke                 5.76": 2, "smoke:C(race)[T.2]    0.57": 3,
                   "smoke:C(race)[T.3]    0.22": 4, "White    0.274": 5, "Black    0.288": 6})
HW3_CELL = c.n
NOTES[c.n] = ("**과제 3.** 동료가 인종별 오즈비(셀 {0})를 보고 \"흡연은 백인 산모에서만 저체중아와 유의하게 관련되었고(OR 5.76, "
              "95% CI 1.78–18.60), 흑인과 기타 산모에서는 관련이 없었다. 흡연의 영향은 인종에 따라 다르다\"라고 썼습니다. "
              "이 결론에서 틀린 곳 두 가지를 찾고, 인종에 따라 효과가 다른지 직접 검정하세요.").format(STRATA_CELL)

print("lab08: cells =", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================== notebook
nb.save_ipynb(
    "실습 8. 신뢰구간, 효과크기, 교란",
    intro=("사회약학 연구방법 노트의 '실습 8. 신뢰구간, 효과크기, 교란'에 딸린 노트북입니다. MASS 패키지의 birthwt 자료"
           "(출생아와 산모 189명)로 '임신 중 흡연은 저체중아 출산과 관련이 있는가, 인종을 고려하면 어떻게 달라지는가'를 "
           "신뢰구간과 효과 지표, 일반화 선형모형과 우도비 검정, 층화 분석, 가변수와 다변수 모형으로 차례로 분석합니다. "
           "셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 8에 있습니다."),
    notes=NOTES)

# ================================================================== VERIFY: numbers quoted in the text (not marked in an output)
if __name__ == "__main__":
    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy import stats

    ns = nb.ns
    df = ns["df"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r1_ = lambda x: f"{x:.1f}"
    r2_ = lambda x: f"{x:.2f}"
    r3_ = lambda x: f"{x:.3f}"
    # 글에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 5: "흡연과 저체중아의 2×2 표", 6: "위험차, 상대위험도, 오즈비, NNH", 18: "인종별 2×2 표와 층별 오즈비",
             19: "Mantel–Haenszel 통합 오즈비와 두 가지 검정", 20: "보정 전과 보정 후 비교", 26: "Table 2 만들기"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    # 대응표
    same("N, smokers, nonsmokers", (len(df), int(df["smoke"].sum()), int((df["smoke"] == 0).sum())), (189, 74, 115))
    same("events", (int(df["low"].sum()), f"{df['low'].mean():.1%}"), (59, "31.2%"))
    t22, st, m_race, full, mp = ns["t22"], ns["st"], ns["m_race"], ns["full"], ns["mp"]
    same("crude OR, RR", (r2_(t22.oddsratio), r2_(t22.riskratio)), ("2.02", "1.61"))
    same("MH OR, adjusted OR, aRR", (r2_(st.oddsratio_pooled), r2_(np.exp(full.params["smoke"])),
                                     r2_(np.exp(mp.params["smoke"]))), ("3.09", "2.36", "1.73"))
    same("EPV", f"{59 / 7:.1f}", "8.4")
    # a 절
    same("first mother lwt", (int(df["lwt"].iloc[0]), r1_(df["lwt_kg"].iloc[0])), (182, "82.6"))
    same("12 mothers Wald half-width", (r3_(7 / 12), r3_(1.96 * np.sqrt(7 / 12 * 5 / 12 / 12))), ("0.583", "0.279"))
    same("Wald minus Wilson upper", r3_(0.862 - 0.807), "0.055")
    bw = ns["bwt"]
    same("95% of babies", (f"{bw.mean() - 1.96 * bw.std():.0f}", f"{bw.mean() + 1.96 * bw.std():.0f}"), ("1515", "4374"))
    ci1 = stats.ttest_1samp(bw, 0).confidence_interval()
    same("ttest_1samp CI", (r1_(ci1.low), r1_(ci1.high)), ("2840.0", "3049.2"))
    same("RR CI geometric mean", r3_(np.sqrt(1.058 * 2.443)), "1.608")
    y, x = ns["y"], ns["x"]
    los, his = [], []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        r = []
        for _ in range(2000):
            i = rng.integers(0, len(y), size=len(y))
            r.append(y[i][x[i] == 1].mean() / y[i][x[i] == 0].mean())
        lo, hi = np.percentile(r, [2.5, 97.5])
        los.append(lo)
        his.append(hi)
    lo2, hi2 = [round(v, 2) for v in los], [round(v, 2) for v in his]
    assert min(lo2) >= 1.04 and max(lo2) <= 1.07 and min(hi2) >= 2.49 and max(hi2) <= 2.54, (los, his)
    n_ok += 1
    # b 절
    a_, b_, c_, d_ = 30, 44, 29, 86
    same("SE of log OR from table", r3_(np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)), "0.320")
    same("nonsmoker odds", r3_(29 / 86), "0.337")
    same("max fitted p (log-binomial) leading digit", f"{ns['lb'].fittedvalues.max():.0e}"[0], "9")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fam = ns["fam"]
        lb2 = smf.glm(ns["f_adj"], df, family=fam.Binomial(fam.links.Log())).fit(start_params=mp.params)
    assert lb2.fittedvalues.max() > 0.999, lb2.fittedvalues.max()     # 경계(예측 확률 1)에 닿음
    n_ok += 1
    # c 절
    tabs = ns["tables"]
    num = [t[0, 0] * t[1, 1] / t.sum() for t in tabs]
    den = [t[0, 1] * t[1, 0] / t.sum() for t in tabs]
    same("MH by hand", (r2_(sum(num)), r2_(sum(den)), r2_(sum(num) / sum(den))), ("13.07", "4.23", "3.09"))
    cmh = st.test_null_odds(correction=True)
    same("CMH corrected", (r2_(cmh.statistic), r3_(cmh.pvalue)), ("8.38", "0.004"))
    bd = st.test_equal_odds(adjust=True)
    same("Breslow-Day Tarone", (r2_(bd.statistic), r2_(bd.pvalue)), ("3.13", "0.21"))
    same("MH RR", r2_(st.riskratio_pooled), "2.15")
    # d 절
    same("other vs white OR", r2_(np.exp(ns["m_cat"].params.iloc[2])), "3.03")
    same("lwt per 10 kg", (r2_(np.exp(10 * full.params["lwt_kg"])),
                           [r2_(v) for v in np.exp(10 * full.conf_int().loc["lwt_kg"])]), ("0.71", ["0.52", "0.96"]))
    same("hypertension mothers", int(df["ht"].sum()), 12)
    # 과제 1
    same("crude NNH", r1_(1 / (ns["p1"] - ns["p0"])), "6.5")
    # 과제 2: OR/RR = (1 - nonsmoker risk) / (1 - smoker risk)
    h2 = pd.DataFrame(ns["rows"], columns=ns["cols"])
    for _, r in h2.iterrows():
        assert abs(r["OR/RR"] - (1 - r["risk_n"]) / (1 - r["risk_s"])) < 1e-12
    n_ok += 1
    same("events by cut", h2["events"].tolist(), [19, 59, 97])
    from statsmodels.stats.contingency_tables import Table2x2
    y20 = (df["bwt"] < 2000).astype(int)
    t20 = Table2x2(pd.crosstab(df["smoke"], y20).loc[[1, 0], [1, 0]].to_numpy())
    same("RR CI at 2000 g", [r2_(v) for v in t20.riskratio_confint()], ["0.48", "2.68"])
    same("OR/RR at 3000 g from rounded risks", r2_((1 - 0.443) / (1 - 0.622)), "1.47")
    same("Black mothers", int((df["race"] == 2).sum()), 26)
    # 과제 3
    mi = ns["m_int"]
    e_ = np.exp(mi.params)
    same("stratum ORs from interaction model", (r2_(e_["smoke"]), r2_(e_["smoke"] * e_["smoke:C(race)[T.2]"]),
                                                r2_(e_["smoke"] * e_["smoke:C(race)[T.3]"])), ("5.76", "3.30", "1.25"))
    wt = mi.wald_test_terms().table
    same("interaction Wald", (r2_(float(wt.loc["smoke:C(race)", "statistic"])), r2_(float(wt.loc["smoke:C(race)", "pvalue"]))),
         ("3.02", "0.22"))
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab08.html passed")
