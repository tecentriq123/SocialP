"""실습 6 (상관분석과 선형회귀) — real cells run with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab06.py
Writes figs/lab06_*.html (insert in content/lab06.html with <!--FIG:lab06_xxx-->).
"""
import html as _html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402


def eol(frag, items, pre_index=0):
    """Append <span class="mk">n</span> at the END of the output line that contains `sub`.

    Keeps the fixed-width columns of statsmodels summaries aligned (inline marks would shift them).
    items: list of (substring, [n, ...]); searched only inside the pre_index-th <pre> of the output.
    """
    head, sep, out = frag.partition('<div class="cell-out">')
    starts = []
    i = 0
    while True:
        j = out.find("<pre>", i)
        if j < 0:
            break
        starts.append(j)
        i = j + 5
    a = starts[pre_index] + 5
    b = out.find("</pre>", a)
    lines = out[a:b].split("\n")
    lens = [len(_html.unescape(x)) for x in lines]
    width = max([n for n in lens if n <= 82] or [0])
    for sub, nums in items:
        e = _html.escape(sub)
        for k, ln in enumerate(lines):
            if e in ln:
                if not ln.endswith("</span>"):
                    ln = ln + " " * max(1, width - len(_html.unescape(ln))) + " "
                lines[k] = ln + "".join(f'<span class="mk">{n}</span>' for n in nums)
                break
        else:
            raise ValueError(f"eol mark text not found: {sub!r}")
    return head + sep + out[:a] + "\n".join(lines) + out[b:]


def mk(frag, marks):
    """Inline marks after the first occurrence of each substring in the OUTPUT (text or table),
    never in the code input or image data. (labkit's own `marks` requires every substring to be
    present in both the printed text and the DataFrame table of a cell, so we mark here instead.)"""
    head, sep, out = frag.partition('<div class="cell-out">')
    stop = out.find("<img")
    stop = len(out) if stop < 0 else stop
    for sub, n in marks.items():
        e = _html.escape(sub)
        i = out.find(e, 0, stop)
        if i < 0:
            raise ValueError(f"mark text not found: {sub!r}")
        j = i + len(e)
        ins = f'<span class="mk">{n}</span>'
        out = out[:j] + ins + out[j:]
        stop += len(ins)
    return head + sep + out


def H(c, marks=None, lines=None):
    f = nb.html(c)
    if marks:
        f = mk(f, marks)
    if lines:
        f = eol(f, lines)
    return f


nb = Notebook("lab06")
S = nb.save_fragment

# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
import pandas as pd
from sklearn.datasets import load_diabetes

data = load_diabetes(scaled=False, as_frame=True)
df = data.frame            # 설명변수 10개 + target 1개
print(df.shape)            # (행 수, 열 수)
df.head()
''', title="당뇨병 자료 불러오기")
S("lab06_load", H(c, marks={"(442, 11)": 1, "151.0": 2}))

c = nb.cell('''
desc = data.DESCR                        # 자료 설명서(문자열)
start = desc.find(":Attribute Information:")
end = desc.find("Source URL")
print(desc[start:end])
''', title="자료 설명서에서 변수 정의 확인")
S("lab06_descr", H(c, marks={"age in years": 1, "- sex": 2, "average blood pressure": 3,
                                    "possibly log of serum triglycerides level": 4,
                                    "Each of these 10 feature variables": 5}))

c = nb.cell('''
print(df["sex"].value_counts())    # sex에 어떤 값이 몇 명씩 있는지
df["sex"] = df["sex"].astype(int)  # 1.0, 2.0 -> 1, 2 (정수로)
df.describe().T.round(2)           # 변수별 요약통계 (행·열 뒤집기)
''', title="요약통계와 sex 코딩 확인", max_rows=15)
S("lab06_describe", H(c, marks={"1.0    235": 1, "26.38": 2, "152.13": 3, "346.00": 4}))

c = nb.cell('''
df["BMI"]      # 열 이름을 대문자로 잘못 쓴 경우
''', title="열 이름을 틀리면", expect_error=True)
S("lab06_keyerror", H(c))

c = nb.cell('''
import matplotlib.pyplot as plt

cols = ["bmi", "bp", "s5", "target"]
pd.plotting.scatter_matrix(df[cols], figsize=(7, 7), alpha=0.35,
                           diagonal="hist", s=12)
plt.suptitle("Scatter matrix: diabetes data (n = 442)")
plt.show()
''', title="산점도 행렬")
S("lab06_scatter", H(c))

# ---------------------------------------------------------------- 나. 상관분석
c = nb.cell('''
from scipy import stats

res = stats.pearsonr(df["bmi"], df["target"])   # Pearson 상관
print(res)
print(res.confidence_interval(confidence_level=0.95))
''', title="Pearson 상관계수, p-value, 95% 신뢰구간")
S("lab06_pearson", H(c, marks={"statistic=np.float64(0.5864501344746887)": 1,
                                      "pvalue=np.float64(3.4660064451669974e-42))": 2,
                                      "high=np.float64(0.6444700712920269))": 3}))

c = nb.cell('''
import numpy as np

n, r = len(df), res.statistic
t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)   # 6장 가 절의 t 공식
z, se = np.arctanh(r), 1 / np.sqrt(n - 3)    # Fisher z 변환과 SE
print(f"t = {t:.3f}, df = {n - 2}")
print("95% CI:", np.tanh([z - 1.96 * se, z + 1.96 * se]).round(4))
''', title="손 공식으로 검산하기")
S("lab06_fisher", H(c, marks={"t = 15.187, df = 440": 1, "0.6445]": 2}))

c = nb.cell('''
rho = stats.spearmanr(df["bmi"], df["target"])   # Spearman 순위상관
print(rho)
''', title="Spearman 순위상관계수")
S("lab06_spearman", H(c, marks={"statistic=np.float64(0.5613820101065616)": 1}))

c = nb.cell('''
cols = ["age", "bmi", "bp", "s1", "s2", "s3", "s5", "target"]
df[cols].corr().round(2)       # 기본은 Pearson
''', title="상관행렬")
S("lab06_corr", H(c))

c = nb.cell('''
bad = df.copy()               # 원본은 그대로 두고 복사본을 만듦
bad.loc[0, "bmi"] = 321       # 32.1을 소수점 없이 입력한 오류 가정
r_bad = stats.pearsonr(bad["bmi"], bad["target"]).statistic
rho_bad = stats.spearmanr(bad["bmi"], bad["target"]).statistic
print(f"Pearson r    : {res.statistic:.3f} -> {r_bad:.3f}")
print(f"Spearman rho : {rho.statistic:.3f} -> {rho_bad:.3f}")
print("bmi max      :", bad["bmi"].max())
''', title="입력 오류 한 건이 상관계수에 주는 영향")
S("lab06_outlier", H(c, marks={"0.586 -> 0.176": 1, "0.561 -> 0.561": 2, "321.0": 3}))

c = nb.cell('''
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(df["bmi"], df["target"], s=10, alpha=0.5)
ax[0].set_title(f"Original data (r = {res.statistic:.2f})")
ax[1].scatter(bad["bmi"], bad["target"], s=10, alpha=0.5)
ax[1].set_title(f"One data-entry error (r = {r_bad:.2f})")
for a in ax:
    a.set_xlabel("BMI (kg/m2)")
    a.set_ylabel("Disease progression")
plt.tight_layout()
plt.show()
''', title="입력 오류가 있는 산점도")
S("lab06_outlier_plot", H(c))

# ---------------------------------------------------------------- 다. 단순회귀분석
c = nb.cell('''
import statsmodels.formula.api as smf

model = smf.ols("target ~ bmi", data=df).fit()   # 모형 적합
print(model.summary())                           # 결과 요약표
''', title="단순회귀 적합과 summary()")
S("lab06_ols", H(c, lines=[
    ("Dep. Variable:", [1, 2]),
    ("Adj. R-squared:", [3]),
    ("F-statistic:", [4]),
    ("Prob (F-statistic):", [5]),
    ("Log-Likelihood:", [6]),
    ("No. Observations:", [7, 8]),
    ("Df Residuals:", [9]),
    ("Df Model:", [10]),
    ("Intercept   -117.7734", [11]),
    ("bmi           10.2331", [12, 13, 14, 15]),
    ("Omnibus:", [16]),
    ("Cond. No.", [17]),
]))

c = nb.cell('''
print(model.pvalues)                     # 계수별 정확한 p-value
print("Pearson p :", res.pvalue)
print("t^2 =", round(model.tvalues["bmi"] ** 2, 2),
      "  F =", round(model.fvalue, 2))
print("r^2 =", round(res.statistic ** 2, 4),
      "  R2 =", round(model.rsquared, 4))
print(model.conf_int().round(3))         # 계수의 95% 신뢰구간
''', title="기울기 검정 = 상관 검정 확인")
S("lab06_same", H(c, marks={"bmi          3.466006e-42": 1, "Pearson p : 3.4660064451669974e-42": 2,
                                   "F = 230.65": 3, "R2 = 0.3439": 4}))

c = nb.cell('''
import statsmodels.api as sm

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(model.fittedvalues, model.resid, s=10, alpha=0.5)
ax[0].axhline(0, color="gray", linestyle="--")
ax[0].set_xlabel("Fitted values")
ax[0].set_ylabel("Residuals")
ax[0].set_title("Residuals vs fitted")
sm.qqplot(model.resid, line="s", ax=ax[1])
ax[1].set_title("Normal Q-Q plot of residuals")
plt.tight_layout()
plt.show()
''', title="잔차 그림 두 가지")
S("lab06_resid", H(c))

c = nb.cell('''
new = pd.DataFrame({"bmi": [25, 30]})     # 예측하고 싶은 BMI 값
pred = model.get_prediction(new)
pred.summary_frame(alpha=0.05).round(1)
''', title="평균의 신뢰구간과 개인의 예측구간")
S("lab06_pred", H(c, marks={"138.1": 1, "3.1": 2, "131.9": 3, "15.0": 4}))

# ---------------------------------------------------------------- 라. 다중회귀분석
c = nb.cell('''
model_adj = smf.ols("target ~ bmi + age + C(sex) + bp + s5",
                    data=df).fit()
print(model_adj.summary())
''', title="다중회귀 적합")
S("lab06_mlr", H(c, lines=[
    ("R-squared:                       0.487", [1]),
    ("Adj. R-squared:", [2]),
    ("F-statistic:", [3]),
    ("Df Model:", [4]),
    ("C(sex)[T.2]", [5]),
    ("bmi             6.4521", [6]),
    ("age            -0.1322", [7]),
    ("s5             51.0543", [8]),
    ("[2] The condition number is large", [9]),
]))

c = nb.cell('''
b_crude = model.params["bmi"]         # 단순회귀의 bmi 계수
b_adj = model_adj.params["bmi"]       # 다중회귀의 bmi 계수
print(f"bmi coefficient: {b_crude:.2f} -> {b_adj:.2f}")
print(f"change: {(b_adj - b_crude) / b_crude * 100:.1f}%")
print(df[["bmi", "bp", "s5", "target"]].corr().round(2))
''', title="보정 전후 BMI 계수 비교")
S("lab06_crude_adj", H(c, marks={"10.23 -> 6.45": 1, "-36.9%": 2}))

c = nb.cell('''
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor)

def vif_table(m):
    X, names = m.model.exog, m.model.exog_names   # 설계행렬, 열 이름
    v = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    s = pd.Series(v, index=names, name="VIF")    # 이름 붙인 Series
    return s.drop("Intercept").round(2)

vif_table(model_adj)
''', title="분산팽창계수(VIF)")
S("lab06_vif", H(c))

c = nb.cell('''
model_all = smf.ols("target ~ bmi + age + C(sex) + bp"
                    " + s1 + s2 + s3 + s4 + s5 + s6", data=df).fit()
print(vif_table(model_all))
print(f"r(s1, s2) = {df['s1'].corr(df['s2']):.2f}")
print("SE of s5:", round(model_adj.bse["s5"], 2), "->",
      round(model_all.bse["s5"], 2))
''', title="혈청 측정값 6개를 모두 넣으면")
S("lab06_vif_all", H(c, marks={"59.20": 1, "r(s1, s2) = 0.90": 2, "5.94 -> 15.67": 3}))

c = nb.cell('''
model_log = smf.ols("np.log(target) ~ bmi + age + C(sex) + bp + s5",
                    data=df).fit()
pct = (np.exp(model_log.params) - 1) * 100          # % 차이
pct_ci = (np.exp(model_log.conf_int()) - 1) * 100   # CI 양 끝 변환
out = pd.concat([model_log.params, pct, pct_ci], axis=1)
out.columns = ["beta (log)", "% diff", "lower %", "upper %"]
out.drop("Intercept").round({"beta (log)": 4, "% diff": 1,
                             "lower %": 1, "upper %": 1})
''', title="로그 변환한 결과변수")
S("lab06_log", H(c, marks={"0.0409": 1, "4.2": 2, "3.1": 3, "-8.7": 4}))

c = nb.cell('''
sk = lambda x: f"{stats.skew(x):.2f}"      # 왜도를 소수 둘째 자리로
print("skewness of target     :", sk(df["target"]))
print("skewness of log(target):", sk(np.log(df["target"])))
print("skewness of residuals  :", sk(model_adj.resid), "(linear),",
      sk(model_log.resid), "(log)")
''', title="로그 변환이 필요했는지 확인")
S("lab06_skew", H(c, marks={"target     : 0.44": 1, "log(target): -0.33": 2, "0.14 (linear), -0.40 (log)": 3}))

c = nb.cell('''
def tidy(m):
    ci = m.conf_int()
    f2 = "{:.2f}".format
    est = (m.params.map(f2) + " (" + ci[0].map(f2) + " to "
           + ci[1].map(f2) + ")")
    p = m.pvalues.map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
    return pd.DataFrame({"beta (95% CI)": est, "P": p})

terms = ["bmi", "age", "C(sex)", "bp", "s5"]
crude = pd.concat([tidy(smf.ols(f"target ~ {t}", data=df).fit())
                   .drop("Intercept") for t in terms])
table = crude.join(tidy(model_adj), lsuffix=" crude",
                   rsuffix=" adjusted")
table
''', title="논문용 결과표 만들기")
S("lab06_table", H(c, marks={"10.23 (8.91 to 11.56)": 1, "6.45 (5.09 to 7.82)": 2,
                               "1.10 (0.56 to 1.65)": 3, "C(sex)[T.2]": 4}))

print("lab06: cells", len(nb.cells))
