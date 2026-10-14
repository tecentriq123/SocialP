"""실습 6 (상관분석과 선형회귀) — real cells run with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab06.py
      then  python3 tools/nbcheck.py lab06   (runs the notebook alone in a fresh process and compares outputs)
Writes figs/lab06_*.html (insert in content/lab06.html with <!--FIG:lab06_xxx-->),
pub/notebooks/lab06.ipynb (download link at the top of content/lab06.html) and checks/lab06.expected.json.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab06.html that are not
marked in a cell output (대응표, the version notes, the 과제 problems and answers).
(2026-10-14: version cell 1 split from the load cell, influence cell in c 절, 과제 1–3 and the notebook added;
cell numbers shifted by 1, and by 2 from the prediction cell on.)
"""
import html as _html
import os
import re
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


def freeze_clock(h):
    """summary()가 찍는 실행 날짜와 시각(Date, Time 줄)을 고정값으로 바꾼다(gen/lab_lab09.py와 같음). 다시 돌려도
    같은 파일이 나오게 하려는 것이고 통계 출력은 건드리지 않는다. 같은 길이의 글자로 바꾸므로 줄 맞춤도 그대로다."""
    h = re.sub(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}", r"\g<1>Wed, 14 Oct 2026", h)
    return re.sub(r"(Time:\s+)\d{2}:\d{2}:\d{2}", r"\g<1>09:10:59", h)


def H(c, marks=None, lines=None):
    f = nb.html(c)
    if marks:
        f = mk(f, marks)
    if lines:
        f = eol(f, lines)
    return freeze_clock(f)


nb = Notebook("lab06")
S = nb.save_fragment
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)

# ---------------------------------------------------------------- a. 실습 데이터 준비
c = nb.cell('''
import pandas as pd
import numpy as np
import scipy, statsmodels, matplotlib, sklearn

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scipy       ", scipy.__version__)
print("statsmodels ", statsmodels.__version__)
print("matplotlib  ", matplotlib.__version__)
print("scikit-learn", sklearn.__version__)
''', title="패키지 불러오기와 버전 확인")
S("lab06_versions", H(c, lines=[("numpy", [1]), ("statsmodels", [2]), ("scikit-learn", [3])]))
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 당뇨병 자료를 읽습니다. "
              "자료는 scikit-learn 안에 들어 있어 인터넷 연결 없이 불러옵니다.")

c = nb.cell('''
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
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

c = nb.cell('''
import matplotlib.pyplot as plt

cols = ["bmi", "bp", "s5", "target"]
pd.plotting.scatter_matrix(df[cols], figsize=(7, 7), alpha=0.35,
                           diagonal="hist", s=12)
plt.suptitle("Scatter matrix: diabetes data (n = 442)")
plt.show()
''', title="산점도 행렬")
S("lab06_scatter", H(c))
NOTES[c.n] = (f"열 이름을 틀려 일부러 오류를 내 보는 셀(셀 {ERR_CELL})은 이 노트북에서 뺐습니다"
              "(사이트의 실습 6 a 절에 있습니다).")

# ---------------------------------------------------------------- b. 상관분석
c = nb.cell('''
from scipy import stats

res = stats.pearsonr(df["bmi"], df["target"])   # Pearson 상관
print(res)
print(res.confidence_interval(confidence_level=0.95))
''', title="Pearson 상관계수, p-value, 95% 신뢰구간")
NOTES[c.n] = "## b. 상관분석"
S("lab06_pearson", H(c, marks={"statistic=np.float64(0.5864501344746887)": 1,
                                      "pvalue=np.float64(3.4660064451669974e-42))": 2,
                                      "high=np.float64(0.6444700712920269))": 3}))

c = nb.cell('''
n, r = len(df), res.statistic
t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)   # 6장 a 절의 t 공식
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

# ---------------------------------------------------------------- c. 단순회귀분석
c = nb.cell('''
import statsmodels.formula.api as smf

model = smf.ols("target ~ bmi", data=df).fit()   # 모형 적합
print(model.summary())                           # 결과 요약표
''', title="단순회귀 적합과 summary()")
NOTES[c.n] = "## c. 단순회귀분석"
OLS_CELL = c.n
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
infl = model.get_influence().summary_frame()    # 환자별 영향력 지표
top = infl.sort_values("cooks_d", ascending=False).head(3)
print(top[["cooks_d", "standard_resid", "hat_diag"]]
      .join(df[["bmi", "target"]]).round(3))
m_wo = smf.ols("target ~ bmi", data=df.drop(index=top.index[0])).fit()
print("slope, all 442 patients:", round(model.params["bmi"], 2))
print("slope, without top 1   :", round(m_wo.params["bmi"], 2))
''', title="영향력이 큰 환자 찾기")
S("lab06_infl", H(c, marks={"0.026": 1, "33.0": 2, "slope, all 442 patients: 10.23": 3,
                           "slope, without top 1   : 10.36": 4}))
INFL_CELL = c.n

c = nb.cell('''
new = pd.DataFrame({"bmi": [25, 30]})     # 예측하고 싶은 BMI 값
pred = model.get_prediction(new)
pred.summary_frame(alpha=0.05).round(1)
''', title="평균의 신뢰구간과 개인의 예측구간")
S("lab06_pred", H(c, marks={"138.1": 1, "3.1": 2, "131.9": 3, "15.0": 4}))

# ---------------------------------------------------------------- d. 다중회귀분석
c = nb.cell('''
model_adj = smf.ols("target ~ bmi + age + C(sex) + bp + s5",
                    data=df).fit()
print(model_adj.summary())
''', title="다중회귀 적합")
NOTES[c.n] = "## d. 다중회귀분석"
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
TABLE_CELL = c.n

# ---------------------------------------------------------------- e. 과제 (정답 셀)
c = nb.cell('''
print(df["bp"].describe().round(1))                # bp의 범위와 표준편차
b = model_adj.params["bp"]                         # 셀 18의 보정 계수
lo, hi = model_adj.conf_int().loc["bp"]
sd_bp = df["bp"].std()
print("per 1 unit :", np.round([b, lo, hi], 2))
print("per 10 units:", np.round(10 * np.array([b, lo, hi]), 1))
print(f"per 1 SD ({sd_bp:.1f}):", np.round(sd_bp * np.array([b, lo, hi]), 1))
''', title="과제 1 정답. 혈압(bp) 행 읽기")
S("lab06_hw1", H(c, marks={"std       13.8": 1, "min       62.0": 2, "[1.04 0.6  1.48]": 3, "[10.4  6.  14.8]": 4,
                          "[14.4  8.2 20.5]": 5}))
NOTES[c.n] = ("## 과제 정답\n\n"
              f"**과제 1.** 셀 {TABLE_CELL}의 결과표에서 bp 행의 보정 계수(1.04, 95% CI 0.60 to 1.48)를 논문의 Results처럼 "
              "한 문장으로 옮기고, 이 숫자가 말해 주지 않는 것을 두 가지 이상 쓰세요. bp의 범위와 표준편차를 확인하고, "
              "10 단위당과 1 표준편차당 계수도 구해 보세요.")

c = nb.cell('''
sets = {"bmi only": "target ~ bmi",
        "+ age, sex": "target ~ bmi + age + C(sex)",
        "+ age, sex, bp": "target ~ bmi + age + C(sex) + bp",
        "+ age, sex, s5": "target ~ bmi + age + C(sex) + s5",
        "+ age, sex, bp, s5": "target ~ bmi + age + C(sex) + bp + s5"}
rows = []
for name, f in sets.items():
    m = smf.ols(f, data=df).fit()
    lo, hi = m.conf_int().loc["bmi"]               # bmi 계수의 95% CI
    rows.append({"model": name, "bmi": m.params["bmi"], "lower": lo,
                 "upper": hi, "SE": m.bse["bmi"], "R2": m.rsquared})
pd.DataFrame(rows).set_index("model").round(3)
''', title="과제 2 정답. 보정 변수를 바꿔 가며 BMI 계수 보기")
S("lab06_hw2", H(c, marks={"9.990": 1, "8.484": 2, "7.278": 3, "6.452": 4}))
NOTES[c.n] = (f"**과제 2.** 셀 {OLS_CELL}의 단순회귀에서 시작해 보정 변수를 하나씩 바꿔 가며 BMI 계수를 다시 구하세요. "
              "나이와 sex만, 여기에 bp만, s5만, bp와 s5를 모두 넣은 모형의 BMI 계수, 95% CI, 표준오차, R²를 한 표로 만들고, "
              "어떤 변수를 넣을 때 계수가 크게 줄어드는지, 왜 그런지 설명하세요.")

c = nb.cell('''
r = model.resid                                  # 셀 13 단순회귀의 잔차
keep = r.abs() <= 2 * r.std()                    # 동료: 잔차가 2 SD 넘는 환자를 뺌
m_trim = smf.ols("target ~ bmi", data=df[keep]).fit()
print("removed:", (~keep).sum(), "patients")
print(df.loc[~keep, ["bmi", "target"]].describe().loc[["min", "max"]])
for name, m in [("all 442", model), ("trimmed", m_trim)]:
    lo, hi = m.conf_int().loc["bmi"]
    print(f"{name:8s} slope {m.params['bmi']:.2f} ({lo:.2f} to {hi:.2f})"
          f"  R2 {m.rsquared:.3f}")
''', title="과제 3 정답. 잔차가 큰 환자를 뺀 분석")
S("lab06_hw3", H(c, marks={"removed: 13": 1, "21.0": 2, "311.0": 3, "slope 10.23 (8.91 to 11.56)  R2 0.344": 4,
                          "slope 10.84 (9.59 to 12.09)  R2 0.406": 5}))
NOTES[c.n] = ("**과제 3.** 동료가 단순회귀의 잔차가 2 표준편차를 넘는 환자를 이상값으로 보고 빼서 다시 적합한 뒤 "
              "\"이상값 13명을 제외하자 모형의 설명력이 R² 0.34에서 0.41로 좋아졌고, BMI 1 kg/m²당 진행 점수가 10.8점"
              "(95% CI 9.6–12.1) 높았다\"라고 썼습니다. 동료의 분석을 재현하고, 무엇이 왜 틀렸는지 두 가지 이상 설명하세요. "
              f"셀 {INFL_CELL}의 영향력 확인과 비교해 보세요.")

print("lab06: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 6. 상관분석과 선형회귀",
    intro=("사회약학 연구방법 노트의 '실습 6. 상관분석과 선형회귀'에 딸린 노트북입니다. scikit-learn에 들어 있는 당뇨병 "
           "환자 442명 자료로 BMI와 1년 뒤 질병 진행 점수의 Pearson·Spearman 상관계수를 구하고(b), 단순회귀(c)와 "
           "다중회귀(d)를 statsmodels로 적합한 뒤 논문용 결과표를 만듭니다. 셀을 위에서부터 차례로 실행하세요. "
           "설명과 출력 읽는 법은 사이트의 실습 6에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import stats as st
    from statsmodels.regression.linear_model import OLS

    ns = nb.ns
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r2 = lambda x: f"{x:.2f}"
    # 글(content/lab06.html)과 노트북 안내, 정답 셀의 주석에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 13: "단순회귀 적합과 summary()", 16: "영향력이 큰 환자 찾기",
             18: "다중회귀 적합", 24: "논문용 결과표 만들기"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    # 셀 1의 출력 읽기에 적은 버전
    same("versions", (ns["np"].__version__, ns["statsmodels"].__version__, ns["sklearn"].__version__),
         ("2.5.3", "0.15.0", "1.9.1"))
    # 버전 노트: 상수항 열이 든 설계행렬이면 옛 방식(다른 열로 회귀한 R²)과 0.15의 VIF가 같다
    m_ = ns["model_adj"]
    X_ = m_.model.exog
    names_ = m_.model.exog_names

    def vif_raw(exog, j):
        mask = np.arange(exog.shape[1]) != j
        return 1 / (1 - OLS(exog[:, j], exog[:, mask]).fit().rsquared)
    old_ = pd.Series([vif_raw(X_, j) for j in range(X_.shape[1])], index=names_).drop("Intercept").round(2)
    same("VIF old method = vif_table", old_.tolist(), ns["vif_table"](m_).tolist())
    # 대응표: 이 실습의 자료 열
    df_, mod_ = ns["df"], ns["model"]
    same("N", len(df_), 442)
    same("target range", (int(df_["target"].min()), int(df_["target"].max())), (25, 346))
    same("r, slope, R2", (r2(ns["res"].statistic), r2(mod_.params["bmi"]), r2(mod_.rsquared)), ("0.59", "10.23", "0.34"))
    same("adjusted bmi", r2(m_.params["bmi"]), "6.45")
    # 대응표: 6장의 예 열 (6장 a–c 절의 환자 8명을 다시 계산)
    age = np.array([42, 48, 53, 58, 62, 66, 71, 80])
    sbp = np.array([125, 127, 133, 141, 129, 139, 137, 149])
    lr = st.linregress(age, sbp)
    same("ch06 8 patients r, slope, R2", (r2(lr.rvalue), f"{lr.slope:.3f}", r2(lr.rvalue ** 2)), ("0.84", "0.547", "0.71"))
    # c 절: 영향력
    infl_ = ns["infl"]
    same("max Cook's D", f"{infl_['cooks_d'].max():.3f}", "0.026")
    same("top patient bmi, target", (df_.loc[infl_["cooks_d"].idxmax(), "bmi"], df_.loc[infl_["cooks_d"].idxmax(), "target"]),
         (33.0, 55.0))
    # 과제 1
    b_, (lo_, hi_) = m_.params["bp"], m_.conf_int().loc["bp"]
    same("bp adjusted (CI)", (r2(b_), r2(lo_), r2(hi_)), ("1.04", "0.60", "1.48"))
    crude_bp = ns["smf"].ols("target ~ bp", data=df_).fit()
    same("bp crude (CI)", (r2(crude_bp.params["bp"]), r2(crude_bp.conf_int().loc["bp", 0]),
                           r2(crude_bp.conf_int().loc["bp", 1])), ("2.46", "1.99", "2.93"))
    same("bp sd, range", (f"{df_['bp'].std():.1f}", int(df_["bp"].min()), int(df_["bp"].max())), ("13.8", 62, 133))
    same("corr bp with bmi, s5, age", [r2(df_["bp"].corr(df_[v])) for v in ("bmi", "s5", "age")], ["0.40", "0.39", "0.34"])
    # 과제 2: 표의 값 (셀 출력에서 반올림한 둘째 자리)
    rows_ = pd.DataFrame(ns["rows"]).set_index("model")
    same("bmi by model", [r2(v) for v in rows_["bmi"]], ["10.23", "9.99", "8.48", "7.28", "6.45"])
    same("R2 by model", [f"{v:.3f}" for v in rows_["R2"]], ["0.344", "0.351", "0.400", "0.462", "0.487"])
    same("corr bmi with bp, s5, age, sex", [r2(df_["bmi"].corr(df_[v])) for v in ("bp", "s5", "age", "sex")],
         ["0.40", "0.45", "0.19", "0.09"])
    same("corr target with s5, bp", [r2(df_["target"].corr(df_[v])) for v in ("s5", "bp")], ["0.57", "0.44"])
    # 과제 3
    keep_ = ns["keep"]
    same("removed", int((~keep_).sum()), 13)
    same("resid SD", r2(ns["r"].std()), "62.44")
    same("removed above / below the line", (int((ns["r"][~keep_] > 0).sum()), int((ns["r"][~keep_] < 0).sum())), (8, 5))
    mt_ = ns["m_trim"]
    same("trimmed SE vs all", (f"{mt_.bse['bmi']:.3f}", f"{mod_.bse['bmi']:.3f}"), ("0.634", "0.674"))
    same("resid SD (n - 2), all vs trimmed", (r2(np.sqrt(mod_.scale)), r2(np.sqrt(mt_.scale)), "%.1f" % np.sqrt(mt_.scale)), ("62.52", "58.25", "58.2"))
    same("SE range across models", (f"{rows_['SE'].min():.3f}", f"{rows_['SE'].max():.3f}"), ("0.674", "0.705"))
    same("max VIF", r2(ns["vif_table"](m_).max()), "1.40")
    rr_ = ns["r"][~keep_]
    same("mean BMI of removed above / below", (f"{df_.loc[rr_[rr_ > 0].index, 'bmi'].mean():.1f}",
                                               f"{df_.loc[rr_[rr_ < 0].index, 'bmi'].mean():.1f}"), ("25.0", "31.2"))
    same("slope change without top 1 (%)", f"{(ns['m_wo'].params['bmi'] / mod_.params['bmi'] - 1) * 100:.1f}", "1.2")
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab06.html passed")
