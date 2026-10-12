"""실습 9 (로지스틱 회귀분석) — real cells run with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab09.py
      then  python3 tools/nbcheck.py lab09   (runs the notebook alone in a fresh process and compares outputs)
Writes figs/lab09_*.html (insert in content/lab09.html with <!--FIG:lab09_xxx-->),
pub/notebooks/lab09.ipynb (download link at the top of content/lab09.html) and checks/lab09.expected.json.
Part 1: MASS birthwt (Rdatasets URL -> offline copy via labkit). Part 2: sklearn load_breast_cancer.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab09.html that are not
marked in a cell output (대응표, the VIF tip, the 과제 problems and answers).
(2026-10-12: version cell 1, VIF cell, 과제 1–3 and the notebook added; cell numbers shifted by 1 and 2.)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402


def eol(frag, items, pre_index=0):
    """Append marks at the END of the output line containing `sub` (keeps fixed-width summaries aligned)."""
    head, sep, out = frag.partition('<div class="cell-out">')
    starts, i = [], 0
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
    width = max([n for n in lens if n <= 88] or [0])
    for sub, nums in items:
        e = _html.escape(sub)
        for k, ln in enumerate(lines):
            if e in ln:
                if not ln.endswith("</span>"):
                    ln = ln + " " * max(1, width - len(_html.unescape(ln))) + " "
                lines[k] = ln + "".join(f'<span class="mk">{n}</span>' for n in nums)
                break
        else:
            if os.environ.get("LABDEBUG"):
                print("  !! eol mark not found:", repr(sub))
                continue
            raise ValueError(f"eol mark text not found: {sub!r}")
    return head + sep + out[:a] + "\n".join(lines) + out[b:]


def mk(frag, marks):
    """Inline marks after the first occurrence of each substring in the OUTPUT (text or table only).
    (labkit's own `marks` needs each substring in both printed text and DataFrame table, so we mark here.)"""
    head, sep, out = frag.partition('<div class="cell-out">')
    stop = out.find("<img")
    stop = len(out) if stop < 0 else stop
    for sub, n in marks.items():
        e = _html.escape(sub)
        i = out.find(e, 0, stop)
        if i < 0:
            if os.environ.get("LABDEBUG"):
                print("  !! mark not found:", repr(sub))
                continue
            raise ValueError(f"mark text not found: {sub!r}")
        j = i + len(e)
        ins = f'<span class="mk">{n}</span>'
        out = out[:j] + ins + out[j:]
        stop += len(ins)
    return head + sep + out


def freeze_clock(h):
    """summary()가 찍는 실행 날짜와 시각(Date, Time 줄)을 고정값으로 바꾼다(gen/lab_lab15.py와 같음). 다시 돌려도
    같은 파일이 나오게 하려는 것이고 통계 출력은 건드리지 않는다. 같은 길이의 글자로 바꾸므로 줄 맞춤도 그대로다."""
    h = re.sub(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}", r"\g<1>Mon, 12 Oct 2026", h)
    return re.sub(r"(Time:\s+)\d{2}:\d{2}:\d{2}", r"\g<1>09:10:59", h)


def H(c, marks=None, lines=None):
    f = nb.html(c)
    if marks:
        f = mk(f, marks)
    if lines:
        f = eol(f, lines)
    return freeze_clock(f)


nb = Notebook("lab09")
S = nb.save_fragment
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)

# ================================================================ 가. 실습 데이터 준비
c = nb.cell('''
import pandas as pd
import numpy as np
import statsmodels, scipy, matplotlib, sklearn

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("statsmodels ", statsmodels.__version__)
print("scipy       ", scipy.__version__)
print("matplotlib  ", matplotlib.__version__)
print("scikit-learn", sklearn.__version__)
''', title="패키지 불러오기와 버전 확인")
S("lab09_versions", H(c, lines=[("statsmodels", [1]), ("scikit-learn", [2])]))
NOTES[c.n] = ("## 가. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 birthwt 자료를 읽습니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
url = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv"
bw = pd.read_csv(url)        # 인터넷의 CSV 파일을 DataFrame으로 읽음
print(bw.shape)
bw.head()
''', title="출생체중 자료 불러오기")
S("lab09_load", H(c, marks={"(189, 11)": 1, "rownames": 2, "2523": 3}))

c = nb.cell('''
print(bw["low"].value_counts())                   # 1 = 저체중아
print((bw["bwt"] < 2500).equals(bw["low"] == 1))  # 정의 확인
bw["race"] = pd.Categorical(
    bw["race"].map({1: "white", 2: "black", 3: "other"}),
    categories=["white", "black", "other"])       # 첫 범주가 기준
bw["ptd"] = (bw["ptl"] > 0).astype(int)   # 이전 조산 1회 이상 = 1
pd.crosstab(bw["ptl"], bw["low"], margins=True)
''', title="결과변수와 설명변수 코딩")
S("lab09_coding", H(c, marks={"0    130": 1, "True": 2}))

c = nb.cell('''
n_event = int(bw["low"].sum())            # 사건(저체중아) 수
n_non = int((bw["low"] == 0).sum())       # 사건이 없는 수
k = 8     # 절편 외 계수: age, lwt, race 2개, smoke, ptd, ht, ui
print("events:", n_event, " non-events:", n_non)
print("proportion:", round(n_event / len(bw), 3))
print("EPV =", round(min(n_event, n_non) / k, 1))
''', title="사건 수와 EPV")
S("lab09_epv", H(c, marks={"events: 59": 1, "proportion: 0.312": 2, "EPV = 7.4": 3}))

# ================================================================ 나. 다변수 로지스틱 회귀
c = nb.cell('''
import statsmodels.formula.api as smf

f = "low ~ age + lwt + C(race) + smoke + ptd + ht + ui"
logit_m = smf.logit(f, data=bw).fit()     # 최대우도추정
print(logit_m.summary())
''', title="다변수 로지스틱 회귀 적합")
NOTES[c.n] = "## 나. 다변수 로지스틱 회귀"
S("lab09_logit", H(c, lines=[
    ("Optimization terminated successfully.", [1]),
    ("Iterations 6", [2]),
    ("Dep. Variable:", [3]),
    ("Df Residuals:", [4]),
    ("Df Model:", [5]),
    ("Pseudo R-squ.:", [6]),
    ("Log-Likelihood:", [7]),
    ("converged:", [8, 9]),
    ("LLR p-value:", [10]),
    ("coef    std err          z      P>|z|", [11]),
    ("Intercept            0.6369", [12]),
    ("C(race)[T.black]     1.2127", [13]),
    ("smoke                0.8464", [14, 15]),
]))

c = nb.cell('''
ci = logit_m.conf_int()                  # 로그 오즈비의 95% CI
or_tab = pd.DataFrame({
    "OR": np.exp(logit_m.params),        # e^β = 오즈비
    "lower": np.exp(ci[0]),
    "upper": np.exp(ci[1]),
    "P": logit_m.pvalues})
or_tab.drop("Intercept").round(3)
''', title="오즈비와 95% 신뢰구간 표")
S("lab09_or", H(c, marks={"2.331": 1, "1.048": 2, "5.187": 3, "0.985": 4, "3.363": 5}))

c = nb.cell('''
def or_per(m, var, c):                  # c 단위당 오즈비와 95% CI
    est = m.params[var]                 # 1단위당 로그 오즈비
    lo, hi = m.conf_int().loc[var]      # 그 95% 신뢰구간
    return np.round(np.exp([c * est, c * lo, c * hi]), 3)

crude_lwt = smf.logit("low ~ lwt", data=bw).fit(disp=0)
crude_age = smf.logit("low ~ age", data=bw).fit(disp=0)
print("lwt per 10 lb, adjusted:", or_per(logit_m, "lwt", 10))
print("lwt per 10 lb, crude   :", or_per(crude_lwt, "lwt", 10))
print("age per 5 yr,  adjusted:", or_per(logit_m, "age", 5))
print("age per 5 yr,  crude   :", or_per(crude_age, "age", 5))
''', title="임상적으로 의미 있는 단위의 오즈비")
S("lab09_unit", H(c, marks={"lwt per 10 lb, adjusted: [0.861": 1, "0.989]": 2,
                            "lwt per 10 lb, crude   : [0.869": 3, "age per 5 yr,  adjusted: [0.828": 4}))

c = nb.cell('''
def or_ci(m):
    ci = np.exp(m.conf_int())
    f2 = "{:.2f}".format
    return (np.exp(m.params).map(f2) + " (" + ci[0].map(f2)
            + " to " + ci[1].map(f2) + ")")

terms = ["age", "lwt", "C(race)", "smoke", "ptd", "ht", "ui"]
crude = pd.concat([or_ci(smf.logit(f"low ~ {t}", data=bw).fit(disp=0))
                   .drop("Intercept") for t in terms])
adj = or_ci(logit_m)[crude.index]         # 같은 순서로 맞춤
pd.DataFrame({"crude OR (95% CI)": crude,
              "adjusted OR (95% CI)": adj})
''', title="보정 전후 오즈비 비교")
S("lab09_crude", H(c, marks={"2.02 (1.08 to 3.78)": 1, "2.33 (1.05 to 5.19)": 2,
                             "4.32 (1.92 to 9.73)": 3, "2.33 (0.94 to 5.77)": 4}))

c = nb.cell('''
new = pd.DataFrame({
    "age":   [25, 25, 20, 20],
    "lwt":   [120, 120, 100, 100],
    "race":  ["white", "white", "black", "black"],
    "smoke": [0, 1, 0, 1],
    "ptd":   [0, 0, 1, 1],
    "ht":    [0, 0, 0, 0],
    "ui":    [0, 0, 1, 1]})
new["p_low"] = logit_m.predict(new).round(3)   # 예측 확률
new
''', title="예시 산모의 예측 확률")
S("lab09_predict", H(c, marks={"0.109": 1, "0.223": 2, "0.823": 3, "0.916": 4}))

c = nb.cell('''
tab = pd.crosstab(bw["smoke"], bw["low"])   # 행 smoke, 열 low
risk = tab[1] / tab.sum(axis=1)             # 저체중아 비율(위험)
odds = tab[1] / tab[0]                      # 오즈
print(tab)
print("risk:", risk.round(3).tolist())
print("RR =", round(risk[1] / risk[0], 2),
      "  OR =", round(odds[1] / odds[0], 2))
''', title="흔한 결과에서 오즈비와 상대위험도")
S("lab09_orrr", H(c, marks={"risk: [0.252, 0.405]": 1, "RR = 1.61": 2, "OR = 2.02": 3}))

c = nb.cell('''
r1 = logit_m.predict(bw.assign(smoke=1)).mean()   # 모두 흡연했다면
r0 = logit_m.predict(bw.assign(smoke=0)).mean()   # 모두 비흡연이라면
print(f"standardized risk: smokers {r1:.3f}, non-smokers {r0:.3f}")
print(f"adjusted RR = {r1 / r0:.2f}, RD = {r1 - r0:.3f}")
''', title="모형 기반 표준화로 보정 상대위험도 구하기")
S("lab09_std", H(c, marks={"smokers 0.404": 1, "non-smokers 0.252": 2, "adjusted RR = 1.60": 3, "RD = 0.152": 4}))

c = nb.cell('''
bw["low12"] = bw["low"] + 1       # 결과를 1/2로 코딩했다고 가정
smf.logit("low12 ~ smoke", data=bw).fit()
''', title="결과변수를 0/1이 아닌 값으로 넣으면", expect_error=True)
S("lab09_err", H(c))
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

# ================================================================ 다. 모형 점검
c = nb.cell('''
from scipy import stats

f_sp = "low ~ age + bs(lwt, df=4) + C(race) + smoke + ptd + ht + ui"
spline_m = smf.logit(f_sp, data=bw).fit(disp=0)
lr = 2 * (spline_m.llf - logit_m.llf)              # 우도비 통계량
ddf = spline_m.df_model - logit_m.df_model         # 늘어난 계수 수
p = stats.chi2.sf(lr, ddf)                         # p-value
print(f"LR = {lr:.2f}, df = {ddf:.0f}, P = {p:.3f}")
print("AIC linear:", round(logit_m.aic, 1),
      " spline:", round(spline_m.aic, 1))
''', title="lwt의 로짓 선형성: 스플라인과 비교")
NOTES[c.n] = (f"결과변수를 1/2로 넣어 일부러 오류를 내 보는 셀(셀 {ERR_CELL})은 이 노트북에서 뺐습니다"
              "(사이트의 실습 9 나 절에 있습니다).\n\n## 다. 모형 점검")
S("lab09_spline", H(c, marks={"LR = 1.99, df = 3, P = 0.574": 1, "AIC linear: 214.8": 2,
                              "spline: 218.8": 3}))

c = nb.cell('''
import matplotlib.pyplot as plt

grid = pd.DataFrame({"lwt": np.arange(85, 251, 5)}).assign(
    age=23, race="white", smoke=0, ptd=0, ht=0, ui=0)
def log_odds(m):
    p = m.predict(grid)                 # 예측 확률
    return np.log(p / (1 - p))          # 확률 -> 로그 오즈
plt.figure(figsize=(6, 3.8))
plt.plot(grid["lwt"], log_odds(logit_m), label="Linear in lwt")
plt.plot(grid["lwt"], log_odds(spline_m), "--",
         label="B-spline (df = 4)")
plt.plot(bw["lwt"], np.full(len(bw), -8), "|", color="gray")  # 관측값
plt.xlabel("Mother's weight at last menstrual period (lb)")
plt.ylabel("Log odds of low birth weight")
plt.legend()
plt.show()
''', title="직선 모형과 스플라인 모형의 로그 오즈")
S("lab09_spline_plot", H(c))
PLT_CELL = c.n     # matplotlib.pyplot을 불러오는 셀

c = nb.cell('''
import statsmodels.api as sm

glm_m = smf.glm(f, data=bw, family=sm.families.Binomial()).fit()
print("same coefficients:", np.allclose(glm_m.params, logit_m.params))
infl = glm_m.get_influence().summary_frame()     # 관측값별 영향력
cols = ["cooks_d", "standard_resid", "hat_diag"]
top = infl[cols].sort_values("cooks_d", ascending=False).head(5)
top.join(bw[["low", "age", "lwt", "race", "smoke", "ptd", "ht",
             "ui"]]).round(3)
''', title="영향력이 큰 관측값 찾기")
S("lab09_infl", H(c, marks={"same coefficients: True": 1, "0.048": 2}))
SM_CELL = c.n      # statsmodels.api를 불러오는 셀

c = nb.cell('''
plt.figure(figsize=(8, 3))
plt.stem(infl.index, infl["cooks_d"], markerfmt=".", basefmt=" ")
plt.xlabel("Observation (row index)")
plt.ylabel("Cook's distance")
plt.title("Influence of each mother on the fitted model")
plt.show()
''', title="Cook의 거리 그림")
S("lab09_cook", H(c))

c = nb.cell('''
m_wo = smf.logit(f, data=bw.drop(index=top.index[0])).fit(disp=0)
comp = pd.DataFrame({"OR all (n=189)": np.exp(logit_m.params),
                     "OR without 1": np.exp(m_wo.params)})
comp.drop("Intercept").round(2)
''', title="가장 영향력이 큰 1명을 빼고 다시 적합")
S("lab09_sens", H(c, marks={"9.74": 1}))

c = nb.cell('''
f_ptl = "low ~ age + lwt + C(race) + smoke + C(ptl) + ht + ui"
sep_m = smf.logit(f_ptl, data=bw).fit()
print(sep_m.summary().tables[1])        # 계수 표만 출력
print("converged:", sep_m.mle_retvals["converged"])
''', title="분리가 생기는 모형: ptl을 범주 그대로 넣으면")
S("lab09_sep", H(c, lines=[
    ("Maximum number of iterations has been exceeded", [1]),
    ("C(ptl)[T.1]", [2]),
    ("C(ptl)[T.3]", [3]),
    ("converged: False", [4]),
]))

c = nb.cell('''
from statsmodels.stats.outliers_influence import variance_inflation_factor

Xd = logit_m.model.exog              # 설계행렬 (상수항 열 + 설명변수 8열)
names = logit_m.model.exog_names     # 각 열의 이름
vif = pd.Series([variance_inflation_factor(Xd, j)
                 for j in range(Xd.shape[1])], index=names, name="VIF")
vif.drop("Intercept").round(2)       # 상수항 행은 빼고 읽음
''', title="다중공선성: 분산팽창계수(VIF)")
S("lab09_vif", H(c, marks={"1.35": 1, "1.07": 2}))
VIF_CELL = c.n

# ================================================================ 라. 예측모형과 성능 평가
c = nb.cell('''
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer(as_frame=True)
X = bc.data                          # 설명변수 30개 (DataFrame)
print(X.shape)
print(bc.target_names)               # target 0, 1의 이름
print(bc.target.value_counts())
y = (bc.target == 0).astype(int)     # 악성 = 1, 양성 = 0 으로 바꿈
print("malignant (y = 1):", y.sum(), " proportion:", round(y.mean(), 3))
''', title="유방암 자료 불러오기와 결과 코딩")
NOTES[c.n] = ("## 라. 예측모형과 성능 평가\n\n유방암 자료는 scikit-learn에 들어 있어 인터넷 없이 불러옵니다. "
              f"그림과 마지막 셀에는 앞에서 불러온 matplotlib(셀 {PLT_CELL})과 statsmodels(셀 {SM_CELL})를 그대로 씁니다.")
S("lab09_bc", H(c, marks={"(569, 30)": 1, "['malignant' 'benign']": 2, "1    357": 3,
                          "malignant (y = 1): 212": 4}))

c = nb.cell('''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
print(X_train.shape, X_test.shape)
print(round(y_train.mean(), 3), round(y_test.mean(), 3))
''', title="훈련 자료와 시험 자료로 나누기")
S("lab09_split", H(c, marks={"(398, 30) (171, 30)": 1, "0.372 0.374": 2}))

c = nb.cell('''
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

clf = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)            # 훈련 자료로만 학습
print(clf.classes_)                  # predict_proba 열의 순서
''', title="표준화 + 로지스틱 회귀 모형 학습")
S("lab09_fit", H(c, marks={"[0 1]": 1}))

c = nb.cell('''
proba = clf.predict_proba(X_test)    # 시험 자료 171명의 확률
print(type(proba), proba.shape)
print(proba[:5].round(3))            # 앞 5명, 두 열
p_test = proba[:, 1]                 # 모든 행, 1번 열 = P(y = 1)
print(p_test[:5].round(3))
print(proba[:5].sum(axis=1))         # 각 행의 합은 1
''', title="predict_proba 결과 살펴보기")
S("lab09_proba", H(c, marks={"(171, 2)": 1, "[[0.887 0.113]": 2, "[0.113 0.002 0.    0.003 0.019]": 3,
                             "[1. 1. 1. 1. 1.]": 4}))

c = nb.cell('''
from sklearn.metrics import roc_curve, roc_auc_score

auc = roc_auc_score(y_test, p_test)          # (실제 0/1, 예측 확률)
fpr, tpr, thr = roc_curve(y_test, p_test)
print(f"AUC = {auc:.4f}")
plt.figure(figsize=(4.6, 4.4))
plt.plot(fpr, tpr, label=f"Logistic model (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
plt.xlabel("1 - Specificity (false positive rate)")
plt.ylabel("Sensitivity (true positive rate)")
plt.legend(loc="lower right")
plt.show()
''', title="ROC 곡선과 AUC")
S("lab09_roc", H(c, marks={"AUC = 0.9975": 1}))

c = nb.cell('''
rng = np.random.default_rng(2026)          # 난수 시드 고정
yt, aucs = y_test.to_numpy(), []
for _ in range(1000):
    i = rng.integers(0, len(yt), len(yt))  # 복원추출한 행 번호
    if yt[i].min() == yt[i].max():         # 한 범주만 뽑히면 건너뜀
        continue
    aucs.append(roc_auc_score(yt[i], p_test[i]))
print(len(aucs), np.percentile(aucs, [2.5, 97.5]).round(3))
''', title="AUC의 부트스트랩 95% 신뢰구간")
S("lab09_boot", H(c, marks={"1000": 1, "[0.992 1.   ]": 2}))

c = nb.cell('''
from sklearn.metrics import confusion_matrix

pred05 = (p_test >= 0.5).astype(int)        # 0.5 이상이면 악성으로
print(confusion_matrix(y_test, pred05))      # 행 = 실제, 열 = 예측
''', title="혼동행렬")
S("lab09_cm", H(c, marks={"[[106   1]": 1, "[  4  60]]": 2}))

c = nb.cell('''
def classify(t):
    pred = (p_test >= t).astype(int)         # 임계값 t 이상 = 악성
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {"threshold": t, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "accuracy": (tp + tn) / len(pred)}

pd.DataFrame([classify(t) for t in [0.5, 0.2]]).round(3)
''', title="임계값별 민감도, 특이도, 정확도")
S("lab09_thr", H(c, marks={"0.938": 1, "0.991": 2, "0.971": 3, "0.984": 4}))

c = nb.cell('''
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(
    y_test, p_test, n_bins=5, strategy="quantile")
print("mean predicted:", prob_pred.round(3))
print("observed      :", prob_true.round(3))
plt.figure(figsize=(4.6, 4.4))
plt.plot(prob_pred, prob_true, "o-", label="Model (5 groups)")
plt.plot([0, 1], [0, 1], "--", color="gray",
         label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed proportion malignant")
plt.legend()
plt.show()
''', title="교정 곡선")
S("lab09_calib", H(c, marks={"mean predicted: [0.    0.002 0.034 0.768 1.   ]": 1,
                             "observed      : [0.    0.    0.029 0.853 1.   ]": 2}))

c = nb.cell('''
from sklearn.metrics import brier_score_loss

brier = brier_score_loss(y_test, p_test)
p_ref = np.full(len(y_test), y_train.mean())   # 모두에게 같은 확률
print(f"Brier (model)    : {brier:.4f}")
print(f"Brier (reference): {brier_score_loss(y_test, p_ref):.4f}")
plt.figure(figsize=(6, 3))
plt.hist(p_test, bins=20)
plt.xlabel("Predicted probability of malignancy")
plt.ylabel("Number of patients")
plt.show()
''', title="Brier 점수와 예측 확률의 분포")
S("lab09_brier", H(c, marks={"Brier (model)    : 0.0187": 1, "Brier (reference): 0.2342": 2}))

c = nb.cell('''
p_half = p_test / 2                        # 모든 확률을 절반으로
acc = lambda p: ((p >= 0.5).astype(int) == y_test).mean()
print("AUC     :", round(roc_auc_score(y_test, p_test), 4), "->",
      round(roc_auc_score(y_test, p_half), 4))
print("Brier   :", round(brier_score_loss(y_test, p_test), 4), "->",
      round(brier_score_loss(y_test, p_half), 4))
print("accuracy:", round(acc(p_test), 3), "->", round(acc(p_half), 3))
print("accuracy if all benign:", round((y_test == 0).mean(), 3))
''', title="AUC와 정확도는 다른 것을 잽니다")
S("lab09_aucacc", H(c, marks={"0.9975 -> 0.9975": 1, "0.0187 -> 0.1134": 2, "0.971 -> 0.632": 3, "all benign: 0.626": 4}))

c = nb.cell('''
cols3 = ["mean radius", "mean texture", "mean smoothness"]
Z = StandardScaler().fit_transform(X_train[cols3])     # 표준화
sm_fit = sm.Logit(y_train.to_numpy(), sm.add_constant(Z)).fit(disp=0)
sk_c1 = LogisticRegression().fit(Z, y_train)            # 기본 C = 1
sk_big = LogisticRegression(C=1e6).fit(Z, y_train)      # 벌점 거의 없음
pd.DataFrame({"statsmodels": sm_fit.params[1:],
              "sklearn C=1": sk_c1.coef_[0],
              "sklearn C=1e6": sk_big.coef_[0]}, index=cols3).round(3)
''', title="statsmodels와 scikit-learn의 계수 비교")
S("lab09_smsk", H(c, marks={"5.047": 1, "3.704": 2}))

# ================================================================ 마. 과제 (정답 셀)
c = nb.cell('''
print(pd.crosstab(bw["ht"], bw["low"], margins=True))
print("adjusted OR (95% CI):",
      or_tab.loc["ht", ["OR", "lower", "upper"]].round(2).tolist())
r_ht1 = logit_m.predict(bw.assign(ht=1)).mean()   # 모두 고혈압 병력이 있다면
r_ht0 = logit_m.predict(bw.assign(ht=0)).mean()   # 모두 없다면
print(f"standardized risk: ht=1 {r_ht1:.3f}, ht=0 {r_ht0:.3f}")
print(f"adjusted RR = {r_ht1 / r_ht0:.2f}")
''', title="과제 1 정답. 고혈압 병력의 오즈비 읽기")
S("lab09_hw1", H(c, marks={"1      5   7   12": 1, "[6.29, 1.58, 24.95]": 2,
                           "ht=1 0.646, ht=0 0.289": 3, "adjusted RR = 2.24": 4}))
HW1_CELL = c.n
NOTES[c.n] = ("## 과제 정답\n\n**과제 1.** 셀 6의 오즈비 표에서 ht 행(고혈압 병력)을 한 문장으로 옮기고, 이 숫자가 말해 주지 "
              "않는 것을 두 가지 이상 쓰세요. 셀 11의 표준화 방법으로 고혈압 병력의 위험의 비를 구해 오즈비와 비교하세요.")

c = nb.cell('''
bw["race_b"] = pd.Categorical(bw["race"],
                              categories=["black", "white", "other"])
bw["lwt10"] = bw["lwt"] / 10                 # 체중을 10 lb 단위로
f_b = "low ~ age + lwt10 + C(race_b) + smoke + ptd + ht + ui"
m_b = smf.logit(f_b, data=bw).fit(disp=0)
print("log-likelihood:", round(logit_m.llf, 3), round(m_b.llf, 3))
print("same predictions:",
      np.allclose(logit_m.predict(bw), m_b.predict(bw)))
np.exp(m_b.params).round(3).rename("OR")
''', title="과제 2 정답. 기준범주와 단위 바꾸기")
S("lab09_hw2", H(c, marks={"-98.417 -98.417": 1, "same predictions: True": 2, "0.297": 3, "0.665": 4,
                           "0.861": 5, "2.331": 6}))
NOTES[c.n] = ("**과제 2.** 인종의 기준범주를 white에서 black으로, 체중(lwt)의 단위를 1 lb에서 10 lb로 바꿔 셀 5의 모형을 "
              "다시 적합하세요. 어떤 오즈비가 바뀌고 어떤 오즈비가 그대로인지, 흡연의 오즈비는 왜 그대로인지 설명하세요.")

c = nb.cell('''
keep = []
for t in terms:                                 # 셀 8의 일곱 변수
    m1 = smf.logit(f"low ~ {t}", data=bw).fit(disp=0)
    p1 = m1.pvalues.drop("Intercept").min()     # 가변수가 둘이면 작은 쪽
    print(f"{t:8} univariable P = {p1:.3f}")
    if p1 < 0.05:
        keep.append(t)

f_sel = "low ~ " + " + ".join(keep)            # 동료가 고른 모형
m_sel = smf.logit(f_sel, data=bw).fit(disp=0)
print(f_sel)
print("smoke, selected     :", or_per(m_sel, "smoke", 1),
      " P =", round(m_sel.pvalues["smoke"], 3))
print("smoke, prespecified :", or_per(logit_m, "smoke", 1))
bw.groupby("race", observed=True)[["smoke", "low"]].mean().round(3)  # 인종별 비율
''', title="과제 3 정답. 단변수 P로 변수를 고른 분석")
S("lab09_hw3", H(c, marks={"C(race)  univariable P = 0.067": 1, "low ~ lwt + smoke + ptd + ht + ui": 2,
                           "[1.648 0.833 3.259]": 3, "P = 0.151": 4, "[2.331 1.048 5.187]": 5,
                           "0.542": 6, "0.240": 7}))
NOTES[c.n] = ("**과제 3.** 동료가 일곱 변수를 하나씩 넣은 단변수 로지스틱 회귀에서 P < 0.05인 변수만 골라 다변수 모형을 "
              "적합하고 \"보정 후 흡연은 저체중아와 관련이 없었다(OR 1.65, 95% CI 0.83–3.26, P = 0.15)\"라고 썼습니다. "
              "동료의 분석을 재현하고, 이 결론에서 틀린 곳 두 가지를 찾아 왜 틀렸는지 설명하세요.")

print("lab09: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 9. 로지스틱 회귀분석",
    intro=("사회약학 연구방법 노트의 '실습 9. 로지스틱 회귀분석'에 딸린 노트북입니다. 앞부분(가–다)은 MASS 패키지의 "
           "birthwt 자료(출생아와 산모 189명)로 저체중아 출산과 산모 특성의 관계를 다변수 로지스틱 회귀로 분석하고 "
           "모형을 점검합니다. 뒷부분(라)은 scikit-learn의 위스콘신 진단 유방암 자료로 예측모형을 만들고 판별과 교정을 "
           "평가합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 9에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    ns = nb.ns
    bw_, m_ = ns["bw"], ns["logit_m"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r2 = lambda x: f"{x:.2f}"
    # 글(content/lab09.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 3: "결과변수와 설명변수 코딩", 5: "다변수 로지스틱 회귀 적합",
             6: "오즈비와 95% 신뢰구간 표", 7: "임상적으로 의미 있는 단위의 오즈비", 8: "보정 전후 오즈비 비교",
             11: "모형 기반 표준화로 보정 상대위험도 구하기", 12: "결과변수를 0/1이 아닌 값으로 넣으면",
             14: "직선 모형과 스플라인 모형의 로그 오즈", 15: "영향력이 큰 관측값 찾기"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    # 대응표 (이 실습의 자료 열)
    same("N", len(bw_), 189)
    same("events, non-events", (int(bw_["low"].sum()), int((bw_["low"] == 0).sum())), (59, 130))
    same("event proportion", f"{bw_['low'].mean():.1%}", "31.2%")
    same("smokers, non-smokers", (int(bw_["smoke"].sum()), int((bw_["smoke"] == 0).sum())), (74, 115))
    same("coefficients (k)", int(m_.df_model), 8)
    same("EPV", f"{59 / 8:.1f}", "7.4")
    same("crude smoke OR", r2(ns["odds"][1] / ns["odds"][0]), "2.02")
    same("adjusted smoke OR", r2(np.exp(m_.params["smoke"])), "2.33")
    same("standardized RR", r2(ns["r1"] / ns["r0"]), "1.60")
    # 다중공선성: VIF 셀의 출력과 '직접 바꿔 보기'의 숫자
    v_ = ns["vif"].drop("Intercept")
    same("max VIF", (v_.idxmax(), r2(v_.max())), ("C(race)[T.other]", "1.35"))
    same("min VIF", (v_.idxmin(), r2(v_.min())), ("ui", "1.07"))
    same("sqrt(max VIF)", r2(np.sqrt(v_.max())), "1.16")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_pp = smf.logit("low ~ age + lwt + C(race) + smoke + ptd + ptl + ht + ui", data=bw_).fit(disp=0)
    X_pp = m_pp.model.exog
    v_pp = pd.Series([variance_inflation_factor(X_pp, j) for j in range(X_pp.shape[1])], index=m_pp.model.exog_names)
    same("VIF ptd, ptl", (r2(v_pp["ptd"]), r2(v_pp["ptl"])), ("6.42", "6.46"))
    same("corr ptd, ptl", r2(np.corrcoef(bw_["ptd"], bw_["ptl"])[0, 1]), "0.92")
    same("SE ptd without/with ptl", (f"{m_.bse['ptd']:.3f}", f"{m_pp.bse['ptd']:.3f}"), ("0.463", "1.202"))
    # 상수항 열을 빼고 넘겼을 때(statsmodels 0.14 이하의 표준화 없는 계산을 그대로 흉내 냄)
    from statsmodels.regression.linear_model import OLS

    def vif_raw(exog, j):
        mask = np.arange(exog.shape[1]) != j
        return 1 / (1 - OLS(exog[:, j], exog[:, mask]).fit().rsquared)
    Xn = ns["Xd"][:, 1:]
    nm = ns["names"][1:]
    raw = dict(zip(nm, [vif_raw(Xn, j) for j in range(Xn.shape[1])]))
    same("VIF without constant, old method (age, lwt)", (f"{raw['age']:.0f}", f"{raw['lwt']:.0f}"), ("14", "15"))
    same("VIF with constant, old method = new", r2(vif_raw(ns["Xd"], nm.index("C(race)[T.other]") + 1)), "1.35")
    same("VIF of the constant column: old method, new", (f"{vif_raw(ns['Xd'], 0):.0f}", r2(ns["vif"]["Intercept"])),
         ("44", "1.00"))
    # 과제 1
    ct = pd.crosstab(bw_["ht"], bw_["low"])
    same("ht mothers, events", (int(ct.loc[1].sum()), int(ct.loc[1, 1])), (12, 7))
    same("ht OR", [r2(x) for x in ns["or_tab"].loc["ht", ["OR", "lower", "upper"]]], ["6.29", "1.58", "24.95"])
    same("ht std risks", (f"{ns['r_ht1']:.1%}", f"{ns['r_ht0']:.1%}", r2(ns["r_ht1"] / ns["r_ht0"])),
         ("64.6%", "28.9%", "2.24"))
    same("ht OR without mother 12", r2(np.exp(ns["m_wo"].params["ht"])), "9.74")
    # 과제 2
    mb = ns["m_b"]
    same("white vs black = 1/black", (r2(np.exp(mb.params["C(race_b)[T.white]"])),
                                       r2(1 / np.exp(m_.params["C(race)[T.black]"]))), ("0.30", "0.30"))
    same("other vs black", f"{np.exp(m_.params['C(race)[T.other]'] - m_.params['C(race)[T.black]']):.3f}", "0.665")
    same("other/black from table", f"{2.235 / 3.363:.3f}", "0.665")
    same("lwt10 = lwt^10", (f"{np.exp(mb.params['lwt10']):.3f}", f"{np.exp(10 * m_.params['lwt']):.3f}"),
         ("0.861", "0.861"))
    same("intercept ratio", r2(np.exp(mb.params["Intercept"]) / np.exp(m_.params["Intercept"])), "3.36")
    # 과제 2 풀이의 white 산모 기준 오즈: 원래 e^0.6369, 새 모형 6.358 × 0.297 (표에 보이는 반올림 값)
    same("white baseline odds, original", (f"{m_.params['Intercept']:.4f}", r2(np.exp(m_.params["Intercept"]))),
         ("0.6369", "1.89"))
    same("white baseline odds, new (rounded table values)", (f"{np.exp(mb.params['Intercept']):.3f}",
         f"{np.exp(mb.params['C(race_b)[T.white]']):.3f}", r2(6.358 * 0.297)), ("6.358", "0.297", "1.89"))
    for v in ("smoke", "ptd", "ht", "ui", "age"):
        assert abs(mb.params[v] - m_.params[v]) < 1e-6, v
        n_ok += 1
    # 과제 3
    same("selected terms", ns["keep"], ["lwt", "smoke", "ptd", "ht", "ui"])
    ms = ns["m_sel"]
    lo, hi = np.exp(ms.conf_int().loc["smoke"])
    same("selected smoke OR (CI, P)", (r2(np.exp(ms.params["smoke"])), r2(lo), r2(hi), r2(ms.pvalues["smoke"])),
         ("1.65", "0.83", "3.26", "0.15"))
    same("smoking by race", [f"{x:.1%}" for x in pd.crosstab(bw_["race"], bw_["smoke"], normalize="index")[1]],
         ["54.2%", "38.5%", "17.9%"])
    same("race ORs", (r2(np.exp(m_.params["C(race)[T.black]"])), r2(np.exp(m_.params["C(race)[T.other]"]))),
         ("3.36", "2.23"))
    same("low birth weight by race", [f"{x:.1%}" for x in bw_.groupby("race", observed=True)["low"].mean()],
         ["24.0%", "42.3%", "37.3%"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_nr = smf.logit("low ~ age + lwt + smoke + ptd + ht + ui", data=bw_).fit(disp=0)
    same("smoke OR without race only (the change comes from race)", r2(np.exp(m_nr.params["smoke"])), "1.62")
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab09.html passed")
