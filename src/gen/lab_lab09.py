"""실습 9 (로지스틱 회귀분석) — real cells run with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab09.py
Writes figs/lab09_*.html (insert in content/lab09.html with <!--FIG:lab09_xxx-->).
Part 1: MASS birthwt (Rdatasets URL -> offline copy via labkit). Part 2: sklearn load_breast_cancer.
"""
import html as _html
import os
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


def H(c, marks=None, lines=None):
    f = nb.html(c)
    if marks:
        f = mk(f, marks)
    if lines:
        f = eol(f, lines)
    return f


nb = Notebook("lab09")
S = nb.save_fragment

# ================================================================ 가. 실습 데이터 준비
c = nb.cell('''
import pandas as pd
import numpy as np

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

print("lab09: cells", len(nb.cells))
