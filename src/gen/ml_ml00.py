"""머신러닝 기초 0장 · 준비 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_ml.py && python3 gen/ml_ml00.py
      (NOMARK=1 python3 gen/ml_ml00.py 로 돌리면 표식 없이 모든 셀의 출력을 화면에 찍는다)

자료는 gen/data_ml.py가 만든 pub/data/ml_claims.csv(과목 전체의 공통 자료)이고, 이 장에서 정한 분할
(test_size=0.25, stratify=y, random_state=2026)을 이후 모든 장이 그대로 쓴다. 끝의 대조 블록에서 본문에
적은 핵심 숫자를 실행 결과와 맞춘다.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("ml00")
NOMARK = bool(os.environ.get("NOMARK"))


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
    if NOMARK:
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        for w in c.warns:
            print("WARN:", w)
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


# ================================================================ 가. 지도학습이란
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
df = pd.read_csv(BASE + "ml_claims.csv")
print(df.shape)
df.head(3).T                     # 앞 3명을 세로로 돌려 보기
''', title="공통 자료 불러오기", max_rows=12)
save("ml00_load", c, marks={"(15000, 42)": 1}, dfmarks={"NaN": 2})

c = nb.cell('''
print(df["admit_2023"].value_counts())
print("admission rate:", round(df["admit_2023"].mean(), 4))
print(df["cost_2023"].describe().round(1))
print("zero cost:", round((df["cost_2023"] == 0).mean(), 3))

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(df["cost_2023"], bins=np.arange(0, 3001, 50))
ax.set_xlabel("cost_2023 (10,000 KRW), shown up to 3,000")
ax.set_ylabel("Number of people")
plt.tight_layout()
''', title="결과 두 개의 분포")
save("ml00_outcomes", c, marks={"1      612": 1, "admission rate: 0.0408": 2, "mean       284.8": 3,
                                  "50%        116.3": 4, "max      35695.3": 5, "zero cost: 0.029": 6})

c = nb.cell('''
miss = df.isna().mean().round(3)          # 열마다 결측 비율
print(miss[miss > 0])

age_grp = pd.cut(df["age"], [39, 64, 79, 99])
print(df["bmi"].isna().groupby(age_grp, observed=True)
      .mean().round(3))
''', title="결측이 있는 열과 나이별 검진 결측")
save("ml00_missing", c, marks={"bmi        0.307": 1, "alcohol    0.324": 2, "(79, 99]    0.576": 3})

c = nb.cell('''
y = df["admit_2023"]                                  # 목표
X = df.drop(columns=["id", "admit_2023", "cost_2023"])  # 특성
print(X.shape, y.shape)
print(X.dtypes.value_counts())
''', title="특성 X와 목표 y")
save("ml00_xy", c, marks={"(15000, 39) (15000,)": 1, "str         4": 2})

c = nb.cell('''
from sklearn.linear_model import LogisticRegression

cols3 = ["age", "n_ed", "n_drugs"]       # 결측이 없는 특성 3개
model = LogisticRegression()             # (1) 모형을 고른다
model.fit(X[cols3], y)                   # (2) 정답이 있는 자료로 배운다

new = pd.DataFrame({"age": [55, 85], "n_ed": [0, 2],
                    "n_drugs": [3, 11]})  # 새로운 두 사람
print(model.predict(new))                # (3) 예측: 0 또는 1
print(model.predict_proba(new).round(3)) #     예측: 확률
''', title="fit과 predict")
save("ml00_fit", c, marks={"[0 0]": 1, "[[0.978 0.022]": 2, "[0.64  0.36 ]": 3})

# ================================================================ 나. 자료 나누기
c = nb.cell('''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=2026)
print(X_train.shape, X_test.shape)
print("events:", y_train.sum(), y_test.sum())
print("rate  :", round(y_train.mean(), 4), round(y_test.mean(), 4))
''', title="이 과목의 분할 (모든 장이 같은 분할)")
save("ml00_split", c, marks={"(11250, 39) (3750, 39)": 1, "events: 459 153": 2, "rate  : 0.0408 0.0408": 3})

c = nb.cell('''
for seed in [1, 2, 3, 4, 5]:
    _, _, _, yt = train_test_split(X, y, test_size=0.25,
                                   random_state=seed)
    print("seed", seed, "events in test:", yt.sum(),
          " rate:", round(yt.mean(), 4))
''', title="stratify 없이 나누면")
save("ml00_nostrat", c, marks={"seed 5 events in test: 138": 1, "seed 4 events in test: 160": 2})

c = nb.cell('''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

cols = ["age", "female", "n_outpt", "n_ed", "n_admit",
        "n_drugs", "cost_2022"]
tree = DecisionTreeClassifier(random_state=0)   # 깊이 제한 없음
tree.fit(X_train[cols], y_train)

p_tr = tree.predict_proba(X_train[cols])[:, 1]
p_te = tree.predict_proba(X_test[cols])[:, 1]
print("train AUC:", round(roc_auc_score(y_train, p_tr), 3))
print("test AUC :", round(roc_auc_score(y_test, p_te), 3))
print("leaves:", tree.get_n_leaves())
''', title="배운 자료로 시험하면")
save("ml00_overfit", c, marks={"train AUC: 1.0": 1, "test AUC : 0.558": 2, "leaves: 838": 3})

# ================================================================ 다. 전처리와 파이프라인
c = nb.cell('''
cat_cols = ["insurance", "region", "smoking", "alcohol"]
num_cols = [c for c in X.columns if c not in cat_cols]
print(len(num_cols), "numeric,", len(cat_cols), "categorical")
X_train[cat_cols].head(4)
''', title="숫자 열과 범주 열")
save("ml00_cols", c, marks={"35 numeric, 4 categorical": 1}, dfmarks={"NaN": 2})

c = nb.cell('''
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median")
imp.fit(X_train[["bmi", "egfr"]])        # 학습 자료에서 중앙값 계산
print("medians:", imp.statistics_)

four = X_train[["bmi", "egfr"]].iloc[:4]
print(four)
print(imp.transform(four))               # 결측을 중앙값으로 채움
''', title="결측 대체 (SimpleImputer)")
save("ml00_impute", c, marks={"medians: [24.4 89. ]": 1, "632     NaN    NaN": 2, "[ 24.4  89. ]": 3})

c = nb.cell('''
from sklearn.preprocessing import OneHotEncoder

smk = X_train[["smoking"]].fillna("missing")   # 결측도 한 범주로
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe.fit(smk)
print(ohe.get_feature_names_out())
print(smk.head(4))
print(ohe.transform(smk.head(4)))
''', title="원-핫 인코딩 (OneHotEncoder)")
save("ml00_onehot", c, marks={"'smoking_missing'": 1, "632    missing": 2, "[0. 0. 1. 0.]": 3})

c = nb.cell('''
from sklearn.preprocessing import StandardScaler

sc = StandardScaler().fit(X_train[["age", "n_outpt"]])
print("mean:", sc.mean_.round(2), " SD:", sc.scale_.round(2))
three = X_train[["age", "n_outpt"]].iloc[:3]
print(three)
print(sc.transform(three).round(2))
''', title="표준화 (StandardScaler)")
save("ml00_scale", c, marks={"mean: [59.01 18.17]": 1, "SD: [12.94 23.72]": 2, "[[-1.16 -0.72]": 3})

c = nb.cell('''
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def make_pipe(model, num=num_cols, cat=cat_cols):
    """전처리(결측 대체, 표준화, 원-핫)와 모형을 묶습니다."""
    num_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())])
    cat_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant",
                                 fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False))])
    prep = ColumnTransformer([("num", num_steps, num),
                              ("cat", cat_steps, cat)])
    return Pipeline([("prep", prep), ("model", model)])

pipe = make_pipe(LogisticRegression(max_iter=1000))
pipe.fit(X_train, y_train)
Z = pipe.named_steps["prep"].transform(X_test)
print("after prep:", Z.shape)
p_test = pipe.predict_proba(X_test)[:, 1]
print("test AUC:", round(roc_auc_score(y_test, p_test), 3))
''', title="ColumnTransformer와 Pipeline")
save("ml00_pipe", c, marks={"after prep: (3750, 48)": 1, "test AUC: 0.74": 2})

c = nb.cell('''
raw = Pipeline([("impute", SimpleImputer(strategy="median")),
                ("model", LogisticRegression())])   # 표준화 없음
raw.fit(X_train[num_cols], y_train)
print("iterations, no scaling:", raw.named_steps["model"].n_iter_)
print("iterations, scaled    :", pipe.named_steps["model"].n_iter_)
''', title="표준화하지 않으면")
save("ml00_converge", c, marks={"no scaling: [100]": 1, "scaled    : [33]": 2})

c = nb.cell('''
rng = np.random.default_rng(0)
# 2023년(결과 기간)의 응급실 방문 수를 흉내 낸 열. 실제로는 쓰면 안 됨
ed_2023 = np.where(y == 1, rng.poisson(1.5, len(y)) + 1,
                   rng.poisson(0.08, len(y)))
X_leak = X.assign(n_ed_2023=ed_2023)

XL_train, XL_test = train_test_split(
    X_leak, test_size=0.25, stratify=y, random_state=2026)
leaky = make_pipe(LogisticRegression(max_iter=1000),
                  num=num_cols + ["n_ed_2023"])
leaky.fit(XL_train, y_train)
p_leak = leaky.predict_proba(XL_test)[:, 1]
print("test AUC with n_ed_2023:", round(roc_auc_score(y_test, p_leak), 3))
''', title="자료 누수: 결과 뒤의 정보가 섞이면")
save("ml00_leak", c, marks={"0.996": 1})

# ================================================================ 라. 교차검증과 조절값 탐색
c = nb.cell('''
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
for k, (tr, va) in enumerate(skf.split(X_train, y_train), start=1):
    print("fold", k, " train:", len(tr), " valid:", len(va),
          " events in valid:", y_train.iloc[va].sum())
''', title="5겹으로 나누기 (StratifiedKFold)")
save("ml00_kfold", c, marks={"train: 9000  valid: 2250": 1, "events in valid: 91": 2})

c = nb.cell('''
from sklearn.model_selection import cross_val_score

auc5 = cross_val_score(make_pipe(LogisticRegression(max_iter=1000)),
                       X_train, y_train, cv=skf, scoring="roc_auc")
print(auc5.round(3))
print("mean:", round(auc5.mean(), 3), " SD:", round(auc5.std(), 3))
''', title="교차검증 AUC (cross_val_score)")
save("ml00_cv", c, marks={"[0.739 0.787 0.756 0.727 0.732]": 1, "mean: 0.748  SD: 0.022": 2})

c = nb.cell('''
from sklearn.model_selection import GridSearchCV

grid = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)),
                  grid, cv=skf, scoring="roc_auc")
gs.fit(X_train, y_train)
print("best:", gs.best_params_, round(gs.best_score_, 4))

res = pd.DataFrame(gs.cv_results_)
show = ["param_model__C", "mean_test_score", "std_test_score",
        "rank_test_score"]
res[show].round(4)
''', title="조절값 C 고르기 (GridSearchCV)")
save("ml00_grid", c, marks={"best: {'model__C': 0.1} 0.7486": 1},
     dfmarks={"0.7415": 2, "0.7480": 3, "0.0220": 4})

c = nb.cell('''
final = gs.best_estimator_      # 고른 C로 학습 자료 전체에 다시 맞춘 모형
p_final = final.predict_proba(X_test)[:, 1]
print("test AUC:", round(roc_auc_score(y_test, p_final), 3))
''', title="시험 자료로 한 번 평가")
save("ml00_final", c, marks={"test AUC: 0.742": 1})

# ================================================================ 마. 성능을 재는 지표
c = nb.cell('''
from sklearn.metrics import accuracy_score

all0 = np.zeros(len(y_test), dtype=int)     # 모두 '입원 안 함'
pred05 = final.predict(X_test)              # 임계값 0.5
print("accuracy, all 0  :", round(accuracy_score(y_test, all0), 4))
print("accuracy, model  :", round(accuracy_score(y_test, pred05), 4))
print("predicted 1 (0.5):", pred05.sum())
''', title="정확도의 함정")
save("ml00_acc", c, marks={"accuracy, all 0  : 0.9592": 1, "accuracy, model  : 0.9632": 2,
                              "predicted 1 (0.5): 29": 3})

c = nb.cell('''
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)

pred10 = (p_final >= 0.10).astype(int)      # 임계값 0.10
print(confusion_matrix(y_test, pred10))
print("precision:", round(precision_score(y_test, pred10), 3))
print("recall   :", round(recall_score(y_test, pred10), 3))
print("F1       :", round(f1_score(y_test, pred10), 3))
''', title="혼동행렬과 정밀도·재현율·F1")
save("ml00_cm", c, marks={"[[3419  178]": 1, "[ 100   53]]": 2, "precision: 0.229": 3, "recall   : 0.346": 4,
                             "F1       : 0.276": 5})

c = nb.cell('''
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             average_precision_score)

fpr, tpr, _ = roc_curve(y_test, p_final)
prec, rec, _ = precision_recall_curve(y_test, p_final)
print("AUC:", round(roc_auc_score(y_test, p_final), 3),
      " AP:", round(average_precision_score(y_test, p_final), 3))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
ax[0].plot(fpr, tpr)
ax[0].plot([0, 1], [0, 1], "--", color="gray")
ax[0].set_xlabel("1 - specificity")
ax[0].set_ylabel("Sensitivity (recall)")
ax[0].set_title("ROC curve")
ax[1].plot(rec, prec)
ax[1].axhline(y_test.mean(), ls="--", color="gray")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].set_title("Precision-recall curve")
plt.tight_layout()
''', title="ROC 곡선과 정밀도-재현율 곡선")
save("ml00_curves", c, marks={"AUC: 0.742": 1, "AP: 0.268": 2})

c = nb.cell('''
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)

c_train = df.loc[X_train.index, "cost_2023"]    # 같은 사람들의 비용
c_test = df.loc[X_test.index, "cost_2023"]
reg = make_pipe(LinearRegression())
reg.fit(X_train, c_train)
c_pred = reg.predict(X_test)

print("MAE :", round(mean_absolute_error(c_test, c_pred), 1))
print("RMSE:", round(np.sqrt(mean_squared_error(c_test, c_pred)), 1))
print("R2  :", round(r2_score(c_test, c_pred), 3))
print("negative predictions:", (c_pred < 0).sum())
''', title="회귀: 2023년 의료비 예측")
save("ml00_reg", c, marks={"MAE : 183.0": 1, "RMSE: 430.1": 2, "R2  : 0.445": 3, "negative predictions: 196": 4})

c = nb.cell('''
err = (c_test - c_pred).to_numpy()
top = np.argsort(np.abs(err))[-10:]        # 오차가 가장 큰 10명
rest = np.setdiff1d(np.arange(len(err)), top)
mae = lambda e: np.abs(e).mean()
rmse = lambda e: np.sqrt((e ** 2).mean())
print("MAE  all / without top 10:",
      round(mae(err), 1), round(mae(err[rest]), 1))
print("RMSE all / without top 10:",
      round(rmse(err), 1), round(rmse(err[rest]), 1))
print("share of squared error from top 10:",
      round((err[top] ** 2).sum() / (err ** 2).sum(), 3))
''', title="치우친 비용에서 MAE와 RMSE")
save("ml00_maerms", c, marks={"183.0 170.4": 1, "430.1 324.5": 2, "0.432": 3})

# ================================================================ 바. 과제
c = nb.cell('''
comorb = ["htn", "dm", "dyslip", "ckd", "hf", "af", "cad", "stroke",
          "copd", "asthma", "depress", "dementia", "cancer",
          "liver", "oa"]
drugs = ["anticoag", "insulin", "opioid", "benzo", "antipsych",
         "diuretic"]
small = make_pipe(LogisticRegression(max_iter=1000),
                  num=["age", "female", "income_q"] + comorb + drugs,
                  cat=["insurance", "region"])
full = make_pipe(LogisticRegression(max_iter=1000))
for name, p in [("dx + drugs only", small), ("all features", full)]:
    s = cross_val_score(p, X_train, y_train, cv=skf,
                        scoring="roc_auc")
    print(f"{name:16s} CV AUC {s.mean():.3f} (SD {s.std():.3f})")
''', title="과제 1 정답. 특성 묶음 바꾸기")
save("ml00_hw1", c)

c = nb.cell('''
for k in [3, 5, 10]:
    cv_k = StratifiedKFold(n_splits=k, shuffle=True,
                           random_state=2026)
    s = cross_val_score(make_pipe(LogisticRegression(max_iter=1000)),
                        X_train, y_train, cv=cv_k, scoring="roc_auc")
    print(f"k={k:2d}  train per fold {len(X_train) * (k - 1) // k:5d}"
          f"  mean {s.mean():.3f}  SD {s.std():.3f}"
          f"  min {s.min():.3f}  max {s.max():.3f}")
''', title="과제 2 정답. k 바꾸기")
save("ml00_hw2", c)

c = nb.cell('''
from sklearn.compose import TransformedTargetRegressor

skewed = ["n_outpt", "los_days", "cost_2022"]
Xtr_log, Xte_log = X_train.copy(), X_test.copy()
for col in skewed:                       # 치우친 특성도 log(1 + x)로
    Xtr_log[col] = np.log1p(Xtr_log[col])
    Xte_log[col] = np.log1p(Xte_log[col])

log_model = TransformedTargetRegressor(
    regressor=LinearRegression(),
    func=np.log1p, inverse_func=np.expm1)    # 목표도 log(1 + 비용)
reg_log = make_pipe(log_model)
reg_log.fit(Xtr_log, c_train)
c_pred_log = reg_log.predict(Xte_log)        # 만원 단위로 되돌린 예측

for name, pr in [("raw", c_pred), ("log", c_pred_log)]:
    print(f"{name}  MAE {mean_absolute_error(c_test, pr):5.1f}"
          f"  RMSE {np.sqrt(mean_squared_error(c_test, pr)):5.1f}"
          f"  R2 {r2_score(c_test, pr):.3f}"
          f"  mean of predictions {pr.mean():5.1f}")
print("mean of actual cost:", round(c_test.mean(), 1))
''', title="과제 3 정답. 로그 변환한 목표")
save("ml00_hw3", c)

print("ml00: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)


# ================================================================ 그림 0-1: 자료 나누기와 5겹 교차검증 (SVG)
def kfold_figure(fold_auc, mean_auc):
    from html import escape
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from svgplot import figure
    W, H = 640, 300
    X0, X1 = 118, 548
    xt = X0 + 0.75 * (X1 - X0)                     # 학습 75% | 시험 25%
    o = []

    def rect(x0, x1, y, h, cls, label=None, lcls="lbl small"):
        o.append(f'<rect x="{x0:.1f}" y="{y}" width="{x1 - x0:.1f}" height="{h}" rx="3" class="{cls}" stroke-width="1"/>')
        if label:
            o.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" class="{lcls}">{escape(label)}</text>')

    def text(x, y, s, anchor="start", cls="lbl"):
        o.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{escape(s)}</text>')

    text(X0 - 10, 37, "All 15,000", "end", "lbl strong")
    rect(X0, xt, 22, 24, "a1 s1", "Training set 11,250 (459 events)")
    rect(xt, X1, 22, 24, "a2 s2", "Test set 3,750")
    o.append(f'<line x1="{X0}" y1="58" x2="{xt:.1f}" y2="58" class="ref" stroke-width="1"/>')
    text(X0, 76, "5-fold cross-validation inside the training set", "start", "lbl mute small")
    seg = (xt - X0) / 5
    for k in range(5):
        y = 88 + k * 30
        text(X0 - 10, y + 15, f"Round {k + 1}", "end")
        for j in range(5):
            a, b = X0 + j * seg, X0 + (j + 1) * seg
            if j == k:
                rect(a + 1, b - 1, y, 20, "f3", "valid", "lbl small strong")
            else:
                rect(a + 1, b - 1, y, 20, "a1 s1", "train" if k == 0 else None)
        rect(xt + 1, X1, y, 20, "a4", "not used" if k == 0 else None, "lbl mute small")
        text(X1 + 10, y + 15, f"AUC {fold_auc[k]:.3f}", "start", "lbl small")
    text(X1 + 10, 250, f"mean {mean_auc:.3f}", "start", "lbl strong")
    o.append(f'<line x1="{X1 + 8}" y1="236" x2="{W - 8}" y2="236" class="ref" stroke-width="1"/>')
    text(X0, 276, "Choose C by the mean, refit on all 11,250, then evaluate once on the test set",
         "start", "lbl mute small")
    svg = (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="자료 나누기와 5겹 교차검증" '
           f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')
    cap = ("그림 0-1. 이 과목의 자료 나누기와 5겹 교차검증. 시험 자료(주황)는 처음에 떼어 두고, 학습 자료를 다섯 겹으로 "
           "나눠 한 겹씩 돌아가며 검증 자료(초록)로 씁니다. 오른쪽 숫자는 셀 17에서 C = 1인 로지스틱 회귀로 구한 "
           "겹별 AUC입니다.")
    return figure(svg, cap)


ns = nb.ns
nb.save_fragment("ml00_fig_kfold", kfold_figure(ns["auc5"], ns["auc5"].mean()))

# ================================================================ notebook
nb.save_ipynb(
    "머신러닝 기초 0장. 준비",
    intro=("사회약학 연구방법 노트 '머신러닝 기초' 0장을 따라 하는 노트북입니다. 과목 전체가 쓰는 가상 청구자료"
           "(40세 이상 15,000명)를 불러와 학습·시험 자료로 나누고, 전처리와 모형을 파이프라인으로 묶고, 교차검증으로 "
           "조절값을 고른 뒤 분류(2023년 응급 입원)와 회귀(2023년 의료비)의 성능 지표를 구합니다. 셀을 위에서부터 "
           "차례로 실행하세요. 자료는 사이트가 만든 가상 자료이며 실제 환자 자료가 아닙니다."),
    notes={
        1: "## 가. 지도학습이란\n\n자료를 불러와 특성(X)과 목표(y)를 정하고 fit → predict를 한 번 해 봅니다.",
        6: "## 나. 자료 나누기\n\n이 과목의 모든 장이 셀 6의 분할을 그대로 씁니다.",
        9: "## 다. 전처리와 파이프라인\n\n결측 대체, 원-핫 인코딩, 표준화를 학습 자료로만 맞추고 모형과 묶습니다.",
        16: "## 라. 교차검증과 조절값 탐색\n\n조절값은 학습 자료 안의 교차검증으로 고르고, 시험 자료는 마지막에 한 번만 씁니다.",
        20: "## 마. 성능을 재는 지표\n\n분류는 정확도, 혼동행렬, 정밀도·재현율·F1, ROC와 정밀도-재현율 곡선, 회귀는 MAE, RMSE, R²를 봅니다.",
        25: ("## 과제 정답\n\n**과제 1.** 진단과 처방(그리고 나이, 성별, 소득, 보험, 지역)만으로 만든 파이프라인과 "
             "모든 특성을 쓴 파이프라인의 교차검증 AUC를 비교합니다."),
        26: "**과제 2.** k를 3, 5, 10으로 바꿔 교차검증 AUC의 평균과 겹 사이 SD를 비교합니다.",
        27: ("**과제 3.** 목표(cost_2023)와 치우친 특성 세 개를 log(1 + x)로 바꿔 선형회귀를 다시 맞추고, "
             "원래 단위의 MAE, RMSE, R²와 예측 평균을 비교합니다."),
    })

# ================================================================ 대조 블록: 본문에 적은 숫자와 실행 결과 맞추기
if __name__ == "__main__":
    import json
    import warnings
    import pandas as pd
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    df_, X_, y_ = ns["df"], ns["X"], ns["y"]
    Xtr, Xte, ytr, yte = ns["X_train"], ns["X_test"], ns["y_train"], ns["y_test"]
    n_ok = 0

    def same(label, a, b, tol=1e-9):
        global n_ok
        a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
        assert a.shape == b.shape and np.all(np.abs(a - b) <= tol), f"{label}: {a} != {b}"
        n_ok += 1

    T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ml_truth.json"), encoding="utf-8"))
    D = T["describe"]
    same("rows, cols", df_.shape, (15000, 42))
    same("events, rate", [y_.sum(), round(y_.mean(), 4)], [D["admit_2023_events"], D["admit_2023_rate"]])
    same("split sizes and events", [len(Xtr), len(Xte), ytr.sum(), yte.sum()],
         [T["check"]["n_train"], T["check"]["n_test"], T["check"]["events_train"], T["check"]["events_test"]])
    same("cost mean, median, max", [round(df_.cost_2023.mean(), 1), round(df_.cost_2023.median(), 1), df_.cost_2023.max()],
         [D["cost_2023"]["mean"], 116.3, D["cost_2023"]["max"]], 0.051)
    same("bmi missing", round(df_.bmi.isna().mean(), 3), 0.307)
    # 나 절: 40명 중 4명이 사건, 10명을 시험 자료로 뽑을 때 사건이 0명일 확률
    p0 = math.comb(36, 10) / math.comb(40, 10)
    same("hypergeometric P(0 events)", round(p0, 3), 0.300)
    same("deep tree train/test AUC", [round(roc_auc_score(ytr, ns["p_tr"]), 3), round(roc_auc_score(yte, ns["p_te"]), 3)],
         [1.0, 0.558])
    same("pipeline AUC (C=1)", round(roc_auc_score(yte, ns["p_test"]), 3), 0.740)
    same("leaky AUC", round(roc_auc_score(yte, ns["p_leak"]), 3), 0.996)
    assert ns["XL_train"].index.equals(Xtr.index) and ns["XL_test"].index.equals(Xte.index)   # 같은 분할
    n_ok += 1
    same("CV mean, SD", [round(ns["auc5"].mean(), 3), round(ns["auc5"].std(), 3)], [0.748, 0.022])
    same("best C", ns["gs"].best_params_["model__C"], 0.1)
    same("final test AUC", round(roc_auc_score(yte, ns["p_final"]), 3), 0.742)
    tn, fp, fn, tp = ns["confusion_matrix"](yte, ns["pred10"]).ravel()
    same("confusion matrix", [tn, fp, fn, tp], [3419, 178, 100, 53])
    same("precision, recall, F1, specificity", [round(tp / (tp + fp), 3), round(tp / (tp + fn), 3),
                                                round(2 * tp / (2 * tp + fp + fn), 3), round(tn / (tn + fp), 3)],
         [0.229, 0.346, 0.276, 0.951])
    same("all-0 accuracy", round((yte == 0).mean(), 4), 0.9592)
    same("regression MAE, RMSE, R2", [round(ns["mae"](ns["err"]), 1), round(ns["rmse"](ns["err"]), 1)], [183.0, 430.1])
    same("hw3 log model", [round(ns["c_pred_log"].mean(), 1), round(ns["c_test"].mean(), 1)], [197.6, 281.5])
    # 마 절: 다섯 명의 오차로 MAE와 RMSE
    e5 = np.array([10, 20, 30, 40, 900.0])
    same("toy MAE/RMSE", [np.abs(e5).mean(), round(np.sqrt((e5 ** 2).mean()), 1),
                          np.abs(e5[:4]).mean(), round(np.sqrt((e5[:4] ** 2).mean()), 1)], [200, 403.2, 25, 27.4])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 시험 AUC의 부트스트랩 95% 신뢰구간 (실습 9 라 절과 같은 방법)
        rng = np.random.default_rng(2026)
        yt, pf, aucs = yte.to_numpy(), ns["p_final"], []
        for _ in range(1000):
            i = rng.integers(0, len(yt), len(yt))
            aucs.append(roc_auc_score(yt[i], pf[i]))
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        print(f"VERIFY bootstrap 95% CI of test AUC: {lo:.3f}-{hi:.3f}")
        same("bootstrap CI (3 decimals)", [round(lo, 3), round(hi, 3)], [0.695, 0.785])
        # 다 절: 전체 자료로 전처리를 맞춘 뒤 나누면 (두 번째 누수 유형)
        mp, LR = ns["make_pipe"], ns["LogisticRegression"]
        prep_all = mp(LR(max_iter=1000)).named_steps["prep"].fit(X_)
        m_all = LR(max_iter=1000).fit(prep_all.transform(Xtr), ytr)
        auc_all = roc_auc_score(yte, m_all.predict_proba(prep_all.transform(Xte))[:, 1])
        print(f"VERIFY preprocessing fitted on all data: test AUC {auc_all:.4f} vs {roc_auc_score(yte, ns['p_test']):.4f}")
        same("prep-on-all AUC", [round(auc_all, 4), round(roc_auc_score(yte, ns["p_test"]), 4)], [0.7398, 0.7396])
        med_all, med_tr = X_["bmi"].median(), Xtr["bmi"].median()
        print("VERIFY bmi median all / train:", med_all, med_tr, " egfr:", X_["egfr"].median(), Xtr["egfr"].median())
        # 다 절 팁: 결측 표시 열을 더하면 (add_indicator=True)
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        p_ind = mp(LR(max_iter=1000))
        p_ind.named_steps["prep"].transformers[0][1].steps[0] = ("impute", SimpleImputer(strategy="median", add_indicator=True))
        s_ind = cross_val_score(p_ind, Xtr, ytr, cv=ns["skf"], scoring="roc_auc")
        print(f"VERIFY add_indicator CV AUC {s_ind.mean():.4f} (SD {s_ind.std():.4f}) vs {ns['auc5'].mean():.4f}")
        # 라 절: 시험 자료로 C를 고르면 (하지 말아야 할 일)
        test_auc_by_C = {}
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            m = mp(LR(C=C, max_iter=1000)).fit(Xtr, ytr)
            test_auc_by_C[C] = round(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]), 4)
        print("VERIFY test AUC by C:", test_auc_by_C)
        same("C chosen on the test set (C, AUC)", [max(test_auc_by_C, key=test_auc_by_C.get), round(test_auc_by_C[0.01], 3)], [0.01, 0.746])
        print("VERIFY grid mean CV:", dict(zip(ns["res"]["param_model__C"].astype(float), ns["res"]["mean_test_score"].round(4))))
        # 가 절: 응급실 방문 2회 이상과 미만의 입원율 (학습 자료)
        g2 = Xtr["n_ed"] >= 2
        print(f"VERIFY train n_ed>=2: n={g2.sum()}, rate={ytr[g2].mean():.3f}; <2: n={(~g2).sum()}, rate={ytr[~g2].mean():.3f}")
        print(f"VERIFY high-risk share at 0.10: {(ns['pred10'] == 1).sum()} ({(ns['pred10'] == 1).mean():.3f})")
        print(f"VERIFY negative predictions share: {(ns['c_pred'] < 0).mean():.3f}; max pred {ns['c_pred'].max():.1f}")
        print(f"VERIFY log-target without log features:")
        from sklearn.compose import TransformedTargetRegressor
        r0 = mp(TransformedTargetRegressor(regressor=ns["LinearRegression"](), func=np.log1p, inverse_func=np.expm1))
        r0.fit(Xtr, ns["c_train"])
        pr0 = r0.predict(Xte)
        print(f"   max pred {pr0.max():.1f}, R2 {ns['r2_score'](ns['c_test'], pr0):.1f}")
        same("log-target w/o log features (max pred, R2)", [round(pr0.max() / 10000, 1), round(ns["r2_score"](ns["c_test"], pr0))],
             [56.6, -256])
        print("VERIFY top-10 actual costs in test:", np.sort(ns["c_test"].to_numpy())[-3:], " largest |error|:",
              np.round(np.sort(np.abs(ns["err"]))[-3:], 1))
        print("VERIFY test cost max:", ns["c_test"].max(), " in train:", ns["c_train"].max())
        # 정밀도-재현율 곡선 기준선
        same("PR baseline (prevalence)", round(yte.mean(), 3), 0.041)
        print("VERIFY iterations:", ns["raw"].named_steps["model"].n_iter_, ns["pipe"].named_steps["model"].n_iter_)
        same("accuracy at 0.10", round((tp + tn) / len(yte), 3), 0.926)
        same("train n_ed>=2", [g2.sum(), round(ytr[g2].mean(), 3), (~g2).sum(), round(ytr[~g2].mean(), 3)],
             [184, 0.207, 11066, 0.038])
        # 과제 1: 같은 다섯 겹에서 겹마다 비교
        s_small = cross_val_score(ns["small"], Xtr, ytr, cv=ns["skf"], scoring="roc_auc")
        s_full = cross_val_score(ns["full"], Xtr, ytr, cv=ns["skf"], scoring="roc_auc")
        print("VERIFY hw1 per fold small", s_small.round(3), "full", s_full.round(3), "full>small in",
              int((s_full > s_small).sum()), "of 5")
        same("hw1 first/last folds and wins", [round(s_small[0], 3), round(s_full[0], 3), round(s_small[-1], 3),
                                                round(s_full[-1], 3), (s_full > s_small).sum()], [0.722, 0.739, 0.716, 0.732, 5])
        # 조절값 표: class_weight="balanced"
        s_bal = cross_val_score(mp(LR(max_iter=1000, class_weight="balanced")), Xtr, ytr, cv=ns["skf"], scoring="roc_auc")
        m_bal = mp(LR(max_iter=1000, class_weight="balanced")).fit(Xtr, ytr)
        print(f"VERIFY class_weight=balanced CV AUC {s_bal.mean():.4f}; mean predicted prob (train) "
              f"{m_bal.predict_proba(Xtr)[:, 1].mean():.3f} vs default {ns['pipe'].predict_proba(Xtr)[:, 1].mean():.3f}")
        same("class_weight=balanced (CV AUC, mean prob, default mean prob)",
             [round(s_bal.mean(), 3), round(m_bal.predict_proba(Xtr)[:, 1].mean(), 3), round(ns["pipe"].predict_proba(Xtr)[:, 1].mean(), 3)],
             [0.748, 0.393, 0.041])
        # 가 절 흔한 오해: 효과가 없도록 만든 처방 의사 수의 단순 비교 (공개 자료로 계산)
        npres = df_["n_prescribers"]
        r_lo, r_hi = df_.loc[npres <= 2, "admit_2023"].mean(), df_.loc[npres >= 5, "admit_2023"].mean()
        print(f"VERIFY n_prescribers <=2: {r_lo:.3f}  >=5: {r_hi:.3f}  (n>=5: {(npres >= 5).sum()})")
        same("n_prescribers rates", [round(r_lo, 3), round(r_hi, 3)], [0.031, 0.105])
        same("first-four check: age z", round((44 - 59.01) / 12.94, 2), -1.16)
        same("CV folds mean by hand", round(np.mean([0.739, 0.787, 0.756, 0.727, 0.732]), 3), 0.748)

    print(f"VERIFY: {n_ok} checks passed")
