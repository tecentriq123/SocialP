"""머신러닝 기초 1장 · 결정트리 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/ml_ml01.py
      (NOMARK=1 python3 gen/ml_ml01.py 로 돌리면 표식 없이 모든 셀의 출력을 화면에 찍는다)

자료와 분할은 0장(gen/ml_ml00.py)과 같다(pub/data/ml_claims.csv, test_size=0.25, stratify=y, random_state=2026,
StratifiedKFold(5, shuffle=True, random_state=2026)). 끝의 대조 블록에서 본문에 적은 핵심 숫자를 실행 결과와 맞추고,
같은 생성식으로 만든 큰 모의 집단(gen/data_ml.py의 simulate, seed 777)에서 고른 나무와 로지스틱 회귀를 비교한다.
걸린 시간은 실행할 때마다 달라서 화면에 보이는 값만 이 환경에서 잰 값으로 고정한다(본문에 적음).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("ml01")
NOMARK = bool(os.environ.get("NOMARK"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 셀 27(joblib)이 현재 폴더에 파일을 쓰므로 임시 폴더에서 돌린다(그림·노트북은 labkit이 절대 경로로 쓴다).
_TMP = __import__("tempfile").TemporaryDirectory(prefix="ml01_")
os.chdir(_TMP.name)


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


def fix_time(c, measured):
    """걸린 시간 줄을 이 환경에서 잰 값으로 고정 (실행할 때마다 달라 그림 파일이 매번 바뀌지 않게)."""
    c.stdout = re.sub(r"(seconds: )[0-9.]+", rf"\g<1>{measured}", c.stdout)


# ================================================================ 가. 나무가 자료를 나누는 방법
c = nb.cell('''
import numpy as np
import pandas as pd

toy = pd.DataFrame({
    "age":   [52, 61, 47, 78, 83, 69, 88, 58, 74, 81],
    "n_ed":  [0, 0, 1, 2, 0, 0, 3, 1, 2, 0],
    "admit": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]},
    index=list("ABCDEFGHIJ"))           # 0장 가 절의 환자 10명

def gini(y):
    """입원 비율 p에서 지니 불순도 1 - p² - (1 - p)²"""
    p = np.mean(y)
    return 1 - p**2 - (1 - p)**2

def split_gini(col, cut):
    """col <= cut과 col > cut으로 나눈 뒤 불순도의 가중 평균"""
    left = toy["admit"][toy[col] <= cut]
    right = toy["admit"][toy[col] > cut]
    return (len(left) * gini(left)
            + len(right) * gini(right)) / len(toy)

g0 = gini(toy["admit"])
print("before split:", round(g0, 3))
for col, cut in [("n_ed", 1.5), ("age", 76)]:
    g = split_gini(col, cut)
    print(f"{col} <= {cut}: after {g:.3f}, decrease {g0 - g:.3f}")
''', title="지니 불순도와 나누기 전후의 감소량")
save("ml01_gini", c, marks={"before split: 0.42": 1, "after 0.305, decrease 0.115": 2,
                             "after 0.150, decrease 0.270": 3})

c = nb.cell('''
ages = np.sort(toy["age"].unique())
cuts = (ages[:-1] + ages[1:]) / 2        # 이웃한 두 나이의 가운데
rows = [(cut, (toy["age"] <= cut).sum(), g0 - split_gini("age", cut))
        for cut in cuts]
pd.DataFrame(rows, columns=["cut", "n_left", "decrease"]).round(3)
''', title="나이의 모든 기준을 시험하기")
save("ml01_cuts", c, dfmarks={"76.0": 1, "0.270": 2})

c = nb.cell('''
from sklearn.tree import DecisionTreeClassifier, export_text

stump = DecisionTreeClassifier(max_depth=1, random_state=2026)
stump.fit(toy[["age", "n_ed"]], toy["admit"])
print(export_text(stump, feature_names=["age", "n_ed"],
                  show_weights=True))
new = pd.DataFrame({"age": [80], "n_ed": [2]})    # 새 환자 K
print(stump.predict_proba(new))
''', title="사이킷런으로 같은 일 하기")
save("ml01_stump", c, marks={"age <= 76.00": 1, "weights: [1.00, 3.00]": 2, "[[0.25 0.75]]": 3})

# ---------------------------------------------------------------- 0장의 준비를 다시 (가 절 안)
c = nb.cell('''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold, cross_val_score)
from sklearn.metrics import roc_auc_score

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
df = pd.read_csv(BASE + "ml_claims.csv", storage_options=UA)
y = df["admit_2023"]
X = df.drop(columns=["id", "admit_2023", "cost_2023"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=2026)
cat_cols = ["insurance", "region", "smoking", "alcohol"]
num_cols = [c for c in X.columns if c not in cat_cols]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
print(X_train.shape, X_test.shape, y_train.sum(), y_test.sum())
''', title="0장의 준비 (자료, 분할, 교차검증 겹)")
save("ml01_setup", c, marks={"(11250, 39) (3750, 39) 459 153": 1})

c = nb.cell('''
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_pipe(model, num=num_cols, cat=cat_cols):
    """0장 셀 13과 같습니다 (로지스틱 회귀와 비교할 때 씀)."""
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
''', title="0장의 make_pipe")
save("ml01_makepipe", c)

c = nb.cell('''
def make_tree_pipe(model, num=num_cols, cat=cat_cols):
    """나무용: 결측 대체와 원-핫만 하고 표준화는 하지 않습니다."""
    cat_steps = Pipeline([
        ("impute", SimpleImputer(strategy="constant",
                                 fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False))])
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", cat_steps, cat)],
        verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", model)])

prep = make_tree_pipe(DecisionTreeClassifier()).named_steps["prep"]
names = prep.fit(X_train).get_feature_names_out()
print(len(names))
print(names[:3], names[-4:])
''', title="나무용 파이프라인 make_tree_pipe")
save("ml01_treepipe", c, marks={"48": 1, "['age' 'female' 'income_q']": 2,
                                 "'alcohol_missing'": 3})

c = nb.cell('''
small = make_tree_pipe(
    DecisionTreeClassifier(max_depth=3, random_state=2026))
small.fit(X_train, y_train)
tree3 = small.named_steps["model"]
print(export_text(tree3, feature_names=list(names),
                  show_weights=True, decimals=1))
''', title="공통 자료로 깊이 3의 나무")
save("ml01_depth3", c, marks={"age <= 87.5": 1, "age <= 76.5": 2, "n_drugs <= 9.5": 3,
                               "n_ed <= 1.5": 4, "insulin >  0.5": 5})

c = nb.cell('''
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(13, 5.5))
plot_tree(tree3, feature_names=list(names),
          class_names=["no", "admit"], filled=True, impurity=True,
          fontsize=8, ax=ax)
plt.tight_layout()
''', title="나무 그림 (plot_tree)")
save("ml01_plot3", c)

c = nb.cell('''
Z_train = small.named_steps["prep"].transform(X_train)
leaf = tree3.apply(Z_train)                  # 사람마다 도착한 잎 번호
p_train = small.predict_proba(X_train)[:, 1]
by_leaf = pd.DataFrame({"leaf": leaf, "admit": y_train.to_numpy(),
                        "proba": p_train}).groupby("leaf").agg(
    n=("admit", "size"), admit_rate=("admit", "mean"),
    proba=("proba", "first"))
by_leaf.round(4)
''', title="잎의 입원 비율이 곧 예측 확률")
save("ml01_leafprob", c, dfmarks={"9769": 1, "0.0229": 2, "0.8571": 3, "1.0000": 4})

# ================================================================ 나. 나무의 크기와 과적합
c = nb.cell('''
from sklearn.model_selection import cross_validate

rows = []
for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, None]:
    m = make_tree_pipe(DecisionTreeClassifier(max_depth=d,
                                              random_state=2026))
    cv = cross_validate(m, X_train, y_train, cv=skf,
                        scoring="roc_auc", return_train_score=True)
    m.fit(X_train, y_train)
    p_te = m.predict_proba(X_test)[:, 1]    # 현상을 보려고만 계산
    rows.append({"max_depth": str(d),
                 "leaves": m.named_steps["model"].get_n_leaves(),
                 "train": cv["train_score"].mean(),
                 "cv": cv["test_score"].mean(),
                 "cv_sd": cv["test_score"].std(),
                 "test": roc_auc_score(y_test, p_te)})
depth_tab = pd.DataFrame(rows)
depth_tab.round(3)
''', title="깊이를 늘리면", max_rows=12)
DEPTH_CELL = c

c = nb.cell('''
num = depth_tab.iloc[:-1]                    # None을 뺀 10줄
d = num["max_depth"].astype(int)
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(d, num["train"], "o-", label="Training folds")
ax.plot(d, num["cv"], "o-", label="Cross-validation")
ax.plot(d, num["test"], "o--", color="gray", label="Test set")
ax.set_xlabel("max_depth")
ax.set_ylabel("AUC")
ax.set_title("Deeper trees fit the training data better")
ax.legend()
plt.tight_layout()
''', title="깊이별 AUC 그림")
save("ml01_depthfig", c)

c = nb.cell('''
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import recall_score

for cw in [None, "balanced"]:
    m = make_tree_pipe(DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=50, class_weight=cw,
        random_state=2026))
    auc = cross_val_score(m, X_train, y_train, cv=skf,
                          scoring="roc_auc").mean()
    p_oof = cross_val_predict(m, X_train, y_train, cv=skf,
                              method="predict_proba")[:, 1]
    pred = (p_oof >= 0.5).astype(int)
    print(f"{str(cw):8s}  CV AUC {auc:.3f}  mean prob "
          f"{p_oof.mean():.3f}  predicted 1: {pred.sum():4d}  "
          f"recall {recall_score(y_train, pred):.3f}")
''', title="class_weight를 바꾸면")
save("ml01_cw", c, marks={"None      CV AUC 0.718": 1, "mean prob 0.041": 2, "predicted 1:   89": 3,
                           "recall 0.089": 4, "balanced  CV AUC 0.724": 5, "mean prob 0.369": 6,
                           "predicted 1: 2466": 7, "recall 0.560": 8})

c = nb.cell('''
Zt = prep.transform(X_train)                # 대체·원-핫만 한 48열
path = DecisionTreeClassifier(random_state=2026)\\
    .cost_complexity_pruning_path(Zt, y_train)
alphas = path.ccp_alphas
print(len(alphas), alphas[:3].round(6), alphas[-2:].round(5))

show = list(alphas[1::6]) + [alphas[-1]]     # 0 빼고 6개에 1개
n_leaves = [DecisionTreeClassifier(ccp_alpha=a, random_state=2026)
            .fit(Zt, y_train).get_n_leaves() for a in show]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
ax[0].step(alphas[1:], path.impurities[1:], where="post")
ax[0].set_ylabel("Total impurity of leaves")
ax[1].step(show, n_leaves, where="post")
ax[1].set_ylabel("Number of leaves")
for a in ax:
    a.set_xscale("log")
    a.set_xlabel("ccp_alpha (log scale)")
plt.tight_layout()
''', title="비용 복잡도 가지치기 경로")
save("ml01_ccp", c, marks={"179": 1})

# ================================================================ 다. 조절값 하나씩 바꿔 보기
c = nb.cell('''
from sklearn.model_selection import validation_curve

depths = list(range(1, 16))
tr, va = validation_curve(
    make_tree_pipe(DecisionTreeClassifier(random_state=2026)),
    X_train, y_train, param_name="model__max_depth",
    param_range=depths, cv=skf, scoring="roc_auc", n_jobs=-1)
print(tr.shape, va.shape)
print("best depth:", depths[va.mean(axis=1).argmax()],
      " CV AUC:", va.mean(axis=1).max().round(3))

fig, ax = plt.subplots(figsize=(6.5, 3.8))
for s, lab in [(tr, "Training folds"), (va, "Cross-validation")]:
    m, sd = s.mean(axis=1), s.std(axis=1)
    ax.plot(depths, m, "o-", label=lab)
    ax.fill_between(depths, m - sd, m + sd, alpha=0.2)
ax.set_xlabel("max_depth")
ax.set_ylabel("AUC (mean +/- SD of 5 folds)")
ax.set_title("Validation curve: max_depth")
ax.legend()
plt.tight_layout()
''', title="검증 곡선 1. max_depth")
save("ml01_vc_depth", c, marks={"(15, 5) (15, 5)": 1, "best depth: 5": 2, "CV AUC: 0.706": 3})

c = nb.cell('''
leaf_sizes = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 600, 800, 1200]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
for d in [None, 4]:
    tr, va = validation_curve(
        make_tree_pipe(DecisionTreeClassifier(max_depth=d,
                                              random_state=2026)),
        X_train, y_train, param_name="model__min_samples_leaf",
        param_range=leaf_sizes, cv=skf, scoring="roc_auc", n_jobs=-1)
    m, sd = va.mean(axis=1), va.std(axis=1)
    best = leaf_sizes[m.argmax()]
    print(f"max_depth={d}: best min_samples_leaf {best}, "
          f"CV AUC {m.max():.3f}")
    ax.plot(leaf_sizes, m, "o-", label=f"CV, max_depth={d}")
    ax.fill_between(leaf_sizes, m - sd, m + sd, alpha=0.15)
    if d is None:
        ax.plot(leaf_sizes, tr.mean(axis=1), "o--", color="gray",
                label="Training folds, max_depth=None")
ax.set_xscale("log")
ax.set_xlabel("min_samples_leaf (log scale)")
ax.set_ylabel("AUC")
ax.set_title("Validation curve: min_samples_leaf")
ax.legend(fontsize=8)
plt.tight_layout()
''', title="검증 곡선 2. min_samples_leaf")
save("ml01_vc_leaf", c, marks={"max_depth=None: best min_samples_leaf 300": 1, "CV AUC 0.741": 2,
                                "max_depth=4: best min_samples_leaf 800": 3, "CV AUC 0.739": 4})

# ================================================================ 라. 격자 탐색으로 튜닝하기
c = nb.cell('''
from sklearn.model_selection import GridSearchCV

grid1 = {"model__max_depth": [3, 4, 5, 6, 8, 10],
         "model__min_samples_leaf": [10, 20, 50, 100, 200]}
gs1 = GridSearchCV(
    make_tree_pipe(DecisionTreeClassifier(random_state=2026)),
    grid1, cv=skf, scoring="roc_auc", n_jobs=-1,
    return_train_score=True)
t0 = time.perf_counter()
gs1.fit(X_train, y_train)
print("seconds:", round(time.perf_counter() - t0, 1))
n_comb = len(gs1.cv_results_["params"])
print("combinations:", n_comb, " fits:", n_comb * skf.get_n_splits())
print("best:", gs1.best_params_, round(gs1.best_score_, 4))
''', title="격자 탐색 1. 처음 정한 범위")
GS1_CELL = c

c = nb.cell('''
grid = {"model__max_depth": [4, 5, 6, 7, 8, 10, 12],
        "model__min_samples_leaf": [100, 150, 200, 300, 400, 600]}
gs = GridSearchCV(
    make_tree_pipe(DecisionTreeClassifier(random_state=2026)),
    grid, cv=skf, scoring="roc_auc", n_jobs=-1,
    return_train_score=True)
t0 = time.perf_counter()
gs.fit(X_train, y_train)
print("seconds:", round(time.perf_counter() - t0, 1))
print("fits:", len(gs.cv_results_["params"]) * 5)
print("best:", gs.best_params_, round(gs.best_score_, 4))
''', title="격자 탐색 2. 범위를 넓혀 다시")
GS2_CELL = c

c = nb.cell('''
grid_list = [
    {"model__max_depth": [5, 6, 7],
     "model__min_samples_leaf": [150, 300]},
    {"model__max_leaf_nodes": [8, 10, 12, 16],
     "model__min_samples_leaf": [150, 300]}]
gs_two = GridSearchCV(
    make_tree_pipe(DecisionTreeClassifier(random_state=2026)),
    grid_list, cv=skf, n_jobs=-1,
    scoring={"auc": "roc_auc", "logloss": "neg_log_loss"},
    refit="auc")
gs_two.fit(X_train, y_train)
r2 = pd.DataFrame(gs_two.cv_results_)
print("combinations:", len(r2), " best:", gs_two.best_params_)
show = r2[["param_model__max_depth", "param_model__max_leaf_nodes",
           "param_model__min_samples_leaf", "mean_test_auc",
           "rank_test_auc", "mean_test_logloss", "rank_test_logloss"]]
show.columns = ["depth", "max_leaves", "min_leaf", "auc", "auc_rank",
                "logloss", "ll_rank"]
show.sort_values("auc_rank").round(4)
''', title="격자를 사전의 목록으로, 지표를 둘로", max_rows=14)
GS3_CELL = c

# ================================================================ 마. 결과를 표와 그림으로 보기
c = nb.cell('''
res = pd.DataFrame(gs.cv_results_)
table = (res[["rank_test_score", "param_model__max_depth",
              "param_model__min_samples_leaf", "mean_test_score",
              "std_test_score", "mean_train_score"]]
         .rename(columns={"rank_test_score": "rank",
                          "param_model__max_depth": "max_depth",
                          "param_model__min_samples_leaf": "min_leaf",
                          "mean_test_score": "cv_auc",
                          "std_test_score": "cv_sd",
                          "mean_train_score": "train_auc"})
         .sort_values("rank"))
table.head(8).round(4)
''', title="cv_results_를 표로")
TABLE_CELL = c

c = nb.cell('''
heat = res.pivot(index="param_model__min_samples_leaf",
                 columns="param_model__max_depth",
                 values="mean_test_score")
fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.imshow(heat.to_numpy(), cmap="Blues", aspect="auto")
ax.set_xticks(range(heat.shape[1]), heat.columns)
ax.set_yticks(range(heat.shape[0]), heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.iloc[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=8, color="white" if v > 0.745 else "black")
ax.set_xlabel("max_depth")
ax.set_ylabel("min_samples_leaf")
ax.set_title("Mean CV AUC")
fig.colorbar(im, ax=ax)
plt.tight_layout()
''', title="두 조절값의 열지도")
save("ml01_heat", c)

c = nb.cell('''
def boot_auc(y_true, p, n_boot=1000, seed=2026):
    """시험 자료를 n_boot번 복원추출해 AUC를 다시 구한 값들"""
    rng = np.random.default_rng(seed)
    yt, p = np.asarray(y_true), np.asarray(p)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        if yt[i].min() == yt[i].max():      # 한 범주만 뽑히면 건너뜀
            continue
        aucs.append(roc_auc_score(yt[i], p[i]))
    return np.array(aucs)

final = gs.best_estimator_              # 학습 자료 전체로 다시 맞춤
p_tree = final.predict_proba(X_test)[:, 1]
b_tree = boot_auc(y_test, p_tree)
print("leaves:", final.named_steps["model"].get_n_leaves())
print("test AUC:", round(roc_auc_score(y_test, p_tree), 3),
      " 95% CI:", np.percentile(b_tree, [2.5, 97.5]).round(3))
''', title="최종 나무를 시험 자료로 한 번 평가")
save("ml01_final", c, marks={"leaves: 11": 1, "test AUC: 0.771": 2, "[0.726 0.81 ]": 3})

c = nb.cell('''
from sklearn.linear_model import LogisticRegression

# 0장 라 절에서 고른 C = 0.1
lr = make_pipe(LogisticRegression(C=0.1, max_iter=1000))
cv_lr = cross_val_score(lr, X_train, y_train, cv=skf,
                        scoring="roc_auc")
lr.fit(X_train, y_train)
p_lr = lr.predict_proba(X_test)[:, 1]
b_lr = boot_auc(y_test, p_lr)           # 같은 seed, 같은 복원추출

i_best = gs.best_index_
comp = pd.DataFrame({
    "cv_auc": [res.loc[i_best, "mean_test_score"], cv_lr.mean()],
    "cv_sd": [res.loc[i_best, "std_test_score"], cv_lr.std()],
    "test_auc": [roc_auc_score(y_test, p_tree),
                 roc_auc_score(y_test, p_lr)],
    "ci_low": [np.percentile(b_tree, 2.5), np.percentile(b_lr, 2.5)],
    "ci_high": [np.percentile(b_tree, 97.5),
                np.percentile(b_lr, 97.5)]},
    index=["decision tree", "logistic (C=0.1)"])
print("difference 95% CI:",
      np.percentile(b_tree - b_lr, [2.5, 97.5]).round(3))
comp.round(3)
''', title="0장 로지스틱 회귀와 비교")
save("ml01_compare", c, marks={"[-0.002  0.061]": 6},
     dfmarks={"0.754": 1, "0.749": 2, "0.771": 3, "0.742": 4, "0.695": 5})

c = nb.cell('''
from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(4.8, 4.4))
for p, lab in [(p_tree, "Decision tree"), (p_lr, "Logistic (C=0.1)")]:
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{lab}, AUC "
            f"{roc_auc_score(y_test, p):.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("1 - specificity")
ax.set_ylabel("Sensitivity")
ax.set_title("ROC curves in the test set")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
''', title="두 모형의 ROC 곡선")
save("ml01_roc", c)

c = nb.cell('''
deep = make_tree_pipe(DecisionTreeClassifier(random_state=2026))
deep.fit(X_train, y_train)              # 깊이 제한 없는 나무
imp_final = pd.Series(final.named_steps["model"].feature_importances_,
                      index=names).sort_values(ascending=False)
imp_deep = pd.Series(deep.named_steps["model"].feature_importances_,
                     index=names).sort_values(ascending=False)
print("nonzero in final tree:", (imp_final > 0).sum())
null4 = ["dyslip", "oa", "n_prescribers", "region_city"]
print("no-effect features, rank in deep tree:",
      [list(imp_deep.index).index(f) + 1 for f in null4])

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for a, imp, t in [(ax[0], imp_final, "Tuned tree (11 leaves)"),
                  (ax[1], imp_deep, "Unrestricted tree")]:
    top = imp[imp > 0].head(12)[::-1]   # 0보다 큰 것만
    a.barh(top.index, top.to_numpy())
    a.set_xlabel("Impurity-based importance")
    a.set_title(t)
plt.tight_layout()
''', title="불순도 기반 변수 중요도")
save("ml01_imp", c, marks={"nonzero in final tree: 7": 1, "[17, 25, 10, 13]": 2})

c = nb.cell('''
fig, ax = plt.subplots(figsize=(14, 7))
plot_tree(final.named_steps["model"], feature_names=list(names),
          class_names=["no", "admit"], filled=True, proportion=True,
          impurity=False, precision=3, fontsize=7, ax=ax)
plt.tight_layout()
''', title="최종 나무 그림")
save("ml01_finaltree", c)

c = nb.cell('''
from sklearn.model_selection import learning_curve

sizes, tr, va = learning_curve(
    make_tree_pipe(DecisionTreeClassifier(        # 셀 17에서 고른 조절값
        max_depth=6, min_samples_leaf=300, random_state=2026)),
    X_train, y_train, train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    cv=skf, scoring="roc_auc", n_jobs=-1)
print("patients:", sizes)
print("train:", tr.mean(axis=1).round(3))
print("CV:   ", va.mean(axis=1).round(3))

fig, ax = plt.subplots(figsize=(6.5, 3.8))
for s, lab in [(tr, "Training folds"), (va, "Cross-validation")]:
    m, sd = s.mean(axis=1), s.std(axis=1)
    ax.plot(sizes, m, "o-", label=lab)
    ax.fill_between(sizes, m - sd, m + sd, alpha=0.2)
ax.set_xlabel("Number of patients used for training")
ax.set_ylabel("AUC (mean +/- SD of 5 folds)")
ax.set_title("Learning curve: max_depth=6, min_samples_leaf=300")
ax.legend()
plt.tight_layout()
''', title="학습 곡선 (learning_curve)")
save("ml01_lc", c, marks={"[ 900 1800 3600 5400 7200 9000]": 1, "0.666": 2, "0.754]": 3, "0.777]": 4})
LC = (nb.ns["sizes"].copy(), nb.ns["tr"].copy(), nb.ns["va"].copy())    # 과제 2 셀이 tr, va를 덮어쓰므로 대조용으로 남김

c = nb.cell('''
import joblib

joblib.dump(final, "tree_admit_2023.joblib")       # 파일로 저장
loaded = joblib.load("tree_admit_2023.joblib")     # 다시 불러오기
p_again = loaded.predict_proba(X_test)[:, 1]
print("same predictions:", np.array_equal(p_again, p_tree))
print(loaded.named_steps["model"].get_params()["min_samples_leaf"])
''', title="모형 저장과 불러오기 (joblib)")
save("ml01_joblib", c, marks={"same predictions: True": 1, "300": 2})

# ================================================================ 바. 과제
c = nb.cell('''
gs_ent = GridSearchCV(
    make_tree_pipe(DecisionTreeClassifier(criterion="entropy",
                                          random_state=2026)),
    grid, cv=skf, scoring="roc_auc", n_jobs=-1)
gs_ent.fit(X_train, y_train)
print("entropy best:", gs_ent.best_params_,
      round(gs_ent.best_score_, 4))
print("gini best   :", gs.best_params_, round(gs.best_score_, 4))
print("leaves:", gs_ent.best_estimator_.named_steps["model"]
      .get_n_leaves())
''', title="과제 1 정답. 엔트로피로 다시 튜닝")
HW1_CELL = c

c = nb.cell('''
alpha_grid = [0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 7e-4, 1e-3, 1.5e-3]
tr, va = validation_curve(
    make_tree_pipe(DecisionTreeClassifier(random_state=2026)),
    X_train, y_train, param_name="model__ccp_alpha",
    param_range=alpha_grid, cv=skf, scoring="roc_auc", n_jobs=-1)
for a, m in zip(alpha_grid, va.mean(axis=1)):
    t = DecisionTreeClassifier(ccp_alpha=a, random_state=2026)
    t.fit(prep.transform(X_train), y_train)
    print(f"ccp_alpha {a:<7}  leaves {t.get_n_leaves():4d}"
          f"  CV AUC {m:.3f}")
print("depth/leaf tuned tree: leaves 11, CV AUC",
      round(gs.best_score_, 3))
''', title="과제 2 정답. 가지치기한 나무와 크기를 제한한 나무")
HW2_CELL = c

c = nb.cell('''
from sklearn.metrics import confusion_matrix

best = dict(max_depth=6, min_samples_leaf=300, random_state=2026)
rate = y_train.mean()                        # 학습 자료의 입원 비율
for cw, thr in [(None, 0.5), ("balanced", 0.5), (None, rate)]:
    m = make_tree_pipe(
        DecisionTreeClassifier(class_weight=cw, **best))
    p = cross_val_predict(m, X_train, y_train, cv=skf,
                          method="predict_proba")[:, 1]
    print(f"class_weight={cw}, threshold {thr:.3f}")
    print(confusion_matrix(y_train, (p >= thr).astype(int)))
''', title="과제 3 정답. class_weight와 임계값")
HW3_CELL = c

# ---------------------------------------------------------------- 걸린 시간 고정, 마크
fix_time(GS1_CELL, 9.6)
fix_time(GS2_CELL, 11.2)
save("ml01_depthtab", DEPTH_CELL, dfmarks={"0.706": 1, "0.729": 2, "674": 3, "1.000": 4})
save("ml01_gs1", GS1_CELL, marks={"seconds: 9.6": 1, "combinations: 30": 2, "fits: 150": 3,
                                   "'model__min_samples_leaf': 200} 0.7478": 4})
save("ml01_gs2", GS2_CELL, marks={"seconds: 11.2": 1, "fits: 210": 2,
                                   "{'model__max_depth': 6, 'model__min_samples_leaf': 300} 0.7544": 3})
save("ml01_gs3", GS3_CELL, marks={"combinations: 14": 1})
save("ml01_table", TABLE_CELL, dfmarks={"0.7544": 1, "0.7538": 2, "0.0251": 3, "0.7766": 4})
save("ml01_hw1", HW1_CELL)
save("ml01_hw2", HW2_CELL)
save("ml01_hw3", HW3_CELL)

print("ml01: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)



# ================================================================ 그림 1-1: 환자 10명의 나무 (SVG)
def toy_tree_figure():
    from html import escape
    sys.path.insert(0, ROOT)
    from svgplot import figure
    W, H = 640, 318
    o = []

    def box(cx, y, w, lines, cls):
        h = 18 + 17 * len(lines)
        o.append(f'<rect x="{cx - w / 2:.1f}" y="{y}" width="{w}" height="{h}" rx="5" class="{cls}" stroke-width="1"/>')
        for k, (t, tc) in enumerate(lines):
            o.append(f'<text x="{cx:.1f}" y="{y + 22 + 17 * k}" text-anchor="middle" class="{tc}">{escape(t)}</text>')
        return y + h

    def text(x, y, t, anchor="middle", cls="lbl small"):
        o.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{escape(t)}</text>')

    def link(x0, y0, x1, y1):
        o.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" class="ref" stroke-width="1.2"/>')

    # 깊이 표시
    for y, t in [(42, "depth 0 (root)"), (152, "depth 1"), (262, "depth 2")]:
        text(8, y, t, "start", "lbl mute small")
    b0 = box(330, 14, 230, [("All 10 patients: 3 admitted (30%)", "lbl small strong"), ("Gini 0.420", "lbl small")], "a4")
    text(330, b0 + 20, "age <= 76 ?", "middle", "lbl strong")
    link(285, b0, 200, 116)
    link(375, b0, 460, 116)
    text(228, b0 + 30, "yes", "middle", "lbl mute small")
    text(432, b0 + 30, "no", "middle", "lbl mute small")
    b1 = box(185, 120, 210, [("A B C F H I: 0 of 6 (0%)", "lbl small strong"), ("Gini 0  ->  leaf, predict 0.00", "lbl small")], "a3 s3")
    b2 = box(460, 120, 210, [("D E G J: 3 of 4 (75%)", "lbl small strong"), ("Gini 0.375", "lbl small")], "a1 s1")
    text(460, b2 + 20, "n_ed <= 1.5 ?", "middle", "lbl strong")
    link(425, b2, 385, 226)
    link(495, b2, 545, 226)
    text(392, b2 + 30, "yes", "middle", "lbl mute small")
    text(530, b2 + 30, "no", "middle", "lbl mute small")
    box(370, 230, 170, [("E J: 1 of 2 (50%)", "lbl small strong"), ("leaf, predict 0.50", "lbl small")], "a3 s3")
    box(555, 230, 150, [("D G: 2 of 2 (100%)", "lbl small strong"), ("leaf, predict 1.00", "lbl small")], "a3 s3")
    text(185, 200, "after the first split:", "middle", "lbl mute small")
    text(185, 216, "0.6 x 0 + 0.4 x 0.375 = 0.150", "middle", "lbl mute small")
    svg = (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="환자 10명으로 만든 깊이 2의 나무" '
           f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')
    cap = ("그림 1-1. 0장 가 절의 환자 10명으로 만든 깊이 2의 나무. 맨 위 마디가 뿌리, 더 나누지 않는 초록 마디가 잎입니다. "
           "잎의 입원 비율이 그 잎에 도착하는 새 사람의 예측 확률이 됩니다. 두 번째 나누기의 응급실 기준은 나이 82세 기준과 "
           "불순도 감소가 같은 동점입니다(본문 ④).")
    return figure(svg, cap)


nb.save_fragment("ml01_fig_toytree", toy_tree_figure())

# ================================================================ notebook
nb.save_ipynb(
    "머신러닝 기초 1장. 결정트리",
    intro=("사회약학 연구방법 노트 '머신러닝 기초' 1장을 따라 하는 노트북입니다. 0장과 같은 가상 청구자료와 분할로 "
           "결정트리가 자료를 나누는 방법을 작은 표로 확인하고, 나무의 크기와 과적합, 검증 곡선, 격자 탐색으로 "
           "조절값을 고른 뒤 결과를 표와 그림으로 정리합니다. 셀을 위에서부터 차례로 실행하세요. 셀 4–5는 0장의 "
           "준비 셀을 다시 실행하는 것이고, 셀 6은 이 장에서 새로 만드는 나무용 전처리 함수입니다. "
           "자료는 사이트가 만든 가상 자료이며 실제 환자 자료가 아닙니다."),
    notes={
        1: "## 가. 나무가 자료를 나누는 방법\n\n환자 10명으로 지니 불순도와 나누기 기준을 계산합니다.",
        4: "**0장의 준비를 다시 합니다.** 자료, 분할, 교차검증 겹, 파이프라인을 만든 뒤 공통 자료로 작은 나무를 맞춥니다.",
        10: "## 나. 나무의 크기와 과적합\n\n깊이를 늘려 가며 학습·교차검증·시험 AUC를 비교하고, class_weight와 가지치기를 봅니다.",
        14: "## 다. 조절값 하나씩 바꿔 보기\n\nvalidation_curve로 max_depth와 min_samples_leaf를 하나씩 바꿉니다.",
        16: "## 라. 격자 탐색으로 튜닝하기\n\nGridSearchCV로 두 조절값을 함께 고릅니다. 셀마다 수십 초가 걸릴 수 있습니다.",
        19: ("## 마. 결과를 표와 그림으로 보기\n\n탐색 결과 표와 열지도, 시험 자료 평가, 로지스틱 회귀와의 비교, "
             "변수 중요도, 나무 그림, 학습 곡선, 모형 저장."),
        28: ("## 과제 정답\n\n**과제 1.** criterion을 \"entropy\"로 바꿔 셀 17과 같은 격자로 다시 튜닝하고 "
             "지니로 고른 결과와 비교합니다."),
        29: ("**과제 2.** ccp_alpha만 바꾼 나무(깊이와 잎 크기 제한 없음)의 교차검증 AUC를 구해 "
             "깊이와 잎 크기로 제한한 나무와 비교합니다."),
        30: ("**과제 3.** 고른 조절값의 나무를 class_weight 없이와 \"balanced\"로 맞춰 교차검증 예측 확률의 "
             "혼동행렬을 임계값 0.5와 입원 비율(0.041)에서 비교합니다."),
    })

# ================================================================ 대조 블록: 본문에 적은 숫자와 실행 결과 맞추기
if __name__ == "__main__":
    import json
    import math
    import warnings
    import pandas as pd
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score, log_loss
    from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier

    ns = nb.ns
    df_, X_, y_ = ns["df"], ns["X"], ns["y"]
    Xtr, Xte, ytr, yte = ns["X_train"], ns["X_test"], ns["y_train"], ns["y_test"]
    mtp, mp, skf = ns["make_tree_pipe"], ns["make_pipe"], ns["skf"]
    n_ok = 0

    def same(label, a, b, tol=1e-9):
        global n_ok
        a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
        assert a.shape == b.shape and np.all(np.abs(a - b) <= tol), f"{label}: {a} != {b}"
        n_ok += 1

    T = json.load(open(os.path.join(ROOT, "gen", "_ml_truth.json"), encoding="utf-8"))
    same("split", [len(Xtr), len(Xte), ytr.sum(), yte.sum()],
         [T["check"]["n_train"], T["check"]["n_test"], T["check"]["events_train"], T["check"]["events_test"]])
    # 가 절: 환자 10명 (0장 가 절의 표와 같은 값)
    toy = ns["toy"]
    same("toy table = ml00 table", [toy["age"].tolist(), toy["n_ed"].tolist(), toy["admit"].tolist()],
         [[52, 61, 47, 78, 83, 69, 88, 58, 74, 81], [0, 0, 1, 2, 0, 0, 3, 1, 2, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]])
    same("toy gini", [ns["g0"], ns["split_gini"]("n_ed", 1.5), ns["split_gini"]("age", 76)],
         [0.42, 0.7 * 12 / 49 + 0.3 * 4 / 9, 0.4 * 0.375], 1e-12)
    print(f"VERIFY toy gini after: n_ed {ns['split_gini']('n_ed', 1.5):.4f}, age {ns['split_gini']('age', 76):.4f}")
    # 오른쪽 4명 안에서 응급실 2회로 나누면 0.375 -> 0.25
    same("toy depth-2 within node", [2 * 0.75 * 0.25, (2 * 0 + 2 * 0.5) / 4], [0.375, 0.25])
    rnode = toy[toy["age"] > 76]
    g_ed = (lambda L, R: (len(L) * ns["gini"](L) + len(R) * ns["gini"](R)) / len(rnode))(
        rnode["admit"][rnode["n_ed"] <= 1.5], rnode["admit"][rnode["n_ed"] > 1.5])
    g_a82 = (lambda L, R: (len(L) * ns["gini"](L) + len(R) * ns["gini"](R)) / len(rnode))(
        rnode["admit"][rnode["age"] <= 82], rnode["admit"][rnode["age"] > 82])
    same("depth-2 tie n_ed vs age 82", [g_ed, g_a82], [0.25, 0.25], 1e-12)
    # 나 절: 잎 크기와 확률의 흔들림 (입원 비율 4%)
    same("leaf of 20: P(0 events), SE; leaf of 300: SE",
         [round(0.96 ** 20, 3), round(math.sqrt(0.04 * 0.96 / 20), 3), round(math.sqrt(0.04 * 0.96 / 300), 3), 0.04 * 300],
         [0.442, 0.044, 0.011, 12])
    # 엔트로피(접힌 상자)
    H = lambda p: 0.0 if p in (0, 1) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))  # noqa: E731
    e0, e_age, e_ed = H(0.3), 0.4 * H(0.75), 0.7 * H(1 / 7) + 0.3 * H(2 / 3)
    print(f"VERIFY entropy root {e0:.3f}, after age {e_age:.3f} (gain {e0 - e_age:.3f}), after n_ed {e_ed:.3f} (gain {e0 - e_ed:.3f})")
    same("entropy", [round(e0, 3), round(e_age, 3), round(e0 - e_age, 3), round(e_ed, 3), round(e0 - e_ed, 3)],
         [0.881, 0.325, 0.557, 0.690, 0.192])
    # 깊이 3의 나무: 첫 분할과 참 구조의 계단
    t3 = ns["tree3"].tree_
    nm = list(ns["names"])
    same("first split age 87.5", [nm.index("age"), t3.threshold[0]], [t3.feature[0], 87.5])
    bl = ns["by_leaf"]
    same("leaf proba = leaf rate", np.abs(bl["admit_rate"] - bl["proba"]).max(), 0, 1e-12)
    same("leaf counts", [bl["n"].sum(), len(bl), bl["n"].max()], [11250, 8, 9769])
    age = df_["age"]
    print("VERIFY truth rates: 85+ {:.3f}, n_ed>=2 {:.3f} (n={}), n_drugs>=10 {:.3f} vs <10 {:.3f}".format(
        df_.loc[age >= 85, "admit_2023"].mean(), df_.loc[df_.n_ed >= 2, "admit_2023"].mean(), (df_.n_ed >= 2).sum(),
        df_.loc[df_.n_drugs >= 10, "admit_2023"].mean(), df_.loc[df_.n_drugs < 10, "admit_2023"].mean()))
    same("truth rates (whole file)", [round(df_.loc[age >= 85, "admit_2023"].mean(), 3),
                                     round(df_.loc[df_.n_ed >= 2, "admit_2023"].mean(), 3),
                                     round(df_.loc[df_.n_drugs >= 10, "admit_2023"].mean(), 3)], [0.308, 0.221, 0.285])
    same("train n_ed>=2 and n_drugs>=10 counts", [(Xtr.n_ed >= 2).sum(), (Xtr.n_drugs >= 10).sum()], [184, 366])
    ag, dr = Xtr["age"], Xtr["n_drugs"] >= 10
    same("n_drugs>=10 by age node (<=76, 77-85, >85)", [(dr & (ag <= 76)).sum(), (dr & (ag > 76) & (ag <= 85)).sum(),
                                                       (dr & (ag > 85)).sum()], [139, 124, 103])
    same("small groups: insulin, hf&ckd", [Xtr["insulin"].sum(), ((Xtr.hf == 1) & (Xtr.ckd == 1)).sum()], [203, 41])
    tr_csv = pd.read_csv(os.path.join(ROOT, "gen", "_ml_truth.csv")).set_index("id")
    p_true_te = tr_csv.loc[df_.loc[Xte.index, "id"], "p_true"].to_numpy()
    same("true-probability test AUC", round(roc_auc_score(yte, p_true_te), 3), 0.810)
    same("cv recall 0.089 -> 41 of 459", round(41 / 459, 3), 0.089)
    # 나 절: 깊이 표
    dt = ns["depth_tab"]
    same("depth table", [dt.loc[4, "cv"].round(3), dt.loc[10, "train"], dt.loc[10, "cv"].round(3), dt.loc[10, "leaves"],
                         dt.loc[10, "test"].round(3), dt.loc[5, "test"].round(3)], [0.706, 1.0, 0.578, 674, 0.581, 0.729])
    same("cv peak depth 5", dt["cv"].idxmax(), 4)
    # class_weight="balanced"의 가중치와 4% 잎의 가중 비율
    n0, n1 = (ytr == 0).sum(), (ytr == 1).sum()
    w0, w1 = len(ytr) / (2 * n0), len(ytr) / (2 * n1)
    pw = 0.04 * w1 / (0.04 * w1 + 0.96 * w0)
    print(f"VERIFY balanced weights {w0:.3f} {w1:.3f} ratio {w1 / w0:.1f}; 4% leaf -> {pw:.3f}")
    same("balanced weights", [round(w0, 3), round(w1, 2), round(w1 / w0, 1), round(pw, 2)], [0.521, 12.25, 23.5, 0.49])
    same("ccp path length", len(ns["alphas"]), 179)
    # 라 절
    same("gs1 best at edge", [ns["gs1"].best_params_["model__min_samples_leaf"], round(ns["gs1"].best_score_, 4)], [200, 0.7478])
    gs = ns["gs"]
    same("gs best", [gs.best_params_["model__max_depth"], gs.best_params_["model__min_samples_leaf"], round(gs.best_score_, 4)],
         [6, 300, 0.7544])
    g2 = ns["gs_two"]
    same("gs_two best", [g2.best_params_["model__max_leaf_nodes"], round(g2.best_score_, 4)], [10, 0.7546])
    same("gs_two leaves", g2.best_estimator_.named_steps["model"].get_n_leaves(), 10)
    # 마 절
    same("final test AUC and CI", [round(roc_auc_score(yte, ns["p_tree"]), 3), *np.percentile(ns["b_tree"], [2.5, 97.5]).round(3)],
         [0.771, 0.726, 0.810])
    same("logistic test AUC and CI (= ml00)", [round(roc_auc_score(yte, ns["p_lr"]), 3), *np.percentile(ns["b_lr"], [2.5, 97.5]).round(3)],
         [0.742, 0.695, 0.785])
    same("difference CI", np.percentile(ns["b_tree"] - ns["b_lr"], [2.5, 97.5]).round(3), [-0.002, 0.061])
    print(f"VERIFY share of bootstrap differences <= 0: {(ns['b_tree'] - ns['b_lr'] <= 0).mean():.3f}; n boot {len(ns['b_tree'])}")
    cvl = ns["cv_lr"]
    same("lr cv", [round(cvl.mean(), 3), round(cvl.std(), 3)], [0.749, 0.022])
    tree_folds = np.array([gs.cv_results_[f"split{k}_test_score"][gs.best_index_] for k in range(5)])
    print("VERIFY folds tree", tree_folds.round(3), "lr", cvl.round(3), "tree wins", int((tree_folds > cvl).sum()))
    same("tree wins folds", (tree_folds > cvl).sum(), 3)
    pt = ns["p_tree"]
    same("final tree distinct probs and max", [len(np.unique(pt)), round(pt.max(), 3)], [11, 0.313])
    imf, imd = ns["imp_final"], ns["imp_deep"]
    same("importance", [(imf > 0).sum(), round(imf["age"], 3), round(imf["n_drugs"], 3)], [7, 0.864, 0.079])
    same("age + n_drugs + bmi share", round(imf[["age", "n_drugs", "bmi"]].sum(), 2), 0.97)
    bands = [round(df_.loc[(age >= lo) & (age <= hi), "admit_2023"].mean(), 3) for lo, hi in [(70, 74), (75, 79), (80, 84), (85, 99)]]
    same("age band rates", bands, [0.030, 0.064, 0.123, 0.308])
    lo87 = Xtr["age"] <= 87.5
    same("depth-3 root split sizes", [lo87.sum(), (~lo87).sum(), ytr[lo87].sum(), ytr[~lo87].sum(),
                                      round(ytr[lo87].mean(), 3), round(ytr[~lo87].mean(), 3)],
         [11018, 232, 373, 86, 0.034, 0.371])
    null_all = ["dyslip", "oa", "n_prescribers", "region_city", "region_metro", "region_rural"]
    print(f"VERIFY no-effect total importance in deep tree {imd[null_all].sum():.3f}; in final {imf[null_all].sum():.3f}")
    same("no-effect importance", [round(imd[null_all].sum(), 3), imf[null_all].sum()], [0.089, 0.0])
    # 최종 나무의 나누기와 참 구조: 잎 크기 300이면 응급실 2회 이상(184명)을 따로 뗄 수 없다
    tf = ns["final"].named_steps["model"].tree_
    used = sorted({nm[f] for f in tf.feature if f >= 0})
    print("VERIFY final tree features:", used, " thresholds:",
          sorted({(nm[f], round(th, 2)) for f, th in zip(tf.feature, tf.threshold) if f >= 0}))
    same("final tree features", len(used), 7)
    # 학습 곡선: 9,000명 지점은 셀 17의 교차검증과 같은 값
    lc_n, lc_tr, lc_va = LC
    same("learning curve", [*lc_n, round(lc_va.mean(1)[0], 3), round(lc_va.mean(1)[-1], 3), round(lc_tr.mean(1)[-1], 3)],
         [900, 1800, 3600, 5400, 7200, 9000, 0.666, 0.754, 0.777])
    same("learning curve at 9000 = gs best", lc_va.mean(1)[-1], ns["gs"].best_score_, 1e-12)
    # 최종 나무의 eGFR 88.5 나누기: 77-85세, 약 5가지 이하에서 88.5 초과 쪽은 대부분 eGFR 결측(중앙값 89로 채움)
    nd = (Xtr.age > 76.5) & (Xtr.age <= 85.5) & (Xtr.n_drugs <= 5.5)
    hi = nd & (Xtr.egfr.isna() | (Xtr.egfr > 88.5))
    same("egfr split: median, n, missing", [Xtr.egfr.median(), hi.sum(), (hi & Xtr.egfr.isna()).sum(),
                                           round(ytr[hi].mean(), 3), round(ytr[nd & ~hi].mean(), 3)], [89, 300, 272, 0.097, 0.056])
    sm = Xtr.smoking.isna()
    print(f"VERIFY smoking missing {sm.mean():.3f}, of whom bmi missing {Xtr.bmi.isna()[sm].mean():.3f}; bmi median {Xtr.bmi.median()}")
    same("smoking_missing share and overlap", [round(sm.mean(), 2), round(Xtr.bmi.isna()[sm].mean(), 2)], [0.32, 0.95])
    assert "n_ed" not in used
    n_ok += 1
    # 표준화해도 나무는 같은 예측 (make_pipe 대 make_tree_pipe)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=300, random_state=2026)
        p_sc = mp(clone(best_tree)).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        same("scaling does not change the tree", np.abs(p_sc - pt).max(), 0, 1e-12)
        # 로그 손실: 제한 없는 나무, 고른 나무, 모두에게 입원 비율
        ll_deep = -cross_val_score(mtp(DecisionTreeClassifier(random_state=2026)), Xtr, ytr, cv=skf, scoring="neg_log_loss").mean()
        r2 = ns["r2"]
        sel = (r2["param_model__max_depth"] == 6) & (r2["param_model__min_samples_leaf"] == 300)
        ll_best = -r2.loc[sel, "mean_test_logloss"].iloc[0]
        rate = ytr.mean()
        ll_base = -(rate * np.log(rate) + (1 - rate) * np.log(1 - rate))
        print(f"VERIFY CV log loss: unrestricted {ll_deep:.3f}, depth6/leaf300 {ll_best:.4f}, base rate {ll_base:.3f}")
        same("log loss", [round(ll_deep, 2), round(ll_best, 3), round(ll_base, 3)], [2.80, 0.149, 0.170])
        # 최종 나무 나누기의 가중 불순도 감소 (min_impurity_decrease 범위)
        N = tf.weighted_n_node_samples[0]
        dec = [(tf.weighted_n_node_samples[i] * tf.impurity[i] - tf.weighted_n_node_samples[l] * tf.impurity[l]
                - tf.weighted_n_node_samples[r] * tf.impurity[r]) / N
               for i, (l, r) in enumerate(zip(tf.children_left, tf.children_right)) if l != -1]
        print(f"VERIFY impurity decreases of final tree: min {min(dec):.6f} max {max(dec):.6f}; root gini {tf.impurity[0]:.4f}")
        same("decrease range", [round(min(dec), 5), round(max(dec), 4), round(tf.impurity[0], 3)], [0.00004, 0.0045, 0.078])
        # 과제
        same("hw1 entropy", [ns["gs_ent"].best_params_["model__max_depth"], ns["gs_ent"].best_params_["model__min_samples_leaf"],
                             round(ns["gs_ent"].best_score_, 4)], [5, 300, 0.7528])
        vam = ns["va"].mean(axis=1)
        same("hw2 ccp best", [ns["alpha_grid"][vam.argmax()], round(vam.max(), 3)], [5e-4, 0.692])
        t_ccp = DecisionTreeClassifier(ccp_alpha=5e-4, random_state=2026).fit(ns["prep"].transform(Xtr), ytr)
        leaf_n = np.bincount(t_ccp.apply(ns["prep"].transform(Xtr)))
        leaf_n = leaf_n[leaf_n > 0]
        print("VERIFY ccp 5e-4 tree leaf sizes:", sorted(leaf_n.tolist()))
        same("ccp tree: 8 leaves, largest 9908, three tiny", [len(leaf_n), leaf_n.max(), *sorted(leaf_n)[:3]], [8, 9908, 7, 14, 17])
        # 과제 3: 혼동행렬
        from sklearn.model_selection import cross_val_predict
        cms = {}
        for cw in [None, "balanced"]:
            p = cross_val_predict(mtp(DecisionTreeClassifier(class_weight=cw, max_depth=6, min_samples_leaf=300,
                                                             random_state=2026)), Xtr, ytr, cv=skf, method="predict_proba")[:, 1]
            cms[cw] = p
        tp = lambda p, t: int(((p >= t) & (ytr == 1)).sum())  # noqa: E731
        npos = lambda p, t: int((p >= t).sum())  # noqa: E731
        same("hw3", [npos(cms[None], 0.5), tp(cms["balanced"], 0.5), npos(cms["balanced"], 0.5), tp(cms[None], rate),
                     npos(cms[None], rate)], [0, 297, 3288, 292, 3060])
        print(f"VERIFY hw3 recall balanced {297 / 459:.3f}, none@rate {292 / 459:.3f}; precision {297 / 3288:.3f} {292 / 3060:.3f}")
        # 큰 모의 집단(같은 생성식, seed 777): 교차검증으로 고른 나무 대 로지스틱 회귀 (작성자 확인용)
        if not os.environ.get("NOPOP"):
            sys.path.insert(0, os.path.join(ROOT, "gen"))
            import data_ml
            big, tb = data_ml.simulate(seed=777, n=100000)
            Xb, yb = big[X_.columns], big["admit_2023"]
            te = np.arange(50000, 100000)
            LR = ns["LogisticRegression"]
            auc_t, auc_l = [], []
            for r in range(3):
                tr = np.arange(r * 11250, (r + 1) * 11250)
                g = GridSearchCV(mtp(DecisionTreeClassifier(random_state=2026)), ns["grid"],
                                 cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=2026),
                                 scoring="roc_auc", n_jobs=-1).fit(Xb.iloc[tr], yb.iloc[tr])
                auc_t.append(roc_auc_score(yb.iloc[te], g.predict_proba(Xb.iloc[te])[:, 1]))
                m = mp(LR(C=0.1, max_iter=1000)).fit(Xb.iloc[tr], yb.iloc[tr])
                auc_l.append(roc_auc_score(yb.iloc[te], m.predict_proba(Xb.iloc[te])[:, 1]))
            print(f"VERIFY population (seed 777): tuned tree {np.round(auc_t, 4)} mean {np.mean(auc_t):.4f}; "
                  f"logistic {np.round(auc_l, 4)} mean {np.mean(auc_l):.4f}")
            same("population means", [round(np.mean(auc_t), 3), round(np.mean(auc_l), 3)], [0.747, 0.743])

    print(f"VERIFY: {n_ok} checks passed")
    os.chdir(ROOT)
    _TMP.cleanup()
