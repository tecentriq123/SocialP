"""머신러닝 기초 2장 · 배깅과 랜덤 포레스트 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/ml_ml02.py          (전체 실행, 이 환경에서 약 20-25분)
      NOMARK=1 python3 gen/ml_ml02.py   표식 없이 모든 셀의 출력을 화면에 찍는다
      ML02_CACHE=<파일> python3 gen/ml_ml02.py            실행 결과(셀 출력과 그림 자료)를 그 파일에 저장
      REPLAY=1 ML02_CACHE=<파일> python3 gen/ml_ml02.py   셀을 다시 돌리지 않고 저장된 결과로 조각만 다시 만든다
                                                           (셀 코드가 바뀌었으면 멈춘다. 표식과 본문을 고칠 때 쓴다)

자료와 분할, 전처리 함수(make_pipe), 5겹 분할기(skf)는 0장(gen/ml_ml00.py), 나무용 전처리 함수(make_tree_pipe)는
1장(gen/ml_ml01.py)과 같다. 걸린 시간은 실행할 때마다
달라지므로 조각에 들어가는 시간 표시는 아래 SECONDS의 값으로 고정한다(대표 실행 한 번에서 잰 값, 본문에
"환경마다 다르다"고 적음). 끝의 대조 블록에서 본문에 적은 핵심 숫자를 실행 결과와 맞춘다.
"""
import json
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import labkit  # noqa: E402
from labkit import Notebook, _mark, _dedent, Cell  # noqa: E402

# 셀 21(joblib)이 현재 폴더에 파일을 쓰므로 임시 폴더에서 돌린다(그림·노트북은 labkit이 절대 경로로 쓴다).
import tempfile  # noqa: E402
_TMP = tempfile.TemporaryDirectory(prefix="ml02_")
_CWD = os.getcwd()
os.chdir(_TMP.name)

NOMARK = bool(os.environ.get("NOMARK"))
REPLAY = bool(os.environ.get("REPLAY"))
CACHE = os.environ.get("ML02_CACHE")
T_START = time.time()


class ReplayNotebook(Notebook):
    """저장된 실행 결과로 셀을 되살린다(코드가 같을 때만)."""

    def __init__(self, lab, cached):
        super().__init__(lab)
        self._cached = cached

    def cell(self, code, title="", **kw):
        code = _dedent(code)
        d = self._cached[len(self.cells)]
        if d["code"] != code:
            raise SystemExit(f"cell {len(self.cells) + 1} code changed; run without REPLAY")
        c = Cell(len(self.cells) + 1, code, title)
        for k in ("stdout", "value_repr", "value_html", "images", "warns", "error"):
            setattr(c, k, d[k])
        self.cells.append(c)
        return c


if REPLAY:
    with open(CACHE, "rb") as f:
        _cache = pickle.load(f)
    nb = ReplayNotebook("ml02", _cache["cells"])
    DATA = _cache["data"]
else:
    nb = Notebook("ml02")
    DATA = {}
ns = nb.ns

SMOKE = bool(os.environ.get("SMOKE"))   # 빠른 점검용: 반복 횟수를 줄여 모든 셀이 오류 없이 도는지만 본다(조각은 버림)
if SMOKE:
    _SMOKE_SUBS = [("range(100)", "range(5)"), ("n_estimators=200", "n_estimators=10"),
                   ("n_estimators=500", "n_estimators=10"), ("n_trials=40", "n_trials=12"),
                   ("n_trials=20", "n_trials=3"), ("n_iter=40", "n_iter=12"), ("max_evals=40", "max_evals=12"),
                   ("range(1, 41)", "range(1, 13)"), ("n_candidates=81", "n_candidates=9"),
                   ("n_repeats=10", "n_repeats=2"), ("for _ in range(10):", "for _ in range(2):"),
                   ("n_boot=1000", "n_boot=50"), ("[25, 50, 75, 100, 150, 200, 300, 400, 500]", "[25, 50]"),
                   ("[2, 4, 6, 12, 24, 48]", "[2, 6]"), ("[1, 5, 20, 50, 100]", "[1, 20]"),
                   ("[10, 20, 40, 80]", "[5, 10, 12, 24]"), ("[10, 50, 100, 200, 500, 1000]", "[10, 20]")]
    _orig_cell = nb.cell

    def _smoke_cell(code, title="", **kw):
        for a, b in _SMOKE_SUBS:
            code = code.replace(a, b)
        return _orig_cell(code, title, **kw)
    nb.cell = _smoke_cell

# 조각에 넣을 걸린 시간(초). 대표 실행 한 번에서 잰 값으로 고정한다. None이면 실제 값을 그대로 쓴다.
SECONDS = {    # 2026-10-09 대표 실행(코어 2개)에서 잰 값
    "t_oob": 1, "t_cv5": 5, "t_warm": 5, "t_grid": 53, "t_rand": 96, "t_halv": 93,
    "t_optuna": 119, "t_hyperopt": 103, "hw3": [0.1, 0.5, 0.8, 1.6, 3.8, 7.5],
}
MARK_ERRORS = []


def grab(key, fn):
    """전체 실행이면 fn()을 계산해 DATA에 담고, REPLAY면 DATA에서 꺼낸다."""
    if not REPLAY:
        DATA[key] = fn()
    return DATA[key]


def secs(key, actual):
    v = SECONDS.get(key)
    return actual if v is None else v


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


def row_last_td(h, label, value):
    """DataFrame 표에서 행 이름이 label인 줄의 마지막 칸을 value로 바꾼다(걸린 시간 열)."""
    m = re.search(r"<th>" + re.escape(str(label)) + r"</th>.*?</tr>", h, re.S)
    if not m:
        MARK_ERRORS.append(f"row {label!r} not found")
        return h
    row = m.group(0)
    k = row.rfind("<td>")
    row2 = row[:k] + re.sub(r"<td>[^<]*</td>", f"<td>{value}</td>", row[k:], count=1)
    return h[:m.start()] + row2 + h[m.end():]


def save(name, c, marks=None, dfmarks=None, subs=None, rowfix=None):
    """subs: [(실제 문자열, 고정 문자열)] — 걸린 시간처럼 실행마다 달라지는 출력을 고정한다."""
    if NOMARK:
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        for w in c.warns:
            print("WARN:", w)
        marks = dfmarks = None
    try:
        h = render(c, marks, dfmarks)
    except ValueError as e:
        MARK_ERRORS.append(f"{name}: {e}")
        h = render(c)
    for a, b in subs or []:
        if a not in h:
            MARK_ERRORS.append(f"{name}: time text not found {a!r}")
        h = h.replace(a, b, 1)
    for label, value in rowfix or []:
        h = row_last_td(h, label, value)
    nb.save_fragment(name, h)


# ================================================================ 준비
c = nb.cell('''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split,
                                     StratifiedKFold, cross_val_score)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
df = pd.read_csv(BASE + "ml_claims.csv")
y = df["admit_2023"]
X = df.drop(columns=["id", "admit_2023", "cost_2023"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=2026)

cat_cols = ["insurance", "region", "smoking", "alcohol"]
num_cols = [c for c in X.columns if c not in cat_cols]

def make_pipe(model, num=num_cols, cat=cat_cols):
    """0장 다 절과 같은 전처리(결측 대체, 표준화, 원-핫) + 모형"""
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

def make_tree_pipe(model, num=num_cols, cat=cat_cols):
    """1장 가 절과 같은 나무용: 결측 대체와 원-핫만 (표준화 없음)"""
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

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
print(X_train.shape, X_test.shape, y_train.sum(), y_test.sum())
''', title="0장과 1장의 준비를 한 번에 (자료, 분할, 파이프라인, skf)")
save("ml02_setup", c, marks={"(11250, 39) (3750, 39) 459 153": 1})

# ================================================================ 가. 여러 나무를 평균 내는 이유
c = nb.cell('''
from sklearn.tree import DecisionTreeClassifier

prep = make_tree_pipe(None).named_steps["prep"]  # 전처리만
Z_train = prep.fit_transform(X_train)          # 11250 × 48 배열
Z_test = prep.transform(X_test)                # 3750 × 48
y_arr = y_train.to_numpy()

rng = np.random.default_rng(2026)
n = len(Z_train)
probs = []                                     # 나무마다 시험 자료 확률
for b in range(100):
    idx = rng.integers(0, n, size=n)           # ① 복원추출로 한 벌
    tree = DecisionTreeClassifier(min_samples_leaf=20,
                                  random_state=b)
    tree.fit(Z_train[idx], y_arr[idx])         # ② 나무 한 그루
    probs.append(tree.predict_proba(Z_test)[:, 1])
probs = np.array(probs)                        # (100, 3750)
p_avg = probs.mean(axis=0)                     # ③ 사람마다 평균

print(probs.shape, " people in the last bag:", len(np.unique(idx)))
who = np.argsort(p_avg)[[1874, 3599, 3749]]    # 중간, 상위 4%, 최고
print(probs[:5, who].round(3))                 # 나무 5그루 × 3명
print("SD over trees:", probs[:, who].std(axis=0).round(3))
print("average      :", p_avg[who].round(3))
aucs = [roc_auc_score(y_test, p) for p in probs]
print("AUC, one tree:", round(min(aucs), 3), "to",
      round(max(aucs), 3))
print("AUC, average :", round(roc_auc_score(y_test, p_avg), 3))
''', title="부트스트랩으로 나무 100그루를 키워 평균 내기")
DATA_boot = grab("boot", lambda: dict(
    shape=list(ns["probs"].shape), uniq=int(len(np.unique(ns["idx"]))),
    first5=ns["probs"][:5, ns["who"]].round(3).tolist(), sd=ns["probs"][:, ns["who"]].std(axis=0).round(3).tolist(),
    avg=ns["p_avg"][ns["who"]].round(3).tolist(), auc_min=float(min(ns["aucs"])), auc_max=float(max(ns["aucs"])),
    auc_mean=float(np.mean(ns["aucs"])), auc_avg=float(ns["roc_auc_score"](ns["y_test"], ns["p_avg"])),
    who_age=ns["X_test"].iloc[ns["who"]]["age"].tolist(), who_ned=ns["X_test"].iloc[ns["who"]]["n_ed"].tolist(),
    who_ndrugs=ns["X_test"].iloc[ns["who"]]["n_drugs"].tolist(), who_y=ns["y_test"].iloc[ns["who"]].tolist()))
save("ml02_boot", c, marks={"(100, 3750)": 1, "people in the last bag: 7021": 2, "[[0.091 0.    0.762]": 3,
                            "SD over trees: [0.044 0.201 0.198]": 4, "average      : [0.019 0.18  0.745]": 5,
                            "AUC, one tree: 0.616 to 0.714": 6, "AUC, average : 0.774": 7})

c = nb.cell('''
from sklearn.ensemble import RandomForestClassifier

def mean_corr(P):
    """나무 두 그루씩 짝지은 예측(행)의 상관계수 평균"""
    C = np.corrcoef(P)
    return C[np.triu_indices_from(C, k=1)].mean()

for mf in [None, "sqrt"]:           # None = 모든 특성 = 배깅
    rf = RandomForestClassifier(n_estimators=200, max_features=mf,
                                min_samples_leaf=20, oob_score=True,
                                random_state=2026, n_jobs=-1)
    rf.fit(Z_train, y_train)
    P = np.array([t.predict_proba(Z_test)[:, 1]
                  for t in rf.estimators_])      # (200, 3750)
    one = np.mean([roc_auc_score(y_test, p) for p in P])
    oob = roc_auc_score(y_train, rf.oob_decision_function_[:, 1])
    print(f"max_features={mf}: corr {mean_corr(P):.3f}, "
          f"one tree AUC {one:.3f}, OOB AUC {oob:.3f}")
''', title="나무끼리 얼마나 닮았나 (배깅과 랜덤 포레스트)")
save("ml02_corr", c, marks={"corr 0.449": 1, "one tree AUC 0.681": 2, "OOB AUC 0.750": 3})

c = nb.cell('''
rf_pipe = make_tree_pipe(RandomForestClassifier(
    n_estimators=200, min_samples_leaf=20, oob_score=True,
    random_state=2026, n_jobs=-1))
t0 = time.perf_counter()
rf_pipe.fit(X_train, y_train)
t_oob = time.perf_counter() - t0
rf = rf_pipe.named_steps["model"]
print("oob_score_ (accuracy):", round(rf.oob_score_, 4))
p_oob = rf.oob_decision_function_[:, 1]       # 사람마다 OOB 확률
print("OOB AUC:", round(roc_auc_score(y_train, p_oob), 4),
      f" ({t_oob:.0f} s)")

t0 = time.perf_counter()
cv5 = cross_val_score(rf_pipe, X_train, y_train, cv=skf,
                      scoring="roc_auc")
t_cv5 = time.perf_counter() - t0
print("5-fold CV AUC:", round(cv5.mean(), 4), f" ({t_cv5:.0f} s)")
''', title="표본 밖 점수(OOB)와 5겹 교차검증")
_t = grab("t_oobcv", lambda: (ns["t_oob"], ns["t_cv5"]))
save("ml02_oob", c, marks={"oob_score_ (accuracy): 0.9596": 1, "OOB AUC: 0.7498": 2, "5-fold CV AUC: 0.7567": 3},
     subs=[(f" ({_t[0]:.0f} s)", f" ({secs('t_oob', round(_t[0])):.0f} s)"),
                          (f" ({_t[1]:.0f} s)", f" ({secs('t_cv5', round(_t[1])):.0f} s)")])

c = nb.cell('''
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

models = {
    "logistic (ch 0)": make_pipe(
        LogisticRegression(C=0.1, max_iter=1000)),
    "tree (ch 1)": make_tree_pipe(DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=300, random_state=2026)),
    "bagging": make_tree_pipe(BaggingClassifier(
        estimator=DecisionTreeClassifier(min_samples_leaf=20),
        n_estimators=200, random_state=2026, n_jobs=-1)),
    "random forest": make_tree_pipe(RandomForestClassifier(
        n_estimators=200, min_samples_leaf=20,
        random_state=2026, n_jobs=-1))}
rows, fitted = [], {}
for name, m in models.items():
    cv = cross_val_score(m, X_train, y_train, cv=skf,
                         scoring="roc_auc")
    fitted[name] = m.fit(X_train, y_train)   # 학습 자료 전체로
    p = fitted[name].predict_proba(X_test)[:, 1]
    rows.append([name, cv.mean(), cv.std(), roc_auc_score(y_test, p)])
pd.DataFrame(rows, columns=["model", "CV AUC", "CV SD", "test AUC"]
             ).round(3)
''', title="로지스틱 회귀, 나무 하나, 배깅, 랜덤 포레스트")
save("ml02_cmp1", c, dfmarks={"0.757": 1, "0.784": 2})

# ================================================================ 나. 랜덤 포레스트의 조절값
c = nb.cell('''
forest = make_tree_pipe(RandomForestClassifier(
    min_samples_leaf=20, oob_score=True, warm_start=True,
    random_state=2026, n_jobs=-1))
sizes = [25, 50, 75, 100, 150, 200, 300, 400, 500]
oob_auc = []
t0 = time.perf_counter()
for k in sizes:
    forest.set_params(model__n_estimators=k)  # 나무 수만 늘린다
    forest.fit(X_train, y_train)              # 늘어난 만큼만 새로 키움
    p = forest.named_steps["model"].oob_decision_function_[:, 1]
    oob_auc.append(roc_auc_score(y_train, p))
t_warm = time.perf_counter() - t0
print(np.round(oob_auc, 4), f" ({t_warm:.0f} s)")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(sizes, oob_auc, marker="o")
ax.set_xlabel("Number of trees (n_estimators)")
ax.set_ylabel("OOB AUC")
plt.tight_layout()
''', title="나무 수를 늘리며 OOB AUC 보기 (warm_start)")
_t = grab("t_warm", lambda: ns["t_warm"])
save("ml02_warm", c, marks={"[0.738": 1, "0.7514": 2}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_warm', round(_t)):.0f} s)")])

c = nb.cell('''
mf_list = [2, 4, 6, 12, 24, 48]       # 6 = "sqrt", 48 = 배깅
leaf_list = [1, 5, 20, 50, 100]
heat = pd.DataFrame(index=leaf_list, columns=mf_list, dtype=float)
for leaf in leaf_list:
    for mf in mf_list:
        m = make_tree_pipe(RandomForestClassifier(
            n_estimators=200, max_features=mf, min_samples_leaf=leaf,
            oob_score=True, random_state=2026, n_jobs=-1))
        m.fit(X_train, y_train)
        p = m.named_steps["model"].oob_decision_function_[:, 1]
        heat.loc[leaf, mf] = roc_auc_score(y_train, p)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
im = ax.imshow(heat.to_numpy(), cmap="viridis")
for i in range(len(leaf_list)):
    for j in range(len(mf_list)):
        v = heat.iloc[i, j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=9, color="black" if v > 0.74 else "white")
ax.set_xticks(range(len(mf_list)), mf_list)
ax.set_yticks(range(len(leaf_list)), leaf_list)
ax.set_xlabel("max_features")
ax.set_ylabel("min_samples_leaf")
ax.set_title("OOB AUC, 200 trees")
fig.colorbar(im)
heat.round(3)
''', title="max_features와 min_samples_leaf의 OOB AUC 열지도")
save("ml02_heat", c, dfmarks={"0.708": 1, "0.757": 2, "0.738": 3})

# ================================================================ 다. 무작위 탐색과 단계적 탐색
c = nb.cell('''
from scipy.stats import randint, uniform, loguniform

print(randint(2, 25).rvs(8, random_state=0))       # 2–24의 정수
print(uniform(0.3, 0.7).rvs(5, random_state=0).round(2))  # 0.3–1.0
print(loguniform(0.001, 100).rvs(5, random_state=0).round(4))
''', title="분포에서 조절값 뽑아 보기 (randint, uniform, loguniform)")
save("ml02_dists", c, marks={"[14 17 23  2  5  5  9 11]": 1, "[0.68 0.8  0.72 0.68 0.6 ]": 2,
                           "[0.5547 3.7666 1.0323 0.5302 0.1313]": 3})

c = nb.cell('''
from sklearn.model_selection import GridSearchCV

skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
rf_base = make_tree_pipe(RandomForestClassifier(
    n_estimators=200, random_state=2026, n_jobs=-1))
grid = {"model__max_features": [3, 6, 12, 24],
        "model__min_samples_leaf": [5, 20, 50, 100]}
gs = GridSearchCV(rf_base, grid, cv=skf3, scoring="roc_auc")
t0 = time.perf_counter()
gs.fit(X_train, y_train)
t_grid = time.perf_counter() - t0
print(gs.best_params_, round(gs.best_score_, 4))
print("candidates: 16, fits: 16 x 3 =", 16 * 3, f" ({t_grid:.0f} s)")
''', title="격자 탐색 16개 조합 (비교의 기준)")
_t = grab("t_grid", lambda: ns["t_grid"])
save("ml02_grid", c, marks={"0.759": 1, "fits: 16 x 3 = 48": 2}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_grid', round(_t)):.0f} s)")])

c = nb.cell('''
from sklearn.model_selection import RandomizedSearchCV

space = {"model__max_features": randint(2, 25),
         "model__min_samples_leaf": randint(1, 101),
         "model__max_samples": uniform(0.3, 0.7),
         "model__criterion": ["gini", "entropy"]}
rs = RandomizedSearchCV(rf_base, space, n_iter=40, cv=skf3,
                        scoring="roc_auc", random_state=2026)
t0 = time.perf_counter()
rs.fit(X_train, y_train)
t_rand = time.perf_counter() - t0
print(rs.best_params_)
print(round(rs.best_score_, 4), " fits: 40 x 3 =", 40 * 3,
      f" ({t_rand:.0f} s)")
''', title="무작위 탐색 40개 조합 (RandomizedSearchCV)")
_t = grab("t_rand", lambda: ns["t_rand"])
save("ml02_random", c, marks={"{'model__criterion': 'entropy'": 1, "0.7597  fits: 40 x 3 = 120": 2}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_rand', round(_t)):.0f} s)")])

c = nb.cell('''
res = pd.DataFrame(rs.cv_results_)
show = res[["param_model__max_features",
            "param_model__min_samples_leaf",
            "param_model__max_samples", "param_model__criterion",
            "mean_test_score", "std_test_score", "rank_test_score"]]
show.columns = ["max_feat", "min_leaf", "max_samp", "criterion",
                "mean AUC", "SD", "rank"]
print("best of the first 16 draws:",
      round(res["mean_test_score"][:16].max(), 4))
show.sort_values("rank").head(6).round(3)
''', title="무작위 탐색 결과를 표로 (cv_results_)")
save("ml02_randres", c, marks={"best of the first 16 draws: 0.7597": 1}, dfmarks={"0.760": 2})

c = nb.cell('''
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

hs = HalvingRandomSearchCV(
    rf_base, space, n_candidates=81, factor=3,
    resource="model__n_estimators", min_resources=22,
    max_resources=200, cv=skf3, scoring="roc_auc",
    random_state=2026)
t0 = time.perf_counter()
hs.fit(X_train, y_train)
t_halv = time.perf_counter() - t0
print(hs.best_params_)
print(round(hs.best_score_, 4), f" ({t_halv:.0f} s)")
steps = pd.DataFrame(hs.cv_results_).groupby("iter").agg(
    candidates=("mean_test_score", "size"),
    trees=("n_resources", "first"),
    best_AUC=("mean_test_score", "max"))
steps.round(4)
''', title="단계적 탐색 (HalvingRandomSearchCV)")
_t = grab("t_halv", lambda: ns["t_halv"])
save("ml02_halving", c, marks={"{'model__criterion': 'entropy'": 1, "0.7597": 2}, dfmarks={"0.7626": 3}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_halv', round(_t)):.0f} s)")])

c = nb.cell('''
hres = pd.DataFrame(hs.cv_results_)
trees_per_fold = [16 * 200, 40 * 200, hres["n_resources"].sum()]
summary = pd.DataFrame({
    "best CV AUC": [gs.best_score_, rs.best_score_, hs.best_score_],
    "candidates": [16, 40, hs.n_candidates_[0]],
    "fits": [16 * 3, 40 * 3, len(hres) * 3],
    "trees grown": [t * 3 for t in trees_per_fold],
    "seconds": np.round([t_grid, t_rand, t_halv]).astype(int)},
    index=["grid", "random", "halving"])
summary.round(4)
''', title="세 방법의 결과와 시간")
_tt = grab("t_three", lambda: [int(round(ns[k])) for k in ("t_grid", "t_rand", "t_halv")])
_fx = [secs("t_grid", _tt[0]), secs("t_rand", _tt[1]), secs("t_halv", _tt[2])]
save("ml02_cmp3", c, dfmarks={"0.7590": 1, "81": 2}, rowfix=list(zip(["grid", "random", "halving"], _fx)))

# ================================================================ 라. 베이즈 최적화
PIP_OUT = """Collecting optuna
  Downloading optuna-4.5.0-py3-none-any.whl (...)
...
Successfully installed ... optuna-4.5.0"""
c = nb.cell('''
!pip install optuna hyperopt
''', title="Optuna와 HyperOpt 설치 (Colab은 세션마다)", shell_output=PIP_OUT)
save("ml02_pip", c)

c = nb.cell('''
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    """조합 하나를 골라 3겹 교차검증 AUC를 돌려준다"""
    model = RandomForestClassifier(
        n_estimators=200, random_state=2026, n_jobs=-1,
        max_features=trial.suggest_int("max_features", 2, 24),
        min_samples_leaf=trial.suggest_int("min_samples_leaf",
                                           1, 100),
        max_samples=trial.suggest_float("max_samples", 0.3, 1.0),
        criterion=trial.suggest_categorical("criterion",
                                            ["gini", "entropy"]))
    auc = cross_val_score(make_tree_pipe(model), X_train, y_train,
                          cv=skf3, scoring="roc_auc")
    return auc.mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)  # 시도별 로그 끄기
study = optuna.create_study(direction="maximize",
                            sampler=TPESampler(seed=2026))
t0 = time.perf_counter()
study.optimize(objective, n_trials=40)
t_optuna = time.perf_counter() - t0
print(study.best_params)
print(round(study.best_value, 4), " trial", study.best_trial.number,
      f" ({t_optuna:.0f} s)")
''', title="Optuna로 40번 시도")
_t = grab("t_optuna", lambda: ns["t_optuna"])
save("ml02_optuna", c, marks={"{'max_features': 20": 1, "trial 16": 2, f" ({_t:.0f} s)": 3}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_optuna', round(_t)):.0f} s)")])

c = nb.cell('''
tdf = study.trials_dataframe(attrs=("number", "value", "params"))
print(tdf.shape)
tdf.sort_values("value", ascending=False).head(5).round(4)
''', title="시도 기록을 표로 (trials_dataframe)")
save("ml02_optuna_df", c, marks={"(40, 6)": 1}, dfmarks={"0.7594": 2})

c = nb.cell('''
from optuna.visualization.matplotlib import (
    plot_optimization_history, plot_param_importances, plot_slice)
from optuna.importance import FanovaImportanceEvaluator

ax1 = plot_optimization_history(study)
ax2 = plot_param_importances(
    study, evaluator=FanovaImportanceEvaluator(seed=2026))
ax3 = plot_slice(study)
''', title="Optuna 그림 세 가지 (이력, 조절값 중요도, 조각)")
save("ml02_optuna_plots", c)

c = nb.cell('''
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval

hp_space = {
    "max_features": hp.quniform("max_features", 2, 24, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 100, 1),
    "max_samples": hp.uniform("max_samples", 0.3, 1.0),
    "criterion": hp.choice("criterion", ["gini", "entropy"])}

def hp_objective(params):
    """조합 하나를 받아 손실(-AUC)을 돌려준다"""
    model = RandomForestClassifier(
        n_estimators=200, random_state=2026, n_jobs=-1,
        max_features=int(params["max_features"]),      # 12.0 → 12
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_samples=params["max_samples"],
        criterion=params["criterion"])
    auc = cross_val_score(make_tree_pipe(model), X_train, y_train,
                          cv=skf3, scoring="roc_auc").mean()
    return {"loss": -auc, "status": STATUS_OK}   # 작을수록 좋게

trials = Trials()
t0 = time.perf_counter()
best = fmin(hp_objective, hp_space, algo=tpe.suggest, max_evals=40,
            trials=trials, rstate=np.random.default_rng(2026))
t_hyperopt = time.perf_counter() - t0
print(best)                          # hp.choice는 번호로 나온다
print(space_eval(hp_space, best))    # 실제 값으로 바꾼 것
print(round(-min(trials.losses()), 4), f" ({t_hyperopt:.0f} s)")
''', title="HyperOpt로 40번 시도")
_t = grab("t_hyperopt", lambda: ns["t_hyperopt"])


def tqdm_last(cell, total_sec, n):
    """fmin의 진행 막대(\r로 덮어쓰는 줄)를 마지막 모습 한 줄로 줄이고 시간 부분을 고정값으로 바꾼다."""
    head, sep, rest = cell.stdout.partition("\n")
    if "\r" not in head:
        return
    last = [x for x in head.split("\r") if x.strip()][-1].strip()
    mm, ss = divmod(int(total_sec), 60)
    last = re.sub(r"\[\d\d:\d\d<00:00, +[\d.]+s/trial", f"[{mm:02d}:{ss:02d}<00:00,  {total_sec / n:.2f}s/trial", last)
    cell.stdout = last + sep + rest


tqdm_last(c, secs("t_hyperopt", round(_t)), 40)
save("ml02_hyperopt", c, marks={"{'criterion': np.int64(0)": 1, "{'criterion': 'gini'": 2,
                               "0.7598": 3}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_hyperopt', round(_t)):.0f} s)")])

c = nb.cell('''
hdf = pd.DataFrame(trials.vals)        # 시도마다 뽑힌 값
hdf["criterion"] = hdf["criterion"].map({0: "gini", 1: "entropy"})
hdf["AUC"] = [-r["loss"] for r in trials.results]
print(hdf.shape)

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(hdf.index, hdf["AUC"], "o", alpha=0.6, label="each trial")
ax.plot(hdf.index, hdf["AUC"].cummax(), label="best so far")
ax.set_xlabel("Trial")
ax.set_ylabel("CV AUC (3-fold)")
ax.legend()
plt.tight_layout()
hdf.sort_values("AUC", ascending=False).head(5).round(4)
''', title="HyperOpt의 Trials를 표와 그림으로")
save("ml02_hyperopt_df", c, marks={"(40, 5)": 1}, dfmarks={"0.7598": 2})

c = nb.cell('''
so_far = pd.DataFrame({
    "random": res["mean_test_score"].cummax().to_numpy(),
    "Optuna": tdf["value"].cummax().to_numpy(),
    "HyperOpt": hdf["AUC"].cummax().to_numpy()},
    index=range(1, 41))
ax = so_far.plot(figsize=(6.5, 3.2), drawstyle="steps-post")
ax.set_xlabel("Trial")
ax.set_ylabel("Best CV AUC so far")
plt.tight_layout()

pd.DataFrame({"best CV AUC": so_far.iloc[-1],
              "first reached at trial": so_far.idxmax(),
              "seconds": np.round([t_rand, t_optuna,
                                   t_hyperopt]).astype(int)}).round(4)
''', title="무작위 탐색, Optuna, HyperOpt 비교 (같은 범위, 40번씩)")
_tt = grab("t_bayes", lambda: [int(round(ns[k])) for k in ("t_rand", "t_optuna", "t_hyperopt")])
_fx = [secs("t_rand", _tt[0]), secs("t_optuna", _tt[1]), secs("t_hyperopt", _tt[2])]
save("ml02_cmp_bayes", c, dfmarks={"best CV AUC": 1, "first reached at trial": 2, "seconds": 3}, rowfix=list(zip(["random", "Optuna", "HyperOpt"], _fx)))

# ================================================================ 마. 변수 중요도와 결과 정리
c = nb.cell('''
import joblib

best_rf = study.best_params                  # 라 절에서 Optuna가 찾은 조합
final = make_tree_pipe(RandomForestClassifier(
    n_estimators=500, random_state=2026, n_jobs=-1, **best_rf))
final.fit(X_train, y_train)                  # 학습 자료 전체로 다시 맞춤
p_final = final.predict_proba(X_test)[:, 1]  # 시험 자료는 여기서 한 번

def boot_ci(y_true, p, n_boot=1000, seed=2026):
    """시험 자료를 복원추출해 AUC의 95% 백분위수 구간"""
    rng = np.random.default_rng(seed)
    yt, out = np.asarray(y_true), []
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        out.append(roc_auc_score(yt[i], p[i]))
    return np.percentile(out, [2.5, 97.5])

lo, hi = boot_ci(y_test, p_final)
print(f"test AUC {roc_auc_score(y_test, p_final):.3f}"
      f" (95% CI {lo:.3f} to {hi:.3f})")
joblib.dump(final, "rf_final.joblib")        # 저장 (1장 마 절과 같음)
''', title="최종 모형: 학습 자료 전체로 맞추고 시험 자료로 한 번 평가")
save("ml02_final", c, marks={"test AUC 0.770 (95% CI 0.722 to 0.815)": 1})

c = nb.cell('''
names = final[:-1].get_feature_names_out()   # 전처리 뒤 48열의 이름
print(names[:3], names[-3:])

def original(name):
    """원-핫 열 이름을 원래 열로: 'region_metro' → 'region'"""
    for c in cat_cols:
        if name.startswith(c + "_"):
            return c
    return name

mdi = pd.Series(final[-1].feature_importances_, index=names)
mdi = mdi.groupby([original(nm) for nm in names]).sum()  # 39개로
print(len(mdi), round(mdi.sum(), 3))
mdi.sort_values(ascending=False).head(8).round(3)
''', title="불순도 기반 중요도 (feature_importances_)")
save("ml02_mdi", c, marks={"['age' 'female' 'income_q']": 1, "39 1.0": 2}, dfmarks={"0.077": 3})

c = nb.cell('''
from sklearn.inspection import permutation_importance

perm = permutation_importance(final, X_test, y_test,
                              scoring="roc_auc", n_repeats=10,
                              random_state=2026)
imp = pd.DataFrame({"impurity": mdi,
                    "perm": pd.Series(perm.importances_mean,
                                      index=X_test.columns),
                    "perm_sd": pd.Series(perm.importances_std,
                                         index=X_test.columns)})
imp["rank_imp"] = imp["impurity"].rank(ascending=False).astype(int)
imp["rank_perm"] = imp["perm"].rank(ascending=False).astype(int)

fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
a = imp.sort_values("impurity").tail(15)
ax[0].barh(a.index, a["impurity"])
ax[0].set_title("Impurity-based (training data)")
b = imp.sort_values("perm").tail(15)
ax[1].barh(b.index, b["perm"], xerr=b["perm_sd"])
ax[1].set_title("Permutation (test data, AUC drop)")
plt.tight_layout()
check = ["age", "n_drugs", "n_ed", "bmi", "hf", "ckd", "insulin",
         "dyslip", "oa", "region", "n_prescribers"]
imp.loc[check, ["rank_imp", "rank_perm", "perm"]].round(4)
''', title="순열 중요도와 나란히 보기 (permutation_importance)", max_rows=15)
save("ml02_perm", c, dfmarks={"0.0507": 1, "0.0329": 2, "-0.0002": 3, "0.0001": 4})

c = nb.cell('''
util = ["n_outpt", "n_drugs", "n_prescribers", "cost_2022"]
print(X_train[util].corr(method="spearman").round(2))

rng = np.random.default_rng(2026)
base = roc_auc_score(y_test, p_final)
drops = []
for _ in range(10):
    Xs = X_test.copy()
    order = rng.permutation(len(Xs))       # 네 열을 같은 순서로 섞기
    Xs[util] = X_test[util].to_numpy()[order]
    drops.append(base - roc_auc_score(
        y_test, final.predict_proba(Xs)[:, 1]))
print("alone   :", imp.loc[util, "perm"].round(4).tolist())
print("together:", round(np.mean(drops), 4))
''', title="서로 상관된 특성은 중요도를 나눠 갖는다")
save("ml02_group", c, marks={"0.73": 1, "alone   : [0.0023, 0.009, 0.0, -0.0022]": 2, "together: 0.0101": 3})

c = nb.cell('''
rng = np.random.default_rng(7)
Xn_train = X_train.assign(noise_num=rng.normal(size=len(X_train)),
                          noise_bin=rng.integers(0, 2, len(X_train)))
Xn_test = X_test.assign(noise_num=rng.normal(size=len(X_test)),
                        noise_bin=rng.integers(0, 2, len(X_test)))
noisy = make_tree_pipe(RandomForestClassifier(
    n_estimators=200, random_state=2026, n_jobs=-1, **best_rf),
    num=num_cols + ["noise_num", "noise_bin"])
noisy.fit(Xn_train, y_train)

nm = noisy[:-1].get_feature_names_out()
mdi_n = pd.Series(noisy[-1].feature_importances_, index=nm)
mdi_n = mdi_n.groupby([original(x) for x in nm]).sum()
perm_n = permutation_importance(noisy, Xn_test, y_test,
                                scoring="roc_auc", n_repeats=10,
                                random_state=2026)
out = pd.DataFrame({
    "impurity": mdi_n,
    "rank_imp": mdi_n.rank(ascending=False).astype(int),
    "perm": pd.Series(perm_n.importances_mean,
                      index=Xn_test.columns)})
out.loc[["noise_num", "noise_bin", "hf", "insulin"]].round(4)
''', title="아무 의미 없는 열을 더해 보면")
save("ml02_noise", c, dfmarks={"0.0248": 1, "-0.0028": 2})

c = nb.cell('''
from sklearn.inspection import PartialDependenceDisplay

screened = X_train[X_train["bmi"].notna()]    # BMI가 있는 사람만
X_pdp = screened.sample(1000, random_state=2026)
X_pdp = X_pdp.astype({c: float for c in num_cols})  # 정수 열을 실수로
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
pdp = PartialDependenceDisplay.from_estimator(
    final, X_pdp, ["age", "bmi", "n_ed", "n_drugs"], ax=ax,
    percentiles=(0.01, 0.99), grid_resolution=40)
plt.tight_layout()
''', title="부분 의존 그림 (PartialDependenceDisplay)")
save("ml02_pdp", c)

c = nb.cell('''
from sklearn.metrics import roc_curve

p_test = {name: m.predict_proba(X_test)[:, 1]
          for name, m in fitted.items()}       # 셀 5에서 맞춘 네 모형
p_test["tuned forest"] = p_final
cv_final = cross_val_score(final, X_train, y_train, cv=skf,
                           scoring="roc_auc")
cv_auc = [r[1] for r in rows] + [cv_final.mean()]

table, fig = [], plt.figure(figsize=(5.2, 4.8))
for (name, p), cv in zip(p_test.items(), cv_auc):
    auc = roc_auc_score(y_test, p)
    lo, hi = boot_ci(y_test, p)
    table.append([name, cv, auc, lo, hi])
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("1 - specificity")
plt.ylabel("Sensitivity")
plt.legend(fontsize=8)
pd.DataFrame(table, columns=["model", "CV AUC", "test AUC",
                             "CI low", "CI high"]).round(3)
''', title="모형 비교 표와 ROC 곡선")
save("ml02_cmp_final", c, dfmarks={"0.770": 1, "0.784": 2})

# ================================================================ 바. 과제
c = nb.cell('''
from sklearn.metrics import confusion_matrix

def objective_bal(trial):
    model = RandomForestClassifier(
        n_estimators=200, random_state=2026, n_jobs=-1,
        class_weight="balanced_subsample",
        max_features=trial.suggest_int("max_features", 2, 24),
        min_samples_leaf=trial.suggest_int("min_samples_leaf",
                                           1, 100),
        max_samples=trial.suggest_float("max_samples", 0.3, 1.0),
        criterion=trial.suggest_categorical("criterion",
                                            ["gini", "entropy"]))
    return cross_val_score(make_tree_pipe(model), X_train, y_train,
                           cv=skf3, scoring="roc_auc").mean()

study_bal = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=2026))
study_bal.optimize(objective_bal, n_trials=20)
print(study_bal.best_params, round(study_bal.best_value, 4))

bal = make_tree_pipe(RandomForestClassifier(
    n_estimators=500, random_state=2026, n_jobs=-1,
    class_weight="balanced_subsample", **study_bal.best_params))
bal.fit(X_train, y_train)
p_bal = bal.predict_proba(X_test)[:, 1]
for name, p in [("default", p_final), ("balanced_subsample", p_bal)]:
    print(f"{name}: test AUC {roc_auc_score(y_test, p):.3f},"
          f" mean prob {p.mean():.3f}")
    print(confusion_matrix(y_test, (p >= 0.10).astype(int)))
''', title="과제 1 정답. class_weight를 바꿔 다시 튜닝")
save("ml02_hw1", c)

c = nb.cell('''
study.optimize(objective, n_trials=40)       # 40번을 더 (모두 80번)
best_by = study.trials_dataframe()["value"].cummax()
for n_t in [10, 20, 40, 80]:
    print(f"after {n_t:2d} trials:",
          f"best CV AUC {best_by[n_t - 1]:.4f}")
print(study.best_params)
''', title="과제 2 정답. 시도 수에 따른 최고 점수")
save("ml02_hw2", c)

c = nb.cell('''
out = []
for k in [10, 50, 100, 200, 500, 1000]:
    m = make_tree_pipe(RandomForestClassifier(
        n_estimators=k, random_state=2026, n_jobs=-1, **best_rf))
    t0 = time.perf_counter()
    m.fit(X_train, y_train)
    sec = time.perf_counter() - t0
    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    out.append([k, round(auc, 4), round(sec, 1)])
pd.DataFrame(out, columns=["n_estimators", "test AUC", "seconds"])
''', title="과제 3 정답. 나무 수에 따른 시험 AUC와 시간")
_tt = grab("t_hw3", lambda: [r[2] for r in ns["out"]])
_fx = SECONDS["hw3"] or _tt
save("ml02_hw3", c, rowfix=list(zip(range(6), _fx)))

print("ml02: cells", len(nb.cells), f" elapsed {time.time() - T_START:.0f} s")
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)


# ================================================================ 자료 모으기 (전체 실행 때만): 그림과 대조 블록에 쓸 값
def collect():
    import warnings
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import partial_dependence
    from sklearn.metrics import roc_auc_score
    D = DATA
    Xtr, Xte, ytr, yte = ns["X_train"], ns["X_test"], ns["y_train"], ns["y_test"]
    mp = ns["make_tree_pipe"]
    D["setup"] = [len(Xtr), len(Xte), int(ytr.sum()), int(yte.sum())]
    D["corr_out"] = nb.cells[2].stdout
    D["oob"] = dict(acc=float(ns["rf"].oob_score_), auc=float(roc_auc_score(ytr, ns["p_oob"])),
                    cv_mean=float(ns["cv5"].mean()), cv_sd=float(ns["cv5"].std()))
    D["cmp1"] = [[r[0], float(r[1]), float(r[2]), float(r[3])] for r in ns["rows"]]
    D["warm"] = dict(sizes=list(ns["sizes"]), auc=[float(v) for v in ns["oob_auc"]])
    D["heat"] = dict(index=list(ns["leaf_list"]), columns=list(ns["mf_list"]),
                     values=ns["heat"].to_numpy().astype(float).tolist())
    gs, rs, hs = ns["gs"], ns["rs"], ns["hs"]
    D["grid"] = dict(mf=[int(p["model__max_features"]) for p in gs.cv_results_["params"]],
                     leaf=[int(p["model__min_samples_leaf"]) for p in gs.cv_results_["params"]],
                     score=[float(v) for v in gs.cv_results_["mean_test_score"]],
                     best=dict(gs.best_params_), best_score=float(gs.best_score_))
    D["rand"] = dict(mf=[int(p["model__max_features"]) for p in rs.cv_results_["params"]],
                     leaf=[int(p["model__min_samples_leaf"]) for p in rs.cv_results_["params"]],
                     samp=[float(p["model__max_samples"]) for p in rs.cv_results_["params"]],
                     crit=[p["model__criterion"] for p in rs.cv_results_["params"]],
                     score=[float(v) for v in rs.cv_results_["mean_test_score"]],
                     best={k: (v if isinstance(v, str) else float(v)) for k, v in rs.best_params_.items()},
                     best_score=float(rs.best_score_),
                     best16=float(ns["res"]["mean_test_score"][:16].max()))
    hres = pd.DataFrame(hs.cv_results_)
    D["halv"] = dict(best={k: (v if isinstance(v, str) else float(v)) for k, v in hs.best_params_.items()},
                     best_score=float(hs.best_score_), n_candidates=[int(v) for v in hs.n_candidates_],
                     n_resources=[int(v) for v in hs.n_resources_], n_fits=int(len(hres) * 3),
                     trees=int(hres["n_resources"].sum() * 3),
                     steps=ns["steps"].reset_index().astype(float).values.tolist())
    st = ns["study"]
    tr40 = [t for t in st.trials if t.number < 40]
    D["optuna"] = dict(values=[float(t.value) for t in tr40],
                       params=[{k: (v if isinstance(v, str) else float(v)) for k, v in t.params.items()} for t in tr40],
                       best=ns["best_rf"], best_value=float(max(t.value for t in tr40)),
                       best_number=int(max(tr40, key=lambda t: t.value).number))
    from optuna.importance import get_param_importances, FanovaImportanceEvaluator
    import optuna
    st40 = optuna.create_study(direction="maximize")
    st40.add_trials(tr40)
    D["optuna_imp"] = {k: float(v) for k, v in
                       get_param_importances(st40, evaluator=FanovaImportanceEvaluator(seed=2026)).items()}
    hdf = ns["hdf"]
    D["hyperopt"] = dict(auc=[float(v) for v in hdf["AUC"]], best=ns["space_eval"](ns["hp_space"], ns["best"]),
                         best_raw={k: float(v) for k, v in ns["best"].items()},
                         mf=[float(v) for v in hdf["max_features"]], leaf=[float(v) for v in hdf["min_samples_leaf"]])
    D["final"] = dict(auc=float(roc_auc_score(yte, ns["p_final"])), lo=float(ns["lo"]), hi=float(ns["hi"]))
    imp = ns["imp"]
    D["imp"] = {k: imp[k].astype(float).tolist() for k in imp.columns}
    D["imp_index"] = list(imp.index)
    D["group"] = dict(together=float(np.mean(ns["drops"])), alone=float(imp.loc[ns["util"], "perm"].sum()),
                      corr=ns["X_train"][ns["util"]].corr(method="spearman").round(2).values.tolist())
    out_n = ns["out"] if False else None  # noqa
    D["noise"] = {k: [float(v) for v in ns["mdi_n"].rank(ascending=False).loc[["noise_num", "noise_bin", "hf", "insulin"]]]
                  for k in ["rank"]}
    pn = pd.Series(ns["perm_n"].importances_mean, index=ns["Xn_test"].columns)
    D["noise"]["perm"] = [float(pn[k]) for k in ["noise_num", "noise_bin", "hf", "insulin"]]
    D["noise"]["imp"] = [float(ns["mdi_n"][k]) for k in ["noise_num", "noise_bin", "hf", "insulin"]]
    D["noise"]["n_feat"] = int(len(ns["mdi_n"]))
    D["pdp"] = {f: dict(grid=[float(v) for v in r["grid_values"][0]], avg=[float(v) for v in r["average"][0]])
                for f, r in zip(["age", "bmi", "n_ed", "n_drugs"], ns["pdp"].pd_results)}
    D["cmp_final"] = [[r[0]] + [float(v) for v in r[1:]] for r in ns["table"]]
    sb = ns["study_bal"]
    D["hw1"] = dict(best=sb.best_params, best_value=float(sb.best_value),
                    auc=float(roc_auc_score(yte, ns["p_bal"])), mean=float(ns["p_bal"].mean()),
                    mean_default=float(ns["p_final"].mean()))
    cm = ns["confusion_matrix"]
    D["hw1"]["cm_default"] = cm(yte, (ns["p_final"] >= 0.10).astype(int)).tolist()
    D["hw1"]["cm_bal"] = cm(yte, (ns["p_bal"] >= 0.10).astype(int)).tolist()
    bb = ns["best_by"]
    D["hw2"] = dict(best_by=[float(bb[n - 1]) for n in ((5, 10, 12, 24) if SMOKE else (10, 20, 40, 80))],
                    best=ns["study"].best_params)
    D["hw3"] = [[int(r[0]), float(r[1]), float(r[2])] for r in ns["out"]]

    # 나 절 조절값 표에 적는 숫자: 기준(200그루, sqrt, 잎 20명)에서 하나씩 바꾼 OOB AUC와 평균 OOB 확률
    def oob(**kw):
        params = dict(n_estimators=10 if SMOKE else 200, min_samples_leaf=20, oob_score=True, random_state=2026,
                      n_jobs=-1)
        params.update(kw)
        m = mp(RandomForestClassifier(**params)).fit(Xtr, ytr).named_steps["model"]
        p = m.oob_decision_function_[:, 1]
        return [float(roc_auc_score(ytr, p)), float(p.mean())]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D["param_effects"] = {
            "base": oob(), "max_depth=6": oob(max_depth=6), "max_depth=12": oob(max_depth=12),
            "max_samples=0.5": oob(max_samples=0.5), "criterion=entropy": oob(criterion="entropy"),
            "balanced": oob(class_weight="balanced"), "balanced_subsample": oob(class_weight="balanced_subsample"),
            "msl1_mss100": oob(min_samples_leaf=1, min_samples_split=100), "max_leaf_nodes=64": oob(max_leaf_nodes=64),
            "ccp_alpha=0.0005": oob(ccp_alpha=0.0005), "min_impurity_decrease=1e-4": oob(min_impurity_decrease=1e-4),
            "default": oob(min_samples_leaf=1),
        }
        # 마 절: 부분 의존의 몇 점 (그림에서 읽는 값)
        D["pdp_pts"] = {}
        for f, pts in [("age", [50, 60, 70, 75, 80, 85, 87]), ("bmi", [18, 20, 23, 24, 25, 28, 32]),
                       ("n_ed", [0, 1, 2, 3]), ("n_drugs", [5, 8, 9, 10, 11, 15])]:
            g, a = np.array(D["pdp"][f]["grid"]), np.array(D["pdp"][f]["avg"])
            D["pdp_pts"][f] = {str(v): float(np.interp(v, g, a)) for v in pts}
        # 참 구조: 나이대별 입원율 (공개 자료)
        df_ = ns["df"]
        D["true_age_rates"] = {str(k): float(v) for k, v in
                               df_.groupby(pd.cut(df_["age"], [39, 59, 69, 74, 79, 84, 99]), observed=True)["admit_2023"].mean().items()}


if not REPLAY:
    collect()
    if CACHE:
        cells = [dict(code=cc.code, stdout=cc.stdout, value_repr=cc.value_repr, value_html=cc.value_html,
                      images=cc.images, warns=cc.warns, error=cc.error) for cc in nb.cells]
        with open(CACHE, "wb") as f:
            pickle.dump({"cells": cells, "data": DATA}, f)
        print("cache written:", CACHE)
print(f"collected, elapsed {time.time() - T_START:.0f} s")
print(json.dumps({k: DATA[k] for k in ["setup", "oob", "cmp1", "warm", "grid", "halv", "final", "group", "noise",
                                       "cmp_final", "hw1", "hw2", "hw3", "param_effects", "pdp_pts", "true_age_rates",
                                       "optuna_imp"]}, ensure_ascii=False, default=str)[:20000])
if MARK_ERRORS:
    print("MARK ERRORS:")
    for e in MARK_ERRORS:
        print("  ", e)


# ================================================================ 대조 블록: 본문에 적은 숫자와 실행 결과 맞추기
def checks():
    n_ok = [0]

    def same(label, a, b, tol=1e-9):
        a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
        assert a.shape == b.shape and np.all(np.abs(a - b) <= tol), f"{label}: {a} != {b}"
        n_ok[0] += 1

    D = DATA
    # 가 절 숫자로 따라가기: 환자 10명, seed 30으로 뽑은 주머니 세 개, 한 번 나누는 나무
    age = np.array([52, 61, 47, 78, 83, 69, 88, 58, 74, 81])
    ed = np.array([0, 0, 1, 2, 0, 0, 3, 1, 2, 0])
    yy = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 1])

    def gini(v):
        return 0.0 if len(v) == 0 else 2 * v.mean() * (1 - v.mean())

    def stump(idx, feats=("age", "ed")):
        best = None
        for f in feats:
            x = age if f == "age" else ed
            xs = np.unique(x[idx])
            for a, b in zip(xs[:-1], xs[1:]):
                t = (a + b) / 2
                L, R = idx[x[idx] <= t], idx[x[idx] > t]
                g = (len(L) * gini(yy[L]) + len(R) * gini(yy[R])) / len(idx)
                if best is None or g < best[0] - 1e-12:
                    best = (g, f, t, yy[L].mean(), yy[R].mean())
        return best

    def predict(st, a, e):
        x = a if st[1] == "age" else e
        return st[4] if x > st[2] else st[3]
    full = stump(np.arange(10))
    assert full[1] == "age" and full[2] == 76.0 and full[4] == 0.75 and full[3] == 0.0
    same("K by full-data stump", predict(full, 72, 2), 0.0)
    rng = np.random.default_rng(30)
    bags = [rng.integers(0, 10, 10) for _ in range(3)]
    letters = "ABCDEFGHIJ"
    assert ["".join(sorted(letters[i] for i in b)) for b in bags] == ["ABCCDEFGHH", "ABBBDDFHHI", "BCCCDEGGHJ"]
    sts = [stump(b) for b in bags]
    assert [(st[1], st[2]) for st in sts] == [("ed", 1.5), ("age", 76.0), ("age", 69.5)]
    pk = [predict(st, 72, 2) for st in sts]
    same("K per bag and mean", pk + [np.mean(pk)], [1.0, 0.0, 0.8, 0.6])
    same("J OOB (bags 1, 2)", np.mean([predict(sts[0], 81, 0), predict(sts[1], 81, 0)]), 0.5)
    st3 = stump(bags[2], ("ed",))
    same("bag 3 with n_ed only", [st3[2], st3[4], predict(st3, 72, 2)], [1.5, 1.0, 1.0])
    same("P(drawn at least once)", round(1 - (1 - 1 / 11250) ** 11250, 3), 0.632)
    same("all-zero accuracy, train", round(1 - 459 / 11250, 4), 0.9592)
    # 셀 출력의 핵심 숫자
    same("split", D["setup"], [11250, 3750, 459, 153])
    b = D["boot"]
    same("boot", [b["uniq"], round(b["auc_min"], 3), round(b["auc_max"], 3), round(b["auc_avg"], 3)],
         [7021, 0.616, 0.714, 0.774])
    for t in ["corr 0.525", "corr 0.449", "one tree AUC 0.688", "one tree AUC 0.681", "OOB AUC 0.738", "OOB AUC 0.750"]:
        assert t in D["corr_out"], t
    n_ok[0] += 1
    o = D["oob"]
    same("OOB", [round(o["acc"], 4), round(o["auc"], 4), round(o["cv_mean"], 4)], [0.9596, 0.7498, 0.7567])
    same("cmp1 CV", [round(r[1], 3) for r in D["cmp1"]], [0.749, 0.754, 0.734, 0.757])
    same("cmp1 test", [round(r[3], 3) for r in D["cmp1"]], [0.742, 0.771, 0.784, 0.772])
    same("warm", [round(D["warm"]["auc"][0], 3), round(D["warm"]["auc"][3], 4), round(D["warm"]["auc"][-1], 4)],
         [0.738, 0.7503, 0.7514])
    H = np.array(D["heat"]["values"])
    same("heat default, max, bagging(20)", [round(H[0, 2], 3), round(H.max(), 3), round(H[2, 5], 3)], [0.708, 0.757, 0.738])
    same("heat leaf-1 row range, leaf>=20 range", [round(H[0].min(), 3), round(H[0].max(), 3), round(H[2:].min(), 3)],
         [0.706, 0.724, 0.738])
    same("searches", [round(D["grid"]["best_score"], 4), round(D["rand"]["best_score"], 4), round(D["rand"]["best16"], 4),
                      round(D["halv"]["best_score"], 4)], [0.7590, 0.7597, 0.7597, 0.7597])
    same("halving steps", [D["halv"]["n_candidates"], D["halv"]["n_resources"]], [[81, 27, 9], [22, 66, 198]])
    same("optuna", [round(D["optuna"]["best_value"], 4), D["optuna"]["best_number"]], [0.7594, 16])
    same("hyperopt", round(max(D["hyperopt"]["auc"]), 4), 0.7598)
    f = D["final"]
    same("final test AUC and CI", [round(f["auc"], 3), round(f["lo"], 3), round(f["hi"], 3)], [0.770, 0.722, 0.815])
    # 나 절 조절값 표의 숫자 (기준에서 하나씩 바꾼 OOB AUC)
    pe = {k: round(v[0], 3) for k, v in D["param_effects"].items()}
    same("param table", [pe["base"], pe["max_samples=0.5"], pe["max_depth=6"], pe["max_depth=12"], pe["criterion=entropy"],
                         pe["balanced_subsample"], pe["msl1_mss100"], pe["max_leaf_nodes=64"],
                         pe["min_impurity_decrease=1e-4"], pe["ccp_alpha=0.0005"], pe["default"]],
         [0.750, 0.754, 0.755, 0.755, 0.746, 0.755, 0.747, 0.755, 0.751, 0.740, 0.708])
    same("class_weight mean OOB prob", [round(D["param_effects"]["base"][1], 3),
                                        round(D["param_effects"]["balanced_subsample"][1], 3)], [0.040, 0.264])
    assert D["param_effects"]["balanced"][0] == 0.0     # 1.9.1: 'balanced'는 입원자의 OOB 예측이 생기지 않음
    n_ok[0] += 1
    # 라 절
    same("optuna importance", [round(D["optuna_imp"][k], 2) for k in ("min_samples_leaf", "max_features", "max_samples",
                                                                       "criterion")], [0.73, 0.13, 0.09, 0.06])
    ov = np.array(D["optuna"]["values"])
    leaf = np.array([q["min_samples_leaf"] for q in D["optuna"]["params"]])
    same("optuna first 10 range, later >0.75, later leaf>=73",
         [round(ov[:10].min(), 3), round(ov[:10].max(), 3), (ov[10:] > 0.75).sum(), (leaf[10:] >= 73).sum()],
         [0.741, 0.759, 28, 25])
    same("lowest three trials' leaf", sorted(leaf[np.argsort(ov)[:3]]), [1, 20, 36])
    hy = np.array(D["hyperopt"]["auc"])
    same("hyperopt best trial (1-based)", int(np.argmax(hy)) + 1, 31)
    # 마 절
    imp = D["imp"]
    idx = D["imp_index"]
    rk = {k: (int(imp["rank_imp"][i]), int(imp["rank_perm"][i])) for i, k in enumerate(idx)}
    same("ranks (impurity, permutation)", [rk[k] for k in ["age", "n_drugs", "bmi", "n_ed", "hf", "insulin", "dyslip", "oa",
                                                            "region", "n_prescribers", "cost_2022", "dementia", "egfr"]],
         [(1, 1), (2, 3), (6, 2), (23, 14), (21, 34), (16, 6), (20, 29), (22, 15), (11, 19), (17, 21), (4, 39), (3, 8), (9, 38)])
    same("group", [round(D["group"]["together"], 4), round(D["group"]["alone"], 4)], [0.0101, 0.0091])
    same("noise", [D["noise"]["rank"][0], round(D["noise"]["perm"][0], 4), D["noise"]["rank"][2], D["noise"]["rank"][3],
                   D["noise"]["rank"][1]], [8, -0.0028, 15, 18, 19])
    pp = {f: {k: round(v, 3) for k, v in d.items()} for f, d in D["pdp_pts"].items()}
    same("pdp age", [pp["age"][k] for k in ("50", "70", "75", "80", "85", "87")], [0.023, 0.027, 0.032, 0.068, 0.110, 0.124])
    same("pdp bmi, n_ed, n_drugs", [pp["bmi"]["18"], pp["bmi"]["24"], pp["bmi"]["32"], pp["n_ed"]["0"], pp["n_ed"]["2"],
                                    pp["n_drugs"]["9"], pp["n_drugs"]["10"]],
         [0.065, 0.025, 0.045, 0.029, 0.032, 0.045, 0.115])
    same("pdp age grid end (99th percentile)", D["pdp"]["age"]["grid"][-1], 87.0)
    ga, aa = np.array(D["pdp"]["age"]["grid"]), np.array(D["pdp"]["age"]["avg"])
    same("pdp age up to 70 (0.022-0.027)", [round(aa[ga <= 70].min(), 3), round(aa[ga <= 70].max(), 3)], [0.022, 0.027])
    same("pdp n_ed grid", D["pdp"]["n_ed"]["grid"], [0, 1, 2, 3, 5])
    same("pdp n_drugs grid end", D["pdp"]["n_drugs"]["grid"][-1], 17.0)
    # 라 절 셀 16 아래 글: Optuna 위쪽 5개의 후보 수 18-20개, 표본 비율 73-89%
    top5 = [D["optuna"]["params"][i] for i in np.argsort(-np.array(D["optuna"]["values"]))[:5]]
    same("optuna top5 max_features and max_samples range",
         [min(q["max_features"] for q in top5), max(q["max_features"] for q in top5),
          round(min(q["max_samples"] for q in top5), 2), round(max(q["max_samples"] for q in top5), 2)], [18, 20, 0.73, 0.89])
    same("true admission rates by age", [round(v, 3) for v in list(D["true_age_rates"].values())[2:5]], [0.030, 0.064, 0.123])
    cf = [[round(v, 3) for v in r[1:]] for r in D["cmp_final"]]
    same("cmp_final", cf, [[0.749, 0.742, 0.695, 0.785], [0.754, 0.771, 0.726, 0.810], [0.734, 0.784, 0.740, 0.825],
                           [0.757, 0.772, 0.726, 0.816], [0.757, 0.770, 0.722, 0.815]])
    h1 = D["hw1"]
    same("hw1", [round(h1["best_value"], 4), round(h1["auc"], 3), round(h1["mean"], 3), round(h1["mean_default"], 3)],
         [0.7589, 0.756, 0.302, 0.040])
    same("hw1 confusion matrices", [h1["cm_default"], h1["cm_bal"]], [[[3360, 237], [85, 68]], [[96, 3501], [0, 153]]])
    same("hw1 recall, precision, flagged share", [round(68 / 153, 2), round(68 / 305, 2), round(3654 / 3750, 2)],
         [0.44, 0.22, 0.97])
    same("hw2", [round(v, 4) for v in D["hw2"]["best_by"]], [0.7585, 0.7594, 0.7594, 0.7605])
    same("hw3 test AUC", [r[1] for r in D["hw3"]], [0.7695, 0.7753, 0.7755, 0.7720, 0.7696, 0.7704])
    print(f"VERIFY: {n_ok[0]} checks passed")
    return n_ok[0]


N_CHECKS = None if SMOKE else checks()


# ================================================================ 그림 2-1, 2-2 (SVG, 실제 탐색 결과로)
def _svgplot():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import svgplot
    return svgplot


def fig_grid_random():
    sp = _svgplot()
    panels = []
    for key, title in [("grid", "Grid search, 16 candidates"), ("rand", "Random search, 40 candidates")]:
        d = DATA[key]
        n = len(d["score"])
        cut = sorted(d["score"], reverse=True)[max(n // 4 - 1, 0)]
        p = sp.Plot((0, 26), (-8, 108), w=420, h=330, ml=56, mr=14, mt=30, mb=50,
                    xlabel="max_features", ylabel="min_samples_leaf",
                    xticks=[0, 5, 10, 15, 20, 25], yticks=[0, 20, 40, 60, 80, 100])
        sp.panel_title(p, title)
        top = [i for i in range(n) if d["score"][i] >= cut]
        rest = [i for i in range(n) if i not in top]
        p.points([d["mf"][i] for i in rest], [d["leaf"][i] for i in rest], s=1, r=4.5, hollow=True)
        p.points([d["mf"][i] for i in top], [d["leaf"][i] for i in top], s=2, r=5)
        b = int(np.argmax(d["score"]))
        p.points([d["mf"][b]], [d["leaf"][b]], s=2, r=8, hollow=True)
        p.text(d["mf"][b], d["leaf"][b], f"best {d['score'][b]:.4f}", anchor="start", cls="lbl small", dx=10, dy=-8)
        for v in sorted(set(d["mf"])):                     # 시험한 max_features 값(아래 눈금)
            p.seg(v, -8, v, -3, cls="ln s4", w=1.6)
        panels.append(p.svg(f"{title}: 시험한 조합과 교차검증 AUC 상위 4분의 1"))
    ng, nr = len(set(DATA["grid"]["mf"])), len(set(DATA["rand"]["mf"]))
    lg, lr = len(set(DATA["grid"]["leaf"])), len(set(DATA["rand"]["leaf"]))
    cap = (f"그림 2-1. 셀 9의 격자 탐색(왼쪽)과 셀 10의 무작위 탐색(오른쪽)이 시험한 조합. 색칠한 점은 교차검증 AUC가 "
           f"상위 4분의 1인 조합이고 큰 원이 최고입니다. 아래 짧은 눈금은 시험한 max_features 값으로, 격자는 {ng}가지, "
           f"무작위는 {nr}가지입니다(min_samples_leaf는 {lg}가지 대 {lr}가지). 무작위 탐색은 같은 축을 훨씬 촘촘하게 보고, "
           f"두 조절값 말고 max_samples와 criterion도 함께 바꿨습니다.")
    nb.save_fragment("ml02_fig_gridrand", sp.figure(panels, cap, cols=2))
    return dict(ng=ng, nr=nr, lg=lg, lr=lr)


def fig_tpe(param="min_samples_leaf", gamma=0.25, bw=8.0):
    sp = _svgplot()
    vals = np.array(DATA["optuna"]["values"])
    xs = np.array([p[param] for p in DATA["optuna"]["params"]], float)
    n_good = int(np.ceil(gamma * len(vals)))
    order = np.argsort(-vals)
    good = np.zeros(len(vals), bool)
    good[order[:n_good]] = True
    cut = vals[order[n_good - 1]]
    lo_y = np.floor(vals.min() * 100) / 100
    hi_y = np.ceil(vals.max() * 100) / 100 + 0.002
    pa = sp.Plot((0, 105), (lo_y, hi_y), w=420, h=320, ml=62, mr=14, mt=30, mb=50,
                 xlabel=param, ylabel="CV AUC (3-fold)", xticks=[0, 20, 40, 60, 80, 100],
                 ytickfmt=lambda v: f"{v:.2f}")
    sp.panel_title(pa, "A. 40 trials split into two groups")
    pa.hline(cut, cls="ref", dash=True)
    pa.points(xs[~good], vals[~good], s=1, r=4.5, hollow=True)
    pa.points(xs[good], vals[good], s=2, r=5)
    pa.text(2, cut, f"top {n_good} above this line", anchor="start", cls="lbl small", dy=-6)
    grid = np.linspace(0, 105, 211)

    def parzen(pts):
        z = (grid[:, None] - pts[None, :]) / bw
        return np.exp(-0.5 * z ** 2).sum(axis=1) / (len(pts) * bw * np.sqrt(2 * np.pi))
    lx, gx = parzen(xs[good]), parzen(xs[~good])
    ymax = float(max(lx.max(), gx.max())) * 1.15
    pb = sp.Plot((0, 105), (0, ymax), w=420, h=320, ml=62, mr=14, mt=30, mb=50,
                 xlabel=param, ylabel="Density", xticks=[0, 20, 40, 60, 80, 100],
                 ytickfmt=lambda v: f"{v:.3f}")
    sp.panel_title(pb, "B. Where good trials are dense")
    pb.line(grid, lx, s=2)
    pb.line(grid, gx, s=1, dash=True)
    ratio = lx / np.maximum(gx, 1e-12)
    ok = (grid >= 1) & (grid <= 100) & (lx >= 0.25 * lx.max())   # 후보는 범위 안, l(x)가 큰 곳에서 뽑힌다
    xbest = float(grid[ok][int(np.argmax(ratio[ok]))])
    pb.vline(xbest, cls="ref", dash=True)
    pb.text(xbest, ymax * 0.97, f"next try near {xbest:.0f}", anchor="end", cls="lbl small", dx=-6)
    pb.legend([("l(x): top group", 2, "line"), ("g(x): the rest", 1, "dash")], X=pb.sx(4), Y=pb.mt + 30)
    cap = (f"그림 2-2. TPE의 생각을 셀 15의 실제 시도 40번으로 단순화해 그린 것. A는 시도마다 {param}과 교차검증 AUC이고, "
           f"점수 위쪽 {n_good}번(25%)을 좋은 무리(주황), 나머지를 다른 무리(파랑)로 나눴습니다. B는 두 무리의 값 주변에 "
           f"종 모양을 하나씩 얹어 더한 밀도입니다. 좋은 무리의 밀도 l(x)가 나머지의 밀도 g(x)보다 상대적으로 가장 큰 곳"
           f"(이 그림에서는 {xbest:.0f} 근처)을 다음에 시험합니다. 실제 TPE는 조절값 네 개를 함께 다루고 후보를 여러 개 뽑아 "
           f"이 비율로 고릅니다.")
    nb.save_fragment("ml02_fig_tpe", sp.figure([pa.svg("TPE 시도와 두 무리"), pb.svg("두 무리의 밀도")], cap, cols=2))
    return dict(n_good=n_good, xbest=xbest, cut=float(cut))


FIGINFO = {}
FIGINFO["gridrand"] = fig_grid_random()
FIGINFO["tpe"] = fig_tpe()
print("FIGINFO", FIGINFO)

# ================================================================ notebook
if not SMOKE:
  nb.save_ipynb(
    "머신러닝 기초 2장. 배깅과 랜덤 포레스트",
    intro=("사회약학 연구방법 노트 '머신러닝 기초' 2장을 따라 하는 노트북입니다. 0장과 같은 가상 청구자료와 분할로 "
           "배깅과 랜덤 포레스트를 만들고, 조절값을 격자 탐색, 무작위 탐색, 단계적 탐색, Optuna, HyperOpt로 튜닝한 뒤 "
           "변수 중요도와 부분 의존 그림으로 결과를 정리합니다. 셀을 위에서부터 차례로 실행하세요. 탐색 셀은 Colab에서 "
           "몇 분씩 걸립니다. 자료는 사이트가 만든 가상 자료이며 실제 환자 자료가 아닙니다."),
    notes={
        1: ("## 준비\n\n0장의 자료 불러오기, 분할, 전처리 함수(make_pipe), 5겹 분할기(skf)와 1장의 나무용 "
            "전처리 함수(make_tree_pipe)를 한 셀에 모았습니다."),
        2: "## 가. 여러 나무를 평균 내는 이유",
        6: "## 나. 랜덤 포레스트의 조절값",
        8: "## 다. 무작위 탐색과 단계적 탐색",
        14: "## 라. 베이즈 최적화: Optuna와 HyperOpt\n\nColab에서는 설치 셀부터 실행합니다(세션마다).",
        21: "## 마. 변수 중요도와 결과 정리",
        28: ("## 과제 정답\n\n**과제 1.** class_weight=\"balanced_subsample\"로 Optuna 탐색을 다시 하고(20번), "
             "기본 모형과 시험 AUC, 평균 예측 확률, 임계값 0.10의 혼동행렬을 비교합니다."),
        29: "**과제 2.** 셀 15의 study를 40번 더 이어서 돌려 시도 10, 20, 40, 80번째까지의 최고 점수를 봅니다.",
        30: "**과제 3.** 최종 조합으로 나무 수를 10–1000으로 바꿔 시험 AUC와 걸린 시간을 봅니다.",
    })

os.chdir(_CWD)
