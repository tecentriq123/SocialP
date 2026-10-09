"""머신러닝 기초 4장 · 스태킹과 모형 비교 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/ml_ml04.py          (전체 실행, 이 환경에서 약 8-12분)
      NOMARK=1 python3 gen/ml_ml04.py   표식 없이 모든 셀의 출력을 화면에 찍는다
      ML04_CACHE=<파일> python3 gen/ml_ml04.py            실행 결과(셀 출력과 자료)를 그 파일에 저장
      REPLAY=1 ML04_CACHE=<파일> python3 gen/ml_ml04.py   셀을 다시 돌리지 않고 저장된 결과로 조각만 다시 만든다
                                                           (셀 코드가 바뀌었으면 멈춘다. 표식과 본문을 고칠 때 쓴다)
      SMOKE=1 python3 gen/ml_ml04.py    반복 횟수를 줄여 모든 셀이 오류 없이 도는지만 본다(조각·노트북은 쓰지 않음)
      ML04_WORKDIR=<폴더>               셀 19(joblib)가 파일을 쓰는 작업 폴더. 없으면 임시 폴더를 만들어 쓰고 지운다.

자료와 분할, 전처리 함수(make_pipe, make_tree_pipe, make_nan_pipe, make_hgb_pipe), 5겹 분할기(skf), boot_auc는
0-3장(gen/ml_ml00.py … ml_ml03.py)과 같다. 기본 모형 일곱 개의 조절값은 앞 장에서 고른 값을 그대로 쓴다
(로지스틱 C = 0.1: 0장 라 절, 나무: 1장 라 절, 배깅·랜덤 포레스트: 2장 가·라 절, XGBoost·히스토그램 부스팅: 3장 라·마 절).

OMP_NUM_THREADS를 1로 둔다. 이 환경은 코어가 2개이고 다른 작업과 함께 돌 때 XGBoost·히스토그램 부스팅의 OpenMP 스레드가
서로 기다리며 수십 배 느려졌다(XGBoost 기본값 5겹 34초 → 1초). 결과(AUC)는 스레드 수와 상관없이 같았다.
랜덤 포레스트와 배깅은 n_jobs=-1(joblib 스레드)이라 영향이 없다. 학생 코드(노트북)에는 이 설정이 들어가지 않는다.

걸린 시간은 실행할 때마다 달라지므로 조각에 들어가는 값은 SECONDS로 고정한다(대표 실행 한 번에서 잰 값, 본문에
"환경마다 다르다"고 적음). 끝의 대조 블록에서 본문에 적은 핵심 숫자와 앞 장의 숫자를 실행 결과와 맞춘다.
LightGBM은 이 환경에 없어서 이 장의 계산에 넣지 않는다(본문은 3장 다 절로 링크만).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # 위 설명 참고. numpy·xgboost를 부르기 전에 정해야 한다.

import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import labkit  # noqa: E402,F401
from labkit import Notebook, _mark, _dedent, Cell  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 셀 19(joblib 저장)가 현재 폴더에 파일을 쓰므로 사이트 밖의 작업 폴더에서 돌린다(그림·노트북은 labkit이 절대 경로로 쓴다).
import tempfile  # noqa: E402
_CWD = os.getcwd()
_WD = os.environ.get("ML04_WORKDIR")
if _WD:
    os.makedirs(_WD, exist_ok=True)
    os.chdir(_WD)
else:
    _TMP = tempfile.TemporaryDirectory(prefix="ml04_")
    os.chdir(_TMP.name)

NOMARK = bool(os.environ.get("NOMARK"))
REPLAY = bool(os.environ.get("REPLAY"))
SMOKE = bool(os.environ.get("SMOKE"))
CACHE = os.environ.get("ML04_CACHE")
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
    nb = ReplayNotebook("ml04", _cache["cells"])
    DATA = _cache["data"]
else:
    nb = Notebook("ml04")
    DATA = {}
ns = nb.ns

if SMOKE:
    _SMOKE_SUBS = [("n_estimators=500", "n_estimators=30"), ("n_estimators=200, random_state=2026, n_jobs=-1",
                                                               "n_estimators=20, random_state=2026, n_jobs=-1"),
                   ("n_boot=1000", "n_boot=30"), ("range(10)", "range(2)"), ("range(20)", "range(2)"),
                   ("n_repeats=5", "n_repeats=1")]
    _orig_cell = nb.cell

    def _smoke_cell(code, title="", **kw):
        for a, b in _SMOKE_SUBS:
            code = code.replace(a, b)
        return _orig_cell(code, title, **kw)
    nb.cell = _smoke_cell
    nb.save_fragment = lambda name, h: None          # 사이트의 조각을 덮어쓰지 않는다

# 조각에 넣을 걸린 시간(초). 대표 실행 한 번에서 잰 값으로 고정한다. None이면 실제 값을 그대로 쓴다.
SECONDS = {    # 2026-10-09 대표 실행(코어 2개, 다른 작업과 함께 돌던 때)에서 잰 값
    "cv_fit": {"logistic": 1, "tree": 0, "bagging": 61, "forest": 36, "xgb_default": 1, "hgb": 2, "xgb": 1},
    "t_stack": 39,         # 셀 7: 스태킹 한 번 맞추기
    "t_outer": 149,        # 셀 9: 스태킹의 바깥 5겹 교차검증
    "t_fit": 38,           # 셀 9: 로그 오즈 메타 모형 스태킹 한 번 맞추기
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
        if c.value_html:
            print("[table]", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c.value_html))[:3000])
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


def num_list(x):
    return [float(v) for v in np.ravel(np.asarray(x, float))]


# ================================================================ 준비
c = nb.cell('''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_val_predict)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder,
                                   OrdinalEncoder)
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
df = pd.read_csv(BASE + "ml_claims.csv", storage_options=UA)
y = df["admit_2023"]
X = df.drop(columns=["id", "admit_2023", "cost_2023"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=2026)
cat_cols = ["insurance", "region", "smoking", "alcohol"]
num_cols = [c for c in X.columns if c not in cat_cols]

def cat_steps():
    """글자 열: 결측을 'missing' 범주로 채운 뒤 원-핫 (0-3장과 같음)"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant",
                                 fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False))])

def make_pipe(model):
    """0장: 중앙값 대체 + 표준화 + 원-핫"""
    num_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())])
    prep = ColumnTransformer([("num", num_steps, num_cols),
                              ("cat", cat_steps(), cat_cols)])
    return Pipeline([("prep", prep), ("model", model)])

def make_tree_pipe(model):
    """1장: 중앙값 대체 + 원-핫 (표준화 없음)"""
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num_cols),
         ("cat", cat_steps(), cat_cols)],
        verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", model)])

def make_nan_pipe(model):
    """3장: 숫자 열은 결측(NaN) 그대로, 글자 열만 원-핫"""
    prep = ColumnTransformer(
        [("num", "passthrough", num_cols),
         ("cat", cat_steps(), cat_cols)],
        verbose_feature_names_out=False).set_output(transform="pandas")
    return Pipeline([("prep", prep), ("model", model)])

def make_hgb_pipe(model):
    """3장 라 절: 글자 열은 정수 번호로(결측은 NaN) 앞에, 숫자 열은 그대로"""
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=np.nan), cat_cols)],
        remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", model)])

def boot_auc(y_true, p, n_boot=1000, seed=2026):
    """1장 마 절: 시험 자료를 복원추출해 다시 구한 AUC n_boot개"""
    rng = np.random.default_rng(seed)
    yt, p, out = np.asarray(y_true), np.asarray(p), []
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        out.append(roc_auc_score(yt[i], p[i]))
    return np.array(out)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
print(X_train.shape, X_test.shape, y_train.sum(), y_test.sum())
''', title="0–3장의 준비를 한 번에 (자료, 분할, 전처리 네 가지, skf, boot_auc)")
grab("setup", lambda: [len(ns["X_train"]), len(ns["X_test"]), int(ns["y_train"].sum()), int(ns["y_test"].sum())])
save("ml04_setup", c, marks={"(11250, 39) (3750, 39) 459 153": 1})

# ================================================================ 가. 여러 모형을 합치는 방법
c = nb.cell('''
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                              HistGradientBoostingClassifier)
from xgboost import XGBClassifier

best_xgb = {"n_estimators": 200,            # 3장 마 절 Optuna가 고른 값
            "learning_rate": 0.08296006039772535,
            "max_depth": 1,
            "min_child_weight": 1.2474747561630017,
            "subsample": 0.6464876912292634,
            "colsample_bytree": 0.39626241159839404,
            "reg_lambda": 13.846707712074963,
            "reg_alpha": 0.0021251668855702768}
base = {
    "logistic": make_pipe(LogisticRegression(C=0.1, max_iter=1000)),
    "tree": make_tree_pipe(DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=300, random_state=2026)),
    "bagging": make_tree_pipe(BaggingClassifier(
        estimator=DecisionTreeClassifier(min_samples_leaf=20),
        n_estimators=200, random_state=2026, n_jobs=-1)),
    "forest": make_tree_pipe(RandomForestClassifier(
        n_estimators=500, max_features=20, min_samples_leaf=85,
        max_samples=0.7339919539656937, random_state=2026, n_jobs=-1)),
    "xgb_default": make_nan_pipe(XGBClassifier(random_state=2026)),
    "hgb": make_hgb_pipe(HistGradientBoostingClassifier(
        categorical_features=[0, 1, 2, 3], learning_rate=0.05,
        max_leaf_nodes=4, min_samples_leaf=50, max_iter=1000,
        early_stopping=True, n_iter_no_change=30, random_state=2026)),
    "xgb": make_nan_pipe(XGBClassifier(random_state=2026, **best_xgb))}

oof = pd.DataFrame(index=X_train.index)     # 겹 밖 예측 (학습 자료)
test_p = pd.DataFrame(index=X_test.index)   # 시험 자료 예측
secs = {}
for name, m in base.items():
    t0 = time.perf_counter()
    oof[name] = cross_val_predict(m, X_train, y_train, cv=skf,
                                  method="predict_proba")[:, 1]
    m.fit(X_train, y_train)                  # 학습 자료 전체로
    test_p[name] = m.predict_proba(X_test)[:, 1]
    secs[name] = time.perf_counter() - t0

folds = list(skf.split(X_train, y_train))   # 다섯 겹의 (학습, 검증) 번호

def fold_aucs(p):
    """겹 밖 예측 p로 겹마다 AUC (cross_val_score와 같은 값)"""
    p = np.asarray(p)
    return np.array([roc_auc_score(y_train.iloc[va], p[va])
                     for tr, va in folds])

fold_auc = pd.DataFrame({nm: fold_aucs(oof[nm]) for nm in base})
summary = pd.DataFrame({
    "CV AUC": fold_auc.mean(), "CV SD": fold_auc.std(ddof=0),
    "test AUC": [roc_auc_score(y_test, test_p[nm]) for nm in base],
    "seconds": pd.Series(secs).round().astype(int)})
summary.round(3)
''', title="앞 장의 모형 일곱 개: 겹 밖 예측, 전체 학습, 시험 예측")
_cvfit = grab("cv_fit_secs", lambda: {k: float(v) for k, v in ns["secs"].items()})
grab("summary", lambda: {k: num_list(ns["summary"][k]) for k in ["CV AUC", "CV SD", "test AUC"]})
grab("models", lambda: list(ns["base"]))
grab("fold_auc", lambda: {k: num_list(v) for k, v in ns["fold_auc"].items()})
_fix = SECONDS["cv_fit"] or {k: int(round(v)) for k, v in _cvfit.items()}
save("ml04_base", c, dfmarks={"0.749": 1, "0.734": 2, "0.698": 3, "0.776": 4}, rowfix=list(_fix.items()))

c = nb.cell('''
three = ["logistic", "forest", "xgb"]     # 서로 다른 집안의 세 모형
P = test_p[three]                         # 시험 자료 3,750명 × 3
votes = (P >= 0.5).sum(axis=1)            # 0.5 이상이라고 한 모형 수
toy = P.assign(votes=votes,
               hard=(votes >= 2).astype(int),          # 다수결
               soft=P.mean(axis=1),                    # 확률 평균
               weighted=(P * [1, 1, 2]).sum(axis=1) / 4,  # xgb 두 배
               admit=y_test)
print("hard vote = 1:", toy["hard"].sum(), "of", len(toy),
      "| admitted among them:", toy.loc[toy["hard"] == 1, "admit"].sum())
print("AUC  soft:", round(roc_auc_score(y_test, toy["soft"]), 3),
      "| number of votes:", round(roc_auc_score(y_test, votes), 3))
print("largest probability:", P.max().round(3).to_dict())
order = toy["soft"].sort_values(ascending=False).index
toy.loc[order[[0, 2, 5, 10, 150, 1875]]].round(3)
''', title="환자 여섯 명으로 하드 투표, 소프트 투표, 가중 평균")
grab("toy6", lambda: dict(
    pmax={k: float(v) for k, v in ns["P"].max().items()},
    rows=ns["toy"].loc[ns["order"][[0, 2, 5, 10, 150, 1875]]].astype(float).round(6).values.tolist(),
    hard_n=int(ns["toy"]["hard"].sum()), hard_admit=int(ns["toy"].loc[ns["toy"]["hard"] == 1, "admit"].sum()),
    vote_counts={int(k): int(v) for k, v in ns["votes"].value_counts().items()},
    auc_soft=float(ns["roc_auc_score"](ns["y_test"], ns["toy"]["soft"])),
    auc_votes=float(ns["roc_auc_score"](ns["y_test"], ns["votes"])),
    who=[int(i) for i in ns["order"][[0, 2, 5, 10, 150, 1875]]],
    who_x=ns["X_test"].loc[ns["order"][[0, 2, 5, 10, 150, 1875]], ["age", "n_ed", "n_drugs", "hf", "ckd"]]
    .astype(float).values.tolist()))
save("ml04_toy6", c, marks={"hard vote = 1: 14 of 3750": 1, "number of votes: 0.577": 2, "'forest': 0.383": 3})

c = nb.cell('''
from sklearn.ensemble import VotingClassifier

members = [(nm, base[nm]) for nm in three]   # (이름, 모형) 짝의 목록
vote_soft = VotingClassifier(members, voting="soft")
vote_hard = VotingClassifier(members, voting="hard")
vote_soft.fit(X_train, y_train)
vote_hard.fit(X_train, y_train)
p_soft = vote_soft.predict_proba(X_test)[:, 1]
print("soft = manual average :", np.allclose(p_soft, toy["soft"]))
print("hard = manual majority:",
      np.array_equal(vote_hard.predict(X_test), toy["hard"]))
try:
    vote_hard.predict_proba(X_test)
except AttributeError as e:
    print("AttributeError:", e)
cv_soft = fold_aucs(oof[three].mean(axis=1))  # 겹 밖 예측의 평균으로
print("soft vote: CV AUC", round(cv_soft.mean(), 4),
      "| test AUC", round(roc_auc_score(y_test, p_soft), 4))
''', title="VotingClassifier로 같은 일 하기")
grab("vote", lambda: dict(cv=num_list(ns["cv_soft"]), test=float(ns["roc_auc_score"](ns["y_test"], ns["p_soft"])),
                          out=c.stdout))
save("ml04_vote", c, marks={"AttributeError": 1, "soft vote: CV AUC 0.7677": 2})

c = nb.cell('''
import itertools

rho = oof.corr(method="spearman")      # 겹 밖 예측끼리의 순위 상관
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
im = ax[0].imshow(rho, vmin=0.4, vmax=1, cmap="viridis")
for i in range(len(rho)):
    for j in range(len(rho)):
        ax[0].text(j, i, f"{rho.iloc[i, j]:.2f}", ha="center",
                   va="center", fontsize=8,
                   color="black" if rho.iloc[i, j] > 0.75 else "white")
ax[0].set_xticks(range(len(rho)), rho.columns, rotation=45, ha="right")
ax[0].set_yticks(range(len(rho)), rho.columns)
ax[0].set_title("Spearman correlation, out-of-fold predictions")
fig.colorbar(im, ax=ax[0])

auc = summary["CV AUC"]
rows = []
for a, b in itertools.combinations(base, 2):
    both = fold_aucs(oof[[a, b]].mean(axis=1)).mean()  # 둘의 평균
    rows.append([a, b, rho.loc[a, b], abs(auc[a] - auc[b]),
                 both - max(auc[a], auc[b])])
pairs = pd.DataFrame(rows, columns=["model 1", "model 2", "rho",
                                    "AUC gap", "gain"])
close = pairs["AUC gap"] < 0.01             # 실력이 비슷한 짝
ax[1].scatter(pairs["rho"][~close], pairs["gain"][~close],
              color="lightgray", label="AUC gap >= 0.01")
ax[1].scatter(pairs["rho"][close], pairs["gain"][close],
              label="AUC gap < 0.01")
ax[1].axhline(0, color="gray", lw=0.8)
ax[1].set_xlabel("Spearman correlation of the two models")
ax[1].set_ylabel("CV AUC of their average - better one")
ax[1].legend()
plt.tight_layout()
pairs[close].sort_values("rho").round(4)
''', title="기본 모형 예측의 상관과 두 모형을 평균한 이득")
grab("corr", lambda: dict(rho={k: num_list(v) for k, v in ns["rho"].items()},
                          pairs=ns["pairs"].assign(rho=ns["pairs"]["rho"].astype(float)).values.tolist()))
save("ml04_corr", c)

# ================================================================ 나. 스태킹 만들기
c = nb.cell('''
from sklearn.ensemble import StackingClassifier

inside = pd.DataFrame({nm: m.predict_proba(X_train)[:, 1]  # 안쪽 예측
                       for nm, m in base.items()}, index=X_train.index)
print(pd.DataFrame({
    "inside": [roc_auc_score(y_train, inside[nm]) for nm in base],
    "out-of-fold": summary["CV AUC"]}, index=list(base)).T.round(3))

leaky = StackingClassifier(list(base.items()), cv="prefit",
                           final_estimator=LogisticRegression(max_iter=1000))
leaky.fit(X_train, y_train)        # 메타 모형이 안쪽 예측으로 배움
honest = LogisticRegression(max_iter=1000).fit(oof, y_train)  # 겹 밖
for name, meta, Z in [("inside", leaky, X_test),
                      ("out-of-fold", honest, test_p[list(base)])]:
    auc_t = roc_auc_score(y_test, meta.predict_proba(Z)[:, 1])
    print(f"test AUC, meta model trained on {name}: {auc_t:.3f}")
leaky3 = StackingClassifier([(nm, base[nm]) for nm in three], cv="prefit",
                            final_estimator=LogisticRegression(max_iter=1000))
leaky3.fit(X_train, y_train)       # 덜 외운 세 모형만 넣으면
honest3 = LogisticRegression(max_iter=1000).fit(oof[three], y_train)
print("three models only, inside vs out-of-fold:",
      round(roc_auc_score(y_test, leaky3.predict_proba(X_test)[:, 1]), 3),
      round(roc_auc_score(y_test, honest3.predict_proba(test_p[three])[:, 1]), 3))
coefs = pd.DataFrame({"inside": leaky.final_estimator_.coef_[0],
                      "out-of-fold": honest.coef_[0]}, index=list(base))
coefs.plot.barh(figsize=(6.5, 3.8))
plt.axvline(0, color="gray", lw=0.8)
plt.xlabel("Meta-model coefficient (probability input)")
plt.tight_layout()
coefs.round(2)
''', title="안쪽 예측으로 메타 모형을 배우면 (누설)")
grab("leak", lambda: dict(
    inside_auc={nm: float(ns["roc_auc_score"](ns["y_train"], ns["inside"][nm])) for nm in ns["base"]},
    coef_in=num_list(ns["leaky"].final_estimator_.coef_[0]), coef_oof=num_list(ns["honest"].coef_[0]),
    test_in=float(ns["roc_auc_score"](ns["y_test"], ns["leaky"].predict_proba(ns["X_test"])[:, 1])),
    test_oof=float(ns["roc_auc_score"](ns["y_test"], ns["honest"].predict_proba(ns["test_p"][list(ns["base"])])[:, 1])),
    test_in3=float(ns["roc_auc_score"](ns["y_test"], ns["leaky3"].predict_proba(ns["X_test"])[:, 1])),
    test_oof3=float(ns["roc_auc_score"](ns["y_test"], ns["honest3"].predict_proba(ns["test_p"][ns["three"]])[:, 1]))))
save("ml04_leak", c, marks={"0.998": 1, "trained on inside: 0.690": 2, "trained on out-of-fold: 0.777": 3,
                           "inside vs out-of-fold: 0.774 0.772": 4}, dfmarks={"14.70": 5})

c = nb.cell('''
stack = StackingClassifier(
    estimators=[(nm, base[nm]) for nm in three],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=skf, stack_method="predict_proba")
t0 = time.perf_counter()
stack.fit(X_train, y_train)        # 3 모형 × (5겹 + 전체 1번) = 18번
t_stack = time.perf_counter() - t0
meta = stack.final_estimator_
print("coef:", meta.coef_.round(2), " intercept:",
      meta.intercept_.round(2), f" ({t_stack:.0f} s)")
same = LogisticRegression(max_iter=1000).fit(oof[three], y_train)
print("largest difference from fitting cell 2's oof:",
      np.abs(same.coef_ - meta.coef_).max().round(6))
p_stack = stack.predict_proba(X_test)[:, 1]
print("test AUC:", round(roc_auc_score(y_test, p_stack), 3))
params = stack.get_params()        # 바깥 이름__안쪽 이름
print(params["forest__model__min_samples_leaf"],
      params["final_estimator__C"], params["cv"])
''', title="StackingClassifier (기본 모형 셋, 메타 모형은 로지스틱 회귀)")
_t = grab("t_stack", lambda: float(ns["t_stack"]))
grab("stack", lambda: dict(coef=num_list(ns["meta"].coef_), icpt=float(ns["meta"].intercept_[0]),
                           diff=float(np.abs(ns["same"].coef_ - ns["meta"].coef_).max()),
                           test=float(ns["roc_auc_score"](ns["y_test"], ns["p_stack"]))))
save("ml04_stack", c, marks={"coef: [[3.65 2.46 3.84]]": 1, "oof: 0.0": 2, "test AUC: 0.772": 3, "85 1.0": 4}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_stack', round(_t)):.0f} s)")])

c = nb.cell('''
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import FunctionTransformer

def to_logit(P):
    """확률을 로그 오즈로 (0과 1은 아주 조금 안쪽으로 당긴 뒤)"""
    P = np.clip(P, 1e-6, 1 - 1e-6)
    return np.log(P / (1 - P))

meta_pipe = Pipeline([("input", "passthrough"),
                      ("lr", LogisticRegression(max_iter=1000))])
grid = {"input": ["passthrough", FunctionTransformer(to_logit)],
        "lr__C": [0.01, 0.1, 1, 10, 100]}
gs_meta = GridSearchCV(meta_pipe, grid, cv=skf, refit="log loss",
                       scoring={"AUC": "roc_auc",
                                "log loss": "neg_log_loss"})
gs_meta.fit(oof[three], y_train)   # 메타 모형의 학습 자료 = 겹 밖 예측
best_lr = gs_meta.best_estimator_["lr"]
print("best C:", gs_meta.best_params_["lr__C"], "| coef:",
      best_lr.coef_.round(2), "| intercept:", best_lr.intercept_.round(2))

same_p = pd.DataFrame([[v] * 3 for v in [0.02, 0.05, 0.1, 0.2, 0.4]],
                      columns=three)    # 세 모형이 같은 확률을 낼 때
print(pd.DataFrame({
    "all three": same_p["xgb"],
    "probability input": meta.predict_proba(same_p.to_numpy())[:, 1],
    "log-odds input": gs_meta.predict_proba(same_p)[:, 1]}).round(3))

res = pd.DataFrame(gs_meta.cv_results_)
res["input"] = ["probability" if p == "passthrough" else "log odds"
                for p in res["param_input"]]
res["AUC"] = res["mean_test_AUC"]
res["log loss"] = -res["mean_test_log loss"]
res.pivot_table(index="param_lr__C", columns="input",
                values=["AUC", "log loss"]).round(4)
''', title="메타 모형 조절: 입력의 척도와 C를 GridSearchCV로")
grab("meta", lambda: dict(best_C=float(ns["gs_meta"].best_params_["lr__C"]), coef=num_list(ns["best_lr"].coef_),
                          icpt=float(ns["best_lr"].intercept_[0]),
                          res=ns["res"][["input", "param_lr__C", "AUC", "log loss"]].astype({"param_lr__C": float})
                          .values.tolist(),
                          map_prob=num_list(ns["meta"].predict_proba(ns["same_p"].to_numpy())[:, 1]),
                          map_logit=num_list(ns["gs_meta"].predict_proba(ns["same_p"])[:, 1]),
                          best_auc=float(ns["gs_meta"].cv_results_["mean_test_AUC"][ns["gs_meta"].best_index_])))
save("ml04_meta", c, marks={"best C: 10": 1, "0.056": 2, "0.097": 3}, dfmarks={"0.1653": 4})

c = nb.cell('''
from sklearn.base import clone

stack_lo = clone(stack).set_params(
    final_estimator=gs_meta.best_estimator_)   # 메타 모형만 바꿈
t0 = time.perf_counter()
cv_stack = cross_val_score(stack_lo, X_train, y_train, cv=skf,
                           scoring="roc_auc")  # 바깥 5겹 × 안쪽 5겹
t_outer = time.perf_counter() - t0
t0 = time.perf_counter()
stack_lo.fit(X_train, y_train)
t_fit = time.perf_counter() - t0
p_stack_lo = stack_lo.predict_proba(X_test)[:, 1]
print("fits: one stack", 3 * (5 + 1), "| outer CV", 5 * 3 * (5 + 1))
print("outer CV AUC:", cv_stack.round(4), round(cv_stack.mean(), 4),
      f"({t_outer:.0f} s)")
print("test AUC:", round(roc_auc_score(y_test, p_stack_lo), 3),
      f"(one fit {t_fit:.0f} s)")
w = pd.Series(stack_lo.final_estimator_["lr"].coef_[0], index=three)
print("meta coef (log-odds input):", w.round(2).to_dict())
print("GridSearchCV over the whole stack, 10 candidates:",
      f"about {10 * t_outer / 60:.0f} min")
w.plot.barh(figsize=(5, 2.2))
plt.axvline(0, color="gray", lw=0.8)
plt.xlabel("Meta-model coefficient (log-odds input)")
plt.tight_layout()
''', title="메타 모형을 바꾼 스태킹과 바깥 교차검증 (겹 속의 겹)")
_t2 = grab("t_outer_fit", lambda: (float(ns["t_outer"]), float(ns["t_fit"])))
grab("stack_lo", lambda: dict(cv=num_list(ns["cv_stack"]), test=float(ns["roc_auc_score"](ns["y_test"], ns["p_stack_lo"])),
                              coef=num_list(ns["w"])))
_to, _tf = secs("t_outer", round(_t2[0])), secs("t_fit", round(_t2[1]))
save("ml04_stack_lo", c, marks={"outer CV 90": 1, "0.7773": 2, "test AUC: 0.778": 3, "'forest': -0.49": 4}, subs=[(f"({_t2[0]:.0f} s)", f"({_to:.0f} s)"), (f"(one fit {_t2[1]:.0f} s)", f"(one fit {_tf:.0f} s)"),
                               (f"about {10 * _t2[0] / 60:.0f} min", f"about {10 * _to / 60:.0f} min")])

# ================================================================ 다. 모형을 공정하게 비교하기
c = nb.cell('''
rows = []
for seed in range(10):
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    a = cross_val_score(base["logistic"], X_train, y_train, cv=cv_s,
                        scoring="roc_auc").mean()
    b = cross_val_score(base["xgb"], X_train, y_train, cv=cv_s,
                        scoring="roc_auc").mean()
    rows.append([seed, a, b, b - a])
seeds = pd.DataFrame(rows, columns=["fold seed", "logistic", "xgb",
                                    "xgb - logistic"])
print("unpaired (each model on its own split):",
      round(seeds["xgb"].min() - seeds["logistic"].max(), 4), "to",
      round(seeds["xgb"].max() - seeds["logistic"].min(), 4))
seeds.drop(columns="fold seed").describe().loc[
    ["mean", "std", "min", "max"]].round(4)
''', title="겹 나누기만 바꿔 보기 (seed 10개)")
grab("seeds", lambda: ns["seeds"].astype(float).values.tolist())
save("ml04_seeds", c, marks={"0.0168 to 0.0359": 1}, dfmarks={"0.0033": 2})

c = nb.cell('''
fold_auc["soft vote"] = cv_soft
fold_auc["stack"] = cv_stack
show = ["logistic", "tree", "forest", "hgb", "soft vote", "stack", "xgb"]
fig, ax = plt.subplots(figsize=(7.5, 4))
for i in range(5):
    ax.plot(show, fold_auc.loc[i, show], marker="o",
            label=f"fold {i + 1}")
ax.set_ylabel("AUC in the validation fold")
ax.legend(fontsize=8, ncol=5)
plt.tight_layout()
diff = fold_auc[show].sub(fold_auc["xgb"], axis=0)  # 겹마다 xgb와의 차이
pd.DataFrame({"fold SD": fold_auc[show].std(ddof=0),
              "mean diff vs xgb": diff.mean(),
              "SD of diff": diff.std(ddof=0),
              "folds xgb higher": (diff < 0).sum()}).round(4)
''', title="겹마다 AUC를 이어 보기 (짝지은 비교)")
grab("paired_folds", lambda: dict(fold_auc={k: num_list(v) for k, v in ns["fold_auc"].items()},
                                  diff_sd={k: float(v) for k, v in ns["diff"].std(ddof=0).items()},
                                  diff_mean={k: float(v) for k, v in ns["diff"].mean().items()},
                                  fold_sd={k: float(v) for k, v in ns["fold_auc"][ns["show"]].std(ddof=0).items()},
                                  wins={k: int(v) for k, v in (ns["diff"] < 0).sum().items()}))
save("ml04_folds", c, dfmarks={"0.0220": 1, "0.0036": 2, "0.0013": 3})

c = nb.cell('''
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import stats

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                               random_state=2026)   # 5겹 × 5번 = 25
rep = pd.DataFrame({nm: cross_val_score(base[nm], X_train, y_train,
                                        cv=rskf, scoring="roc_auc")
                    for nm in ["logistic", "tree", "hgb", "xgb"]})

def paired_t(d, corrected=True, ratio=1 / 4):
    """겹마다의 차이 d로 짝지은 t. corrected=True면 Nadeau-Bengio 보정
    (ratio = 검증 겹 크기 ÷ 학습 겹 크기, 5겹이면 1/4)"""
    J = len(d)
    var = d.var(ddof=1) * (1 / J + (ratio if corrected else 0))
    t = d.mean() / np.sqrt(var)
    return f"t = {t:.2f}, p = {2 * stats.t.sf(abs(t), J - 1):.2g}"

for a, b in [("xgb", "logistic"), ("xgb", "hgb")]:
    d = rep[a] - rep[b]
    print(f"{a} - {b}: mean {d.mean():.4f}, SD {d.std():.4f}")
    print("   usual    :", paired_t(d, corrected=False))
    print("   corrected:", paired_t(d))
rep.plot.box(figsize=(6, 3.6))
plt.ylabel("AUC, 5-fold CV repeated 5 times")
plt.tight_layout()
rep.describe().loc[["mean", "std", "min", "max"]].round(4)
''', title="반복 교차검증과 보정한 t 검정")
grab("rep", lambda: {k: num_list(v) for k, v in ns["rep"].items()})
save("ml04_repeat", c, marks={"t = 8.81, p = 5.5e-09": 1, "t = 3.27, p = 0.0032": 2, "t = 3.44, p = 0.0022": 3,
                             "t = 1.28, p = 0.21": 4})

c = nb.cell('''
from sklearn.model_selection import cross_validate

grid_x = {"model__max_depth": [1, 2, 3],
          "model__learning_rate": [0.03, 0.1, 0.3]}
xgb200 = make_nan_pipe(XGBClassifier(n_estimators=200,
                                     random_state=2026))
usual = GridSearchCV(xgb200, grid_x, cv=skf, scoring="roc_auc")
usual.fit(X_train, y_train)
print("usual :", usual.best_params_, round(usual.best_score_, 4))

inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
nested = cross_validate(
    GridSearchCV(xgb200, grid_x, cv=inner, scoring="roc_auc"),
    X_train, y_train, cv=skf, scoring="roc_auc", return_estimator=True)
for k, (s, g) in enumerate(zip(nested["test_score"],
                               nested["estimator"])):
    print(f"outer fold {k + 1}: {g.best_params_}")
    print(f"     inner best {g.best_score_:.4f}, outer {s:.4f}")
print(f"nested: {nested['test_score'].mean():.4f}")
''', title="중첩 교차검증 (XGBoost, 작은 격자 9개)")
grab("nested", lambda: dict(usual=float(ns["usual"].best_score_), usual_params=dict(ns["usual"].best_params_),
                            outer=num_list(ns["nested"]["test_score"]),
                            inner=[float(g.best_score_) for g in ns["nested"]["estimator"]],
                            params=[dict(g.best_params_) for g in ns["nested"]["estimator"]],
                            usual_table=num_list(ns["usual"].cv_results_["mean_test_score"])))
save("ml04_nested", c, marks={"0.7717": 1, "nested: 0.7700": 2})

c = nb.cell('''
rows = []
for seed in range(20):
    m = make_nan_pipe(XGBClassifier(**{**best_xgb, "random_state": seed}))
    cv_auc = cross_val_score(m, X_train, y_train, cv=skf,
                             scoring="roc_auc").mean()
    m.fit(X_train, y_train)
    rows.append([seed, cv_auc,
                 roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])])
runs = pd.DataFrame(rows, columns=["seed", "CV AUC", "test AUC"])
top = runs.loc[runs["CV AUC"].idxmax()]
print("best of 20 by CV: seed", int(top["seed"]),
      top[["CV AUC", "test AUC"]].round(4).to_dict())
print("its test AUC ranks", int(runs["test AUC"].rank(ascending=False)
                                [top.name]), "of 20")
print("mean of 20      :",
      runs[["CV AUC", "test AUC"]].mean().round(4).to_dict())
print("seed 2026 (ch 3):", summary.loc["xgb", ["CV AUC", "test AUC"]]
      .astype(float).round(4).to_dict())
''', title="같은 조합을 seed만 바꿔 20번 (1등을 고르는 일의 낙관)")
grab("runs", lambda: ns["runs"].astype(float).values.tolist())
save("ml04_winner", c, marks={"seed 5": 1, "ranks 10 of 20": 2, "'CV AUC': 0.7733": 3,
                             "seed 2026 (ch 3): {'CV AUC': 0.7761": 4})

c = nb.cell('''
test_p["soft vote"] = p_soft
test_p["stack"] = p_stack_lo
boots = {nm: boot_auc(y_test, test_p[nm]) for nm in test_p}  # 같은 1,000벌

def delong(y, p1, p2):
    """DeLong(1988): 같은 사람들로 구한 두 AUC의 차이, 표준오차, p"""
    y = np.asarray(y)
    parts = []
    for p in (np.asarray(p1), np.asarray(p2)):
        pos, neg = p[y == 1], p[y == 0]
        win = (pos[:, None] > neg) + 0.5 * (pos[:, None] == neg)
        parts.append((win.mean(axis=1), win.mean(axis=0)))
    (v1, w1), (v2, w2) = parts     # v: 입원한 사람별, w: 안 한 사람별
    se = np.sqrt(np.var(v1 - v2, ddof=1) / len(v1)
                 + np.var(w1 - w2, ddof=1) / len(w1))
    d = v1.mean() - v2.mean()
    return d, se, 2 * stats.norm.sf(abs(d) / se)

rows = []
for a in ["stack", "soft vote", "hgb", "bagging", "forest", "logistic"]:
    d = boots[a] - boots["xgb"]          # 같은 표본에서의 차이
    diff, se, p = delong(y_test, test_p[a], test_p["xgb"])
    rows.append([a + " - xgb", diff, *np.percentile(d, [2.5, 97.5]),
                 d.std(ddof=1), se, p])
pd.DataFrame(rows, columns=["pair", "diff", "boot low", "boot high",
                            "boot SE", "DeLong SE", "DeLong p"]).round(3)
''', title="시험 자료에서 짝지은 부트스트랩과 DeLong 검정")
grab("boot_pairs", lambda: [[r[0]] + [float(v) for v in r[1:]] for r in ns["rows"]])
grab("who_stack", lambda: [float(ns["test_p"].loc[i, "stack"]) for i in DATA["toy6"]["who"]])
save("ml04_boot", c, dfmarks={"0.322": 1, "0.010": 2, "-0.066": 3})

# ================================================================ 라. 최종 모형 고르기와 보고
c = nb.cell('''
from sklearn.metrics import roc_curve

order = ["logistic", "tree", "bagging", "forest", "xgb_default", "hgb",
         "xgb", "soft vote", "stack"]
secs["soft vote"] = sum(secs[nm] for nm in three)
secs["stack"] = t_outer + t_fit
rows = []
fig = plt.figure(figsize=(5.6, 5))
for nm in order:
    p = test_p[nm]
    lo, hi = np.percentile(boots[nm], [2.5, 97.5])
    rows.append([nm, fold_auc[nm].mean(), fold_auc[nm].std(ddof=0),
                 roc_auc_score(y_test, p), lo, hi, log_loss(y_test, p),
                 brier_score_loss(y_test, p), round(secs[nm])])
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, lw=1.1, label=nm)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("1 - specificity")
plt.ylabel("Sensitivity")
plt.legend(fontsize=7)
table = pd.DataFrame(rows, columns=["model", "CV AUC", "CV SD",
                                    "test AUC", "CI low", "CI high",
                                    "log loss", "Brier", "seconds"])
table = table.set_index("model")
table.round(4)
''', title="모형 비교 표와 ROC 곡선")
grab("table", lambda: {k: num_list(ns["table"][k]) for k in ns["table"].columns if k != "seconds"})
_fix2 = dict(_fix)
_fix2["soft vote"] = sum(_fix[k] for k in ["logistic", "forest", "xgb"])
_fix2["stack"] = _to + _tf
save("ml04_table", c, dfmarks={"0.7773": 1, "0.7837": 2}, rowfix=[(k, _fix2[k]) for k in ["logistic", "tree", "bagging", "forest", "xgb_default", "hgb",
                                                   "xgb", "soft vote", "stack"]])

c = nb.cell('''
best = table["CV AUC"].idxmax()
se = fold_auc[best].std(ddof=1) / np.sqrt(5)  # 겹 5개 평균의 표준오차
cut = table.loc[best, "CV AUC"] - se
print(f"best: {best}, CV AUC {table.loc[best, 'CV AUC']:.4f},",
      f"SE {se:.4f} -> keep models with CV AUC >= {cut:.4f}")
table.assign(within_1SE=table["CV AUC"] >= cut).sort_values(
    "CV AUC", ascending=False)[["CV AUC", "CV SD", "within_1SE"]].round(4)
''', title="한 표준오차 규칙")
grab("onese", lambda: dict(best=ns["best"], se=float(ns["se"]), cut=float(ns["cut"])))
save("ml04_onese", c, marks={"keep models with CV AUC >= 0.7705": 1})

c = nb.cell('''
from sklearn.calibration import calibration_curve

test_p["stack, probability meta"] = p_stack     # 셀 7의 스태킹
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
for nm in ["logistic", "xgb", "stack", "stack, probability meta"]:
    obs, pred = calibration_curve(y_test, test_p[nm], n_bins=10,
                                  strategy="quantile")
    for a in ax:                                # 같은 점을 두 그림에
        a.plot(pred, obs, marker="o", ms=4, label=f"{nm} "
               f"(Brier {brier_score_loss(y_test, test_p[nm]):.4f})")
for a, lim in zip(ax, [0.3, 0.08]):             # 오른쪽은 아래쪽 확대
    a.plot([0, lim], [0, lim], "--", color="gray")
    a.set_xlim(0, lim)
    a.set_ylim(0, lim)
    a.set_xlabel("Mean predicted probability (10 groups)")
ax[0].set_ylabel("Observed admission rate")
ax[0].legend(fontsize=7)
ax[1].set_title("Zoom: probabilities below 0.08")
plt.tight_layout()
flat = np.full(len(y_test), y_train.mean())     # 모두에게 4.08%
print("Brier, 4.08% for everyone:",
      round(brier_score_loss(y_test, flat), 4))
''', title="교정 그림과 Brier 점수 (맛보기)")


def _calib():
    from sklearn.calibration import calibration_curve
    out = {}
    for nm in ["logistic", "xgb", "stack", "stack, probability meta"]:
        o, p = calibration_curve(ns["y_test"], ns["test_p"][nm], n_bins=10, strategy="quantile")
        out[nm] = dict(obs=num_list(o), pred=num_list(p))
    out["flat"] = float(ns["brier_score_loss"](ns["y_test"], ns["flat"]))
    out["brier_prob_stack"] = float(ns["brier_score_loss"](ns["y_test"], ns["p_stack"]))
    return out


grab("calib", _calib)
save("ml04_calib", c, marks={"0.0391": 1})

c = nb.cell('''
import joblib
import sklearn
import xgboost

final = clone(base["xgb"]).fit(X_train, y_train)  # 학습 자료 전체로 다시
p_final = final.predict_proba(X_test)[:, 1]       # 시험 자료는 한 번
lo, hi = np.percentile(boot_auc(y_test, p_final), [2.5, 97.5])
print(f"test AUC {roc_auc_score(y_test, p_final):.3f}"
      f" (95% CI {lo:.3f} to {hi:.3f})")
joblib.dump(final, "admit_model_final.joblib")
again = joblib.load("admit_model_final.joblib")
print("same after loading:",
      np.array_equal(again.predict_proba(X_test)[:, 1], p_final))
print("scikit-learn", sklearn.__version__, "| xgboost", xgboost.__version__)
''', title="최종 모형: 학습 자료 전체로 다시 맞추고 저장하기")
grab("final", lambda: dict(auc=float(ns["roc_auc_score"](ns["y_test"], ns["p_final"])), lo=float(ns["lo"]),
                           hi=float(ns["hi"])))
save("ml04_final", c, marks={"test AUC 0.784 (95% CI 0.738 to 0.825)": 1, "same after loading: True": 2})

c = nb.cell('''
from sklearn.inspection import partial_dependence

X_pdp = X_train.sample(1000, random_state=2026)
X_pdp = X_pdp.astype({c: float for c in num_cols})  # 정수 열을 실수로
show_pd = {"logistic": base["logistic"], "forest": base["forest"],
           "xgb": base["xgb"], "stack": stack_lo}
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
pdv = {}
for a, f in zip(ax, ["age", "n_ed"]):
    for nm, m in show_pd.items():
        r = partial_dependence(m, X_pdp, [f], percentiles=(0.01, 0.99),
                               grid_resolution=30)
        pdv[f, nm] = pd.Series(r["average"][0],
                               index=r["grid_values"][0])
        a.plot(r["grid_values"][0], r["average"][0], marker=".",
               label=nm)
    a.set_xlabel(f)
ax[0].set_ylabel("Average predicted probability")
ax[0].legend()
plt.tight_layout()
pd.DataFrame({nm: pdv["n_ed", nm] for nm in show_pd}).round(3)
''', title="부분 의존 그림: 개별 모형과 스태킹")
grab("pdp", lambda: {f"{f}|{nm}": dict(grid=num_list(s.index), avg=num_list(s.values)) for (f, nm), s in ns["pdv"].items()})
grab("pdp_ids", lambda: [int(v) for v in ns["df"].loc[ns["X_pdp"].index, "id"]])
save("ml04_pdp", c)

# ================================================================ 마. 과제
c = nb.cell('''
for w in [[1, 1, 1], [1, 1, 2], [1, 1, 4], [0, 0, 1]]:
    p_cv = (oof[three] * w).sum(axis=1) / sum(w)    # 겹 밖 예측의 가중 평균
    p_te = (test_p[three] * w).sum(axis=1) / sum(w)
    print(w, "CV AUC", round(fold_aucs(p_cv).mean(), 4),
          "| test AUC", round(roc_auc_score(y_test, p_te), 4))
''', title="과제 1 정답. 가중치를 바꾼 소프트 투표")
grab("hw1", lambda: c.stdout)
save("ml04_hw1", c)

c = nb.cell('''
seven = list(base)
meta7 = clone(gs_meta.best_estimator_).fit(oof[seven], y_train)
cv7 = cross_val_score(clone(gs_meta.best_estimator_), oof[seven],
                      y_train, cv=skf, scoring="roc_auc")
p7 = meta7.predict_proba(test_p[seven])[:, 1]
print(pd.Series(meta7["lr"].coef_[0], index=seven).round(2).to_dict())
print("meta-level CV AUC: 7 models", round(cv7.mean(), 4),
      "| 3 models", round(gs_meta.cv_results_["mean_test_AUC"]
                           [gs_meta.best_index_], 4))
print("test AUC: 7 models", round(roc_auc_score(y_test, p7), 4),
      "| 3 models", round(roc_auc_score(y_test, p_stack_lo), 4))
''', title="과제 2 정답. 기본 모형 일곱 개로 스태킹")
grab("hw2", lambda: dict(coef=num_list(ns["meta7"]["lr"].coef_[0]), cv7=float(ns["cv7"].mean()),
                         test7=float(ns["roc_auc_score"](ns["y_test"], ns["p7"]))))
save("ml04_hw2", c)

c = nb.cell('''
stack_b = StackingClassifier(
    [(nm, base[nm]) for nm in ["logistic", "hgb", "xgb"]],
    final_estimator=gs_meta.best_estimator_, cv=skf)
cv_b = cross_val_score(stack_b, X_train, y_train, cv=skf,
                       scoring="roc_auc")
d_cv = cv_b - fold_auc["xgb"].to_numpy()          # 같은 겹에서의 차이
print("outer CV AUC", round(cv_b.mean(), 4), "| diff vs xgb by fold",
      d_cv.round(4), "mean", round(d_cv.mean(), 4))
p_b = stack_b.fit(X_train, y_train).predict_proba(X_test)[:, 1]
d = boot_auc(y_test, p_b) - boots["xgb"]
diff, se, p = delong(y_test, p_b, test_p["xgb"])
print("test AUC", round(roc_auc_score(y_test, p_b), 4),
      "| diff", round(diff, 4), "boot 95% CI",
      np.percentile(d, [2.5, 97.5]).round(3), "| DeLong p", round(p, 3))
print(pd.Series(stack_b.final_estimator_["lr"].coef_[0],
                index=["logistic", "hgb", "xgb"]).round(2).to_dict())
''', title="과제 3 정답. 다른 세 모형으로 스태킹하고 공정하게 비교")
grab("hw3", lambda: dict(cv=num_list(ns["cv_b"]), d_cv=num_list(ns["d_cv"]),
                         test=float(ns["roc_auc_score"](ns["y_test"], ns["p_b"])), diff=float(ns["diff"]),
                         ci=num_list(np.percentile(ns["d"], [2.5, 97.5])), p=float(ns["p"]),
                         coef=num_list(ns["stack_b"].final_estimator_["lr"].coef_[0])))
save("ml04_hw3", c)

print("ml04: cells", len(nb.cells), f" elapsed {time.time() - T_START:.0f} s")
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)


# ================================================================ 생성기 안에서만: 참 구조와 견주기, 손 계산 예
def collect_truth():
    import pandas as pd
    from scipy import stats
    tr = pd.read_csv(os.path.join(ROOT, "gen", "_ml_truth.csv")).set_index("id")
    df_ = pd.read_csv(os.path.join(ROOT, "pub", "data", "ml_claims.csv"))
    from sklearn.model_selection import train_test_split
    y_ = df_["admit_2023"]
    X_ = df_.drop(columns=["id", "admit_2023", "cost_2023"])
    _, Xte, _, yte = train_test_split(X_, y_, test_size=0.25, stratify=y_, random_state=2026)
    pt = tr.loc[df_.loc[Xte.index, "id"], "p_true"].to_numpy()
    from sklearn.metrics import roc_auc_score
    out = {"true_auc": float(roc_auc_score(yte, pt))}
    if not REPLAY:
        tp = ns["test_p"]
        out["mae"] = {nm: float(np.abs(tp[nm].to_numpy() - pt).mean()) for nm in tp.columns}
        out["spearman"] = {nm: float(stats.spearmanr(tp[nm].to_numpy(), pt)[0]) for nm in tp.columns}
        out["true_brier"] = float(np.mean((pt - yte.to_numpy()) ** 2))
    # 부분 의존의 참값: 같은 1,000명에서 나이(또는 응급실 방문)만 바꾼 참 확률의 평균
    ids = DATA["pdp_ids"]
    lp = tr.loc[ids, "lp_true"].to_numpy()
    sub = df_.set_index("id").loc[ids]
    age, ins, ned = sub["age"].to_numpy(float), sub["insulin"].to_numpy(float), sub["n_ed"].to_numpy(float)

    def fa(a):
        return 0.005 * (a - 60) + 0.085 * np.clip(a - 70, 0, 22) + ins * 0.06 * np.maximum(a - 60, 0)

    def ga(k):
        return np.where(k >= 2, 1.2, np.where(k == 1, 0.1, 0.0))

    def sig(z):
        return 1 / (1 + np.exp(-z))
    g_age = DATA["pdp"]["age|xgb"]["grid"]
    g_ned = DATA["pdp"]["n_ed|xgb"]["grid"]
    out["pd_true_age"] = [float(sig(lp - fa(age) + fa(np.full_like(age, a))).mean()) for a in g_age]
    out["pd_true_ned"] = [float(sig(lp - ga(ned) + ga(np.full_like(ned, k))).mean()) for k in g_ned]
    out["pd_mae"] = {}
    for f, tv in [("age", out["pd_true_age"]), ("n_ed", out["pd_true_ned"])]:
        for nm in ["logistic", "forest", "xgb", "stack"]:
            out["pd_mae"][f"{f}|{nm}"] = float(np.mean(np.abs(np.array(tv) - np.array(DATA["pdp"][f"{f}|{nm}"]["avg"]))))
    pts = [50, 60, 70, 75, 80, 85, 90]
    out["pd_pts_age"] = {nm: [float(np.interp(a, g_age, DATA["pdp"][f"age|{nm}"]["avg"])) for a in pts]
                         for nm in ["logistic", "forest", "xgb", "stack"]}
    out["pd_pts_age"]["truth"] = [float(np.interp(a, g_age, out["pd_true_age"])) for a in pts]
    out["age_grid_end"] = [float(g_age[0]), float(g_age[-1])]
    return out


def toy_stack():
    """나 절 '숫자로 따라가기': 환자 8명, 2겹, 단순한 기본 모형 셋(나이 규칙, 응급실 규칙, 가장 닮은 환자 외우기)."""
    import itertools
    age = np.array([52, 78, 66, 83, 71, 88, 59, 80])
    ned = np.array([0, 2, 0, 0, 0, 1, 1, 2])
    yy = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    f1, f2 = np.arange(4), np.arange(4, 8)

    def m1(tr, te):
        g = age[tr] >= 75
        return np.where(age[te] >= 75, yy[tr][g].mean(), yy[tr][~g].mean())

    def m2(tr, te):
        g = ned[tr] >= 1
        return np.where(ned[te] >= 1, yy[tr][g].mean(), yy[tr][~g].mean())

    def m3(tr, te):
        return np.array([yy[tr][np.argmin(np.abs(age[tr] - a))] for a in age[te]], float)
    allx = np.arange(8)
    ins = np.column_stack([m(allx, allx) for m in (m1, m2, m3)])
    oof = np.zeros((8, 3))
    for tr, te in [(f2, f1), (f1, f2)]:
        oof[te] = np.column_stack([m(tr, te) for m in (m1, m2, m3)])
    W = [w for w in itertools.product(np.arange(0, 11) / 10, repeat=3) if abs(sum(w) - 1) < 1e-9]

    def brier(P, w):
        return float(np.mean((P @ np.array(w) - yy) ** 2))
    best_in = min(W, key=lambda w: (round(brier(ins, w), 12), w))
    best_oof = min(W, key=lambda w: (round(brier(oof, w), 12), w))
    return dict(ins=ins.tolist(), oof=oof.tolist(), n_w=len(W), best_in=[float(v) for v in best_in],
                best_oof=[float(v) for v in best_oof], brier_in_best=brier(ins, best_in),
                brier_oof_best=brier(oof, best_oof), brier_oof_equal=brier(oof, (1 / 3,) * 3),
                brier_in_equal=brier(ins, (1 / 3,) * 3),
                brier_oof_each=[brier(oof, w) for w in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
                brier_in_each=[brier(ins, w) for w in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
                brier_oof_half=brier(oof, (0.5, 0.5, 0)),
                pred_oof_best=(oof @ np.array(best_oof)).tolist())


if not SMOKE and not REPLAY:
    DATA["truth"] = collect_truth()
if not SMOKE:
    DATA["toy8"] = toy_stack()
if not REPLAY and CACHE and not SMOKE:
    cells = [dict(code=cc.code, stdout=cc.stdout, value_repr=cc.value_repr, value_html=cc.value_html,
                  images=cc.images, warns=cc.warns, error=cc.error) for cc in nb.cells]
    with open(CACHE, "wb") as f:
        pickle.dump({"cells": cells, "data": DATA}, f)
    print("cache written:", CACHE)
print(f"collected, elapsed {time.time() - T_START:.0f} s")
if not SMOKE:
    print(json.dumps({k: DATA[k] for k in DATA if k not in ("corr", "pdp", "pdp_ids", "fold_auc", "paired_folds")},
                     ensure_ascii=False, default=str)[:30000])
if MARK_ERRORS:
    print("MARK ERRORS:")
    for e in MARK_ERRORS:
        print("  ", e)


# ================================================================ 대조 블록: 본문에 적은 숫자와 실행 결과 맞추기
def _frag_text(name):
    """앞 장 조각(figs/<name>.html)의 출력 글자만 꺼낸다(대조용)."""
    import html as _h
    t = open(os.path.join(ROOT, "figs", name + ".html"), encoding="utf-8").read()
    t = t[t.find('class="cell-out"'):]
    t = re.sub(r"<img[^>]*>", " ", t)
    t = re.sub(r'<span class="mk">\d+</span>', "", t)
    return re.sub(r"\s+", " ", _h.unescape(re.sub(r"<[^>]+>", " ", t)))


def checks():
    n_ok = [0]

    def same(label, a, b, tol=1e-9):
        a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
        assert a.shape == b.shape and np.all(np.abs(a - b) <= tol), f"{label}: {a} != {b}"
        n_ok[0] += 1

    def r(x, k):
        return np.round(np.asarray(x, float), k)

    D = DATA
    M = D["models"]
    idx = {m: i for i, m in enumerate(M)}
    # ---------------- 앞 장의 숫자 재현 (본문 가 절 셀 2, 라 절 셀 16·19)
    same("split", D["setup"], [11250, 3750, 459, 153])
    S = D["summary"]
    same("CV AUC (ch 0-3)", r(S["CV AUC"], 3), [0.749, 0.754, 0.734, 0.757, 0.698, 0.768, 0.776])
    same("CV SD (ch 0-3)", r(S["CV SD"], 3), [0.022, 0.025, 0.020, 0.025, 0.023, 0.012, 0.016])
    same("test AUC (ch 0-3)", r(S["test AUC"], 3), [0.742, 0.771, 0.784, 0.770, 0.717, 0.783, 0.784])
    T = D["table"]
    same("test CIs (ch 0-3)", r(np.column_stack([T["CI low"][:7], T["CI high"][:7]]), 3),
         [[0.695, 0.785], [0.726, 0.810], [0.740, 0.825], [0.722, 0.815], [0.673, 0.761], [0.738, 0.824], [0.738, 0.825]])
    # 앞 장 조각의 출력과 직접 대조 (3장이 다시 생성되어 숫자가 바뀌면 여기서 멈춘다)
    t3 = _frag_text("ml03_compare")
    for row in ["logistic (ch 0) 0.749 0.742 0.695 0.785", "tree (ch 1) 0.754 0.771 0.726 0.810",
                "random forest (ch 2) 0.757 0.770 0.722 0.815", "HistGB tuned 0.768 0.783 0.738 0.824",
                "XGBoost default 0.698 0.717 0.673 0.761", "XGBoost tuned 0.776 0.784 0.738 0.825",
                "[0.019 0.066]", "[0.002 0.028]"]:
        assert row in t3, f"ml03_compare: {row!r} not found"
    n_ok[0] += 1
    assert "bagging 0.734 0.784 0.740 0.825" in _frag_text("ml02_cmp_final"), "ml02 bagging row"
    n_ok[0] += 1
    BP = {row[0]: row[1:] for row in D["boot_pairs"]}
    same("ch3 paired CIs (sign flipped)", [r(BP["logistic - xgb"][1:3], 3), r(BP["forest - xgb"][1:3], 3)],
         [[-0.066, -0.019], [-0.028, -0.002]])
    same("final model", r([D["final"]["auc"], D["final"]["lo"], D["final"]["hi"]], 3), [0.784, 0.738, 0.825])
    # ---------------- 가 절
    t6 = D["toy6"]
    same("toy6 hard / admitted / AUCs", [t6["hard_n"], t6["hard_admit"], round(t6["auc_votes"], 3), round(t6["auc_soft"], 3)],
         [14, 11, 0.577, 0.772])
    same("toy6 rows 1 and 4", r([t6["rows"][0][:3], t6["rows"][3][:3]], 3), [[0.891, 0.374, 0.797], [0.825, 0.290, 0.351]])
    same("toy6 row 4 soft, weighted (by hand from rounded)", [round((0.825 + 0.290 + 0.351) / 3, 3),
                                                              round((0.825 + 0.290 + 2 * 0.351) / 4, 3)], [0.489, 0.454])
    same("toy6 row 4 soft, weighted (cell)", r([t6["rows"][3][5], t6["rows"][3][6]], 3), [0.489, 0.454])
    same("toy6 votes/hard row 1 and 4", [t6["rows"][0][3], t6["rows"][0][4], t6["rows"][3][3], t6["rows"][3][4]], [2, 1, 1, 0])
    assert t6["pmax"]["forest"] < 0.4 and t6["rows"][3][7] == 1 and t6["who_x"][3] == [82.0, 0.0, 29.0, 1.0, 1.0]
    n_ok[0] += 1
    same("soft vote CV, test", [round(np.mean(D["vote"]["cv"]), 4), round(D["vote"]["test"], 4)], [0.7677, 0.7721])
    pr = {(p_[0], p_[1]): p_ for p_ in D["corr"]["pairs"]}
    same("pairs", [round(pr["logistic", "tree"][2], 2), round(pr["logistic", "tree"][4], 4), round(pr["hgb", "xgb"][2], 2),
                   round(pr["hgb", "xgb"][4], 4), round(pr["forest", "xgb"][2], 2),
                   round(S["CV AUC"][idx["tree"]] + pr["logistic", "tree"][4], 3)],
         [0.59, 0.0142, 0.91, -0.0026, 0.91, 0.769])
    gray = [p_ for p_ in D["corr"]["pairs"] if p_[3] >= 0.01]
    assert sum(p_[4] < 0 for p_ in gray) >= 0.8 * len(gray)        # "회색 점은 대부분 0 아래"
    n_ok[0] += 1
    # ---------------- 나 절
    k8 = D["toy8"]
    same("toy8 oof", k8["oof"], [[0, 0, 0], [1, 2 / 3, 1], [0, 0, 0], [1, 0, 1], [0, 0, 0], [0.5, 1, 0], [0, 1, 0], [0.5, 1, 1]])
    same("toy8 in-sample", k8["ins"], [[0, 0, 0], [.75, .75, 1], [0, 0, 0], [.75, 0, 0], [0, 0, 0], [.75, .75, 1],
                                        [0, .75, 0], [.75, .75, 1]])
    same("toy8 best weights", [k8["best_in"], k8["best_oof"]], [[0, 0, 1], [0.4, 0.6, 0]])
    yy = np.array([0, 1, 0, 0, 0, 1, 0, 1])
    b_in_46 = float(np.mean((np.array(k8["ins"]) @ np.array([0.4, 0.6, 0]) - yy) ** 2))
    same("toy8 Brier table", r([k8["brier_in_each"], k8["brier_oof_each"]], 3), [[0.094, 0.094, 0], [0.188, 0.139, 0.250]])
    same("toy8 Brier equal / best", r([k8["brier_in_equal"], k8["brier_oof_equal"], b_in_46, k8["brier_oof_best"]], 3),
         [0.026, 0.106, 0.060, 0.080])
    same("toy8 n weights", k8["n_w"], 66)
    L = D["leak"]
    same("leak inside AUCs", r([L["inside_auc"][m] for m in ["xgb_default", "bagging", "tree", "logistic"]], 3),
         [0.998, 0.953, 0.775, 0.762])
    same("leak test AUCs", r([L["test_in"], L["test_oof"], L["test_in3"], L["test_oof3"]], 3), [0.690, 0.777, 0.774, 0.772])
    same("leak coef xgb_default", r([L["coef_in"][idx["xgb_default"]], L["coef_oof"][idx["xgb_default"]]], 2), [14.70, 0.38])
    st = D["stack"]
    same("stack (probability meta)", r(st["coef"] + [st["icpt"]], 2), [3.65, 2.46, 3.84, -3.83])
    same("stack test, diff", [round(st["test"], 3), st["diff"] < 1e-6], [0.772, 1])
    me = D["meta"]
    same("meta best", [me["best_C"]] + list(r(me["coef"] + [me["icpt"]], 2)), [10, 0.20, -0.49, 1.24, -0.14])
    same("meta mapping", [r(me["map_prob"], 3), r(me["map_logit"], 3)],
         [[0.026, 0.035, 0.056, 0.138, 0.539], [0.021, 0.050, 0.097, 0.189, 0.371]])
    res = {(row[0], row[1]): row[2:] for row in me["res"]}
    same("meta grid", r([res["probability", 0.01][1], res["probability", 1.0][1], res["probability", 1.0][0]], 4),
         [0.1653, 0.1431, 0.7685])
    lo_auc = [res["log odds", c][0] for c in (0.1, 1.0, 10.0, 100.0)]
    lo_ll = [res["log odds", c][1] for c in (0.1, 1.0, 10.0, 100.0)]
    pa = [res["probability", c][0] for c in (0.01, 0.1, 1.0, 10.0, 100.0)]
    same("meta grid ranges", [round(min(lo_auc), 4), round(max(lo_auc), 4), round(min(lo_ll), 4), round(max(lo_ll), 4),
                              round(min(pa), 3), round(max(pa), 3)], [0.7743, 0.7752, 0.1398, 0.1400, 0.768, 0.769])
    sl = D["stack_lo"]
    same("stack log-odds", [round(np.mean(sl["cv"]), 4), round(sl["test"], 3), round(sl["coef"][1], 2)], [0.7773, 0.778, -0.49])
    same("fits", [3 * (5 + 1), 5 * 3 * (5 + 1)], [18, 90])
    # ---------------- 다 절
    sd = np.array(D["seeds"])
    same("fold seeds", [round(sd[:, 1].min(), 4), round(sd[:, 1].max(), 4), round(sd[:, 2].min(), 4), round(sd[:, 2].max(), 4),
                        round(sd[:, 3].min(), 4), round(sd[:, 3].max(), 4), round(sd[:, 3].std(ddof=1), 4),
                        round(sd[:, 2].min() - sd[:, 1].max(), 4), round(sd[:, 2].max() - sd[:, 1].min(), 4)],
         [0.7397, 0.7475, 0.7643, 0.7756, 0.0215, 0.0316, 0.0033, 0.0168, 0.0359])
    pf = D["paired_folds"]
    same("paired folds", [round(pf["fold_sd"]["logistic"], 4), round(pf["diff_sd"]["logistic"], 4), round(pf["fold_sd"]["soft vote"], 4),
                          round(pf["diff_sd"]["soft vote"], 4), round(pf["diff_mean"]["soft vote"], 4), pf["wins"]["soft vote"],
                          round(pf["diff_mean"]["stack"], 4), pf["wins"]["stack"]],
         [0.0220, 0.0108, 0.0191, 0.0036, -0.0083, 5, 0.0013, 2])
    from scipy import stats as _st
    rp = {k: np.array(v) for k, v in D["rep"].items()}

    def _t(d, corr):
        J = len(d)
        t = d.mean() / np.sqrt(d.var(ddof=1) * (1 / J + (0.25 if corr else 0)))
        return round(t, 2), 2 * _st.t.sf(abs(t), J - 1)
    d1, d2 = rp["xgb"] - rp["logistic"], rp["xgb"] - rp["hgb"]
    same("repeated CV t", [_t(d1, 0)[0], _t(d1, 1)[0], round(_t(d1, 1)[1], 4), _t(d2, 0)[0], round(_t(d2, 0)[1], 4), _t(d2, 1)[0],
                           round(_t(d2, 1)[1], 2), round(d1.mean(), 3), round(d1.std(ddof=1), 3), round(d2.mean(), 3)],
         [8.81, 3.27, 0.0032, 3.44, 0.0022, 1.28, 0.21, 0.027, 0.015, 0.006])
    same("NB factor", [round(np.sqrt((1 / 25 + 0.25) / (1 / 25)), 2), 2250 / 9000], [2.69, 0.25])
    ne = D["nested"]
    same("nested", [round(ne["usual"], 4), round(np.mean(ne["outer"]), 4), round(ne["usual"] - np.mean(ne["outer"]), 4),
                    round(min(ne["outer"]), 3), round(max(ne["outer"]), 3), round(min(ne["inner"]), 3), round(max(ne["inner"]), 3)],
         [0.7717, 0.7700, 0.0017, 0.745, 0.792, 0.763, 0.775])
    assert ne["usual_params"] == {"model__learning_rate": 0.03, "model__max_depth": 2}
    assert all(p_["model__max_depth"] in (1, 2) for p_ in ne["params"])
    n_ok[0] += 1
    ru = np.array(D["runs"])
    b_ = int(np.argmax(ru[:, 1]))
    rank = int((ru[:, 2] > ru[b_, 2]).sum() + 1)
    same("winner", [round(ru[:, 1].min(), 3), round(ru[:, 1].max(), 3), ru[b_, 0], round(ru[b_, 1], 4), rank,
                    round(ru[:, 1].mean(), 4), round(ru[:, 2].mean(), 4), round(ru[:, 1].mean() - S["CV AUC"][idx["xgb"]], 3)],
         [0.770, 0.776, 5, 0.7763, 10, 0.7733, 0.7857, -0.003])
    same("boot pairs", [r(BP[k][:3], 3) for k in ["stack - xgb", "soft vote - xgb", "forest - xgb", "logistic - xgb"]],
         [[-0.006, -0.019, 0.005], [-0.012, -0.021, -0.003], [-0.015, -0.028, -0.002], [-0.042, -0.066, -0.019]])
    same("boot pairs hgb, bagging", [r(BP[k][:3], 3) for k in ["hgb - xgb", "bagging - xgb"]],
         [[-0.002, -0.020, 0.016], [-0.001, -0.020, 0.021]])
    same("DeLong p", r([BP[k][5] for k in ["stack - xgb", "soft vote - xgb", "forest - xgb", "logistic - xgb"]], 3),
         [0.322, 0.010, 0.020, 0.001])
    ses = np.array([[v[3], v[4]] for v in BP.values()])
    same("SE agree, SE range", [np.abs(ses[:, 0] - ses[:, 1]).max() < 0.001, round(ses.min(), 3), round(ses.max(), 3),
                                round(1.96 * ses.min(), 2), round(1.96 * ses.max(), 3)], [1, 0.005, 0.013, 0.01, 0.025])
    same("Bonferroni", [round(0.05 / 6, 4), sum(v[5] < 0.05 / 6 for v in BP.values())], [0.0083, 1])
    # ---------------- 라 절
    O = ["logistic", "tree", "bagging", "forest", "xgb_default", "hgb", "xgb", "soft vote", "stack"]
    tb = {k: dict(zip(O, v)) for k, v in T.items()}
    same("table CV", r([tb["CV AUC"][m] for m in ["stack", "xgb", "hgb", "soft vote", "logistic"]], 4),
         [0.7773, 0.7761, 0.7679, 0.7677, 0.7486])
    same("table test", r([tb["test AUC"][m] for m in ["xgb", "bagging", "hgb"]], 4), [0.7843, 0.7837, 0.7828])
    same("table loss", r([tb["log loss"][m] for m in ["hgb", "stack", "xgb", "xgb_default"]], 4), [0.1384, 0.1391, 0.1387, 0.1696])
    same("table Brier", r([tb["Brier"][m] for m in ["hgb", "stack", "xgb", "xgb_default", "logistic"]], 4),
         [0.0331, 0.0332, 0.0334, 0.0360, 0.0340])
    same("soft vote CI", r([tb["CI low"]["soft vote"], tb["CI high"]["soft vote"]], 3), [0.725, 0.814])
    o1 = D["onese"]
    assert o1["best"] == "stack"
    same("one SE", [round(o1["se"], 4), round(o1["cut"], 4), round(o1["cut"] - tb["CV AUC"]["logistic"], 3),
                    round((tb["CV AUC"]["stack"] - tb["CV AUC"]["xgb"]) / o1["se"], 2)], [0.0068, 0.7705, 0.022, 0.19])
    assert sum(tb["CV AUC"][m] >= o1["cut"] for m in O) == 2
    n_ok[0] += 1
    ca = D["calib"]
    pm = ca["stack, probability meta"]
    same("calibration", [round(ca["flat"], 4), round(min(pm["pred"][:9]), 3), round(max(pm["pred"][:9]), 3),
                         round(min(pm["obs"][:9]), 3), round(max(pm["obs"][:9]), 3), round(ca["brier_prob_stack"], 4)],
         [0.0391, 0.024, 0.036, 0.008, 0.059, 0.0335])
    briers = [tb["Brier"]["logistic"], tb["Brier"]["xgb"], tb["Brier"]["stack"], ca["brier_prob_stack"]]
    same("Brier range", [round(min(briers), 4), round(max(briers), 4)], [0.0332, 0.0340])
    cnt = [round(v * 375) for nm in ["logistic", "xgb", "stack", "stack, probability meta"] for v in ca[nm]["obs"]]
    top = [round(ca[nm]["obs"][-1] * 375) for nm in ["logistic", "xgb", "stack", "stack, probability meta"]]
    same("calibration counts", [min(cnt), max(c for c in cnt if c not in top), min(top), max(top)], [2, 27, 64, 75])
    tr = D["truth"]
    same("truth", [round(tr["true_auc"], 3)] + list(r([tr["mae"][m] for m in ["xgb", "stack", "hgb", "soft vote",
                                                                          "stack, probability meta", "forest", "logistic"]], 4)),
         [0.810, 0.0144, 0.0147, 0.0153, 0.0163, 0.0182, 0.0196, 0.0200])
    pt = tr["pd_pts_age"]
    same("PD age table", r([pt[k][1:] for k in ["truth", "logistic", "forest", "xgb", "stack"]], 3),
         [[0.027, 0.030, 0.045, 0.068, 0.100, 0.144], [0.035, 0.046, 0.052, 0.059, 0.068, 0.077],
          [0.032, 0.035, 0.042, 0.079, 0.114, 0.182], [0.029, 0.029, 0.034, 0.071, 0.090, 0.125],
          [0.030, 0.030, 0.034, 0.065, 0.076, 0.094]])
    same("PD n_ed grid", D["pdp"]["n_ed|xgb"]["grid"], [0, 1, 2, 3, 5])
    same("PD n_ed table", r([tr["pd_true_ned"][:4]] + [D["pdp"][f"n_ed|{m}"]["avg"][:4] for m in ["logistic", "forest", "xgb", "stack"]], 3),
         [[0.038, 0.041, 0.100, 0.100], [0.039, 0.048, 0.060, 0.074], [0.041, 0.042, 0.044, 0.044],
          [0.040, 0.040, 0.067, 0.067], [0.039, 0.040, 0.078, 0.081]])
    pm_ = tr["pd_mae"]
    same("PD MAE", r([pm_[f"age|{m}"] for m in ["logistic", "forest", "xgb", "stack"]] +
                     [pm_[f"n_ed|{m}"] for m in ["logistic", "forest", "xgb", "stack"]], 3),
         [0.013, 0.009, 0.005, 0.009, 0.017, 0.035, 0.020, 0.011])
    same("PD age grid ends", r(tr["age_grid_end"], 1), [40.0, 90.6])
    # ---------------- 과제
    h1 = [float(v) for v in re.findall(r"CV AUC ([\d.]+)", D["hw1"])]
    h1t = [float(v) for v in re.findall(r"test AUC ([\d.]+)", D["hw1"])]
    same("hw1", h1, [0.7677, 0.7706, 0.7731, 0.7761])
    assert h1t == sorted(h1t)
    n_ok[0] += 1
    h2 = D["hw2"]
    same("hw2", [round(h2["coef"][idx["xgb"]], 2), round(h2["coef"][idx["forest"]], 2), round(h2["coef"][idx["bagging"]], 2),
                 round(h2["coef"][idx["xgb_default"]], 2), round(h2["cv7"], 4), round(D["meta"]["best_auc"], 4),
                 round(h2["test7"], 4), round(D["stack_lo"]["test"], 4)],
         [1.06, -0.39, -0.16, 0.01, 0.7729, 0.7743, 0.7744, 0.7779])
    h3 = D["hw3"]
    same("hw3", [round(np.mean(h3["cv"]), 4), round(np.mean(h3["d_cv"]), 4), int(np.sum(np.array(h3["d_cv"]) < 0)),
                 round(h3["test"], 4), round(h3["diff"], 4)] + list(r(h3["ci"], 3)) + [round(h3["p"], 2)] + list(r(h3["coef"], 2)),
         [0.7727, -0.0033, 4, 0.7825, -0.0018, -0.008, 0.004, 0.53, 0.19, 0.20, 0.65])
    # ---------------- 그림 4-1의 숫자
    same("figure 4-1 final probability = stack prediction", round(FIGINFO["stack"]["pf"], 6), round(D["who_stack"][3], 6),
         tol=2e-6)
    same("figure 4-1 numbers", [round(FIGINFO["stack"]["pf"], 3)] + list(r(FIGINFO["stack"]["p3"], 3)), [0.459, 0.825, 0.290, 0.351])
    print(f"VERIFY: {n_ok[0]} checks passed")
    return n_ok[0]




# ================================================================ 그림 4-1 (SVG): 스태킹의 두 단계
def _svg(W, H, body, aria):
    return (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="{aria}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(body)}</svg>')


def _t(o, x, y, s, anchor="start", cls="lbl"):
    from html import escape
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{escape(s)}</text>')


def _arrow(o, x1, y1, x2, y2):
    import math
    o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="ref" stroke-width="1.3"/>')
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + 2.6, ang - 2.6
    o.append(f'<polygon points="{x2:.1f},{y2:.1f} {x2 + 7 * math.cos(a1):.1f},{y2 + 7 * math.sin(a1):.1f} '
             f'{x2 + 7 * math.cos(a2):.1f},{y2 + 7 * math.sin(a2):.1f}" class="f4"/>')


def _box(o, x, y, w, h, label, cls="a4", lcls="lbl small", label2=None):
    o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" class="{cls}" stroke-width="1"/>')
    if label2 is None:
        _t(o, x + w / 2, y + h / 2 + 4, label, "middle", lcls)
    else:
        _t(o, x + w / 2, y + h / 2 - 3, label, "middle", lcls)
        _t(o, x + w / 2, y + h / 2 + 11, label2, "middle", "lbl mute small")


def fig_stacking():
    sys.path.insert(0, ROOT)
    from svgplot import figure
    W, H = 420, 340
    o = []
    _t(o, 12, 20, "A. Training data for the meta model", "start", "ptitle")
    cols = [("logistic", 84), ("forest", 84), ("xgb", 84), ("admit", 52)]
    x0, y0, rh = 74, 52, 30
    xs = []
    x = x0
    for name, w in cols:
        xs.append((x, w))
        _t(o, x + w / 2, y0 - 8, name, "middle", "lbl strong small")
        x += w + 6
    for i in range(5):
        y = y0 + i * (rh + 6)
        _box(o, 12, y, 54, rh, f"fold {i + 1}", "a4")
        for j, (x, w) in enumerate(xs):
            if j < 3:
                _box(o, x, y, w, rh, f"model w/o {i + 1}", "a1 s1")
            else:
                _box(o, x, y, w, rh, "0 or 1", "a2 s2")
    yb = y0 + 5 * (rh + 6)
    _arrow(o, 210, yb, 210, yb + 26)
    _box(o, 60, yb + 30, 300, 34, "Meta model: one weight per column", "a1 s1", "lbl small strong")
    _t(o, 210, yb + 84, "Each row's 3 probabilities come from models", "middle", "lbl mute small")
    _t(o, 210, yb + 100, "that never saw that patient (out-of-fold).", "middle", "lbl mute small")
    a = _svg(W, H, o, "겹 밖 예측으로 메타 모형의 학습 자료를 만드는 과정")
    # B: 새 환자 한 명 (셀 3의 넷째 줄 환자)
    r = DATA["toy6"]["rows"][3]
    p3 = r[:3]
    m = DATA["meta"]
    z = m["icpt"] + sum(c * np.log(p / (1 - p)) for c, p in zip(m["coef"], p3))
    pf = 1 / (1 + np.exp(-z))
    o = []
    _t(o, 12, 20, "B. Predicting a new patient", "start", "ptitle")
    _box(o, 110, 38, 200, 28, "New patient (39 features)", "a4")
    bx = [(14, "logistic"), (150, "forest"), (286, "xgb")]
    for x, name in bx:
        _arrow(o, 210, 66, x + 60, 96)
        _box(o, x, 98, 120, 40, name, "a1 s1", "lbl small strong", label2="refit on all 11,250")
    for (x, name), p in zip(bx, p3):
        _arrow(o, x + 60, 138, x + 60, 160)
        _t(o, x + 60, 176, f"p = {p:.3f}", "middle", "lbl small")
    for x, _ in bx:
        _arrow(o, x + 60, 184, 210, 214)
    _box(o, 60, 216, 300, 40, "Meta model (log-odds input)", "a1 s1", "lbl small strong",
         label2=f"{m['coef'][0]:.2f}, {m['coef'][1]:.2f}, {m['coef'][2]:.2f}; intercept {m['icpt']:.2f}")
    _arrow(o, 210, 256, 210, 280)
    _box(o, 120, 282, 180, 28, f"Final probability {pf:.3f}", "a2 s2", "lbl small strong")
    _t(o, 210, 330, "Same base models as in A, but fitted once on all data.", "middle", "lbl mute small")
    b = _svg(W, H, o, "새 환자의 예측: 기본 모형 셋의 확률을 메타 모형이 합친다")
    cap = ("그림 4-1. 스태킹의 두 단계. A는 메타 모형의 학습 자료를 만드는 과정입니다. 학습 자료를 다섯 겹으로 나누고, "
           "겹마다 그 겹을 뺀 나머지 네 겹으로 기본 모형 셋을 맞춰 그 겹 사람들의 확률을 냅니다(파란 칸의 'w/o 1'은 "
           "1겹을 빼고 배운 모형이라는 뜻). 이렇게 모은 11,250명 × 3열의 겹 밖 예측과 실제 결과(주황 칸)로 메타 모형이 "
           "열마다 무게를 배웁니다. B는 새 환자 한 명의 예측입니다. 기본 모형 셋은 학습 자료 전체로 한 번 더 맞춘 것을 쓰고, "
           f"셋의 확률 {p3[0]:.3f}, {p3[1]:.3f}, {p3[2]:.3f}을 로그 오즈로 바꿔 메타 모형의 무게를 곱해 더하고 "
           f"절편을 더한 뒤 확률로 되돌리면 {pf:.3f}가 됩니다. 이 환자는 셀 3 표의 넷째 줄(82세, 약 29가지, 심부전과 만성콩팥병이 함께 있음)이고 실제로 "
           "입원했습니다.")
    nb.save_fragment("ml04_fig_stack", figure([a, b], cap, cols=2))
    return dict(p3=p3, pf=float(pf))


FIGINFO = {}
if not SMOKE:
    FIGINFO["stack"] = fig_stacking()
    print("FIGINFO", FIGINFO)

N_CHECKS = None if SMOKE else checks()

# ================================================================ notebook
if not SMOKE:
    nb.save_ipynb(
        "머신러닝 기초 4장. 스태킹과 모형 비교",
        intro=("사회약학 연구방법 노트 '머신러닝 기초' 4장을 따라 하는 노트북입니다. 0–3장과 같은 가상 청구자료와 분할로 "
               "앞 장의 모형 일곱 개를 다시 만들고, 투표·평균·스태킹으로 합친 뒤, 같은 겹과 짝지은 비교, 반복 교차검증, "
               "중첩 교차검증, 짝지은 부트스트랩으로 모형을 공정하게 비교하고 최종 모형을 골라 저장합니다. 셀을 위에서부터 "
               "차례로 실행하세요. 셀 2와 셀 9는 Colab에서 몇 분씩 걸립니다. 자료는 사이트가 만든 가상 자료이며 실제 환자 "
               "자료가 아닙니다."),
        notes={
            1: ("## 준비\n\n0장의 자료 불러오기와 분할, 0–3장의 전처리 함수 네 가지, 5겹 분할기(skf), 1장의 `boot_auc`를 "
                "한 셀에 모았습니다."),
            2: "## 가. 여러 모형을 합치는 방법",
            6: "## 나. 스태킹 만들기",
            10: "## 다. 모형을 공정하게 비교하기",
            16: "## 라. 최종 모형 고르기와 보고",
            21: ("## 과제 정답\n\n**과제 1.** 셀 3의 세 모형에 가중치 [1, 1, 1], [1, 1, 2], [1, 1, 4], [0, 0, 1]을 주어 "
                 "겹 밖 예측으로 교차검증 AUC를, 시험 예측으로 시험 AUC를 구합니다."),
            22: "**과제 2.** 기본 모형 일곱 개의 겹 밖 예측으로 셀 8에서 고른 메타 모형을 맞추고 계수와 AUC를 봅니다.",
            23: ("**과제 3.** 로지스틱 회귀, 히스토그램 부스팅, XGBoost로 스태킹을 만들어 바깥 교차검증과 시험 자료에서 "
                 "XGBoost 하나와 짝지어 비교합니다."),
        })

os.chdir(_CWD)
