"""머신러닝 기초 3장 · 부스팅 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/ml_ml03.py          (전체 실행, 이 환경에서 약 10분)
      NOMARK=1 python3 gen/ml_ml03.py   표식 없이 모든 셀의 출력을 화면에 찍는다
      ML03_CACHE=<파일> python3 gen/ml_ml03.py            실행 결과(셀 출력과 자료)를 그 파일에 저장
      REPLAY=1 ML03_CACHE=<파일> python3 gen/ml_ml03.py   셀을 다시 돌리지 않고 저장된 결과로 조각만 다시 만든다
      SMOKE=1 python3 gen/ml_ml03.py    반복 횟수를 줄여 모든 셀이 오류 없이 도는지만 본다(조각·노트북은 버림)
      COLAB_NB=<실행된 ml03_colab.ipynb> ... Colab 셀의 자리 표시 조각을 실행된 노트북의 출력으로 바꾼다

자료와 분할, 전처리 함수는 0–2장(gen/ml_ml00.py, ml_ml01.py, ml_ml02.py)과 같다. 이 장에서 새로 쓰는
make_nan_pipe는 숫자 열의 결측을 대체하지 않고 NaN 그대로 넘긴다(글자 열은 0–2장처럼 'missing' 범주 + 원-핫).

이 환경에는 LightGBM, CatBoost, shap이 없다. 그런 셀(COLAB_CELLS)은 실행하지 않고 코드만 노트북에 넣으며,
본문 조각(figs/ml03_lgb_*.html 등)은 "Colab 실행 결과를 넣을 자리"로 만든다. 같은 코드를 pub/colab/ml03_colab.ipynb에
모으고, 셀 이름 ↔ 본문 셀 번호 ↔ Colab 노트북 셀 위치를 gen/_ml03_colab_map.json에 적는다. 주인이 Colab에서 돌린
노트북을 받으면 COLAB_NB=<그 파일>로 이 스크립트를 다시 돌려(REPLAY와 함께 써도 됨) 자리 표시를 실제 출력으로 바꾼다.

걸린 시간은 실행할 때마다 달라지므로 조각에 들어가는 값은 SECONDS로 고정한다(대표 실행 한 번에서 잰 값).
끝의 대조 블록에서 본문에 적은 핵심 숫자를 실행 결과와 맞춘다.
"""
import hashlib
import json
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import labkit  # noqa: E402,F401
from labkit import Notebook, _mark, _dedent, Cell  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 셀(저장 시연)이 현재 폴더에 파일을 쓰므로 임시 폴더에서 돌린다(그림·노트북은 labkit이 절대 경로로 쓴다).
import tempfile  # noqa: E402
_TMP = tempfile.TemporaryDirectory(prefix="ml03_")
_CWD = os.getcwd()
os.chdir(_TMP.name)

NOMARK = bool(os.environ.get("NOMARK"))
REPLAY = bool(os.environ.get("REPLAY"))
SMOKE = bool(os.environ.get("SMOKE"))
CACHE = os.environ.get("ML03_CACHE")
COLAB_NB = os.environ.get("COLAB_NB")
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
    nb = ReplayNotebook("ml03", _cache["cells"])
    DATA = _cache["data"]
else:
    nb = Notebook("ml03")
    DATA = {}
ns = nb.ns

if SMOKE:
    _SMOKE_SUBS = [("n_trials=60", "n_trials=4"), ("n_trials=30", "n_trials=3"), ("max_evals=30", "max_evals=3"),
                   ("n_estimators=500, learning_rate=0.1", "n_estimators=60, learning_rate=0.1"),
                   ("range(1, 501)", "range(1, 61)"), ("[1, 50, best, 500]", "[1, 50, best, 60]"),
                   ("n_repeats=10", "n_repeats=2"), ("n_boot=1000", "n_boot=50"),
                   ("RandomForestClassifier(n_estimators=500", "RandomForestClassifier(n_estimators=20")]
    _orig_cell = nb.cell

    def _smoke_cell(code, title="", **kw):
        for a, b in _SMOKE_SUBS:
            code = code.replace(a, b)
        return _orig_cell(code, title, **kw)
    nb.cell = _smoke_cell

# 조각에 넣을 걸린 시간(초). 대표 실행 한 번에서 잰 값으로 고정한다. None이면 실제 값을 그대로 쓴다.
SECONDS = {"t_gbm": 8, "t_optuna": 104, "t_hyperopt": 68, "t_steps": 23}   # 2026-10-09 대표 실행(코어 2개)
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


def save(name, c, marks=None, dfmarks=None, subs=None):
    """subs: [(실제 문자열, 고정 문자열)] — 걸린 시간처럼 실행마다 달라지는 출력을 고정한다."""
    if NOMARK:
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        if c.value_html:
            print("[table]", re.sub(r"<[^>]+>", " ", c.value_html)[:3000])
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
    nb.save_fragment(name, h)


# ================================================================ Colab에서만 도는 셀 (LightGBM, CatBoost, shap)
PLACEHOLDER = ("Colab 실행 결과를 넣을 자리\n"
               "(이 셀은 {pkg}이(가) 필요해 이 쪽을 만든 환경에서는 실행하지 않았습니다.\n"
               " Colab에서는 그대로 실행됩니다.)")
COLAB_CELLS = []          # [dict(name, cell, section, pkg)]


def _colab_outputs(path):
    """실행된 Colab 노트북에서 '셀 이름 → 출력 목록'을 읽는다(마크다운의 <!-- ml03_xxx --> 표시로 찾음)."""
    with open(path, encoding="utf-8") as f:
        nbj = json.load(f)
    out, pending = {}, None
    for cc in nbj["cells"]:
        src = "".join(cc.get("source", []))
        if cc["cell_type"] == "markdown":
            m = re.search(r"<!-- (ml03_[a-z0-9_]+) -->", src)
            pending = m.group(1) if m else None
        elif pending:
            out[pending] = (src, cc.get("outputs", []))
            pending = None
    return out


_COLAB_OUT = _colab_outputs(COLAB_NB) if COLAB_NB else {}


def _fill_from_colab(c, outputs):
    """Jupyter 출력(stream, execute_result, display_data)을 labkit Cell의 칸으로 옮긴다."""
    txt, warns, imgs = [], [], []
    for o in outputs:
        t = o.get("output_type")
        if t == "stream":
            s = "".join(o.get("text", []))
            if o.get("name") == "stderr":
                # '/usr/.../sklearn.py:123: FutureWarning: 내용' -> 'FutureWarning: 내용' (labkit과 같은 모양)
                for ln in s.splitlines():
                    m = re.search(r"\b(\w*Warning): (.*)", ln)
                    if m and f"{m.group(1)}: {m.group(2)}" not in warns:
                        warns.append(f"{m.group(1)}: {m.group(2)}")
            else:
                txt.append(s)
        elif t in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                img = d["image/png"]
                imgs.append(("".join(img) if isinstance(img, list) else img).replace("\n", ""))
            elif "text/html" in d and "<table" in "".join(d["text/html"]):
                # Colab의 DataFrame 출력에는 표 말고도 단추, <svg>, <style>, <script>가 붙어 있으므로 표만 꺼낸다
                m = re.search(r"<table.*?</table>", "".join(d["text/html"]), flags=re.S)
                h = m.group(0).replace('border="1" ', 'border="0" ')
                c.value_html = h.replace('class="dataframe"', 'class="dataframe dfout"')
            elif "text/plain" in d and t == "execute_result":
                c.value_repr = "".join(d["text/plain"])
        elif t == "error":
            c.error = f"{o.get('ename')}: {o.get('evalue')}"
    c.stdout, c.warns, c.images = "".join(txt), warns, imgs


def colab_cell(name, code, title, section, pkg, marks=None, dfmarks=None):
    """실행하지 않는 셀: 노트북에는 코드로 넣고, 본문 조각은 자리 표시(또는 실행된 Colab 노트북의 출력)로."""
    code = _dedent(code)
    c = Cell(len(nb.cells) + 1, code, title)
    nb.cells.append(c)
    COLAB_CELLS.append(dict(name=name, cell=c, section=section, pkg=pkg))
    if name in _COLAB_OUT and not _COLAB_OUT[name][1]:      # '모두 실행'이 앞에서 멈춰 출력이 없는 셀
        MARK_ERRORS.append(f"{name}: no output in the Colab notebook (cell not run?); placeholder kept")
    if name in _COLAB_OUT and _COLAB_OUT[name][1]:
        src, outs = _COLAB_OUT[name]
        if src.strip() != code.strip():
            MARK_ERRORS.append(f"{name}: Colab notebook code differs from the site cell")
        _fill_from_colab(c, outs)
        save(name, c, marks=marks, dfmarks=dfmarks)
    else:
        c.stdout = PLACEHOLDER.format(pkg=pkg)
        nb.save_fragment(name, nb.html(c))
    return c


# ================================================================ 준비
PIP_OUT = """Collecting optuna
  Downloading optuna-4.5.0-py3-none-any.whl (...)
Collecting catboost
  Downloading catboost-...-manylinux2014_x86_64.whl (...)
...
Successfully installed ... catboost-... optuna-4.5.0 ..."""
c = nb.cell('''
!pip install optuna hyperopt lightgbm catboost shap
''', title="Optuna, HyperOpt, LightGBM, CatBoost, shap 설치 (Colab은 세션마다)", shell_output=PIP_OUT)
save("ml03_pip", c)

c = nb.cell('''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, cross_validate)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss

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
    """글자 열: 결측을 'missing' 범주로 채운 뒤 원-핫 (0-2장과 같음)"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant",
                                 fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False))])

def make_pipe(model, num=num_cols, cat=cat_cols):
    """0장: 중앙값 대체 + 표준화 + 원-핫"""
    num_steps = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())])
    prep = ColumnTransformer([("num", num_steps, num),
                              ("cat", cat_steps(), cat)])
    return Pipeline([("prep", prep), ("model", model)])

def make_tree_pipe(model, num=num_cols, cat=cat_cols):
    """1장: 중앙값 대체 + 원-핫 (표준화 없음)"""
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", cat_steps(), cat)],
        verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", model)])

def make_nan_pipe(model, num=num_cols, cat=cat_cols):
    """3장: 숫자 열은 결측(NaN) 그대로, 글자 열만 원-핫"""
    prep = ColumnTransformer(
        [("num", "passthrough", num), ("cat", cat_steps(), cat)],
        verbose_feature_names_out=False).set_output(transform="pandas")
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
''', title="0–2장의 준비를 한 번에 (자료, 분할, 파이프라인 세 가지, skf)")
save("ml03_setup", c, marks={"(11250, 39) (3750, 39) 459 153": 1})

# ================================================================ 가. 앞 나무의 오차를 다음 나무가 배운다
c = nb.cell('''
from sklearn.tree import DecisionTreeRegressor

toy = pd.DataFrame({
    "age":   [52, 61, 47, 78, 83, 69, 88, 74],
    "n_ed":  [0, 0, 1, 2, 0, 2, 1, 1],
    "admit": [0, 0, 0, 1, 0, 1, 1, 0]}, index=range(1, 9))
lr = 0.5                                   # 학습률 (보기 쉽게 크게)
p = np.full(8, toy["admit"].mean())        # ① 모두 같은 출발 확률
for r in [1, 2]:
    resid = toy["admit"] - p               # ② 남은 오차
    stump = DecisionTreeRegressor(max_depth=1, random_state=2026)
    stump.fit(toy[["age", "n_ed"]], resid) # ③ 작은 나무가 오차를 배움
    step = stump.predict(toy[["age", "n_ed"]])
    p = p + lr * step                      # ④ 학습률만큼만 고침
    f = ["age", "n_ed"][stump.tree_.feature[0]]
    print(f"round {r}: {f} <= {stump.tree_.threshold[0]}")
    toy[f"resid{r}"], toy[f"tree{r}"] = resid, step
    toy[f"p{r}"] = p
toy.round(3)
''', title="환자 8명으로 그레이디언트 부스팅 두 번 돌리기")
grab("toy", lambda: dict(table=ns["toy"].round(4).to_dict(orient="list"), out=c.stdout))
save("ml03_toy", c, marks={"round 1: n_ed <= 1.5": 1, "round 2: age <= 85.5": 2},
     dfmarks={"0.688": 3, "0.729": 4, "0.635": 5})

c = nb.cell('''
from sklearn.ensemble import GradientBoostingClassifier

X_tr, X_val, y_tr, y_val = train_test_split(   # 학습 자료를 다시 나눔
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=2026)
prep = make_tree_pipe(None).named_steps["prep"]
A_tr = prep.fit_transform(X_tr)                 # 9000 × 48
A_val = prep.transform(X_val)                   # 2250 × 48

gbm = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1,
                                 max_depth=3, random_state=2026)
gbm.fit(A_tr, y_tr)
loss_tr = [log_loss(y_tr, p[:, 1])
           for p in gbm.staged_predict_proba(A_tr)]
loss_val = [log_loss(y_val, p[:, 1])
            for p in gbm.staged_predict_proba(A_val)]
best = int(np.argmin(loss_val)) + 1             # 검증 손실이 가장 작은 나무 수
for k in [1, 50, best, 500]:
    print(f"{k:3d} trees: train {loss_tr[k - 1]:.4f},"
          f" validation {loss_val[k - 1]:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(range(1, 501), loss_tr, label="training (9,000)")
ax.plot(range(1, 501), loss_val, label="validation (2,250)")
ax.axvline(best, ls="--", color="gray")
ax.set_xlabel("Number of trees")
ax.set_ylabel("Log loss")
ax.legend()
plt.tight_layout()
''', title="나무 수에 따른 학습·검증 손실 (staged_predict_proba)")
grab("gbm", lambda: dict(best=int(ns["best"]), tr=[float(v) for v in ns["loss_tr"]],
                         val=[float(v) for v in ns["loss_val"]]))
save("ml03_gbm_curve", c, marks={" 69 trees: train 0.1203, validation 0.1396": 1,
                                "500 trees: train 0.0653, validation 0.1630": 2})

c = nb.cell('''
t0 = time.perf_counter()
cv_gbm = cross_val_score(
    make_tree_pipe(GradientBoostingClassifier(random_state=2026)),
    X_train, y_train, cv=skf, scoring="roc_auc")
t_gbm = time.perf_counter() - t0
print(f"CV AUC {cv_gbm.mean():.4f} (SD {cv_gbm.std():.4f}),"
      f" {t_gbm:.0f} s")
''', title="사이킷런 그레이디언트 부스팅의 기본값으로 5겹 교차검증")
_t = grab("t_gbm", lambda: ns["t_gbm"])
grab("gbm_cv", lambda: [float(ns["cv_gbm"].mean()), float(ns["cv_gbm"].std())])
save("ml03_gbm_cv", c, marks={"CV AUC 0.7658 (SD 0.0154)": 1}, subs=[(f" {_t:.0f} s", f" {secs('t_gbm', round(_t)):.0f} s")])

# ================================================================ 나. XGBoost
c = nb.cell('''
import xgboost as xgb
from xgboost import XGBClassifier
print("xgboost", xgb.__version__)

shallow = dict(max_depth=2, learning_rate=0.05, n_estimators=300)
rows = []
for how, maker in [("median imputed", make_tree_pipe),
                   ("NaN as is", make_nan_pipe)]:
    for name, params in [("default", {}), ("shallow", shallow)]:
        model = maker(XGBClassifier(random_state=2026, **params))
        auc = cross_val_score(model, X_train, y_train, cv=skf,
                              scoring="roc_auc")
        rows.append([how, name, auc.mean(), auc.std()])
pd.DataFrame(rows, columns=["missing", "setting", "CV AUC", "SD"]
             ).round(4)
''', title="XGBoost를 파이프라인 안에서: 결측 대체와 NaN 그대로")
grab("xgb_basic", lambda: [[r[0], r[1], float(r[2]), float(r[3])] for r in ns["rows"]] + [ns["xgb"].__version__])
save("ml03_xgb_basic", c, dfmarks={"0.6904": 1, "0.7697": 2})

c = nb.cell('''
import json

prep_nan = make_nan_pipe(None).named_steps["prep"]
Z_train = prep_nan.fit_transform(X_train)   # DataFrame, NaN 그대로
Z_test = prep_nan.transform(X_test)
small = XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1,
                      random_state=2026).fit(Z_train, y_train)

cfg = json.loads(small.get_booster().save_config())   # 실제로 쓴 설정
tp = cfg["learner"]["gradient_booster"]["tree_train_param"]
print({k: tp[k] for k in ["eta", "max_depth", "min_child_weight",
                          "lambda", "alpha", "gamma", "max_bin"]})
trees = small.get_booster().trees_to_dataframe()
cols = ["Tree", "ID", "Feature", "Split", "Yes", "No", "Missing", "Gain"]
print(trees.loc[trees["Tree"] == 0, cols].to_string(index=False))
bmi = trees[trees["Feature"] == "bmi"]
print("bmi splits:", len(bmi), " NaN sent to the 'Yes' side:",
      (bmi["Missing"] == bmi["Yes"]).sum())
''', title="XGBoost 안을 들여다보기: 실제 기본값과 결측의 방향")
grab("inside", lambda: dict(
    n_bmi=int(len(ns["bmi"])), n_yes=int((ns["bmi"]["Missing"] == ns["bmi"]["Yes"]).sum()),
    tree0=ns["trees"][ns["trees"]["Tree"] == 0][["ID", "Feature", "Split", "Yes", "No", "Missing", "Gain"]].astype(str).values.tolist(),
    cfg={k: ns["tp"][k] for k in ["eta", "max_depth", "min_child_weight", "lambda", "alpha", "gamma", "max_bin"]}))
save("ml03_xgb_inside", c, marks={"'min_child_weight': '1'": 1, "age   84.0": 2, "0.962585": 3,
                                 "NaN sent to the 'Yes' side: 24": 4})

c = nb.cell('''
from sklearn.model_selection import validation_curve

depths = [1, 2, 3, 4, 5, 6, 8]
tr, va = validation_curve(
    make_nan_pipe(XGBClassifier(learning_rate=0.05, n_estimators=300,
                                random_state=2026)),
    X_train, y_train, param_name="model__max_depth",
    param_range=depths, cv=skf, scoring="roc_auc")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for s, lab in [(tr, "training folds"), (va, "cross-validation")]:
    m, sd = s.mean(axis=1), s.std(axis=1)
    ax.plot(depths, m, "o-", label=lab)
    ax.fill_between(depths, m - sd, m + sd, alpha=0.2)
ax.set_xlabel("max_depth (learning_rate 0.05, 300 trees)")
ax.set_ylabel("AUC")
ax.legend()
plt.tight_layout()
pd.DataFrame({"max_depth": depths, "train AUC": tr.mean(axis=1),
              "CV AUC": va.mean(axis=1), "CV SD": va.std(axis=1)}).round(4)
''', title="max_depth의 검증 곡선")
grab("depth", lambda: dict(tr=ns["tr"].mean(axis=1).tolist(), va=ns["va"].mean(axis=1).tolist(),
                           sd=ns["va"].std(axis=1).tolist()))
save("ml03_xgb_depth", c, dfmarks={"0.7712": 1, "0.7124": 2})

# ================================================================ 다. LightGBM (Colab)
colab_cell("ml03_lgb_default", '''
import lightgbm as lgb
from lightgbm import LGBMClassifier
print("lightgbm", lgb.__version__)

few = dict(num_leaves=4, learning_rate=0.05, n_estimators=300,
           min_child_samples=50)
for name, params in [("default", {}), ("4 leaves", few)]:
    model = make_nan_pipe(LGBMClassifier(random_state=2026, verbose=-1,
                                         **params))
    t0 = time.perf_counter()
    auc = cross_val_score(model, X_train, y_train, cv=skf,
                          scoring="roc_auc")
    print(f"{name:9s} CV AUC {auc.mean():.4f} (SD {auc.std():.4f}),"
          f" {time.perf_counter() - t0:.1f} s")
''', "LightGBM 기본값과 잎을 줄인 설정", "ml03-s3", "LightGBM")

colab_cell("ml03_lgb_leaves", '''
leaves = [2, 4, 8, 16, 31, 64]
rows = []
for nl in leaves:
    model = make_nan_pipe(LGBMClassifier(
        num_leaves=nl, learning_rate=0.05, n_estimators=300,
        random_state=2026, verbose=-1))
    r = cross_validate(model, X_train, y_train, cv=skf,
                       scoring="roc_auc", return_train_score=True)
    rows.append([nl, r["train_score"].mean(), r["test_score"].mean(),
                 r["test_score"].std()])
res_leaves = pd.DataFrame(rows, columns=["num_leaves", "train AUC",
                                         "CV AUC", "CV SD"])
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(leaves, res_leaves["train AUC"], "o-", label="training folds")
ax.plot(leaves, res_leaves["CV AUC"], "o-", label="cross-validation")
ax.set_xscale("log", base=2)
ax.set_xlabel("num_leaves (learning_rate 0.05, 300 trees)")
ax.set_ylabel("AUC")
ax.legend()
plt.tight_layout()
res_leaves.round(4)
''', "num_leaves의 검증 곡선", "ml03-s3", "LightGBM")

colab_cell("ml03_lgb_cat", '''
Xl_train = X_train.astype({c: "category" for c in cat_cols})
print([str(t) for t in Xl_train[cat_cols].dtypes])
for name, model, data in [
        ("one-hot (make_nan_pipe)",
         make_nan_pipe(LGBMClassifier(random_state=2026, verbose=-1,
                                      **few)), X_train),
        ("category dtype, no pipeline",
         LGBMClassifier(random_state=2026, verbose=-1, **few), Xl_train)]:
    auc = cross_val_score(model, data, y_train, cv=skf,
                          scoring="roc_auc")
    print(f"{name:28s} CV AUC {auc.mean():.4f}")
''', "범주형 열을 그대로 넘기기 (pandas category)", "ml03-s3", "LightGBM")

# ================================================================ 라. 히스토그램 부스팅과 CatBoost
c = nb.cell('''
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

def make_hgb_pipe(model):
    """글자 열은 정수 번호로(결측은 NaN 그대로) 앞에, 숫자 열은 그대로"""
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=np.nan), cat_cols)],
        remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([("prep", prep), ("model", model)])

settings = {
    "default": {},
    "early_stopping=True": dict(early_stopping=True),
    "tuned": dict(learning_rate=0.05, max_leaf_nodes=4,
                  min_samples_leaf=50, max_iter=1000,
                  early_stopping=True, n_iter_no_change=30)}
rows = []
for name, params in settings.items():
    hgb = make_hgb_pipe(HistGradientBoostingClassifier(
        categorical_features=[0, 1, 2, 3], random_state=2026, **params))
    auc = cross_val_score(hgb, X_train, y_train, cv=skf,
                          scoring="roc_auc")
    hgb.fit(X_train, y_train)                 # 11,250명 전체로
    rows.append([name, auc.mean(), auc.std(), hgb[-1].n_iter_])
pd.DataFrame(rows, columns=["setting", "CV AUC", "SD",
                            "trees (all 11,250)"]).round(4)
''', title="히스토그램 부스팅 (HistGradientBoostingClassifier)")
grab("hgb", lambda: [[r[0], float(r[1]), float(r[2]), int(r[3])] for r in ns["rows"]])
save("ml03_hgb", c, dfmarks={"0.7243": 1, "0.7469": 2, "0.7679": 3, "256": 4})

colab_cell("ml03_cat", '''
import catboost
from catboost import CatBoostClassifier
print("catboost", catboost.__version__)

Xk_train, Xk_test = X_train.copy(), X_test.copy()
Xk_train[cat_cols] = Xk_train[cat_cols].fillna("missing")  # 글자 열의 NaN은
Xk_test[cat_cols] = Xk_test[cat_cols].fillna("missing")    # 글자로 바꿔야 함

def cv_catboost(**params):
    """CatBoost의 5겹 교차검증 AUC (겹은 skf와 같음)"""
    aucs = []
    for tr, va in skf.split(Xk_train, y_train):
        m = CatBoostClassifier(cat_features=cat_cols, random_seed=2026,
                               verbose=0, **params)
        m.fit(Xk_train.iloc[tr], y_train.iloc[tr])
        p = m.predict_proba(Xk_train.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y_train.iloc[va], p))
    return np.mean(aucs), np.std(aucs), m

for name, params in [("default", {}),
                     ("ordered", {"boosting_type": "Ordered"})]:
    t0 = time.perf_counter()
    mean, sd, m = cv_catboost(**params)
    if name == "default":
        cv_cat_default = mean
    used = m.get_all_params()
    print(f"{name:8s} CV AUC {mean:.4f} (SD {sd:.4f}),"
          f" {time.perf_counter() - t0:.0f} s")
    print("   ", {k: used[k] for k in ["boosting_type", "iterations",
                                       "learning_rate", "depth",
                                       "l2_leaf_reg"]})
''', "CatBoost: 범주형 열을 그대로, 기본값으로", "ml03-s4", "CatBoost")

# ================================================================ 마. 조기 종료와 튜닝
c = nb.cell('''
Z_tr, Z_val = Z_train.loc[X_tr.index], Z_train.loc[X_val.index]
es_model = XGBClassifier(n_estimators=2000, learning_rate=0.05,
                         max_depth=2, early_stopping_rounds=50,
                         eval_metric="logloss", random_state=2026)
es_model.fit(Z_tr, y_tr, eval_set=[(Z_tr, y_tr), (Z_val, y_val)],
             verbose=False)
hist = es_model.evals_result()
print(list(hist), " trees grown:", len(hist["validation_1"]["logloss"]))
print("best_iteration:", es_model.best_iteration,
      " best validation log loss:", round(es_model.best_score, 4))
p_val = es_model.predict_proba(Z_val)[:, 1]   # 가장 좋았던 지점까지의 나무로
print("validation AUC:", round(roc_auc_score(y_val, p_val), 4))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(hist["validation_0"]["logloss"], label="training (9,000)")
ax.plot(hist["validation_1"]["logloss"], label="validation (2,250)")
ax.axvline(es_model.best_iteration, ls="--", color="gray")
ax.set_xlabel("Boosting round (tree number - 1)")
ax.set_ylabel("Log loss")
ax.legend()
plt.tight_layout()
''', title="XGBoost의 조기 종료 (eval_set, early_stopping_rounds, evals_result)")
grab("es", lambda: dict(best=int(ns["es_model"].best_iteration), score=float(ns["es_model"].best_score),
                        grown=len(ns["hist"]["validation_1"]["logloss"]),
                        auc=float(ns["roc_auc_score"](ns["y_val"], ns["p_val"])),
                        val=[float(v) for v in ns["hist"]["validation_1"]["logloss"]],
                        trn=[float(v) for v in ns["hist"]["validation_0"]["logloss"]]))
save("ml03_es", c, marks={"trees grown: 210": 1, "best_iteration: 159": 2, "validation AUC: 0.7958": 3})

colab_cell("ml03_lgb_es", '''
lgb_es = LGBMClassifier(n_estimators=2000, learning_rate=0.05,
                        num_leaves=4, random_state=2026, verbose=-1)
lgb_es.fit(Z_tr, y_tr, eval_set=[(Z_tr, y_tr), (Z_val, y_val)],
           eval_metric="binary_logloss",
           callbacks=[lgb.early_stopping(50, verbose=False),
                      lgb.log_evaluation(0)])
print("best_iteration_:", lgb_es.best_iteration_)
print({k: round(v["binary_logloss"], 4)
       for k, v in lgb_es.best_score_.items()})
p_lval = lgb_es.predict_proba(Z_val)[:, 1]
print("validation AUC:", round(roc_auc_score(y_val, p_lval), 4))
ax = lgb.plot_metric(lgb_es, metric="binary_logloss", figsize=(6.5, 3.4))
''', "LightGBM의 조기 종료 (callbacks)", "ml03-s5", "LightGBM")

c = nb.cell('''
def fit_es(Z_a, y_a, Z_b, y_b, metric):
    """Z_b로 멈출 곳을 정하는 XGBoost (깊이 2, 학습률 0.05)"""
    m = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=2,
                      early_stopping_rounds=50, eval_metric=metric,
                      random_state=2026)
    return m.fit(Z_a, y_a, eval_set=[(Z_b, y_b)], verbose=False)

for metric in ["logloss", "auc"]:
    leak, clean = [], []
    for tr, va in skf.split(Z_train, y_train):
        Zb, yb = Z_train.iloc[va], y_train.iloc[va]   # 검증 겹
        Za, Zs, ya, ys = train_test_split(            # 학습 겹을 8 대 2로
            Z_train.iloc[tr], y_train.iloc[tr], test_size=0.2,
            stratify=y_train.iloc[tr], random_state=2026)
        m = fit_es(Za, ya, Zb, yb, metric)            # 검증 겹으로 멈춤
        leak.append(roc_auc_score(yb, m.predict_proba(Zb)[:, 1]))
        m = fit_es(Za, ya, Zs, ys, metric)            # 떼어 둔 20%로 멈춤
        clean.append(roc_auc_score(yb, m.predict_proba(Zb)[:, 1]))
    d = np.array(leak) - np.array(clean)
    print(f"stop on {metric:7s}: leaky {np.mean(leak):.4f},"
          f" clean {np.mean(clean):.4f}, leaky higher in"
          f" {(d > 0).sum()} of 5 folds")
''', title="교차검증 안의 조기 종료: 검증 겹을 두 번 쓰면")
grab("es_leak", lambda: c.stdout)
save("ml03_es_leak", c, marks={"leaky 0.7707, clean 0.7698, leaky higher in 4 of 5 folds": 1,
                               "leaky 0.7770, clean 0.7719, leaky higher in 5 of 5 folds": 2})

c = nb.cell('''
from sklearn.model_selection import GridSearchCV

params = {"learning_rate": 0.1, "n_estimators": 200}   # 학습률을 크게
steps = [{"max_depth": [1, 2, 3, 4],                  # 1 나무 구조
          "min_child_weight": [1, 5, 20]},
         {"subsample": [0.6, 0.8, 1.0],               # 2 표본·특성 비율
          "colsample_bytree": [0.4, 0.7, 1.0]},
         {"reg_lambda": [0.1, 1, 10, 50]}]            # 3 규제
t0 = time.perf_counter()
for k, grid in enumerate(steps, start=1):
    gs = GridSearchCV(
        make_nan_pipe(XGBClassifier(random_state=2026, **params)),
        {"model__" + name: v for name, v in grid.items()},
        cv=skf, scoring="roc_auc")
    gs.fit(X_train, y_train)
    params.update({name.replace("model__", ""): v
                   for name, v in gs.best_params_.items()})
    print(f"step {k}: CV AUC {gs.best_score_:.4f}", gs.best_params_)
params.update(learning_rate=0.02, n_estimators=1000)  # 4 학습률 ÷ 5, 나무 × 5
cv4 = cross_val_score(make_nan_pipe(XGBClassifier(random_state=2026,
                                                  **params)),
                      X_train, y_train, cv=skf, scoring="roc_auc")
t_steps = time.perf_counter() - t0
print(f"step 4: CV AUC {cv4.mean():.4f}  ({t_steps:.0f} s)")
step_params = params
''', title="튜닝 순서의 실전 요령을 격자 탐색으로")
_t = grab("t_steps", lambda: ns["t_steps"])
grab("steps", lambda: dict(out=c.stdout, params={k: v for k, v in ns["step_params"].items()}, cv4=float(ns["cv4"].mean())))
save("ml03_steps", c, marks={"step 1: CV AUC 0.7713": 1, "step 2: CV AUC 0.7728": 2, "step 4: CV AUC 0.7713": 3},
     subs=[(f"({_t:.0f} s)", f"({secs('t_steps', round(_t)):.0f} s)")])

c = nb.cell('''
import optuna
from optuna.samplers import TPESampler

def objective(trial):
    """조합 하나를 골라 5겹 교차검증 AUC를 돌려준다 (겹은 늘 skf)"""
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000,
                                       step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                          log=True),
        max_depth=trial.suggest_int("max_depth", 1, 6),
        min_child_weight=trial.suggest_float("min_child_weight", 0.5,
                                             50, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3,
                                             1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.1, 100,
                                       log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.001, 10,
                                      log=True))
    model = make_nan_pipe(XGBClassifier(random_state=2026, **params))
    return cross_val_score(model, X_train, y_train, cv=skf,
                           scoring="roc_auc").mean()

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=TPESampler(seed=2026))
t0 = time.perf_counter()
study.optimize(objective, n_trials=60)
t_optuna = time.perf_counter() - t0
best_xgb = study.best_params
print({k: round(v, 4) for k, v in best_xgb.items()})
print(round(study.best_value, 4), " trial", study.best_trial.number,
      f" ({t_optuna:.0f} s)")
''', title="Optuna로 XGBoost 튜닝 (60번 시도)")
_t = grab("t_optuna", lambda: ns["t_optuna"])
grab("optuna", lambda: dict(values=[float(t.value) for t in ns["study"].trials],
                            params=[{k: float(v) for k, v in t.params.items()} for t in ns["study"].trials],
                            best=dict(ns["study"].best_params), best_value=float(ns["study"].best_value),
                            best_number=int(ns["study"].best_trial.number)))
save("ml03_optuna", c, marks={"'max_depth': 1": 1, "0.7761  trial 53": 2}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_optuna', round(_t)):.0f} s)")])

c = nb.cell('''
from optuna.visualization.matplotlib import (
    plot_optimization_history, plot_param_importances, plot_slice)
from optuna.importance import FanovaImportanceEvaluator

ax1 = plot_optimization_history(study)
ax2 = plot_param_importances(
    study, evaluator=FanovaImportanceEvaluator(seed=2026))
ax3 = plot_slice(study, params=["learning_rate", "n_estimators",
                                "max_depth"])
tdf = study.trials_dataframe(attrs=("number", "value", "params"))
tdf.columns = [c.replace("params_", "") for c in tdf.columns]
tdf.sort_values("value", ascending=False).head(5).round(3)
''', title="Optuna 결과를 그림과 표로")
def _imp():
    from optuna.importance import get_param_importances, FanovaImportanceEvaluator
    return {k: float(v) for k, v in get_param_importances(ns["study"], evaluator=FanovaImportanceEvaluator(seed=2026)).items()}
grab("optuna_imp", _imp)
save("ml03_optuna_plots", c)

c = nb.cell('''
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

hp_space = {
    "max_depth": hp.quniform("max_depth", 1, 6, 1),
    "n_estimators": hp.quniform("n_estimators", 100, 1000, 50),
    "learning_rate": hp.loguniform("learning_rate",
                                   np.log(0.01), np.log(0.3)),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1.0)}

def hp_objective(params):
    model = make_nan_pipe(XGBClassifier(
        random_state=2026, max_depth=int(params["max_depth"]),
        n_estimators=int(params["n_estimators"]),
        learning_rate=params["learning_rate"],
        colsample_bytree=params["colsample_bytree"]))
    auc = cross_val_score(model, X_train, y_train, cv=skf,
                          scoring="roc_auc").mean()
    return {"loss": -auc, "status": STATUS_OK}

trials = Trials()
t0 = time.perf_counter()
best_hp = fmin(hp_objective, hp_space, algo=tpe.suggest, max_evals=30,
               trials=trials, rstate=np.random.default_rng(2026),
               show_progressbar=False)
t_hyperopt = time.perf_counter() - t0
print({k: round(float(v), 4) for k, v in best_hp.items()})
print(round(-min(trials.losses()), 4), f" ({t_hyperopt:.0f} s)")
''', title="HyperOpt로 같은 일 (교재의 방식, 30번)")
_t = grab("t_hyperopt", lambda: ns["t_hyperopt"])
grab("hyperopt", lambda: dict(best={k: float(v) for k, v in ns["best_hp"].items()}, auc=float(-min(ns["trials"].losses())),
                              losses=[float(v) for v in ns["trials"].losses()]))
save("ml03_hyperopt", c, marks={"'max_depth': 2.0": 1, "0.7735": 2}, subs=[(f" ({_t:.0f} s)", f" ({secs('t_hyperopt', round(_t)):.0f} s)")])

c = nb.cell('''
cands = {"default": {},
         "stepwise grid": step_params,
         "Optuna": best_xgb}
rows, fitted = [], {}
for name, params in cands.items():
    model = make_nan_pipe(XGBClassifier(random_state=2026, **params))
    cv = cross_val_score(model, X_train, y_train, cv=skf,
                         scoring="roc_auc")
    fitted[name] = model.fit(X_train, y_train)    # 학습 자료 전체로
    p = fitted[name].predict_proba(X_test)[:, 1]
    lo, hi = np.percentile(boot_auc(y_test, p), [2.5, 97.5])
    rows.append([name, cv.mean(), cv.std(), roc_auc_score(y_test, p),
                 lo, hi])
xgb_final = fitted["Optuna"]                       # 교차검증으로 고른 모형
p_xgb = xgb_final.predict_proba(X_test)[:, 1]
pd.DataFrame(rows, columns=["XGBoost", "CV AUC", "CV SD", "test AUC",
                            "CI low", "CI high"]).round(3)
''', title="기본값, 단계별 격자, Optuna의 교차검증과 시험 AUC")
grab("tuned", lambda: [[r[0]] + [float(v) for v in r[1:]] for r in ns["rows"]])
save("ml03_tuned", c, dfmarks={"0.717": 1, "0.792": 2, "0.776": 3, "0.784": 4})

colab_cell("ml03_lgb_optuna", '''
def lgb_objective(trial):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000,
                                       step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                          log=True),
        num_leaves=trial.suggest_int("num_leaves", 2, 64, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 200,
                                            log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3,
                                             1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 100,
                                       log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.001, 10,
                                      log=True))
    model = make_nan_pipe(LGBMClassifier(
        subsample_freq=1, random_state=2026, verbose=-1, **params))
    return cross_val_score(model, X_train, y_train, cv=skf,
                           scoring="roc_auc").mean()

study_lgb = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=2026))
t0 = time.perf_counter()
study_lgb.optimize(lgb_objective, n_trials=40)
print({k: round(v, 4) for k, v in study_lgb.best_params.items()})
print(round(study_lgb.best_value, 4), " trial",
      study_lgb.best_trial.number,
      f" ({time.perf_counter() - t0:.0f} s)")
''', "Optuna로 LightGBM 튜닝 (40번 시도)", "ml03-s5", "LightGBM")

# ================================================================ 바. 변수 기여도와 결과 정리
c = nb.cell('''
from sklearn.inspection import permutation_importance

booster = xgb_final[-1].get_booster()
imp = pd.DataFrame({t: pd.Series(booster.get_score(importance_type=t))
                    for t in ["weight", "gain", "cover", "total_gain"]})
imp = imp.reindex(Z_train.columns).fillna(0)  # 쓰이지 않은 열은 0
perm = permutation_importance(xgb_final, X_test, y_test,
                              scoring="roc_auc", n_repeats=10,
                              random_state=2026)
perm = pd.Series(perm.importances_mean, index=X_test.columns)
print("features used:", (imp["weight"] > 0).sum(), "of", len(imp))
show = ["age", "n_drugs", "bmi", "cost_2022", "insulin", "n_ed", "ckd",
        "hf", "dyslip", "oa", "n_prescribers"]
out = imp.loc[show].round(1)
out["rank_weight"] = imp["weight"].rank(ascending=False,
                                        method="min").loc[show]
out["rank_gain"] = imp["gain"].rank(ascending=False,
                                    method="min").loc[show]
out["perm"] = perm.loc[show].round(4)
out["rank_perm"] = perm.rank(ascending=False, method="min").loc[show]
out.astype({c: int for c in ["rank_weight", "rank_gain", "rank_perm"]})
''', title="XGBoost의 중요도 네 가지와 순열 중요도", max_rows=12)
grab("imp", lambda: dict(imp={k: ns["imp"][k].astype(float).to_dict() for k in ns["imp"].columns},
                         perm=ns["perm"].astype(float).to_dict()))
save("ml03_imp", c, dfmarks={"52.0": 1, "59.4": 2, "266.3": 3, "0.0528": 4})

c = nb.cell('''
dtest = xgb.DMatrix(Z_test)
contrib = booster.predict(dtest, pred_contribs=True)  # (3750, 48 + 1)
margin = booster.predict(dtest, output_margin=True)   # 로그 오즈
print(contrib.shape, " largest |sum - log odds|:",
      np.abs(contrib.sum(axis=1) - margin).max().round(6))
shap_df = pd.DataFrame(contrib[:, :-1], columns=Z_test.columns,
                       index=Z_test.index).astype(float)
print("base value:", contrib[0, -1].round(3))

mean_abs = shap_df.abs().mean().sort_values(ascending=False)
top = mean_abs.index[:12][::-1]                    # 위 12개, 아래부터
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4),
                       gridspec_kw={"width_ratios": [1, 1.4]})
ax[0].barh(top, mean_abs[top])
ax[0].set_xlabel("Mean |SHAP| (log odds)")
rng = np.random.default_rng(2026)
for k, f in enumerate(top):
    v = Z_test[f]
    col = v.rank(pct=True)                         # 값의 상대적 크기 0-1
    ok = v.notna().to_numpy()
    yy = k + rng.uniform(-0.3, 0.3, len(v))
    ax[1].scatter(shap_df[f][~ok], yy[~ok], s=4, color="lightgray")
    sc = ax[1].scatter(shap_df[f][ok], yy[ok], s=4, c=col[ok],
                       cmap="coolwarm", vmin=0, vmax=1)
ax[1].set_yticks(range(len(top)), top)
ax[1].axvline(0, color="gray", lw=0.8)
ax[1].set_xlabel("SHAP value (log odds); gray = missing")
fig.colorbar(sc, ax=ax[1], label="Feature value (low to high)")
plt.tight_layout()
''', title="SHAP 값을 XGBoost 자체 기능으로 (pred_contribs)")
grab("shap", lambda: dict(base=float(ns["contrib"][0, -1]),
                          addit=float(np.abs(ns["contrib"].sum(axis=1) - ns["margin"]).max()),
                          mean_abs=ns["shap_df"].abs().mean().astype(float).to_dict()))
save("ml03_shap", c, marks={"largest |sum - log odds|: 3e-06": 1, "base value: -3.19": 2})

c = nb.cell('''
who = int(np.argsort(-p_xgb)[75])            # 예측 위험이 76번째로 높은 사람
row = shap_df.iloc[who]
base = contrib[who, -1]
big = row[row.abs() >= 0.05].sort_values(key=abs, ascending=False)
print(X_test.iloc[who][["age", "n_ed", "n_drugs", "bmi", "insulin",
                        "hf", "ckd"]].to_dict())
print("base value (log odds):", round(base, 3),
      " = probability", round(1 / (1 + np.exp(-base)), 3))
print(big.round(3).to_string())
print("all other features   :", round(row.sum() - big.sum(), 3))
total = base + row.sum()
print("total (log odds):", round(total, 3), " = probability",
      round(1 / (1 + np.exp(-total)), 3), " | predict_proba:",
      round(p_xgb[who], 3))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
vals = list(big.values) + [row.sum() - big.sum()]
labs = [f"{f} = {Z_test.iloc[who][f]}" for f in big.index] + ["others"]
ax.barh(labs[::-1], vals[::-1],
        color=["tab:red" if v > 0 else "tab:blue" for v in vals[::-1]])
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("SHAP value (log odds)")
plt.tight_layout()
''', title="환자 한 명의 예측을 변수별 기여로 쪼개기")
grab("one", lambda: dict(who=int(ns["who"]), p=float(ns["p_xgb"][ns["who"]]), base=float(ns["base"]),
                         row=ns["row"].astype(float).to_dict(), x=ns["X_test"].iloc[ns["who"]].astype(str).to_dict(),
                         y=int(ns["y_test"].iloc[ns["who"]]), out=c.stdout))
save("ml03_shap_one", c, marks={"base value (log odds): -3.19": 1, "age          1.039": 2,
                               "insulin      0.764": 3, "bmi          0.229": 4, "predict_proba: 0.282": 5})

c = nb.cell('''
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
for a, f in zip(ax, ["age", "bmi", "n_ed", "n_drugs"]):
    a.scatter(Z_test[f], shap_df[f], s=5, alpha=0.4)
    a.axhline(0, color="gray", lw=0.8)
    a.set_xlabel(f)
ax[0].set_ylabel("SHAP value (log odds)")
plt.tight_layout()

def by(f, groups):
    """특성 f의 SHAP 평균을 groups(사람별 무리 이름)로 묶어서"""
    return shap_df[f].groupby(groups).mean().round(2).to_dict()

age = pd.cut(X_test["age"], [39, 69, 74, 79, 84, 99])
print("age    :", by("age", age.astype(str)))
bmi = pd.cut(X_test["bmi"], [0, 18.5, 23, 25, 30, 60]).astype(str)
bmi[X_test["bmi"].isna()] = "missing"       # 검진을 받지 않은 사람
print("bmi    :", by("bmi", bmi))
print("n_ed   :", by("n_ed", X_test["n_ed"].clip(upper=2)))
print("n_drugs:", by("n_drugs", X_test["n_drugs"] >= 10))
both = X_test["hf"].astype(str) + "/" + X_test["ckd"].astype(str)
print("hf (hf/ckd) :", by("hf", both))
print("ckd (hf/ckd):", by("ckd", both))
''', title="SHAP 의존 그림과 무리별 평균 (참 구조와 견주기)")
grab("dep", lambda: c.stdout)
save("ml03_shap_dep", c, marks={"'(84, 99]': 1.33": 1, "'(0.0, 18.5]': 0.86": 2, "2: 0.6}": 3,
                               "True: 0.87": 4, "'1/1': 0.14": 5})

c = nb.cell('''
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

models = {
    "logistic (ch 0)": make_pipe(
        LogisticRegression(C=0.1, max_iter=1000)),
    "tree (ch 1)": make_tree_pipe(DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=300, random_state=2026)),
    "random forest (ch 2)": make_tree_pipe(RandomForestClassifier(
        n_estimators=500, max_features=20, min_samples_leaf=85,
        max_samples=0.7339919539656937, random_state=2026, n_jobs=-1)),
    "sklearn GBM": make_tree_pipe(
        GradientBoostingClassifier(random_state=2026)),
    "HistGB tuned": make_hgb_pipe(HistGradientBoostingClassifier(
        categorical_features=[0, 1, 2, 3], random_state=2026,
        **settings["tuned"])),
    "XGBoost default": make_nan_pipe(XGBClassifier(random_state=2026)),
    "XGBoost tuned": xgb_final}
table, boots = [], {}
fig = plt.figure(figsize=(5.2, 4.8))
for name, m in models.items():
    cv = cross_val_score(m, X_train, y_train, cv=skf, scoring="roc_auc")
    p = m.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    boots[name] = boot_auc(y_test, p)
    lo, hi = np.percentile(boots[name], [2.5, 97.5])
    table.append([name, cv.mean(), roc_auc_score(y_test, p), lo, hi])
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, lw=1.2, label=name)
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("1 - specificity")
plt.ylabel("Sensitivity")
plt.legend(fontsize=7)
for ref in ["logistic (ch 0)", "random forest (ch 2)"]:
    d = boots["XGBoost tuned"] - boots[ref]
    print(f"XGBoost tuned - {ref}: 95% CI",
          np.percentile(d, [2.5, 97.5]).round(3))
pd.DataFrame(table, columns=["model", "CV AUC", "test AUC", "CI low",
                             "CI high"]).round(3)
''', title="0–3장 모형 비교 표와 ROC 곡선")
grab("compare", lambda: dict(table=[[r[0]] + [float(v) for v in r[1:]] for r in ns["table"]],
                             diff={ref: np.percentile(ns["boots"]["XGBoost tuned"] - ns["boots"][ref], [2.5, 97.5]).tolist()
                                   for ref in ["logistic (ch 0)", "random forest (ch 2)"]}))
save("ml03_compare", c, marks={"[0.019 0.066]": 1, "[0.002 0.028]": 2}, dfmarks={"0.783": 3, "0.717": 4})

c = nb.cell('''
import os
import joblib

xgb_final[-1].save_model("xgb_admit_2023.json")  # XGBoost 형식: 나무만
joblib.dump(xgb_final, "xgb_admit_2023.joblib")  # 전처리까지 통째로

only_trees = XGBClassifier()
only_trees.load_model("xgb_admit_2023.json")
p1 = only_trees.predict_proba(xgb_final[:-1].transform(X_test))[:, 1]
p2 = joblib.load("xgb_admit_2023.joblib").predict_proba(X_test)[:, 1]
print("max difference:", np.abs(p1 - p_xgb).max(),
      np.abs(p2 - p_xgb).max())
print({f: f"{os.path.getsize(f) / 1024:.0f} KB"
       for f in ["xgb_admit_2023.json", "xgb_admit_2023.joblib"]})
''', title="모형 저장과 불러오기 (save_model과 joblib)")
grab("save", lambda: c.stdout)
save("ml03_save", c, marks={"max difference: 0.0 0.0": 1, "'xgb_admit_2023.json': '106 KB'": 2})

colab_cell("ml03_shap_pkg", '''
import shap
print("shap", shap.__version__)

explainer = shap.TreeExplainer(xgb_final[-1])
sv = explainer(Z_test)               # values, base_values, data
print(sv.values.shape, " largest difference from pred_contribs:",
      np.abs(sv.values - contrib[:, :-1]).max())
shap.plots.bar(sv, max_display=12)
shap.plots.beeswarm(sv, max_display=12)
shap.plots.waterfall(sv[who], max_display=10)
''', "shap 패키지로 같은 그림", "ml03-s6", "shap")

colab_cell("ml03_colab_final", '''
lgb_final = make_nan_pipe(LGBMClassifier(
    subsample_freq=1, random_state=2026, verbose=-1,
    **study_lgb.best_params)).fit(X_train, y_train)
cat_final = CatBoostClassifier(cat_features=cat_cols, random_seed=2026,
                               verbose=0).fit(Xk_train, y_train)
b_xgb = boot_auc(y_test, p_xgb)
rows = []
for name, p, cv in [
        ("LightGBM tuned", lgb_final.predict_proba(X_test)[:, 1],
         study_lgb.best_value),
        ("CatBoost default", cat_final.predict_proba(Xk_test)[:, 1],
         cv_cat_default)]:
    b = boot_auc(y_test, p)
    lo, hi = np.percentile(b, [2.5, 97.5])
    d_lo, d_hi = np.percentile(b - b_xgb, [2.5, 97.5])
    rows.append([name, cv, roc_auc_score(y_test, p), lo, hi, d_lo, d_hi])
pd.DataFrame(rows, columns=["model", "CV AUC", "test AUC", "CI low",
                            "CI high", "vs XGB low", "vs XGB high"]
             ).round(3)
''', "LightGBM과 CatBoost의 시험 AUC (비교 표에 더할 줄)", "ml03-s6", "LightGBM, CatBoost")

# ================================================================ 사. 과제
c = nb.cell('''
for lr in [0.3, 0.1, 0.03]:
    m = XGBClassifier(n_estimators=5000, learning_rate=lr, max_depth=2,
                      early_stopping_rounds=50, eval_metric="logloss",
                      random_state=2026)
    m.fit(Z_tr, y_tr, eval_set=[(Z_val, y_val)], verbose=False)
    auc = roc_auc_score(y_val, m.predict_proba(Z_val)[:, 1])
    print(f"learning_rate {lr}: trees {m.best_iteration + 1},"
          f" log loss {m.best_score:.4f}, validation AUC {auc:.4f}")
''', title="과제 1 정답. 학습률과 조기 종료")
grab("hw1", lambda: c.stdout)
save("ml03_hw1", c)

c = nb.cell('''
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

spw = (y_train == 0).sum() / (y_train == 1).sum()   # 10791 / 459
for w in [1, spw]:
    m = make_nan_pipe(XGBClassifier(random_state=2026,
                                    scale_pos_weight=w, **best_xgb))
    p = cross_val_predict(m, X_train, y_train, cv=skf,
                          method="predict_proba")[:, 1]
    print(f"scale_pos_weight={w:.1f}: AUC {roc_auc_score(y_train, p):.4f},"
          f" mean prob {p.mean():.3f}")
    print(confusion_matrix(y_train, (p >= 0.5).astype(int)))
''', title="과제 2 정답. scale_pos_weight와 확률, 혼동행렬")
grab("hw2", lambda: c.stdout)
save("ml03_hw2", c)

c = nb.cell('''
def hgb_objective(trial):
    model = make_hgb_pipe(HistGradientBoostingClassifier(
        categorical_features=[0, 1, 2, 3], early_stopping=False,
        random_state=2026,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                          log=True),
        max_iter=trial.suggest_int("max_iter", 50, 1000, step=50),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 2, 32,
                                         log=True),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 200,
                                           log=True),
        l2_regularization=trial.suggest_float("l2_regularization",
                                              0.001, 10, log=True)))
    return cross_val_score(model, X_train, y_train, cv=skf,
                           scoring="roc_auc").mean()

study_hgb = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=2026))
study_hgb.optimize(hgb_objective, n_trials=30)
print({k: round(v, 4) for k, v in study_hgb.best_params.items()})
print(round(study_hgb.best_value, 4), " trial",
      study_hgb.best_trial.number)
''', title="과제 3 정답. 히스토그램 부스팅을 Optuna로")
grab("hw3", lambda: dict(best=dict(ns["study_hgb"].best_params), best_value=float(ns["study_hgb"].best_value),
                         number=int(ns["study_hgb"].best_trial.number)))
save("ml03_hw3", c)

print("ml03: cells", len(nb.cells), f" elapsed {time.time() - T_START:.0f} s")
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)
# ================================================================ 자료 모으기 (전체 실행 때만): 그림과 대조 블록에 쓸 값
def collect():
    import warnings
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    D = DATA
    Xtr, Xte, ytr, yte = ns["X_train"], ns["X_test"], ns["y_train"], ns["y_test"]
    D["setup"] = [len(Xtr), len(Xte), int(ytr.sum()), int(yte.sum())]
    df_ = ns["df"]
    D["true_rates"] = {
        "age": {str(k): float(v) for k, v in
                df_.groupby(pd.cut(df_["age"], [39, 69, 74, 79, 84, 99]), observed=True)["admit_2023"].mean().items()},
        "n_ed2": float(df_.loc[df_.n_ed >= 2, "admit_2023"].mean()),
    }
    D["groups_train"] = dict(n_ed2=int((Xtr.n_ed >= 2).sum()), hf_ckd=int(((Xtr.hf == 1) & (Xtr.ckd == 1)).sum()),
                             insulin=int(Xtr.insulin.sum()), drugs10=int((Xtr.n_drugs >= 10).sum()))
    tr_csv = pd.read_csv(os.path.join(ROOT, "gen", "_ml_truth.csv")).set_index("id")
    p_true_te = tr_csv.loc[df_.loc[Xte.index, "id"], "p_true"].to_numpy()
    D["true_auc"] = float(roc_auc_score(yte, p_true_te))
    r = float(ytr.mean())
    D["hess_per_person"] = r * (1 - r)
    # 본문의 '지역의 순열 중요도 0.0026(SD 0.0032)': 셀 23과 같은 호출을 다시 해 SD까지 꺼낸다(셀은 평균만 남김)
    from sklearn.inspection import permutation_importance
    pr = permutation_importance(ns["xgb_final"], Xte, yte, scoring="roc_auc", n_repeats=10, random_state=2026)
    j = list(Xte.columns).index("region")
    D["perm_region"] = [float(pr.importances_mean[j]), float(pr.importances_std[j])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        D["ada"] = adaboost_toy()


def adaboost_toy(rounds=3):
    """그림 3-1용: 이산 AdaBoost(y = ±1). 특성은 나이와 응급실, 기준은 이웃 값의 가운데. 동점이면 먼저 찾은 것."""
    age = np.array([52, 61, 47, 78, 83, 69, 88, 74], float)
    ed = np.array([0, 0, 1, 2, 0, 2, 1, 1], float)
    yy = np.array([0, 0, 0, 1, 0, 1, 1, 0]) * 2 - 1
    w = np.full(8, 1 / 8)
    out = []
    F = np.zeros(8)
    for _ in range(rounds):
        best = None
        for fname, x in [("age", age), ("n_ed", ed)]:
            xs = np.unique(x)
            for a, b in zip(xs[:-1], xs[1:]):
                t = (a + b) / 2
                for sgn in (1, -1):
                    h = np.where(x > t, sgn, -sgn)
                    err = w[h != yy].sum()
                    if best is None or err < best[0] - 1e-12:
                        best = (err, fname, t, sgn, h)
        err, fname, t, sgn, h = best
        alpha = 0.5 * np.log((1 - err) / err)
        out.append(dict(w=w.tolist(), feature=fname, t=float(t), sign=int(sgn), err=float(err), alpha=float(alpha),
                        wrong=(h != yy).tolist(), h=h.tolist()))
        F = F + alpha * h
        w = w * np.exp(-alpha * yy * h)
        w = w / w.sum()
    return dict(rounds=out, F=F.tolist(), w_last=w.tolist())


if not REPLAY:
    collect()
    if CACHE:
        cells = [dict(code=cc.code, stdout=cc.stdout, value_repr=cc.value_repr, value_html=cc.value_html,
                      images=cc.images, warns=cc.warns, error=cc.error) for cc in nb.cells]
        with open(CACHE, "wb") as f:
            pickle.dump({"cells": cells, "data": DATA}, f)
        print("cache written:", CACHE)
print(f"collected, elapsed {time.time() - T_START:.0f} s")
if os.environ.get("SHOWDATA"):
    print(json.dumps({k: v for k, v in DATA.items() if k not in ("gbm", "es", "optuna")}, ensure_ascii=False, default=str)[:30000])
if MARK_ERRORS:
    print("MARK ERRORS:")
    for e in MARK_ERRORS:
        print("  ", e)



# ================================================================ 대조 블록: 본문에 적은 숫자와 실행 결과 맞추기
def checks():
    import math
    import pandas as pd
    n_ok = [0]

    def same(label, a, b, tol=1e-9):
        a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
        assert a.shape == b.shape and np.all(np.abs(a - b) <= tol), f"{label}: {a} != {b}"
        n_ok[0] += 1

    D = DATA
    same("split", D["setup"], [11250, 3750, 459, 153])
    # 가 절: 환자 8명 (숫자로 따라가기의 표와 손계산)
    T = D["toy"]["table"]
    same("toy round 1", [T["resid1"][0], T["tree1"][0], T["p1"][0], T["tree1"][3], T["p1"][3]],
         [-0.375, round(-1.25 / 6, 4), 0.2708, 0.625, 0.6875], 1e-4)
    same("toy round 2", [T["resid2"][6], T["tree2"][6], T["tree2"][0], T["p2"][3], T["p2"][6], T["p2"][0]],
         [0.7292, 0.7292, -0.1042, 0.6354, 0.6354, 0.2187], 1e-4)
    assert "round 1: n_ed <= 1.5" in D["toy"]["out"] and "round 2: age <= 85.5" in D["toy"]["out"]
    lo0 = math.log(0.375 / 0.625)
    w0, w1 = 1.25 / (2 * 0.375 * 0.625), 1.25 / (2 * 0.375 * 0.625 + 1)
    sig = lambda v: 1 / (1 + math.exp(-v))  # noqa: E731
    same("log-odds fold", [round(lo0, 3), round(2 * 0.375 * 0.625, 3), round(w0, 3), round(lo0 + 0.5 * w0, 3),
                           round(sig(lo0 + 0.5 * w0), 3), round(w1, 3), round(sig(lo0 + 0.5 * w1), 3)],
         [-0.511, 0.469, 2.667, 0.823, 0.695, 0.851, 0.479])
    A = D["ada"]
    same("adaboost", [round(r["err"], 3) for r in A["rounds"]] + [round(r["alpha"], 2) for r in A["rounds"]]
         + [r["t"] for r in A["rounds"]] + [round(A["rounds"][1]["w"][6], 2)], [0.125, 0.143, 0.083, 0.97, 0.90, 1.20, 1.5, 65, 85.5, 0.5])
    assert [r["feature"] for r in A["rounds"]] == ["n_ed", "age", "age"] and np.all(np.sign(A["F"]) == [-1, -1, -1, 1, -1, 1, 1, -1])
    gb = D["gbm"]
    same("gbm curve", [gb["best"], round(min(gb["val"]), 4), round(gb["val"][0], 4), round(gb["tr"][-1], 4), round(gb["val"][-1], 4)],
         [69, 0.1396, 0.1602, 0.0653, 0.1630])
    same("gbm cv", [round(v, 4) for v in D["gbm_cv"]], [0.7658, 0.0154])
    # 나 절
    xb = D["xgb_basic"]
    same("xgb basic", [round(r[2], 4) for r in xb[:4]], [0.6904, 0.7682, 0.6977, 0.7697])
    assert xb[-1] == "3.4.2"
    ins = D["inside"]
    same("inside", [ins["n_bmi"], ins["n_yes"], float(ins["tree0"][0][2]), float(ins["tree0"][6][6])], [43, 24, 84.0, 0.962585], 1e-6)
    assert ins["cfg"]["min_child_weight"] == "1" and ins["cfg"]["lambda"] == "1" and ins["cfg"]["max_bin"] == "256"
    dp = D["depth"]
    same("depth curve", [round(dp["va"][0], 4), round(dp["va"][-1], 4), int(np.argmax(dp["va"])), round(dp["tr"][-1], 3)],
         [0.7712, 0.7124, 0, 0.998])
    same("hess per person, min_child_weight 1 ~ people", [round(D["hess_per_person"], 3), round(1 / D["hess_per_person"])], [0.039, 26])
    # 라 절
    hg = D["hgb"]
    same("hgb", [round(r[1], 4) for r in hg] + [r[3] for r in hg], [0.7243, 0.7469, 0.7679, 50, 50, 256])
    # 마 절
    es = D["es"]
    same("early stopping", [es["grown"], es["best"], round(es["score"], 4), round(es["auc"], 4)], [210, 159, 0.1373, 0.7958])
    for t in ["stop on logloss: leaky 0.7707, clean 0.7698, leaky higher in 4 of 5 folds",
              "stop on auc    : leaky 0.7770, clean 0.7719, leaky higher in 5 of 5 folds"]:
        assert t in D["es_leak"], t
    same("es_leak differences", [round(0.7707 - 0.7698, 3), round(0.7770 - 0.7719, 3)], [0.001, 0.005])
    st = D["steps"]
    assert "step 1: CV AUC 0.7713 {'model__max_depth': 1" in st["out"] and "step 2: CV AUC 0.7728" in st["out"]
    assert "'model__colsample_bytree': 0.4" in st["out"] and "{'model__reg_lambda': 1}" in st["out"]
    same("steps 4", round(st["cv4"], 4), 0.7713)
    same("steps fits", [(12 + 9 + 4) * 5 + 5, (12 + 9 + 4) * 5 + 3 + 5], [130, 133])
    o = D["optuna"]
    v = np.array(o["values"])
    same("optuna best", [round(o["best_value"], 4), o["best_number"], o["best"]["max_depth"], o["best"]["n_estimators"],
                         round(o["best"]["learning_rate"], 3), round(o["best"]["colsample_bytree"], 2),
                         round(o["best"]["subsample"], 2), round(o["best"]["reg_lambda"], 1)],
         [0.7761, 53, 1, 200, 0.083, 0.40, 0.65, 13.8])
    same("optuna history", [round(v[:10].min(), 3), round(v[:10].max(), 3), (v[10:] > 0.77).sum()], [0.728, 0.773, 22])
    top10 = np.argsort(-v)[:10]
    assert all(o["params"][i]["max_depth"] == 1 for i in top10)
    same("top5 spread", round(v[top10[0]] - v[top10[4]], 4) <= 0.003, 1)
    t5 = sorted((round(o["params"][i]["learning_rate"], 3), o["params"][i]["n_estimators"]) for i in top10[:5])
    same("top5 learning rate and trees (slice-plot text)", t5,
         [(0.045, 750), (0.083, 200), (0.083, 300), (0.098, 250), (0.104, 250)])
    oi = D["optuna_imp"]
    same("optuna importance", [round(oi["max_depth"], 2), round(oi["learning_rate"], 2), round(oi["n_estimators"], 2)], [0.41, 0.23, 0.10])
    assert all(oi[k] < 0.1 for k in ["colsample_bytree", "reg_lambda", "subsample", "min_child_weight", "reg_alpha"])
    h = D["hyperopt"]
    same("hyperopt", [round(h["auc"], 4), h["best"]["max_depth"], round(h["best"]["learning_rate"], 3), h["best"]["n_estimators"]],
         [0.7735, 2, 0.019, 350])
    tu = {r[0]: r[1:] for r in D["tuned"]}
    same("tuned table", [[round(x, 3) for x in tu[k]] for k in ["default", "stepwise grid", "Optuna"]],
         [[0.698, 0.023, 0.717, 0.673, 0.761], [0.771, 0.014, 0.792, 0.746, 0.831], [0.776, 0.016, 0.784, 0.738, 0.825]])
    # 바 절
    im, pm = D["imp"]["imp"], D["imp"]["perm"]
    w = pd.Series(im["weight"]); gn = pd.Series(im["gain"]); cv_ = pd.Series(im["cover"])
    same("importance", [w["bmi"], w.rank(ascending=False, method="min")["bmi"], gn.rank(ascending=False, method="min")["bmi"],
                        w["age"], round(gn["age"], 1), gn.rank(ascending=False, method="min")["age"], (w > 0).sum()],
         [52, 1, 7, 22, 59.4, 1, 26])
    used = cv_[w > 0]
    same("cover range", [round(used.min()), round(used.max())], [246, 266])
    pmS = pd.Series(pm)
    same("perm", [round(pmS["bmi"], 4), round(pmS["age"], 4), round(pmS["n_drugs"], 4), round(pmS["insulin"], 4),
                  round(pmS["n_ed"], 4), round(pmS["dyslip"], 4), round(pmS["oa"], 4), round(pmS["n_prescribers"], 4),
                  round(pmS["region"], 4), pmS.rank(ascending=False)["region"]],
         [0.0528, 0.0519, 0.0067, 0.0065, 0.0060, -0.0003, 0.0, -0.0006, 0.0026, 6])
    same("perm region mean and SD (text)", [round(v_, 4) for v_ in D["perm_region"]], [0.0026, 0.0032])
    sh = D["shap"]
    ma = pd.Series(sh["mean_abs"]).sort_values(ascending=False)
    same("shap", [round(sh["base"], 2), round(1 / (1 + math.exp(-sh["base"])), 3), sh["addit"] < 1e-5,
                  round(ma["bmi"], 2), round(ma["age"], 2), round(ma["n_drugs"], 2), ma.iloc[3] < 0.06,
                  round(ma["dyslip"], 3), ma["oa"], round(ma["region_metro"], 3), ma.rank(ascending=False)["region_metro"],
                  round(ma["n_prescribers"], 3)],
         [-3.19, 0.040, 1, 0.33, 0.33, 0.19, 1, 0.007, 0.0, 0.036, 7, 0.012])
    one = D["one"]
    r = pd.Series(one["row"])
    same("one patient", [one["y"], int(float(one["x"]["age"])), int(float(one["x"]["insulin"])), int(float(one["x"]["n_drugs"])),
                         int(float(one["x"]["n_ed"])), round(r["age"], 2), round(r["insulin"], 2), round(r["n_drugs"], 2),
                         round(r["bmi"], 2), round(r["diuretic"], 2), round(r["cost_2022"], 2),
                         round(r[r.abs() < 0.05].sum(), 2), round(one["base"] + r.sum(), 2), round(one["p"], 3)],
         [1, 85, 1, 7, 0, 1.04, 0.76, 0.25, 0.23, 0.08, -0.08, -0.03, -0.94, 0.282])
    assert pd.isna(one["x"]["bmi"]) or one["x"]["bmi"] == "nan"
    same("one patient: hand sum", round(-3.19 + 1.04 + 0.76 + 0.25 + 0.23 + 0.08 - 0.08 - 0.03, 2), -0.94)
    same("one patient: unrounded total and prob", [round(one["base"] + r.sum(), 3), round(1 / (1 + math.exp(0.936)), 3)],
         [-0.936, 0.282])
    same("odds factor e^1.04", round(math.exp(1.04), 1), 2.8)
    for t in ["'(39, 69]': -0.27", "'(69, 74]': -0.2", "'(74, 79]': 0.32", "'(79, 84]': 0.79", "'(84, 99]': 1.33",
              "'(0.0, 18.5]': 0.86", "'(23.0, 25.0]': -0.41", "'(30.0, 60.0]': 0.24", "'missing': 0.23",
              "0: -0.03, 1: -0.03, 2: 0.6}", "False: -0.13, True: 0.87", "'1/0': 0.14, '1/1': 0.14", "'0/1': 0.25", "'1/1': 0.25"]:
        assert t in D["dep"], t
    n_ok[0] += 1
    cp = {r_[0]: r_[1:] for r_ in D["compare"]["table"]}
    same("compare", [[round(x, 3) for x in cp[k]] for k in cp],
         [[0.749, 0.742, 0.695, 0.785], [0.754, 0.771, 0.726, 0.810], [0.757, 0.770, 0.722, 0.815], [0.766, 0.776, 0.728, 0.818],
          [0.768, 0.783, 0.738, 0.824], [0.698, 0.717, 0.673, 0.761], [0.776, 0.784, 0.738, 0.825]])
    same("diff CI", [[round(x, 3) for x in D["compare"]["diff"][k]] for k in ["logistic (ch 0)", "random forest (ch 2)"]],
         [[0.019, 0.066], [0.002, 0.028]])
    same("paper differences", [round(cp["XGBoost tuned"][1] - cp["logistic (ch 0)"][1], 3),
                               round(cp["XGBoost tuned"][1] - cp["random forest (ch 2)"][1], 3),
                               round(round(cp["XGBoost tuned"][1], 3) - round(cp["random forest (ch 2)"][1], 3), 3)], [0.042, 0.015, 0.014])
    same("true prob AUC and gap", [round(D["true_auc"], 3), round(D["true_auc"] - cp["XGBoost tuned"][1], 3)], [0.810, 0.026])
    same("small groups (train)", [D["groups_train"]["hf_ckd"], D["groups_train"]["n_ed2"]], [41, 184])
    assert "max difference: 0.0 0.0" in D["save"] and "'xgb_admit_2023.json': '106 KB'" in D["save"]
    for t in ["learning_rate 0.3: trees 13, log loss 0.1415, validation AUC 0.7994",
              "learning_rate 0.1: trees 80, log loss 0.1374, validation AUC 0.7975",
              "learning_rate 0.03: trees 273, log loss 0.1370, validation AUC 0.7975"]:
        assert t in D["hw1"], t
    for t in ["scale_pos_weight=1.0: AUC 0.7747, mean prob 0.041", "[[10775    16]", "[  416    43]]",
              "scale_pos_weight=23.5: AUC 0.7681, mean prob 0.378", "[[8832 1959]", "[ 185  274]]"]:
        assert t in D["hw2"], t
    same("hw2 numbers", [16 + 43, 1959 + 274, round(274 / 459, 2), round(274 / 2233, 2), round(10791 / 459, 1)],
         [59, 2233, 0.60, 0.12, 23.5])
    h3 = D["hw3"]
    same("hw3", [round(h3["best_value"], 4), h3["best"]["max_leaf_nodes"], round(h3["best"]["learning_rate"], 3),
                 h3["best"]["max_iter"], h3["best"]["min_samples_leaf"]], [0.7717, 2, 0.017, 850, 80])
    n_ok[0] += 4
    # 지역: 학습·시험 자료의 입원 비율 (본문의 우연한 무늬)
    Xd = pd.read_csv(os.path.join(ROOT, "pub", "data", "ml_claims.csv"))
    from sklearn.model_selection import train_test_split
    yy = Xd["admit_2023"]
    tr_i, te_i = train_test_split(Xd.index, test_size=0.25, stratify=yy, random_state=2026)
    rt = yy[tr_i].groupby(Xd.loc[tr_i, "region"]).mean()
    re_ = yy[te_i].groupby(Xd.loc[te_i, "region"]).mean()
    same("region rates", [round(rt["city"], 3), round(rt["metro"], 3), round(re_["city"], 3), round(re_["metro"], 3)],
         [0.045, 0.034, 0.037, 0.039])
    # 큰 모의 집단(같은 생성식, seed 777): 이 장의 XGBoost 조합 대 2장의 랜덤 포레스트 조합 (작성자 확인용, 약 1분)
    if not os.environ.get("NOPOP"):
        import warnings
        from sklearn.metrics import roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        sys.path.insert(0, os.path.join(ROOT, "gen"))
        import data_ml
        big, _tb = data_ml.simulate(seed=777, n=100000)
        cat = ["insurance", "region", "smoking", "alcohol"]
        cols = [c for c in Xd.columns if c not in ("id", "admit_2023", "cost_2023")]
        num = [c for c in cols if c not in cat]
        Xb, yb = big[cols], big["admit_2023"]

        def pipe(model, num_step):
            cs = Pipeline([("i", SimpleImputer(strategy="constant", fill_value="missing")),
                           ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
            ct = ColumnTransformer([("num", num_step, num), ("cat", cs, cat)], verbose_feature_names_out=False)
            return Pipeline([("prep", ct.set_output(transform="pandas")), ("model", model)])
        te = np.arange(50000, 100000)
        ax_, ar_ = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(3):
                tr = np.arange(k * 11250, (k + 1) * 11250)
                mx = pipe(XGBClassifier(random_state=2026, **D["optuna"]["best"]), "passthrough").fit(Xb.iloc[tr], yb.iloc[tr])
                mr = pipe(RandomForestClassifier(n_estimators=500, max_features=20, min_samples_leaf=85,
                                                 max_samples=0.7339919539656937, random_state=2026, n_jobs=-1),
                          SimpleImputer(strategy="median")).fit(Xb.iloc[tr], yb.iloc[tr])
                ax_.append(roc_auc_score(yb.iloc[te], mx.predict_proba(Xb.iloc[te])[:, 1]))
                ar_.append(roc_auc_score(yb.iloc[te], mr.predict_proba(Xb.iloc[te])[:, 1]))
        print(f"VERIFY population (seed 777): XGBoost {np.round(ax_, 4)} mean {np.mean(ax_):.4f}; "
              f"RF {np.round(ar_, 4)} mean {np.mean(ar_):.4f}")
        same("population means", [round(np.mean(ax_), 3), round(np.mean(ar_), 3), round(np.mean(ax_) - np.mean(ar_), 3)],
             [0.776, 0.767, 0.009])
    print(f"VERIFY: {n_ok[0]} checks passed")
    return n_ok[0]


N_CHECKS = None if SMOKE else checks()

# ================================================================ 그림 3-1, 3-2, 3-3 (SVG)
def _fig(svg_list, cap, cols=1):
    sys.path.insert(0, ROOT)
    from svgplot import figure
    return figure(svg_list, cap, cols=cols)


def _svg(W, H, body, aria):
    return (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="{aria}" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(body)}</svg>')


def _t(o, x, y, s, anchor="start", cls="lbl"):
    from html import escape
    o.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{cls}">{escape(s)}</text>')


def fig_adaboost():
    """그림 3-1. AdaBoost: 틀린 환자의 가중치가 커진다(원의 넓이 ∝ 가중치)."""
    A = DATA["ada"]
    adm = [0, 0, 0, 1, 0, 1, 1, 0]
    W, H = 700, 352
    X0, dx = 136, 50
    o = []
    for k in range(8):
        _t(o, X0 + k * dx, 22, f"{k + 1}", "middle", "lbl strong small")
    _t(o, X0 - 26, 22, "Patient", "end", "lbl mute small")
    rule = {"n_ed": "ED visits", "age": "age"}
    for r, rd in enumerate(A["rounds"]):
        y = 66 + r * 74
        _t(o, 14, y + 4, f"Round {r + 1}", "start", "lbl strong")
        for k in range(8):
            rad = 40 * np.sqrt(rd["w"][k])
            cls = "f2" if adm[k] else "a1 s1"
            o.append(f'<circle cx="{X0 + k * dx}" cy="{y}" r="{rad:.1f}" class="{cls}" stroke-width="1.2"/>')
            if rd["wrong"][k]:
                o.append(f'<circle cx="{X0 + k * dx}" cy="{y}" r="{rad + 4:.1f}" fill="none" class="strongref" '
                         f'stroke-width="1.6" stroke-dasharray="3 3"/>')
        thr = rd["t"]
        thr_s = f"{thr:g}"
        _t(o, X0 + 7 * dx + 36, y - 6, f"{rule[rd['feature']]} > {thr_s} → admit", "start", "lbl small")
        _t(o, X0 + 7 * dx + 36, y + 12, f"error {rd['err']:.3f}, say {rd['alpha']:.2f}", "start", "lbl mute small")
    y = 66 + 3 * 74 - 8
    _t(o, 14, y + 4, "Weighted vote", "start", "lbl strong")
    for k in range(8):
        v = A["F"][k]
        _t(o, X0 + k * dx, y + 4, f"{v:+.2f}", "middle", "lbl strong small" if v > 0 else "lbl small")
    _t(o, X0 + 7 * dx + 36, y + 4, "> 0 → admit", "start", "lbl mute small")
    _t(o, 14, H - 10, "Circle area = weight at the start of the round. Dashed ring = misclassified by that round's stump.",
       "start", "lbl mute small")
    svg = _svg(W, H, o, "AdaBoost의 가중치 변화와 가중 투표")
    r1, r2, r3 = A["rounds"]
    cap = (f"그림 3-1. AdaBoost를 환자 8명(가 절의 표, 주황은 입원한 환자)으로 세 번 돌린 모습. 원의 넓이가 그 회차의 가중치입니다. "
           f"1회차 그루터기(응급실 2회 이상 → 입원)는 7번만 틀려(점선 고리) 오차 {r1['err']:.3f}, 발언권 {r1['alpha']:.2f}이고, "
           f"그 결과 7번의 가중치가 커집니다(1/8 → {r2['w'][6]:.2f}). 2회차는 7번을 맞히려고 나이 {r2['t']:g}세를 고르고, "
           f"3회차는 2회차에 틀린 사람과 7번을 함께 보며 나이 {r3['t']:g}세를 고릅니다. 맨 아래 줄은 세 그루터기의 "
           f"'입원(+1), 아님(−1)'에 발언권을 곱해 더한 값이며 0보다 크면 입원으로 분류합니다. 발언권은 "
           f"½ log((1 − 오차) ÷ 오차)입니다.")
    nb.save_fragment("ml03_fig_ada", _fig([svg], cap))
    return dict(err=[r["err"] for r in A["rounds"]], alpha=[r["alpha"] for r in A["rounds"]],
                t=[r["t"] for r in A["rounds"]], F=A["F"])


def _tree_icon(o, cx, cy, s=1.0, cls="s1"):
    """작은 나무 그림: 마디 셋과 가지 둘"""
    pts = [(cx, cy - 18 * s), (cx - 16 * s, cy + 10 * s), (cx + 16 * s, cy + 10 * s)]
    for a in pts[1:]:
        o.append(f'<line x1="{pts[0][0]:.1f}" y1="{pts[0][1]:.1f}" x2="{a[0]:.1f}" y2="{a[1]:.1f}" class="ln {cls}" stroke-width="1.6"/>')
    for (x, y) in pts:
        o.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5.5 * s:.1f}" class="f{cls[-1]}"/>')


def _arrow(o, x1, y1, x2, y2):
    import math
    o.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="ref" stroke-width="1.3"/>')
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + 2.6, ang - 2.6
    o.append(f'<polygon points="{x2:.1f},{y2:.1f} {x2 + 7 * math.cos(a1):.1f},{y2 + 7 * math.sin(a1):.1f} '
             f'{x2 + 7 * math.cos(a2):.1f},{y2 + 7 * math.sin(a2):.1f}" class="f4"/>')


def _box(o, x, y, w, h, label, cls="a4", lcls="lbl small"):
    o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="4" class="{cls}" stroke-width="1"/>')
    _t(o, x + w / 2, y + h / 2 + 4, label, "middle", lcls)


def fig_forest_vs_boost():
    """그림 3-2. 랜덤 포레스트(나란히 키워 평균)와 부스팅(차례로 키워 고침)."""
    W, H = 420, 330
    o = []
    _t(o, 12, 20, "A. Random forest (chapter 2)", "start", "ptitle")
    _box(o, 120, 36, 180, 26, "Training data", "a4")
    xs = [80, 210, 340]
    for k, x in enumerate(xs):
        _arrow(o, 210, 62, x, 90)
        _box(o, x - 52, 92, 104, 24, f"Bootstrap {k + 1}", "a1 s1")
        _arrow(o, x, 116, x, 140)
        _tree_icon(o, x, 170, 1.0, "s1")
        _arrow(o, x, 196, 210 + (x - 210) * 0.25, 232)
    _box(o, 110, 236, 200, 28, "Average the probabilities", "a1 s1", "lbl small strong")
    _t(o, 210, 290, "Trees grow in parallel and do not", "middle", "lbl mute small")
    _t(o, 210, 306, "know about each other.", "middle", "lbl mute small")
    a = _svg(W, H, o, "랜덤 포레스트: 나란히 키워 평균")
    o = []
    _t(o, 12, 20, "B. Boosting (this chapter)", "start", "ptitle")
    _box(o, 20, 36, 380, 26, "Start: same risk for everyone (4.1%)", "a4")
    ys = [104, 178, 252]
    _arrow(o, 95, 62, 95, ys[0] - 22)
    for k, y in enumerate(ys):
        if k:                                   # 앞 회차의 '+ lr × tree' → 다음 회차의 남은 오차
            yp = ys[k - 1]
            o.append(f'<polyline points="350,{yp + 4} 350,{yp + 22} 95,{yp + 22}" fill="none" class="ref" stroke-width="1.3"/>')
            _arrow(o, 95, yp + 22, 95, y - 22)
        _box(o, 20, y - 20, 150, 24, "Errors left so far" if k else "Errors (actual − risk)", "a2 s2")
        _arrow(o, 172, y - 8, 214, y - 8)
        _tree_icon(o, 240, y - 6, 0.85, "s2")
        _arrow(o, 266, y - 8, 296, y - 8)
        _box(o, 298, y - 20, 104, 24, f"+ lr × tree {k + 1}", "a2 s2")
    _t(o, 210, 300, "Each small tree learns what the trees", "middle", "lbl mute small")
    _t(o, 210, 316, "before it got wrong; risk = sum of steps.", "middle", "lbl mute small")
    b = _svg(W, H, o, "부스팅: 차례로 키워 고침")
    cap = ("그림 3-2. 랜덤 포레스트와 부스팅. A는 2장의 랜덤 포레스트로, 부트스트랩 표본마다 깊은 나무를 따로 키워 확률을 "
           "평균합니다. 나무끼리 서로를 모르므로 동시에 키울 수 있습니다. B의 부스팅은 모두 같은 출발 위험에서 시작해, "
           "지금까지 틀린 만큼(남은 오차)을 작은 나무가 배우고 그 예측에 학습률(lr)을 곱해 더하는 일을 차례로 되풀이합니다. "
           "앞 나무가 끝나야 다음 나무가 무엇을 배울지 정해지므로 나무를 차례로 키웁니다.")
    nb.save_fragment("ml03_fig_rf_boost", _fig([a, b], cap, cols=2))


def fig_leafwise():
    """그림 3-3. 층 우선 성장(depthwise)과 잎 우선 성장(leaf-wise), 나누기 세 번."""
    def tree(title, nodes, edges, sub):
        W, H = 420, 270
        o = []
        _t(o, 12, 20, title, "start", "ptitle")
        for (a, b) in edges:
            (x1, y1, *_), (x2, y2, k2, _l) = nodes[a], nodes[b]
            y2 = y2 - 12 if k2 == "leaf" else y2
            o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln s4" stroke-width="1.6"/>')
        for key, (x, y, kind, lab) in nodes.items():
            if kind == "split":
                o.append(f'<circle cx="{x}" cy="{y}" r="15" class="f1"/>')
                o.append(f'<text x="{x}" y="{y + 5}" text-anchor="middle" class="lbl strong" fill="#fff" '
                         f'style="fill:#fff">{lab}</text>')
            else:
                o.append(f'<rect x="{x - 15}" y="{y - 12}" width="30" height="24" rx="3" class="a2 s2" stroke-width="1.2"/>')
        _t(o, 210, 250, sub, "middle", "lbl mute small")
        return _svg(W, H, o, title)
    n1 = {"r": (210, 60, "split", "1"), "L": (120, 130, "split", "2"), "R": (300, 130, "split", "3"),
          "LL": (80, 200, "leaf", ""), "LR": (160, 200, "leaf", ""), "RL": (260, 200, "leaf", ""), "RR": (340, 200, "leaf", "")}
    e1 = [("r", "L"), ("r", "R"), ("L", "LL"), ("L", "LR"), ("R", "RL"), ("R", "RR")]
    n2 = {"r": (210, 52, "split", "1"), "L": (140, 106, "split", "2"), "R": (280, 106, "leaf", ""),
          "LL": (90, 160, "split", "3"), "LR": (190, 160, "leaf", ""), "LLL": (50, 214, "leaf", ""), "LLR": (130, 214, "leaf", "")}
    e2 = [("r", "L"), ("r", "R"), ("L", "LL"), ("L", "LR"), ("LL", "LLL"), ("LL", "LLR")]
    a = tree("A. Level-wise (XGBoost default, sklearn GBM)", n1, e1, "Split every node of a level, then go deeper")
    b = tree("B. Leaf-wise (LightGBM, HistGradientBoosting)", n2, e2, "Always split the leaf that cuts the loss most")
    cap = ("그림 3-3. 나누기 세 번(파란 원 안의 숫자는 나눈 순서)으로 잎 네 개(주황 네모)를 만든 두 가지 방법. "
           "A의 층 우선 성장은 한 층의 마디를 모두 나눈 뒤 다음 층으로 가므로 깊이(max_depth)로 크기를 정합니다. "
           "B의 잎 우선 성장은 지금 있는 잎 가운데 손실을 가장 많이 줄이는 잎 하나만 골라 나누므로 한쪽으로 깊어질 수 있고, "
           "그래서 잎의 수(num_leaves, max_leaf_nodes)로 크기를 정합니다. 같은 잎 수라면 B가 손실을 더 많이 줄이지만 "
           "작은 무리를 깊게 파고들어 과적합하기도 쉽습니다.")
    nb.save_fragment("ml03_fig_leafwise", _fig([a, b], cap, cols=2))


FIGINFO = {}
if not SMOKE:
    FIGINFO["ada"] = fig_adaboost()
    fig_forest_vs_boost()
    fig_leafwise()
    print("FIGINFO", FIGINFO)


# ================================================================ Colab 노트북 (주인 실행용)과 대응표
SEC_NAME = {"ml03-s1": "가 절", "ml03-s2": "나 절", "ml03-s3": "다 절", "ml03-s4": "라 절", "ml03-s5": "마 절",
            "ml03-s6": "바 절", "ml03-s7": "사 절"}
COLAB_INTRO = """# 머신러닝 기초 3장 · Colab 실행용 노트북 (LightGBM, CatBoost, shap)

사이트 '머신러닝 기초' 3장(부스팅)에서 LightGBM, CatBoost, shap이 필요한 셀만 모은 노트북입니다. 사이트를 만든 컴퓨터에는 이 세 패키지가 없어서, Colab에서 한 번 돌려 나온 출력을 본문에 넣습니다.

**할 일 (네 가지)**

1. Colab(https://colab.research.google.com)에서 **파일 → 노트북 업로드**를 눌러 이 파일(ml03_colab.ipynb)을 엽니다.
2. 메뉴의 **런타임 → 모두 실행**을 누릅니다. 무료 Colab에서 10분 안팎 걸립니다. 맨 위 설치 셀 뒤에 '세션을 다시 시작하라(Restart session)'는 안내가 나오면 그 버튼을 누른 뒤 다시 **런타임 → 모두 실행**을 누릅니다.
3. 맨 아래 셀까지 끝나면 **파일 → 다운로드 → .ipynb 다운로드**로 파일을 받습니다(셀의 출력이 파일 안에 함께 저장됩니다).
4. 받은 파일을 구글 드라이브의 **'사회약학 연구실'** 폴더에 올립니다.

중간에 빨간 오류가 나면 거기서 멈춘 상태 그대로 3, 4를 하면 됩니다(어느 셀에서 왜 멈췄는지가 파일에 남습니다). 셀 위의 제목에 적힌 '3장 ○ 절 셀 n'은 사이트 본문에서 그 출력이 들어갈 셀 번호입니다. 셀의 코드는 고치지 않고 그대로 실행해 주세요.
"""
COLAB_VERSIONS = '''
import os, sys
import numpy, pandas, sklearn, xgboost, lightgbm, catboost, shap, optuna
print("python", sys.version.split()[0], " CPU", os.cpu_count())
for m in [numpy, pandas, sklearn, xgboost, lightgbm, catboost, shap, optuna]:
    print(m.__name__, m.__version__)
'''


def colab_prep_code(cno):
    best = DATA["optuna"]["best"]
    best_s = "{" + ",\n            ".join(f'"{k}": {v!r}' for k, v in best.items()) + "}"
    return f'''
# 이 셀은 본문에 들어가지 않습니다. 본문의 앞 셀(XGBoost 등)이 만든 것 가운데
# 아래 셀들이 쓰는 것만 같은 코드로 다시 만듭니다.
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

X_tr, X_val, y_tr, y_val = train_test_split(              # 본문 셀 {cno["ml03_gbm_curve"]}
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=2026)
prep_nan = make_nan_pipe(None).named_steps["prep"]        # 본문 셀 {cno["ml03_xgb_inside"]}
Z_train = prep_nan.fit_transform(X_train)
Z_test = prep_nan.transform(X_test)
Z_tr, Z_val = Z_train.loc[X_tr.index], Z_train.loc[X_val.index]   # 본문 셀 {cno["ml03_es"]}

# 본문 셀 {cno["ml03_optuna"]}에서 Optuna가 고른 조합 (사이트를 만든 환경, xgboost {DATA["xgb_basic"][-1]})
best_xgb = {best_s}
xgb_final = make_nan_pipe(XGBClassifier(random_state=2026, **best_xgb))
xgb_final.fit(X_train, y_train)                           # 본문 셀 {cno["ml03_tuned"]}
p_xgb = xgb_final.predict_proba(X_test)[:, 1]
booster = xgb_final[-1].get_booster()                     # 본문 셀 {cno["ml03_imp"]}
contrib = booster.predict(xgb.DMatrix(Z_test), pred_contribs=True)   # 셀 {cno["ml03_shap"]}
who = int(np.argsort(-p_xgb)[75])                         # 본문 셀 {cno["ml03_shap_one"]}
print("XGBoost tuned, test AUC:", round(roc_auc_score(y_test, p_xgb), 4),
      " (site: {DATA["tuned"][2][3]:.4f})")
print("patient who:", who, " (site: {DATA["one"]["who"]})")
'''


def write_colab():
    cno = {}
    for cc in nb.cells:
        pass
    # 조각 이름 → 셀 번호
    for fn in sorted(os.listdir(os.path.join(ROOT, "figs"))):
        if fn.startswith("ml03_") and fn.endswith(".html"):
            m = re.search(r'<span class="cell-n">셀 (\d+)</span>', open(os.path.join(ROOT, "figs", fn), encoding="utf-8").read())
            if m:
                cno[fn[:-5]] = int(m.group(1))

    def md(t):
        return {"cell_type": "markdown", "metadata": {}, "source": _dedent(t).splitlines(keepends=True)}

    def code(t):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": _dedent(t).splitlines(keepends=True)}
    setup = [cc for cc in nb.cells if "def make_nan_pipe" in cc.code][0]
    cells = [md(COLAB_INTRO), md("## 설치와 버전"), code("!pip install optuna lightgbm catboost shap"),
             code(COLAB_VERSIONS),
             md(f"## 준비\n\n본문 셀 {setup.n}의 코드 그대로입니다. 자료를 사이트 주소에서 읽고 0장과 같은 분할과 전처리를 만듭니다."),
             code(setup.code),
             md("## 본문에 들어가지 않는 준비\n\n아래 셀들이 쓰는 XGBoost 모형과 자료를 다시 만듭니다. 출력의 시험 AUC가 "
                "사이트의 값과 같거나 끝자리만 다르면 정상입니다(XGBoost 버전이 다르면 조금 다를 수 있습니다)."),
             code(colab_prep_code(cno))]
    cmap = []
    # shap 셀은 XGBoost·shap 버전 조합에 따라 오류가 날 수 있으므로 맨 뒤에 둔다('모두 실행'이 그 오류에서 멈춰도
    # LightGBM·CatBoost의 결과는 모두 남도록). 다른 셀은 shap 셀이 만든 것을 쓰지 않는다.
    for item in sorted(COLAB_CELLS, key=lambda it: it["pkg"] == "shap"):
        c = item["cell"]
        cells.append(md(f"<!-- {item['name']} -->\n### 3장 {SEC_NAME[item['section']]} 셀 {c.n}에 들어갈 출력 · {c.title}"))
        cells.append(code(c.code))
        cmap.append(dict(name=item["name"], site_cell=c.n, title=c.title, section=item["section"], needs=item["pkg"],
                         fig=f"figs/{item['name']}.html", colab_cell_index=len(cells) - 1,
                         code_sha1=hashlib.sha1(c.code.encode("utf-8")).hexdigest()))
    nbj = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
           "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.join(ROOT, "pub", "colab"), exist_ok=True)
    path = os.path.join(ROOT, "pub", "colab", "ml03_colab.ipynb")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nbj, f, ensure_ascii=False, indent=1)
    # 본문의 [[COLAB:키]] 표시 목록 (본문을 읽어서)
    markers = []
    cp = os.path.join(ROOT, "content", "ml03.html")
    if os.path.exists(cp):
        src = open(cp, encoding="utf-8").read()
        secs_ = [(m.start(), m.group(1)) for m in re.finditer(r'<section class="sec" id="(ml03-s\d)">', src)]
        for m in re.finditer(r"\[\[COLAB:([a-z0-9_]+)\]\]", src):
            sec = [s for p, s in secs_ if p < m.start()]
            ctx = re.sub(r"<[^>]+>", "", src[max(0, m.start() - 160):m.start()])[-110:]
            markers.append(dict(key=m.group(1), section=sec[-1] if sec else None,
                                source=COLAB_KEYS.get(m.group(1), ""), context=ctx.strip()))
    with open(os.path.join(ROOT, "gen", "_ml03_colab_map.json"), "w", encoding="utf-8") as f:
        json.dump({"chapter": "ml03", "colab_notebook": "pub/colab/ml03_colab.ipynb",
                   "how_to_fill": ("1) 실행된 노트북을 받아 COLAB_NB=<그 파일> python3 gen/ml_ml03.py "
                                   "(REPLAY=1 ML03_CACHE=<캐시>와 함께 쓰면 다른 셀은 다시 돌리지 않음)로 figs/<name>.html을 "
                                   "실제 출력으로 바꾼다. 마크다운의 <!-- ml03_xxx --> 표시로 셀을 찾고, 코드가 본문과 다르면 "
                                   "MARK ERRORS에 적는다. 2) content/ml03.html의 [[COLAB:키]]를 아래 markers의 source가 "
                                   "가리키는 출력의 숫자로 바꾼다. 3) 출력 읽기의 표식(marks)은 colab_cell(..., marks=)로 더한다."),
                   "cells": cmap, "markers": markers}, f, ensure_ascii=False, indent=1)
    return path, cmap, markers


# 본문의 [[COLAB:키]]가 어느 셀의 어느 출력에서 오는지
COLAB_KEYS = {
    "lgb_default_cv": "ml03_lgb_default: default 줄의 CV AUC",
    "lgb_few_cv": "ml03_lgb_default: 4 leaves 줄의 CV AUC",
    "lgb_time": "ml03_lgb_default: 두 줄의 걸린 초 (5겹)",
    "lgb_leaves_best": "ml03_lgb_leaves: CV AUC가 가장 높은 num_leaves와 그 CV AUC",
    "lgb_leaves_31": "ml03_lgb_leaves: num_leaves 31의 CV AUC",
    "lgb_cat_cv": "ml03_lgb_cat: 두 줄의 CV AUC (one-hot / category)",
    "cat_default_cv": "ml03_cat: default 줄의 CV AUC",
    "cat_ordered_cv": "ml03_cat: ordered 줄의 CV AUC",
    "cat_lr": "ml03_cat: default 줄 아래 learning_rate (자동으로 정한 값)",
    "cat_time": "ml03_cat: default와 ordered의 걸린 초",
    "lgb_es_best": "ml03_lgb_es: best_iteration_",
    "lgb_es_auc": "ml03_lgb_es: validation AUC",
    "lgb_opt_cv": "ml03_lgb_optuna: 최고 CV AUC와 그 시도의 조합",
    "lgb_test": "ml03_colab_final: LightGBM tuned 줄의 test AUC와 CI",
    "cat_test": "ml03_colab_final: CatBoost default 줄의 test AUC와 CI",
    "lgb_vs_xgb": "ml03_colab_final: LightGBM tuned 줄의 vs XGB low–high",
    "cat_vs_xgb": "ml03_colab_final: CatBoost default 줄의 vs XGB low–high",
    "shap_diff": "ml03_shap_pkg: largest difference from pred_contribs",
}

if not SMOKE:
    _cp, _cmap, _markers = write_colab()
    print("colab notebook:", _cp, " cells mapped:", [(m["name"], m["site_cell"], m["colab_cell_index"]) for m in _cmap])
    print("markers:", len(_markers), sorted({m["key"] for m in _markers}))


# ================================================================ 학생용 노트북 (LightGBM·CatBoost·shap 셀은 실행하지 않은 채 포함)
def _sec_start(code_piece):
    return [cc.n for cc in nb.cells if code_piece in cc.code][0]


if not SMOKE:
    notes = {
        1: ("## 설치\n\nColab에서는 세션마다 이 셀부터 실행합니다. LightGBM과 shap은 Colab에 이미 있을 수 있고, 그러면 "
            "`Requirement already satisfied`로 나옵니다. 설치 뒤 세션을 다시 시작하라는 안내가 나오면 따르고 셀 2부터 다시 실행합니다."),
        2: "## 준비\n\n0장의 자료 불러오기와 분할, 0–1장의 전처리 함수, 이 장에서 새로 쓰는 `make_nan_pipe`(숫자 열의 결측을 그대로 두는 전처리), 1장의 `boot_auc`를 한 셀에 모았습니다.",
        _sec_start("toy = pd.DataFrame"): "## 가. 앞 나무의 오차를 다음 나무가 배운다",
        _sec_start("shallow = dict("): "## 나. XGBoost",
        _sec_start("import lightgbm as lgb"): "## 다. LightGBM\n\n이 절의 셀은 LightGBM이 필요합니다(Colab에는 있습니다). 사이트의 출력은 Colab에서 실행한 결과입니다.",
        _sec_start("settings = {"): "## 라. 히스토그램 부스팅과 CatBoost",
        _sec_start("es_model = XGBClassifier"): "## 마. 조기 종료와 튜닝",
        _sec_start("from sklearn.inspection import permutation_importance"): "## 바. 변수 기여도와 결과 정리",
        _sec_start("for lr in [0.3, 0.1, 0.03]"): ("## 과제 정답\n\n**과제 1.** 학습률 0.3, 0.1, 0.03에서 조기 종료가 고른 나무 수와 검증 손실, "
                                                  "검증 AUC를 비교합니다."),
        _sec_start("spw = "): "**과제 2.** scale_pos_weight를 1과 (음성 수 ÷ 양성 수)로 바꿔 겹 밖 예측의 AUC, 평균 확률, 혼동행렬을 비교합니다.",
        _sec_start("def hgb_objective"): "**과제 3.** 히스토그램 부스팅을 Optuna로 30번 튜닝합니다.",
    }
    nb.save_ipynb(
        "머신러닝 기초 3장. 부스팅",
        intro=("사회약학 연구방법 노트 '머신러닝 기초' 3장을 따라 하는 노트북입니다. 0장과 같은 가상 청구자료와 분할로 "
               "그레이디언트 부스팅의 원리를 손으로 따라가고, XGBoost, LightGBM, 히스토그램 부스팅, CatBoost를 만들어 "
               "조기 종료와 Optuna로 튜닝한 뒤 SHAP 값으로 결과를 정리합니다. 셀을 위에서부터 차례로 실행하세요. "
               "튜닝 셀은 Colab에서 몇 분씩 걸립니다. 자료는 사이트가 만든 가상 자료이며 실제 환자 자료가 아닙니다."),
        notes=notes)

os.chdir(_CWD)
