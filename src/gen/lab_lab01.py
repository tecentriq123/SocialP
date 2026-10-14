"""실습 1 · 자료 탐색과 정규성 검정 — real cells run with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab01.py
      then  python3 tools/nbcheck.py lab01   (runs the notebook alone in a fresh process and compares outputs)
Writes figs/lab01_*.html (insert in content/lab01.html with <!--FIG:lab01_xxx-->),
pub/notebooks/lab01.ipynb (download link at the top of content/lab01.html) and checks/lab01.expected.json.
The Table 1 paper-box table is generated from the same results (figs/lab01_t1.html) so that the paper box
and the cell outputs cannot disagree. The VERIFY block at the end asserts the numbers quoted in the text of
content/lab01.html that are not marked in a cell output (대응표, prose, paper boxes, 직접 바꿔 보기, 과제).
(2026-10-14: version cell 1, the 1장 c 절 five-p-value cell 19, 과제 1–3 and the notebook added;
 old cells 1–17 became 2–18 and old 18–22 became 20–24.)
"""
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labkit import Notebook  # noqa: E402

nb = Notebook("lab01")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


def mark_out(h, marks):
    """<span class="mk">n</span> after the first occurrence of each text in the cell OUTPUT (text nodes only).
    labkit's own `marks` needs every mark in both the printed text and the table, so it is not used here."""
    if not marks:
        return h
    i0 = h.find('<div class="cell-out">')
    head, body = h[:i0], h[i0:]
    for sub, n in marks.items():
        parts = re.split(r"(<[^>]+>)", body)
        done = False
        for k, p in enumerate(parts):
            if p.startswith("<"):
                continue
            for e in (html.escape(sub), html.escape(sub, quote=False)):
                j = p.find(e)
                if j >= 0:
                    parts[k] = p[:j + len(e)] + f'<span class="mk">{n}</span>' + p[j + len(e):]
                    done = True
                    break
            if done:
                break
        if not done:
            raise ValueError(f"mark text not found in output: {sub!r}")
        body = "".join(parts)
    return head + body


def save(name, cell, marks=None):
    nb.save_fragment(name, mark_out(nb.html(cell), marks))


def show(cell):
    print(f"--- cell {cell.n}: {cell.title}")
    print(cell.stdout.rstrip())
    if cell.value_repr is not None:
        print(cell.value_repr)
    for w in cell.warns:
        print("WARN", w)


# ---------------------------------------------------------------- a. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer
import scipy, matplotlib, statsmodels, sklearn

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scipy       ", scipy.__version__)
print("matplotlib  ", matplotlib.__version__)
print("statsmodels ", statsmodels.__version__)
print("scikit-learn", sklearn.__version__)
''', title="패키지 불러오기와 버전 확인")
_v = {k: nb.ns[k].__version__ for k in ("pd", "scipy", "statsmodels")}
show(c); save("lab01_versions", c, marks={f"pandas       {_v['pd']}": 1, f"scipy        {_v['scipy']}": 2,
                                          f"statsmodels  {_v['statsmodels']}": 3})
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤, scikit-learn에 들어 있는 "
              "위스콘신 유방암 진단 자료를 읽습니다. 인터넷 연결은 필요 없습니다.")

c = nb.cell('''
bc = load_breast_cancer(as_frame=True)
df = bc.frame.copy()     # 설명변수 30개 + target 1개
print(df.shape)
print(df["target"].value_counts())
''', title="데이터 불러오기")
show(c); save("lab01_load", c, marks={"(569, 31)": 1, "357": 2, "212": 3})

c = nb.cell('''
labels = {0: "malignant", 1: "benign"}      # sklearn의 코딩
df["diagnosis"] = pd.Categorical(df["target"].map(labels),
                                 categories=["benign", "malignant"])
df["malignant"] = (df["target"] == 0).astype(int)   # 1 = 악성
print(df["malignant"].sum())           # 악성 건수
pd.crosstab(df["target"], df["diagnosis"])   # 바꾼 결과 확인
''', title="진단 변수를 읽기 쉽게 바꾸기")
show(c); save("lab01_recode", c, marks={"212": 1})

c = nb.cell('''
features = bc.feature_names.tolist()   # 설명변수 30개의 이름
print(len(features))
print(features[:10])    # 0–9번: mean 계열
print(features[10:13])  # 10–19번: error 계열 (앞의 3개만)
print(features[20:23])  # 20–29번: worst 계열 (앞의 3개만)
''', title="설명변수 30개의 구성")
show(c); save("lab01_features", c, marks={"30": 1, "'radius error'": 2, "'worst radius'": 3})

c = nb.cell('''
df[["mean radius", "mean smoothness", "area error",
    "diagnosis", "malignant"]].head()
''', title="앞의 5행")
show(c); save("lab01_head", c)

# ---------------------------------------------------------------- b. 요약통계와 그래프
c = nb.cell('''
cols = ["mean radius", "mean smoothness", "mean area", "area error"]
df[cols].describe().round(3)
''', title="describe()")
show(c); save("lab01_describe", c, marks={"40.337": 1, "24.530": 2, "542.200": 3})
NOTES[c.n] = "## b. 요약통계와 그래프"

c = nb.cell('''
summary = pd.DataFrame({
    "mean": df[cols].mean(),
    "median": df[cols].median(),
    "skewness": df[cols].apply(stats.skew),   # 열마다 왜도
})
summary.round(3)
''', title="평균, 중앙값, 왜도")
show(c); save("lab01_skew", c, marks={"0.455": 1, "5.433": 2})

c = nb.cell('''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, col in zip(axes, ["mean smoothness", "area error"]):
    ax.hist(df[col], bins=30, color="lightgray", edgecolor="gray")
    ax.axvline(df[col].mean(), color="C0", label="Mean")
    ax.axvline(df[col].median(), color="C1", linestyle="--",
               label="Median")
    ax.set_title(col)
    ax.set_xlabel("Value")
    ax.set_ylabel("Number of samples")
    ax.legend()
plt.tight_layout()
plt.show()
''', title="히스토그램")
show(c); save("lab01_hist", c)
PLT_CELL = c.n

c = nb.cell('''
df.groupby("diagnosis", observed=True)["mean radius"].agg(
    ["count", "mean", "std", "median"]).round(2)
''', title="진단별 요약")
show(c); save("lab01_group", c, marks={"12.15": 1, "17.46": 2})

c = nb.cell('''
groups = ["benign", "malignant"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, col in zip(axes, ["mean radius", "area error"]):
    data = [df.loc[df["diagnosis"] == g, col] for g in groups]
    ax.boxplot(data)
    ax.set_xticks([1, 2], groups)
    ax.set_title(col)
    ax.set_ylabel("Value")
plt.tight_layout()
plt.show()
''', title="진단별 상자그림")
show(c); save("lab01_box", c)

c = nb.cell('''
t1_vars = ["mean radius", "mean texture", "mean smoothness",
           "mean area", "area error"]
g = df.groupby("diagnosis", observed=True)[t1_vars]
m = g.mean().T     # 행: 변수, 열: benign / malignant
s = g.std().T      # 표준편차 (n - 1로 나눔)
pd.concat({"mean": m, "sd": s}, axis=1).round(3)
''', title="군별 평균과 표준편차")
show(c); save("lab01_t1_msd", c, marks={"12.147": 1, "1.781": 2})
T1_FIRST = c.n

c = nb.cell('''
q = g.quantile([0.25, 0.5, 0.75]).T   # 1사분위수, 중앙값, 3사분위수
q.round(2)
''', title="군별 사분위수")
show(c); save("lab01_t1_q", c, marks={"458.40": 1})

c = nb.cell('''
# SMD = (평균 차이) / 두 군 분산의 평균의 제곱근
smd = (m["malignant"] - m["benign"]) / np.sqrt(
    (s["malignant"] ** 2 + s["benign"] ** 2) / 2)
print(smd.round(2))
''', title="표준화 평균차(SMD)")
show(c); save("lab01_smd", c, marks={"2.05": 1})
SMD_CELL = c.n

# Table 1 for the paper box, generated from the notebook's own results
ns = nb.ns
m, s, q, smd = ns["m"], ns["s"], ns["q"], ns["smd"]
n_b = int((ns["df"]["diagnosis"] == "benign").sum())
n_m = int((ns["df"]["diagnosis"] == "malignant").sum())
rows = [("Radius (mean), mean ± SD", "mean radius", "msd", 2, 1),
        ("Texture (mean), mean ± SD", "mean texture", "msd", 2, None),
        ("Smoothness (mean), mean ± SD", "mean smoothness", "msd", 3, 2),
        ("Area (mean), median (IQR)", "mean area", "iqr", 1, 3),
        ("Area (standard error), median (IQR)", "area error", "iqr", 1, None)]
body = []
T1 = {}
for label, v, kind, d, mk in rows:
    cells = []
    for grp in ("malignant", "benign"):
        if kind == "msd":
            txt = f"{m.loc[v, grp]:.{d}f} ± {s.loc[v, grp]:.{d}f}"
        else:
            txt = (f"{q.loc[v, (grp, 0.5)]:.{d}f} "
                   f"({q.loc[v, (grp, 0.25)]:.{d}f}–{q.loc[v, (grp, 0.75)]:.{d}f})")
        cells.append(txt)
        T1[(v, grp)] = txt
    T1[(v, "smd")] = f"{smd[v]:.2f}"
    lab = (f'<span class="mk">{mk}</span> ' if mk else "") + label
    smd_txt = ('<span class="mk">4</span> ' if v == "mean radius" else "") + f"{smd[v]:.2f}"
    body.append(f'<tr><td>{lab}</td><td class="r">{cells[0]}</td><td class="r">{cells[1]}</td>'
                f'<td class="r">{smd_txt}</td></tr>')
t1_html = ('<div class="tbl-wrap"><table class="jt">'
           '<caption><b>Table 1.</b> Nuclear features of fine-needle aspirates by diagnosis</caption>'
           f'<thead><tr><th>Feature</th><th class="r">Malignant<br>(n = {n_m})</th>'
           f'<th class="r">Benign<br>(n = {n_b})</th><th class="r">SMD</th></tr></thead>'
           f'<tbody>{"".join(body)}</tbody></table></div>')
nb.save_fragment("lab01_t1", t1_html)
print("TABLE1", T1)

# ---------------------------------------------------------------- c. Q-Q plot과 정규성 검정
c = nb.cell('''
benign = df[df["diagnosis"] == "benign"]      # 양성 357건
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, col in zip(axes, ["mean radius", "area error"]):
    stats.probplot(benign[col], dist="norm", plot=ax)
    ax.set_title(col + " (benign)")
plt.tight_layout()
plt.show()
''', title="Q-Q plot: 양성 종양 안에서")
show(c); save("lab01_qq", c)
NOTES[c.n] = "## c. Q-Q plot과 정규성 검정"
QQ_CELL = c.n

c = nb.cell('''
for col in ["mean radius", "area error"]:
    x = benign[col]
    w, p = stats.shapiro(x)       # 검정통계량 W와 p-value
    print(f"{col}: n = {len(x)}, skewness = {stats.skew(x):.2f}, "
          f"W = {w:.3f}, p = {p:.3g}")
''', title="Shapiro-Wilk 검정")
show(c); save("lab01_shapiro", c, marks={"W = 0.997": 1, "p = 0.668": 2, "W = 0.896": 3, "p = 6.76e-15": 4})
SW_CELL = c.n

c = nb.cell('''
x = df["mean smoothness"]             # 전체 569건
w, p = stats.shapiro(x)
print(f"n = {len(x)}, skewness = {stats.skew(x):.2f}, "
      f"W = {w:.3f}, p = {p:.2g}")

fig, ax = plt.subplots(figsize=(4.5, 3.6))
stats.probplot(x, dist="norm", plot=ax)
ax.set_title("mean smoothness (all, n = 569)")
plt.show()
''', title="표본이 클 때")
show(c); save("lab01_bign", c, marks={"W = 0.987": 1, "p = 8.6e-05": 2})
BIGN_CELL = c.n

c = nb.cell('''
rng = np.random.default_rng(2026)      # 난수 시드 고정
xs = df["mean smoothness"].to_numpy()
for n in [20, 50, 100, 200, 400]:
    rejected = 0
    for i in range(1000):              # 표본을 1000번 뽑아 검정
        sample = rng.choice(xs, size=n, replace=False)
        if stats.shapiro(sample).pvalue < 0.05:
            rejected += 1
    print(f"n = {n:3d}: p < 0.05 in {rejected / 1000:.1%}")
''', title="표본크기와 정규성 검정")
show(c); save("lab01_nsim", c, marks={"n =  20: p < 0.05 in 9.1%": 1, "n = 400: p < 0.05 in 87.7%": 2})
NSIM_CELL = c.n

# ---------------------------------------------------------------- d. 다중비교 보정
c = nb.cell('''
mal = df[df["diagnosis"] == "malignant"]
ben = df[df["diagnosis"] == "benign"]
res = stats.ttest_ind(mal[features], ben[features], equal_var=False)
tt = pd.DataFrame({"t": res.statistic, "p": res.pvalue},
                  index=features)
print((tt["p"] < 0.05).sum(), "of", len(tt), "features: p < 0.05")
tt.sort_values("p").head(3)
''', title="30개 변수의 t 검정")
show(c); save("lab01_ttest", c, marks={"26 of 30": 1})
NOTES[c.n] = "## d. 다중비교 보정"
TT_CELL = c.n

c = nb.cell('''
from statsmodels.stats.multitest import multipletests

p5 = [0.004, 0.012, 0.021, 0.038, 0.260]   # 1장 c 절 표의 원래 p
for method in ["bonferroni", "holm", "fdr_bh"]:
    reject, p_adj, _, _ = multipletests(p5, alpha=0.05, method=method)
    print(f"{method:10s}: {p_adj.round(4)}  {reject.sum()} significant")
''', title="1장 c 절의 다섯 p값으로 확인하기")
show(c); save("lab01_mt5", c, marks={"bonferroni: [0.02  0.06  0.105 0.19  1.   ]  1 significant": 1,
                                     "holm      : [0.02  0.048 0.063 0.076 0.26 ]  2 significant": 2,
                                     "0.0475": 3, "4 significant": 4})
MT5_CELL = c.n

c = nb.cell('''
for method in ["bonferroni", "holm", "fdr_bh"]:
    reject, p_adj, _, _ = multipletests(tt["p"], alpha=0.05,
                                        method=method)
    tt["p_" + method] = p_adj          # 보정 p-value를 새 열로
    print(f"{method:10s}: {reject.sum()} significant")
''', title="Bonferroni, Holm, Benjamini-Hochberg")
show(c); save("lab01_mt", c, marks={"bonferroni: 25": 1, "holm      : 25": 2, "fdr_bh    : 26": 3})
MT_CELL = c.n

c = nb.cell('''
show_cols = ["p", "p_bonferroni", "p_holm", "p_fdr_bh"]
tt.sort_values("p")[show_cols].tail(6)   # p가 큰 쪽 6개
''', title="경계에 있는 변수들")
show(c); save("lab01_mt_rows", c, marks={"fractal dimension error": 1, "2.110119e-01": 2, "4.869505e-02": 3})
ROWS_CELL = c.n

c = nb.cell('''
# 양성 357건을 무작위로 둘로 나눔: 두 군은 같은 집단 (귀무가설이 참)
half_a = ben.sample(frac=0.5, random_state=3)
half_b = ben.drop(half_a.index)
p_null = stats.ttest_ind(half_a[features], half_b[features],
                         equal_var=False).pvalue
print(len(half_a), len(half_b))
print("no correction:", (p_null < 0.05).sum())
for method in ["bonferroni", "holm", "fdr_bh"]:
    print(f"{method}:", multipletests(p_null, method=method)[0].sum())
''', title="귀무가설이 참일 때 (한 번)")
show(c); save("lab01_null1", c, marks={"no correction: 2": 1})
NULL1_CELL = c.n

c = nb.cell('''
rows = []
for i in range(1000):                 # 무작위 분할을 1000번 반복
    a = ben.sample(frac=0.5, random_state=i)
    b = ben.drop(a.index)
    p = stats.ttest_ind(a[features], b[features],
                        equal_var=False).pvalue
    row = {"none": (p < 0.05).sum()}
    for method in ["bonferroni", "holm", "fdr_bh"]:
        row[method] = multipletests(p, method=method)[0].sum()
    rows.append(row)
sim = pd.DataFrame(rows)       # 1000행: 반복마다 거짓양성 수
pd.DataFrame({"FWER": (sim > 0).mean(),   # 1개 이상 나온 비율
              "mean_FP": sim.mean()})     # 평균 거짓양성 수
''', title="귀무가설이 참일 때 (1000번 반복)")
show(c); save("lab01_nullsim", c, marks={"0.481": 1, "0.024": 2, "0.029": 3, "1.414": 4})
NULLSIM_CELL = c.n

c = nb.cell('''
# 악성 10건, 양성 10건만 뽑은 작은 연구 (실제 차이는 있음)
small_m = mal.sample(10, random_state=0)
small_b = ben.sample(10, random_state=0)
p_small = stats.ttest_ind(small_m[features], small_b[features],
                          equal_var=False).pvalue
print("no correction:", (p_small < 0.05).sum())
for method in ["bonferroni", "holm", "fdr_bh"]:
    print(f"{method}:", multipletests(p_small, method=method)[0].sum())
''', title="작은 표본에서 세 방법의 차이")
show(c); save("lab01_small", c, marks={"no correction: 21": 1, "bonferroni: 7": 2, "fdr_bh: 18": 3})
SMALL_CELL = c.n

# ---------------------------------------------------------------- e. 과제 (정답 셀)
c = nb.cell('''
v = "fractal dimension error"
r = stats.ttest_ind(mal[v], ben[v], equal_var=False)   # 셀 18과 같은 검정
ci = r.confidence_interval()                           # 평균 차이의 95% CI
diff = mal[v].mean() - ben[v].mean()
smd_v = diff / np.sqrt((mal[v].var() + ben[v].var()) / 2)
print(f"mean: malignant {mal[v].mean():.5f}, benign {ben[v].mean():.5f}")
print(f"difference {diff:.5f} (95% CI {ci.low:.5f} to {ci.high:.5f})")
print(f"difference / benign mean = {diff / ben[v].mean():.1%}")
print(f"SMD {smd_v:.2f}  (mean radius: {smd['mean radius']:.2f})")
''', title="과제 1 정답. 경계에 있는 변수의 차이 크기")
show(c); save("lab01_hw1", c, marks={"benign 0.00364": 1, "difference 0.00043": 2,
                                     "(95% CI 0.00002 to 0.00084)": 3, "11.7%": 4, "SMD 0.17": 5})
HW1_CELL = c.n
NOTES[c.n] = (f"## e. 과제 정답\n\n**과제 1.** 셀 {ROWS_CELL}의 표에서 fractal dimension error 행을 말로 옮기고, "
              "이 행이 말해 주지 않는 것을 두 가지 이상 쓰세요. 두 군의 평균 차이와 그 95% 신뢰구간, SMD를 구해 "
              f"셀 {SMD_CELL}의 mean radius와 비교하세요.")

c = nb.cell('''
tt["p_mw"] = [stats.mannwhitneyu(mal[f], ben[f]).pvalue
              for f in features]               # 30개 변수의 Mann-Whitney 검정
holm_mw = multipletests(tt["p_mw"], method="holm")[0]
print("Welch       : p < 0.05", (tt["p"] < 0.05).sum(),
      "| after Holm", (tt["p_holm"] < 0.05).sum())
print("Mann-Whitney: p < 0.05", (tt["p_mw"] < 0.05).sum(),
      "| after Holm", holm_mw.sum())
two = ["fractal dimension error", "symmetry error"]
for f in two:
    print(f"{f:23}: Welch p = {tt.loc[f, 'p']:.3f}, "
          f"Mann-Whitney p = {tt.loc[f, 'p_mw']:.2g}")
df.groupby("diagnosis", observed=True)[two].agg(
    ["median", "mean", "std", "max"]).T.round(4)
''', title="과제 2 정답. Welch t 검정을 Mann-Whitney 검정으로 바꾸기")
show(c); save("lab01_hw2", c, marks={"after Holm 25": 1, "after Holm 26": 2,
                                     "Welch p = 0.042, Mann-Whitney p = 1.6e-06": 3,
                                     "Welch p = 0.887, Mann-Whitney p = 0.028": 4, "0.0298": 5})
HW2_CELL = c.n
NOTES[c.n] = (f"**과제 2.** 셀 {TT_CELL}의 Welch t 검정을 Mann-Whitney 검정(`stats.mannwhitneyu`)으로 바꿔 30개 변수를 "
              "다시 검정하고, Holm 보정 후 유의한 개수를 비교하세요. fractal dimension error와 symmetry error의 p가 "
              "어떻게 바뀌는지, 왜 그런지 두 군의 중앙값, 평균, SD, 최댓값으로 설명하세요.")

c = nb.cell('''
p_sig = p_null[p_null < 0.05]                  # 동료: 유의한 p만 골라서
print("colleague:", p_sig.round(4), "->",
      multipletests(p_sig, method="holm")[1].round(4))
print("all 30   :", multipletests(p_null, method="holm")[0].sum(),
      "significant")

wrong, right = [], []
for i in range(1000):                          # 셀 23과 같은 분할 1000개
    a = ben.sample(frac=0.5, random_state=i)
    b = ben.drop(a.index)
    p = stats.ttest_ind(a[features], b[features],
                        equal_var=False).pvalue
    sig = p[p < 0.05]
    wrong.append(len(sig) > 0 and multipletests(sig, method="holm")[0].any())
    right.append(multipletests(p, method="holm")[0].any())
print("FWER, only p < 0.05 adjusted:", np.mean(wrong))
print("FWER, all 30 adjusted       :", np.mean(right))
''', title="과제 3 정답. 유의한 p만 골라 보정하면")
show(c); save("lab01_hw3", c, marks={"[0.0215 0.0164]": 1, "[0.0327 0.0327]": 2, "all 30   : 0": 3,
                                     "only p < 0.05 adjusted: 0.357": 4, "all 30 adjusted       : 0.024": 5})
HW3_CELL = c.n
NOTES[c.n] = (f"**과제 3.** 동료가 셀 {NULL1_CELL}의 분할 결과에서 p < 0.05인 변수만 골라 `multipletests`에 넣고 "
              "\"Holm 보정 후에도 두 변수가 유의했다\"고 썼습니다. 동료의 계산을 재현하고, 무엇이 틀렸는지 설명하세요. "
              f"셀 {NULLSIM_CELL}처럼 분할을 1000번 되풀이해 두 방법의 가족단위 오류율을 비교하세요.")

print("cells:", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 1. 자료 탐색과 정규성 검정",
    intro=("사회약학 연구방법 노트의 '실습 1. 자료 탐색과 정규성 검정'에 딸린 노트북입니다. scikit-learn에 들어 있는 "
           "위스콘신 유방암 진단 자료(유방 종괴 세침흡인 검체 569건)로 요약통계와 그래프를 만들고, Table 1의 평균 ± SD, "
           "중앙값(IQR), SMD를 계산하고, Q-Q plot과 Shapiro-Wilk 검정으로 분포를 확인한 뒤, 30개 변수를 한꺼번에 검정할 "
           "때의 다중비교 보정을 실행합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 "
           "실습 1에 있습니다."),
    notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    ns = nb.ns
    df_, tt_ = ns["df"], ns["tt"]
    mal_, ben_ = ns["mal"], ns["ben"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    # 글(content/lab01.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 2: "데이터 불러오기", 3: "진단 변수를 읽기 쉽게 바꾸기",
             4: "설명변수 30개의 구성", 8: "히스토그램", 10: "진단별 상자그림",
             11: "군별 평균과 표준편차", 12: "군별 사분위수", 13: "표준화 평균차(SMD)", 14: "Q-Q plot: 양성 종양 안에서",
             15: "Shapiro-Wilk 검정", 16: "표본이 클 때", 17: "표본크기와 정규성 검정", 18: "30개 변수의 t 검정",
             19: "1장 c 절의 다섯 p값으로 확인하기", 20: "Bonferroni, Holm, Benjamini-Hochberg",
             21: "경계에 있는 변수들", 22: "귀무가설이 참일 때 (한 번)", 23: "귀무가설이 참일 때 (1000번 반복)",
             24: "작은 표본에서 세 방법의 차이", 25: "과제 1 정답. 경계에 있는 변수의 차이 크기",
             26: "과제 2 정답. Welch t 검정을 Mann-Whitney 검정으로 바꾸기", 27: "과제 3 정답. 유의한 p만 골라 보정하면"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    same("number of cells", len(nb.cells), 27)
    # 버전 셀의 설명
    import scipy as _sp, statsmodels as _sm
    same("versions", (pd.__version__, _sp.__version__, _sm.__version__), ("3.0.5", "1.18.1", "0.15.0"))
    # 대응표와 a 절
    same("N, malignant, benign", (len(df_), int(df_["malignant"].sum()), int((df_["malignant"] == 0).sum())),
         (569, 212, 357))
    same("malignant proportion", f"{212 / 569:.1%}", "37.3%")
    ct = pd.crosstab(df_["target"], df_["diagnosis"])
    same("recode crosstab", (int(ct.loc[0, "malignant"]), int(ct.loc[1, "benign"]), int(ct.loc[0, "benign"]),
                             int(ct.loc[1, "malignant"])), (212, 357, 0, 0))
    feats = ns["features"]
    same("feature blocks", (all(f.startswith("mean ") for f in feats[:10]), all(f.endswith(" error") for f in feats[10:20]),
                            all(f.startswith("worst ") for f in feats[20:])), (True, True, True))
    # b 절: describe, 왜도
    ae = df_["area error"]
    same("area error mean-median", f"{ae.mean() - ae.median():.1f}", "15.8")
    same("area error Q1, Q3, min, SD", (f3(ae.quantile(.25)), f3(ae.quantile(.75)), f3(ae.min()), f3(ae.std())),
         ("17.850", "45.190", "6.802", "45.491"))
    same("area error max / median", f"{ae.max() / ae.median():.1f}", "22.1")
    ms = df_["mean smoothness"]
    same("mean smoothness mean, median", (f3(ms.mean()), f3(ms.median())), ("0.096", "0.096"))
    same("skew mean area, mean radius", (f3(stats.skew(df_["mean area"])), f3(stats.skew(df_["mean radius"]))),
         ("1.641", "0.940"))
    same("pandas skew area error", f3(ae.skew()), "5.447")
    same("skew area error (2 d.p.)", f2(stats.skew(ae)), "5.43")
    gm = df_.groupby("diagnosis", observed=True)["mean radius"].median()
    same("median mean radius by group", (f2(gm["benign"]), f2(gm["malignant"])), ("12.20", "17.33"))
    same("whole-sample skew of mean radius", f2(stats.skew(df_["mean radius"])), "0.94")
    m_, s_, q_, smd_ = ns["m"], ns["s"], ns["q"], ns["smd"]
    same("mean radius benign mean, SD (2 d.p.)", (f2(m_.loc["mean radius", "benign"]), f2(s_.loc["mean radius", "benign"])),
         ("12.15", "1.78"))
    same("mean area benign median, IQR, mean",
         (f2(q_.loc["mean area", ("benign", 0.5)]), f2(q_.loc["mean area", ("benign", 0.25)]),
          f2(q_.loc["mean area", ("benign", 0.75)]), f3(m_.loc["mean area", "benign"])),
         ("458.40", "378.20", "551.10", "462.790"))
    same("mean area benign IQR width", f"{q_.loc['mean area', ('benign', 0.75)] - q_.loc['mean area', ('benign', 0.25)]:.1f}",
         "172.9")
    mr = "mean radius"
    same("SMD hand check", (f3(m_.loc[mr, "malignant"]), f3(s_.loc[mr, "malignant"]), f3(s_.loc[mr, "benign"]),
                            f3(m_.loc[mr, "malignant"] - m_.loc[mr, "benign"]),
                            f3(np.sqrt((s_.loc[mr, "malignant"] ** 2 + s_.loc[mr, "benign"] ** 2) / 2)), f2(smd_[mr])),
         ("17.463", "3.204", "1.781", "5.316", "2.592", "2.05"))
    same("smoothness means, SD, SMD", (f"{m_.loc['mean smoothness', 'malignant']:.4f}",
                                       f"{m_.loc['mean smoothness', 'benign']:.4f}",
                                       f3(s_.loc["mean smoothness", "malignant"]), f3(s_.loc["mean smoothness", "benign"]),
                                       f2(smd_["mean smoothness"])),
         ("0.1029", "0.0925", "0.013", "0.013", "0.80"))
    same("Table 1 radius malignant", ns["T1"][("mean radius", "malignant")] if "T1" in ns else T1[("mean radius", "malignant")],
         "17.46 ± 3.20")
    same("skew mean area, area error (2 d.p.)", (f2(stats.skew(df_["mean area"])), f2(stats.skew(ae))), ("1.64", "5.43"))
    # c 절
    bn = ns["benign"]
    same("benign skew mean radius, area error", (f2(stats.skew(bn["mean radius"])), f2(stats.skew(bn["area error"]))),
         ("-0.08", "1.64"))
    same("big-n p", f"{stats.shapiro(df_['mean smoothness']).pvalue:.6f}", "0.000086")
    xm = mal_["mean radius"]
    same("malignant mean radius: SW p < 0.05, skew", (stats.shapiro(xm).pvalue < 0.05, f"{stats.skew(xm):.1f}"), (True, "0.5"))
    rng = np.random.default_rng(2026)
    xa = df_["area error"].to_numpy()
    rate = np.mean([stats.shapiro(rng.choice(xa, size=20, replace=False)).pvalue < 0.05 for _ in range(1000)])
    same("area error n = 20 rejection rate (most)", rate > 0.9, True)
    same("SW p for paper box", (f2(stats.shapiro(bn["mean radius"]).pvalue), stats.shapiro(bn["area error"]).pvalue < 0.001),
         ("0.67", True))
    # d 절
    same("smallest p", f"{tt_['p'].min():.6e}", "1.061454e-96")
    v = "fractal dimension error"
    rk = int(tt_["p"].rank()[v])
    same("fractal dim p, rank", (f"{tt_.loc[v, 'p']:.4f}", f"{tt_.loc[v, 'p']:.6e}", rk), ("0.0422", "4.220238e-02", 26))
    same("Bonferroni raw x30", f2(tt_.loc[v, "p"] * 30), "1.27")
    same("Holm, BH", (f3(tt_.loc[v, "p_holm"]), f"{tt_.loc[v, 'p_fdr_bh']:.4f}"), ("0.211", "0.0487"))
    same("Holm x5, BH x30/26", (f3(tt_.loc[v, "p"] * 5), f"{tt_.loc[v, 'p'] * 30 / 26:.4f}"), ("0.211", "0.0487"))
    same("counts after correction", ((tt_["p"] < 0.05).sum(), (tt_["p_bonferroni"] < 0.05).sum(),
                                     (tt_["p_holm"] < 0.05).sum(), (tt_["p_fdr_bh"] < 0.05).sum()), (26, 25, 25, 26))
    p_null = ns["p_null"]
    same("null split: sizes, corrected counts", (len(ns["half_a"]), len(ns["half_b"]),
                                                 [int(multipletests(p_null, method=mt)[0].sum())
                                                  for mt in ("bonferroni", "holm", "fdr_bh")]), (178, 179, [0, 0, 0]))
    cnt = []
    for sd in range(1, 12):
        a = ben_.sample(frac=0.5, random_state=sd)
        cnt.append(int((stats.ttest_ind(a[feats], ben_.drop(a.index)[feats], equal_var=False).pvalue < 0.05).sum()))
    same("random_state 1-11 range", (min(cnt), max(cnt)), (0, 5))
    same("independent FWER for 30", f"{1 - 0.95 ** 30:.3f}", "0.785")
    sim = ns["sim"]
    same("null FWER by method", [f3((sim[c_] > 0).mean()) for c_ in ("none", "bonferroni", "holm", "fdr_bh")],
         ["0.481", "0.024", "0.024", "0.029"])
    same("small sample: holm", int(multipletests(ns["p_small"], method="holm")[0].sum()), 7)
    same("0.05/30", f"{0.05 / 30:.4f}", "0.0017")
    # 비교표(d 절 끝)
    same("comparison table: real data", [(tt_["p"] < 0.05).sum()] + [(tt_["p_" + mt] < 0.05).sum()
                                         for mt in ("bonferroni", "holm", "fdr_bh")], [26, 25, 25, 26])
    same("comparison table: small", [int((ns["p_small"] < 0.05).sum())] +
         [int(multipletests(ns["p_small"], method=mt)[0].sum()) for mt in ("bonferroni", "holm", "fdr_bh")], [21, 7, 7, 18])
    # 과제 1
    r = stats.ttest_ind(mal_[v], ben_[v], equal_var=False)
    same("hw1 malignant mean", f"{mal_[v].mean():.5f}", "0.00406")
    # 과제 2
    same("hw2 MW counts", ((tt_["p_mw"] < 0.05).sum(), int(multipletests(tt_["p_mw"], method="holm")[0].sum())), (27, 26))
    same("hw2 MW p fractal dim", f"{tt_.loc[v, 'p_mw']:.1e}", "1.6e-06")
    same("hw2 symmetry error Welch p, MW p", (f2(tt_.loc["symmetry error", "p"]), f3(tt_.loc["symmetry error", "p_mw"])),
         ("0.89", "0.028"))
    agg = df_.groupby("diagnosis", observed=True)[[v, "symmetry error"]].agg(["median", "mean", "std", "max"])
    same("hw2 fractal dim medians, SD, max",
         [f"{agg.loc[g_, (v, st)]:.4f}" for g_ in ("benign", "malignant") for st in ("median", "std", "max")],
         ["0.0028", "0.0029", "0.0298", "0.0037", "0.0020", "0.0128"])
    same("hw2 symmetry medians, means", [f"{agg.loc[g_, ('symmetry error', st)]:.4f}"
                                        for g_ in ("benign", "malignant") for st in ("median", "mean", "std")],
         ["0.0191", "0.0206", "0.0070", "0.0177", "0.0205", "0.0101"])
    same("hw2 benign max / median (fractal)", f"{agg.loc['benign', (v, 'max')] / agg.loc['benign', (v, 'median')]:.0f}", "11")
    n12 = len(mal_) * len(ben_)
    same("hw2 MW direction: fractal higher in malignant, symmetry lower",
         (stats.mannwhitneyu(mal_[v], ben_[v]).statistic / n12 > 0.5,
          stats.mannwhitneyu(mal_["symmetry error"], ben_["symmetry error"]).statistic / n12 < 0.5), (True, True))
    same("hw2 Welch p fractal dim (3 d.p.)", f3(tt_.loc[v, "p"]), "0.042")
    same("hw1 Holm, BH rounded for the sentence", (f2(tt_.loc[v, "p_holm"]), f3(tt_.loc[v, "p_fdr_bh"])), ("0.21", "0.049"))
    same("hw1 SMD ratio to mean radius", f"{smd_['mean radius'] / ns['smd_v']:.0f}", "12")
    same("hw3 FWER no correction (cell 23) for the comparison", f3((sim["none"] > 0).mean()), "0.481")
    same("hw3 Holm in the colleague's calculation (smaller p x 2)", f"{sorted(p_null[p_null < 0.05])[0] * 2:.4f}", "0.0327")
    # 과제 3
    same("hw3 colleague's variables", list(np.array(feats)[p_null < 0.05]), ["texture error", "symmetry error"])
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab01.html passed")
