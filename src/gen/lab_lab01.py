"""실습 1 · 자료 탐색과 정규성 검정 — run with: source /home/claude/pylibs/env.sh && python3 gen/lab_lab01.py

All cells are executed for real through labkit; the Table 1 paper-box table is generated from the same
results (figs/lab01_t1.html) so that the paper box and the cell outputs cannot disagree.
"""
import html, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labkit import Notebook  # noqa: E402

nb = Notebook("lab01")


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


# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_breast_cancer

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

# ---------------------------------------------------------------- 나. 요약통계와 그래프
c = nb.cell('''
cols = ["mean radius", "mean smoothness", "mean area", "area error"]
df[cols].describe().round(3)
''', title="describe()")
show(c); save("lab01_describe", c, marks={"40.337": 1, "24.530": 2, "542.200": 3})

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

# ---------------------------------------------------------------- 다. Q-Q plot과 정규성 검정
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

c = nb.cell('''
for col in ["mean radius", "area error"]:
    x = benign[col]
    w, p = stats.shapiro(x)       # 검정통계량 W와 p-value
    print(f"{col}: n = {len(x)}, skewness = {stats.skew(x):.2f}, "
          f"W = {w:.3f}, p = {p:.3g}")
''', title="Shapiro-Wilk 검정")
show(c); save("lab01_shapiro", c, marks={"W = 0.997": 1, "p = 0.668": 2, "W = 0.896": 3, "p = 6.76e-15": 4})

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

# ---------------------------------------------------------------- 라. 다중비교 보정
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

c = nb.cell('''
from statsmodels.stats.multitest import multipletests

for method in ["bonferroni", "holm", "fdr_bh"]:
    reject, p_adj, _, _ = multipletests(tt["p"], alpha=0.05,
                                        method=method)
    tt["p_" + method] = p_adj          # 보정 p-value를 새 열로
    print(f"{method:10s}: {reject.sum()} significant")
''', title="Bonferroni, Holm, Benjamini-Hochberg")
show(c); save("lab01_mt", c, marks={"bonferroni: 25": 1, "holm      : 25": 2, "fdr_bh    : 26": 3})

c = nb.cell('''
show_cols = ["p", "p_bonferroni", "p_holm", "p_fdr_bh"]
tt.sort_values("p")[show_cols].tail(6)   # p가 큰 쪽 6개
''', title="경계에 있는 변수들")
show(c); save("lab01_mt_rows", c, marks={"fractal dimension error": 1})

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

print("cells:", len(nb.cells))
