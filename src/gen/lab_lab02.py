"""실습 2 (lab02) — 두 군의 크기 비교: birthwt, 흡연 여부별 출생체중.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab02.py
All outputs are produced by actually running the cells (labkit).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook

nb = Notebook("lab02")
S = nb.save_fragment


def H(c, marks=None):
    """nb.html + marks placed once in the whole output block (text or table).

    labkit's own `marks` requires every mark to appear in BOTH the printed text and the
    DataFrame table of a cell; this helper marks the first occurrence anywhere in the
    output, skipping embedded images."""
    import html as _h
    s = nb.html(c)
    if not marks:
        return s
    i = s.index('<div class="cell-out">')
    j = s.find("<img", i)
    j = len(s) if j < 0 else j
    head, body, tail = s[:i], s[i:j], s[j:]
    for sub, n in marks.items():
        e = _h.escape(sub)
        k = body.find(e)
        if k < 0:
            if os.environ.get("LAB_DRY"):
                print("  [dry] mark not found:", repr(sub))
                continue
            raise ValueError(f"mark text not found in output: {sub!r}")
        k += len(e)
        body = body[:k] + f'<span class="mk">{n}</span>' + body[k:]
    return head + body + tail

# ------------------------------------------------------------------ 가. 데이터 준비
c = nb.cell('''
import pandas as pd
import numpy as np
from scipy import stats

url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/MASS/birthwt.csv")
df = pd.read_csv(url)
print(df.shape)
df.head()
''', title="데이터 불러오기")
S("lab02_load", H(c, marks={"(189, 11)": 1, "2523": 2}))

c = nb.cell('''
df["smoke_f"] = pd.Categorical(
    df["smoke"].map({0: "Nonsmoker", 1: "Smoker"}),
    categories=["Nonsmoker", "Smoker"])
df["smoke_f"].value_counts(sort=False)
''', title="흡연 여부에 이름표 붙이기")
S("lab02_recode", H(c, marks={"115": 1, "74": 2}))

c = nb.cell('''
summary = (df.groupby("smoke_f", observed=True)["bwt"]
             .agg(["count", "mean", "std", "median"]))
summary.round(1)
''', title="군별 요약통계")
S("lab02_summary", H(c, marks={"3055.7": 1, "752.7": 2, "2771.9": 3}))

c = nb.cell('''
df["BWT"].mean()
''', title="열 이름을 잘못 쓰면", expect_error=True)
S("lab02_keyerror", H(c))

c = nb.cell('''
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(data=df, x="smoke_f", y="bwt", ax=ax,
            color="0.85", width=0.5, showfliers=False)
np.random.seed(0)        # 점 흩뿌리기(jitter) 위치 고정
sns.stripplot(data=df, x="smoke_f", y="bwt", ax=ax,
              color="black", alpha=0.4, jitter=0.2)
ax.axhline(2500, ls="--", color="gray")
ax.set_xlabel("Smoking during pregnancy")
ax.set_ylabel("Birth weight (g)")
plt.show()
''', title="상자그림과 개별 점")
S("lab02_plot", H(c))

# ------------------------------------------------------------------ 나. 독립표본 t 검정
c = nb.cell('''
smoker = df.loc[df["smoke"] == 1, "bwt"]
nonsmoker = df.loc[df["smoke"] == 0, "bwt"]

student = stats.ttest_ind(smoker, nonsmoker, equal_var=True)
welch = stats.ttest_ind(smoker, nonsmoker, equal_var=False)
print(welch)
print(f"Student: t = {student.statistic:.3f}, "
      f"p = {student.pvalue:.4f}, df = {student.df:.0f}")
print(f"Welch:   t = {welch.statistic:.3f}, "
      f"p = {welch.pvalue:.4f}, df = {welch.df:.1f}")
''', title="Student t 검정과 Welch t 검정")
S("lab02_ttest", H(c, marks={"TtestResult": 1, "df = 187": 2, "Welch:   t = -2.730": 3,
                              "p = 0.0070": 4, "df = 170.1": 5}))

c = nb.cell('''
import scipy
print("scipy", scipy.__version__)

diff = smoker.mean() - nonsmoker.mean()
ci = welch.confidence_interval(confidence_level=0.95)
print(f"평균 차이 = {diff:.1f} g")
print(f"95% CI = {ci.low:.1f} ~ {ci.high:.1f} g")
''', title="평균 차이와 95% 신뢰구간")
S("lab02_ci", H(c, marks={"scipy 1.17.1": 1, "-283.8 g": 2, "-489.0 ~ -78.6 g": 3}))

c = nb.cell('''
n1, n2 = len(smoker), len(nonsmoker)
v1, v2 = smoker.var(), nonsmoker.var()   # 분산 (n - 1로 나눔)
se = np.sqrt(v1 / n1 + v2 / n2)          # 차이의 표준오차
df_w = (v1/n1 + v2/n2)**2 / (
    (v1/n1)**2 / (n1 - 1) + (v2/n2)**2 / (n2 - 1))
t_crit = stats.t.ppf(0.975, df_w)        # 양측 5% 임계값
print(f"SE = {se:.2f}, t = {diff / se:.3f}, df = {df_w:.2f}")
print(f"CI = {diff - t_crit*se:.1f} ~ {diff + t_crit*se:.1f}")
''', title="같은 값을 공식으로 계산하기")
S("lab02_manual", H(c, marks={"SE = 103.95": 1, "df = 170.10": 2, "CI = -489.0 ~ -78.6": 3}))

c = nb.cell('''
sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
d = diff / sp
print(f"합동 SD = {sp:.1f} g")
print(f"Cohen's d = {d:.2f}")
''', title="Cohen의 d")
S("lab02_cohen", H(c, marks={"717.8 g": 1, "-0.40": 2}))

c = nb.cell('''
lev_med = stats.levene(smoker, nonsmoker)   # 중앙값 기준 (기본값)
lev_mean = stats.levene(smoker, nonsmoker, center="mean")
print(lev_med)
print(f"median: p = {lev_med.pvalue:.3f}")
print(f"mean:   p = {lev_mean.pvalue:.3f}")
''', title="Levene 검정")
S("lab02_levene", H(c, marks={"median: p = 0.229": 1, "mean:   p = 0.211": 2}))

# ------------------------------------------------------------------ 다. Mann-Whitney
c = nb.cell('''
q = (df.groupby("smoke_f", observed=True)["bwt"]
       .quantile([0.25, 0.5, 0.75])
       .unstack())
q
''', title="중앙값과 사분위수")
S("lab02_quant", H(c, marks={"3100.0": 1, "2775.5": 2}))

c = nb.cell('''
mw = stats.mannwhitneyu(smoker, nonsmoker,
                        alternative="two-sided")
print(mw)
n1n2 = n1 * n2
print("U(smoker) =", mw.statistic)
print("U(nonsmoker) =", n1n2 - mw.statistic)
print(f"p = {mw.pvalue:.4f}")
''', title="Mann-Whitney 검정")
S("lab02_mw", H(c, marks={"U(smoker) = 3260.5": 1, "U(nonsmoker) = 5249.5": 2,
                           "p = 0.0068": 3}))

c = nb.cell('''
s = smoker.to_numpy()[:, None]      # 74행 1열
ns = nonsmoker.to_numpy()[None, :]  # 1행 115열
greater = (s > ns).sum()            # 흡연군 쪽이 더 무거운 쌍
ties = (s == ns).sum()              # 체중이 같은 쌍
print(greater, ties, greater + 0.5 * ties)
''', title="U를 직접 세어 보기")
S("lab02_count", H(c, marks={"3260.5": 1}))

c = nb.cell('''
ps = mw.statistic / n1n2
print(f"P(smoker > nonsmoker) = {ps:.3f}")

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(df["smoke"], df["bwt"])
print(f"AUC = {auc:.3f}")
''', title="우월 확률과 AUC")
S("lab02_auc", H(c, marks={"= 0.383": 1, "AUC = 0.383": 2}))

c = nb.cell('''
diffs = np.subtract.outer(smoker.to_numpy(),
                          nonsmoker.to_numpy())
print(diffs.shape)
hl = np.median(diffs)

d_sorted = np.sort(diffs.ravel())
N = n1 + n2
k = round(n1n2 / 2 - 1.96 * np.sqrt(n1n2 * (N + 1) / 12))
lo, hi = d_sorted[k - 1], d_sorted[n1n2 - k]
print(f"HL = {hl:.1f} g, 95% CI {lo} ~ {hi}")
''', title="Hodges-Lehmann 추정치")
S("lab02_hl", H(c, marks={"(74, 115)": 1, "HL = -307.0 g": 2, "95% CI -512 ~ -85": 3}))

print("lab02: cells =", len(nb.cells))
