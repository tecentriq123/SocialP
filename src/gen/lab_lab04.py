"""실습 4 (lab04) — 세 군 이상의 크기 비교: birthwt, 인종별 출생체중과 어머니 체중 삼분위의 추세.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab04.py
All outputs are produced by actually running the cells (labkit).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook

nb = Notebook("lab04")
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
df["race_f"] = pd.Categorical(
    df["race"].map({1: "White", 2: "Black", 3: "Other"}),
    categories=["White", "Black", "Other"])
df["race_f"].value_counts(sort=False)
''', title="데이터 불러오기와 군 이름표")
S("lab04_load", H(c, marks={"96": 1, "26": 2, "67": 3}))

c = nb.cell('''
g = df.groupby("race_f", observed=True)["bwt"]
g.agg(["count", "mean", "std", "median"]).round(1)
''', title="군별 요약통계")
S("lab04_summary", H(c, marks={"3102.7": 1, "2719.7": 2, "638.7": 3}))

c = nb.cell('''
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="race_f", y="bwt", ax=ax,
            color="0.85", width=0.5, showfliers=False)
np.random.seed(0)        # 점 흩뿌리기(jitter) 위치 고정
sns.stripplot(data=df, x="race_f", y="bwt", ax=ax,
              color="black", alpha=0.4, jitter=0.2)
ax.set_xlabel("Maternal race")
ax.set_ylabel("Birth weight (g)")
plt.show()
''', title="상자그림과 개별 점")
S("lab04_plot", H(c))

c = nb.cell('''
pd.crosstab(df["race_f"], df["smoke"], normalize="index").round(2)
''', title="군별 흡연 비율")
S("lab04_smoke", H(c, marks={"0.54": 1, "0.18": 2}))

# ------------------------------------------------------------------ 나. 분산분석과 사후분석
c = nb.cell('''
white = df.loc[df["race_f"] == "White", "bwt"]
black = df.loc[df["race_f"] == "Black", "bwt"]
other = df.loc[df["race_f"] == "Other", "bwt"]

res = stats.f_oneway(white, black, other)
print(res)
print(f"F = {res.statistic:.3f}, p = {res.pvalue:.4f}")
''', title="일원배치 분산분석 (scipy)")
S("lab04_foneway", H(c, marks={"F = 4.913": 1, "p = 0.0083": 2}))

c = nb.cell('''
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols("bwt ~ race_f", data=df).fit()
table = sm.stats.anova_lm(model)
table
''', title="분산분석표 (statsmodels)")
S("lab04_anova", H(c, marks={"2.0": 1, "5.015725e+06": 2, "2.507863e+06": 3, "4.912513": 4,
                             "0.008336": 5, "186.0": 6, "5.105050e+05": 7}))

c = nb.cell('''
ss_b = table.loc["race_f", "sum_sq"]      # 군간 제곱합
ss_w = table.loc["Residual", "sum_sq"]    # 군내 제곱합
ms_w = table.loc["Residual", "mean_sq"]   # 군내 평균제곱
eta2 = ss_b / (ss_b + ss_w)
omega2 = (ss_b - 2 * ms_w) / (ss_b + ss_w + ms_w)
print(f"SS_B = {ss_b:,.0f}, SS_W = {ss_w:,.0f}")
print(f"eta^2 = {eta2:.3f}, omega^2 = {omega2:.3f}")
print(f"pooled SD = {np.sqrt(ms_w):.1f} g")
''', title="효과크기 η²와 ω²")
S("lab04_eta", H(c, marks={"SS_B = 5,015,725": 1, "eta^2 = 0.050": 2, "omega^2 = 0.040": 3,
                            "pooled SD = 714.5 g": 4}))

c = nb.cell('''
from statsmodels.stats.oneway import anova_oneway

lev = stats.levene(white, black, other)       # 중앙값 기준
print(f"Levene: p = {lev.pvalue:.3f}")
welch = anova_oneway(df["bwt"], df["race_f"], use_var="unequal")
print(f"Welch F = {welch.statistic:.3f}, "
      f"df = ({welch.df_num:.0f}, {welch.df_denom:.1f}), "
      f"p = {welch.pvalue:.4f}")
''', title="등분산 확인과 Welch 분산분석")
S("lab04_welch", H(c, marks={"Levene: p = 0.627": 1, "Welch F = 5.060": 2, "72.4)": 3, "p = 0.0088": 4}))

c = nb.cell('''
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(df["bwt"], df["race_f"], alpha=0.05)
print(tukey)
''', title="Tukey HSD 사후분석")
S("lab04_tukey", H(c, marks={"FWER=0.05": 1, "meandiff": 2, "0.8624": 3,
                              " Black  White 383.0264 0.0428": 4,
                              " Other  White 297.4352  0.026": 5}))

# ------------------------------------------------------------------ 다. Kruskal-Wallis
c = nb.cell('''
kw = stats.kruskal(white, black, other)
print(f"H = {kw.statistic:.3f}, p = {kw.pvalue:.4f}")

df["rank"] = df["bwt"].rank()      # 189명 전체에서 매긴 순위
print(df.groupby("race_f", observed=True)["rank"].mean().round(1))
''', title="Kruskal-Wallis 검정")
S("lab04_kw", H(c, marks={"H = 8.520": 1, "p = 0.0141": 2, "106.1": 3, "77.5": 4}))

c = nb.cell('''
from statsmodels.stats.multitest import multipletests

groups = {"White": white, "Black": black, "Other": other}
pairs = [("White", "Black"), ("White", "Other"),
         ("Black", "Other")]
p_mw, sup = [], []
for a, b in pairs:
    r = stats.mannwhitneyu(groups[b], groups[a])
    p_mw.append(r.pvalue)
    sup.append(r.statistic / (len(groups[a]) * len(groups[b])))
p_mw_holm = multipletests(p_mw, method="holm")[1]
print("P(b > a):", np.round(sup, 3))
print("p:", np.round(p_mw, 4), " Holm:", np.round(p_mw_holm, 4))
''', title="쌍별 Mann-Whitney 검정과 Holm 보정")
S("lab04_pairmw", H(c, marks={"P(b > a): [0.346 0.394 0.541]": 1, "[0.0165 0.021  0.5461]": 2,
                                "[0.0495 0.0495 0.5461]": 3}))

c = nb.cell('''
N = len(df)
t = df["bwt"].value_counts()               # 동점 묶음의 크기
tie = (t**3 - t).sum() / (12 * (N - 1))    # 동점 보정 항
mean_rank = df.groupby("race_f", observed=True)["rank"].mean()
n = df["race_f"].value_counts()

p_dunn = []
for a, b in pairs:
    se = np.sqrt((N * (N + 1) / 12 - tie) * (1 / n[a] + 1 / n[b]))
    z = (mean_rank[a] - mean_rank[b]) / se
    p_dunn.append(2 * stats.norm.sf(abs(z)))
p_dunn_holm = multipletests(p_dunn, method="holm")[1]
print(np.round(p_dunn, 4), np.round(p_dunn_holm, 4))
''', title="Dunn 검정 직접 계산하기")
S("lab04_dunn", H(c, marks={"[0.0179 0.0197 0.5096]": 1, "[0.0537 0.0537 0.5096]": 2}))

# ------------------------------------------------------------------ 라. Jonckheere-Terpstra
c = nb.cell('''
# 사전 가설: 어머니 체중(lwt)이 무거울수록 출생체중이 크다
df["lwt3"] = pd.qcut(df["lwt"], q=3, labels=["T1", "T2", "T3"])
df.groupby("lwt3", observed=True).agg(
    n=("bwt", "size"), lwt_min=("lwt", "min"),
    lwt_max=("lwt", "max"), bwt_median=("bwt", "median"))
''', title="어머니 체중 삼분위 만들기")
S("lab04_tertile", H(c, marks={"115": 1, "2722.0": 2, "3288.5": 3}))

c = nb.cell('''
def jt_J(groups):
    """J = 모든 (앞 군, 뒤 군) 쌍에서 뒤 군 값이 더 큰 짝의 수"""
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = np.asarray(groups[i])[:, None]   # 앞 군 (세로)
            b = np.asarray(groups[j])[None, :]   # 뒤 군 (가로)
            J += (b > a).sum() + 0.5 * (b == a).sum()
    return J

order = ["T1", "T2", "T3"]
tert = [df.loc[df["lwt3"] == k, "bwt"] for k in order]
print("J =", jt_J(tert))
''', title="J 통계량 계산 함수")
S("lab04_jtJ", H(c, marks={"J = 7356.0": 1}))

c = nb.cell('''
U12 = stats.mannwhitneyu(tert[1], tert[0]).statistic   # T2 > T1
U13 = stats.mannwhitneyu(tert[2], tert[0]).statistic   # T3 > T1
U23 = stats.mannwhitneyu(tert[2], tert[1]).statistic   # T3 > T2
print(U12, U13, U23, U12 + U13 + U23)
''', title="J는 Mann-Whitney U의 합")
S("lab04_jtU", H(c, marks={"7356.0": 1}))

c = nb.cell('''
def jt_test(groups):
    """Jonckheere-Terpstra 검정: 동점 보정 정규근사, 양측 p"""
    J = jt_J(groups)
    n = np.array([len(x) for x in groups])      # 군별 인원
    N = n.sum()
    allv = np.concatenate([np.asarray(x) for x in groups])
    t = np.unique(allv, return_counts=True)[1]  # 동점 묶음 크기
    EJ = (N**2 - (n**2).sum()) / 4               # 귀무가설 아래 평균
    v1 = (N*(N-1)*(2*N+5) - (n*(n-1)*(2*n+5)).sum()
          - (t*(t-1)*(2*t+5)).sum()) / 72
    v2 = ((n*(n-1)*(n-2)).sum() * (t*(t-1)*(t-2)).sum()
          / (36 * N * (N-1) * (N-2)))
    v3 = (n*(n-1)).sum() * (t*(t-1)).sum() / (8 * N * (N-1))
    z = (J - EJ) / np.sqrt(v1 + v2 + v3)
    return J, EJ, z, 2 * stats.norm.sf(abs(z))

J, EJ, z, p = jt_test(tert)
print(f"J = {J:.0f}, E(J) = {EJ:.0f}, z = {z:.2f}, p = {p:.4f}")
''', title="정규근사로 p-value 구하기")
S("lab04_jttest", H(c, marks={"J = 7356": 1, "E(J) = 5947": 2, "z = 3.44": 3, "p = 0.0006": 4}))

c = nb.cell('''
import statsmodels
print("statsmodels", statsmodels.__version__)
from statsmodels.stats.nonparametric import jonckheere_terpstra

res_sm = jonckheere_terpstra(tert, alternative="two-sided")
print(f"J = {res_sm.statistic:.0f}, z = {res_sm.zstat:.2f}, "
      f"p = {res_sm.pvalue:.4f}")
''', title="statsmodels 0.15의 함수와 비교")
S("lab04_jtsm", H(c, marks={"statsmodels 0.15.0": 1, "J = 7356, z = 3.44, p = 0.0006": 2}))

c = nb.cell('''
rng = np.random.default_rng(2026)      # 난수 시드 고정
y = df["bwt"].to_numpy()
lab = df["lwt3"].to_numpy()
B = 10000
J_perm = np.empty(B)
for r in range(B):
    s = rng.permutation(lab)           # 군 이름표를 무작위로 섞음
    J_perm[r] = jt_J([y[s == k] for k in order])
extreme = np.abs(J_perm - EJ) >= abs(J - EJ)   # 관측값만큼 극단적
print("extreme:", extreme.sum(), "of", B)
print(f"permutation p = {(extreme.sum() + 1) / (B + 1):.4f}")
''', title="순열검정으로 확인하기")
S("lab04_jtperm", H(c, marks={"extreme: 4 of 10000": 1, "permutation p = 0.0005": 2}))

c = nb.cell('''
kw3 = stats.kruskal(*tert)
print(f"H = {kw3.statistic:.2f}, p = {kw3.pvalue:.4f}")
''', title="같은 삼분위에 Kruskal-Wallis 검정")
S("lab04_jtkw", H(c, marks={"H = 12.37": 1, "p = 0.0021": 2}))

c = nb.cell('''
df["ftv3"] = pd.cut(df["ftv"], bins=[-1, 0, 1, 99],
                    labels=["0", "1", "2+"])
visits = [df.loc[df["ftv3"] == k, "bwt"] for k in ["0", "1", "2+"]]
g3 = df.groupby("ftv3", observed=True)["bwt"]
print(g3.agg(["count", "median"]))
J, EJ, z, p = jt_test(visits)
print(f"JT: z = {z:.2f}, p = {p:.3f}")
print(f"KW: p = {stats.kruskal(*visits).pvalue:.3f}")
''', title="순서대로 커지지 않는 경우")
S("lab04_ftv", H(c, marks={"3033.0": 1, "JT: z = 1.05, p = 0.292": 2, "KW: p = 0.269": 3}))

print("lab04: cells =", len(nb.cells))
