"""실습 4 (lab04) — 세 군 이상의 크기 비교: birthwt, 인종별 출생체중과 어머니 체중 삼분위의 추세.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab04.py
      then  python3 tools/nbcheck.py lab04   (runs the notebook alone in a fresh process and compares outputs)
All outputs are produced by actually running the cells (labkit).
Writes figs/lab04_*.html (insert in content/lab04.html with <!--FIG:lab04_xxx-->),
pub/notebooks/lab04.ipynb (download link at the top of content/lab04.html) and checks/lab04.expected.json.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab04.html that are not
marked in a cell output (대응표, the '직접 바꿔 보기' tips, the 과제 problems and answers).
(2026-10-14: version cell 1 split from the old load cell, residual normality cell added before the Welch cell,
과제 1–3 and the notebook added; old cells 1–7 are now 2–8 and old cells 8–20 are now 10–22.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook

nb = Notebook("lab04")
S = nb.save_fragment
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


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

# ------------------------------------------------------------------ a. 데이터 준비
c = nb.cell('''
import pandas as pd
import numpy as np
from scipy import stats
import scipy, statsmodels, matplotlib, seaborn

print("pandas     ", pd.__version__)
print("numpy      ", np.__version__)
print("scipy      ", scipy.__version__)
print("statsmodels", statsmodels.__version__)
print("matplotlib ", matplotlib.__version__)
print("seaborn    ", seaborn.__version__)
''', title="패키지 불러오기와 버전 확인")
S("lab04_versions", H(c, marks={"scipy       1.18.1": 1, "statsmodels 0.15.0": 2}))
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 birthwt 자료를 읽습니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
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
SMOKE_CELL = c.n

# ------------------------------------------------------------------ b. 분산분석과 사후분석
c = nb.cell('''
white = df.loc[df["race_f"] == "White", "bwt"]
black = df.loc[df["race_f"] == "Black", "bwt"]
other = df.loc[df["race_f"] == "Other", "bwt"]

res = stats.f_oneway(white, black, other)
print(res)
print(f"F = {res.statistic:.3f}, p = {res.pvalue:.4f}")
''', title="일원배치 분산분석 (scipy)")
S("lab04_foneway", H(c, marks={"F = 4.913": 1, "p = 0.0083": 2}))
NOTES[c.n] = "## b. 분산분석과 사후분석"

c = nb.cell('''
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols("bwt ~ race_f", data=df).fit()
table = sm.stats.anova_lm(model)
table
''', title="분산분석표 (statsmodels)")
S("lab04_anova", H(c, marks={"2.0": 1, "5.015725e+06": 2, "2.507863e+06": 3, "4.912513": 4,
                             "0.008336": 5, "186.0": 6, "5.105050e+05": 7}))
ANOVA_CELL = c.n

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
resid = model.resid                 # 각 아기의 값 - 자기 군 평균
sw = stats.shapiro(resid)
print(f"Shapiro-Wilk W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}")

fig, ax = plt.subplots(figsize=(4.2, 3.6))
stats.probplot(resid, dist="norm", plot=ax)   # 잔차 189개의 정규 Q-Q plot
ax.set_title("Normal Q-Q plot of residuals")
plt.show()
''', title="잔차의 정규성")
S("lab04_resid", H(c, marks={"W = 0.989": 1, "p = 0.174": 2}))

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
TUKEY_CELL = c.n

# ------------------------------------------------------------------ c. Kruskal-Wallis
c = nb.cell('''
kw = stats.kruskal(white, black, other)
print(f"H = {kw.statistic:.3f}, p = {kw.pvalue:.4f}")

df["rank"] = df["bwt"].rank()      # 189명 전체에서 매긴 순위
print(df.groupby("race_f", observed=True)["rank"].mean().round(1))
''', title="Kruskal-Wallis 검정")
S("lab04_kw", H(c, marks={"H = 8.520": 1, "p = 0.0141": 2, "106.1": 3, "77.5": 4}))
NOTES[c.n] = "## c. Kruskal-Wallis 검정"
KW_CELL = c.n

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
DUNN_CELL = c.n

# ------------------------------------------------------------------ d. Jonckheere-Terpstra
c = nb.cell('''
# 사전 가설: 어머니 체중(lwt)이 무거울수록 출생체중이 크다
df["lwt3"] = pd.qcut(df["lwt"], q=3, labels=["T1", "T2", "T3"])
df.groupby("lwt3", observed=True).agg(
    n=("bwt", "size"), lwt_min=("lwt", "min"),
    lwt_max=("lwt", "max"), bwt_median=("bwt", "median"))
''', title="어머니 체중 삼분위 만들기")
S("lab04_tertile", H(c, marks={"115": 1, "2722.0": 2, "3288.5": 3}))
NOTES[c.n] = "## d. Jonckheere-Terpstra 검정"

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
JT_CELL = c.n

c = nb.cell('''
import statsmodels
print("statsmodels", statsmodels.__version__)
from statsmodels.stats.nonparametric import jonckheere_terpstra

res_sm = jonckheere_terpstra(tert, alternative="two-sided")
print(f"J = {res_sm.statistic:.0f}, z = {res_sm.zstat:.2f}, "
      f"p = {res_sm.pvalue:.4f}")
''', title="statsmodels 0.15의 함수와 비교")
S("lab04_jtsm", H(c, marks={"statsmodels 0.15.0": 1, "J = 7356, z = 3.44, p = 0.0006": 2}))
SM_JT_CELL = c.n
NOTES[c.n] = ("statsmodels가 0.15보다 낮으면 이 셀에서 `ImportError`가 납니다. 그때는 이 셀을 건너뛰고 다음 셀부터 "
              "실행하세요. 다음 셀들은 직접 만든 `jt_test`를 씁니다.")

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
PERM_CELL = c.n

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

# ------------------------------------------------------------------ e. 과제 (정답 셀)
c = nb.cell('''
m_smoke = smf.ols("bwt ~ race_f + smoke", data=df).fit()   # 흡연을 함께 넣음
print("adjusted 95% CI, Other:")
print(m_smoke.conf_int().loc["race_f[T.Other]"].round(1).tolist())
pd.DataFrame({"crude": model.params,
              "adjusted": m_smoke.params}).round(1)
''', title="과제 1 정답. Tukey 결과의 한 행 읽기")
S("lab04_hw1", H(c, marks={"[-682.7, -223.1]": 1, "-297.4": 2, "-452.9": 3, "-428.7": 4}))
NOTES[c.n] = (f"## 과제 정답\n\n**과제 1.** 셀 {TUKEY_CELL}의 Tukey 결과에서 `Other  White 297.4352  0.026` 행을 "
              "한 문장으로 옮기고, 이 결과가 말해 주지 않는 것을 두 가지 이상 쓰세요. "
              f"셀 {ANOVA_CELL}의 모형에 흡연(smoke)을 더해 White와 Other의 차이가 어떻게 바뀌는지 확인하세요.")

c = nb.cell('''
dun = stats.dunnett(black, other, control=white, rng=2026)   # 대조군 = White
ci_d = dun.confidence_interval()
for k, name in enumerate(["Black", "Other"]):      # 넣은 순서대로
    print(f"{name} - White: p = {dun.pvalue[k]:.3f}")
    print(f"  95% CI {ci_d.low[k]:.1f} ~ {ci_d.high[k]:.1f}")
''', title="과제 2 정답. Tukey 대신 Dunnett 방법으로")
S("lab04_hw2", H(c, marks={"p = 0.032": 1, "-737.9": 2, "-28.1": 3, "p = 0.019": 4}))
NOTES[c.n] = (f"**과제 2.** 연구 질문이 처음부터 \"White 산모와 비교해 나머지 두 군의 출생체중이 다른가\"였다고 하고, "
              f"셀 {TUKEY_CELL}의 Tukey 대신 Dunnett 방법(`stats.dunnett`, 대조군 White)으로 다시 비교하세요. "
              "Tukey와 비교해 P와 신뢰구간이 어떻게 바뀌는지, 왜 그런지, 그리고 Dunnett 결과로 말할 수 없는 것이 "
              "무엇인지 설명하세요.")

c = nb.cell('''
med = df.groupby("race_f", observed=True)["bwt"].median()
order_seen = med.sort_values().index.tolist()     # 자료를 본 뒤 정한 순서
print("medians:", order_seen)
print("means  :", g.mean().sort_values().index.tolist())
J_c, EJ_c, z_c, p_c = jt_test(
    [df.loc[df["race_f"] == k, "bwt"] for k in order_seen])
print(f"z = {z_c:.2f}, P for trend = {p_c:.3f}")

rng3 = np.random.default_rng(2026)
y3, race_lab = df["bwt"].to_numpy(), df["race_f"].to_numpy()
B3, hit_jt, hit_kw = 2000, 0, 0
for _ in range(B3):
    s = rng3.permutation(race_lab)         # 인종과 무관하게 섞음
    gs = sorted([y3[s == k] for k in ["White", "Black", "Other"]],
                key=np.median)             # 매번 중앙값 순서로 늘어놓음
    hit_jt += jt_test(gs)[3] < 0.05
    hit_kw += stats.kruskal(*gs).pvalue < 0.05
print(f"'significant' in {B3} null data sets:")
print(f"  JT, order from data {hit_jt / B3:.3f}")
print(f"  Kruskal-Wallis      {hit_kw / B3:.3f}")
''', title="과제 3 정답. 자료를 본 뒤 정한 순서로 한 추세 검정")
S("lab04_hw3", H(c, marks={"['Other', 'Black', 'White']": 1, "['Black', 'Other', 'White']": 2,
                           "P for trend = 0.011": 3, "order from data 0.112": 4, "Kruskal-Wallis      0.056": 5}))
NOTES[c.n] = ("**과제 3.** 동료가 인종 분류의 세 군을 출생체중 중앙값이 작은 순서(Other, Black, White)로 늘어놓고 "
              f"셀 {JT_CELL}의 `jt_test`로 Jonckheere-Terpstra 검정을 한 뒤 \"출생체중은 Other, Black, White 순서로 "
              "커지는 추세가 있었다(z = 2.54, P for trend = 0.011)\"라고 썼습니다. 동료의 분석을 재현하고, "
              "인종과 출생체중이 무관한 자료를 여러 번 만들어 같은 절차를 되풀이하면 '유의'가 몇 %나 나오는지 확인한 뒤, "
              "무엇이 왜 틀렸는지 설명하세요.")

print("lab04: cells =", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 4. 세 군 이상의 크기 비교",
    intro=("사회약학 연구방법 노트의 '실습 4. 세 군 이상의 크기 비교'에 딸린 노트북입니다. MASS 패키지의 birthwt 자료"
           "(출생아와 산모 189명)로 어머니 인종 분류에 따른 세 군의 출생체중을 분산분석, Welch 분산분석, Tukey 사후분석, "
           "Kruskal-Wallis 검정과 Dunn 검정으로 비교하고, 어머니 체중 삼분위에 Jonckheere-Terpstra 검정을 직접 만들어 "
           "적용합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 4에 있습니다."),
    notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy import stats

    ns = nb.ns
    df_ = ns["df"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r1 = lambda x: f"{x:.1f}"
    r2 = lambda x: f"{x:.2f}"
    # 글(content/lab04.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 5: "군별 흡연 비율", 7: "분산분석표 (statsmodels)",
             9: "잔차의 정규성", 11: "Tukey HSD 사후분석", 12: "Kruskal-Wallis 검정",
             14: "Dunn 검정 직접 계산하기", 18: "정규근사로 p-value 구하기", 19: "statsmodels 0.15의 함수와 비교",
             20: "순열검정으로 확인하기", 25: "과제 3 정답. 자료를 본 뒤 정한 순서로 한 추세 검정"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    same("cell constants", (SMOKE_CELL, ANOVA_CELL, TUKEY_CELL, KW_CELL, DUNN_CELL, JT_CELL, SM_JT_CELL, PERM_CELL),
         (5, 7, 11, 12, 14, 18, 19, 20))
    # 대응표 (이 실습의 자료 열)
    same("N", len(df_), 189)
    same("race n", df_["race_f"].value_counts(sort=False).tolist(), [96, 26, 67])
    same("smoking by race", [f"{x:.0%}" for x in pd.crosstab(df_["race_f"], df_["smoke"], normalize="index")[1]],
         ["54%", "38%", "18%"])
    same("ANOVA F, P", (r2(ns["res"].statistic), f"{ns['res'].pvalue:.3f}"), ("4.91", "0.008"))
    same("tertile sizes", df_["lwt3"].value_counts(sort=False).tolist(), [67, 60, 62])
    from statsmodels.stats.nonparametric import jonckheere_terpstra
    same("tertile JT p", f"{jonckheere_terpstra(ns['tert'], alternative='two-sided').pvalue:.4f}", "0.0006")
    # a·b 절 글
    same("SD ratio", f"{727.9 / 638.7:.2f}", "1.14")
    same("Other mean", r1(ns["g"].mean()["Other"]), "2805.3")
    same("SD range", (r1(ns["g"].std().min()), r1(ns["g"].std().max())), ("638.7", "727.9"))
    # '직접 바꿔 보기' (b 절)
    th = stats.tukey_hsd(ns["white"], ns["black"], ns["other"])
    same("scipy tukey_hsd p", (f"{th.pvalue[0, 1]:.3f}", f"{th.pvalue[0, 2]:.3f}", f"{th.pvalue[1, 2]:.3f}"),
         ("0.043", "0.026", "0.862"))
    pm = ns["model"].params
    same("model params", (r1(pm["Intercept"]), r1(pm["race_f[T.Black]"]), r1(pm["race_f[T.Other]"])),
         ("3102.7", "-383.0", "-297.4"))
    swl = stats.f_oneway(ns["white"], ns["black"], ns["other"], equal_var=False)
    same("scipy Welch (equal_var=False)", (f"{swl.statistic:.3f}", f"{swl.pvalue:.4f}"), ("5.060", "0.0088"))
    # c 절 글
    same("Holm second step", f"{sorted(ns['p_mw'])[1] * 2:.4f}", "0.0419")
    same("superiority", [r2(x) for x in ns["sup"][:2]], ["0.35", "0.39"])
    # 과제 1
    ms = ns["m_smoke"]
    same("hw1 adjusted Black, smoke", (r1(ms.params["race_f[T.Black]"]), r1(ms.params["race_f[T.Other]"])),
         ("-450.4", "-452.9"))
    same("hw1 eta2", f"{ns['eta2']:.2f}", "0.05")
    # 과제 2
    dn = ns["dun"]
    ci = dn.confidence_interval()
    same("hw2 Dunnett p", (f"{dn.pvalue[0]:.3f}", f"{dn.pvalue[1]:.3f}"), ("0.032", "0.019"))
    same("hw2 Dunnett CI", (r1(ci.low[0]), r1(ci.high[0]), r1(ci.low[1]), r1(ci.high[1])),
         ("-737.9", "-28.1", "-553.0", "-41.9"))
    same("hw2 widths", (f"{ci.high[0] - ci.low[0]:.0f}", f"{756.2363 - 9.8166:.0f}"), ("710", "746"))
    # 과제 3
    same("hw3 z", r2(ns["z_c"]), "2.54")
    same("hw3 medians", (r1(ns["med"]["Other"]), r1(ns["med"]["Black"]), r1(ns["med"]["White"])),
         ("2835.0", "2849.0", "3062.0"))
    same("hw3 KW", f"{ns['kw'].pvalue:.3f}", "0.014")
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab04.html passed")
