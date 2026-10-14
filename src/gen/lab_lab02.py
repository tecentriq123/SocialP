"""실습 2 (lab02) — 두 군의 크기 비교: birthwt, 흡연 여부별 출생체중.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab02.py
      then  python3 tools/nbcheck.py lab02   (runs the notebook alone in a fresh process and compares outputs)
All outputs are produced by actually running the cells (labkit). Writes figs/lab02_*.html,
pub/notebooks/lab02.ipynb (download link at the top of content/lab02.html) and checks/lab02.expected.json.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab02.html that are not marked in
a cell output (대응표, prose, 직접 바꿔 보기, the paper boxes, 과제).
(2026-10-14: version cell 1 added and the imports moved there; the scipy version line left the CI cell;
 과제 1–3 and the notebook added; old cells 1–15 became 2–16.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab02")
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
import scipy, matplotlib, seaborn, sklearn

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scipy       ", scipy.__version__)
print("matplotlib  ", matplotlib.__version__)
print("seaborn     ", seaborn.__version__)
print("scikit-learn", sklearn.__version__)
''', title="패키지 불러오기와 버전 확인")
_v = {k: nb.ns[k].__version__ for k in ("pd", "scipy", "seaborn")}
S("lab02_versions", H(c, marks={f"pandas       {_v['pd']}": 1, f"scipy        {_v['scipy']}": 2,
                                f"seaborn      {_v['seaborn']}": 3}))
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 birthwt 자료를 읽습니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
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
SUMMARY_CELL = c.n

c = nb.cell('''
df["BWT"].mean()
''', title="열 이름을 잘못 쓰면", expect_error=True)
S("lab02_keyerror", H(c))
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

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
''', title="상자그림과 개별 점", show_warnings=False)
S("lab02_plot", H(c))
NOTES[c.n] = (f"열 이름을 잘못 써서 일부러 오류를 내 보는 셀(셀 {ERR_CELL})은 이 노트북에서 뺐습니다"
              "(사이트의 실습 2 a 절에 있습니다).")

# ------------------------------------------------------------------ b. 독립표본 t 검정
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
NOTES[c.n] = "## b. 독립표본 t 검정"
TT_CELL = c.n

c = nb.cell('''
diff = smoker.mean() - nonsmoker.mean()
ci = welch.confidence_interval(confidence_level=0.95)
print(f"평균 차이 = {diff:.1f} g")
print(f"95% CI = {ci.low:.1f} ~ {ci.high:.1f} g")
''', title="평균 차이와 95% 신뢰구간")
S("lab02_ci", H(c, marks={"-283.8 g": 1, "-489.0 ~ -78.6 g": 2}))
CI_CELL = c.n

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

# ------------------------------------------------------------------ c. Mann-Whitney
c = nb.cell('''
q = (df.groupby("smoke_f", observed=True)["bwt"]
       .quantile([0.25, 0.5, 0.75])
       .unstack())
q
''', title="중앙값과 사분위수")
S("lab02_quant", H(c, marks={"3100.0": 1, "2775.5": 2}))
NOTES[c.n] = "## c. Mann-Whitney 검정"
QUANT_CELL = c.n

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
MW_CELL = c.n

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
AUC_CELL = c.n

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
HL_CELL = c.n

# ------------------------------------------------------------------ d. 과제 (정답 셀)
c = nb.cell('''
above = (smoker > nonsmoker.mean()).sum()      # 비흡연군 평균보다 무거운 흡연군 아기
print("smokers' infants above the nonsmoker mean:", above, "of", n1)

df["race_f"] = pd.Categorical(
    df["race"].map({1: "white", 2: "black", 3: "other"}),
    categories=["white", "black", "other"])
print(pd.crosstab(df["race_f"], df["smoke_f"], normalize="index").round(3))
df.groupby(["race_f", "smoke_f"], observed=True)["bwt"].mean().unstack().round(1)
''', title="과제 1 정답. 평균 차이가 말해 주지 않는 것")
S("lab02_hw1", H(c, marks={"above the nonsmoker mean: 25 of 74": 1, "0.542": 2, "0.179": 3,
                           "3428.8": 4, "2826.8": 5}))
HW1_CELL = c.n
NOTES[c.n] = (f"## d. 과제 정답\n\n**과제 1.** 셀 {CI_CELL}의 평균 차이와 95% 신뢰구간을 논문의 Results처럼 한 문장으로 "
              "옮기고, 이 숫자가 말해 주지 않는 것을 두 가지 이상 쓰세요. 비흡연군 평균보다 무거운 흡연군 아기가 몇 명인지, "
              "인종별 흡연 비율과 출생체중도 확인해 보세요.")

c = nb.cell('''
ht1 = df.loc[df["ht"] == 1, "bwt"]               # 고혈압 병력 있음
ht0 = df.loc[df["ht"] == 0, "bwt"]               # 없음
print(f"n = {len(ht1)} vs {len(ht0)}, "
      f"SD = {ht1.std():.1f} vs {ht0.std():.1f}")
for name, ev in [("Student", True), ("Welch", False)]:
    r = stats.ttest_ind(ht1, ht0, equal_var=ev)
    ci_ht = r.confidence_interval()
    print(f"{name:7}: diff = {ht1.mean() - ht0.mean():.1f}, "
          f"t = {r.statistic:.2f}, df = {r.df:.1f}, p = {r.pvalue:.3f}, "
          f"95% CI {ci_ht.low:.0f} ~ {ci_ht.high:.0f}")
''', title="과제 2 정답. 고혈압 병력으로 나눈 두 군의 Student와 Welch")
S("lab02_hw2", H(c, marks={"n = 12 vs 177": 1, "SD = 917.4 vs 709.4": 2, "df = 187.0, p = 0.045": 3,
                           "95% CI -861 ~ -10": 4, "df = 11.9, p = 0.133": 5, "95% CI -1024 ~ 154": 6}))
HW2_CELL = c.n
NOTES[c.n] = (f"**과제 2.** 셀 {TT_CELL}의 두 t 검정을 흡연 대신 고혈압 병력(`ht`)으로 두 군을 나눠 다시 하세요. "
              "Student와 Welch의 P와 95% 신뢰구간이 어떻게 다른지, 흡연에서는 비슷했던 두 결과가 왜 여기서는 크게 "
              "갈리는지 두 군의 인원과 SD로 설명하세요.")

c = nb.cell('''
print("median difference:", smoker.median() - nonsmoker.median())
q75 = smoker.quantile(0.75)
smoker_hi = smoker.where(smoker <= q75, smoker + 500)  # 위쪽 4분의 1만 +500 g
for name, x in [("original", smoker), ("top quarter +500", smoker_hi)]:
    r = stats.mannwhitneyu(x, nonsmoker, alternative="two-sided")
    print(f"{name:16}: median diff = {x.median() - nonsmoker.median():.1f}, "
          f"P(smoker > nonsmoker) = {r.statistic / n1n2:.3f}, "
          f"p = {r.pvalue:.4f}")
print(f"Hodges-Lehmann = {hl:.1f} g, 95% CI {lo} ~ {hi}")
''', title="과제 3 정답. Mann-Whitney 검정의 P는 중앙값의 차이를 검정하지 않습니다")
S("lab02_hw3", H(c, marks={"median difference: -324.5": 1, "P(smoker > nonsmoker) = 0.383, p = 0.0068": 2,
                           "P(smoker > nonsmoker) = 0.431, p = 0.1110": 3, "Hodges-Lehmann = -307.0 g": 4}))
HW3_CELL = c.n
NOTES[c.n] = ("**과제 3.** 동료가 결과를 \"흡연군 아기의 출생체중은 2,772 ± 660 g, 비흡연군은 3,056 ± 753 g였고, "
              "Mann-Whitney 검정에서 흡연군의 중앙값이 324.5 g 유의하게 낮았다(P = 0.007)\"라고 썼습니다. 틀린 곳 두 가지를 "
              "찾아 설명하세요. 흡연군의 위쪽 4분의 1 값만 500 g 무겁게 바꾸면 중앙값의 차이와 P가 어떻게 되는지 확인해 "
              "보세요.")

print("lab02: cells =", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 2. 두 군의 크기 비교",
    intro=("사회약학 연구방법 노트의 '실습 2. 두 군의 크기 비교'에 딸린 노트북입니다. MASS 패키지의 birthwt 자료"
           "(1986년 미국 한 병원의 출생아 189명)로 임신 중 흡연한 어머니와 흡연하지 않은 어머니의 출생아 체중을 "
           "Student·Welch t 검정과 Mann-Whitney 검정으로 비교하고, 평균 차이의 신뢰구간, Cohen의 d, 우월 확률, "
           "Hodges–Lehmann 추정치를 구합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 "
           "실습 2에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import stats

    ns = nb.ns
    df_ = ns["df"]
    sm, nsm = ns["smoker"], ns["nonsmoker"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    f1 = lambda x: f"{x:.1f}"
    f3 = lambda x: f"{x:.3f}"
    # 글(content/lab02.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 3: "흡연 여부에 이름표 붙이기", 4: "군별 요약통계", 6: "상자그림과 개별 점", 7: "Student t 검정과 Welch t 검정",
             8: "평균 차이와 95% 신뢰구간", 9: "같은 값을 공식으로 계산하기", 11: "Levene 검정",
             12: "중앙값과 사분위수", 13: "Mann-Whitney 검정", 15: "우월 확률과 AUC", 16: "Hodges-Lehmann 추정치",
             17: "과제 1 정답. 평균 차이가 말해 주지 않는 것",
             18: "과제 2 정답. 고혈압 병력으로 나눈 두 군의 Student와 Welch",
             19: "과제 3 정답. Mann-Whitney 검정의 P는 중앙값의 차이를 검정하지 않습니다"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    same("number of cells", len(nb.cells), 19)
    import scipy as _sp, seaborn as _sb
    same("versions", (pd.__version__, _sp.__version__, _sb.__version__), ("3.0.5", "1.18.1", "0.13.2"))
    # 대응표, a 절
    same("N, smokers, nonsmokers", (len(df_), len(sm), len(nsm)), (189, 74, 115))
    same("smoker SD", f1(sm.std()), "659.6")
    same("mean difference (rounded)", f"{sm.mean() - nsm.mean():.0f}", "-284")
    same("smoker min", int(sm.min()), 709)
    same("LBW by group", ((sm < 2500).sum(), f"{(sm < 2500).mean():.1%}", (nsm < 2500).sum(), f"{(nsm < 2500).mean():.1%}"),
         (30, "40.5%", 29, "25.2%"))
    # b 절
    st, we = ns["student"], ns["welch"]
    same("Student t, p", (f3(st.statistic), f"{st.pvalue:.4f}"), ("-2.653", "0.0087"))
    same("t crit", f3(ns["t_crit"]), "1.974")
    same("Welch t check", f3((sm.mean() - nsm.mean()) / ns["se"]), "-2.730")
    same("scipy CI direction swapped", (f1(-we.confidence_interval().high), f1(-we.confidence_interval().low)), ("78.6", "489.0"))
    r_sw = stats.ttest_ind(nsm, sm, equal_var=False)
    same("swap: t, p", (f"{r_sw.statistic:.2f}", f"{r_sw.pvalue:.4f}"), ("2.73", f"{we.pvalue:.4f}"))
    same("one-sided p", f"{stats.ttest_ind(sm, nsm, equal_var=False, alternative='less').pvalue:.4f}", "0.0035")
    same("Cohen SDs", (f1(sm.std()), f1(nsm.std())), ("659.6", "752.7"))
    # 직접 바꿔 보기: 709 -> 7090
    sm2 = sm.replace(709, 7090)
    same("outlier: SD, Welch p", (f"{sm2.std():.0f}", f3(stats.ttest_ind(sm2, nsm, equal_var=False).pvalue)), ("790", "0.090"))
    mw2 = stats.mannwhitneyu(sm2, nsm, alternative="two-sided")
    same("outlier: U, MW p", (mw2.statistic, f3(mw2.pvalue)), (3375.5, "0.017"))
    same("MW swapped statistic, p", (stats.mannwhitneyu(nsm, sm).statistic, f"{stats.mannwhitneyu(nsm, sm).pvalue:.4f}"),
         (5249.5, "0.0068"))
    same("MW no continuity p", f"{stats.mannwhitneyu(sm, nsm, use_continuity=False).pvalue:.4f}", "0.0067")
    # c 절
    q = ns["q"]
    same("quartiles", (q.loc["Nonsmoker", 0.25], q.loc["Nonsmoker", 0.75], q.loc["Smoker", 0.25], q.loc["Smoker", 0.75]),
         (2509.0, 3621.5, 2370.5, 3245.75))
    same("count: greater, ties", (int(ns["greater"]), int(ns["ties"])), (3241, 39))
    same("pairs", ns["n1n2"], 8510)
    same("AUC reversed", f3(1 - ns["auc"]), "0.617")
    same("median diff", sm.median() - nsm.median(), -324.5)
    # HL 셀의 코드를 2장 b 절의 오피오이드 예제에 적용하면 −40 ~ −3 MME
    _p = np.array([12, 15, 18, 20, 22, 26, 31, 68]); _u = np.array([24, 29, 36, 41, 45, 58, 115])
    _d = np.sort(np.subtract.outer(_p, _u).ravel()); _nn = len(_p) * len(_u)
    _k = round(_nn / 2 - 1.96 * np.sqrt(_nn * (len(_p) + len(_u) + 1) / 12))
    same("opioid HL CI with the lab code", (int(_d[_k - 1]), int(_d[_nn - _k]), float(np.median(_d))), (-40, -3, -18.5))
    # 과제 1
    race = df_["race"].map({1: "white", 2: "black", 3: "other"})
    tab = pd.crosstab(race, df_["smoke"], normalize="index")[1]
    same("smoking by race", [f"{tab[r]:.1%}" for r in ("white", "black", "other")], ["54.2%", "38.5%", "17.9%"])
    mb = df_.groupby(race)["bwt"].mean()
    same("mean bwt by race", [f1(mb[r]) for r in ("white", "black", "other")], ["3102.7", "2719.7", "2805.3"])
    g = df_.groupby([race, "smoke"])["bwt"].mean().unstack()
    g = g.round(1)       # 셀 출력에 보이는 반올림 값으로 뺌
    same("within-race differences (from the rounded table)", [f1(g.loc[r, 1] - g.loc[r, 0]) for r in ("white", "black", "other")],
         ["-602.0", "-350.5", "-58.6"])
    same("smokers per race", [int(((race == r) & (df_["smoke"] == 1)).sum()) for r in ("white", "black", "other")], [52, 10, 12])
    same("share above mean", f"{25 / 74:.1%}", "33.8%")
    same("nonsmokers above own mean", f"{(nsm > nsm.mean()).mean():.1%}", "55.7%")
    same("smoking ratio white/other > 3", tab["white"] / tab["other"] > 3, True)
    same("white nonsmoker heaviest of six", float(g.max().max()), 3428.8)
    # 과제 2
    h1, h0 = df_.loc[df_["ht"] == 1, "bwt"], df_.loc[df_["ht"] == 0, "bwt"]
    same("ht diff", f1(h1.mean() - h0.mean()), "-435.4")
    sp_ht = np.sqrt(((len(h1) - 1) * h1.var() + (len(h0) - 1) * h0.var()) / (len(h1) + len(h0) - 2))
    same("ht pooled SD", f1(sp_ht), "723.3")
    same("ht SE Student, Welch", (f1(sp_ht * np.sqrt(1 / len(h1) + 1 / len(h0))), f1(np.sqrt(h1.var() / len(h1) + h0.var() / len(h0)))),
         ("215.8", "270.1"))
    same("ht weights", (f"{(len(h1) - 1) / (len(h1) + len(h0) - 2):.0%}", f"{(len(h0) - 1) / (len(h1) + len(h0) - 2):.0%}"),
         ("6%", "94%"))
    same("ht size ratio about 15", f"{len(h0) / len(h1):.0f}", "15")
    same("ht t", (f"{stats.ttest_ind(h1, h0).statistic:.2f}", f"{stats.ttest_ind(h1, h0, equal_var=False).statistic:.2f}"),
         ("-2.02", "-1.61"))
    # 과제 3
    same("IQR widths", (q.loc["Smoker", 0.75] - q.loc["Smoker", 0.25], q.loc["Nonsmoker", 0.75] - q.loc["Nonsmoker", 0.25]),
         (875.25, 1112.5))
    same("upper quarter count", int((sm > sm.quantile(0.75)).sum()), 19)
    same("shifted median unchanged", ns["smoker_hi"].median(), sm.median())
    same("HL CI", (int(ns["lo"]), int(ns["hi"])), (-512, -85))
    same("colleague's mean ± SD", (f"{sm.mean():.0f}", f"{sm.std():.0f}", f"{nsm.mean():.0f}", f"{nsm.std():.0f}"),
         ("2772", "660", "3056", "753"))
    # 셀 6의 seaborn 상자그림은 matplotlib 3.11에서 vert 인자에 대한 MatplotlibDeprecationWarning을 낸다(그림에는 영향 없음).
    # 출력에는 싣지 않고(show_warnings=False) 글(a 절 '상자그림과 개별 점')에 한 문장으로 적는다.
    import warnings
    import matplotlib.pyplot as plt
    import seaborn as sns
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        sns.boxplot(data=df_, x="smoke_f", y="bwt", ax=ax)
        plt.close(fig)
    same("boxplot warning", any(w.category.__name__ == "MatplotlibDeprecationWarning" and "vert" in str(w.message) for w in wl),
         True)
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab02.html passed")
