"""실습 3 (lab03) — 치료 전과 후의 크기 비교: sleep (Student 1908), 대응 자료.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab03.py
All outputs are produced by actually running the cells (labkit).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook

nb = Notebook("lab03")
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
import matplotlib.pyplot as plt

url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/datasets/sleep.csv")
sleep = pd.read_csv(url)
sleep
''', title="데이터 불러오기", max_rows=20)
S("lab03_load", H(c, marks={"0.7": 1, "1.9": 2}))

c = nb.cell('''
pd.crosstab(sleep["ID"], sleep["group"])
''', title="짝 확인하기")
S("lab03_crosstab", H(c))

c = nb.cell('''
wide = sleep.pivot(index="ID", columns="group", values="extra")
wide.columns = ["drug1", "drug2"]
wide["d"] = wide["drug2"] - wide["drug1"]   # 환자별 차이
wide
''', title="넓은 형식으로 바꾸기")
S("lab03_wide", H(c, marks={"1.2": 1, "-0.1": 2, "4.6": 3}))

c = nb.cell('''
dup = pd.concat([sleep, sleep.iloc[[0]]])   # 첫 행이 두 번 들어감
dup.pivot(index="ID", columns="group", values="extra")
''', title="같은 짝이 두 번 있으면", expect_error=True)
S("lab03_duperror", H(c))

c = nb.cell('''
r = wide["drug1"].corr(wide["drug2"])
print(f"두 약의 상관계수 r = {r:.3f}")
wide.describe().round(2)
''', title="요약통계와 상관")
S("lab03_describe", H(c, marks={"r = 0.795": 1, "1.58": 2, "1.23": 3}))

c = nb.cell('''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.6))
for pid, row in wide.iterrows():            # 환자 한 명씩
    ax1.plot([1, 2], [row["drug1"], row["drug2"]],
             marker="o", color="gray")
ax1.set_xticks([1, 2], ["Drug 1", "Drug 2"])
ax1.set_xlim(0.7, 2.3)
ax1.set_ylabel("Extra sleep (hours)")

ax2.scatter(wide.index, wide["d"], color="black")
ax2.axhline(0, ls="--", color="gray")
ax2.set_xticks(wide.index)
ax2.set_xlabel("Patient ID")
ax2.set_ylabel("Drug 2 - Drug 1 (hours)")
plt.tight_layout()
plt.show()
''', title="짝지은 그림")
S("lab03_plot", H(c))

# ------------------------------------------------------------------ 나. 대응표본 t 검정
c = nb.cell('''
res = stats.ttest_rel(wide["drug2"], wide["drug1"])
print(res)
print(f"t = {res.statistic:.3f}, p = {res.pvalue:.4f}, "
      f"df = {res.df}")
ci = res.confidence_interval()
print(f"mean d = {wide['d'].mean():.2f} h, "
      f"95% CI {ci.low:.2f} ~ {ci.high:.2f}")
''', title="대응표본 t 검정")
S("lab03_ttest", H(c, marks={"t = 4.062": 1, "p = 0.0028": 2, "df = 9": 3,
                              "mean d = 1.58 h": 4, "95% CI 0.70 ~ 2.46": 5}))

c = nb.cell('''
d = wide["d"]
one = stats.ttest_1samp(d, popmean=0)
print(f"one-sample: t = {one.statistic:.3f}, p = {one.pvalue:.4f}")

se = d.std() / np.sqrt(len(d))       # 차이의 표준오차
print(f"mean = {d.mean():.2f}, SD = {d.std():.3f}, "
      f"SE = {se:.3f}, t = {d.mean() / se:.3f}")
''', title="차이 d에 대한 일표본 t 검정")
S("lab03_onesample", H(c, marks={"one-sample: t = 4.062, p = 0.0028": 1, "SD = 1.230": 2,
                                  "SE = 0.389": 3}))

c = nb.cell('''
wrong = stats.ttest_ind(wide["drug2"], wide["drug1"])  # 잘못된 분석
ci_w = wrong.confidence_interval()
print(f"t = {wrong.statistic:.3f}, p = {wrong.pvalue:.4f}, "
      f"df = {wrong.df:.0f}")
print(f"95% CI {ci_w.low:.2f} ~ {ci_w.high:.2f}")
''', title="잘못된 분석: 독립표본 t 검정")
S("lab03_wrong", H(c, marks={"t = 1.861": 1, "p = 0.0792": 2, "df = 18": 3, "95% CI -0.20 ~ 3.36": 4}))

c = nb.cell('''
s1, s2 = wide["drug1"].std(), wide["drug2"].std()
sd_paired = np.sqrt(s1**2 + s2**2 - 2 * r * s1 * s2)
sd_indep = np.sqrt(s1**2 + s2**2)            # r = 0으로 본 경우
print(f"SD1 = {s1:.3f}, SD2 = {s2:.3f}, r = {r:.3f}")
print(f"SD of d: paired {sd_paired:.3f}, "
      f"if r = 0 {sd_indep:.3f}")
''', title="상관이 표준오차를 줄이는 원리")
S("lab03_var", H(c, marks={"SD1 = 1.789": 1, "paired 1.230": 2, "if r = 0 2.685": 3}))

c = nb.cell('''
sw = stats.shapiro(d)
print(f"Shapiro-Wilk W = {sw.statistic:.3f}, p = {sw.pvalue:.3f}")
''', title="차이 d의 정규성")
S("lab03_shapiro", H(c, marks={"W = 0.830": 1, "p = 0.033": 2}))

# ------------------------------------------------------------------ 다. Wilcoxon
c = nb.cell('''
w = stats.wilcoxon(d)
print(w)
''', title="Wilcoxon 부호순위 검정")
S("lab03_wilcoxon", H(c, marks={"statistic=np.float64(0.0)": 1, "pvalue=np.float64(0.00390625)": 2}))

c = nb.cell('''
nz = d[d != 0]                   # 차이가 0인 쌍은 제외
rk = stats.rankdata(nz.abs())    # |d|에 순위 (동점은 평균순위)
print(pd.DataFrame({"d": nz, "rank": rk}).T)
w_plus = rk[nz > 0].sum()        # 늘어난 쪽의 순위합
w_minus = rk[nz < 0].sum()       # 줄어든 쪽의 순위합
print("n =", len(nz), " W+ =", w_plus, " W- =", w_minus)
''', title="부호순위를 직접 계산하기")
S("lab03_ranks", H(c, marks={"4.5": 1, "n = 9": 2, "W+ = 45.0": 3, "W- = 0.0": 4}))

c = nb.cell('''
g = stats.wilcoxon(d, alternative="greater")
print(f"greater: statistic = {g.statistic}, p = {g.pvalue:.5f}")
a = stats.wilcoxon(d, method="asymptotic")
print(f"asymptotic: z = {a.zstatistic:.3f}, p = {a.pvalue:.4f}")
ac = stats.wilcoxon(d, method="asymptotic", correction=True)
print(f"asymptotic + correction: p = {ac.pvalue:.4f}")
pr = stats.wilcoxon(d, zero_method="pratt", method="asymptotic")
print(f"pratt (asymptotic): p = {pr.pvalue:.4f}")
''', title="p-value 계산 방식 비교")
S("lab03_methods", H(c, marks={"greater: statistic = 45.0": 1, "p = 0.00195": 2, "z = -2.668": 3,
                                "p = 0.0076": 4, "correction: p = 0.0091": 5,
                                "pratt (asymptotic): p = 0.0058": 6}))

c = nb.cell('''
n_pos = int((d > 0).sum())      # drug2에서 더 잔 환자
n_neg = int((d < 0).sum())      # drug1에서 더 잔 환자
print("+:", n_pos, " -:", n_neg, " 0:", int((d == 0).sum()))
print(stats.binomtest(n_pos, n_pos + n_neg, p=0.5))
''', title="부호검정")
S("lab03_sign", H(c, marks={"+: 9": 1, "pvalue=0.00390625": 2}))

c = nb.cell('''
print(wide.median())
print(d.quantile([0.25, 0.5, 0.75]))
''', title="중앙값과 사분위수")
S("lab03_median", H(c, marks={"drug1    0.35": 1, "drug2    1.75": 2, "0.25    1.05": 3, "0.75    1.70": 4}))

print("lab03: cells =", len(nb.cells))
