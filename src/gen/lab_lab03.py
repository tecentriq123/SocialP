"""실습 3 (lab03) — 치료 전과 후의 크기 비교: sleep (Student 1908), 대응 자료.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab03.py
      then  python3 tools/nbcheck.py lab03   (runs the notebook alone in a fresh process and compares outputs)
All outputs are produced by actually running the cells (labkit).
Writes figs/lab03_*.html (insert in content/lab03.html with <!--FIG:lab03_xxx-->),
pub/notebooks/lab03.ipynb (download link at the top of content/lab03.html) and checks/lab03.expected.json.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab03.html that are not
marked in a cell output (대응표, the swapped-argument t test, the 과제 problems and answers).
(2026-10-14: version cell 1 split from the old load cell, Q-Q plot added to the normality cell, 과제 1–3 and the
notebook added; old cells 1–16 are now cells 2–17.)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook

nb = Notebook("lab03")
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
import matplotlib.pyplot as plt
import scipy, matplotlib

print("pandas    ", pd.__version__)
print("numpy     ", np.__version__)
print("scipy     ", scipy.__version__)
print("matplotlib", matplotlib.__version__)
''', title="패키지 불러오기와 버전 확인")
S("lab03_versions", H(c, marks={"scipy      1.18.1": 1}))
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 sleep 자료를 읽습니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/datasets/sleep.csv")
sleep = pd.read_csv(url)
sleep
''', title="데이터 불러오기", max_rows=20)
S("lab03_load", H(c, marks={"rownames": 1, "0.7": 2, "1.9": 3}))

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
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

c = nb.cell('''
r = wide["drug1"].corr(wide["drug2"])
print(f"두 약의 상관계수 r = {r:.3f}")
wide.describe().round(2)
''', title="요약통계와 상관")
S("lab03_describe", H(c, marks={"r = 0.795": 1, "1.58": 2, "1.23": 3}))
NOTES[c.n] = (f"pivot에 같은 짝이 두 번 있으면 나는 오류를 일부러 내 보는 셀(셀 {ERR_CELL})은 이 노트북에서 뺐습니다"
              "(사이트의 실습 3 a 절에 있습니다).")

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

# ------------------------------------------------------------------ b. 대응표본 t 검정
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
NOTES[c.n] = "## b. 대응표본 t 검정"
TTEST_CELL = c.n

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

fig, ax = plt.subplots(figsize=(4.2, 3.6))
stats.probplot(d, dist="norm", plot=ax)     # 차이 d의 정규 Q-Q plot
ax.set_title("Normal Q-Q plot of d")
plt.show()
''', title="차이 d의 정규성")
S("lab03_shapiro", H(c, marks={"W = 0.830": 1, "p = 0.033": 2}))
SHAPIRO_CELL = c.n

# ------------------------------------------------------------------ c. Wilcoxon
c = nb.cell('''
w = stats.wilcoxon(d)
print(w)
''', title="Wilcoxon 부호순위 검정")
S("lab03_wilcoxon", H(c, marks={"statistic=np.float64(0.0)": 1, "pvalue=np.float64(0.00390625)": 2}))
NOTES[c.n] = "## c. Wilcoxon 부호순위 검정"

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

# ------------------------------------------------------------------ d. 과제 (정답 셀)
c = nb.cell('''
print(d.round(1).sort_values().tolist())       # 환자별 차이, 작은 순서로
print("median d:", d.median())
print("d >= 1 h:", int((d >= 1).sum()), "of", len(d))
wide[["drug1", "drug2"]].mean().round(2).rename("mean")   # 약을 쓰지 않은 밤 대비
''', title="과제 1 정답. 대응표본 t 검정의 한 행 읽기")
S("lab03_hw1", H(c, marks={"[0.0": 1, "median d: 1.3": 2, "8 of 10": 3, "0.75": 4, "2.33": 5}))
NOTES[c.n] = (f"## 과제 정답\n\n**과제 1.** 셀 {TTEST_CELL}의 출력 가운데 `mean d = 1.58 h, 95% CI 0.70 ~ 2.46` 행을 "
              "p-value와 함께 논문의 Results처럼 한 문장으로 옮기고, 이 결과가 말해 주지 않는 것을 두 가지 이상 쓰세요. "
              "환자별 차이와 각 약의 평균을 확인해 근거로 쓰세요.")

c = nb.cell('''
def three_tests(x):                    # 차이 x에 세 검정을 한 번에
    t = stats.ttest_1samp(x, 0)
    return {"mean": x.mean(), "SD": x.std(),
            "paired t p": t.pvalue,
            "signed-rank p": stats.wilcoxon(x).pvalue,
            "sign test p": stats.binomtest(int((x > 0).sum()),
                                           int((x != 0).sum())).pvalue}

d2 = d.copy()
d2[9] = 14.6                           # 환자 9의 차이만 4.6 -> 14.6
pd.DataFrame({"d": three_tests(d), "d2": three_tests(d2)}).round(4)
''', title="과제 2 정답. 극단값 하나가 커지면 세 검정은")
S("lab03_hw2", H(c, marks={"2.5800": 1, "4.2690": 2, "0.0883": 3, "signed-rank p": 4}))
NOTES[c.n] = ("**과제 2.** 환자 9의 차이 d가 4.6이 아니라 14.6이었다고 하고(약 2의 효과가 더 커 보이는 방향), "
              "대응표본 t 검정, Wilcoxon 부호순위 검정, 부호검정을 원래 자료와 나란히 다시 돌리세요. "
              "어느 검정의 P가 바뀌고 어느 것이 그대로인지, 왜 그런지 설명하세요.")

c = nb.cell('''
s = sleep.sort_values("extra")                  # 동료의 코드: 값 크기순 정렬
drug1_s = s.loc[s["group"] == 1, ["ID", "extra"]]
drug2_s = s.loc[s["group"] == 2, ["ID", "extra"]]
print("Shapiro p (drug 1, drug 2):",
      round(stats.shapiro(drug1_s["extra"]).pvalue, 2),
      round(stats.shapiro(drug2_s["extra"]).pvalue, 2))
res_c = stats.ttest_rel(drug2_s["extra"], drug1_s["extra"])
ci_c = res_c.confidence_interval()
print(f"colleague: t = {res_c.statistic:.2f}, p = {res_c.pvalue:.1e}")
print(f"           95% CI {ci_c.low:.2f} ~ {ci_c.high:.2f}")
pd.DataFrame({"ID (drug 1)": drug1_s["ID"].to_numpy(),
              "ID (drug 2)": drug2_s["ID"].to_numpy(),
              "d used": (drug2_s["extra"].to_numpy()
                         - drug1_s["extra"].to_numpy()).round(1)})
''', title="과제 3 정답. 정렬한 뒤 행 순서로 짝지은 분석")
S("lab03_hw3", H(c, marks={"0.41 0.35": 1, "t = 9.27": 2, "p = 6.7e-06": 3, "95% CI 1.19 ~ 1.97": 4}))
NOTES[c.n] = ("**과제 3.** 동료가 sleep 자료를 값 크기순으로 정렬한 뒤 약 1과 약 2의 값을 각각 꺼내 정규성을 확인하고 "
              "`stats.ttest_rel`에 넣었습니다. 그리고 \"두 약의 값이 각각 정규분포를 따랐으므로(Shapiro-Wilk P = 0.41, 0.35) "
              "대응표본 t 검정을 했다. 약 2는 약 1보다 수면을 평균 1.58시간 더 늘렸다(95% CI 1.19–1.97, t(9) = 9.27, "
              "P < 0.001)\"라고 썼습니다. 동료의 분석을 재현하고, 틀린 곳 두 가지를 찾아 왜 틀렸는지 설명하세요.")

print("lab03: cells =", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 3. 치료 전과 후의 크기 비교",
    intro=("사회약학 연구방법 노트의 '실습 3. 치료 전과 후의 크기 비교'에 딸린 노트북입니다. Student(1908)가 t 검정의 "
           "예제로 쓴 sleep 자료(같은 환자 10명이 두 수면제를 모두 복용)로 대응표본 t 검정과 Wilcoxon 부호순위 검정을 "
           "합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 3에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import stats

    ns = nb.ns
    wide_, d_, sleep_ = ns["wide"], ns["d"], ns["sleep"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r2 = lambda x: f"{x:.2f}"
    # 글(content/lab03.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 5: "같은 짝이 두 번 있으면", 6: "요약통계와 상관", 8: "대응표본 t 검정",
             9: "차이 d에 대한 일표본 t 검정", 12: "차이 d의 정규성", 13: "Wilcoxon 부호순위 검정",
             16: "부호검정", 17: "중앙값과 사분위수"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    same("ERR_CELL", ERR_CELL, 5)
    same("TTEST_CELL", TTEST_CELL, 8)
    same("SHAPIRO_CELL", SHAPIRO_CELL, 12)
    # 대응표 (이 실습의 자료 열)와 a 절의 글
    same("rows, patients", (len(sleep_), sleep_["ID"].nunique()), (20, 10))
    same("d > 0, d == 0, d < 0", (int((d_ > 0).sum()), int((d_ == 0).sum()), int((d_ < 0).sum())), (9, 1, 0))
    same("patient 5 zero, patient 9 largest", (float(d_[5]), float(round(d_[9], 1)), int(d_.idxmax())), (0.0, 4.6, 9))
    same("other patients' d range", (float(round(d_.drop(9).min(), 1)), float(round(d_.drop(9).max(), 1))), (0.0, 2.4))
    same("drug 1 range", (float(wide_["drug1"].min()), float(wide_["drug1"].max())), (-1.6, 3.7))
    same("drug SDs", (r2(wide_["drug1"].std()), r2(wide_["drug2"].std())), ("1.79", "2.00"))
    res_ = ns["res"]
    same("paired t (rounded)", (r2(res_.statistic), f"{res_.pvalue:.3f}", r2(ns["ci"].low), r2(ns["ci"].high)),
         ("4.06", "0.003", "0.70", "2.46"))
    # 인자 순서를 바꾼 대응표본 t 검정 (b 절 셀 8의 설명)
    sw_ = stats.ttest_rel(wide_["drug1"], wide_["drug2"])
    ci_sw = sw_.confidence_interval()
    same("swapped arguments", (f"{sw_.statistic:.3f}", r2(ci_sw.low), r2(ci_sw.high)), ("-4.062", "-2.46", "-0.70"))
    # 표준오차의 비, 정규근사 계산 (b·c 절 해설)
    same("SE ratio", f"{(ns['sd_indep'] / np.sqrt(10)) / ns['se']:.1f}", "2.2")
    same("SE independent", f"{ns['sd_indep'] / np.sqrt(10):.3f}", "0.849")
    same("p ratio wrong/paired", f"{ns['wrong'].pvalue / res_.pvalue:.0f}", "28")
    same("t critical df 9", f"{stats.t.ppf(0.975, 9):.3f}", "2.262")
    same("signed-rank var", f"{9 * 10 * 19 / 24 - (2**3 - 2) / 48:.3f}", "71.125")
    same("continuity z", f"{(0 + 0.5 - 22.5) / np.sqrt(71.125):.3f}", "-2.609")
    same("pratt default method p", f"{stats.wilcoxon(d_, zero_method='pratt').pvalue:.4f}", "0.0039")
    # c 절 비교표 (p-value 계산 방식)
    same("methods table", (f"{ns['w'].pvalue:.4f}", f"{ns['g'].pvalue:.4f}", f"{ns['a'].pvalue:.4f}",
                           f"{ns['ac'].pvalue:.4f}", f"{ns['pr'].pvalue:.4f}"),
         ("0.0039", "0.0020", "0.0076", "0.0091", "0.0058"))
    same("one-sided p (5 digits)", f"{ns['g'].pvalue:.5f}", "0.00195")
    # 과제 1
    same("hw1 d >= 1", int((d_ >= 1).sum()), 8)
    same("hw1 median, mean", (float(d_.median()), r2(d_.mean())), (1.3, "1.58"))
    same("hw1 patients with d < 1", [int(i) for i in d_[d_ < 1].index], [5, 8])
    same("hw1 d of patient 8", float(round(d_[8], 1)), 0.8)
    same("hw1 mean without patient 9", r2(d_.drop(9).mean()), "1.24")
    same("hw1 drug means", (r2(wide_["drug1"].mean()), r2(wide_["drug2"].mean())), ("0.75", "2.33"))
    # 과제 2
    d2_ = ns["d2"]
    t2 = stats.ttest_1samp(d2_, 0)
    ci2 = t2.confidence_interval()
    same("hw2 t, CI", (f"{t2.statistic:.2f}", r2(ci2.low), r2(ci2.high)), ("1.91", "-0.47", "5.63"))
    same("hw2 SD ratio", f"{d2_.std() / d_.std():.1f}", "3.5")
    same("hw2 SE", (f"{d_.std() / np.sqrt(10):.2f}", f"{d2_.std() / np.sqrt(10):.2f}"), ("0.39", "1.35"))
    rank9 = lambda x: float(pd.Series(stats.rankdata(x[x != 0].abs()), index=x[x != 0].index)[9])
    same("hw2 rank of patient 9 (largest of 9)", (rank9(d_), rank9(d2_)), (9.0, 9.0))
    # 과제 3
    d1s, d2s = ns["drug1_s"], ns["drug2_s"]
    dw = d2s["extra"].to_numpy() - d1s["extra"].to_numpy()
    same("hw3 mean of wrong d", r2(dw.mean()), "1.58")
    same("hw3 SD of wrong d", f"{dw.std(ddof=1):.2f}", "0.54")
    same("hw3 corr of sorted columns", r2(np.corrcoef(d1s["extra"], d2s["extra"])[0, 1]), "0.97")
    same("hw3 matched pairs (IDs)", [int(x) for x in d1s["ID"].to_numpy()[d1s["ID"].to_numpy() == d2s["ID"].to_numpy()]],
         [4, 1, 7])
    same("hw3 first pair", (int(d1s["ID"].iloc[0]), int(d2s["ID"].iloc[0])), (2, 5))
    same("hw3 Shapiro d", f"{stats.shapiro(d_).pvalue:.3f}", "0.033")
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab03.html passed")
