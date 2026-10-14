"""실습 13 · 동등성·비열등성 검정 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab13.py
      then  python3 tools/nbcheck.py lab13   (runs the notebook alone in a fresh process and compares outputs)
      (LABDEBUG=1 prints every cell's text output, for writing marks)
Writes figs/lab13_*.html (insert in content/lab13.html with <!--FIG:lab13_xxx-->),
pub/notebooks/lab13.ipynb (download link at the top of content/lab13.html) and checks/lab13.expected.json.
Data: simulated HbA1c trial (seed 2024), 13장 summary data (H. pylori trial, AUC of 12 subjects),
datasets::Theoph (Rdatasets URL -> offline copy via labkit), simulated 2x2 crossover (seed 13).
The VERIFY block at the end asserts the numbers quoted in the text of content/lab13.html that are not
marked in a cell output (대응표, the explanations and tips, the 과제 problems and answers).
(2026-10-14: version cell 1, one-sided CI lines in cell 4, summary-statistics cell 6, 과제 1–3 and the notebook
added; old cells 1–4 moved to 2–5 and old cells 5–16 to 7–18.)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab13")
DEBUG = os.environ.get("LABDEBUG")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


def save(name, c, marks=None, **kw):
    """Render a cell. Marks are placed after the first match in the OUTPUT area
    (plain substring, or a regex when the key starts with '~'). labkit's own
    marks= needs the text in both print output and DataFrame table, hence this."""
    if DEBUG:
        print(f"\n===== {name} (셀 {c.n}) =====\n{c.stdout}{c.value_repr or ''}")
        if c.warns:
            print("WARN:", c.warns)
        if c.error:
            print("ERR:", c.error)
    h = nb.html(c, **kw)
    if marks:
        i = h.index('<div class="cell-out">')
        head, out = h[:i], h[i:]
        for sub, n in marks.items():
            if sub.startswith("~"):
                m = re.search(sub[1:], out)
                j = m.end() if m else -1
            else:
                e = _html.escape(sub)
                j = out.find(e)
                j = j + len(e) if j >= 0 else -1
            if j < 0:
                raise ValueError(f"{name}: mark text not found: {sub!r}")
            out = out[:j] + f'<span class="mk">{n}</span>' + out[j:]
        h = head + out
    nb.save_fragment(name, h)


# ------------------------------------------------------------------ 가
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import scipy, statsmodels, matplotlib

print("numpy       ", np.__version__)
print("pandas      ", pd.__version__)
print("scipy       ", scipy.__version__)
print("statsmodels ", statsmodels.__version__)
print("matplotlib  ", matplotlib.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab13_versions", c, marks={"~scipy\\s+[0-9.]+": 1, "~statsmodels\\s+[0-9.]+": 2})
NOTES[c.n] = ("## a. 평균 차이의 비열등성\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인합니다. c 절의 Theoph 자료는 "
              "인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
rng = np.random.default_rng(2024)    # 시드를 고정해 같은 난수 재현
new = rng.normal(loc=-0.74, scale=1.02, size=185)   # 신약 X
ref = rng.normal(loc=-0.83, scale=1.06, size=183)   # 대조약
print(new[:5].round(2))
for name, x in [("drug X", new), ("comparator", ref)]:
    print(f"{name}: n={x.size}, mean={x.mean():.3f}, "
          f"SD={x.std(ddof=1):.3f}")
''', title="모의 임상시험 자료 만들기")
save("lab13_sim", c, marks={"mean=-0.749": 1, "mean=-0.893": 2})

c = nb.cell('''
res = stats.ttest_ind(new, ref, equal_var=False)   # Welch t 검정
d = new.mean() - ref.mean()                        # 신약 - 대조약
ci = res.confidence_interval(confidence_level=0.95)
print(f"difference = {d:.3f}")
print(f"95% CI {ci.low:.3f} to {ci.high:.3f}")
print(f"P (difference = 0) = {res.pvalue:.3f}")
''', title="차이와 95% 신뢰구간")
save("lab13_ci", c, marks={"difference = 0.145": 1, "0.353": 2, "P (difference = 0) = 0.172": 3})
CI_CELL = c.n

c = nb.cell('''
margin = 0.4
ni = stats.ttest_ind(new - margin, ref, equal_var=False,
                     alternative="less")
print(ni)
se = np.sqrt(new.var(ddof=1) / new.size + ref.var(ddof=1) / ref.size)
print(f"SE = {se:.4f}, (d - margin) / SE = {(d - margin) / se:.3f}")
ci1 = ni.confidence_interval(confidence_level=0.975)   # 단측 97.5% 구간
print(f"one-sided 97.5% CI: {ci1.low} to {ci1.high + margin:.3f}")
''', title="비열등성 검정 (단측)")
save("lab13_ni", c, marks={r"~statistic=np\.float64\(-2\.41\d*\)": 1, r"~pvalue=np\.float64\(0\.0081\d*\)": 2,
                           "SE = 0.1058": 3, "-inf to 0.353": 4})
NI_CELL = c.n

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7, 2.4))
ax.errorbar(d, 0, xerr=[[d - ci.low], [ci.high - d]], fmt="s",
            color="C0", capsize=6, label="Difference (95% CI)")
ax.axvline(0, color="black", lw=1)
ax.axvline(margin, color="C1", ls="--", label="+margin (0.4)")
ax.axvline(-margin, color="grey", ls=":", label="-margin (-0.4)")
ax.set_xlim(-0.6, 0.6)
ax.set_yticks([])
ax.set_xlabel("HbA1c change, drug X minus comparator (%-points)\\n"
              "<- favors drug X          favors comparator ->")
ax.legend(loc="upper left", fontsize=8)
plt.show()
''', title="신뢰구간과 한계를 한 그림에")
save("lab13_niplot", c)

c = nb.cell('''
def ni_from_stats(label, m1, s1, n1, m2, s2, n2, margin, higher_better):
    v1, v2 = s1**2 / n1, s2**2 / n2               # 두 평균의 분산(SE의 제곱)
    se = np.sqrt(v1 + v2)                          # 차이의 표준오차
    df = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))  # Welch 자유도
    tc = stats.t.ppf(0.975, df)                    # 가운데 95%의 임계값
    d = m1 - m2                                    # 신약 - 대조
    if higher_better:      # 한계 -margin: 신약에 margin을 더하고 "greater"
        r = stats.ttest_ind_from_stats(m1 + margin, s1, n1, m2, s2, n2,
                                       equal_var=False, alternative="greater")
    else:                  # 한계 +margin: 신약에서 margin을 빼고 "less"
        r = stats.ttest_ind_from_stats(m1 - margin, s1, n1, m2, s2, n2,
                                       equal_var=False, alternative="less")
    print(f"{label}: difference {d:.2f}, SE {se:.4f}, df {df:.1f},"
          f" 95% CI {d - tc * se:.2f} to {d + tc * se:.2f},"
          f" t {r.statistic:.2f}, one-sided P {r.pvalue:.4f}")

ni_from_stats("PDC (ch13 b)", 83.9, 16.8, 304, 85.2, 17.5, 304, 5, True)
ni_from_stats("HbA1c (ch13 c)", -0.74, 1.02, 185, -0.83, 1.06, 183, 0.4, False)
''', title="요약 통계로 13장 b·c 절의 숫자 재현")
save("lab13_stats", c, marks={"difference -1.30": 1, "-4.03 to 1.43": 2, "P 0.0040": 3,
                              "difference 0.09": 4, "-0.12 to 0.30": 5, "t -2.86": 6})

# ------------------------------------------------------------------ 나
c = nb.cell('''
from statsmodels.stats.proportion import confint_proportions_2indep

trials = {"main": (186, 224, 190, 224),     # 성공 수, 인원 (신약, 표준)
          "small": (48, 50, 50, 50)}
rows = []
for name, (x1, n1, x2, n2) in trials.items():
    for m in ["wald", "newcomb"]:
        lo, hi = confint_proportions_2indep(x1, n1, x2, n2,
                                            method=m, compare="diff")
        rows.append([name, m, x1 / n1 - x2 / n2, lo, hi])
tab = pd.DataFrame(rows, columns=["trial", "method", "diff",
                                  "lower", "upper"])
tab["NI (lower > -0.10)"] = tab["lower"] > -0.10
tab.round(4)
''', title="위험차의 Wald 구간과 Newcombe 구간")
save("lab13_prop_ci", c)
NOTES[c.n] = "## b. 비율 차이의 비열등성"
PROP_CELL = c.n

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7, 2.8))
for i, r in tab.iterrows():
    ax.errorbar(r["diff"] * 100, -i, capsize=4, fmt="s",
                xerr=[[(r["diff"] - r["lower"]) * 100],
                      [(r["upper"] - r["diff"]) * 100]],
                color="C0" if r["method"] == "wald" else "C1")
ax.axvline(0, color="black", lw=1)
ax.axvline(-10, color="grey", ls=":")
ax.set_yticks(-tab.index, tab["trial"] + ", " + tab["method"])
ax.set_xlabel("Difference in eradication rate, new minus standard "
              "(%-points)")
plt.show()
''', title="네 신뢰구간 그리기")
save("lab13_prop_plot", c)

c = nb.cell('''
confint_proportions_2indep(48, 50, 50, 50, method="score",
                           compare="diff")
''', title="score 방법이 실패하는 경우", expect_error=True)
save("lab13_score_err", c)
ERR_CELL = c.n     # 일부러 오류를 내는 셀: 노트북에서는 뺀다(위에서부터 모두 실행이 멈추지 않게)

c = nb.cell('''
from statsmodels.stats.proportion import test_proportions_2indep

fm = test_proportions_2indep(186, 224, 190, 224, value=-0.10,
                             method="score", compare="diff",
                             alternative="larger", correction=False)
print(f"z = {fm.statistic:.3f}, one-sided P = {fm.pvalue:.4f}")
''', title="statsmodels의 score(Farrington–Manning) 검정")
save("lab13_fm_sm", c, marks={"z = 2.309": 1, "P = 0.0105": 2})
NOTES[c.n] = (f"score 방법이 실패하는 것을 보여 주는 셀(셀 {ERR_CELL})은 일부러 오류를 내므로 이 노트북에서 뺐습니다"
              "(사이트의 실습 13 b 절에 있습니다).")

c = nb.cell('''
from scipy.optimize import minimize_scalar

x1, n1, x2, n2, delta = 186, 224, 190, 224, -0.10

def negloglik(p2):          # 귀무가설 p1 - p2 = delta 아래의 -로그우도
    p1 = p2 + delta
    return -(stats.binom.logpmf(x1, n1, p1)
             + stats.binom.logpmf(x2, n2, p2))

opt = minimize_scalar(negloglik, bounds=(0.1001, 0.9999),
                      method="bounded")
p2t = opt.x
p1t = p2t + delta
se0 = np.sqrt(p1t * (1 - p1t) / n1 + p2t * (1 - p2t) / n2)
z = (x1 / n1 - x2 / n2 - delta) / se0
print(f"restricted MLE: new {p1t:.4f}, standard {p2t:.4f}")
print(f"z = {z:.3f}, one-sided P = {stats.norm.sf(z):.4f}")
''', title="Farrington–Manning 검정을 직접 계산")
save("lab13_fm_hand", c, marks={"new 0.7795, standard 0.8795": 1, "z = 2.333": 2,
                                "P = 0.0098": 3})

# ------------------------------------------------------------------ 다
c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/csv/"
       "datasets/Theoph.csv")
theo = pd.read_csv(url).drop(columns="rownames")
print(theo.shape, theo["Subject"].nunique())
theo.head(11)
''', title="Theoph 불러오기", max_rows=11)
save("lab13_theo", c, marks={"(132, 5) 12": 1})
NOTES[c.n] = "## c. 생물학적 동등성"

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7, 4))
for sid, g in theo.groupby("Subject"):
    ax.plot(g["Time"], g["conc"], marker="o", ms=3, alpha=0.7)
ax.set_xlabel("Time since dose (h)")
ax.set_ylabel("Theophylline concentration (mg/L)")
plt.show()
''', title="혈중농도-시간 곡선")
save("lab13_theo_plot", c)

c = nb.cell('''
from scipy.integrate import trapezoid

rows = []
for sid, g in theo.groupby("Subject"):
    g = g.sort_values("Time")
    rows.append({"Subject": sid,
                 "AUC_0_t": trapezoid(g["conc"], g["Time"]),
                 "Cmax": g["conc"].max(),
                 "Tmax": g.loc[g["conc"].idxmax(), "Time"]})
pk = pd.DataFrame(rows)
pk.round(2)
''', title="사다리꼴 규칙으로 AUC 계산", max_rows=12)
save("lab13_auc", c)

c = nb.cell('''
auc_R = np.array([412, 538, 365, 620, 488, 297,
                  554, 431, 702, 389, 515, 460])   # 대조약
auc_T = np.array([366, 455, 303, 548, 556, 268,
                  535, 510, 639, 429, 412, 525])   # 시험약
res = stats.ttest_rel(np.log(auc_T), np.log(auc_R))
ci90 = res.confidence_interval(confidence_level=0.90)
gmr = np.exp(np.mean(np.log(auc_T) - np.log(auc_R)))
print(f"GMR = {gmr * 100:.2f}%")
print(f"90% CI {np.exp(ci90.low) * 100:.2f}% "
      f"to {np.exp(ci90.high) * 100:.2f}%")
''', title="기하평균비와 90% 신뢰구간 (대응 자료)")
save("lab13_be", c, marks={"GMR = 95.75%": 1, "89.10% to 102.89%": 2})

c = nb.cell('''
dl = np.log(auc_T) - np.log(auc_R)          # 대상자별 ln(T/R)
lower = stats.ttest_1samp(dl, np.log(0.80), alternative="greater")
upper = stats.ttest_1samp(dl, np.log(1.25), alternative="less")
print(f"t_L = {lower.statistic:.2f}, P = {lower.pvalue:.5f}")
print(f"t_U = {upper.statistic:.2f}, P = {upper.pvalue:.5f}")
print(f"paired t test of no difference: P = {res.pvalue:.2f}")
''', title="TOST: 두 번의 단측 검정")
save("lab13_tost", c, marks={"t_L = 4.48, P = 0.00046": 1, "t_U = -6.65, P = 0.00002": 2,
                             "P = 0.30": 3})

c = nb.cell('''
rng = np.random.default_rng(13)
rows = []
for i in range(1, 25):                      # 대상자 24명
    seq = "TR" if i <= 12 else "RT"         # 순서군
    u = rng.normal(0, 0.25)                 # 개인의 흡수 수준
    for period in (1, 2):
        form = seq[period - 1]              # 이번 기간의 제제
        mu = (np.log(4500) + u + 0.02 * (period == 2)
              + np.log(0.97) * (form == "T"))
        rows.append([i, seq, period, form,
                     np.exp(mu + rng.normal(0, 0.13))])
be = pd.DataFrame(rows, columns=["subject", "seq", "period",
                                 "form", "auc"])
be.head(4)
''', title="2×2 교차설계 모의자료")
save("lab13_xo_sim", c)

c = nb.cell('''
import statsmodels.formula.api as smf

fit = smf.ols("np.log(auc) ~ C(subject) + C(period) + C(form)",
              data=be).fit()
t = "C(form)[T.T]"
lo, hi = fit.conf_int(alpha=0.10).loc[t]       # 90% 신뢰구간
print(f"GMR = {np.exp(fit.params[t]) * 100:.2f}%, "
      f"90% CI {np.exp(lo) * 100:.2f}% to {np.exp(hi) * 100:.2f}%")
print(f"residual df = {fit.df_resid:.0f}, MSE = {fit.mse_resid:.4f}")
print(f"CV_intra = {np.sqrt(np.exp(fit.mse_resid) - 1) * 100:.1f}%")
''', title="교차설계 분산분석으로 90% 신뢰구간")
save("lab13_xo_fit", c, marks={"GMR = 95.82%": 1, "89.71% to 102.36%": 2,
                               "residual df = 22": 3, "CV_intra = 13.4%": 4})
XO_CELL = c.n

# ------------------------------------------------------------------ d. 과제 (정답 셀)
c = nb.cell('''
wide = be.pivot(index="subject", columns="form", values="auc")  # 한 사람 = 한 행
ratio = wide["T"] / wide["R"]                     # 대상자별 T/R
print(wide.head(3).round(0))
print("geometric mean of T/R:",
      round(np.exp(np.log(ratio).mean()) * 100, 2), "%")
print("range of T/R:", round(ratio.min(), 3), "to", round(ratio.max(), 3))
outside = (ratio < 0.80) | (ratio > 1.25)
print("outside 80-125%:", outside.sum(), "of", len(ratio))
''', title="과제 1 정답. 기하평균비의 90% 신뢰구간 읽기")
save("lab13_hw1", c, marks={"95.82 %": 1, "0.563 to 1.304": 2, "outside 80-125%: 5 of 24": 3})
NOTES[c.n] = ("## 과제 정답\n\n"
              f"**과제 1.** 셀 {XO_CELL}의 결과(GMR 95.82%, 90% CI 89.71–102.36%)를 논문의 Results처럼 한 문장으로 "
              "옮기고, 이 결과가 말해 주지 않는 것을 두 가지 이상 쓰세요. 대상자마다 시험약 AUC ÷ 대조약 AUC를 구해 "
              "80–125% 밖인 사람이 몇 명인지도 확인하세요.")

c = nb.cell('''
counts = {"success": (186, 224, 190, 224),     # 제균 성공 수, 인원 (신약, 표준)
          "failure": (38, 224, 34, 224)}       # 실패 수 = 224 - 성공 수
margins = {"success": 0.75 / 0.85,             # 85%가 75%로: 성공의 비 한계
           "failure": 0.25 / 0.15}             # 15%가 25%로: 실패의 비 한계
for name, (x1, n1, x2, n2) in counts.items():
    lo, hi = confint_proportions_2indep(x1, n1, x2, n2,
                                        method="log", compare="ratio")
    se_log = np.sqrt(1 / x1 - 1 / n1 + 1 / x2 - 1 / n2)   # ln(비)의 SE
    print(f"{name}: ratio {x1 / n1 / (x2 / n2):.3f},"
          f" 95% CI {lo:.3f} to {hi:.3f}, margin {margins[name]:.3f},"
          f" SE(log) {se_log:.3f}")
''', title="과제 2 정답. 위험차 대신 비의 척도로 판정하기")
save("lab13_hw2", c, marks={"95% CI 0.903 to 1.062, margin 0.882": 1,
                            "95% CI 0.731 to 1.708, margin 1.667": 2,
                            "SE(log) 0.041": 3, "SE(log) 0.216": 4})
NOTES[c.n] = (f"**과제 2.** 셀 {PROP_CELL}의 본 시험(186/224 대 190/224)을 위험차 대신 비의 척도로 다시 판정하세요. "
              "`confint_proportions_2indep`에서 `compare=\"ratio\"`, `method=\"log\"`로 바꿔 제균 성공의 비와 실패의 비의 "
              "95% 신뢰구간을 구하고, 설계 때 가정한 제균율 85%에서 −10%p 한계가 각각 어떤 비의 한계가 되는지 계산해 "
              "결론을 비교하세요.")

c = nb.cell('''
print("Fisher exact P (difference = 0):",
      round(stats.fisher_exact([[48, 2], [50, 0]]).pvalue, 3))
print(tab[tab["trial"] == "small"].round(4))      # 셀의 Wald와 Newcombe 구간
for m in ["wald", "score"]:
    r = test_proportions_2indep(48, 50, 50, 50, value=-0.10, method=m,
                                compare="diff", alternative="larger",
                                correction=False)
    print(f"NI test, {m:5}: z = {r.statistic:.3f},"
          f" one-sided P = {r.pvalue:.4f}")
''', title="과제 3 정답. 소규모 시험을 '차이 없음'과 Wald 구간으로 판단한 분석")
save("lab13_hw3", c, marks={"Fisher exact P (difference = 0): 0.495": 1, "-0.0943": 2, "-0.1346": 3,
                            "wald : z = 2.165, one-sided P = 0.0152": 4,
                            "score: z = 1.414, one-sided P = 0.0786": 5})
NOTES[c.n] = ("**과제 3.** 동료가 소규모 시험(48/50 대 50/50)을 이렇게 요약했습니다. \"Fisher 정확검정에서 두 요법의 "
              "제균율 차이는 유의하지 않았다(P = 0.49). 위험차의 95% CI는 −9.4 ~ 1.4%p(Wald)로 하한이 한계 −10%p보다 "
              "크므로 새 요법은 표준 요법에 비열등하다. 따라서 두 요법의 효과는 같다.\" 이 결론에서 틀린 곳 세 가지를 "
              "찾아 왜 틀렸는지 설명하고, 바르게 고쳐 쓰세요.")

if DEBUG:
    print("\ncells:", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 13. 동등성·비열등성 검정",
    intro=("사회약학 연구방법 노트의 '실습 13. 동등성·비열등성 검정'에 딸린 노트북입니다. a 절은 13장 c 절의 HbA1c "
           "시험을 흉내 낸 모의자료로 평균 차이의 비열등성을, b 절은 13장 d 절의 제균 시험 요약자료로 비율 차이의 "
           "비열등성을, c 절은 공개 약동학 자료(Theoph)와 13장의 AUC 자료, 교차설계 모의자료로 생물학적 동등성을 "
           "계산합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 13에 있습니다."),
    skip=(ERR_CELL,), notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import warnings
    import numpy as np
    import pandas as pd
    from scipy import stats
    from statsmodels.stats.proportion import confint_proportions_2indep, test_proportions_2indep
    from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind

    ns = nb.ns
    new_, ref_ = ns["new"], ns["ref"]
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    f4 = lambda x: f"{x:.4f}"
    # 글과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 2: "모의 임상시험 자료 만들기", 3: "차이와 95% 신뢰구간", 4: "비열등성 검정 (단측)",
             6: "요약 통계로 13장 b·c 절의 숫자 재현", 7: "위험차의 Wald 구간과 Newcombe 구간",
             9: "score 방법이 실패하는 경우", 10: "statsmodels의 score(Farrington–Manning) 검정",
             11: "Farrington–Manning 검정을 직접 계산", 15: "기하평균비와 90% 신뢰구간 (대응 자료)",
             17: "2×2 교차설계 모의자료", 18: "교차설계 분산분석으로 90% 신뢰구간"}
    for k, tt in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, tt)
    # a 절
    ci_ = ns["ci"]
    same("sim CI", (f3(ci_.low), f3(ci_.high)), ("-0.063", "0.353"))
    same("margin to spare", f3(0.4 - ci_.high), "0.047")
    same("NI t, P, df", (f3(ns["ni"].statistic), f4(ns["ni"].pvalue), f"{ns['ni'].df:.1f}"),
         ("-2.414", "0.0081", "365.5"))
    same("sim difference rounded for the paper box", (f2(new_.mean()), f2(ref_.mean()), f"{ns['d']:.4f}"),
         ("-0.75", "-0.89", "0.1447"))
    sm_t, sm_p, sm_df = sm_ttest_ind(new_, ref_, alternative="smaller", usevar="unequal", value=0.4)
    same("statsmodels weightstats ttest_ind", (f3(sm_t), f4(sm_p)), ("-2.414", "0.0081"))
    ci95 = ns["ni"].confidence_interval()                 # 기본값 0.95(단측) = 양측 90%의 상한
    same("default one-sided 95% upper limit", f3(ci95.high + 0.4), "0.319")
    r2_ = stats.ttest_ind(new_, ref_, equal_var=False).confidence_interval(confidence_level=0.90)
    same("two-sided 90% upper = one-sided 95% upper", f3(r2_.high), "0.319")
    rng_ = np.random.default_rng(13)
    n13 = rng_.normal(loc=-0.74, scale=1.02, size=185)
    r13 = rng_.normal(loc=-0.83, scale=1.06, size=183)
    c13 = stats.ttest_ind(n13, r13, equal_var=False).confidence_interval()
    same("seed 13 tip", (f3(n13.mean() - r13.mean()), f3(c13.low), f3(c13.high)), ("0.259", "0.036", "0.482"))
    # 대응표: 13장 숫자와 같은지(요약 통계 셀) — 13장 c 절 0.09 (−0.12 to 0.30), P .002; b 절 −1.30 (−4.03 ~ 1.43), .004
    out5 = nb.cells[5].stdout
    assert "difference 0.09, SE 0.1085, df 365.1" in out5 and "t -2.86, one-sided P 0.0023" in out5, out5
    assert "difference -1.30, SE 1.3913, df 605.0" in out5 and "t 2.66, one-sided P 0.0040" in out5, out5
    n_ok += 2
    same("t critical for df 365.1", f3(stats.t.ppf(0.975, 365.1)), "1.966")
    # b 절
    tab_ = ns["tab"]
    same("main Wald, Newcombe", [f4(x) for x in tab_.loc[:1, ["lower", "upper"]].to_numpy().ravel()],
         ["-0.0859", "0.0501", "-0.0862", "0.0506"])
    same("small Wald, Newcombe", [f4(x) for x in tab_.loc[2:, ["lower", "upper"]].to_numpy().ravel()],
         ["-0.0943", "0.0143", "-0.1346", "0.0370"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci_sc = confint_proportions_2indep(186, 224, 190, 224, method="score", compare="diff")
    same("statsmodels score CI (%p)", (f2(ci_sc[0] * 100), f2(ci_sc[1] * 100)), ("-8.73", "5.10"))
    # statsmodels score 검정이 쓴 제약 비율: z = 2.309가 나오게 하는 비율을 거꾸로 찾음
    from scipy.optimize import brentq
    obs = 186 / 224 - 190 / 224

    def z_at(p2):
        p1 = p2 - 0.10
        return (obs + 0.10) / np.sqrt(p1 * (1 - p1) / 224 + p2 * (1 - p2) / 224)
    p2_sm = brentq(lambda p: z_at(p) - ns["fm"].statistic, 0.80, 0.8795)
    same("statsmodels restricted proportions (%)", (f2((p2_sm - 0.10) * 100), f2(p2_sm * 100)), ("77.52", "87.52"))
    ll = lambda p2: stats.binom.logpmf(186, 224, p2 - 0.10) + stats.binom.logpmf(190, 224, p2)
    same("log-likelihoods (= -negloglik of cell 11)", (f3(ll(p2_sm)), f3(ll(ns["p2t"]))), ("-8.042", "-8.008"))
    # c 절
    th = ns["theo"]
    s1 = th[th["Subject"] == 1].sort_values("Time")
    same("subject 1 peak and last", (f2(s1["Time"].iloc[s1["conc"].argmax()]), f2(s1["conc"].max()),
                                      f2(s1["Time"].iloc[-1]), f2(s1["conc"].iloc[-1])), ("1.12", "10.50", "24.37", "3.28"))
    same("subject 1 first trapezoid", (f2(s1["conc"].iloc[0]), f2(s1["conc"].iloc[1]), f2(s1["Time"].iloc[1]),
                                        f4((s1["conc"].iloc[0] + s1["conc"].iloc[1]) / 2 * s1["Time"].iloc[1])),
         ("0.74", "2.84", "0.25", "0.4475"))
    pk_ = ns["pk"]
    same("Tmax range", (f2(pk_["Tmax"].min()), f2(pk_["Tmax"].max())), ("0.63", "3.55"))
    same("AUC range", (f2(pk_["AUC_0_t"].min()), f2(pk_["AUC_0_t"].max())), ("73.78", "148.92"))
    same("ln 1.25", f4(np.log(1.25)), "0.2231")
    be_ = ns["be"]
    same("crossover subject 1", [f"{x:.0f}" for x in be_.loc[be_["subject"] == 1, "auc"]], ["4619", "8210"])
    same("MSE", f4(ns["fit"].mse_resid), "0.0177")
    # 과제 1
    rt_ = ns["ratio"]
    same("GMR of individual ratios = model GMR", f2(np.exp(np.log(rt_).mean()) * 100),
         f2(np.exp(ns["fit"].params["C(form)[T.T]"]) * 100))
    same("below 80%, above 125%", (int((rt_ < 0.8).sum()), int((rt_ > 1.25).sum())), (4, 1))
    # 과제 2
    same("success, failure counts", (224 - 186, 224 - 190), (38, 34))
    same("relative margins", (f"{1 - 0.75 / 0.85:.1%}", f"{0.25 / 0.15 - 1:.0%}"), ("11.8%", "67%"))
    # 과제 3
    same("Fisher P rounded", f2(stats.fisher_exact([[48, 2], [50, 0]]).pvalue), "0.49")
    same("FM by hand at the boundary p2 = 1", f3((0.96 - 1.0 + 0.10) / np.sqrt(0.9 * 0.1 / 50)), "1.414")
    same("Wald SE small trial", f4(np.sqrt(0.96 * 0.04 / 50)), "0.0277")
    same("t critical for df 605.0", f3(stats.t.ppf(0.975, 605.0)), "1.964")
    lo_xo = np.exp(ns["fit"].conf_int(alpha=0.10).loc["C(form)[T.T]", 0])
    same("largest reduction inside the 90% CI", f"{1 - lo_xo:.0%}", "10%")
    se_s = np.sqrt(1 / 186 - 1 / 224 + 1 / 190 - 1 / 224)
    se_f = np.sqrt(1 / 38 - 1 / 224 + 1 / 34 - 1 / 224)
    same("failure SE / success SE", f"{se_f / se_s:.1f}", "5.2")
    same("small trial: new rate, difference", (f"{48 / 50:.1%}", f"{48 / 50 - 1:.1%}"), ("96.0%", "-4.0%"))
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab13.html passed")
