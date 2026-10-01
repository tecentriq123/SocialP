"""실습 13 · 동등성·비열등성 검정 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab13.py
      (LABDEBUG=1 prints every cell's text output, for writing marks)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab13")
DEBUG = os.environ.get("LABDEBUG")


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

c = nb.cell('''
margin = 0.4
ni = stats.ttest_ind(new - margin, ref, equal_var=False,
                     alternative="less")
print(ni)
se = np.sqrt(new.var(ddof=1) / new.size + ref.var(ddof=1) / ref.size)
print(f"SE = {se:.4f}, (d - margin) / SE = {(d - margin) / se:.3f}")
''', title="비열등성 검정 (단측)")
save("lab13_ni", c, marks={"statistic=np.float64(-2.41": 1, "pvalue=np.float64(0.0081": 2,
                           "SE = 0.1058": 3})

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

c = nb.cell('''
from statsmodels.stats.proportion import test_proportions_2indep

fm = test_proportions_2indep(186, 224, 190, 224, value=-0.10,
                             method="score", compare="diff",
                             alternative="larger", correction=False)
print(f"z = {fm.statistic:.3f}, one-sided P = {fm.pvalue:.4f}")
''', title="statsmodels의 score(Farrington–Manning) 검정")
save("lab13_fm_sm", c, marks={"z = 2.309": 1, "P = 0.0105": 2})

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

if DEBUG:
    print("\ncells:", len(nb.cells))
