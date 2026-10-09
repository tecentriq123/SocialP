"""실습 19 · 메타분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab19.py && python3 gen/lab_lab19.py
(NOMARK=1 prints every cell's raw output instead of placing marks.)

The data (pub/data/acs_trials.csv, acs_trials_subgroup.csv) are the chapter-19 example exported by gen/data_lab19.py.
The block at the end checks the lab's numbers against gen/nums_ch19.py compute() and the values printed in ch19.html.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("lab19")


def render(c, marks=None, dfmarks=None):
    """text marks -> first text <pre> only; table marks -> DataFrame only."""
    h = nb.html(c)
    if marks:
        i = h.index('<div class="cell-out">')
        j = h.index("<pre>", i)
        k = h.index("</pre>", j)
        h = h[:j] + _mark(h[j:k], marks) + h[k:]
    if dfmarks:
        j = h.index('<div class="df-wrap">')
        k = h.index("</table>", j)
        seg = h[j:k]
        for sub, n in dfmarks.items():
            if sub.startswith("<td>"):          # whole-cell match (labkit._mark escapes its key, so do it here)
                i = seg.index(sub) + len(sub) - len("</td>")
                seg = seg[:i] + f'<span class="mk">{n}</span>' + seg[i:]
            else:
                seg = _mark(seg, {sub: n})
        h = h[:j] + seg + h[k:]
    return h


def save(name, c, marks=None, dfmarks=None):
    if os.environ.get("NOMARK"):
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        for w_ in c.warns:
            print("  [warn]", w_)
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
tr = pd.read_csv(BASE + "acs_trials.csv", storage_options=UA)
print(tr.shape)
tr
''', title="시험 수준 자료 불러오기")
save("lab19_load", c, marks={"(10, 12)": 1},
     dfmarks={"some concerns": 2, "<td>122</td>": 3, "<td>0</td>": 4})

c = nb.cell('''
sub = pd.read_csv(BASE + "acs_trials_subgroup.csv",
                  storage_options=UA)
print(sub.shape)
sub
''', title="하위군 자료 불러오기", max_rows=12)
save("lab19_loadsub", c, marks={"(12, 6)": 1}, dfmarks={"<td>341</td>": 2, "<td>279</td>": 3})

c = nb.cell('''
cols = ["n_x", "mace_x", "n_pbo", "mace_pbo"]
tot = tr[cols].sum()
print(tot.to_string())
r_x = tot["mace_x"] / tot["n_x"]
r_p = tot["mace_pbo"] / tot["n_pbo"]
print(f"summed risk: {r_x:.1%} vs {r_p:.1%}")
print(f"crude OR   : {(r_x / (1 - r_x)) / (r_p / (1 - r_p)):.2f}")

short = tr[tr["followup_mo"] < 12]          # 추적이 짧은 2상 시험
sh_x = short["n_x"].sum() / tot["n_x"]
sh_p = short["n_pbo"].sum() / tot["n_pbo"]
print(f"short trials: {sh_x:.1%} of drug X, {sh_p:.1%} of placebo")
''', title="사건 수를 그냥 더하면 안 되는 이유")
save("lab19_naive", c, marks={"n_x         16850": 1, "7.5% vs 9.3%": 2, "0.79": 3, "22.8% of drug X": 4})

# ---------------------------------------------------------------- 나. 효과의 통합
c = nb.cell('''
t = tr[tr["trial"] == "Trial I"].iloc[0]
a, c = t["mace_x"], t["mace_pbo"]           # 사건이 생긴 사람
b, d = t["n_x"] - a, t["n_pbo"] - c         # 생기지 않은 사람
or_i = (a * d) / (b * c)
y_i = np.log(or_i)                          # 로그 오즈비
v_i = 1/a + 1/b + 1/c + 1/d                 # 그 분산
se_i = np.sqrt(v_i)
Z = stats.norm.ppf(0.975)                   # 1.959964
print("a, b, c, d =", a, b, c, d)
print(f"OR {or_i:.3f}, log OR {y_i:.4f},"
      f" var {v_i:.4f}, SE {se_i:.4f}")
print(f"95% CI {np.exp(y_i - Z*se_i):.2f}-{np.exp(y_i + Z*se_i):.2f}")
print(f"weight = 1/var = {1 / v_i:.1f}")
''', title="시험 하나의 로그 오즈비와 표준오차 (Trial I)")
save("lab19_trialI", c, marks={"122 1678 167 1623": 1, "OR 0.707": 2, "log OR -0.3473": 3, "SE 0.1241": 4,
                               "0.55-0.90": 5, "64.9": 6})

c = nb.cell('''
def log_or(e1, n1, e0, n0):
    """로그 오즈비와 분산 (0인 칸이 있는 시험만 0.5씩 더함)"""
    a, c = e1.astype(float), e0.astype(float)
    b, d = n1 - a, n0 - c
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    a, b, c, d = [x + 0.5 * zero for x in (a, b, c, d)]
    return np.log(a * d / (b * c)), 1/a + 1/b + 1/c + 1/d

es = tr[["trial"]].copy()
es["yi"], es["vi"] = log_or(tr["mace_x"], tr["n_x"],
                            tr["mace_pbo"], tr["n_pbo"])
es["se"] = np.sqrt(es["vi"])
es["or"] = np.exp(es["yi"])
es["lo"] = np.exp(es["yi"] - Z * es["se"])
es["hi"] = np.exp(es["yi"] + Z * es["se"])
es.round(3)
''', title="열 개 시험의 효과크기 표")
save("lab19_es", c, dfmarks={"<td>0.686</td>": 1, "<td>0.063</td>": 2, "<td>-0.347</td>": 3, "<td>1.279</td>": 4})

c = nb.cell('''
w = 1 / es["vi"]                            # 역분산 가중치
fe = np.sum(w * es["yi"]) / np.sum(w)       # 가중평균
se_fe = 1 / np.sqrt(np.sum(w))
print(f"sum w = {np.sum(w):.1f}, sum wy = {np.sum(w * es['yi']):.1f}")
print(f"log OR {fe:.4f}, SE {se_fe:.4f}")
print(f"OR {np.exp(fe):.2f} ({np.exp(fe - Z*se_fe):.2f}"
      f"-{np.exp(fe + Z*se_fe):.2f})")
''', title="고정효과 모형 (역분산 가중평균)")
save("lab19_fixed", c, marks={"sum w = 596.4": 1, "sum wy = -101.5": 2, "log OR -0.1702": 3, "SE 0.0409": 4,
                              "OR 0.84 (0.78-0.91)": 5})

c = nb.cell('''
k = len(es)                                 # 시험 수
Q = np.sum(w * (es["yi"] - fe)**2)
C = np.sum(w) - np.sum(w**2) / np.sum(w)
tau2 = max(0, (Q - (k - 1)) / C)            # DerSimonian-Laird
ws = 1 / (es["vi"] + tau2)                  # 무작위효과 가중치
re = np.sum(ws * es["yi"]) / np.sum(ws)
se_re = 1 / np.sqrt(np.sum(ws))
p_re = 2 * stats.norm.sf(abs(re / se_re))
print(f"Q {Q:.2f}, C {C:.1f}, tau2 {tau2:.4f},"
      f" sum w* = {np.sum(ws):.1f}")
print(f"log OR {re:.4f}, SE {se_re:.4f},"
      f" z {re / se_re:.2f}, P {p_re:.4f}")
print(f"OR {np.exp(re):.2f} ({np.exp(re - Z*se_re):.2f}"
      f"-{np.exp(re + Z*se_re):.2f})")

es["w_fe"] = 100 * w / np.sum(w)            # 가중치 비율(%)
es["w_re"] = 100 * ws / np.sum(ws)
es[["trial", "se", "w_fe", "w_re"]].round(3)
''', title="연구 간 분산과 무작위효과 모형")
save("lab19_random", c, marks={"tau2 0.0098": 1, "sum w* = 266.7": 2, "log OR -0.1921": 3, "z -3.14": 4,
                               "P 0.0017": 5, "OR 0.83 (0.73-0.93)": 6},
     dfmarks={"<td>0.779</td>": 7, "<td>42.712</td>": 8, "<td>27.263</td>": 9})

c = nb.cell('''
def pool(y, v):
    """역분산 고정효과(fe)와 DerSimonian-Laird 무작위효과(re)"""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k, w = len(y), 1 / v
    fe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - fe)**2)
    C = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0, (Q - (k - 1)) / C)
    ws = 1 / (v + tau2)
    re = np.sum(ws * y) / np.sum(ws)
    return {"k": k, "fe": fe, "se_fe": 1 / np.sqrt(np.sum(w)),
            "re": re, "se_re": 1 / np.sqrt(np.sum(ws)),
            "Q": Q, "I2": max(0, (Q - (k - 1)) / Q), "tau2": tau2}

def orci(est, se):
    """로그 척도의 추정값과 표준오차 -> 'OR (하한-상한)'"""
    lo, hi = np.exp(est - Z * se), np.exp(est + Z * se)
    return f"{np.exp(est):.2f} ({lo:.2f}-{hi:.2f})"

m = pool(es["yi"], es["vi"])
print("fixed  :", orci(m["fe"], m["se_fe"]))
print("random :", orci(m["re"], m["se_re"]))
''', title="통합 계산을 함수로 묶기")
save("lab19_pool", c, marks={"0.84 (0.78-0.91)": 1, "0.83 (0.73-0.93)": 2})

c = nb.cell('''
from statsmodels.stats.meta_analysis import (
    effectsize_2proportions, combine_effects)

cols = ["mace_x", "n_x", "mace_pbo", "n_pbo"]
args = [tr[c].to_numpy() for c in cols]     # 열 네 개를 배열로
eff, var = effectsize_2proportions(*args, statistic="odds-ratio")
res = combine_effects(eff, var, method_re="chi2",
                      row_names=list(tr["trial"]))
print(res.summary_frame().round(4))
print(round(res.q, 4), round(res.i2, 4), round(res.tau2, 6))
print("default (iterated) tau2:",
      round(combine_effects(eff, var).tau2, 6))
''', title="statsmodels로 같은 값 확인하기")
save("lab19_sm", c, marks={"Trial I           -0.3473": 1, "0.4271  0.2726": 2,
                           "fixed effect      -0.1702  0.0409": 3, "random effect     -0.1921  0.0612": 4,
                           "random effect wls -0.1921  0.0591": 5, "13.2849 0.3225 0.009827": 6,
                           "default (iterated) tau2:": 7})

c = nb.cell('''
from statsmodels.stats.contingency_tables import StratifiedTable

tables = [[[r.mace_x, r.n_x - r.mace_x],
           [r.mace_pbo, r.n_pbo - r.mace_pbo]]
          for r in tr.itertuples()]
mh = StratifiedTable(tables)
lo, hi = mh.oddsratio_pooled_confint()
print(f"Mantel-Haenszel OR {mh.oddsratio_pooled:.2f}"
      f" ({lo:.2f}-{hi:.2f})")
''', title="Mantel–Haenszel 방법")
save("lab19_mh", c, marks={"0.84 (0.78-0.91)": 1})

c = nb.cell('''
bl = tr[["trial"]].copy()
bl["yi"], bl["vi"] = log_or(tr["bleed_x"], tr["n_x"],
                            tr["bleed_pbo"], tr["n_pbo"])
mb = pool(bl["yi"], bl["vi"])
print("Trial C:", orci(bl.loc[2, "yi"], np.sqrt(bl.loc[2, "vi"])))
print("pooled :", orci(mb["re"], mb["se_re"]),
      f"Q {mb['Q']:.2f}, I2 {mb['I2']:.0%}, tau2 {mb['tau2']:.4f}")

cols_b = ["bleed_x", "n_x", "bleed_pbo", "n_pbo"]
args_b = [tr[c].to_numpy() for c in cols_b]
e0, v0 = effectsize_2proportions(*args_b, statistic="odds-ratio")
print("statsmodels, Trial C, no correction:", e0[2], v0[2])
e5, v5 = effectsize_2proportions(*args_b, statistic="odds-ratio",
                                 zero_correction=0.5)
m5 = pool(e5, v5)
print("statsmodels, zero_correction=0.5  :",
      orci(m5["re"], m5["se_re"]))
''', title="주요 출혈과 0건인 칸")
save("lab19_bleed", c, marks={"3.55 (0.18-69.32)": 1, "2.28 (1.90-2.74)": 2, "I2 0%": 3, "inf inf": 4,
                              "2.24 (1.87-2.68)": 5})

c = nb.cell('''
pi_half = stats.t.ppf(0.975, k - 2) * np.sqrt(tau2 + se_re**2)
pi = np.exp([re - pi_half, re + pi_half])   # 예측구간(다 절)
ypos = np.arange(k, 0, -1)                  # 10, 9, ..., 1

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.hlines(ypos, es["lo"], es["hi"], color="black", lw=1)
ax.scatter(es["or"], ypos, s=es["w_re"] * 12, marker="s",
           color="C0", zorder=3)
ax.fill(np.exp([re - Z*se_re, re, re + Z*se_re, re]),
        [0, 0.3, 0, -0.3], color="black")   # 마름모
ax.hlines(-1, pi[0], pi[1], color="gray", lw=3)
ax.axvline(1, color="gray", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlim(0.2, 6)
ax.set_xticks([0.25, 0.5, 1, 2, 4])
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_yticks(list(ypos) + [0, -1])
ax.set_yticklabels(list(es["trial"]) + ["Random effects",
                                         "Prediction interval"])
for y, i in zip(ypos, es.index):
    ax.text(6.5, y, orci(es["yi"][i], es["se"][i])
            + f"  {es['w_re'][i]:.1f}%", va="center", fontsize=9)
ax.text(6.5, 0, orci(re, se_re), va="center", fontsize=9)
ax.text(6.5, -1, f"{pi[0]:.2f}-{pi[1]:.2f}", va="center", fontsize=9)
ax.set_xlabel("Odds ratio (95% CI, log scale)")
plt.tight_layout()
''', title="포레스트 플롯 그리기")
save("lab19_forest", c)

# ---------------------------------------------------------------- 다. 이질성과 하위군 분석
c = nb.cell('''
p_q = stats.chi2.sf(m["Q"], m["k"] - 1)
tau = np.sqrt(m["tau2"])
tcrit = stats.t.ppf(0.975, m["k"] - 2)
half = tcrit * np.sqrt(m["tau2"] + m["se_re"]**2)
print(f"Q = {m['Q']:.2f}, df = {m['k'] - 1}, P = {p_q:.2f}")
print(f"I2 = {m['I2']:.0%}")
print(f"tau2 = {m['tau2']:.4f}, tau = {tau:.3f}")
print(f"exp(mean +/- 1.96 tau): {np.exp(m['re'] - Z*tau):.2f}"
      f"-{np.exp(m['re'] + Z*tau):.2f}")
print(f"t({m['k'] - 2}) = {tcrit:.3f}, half-width = {half:.3f}")
print(f"95% prediction interval: {np.exp(m['re'] - half):.2f}"
      f"-{np.exp(m['re'] + half):.2f}")
''', title="이질성 지표와 예측구간")
save("lab19_het", c, marks={"Q = 13.28, df = 9, P = 0.15": 1, "I2 = 32%": 2, "tau = 0.099": 3,
                            "0.68-1.00": 4, "t(8) = 2.306": 5, "0.63-1.08": 6})

c = nb.cell('''
sub["yi"], sub["vi"] = log_or(sub["mace_x"], sub["n_x"],
                              sub["mace_pbo"], sub["n_pbo"])
rows = []
for g in ["STEMI", "NSTE-ACS"]:
    s = sub[sub["subgroup"] == g]           # 그 하위군의 6줄
    r = pool(s["yi"], s["vi"])
    r["subgroup"] = g
    r["drug X"] = f"{s['mace_x'].sum()}/{s['n_x'].sum()}"
    r["placebo"] = f"{s['mace_pbo'].sum()}/{s['n_pbo'].sum()}"
    r["OR (95% CI)"] = orci(r["re"], r["se_re"])
    r["P"] = 2 * stats.norm.sf(abs(r["re"] / r["se_re"]))
    rows.append(r)
sg = pd.DataFrame(rows).set_index("subgroup")
sg[["k", "drug X", "placebo", "OR (95% CI)", "P", "I2"]].round(4)
''', title="하위군별 통합")
save("lab19_subgroup", c, dfmarks={"528/7006": 1, "0.78 (0.69-0.89)": 2, "0.92 (0.81-1.06)": 3, "0.2393": 4,
                                   "0.0312": 5})

c = nb.cell('''
d = sg.loc["STEMI", "re"] - sg.loc["NSTE-ACS", "re"]
se_d = np.sqrt(sg.loc["STEMI", "se_re"]**2
               + sg.loc["NSTE-ACS", "se_re"]**2)
z = d / se_d
print(f"difference {d:.3f}, SE {se_d:.3f}, z {z:.2f}")
print(f"Q between = z^2 = {z**2:.2f} (df = 1)")
print(f"P for interaction = {2 * stats.norm.sf(abs(z)):.3f}")
print("ratio of ORs:", orci(d, se_d))

d_fe = sg.loc["STEMI", "fe"] - sg.loc["NSTE-ACS", "fe"]
se_fe_d = np.sqrt((sg["se_fe"]**2).sum())
print("fixed-effect subtotals: P =",
      round(2 * stats.norm.sf(abs(d_fe / se_fe_d)), 3))
''', title="하위군 간 차이 검정")
save("lab19_interaction", c, marks={"difference -0.164, SE 0.094, z -1.74": 1, "3.03": 2,
                                    "P for interaction = 0.082": 3, "0.85 (0.71-1.02)": 4, "P = 0.043": 5})

c = nb.cell('''
import statsmodels.formula.api as smf

es["stemi10"] = (tr["stemi_pct"] - 50) / 10   # 10%p 단위, 50%가 0
m0 = smf.wls("yi ~ stemi10", data=es, weights=w).fit()
QE = np.sum(w * m0.resid**2)                  # 직선 주위의 흩어짐
X = m0.model.exog                             # 10행 x [1, stemi10]
W = np.diag(w)                                # 대각선이 가중치인 행렬
A = np.linalg.inv(X.T @ W @ X)
C_reg = np.trace(W) - np.trace(A @ X.T @ W @ W @ X)
tau2_res = max(0, (QE - (k - 2)) / C_reg)     # 남은 연구 간 분산

mr = smf.wls("yi ~ stemi10", data=es,
             weights=1 / (es["vi"] + tau2_res)).fit(
                 cov_type="fixed scale")
b, se_b = mr.params["stemi10"], mr.bse["stemi10"]
print(f"QE {QE:.2f}, residual tau2 {tau2_res:.4f}")
p_b = 2 * stats.norm.sf(abs(b / se_b))
print(f"slope {b:.3f} (SE {se_b:.3f}), z {b / se_b:.2f}, P {p_b:.2f}")
print("ratio of ORs per 10 %p:", orci(b, se_b))
''', title="메타회귀 (STEMI 환자 비율)")
save("lab19_metareg", c, marks={"residual tau2 0.0044": 1, "slope -0.094 (SE 0.058)": 2, "P 0.11": 3,
                                "0.91 (0.81-1.02)": 4})

# ---------------------------------------------------------------- 라. 깔때기 그림과 민감도 분석
c = nb.cell('''
s_max = 0.75
fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(es["yi"], es["se"], zorder=3)
ax.plot([fe - Z*s_max, fe, fe + Z*s_max], [s_max, 0, s_max],
        ls="--", color="gray")              # 95%가 들어올 범위
ax.axvline(fe, color="black", lw=1)         # 고정효과 통합값
for x, y, name in zip(es["yi"], es["se"], es["trial"]):
    ax.annotate(name[-1], (x, y), xytext=(5, 3),
                textcoords="offset points")
ax.set_ylim(s_max, 0)                       # 위가 정밀한 시험
ax.set_xticks(np.log([0.25, 0.5, 1, 2, 4]))
ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
ax.set_xlabel("Odds ratio (log scale)")
ax.set_ylabel("Standard error")
plt.tight_layout()
''', title="깔때기 그림")
save("lab19_funnel", c)

c = nb.cell('''
eg = smf.wls("yi ~ se", data=es, weights=1 / es["vi"]).fit()
print(f"slope {eg.params['se']:.3f} (SE {eg.bse['se']:.3f}),"
      f" t {eg.tvalues['se']:.2f}, df {eg.df_resid:.0f},"
      f" P {eg.pvalues['se']:.2f}")

es["snd"] = es["yi"] / es["se"]             # 표준화한 효과
es["prec"] = 1 / es["se"]                   # 정밀도
eg2 = smf.ols("snd ~ prec", data=es).fit()
print(f"original form: intercept {eg2.params['Intercept']:.3f},"
      f" P {eg2.pvalues['Intercept']:.2f}")
''', title="Egger 검정")
save("lab19_egger", c, marks={"slope -0.505 (SE 0.667)": 1, "t -0.76, df 8": 2, "P 0.47": 3,
                              "intercept -0.505": 4})

c = nb.cell('''
rows = []
for i in es.index:
    rest = es.drop(index=i)                 # i번 시험을 뺀 9개
    r = pool(rest["yi"], rest["vi"])
    rows.append({"left out": es.loc[i, "trial"],
                 "OR (95% CI)": orci(r["re"], r["se_re"]),
                 "OR": round(np.exp(r["re"]), 3),
                 "I2 (%)": int(round(100 * r["I2"]))})
loo = pd.DataFrame(rows)
print(f"range: {loo['OR'].min():.2f}-{loo['OR'].max():.2f}")
loo
''', title="한 시험씩 빼기")
save("lab19_loo", c, marks={"0.79-0.85": 1},
     dfmarks={"0.79 (0.72-0.87)": 2, "<td>0</td>": 3, "0.81 (0.69-0.96)": 4})

c = nb.cell('''
low = tr["rob"] == "low"                    # 편향 위험 '낮음' 8개
m_low = pool(es.loc[low, "yi"], es.loc[low, "vi"])
q_hk = np.sum(ws * (es["yi"] - re)**2) / (k - 1)
se_hk = se_re * np.sqrt(q_hk)               # Hartung-Knapp 표준오차
t9 = stats.t.ppf(0.975, k - 1)
print("main (random effects):", orci(m["re"], m["se_re"]))
print("fixed effect         :", orci(m["fe"], m["se_fe"]))
print(f"low risk of bias, k={m_low['k']}:",
      orci(m_low["re"], m_low["se_re"]))
print(f"Hartung-Knapp        : SE {se_hk:.4f}, 95% CI"
      f" {np.exp(re - t9*se_hk):.2f}-{np.exp(re + t9*se_hk):.2f}")
''', title="그 밖의 민감도 분석")
save("lab19_sens", c, marks={"0.84 (0.78-0.91)": 1, "0.83 (0.73-0.94)": 2, "SE 0.0591": 3, "0.72-0.94": 4})

c = nb.cell('''
def risk_with(p0, OR):
    odds = p0 / (1 - p0) * OR               # 위험 -> 오즈, x 오즈비
    return odds / (1 + odds)                # 오즈 -> 위험

def absolute(p0, est, se):
    ORs = np.exp([est, est - Z * se, est + Z * se])
    rd = risk_with(p0, ORs) - p0            # 위험차 [추정값, 양 끝]
    lo, hi = sorted(1 / np.abs(rd[1:]))     # NNT의 양 끝
    print(f"  risk {p0:.1%} -> {risk_with(p0, ORs[0]):.2%}")
    print("  RD %p   : {:.2f} ({:.2f}, {:.2f})".format(*(100 * rd)))
    print("  per 1000: {:.0f} ({:.0f}, {:.0f})".format(*(1000 * rd)))
    print(f"  NNT/NNH : {1 / abs(rd[0]):.0f} ({lo:.0f}-{hi:.0f})")

y12 = tr[tr["followup_mo"] == 12]           # 12개월 추적 시험 D, H, I
n_ctl = y12["n_pbo"].sum()
p0 = round(y12["mace_pbo"].sum() / n_ctl, 3)
pb0 = round(y12["bleed_pbo"].sum() / n_ctl, 3)
print("MACE, placebo:", y12["mace_pbo"].sum(), "/", n_ctl)
absolute(p0, m["re"], m["se_re"])
print("Major bleeding, placebo:", y12["bleed_pbo"].sum(), "/", n_ctl)
absolute(pb0, mb["re"], mb["se_re"])
''', title="NNT와 NNH")
save("lab19_nnt", c, marks={"662 / 7870": 1, "8.4% -> 7.04%": 2, "-1.36 (-2.11, -0.54)": 3,
                            "-14 (-21, -5)": 4, "73 (47-186)": 5, "1.2% -> 2.70%": 6,
                            "15 (11, 20)": 7, "67 (50-94)": 8})

# ---------------------------------------------------------------- 과제 (정답 셀)
c = nb.cell('''
es["phase"] = tr["phase"]
rows = []
for ph in [2, 3]:
    s = es[es["phase"] == ph]
    r = pool(s["yi"], s["vi"])
    r["phase"] = ph
    r["OR (95% CI)"] = orci(r["re"], r["se_re"])
    rows.append(r)
pg = pd.DataFrame(rows).set_index("phase")
d = pg.loc[2, "re"] - pg.loc[3, "re"]
se_d = np.sqrt((pg["se_re"]**2).sum())
print("P for subgroup difference:",
      round(2 * stats.norm.sf(abs(d / se_d)), 2))
pg[["k", "OR (95% CI)", "I2", "tau2"]].round(3)
''', title="과제 1 정답. 임상시험 단계에 따른 하위군 분석")
save("lab19_hw1", c, marks={"0.36": 1}, dfmarks={"0.73 (0.57-0.95)": 2, "0.85 (0.73-0.98)": 3, "0.626": 4})

c = nb.cell('''
r1 = tr["mace_x"] / tr["n_x"]               # 약물 X군의 위험
r0 = tr["mace_pbo"] / tr["n_pbo"]           # 위약군의 위험

y_rr = np.log(r1 / r0)                      # 로그 상대위험도
v_rr = (1/tr["mace_x"] - 1/tr["n_x"]
        + 1/tr["mace_pbo"] - 1/tr["n_pbo"])
m_rr = pool(y_rr, v_rr)
print("RR:", orci(m_rr["re"], m_rr["se_re"]), f"I2 {m_rr['I2']:.0%}")

y_rd = r1 - r0                              # 위험차 (로그 없음)
v_rd = r1 * (1 - r1) / tr["n_x"] + r0 * (1 - r0) / tr["n_pbo"]
m_rd = pool(y_rd, v_rd)
lo = m_rd["re"] - Z * m_rd["se_re"]
hi = m_rd["re"] + Z * m_rd["se_re"]
print(f"RD: {100 * m_rd['re']:.2f} ({100 * lo:.2f} to {100 * hi:.2f})"
      f" %p, I2 {m_rd['I2']:.0%}")
''', title="과제 2 정답. 상대위험도와 위험차로 통합")
save("lab19_hw2", c, marks={"0.84 (0.75-0.94)": 1, "-1.24 (-2.02 to -0.46)": 2, "I2 34%": 3})

c = nb.cell('''
keep = bl["trial"] != "Trial C"
m_noC = pool(bl.loc[keep, "yi"], bl.loc[keep, "vi"])
tables_b = [[[r.bleed_x, r.n_x - r.bleed_x],
             [r.bleed_pbo, r.n_pbo - r.bleed_pbo]]
            for r in tr.itertuples()]
mh_b = StratifiedTable(tables_b)
lo, hi = mh_b.oddsratio_pooled_confint()
print("0.5 added to Trial C:", orci(mb["re"], mb["se_re"]))
print("Trial C excluded    :", orci(m_noC["re"], m_noC["se_re"]))
print(f"Mantel-Haenszel     : {mh_b.oddsratio_pooled:.2f}"
      f" ({lo:.2f}-{hi:.2f})")

wb = 1 / bl["vi"]                           # tau2 = 0: 두 모형 공통
share = 100 * wb / wb.sum()
print(f"weight of Trial C: {share[2]:.1f}%,"
      f" four phase 3 trials: {share[tr['phase'] == 3].sum():.1f}%")
''', title="과제 3 정답. 주요 출혈의 민감도 분석")
save("lab19_hw3", c, marks={"2.28 (1.90-2.74)": 1, "2.28 (1.90-2.73)": 2, "2.31 (1.93-2.77)": 3,
                            "0.4%": 4, "94.2%": 5})

print("lab19: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- 노트북
nb.save_ipynb(
    "실습 19. 메타분석",
    intro=("사회약학 연구방법 노트의 '실습 19. 메타분석'에 나오는 셀을 차례로 모은 노트북입니다. "
           "19장의 예제(급성관상동맥증후군 뒤 약물 X 대 위약, 가상의 무작위배정 시험 10개)로 시험별 로그 오즈비와 표준오차, "
           "고정효과·무작위효과 통합, 이질성과 예측구간, 하위군 분석과 메타회귀, 깔때기 그림과 Egger 검정, 민감도 분석, "
           "NNT·NNH를 numpy와 pandas로 직접 계산하고 statsmodels로 확인합니다. 셀을 위에서부터 차례로 실행하세요. "
           "자료는 사이트가 만든 가상 자료이며 인터넷 연결이 필요합니다."),
    notes={
        1: "## 가. 실습 데이터 준비\n\n시험 한 개가 한 줄인 표와 시험 × 하위군이 한 줄인 표를 읽습니다.",
        4: "## 나. 효과의 통합\n\n시험 하나의 계산에서 시작해 고정효과·무작위효과 통합, statsmodels 확인, 포레스트 플롯까지 갑니다.",
        13: "## 다. 이질성과 하위군 분석\n\nQ, I², τ², 예측구간을 읽고 하위군 분석과 메타회귀를 합니다.",
        17: "## 라. 깔때기 그림과 민감도 분석\n\n깔때기 그림, Egger 검정, 한 시험씩 빼기, 그리고 NNT·NNH를 구합니다.",
        22: ("# 과제 정답\n\n먼저 스스로 풀어 본 뒤 실행하세요.\n\n**과제 1.** 임상시험 단계(2상, 3상)로 하위군 분석을 하고 "
             "하위군 간 차이 검정의 P를 구합니다."),
        23: "**과제 2.** 효과 지표를 상대위험도와 위험차로 바꿔 무작위효과 모형으로 통합합니다.",
        24: ("**과제 3.** 주요 출혈에서 Trial C를 뺀 결과와 Mantel–Haenszel 결과를 0.5를 더한 결과와 견주고, "
             "Trial C와 3상 시험 네 개의 가중치 비율을 구합니다."),
    })

# ---------------------------------------------------------------- 대조 블록: 본문(19장)과 nums_ch19.compute()의 값과 맞춘다
if __name__ == "__main__":
    import nums_ch19 as N19
    ns = nb.ns
    R = N19.compute()
    E, B = R["eff"], R["bleed"]
    es_, m_, mb_, sg_ = ns["es"], ns["m"], ns["mb"], ns["sg"]
    orci_, pool_ = ns["orci"], ns["pool"]
    n_ok = 0

    def close(a, b, tol=1e-9, what=""):
        global n_ok
        assert abs(float(a) - float(b)) < tol, (what, a, b)
        n_ok += 1

    def same(a, b, what=""):
        global n_ok
        assert a == b, (what, a, b)
        n_ok += 1

    tr_ = ns["tr"]
    same(int(tr_["n_x"].sum() + tr_["n_pbo"].sum()), 31725, "N")
    assert np.allclose(es_["yi"], E["y"]) and np.allclose(es_["vi"], E["v"]); n_ok += 1
    for key in ("fe", "re", "Q", "I2"):
        close(m_[key], E[key], what=key)
    close(m_["tau2"], E["t2"])
    close(m_["se_fe"], E["fe_se"]); close(m_["se_re"], E["re_se"])
    close(ns["fe"], E["fe"]); close(ns["re"], E["re"]); close(ns["tau2"], E["t2"]); close(ns["p_re"], E["re_p"])
    assert np.allclose(es_["w_fe"], E["wfe"] * 100) and np.allclose(es_["w_re"], E["wre"] * 100); n_ok += 1
    same(orci_(m_["fe"], m_["se_fe"]), "0.84 (0.78-0.91)"); same(orci_(m_["re"], m_["se_re"]), "0.83 (0.73-0.93)")
    close(ns["res"].tau2, E["t2"]); close(ns["res"].q, E["Q"]); close(ns["res"].i2, E["I2"])
    close(ns["mh"].oddsratio_pooled, R["mh_eff"][0])
    close(ns["pi"][0], E["pi_or"][0]); close(ns["pi"][1], E["pi_or"][1])
    close(np.exp(m_["re"] - ns["half"]), E["pi_or"][0])
    close(ns["p_q"], E["pQ"]); close(ns["tcrit"], E["tcrit"])
    # bleeding / zero cell
    close(mb_["re"], B["re"]); close(mb_["se_re"], B["re_se"]); close(mb_["Q"], B["Q"])
    same(orci_(mb_["re"], mb_["se_re"]), "2.28 (1.90-2.74)")
    close(ns["m5"]["re"], R["bleed_all05"]["re"])
    # subgroups
    close(sg_.loc["STEMI", "re"], R["sub"]["S"]["re"]); close(sg_.loc["NSTE-ACS", "re"], R["sub"]["N"]["re"])
    close(sg_.loc["STEMI", "I2"], R["sub"]["S"]["I2"]); close(sg_.loc["NSTE-ACS", "I2"], R["sub"]["N"]["I2"])
    close(ns["z"], R["sub"]["zdiff"]); close(ns["z"] ** 2, R["sub"]["Qb"])
    d_sg = sg_.loc["STEMI", "re"] - sg_.loc["NSTE-ACS", "re"]      # (the notebook's d, se_d are reused by 과제 1)
    se_sg = np.sqrt(sg_.loc["STEMI", "se_re"] ** 2 + sg_.loc["NSTE-ACS", "se_re"] ** 2)
    same(orci_(d_sg, se_sg), "0.85 (0.71-1.02)"); close(2 * N19.st.norm.sf(abs(d_sg / se_sg)), R["sub"]["pint"])
    # meta-regression
    MR = R["mr"]
    close(ns["b"], MR["b"][1]); close(ns["se_b"], MR["se"][1]); close(ns["tau2_res"], MR["t2"]); close(ns["p_b"], MR["p"][1])
    close(ns["QE"], MR["QE"])
    # Egger, leave-one-out, sensitivity
    EG = R["egger"]
    close(ns["eg"].params["se"], EG["b0"], 1e-8); close(ns["eg"].pvalues["se"], EG["p"], 1e-8)
    close(ns["eg2"].params["Intercept"], EG["b0"], 1e-8); close(ns["eg2"].pvalues["Intercept"], EG["p"], 1e-8)
    for row, ref in zip(ns["loo"].itertuples(), R["loo"]):
        close(row.OR, ref["or_"][0], 6e-4, "loo " + ref["nm"])
        same(int(row._4), int(round(ref["I2"] * 100)), "loo I2 " + ref["nm"])
    close(ns["m_low"]["re"], R["lowrob"]["re"])
    close(np.exp(ns["re"] - ns["t9"] * ns["se_hk"]), R["hk"][0]); close(np.exp(ns["re"] + ns["t9"] * ns["se_hk"]), R["hk"][1])
    # absolute effects
    same(float(ns["p0"]), 0.084); same(float(ns["pb0"]), 0.012)
    rw = ns["risk_with"]
    close(1 / abs(rw(0.084, np.exp(m_["re"])) - 0.084), R["abs_e"]["n"][0])
    close(1 / abs(rw(0.012, np.exp(mb_["re"])) - 0.012), R["abs_b"]["n"][0])
    # homework
    close(ns["pg"].loc[2, "re"], R["phase"]["P2"]["re"]); close(ns["pg"].loc[3, "re"], R["phase"]["P3"]["re"])
    close(ns["m_rr"]["re"], R["rr"]["re"]); close(ns["m_rd"]["re"], R["rdm"]["re"]); close(ns["m_rd"]["se_re"], R["rdm"]["re_se"])
    close(ns["m_noC"]["re"], R["bleed_exC"]["re"]); close(ns["mh_b"].oddsratio_pooled, R["mh_bleed"][0])
    print("대조 블록: 본문 값과 일치", n_ok, "개")

    # ---- numbers quoted in the text only (not shown as cells)
    import warnings
    from statsmodels.stats.meta_analysis import combine_effects as ce
    pm = ce(ns["eff"], ns["var"])
    sf = pm.summary_frame()
    print("VERIFY default iterated: tau2 %.4f, RE OR %.3f (%.3f-%.3f)" % (
        pm.tau2, *np.exp(sf.loc["random effect", ["eff", "ci_low", "ci_upp"]].astype(float))))
    print("VERIFY trial C bleed 2x2 after 0.5:", 3.5, 237.5, 0.5, 120.5, " weight share %.2f%%" % (
        100 * (1 / ns["bl"]["vi"][2]) / (1 / ns["bl"]["vi"]).sum()))
    print("VERIFY subgroup FE:", orci_(sg_.loc["STEMI", "fe"], sg_.loc["STEMI", "se_fe"]),
          orci_(sg_.loc["NSTE-ACS", "fe"], sg_.loc["NSTE-ACS", "se_fe"]))
    print("VERIFY subgroup P:", sg_["P"].round(5).to_dict(), "tau2", sg_["tau2"].round(4).to_dict())
    for p0_, mm in ((0.15, m_), (0.04, m_), (0.03, mb_)):
        ORs = np.exp([mm["re"], mm["re"] - ns["Z"] * mm["se_re"], mm["re"] + ns["Z"] * mm["se_re"]])
        rd = rw(p0_, ORs) - p0_
        print("VERIFY baseline %.2f: risk %.4f RD %s NNT/NNH %s per1000 %s" % (
            p0_, rw(p0_, ORs[0]), np.round(100 * rd, 2), np.round(1 / np.abs(rd), 1), np.round(1000 * rd, 1)))
    print("VERIFY metareg intercept OR %.3f; fitted OR at 40%% %.3f, 60%% %.3f" % (
        np.exp(ns["mr"].params["Intercept"]), np.exp(ns["mr"].params["Intercept"] - ns["b"]),
        np.exp(ns["mr"].params["Intercept"] + ns["b"])))
    print("VERIFY mr.pvalues (statsmodels, this version):", round(float(ns["mr"].pvalues["stemi10"]), 4), " use_t:", ns["mr"].use_t)
    print("VERIFY C_reg %.2f  k-2 %d" % (ns["C_reg"], ns["k"] - 2))
    print("VERIFY Egger: funnel centre FE OR %.3f; weights C+G (RE) %.1f%%" % (
        np.exp(ns["fe"]), es_["w_re"][[2, 6]].sum()))
    print("VERIFY pooled RD -> NNT %.0f; RR I2 %.3f RD I2 %.3f; phase tau2 %s" % (
        -1 / ns["m_rd"]["re"], ns["m_rr"]["I2"], ns["m_rd"]["I2"], ns["pg"]["tau2"].round(4).to_dict()))
    print("VERIFY bleed 12-mo: %.4f %.4f ; exp(mean±1.96tau) %.3f-%.3f" % (
        662 / 7870, 93 / 7870, np.exp(m_["re"] - ns["Z"] * ns["tau"]), np.exp(m_["re"] + ns["Z"] * ns["tau"])))
    print("VERIFY sum(w^2) %.0f ; Q-df %.2f" % ((ns["w"] ** 2).sum(), ns["Q"] - 9))
