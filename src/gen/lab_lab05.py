"""실습 5 (lab05) — 비율 비교: birthwt(카이제곱·Fisher), esoph(선형 대 선형 결합).

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab05.py
All outputs are produced by actually running the cells (labkit).
Set LABDEBUG=1 to print every cell's text output to the console.
"""
import html as _h
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab05")
DEBUG = os.environ.get("LABDEBUG")


def H(c, marks=None):
    """nb.html with each mark placed after the first occurrence anywhere in the
    output block (printed text or DataFrame table), skipping embedded images."""
    if DEBUG:
        print(f"----- cell {c.n}: {c.title}")
        print(c.stdout, c.value_repr if c.value_html is None else c.value_repr)
        for w in c.warns:
            print("WARN:", w)
    s = nb.html(c)
    if not marks:
        return s
    i = s.index('<div class="cell-out">')
    j = s.find("<img", i)
    j = len(s) if j < 0 else j
    head, body, tail = s[:i], s[i:j], s[j:]
    for sub, n in marks.items():
        # "<td>30</td>" style keys match raw table HTML; the mark goes inside the cell
        raw = sub.startswith("<")
        e = sub if raw else _h.escape(sub)
        k = body.find(e)
        if k < 0:
            raise ValueError(f"mark text not found in output of cell {c.n}: {sub!r}")
        k += e.rfind("</") if raw else len(e)
        body = body[:k] + f'<span class="mk">{n}</span>' + body[k:]
    return head + body + tail


def S(name, c, marks=None):
    nb.save_fragment(name, H(c, marks))


# ================================================================== 가. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
from scipy import stats

url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/MASS/birthwt.csv")
df = pd.read_csv(url)
print(df.shape)
df.head()
''', title="패키지와 데이터 불러오기")
S("lab05_load", c, {"(189, 11)": 1})

c = nb.cell('''
def label(col, mapping):
    # mapping에 적은 순서가 곧 범주의 순서가 됩니다
    return pd.Categorical(df[col].map(mapping),
                          categories=list(mapping.values()))

df["ptl_any"] = (df["ptl"] > 0).astype(int)  # 조산 경험 1회 이상
df["low_f"] = label("low", {1: "LBW", 0: "Not LBW"})
df["smoke_f"] = label("smoke", {1: "Smoker", 0: "Nonsmoker"})
df["ht_f"] = label("ht", {1: "HT", 0: "No HT"})
df["ptl_f"] = label("ptl_any", {1: "Preterm", 0: "None"})
df["race_f"] = label("race", {1: "White", 2: "Black", 3: "Other"})
''', title="범주에 이름표와 순서 붙이기")
S("lab05_label", c)

c = nb.cell('''
for col in ["low_f", "smoke_f", "ht_f", "ptl_f", "race_f"]:
    print(col, df[col].value_counts(sort=False).to_dict())
print("LBW 비율:", round(df["low"].mean(), 3))
''', title="범주별 빈도 확인")
S("lab05_counts", c, {"'LBW': 59": 1, "'HT': 12": 2, "'Preterm': 30": 3, "0.312": 4})

# ================================================================== 나. 카이제곱 검정
c = nb.cell('''
pd.crosstab(df["smoke_f"], df["low_f"], margins=True)
''', title="교차표와 합계")
S("lab05_crosstab", c, {"<td>30</td>": 1, "<td>29</td>": 2, "<td>189</td>": 3})

c = nb.cell('''
pct = pd.crosstab(df["smoke_f"], df["low_f"], normalize="index")
(pct * 100).round(1)
''', title="행 백분율")
S("lab05_rowpct", c, {"40.5": 1, "25.2": 2})

c = nb.cell('''
obs = pd.crosstab(df["smoke_f"], df["low_f"])  # 합계 없는 2×2 표
res = stats.chi2_contingency(obs, correction=False)
print("statistic    :", res.statistic)
print("pvalue       :", res.pvalue)
print("dof          :", res.dof)
print("expected_freq:")
print(res.expected_freq)
''', title="카이제곱 검정 (연속성 보정 없음)")
S("lab05_chi2", c, {"4.923705434361292": 1, "0.026490642530502487": 2, "dof          : 1": 3,
                    "[[23.1005291": 4})

c = nb.cell('''
yates = stats.chi2_contingency(obs)  # correction=True가 기본값
print(f"Pearson: chi2 = {res.statistic:.3f}, p = {res.pvalue:.4f}")
print(f"Yates:   chi2 = {yates.statistic:.3f}, p = {yates.pvalue:.4f}")
print("가장 작은 기대빈도:", res.expected_freq.min().round(1))
''', title="Yates 연속성 보정과 비교")
S("lab05_yates", c, {"p = 0.0265": 1, "p = 0.0396": 2, "23.1": 3})

c = nb.cell('''
from statsmodels.stats.contingency_tables import Table2x2

t22 = Table2x2(obs.to_numpy())  # DataFrame 대신 숫자 배열로
print(t22.summary())
''', title="상대위험도와 오즈비")
S("lab05_t22", c, {"2.022": 1, "0.320": 2, "1.081 3.783": 3, "1.608": 4, "1.058 2.443": 5})

c = nb.cell('''
wrong = Table2x2(obs)  # 이름표가 붙은 DataFrame을 그대로 넣으면
print(wrong.table.shape, wrong.riskratio)
''', title="DataFrame을 그대로 넣으면")
S("lab05_t22wrong", c, {"(4, 4)": 1, "1.0": 2})

c = nb.cell('''
from statsmodels.stats.proportion import confint_proportions_2indep

(a, b), (c, d) = obs.to_numpy()     # 네 칸의 빈도
p1, p0 = a / (a + b), c / (c + d)   # 흡연군, 비흡연군의 위험
lo, hi = confint_proportions_2indep(a, a + b, c, c + d,
                                    compare="diff", method="wald")
print(f"p1 = {p1:.3f}, p0 = {p0:.3f}")
print(f"RD = {p1 - p0:.3f} (95% CI {lo:.3f} to {hi:.3f})")
''', title="위험차와 95% 신뢰구간")
S("lab05_rd", c, {"RD = 0.153": 1, "0.016 to 0.290": 2})

c = nb.cell('''
rr, or_ = t22.riskratio, t22.oddsratio
print(f"RR = {rr:.3f}, OR = {or_:.3f}")
print("RR x (1-p0)/(1-p1) =", round(rr * (1 - p0) / (1 - p1), 3))
print("OR/(1-p0+p0*OR) =", round(or_ / (1 - p0 + p0 * or_), 3))
''', title="오즈비와 상대위험도의 관계")
S("lab05_orrr", c, {"RR x (1-p0)/(1-p1) = 2.022": 1, "OR/(1-p0+p0*OR) = 1.608": 2})

c = nb.cell('''
obs_r = pd.crosstab(df["race_f"], df["low_f"])
res_r = stats.chi2_contingency(obs_r)  # 2×2가 아니면 보정 없음
print(f"chi2 = {res_r.statistic:.3f}, dof = {res_r.dof}, "
      f"p = {res_r.pvalue:.4f}")
pct_r = pd.crosstab(df["race_f"], df["low_f"], normalize="index")
(pct_r * 100).round(1)
''', title="세 군 이상: 인종별 저체중아 비율")
S("lab05_race", c, {"chi2 = 5.005": 1, "dof = 2": 2, "p = 0.0819": 3})

c = nb.cell('''
from statsmodels.stats.contingency_tables import Table

Table(obs_r).standardized_resids.round(2)
''', title="수정 표준화 잔차")
S("lab05_resid", c, {"-2.19": 1, "1.31": 2, "1.34": 3})

c = nb.cell('''
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
for ax, col in zip(axes, ["smoke_f", "race_f"]):
    pct = df.groupby(col, observed=True)["low"].mean() * 100
    bars = ax.bar(pct.index.astype(str), pct.values)
    ax.bar_label(bars, fmt="%.1f")
    ax.set_xlabel(col)
axes[0].set_ylabel("Low birth weight (%)")
axes[0].set_ylim(0, 50)
plt.show()
''', title="군별 저체중아 비율 그래프")
S("lab05_barplot", c)

# ================================================================== 다. Fisher의 정확한 검정
c = nb.cell('''
obs_h = pd.crosstab(df["ht_f"], df["low_f"])
print(obs_h)
exp_h = stats.chi2_contingency(obs_h).expected_freq
print(exp_h.round(2))
print("기대빈도 5 미만인 칸:", (exp_h < 5).sum(), "/ 4")
''', title="고혈압 병력과 저체중아: 기대빈도 확인")
S("lab05_ht", c, {"3.75": 1, "기대빈도 5 미만인 칸: 1 / 4": 2})

c = nb.cell('''
fe = stats.fisher_exact(obs_h)
print("statistic:", fe.statistic)
print("pvalue   :", fe.pvalue)
chi_u = stats.chi2_contingency(obs_h, correction=False)
chi_y = stats.chi2_contingency(obs_h)
print(f"Pearson p = {chi_u.pvalue:.4f}, Yates p = {chi_y.pvalue:.4f}")
''', title="Fisher의 정확한 검정")
S("lab05_fisher", c, {"3.3653846153846154": 1,
                      "0.05161187156764768": 2,
                      "Pearson p = 0.0362": 3, "Yates p = 0.0763": 4})

c = nb.cell('''
from scipy.stats.contingency import odds_ratio

for kind in ["sample", "conditional"]:
    o = odds_ratio(obs_h.to_numpy(), kind=kind)
    ci = o.confidence_interval(confidence_level=0.95)
    print(f"{kind:11s} OR = {o.statistic:.3f} "
          f"(95% CI {ci.low:.2f} to {ci.high:.2f})")
''', title="오즈비 두 가지")
S("lab05_or2", c, {"OR = 3.365": 1, "(95% CI 1.02 to 11.09)": 2, "OR = 3.341": 3,
                   "(95% CI 0.87 to 14.00)": 4})

c = nb.cell('''
total, lbw, ht_n = 189, 59, 12   # 전체, LBW 합계, HT 산모 수
a = np.arange(0, ht_n + 1)        # HT 산모 중 LBW 수: 0 ~ 12
prob = stats.hypergeom.pmf(a, total, lbw, ht_n)
p_obs = prob[7]                   # 관측된 표 (a = 7)
extreme = prob <= p_obs * (1 + 1e-7)
print(prob.round(4).tolist())
print(f"P(a = 7) = {p_obs:.4f}, 양측 p = {prob[extreme].sum():.4f}")
''', title="가능한 모든 표의 확률")
S("lab05_hyper", c, {"P(a = 7) = 0.0322": 1, "양측 p = 0.0516": 2})

c = nb.cell('''
plt.figure(figsize=(6, 3))
plt.bar(a, prob, color=np.where(extreme, "tab:orange", "tab:gray"))
plt.axvline(ht_n * lbw / total, ls="--", color="k")  # 기대 3.75
plt.xlabel("Number of LBW among 12 mothers with hypertension")
plt.ylabel("Probability")
plt.show()
''', title="초기하분포 그래프")
S("lab05_hyperplot", c)

# ================================================================== 라. 선형 대 선형 결합
c = nb.cell('''
url2 = ("https://vincentarelbundock.github.io/Rdatasets/"
        "csv/datasets/esoph.csv")
es = pd.read_csv(url2)
print(es.shape, es["ncases"].sum(), es["ncontrols"].sum())
es.head(6)
''', title="esoph 데이터 불러오기")
S("lab05_esoph", c, {"(88, 6) 200 775": 1})

c = nb.cell('''
es.groupby("alcgp")[["ncases", "ncontrols"]].sum()
''', title="순서를 지정하지 않고 합치면")
S("lab05_badorder", c, {"120+": 1})

c = nb.cell('''
alc_order = ["0-39g/day", "40-79", "80-119", "120+"]
es["alcgp"] = pd.Categorical(es["alcgp"], categories=alc_order,
                             ordered=True)
alc = (es.groupby("alcgp", observed=True)
         [["ncases", "ncontrols"]].sum())
n_grp = alc["ncases"] + alc["ncontrols"]
alc["pct_case"] = 100 * alc["ncases"] / n_grp
alc["odds"] = alc["ncases"] / alc["ncontrols"]
alc["OR"] = alc["odds"] / alc["odds"].iloc[0]
alc.round(3)
''', title="음주량 순서대로 합치기")
S("lab05_alc", c, {"6.988": 1, "0.075": 2, "67.164": 3, "27.226": 4})

c = nb.cell('''
ref = alc.iloc[0]            # 기준범주: 0-39 g/day
for g, row in alc.iloc[1:].iterrows():
    t = Table2x2(np.array([[row["ncases"], row["ncontrols"]],
                           [ref["ncases"], ref["ncontrols"]]]))
    lo, hi = t.oddsratio_confint()
    print(f"{g:>6}: OR {t.oddsratio:5.2f} ({lo:.2f} to {hi:.2f})")
''', title="음주량별 오즈비와 95% 신뢰구간")
S("lab05_alcor", c, {"(2.26 to 5.62)": 1, "(14.44 to 51.34)": 2})

c = nb.cell('''
cc = alc[["ncases", "ncontrols"]]   # 환자 수, 대조군 수 두 열만
(cc / cc.sum() * 100).round(1)       # 열 백분율
''', title="환자군과 대조군의 음주 분포")
S("lab05_colpct", c, {"<td>14.5</td>": 1, "<td>49.8</td>": 2})

c = nb.cell('''
plt.figure(figsize=(6, 3))
bars = plt.bar(alc.index.astype(str), alc["pct_case"])
plt.bar_label(bars, labels=[f"{x}/{m}" for x, m
                            in zip(alc["ncases"], n_grp)])
plt.xlabel("Alcohol consumption (g/day)")
plt.ylabel("Cases among subjects (%)")
plt.ylim(0, 80)
plt.show()
''', title="음주량별 환자 비율 그래프")
S("lab05_alcplot", c)

c = nb.cell('''
pear = stats.chi2_contingency(cc)
print(f"Pearson chi2 = {pear.statistic:.2f}, dof = {pear.dof}, "
      f"p = {pear.pvalue:.2e}")
''', title="Pearson 카이제곱 검정 (자유도 3)")
S("lab05_alcchi2", c, {"chi2 = 158.95": 1, "dof = 3": 2, "p = 3.08e-34": 3})

c = nb.cell('''
score = np.array([1, 2, 3, 4])     # 음주 범주의 점수
case_n = cc["ncases"].to_numpy()
ctrl_n = cc["ncontrols"].to_numpy()
x = np.r_[np.repeat(score, case_n), np.repeat(score, ctrl_n)]
y = np.r_[np.ones(case_n.sum()), np.zeros(ctrl_n.sum())]
print(x.shape, x[y == 1].mean(), x[y == 0].mean().round(3))
r = np.corrcoef(x, y)[0, 1]
M2 = (len(x) - 1) * r**2
print(f"r = {r:.4f}, M2 = {M2:.2f}, p = {stats.chi2.sf(M2, 1):.2e}")
''', title="M² = (N − 1)r² 직접 계산")
S("lab05_m2", c, {"(975,)": 1, "2.56": 2, "1.671": 3, "r = 0.3963": 4, "M2 = 152.97": 5,
                  "p = 3.88e-35": 6})

c = nb.cell('''
lbl = Table(cc).test_ordinal_association(
    row_scores=score, col_scores=np.array([1, 0]))
print(f"z = {lbl.zscore:.3f}, z^2 = {lbl.zscore**2:.2f}, "
      f"p = {lbl.pvalue:.2e}")
print(f"Cochran-Armitage (N r^2) = {len(x) * r**2:.2f}")
dev = pear.statistic - M2          # 직선에서 벗어난 부분
print(f"departure = {dev:.2f}, p = {stats.chi2.sf(dev, 2):.3f}")
''', title="statsmodels로 같은 검정")
S("lab05_ordinal", c, {"z = 12.368": 1, "z^2 = 152.97": 2, "(N r^2) = 153.13": 3,
                       "departure = 5.98": 4, "p = 0.050": 5})

# ---------------------------------------------------------------- numbers quoted in the text
ns = nb.ns
_st, _np, _pd = ns["stats"], ns["np"], ns["pd"]
from statsmodels.stats.proportion import confint_proportions_2indep as _ci2
from statsmodels.stats.contingency_tables import Table as _T
print("RD newcombe (default):", _np.round(_ci2(30, 74, 29, 115, compare="diff"), 3))
_cc = ns["cc"]
print("scores 20,60,100,160 M2:", round(_T(_cc).test_ordinal_association(
    row_scores=_np.array([20, 60, 100, 160]), col_scores=_np.array([1, 0])).zscore ** 2, 2))
_es = ns["es"].copy()
_es["tobgp"] = _pd.Categorical(_es["tobgp"], ["0-9g/day", "10-19", "20-29", "30+"], ordered=True)
_tb = _es.groupby("tobgp", observed=True)[["ncases", "ncontrols"]].sum()
_p = _st.chi2_contingency(_tb)
_o = _T(_tb).test_ordinal_association(col_scores=_np.array([1, 0]))
print("tobgp:", _tb.values.tolist(), "Pearson", round(_p.statistic, 2), f"{_p.pvalue:.2e}",
      "M2", round(_o.zscore ** 2, 2), f"{_o.pvalue:.2e}")
_df = ns["df"]
print("ptl expected:", _st.chi2_contingency(_pd.crosstab(_df["ptl_f"], _df["low_f"])).expected_freq.round(2).tolist())
print("ht LBW%:", round(7 / 12 * 100, 1), round(52 / 177 * 100, 1))
print("white resid check:", round((23 - 96 * 59 / 189) / _np.sqrt(96 * 59 / 189 * (1 - 96 / 189) * (1 - 59 / 189)), 3))
print("lab05: cells =", len(nb.cells))
