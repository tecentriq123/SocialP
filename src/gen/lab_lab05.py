"""실습 5 (lab05) — 비율 비교: birthwt(카이제곱·Fisher), esoph(선형 대 선형 결합).

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab05.py
      then  python3 tools/nbcheck.py lab05   (runs the notebook alone in a fresh process and compares outputs)
All outputs are produced by actually running the cells (labkit).
Writes figs/lab05_*.html (insert in content/lab05.html with <!--FIG:lab05_xxx-->),
pub/notebooks/lab05.ipynb (download link at the top of content/lab05.html) and checks/lab05.expected.json.
Set LABDEBUG=1 to print every cell's text output to the console.
The VERIFY block at the end asserts the numbers quoted in the text of content/lab05.html that are not
marked in a cell output (대응표, the p-value comparison table in c 절, the '직접 바꿔 보기' tips, the 과제).
(2026-10-14: version cell 1 split from the load cell, 과제 1–3 and the notebook added; cell numbers shifted by 1.)
"""
import html as _h
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab05")
DEBUG = os.environ.get("LABDEBUG")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


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


# ================================================================== a. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
from scipy import stats
import scipy, statsmodels, matplotlib

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scipy       ", scipy.__version__)
print("statsmodels ", statsmodels.__version__)
print("matplotlib  ", matplotlib.__version__)
''', title="패키지 불러오기와 버전 확인")
S("lab05_versions", c, {nb.ns["scipy"].__version__: 1, nb.ns["statsmodels"].__version__: 2})
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 birthwt 자료를 읽습니다. "
              "birthwt와 d 절의 esoph는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/"
       "csv/MASS/birthwt.csv")
df = pd.read_csv(url)
print(df.shape)
df.head()
''', title="데이터 불러오기")
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

# ================================================================== b. 카이제곱 검정
c = nb.cell('''
pd.crosstab(df["smoke_f"], df["low_f"], margins=True)
''', title="교차표와 합계")
NOTES[c.n] = "## b. 카이제곱 검정"
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
T22_CELL = c.n

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

# ================================================================== c. Fisher의 정확한 검정
c = nb.cell('''
obs_h = pd.crosstab(df["ht_f"], df["low_f"])
print(obs_h)
exp_h = stats.chi2_contingency(obs_h).expected_freq
print(exp_h.round(2))
print("기대빈도 5 미만인 칸:", (exp_h < 5).sum(), "/ 4")
''', title="고혈압 병력과 저체중아: 기대빈도 확인")
NOTES[c.n] = "## c. Fisher의 정확한 검정"
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
FISHER_CELL = c.n

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

# ================================================================== d. 선형 대 선형 결합
c = nb.cell('''
url2 = ("https://vincentarelbundock.github.io/Rdatasets/"
        "csv/datasets/esoph.csv")
es = pd.read_csv(url2)
print(es.shape, es["ncases"].sum(), es["ncontrols"].sum())
es.head(6)
''', title="esoph 데이터 불러오기")
NOTES[c.n] = "## d. 선형 대 선형 결합"
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
ALCOR_CELL = c.n

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

# ================================================================== e. 과제 (정답 셀)
c = nb.cell('''
row, ref = alc.loc["80-119"], alc.iloc[0]   # 80-119 g군, 기준군(0-39 g)
t80 = Table2x2(np.array([[row["ncases"], row["ncontrols"]],
                         [ref["ncases"], ref["ncontrols"]]]))
lo, hi = t80.oddsratio_confint()
print(f"80-119 vs 0-39: OR {t80.oddsratio:.2f} ({lo:.2f} to {hi:.2f})")
for k in [1, 2]:                            # 대조군을 k배로 뽑았다면
    pct80 = 100 * row["ncases"] / (row["ncases"] + k * row["ncontrols"])
    pct0 = 100 * ref["ncases"] / (ref["ncases"] + k * ref["ncontrols"])
    or_k = ((row["ncases"] / (k * row["ncontrols"]))
            / (ref["ncases"] / (k * ref["ncontrols"])))
    print(f"controls x{k}: case % {pct80:.1f} vs {pct0:.1f}, OR {or_k:.2f}")
mid = alc.loc["40-79"]                      # 이웃한 40-79 g군
print("80-119 vs 40-79: OR", round((row["ncases"] / row["ncontrols"])
                                   / (mid["ncases"] / mid["ncontrols"]), 2))
''', title="과제 1 정답. 음주량 80–119 g군의 오즈비 읽기")
S("lab05_hw1", c, {"OR 7.80 (4.68 to 13.02)": 1, "case % 37.0 vs 7.0": 2, "case % 22.7 vs 3.6, OR 7.80": 3,
                   "80-119 vs 40-79: OR 2.19": 4})
NOTES[c.n] = ("## 과제 정답\n\n"
              f"**과제 1.** 셀 {ALCOR_CELL}의 출력에서 80–119 g 행(OR 7.80, 95% CI 4.68–13.02)을 한 문장으로 옮기고, "
              "이 숫자가 말해 주지 않는 것을 두 가지 이상 쓰세요. 대조군을 두 배로 뽑았다면 이 군의 환자 비율과 오즈비가 "
              "어떻게 되는지, 40–79 g군과 견준 오즈비는 얼마인지도 계산해 보세요.")

c = nb.cell('''
tabs = {"original": obs.to_numpy(),                 # 흡연군, LBW가 첫 행·첫 열
        "rows swapped": obs.to_numpy()[::-1],        # 비흡연군을 첫 행으로
        "columns swapped": obs.to_numpy()[:, ::-1]}  # 'Not LBW'를 첫 열(사건)로
for name, arr in tabs.items():
    t = Table2x2(arr)
    r1, r0 = arr[0, 0] / arr[0].sum(), arr[1, 0] / arr[1].sum()
    print(f"{name:15s} risks {r1:.3f} {r0:.3f}  RD {r1 - r0:+.3f}"
          f"  RR {t.riskratio:.3f}  OR {t.oddsratio:.3f}")
''', title="과제 2 정답. 행과 열의 순서 바꾸기")
S("lab05_hw2", c, {"RD +0.153  RR 1.608  OR 2.022": 1, "RR 0.622  OR 0.495": 2, "risks 0.595 0.748": 3,
                   "RR 0.795  OR 0.495": 4})
NOTES[c.n] = (f"**과제 2.** 셀 {T22_CELL}의 `Table2x2`는 첫 행을 노출군, 첫 열을 사건으로 읽습니다. 흡연 교차표 `obs`의 "
              "행 순서를 바꿔 비흡연군을 첫 행에 두거나, 열 순서를 바꿔 'Not LBW'를 사건으로 두고 위험차, 상대위험도, "
              "오즈비를 다시 구하세요. 각 지표가 어떻게 바뀌는지, 열을 바꿨을 때 상대위험도가 원래 값의 역수"
              "(1 ÷ 1.608 = 0.622)가 되지 않는 이유는 무엇인지 설명하세요.")

c = nb.cell('''
pct_h = pd.crosstab(df["ht_f"], df["low_f"], normalize="index") * 100
print(pct_h.round(1))
bad = stats.chi2_contingency(pct_h, correction=False)   # 동료의 계산
print(f"percent table: total {pct_h.to_numpy().sum():.0f}, "
      f"chi2 = {bad.statistic:.2f}, p = {bad.pvalue:.1e}")
print(f"OR from percents {Table2x2(pct_h.to_numpy()).oddsratio:.3f}, "
      f"from counts {Table2x2(obs_h.to_numpy()).oddsratio:.3f}")
rr_h = Table2x2(obs_h.to_numpy()).riskratio             # 실제 인원으로
print(f"counts: Fisher p = {fe.pvalue:.3f}, RR = {rr_h:.2f}")
''', title="과제 3 정답. 백분율 표로 한 카이제곱 검정")
S("lab05_hw3", c, {"58.3": 1, "total 200": 2, "chi2 = 17.02, p = 3.7e-05": 3,
                   "OR from percents 3.365, from counts 3.365": 4, "Fisher p = 0.052": 5, "RR = 1.99": 6})
NOTES[c.n] = ("**과제 3.** 동료가 논문의 표에 있는 비율을 그대로 쓰려고 고혈압 병력과 저체중아의 행 백분율 표를 만들어 "
              "`stats.chi2_contingency(pct_h, correction=False)`에 넣었습니다(`pct_h = pd.crosstab(df[\"ht_f\"], "
              "df[\"low_f\"], normalize=\"index\") * 100`). 그리고 \"고혈압 병력이 있는 산모는 저체중아 출산 위험이 "
              "3.37배 높았다(χ² = 17.02, P < 0.001)\"라고 썼습니다. 동료의 계산을 재현하고, 이 결론에서 틀린 곳 두 가지를 "
              "찾아 왜 틀렸는지 설명하세요.")

print("lab05: cells =", len(nb.cells))
for cc_ in nb.cells:
    if cc_.warns:
        print("WARN cell", cc_.n, cc_.warns)

# ================================================================== notebook
nb.save_ipynb(
    "실습 5. 비율을 비교하는 방법",
    intro=("사회약학 연구방법 노트의 '실습 5. 비율을 비교하는 방법'에 딸린 노트북입니다. 앞부분(a–c)은 MASS 패키지의 "
           "birthwt 자료(출생아와 산모 189명)로 흡연, 고혈압 병력, 인종과 저체중아 출생의 관련을 카이제곱 검정과 "
           "Fisher의 정확한 검정으로 비교하고 위험차, 상대위험도, 오즈비를 구합니다. 뒷부분(d)은 식도암 환자-대조군 "
           "자료 esoph로 음주량에 따른 추세를 검정합니다. 셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 "
           "사이트의 실습 5에 있습니다."),
    notes=NOTES)

# ================================================================== 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import stats as st
    from statsmodels.stats.contingency_tables import Table as _T, Table2x2 as _T22
    from statsmodels.stats.proportion import confint_proportions_2indep as _ci2

    ns = nb.ns
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    r2 = lambda x: f"{x:.2f}"
    r3 = lambda x: f"{x:.3f}"
    # 글(content/lab05.html)과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 9: "상대위험도와 오즈비", 17: "Fisher의 정확한 검정",
             23: "음주량 순서대로 합치기", 24: "음주량별 오즈비와 95% 신뢰구간"}
    for k, t in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, t)
    # 셀 1의 출력 읽기에 적은 버전
    same("scipy version", ns["scipy"].__version__, "1.18.1")
    same("statsmodels version", ns["statsmodels"].__version__, "0.15.0")
    # 대응표: 이 실습의 자료 열
    df_, obs_ = ns["df"], ns["obs"].to_numpy()
    same("N", len(df_), 189)
    same("smokers, nonsmokers", (int(obs_[0].sum()), int(obs_[1].sum())), (74, 115))
    same("LBW n, %", (int(df_["low"].sum()), f"{df_['low'].mean():.1%}"), (59, "31.2%"))
    same("smoke risks", (f"{obs_[0, 0] / obs_[0].sum():.1%}", f"{obs_[1, 0] / obs_[1].sum():.1%}"), ("40.5%", "25.2%"))
    same("smoke chi2, P", (r2(ns["res"].statistic), r3(ns["res"].pvalue)), ("4.92", "0.026"))
    same("smoke RR, OR", (r2(ns["t22"].riskratio), r2(ns["t22"].oddsratio)), ("1.61", "2.02"))
    oh = ns["obs_h"].to_numpy()
    same("ht table", (int(oh[0, 0]), int(oh[0].sum()), int(oh[1, 0]), int(oh[1].sum())), (7, 12, 52, 177))
    same("ht Fisher P", r3(ns["fe"].pvalue), "0.052")
    same("esoph cases, controls", (int(ns["es"]["ncases"].sum()), int(ns["es"]["ncontrols"].sum())), (200, 775))
    same("esoph M2", r2(ns["M2"]), "152.97")
    # 대응표: 5장의 예 열 (5장 본문의 숫자를 다시 계산)
    ab = np.array([[48, 352], [102, 498]])
    chi_ab = st.chi2_contingency(ab, correction=False)
    t_ab = _T22(ab)
    same("ch05 a chi2, P, RR, OR", (r2(chi_ab.statistic), r3(chi_ab.pvalue), r2(t_ab.riskratio), r2(t_ab.oddsratio)),
         ("4.71", "0.030", "0.71", "0.67"))
    same("ch05 b Fisher P", r3(st.fisher_exact([[1, 14], [6, 8]]).pvalue), "0.035")
    tr = _T(np.array([[30, 170], [36, 264], [40, 460]])).test_ordinal_association(
        row_scores=np.array([1, 2, 3]), col_scores=np.array([1, 0]))
    same("ch05 c M2, P for trend", (r2(tr.zscore ** 2), r3(tr.pvalue)), ("8.21", "0.004"))
    # c 절의 p 비교표: 흡연 표의 Fisher P와 두 표의 세 p
    same("smoke Fisher P", r3(st.fisher_exact(obs_).pvalue), "0.036")
    same("smoke Yates P", r3(ns["yates"].pvalue), "0.040")
    same("ht Pearson, Yates P", (r3(ns["chi_u"].pvalue), r3(ns["chi_y"].pvalue)), ("0.036", "0.076"))
    same("min expected (smoke, ht)", (f"{ns['res'].expected_freq.min():.1f}", r2(ns["exp_h"].min())), ("23.1", "3.75"))
    # '직접 바꿔 보기'와 본문의 표식 없는 숫자
    same("RD Newcombe CI", [r3(v) for v in _ci2(30, 74, 29, 115, compare="diff")], ["0.018", "0.287"])
    ptl_e = st.chi2_contingency(pd.crosstab(df_["ptl_f"], df_["low_f"])).expected_freq.ravel()
    same("ptl expected", [r2(v) for v in ptl_e], ["9.37", "20.63", "49.63", "109.37"])
    cc_ = ns["cc"]
    same("scores 20,60,100,160 M2", r2(_T(cc_).test_ordinal_association(
        row_scores=np.array([20, 60, 100, 160]), col_scores=np.array([1, 0])).zscore ** 2), "157.32")
    es_ = ns["es"].copy()
    es_["tobgp"] = pd.Categorical(es_["tobgp"], ["0-9g/day", "10-19", "20-29", "30+"], ordered=True)
    tb = es_.groupby("tobgp", observed=True)[["ncases", "ncontrols"]].sum()
    same("tobgp Pearson, M2", (r2(st.chi2_contingency(tb).statistic),
                               r2(_T(tb).test_ordinal_association(col_scores=np.array([1, 0])).zscore ** 2)),
         ("29.36", "26.93"))
    same("ht LBW %", (f"{7 / 12:.1%}", f"{52 / 177:.1%}"), ("58.3%", "29.4%"))
    e_w = 96 * 59 / 189
    same("white resid by hand", (r2(e_w), r2((23 - e_w) / np.sqrt(e_w * (1 - 96 / 189) * (1 - 59 / 189)))),
         ("29.97", "-2.19"))
    same("Bonferroni 3 cells", r2(st.norm.ppf(1 - 0.05 / 6)), "2.39")
    pr = ns["prob"]
    same("hypergeom a=4, a=0, a=1", (f"{pr[4]:.4f}", f"{pr[0]:.4f}", f"{pr[1]:.4f}"), ("0.2434", "0.0095", "0.0565"))
    same("Yates chi2 (paper box)", r2(ns["yates"].statistic), "4.24")
    # 과제 1
    lo80, hi80 = ns["t80"].oddsratio_confint()
    same("80-119 OR (CI)", (r2(ns["t80"].oddsratio), r2(lo80), r2(hi80)), ("7.80", "4.68", "13.02"))
    alc_ = ns["alc"]
    same("case % by group", [f"{v:.1f}" for v in alc_["pct_case"]], ["7.0", "21.1", "37.0", "67.2"])
    # 과제 2
    same("RR not-LBW by hand", r3((44 / 74) / (86 / 115)), "0.795")
    same("1/RR, 1/OR", (r3(1 / ns["t22"].riskratio), r3(1 / ns["t22"].oddsratio)), ("0.622", "0.495"))
    # 과제 3
    same("ht RR", r2((7 / 12) / (52 / 177)), "1.99")
    same("percent table min expected", f"{ns['bad'].expected_freq.min():.1f}", "43.9")
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab05.html passed")
