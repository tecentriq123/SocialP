"""실습 12 · 포아송 회귀와 음이항 회귀 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab12.py
      then  python3 tools/nbcheck.py lab12   (runs the notebook alone in a fresh process and compares outputs)
      (LABDEBUG=1 prints every cell's text output, for writing marks)
Writes figs/lab12_*.html (insert in content/lab12.html with <!--FIG:lab12_xxx-->),
pub/notebooks/lab12.ipynb (download link at the top of content/lab12.html) and checks/lab12.expected.json.
Data: survival::cgd (Rdatasets URL -> offline copy via labkit).
The VERIFY block at the end asserts the numbers quoted in the text of content/lab12.html that are not
marked in a cell output (대응표, the explanations, the 과제 problems and answers).
(2026-10-14: version cell 1, quasi-Poisson cell 14, standardization cell 18, 과제 1–3 and the notebook added;
cell numbers shifted by 1 from cell 1, by 2 from the old cell 13 and by 3 from the old cell 15.)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab12")
DEBUG = os.environ.get("LABDEBUG")
NOTES = {}     # 노트북에서 셀 앞에 넣을 짧은 안내 (절 제목, 과제 문제)


def freeze_clock(h):
    """summary()가 찍는 실행 날짜와 시각(Date, Time 줄)을 고정값으로 바꾼다(gen/lab_lab09.py와 같음). 다시 돌려도
    같은 파일이 나오게 하려는 것이고 통계 출력은 건드리지 않는다. 같은 길이의 글자로 바꾸므로 줄 맞춤도 그대로다."""
    h = re.sub(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}", r"\g<1>Wed, 14 Oct 2026", h)
    return re.sub(r"(Time:\s+)\d{2}:\d{2}:\d{2}", r"\g<1>09:10:59", h)


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
    nb.save_fragment(name, freeze_clock(h))


# ------------------------------------------------------------------ 가
c = nb.cell('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy, statsmodels, matplotlib

print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scipy       ", scipy.__version__)
print("statsmodels ", statsmodels.__version__)
print("matplotlib  ", matplotlib.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab12_versions", c, marks={"~scipy\\s+[0-9.]+": 1, "~statsmodels\\s+[0-9.]+": 2})
NOTES[c.n] = ("## a. 실습 데이터 준비\n\n이 실습이 쓰는 패키지를 불러와 버전을 확인한 뒤 cgd 자료를 읽습니다. "
              "자료는 인터넷에서 읽어 오므로 인터넷 연결이 필요합니다.")

c = nb.cell('''
url = ("https://vincentarelbundock.github.io/Rdatasets/csv/"
       "survival/cgd.csv")
cgd = pd.read_csv(url).drop(columns="rownames")
print(cgd.shape, cgd["id"].nunique())
cols = ["id", "treat", "tstart", "tstop", "status", "enum"]
cgd[cols].head(12)
''', title="cgd 불러오기 (계수과정 형식)", max_rows=12)
save("lab12_load", c, marks={"(203, 16) 128": 1})

c = nb.cell('''
first = cgd.groupby("id").first()      # 환자마다 첫 행 하나
print(first["treat"].value_counts())
print(cgd["status"].value_counts())
''', title="코딩 확인")
save("lab12_coding", c, marks={"placebo    65": 1, "~1\\s+76": 2})

c = nb.cell('''
pt = cgd.groupby("id").agg(
    treat=("treat", "first"),
    age=("age", "first"),
    inherit=("inherit", "first"),
    propylac=("propylac", "first"),
    n_inf=("status", "sum"),       # 감염 건수
    days=("tstop", "max"),         # 추적 일수
).reset_index()
pt["py"] = pt["days"] / 365.25     # 인년
pt["treat"] = pd.Categorical(pt["treat"],
                             categories=["placebo", "rIFN-g"])
print(pt.shape)
pt.head()
''', title="환자 한 명 = 한 행으로 요약")
save("lab12_agg", c, marks={"(128, 8)": 1})

c = nb.cell('''
span = (cgd["tstop"] - cgd["tstart"]).groupby(cgd["id"]).sum()
print((span.values == pt["days"].values).all())  # 빈틈 없는 추적?
print(pt["n_inf"].sum(), round(pt["py"].sum(), 1))
pt["n_inf"].value_counts().sort_index()
''', title="요약이 맞는지 확인")
save("lab12_aggcheck", c, marks={"True": 1, "76 102.6": 2})

# ------------------------------------------------------------------ 나
c = nb.cell('''
from scipy import stats

rt = pt.groupby("treat", observed=True).agg(
    patients=("id", "size"),
    events=("n_inf", "sum"),
    py=("py", "sum"))
d, T = rt["events"], rt["py"]
rt["rate"] = d / T * 100                          # 100인년당
rt["lower"] = stats.chi2.ppf(0.025, 2 * d) / 2 / T * 100
rt["upper"] = stats.chi2.ppf(0.975, 2 * d + 2) / 2 / T * 100
rt.round(2)
''', title="군별 발생률과 정확 신뢰구간")
save("lab12_rates", c, marks={"110.42": 1, "83.41": 2, "38.54": 3})
NOTES[c.n] = "## b. 발생률과 발생률비"

c = nb.cell('''
irr = rt.loc["rIFN-g", "rate"] / rt.loc["placebo", "rate"]
se = np.sqrt(1 / d["rIFN-g"] + 1 / d["placebo"])  # log IRR의 SE
lo = np.exp(np.log(irr) - 1.96 * se)
hi = np.exp(np.log(irr) + 1.96 * se)
print(f"IRR = {irr:.3f}  95% CI {lo:.3f} to {hi:.3f}")
print(f"SE(log IRR) = {se:.3f}")
''', title="조발생률비와 Wald 신뢰구간")
save("lab12_irr", c, marks={"IRR = 0.349": 1, "0.209 to 0.582": 2, "0.260": 3})

c = nb.cell('''
share = T["rIFN-g"] / T.sum()        # rIFN-g군 인년의 비율
res = stats.binomtest(int(d["rIFN-g"]), int(d.sum()), share)
print(f"expected share {share:.3f}, observed {d['rIFN-g']}/{d.sum()}")
print("exact P =", res.pvalue)
ci = res.proportion_ci()             # 정확(Clopper-Pearson) 구간
k = T["placebo"] / T["rIFN-g"]
print(f"exact 95% CI {ci.low / (1 - ci.low) * k:.3f}"
      f" to {ci.high / (1 - ci.high) * k:.3f}")
''', title="조건부 정확검정")
save("lab12_exact", c, marks={"0.506": 1, "20/76": 2, "~exact P = [0-9.e-]+": 3, "0.198 to 0.591": 4})

# ------------------------------------------------------------------ 다
c = nb.cell('''
import statsmodels.api as sm
import statsmodels.formula.api as smf

pt["inherit"] = pd.Categorical(pt["inherit"],
                               categories=["X-linked", "autosomal"])
form = "n_inf ~ treat + age + inherit + propylac"
pois = smf.glm(form, data=pt, family=sm.families.Poisson(),
               offset=np.log(pt["py"])).fit()
print(pois.summary())
''', title="포아송 회귀 (오프셋 = log 인년)")
save("lab12_pois", c, marks={"No. Observations:                  128": 1,
                             "~Link Function:\\s+Log": 2,
                             "~Scale:\\s+1\\.0000": 3,
                             "~Deviance:\\s+[0-9.]+": 4, "~Pearson chi2:\\s+[0-9.]+": 5,
                             r"~Intercept\s+0\.9004[^\n]*": 6,
                             r"~treat\[T\.rIFN-g\]\s+-1\.0157[^\n]*": 7, r"~age\s+-0\.0371[^\n]*": 8})
NOTES[c.n] = "## c. 포아송 회귀와 과산포"
POIS_CELL = c.n

c = nb.cell('''
ci = pois.conf_int()
pd.DataFrame({"IRR": np.exp(pois.params),
              "2.5%": np.exp(ci[0]),
              "97.5%": np.exp(ci[1])}).round(3)
''', title="발생률비와 95% 신뢰구간")
save("lab12_pois_irr", c)
IRR_CELL = c.n

c = nb.cell('''
print("deviance / df:", round(pois.deviance / pois.df_resid, 3))
print("Pearson chi2 / df:",
      round(pois.pearson_chi2 / pois.df_resid, 3))
''', title="과산포 확인")
save("lab12_disp", c, marks={"1.228": 1, "1.386": 2})

c = nb.cell('''
mu = pois.fittedvalues            # 환자별 기대 건수 (인년 반영)
k = np.arange(0, 8)
obs = [(pt["n_inf"] == i).sum() for i in k]
exp_p = [stats.poisson.pmf(i, mu).sum() for i in k]
print("zeros: observed", obs[0], " Poisson", round(exp_p[0], 1))

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(k - 0.2, obs, width=0.4, label="Observed")
ax.bar(k + 0.2, exp_p, width=0.4, label="Expected, Poisson")
ax.set_xlabel("Number of serious infections per patient")
ax.set_ylabel("Number of patients")
ax.legend()
plt.show()
''', title="관측 빈도와 포아송 기대 빈도")
save("lab12_obsexp", c, marks={"Poisson 75.7": 1})

c = nb.cell('''
pois_r = smf.glm(form, data=pt, family=sm.families.Poisson(),
                 offset=np.log(pt["py"])).fit(cov_type="HC0")
t = "treat[T.rIFN-g]"
print("model-based SE:", round(pois.bse[t], 3))
print("robust SE     :", round(pois_r.bse[t], 3))
print("robust 95% CI :", np.exp(pois_r.conf_int().loc[t]).round(3).values)
''', title="강건(샌드위치) 표준오차")
save("lab12_robust", c, marks={"model-based SE: 0.262": 1, "robust SE     : 0.309": 2, "[0.198 0.664]": 3})
ROBUST_CELL = c.n

c = nb.cell('''
pois_q = smf.glm(form, data=pt, family=sm.families.Poisson(),
                 offset=np.log(pt["py"])).fit(scale="X2")
phi = pois.pearson_chi2 / pois.df_resid      # Pearson chi2 / df
print("phi =", round(phi, 3), "  sqrt(phi) =", round(np.sqrt(phi), 3))
print("same coefficients:", np.allclose(pois_q.params, pois.params))
print("quasi-Poisson SE:", round(pois_q.bse[t], 4),
      "  model-based SE x sqrt(phi):", round(pois.bse[t] * np.sqrt(phi), 4))
print("quasi-Poisson 95% CI:", np.exp(pois_q.conf_int().loc[t]).round(3).values)
''', title="준포아송: 표준오차에 √φ 곱하기")
save("lab12_quasi", c, marks={"sqrt(phi) = 1.177": 1, "same coefficients: True": 2,
                              "quasi-Poisson SE: 0.3085": 3, "[0.198 0.663]": 4})

# ------------------------------------------------------------------ 라
c = nb.cell('''
nb2 = smf.negativebinomial(form, data=pt,
                           offset=np.log(pt["py"])).fit()
print(nb2.summary())
''', title="음이항 회귀 (NB2)")
save("lab12_nb", c, marks={"Optimization terminated successfully.": 1,
                           "~converged:\\s+True": 2,
                           r"~treat\[T\.rIFN-g\]\s+-1\.0096[^\n]*": 4, r"~alpha\s+0\.7153[^\n]*": 5,
                           "~LLR p-value:\\s+[0-9.]+": 3})
NOTES[c.n] = "## d. 음이항 회귀"
NB_CELL = c.n

c = nb.cell('''
fits = {"Poisson": pois, "Quasi-Poisson": pois_q,
        "Poisson, robust SE": pois_r, "Negative binomial": nb2}
tab = pd.DataFrame(
    {"IRR": [np.exp(f.params[t]) for f in fits.values()],
     "lower": [np.exp(f.conf_int().loc[t, 0]) for f in fits.values()],
     "upper": [np.exp(f.conf_int().loc[t, 1]) for f in fits.values()],
     "SE": [f.bse[t] for f in fits.values()]},
    index=fits.keys())
print("AIC  Poisson:", round(pois.aic, 1), " NB:", round(nb2.aic, 1))
lr = 2 * (nb2.llf - pois.llf)
print("LR =", round(lr, 2), " p =", round(stats.chi2.sf(lr, 1) / 2, 4))
tab.round(3)
''', title="네 방법 비교, AIC와 우도비 검정")
save("lab12_cmp", c, marks={"264.1": 1, "256.6": 2, "LR = 9.55": 3})

c = nb.cell('''
a = nb2.params["alpha"]
mu_nb = nb2.predict()                 # 환자별 기대 건수
p0_nb = (1 + a * mu_nb) ** (-1 / a)   # 음이항 분포의 P(0건)
print("zeros: observed", obs[0], " Poisson", round(exp_p[0], 1),
      " NB", round(p0_nb.sum(), 1))
''', title="0건 환자 수 예측 비교")
save("lab12_zeros", c, marks={"NB 83.2": 1})
ZEROS_CELL = c.n

c = nb.cell('''
one_py = np.zeros(len(pt))            # 오프셋 ln(1인년) = 0
r0 = nb2.predict(pt.assign(treat="placebo"), offset=one_py)
r1 = nb2.predict(pt.assign(treat="rIFN-g"), offset=one_py)
print(f"standardized rate per 100 PY: placebo {100 * r0.mean():.1f},"
      f" rIFN-g {100 * r1.mean():.1f}")
print(f"ratio {r1.mean() / r0.mean():.3f},"
      f" difference {100 * (r1.mean() - r0.mean()):.1f} per 100 PY")
''', title="표준화로 발생률과 발생률차 구하기")
save("lab12_std", c, marks={"placebo 105.7": 1, "rIFN-g 38.5": 2, "ratio 0.364": 3,
                            "difference -67.2 per 100 PY": 4})
STD_CELL = c.n

c = nb.cell('''
nb_glm = smf.glm(form, data=pt,
                 family=sm.families.NegativeBinomial(alpha=a),
                 offset=np.log(pt["py"])).fit()
both = pd.concat({"discrete NB": nb2.params,
                  "GLM NB (alpha fixed)": nb_glm.params}, axis=1)
print(both.round(4))
''', title="GLM의 음이항 family와 비교")
save("lab12_nbglm", c)

c = nb.cell('''
fam = sm.families.NegativeBinomial()   # alpha를 주지 않으면?
''', title="alpha를 빠뜨리면")
save("lab12_nbwarn", c)

# ------------------------------------------------------------------ e. 과제 (정답 셀)
c = nb.cell('''
print(pd.crosstab(pt["propylac"], pt["treat"], margins=True))
print(pt.groupby("propylac")[["n_inf", "py"]].sum().round(1))
for name, f in [("Poisson", pois), ("NB", nb2)]:
    lo, hi = np.exp(f.conf_int().loc["propylac"])
    print(f"{name:8} IRR {np.exp(f.params['propylac']):.3f}"
          f" (95% CI {lo:.3f} to {hi:.3f}), P = {f.pvalues['propylac']:.3f}")
''', title="과제 1 정답. 예방적 항생제의 발생률비 읽기")
save("lab12_hw1", c, marks={"~0\\s+10\\s+7\\s+17": 1, "~0\\s+14\\s+13\\.1": 2,
                            "Poisson  IRR 0.589 (95% CI 0.326 to 1.064), P = 0.079": 3,
                            "NB       IRR 0.561 (95% CI 0.256 to 1.229), P = 0.149": 4})
NOTES[c.n] = ("## 과제 정답\n\n"
              f"**과제 1.** 셀 {IRR_CELL}의 발생률비 표에서 propylac 행(예방적 항생제 사용)을 한 문장으로 옮기고, "
              "이 숫자가 말해 주지 않는 것을 두 가지 이상 쓰세요. 예방적 항생제를 쓰지 않은 환자가 몇 명인지, "
              f"셀 {NB_CELL}의 음이항 회귀에서는 이 행이 어떻게 나오는지도 확인하세요.")

c = nb.cell('''
m_days = smf.glm(form, data=pt, family=sm.families.Poisson(),
                 offset=np.log(pt["days"])).fit()     # 오프셋을 일 단위로
m_none = smf.glm(form, data=pt,
                 family=sm.families.Poisson()).fit()  # 오프셋 없음
coefs = pd.DataFrame({"ln(PY)": pois.params, "ln(days)": m_days.params,
                      "none": m_none.params})
print(coefs.round(4))
print("ln(365.25) =", round(np.log(365.25), 4))
print("treat IRR:", np.exp(coefs.loc[t]).round(3).tolist())
pt.groupby("treat", observed=True)["days"].mean().round(1)
''', title="과제 2 정답. 오프셋의 단위를 바꾸거나 빼기")
save("lab12_hw2", c, marks={"~Intercept\\s+0\\.9004\\s+-5\\.0002\\s+0\\.6652": 1, "ln(365.25) = 5.9006": 2,
                            "[0.362, 0.362, 0.387]": 3, "~placebo</th>\\s*<td>285\\.0": 4})
NOTES[c.n] = (f"**과제 2.** 셀 {POIS_CELL}의 포아송 회귀에서 오프셋을 (가) 인년 대신 일수의 로그 "
              "`np.log(pt[\"days\"])`로 바꾸고 (나) 아예 빼고 다시 적합하세요. 치료군 발생률비와 절편이 각각 "
              "어떻게 달라지는지, 왜 그런지 설명하세요.")

c = nb.cell('''
rows = cgd.copy()                                        # 동료가 쓴 203행 자료
rows["py_row"] = (rows["tstop"] - rows["tstart"]) / 365.25
rows["treat"] = pd.Categorical(rows["treat"],
                               categories=["placebo", "rIFN-g"])
rows["inherit"] = pd.Categorical(rows["inherit"],
                                 categories=["X-linked", "autosomal"])
f_row = "status ~ treat + age + inherit + propylac"
glm_row = smf.glm(f_row, data=rows, family=sm.families.Poisson(),
                  offset=np.log(rows["py_row"]))
m_hc0 = glm_row.fit(cov_type="HC0")                     # 동료의 분석
m_cl = glm_row.fit(cov_type="cluster", cov_kwds={"groups": rows["id"]})
fs = {"patients, model-based": pois, "patients, robust": pois_r,
      "intervals, robust (HC0)": m_hc0,
      "intervals, clustered by id": m_cl}
pd.DataFrame(
    {"coef": [f.params[t] for f in fs.values()],
     "SE": [f.bse[t] for f in fs.values()],
     "lower": [np.exp(f.conf_int().loc[t, 0]) for f in fs.values()],
     "upper": [np.exp(f.conf_int().loc[t, 1]) for f in fs.values()]},
    index=fs.keys()).round(4)
''', title="과제 3 정답. 203행 자료에 강건 표준오차를 쓴 분석")
save("lab12_hw3", c, marks={"0.2620": 1, "0.3094": 2, "0.2561": 3, "0.3137": 4})
NOTES[c.n] = ("**과제 3.** 동료가 환자 단위로 요약하지 않고 계수과정 형식의 203행을 그대로 분석했습니다. 행마다 "
              "status를 결과로, 그 구간 길이(인년)의 로그를 오프셋으로 넣은 포아송 회귀에 강건 표준오차(HC0)를 쓰고, "
              "\"강건 표준오차를 썼으므로 과산포도 처리했다. 보정 발생률비는 0.36(95% CI 0.22–0.60)으로 환자 단위 "
              "분석보다 정밀하다\"라고 썼습니다. 동료의 분석을 재현하고, 맞는 곳과 틀린 곳을 찾아 설명하세요.")

if DEBUG:
    print("\ncells:", len(nb.cells))
for cc in nb.cells:
    if cc.warns and cc.title != "alpha를 빠뜨리면":
        print("WARN cell", cc.n, cc.warns)

# ================================================================ notebook
nb.save_ipynb(
    "실습 12. 포아송 회귀와 음이항 회귀",
    intro=("사회약학 연구방법 노트의 '실습 12. 포아송 회귀와 음이항 회귀'에 딸린 노트북입니다. R survival 패키지의 "
           "cgd 자료(만성 육아종병 환자 128명의 감마 인터페론 무작위배정 시험)를 환자 단위로 요약하고, 군별 발생률과 "
           "발생률비, 오프셋을 넣은 포아송 회귀, 과산포 확인과 준포아송·강건 표준오차, 음이항 회귀를 차례로 해 봅니다. "
           "셀을 위에서부터 차례로 실행하세요. 설명과 출력 읽는 법은 사이트의 실습 12에 있습니다."),
    notes=NOTES)

# ================================================================ 대조 블록: 글에 적은 숫자 확인
if __name__ == "__main__":
    import warnings
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats

    ns = nb.ns
    cgd_, pt_ = ns["cgd"], ns["pt"]
    pois_, pr_, pq_, nb_ = ns["pois"], ns["pois_r"], ns["pois_q"], ns["nb2"]
    t_ = "treat[T.rIFN-g]"
    n_ok = 0

    def same(label, a, b):
        global n_ok
        assert a == b, f"{label}: {a!r} != {b!r}"
        n_ok += 1

    f1 = lambda x: f"{x:.1f}"
    f2 = lambda x: f"{x:.2f}"
    f3 = lambda x: f"{x:.3f}"
    # 글과 노트북 안내에서 번호로 가리키는 셀
    cited = {1: "패키지 불러오기와 버전 확인", 3: "코딩 확인", 4: "환자 한 명 = 한 행으로 요약",
             6: "군별 발생률과 정확 신뢰구간", 9: "포아송 회귀 (오프셋 = log 인년)", 10: "발생률비와 95% 신뢰구간",
             11: "과산포 확인", 13: "강건(샌드위치) 표준오차", 14: "준포아송: 표준오차에 √φ 곱하기",
             15: "음이항 회귀 (NB2)", 16: "네 방법 비교, AIC와 우도비 검정", 17: "0건 환자 수 예측 비교",
             18: "표준화로 발생률과 발생률차 구하기"}
    for k, tt in cited.items():
        same(f"cell {k} title", nb.cells[k - 1].title, tt)
    # 대응표 (12장 b 절 COPD 예는 ch12의 숫자, 이 실습 열은 아래에서 확인)
    same("events, PY", (int(pt_["n_inf"].sum()), f1(pt_["py"].sum())), (76, "102.6"))
    same("patients by arm", pt_["treat"].value_counts().sort_index().tolist(), [65, 63])
    same("COPD events, PY (ch12 b)", (554 + 998, f"{940.0 + 1563.6:.1f}"), (1552, "2503.6"))
    same("Pearson/df, alpha", (f2(pois_.pearson_chi2 / pois_.df_resid), f2(nb_.params["alpha"])), ("1.39", "0.72"))
    lo, hi = np.exp(nb_.conf_int().loc[t_])
    same("NB IRR (CI)", (f2(np.exp(nb_.params[t_])), f2(lo), f2(hi)), ("0.36", "0.20", "0.66"))
    same("crude IRR", f2(ns["irr"]), "0.35")
    # a 절
    same("status 0 rows", int((cgd_["status"] == 0).sum()), 127)
    last = cgd_.groupby("id").tail(1)
    same("patient whose follow-up ended with an infection", last.loc[last["status"] == 1, "id"].tolist(), [87])
    same("patient 1, 2, 3 rows", [int((cgd_["id"] == i).sum()) for i in (1, 2, 3)], [3, 8, 1])
    same("patient 1 py", f2(pt_.loc[pt_["id"] == 1, "py"].iloc[0]), "1.13")
    same("follow-up days min, max, mean", (int(pt_["days"].min()), int(pt_["days"].max()),
                                           f"{pt_['days'].mean():.0f}"), (91, 439, "293"))
    n0 = int((pt_["n_inf"] == 0).sum())
    same("no infection", (n0, f"{n0 / 128:.0%}", 128 - n0), (84, "66%", 44))
    vc = pt_["n_inf"].value_counts()
    same("1 infection, 2+ infections, max", (int(vc[1]), int(vc[vc.index >= 2].sum()), int(pt_["n_inf"].max())),
         (27, 17, 7))
    same("steroids users", int(cgd_.groupby("id")["steroids"].first().sum()), 3)
    # b 절
    rt_ = ns["rt"]
    same("placebo events, PY", (int(rt_.loc["placebo", "events"]), f2(rt_.loc["placebo", "py"])), (56, "50.72"))
    same("rIFN-g events, PY", (int(rt_.loc["rIFN-g", "events"]), f2(rt_.loc["rIFN-g", "py"])), (20, "51.89"))
    same("placebo CI distance", (f1(rt_.loc["placebo", "rate"] - rt_.loc["placebo", "lower"]),
                                 f1(rt_.loc["placebo", "upper"] - rt_.loc["placebo", "rate"])), ("27.0", "33.0"))
    same("relative SE", (f"{1 / np.sqrt(20):.0%}", f"{1 / np.sqrt(56):.0%}"), ("22%", "13%"))
    same("observed share", f3(20 / 76), "0.263")
    same("exact P", f"{ns['res'].pvalue:.1e}", "2.6e-05")
    same("Wald z", f2(np.log(ns["irr"]) / ns["se"]), "-4.04")
    anyinf = pt_.groupby("treat", observed=True)["n_inf"].agg(lambda s: int((s > 0).sum()))
    same("patients with >=1 infection", anyinf.tolist(), [30, 14])
    same("risk of >=1 infection", (f"{30 / 65:.1%}", f"{14 / 63:.1%}", f2((14 / 63) / (30 / 65))),
         ("46.2%", "22.2%", "0.48"))
    same("mean follow-up by arm", [f"{x:.0f}" for x in pt_.groupby("treat", observed=True)["days"].mean()],
         ["285", "301"])
    # c 절
    same("Pearson chi2", f1(pois_.pearson_chi2), "170.5")
    same("Df Residuals", int(pois_.df_resid), 123)
    same("pois treat IRR, z", (f3(np.exp(pois_.params[t_])), f2(pois_.tvalues[t_])), ("0.362", "-3.88"))
    same("age per year, per 10 years", (f3(np.exp(pois_.params["age"])), f2(np.exp(10 * pois_.params["age"]))),
         ("0.964", "0.69"))
    same("pois CI", [f3(x) for x in np.exp(pois_.conf_int().loc[t_])], ["0.217", "0.605"])
    m_exp = smf.glm(ns["form"], data=pt_, family=sm.families.Poisson(), exposure=pt_["py"]).fit()
    same("exposure=py gives the same fit as offset=log(py)", bool(np.allclose(m_exp.params, pois_.params)), True)
    same("sqrt(Pearson/df)", f2(np.sqrt(pois_.pearson_chi2 / pois_.df_resid)), "1.18")
    same("robust/model SE", f2(pr_.bse[t_] / pois_.bse[t_]), "1.18")
    m_by = pt_.groupby("treat", observed=True)["n_inf"].agg(["mean", "var"])
    same("mean, variance by arm", [f2(x) for x in m_by.to_numpy().ravel()], ["0.86", "1.81", "0.32", "0.45"])
    same("quasi CI", [f3(x) for x in np.exp(pq_.conf_int().loc[t_])], ["0.198", "0.663"])
    # d 절
    same("NB function value", (f2(-nb_.llf), f"{-nb_.llf / 128:.4f}"), ("122.28", "0.9553"))
    same("NB LL-Null", f2(nb_.llnull), "-130.90")
    same("NB treat", (f3(np.exp(nb_.params[t_])), f"{nb_.pvalues[t_]:.5f}", f3(nb_.bse[t_])),
         ("0.364", "0.00097", "0.306"))
    same("NB CI log", [f3(x) for x in nb_.conf_int().loc[t_]], ["-1.610", "-0.410"])
    same("NB CI", [f3(x) for x in np.exp(nb_.conf_int().loc[t_])], ["0.200", "0.664"])
    al = nb_.params["alpha"]
    same("alpha, sqrt, 1+alpha, theta, P", (f3(al), f2(np.sqrt(al)), f2(1 + al), f2(1 / al),
                                            f3(nb_.pvalues["alpha"])), ("0.715", "0.85", "1.72", "1.40", "0.047"))
    same("LR P (halved)", f"{stats.chi2.sf(2 * (nb_.llf - pois_.llf), 1) / 2:.4f}", "0.0010")
    same("pois llf", f2(pois_.llf), "-127.05")
    same("AIC difference of the printed values", f1(round(pois_.aic, 1) - round(nb_.aic, 1)), "7.5")
    same("robust CI", [f3(x) for x in np.exp(pr_.conf_int().loc[t_])], ["0.198", "0.664"])
    same("GLM NB SE vs discrete NB SE", (f3(ns["nb_glm"].bse[t_]), f3(nb_.bse[t_])), ("0.309", "0.306"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g1 = smf.glm(ns["form"], data=pt_, family=sm.families.NegativeBinomial(),
                     offset=np.log(pt_["py"])).fit()
    same("GLM NB with alpha = 1, treat coef", f3(g1.params[t_]), "-1.009")
    r0_, r1_ = ns["r0"].mean(), ns["r1"].mean()
    same("standardized rates per PY", (f2(r0_), f2(r1_)), ("1.06", "0.38"))
    same("standardized ratio = exp(NB coef)", abs(r1_ / r0_ - np.exp(nb_.params[t_])) < 1e-9, True)
    same("crude rate difference per 100 PY", f1(rt_.loc["rIFN-g", "rate"] - rt_.loc["placebo", "rate"]), "-71.9")
    # 과제 1
    ct = pd.crosstab(pt_["propylac"], pt_["treat"])
    same("no prophylaxis: patients", (int(ct.loc[0].sum()), int(ct.loc[0, "placebo"]), int(ct.loc[0, "rIFN-g"])),
         (17, 10, 7))
    g = pt_.groupby("propylac")[["n_inf", "py"]].sum()
    same("no prophylaxis: events, PY", (int(g.loc[0, "n_inf"]), f1(g.loc[0, "py"])), (14, "13.1"))
    same("prophylaxis: patients, events, PY", (int(ct.loc[1].sum()), int(g.loc[1, "n_inf"]), f1(g.loc[1, "py"])),
         (111, 62, "89.5"))
    same("crude rates by propylac per 100 PY", (f1(100 * g.loc[0, "n_inf"] / g.loc[0, "py"]),
                                                 f1(100 * g.loc[1, "n_inf"] / g.loc[1, "py"])), ("106.5", "69.3"))
    same("robust propylac CI", [f2(x) for x in np.exp(pr_.conf_int().loc["propylac"])], ["0.29", "1.20"])
    # 과제 2
    co = ns["coefs"]
    same("intercept shift = ln(365.25)", f"{co.loc['Intercept', 'ln(PY)'] - co.loc['Intercept', 'ln(days)']:.4f}",
         "5.9006")
    same("rate per day = rate per year / 365.25", (f"{np.exp(co.loc['Intercept', 'ln(days)']):.5f}",
                                                   f"{np.exp(co.loc['Intercept', 'ln(PY)']) / 365.25:.5f}"),
         ("0.00674", "0.00674"))
    same("e^intercept per year", f2(np.exp(co.loc["Intercept", "ln(PY)"])), "2.46")
    for v in ("treat[T.rIFN-g]", "inherit[T.autosomal]", "age", "propylac"):
        assert abs(co.loc[v, "ln(PY)"] - co.loc[v, "ln(days)"]) < 1e-6, v
        n_ok += 1
    same("no-offset IRR", f3(np.exp(co.loc[t_, "none"])), "0.387")
    same("mean counts per patient", (f2(56 / 65), f2(20 / 63)), ("0.86", "0.32"))
    # 과제 3
    mh, mc = ns["m_hc0"], ns["m_cl"]
    same("row-level coefficient = patient-level", abs(mh.params[t_] - pois_.params[t_]) < 1e-6, True)
    same("row-level model-based SE = patient-level", abs(ns["glm_row"].fit().bse[t_] - pois_.bse[t_]) < 1e-6, True)
    same("colleague's CI", [f2(x) for x in np.exp(mh.conf_int().loc[t_])], ["0.22", "0.60"])
    same("cluster CI", [f3(x) for x in np.exp(mc.conf_int().loc[t_])], ["0.196", "0.670"])
    m_nc = ns["glm_row"].fit(cov_type="cluster", cov_kwds={"groups": ns["rows"]["id"], "use_correction": False})
    same("cluster without small-sample correction = patient-level robust", abs(m_nc.bse[t_] - pr_.bse[t_]) < 1e-6,
         True)
    same("rows: max events per row", int(ns["rows"]["status"].max()), 1)
    print(f"VERIFY: {n_ok} checks of numbers quoted in content/lab12.html passed")
