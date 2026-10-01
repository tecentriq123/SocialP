"""실습 12 · 포아송 회귀와 음이항 회귀 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab12.py
      (LABDEBUG=1 prints every cell's text output, for writing marks)
"""
import html as _html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labkit import Notebook  # noqa: E402

nb = Notebook("lab12")
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

c = nb.cell('''
ci = pois.conf_int()
pd.DataFrame({"IRR": np.exp(pois.params),
              "2.5%": np.exp(ci[0]),
              "97.5%": np.exp(ci[1])}).round(3)
''', title="발생률비와 95% 신뢰구간")
save("lab12_pois_irr", c)

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
save("lab12_robust", c, marks={"model-based SE: 0.262": 1, "robust SE     : 0.309": 2})

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

c = nb.cell('''
fits = {"Poisson": pois, "Poisson, robust SE": pois_r,
        "Negative binomial": nb2}
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
''', title="세 방법 비교, AIC와 우도비 검정")
save("lab12_cmp", c, marks={"264.1": 1, "256.6": 2, "LR = 9.55": 3})

c = nb.cell('''
a = nb2.params["alpha"]
mu_nb = nb2.predict()                 # 환자별 기대 건수
p0_nb = (1 + a * mu_nb) ** (-1 / a)   # 음이항 분포의 P(0건)
print("zeros: observed", obs[0], " Poisson", round(exp_p[0], 1),
      " NB", round(p0_nb.sum(), 1))
''', title="0건 환자 수 예측 비교")
save("lab12_zeros", c, marks={"NB 83.2": 1})

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

if DEBUG:
    print("\ncells:", len(nb.cells))
