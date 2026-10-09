"""실습 18 · 중단시계열분석과 이중차분법 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab18.py && python3 gen/lab_lab18.py
      (LABDEBUG=1 prints every cell's text output, for writing marks)

자료: pub/data/sedative_district_month.csv (gen/data_lab18.py가 18장의 모의자료를 그대로 내보낸 것).
끝의 대조 블록에서 실습 결과를 gen/_ch18_nums.json(18장 본문 숫자)과 맞춘다.
"""
import html as _html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab18")
DEBUG = os.environ.get("LABDEBUG")


def save(name, c, marks=None, dfmarks=None, **kw):
    """marks: on the text output (<pre>); dfmarks: on the DataFrame table. '~' prefix = regex."""
    if DEBUG:
        print(f"\n===== {name} (셀 {c.n}) =====\n{c.stdout}{c.value_repr or ''}")
        if c.warns:
            print("WARN:", c.warns)
    h = nb.html(c, **kw)

    def put(seg, mk):
        for sub, n in mk.items():
            if sub.startswith("~"):
                m = re.search(sub[1:], seg)
                j = m.end() if m else -1
            else:
                e = _html.escape(sub)
                j = seg.find(e)
                j = j + len(e) if j >= 0 else -1
            if j < 0:
                raise ValueError(f"{name}: mark text not found: {sub!r}")
            seg = seg[:j] + f'<span class="mk">{n}</span>' + seg[j:]
        return seg

    if marks:
        i = h.index('<div class="cell-out">')
        j = h.index("<pre>", i)
        k = h.index("</pre>", j)
        h = h[:j] + put(h[j:k], marks) + h[k:]
    if dfmarks:
        j = h.index('<div class="df-wrap">')
        k = h.index("</table>", j)
        h = h[:j] + put(h[j:k], dfmarks) + h[k:]
    nb.save_fragment(name, h)


# ================================================================== 가. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
raw = pd.read_csv(BASE + "sedative_district_month.csv",
                  storage_options=UA)
print(raw.shape)
raw.head(3)
''', title="자료 불러오기")
save("lab18_load", c, marks={"(3600, 5)": 1}, dfmarks={"2015-01-01": 2, "747": 3})

c = nb.cell('''
print(raw["unit"].nunique(), "개 시군구,",
      raw["month"].nunique(), "개월")
print(raw["month"].min(), "~", raw["month"].max())

units = raw.groupby("unit").agg(treat=("treat", "first"),
                                months=("month", "size"),
                                mean_pat=("n_pat", "mean"))
print(units["treat"].value_counts())
units.groupby("treat")["mean_pat"].describe().round(0)
''', title="자료 구조 확인")
save("lab18_structure", c, marks={"60 개 시군구, 60 개월": 1, "~0    40": 2, "~1    20": 3},
     dfmarks={"30076.0": 4})

c = nb.cell('''
d = raw.copy()
d["date"] = pd.to_datetime(d["month"])
d["time"] = ((d["date"].dt.year - 2015) * 12
             + d["date"].dt.month)              # 1, 2, ..., 60
d["post"] = (d["time"] > 36).astype(int)        # 2018년 1월부터 1
d["time_after"] = np.where(d["post"] == 1, d["time"] - 36, 0)
d["rate"] = d["n_sed"] / d["n_pat"] * 1000      # 1,000명당

cols = ["month", "time", "post", "time_after",
        "n_pat", "n_sed", "rate"]
d.loc[d["unit"] == 1, cols].iloc[[0, 35, 36, 37, 59]].round(2)
''', title="시간 변수와 처방률 만들기")
save("lab18_timevars", c, dfmarks={"2017-12-01": 1, "2018-01-01": 2, "93.38": 3})

c = nb.cell('''
ag = (d.groupby(["treat", "time", "post", "time_after"])
        [["n_sed", "n_pat"]].sum().reset_index())
ag["rate"] = ag["n_sed"] / ag["n_pat"] * 1000

its = ag[ag["treat"] == 1].reset_index(drop=True)    # 시범 지역
comp = ag[ag["treat"] == 0].reset_index(drop=True)   # 비교 지역
print(len(ag), len(its), len(comp))
its.iloc[[0, 1, 35, 36, 37, 59]].round(2)
''', title="지역을 합친 월별 계열 만들기")
save("lab18_series", c, marks={"120 60 60": 1}, dfmarks={"22725": 2, "96.17": 3, "81.68": 4})

c = nb.cell('''
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(its["time"], its["rate"], "o-", ms=3,
        label="Pilot districts")
ax.plot(comp["time"], comp["rate"], "s-", ms=3,
        label="Comparison districts")
ax.axvline(36.5, color="gray", ls="--")           # 시행 시점
ax.set_xticks([1, 13, 25, 37, 49])
ax.set_xticklabels(["2015-01", "2016-01", "2017-01",
                    "2018-01", "2019-01"])
ax.set_xlabel("Month")
ax.set_ylabel("Patients prescribed per 1,000")
ax.legend()
plt.tight_layout()
''', title="두 계열의 그림")
save("lab18_plot", c)

c = nb.cell('''
from scipy import stats

for name, s in [("pilot", its), ("comparison", comp)]:
    pre = s.loc[s["post"] == 0, "rate"]
    po = s.loc[s["post"] == 1, "rate"]
    tt = stats.ttest_ind(po, pre, equal_var=False)
    ci = tt.confidence_interval()
    print(f"{name:<10} before {pre.mean():.2f}  after {po.mean():.2f}"
          f"  diff {po.mean() - pre.mean():.2f}"
          f" ({ci.low:.2f}, {ci.high:.2f})  p = {tt.pvalue:.0e}")
''', title="전후 평균만 비교하면")
save("lab18_naive", c, marks={"diff -17.46 (-19.25, -15.67)": 1, "diff -10.06 (-11.40, -8.72)": 2})

# ================================================================== 나. 중단시계열분석
S2 = len(nb.cells) + 1
c = nb.cell('''
import statsmodels.formula.api as smf
from statsmodels.stats.stattools import durbin_watson

f = "rate ~ time + post + time_after"
fit0 = smf.ols(f, data=its).fit()
print(fit0.summary().tables[1])
print("Durbin-Watson:", round(durbin_watson(fit0.resid), 3))
''', title="분절회귀 (보통의 최소제곱)")
save("lab18_ols", c, marks={"~Intercept\\s+95\\.6211": 1, "~time\\s+-0\\.2421": 2, "~post\\s+-7\\.3707": 3,
                            "~time_after\\s+-0\\.2263": 4, "Durbin-Watson: 0.945": 5})

c = nb.cell('''
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

r = acf(fit0.resid, nlags=12)
print("lag 1-6 :", np.round(r[1:7], 3))
print("lag 7-12:", np.round(r[7:13], 3))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].plot(its["time"], fit0.resid, "o-", ms=3)
axes[0].axhline(0, color="gray")
axes[0].axvline(36.5, color="gray", ls="--")
axes[0].set_xlabel("Month")
axes[0].set_ylabel("Residual")
plot_acf(fit0.resid, lags=12, zero=False, ax=axes[1])
axes[1].set_xlabel("Lag (months)")
plt.tight_layout()
''', title="잔차의 자기상관 그림")
save("lab18_acf", c, marks={"lag 1-6 : [ 0.513": 1, "~lag 7-12: \\[[^\\]]*\\]": 2})

c = nb.cell('''
fit = smf.ols(f, data=its).fit(cov_type="HAC",
                               cov_kwds={"maxlags": 3})
print(fit.summary())
''', title="Newey-West(HAC) 표준오차로 다시 적합")
# summary()의 Date·Time 줄은 실행한 시각이라 돌릴 때마다 달라진다. 다시 돌려도 같은 조각이 나오도록
# 처음 실행한 시각(2026-10-03 16:32:52)으로 고정한다. 글자 수가 같아 표의 모양은 그대로다.
c.stdout = re.sub(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}", r"\g<1>Sat, 03 Oct 2026", c.stdout)
c.stdout = re.sub(r"(Time:\s+)\d{2}:\d{2}:\d{2}", r"\g<1>16:32:52", c.stdout)
assert "Sat, 03 Oct 2026" in c.stdout and "16:32:52" in c.stdout
save("lab18_hac", c, marks={"~Covariance Type:\\s+HAC": 1, "~time\\s+-0\\.2421\\s+0\\.026[^\\n]*": 2,
                            "~post\\s+-7\\.3707\\s+1\\.117[^\\n]*": 3, "~time_after\\s+-0\\.2263\\s+0\\.063[^\\n]*": 4,
                            "~Durbin-Watson:\\s+0\\.945": 5, "Notes:": 6})

c = nb.cell('''
print(fit.t_test("post + 12 * time_after = 0"))   # 12개월째의 효과
print(fit.t_test("time + time_after = 0"))        # 시행 후 기울기

b = fit.params
cf12 = b["Intercept"] + b["time"] * 48           # 반사실 (time = 48)
eff12 = b["post"] + 12 * b["time_after"]         # 12개월째의 효과
print(f"counterfactual: {cf12:.2f}")
print(f"fitted        : {cf12 + eff12:.2f}")
print(f"relative      : {100 * eff12 / cf12:.1f}%")
''', title="시행 12개월째의 효과와 시행 후 기울기")
save("lab18_eff12", c, marks={"~c0\\s+-10\\.0860[^\\n]*": 1, "~c0\\s+-0\\.4684[^\\n]*": 2, "counterfactual: 84.00": 3, "fitted        : 73.91": 4,
                              "relative      : -12.0%": 5})

c = nb.cell('''
its["fitted"] = fit.predict(its)
its["cf"] = fit.predict(its.assign(post=0, time_after=0))
its["effect"] = its["fitted"] - its["cf"]

pre, po = its[its["post"] == 0], its[its["post"] == 1]
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(its["time"], its["rate"], "o", ms=3, color="gray",
        label="Observed")
ax.plot(pre["time"], pre["fitted"], color="C0", label="Fitted")
ax.plot(po["time"], po["fitted"], color="C0")
ax.plot(po["time"], po["cf"], "--", color="C1",
        label="Counterfactual")
ax.axvline(36.5, color="gray", ls="--")
ax.set_xlabel("Month (1 = January 2015)")
ax.set_ylabel("Patients prescribed per 1,000")
ax.legend()
plt.tight_layout()
its.loc[its["time"].isin([37, 48, 60]),
        ["time", "rate", "fitted", "cf", "effect"]].round(2)
''', title="반사실 선 그리기")
save("lab18_cf", c, dfmarks={"86.66": 1, "-10.09": 2, "-12.80": 3})

c = nb.cell('''
its["sin12"] = np.sin(2 * np.pi * its["time"] / 12)
its["cos12"] = np.cos(2 * np.pi * its["time"] / 12)

fit_s = smf.ols(f + " + sin12 + cos12", data=its).fit(
    cov_type="HAC", cov_kwds={"maxlags": 3})
print(fit_s.summary().tables[1])
print("Durbin-Watson:", round(durbin_watson(fit_s.resid), 3))
print("residual SD  :", round(np.sqrt(fit0.scale), 2), "->",
      round(np.sqrt(fit_s.scale), 2))
''', title="계절 항을 넣은 모형")
save("lab18_season", c, marks={"~post\\s+-7\\.7717[^\\n]*": 1, "~time_after\\s+-0\\.2153[^\\n]*": 2,
                               "~cos12\\s+0\\.9506": 3, "Durbin-Watson: 1.296": 4, "residual SD  : 1.52 -> 1.26": 5})

c = nb.cell('''
import statsmodels.api as sm

X = sm.add_constant(its[["time", "post", "time_after"]])
gls = sm.GLSAR(its["rate"], X, rho=1).iterative_fit(maxiter=50)
print("rho:", np.round(gls.model.rho, 3))

def row(res):
    ci = res.conf_int()
    return [res.params["post"], *ci.loc["post"],
            res.params["time_after"], *ci.loc["time_after"]]

tab = pd.DataFrame(
    {"OLS": row(fit0), "Newey-West (lag 3)": row(fit),
     "AR(1) errors": row(gls), "Seasonal + Newey-West": row(fit_s)},
    index=["level", "lower", "upper", "slope", "lower", "upper"]).T
tab.round(3)
''', title="AR(1) 오차 모형과 네 가지 방법의 비교")
save("lab18_methods", c, marks={"rho: [0.527]": 1}, dfmarks={"-8.991": 2, "-9.559": 3, "-7.216": 4, "-7.772": 5})

# ================================================================== 다. 대조군이 있는 중단시계열
S3 = len(nb.cells) + 1
c = nb.cell('''
diff = its[["time", "post", "time_after"]].copy()
diff["rate"] = its["rate"] - comp["rate"]       # 시범 - 비교

def seg(data):
    return smf.ols(f, data=data).fit(cov_type="HAC",
                                     cov_kwds={"maxlags": 3})

fits = {"pilot": seg(its), "comparison": seg(comp),
        "difference": seg(diff)}
tab3 = pd.DataFrame({k: v.params for k, v in fits.items()})
tab3.loc["effect at 12 months"] = (tab3.loc["post"]
                                   + 12 * tab3.loc["time_after"])
tab3.round(3)
''', title="세 계열에 같은 분절회귀 적합")
save("lab18_three", c, dfmarks={"-2.528": 1, "-4.843": 2, "-0.198": 3, "-7.223": 4})

c = nb.cell('''
fit_d = fits["difference"]
print(fit_d.summary().tables[1])
print(fit_d.t_test("post + 12 * time_after = 0"))
print("Durbin-Watson:", round(durbin_watson(fit_d.resid), 2),
      " residual SD:", round(np.sqrt(fit_d.scale), 2))
''', title="차이의 계열 분석 결과")
save("lab18_diff", c, marks={"~time\\s+-0\\.0026[^\\n]*": 1, "~post\\s+-4\\.8426[^\\n]*": 2,
                             "~time_after\\s+-0\\.1984[^\\n]*": 3, "~c0\\s+-7\\.2230[^\\n]*": 4,
                             "Durbin-Watson: 1.76": 5})

c = nb.cell('''
diff["fitted"] = fit_d.predict(diff)
diff["cf"] = fit_d.predict(diff.assign(post=0, time_after=0))
pre, po = diff[diff["post"] == 0], diff[diff["post"] == 1]

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(diff["time"], diff["rate"], "o", ms=3, color="gray",
        label="Observed difference")
ax.plot(pre["time"], pre["fitted"], color="C0", label="Fitted")
ax.plot(po["time"], po["fitted"], color="C0")
ax.plot(po["time"], po["cf"], "--", color="C1",
        label="Counterfactual")
ax.axvline(36.5, color="gray", ls="--")
ax.set_xlabel("Month (1 = January 2015)")
ax.set_ylabel("Pilot minus comparison, per 1,000")
ax.legend()
plt.tight_layout()
''', title="차이의 계열 그림")
save("lab18_diffplot", c)

c = nb.cell('''
cols = ["treat", "time", "post", "time_after", "rate"]
both = pd.concat([its[cols], comp[cols]], ignore_index=True)
both = both.rename(columns={"treat": "group"})   # 시범 지역 = 1

f_c = "rate ~ (time + post + time_after) * group"
cits = smf.ols(f_c, data=both).fit(
    cov_type="cluster", cov_kwds={"groups": both["time"]})
print(cits.summary().tables[1])
print("ordinary SE:")
print(smf.ols(f_c, data=both).fit().bse[
    ["post:group", "time_after:group"]].round(3))
''', title="교호작용 모형으로 한 번에 적합")
save("lab18_cits", c, marks={"~post\\s+-2\\.5281[^\\n]*": 1, "~time:group\\s+-0\\.0026[^\\n]*": 2,
                             "~post:group\\s+-4\\.8426\\s+0\\.429[^\\n]*": 3,
                             "~time_after:group\\s+-0\\.1984[^\\n]*": 4, "~post:group\\s+1\\.089": 5})

# ================================================================== 라. 이중차분법
S4 = len(nb.cells) + 1
c = nb.cell('''
m = both.groupby(["group", "post"])["rate"].mean().unstack("post")
m.columns = ["before", "after"]
m["change"] = m["after"] - m["before"]
m.loc["difference"] = m.loc[1] - m.loc[0]
m.round(2)
''', title="2×2 평균표와 이중차분 추정값")
save("lab18_did22", c, dfmarks={"-17.46": 1, "-10.06": 2, "11.48": 3, "4.08": 4, "-7.40": 5})

c = nb.cell('''
panel = d.copy()
panel["w"] = panel.groupby("unit")["n_pat"].transform("mean")

did = smf.wls("rate ~ treat * post", data=panel,
              weights=panel["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": panel["unit"]})
print(did.summary().tables[1])
print("rows:", int(did.nobs), " clusters:", panel["unit"].nunique())
''', title="회귀로 구하는 이중차분 추정값 (군집 강건 표준오차)")
save("lab18_didreg", c, marks={"~Intercept\\s+79\\.6646": 1, "~treat\\s+11\\.4771": 2, "~post\\s+-10\\.0615": 3,
                               "~treat:post\\s+-7\\.4015[^\\n]*": 4, "rows: 3600  clusters: 60": 5})

c = nb.cell('''
pre = panel[panel["post"] == 0]
pt = smf.wls("rate ~ treat * time", data=pre,
             weights=pre["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": pre["unit"]})
print(pt.summary().tables[1])
lo, hi = pt.conf_int().loc["treat:time"]
print(f"x 30 months: {30 * lo:.2f} ~ {30 * hi:.2f}")
''', title="시행 전 추세의 차이")
save("lab18_pretrend", c, marks={"~treat:time\\s+-0\\.0026[^\\n]*": 1, "~x 30 months: [^\\n]*": 2})

c = nb.cell('''
panel["half"] = (panel["time"] - 1) // 6 - 6   # -6..-1 전, 0..3 후
terms = []
for h in range(-6, 4):
    if h == -1:                 # 시행 직전 반기 = 기준
        continue
    name = f"lead{-h}" if h < 0 else f"lag{h + 1}"
    panel[name] = ((panel["half"] == h)
                   & (panel["treat"] == 1)).astype(int)
    terms.append(name)

f_es = "rate ~ " + " + ".join(terms) + " + C(unit) + C(time)"
es = smf.wls(f_es, data=panel, weights=panel["w"]).fit(
    cov_type="cluster", cov_kwds={"groups": panel["unit"]})
ev = es.conf_int().loc[terms]
ev.columns = ["lower", "upper"]
ev.insert(0, "coef", es.params[terms])
print(ev.round(2))
''', title="사건연구 모형")
save("lab18_event", c, marks={"~lead6[^\\n]*": 1, "~lead2[^\\n]*": 2, "~lag1[^\\n]*": 3, "~lag4[^\\n]*": 4})

c = nb.cell('''
leads = ", ".join(f"{t} = 0" for t in terms if "lead" in t)
jt = es.wald_test(leads, scalar=True)        # 시행 전 계수 결합 검정
print(f"joint test of 5 leads: chi2 = {float(jt.statistic):.2f},"
      f" p = {float(jt.pvalue):.2f}")

x = [h for h in range(-6, 4) if h != -1]       # 반기 번호
err = [ev["coef"] - ev["lower"], ev["upper"] - ev["coef"]]

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.errorbar(x, ev["coef"], yerr=err, fmt="o", capsize=3)
ax.plot(-1, 0, "o", color="gray")              # 기준 반기
ax.axhline(0, color="gray")
ax.axvline(-0.5, color="gray", ls="--")
ax.set_xlabel("Half-year relative to the programme start")
ax.set_ylabel("Difference vs. last pre half-year")
plt.tight_layout()
''', title="시행 전 계수의 결합 검정과 사건연구 그림")
save("lab18_eventplot", c, marks={"p = 0.76": 1})

N_MAIN = len(nb.cells)

# ================================================================== 과제
c = nb.cell('''
short = its[(its["time"] > 24) & (its["time"] <= 48)]
fit_12 = smf.ols(f, data=short).fit(cov_type="HAC",
                                    cov_kwds={"maxlags": 3})
print(len(short), "months")
print(fit_12.summary().tables[1])
''', title="과제 1 정답. 시행 전후 12개월씩만 쓴 분절회귀")
save("lab18_hw1", c)

c = nb.cell('''
keep = its[~its["time"].isin([37, 38, 39])]     # 2018년 1-3월 제외
fit_ph = smf.ols(f, data=keep).fit(cov_type="HAC",
                                   cov_kwds={"maxlags": 3})
print(len(keep), "months")
print(fit_ph.summary().tables[1])
print(fit_ph.t_test("post + 12 * time_after = 0"))
''', title="과제 2 정답. 적응 기간 3개월을 뺀 분석")
save("lab18_hw2", c)

c = nb.cell('''
# (1) 가중치 없는 이중차분
did_u = smf.ols("rate ~ treat * post", data=panel).fit(
    cov_type="cluster", cov_kwds={"groups": panel["unit"]})
lo, hi = did_u.conf_int().loc["treat:post"]
print(f"unweighted DID: {did_u.params['treat:post']:.2f}"
      f" ({lo:.2f}, {hi:.2f})")

# (2) 시군구마다 변화량 하나
chg = (panel.groupby(["unit", "treat", "w", "post"])["rate"]
            .mean().unstack("post").reset_index())
chg["chg"] = chg[1] - chg[0]
print(chg.groupby("treat")["chg"].describe().round(1)[
    ["count", "mean", "std", "min", "max"]])
fit_chg = smf.wls("chg ~ treat", data=chg,
                  weights=chg["w"]).fit(cov_type="HC1")
lo, hi = fit_chg.conf_int().loc["treat"]
print(f"district-level: {fit_chg.params['treat']:.2f}"
      f" ({lo:.2f}, {hi:.2f})  n = {int(fit_chg.nobs)}")
''', title="과제 3 정답. 가중치 없는 분석과 시군구별 변화량")
save("lab18_hw3", c)

print("lab18: cells", len(nb.cells), "(main", N_MAIN, "+ homework", len(nb.cells) - N_MAIN, ")")
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ================================================================== 노트북
nb.save_ipynb(
    "실습 18. 중단시계열분석과 이중차분법",
    intro=("사회약학 연구방법 노트의 '실습 18 중단시계열분석과 이중차분법'을 따라 하는 노트북입니다. "
           "18장의 예제(65세 이상 외래 환자의 수면진정제 처방 관리 시범사업, 가상 자료)를 시군구 × 월 자료에서 "
           "직접 집계해 분절회귀, 대조군이 있는 중단시계열, 이중차분법으로 분석합니다. "
           "셀을 위에서부터 차례로 실행하세요. 자료는 사이트에서 내려받으므로 인터넷 연결이 필요합니다."),
    notes={
        1: "## 가. 실습 데이터 준비\n\n시군구 × 월 자료를 읽고, 시간 변수와 처방률을 만들고, 지역을 합친 월별 계열을 만듭니다.",
        S2: "## 나. 중단시계열분석\n\n시범 지역의 월별 계열(its)에 분절회귀를 적합하고, 자기상관을 확인한 뒤 Newey-West 표준오차를 씁니다.",
        S3: "## 다. 대조군이 있는 중단시계열\n\n비교 지역의 계열과 차이의 계열(시범 − 비교)에도 같은 모형을 적합합니다.",
        S4: "## 라. 이중차분법\n\n2×2 평균표로 먼저 구하고, 시군구 × 월 자료의 회귀로 같은 값과 군집 강건 표준오차를 구합니다.",
        N_MAIN + 1: ("## 과제 정답\n\n**과제 1.** 시범 지역의 계열에서 시행 전 12개월과 시행 후 12개월(time 25–48)만 남겨 "
                     "나 절의 분절회귀(Newey-West, 시차 3)를 다시 적합하고, 수준 변화와 기울기 변화의 95% 신뢰구간을 60개월 분석과 비교하세요."),
        N_MAIN + 2: ("**과제 2.** 시행 직후 3개월(2018년 1–3월, time 37–39)을 적응 기간으로 보고 빼고 분석하세요. "
                     "time_after는 그대로 둡니다. 수준 변화, 기울기 변화, 12개월째의 효과를 구하세요."),
        N_MAIN + 3: ("**과제 3.** (1) 가중치 없이 이중차분 추정값과 95% 신뢰구간을 구하세요. "
                     "(2) 시군구마다 '시행 후 평균 − 시행 전 평균'을 하나씩 구해 60개의 변화량으로 두 지역을 비교하세요."),
    })

# ================================================================== 본문 숫자와의 대조 (18장 = gen/_ch18_nums.json)
if __name__ == "__main__":
    import warnings
    J = json.load(open(os.path.join(HERE, "_ch18_nums.json")))
    ns = nb.ns
    n_ok = 0

    def eq(a, b, tol=1e-6, what=""):
        global n_ok
        assert abs(float(a) - float(b)) < tol, f"MISMATCH {what}: lab {a} vs chapter {b}"
        n_ok += 1

    its, comp, fit0, fit = ns["its"], ns["comp"], ns["fit0"], ns["fit"]
    # 자료: 월별 분자·분모·율
    assert its["n_sed"].tolist() == J["series"]["cp"] and comp["n_pat"].tolist() == J["series"]["nc"]; n_ok += 1
    eq(np.abs(its["rate"].values - np.array(J["series"]["yp"])).max(), 0, 1e-9, "pilot series")
    eq(np.abs(comp["rate"].values - np.array(J["series"]["yc"])).max(), 0, 1e-9, "comparison series")
    # 가: 전후 평균
    for nm, s in (("pilot", its), ("comp", comp)):
        eq(s.loc[s.post == 0, "rate"].mean(), J["naive"][nm]["pre"], what=nm + " pre")
        eq(s.loc[s.post == 1, "rate"].mean(), J["naive"][nm]["post"], what=nm + " post")
    tt = ns["stats"].ttest_ind(its.loc[its.post == 1, "rate"], its.loc[its.post == 0, "rate"], equal_var=False)
    eq(tt.confidence_interval().low, J["naive"]["pilot"]["lo"], what="naive lo")
    # 나: 분절회귀
    for i, k in enumerate(["Intercept", "time", "post", "time_after"]):
        eq(fit.params[k], J["its"]["b"][i], what="its b " + k)
        eq(fit.bse[k], J["its"]["se"][i], what="its HAC se " + k)
        eq(fit0.bse[k], J["its"]["se_ols"][i], what="its OLS se " + k)
        eq(fit.conf_int().loc[k, 0], J["its"]["ci"][i][0], what="its ci lo " + k)
        eq(fit.conf_int().loc[k, 1], J["its"]["ci"][i][1], what="its ci hi " + k)
    eq(ns["durbin_watson"](fit0.resid), J["its"]["dw"], what="DW")
    eq(np.abs(ns["r"] - np.array(J["its"]["acf"])).max(), 0, 1e-9, "acf")
    t12 = fit.t_test("post + 12 * time_after = 0")
    eq(t12.effect[0], J["its_eff"]["12"]["diff"], what="eff12")
    eq(t12.conf_int()[0][0], J["its_eff"]["12"]["lo"], what="eff12 lo")
    eq(t12.conf_int()[0][1], J["its_eff"]["12"]["hi"], what="eff12 hi")
    eq(ns["cf12"], J["its_eff"]["12"]["cf"], what="cf12")
    eq(ns["eff12"] / ns["cf12"], J["its_eff"]["12"]["rel"], what="rel12")
    ps = fit.t_test("time + time_after = 0")
    eq(ps.effect[0], J["its"]["post_slope"][0], what="post slope")
    eq(ps.conf_int()[0][0], J["its"]["post_slope"][1], what="post slope lo")
    eq(its.loc[its.time == 60, "effect"].iloc[0], J["its_eff"]["24"]["diff"], what="eff24")
    eq(its.loc[its.time == 37, "cf"].iloc[0], J["its_eff"]["1"]["cf"], what="cf at 37")
    fs, A = ns["fit_s"], J["alt"]["Fourier(1 pair)+HAC3"]
    eq(fs.params["post"], A["level"], what="season level"); eq(fs.params["time_after"], A["slope"], what="season slope")
    eq(fs.conf_int().loc["post", 0], A["ci_level"][0], what="season ci"); eq(fs.conf_int().loc["time_after", 1], A["ci_slope"][1], what="season ci")
    eq(ns["durbin_watson"](fs.resid), A["dw"], what="season DW"); eq(np.sqrt(fs.scale), A["sigma"], what="season sigma")
    eq(np.hypot(fs.params["sin12"], fs.params["cos12"]), A["amp"], what="season amplitude")
    eq(np.sqrt(fit0.scale), J["its"]["sigma"], what="sigma")
    g = ns["gls"]
    eq(g.model.rho[0], J["glsar"]["rho"], what="rho")
    for i, k in enumerate(["const", "time", "post", "time_after"]):
        eq(g.params[k], J["glsar"]["b"][i], what="glsar b"); eq(g.bse[k], J["glsar"]["se"][i], what="glsar se")
    eq(g.conf_int().loc["post", 0], J["glsar"]["ci"][2][0], what="glsar ci")
    # 다: 대조군이 있는 중단시계열
    for nm, key in (("pilot", "pilot"), ("comparison", "comp"), ("difference", "diff")):
        fm, CJ = ns["fits"][nm], J["cits"][key]
        for i, k in enumerate(["Intercept", "time", "post", "time_after"]):
            eq(fm.params[k], CJ["b"][i], what=f"cits {nm} b {k}")
            eq(fm.conf_int().loc[k, 0], CJ["ci"][i][0], what=f"cits {nm} lo {k}")
            eq(fm.conf_int().loc[k, 1], CJ["ci"][i][1], what=f"cits {nm} hi {k}")
        e = fm.t_test("post + 12 * time_after = 0")
        eq(e.effect[0], CJ["e12"][0], what=f"cits {nm} e12"); eq(e.conf_int()[0][0], CJ["e12"][1], what="e12 lo")
        eq(e.conf_int()[0][1], CJ["e12"][2], what="e12 hi")
        eq(ns["durbin_watson"](fm.resid), CJ["dw"], what="cits dw"); eq(np.sqrt(fm.scale), CJ["sigma"], what="cits sigma")
    eav = ns["fit_d"].t_test("post + 12.5 * time_after = 0")
    eq(eav.effect[0], J["cits"]["diff"]["eav"][0], what="cits 24-month average")
    ct, S = ns["cits"], J["stack"]
    for a, b_ in (("post:group", "grp:post"), ("time_after:group", "grp:taft"), ("time:group", "grp:time"), ("group", "grp"),
                  ("post", "post"), ("time", "time")):
        eq(ct.params[a], S["params"][b_], what="stack " + a)
        eq(ct.bse[a], S["se_cluster_month"][b_], what="stack cluster se " + a)
    o = ns["smf"].ols(ns["f_c"], data=ns["both"]).fit()
    eq(o.bse["post:group"], S["se_ols"]["grp:post"], what="stack ols se")
    # 라: 이중차분법
    m = ns["m"]
    eq(m.loc[1, "before"], J["did"]["tp"], what="2x2"); eq(m.loc[1, "after"], J["did"]["tpost"], what="2x2")
    eq(m.loc[0, "before"], J["did"]["cp"], what="2x2"); eq(m.loc[0, "after"], J["did"]["cpost"], what="2x2")
    eq(m.loc["difference", "change"], J["did"]["did"], what="DID by hand")
    dd, R = ns["did"], J["did_reg"]
    for i, k in enumerate(["Intercept", "treat", "post", "treat:post"]):
        eq(dd.params[k], R["params"][k], what="did " + k); eq(dd.bse[k], R["se_cluster"][k], what="did se " + k)
        eq(dd.conf_int().loc[k, 0], R["ci_cluster"][i][0], what="did lo"); eq(dd.conf_int().loc[k, 1], R["ci_cluster"][i][1], what="did hi")
    assert int(dd.nobs) == R["nobs"] and ns["panel"]["unit"].nunique() == R["nclu"]; n_ok += 1
    pt, P = ns["pt"], J["pretrend"]
    eq(pt.params["treat:time"], P["b"], what="pretrend"); eq(pt.conf_int().loc["treat:time", 0], P["lo"], what="pretrend lo")
    eq(pt.conf_int().loc["treat:time", 1], P["hi"], what="pretrend hi"); eq(pt.pvalues["treat:time"], P["p"], what="pretrend p")
    ev = ns["ev"]
    for e in J["event"]:
        if e.get("ref"):
            continue
        nm = ("lead%d" % (-e["h"])) if e["h"] < 0 else ("lag%d" % (e["h"] + 1))
        eq(ev.loc[nm, "coef"], e["b"], what="event " + nm); eq(ev.loc[nm, "lower"], e["lo"], what="event lo " + nm)
        eq(ev.loc[nm, "upper"], e["hi"], what="event hi " + nm)
    jt = ns["jt"]
    eq(jt.statistic, J["event_joint"][0], what="joint chi2"); eq(jt.pvalue, J["event_joint"][1], what="joint p")
    # 과제
    A = J["alt"]["12 pre + 12 post only"]
    f12 = ns["fit_12"]
    eq(f12.params["post"], A["level"], what="hw1 level"); eq(f12.conf_int().loc["post", 0], A["ci_level"][0], what="hw1")
    eq(f12.params["time_after"], A["slope"], what="hw1 slope"); eq(f12.conf_int().loc["time_after", 1], A["ci_slope"][1], what="hw1")
    A = J["alt"]["drop first 3 post months"]
    fp = ns["fit_ph"]
    eq(fp.params["post"], A["level"], what="hw2 level"); eq(fp.conf_int().loc["post", 0], A["ci_level"][0], what="hw2")
    eq(fp.params["time_after"], A["slope"], what="hw2 slope"); eq(fp.conf_int().loc["time_after", 1], A["ci_slope"][1], what="hw2")
    du = ns["did_u"]
    eq(du.params["treat:post"], J["did_unw"][0], what="hw3 unweighted"); eq(du.conf_int().loc["treat:post", 0], J["did_unw"][1], what="hw3")
    fc = ns["fit_chg"]
    eq(fc.params["treat"], J["did_chg"]["est"][0], what="hw3 district-level"); eq(fc.conf_int().loc["treat", 0], J["did_chg"]["est"][1], what="hw3")
    eq(fc.conf_int().loc["treat", 1], J["did_chg"]["est"][2], what="hw3")
    print(f"CHECK: {n_ok} comparisons with gen/_ch18_nums.json all agree (tol 1e-6)")

    # ---------------- numbers quoted in the text but not shown in a cell
    smf_, pd_ = ns["smf"], ns["pd"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("VERIFY denominators pilot month 1/60:", its.n_pat.iloc[0], its.n_pat.iloc[-1], " comparison:", comp.n_pat.iloc[0], comp.n_pat.iloc[-1])
        d_ = ns["d"]
        mr = d_[d_.treat == 1].groupby("time")["rate"].mean()
        print("VERIFY mean of district rates (pilot, month 1) %.2f vs pooled %.2f; max abs gap over months %.2f" %
              (mr.iloc[0], its.rate.iloc[0], np.abs(mr.values - its.rate.values).max()))
        print("VERIFY unit size range:", ns["units"].mean_pat.min().round(0), ns["units"].mean_pat.max().round(0))
        for L in (1, 3, 6):
            r_ = fit0.get_robustcov_results(cov_type="HAC", maxlags=L)
            print("VERIFY get_robustcov_results maxlags", L, "SE post", round(float(np.asarray(r_.bse)[2]), 3), type(r_.bse).__name__)
        print("VERIFY HAC level-change CI width ratio vs OLS:",
              round((fit.conf_int().loc["post", 1] - fit.conf_int().loc["post", 0]) / (fit0.conf_int().loc["post", 1] - fit0.conf_int().loc["post", 0]), 2))
        e24 = fit.t_test("post + 24 * time_after = 0")
        print("VERIFY 24-month effect", np.round(e24.effect, 2), np.round(e24.conf_int(), 2), "cf", round(float(its.loc[its.time == 60, "cf"].iloc[0]), 2))
        e1 = fit.t_test("post + 1 * time_after = 0")
        print("VERIFY first-month effect", np.round(e1.effect, 2))
        print("VERIFY p-values HAC:", fit.pvalues.round(5).to_dict())
        print("VERIFY seasonal coefs:", ns["fit_s"].params.round(3).to_dict(), "amp", round(float(np.hypot(ns["fit_s"].params["sin12"], ns["fit_s"].params["cos12"])), 2))
        print("VERIFY season eff12:", np.round(ns["fit_s"].t_test("post + 12 * time_after = 0").effect, 2), np.round(ns["fit_s"].t_test("post + 12 * time_after = 0").conf_int(), 2))
        print("VERIFY GLSAR nobs", int(g.nobs), "bse", g.bse.round(3).to_dict())
        fc_ = ns["fits"]["comparison"]
        print("VERIFY comparison CI:", fc_.conf_int().round(3).values.tolist(), "p", fc_.pvalues.round(4).to_dict())
        print("VERIFY comparison e12:", np.round(fc_.t_test("post + 12 * time_after = 0").conf_int(), 2))
        fd = ns["fit_d"]
        print("VERIFY diff CI:", fd.conf_int().round(3).values.tolist(), "se", fd.bse.round(3).to_dict(), "p", fd.pvalues.round(4).to_dict())
        print("VERIFY diff 24-month average:", np.round(fd.t_test("post + 12.5 * time_after = 0").effect, 2), np.round(fd.t_test("post + 12.5 * time_after = 0").conf_int(), 2))
        print("VERIFY resid corr pilot/comparison:", round(float(np.corrcoef(ns["fits"]["pilot"].resid, fc_.resid)[0, 1]), 2))
        print("VERIFY cits cluster CI:", ct.conf_int().round(3).loc[["post:group", "time_after:group"]].values.tolist())
        pan = ns["panel"]
        nv = smf_.wls("rate ~ treat * post", data=pan, weights=pan["w"]).fit()
        print("VERIFY DID without cluster: SE %.3f CI %s ; cluster SE %.3f" % (nv.bse["treat:post"], nv.conf_int().loc["treat:post"].round(2).tolist(), dd.bse["treat:post"]))
        fe = smf_.wls("rate ~ treat:post + C(unit) + C(time)", data=pan, weights=pan["w"])
        fe_n, fe_c = fe.fit(), fe.fit(cov_type="cluster", cov_kwds={"groups": pan["unit"]})
        print("VERIFY TWFE: est %.4f ordinary SE %.3f cluster SE %.3f" % (fe_c.params["treat:post"], fe_n.bse["treat:post"], fe_c.bse["treat:post"]))
        print("VERIFY between-district SD of pre-period mean rate:", pan[pan.post == 0].groupby(["treat", "unit"])["rate"].mean().groupby("treat").std().round(2).to_dict())
        print("VERIFY DID p:", dd.pvalues["treat:post"], " pretrend p:", round(float(pt.pvalues["treat:time"]), 3))
        print("VERIFY weights vs 2x2: did reg %.6f, by hand %.6f" % (dd.params["treat:post"], m.loc["difference", "change"]))
        print("VERIFY event:", ev.round(2).to_dict("index"))
        print("VERIFY mean of 4 lags:", round(float(ev.loc[["lag1", "lag2", "lag3", "lag4"], "coef"].mean()), 2))
        print("VERIFY hw1:", f12.params.round(3).to_dict(), f12.conf_int().round(3).values.tolist())
        print("VERIFY hw2:", fp.params.round(3).to_dict(), fp.conf_int().round(3).values.tolist(),
              np.round(fp.t_test("post + 12 * time_after = 0").conf_int(), 2))
        chg = ns["chg"]
        print("VERIFY hw3 chg:", chg.groupby("treat")["chg"].agg(["mean", "std", "min", "max"]).round(2).to_dict("index"))
