"""실습 21 · 비용 자료 분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab21.py && python3 gen/lab_lab21.py
      (NOMARK=1 을 앞에 붙이면 표식 없이 셀 출력만 화면에 찍는다)

자료: pub/data/rcc_cost.csv (gen/data_lab21.py가 21장의 gen/_ch21_cost.csv에서 만든다).
끝의 대조 블록이 실습 결과를 gen/_ch21_nums.json(21장 본문의 숫자)과 맞춘다.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab21")


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
        h = h[:j] + _mark(h[j:k], dfmarks) + h[k:]
    return h


def save(name, c, marks=None, dfmarks=None):
    if os.environ.get("NOMARK"):
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        for w in c.warns:
            print("WARN:", w)
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
df = pd.read_csv(BASE + "rcc_cost.csv", storage_options=UA)
print(df.shape)
df.head()
''', title="패키지와 자료 불러오기")
save("lab21_load", c, marks={"(3000, 10)": 1}, dfmarks={"arm": 2, "cost_drug": 3, "1974.0": 4})

c = nb.cell('''
df["trt"] = (df["arm"] == "A").astype(int)   # 신약 A = 1, B = 0
print(df["arm"].value_counts())
print("결측값:", df.isna().sum().sum(), "개")

items = ["cost_drug", "cost_op", "cost_inp", "cost"]
zero = (df[items] == 0).groupby(df["arm"]).mean() * 100
zero.round(1)
''', title="치료군 변수 만들기, 결측값과 0원 확인")
save("lab21_trt", c, marks={"B    1797": 1, "A    1203": 2, "결측값: 0 개": 3}, dfmarks={"43.6": 4, "35.7": 5})

c = nb.cell('''
print(df.groupby("arm")["cost"].describe().round(1).T)

df.groupby("arm")[items].agg(["mean", "std"]).round(1).T
''', title="군별 요약통계")
save("lab21_summary", c, marks={"2668.9": 1, "1303.1": 2, "2462.3": 3, "21727.1": 4},
     dfmarks={"1558.6": 5, "1202.7": 6, "2272.8": 7})

c = nb.cell('''
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for ax, g in zip(axes, ["A", "B"]):
    y = df.loc[df["arm"] == g, "cost"]
    ax.hist(y, bins=np.arange(0, 10200, 200))
    ax.axvline(y.mean(), color="C1", label="mean")
    ax.axvline(y.median(), color="C1", ls="--", label="median")
    ax.set_title("Arm " + g)
    ax.set_xlabel("1-year cost (10,000 KRW)")
axes[0].set_ylabel("Number of patients")
axes[0].legend()
plt.tight_layout()
print("10,000만원을 넘어 그림 밖에 있는 환자:",
      (df["cost"] > 10000).groupby(df["arm"]).sum().to_dict())
''', title="비용 분포의 히스토그램")
save("lab21_hist", c, marks={"{'A': 5, 'B': 11}": 1})

c = nb.cell('''
a = df.loc[df["arm"] == "A", "cost"].to_numpy()   # A군 1,203명의 비용
b = df.loc[df["arm"] == "B", "cost"].to_numpy()   # B군 1,797명의 비용

print("B군 총액      :", round(b.sum()))
print("평균 x 인원   :", round(b.mean() * len(b)))
print("중앙값 x 인원 :", round(np.median(b) * len(b)))
print("평균보다 적게 쓴 환자(%):",
      round((b < b.mean()).mean() * 100, 1))
k = round(len(b) * 0.05)                          # 상위 5%는 90명
top = np.sort(b)[::-1][:k]
print("상위 5%가 쓴 몫(%)      :",
      round(top.sum() / b.sum() * 100, 1))
''', title="평균, 중앙값, 총액")
save("lab21_total", c, marks={"4084201": 1, "3300011": 2, "66.5": 3, "15.5": 4})

# ---------------------------------------------------------------- 나. 평균 비용의 비교와 부트스트랩
c = nb.cell('''
diff = a.mean() - b.mean()
tt = stats.ttest_ind(a, b, equal_var=False)       # Welch t 검정
ci = tt.confidence_interval()
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))

print("평균 차이(A - B):", round(diff, 1))
print("SE:", round(se, 1), " t =", round(tt.statistic, 2),
      " df =", round(tt.df, 1), " p =", tt.pvalue)
print("95% CI:", round(ci.low, 1), "~", round(ci.high, 1))
''', title="평균 차이와 Welch t 검정")
save("lab21_welch", c, marks={"396.1": 1, "SE: 51.0": 2, "t = 7.77": 3, "296.1 ~ 496.1": 4})

c = nb.cell('''
rng = np.random.default_rng(21)
idx = rng.integers(0, len(a), len(a))    # 0~1202 가운데 1,203개
print(idx[:8])
print("한 번 이상 뽑힌 환자:", len(np.unique(idx)), "명")
print("원래 A군의 평균     :", round(a.mean(), 1))
print("다시 뽑은 A군의 평균:", round(a[idx].mean(), 1))
''', title="A군을 한 번 다시 뽑아 보기")
save("lab21_once", c, marks={"[362 939 464 728 560 853 413 107]": 1, "750 명": 2, "다시 뽑은 A군의 평균: 2667.4": 3})

c = nb.cell('''
B = 9999                                     # 반복 횟수
rng = np.random.default_rng(21)              # 난수를 처음부터 다시
ia = rng.integers(0, len(a), (B, len(a)))    # A군 번호표
ib = rng.integers(0, len(b), (B, len(b)))    # B군 번호표
print(ia.shape, ib.shape)

boot = a[ia].mean(axis=1) - b[ib].mean(axis=1)
print(boot.shape, boot[:4].round(1))

lo, hi = np.percentile(boot, [2.5, 97.5])
print("95% CI (percentile):", round(lo, 1), "~", round(hi, 1))
print("bootstrap SE:", round(boot.std(ddof=1), 1))
''', title="군 안에서 9,999번 다시 뽑기")
save("lab21_boot", c, marks={"(9999, 1203)": 1, "(9999,)": 2, "296.1 ~ 496.3": 3, "bootstrap SE: 51.0": 4})

c = nb.cell('''
plt.figure(figsize=(6.4, 3.4))
plt.hist(boot, bins=50)
plt.axvline(diff, color="C1", label="observed difference")
plt.axvline(lo, color="k", ls="--", label="2.5% and 97.5%")
plt.axvline(hi, color="k", ls="--")
plt.xlabel("Bootstrap mean difference, A - B (10,000 KRW)")
plt.ylabel("Number of resamples")
plt.legend()
plt.tight_layout()
print("0 이하인 값:", (boot <= 0).sum(), "개")
''', title="부트스트랩 분포 그림")
save("lab21_bootplot", c)

c = nb.cell('''
def mean_diff(x, y, axis=-1):
    return x.mean(axis=axis) - y.mean(axis=axis)

for method in ["percentile", "BCa"]:
    res = stats.bootstrap((a, b), mean_diff, n_resamples=9999,
                          method=method,
                          random_state=np.random.default_rng(21))
    bci = res.confidence_interval
    print(method, round(bci.low, 1), round(bci.high, 1),
          round(res.standard_error, 1))
''', title="scipy 함수로 확인하고 BCa 구간 구하기")
save("lab21_scipy", c, marks={"percentile 296.1 496.3": 1, "BCa 298.0 498.9": 2})

c = nb.cell('''
med_diff = np.median(a) - np.median(b)
mw = stats.mannwhitneyu(a, b)
print("중앙값 A, B:", np.median(a), np.median(b),
      " 차이:", round(med_diff, 1))
print("Mann-Whitney U =", mw.statistic, " p =", mw.pvalue)
print("A군 환자의 비용이 더 클 확률:",
      round(mw.statistic / (len(a) * len(b)), 3))

print("총액 차이, 평균 차이 x 1,203명  :", round(diff * len(a)))
print("총액 차이, 중앙값 차이 x 1,203명:", round(med_diff * len(a)))
''', title="중앙값 차이와 Mann-Whitney 검정")
save("lab21_mw", c, marks={"차이: 625.9": 1, "p = 2.0": 2, "0.678": 3, "476477": 4, "752958": 5})

c = nb.cell('''
b2 = b.copy()
top = b2 >= np.percentile(b2, 95)       # B군에서 비용이 큰 5%
b2[top] = b2[top] * 3                   # 이 환자들의 비용만 3배로

print("바꾼 환자:", top.sum(), "명")
print("평균 차이(A - B):", round(a.mean() - b2.mean(), 1))
print("중앙값 차이      :", round(np.median(a) - np.median(b2), 1))
mw2 = stats.mannwhitneyu(a, b2)
print("A군이 더 클 확률 :",
      round(mw2.statistic / (len(a) * len(b2)), 3),
      " p =", mw2.pvalue)
''', title="고액 환자의 비용을 키워 보는 실험")
save("lab21_tail", c, marks={"평균 차이(A - B): -310.6": 1, "중앙값 차이      : 625.9": 2, "0.677": 3})

c = nb.cell('''
la, lb = np.log(a), np.log(b)
gm_a, gm_b = np.exp(la.mean()), np.exp(lb.mean())
print("로그 비용의 SD A, B:", round(la.std(ddof=1), 2),
      round(lb.std(ddof=1), 2))
print("기하평균 A, B:", round(gm_a, 1), round(gm_b, 1))

tl = stats.ttest_ind(la, lb, equal_var=False)
cl = tl.confidence_interval()
print("기하평균의 비:", round(gm_a / gm_b, 3),
      np.exp([cl.low, cl.high]).round(3))
print("산술평균의 비:", round(a.mean() / b.mean(), 3))

k_a = np.exp(la - la.mean()).mean()      # smearing 계수
k_b = np.exp(lb - lb.mean()).mean()
print("smearing 계수 A, B:", round(k_a, 3), round(k_b, 3))
print("기하평균 x 계수   :",
      round(gm_a * k_a, 1), round(gm_b * k_b, 1))
''', title="로그 변환, 기하평균, smearing 계수")
save("lab21_log", c, marks={"0.43 0.54": 1, "2441.0 1961.7": 2, "1.244": 3, "1.174": 4, "1.093 1.159": 5,
                            "2668.9 2272.8": 6})

# ---------------------------------------------------------------- 다. 비용의 회귀분석
c = nb.cell('''
covs = ["age", "male", "cci", "stage4"]
df.groupby("arm")[covs].agg(["mean", "std"]).round(2).T
''', title="두 군의 기저 특성")
save("lab21_base", c, dfmarks={"61.51": 1, "1.00": 2, "0.57": 3})

c = nb.cell('''
f = "cost ~ trt + age + male + cci + stage4"
ols = smf.ols(f, data=df).fit(cov_type="HC3")
print(ols.summary().tables[1])
''', title="원래 비용의 선형회귀 (강건 표준오차)")
save("lab21_ols", c, marks={"512.5732": 1, "412.693": 2, "200.1425": 3})

c = nb.cell('''
logm = smf.ols("np.log(cost) ~ trt + age + male + cci + stage4",
               data=df).fit()
print("exp(b):", np.exp(logm.params["trt"]).round(3),
      np.exp(logm.conf_int().loc["trt"]).round(3).tolist())

dA, dB = df.assign(trt=1), df.assign(trt=0)   # 모두 A, 모두 B
nA = np.exp(logm.predict(dA)).mean()
nB = np.exp(logm.predict(dB)).mean()
print("그대로 되돌림  :", round(nA, 1), round(nB, 1),
      round(nA - nB, 1))

s = np.exp(logm.resid).mean()                 # smearing 계수 하나
print("smearing 계수  :", round(s, 3))
print("계수 하나로 보정:", round(nA * s, 1), round(nB * s, 1),
      round((nA - nB) * s, 1))

sA = np.exp(logm.resid[df["trt"] == 1]).mean()
sB = np.exp(logm.resid[df["trt"] == 0]).mean()
print("군별 계수 A, B :", round(sA, 3), round(sB, 3))
print("군별 계수로 보정:", round(nA * sA, 1), round(nB * sB, 1),
      round(nA * sA - nB * sB, 1))
''', title="로그 변환 회귀와 되돌리기")
save("lab21_logols", c, marks={"1.295": 1, "570.7": 2, "1.127": 3, "643.2": 4, "1.091 1.151": 5, "507.3": 6})

# summary()의 Date·Time 줄은 실행한 시각이어서 다시 돌릴 때마다 달라진다.
# 이 셀을 돌리는 동안만 시각을 고정해 figs/lab21_glm.html이 매번 같게 나오게 한다(학생에게 보이는 코드는 그대로).
import time as _time  # noqa: E402
_real_localtime = _time.localtime
_time.localtime = lambda *a: _time.struct_time((2026, 10, 3, 12, 0, 0, 5, 276, 0))
try:
    c = nb.cell('''
    fam = sm.families.Gamma(link=sm.families.links.Log())
    glm = smf.glm(f, data=df, family=fam).fit()
    print(glm.summary())
    ''', title="감마 GLM (로그 연결)")
finally:
    _time.localtime = _real_localtime
save("lab21_glm", c, marks={"Gamma": 1, "Log": 2, "0.31751": 3, "0.2159": 4, "0.0832": 5})

c = nb.cell('''
ratio = pd.DataFrame({"cost_ratio": np.exp(glm.params),
                      "lower": np.exp(glm.conf_int()[0]),
                      "upper": np.exp(glm.conf_int()[1]),
                      "p": glm.pvalues})
ratio.round(3)
''', title="계수를 비용 비로 바꾸기")
save("lab21_ratio", c, dfmarks={"1.241": 1, "1.087": 2, "2037.101": 3})

c = nb.cell('''
pA = glm.predict(df.assign(trt=1))   # 모두 A를 썼다고 놓은 예측
pB = glm.predict(df.assign(trt=0))   # 모두 B를 썼다고 놓은 예측
print("모두 A, 모두 B, 차이:", round(pA.mean(), 1),
      round(pB.mean(), 1), round(pA.mean() - pB.mean(), 1))
print("비:", round(pA.mean() / pB.mean(), 3))

d_i = pA - pB                        # 환자 한 명 한 명의 예측 차이
print("환자별 차이의 범위:", round(d_i.min(), 1), "~",
      round(d_i.max(), 1))

two = pd.DataFrame({"trt": [1, 0, 1, 0], "age": [64, 64, 75, 75],
                    "male": 1, "cci": [1, 1, 4, 4], "stage4": 1})
print(glm.predict(two).round(1).tolist())
''', title="모두 A일 때와 모두 B일 때의 예측 평균")
save("lab21_std", c, marks={"2759.8 2223.8 536.0": 1, "비: 1.241": 2, "445.1 ~ 795.5": 3, "[2712.5, 2185.7, 3435.1, 2768.0]": 4})

c = nb.cell('''
rng = np.random.default_rng(2101)
n = len(df)
R = 1000                                  # 반복 횟수
bs = np.empty((R, 3))                     # 1,000행 x 3열의 빈 상자
for i in range(R):
    ix = rng.integers(0, n, n)            # 환자 3,000명을 다시 뽑음
    d = df.iloc[ix]
    m = smf.glm(f, data=d, family=fam).fit()
    mA = m.predict(d.assign(trt=1)).mean()
    mB = m.predict(d.assign(trt=0)).mean()
    bs[i] = [mA, mB, mA - mB]

ci3 = np.percentile(bs, [2.5, 97.5], axis=0).round(1)
print("모두 A :", ci3[:, 0])
print("모두 B :", ci3[:, 1])
print("차이   :", ci3[:, 2], " SE:", round(bs[:, 2].std(ddof=1), 1))
''', title="표준화한 차이의 부트스트랩 신뢰구간")
save("lab21_stdboot", c, marks={"[2684.6 2849.8]": 1, "[437.4 642.1]": 2, "SE: 52.6": 3})

c = nb.cell('''
chk = pd.DataFrame({"cost": df["cost"], "pred": glm.predict(df)})
chk["res2"] = (chk["cost"] - chk["pred"]) ** 2
chk["ln_pred"] = np.log(chk["pred"])
park = smf.glm("res2 ~ ln_pred", data=chk,
               family=fam).fit(cov_type="HC0")
print("기울기:", park.params["ln_pred"].round(2),
      park.conf_int().loc["ln_pred"].round(2).tolist())

chk["tenth"] = pd.qcut(chk["pred"], 10, labels=False) + 1
chk.groupby("tenth")[["cost", "pred"]].mean().round(0)
''', title="분포와 평균 구조 점검")
save("lab21_check", c, marks={"기울기: 2.02": 1}, dfmarks={"2878.0": 2, "3019.0": 3})

c = nb.cell('''
res_tab = pd.DataFrame(
    {"all_A": [a.mean(), ols.predict(dA).mean(), nA, nA * s,
               nA * sA, pA.mean()],
     "all_B": [b.mean(), ols.predict(dB).mean(), nB, nB * s,
               nB * sB, pB.mean()]},
    index=["unadjusted", "OLS", "log OLS, naive", "log OLS, smearing",
           "log OLS, smearing by arm", "gamma GLM"])
res_tab["diff"] = res_tab["all_A"] - res_tab["all_B"]
res_tab.round(0)
''', title="방법별 결과 모으기")
save("lab21_table", c, dfmarks={"396.0": 1, "513.0": 2, "571.0": 3, "643.0": 4, "507.0": 5, "536.0": 6})

# ---------------------------------------------------------------- 라. 과제
c = nb.cell('''
def boot_diff(col, B=9999, seed=21):
    x = df.loc[df["arm"] == "A", col].to_numpy()
    y = df.loc[df["arm"] == "B", col].to_numpy()
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), (B, len(x)))
    iy = rng.integers(0, len(y), (B, len(y)))
    bt = x[ix].mean(axis=1) - y[iy].mean(axis=1)
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return [x.mean(), y.mean(), x.mean() - y.mean(), lo, hi]

hw1 = pd.DataFrame([boot_diff(col) for col in items], index=items,
                   columns=["mean_A", "mean_B", "diff",
                            "lower", "upper"])
hw1.round(0)
''', title="과제 1 정답. 항목별 평균 차이와 부트스트랩 구간")
save("lab21_hw1", c, dfmarks={"740.0": 1, "-169.0": 2, "-175.0": 3, "396.0": 4})

c = nb.cell('''
rows = []
for B_ in [200, 1000, 9999]:
    lows, highs = [], []
    for seed in range(1, 11):                 # seed 1~10
        rng = np.random.default_rng(seed)
        ia_ = rng.integers(0, len(a), (B_, len(a)))
        ib_ = rng.integers(0, len(b), (B_, len(b)))
        bt = a[ia_].mean(axis=1) - b[ib_].mean(axis=1)
        l_, h_ = np.percentile(bt, [2.5, 97.5])
        lows.append(l_)
        highs.append(h_)
    rows.append([B_, min(lows), max(lows), min(highs), max(highs)])

hw2 = pd.DataFrame(rows, columns=["B", "lower_min", "lower_max",
                                  "upper_min", "upper_max"])
hw2["lower_range"] = hw2["lower_max"] - hw2["lower_min"]
hw2["upper_range"] = hw2["upper_max"] - hw2["upper_min"]
hw2.round(1)
''', title="과제 2 정답. 반복 횟수와 구간의 흔들림")
save("lab21_hw2", c)

c = nb.cell('''
df["anyinp"] = (df["cost_inp"] > 0).astype(int)  # 입원비가 생겼으면 1
rhs = "trt + age + male + cci + stage4"
part1 = smf.logit("anyinp ~ " + rhs, data=df).fit(disp=0)
pos = df[df["cost_inp"] > 0]                     # 입원비가 있는 환자만
part2 = smf.glm("cost_inp ~ " + rhs, data=pos, family=fam).fit()
print("입원비가 0보다 큰 환자:", len(pos), "명")
print("1부 오즈비:", np.exp(part1.params["trt"]).round(3),
      np.exp(part1.conf_int().loc["trt"]).round(2).tolist())
print("2부 비용 비:", np.exp(part2.params["trt"]).round(3),
      np.exp(part2.conf_int().loc["trt"]).round(2).tolist())

dA, dB = df.assign(trt=1), df.assign(trt=0)
eA = (part1.predict(dA) * part2.predict(dA)).mean()
eB = (part1.predict(dB) * part2.predict(dB)).mean()
print("모두 A, 모두 B, 차이:", round(eA, 1), round(eB, 1),
      round(eA - eB, 1))
''', title="과제 3 정답. 입원비의 이단계 모형")
save("lab21_hw3", c, marks={"1835": 1, "0.935": 2, "0.923": 3, "645.1 715.9 -70.7": 4})

# ---------------------------------------------------------------- 대조 블록: 21장 본문의 숫자와 맞추기
ns = nb.ns
N = json.load(open(os.path.join(HERE, "_ch21_nums.json"), encoding="utf-8"))
CHECKS = []


def chk(name, got, want, tol=1e-6):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    err = float(np.max(np.abs(got - want)))
    CHECKS.append((name, err))
    assert err < tol, f"{name}: lab {got} vs chapter {want} (err {err:g})"


G, C, LG, RG, TP = N["grp"], N["comp"], N["log"], N["reg"], N["reg"]["tp"]
a_, b_ = ns["a"], ns["b"]
# 가 절: 인원, 평균(SD), 중앙값(사분위수), 총액, 0원 비율
chk("n A, B", [len(a_), len(b_)], [G["A"]["n"], G["B"]["n"]])
chk("mean, SD", [a_.mean(), a_.std(ddof=1), b_.mean(), b_.std(ddof=1)],
    [C["cost"]["mA"], C["cost"]["sA"], C["cost"]["mB"], C["cost"]["sB"]])
chk("median, quartiles", np.percentile(a_, [25, 50, 75]).tolist() + np.percentile(b_, [25, 50, 75]).tolist(),
    [G["A"]["q1"], G["A"]["med"], G["A"]["q3"], G["B"]["q1"], G["B"]["med"], G["B"]["q3"]])
chk("B total, median x n", [b_.sum(), np.median(b_) * len(b_)], [G["B"]["total"], N["budget"]["tot_by_med"]], 1e-4)
chk("below mean, top 5% (B)", [(b_ < b_.mean()).mean(), ns["top"].sum()], [G["B"]["below_mean"], 90], 1e-9)
chk("zero inpatient % by arm", ns["zero"]["cost_inp"].tolist(), [C["inp"]["zeroA"] * 100, C["inp"]["zeroB"] * 100])
# 나 절: 평균 차이, Welch t, 부트스트랩(백분위수, BCa), Mann-Whitney, 로그 변환
chk("mean difference", ns["diff"], C["cost"]["diff"])
chk("Welch t, df, CI", [ns["tt"].statistic, ns["tt"].df, ns["ci"].low, ns["ci"].high],
    [C["cost"]["t"], C["cost"]["dfw"]] + C["cost"]["t_ci"])
chk("Welch p", ns["tt"].pvalue, C["cost"]["p"], 1e-20)
chk("SE (formula)", ns["se"], N["boot"]["se_formula"])
chk("bootstrap distribution (9,999 values)", ns["boot"], np.load(os.path.join(HERE, "_ch21_boot.npy")), 1e-9)
chk("bootstrap percentile CI", [ns["lo"], ns["hi"]], N["boot"]["pct"], 1e-9)
chk("bootstrap SE", ns["boot"].std(ddof=1), N["boot"]["se"], 1e-9)
chk("scipy BCa CI", [ns["bci"].low, ns["bci"].high], N["boot"]["bca"], 1e-9)
chk("median difference, P(A > B)", [ns["med_diff"], ns["mw"].statistic / (len(a_) * len(b_))],
    [LG["med_diff"], C["cost"]["auc"]])
chk("Mann-Whitney p", ns["mw"].pvalue, C["cost"]["mw_p"], 1e-65)
chk("budget difference", [ns["diff"] * len(a_), ns["med_diff"] * len(a_)], [N["budget"]["d_mean"], N["budget"]["d_med"]], 1e-4)
chk("geometric means, ratio, CI", [ns["gm_a"], ns["gm_b"], ns["gm_a"] / ns["gm_b"], np.exp(ns["cl"].low), np.exp(ns["cl"].high)],
    [LG["gmA"], LG["gmB"], LG["gm_ratio"]] + LG["gm_ci"])
chk("smearing factors by arm, log SD", [ns["k_a"], ns["k_b"], ns["la"].std(ddof=1), ns["lb"].std(ddof=1)],
    [LG["smearA"], LG["smearB"], LG["sdlA"], LG["sdlB"]])
# 다 절: 기저 특성, 선형회귀, 로그 변환 회귀, 감마 GLM, 표준화, 부트스트랩, 점검
df_ = ns["df"]
for v in ("age", "male", "cci", "stage4"):
    g = df_.groupby("arm")[v].agg(["mean", "std"])
    chk("baseline " + v, [g.loc["A", "mean"], g.loc["A", "std"], g.loc["B", "mean"], g.loc["B", "std"]],
        [N["base"][v][k] for k in ("mA", "sA", "mB", "sB")])
chk("OLS (HC3) difference, CI", [ns["ols"].params["trt"]] + ns["ols"].conf_int().loc["trt"].tolist(),
    [RG["ols"]["diff"]] + RG["ols"]["ci"])
chk("OLS standardized means", [ns["ols"].predict(ns["dA"]).mean(), ns["ols"].predict(ns["dB"]).mean()],
    [RG["ols"]["predA"], RG["ols"]["predB"]])
L_ = RG["logols"]
chk("log OLS exp(b), CI", [np.exp(ns["logm"].params["trt"])] + np.exp(ns["logm"].conf_int().loc["trt"]).tolist(),
    [L_["ratio"]] + L_["ratio_ci"])
chk("log OLS retransformation", [ns["nA"], ns["nB"], ns["s"], ns["nA"] * ns["s"], ns["nB"] * ns["s"],
                                 (ns["nA"] - ns["nB"]) * ns["s"], ns["sA"], ns["sB"],
                                 ns["nA"] * ns["sA"] - ns["nB"] * ns["sB"]],
    [L_["naiveA"], L_["naiveB"], L_["smear"], L_["smA"], L_["smB"], L_["smear_diff"], L_["smearA"], L_["smearB"],
     L_["grp_diff"]])
glm_ = ns["glm"]
chk("gamma GLM coefficients", [glm_.params[k] for k in ("Intercept", "trt", "age", "male", "cci", "stage4")],
    [RG["glm"]["coef"][k] for k in ("Intercept", "A", "age", "male", "cci", "stage4")], 1e-8)
chk("cost ratio, CI, scale", [np.exp(glm_.params["trt"])] + np.exp(glm_.conf_int().loc["trt"]).tolist() + [glm_.scale],
    [RG["glm"]["ratio"]] + RG["glm"]["ratio_ci"] + [RG["glm"]["scale"]], 1e-8)
chk("standardized means, difference", [ns["pA"].mean(), ns["pB"].mean(), ns["pA"].mean() - ns["pB"].mean()],
    [RG["glm"]["predA"], RG["glm"]["predB"], RG["glm"]["diff"]])
two_ = glm_.predict(ns["two"]).to_numpy()
chk("two example patients", [two_[0] - two_[1], two_[2] - two_[3]], [RG["glm"]["one"]["diff"], RG["glm"]["one2"]["diff"]])
bs_ = ns["bs"]
chk("bootstrap CI of adjusted difference", np.percentile(bs_[:, 2], [2.5, 97.5]), RG["glm"]["ci"], 1e-5)
chk("bootstrap CI of adjusted means", np.percentile(bs_[:, :2], [2.5, 97.5], axis=0).T.ravel(),
    RG["glm"]["predA_ci"] + RG["glm"]["predB_ci"], 1e-5)
chk("bootstrap SE of adjusted difference", bs_[:, 2].std(ddof=1), RG["glm"]["boot_se"], 1e-5)
chk("modified Park test", [ns["park"].params["ln_pred"]] + ns["park"].conf_int().loc["ln_pred"].tolist(),
    [RG["park"]["lam"]] + RG["park"]["ci"], 1e-6)
dec_ = ns["chk"].groupby("tenth")[["cost", "pred"]].mean()
chk("observed and predicted means by tenth", dec_["cost"].tolist() + dec_["pred"].tolist(),
    RG["check"]["obs"] + RG["check"]["glm"], 1e-6)
# 과제: 항목별 부트스트랩 구간, 이단계 모형
for col, key in (("cost_drug", "drug"), ("cost_op", "op"), ("cost_inp", "inp"), ("cost", "cost")):
    chk("item " + key + " diff, bootstrap CI", ns["hw1"].loc[col, ["diff", "lower", "upper"]].tolist(),
        [C[key]["diff"]] + C[key]["boot"], 1e-9)
chk("two-part model", [np.exp(ns["part1"].params["trt"]), np.exp(ns["part2"].params["trt"]), ns["eA"], ns["eB"],
                       ns["eA"] - ns["eB"], len(ns["pos"])],
    [TP["or_"], TP["ratio"], TP["predA"], TP["predB"], TP["diff"], TP["n_pos"]], 1e-6)
chk("two-part CIs", np.exp(ns["part1"].conf_int().loc["trt"]).tolist() + np.exp(ns["part2"].conf_int().loc["trt"]).tolist(),
    TP["or_ci"] + TP["ratio_ci"], 1e-6)
print(f"대조: {len(CHECKS)}개 항목 모두 일치 (가장 큰 차이 {max(e for _, e in CHECKS):.2e}, "
      f"{max(CHECKS, key=lambda t: t[1])[0]})")

# ---------------------------------------------------------------- 본문 문장에 쓴 그 밖의 숫자 확인
if os.environ.get("VERIFY") or os.environ.get("NOMARK"):
    import warnings
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy import stats
    fam_ = sm.families.Gamma(link=sm.families.links.Log())
    f_ = ns["f"]
    d1, d0 = df_.assign(trt=1), df_.assign(trt=0)
    poi = smf.glm(f_, df_, family=sm.families.Poisson()).fit(cov_type="HC0")
    print("VERIFY Poisson(HC0) standardized diff:", round(poi.predict(d1).mean() - poi.predict(d0).mean(), 1),
          "chapter", round(RG["poisson"]["diff"], 1))
    gi = smf.glm("cost ~ trt * (age + male + cci + stage4)", df_, family=fam_).fit()
    print("VERIFY interaction GLM standardized diff:", round(gi.predict(d1).mean() - gi.predict(d0).mean(), 1),
          "chapter", round(N["het"]["sample_inter"], 1))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        g0 = smf.glm(f_, df_, family=sm.families.Gamma()).fit()
        print("VERIFY Gamma() default link:", type(g0.family.link).__name__, "trt coef", g0.params["trt"],
              [str(w.message)[:80] for w in wl])
    print("VERIFY OLS plain SE", round(smf.ols(f_, df_).fit().bse["trt"], 1), "HC3", round(ns["ols"].bse["trt"], 1))
    print("VERIFY unique share:", round(len(np.unique(ns["idx"])) / len(a_), 3), " 1 - 1/e =", round(1 - np.exp(-1), 3))
    print("VERIFY ia[0][:8] == idx[:8]:", (ns["ia"][0] == ns["idx"]).all(), " MB:",
          round(ns["ia"].nbytes / 1e6), round(ns["ib"].nbytes / 1e6))
    rng = np.random.default_rng(21)
    d3 = np.empty(9999)
    for i in range(9999):
        d3[i] = a_[rng.integers(0, len(a_), len(a_))].mean() - b_[rng.integers(0, len(b_), len(b_))].mean()
    print("VERIFY interleaved loop CI:", np.percentile(d3, [2.5, 97.5]).round(1))
    for sd in (1, 2, 3):
        r_ = np.random.default_rng(sd)
        x_ = a_[r_.integers(0, len(a_), (9999, len(a_)))].mean(axis=1) - b_[r_.integers(0, len(b_), (9999, len(b_)))].mean(axis=1)
        print("VERIFY seed", sd, np.percentile(x_, [2.5, 97.5]).round(1))
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            m0 = smf.ols("np.log(cost_inp) ~ trt", df_).fit()
            print("VERIFY log(0) OLS params:", m0.params.tolist(), [w.category.__name__ + ": " + str(w.message)[:60] for w in wl])
        except Exception as e:  # noqa
            print("VERIFY log(0) OLS error:", type(e).__name__, str(e)[:200], [str(w.message)[:60] for w in wl])
    print("VERIFY B top 1% removed mean:", round(np.sort(b_)[::-1][18:].mean(), 1), " max A, B:", a_.max(), b_.max())
    print("VERIFY over 10000:", (a_ > 10000).sum(), (b_ > 10000).sum(), " bins max", df_["cost"].max())
    print("VERIFY tail experiment: B mean after", round(ns["b2"].mean(), 1), " cut-off", round(np.percentile(b_, 95), 1),
          " A above cut-off:", (a_ >= np.percentile(b_, 95)).sum())
    print("VERIFY ratio check 2760/2224:", round(ns["pA"].mean() / ns["pB"].mean(), 4), " exp(b)", round(float(np.exp(glm_.params["trt"])), 4))
    print("VERIFY GLM mean of fitted vs observed:", round(glm_.predict(df_).mean(), 1), round(df_["cost"].mean(), 1))
    print("VERIFY IQR A, B:", np.percentile(a_, [25, 75]), np.percentile(b_, [25, 75]))
    print("VERIFY cci ratio CI:", np.exp(glm_.conf_int().loc["cci"]).round(3).tolist())
    print("VERIFY crude ratio of means", round(a_.mean() / b_.mean(), 3), " share of A patients with cci>=2:",
          round((df_.loc[df_.arm == "A", "cci"] >= 2).mean(), 3), round((df_.loc[df_.arm == "B", "cci"] >= 2).mean(), 3))
    print("VERIFY hw2:\n", ns["hw2"].round(1).to_string())
    print("VERIFY two-part obs:", TP["obs_pA"], TP["obs_pB"], TP["obs_mA"], TP["obs_mB"], " tp p-values", TP["or_p"], TP["ratio_p"])
    h3 = ns["hw1"]
    print("VERIFY hw1 raw:\n", h3.round(1).to_string())

if os.environ.get("NOMARK"):
    sys.exit(0)

# ---------------------------------------------------------------- 노트북
HW = {
    23: ("## 과제 정답\n\n**과제 1.** 약제비(`cost_drug`), 외래 진료비(`cost_op`), 입원 진료비(`cost_inp`), 총비용(`cost`) 각각에 대해 "
         "평균 차이(A − B)와 부트스트랩 백분위수 95% 신뢰구간(9,999회, seed 21)을 구합니다. 먼저 스스로 풀어 본 뒤 아래 셀을 실행하세요."),
    24: ("**과제 2.** 반복 횟수를 200, 1,000, 9,999로 바꾸고 seed를 1부터 10까지 바꿔 가며 총비용 평균 차이의 백분위수 구간을 구해, "
         "구간의 하한과 상한이 seed에 따라 얼마나 달라지는지 비교합니다."),
    25: ("**과제 3.** 입원 진료비(`cost_inp`)는 38.8%가 0원입니다. 이단계 모형(1부 로지스틱 회귀, 2부 감마 GLM)으로 "
         "모두 A일 때와 모두 B일 때의 1인당 입원비와 그 차이를 구합니다."),
}
notes = {
    1: "## 가. 실습 데이터 준비\n\n환자 한 명이 한 줄인 1년 의료비 자료(가상)를 불러와 군별로 요약하고 분포를 봅니다.",
    6: "## 나. 평균 비용의 비교와 부트스트랩\n\n평균 차이의 신뢰구간을 Welch t 검정과 부트스트랩으로 구하고, 중앙값·Mann-Whitney 검정·로그 변환이 답하는 질문과 견줍니다.",
    14: "## 다. 비용의 회귀분석\n\n두 군의 환자 구성 차이를 선형회귀, 로그 변환 회귀, 감마 GLM으로 보정합니다. 셀 20은 모형을 1,000번 다시 적합하므로 1–2분 걸립니다.",
}
notes.update(HW)
path = nb.save_ipynb(
    "실습 21. 비용 자료 분석",
    intro=("사회약학 연구방법 노트의 '실습 21 비용 자료 분석'에 나오는 셀을 차례로 모은 노트북입니다. "
           "진행성 신세포암 1차 치료(신약 A 대 표준요법 B, 가상 자료) 환자 3,000명의 1년 의료비로 "
           "평균 비용의 차이와 부트스트랩 신뢰구간을 구하고, 감마 GLM으로 환자 구성의 차이를 보정합니다. "
           "셀을 위에서부터 차례로 실행하세요. 자료는 사이트에서 바로 읽으므로 인터넷 연결이 필요합니다. "
           "설명과 출력 읽는 법은 사이트의 실습 21 쪽에 있습니다."),
    notes=notes)
print("notebook:", path, len(nb.cells), "cells")
