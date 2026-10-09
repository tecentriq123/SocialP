"""실습 15 · 성향점수 분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab15.py && python3 gen/lab_lab15.py
      (NOMARK=1 python3 gen/lab_lab15.py 로 돌리면 표식 없이 모든 셀의 출력을 화면에 찍는다)

자료는 gen/data_lab15.py가 만든 pub/data/ps_cohort.csv(15장의 예제 코호트 그대로)이고, 끝의 대조 블록에서
실습 결과를 gen/nums_ch15.py의 계산값과 맞춘다.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

nb = Notebook("lab15")


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


def freeze_clock(h):
    """summary()가 찍는 실행 날짜와 시각(Date, Time 줄)을 고정값으로 바꾼다. 다시 돌려도 같은 파일이 나오게
    하려는 것이고 통계 출력은 건드리지 않는다(같은 길이의 글자로 바꾸므로 줄 맞춤도 그대로다)."""
    h = re.sub(r"(Date:\s+)\w{3}, \d{2} \w{3} \d{4}", r"\g<1>Sat, 03 Oct 2026", h)
    return re.sub(r"(Time:\s+)\d{2}:\d{2}:\d{2}", r"\g<1>16:34:00", h)


def save(name, c, marks=None, dfmarks=None):
    if os.environ.get("NOMARK"):
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        marks = dfmarks = None
    nb.save_fragment(name, freeze_clock(render(c, marks, dfmarks)))


PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

# ---------------------------------------------------------------- 가. 실습 데이터 준비
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab15_install", c)

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
df = pd.read_csv(BASE + "ps_cohort.csv", storage_options=UA)
print(df.shape)
df.head(3)
''', title="코호트 자료 불러오기")
save("lab15_load", c, marks={"(10000, 13)": 1}, dfmarks={"drugA": 2, "72.366034": 3, "349": 4})

c = nb.cell('''
covs = ["age", "female", "hf", "ckd", "prior_hosp",
        "n_drug", "tertiary"]
print(df["drugA"].value_counts())

t1 = df.groupby("drugA")[covs + ["hosp_1y"]].mean().T
t1.columns = ["drug B", "drug A"]
print("crude RR:",
      round(t1.loc["hosp_1y", "drug A"] / t1.loc["hosp_1y", "drug B"], 3))
t1.round(3)
''', title="보정 전 두 군의 기저특성과 입원 위험", max_rows=12)
save("lab15_baseline", c, marks={"5947": 1, "crude RR: 1.328": 4},
     dfmarks={"67.855": 2, "0.148": 3})

# ---------------------------------------------------------------- 나. 성향점수 추정과 겹침
c = nb.cell('''
f_ps = ("drugA ~ age + female + hf + ckd + prior_hosp"
        " + n_drug + tertiary")
ps_fit = smf.logit(f_ps, data=df).fit(disp=0)
print(ps_fit.summary())
''', title="성향점수 모형 (로지스틱 회귀)")
save("lab15_psmodel", c, marks={"drugA": 1, "0.06109": 2, "-1.779      0.075": 3, "0.7566": 4})

c = nb.cell('''
from sklearn.metrics import roc_auc_score

df["ps"] = ps_fit.predict(df)             # 환자별 성향점수
print(df.groupby("drugA")["ps"]
        .agg(["mean", "min", "max"]).round(3))
print("C statistic:",
      round(roc_auc_score(df["drugA"], df["ps"]), 3))

new = pd.DataFrame({
    "age": [78, 58], "female": [1, 0], "hf": [1, 0],
    "ckd": [1, 0], "prior_hosp": [1, 0], "n_drug": [9, 4],
    "tertiary": [1, 0]})
print("two patients:", ps_fit.predict(new).round(3).tolist())
''', title="환자별 성향점수와 C 통계량")
save("lab15_ps", c, marks={"0.372  0.130  0.896": 1, "0.454  0.149  0.932": 2,
                           "C statistic: 0.658": 3, "[0.881, 0.253]": 4})

c = nb.cell('''
a = df["drugA"] == 1                      # A 사용자면 True
bins = np.linspace(0, 1, 41)              # 0.025 간격의 막대
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(df.loc[~a, "ps"], bins=bins, density=True,
        alpha=0.5, label="Drug B")
ax.hist(df.loc[a, "ps"], bins=bins, density=True,
        alpha=0.5, label="Drug A")
ax.set_xlabel("Propensity score (probability of drug A)")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()

ps_lo = max(df.loc[a, "ps"].min(), df.loc[~a, "ps"].min())
ps_hi = min(df.loc[a, "ps"].max(), df.loc[~a, "ps"].max())
outside = (df["ps"] < ps_lo) | (df["ps"] > ps_hi)
print(f"common range {ps_lo:.3f}-{ps_hi:.3f},",
      "outside:", outside.sum())
''', title="두 군의 성향점수 분포와 겹침")
save("lab15_overlap", c, marks={"common range 0.149-0.896": 1, "outside: 13": 2})

# ---------------------------------------------------------------- 다. 매칭과 가중
c = nb.cell('''
df["logit"] = np.log(df["ps"] / (1 - df["ps"]))   # 로짓 성향점수
sd = df["logit"].std()
caliper = 0.2 * sd                        # 허용 한계
print("SD of logit PS:", round(sd, 3))
print("caliper       :", round(caliper, 3))
''', title="로짓 성향점수와 caliper")
save("lab15_caliper", c, marks={"0.616": 1, "0.123": 2})

c = nb.cell('''
def match_1to1(logit, treat, caliper, seed=7):
    """1:1 최근접 매칭 (비복원, caliper 적용)."""
    rng = np.random.default_rng(seed)
    a_rows = np.where(treat == 1)[0]      # A 사용자의 행 번호
    b_rows = np.where(treat == 0)[0]      # B 사용자의 행 번호
    b_logit = logit[b_rows].copy()        # 아직 남은 B의 로짓
    pairs = []
    for i in rng.permutation(a_rows):     # A를 무작위 순서로
        dist = np.abs(b_logit - logit[i])     # 모든 B와의 거리
        j = np.argmin(dist)                   # 가장 가까운 B
        if dist[j] <= caliper:                # 한계 안이면
            pairs.append((i, b_rows[j]))      # 짝으로 기록
            b_logit[j] = np.inf               # 쓴 B는 지움
    return pd.DataFrame(pairs, columns=["a_row", "b_row"])

pairs = match_1to1(df["logit"].to_numpy(),
                   df["drugA"].to_numpy(), caliper)
print(len(pairs), "pairs")
pairs.head(3)
''', title="1:1 최근접 매칭 함수")
save("lab15_match", c, marks={"3516 pairs": 1}, dfmarks={"8203": 2})

c = nb.cell('''
ma = df.iloc[pairs["a_row"]].copy()       # 짝을 찾은 A 사용자
mb = df.iloc[pairs["b_row"]].copy()       # 짝이 된 B 사용자
ma["pair"] = range(len(pairs))            # 짝 번호 0, 1, 2, ...
mb["pair"] = range(len(pairs))
m = pd.concat([ma, mb])                   # 매칭 코호트 (긴 형식)

gap = np.abs(ma["logit"].to_numpy() - mb["logit"].to_numpy())
print("matched A:", len(ma), "of", a.sum(),
      f"({len(ma) / a.sum():.1%})")
print("largest distance in a pair:", round(gap.max(), 3))
show = ["pair", "id", "drugA", "age", "hf", "ps"]
m.sort_values(["pair", "drugA"])[show].head(4).round(3)
''', title="매칭 코호트 만들기")
save("lab15_matched", c, marks={"3516 of 4053": 1, "(86.8%)": 2, "0.122": 3})

c = nb.cell('''
df["matched"] = df["id"].isin(m["id"]).astype(int)
um = df[a].groupby("matched")[["ps", "age", "hf",
                               "hosp_1y"]].mean()
um.insert(0, "n", df[a]["matched"].value_counts())
print("unused B users:", ((~a) & (df["matched"] == 0)).sum())
um.round(3)
''', title="짝을 못 찾은 A 사용자는 누구인가")
save("lab15_unmatched", c, marks={"unused B users: 2431": 4},
     dfmarks={"537": 1, "0.687": 2, "0.527": 3})

c = nb.cell('''
pA = df["drugA"].mean()                   # A를 받은 비율 0.4053
ps = df["ps"]
df["w"] = np.where(a, 1 / ps, 1 / (1 - ps))            # IPTW
df["sw"] = np.where(a, pA / ps, (1 - pA) / (1 - ps))   # 안정화

print(df[["w", "sw"]].agg(["mean", "min", "max"]).round(3))
print(df.groupby("drugA")[["w", "sw"]].sum().round(0))

wb = np.linspace(0, 6, 61)                # 0.1 간격의 막대
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.hist(df.loc[~a, "sw"], bins=wb, alpha=0.5, label="Drug B")
ax.hist(df.loc[a, "sw"], bins=wb, alpha=0.5, label="Drug A")
ax.set_xlabel("Stabilized weight")
ax.set_ylabel("Number of patients")
ax.legend()
plt.tight_layout()
''', title="역확률 가중치와 안정화 가중치")
save("lab15_weights", c, marks={"1.000": 1, "1.072  0.435": 2, "9.583  5.699": 3,
                                "10040.0  4069.0": 4})

# ---------------------------------------------------------------- 라. 균형 진단과 효과 추정
c = nb.cell('''
def smd(x, treat, w=None):
    """표준화 평균차. w를 주면 가중 평균과 가중 분산으로."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(treat) == 1
    w = np.ones(len(x)) if w is None else np.asarray(w)
    m1 = np.average(x[t], weights=w[t])           # A군 평균
    m0 = np.average(x[~t], weights=w[~t])         # B군 평균
    v1 = np.average((x[t] - m1) ** 2, weights=w[t])     # 분산
    v0 = np.average((x[~t] - m0) ** 2, weights=w[~t])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2)

print("hf, before:", round(smd(df["hf"], df["drugA"]), 3))
''', title="표준화 평균차(SMD) 함수")
save("lab15_smd", c, marks={"0.292": 1})

c = nb.cell('''
bal = pd.DataFrame({
    "before": [smd(df[v], df["drugA"]) for v in covs],
    "matched": [smd(m[v], m["drugA"]) for v in covs],
    "IPTW": [smd(df[v], df["drugA"], df["w"]) for v in covs],
}, index=covs).abs()
print(bal.max().round(3))
bal.round(3)
''', title="보정 전, 매칭 뒤, 가중 뒤의 균형 표")
save("lab15_balance", c, marks={"0.346": 1, "0.018": 2, "0.006": 3},
     dfmarks={"0.042": 4})

c = nb.cell('''
y = np.arange(len(covs))
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for col, mk in [("before", "o"), ("matched", "s"), ("IPTW", "^")]:
    ax.scatter(bal[col], y, marker=mk, label=col)
ax.axvline(0.1, ls="--", color="gray")    # 기준선 0.1
ax.set_yticks(y)
ax.set_yticklabels(covs)
ax.invert_yaxis()                         # 첫 변수를 맨 위에
ax.set_xlabel("Absolute standardized mean difference")
ax.legend()
plt.tight_layout()
''', title="Love plot")
save("lab15_love", c)

c = nb.cell('''
def risks(d, w=None):
    """A군 위험, B군 위험, 상대위험도, 위험차 (w는 가중치 열 이름)."""
    t = d["drugA"] == 1
    w = np.ones(len(d)) if w is None else d[w].to_numpy()
    rA = np.average(d.loc[t, "hosp_1y"], weights=w[t])
    rB = np.average(d.loc[~t, "hosp_1y"], weights=w[~t])
    return [rA, rB, rA / rB, rA - rB]

est = pd.DataFrame(
    {"crude": risks(df), "matched": risks(m), "IPTW": risks(df, "w")},
    index=["risk A", "risk B", "RR", "RD"]).T
est.round(4)
''', title="세 분석의 입원 위험과 상대위험도")
save("lab15_risks", c, dfmarks={"1.3279": 1, "0.9076": 2, "-0.0111": 3, "0.9535": 4})

c = nb.cell('''
def gee_fit(formula, data, groups, weights=None, family=None):
    """독립 작업상관 GEE. 기본은 포아송(상대위험도)."""
    if family is None:
        family = sm.families.Poisson()
    return smf.gee(formula, groups=groups, data=data,
                   family=family, weights=weights,
                   cov_struct=sm.cov_struct.Independence()).fit()

gm = gee_fit("hosp_1y ~ drugA", m, "pair")
print(gm.summary().tables[1])
print("clusters:", gm.model.num_group,
      " cov type:", gm.cov_type)
ci = gm.conf_int().loc["drugA"]
print("RR:", round(np.exp(gm.params["drugA"]), 3),
      " 95% CI:", np.exp(ci).round(3).tolist())
''', title="매칭 코호트의 상대위험도와 강건 신뢰구간")
save("lab15_gee", c, marks={"-0.0970": 1, "0.066": 2, "clusters: 3516": 3,
                            "RR: 0.908": 4, "[0.797, 1.033]": 5})

c = nb.cell('''
def rr_ci(g):
    """GEE 결과에서 RR과 95% 신뢰구간을 꺼냅니다."""
    lo, hi = g.conf_int().loc["drugA"]
    return np.exp([g.params["drugA"], lo, hi])

g0 = gee_fit("hosp_1y ~ drugA", df, "id")             # 보정 전
gw = gee_fit("hosp_1y ~ drugA", df, "id", df["sw"])   # IPTW
rr = pd.DataFrame([rr_ci(g0), rr_ci(gm), rr_ci(gw)],
                  index=["crude", "matched", "IPTW"],
                  columns=["RR", "lower", "upper"])
print(rr.round(3))

gd = gee_fit("hosp_1y ~ drugA", m, "pair",
             family=sm.families.Gaussian())           # 위험차
lo, hi = 100 * gd.conf_int().loc["drugA"]
print("matched RD (%p):", round(100 * gd.params["drugA"], 1),
      f"({lo:.1f}, {hi:.1f})")
''', title="세 분석의 상대위험도와 매칭 코호트의 위험차")
save("lab15_rr", c, marks={"1.328  1.191  1.481": 1, "0.908  0.797  1.033": 2,
                           "0.953  0.848  1.072": 3, "-1.1 (-2.6, 0.4)": 4})

c = nb.cell('''
from lifelines import CoxPHFitter

cols = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]
c1 = CoxPHFitter().fit(df, "hosp_day", "hosp_1y", formula="drugA")
c2 = CoxPHFitter().fit(m, "hosp_day", "hosp_1y", formula="drugA",
                       cluster_col="pair")
c3 = CoxPHFitter().fit(df, "hosp_day", "hosp_1y", formula="drugA",
                       weights_col="sw", robust=True)
hr = pd.concat([c.summary[cols] for c in (c1, c2, c3)])
hr.index = ["crude", "matched", "IPTW"]
hr.columns = ["HR", "lower", "upper"]
hr.round(3)
''', title="입원까지의 시간으로 본 위험비 (Cox 모형)")
save("lab15_cox", c, dfmarks={"1.354": 1, "0.903": 2, "0.952": 3})

c = nb.cell('''
fr = [smd(df["frail"], df["drugA"]), smd(m["frail"], m["drugA"]),
      smd(df["frail"], df["drugA"], df["w"])]
print("frailty SMD (before, matched, IPTW):", np.round(fr, 2))
print(m.groupby("drugA")[["frail", "injury_ed"]].mean().round(3))

nc = gee_fit("injury_ed ~ drugA", m, "pair")     # 음성 대조 결과
print("negative control RR:", rr_ci(nc).round(2))
''', title="측정하지 않은 변수와 음성 대조 결과")
save("lab15_frail", c, marks={"[0.4  0.34 0.33]": 1, "0.249": 2,
                              "negative control RR: [1.29 1.09 1.53]": 3})

# ---------------------------------------------------------------- 과제
c = nb.cell('''
def evalue(rr):
    """E-value. RR이 1보다 작으면 역수를 먼저 취합니다."""
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))

r = rr.loc["matched"]                     # RR, lower, upper
print("point estimate:", round(evalue(r["RR"]), 2))
print("rounded 0.91  :", round(evalue(0.91), 2))
covers_1 = r["lower"] <= 1 <= r["upper"]
print("CI limit      :",
      1.0 if covers_1 else round(evalue(r["upper"]), 2))
c0 = rr.loc["crude"]
print("crude         :", round(evalue(c0["RR"]), 2),
      round(evalue(c0["lower"]), 2))
''', title="과제 1 정답. E-value")
save("lab15_hw1", c, marks={"point estimate: 1.44": 1, "rounded 0.91  : 1.43": 2,
                            "CI limit      : 1.0": 3, "crude         : 1.99 1.67": 4})

c = nb.cell('''
lg = df["logit"].to_numpy()
tr = df["drugA"].to_numpy()
rows = []
for k in [0.05, 0.1, 0.2, 0.5, 1.0, np.inf]:
    p = match_1to1(lg, tr, k * sd)
    mk = pd.concat([df.iloc[p["a_row"]], df.iloc[p["b_row"]]])
    worst = max(abs(smd(mk[v], mk["drugA"])) for v in covs)
    rows.append([k, len(p), round(len(p) / a.sum(), 3),
                 round(worst, 3), round(risks(mk)[2], 3)])
pd.DataFrame(rows, columns=["caliper (SD)", "pairs",
                            "share of A", "max SMD", "RR"])
''', title="과제 2 정답. caliper 폭 바꾸기")
save("lab15_hw2", c)

c = nb.cell('''
f2 = "drugA ~ age + female + ckd + prior_hosp + n_drug + tertiary"
ps2 = smf.logit(f2, data=df).fit(disp=0).predict(df)
df["sw2"] = np.where(a, pA / ps2, (1 - pA) / (1 - ps2))

for name, w in [("full model", "sw"), ("without hf", "sw2")]:
    print(f"hf SMD, {name}: {smd(df['hf'], df['drugA'], df[w]):.3f}")
g2 = gee_fit("hosp_1y ~ drugA", df, "id", df["sw2"])
print("IPTW RR, without hf:", rr_ci(g2).round(3))
print("IPTW RR, full model:", rr.loc["IPTW"].to_numpy().round(3))
''', title="과제 3 정답. 심부전을 뺀 성향점수 모형")
save("lab15_hw3", c)

print("lab15: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- notebook
nb.save_ipynb(
    "실습 15. 성향점수 분석",
    intro=("사회약학 연구방법 노트의 '실습 15 성향점수 분석'을 따라 하는 노트북입니다. 15장의 예제 코호트"
           "(가상 청구자료의 신규 사용자 10,000명)로 성향점수를 추정하고, 1:1 매칭과 역확률 가중을 한 뒤 "
           "균형을 진단하고 1년 입원의 상대위험도를 추정합니다. 셀을 위에서부터 차례로 실행하세요. "
           "자료는 사이트가 만든 가상 자료이며 실제 환자 자료가 아닙니다."),
    notes={
        1: "## 가. 실습 데이터 준비\n\nColab에서는 런타임이 새로 배정될 때마다 lifelines를 설치합니다(셀 18에서 씁니다).",
        4: "## 나. 성향점수 추정과 겹침\n\n결과변수 자리에 치료(drugA)를 놓은 로지스틱 회귀입니다.",
        7: "## 다. 매칭과 가중\n\n로짓 성향점수로 짝을 짓고, 같은 성향점수로 가중치를 만듭니다.",
        12: "## 라. 균형 진단과 효과 추정\n\n균형(SMD)을 먼저 확인한 다음 결과를 비교합니다.",
        20: ("## 과제 정답\n\n**과제 1.** 매칭 코호트의 상대위험도와 그 신뢰구간에 대한 E-value를 구합니다. "
             "보정 전 상대위험도의 E-value와 비교합니다."),
        21: ("**과제 2.** caliper를 로짓 성향점수 표준편차의 0.05, 0.1, 0.2, 0.5, 1.0배와 caliper 없음으로 바꿔 "
             "매칭 쌍 수, 가장 큰 SMD, 상대위험도를 표로 만듭니다."),
        22: ("**과제 3.** 성향점수 모형에서 심부전(hf)을 빼고 안정화 가중치를 다시 만들어, 가중 뒤 심부전의 "
             "SMD와 상대위험도가 어떻게 달라지는지 봅니다."),
    })

# ---------------------------------------------------------------- 대조 블록: 본문(gen/nums_ch15.py)의 숫자와 맞추기
if __name__ == "__main__":
    import warnings
    import nums_ch15 as N15

    ns = nb.ns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = N15.ps_sim()
    df_, m_, pairs_, rr_, hr_, bal_, est_ = (ns[k] for k in ("df", "m", "pairs", "rr", "hr", "bal", "est"))
    n_ok = 0

    def same(label, a, b, tol=1e-9):
        global n_ok
        a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
        assert a.shape == b.shape and np.all(np.abs(a - b) < tol), f"{label}: {a} != {b}"
        n_ok += 1

    flat = lambda t: [t[0], t[1][0], t[1][1]]                       # (est, (lo, hi), ...) -> [est, lo, hi]
    same("N, nA, nB", [len(df_), df_["drugA"].sum(), (df_["drugA"] == 0).sum()], [S["N"], S["nA"], S["nB"]])
    same("events A, B", df_.groupby("drugA")["hosp_1y"].sum().loc[[1, 0]], [S["eA"], S["eB"]])
    same("PS coefficients", ns["ps_fit"].params.to_numpy(), S["res"].params.to_numpy(), 1e-8)
    same("PS", df_["ps"], S["ps"], 1e-12)
    same("PS range", df_.groupby("drugA")["ps"].agg(["min", "max"]).loc[[1, 0]].to_numpy(),
         [*S["ps_rng"]["A"], *S["ps_rng"]["B"]], 1e-12)
    same("C statistic", ns["roc_auc_score"](df_["drugA"], df_["ps"]), S["c_ps"], 1e-12)
    same("outside common support", [ns["ps_lo"], ns["ps_hi"], ns["outside"].sum()],
         [S["overlap"]["lo"], S["overlap"]["hi"], S["overlap"]["n_out"]], 1e-12)
    same("caliper", ns["caliper"], S["cal"], 1e-12)
    same("number of pairs", len(pairs_), S["npairs"])
    same("matched pairs (same rows, same order)", pairs_.to_numpy(), np.column_stack([S["ti"], S["ci"]]))
    u = S["unmatched"]
    um_ = ns["um"]
    same("unmatched A users", um_.loc[0, ["n", "ps", "age", "hf", "hosp_1y"]],
         [u["n"], u["ps_un"], u["age_un"], u["hf_un"], u["risk_un"]], 1e-9)
    same("weights", [df_["w"].min(), df_["w"].max(), df_["sw"].min(), df_["sw"].max(), df_["sw"].mean()],
         [*S["wsum"]["ate_rng"], *S["wsum"]["st_rng"], S["wsum"]["st_mean"]], 1e-9)
    same("weighted n", df_.groupby("drugA")["w"].sum().loc[[1, 0]], [S["wsum"]["sumA"], S["wsum"]["sumB"]], 1e-6)
    tab = {r["key"]: r for r in S["tab"]}
    keymap = dict(prior_hosp="prior", n_drug="ndrug", tertiary="tert")
    for col, k in (("before", "smd_pre"), ("matched", "smd_m"), ("IPTW", "smd_w")):
        same("SMD " + col, bal_[col], [abs(tab[keymap.get(v, v)][k]) for v in ns["covs"]], 1e-9)
    same("risks", est_[["risk A", "risk B"]].to_numpy(),
         [[S["rA"], S["rB"]], [S["rA_m"], S["rB_m"]], [S["rA_w"], S["rB_w"]]], 1e-9)
    same("RR crude", rr_.loc["crude"], flat(S["crude"]), 1e-7)
    same("RR matched", rr_.loc["matched"], flat(S["matched"]), 1e-7)
    same("RR IPTW", rr_.loc["IPTW"], flat(S["iptw"]), 1e-7)
    gd_ = ns["gd"]
    same("RD matched", [gd_.params["drugA"], *gd_.conf_int().loc["drugA"]], flat(S["matched_rd"]), 1e-7)
    same("HR crude", hr_.loc["crude"], flat(S["hr_crude"]), 1e-6)
    same("HR matched", hr_.loc["matched"], flat(S["hr_match"]), 1e-6)
    same("HR IPTW", hr_.loc["IPTW"], flat(S["hr_iptw"]), 1e-6)
    same("frailty SMD", ns["fr"], [tab["frail"][k] for k in ("smd_pre", "smd_m", "smd_w")], 1e-9)
    nc_ = ns["nc"]
    same("negative control RR", np.exp([nc_.params["drugA"], *nc_.conf_int().loc["drugA"]]), flat(S["nc"]["matched"]), 1e-7)
    same("E-value", [ns["evalue"](rr_.loc["matched", "RR"]), ns["evalue"](0.91), ns["evalue"](rr_.loc["crude", "RR"]),
                     ns["evalue"](rr_.loc["crude", "lower"])],
         [S["ev"]["matched"], N15.evalue(0.91), S["ev"]["crude"], S["ev"]["crude_ci"]], 1e-7)
    # the two-decimal strings quoted in content/ch15.html
    q = lambda r: "%.2f (%.2f-%.2f)" % tuple(r)
    quoted = {"crude": "1.33 (1.19-1.48)", "matched": "0.91 (0.80-1.03)", "IPTW": "0.95 (0.85-1.07)"}
    quoted_hr = {"crude": "1.35 (1.21-1.52)", "matched": "0.90 (0.79-1.04)", "IPTW": "0.95 (0.84-1.08)"}
    for k in quoted:
        assert q(rr_.loc[k]) == quoted[k], (k, q(rr_.loc[k]))
        assert q(hr_.loc[k]) == quoted_hr[k], (k, q(hr_.loc[k]))
        n_ok += 2
    print(f"VERIFY: {n_ok} checks against gen/nums_ch15.py and the chapter text passed")

    # ---- numbers quoted in the text of content/lab15.html (not shown as cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm_, smf_, gee_fit, smd_, match_ = ns["sm"], ns["smf"], ns["gee_fit"], ns["smd"], ns["match_1to1"]
        lo_, hi_ = np.quantile(df_["sw"], [0.01, 0.99])
        gt = gee_fit("hosp_1y ~ drugA", df_.assign(sw_t=df_["sw"].clip(lo_, hi_)), "id", df_["sw"].clip(lo_, hi_))
        print("VERIFY truncated sw at", round(lo_, 3), round(hi_, 3), "RR",
              np.exp([gt.params["drugA"], *gt.conf_int().loc["drugA"]]).round(3), "ch15 nums", np.round(flat(S["iptw_tr"]), 3))
        print("VERIFY sklearn max |diff| no penalty %.4f, default %.4f" % (S["sk_maxdiff"], S["sk_def_maxdiff"]))
        for seed in (1, 2, 3):
            p_ = match_(df_["logit"].to_numpy(), df_["drugA"].to_numpy(), ns["caliper"], seed=seed)
            mm = pd_ = None
            import pandas as pd_
            mm = pd_.concat([df_.iloc[p_["a_row"]], df_.iloc[p_["b_row"]]])
            print("VERIFY seed", seed, "pairs", len(p_), "RR", round(ns["risks"](mm)[2], 3),
                  "max SMD", round(max(abs(smd_(mm[v], mm["drugA"])) for v in ns["covs"]), 3))
        g_naive = smf_.glm("hosp_1y ~ drugA", data=df_, family=sm_.families.Poisson(), freq_weights=df_["w"]).fit()
        print("VERIFY naive CI (weights as counts):", np.exp(g_naive.conf_int().loc["drugA"]).round(3).tolist(),
              "ch15", np.round(S["iptw_naive"][1], 3))
        g_unp = gee_fit("hosp_1y ~ drugA", m_.assign(one=range(len(m_))), "one")
        print("VERIFY matched, pairing ignored:", np.exp(g_unp.conf_int().loc["drugA"]).round(4).tolist(),
              " paired:", rr_.loc["matched"].round(4).tolist())
        print("VERIFY matched Table 1 (A, B):")
        print(m_.groupby("drugA")[ns["covs"]].agg(["mean", "std"]).T.round(3).to_string())
        print(m_.groupby("drugA")[["female", "hf", "ckd", "prior_hosp", "tertiary", "hosp_1y", "injury_ed", "frail"]].sum().T.to_string())
        print("VERIFY before: counts")
        print(df_.groupby("drugA")[["female", "hf", "ckd", "prior_hosp", "tertiary", "hosp_1y"]].sum().T.to_string())
        print(df_.groupby("drugA")[["age", "n_drug"]].agg(["mean", "std"]).round(2).T.to_string())
        print("VERIFY sw by group:", df_.groupby("drugA")["sw"].agg(["min", "max", "mean"]).round(3).to_dict())
        print("VERIFY w>10:", int((df_["w"] > 10).sum()), " max w row:", df_.loc[df_["w"].idxmax(), ["drugA", "ps", "w"]].round(3).to_dict())
        print("VERIFY weighted risks:", est_.round(4).to_dict())
        print("VERIFY p values: matched", round(ns["gm"].pvalues["drugA"], 3), "IPTW", round(ns["gw"].pvalues["drugA"], 3))
        print("VERIFY PS of unmatched A: min", round(df_.loc[(df_["drugA"] == 1) & (df_["matched"] == 0), "ps"].min(), 3))
        gw1 = gee_fit("hosp_1y ~ drugA", df_, "id", df_["w"])
        print("VERIFY IPTW with w instead of sw: max |diff| in RR and CI",
              float(np.abs(ns["rr_ci"](gw1) - rr_.loc["IPTW"].to_numpy()).max()))
        from sklearn.linear_model import LogisticRegression
        for name, mdl in (("default (L2)", LogisticRegression(max_iter=5000)),
                          ("C=np.inf", LogisticRegression(C=np.inf, max_iter=5000))):
            ps_sk = mdl.fit(df_[ns["covs"]], df_["drugA"]).predict_proba(df_[ns["covs"]])[:, 1]
            lg_sk = np.log(ps_sk / (1 - ps_sk))
            p_ = match_(lg_sk, df_["drugA"].to_numpy(), 0.2 * lg_sk.std(ddof=1))
            y_ = df_["hosp_1y"].to_numpy()
            print(f"VERIFY sklearn {name}: max |PS diff| {np.abs(ps_sk - df_['ps']).max():.4f}, pairs {len(p_)},"
                  f" matched RR {y_[p_['a_row']].mean() / y_[p_['b_row']].mean():.3f}")
        print("VERIFY hw3: hf SMD before", round(smd_(df_["hf"], df_["drugA"]), 3))
        print("VERIFY discordant pairs:", S["disc"])
