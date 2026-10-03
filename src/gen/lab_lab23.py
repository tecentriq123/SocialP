"""실습 23 · 결정분석 모형 만들기 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab23.py && python3 gen/lab_lab23.py
      (NOMARK=1 을 앞에 붙이면 표식 없이 셀 출력만 화면에 찍는다)

자료: pub/data/rcc_trial.csv (gen/data_lab23.py가 23장의 gen/_ch23_trial.csv에서 만든다).
학생 코드는 gen/lib_p4.py를 쓰지 않는다(입력값 사전과 짧은 함수로 다시 짠다).
끝의 대조 블록이 실습 결과를 lib_p4.run()(20–25장 공통 예시의 기준 분석)과 gen/_ch23_nums.json(23장 라 절)의 값과 맞춘다.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab23")


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


PIP_OUT = """Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl (...)
...
Successfully installed ... lifelines-0.30.3"""

# ---------------------------------------------------------------- 가. 생존곡선의 적합과 외삽
c = nb.cell('''
!pip install lifelines
''', title="lifelines 설치", shell_output=PIP_OUT)
save("lab23_install", c)

c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
trial = pd.read_csv(BASE + "rcc_trial.csv")
print(trial.shape)
print(trial["event"].value_counts().to_dict())
trial.head()
''', title="패키지와 시험 자료 불러오기")
save("lab23_load", c, marks={"(300, 3)": 1, "{0: 155, 1: 145}": 2}, dfmarks={"24.903896": 3, "11.711834": 4})

c = nb.cell('''
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time

T, E = trial["months"], trial["event"]
kmf = KaplanMeierFitter().fit(T, E, label="Kaplan-Meier")
print("12, 24개월 생존율:", kmf.predict([12, 24]).round(3).tolist())
print("마지막 관찰 시점(개월):", round(T.max(), 1),
      " 그때의 생존율:", round(kmf.survival_function_.iloc[-1, 0], 3))
print("중앙생존기간:", kmf.median_survival_time_)
print("30개월까지의 곡선 아래 면적(개월):",
      round(restricted_mean_survival_time(kmf, t=30), 1))
''', title="Kaplan-Meier 추정")
save("lab23_km", c, marks={"[0.793, 0.557]": 1, "0.503": 2, "inf": 3, "21.9": 4})

c = nb.cell('''
from lifelines import (ExponentialFitter, WeibullFitter,
                       LogLogisticFitter, LogNormalFitter,
                       GeneralizedGammaFitter)

grid = np.arange(0, 241)               # 0, 1, ..., 240개월
models = {"Exponential": ExponentialFitter(),
          "Weibull": WeibullFitter(),
          "Log-logistic": LogLogisticFitter(),
          "Log-normal": LogNormalFitter(),
          "Gen. gamma": GeneralizedGammaFitter()}
rows, S = [], {}
for name, f in models.items():
    f.fit(T, E)                        # 최대우도추정
    s = f.survival_function_at_times(grid).to_numpy()
    S[name] = s                        # 20년까지 늘린 곡선
    mean_y = ((s[:-1] + s[1:]) / 2).sum() / 12
    rows.append([name, len(f.params_), f.AIC_, f.BIC_,
                 s[24], s[60], s[120], s[240], mean_y])
cols = ["model", "k", "AIC", "BIC", "S(2y)", "S(5y)",
        "S(10y)", "S(20y)", "mean_y"]
fit_tab = pd.DataFrame(rows, columns=cols).set_index("model")
fit_tab.round(3).round({"AIC": 1, "BIC": 1})
''', title="모수 모형 다섯 개의 적합과 외삽")
save("lab23_fit", c, dfmarks={"1375.9": 1, "1380.6": 2, "0.564": 3, "0.058": 4, "0.177": 5, "0.081": 6,
                              "2.642": 7, "5.187": 8})

c = nb.cell('''
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
for ax, xmax in zip(axes, [30, 240]):
    ax.step(kmf.timeline, kmf.survival_function_.iloc[:, 0],
            where="post", color="black", lw=2, label="Kaplan-Meier")
    for name, s in S.items():
        ax.plot(grid, s, label=name)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Months since randomization")
axes[0].set_ylim(0.4, 1)
axes[0].set_ylabel("Overall survival")
axes[0].set_title("Observed period (0-30 months)")
axes[1].set_title("Extrapolated to 20 years")
axes[1].axvline(30, color="gray", ls=":")
axes[1].legend(fontsize=8)
plt.tight_layout()
''', title="Kaplan-Meier 곡선과 외삽 곡선")
save("lab23_fitplot", c)

c = nb.cell('''
w = models["Weibull"]
print("lifelines: lambda_ =", round(w.lambda_, 3),
      " rho_ =", round(w.rho_, 4))

gam_fit = w.rho_                       # 모양 모수는 그대로
lam_fit = w.lambda_ ** (-w.rho_)       # lambda_는 바꿔야 한다
print("이 실습의 식: lam =", round(lam_fit, 5),
      " gam =", round(gam_fit, 4))

s_ours = np.exp(-lam_fit * grid ** gam_fit)
print("두 식으로 구한 생존율의 최대 차이:",
      np.abs(s_ours - S["Weibull"]).max())
print("중앙값(개월):", round((np.log(2) / lam_fit) ** (1 / gam_fit), 1))
print("관찰 기간(30개월)의 몫:",
      round(((s_ours[:30] + s_ours[1:31]) / 2).sum()
            / ((s_ours[:-1] + s_ours[1:]) / 2).sum(), 2))
''', title="lifelines의 와이블 모수를 이 실습의 식으로 바꾸기")
save("lab23_weib", c, marks={"39.059": 1, "1.1454": 2, "0.01502": 3, "두 식으로 구한 생존율의 최대 차이:": 4,
                             "중앙값(개월): 28.4": 5, "몫: 0.58": 6})

# ---------------------------------------------------------------- 나. 분할생존모형 만들기
c = nb.cell('''
def weib_lam(median, gam):
    """중앙값(개월)과 모양 모수 -> lam. S(중앙값) = 0.5가 되게"""
    return np.log(2) / median ** gam

p = {
    # 표준요법 B의 와이블 곡선: 중앙 PFS 10개월, 중앙 OS 28개월
    "pfs_gam": 0.95, "pfs_lam": weib_lam(10, 0.95),
    "os_gam": 1.15, "os_lam": weib_lam(28, 1.15),
    # 신약 A의 위험비 (B 대비)
    "hr_pfs": 0.65, "hr_os": 0.75,
    # 상태별 효용, 이상반응의 QALY 손실 (1회)
    "u_pf": 0.78, "u_pd": 0.62,
    "du_ae_A": 0.012, "du_ae_B": 0.008,
    # 비용 (만원): 월 약값, 상태별 월 비용, 임종기와 이상반응 (1회)
    "c_drug_A": 190.0, "c_drug_B": 120.0,
    "c_pf": 40.0, "c_pd": 250.0, "c_death": 800.0,
    "c_ae_A": 120.0, "c_ae_B": 80.0,
    "disc": 0.045,                     # 연 할인율
}
print(len(p), "개 입력값")
print("pfs_lam =", round(p["pfs_lam"], 5),
      " os_lam =", round(p["os_lam"], 5))
print("os_lam에서 되돌린 중앙값:",
      round((np.log(2) / p["os_lam"]) ** (1 / p["os_gam"]), 1))
''', title="입력값 사전")
save("lab23_inputs", c, marks={"18 개 입력값": 1, "0.07777": 2, "0.01502": 3, "28.0": 4})

c = nb.cell('''
t = np.arange(0, 241)                  # 주기 경계: 0, 1, ..., 240개월

def surv(lam, gam, t):
    """와이블 생존함수 S(t) = exp(-lam * t^gam)"""
    return np.exp(-lam * t ** gam)

def curves(p, arm, t):
    """군별 무진행생존(PFS), 전체생존(OS) 곡선"""
    hr_pfs = p["hr_pfs"] if arm == "A" else 1.0   # B는 기준군
    hr_os = p["hr_os"] if arm == "A" else 1.0
    s_pfs = surv(p["pfs_lam"] * hr_pfs, p["pfs_gam"], t)
    s_os = surv(p["os_lam"] * hr_os, p["os_gam"], t)
    return s_pfs, s_os

pfs_B, os_B = curves(p, "B", t)
pfs_A, os_A = curves(p, "A", t)
show = [0, 12, 24, 36, 60, 120, 240]
pd.DataFrame({"PFS_B": pfs_B, "OS_B": os_B,
              "PFS_A": pfs_A, "OS_A": os_A}).loc[show].round(4)
''', title="시간 격자와 군별 생존곡선")
save("lab23_curves", c, dfmarks={"0.4386": 1, "0.5596": 2, "0.6470": 3, "0.0003": 4, "0.0021": 5})

c = nb.cell('''
def occupancy(p, arm, t):
    """시점별 상태 비율: 무진행(pf), 진행(pd), 사망(dead)"""
    s_pfs, s_os = curves(p, arm, t)
    s_pfs = np.minimum(s_pfs, s_os)    # PFS가 OS를 넘지 않게
    return pd.DataFrame({"pf": s_pfs, "pd": s_os - s_pfs,
                         "dead": 1 - s_os}, index=t)

occ_B = occupancy(p, "B", t)
occ_A = occupancy(p, "A", t)
print(occ_B.shape)
print("B군 24개월:", occ_B.loc[24].round(4).tolist(),
      " 합:", occ_B.loc[24].sum())
both = pd.concat({"B": occ_B, "A": occ_A}, axis=1)
(both.loc[show] * 100).round(1)
''', title="두 곡선에서 상태 비율 만들기")
save("lab23_occ", c, marks={"(241, 3)": 1, "[0.2035, 0.3561, 0.4404]": 2},
     dfmarks={"43.9": 3, "33.1": 4, "23.0": 5, "58.5": 6})

c = nb.cell('''
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for ax, occ, name in [(axes[0], occ_B, "Standard therapy B"),
                      (axes[1], occ_A, "Drug A")]:
    ax.stackplot(occ.index, occ["pf"], occ["pd"], occ["dead"],
                 labels=["Progression-free", "Progressed", "Dead"],
                 colors=["tab:blue", "tab:orange", "lightgray"])
    ax.set_title(name)
    ax.set_xlabel("Months")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
axes[0].set_ylabel("Proportion of cohort")
axes[0].legend(loc="upper right", fontsize=8)
plt.tight_layout()
''', title="상태 비율의 쌓은 면적 그림")
save("lab23_occplot", c)

c = nb.cell('''
pf_t = occ_B["pf"].to_numpy()          # 241개: 0, 1, ..., 240개월
pd_t = occ_B["pd"].to_numpy()
pf = (pf_t[:-1] + pf_t[1:]) / 2        # 240개: 주기 시작과 끝의 평균
pd_ = (pd_t[:-1] + pd_t[1:]) / 2
print(len(pf_t), "->", len(pf))
print("첫 주기의 무진행:", pf_t[0], pf_t[1].round(4),
      "-> 평균", pf[0].round(4))

print("무진행 기간(년):", round(pf.sum() / 12, 3))
print("진행 기간(년)  :", round(pd_.sum() / 12, 3))
print("생존연수       :", round((pf + pd_).sum() / 12, 3))
print("보정 없이 주기 시작 값으로 세면:",
      round((pf_t[:-1] + pd_t[:-1]).sum() / 12, 3))
''', title="반주기 보정과 상태별 평균 기간 (표준요법 B)")
save("lab23_half", c, marks={"241 -> 240": 1, "평균 0.9626": 2, "1.255": 3, "1.799": 4, "3.054": 5, "3.095": 6})

# ---------------------------------------------------------------- 다. 비용과 QALY, ICER
c = nb.cell('''
died = os_B[:-1] - os_B[1:]            # 그 주기에 사망한 비율
mid = (t[:-1] + 0.5) / 12              # 주기 중간 시점(년)
disc = 1 / (1 + p["disc"]) ** mid      # 할인 계수

cyc = pd.DataFrame({
    "pf": pf, "pd": pd_, "died": died,
    "c_drug": pf * p["c_drug_B"],      # 약값: 무진행 상태에서만
    "c_pf": pf * p["c_pf"],            # 무진행 상태 관리비
    "c_pd": pd_ * p["c_pd"],           # 진행 상태 비용
    "c_death": died * p["c_death"],    # 임종기 비용
    "qaly": (pf * p["u_pf"] + pd_ * p["u_pd"]) / 12,
    "disc": disc}, index=t[1:])
cyc.loc[[1, 2, 12, 120, 240]].round(4).round(
    {"c_drug": 1, "c_pf": 1, "c_pd": 1, "c_death": 1})
''', title="주기별 비용과 QALY, 할인 계수 (표준요법 B)")
save("lab23_cycle", c, dfmarks={"115.5": 1, "11.9": 2, "0.0641": 3, "0.9982": 4, "0.6451": 5})

c = nb.cell('''
items = ["c_drug", "c_pf", "c_pd", "c_death"]
undisc = cyc[items].sum()                        # 할인 전 합계
after = cyc[items].mul(cyc["disc"], axis=0).sum()   # 할인 후 합계
print(pd.DataFrame({"undiscounted": undisc,
                    "discounted": after}).round(1))

cost_B = after.sum() + p["c_ae_B"]               # 이상반응 비용은 1회
qaly_B = (cyc["qaly"] * cyc["disc"]).sum() - p["du_ae_B"]
print("B 총비용:", round(cost_B, 1), " QALY:", round(qaly_B, 3))
print("할인 전 QALY:", round(cyc["qaly"].sum() - p["du_ae_B"], 3))
''', title="240주기를 더해 표준요법 B의 총비용과 QALY 구하기")
save("lab23_sumB", c, marks={"1707.8": 1, "4628.6": 2, "799.8": 3, "7689.5": 4, "QALY: 1.874": 5, "2.086": 6})

c = nb.cell('''
def run_model(p, horizon=240):
    """입력값 사전 p -> 결과 사전 (시간 개월, 비용 만원)"""
    t = np.arange(0, horizon + 1)            # 주기 경계 시점
    disc = 1 / (1 + p["disc"]) ** ((t[:-1] + 0.5) / 12)
    res = {}
    for arm in ["A", "B"]:
        hr_pfs = p["hr_pfs"] if arm == "A" else 1.0
        hr_os = p["hr_os"] if arm == "A" else 1.0
        s_os = np.exp(-p["os_lam"] * hr_os * t ** p["os_gam"])
        s_pfs = np.exp(-p["pfs_lam"] * hr_pfs * t ** p["pfs_gam"])
        s_pfs = np.minimum(s_pfs, s_os)      # PFS가 OS를 넘지 않게
        pd_t = s_os - s_pfs                  # 진행 = OS - PFS
        pf = (s_pfs[:-1] + s_pfs[1:]) / 2    # 반주기 보정
        pd_ = (pd_t[:-1] + pd_t[1:]) / 2
        died = s_os[:-1] - s_os[1:]          # 주기별 사망
        q = (pf * p["u_pf"] + pd_ * p["u_pd"]) / 12
        r = {"ly_pf": pf.sum() / 12, "ly_pd": pd_.sum() / 12,
             "ly": (pf + pd_).sum() / 12,
             "ly_d": ((pf + pd_) * disc).sum() / 12,
             "qaly_undisc": q.sum() - p["du_ae_" + arm],
             "qaly": (q * disc).sum() - p["du_ae_" + arm],
             "c_drug": (pf * p["c_drug_" + arm] * disc).sum(),
             "c_pf": (pf * p["c_pf"] * disc).sum(),
             "c_pd": (pd_ * p["c_pd"] * disc).sum(),
             "c_death": (died * p["c_death"] * disc).sum(),
             "c_ae": p["c_ae_" + arm]}
        r["cost"] = (r["c_drug"] + r["c_pf"] + r["c_pd"]
                     + r["c_death"] + r["c_ae"])
        res[arm] = r
    res["d_cost"] = res["A"]["cost"] - res["B"]["cost"]
    res["d_qaly"] = res["A"]["qaly"] - res["B"]["qaly"]
    res["d_ly"] = res["A"]["ly_d"] - res["B"]["ly_d"]
    dq = res["d_qaly"]
    res["icer"] = res["d_cost"] / dq if dq != 0 else np.nan
    return res

base = run_model(p)
print("함수:", round(base["B"]["cost"], 6), round(base["B"]["qaly"], 6))
print("단계:", round(cost_B, 6), round(qaly_B, 6))
''', title="모형을 함수 하나로 묶기")
save("lab23_runmodel", c, marks={"함수: 7689.48472": 1, "단계: 7689.48472": 2})

c = nb.cell('''
tab = pd.DataFrame({"A": base["A"], "B": base["B"]})
tab["diff"] = tab["A"] - tab["B"]
tab.round(3)
''', title="군별 결과 표", max_rows=20)
save("lab23_table", c, dfmarks={"1.974": 1, "3.916": 2, "3.399": 3, "2.732": 4, "2.385": 5, "4124.384": 6,
                                "-24.207": 7, "10561.778": 8})

c = nb.cell('''
threshold = 5000                       # 이 예시에서 가정한 임계값
print("증분비용  :", round(base["d_cost"], 1))
print("증분 QALY :", round(base["d_qaly"], 4))
print("ICER      :", round(base["icer"]), "만원/QALY")
print("생존연수당:", round(base["d_cost"] / base["d_ly"]),
      "만원/생존연수")
nmb = threshold * base["d_qaly"] - base["d_cost"]
print("증분 순금전편익:", round(nmb, 1), "만원")
''', title="증분, ICER, 순금전편익")
save("lab23_icer", c, marks={"2872.3": 1, "0.511": 2, "5621": 3, "4287": 4, "-317.1": 5})

c = nb.cell('''
p_fit = dict(p, os_lam=lam_fit, os_gam=gam_fit)   # 두 값만 바꾼 사본
r_fit = run_model(p_fit)
print("p는 그대로:", round(p["os_gam"], 4),
      " p_fit:", round(p_fit["os_gam"], 4))
print("적합한 와이블 곡선: 생존연수 B",
      round(r_fit["B"]["ly"], 2), " A", round(r_fit["A"]["ly"], 2))
print("증분비용", round(r_fit["d_cost"]),
      " 증분 QALY", round(r_fit["d_qaly"], 3),
      " ICER", round(r_fit["icer"]))
''', title="입력값을 바꿔 다시 돌리기 (가 절에서 적합한 곡선)")
save("lab23_refit", c, marks={"p_fit: 1.1454": 1, "A 3.98": 2, "5610": 3})

# ---------------------------------------------------------------- 라. 모형 점검
c = nb.cell('''
for arm in ["A", "B"]:
    x = occupancy(p, arm, t).to_numpy()          # 241행 x 3열
    assert np.isclose(x.sum(axis=1), 1).all(), "합이 1이 아닌 주기"
    assert (x >= 0).all(), "음수인 비율"
    s_pfs, s_os = curves(p, arm, t)              # 자르기 전 곡선
    assert (s_pfs <= s_os).all(), "PFS가 OS를 넘는 주기"
    print(arm, "합의 범위:", x.sum(axis=1).min(), x.sum(axis=1).max(),
          " 가장 작은 비율:", x.min().round(6),
          " OS - PFS의 최솟값:", (s_os - s_pfs).min().round(6))
print("점검 1, 2 통과")
''', title="점검 1과 2. 상태 비율의 합과 부호, 두 곡선의 순서")
save("lab23_chk12", c, marks={"0.9999999999999999": 1, "점검 1, 2 통과": 2})

c = nb.cell('''
same = dict(p, hr_pfs=1.0, hr_os=1.0,            # 효과가 같고
            c_drug_A=p["c_drug_B"],              # 약값이 같고
            c_ae_A=p["c_ae_B"], du_ae_A=p["du_ae_B"])
r = run_model(same)
print("증분비용:", r["d_cost"], " 증분 QALY:", r["d_qaly"],
      " ICER:", r["icer"])
assert r["d_cost"] == 0 and r["d_qaly"] == 0
print("점검 3 통과")
''', title="점검 3. 두 군에 같은 입력값을 넣으면 증분이 0")
save("lab23_chk3", c, marks={"증분비용: 0.0": 1, "nan": 2})

c = nb.cell('''
r0 = run_model(dict(p, disc=0.0))                # 4. 할인율 0%
r1 = run_model(dict(p, u_pf=1.0, u_pd=1.0,       # 5. 효용 1, 손실 0
                    du_ae_A=0.0, du_ae_B=0.0))
r2 = run_model(dict(p, c_drug_A=0.0, c_drug_B=0.0))   # 6. 약값 0
for arm in ["A", "B"]:
    assert np.isclose(r0[arm]["qaly"], base[arm]["qaly_undisc"])
    assert np.isclose(r0[arm]["ly_d"], base[arm]["ly"])
    assert np.isclose(r1[arm]["qaly"], base[arm]["ly_d"])
    assert r2[arm]["c_drug"] == 0
    assert r2[arm]["qaly"] == base[arm]["qaly"]
print("4. 할인율 0%: A의 총비용", round(r0["A"]["cost"]),
      " QALY", round(r0["A"]["qaly"], 3), " ICER", round(r0["icer"]))
print("5. 효용 1: A의 QALY", round(r1["A"]["qaly"], 3),
      "= 할인한 생존연수", round(base["A"]["ly_d"], 3))
print("6. 약값 0: A의 총비용", round(r2["A"]["cost"]),
      " 증분비용", round(r2["d_cost"]), " ICER", round(r2["icer"]))
print("점검 4, 5, 6 통과")
''', title="점검 4, 5, 6. 할인율 0%, 효용 1, 약값 0")
save("lab23_chk456", c, marks={"12194": 1, "QALY 2.732": 2, "3.399": 3, "6437": 4, "점검 4, 5, 6 통과": 5})

c = nb.cell('''
p_bad = dict(p, pfs_lam=weib_lam(30, 0.95))      # 중앙 PFS 30개월
s_pfs, s_os = curves(p_bad, "B", t)
print("PFS가 OS를 넘는 시점의 수:", (s_pfs > s_os).sum())
assert (s_pfs <= s_os).all(), "PFS가 OS를 넘는 주기가 있습니다"
''', title="점검에 걸리는 입력값", expect_error=True)
save("lab23_chkfail", c, marks={"220": 1})

# ---------------------------------------------------------------- 마. 과제
c = nb.cell('''
rows = []
for years in [3, 5, 10, 20]:
    r = run_model(p, horizon=years * 12)
    alive = np.exp(-p["os_lam"] * p["hr_os"]
                   * (years * 12) ** p["os_gam"])
    rows.append([years, r["d_cost"], r["d_qaly"], r["icer"], alive])
hw1 = pd.DataFrame(rows, columns=["years", "d_cost", "d_qaly",
                                  "icer", "alive_A"])
hw1.round({"d_cost": 0, "d_qaly": 3, "icer": 0, "alive_A": 3})
''', title="과제 1 정답. 분석기간에 따른 결과")
save("lab23_hw1", c, dfmarks={"7994.0": 1, "0.500": 2, "5686.0": 3, "5621.0": 4})

c = nb.cell('''
rows = []
for rate in [0.0, 0.03, 0.045]:
    r = run_model(dict(p, disc=rate))
    rows.append([rate, r["A"]["cost"], r["A"]["qaly"],
                 r["d_cost"], r["d_qaly"], r["icer"]])
hw2 = pd.DataFrame(rows, columns=["rate", "cost_A", "qaly_A",
                                  "d_cost", "d_qaly", "icer"])
print(hw2.round({"cost_A": 0, "qaly_A": 3, "d_cost": 0,
                 "d_qaly": 3, "icer": 0}).to_string(index=False))

dq_undisc = base["A"]["qaly_undisc"] - base["B"]["qaly_undisc"]
print("비용만 4.5%로 할인하고 QALY는 할인하지 않으면:",
      round(base["d_cost"] / dq_undisc))
''', title="과제 2 정답. 할인율에 따른 결과")
save("lab23_hw2", c, marks={"5431.0": 1, "5555.0": 2, "4447": 3})

c = nb.cell('''
months = base["A"]["c_drug"] / p["c_drug_A"]     # 할인한 투약 개월 수
nmb = threshold * base["d_qaly"] - base["d_cost"]   # 순금전편익
price = p["c_drug_A"] + nmb / months
print("A군의 할인한 투약 개월 수:", round(months, 3))
print("임계 가격:", round(price, 2), "만원,",
      "인하율:", round((1 - price / p["c_drug_A"]) * 100, 1), "%")

r = run_model(dict(p, c_drug_A=price))
print("그 가격에서의 ICER:", round(r["icer"], 4))
for x in [190, 180, 170]:
    print(x, round(run_model(dict(p, c_drug_A=x))["icer"]))
''', title="과제 3 정답. ICER가 임계값과 같아지는 약값")
save("lab23_hw3", c, marks={"21.707": 1, "175.39": 2, "인하율: 7.7": 3, "5000.0": 4})

print("lab23: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- 대조 블록: lib_p4.run()과 본문의 숫자에 맞추기
import lib_p4 as L  # noqa: E402  (읽기만 한다. 학생 코드는 이 파일을 쓰지 않는다)

ns = nb.ns
N = json.load(open(os.path.join(HERE, "_ch23_nums.json"), encoding="utf-8"))
CHECKS = []


def chk(name, got, want, tol=1e-9, grp="model"):
    """grp: "model" = lib_p4와 견준 모형 계산, "fit" = 23장 라 절의 적합 결과(_ch23_nums.json)와 견준 것"""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    err = float(np.max(np.abs(got - want)))
    CHECKS.append((name, err, grp))
    assert err < tol, f"{name}: lab {got} vs reference {want} (err {err:g})"


P0, REF = L.base_params(), L.run()
p_, base_ = ns["p"], ns["base"]
# 입력값 사전: key와 값이 base_params()와 같다
assert set(p_) == set(P0), set(p_) ^ set(P0)
chk("18 inputs = lib_p4.base_params()", [p_[k] for k in P0], [P0[k] for k in P0], 1e-15)
# 기준 분석: 군별 12개 결과와 증분, ICER, 순금전편익
KEYS = ("ly_pf", "ly_pd", "ly", "ly_d", "qaly_undisc", "qaly", "c_drug", "c_pf", "c_pd", "c_death", "c_ae", "cost")
for arm in ("A", "B"):
    assert set(base_[arm]) == set(KEYS)
    chk(f"run_model arm {arm}: 12 results", [base_[arm][k] for k in KEYS], [REF[arm][k] for k in KEYS])
chk("d_cost, d_qaly, d_ly, ICER", [base_[k] for k in ("d_cost", "d_qaly", "d_ly", "icer")],
    [REF[k] for k in ("d_cost", "d_qaly", "d_ly", "icer")])
chk("NMB, cost per life-year", [L.THRESHOLD * base_["d_qaly"] - base_["d_cost"], base_["d_cost"] / base_["d_ly"]],
    [L.nmb(REF), REF["icer_ly"]])
# 본문(20·22·23장)에 적힌 반올림 값
assert (round(base_["A"]["cost"]), round(base_["B"]["cost"]), round(base_["d_cost"]), round(base_["icer"])) == (10562, 7689, 2872, 5621)
assert (round(base_["A"]["qaly"], 3), round(base_["B"]["qaly"], 3), round(base_["d_qaly"], 3)) == (2.385, 1.874, 0.511)
assert [round(base_[a][k], 3) for a in "AB" for k in ("ly_pf", "ly_pd", "ly")] == [1.974, 1.942, 3.916, 1.255, 1.799, 3.054]
assert [round(base_[a]["ly_d"], 3) for a in "AB"] == [3.399, 2.729] and round(base_["d_ly"], 3) == 0.670
assert [round(base_[a][k]) for a in "AB" for k in ("c_drug", "c_pf", "c_pd", "c_death")] == [4124, 868, 4769, 680, 1708, 569, 4629, 704]
assert round(L.nmb(REF)) == -317 and round(REF["icer_ly"]) == 4287
# 나 절: 곡선과 상태 비율(lib_p4의 trace), 반주기 보정
for arm in ("A", "B"):
    tr, occ = REF[arm]["trace"], ns["occ_" + arm]
    chk(f"occupancy arm {arm} (241 x 3)", occ[["pf", "pd", "dead"]].to_numpy().T, [tr["pf"], tr["pd"], tr["dead"]], 1e-12)
    s_pfs, s_os = L.curves(P0, arm, tr["t"])
    chk(f"curves arm {arm}", [ns["pfs_" + arm], ns["os_" + arm]], [s_pfs, s_os], 1e-12)
chk("B: PF, PD years (step by step)", [ns["pf"].sum() / 12, ns["pd_"].sum() / 12], [REF["B"]["ly_pf"], REF["B"]["ly_pd"]])
chk("B: life-years without half-cycle correction", (ns["pf_t"][:-1] + ns["pd_t"][:-1]).sum() / 12, L.run(half_cycle=False)["B"]["ly"])
# 다 절: 단계별 계산(표준요법 B)과 함수의 결과
chk("B: cost, QALY (step by step)", [ns["cost_B"], ns["qaly_B"]], [REF["B"]["cost"], REF["B"]["qaly"]])
chk("B: discounted cost items (step by step)", ns["after"][["c_drug", "c_pf", "c_pd", "c_death"]].tolist(),
    [REF["B"][k] for k in ("c_drug", "c_pf", "c_pd", "c_death")])
chk("discount factors: cycles 1, 12, 120", ns["cyc"].loc[[1, 12, 120], "disc"].tolist(),
    [1 / 1.045 ** (0.5 / 12), 1 / 1.045 ** (11.5 / 12), 1 / 1.045 ** (119.5 / 12)], 1e-12)
# 가 절: Kaplan-Meier, 다섯 모형의 적합(23장 표 23-11), 와이블 모수 변환
KM = N["km"]
trial_ = ns["trial"]
assert (len(trial_), int(trial_["event"].sum()), int((trial_["event"] == 0).sum())) == (KM["n"], KM["events"], KM["cens"])
chk("KM S(12), S(24), S(last)", ns["kmf"].predict([12, 24]).tolist() + [ns["kmf"].survival_function_.iloc[-1, 0]],
    [KM["s12"], KM["s24"], KM["s_last"]], 1e-9, "fit")
assert np.isinf(ns["kmf"].median_survival_time_) and np.isinf(KM["median"])
ft = ns["fit_tab"]
for name, key in (("Exponential", "exp"), ("Weibull", "weib"), ("Log-logistic", "llog"), ("Log-normal", "lnorm"),
                  ("Gen. gamma", "ggam")):
    x = N["fit"][key]
    assert int(ft.loc[name, "k"]) == x["k"]
    # 본문은 반올림하기 전의 모의 자료로, 실습은 소수 여섯째 자리까지 저장한 CSV로 적합하므로 1e-5쯤의 차이가 난다
    chk(f"fit {name}: AIC, BIC, S(2y, 5y, 10y, 20y), mean", ft.loc[name, ["AIC", "BIC", "S(2y)", "S(5y)", "S(10y)", "S(20y)", "mean_y"]],
        [x["aic"], x["bic"], x["s24"], x["s60"], x["s120"], x["s240"], x["mean_B"]], 1e-4, "fit")
W = N["fit"]["weib"]
chk("Weibull: lambda_, rho_", [ns["w"].lambda_, ns["w"].rho_], [W["params"]["lambda_"], W["params"]["rho_"]], 1e-4, "fit")
chk("Weibull: lam, gam (lib_p4.lifelines_weibull_to_lib)", [ns["lam_fit"], ns["gam_fit"]],
    L.lifelines_weibull_to_lib(ns["w"].lambda_, ns["w"].rho_), 1e-15)
chk("weib_lam = lib_p4.weib_lam", [ns["weib_lam"](10, 0.95), ns["weib_lam"](28, 1.15)], [L.weib_lam(10.0, 0.95), L.weib_lam(28.0, 1.15)], 1e-15)
rf = ns["r_fit"]
chk("fitted Weibull in the model (Table 23-12): LY B, LY A, d_cost, d_qaly, ICER",
    [rf["B"]["ly"], rf["A"]["ly"], rf["d_cost"], rf["d_qaly"], rf["icer"]], [W["ly_B"], W["ly_A"], W["d_cost"], W["d_qaly"], W["icer"]], 0.02, "fit")
assert round(rf["icer"]) == 5610 and round(rf["d_cost"]) == 2902 and round(rf["d_qaly"], 3) == 0.517
# 라 절: 점검에 쓴 값
chk("discount 0%: A cost, A QALY, ICER", [ns["r0"]["A"]["cost"], ns["r0"]["A"]["qaly"], ns["r0"]["icer"]],
    [L.run(disc=0.0)["A"]["cost"], REF["A"]["qaly_undisc"], L.run(disc=0.0)["icer"]])
assert round(ns["r0"]["A"]["cost"]) == 12194 and round(ns["r0"]["A"]["qaly"], 3) == 2.732          # 20장 '할인'
# 과제: 분석기간(20장 표 20-6), 할인율(표 20-8), 임계 가격(24장 가 절)
hw1_, hw2_ = ns["hw1"], ns["hw2"]
for i, yrs in enumerate((3, 5, 10, 20)):
    q = L.run(horizon=yrs * 12)
    chk(f"horizon {yrs} y: d_cost, d_qaly, ICER", hw1_.loc[i, ["d_cost", "d_qaly", "icer"]], [q["d_cost"], q["d_qaly"], q["icer"]])
assert [(round(a), round(b, 3), round(c)) for a, b, c in hw1_[["d_cost", "d_qaly", "icer"]].to_numpy()] == \
    [(1316, 0.165, 7994), (1875, 0.297, 6324), (2640, 0.464, 5686), (2872, 0.511, 5621)]
for i, rate in enumerate((0.0, 0.03, 0.045)):
    q = L.run(disc=rate)
    chk(f"discount {rate}: d_cost, d_qaly, ICER", hw2_.loc[i, ["d_cost", "d_qaly", "icer"]], [q["d_cost"], q["d_qaly"], q["icer"]])
assert [(round(a), round(b, 3), round(c)) for a, b, c in hw2_[["d_cost", "d_qaly", "icer"]].to_numpy()[:2]] == \
    [(3508, 0.646, 5431), (3058, 0.550, 5555)]
assert round(base_["d_cost"] / ns["dq_undisc"]) == 4447                                             # 20장 다 절 흔한 오해
chk("threshold price = lib_p4.threshold_price()", ns["price"], L.threshold_price())
assert round(ns["price"], 1) == 175.4 and round(ns["months"], 1) == 21.7
print(f"대조: {len(CHECKS)}개 항목 모두 허용 오차 안 (lib_p4와 견준 {sum(g == 'model' for _, _, g in CHECKS)}개의 가장 큰 차이 "
      f"{max(e for _, e, g in CHECKS if g == 'model'):.2e}, "
      f"23장 라 절과 견준 {sum(g == 'fit' for _, _, g in CHECKS)}개의 가장 큰 차이 {max(e for _, e, g in CHECKS if g == 'fit'):.2e})")

# ---------------------------------------------------------------- 본문 문장에 쓴 그 밖의 숫자 확인
if os.environ.get("VERIFY") or os.environ.get("NOMARK"):
    print("VERIFY drug share of incremental cost:", round((base_["A"]["c_drug"] - base_["B"]["c_drug"]) / base_["d_cost"], 3))
    print("VERIFY 20-year survival A, B:", round(float(ns["os_A"][240]), 4), round(float(ns["os_B"][240]), 5))
    print("VERIFY medians of fitted models:", {n_: round(float(f.median_survival_time_), 1) for n_, f in ns["models"].items()})
    print("VERIFY B undiscounted total cost:", round(ns["r0"]["B"]["cost"], 1), " first-cycle cost:",
          round(float(ns["cyc"].loc[1, ["c_drug", "c_pf", "c_pd", "c_death"]].sum()), 1))
    print("VERIFY price 180, 170 ICER:", [round(ns["run_model"](dict(p_, c_drug_A=x))["icer"]) for x in (180, 170)])

# ---------------------------------------------------------------- 노트북 (pub/notebooks/lab23.ipynb)
notes = {
    1: "## 가. 생존곡선의 적합과 외삽\n\n가상 임상시험(표준요법 B군 300명)의 전체생존 자료에 모수 모형 다섯 개를 적합하고 20년까지 늘려 그립니다.",
    7: "## 나. 분할생존모형 만들기\n\n입력값 사전, 시간 격자, 군별 생존곡선, 상태 비율을 차례로 만듭니다.",
    12: "## 다. 비용과 QALY, ICER\n\n주기별 비용과 QALY를 할인해 더하고, 모형 전체를 함수 `run_model(p)`로 묶습니다.",
    18: "## 라. 모형 점검\n\n답을 미리 아는 입력값을 넣어 모형이 제대로 계산하는지 확인합니다. "
        "사이트의 셀 21(점검에 걸리는 입력값을 넣어 일부러 오류를 내는 셀)은 이 노트북에서 뺐습니다.",
    22: "## 과제 정답\n\n**과제 1.** 분석기간을 3년, 5년, 10년, 20년으로 바꿔 증분비용, 증분 QALY, ICER를 구하세요.",
    23: "**과제 2.** 할인율을 0%, 3%, 4.5%로 바꿔 신약 A의 총비용과 QALY, 증분비용, 증분 QALY, ICER를 구하세요.",
    24: "**과제 3.** ICER가 5,000만원/QALY(이 예시에서 가정한 임계값)가 되는 신약 A의 월 약값을 구하세요.",
}
path = nb.save_ipynb(
    "실습 23. 결정분석 모형 만들기",
    intro=("사회약학 연구방법 노트의 '실습 23 결정분석 모형 만들기'에 나오는 셀을 차례로 모은 노트북입니다. "
           "진행성 신세포암 1차 치료(신약 A 대 표준요법 B, 가상의 예시)의 세 상태 분할생존모형을 numpy와 pandas만으로 만들어 "
           "20–25장 공통 예시의 기준 분석(ICER 5,621만원/QALY)을 재현합니다. "
           "셀을 위에서부터 차례로 실행하세요. 가 절의 자료는 사이트에서 바로 읽으므로 인터넷 연결이 필요합니다. "
           "설명과 출력 읽는 법은 사이트의 실습 23 쪽에 있습니다."),
    skip=(21,), notes=notes)
print("notebook:", path, len(nb.cells), "cells")
