"""실습 22 · 효용과 QALY 계산 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab22.py && python3 gen/lab_lab22.py
      then  python3 tools/nbcheck.py lab22   (runs the notebook alone in a fresh process and compares outputs)
      (NOMARK=1 을 앞에 붙이면 표식 없이 셀 출력만 화면에 찍는다. VERIFY=1 이면 본문 문장에 쓴 그 밖의 숫자도 찍는다)

자료: pub/data/lab22_eq5d.csv, pub/data/lab22_patients.csv (gen/data_lab22.py가 22장의 gen/_ch22_trial.csv, gen/_ch22_pts.csv에서 만든다.
이 생성기는 자료를 다시 만들지 않는다). 학생 코드는 gen/lib_p4.py를 쓰지 않는다(가치 세트와 모형을 짧은 함수로 다시 짠다).
가치 세트의 계수: 한국 5L과 한국 3L은 P4_FACTS.md 5.4, 5.3. 영국 3L은 P4_FACTS.md 5.5(22장 gen/nums_ch22.py의 UK3와 같은 값,
출처는 R 패키지 eq5d의 data-raw/TTO.csv, UK 열).
끝의 대조 블록이 실습 결과를 gen/_ch22_nums.json(22장 본문의 숫자), gen/lib_p4.py(공통 예시 모형)와 맞추고,
22장 작성자의 작업 공간 스크립트(calc22_new.py)가 낸 값(VAS ÷ 100, 생존자만의 평균, 할인 시점, 마지막 값 유지)을 반올림 자릿수로 확인한다.
(2026-10-15 새로 만듦. 22장을 새 틀로 고친 뒤 PART 4 확장의 '실습 20·22(새로 만듦, 작게)'에 따라.)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab22")


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


# ---------------------------------------------------------------- a. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.integrate import trapezoid
import matplotlib, scipy, statsmodels

print("numpy      ", np.__version__)
print("pandas     ", pd.__version__)
print("matplotlib ", matplotlib.__version__)
print("scipy      ", scipy.__version__)
print("statsmodels", statsmodels.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab22_versions", c, marks={"pandas      3.0.5": 1, "scipy       1.18.1": 2, "statsmodels 0.15.0": 3})

c = nb.cell('''
BASE = "https://socialp-ajou.tecentriq12.workers.dev/data/"
UA = {"User-Agent": "Mozilla/5.0"}   # 사이트가 파이썬 기본 요청을 막아 브라우저처럼 보이게 함
df = pd.read_csv(BASE + "lab22_eq5d.csv", storage_options=UA,
                 dtype={"eq5d": str})          # 응답 코드는 글자로 읽는다
pts = pd.read_csv(BASE + "lab22_patients.csv", storage_options=UA)
print(df.shape, pts.shape)
print(pts.head(3))
df.head()
''', title="설문 응답과 환자 자료 불러오기")
save("lab22_load", c, marks={"(1593, 7)": 1, "(320, 4)": 2, "17.300321": 3}, dfmarks={"42444": 4})

c = nb.cell('''
print("환자:", pts.groupby("arm").size().to_dict())
print("시점별 응답:", df.groupby("month").size().to_dict())
print("상태별 응답:", df["state"].value_counts().to_dict())
print("진행 뒤에 응답한 환자:", df.loc[df["state"] == "PD", "id"].nunique(), "명")

died = pts["t_death"] <= 24                     # 24개월 안에 사망
print("24개월 안의 사망:", pts[died].groupby("arm").size().to_dict())
print("진행 없이 사망(t_prog가 inf):", np.isinf(pts["t_prog"]).sum(), "명")

visits = np.array([0, 3, 6, 12, 18, 24])
alive = pts["t_death"].apply(lambda d: (visits < d).sum())   # 환자마다 살아 있던 방문 수
print("살아 있던 방문:", alive.sum(), "번, 응답:", len(df), "건")
''', title="자료 확인")
save("lab22_check", c, marks={"{'A': 160, 'B': 160}": 1, "{'PF': 1184, 'PD': 409}": 2, "174 명": 3,
                             "{'A': 55, 'B': 68}": 4, "47 명": 5, "1593 번": 6})

c = nb.cell('''
b0 = df[df["month"] == 0].copy()                # 치료 시작 시점, 환자마다 한 행
dims = ["mobility", "selfcare", "activity", "pain", "anxiety"]
for i, d in enumerate(dims):
    b0[d] = b0["eq5d"].str[i].astype(int)       # 코드의 i번째 자리 = 그 영역의 수준

print((b0[dims] > 1).groupby(b0["arm"]).sum())  # 수준 2-5(문제 있음)인 환자 수
b0.groupby("arm").agg(utility=("utility", "mean"), utility_sd=("utility", "std"),
                      vas=("vas", "mean"), vas_sd=("vas", "std")).round(3)
''', title="치료 시작 시점의 세 가지 숫자")
save("lab22_base", c, marks={"119": 1, "129": 2}, dfmarks={"0.786": 3, "73.456": 4})

# ---------------------------------------------------------------- b. 효용 점수 구하기
c = nb.cell('''
# 한국 EQ-5D-5L 가치 세트(Kim 등, 2016)의 감점. 영역마다 수준 1, 2, 3, 4, 5
KR5 = {"mobility": [0, 0.046, 0.058, 0.133, 0.251],
       "selfcare": [0, 0.032, 0.050, 0.078, 0.122],
       "activity": [0, 0.021, 0.051, 0.100, 0.175],
       "pain":     [0, 0.042, 0.053, 0.166, 0.207],
       "anxiety":  [0, 0.033, 0.046, 0.102, 0.137]}

def kr5l(code):
    """다섯 자리 응답 코드(글자) -> 한국 EQ-5D-5L 효용"""
    lv = [int(x) for x in code]              # "21222" -> [2, 1, 2, 2, 2]
    if lv == [1, 1, 1, 1, 1]:
        return 1.0                           # 완전한 건강은 감점 없음
    u = 1 - 0.096                            # 11111이 아니면 붙는 상수 감점
    for dim, level in zip(KR5, lv):
        u = u - KR5[dim][level - 1]          # 영역별 감점 (수준 1이면 0)
    if max(lv) >= 4:
        u = u - 0.078                        # 수준 4나 5가 하나라도 있으면
    return round(u, 3)

for code in ["21222", "31342", "11211", "55555", "11111"]:
    print(code, kr5l(code))
''', title="한국 EQ-5D-5L 가치 세트 함수")
save("lab22_kr5l", c, marks={"21222 0.762": 1, "31342 0.518": 2, "11211 0.883": 3, "55555 -0.066": 4, "11111 1.0": 5})

c = nb.cell('''
df["u_calc"] = df["eq5d"].map(kr5l)            # 1,593건 모두
print("서로 다른 응답 코드:", df["eq5d"].nunique(), "가지")
print("utility 열과의 차이(가장 큰 값):", (df["u_calc"] - df["utility"]).abs().max())
print("11111(효용 1) 응답:", (df["eq5d"] == "11111").sum(), "건")
df[df["u_calc"] < 0][["id", "arm", "month", "state", "eq5d", "u_calc"]]
''', title="응답 1,593건을 효용으로 바꾸고 utility 열과 비교하기")
save("lab22_apply", c, marks={"227": 1, "(가장 큰 값): 0.0": 2, "180 건": 3}, dfmarks={"-0.031": 4})

c = nb.cell('''
# EQ-5D-3L 가치 세트: 상수 감점, 수준 3이 하나라도 있으면 붙는 감점, 영역별 (수준 2, 수준 3) 감점
KR3 = {"const": 0.050, "n3": 0.050,             # 한국 (Lee 등, 2009)
       "dec": [(0.096, 0.418), (0.046, 0.136), (0.051, 0.208),
               (0.037, 0.151), (0.043, 0.158)]}
UK3 = {"const": 0.081, "n3": 0.269,             # 영국 (Dolan, 1997)
       "dec": [(0.069, 0.314), (0.104, 0.214), (0.036, 0.094),
               (0.123, 0.386), (0.071, 0.236)]}

def eq3l(code, vs):
    """다섯 자리 3L 응답 코드 -> 가치 세트 vs의 효용"""
    lv = [int(x) for x in code]
    if lv == [1, 1, 1, 1, 1]:
        return 1.0
    u = 1 - vs["const"]
    for (d2, d3), level in zip(vs["dec"], lv):
        u = u - {1: 0, 2: d2, 3: d3}[level]
    if 3 in lv:
        u = u - vs["n3"]
    return round(u, 3)

print("검산 21232 (한국):", eq3l("21232", KR3))
codes = ["11121", "21222", "33333"]
t3 = pd.DataFrame({"korea": [eq3l(x, KR3) for x in codes],
                   "uk": [eq3l(x, UK3) for x in codes]}, index=codes)
t3["diff"] = t3["korea"] - t3["uk"]
t3.round(3)
''', title="EQ-5D-3L 가치 세트 두 개로 같은 응답 바꾸기")
save("lab22_eq3l", c, marks={"0.559": 1}, dfmarks={"0.913": 2, "0.796": 3, "0.423": 4})

c = nb.cell('''
naive = smf.ols("utility ~ 0 + state", data=df).fit()
clus = smf.ols("utility ~ 0 + state", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["id"]})   # 환자 단위로 묶음

ci = clus.conf_int()
st = pd.DataFrame({"mean": clus.params,
                   "se_naive": naive.bse,
                   "se_cluster": clus.bse,
                   "lo": ci[0], "hi": ci[1]})
st.index = [k[6:-1] for k in st.index]         # "state[PF]" -> "PF"
st["n"] = df["state"].value_counts()
st.loc[["PF", "PD"]].round({"mean": 3, "se_naive": 4, "se_cluster": 4,
                             "lo": 3, "hi": 3})
''', title="상태별 평균 효용과 환자 단위로 묶은 신뢰구간")
save("lab22_state", c, dfmarks={"0.780": 1, "0.0045": 2, "0.0074": 3, "0.765": 4, "0.621": 5})

c = nb.cell('''
by = df.groupby("state").agg(n=("id", "size"), utility=("utility", "mean"),
                             vas=("vas", "mean"))
by["vas_100"] = by["vas"] / 100                # EQ VAS를 100으로 나눈 값
by = by.loc[["PF", "PD"]]
gap = by.loc["PF"] - by.loc["PD"]              # 무진행 - 진행
print("두 상태의 차이  효용:", round(gap["utility"], 3),
      "  VAS/100:", round(gap["vas_100"], 3))
by.round(3)
''', title="EQ VAS ÷ 100과 효용 비교하기")
save("lab22_vas", c, marks={"효용: 0.158": 1, "VAS/100: 0.096": 2}, dfmarks={"0.732": 3, "0.636": 4})

# ---------------------------------------------------------------- c. QALY와 할인
c = nb.cell('''
def path(month, u, t_death, end=24):
    """측정 시점(개월)과 효용 -> 사망과 추적 끝을 이은 (시점, 효용) 경로"""
    t, u = list(month), list(u)
    if t_death <= end:                    # 추적 기간 안에 사망
        t, u = t + [t_death], u + [0.0]   # 사망 시점의 효용은 0
        if t_death < end:
            t, u = t + [end], u + [0.0]   # 그 뒤 추적 끝까지 0
    return np.array(t, dtype=float), np.array(u, dtype=float)

# 환자 다 (22장 표 22-4). 0, 3, 6, 12개월에 응답하고 15개월에 사망
u_c = [kr5l(x) for x in ["11221", "21222", "21232", "31241"]]
t, u = path([0, 3, 6, 12], u_c, t_death=15)
print("시점:", t)
print("효용:", u)
print("구간별 QALY:", np.round(np.diff(t) * (u[:-1] + u[1:]) / 2 / 12, 3))
print("QALY:", round(trapezoid(u, t) / 12, 3), " 생존연수:", 15 / 12)
''', title="환자 한 명의 QALY (사다리꼴 공식)")
save("lab22_one", c, marks={"[0.2   0.189 0.333 0.073 0.   ]": 1, "QALY: 0.795": 2})

c = nb.cell('''
paths, q, ly, base = {}, [], [], []
for r in pts.itertuples():                     # 환자 한 명씩
    g = df[df["id"] == r.id]                   # 이 환자의 응답 (시점 순서)
    t, u = path(g["month"], g["utility"], r.t_death)
    paths[r.id] = (t, u)                       # 뒤의 셀에서 다시 씀
    q.append(trapezoid(u, t) / 12)
    ly.append(min(r.t_death, 24) / 12)
    base.append(g["utility"].iloc[0])          # 치료 시작 시점의 효용
pts["qaly"], pts["ly"], pts["base"] = q, ly, base

print(pts.head(3).round(3))
pts.groupby("arm").agg(qaly=("qaly", "mean"), qaly_sd=("qaly", "std"),
                       ly=("ly", "mean"), base=("base", "mean")).round(3)
''', title="환자 320명의 24개월 QALY와 생존연수")
save("lab22_all", c, marks={"0.799": 1}, dfmarks={"1.204": 2, "0.529": 3, "1.640": 4, "1.571": 5})

c = nb.cell('''
pts["A"] = (pts["arm"] == "A").astype(int)       # 신약 A = 1, B = 0
m0 = smf.ols("qaly ~ A", data=pts).fit()
m1 = smf.ols("qaly ~ A + base", data=pts).fit()
for name, m in [("보정 전", m0), ("보정 후", m1)]:
    lo, hi = m.conf_int().loc["A"]
    print(name, " 차이", round(m.params["A"], 3),
          " 95% CI", round(lo, 3), round(hi, 3), " SE", round(m.bse["A"], 3))

mb = pts.groupby("arm")["base"].mean()
print("기저 효용의 계수:", round(m1.params["base"], 3))
print("기저 효용의 차이", round(mb["A"] - mb["B"], 4),
      "x 계수 =", round((mb["A"] - mb["B"]) * m1.params["base"], 3))
''', title="기저 효용 보정 전후의 QALY 차이")
save("lab22_reg", c, marks={"차이 0.117": 1, "0.001 0.232": 2, "차이 0.094": 3, "-0.017 0.204": 4, "1.186": 5, "= 0.023": 6})

c = nb.cell('''
# 행: 시점, 열: 군. 그 시점에 살아서 답한 사람만의 평균
alive_mean = df.groupby(["month", "arm"])["utility"].mean().unstack()
alive_n = df.groupby(["month", "arm"]).size().unstack()
print(pd.concat({"mean": alive_mean.round(3), "n": alive_n}, axis=1))

wrong = trapezoid(alive_mean, visits, axis=0) / 12   # 생존자만의 평균을 이은 곡선 아래 넓이
print("생존자만의 평균 곡선:", wrong.round(3))
print("환자별 QALY의 평균  :", pts.groupby("arm")["qaly"].mean().round(3).to_numpy())
''', title="생존자만의 평균 효용으로 계산하면")
save("lab22_surv", c, marks={"0.719": 1, "[1.496 1.411]": 2, "[1.204 1.087]": 3})

c = nb.cell('''
grid = np.union1d(visits, pts.loc[died, "t_death"])   # 방문 시점과 모든 사망 시점
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, arm in zip(axes, ["A", "B"]):
    ids = pts.loc[pts["arm"] == arm, "id"]
    curve = np.mean([np.interp(grid, *paths[i]) for i in ids], axis=0)
    ax.plot(grid, curve, label="All patients (dead = 0)")
    ax.plot(visits, alive_mean[arm], "o--", label="Survivors only")
    ax.set_title("Drug A" if arm == "A" else "Standard therapy B")
    ax.set_xlabel("Months since randomization")
    ax.set_xticks(visits)
    print(arm, "평균 곡선 아래 넓이:", round(trapezoid(curve, grid) / 12, 3))
axes[0].set_ylabel("Mean utility")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=8)
plt.tight_layout()
''', title="두 평균 곡선 그리기")
save("lab22_survplot", c, marks={"넓이: 1.204": 1, "넓이: 1.087": 2})

c = nb.cell('''
def split_qaly(t, u, cut=12):
    """경로를 cut개월에서 나눠 (앞부분의 QALY, 뒷부분의 QALY)"""
    uc = np.interp(cut, t, u)                  # cut개월의 효용 (직선 위의 값)
    t1, u1 = np.append(t[t < cut], cut), np.append(u[t < cut], uc)
    t2, u2 = np.insert(t[t > cut], 0, cut), np.insert(u[t > cut], 0, uc)
    return trapezoid(u1, t1) / 12, trapezoid(u2, t2) / 12

yr = np.array([split_qaly(*paths[i]) for i in pts["id"]])   # 320행 x 2열
pts["qaly_d"] = yr[:, 0] + yr[:, 1] / 1.045               # 둘째 해만 할인
print("첫해 + 둘째 해 = 할인 전 QALY:", np.allclose(yr.sum(axis=1), pts["qaly"]))
mq = pts.groupby("arm")[["qaly", "qaly_d"]].mean()
print(mq.round(3))
print("차이  할인 전", round(mq.loc["A", "qaly"] - mq.loc["B", "qaly"], 3),
      " 둘째 해 할인", round(mq.loc["A", "qaly_d"] - mq.loc["B", "qaly_d"], 3))
''', title="둘째 해만 할인한 24개월 QALY")
save("lab22_disc2", c, marks={"True": 1, "1.181": 2, "둘째 해 할인 0.113": 3})

c = nb.cell('''
p = {"pfs_med": 10, "pfs_gam": 0.95,    # 표준요법 B 무진행생존: 중앙값(개월), 모양
     "os_med": 28, "os_gam": 1.15,      # 표준요법 B 전체생존
     "hr_pfs": 0.65, "hr_os": 0.75,     # 신약 A의 위험비
     "u_pf": 0.78, "u_pd": 0.62,        # 상태별 효용 (셀 8의 0.780, 0.621)
     "du_ae_A": 0.012, "du_ae_B": 0.008,                         # 이상반응 QALY 손실
     "c_drug_A": 190, "c_drug_B": 120, "c_pf": 40, "c_pd": 250,  # 월 비용(만원)
     "c_death": 800, "c_ae_A": 120, "c_ae_B": 80,                # 1회 비용(만원)
     "disc": 0.045}                     # 연 할인율

def model(p, when="mid", months=240):
    """공통 예시의 분할생존모형 (1개월 주기, 20년). when: 할인 계수를 매기는 시점"""
    t = np.arange(months + 1)           # 주기 경계: 0, 1, ..., 240개월
    k = np.arange(months)               # 주기 번호: 0, 1, ..., 239
    years = {"mid": (k + 0.5) / 12,     # 각 달의 한가운데 (기준 분석)
             "start": k / 12,           # 각 달의 시작
             "end": (k + 1) / 12,       # 각 달의 끝
             "annual": k // 12,         # 연 단위 (첫해는 0년, 할인 없음)
             "wrong": k + 0.5}[when]    # 잘못: 개월 수를 연수 자리에
    disc = 1 / (1 + p["disc"]) ** years # 주기별 할인 계수 240개
    res = {}
    for arm in ["A", "B"]:
        hp = p["hr_pfs"] if arm == "A" else 1
        ho = p["hr_os"] if arm == "A" else 1
        s_os = np.exp(-np.log(2) * ho * (t / p["os_med"]) ** p["os_gam"])
        s_pfs = np.exp(-np.log(2) * hp * (t / p["pfs_med"]) ** p["pfs_gam"])
        s_pfs = np.minimum(s_pfs, s_os)
        pf = (s_pfs[:-1] + s_pfs[1:]) / 2                 # 반주기 보정
        pd_ = ((s_os - s_pfs)[:-1] + (s_os - s_pfs)[1:]) / 2
        died_k = s_os[:-1] - s_os[1:]                     # 주기별 사망
        r = {"pf_years": (pf * disc).sum() / 12,          # 할인한 상태별 기간(년)
             "pd_years": (pd_ * disc).sum() / 12}
        r["qaly"] = (p["u_pf"] * r["pf_years"] + p["u_pd"] * r["pd_years"]
                     - p["du_ae_" + arm])
        r["cost"] = ((pf * (p["c_drug_" + arm] + p["c_pf"]) + pd_ * p["c_pd"]
                      + died_k * p["c_death"]) * disc).sum() + p["c_ae_" + arm]
        res[arm] = r
    return res

res = model(p)
print("QALY   A", round(res["A"]["qaly"], 3), " B", round(res["B"]["qaly"], 3))
print("총비용 A", round(res["A"]["cost"], 1), " B", round(res["B"]["cost"], 1))
print("ICER", round((res["A"]["cost"] - res["B"]["cost"])
                    / (res["A"]["qaly"] - res["B"]["qaly"])))
''', title="공통 예시 모형을 함수 하나로")
save("lab22_model", c, marks={"A 2.385": 1, "B 1.874": 2, "ICER 5621": 3})

c = nb.cell('''
rows = {}
for arm in ["A", "B"]:
    r = res[arm]
    rows[arm] = {"pf_years": r["pf_years"], "pd_years": r["pd_years"],
                 "life_years": r["pf_years"] + r["pd_years"],
                 "qaly_pf": p["u_pf"] * r["pf_years"],
                 "qaly_pd": p["u_pd"] * r["pd_years"],
                 "ae_loss": -p["du_ae_" + arm],
                 "qaly": r["qaly"]}
tab = pd.DataFrame(rows)
tab["diff"] = tab["A"] - tab["B"]
tab.round(3)
''', title="모형의 20년 QALY를 상태별로 나누기")
save("lab22_decomp", c, dfmarks={"1.809": 1, "0.623": 2, "0.670": 3, "0.486": 4, "0.511": 5})

c = nb.cell('''
rows = []
for when in ["mid", "start", "end", "annual", "wrong"]:
    r = model(p, when)
    dq = r["A"]["qaly"] - r["B"]["qaly"]
    dc = r["A"]["cost"] - r["B"]["cost"]
    rows.append([when, r["A"]["qaly"], r["B"]["qaly"], dq, dc / dq])
when_tab = pd.DataFrame(rows, columns=["when", "qaly_A", "qaly_B",
                                       "d_qaly", "icer"]).set_index("when")
print("12번째 달의 할인 계수  기준:", round(1 / 1.045 ** (11.5 / 12), 3),
      " wrong:", round(1 / 1.045 ** 11.5, 3))
when_tab.round(3).round({"icer": 0})
''', title="할인 계수를 매기는 시점 바꾸기")
save("lab22_when", c, marks={"기준: 0.959": 1, "wrong: 0.603": 2}, dfmarks={"5620.0": 3, "0.523": 4, "9140.0": 5})

# ---------------------------------------------------------------- d. 과제
c = nb.cell('''
pf_resp = df[df["state"] == "PF"]
print("응답", len(pf_resp), "건, 환자", pf_resp["id"].nunique(), "명")
print("효용이 1인 응답의 비율:", round((pf_resp["utility"] == 1).mean(), 3))
print("개별 응답의 가운데 95% 범위:",
      pf_resp["utility"].quantile([0.025, 0.975]).round(3).tolist())
print("군별 평균:", pf_resp.groupby("arm")["utility"].mean().round(3).to_dict())
''', title="과제 1 정답. 무진행 상태의 효용 한 행 읽기")
save("lab22_hw1", c, marks={"0.147": 1, "[0.424, 1.0]": 2, "{'A': 0.783, 'B': 0.775}": 3})

c = nb.cell('''
def path_locf(month, u, t_death, end=24):
    """마지막 측정값이 사망 직전까지 유지된다고 보는 경로"""
    t, u = list(month), list(u)
    if t_death <= end:                    # 사망 직전까지 마지막 값
        t, u = t + [t_death, t_death], u + [u[-1], 0.0]
        if t_death < end:
            t, u = t + [end], u + [0.0]
    return np.array(t, dtype=float), np.array(u, dtype=float)

t, u = path_locf([0, 3, 6, 12], u_c, t_death=15)
print("환자 다:", round(trapezoid(u, t) / 12, 3))

q2 = []
for r in pts.itertuples():
    g = df[df["id"] == r.id]
    t, u = path_locf(g["month"], g["utility"], r.t_death)
    q2.append(trapezoid(u, t) / 12)
pts["qaly_locf"] = q2
print(pts.groupby("arm")[["qaly", "qaly_locf"]].mean().round(3))
for f in ["qaly_locf ~ A", "qaly_locf ~ A + base"]:
    m = smf.ols(f, data=pts).fit()
    lo, hi = m.conf_int().loc["A"]
    print(f, " 차이", round(m.params["A"], 3),
          " 95% CI", round(lo, 3), round(hi, 3))
''', title="과제 2 정답. 마지막 측정값을 사망 직전까지 유지")
save("lab22_hw2", c, marks={"환자 다: 0.868": 1, "1.230": 2, "1.118": 3, "차이 0.112": 4, "차이 0.089": 5})

c = nb.cell('''
# 동료의 코드: 효용 대신 EQ VAS ÷ 100
q_vas = []
for r in pts.itertuples():
    g = df[df["id"] == r.id]
    t, u = path(g["month"], g["vas"] / 100, r.t_death)
    q_vas.append(trapezoid(u, t) / 12)
pts["qaly_vas"] = q_vas
m = smf.ols("qaly_vas ~ A", data=pts).fit()
lo, hi = m.conf_int().loc["A"]
print("VAS/100 QALY:", pts.groupby("arm")["qaly_vas"].mean().round(3).to_dict(),
      " 차이", round(m.params["A"], 3), " 95% CI", round(lo, 3), round(hi, 3))

p_vas = dict(p, u_pf=0.73, u_pd=0.64)          # 셀 9의 상태별 VAS/100 평균
for name, pp in [("효용", p), ("VAS/100", p_vas)]:
    r = model(pp)
    dq = r["A"]["qaly"] - r["B"]["qaly"]
    print(name, " 증분 QALY", round(dq, 3),
          " ICER", round((r["A"]["cost"] - r["B"]["cost"]) / dq))
''', title="과제 3 정답. EQ VAS ÷ 100으로 계산한 동료의 QALY")
save("lab22_hw3", c, marks={"{'A': 1.153, 'B': 1.052}": 1, "차이 0.101": 2, "-0.004 0.207": 3,
                           "VAS/100  증분 QALY 0.481": 4, "ICER 5974": 5})

print("lab22: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- 대조 블록: 22장의 숫자(gen/_ch22_nums.json)와 lib_p4에 맞추기
import lib_p4 as L  # noqa: E402  (읽기만 한다. 학생 코드는 이 파일을 쓰지 않는다)

ns = nb.ns
N = json.load(open(os.path.join(HERE, "_ch22_nums.json"), encoding="utf-8"))
T, M, EQ = N["trial"], N["model"], N["eq"]
CHECKS = []


def chk(name, got, want, tol=1e-9, grp="ch22"):
    """grp: "ch22" = 22장 본문의 숫자(_ch22_nums.json), "model" = lib_p4의 공통 예시 모형"""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    err = float(np.max(np.abs(got - want)))
    CHECKS.append((name, err, grp))
    assert err < tol, f"{name}: lab {got} vs reference {want} (err {err:g})"


df_, pts_ = ns["df"], ns["pts"]
# a 절: 자료의 크기, 상태별 응답, 사건, 기저 시점(22장 a 절 4단계 Table 1)
assert len(df_) == T["n_obs"] == 1593 and len(pts_) == 320
assert (int((df_["state"] == "PF").sum()), int((df_["state"] == "PD").sum())) == (T["state"]["n_pf"], T["state"]["n_pd"])
assert {a: int(((pts_["arm"] == a) & (pts_["t_death"] <= 24)).sum()) for a in "AB"} == {a: T["events"][a]["dead24"] for a in "AB"}
b0_ = ns["b0"]
for a in "AB":
    x = T["base"]["arm" + a]
    got = [int(((b0_["arm"] == a) & (b0_[d] > 1)).sum()) for d in ns["dims"]]
    assert got == [x["n_prob"][k] for k in ("MO", "SC", "UA", "PD", "AD")], (a, got)
    chk(f"baseline arm {a}: utility mean, SD, VAS mean, SD",
        [b0_.loc[b0_["arm"] == a, "utility"].mean(), b0_.loc[b0_["arm"] == a, "utility"].std(),
         b0_.loc[b0_["arm"] == a, "vas"].mean(), b0_.loc[b0_["arm"] == a, "vas"].std()],
        [T["base"][a], T["base"]["sd" + a], x["vas"], x["vas_sd"]], 1e-12)
# b 절: 가치 세트(22장 b 절과 P4_FACTS.md 5.3, 5.4의 계산 예), 상태별 효용
kr5l_, eq3l_, KR3_, UK3_ = ns["kr5l"], ns["eq3l"], ns["KR3"], ns["UK3"]
chk("KR 5L: 21222, 11121, 11211, 31241, 55555, 11111", [kr5l_(x) for x in ("21222", "11121", "11211", "31241", "55555", "11111")],
    [EQ[k] for k in ("kr5_21222", "kr5_11121", "kr5_11211", "kr5_31241", "kr5_55555", "kr5_11111")], 1e-12)
assert kr5l_("31342") == 0.518 and kr5l_("12233") == 0.752 and kr5l_("42444") == 0.293
chk("3L KR, UK (Table 22-3) and KR 21232", [eq3l_(x, KR3_) for x in ("11121", "21222", "33333", "21232")] + [eq3l_(x, UK3_) for x in ("11121", "21222", "33333")],
    [EQ[k] for k in ("kr3_11121", "kr3_21222", "kr3_33333", "kr3_21232", "uk3_11121", "uk3_21222", "uk3_33333")], 1e-12)
assert eq3l_("32322", KR3_) == 0.148                      # Lee 2009 본문의 계산 예
assert eq3l_("11112", UK3_) == 0.848                      # gen/nums_ch22.py가 확인한 영국 3L 값
assert float((df_["u_calc"] - df_["utility"]).abs().max()) == 0.0
s_, S = ns["st"], T["state"]
# 22장은 진행 상태의 구간을 평균 ± 1.96 × SE로, 이 실습은 conf_int()(정규분포 1.959964…)로 구해 1e-6쯤 다르다
chk("state means, cluster SE, 95% CI", [s_.loc["PF", "mean"], s_.loc["PD", "mean"], s_.loc["PF", "se_cluster"], s_.loc["PD", "se_cluster"],
                                        s_.loc["PF", "lo"], s_.loc["PF", "hi"], s_.loc["PD", "lo"], s_.loc["PD", "hi"]],
    [S["pf"], S["pd"], S["pf_se"], S["pd_se"], S["pf_lo"], S["pf_hi"], S["pd_lo"], S["pd_hi"]], 1e-5)
assert [round(s_.loc[k, c], 3) for k in ("PF", "PD") for c in ("mean", "lo", "hi")] == [0.780, 0.765, 0.794, 0.621, 0.595, 0.647]
assert int((df_["utility"] < 0).sum()) == S["neg"] == 2 and abs(df_["utility"].min() - S["min"]) < 1e-12
by_ = ns["by"]
assert (round(by_.loc["PF", "vas_100"], 4), round(by_.loc["PD", "vas_100"], 4)) == (0.7324, 0.6362)   # calc22_new.py 1번
assert (round(by_.loc["PF", "vas_100"], 2), round(by_.loc["PD", "vas_100"], 2)) == (0.73, 0.64)        # 22장 a 절 3단계
# c 절: 환자 다, 24개월 QALY와 생존연수, 회귀, 생존자만의 평균, 둘째 해 할인
t_, u_ = ns["path"]([0, 3, 6, 12], ns["u_c"], 15)
assert list(u_[:4]) == [round(x, 3) for x in N["ex"]["다"]["u"]]
chk("patient 다: QALY (Table 22-4)", np.trapezoid(u_, t_) / 12, N["ex"]["다"]["qaly"], 1e-12)
Q = T["qaly"]
# 공개 자료는 사망 시점을 소수 여섯째 자리로 저장했으므로 22장 값과 1e-7 안쪽으로 다르다
chk("24-month QALY, LY by arm", [pts_.loc[pts_["arm"] == a, k].mean() for k in ("qaly", "ly") for a in "AB"],
    [Q["A"], Q["B"], Q["lyA"], Q["lyB"]], 1e-6)
chk("24-month QALY SD by arm", [pts_.loc[pts_["arm"] == a, "qaly"].std() for a in "AB"], [Q["sdA"], Q["sdB"]], 1e-6)
m0_, m1_ = ns["m0"], ns["m1"]
for tag, m_, ref in (("unadjusted", m0_, Q["unadj"]), ("adjusted", m1_, Q["adj"])):
    lo_, hi_ = m_.conf_int().loc["A"]
    chk(f"QALY difference {tag}: diff, SE, CI", [m_.params["A"], m_.bse["A"], lo_, hi_], [ref["diff"], ref["se"], ref["lo"], ref["hi"]], 1e-6)
chk("baseline utility coefficient", m1_.params["base"], Q["b_base"], 1e-6)
am_ = ns["alive_mean"]
chk("survivors-only means by visit", [am_.loc[v, a] for a in "AB" for v in (0, 3, 6, 12, 18, 24)],
    [T["visit"][f"{a}{v}"]["mean_alive"] for a in "AB" for v in (0, 3, 6, 12, 18, 24)], 1e-12)
w_ = ns["wrong"]
chk("survivors-only area", w_, [np.trapezoid([T["visit"][f"{a}{v}"]["mean_alive"] for v in (0, 3, 6, 12, 18, 24)], [0, 3, 6, 12, 18, 24]) / 12 for a in "AB"], 1e-12)
assert (round(w_[0], 3), round(w_[1], 3), round(w_[0] - w_[1], 3)) == (1.496, 1.411, 0.084)
for a in "AB":
    ids_ = pts_.loc[pts_["arm"] == a, "id"]
    curve_ = np.mean([np.interp(ns["grid"], *ns["paths"][i]) for i in ids_], axis=0)
    chk(f"mean-curve area = mean QALY, arm {a}", np.trapezoid(curve_, ns["grid"]) / 12, pts_.loc[pts_["arm"] == a, "qaly"].mean(), 1e-12)
chk("second-year discounting by arm", [pts_.loc[pts_["arm"] == a, "qaly_d"].mean() for a in "AB"], [Q["dA"], Q["dB"]], 1e-6)
# c 절: 모형(lib_p4.run, 22장 표 22-5)과 할인 시점(22장 c 절 3단계와 '더 알아보기'의 표)
P0, REF = L.base_params(), L.run()
p_, res_ = ns["p"], ns["res"]
chk("model: QALY, cost A, B (lib_p4.run)", [res_[a][k] for a in "AB" for k in ("qaly", "cost")], [REF[a][k] for a in "AB" for k in ("qaly", "cost")], 1e-9, "model")
for a in "AB":
    chk(f"model decomposition arm {a} (Table 22-5)", [res_[a]["pf_years"], res_[a]["pd_years"], p_["u_pf"] * res_[a]["pf_years"], p_["u_pd"] * res_[a]["pd_years"]],
        [M[a]["lyd_pf"], M[a]["lyd_pd"], M[a]["q_pf"], M[a]["q_pd"]], 1e-9)
    chk(f"model life-years arm {a} = lib_p4 ly_d", res_[a]["pf_years"] + res_[a]["pd_years"], REF[a]["ly_d"], 1e-9, "model")
assert [round(x, 3) for x in ns["tab"]["diff"]] == [0.623, 0.047, 0.670, 0.486, 0.029, -0.004, 0.511]


def ref_when(when, p=P0, r=None):
    """lib_p4.psm의 상태 비율(반주기 보정 전)로 다시 계산한 할인 시점별 결과(calc22_new.py의 run_disc와 같은 계산)"""
    r = p["disc"] if r is None else r
    out = {}
    for arm in L.ARMS:
        tr = L.psm(p, arm)["trace"]
        t = tr["t"]
        pf = (tr["pf"][:-1] + tr["pf"][1:]) / 2
        pd_ = (tr["pd"][:-1] + tr["pd"][1:]) / 2
        s_os = 1 - tr["dead"]
        yrs = {"mid": (t[:-1] + 0.5) / 12, "start": t[:-1] / 12, "end": t[1:] / 12, "annual": np.floor(t[:-1] / 12), "wrong": t[:-1] + 0.5}[when]
        d = 1 / (1 + r) ** yrs
        q = ((pf * p["u_pf"] + pd_ * p["u_pd"]) / 12 * d).sum() - p["du_ae_" + arm]
        c = ((pf * (p["c_drug_" + arm] + p["c_pf"]) + pd_ * p["c_pd"] + (s_os[:-1] - s_os[1:]) * p["c_death"]) * d).sum() + p["c_ae_" + arm]
        out[arm] = (q, c)
    dq, dc = out["A"][0] - out["B"][0], out["A"][1] - out["B"][1]
    return [out["A"][0], out["B"][0], dq, dc / dq]


wt_ = ns["when_tab"]
for when in ("mid", "start", "end", "annual", "wrong"):
    chk(f"discount timing {when}", wt_.loc[when].to_numpy(), ref_when(when), 1e-8, "model")
assert [(round(a, 3), round(b, 3), round(c, 3), round(d)) for a, b, c, d in wt_.to_numpy()] == \
    [(2.385, 1.874, 0.511, 5621), (2.389, 1.877, 0.512, 5620), (2.380, 1.870, 0.510, 5621), (2.435, 1.913, 0.523, 5612), (0.969, 0.870, 0.099, 9140)]
# 과제: 마지막 값 유지(22장 c 절 3단계의 환자 다 0.868, calc22_new.py 8번), VAS ÷ 100(22장 a 절 '더 알아보기', calc22_new.py 1번)
t_, u_ = ns["path_locf"]([0, 3, 6, 12], ns["u_c"], 15)
chk("patient 다 with the last value carried to death", np.trapezoid(u_, t_) / 12, N["ex"]["다"]["qaly_locf"], 1e-12)
ql_ = pts_.groupby("arm")["qaly_locf"].mean()
import statsmodels.formula.api as _smf  # noqa: E402
_r0, _r1 = _smf.ols("qaly_locf ~ A", data=pts_).fit(), _smf.ols("qaly_locf ~ A + base", data=pts_).fit()
assert (round(ql_["A"], 4), round(ql_["B"], 4), round(_r0.params["A"], 4), round(_r1.params["A"], 4)) == (1.2303, 1.1182, 0.1120, 0.0889)
assert [round(x, 3) for x in list(_r0.conf_int().loc["A"]) + list(_r1.conf_int().loc["A"])] == [0.001, 0.223, -0.017, 0.195]
qv_ = pts_.groupby("arm")["qaly_vas"].mean()
_rv = _smf.ols("qaly_vas ~ A", data=pts_).fit()
chk("VAS/100 QALY A, B, diff, CI (calc22_new.py)", [qv_["A"], qv_["B"], _rv.params["A"], *_rv.conf_int().loc["A"]],
    [1.152945, 1.051535, 0.101410, -0.004279, 0.207100], 1e-5)
RV = L.run(dict(P0, u_pf=0.73, u_pd=0.64))
_m = ns["model"](ns["p_vas"])
chk("VAS/100 in the model: d_qaly, ICER (lib_p4.run)", [_m["A"]["qaly"] - _m["B"]["qaly"], (_m["A"]["cost"] - _m["B"]["cost"]) / (_m["A"]["qaly"] - _m["B"]["qaly"])],
    [RV["d_qaly"], RV["icer"]], 1e-8, "model")
assert round(RV["icer"]) == 5974 and round(RV["d_qaly"], 3) == 0.481
assert round(S["full_pf"], 3) == 0.147
print(f"대조: {len(CHECKS)}개 항목 모두 허용 오차 안 (22장 숫자와 비교한 {sum(g == 'ch22' for _, _, g in CHECKS)}개의 가장 큰 차이 "
      f"{max(e for _, e, g in CHECKS if g == 'ch22'):.2e}, lib_p4와 비교한 {sum(g == 'model' for _, _, g in CHECKS)}개의 가장 큰 차이 "
      f"{max(e for _, e, g in CHECKS if g == 'model'):.2e})")

# ---------------------------------------------------------------- 본문 문장에 쓴 그 밖의 숫자 확인
if os.environ.get("VERIFY") or os.environ.get("NOMARK"):
    print("VERIFY baseline pain/discomfort %:", [round(100 * T["base"]["arm" + a]["n_prob"]["PD"] / 160, 1) for a in "AB"])
    print("VERIFY baseline utility gap:", round(T["base"]["A"] - T["base"]["B"], 4))
    print("VERIFY SE ratio cluster/naive PF, PD:", round(s_.loc["PF", "se_cluster"] / s_.loc["PF", "se_naive"], 2),
          round(s_.loc["PD", "se_cluster"] / s_.loc["PD", "se_naive"], 2))
    print("VERIFY VAS gap / utility gap:", round((by_.loc["PF", "vas_100"] - by_.loc["PD", "vas_100"]) / (by_.loc["PF", "utility"] - by_.loc["PD", "utility"]), 2))
    print("VERIFY utilities strictly between 0.883 and 1:", int(((df_["utility"] > 0.883) & (df_["utility"] < 1)).sum()))
    print("VERIFY LY difference:", round(Q["lyA"] - Q["lyB"], 4), " QALY difference:", round(Q["A"] - Q["B"], 4))
    print("VERIFY coefficient x 0.1:", round(0.1 * m1_.params["base"], 3), " unadj - adj:", round(m0_.params["A"] - m1_.params["A"], 4))
    print("VERIFY survivors-only inflation A, B:", round(w_[0] - Q["A"], 3), round(w_[1] - Q["B"], 3))
    yr_ = ns["yr"]
    print("VERIFY year-1, year-2 QALY means by arm:", {a: [round(yr_[(pts_["arm"] == a).to_numpy(), j].mean(), 3) for j in (0, 1)] for a in "AB"})
    print("VERIFY wrong / base ICER:", round(wt_.loc["wrong", "icer"] / wt_.loc["mid", "icer"], 2))
    print("VERIFY LOCF increase A, B:", round(ql_["A"] - Q["A"], 3), round(ql_["B"] - Q["B"], 3), " diff shrink:", round((Q["A"] - Q["B"]) - (ql_["A"] - ql_["B"]), 4))
    print("VERIFY PF responses, patients:", int((df_["state"] == "PF").sum()), df_.loc[df_["state"] == "PF", "id"].nunique())
    print("VERIFY model decomposition share of d_qaly from PF:", round(M["d"]["q_pf"] / M["d"]["qaly"], 2))
    print("VERIFY responses by month:", df_.groupby("month").size().to_dict())
    print("VERIFY responses per patient (min, max):", df_.groupby("id").size().min(), df_.groupby("id").size().max())
    print("VERIFY patient 다 LOCF gain (0.581 x 3 / 2 / 12):", round(0.581 * 3 / 2 / 12, 3), round(N["ex"]["다"]["qaly_locf"] - N["ex"]["다"]["qaly"], 3))
    print("VERIFY 1 - 1/1.045:", round(1 - 1 / 1.045, 3), " year-2 QALY mean, all patients:", round(yr_[:, 1].mean(), 3))

# ---------------------------------------------------------------- 노트북 (pub/notebooks/lab22.ipynb)
notes = {
    1: ("## a. 실습 데이터 준비\n\n22장의 가상 임상시험(진행성 신세포암, 신약 A 대 표준요법 B, 군당 160명)의 EQ-5D-5L 응답과 "
        "환자별 진행·사망 시점을 읽고 확인합니다. 셀 1은 출력을 만든 패키지의 버전을 찍습니다. 내 화면의 출력이 사이트와 다를 때 먼저 비교합니다."),
    5: ("## b. 효용 점수 구하기\n\n한국 EQ-5D-5L 가치 세트를 함수로 짜서 응답 코드를 효용으로 바꾸고, 3L 가치 세트 두 개(한국, 영국)를 비교한 뒤, "
        "상태별 평균 효용을 환자 단위로 묶은 신뢰구간과 함께 구합니다."),
    10: ("## c. QALY와 할인\n\n사다리꼴 공식으로 환자별 24개월 QALY를 구해 두 군을 비교하고(기저 효용 보정 전후), 생존자만의 평균으로 계산하면 "
         "얼마나 틀리는지 봅니다. 끝에서는 공통 예시 모형의 20년 QALY를 상태별로 나누고 할인 시점을 바꿔 봅니다."),
    19: ("## 과제 정답\n\n**과제 1.** 셀 8의 무진행 행(평균 0.780, 95% CI 0.765~0.794, 응답 1,184건)을 한 문장으로 옮기고, "
         "이 숫자가 뜻하지 않는 것을 두 가지 이상 씁니다."),
    20: ("**과제 2.** 마지막 측정값이 사망 직전까지 유지되고 사망 시점에 0이 된다고 보는 규칙으로 바꿔 환자 다의 QALY, "
         "두 군의 평균 QALY, 기저 효용 보정 전후의 차이를 다시 구하고 왜 달라지는지 설명합니다."),
    21: ("**과제 3.** 효용 대신 EQ VAS ÷ 100으로 QALY를 계산하고 모형에도 상태별 VAS ÷ 100 평균을 넣은 동료의 분석을 재현하고, "
         "틀린 곳 두 가지를 찾습니다."),
}
path = nb.save_ipynb(
    "실습 22. 효용과 QALY 계산",
    intro=("사회약학 연구방법 노트의 '실습 22 효용과 QALY 계산'에 나오는 셀을 차례로 모은 노트북입니다. "
           "22장의 가상 임상시험 자료(환자 320명, EQ-5D-5L 응답 1,593건)로 한국 가치 세트의 효용, 상태별 효용(0.780과 0.621), "
           "환자별 24개월 QALY와 기저 효용 보정(0.117에서 0.094)을 계산하고, 20–25장 공통 예시 모형의 20년 QALY(증분 0.511)를 "
           "상태별로 나눈 뒤 할인 시점을 바꿔 봅니다. 셀을 위에서부터 차례로 실행하세요. 자료는 사이트에서 바로 읽으므로 "
           "인터넷 연결이 필요합니다. 설명과 출력 읽는 법은 사이트의 실습 22 쪽에 있습니다."),
    notes=notes)
print("notebook:", path, len(nb.cells), "cells")
