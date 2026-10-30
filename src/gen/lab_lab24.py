"""실습 24 · 민감도 분석 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab24.py
      then  python3 tools/nbcheck.py lab24   (runs the notebook alone in a fresh process and compares outputs)
      (NOMARK=1 을 앞에 붙이면 표식 없이 셀 출력만 화면에 찍는다. VERIFY=1 이면 본문 문장에 쓴 그 밖의 숫자도 찍는다)

자료 파일은 없다. 입력값 18개와 분포를 노트북에 직접 적는다.
학생 코드는 gen/lib_p4.py를 쓰지 않는다. 셀 2의 입력값 사전과 셀 3의 run_model()은 gen/lab_lab23.py의 셀 8, 15와 같은 코드이고,
셀 11의 arm_result(), two_arms()는 lab_lab23.py의 셀 19와 같은 코드이다(2026-10-15 실습 23 개편 뒤 번호).
끝의 대조 블록이 실습 결과를 lib_p4의 oneway(), threshold_price(), scenario_table(), run_waning(), psa(n=5000, seed=20261002),
psa_inputs(), ceac(), evpi(), twoway()와 gen/_ch24_nums.json(24장 본문의 숫자), 24장 본문에 적힌 숫자에 맞춘다.
(2026-10-15: 24장을 새 틀로 고친 뒤 '실습 24에 더할 것'을 반영했다. 버전 셀 1, 이원 민감도 분석 셀 10(옛 과제 2),
 치료 효과 감소·소실을 넣은 모형과 시나리오 분석 셀 11–12, 모의실험 횟수 셀 19(옛 과제 1), 같은 5,000번을 효과 감소 구조로
 돌리는 셀 21, 수용곡선과 'ICER가 임계값 이하인 비율' 셀 25, 50% 지점·평균끼리 나눈 ICER·모의실험별 ICER 셀 26을 더했다.
 과제는 ① 신약 C의 출력 해석 ② 약값을 바꾼 수용곡선(옛 과제 3) ③ 표준편차를 표준오차 자리에 넣은 분석으로 바꿨다.
 셀 번호가 모두 바뀌었다. 5,000번 셀은 걸린 초 대신 "10초 안에 끝났는가"를 찍는다(nbcheck가 셀 출력을 그대로 비교하므로).)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab24")


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


# ---------------------------------------------------------------- a. 실습 준비
c = nb.cell('''
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib, scipy

print("numpy     ", np.__version__)
print("pandas    ", pd.__version__)
print("matplotlib", matplotlib.__version__)
print("scipy     ", scipy.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab24_versions", c, marks={"numpy      2.5.3": 1, "pandas     3.0.5": 2, "scipy      1.18.1": 3})

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
''', title="입력값 사전")
save("lab24_setup", c, marks={"18 개 입력값": 1, "0.07777": 2})

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
threshold = 5000                       # 이 예시에서 가정한 임계값
print("증분비용  :", round(base["d_cost"], 1))
print("증분 QALY :", round(base["d_qaly"], 4))
print("ICER      :", round(base["icer"]), "만원/QALY")
nmb = threshold * base["d_qaly"] - base["d_cost"]
print("증분 순금전편익:", round(nmb, 1), "만원")
''', title="모형 함수와 기준 분석")
save("lab24_model", c, marks={"2872.3": 1, "0.511": 2, "5621": 3, "-317.1": 4})

c = nb.cell('''
def get_input(p, key):
    """입력값 표의 표현으로 읽기. pfs_med, os_med는 중앙값(개월)"""
    if key.endswith("_med"):
        c = key[:-4]                   # "pfs" 또는 "os"
        return (np.log(2) / p[c + "_lam"]) ** (1 / p[c + "_gam"])
    return p[key]

def set_input(p, key, value):
    """p의 사본에서 key 하나를 value로 바꾼다 (원래 p는 그대로)"""
    q = dict(p)
    if key.endswith("_med"):           # 중앙값을 바꾼다. 모양은 그대로
        c = key[:-4]
        q[c + "_lam"] = weib_lam(value, q[c + "_gam"])
    elif key.endswith("_gam"):         # 모양을 바꾼다. 중앙값은 그대로
        c = key[:-4]
        med = get_input(p, c + "_med")
        q[key] = value
        q[c + "_lam"] = weib_lam(med, value)
    else:                              # 그 밖의 입력값
        q[key] = value
    return q

print("중앙값:", round(get_input(p, "pfs_med"), 4),
      round(get_input(p, "os_med"), 4))
q1 = set_input(p, "os_gam", 1.29)      # 중앙값을 지키는 방법
q2 = dict(p, os_gam=1.29)              # 사전의 값만 바꾼 것
for name, q in [("set_input", q1), ("dict     ", q2)]:
    s60 = np.exp(-q["os_lam"] * 60 ** q["os_gam"])
    print(name, " os_lam", round(q["os_lam"], 5),
          " 중앙값", round(get_input(q, "os_med"), 1),
          " 60개월 생존율", round(s60, 3),
          " ICER", round(run_model(q)["icer"]))
''', title="곡선을 중앙값과 모양 모수로 읽고 바꾸는 함수")
save("lab24_setinput", c, marks={"중앙값: 10.0 28.0": 1, "0.00942": 2, "0.157": 3, "중앙값 19.5": 4, "6021": 5})

# ---------------------------------------------------------------- b. 일원 민감도 분석과 토네이도 그림
c = nb.cell('''
spec = {   # 이름: (분포, 불확실성의 크기)
    "hr_pfs": ("lognormal", 0.113), "hr_os": ("lognormal", 0.131),
    "pfs_med": ("lognormal", 0.07), "pfs_gam": ("lognormal", 0.05),
    "os_med": ("lognormal", 0.08), "os_gam": ("lognormal", 0.06),
    "u_pf": ("beta", 0.03), "u_pd": ("beta", 0.05),
    "du_ae_A": ("gamma", 0.25), "du_ae_B": ("gamma", 0.25),
    "c_pf": ("gamma", 0.20), "c_pd": ("gamma", 0.20),
    "c_death": ("gamma", 0.20),
    "c_ae_A": ("gamma", 0.25), "c_ae_B": ("gamma", 0.25),
}

def beta_ab(m, se):
    """평균, 표준오차 -> 베타분포의 alpha, beta"""
    n = m * (1 - m) / se ** 2 - 1
    return m * n, (1 - m) * n

def gamma_ks(m, se):
    """평균, 표준오차 -> 감마분포의 모양, 척도"""
    shape = (m / se) ** 2
    return shape, m / shape

print(len(spec), "개 입력값에 분포")
print("무진행 효용 0.78 (SE 0.03):", np.round(beta_ab(0.78, 0.03), 2))
print("진행 상태 비용 250 (SE 50):", gamma_ks(250, 250 * 0.20))
''', title="입력값의 분포와 적률법")
save("lab24_dist", c, marks={"15 개 입력값에 분포": 1, "[147.94  41.73]": 2, "(25.0, 10.0)": 3})

c = nb.cell('''
z = stats.norm.ppf(0.975)              # 1.96

def range95(p, key):
    """분포의 2.5, 97.5 백분위수"""
    dist, u = spec[key]
    m = get_input(p, key)
    if dist == "lognormal":            # u = 로그 척도의 표준오차
        return np.exp(np.log(m) - z * u), np.exp(np.log(m) + z * u)
    if dist == "beta":                 # u = 표준오차
        a, b = beta_ab(m, u)
        return tuple(stats.beta.ppf([0.025, 0.975], a, b))
    k, s = gamma_ks(m, m * u)          # u = 표준오차 / 평균
    return tuple(stats.gamma.ppf([0.025, 0.975], k, scale=s))

rows = []
for key in spec:
    low, high = range95(p, key)
    rows.append([key, get_input(p, key), low, high, "95% 구간"])
a, b = p["c_drug_A"], p["c_drug_B"]
rows.append(["c_drug_A", a, a * (1 - 0.20), a, "20% 인하까지"])
rows.append(["c_drug_B", b, b * (1 - 0.20), b * (1 + 0.20),
             "±20% (임의)"])
ranges = pd.DataFrame(rows, columns=["key", "base", "low", "high",
                                     "basis"]).set_index("key")
ranges.round(3)
''', title="입력값별 범위 표", max_rows=20)
save("lab24_ranges", c, dfmarks={"0.521": 1, "8.718": 2, "0.719": 3, "161.787": 4, "152.000": 5, "96.000": 6})

c = nb.cell('''
rows = []
for key in ranges.index:
    low, high = ranges.loc[key, "low"], ranges.loc[key, "high"]
    r_low = run_model(set_input(p, key, low))     # 낮은 값으로
    r_high = run_model(set_input(p, key, high))   # 높은 값으로
    rows.append([key, r_low["icer"], r_high["icer"]])
ow = pd.DataFrame(rows, columns=["key", "icer_low",
                                 "icer_high"]).set_index("key")
ow["swing"] = (ow["icer_high"] - ow["icer_low"]).abs()
lowest = ow[["icer_low", "icer_high"]].min(axis=1)
ow["below"] = lowest < threshold       # 임계값 아래로 내려가는가
ow = ow.sort_values("swing", ascending=False)
print("기준 분석 ICER:", round(base["icer"]))
ow.round(0)
''', title="입력값을 하나씩 양 끝으로 바꿔 ICER 계산", max_rows=20)
save("lab24_oneway", c, marks={"기준 분석 ICER: 5621": 1},
     dfmarks={"5232.0": 2, "7922.0": 3, "2690.0": 4, "True": 5, "4006.0": 6, "4952.0": 7, "6076.0": 8, "5600.0": 9})

c = nb.cell('''
labels = {
    "hr_os": "HR, overall survival",
    "hr_pfs": "HR, progression-free survival",
    "os_med": "Median OS, therapy B",
    "os_gam": "Weibull shape, OS",
    "pfs_med": "Median PFS, therapy B",
    "pfs_gam": "Weibull shape, PFS",
    "u_pf": "Utility, progression-free",
    "u_pd": "Utility, progressed",
    "du_ae_A": "AE QALY loss, drug A",
    "du_ae_B": "AE QALY loss, therapy B",
    "c_drug_A": "Monthly price, drug A",
    "c_drug_B": "Monthly price, therapy B",
    "c_pf": "Monthly cost, progression-free",
    "c_pd": "Monthly cost, progressed",
    "c_death": "End-of-life cost",
    "c_ae_A": "AE cost, drug A",
    "c_ae_B": "AE cost, therapy B",
}
b0 = base["icer"]
y = np.arange(len(ow))                 # 0, 1, ..., 16
fig, ax = plt.subplots(figsize=(8, 5.6))
ax.barh(y, ow["icer_low"] - b0, left=b0, color="tab:blue",
        label="Lower value of input")
ax.barh(y, ow["icer_high"] - b0, left=b0, color="tab:orange",
        label="Upper value of input")
ax.axvline(b0, color="black", lw=1)
ax.axvline(threshold, color="gray", ls="--")
ax.set_yticks(y)
ax.set_yticklabels([labels[k] for k in ow.index])
ax.invert_yaxis()                      # 폭이 큰 것을 위에
ax.set_xlabel("ICER (10,000 KRW per QALY)")
ax.legend(loc="lower right")
plt.tight_layout()
''', title="토네이도 그림")
save("lab24_tornado", c)

c = nb.cell('''
prices = [152.0, 190.0]
icers = [run_model(dict(p, c_drug_A=x))["icer"] for x in prices]
print("ICER:", np.round(icers, 1))
price_star = np.interp(threshold, icers, prices)
print("임계 가격:", round(price_star, 2), "만원,",
      "인하율:", round((1 - price_star / 190) * 100, 1), "%")
r = run_model(dict(p, c_drug_A=price_star))
print("그 가격에서의 ICER:", round(r["icer"], 4))
slope = (icers[1] - icers[0]) / (prices[1] - prices[0])
print("약값 1만원당 ICER 변화:", round(slope, 1))
''', title="ICER가 임계값과 같아지는 약값")
save("lab24_price", c, marks={"[4006.5 5620.6]": 1, "175.39": 2, "인하율: 7.7": 3, "5000.0": 4, "42.5": 5})

c = nb.cell('''
hr_low = ranges.loc["hr_pfs", "low"]     # 95% 구간의 양 끝
hr_high = ranges.loc["hr_pfs", "high"]
hrs = [hr_low, 0.65, hr_high]
rows = []
for price in [190.0, 175.0, 160.0]:
    row = []
    for hr in hrs:
        q = set_input(set_input(p, "c_drug_A", price), "hr_pfs", hr)
        row.append(run_model(q)["icer"])
    rows.append(row)
two = pd.DataFrame(rows, index=[190, 175, 160],
                   columns=np.round(hrs, 2))
two.round(0)
''', title="약값과 무진행생존 위험비를 함께 바꾸기 (이원 민감도 분석)")
save("lab24_twoway", c, dfmarks={"4784.0": 1, "5621.0": 2, "4983.0": 3, "5345.0": 4})

c = nb.cell('''
def arm_result(p, arm, pf_t, pd_t, half=True):
    """시점별 상태 비율(0-n개월의 n+1개) -> 한 군의 생존연수, 비용, QALY"""
    n = len(pf_t) - 1
    if half:                                  # 반주기 보정
        pf = (pf_t[:-1] + pf_t[1:]) / 2
        pd_ = (pd_t[:-1] + pd_t[1:]) / 2
    else:                                     # 주기 시작의 값으로 셈
        pf, pd_ = pf_t[:-1], pd_t[:-1]
    alive = pf_t + pd_t
    died = alive[:-1] - alive[1:]             # 주기별 사망
    disc = 1 / (1 + p["disc"]) ** ((np.arange(n) + 0.5) / 12)
    cost = ((pf * (p["c_drug_" + arm] + p["c_pf"]) + pd_ * p["c_pd"]
             + died * p["c_death"]) * disc).sum() + p["c_ae_" + arm]
    qaly = (((pf * p["u_pf"] + pd_ * p["u_pd"]) / 12 * disc).sum()
            - p["du_ae_" + arm])
    return {"ly_pf": pf.sum() / 12, "ly_pd": pd_.sum() / 12,
            "ly": (pf + pd_).sum() / 12, "qaly": qaly, "cost": cost}

def two_arms(rA, rB):
    """두 군의 결과 -> 증분과 ICER"""
    dc, dq = rA["cost"] - rB["cost"], rA["qaly"] - rB["qaly"]
    return {"A": rA, "B": rB, "d_cost": dc, "d_qaly": dq, "icer": dc / dq}

def run_waning(p, start=36, end=60, horizon=240):
    """신약 A의 위험비가 start개월부터 줄어 end개월에 1이 되는 모형"""
    t = np.arange(0, horizon + 1)
    mid = t[:-1] + 0.5                        # 주기 중간 시점
    if end > start:                           # 0이면 효과 그대로, 1이면 없음
        w = np.clip((mid - start) / (end - start), 0, 1)
    else:                                     # start = end: 그 달부터 바로 1
        w = (mid >= start) * 1.0
    res = {}
    for arm in ["A", "B"]:
        s = {}
        for k in ["pfs", "os"]:
            dH = np.diff(p[k + "_lam"] * t ** p[k + "_gam"])  # B의 누적위험 증가
            hr = p["hr_" + k] + (1 - p["hr_" + k]) * w if arm == "A" else 1.0
            s[k] = np.exp(-np.r_[0, np.cumsum(hr * dH)])
        s_pfs = np.minimum(s["pfs"], s["os"])
        res[arm] = arm_result(p, arm, s_pfs, s["os"] - s_pfs)
    out = two_arms(res["A"], res["B"])
    out["hr_os_t"] = p["hr_os"] + (1 - p["hr_os"]) * w    # 달마다의 위험비
    return out

r0 = run_waning(p, start=999, end=1000)   # 효과가 줄지 않는 경우
print("효과가 그대로이면 ICER:", round(r0["icer"], 6),
      " run_model:", round(base["icer"], 6))
r1 = run_waning(p, 36, 60)
print("3년 뒤부터 감소: 48번째 달의 위험비", round(r1["hr_os_t"][47], 3),
      " ICER", round(r1["icer"]))
r2 = run_waning(p, 30, 30)
print("30개월 뒤 소실: 30, 31번째 달의 위험비", r2["hr_os_t"][29:31])
print("신약 A 생존연수: 효과 소실", round(r2["A"]["ly"], 3),
      " 기준 분석", round(base["A"]["ly"], 3))
''', title="치료 효과 감소와 소실을 넣은 모형 함수")
save("lab24_waning", c, marks={"run_model: 5620.610246": 1, "위험비 0.87": 2, "ICER 6150": 3, "[0.75 1.  ]": 4, "효과 소실 3.455": 5})

c = nb.cell('''
scen = {"base": base,
        "horizon 5y": run_model(p, horizon=60),
        "horizon 10y": run_model(p, horizon=120),
        "horizon 15y": run_model(p, horizon=180),
        "discount 0%": run_model(dict(p, disc=0.0)),
        "discount 3%": run_model(dict(p, disc=0.03)),
        "waning 36-60": run_waning(p, 36, 60),
        "waning 60-84": run_waning(p, 60, 84),
        "loss after 30": run_waning(p, 30, 30)}
rows = [[k, r["d_cost"], r["d_qaly"], r["icer"]] for k, r in scen.items()]
sc = pd.DataFrame(rows, columns=["scenario", "d_cost", "d_qaly",
                                 "icer"]).set_index("scenario")
sc["vs_base"] = (sc["icer"] / base["icer"] - 1) * 100   # %
sc.round({"d_cost": 0, "d_qaly": 3, "icer": 0, "vs_base": 1})
''', title="시나리오 분석")
save("lab24_scen", c, dfmarks={"6324.0": 1, "5431.0": 2, "6150.0": 3, "5815.0": 4, "0.282": 5, "22.3": 6})

# ---------------------------------------------------------------- c. 확률적 민감도 분석
c = nb.cell('''
rng0 = np.random.default_rng(1)        # 연습용 난수 (본 분석과 별개)
z1 = rng0.normal(size=10000)           # 표준정규 난수 1만 개
e = rng0.normal(size=10000)            # z1과 독립인 난수
rho = 0.5
z2 = rho * z1 + np.sqrt(1 - rho ** 2) * e

print("z1과 e의 상관 :", round(np.corrcoef(z1, e)[0, 1], 3))
print("z1과 z2의 상관:", round(np.corrcoef(z1, z2)[0, 1], 3))
print("z2의 평균, 표준편차:", round(z2.mean(), 3), round(z2.std(), 3))
print("z1 > 0일 때 z2 > 0인 비율:", round(np.mean(z2[z1 > 0] > 0), 3))

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
axes[0].scatter(z1[:2000], e[:2000], s=3, alpha=0.3)
axes[0].set_title("Independent (rho = 0)")
axes[1].scatter(z1[:2000], z2[:2000], s=3, alpha=0.3)
axes[1].set_title("Correlated (rho = 0.5)")
for ax in axes:
    ax.set_xlabel("z1")
axes[0].set_ylabel("Second draw")
plt.tight_layout()
''', title="상관 있는 두 정규 난수 만들기")
save("lab24_corr", c, marks={"0.033": 1, "0.524": 2, "1.006": 3, "0.67": 4})

c = nb.cell('''
pairs = [("pfs_med", "os_med"),        # 함께 뽑는 짝 (PFS, OS)
         ("pfs_gam", "os_gam"),
         ("hr_pfs", "hr_os")]

def draw_curves(rng, p, rho=0.5):
    """곡선과 위험비 한 세트. 짝마다 로그 척도에서 상관 rho"""
    v = {}
    for k1, k2 in pairs:
        z1, z2 = rng.normal(size=2)    # 독립인 표준정규 난수 둘
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2
        v[k1] = get_input(p, k1) * np.exp(spec[k1][1] * z1)
        v[k2] = get_input(p, k2) * np.exp(spec[k2][1] * z2)
    q = dict(p)
    for k in ["hr_pfs", "hr_os", "pfs_gam", "os_gam"]:
        q[k] = v[k]
    q["pfs_lam"] = weib_lam(v["pfs_med"], v["pfs_gam"])
    q["os_lam"] = weib_lam(v["os_med"], v["os_gam"])
    return q

def crossed(p, horizon=240):
    """어느 군에서든 PFS 곡선이 OS 곡선을 넘으면 True"""
    t = np.arange(0, horizon + 1)
    for hr_pfs, hr_os in [(p["hr_pfs"], p["hr_os"]), (1.0, 1.0)]:
        s_pfs = np.exp(-p["pfs_lam"] * hr_pfs * t ** p["pfs_gam"])
        s_os = np.exp(-p["os_lam"] * hr_os * t ** p["os_gam"])
        if (s_pfs > s_os).any():
            return True
    return False

rng = np.random.default_rng(20261002)
q = draw_curves(rng, p)
keys = [k for pair in pairs for k in pair]
print(pd.DataFrame({"base": [get_input(p, k) for k in keys],
                    "draw": [get_input(q, k) for k in keys]},
                   index=keys).round(3))
print("곡선이 엇갈리는가:", crossed(q))
''', title="곡선과 위험비 한 세트 뽑기")
save("lab24_curves", c, marks={"9.848": 1, "29.601": 2, "0.581": 3, "False": 4})

c = nb.cell('''
def draw_inputs(rng, p, count, rho=0.5, redraw=True):
    """입력값 한 세트를 분포에서 뽑는다 (약값과 할인율은 고정)"""
    while True:
        q = draw_curves(rng, p, rho)   # 1) 곡선과 위험비
        count["tries"] += 1
        if not crossed(q):
            break                      # 엇갈리지 않으면 채택
        count["crossed"] += 1
        if not redraw:
            break                      # 다시 뽑기를 끈 경우
    for key, (dist, u) in spec.items():    # 2) 나머지 입력값
        if dist == "beta":
            a, b = beta_ab(p[key], u)
            q[key] = rng.beta(a, b)
        elif dist == "gamma":
            k, s = gamma_ks(p[key], p[key] * u)
            q[key] = rng.gamma(k, s)
    return q                           # 로그정규는 1)에서 뽑았다

rng = np.random.default_rng(20261002)  # 같은 seed로 처음부터
count = {"tries": 0, "crossed": 0}
q = draw_inputs(rng, p, count)
show = ["u_pf", "u_pd", "c_pf", "c_pd", "c_death", "c_drug_A"]
print({k: round(float(q[k]), 3) for k in show})
r = run_model(q)
print("첫 반복: 증분비용", round(r["d_cost"]),
      " 증분 QALY", round(r["d_qaly"], 3), " ICER", round(r["icer"]))
print(count)
''', title="입력값 한 세트 뽑기와 다시 뽑기")
save("lab24_draw", c, marks={"'u_pf': 0.839": 1, "'c_drug_A': 190.0": 2, "2709": 3, "{'tries': 1, 'crossed': 0}": 4})

c = nb.cell('''
def run_psa(p, n=5000, seed=20261002, rho=0.5, redraw=True,
            model=run_model):
    """입력값 n번을 뽑아 모형을 n번 돌린다"""
    rng = np.random.default_rng(seed)
    count = {"tries": 0, "crossed": 0}
    dc, dq = np.empty(n), np.empty(n)  # 결과를 담을 빈 배열
    for i in range(n):
        r = model(draw_inputs(rng, p, count, rho, redraw))
        dc[i], dq[i] = r["d_cost"], r["d_qaly"]
    return dc, dq, count

t0 = time.time()
dc, dq, count = run_psa(p)
print("10초 안에 끝났는가:", time.time() - t0 < 10)
print("뽑은 횟수", count["tries"], " 버린 횟수", count["crossed"],
      " 비율", round(count["crossed"] / count["tries"], 4))
print(dc.shape, dq.shape)
print("처음 세 반복의 증분비용:", dc[:3].round(1))
print("처음 세 반복의 증분 QALY:", dq[:3].round(4))
''', title="5,000번 돌리기")
save("lab24_psa", c, marks={"10초 안에 끝났는가: True": 1, "뽑은 횟수 5137": 2, "버린 횟수 137": 3, "(5000,) (5000,)": 4,
                            "[2709.5 4450.7 4428.8]": 5})

c = nb.cell('''
inmb = threshold * dq - dc             # 반복마다의 증분 순금전편익
rows = []
for name, x in [("d_cost", dc), ("d_qaly", dq), ("inmb", inmb)]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    rows.append([name, x.mean(), lo, hi])
summ = pd.DataFrame(rows, columns=["result", "mean", "p2.5",
                                   "p97.5"]).set_index("result")
print(summ.round(3))
print("ICER (평균 / 평균):", round(dc.mean() / dq.mean()))
p_ce5 = np.mean(inmb > 0)
print("비용효과적일 확률:", p_ce5,
      " 몬테카를로 표준오차:",
      round(np.sqrt(p_ce5 * (1 - p_ce5) / len(dc)), 4))
''', title="결과 요약")
save("lab24_summary", c, marks={"2934.924": 1, "1060.857": 2, "0.525": 3, "-307.745": 4, "5586": 5, "0.1604": 6, "0.0052": 7})

c = nb.cell('''
fig, ax = plt.subplots(figsize=(6.5, 4.6))
ax.scatter(dq, dc, s=3, alpha=0.25, label="5,000 simulations")
x = np.array([-0.2, 1.6])
ax.plot(x, threshold * x, color="black", ls="--",
        label="Threshold: 5,000 per QALY")
ax.scatter(base["d_qaly"], base["d_cost"], color="red",
           marker="x", s=70, zorder=3, label="Base case")
ax.axhline(0, color="gray", lw=0.8)
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("Incremental QALYs")
ax.set_ylabel("Incremental cost (10,000 KRW)")
ax.legend(loc="upper left")
plt.tight_layout()

quad = {"NE": (dq > 0) & (dc > 0), "SE": (dq > 0) & (dc <= 0),
        "NW": (dq <= 0) & (dc > 0), "SW": (dq <= 0) & (dc <= 0)}
print({k: int(v.sum()) for k, v in quad.items()})
print("임계값 선 아래의 점:", int((inmb > 0).sum()))
''', title="비용효과평면과 사분면")
save("lab24_plane", c, marks={"'NE': 4958": 1, "'NW': 37": 2, "802": 3})

c = nb.cell('''
ok = (5000 * dq - dc) > 0              # 반복마다 비용효과적인가
for n in [500, 1000, 5000]:
    pr = ok[:n].mean()                 # 처음 n번만 쓴 확률
    se = np.sqrt(pr * (1 - pr) / n)    # 몬테카를로 표준오차
    print(n, " 확률", round(pr, 4), " 표준오차", round(se, 4),
          " 범위", round(pr - 1.96 * se, 3), "-",
          round(pr + 1.96 * se, 3))

for seed in [1, 2, 3]:
    c2, q2, n2 = run_psa(p, seed=seed)
    print("seed", seed, " 확률", np.mean(5000 * q2 - c2 > 0))
''', title="모의실험 횟수와 몬테카를로 오차")
save("lab24_mcse", c, marks={"0.142": 1, "0.0052": 2, "0.1544": 3})

c = nb.cell('''
rows = []
for rho, redraw in [(0.5, True), (0.0, True),
                    (0.5, False), (0.0, False)]:
    c2, q2, n2 = run_psa(p, rho=rho, redraw=redraw)
    rows.append([rho, redraw, n2["tries"], n2["crossed"],
                 n2["crossed"] / n2["tries"],
                 np.mean(threshold * q2 - c2 > 0)])
pd.DataFrame(rows, columns=["rho", "redraw", "tries", "crossed",
                            "share", "p_ce"]).round(4)
''', title="상관과 다시 뽑기를 바꿔 보기")
save("lab24_vary", c, dfmarks={"5137": 1, "5503": 2, "0.0914": 3, "141": 4, "427": 5, "0.1702": 6})

c = nb.cell('''
dc_w, dq_w, n_w = run_psa(p, model=run_waning)   # 같은 seed, 같은 반복
print("같은 반복을 뽑았는가:", n_w == count)
print("효과 감소 구조: 평균 증분비용", round(dc_w.mean()),
      " 평균 증분 QALY", round(dq_w.mean(), 3))
for lam in [5000, 6000]:
    print("임계값", lam, " 기준 구조", np.mean(lam * dq - dc > 0),
          " 효과 감소", np.mean(lam * dq_w - dc_w > 0))
''', title="같은 5,000번을 효과 감소 구조로 돌리기")
save("lab24_psawane", c, marks={"같은 반복을 뽑았는가: True": 1, "0.371": 2, "효과 감소 0.0508": 3, "효과 감소 0.4204": 4})

# ---------------------------------------------------------------- d. 비용효과 수용곡선
c = nb.cell('''
first = pd.DataFrame({"d_qaly": dq[:5], "d_cost": dc[:5]},
                     index=[1, 2, 3, 4, 5])
first["inmb_5000"] = 5000 * first["d_qaly"] - first["d_cost"]
first["inmb_6000"] = 6000 * first["d_qaly"] - first["d_cost"]
print(first.round({"d_qaly": 3, "d_cost": 0, "inmb_5000": 0,
                   "inmb_6000": 0}))

inmb = 5000 * dq - dc                  # 5,000번 모두
print("양수인 반복:", int((inmb > 0).sum()), "/", len(inmb),
      "=", np.mean(inmb > 0))
''', title="증분 순금전편익과 비용효과적일 확률")
save("lab24_inmb", c, marks={"143.0": 1, "-949.0": 2, "118.0": 3, "802 / 5000": 4, "0.1604": 5})

c = nb.cell('''
def p_ce(dc, dq, lam):
    """임계값 lam에서 순금전편익이 양수인 반복의 비율"""
    return np.mean(lam * dq - dc > 0)

def evpi(dc, dq, lam):
    """1인당 완전정보의 기대가치 (대안이 둘일 때)"""
    nb = lam * dq - dc
    return np.maximum(nb, 0).mean() - max(nb.mean(), 0)

rows = []
for lam in [3000, 4000, 5000, 5500, 6000, 7000, 8000, 10000, 12000]:
    rows.append([lam, (lam * dq - dc).mean(),
                 p_ce(dc, dq, lam), evpi(dc, dq, lam)])
cols = ["threshold", "mean_inmb", "p_ce", "evpi"]
tab = pd.DataFrame(rows, columns=cols).set_index("threshold")
tab.round({"mean_inmb": 0, "p_ce": 4, "evpi": 1})
''', title="임계값별 확률과 EVPI")
save("lab24_ceac", c, dfmarks={"0.0022": 1, "-308.0": 2, "0.1604": 3, "35.9": 4, "139.1": 5, "0.6586": 6, "0.9300": 7, "0.9752": 8})

c = nb.cell('''
lams = np.arange(0, 12001, 100)        # 0, 100, ..., 12000
curve = np.array([p_ce(dc, dq, lam) for lam in lams])
print(len(lams), "개 임계값. 양 끝의 확률:", curve[0], curve[-1])

fine = np.arange(0, 12001, 10)         # 10만원 간격
curve_f = np.array([p_ce(dc, dq, lam) for lam in fine])
lam50 = fine[np.argmax(curve_f >= 0.5)]
print("확률이 50%를 처음 넘는 임계값:", lam50)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(lams, curve)
ax.axvline(threshold, color="gray", ls="--")
ax.axhline(0.5, color="gray", ls=":")
ax.set_xlabel("Threshold (10,000 KRW per QALY)")
ax.set_ylabel("Probability that drug A is cost-effective")
ax.set_ylim(0, 1)
plt.tight_layout()
''', title="수용곡선 그리기")
save("lab24_ceacplot", c, marks={"0.001 0.9752": 1, "5650": 2})

c = nb.cell('''
icer_i = dc / dq                       # 모의실험마다의 ICER
print("ICER가 음수인 반복:", int((icer_i < 0).sum()))
for lam in [0, 6000, 12000, 300000]:
    print("임계값", lam, " 수용곡선", p_ce(dc, dq, lam),
          " ICER가 임계값 이하", np.mean(icer_i <= lam))

wide = np.arange(0, 300001, 1000)      # 30억 원/QALY까지
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, x in [(axes[0], lams), (axes[1], wide)]:
    ax.plot(x, [p_ce(dc, dq, lam) for lam in x], label="CEAC")
    ax.plot(x, [np.mean(icer_i <= lam) for lam in x], ls="--",
            label="Share of ICERs <= threshold")
    ax.set_xlabel("Threshold (10,000 KRW per QALY)")
axes[0].set_ylabel("Proportion of simulations")
axes[0].legend(loc="lower right", fontsize=8)
axes[1].set_xticks([0, 100000, 200000, 300000])
axes[1].set_ylim(0.95, 1.002)
axes[1].set_title("Upper end (y from 0.95)")
plt.tight_layout()
''', title="수용곡선과 'ICER가 임계값 이하인 비율'")
save("lab24_ceaccdf", c, marks={"ICER가 음수인 반복: 40": 1, "이하 0.008": 2, "이하 0.983": 3, "수용곡선 0.9922": 4, "이하 1.0": 5})

c = nb.cell('''
lam_mean = dc.mean() / dq.mean()       # 순편익의 평균이 0이 되는 임계값
print("확률이 50%를 처음 넘는 임계값:", lam50)
print("순편익의 평균이 0이 되는 임계값:", round(lam_mean),
      " 그때의 확률:", p_ce(dc, dq, lam_mean))
print("기준 분석의 ICER:", round(base["icer"]))
print("모의실험별 ICER의 중앙값:", round(np.median(icer_i)),
      " 평균:", round(icer_i.mean()))
print("모의실험별 ICER의 2.5, 97.5 백분위수:",
      np.percentile(icer_i, [2.5, 97.5]).round(0))
''', title="50% 지점, 평균끼리 나눈 ICER, 모의실험별 ICER")
save("lab24_lam50", c, marks={"임계값: 5650": 1, "임계값: 5586": 2, "0.4718": 3, "중앙값: 5625": 4, "평균: 4237": 5, "10187.]": 6})

c = nb.cell('''
pos = inmb[inmb > 0]                   # 신약 A가 더 나았던 반복
print("순금전편익의 평균:", round(inmb.mean(), 1))
print("양수인 비율:", np.mean(inmb > 0),
      " 양수인 반복의 평균:", round(pos.mean(), 1))
print("둘의 곱:", round(np.mean(inmb > 0) * pos.mean(), 1),
      " evpi():", round(evpi(dc, dq, 5000), 1))

ev = np.array([evpi(dc, dq, lam) for lam in fine])
print("EVPI가 가장 큰 임계값:", fine[np.argmax(ev)],
      " 그때의 EVPI:", round(ev.max(), 1))
print("환자 3,000명이면(억원):",
      round(evpi(dc, dq, 5000) * 3000 / 10000, 1))
''', title="EVPI 풀어 보기")
save("lab24_evpi", c, marks={"224.1": 1, "둘의 곱: 35.9": 2, "5590": 3, "164.1": 4, "10.8": 5})

# ---------------------------------------------------------------- e. 과제
c = nb.cell('''
pC = dict(p, hr_pfs=0.60, hr_os=0.90)   # 신약 C의 위험비
print("기준 분석 ICER:", round(run_model(pC)["icer"]))
dc_C, dq_C, n_C = run_psa(pC)           # 같은 분포와 seed로 5,000번
print("다시 뽑은 비율:", round(n_C["crossed"] / n_C["tries"], 4))
inmb_C = 5000 * dq_C - dc_C
for name, x in [("d_cost", dc_C), ("d_qaly", dq_C), ("inmb", inmb_C)]:
    print(name, round(x.mean(), 3),
          np.percentile(x, [2.5, 97.5]).round(3))
for lam in [5000, 6000]:
    print("임계값", lam, " 확률", p_ce(dc_C, dq_C, lam))
print("증분 QALY가 0 이하인 반복:", np.mean(dq_C <= 0))
''', title="과제 1 정답. 신약 C의 확률적 분석")
save("lab24_hw1", c, marks={"기준 분석 ICER: 5865": 1, "0.2534": 2, "d_cost 1712.129": 3, "[-0.088": 4, "inmb -226.539": 5,
                           "확률 0.2754": 6, "확률 0.5482": 7, "0.0674": 8})

c = nb.cell('''
fig, ax = plt.subplots(figsize=(6.5, 4))
for price in [190.0, price_star, 160.0]:
    p2 = dict(p, c_drug_A=price)       # 약값만 바꾼 입력값
    c2, q2, n2 = run_psa(p2)           # 같은 seed로 5,000번
    print("약값", round(price, 1),
          " ICER", round(run_model(p2)["icer"]),
          " 평균 순금전편익", round((5000 * q2 - c2).mean()),
          " 확률", p_ce(c2, q2, 5000),
          " QALY가 같은가", np.allclose(q2, dq))
    ax.plot(lams, [p_ce(c2, q2, lam) for lam in lams],
            label="Price " + str(round(price, 1)))
ax.axvline(threshold, color="gray", ls="--")
ax.axhline(0.5, color="gray", ls=":")
ax.set_xlabel("Threshold (10,000 KRW per QALY)")
ax.set_ylabel("Probability that drug A is cost-effective")
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
''', title="과제 2 정답. 약값을 낮췄을 때의 확률과 수용곡선")
save("lab24_hw2", c, marks={"확률 0.1604": 1, "확률 0.4826": 2, "평균 순금전편익 12": 3, "확률 0.8254": 4, "True": 5})

c = nb.cell('''
spec_ok = spec                          # 셀 5의 사전을 따로 둔다
spec = dict(spec_ok, u_pf=("beta", 0.20), u_pd=("beta", 0.20))
print("동료의 무진행 효용 95% 구간:", np.round(range95(p, "u_pf"), 2))
c3, q3, n3 = run_psa(p)                 # 동료의 분석
spec = spec_ok                          # 셀 5의 사전으로 되돌린다
print("셀 6의 구간:", np.round(range95(p, "u_pf"), 2))

for name, c_, q_ in [("동료", c3, q3), ("셀 16", dc, dq)]:
    print(name, " 증분 QALY 구간", np.percentile(q_, [2.5, 97.5]).round(3),
          " 순편익 평균", round((5000 * q_ - c_).mean()),
          " 확률", p_ce(c_, q_, 5000),
          " EVPI", round(evpi(c_, q_, 5000), 1))
print("환자 100명, 표준편차 0.20이면 표준오차:", 0.20 / np.sqrt(100))
''', title="과제 3 정답. 효용의 표준편차를 표준오차 자리에 넣은 분석")
save("lab24_hw3", c, marks={"[0.28 1.  ]": 1, "[0.032 1.081]": 2, "순편익 평균 -293": 3, "확률 0.3618": 4, "EVPI 173.4": 5,
                           "표준오차: 0.02": 6})

print("lab24: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# 셀 번호(본문에 '셀 N'으로 적은 것)가 바뀌지 않았는지 확인한다
TITLES = {1: "패키지 불러오기", 2: "입력값 사전", 3: "모형 함수", 4: "곡선을 중앙값", 5: "입력값의 분포", 6: "입력값별 범위",
          7: "입력값을 하나씩", 8: "토네이도", 9: "ICER가 임계값과", 10: "약값과 무진행생존", 11: "치료 효과 감소",
          12: "시나리오", 13: "상관 있는", 14: "곡선과 위험비", 15: "입력값 한 세트", 16: "5,000번", 17: "결과 요약",
          18: "비용효과평면", 19: "모의실험 횟수", 20: "상관과 다시", 21: "같은 5,000번", 22: "증분 순금전편익",
          23: "임계값별", 24: "수용곡선 그리기", 25: "수용곡선과", 26: "50% 지점", 27: "EVPI 풀어", 28: "과제 1",
          29: "과제 2", 30: "과제 3"}
assert len(nb.cells) == len(TITLES) and all(nb.cells[n - 1].title.startswith(t) for n, t in TITLES.items()), \
    [(cc.n, cc.title) for cc in nb.cells]

# ---------------------------------------------------------------- 대조 블록: lib_p4와 24장 본문의 숫자에 맞추기
import lib_p4 as L  # noqa: E402  (읽기만 한다. 학생 코드는 이 파일을 쓰지 않는다)

ns = nb.ns
N = json.load(open(os.path.join(HERE, "_ch24_nums.json"), encoding="utf-8"))
SEED = 20261002
CHECKS, BOOK = [], []


def chk(name, got, want, tol=1e-9):
    """실습 값(got)과 lib_p4 또는 _ch24_nums.json의 값(want)의 가장 큰 차이가 tol보다 작은지 본다."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape, f"{name}: shape {got.shape} vs {want.shape}"
    err = float(np.max(np.abs(got - want)))
    CHECKS.append((name, err))
    assert err < tol, f"{name}: max abs difference {err:g} (tol {tol:g})"
    return err


def book(name, got, want):
    """본문(24장)에 적힌 반올림 값과 실습 값을 같은 자릿수로 반올림해 비교한다. want는 본문에 적힌 수
    (int이면 정수 자리, float이면 적힌 소수 자릿수까지. 목록 안에 int와 float를 섞어 적어도 각자의 자릿수로 비교한다)."""
    got = list(np.atleast_1d(got))
    want = list(want) if isinstance(want, (list, tuple)) else [want]
    assert len(got) == len(want), name
    for g, w in zip(got, want):
        nd = len(str(w).split(".")[1]) if isinstance(w, float) else 0
        ok = round(float(g), nd) == w
        BOOK.append((name, float(g), w, ok))
        assert ok, f"{name}: lab {g} vs chapter {w}"


P0, REF = L.base_params(), L.run()
p_, base_ = ns["p"], ns["base"]
run_model, get_input, set_input = ns["run_model"], ns["get_input"], ns["set_input"]

# 가. 입력값, 기준 분석, get_input / set_input
assert set(p_) == set(P0)
chk("18 inputs = lib_p4.base_params()", [p_[k] for k in P0], [P0[k] for k in P0], 1e-15)
chk("base case: d_cost, d_qaly, ICER, NMB", [base_["d_cost"], base_["d_qaly"], base_["icer"], ns["nmb"]],
    [REF["d_cost"], REF["d_qaly"], REF["icer"], L.nmb(REF)])
chk("base case = _ch24_nums.json", [base_["d_cost"], base_["d_qaly"], base_["icer"]], [N["base"][k] for k in ("d_cost", "d_qaly", "icer")])
book("base case (d_cost, ICER, NMB)", [base_["d_cost"], base_["icer"], ns["nmb"]], [2872, 5621, -317])
book("base case d_qaly", base_["d_qaly"], 0.511)
RANGES = L.dsa_ranges()
assert list(ns["spec"]) == list(L.PSA_SPEC) and all(tuple(ns["spec"][k]) == tuple(L.PSA_SPEC[k]) for k in L.PSA_SPEC)
chk("get_input for 17 inputs", [get_input(p_, k) for k in RANGES], [L.get_input(P0, k) for k in RANGES], 1e-14)
_d = 0.0
for k, (lo, hi, _, _) in RANGES.items():
    for v in (lo, hi):
        a, b = set_input(p_, k, v), L.set_input(P0, k, v)
        assert set(a) == set(b)
        _d = max(_d, max(abs(a[x] - b[x]) for x in b))
chk("set_input at both ends of 17 ranges (all 18 dict values)", _d, 0.0, 1e-14)
for k in ("os_gam", "pfs_gam"):                     # 모양을 바꿔도 중앙값은 그대로
    for v in RANGES[k][:2]:
        assert abs(get_input(set_input(p_, k, v), k[:-3] + "med") - get_input(p_, k[:-3] + "med")) < 1e-9

# 나. 범위, 일원 민감도 분석, 임계 가격, 이원 분석, 시나리오
rg, ow = ns["ranges"], ns["ow"]
assert list(rg.index) == list(RANGES)
chk("17 ranges (low, high) = lib_p4.dsa_ranges()", rg[["low", "high"]].to_numpy(), [[RANGES[k][0], RANGES[k][1]] for k in RANGES], 1e-12)
for k in ("u_pf", "c_pd", "hr_os"):
    d = N["dist"][k]
    chk(f"range of {k} = _ch24_nums.json dist", rg.loc[k, ["low", "high"]], [d["lo"], d["hi"]], 1e-12)
chk("beta_ab, gamma_ks (method of moments)", list(ns["beta_ab"](0.78, 0.03)) + list(ns["gamma_ks"](250, 50.0)),
    [N["dist"]["u_pf"]["alpha"], N["dist"]["u_pf"]["beta"], N["dist"]["c_pd"]["shape"], N["dist"]["c_pd"]["scale"]], 1e-12)
OW = L.oneway()
assert list(ow.index) == [r["key"] for r in OW], "토네이도 순위가 lib_p4.oneway()와 다르다"
assert list(ow.index) == [r["key"] for r in N["oneway"]]
chk("one-way: ICER at low, high, swing (17 x 3) = lib_p4.oneway()", ow[["icer_low", "icer_high", "swing"]].to_numpy(),
    [[r["icer_low"], r["icer_high"], r["swing"]] for r in OW], 1e-8)
assert [bool(x) for x in ow["below"]] == [bool(r["crosses"]) for r in OW]
_BOOK_OW = {  # 24장 표 24-2와 '더 알아보기'의 나머지 11개 표: 범위(낮은 값, 높은 값), 낮은 값·높은 값일 때 ICER, 폭
    "hr_os": (0.58, 0.97, 5232, 7922, 2690), "hr_pfs": (0.52, 0.81, 4784, 6499, 1715), "c_drug_A": (152, 190, 4006, 5621, 1614),
    "c_drug_B": (96, 144, 6289, 4952, 1337), "u_pf": (0.72, 0.84, 6076, 5262, 814), "c_pf": (26, 57, 5414, 5871, 457),
    "c_ae_A": (69, 186, 5520, 5749, 229), "c_pd": (162, 357, 5523, 5739, 215), "os_gam": (1.02, 1.29, 5519, 5734, 215),
    "pfs_gam": (0.86, 1.05, 5515, 5714, 199), "pfs_med": (8.7, 11.5, 5537, 5715, 177), "os_med": (23.9, 32.8, 5706, 5546, 160),
    "c_ae_B": (46, 124, 5688, 5535, 153), "du_ae_A": (0.007, 0.019, 5565, 5694, 129), "u_pd": (0.52, 0.72, 5673, 5572, 101),
    "du_ae_B": (0.005, 0.012, 5659, 5573, 86), "c_death": (518, 1143, 5637, 5600, 37)}
assert list(_BOOK_OW) == list(ow.index)             # 본문 표의 줄 순서와 같다
for k, (lo, hi, il, ih, sw) in _BOOK_OW.items():
    book(f"one-way {k}: range", [rg.loc[k, "low"], rg.loc[k, "high"]], [lo, hi])
    book(f"one-way {k}: ICER low, high, swing", [ow.loc[k, "icer_low"], ow.loc[k, "icer_high"], ow.loc[k, "swing"]], [il, ih, sw])
chk("threshold price = lib_p4.threshold_price(), _ch24_nums.json", [ns["price_star"]] * 2, [L.threshold_price(), N["price"]["tp"]])
book("threshold price, cut %, ICER change per 1 unit of price", [ns["price_star"], (1 - ns["price_star"] / 190) * 100, ns["slope"]], [175.4, 7.7, 42.5])
assert abs(run_model(dict(p_, c_drug_A=ns["price_star"]))["icer"] - 5000) < 1e-8
two = ns["two"]
chk("two-way table = lib_p4.twoway(), _ch24_nums.json", np.r_[two.to_numpy(), two.to_numpy()],
    np.r_[L.twoway("c_drug_A", [190.0, 175.0, 160.0], "hr_pfs", ns["hrs"]), np.array(N["twoway"]["icer"])], 1e-8)
book("two-way table ('더 알아보기')", two.to_numpy().ravel().tolist(), [4784, 5621, 6499, 4090, 4983, 5922, 3395, 4346, 5345])
# 셀 11–12: 효과 감소·소실 모형과 시나리오 표(24장 표 24-3의 방법론적 선택과 효과 감소·소실 줄)
run_waning = ns["run_waning"]
chk("run_waning without waning = run_model", [ns["r0"]["d_cost"], ns["r0"]["d_qaly"], ns["r0"]["icer"]],
    [base_["d_cost"], base_["d_qaly"], base_["icer"]], 1e-8)
SC = {r["key"]: r for r in L.scenario_table(os_models=False)}
sc = ns["sc"]
_SCKEY = {"base": "base", "horizon 5y": "horizon5", "horizon 10y": "horizon10", "horizon 15y": "horizon15",
          "discount 0%": "disc0", "discount 3%": "disc3", "waning 36-60": "wane_3_5", "waning 60-84": "wane_5_7"}
for mine, key in _SCKEY.items():
    chk(f"scenario {mine} = lib_p4.scenario_table()", sc.loc[mine, ["d_cost", "d_qaly", "icer"]],
        [SC[key]["d_cost"], SC[key]["d_qaly"], SC[key]["icer"]], 1e-8)
W30 = L.run_waning(start=30.0, end=30.0)
chk("loss after 30 months = lib_p4.run_waning(30, 30)", sc.loc["loss after 30", ["d_cost", "d_qaly", "icer"]],
    [W30["d_cost"], W30["d_qaly"], W30["icer"]], 1e-8)
chk("loss after 30: LY of drug A", ns["r2"]["A"]["ly"], W30["A"]["ly"], 1e-10)
chk("hr path 36-60 = lib_p4", ns["r1"]["hr_os_t"], L.run_waning(start=36.0, end=60.0)["hr_t"]["os"], 1e-15)
_BOOK_SC = {"base": (2872, 0.511, 5621), "horizon 5y": (1875, 0.297, 6324), "horizon 10y": (2640, 0.464, 5686),
            "horizon 15y": (2836, 0.504, 5628), "discount 0%": (3508, 0.646, 5431), "discount 3%": (3058, 0.550, 5555),
            "waning 36-60": (2285, 0.372, 6150), "waning 60-84": (2576, 0.443, 5815), "loss after 30": (1939, 0.282, 6876)}
assert list(_BOOK_SC) == list(sc.index)
for k, v in _BOOK_SC.items():
    book(f"Table 24-3 / loss scenario {k}", sc.loc[k, ["d_cost", "d_qaly", "icer"]].tolist(), list(v))
book("scenario change vs base %", sc["vs_base"].tolist()[1:], [12.5, 1.2, 0.1, -3.4, -1.2, 9.4, 3.5, 22.3])
book("hr at month 48 (36-60), LY of A under loss", [ns["r1"]["hr_os_t"][47], ns["r2"]["A"]["ly"], base_["A"]["ly"]], [0.87, 3.455, 3.916])

# 다. 확률적 민감도 분석: 5,000번이 lib_p4.psa()와 배열 단위로 같은가
DC, DQ = L.psa(n=5000, seed=SEED)
dc, dq = ns["dc"], ns["dq"]
E_DC = chk("PSA 5,000 incremental costs = lib_p4.psa() (array)", dc, DC, 1e-9)
E_DQ = chk("PSA 5,000 incremental QALYs = lib_p4.psa() (array)", dq, DQ, 1e-12)
DRAWS, CNT = L.psa_inputs(5000, SEED)
_rng = np.random.default_rng(SEED)
_cnt = {"tries": 0, "crossed": 0}
_mine = [ns["draw_inputs"](_rng, p_, _cnt) for _ in range(5000)]
assert all(set(a) == set(b) for a, b in zip(_mine, DRAWS))
E_IN = chk("PSA 5,000 input sets x 18 values = lib_p4.psa_inputs()", [[a[k] for k in P0] for a in _mine], [[b[k] for k in P0] for b in DRAWS], 1e-13)
assert not any(ns["crossed"](q) for q in _mine) and not any(L.curves_cross(q) for q in _mine)
assert (ns["count"]["tries"], ns["count"]["crossed"]) == (CNT["tries"], CNT["rejected"]) == (N["cap"]["tries"], N["cap"]["rejected"]) == (5137, 137)
book("redraw share %", 100 * ns["count"]["crossed"] / ns["count"]["tries"], 2.7)
PS = L.psa_summary(DC, DQ)
sm = ns["summ"]
chk("PSA summary: mean, 2.5th, 97.5th of d_cost, d_qaly, INMB = lib_p4.psa_summary()", sm.to_numpy(),
    [[PS[k]["mean"], PS[k]["lo"], PS[k]["hi"]] for k in ("d_cost", "d_qaly", "inmb")], 1e-9)
chk("PSA summary = _ch24_nums.json", sm.to_numpy(), [[N["psa"][k]["mean"], N["psa"][k]["lo"], N["psa"][k]["hi"]] for k in ("d_cost", "d_qaly", "inmb")], 1e-9)
assert ns["p_ce5"] == PS["p_ce"] == N["psa"]["p_ce"] == 0.1604
chk("ICER of means, MC standard error", [dc.mean() / dq.mean(), np.sqrt(ns["p_ce5"] * (1 - ns["p_ce5"]) / 5000)], [PS["icer"], PS["mcse"]], 1e-9)
quad = {k: int(v.sum()) for k, v in ns["quad"].items()}
assert quad == {k: round(v * 5000) for k, v in PS["quadrants"].items()} == {"NE": 4958, "SE": 3, "NW": 37, "SW": 2}
book("PSA d_cost mean (2.5, 97.5)", sm.loc["d_cost"].tolist(), [2935, 1061, 5102])
book("PSA d_qaly mean (2.5, 97.5)", sm.loc["d_qaly"].tolist(), [0.525, 0.101, 1.008])
book("PSA INMB mean (2.5, 97.5)", sm.loc["inmb"].tolist(), [-308, -919, 412])
book("PSA ICER, n cost-effective, quadrant counts", [dc.mean() / dq.mean(), int((ns["inmb"] > 0).sum()), quad["NE"], quad["NW"], quad["SE"], quad["SW"]],
     [5586, 802, 4958, 37, 3, 2])
book("PSA probability % at 5,000, MC standard error", 100 * ns["p_ce5"], 16.0)
book("MC standard error", np.sqrt(ns["p_ce5"] * (1 - ns["p_ce5"]) / 5000), 0.0052)
# 상관 있는 난수를 만드는 줄: 채택된 5,000번에서 로그 값의 상관이 지정한 0.5 근처인가
for k1, k2 in ns["pairs"]:
    r_ = np.corrcoef(np.log([get_input(q, k1) for q in _mine]), np.log([get_input(q, k2) for q in _mine]))[0, 1]
    assert 0.45 < r_ < 0.56, (k1, k2, r_)
# 셀 19: 모의실험 횟수(24장 b 절 2단계 '모의실험 횟수는 어떻게 정하나?')
ok_ = ns["ok"]
for n_ in (500, 1000, 5000):
    assert ok_[:n_].mean() == N["conv"]["at"][str(n_)]
book("running probability % at 500, 1,000, 5,000", [100 * ok_[:500].mean(), 100 * ok_[:1000].mean(), 100 * ok_.mean()], [14.2, 15.2, 16.0])
MC_SEEDS = {}
for seed in (1, 2, 3):
    c_, q_ = L.psa(5000, seed=seed)
    c2, q2, _ = ns["run_psa"](p_, seed=seed)
    chk(f"PSA seed {seed}: arrays = lib_p4.psa()", np.r_[c2, q2], np.r_[c_, q_], 1e-9)
    MC_SEEDS[seed] = float(np.mean(5000 * q2 - c2 > 0))
    assert N["conv"]["seeds"]["5000"]["min"] <= MC_SEEDS[seed] <= N["conv"]["seeds"]["5000"]["max"]


def lib_psa(corr, redraw, n=5000, seed=SEED):
    """lib_p4로 같은 계산을 한다. corr: 세 짝의 상관(PSA_CORR을 잠시 바꾼다). redraw=False이면 lib_p4.draw()의 순서 그대로
    (draw_survival → 나머지 입력값) 뽑되 곡선이 엇갈려도 다시 뽑지 않는다. 반환: (dc, dq, 뽑은 횟수, 엇갈린 횟수)."""
    keep = L.PSA_CORR
    L.PSA_CORR = {k: corr for k in keep}
    try:
        if redraw:
            c_, q_ = L.psa(n, seed)
            cnt = L.psa_inputs(n, seed)[1]
            return c_, q_, cnt["tries"], cnt["rejected"]
        rng = np.random.default_rng(seed)
        base, paired = L.base_params(), {k for pair in keep for k in pair}
        c_, q_, bad = np.empty(n), np.empty(n), 0
        for i in range(n):
            v = L.draw_survival(rng, base)
            bad += L.curves_cross(v)
            for k, (dist, u) in L.PSA_SPEC.items():
                if k in paired:
                    continue
                m = v[k]
                if dist == "beta":
                    nn = m * (1 - m) / u ** 2 - 1
                    v[k] = float(rng.beta(m * nn, (1 - m) * nn))
                elif dist == "gamma":
                    se = m * u
                    shape = (m / se) ** 2
                    v[k] = float(rng.gamma(shape, m / shape))
            r = L.run(v)
            c_[i], q_[i] = r["d_cost"], r["d_qaly"]
        return c_, q_, n, bad
    finally:
        L.PSA_CORR = keep


_rows = []
for rho, redraw in [(0.5, True), (0.0, True), (0.5, False), (0.0, False)]:
    c_, q_, tries, bad = lib_psa(rho, redraw)
    _rows.append([rho, redraw, tries, bad, bad / tries, float(np.mean(5000 * q_ - c_ > 0))])
    c2, q2, n2 = ns["run_psa"](p_, rho=rho, redraw=redraw)
    chk(f"PSA rho={rho}, redraw={redraw}: arrays = lib_p4", np.r_[c2, q2], np.r_[c_, q_], 1e-9)
    assert (n2["tries"], n2["crossed"]) == (tries, bad), (rho, redraw, n2, tries, bad)
assert (_rows[1][2], _rows[1][3]) == (N["cap"]["tries_indep"], N["cap"]["rejected_indep"]) == (5503, 503)
book("redraw share % when pairs are drawn independently", 100 * _rows[1][4], 9.1)
book("no redraw: crossed sets, probability % (rho 0.5; rho 0)", [_rows[2][3], _rows[3][3], 100 * _rows[3][5]], [141, 427, 17.0])
VARY = _rows
# 셀 21: 같은 5,000번을 효과 감소 구조로(24장 b 절 3단계 '모형 구조의 불확실성은 들어 있나?')
_rw = [L.run_waning(q, 36.0, 60.0) for q in DRAWS]
DCW, DQW = np.array([r["d_cost"] for r in _rw]), np.array([r["d_qaly"] for r in _rw])
chk("PSA under waning 36-60: arrays = lib_p4.run_waning on psa_inputs()", np.r_[ns["dc_w"], ns["dq_w"]], np.r_[DCW, DQW], 1e-9)
assert ns["n_w"] == ns["count"]
book("PSA under waning: probability % at 5,000 and 6,000", [100 * L.ceac(DCW, DQW, 5000.0), 100 * L.ceac(DCW, DQW, 6000.0)], [5.1, 42.0])
book("PSA under waning: mean d_cost, d_qaly", [DCW.mean(), DQW.mean()], [2288, 0.371])

# 라. 수용곡선, EVPI, ICER의 누적분포와의 차이, 50% 지점
chk("CEAC on 0-12,000 by 100 (121 values) = lib_p4.ceac(), _ch24_nums.json", np.r_[ns["curve"], ns["curve"]], np.r_[L.ceac(DC, DQ), N["ceac"]["p"]], 1e-12)
tab = ns["tab"]
chk("threshold table: mean INMB, probability, EVPI (9 x 3) = _ch24_nums.json", tab.to_numpy(),
    [[N["ceac"]["tab"][str(l)][k] for k in ("nmb_mean", "p", "evpi")] for l in tab.index], 1e-8)
chk("EVPI at 5,000 = lib_p4.evpi()", ns["evpi"](dc, dq, 5000), L.evpi(DC, DQ), 1e-9)
chk("EVPI on fine grid = lib_p4.evpi()", ns["ev"], L.evpi(DC, DQ, ns["fine"].astype(float)), 1e-9)
assert int(ns["lam50"]) == int(N["ceac"]["lam50"]) == 5650
assert int(ns["fine"][np.argmax(ns["ev"])]) == int(N["ceac"]["lam_evpi_max"]) == 5590
chk("mean of positive INMB, max EVPI", [ns["pos"].mean(), ns["ev"].max()], [N["inmb"]["pos_mean"], N["ceac"]["evpi_max"]], 1e-8)
f5 = ns["first"]
chk("first five simulations = _ch24_nums.json", f5.to_numpy(), [[x["dq"], x["dc"], x["nb5"], x["nb6"]] for x in N["first5"]], 1e-9)
book("Table 24-6 d_qaly", f5["d_qaly"].tolist(), [0.570, 0.700, 0.985, 0.551, 0.317])
book("Table 24-6 d_cost, INMB(5,000), INMB(6,000)", f5[["d_cost", "inmb_5000", "inmb_6000"]].to_numpy().ravel().tolist(),
     [2709, 143, 713, 4451, -949, -248, 4429, 497, 1482, 3187, -433, 118, 2101, -516, -199])
book("Table 24-7 mean INMB", tab["mean_inmb"].tolist(), [-1359, -833, -308, -45, 218, 743, 1269, 2319, 3370])
book("Table 24-7 probability %", (100 * tab["p_ce"]).tolist(), [0.2, 0.7, 16.0, 42.0, 65.9, 86.7, 93.0, 96.5, 97.5])
book("EVPI by threshold (c 절 '더 알아보기')", tab["evpi"].tolist(), [0.5, 1.3, 35.9, 139.1, 100.7, 42.6, 27.0, 18.4, 16.0])
book("CEAC ends %: lambda 0, 12,000", [100 * ns["curve"][0], 100 * ns["curve"][-1]], [0.1, 97.5])
book("50% point, EVPI max threshold, EVPI max", [ns["lam50"], ns["fine"][np.argmax(ns["ev"])], ns["ev"].max()], [5650, 5590, 164])
book("mean INMB when positive", ns["pos"].mean(), 224.1)
# 셀 25–26: 24장 c 절 4단계 '곡선을 읽을 때 틀리기 쉬운 것' 1–2, b 절 4단계 'ICER의 신뢰구간을 쓰지 않는 이유'
icer_i = ns["icer_i"]
chk("per-simulation ICER = DC / DQ (relative difference; ICER is up to 26.5억 when d_qaly is near 0)", icer_i / (DC / DQ) - 1, np.zeros(5000), 1e-9)
book("share of ICER <= lambda % at 0, CEAC at 0, ceiling of CEAC (share d_qaly > 0)",
     [100 * np.mean(icer_i <= 0), 100 * L.ceac(DC, DQ, 0.0), 100 * np.mean(DQ > 0)], [0.8, 0.1, 99.2])
assert int((icer_i < 0).sum()) == 40 and np.mean(icer_i <= 300000) == 1.0 and L.ceac(DC, DQ, 300000.0) == 0.9922
lam_mean = ns["lam_mean"]
chk("threshold where mean INMB = 0 equals ICER of means", lam_mean, PS["icer"], 1e-9)
assert abs(L.inmb(DC, DQ, lam_mean).mean()) < 1e-8
book("ICER of means, probability % there, median and mean of per-simulation ICER",
     [lam_mean, 100 * L.ceac(DC, DQ, lam_mean), np.median(icer_i), icer_i.mean()], [5586, 47.2, 5625, 4237])
book("per-simulation ICER 2.5th, 97.5th percentiles", np.percentile(icer_i, [2.5, 97.5]).tolist(), [4371, 10187])

# 마. 과제
# 과제 1: 신약 C(24장 c 절 스스로 확인하기 2번)
pc = L.set_param(L.set_param(P0, "hr_pfs", 0.60), "hr_os", 0.90)
DCC, DQC = L.psa(5000, SEED, pc)
CNTC = L.psa_inputs(5000, SEED, pc)[1]
chk("drug C PSA arrays = lib_p4.psa(p=drug C)", np.r_[ns["dc_C"], ns["dq_C"]], np.r_[DCC, DQC], 1e-9)
assert (ns["n_C"]["tries"], ns["n_C"]["crossed"]) == (CNTC["tries"], CNTC["rejected"])
SC_C = L.psa_summary(DCC, DQC)
book("drug C base ICER", L.run(pc)["icer"], 5865)
book("drug C d_cost mean (2.5, 97.5)", [SC_C["d_cost"]["mean"], SC_C["d_cost"]["lo"], SC_C["d_cost"]["hi"]], [1712, -136, 3579])
book("drug C d_qaly mean (2.5, 97.5)", [SC_C["d_qaly"]["mean"], SC_C["d_qaly"]["lo"], SC_C["d_qaly"]["hi"]], [0.297, -0.088, 0.732])
book("drug C mean INMB, probability % at 5,000 and 6,000, redraw %",
     [SC_C["inmb"]["mean"], 100 * SC_C["p_ce"], 100 * L.ceac(DCC, DQC, 6000.0), 100 * CNTC["rejected"] / CNTC["tries"]],
     [-227, 27.5, 54.8, 25.3])
# 과제 2: 약값을 바꾼 확률적 분석(24장 c 절 '더 알아보기'의 '약값에 따른 비용효과적일 확률')
HW2 = {}
for price, key in ((190.0, "190.0"), (ns["price_star"], "175.39"), (160.0, "160.0")):
    c_, q_ = L.psa(5000, seed=SEED, p=L.set_param(P0, "c_drug_A", price))
    c2, q2, _ = ns["run_psa"](dict(p_, c_drug_A=price))
    chk(f"PSA at price {price:.2f}: arrays = lib_p4.psa()", np.r_[c2, q2], np.r_[c_, q_], 1e-9)
    HW2[key] = (float(np.mean(5000 * q2 - c2 > 0)), float((5000 * q2 - c2).mean()), run_model(dict(p_, c_drug_A=price))["icer"])
    assert HW2[key][0] == N["ceac"]["price_p"][key]["p"]
    chk(f"price {price:.2f}: mean INMB, ICER = _ch24_nums.json", HW2[key][1:], [N["ceac"]["price_p"][key]["nmb"], N["ceac"]["price_p"][key]["icer"]], 1e-7)
book("probability % at prices 190, 175.4, 160", [100 * HW2[k][0] for k in ("190.0", "175.39", "160.0")], [16.0, 48.3, 82.5])
book("mean INMB and ICER at prices 175.4, 160", [HW2["175.39"][1], HW2["175.39"][2], HW2["160.0"][1], HW2["160.0"][2]], [12, 5000, 348, 4346])
# 과제 3: 효용의 표준편차 0.20을 표준오차 자리에(24장 b 절 3단계 '분포의 폭은 무엇으로 정하나?')
assert ns["spec"] is ns["spec_ok"] and dict(ns["spec"]) == dict(L.PSA_SPEC)     # 셀이 사전을 되돌렸다
_keep = dict(L.PSA_SPEC)
try:
    L.PSA_SPEC["u_pf"] = ("beta", 0.20)
    L.PSA_SPEC["u_pd"] = ("beta", 0.20)
    DSU = L.dist_params("u_pf")
    DCS, DQS = L.psa(5000, SEED)
finally:
    L.PSA_SPEC.clear()
    L.PSA_SPEC.update(_keep)
chk("SD as SE: PSA arrays = lib_p4.psa() with PSA_SPEC utilities 0.20", np.r_[ns["c3"], ns["q3"]], np.r_[DCS, DQS], 1e-9)
SS = L.psa_summary(DCS, DQS)
book("SD as SE: utility interval, probability %", [DSU["lo"], DSU["hi"], 100 * SS["p_ce"]], [0.28, 1.00, 36.2])
book("SD as SE: mean INMB, d_qaly interval, EVPI; base EVPI",
     [SS["inmb"]["mean"], SS["d_qaly"]["lo"], SS["d_qaly"]["hi"], SS["evpi"], PS["evpi"]], [-293, 0.032, 1.081, 173.4, 35.9])
# 본문 문장에만 쓴(표식 없는) 숫자
book("loss after 30: LY shorter by, d_qaly as % of base", [base_["A"]["ly"] - ns["r2"]["A"]["ly"], 100 * sc.loc["loss after 30", "d_qaly"] / base_["d_qaly"]], [0.46, 55])
book("waning PSA: d_qaly smaller by", PS["d_qaly"]["mean"] - DQW.mean(), 0.15)
assert 9 < (CNTC["rejected"] / CNTC["tries"]) / (CNT["rejected"] / CNT["tries"]) < 10           # "아홉 배가 넘습니다"
book("width of INMB 95% interval: drug C, drug A", [SC_C["inmb"]["hi"] - SC_C["inmb"]["lo"], PS["inmb"]["hi"] - PS["inmb"]["lo"]], [1780, 1331])
book("share of d_qaly <= 0 %: drug C, drug A", [100 * np.mean(DQC <= 0), 100 * np.mean(DQ <= 0)], [6.7, 0.8])
book("SD as SE: positive simulations (base 802)", int((L.inmb(DCS, DQS) > 0).sum()), 1809)
_w0 = rg.loc["u_pf", "high"] - rg.loc["u_pf", "low"]
book("SD as SE: utility interval width ratio, EVPI ratio", [(DSU["hi"] - DSU["lo"]) / _w0, SS["evpi"] / PS["evpi"]], [6.1, 4.8])

print(f"대조: lib_p4·_ch24_nums.json과 비교한 {len(CHECKS)}개 항목 모두 허용 오차 안 (가장 큰 차이 {max(e for _, e in CHECKS):.2e})")
print(f"      5,000번 배열: 증분비용의 가장 큰 차이 {E_DC:.2e}만원, 증분 QALY {E_DQ:.2e}, 입력값 5,000번 × 18개 {E_IN:.2e}"
      f" (비트 단위로 같은가: {np.array_equal(dc, DC)}, {np.array_equal(dq, DQ)})")
print(f"      본문에 적힌 숫자 {len(BOOK)}개 가운데 {sum(b[3] for b in BOOK)}개가 같은 자릿수에서 일치")

# ---------------------------------------------------------------- 본문 문장에 쓴 그 밖의 숫자 확인
if os.environ.get("VERIFY") or os.environ.get("NOMARK"):
    print("VERIFY vary rows (rho, redraw, tries, crossed, share, p_ce):", VARY)
    print("VERIFY mc seeds:", MC_SEEDS, " chapter range", N["conv"]["seeds"]["5000"])
    print("VERIFY hw2:", HW2)
    q_ = dict(p_, os_gam=1.29)
    print("VERIFY naive dict(p, os_gam=1.29): median", get_input(q_, "os_med"), " set_input: lam", set_input(p_, "os_gam", 1.29)["os_lam"])
    print("VERIFY base S_OS(60) B:", float(np.exp(-p_["os_lam"] * 60 ** p_["os_gam"])))
    arr = {k: np.array([get_input(q, k) for q in _mine]) for k in ("pfs_med", "os_med")}
    up = arr["pfs_med"] > get_input(p_, "pfs_med")
    print("VERIFY share os_med > 28 given pfs_med > 10:", float(np.mean(arr["os_med"][up] > get_input(p_, "os_med"))), "(chapter", N["cap"]["same_side"], ")")
    print("VERIFY sqrt(1 - 0.25):", np.sqrt(0.75), " theory share:", 0.5 + np.arcsin(0.5) / np.pi)
    print("VERIFY d_qaly > 0 share:", float(np.mean(dq > 0)), " d_cost < 0 share:", float(np.mean(dc < 0)))
    print("VERIFY p_ce at 5,586 and base ICER:", ns["p_ce"](dc, dq, dc.mean() / dq.mean()), ns["p_ce"](dc, dq, base_["icer"]))
    print("VERIFY EVPI at 6,000:", ns["evpi"](dc, dq, 6000), " E[max] at 6,000:", float(np.maximum(6000 * dq - dc, 0).mean()))
    print("VERIFY one-way top rows:", ow.head(6).round(1).to_dict("index"))
    print("VERIFY rest max swing:", float(ow["swing"].iloc[6:].max()))
    print("VERIFY ICER change at hr_os ends (%):", [round(100 * (x / base_["icer"] - 1), 1) for x in ow.loc["hr_os", ["icer_low", "icer_high"]]])
    print("VERIFY c_drug_B +-10%: swing", abs(run_model(set_input(p_, "c_drug_B", 108.0))["icer"] - run_model(set_input(p_, "c_drug_B", 132.0))["icer"]))
    print("VERIFY scenario ICER range:", float(ns["sc"]["icer"].min()), float(ns["sc"]["icer"].max()))
    print("VERIFY waning PSA: EVPI 5,000", L.evpi(DCW, DQW), " NE share", float(np.mean((DQW > 0) & (DCW > 0))))
    print("VERIFY CEAC vs ICER<=lam at 6000, 12000:", L.ceac(DC, DQ, 6000.0), float(np.mean(icer_i <= 6000)), L.ceac(DC, DQ, 12000.0), float(np.mean(icer_i <= 12000)))
    print("VERIFY median INMB at 5640, 5650:", float(np.median(5640 * DQ - DC)), float(np.median(5650 * DQ - DC)))
    print("VERIFY drug C: d_qaly <= 0 share", float(np.mean(DQC <= 0)), " ceac 12000", L.ceac(DCC, DQC, 12000.0), " NE", SC_C["quadrants"], " EVPI", SC_C["evpi"])
    print("VERIFY SD as SE: d_cost", SS["d_cost"], " 6000", L.ceac(DCS, DQS, 6000.0), " NE", SS["quadrants"]["NE"])
    print("VERIFY pandas/numpy/scipy/matplotlib:", ns["pd"].__version__, np.__version__, __import__("scipy").__version__, __import__("matplotlib").__version__)

# ---------------------------------------------------------------- 노트북 (pub/notebooks/lab24.ipynb)
notes = {
    1: ("## a. 실습 준비\n\n출력을 만든 패키지의 버전을 확인하고, 실습 23에서 만든 입력값 사전과 모형 함수 `run_model(p)`를 다시 적은 뒤, "
        "곡선을 중앙값과 모양 모수로 읽고 바꾸는 함수를 만듭니다."),
    5: "## b. 일원 민감도 분석과 토네이도 그림\n\n입력값마다 범위를 정하고, 하나씩 양 끝으로 바꿔 ICER를 다시 계산합니다. 끝에서 임계 가격, 이원 민감도 분석, 시나리오 분석을 합니다.",
    11: "치료 효과가 줄거나 사라지는 가정을 넣은 모형 함수를 만들고(셀 11), 분석기간·할인율과 함께 시나리오 표를 만듭니다(셀 12).",
    13: "## c. 확률적 민감도 분석\n\n입력값을 분포에서 한꺼번에 뽑아 모형을 5,000번 돌립니다. 셀 16, 19, 20, 21은 몇 초 걸릴 수 있습니다.",
    22: "## d. 비용효과 수용곡선\n\n셀 16에서 얻은 5,000번의 증분비용 `dc`와 증분 QALY `dq`를 임계값별로 요약합니다.",
    28: ("## 과제 정답\n\n**과제 1.** 신약 C(무진행생존 위험비 0.60, 전체생존 위험비 0.90)를 같은 분포와 seed로 5,000번 돌리고, "
         "임계값 5,000만원/QALY의 결과를 말로 옮긴 뒤 이 확률이 말해 주지 않는 것을 씁니다."),
    29: ("**과제 2.** 신약 A의 월 약값을 임계 가격(셀 9의 `price_star`)과 160만원으로 낮춰 확률적 민감도 분석을 다시 돌리고, "
         "임계값 5,000만원/QALY에서 비용효과적일 확률을 구해 세 가격의 수용곡선을 한 그림에 그립니다."),
    30: "**과제 3.** 효용의 표준오차 자리에 환자 사이 표준편차 0.20을 넣은 동료의 분석을 재현하고, 틀린 곳 두 가지를 찾습니다.",
}
path = nb.save_ipynb(
    "실습 24. 민감도 분석",
    intro=("사회약학 연구방법 노트의 '실습 24 민감도 분석'에 나오는 셀을 차례로 모은 노트북입니다. "
           "진행성 신세포암 1차 치료(신약 A 대 표준요법 B, 가상의 예시)의 분할생존모형에 일원 민감도 분석, 시나리오 분석, "
           "확률적 민감도 분석 5,000회, 비용효과 수용곡선을 직접 돌려 24장 본문의 숫자를 재현합니다. "
           "자료 파일이 필요 없고 numpy, pandas, matplotlib, scipy만 씁니다. 셀을 위에서부터 차례로 실행하세요. "
           "설명과 출력 읽는 법은 사이트의 실습 24 쪽에 있습니다."),
    notes=notes)
print("notebook:", path, len(nb.cells), "cells")
