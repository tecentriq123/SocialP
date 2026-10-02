"""22장(효용과 QALY)의 모든 숫자.

실행: source /home/claude/pylibs/env.sh && python3 gen/nums_ch22.py
출력: gen/_ch22_nums.json (그림 스크립트와 본문 검산용), gen/_ch22_trial.csv (가상의 임상시험 자료, 긴 형식),
      gen/_ch22_pts.csv (환자별 진행·사망 시점과 QALY), gen/_ch22_cell.html (본문에 붙여 넣은 코드 셀과 실제 출력)

내용
  1. 직접 측정법 예(표준 도박, 시간교환, 시각 아날로그 척도)
  2. EQ-5D 건강상태 코드를 지수로 바꾸는 계산(한국 3L, 5L 가치 세트. 계수는 P4_FACTS.md 5.3, 5.4)
  3. 예시 환자 세 명의 QALY(사다리꼴 공식)
  4. 가상의 임상시험(진행성 신세포암, 신약 A 대 표준요법 B, 군당 160명, EQ-5D-5L 6회 측정)
  5. 공통 예시 모형(gen/lib_p4.py)의 QALY 분해
  6. 본문 '파이썬으로 계산해 보기'의 코드와 실제 출력
"""
import contextlib
import io
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib_p4 as L  # noqa: E402

N = {}

# ------------------------------------------------------------------ 1. 직접 측정법
N["sg"] = {"p": 0.75, "u": 0.75}
N["tto"] = {"t_state": 10, "t_full": 7, "u": 7 / 10}
# 시각 아날로그 척도: 상태 X 60점, 사망을 20점에 표시한 응답자라면 (60-20)/(100-20)
N["vas"] = {"x": 60, "dead": 20, "raw": 60 / 100, "rescaled": (60 - 20) / (100 - 20)}

# ------------------------------------------------------------------ 2. EQ-5D 가치 세트
# 한국 EQ-5D-3L (Lee 2009, N3 모형)과 EQ-5D-5L (Kim 2016). 계수 출처: P4_FACTS.md 5.3, 5.4
# 검수(2026-10-03)에서 확인한 것
#   - 세 가치 세트 모두 R 패키지 eq5d(fragla/eq5d, data-raw/TTO.csv·VT.csv)의 계수와 같고, 이 계수로 계산한 값이
#     eq5dsuite-value-sets(MathsInHealth, EQ-5D-3L/KR.csv·GB.csv, EQ-5D-5L/KR.csv)의 상태별 값 표와
#     243 + 243 + 3,125개 상태 전부에서 일치한다(서로 다른 두 공개 구현).
#   - 한국 3L: Lee 2009 원문(ScienceDirect, 요약 도구로 읽음)의 Table 3 계수와 계산 예(32322 → 0.148)와 일치.
#   - 한국 5L(Kim 2016), 영국 3L(Dolan 1997): 원 논문의 표는 열지 못했다. 본문에도 그렇게 적는다.
KR3 = {"const": 0.050, "n3": 0.050,
       "MO": [0, 0.096, 0.418], "SC": [0, 0.046, 0.136], "UA": [0, 0.051, 0.208],
       "PD": [0, 0.037, 0.151], "AD": [0, 0.043, 0.158]}
KR5 = {"const": 0.096, "n4": 0.078,
       "MO": [0, 0.046, 0.058, 0.133, 0.251], "SC": [0, 0.032, 0.050, 0.078, 0.122],
       "UA": [0, 0.021, 0.051, 0.100, 0.175], "PD": [0, 0.042, 0.053, 0.166, 0.207],
       "AD": [0, 0.033, 0.046, 0.102, 0.137]}
# 영국 EQ-5D-3L (Dolan 1997, MVH N3 모형). 계수는 R 패키지 eq5d의 자료 파일(TTO.csv, UK 열)에서 확인
UK3 = {"const": 0.081, "n3": 0.269,
       "MO": [0, 0.069, 0.314], "SC": [0, 0.104, 0.214], "UA": [0, 0.036, 0.094],
       "PD": [0, 0.123, 0.386], "AD": [0, 0.071, 0.236]}
DIMS = ["MO", "SC", "UA", "PD", "AD"]


def eq3(state, vs):
    lv = [int(c) for c in str(state)]
    if all(v == 1 for v in lv):
        return 1.0
    u = 1 - vs["const"] - sum(vs[d][v - 1] for d, v in zip(DIMS, lv))
    if any(v == 3 for v in lv):
        u -= vs["n3"]
    return u


def eq5(state, vs=KR5):
    lv = [int(c) for c in str(state)]
    if all(v == 1 for v in lv):
        return 1.0
    u = 1 - vs["const"] - sum(vs[d][v - 1] for d, v in zip(DIMS, lv))
    if any(v >= 4 for v in lv):
        u -= vs["n4"]
    return u


# P4_FACTS.md의 계산 예와 일치하는지 확인
assert abs(eq3("11121", KR3) - 0.913) < 1e-9 and abs(eq3("21232", KR3) - 0.559) < 1e-9
assert abs(eq3("33333", KR3) - (-0.171)) < 1e-9
assert abs(eq3("32322", KR3) - 0.148) < 1e-9     # Lee 2009 본문의 계산 예
assert abs(eq5("11121") - 0.862) < 1e-9 and abs(eq5("21222") - 0.762) < 1e-9
assert abs(eq5("12233") - 0.752) < 1e-9 and abs(eq5("31241") - 0.581) < 1e-9
assert abs(eq5("55555") - (-0.066)) < 1e-9 and abs(eq5("11211") - 0.883) < 1e-9
assert abs(eq3("33333", UK3) - (-0.594)) < 1e-9 and abs(eq3("11112", UK3) - 0.848) < 1e-9

N["eq"] = {
    "kr5_21222": eq5("21222"), "kr5_11121": eq5("11121"), "kr5_11211": eq5("11211"),
    "kr5_31241": eq5("31241"), "kr5_55555": eq5("55555"), "kr5_11111": eq5("11111"),
    "kr3_21232": eq3("21232", KR3), "kr3_21222": eq3("21222", KR3), "uk3_21222": eq3("21222", UK3),
    "kr3_33333": eq3("33333", KR3), "uk3_33333": eq3("33333", UK3),
    "kr3_11121": eq3("11121", KR3), "uk3_11121": eq3("11121", UK3),
    "n3L": 3 ** 5, "n5L": 5 ** 5,
}

# ------------------------------------------------------------------ 3. 예시 환자
VISITS = np.array([0, 3, 6, 12, 18, 24], float)

# 환자 가: 신약 A, 24개월 동안 무진행. 환자 나: 표준요법 B, 9개월째 진행, 24개월까지 생존
# 환자 다: 표준요법 B, 7개월째 진행, 15개월째 사망
EX = {
    "가": {"states": ["11121", "11221", "11121", "11121", "11221", "11121"], "t": VISITS.tolist(), "death": None},
    "나": {"states": ["11121", "11122", "21222", "21241", "21242", "31342"], "t": VISITS.tolist(), "death": None},
    "다": {"states": ["11221", "21222", "21232", "31241"], "t": [0, 3, 6, 12], "death": 15.0},
}


def path(t, u, death=None, end=24.0):
    """측정 시점과 효용으로 (시간, 효용) 경로를 만든다. 사망하면 사망 시점의 효용을 0으로 두고 그 뒤는 0."""
    t, u = list(map(float, t)), list(map(float, u))
    if death is not None and death <= end:
        t, u = t + [float(death)], u + [0.0]
        if death < end:
            t, u = t + [end], u + [0.0]
    return np.array(t), np.array(u)


def qaly(t, u):
    """사다리꼴 공식. 시간 단위가 개월이므로 12로 나눠 연 단위로 바꾼다."""
    return float(np.trapezoid(u, t) / 12.0)


N["ex"] = {}
for k, e in EX.items():
    u = [eq5(s) for s in e["states"]]
    t, uu = path(e["t"], u, e["death"])
    steps = [{"t0": t[i], "t1": t[i + 1], "u0": uu[i], "u1": uu[i + 1],
              "area": (uu[i] + uu[i + 1]) / 2 * (t[i + 1] - t[i]) / 12} for i in range(len(t) - 1)]
    q = qaly(t, uu)
    assert abs(q - sum(s["area"] for s in steps)) < 1e-12
    ly = (e["death"] if e["death"] else 24.0) / 12
    N["ex"][k] = {"states": e["states"], "u": u, "t": t.tolist(), "path_u": uu.tolist(), "steps": steps,
                  "qaly": q, "ly": ly, "death": e["death"]}
# 환자 다: 마지막 측정값을 사망까지 유지한다고 볼 때(대안 규칙)
e = N["ex"]["다"]
alt_t = np.array([0, 3, 6, 12, 15, 15, 24], float)
alt_u = np.array(e["u"] + [e["u"][-1], 0, 0], float)
N["ex"]["다"]["qaly_locf"] = qaly(alt_t, alt_u)

# 생존기간이 같고 독성이 다른 두 약(가 절 도입 예)
N["intro"] = {"years": 3, "u_x": 0.80, "u_y": 0.60, "q_x": 3 * 0.80, "q_y": 3 * 0.60}

# ------------------------------------------------------------------ 4. 가상의 임상시험
P = L.base_params()
# 씨앗값은 250개 후보 가운데 (1) 상태별 평균 효용이 모형 입력값(0.78, 0.62)에 가깝고
# (2) 기저 효용이 우연히 A군에서 약 0.02 높게 나온 것을 골랐다(다 절의 기저 효용 보정 설명용).
SEED = 20262348
N_ARM = 160
RHO = 0.7          # 진행 시간과 사망 시간의 상관(정규 코퓰라)
# 잠재 중증도에서 영역별 수준을 정하는 설정. 무진행 평균 약 0.78, 진행 후 약 0.62가 되게 맞췄다
MU_PF, MU_PD = 0.10, 1.07
SD_PT, SD_VIS, SD_DIM = 0.75, 0.45, 0.55
DIM_SHIFT = {"MO": -0.35, "SC": -1.05, "UA": -0.20, "PD": 0.25, "AD": -0.10}
CUTS = [-0.45, 0.95, 1.9, 2.9]   # 수준 2, 3, 4, 5의 경계


def simulate(seed=SEED, n_arm=N_ARM):
    rng = np.random.default_rng(seed)
    rows, pts = [], []
    pid = 0
    for arm in ("A", "B"):
        h_p = P["hr_pfs"] if arm == "A" else 1.0
        h_o = P["hr_os"] if arm == "A" else 1.0
        z = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], size=n_arm)
        uu = stats.norm.cdf(z)
        t_prog = (-np.log(uu[:, 0]) / (P["pfs_lam"] * h_p)) ** (1 / P["pfs_gam"])
        t_os = (-np.log(uu[:, 1]) / (P["os_lam"] * h_o)) ** (1 / P["os_gam"])
        b = rng.normal(0, SD_PT, n_arm)
        for i in range(n_arm):
            pid += 1
            prog = t_prog[i] if t_prog[i] < t_os[i] else np.inf    # 진행 없이 사망하면 진행 시점 없음
            death = t_os[i]
            pts.append({"id": pid, "arm": arm, "t_prog": prog, "t_death": death})
            for v in VISITS:
                if v >= death:
                    continue
                st = "PD" if v >= prog else "PF"
                zz = b[i] + (MU_PD if st == "PD" else MU_PF) + rng.normal(0, SD_VIS)
                lv = []
                for d in DIMS:
                    x = zz + DIM_SHIFT[d] + rng.normal(0, SD_DIM)
                    lv.append(1 + int(np.searchsorted(CUTS, x)))
                code = "".join(map(str, lv))
                u = eq5(code)
                vas = float(np.clip(np.round(100 * (0.25 + 0.62 * u) + rng.normal(0, 8)), 0, 100))
                rows.append({"id": pid, "arm": arm, "month": v, "state": st, "eq5d": code, "utility": u, "vas": vas})
    return pd.DataFrame(pts), pd.DataFrame(rows)


def analyse(pts, obs):
    out = {}
    out["n"] = {a: int((pts.arm == a).sum()) for a in "AB"}
    # 상태별 평균 효용(환자 내 상관을 고려한 군집 강건 표준오차)
    m = smf.ols("utility ~ C(state, Treatment('PF'))", obs).fit(cov_type="cluster", cov_kwds={"groups": obs["id"]})
    pf = m.params.iloc[0]
    d = m.params.iloc[1]
    ci = m.conf_int()
    cov = m.cov_params().values
    se_pd = math.sqrt(cov[0, 0] + cov[1, 1] + 2 * cov[0, 1])
    out["state"] = {
        "pf": pf, "pf_se": m.bse.iloc[0], "pf_lo": ci.iloc[0, 0], "pf_hi": ci.iloc[0, 1],
        "pd": pf + d, "pd_se": se_pd, "pd_lo": pf + d - 1.96 * se_pd, "pd_hi": pf + d + 1.96 * se_pd,
        "diff": d, "diff_lo": ci.iloc[1, 0], "diff_hi": ci.iloc[1, 1],
        "n_pf": int((obs.state == "PF").sum()), "n_pd": int((obs.state == "PD").sum()),
        "sd_pf": obs.utility[obs.state == "PF"].std(), "sd_pd": obs.utility[obs.state == "PD"].std(),
        "full_pf": float((obs.utility[obs.state == "PF"] == 1).mean()),
        "full_pd": float((obs.utility[obs.state == "PD"] == 1).mean()),
        "neg": int((obs.utility < 0).sum()), "min": float(obs.utility.min()),
    }
    # 군·시점별
    vis = {}
    for a in "AB":
        na = out["n"][a]
        for v in VISITS:
            s = obs[(obs.arm == a) & (obs.month == v)]
            vis[f"{a}{int(v)}"] = {"alive": len(s), "mean_alive": s.utility.mean(), "sd_alive": s.utility.std(),
                                   "mean_all": s.utility.sum() / na, "pd": int((s.state == "PD").sum()),
                                   "dead": na - len(s)}
    out["visit"] = vis
    # 기저 시점 요약
    b = obs[obs.month == 0]
    out["base"] = {
        "A": b.utility[b.arm == "A"].mean(), "B": b.utility[b.arm == "B"].mean(),
        "sdA": b.utility[b.arm == "A"].std(), "sdB": b.utility[b.arm == "B"].std(),
        "all": b.utility.mean(), "sd": b.utility.std(), "vas": b.vas.mean(), "vas_sd": b.vas.std(),
        "full": float((b.utility == 1).mean()),
        "prob": {d: float((b.eq5d.str[i].astype(int) > 1).mean()) for i, d in enumerate(DIMS)},
        "p_diff": float(stats.ttest_ind(b.utility[b.arm == "A"], b.utility[b.arm == "B"]).pvalue),
    }
    out["base"]["diff"] = out["base"]["A"] - out["base"]["B"]
    for a in "AB":
        ba = b[b.arm == a]
        out["base"]["arm" + a] = {
            "vas": ba.vas.mean(), "vas_sd": ba.vas.std(), "full": float((ba.utility == 1).mean()),
            "n_full": int((ba.utility == 1).sum()),
            "prob": {d: float((ba.eq5d.str[i].astype(int) > 1).mean()) for i, d in enumerate(DIMS)},
            "n_prob": {d: int((ba.eq5d.str[i].astype(int) > 1).sum()) for i, d in enumerate(DIMS)},
        }
    # 환자별 24개월 QALY와 생존연수
    q, q_d, ly, base = [], [], [], []
    grid = np.linspace(0, 24, 97)
    prof = {a: np.zeros_like(grid) for a in "AB"}
    for r in pts.itertuples():
        o = obs[obs.id == r.id]
        t, u = path(o.month.values, o.utility.values, r.t_death if r.t_death <= 24 else None)
        q.append(qaly(t, u))
        # 2년차 QALY를 4.5%로 할인(1년차는 할인하지 않음)
        t1, u1 = np.append(t[t < 12], 12.0), np.append(u[t < 12], np.interp(12.0, t, u))
        t2, u2 = np.insert(t[t > 12], 0, 12.0), np.insert(u[t > 12], 0, np.interp(12.0, t, u))
        q_d.append(qaly(t1, u1) + qaly(t2, u2) / (1 + P["disc"]))
        ly.append(min(r.t_death, 24.0) / 12)
        base.append(o.utility.values[0])
        prof[r.arm] += np.interp(grid, t, u)
    pts = pts.assign(qaly=q, qaly_d=q_d, ly=ly, base=base)
    for a in "AB":
        prof[a] /= out["n"][a]
    out["grid"] = grid.tolist()
    out["prof"] = {a: prof[a].tolist() for a in "AB"}
    qa, qb = pts.qaly[pts.arm == "A"], pts.qaly[pts.arm == "B"]
    out["qaly"] = {"A": qa.mean(), "B": qb.mean(), "sdA": qa.std(), "sdB": qb.std(),
                   "lyA": pts.ly[pts.arm == "A"].mean(), "lyB": pts.ly[pts.arm == "B"].mean(),
                   "dA": pts.qaly_d[pts.arm == "A"].mean(), "dB": pts.qaly_d[pts.arm == "B"].mean()}
    out["qaly"]["ly_diff"] = out["qaly"]["lyA"] - out["qaly"]["lyB"]
    out["qaly"]["d_diff"] = out["qaly"]["dA"] - out["qaly"]["dB"]
    pts["A"] = (pts.arm == "A").astype(int)
    m0 = smf.ols("qaly ~ A", pts).fit()
    m1 = smf.ols("qaly ~ A + base", pts).fit()
    for nm, mm in (("unadj", m0), ("adj", m1)):
        c = mm.conf_int().loc["A"]
        out["qaly"][nm] = {"diff": mm.params["A"], "se": mm.bse["A"], "lo": c[0], "hi": c[1], "p": mm.pvalues["A"]}
    out["qaly"]["b_base"] = m1.params["base"]
    out["qaly"]["b_base_se"] = m1.bse["base"]
    out["qaly"]["r_base"] = float(np.corrcoef(pts.base, pts.qaly)[0, 1])
    # 평균 곡선 아래 면적 = 환자별 QALY의 평균인지 확인(격자에 꺾이는 점이 다 들어 있지는 않으므로 근사)
    for a in "AB":
        auc = float(np.trapezoid(prof[a], grid) / 12)
        assert abs(auc - out["qaly"][a]) < 0.004, (a, auc, out["qaly"][a])
    out["events"] = {a: {"dead24": int(((pts.arm == a) & (pts.t_death <= 24)).sum()),
                         "prog24": int(((pts.arm == a) & (pts.t_prog <= 24)).sum()),
                         "pf24": int(((pts.arm == a) & (pts.t_prog > 24) & (pts.t_death > 24)).sum())} for a in "AB"}
    out["alive24"] = {a: float(((pts.arm == a) & (pts.t_death > 24)).mean() / (pts.arm == a).mean()) for a in "AB"}
    out["n_obs"] = len(obs)
    return out, pts


pts, obs = simulate()
T, pts = analyse(pts, obs)
N["trial"] = T
obs.to_csv(os.path.join(HERE, "_ch22_trial.csv"), index=False)
pts.to_csv(os.path.join(HERE, "_ch22_pts.csv"), index=False)

# ------------------------------------------------------------------ 5. 공통 예시 모형의 QALY 분해
r = L.run()
M = {}
for arm in L.ARMS:
    x = r[arm]
    tr = x["trace"]
    t = tr["t"]
    pf = (tr["pf"][:-1] + tr["pf"][1:]) / 2
    pd_ = (tr["pd"][:-1] + tr["pd"][1:]) / 2
    mid = (t[:-1] + L.CYCLE_MONTHS / 2) / 12.0
    d = 1.0 / (1.0 + P["disc"]) ** mid
    yr = L.CYCLE_MONTHS / 12.0
    lyd_pf, lyd_pd = float((pf * yr * d).sum()), float((pd_ * yr * d).sum())
    q_pf, q_pd = P["u_pf"] * lyd_pf, P["u_pd"] * lyd_pd
    du = P["du_ae_" + arm]
    assert abs(lyd_pf + lyd_pd - x["ly_d"]) < 1e-9 and abs(q_pf + q_pd - du - x["qaly"]) < 1e-9
    M[arm] = {"ly": x["ly"], "ly_pf": x["ly_pf"], "ly_pd": x["ly_pd"], "ly_d": x["ly_d"],
              "lyd_pf": lyd_pf, "lyd_pd": lyd_pd, "q_pf": q_pf, "q_pd": q_pd, "du": du,
              "qaly": x["qaly"], "qaly_undisc": x["qaly_undisc"],
              "qu_pf": P["u_pf"] * x["ly_pf"], "qu_pd": P["u_pd"] * x["ly_pd"]}
M["d"] = {k: M["A"][k] - M["B"][k] for k in M["A"]}
M["icer"], M["icer_ly"], M["d_cost"] = r["icer"], r["icer_ly"], r["d_cost"]
M["u_pf"], M["u_pd"] = P["u_pf"], P["u_pd"]
M["ratio"] = M["d"]["qaly"] / M["d"]["ly_d"]
# 이상반응 QALY 손실의 풀이 예: 발생률 × 효용 감소 × 지속 기간(년)
AE = {"du": 0.12, "months": 4, "pA": 0.30, "pB": 0.20}
AE["lossA"] = AE["pA"] * AE["du"] * AE["months"] / 12
AE["lossB"] = AE["pB"] * AE["du"] * AE["months"] / 12
assert abs(AE["lossA"] - P["du_ae_A"]) < 1e-12 and abs(AE["lossB"] - P["du_ae_B"]) < 1e-12
M["ae"] = AE
# 모형의 처음 24개월(시험의 추적 기간)에서 생기는 증분 QALY와 그 뒤의 외삽 구간에서 생기는 몫
_r24 = L.run(horizon=24)
M["d_qaly_24"] = _r24["d_qaly"]
M["d_qaly_after24"] = r["d_qaly"] - _r24["d_qaly"]
M["d_qaly_24_undisc"] = L.run(horizon=24, disc=0)["d_qaly"]
# 진행 상태의 효용을 0.62에서 0.45로 낮출 때(나 절 연습 문제): 증분 QALY는 진행 기간의 차이만큼 움직인다
_p45 = dict(P); _p45["u_pd"] = 0.45
_r45 = L.run(_p45)
M["upd45"] = {"d_qaly": _r45["d_qaly"], "icer": _r45["icer"]}
assert abs(_r45["d_qaly"] - (r["d_qaly"] - (P["u_pd"] - 0.45) * M["d"]["lyd_pd"])) < 1e-9
# 전체 생존은 늘지 않고 진행만 늦춰진다면(hr_os = 1): 늘어난 무진행 기간만큼 진행 기간이 줄어든다
_pos = dict(P); _pos["hr_os"] = 1.0
_ros = L.run(_pos)
M["os_same"] = {"d_qaly": _ros["d_qaly"], "d_ly": _ros["d_ly"], "icer": _ros["icer"]}
# 효용 입력값의 불확실성(24장에서 쓰는 분포)
M["se"] = {"u_pf": L.PSA_SPEC["u_pf"][1], "u_pd": L.PSA_SPEC["u_pd"][1]}
for k in ("u_pf", "u_pd"):
    m_, s_ = P[k], L.PSA_SPEC[k][1]
    n_ = m_ * (1 - m_) / s_ ** 2 - 1
    M["se"][k + "_a"], M["se"][k + "_b"] = m_ * n_, (1 - m_) * n_
    M["se"][k + "_lo"], M["se"][k + "_hi"] = [float(v) for v in stats.beta.ppf([0.025, 0.975], m_ * n_, (1 - m_) * n_)]
# 생존 연장 없이 삶의 질만 좋아지는 약의 QALY(연습 문제)
M["qonly"] = {"du": 0.10, "years": 2, "gain": 0.10 * 2}
# 할인(수식으로 보기): 3년 뒤의 1 QALY
M["disc3"] = 1 / (1 + P["disc"]) ** 3
N["model"] = M

# 연습 문제
PR = {}
PR["tto"] = 13 / 20
PR["vas"] = (50 - 10) / (100 - 10)
PR["eq_11122"] = eq5("11122")
_t, _u = np.array([0, 6, 12, 18.0]), np.array([0.80, 0.70, 0.60, 0.0])
PR["q1"] = {"qaly": qaly(_t, _u), "parts": [float((_u[i] + _u[i + 1]) / 2 * 6 / 12) for i in range(3)], "ly": 1.5}
PR["cost24"] = 1400.0
PR["icer_unadj"] = 1400.0 / 0.117
PR["icer_adj"] = 1400.0 / 0.094
N["practice"] = PR

# NICE 중증도 가중(연습 문제용): 절대 부족분, 비례 부족분
# 예: 질환이 없을 때 기대 QALY 12.0, 현재 치료에서 2.0 → 절대 부족분 10.0, 비례 부족분 0.833 → 가중치 1
# (P4_FACTS.md 3부: ×1.2는 비례 0.85~0.95 또는 절대 12~18, ×1.7은 비례 0.95 이상 또는 절대 18 이상)
N["sev"] = {"exp": 12.0, "cur": 2.0, "abs": 10.0, "prop": 10.0 / 12.0}

# ------------------------------------------------------------------ 6. 본문에 싣는 파이썬 코드와 실제 출력
_e = N["ex"]
_f = lambda xs: ", ".join(f"{v:.3f}" for v in xs)
CODE = f'''import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# 환자 다: 측정 시점(개월)과 그때의 효용. 15개월에 사망했으므로 그 시점의 효용은 0
t = np.array([0, 3, 6, 12, 15])
u = np.array([{_f(_e["다"]["u"])}, 0.0])

area = trapezoid(u, t)             # 곡선 아래 면적(단위: 효용 × 개월)
print("면적:", round(area, 3), "→ QALY:", round(area / 12, 3))

# 사다리꼴을 하나씩 계산해 같은 값인지 확인
width = np.diff(t)                 # 구간의 길이: 3, 3, 6, 3
height = (u[:-1] + u[1:]) / 2      # 구간 양 끝 효용의 평균
print("구간별 QALY:", np.round(width * height / 12, 3))

# 환자가 여러 명일 때: 긴 형식의 표(한 행 = 측정 한 번)에서 환자별로 계산
df = pd.DataFrame({{
    "id":    ["가"] * 6 + ["나"] * 6 + ["다"] * 5,
    "month": [0, 3, 6, 12, 18, 24] * 2 + [0, 3, 6, 12, 15],
    "utility": [{_f(_e["가"]["u"])},
                {_f(_e["나"]["u"])},
                {_f(_e["다"]["u"])}, 0.0],
}})
qaly = df.groupby("id")[["month", "utility"]].apply(
    lambda g: trapezoid(g["utility"], g["month"]) / 12)
print(qaly.round(3))
'''
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(CODE, {})
N["code"] = {"src": CODE, "out": buf.getvalue(), "numpy": np.__version__, "has_trapz": hasattr(np, "trapz"),
             "has_trapezoid": hasattr(np, "trapezoid")}
# 본문에 붙여 넣는 코드 셀(HTML). content/ch22.html의 '파이썬으로 계산해 보기'와 같아야 한다
from html import escape as _esc  # noqa: E402
CELL = ('<div class="cell"><div class="cell-h"><span class="cell-n">직접 실행</span><span class="cell-t">사다리꼴 공식으로 QALY 구하기</span>'
        '<button class="copy" type="button" aria-label="코드 복사">복사</button></div><pre class="cell-in"><code class="language-python">'
        + _esc(CODE.rstrip("\n")) + '</code></pre><div class="cell-out"><div class="cell-out-h">출력</div><pre>'
        + _esc(buf.getvalue().rstrip("\n")) + '</pre></div></div>')
open(os.path.join(HERE, "_ch22_cell.html"), "w", encoding="utf-8").write(CELL)
# 본문 코드는 scipy.integrate.trapezoid를 쓴다(SciPy 1.6 이상, numpy 버전과 무관). numpy 2.0 이상의 np.trapezoid와 같은 값인지 확인
import scipy  # noqa: E402
from scipy.integrate import trapezoid as sp_trap  # noqa: E402
assert abs(sp_trap(N["ex"]["다"]["u"] + [0.0], [0, 3, 6, 12, 15]) / 12 - N["ex"]["다"]["qaly"]) < 1e-12
N["code"]["scipy"] = scipy.__version__


# ------------------------------------------------------------------ 본문에 적힌 유도값의 검산
r3 = lambda v: round(v + 1e-12, 3)
assert r3(N["eq"]["kr3_11121"] - N["eq"]["uk3_11121"]) == 0.117 and r3(N["eq"]["kr3_21222"] - N["eq"]["uk3_21222"]) == 0.103
assert r3(N["eq"]["kr3_33333"] - N["eq"]["uk3_33333"]) == 0.423 and r3(eq5("31342")) == 0.518
assert r3(N["ex"]["가"]["qaly"]) == 1.708 and r3(N["ex"]["나"]["qaly"]) == 1.307 and r3(N["ex"]["다"]["qaly"]) == 0.795
assert round(N["ex"]["가"]["qaly"] - N["ex"]["나"]["qaly"], 2) == 0.40 and r3(N["ex"]["다"]["qaly_locf"]) == 0.868
# 본문에 적는 값이 반올림 경계(…5)에 걸리지 않는지 확인
for _k in "가나다":
    _v = N["ex"][_k]["qaly"] * 1000
    assert abs(_v - math.floor(_v) - 0.5) > 0.05, _k
for _s in N["ex"]["다"]["steps"]:
    _v = _s["area"] * 1000
    assert abs(_v - math.floor(_v) - 0.5) > 0.05, _s
_q = T["qaly"]
assert r3(_q["unadj"]["diff"]) == 0.117 and r3(_q["adj"]["diff"]) == 0.094
assert abs(_q["unadj"]["diff"] - _q["adj"]["diff"] - _q["b_base"] * T["base"]["diff"]) < 1e-9   # 보정으로 줄어든 크기
assert r3(_q["b_base"] * T["base"]["diff"]) == 0.023 and round(_q["ly_diff"] * 365) == 25
assert round(T["alive24"]["A"] * 100 + 1e-9) == 66 and round(T["alive24"]["B"] * 100 + 1e-9) == 58
_m = N["model"]
assert r3(_m["A"]["q_pf"] + _m["A"]["q_pd"] - _m["A"]["du"]) == 2.385 and r3(_m["B"]["q_pf"] + _m["B"]["q_pd"] - _m["B"]["du"]) == 1.874
assert r3(_m["d"]["q_pf"]) == 0.486 and r3(_m["d"]["q_pd"]) == 0.029 and r3(_m["d"]["qaly"]) == 0.511 and r3(_m["d"]["ly_d"]) == 0.670
assert round(_m["d"]["ly"], 2) == 0.86 and round(_m["icer"]) == 5621 and round(_m["icer_ly"]) == 4287 and round(_m["d_cost"]) == 2872
assert round(_m["d_qaly_24"], 2) == 0.09 and round(_m["d_qaly_after24"], 2) == 0.42
assert round(N["practice"]["icer_unadj"]) == 11966 and round(N["practice"]["icer_adj"]) == 14894
assert r3(_m["os_same"]["d_qaly"]) == 0.096 and abs(_m["os_same"]["d_ly"]) < 1e-9
assert r3(_m["upd45"]["d_qaly"]) == 0.503 and round(_m["upd45"]["icer"]) == 5710 and r3(_m["d"]["lyd_pd"]) == 0.047


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(clean(N), open(os.path.join(HERE, "_ch22_nums.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    f3 = lambda v: f"{v:.3f}"
    print("== 직접 측정 ==", N["sg"], N["tto"], N["vas"])
    print("== EQ-5D ==", {k: round(v, 3) for k, v in N["eq"].items()})
    print("== 예시 환자 ==")
    for k, e in N["ex"].items():
        print(k, e["states"], [f3(v) for v in e["u"]], "QALY", f3(e["qaly"]), "LY", e["ly"],
              "steps", [f3(s["area"]) for s in e["steps"]], "locf", e.get("qaly_locf"))
    print("== 임상시험 ==")
    s = T["state"]
    print("관측", T["n_obs"], "PF n", s["n_pf"], "PD n", s["n_pd"])
    print(f"PF {s['pf']:.4f} (SE {s['pf_se']:.4f}, {s['pf_lo']:.3f}-{s['pf_hi']:.3f}) SD {s['sd_pf']:.3f} 완전건강 {s['full_pf']:.3f}")
    print(f"PD {s['pd']:.4f} (SE {s['pd_se']:.4f}, {s['pd_lo']:.3f}-{s['pd_hi']:.3f}) SD {s['sd_pd']:.3f} 완전건강 {s['full_pd']:.3f}")
    print(f"차이 {s['diff']:.4f} ({s['diff_lo']:.3f}, {s['diff_hi']:.3f}); 음수 {s['neg']}건, 최솟값 {s['min']:.3f}")
    b = T["base"]
    print("기저:", {k: round(v, 4) for k, v in b.items() if not isinstance(v, dict)}, {k: round(v, 3) for k, v in b["prob"].items()})
    for a in "AB":
        print(a, "사건", T["events"][a])
        for v in VISITS:
            x = T["visit"][f"{a}{int(v)}"]
            print(f"   {int(v):>2}개월 생존 {x['alive']:>3} (진행 {x['pd']:>3}, 사망 {x['dead']:>3}) 생존자 평균 {x['mean_alive']:.3f} (SD {x['sd_alive']:.3f}) 전체 평균(사망=0) {x['mean_all']:.3f}")
    q = T["qaly"]
    print(f"QALY A {q['A']:.4f} (SD {q['sdA']:.3f}) B {q['B']:.4f} (SD {q['sdB']:.3f}); LY A {q['lyA']:.4f} B {q['lyB']:.4f} 차이 {q['ly_diff']:.4f}")
    for nm in ("unadj", "adj"):
        x = q[nm]
        print(f"   {nm}: {x['diff']:.4f} (SE {x['se']:.4f}; {x['lo']:.3f}, {x['hi']:.3f}) p {x['p']:.4f}")
    print(f"   기저 효용 계수 {q['b_base']:.3f} (SE {q['b_base_se']:.3f}), 상관 {q['r_base']:.3f}; 할인 QALY A {q['dA']:.4f} B {q['dB']:.4f} 차이 {q['d_diff']:.4f}")
    print("== 모형 ==")
    for a in ("A", "B", "d"):
        print(a, {k: round(v, 4) for k, v in M[a].items()})
    print("ICER", round(M["icer"]), "ICER/LY", round(M["icer_ly"]), "증분비용", round(M["d_cost"]), "QALY/LY 비", round(M["ratio"], 3))
    print("AE", M["ae"], "SE", {k: round(v, 3) for k, v in M["se"].items()}, "disc3", round(M["disc3"], 4))
    print("기저(군별):", {a: T["base"]["arm" + a] for a in "AB"})
    print("연습:", N["practice"])
    print("24개월 생존 비율", T["alive24"], "생존자 평균×2", T["visit"]["A24"]["mean_alive"] * 2, "LY 차이(일)", q["ly_diff"] * 365)
    print("u_pd 0.45:", M["upd45"], "hr_os 1:", M["os_same"], "효용 차 × 무진행 기간 차", (P["u_pf"] - P["u_pd"]) * M["d"]["lyd_pf"])
    print("== 코드 출력 ==")
    print(N["code"]["out"], N["code"]["numpy"], N["code"]["has_trapz"], N["code"]["has_trapezoid"])
