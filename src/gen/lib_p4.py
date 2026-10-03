"""PART 4 (약물경제성 평가, 20–25장) 공통 예시 모형.

가상의 급여 신청 사례: 진행성 신세포암 1차 치료에서 신약 A와 표준요법 B의 비교.
세 건강상태(무진행 PF, 진행 PD, 사망 D)의 분할생존모형(partitioned survival model).
모든 값은 설명을 위해 지어낸 것이며 실제 약제의 자료가 아니다.

20–25장은 이 파일의 기준 분석(base case) 숫자를 같이 쓴다. 기준 입력값(base_params)은
바꾸지 않는다. 함수 추가는 23장(모형)·24장(불확실성) 담당만 한다.

확률적 민감도 분석에서 입력값을 뽑는 방법(2026-10-03 고침. 기준 입력값과 기준 분석 결과는 그대로):
  곡선의 불확실성은 (중앙 생존기간, 모양 모수)로 적고(PSA_SPEC의 pfs_med, os_med, pfs_gam, os_gam),
  무진행생존과 전체생존의 중앙값끼리·모양끼리·위험비끼리 상관 0.5로 함께 뽑으며(PSA_CORR),
  분석기간 안에 무진행생존 곡선이 전체생존 곡선을 넘는 벌은 버리고 다시 뽑는다(draw, curves_cross).
  쓰는 법: psa(n, seed) → (증분비용, 증분QALY) 배열, psa_inputs(n, seed) → (입력값 n벌, 다시 뽑기 기록),
           get_input / set_input(p, key, value) → 입력값 표의 표현(중앙값·모양)으로 읽고 바꾸기.

단위: 시간 = 개월(주기 1개월), 비용 = 만원, 할인율 = 연 4.5%(비용·효과 동일).
생존함수: Weibull 비례위험 꼴 S(t) = exp(-lam * t**gam). 위험비(HR)는 lam에 곱한다.
"""
import math
import numpy as np

ARMS = ("A", "B")
CYCLE_MONTHS = 1.0
HORIZON_MONTHS = 240          # 20년. 이 시점에 두 군 모두 생존 1% 미만
THRESHOLD = 5000.0            # 예시에서 가정한 비용효과 임계값(만원/QALY)


def weib_lam(median, gam):
    """중앙값(개월)과 모양모수로 lam 계산: S(median) = 0.5."""
    return math.log(2) / median ** gam


def base_params():
    """기준 분석 입력값. 평균과, 확률적 민감도 분석에 쓰는 분포 정보(se, dist)를 함께 둔다."""
    return {
        # 표준요법 B의 생존곡선(Weibull). 중앙 무진행생존 10개월, 중앙 전체생존 28개월
        "pfs_gam": 0.95, "pfs_lam": weib_lam(10.0, 0.95),
        "os_gam": 1.15, "os_lam": weib_lam(28.0, 1.15),
        # 신약 A의 상대효과(위험비, B 대비)
        "hr_pfs": 0.65, "hr_os": 0.75,
        # 효용(상태별), 이상반응으로 인한 1회성 QALY 손실
        "u_pf": 0.78, "u_pd": 0.62,
        "du_ae_A": 0.012, "du_ae_B": 0.008,
        # 비용(만원). 약값은 무진행 상태에 머무는 동안 매달 든다(진행 시까지 투여)
        "c_drug_A": 190.0, "c_drug_B": 120.0,
        "c_pf": 40.0,        # 무진행 상태 월 관리비(외래, 검사)
        "c_pd": 250.0,       # 진행 상태 월 비용(후속치료 포함)
        "c_death": 800.0,    # 사망 직전 임종기 비용(1회)
        "c_ae_A": 120.0, "c_ae_B": 80.0,   # 이상반응 치료비(1회)
        "disc": 0.045,
    }


# 확률적 민감도 분석용 분포. (분포, 불확실성 크기)
#   lognormal: 로그 척도의 표준오차 / beta: 표준오차 / gamma: 표준오차 = 평균 × 비율
# 생존곡선(표준요법 B의 와이블 곡선)의 불확실성은 척도 모수 lam이 아니라 중앙 생존기간과 모양 모수로 적는다.
#   pfs_med, os_med: 중앙 무진행생존·전체생존기간(개월). base_params()에는 없는 key이고 get_input()이 lam, gam에서 계산한다
#   (기준값 10, 28개월). 뽑은 중앙값과 모양 모수로 lam = weib_lam(중앙값, 모양)을 다시 계산한다.
#   표준오차의 근거(gen/nums_ch24.py에서 검산): 23장 라 절의 가상 시험(표준요법 B군 300명, 추적 24–30개월)에 와이블을 적합하면
#   전체생존은 ln 중앙값의 표준오차 0.074, ln 모양의 표준오차 0.077이고, 같은 설계의 무진행생존은 0.069, 0.053이다.
#   두 군(각 300명)을 함께 적합하면 모양의 표준오차는 0.058, 0.040으로 줄어든다. 이 사이의 값으로 정했다.
PSA_SPEC = {
    "hr_pfs": ("lognormal", 0.113),   # 95% CI 약 0.52–0.81
    "hr_os": ("lognormal", 0.131),    # 95% CI 약 0.58–0.97
    "pfs_med": ("lognormal", 0.07), "pfs_gam": ("lognormal", 0.05),
    "os_med": ("lognormal", 0.08), "os_gam": ("lognormal", 0.06),
    "u_pf": ("beta", 0.03), "u_pd": ("beta", 0.05),
    "du_ae_A": ("gamma", 0.25), "du_ae_B": ("gamma", 0.25),
    "c_pf": ("gamma", 0.20), "c_pd": ("gamma", 0.20), "c_death": ("gamma", 0.20),
    "c_ae_A": ("gamma", 0.25), "c_ae_B": ("gamma", 0.25),
}
# 약값(c_drug_A, c_drug_B)과 할인율은 확률분포를 주지 않고 시나리오·임계값 분석에서 바꾼다.

# 함께 뽑는 짝과 로그 척도에서의 상관계수. 무진행생존과 전체생존은 같은 환자에게서 추정하므로
# 중앙값끼리, 모양 모수끼리, 위험비끼리 양의 상관이 있다(진행이 늦은 환자가 대체로 오래 산다).
#   근거(gen/nums_ch24.py에서 검산): 진행 시간과 사망 시간의 상관이 0.5–0.7(22장의 가상 시험은 0.7)인 군당 300명의 시험을
#   600번씩 만들어 적합하면 추정값 사이의 상관은 중앙값 0.51–0.61, 모양 0.40–0.45, 위험비 0.56–0.64이다. 어림해 모두 0.5로 둔다.
PSA_CORR = {("pfs_med", "os_med"): 0.5, ("pfs_gam", "os_gam"): 0.5, ("hr_pfs", "hr_os"): 0.5}

_CURVE_MEDIAN = {"pfs_med": ("pfs_lam", "pfs_gam"), "os_med": ("os_lam", "os_gam")}
_CURVE_SHAPE = {"pfs_gam": "pfs_lam", "os_gam": "os_lam"}


def get_input(p, key):
    """입력값 표의 표현으로 값 하나를 읽는다. pfs_med, os_med(중앙값, 개월)는 lam과 gam에서 계산하고 나머지는 p[key]."""
    if key in _CURVE_MEDIAN:
        k_lam, k_gam = _CURVE_MEDIAN[key]
        return (math.log(2) / p[k_lam]) ** (1.0 / p[k_gam])
    return p[key]


def set_input(p, key, value):
    """입력값 dict p를 복사해 입력값 표의 표현으로 key 하나를 value로 바꾼 새 dict를 돌려준다(원래 dict는 그대로).

    곡선은 (중앙값, 모양 모수)로 다룬다:
      pfs_med, os_med : 모양 모수는 그대로 두고 중앙값이 value가 되도록 lam을 다시 계산한다.
      pfs_gam, os_gam : 중앙값은 그대로 두고 모양 모수를 value로 바꾼다(lam을 다시 계산한다).
    그 밖의 key는 set_param()과 같다. lam을 직접 바꾸려면 set_param(p, "os_lam", …)을 쓴다.
    """
    q = dict(p)
    if key in _CURVE_MEDIAN:
        k_lam, k_gam = _CURVE_MEDIAN[key]
        q[k_lam] = weib_lam(value, q[k_gam])
    elif key in _CURVE_SHAPE:
        med = get_input(p, key[:-3] + "med")
        q[key] = value
        q[_CURVE_SHAPE[key]] = weib_lam(med, value)
    else:
        q[key] = value
    return q


def surv(lam, gam, t):
    return np.exp(-lam * np.asarray(t, float) ** gam)


def curves_raw(p, arm, t):
    """군별 무진행생존 S_PFS(t), 전체생존 S_OS(t)를 자르지 않고 돌려준다(두 곡선이 엇갈리는지 볼 때 쓴다)."""
    h_pfs = p["hr_pfs"] if arm == "A" else 1.0
    h_os = p["hr_os"] if arm == "A" else 1.0
    s_os = surv(p["os_lam"] * h_os, p["os_gam"], t)
    return surv(p["pfs_lam"] * h_pfs, p["pfs_gam"], t), s_os


def curves(p, arm, t):
    """군별 무진행생존 S_PFS(t), 전체생존 S_OS(t). 무진행생존은 전체생존을 넘지 못하게 자른다."""
    s_pfs, s_os = curves_raw(p, arm, t)
    return np.minimum(s_pfs, s_os), s_os


def curves_cross(p, horizon=HORIZON_MONTHS):
    """분석기간 안의 주기 경계 시점(0, 1, …, horizon개월)에서 어느 군에서든 무진행생존 곡선이 전체생존 곡선을 넘으면 True.
    False이면 curves()의 np.minimum이 값을 하나도 바꾸지 않는다."""
    t = np.arange(0, horizon + 1) * CYCLE_MONTHS
    for arm in ARMS:
        s_pfs, s_os = curves_raw(p, arm, t)
        if (s_pfs > s_os).any():
            return True
    return False


def psm(p, arm, horizon=HORIZON_MONTHS, half_cycle=True, disc=None):
    """분할생존모형 한 군의 결과.

    상태 점유: PF = S_PFS, PD = S_OS - S_PFS, D = 1 - S_OS.
    한 주기의 점유는 주기 시작과 끝의 평균(반주기 보정). 할인은 주기 중간 시점 기준.
    """
    r = p["disc"] if disc is None else disc
    t = np.arange(0, horizon + 1) * CYCLE_MONTHS
    s_pfs, s_os = curves(p, arm, t)
    pf_t, pd_t = s_pfs, s_os - s_pfs
    if half_cycle:
        pf = (pf_t[:-1] + pf_t[1:]) / 2
        pd_ = (pd_t[:-1] + pd_t[1:]) / 2
    else:
        pf, pd_ = pf_t[:-1], pd_t[:-1]
    deaths = s_os[:-1] - s_os[1:]
    mid = (t[:-1] + CYCLE_MONTHS / 2) / 12.0          # 주기 중간(년)
    d = 1.0 / (1.0 + r) ** mid
    yr = CYCLE_MONTHS / 12.0
    ly_pf, ly_pd = (pf * yr), (pd_ * yr)
    out = {
        "arm": arm,
        "ly_pf": ly_pf.sum(), "ly_pd": ly_pd.sum(), "ly": (ly_pf + ly_pd).sum(),
        "ly_d": ((ly_pf + ly_pd) * d).sum(),
        "qaly": ((ly_pf * p["u_pf"] + ly_pd * p["u_pd"]) * d).sum() - p["du_ae_" + arm],
        "qaly_undisc": (ly_pf * p["u_pf"] + ly_pd * p["u_pd"]).sum() - p["du_ae_" + arm],
        "c_drug": (pf * p["c_drug_" + arm] * d).sum(),
        "c_pf": (pf * p["c_pf"] * d).sum(),
        "c_pd": (pd_ * p["c_pd"] * d).sum(),
        "c_death": (deaths * p["c_death"] * d).sum(),
        "c_ae": p["c_ae_" + arm],
        "trace": {"t": t, "pf": pf_t, "pd": pd_t, "dead": 1 - s_os},
    }
    out["cost"] = out["c_drug"] + out["c_pf"] + out["c_pd"] + out["c_death"] + out["c_ae"]
    return out


def run(p=None, **kw):
    """두 군을 돌리고 증분 결과를 붙인다. ICER 단위: 만원/QALY."""
    p = base_params() if p is None else p
    a, b = psm(p, "A", **kw), psm(p, "B", **kw)
    dc, dq, dl = a["cost"] - b["cost"], a["qaly"] - b["qaly"], a["ly_d"] - b["ly_d"]
    return {"A": a, "B": b, "d_cost": dc, "d_qaly": dq, "d_ly": dl,
            "icer": dc / dq if dq != 0 else float("nan"),
            "icer_ly": dc / dl if dl != 0 else float("nan")}


def nmb(res, lam=THRESHOLD):
    """증분 순금전편익(incremental net monetary benefit) = lam × 증분QALY − 증분비용."""
    return lam * res["d_qaly"] - res["d_cost"]


def draw_survival(rng, p=None):
    """생존곡선과 위험비 한 벌을 뽑는다(곡선이 엇갈리는지는 보지 않는다). 다른 입력값은 p의 값 그대로.

    PSA_CORR의 짝(중앙값, 모양 모수, 위험비)마다 표준정규 난수 두 개 z1, z2를 뽑아
    둘째를 rho × z1 + sqrt(1 − rho²) × z2로 바꾼 뒤(2 × 2 촐레스키 분해), 로그 척도에서 기준값에 더한다.
    뽑은 중앙값과 모양 모수로 lam을 다시 계산한다."""
    p = dict(base_params() if p is None else p)
    val = {}
    for (k1, k2), rho in PSA_CORR.items():
        z1, z2 = rng.normal(size=2)
        z2 = rho * z1 + math.sqrt(1.0 - rho ** 2) * z2
        val[k1] = get_input(p, k1) * math.exp(PSA_SPEC[k1][1] * z1)
        val[k2] = get_input(p, k2) * math.exp(PSA_SPEC[k2][1] * z2)
    for k in ("hr_pfs", "hr_os", "pfs_gam", "os_gam"):
        p[k] = float(val[k])
    p["pfs_lam"] = weib_lam(val["pfs_med"], val["pfs_gam"])
    p["os_lam"] = weib_lam(val["os_med"], val["os_gam"])
    return p


def draw(rng, p=None, count=None):
    """PSA_SPEC에 따라 입력값 한 벌을 뽑는다.

    1) 생존곡선과 위험비: draw_survival()로 뽑고, 분석기간 안에 어느 군에서든 무진행생존 곡선이 전체생존 곡선을 넘으면
       (curves_cross) 그 벌을 버리고 다시 뽑는다. count(dict)를 주면 count["tries"](뽑은 횟수)와 count["rejected"](버린 횟수)에 더한다.
    2) 효용, 비용, QALY 손실: 서로 독립으로 뽑는다.
    """
    base = dict(base_params() if p is None else p)
    while True:
        p = draw_survival(rng, base)
        if count is not None:
            count["tries"] = count.get("tries", 0) + 1
        if not curves_cross(p):
            break
        if count is not None:
            count["rejected"] = count.get("rejected", 0) + 1
    paired = {k for pair in PSA_CORR for k in pair}
    for k, (dist, u) in PSA_SPEC.items():
        if k in paired:
            continue
        m = p[k]
        if dist == "lognormal":
            p[k] = float(np.exp(rng.normal(math.log(m), u)))
        elif dist == "beta":
            n = m * (1 - m) / u ** 2 - 1
            p[k] = float(rng.beta(m * n, (1 - m) * n))
        elif dist == "gamma":
            se = m * u
            shape = (m / se) ** 2
            p[k] = float(rng.gamma(shape, m / shape))
    return p


def psa_inputs(n=5000, seed=20261002, p=None):
    """psa()와 같은 난수로 뽑은 입력값 n벌(dict의 list)과 다시 뽑기 기록을 돌려준다.
    기록 dict: tries(뽑은 횟수 = n + rejected), rejected(곡선이 엇갈려 버린 횟수), share(= rejected ÷ tries)."""
    rng = np.random.default_rng(seed)
    count = {"tries": 0, "rejected": 0}
    draws = [draw(rng, p, count) for _ in range(n)]
    count["share"] = count["rejected"] / count["tries"]
    return draws, count


def psa(n=5000, seed=20261002, p=None):
    """확률적 민감도 분석: n벌의 (증분비용, 증분QALY)."""
    rng = np.random.default_rng(seed)
    dc, dq = np.empty(n), np.empty(n)
    for i in range(n):
        r = run(draw(rng, p))
        dc[i], dq[i] = r["d_cost"], r["d_qaly"]
    return dc, dq



# ======================================================================================
# 23장(결정분석 모형)에서 추가한 함수. 위의 기준 분석 함수와 입력값은 그대로 두고 덧붙인 것이다.
#   결정수형      : ae_tree_params, tree_rollback, tree_paths, ae_tree
#   율과 확률     : rate_to_prob, prob_to_rate, weib_median
#   상태 점유 → 결과: occupancy_result (psm과 같은 계산을 임의의 상태 점유에 적용)
#   마르코프 모형 : markov_matrix, markov_trace, markov_inputs, markov, run_markov
#   곡선 바꿔 넣기: psm_curves, run_os, lifelines_weibull_to_lib
# ======================================================================================

STATES = ("PF", "PD", "D")


def ae_tree_params():
    """결정수형 예시(신약 A 투여 초기의 중증 이상반응과 예방 요법)의 입력값. 가상의 값.

    '예방 요법 없음' 전략의 기대값이 base_params()의 c_ae_A(120만원), du_ae_A(0.012 QALY)와
    같아지도록 정했다: 0.30 × (0.40 × 700 + 0.60 × 200) = 120, 0.30 × (0.40 × 0.07 + 0.60 × 0.02) = 0.012.
    """
    return {
        "p_ae": 0.30,       # 예방 요법 없이 신약 A를 쓸 때 중증 이상반응이 생길 확률
        "rr_proph": 0.60,   # 예방 요법의 상대위험도(중증 이상반응)
        "p_hosp": 0.40,     # 중증 이상반응이 생겼을 때 입원할 확률(나머지는 외래 치료)
        "c_hosp": 700.0, "c_out": 200.0,     # 입원, 외래 치료의 비용(만원)
        "dq_hosp": 0.07, "dq_out": 0.02,     # 입원, 외래 치료에 따른 QALY 손실
        "c_proph": 60.0,    # 예방 요법의 비용(만원)
    }


def tree_rollback(node):
    """결정수형의 한 가지(전략)를 뒤에서부터 접어(roll back) 기대값을 구한다.

    node는 dict. 종결 마디는 {"cost": 비용, "qaly": QALY(또는 QALY 변화)},
    확률 마디는 {"branches": [(확률, 자식 마디), ...]} (확률의 합은 1).
    선택적으로 "name"을 둘 수 있다. 반환값: (기대비용, 기대QALY).
    """
    if "branches" not in node:
        return float(node["cost"]), float(node["qaly"])
    ps = [pr for pr, _ in node["branches"]]
    if abs(sum(ps) - 1.0) > 1e-9:
        raise ValueError("확률 마디에서 나가는 가지의 확률 합이 1이 아닙니다: %r" % (ps,))
    c = q = 0.0
    for pr, child in node["branches"]:
        cc, qq = tree_rollback(child)
        c += pr * cc
        q += pr * qq
    return c, q


def tree_paths(node, prob=1.0, names=()):
    """결정수형의 한 전략에서 종결 마디까지의 모든 경로를 [(경로 이름들, 경로 확률, 비용, QALY)]로 돌려준다.
    경로 확률은 지나온 가지의 확률을 모두 곱한 값이고, 한 전략의 경로 확률을 모두 더하면 1이다."""
    names = names + ((node["name"],) if node.get("name") else ())
    if "branches" not in node:
        return [(names, prob, float(node["cost"]), float(node["qaly"]))]
    out = []
    for pr, child in node["branches"]:
        out += tree_paths(child, prob * pr, names)
    return out


def ae_tree(tp=None):
    """중증 이상반응 예방 요법의 결정수형. 두 전략('none' 예방 요법 없음, 'proph' 예방 요법)의
    나무(tree), 경로(paths), 기대비용(cost)·기대 QALY 변화(qaly; 이상반응이 없을 때 대비, 음수)와
    증분 결과(d_cost, d_qaly, icer; proph − none)를 돌려준다. 비용 단위 만원."""
    tp = ae_tree_params() if tp is None else tp

    def strategy(p_ae, c_add):
        return {"branches": [
            (p_ae, {"name": "중증 이상반응", "branches": [
                (tp["p_hosp"], {"name": "입원", "cost": c_add + tp["c_hosp"], "qaly": -tp["dq_hosp"]}),
                (1 - tp["p_hosp"], {"name": "외래 치료", "cost": c_add + tp["c_out"], "qaly": -tp["dq_out"]}),
            ]}),
            (1 - p_ae, {"name": "중증 이상반응 없음", "cost": c_add, "qaly": 0.0}),
        ]}

    trees = {"none": strategy(tp["p_ae"], 0.0),
             "proph": strategy(tp["p_ae"] * tp["rr_proph"], tp["c_proph"])}
    out = {}
    for k, tr in trees.items():
        c, q = tree_rollback(tr)
        out[k] = {"tree": tr, "paths": tree_paths(tr), "cost": c, "qaly": q}
    out["d_cost"] = out["proph"]["cost"] - out["none"]["cost"]
    out["d_qaly"] = out["proph"]["qaly"] - out["none"]["qaly"]
    out["icer"] = out["d_cost"] / out["d_qaly"] if out["d_qaly"] != 0 else float("nan")
    return out


def rate_to_prob(r, t=1.0):
    """율(rate, 단위 시간당 사건 수) r이 일정할 때 길이 t인 기간 안에 사건이 일어날 확률: p = 1 − exp(−r t)."""
    return 1.0 - np.exp(-np.asarray(r, float) * t)


def prob_to_rate(p, t=1.0):
    """길이 t인 기간의 확률 p를 일정한 율로 바꾼다: r = −ln(1 − p) / t."""
    return -np.log(1.0 - np.asarray(p, float)) / t


def weib_median(lam, gam):
    """S(t) = exp(−lam t^gam)의 중앙값(개월). weib_lam()의 역."""
    return (math.log(2) / lam) ** (1.0 / gam)


def occupancy_result(p, arm, pf_t, pd_t, half_cycle=True, disc=None):
    """주기 경계 시점(0, 1, …, n개월)의 상태 점유(pf_t, pd_t; 길이 n+1)에서 비용·생존연수·QALY를 계산한다.

    psm()과 같은 규칙이다: 반주기 보정(주기 시작과 끝의 평균), 주기 중간 시점 기준 할인, 약값은 무진행 상태에서만,
    임종기 비용은 그 주기에 사망한 사람 수에, 이상반응 비용과 QALY 손실은 1회. 반환 dict의 key도 psm()과 같다.
    markov()와 psm_curves()가 이 함수를 쓴다. 따라서 마르코프 모형과 분할생존모형의 결과 차이는 상태 점유에서만 생긴다.
    (psm()은 원래 코드 그대로이며, psm_curves()가 psm()과 같은 값을 내는 것은 gen/nums_ch23.py에서 확인한다.)
    """
    r = p["disc"] if disc is None else disc
    pf_t, pd_t = np.asarray(pf_t, float), np.asarray(pd_t, float)
    n = len(pf_t) - 1
    t = np.arange(0, n + 1) * CYCLE_MONTHS
    alive = pf_t + pd_t
    if half_cycle:
        pf = (pf_t[:-1] + pf_t[1:]) / 2
        pd_ = (pd_t[:-1] + pd_t[1:]) / 2
    else:
        pf, pd_ = pf_t[:-1], pd_t[:-1]
    deaths = alive[:-1] - alive[1:]
    mid = (t[:-1] + CYCLE_MONTHS / 2) / 12.0
    d = 1.0 / (1.0 + r) ** mid
    yr = CYCLE_MONTHS / 12.0
    ly_pf, ly_pd = (pf * yr), (pd_ * yr)
    out = {
        "arm": arm,
        "ly_pf": ly_pf.sum(), "ly_pd": ly_pd.sum(), "ly": (ly_pf + ly_pd).sum(),
        "ly_d": ((ly_pf + ly_pd) * d).sum(),
        "qaly": ((ly_pf * p["u_pf"] + ly_pd * p["u_pd"]) * d).sum() - p["du_ae_" + arm],
        "qaly_undisc": (ly_pf * p["u_pf"] + ly_pd * p["u_pd"]).sum() - p["du_ae_" + arm],
        "qaly_pf": (ly_pf * p["u_pf"] * d).sum(), "qaly_pd": (ly_pd * p["u_pd"] * d).sum(),
        "c_drug": (pf * p["c_drug_" + arm] * d).sum(),
        "c_pf": (pf * p["c_pf"] * d).sum(),
        "c_pd": (pd_ * p["c_pd"] * d).sum(),
        "c_death": (deaths * p["c_death"] * d).sum(),
        "c_ae": p["c_ae_" + arm],
        "trace": {"t": t, "pf": pf_t, "pd": pd_t, "dead": 1 - alive},
    }
    out["cost"] = out["c_drug"] + out["c_pf"] + out["c_pd"] + out["c_death"] + out["c_ae"]
    return out


def markov_matrix(r_prog, r_death_pf, r_death_pd, t=CYCLE_MONTHS):
    """세 상태(PF, PD, D) 마르코프 모형의 한 주기 전이확률 행렬(3×3, 행 = 출발 상태, 열 = 도착 상태, 행의 합 1).

    입력은 월 단위 율: r_prog(PF→PD), r_death_pf(PF→D), r_death_pd(PD→D).
    PF를 떠날 확률은 1 − exp(−(r_prog + r_death_pf) t)이고, 두 도착지에는 율의 비대로 나눈다(경쟁하는 두 사건).
    """
    tot = r_prog + r_death_pf
    p_exit = 1.0 - math.exp(-tot * t)
    p_pd = p_exit * r_prog / tot if tot > 0 else 0.0
    p_d = p_exit * r_death_pf / tot if tot > 0 else 0.0
    q = 1.0 - math.exp(-r_death_pd * t)
    return np.array([[1.0 - p_exit, p_pd, p_d],
                     [0.0, 1.0 - q, q],
                     [0.0, 0.0, 1.0]])


def markov_trace(P, horizon=HORIZON_MONTHS, start=(1.0, 0.0, 0.0)):
    """코호트 추적(cohort trace): 상태 분포 벡터에 전이확률 행렬을 주기마다 곱한다.
    반환: (horizon + 1) × 3 배열. 0행이 시작 분포, k행이 k주기 뒤의 (PF, PD, D) 비율."""
    x = np.asarray(start, float)
    out = np.empty((int(horizon) + 1, len(x)))
    out[0] = x
    for k in range(1, int(horizon) + 1):
        x = x @ P
        out[k] = x
    return out


def _trace_median(P, max_cycles=1200):
    """마르코프 코호트의 중앙 전체생존(개월): 생존 비율이 0.5가 되는 시점(주기 사이는 직선 보간)."""
    tr = markov_trace(P, max_cycles)
    s = tr[:, 0] + tr[:, 1]
    i = int(np.argmax(s <= 0.5))
    if i == 0:
        return float("inf")
    return float((i - 1) + (s[i - 1] - 0.5) / (s[i - 1] - s[i]))


def markov_inputs(p=None, share_death_pf=0.10, pd_death_by_arm=False):
    """공통 예시를 '전이확률이 일정한' 세 상태 마르코프 모형으로 옮길 때의 월 단위 율과 전이확률 행렬.

    정하는 방법(표준요법 B):
      1) PF를 떠나는 율 = ln2 / (B군 중앙 무진행생존). 중앙값은 p의 와이블 곡선에서 구한다(기준 분석 10개월).
      2) 그 가운데 share_death_pf(기본 10%)는 진행 없이 사망, 나머지는 진행(이 장에서 덧붙인 가정).
      3) PD→D 율은 코호트 추적의 중앙 전체생존이 B군 중앙값(기준 분석 28개월)과 같아지도록 수치적으로 맞춘다.
    신약 A:
      PF를 떠나는 두 율에 hr_pfs를 곱한다. PD→D 율은 pd_death_by_arm=False(기본)이면 B와 같게 두고
      (진행 뒤의 사망 속도는 두 군이 같다는 가정. hr_os는 쓰지 않는다),
      True이면 A군의 중앙 전체생존(와이블에 hr_os를 적용한 값, 기준 분석 약 36.0개월)에 따로 맞춘다.
    반환: {"B": {...}, "A": {...}} 각 군에 r_prog, r_death_pf, r_death_pd(월 단위 율), P(3×3 행렬),
          med_pfs, med_os(목표 중앙값), med_os_model(모형이 내는 중앙 전체생존).
    """
    from scipy.optimize import brentq
    p = base_params() if p is None else p
    out = {}
    for arm in ("B", "A"):
        h_pfs = p["hr_pfs"] if arm == "A" else 1.0
        h_os = p["hr_os"] if arm == "A" else 1.0
        med_pfs_b = weib_median(p["pfs_lam"], p["pfs_gam"])
        med_os = weib_median(p["os_lam"] * h_os, p["os_gam"])
        r_exit = math.log(2) / med_pfs_b * h_pfs
        r_prog, r_dpf = r_exit * (1 - share_death_pf), r_exit * share_death_pf
        if arm == "B" or pd_death_by_arm:
            r_dpd = brentq(lambda r: _trace_median(markov_matrix(r_prog, r_dpf, r)) - med_os, 1e-4, 2.0, xtol=1e-12)
        else:
            r_dpd = out["B"]["r_death_pd"]
        P = markov_matrix(r_prog, r_dpf, r_dpd)
        out[arm] = {"r_prog": r_prog, "r_death_pf": r_dpf, "r_death_pd": r_dpd, "P": P,
                    "med_pfs": math.log(2) / r_exit, "med_os": med_os, "med_os_model": _trace_median(P)}
    return out


def markov(p=None, arm="B", horizon=HORIZON_MONTHS, half_cycle=True, disc=None,
           share_death_pf=0.10, pd_death_by_arm=False, P=None):
    """세 상태 마르코프 코호트 모형 한 군의 결과(psm()과 같은 key의 dict + "P" 전이확률 행렬).

    전이확률 행렬 P를 직접 주지 않으면 markov_inputs()로 만든다. 비용·효용·할인·반주기 보정은 psm()과 같다.
    분할생존모형 기준 분석을 단순하게 옮긴 설명용 모형이며 결과는 기준 분석과 다르다.
    """
    p = base_params() if p is None else p
    if P is None:
        P = markov_inputs(p, share_death_pf, pd_death_by_arm)[arm]["P"]
    tr = markov_trace(P, horizon)
    out = occupancy_result(p, arm, tr[:, 0], tr[:, 1], half_cycle=half_cycle, disc=disc)
    out["P"] = P
    return out


def run_markov(p=None, **kw):
    """마르코프 모형으로 두 군을 돌리고 증분 결과를 붙인다(run()과 같은 꼴). kw는 markov()로 넘긴다."""
    p = base_params() if p is None else p
    a, b = markov(p, "A", **kw), markov(p, "B", **kw)
    dc, dq, dl = a["cost"] - b["cost"], a["qaly"] - b["qaly"], a["ly_d"] - b["ly_d"]
    return {"A": a, "B": b, "d_cost": dc, "d_qaly": dq, "d_ly": dl,
            "icer": dc / dq if dq != 0 else float("nan"),
            "icer_ly": dc / dl if dl != 0 else float("nan")}


def psm_curves(p, arm, s_pfs, s_os, half_cycle=True, disc=None):
    """주기 경계 시점(0, 1, …, n개월)의 생존곡선 값 배열 s_pfs, s_os를 직접 받아 분할생존모형을 계산한다.
    무진행생존은 전체생존을 넘지 못하게 자른다(curves()와 같은 규칙). 반환 dict는 psm()과 같다.
    psm_curves(p, arm, *curves(p, arm, t))는 psm(p, arm)과 같은 결과를 낸다."""
    s_os = np.asarray(s_os, float)
    s_pfs = np.minimum(np.asarray(s_pfs, float), s_os)
    return occupancy_result(p, arm, s_pfs, s_os - s_pfs, half_cycle=half_cycle, disc=disc)


def run_os(os_b, p=None, horizon=HORIZON_MONTHS, **kw):
    """표준요법 B의 전체생존 곡선을 다른 것으로 바꿔 넣고 두 군을 돌린다(외삽 모형 선택의 영향을 볼 때 쓴다).

    os_b: 개월 단위 시점 배열 t를 받아 S_OS(t)를 돌려주는 함수(예: lifelines 적합 결과로 만든 함수).
    신약 A의 전체생존은 비례위험 가정으로 S_A(t) = S_B(t) ** hr_os. 무진행생존은 p의 와이블 곡선 그대로.
    반환은 run()과 같은 꼴이다.
    """
    p = base_params() if p is None else p
    t = np.arange(0, horizon + 1) * CYCLE_MONTHS
    sb = np.asarray(os_b(t), float)
    res = {}
    for arm in ARMS:
        h_pfs = p["hr_pfs"] if arm == "A" else 1.0
        s_os = sb ** p["hr_os"] if arm == "A" else sb
        s_pfs = surv(p["pfs_lam"] * h_pfs, p["pfs_gam"], t)
        res[arm] = psm_curves(p, arm, s_pfs, s_os, **kw)
    a, b = res["A"], res["B"]
    dc, dq, dl = a["cost"] - b["cost"], a["qaly"] - b["qaly"], a["ly_d"] - b["ly_d"]
    return {"A": a, "B": b, "d_cost": dc, "d_qaly": dq, "d_ly": dl,
            "icer": dc / dq if dq != 0 else float("nan"),
            "icer_ly": dc / dl if dl != 0 else float("nan")}


def lifelines_weibull_to_lib(lambda_, rho_):
    """lifelines WeibullFitter의 모수(S(t) = exp(−(t/lambda_)^rho_))를 이 파일의 꼴(S(t) = exp(−lam t^gam))로 바꾼다.
    반환: (lam, gam) = (lambda_ ** (−rho_), rho_)."""
    return float(lambda_) ** (-float(rho_)), float(rho_)


# ======================================================================================
# 24장(불확실성 분석)에서 추가한 함수. 위의 기준 분석 함수와 입력값(base_params)은 그대로 두고 덧붙인 것이다.
#   분포와 범위    : dist_params, DSA_LABELS, dsa_ranges, DSA_RANGES
#   결정론적 분석  : set_param, oneway, twoway, threshold_value, threshold_price
#                    (oneway, twoway, threshold_value는 위의 set_input()으로 값을 바꾼다. 곡선은 중앙값·모양 모수로 다룬다)
#   시나리오       : run_waning, os_alternatives, scenario_table
#   확률적 분석 요약: CEAC_LAMS, inmb, ceac, evpi, quadrants, psa_summary
# 확률적 민감도 분석 자체는 위의 psa(n=5000, seed=20261002)를 그대로 쓴다. 뽑힌 입력값과 다시 뽑은 비율은 psa_inputs().
# ======================================================================================

DSA_LABELS = {
    "hr_os": "전체생존 위험비", "hr_pfs": "무진행생존 위험비",
    "os_med": "중앙 전체생존기간, 표준요법 B", "os_gam": "전체생존 곡선의 모양 모수 γ",
    "pfs_med": "중앙 무진행생존기간, 표준요법 B", "pfs_gam": "무진행생존 곡선의 모양 모수 γ",
    "u_pf": "무진행 상태의 효용", "u_pd": "진행 상태의 효용",
    "du_ae_A": "이상반응 QALY 손실, 신약 A", "du_ae_B": "이상반응 QALY 손실, 표준요법 B",
    "c_drug_A": "신약 A의 월 약값", "c_drug_B": "표준요법 B의 월 약값",
    "c_pf": "무진행 상태 월 관리비", "c_pd": "진행 상태 월 비용", "c_death": "임종기 비용",
    "c_ae_A": "이상반응 치료비, 신약 A", "c_ae_B": "이상반응 치료비, 표준요법 B",
    "disc": "할인율",
}


def dist_params(key, p=None):
    """PSA_SPEC에 있는 입력값 하나의 분포 정보를 dict로 돌려준다(draw()가 쓰는 것과 같은 모수화).
    pfs_med, os_med(중앙 생존기간)의 기준값은 get_input()으로 lam, gam에서 계산한다. 함께 뽑는 짝과 상관계수는 PSA_CORR에 있다.

    공통 key: dist("lognormal" | "beta" | "gamma"), base(기준값), se(자연 척도의 표준오차; 로그정규는 근사),
              mean(분포의 평균), lo, hi(분포의 2.5, 97.5 백분위수).
    lognormal: mu = ln(기준값), sigma = PSA_SPEC의 값(로그 척도의 표준오차). 기준값이 분포의 중앙값이 된다.
    beta     : alpha, beta. 적률법: n = m(1 − m)/se² − 1, alpha = m n, beta = (1 − m) n.
    gamma    : shape, scale. 적률법: shape = (m/se)², scale = m/shape = se²/m. (se = m × PSA_SPEC의 비율)
    """
    from scipy import stats
    p = base_params() if p is None else p
    dist, u = PSA_SPEC[key]
    m = get_input(p, key)
    if dist == "lognormal":
        lo, hi = (math.exp(math.log(m) + z * u) for z in (-1.959963984540054, 1.959963984540054))
        return {"dist": dist, "base": m, "mu": math.log(m), "sigma": u, "se": m * u,
                "mean": m * math.exp(u ** 2 / 2), "lo": lo, "hi": hi}
    if dist == "beta":
        n = m * (1 - m) / u ** 2 - 1
        a, b = m * n, (1 - m) * n
        lo, hi = stats.beta.ppf([0.025, 0.975], a, b)
        return {"dist": dist, "base": m, "alpha": a, "beta": b, "se": u, "mean": m, "lo": float(lo), "hi": float(hi)}
    if dist == "gamma":
        se = m * u
        shape = (m / se) ** 2
        scale = m / shape
        lo, hi = stats.gamma.ppf([0.025, 0.975], shape, scale=scale)
        return {"dist": dist, "base": m, "shape": shape, "scale": scale, "se": se, "mean": m, "lo": float(lo), "hi": float(hi)}
    raise ValueError("알 수 없는 분포: %r" % (dist,))


def dsa_ranges(p=None, pm=0.20, price_cut=0.20):
    """일원 민감도 분석의 범위 {입력값 key: (낮은 값, 높은 값, 우리말 이름, 범위의 근거)}.

    근거 "ci"      : PSA_SPEC에 분포가 있는 입력값. 그 분포의 2.5–97.5 백분위수(95% 구간).
    근거 "scenario": 신약 A의 약값. 표시 가격에서 price_cut(기본 20%) 인하한 값부터 표시 가격까지(가격 협상 시나리오).
    근거 "pm"      : 표준요법 B의 약값. 불확실성의 크기를 알려 주는 자료가 없어 기준값 ±pm(기본 ±20%)으로 임의로 정함.
    할인율과 분석기간은 여기에 넣지 않고 scenario_table()에서 바꾼다.
    곡선은 PSA_SPEC과 같은 표현(중앙값 pfs_med·os_med와 모양 모수 pfs_gam·os_gam)으로 들어 있고, oneway()가 set_input()으로 바꿔 넣는다.
    """
    p = base_params() if p is None else p
    out = {}
    for k in PSA_SPEC:
        d = dist_params(k, p)
        out[k] = (d["lo"], d["hi"], DSA_LABELS[k], "ci")
    out["c_drug_A"] = (p["c_drug_A"] * (1 - price_cut), p["c_drug_A"], DSA_LABELS["c_drug_A"], "scenario")
    out["c_drug_B"] = (p["c_drug_B"] * (1 - pm), p["c_drug_B"] * (1 + pm), DSA_LABELS["c_drug_B"], "pm")
    return out


DSA_RANGES = dsa_ranges()


def set_param(p, key, value):
    """입력값 dict p를 복사해 key 하나만 value로 바꾼 새 dict를 돌려준다(원래 dict는 그대로).
    dict의 값을 그대로 바꿀 뿐이다. 곡선을 중앙값·모양 모수로 바꾸려면 set_input()을 쓴다."""
    q = dict(p)
    q[key] = value
    return q


def oneway(p=None, ranges=None, lam=THRESHOLD):
    """일원 민감도 분석(one-way sensitivity analysis). 입력값을 하나씩 범위의 양 끝으로 바꾸고 나머지는 기준값에 둔다.
    곡선은 set_input()으로 바꾼다: 중앙값을 바꿀 때는 모양 모수를, 모양 모수를 바꿀 때는 중앙값을 기준값에 둔다.

    ranges는 dsa_ranges()와 같은 꼴(기본 DSA_RANGES). 반환: ICER가 움직인 폭(swing)이 큰 순서로 정렬한 dict의 list
    (토네이도 그림의 위에서 아래 순서). 각 dict의 key:
      key, label, basis, base(기준값), low, high(입력값의 범위),
      icer_low, icer_high(입력값이 low, high일 때의 ICER), icer_min, icer_max, swing(= icer_max − icer_min),
      nmb_low, nmb_high(임계값 lam에서의 증분 순금전편익), d_cost_low, d_cost_high, d_qaly_low, d_qaly_high,
      crosses(범위 안에서 ICER가 lam을 넘나드는지).
    """
    p = base_params() if p is None else p
    ranges = DSA_RANGES if ranges is None else ranges
    rows = []
    for k, (lo, hi, label, basis) in ranges.items():
        r_lo, r_hi = run(set_input(p, k, lo)), run(set_input(p, k, hi))
        i_lo, i_hi = r_lo["icer"], r_hi["icer"]
        rows.append({
            "key": k, "label": label, "basis": basis, "base": get_input(p, k), "low": lo, "high": hi,
            "icer_low": i_lo, "icer_high": i_hi, "icer_min": min(i_lo, i_hi), "icer_max": max(i_lo, i_hi),
            "swing": abs(i_hi - i_lo),
            "nmb_low": nmb(r_lo, lam), "nmb_high": nmb(r_hi, lam),
            "d_cost_low": r_lo["d_cost"], "d_cost_high": r_hi["d_cost"],
            "d_qaly_low": r_lo["d_qaly"], "d_qaly_high": r_hi["d_qaly"],
            "crosses": (nmb(r_lo, lam) > 0) != (nmb(r_hi, lam) > 0),
        })
    rows.sort(key=lambda x: -x["swing"])
    return rows


def twoway(key1, values1, key2, values2, p=None):
    """이원 민감도 분석(two-way sensitivity analysis): 두 입력값을 함께 바꾼 ICER의 표.
    반환: len(values1) × len(values2) numpy 배열. [i, j]는 key1 = values1[i], key2 = values2[j]일 때의 ICER."""
    p = base_params() if p is None else p
    out = np.empty((len(values1), len(values2)))
    for i, v1 in enumerate(values1):
        for j, v2 in enumerate(values2):
            out[i, j] = run(set_input(set_input(p, key1, v1), key2, v2))["icer"]
    return out


def threshold_value(key, lo, hi, lam=THRESHOLD, p=None):
    """임계값 분석(threshold analysis): 입력값 key를 [lo, hi] 안에서 바꿀 때 증분 순금전편익이 0이 되는 값
    (증분 QALY가 양수이면 ICER = lam이 되는 값). scipy.optimize.brentq로 푼다. 구간 안에 그런 값이 없으면 None."""
    from scipy.optimize import brentq
    p = base_params() if p is None else p
    f = lambda v: nmb(run(set_input(p, key, v)), lam)
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0:
        return float(lo)
    if f_hi == 0:
        return float(hi)
    if (f_lo > 0) == (f_hi > 0):
        return None
    return float(brentq(f, lo, hi, xtol=1e-10))


def threshold_price(lam=THRESHOLD, p=None, key="c_drug_A"):
    """ICER가 임계값 lam과 같아지는 신약 A의 월 약값(만원). 기준 분석에서 lam = 5,000이면 약 175.4만원.

    비용은 약값의 1차식이므로 식으로 바로 풀 수 있다: 증분 순금전편익 ÷ (신약 A군의 할인된 무진행 개월 수)만큼 값을 조정한다.
    threshold_value("c_drug_A", ...)로 수치적으로 풀어도 같은 값이 나온다(gen/nums_ch24.py에서 확인).
    """
    p = base_params() if p is None else p
    r = run(p)
    months = r["A"]["c_drug"] / p[key] if key == "c_drug_A" else -r["B"]["c_drug"] / p[key]
    return p[key] + nmb(r, lam) / months


def run_waning(p=None, start=36.0, end=60.0, apply_to=("os", "pfs"), horizon=HORIZON_MONTHS, **kw):
    """치료 효과 감소(treatment effect waning) 시나리오로 두 군을 돌린다(run()과 같은 꼴 + "hr_t").

    기준 분석은 위험비가 분석기간 내내 유지된다고 본다. 여기서는 위험비가 start개월까지는 그대로이고,
    start–end개월 사이에 1을 향해 직선으로 올라가며, end개월 뒤에는 1(두 군의 위험률이 같음)이라고 가정한다.
    신약 A의 곡선은 주기마다 S_A(t_k) = S_A(t_{k−1}) × exp(−HR(주기 중간 시점) × [H_B(t_k) − H_B(t_{k−1})])로 만든다
    (H_B = lam t^gam, 표준요법 B의 누적위험). 위험비가 일정하면 run()과 같은 결과가 된다.
    apply_to: 효과 감소를 적용할 곡선("os", "pfs"). 표준요법 B와 나머지 입력값은 기준 분석과 같다.
    """
    p = base_params() if p is None else p
    t = np.arange(0, horizon + 1) * CYCLE_MONTHS
    mid = t[:-1] + CYCLE_MONTHS / 2
    if end > start:
        w = np.clip((mid - start) / (end - start), 0.0, 1.0)     # 0 = 효과 그대로, 1 = 효과 없음
    else:
        w = (mid >= start).astype(float)

    def curve_a(lam, gam, hr, wane):
        d_h = np.diff(lam * t ** gam)                             # 주기마다 늘어나는 B군의 누적위험
        hr_t = hr + (1.0 - hr) * w if wane else np.full(len(mid), hr)
        return np.exp(-np.concatenate([[0.0], np.cumsum(hr_t * d_h)])), hr_t

    s_os, hr_os_t = curve_a(p["os_lam"], p["os_gam"], p["hr_os"], "os" in apply_to)
    s_pfs, hr_pfs_t = curve_a(p["pfs_lam"], p["pfs_gam"], p["hr_pfs"], "pfs" in apply_to)
    a = psm_curves(p, "A", s_pfs, s_os, **kw)
    b = psm_curves(p, "B", *curves(p, "B", t), **kw)
    dc, dq, dl = a["cost"] - b["cost"], a["qaly"] - b["qaly"], a["ly_d"] - b["ly_d"]
    return {"A": a, "B": b, "d_cost": dc, "d_qaly": dq, "d_ly": dl,
            "icer": dc / dq if dq != 0 else float("nan"),
            "icer_ly": dc / dl if dl != 0 else float("nan"),
            "hr_t": {"t_mid": mid, "os": hr_os_t, "pfs": hr_pfs_t}}


def os_alternatives(path=None):
    """23장 라 절의 가상 시험 자료(gen/_ch23_trial.csv, 표준요법 B군 300명)에 lifelines의 모수 모형을 적합해
    전체생존 외삽 곡선의 후보를 돌려준다: {key: (우리말 이름, 영어 이름, 함수 t(개월) → S_OS(t))}.
    key는 exp, weib, llog, lnorm, ggam. run_os()에 함수를 넘겨 쓰면 23장 표 23-12와 같은 결과가 나온다.
    lifelines가 없거나 자료 파일이 없으면 빈 dict를 돌려준다(자료 파일은 gen/nums_ch23.py가 만든다)."""
    import os
    import warnings
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ch23_trial.csv") if path is None else path
    try:
        from lifelines import (ExponentialFitter, WeibullFitter, LogLogisticFitter, LogNormalFitter,
                               GeneralizedGammaFitter)
        d = np.loadtxt(path, delimiter=",", skiprows=1)
    except (ImportError, OSError):
        return {}

    def curve(f):
        def s_os(x):
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                return f.survival_function_at_times(np.asarray(x, float)).values
        return s_os

    out = {}
    for key, ko, en, fitter in (("exp", "지수", "Exponential", ExponentialFitter),
                                ("weib", "와이블", "Weibull", WeibullFitter),
                                ("llog", "로그-로지스틱", "Log-logistic", LogLogisticFitter),
                                ("lnorm", "로그-정규", "Log-normal", LogNormalFitter),
                                ("ggam", "일반화 감마", "Generalized gamma", GeneralizedGammaFitter)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = fitter().fit(d[:, 0], d[:, 1])
        out[key] = (ko, en, curve(f))
    return out


def scenario_table(p=None, lam=THRESHOLD, os_models=True):
    """시나리오 분석 표. 기준 분석에서 가정 하나를 바꾼 결과를 dict의 list로 돌려준다.

    각 dict: key, group, label(우리말), label_en, d_cost, d_qaly, icer, nmb(임계값 lam에서의 증분 순금전편익).
    group은 "기준 분석", "방법론적 선택"(분석기간, 할인율), "구조적 불확실성"(외삽 모형, 치료 효과 감소, 모형 구조).
      분석기간 5·10·15년, 할인율 0%·3%(비용과 효과에 같은 율. 국내 지침 요약의 민감도 분석 값),
      전체생존 외삽 모형 4가지(os_alternatives(); os_models=False이거나 lifelines·자료가 없으면 생략),
      치료 효과 감소 2가지(run_waning: 3→5년, 5→7년 사이에 사라짐), 마르코프 모형 2가지(run_markov).
    """
    p = base_params() if p is None else p
    rows = []

    def add(key, group, label, label_en, r):
        rows.append({"key": key, "group": group, "label": label, "label_en": label_en,
                     "d_cost": r["d_cost"], "d_qaly": r["d_qaly"], "icer": r["icer"], "nmb": nmb(r, lam)})

    add("base", "기준 분석", "기준 분석(분석기간 20년, 할인율 4.5%, 와이블, 분할생존모형)", "Base case", run(p))
    for yrs in (5, 10, 15):
        add("horizon%d" % yrs, "방법론적 선택", "분석기간 %d년" % yrs, "Time horizon %d years" % yrs, run(p, horizon=yrs * 12))
    for rate in (0.0, 0.03):
        add("disc%g" % (rate * 100), "방법론적 선택", "할인율 %g%% (비용과 효과)" % (rate * 100),
            "Discount rate %g%% (costs and QALYs)" % (rate * 100), run(p, disc=rate))
    if os_models:
        alts = os_alternatives()
        for key in ("exp", "llog", "lnorm", "ggam"):
            if key in alts:
                ko, en, fn = alts[key]
                add("os_" + key, "구조적 불확실성", "전체생존 외삽: " + ko, "OS extrapolation: " + en.lower(), run_os(fn, p))
    add("wane_3_5", "구조적 불확실성", "치료 효과가 3년 뒤부터 줄어 5년에 사라짐", "Treatment effect waning from year 3 to year 5",
        run_waning(p, 36.0, 60.0))
    add("wane_5_7", "구조적 불확실성", "치료 효과가 5년 뒤부터 줄어 7년에 사라짐", "Treatment effect waning from year 5 to year 7",
        run_waning(p, 60.0, 84.0))
    add("markov", "구조적 불확실성", "마르코프 모형(진행 후 사망확률이 두 군에서 같음)", "Markov model, same post-progression mortality",
        run_markov(p))
    add("markov_arm", "구조적 불확실성", "마르코프 모형(진행 후 사망확률을 군별로 맞춤)", "Markov model, arm-specific post-progression mortality",
        run_markov(p, pd_death_by_arm=True))
    return rows


CEAC_LAMS = np.arange(0.0, 12001.0, 100.0)     # 비용효과 수용곡선을 그릴 임계값 격자(만원/QALY): 0, 100, …, 12,000


def inmb(dc, dq, lam=THRESHOLD):
    """증분 순금전편익(incremental net monetary benefit) = lam × 증분QALY − 증분비용. dc, dq는 수 또는 배열(PSA의 모의실험 결과)."""
    return lam * np.asarray(dq, float) - np.asarray(dc, float)


def ceac(dc, dq, lams=None):
    """비용효과 수용곡선(cost-effectiveness acceptability curve): 임계값마다 신약이 비용효과적일 확률,
    곧 증분 순금전편익이 양수인 모의실험의 비율. lams(기본 CEAC_LAMS)와 같은 길이의 배열을 돌려준다(수 하나를 주면 수 하나)."""
    lams = CEAC_LAMS if lams is None else lams
    if np.ndim(lams) == 0:
        return float(np.mean(inmb(dc, dq, lams) > 0))
    return np.array([np.mean(inmb(dc, dq, l) > 0) for l in lams])


def evpi(dc, dq, lam=THRESHOLD):
    """환자 1인당 완전정보의 기대가치(expected value of perfect information, 만원). 대안이 둘일 때:
    EVPI = E[max(증분 순금전편익, 0)] − max(E[증분 순금전편익], 0).
    앞 항은 모의실험마다 참값을 알고 더 나은 대안을 고를 때의 기대 순편익, 뒤 항은 지금의 정보(평균)로 한 번 고를 때의 기대 순편익이다.
    lam이 배열이면 같은 길이의 배열을 돌려준다."""
    if np.ndim(lam) == 0:
        nb = inmb(dc, dq, lam)
        return float(np.maximum(nb, 0.0).mean() - max(nb.mean(), 0.0))
    return np.array([evpi(dc, dq, l) for l in lam])


def quadrants(dc, dq):
    """비용효과평면의 사분면별 모의실험 비율. NE = 더 비싸고 더 효과적, SE = 더 싸고 더 효과적(우월),
    NW = 더 비싸고 덜 효과적(열등), SW = 더 싸고 덜 효과적."""
    dc, dq = np.asarray(dc, float), np.asarray(dq, float)
    return {"NE": float(np.mean((dq > 0) & (dc > 0))), "SE": float(np.mean((dq > 0) & (dc <= 0))),
            "NW": float(np.mean((dq <= 0) & (dc > 0))), "SW": float(np.mean((dq <= 0) & (dc <= 0)))}


def psa_summary(dc, dq, lam=THRESHOLD):
    """확률적 민감도 분석 결과의 요약 dict.
    n, d_cost·d_qaly·inmb 각각의 mean과 lo, hi(2.5, 97.5 백분위수), icer(= 평균 증분비용 ÷ 평균 증분QALY),
    p_ce(비용효과적일 확률), mcse(그 확률의 몬테카를로 표준오차 √(p(1 − p)/n)), evpi(1인당, 만원), quadrants."""
    dc, dq = np.asarray(dc, float), np.asarray(dq, float)
    nb = inmb(dc, dq, lam)
    pr = float(np.mean(nb > 0))
    out = {"n": int(len(dc)), "lam": float(lam), "icer": float(dc.mean() / dq.mean()),
           "p_ce": pr, "mcse": math.sqrt(pr * (1 - pr) / len(dc)), "evpi": evpi(dc, dq, lam), "quadrants": quadrants(dc, dq)}
    for name, x in (("d_cost", dc), ("d_qaly", dq), ("inmb", nb)):
        lo, hi = np.percentile(x, [2.5, 97.5])
        out[name] = {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}
    return out


if __name__ == "__main__":
    r = run()
    for arm in ARMS:
        x = r[arm]
        print(arm, f"LY {x['ly']:.3f} (PF {x['ly_pf']:.3f}, PD {x['ly_pd']:.3f}) | 할인 LY {x['ly_d']:.3f} | QALY {x['qaly']:.3f}"
              f" | 비용 {x['cost']:.0f} = 약값 {x['c_drug']:.0f} + PF {x['c_pf']:.0f} + PD {x['c_pd']:.0f}"
              f" + 임종 {x['c_death']:.0f} + AE {x['c_ae']:.0f}")
    print(f"증분비용 {r['d_cost']:.0f}만원, 증분QALY {r['d_qaly']:.3f}, 증분LY {r['d_ly']:.3f}")
    print(f"ICER {r['icer']:.0f}만원/QALY, {r['icer_ly']:.0f}만원/LY, 증분 순편익(λ={THRESHOLD:.0f}) {nmb(r):.0f}만원")
    t = np.array([12, 24, 36, 60, 120, 240])
    for arm in ARMS:
        s1, s2 = curves(base_params(), arm, t)
        print(arm, "PFS", np.round(s1, 3), "OS", np.round(s2, 3))
    dc, dq = psa(2000)
    print("PSA 2000회: P(비용효과적, λ=5000) =", round(float(np.mean(THRESHOLD * dq - dc > 0)), 3),
          "| 평균 증분비용", round(float(dc.mean())), "평균 증분QALY", round(float(dq.mean()), 3))
