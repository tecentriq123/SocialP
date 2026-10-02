"""23장(결정분석 모형)의 모든 숫자.

실행: source /home/claude/pylibs/env.sh && python3 gen/nums_ch23.py
  - 공통 예시 모형(gen/lib_p4.py)의 기준 분석과, 이 장에서 lib_p4에 덧붙인 함수(결정수형, 마르코프, 곡선 바꿔 넣기)를 쓴다.
  - 본문의 '파이썬으로 계산해 보기' 코드(CODE_*)를 실제로 실행해 출력을 얻고, lib_p4의 결과와 같은지 확인한다.
  - 결과는 gen/_ch23_nums.json(그림 스크립트용)에 저장하고 화면에도 찍는다.
모든 값은 가상의 예시이다. 난수 seed는 20261002로 고정.
"""
import contextlib, io, json, math, os, sys, warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy import stats
import lib_p4 as L

SEED = 20261002
p = L.base_params()
N = {}                      # json으로 저장할 값
T = np.arange(0, L.HORIZON_MONTHS + 1) * L.CYCLE_MONTHS


def show(title):
    print("\n" + "=" * 12, title)


# ====================================================================== 가. 결정수형
show("가. 결정수형")
tp = L.ae_tree_params()
TREE = L.ae_tree(tp)
for k in ("none", "proph"):
    x = TREE[k]
    print(k, "기대비용", round(x["cost"], 4), "기대 QALY 변화", round(x["qaly"], 5))
    for names, pr, c, q in x["paths"]:
        print("   ", " > ".join(names), "경로확률", round(pr, 4), "비용", c, "QALY", q, "| 확률×비용", round(pr * c, 3), "확률×QALY", round(pr * q, 5))
    assert abs(sum(pr for _, pr, _, _ in x["paths"]) - 1) < 1e-12
assert abs(TREE["none"]["cost"] - p["c_ae_A"]) < 1e-9 and abs(-TREE["none"]["qaly"] - p["du_ae_A"]) < 1e-12
# 접어 올라가기의 중간값: 중증 이상반응 마디의 기대값
TREE_MID = {
    "none_ae_cost": tp["p_hosp"] * tp["c_hosp"] + (1 - tp["p_hosp"]) * tp["c_out"],
    "ae_qloss": tp["p_hosp"] * tp["dq_hosp"] + (1 - tp["p_hosp"]) * tp["dq_out"],
    "proph_ae_cost": tp["c_proph"] + tp["p_hosp"] * tp["c_hosp"] + (1 - tp["p_hosp"]) * tp["c_out"],
    "p_ae_proph": tp["p_ae"] * tp["rr_proph"],
}
TREE_NMB = L.THRESHOLD * TREE["d_qaly"] - TREE["d_cost"]
print("중간값", TREE_MID)
print("증분비용", TREE["d_cost"], "증분QALY", TREE["d_qaly"], "ICER", TREE["icer"], "증분 순편익", TREE_NMB)
# 예방 요법을 쓰는 것으로 공통 모형의 이상반응 입력값을 바꾸면
p_pro = dict(p); p_pro["c_ae_A"] = TREE["proph"]["cost"]; p_pro["du_ae_A"] = -TREE["proph"]["qaly"]
BASE = L.run()
BASE_PRO = L.run(p_pro)
# 기준 분석이 20–25장이 함께 인용하는 값과 같은지 확인(lib_p4의 기준 입력값이 바뀌지 않았는지)
assert (round(BASE["A"]["cost"]), round(BASE["B"]["cost"]), round(BASE["icer"])) == (10562, 7689, 5621)
assert (round(BASE["A"]["qaly"], 3), round(BASE["B"]["qaly"], 3)) == (2.385, 1.874)
print("공통 모형 ICER", BASE["icer"], "→ 예방 요법 반영", BASE_PRO["icer"], BASE_PRO["d_cost"], BASE_PRO["d_qaly"])


# 스스로 확인하기: 상대위험도가 달라질 때
def tree_rr(rr):
    q = dict(tp); q["rr_proph"] = rr
    return L.ae_tree(q)


TREE_RR = {rr: tree_rr(rr) for rr in (0.5, 0.8)}
for rr, x in TREE_RR.items():
    print("RR", rr, "예방 요법 비용", x["proph"]["cost"], "증분비용", x["d_cost"], "증분QALY", x["d_qaly"], "ICER", x["icer"])
# ICER가 임계값과 같아지는 상대위험도: 60 − 120(1−rr) = 5000 × 0.012(1−rr)
per_ae_cost = TREE_MID["none_ae_cost"]; per_ae_q = TREE_MID["ae_qloss"]
RR_THR = 1 - tp["c_proph"] / (tp["p_ae"] * (per_ae_cost + L.THRESHOLD * per_ae_q))
RR_SAVE = 1 - tp["c_proph"] / (tp["p_ae"] * per_ae_cost)
print("ICER = 임계값이 되는 RR", RR_THR, tree_rr(RR_THR)["icer"], "| 비용이 같아지는 RR", RR_SAVE)

# ====================================================================== 나. 마르코프 코호트 모형
show("나. 율과 확률")
med_pfs_B, med_os_B = L.weib_median(p["pfs_lam"], p["pfs_gam"]), L.weib_median(p["os_lam"], p["os_gam"])
r_exit = math.log(2) / med_pfs_B
RP = {
    "med_pfs": med_pfs_B, "med_os": med_os_B,
    "r_exit": r_exit,                                   # 월 단위 율
    "p_month": float(L.rate_to_prob(r_exit)),           # 월 확률
    "p_year": float(L.rate_to_prob(r_exit, 12)),        # 1년 확률
    "p_year_div12": float(L.rate_to_prob(r_exit, 12)) / 12,   # 틀린 방법: 연 확률 ÷ 12
    "p_month_x12": float(L.rate_to_prob(r_exit)) * 12,        # 틀린 방법: 월 확률 × 12
    "r_exit_A": r_exit * p["hr_pfs"],
    "p_month_A": float(L.rate_to_prob(r_exit * p["hr_pfs"])),
    "p_month_A_wrong": float(L.rate_to_prob(r_exit)) * p["hr_pfs"],
    "p_year_A": float(L.rate_to_prob(r_exit * p["hr_pfs"], 12)),
    "p_year_A_wrong": float(L.rate_to_prob(r_exit, 12)) * p["hr_pfs"],
}
RP["r_back"] = float(L.prob_to_rate(RP["p_year"], 12))
for k, v in RP.items():
    print(f"  {k}: {v:.5f}")
# 스스로 확인하기: 연 확률 20%를 월 확률로, 위험비 0.7 적용
PRQ = {"p_year": 0.20}
PRQ["rate_m"] = float(L.prob_to_rate(0.20, 12)); PRQ["p_month"] = float(L.rate_to_prob(PRQ["rate_m"]))
PRQ["div12"] = 0.20 / 12
PRQ["p_year_hr"] = float(L.rate_to_prob(PRQ["rate_m"] * 0.7, 12)); PRQ["p_year_hr_wrong"] = 0.20 * 0.7
PRQ["rate_y"] = float(L.prob_to_rate(0.20, 1))
print("연습:", PRQ)

show("나. 전이확률과 코호트 추적")
MI = L.markov_inputs(p)
MI2 = L.markov_inputs(p, pd_death_by_arm=True)
for arm in ("B", "A"):
    m = MI[arm]
    print(arm, "율: 진행", round(m["r_prog"], 5), "PF 사망", round(m["r_death_pf"], 5), "PD 사망", round(m["r_death_pd"], 5),
          "| 중앙 PFS", round(m["med_pfs"], 2), "목표 OS 중앙값", round(m["med_os"], 2), "모형 OS 중앙값", round(m["med_os_model"], 2))
    print(np.round(m["P"], 5))
    assert np.allclose(m["P"].sum(axis=1), 1)
print("A군 PD→D를 군별로 맞추면: 율", round(MI2["A"]["r_death_pd"], 5), "확률", round(MI2["A"]["P"][1, 2], 5))
MK = L.run_markov(p)
MK2 = L.run_markov(p, pd_death_by_arm=True)
MK_NOH = L.run_markov(p, half_cycle=False)
PSM_NOH = L.run(half_cycle=False)
trB, trA = MK["B"]["trace"], MK["A"]["trace"]
TRACE_ROWS = [0, 1, 2, 3, 6, 12, 24, 36, 60, 120, 240]
print("코호트 추적 (B): 주기, PF, PD, 사망 / (A)")
for k in TRACE_ROWS:
    print(f"  {k:4d}  B {trB['pf'][k]:.4f} {trB['pd'][k]:.4f} {trB['dead'][k]:.4f}   A {trA['pf'][k]:.4f} {trA['pd'][k]:.4f} {trA['dead'][k]:.4f}")
# 첫 주기 손계산(표준요법 B)
PB = MI["B"]["P"]
x0 = np.array([1.0, 0, 0]); x1 = x0 @ PB; x2 = x1 @ PB; x3 = x2 @ PB
d1 = 1 / (1 + p["disc"]) ** (0.5 / 12)
HAND = {
    "x1": x1.tolist(), "x2": x2.tolist(), "x3": x3.tolist(),
    "pd2_from_pf": x1[0] * PB[0, 1], "pd2_stay": x1[1] * PB[1, 1],
    "d2_from_pf": x1[0] * PB[0, 2], "d2_from_pd": x1[1] * PB[1, 2],
    "pf_avg1": (1 + x1[0]) / 2, "pd_avg1": x1[1] / 2, "deaths1": x1[2],
    "disc1": d1,
}
HAND["cost_pf1"] = HAND["pf_avg1"] * (p["c_drug_B"] + p["c_pf"])
HAND["cost_pd1"] = HAND["pd_avg1"] * p["c_pd"]
HAND["cost_death1"] = HAND["deaths1"] * p["c_death"]
HAND["cost1"] = HAND["cost_pf1"] + HAND["cost_pd1"] + HAND["cost_death1"]
HAND["cost1_disc"] = HAND["cost1"] * d1
HAND["qaly1"] = (HAND["pf_avg1"] * p["u_pf"] + HAND["pd_avg1"] * p["u_pd"]) / 12
HAND["qaly1_disc"] = HAND["qaly1"] * d1
HAND["disc_12"] = 1 / (1 + p["disc"]) ** (11.5 / 12)        # 12번째 주기(11.5개월)
HAND["disc_120"] = 1 / (1 + p["disc"]) ** (119.5 / 12)      # 120번째 주기
for k, v in HAND.items():
    print(f"  {k}: {v}")
show("나. 마르코프 모형의 결과")


def line(tag, r):
    for arm in ("A", "B"):
        x = r[arm]
        print(f"  {tag} {arm}: LY {x['ly']:.3f} (PF {x['ly_pf']:.3f}, PD {x['ly_pd']:.3f}) QALY {x['qaly']:.3f} 비용 {x['cost']:.1f} "
              f"= 약값 {x['c_drug']:.1f} + PF {x['c_pf']:.1f} + PD {x['c_pd']:.1f} + 임종 {x['c_death']:.1f} + AE {x['c_ae']:.0f}")
    print(f"  {tag} 증분비용 {r['d_cost']:.1f} 증분QALY {r['d_qaly']:.4f} 증분LY(할인 전) {r['A']['ly'] - r['B']['ly']:.3f} ICER {r['icer']:.1f} 순편익 {L.nmb(r):.1f}")


line("마르코프", MK)
line("마르코프(진행 후 사망을 군별로 맞춤)", MK2)
line("분할생존(기준)", BASE)
line("마르코프, 반주기 보정 없음", MK_NOH)
line("분할생존, 반주기 보정 없음", PSM_NOH)
# 반주기 보정의 작은 예: 1년 주기, 연초 생존자 100, 60, 30, 10, 0
HC_TOY = {"alive": [100, 60, 30, 10, 0]}
HC_TOY["start"] = sum(HC_TOY["alive"][:-1]); HC_TOY["end"] = sum(HC_TOY["alive"][1:])
HC_TOY["avg"] = (HC_TOY["start"] + HC_TOY["end"]) / 2
print("반주기 예:", HC_TOY)
# 연 단위 주기였다면(마르코프 B): 12개월마다 센 생존자 합
aliveB = trB["pf"] + trB["pd"]
HC_YEAR = {"start": float(aliveB[0:240:12].sum()), "end": float(aliveB[12:241:12].sum())}
HC_YEAR["avg"] = (HC_YEAR["start"] + HC_YEAR["end"]) / 2
print("연 주기로 셌다면 B 생존연수: 시작 기준", HC_YEAR["start"], "끝 기준", HC_YEAR["end"], "평균", HC_YEAR["avg"], "| 월 주기 보정값", MK["B"]["ly"])
# 기억 없음: 와이블 곡선에서 달마다의 사망확률(시간에 따라 달라지는 전이확률)
s_os_B = L.surv(p["os_lam"], p["os_gam"], T)
TD = {m: float(1 - s_os_B[m] / s_os_B[m - 1]) for m in (1, 12, 24, 60, 120)}
print("와이블 OS의 월 사망확률:", TD)
# 마르코프 모형이 내는 생존곡선과 분할생존모형의 곡선
CMP_T = [12, 24, 36, 60, 120]
for arm in ("B", "A"):
    spf, sos = L.curves(p, arm, T)
    tr = MK[arm]["trace"]
    print(arm, "OS  PSM", np.round(sos[CMP_T], 3), "마르코프", np.round((tr["pf"] + tr["pd"])[CMP_T], 3))
    print(arm, "PFS PSM", np.round(spf[CMP_T], 3), "마르코프", np.round(tr["pf"][CMP_T], 3))

# ====================================================================== 다. 분할생존모형
show("다. 분할생존모형")
OCC_T = [0, 6, 12, 24, 36, 60, 120]
for arm in ("B", "A"):
    tr = BASE[arm]["trace"]
    for m in OCC_T:
        print(f"  {arm} {m:4d}개월  PFS {tr['pf'][m]:.4f}  OS {tr['pf'][m] + tr['pd'][m]:.4f}  PF {tr['pf'][m]:.4f} PD {tr['pd'][m]:.4f} 사망 {tr['dead'][m]:.4f}")
DET = {arm: L.psm_curves(p, arm, *L.curves(p, arm, T)) for arm in L.ARMS}
for arm in L.ARMS:
    for k in ("cost", "qaly", "ly", "ly_pf", "ly_pd", "c_drug", "c_pf", "c_pd", "c_death"):
        assert abs(DET[arm][k] - BASE[arm][k]) < 1e-9
    x = DET[arm]
    print(f"  {arm} QALY 구성: PF {x['qaly_pf']:.4f} + PD {x['qaly_pd']:.4f} − AE {p['du_ae_' + arm]} = {x['qaly']:.4f} | 할인 LY {x['ly_d']:.3f} | 할인 전 QALY {x['qaly_undisc']:.3f}")
line("기준", BASE)
print("  증분 구성: 약값", BASE["A"]["c_drug"] - BASE["B"]["c_drug"], "PF", BASE["A"]["c_pf"] - BASE["B"]["c_pf"], "PD", BASE["A"]["c_pd"] - BASE["B"]["c_pd"],
      "임종", BASE["A"]["c_death"] - BASE["B"]["c_death"], "AE", BASE["A"]["c_ae"] - BASE["B"]["c_ae"])
# 곡선이 만나는 시점: lam_pfs t^g1 = lam_os t^g2
CROSS = {}
for arm in ("B", "A"):
    h1 = p["hr_pfs"] if arm == "A" else 1.0
    h2 = p["hr_os"] if arm == "A" else 1.0
    CROSS[arm] = ((p["pfs_lam"] * h1) / (p["os_lam"] * h2)) ** (1 / (p["os_gam"] - p["pfs_gam"]))
print("PFS와 OS 곡선이 만나는 시점(개월):", CROSS)
# 확률적 민감도 분석의 분포(PSA_SPEC)에서 뽑은 입력값 가운데 두 곡선이 분석기간(240개월) 안에 만나는 비율
_rng = np.random.default_rng(SEED)
_hit, _area = 0, []
for _ in range(5000):
    q = L.draw(_rng)
    a_max = 0.0
    for arm in ("A", "B"):
        h1 = q["hr_pfs"] if arm == "A" else 1.0
        h2 = q["hr_os"] if arm == "A" else 1.0
        d = L.surv(q["pfs_lam"] * h1, q["pfs_gam"], T) - L.surv(q["os_lam"] * h2, q["os_gam"], T)
        if (d > 1e-12).any():
            a_max = max(a_max, float(d[d > 0].sum() / 12))       # 잘려 나간 면적(년)
    if a_max > 0:
        _hit += 1; _area.append(a_max)
CROSS_PSA = {"share": _hit / 5000, "area_median": float(np.median(_area)), "share_area_gt_0.05y": float(np.mean(np.array(_area) > 0.05)) * _hit / 5000}
print("PSA 5000벌 가운데 두 곡선이 20년 안에 만나는 비율:", CROSS_PSA)
# 중앙값(군별)
MED = {"B": {"pfs": med_pfs_B, "os": med_os_B},
       "A": {"pfs": L.weib_median(p["pfs_lam"] * p["hr_pfs"], p["pfs_gam"]), "os": L.weib_median(p["os_lam"] * p["hr_os"], p["os_gam"])}}
print("중앙값:", MED)
print("중앙값 차이: PFS", MED["A"]["pfs"] - MED["B"]["pfs"], "OS", MED["A"]["os"] - MED["B"]["os"],
      "| 평균(20년 제한) 차이(개월): PF", (BASE["A"]["ly_pf"] - BASE["B"]["ly_pf"]) * 12, "OS", (BASE["A"]["ly"] - BASE["B"]["ly"]) * 12)
# 입력값 표(논문 상자)의 95% 구간: PSA_SPEC의 분포에서
CI = {}
for k, (dist, u) in L.PSA_SPEC.items():
    m = p[k]
    if dist == "lognormal":
        lo, hi = math.exp(math.log(m) - 1.959964 * u), math.exp(math.log(m) + 1.959964 * u)
    elif dist == "beta":
        n = m * (1 - m) / u ** 2 - 1
        lo, hi = stats.beta.ppf([0.025, 0.975], m * n, (1 - m) * n)
    else:
        se = m * u; shape = (m / se) ** 2
        lo, hi = stats.gamma.ppf([0.025, 0.975], shape, scale=m / shape)
    CI[k] = (float(lo), float(hi))
    print(f"  {k}: {m:.5g} ({dist}, {u})  95% {lo:.4g} – {hi:.4g}")
# 임계값에 맞는 약값
def price_at_threshold(r, lam=L.THRESHOLD):
    pf_disc = r["A"]["c_drug"] / p["c_drug_A"]              # 할인된 무진행 개월 수
    return p["c_drug_A"] - (r["d_cost"] - lam * r["d_qaly"]) / pf_disc


print("임계값에 맞는 월 약값:", price_at_threshold(BASE))

# ====================================================================== 라. 생존곡선의 외삽
show("라. 외삽")
from lifelines import (ExponentialFitter, WeibullFitter, LogNormalFitter, LogLogisticFitter,
                       GeneralizedGammaFitter, KaplanMeierFitter)
N_TRIAL, FU_LO, FU_HI = 300, 24.0, 30.0


def simulate_trial(seed=SEED, n=N_TRIAL):
    """표준요법 B군의 가상 임상시험 자료: 기준 분석의 와이블 전체생존에서 생존시간을 뽑고,
    환자마다 24–30개월 사이의 자료 마감 시점에서 중도절단한다."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    t_event = (-np.log(u) / p["os_lam"]) ** (1 / p["os_gam"])     # S(t) = u 를 t에 대해 푼 것
    t_cens = rng.uniform(FU_LO, FU_HI, size=n)                    # 등록 시점에 따라 추적 가능 기간이 다름
    return np.minimum(t_event, t_cens), (t_event <= t_cens).astype(int)


dur, ev = simulate_trial()
np.savetxt(os.path.join(HERE, "_ch23_trial.csv"), np.column_stack([dur, ev]), delimiter=",", header="months,event", comments="", fmt=["%.6f", "%d"])
km = KaplanMeierFitter().fit(dur, ev)
KM = {"n": int(len(dur)), "events": int(ev.sum()), "cens": int((1 - ev).sum()),
      "s12": float(km.predict(12)), "s24": float(km.predict(24)), "s_last": float(km.survival_function_.iloc[-1, 0]),
      "t_max": float(dur.max()), "median": float(km.median_survival_time_),
      "true_s12": float(s_os_B[12]), "true_s24": float(s_os_B[24]), "true_s30": float(s_os_B[30])}
print("KM:", KM)
FITTERS = [("exp", "지수", "Exponential", ExponentialFitter, 1),
           ("weib", "와이블", "Weibull", WeibullFitter, 2),
           ("lnorm", "로그-정규", "Log-normal", LogNormalFitter, 2),
           ("llog", "로그-로지스틱", "Log-logistic", LogLogisticFitter, 2),
           ("ggam", "일반화 감마", "Generalized gamma", GeneralizedGammaFitter, 3)]
FIT, FOBJ = {}, {}
for key, ko, en, F, k in FITTERS:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f = F().fit(dur, ev)
    FOBJ[key] = f
    S = f.survival_function_at_times(T).values
    r = L.run_os(lambda x, f=f: f.survival_function_at_times(x).values, p)
    FIT[key] = {"ko": ko, "en": en, "k": k, "ll": float(f.log_likelihood_), "aic": float(f.AIC_), "bic": float(f.BIC_),
                "aic_hand": float(2 * k - 2 * f.log_likelihood_), "bic_hand": float(k * math.log(len(dur)) - 2 * f.log_likelihood_),
                "s24": float(S[24]), "s60": float(S[60]), "s120": float(S[120]), "s240": float(S[240]),
                "median": float(f.median_survival_time_),
                "mean_B": float(np.trapezoid(S, T) / 12), "ly_B": r["B"]["ly"], "ly_A": r["A"]["ly"],
                "d_ly": r["A"]["ly"] - r["B"]["ly"], "d_qaly": r["d_qaly"], "d_cost": r["d_cost"], "icer": r["icer"],
                "nmb": L.nmb(r), "price": price_at_threshold(r), "params": {a: float(b) for a, b in f.params_.items()},
                "warn": len(w), "S": S.tolist()}
    assert abs(FIT[key]["aic"] - FIT[key]["aic_hand"]) < 1e-6 and abs(FIT[key]["bic"] - FIT[key]["bic_hand"]) < 1e-6
    assert abs(FIT[key]["mean_B"] - FIT[key]["ly_B"]) < 1e-9
    x = FIT[key]
    print(f"  {en:18s} k={k} logL {x['ll']:.2f} AIC {x['aic']:.1f} BIC {x['bic']:.1f} | S24 {x['s24']:.3f} S60 {x['s60']:.3f} S120 {x['s120']:.3f} S240 {x['s240']:.4f}"
          f" | 중앙값 {x['median']:.1f} 평균 {x['mean_B']:.2f}년 A {x['ly_A']:.2f} | 증분LY {x['d_ly']:.3f} 증분QALY {x['d_qaly']:.3f} 증분비용 {x['d_cost']:.0f} ICER {x['icer']:.0f}"
          f" 순편익 {x['nmb']:.0f} 임계가격 {x['price']:.1f} warn {x['warn']} {x['params']}")
# 일반화 감마의 수렴 확인: 헤시안이 양의 정부호인지, 셋째 모수(lambda_; 1이면 와이블, 0이면 로그-정규)의 신뢰구간
_g = FOBJ["ggam"]
GG = {"hess_eig_min": float(np.linalg.eigvalsh(_g._hessian_).min()),
      "lambda": float(_g.lambda_), "lambda_lo": float(_g.summary.loc["lambda_", "coef lower 95%"]),
      "lambda_hi": float(_g.summary.loc["lambda_", "coef upper 95%"]),
      "ll_minus_weibull": FIT["ggam"]["ll"] - FIT["weib"]["ll"], "s120": FIT["ggam"]["s120"]}
GG["lrt_p_vs_weibull"] = float(stats.chi2.sf(2 * GG["ll_minus_weibull"], 1))
# 초깃값을 바꿔도 같은 최대우도에 이르는지
for _ip in ([3.4, 0.3, 0.1], [3.8, -1.0, 3.0], [3.6, 0.0, 1.0]):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _g2 = GeneralizedGammaFitter().fit(dur, ev, initial_point=np.array(_ip))
    assert abs(_g2.log_likelihood_ - _g.log_likelihood_) < 1e-4 and abs(_g2.lambda_ - _g.lambda_) < 1e-2
print("  일반화 감마:", GG)
assert GG["hess_eig_min"] > 0 and FIT["ggam"]["warn"] == 0
aic_min = min(x["aic"] for x in FIT.values()); bic_min = min(x["bic"] for x in FIT.values())
for x in FIT.values():
    x["d_aic"] = x["aic"] - aic_min; x["d_bic"] = x["bic"] - bic_min
print("  기준 분석(참값 와이블): 평균", BASE["B"]["ly"], "ICER", BASE["icer"], "S60", s_os_B[60], "S120", s_os_B[120], "S240", s_os_B[240])
# lifelines 와이블 모수 → lib 꼴
w = FOBJ["weib"]
lam_w, gam_w = L.lifelines_weibull_to_lib(w.lambda_, w.rho_)
WB = {"lambda_": float(w.lambda_), "rho_": float(w.rho_), "lam": lam_w, "gam": gam_w,
      "se_lambda": float(w.summary.loc["lambda_", "se(coef)"]), "se_rho": float(w.summary.loc["rho_", "se(coef)"]),
      "rho_lo": float(w.summary.loc["rho_", "coef lower 95%"]), "rho_hi": float(w.summary.loc["rho_", "coef upper 95%"]),
      "maxdiff": float(np.abs(L.surv(lam_w, gam_w, T) - w.survival_function_at_times(T).values).max()),
      "median": float(w.median_survival_time_), "median_lib": L.weib_median(lam_w, gam_w),
      "true_lam": p["os_lam"], "true_gam": p["os_gam"]}
print("와이블 변환:", WB)
assert WB["maxdiff"] < 1e-12
# 관찰 기간(30개월)까지의 면적과 20년 전체 면적
from lifelines.utils import restricted_mean_survival_time
RM = {"km30": float(restricted_mean_survival_time(km, t=30.0)),                       # Kaplan-Meier, 30개월까지(개월)
      "weib30": float(np.trapezoid(np.array(FIT["weib"]["S"])[:31], T[:31])),         # 적합한 와이블, 30개월까지(개월)
      "weib_total": FIT["weib"]["mean_B"] * 12}
RM["share30"] = RM["weib30"] / RM["weib_total"]
print("면적:", RM)
# numpy 한 줄 검산(가 절), matrix_power 검산(나 절)
assert abs((np.array([0.12, 0.18, 0.70]) * np.array([700, 200, 0])).sum() - 120) < 1e-9
assert np.allclose(np.linalg.matrix_power(MI["B"]["P"], 24)[0], [trB["pf"][24], trB["pd"][24], trB["dead"][24]])
assert np.allclose(MI["B"]["P"].sum(axis=1), 1)
# 마지막 관찰 시점에서 적합 곡선들이 얼마나 비슷한가
print("  S(24) 범위", min(x["s24"] for x in FIT.values()), max(x["s24"] for x in FIT.values()), "KM", KM["s24"])
# 진행 상태 비용이 낮다면(월 100만원) 외삽 모형에 따라 ICER가 얼마나 달라지나
p_low = dict(p); p_low["c_pd"] = 100.0
LOWPD = {key: L.run_os(lambda x, f=FOBJ[key]: f.survival_function_at_times(x).values, p_low)["icer"] for key in FIT}
LOWPD["base"] = L.run(p_low)["icer"]
print("  진행 상태 월 100만원일 때 ICER:", {k: round(v) for k, v in LOWPD.items()})
# 스스로 확인하기(라): 평균 생존 차이에 PD 비용·효용을 곱해 보기
PD_CPQ = p["c_pd"] * 12 / p["u_pd"]
print("진행 상태 1 QALY당 비용:", PD_CPQ)

# ====================================================================== 본문의 파이썬 코드(실제로 실행해 출력을 얻는다)
CODE_MARKOV = r'''import numpy as np

def matrix(r_prog, r_death_pf, r_death_pd):
    """월 단위 율 세 개 → 월 전이확률 행렬 (행: 출발, 열: 도착; 순서는 무진행, 진행, 사망)"""
    leave = 1 - np.exp(-(r_prog + r_death_pf))      # 무진행 상태를 떠날 확률
    to_pd = leave * r_prog / (r_prog + r_death_pf)  # 그중 진행으로 가는 몫
    to_d = leave - to_pd                            # 나머지는 사망
    q = 1 - np.exp(-r_death_pd)                     # 진행 상태에서 사망할 확률
    return np.array([[1 - leave, to_pd, to_d],
                     [0.0, 1 - q, q],
                     [0.0, 0.0, 1.0]])

def run(P, c_drug, c_ae, dq_ae, n=240):
    x = np.array([1.0, 0.0, 0.0])                   # 시작: 모두 무진행
    trace = [x]
    for k in range(n):                              # 240개월 = 20년
        x = x @ P                                   # 행렬 곱 한 번 = 한 달
        trace.append(x)
    trace = np.array(trace)                         # 241행 × 3열
    pf = (trace[:-1, 0] + trace[1:, 0]) / 2         # 반주기 보정: 주기 시작과 끝의 평균
    pd_ = (trace[:-1, 1] + trace[1:, 1]) / 2
    died = trace[1:, 2] - trace[:-1, 2]             # 그 달에 사망한 비율
    disc = 1 / 1.045 ** ((np.arange(n) + 0.5) / 12) # 주기 중간 시점의 할인 계수
    cost = ((pf * (c_drug + 40) + pd_ * 250 + died * 800) * disc).sum() + c_ae
    qaly = ((pf * 0.78 + pd_ * 0.62) / 12 * disc).sum() - dq_ae
    return trace, cost, qaly

r_exit = np.log(2) / 10                             # 중앙 무진행생존 10개월 → 월 단위 율
r_pd_d = 0.0470602                                  # 진행 → 사망 (중앙 전체생존 28개월에 맞춘 값)
P_B = matrix(0.9 * r_exit, 0.1 * r_exit, r_pd_d)
P_A = matrix(0.9 * r_exit * 0.65, 0.1 * r_exit * 0.65, r_pd_d)   # 위험비는 율에 곱한다

np.set_printoptions(precision=4, suppress=True)
print(P_B)
tB, cB, qB = run(P_B, c_drug=120, c_ae=80, dq_ae=0.008)
tA, cA, qA = run(P_A, c_drug=190, c_ae=120, dq_ae=0.012)
print(tB[:4])
print("B: 비용", round(cB), "QALY", round(qB, 3))
print("A: 비용", round(cA), "QALY", round(qA, 3))
print("ICER", round((cA - cB) / (qA - qB)))'''

CODE_PSM = r'''import numpy as np

t = np.arange(0, 241)                              # 0, 1, ..., 240개월
def S(median, shape, hr=1.0):                      # 와이블 생존함수 S(t) = exp(-lam * t^shape)
    lam = np.log(2) / median ** shape              # 중앙값에서 lam을 구함
    return np.exp(-lam * hr * t ** shape)          # 위험비는 lam에 곱한다

def psm(pfs, os_, c_drug, c_ae, dq_ae):
    pfs = np.minimum(pfs, os_)                     # 무진행생존이 전체생존을 넘지 않게
    pf_t, pd_t = pfs, os_ - pfs                    # 상태 점유: 무진행, 진행
    pf = (pf_t[:-1] + pf_t[1:]) / 2                # 반주기 보정
    pd_ = (pd_t[:-1] + pd_t[1:]) / 2
    died = os_[:-1] - os_[1:]                      # 그 달에 사망한 비율
    disc = 1 / 1.045 ** ((t[:-1] + 0.5) / 12)      # 할인 계수
    cost = ((pf * (c_drug + 40) + pd_ * 250 + died * 800) * disc).sum() + c_ae
    qaly = ((pf * 0.78 + pd_ * 0.62) / 12 * disc).sum() - dq_ae
    return pf.sum() / 12, pd_.sum() / 12, cost, qaly

B = psm(S(10, 0.95), S(28, 1.15), c_drug=120, c_ae=80, dq_ae=0.008)
A = psm(S(10, 0.95, hr=0.65), S(28, 1.15, hr=0.75), c_drug=190, c_ae=120, dq_ae=0.012)
for name, r in (("A", A), ("B", B)):
    print(name, "무진행", round(r[0], 3), "년, 진행", round(r[1], 3), "년, 비용", round(r[2]), "QALY", round(r[3], 3))
print("증분비용", round(A[2] - B[2]), "증분QALY", round(A[3] - B[3], 3), "ICER", round((A[2] - B[2]) / (A[3] - B[3])))'''

CODE_FIT = r'''import numpy as np
import pandas as pd
from lifelines import (KaplanMeierFitter, ExponentialFitter, WeibullFitter,
                       LogNormalFitter, LogLogisticFitter, GeneralizedGammaFitter)

# 가상의 시험 자료 만들기: 표준요법 B군 300명, 추적 24-30개월
rng = np.random.default_rng(20261002)
lam, gam = np.log(2) / 28 ** 1.15, 1.15            # 중앙 전체생존 28개월인 와이블
t_event = (-np.log(rng.uniform(size=300)) / lam) ** (1 / gam)
t_cens = rng.uniform(24, 30, size=300)             # 환자마다 다른 자료 마감 시점
months = np.minimum(t_event, t_cens)               # 관찰된 시간
event = (t_event <= t_cens).astype(int)            # 1 = 사망, 0 = 중도절단
print("사망", event.sum(), "명, 중도절단", (1 - event).sum(), "명")
print("Kaplan-Meier 24개월 생존율", round(KaplanMeierFitter().fit(months, event).predict(24), 3))

grid = np.arange(0, 241)                           # 0-240개월
rows = []
for name, Fitter in [("Exponential", ExponentialFitter), ("Weibull", WeibullFitter),
                     ("Log-normal", LogNormalFitter), ("Log-logistic", LogLogisticFitter),
                     ("Generalized gamma", GeneralizedGammaFitter)]:
    f = Fitter().fit(months, event)                # 최대우도추정
    s = f.survival_function_at_times(grid).values  # 20년까지 늘린 생존곡선
    rows.append([name, f.AIC_, f.BIC_, s[24], s[60], s[120], np.trapezoid(s, grid) / 12])
out = pd.DataFrame(rows, columns=["model", "AIC", "BIC", "S(2y)", "S(5y)", "S(10y)", "mean_years"])
print(out.round(3).to_string(index=False))

w = WeibullFitter().fit(months, event)
print("lifelines:", round(w.lambda_, 3), round(w.rho_, 4))
print("S(t) = exp(-lam * t^gam) 꼴:", round(w.lambda_ ** -w.rho_, 5), round(w.rho_, 4))'''


def run_code(src):
    buf = io.StringIO()
    ns = {}
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(src, ns)
    return buf.getvalue().rstrip("\n"), ns


CODE = {}
for name, src in (("markov", CODE_MARKOV), ("psm", CODE_PSM), ("fit", CODE_FIT)):
    out, ns = run_code(src)
    CODE[name] = {"src": src, "out": out}
    show("코드 출력: " + name)
    print(out)
    if name == "markov":
        assert round(ns["cB"]) == round(MK["B"]["cost"]) and round(ns["cA"]) == round(MK["A"]["cost"])
        assert round(ns["qB"], 3) == round(MK["B"]["qaly"], 3) and round(ns["qA"], 3) == round(MK["A"]["qaly"], 3)
        assert round((ns["cA"] - ns["cB"]) / (ns["qA"] - ns["qB"])) == round(MK["icer"])
    if name == "psm":
        A_, B_ = ns["A"], ns["B"]
        assert abs(A_[2] - BASE["A"]["cost"]) < 1e-6 and abs(B_[3] - BASE["B"]["qaly"]) < 1e-9
    if name == "fit":
        assert np.allclose(ns["months"], dur) and (ns["event"] == ev).all()

# ====================================================================== 저장
N.update({
    "tp": tp, "tree": {k: {"cost": TREE[k]["cost"], "qaly": TREE[k]["qaly"], "paths": [[list(a), b, c, d] for a, b, c, d in TREE[k]["paths"]]} for k in ("none", "proph")},
    "tree_d_cost": TREE["d_cost"], "tree_d_qaly": TREE["d_qaly"], "tree_icer": TREE["icer"],
    "markov": {arm: {"P": MI[arm]["P"].tolist(), "pf": MK[arm]["trace"]["pf"].tolist(), "pd": MK[arm]["trace"]["pd"].tolist()} for arm in L.ARMS},
    "fit": FIT, "km": KM,
})
with open(os.path.join(HERE, "_ch23_nums.json"), "w", encoding="utf-8") as f:
    json.dump(N, f, ensure_ascii=False)
print("\nsaved _ch23_nums.json, _ch23_trial.csv")
