"""24장(불확실성 분석)의 모든 숫자를 계산한다.
실행: source /home/claude/pylibs/env.sh && python3 gen/nums_ch24.py   (그 뒤 python3 gen/fig_ch24.py)

공통 예시(gen/lib_p4.py)의 기준 분석과 확률적 민감도 분석 L.psa(n=5000, seed=20261002)를 그대로 쓴다.
이 장에서 lib_p4.py에 덧붙인 함수(dist_params, DSA_RANGES, oneway, twoway, threshold_value, threshold_price,
run_waning, os_alternatives, scenario_table, CEAC_LAMS, inmb, ceac, evpi, quadrants, psa_summary)를 여기서 검산한다.
모든 값은 가상의 예시이다. 결과는 gen/_ch24_nums.json(그림용)과 gen/_ch24_cells.html(본문 코드 상자)에 저장한다.
"""
import contextlib, html, io, json, math, os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import lib_p4 as L

SEED = 20261002
LAM = L.THRESHOLD
N = {}


def show(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


p = L.base_params()
BASE = L.run()
show("기준 분석")
print(f"증분비용 {BASE['d_cost']:.1f} 증분QALY {BASE['d_qaly']:.4f} ICER {BASE['icer']:.1f} 순편익 {L.nmb(BASE):.1f} 순건강편익 {L.nmb(BASE) / LAM:.4f}")
assert round(BASE["icer"]) == 5621 and round(BASE["d_cost"]) == 2872 and round(BASE["d_qaly"], 3) == 0.511 and round(L.nmb(BASE)) == -317
N["base"] = {"d_cost": BASE["d_cost"], "d_qaly": BASE["d_qaly"], "icer": BASE["icer"], "nmb": L.nmb(BASE),
             "cost_A": BASE["A"]["cost"], "cost_B": BASE["B"]["cost"], "qaly_A": BASE["A"]["qaly"], "qaly_B": BASE["B"]["qaly"]}

# ====================================================================== 가. 결정론적 민감도 분석
show("가. 일원 민감도 분석 (범위와 ICER)")
OW = L.oneway()
for i, r in enumerate(OW, 1):
    print(f"{i:2d} {r['label']:22s} [{r['basis']:8s}] 기준 {r['base']:.5g}  범위 {r['low']:.5g} – {r['high']:.5g} | ICER {r['icer_low']:.0f} → {r['icer_high']:.0f}"
          f" (폭 {r['swing']:.0f}) | 순편익 {r['nmb_low']:.0f} → {r['nmb_high']:.0f} | 증분비용 {r['d_cost_low']:.0f} → {r['d_cost_high']:.0f}"
          f" 증분QALY {r['d_qaly_low']:.3f} → {r['d_qaly_high']:.3f} | 임계값 넘나듦 {r['crosses']}")
N["oneway"] = OW
print("ICER 변화율(기준 대비 %):")
for r in OW[:8]:
    print(f"   {r['label']}: {100 * (r['icer_low'] / BASE['icer'] - 1):+.1f}% / {100 * (r['icer_high'] / BASE['icer'] - 1):+.1f}%")
print("나머지(7번째 이하) 폭의 최댓값:", max(r["swing"] for r in OW[6:]), " 11번째 이하:", max(r["swing"] for r in OW[10:]))
# 범위가 분포의 95% 구간과 같은지(위험비는 PSA_SPEC 주석의 신뢰구간과 같은지)
for k, (lo, hi) in {"hr_pfs": (0.52, 0.81), "hr_os": (0.58, 0.97)}.items():
    assert round(L.DSA_RANGES[k][0], 2) == lo and round(L.DSA_RANGES[k][1], 2) == hi
# 로그 척도 표준오차는 신뢰구간에서: (ln 상한 − ln 하한) / (2 × 1.96)
# 표 24-2의 γ 행 각주: 척도 모수 λ를 고정하고 모양 모수 γ만 바꾸므로 전체생존 곡선의 중앙값도 함께 달라진다
for _g, _med in zip(L.DSA_RANGES["os_gam"][:2], (42, 19)):
    _m = L.weib_median(p["os_lam"], _g)
    print(f"os_gam {_g:.4f}: 표준요법 B의 중앙 전체생존 {_m:.2f}개월 (기준 {L.weib_median(p['os_lam'], p['os_gam']):.1f}개월), "
          f"신약 A {L.weib_median(p['os_lam'] * p['hr_os'], _g):.2f}개월")
    assert round(_m) == _med and abs(_m - round(_m)) < 0.45
assert round(L.weib_median(p["os_lam"], p["os_gam"])) == 28
print("hr_os 로그 SE (CI 0.58–0.97에서):", (math.log(0.97) - math.log(0.58)) / (2 * 1.96), " hr_pfs:", (math.log(0.81) - math.log(0.52)) / (2 * 1.96))

show("가. 전체생존 위험비를 바꿀 때 ICER가 덜 내려가는 이유")
for hr in (L.DSA_RANGES["hr_os"][0], 0.75, L.DSA_RANGES["hr_os"][1]):
    r = L.run(L.set_param(p, "hr_os", hr))
    print(f"  HR_OS {hr:.3f}: 생존연수 A {r['A']['ly']:.3f} 증분QALY {r['d_qaly']:.3f} 증분비용 {r['d_cost']:.0f} (진행 상태 비용 증분 {r['A']['c_pd'] - r['B']['c_pd']:.0f}) ICER {r['icer']:.0f}")
print("  진행 상태 1 QALY당 비용:", p["c_pd"] * 12 / p["u_pd"])

show("가. 약값과 ICER, 임계값 분석")
TP = L.threshold_price()
TP_num = L.threshold_value("c_drug_A", 0.0, 400.0)
pf_months_A = BASE["A"]["c_drug"] / p["c_drug_A"]
tp_brentq = brentq(lambda v: L.run(L.set_param(p, "c_drug_A", v))["icer"] - LAM, 100, 190)
print(f"임계 가격 {TP:.4f} (수치해 {TP_num:.4f}, ICER식 brentq {tp_brentq:.4f}), 인하율 {100 * (1 - TP / 190):.2f}%, 인하액 {190 - TP:.2f}")
print(f"신약 A군의 할인된 무진행 개월 수 {pf_months_A:.3f}, 손계산: 190 − 317.15/{pf_months_A:.3f} = {190 - (-L.nmb(BASE)) / pf_months_A:.3f}")
print(f"약값 1만원당 ICER 변화 {pf_months_A / BASE['d_qaly']:.2f}")
assert abs(TP - TP_num) < 1e-6 and abs(TP - tp_brentq) < 1e-6
assert abs(L.run(L.set_param(p, "c_drug_A", TP))["icer"] - LAM) < 1e-6
PRICE = {}
for cut in (0, 5, 10, 15, 20):
    v = 190 * (1 - cut / 100)
    r = L.run(L.set_param(p, "c_drug_A", v))
    PRICE[cut] = {"price": v, "icer": r["icer"], "nmb": L.nmb(r), "d_cost": r["d_cost"]}
    print(f"  {cut:2d}% 인하 → {v:.1f}만원: 증분비용 {r['d_cost']:.0f} ICER {r['icer']:.0f} 순편익 {L.nmb(r):.0f}")
TP_LAMS = {lam: L.threshold_price(lam) for lam in (3000, 4000, 5000, 6000)}
print("임계값별 임계 가격:", {k: round(v, 2) for k, v in TP_LAMS.items()})
TV = {"hr_os": L.threshold_value("hr_os", 0.2, 1.2), "hr_pfs": L.threshold_value("hr_pfs", 0.2, 1.2),
      "u_pf": L.threshold_value("u_pf", 0.5, 1.0), "c_pd": L.threshold_value("c_pd", 0, 1000), "c_drug_B": L.threshold_price(key="c_drug_B")}
print("다른 입력값의 임계값(ICER = 5,000이 되는 값):", TV)
for k in ("hr_os", "hr_pfs", "u_pf"):
    lo, hi = L.DSA_RANGES[k][:2]
    print(f"   {k}: {TV[k]:.4f}  95% 구간 {lo:.3f}–{hi:.3f} 안인가? {lo <= TV[k] <= hi}")
# 전체생존 위험비의 양 끝에서 임계 가격
TP_HR = {hr: L.threshold_price(p=L.set_param(p, "hr_os", hr)) for hr in (L.DSA_RANGES["hr_os"][0], L.DSA_RANGES["hr_os"][1])}
TP_HRP = {hr: L.threshold_price(p=L.set_param(p, "hr_pfs", hr)) for hr in (L.DSA_RANGES["hr_pfs"][0], L.DSA_RANGES["hr_pfs"][1])}
print("전체생존 위험비 양 끝에서의 임계 가격:", TP_HR, " 무진행생존 위험비:", TP_HRP)
N["price"] = {"tp": TP, "cut_pct": 100 * (1 - TP / 190), "pf_months_A": pf_months_A, "by_cut": PRICE, "tp_lams": TP_LAMS}

show("가. 이원 민감도 분석: 약값 × 무진행생존 위험비")
TW_PRICES = [190.0, 175.0, 160.0]
TW_HRS = [L.DSA_RANGES["hr_pfs"][0], 0.65, L.DSA_RANGES["hr_pfs"][1]]
TW = L.twoway("c_drug_A", TW_PRICES, "hr_pfs", TW_HRS)
print("   열: HR_PFS", [round(h, 2) for h in TW_HRS])
for pr, row in zip(TW_PRICES, TW):
    print(f"   약값 {pr:.0f}: ", [round(float(x)) for x in row])
N["twoway"] = {"prices": TW_PRICES, "hrs": TW_HRS, "icer": TW.tolist()}

show("가. 시나리오 분석")
SC = L.scenario_table()
for r in SC:
    print(f"  [{r['group']}] {r['label']:44s} 증분비용 {r['d_cost']:.0f} 증분QALY {r['d_qaly']:.3f} ICER {r['icer']:.0f} ({100 * (r['icer'] / BASE['icer'] - 1):+.1f}%) 순편익 {r['nmb']:.0f}")
N["scen"] = SC
ic = [r["icer"] for r in SC]
print("시나리오 ICER 범위:", min(ic), max(ic), " 임계값 아래로 내려가는 시나리오 수:", sum(x < LAM for x in ic))
# 23장 표 23-10, 23-12와 같은 값인지
S = {r["key"]: r for r in SC}
N23 = json.load(open(os.path.join(HERE, "_ch23_nums.json"), encoding="utf-8"))
for k in ("exp", "llog", "lnorm", "ggam"):
    print(f"   23장 표 23-12와 비교 {k}: {S['os_' + k]['icer']:.3f} / {N23['fit'][k]['icer']:.3f}")
    # 자료 파일은 소수 6자리로 저장되어 있어 적합 결과가 아주 조금 다를 수 있다. 표에 적는 자릿수에서는 같아야 한다.
    assert round(S["os_" + k]["icer"]) == round(N23["fit"][k]["icer"]) and round(S["os_" + k]["d_qaly"], 3) == round(N23["fit"][k]["d_qaly"], 3), k
assert round(S["markov"]["icer"]) == 5858 and round(S["markov_arm"]["icer"]) == 5723
N20 = json.load(open(os.path.join(HERE, "_ch20_nums.json"), encoding="utf-8"))
print("20장 그림 20-1의 값과 비교:", {y: round(N20["horizon"][y]["icer"]) for y in ("5", "10", "20")})
assert abs(N20["horizon"]["5"]["icer"] - S["horizon5"]["icer"]) < 1e-6 and abs(N20["horizon"]["10"]["icer"] - S["horizon10"]["icer"]) < 1e-6
# 치료 효과 감소: 위험비가 일정하면 기준 분석과 같아야 한다
w0 = L.run_waning(start=1e9, end=2e9)
assert abs(w0["d_cost"] - BASE["d_cost"]) < 1e-8 and abs(w0["d_qaly"] - BASE["d_qaly"]) < 1e-10
for s, e in ((36, 60), (60, 84)):
    w = L.run_waning(start=s, end=e)
    print(f"  효과 감소 {s}→{e}개월: 생존연수 A {w['A']['ly']:.3f} (기준 {BASE['A']['ly']:.3f}) 증분QALY {w['d_qaly']:.3f} ICER {w['icer']:.0f};"
          f" 48개월 시점 위험비(OS) {w['hr_t']['os'][47]:.3f}")
print("  20년 시점 생존율 A, B:", L.curves(p, "A", np.array([240.0]))[1], L.curves(p, "B", np.array([240.0]))[1])

# ====================================================================== 나. 확률적 민감도 분석
show("나. 분포의 모수 (적률법)")
DP = {k: L.dist_params(k) for k in L.PSA_SPEC}
for k, d in DP.items():
    print(f"  {k:9s} {d['dist']:9s} 기준 {d['base']:.5g} se {d['se']:.4g} | " + ", ".join(f"{a} {d[a]:.5g}" for a in d if a not in ("dist", "base", "se")))
N["dist"] = DP
# 손계산 검산
u = DP["u_pf"]; n_ = 0.78 * 0.22 / 0.03 ** 2 - 1
print(f"  베타(무진행 효용): n = 0.78×0.22/0.0009 − 1 = {n_:.2f}, α = {0.78 * n_:.2f}, β = {0.22 * n_:.2f}; 평균 {u['alpha'] / (u['alpha'] + u['beta']):.4f},"
      f" SD {math.sqrt(u['alpha'] * u['beta'] / ((u['alpha'] + u['beta']) ** 2 * (u['alpha'] + u['beta'] + 1))):.4f}")
g = DP["c_pd"]
print(f"  감마(진행 상태 비용): shape = (250/50)² = {g['shape']:.1f}, scale = 250/25 = {g['scale']:.1f}; 평균 {g['shape'] * g['scale']:.1f}, SD {math.sqrt(g['shape']) * g['scale']:.1f}")
h = DP["hr_os"]
print(f"  로그정규(전체생존 위험비): mu = ln 0.75 = {h['mu']:.4f}, sigma {h['sigma']}, 95% {h['lo']:.3f}–{h['hi']:.3f}, 평균 {h['mean']:.4f}")
# 정규분포를 쓰면 안 되는 이유: 범위를 벗어날 확률
print("  효용 0.92 (SE 0.05)를 정규분포로 뽑으면 1을 넘을 확률:", 1 - stats.norm.cdf(1, 0.92, 0.05))
print("  비용 80 (SE 60)을 정규분포로 뽑으면 음수일 확률:", stats.norm.cdf(0, 80, 60))
print("  진행 상태 비용 감마분포: 중앙값", stats.gamma.ppf(0.5, 25, scale=10), "왜도", 2 / math.sqrt(25))

show("나. 확률적 민감도 분석 5,000회 (L.psa)")
DC, DQ = L.psa(n=5000, seed=SEED)
PS = L.psa_summary(DC, DQ)
print(json.dumps(PS, indent=1, ensure_ascii=False))
P6000 = L.ceac(DC, DQ, 6000.0)
print("P(비용효과적) 5,000:", PS["p_ce"], " 6,000:", P6000)
assert abs(PS["p_ce"] - 0.1762) < 1e-12 and abs(P6000 - 0.6522) < 1e-12
Q = PS["quadrants"]
print("사분면별 횟수:", {k: round(v * 5000) for k, v in Q.items()}, " 합", sum(Q.values()))
print("표준편차: 증분비용", DC.std(ddof=1), "증분QALY", DQ.std(ddof=1), " 상관", np.corrcoef(DC, DQ)[0, 1])
print("최소·최대 증분QALY", DQ.min(), DQ.max(), " 증분비용", DC.min(), DC.max())
# 기준 분석과 PSA 평균의 차이
pm = dict(p)
for k in L.PSA_SPEC:
    pm[k] = DP[k]["mean"]
RM = L.run(pm)
print(f"기준 분석: 증분비용 {BASE['d_cost']:.0f}, 증분QALY {BASE['d_qaly']:.3f}, ICER {BASE['icer']:.0f}, 순편익 {L.nmb(BASE):.0f}")
print(f"PSA 평균 : 증분비용 {PS['d_cost']['mean']:.0f}, 증분QALY {PS['d_qaly']['mean']:.3f}, ICER(평균÷평균) {PS['icer']:.0f}, 순편익 {PS['inmb']['mean']:.0f}")
print(f"분포의 평균값을 넣은 결정론적 계산: 증분비용 {RM['d_cost']:.0f}, 증분QALY {RM['d_qaly']:.3f}, ICER {RM['icer']:.0f}")
print(f"비선형성 예: HR_OS 하한에서 증분QALY {OW[0]['d_qaly_low']:.3f} (기준 대비 {OW[0]['d_qaly_low'] - BASE['d_qaly']:+.3f}), 상한에서 {OW[0]['d_qaly_high']:.3f} ({OW[0]['d_qaly_high'] - BASE['d_qaly']:+.3f})")
# ICER의 분포
ICER = DC / DQ
print("모의실험별 ICER의 2.5·50·97.5 백분위수:", np.percentile(ICER, [2.5, 50, 97.5]), " 평균:", ICER.mean(), " 음수 ICER 수:", int((ICER < 0).sum()),
      " (우월", int(((DQ > 0) & (DC <= 0)).sum()), "열등", int(((DQ <= 0) & (DC > 0)).sum()), ")", " 최소", ICER.min(), "최대", ICER.max())
order = np.argsort(ICER)
print("ICER가 가장 낮은 5개의 (증분QALY, 증분비용, ICER):", [(round(DQ[i], 3), round(DC[i]), round(ICER[i])) for i in order[:5]])
print("ICER가 가장 높은 3개:", [(round(DQ[i], 3), round(DC[i]), round(ICER[i])) for i in order[-3:]])
N["psa"] = PS; N["p6000"] = P6000
N["icer_pct"] = [float(x) for x in np.percentile(ICER, [2.5, 50, 97.5])]

show("나. 모의실험 횟수와 몬테카를로 오차")
RUNP = np.cumsum(L.inmb(DC, DQ) > 0) / np.arange(1, 5001)
CONV = {n: float(RUNP[n - 1]) for n in (100, 200, 500, 1000, 2000, 3000, 4000, 5000)}
print("누적 확률:", CONV)
for n in (500, 1000, 5000, 10000):
    print(f"  n = {n}: 몬테카를로 표준오차 {math.sqrt(PS['p_ce'] * (1 - PS['p_ce']) / n):.4f}, 95% 범위 ±{1.96 * math.sqrt(PS['p_ce'] * (1 - PS['p_ce']) / n):.4f}")
print("  5,000회의 95% 범위:", PS["p_ce"] - 1.96 * PS["mcse"], PS["p_ce"] + 1.96 * PS["mcse"])
SEEDS = {}
for n in (1000, 5000):
    ps = [L.ceac(*L.psa(n, seed=s), LAM) for s in range(1, 21)]
    SEEDS[n] = {"min": float(min(ps)), "max": float(max(ps)), "sd": float(np.std(ps, ddof=1)), "mean": float(np.mean(ps))}
    print(f"  난수 seed 1–20으로 {n}회씩: 확률 {min(ps):.3f}–{max(ps):.3f}, 평균 {np.mean(ps):.4f}, SD {np.std(ps, ddof=1):.4f}")
BIG_DC, BIG_DQ = L.psa(100000, seed=1)
BIG = L.psa_summary(BIG_DC, BIG_DQ)
print(f"  10만 회(seed 1): 확률 {BIG['p_ce']:.4f} (6,000에서 {L.ceac(BIG_DC, BIG_DQ, 6000.0):.4f}), 평균 증분비용 {BIG['d_cost']['mean']:.0f}, 증분QALY {BIG['d_qaly']['mean']:.4f}, 순편익 {BIG['inmb']['mean']:.1f}, EVPI {BIG['evpi']:.1f}")
print("  평균 순편익의 몬테카를로 표준오차(5,000회):", L.inmb(DC, DQ).std(ddof=1) / math.sqrt(5000))
N["conv"] = {"run": RUNP[::10].tolist(), "run_n": list(range(1, 5001, 10)), "at": CONV, "seeds": SEEDS, "big_p": BIG["p_ce"]}

show("나. 무진행생존 곡선이 전체생존 곡선을 넘어 잘린 모의실험 (입력값을 독립으로 뽑은 결과)")


def cap_stats(n, seed):
    """L.psa와 같은 난수로 입력값 n벌을 다시 뽑아, 군별로 무진행생존 곡선(자르기 전)이 전체생존 곡선을 넘는지 센다.
    반환: (A군에서 처음 넘는 개월, 그때의 전체생존, 잘려 나간 무진행 기간(개월), B군에서 넘는지) 배열."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, L.HORIZON_MONTHS + 1) * L.CYCLE_MONTHS
    first, os_at, lost, cap_b = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.zeros(n, bool)
    for i in range(n):
        q = L.draw(rng)
        so_a, sp_a = L.surv(q["os_lam"] * q["hr_os"], q["os_gam"], t), L.surv(q["pfs_lam"] * q["hr_pfs"], q["pfs_gam"], t)
        so_b, sp_b = L.surv(q["os_lam"], q["os_gam"], t), L.surv(q["pfs_lam"], q["pfs_gam"], t)
        ex = sp_a > so_a
        if ex.any():
            k = int(np.argmax(ex))
            first[i], os_at[i], lost[i] = t[k], so_a[k], np.maximum(sp_a - so_a, 0).sum()
        cap_b[i] = bool((sp_b > so_b).any())
    return first, os_at, lost, cap_b


c_first, c_os, c_lost, c_b = cap_stats(5000, SEED)
c_a = ~np.isnan(c_first)
c_any = c_a | c_b
nb_ = L.inmb(DC, DQ)
print(f"  5,000벌: 적어도 한 군에서 잘림 {int(c_any.sum())}벌 ({c_any.mean():.4f}); 신약 A군 {c_a.mean():.4f}, 표준요법 B군 {c_b.mean():.4f}")
print(f"  A군에서 잘린 것 가운데 만나는 시점의 전체생존이 5% 미만인 비율 {np.mean(c_os[c_a] < 0.05):.3f}")
print(f"  전체 모의실험 가운데 A군의 두 곡선이 5년 안에 만나는 비율 {np.mean(c_first < 60):.4f}, 잘려 나간 무진행 기간이 1개월을 넘는 비율 {np.mean(c_lost > 1.0):.4f}")
print(f"  비용효과적일 확률: 잘린 {int(c_any.sum())}벌 {np.mean(nb_[c_any] > 0):.4f}, 잘리지 않은 {int((~c_any).sum())}벌 {np.mean(nb_[~c_any] > 0):.4f}")
b_first, _, _, b_b = cap_stats(100000, 1)
print(f"  10만 벌(seed 1): 적어도 한 군에서 잘림 {np.mean(~np.isnan(b_first) | b_b):.4f}")
assert int(c_any.sum()) == 1122
N["cap"] = {"any": float(c_any.mean()), "A": float(c_a.mean()), "B": float(c_b.mean()), "n_any": int(c_any.sum()),
            "tail_share": float(np.mean(c_os[c_a] < 0.05)), "within5y": float(np.mean(c_first < 60)), "lost_gt1": float(np.mean(c_lost > 1.0)),
            "p_ce_cap": float(np.mean(nb_[c_any] > 0)), "p_ce_nocap": float(np.mean(nb_[~c_any] > 0)),
            "any_100k": float(np.mean(~np.isnan(b_first) | b_b))}

show("나(심화). 와이블 모양·척도 모수의 상관과 촐레스키 분해")
from lifelines import WeibullFitter
trial = np.loadtxt(os.path.join(HERE, "_ch23_trial.csv"), delimiter=",", skiprows=1)
wf = WeibullFitter().fit(trial[:, 0], trial[:, 1])
lam_, rho_ = wf.params_.values
V = wf.variance_matrix_.values
J = np.array([[-rho_ / lam_, -math.log(lam_)], [0.0, 1.0 / rho_]])       # (lambda_, rho_) → (ln lam, ln gam)의 델타법
C = J @ V @ J.T
sd1, sd2 = math.sqrt(C[0, 0]), math.sqrt(C[1, 1]); corr = C[0, 1] / (sd1 * sd2)
CH = np.linalg.cholesky(C)
print(f"  ln λ의 SE {sd1:.4f}, ln γ의 SE {sd2:.4f}, 상관 {corr:.3f}; 공분산 행렬 {C.tolist()}")
print(f"  촐레스키 L = {CH.tolist()}; 검산 L Lᵀ = {(CH @ CH.T).tolist()}")
mu = np.array([-rho_ * math.log(lam_), math.log(rho_)])
rng = np.random.default_rng(SEED)
z = rng.standard_normal((20000, 2))
cor_draw = mu + z @ CH.T
ind_draw = mu + z * np.array([sd1, sd2])
def s24(d):
    return np.exp(-np.exp(d[:, 0]) * 24.0 ** np.exp(d[:, 1]))
def s60(d):
    return np.exp(-np.exp(d[:, 0]) * 60.0 ** np.exp(d[:, 1]))
print("  뽑은 값의 상관(상관 반영):", np.corrcoef(cor_draw.T)[0, 1], " (독립):", np.corrcoef(ind_draw.T)[0, 1])
CORR = {"sd_lnlam": sd1, "sd_lngam": sd2, "corr": corr, "chol": CH.tolist(),
        "s24_cor": [float(x) for x in np.percentile(s24(cor_draw), [2.5, 97.5])], "s24_ind": [float(x) for x in np.percentile(s24(ind_draw), [2.5, 97.5])],
        "s60_cor": [float(x) for x in np.percentile(s60(cor_draw), [2.5, 97.5])], "s60_ind": [float(x) for x in np.percentile(s60(ind_draw), [2.5, 97.5])]}
print("  24개월 생존율의 95% 구간: 상관 반영", CORR["s24_cor"], " 독립으로 뽑음", CORR["s24_ind"])
print("  60개월 생존율의 95% 구간: 상관 반영", CORR["s60_cor"], " 독립으로 뽑음", CORR["s60_ind"])
from lifelines import KaplanMeierFitter
km = KaplanMeierFitter().fit(trial[:, 0], trial[:, 1])
ci24 = km.confidence_interval_survival_function_.loc[:24.0].iloc[-1].values
print("  Kaplan-Meier 24개월 생존율", float(km.predict(24.0)), "95% CI", ci24)
N["corr"] = CORR

# ====================================================================== 다. 수용곡선과 순편익
show("다. 순편익과 수용곡선")
NB = L.inmb(DC, DQ)
print(f"증분 순금전편익(λ=5,000): 평균 {NB.mean():.1f}, 2.5–97.5 백분위수 {np.percentile(NB, 2.5):.0f} – {np.percentile(NB, 97.5):.0f}, 양수 비율 {np.mean(NB > 0):.4f}, 양수 {int((NB > 0).sum())}회")
print(f"순건강편익(QALY): 기준 분석 {L.nmb(BASE) / LAM:.4f}, PSA 평균 {NB.mean() / LAM:.4f}")
print(f"양수인 것만의 평균 {NB[NB > 0].mean():.1f}, 음수인 것만의 평균 {NB[NB <= 0].mean():.1f}")
print("처음 다섯 번의 모의실험 (증분QALY, 증분비용, 순편익(5,000), 순편익(6,000)):")
FIRST = []
for i in range(5):
    FIRST.append({"dq": float(DQ[i]), "dc": float(DC[i]), "nb5": float(5000 * DQ[i] - DC[i]), "nb6": float(6000 * DQ[i] - DC[i]), "icer": float(DC[i] / DQ[i])})
    print(f"   {i + 1}: {DQ[i]:.3f}  {DC[i]:8.0f}  {5000 * DQ[i] - DC[i]:8.0f}  {6000 * DQ[i] - DC[i]:8.0f}   ICER {DC[i] / DQ[i]:.0f}")
N["first5"] = FIRST
CE = L.ceac(DC, DQ)
TAB_LAMS = [0, 2000, 3000, 4000, 5000, 5500, 6000, 7000, 8000, 10000, 12000]
EV_TAB = {}
for lam in TAB_LAMS:
    nb = L.inmb(DC, DQ, lam)
    EV_TAB[lam] = {"p": L.ceac(DC, DQ, float(lam)), "evpi": L.evpi(DC, DQ, float(lam)), "nmb_mean": float(nb.mean()), "nmb_det": L.nmb(BASE, lam)}
    print(f"  λ {lam:6d}: 확률 {EV_TAB[lam]['p']:.4f}  평균 순편익 {nb.mean():8.1f} (결정론적 {L.nmb(BASE, lam):8.1f})  EVPI {EV_TAB[lam]['evpi']:.1f}")
fine = np.arange(0.0, 20001.0, 10.0)
ce_fine = L.ceac(DC, DQ, fine)
lam50 = float(fine[np.argmax(ce_fine >= 0.5)])
ev_fine = L.evpi(DC, DQ, fine)
lam_ev = float(fine[np.argmax(ev_fine)])
print(f"수용곡선이 50%를 처음 넘는 임계값 {lam50:.0f}; ICER(기준 분석) {BASE['icer']:.0f}; ICER(PSA 평균의 비) {PS['icer']:.0f}; 모의실험별 ICER의 중앙값 {np.median(ICER):.0f}")
print(f"ICER(기준 분석)에서의 확률 {L.ceac(DC, DQ, BASE['icer']):.4f}; ICER(PSA)에서의 확률 {L.ceac(DC, DQ, PS['icer']):.4f}")
print(f"λ=0에서의 확률 {ce_fine[0]:.4f} (= 증분비용이 음수인 비율 {np.mean(DC < 0):.4f}); λ가 아주 클 때의 한계 = 증분QALY가 양수인 비율 {np.mean(DQ > 0):.4f}; λ=12,000 {L.ceac(DC, DQ, 12000.0):.4f}")
print(f"90%, 95%를 넘는 임계값: {fine[np.argmax(ce_fine >= 0.9)]:.0f}, {fine[np.argmax(ce_fine >= 0.95)]:.0f}")
print(f"EVPI 최댓값 {ev_fine.max():.1f} (λ = {lam_ev:.0f}); λ=5,000에서 {L.evpi(DC, DQ):.2f}")
assert abs(L.evpi(DC, DQ) - np.maximum(NB, 0).mean()) < 1e-9        # 평균 순편익이 음수이므로 뒤 항은 0
# EVPI 손계산: 양수 비율 × 양수일 때의 평균
print(f"  EVPI(5,000) = {np.mean(NB > 0):.4f} × {NB[NB > 0].mean():.1f} = {np.mean(NB > 0) * NB[NB > 0].mean():.2f}")
POP = 3000
print(f"  인구 EVPI 예: 대상 환자 {POP}명이면 {L.evpi(DC, DQ) * POP / 10000:.1f}억원")
# 가격과 확률
PP = {}
for pr in (190.0, 180.0, TP, 170.0, 160.0, 152.0):
    c, q = L.psa(n=5000, seed=SEED, p=L.set_param(p, "c_drug_A", pr))
    assert np.allclose(q, DQ)
    PP[round(pr, 2)] = {"icer": L.run(L.set_param(p, "c_drug_A", pr))["icer"], "p": L.ceac(c, q, LAM), "nmb": float(L.inmb(c, q).mean()), "evpi": L.evpi(c, q)}
    print(f"  약값 {pr:7.2f}: 결정론적 ICER {PP[round(pr, 2)]['icer']:.0f}, 확률 {PP[round(pr, 2)]['p']:.4f}, 평균 순편익 {PP[round(pr, 2)]['nmb']:.1f}, EVPI {PP[round(pr, 2)]['evpi']:.1f}")
def p_at_price(pr):
    c, q = L.psa(n=5000, seed=SEED, p=L.set_param(p, "c_drug_A", pr))
    return L.ceac(c, q, LAM)
for target in (0.5, 0.8, 0.9):
    lo, hi = 100.0, 190.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if p_at_price(mid) >= target:
            lo = mid
        else:
            hi = mid
    print(f"  확률이 {target:.0%} 이상이 되는 가장 높은 약값: {lo:.1f}만원 ({100 * (1 - lo / 190):.1f}% 인하)")
N["ceac"] = {"lams": L.CEAC_LAMS.tolist(), "p": CE.tolist(), "evpi": L.evpi(DC, DQ, L.CEAC_LAMS).tolist(), "tab": {str(k): v for k, v in EV_TAB.items()},
             "lam50": lam50, "lam_evpi_max": lam_ev, "evpi_max": float(ev_fine.max()), "p_dq_pos": float(np.mean(DQ > 0)), "p_dc_neg": float(np.mean(DC < 0)),
             "price_p": {str(k): v for k, v in PP.items()}}
N["inmb"] = {"mean": float(NB.mean()), "lo": float(np.percentile(NB, 2.5)), "hi": float(np.percentile(NB, 97.5)),
             "pos_mean": float(NB[NB > 0].mean()), "neg_mean": float(NB[NB <= 0].mean()), "n_pos": int((NB > 0).sum()),
             "hist": np.histogram(NB, bins=np.arange(-2000, 2001, 100))[0].tolist(), "below": int((NB < -2000).sum()), "above": int((NB > 2000).sum())}
print("  순편익 히스토그램 범위 밖:", N["inmb"]["below"], N["inmb"]["above"], " 최소·최대", NB.min(), NB.max())

show("다. 수용곡선의 50% 지점과 ICER가 다른 예 (스스로 확인하기)")
# 치우친 분포의 작은 예: 다섯 번의 모의실험
ex_dq = np.array([0.2, 0.4, 0.5, 0.6, 1.3]); ex_dc = np.array([2000.0, 2400.0, 2500.0, 2600.0, 3000.0])
print("  평균 증분비용", ex_dc.mean(), "평균 증분QALY", ex_dq.mean(), "ICER", ex_dc.mean() / ex_dq.mean(), " 모의실험별 ICER", (ex_dc / ex_dq).round(0))
for lam in (3000, 4000, 4167, 4500, 5000, 5500, 6500, 10500):
    print(f"   λ {lam}: 확률 {np.mean(lam * ex_dq - ex_dc > 0):.1f}, 평균 순편익 {np.mean(lam * ex_dq - ex_dc):.0f}")
N["ex5"] = {"dq": ex_dq.tolist(), "dc": ex_dc.tolist(), "icer": float(ex_dc.mean() / ex_dq.mean()), "icers": (ex_dc / ex_dq).tolist()}

# 스스로 확인하기: 작은 PSA 표(10회)로 확률과 EVPI
show("스스로 확인하기: 10회 모의실험")
ex10 = np.array([-900, -650, -520, -400, -310, -250, -120, -40, 180, 510], float)
print("  평균", ex10.mean(), "양수 비율", np.mean(ex10 > 0), "EVPI", np.maximum(ex10, 0).mean() - max(ex10.mean(), 0))
ex10b = ex10 + 400
print("  +400 이동: 평균", ex10b.mean(), "양수 비율", np.mean(ex10b > 0), "E[max]", np.maximum(ex10b, 0).mean(), "EVPI", np.maximum(ex10b, 0).mean() - max(ex10b.mean(), 0))
# 스스로 확인하기(가): 토네이도 읽기, 임계 가격 손계산
print("  임계 가격 손계산(λ=6,000):", 190 + L.nmb(BASE, 6000) / pf_months_A, " 순편익(6,000):", L.nmb(BASE, 6000))
# 스스로 확인하기(나): 적률법
m_, se_ = 0.70, 0.04
n2 = m_ * (1 - m_) / se_ ** 2 - 1
print(f"  베타 연습: 평균 0.70, SE 0.04 → n {n2:.2f}, α {m_ * n2:.2f}, β {(1 - m_) * n2:.2f}; 95% {stats.beta.ppf([0.025, 0.975], m_ * n2, (1 - m_) * n2)}")
print(f"  감마 연습: 평균 300, SE 60 → shape {(300 / 60) ** 2:.1f}, scale {300 / 25:.1f}; 95% {stats.gamma.ppf([0.025, 0.975], 25, scale=12)}")
print(f"  MC 오차 연습: p 0.40, n 1,000 → SE {math.sqrt(0.4 * 0.6 / 1000):.4f}, ±{1.96 * math.sqrt(0.4 * 0.6 / 1000):.3f}; n 10,000 → ±{1.96 * math.sqrt(0.4 * 0.6 / 10000):.3f}")

# ====================================================================== 본문의 파이썬 코드(실제로 실행해 출력을 얻는다)
CODE_DSA = r'''import numpy as np
from scipy import stats
from scipy.optimize import brentq

t = np.arange(0, 241)                                   # 0, 1, ..., 240개월
base = dict(hr_pfs=0.65, hr_os=0.75,                    # 기준 분석 입력값(23장 다 절의 표)
            pfs_lam=np.log(2) / 10 ** 0.95, pfs_gam=0.95, os_lam=np.log(2) / 28 ** 1.15, os_gam=1.15,
            u_pf=0.78, u_pd=0.62, du_ae_A=0.012, du_ae_B=0.008,
            c_pf=40.0, c_pd=250.0, c_death=800.0, c_ae_A=120.0, c_ae_B=80.0,
            c_drug_A=190.0, c_drug_B=120.0)

def model(v):                                           # 입력값 한 벌 → (증분비용, 증분QALY)
    res = {}
    for arm, hr_pfs, hr_os in (("A", v["hr_pfs"], v["hr_os"]), ("B", 1.0, 1.0)):
        os_ = np.exp(-v["os_lam"] * hr_os * t ** v["os_gam"])
        pfs = np.minimum(np.exp(-v["pfs_lam"] * hr_pfs * t ** v["pfs_gam"]), os_)
        pf = (pfs[:-1] + pfs[1:]) / 2                   # 무진행 비율(반주기 보정)
        pd_ = ((os_ - pfs)[:-1] + (os_ - pfs)[1:]) / 2  # 진행 비율
        died = os_[:-1] - os_[1:]
        disc = 1 / 1.045 ** ((t[:-1] + 0.5) / 12)
        cost = ((pf * (v["c_drug_" + arm] + v["c_pf"]) + pd_ * v["c_pd"] + died * v["c_death"]) * disc).sum() + v["c_ae_" + arm]
        qaly = ((pf * v["u_pf"] + pd_ * v["u_pd"]) / 12 * disc).sum() - v["du_ae_" + arm]
        res[arm] = (cost, qaly)
    return res["A"][0] - res["B"][0], res["A"][1] - res["B"][1]

dc, dq = model(base)
print("기준 분석: 증분비용", round(dc), "증분QALY", round(dq, 3), "ICER", round(dc / dq))

z = np.array([-1.96, 1.96])
ranges = {"hr_os": np.exp(np.log(0.75) + z * 0.131),    # 위험비: 로그 척도에서 ±1.96 × 표준오차
          "hr_pfs": np.exp(np.log(0.65) + z * 0.113),
          "c_drug_A": np.array([152.0, 190.0]),         # 표시 가격에서 20% 인하한 값까지
          "u_pf": stats.beta.ppf([0.025, 0.975], 147.94, 41.727)}   # 베타분포의 95% 구간(나 절)
for name, (low, high) in ranges.items():                # 일원 민감도 분석: 하나씩만 바꾼다
    icers = []
    for value in (low, high):
        c, q = model({**base, name: value})
        icers.append(round(c / q))
    print(name, "범위", round(low, 3), "-", round(high, 3), " ICER", icers)

def gap(price):                                         # 그 가격에서의 ICER - 5,000
    c, q = model({**base, "c_drug_A": price})
    return c / q - 5000
print("ICER가 5,000이 되는 월 약값:", round(brentq(gap, 100, 190), 2))'''

CODE_PSA = r'''spec = dict(hr_pfs=("lognormal", 0.113), hr_os=("lognormal", 0.131),      # (분포, 불확실성의 크기)
            pfs_lam=("lognormal", 0.08), pfs_gam=("lognormal", 0.05),
            os_lam=("lognormal", 0.10), os_gam=("lognormal", 0.06),
            u_pf=("beta", 0.03), u_pd=("beta", 0.05),
            du_ae_A=("gamma", 0.25), du_ae_B=("gamma", 0.25),
            c_pf=("gamma", 0.20), c_pd=("gamma", 0.20), c_death=("gamma", 0.20),
            c_ae_A=("gamma", 0.25), c_ae_B=("gamma", 0.25))

def draw(rng):                                          # 입력값 한 벌을 분포에서 뽑는다
    v = dict(base)                                      # 약값은 base의 값 그대로(고정)
    for name, (dist, u) in spec.items():
        m = base[name]
        if dist == "lognormal":                         # u = 로그 척도의 표준오차
            v[name] = np.exp(rng.normal(np.log(m), u))
        elif dist == "beta":                            # u = 표준오차
            n = m * (1 - m) / u ** 2 - 1
            v[name] = rng.beta(m * n, (1 - m) * n)
        else:                                           # gamma: u = 표준오차 / 평균
            se = m * u
            shape = (m / se) ** 2
            v[name] = rng.gamma(shape, m / shape)
    return v

rng = np.random.default_rng(20261002)                   # 난수 seed를 고정해 결과를 재현
sims = np.array([model(draw(rng)) for _ in range(5000)])
dc, dq = sims[:, 0], sims[:, 1]                         # 5,000벌의 증분비용, 증분QALY

print("평균 증분비용", round(dc.mean()), " 95% 구간", np.percentile(dc, [2.5, 97.5]).round(0))
print("평균 증분QALY", round(dq.mean(), 3), " 95% 구간", np.percentile(dq, [2.5, 97.5]).round(3))
print("ICER(평균 ÷ 평균)", round(dc.mean() / dq.mean()))
print("더 비싸고 더 효과적인 비율", np.mean((dc > 0) & (dq > 0)))
inmb = 5000 * dq - dc                                   # 임계값 5,000에서의 증분 순금전편익
print("비용효과적일 확률", np.mean(inmb > 0), " 몬테카를로 표준오차", round(np.sqrt(np.mean(inmb > 0) * np.mean(inmb <= 0) / 5000), 4))'''

CODE_CEAC = r'''lams = np.array([3000, 4000, 5000, 6000, 7000, 8000, 10000])
for lam in lams:
    inmb = lam * dq - dc                                # 모의실험마다의 증분 순금전편익
    p_ce = np.mean(inmb > 0)                            # 양수인 비율 = 비용효과적일 확률
    evpi = np.maximum(inmb, 0).mean() - max(inmb.mean(), 0)
    print(lam, " 확률", round(p_ce, 3), " 평균 순편익", round(inmb.mean()), " EVPI", round(evpi, 1))'''


def run_code(src, ns=None):
    buf = io.StringIO()
    ns = {} if ns is None else ns
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(src, ns)
    return buf.getvalue().rstrip("\n"), ns


CODE = {}
out, ns = run_code(CODE_DSA)
CODE["dsa"] = {"src": CODE_DSA, "out": out, "title": "일원 민감도 분석과 임계 가격"}
show("코드 출력: dsa"); print(out)
c0, q0 = ns["model"](ns["base"])
assert abs(c0 - BASE["d_cost"]) < 1e-6 and abs(q0 - BASE["d_qaly"]) < 1e-9
assert abs(brentq(ns["gap"], 100, 190) - TP) < 1e-6
_ow = {r["key"]: r for r in OW}
for _k, (_lo, _hi) in ns["ranges"].items():        # 본문 코드의 범위(1.96, 반올림한 베타 모수)와 표 24-2의 ICER가 같은 정수인지
    _a = [round(ns["model"]({**ns["base"], _k: v})[0] / ns["model"]({**ns["base"], _k: v})[1]) for v in (_lo, _hi)]
    assert _a == [round(_ow[_k]["icer_low"]), round(_ow[_k]["icer_high"])], (_k, _a)
out, ns = run_code(CODE_PSA, ns)
CODE["psa"] = {"src": CODE_PSA, "out": out, "title": "확률적 민감도 분석 5,000회"}
show("코드 출력: psa"); print(out)
assert np.allclose(ns["dc"], DC, atol=1e-6) and np.allclose(ns["dq"], DQ, atol=1e-9), "본문 코드가 L.psa와 다른 결과를 냄"
assert np.mean(ns["inmb"] > 0) == PS["p_ce"]
out, ns = run_code(CODE_CEAC, ns)
CODE["ceac"] = {"src": CODE_CEAC, "out": out, "title": "임계값별 확률과 EVPI"}
show("코드 출력: ceac"); print(out)


def cell(c):
    return ('<div class="cell"><div class="cell-h"><span class="cell-n">직접 실행</span><span class="cell-t">' + c["title"]
            + '</span><button class="copy" type="button" aria-label="코드 복사">복사</button></div><pre class="cell-in"><code class="language-python">'
            + html.escape(c["src"]) + '</code></pre><div class="cell-out"><div class="cell-out-h">출력</div><pre>' + html.escape(c["out"]) + '</pre></div></div>')


with open(os.path.join(HERE, "_ch24_cells.html"), "w", encoding="utf-8") as f:
    for k in ("dsa", "psa", "ceac"):
        f.write("<!-- " + k + " -->\n" + cell(CODE[k]) + "\n")

# ====================================================================== 저장
N["scatter"] = {"dc": DC[:1000].round(1).tolist(), "dq": DQ[:1000].round(4).tolist()}
N["scatter_p1000"] = float(np.mean(L.inmb(DC[:1000], DQ[:1000]) > 0))
with open(os.path.join(HERE, "_ch24_nums.json"), "w", encoding="utf-8") as f:
    json.dump(N, f, ensure_ascii=False, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
print("\nsaved _ch24_nums.json, _ch24_cells.html")
