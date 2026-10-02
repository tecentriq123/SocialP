"""25장(경제성 평가 논문 읽기)의 모든 숫자를 계산한다.
실행: source /home/claude/pylibs/env.sh && python3 gen/nums_ch25.py   (그 뒤 python3 gen/fig_ch25.py)

공통 예시(gen/lib_p4.py)의 기준 분석, 일원 민감도 분석 L.oneway(), 시나리오 L.scenario_table(), 임계 가격 L.threshold_price(),
확률적 민감도 분석 L.psa(n=5000, seed=20261002)와 L.ceac, L.evpi, L.psa_summary를 그대로 불러 쓴다(lib_p4.py는 고치지 않는다).
이 장에서 새로 계산하는 것:
  나 절  증분 QALY·증분비용 가운데 시험 추적(30개월) 뒤에 생기는 몫, 가상 논문의 표(₩ thousand, ₩ million 단위)
  다 절  환급률과 표시 가격·실제 가격의 ICER, 5년 재정영향분석(가상의 환자 수와 점유율)
모든 값은 가상의 예시이다. 결과는 gen/_ch25_nums.json(그림용)에 저장한다.
"""
import contextlib, io, json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import lib_p4 as L

SEED = 20261002
LAM = L.THRESHOLD
N = {}


def show(title):
    print("\n" + "=" * 100 + "\n" + title + "\n" + "=" * 100)


def r0(x):
    """반올림(사사오입이 아니라 파이썬 round는 은행가 반올림이므로 0.5를 올림으로 고정)."""
    return int(math.floor(x + 0.5))


p = L.base_params()
BASE = L.run()
A, B = BASE["A"], BASE["B"]

# ====================================================================== 기준 분석 (가·나 절의 가상 논문)
show("기준 분석: 만원 단위와 논문 표기(₩ thousand = 천 원, ₩ million = 백만 원)")
assert round(BASE["icer"]) == 5621 and round(BASE["d_cost"]) == 2872 and round(BASE["d_qaly"], 3) == 0.511 and round(L.nmb(BASE)) == -317
comp = [("c_drug", "Drug acquisition"), ("c_pf", "Disease management, progression-free"), ("c_pd", "Disease management, progressed disease"),
        ("c_death", "End-of-life care"), ("c_ae", "Adverse events")]
tab = {}
for k, en in comp + [("cost", "Total costs")]:
    a_k, b_k = r0(A[k] * 10), r0(B[k] * 10)          # 만원 → 천 원
    tab[k] = (a_k, b_k, a_k - b_k)
    print(f"  {en:42s} A {a_k:>8,d}  B {b_k:>8,d}  증분 {a_k - b_k:>7,d} (천 원) | 만원 {A[k]:.1f} {B[k]:.1f} {A[k] - B[k]:.1f}")
assert sum(tab[k][0] for k, _ in comp) == tab["cost"][0] and sum(tab[k][1] for k, _ in comp) == tab["cost"][1]
print(f"  약값이 증분비용에서 차지하는 몫 {tab['c_drug'][2] / tab['cost'][2]:.3f}")
print(f"  생존연수(할인 전) A {A['ly']:.3f} B {B['ly']:.3f} 증분 {A['ly'] - B['ly']:.3f} ({(A['ly'] - B['ly']) * 12:.1f}개월)")
print(f"  생존연수(할인 후) A {A['ly_d']:.3f} B {B['ly_d']:.3f} 증분 {BASE['d_ly']:.3f}")
print(f"  무진행 기간(할인 전, 년) A {A['ly_pf']:.3f} B {B['ly_pf']:.3f} 증분 {A['ly_pf'] - B['ly_pf']:.3f} ({(A['ly_pf'] - B['ly_pf']) * 12:.1f}개월)")
print(f"  QALY A {A['qaly']:.3f} B {B['qaly']:.3f} 증분 {BASE['d_qaly']:.4f}")
qa, qb = round(A["qaly"], 3), round(B["qaly"], 3)
la, lb = round(A["ly_d"], 3), round(B["ly_d"], 3)
print(f"  표의 반올림 값으로 검산: 증분 QALY {qa - qb:.3f}, ICER {tab['cost'][2] / (qa - qb) / 1000:.2f} 백만 원/QALY, 생존연수당 {tab['cost'][2] / (la - lb) / 1000:.2f}")
print(f"  ₩ million: 비용 A {A['cost'] / 100:.1f} B {B['cost'] / 100:.1f} 증분 {BASE['d_cost'] / 100:.1f} | ICER {BASE['icer'] / 100:.1f} | 생존연수당 {BASE['icer_ly'] / 100:.1f}")
print(f"  표의 ₩ million 값으로: ({A['cost'] / 100:.1f} − {B['cost'] / 100:.1f}) ÷ {qa - qb:.3f} = {(round(A['cost'] / 100, 1) - round(B['cost'] / 100, 1)) / (qa - qb):.2f}")
print(f"  평균 비용효과비 A {A['cost'] / A['qaly']:.0f} B {B['cost'] / B['qaly']:.0f} (ICER가 아님)")
N["base"] = {"tab": tab, "icer": BASE["icer"], "d_cost": BASE["d_cost"], "d_qaly": BASE["d_qaly"], "icer_ly": BASE["icer_ly"],
             "qaly_A": A["qaly"], "qaly_B": B["qaly"], "ly_A": A["ly_d"], "ly_B": B["ly_d"], "lyu_A": A["ly"], "lyu_B": B["ly"],
             "nmb": L.nmb(BASE)}

show("입력값(₩ thousand)과 중앙값")
for k in ("c_drug_A", "c_drug_B", "c_pf", "c_pd", "c_death", "c_ae_A", "c_ae_B"):
    print(f"  {k}: {p[k]:.0f}만원 = ₩{p[k] * 10:,.0f} thousand" + (f" (SE {p[k] * L.PSA_SPEC[k][1] * 10:,.0f})" if k in L.PSA_SPEC else ""))
med = {"pfs_B": L.weib_median(p["pfs_lam"], p["pfs_gam"]), "os_B": L.weib_median(p["os_lam"], p["os_gam"]),
       "pfs_A": L.weib_median(p["pfs_lam"] * p["hr_pfs"], p["pfs_gam"]), "os_A": L.weib_median(p["os_lam"] * p["hr_os"], p["os_gam"])}
print("  중앙값(개월):", {k: round(v, 1) for k, v in med.items()})
print(f"  와이블 모수: PFS lam {p['pfs_lam']:.4f} gam {p['pfs_gam']}, OS lam {p['os_lam']:.4f} gam {p['os_gam']}")
for k in ("hr_pfs", "hr_os", "u_pf", "u_pd"):
    d = L.dist_params(k)
    print(f"  {k}: {d['base']} (95% {d['lo']:.2f}–{d['hi']:.2f})")
N["median"] = med

# ====================================================================== 나 절: 시험 추적 30개월 뒤에 생기는 몫
show("나. 증분 결과 가운데 시험 추적 기간(30개월) 뒤에 생기는 몫")
FU = 30
t_chk = np.array([12, 24, FU, 60, 120, 240])
for arm in L.ARMS:
    s_pfs, s_os = L.curves(p, arm, t_chk)
    print(f"  {arm}: t = {t_chk.tolist()}  OS {np.round(s_os * 100, 1).tolist()}  PFS {np.round(s_pfs * 100, 1).tolist()}")
sA30, sB30 = float(L.curves(p, "A", [FU])[1][0]), float(L.curves(p, "B", [FU])[1][0])
print(f"  30개월 생존율: A {sA30:.3f}, B {sB30:.3f}")
rows = []
for h in (12, 24, FU, 36, 60, 120, 180, 240):
    r = L.run(horizon=h)
    rows.append((h, r["d_qaly"], r["d_cost"], r["A"]["ly"] - r["B"]["ly"], r["icer"]))
    print(f"  분석기간 {h:3d}개월: 증분 QALY {r['d_qaly']:.4f} ({r['d_qaly'] / BASE['d_qaly'] * 100:5.1f}%)  증분비용 {r['d_cost']:7.1f} ({r['d_cost'] / BASE['d_cost'] * 100:5.1f}%)"
          f"  증분 생존연수(할인 전) {r['A']['ly'] - r['B']['ly']:.3f}  ICER {r['icer']:.0f}")
R30 = L.run(horizon=FU)
share_q = 1 - R30["d_qaly"] / BASE["d_qaly"]
share_c = 1 - R30["d_cost"] / BASE["d_cost"]
share_ly = 1 - (R30["A"]["ly"] - R30["B"]["ly"]) / (A["ly"] - B["ly"])
print(f"  ⇒ 30개월 안: 증분 QALY {R30['d_qaly']:.3f}, 증분비용 {R30['d_cost']:.0f}, ICER {R30['icer']:.0f}")
print(f"  ⇒ 30개월 뒤의 몫: 증분 QALY {share_q * 100:.1f}% ({BASE['d_qaly'] - R30['d_qaly']:.3f}), 증분비용 {share_c * 100:.1f}%, 증분 생존연수 {share_ly * 100:.1f}%")
# 군별 QALY 가운데 30개월 뒤의 몫
for arm in L.ARMS:
    print(f"  {arm}군 QALY 가운데 30개월 뒤: {(1 - R30[arm]['qaly'] / BASE[arm]['qaly']) * 100:.1f}%")
# 누적 증분 QALY 곡선(그림용): 분석기간 h개월까지의 증분 QALY
hs = list(range(0, 241, 3))
cum = [0.0] + [L.run(horizon=h)["d_qaly"] for h in hs[1:]]
cum[0] = -(p["du_ae_A"] - p["du_ae_B"])
N["extrap"] = {"fu": FU, "sA30": sA30, "sB30": sB30, "dq30": R30["d_qaly"], "dc30": R30["d_cost"], "icer30": R30["icer"],
               "share_q": share_q, "share_c": share_c, "share_ly": share_ly, "hs": hs, "cum": cum}
# 전체생존 이득이 없다면(22장 다 절): hr_os = 1
r_noos = L.run(L.set_param(p, "hr_os", 1.0))
print(f"  전체생존 위험비를 1로 두면 증분 QALY {r_noos['d_qaly']:.3f}, 증분비용 {r_noos['d_cost']:.0f}, ICER {r_noos['icer']:.0f}")
N["no_os"] = {"d_qaly": r_noos["d_qaly"], "d_cost": r_noos["d_cost"], "icer": r_noos["icer"]}

# ====================================================================== 나 절: 일원 민감도 분석과 시나리오
show("나. 일원 민감도 분석 (L.oneway) — 논문 그림과 표, ₩ million per QALY")
OW = L.oneway()
for r in OW[:6]:
    print(f"  {r['label']:22s} {r['base']:.5g} ({r['low']:.5g}–{r['high']:.5g}) [{r['basis']}]  ICER {r['icer_low']:.0f} → {r['icer_high']:.0f}"
          f"  = ₩{r['icer_low'] / 100:.1f} → {r['icer_high'] / 100:.1f} million  폭 {r['swing']:.0f}  임계값 넘나듦 {r['crosses']}")
print("  7번째 이하 폭의 최댓값:", round(max(r["swing"] for r in OW[6:])))
N["oneway"] = [{k: r[k] for k in ("key", "label", "base", "low", "high", "icer_low", "icer_high", "swing", "basis", "crosses")} for r in OW[:6]]

show("나. 시나리오 분석 (L.scenario_table)")
with contextlib.redirect_stderr(io.StringIO()):
    SC = L.scenario_table()
for r in SC:
    print(f"  {r['key']:12s} {r['label_en']:52s} 증분비용 {r['d_cost']:6.0f} 증분QALY {r['d_qaly']:.3f} ICER {r['icer']:5.0f} (₩{r['icer'] / 100:.1f} million) {100 * (r['icer'] / BASE['icer'] - 1):+.1f}%")
ic = [r["icer"] for r in SC if r["key"] != "base"]
print(f"  시나리오 {len(ic)}개의 ICER 범위 {min(ic):.0f}–{max(ic):.0f}")
N["scen"] = {r["key"]: {"d_cost": r["d_cost"], "d_qaly": r["d_qaly"], "icer": r["icer"], "label_en": r["label_en"]} for r in SC}
N["scen_range"] = [min(ic), max(ic), len(ic)]

# ====================================================================== 나·다 절: 임계 가격과 환급
show("임계 가격 (L.threshold_price)과 환급률")
TP = L.threshold_price()
cut = 1 - TP / p["c_drug_A"]
months_A = A["c_drug"] / p["c_drug_A"]
print(f"  ICER = 5,000이 되는 월 약값 {TP:.2f}만원 (표시 가격 {p['c_drug_A']:.0f}만원에서 {p['c_drug_A'] - TP:.1f}만원, {cut * 100:.1f}% 인하)")
print(f"  신약 A군이 약을 쓰는 할인된 개월 수 {months_A:.2f}; 약값 1만원당 ICER {months_A / BASE['d_qaly']:.1f}")
price_rows = []
for rebate in (0.0, 0.05, cut, 0.10, 0.15, 0.20):
    pr = p["c_drug_A"] * (1 - rebate)
    r = L.run(L.set_param(p, "c_drug_A", pr))
    price_rows.append((rebate, pr, r["d_cost"], r["icer"], L.nmb(r)))
    print(f"  환급률 {rebate * 100:5.1f}% → 실제 가격 {pr:6.1f}만원: 증분비용 {r['d_cost']:.0f}, ICER {r['icer']:.0f}, 순편익 {L.nmb(r):.0f}")
for lam in (4000.0, 5000.0, 6000.0):
    print(f"  임계값 {lam:.0f}에서의 임계 가격 {L.threshold_price(lam):.1f}만원")
N["price"] = {"tp": TP, "cut": cut, "months_A": months_A, "rows": price_rows}

# ====================================================================== 나 절: 확률적 민감도 분석
show("나. 확률적 민감도 분석 (L.psa 5,000회, seed 20261002)")
dc, dq = L.psa(n=5000, seed=SEED)
PS = L.psa_summary(dc, dq)
print(f"  평균 증분비용 {PS['d_cost']['mean']:.0f} ({PS['d_cost']['lo']:.0f}–{PS['d_cost']['hi']:.0f}) = ₩{PS['d_cost']['mean'] / 100:.1f} ({PS['d_cost']['lo'] / 100:.1f}–{PS['d_cost']['hi'] / 100:.1f}) million")
print(f"  평균 증분 QALY {PS['d_qaly']['mean']:.3f} ({PS['d_qaly']['lo']:.3f}–{PS['d_qaly']['hi']:.3f})")
print(f"  확률적 ICER {PS['icer']:.0f}; 순편익 {PS['inmb']['mean']:.0f} ({PS['inmb']['lo']:.0f}–{PS['inmb']['hi']:.0f}); 사분면 {PS['quadrants']}")
print(f"  비용효과적일 확률(5,000) {PS['p_ce']:.4f} (MCSE {PS['mcse']:.4f}); EVPI {PS['evpi']:.1f}만원")
ce = {}
for lam in (3000, 4000, 5000, 5500, 6000, 7000, 8000, 10000):
    ce[lam] = (L.ceac(dc, dq, float(lam)), L.evpi(dc, dq, float(lam)))
    print(f"  임계값 {lam:6d}: 확률 {ce[lam][0] * 100:5.1f}%  EVPI {ce[lam][1]:.1f}")
curve = L.ceac(dc, dq)
i50 = int(np.argmax(curve >= 0.5))
print(f"  수용곡선이 50%를 넘는 임계값(100만원 격자): {L.CEAC_LAMS[i50]:.0f}")
print(f"  QALY가 늘어나는 모의실험 비율 {np.mean(dq > 0):.4f}")
# 임계 가격에서의 확률(같은 난수)
p_tp = dict(p); p_tp["c_drug_A"] = TP
dc_tp, dq_tp = L.psa(n=5000, seed=SEED, p=p_tp)
print(f"  임계 가격 {TP:.1f}만원에서 비용효과적일 확률 {L.ceac(dc_tp, dq_tp, LAM) * 100:.1f}%")
N["psa"] = {"summary": PS, "ceac": {str(k): v[0] for k, v in ce.items()}, "evpi": {str(k): v[1] for k, v in ce.items()},
            "lams": L.CEAC_LAMS.tolist(), "curve": curve.tolist(), "lam50": float(L.CEAC_LAMS[i50]), "p_qaly_gain": float(np.mean(dq > 0)),
            "p_ce_tp": L.ceac(dc_tp, dq_tp, LAM), "scatter_dq": dq[:500].tolist(), "scatter_dc": dc[:500].tolist()}

# ====================================================================== 다 절: 재정영향분석
show("다. 재정영향분석 (가상의 환자 수와 점유율, 할인하지 않음)")
YEARS = 5
N_NEW = 600                                   # 해마다 1차 치료를 새로 시작하는 환자 수(가상)
UPTAKE = [0.10, 0.20, 0.30, 0.40, 0.50]       # 그해 새로 시작하는 환자 가운데 신약 A로 시작하는 비율(가상)


def yearly_cost(pp, arm, years=YEARS, comp_key="cost"):
    """치료 시작 후 k년차(1..years)에 환자 1명에게 드는 평균 비용(만원, 할인하지 않음).
    L.psm(…, horizon=12k, disc=0)의 누적 비용을 한 해씩 뺀다. 이상반응 비용(1회)은 1년차에 들어간다."""
    cum = [0.0]
    for k in range(1, years + 1):
        x = L.psm(pp, arm, horizon=12 * k, disc=0.0)
        v = x[comp_key]
        cum.append(v)
    out = np.diff(cum)
    return out


cA, cB = yearly_cost(p, "A"), yearly_cost(p, "B")
dA, dB = yearly_cost(p, "A", comp_key="c_drug"), yearly_cost(p, "B", comp_key="c_drug")
print("  치료 시작 후 연차별 1인당 비용(만원, 할인 전)")
print("   연차      :", list(range(1, YEARS + 1)))
print("   A 전체    :", np.round(cA, 1).tolist(), "5년 합", round(cA.sum(), 1))
print("   B 전체    :", np.round(cB, 1).tolist(), "5년 합", round(cB.sum(), 1))
print("   차이(A−B) :", np.round(cA - cB, 1).tolist(), "5년 합", round((cA - cB).sum(), 1))
print("   A 약값    :", np.round(dA, 1).tolist())
print("   B 약값    :", np.round(dB, 1).tolist())
# 연차별 생존·무진행 비율(연초 기준)로 검산용 출력
for arm in L.ARMS:
    s_pfs, s_os = L.curves(p, arm, np.arange(0, 61, 12))
    print(f"   {arm} 연초 무진행 비율 {np.round(s_pfs, 3).tolist()} 생존 {np.round(s_os, 3).tolist()}")


def bia(cA, cB, dA, n_new=N_NEW, uptake=UPTAKE):
    """달력 연도 t(1..5)의 지출(만원). 해마다 n_new명이 새로 치료를 시작하고(열린 코호트),
    그해 시작한 환자의 uptake[s]가 신약 A로, 나머지는 표준요법 B로 시작한다. 이미 B로 치료 중인 환자는 바꾸지 않는다고 가정."""
    Y = len(uptake)
    n_A = [n_new * u for u in uptake]                       # s년에 A로 시작한 환자 수
    world0 = np.zeros(Y); world1 = np.zeros(Y); gross_A = np.zeros(Y); displaced = np.zeros(Y); drug_A = np.zeros(Y)
    on_A = np.zeros(Y)
    for t in range(Y):
        for s in range(t + 1):
            k = t - s                                       # 치료 시작 후 k+1년차
            world0[t] += n_new * cB[k]
            world1[t] += n_A[s] * cA[k] + (n_new - n_A[s]) * cB[k]
            gross_A[t] += n_A[s] * cA[k]                    # A로 시작한 환자에게 드는 비용 전체
            displaced[t] += n_A[s] * cB[k]                  # 그 환자들이 B를 썼다면 들었을 비용
            drug_A[t] += n_A[s] * dA[k]                     # 신약 A의 약품비만
    return {"n_A": n_A, "world0": world0, "world1": world1, "net": world1 - world0, "gross_A": gross_A, "displaced": displaced, "drug_A": drug_A}


BI = bia(cA, cB, dA)
eok = lambda v: v / 10000.0                                # 만원 → 억 원
print("  연도        :", list(range(1, YEARS + 1)))
print("  A로 시작한 환자 수:", [round(x) for x in BI["n_A"]], "누적", [round(x) for x in np.cumsum(BI["n_A"])])
for k, lab in (("world0", "신약 A가 없을 때의 지출"), ("world1", "신약 A를 급여할 때의 지출"), ("net", "재정영향(순증가)"),
               ("gross_A", "A로 시작한 환자의 비용 전체"), ("displaced", "대체된 표준요법 B의 비용"), ("drug_A", "신약 A 약품비")):
    print(f"  {lab:28s} (억 원): {np.round(eok(BI[k]), 1).tolist()}  5년 합 {eok(BI[k].sum()):.1f}")
assert np.allclose(BI["net"], BI["gross_A"] - BI["displaced"])
print(f"  3년차 재정영향 {eok(BI['net'][2]):.1f}억 원, 5년 합 {eok(BI['net'].sum()):.1f}억 원; 신약 약품비 5년 합 {eok(BI['drug_A'].sum()):.1f}억 원 (순증가의 {BI['drug_A'].sum() / BI['net'].sum():.2f}배)")
print(f"  기준 지출 대비 증가율(5년차) {BI['net'][4] / BI['world0'][4] * 100:.1f}%")
print(f"  공단 부담금만(산정특례 본인부담 5%를 뺀 0.95배) 5년 합 {eok(BI['net'].sum()) * 0.95:.1f}억 원, 3년차 {eok(BI['net'][2]) * 0.95:.1f}")
N["bia"] = {"n_new": N_NEW, "uptake": UPTAKE, "cA": cA.tolist(), "cB": cB.tolist(), "dA": dA.tolist(), "dB": dB.tolist(),
            **{k: (np.asarray(v).tolist()) for k, v in BI.items()}}

show("다. 재정영향분석의 시나리오와 흔한 실수")
# (1) 임계 가격(175.4만원)에서의 재정영향: 비용효과적이어도 재정 부담은 남는다
cA_tp, dA_tp = yearly_cost(p_tp, "A"), yearly_cost(p_tp, "A", comp_key="c_drug")
BI_tp = bia(cA_tp, cB, dA_tp)
print(f"  임계 가격 {TP:.1f}만원: 재정영향 {np.round(eok(BI_tp['net']), 1).tolist()} 5년 합 {eok(BI_tp['net'].sum()):.1f}억 원 (표시 가격 대비 {BI_tp['net'].sum() / BI['net'].sum() * 100:.1f}%)")
# (2) 점유율이 절반일 때 / 두 배 빠를 때
for nm, up in (("점유율 절반", [u / 2 for u in UPTAKE]), ("점유율 1.5배(상한 75%)", [min(u * 1.5, 0.75) for u in UPTAKE]), ("환자 수 900명", UPTAKE)):
    b2 = bia(cA, cB, dA, n_new=900 if "900" in nm else N_NEW, uptake=up)
    print(f"  {nm}: 재정영향 {np.round(eok(b2['net']), 1).tolist()} 5년 합 {eok(b2['net'].sum()):.1f}억 원")
N["bia_alt"] = {"tp": {"net": BI_tp["net"].tolist()},
                "half": bia(cA, cB, dA, uptake=[u / 2 for u in UPTAKE])["net"].tolist(),
                "n900": bia(cA, cB, dA, n_new=900)["net"].tolist()}
# (3) 실수: 5년치를 할인하면
disc = np.array([1 / 1.045 ** (t + 0.5) for t in range(YEARS)])
print(f"  (실수) 연 4.5%로 할인한 5년 합 {eok((BI['net'] * disc).sum()):.1f}억 원 (할인하지 않은 값 {eok(BI['net'].sum()):.1f})")
# (4) 실수: 평생 증분비용 × 5년간 A로 시작한 환자 수
n_tot = sum(BI["n_A"])
print(f"  (실수) 1인당 평생 증분비용 {BASE['d_cost']:.0f}만원 × 5년간 A로 시작한 {n_tot:.0f}명 = {eok(BASE['d_cost'] * n_tot):.1f}억 원 (5년 안에 실제로 드는 순증가 {eok(BI['net'].sum()):.1f})")
# (5) 대비: 값싸고 비용효과적이지만 환자가 많은 약(가상)
n_big, dcost_big, dq_big = 300000, 60.0, 0.04      # 환자 30만 명, 1인당 연 60만원 추가, 연 0.04 QALY 추가
print(f"  (대비) 흔한 만성질환의 가상 신약: ICER {dcost_big / dq_big:.0f}만원/QALY, 연간 재정영향 {eok(n_big * dcost_big):.0f}억 원")
N["contrast"] = {"n": n_big, "dcost": dcost_big, "dq": dq_big, "icer": dcost_big / dq_big, "bi": eok(n_big * dcost_big)}

show("스스로 확인하기용 계산")
# 가 절: 초록의 숫자 검산
print(f"  초록: (105.6 − 76.9) ÷ (2.385 − 1.874) = {(105.6 - 76.9) / (2.385 - 1.874):.2f}")
# 나 절: 다른 논문 예 — 분석기간 내 비율
# 다 절: 재정영향 연습 문제 (환자 2,000명, 점유율 10·20·30%, 연 비용 신약 900 기존 500)
n_q, up_q, c_new, c_old = 2000, [0.10, 0.20, 0.30], 900.0, 500.0
net_q = [n_q * u * (c_new - c_old) for u in up_q]
gross_q = [n_q * u * c_new for u in up_q]
print(f"  연습: 순증가 {[round(eok(x), 1) for x in net_q]}억 원 (3년 합 {eok(sum(net_q)):.1f}), 신약 약품비 {[round(eok(x), 1) for x in gross_q]} (3년 합 {eok(sum(gross_q)):.1f})")
N["practice"] = {"net": net_q, "gross": gross_q}
# 환급률 연습: 표시 가격 200, 환급 15% → 실제 170
print(f"  환급형 예: 표시 가격 190만원, 환급률 {cut * 100:.1f}% → 실제 {TP:.1f}만원; 환급액 월 {p['c_drug_A'] - TP:.1f}만원")

show("추가: 평균 생존(개월), 3년차 재정영향의 분해, 논문 상자의 재정영향 표(₩ million), 희귀질환 대비 예")
print(f"  평균 생존(할인 전): A {A['ly'] * 12:.2f}개월, B {B['ly'] * 12:.2f}개월, 차이 {(A['ly'] - B['ly']) * 12:.2f}개월; 년 A {A['ly']:.2f} B {B['ly']:.2f} 차이 {A['ly'] - B['ly']:.2f}")
d = cA - cB
terms = [BI["n_A"][s] * d[2 - s] for s in range(3)]
print("  3년차 재정영향 = " + " + ".join(f"{BI['n_A'][s]:.0f}명 × {d[2 - s]:.1f}" for s in range(3)) + " = " + " + ".join(f"{x:,.0f}" for x in terms) + f" = {sum(terms):,.0f}만원 = {eok(sum(terms)):.2f}억 원")
print("  5년차 재정영향 = " + " + ".join(f"{BI['n_A'][s]:.0f} × {d[4 - s]:.1f}" for s in range(5)) + f" = {sum(BI['n_A'][s] * d[4 - s] for s in range(5)):,.0f}만원")
m0 = [r0(x / 100) for x in BI["world0"]]; m1 = [r0(x / 100) for x in BI["world1"]]
mnet = [b - a for a, b in zip(m0, m1)]
print("  ₩ million  without:", m0, "합", sum(m0))
print("  ₩ million  with   :", m1, "합", sum(m1))
print("  ₩ million  net    :", mnet, "합", sum(mnet), "| 반올림 전 값의 반올림:", [r0(x / 100) for x in BI["net"]], r0(BI["net"].sum() / 100))
print("  ₩ million  신약 A 약품비:", [r0(x / 100) for x in BI["drug_A"]], "합", r0(BI["drug_A"].sum() / 100))
print("  증가율(%):", [round(n_ / w * 100, 1) for n_, w in zip(mnet, m0)])
N["bia_paper"] = {"without": m0, "with": m1, "net": mnet, "drug_A": [r0(x / 100) for x in BI["drug_A"]]}
n_rare, dcost_rare, dq_rare = 40, 9000.0, 0.45       # 환자 40명, 1인당 연 9,000만원 추가, 연 0.45 QALY 추가(가상)
print(f"  (대비) 희귀질환의 가상 신약: ICER {dcost_rare / dq_rare:.0f}만원/QALY, 연간 재정영향 {eok(n_rare * dcost_rare):.0f}억 원")
N["contrast_rare"] = {"n": n_rare, "dcost": dcost_rare, "dq": dq_rare, "icer": dcost_rare / dq_rare, "bi": eok(n_rare * dcost_rare)}

show("본문 접힌 코드 상자의 실제 실행 결과")
CELLS = {}
CELLS["share"] = '''import numpy as np

def model(horizon):                                   # 분석기간(개월)을 넣으면 (증분비용, 증분QALY)를 돌려준다
    t = np.arange(0, horizon + 1)
    res = {}
    for arm, hr_pfs, hr_os, c_drug, c_ae, dq_ae in (("A", 0.65, 0.75, 190, 120, 0.012), ("B", 1.0, 1.0, 120, 80, 0.008)):
        os_ = np.exp(-np.log(2) / 28 ** 1.15 * hr_os * t ** 1.15)                    # 전체생존 곡선
        pfs = np.minimum(np.exp(-np.log(2) / 10 ** 0.95 * hr_pfs * t ** 0.95), os_)  # 무진행생존 곡선
        pf = (pfs[:-1] + pfs[1:]) / 2                 # 무진행 비율(반주기 보정)
        pd_ = ((os_ - pfs)[:-1] + (os_ - pfs)[1:]) / 2
        died = os_[:-1] - os_[1:]
        disc = 1 / 1.045 ** ((t[:-1] + 0.5) / 12)     # 할인 계수
        cost = ((pf * (c_drug + 40) + pd_ * 250 + died * 800) * disc).sum() + c_ae
        qaly = ((pf * 0.78 + pd_ * 0.62) / 12 * disc).sum() - dq_ae
        res[arm] = (cost, qaly)
    return res["A"][0] - res["B"][0], res["A"][1] - res["B"][1]

dc_all, dq_all = model(240)                           # 기준 분석: 20년
dc_obs, dq_obs = model(30)                            # 시험이 관찰한 30개월에서 끊으면
print("20년  :", round(dc_all), round(dq_all, 3), round(dc_all / dq_all))
print("30개월:", round(dc_obs), round(dq_obs, 3), round(dc_obs / dq_obs))
print("30개월 뒤의 몫: QALY", round(1 - dq_obs / dq_all, 3), " 비용", round(1 - dc_obs / dc_all, 3))'''
CELLS["bia"] = '''import numpy as np

cost_A = np.array([2820.7, 2226.8, 1763.1, 1366.9, 1044.3])   # 치료 시작 후 1-5년차의 1인당 비용(만원), 신약 A
cost_B = np.array([2200.1, 1821.0, 1398.5, 1021.5, 723.8])    # 표준요법 B
n_new = 600                                                    # 해마다 치료를 새로 시작하는 환자 수
share = np.array([0.10, 0.20, 0.30, 0.40, 0.50])               # 그해 시작하는 환자 가운데 신약 A의 비율
start_A = n_new * share                                        # 연도별로 신약 A를 시작한 환자 수

net = np.zeros(5)
for year in range(5):                                          # 급여 후 연차(0 = 첫해)
    for start in range(year + 1):                              # 그해까지 시작한 코호트를 모두 더한다
        k = year - start                                       # 그 코호트의 치료 연차(0 = 1년차)
        net[year] += start_A[start] * (cost_A[k] - cost_B[k])
print(start_A)
print(np.round(net / 10000, 1), round(net.sum() / 10000, 1))   # 억 원'''
outs = {}
for k, code in CELLS.items():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    outs[k] = buf.getvalue().rstrip()
    print(f"--- {k}\n{outs[k]}")
import html as _html
with open(os.path.join(HERE, "_ch25_cells.html"), "w", encoding="utf-8") as f:
    for k, code in CELLS.items():
        f.write(f"<!-- {k} -->\n<pre class=\"cell-in\"><code class=\"language-python\">{_html.escape(code)}</code></pre>"
                f"<div class=\"cell-out\"><div class=\"cell-out-h\">출력</div><pre>{_html.escape(outs[k])}</pre></div>\n\n")

with open(os.path.join(HERE, "_ch25_nums.json"), "w", encoding="utf-8") as f:
    json.dump(N, f, ensure_ascii=False, indent=1, default=float)
print("\nwrote gen/_ch25_nums.json")
