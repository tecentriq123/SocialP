"""Numbers for 종합 5 (qz95, economic evaluation for reimbursement). Hypothetical: moderate-to-severe plaque psoriasis,
new biologic B vs the most-used biologic U (and older biologic T). Health-system perspective, 40-year horizon,
3-month cycles, 4.5%/yr discount on costs and QALYs, life-table (half-cycle) correction.
States: on drug with response (R), off drug / best supportive care (S), dead (D). Response assessed at week 12.
Run: python3 gen/quiz_rv5.py"""
import math
import numpy as np

CYC = 0.25; YEARS = 40; NC = int(YEARS / CYC); DISC = 0.045; AGE0 = 45
U_R, U_S = 0.89, 0.60                         # EQ-5D-5L (Korean value set) utilities, hypothetical
C_MON, C_S = 15.0, 150.0                       # monitoring per cycle on drug; BSC cost per cycle (만원)
DRUG = {"B": 420.0, "U": 400.0, "T": 420.0}    # drug cost per 3-month cycle (만원)
RESP_U = 0.70; OR_BU = 1.80; RESP_T = 0.55    # week-12 PASI 75 response; NMA OR for B vs U
P_STOP_YR = 0.30                              # 1-year discontinuation among responders (Kaplan-Meier 1 - S(1y))


def resp_from_or(p0, OR):
    o = p0 / (1 - p0) * OR
    return o / (1 + o)


RESP = {"U": RESP_U, "B": resp_from_or(RESP_U, OR_BU), "T": RESP_T}
r_stop = -math.log(1 - P_STOP_YR); p_stop = 1 - math.exp(-r_stop * CYC)


def p_die(age):
    q = 0.004 * math.exp(0.09 * (age - 45))   # annual mortality, Gompertz-like, hypothetical
    q = min(q, 0.99)
    return 1 - (1 - q) ** CYC


def run(arm, drug_cost=None, resp=None, u_r=U_R, u_s=U_S, horizon=YEARS, p_stop_c=None):
    dc = DRUG[arm] if drug_cost is None else drug_cost
    rp = RESP[arm] if resp is None else resp
    ps = p_stop if p_stop_c is None else p_stop_c
    nc = int(horizon / CYC)
    # cycle 1: induction, everyone on drug; response realised at end of cycle 1
    R, S, D = rp, 1 - rp, 0.0
    d = 1.0
    cost = dc + C_MON
    qaly = (u_s + 0.5 * rp * (u_r - u_s)) * CYC  # utility rises during induction for responders
    drug_years = CYC
    for t in range(1, nc):
        age = AGE0 + t * CYC; pdie = p_die(age)
        R1 = R * (1 - pdie) * (1 - ps); S1 = S * (1 - pdie) + R * (1 - pdie) * ps; D1 = D + (R + S) * pdie
        Rm, Sm = (R + R1) / 2, (S + S1) / 2      # life-table (half-cycle) correction
        d = 1 / (1 + DISC) ** (t * CYC)
        cost += d * (Rm * (dc + C_MON) + Sm * C_S)
        qaly += d * CYC * (Rm * u_r + Sm * u_s)
        drug_years += d * CYC * Rm
        R, S, D = R1, S1, D1
    return cost, qaly, drug_years


res = {a: run(a) for a in ("B", "U", "T")}
print("response: U %.1f%% B %.1f%% (OR %.2f) T %.1f%% | wrong B=0.70*1.8 = %.0f%%" % (100 * RESP["U"], 100 * RESP["B"], OR_BU, 100 * RESP["T"], 100 * 0.70 * 1.8))
print("discontinuation: 1y 30%% -> rate %.4f -> 3-month prob %.2f%% | wrong 30/4 = 7.50%%" % (r_stop, 100 * p_stop))
for a in ("B", "U", "T"):
    c, q, dy = res[a]; print("  %s cost %.0f QALY %.3f discounted drug-years %.2f" % (a, c, q, dy))
dC = res["B"][0] - res["U"][0]; dQ = res["B"][1] - res["U"][1]
print("B vs U: dC %.1f dQ %.4f ICER %.0f" % (dC, dQ, dC / dQ))
dC_T = res["B"][0] - res["T"][0]; dQ_T = res["B"][1] - res["T"][1]
print("B vs T: dC %.0f dQ %.3f ICER %.0f" % (dC_T, dQ_T, dC_T / dQ_T))
dC_UT = res["U"][0] - res["T"][0]; dQ_UT = res["U"][1] - res["T"][1]
print("U vs T: dC %.0f dQ %.3f ICER %.0f" % (dC_UT, dQ_UT, dC_UT / dQ_UT))
print("average ratio B %.0f U %.0f" % (res["B"][0] / res["B"][1], res["U"][0] / res["U"][1]))
for thr in (3000, 5000, 7000):
    print("  NMB at %d: %.0f" % (thr, thr * dQ - dC))
print("threshold: max dC at 5000 = %.0f -> B total cost may rise by %.0f" % (5000 * dQ, 5000 * dQ - dC))
# threshold B price per cycle making ICER = 5000
lo, hi = DRUG["B"] * 0.8, DRUG["B"] * 1.2
for _ in range(60):
    mid = (lo + hi) / 2; c, q, _ = run("B", drug_cost=mid)
    if (c - res["U"][0]) / (q - res["U"][1]) < 5000: lo = mid
    else: hi = mid
print("threshold B price per 3 months at ICER 5000: %.1f (annual %.0f) vs base %.0f (annual %.0f)" % (lo, lo * 4, DRUG["B"], DRUG["B"] * 4))
# discount factor check
print("discount: QALY 1.0 received 5 years from now -> %.3f ; 10 years -> %.3f" % (1 / 1.045 ** 5, 1 / 1.045 ** 10))

# one-way sensitivity (tornado)
def icer(**kw):
    kB = {k: v for k, v in kw.items() if k not in ("respU",)}
    respU = kw.get("respU", RESP_U)
    orbu = kw.get("orbu", OR_BU)
    kB.pop("orbu", None)
    cB, qB, _ = run("B", resp=resp_from_or(respU, orbu), **kB)
    cU, qU, _ = run("U", resp=respU, **{k: v for k, v in kB.items() if k in ("u_r", "u_s", "horizon", "p_stop_c")})
    return (cB - cU) / (qB - qU)
tests = {
    "OR B vs U 1.30-2.50": [dict(orbu=1.30), dict(orbu=2.50)],
    "utility responder 0.85-0.93": [dict(u_r=0.85), dict(u_r=0.93)],
    "utility BSC 0.55-0.65": [dict(u_s=0.55), dict(u_s=0.65)],
    "horizon 10y-40y": [dict(horizon=10), dict(horizon=40)],
    "discontinuation 20%-40%/yr": [dict(p_stop_c=1 - math.exp(math.log(0.8) * CYC)), dict(p_stop_c=1 - math.exp(math.log(0.6) * CYC))],
    "U response 0.60-0.80": [dict(respU=0.60), dict(respU=0.80)],
}
for k, (a, b) in tests.items():
    print("  OWSA %-30s %.0f / %.0f" % (k, icer(**a), icer(**b)))

# PSA
rng = np.random.default_rng(95); N = 2000
def beta_ms(m, se):
    a = m * (m * (1 - m) / se ** 2 - 1); b = (1 - m) * (m * (1 - m) / se ** 2 - 1); return a, b
se_lor = (math.log(2.70) - math.log(1.20)) / (2 * 1.96)   # NMA OR 1.80 (95% CrI 1.20-2.70)
nb = []; dcs = []; dqs = []
for i in range(N):
    respU = rng.beta(*beta_ms(0.70, 0.03)); orbu = math.exp(rng.normal(math.log(OR_BU), se_lor))
    ur = rng.beta(*beta_ms(0.89, 0.02)); us = rng.beta(*beta_ms(0.60, 0.03))
    stop = rng.beta(*beta_ms(0.30, 0.04)); ps = 1 - math.exp(math.log(1 - stop) * CYC)
    cB, qB, _ = run("B", resp=resp_from_or(respU, orbu), u_r=ur, u_s=us, p_stop_c=ps)
    cU, qU, _ = run("U", resp=respU, u_r=ur, u_s=us, p_stop_c=ps)
    dcs.append(cB - cU); dqs.append(qB - qU)
dcs, dqs = np.array(dcs), np.array(dqs)
for thr in (3000, 5000, 7000):
    nbv = thr * dqs - dcs
    print("  PSA thr %d: P(CE) %.0f%% mean NB %.0f" % (thr, 100 * np.mean(nbv > 0), nbv.mean()))
print("  PSA mean dC %.0f mean dQ %.3f ratio of means %.0f | mean of ratios %.0f" % (dcs.mean(), dqs.mean(), dcs.mean() / dqs.mean(), np.mean(dcs / dqs)))
nbv = 5000 * dqs - dcs; print("  EVPI per patient at 5000: %.0f" % (np.mean(np.maximum(nbv, 0)) - max(nbv.mean(), 0)))

# Budget impact (undiscounted, annual, payer): eligible patients on biologics, B takes share from U
pts = 6000; share = 0.10
ann_B = 4 * (DRUG["B"] + C_MON); ann_U = 4 * (DRUG["U"] + C_MON)
print("BIA: annual per-patient on-drug cost B %.0f U %.0f diff %.0f | %d switched -> %.1f 억원 | wrong (lifetime dC x n) %.1f 억원" %
      (ann_B, ann_U, ann_B - ann_U, pts * share, pts * share * (ann_B - ann_U) / 10000, pts * share * dC / 10000))

# Budget impact, ch25 style: per-patient undiscounted cost in treatment year 1 (horizon 1 year, no discounting)
DISC_SAVE = DISC
DISC = 0.0
y1 = {a: run(a, horizon=1)[0] for a in ("B", "U")}
DISC = DISC_SAVE
print("BIA year-1 per patient (undiscounted): B %.1f U %.1f diff %.1f | 600 starters -> %.2f 억원 | price-only diff 4x20=80 -> %.1f 억원" %
      (y1["B"], y1["U"], y1["B"] - y1["U"], 600 * (y1["B"] - y1["U"]) / 10000, 600 * 80 / 10000))
