"""'머신러닝 기초' 과목의 공통 자료 · pub/data/ml_claims.csv

run:  source /home/claude/pylibs/env.sh && python3 gen/data_ml.py          (자료 + 참 구조 + 점검)
      python3 gen/data_ml.py --nocheck                                     (자료와 참 구조만, 점검 생략)

40세 이상 건강보험 가입자(가상) 한 사람이 한 줄. 기준 연도 2022년의 인구학적 특성, 진단, 의료이용, 처방,
건강검진 결과로 2023년의 응급 입원(admit_2023, 분류)과 총 의료비(cost_2023, 회귀)를 예측한다.

만드는 파일
  pub/data/ml_claims.csv   공개 자료 (학생 코드가 사이트 주소로 읽는다)
  gen/_ml_truth.csv        사람별 참 확률(p_true), 참 기대비용(cost_true)과 생성에 쓴 숨은 값 (공개하지 않음)
  gen/_ml_truth.json       참 구조(계수, 비선형, 교호작용, 효과 없는 변수)와 점검 결과 (공개하지 않음)

결정적이다. 난수는 SEED 하나에서 정해진 순서로 뽑고, CSV의 숫자는 정해진 자릿수의 글자로 적는다.
점검(check)은 0장의 분할(test_size=0.25, stratify=y, random_state=2026)에서 여러 모형의 시험 자료 성능을
구해 _ml_truth.json에 남긴다(ML_DATA.md 끝의 '작성자 메모'와 같은 숫자).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "pub", "data", "ml_claims.csv")
TRUTH_CSV = os.path.join(HERE, "_ml_truth.csv")
TRUTH_JSON = os.path.join(HERE, "_ml_truth.json")

SEED = 3
N = 15000
SPLIT = dict(test_size=0.25, random_state=2026)      # + stratify=y

COMORB = ["htn", "dm", "dyslip", "ckd", "hf", "af", "cad", "stroke", "copd", "asthma",
          "depress", "dementia", "cancer", "liver", "oa"]
DRUGS = ["anticoag", "insulin", "opioid", "benzo", "antipsych", "diuretic"]
UTIL = ["n_outpt", "n_ed", "n_admit", "los_days", "n_drugs", "n_prescribers", "cost_2022"]
SCREEN = ["bmi", "sbp", "fbg", "egfr", "smoking", "alcohol"]
DEMO = ["age", "female", "insurance", "income_q", "region"]
FEATURES = DEMO + COMORB + UTIL + DRUGS + SCREEN

# ---------------------------------------------------------------- 참 구조: 2023년 응급 입원의 로짓
# 선형 항(로그 오즈). 여기 없는 특성(dyslip, oa, region, n_prescribers)은 효과가 없다(계수 0).
B0 = -4.45
LIN = {
    "female": -0.10, "medaid": 0.25, "income_q_per_step": -0.03,
    "htn": 0.03, "dm": 0.06, "ckd": 0.12, "hf": 0.25, "af": 0.10, "cad": 0.10, "stroke": 0.10,
    "copd": 0.20, "asthma": 0.08, "depress": 0.08, "dementia": 0.15, "cancer": 0.30, "liver": 0.15,
    "n_admit_per_admission_max3": 0.06, "los_days_per_day_max60": 0.002, "log1p_n_outpt": 0.02,
    "n_drugs_per_drug": 0.01,
    "anticoag": 0.08, "opioid": 0.10, "benzo": 0.08, "antipsych": 0.10, "diuretic": 0.05,
    "sbp_per_mmHg_above_140": 0.004, "fbg_per_mgdl_above_126": 0.002, "egfr_per_unit_below_60": 0.01,
    "smoking_former": 0.05, "smoking_current": 0.15, "alcohol_heavy": 0.12, "no_screening": 0.35,
}
# 비선형 항과 교호작용의 크기 (simulate()가 이 값을 그대로 쓴다)
NL = dict(age_slope=0.005, age_knot=70, age_extra=0.085, age_cap=22,   # 나이: 70세 이후 기울기 0.005 -> 0.09 (92세까지)
          ed1=0.10, ed2=1.20,                                          # 응급실 1회 +0.10, 2회 이상 +1.20
          drugs10=0.80,                                                # 약 10가지 이상 +0.80
          bmi_lo_knot=23.0, bmi_lo=0.09, bmi_hi_knot=25.0, bmi_hi=0.03, bmi_cap=1.8,   # BMI U자
          hf_ckd=0.80,                                                 # 심부전 x 만성콩팥병
          ins_base=0.20, ins_age=0.06)                                 # 인슐린 x 나이
NONLIN = {
    "age": f"{NL['age_slope']}*(age-60) + {NL['age_extra']}*min(max(age-{NL['age_knot']}, 0), {NL['age_cap']})",
    "n_ed": f"+{NL['ed1']} if n_ed == 1, +{NL['ed2']} if n_ed >= 2",
    "n_drugs": f"{LIN['n_drugs_per_drug']}*n_drugs + {NL['drugs10']}*(n_drugs >= 10)",
    "bmi": (f"min({NL['bmi_lo']}*({NL['bmi_lo_knot']}-bmi)^2, {NL['bmi_cap']}) below {NL['bmi_lo_knot']}, "
            f"min({NL['bmi_hi']}*(bmi-{NL['bmi_hi_knot']})^2, {NL['bmi_cap']}) above {NL['bmi_hi_knot']}, 0 between (U shape)"),
    "hf_x_ckd": f"+{NL['hf_ckd']} when hf == 1 and ckd == 1",
    "insulin_x_age": f"insulin * ({NL['ins_base']} + {NL['ins_age']}*max(age-60, 0))",
}
# 2023년 의료비를 만드는 값 (simulate()가 그대로 쓴다)
COST = dict(use0=0.2, use_outpt=3.6, use_age=0.03, use_chronic=0.5,          # 의료 이용이 있을 확률(로짓)
            mu0=1.70, mu_lcost=0.62, mu_age=0.005, mu_dm=0.20, mu_ckd=0.30, mu_hf=0.25,
            mu_cancer=0.55, mu_dementia=0.20, mu_ndrugs=0.04, mu_copd=0.15, sigma=0.75,   # 로그 비용
            hosp0=400, hosp_age=6, hosp_cancer=260, hosp_hf=160, hosp_ckd=120, hosp_shape=1.6)  # 입원 비용(만원)
NULL_FEATURES = ["dyslip", "oa", "region", "n_prescribers"]


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(seed=SEED, n=N):
    rng = np.random.default_rng(seed)
    u = lambda: rng.random(n)                     # noqa: E731  균등난수 한 벌
    nrm = lambda s=1.0: rng.normal(0.0, s, n)     # noqa: E731

    # ---------- 인구학적 특성
    grp = rng.choice(5, size=n, p=[0.28, 0.29, 0.22, 0.13, 0.08])      # 40대 … 80세 이상
    age = np.empty(n, dtype=int)
    for g, lo in enumerate([40, 50, 60, 70]):
        m = grp == g
        age[m] = rng.integers(lo, lo + 10, m.sum())
    m = grp == 4
    age[m] = 80 + np.minimum(rng.geometric(0.16, m.sum()) - 1, 19)       # 80–99세, 나이가 많을수록 드묾
    a = (age - 60) / 10.0
    female = (u() < 0.48 + 0.0035 * (age - 40)).astype(int)
    male = 1 - female
    inc_latent = nrm() - 0.012 * (age - 40)
    cuts = np.quantile(inc_latent, [0.2, 0.4, 0.6, 0.8])
    income_q = 1 + np.searchsorted(cuts, inc_latent)                     # 소득 5분위(1 = 가장 낮음)
    medaid = ((income_q == 1) & (u() < logistic(-1.7 + 0.035 * (age - 60)))).astype(int)
    p_rural = 0.07 + 0.0045 * (age - 40)
    p_metro = 0.49 - 0.0025 * (age - 40)
    ur = u()
    region = np.where(ur < p_rural, "rural", np.where(ur < p_rural + p_metro, "metro", "city"))

    z = nrm()                                     # 숨은 건강 상태(관측되지 않음). 특성들 사이의 상관을 만든다

    # ---------- 생활습관(참값, 모두에게 있음. 검진을 받은 사람에게서만 관측)
    us = u()
    p_cur = np.where(male == 1, 0.36 - 0.004 * (age - 40), 0.05 - 0.0005 * (age - 40))
    p_for = np.where(male == 1, 0.22 + 0.006 * (age - 40), 0.03)
    smoking = np.where(us < p_cur, "current", np.where(us < p_cur + p_for, "former", "never"))
    ua = u()
    p_heavy = np.where(male == 1, 0.24 - 0.003 * (age - 40), 0.04 - 0.0006 * (age - 40)).clip(0.005)
    p_mod = np.where(male == 1, 0.46 - 0.004 * (age - 40), 0.34 - 0.005 * (age - 40)).clip(0.05)
    alcohol = np.where(ua < p_heavy, "heavy", np.where(ua < p_heavy + p_mod, "moderate", "none"))
    cur, former, heavy = (smoking == "current"), (smoking == "former"), (alcohol == "heavy")

    def flag(lp):
        return (u() < logistic(lp)).astype(int)

    # ---------- 동반질환 (2022년 진단)
    htn = flag(-1.0 + 0.80 * a + 0.30 * z - 0.15 * female + 0.20 * medaid)
    dm = flag(-2.0 + 0.45 * a + 0.35 * z + 0.60 * htn + 0.20 * medaid - 0.20 * female)
    dyslip = flag(-1.45 + 0.30 * a + 0.70 * dm + 0.55 * htn + 0.25 * female * (age >= 55) + 0.1 * z)
    ckd = flag(-4.7 + 0.75 * a + 0.90 * dm + 0.70 * htn + 0.40 * z)
    cad = flag(-4.1 + 0.60 * a + 0.50 * htn + 0.50 * dm + 0.40 * dyslip - 0.50 * female + 0.30 * z)
    af = flag(-5.1 + 0.90 * a + 0.40 * htn - 0.30 * female + 0.20 * z)
    hf = flag(-5.6 + 0.80 * a + 0.70 * htn + 0.60 * dm + 1.00 * cad + 1.20 * af + 0.50 * ckd + 0.40 * z)
    stroke = flag(-4.7 + 0.65 * a + 0.60 * htn + 0.40 * dm + 0.80 * af + 0.30 * z - 0.10 * female)
    copd = flag(-4.3 + 0.65 * a + 1.10 * cur + 0.70 * former + 0.30 * z - 0.30 * female + 0.30 * medaid)
    asthma = flag(-3.4 + 0.15 * a + 0.30 * z + 0.20 * female)
    depress = flag(-3.3 + 0.10 * a + 0.60 * female + 0.35 * z + 0.40 * medaid)
    dementia = flag(-6.4 + 0.18 * (age - 60) + 0.25 * z + 0.30 * female + 0.40 * stroke)
    cancer = flag(-3.9 + 0.45 * a + 0.20 * male + 0.20 * z)
    liver = flag(-3.5 + 0.10 * a + 0.60 * heavy + 0.30 * male + 0.20 * z)
    oa = flag(-2.5 + 0.75 * a + 0.75 * female + 0.10 * z)

    # ---------- 처방 (2022년, 있으면 1)
    anticoag = np.where(af == 1, u() < 0.55 + 0.01 * (age - 70).clip(0, 15), u() < 0.004).astype(int)
    insulin = (dm * (u() < logistic(-2.3 + 0.30 * z + 0.50 * ckd + 0.15 * a))).astype(int)
    opioid = flag(-3.5 + 0.20 * a + 0.90 * oa + 1.40 * cancer + 0.30 * z + 0.20 * female)
    benzo = flag(-3.1 + 0.30 * a + 1.50 * depress + 0.30 * female + 0.20 * z + 0.40 * dementia)
    antipsych = flag(-5.0 + 2.50 * dementia + 2.00 * depress + 0.20 * z)
    diuretic = flag(-3.9 + 1.30 * htn + 2.20 * hf + 0.60 * ckd + 0.30 * a + 0.30 * liver)

    chronic = htn + dm + dyslip + ckd + hf + af + cad + stroke + copd + asthma + depress + dementia + cancer + liver + oa
    serious = ckd + hf + af + cad + stroke + copd + dementia + cancer + liver

    # ---------- 의료이용 (2022년)
    mu_out = np.exp(2.40 + 0.22 * a + 0.20 * chronic + 0.30 * z + 0.15 * female + 0.15 * medaid)
    n_outpt = np.minimum(rng.poisson(mu_out * rng.gamma(1.6, 1 / 1.6, n)), 300)
    mu_ed = np.exp(-2.45 + 0.25 * a + 0.35 * z + 0.25 * serious + 0.60 * medaid + 0.30 * dementia + 0.30 * heavy)
    n_ed = np.minimum(rng.poisson(mu_ed * rng.gamma(0.45, 1 / 0.45, n)), 12)
    mu_ad = np.exp(-2.75 + 0.30 * a + 0.35 * z + 0.30 * serious + 0.45 * np.minimum(n_ed, 3) + 0.30 * medaid)
    n_admit = np.minimum(rng.poisson(mu_ad * rng.gamma(1.0, 1.0, n)), 10)
    los_days = np.zeros(n, dtype=int)
    for k in range(1, int(n_admit.max()) + 1):
        has = n_admit >= k
        scale = (5.5 + 0.18 * (age - 40)) / 1.3
        los_days += np.where(has, np.ceil(rng.gamma(1.3, scale)), 0).astype(int)
    los_days = np.minimum(los_days, 365)
    drugflags = anticoag + insulin + opioid + benzo + antipsych + diuretic
    mu_dr = 0.3 + 1.45 * chronic + 0.60 * drugflags + 0.02 * n_outpt + 0.50 * np.maximum(z, 0)
    dr_draw = rng.poisson(mu_dr)
    n_drugs = np.where(n_outpt > 0, dr_draw, 0)
    n_drugs = np.maximum(n_drugs, np.where(n_outpt > 0, drugflags, 0))
    pr_draw = rng.poisson(0.22 * np.sqrt(n_outpt) + 0.10 * n_drugs)
    # 2022년 진단이나 처방이 있는데 외래·응급실·입원이 모두 0회인 사람은 청구자료에서 생길 수 없으므로
    # 외래 1회로 둔다. 난수는 위에서 모두 뽑았으므로 다른 사람의 값과 이후 난수의 순서는 바뀌지 않는다.
    no_claim = ((chronic + drugflags) > 0) & ((n_outpt + n_ed + n_admit) == 0)
    n_outpt = np.where(no_claim, 1, n_outpt)
    n_drugs = np.maximum(np.where(n_outpt > 0, dr_draw, 0), np.where(n_outpt > 0, drugflags, 0))
    n_prescribers = np.where(n_outpt > 0, 1 + pr_draw, 0)
    # 2022년 총 의료비(만원): 외래 + 장기 처방약 + 응급실 + 입원 + 암 진료, 곱하기 꼴의 잡음
    c_out = n_outpt * 5.0 * np.exp(rng.normal(0, 0.35, n))
    c_drug = n_drugs * 20.0 * np.exp(rng.normal(0, 0.45, n))
    c_ed = n_ed * 25.0
    c_ip = los_days * 45.0 * np.exp(rng.normal(0, 0.30, n))
    c_ca = cancer * 600.0 * np.exp(rng.normal(0, 0.8, n))
    cost_2022 = (c_out + c_drug + c_ed + c_ip + c_ca) * np.exp(rng.normal(0, 0.25, n))
    cost_2022 = np.where((n_outpt + n_ed + n_admit) == 0, 0.0, cost_2022)

    # ---------- 건강검진 (받은 사람만 관측, 약 30% 결측)
    attend_lp = (0.85 - 0.06 * np.maximum(age - 70, 0) - 0.80 * medaid - 1.30 * dementia
                 - 0.45 * (n_admit > 0) - 0.30 * cancer + 0.08 * (income_q - 3) - 0.15 * (region == "rural")
                 + 0.25 * (n_outpt > 0) - 0.20 * z)
    attend = (u() < logistic(attend_lp)).astype(int)
    bmi = (23.6 + 0.8 * male + 1.2 * dm + 0.9 * htn + 0.5 * dyslip - 0.05 * np.maximum(age - 70, 0)
           - 1.3 * cancer - 1.6 * dementia - 1.0 * copd + nrm(3.1)).clip(14.5, 42.0)
    bmi = np.round(bmi, 1)
    sbp = np.round(116 + 0.30 * (age - 40) + 9 * htn + 0.25 * (bmi - 24) + nrm(13)).clip(82, 210)
    fbg = np.where(dm == 1, 136 + 12 * insulin + nrm(32), 92 + 0.12 * (age - 40) + 0.6 * (bmi - 24) + nrm(9))
    fbg = np.round(fbg).clip(62, 400)
    egfr = np.round(103 - 0.75 * (age - 40) - 28 * ckd - 4 * dm - 3 * htn - 6 * hf + nrm(12)).clip(5, 140)

    # ---------- 참 확률 (2023년 응급 입원)
    bmi_term = np.where(bmi < NL["bmi_lo_knot"], np.minimum(NL["bmi_lo"] * (NL["bmi_lo_knot"] - bmi) ** 2, NL["bmi_cap"]),
                        np.where(bmi > NL["bmi_hi_knot"],
                                 np.minimum(NL["bmi_hi"] * (bmi - NL["bmi_hi_knot"]) ** 2, NL["bmi_cap"]), 0.0))
    lp = (B0
          + NL["age_slope"] * (age - 60) + NL["age_extra"] * np.clip(age - NL["age_knot"], 0, NL["age_cap"])
          + LIN["female"] * female + LIN["medaid"] * medaid + LIN["income_q_per_step"] * (income_q - 3)
          + sum(LIN[c] * v for c, v in [("htn", htn), ("dm", dm), ("ckd", ckd), ("hf", hf), ("af", af),
                                         ("cad", cad), ("stroke", stroke), ("copd", copd), ("asthma", asthma),
                                         ("depress", depress), ("dementia", dementia), ("cancer", cancer),
                                         ("liver", liver)])
          + NL["hf_ckd"] * hf * ckd
          + NL["ed1"] * (n_ed == 1) + NL["ed2"] * (n_ed >= 2)
          + LIN["n_admit_per_admission_max3"] * np.minimum(n_admit, 3)
          + LIN["los_days_per_day_max60"] * np.minimum(los_days, 60)
          + LIN["log1p_n_outpt"] * np.log1p(n_outpt)
          + LIN["n_drugs_per_drug"] * n_drugs + NL["drugs10"] * (n_drugs >= 10)
          + sum(LIN[c] * v for c, v in [("anticoag", anticoag), ("opioid", opioid), ("benzo", benzo),
                                         ("antipsych", antipsych), ("diuretic", diuretic)])
          + insulin * (NL["ins_base"] + NL["ins_age"] * np.maximum(age - 60, 0))
          + bmi_term
          + LIN["sbp_per_mmHg_above_140"] * np.maximum(sbp - 140, 0)
          + LIN["fbg_per_mgdl_above_126"] * np.maximum(fbg - 126, 0)
          + LIN["egfr_per_unit_below_60"] * np.maximum(60 - egfr, 0)
          + LIN["smoking_former"] * former + LIN["smoking_current"] * cur + LIN["alcohol_heavy"] * heavy
          + LIN["no_screening"] * (1 - attend))
    p_true = logistic(lp)
    admit_2023 = (u() < p_true).astype(int)

    # ---------- 2023년 총 의료비 (만원): 외래·약제 등(로그 정규) + 응급 입원 비용(감마)
    C = COST
    p_use = logistic(C["use0"] + C["use_outpt"] * (n_outpt > 0) + C["use_age"] * (age - 60) + C["use_chronic"] * (chronic > 0))
    mu_c = (C["mu0"] + C["mu_lcost"] * np.log1p(cost_2022) + C["mu_age"] * (age - 60) + C["mu_dm"] * dm
            + C["mu_ckd"] * ckd + C["mu_hf"] * hf + C["mu_cancer"] * cancer + C["mu_dementia"] * dementia
            + C["mu_ndrugs"] * n_drugs + C["mu_copd"] * copd)
    any_use = (u() < p_use) | (admit_2023 == 1)                 # 응급 입원이 있으면 의료비가 있다
    c_amb = np.where(any_use, np.exp(mu_c + rng.normal(0, C["sigma"], n)), 0.0)
    hosp_mean = (C["hosp0"] + C["hosp_age"] * np.maximum(age - 50, 0) + C["hosp_cancer"] * cancer
                 + C["hosp_hf"] * hf + C["hosp_ckd"] * ckd)
    c_hosp = admit_2023 * rng.gamma(C["hosp_shape"], hosp_mean / C["hosp_shape"])
    cost_2023 = c_amb + c_hosp
    p_any = p_use + (1 - p_use) * p_true
    cost_true = p_any * np.exp(mu_c + C["sigma"] ** 2 / 2) + p_true * hosp_mean   # 참 기대비용(특성이 주어졌을 때)

    # ---------- 관측: 검진 결측(미수검 전체 + 문항 결측 약간)
    def obs(x, extra):
        x = np.asarray(x, dtype=object if np.asarray(x).dtype.kind in "UO" else float).copy()
        miss = (attend == 0) | (u() < extra)
        x[miss] = np.nan
        return x

    bmi_o, sbp_o = obs(bmi, 0.004), obs(sbp, 0.004)
    fbg_o, egfr_o = obs(fbg, 0.012), obs(egfr, 0.015)
    smk_o, alc_o = obs(smoking, 0.025), obs(alcohol, 0.03)

    pub = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "age": age, "female": female, "insurance": np.where(medaid == 1, "medaid", "nhi"),
        "income_q": income_q, "region": region,
        "htn": htn, "dm": dm, "dyslip": dyslip, "ckd": ckd, "hf": hf, "af": af, "cad": cad, "stroke": stroke,
        "copd": copd, "asthma": asthma, "depress": depress, "dementia": dementia, "cancer": cancer,
        "liver": liver, "oa": oa,
        "n_outpt": n_outpt, "n_ed": n_ed, "n_admit": n_admit, "los_days": los_days, "n_drugs": n_drugs,
        "n_prescribers": n_prescribers, "cost_2022": np.round(cost_2022, 1),
        "anticoag": anticoag, "insulin": insulin, "opioid": opioid, "benzo": benzo, "antipsych": antipsych,
        "diuretic": diuretic,
        "bmi": bmi_o, "sbp": sbp_o, "fbg": fbg_o, "egfr": egfr_o, "smoking": smk_o, "alcohol": alc_o,
        "admit_2023": admit_2023, "cost_2023": np.round(cost_2023, 1),
    })
    truth = pd.DataFrame({
        "id": pub["id"], "p_true": p_true, "lp_true": lp, "cost_true": cost_true,
        "attend": attend, "z_latent": z,
        "bmi_full": bmi, "sbp_full": sbp, "fbg_full": fbg, "egfr_full": egfr,
        "smoking_full": smoking, "alcohol_full": alcohol,
        "cost_amb": c_amb, "cost_hosp": c_hosp,
    })
    return pub, truth


# ---------------------------------------------------------------- 결정적인 CSV 쓰기
INT_COLS = ["id", "age", "female", "income_q"] + COMORB + ["n_outpt", "n_ed", "n_admit", "los_days", "n_drugs",
                                                            "n_prescribers"] + DRUGS + ["admit_2023"]


def to_csv_text(df):
    cols = {}
    for c in df.columns:
        v = df[c]
        if c in INT_COLS:
            cols[c] = v.astype(int).astype(str)
        elif c in ("sbp", "fbg", "egfr"):
            cols[c] = v.map(lambda x: "" if pd.isna(x) else str(int(x)))
        elif c in ("bmi", "cost_2022", "cost_2023"):
            cols[c] = v.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
        else:
            cols[c] = v.map(lambda x: "" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x))
    lines = [",".join(df.columns)]
    rows = zip(*[cols[c].tolist() for c in df.columns])
    lines += [",".join(r) for r in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 점검: 0장의 분할에서 모형별 시험 성능
def _models():
    """점검에 쓰는 전처리와 모형 (분류 5개, 회귀 4개)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                                  RandomForestClassifier, RandomForestRegressor)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    cat = ["insurance", "region", "smoking", "alcohol"]
    num = [c for c in FEATURES if c not in cat]
    lin_prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    tree_prep = ColumnTransformer([          # HistGradientBoosting은 결측을 그대로 받는다
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cat)])
    tree_prep_imp = ColumnTransformer([      # 결정트리와 랜덤 포레스트에는 중앙값·최빈값 대체
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oe", OrdinalEncoder())]), cat)])
    clf = {
        "logistic (median impute + scale + one-hot, C=1)":
            Pipeline([("p", lin_prep), ("m", LogisticRegression(max_iter=2000))]),
        "decision tree (max_depth=5, min_samples_leaf=50)":
            Pipeline([("p", tree_prep_imp), ("m", DecisionTreeClassifier(max_depth=5, min_samples_leaf=50,
                                                                          random_state=0))]),
        "random forest (500 trees, min_samples_leaf=20)":
            Pipeline([("p", tree_prep_imp), ("m", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=0))]),
        "HistGradientBoosting (defaults)":
            Pipeline([("p", tree_prep), ("m", HistGradientBoostingClassifier(random_state=0))]),
        "HistGradientBoosting (learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=50, l2=1, max_iter=500)":
            Pipeline([("p", tree_prep), ("m", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=500, max_leaf_nodes=8, min_samples_leaf=50,
                l2_regularization=1.0, random_state=0))]),
    }
    reg = {
        "linear regression (raw cost)": Pipeline([("p", lin_prep), ("m", LinearRegression())]),
        "decision tree (max_depth=6, min_samples_leaf=50)":
            Pipeline([("p", tree_prep_imp), ("m", DecisionTreeRegressor(max_depth=6, min_samples_leaf=50,
                                                                         random_state=0))]),
        "random forest (500 trees, min_samples_leaf=20)":
            Pipeline([("p", tree_prep_imp), ("m", RandomForestRegressor(
                n_estimators=500, min_samples_leaf=20, max_features=0.33, n_jobs=-1, random_state=0))]),
        "HistGradientBoosting (defaults)":
            Pipeline([("p", tree_prep), ("m", HistGradientBoostingRegressor(random_state=0))]),
    }
    return clf, reg


def _reg_scores(y, pr):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"rmse": round(float(np.sqrt(mean_squared_error(y, pr))), 1),
            "mae": round(float(mean_absolute_error(y, pr)), 1),
            "r2": round(float(r2_score(y, pr)), 4)}


def check(pub, truth):
    """0장의 분할에서 모형별 시험 자료 성능 (분류 AUC, 회귀 RMSE·MAE·R²)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X = pub[FEATURES]
    y = pub["admit_2023"]
    X_tr, X_te, y_tr, y_te, c_tr, c_te, t_tr, t_te = train_test_split(
        X, y, pub["cost_2023"], truth, stratify=y, **SPLIT)
    res = {"n_train": int(len(y_tr)), "n_test": int(len(y_te)), "events_train": int(y_tr.sum()),
           "events_test": int(y_te.sum())}
    clf, reg = _models()
    auc = {}
    for name, m in clf.items():
        m.fit(X_tr, y_tr)
        auc[name] = round(float(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])), 4)
    auc["true probability (ceiling)"] = round(float(roc_auc_score(y_te, t_te["p_true"])), 4)
    auc["true probability, whole file"] = round(float(roc_auc_score(y, truth["p_true"])), 4)
    res["classification_test_auc"] = auc
    regres = {}
    for name, m in reg.items():
        m.fit(X_tr, c_tr)
        regres[name] = _reg_scores(c_te, m.predict(X_te))
    regres["true expected cost (ceiling)"] = _reg_scores(c_te, t_te["cost_true"].to_numpy())
    res["regression_test"] = regres
    return res


def population_reference(n_big=100000, reps=3):
    """같은 생성식으로 만든 큰 모의 집단(공개 자료와 다른 seed)에서 11,250명으로 학습해 50,000명으로 평가한
    값의 평균. 공개 자료 한 번의 분할에서 나온 숫자가 우연히 크거나 작은지 판단하는 기준이다."""
    from sklearn.metrics import roc_auc_score
    big, tb = simulate(seed=777, n=n_big)
    X, y, c = big[FEATURES], big["admit_2023"], big["cost_2023"]
    te = np.arange(n_big // 2, n_big)
    clf, reg = _models()
    out = {"auc": {}, "r2": {}}
    for r in range(reps):
        tr = np.arange(r * 11250, (r + 1) * 11250)
        for name, m in clf.items():
            m.fit(X.iloc[tr], y.iloc[tr])
            out["auc"].setdefault(name, []).append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
        for name, m in reg.items():
            m.fit(X.iloc[tr], c.iloc[tr])
            out["r2"].setdefault(name, []).append(_reg_scores(c.iloc[te], m.predict(X.iloc[te]))["r2"])
    out = {k: {nm: round(float(np.mean(v)), 4) for nm, v in d.items()} for k, d in out.items()}
    out["auc"]["true probability (ceiling)"] = round(float(roc_auc_score(y.iloc[te], tb["p_true"].iloc[te])), 4)
    out["r2"]["true expected cost (ceiling)"] = _reg_scores(c.iloc[te], tb["cost_true"].iloc[te])["r2"]
    out["note"] = f"seed 777, n={n_big}; train 11,250 x {reps} disjoint draws, test = last {n_big // 2}"
    return out


def describe(pub, truth):
    d = {}
    d["n"] = int(len(pub))
    d["admit_2023_rate"] = round(float(pub["admit_2023"].mean()), 4)
    d["admit_2023_events"] = int(pub["admit_2023"].sum())
    c = pub["cost_2023"]
    d["cost_2023"] = {"mean": round(float(c.mean()), 1), "median": round(float(c.median()), 1),
                      "p90": round(float(c.quantile(0.9)), 1), "p99": round(float(c.quantile(0.99)), 1),
                      "max": round(float(c.max()), 1), "zero_share": round(float((c == 0).mean()), 4)}
    c2 = pub["cost_2022"]
    d["cost_2022"] = {"mean": round(float(c2.mean()), 1), "median": round(float(c2.median()), 1),
                      "zero_share": round(float((c2 == 0).mean()), 4)}
    d["missing_share"] = {k: round(float(v), 4) for k, v in pub.isna().mean().items() if v > 0}
    d["screening_attended"] = round(float(truth["attend"].mean()), 4)
    d["age_mean"] = round(float(pub["age"].mean()), 1)
    d["prevalence"] = {k: round(float(pub[k].mean()), 4) for k in COMORB + DRUGS + ["female"]}
    d["p_true_mean"] = round(float(truth["p_true"].mean()), 4)
    ag = pd.cut(pub["age"], [39, 49, 59, 69, 74, 79, 84, 120],
                labels=["40-49", "50-59", "60-69", "70-74", "75-79", "80-84", "85+"])
    by = pub.assign(ag=ag, bmi_miss=pub["bmi"].isna()).groupby("ag", observed=True)
    d["by_age"] = {str(k): {"n": int(len(g)), "admit": round(float(g["admit_2023"].mean()), 4),
                            "cost_2023_mean": round(float(g["cost_2023"].mean()), 1),
                            "screening_missing": round(float(g["bmi_miss"].mean()), 3),
                            "n_drugs_ge10": round(float((g["n_drugs"] >= 10).mean()), 3)} for k, g in by}
    rate = lambda m: round(float(pub.loc[m, "admit_2023"].mean()), 4)    # noqa: E731
    bf = truth["bmi_full"]
    d["admit_rate_by_group"] = {
        "n_ed 0 / 1 / 2+": [rate(pub["n_ed"] == 0), rate(pub["n_ed"] == 1), rate(pub["n_ed"] >= 2)],
        "n_drugs <10 / >=10": [rate(pub["n_drugs"] < 10), rate(pub["n_drugs"] >= 10)],
        "hf0ckd0 / hf0ckd1 / hf1ckd0 / hf1ckd1": [rate((pub["hf"] == h) & (pub["ckd"] == k)) for h in (0, 1) for k in (0, 1)],
        "bmi(full) <18.5 / 18.5-23 / 23-25 / 25-30 / 30+": [rate(bf < 18.5), rate((bf >= 18.5) & (bf < 23)),
                                                            rate((bf >= 23) & (bf < 25)), rate((bf >= 25) & (bf < 30)),
                                                            rate(bf >= 30)],
        "screening observed / missing": [rate(pub["bmi"].notna()), rate(pub["bmi"].isna())],
        "null features, marginal (0 / 1): dyslip": [rate(pub["dyslip"] == 0), rate(pub["dyslip"] == 1)],
        "null features, marginal (0 / 1): oa": [rate(pub["oa"] == 0), rate(pub["oa"] == 1)],
        "null features, marginal: region metro / city / rural": [rate(pub["region"] == r) for r in ("metro", "city", "rural")],
        "null features, marginal: n_prescribers <=2 / 3-4 / 5+": [rate(pub["n_prescribers"] <= 2),
                                                                  rate(pub["n_prescribers"].between(3, 4)),
                                                                  rate(pub["n_prescribers"] >= 5)],
    }
    d["shares"] = {"n_ed>=1": round(float((pub["n_ed"] >= 1).mean()), 4),
                   "n_ed>=2": round(float((pub["n_ed"] >= 2).mean()), 4),
                   "n_admit>=1": round(float((pub["n_admit"] >= 1).mean()), 4),
                   "n_drugs>=5": round(float((pub["n_drugs"] >= 5).mean()), 4),
                   "n_drugs>=10": round(float((pub["n_drugs"] >= 10).mean()), 4),
                   "n_outpt==0": round(float((pub["n_outpt"] == 0).mean()), 4)}
    return d


if __name__ == "__main__":
    import hashlib
    pub, truth = simulate()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    text = to_csv_text(pub)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    csv_md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
    truth.round(6).to_csv(TRUTH_CSV, index=False)
    back = pd.read_csv(OUT)
    assert back.shape == pub.shape, back.shape
    info = {
        "seed": SEED, "n": N, "csv_md5": csv_md5, "split": dict(SPLIT, stratify="admit_2023"),
        "admit_logit": {"intercept": B0, "linear": LIN, "nonlinear_and_interactions": NONLIN,
                        "no_effect_features": NULL_FEATURES,
                        "unobserved": "z_latent (건강 상태) does NOT enter the outcome directly; it only makes "
                                      "features correlated. Screening values enter through their full (true) "
                                      "values, which are missing in the public file for non-attenders."},
        "cost_2023": {"params": COST,
                      "any_use": "logistic(use0 + use_outpt*(n_outpt>0) + use_age*(age-60) + use_chronic*(any chronic dx)); "
                                 "anyone with admit_2023 = 1 has use",
                      "log_cost_if_use": "mu0 + mu_lcost*log1p(cost_2022) + mu_age*(age-60) + mu_dm*dm + mu_ckd*ckd "
                                         "+ mu_hf*hf + mu_cancer*cancer + mu_dementia*dementia + mu_ndrugs*n_drugs "
                                         "+ mu_copd*copd, normal error SD sigma",
                      "admission_cost": "admit_2023 * Gamma(shape hosp_shape, mean hosp0 + hosp_age*max(age-50,0) "
                                        "+ hosp_cancer*cancer + hosp_hf*hf + hosp_ckd*ckd)",
                      "cost_true": "P(any use)*exp(mu + sigma^2/2) + p_true*admission mean, "
                                   "P(any use) = p_use + (1-p_use)*p_true"},
        "describe": describe(pub, truth),
    }
    if "--nocheck" not in sys.argv:
        info["check"] = check(pub, truth)
        info["population_reference"] = population_reference()
    elif os.path.exists(TRUTH_JSON):
        # 점검을 건너뛸 때는, 같은 CSV(md5 일치)에 대해 이전에 구한 점검 결과를 그대로 남긴다
        old = json.load(open(TRUTH_JSON, encoding="utf-8"))
        if old.get("csv_md5") == csv_md5:
            for k in ("check", "population_reference"):
                if k in old:
                    info[k] = old[k]
    with open(TRUTH_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(OUT, f"{os.path.getsize(OUT) / 1024:.0f} KB", back.shape)
    print(json.dumps(info["describe"], ensure_ascii=False, indent=1))
    if "check" in info:
        print(json.dumps(info["check"], ensure_ascii=False, indent=1))
        print(json.dumps(info["population_reference"], ensure_ascii=False, indent=1))
