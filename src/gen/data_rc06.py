"""R과 통계분석 6장 자료: pub/data/rcc_switch.csv (대조군의 처치 전환을 허용한 가상 임상시험, 신약 A 대 표준요법 B, 군마다 300명).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_rc06.py      (정답 값은 gen/_rc06_truth.json)

만드는 방식 (모든 환자는 진행한 뒤 사망한다고 단순화)
  - 치료받지 않았을 때(표준요법만)의 시간 U = 진행까지 P0 + 진행 뒤 S0
      P0 ~ 와이블(모양 0.95, 중앙 10개월)
      진행 때의 수행능력 저하 ps_prog ~ 베르누이(0.25 + 0.20 × ecog)
      S0 ~ 와이블(모양 1.1, 중앙 14개월), 위험률에 ecog 1이면 ×1.4, ps_prog 1이면 ×2.6, 나이 10세당 ×1.1
  - 신약 A는 받는 동안의 시간을 AF = exp(0.3) = 1.35배로 늘린다(RPSFTM의 ψ = −0.3, 모든 환자에 같은 효과)
      A군: 진행 1.35 × P0, 사망 1.35 × U
      B군: 진행 P0에 전환 여부를 정하고, 전환하면 진행 뒤 시간이 1.35 × S0
  - 전환 확률: ps_prog 0이면 0.80, 1이면 0.30 (진행 때 상태가 좋은 환자가 더 많이 전환 → 전환은 예후와 관련)
  - 등록 0–12개월 고르게, 자료 마감 42개월 → 잠재 중도절단 시간 cens_time = 42 − 등록 시점(30–42개월)
열: pid, arm(1 = 신약 A, 0 = 표준요법 B), age, ecog, pfs, pfs_event, ps_prog(진행 전 마감이면 빈칸), switch(0/1, A군은 0),
    switch_time(전환 시점, 전환하지 않았으면 빈칸), os, os_event, cens_time
정답(전환이 없었을 때): ψ = −0.3, 같은 시험 기간(마감 방식 같음)의 전체생존 위험비와 28·36개월 제한평균생존시간 차이를 큰 가상 인구로 구한다.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "pub", "data", "rcc_switch.csv")
TRUTH = os.path.join(HERE, "_rc06_truth.json")
AF = float(np.exp(0.3))


def weib(rng, n, shape, median, hr=1.0):
    lam = np.log(2) / median ** shape * hr
    return (-np.log(rng.random(n)) / lam) ** (1 / shape)


def simulate(rng, n_per_arm, switching=True):
    n = 2 * n_per_arm
    arm = np.repeat([1, 0], n_per_arm)
    age = np.round(rng.normal(62, 9, n)).clip(35, 85)
    ecog = (rng.random(n) < 0.45).astype(int)
    p0 = weib(rng, n, 0.95, 10.0)
    ps_prog = (rng.random(n) < 0.25 + 0.20 * ecog).astype(int)
    hr_s = 1.4 ** ecog * 2.6 ** ps_prog * 1.1 ** ((age - 62) / 10)
    s0 = weib(rng, n, 1.1, 14.0, hr_s)
    cens = 42.0 - rng.uniform(0, 12, n)
    sw_draw = rng.random(n) < np.where(ps_prog == 1, 0.30, 0.80)
    prog = np.where(arm == 1, AF * p0, p0)
    progressed = prog < cens
    switch = (arm == 0) & progressed & sw_draw & switching
    post = np.where((arm == 1) | switch, AF * s0, s0)
    death = prog + post
    os_t = np.minimum(death, cens)
    os_e = (death <= cens).astype(int)
    pfs_t = np.minimum(prog, cens)
    pfs_e = progressed.astype(int)
    return pd.DataFrame({"arm": arm, "age": age.astype(int), "ecog": ecog,
                         "pfs": pfs_t, "pfs_event": pfs_e,
                         "ps_prog": np.where(progressed, ps_prog, np.nan),
                         "switch": switch.astype(int), "switch_time": np.where(switch, prog, np.nan),
                         "os": os_t, "os_event": os_e, "cens_time": cens})


def truth(rng):
    """전환이 없었다면: 큰 인구에서 Cox 위험비(같은 마감 방식)와 36개월 RMST 차이"""
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time as rmst
    big = simulate(rng, 100000, switching=False)
    hr = float(np.exp(CoxPHFitter().fit(big[["os", "os_event", "arm"]], "os", "os_event").params_["arm"]))
    out = {"psi": -0.3, "af": AF, "hr_no_switch": round(hr, 4)}
    for tau in (28, 36):
        r = {}
        for a in (1, 0):
            s = big[big["arm"] == a]
            r[a] = float(rmst(KaplanMeierFitter().fit(s["os"], s["os_event"]), t=tau))
        out.update({f"rmst{tau}_A": round(r[1], 3), f"rmst{tau}_B": round(r[0], 3), f"rmst{tau}_diff": round(r[1] - r[0], 3)})
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    d = simulate(rng, 300)
    d.insert(0, "pid", [f"S{i:03d}" for i in range(1, len(d) + 1)])
    for c in ("pfs", "os", "switch_time", "cens_time"):
        d[c] = d[c].round(3)
    assert (d.loc[d["arm"] == 1, "switch"] == 0).all()
    assert (d["os"] >= d["pfs"]).all()
    assert d.loc[d["switch"] == 1, "switch_time"].notna().all()
    d.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    print(OUT, d.shape)
    b = d[d["arm"] == 0]
    print("B: progressed", int(b["pfs_event"].sum()), " switched", int(b["switch"].sum()),
          f"({b['switch'].sum() / b['pfs_event'].sum():.0%} of progressed)")
    print("OS events A/B:", int(d.loc[d.arm == 1, "os_event"].sum()), int(b["os_event"].sum()))
    t = truth(np.random.default_rng(7))
    json.dump(t, open(TRUTH, "w"), indent=1)
    print(t)
