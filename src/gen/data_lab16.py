"""실습 16 자료 · pub/data/ami_cohort.csv (가상의 급성 심근경색 퇴원 환자 6,000명, 한 사람 한 줄).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab16.py

16장의 예제 코호트(gen/nums_ch16.py 의 sim_it(seed=2), 약물 X는 사망에 효과 없음)를 그대로 내보낸다.
nums_ch16.py 를 import 하면 스크립트 전체가 다시 돌고 _ch16_nums.json 을 덮어쓰므로, 생성 코드 10줄을
여기에 그대로 옮겨 두고(결정적, seed 고정) nums_ch16.py 가 저장한 gen/_ch16_cohort.csv 와 같은지 확인한다.

열 (모두 퇴원일 = 0일 기준, 1년 = 365일)
  id       환자 번호 (0-5999, 16장의 출력 상자와 같은 번호)
  fu_days  퇴원일부터 사망 또는 365일까지의 일수 (소수 6자리)
  died     추적 중 사망 = 1, 365일까지 생존 = 0
  rx_day   퇴원일부터 약물 X 첫 처방일까지의 일수. 추적 중 처방이 없으면 빈칸
본문의 모형은 공변량을 보정하지 않으므로(교란이 없는 자료) 공변량 열은 없다.
소수 6자리로 반올림해도 사망 시점과 처방 시점의 순서가 바뀌지 않고 동률이 생기지 않는다(아래 assert).
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DAYS = 365.0
LAM, KSH = 1 / 0.16, 0.8      # Weibull scale (years) and shape, as in nums_ch16.py


def sim_it(seed=2, n=6000):
    """copy of nums_ch16.sim_it with theta = 1.0 (drug X has no effect)"""
    rng = np.random.default_rng(seed)
    t0 = rng.weibull(KSH, n) * LAM                 # death time (years)
    tinit = rng.exponential(1 / 0.7, n)            # first prescription of drug X (years)
    end = np.minimum(t0, 1.0)
    died = (t0 <= 1.0).astype(int)
    user = (tinit < end).astype(int)
    return pd.DataFrame({"id": np.arange(n), "end": end, "died": died,
                         "tinit": np.where(user == 1, tinit, np.nan), "user": user})


dd = sim_it(2)
ref = os.path.join(HERE, "_ch16_cohort.csv")
if os.path.exists(ref):
    r = pd.read_csv(ref)
    assert (r.id.values == dd.id.values).all() and (r.died.values == dd.died.values).all()
    assert np.allclose(r.end.values, dd.end.values, rtol=0, atol=1e-12)
    assert np.allclose(r.tinit.values, dd.tinit.values, rtol=0, atol=1e-12, equal_nan=True)

out = pd.DataFrame({"id": dd.id,
                    "fu_days": (dd.end * DAYS).round(6),
                    "died": dd.died,
                    "rx_day": (dd.tinit * DAYS).round(6)})

# rounding must not create ties or change the order of events and prescriptions
ev = np.concatenate([out.fu_days[out.died == 1].values, out.rx_day.dropna().values])
assert len(np.unique(ev)) == len(ev)
assert ((out.rx_day < out.fu_days) | out.rx_day.isna()).all()
assert (np.argsort(ev, kind="stable") ==
        np.argsort(np.concatenate([(dd.end * DAYS)[dd.died == 1].values, (dd.tinit * DAYS).dropna().values]),
                   kind="stable")).all()
assert len(out) == 6000 and int(out.died.sum()) == 1234 and int(out.rx_day.notna().sum()) == 2734

path = os.path.join(ROOT, "pub", "data", "ami_cohort.csv")
os.makedirs(os.path.dirname(path), exist_ok=True)
out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
print("wrote", path, "rows", len(out), "bytes", os.path.getsize(path))
print(out.head())
