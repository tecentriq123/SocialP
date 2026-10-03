"""실습 14 자료 · pub/data/claims_person.csv, claims_visit.csv, claims_rx.csv, sglt2_cohort.csv, sglt2_cohort_full.csv

run:  source /home/claude/pylibs/env.sh && python3 gen/data_lab14.py

14장 예제 코호트(gen/nums_ch14.py, seed 20261002)를 청구자료 모양의 원자료 표 세 개로 거꾸로 풀어 쓴다.

왜 표본인가
    본문의 출발 인구는 300,000명, 분석 코호트는 166,548명이고 처방은 약 270만 건이다. 한 파일 3MB 제한에서는
    한 사람 한 줄짜리 코호트 파일조차 다 들어가지 않는다(약 4.3MB). 그래서 30분의 1 표본(출발 10,000명, 분석
    코호트 5,552명)을 만든다. 표본에 든 코호트 5,552명은 nums_ch14.py가 만든 사람을 그대로 옮긴 것이다(진입일,
    군, 공변량, 처방 간격, 변경일, 심부전 입원일, 사망일이 같다. 나이만 정수로 반올림).

표본을 고르는 방법 (select_sample)
    1) 군 × as-treated(유예 60일) 종료 사유 × 사건이 잡히는 추적 규칙(유예 30일, 60일, 90일, ITT)별로 본문 인원의 1/30을
       층화 추출한다. 사건 수(본문 사건 수의 1/30을 반올림)는 층의 크기로 정확히 정해진다.
    2) 같은 층 안에서 사람을 맞바꾸는 국소 탐색으로 표본의 요약값을 목표에 맞춘다. 목표는 인년(사건 수가 정수라서
       발생률이 본문과 같아지는 인년으로 잡는다), 기저 특성의 인원과 나이 평균·SD, Cox 점수 방정식의 합(전체
       코호트의 조·보정 모형에서 구한 사람별 기대 사건 수를 쓴다)이다.
    3) 표본에서 Cox 모형을 다시 적합해 본문의 위험비와 견주고, 어긋난 만큼 점수 방정식의 목표를 옮겨 맞바꾸기를
       이어 간다(위험비 되먹임). seed 1–20 가운데 발생률 8개, 발생률비 4개, 조·보정 위험비 8개(유예 60일, 30일,
       90일, ITT)가 모두 본문과 소수 둘째 자리까지 같은 것을 SEED_SELECT로 정했다. 사건 수가 1/30이므로 신뢰구간은
       본문보다 넓다(이것은 맞출 수 없다. 전체 코호트 파일로 확인한다).
    난수는 모두 고정 seed라서 같은 환경에서 다시 돌리면 같은 파일이 나온다(두 번 돌려 md5로 확인). 국소 탐색은
    부동소수점 비교에 기대므로 lifelines·numpy 버전이 바뀌면 다른 표본이 뽑힐 수 있다. 그때는 gen/lab_lab14.py의
    대조 블록이 실패하므로 SEED_SELECT를 다시 찾는다.

유예기간 규칙 (2026-10-04)
    nums_ch14.disc_day()는 처방 101건 안에 중단이 없는 사람을 첫 처방에서 중단한 것으로 잘못 처리하고 있었다(유예
    60일에서 4,114명). 그 버그를 고친 뒤의 nums_ch14.py를 쓴다. 이제 dc30 <= dc60 <= dc90이 모든 사람에서 성립하고,
    한 벌의 처방 기록에서 세 유예기간의 중단일이 모두 재현되므로 유예 90일의 결과도 본문에 맞춘다.

전체 코호트 파일 (sglt2_cohort_full.csv)
    166,548명 전원을 한 사람 한 줄로 쓴다. 5MB 제한 때문에 분석에 쓰는 열만 둔다: sglt2, age(정수), female, htn,
    ihd, hf, ckd, met, ins, time, event, time_itt, event_itt. 표본 파일에 있는 pid, arm, index_date, end_date,
    reason은 뺐다(arm은 sglt2에서 만들 수 있다). 그래도 5.3MB여서, as-treated 추적이 사건·사망·자료 종료로 끝난
    사람(42.5%. 이들은 ITT 추적이 as-treated 추적과 같다)의 time_itt, event_itt는 빈칸으로 둔다(4.99MB). 읽은 뒤
    fillna로 time, event의 값을 채우면 된다. 이 파일로 발생률과 Cox 모형을 돌리면 본문 표 14-5가 신뢰구간까지
    나온다(나이를 정수로 반올림해도 보정 위험비와 신뢰구간이 소수 둘째 자리까지 같다. __main__에서 확인).

제외되는 사람(4,448명)
    nums_ch14.py의 선정 흐름은 30만 명의 표지(기존 사용자, 동시 시작, 18세 미만, 말기신부전, 심부전 입원력)만
    만들고 그 사람들의 기록은 만들지 않는다. 여기서는 단계 × 계열별로 본문 인원의 1/30(반올림)만큼 표지를
    그대로 뽑아 와서, 그 표지에 맞는 처방·상병 기록을 새로 만든다.

nums_ch14.py와 이론 장은 고치지 않는다. nums_ch14.py의 코호트 생성 부분만 exec로 불러 쓴다(_ch14_nums.json을
다시 쓰는 뒷부분은 실행하지 않는다).
"""
import contextlib
import io
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PUBDATA = os.path.join(ROOT, "pub", "data")

FRAC = 30                       # 30분의 1 표본
SEED_SELECT = 11                # 표본 선택의 seed. 1–20을 돌려 발생률 8개, 발생률비 4개, 조·보정 위험비 8개가 모두 본문과
                                # 소수 둘째 자리까지 같은 것을 골랐다(유예 60일, 30일, 90일, ITT)
SEED_TABLES = 20261004          # 원자료 표를 꾸미는 난수(방문일, 관련 없는 진료 등)
DAY0 = pd.Timestamp("2016-01-01")          # nums_ch14.py의 day 0
DATA_START = -365                           # 2015-01-01
END = 2556                                  # 2022-12-31
COVS = ["age", "female", "htn", "ihd", "hf", "ckd", "met", "ins"]
REASONS = ["hhf", "death", "switch", "disc", "admin"]      # nums_ch14.follow()의 동률 우선순위
# 본문 값이 반올림 경계에 바짝 붙은 곳(유예 30일 발생률비 0.4149, 유예 60일 DPP-4 억제제군 12.9545, ITT 12.1845,
# 유예 90일 SGLT2 억제제군 5.4952)에서 표본의 값도 같은 쪽으로 반올림되도록 인년 목표를 조금(0.04% 안쪽) 옮긴다.
PY_NUDGE = {("at30", "D"): -1.5, ("at60", "D"): 1.5, ("itt", "D"): 1.5, ("at90", "S"): -0.8}
N_EXTRA = 150                   # 등록 기간(2016–2021) 밖에서만 처방받은 사람(출발 인구가 아님)


# ====================================================================================== 1. 본문 코호트 다시 만들기
def load_ch14():
    """nums_ch14.py의 앞부분(선정 흐름 표지와 코호트 시뮬레이션)만 실행하고, fills()가 돌려준 처방 간격을 붙잡는다."""
    path = os.path.join(HERE, "nums_ch14.py")
    src = open(path, encoding="utf-8").read()
    a = src.index("NN = 60_000\n")
    b = src.index("# ---------------------------------------------------------------- baseline table")
    ns = {"__name__": "nums_ch14_for_lab14", "__file__": path}
    cap = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src[:a], path, "exec"), ns)
        orig = ns["fills"]

        def fills(n, kind):
            out = orig(n, kind)          # 난수 흐름은 그대로
            cap[kind] = out              # (supply, start, runout, g, L)
            return out

        ns["fills"] = fills
        exec(compile(src[a:b], path, "exec"), ns)
    return ns, cap


def follow(d, mode, G=60):
    """nums_ch14.follow()와 같은 규칙. time(일), event, reason"""
    cand = {"hhf": d["hhf_day"].values, "death": d["death_day"].values, "admin": d["admin"].values}
    if mode == "at":
        cand["disc"] = d[f"dc{G}"].values
        cand["switch"] = d["switch"].values
    names = [k for k in REASONS if k in cand]
    M = np.column_stack([cand[k] for k in names])
    j = M.argmin(axis=1)
    t = M[np.arange(len(d)), j]
    reason = np.array(names)[j]
    return t.astype(np.int64), (reason == "hhf").astype(int), reason


ANALYSES = (("at60", "at", 60), ("at30", "at", 30), ("at90", "at", 90), ("itt", "itt", None))


def add_followup(co):
    for key, mode, G in ANALYSES:
        t, e, r = follow(co, mode, G or 60)
        co[f"t_{key}"], co[f"e_{key}"], co[f"r_{key}"] = t, e, r
    co["age_int"] = np.round(co["age"].values).astype(int)
    return co


def check_against_json(co, J):
    """다시 만든 코호트가 본문 숫자(_ch14_nums.json)와 같은지 확인"""
    assert len(co) == J["flow"]["n"] and int(co.sglt2.sum()) == J["flow"]["nS"]
    for key, _, _ in ANALYSES:
        for lab, k in (("S", 1), ("D", 0)):
            m = co.sglt2.values == k
            ev, py = int(co.loc[m, f"e_{key}"].sum()), co.loc[m, f"t_{key}"].sum() / 365.25
            assert ev == J["fu"][key][lab]["ev"], (key, lab, ev)
            assert abs(py - J["fu"][key][lab]["py"]) < 1e-6, (key, lab, py)
    for v in COVS[1:]:
        assert int(co.loc[co.sglt2 == 1, v].sum()) == J["base"][v]["S"][0]


# ====================================================================================== 2. 표본 고르기
def expected_events(co, key, adjusted=True):
    """전체 코호트의 Cox 모형(나이는 정수)에서 사람별 기대 사건 수 E_i = H0(t_i) exp(x_i b).
    군별로 sum(E_i) = 사건 수, 공변량별로 sum(x_i (d_i - E_i)) = 0 이 Cox의 점수 방정식이다.
    표본에서도 이 합들이 0에 가까우면 표본의 위험비가 전체 코호트의 위험비와 가까워진다."""
    from lifelines import CoxPHFitter
    df = co[["sglt2"] + (["female", "htn", "ihd", "hf", "ckd", "met", "ins"] if adjusted else [])].copy()
    if adjusted:
        df["age"] = co["age_int"].values
    df["t"], df["e"] = co[f"t_{key}"].values, co[f"e_{key}"].values
    m = CoxPHFitter().fit(df, "t", "e")
    H0 = m.baseline_cumulative_hazard_.iloc[:, 0]
    h = np.interp(df["t"].values, H0.index.values.astype(float), H0.values)
    ph = m.predict_partial_hazard(df).values
    return h * ph, float(np.exp(m.params_["sglt2"]))


def pct_count(p, n):
    """n명 가운데 비율 p에 가장 가까운 정수. 소수 첫째 자리 %가 본문과 같아지는 정수가 있으면 그것을 고른다."""
    c0 = int(round(p * n))
    want = round(100 * p, 1)
    for c in sorted(range(c0 - 2, c0 + 3), key=lambda c: abs(c / n - p)):
        if round(100 * c / n, 1) == want:
            return c
    return c0


def largest_remainder(counts, total):
    q = np.asarray(counts, float) / FRAC
    base = np.floor(q).astype(int)
    for i in np.argsort(-(q - base))[: total - base.sum()]:
        base[i] += 1
    return base


def event_level(co):
    """사건이 어느 추적 규칙부터 잡히는가: 0 = 유예 30일, 1 = 60일부터, 2 = 90일부터, 3 = ITT에서만, 4 = 사건 없음.
    dc30 <= dc60 <= dc90 이므로 사건 집합은 포함 관계다."""
    e30, e60, e90, eitt = (co[f"e_{k}"].values for k in ("at30", "at60", "at90", "itt"))
    assert (e30 <= e60).all() and (e60 <= e90).all() and (e90 <= eitt).all()
    return 4 - (e30 + e60 + e90 + eitt)


def sample_hrs(co, rows):
    """표본(rows)에서 다시 적합한 조·보정 위험비. {분석: (조, 보정)}"""
    from lifelines import CoxPHFitter
    d = co.iloc[rows]
    out = {}
    for key, _, _ in ANALYSES:
        df = d[["sglt2"] + COVS[1:]].copy()
        df["age"] = d["age_int"].values
        df["t"], df["e"] = d[f"t_{key}"].values, d[f"e_{key}"].values
        cr = CoxPHFitter().fit(df[["sglt2", "t", "e"]], "t", "e")
        ad = CoxPHFitter().fit(df, "t", "e")
        out[key] = (float(np.exp(cr.params_["sglt2"])), float(np.exp(ad.params_["sglt2"])))
    return out


def select_sample(co, J, seed=SEED_SELECT, iters=40_000, ncand=150, rounds=25, verbose=False):
    """표본에 넣을 코호트 행 번호(정렬됨)와 (소수 둘째 자리까지 맞은 개수(발생률 8, 발생률비 4, 위험비 8), 위험비 거리)를
    돌려준다."""
    rng = np.random.default_rng(seed)
    nS_ = int(round(J["flow"]["nS"] / FRAC))
    n = {"S": nS_, "D": int(round(J["flow"]["n"] / FRAC)) - nS_}
    grp, reason, level = co["grp"].values, co["r_at60"].values, event_level(co)
    fu = J["fu"]
    feats, target, tol, names = [], [], [], []

    def add(name, col, tgt, tl):
        names.append(name); feats.append(np.asarray(col, float)); target.append(float(tgt)); tol.append(float(tl))

    age = co["age_int"].values.astype(float)
    ev = {lab: {k: int(round(fu[k][lab]["ev"] / FRAC)) for k, _, _ in ANALYSES} for lab in ("S", "D")}
    for lab in ("S", "D"):
        g = (grp == lab).astype(float)
        # 인년: 사건 수가 정수라서, 발생률이 본문과 같아지도록 인년을 정한다(본문 인년의 1/30과 0.2–3.5% 다르다)
        for k, _, _ in ANALYSES:
            add(f"py_{k}_{lab}", g * co[f"t_{k}"].values / 365.25,
                ev[lab][k] / (fu[k][lab]["rate"] / 1000) + PY_NUDGE.get((k, lab), 0.0), 0.15)
        add(f"dthitt_{lab}", g * (co["r_itt"].values == "death"), round(fu["itt"]["reasons"][lab]["death"] / FRAC), 0.6)
        for v in COVS[1:]:
            add(f"{v}_{lab}", g * co[v].values, pct_count(J["base"][v][lab][1], n[lab]), 0.4)
        mu, sd = J["base"]["age"][lab]
        add(f"age_{lab}", g * age, mu * n[lab], 2.0)
        add(f"age2_{lab}", g * age * age, (sd ** 2 * (n[lab] - 1) / n[lab] + mu ** 2) * n[lab], 300.0)
        for k, _, _ in ANALYSES:           # 군별 잔차 합 = 0  (치료 변수의 점수 방정식)
            add(f"res_{k}_{lab}", g * (co[f"e_{k}"].values - co[f"E_{k}"].values), 0.0, 0.08)
            add(f"cru_{k}_{lab}", g * (co[f"e_{k}"].values - co[f"C_{k}"].values), 0.0, 0.08)
    for k, _, _ in ANALYSES:               # 공변량의 점수 방정식
        r = co[f"e_{k}"].values - co[f"E_{k}"].values
        for v in COVS:
            x = age if v == "age" else co[v].values.astype(float)
            add(f"sc_{k}_{v}", x * r, 0.0, 5.0 if v == "age" else 0.12)
    F = np.column_stack(feats)
    T, W = np.array(target), 1 / np.array(tol) ** 2

    # 1) 군 × 종료 사유(유예 60일) × 사건 수준으로 층화 추출. 사건 수는 층의 크기로 정확히 정해진다
    strata, pick_ = [], []
    for lab in ("S", "D"):
        R = largest_remainder([fu["at60"]["reasons"][lab][r] for r in REASONS], n[lab])
        R[0] = ev[lab]["at60"]
        R[-1] += n[lab] - R.sum()
        Lv = [ev[lab]["at30"], ev[lab]["at60"] - ev[lab]["at30"], ev[lab]["at90"] - ev[lab]["at60"],
              ev[lab]["itt"] - ev[lab]["at90"]]
        assert min(Lv) >= 0
        alloc = np.zeros((5, 5), int)                    # 사유 × 수준
        alloc[0, 0], alloc[0, 1] = Lv[0], Lv[1]
        for lv in (2, 3):
            cnt = np.array([((grp == lab) & (reason == r) & (level == lv)).sum() for r in REASONS[1:]], float)
            q = cnt / cnt.sum() * Lv[lv]
            base = np.floor(q).astype(int)
            for i in np.argsort(-(q - base))[: Lv[lv] - base.sum()]:
                base[i] += 1
            alloc[1:, lv] = base
        alloc[1:, 4] = R[1:] - alloc[1:, 2] - alloc[1:, 3]
        assert (alloc >= 0).all() and alloc.sum() == n[lab]
        for a, r in enumerate(REASONS):
            for lv in range(5):
                m = int(alloc[a, lv])
                if m == 0:
                    continue
                pool = np.flatnonzero((grp == lab) & (reason == r) & (level == lv))
                ins = rng.choice(pool, size=m, replace=False)
                strata.append((np.setdiff1d(pool, ins), len(pick_), m))
                pick_.extend(ins.tolist())
    pick_ = np.array(pick_)
    S = F[pick_].sum(axis=0)
    # 2) 같은 층 안에서 맞바꾸기: 표본의 한 사람을 빼고, 후보 ncand명 가운데 목적함수를 가장 줄이는 사람을 넣는다
    sizes = np.array([m if len(o) else 0 for o, _, m in strata], float)
    prob = sizes / sizes.sum()

    def swap(n_iter, S):
        obj = float((W * (S - T) ** 2).sum())
        for it in range(n_iter):
            h = rng.choice(len(strata), p=prob)
            out, off, m = strata[h]
            i = off + rng.integers(m)
            cand = rng.integers(len(out), size=min(ncand, len(out)))
            S2 = (S - F[pick_[i]])[None, :] + F[out[cand]]
            o2 = (W * (S2 - T) ** 2).sum(axis=1)
            b = int(o2.argmin())
            if o2[b] < obj:
                j = cand[b]
                pick_[i], out[j] = out[j], pick_[i]
                S, obj = S2[b], float(o2[b])
        return S, obj

    S, obj = swap(iters, S)
    # 3) 위험비 되먹임: 표본에서 Cox 모형을 다시 적합해 본문의 위험비와 견주고, 어긋난 만큼 SGLT2 억제제군의 잔차 합
    #    목표를 옮긴 뒤(위험비가 목표의 r배면 -사건 수 × ln r) 맞바꾸기를 이어 간다. 본문에 가장 가까웠던 표본을 돌려준다.
    want = {k: (fu[k]["crude"]["hr"], fu[k]["adj"]["hr"]) for k, _, _ in ANALYSES}

    def rates_ok():
        """표본의 발생률(군별)과 발생률비가 본문과 소수 둘째 자리까지 같은 개수 (최대 12)"""
        d, ok = co.iloc[pick_], 0
        for k in want:
            r = {}
            for lab, v in (("S", 1), ("D", 0)):
                m = d["sglt2"].values == v
                r[lab] = 1000 * d[f"e_{k}"].values[m].sum() / (d[f"t_{k}"].values[m].sum() / 365.25)
                ok += round(r[lab], 2) == round(fu[k][lab]["rate"], 2)
            ok += round(r["S"] / r["D"], 2) == round(fu[k]["irr"], 2)
        return ok

    best, shift = None, {}
    for rnd in range(rounds + 1):
        hr = sample_hrs(co, pick_)
        nok = rates_ok() + sum(round(hr[k][i], 2) == round(want[k][i], 2) for k in want for i in (0, 1))
        dist = sum(abs(hr[k][i] - want[k][i]) for k in want for i in (0, 1))
        if verbose:
            print(f"    round {rnd}: objective {obj:8.1f}  matching to 2 decimals {nok}/{5 * len(want)}  HR distance {dist:.4f}")
        if best is None or (nok, -dist) > (best[0], -best[1]):
            best = (nok, dist, pick_.copy(), S.copy(), obj, hr)
        if rnd == rounds or (nok == 5 * len(want) and dist < 0.012):
            break
        for k in want:
            for i, pre in ((0, "cru"), (1, "res")):
                name = f"{pre}_{k}_S"
                new = shift.get(name, 0.0) - 0.6 * ev["S"][k] * np.log(hr[k][i] / want[k][i])
                new = float(np.clip(new, -1.5, 1.5))             # 목표가 달아나지 않게 묶어 둔다
                T[names.index(name)] += new - shift.get(name, 0.0)
                shift[name] = new
        S, obj = swap(iters // 6, S)
    nok, dist, pick_, S, obj, hr = best
    if verbose:
        for nm, s_, t_, tl in zip(names, S, T, tol):
            flag = "" if abs(s_ - t_) <= 2 * tl else "   <-- off"
            print(f"    {nm:16s} sample {s_:12.3f}  target {t_:12.3f}{flag}")
        print(f"    objective {obj:.2f}; sample HRs (crude, adjusted): " + ", ".join(f"{k} {a:.4f} {b:.4f}" for k, (a, b) in hr.items()))
    return np.sort(pick_), (nok, dist)


# ====================================================================================== 3. 선정 흐름의 제외 대상
def flow_design(J):
    """단계 × 계열별 표본 인원(본문 인원의 1/30을 반올림하고, 합이 맞도록 끝자리를 조정)"""
    F = J["flow"]
    N0 = F["N0"] // FRAC
    step_n = [int(round(s["n"] / FRAC)) for s in F["steps"]]
    n_final = int(round(F["n"] / FRAC))
    assert N0 - sum(step_n) == n_final, (N0, step_n, n_final)
    design = []
    for s, tot in zip(F["steps"], step_n):
        q = np.array([s["S"], s["D"], s["B"]], float) / FRAC
        base = np.floor(q).astype(int)
        for i in np.argsort(-(q - base))[: tot - base.sum()]:
            base[i] += 1
        design.append(dict(zip("SDB", base.tolist())))
    return N0, step_n, n_final, design


def sample_excluded(ns, J, rng):
    """nums_ch14의 30만 명 표지에서 단계 × 계열별로 정해진 수만큼 뽑는다(표지는 그대로 가져온다)."""
    cls = ns["cls"]
    flags = dict(prev=ns["prev"], both=ns["both"], minor=ns["minor"], esrd=ns["esrd"], phhf=ns["phhf"])
    order = ["prev", "both", "minor", "esrd", "phhf"]
    first = np.full(len(cls), -1)
    for k, name in reversed(list(enumerate(order))):
        first = np.where(flags[name], k, first)
    _, _, _, design = flow_design(J)
    rows = []
    for k, d in enumerate(design):
        for c, m in d.items():
            pool = np.flatnonzero((first == k) & (cls == c))
            for i in rng.choice(pool, size=m, replace=False):
                rows.append(dict(cls=c, **{name: bool(flags[name][i]) for name in order}))
    return pd.DataFrame(rows)


# ====================================================================================== 4. 원자료 표 만들기
E11 = (["E119"] * 6 + ["E118"] * 2 + ["E116", "E114"])
DX = dict(htn=["I10"], ihd=["I209", "I209", "I251", "I259", "I252", "I200"], hf=["I509", "I500", "I501"],
          ckd=["N183", "N189", "N184", "N182"])
HHF_MAIN = ["I500", "I500", "I509", "I501"]
ESRD = ["N185", "N185", "Z491", "Z992"]
OTHER_OUT = ["J00", "J209", "K210", "M545", "H259", "E785", "J304", "M170", "K297", "L239"]
OTHER_IN = ["J189", "S720", "K800", "I639", "N390", "K566"]
IN_WITH_HF_SUB = ["I269", "J189", "I480", "N179"]
SUPPLIES = np.array([30, 60, 90])


class Tables:
    def __init__(self):
        self.person, self.visit, self.rx = [], [], []

    def add_visit(self, k, day, setting, codes):
        codes = list(dict.fromkeys(codes))[:3] + ["", ""]
        self.visit.append((k, int(day), setting, codes[0], codes[1], codes[2]))

    def add_rx(self, k, day, drug, days):
        self.rx.append((k, int(day), drug, int(days)))


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def days_in(rng, lo, hi, k):
    """lo..hi(포함)에서 서로 다른 날 k개, 오름차순"""
    k = int(min(k, hi - lo + 1))
    return np.sort(rng.choice(np.arange(lo, hi + 1), size=k, replace=False))


def baseline_block(T, rng, k, idx, cov, has_e11=True):
    """index date 이전 360일..1일 사이의 외래 방문 2–3건에 동반질환 코드를 나눠 적는다. 방문일 목록을 돌려준다."""
    need = [pick(rng, DX[v]) for v in ("htn", "ihd", "hf", "ckd") if cov.get(v)]
    rng.shuffle(need)
    nv = 2 + int(rng.random() < 0.45) + int(len(need) > 3)
    lo = max(idx - 360, DATA_START)
    days = days_in(rng, lo, idx - 1, nv)
    slots = [[] for _ in days]
    for i in range(len(days)):                 # 주상병: 대개 당뇨병, 가끔 동반질환
        if need and rng.random() < 0.25:
            slots[i].append(need.pop())
        else:
            slots[i].append(pick(rng, E11))
    i = 0
    while need:                                # 남은 코드는 부상병 자리에
        if len(slots[i % len(slots)]) < 3:
            slots[i % len(slots)].append(need.pop())
        i += 1
        if i > 50:
            slots.append([pick(rng, E11), need.pop()])
            days = np.append(days, days[-1])
    for d, codes in zip(days, slots):
        if len(codes) < 3 and rng.random() < 0.35:
            codes.append(pick(rng, ["E785", "E785", "K210", "M545"]) if codes[0] in E11 else pick(rng, E11))
        T.add_visit(k, d, "O", codes)
    return [int(d) for d in days]


def comed_block(T, rng, k, idx, bdays, met, ins, lim, light=False):
    """기저 기간의 메트포르민·인슐린 처방과, 기간 밖의 미끼 처방. 파일 크기 때문에 건수는 최소로 둔다."""
    if met:
        T.add_rx(k, bdays[0], "metformin", pick(rng, SUPPLIES))
        if not light and rng.random() < 0.12:
            T.add_rx(k, idx, "metformin", pick(rng, SUPPLIES))
    elif not light:
        if idx - 400 >= DATA_START and rng.random() < 0.05:           # 365일보다 오래전에 쓰다 끊음
            T.add_rx(k, int(rng.integers(DATA_START, idx - 399)), "metformin", pick(rng, SUPPLIES))
        if lim >= idx + 40 and rng.random() < 0.06:                    # index date 뒤에 시작
            T.add_rx(k, int(rng.integers(idx + 30, lim + 1)), "metformin", pick(rng, SUPPLIES))
    if ins:
        T.add_rx(k, bdays[-1], "insulin", 30)
    elif not light and lim >= idx + 40 and rng.random() < 0.02:
        T.add_rx(k, int(rng.integers(idx + 30, lim + 1)), "insulin", 30)
    if not light and rng.random() < 0.06:
        T.add_rx(k, bdays[0], "sulfonylurea", pick(rng, SUPPLIES))
    if not light and rng.random() < 0.08:
        T.add_rx(k, pick(rng, bdays + [idx]), "statin", pick(rng, SUPPLIES))


def noise_visits(T, rng, k, lim, rate=0.9):
    for _ in range(rng.poisson(rate)):
        d = int(rng.integers(DATA_START, lim + 1))
        if rng.random() < 0.06:
            T.add_visit(k, d, "I", [pick(rng, OTHER_IN)] + ([pick(rng, E11)] if rng.random() < 0.6 else []))
        else:
            T.add_visit(k, d, "O", [pick(rng, OTHER_OUT)] + ([pick(rng, E11)] if rng.random() < 0.2 else []))


def build_cohort_people(T, rng, co, cap, sel, nS):
    """표본에 든 코호트 사람들의 기록. nums_ch14의 값을 그대로 날짜로 옮긴다."""
    truth = []
    for gi in sel:
        r = co.iloc[gi]
        kind = r["grp"]
        li = gi if kind == "S" else gi - nS
        supply, start, runout, g, L = cap[kind]
        idx = int(r["idx"])
        adm, dth, hhf, sw = END, idx + int(r["death_day"]), idx + int(r["hhf_day"]), idx + int(r["switch"])
        lim = min(adm, dth)                              # 이날 뒤로는 기록이 없다
        k = len(T.person)
        own, other = ("SGLT2i", "DPP4i") if kind == "S" else ("DPP4i", "SGLT2i")
        cov = {v: int(r[v]) for v in COVS[1:]}
        age = int(r["age_int"])
        T.person.append(dict(sex="F" if cov["female"] else "M", birth_year=(DAY0 + pd.Timedelta(days=idx)).year - age,
                             death=dth if dth <= adm else None))
        # ---- 연구 약 처방: fills()가 만든 처방일(진입일 기준)과 공급일수 그대로
        s = int(supply[li])
        fill_days = []
        for j in range(int(min(L[li], start.shape[1]))):
            d = idx + int(start[li, j])
            if d > lim:
                break
            T.add_rx(k, d, own, s)
            fill_days.append(d)
        # ---- 비교약으로 변경·추가
        if sw <= lim:
            s2 = int(pick(rng, SUPPLIES))
            T.add_rx(k, sw, other, s2)
            d2 = sw + s2 + int(rng.integers(0, 8))
            if d2 <= lim and rng.random() < 0.6:
                T.add_rx(k, d2, other, s2)
        # ---- 기저 기간(이전 365일)의 상병과 병용약
        bdays = baseline_block(T, rng, k, idx, cov)
        if cov["hf"] and rng.random() < 0.12:            # 심부전이 부상병인 입원(주상병이 아니므로 제외 기준에 걸리지 않음)
            T.add_visit(k, pick(rng, bdays), "I", [pick(rng, IN_WITH_HF_SUB), pick(rng, DX["hf"])])
        comed_block(T, rng, k, idx, bdays, cov["met"], cov["ins"], lim)
        subs = [pick(rng, DX[v]) for v in ("htn", "ihd", "ckd") if cov[v] and rng.random() < 0.6]
        T.add_visit(k, idx, "O", [pick(rng, E11)] + (subs or (["E785"] if rng.random() < 0.3 else [])))
        # ---- 365일보다 오래된 기록(기저 기간 밖이므로 세지 않아야 함)
        if idx - 400 >= DATA_START:
            hi = idx - 400
            if rng.random() < 0.012:
                T.add_visit(k, int(rng.integers(DATA_START, hi + 1)), "I", [pick(rng, HHF_MAIN), pick(rng, E11)])
            for v in ("ihd", "ckd", "htn"):
                if not cov[v] and rng.random() < 0.02:
                    T.add_visit(k, int(rng.integers(DATA_START, hi + 1)), "O", [pick(rng, DX[v]), pick(rng, E11)])
        # ---- 결과: 주상병이 심부전인 입원
        if hhf <= lim:
            T.add_visit(k, hhf, "I", [pick(rng, HHF_MAIN), pick(rng, E11)] + (["I10"] if cov["htn"] else []))
            if rng.random() < 0.5 and hhf + 30 <= lim:   # 퇴원 뒤 외래(주상병 심부전이지만 입원이 아님)
                T.add_visit(k, hhf + int(rng.integers(7, 31)), "O", [pick(rng, DX["hf"]), pick(rng, E11)])
            d2 = hhf + int(rng.integers(20, 400))
            if rng.random() < 0.3 and d2 <= lim:         # 재입원(첫 입원만 사건으로 센다)
                T.add_visit(k, d2, "I", [pick(rng, HHF_MAIN), pick(rng, E11)])
        # ---- 추적 중의 외래(처방일 가운데 몇 번), 새로 생긴 진단, 심부전이 부상병인 입원
        later = [d for d in fill_days[1:]]
        for d in (rng.choice(later, size=min(len(later), 2), replace=False) if later else []):
            codes = [pick(rng, E11)] + [pick(rng, DX[v]) for v in ("htn", "hf", "ihd", "ckd") if cov[v] and rng.random() < 0.4]
            T.add_visit(k, d, "O", codes)
        if lim > idx + 30:
            for v in ("ckd", "ihd", "htn", "hf"):
                if not cov[v] and rng.random() < 0.035:
                    T.add_visit(k, int(rng.integers(idx + 20, lim + 1)), "O", [pick(rng, E11), pick(rng, DX[v])])
            if rng.random() < 0.02:
                T.add_visit(k, int(rng.integers(idx + 20, lim + 1)), "I", [pick(rng, IN_WITH_HF_SUB), pick(rng, DX["hf"]), pick(rng, E11)])
        noise_visits(T, rng, k, lim)
        truth.append(dict(k=k, gi=int(gi), drug=own, sglt2=int(kind == "S"), idx=idx, age=age, **cov,
                          **{f"{a}_{key}": r[f"{a}_{key}"] for key, _, _ in ANALYSES for a in ("t", "e", "r")}))
    return pd.DataFrame(truth)


def simple_fills(T, rng, k, d0, drug, lim, nmax):
    s = int(pick(rng, SUPPLIES))
    d = d0
    for _ in range(nmax):
        if d > lim:
            break
        T.add_rx(k, d, drug, s)
        d = d + s + int(rng.integers(0, 9))


def build_excluded_people(T, rng, ex):
    """선정 기준에서 빠지는 사람들. 표지(prev, both, minor, esrd, phhf)에 맞는 기록을 만든다."""
    for r in ex.itertuples(index=False):
        k = len(T.person)
        first = "SGLT2i" if r.cls == "S" else "DPP4i"
        if r.prev:
            # 2015년부터 쓰던 사람. 2016년의 첫 처방이 index date가 된다
            lapsed = rng.random() < 0.08                 # 마지막 처방이 190–360일 전(세척 180일이면 신규 사용자)
            s = int(pick(rng, SUPPLIES))
            delta = int(rng.integers(190, 361)) if lapsed else s + int(rng.integers(0, 21))
            idx = int(rng.integers(0, delta))            # 0 .. delta-1  → 이전 처방은 2015년
            if r.cls == "S":
                prior = "SGLT2i" if rng.random() < 0.55 else "DPP4i"      # DPP-4 억제제를 쓰다 바꾼 사람 포함
            elif r.cls == "D":
                prior = "DPP4i" if rng.random() < 0.96 else "SGLT2i"
            else:
                prior = pick(rng, ["DPP4i", "DPP4i", "SGLT2i"])
            d = idx - delta
            T.add_rx(k, d, prior, s)
            d2 = d - s - int(rng.integers(0, 11))
            if d2 >= DATA_START and rng.random() < 0.12:
                T.add_rx(k, d2, prior, s)
        else:
            a, b = (1.9, 1.0) if r.cls == "S" else (1.1, 1.0)
            idx = int(np.floor(2192 * rng.beta(a, b)))
        age = int(rng.integers(10, 18)) if r.minor else int(np.clip(round(rng.normal(61, 12)), 18, 95))
        dth = idx + int(rng.integers(30, 1500)) if rng.random() < 0.03 else None
        if dth is not None and dth > END:
            dth = None
        lim = min(END, dth) if dth is not None else END
        T.person.append(dict(sex="F" if rng.random() < 0.45 else "M",
                             birth_year=(DAY0 + pd.Timedelta(days=idx)).year - age, death=dth))
        # ---- index date의 처방과 그 뒤 몇 번
        nmax = (1 + int(rng.random() < 0.1)) if r.prev else int(rng.integers(2, 6))
        if r.cls == "B":
            simple_fills(T, rng, k, idx, "SGLT2i", lim, nmax)
            simple_fills(T, rng, k, idx, "DPP4i", lim, nmax)
        else:
            simple_fills(T, rng, k, idx, first, lim, nmax)
        # ---- 기저 기간의 상병
        cov = dict(htn=rng.random() < 0.6, ihd=rng.random() < 0.13, hf=rng.random() < (0.5 if r.phhf else 0.06),
                   ckd=rng.random() < (0.6 if r.esrd else 0.07))
        bdays = baseline_block(T, rng, k, idx, cov)
        lo = max(idx - 360, DATA_START)
        if r.esrd:
            for d in days_in(rng, lo, idx - 1, 1 + int(rng.random() < 0.5)):
                T.add_visit(k, d, "O" if rng.random() < 0.8 else "I", [pick(rng, ESRD), pick(rng, E11)])
        if r.phhf:
            T.add_visit(k, int(rng.integers(lo, idx)), "I", [pick(rng, HHF_MAIN), pick(rng, E11)])
        comed_block(T, rng, k, idx, bdays, rng.random() < 0.3, rng.random() < 0.05, lim, light=True)
        T.add_visit(k, idx, "O", [pick(rng, E11)] + (["I10"] if cov["htn"] and rng.random() < 0.5 else []))
        noise_visits(T, rng, k, lim, rate=0.5)


def build_extra_people(T, rng, n):
    """등록 기간(2016-01-01..2021-12-31) 밖에서만 두 계열을 처방받은 사람. 출발 인구에 들지 않는다."""
    for i in range(n):
        k = len(T.person)
        drug = "DPP4i" if rng.random() < 0.7 else "SGLT2i"
        age = int(np.clip(round(rng.normal(60, 12)), 20, 92))
        if i < n * 2 // 3:                               # 2022년에 처음 시작
            d0 = int(rng.integers(2192, END - 20))
            simple_fills(T, rng, k, d0, drug, END, int(rng.integers(1, 4)))
        else:                                            # 2015년에만 쓰고 끊음
            d0 = int(rng.integers(DATA_START, -150))
            s = int(pick(rng, SUPPLIES))
            T.add_rx(k, d0, drug, s)
            if d0 + s + 5 < -10 and rng.random() < 0.6:
                T.add_rx(k, d0 + s + int(rng.integers(0, 6)), drug, s)
        T.person.append(dict(sex="F" if rng.random() < 0.45 else "M",
                             birth_year=(DAY0 + pd.Timedelta(days=d0)).year - age, death=None))
        T.add_visit(k, d0, "O", [pick(rng, E11)] + (["I10"] if rng.random() < 0.5 else []))
        if rng.random() < 0.7:
            T.add_rx(k, d0, "metformin", pick(rng, SUPPLIES))
        noise_visits(T, rng, k, END, rate=0.8)


def to_date(days):
    return (DAY0 + pd.to_timedelta(np.asarray(days, dtype="int64"), unit="D")).strftime("%Y-%m-%d")


def assemble(T, truth, rng):
    """사람 번호를 섞어 매기고(10001부터) 세 표와 정답 코호트를 DataFrame으로 만든다."""
    n = len(T.person)
    pid = 10001 + rng.permutation(n)
    p = pd.DataFrame(T.person)
    person = pd.DataFrame({"pid": pid, "sex": p["sex"], "birth_year": p["birth_year"].astype(int),
                           "elig_start": "2015-01-01", "elig_end": "2022-12-31", "death_date": ""})
    has = p["death"].notna().values
    dd = to_date(p.loc[has, "death"].astype(int).values)
    person.loc[has, "death_date"] = np.asarray(dd)
    person.loc[has, "elig_end"] = np.asarray(dd)                     # 사망하면 그날 자격이 끝난다
    person = person.sort_values("pid").reset_index(drop=True)

    v = pd.DataFrame(T.visit, columns=["k", "day", "setting", "dx_main", "dx_sub1", "dx_sub2"])
    v = v.drop_duplicates().copy()
    v["pid"] = pid[v["k"].values]
    v = v.sort_values(["day", "pid", "setting", "dx_main"]).reset_index(drop=True)
    visit = pd.DataFrame({"pid": v["pid"], "visit_date": to_date(v["day"].values), "setting": v["setting"],
                          "dx_main": v["dx_main"], "dx_sub1": v["dx_sub1"], "dx_sub2": v["dx_sub2"]})

    x = pd.DataFrame(T.rx, columns=["k", "day", "drug", "days"]).drop_duplicates(["k", "day", "drug"]).copy()
    x["pid"] = pid[x["k"].values]
    x = x.sort_values(["day", "pid", "drug"]).reset_index(drop=True)
    rx = pd.DataFrame({"pid": x["pid"], "rx_date": to_date(x["day"].values), "drug": x["drug"], "days": x["days"]})

    c = truth.copy()
    c["pid"] = pid[c["k"].values]
    lab = {"hhf": "hhf", "death": "death", "switch": "switch", "disc": "stop", "admin": "study_end"}
    cohort = pd.DataFrame({
        "pid": c["pid"], "arm": c["drug"], "sglt2": c["sglt2"], "index_date": to_date(c["idx"].values),
        "age": c["age"], **{v_: c[v_] for v_ in COVS[1:]},
        "end_date": to_date((c["idx"] + c["t_at60"]).values), "reason": c["r_at60"].map(lab),
        "time": c["t_at60"].astype(int), "event": c["e_at60"].astype(int),
        "time_itt": c["t_itt"].astype(int), "event_itt": c["e_itt"].astype(int)})
    cohort = cohort.sort_values("pid").reset_index(drop=True)
    return person, visit, rx, cohort


def full_cohort(co):
    """전체 코호트(166,548명)의 분석용 표. ITT 추적이 as-treated 추적과 같은 사람은 ITT 칸을 비운다(모듈 설명)."""
    same = (co["t_itt"].values == co["t_at60"].values) & (co["e_itt"].values == co["e_at60"].values)
    assert set(co.loc[~same, "r_at60"]) <= {"disc", "switch"}
    full = pd.DataFrame({"sglt2": co["sglt2"].astype(int), "age": co["age_int"].astype(int),
                         **{v: co[v].astype(int) for v in COVS[1:]},
                         "time": co["t_at60"].astype(int), "event": co["e_at60"].astype(int),
                         "time_itt": co["t_itt"].astype("Int64").where(~same),
                         "event_itt": co["e_itt"].astype("Int64").where(~same)})
    return full.reset_index(drop=True)


def check_full(path, J):
    """쓴 파일을 다시 읽어 표 14-5의 숫자(사건, 인년, 조·보정 위험비와 신뢰구간)가 나오는지 확인한다."""
    from lifelines import CoxPHFitter
    f = pd.read_csv(path)
    f["time_itt"] = f["time_itt"].fillna(f["time"])
    f["event_itt"] = f["event_itt"].fillna(f["event"])
    for key, tc, ec in (("at60", "time", "event"), ("itt", "time_itt", "event_itt")):
        for lab, k in (("S", 1), ("D", 0)):
            g = f[f["sglt2"] == k]
            assert int(g[ec].sum()) == J["fu"][key][lab]["ev"]
            assert abs(g[tc].sum() / 365.25 - J["fu"][key][lab]["py"]) < 1e-6
        for nm, cols in (("crude", ["sglt2"]), ("adj", ["sglt2"] + COVS)):
            r = CoxPHFitter().fit(f[[tc, ec] + cols], tc, ec).summary.loc["sglt2"]
            got = (r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"])
            ref = (J["fu"][key][nm]["hr"], J["fu"][key][nm]["lo"], J["fu"][key][nm]["hi"])
            assert all(round(a, 2) == round(b, 2) for a, b in zip(got, ref)), (key, nm, got, ref)
    return len(f)


def build(verbose=True):
    J = json.load(open(os.path.join(HERE, "_ch14_nums.json")))
    ns, cap = load_ch14()
    co = add_followup(ns["co"])
    check_against_json(co, J)
    assert (co["dc30"] <= co["dc60"]).all() and (co["dc60"] <= co["dc90"]).all()
    for key, _, _ in ANALYSES:
        co[f"E_{key}"], hr = expected_events(co, key)
        co[f"C_{key}"], hr0 = expected_events(co, key, adjusted=False)
        if verbose:
            print(f"  full cohort HR ({key}, integer age): crude {hr0:.4f} adjusted {hr:.4f}"
                  f"  (chapter {J['fu'][key]['crude']['hr']:.4f}, {J['fu'][key]['adj']['hr']:.4f})")
    sel, obj = select_sample(co, J, verbose=verbose)
    rng = np.random.default_rng(SEED_TABLES)
    ex = sample_excluded(ns, J, rng)
    T = Tables()
    truth = build_cohort_people(T, rng, co, cap, sel, ns["nS"])
    build_excluded_people(T, rng, ex)
    build_extra_people(T, rng, N_EXTRA)
    return assemble(T, truth, rng), (J, co, sel, ex)


if __name__ == "__main__":
    (person, visit, rx, cohort), (J, co, sel, ex) = build()
    os.makedirs(PUBDATA, exist_ok=True)
    for name, df in (("claims_person", person), ("claims_visit", visit), ("claims_rx", rx), ("sglt2_cohort", cohort)):
        path = os.path.join(PUBDATA, name + ".csv")
        df.to_csv(path, index=False)
        size = os.path.getsize(path)
        assert size < 3_000_000, (name, size)
        print(f"{name}.csv  {len(df):7d} rows  {size / 1e6:.2f} MB")
    path = os.path.join(PUBDATA, "sglt2_cohort_full.csv")
    full_cohort(co).to_csv(path, index=False)
    size = os.path.getsize(path)
    assert size < 5_000_000, size
    print(f"sglt2_cohort_full.csv  {check_full(path, J):7d} rows  {size / 1e6:.2f} MB  (table 14-5 reproduced)")
    N0, step_n, n_final, design = flow_design(J)
    assert len(person) == N0 + N_EXTRA and len(cohort) == n_final
    print("flow design (1/30):", N0, step_n, n_final, design)
    print("cohort:", cohort["arm"].value_counts().to_dict(), " events:", cohort.groupby("arm")["event"].sum().to_dict())
