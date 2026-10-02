"""21장(비용 자료 분석)의 모든 숫자를 계산한다.
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch21.py
결과: gen/_ch21_nums.json (본문·그림이 읽는 숫자), gen/_ch21_cost.csv (환자 1명 1행 모의 자료),
      gen/_ch21_boot.npy (나 절 부트스트랩 분포), 화면 출력(본문의 코드 상자에 옮긴 실제 출력)

모의 자료: 암 등록 자료와 연계한 건강보험 청구자료(가상)에서 진행성 신세포암 1차 치료로
신약 A 또는 표준요법 B를 시작한 3,000명의 치료 시작 후 1년 의료비(만원, 급여 진료비 기준).
자료 생성은 gen/lib_p4.py의 입력값(무진행·전체 생존곡선, 월 약값 190/120, 무진행 월 40, 진행 월 250,
임종기 800, 이상반응 120/80)을 환자 단위로 풀어 쓴 것이며, 환자마다 A를 썼을 때와 B를 썼을 때의
비용을 모두 만들어 '참값'(같은 환자에게 A를 쓰면 B보다 얼마나 더 드는가)을 안다.
"""
import io
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lib_p4 as L

warnings.filterwarnings("ignore")
P = L.base_params()
N = {}                      # every number used in the chapter


def hdr(s):
    print("\n" + "=" * 8, s)


# ====================================================================== 자료 생성
def lognorm_mean(rng, mean, sigma, size):
    """평균이 mean이 되도록 맞춘 로그정규 난수"""
    return rng.lognormal(math.log(mean) - sigma ** 2 / 2, sigma, size)


def simulate(n, seed, cens=False, cens_seed=1):
    """환자 n명. 두 치료의 잠재 결과(1년 비용)를 같은 난수로 만든다."""
    rng = np.random.default_rng(seed)
    age = np.clip(np.round(rng.normal(64, 9, n)), 40, 88)
    male = (rng.random(n) < 0.72).astype(int)
    cci = np.minimum(rng.poisson(np.exp(0.30 + 0.025 * (age - 64))), 6)       # 동반질환 점수 0–6
    stage4 = (rng.random(n) < 0.62).astype(int)                               # 원격 전이(4기)
    # 치료 배정: 젊고 동반질환이 적은 환자가 신약 A를 더 많이 받는다(적응증에 따른 교란)
    lp = -0.42 - 0.050 * (age - 64) - 0.40 * (cci - 1.4) - 0.25 * (stage4 - 0.62)
    a = (rng.random(n) < 1 / (1 + np.exp(-lp))).astype(int)
    # 공통 난수
    u_p, u_d = rng.random(n), rng.random(n)
    rdi = rng.beta(12, 1, n)                              # 상대 용량강도(감량·휴약)
    g_op = rng.gamma(1 / 0.40 ** 2, 0.40 ** 2, n)         # 외래 이용 강도(평균 1, 변동계수 0.40)
    frail = rng.gamma(1 / 0.30 ** 2, 0.30 ** 2, n)        # 환자별 의료이용 성향(평균 1, 변동계수 0.30)
    u_ae, t_ae = rng.random(n), rng.random(n) * 3
    c_ae = lognorm_mean(rng, 600, 0.8, n)
    u_term = rng.random(n)
    c_term = lognorm_mean(rng, 940, 0.9, n)
    KMAX = 6
    u_pd = rng.random(n)
    u_bg = rng.random(n)
    c_pdadm = lognorm_mean(rng, 700, 0.9, (n, KMAX))
    w_pdadm = rng.random((n, KMAX))
    c_bg = lognorm_mean(rng, 500, 1.0, (n, KMAX))
    w_bg = rng.random((n, KMAX))
    # 자료 마감까지 남은 기간(개월). 다른 난수와 섞이지 않게 따로 뽑는다
    cdate = np.random.default_rng(cens_seed).uniform(6, 30, n) if cens else np.full(n, np.inf)

    m_p = np.exp(0.35 * (stage4 - 0.62) + 0.05 * (cci - 1.4))
    m_d = np.exp(0.50 * (stage4 - 0.62) + 0.18 * (cci - 1.4) + 0.025 * (age - 64))
    r_pd = 0.13 * np.exp(0.15 * (cci - 1.4))
    r_bg = 0.020 * np.exp(0.35 * (cci - 1.4) + 0.03 * (age - 64))
    out = {}
    for arm in ("A", "B"):
        hp = P["hr_pfs"] if arm == "A" else 1.0
        hd = P["hr_os"] if arm == "A" else 1.0
        tp = (-np.log(u_p) / (P["pfs_lam"] * hp * m_p)) ** (1 / P["pfs_gam"])
        td = (-np.log(u_d) / (P["os_lam"] * hd * m_d)) ** (1 / P["os_gam"])
        pfs = np.minimum(tp, td)

        def upto(c):
            """치료 시작부터 c개월(환자별)까지 쌓인 비용: 약제비, 외래·검사비, 입원비"""
            end = np.minimum(np.minimum(td, 12.0), c)                 # 관찰이 끝나는 시점
            mpf = np.minimum(pfs, end)
            mpd = end - mpf
            fills = np.minimum(np.ceil(np.minimum(pfs, 12.0)), np.ceil(end))   # 30일 처방 횟수
            drug = P["c_drug_" + arm] * fills * rdi
            op_pf = 23.0 * mpf * g_op * frail * np.exp(0.15 * (cci - 1.4))
            op_pd = 150.0 * mpd * g_op * frail * np.exp(0.15 * (cci - 1.4))
            op = op_pf + op_pd
            inp = np.zeros(n)
            i_pd = np.zeros(n)        # 진행 후 입원비
            i_bg = np.zeros(n)        # 무진행 기간의 그 밖의 입원비
            p_ae = (0.20 if arm == "A" else 0.80 / 6)
            inp += np.where((u_ae < p_ae) & (t_ae < end), c_ae, 0.0)
            # 진행 후 입원: 진행 기간(1년 안, 사망 전)에 포아송 과정
            full_pd = np.minimum(td, 12.0) - np.minimum(pfs, 12.0)
            k_pd = np.minimum(st.poisson.ppf(u_pd, r_pd * full_pd).astype(int), KMAX)
            k_bg = np.minimum(st.poisson.ppf(u_bg, r_bg * np.minimum(td, 12.0)).astype(int), KMAX)
            for j in range(KMAX):
                t_ev = np.minimum(pfs, 12.0) + w_pdadm[:, j] * full_pd
                i_pd += np.where((j < k_pd) & (t_ev <= end), c_pdadm[:, j], 0.0)
                t_ev = w_bg[:, j] * np.minimum(td, 12.0)
                i_bg += np.where((j < k_bg) & (t_ev <= end), c_bg[:, j], 0.0)
            inp += i_pd + i_bg
            inp += np.where((td < 12.0) & (u_term < 0.85) & (td <= end), c_term, 0.0)   # 임종기 입원
            inp = inp * frail
            self_ = dict(mpf=mpf, mpd=mpd, c_pfstate=op_pf + i_bg * frail, c_pdstate=op_pd + i_pd * frail)
            return drug, op, inp, self_

        d, o, i, ph = upto(np.full(n, 12.0))
        out[arm] = dict(drug=d, op=o, inp=i, td=td, pfs=pfs, **ph)
        if cens:
            d2, o2, i2, _ = upto(cdate)
            out[arm].update(drug_c=d2, op_c=o2, inp_c=i2)

    df = pd.DataFrame(dict(age=age, male=male, cci=cci, stage4=stage4, A=a))
    for comp in ("drug", "op", "inp"):
        df[comp] = np.round(np.where(a == 1, out["A"][comp], out["B"][comp]), 1)
    df["cost"] = np.round(df.drug + df.op + df.inp, 1)
    for k in ("td", "pfs", "mpf", "mpd", "c_pfstate", "c_pdstate"):
        df[k] = np.where(a == 1, out["A"][k], out["B"][k])
    po = {arm: out[arm]["drug"] + out[arm]["op"] + out[arm]["inp"] for arm in ("A", "B")}
    po_inp = {arm: out[arm]["inp"] for arm in ("A", "B")}
    if cens:
        df["cdate"] = cdate
        df["cost_c"] = np.round(sum(np.where(a == 1, out["A"][k], out["B"][k]) for k in ("drug_c", "op_c", "inp_c")), 1)
    return df, po, po_inp


SEED = 20261359      # 표시한 반올림 값끼리 검산이 맞는(합·차가 어긋나지 않는) 난수 씨앗을 골랐다
df, po, po_inp = simulate(3000, SEED, cens=True)
df.insert(0, "pid", [f"R{i + 1:04d}" for i in range(len(df))])
df["grp"] = np.where(df.A == 1, "A", "B")


def boot_ci(x, lo=2.5, hi=97.5):
    return [float(np.percentile(x, lo)), float(np.percentile(x, hi))]


def main():
    out_txt = {}                      # 본문 코드 상자에 옮길 실제 출력

    # ================================================================== 참값과 모형 입력값
    hdr("참값(200만 명 모의) / lib_p4 1년치")
    big, pob, pob_inp = simulate(2_000_000, 7)
    N["true"] = dict(ate=float((pob["A"] - pob["B"]).mean()), yA=float(pob["A"].mean()), yB=float(pob["B"].mean()),
                     ate_inp=float((pob_inp["A"] - pob_inp["B"]).mean()),
                     inpA=float(pob_inp["A"].mean()), inpB=float(pob_inp["B"].mean()),
                     crude=float(big.cost[big.A == 1].mean() - big.cost[big.A == 0].mean()),
                     ratio=float(pob["A"].mean() / pob["B"].mean()),
                     # 이 3,000명 각자의 두 잠재 비용(모의 자료라서 알 수 있는 값)
                     s_yA=float(po["A"].mean()), s_yB=float(po["B"].mean()), s_ate=float((po["A"] - po["B"]).mean()),
                     s_inpA=float(po_inp["A"].mean()), s_inpB=float(po_inp["B"].mean()),
                     s_ate_inp=float((po_inp["A"] - po_inp["B"]).mean()),
                     s_ratio=float(po["A"].mean() / po["B"].mean()))
    print(N["true"])
    # 비용의 비가 환자마다 같은가: 동반질환 점수별 참값(200만 명), 큰 표본(40만 명)에서의 모형별 추정
    het = {}
    for c in (0, 2, 4):
        k = big.cci.values == c
        het[f"cci{c}"] = dict(ratio=float(pob["A"][k].mean() / pob["B"][k].mean()),
                              diff=float((pob["A"][k] - pob["B"][k]).mean()), yB=float(pob["B"][k].mean()))
    sub = big.iloc[:400000]
    s1, s0 = sub.assign(A=1), sub.assign(A=0)
    fam_ = sm.families.Gamma(link=sm.families.links.Log())
    gm_ = smf.glm("cost ~ A + age + male + cci + stage4", sub, family=fam_).fit()
    gi_ = smf.glm("cost ~ A * (age + male + cci + stage4)", sub, family=fam_).fit()
    ol_ = smf.ols("cost ~ A + age + male + cci + stage4", sub).fit()
    het["big"] = dict(n=len(sub), truth=float((pob["A"][:400000] - pob["B"][:400000]).mean()),
                      crude=float(sub.cost[sub.A == 1].mean() - sub.cost[sub.A == 0].mean()),
                      glm=float(gm_.predict(s1).mean() - gm_.predict(s0).mean()),
                      glm_inter=float(gi_.predict(s1).mean() - gi_.predict(s0).mean()), ols=float(ol_.params["A"]))
    gs_ = smf.glm("cost ~ A * (age + male + cci + stage4)", df, family=fam_).fit()
    het["sample_inter"] = float(gs_.predict(df.assign(A=1)).mean() - gs_.predict(df.assign(A=0)).mean())
    N["het"] = het
    print(het)
    lib = {}
    for arm in "AB":
        r = L.psm(P, arm, horizon=12, disc=0.0)
        lib[arm] = float(r["cost"])
    lib["diff"] = lib["A"] - lib["B"]
    N["lib1y"] = lib
    print("lib 1-year undiscounted cost", lib)
    N["state"] = dict(pf=float(df.c_pfstate.sum() / df.mpf.sum()), pd=float(df.c_pdstate.sum() / df.mpd.sum()),
                      pm_pf=float(df.mpf.sum()), pm_pd=float(df.mpd.sum()),
                      sum_pf=float(df.c_pfstate.sum()), sum_pd=float(df.c_pdstate.sum()))
    print("state monthly cost (non-drug): PF, PD", N["state"])

    # ================================================================== 가. 청구 명세서 → 환자당 비용
    hdr("가. 청구 명세서 예")
    code_a = '''
import pandas as pd
cohort = pd.DataFrame({"pid": ["P01", "P02", "P03"],
                       "index_date": pd.to_datetime(["2024-03-04", "2024-05-20", "2024-07-01"])})
claims = pd.DataFrame(
    [("P01", "2024-03-04", "외래", 38.0, 1.9), ("P01", "2024-03-04", "약국", 190.0, 9.5),
     ("P01", "2024-04-03", "약국", 190.0, 9.5), ("P01", "2024-06-10", "입원", 412.0, 20.6),
     ("P01", "2025-03-10", "외래", 25.0, 1.3), ("P02", "2024-05-20", "외래", 41.0, 2.1),
     ("P02", "2024-05-20", "약국", 120.0, 6.0), ("P02", "2024-06-19", "약국", 120.0, 6.0),
     ("P03", "2024-02-15", "입원", 300.0, 15.0), ("P03", "2024-07-01", "외래", 36.0, 1.8),
     ("P03", "2024-07-01", "약국", 190.0, 9.5)],
    columns=["pid", "date", "type", "total", "oop"])
claims["date"] = pd.to_datetime(claims["date"])

m = claims.merge(cohort, on="pid")
m["day"] = (m["date"] - m["index_date"]).dt.days
win = m[(m["day"] >= 0) & (m["day"] < 365)]
cost = win.pivot_table(index="pid", columns="type", values="total", aggfunc="sum", fill_value=0)
cost = cost.reindex(cohort["pid"], fill_value=0)
cost["합계"] = cost.sum(axis=1)
cost["보험자부담"] = (win["total"] - win["oop"]).groupby(win["pid"]).sum()
print(cost)
print(cost.mean().round(1))
print(win[win["type"] == "입원"].groupby("pid")["total"].sum().mean())
'''
    buf = io.StringIO()
    so = sys.stdout
    sys.stdout = buf
    env = {}
    exec(code_a, env)
    sys.stdout = so
    out_txt["claims"] = buf.getvalue()
    print(out_txt["claims"])
    c = env["cost"]
    N["claims"] = dict(rows=len(env["claims"]), kept=len(env["win"]),
                       table={pid: {k: float(v) for k, v in c.loc[pid].items()} for pid in c.index},
                       mean={k: float(v) for k, v in c.mean().items()},
                       inp_users_mean=float(env["win"][env["win"]["type"] == "입원"].groupby("pid")["total"].sum().mean()))
    # 물가 보정·할인 예
    N["adj"] = dict(idx0=100.0, idx1=104.5, c0=1000.0, c1=1000.0 * 104.5 / 100.0,
                    disc1=1000.0 / 1.045, disc2=1000.0 / 1.045 ** 2, disc5=1000.0 / 1.045 ** 5)
    # 관점별 계산 예(스스로 확인하기): 급여 총액, 그중 본인부담, 비급여, 교통비, 간병비
    q = dict(cov=2400.0, oop=120.0, noncov=310.0, trans=45.0, care=280.0)
    q.update(payer=q["cov"] - q["oop"], system=q["cov"] + q["noncov"], claims=q["cov"],
             societal=q["cov"] + q["noncov"] + q["trans"] + q["care"])
    N["q_persp"] = q
    print(N["adj"], q)

    # ================================================================== 나. 분포와 평균 비교
    hdr("나. 분포와 평균 비교")
    a, b = df.cost[df.A == 1].values, df.cost[df.A == 0].values
    G = {}
    for g, y in (("A", a), ("B", b), ("all", df.cost.values)):
        srt = np.sort(y)[::-1]
        q1, med, q3 = np.percentile(y, [25, 50, 75])
        k1, k5, k10 = max(1, round(len(y) * 0.01)), round(len(y) * 0.05), round(len(y) * 0.10)
        G[g] = dict(n=len(y), mean=float(y.mean()), sd=float(y.std(ddof=1)), med=float(med), q1=float(q1), q3=float(q3),
                    min=float(y.min()), max=float(y.max()), total=float(y.sum()), skew=float(st.skew(y)),
                    top1=float(srt[:k1].sum() / y.sum()), top5=float(srt[:k5].sum() / y.sum()),
                    top10=float(srt[:k10].sum() / y.sum()), k1=int(k1),
                    mean_trim1=float(srt[k1:].mean()), below_mean=float((y < y.mean()).mean()),
                    gm=float(np.exp(np.log(y).mean())), sdlog=float(np.log(y).std(ddof=1)),
                    over5000=int((y > 5000).sum()), over10000=int((y > 10000).sum()))
        print(g, {k: round(v, 3) if isinstance(v, float) else v for k, v in G[g].items()})
    N["grp"] = G
    comp = {}
    for cname in ("drug", "op", "inp", "cost"):
        ya, yb = df[cname][df.A == 1].values, df[cname][df.A == 0].values
        w = st.ttest_ind(ya, yb, equal_var=False)
        ci = w.confidence_interval()
        bs = st.bootstrap((ya, yb), lambda x, y, axis=-1: x.mean(axis=axis) - y.mean(axis=axis), n_resamples=9999,
                          method="percentile", vectorized=True, random_state=np.random.default_rng(21))
        comp[cname] = dict(mA=float(ya.mean()), sA=float(ya.std(ddof=1)), mB=float(yb.mean()), sB=float(yb.std(ddof=1)),
                           medA=float(np.median(ya)), medB=float(np.median(yb)),
                           iqrA=[float(v) for v in np.percentile(ya, [25, 75])],
                           iqrB=[float(v) for v in np.percentile(yb, [25, 75])],
                           zeroA=float((ya == 0).mean()), zeroB=float((yb == 0).mean()),
                           nzA=int((ya == 0).sum()), nzB=int((yb == 0).sum()),
                           diff=float(ya.mean() - yb.mean()), t=float(w.statistic), p=float(w.pvalue),
                           t_ci=[float(ci.low), float(ci.high)], dfw=float(w.df),
                           boot=[float(bs.confidence_interval.low), float(bs.confidence_interval.high)],
                           boot_se=float(bs.standard_error),
                           mw_p=float(st.mannwhitneyu(ya, yb).pvalue),
                           auc=float(st.mannwhitneyu(ya, yb).statistic / (len(ya) * len(yb))))
        print(cname, {k: (np.round(v, 3).tolist() if not isinstance(v, int) else v) for k, v in comp[cname].items()})
    N["comp"] = comp
    ib = np.sort(df.inp[df.A == 0].values)[::-1]
    N["inpB"] = dict(top10=float(ib[:round(len(ib) * 0.10)].sum() / ib.sum()), max=float(ib.max()),
                     pos_mean=float(ib[ib > 0].mean()), pos_med=float(np.median(ib[ib > 0])), npos=int((ib > 0).sum()))
    N["budget"] = dict(tot_by_mean=G["B"]["mean"] * G["B"]["n"], tot_by_med=G["B"]["med"] * G["B"]["n"],
                       d_mean=comp["cost"]["diff"] * G["A"]["n"],
                       d_med=(G["A"]["med"] - G["B"]["med"]) * G["A"]["n"])
    print(N["inpB"], N["budget"])

    # --- 본문 코드 상자: scipy.stats.bootstrap (실제 출력)
    df.drop(columns=["grp"]).to_csv(os.path.join(HERE, "_ch21_cost.csv"), index=False)
    code_b = '''
import numpy as np, pandas as pd
from scipy import stats
df = pd.read_csv("gen/_ch21_cost.csv")
a = df.loc[df["A"] == 1, "cost"].to_numpy()      # 신약 A군 1,203명의 1년 비용(만원)
b = df.loc[df["A"] == 0, "cost"].to_numpy()      # 표준요법 B군 1,797명

def mean_diff(x, y, axis=-1):
    return x.mean(axis=axis) - y.mean(axis=axis)

print(round(mean_diff(a, b), 1))
ci = stats.ttest_ind(a, b, equal_var=False).confidence_interval()
print(round(ci.low, 1), round(ci.high, 1))
for method in ["percentile", "BCa"]:
    res = stats.bootstrap((a, b), mean_diff, n_resamples=9999, method=method,
                          random_state=np.random.default_rng(21))
    ci = res.confidence_interval
    print(method, round(ci.low, 1), round(ci.high, 1), round(res.standard_error, 1))
'''
    buf = io.StringIO()
    sys.stdout = buf
    env = {}
    cwd = os.getcwd()
    os.chdir(os.path.join(HERE, ".."))
    exec(code_b, env)
    os.chdir(cwd)
    sys.stdout = so
    out_txt["boot"] = buf.getvalue()
    print(out_txt["boot"])
    res_p = st.bootstrap((a, b), env["mean_diff"], n_resamples=9999, method="percentile", random_state=np.random.default_rng(21))
    res_b = st.bootstrap((a, b), env["mean_diff"], n_resamples=9999, method="BCa", random_state=np.random.default_rng(21))
    dist = res_p.bootstrap_distribution
    np.save(os.path.join(HERE, "_ch21_boot.npy"), dist)
    N["boot"] = dict(B=9999, pct=[float(res_p.confidence_interval.low), float(res_p.confidence_interval.high)],
                     bca=[float(res_b.confidence_interval.low), float(res_b.confidence_interval.high)],
                     se=float(res_p.standard_error), dmean=float(dist.mean()), dmin=float(dist.min()), dmax=float(dist.max()),
                     p_le0=float((dist <= 0).mean()),
                     se_formula=float(math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))))
    print(N["boot"])
    # --- 중앙값, Mann-Whitney, 로그 변환 t 검정
    la, lb = np.log(a), np.log(b)
    tl = st.ttest_ind(la, lb, equal_var=False)
    cil = tl.confidence_interval()
    N["log"] = dict(mlA=float(la.mean()), mlB=float(lb.mean()), gmA=float(np.exp(la.mean())), gmB=float(np.exp(lb.mean())),
                    gm_ratio=float(np.exp(la.mean() - lb.mean())), gm_ci=[float(np.exp(cil.low)), float(np.exp(cil.high))],
                    p=float(tl.pvalue), am_ratio=float(a.mean() / b.mean()),
                    med_diff=float(np.median(a) - np.median(b)), med_ratio=float(np.median(a) / np.median(b)),
                    smearA=float(np.exp(la - la.mean()).mean()), smearB=float(np.exp(lb - lb.mean()).mean()),
                    sdlA=float(la.std(ddof=1)), sdlB=float(lb.std(ddof=1)),
                    under_A=float(1 - np.exp(la.mean()) / a.mean()), under_B=float(1 - np.exp(lb.mean()) / b.mean()))
    N["log"]["backA"] = N["log"]["gmA"] * N["log"]["smearA"]
    N["log"]["backB"] = N["log"]["gmB"] * N["log"]["smearB"]
    N["log"]["gm_diff"] = N["log"]["gmA"] - N["log"]["gmB"]
    print("log", {k: np.round(v, 4).tolist() for k, v in N["log"].items()})

    # --- 중도절단된 비용 (심화)
    hdr("나. 중도절단된 비용")
    from lifelines import KaplanMeierFitter
    T = np.minimum(df.td.values, 12.0)                 # 1년 비용이 완성되는 시점(사망 또는 12개월)
    C = df.cdate.values
    comp_ = (C >= T)
    X = np.minimum(T, C)
    kmf = KaplanMeierFitter().fit(X, event_observed=(~comp_).astype(int))     # 중도절단의 '생존함수' K(t)
    K = kmf.survival_function_at_times(T).values
    cen = {}
    for g, msk in (("A", df.A.values == 1), ("B", df.A.values == 0)):
        n = int(msk.sum())
        full = float(df.cost.values[msk].mean())
        naive_all = float(df.cost_c.values[msk].mean())
        cc = float(df.cost.values[msk & comp_].mean())
        # 군별로 K를 따로 추정
        k_g = KaplanMeierFitter().fit(X[msk], event_observed=(~comp_[msk]).astype(int))
        Kg = k_g.survival_function_at_times(T[msk]).values
        ipw = float((comp_[msk] * df.cost.values[msk] / Kg).sum() / n)
        cen[g] = dict(n=n, n_cens=int((~comp_[msk]).sum()), p_cens=float((~comp_[msk]).mean()), full=full,
                      naive_all=naive_all, cc=cc, ipw=ipw, k12=float(k_g.survival_function_at_times(12.0).values[0]),
                      died_cc=float((df.td.values[msk & comp_] < 12).mean()), died_all=float((df.td.values[msk] < 12).mean()))
    cen["diff"] = {k: cen["A"][k] - cen["B"][k] for k in ("full", "naive_all", "cc", "ipw")}
    cen["p_cens"] = float((~comp_).mean())
    cen["n_cens"] = int((~comp_).sum())
    # 중도절단 시점만 500번 다시 뽑아 같은 계산을 반복(치우침과 우연 변동을 구분)
    rep_ = {k: [] for k in ("naive_all", "cc", "ipw")}
    for k in range(500):
        d2, _, _ = simulate(3000, SEED, cens=True, cens_seed=1000 + k)
        msk = d2.A.values == 0
        T2 = np.minimum(d2.td.values, 12.0)[msk]
        c2 = d2.cdate.values[msk] >= T2
        X2 = np.minimum(T2, d2.cdate.values[msk])
        K2 = KaplanMeierFitter().fit(X2, event_observed=(~c2).astype(int)).survival_function_at_times(T2).values
        rep_["naive_all"].append(d2.cost_c.values[msk].mean())
        rep_["cc"].append(d2.cost.values[msk][c2].mean())
        rep_["ipw"].append((c2 * d2.cost.values[msk] / K2).sum() / msk.sum())
    cen["rep"] = {k: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1))) for k, v in rep_.items()}
    N["cens"] = cen
    print(json.dumps(cen, indent=1))

    # ================================================================== 다. 회귀분석
    hdr("다. 회귀분석")
    base = {}
    for v in ("age", "male", "cci", "stage4"):
        xa, xb = df[v][df.A == 1], df[v][df.A == 0]
        sp = math.sqrt((xa.var(ddof=1) + xb.var(ddof=1)) / 2)
        base[v] = dict(mA=float(xa.mean()), sA=float(xa.std(ddof=1)), mB=float(xb.mean()), sB=float(xb.std(ddof=1)),
                       smd=float((xa.mean() - xb.mean()) / sp))
    base["cci2"] = dict(A=float((df.cci[df.A == 1] >= 2).mean()), B=float((df.cci[df.A == 0] >= 2).mean()))
    N["base"] = base
    print(json.dumps(base, indent=1))

    f = "cost ~ A + age + male + cci + stage4"
    d1, d0 = df.assign(A=1), df.assign(A=0)
    fam = sm.families.Gamma(link=sm.families.links.Log())
    ols = smf.ols(f, df).fit(cov_type="HC3")
    ols_plain = smf.ols(f, df).fit()
    lo = smf.ols("np.log(cost) ~ A + age + male + cci + stage4", df).fit()
    glm = smf.glm(f, df, family=fam).fit()
    poi = smf.glm(f, df, family=sm.families.Poisson()).fit(cov_type="HC0")
    R = {}
    R["crude"] = dict(diff=comp["cost"]["diff"], ci=comp["cost"]["boot"])
    R["ols"] = dict(diff=float(ols.params["A"]), ci=[float(v) for v in ols.conf_int().loc["A"]], p=float(ols.pvalues["A"]),
                    se=float(ols.bse["A"]), se_plain=float(ols_plain.bse["A"]),
                    coef={k: float(v) for k, v in ols.params.items()})
    R["ols"]["predA"], R["ols"]["predB"] = float(ols.predict(d1).mean()), float(ols.predict(d0).mean())
    sme = float(np.exp(lo.resid).mean())
    smeA, smeB = float(np.exp(lo.resid[df.A == 1]).mean()), float(np.exp(lo.resid[df.A == 0]).mean())
    nA, nB = float(np.exp(lo.predict(d1)).mean()), float(np.exp(lo.predict(d0)).mean())
    R["logols"] = dict(ratio=float(np.exp(lo.params["A"])), ratio_ci=[float(v) for v in np.exp(lo.conf_int().loc["A"])],
                       naiveA=nA, naiveB=nB, naive=nA - nB, smear=sme, smA=nA * sme, smB=nB * sme, smear_diff=(nA - nB) * sme,
                       smearA=smeA, smearB=smeB, grpA=nA * smeA, grpB=nB * smeB, grp_diff=nA * smeA - nB * smeB,
                       resid_sdA=float(lo.resid[df.A == 1].std(ddof=1)), resid_sdB=float(lo.resid[df.A == 0].std(ddof=1)))
    gA, gB = float(glm.predict(d1).mean()), float(glm.predict(d0).mean())
    R["glm"] = dict(ratio=float(np.exp(glm.params["A"])), ratio_ci=[float(v) for v in np.exp(glm.conf_int().loc["A"])],
                    p=float(glm.pvalues["A"]), predA=gA, predB=gB, diff=gA - gB,
                    coef={k: float(v) for k, v in glm.params.items()},
                    expcoef={k: float(np.exp(v)) for k, v in glm.params.items()},
                    expci={k: [float(x) for x in np.exp(glm.conf_int().loc[k])] for k in glm.params.index},
                    scale=float(glm.scale), obs_mean=float(df.cost.mean()), pred_mean=float(glm.predict(df).mean()))
    R["poisson"] = dict(ratio=float(np.exp(poi.params["A"])), diff=float(poi.predict(d1).mean() - poi.predict(d0).mean()))
    # 한 환자의 예측: 64세 남성, 동반질환 점수 1, 4기
    one = pd.DataFrame(dict(A=[1, 0], age=64, male=1, cci=1, stage4=1))
    pr = glm.predict(one).values
    one2 = pd.DataFrame(dict(A=[1, 0], age=75, male=1, cci=4, stage4=1))
    pr2 = glm.predict(one2).values
    R["glm"]["one"] = dict(A=float(pr[0]), B=float(pr[1]), diff=float(pr[0] - pr[1]))
    R["glm"]["one2"] = dict(A=float(pr2[0]), B=float(pr2[1]), diff=float(pr2[0] - pr2[1]))
    # modified Park test
    mu = glm.predict(df)
    park = sm.GLM((df.cost - mu) ** 2, sm.add_constant(np.log(mu)), family=fam).fit(cov_type="HC0")
    R["park"] = dict(lam=float(park.params.iloc[1]), ci=[float(v) for v in park.conf_int().iloc[1]])
    # 평균 구조 점검(검수에서 추가): 감마 GLM 예측값의 십분위별 관측 평균과 예측 평균, 연결 검정(link test)
    dec = pd.qcut(mu, 10, labels=False)
    chk = pd.DataFrame(dict(dec=dec, obs=df.cost, glm=mu, ols=ols.predict(df))).groupby("dec").mean()
    xb = np.log(mu)
    link = sm.GLM(df.cost, sm.add_constant(np.column_stack([xb, xb ** 2])), family=fam).fit(cov_type="HC0")
    R["check"] = dict(n_top=int((dec == 9).sum()), obs=[float(v) for v in chk.obs], glm=[float(v) for v in chk.glm],
                      ols=[float(v) for v in chk.ols],
                      top_glm_gap=float(chk.glm.iloc[9] - chk.obs.iloc[9]), top_ols_gap=float(chk.ols.iloc[9] - chk.obs.iloc[9]),
                      rest_glm_maxgap=float((chk.glm - chk.obs).iloc[:9].abs().max()),
                      rest_ols_maxgap=float((chk.ols - chk.obs).iloc[:9].abs().max()),
                      link_p=float(link.pvalues.iloc[2]))
    # two-part model: 입원비
    df["anyinp"] = (df.inp > 0).astype(int)
    d1, d0 = df.assign(A=1), df.assign(A=0)
    rhs = "A + age + male + cci + stage4"
    lg = smf.logit("anyinp ~ " + rhs, df).fit(disp=0)
    pos = df[df.inp > 0]
    gg = smf.glm("inp ~ " + rhs, pos, family=fam).fit()
    p1, p0 = lg.predict(d1), lg.predict(d0)
    m1, m0 = gg.predict(d1), gg.predict(d0)
    mu2 = gg.predict(pos)
    park2 = sm.GLM((pos.inp - mu2) ** 2, sm.add_constant(np.log(mu2)), family=fam).fit(cov_type="HC0")
    ols_i = smf.ols("inp ~ " + rhs, df).fit(cov_type="HC3")
    ci_ = comp["inp"]
    R["tp"] = dict(or_=float(np.exp(lg.params["A"])), or_ci=[float(v) for v in np.exp(lg.conf_int().loc["A"])], or_p=float(lg.pvalues["A"]),
                   ratio=float(np.exp(gg.params["A"])), ratio_ci=[float(v) for v in np.exp(gg.conf_int().loc["A"])], ratio_p=float(gg.pvalues["A"]),
                   pA=float(p1.mean()), pB=float(p0.mean()), mA=float(m1.mean()), mB=float(m0.mean()),
                   predA=float((p1 * m1).mean()), predB=float((p0 * m0).mean()),
                   diff=float((p1 * m1).mean() - (p0 * m0).mean()),
                   crude=ci_["diff"], crude_ci=ci_["boot"], n_pos=int(len(pos)), park=float(park2.params.iloc[1]),
                   ols=float(ols_i.params["A"]), ols_ci=[float(v) for v in ols_i.conf_int().loc["A"]],
                   obs_pA=float(df.anyinp[df.A == 1].mean()), obs_pB=float(df.anyinp[df.A == 0].mean()),
                   obs_mA=float(df.inp[(df.A == 1) & (df.inp > 0)].mean()), obs_mB=float(df.inp[(df.A == 0) & (df.inp > 0)].mean()),
                   n_posA=int(((df.A == 1) & (df.inp > 0)).sum()), n_posB=int(((df.A == 0) & (df.inp > 0)).sum()))
    # 한 환자 예: 64세 남성, CCI 1, 4기, B
    onep = pd.DataFrame(dict(A=[0], age=64, male=1, cci=1, stage4=1))
    R["tp"]["one"] = dict(p=float(lg.predict(onep).iloc[0]), m=float(gg.predict(onep).iloc[0]))
    R["tp"]["one"]["pm"] = R["tp"]["one"]["p"] * R["tp"]["one"]["m"]

    # --- 부트스트랩(환자 복원추출 1,000회)으로 표준화 차이의 CI
    Xn = np.column_stack([np.ones(len(df)), df.A, df.age, df.male, df.cci, df.stage4]).astype(float)
    y = df.cost.values
    yi = df.inp.values
    X1, X0 = Xn.copy(), Xn.copy()
    X1[:, 1], X0[:, 1] = 1, 0
    rb = np.random.default_rng(2101)
    Bn = 1000
    bs = {k: np.empty(Bn) for k in ("glm", "glm_ratio", "naive", "smear", "grp", "tp", "ols", "predA", "predB")}
    n = len(df)
    for i in range(Bn):
        ix = rb.integers(0, n, n)
        xb_, yb_, yib = Xn[ix], y[ix], yi[ix]
        g_ = sm.GLM(yb_, xb_, family=fam).fit()
        e1, e0 = np.exp(X1[ix] @ g_.params), np.exp(X0[ix] @ g_.params)
        bs["glm"][i] = e1.mean() - e0.mean()
        bs["predA"][i], bs["predB"][i] = e1.mean(), e0.mean()
        bs["glm_ratio"][i] = math.exp(g_.params[1])
        bl = np.linalg.lstsq(xb_, np.log(yb_), rcond=None)[0]
        res = np.log(yb_) - xb_ @ bl
        l1, l0 = np.exp(X1[ix] @ bl), np.exp(X0[ix] @ bl)
        bs["naive"][i] = l1.mean() - l0.mean()
        bs["smear"][i] = (l1.mean() - l0.mean()) * np.exp(res).mean()
        am = xb_[:, 1] == 1
        bs["grp"][i] = l1.mean() * np.exp(res[am]).mean() - l0.mean() * np.exp(res[~am]).mean()
        bo = np.linalg.lstsq(xb_, yb_, rcond=None)[0]
        bs["ols"][i] = bo[1]
        lg_ = sm.Logit((yib > 0).astype(float), xb_).fit(disp=0)
        ps = yib > 0
        gg_ = sm.GLM(yib[ps], xb_[ps], family=fam).fit()
        q1 = 1 / (1 + np.exp(-X1[ix] @ lg_.params)) * np.exp(X1[ix] @ gg_.params)
        q0 = 1 / (1 + np.exp(-X0[ix] @ lg_.params)) * np.exp(X0[ix] @ gg_.params)
        bs["tp"][i] = q1.mean() - q0.mean()
    R["glm"]["ci"] = boot_ci(bs["glm"])
    R["glm"]["predA_ci"], R["glm"]["predB_ci"] = boot_ci(bs["predA"]), boot_ci(bs["predB"])
    R["glm"]["ratio_boot"] = boot_ci(bs["glm_ratio"])
    R["glm"]["boot_se"] = float(bs["glm"].std(ddof=1))
    R["logols"]["naive_ci"] = boot_ci(bs["naive"])
    R["logols"]["smear_ci"] = boot_ci(bs["smear"])
    R["logols"]["grp_ci"] = boot_ci(bs["grp"])
    R["ols"]["boot_ci"] = boot_ci(bs["ols"])
    R["tp"]["ci"] = boot_ci(bs["tp"])
    R["B"] = Bn
    N["reg"] = R
    print(json.dumps({k: v for k, v in R.items() if k != "B"}, indent=1))

    # --- 본문 코드 상자: 감마 GLM (실제 출력)
    code_c = '''
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
df = pd.read_csv("gen/_ch21_cost.csv")
fam = sm.families.Gamma(link=sm.families.links.Log())
glm = smf.glm("cost ~ A + age + male + cci + stage4", data=df, family=fam).fit()
print(glm.summary().tables[1])
print(np.exp(glm.params["A"]).round(3), np.exp(glm.conf_int().loc["A"]).round(3).tolist())
mA = glm.predict(df.assign(A=1)).mean()      # 3,000명 모두 A를 썼다고 놓고 예측한 평균
mB = glm.predict(df.assign(A=0)).mean()      # 3,000명 모두 B를 썼다고 놓고 예측한 평균
print(round(mA, 1), round(mB, 1), round(mA - mB, 1))
'''
    buf = io.StringIO()
    sys.stdout = buf
    os.chdir(os.path.join(HERE, ".."))
    exec(code_c, {})
    sys.stdout = so
    out_txt["glm"] = buf.getvalue()
    print(out_txt["glm"])
    code_d = '''
import numpy as np, pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
df = pd.read_csv("gen/_ch21_cost.csv")
df["anyinp"] = (df["inp"] > 0).astype(int)
rhs = "A + age + male + cci + stage4"
part1 = smf.logit("anyinp ~ " + rhs, data=df).fit(disp=0)
part2 = smf.glm("inp ~ " + rhs, data=df[df["inp"] > 0],
                family=sm.families.Gamma(link=sm.families.links.Log())).fit()
print(np.exp(part1.params["A"]).round(3), np.exp(part2.params["A"]).round(3))
dA, dB = df.assign(A=1), df.assign(A=0)
eA = (part1.predict(dA) * part2.predict(dA)).mean()
eB = (part1.predict(dB) * part2.predict(dB)).mean()
print(round(eA, 1), round(eB, 1), round(eA - eB, 1))
'''
    buf = io.StringIO()
    sys.stdout = buf
    exec(code_d, {})
    sys.stdout = so
    os.chdir(cwd)
    out_txt["tp"] = buf.getvalue()
    print(out_txt["tp"])
    N["out"] = out_txt

    # ================================================================== 스스로 확인하기용 숫자
    hdr("연습문제")
    # 나-1: 예산. 다음 해 B군 환자 800명
    N["q_budget"] = dict(n=800, mean=G["B"]["mean"], med=G["B"]["med"], by_mean=G["B"]["mean"] * 800, by_med=G["B"]["med"] * 800)
    N["q_budget"]["gap"] = N["q_budget"]["by_mean"] - N["q_budget"]["by_med"]
    # 다-계산: 이단계 모형 손계산 (가상의 다른 연구)
    qt = dict(pA=0.30, mA=820.0, pB=0.38, mB=790.0)
    qt.update(eA=qt["pA"] * qt["mA"], eB=qt["pB"] * qt["mB"])
    qt["diff"] = qt["eA"] - qt["eB"]
    N["q_tp"] = qt
    # 다-논문 읽기: 비 1.xx와 기준 평균으로 금액 가늠
    N["q_ratio"] = dict(ratio=R["glm"]["ratio"], predB=gB, approx=gB * (R["glm"]["ratio"] - 1))
    print(N["q_budget"], qt, N["q_ratio"])

    with open(os.path.join(HERE, "_ch21_nums.json"), "w", encoding="utf-8") as fo:
        json.dump(N, fo, ensure_ascii=False, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
