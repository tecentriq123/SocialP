"""Every number quoted in 부록 A c–f 절 (content/ap01.html; once 14장 아·자·차·카 절) is computed and printed here.
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch14c.py
fig_ch14c.py imports the objects defined here.
"""
import math
import numpy as np
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from statsmodels.stats.inter_rater import cohens_kappa
from statsmodels.stats.contingency_tables import mcnemar

Z975 = st.norm.ppf(0.975)
Z80 = st.norm.ppf(0.80)
Z90 = st.norm.ppf(0.90)


def hdr(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


# ======================================================================
# 아. 의료비용 자료의 분석
# pragmatic RCT, 65세 이상 다제약물 환자 1,000명, 약사 주도 약물검토(I) vs 통상관리(C)
# 결과: 12개월 입원 진료비 (만원)
# ======================================================================
hdr("아. 의료비용")
N_COST = 500
_r = np.random.default_rng(12)
COST = {}
for g, p, mu in (("C", 0.40, 700), ("I", 0.32, 680)):
    anyh = _r.random(N_COST) < p
    pos = _r.lognormal(np.log(mu) - 0.5, 1.0, N_COST)
    COST[g] = np.round(np.where(anyh, pos, 0.0), 1)
cC, cI = COST["C"], COST["I"]

CS = {}
for g, y in (("C", cC), ("I", cI)):
    pos = y[y > 0]
    srt = np.sort(y)[::-1]
    q1, med, q3 = np.percentile(y, [25, 50, 75])
    CS[g] = dict(n=len(y), npos=int((y > 0).sum()), ppos=(y > 0).mean(), mean=y.mean(), sd=y.std(ddof=1),
                 q1=q1, med=med, q3=q3, pos_mean=pos.mean(), pos_med=np.median(pos), pos_sd=pos.std(ddof=1),
                 max=y.max(), top10=srt[: len(y) // 10].sum() / y.sum(), total=y.sum())
    print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in CS[g].items()})

tt = st.ttest_ind(cI, cC, equal_var=False)
ttci = tt.confidence_interval()
mw = st.mannwhitneyu(cI, cC)
DIFF = cI.mean() - cC.mean()
RATIO = cI.mean() / cC.mean()
print(f"mean diff I-C = {DIFF:.2f}; Welch t = {tt.statistic:.3f}, df = {tt.df:.1f}, p = {tt.pvalue:.4f}, "
      f"95% CI {ttci.low:.1f} to {ttci.high:.1f}")
print(f"Mann-Whitney p = {mw.pvalue:.4f}")

# percentile bootstrap with scipy (same tool as 21장): each group resampled separately, 9,999 times
B = 9999


def _mdiff(x, y, axis=-1):
    return x.mean(axis=axis) - y.mean(axis=axis)


def _mratio(x, y, axis=-1):
    return x.mean(axis=axis) / y.mean(axis=axis)


_bs_d = st.bootstrap((cI, cC), _mdiff, n_resamples=B, method="percentile", rng=np.random.default_rng(2026))
_bs_r = st.bootstrap((cI, cC), _mratio, n_resamples=B, method="percentile", rng=np.random.default_rng(2026))
_bs_bca = st.bootstrap((cI, cC), _mdiff, n_resamples=B, rng=np.random.default_rng(2026))  # scipy default method = BCa
BOOT_D = np.array([_bs_d.confidence_interval.low, _bs_d.confidence_interval.high])
BOOT_R = np.array([_bs_r.confidence_interval.low, _bs_r.confidence_interval.high])
BOOT_BCA = np.array([_bs_bca.confidence_interval.low, _bs_bca.confidence_interval.high])
BOOT_SE = _bs_d.standard_error
print(f"scipy bootstrap percentile (B={B}, rng 2026) diff 95% CI {BOOT_D[0]:.1f} to {BOOT_D[1]:.1f} (SE {BOOT_SE:.1f}); "
      f"ratio {RATIO:.3f} ({BOOT_R[0]:.2f}-{BOOT_R[1]:.2f})")
print(f"  scipy default (BCa) diff 95% CI {BOOT_BCA[0]:.1f} to {BOOT_BCA[1]:.1f}")
print(f"  Student t (ttest_ind default equal_var=True) CI: {st.ttest_ind(cI, cC).confidence_interval().low:.2f} "
      f"to {st.ttest_ind(cI, cC).confidence_interval().high:.2f}")
print(f"  zeros: C {int((cC == 0).sum())}, I {int((cI == 0).sum())}, total {int((cC == 0).sum() + (cI == 0).sum())}")

# two-part decomposition
pc, pi = CS["C"]["ppos"], CS["I"]["ppos"]
mc, mi = CS["C"]["pos_mean"], CS["I"]["pos_mean"]
TP = dict(pc=pc, pi=pi, mc=mc, mi=mi, rr=pi / pc, orr=(pi / (1 - pi)) / (pc / (1 - pc)),
          cond_ratio=mi / mc, freq=(pi - pc) * mc, sev=pi * (mi - mc))
print(f"P(any): C {pc:.3f}, I {pi:.3f}; RR {TP['rr']:.3f}; OR {TP['orr']:.3f}")
print(f"cond mean: C {mc:.1f}, I {mi:.1f}; ratio {TP['cond_ratio']:.3f}")
print(f"check mean = p x cond: C {pc * mc:.2f} (={cC.mean():.2f}), I {pi * mi:.2f} (={cI.mean():.2f})")
print(f"decomposition: frequency {(pi - pc) * mc:.2f} + severity {pi * (mi - mc):.2f} = {TP['freq'] + TP['sev']:.2f}")

yall = np.r_[cC, cI]
xall = np.r_[np.zeros(N_COST), np.ones(N_COST)]
X = sm.add_constant(xall)
lg = sm.Logit((yall > 0).astype(float), X).fit(disp=0)
TP["or_ci"] = np.exp(lg.conf_int()[1])
TP["or_p"] = lg.pvalues[1]
print(f"logit OR {np.exp(lg.params[1]):.3f} ({TP['or_ci'][0]:.2f}-{TP['or_ci'][1]:.2f}), p = {lg.pvalues[1]:.4f}")
pos = yall > 0
gm = sm.GLM(yall[pos], X[pos], family=sm.families.Gamma(sm.families.links.Log())).fit()
TP["g_ratio"] = np.exp(gm.params[1])
TP["g_ci"] = np.exp(gm.conf_int()[1])
TP["g_p"] = gm.pvalues[1]
print(f"gamma GLM (positive costs) ratio {TP['g_ratio']:.3f} ({TP['g_ci'][0]:.2f}-{TP['g_ci'][1]:.2f}), p = {gm.pvalues[1]:.3f}")

# arbitrary constant in log(y + c)
LOGC = {}
for cst in (1, 100):
    ol = sm.OLS(np.log(yall + cst), X).fit()
    LOGC[cst] = np.exp(ol.params[1])
    print(f"OLS on log(cost + {cst}) : exp(b) = {LOGC[cst]:.3f}")
# histogram data for the figure
HIST_EDGES = np.arange(0, 3100, 100)


# ======================================================================
# 자. 결측자료와 다중대체
# RCT 300명(150:150), 약사 주도 혈압관리 vs 통상치료, 결과: 12개월 SBP, 보정: 기저 SBP
# 12개월 결측은 3개월 SBP(관측됨)가 높을수록, 통상치료군에서 더 많음 (MAR)
# ======================================================================
hdr("자. 결측자료")


def mi_gen(rng, n=300):
    trt = np.r_[np.ones(n // 2), np.zeros(n // 2)]
    s0 = rng.normal(152, 12, n)
    s3 = 20 + 0.85 * s0 - 4 * trt + rng.normal(0, 9, n)
    s12 = 15 + 0.35 * s0 + 0.55 * s3 - 3 * trt + rng.normal(0, 7, n)
    lp = -2.1 + 0.13 * (s3 - 146) + 0.8 * (1 - trt)
    miss = rng.random(n) < 1 / (1 + np.exp(-lp))
    return trt, s0, s3, s12, miss


def ancova(y, trt, s0):
    r = sm.OLS(y, sm.add_constant(np.c_[trt, s0])).fit()
    return r.params[1], r.bse[1], r.df_resid


def impute_once(rng, y, miss, Xi):
    """one proper imputation from a Bayesian normal linear regression (parameter draw + residual draw)"""
    Xo, yo = Xi[~miss], y[~miss]
    V = np.linalg.inv(Xo.T @ Xo)
    bh = V @ Xo.T @ yo
    res = yo - Xo @ bh
    df = len(yo) - Xo.shape[1]
    s2 = res @ res / rng.chisquare(df)
    b = rng.multivariate_normal(bh, s2 * V)
    out = y.copy()
    out[miss] = Xi[miss] @ b + rng.normal(0, math.sqrt(s2), miss.sum())
    return out


def rubin(q, u):
    q, u = np.asarray(q), np.asarray(u)
    m = len(q)
    Q = q.mean()
    W = np.mean(u ** 2)
    Bv = q.var(ddof=1)
    T = W + (1 + 1 / m) * Bv
    nu = (m - 1) * (1 + W / ((1 + 1 / m) * Bv)) ** 2
    tcrit = st.t.ppf(0.975, nu)
    return dict(m=m, Q=Q, W=W, B=Bv, T=T, se=math.sqrt(T), nu=nu, tcrit=tcrit,
                lo=Q - tcrit * math.sqrt(T), hi=Q + tcrit * math.sqrt(T), fmi=(1 + 1 / m) * Bv / T)


def mi_run(data, rng, Xi, m, keep=None):
    trt, s0, s3, s12, miss = data
    q, u = [], []
    for _ in range(m):
        yy = impute_once(rng, s12, miss, Xi)
        if keep is not None:
            keep.append(yy)
        a, b_, _ = ancova(yy, trt, s0)
        q.append(a)
        u.append(b_)
    return q, u


MI_DATA = mi_gen(np.random.default_rng(129))
trt, s0, s3, s12, miss = MI_DATA
NM = dict(n=len(trt), miss_I=int(miss[trt == 1].sum()), miss_C=int(miss[trt == 0].sum()), miss=int(miss.sum()))
print(NM, f"overall {miss.mean():.3f}; I {miss[trt == 1].mean():.3f}; C {miss[trt == 0].mean():.3f}")
# 3-month SBP by dropout status
S3TAB = {}
for g, lab in ((1, "I"), (0, "C")):
    k = trt == g
    S3TAB[lab] = dict(comp=s3[k & ~miss].mean(), drop=s3[k & miss].mean(), s0_comp=s0[k & ~miss].mean(),
                      s0_drop=s0[k & miss].mean(), s12_comp=s12[k & ~miss].mean(), s12_drop_true=s12[k & miss].mean(),
                      s12_all_true=s12[k].mean())
    print(lab, {kk: round(v, 1) for kk, v in S3TAB[lab].items()})

RES = {}
RES["full"] = ancova(s12, trt, s0)
RES["cc"] = ancova(s12[~miss], trt[~miss], s0[~miss])
ymean = s12.copy()
for g in (0, 1):
    k = trt == g
    ymean[k & miss] = s12[k & ~miss].mean()
RES["mean"] = ancova(ymean, trt, s0)
M_MI = 30
_rmi = np.random.default_rng(7)
Q_NOAUX, U_NOAUX = mi_run(MI_DATA, _rmi, sm.add_constant(np.c_[trt, s0]), M_MI)
IMP_AUX = []
Q_AUX, U_AUX = mi_run(MI_DATA, _rmi, sm.add_constant(np.c_[trt, s0, s3]), M_MI, keep=IMP_AUX)
RUB_NOAUX = rubin(Q_NOAUX, U_NOAUX)
RUB_AUX = rubin(Q_AUX, U_AUX)
RUB5 = rubin(Q_AUX[:5], U_AUX[:5])


def ci_t(est, se, df):
    t = st.t.ppf(0.975, df)
    return est - t * se, est + t * se


MI_TAB = []
for key, lab in (("full", "결측 없는 전체 자료"), ("cc", "완전사례 분석"), ("mean", "군별 평균으로 단일 대체")):
    e, se, df = RES[key]
    lo, hi = ci_t(e, se, df)
    MI_TAB.append(dict(key=key, lab=lab, est=e, se=se, lo=lo, hi=hi, n=(len(trt) if key != "cc" else int((~miss).sum()))))
for key, lab, R in (("noaux", "다중대체 (군, 기저 SBP)", RUB_NOAUX), ("aux", "다중대체 (군, 기저 SBP, 3개월 SBP)", RUB_AUX)):
    MI_TAB.append(dict(key=key, lab=lab, est=R["Q"], se=R["se"], lo=R["lo"], hi=R["hi"], n=len(trt)))
for r in MI_TAB:
    print(f"{r['lab']:<34} n={r['n']:<4} est {r['est']:.2f} SE {r['se']:.2f} 95% CI {r['lo']:.2f} to {r['hi']:.2f}")
print("RUB_AUX", {k: round(v, 4) for k, v in RUB_AUX.items()})
print("RUB_NOAUX", {k: round(v, 4) for k, v in RUB_NOAUX.items()})
print("first 5 imputations (aux):", [f"{q:.2f} (SE {u:.2f})" for q, u in zip(Q_AUX[:5], U_AUX[:5])])
print("RUB5", {k: round(v, 4) for k, v in RUB5.items()})
print(f"  mean of SE^2 = {RUB5['W']:.4f}; var of estimates = {RUB5['B']:.4f}; "
      f"T = {RUB5['W']:.4f} + 1.2 x {RUB5['B']:.4f} = {RUB5['T']:.4f}; SE = {RUB5['se']:.3f}")
print(f"  naive CI using mean SE only: {RUB5['Q'] - 1.96 * math.sqrt(RUB5['W']):.2f} to {RUB5['Q'] + 1.96 * math.sqrt(RUB5['W']):.2f}")
p_mi = 2 * st.t.sf(abs(RUB_AUX["Q"] / RUB_AUX["se"]), RUB_AUX["nu"])
p_cc = 2 * st.t.sf(abs(RES["cc"][0] / RES["cc"][1]), RES["cc"][2])
print(f"p MI(aux) = {p_mi:.2g}; p CC = {p_cc:.2g}")

# ---- 결측 기전만 바꿔 400번 반복: 탈락이 무엇에 달렸느냐에 따라 완전사례와 다중대체가 어떻게 달라지나
# 자료를 만드는 식은 mi_gen과 같고, 탈락 확률의 선형예측만 바꾼다. 참값(전체 자료 기대값)은 -3 - 0.55 x 4 = -5.2.
# 기울기는 원래 설계(3개월 SBP 1 mmHg당 0.13)를 각 변수의 표준편차 단위로 맞췄다.
SD_S3 = math.sqrt((0.85 * 12) ** 2 + 9 ** 2)
SD_S12 = math.sqrt(((0.35 + 0.55 * 0.85) * 12) ** 2 + (0.55 * 9) ** 2 + 7 ** 2)
BETA_SD = 0.13 * SD_S3


def mech_miss(rng, mech, trt, s0, s3, s12):
    if mech == "mcar":
        lp = np.full(len(trt), -1.0)
    elif mech == "model":
        lp = -1.9 + BETA_SD * (s0 - 152) / 12 + 0.8 * (1 - trt)
    elif mech == "s3":
        lp = -2.1 + 0.13 * (s3 - 146) + 0.8 * (1 - trt)
    else:
        lp = -1.9 + BETA_SD * (s12 - 147.7) / SD_S12 + 0.8 * (1 - trt)
    return rng.random(len(trt)) < 1 / (1 + np.exp(-lp))


MECH_LAB = {"mcar": "무관(MCAR)", "model": "배정군과 기저 SBP", "s3": "3개월 SBP", "s12": "12개월 SBP 자체"}
MECH = {}
_rmech = np.random.default_rng(5)
NMECH = 400
for mech in MECH_LAB:
    acc = {k: [] for k in ("miss", "full", "cc", "mi")}
    for _ in range(NMECH):
        t_ = np.r_[np.ones(150), np.zeros(150)]
        a0 = _rmech.normal(152, 12, 300)
        a3 = 20 + 0.85 * a0 - 4 * t_ + _rmech.normal(0, 9, 300)
        y12 = 15 + 0.35 * a0 + 0.55 * a3 - 3 * t_ + _rmech.normal(0, 7, 300)
        ms = mech_miss(_rmech, mech, t_, a0, a3, y12)
        acc["miss"].append(ms.mean())
        acc["full"].append(ancova(y12, t_, a0)[0])
        acc["cc"].append(ancova(y12[~ms], t_[~ms], a0[~ms])[0])
        q, u = mi_run((t_, a0, a3, y12, ms), _rmech, sm.add_constant(np.c_[t_, a0, a3]), 10)
        acc["mi"].append(np.mean(q))
    MECH[mech] = {k: float(np.mean(v)) for k, v in acc.items()}
    print(f"mechanism {MECH_LAB[mech]:<14} miss {MECH[mech]['miss'] * 100:.1f}%  full {MECH[mech]['full']:.2f}  "
          f"CC {MECH[mech]['cc']:.2f}  MI(+3m) {MECH[mech]['mi']:.2f}")

# ---- delta 조정(티핑 포인트): 중재군 탈락자의 대체값에 delta를 더한 뒤 같은 30벌을 Rubin의 규칙으로 합친다
DELTA = {}
for dl in np.arange(0, 25.01, 0.5):
    q, u = [], []
    for yy in IMP_AUX:
        y2 = yy.copy()
        y2[miss & (trt == 1)] += dl
        a, b_, _ = ancova(y2, trt, s0)
        q.append(a)
        u.append(b_)
    DELTA[float(dl)] = rubin(q, u)
for dl in (0.0, 5.0, 10.0, 15.0, 20.0):
    R = DELTA[dl]
    print(f"delta {dl:>4.0f}: est {R['Q']:.2f}  95% CI {R['lo']:.2f} to {R['hi']:.2f}")
TIP = min(d for d, R in DELTA.items() if R["hi"] >= 0)
print(f"tipping point (first delta on 0.5 grid where upper CI >= 0): {TIP}")

# repeated simulation (averages)
_rs = np.random.default_rng(1)
SIMR = {k: [] for k in ("full", "cc", "mean", "noaux", "aux")}
SIMSE = {k: [] for k in SIMR}
NSIM = 300
for _ in range(NSIM):
    d = mi_gen(_rs)
    t_, a0, a3, y12, ms = d
    for key, val in (("full", ancova(y12, t_, a0)), ("cc", ancova(y12[~ms], t_[~ms], a0[~ms]))):
        SIMR[key].append(val[0])
        SIMSE[key].append(val[1])
    ym = y12.copy()
    for g in (0, 1):
        k = t_ == g
        ym[k & ms] = y12[k & ~ms].mean()
    v = ancova(ym, t_, a0)
    SIMR["mean"].append(v[0])
    SIMSE["mean"].append(v[1])
    for key, Xi in (("noaux", sm.add_constant(np.c_[t_, a0])), ("aux", sm.add_constant(np.c_[t_, a0, a3]))):
        q, u = mi_run(d, _rs, Xi, 10)
        R = rubin(q, u)
        SIMR[key].append(R["Q"])
        SIMSE[key].append(R["se"])
SIMAVG = {k: (np.mean(SIMR[k]), np.std(SIMR[k]), np.mean(SIMSE[k])) for k in SIMR}
for k, v in SIMAVG.items():
    print(f"sim {k:<6} mean est {v[0]:.2f}  empirical SD {v[1]:.2f}  mean reported SE {v[2]:.2f}")


# ======================================================================
# 차. 표본크기와 검정력
# ======================================================================
hdr("차. 표본크기")
zsum2 = (Z975 + Z80) ** 2
n_mean = 2 * zsum2 * 7 ** 2 / 4 ** 2
n_mean_t = TTestIndPower().solve_power(effect_size=4 / 7, alpha=0.05, power=0.8)
for lab, sd_, dl, zb in (("기본", 7, 4, Z80), ("SD 8", 8, 4, Z80), ("차이 3", 7, 3, Z80), ("검정력 90%", 7, 4, Z90), ("차이 2 (절반)", 7, 2, Z80)):
    nn = 2 * (Z975 + zb) ** 2 * sd_ ** 2 / dl ** 2
    print(f"  ingredient {lab}: {nn:.2f} -> {math.ceil(nn)}")
print(f"two means: z-formula {n_mean:.2f} -> {math.ceil(n_mean)}; t-based (statsmodels) {n_mean_t:.2f} -> {math.ceil(n_mean_t)}; "
      f"dropout10% {math.ceil(n_mean) / 0.9:.1f} -> {math.ceil(math.ceil(n_mean) / 0.9)}")


def n_two_prop(p1, p2, za=Z975, zb=Z80):
    pb = (p1 + p2) / 2
    num = (za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return num / (p1 - p2) ** 2


PC, PI = 0.50, 0.65
n_prop = n_two_prop(PC, PI)
n_prop90 = n_two_prop(PC, PI, zb=Z90)
h = proportion_effectsize(PI, PC)
n_prop_sm = NormalIndPower().solve_power(effect_size=h, alpha=0.05, power=0.8)
pb = (PC + PI) / 2
print(f"two props {PC}->{PI}: pbar {pb}, sqrt(2pq) {math.sqrt(2 * pb * (1 - pb)):.4f}, sqrt(p1q1+p2q2) {math.sqrt(PC * (1 - PC) + PI * (1 - PI)):.4f}")
print(f"  n per group {n_prop:.2f} -> {math.ceil(n_prop)}; 90% power {n_prop90:.2f} -> {math.ceil(n_prop90)}; "
      f"Cohen h {h:.4f}, statsmodels NormalIndPower {n_prop_sm:.2f} -> {math.ceil(n_prop_sm)}")
N_PROP = math.ceil(n_prop)
n_cc = n_prop / 4 * (1 + math.sqrt(1 + 4 / (n_prop * abs(PI - PC)))) ** 2
print(f"  with continuity correction (Fleiss): {n_cc:.2f} -> {math.ceil(n_cc)}")
print(f"  170 x 1.15 = {170 * 1.15:.1f}; 196 x 0.85 = {196 * 0.85:.1f}")
print(f"  dropout 15%: {N_PROP}/0.85 = {N_PROP / 0.85:.1f} -> 200 per group")
for d in (0.10, 0.075):
    nn = n_two_prop(0.5, 0.5 + d)
    print(f"  difference {d * 100:.1f}%p: n per group {nn:.1f} -> {math.ceil(nn)}")
# power actually achieved if true difference is 10%p with 170 per group
def power_two_prop(p1, p2, n):
    pb = (p1 + p2) / 2
    se0 = math.sqrt(2 * pb * (1 - pb) / n)
    se1 = math.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return st.norm.cdf((abs(p2 - p1) - Z975 * se0) / se1)
print(f"  power with 170/group if true 60% vs 50%: {power_two_prop(0.5, 0.6, 170):.3f}")
# ICC design effect
DE = 1 + (12.5 - 1) * 0.05
print(f"design effect {DE:.3f}; {N_PROP} x DE = {N_PROP * DE:.1f} -> {math.ceil(N_PROP * DE)}")

# survival: Schoenfeld
def events(hr, p=0.5, zb=Z80):
    return (Z975 + zb) ** 2 / (p * (1 - p) * math.log(hr) ** 2)
D75 = events(0.75)
D80 = events(0.80)
D75_90 = events(0.75, zb=Z90)
S0_3y = 0.70
p_ev_c = 1 - S0_3y
p_ev_i = 1 - S0_3y ** 0.75
p_ev = (p_ev_c + p_ev_i) / 2
print(f"events HR0.75 {D75:.1f} -> {math.ceil(D75)}; HR0.80 {D80:.1f} -> {math.ceil(D80)}; HR0.75 90% {D75_90:.1f} -> {math.ceil(D75_90)}")
print(f"3y event prob C {p_ev_c:.3f}, I = 1-0.7^0.75 = {p_ev_i:.4f}, avg {p_ev:.4f}; patients {math.ceil(D75) / p_ev:.1f}")
# post-hoc (observed) power as a function of p
OBS_POW = {}
for p in (0.05, 0.20, 0.50):
    z = st.norm.ppf(1 - p / 2)
    OBS_POW[p] = st.norm.cdf(z - Z975) + st.norm.cdf(-z - Z975)
    print(f"observed power at p={p}: {OBS_POW[p]:.3f}")
# ---- 발생률(인년) 비교: 두 군의 관찰 인년이 같을 때 필요한 총 사건 수
def events_rate(rr, zb=Z80):
    return (Z975 + zb) ** 2 * (1 + rr) ** 2 / (rr * math.log(rr) ** 2)


D_RATE75 = events_rate(0.75)
PY_C = D_RATE75 / (1 + 0.75) / 0.10
print(f"rate ratio 0.75: total events {D_RATE75:.1f}; control events {D_RATE75 / 1.75:.1f}; "
      f"person-years per group at 0.10/PY = {PY_C:.0f}")
# 사건이 0건일 때 발생률 상한(정확 포아송 95% 양측 상한 = chi2(0.975, 2)/2)
UP0 = st.chi2.ppf(0.975, 2) / 2
print(f"zero events: upper limit = {UP0:.3f} / person-years; 2,300 PY -> {UP0 / 2300 * 1000:.2f} per 1,000 PY; "
      f"to bound rate below 1/1,000 PY need {UP0 * 1000:.0f} PY")
# ---- 같은 사람을 전후로 재는 설계: 차이 4, SD 7, 전후 상관 rho
from statsmodels.stats.power import TTestPower
PREPOST = {}
for rho in (0.3, 0.5, 0.7):
    sd_ch = 7 * math.sqrt(2 * (1 - rho))
    n1 = TTestPower().solve_power(effect_size=4 / sd_ch, alpha=0.05, power=0.8)
    n_anc = n_mean * (1 - rho ** 2)
    PREPOST[rho] = dict(sd=sd_ch, n1=n1, n_anc=n_anc)
    print(f"rho {rho}: SD of change {sd_ch:.2f}; one-group pre-post n {n1:.1f} -> {math.ceil(n1)}; "
          f"two-group ANCOVA per group {n_mean:.1f} x (1 - rho^2) = {n_anc:.1f} -> {math.ceil(n_anc)}")
# curve for figure: n per group vs difference, p_C = 0.5
CURVE_D = np.linspace(0.05, 0.25, 81)
CURVE80 = np.array([n_two_prop(0.5, 0.5 + d) for d in CURVE_D])
CURVE90 = np.array([n_two_prop(0.5, 0.5 + d, zb=Z90) for d in CURVE_D])
for d in (0.05, 0.10, 0.15, 0.20):
    print(f"  curve d={d:.2f}: 80% {n_two_prop(0.5, 0.5 + d):.1f}, 90% {n_two_prop(0.5, 0.5 + d, zb=Z90):.1f}")


# ======================================================================
# 카. 일치도 분석
# ======================================================================
hdr("카. 일치도")
_ra = np.random.default_rng(31)
N_BP = 50
true = _ra.normal(138, 17, N_BP)
REF = np.round(true + _ra.normal(0, 3, N_BP))
NEW = np.round(true + 3 + _ra.normal(0, 6, N_BP))
dif = NEW - REF
avg = (NEW + REF) / 2
BA = dict(mean=dif.mean(), sd=dif.std(ddof=1), n=N_BP)
BA["lo"] = BA["mean"] - 1.96 * BA["sd"]
BA["hi"] = BA["mean"] + 1.96 * BA["sd"]
tcrit = st.t.ppf(0.975, N_BP - 1)
BA["m_lo"] = BA["mean"] - tcrit * BA["sd"] / math.sqrt(N_BP)
BA["m_hi"] = BA["mean"] + tcrit * BA["sd"] / math.sqrt(N_BP)
BA["r"], BA["r_p"] = st.pearsonr(NEW, REF)
BA["within5"] = np.mean(np.abs(dif) <= 5)
BA["outside"] = int(np.sum((dif < BA["lo"]) | (dif > BA["hi"])))
BA["ref_mean"], BA["new_mean"] = REF.mean(), NEW.mean()
BA["ref_sd"], BA["new_sd"] = REF.std(ddof=1), NEW.std(ddof=1)
BA["ref_min"], BA["ref_max"] = REF.min(), REF.max()
# 일치 한계의 95% 신뢰구간 (근사: SE = sqrt(3 s^2 / n), t(n-1))
BA["loa_se"] = math.sqrt(3 * BA["sd"] ** 2 / N_BP)
BA["lo_ci"] = (BA["lo"] - tcrit * BA["loa_se"], BA["lo"] + tcrit * BA["loa_se"])
BA["hi_ci"] = (BA["hi"] - tcrit * BA["loa_se"], BA["hi"] + tcrit * BA["loa_se"])
# statsmodels mean_diff_plot draws limits with np.std (ddof=0)
BA["lo_ddof0"] = BA["mean"] - 1.96 * dif.std()
BA["hi_ddof0"] = BA["mean"] + 1.96 * dif.std()
# trend of difference vs average
sl = st.linregress(avg, dif)
BA["slope"], BA["slope_p"] = sl.slope, sl.pvalue
# paired t (systematic bias)
pt = st.ttest_rel(NEW, REF)
BA["t"], BA["p"] = pt.statistic, pt.pvalue
# r within restricted range
k = (avg >= 130) & (avg <= 150)
BA["r_narrow"] = st.pearsonr(NEW[k], REF[k])[0]
BA["n_narrow"] = int(k.sum())
BA["sd_narrow"] = dif[k].std(ddof=1)
BA["mean_narrow"] = dif[k].mean()


def icc2(Y):
    """Y: n x k. returns ICC(A,1) absolute agreement and ICC(C,1) consistency (two-way model)"""
    n, kk = Y.shape
    gm_ = Y.mean()
    msr = kk * np.sum((Y.mean(1) - gm_) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(0) - gm_) ** 2) / (kk - 1)
    sse = np.sum((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm_) ** 2)
    mse = sse / ((n - 1) * (kk - 1))
    icc_a = (msr - mse) / (msr + (kk - 1) * mse + kk * (msc - mse) / n)
    icc_c = (msr - mse) / (msr + (kk - 1) * mse)
    return icc_a, icc_c


BA["icc_a"], BA["icc_c"] = icc2(np.c_[REF, NEW])
# wrist device shifted up by a further 7 mmHg: r and ICC(C) same, ICC(A) drops
BA["icc_a10"], BA["icc_c10"] = icc2(np.c_[REF, NEW + 7])
BA["r10"] = st.pearsonr(NEW + 7, REF)[0]
for kk_, v in BA.items():
    print(f"{kk_:<10} {v:.4f}" if isinstance(v, (float, np.floating)) else f"{kk_:<10} {v}")

# kappa: 자기보고 순응 vs PDC >= 80%, 200명
#            PDC>=80  PDC<80
# SR adh       110      50
# SR non        10      30
KT = np.array([[110, 50], [10, 30]])
n_k = KT.sum()
po = np.trace(KT) / n_k
row = KT.sum(1) / n_k
col = KT.sum(0) / n_k
pe = float(row @ col)
kap = (po - pe) / (1 - pe)
ck = cohens_kappa(KT)
mc_ = mcnemar(KT, exact=True)
mc_chi = mcnemar(KT, exact=False, correction=False)
KAP = dict(po=po, pe=pe, kappa=kap, se=ck.std_kappa, lo=ck.kappa_low, hi=ck.kappa_upp,
           row=row, col=col, mc_p=mc_.pvalue, mc_chi=mc_chi.statistic, mc_chi_p=mc_chi.pvalue)
print(f"kappa table {KT.tolist()}: po {po:.3f}, row {row}, col {col}, pe {pe:.3f}, kappa {kap:.4f}")
print(f"statsmodels kappa {ck.kappa:.4f}, SE {ck.std_kappa:.4f}, 95% CI {ck.kappa_low:.3f}-{ck.kappa_upp:.3f}")
print(f"McNemar exact p = {mc_.pvalue:.2e}; chi2 = {mc_chi.statistic:.2f}, p = {mc_chi.pvalue:.2e}")
print(f"expected agreement cells: both adh {row[0] * col[0] * n_k:.1f}, both non {row[1] * col[1] * n_k:.1f}")
# low prevalence paradox example: same po 0.90 but different kappa
for tab in ([[85, 5], [5, 5]], [[45, 5], [5, 45]]):
    tt_ = np.array(tab, float)
    ck2 = cohens_kappa(tt_)
    print(f"paradox table {tab}: po {np.trace(tt_) / tt_.sum():.2f}, kappa {ck2.kappa:.3f}")
