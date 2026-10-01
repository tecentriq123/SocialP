"""Chapter 3 numbers: paired t-test and Wilcoxon signed rank test.
Run: python3 gen/nums_ch03.py   (prints every statistic quoted in content/ch03.html)
"""
import numpy as np, scipy.stats as st
from itertools import product

print("=" * 20, "A. paired SBP example (home BP education), 12 patients")
before = np.array([131, 146, 144, 154, 164, 159, 155, 164, 149, 161, 155, 142.])
after = np.array([126, 144, 131, 147, 162, 150, 154, 147, 142, 154, 139, 144.])
d = after - before
n = len(d)
print("d", d.astype(int).tolist(), "sum", d.sum())
mb, ma, md = before.mean(), after.mean(), d.mean()
sb, sa, sd = before.std(ddof=1), after.std(ddof=1), d.std(ddof=1)
r = np.corrcoef(before, after)[0, 1]
print(f"before {mb:.4f} ± {sb:.4f}; after {ma:.4f} ± {sa:.4f}; d {md:.4f} ± {sd:.4f}; r={r:.4f}")
print("sum (d-mean)^2", ((d - md) ** 2).sum(), "var d", sd ** 2)
se = sd / np.sqrt(n); t = md / se; tc = st.t.ppf(.975, n - 1); p = 2 * st.t.sf(abs(t), n - 1)
print(f"SE={se:.4f} t={t:.4f} df={n-1} p={p:.6f} tcrit={tc:.4f} CI=({md - tc*se:.4f}, {md + tc*se:.4f})")
print("scipy", st.ttest_rel(after, before))
# independent analysis of the same data
sp = np.sqrt((sb ** 2 + sa ** 2) / 2); sei = sp * np.sqrt(2 / n); ti = md / sei; pi = 2 * st.t.sf(abs(ti), 2 * n - 2)
tci = st.t.ppf(.975, 2 * n - 2)
print(f"independent: sp={sp:.4f} SE={sei:.4f} t={ti:.4f} df={2*n-2} p={pi:.4f} CI=({md - tci*sei:.3f},{md + tci*sei:.3f})")
print("scipy ind", st.ttest_ind(after, before))
# variance identity
vid = sb ** 2 + sa ** 2 - 2 * r * sb * sa
print("Var identity:", sb ** 2, sa ** 2, 2 * r * sb * sa, "=", vid, "sqrt", np.sqrt(vid))
print("if r=0: SD_d=", np.sqrt(sb ** 2 + sa ** 2))
print("ratio of SEs", sei / se)
# normality of differences (small n)
print("Shapiro on d", st.shapiro(d))
print("patients with decrease", (d < 0).sum(), "increase", (d > 0).sum())

print("=" * 20, "B. SD of difference vs rho (sigma1=sigma2=10)")
for rho in (0, 0.5, 0.8, 0.82, 0.9):
    print(rho, np.sqrt(200 * (1 - rho)))

print("=" * 20, "C. pre-post table: intervention n=30 vs control n=30")
def within(m, s, nn):
    se_ = s / np.sqrt(nn); t_ = m / se_; tc_ = st.t.ppf(.975, nn - 1)
    return se_, t_, 2 * st.t.sf(abs(t_), nn - 1), m - tc_ * se_, m + tc_ * se_
for lab, m, s in (("int", -7.0, 11.0), ("ctl", -3.5, 10.5)):
    se_, t_, p_, lo, hi = within(m, s, 30)
    print(lab, f"SE={se_:.3f} t={t_:.3f} p={p_:.5f} CI=({lo:.2f},{hi:.2f})")
v1, v2 = 11.0 ** 2 / 30, 10.5 ** 2 / 30
seb = np.sqrt(v1 + v2); tb = -3.5 / seb; dfb = (v1 + v2) ** 2 / (v1 ** 2 / 29 + v2 ** 2 / 29)
pb = 2 * st.t.sf(abs(tb), dfb); tcb = st.t.ppf(.975, dfb)
print(f"between: diff=-3.5 SE={seb:.4f} t={tb:.4f} df={dfb:.2f} p={pb:.4f} CI=({-3.5 - tcb*seb:.2f},{-3.5 + tcb*seb:.2f})")
# implied correlations
for lab, s0, s1, sc in (("int", 10.1, 11.8, 11.0), ("ctl", 9.7, 10.9, 10.5)):
    print(lab, "implied rho", (s0 ** 2 + s1 ** 2 - sc ** 2) / (2 * s0 * s1))
print("follow-up diff", 145.4 - 148.3, "baseline diff", 152.4 - 151.8, "change diff", -7.0 - (-3.5))
print("check means: int 152.4-7.0=", 152.4 - 7.0, " ctl 151.8-3.5=", 151.8 - 3.5)
# what if SD of change assumed from r=0 or r=0.5
for rr in (0.0, 0.5, 0.8):
    print("r", rr, "SD change int", np.sqrt(10.1 ** 2 + 11.8 ** 2 - 2 * rr * 10.1 * 11.8))

print("=" * 20, "D. regression to the mean")
mu, sb_, se_m = 140.0, 12.0, 8.0
sobs = np.sqrt(sb_ ** 2 + se_m ** 2); rel = sb_ ** 2 / sobs ** 2
c = 160.0
z = (c - mu) / sobs; lam = st.norm.pdf(z) / st.norm.sf(z)
e1 = mu + sobs * lam; e2 = mu + rel * (e1 - mu)
print(f"obs SD={sobs:.3f} reliability={rel:.4f} z={z:.4f} P(select)={st.norm.sf(z):.4f} E[x1|sel]={e1:.2f} E[x2|sel]={e2:.2f} drop={e1-e2:.2f}")
rng = np.random.default_rng(314)
N = 200000
tru = rng.normal(mu, sb_, N); x1 = tru + rng.normal(0, se_m, N); x2 = tru + rng.normal(0, se_m, N)
sel = x1 >= c
print("sim: frac", sel.mean(), "mean x1", x1[sel].mean(), "mean x2", x2[sel].mean(), "drop", (x1[sel] - x2[sel]).mean())
# figure subsample
rng2 = np.random.default_rng(512)  # same sample as figure 3-3
M = 600
tr = rng2.normal(mu, sb_, M); a1 = tr + rng2.normal(0, se_m, M); a2 = tr + rng2.normal(0, se_m, M)
s2 = a1 >= c
print("fig sample: n sel", s2.sum(), "mean x1", a1[s2].mean(), "mean x2", a2[s2].mean())

print("=" * 20, "E. Wilcoxon signed rank: deprescribing, 14 patients")
b = np.array([12, 10, 14, 9, 11, 13, 10, 12, 15, 11, 9, 13, 10, 12])
a = np.array([9, 9, 11, 9, 10, 10, 8, 12, 11, 9, 10, 10, 12, 9])
dd = a - b
print("d", dd.tolist())
nz = dd[dd != 0]
rk = st.rankdata(np.abs(nz))
full_rank = np.full(len(dd), np.nan); full_rank[dd != 0] = rk
print("ranks by patient", full_rank.tolist())
Wp = rk[nz > 0].sum(); Wm = rk[nz < 0].sum(); m = len(nz)
print("n nonzero", m, "W+", Wp, "W-", Wm, "total", m * (m + 1) / 2)
vals, tcount = np.unique(np.abs(nz), return_counts=True)
print("ties", dict(zip(vals.tolist(), tcount.tolist())))
EW = m * (m + 1) / 4
V0 = m * (m + 1) * (2 * m + 1) / 24
corr = ((tcount ** 3 - tcount).sum()) / 48
V = V0 - corr
print("E", EW, "V0", V0, "tie corr", corr, "V", V, "SD", np.sqrt(V))
zz = (Wp - EW) / np.sqrt(V); zc = (Wp - EW + 0.5) / np.sqrt(V)
print("z", zz, "p", 2 * st.norm.sf(abs(zz)), "z_cc", zc, "p_cc", 2 * st.norm.sf(abs(zc)))
print("scipy wilcox approx cc", st.wilcoxon(dd, zero_method="wilcox", correction=True, method="approx"))
print("scipy wilcox approx", st.wilcoxon(dd, zero_method="wilcox", correction=False, method="approx"))
# exact conditional permutation distribution with midranks
dist = {}
for signs in product((0, 1), repeat=m):
    w = float(np.dot(signs, rk))
    dist[w] = dist.get(w, 0) + 1
tot = 2 ** m
le = sum(c_ for w, c_ in dist.items() if w <= Wp)
print("perm: total", tot, "P(W+<=obs)", le, le / tot, "two-sided", 2 * le / tot)
PERM = dist
# sign test
k_pos = int((nz > 0).sum()); k_neg = int((nz < 0).sum())
print("sign test", k_pos, k_neg, st.binomtest(k_pos, m, 0.5))
# paired t for reference
print("paired t", st.ttest_rel(a, b), "mean d", dd.mean(), "sd", dd.std(ddof=1))
print("medians before", np.median(b), "after", np.median(a), "median d", np.median(dd))
print("IQR before", np.percentile(b, [25, 75]), "after", np.percentile(a, [25, 75]), "d", np.percentile(dd, [25, 75]))
print("means", b.mean(), a.mean())
# Walsh averages pseudo-median (all d incl zeros, as R does after removing zeros? R removes zeros for estimate too)
w_all = [(nz[i] + nz[j]) / 2 for i in range(m) for j in range(i, m)]
print("pseudo-median (nonzero d)", np.median(w_all))
# exact null distribution when no ties (n=12), for reference
cnt = np.zeros(79, int)
for signs in product((0, 1), repeat=12):
    cnt[int(np.dot(signs, np.arange(1, 13)))] += 1
print("no-tie exact P(W+<=7)*2", 2 * cnt[:8].sum() / 4096)

print("=" * 20, "F. 파이썬 출력 상자 (scipy 1.17; run with source /home/claude/pylibs/env.sh)")
# 가 절: 대응표본 t 검정
res = st.ttest_rel(after, before)
print(repr(res)); print(repr(res.confidence_interval()))
print(repr(np.mean(after - before)))
print("one-sample", st.ttest_1samp(after - before, 0), "independent (wrong)", st.ttest_ind(after, before))
# 나 절: Wilcoxon 부호순위 검정 (0 포함 14쌍 -> scipy 1.17 method='auto'는 정규근사)
print(repr(st.wilcoxon(a, b)))                                   # 기본값: correction=False
print(repr(st.wilcoxon(a, b, correction=True)))                  # 연속성 보정
print(repr(st.wilcoxon(a, b, method=st.PermutationMethod(n_resamples=np.inf))))  # 조건부 정확 p
print("swapped", st.wilcoxon(b, a), "zstat", st.wilcoxon(a, b, method="asymptotic").zstatistic,
      st.wilcoxon(a, b, method="asymptotic", correction=True).zstatistic)
print("pratt", st.wilcoxon(a, b, zero_method="pratt"), "nonzero only (auto -> permutation)", st.wilcoxon(nz))
