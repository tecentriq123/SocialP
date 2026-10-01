"""Chapter 2 numbers: independent t-test (Student/Welch) and Mann-Whitney.
Run: python3 gen/nums_ch02.py   (prints every statistic quoted in content/ch02.html)
"""
import numpy as np, scipy.stats as st
from itertools import combinations


def exact_data(m, s, n, seed):
    """individual values with exactly mean m and SD s (used for figures / R output)"""
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return m + s * z


def student(m1, s1, n1, m2, s2, n2):
    df = n1 + n2 - 2
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / df)
    se = sp * np.sqrt(1 / n1 + 1 / n2)
    t = (m1 - m2) / se
    tc = st.t.ppf(.975, df)
    return dict(sp=sp, se=se, t=t, df=df, p=2 * st.t.sf(abs(t), df), tc=tc,
                lo=(m1 - m2) - tc * se, hi=(m1 - m2) + tc * se)


def welch(m1, s1, n1, m2, s2, n2):
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    se = np.sqrt(v1 + v2)
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    tc = st.t.ppf(.975, df)
    return dict(v1=v1, v2=v2, se=se, t=t, df=df, p=2 * st.t.sf(abs(t), df), tc=tc,
                lo=(m1 - m2) - tc * se, hi=(m1 - m2) + tc * se)


def show(tag, d):
    print(tag, {k: (round(float(v), 5) if not isinstance(v, int) else v) for k, v in d.items()})


print("=" * 20, "A. main example: pharmacist counseling vs usual care, HbA1c change (n=36/36)")
A = (-0.90, 0.80, 36, -0.40, 0.90, 36)
print("SE each mean", 0.80 / 6, 0.90 / 6, "squares", (0.80 / 6) ** 2, (0.90 / 6) ** 2)
show("student", student(*A))
show("welch", welch(*A))
sA = student(*A)
print("Cohen d", -0.5 / sA["sp"])
# per-group 95% CI (figure 2-2, paper figure)
for m, s, n in ((-0.90, 0.80, 36), (-0.40, 0.90, 36)):
    se = s / np.sqrt(n); tc = st.t.ppf(.975, n - 1)
    print(f"group m={m}: SE={se:.4f} tcrit={tc:.4f} 95%CI=({m - tc * se:.3f},{m + tc * se:.3f}) margin={tc * se:.4f}  SEbar=({m - se:.3f},{m + se:.3f}) SDbar=({m - s:.2f},{m + s:.2f})")
x1 = exact_data(-0.90, 0.80, 36, 11); x2 = exact_data(-0.40, 0.90, 36, 12)
print("raw check", x1.mean(), x1.std(ddof=1), x2.mean(), x2.std(ddof=1))
print("R welch", st.ttest_ind(x1, x2, equal_var=False), "student", st.ttest_ind(x1, x2))
print("Levene(median) A", st.levene(x1, x2, center="median"), "F", st.bartlett(x1, x2))
# overlap of the two 95% CIs
m1a, m2a = -0.90, -0.40
mg1 = st.t.ppf(.975, 35) * 0.80 / 6; mg2 = st.t.ppf(.975, 35) * 0.90 / 6
print("CI overlap: upper g1", m1a + mg1, "lower g2", m2a - mg2, "overlap", (m1a + mg1) - (m2a - mg2), "avg margin", (mg1 + mg2) / 2)
# SE bars just touching -> p ?
print("p when SE bars just touch (equal SE, large df): z=", 2 / np.sqrt(2), 2 * st.norm.sf(2 / np.sqrt(2)))
print("p when 95% CI just touch (equal SE): z=", 2 * 1.96 / np.sqrt(2), 2 * st.norm.sf(2 * 1.96 / np.sqrt(2)))

print("=" * 20, "B. unequal n and SD (n=40, SD 0.6 vs n=15, SD 1.2)")
B = (-0.90, 0.60, 40, -0.40, 1.20, 15)
show("student", student(*B))
show("welch", welch(*B))
F = 0.60 ** 2 / 1.20 ** 2
pF = 2 * min(st.f.cdf(F, 39, 14), st.f.sf(F, 39, 14))
print("F test var ratio", F, "df 39,14 p", pF)
b1 = exact_data(-0.90, 0.60, 40, 21); b2 = exact_data(-0.40, 1.20, 15, 22)
print("Levene(median) B", st.levene(b1, b2, center="median"), "Levene(mean)", st.levene(b1, b2, center="mean"))
print("sp^2 B", ((39 * .36 + 14 * 1.44) / 53), "weights", 39 / 53, 14 / 53)

print("=" * 20, "C. simulation: actual type I error (200,000 reps each)")
rng = np.random.default_rng(2024)
R = 200000


def sim(n1, s1, n2, s2):
    x = rng.normal(0, s1, (R, n1)); y = rng.normal(0, s2, (R, n2))
    m1, m2 = x.mean(1), y.mean(1); v1, v2 = x.var(1, ddof=1), y.var(1, ddof=1)
    df = n1 + n2 - 2; sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    ps = 2 * st.t.sf(abs((m1 - m2) / np.sqrt(sp2 * (1 / n1 + 1 / n2))), df)
    a, b = v1 / n1, v2 / n2
    dfw = (a + b) ** 2 / (a ** 2 / (n1 - 1) + b ** 2 / (n2 - 1))
    pw = 2 * st.t.sf(abs((m1 - m2) / np.sqrt(a + b)), dfw)
    return (ps < .05).mean(), (pw < .05).mean()


SIM = [(36, 1, 36, 1), (36, 1, 36, 2), (40, 1, 15, 1), (40, 1.2, 15, 0.6), (40, 0.6, 15, 1.2)]
SIMRES = [sim(*sc) for sc in SIM]
for sc, r in zip(SIM, SIMRES):
    print(sc, "student %.4f  welch %.4f" % r)
# power loss of Welch when variances are equal (scenario A-like, true diff 0.5, SD 0.85)
rngp = np.random.default_rng(5)
x = rngp.normal(-0.9, 0.85, (R, 36)); y = rngp.normal(-0.4, 0.85, (R, 36))
print("power equal var: student", (st.ttest_ind(x, y, axis=1).pvalue < .05).mean(),
      "welch", (st.ttest_ind(x, y, axis=1, equal_var=False).pvalue < .05).mean())

print("=" * 20, "D. Table 2 (paper box), Welch for each row, n=36/36")
rows = [("HbA1c", -0.90, 0.80, -0.40, 0.90), ("FPG", -18.4, 30.2, -6.1, 33.5),
        ("PDC", 86.2, 11.5, 78.4, 16.9), ("SBP", -4.1, 12.8, -2.6, 13.5)]
for nm, a, sa, b, sb in rows:
    w = welch(a, sa, 36, b, sb, 36); s_ = student(a, sa, 36, b, sb, 36)
    print(nm, "diff", round(a - b, 2), "welch CI (%.2f, %.2f) p=%.4f t=%.3f df=%.1f" % (w["lo"], w["hi"], w["p"], w["t"], w["df"]),
          "| student p=%.4f" % s_["p"], "d=%.2f" % ((a - b) / s_["sp"]))
print("PDC: 100-mean =", 100 - 86.2, " 2SD =", 2 * 11.5, " usual:", 100 - 78.4, 2 * 16.9)

print("=" * 20, "E. Mann-Whitney hand example (opioid MME 0-72h), protocol n=8 vs usual n=7")
P = np.array([12, 15, 18, 20, 22, 26, 31, 68.])
U = np.array([24, 29, 36, 41, 45, 58, 115.])
allv = np.concatenate([P, U]); r = st.rankdata(allv)
print("ranks P", r[:8], "R_P", r[:8].sum(), "ranks U", r[8:], "R_U", r[8:].sum())
U_P = r[:8].sum() - 8 * 9 / 2; U_U = r[8:].sum() - 7 * 8 / 2
print("U_P", U_P, "U_U", U_U, "sum", U_P + U_U, "P(P>U)", U_P / 56, "P(U>P)", U_U / 56)
# pair count check
cnt = sum((p > u) + 0.5 * (p == u) for p in P for u in U); print("pair count P>U", cnt)
print("per protocol patient: # usual below", [int((U < p).sum()) for p in P])
print("medians", np.median(P), np.median(U), "means", P.mean(), U.mean(), "SD", P.std(ddof=1), U.std(ddof=1))
print("IQR (type7)", np.percentile(P, [25, 75]), np.percentile(U, [25, 75]))
print("exact", st.mannwhitneyu(P, U, method="exact"))
mu, sd = 56 / 2, np.sqrt(8 * 7 * 16 / 12)
z = (U_P - mu) / sd; zc = (U_P + 0.5 - mu) / sd
print("E[U]", mu, "SD", sd, "z", z, "p", 2 * st.norm.sf(abs(z)), "z_cc", zc, "p_cc", 2 * st.norm.sf(abs(zc)))
print("r = |z|/sqrt(N)", abs(z) / np.sqrt(15))
print("t-test", st.ttest_ind(P, U), "welch", st.ttest_ind(P, U, equal_var=False))

# exact null distribution of U by enumeration of rank subsets (C(15,8)=6435)
Ns = 15
cnts = np.zeros(57, int)
for comb in combinations(range(1, Ns + 1), 8):
    cnts[sum(comb) - 36] += 1
print("number of arrangements", cnts.sum(), "P(U<=9)", cnts[:10].sum(), cnts[:10].sum() / cnts.sum(),
      "two-sided", 2 * cnts[:10].sum() / cnts.sum())
np.save  # noqa
UNULL = cnts

# Hodges-Lehmann and exact CI the way R's wilcox.test(conf.int=TRUE) does it
diffs = np.sort(np.subtract.outer(P, U).ravel())
cdf = np.cumsum(cnts) / cnts.sum()


def qwilcox(pr):
    return int(np.argmax(cdf >= pr - 1e-12))


qu = qwilcox(0.025)
if qu == 0:
    qu = 1
ql = 56 - qu
print("HL", np.median(diffs), "qu", qu, "CI", diffs[qu - 1], diffs[ql], "achieved conf", 1 - 2 * cdf[qu - 1])
print("sorted diffs", diffs.astype(int).tolist())

print("=" * 20, "F. ties: 5-point satisfaction, counseling n=20 vs usual n=20")
fc = np.array([0, 2, 5, 8, 5]); fu = np.array([2, 5, 7, 4, 2])
tot = fc + fu; ends = np.cumsum(tot); starts = ends - tot + 1
mid = (starts + ends) / 2
print("totals", tot, "midranks", mid)
Rc = (fc * mid).sum(); Uc = Rc - 20 * 21 / 2
print("R_c", Rc, "U_c", Uc, "U_u", 400 - Uc, "U/n1n2", Uc / 400)
tt = tot[tot > 0]
v0 = 20 * 20 * 41 / 12
v1 = 20 * 20 / 12 * (41 - (tt ** 3 - tt).sum() / (40 * 39))
print("sum t^3-t", (tt ** 3 - tt).sum(), "var no-tie", v0, np.sqrt(v0), "var tie", v1, np.sqrt(v1))
for v in (v0, v1):
    z = (Uc - 200) / np.sqrt(v); print("z", z, "p", 2 * st.norm.sf(abs(z)), "z_cc", (Uc - 200 - 0.5) / np.sqrt(v), 2 * st.norm.sf(abs((Uc - 200 - .5) / np.sqrt(v))))
xc = np.repeat(np.arange(1, 6), fc); xu = np.repeat(np.arange(1, 6), fu)
print("scipy MW", st.mannwhitneyu(xc, xu), st.mannwhitneyu(xc, xu, use_continuity=False))
print("medians", np.median(xc), np.median(xu), "IQR", np.percentile(xc, [25, 75]), np.percentile(xu, [25, 75]))
p_gt = sum((a > b) for a in xc for b in xu) / 400; p_eq = sum((a == b) for a in xc for b in xu) / 400
print("P(c>u)", p_gt, "P(tie)", p_eq, "P(c<u)", 1 - p_gt - p_eq)

print("=" * 20, "G. 파이썬 출력 상자 (scipy 1.17, pandas 3.0; run with source /home/claude/pylibs/env.sh)")
import pandas as pd
# 가 절: Welch t 검정. x1, x2 = exact_data (평균·SD가 표와 정확히 같은 개인 값)
tdf = pd.DataFrame({"group": ["counseling"] * 36 + ["usual"] * 36, "hba1c_chg": np.r_[x1, x2]})
print(tdf.groupby("group")["hba1c_chg"].agg(["count", "mean", "std"]))
counseling = tdf.loc[tdf["group"] == "counseling", "hba1c_chg"]
usual = tdf.loc[tdf["group"] == "usual", "hba1c_chg"]
res = st.ttest_ind(counseling, usual, equal_var=False)
print(repr(res)); print(repr(res.confidence_interval()))
res_s = st.ttest_ind(counseling, usual)          # 기본값 equal_var=True -> Student
print(repr(res_s)); print(repr(res_s.confidence_interval()))
# 나 절: Mann-Whitney + Hodges-Lehmann
protocol = np.array([12, 15, 18, 20, 22, 26, 31, 68])
usual_mme = np.array([24, 29, 36, 41, 45, 58, 115])
print(repr(st.mannwhitneyu(protocol, usual_mme)))
print(repr(st.mannwhitneyu(usual_mme, protocol)))
pdiffs = np.subtract.outer(protocol, usual_mme).ravel()
print(repr(np.median(pdiffs)))
print(repr(np.sort(pdiffs)[[10, 45]]))
print("P(U<=10)", cnts[:11].sum() / cnts.sum(), "P(U<=9)", cnts[:10].sum() / cnts.sum(), "qu", qu)
print("satisfaction: default", st.mannwhitneyu(xc, xu), "exact(no tie corr)", st.mannwhitneyu(xc, xu, method="exact"))
