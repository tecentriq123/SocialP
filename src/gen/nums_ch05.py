"""Chapter 5 numbers: every statistic quoted in content/ch05.html is computed here.
Run: python3 gen/nums_ch05.py
"""
import numpy as np
import scipy.stats as st
from scipy.special import comb
from scipy.optimize import brentq

np.set_printoptions(suppress=True, linewidth=120)


def hr(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def expected(tab):
    tab = np.asarray(tab, float)
    return np.outer(tab.sum(1), tab.sum(0)) / tab.sum()


def pearson(tab, yates=False):
    tab = np.asarray(tab, float)
    E = expected(tab)
    d = np.abs(tab - E)
    if yates:
        d = np.maximum(d - 0.5, 0)  # R: min(0.5, |O-E|)
    chi = ((d ** 2) / E).sum()
    df = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return chi, df, st.chi2.sf(chi, df)


def lr_g2(tab):
    tab = np.asarray(tab, float)
    E = expected(tab)
    m = tab > 0
    g = 2 * (tab[m] * np.log(tab[m] / E[m])).sum()
    df = (tab.shape[0] - 1) * (tab.shape[1] - 1)
    return g, st.chi2.sf(g, df)


def adj_resid(tab):
    tab = np.asarray(tab, float)
    N = tab.sum()
    E = expected(tab)
    rp = tab.sum(1, keepdims=True) / N
    cp = tab.sum(0, keepdims=True) / N
    return (tab - E) / np.sqrt(E * (1 - rp) * (1 - cp))


def effect(a, b, c, d):
    """rows: exposed (a events, b non), unexposed (c events, d non)"""
    n1, n0 = a + b, c + d
    p1, p0 = a / n1, c / n0
    rd = p1 - p0
    se_rd = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    rr = p1 / p0
    se_lrr = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n0)
    orr = a * d / (b * c)
    se_lor = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = 1.959964
    return dict(p1=p1, p0=p0, rd=rd, rd_ci=(rd - z * se_rd, rd + z * se_rd), se_rd=se_rd,
                rr=rr, rr_ci=(rr * np.exp(-z * se_lrr), rr * np.exp(z * se_lrr)), se_lrr=se_lrr,
                orr=orr, or_ci=(orr * np.exp(-z * se_lor), orr * np.exp(z * se_lor)), se_lor=se_lor)


# ---------- Fisher exact, replicating R's fisher.test (2x2) ----------
def fisher_R(tab):
    tab = np.asarray(tab, int)
    x = tab[0, 0]
    m = tab[:, 0].sum()
    n = tab[:, 1].sum()
    k = tab[0, :].sum()
    lo, hi = max(0, k - n), min(k, m)
    support = np.arange(lo, hi + 1)
    logdc = st.hypergeom.logpmf(support, m + n, m, k)

    def dnhyper(ncp):
        d = logdc + np.log(ncp) * support
        d = np.exp(d - d.max())
        return d / d.sum()

    def mnhyper(ncp):
        if ncp == 0:
            return lo
        if np.isinf(ncp):
            return hi
        return (support * dnhyper(ncp)).sum()

    def pnhyper(q, ncp, upper=False):
        if ncp == 1:
            return st.hypergeom.sf(x - 1, m + n, m, k) if upper else st.hypergeom.cdf(x, m + n, m, k)
        d = dnhyper(ncp)
        return d[support >= q].sum() if upper else d[support <= q].sum()

    d1 = dnhyper(1.0)
    pv = d1[d1 <= d1[x - lo] * (1 + 1e-7)].sum()
    p_less = pnhyper(x, 1.0)
    p_greater = pnhyper(x, 1.0, upper=True)

    eps = np.finfo(float).eps

    def ncpU(alpha):
        if x == hi:
            return np.inf
        p = pnhyper(x, 1)
        if p < alpha:
            return brentq(lambda t: pnhyper(x, t) - alpha, 1e-12, 1, xtol=1e-12)
        elif p > alpha:
            return 1 / brentq(lambda t: pnhyper(x, 1 / t) - alpha, eps, 1, xtol=1e-14)
        return 1

    def ncpL(alpha):
        if x == lo:
            return 0
        p = pnhyper(x, 1, upper=True)
        if p > alpha:
            return brentq(lambda t: pnhyper(x, t, upper=True) - alpha, 1e-12, 1, xtol=1e-14)
        elif p < alpha:
            return 1 / brentq(lambda t: pnhyper(x, 1 / t, upper=True) - alpha, eps, 1, xtol=1e-14)
        return 1

    def mle():
        if x == lo:
            return 0
        if x == hi:
            return np.inf
        mu = mnhyper(1)
        if mu > x:
            return brentq(lambda t: mnhyper(t) - x, 1e-12, 1, xtol=1e-14)
        elif mu < x:
            return 1 / brentq(lambda t: mnhyper(1 / t) - x, eps, 1, xtol=1e-14)
        return 1

    return dict(p=pv, p_less=p_less, p_greater=p_greater, mle=mle(), ci=(ncpL(0.025), ncpU(0.025)),
                support=support, probs=d1)


def rfmt(v, digits=7):
    """mimic R print: signif to `digits`"""
    return f"{v:.{digits}g}"


# =====================================================================
hr("가. 2x2 main example: drug A 48/400 vs drug B 102/600, 1-year hospitalization")
T = np.array([[48, 352], [102, 498]])
N = T.sum()
E = expected(T)
print("observed\n", T, "\nrow totals", T.sum(1), "col totals", T.sum(0), "N", N)
print("expected\n", E)
print("O-E\n", T - E)
contrib = (T - E) ** 2 / E
print("contrib (O-E)^2/E\n", contrib.round(4), "sum", contrib.sum())
chi, df, p = pearson(T)
print(f"Pearson chi2 = {chi:.6f}, df = {df}, p = {p:.6f}")
chiY, _, pY = pearson(T, yates=True)
print(f"Yates chi2 = {chiY:.6f}, p = {pY:.6f}")
print("Yates contribs", ((np.abs(T - E) - 0.5) ** 2 / E).round(4))
g2, pg = lr_g2(T)
print(f"LR G2 = {g2:.6f}, p = {pg:.6f}")
fr = fisher_R(T)
print(f"Fisher 2-sided p = {fr['p']:.6f}, less = {fr['p_less']:.6f}, greater = {fr['p_greater']:.6f}")
print(f"Fisher cond MLE OR = {fr['mle']:.6f}, CI = {fr['ci']}")
lbl = chi * (N - 1) / N
print(f"Linear-by-linear (N-1)/N*chi2 = {lbl:.6f}, p = {st.chi2.sf(lbl, 1):.6f}")
print("scipy check", st.chi2_contingency(T, correction=False)[:2], st.chi2_contingency(T, correction=True)[:2],
      st.fisher_exact(T))
# z-test for two proportions
p1, p2 = 48 / 400, 102 / 600
pp = 150 / 1000
se = np.sqrt(pp * (1 - pp) * (1 / 400 + 1 / 600))
z = (p1 - p2) / se
print(f"z test: p1={p1}, p2={p2}, pooled={pp}, SE={se:.6f}, z={z:.6f}, z^2={z * z:.6f}, p={2 * st.norm.sf(abs(z)):.6f}")
print("phi", np.sqrt(chi / N))
print("critical chi2(1) .95", st.chi2.ppf(.95, 1))
ef = effect(48, 352, 102, 498)
for k_, v in ef.items():
    print(" ", k_, v)
print("NNT-ish 1/|RD|", 1 / abs(ef["rd"]))

hr("가. significance vs strength: same proportions, 10x sample")
T10 = T * 10
c10 = pearson(T10)
print("10x:", c10, effect(480, 3520, 1020, 4980)["rr"])
Tsmall = np.array([[12, 88], [17, 83]])
print("percent-as-counts table", pearson(Tsmall), "yates", pearson(Tsmall, True))
# small sample same pct: 40 vs 60? keep for reference
T4 = np.array([[12, 88], [17, 83]])

hr("가. OR vs RR as baseline risk rises")
for rr_ in (2.0, 0.5):
    for p0 in (0.01, 0.05, 0.10, 0.20, 0.30, 0.40):
        p1_ = rr_ * p0
        if p1_ >= 1:
            continue
        or_ = (p1_ / (1 - p1_)) / (p0 / (1 - p0))
        print(f"RR={rr_} p0={p0:.2f} p1={p1_:.2f} OR={or_:.3f}")
# common-outcome worked example: nonadherence 60% vs 40%
for (a, n1, c, n0) in [(60, 100, 40, 100), (30, 100, 15, 100), (24, 100, 12, 100)]:
    e = effect(a, n1 - a, c, n0 - c)
    print(f"{a}/{n1} vs {c}/{n0}: RR={e['rr']:.4f} OR={e['orr']:.4f}")
# Zhang-Yu
def zy(orr, p0):
    return orr / ((1 - p0) + p0 * orr)
print("ZhangYu OR=0.666 p0=0.17 ->", zy(ef['orr'], 102 / 600))
print("ZhangYu OR=3.5 p0=0.3 ->", zy(3.5, 0.3), "OR=2.25 p0=.1", zy(2.25, 0.1))

hr("가. 3x2 hypoglycemia by add-on class (SGLT2i, DPP-4i, SU)")
T3 = np.array([[12, 288], [15, 285], [33, 267]])
E3 = expected(T3)
print("expected\n", E3)
c3 = pearson(T3)
print("chi2", c3)
print("contrib\n", ((T3 - E3) ** 2 / E3).round(4))
R3 = adj_resid(T3)
print("adjusted std residuals\n", R3.round(4))
print("Pearson resid (O-E)/sqrt(E)\n", ((T3 - E3) / np.sqrt(E3)).round(4))
print("bonferroni z for 3 tests:", st.norm.ppf(1 - 0.05 / 6))
print("two-sided p of residuals", 2 * st.norm.sf(np.abs(R3[:, 0])))
print("LR G2", lr_g2(T3))
names = ["SGLT2i", "DPP-4i", "SU"]
for i in range(3):
    for j in range(i + 1, 3):
        sub = T3[[i, j]]
        c = pearson(sub)
        f = st.fisher_exact(sub)
        print(f"{names[i]} vs {names[j]}: chi2={c[0]:.4f} p={c[2]:.6f} bonf={min(1, 3 * c[2]):.6f}  fisher p={f[1]:.5f}")
print("props", T3[:, 0] / 300)

hr("가. Paper Table 2 (outcomes) rows, n=400 vs 600")
rows = {"All-cause hospitalization": (48, 102), "Emergency department visit": (64, 111),
        "Hyperkalemia requiring hospitalization": (2, 9), "All-cause death": (3, 6)}
for name, (a, c) in rows.items():
    t = np.array([[a, 400 - a], [c, 600 - c]])
    Ee = expected(t)
    nsmall = (Ee < 5).sum()
    ch = pearson(t)
    fp = fisher_R(t)["p"]
    print(f"{name}: {a} ({100 * a / 400:.1f}) vs {c} ({100 * c / 600:.1f}); E min={Ee.min():.2f}, cells<5={nsmall}; "
          f"chi2={ch[0]:.4f} p={ch[2]:.4f}; yates p={pearson(t, True)[2]:.4f}; fisher p={fp:.4f}")
    e = effect(a, 400 - a, c, 600 - c)
    print(f"    RR={e['rr']:.3f} ({e['rr_ci'][0]:.3f}-{e['rr_ci'][1]:.3f}), RD={e['rd'] * 100:.2f}")

# =====================================================================
hr("나. Fisher: drug X 1/15 vs drug Y 6/14 grade>=3 hepatotoxicity")
F = np.array([[1, 14], [6, 8]])
print("expected\n", expected(F))
fr = fisher_R(F)
tot = comb(29, 7, exact=True)
print("C(29,7) =", tot)
for a, pr in zip(fr["support"], fr["probs"]):
    b = 15 - a
    c = 7 - a
    d = 14 - c
    num = comb(15, a, exact=True) * comb(14, 7 - a, exact=True)
    print(f"a={a}: table [[{a},{b}],[{c},{d}]] C(15,{a})*C(14,{7 - a}) = {comb(15, a, exact=True)}*{comb(14, 7 - a, exact=True)} = {num}  P={pr:.6f}  {'<=obs' if pr <= fr['probs'][1] * (1 + 1e-7) else ''}")
print("sum probs", fr["probs"].sum())
print(f"two-sided p = {fr['p']:.6f}; one-sided less = {fr['p_less']:.6f}; greater = {fr['p_greater']:.6f}")
print("doubling one-sided:", 2 * fr['p_less'])
print(f"cond MLE OR = {fr['mle']:.7f}; exact CI = {fr['ci'][0]:.9f} {fr['ci'][1]:.9f}")
print("sample OR", (1 * 8) / (14 * 6))
e = effect(1, 14, 6, 8)
print("effects", e)
cp = pearson(F)
cy = pearson(F, True)
print("Pearson", cp, "Yates", cy, "N-1 chi2", cp[0] * 28 / 29, st.chi2.sf(cp[0] * 28 / 29, 1))
Pobs = fr["probs"][1]
print("P(obs) =", Pobs, "mid-p (two-sided: sum(P<Pobs)+0.5Pobs) =", fr['p'] - 0.5 * Pobs)
# actual size at alpha 0.05 given these margins
probs = fr["probs"]
pv_each = np.array([probs[probs <= q * (1 + 1e-7)].sum() for q in probs])
print("p-values per possible table", dict(zip(fr["support"], pv_each.round(5))))
print("actual type I error at .05:", probs[pv_each <= 0.05].sum())
# Also scipy
print("scipy fisher", st.fisher_exact(F), st.fisher_exact(F, alternative='less'))
print("RD", 1 / 15 - 6 / 14)

hr("나. Paper AE table, X n=15, Y n=14")
ae = {"Any grade >=3 adverse event": (6, 10), "Hepatotoxicity": (1, 6), "Hypertension": (3, 4),
      "Hand-foot skin reaction": (2, 1), "Diarrhea": (1, 2)}
for name, (a, c) in ae.items():
    t = np.array([[a, 15 - a], [c, 14 - c]])
    f = fisher_R(t)
    print(f"{name}: {a} ({100 * a / 15:.1f}) vs {c} ({100 * c / 14:.1f}); fisher p={f['p']:.4f}; minE={expected(t).min():.2f}; chi2 p={pearson(t)[2]:.4f}")
print("pct", 1 / 15, 6 / 14)

# =====================================================================
hr("다. PDC categories: <40, 40-79, >=80 ; hospitalization")


def trend_stats(events, ns, scores):
    events, ns, scores = map(lambda v: np.asarray(v, float), (events, ns, scores))
    tab = np.column_stack([events, ns - events])
    Ntot = ns.sum()
    # individual-level r between score and outcome
    s = np.repeat(scores, ns.astype(int))
    y = np.concatenate([np.r_[np.ones(int(e)), np.zeros(int(n - e))] for e, n in zip(events, ns)])
    r = np.corrcoef(s, y)[0, 1]
    M2 = (Ntot - 1) * r ** 2
    CA = Ntot * r ** 2
    ch = pearson(tab)
    return dict(r=r, M2=M2, pM2=st.chi2.sf(M2, 1), CA=CA, pCA=st.chi2.sf(CA, 1), pearson=ch,
                depart=ch[0] - CA, pdepart=st.chi2.sf(ch[0] - CA, len(ns) - 2), props=events / ns,
                lr=lr_g2(tab))


ev, ns = [30, 36, 40], [200, 300, 500]
t1 = trend_stats(ev, ns, [1, 2, 3])
for k_, v in t1.items():
    print(" ", k_, v)
tab = np.column_stack([ev, np.array(ns) - np.array(ev)])
print("expected\n", expected(tab))
print("contrib\n", ((tab - expected(tab)) ** 2 / expected(tab)).round(4))
print("adj resid\n", adj_resid(tab).round(3))
# manual r pieces
s = np.array([1, 2, 3.]); n_ = np.array(ns, float); e_ = np.array(ev, float)
sbar = (n_ * s).sum() / n_.sum()
pbar = e_.sum() / n_.sum()
print("mean score", sbar, "pbar", pbar)
print("mean score among events", (e_ * s).sum() / e_.sum(), "among non-events", ((n_ - e_) * s).sum() / (n_ - e_).sum())
cov = ((e_ * s).sum() / n_.sum()) - sbar * pbar
var_s = (n_ * s ** 2).sum() / n_.sum() - sbar ** 2
var_y = pbar * (1 - pbar)
print("cov", cov, "var_s", var_s, "sd_s", np.sqrt(var_s), "var_y", var_y, "sd_y", np.sqrt(var_y), "r", cov / np.sqrt(var_s * var_y))
# weighted slope of proportion on score
w = n_
b = ((w * (s - sbar) * (e_ / n_ - pbar)).sum()) / ((w * (s - sbar) ** 2).sum())
print("weighted LS slope (proportion per category)", b, "intercept", pbar - b * sbar)

print("\n-- midpoint scores 20, 60, 90")
t2 = trend_stats(ev, ns, [20, 60, 90])
print({k_: v for k_, v in t2.items() if k_ in ("r", "M2", "pM2", "CA", "pCA")})
print("\n-- scores 1,2,4 (unequal)")
t2b = trend_stats(ev, ns, [1, 2, 4])
print({k_: v for k_, v in t2b.items() if k_ in ("r", "M2", "pM2")})

print("\n-- half sample (100,150,250 ; 15,18,20)")
t3 = trend_stats([15, 18, 20], [100, 150, 250], [1, 2, 3])
for k_, v in t3.items():
    print(" ", k_, v)

print("\n-- inverted U (non-monotone) with same n: 200,300,500")
for evu in ([16, 45, 45], [16, 48, 42], [14, 45, 47], [16, 46, 44], [15, 46, 45]):
    tu = trend_stats(evu, ns, [1, 2, 3])
    print(evu, "props", tu["props"], "pearson", tu["pearson"][0], tu["pearson"][2], "M2", tu["M2"], tu["pM2"])

hr("다. SPSS-like output rows for PDC table")
print("Pearson", t1["pearson"], "LR", t1["lr"], "LbL", t1["M2"], t1["pM2"])
print("min expected", expected(tab).min())

hr("다. Paper table (P for trend) crude RR vs >=80% ref")
for e1, n1 in zip(ev, ns):
    ee = effect(e1, n1 - e1, 40, 460)
    print(f"{e1}/{n1} ({100 * e1 / n1:.1f}%) RR vs >=80%: {ee['rr']:.3f} ({ee['rr_ci'][0]:.3f}-{ee['rr_ci'][1]:.3f})")
print("CA p", t1["pCA"], "M2 p", t1["pM2"], "pearson p", t1["pearson"][2])

hr("다. 2x2 linear-by-linear equals (N-1)/N Pearson")
print(chi * 999 / 1000)

hr("추가: exact (Clopper-Pearson) CIs, AE-row RR CIs, misc")


def clopper(x, n, a=0.05):
    lo = 0 if x == 0 else st.beta.ppf(a / 2, x, n - x + 1)
    hi = 1 if x == n else st.beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


print("1/15", clopper(1, 15), "6/14", clopper(6, 14))
e = effect(3, 12, 4, 10)
print("hypertension RR", e["rr"], e["rr_ci"])
print("reference <40%: RRs", 0.08 / 0.15, 0.12 / 0.15)
print("chi2 crit df2", st.chi2.ppf(.95, 2), "df1", st.chi2.ppf(.95, 1))
print("hand-foot probs", st.hypergeom.pmf(np.arange(4), 29, 15, 3))
print("adj resid denom", np.sqrt(20 * (2 / 3) * (1 - 60 / 900)))
print("odds A", 48 / 352, "odds B", 102 / 498)
print("A->B Fisher one-sided x2", 2 * 0.018023)

# =====================================================================
hr("파이썬 출력 상자 (scipy/statsmodels; run with: source /home/claude/pylibs/env.sh)")
try:
    from scipy.stats.contingency import odds_ratio
    from statsmodels.stats.proportion import proportions_ztest
    from statsmodels.stats.contingency_tables import Table
    F = [[1, 14], [6, 8]]
    res = st.fisher_exact(F)
    print("fisher_exact: pvalue", res.pvalue, "statistic (sample OR)", res.statistic,
          "less", st.fisher_exact(F, alternative="less").pvalue)
    orc = odds_ratio(F)  # kind="conditional"
    print("odds_ratio conditional", orc.statistic, orc.confidence_interval(confidence_level=0.95))
    print("A/B table: sample OR", st.fisher_exact(T).statistic, "conditional", odds_ratio(T).statistic)
    print("chi2_contingency default (Yates)", st.chi2_contingency(T)[:2], "no correction",
          st.chi2_contingency(T, correction=False)[:2])
    print("proportions_ztest", proportions_ztest([48, 102], [400, 600]))
    tt = Table(np.column_stack([[30, 36, 40], [170, 264, 460]])).test_ordinal_association()
    print("test_ordinal_association z^2 =", tt.zscore ** 2, "p =", tt.pvalue)
    print("r x c fisher_exact (Monte Carlo default)", st.fisher_exact(T3))
except ImportError as e:
    print("  (skipped:", e, ")")
