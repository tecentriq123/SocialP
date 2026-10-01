"""Chapter 4 numbers: every statistic quoted in content/ch04.html is computed here.

Running example: 6-month PDC (%) in three pharmacist-intervention arms
(usual care / brief counselling / intensive counselling + phone follow-up), n = 30 each.
Tiny example: pilot study, 5 patients per arm (hand calculation in 다, 라, 마).
Run: python3 gen/nums_ch04.py
"""
import itertools
import numpy as np
import scipy.stats as st

ARMS = ["Usual care", "Brief counselling", "Intensive counselling"]
KO = ["일반 복약지도", "1회 상담", "집중 상담"]


def running_data():
    """Return list of 3 arrays (usual, brief, intensive), PDC % rounded to 0.1, capped at 100."""
    rng = np.random.default_rng(33)
    base = np.array([70.0, 76.0, 79.0])
    mu0 = base.mean()
    c = 0.965
    out = []
    e = [rng.normal(0, 13.0, 30) for _ in range(3)]
    for i in range(3):
        y = np.round(np.clip(e[i] - e[i].mean() + (mu0 + c * (base[i] - mu0)), 0, 100), 1)
        out.append(y)
    return out


def knowledge_scores():
    """Secondary outcome for the Table in 가: medication knowledge score 0-10 (integers)."""
    rng = np.random.default_rng(493)
    out = []
    for m in (6.0, 7.0, 7.3):
        out.append(np.clip(np.round(rng.normal(m, 1.8, 30)), 0, 10))
    return out


TINY = [np.array([58, 63, 67, 69, 73.]), np.array([62, 68, 71, 77, 82.]), np.array([70, 74, 79, 81, 86.])]


# ---------------------------------------------------------------- helpers
def anova(groups):
    n = np.array([len(g) for g in groups]); N = n.sum(); k = len(groups)
    allv = np.concatenate(groups); gm = allv.mean()
    means = np.array([g.mean() for g in groups])
    ssb = (n * (means - gm) ** 2).sum()
    ssw = sum(((g - g.mean()) ** 2).sum() for g in groups)
    sst = ((allv - gm) ** 2).sum()
    dfb, dfw = k - 1, N - k
    msb, msw = ssb / dfb, ssw / dfw
    F = msb / msw
    p = st.f.sf(F, dfb, dfw)
    eta2 = ssb / sst
    omega2 = (ssb - dfb * msw) / (sst + msw)
    return dict(n=n, N=N, k=k, gm=gm, means=means, ssb=ssb, ssw=ssw, sst=sst, dfb=dfb, dfw=dfw,
                msb=msb, msw=msw, F=F, p=p, eta2=eta2, omega2=omega2)


def welch_anova(groups):
    n = np.array([len(g) for g in groups], float); k = len(groups)
    m = np.array([g.mean() for g in groups]); v = np.array([g.var(ddof=1) for g in groups])
    w = n / v; W = w.sum(); mw = (w * m).sum() / W
    A = (w * (m - mw) ** 2).sum() / (k - 1)
    tmp = (((1 - w / W) ** 2) / (n - 1)).sum()
    B = 1 + 2 * (k - 2) / (k ** 2 - 1) * tmp
    F = A / B; df2 = (k ** 2 - 1) / (3 * tmp)
    return F, k - 1, df2, st.f.sf(F, k - 1, df2)


def pairs(k):
    return [(i, j) for i in range(k) for j in range(i + 1, k)]


def pairwise_all(groups, a=None):
    a = a or anova(groups)
    k, dfw, msw = a["k"], a["dfw"], a["msw"]
    res = []
    tcrit = st.t.ppf(0.975, dfw)
    qcrit = st.studentized_range.ppf(0.95, k, dfw)
    for i, j in pairs(k):
        gi, gj = groups[i], groups[j]
        ni, nj = len(gi), len(gj)
        d = gj.mean() - gi.mean()
        se = np.sqrt(msw * (1 / ni + 1 / nj))
        t = d / se
        p_unadj = 2 * st.t.sf(abs(t), dfw)
        q = abs(d) / np.sqrt(msw / 2 * (1 / ni + 1 / nj))
        p_tukey = st.studentized_range.sf(q, k, dfw)
        hw_tukey = qcrit * np.sqrt(msw / 2 * (1 / ni + 1 / nj))
        Fs = t ** 2 / (k - 1)
        p_scheffe = st.f.sf(Fs, k - 1, dfw)
        # Games-Howell
        vi, vj = gi.var(ddof=1) / ni, gj.var(ddof=1) / nj
        se_gh = np.sqrt(vi + vj)
        df_gh = (vi + vj) ** 2 / (vi ** 2 / (ni - 1) + vj ** 2 / (nj - 1))
        p_gh = st.studentized_range.sf(abs(d) / se_gh * np.sqrt(2), k, df_gh)
        res.append(dict(i=i, j=j, diff=d, se=se, t=t, p=p_unadj, lo=d - tcrit * se, hi=d + tcrit * se,
                        q=q, p_tukey=p_tukey, tlo=d - hw_tukey, thi=d + hw_tukey, p_scheffe=p_scheffe,
                        p_gh=p_gh, df_gh=df_gh))
    ps = np.array([r["p"] for r in res]); m = len(ps)
    bon = np.minimum(ps * m, 1)
    o = np.argsort(ps)
    holm_sorted = np.minimum(np.maximum.accumulate(ps[o] * (m - np.arange(m))), 1)
    holm = np.empty(m); holm[o] = holm_sorted
    for r, b, h in zip(res, bon, holm):
        r["p_bon"], r["p_holm"] = b, h
    return res, tcrit, qcrit


def kw(groups):
    n = np.array([len(g) for g in groups]); N = n.sum()
    allv = np.concatenate(groups)
    r = st.rankdata(allv)
    idx = np.cumsum(np.r_[0, n])
    R = np.array([r[idx[i]:idx[i + 1]].sum() for i in range(len(groups))])
    H = 12 / (N * (N + 1)) * (R ** 2 / n).sum() - 3 * (N + 1)
    _, t = np.unique(allv, return_counts=True)
    C = 1 - (t ** 3 - t).sum() / (N ** 3 - N)
    Hc = H / C
    return dict(R=R, mean_rank=R / n, H=H, C=C, Hc=Hc, p=st.chi2.sf(Hc, len(groups) - 1), ranks=r, ties=t[t > 1])


def dunn(groups):
    k = kw(groups)
    n = np.array([len(g) for g in groups]); N = n.sum()
    allv = np.concatenate(groups); _, t = np.unique(allv, return_counts=True)
    s2 = N * (N + 1) / 12 - (t ** 3 - t).sum() / (12 * (N - 1))
    res = []
    for i, j in pairs(len(groups)):
        z = (k["mean_rank"][j] - k["mean_rank"][i]) / np.sqrt(s2 * (1 / n[i] + 1 / n[j]))
        res.append(dict(i=i, j=j, z=z, p=2 * st.norm.sf(abs(z))))
    ps = np.array([r["p"] for r in res]); m = len(ps)
    o = np.argsort(ps)
    hs = np.minimum(np.maximum.accumulate(ps[o] * (m - np.arange(m))), 1)
    holm = np.empty(m); holm[o] = hs
    for r, h in zip(res, holm):
        r["p_holm"], r["p_bon"] = h, min(r["p"] * m, 1)
    return res


def jt(groups):
    """Jonckheere-Terpstra: J = sum over ordered pairs i<j of #(x_i < x_j) + 0.5 #(ties). Tie-corrected variance."""
    J = 0.0; U = {}
    for i, j in pairs(len(groups)):
        a = groups[i][:, None]; b = groups[j][None, :]
        u = (a < b).sum() + 0.5 * (a == b).sum()
        U[(i, j)] = u; J += u
    n = np.array([len(g) for g in groups]); N = n.sum()
    allv = np.concatenate(groups); _, t = np.unique(allv, return_counts=True)
    E = (N ** 2 - (n ** 2).sum()) / 4
    V = ((N * (N - 1) * (2 * N + 5) - (n * (n - 1) * (2 * n + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72
         + ((n * (n - 1) * (n - 2)).sum() * (t * (t - 1) * (t - 2)).sum()) / (36 * N * (N - 1) * (N - 2))
         + ((n * (n - 1)).sum() * (t * (t - 1)).sum()) / (8 * N * (N - 1)))
    z = (J - E) / np.sqrt(V)
    return dict(U=U, J=J, E=E, V=V, sd=np.sqrt(V), z=z, p1=st.norm.sf(z), p2=2 * st.norm.sf(abs(z)))


# ---------------------------------------------------------------- vectorized simulations
def _ranks(M):
    """row-wise ranks (continuous data, no ties)."""
    return np.argsort(np.argsort(M, axis=1), axis=1) + 1.0


def sim_tests(groups):
    """groups: list of arrays shaped (R, n_i). Returns p-values (R,) for ANOVA, Welch, KW, JT(two-sided)."""
    n = np.array([g.shape[1] for g in groups], float); k = len(groups); N = n.sum()
    m = np.stack([g.mean(1) for g in groups], 1)                       # (R,k)
    v = np.stack([g.var(1, ddof=1) for g in groups], 1)
    allv = np.concatenate(groups, 1)
    gm = allv.mean(1, keepdims=True)
    ssb = (n * (m - gm) ** 2).sum(1); ssw = ((n - 1) * v).sum(1)
    F = (ssb / (k - 1)) / (ssw / (N - k)); p_an = st.f.sf(F, k - 1, N - k)
    w = n / v; W = w.sum(1, keepdims=True); mw = (w * m).sum(1, keepdims=True) / W
    A = (w * (m - mw) ** 2).sum(1) / (k - 1)
    tmp = (((1 - w / W) ** 2) / (n - 1)).sum(1)
    Fw = A / (1 + 2 * (k - 2) / (k ** 2 - 1) * tmp); p_w = st.f.sf(Fw, k - 1, (k ** 2 - 1) / (3 * tmp))
    r = _ranks(allv); idx = np.cumsum(np.r_[0, n]).astype(int)
    rb = np.stack([r[:, idx[i]:idx[i + 1]].mean(1) for i in range(k)], 1)
    H = 12 / (N * (N + 1)) * (n * (rb - (N + 1) / 2) ** 2).sum(1); p_kw = st.chi2.sf(H, k - 1)
    J = 0
    for i, j in pairs(k):
        J = J + (groups[i][:, :, None] < groups[j][:, None, :]).sum((1, 2))
    E = (N ** 2 - (n ** 2).sum()) / 4
    V = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72
    p_jt = 2 * st.norm.sf(np.abs(J - E) / np.sqrt(V))
    return dict(anova=p_an, welch=p_w, kw=p_kw, jt=p_jt)


def power_sim(reps=4000, seed=99, n=10, deltas=None):
    deltas = np.round(np.arange(0, 1.501, 0.125), 3) if deltas is None else deltas
    rs = np.random.default_rng(seed)
    out = {"deltas": list(deltas)}
    for pattern in ("mono", "umbrella"):
        res = {"kw": [], "jt": [], "anova": []}
        for d in deltas:
            mu = (0, d / 2, d) if pattern == "mono" else (0, d, 0)
            gs = [rs.normal(m_, 1, (reps, n)) for m_ in mu]
            p = sim_tests(gs)
            for key in res:
                res[key].append(float((p[key] < 0.05).mean()))
        out[pattern] = res
    return out


def typeI_sim(reps=20000, seed=2024):
    rs = np.random.default_rng(seed)
    rows = []
    for ns, sds in (((30, 30, 30), (15, 10, 5)), ((10, 20, 40), (15, 10, 5)), ((10, 20, 40), (5, 10, 15)), ((10, 20, 40), (10, 10, 10))):
        gs = [rs.normal(0, s, (reps, n)) for n, s in zip(ns, sds)]
        p = sim_tests(gs)
        rows.append((ns, sds, {k: float((v < 0.05).mean()) for k, v in p.items()}))
    return rows


def summ(g):
    q1, med, q3 = np.percentile(g, [25, 50, 75])
    return dict(n=len(g), mean=g.mean(), sd=g.std(ddof=1), med=med, q1=q1, q3=q3, min=g.min(), max=g.max())


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 70, "\n가. overview")
    for k in (2, 3, 4, 5, 6):
        m = k * (k - 1) // 2
        print(f"k={k}: comparisons={m}, FWER indep={1 - 0.95 ** m:.4f}")
    # actual FWER of unadjusted pairwise t tests, equal n=30, via studentized range
    for k in (3, 4, 5, 6):
        df = 30 * k - k
        tc = st.t.ppf(0.975, df)
        print(f"k={k}: actual FWER (all pairwise, n=30/grp) = {st.studentized_range.sf(tc * np.sqrt(2), k, df):.4f}")

    G = running_data()
    for nm, g in zip(ARMS, G):
        s = summ(g)
        print(f"{nm:22s} n={s['n']} mean={s['mean']:.2f} sd={s['sd']:.2f} median={s['med']:.2f} IQR={s['q1']:.2f}-{s['q3']:.2f} "
              f"min={s['min']} max={s['max']} n100={(g == 100).sum()} >=80: {(g >= 80).sum()}")
        se = s["sd"] / np.sqrt(30); tc = st.t.ppf(0.975, 29)
        print(f"    95% CI of mean {s['mean'] - tc * se:.2f} to {s['mean'] + tc * se:.2f}")
    a = anova(G)
    print("ANOVA:", {k: (np.round(v, 4) if not isinstance(v, np.ndarray) else v) for k, v in a.items()})
    print("scipy f_oneway", st.f_oneway(*G))
    print("F crit(2,87)", st.f.ppf(0.95, 2, 87))
    print("Welch ANOVA", welch_anova(G), "scipy", st.f_oneway(*G, equal_var=False))
    print("Levene mean", st.levene(*G, center="mean"), "Brown-Forsythe (median)", st.levene(*G, center="median"))
    print("Bartlett", st.bartlett(*G))
    res_all = np.concatenate([g - g.mean() for g in G])
    print("Shapiro residuals", st.shapiro(res_all), "per group", [round(st.shapiro(g).pvalue, 3) for g in G])
    print("SD ratio max/min", max(g.std(ddof=1) for g in G) / min(g.std(ddof=1) for g in G))
    print("sqrt MSW (pooled SD)", np.sqrt(a["msw"]))

    # secondary rows for 가 table
    ge80 = np.array([[(g >= 80).sum(), (g < 80).sum()] for g in G])
    chi = st.chi2_contingency(ge80, correction=False)
    print("PDC>=80 counts", ge80[:, 0], "pct", np.round(ge80[:, 0] / 30 * 100, 1), "chi2", chi.statistic, chi.pvalue, chi.dof)
    Ks = knowledge_scores()
    for g in Ks:
        s = summ(g); print("  knowledge", s["med"], s["q1"], s["q3"], "mean", round(s["mean"], 2))
    kk = kw(Ks)
    print("  knowledge KW", st.kruskal(*Ks), "H uncorrected", kk["H"], "C", kk["C"], "Hc", kk["Hc"], "ties", kk["ties"])

    print("=" * 70, "\n나. post hoc (running)")
    res, tcrit, qcrit = pairwise_all(G, a)
    print("tcrit(87)", tcrit, "qcrit(3,87)", qcrit, "qcrit/sqrt2", qcrit / np.sqrt(2))
    for r in res:
        print(f"{ARMS[r['j']]} - {ARMS[r['i']]}: diff={r['diff']:.3f} se={r['se']:.3f} t={r['t']:.3f} p={r['p']:.4f} "
              f"CI {r['lo']:.2f} {r['hi']:.2f} | bon={r['p_bon']:.4f} holm={r['p_holm']:.4f} | tukey q={r['q']:.3f} "
              f"p={r['p_tukey']:.4f} CI {r['tlo']:.2f} {r['thi']:.2f} | scheffe={r['p_scheffe']:.4f} | GH p={r['p_gh']:.4f} df={r['df_gh']:.1f}")
    tk = st.tukey_hsd(*G)
    ci = tk.confidence_interval(0.95)
    print("scipy tukey p", tk.pvalue[[0, 0, 1], [1, 2, 2]], "CI low", ci.low[[1, 2, 2], [0, 0, 1]], "high", ci.high[[1, 2, 2], [0, 0, 1]])
    gh = st.tukey_hsd(*G, equal_var=False)
    print("scipy games-howell p", gh.pvalue[[0, 0, 1], [1, 2, 2]])
    d = st.dunnett(G[1], G[2], control=G[0], rng=np.random.default_rng(1))
    dci = d.confidence_interval(0.95)
    print("Dunnett p", d.pvalue, "stat", d.statistic, "CI", dci.low, dci.high)
    # Scheffe contrast usual vs mean(brief, intensive)
    c = np.array([-1, 0.5, 0.5])
    L = (c * a["means"]).sum(); seL = np.sqrt(a["msw"] * (c ** 2 / a["n"]).sum())
    print("contrast usual vs avg(interventions): L", L, "se", seL, "t", L / seL, "p_unadj", 2 * st.t.sf(abs(L / seL), 87),
          "Scheffe p", st.f.sf((L / seL) ** 2 / 2, 2, 87))
    # linear trend contrast
    c = np.array([-1, 0, 1.])
    L = (c * a["means"]).sum(); seL = np.sqrt(a["msw"] * (c ** 2 / a["n"]).sum())
    print("linear contrast", L, seL, L / seL, 2 * st.t.sf(abs(L / seL), 87))

    # ANOVA significant but no Tukey pair significant (summary-statistic example)
    print("-- summary example: significant F, no significant Tukey pair")
    for ms in ([70.0, 76.8, 77.0], [70.0, 76.6, 76.8], [70.0, 76.9, 76.9]):
        sd = 12.0; n = 30
        ms = np.array(ms); gm = ms.mean()
        ssb = n * ((ms - gm) ** 2).sum(); msw = sd ** 2
        F = ssb / 2 / msw; p = st.f.sf(F, 2, 87)
        out = []
        for i, j in pairs(3):
            q = abs(ms[j] - ms[i]) / np.sqrt(msw / n)
            out.append(round(st.studentized_range.sf(q, 3, 87), 4))
        cc = np.array([-1, 0.5, 0.5]); L = (cc * ms).sum(); seL = np.sqrt(msw * (cc ** 2).sum() / n)
        print(ms, "F", round(F, 3), "p", round(p, 4), "tukey", out, "contrast L", L, "Scheffe p", round(st.f.sf((L / seL) ** 2 / 2, 2, 87), 4),
              "t", L / seL)

    print("=" * 70, "\n다. tiny ANOVA")
    t = anova(TINY)
    for g in TINY:
        print(" group", g, "mean", g.mean(), "SS within", ((g - g.mean()) ** 2).sum(), "dev", g - g.mean())
    print({k: v for k, v in t.items()})
    print("scipy", st.f_oneway(*TINY))
    print("-- running ANOVA table")
    print(f"SSB={a['ssb']:.2f} SSW={a['ssw']:.2f} SST={a['sst']:.2f} MSB={a['msb']:.2f} MSW={a['msw']:.2f} F={a['F']:.4f} p={a['p']:.5f} eta2={a['eta2']:.4f} omega2={a['omega2']:.4f}")
    print("grand mean", a["gm"], "means", a["means"])
    # dummy regression
    y = np.concatenate(G)
    X = np.column_stack([np.ones(90), np.r_[np.zeros(30), np.ones(30), np.zeros(30)], np.r_[np.zeros(60), np.ones(30)]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b; sse = (resid ** 2).sum(); s2 = sse / 87
    cov = s2 * np.linalg.inv(X.T @ X); seb = np.sqrt(np.diag(cov))
    print("dummy regression b", b, "se", seb, "t", b / seb, "p", 2 * st.t.sf(np.abs(b / seb), 87), "R2", 1 - sse / a["sst"])
    Freg = ((a["sst"] - sse) / 2) / s2
    print("regression F", Freg)
    # spread intuition figure data
    rng = np.random.default_rng(11)
    base = [rng.normal(0, 1, 10) for _ in range(3)]
    base = [(b_ - b_.mean()) / b_.std(ddof=1) for b_ in base]
    for sd in (4.0, 12.0):
        gs = [np.array(m + sd * b_) for m, b_ in zip((66, 72, 78), base)]
        aa = anova(gs)
        print(f"spread sd={sd}: F={aa['F']:.2f} p={aa['p']:.2g} MSB={aa['msb']:.1f} MSW={aa['msw']:.1f}")

    print("-- type I error simulation (20000 reps, vectorized)")
    for ns, sds, r in typeI_sim():
        print(ns, sds, {k: round(v, 4) for k, v in r.items()})

    print("=" * 70, "\n라. Kruskal-Wallis")
    kt = kw(TINY)
    allv = np.concatenate(TINY)
    print("tiny sorted", sorted(allv))
    print("tiny ranks by group", [kt["ranks"][i * 5:(i + 1) * 5] for i in range(3)], "R", kt["R"], "mean rank", kt["mean_rank"], "H", kt["H"], "p", kt["p"])
    print("scipy", st.kruskal(*TINY))
    # exact permutation distribution for tiny (756756 partitions)
    vals = allv
    Hs = []; Js = []
    idx = np.arange(15)
    rk = st.rankdata(vals)
    for A in itertools.combinations(idx, 5):
        rest = [i for i in idx if i not in A]
        for B in itertools.combinations(rest, 5):
            C = [i for i in rest if i not in B]
            RA, RB, RC = rk[list(A)].sum(), rk[list(B)].sum(), rk[C].sum()
            Hs.append(12 / 240 * (RA ** 2 + RB ** 2 + RC ** 2) / 5 - 48)
            # JT: counts
            a_, b_, c_ = vals[list(A)], vals[list(B)], vals[C]
            Js.append((a_[:, None] < b_[None, :]).sum() + (a_[:, None] < c_[None, :]).sum() + (b_[:, None] < c_[None, :]).sum())
    Hs = np.array(Hs); Js = np.array(Js)
    print("n partitions", len(Hs))
    print("exact KW p", (Hs >= kt["H"] - 1e-9).mean())
    jtt = jt(TINY)
    print("tiny JT", jtt)
    print("exact JT p one-sided", (Js >= jtt["J"]).mean(), "two-sided (2x)", 2 * (Js >= jtt["J"]).mean(),
          "two-sided |J-E|", (np.abs(Js - 37.5) >= abs(jtt["J"] - 37.5) - 1e-9).mean())
    print("perm var J", Js.var(), "formula", jtt["V"])
    print("tiny ANOVA p", t["p"])

    k2 = kw(G)
    print("running KW: R", k2["R"], "mean rank", k2["mean_rank"], "H", k2["H"], "C", k2["C"], "Hc", k2["Hc"], "p", k2["p"], "ties", k2["ties"])
    print("scipy", st.kruskal(*G))
    for r in dunn(G):
        print(f"Dunn {ARMS[r['j']]} - {ARMS[r['i']]}: z={r['z']:.4f} p={r['p']:.4f} holm={r['p_holm']:.4f} bon={r['p_bon']:.4f}")
    for i, j in pairs(3):
        mw = st.mannwhitneyu(G[j], G[i], alternative="two-sided")
        print(f"pairwise MW {ARMS[j]} vs {ARMS[i]}: U={mw.statistic} p={mw.pvalue:.4f}")
    # epsilon squared
    print("epsilon^2 (H/(N-1))", k2["Hc"] / 89, "eta2_H (H-k+1)/(N-k)", (k2["Hc"] - 3 + 1) / (90 - 3))

    print("=" * 70, "\n마. Jonckheere-Terpstra")
    jr = jt(G)
    print("running JT", jr)
    # permutation check of variance with ties (running data)
    rp = np.random.default_rng(5)
    allv = np.concatenate(G); Jp = []
    for _ in range(4000):
        pm = rp.permutation(allv)
        Jp.append(jt([pm[:30], pm[30:60], pm[60:]])["J"])
    Jp = np.array(Jp)
    print("perm mean/var", Jp.mean(), Jp.var(), "perm p one-sided", (Jp >= jr["J"]).mean())
    # umbrella example: same data, reorder so that brief is highest? use tiny reorder
    print("P(X_j > X_i) = U/(n_i n_j):", {k: v / 900 for k, v in jr["U"].items()})
    # parametric trend: regression on ordinal score 1,2,3
    x = np.repeat([1, 2, 3], 30); yy = np.concatenate(G)
    lr = st.linregress(x, yy)
    print("score regression slope", lr.slope, "se", lr.stderr, "t", lr.slope / lr.stderr, "p", lr.pvalue)
    # proportion trend (PDC >= 80): linear-by-linear association (N-1) r^2, Cochran-Armitage N r^2
    yb = np.concatenate([(g >= 80).astype(float) for g in G]); r = np.corrcoef(x, yb)[0, 1]
    print("linear-by-linear M2", 89 * r ** 2, "p", st.chi2.sf(89 * r ** 2, 1), "| Cochran-Armitage chi2", 90 * r ** 2, st.chi2.sf(90 * r ** 2, 1))
    print("-- power simulation (normal, n=10/grp, 4000 reps, vectorized)")
    ps = power_sim()
    for pat in ("mono", "umbrella"):
        for i, d in enumerate(ps["deltas"]):
            print(pat, d, {k: round(ps[pat][k][i], 4) for k in ("kw", "jt", "anova")})

def python_outputs():
    """파이썬 출력 상자 (scipy 1.17, statsmodels 0.15, pandas 3.0).
    Run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch04.py"""
    import pandas as pd
    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from statsmodels.stats.oneway import anova_oneway
    except ImportError:
        print("statsmodels not found: run with  source /home/claude/pylibs/env.sh")
        return
    print("=" * 70, "\n파이썬 출력 상자")
    usual, brief, intensive = running_data()
    df = pd.DataFrame({"arm": pd.Categorical(np.repeat(["usual", "brief", "intensive"], 30),
                                             categories=["usual", "brief", "intensive"]),
                       "pdc": np.concatenate([usual, brief, intensive])})
    # 나 절: Tukey HSD (0 = usual, 1 = brief, 2 = intensive)
    res = st.tukey_hsd(usual, brief, intensive)
    print(res)
    # 나 절 본문: 합동분산(MSW)을 쓰는 쌍별 t 검정 + Bonferroni/Holm
    fit = smf.ols("pdc ~ arm", data=df).fit()
    print(fit.t_test_pairwise("arm", method=["bonferroni", "holm"]).result_frame.round(4).to_string())
    # 다 절: 분산분석표
    print(sm.stats.anova_lm(fit))
    # 다 절: Welch 분산분석
    print(repr(st.f_oneway(usual, brief, intensive, equal_var=False)))
    print(repr(st.f_oneway(usual, brief, intensive)))
    w = anova_oneway(df["pdc"], df["arm"], use_var="unequal")
    print(repr(w.df))
    print("levene median", st.levene(usual, brief, intensive), "mean", st.levene(usual, brief, intensive, center="mean"))
    # 라 절: Kruskal-Wallis (+ 작은 예제의 정확 p를 순열검정으로)
    print(repr(st.kruskal(usual, brief, intensive)))
    perm = st.permutation_test(TINY, lambda *s, axis=-1: st.kruskal(*s, axis=axis).statistic,
                               permutation_type="independent", alternative="greater",
                               n_resamples=np.inf, vectorized=True)
    print("tiny KW exact permutation p", perm.pvalue)


if __name__ == "__main__":
    main()
    python_outputs()
