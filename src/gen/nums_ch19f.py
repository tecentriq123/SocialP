"""Numbers for chapter 19 f 절 (네트워크 메타분석).

Network (hypothetical): the 10 drug X vs placebo trials of nums_ch19.py (A–J) plus simulated trials of
drug Y vs placebo (K–O), drug X vs drug Y (Q, R) and drug Z vs placebo (S, T); outcome MACE.
The seed was chosen so that the Y vs placebo pooled OR equals the value assumed in 19장 e 절
('대조약이 다른 시험은 어떻게 잇나?', 0.90, 95% CI 0.80–1.01), the pairwise τ² of Y–P, X–Y and Z–P is 0,
direct and indirect X vs Y differ mildly, and drug Z has the best point estimate with a wide interval.
Run: python3 gen/nums_ch19f.py   (writes gen/ch19f_network.csv; NMA itself: Rscript gen/nma_ch19f.R)
"""
import os, sys, csv
import numpy as np
import scipy.stats as st
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nums_ch19 as N

SEED = 21471
YP = [("K", 2012, 1300, 1300, 12), ("L", 2013, 800, 800, 12), ("M", 2015, 2300, 2300, 18), ("N", 2016, 500, 500, 6), ("O", 2019, 1700, 1700, 12)]
XY = [("Q", 2017, 2100, 2100, 12), ("R", 2020, 1600, 1600, 12)]
ZP = [("S", 2021, 420, 420, 6), ("T", 2022, 560, 560, 9)]


def lor(a, n1, c, n0):
    b = n1 - a; d = n0 - c
    return np.log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d


def network():
    rows = [dict(study=r["nm"], year=r["yr"], t1="X", t0="P", n1=r["nT"], n0=r["nC"], e1=r["eT"], e0=r["eC"], mo=r["mo"]) for r in N.simulate()]
    rng = np.random.default_rng(SEED)
    for spec, t1, t0, lo1, base_or in ((YP, "Y", "P", np.log(0.90), None), (XY, "X", "Y", np.log(0.84 / 0.90), 0.90), (ZP, "Z", "P", np.log(0.80), None)):
        for nm, yr, n1, n0, mo in spec:
            u = rng.normal(0, 0.05)
            p0 = N.base_mace(mo) if base_or is None else N.or2p(N.base_mace(mo), np.log(base_or))
            e1 = int(rng.binomial(n1, N.or2p(p0, lo1 + u))); e0 = int(rng.binomial(n0, p0))
            rows.append(dict(study=nm, year=yr, t1=t1, t0=t0, n1=n1, n0=n0, e1=e1, e0=e0, mo=mo))
    return rows


def compute():
    rows = network()
    g = lambda a, b: [lor(r["e1"], r["n1"], r["e0"], r["n0"]) for r in rows if r["t1"] == a and r["t0"] == b]
    out = {k: N.pool(*zip(*g(*k.split("-")))) for k in ("X-P", "Y-P", "X-Y", "Z-P")}
    xp, yp, xy = out["X-P"], out["Y-P"], out["X-Y"]
    # Bucher indirect X vs Y through placebo (X: random-effects mean with its model SE, as in 19장 e 절)
    ind = xp["re"] - yp["re"]; se_i = np.sqrt(xp["re_se"] ** 2 + yp["re_se"] ** 2)
    dr, se_d = xy["re"], xy["re_se"]
    wd, wi = 1 / se_d ** 2, 1 / se_i ** 2
    comb = (wd * dr + wi * ind) / (wd + wi); se_c = 1 / np.sqrt(wd + wi)
    z = (dr - ind) / np.sqrt(se_d ** 2 + se_i ** 2)
    out.update(rows=rows, ind=ind, se_i=se_i, dr=dr, se_d=se_d, comb=comb, se_c=se_c, share_d=wd / (wd + wi),
               z=z, p_diff=2 * st.norm.sf(abs(z)), ror=np.exp(dr - ind))
    # characteristics by comparison (transitivity table): years, follow-up, placebo/common-comparator 12-month risk
    ch = {}
    for (a, b) in (("X", "P"), ("Y", "P"), ("X", "Y"), ("Z", "P")):
        rr = [r for r in rows if r["t1"] == a and r["t0"] == b]
        risk12 = np.average([1 - (1 - r["e0"] / r["n0"]) ** (12 / r["mo"]) for r in rr], weights=[r["n0"] for r in rr])
        ch[f"{a}-{b}"] = dict(k=len(rr), n=sum(r["n1"] + r["n0"] for r in rr), yr=(min(r["year"] for r in rr), max(r["year"] for r in rr)),
                              mo=(min(r["mo"] for r in rr), max(r["mo"] for r in rr)), risk12=risk12)
    out["chars"] = ch
    return out


if __name__ == "__main__":
    R = compute()
    with open(os.path.join(HERE, "ch19f_network.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(R["rows"][0])); w.writeheader(); w.writerows(R["rows"])
    for k in ("X-P", "Y-P", "X-Y", "Z-P"):
        p = R[k]; print(k, "k", p["k"], "RE", [round(x, 3) for x in p["re_or"]], p["ci_method"], "FE", [round(x, 3) for x in p["fe_or"]], "tau2", round(p["t2"], 4))
    e = lambda m, s: [round(float(np.exp(x)), 3) for x in (m, m - 1.96 * s, m + 1.96 * s)]
    print("indirect", e(R["ind"], R["se_i"]), "se", round(R["se_i"], 4))
    print("direct", e(R["dr"], R["se_d"]), "se", round(R["se_d"], 4))
    print("combined", e(R["comb"], R["se_c"]), "se", round(R["se_c"], 4), "direct share", round(R["share_d"], 3))
    print("RoR", round(R["ror"], 3), "z", round(R["z"], 3), "p", round(R["p_diff"], 3))
    for k, v in R["chars"].items(): print(k, v)
    print("N", sum(r["n1"] + r["n0"] for r in R["rows"]), "trials", len(R["rows"]))
