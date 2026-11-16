"""Numbers for 14장 e 절 (자발적 보고 자료와 불균형 분석). Hypothetical spontaneous-report database.
Run: python3 gen/nums_ch14e.py   (prints every number used in the section; R check in gen/disp_ch14e.R)"""
import math
from scipy import stats

Z = stats.norm.ppf(0.975)

# ---- hypothetical database (after de-duplication), reports 2014-2023
N_ALL = 1_000_000
REP = {"SGLT2": 5_000, "DPP4": 10_000, "OTHER_DM": 35_000}          # reports with each class as suspect drug
DKA = {"SGLT2": 60, "DPP4": 40, "OTHER_DM": 140, "NON_DM": 160}     # DKA reports
N_DM = sum(REP.values())                                              # 50,000 reports with a glucose-lowering drug
DKA_ALL = sum(DKA.values())                                           # 400

def table(drug, background):
    a = DKA[drug]; n_drug = REP[drug]
    if background == "all":
        N = N_ALL; ev = DKA_ALL
    else:
        N = N_DM; ev = DKA_ALL - DKA["NON_DM"]
    return a, n_drug - a, ev - a, N - n_drug - (ev - a)

def measures(a, b, c, d):
    N = a + b + c + d
    ror = a * d / (b * c); se = math.sqrt(1/a + 1/b + 1/c + 1/d)
    prr = (a / (a + b)) / (c / (c + d)); sep = math.sqrt(1/a - 1/(a + b) + 1/c - 1/(c + d))
    E = (a + b) * (a + c) / N
    ic = math.log2((a + 0.5) / (E + 0.5))
    ic025 = math.log2(stats.gamma.ppf(0.025, a + 0.5, scale=1 / (E + 0.5)))
    ic975 = math.log2(stats.gamma.ppf(0.975, a + 0.5, scale=1 / (E + 0.5)))
    # Pearson chi-square with Yates correction (R chisq.test default for 2x2)
    def chi(yates):
        obs = [[a, b], [c, d]]; rs = [a + b, c + d]; cs = [a + c, b + d]; s = 0
        for i in range(2):
            for j in range(2):
                e = rs[i] * cs[j] / N; dev = abs(obs[i][j] - e) - (0.5 if yates else 0)
                s += max(dev, 0) ** 2 / e
        return s
    return dict(a=a, b=b, c=c, d=d, N=N, E=E, oe=a / E,
                ror=ror, ror_lo=math.exp(math.log(ror) - Z * se), ror_hi=math.exp(math.log(ror) + Z * se), se_ror=se,
                prr=prr, prr_lo=math.exp(math.log(prr) - Z * sep), prr_hi=math.exp(math.log(prr) + Z * sep), se_prr=sep,
                chi_y=chi(True), chi=chi(False), ic=ic, ic025=ic025, ic975=ic975,
                p_drug=a / (a + b), p_other=c / (c + d))

def show(tag, m):
    print(f"{tag}: a={m['a']} b={m['b']:,} c={m['c']} d={m['d']:,} N={m['N']:,}")
    print(f"   p_drug={100*m['p_drug']:.2f}%  p_other={100*m['p_other']:.3f}%  E={m['E']:.2f}  O/E={m['oe']:.2f}")
    print(f"   ROR {m['ror']:.2f} ({m['ror_lo']:.2f}-{m['ror_hi']:.2f}) se={m['se_ror']:.4f}")
    print(f"   PRR {m['prr']:.2f} ({m['prr_lo']:.2f}-{m['prr_hi']:.2f}) se={m['se_prr']:.4f}  chi2_yates={m['chi_y']:.1f} chi2={m['chi']:.1f}")
    print(f"   IC {m['ic']:.2f}  IC025 {m['ic025']:.2f}  IC975 {m['ic975']:.2f}")

RES = {}
for drug in ("SGLT2", "DPP4"):
    for bg in ("all", "dm"):
        RES[(drug, bg)] = measures(*table(drug, bg))
# first year only (reports received by end of 2014), glucose-lowering-drug background
RES[("SGLT2", "fy")] = measures(4, 296, 23, 5_677)
# masking: DPP-4 inhibitors against glucose-lowering-drug reports without SGLT2 inhibitor reports
RES[("DPP4", "dm_noSGLT2")] = measures(DKA["DPP4"], REP["DPP4"] - DKA["DPP4"], DKA["OTHER_DM"],
                                       REP["OTHER_DM"] - DKA["OTHER_DM"])

if __name__ == "__main__":
    for k, m in RES.items():
        show(f"{k[0]} vs {k[1]}", m)
    print("DKA share among reports without a glucose-lowering drug:", f"{100*DKA['NON_DM']/(N_ALL-N_DM):.3f}%",
          "| with glucose-lowering drugs:", f"{100*(DKA_ALL-DKA['NON_DM'])/N_DM:.2f}%")
    print("share of DKA reports that carry a glucose-lowering drug:", f"{100*(DKA_ALL-DKA['NON_DM'])/DKA_ALL:.0f}%")
