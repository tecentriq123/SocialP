"""Numbers for 종합 4 (qz94, systematic review and meta-analysis). Hypothetical: drug P added to standard prophylaxis vs placebo
for postoperative nausea and vomiting (PONV) within 24 h after general anaesthesia; 12 RCTs with event counts.
Main analysis: random effects, tau^2 by REML, CI by HKSJ (t, k-1 df); prediction interval with t (k-2 df).
Run: python3 gen/quiz_rv4.py"""
import math
import numpy as np
from scipy import optimize, stats

# name, events_P, n_P, events_C, n_C, dose ('high'/'low'), open-label (True = outcome assessors aware of allocation)
T = [
    ("A", 80, 400, 124, 400, "high", False),
    ("B", 84, 380, 112, 380, "low", False),
    ("C", 112, 360, 110, 360, "low", False),
    ("D", 27, 150, 46, 150, "high", False),
    ("E", 26, 120, 36, 120, "low", False),
    ("F", 11, 80, 27, 80, "high", True),
    ("G", 22, 100, 31, 100, "high", False),
    ("H", 8, 60, 21, 60, "low", True),
    ("I", 20, 90, 30, 90, "high", False),
    ("J", 11, 50, 15, 50, "low", False),
    ("K", 6, 45, 17, 45, "high", True),
    ("L", 17, 70, 20, 70, "low", False),
]


def lor(a, n1, c, n0):
    b, d = n1 - a, n0 - c
    return math.log(a * d / (b * c)), math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)


def reml_tau2(y, v):
    def nll(t2):
        w = 1 / (v + t2); mu = np.sum(w * y) / np.sum(w)
        return 0.5 * (np.sum(np.log(v + t2)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))
    r = optimize.minimize_scalar(nll, bounds=(0, 5), method="bounded")
    return max(0.0, r.x)


def meta(rows, label, verbose=True):
    y = np.array([lor(r[1], r[2], r[3], r[4])[0] for r in rows]); se = np.array([lor(r[1], r[2], r[3], r[4])[1] for r in rows])
    v = se ** 2; k = len(y)
    w = 1 / v; fe = np.sum(w * y) / np.sum(w); sefe = math.sqrt(1 / np.sum(w))
    Q = np.sum(w * (y - fe) ** 2); I2 = max(0, (Q - (k - 1)) / Q)
    t2 = reml_tau2(y, v)
    wr = 1 / (v + t2); re = np.sum(wr * y) / np.sum(wr)
    # HKSJ
    q = np.sum(wr * (y - re) ** 2) / (k - 1); sehk = math.sqrt(q / np.sum(wr)); tq = stats.t.ppf(0.975, k - 1)
    sewald = math.sqrt(1 / np.sum(wr))
    p_hk = 2 * stats.t.sf(abs(re / sehk), k - 1)
    # prediction interval: HKSJ SE and t(k-1), the rule ch19 uses
    half = tq * math.sqrt(t2 + sehk ** 2)  # HKSJ SE, t with k-1 df (as in ch19)
    out = dict(k=k, fe=math.exp(fe), fe_lo=math.exp(fe - 1.96 * sefe), fe_hi=math.exp(fe + 1.96 * sefe), Q=Q, I2=I2, tau2=t2,
               re=math.exp(re), lo=math.exp(re - tq * sehk), hi=math.exp(re + tq * sehk), p=p_hk,
               pi_lo=math.exp(re - half), pi_hi=math.exp(re + half), lre=re, sere=sehk, y=y, se=se, wr=wr,
               wald_lo=math.exp(re - 1.96 * sewald), wald_hi=math.exp(re + 1.96 * sewald))
    if verbose:
        print("%s k=%d FE %.2f (%.2f-%.2f) | RE %.2f (%.2f-%.2f) P=%.4f | Q %.2f I2 %.0f%% tau2 %.3f | PI %.2f-%.2f | Wald %.2f-%.2f" %
              (label, k, out["fe"], out["fe_lo"], out["fe_hi"], out["re"], out["lo"], out["hi"], p_hk, Q, 100 * I2, t2,
               out["pi_lo"], out["pi_hi"], out["wald_lo"], out["wald_hi"]))
    return out


for r in T:
    l, s = lor(r[1], r[2], r[3], r[4])
    print("  %s OR %.2f (%.2f-%.2f) risk P %.1f%% C %.1f%% %s %s" % (r[0], math.exp(l), math.exp(l - 1.96 * s), math.exp(l + 1.96 * s),
          100 * r[1] / r[2], 100 * r[3] / r[4], r[5], "open" if r[6] else ""))
nsig = sum(1 for r in T if abs(lor(*r[1:5])[0] / lor(*r[1:5])[1]) > 1.96)
print("vote count: significant %d of %d" % (nsig, len(T)))
m = meta(T, "ALL")
tot = [sum(r[i] for r in T) for i in (1, 2, 3, 4)]
a, n1, c, n0 = tot; l, s = lor(a, n1, c, n0)
print("raw-summed: %d/%d (%.1f%%) vs %d/%d (%.1f%%) OR %.2f" % (a, n1, 100 * a / n1, c, n0, 100 * c / n0, math.exp(l)))
# relative weights RE
wr = m["wr"] / m["wr"].sum(); print("RE weights %%:", " ".join("%s %.1f" % (r[0], 100 * w) for r, w in zip(T, wr)))
hi = meta([r for r in T if r[5] == "high"], "HIGH"); lo = meta([r for r in T if r[5] == "low"], "LOW")
d = hi["lre"] - lo["lre"]; sed = math.sqrt(hi["sere"] ** 2 + lo["sere"] ** 2); z = d / sed
print("subgroup diff: ratio %.2f (%.2f-%.2f) z %.2f P %.3f" % (math.exp(d), math.exp(d - 1.96 * sed), math.exp(d + 1.96 * sed), z, 2 * stats.norm.sf(abs(z))))
meta([r for r in T if not r[6]], "BLINDED-ONLY")
# Egger test: regress y/se on 1/se
y, se = m["y"], m["se"]; X = np.column_stack([np.ones_like(se), 1 / se]); Y = y / se
beta, res, *_ = np.linalg.lstsq(X, Y, rcond=None); dof = len(Y) - 2
s2 = np.sum((Y - X @ beta) ** 2) / dof; cov = s2 * np.linalg.inv(X.T @ X); tint = beta[0] / math.sqrt(cov[0, 0])
print("Egger intercept %.2f P %.3f" % (beta[0], 2 * stats.t.sf(abs(tint), dof)))

# Peters test (regress lnOR on 1/N, weights 1/(1/events + 1/non-events))
yy, xx, ww = [], [], []
for r in T:
    a_, n1_, c_, n0_ = r[1:5]; yy.append(lor(a_, n1_, c_, n0_)[0]); N = n1_ + n0_; xx.append(1 / N); e_ = a_ + c_; ww.append(1 / (1 / e_ + 1 / (N - e_)))
yy, xx, ww = map(np.array, (yy, xx, ww)); Xp = np.column_stack([np.ones_like(xx), xx]); W = np.diag(ww)
bp = np.linalg.solve(Xp.T @ W @ Xp, Xp.T @ W @ yy); rp = yy - Xp @ bp; s2p = (rp @ W @ rp) / (len(yy) - 2)
tp_ = bp[1] / math.sqrt((s2p * np.linalg.inv(Xp.T @ W @ Xp))[1, 1]); print("Peters P %.3f" % (2 * stats.t.sf(abs(tp_), len(yy) - 2)))

# b1: trial reporting only OR 0.60 (95% CI 0.40-0.90)
lo_, hi_ = math.log(0.40), math.log(0.90); se1 = (hi_ - lo_) / (2 * 1.96)
print("b1 log OR %.3f SE %.3f FE weight %.1f | wrong SE using ratio (0.90-0.40)/3.92 %.3f" % (math.log(0.60), se1, 1 / se1 ** 2, (0.90 - 0.40) / 3.92))
print("   check symmetric: ln0.60-ln0.40 %.3f ; ln0.90-ln0.60 %.3f" % (math.log(0.60) - lo_, hi_ - math.log(0.60)))

# d1: apply pooled OR to baseline risk 30%
OR = m["re"]
for base in (0.30, 0.10):
    o = base / (1 - base) * OR; r1 = o / (1 + o); rd = base - r1
    print("d1 base %.0f%%: treated %.1f%% RD %.1f%%p NNT %.1f | direct multiply %.1f%% RD %.1f%%p NNT %.1f" %
          (100 * base, 100 * r1, 100 * rd, 1 / rd, 100 * base * OR, 100 * (base - base * OR), 1 / (base - base * OR)))
