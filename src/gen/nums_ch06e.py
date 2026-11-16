"""Chapter 6 e 절 (공분산분석) numbers.

The trial is the pharmacist-counseling HbA1c trial of 2장 a 절 (36 vs 36). The 6-month change values are
exactly those of gen/nums_ch02.py (exact_data seeds 11, 12); baseline HbA1c is added here (seed 33), and the
6-month value is baseline + change. Run: python3 gen/nums_ch06e.py
"""
import numpy as np, scipy.stats as st

def exact_data(m, s, n, seed):
    z = np.random.default_rng(seed).normal(size=n); z = (z - z.mean()) / z.std(ddof=1); return m + s * z

c1 = exact_data(-0.90, 0.80, 36, 11)   # counseling, change (2장)
c2 = exact_data(-0.40, 0.90, 36, 12)   # usual care, change (2장)
rng = np.random.default_rng(33)
d1 = rng.normal(0, 0.80, 36); d2 = rng.normal(0, 0.80, 36)
b1 = -0.45 * (c1 - c1.mean()) + d1; b2 = -0.45 * (c2 - c2.mean()) + d2
b1 = b1 - b1.mean() + 8.36; b2 = b2 - b2.mean() + 8.12
base = np.round(np.r_[b1, b2], 2); chg = np.round(np.r_[c1, c2], 2)
post = np.round(base + chg, 2); chg = np.round(post - base, 2)
arm = np.r_[np.ones(36), np.zeros(36)]

def ols(cols, y, names):
    X = np.column_stack([np.ones(len(y))] + cols); XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ y
    e = y - X @ b; df = len(y) - X.shape[1]; s2 = e @ e / df; se = np.sqrt(np.diag(s2 * XtXi))
    t = b / se; p = 2 * st.t.sf(abs(t), df); tc = st.t.ppf(.975, df)
    out = {n: dict(b=b[i], se=se[i], t=t[i], p=p[i], lo=b[i] - tc * se[i], hi=b[i] + tc * se[i]) for i, n in enumerate(names)}
    out['_'] = dict(df=df, sigma=np.sqrt(s2), R2=1 - (e @ e) / ((y - y.mean()) @ (y - y.mean())), XtXi=XtXi, b=b)
    return out

if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    for g, m in (('counsel', arm == 1), ('usual', arm == 0)):
        print(g, 'base', base[m].mean().round(3), base[m].std(ddof=1).round(3), 'post', post[m].mean().round(3), post[m].std(ddof=1).round(3),
              'chg', chg[m].mean().round(3), chg[m].std(ddof=1).round(3), 'range base', base[m].min(), base[m].max(), 'post', post[m].min(), post[m].max())
    print('baseline diff', (base[arm == 1].mean() - base[arm == 0].mean()).round(4), 'all base mean', base.mean().round(4))
    r_within = np.corrcoef(np.r_[base[arm == 1] - base[arm == 1].mean(), base[arm == 0] - base[arm == 0].mean()],
                           np.r_[post[arm == 1] - post[arm == 1].mean(), post[arm == 0] - post[arm == 0].mean()])[0, 1]
    print('pooled within-group r(base, post)', round(r_within, 4))
    A = ols([arm], post, ['int', 'arm']); B = ols([arm], chg, ['int', 'arm']); C = ols([arm, base], post, ['int', 'arm', 'base'])
    C2 = ols([arm, base], chg, ['int', 'arm', 'base']); I = ols([arm, base, arm * (base - base.mean())], post, ['int', 'arm', 'base', 'ix'])
    for tag, M in (('post only', A), ('change', B), ('ANCOVA', C), ('ANCOVA chg', C2), ('interaction', I)):
        print(tag, {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in M.items() if k != '_'}, 'sigma', round(M['_']['sigma'], 4), 'df', M['_']['df'], 'R2', round(M['_']['R2'], 4))
    print('welch change', st.ttest_ind(chg[arm == 1], chg[arm == 0], equal_var=False))
    bC = C['_']['b']; mb = base.mean()
    ls1 = bC[0] + bC[1] + bC[2] * mb; ls0 = bC[0] + bC[2] * mb
    # SE of LS means
    for a, v in ((1, ls1), (0, ls0)):
        x = np.array([1, a, mb]); se = C['_']['sigma'] * np.sqrt(x @ C['_']['XtXi'] @ x); print('LS mean arm', a, round(v, 4), 'SE', round(se, 4))
    print('check: arm diff = post diff - beta*base diff', round(post[arm == 1].mean() - post[arm == 0].mean() - bC[2] * (base[arm == 1].mean() - base[arm == 0].mean()), 4))
    import csv
    with open('/home/claude/qref/e/hba1c_trial.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id', 'arm', 'hba1c_0', 'hba1c_6m'])
        for i in range(72): w.writerow([i + 1, int(arm[i]), f'{base[i]:.2f}', f'{post[i]:.2f}'])
