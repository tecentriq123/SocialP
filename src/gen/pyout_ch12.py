"""Python (statsmodels) outputs pasted into content/ch12.html ('파이썬 출력에서는 이렇게 보입니다').

run:  source /home/claude/pylibs/env.sh && python3 gen/pyout_ch12.py
Data: the chapter's simulated COPD cohort (lib_ch12.cohort()).  Every block is produced by actually
running statsmodels (0.15.0 here).  The script prints the HTML of each <pre class="out"> block
(escaped, with <span class="mk">n</span> markers), then the numbers quoted in the text, and compares
them with the chapter's own numpy implementation (lib_ch12).

Note on the negative binomial standard errors.  smf.negativebinomial() estimates beta and alpha jointly
and takes its SEs from the observed information (Hessian of the full log-likelihood).  lib_ch12.nb_glm()
follows the GLM approach (IRLS at fixed alpha, expected information), as does statsmodels' GLM with
family=NegativeBinomial(alpha=alpha_hat).  The two sets of SEs differ in the 3rd-4th significant digit
(drug A: 0.07183 vs 0.07142).  Since 2026-09-30 the chapter quotes the smf.negativebinomial() values
(SE 0.0718; Table 3 female 0.87 (0.75-1.01), CCI 1.12 (1.08-1.17)); all other displayed numbers are identical either way.
"""
import sys, os, io, ast, html, contextlib, warnings, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lib_ch12 import cohort, design, poisson_glm, nb_glm, Z


# ------------------------------------------------------------------ tiny REPL recorder (same as pyout_ch09)
class Console:
    """Runs code like the Python prompt and records a transcript of input and output lines."""

    def __init__(self):
        self.ns = {}
        self.blocks = {}

    def run(self, block, src, filt=None, show=True):
        lines = src.strip("\n").split("\n")
        rec = self.blocks.setdefault(block, [])
        if show:
            depth = 0
            for l in lines:
                cont = depth > 0 or l[:1].isspace()
                rec.append(["in", ("... " if cont else ">>> ") + l, []])
                depth += sum(l.count(c) for c in "([{") - sum(l.count(c) for c in ")]}")
        buf = io.StringIO()
        tree = ast.parse("\n".join(lines))
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(buf):
            warnings.simplefilter("default")
            exec(compile(tree, "<stdin>", "exec"), self.ns)
            if last is not None:
                val = eval(compile(last, "<stdin>", "eval"), self.ns)
                if val is not None:
                    print(repr(val))
        out = buf.getvalue().rstrip("\n").split("\n") if buf.getvalue().strip() else []
        for w in caught:
            fn = re.sub(r"^.*?(statsmodels/)", r"…/\1", w.filename)
            out.append(f"{fn}:{w.lineno}: {w.category.__name__}: {w.message}")
        if filt is not None:
            out = filt(out)
        for l in out:
            rec.append(["out", l, []])
        return out

    def mark(self, block, substr, n, nth=1):
        k = 0
        for row in self.blocks[block]:
            if substr in row[1]:
                k += 1
                if k == nth:
                    row[2].append(n)
                    return
        raise KeyError(f"{block}: '{substr}' not found")

    def html(self, block, pad=1):
        out = []
        for kind, text, mks in self.blocks[block]:
            t = html.escape(text, quote=False)
            if kind == "in":
                t = f'<span class="cm">{t}</span>'
            if mks:
                t += " " * pad + " ".join(f'<span class="mk">{m}</span>' for m in mks)
            out.append(t)
        return "<pre class=\"out\">" + "\n".join(out) + "</pre>"


# ------------------------------------------------------------------ data frame (same data as nums_ch12)
c = cohort()
copd = pd.DataFrame(dict(
    exac=c["y"].astype(int), py=c["py"],
    drug=pd.Categorical(np.where(c["drug"] == 1, "A", "B"), categories=["B", "A"]),
    agegrp=pd.Categorical(np.array(["40-64", "65-74", "75+"])[c["age"]], categories=["40-64", "65-74", "75+"]),
    female=c["female"], cci=c["cci"]))

C = Console()
C.ns.update(copd=copd)

# ------------------------------------------------------------------ block P: Poisson regression
C.run("P", "import numpy as np\nimport statsmodels.api as sm\nimport statsmodels.formula.api as smf")
C.run("P", 'fit = smf.glm("exac ~ drug + agegrp + female + cci", data=copd,\n'
           '              family=sm.families.Poisson(),\n'
           '              offset=np.log(copd["py"])).fit()')
C.run("P", "print(fit.summary())")
C.run("P", 'print(fit.bse["drug[T.A]"], fit.pvalues["drug[T.A]"])')
C.run("P", "print(round(fit.null_deviance, 1), round(fit.aic, 1), round(fit.pearson_chi2, 1))")
C.mark("P", "drug[T.A]  ", 1)
C.mark("P", "0.0548989", 2)
C.mark("P", "Intercept  ", 3)
C.mark("P", "agegrp[T.65-74]", 4)
C.mark("P", "cci   ", 5)
C.mark("P", "Scale:", 6)
C.mark("P", "Deviance:", 7)
C.mark("P", "3233.5 5125.7", 7)
fit = C.ns["fit"]

# ------------------------------------------------------------------ block N: negative binomial (NB2)
C.run("N", 'nb = smf.negativebinomial("exac ~ drug + agegrp + female + cci",\n'
           '                          data=copd, offset=np.log(copd["py"])).fit()')
C.run("N", "print(nb.summary())")
C.run("N", "print(round(nb.llf, 3), round(nb.aic, 1), round(2 * (nb.llf - fit.llf), 1))")
C.run("N", 'alpha = nb.params["alpha"]')
C.run("N", 'g = smf.glm("exac ~ drug + agegrp + female + cci", data=copd,\n'
           '            family=sm.families.NegativeBinomial(alpha=alpha),\n'
           '            offset=np.log(copd["py"])).fit()')
C.run("N", 'print(round(g.bse["drug[T.A]"], 5), round(g.deviance / g.df_resid, 2),\n'
           '      round(g.pearson_chi2 / g.df_resid, 2))')
C.mark("N", "drug[T.A]  ", 1)
C.mark("N", "alpha      ", 2)
C.mark("N", "Log-Likelihood:", 3)
C.mark("N", "-2418.256 4850.5", 3)
C.mark("N", "round(g.pearson_chi2", 4)
nb, g = C.ns["nb"], C.ns["g"]

if __name__ == "__main__":
    print("statsmodels", statsmodels.__version__, " pandas", pd.__version__, " numpy", np.__version__)
    for b in ("P", "N"):
        print("\n" + "=" * 30, "block", b, "=" * 30)
        print(C.html(b))

    X = design(c)
    off = np.log(c["py"])
    pf = poisson_glm(X, c["y"], off)
    own = nb_glm(X, c["y"], off)
    assert np.allclose(fit.params.values, pf["beta"], atol=1e-6) and np.allclose(fit.bse.values, pf["se"], atol=1e-6)
    assert np.allclose(nb.params.values[:-1], own["beta"], atol=2e-5) and abs(nb.params["alpha"] - own["alpha"]) < 1e-4
    assert np.allclose(g.bse.values, own["se"], atol=1e-6)      # GLM at fixed alpha = chapter's nb_glm SEs

    print("\n--- Poisson: numbers in text and marks")
    b, s = fit.params["drug[T.A]"], fit.bse["drug[T.A]"]
    print(f"  drug A IRR {np.exp(b):.4f} ({np.exp(b - Z * s):.3f}-{np.exp(b + Z * s):.3f}) z {b / s:.3f} p {fit.pvalues['drug[T.A]']:.3g}")
    print(f"  exp(intercept) {np.exp(fit.params['Intercept']):.4f}; age 65-74 {np.exp(fit.params['agegrp[T.65-74]']):.3f} "
          f"75+ {np.exp(fit.params['agegrp[T.75+]']):.3f} ratio {np.exp(fit.params['agegrp[T.75+]'] - fit.params['agegrp[T.65-74]']):.3f}; "
          f"cci {np.exp(fit.params['cci']):.4f} x3 {np.exp(3 * fit.params['cci']):.4f}")
    print(f"  deviance {fit.deviance:.1f}/{fit.df_resid:.0f} = {fit.deviance / fit.df_resid:.3f}; Pearson {fit.pearson_chi2:.1f}/"
          f"{fit.df_resid:.0f} = {fit.pearson_chi2 / fit.df_resid:.3f}; null {fit.null_deviance:.1f}; LR {fit.null_deviance - fit.deviance:.1f}; "
          f"AIC {fit.aic:.1f}; iterations {fit.fit_history['iteration']}; llf {fit.llf:.3f}")
    q = smf.glm("exac ~ drug + agegrp + female + cci", data=copd, family=sm.families.Poisson(),
                offset=np.log(copd["py"])).fit(scale="X2")
    r = smf.glm("exac ~ drug + agegrp + female + cci", data=copd, family=sm.families.Poisson(),
                offset=np.log(copd["py"])).fit(cov_type="HC0")
    print(f"  quasi-Poisson SE {q.bse['drug[T.A]']:.4f} p {q.pvalues['drug[T.A]']:.4f};  robust HC0 SE {r.bse['drug[T.A]']:.4f} "
          f"p {r.pvalues['drug[T.A]']:.4f} CI {np.exp(r.conf_int().loc['drug[T.A]'].values).round(3)}")

    print("\n--- negative binomial: numbers in text, marks and Table 3")
    ci = np.exp(nb.conf_int())
    for k in nb.params.index[:-1]:
        print(f"  {k:16s} coef {nb.params[k]:.5f} se {nb.bse[k]:.5f} IRR {np.exp(nb.params[k]):.4f} "
              f"({ci.loc[k, 0]:.4f}-{ci.loc[k, 1]:.4f}) p {nb.pvalues[k]:.4g}   [GLM fixed alpha: se {g.bse[k]:.5f}]")
    a, sa = nb.params["alpha"], nb.bse["alpha"]
    print(f"  alpha {a:.4f} se {sa:.4f} CI {nb.conf_int().loc['alpha'].values.round(3)}  theta = 1/alpha {1 / a:.4f}")
    print(f"  llf {nb.llf:.4f} 2llf {2 * nb.llf:.3f} AIC {nb.aic:.2f} LR vs Poisson {2 * (nb.llf - fit.llf):.2f} "
          f"llnull {nb.llnull:.3f}; GLM(fixed alpha) deviance/df {g.deviance / g.df_resid:.4f} Pearson/df {g.pearson_chi2 / g.df_resid:.4f}")
    nbc = smf.negativebinomial("exac ~ drug", data=copd, offset=np.log(copd["py"])).fit(disp=0)
    print(f"  crude NB IRR {np.exp(nbc.params['drug[T.A]']):.4f} ({np.exp(nbc.conf_int().loc['drug[T.A]'].values).round(3)})")
    # null model of smf.negativebinomial keeps the offset?
    n0 = smf.negativebinomial("exac ~ 1", data=copd, offset=np.log(copd["py"])).fit(disp=0)
    print(f"  intercept-only NB with offset llf {n0.llf:.3f} (compare llnull {nb.llnull:.3f})")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        smf.glm("exac ~ drug", data=copd, family=sm.families.NegativeBinomial(), offset=np.log(copd["py"])).fit()
    print("  GLM NegativeBinomial() without alpha ->", [f"{x.category.__name__}: {x.message}" for x in w])
    print("  all statsmodels fits agree with lib_ch12 (assertions passed)")
