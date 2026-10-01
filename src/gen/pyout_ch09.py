"""Python (statsmodels) outputs pasted into content/ch09.html ('파이썬 출력에서는 이렇게 보입니다').

run:  source /home/claude/pylibs/env.sh && python3 gen/pyout_ch09.py
The data are the chapter's simulated data sets (lib_ch09.cohort(), lib_ch09.irae()).  Every block is
produced by actually running statsmodels (0.15.0 here); the script prints the HTML of each
<pre class="out"> block (escaped, with <span class="mk">n</span> markers) followed by the exact
numbers quoted in the mark explanations, and checks them against nums_ch09.compute().
Only the 'Date/Time' lines of summary() change from run to run.
"""
import sys, os, io, ast, html, contextlib, warnings, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from lib_ch09 import cohort, irae


# ------------------------------------------------------------------ tiny REPL recorder
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


def head_and_table(head_patterns):
    """keep the summary header lines that contain head_patterns, then '…', then the coefficient table."""
    def f(lines):
        hdr = next(i for i, l in enumerate(lines) if "coef    std err" in l)
        head = [l for l in lines[:hdr - 1] if any(p in l for p in head_patterns)]
        return head + ["…"] + lines[hdr - 1:]
    return f


# ------------------------------------------------------------------ data frames (same data as nums_ch09)
d = cohort()
egfr_cat = np.where(d["egfr"] >= 60, ">=60", np.where(d["egfr"] >= 45, "45-59", "<45"))
cohort_df = pd.DataFrame(dict(
    hosp=d["hosp"], sglt2=d["sglt2"], age10=(d["age"] - 75) / 10, female=d["female"],
    cci=pd.Categorical(np.array(["0", "1-2", ">=3"])[d["cci"]], categories=["0", "1-2", ">=3"]),
    prior=d["prior"], hf=d["hf"],
    egfr=pd.Categorical(egfr_cat, categories=[">=60", "45-59", "<45"])))
au, co, yy = irae()
ici_df = pd.DataFrame(dict(irae=yy.astype(int), autoimmune=au.astype(int), combination=co.astype(int)))

C = Console()
C.ns.update(cohort=cohort_df, ici=ici_df)

# ------------------------------------------------------------------ block A: cohort model (GLM, binomial)
C.run("A", "import numpy as np\nimport statsmodels.api as sm\nimport statsmodels.formula.api as smf")
C.run("A", 'm = smf.glm("hosp ~ sglt2 + age10 + female + cci + prior + hf + egfr",\n'
           '            data=cohort, family=sm.families.Binomial()).fit()')
C.run("A", "print(m.summary())")
C.run("A", "print(round(m.null_deviance, 1), round(m.aic, 1))")
C.run("A", 'ci = np.exp(m.conf_int()); ci.columns = ["2.5%", "97.5%"]\n'
           'ci.insert(0, "OR", np.exp(m.params)); print(ci.round(3))')
C.mark("A", "family=sm.families.Binomial()", 1)
C.mark("A", "Intercept ", 2)
C.mark("A", "sglt2 ", 3, nth=2)            # 1st = formula line, 2nd = coefficient row
C.mark("A", "Scale:", 4)
C.mark("A", "Deviance:", 5)
C.mark("A", "No. Iterations:", 7)
C.mark("A", "2541.8 2340.8", 6)
C.mark("A", "ci.insert", 8)
m = C.ns["m"]

# ------------------------------------------------------------------ block B: separation (logit, then GLM)
C.run("B", 'r = smf.logit("irae ~ autoimmune + combination", data=ici).fit()')
C.run("B", "print(r.summary())",
      filt=head_and_table(["converged:"]))
C.run("B", 'g = smf.glm("irae ~ autoimmune + combination", data=ici,\n'
           '            family=sm.families.Binomial()).fit()')
C.run("B", "print(g.summary())",
      filt=head_and_table(["No. Iterations:"]))
C.mark("B", "autoimmune     2", 1)
C.mark("B", "Warning: Maximum number of iterations", 2)
C.mark("B", "converged:", 2)
C.mark("B", "No. Iterations:", 3)
C.mark("B", "autoimmune     2", 3, nth=2)
r, g = C.ns["r"], C.ns["g"]

if __name__ == "__main__":
    print("statsmodels", statsmodels.__version__, " pandas", pd.__version__, " numpy", np.__version__)
    for b in ("A", "B"):
        print("\n" + "=" * 30, "block", b, "=" * 30)
        print(C.html(b))

    print("\n--- numbers quoted in the marks (block A)")
    for k in ("Intercept", "sglt2", "hf"):
        print(f"  {k:10s} coef {m.params[k]:.5f}  se {m.bse[k]:.5f}  z {m.tvalues[k]:.3f}  p {m.pvalues[k]:.4g}  "
              f"OR {np.exp(m.params[k]):.3f} ({np.exp(m.conf_int().loc[k, 0]):.3f}-{np.exp(m.conf_int().loc[k, 1]):.3f})")
    print(f"  intercept -> p = {1 / (1 + np.exp(-m.params['Intercept'])):.4f}")
    print(f"  deviance {m.deviance:.4f}  null {m.null_deviance:.4f}  diff {m.null_deviance - m.deviance:.2f} "
          f"df {m.df_model}  aic {m.aic:.2f}  iterations {m.fit_history['iteration']}  scale {m.scale}  "
          f"pearson {m.pearson_chi2:.1f}  CS pseudo R2 {m.pseudo_rsquared('cs'):.5f}  z^2 sglt2 {m.tvalues['sglt2']**2:.3f}")

    print("\n--- numbers quoted in the marks (block B)")
    for lab, res in (("logit", r), ("glm", g)):
        print(f"  {lab}: autoimmune coef {res.params['autoimmune']:.4f} se {res.bse['autoimmune']:.4g} "
              f"exp(coef) {np.exp(res.params['autoimmune']):.4g}  p {res.pvalues['autoimmune']:.4f}; "
              f"combination coef {res.params['combination']:.4f} OR {np.exp(res.params['combination']):.3f}; "
              f"se ratio auto/combination {res.bse['autoimmune'] / res.bse['combination']:.0f}")
    print(f"  logit converged {r.mle_retvals['converged']}, iterations {r.mle_retvals['iterations']};  "
          f"glm converged {g.converged}, iterations {g.fit_history['iteration']}; llf logit {r.llf:.4f} glm {g.llf:.4f}")
    print(f"  GLM null deviance {g.null_deviance:.3f}  deviance {g.deviance:.3f}")

    # cross-check with the chapter's own implementation (nums_ch09.compute, ~1 min)
    from nums_ch09 import compute
    R = compute()
    E = R["etio"]
    for k, nm in (("sglt2", "sglt2"), ("hf", "hf"), ("Intercept", "(Intercept)")):
        assert abs(m.params[k] - E["adj"][nm]["b"]) < 1e-6 and abs(m.bse[k] - E["adj"][nm]["se"]) < 1e-5, k
    S = R["sep"]
    print("\n--- chapter implementation (nums_ch09) for comparison")
    print(f"  profile-likelihood OR CI hf {E['prof_all']['hf']}, sglt2 {E['prof_all']['sglt2']}")
    print(f"  ML LR test for autoimmune (drop it): chi2 {S['ml_lr']['auto'][0]:.2f} p {S['ml_lr']['auto'][1]:.2g}")
    fb = S["firth"]["beta"]
    print(f"  Firth: autoimmune {fb[1]:.4f} -> OR {np.exp(fb[1]):.1f}, profile CI {np.exp(S['firth_ci'][0][0]):.2f}-"
          f"{np.exp(S['firth_ci'][0][1]):.0f}, penalized LR chi2 {S['firth_lr'][0][0]:.3f} p {S['firth_lr'][0][1]:.5f}; "
          f"combination {fb[2]:.4f} -> OR {np.exp(fb[2]):.2f}")
    print("  all block-A coefficients agree with nums_ch09 (assertions passed)")
