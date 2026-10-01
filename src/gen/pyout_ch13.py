"""Python (scipy) output pasted into content/ch13.html, 나 절 ('파이썬 출력에서는 이렇게 보입니다').

run:  source /home/claude/pylibs/env.sh && python3 gen/pyout_ch13.py
The ITT summary statistics of the telepharmacy PDC trial (nums_ch13.py: remote 83.9 +/- 16.8, n 304;
in-person 85.2 +/- 17.5, n 304) are turned into individual-level data with exactly these means and SDs
(left-skewed PDC-like values inside 0-100, seed 1302), and scipy.stats.ttest_ind is then actually run.
The script prints the HTML of the <pre class="out"> block and checks it against the Welch formulas
used in nums_ch13.py.
"""
import sys, os, io, ast, html, contextlib, warnings, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy
from scipy import stats


def pdc_sample(rng, n, mean, sd, q=0.35, a=1.0, b=2.5):
    """left-skewed values in [0, 100] rescaled to exactly this mean and SD (ddof=1)."""
    for _ in range(2000):
        gap = np.where(rng.random(n) < q, 0.0, 100 * rng.beta(a, b, n))   # days not covered, %
        x = 100 - gap
        z = (x - x.mean()) / x.std(ddof=1)
        y = mean + sd * z
        if y.min() >= 0 and y.max() <= 100:
            return y
    raise RuntimeError("no admissible sample")


rng = np.random.default_rng(1302)
remote = pdc_sample(rng, 304, 83.9, 16.8)
inperson = pdc_sample(rng, 304, 85.2, 17.5)


class Console:
    """Runs code like the Python prompt and records a transcript (same helper as pyout_ch09/12)."""

    def __init__(self):
        self.ns = {}
        self.rows = []

    def run(self, src):
        lines = src.strip("\n").split("\n")
        depth = 0
        for l in lines:
            cont = depth > 0 or l[:1].isspace()
            self.rows.append(["in", ("... " if cont else ">>> ") + l, []])
            depth += sum(l.count(c) for c in "([{") - sum(l.count(c) for c in ")]}")
        buf = io.StringIO()
        tree = ast.parse("\n".join(lines))
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        with warnings.catch_warnings(record=True), contextlib.redirect_stdout(buf):
            exec(compile(tree, "<stdin>", "exec"), self.ns)
            if last is not None:
                val = eval(compile(last, "<stdin>", "eval"), self.ns)
                if val is not None:
                    print(repr(val))
        for l in (buf.getvalue().rstrip("\n").split("\n") if buf.getvalue().strip() else []):
            self.rows.append(["out", l, []])

    def mark(self, substr, n):
        for row in self.rows:
            if substr in row[1]:
                row[2].append(n)
                return
        raise KeyError(substr)

    def html(self, pad=1):
        out = []
        for kind, text, mks in self.rows:
            t = html.escape(text, quote=False)
            if kind == "in":
                t = f'<span class="cm">{t}</span>'
            if mks:
                t += " " * pad + " ".join(f'<span class="mk">{m}</span>' for m in mks)
            out.append(t)
        return "<pre class=\"out\">" + "\n".join(out) + "</pre>"


C = Console()
C.ns.update(remote=remote, inperson=inperson)
C.run("from scipy import stats")
C.run('res = stats.ttest_ind(remote + 5, inperson, equal_var=False,\n'
      '                      alternative="greater")')
C.run("res")
C.run("print(round(res.statistic, 4), round(res.pvalue, 6), round(res.df, 2))")
C.run("ci = res.confidence_interval(confidence_level=0.975)")
C.run("print(ci.low - 5, ci.high - 5)")
C.run("print(round(remote.mean(), 1), round(inperson.mean(), 1))")
C.mark('alternative="greater")', 1)
C.mark("print(round(res.statistic", 2)
C.mark("print(ci.low - 5", 3)
C.mark("print(round(remote.mean()", 4)
res = C.ns["res"]

if __name__ == "__main__":
    print("scipy", scipy.__version__, " numpy", np.__version__)
    print(C.html())
    # checks against the summary-statistic formulas of nums_ch13.py
    v1, v2 = 16.8 ** 2 / 304, 17.5 ** 2 / 304
    se = math.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / 303 + v2 ** 2 / 303)
    t = (-1.3 + 5) / se
    assert abs(res.statistic - t) < 1e-9 and abs(res.df - df) < 1e-6
    assert abs(res.pvalue - stats.t.sf(t, df)) < 1e-12
    print(f"\nWelch by formula: t {t:.4f} df {df:.2f} p {stats.t.sf(t, df):.6f} SE {se:.4f}")
    print(f"data: remote {remote.mean():.4f} +/- {remote.std(ddof=1):.4f} range {remote.min():.1f}-{remote.max():.1f};"
          f" in-person {inperson.mean():.4f} +/- {inperson.std(ddof=1):.4f} range {inperson.min():.1f}-{inperson.max():.1f}")
    print("default 0.95 one-sided lower bound:", res.confidence_interval().low - 5)
    two = stats.ttest_ind(remote, inperson, equal_var=False)
    print("two-sided difference test:", two.statistic, two.pvalue, " 95% CI", two.confidence_interval())
    from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind
    print("statsmodels ttest_ind(value=-5):", sm_ttest_ind(remote, inperson, alternative="larger", usevar="unequal", value=-5))
