"""labkit — run Python lab cells for real and render them as site HTML.

Usage (inside gen/lab_labNN.py; run with `source /home/claude/pylibs/env.sh`):

    from labkit import Notebook
    nb = Notebook("lab02")
    c = nb.cell('''
    import pandas as pd
    url = "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv"
    df = pd.read_csv(url)
    df.head()
    ''', title="데이터 불러오기")
    html = nb.html(c, marks={"2523": 1})     # <div class="cell"> … real input + real output
    nb.save_fragment("lab02_load", html)     # -> figs/lab02_load.html  (insert with <!--FIG:lab02_load-->)

What it does
- Executes cells in ONE shared namespace, in order (like a Jupyter kernel).
- Rewrites Rdatasets URLs (…/Rdatasets/csv/<pkg>/<name>.csv) to the offline copies in site/data/<name>.csv,
  so the code shown to students keeps the real URL while the sandbox runs offline.
- Captures print() output, the value of the last expression (like Jupyter's Out[]), warnings, and every
  matplotlib figure the cell creates (embedded as PNG data URIs).
- DataFrames / Series shown as the last expression are rendered as an HTML table (as Jupyter does).
- `marks` puts <span class="mk">n</span> right after the first occurrence of each substring in the text output.
- Raises if a cell errors, unless expect_error=True (then the traceback's last line is shown, for teaching).
"""
import ast, base64, contextlib, html, io, os, re, sys, traceback, warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Korean-capable font for figures, if present
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
    try:
        from matplotlib import font_manager as _fm
        if any(_f in x.name for x in _fm.fontManager.ttflist):
            plt.rcParams["font.family"] = _f
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

_URL = re.compile(r"https?://vincentarelbundock\.github\.io/Rdatasets/csv/[^/]+/([^/]+\.csv)")
# Site-hosted synthetic data for PART 5 B labs (2026-10-04): the code shown to students reads
#   https://socialp-ajou.tecentriq12.workers.dev/data/<name>.csv
# and the sandbox reads the same file from site/pub/data/<name>.csv (deploy.py publishes pub/data as /data).
SITE_DATA_URL = "https://socialp-ajou.tecentriq12.workers.dev/data/"
PUB = os.path.join(ROOT, "pub")
_URL2 = re.compile(r"https?://socialp-ajou\.tecentriq12\.workers\.dev/data/([^/]+\.csv)")
_orig_read_csv = pd.read_csv


def _read_csv(path, *a, **k):
    if isinstance(path, str):
        m = _URL.match(path)
        if m:
            local = os.path.join(DATA, m.group(1))
            if not os.path.exists(local):
                raise FileNotFoundError(f"offline copy missing: {local}")
            path = local
        m2 = _URL2.match(path)
        if m2:
            local = os.path.join(PUB, "data", m2.group(1))
            if not os.path.exists(local):
                raise FileNotFoundError(f"site data file missing: {local}")
            path = local
    return _orig_read_csv(path, *a, **k)


pd.read_csv = _read_csv


class Cell:
    def __init__(self, n, code, title):
        self.n, self.code, self.title = n, code, title
        self.stdout = ""
        self.value_repr = None
        self.value_html = None
        self.images = []
        self.warns = []
        self.error = None


class Notebook:
    def __init__(self, lab):
        self.lab = lab
        self.ns = {"__name__": "__main__"}
        self.cells = []

    def cell(self, code, title="", expect_error=False, show_warnings=True, max_rows=10, shell_output=None):
        """shell_output: text to show for notebook shell lines (!pip ...), which are NOT executed here."""
        code = _dedent(code)
        c = Cell(len(self.cells) + 1, code, title)
        self.cells.append(c)
        # notebook shell/magic lines (!pip install ..., %matplotlib inline) are shown but not executed
        run_code = "\n".join("pass" if ln.lstrip().startswith(("!", "%")) else ln for ln in code.split("\n"))
        tree = ast.parse(run_code)
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        buf = io.StringIO()
        plt.close("all")
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("default")
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(tree, f"<{self.lab}-cell{c.n}>", "exec"), self.ns)
                    if last_expr is not None:
                        val = eval(compile(last_expr, f"<{self.lab}-cell{c.n}>", "eval"), self.ns)
                        if val is not None:
                            self._set_value(c, val, max_rows)
            except Exception as e:  # noqa
                if not expect_error:
                    raise
                c.error = "".join(traceback.format_exception_only(type(e), e)).strip()
        c.stdout = (shell_output.rstrip() + "\n" if shell_output else "") + buf.getvalue()
        if show_warnings:
            seen = set()
            for w in wl:
                # environment-only noise (this sandbox runs pandas 3 / statsmodels dev builds); students won't see these
                if w.category.__name__ in ("Pandas4Warning", "DeprecationWarning", "PendingDeprecationWarning"):
                    continue
                msg = f"{w.category.__name__}: {w.message}"
                if msg not in seen:
                    seen.add(msg)
                    c.warns.append(msg)
        for num in plt.get_fignums():
            fig = plt.figure(num)
            b = io.BytesIO()
            fig.savefig(b, format="png", bbox_inches="tight", dpi=100)
            c.images.append(base64.b64encode(b.getvalue()).decode())
        plt.close("all")
        return c

    def _set_value(self, c, val, max_rows):
        if isinstance(val, (pd.DataFrame, pd.Series)):
            df = val.to_frame() if isinstance(val, pd.Series) else val
            c.value_html = df.to_html(max_rows=max_rows, classes="dfout", border=0)
            c.value_repr = repr(val)
        else:
            c.value_repr = repr(val)

    # ---------- rendering ----------
    def html(self, c, marks=None, title=None, show_code=True, out_label="출력"):
        t = html.escape(title if title is not None else c.title)
        head = (f'<div class="cell-h"><span class="cell-n">셀 {c.n}</span><span class="cell-t">{t}</span>'
                f'<button class="copy" type="button" aria-label="코드 복사">복사</button></div>')
        inp = f'<pre class="cell-in"><code class="language-python">{html.escape(c.code)}</code></pre>' if show_code else ""
        outs = []
        text = c.stdout
        if c.value_repr is not None and c.value_html is None:
            text = (text + ("\n" if text and not text.endswith("\n") else "") + c.value_repr)
        if text.strip():
            outs.append(f"<pre>{_mark(html.escape(text.rstrip()), marks)}</pre>")
        if c.value_html is not None:
            outs.append(f'<div class="df-wrap">{_mark(c.value_html, marks)}</div>')
        for w in c.warns:
            outs.append(f'<pre class="warn-out">{html.escape(w)}</pre>')
        if c.error:
            outs.append(f'<pre class="err-out">{html.escape(c.error)}</pre>')
        for img in c.images:
            outs.append(f'<img class="cell-img" alt="셀 {c.n} 그래프" src="data:image/png;base64,{img}">')
        out = f'<div class="cell-out"><div class="cell-out-h">{out_label}</div>{"".join(outs)}</div>' if outs else ""
        return f'<div class="cell">{head}{inp}{out}</div>'

    def save_ipynb(self, title, intro="", skip=(), notes=None):
        """Write pub/notebooks/<lab>.ipynb (published as /notebooks/<lab>.ipynb): one markdown title cell, then
        every executed cell as a code cell WITHOUT outputs, each preceded by a markdown cell '### 셀 n. <title>'.
        skip: cell numbers to leave out (e.g. expect_error demos). notes: {cell number: markdown text shown
        before that cell} for short Korean guidance (section headings, exercise text)."""
        import json as _json
        notes = notes or {}
        cells = [{"cell_type": "markdown", "metadata": {},
                  "source": (f"# {title}\n\n{intro}".strip() + "\n").splitlines(keepends=True)}]
        for c in self.cells:
            if c.n in skip:
                continue
            md = (notes.get(c.n, "") + "\n\n" if notes.get(c.n) else "") + f"### 셀 {c.n}. {c.title}".rstrip(". ")
            cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": c.code.splitlines(keepends=True)})
        nbj = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
               "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        os.makedirs(os.path.join(PUB, "notebooks"), exist_ok=True)
        path = os.path.join(PUB, "notebooks", f"{self.lab}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(nbj, f, ensure_ascii=False, indent=1)
        return path

    def save_fragment(self, name, html_str):
        os.makedirs(FIGS, exist_ok=True)
        with open(os.path.join(FIGS, name + ".html"), "w", encoding="utf-8") as f:
            f.write(html_str)


def _dedent(code):
    import textwrap
    return textwrap.dedent(code).strip("\n")


def _mark(escaped, marks):
    if not marks:
        return escaped
    for sub, n in marks.items():
        e = html.escape(sub)
        i = escaped.find(e)
        if i < 0:
            raise ValueError(f"mark text not found in output: {sub!r}")
        j = i + len(e)
        escaped = escaped[:j] + f'<span class="mk">{n}</span>' + escaped[j:]
    return escaped
