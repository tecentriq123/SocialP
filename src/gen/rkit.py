"""rkit — run R cells for real (one R session per chapter) and render them as site HTML, like labkit does for Python.

Usage (inside gen/r_rcNN.py; R lives in /home/claude/renv, installed from conda-forge):

    from rkit import RNotebook
    nb = RNotebook("rc04")
    c1 = nb.cell('''
    library(survival)
    d <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/rcc_trial.csv")
    head(d)
    ''', title="자료 불러오기")
    nb.run()                                   # runs every queued cell, in order, in ONE R session
    nb.save_fragment("rc04_load", nb.html(c1, marks={"T001": 1}))   # -> figs/rc04_load.html
    nb.save_ipynb("4장 …")                     # -> pub/notebooks/rc04.ipynb (R kernel, opens in Colab's R runtime)

What it does
- The code shown keeps the site URL; the run reads the same file from site/pub/data (the sandbox is offline).
- install.packages(...) lines are shown but not executed (the packages are already in the sandbox R).
- Output is what the R console prints: printed values, messages, warnings, errors (expect_error=True to show one).
- Plots a cell draws are embedded as PNG (7 x 4.5 in at 100 dpi unless fig=(w, h) is given).
- marks puts <span class="mk">n</span> right after the first occurrence of each substring in the text output.
"""
import html, json, os, re, subprocess, tempfile, textwrap, base64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
PUB = os.path.join(ROOT, "pub")
RSCRIPT = "/home/claude/renv/bin/Rscript"
SITE_DATA = "https://socialp-ajou.tecentriq12.workers.dev/data/"

DRIVER = r'''
local({
  args <- commandArgs(trailingOnly = TRUE)
  cells <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
  outdir <- args[2]
  options(width = 80, digits = 7, warn = 1, cli.unicode = FALSE, cli.num_colors = 1, crayon.enabled = FALSE)
  res <- list()
  for (i in seq_along(cells)) {
    cl <- cells[[i]]
    ev <- evaluate::evaluate(cl$run, envir = globalenv(), new_device = TRUE, stop_on_error = 1L,
                             output_handler = evaluate::new_output_handler())
    txt <- character(0); warns <- character(0); err <- NULL; imgs <- character(0); k <- 0
    for (it in ev) {
      if (is.character(it)) { txt <- c(txt, it) }
      else if (inherits(it, "source")) { }
      else if (inherits(it, "error")) {
        cl0 <- conditionCall(it)
        err <- if (is.null(cl0)) paste0("Error: ", conditionMessage(it))
               else paste0("Error in ", paste(deparse(cl0), collapse = " "), " : ", conditionMessage(it))
      }
      else if (inherits(it, "warning")) {
        cl0 <- conditionCall(it)
        warns <- c(warns, if (is.null(cl0)) paste0("Warning message:\n", conditionMessage(it))
                          else paste0("Warning message:\nIn ", paste(deparse(cl0), collapse = " "), " :\n  ", conditionMessage(it)))
      }
      else if (inherits(it, "message")) { if (isTRUE(cl$messages)) txt <- c(txt, conditionMessage(it)) }
      else if (inherits(it, "recordedplot")) {
        k <- k + 1
        f <- file.path(outdir, sprintf("cell%02d_%d.png", i, k))
        png(f, width = cl$fig[[1]], height = cl$fig[[2]], units = "in", res = 100, type = "cairo")
        replayPlot(it); dev.off()
        imgs <- c(imgs, f)
      }
    }
    res[[i]] <- list(text = paste(txt, collapse = ""), warns = as.list(warns), error = err, images = as.list(imgs))
    if (!is.null(err) && !isTRUE(cl$expect_error)) { res[[i]]$fatal <- TRUE; break }
  }
  writeLines(jsonlite::toJSON(res, auto_unbox = TRUE, null = "null"), file.path(outdir, "result.json"))
})
'''


class RCell:
    def __init__(self, n, code, title, expect_error, fig, messages, shell_output):
        self.n, self.code, self.title = n, code, title
        self.expect_error, self.fig, self.messages = expect_error, fig, messages
        self.shell_out = (shell_output.rstrip() + "\n") if shell_output else ""
        self.text, self.warns, self.error, self.images = "", [], None, []


class RNotebook:
    def __init__(self, name):
        self.name = name
        self.cells = []
        self.ran = False

    def cell(self, code, title="", expect_error=False, fig=(7, 4.5), messages=True, shell_output=None):
        code = textwrap.dedent(code).strip("\n")
        c = RCell(len(self.cells) + 1, code, title, expect_error, fig, messages, shell_output)
        self.cells.append(c)
        return c

    @staticmethod
    def _runnable(code):
        lines = []
        for ln in code.split("\n"):
            lines.append("invisible(NULL)" if ln.lstrip().startswith("install.packages(") else ln)
        run = "\n".join(lines)
        return run.replace(SITE_DATA, os.path.join(PUB, "data") + "/")

    def run(self):
        with tempfile.TemporaryDirectory(prefix=f"rkit_{self.name}_") as tmp:
            spec = [{"run": self._runnable(c.code), "expect_error": c.expect_error, "fig": list(c.fig),
                     "messages": c.messages} for c in self.cells]
            sp = os.path.join(tmp, "cells.json"); dp = os.path.join(tmp, "driver.R")
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(spec, f, ensure_ascii=False)
            with open(dp, "w", encoding="utf-8") as f:
                f.write(DRIVER)
            env = dict(os.environ, LANG="en_US.UTF-8", LC_ALL="C.UTF-8")
            p = subprocess.run([RSCRIPT, "--vanilla", dp, sp, tmp], capture_output=True, text=True, cwd=tmp, env=env)
            rp = os.path.join(tmp, "result.json")
            if p.returncode != 0 or not os.path.exists(rp):
                raise RuntimeError(f"R driver failed ({p.returncode}):\n{p.stderr[-3000:]}")
            res = json.load(open(rp, encoding="utf-8"))
            for c, r in zip(self.cells, res):
                c.text = c.shell_out + (r.get("text") or "")
                c.warns = r.get("warns") or []
                c.error = r.get("error")
                c.images = [base64.b64encode(open(f, "rb").read()).decode() for f in (r.get("images") or [])]
                if r.get("fatal"):
                    raise RuntimeError(f"{self.name} cell {c.n} ({c.title}) failed:\n{c.code}\n---\n{c.error}")
            if len(res) < len(self.cells):
                raise RuntimeError(f"{self.name}: only {len(res)} of {len(self.cells)} cells ran")
        self.ran = True
        return self

    # ---------- rendering (same markup as labkit) ----------
    def html(self, c, marks=None, title=None, out_label="출력", show_images=True):
        assert self.ran, "call run() first"
        t = html.escape(title if title is not None else c.title)
        head = (f'<div class="cell-h"><span class="cell-n">셀 {c.n}</span><span class="cell-t">{t}</span>'
                f'<button class="copy" type="button" aria-label="코드 복사">복사</button></div>')
        inp = f'<pre class="cell-in"><code class="language-r">{html.escape(c.code)}</code></pre>'
        outs = []
        if c.text.strip():
            outs.append(f"<pre>{_mark(html.escape(c.text.rstrip()), marks)}</pre>")
        for w in c.warns:
            outs.append(f'<pre class="warn-out">{html.escape(w)}</pre>')
        if c.error:
            outs.append(f'<pre class="err-out">{html.escape(c.error)}</pre>')
        if show_images:
            for img in c.images:
                outs.append(f'<img class="cell-img" alt="셀 {c.n} 그래프" src="data:image/png;base64,{img}">')
        out = f'<div class="cell-out"><div class="cell-out-h">{out_label}</div>{"".join(outs)}</div>' if outs else ""
        return f'<div class="cell">{head}{inp}{out}</div>'

    def save_fragment(self, name, html_str):
        os.makedirs(FIGS, exist_ok=True)
        with open(os.path.join(FIGS, name + ".html"), "w", encoding="utf-8") as f:
            f.write(html_str)

    def save_ipynb(self, title, intro="", skip=(), notes=None):
        """pub/notebooks/<name>.ipynb with an R kernel (Colab: 런타임 유형을 R로). Cells without outputs."""
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
        nbj = {"cells": cells, "metadata": {"kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
               "language_info": {"name": "R"}}, "nbformat": 4, "nbformat_minor": 5}
        os.makedirs(os.path.join(PUB, "notebooks"), exist_ok=True)
        path = os.path.join(PUB, "notebooks", f"{self.name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nbj, f, ensure_ascii=False, indent=1)
        return path


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
