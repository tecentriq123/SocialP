# Brief for Part 3 (파이썬 실습) writers

You are writing Python practice chapters ("실습") for a Korean-language statistics learning website for new graduate students in a social-pharmacy lab (pharmacoepidemiology, pharmacoeconomics/HEOR). Project root: /home/claude/site. Parts 1–2 (theory chapters 1–13, `content/ch01.html` … `ch13.html`) are written. Part 3 collects all hands-on practice: **실습 N ↔ N장** (ids `lab00` … `lab13`; lab00 = environment setup).

## What the site owner asked for (their words, paraphrased)
- "우리 연구실은 파이썬을 사용하니까 파이썬 기준으로."
- "비전공자도 실제 실습을 할 수 있게 … 사이킷런에서 위스콘신 유방암 데이터를 받아오듯이 데이터를 받아와서 실습할 수 있었으면."
- "파이썬 코드나 셀도 비전공자가 이해하고 직접 해 볼 수 있게 자세히 설명."
- Their standing preferences for coding explanations: beginner level, **line by line — what each line does and how the data change** (e.g., `predict_proba(X_test)[:, 1]` → what is returned, what rows/columns mean, why `[:, 1]` is the class-1 probability); always say **where to run it** (Google Colab cell, Jupyter cell, Anaconda Prompt for installs), and not just "type this" but **"type this here → this output appears → look at this part of it"**. Main computer is **Windows 11**. Korean 합니다체, statistics terms Korean + English on first use, plain headings (no flashy titles or colon subtitles), accuracy over agreeableness.

## Read first
1. `STYLE.md` (general site style, components, terminology table, and the "2026-09-30 변경 사항" section).
2. The theory chapter(s) your lab accompanies (`content/chNN.html`) — use the same terms, link back to specific sections (`<a href="#ch07-s2">7장 나 절</a>`), and mirror its '논문에서는 이렇게 보입니다' logic by ending analyses with a paper-style reporting sentence built from the real result.
3. `build.py` (the CHAPTERS list defines each lab's exact section titles and order — use ids `labNN-s1`, `labNN-s2`, …), `gen/labkit.py` (the notebook runner — read its docstring), and `content/ch01.html` for the component markup.

## Environment (real code, real output)
- Run everything with `source /home/claude/pylibs/env.sh` (statsmodels 0.15.0, lifelines 0.30.3, scipy 1.17, pandas 3.0, numpy 2.4, scikit-learn 1.8, matplotlib, seaborn). No internet: `gen/labkit.py` maps Rdatasets URLs to offline copies in `site/data/` (birthwt, sleep, cancer (= NCCTG lung; `lung.csv` on Rdatasets is broken — use `survival/cancer.csv`), pbc, cgd, veteran, sleepstudy, epil, esoph, Insurance, Theoph, Indometh; variable docs in `data/doc/*.html`). scikit-learn's bundled datasets (`load_breast_cancer`, `load_diabetes(scaled=False)`, …) need no download.
- Student-facing data code must use the real URL: `pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv")` (pattern `…/csv/<package>/<name>.csv`), or the sklearn loader. Explain once that this needs internet and what the site is (Rdatasets: a public collection of R datasets as CSV).
- Write `gen/lab_labNN.py` that builds each cell with `labkit.Notebook`, runs it for real, and saves HTML fragments (`nb.save_fragment("labNN_xxx", nb.html(cell, marks={...}))` → `figs/labNN_xxx.html`), which you insert in `content/labNN.html` with `<!--FIG:labNN_xxx-->`. **Never hand-type outputs.** Re-run the script after any code change.
- Colab/other machines may run pandas 2.x and statsmodels 0.14: write code that works on both (e.g., set category order explicitly with `pd.Categorical(..., categories=[...])`, don't depend on the `str` vs `object` dtype, avoid APIs added only in the newest versions). In lab00 (and briefly where relevant) say that outputs were produced with the versions above and small formatting differences are normal.
- Plots: use **English axis labels/titles** in matplotlib (Korean text shows as boxes in Colab unless a Korean font is installed); explain the labels in Korean in the text. Keep 1–3 plots per lab section.
- Set random seeds wherever randomness appears.

## Page structure for each lab
```html
<p class="lead">무엇을 하는지 2–3문장 + 관련 장 링크</p>
<dl class="glance">
  <div><dt>관련 이론</dt><dd><a href="#ch07">7장 …</a></dd></div>
  <div><dt>데이터</dt><dd>이름, 출처, 몇 명·몇 변수, 연구 배경 한 줄</dd></div>
  <div><dt>사용 패키지</dt><dd>pandas, scipy, statsmodels, lifelines …</dd></div>
  <div><dt>실행 환경</dt><dd>Google Colab 권장 (실습 0 참고). lifelines는 첫 셀에서 설치</dd></div>
</dl>
<section class="sec" id="labNN-s1"><h2><span class="sec-no">가.</span>제목</h2> … </section>
```
Within each section, for each step:
1. A short paragraph: **why** we do this step (link to the theory section).
2. The cell (`<!--FIG:labNN_xxx-->`, produced by labkit: numbered "셀 n", title, copy button, real output). Keep cells short (≈3–12 lines) with brief Korean `#` comments.
3. **코드 한 줄씩** — `<ol class="lines"><li><code>코드 조각</code><span>설명: 무엇을 하는지, 데이터(행·열·자료형)가 어떻게 바뀌는지, 괄호 안 인자의 뜻</span></li>…</ol>`. Explain every non-trivial line; for repeated patterns later in the lab, explain only what is new.
4. **출력 읽기** — marks in the output (`marks={"text in output": n}`) with `<ol class="marks">` explanations: which number to look at, what it means, how it maps to the theory chapter and to what a paper would report.
5. Where useful: `<div class="callout tip"><p class="ct">직접 바꿔 보기</p>…</div>` (a small variation to try and what should change), `<div class="callout warn"><p class="ct">이런 오류가 나면</p>…</div>` (typical errors: `ModuleNotFoundError`, `KeyError` from a column-name typo, wrong event coding like `status == 2`, etc.; you can show a real error with `expect_error=True`).
6. End each analysis with **"논문에는 이렇게 씁니다"**: a `<div class="paper">` box containing an English Results/Methods sentence built from the real numbers, with marks explained (reuse the paper-box markup from the theory chapters).
7. End each section with the `<div class="keypoints">` 정리.

Also: `<details class="code">` is not used in labs (cells replace it). Do not add R code anywhere. Cite references only if needed (dataset origin papers, if you are certain of details; `refs_add/labNN.py`, same format as other refs_add files).

## Mechanics
- Write only your own files: `content/labNN.html`, `gen/lab_labNN.py`, `figs/labNN_*.html`, optional `refs_add/labNN.py`. Do not edit shared files (`shell.html`, `build.py`, `refs.py`, `svgplot.py`, `gen/labkit.py`, STYLE/briefs, theory chapters). If labkit needs a feature, work around it in your own script and mention it in your report.
- Verify: `python3 build.py --only labNN --out /tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<yourdir>/test.html` must show no warnings (it warns if a section id from build.py is missing); `python3 tools/preview.py labNN <scratch dir>` and with `--mobile --dark`; open the PNGs (cells, lines, papers) with Read and fix layout problems (long code lines scroll inside the cell — keep lines ≤ 70 characters where possible).
- Check statistical interpretations carefully (they must agree with the theory chapters). Final reply (under 250 words): files, number of cells/figures, datasets used, anything uncertain (e.g., version-dependent output), requests for shared-file changes.

## Note (2026-09-30)
`build.py` automatically folds every `<ol class="lines">` (and a directly preceding `<h4>코드 한 줄씩</h4>`) into a closed `<details class="lines-box">` box ("코드 한 줄씩 설명 · 눌러서 펼치기"). Keep writing `<h4>코드 한 줄씩</h4>` + `<ol class="lines">` as before; do not add your own `<details>`.
