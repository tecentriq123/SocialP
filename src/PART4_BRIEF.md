# Brief: write a PART 4 chapter (약물경제성 평가, chapters 20–25) — 2026-10-02

Site: Korean study notes for new graduate students in a social-pharmacy lab (source in /home/claude/site, built by `python3 build.py` → dist/stats.html). Readers are pharmacists with little statistics or economics background. The owner (a pharmacoeconomics graduate student) wants them to **read** economic evaluation papers correctly, **build and report** a simple model, and **run it in Python**. PART 4 is new: 20 경제성 평가의 틀, 21 비용 자료 분석, 22 효용과 QALY, 23 결정분석 모형, 24 불확실성 분석, 25 경제성 평가 논문 읽기. Chapter and section titles are fixed in `build.py` CHAPTERS (section ids `chNN-s1…`, letters 가, 나, …).

Owner's decisions for PART 4:
- **Standard = 건강보험심사평가원 「의약품 경제성 평가 지침」 (the Korean guideline), plus CHEERS 2022 for reporting. Where international practice differs (NICE, ISPOR good practice, US Second Panel), point out the difference briefly** — one sentence or a small table, not a parallel treatment.
- Models: decision tree → Markov cohort model → **partitioned survival model** → probabilistic sensitivity analysis. No microsimulation (one sentence saying it exists is enough).
- Lean and intuitive: intuition first with a numeric example, few formulas, everything not needed on first reading folded.

## Read first
- `STYLE.md` (terminology, markup, no R, Python only) and `FOLD_BRIEF.md` (folding: `심화` tag on h3/h4/boxes, `<!--FOLD:title-->…<!--/FOLD-->`, '수식으로 보기' auto-folded).
- One finished model chapter in full: `content/ch17.html` (short PART 3 chapter with every component: route box, 쉽게 말하면, glance, 숫자로 따라가기, 흔한 오해, 내 연구에 쓸 때, 논문에서 읽어 보기 with numbered markers, 수식으로 보기, 정리, 스스로 확인하기). Skim `content/ch19.html` section 마 (통합 결과의 활용: NNT·NNH와 경제성 평가) and `content/ap01.html` section 다 (의료비용 자료의 분석) so that you build on them instead of repeating them.
- `P4_FACTS.md`: the verified fact sheet for PART 4 (Korean guideline items, international comparison, value sets, reporting standards, reference keys in `refs_add/p4.py`). **Guideline facts in your chapter must agree with it.** If you need a fact that is not there, verify it yourself (official or peer-reviewed source via WebSearch/WebFetch, PubMed tools) or stay general. Never state a guideline number from memory.
- `gen/lib_p4.py`: the shared example model (below). Run `python3 gen/lib_p4.py`.

## The shared example (use it in every chapter)
A hypothetical reimbursement submission: **진행성 신세포암 1차 치료, 신약 A 대 표준요법 B** (always labelled 가상의 예시; never name a real drug). Three health states: 무진행(PF), 진행(PD), 사망. Monthly cycle, 20-year horizon, costs in 만원, 보건의료체계 관점, discount 4.5% per year for costs and effects. Base case from `gen/lib_p4.py` (`run()`):

| | 신약 A | 표준요법 B | 증분 |
|---|---|---|---|
| 총비용(만원) | 10,562 | 7,689 | 2,872 |
| 생존연수(할인 전) | 3.916 | 3.054 | |
| QALY(할인 후) | 2.385 | 1.874 | 0.511 |
| ICER | | | 5,621만원/QALY |

Assumed threshold in the examples: 5,000만원/QALY (always written as "이 예시에서 가정한 임계값"; Korea has no official explicit threshold — see P4_FACTS.md). At that threshold A is not cost-effective at the list price (증분 순편익 −317만원), the probability of being cost-effective in the PSA is about 16%, and ICER falls to the threshold at a monthly price of about 175만원 (list 190만원). This story — a drug slightly above the threshold, uncertainty, price negotiation — runs through chapters 20, 23, 24, 25.
- Import the numbers (`sys.path.insert(0, "gen"); import lib_p4 as L`); never retype or re-derive them differently. **Do not change `base_params()`, `PSA_SPEC` or the existing functions.** (2026-10-03: the PSA sampling was revised once, without touching `base_params()` or the base case — curve uncertainty is now drawn as median and shape, PFS/OS pairs are drawn with correlation 0.5 (`PSA_CORR`), and parameter sets whose PFS curve exceeds the OS curve are redrawn; see P4_FACTS.md and the header of `gen/lib_p4.py`.) Chapter 23 may add functions (decision tree, Markov version of the same disease, parametric fits); chapter 24 may add functions (one-way analysis, CEAC, EVPI). Other chapters do not edit the file.
- Chapters 21 and 22 use their own simulated data set in the same clinical story (ch21: 청구자료에서 뽑은 두 군의 1년 의료비; ch22: 임상시험에서 반복 측정한 EQ-5D) — their means should be compatible with the lib inputs (e.g., utility 0.78 in PF, 0.62 in PD; monthly PD cost about 250만원) but need not reproduce them exactly; say that the model uses rounded inputs.
- Chapter 20 section 라 needs a 3–4 strategy example for dominance and extended dominance: invent extra comparators (e.g., 표준요법 B, 신약 A, 기존약 C, 병용요법 D) around the A-vs-B numbers above.

## Chapter structure (same as PART 3)
1. `<p class="lead">` (2–3 sentences) + `<div class="route">` '처음 읽을 때' box (미리 알아 둘 것 with links, 먼저 읽을 부분, 나중에 읽을 부분, 이 장을 마치면 ①–④) — same markup as ch17.
2. Each section (`<section class="sec" id="chNN-sK">`, `<h2><span class="sec-no">가.</span>Title</h2>` exactly as in build.py):
   - `<div class="easy">` 쉽게 말하면 (gist, everyday or pharmacy analogy, how it appears in a paper, one caution) and `<dl class="glance">`.
   - Explanation subsections (h3). '숫자로 따라가기': setup and final result visible, intermediate arithmetic in a FOLD ("계산 과정 보기"). Tables and figures where they help (figures with `svgplot.py`, see `gen/fig_ch17.py`; captions "그림 NN-k." in order of appearance; "표 NN-k." if captioned). Typical figures: cost-effectiveness plane, efficiency frontier, cost histogram, state-transition diagram, cohort trace, survival curves with extrapolation, tornado diagram, PSA scatter, CEAC.
   - `<div class="callout warn">` 흔한 오해 with `<span class="wrong">…</span>` statements. Errors to correct actively in PART 4: "ICER가 낮을수록 항상 좋은 약" (negative ICERs, dominance); average cost-effectiveness ratio read as ICER; "비용효과적 = 비용 절감"; comparing costs by medians or Mann–Whitney; log-transformed cost back-transformed without smearing; charges read as costs; utility read as a 0–100 quality-of-life score or as a probability; QALY gain = life-years gained; rate used as a probability; forgetting discounting or half-cycle correction; Markov "memoryless" ignored; PFS gain assumed to equal OS gain; best statistical fit (AIC) taken as the right extrapolation; one-way sensitivity analysis taken as showing overall uncertainty; CEAC read as "probability the drug is effective" or its 50% point read as the ICER; "95% CI of ICER" interpreted naively when it spans quadrants; CHEERS used as a quality score.
   - `<div class="callout use">` 내 연구에 쓸 때: what to decide before analysing (per the Korean guideline), what to report (Methods/Results wording, CHEERS 2022 item), the difference from international practice in one or two sentences, and the Python tools (numpy/scipy/pandas/statsmodels/lifelines functions you have **run**; "직접 구현" where none exists — there is no standard Python package for decision models, so models are written with numpy arrays).
   - `<h3>논문에서 읽어 보기</h3>`: at least one '논문에서는 이렇게 보입니다' paper box per section where a paper would show it (hypothetical, labelled 가상의 예시; English journal-style table or Methods/Results excerpt; `<span class="mk">n</span>` BEFORE each highlighted phrase; every marker explained for a beginner in `<ol class="marks">`). It comes after the explanations it relies on.
   - `<h3>수식으로 보기</h3>` + `<p class="math-note">` + a few `<div class="eq">` formulas with the example plugged in (auto-folded). Keep it short.
   - `<div class="keypoints">` 정리 (3–5 bullets), then `<div class="practice">` 스스로 확인하기 (2–3 questions; types 계산 / 해석 고치기 / 분석 고르기 / 논문 읽기; answers in `<details><summary>답과 풀이</summary><div>…</div></details>`).
3. **Python inside the chapter**: where the reader should see how a result is produced (Markov trace, PSM areas, PSA loop, gamma GLM, QALY area), show a short runnable code block in a FOLD ("파이썬으로 계산해 보기") with the real output (run it; never hand-type output) and 3–6 line-by-line notes for a beginner. Full step-by-step labs come later in PART 5 — do not write a lab and do not add lab links.
4. **Length**: visible text (folds closed) about 5,000–8,000 characters per section; derivations, variants, edge cases, long arithmetic and code go into folds. Nothing visible may depend on a term explained only inside a fold.
5. Terms Korean + English on first visible use (증분비용효과비(incremental cost-effectiveness ratio, ICER), 질보정생존연수(quality-adjusted life year, QALY), 효용(utility), 마르코프 모형(Markov model), 분할생존모형(partitioned survival model), 확률적 민감도 분석(probabilistic sensitivity analysis, PSA), 비용효과 수용곡선(cost-effectiveness acceptability curve, CEAC), 순금전편익(net monetary benefit, NMB), 재정영향분석(budget impact analysis)). Plain headings: no colons, no catchy titles. 합니다체.

## Accuracy
- Every number in text, tables, paper boxes, figures and practice answers is computed by your `gen/nums_chNN.py` (fixed seeds; `source /home/claude/pylibs/env.sh`: numpy, scipy, pandas, statsmodels, lifelines, scikit-learn). Paper boxes must be internally consistent (increments equal differences of the arms shown, ICER = Δcost/ΔQALY of the rounded table to within rounding, CI contains estimate, percentages add up).
- Citations: `<cite data-ref="key"></cite>`; reuse existing keys (`refs.py`, `refs_add/*.py`, especially `refs_add/p4.py`); add new ones in `refs_add/chNN.py` (same dict format; **verify each on PubMed** — ToolSearch "PubMed": `mcp__PubMed__lookup_article_by_citation`, `get_article_metadata` — or the publisher page; never from memory alone). Cite methodological sources for key claims.
- Cross-references: 6장 회귀, 7장 Kaplan–Meier, 8장 라 절 일반화 선형모형 (`#ch08-s4`), 11장 Cox, 14장 청구자료, 15장 성향점수, 19장 메타분석 (마 절 `#ch19-s5`), 부록 A 다 절 의료비용 (`#ap01-s3`); within PART 4: 20 틀, 21 비용, 22 효용·QALY, 23 모형, 24 불확실성, 25 논문 읽기 (links like `<a href="#ch23-s3">23장 다 절</a>`; all six chapters are being written now, section titles as in build.py).

## Real-paper guides (optional, at most one per chapter)
Read `REAL_BRIEF.md`. Only if you find an open-access (PMC, CC BY or similar) economic evaluation that fits a section's paper box, add `realpapers/chNN-sK.html` (`data-after="1"`): quote only a few numbers and ≤ 15-word phrases, link to the article, verify every quoted number against the full text. Skip it rather than guess.

## Files (edit only these)
`content/chNN.html`, `gen/nums_chNN.py`, `gen/fig_chNN.py`, `figs/chNN_*.html`, `refs_add/chNN.py`, `realpapers/chNN-s*.html` (+ `gen/lib_p4.py` additions for ch23/ch24 only). Do not edit build.py, shell.html, other chapters, plan/, P4_FACTS.md, refs_add/p4.py. Other agents write the other PART 4 chapters in parallel.

## Verify and report
`python3 build.py --only chNN --out <scratch>/t.html` (no warnings; an unknown ref key or missing figure makes the build exit); visible length per section (`tools/visible.py` has a `vis()` helper; strip `<details>`); screenshots with your own small Playwright script at 1200 px and 390 px (Chromium at /opt/pw-browsers; block requests whose URL starts with http; folded boxes closed; no horizontal overflow: `document.documentElement.scrollWidth` ≤ viewport) and look at a few of them; every `<ol class="marks">` number has exactly one `<span class="mk">` before its phrase. Then read the visible text once as a beginner.
Final reply (under 250 words): sections with visible/total length, examples used, figures, new ref keys (verified how), Python functions you ran, additions to lib_p4.py, anything uncertain or needing the owner's judgment.
