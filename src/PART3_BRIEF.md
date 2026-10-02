# Brief: write a PART 3 chapter (약물역학 연구 설계, chapters 14–19) — 2026-10-02

Site: Korean study notes for new graduate students in a social-pharmacy lab (source in /home/claude/site, built by `python3 build.py` → dist/stats.html). Readers have little statistics background. The owner's goal for them: **read** claims-data cohort, pharmacoeconomic and policy-evaluation/meta-analysis papers correctly, **choose and report** the right analysis, and **run it in Python**. PART 3 is new: chapter 14 청구자료 코호트 연구의 설계, 15 성향점수, 16 시간과 관련된 편향, 17 경쟁위험 분석, 18 정책 효과의 평가, 19 메타분석. Chapter and section titles are fixed in `build.py` CHAPTERS (your chapter's section ids are `chNN-s1…`, letters 가, 나, …).

Until now these topics existed only as short, deliberately light sections of the old chapter 14 (now split). You rewrite your topic **at the level of the main chapters** (like ch09/ch11/ch12), but lean: the owner complained that the site was too long, so everything not needed on first reading is folded.

## Read first
- `STYLE.md` (terminology: 상대위험도 = RR, 위험비 = HR, 위험률 = hazard, 오즈비 = OR, 교정 = calibration; markup; no R; Python only).
- `FOLD_BRIEF.md` (how folding works: `심화` tag on h3/h4/boxes, `<!--FOLD:title-->…<!--/FOLD-->`, '수식으로 보기' auto-folded) and one finished model chapter in full: `content/ch12.html` (route box, 쉽게 말하면, glance, 숫자로 따라가기, 흔한 오해, 내 연구에 쓸 때, 논문에서 읽어 보기 with numbered markers, 수식으로 보기, 정리, 스스로 확인하기). Skim `content/ch11.html` for a single long topic handled with folds.
- Source material for your topic (reuse its good parts — examples, paper boxes, numbers — and deepen): `plan/old14/sa.html` (old 가–다: McNemar, ROC, **성향점수**), `sb.html` (old 라–사: **경쟁위험, 시간의존·불멸시간, 메타분석, ITS·DID**), with their scripts `gen/nums_ch14a.py`/`nums_ch14b.py`, `gen/fig_ch14a.py`/`fig_ch14b.py`, refs in `refs_add/ch14a.py`/`ch14b.py` (keys already valid; reuse them), and the verified real-paper guides in `plan/old14/realpapers/ch14-s3…s7.html` (s3 성향점수, s4 경쟁위험, s5 불멸시간, s6 메타분석, s7 ITS). Other related material is named in your prompt.
- The plan doc summary for PART 3 chapters is in your prompt.

## Chapter structure
1. `<p class="lead">` (2–3 sentences) + `<div class="route">` '처음 읽을 때' box (미리 알아 둘 것 with links, 먼저 읽을 부분, 나중에 읽을 부분, 이 장을 마치면 ①–④) — same markup as ch12.
2. Each section (`<section class="sec" id="chNN-sK">`, `<h2><span class="sec-no">가.</span>Title</h2>` exactly as in build.py):
   - `<div class="easy">` 쉽게 말하면 (gist, everyday or pharmacy analogy, how it appears in a paper, one caution) and `<dl class="glance">`.
   - Explanation subsections (h3) built around **one running hypothetical claims-data example for the whole chapter** (state it once; e.g. an SGLT2 vs DPP-4 inhibitor new-user cohort from 청구자료 with realistic sizes). '숫자로 따라가기': setup and final result visible, intermediate arithmetic inside a FOLD ("계산 과정 보기"). Use tables/figures where they help (figures: `svgplot.py`, see any `gen/fig_chNN.py`; captions "그림 NN-k." numbered in order of appearance; tables "표 NN-k." if you caption them).
   - `<div class="callout warn">` 흔한 오해 with `<span class="wrong">…</span>` statements (the owner wants these errors corrected actively: OR read as RR; HR read as a risk ratio; "PS matching removes all bias / like an RCT"; AUC = accuracy; correlation = causation; "not significant = no effect"; adjusting for mediators/colliders).
   - `<div class="callout use">` 내 연구에 쓸 때: when to use, what to decide before analysing, what to report (Methods/Results wording), and one line naming the Python tools that exist (statsmodels / lifelines / scikit-learn / scipy function names you have **run** in this environment; say "직접 구현" where no function exists). No lab link (PART 5 labs for these chapters come later).
   - `<h3>논문에서 읽어 보기</h3>`: at least one '논문에서는 이렇게 보입니다' paper box per section where a paper would show it (hypothetical, labelled 가상의 예시; English journal-style table or Methods/Results excerpt; `<span class="mk">n</span>` placed BEFORE each highlighted phrase; every marker explained for a beginner in `<ol class="marks">`). It comes after the explanations it relies on.
   - `<h3>수식으로 보기</h3>` + `<p class="math-note">` + a few `<div class="eq">` formulas with the example plugged in (auto-folded).
   - `<div class="keypoints">` 정리 (3–5 bullets), then `<div class="practice">` 스스로 확인하기 (2–3 questions; types 계산 / 해석 고치기 / 분석 고르기 / 논문 읽기; answers in `<details><summary>답과 풀이</summary><div>…</div></details>`).
3. **Length**: visible text (folds closed) about 6,000–9,000 characters per section; put derivations, extra variants, software output reading, edge cases and long arithmetic into folds (심화 h3/h4, FOLD markers). Nothing visible may depend on a term explained only inside a fold.
4. Korean context matters: where relevant use 건강보험심사평가원(HIRA)·국민건강보험공단(NHIS) 청구자료 terms (명세서, 원외처방, 상병코드 KCD, 주성분코드, 요양기관). State only facts you are sure of or have verified (WebSearch/WebFetch on official or peer-reviewed sources); otherwise stay general.

## Accuracy
- Every number in text, tables, paper boxes, figures and practice answers is computed by your `gen/nums_chNN.py` (simulate the running example with a fixed seed; `source /home/claude/pylibs/env.sh`: numpy, scipy, pandas, statsmodels 0.15, lifelines 0.30.3, scikit-learn 1.8). Paper boxes must be internally consistent (CI contains estimate, P agrees with CI, counts add up).
- Citations: `<cite data-ref="key"></cite>`; reuse existing keys (`refs.py`, `refs_add/*.py`); add new ones in `refs_add/chNN.py` (same dict format; **verify each on PubMed** — ToolSearch "PubMed": `mcp__PubMed__lookup_article_by_citation`, `get_article_metadata` — or the publisher page; never from memory alone). Cite methodological sources for key claims (e.g., new-user design, immortal time, Fine–Gray, ITS tutorial, Cochrane Handbook).
- Cross-references use the new numbering: 성향점수 = 15장, 시간 관련 편향 = 16장, 경쟁위험 = 17장, 정책평가 = 18장, 메타분석 = 19장, 설계 = 14장; McNemar/ROC/의료비용/결측/표본크기/일치도 = 부록 A 가–바 절 (`#ap01-s1…s6`); links like `<a href="#ch08-s6">8장 바 절</a>`. PART 4 (20–25장, 약물경제성 평가) is not written yet — you may say "PART 4에서 다룹니다" without a link.

## Real-paper guides
Read `REAL_BRIEF.md`. Put the old verified guide for your topic (named in your prompt) into `realpapers/chNN-sK.html` for the section whose paper box it fits (`data-after="1"` = under the first paper box of that section; adjust wording that refers to the old section). Do not change its verified numbers. You may add one more guide only if your prompt asks for it.

## Files (edit only these)
`content/chNN.html`, `gen/nums_chNN.py`, `gen/fig_chNN.py`, `figs/chNN_*.html`, `refs_add/chNN.py`, `realpapers/chNN-s*.html`. Do not edit build.py, shell.html, other chapters, plan/. Other agents write the other PART 3 chapters in parallel.

## Verify and report
`python3 build.py --only chNN --out /tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<dir>/t.html` (no warnings; an unknown ref key makes the build exit); visible length per section (strip `<details>` blocks; `tools/visible.py` has a `vis()` helper); screenshots with your own small Playwright script at 1200 px and 390 px (block requests whose URL starts with http; folded boxes closed; no horizontal overflow: `document.documentElement.scrollWidth` ≤ viewport); every `<ol class="marks">` number has exactly one `<span class="mk">` before its phrase. Then read the visible text once as a beginner.
Final reply (under 250 words): sections with visible/total length, the running example, figures, new ref keys (verified how), real-paper guides placed, Python functions you ran, anything uncertain.
