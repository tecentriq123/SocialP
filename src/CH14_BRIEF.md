# Brief: chapter 14 "추가로 알아야 할 검정" (write in the beginner style, but lighter)

The site owner's instruction for chapter 14 (their words, paraphrased): "Unlike the earlier parts, don't go too deep. Go intuitive explanation → paper or numeric example → additional formula explanation, and keep the formula part simple."

Read first: `STYLE.md` (terminology table, markup, 2026-09-30 policy), `REWRITE_BRIEF.md` (the beginner structure used by all chapters), and the model `content/ch03.html` (structure: easy box → glance → 논문에서 먼저 보기 paper box → 숫자로 따라가기 → cautions → 내 연구에 쓸 때 → 수식으로 보기 → 정리). Also skim `FORWARD_REFS_14.txt`: sentences in chapters 1–13 and the labs that promise "14장 X 절에서 다룹니다" — your sections must cover what they promise (at a light level) so those pointers make sense.

## Per section (keep each section about half the length of a Part 2 section)
1. `<h2><span class="sec-no">가.</span>Title exactly as in build.py CHAPTERS</h2>`, id `ch14-sN`.
2. `<div class="easy">` intuition box (same markup as ch03): 1–2 sentence gist, an everyday analogy, how it looks in a paper and how to read it aloud, one caution.
3. `<dl class="glance">` (언제 쓰나 / 핵심 질문 / 가정·조건 / 논문 보고 형식).
4. `<h3>논문에서 먼저 보기</h3>` + ONE main '논문에서는 이렇게 보입니다' paper box (hypothetical, label 가상의 예시; English excerpt or journal table; markers `<span class="mk">n</span>` placed BEFORE the `<span class="hl">` phrase they explain; every marker explained in `<ol class="marks">` for a beginner). This must be the first `<div class="paper">` in the section (a real-paper guide will be attached under it later).
5. `<h3>숫자로 따라가기</h3>`: one small worked example with concrete numbers (computed in a script — see Mechanics), at most one or two figures (svgplot, captions "그림 14-n." numbered in order across the whole chapter — coordinate: use the section's letter in the fig file name, e.g. `figs/ch14_c_love.html`, and number captions provisionally; I will renumber after merging if needed).
6. Optional short `<div class="callout warn"><p class="ct">흔한 오해 · …</p>…` with `<span class="wrong">…</span>` statements (CSS appends "(X)").
7. `<div class="callout use"><p class="ct">내 연구에 쓸 때</p>…` — when to use it, practical pitfalls, confounding/adjustment notes where relevant, what to report, and (one line) which Python function/package does it if you are sure (e.g., statsmodels `mcnemar`, scikit-learn `roc_auc_score`, lifelines `AalenJohansenFitter`) — do not write code blocks.
8. `<h3>수식으로 보기</h3>` + `<p class="math-note">계산 원리가 궁금할 때 읽는 부분입니다. 건너뛰어도 논문을 읽는 데는 지장이 없습니다.</p>` + at most 1–2 short `<div class="eq">` formulas with one line of explanation each and the numeric example plugged in. Keep it simple.
9. `<div class="keypoints">` 정리 (3–4 bullets). No lab link (there is no lab 14).

Accuracy: the owner wants these errors actively corrected wherever relevant: OR vs RR, HR read as risk ratio, correlation = causation, **"PSM removes all bias"**, **"AUC = accuracy"**. Use `<cite data-ref="key">` only for references whose details you are certain of (reuse keys in refs.py / refs_add/*.py; add new ones in your own `refs_add/ch14X.py`, X = your part letter). Korean 합니다체, plain headings, statistics terms Korean + English on first use, no emoji, no R.

## Mechanics
- Write ONLY your own files: `content/_ch14/sX.html` (your sections, in order; X = your part letter a/b/c), `gen/nums_ch14X.py`, `gen/fig_ch14X.py`, `figs/ch14_*` with your section letter in the name, `refs_add/ch14X.py`. Other agents write the other sections of chapter 14 in parallel.
- Assemble and test: `python3 tools/merge14.py && python3 build.py --only ch14 --out /tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<dir>/t.html` (warnings about other agents' missing sections are expected; none about yours). `python3 tools/move_marks.py content/ch14.html` should show 0 unhandled marks for your sections. Preview: `python3 tools/preview.py ch14 <dir>` (+ `--mobile --dark`), open a few PNGs.
- Python: `source /home/claude/pylibs/env.sh` gives numpy, scipy, pandas, statsmodels 0.15, lifelines 0.30.3, scikit-learn; compute every number.
- Final reply (under 200 words): sections written, figures, new ref keys, anything uncertain.
