# Brief: making the theory chapters beginner-friendly (2026-09-30 request)

The site owner (pharmacist, social-pharmacy grad student) reviewed the site and asked for changes to ALL theory chapters (1–13). Their words, paraphrased:

- "Many new students enter with almost no statistics — at most the pharmacy statistics course, or none. Please add more **intuitive** explanation."
- "The goal is not 'master statistics'. The goal is that students can **read a paper and understand what the numbers mean**, and when they run their own analysis, know **how to use the method, how to control confounders and how to adjust**."
- "So the order **intuitive explanation → paper or numeric example → additional formula explanation** would be easier to understand."
- "In the '논문에서는 이렇게 보입니다' boxes the marker numbers come after the sentence, which is hard to read — **put the numbers in front**."

**Model to copy: `content/ch03.html`** (rewritten by me in this style — read it fully, compare with `/tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/ch03_before.html` to see exactly what changed). Also read STYLE.md (conventions, terminology table, 2026-09-30 policy).

## What to do in each section (`<section class="sec">`)

1. **Start with an intuition box** right after the `<h2>`:
```html
<div class="easy">
<p class="et">쉽게 말하면</p>
<p class="big">One or two plain sentences: what this method/concept is and what question it answers.</p>
<ul>
<li><b>왜 필요한가 / 비유.</b> An everyday analogy or a concrete picture (no formulas, no jargon without explanation).</li>
<li><b>논문에서는.</b> The typical way it appears in a paper (a short realistic string such as "HR 0.73 (95% CI 0.55–0.96)") and how to read it aloud in plain Korean.</li>
<li><b>기억할 한 가지.</b> The single most important caution (e.g., "변했다 ≠ 치료 때문에 변했다").</li>
</ul>
</div>
```
   Write for someone who never took statistics beyond a pharmacy-school course. Explain any term the first time it appears (Korean + English per STYLE.md).
2. Keep the `glance` box (after the easy box).
3. **Paper or numeric example next**: move the section's main '논문에서는 이렇게 보입니다' box up so it comes early (under an `<h3>논문에서 먼저 보기</h3>` with a one-sentence setup of the hypothetical study), or, for concept sections where a numeric example is the natural entry, put the numeric example first and the paper box right after it. **The first `<div class="paper">` in each section must be a paper-style box (not a Python output box)** — collapsed real-paper guides will be attached under it automatically. Then the numeric walk-through ("숫자로 따라가기"), described in words and numbers.
4. **Formulas last**: move display formulas (`<div class="eq">`) and formula-heavy derivations into a final `<h3>수식으로 보기</h3>` subsection placed just before the lab link and 정리, starting with `<p class="math-note">계산 원리가 궁금할 때 읽는 부분입니다. 건너뛰어도 논문을 읽는 데는 지장이 없습니다.</p>`. In the earlier text, replace a moved formula with a plain-words version ("신호 ÷ 잡음: 평균 변화 ÷ 그 불확실성") and a pointer ("정확한 식은 '수식으로 보기'"). Numbers in worked examples can stay (they are examples, not formulas). Short inline math that is essential to reading a paper (e.g., OR = e^β) may stay in place, explained in words.
5. **Add a "내 연구에 쓸 때" box** in method sections (and wherever it helps in concept sections):
```html
<div class="callout use"><p class="ct">내 연구에 쓸 때</p><ul>
<li><strong>고르는 기준</strong>: when to choose this vs alternatives.</li>
<li><strong>교란과 보정</strong>: in observational (e.g., claims) data what confounding problem arises and how to adjust — which model/chapter (multivariable regression 6·9·11·12장, stratification/Mantel–Haenszel 8장 바 절, propensity scores 14장 다 절) — and what adjustment cannot fix.</li>
<li><strong>보고할 것</strong>: what to report (estimates, CI, method, n).</li>
</ul></div>
```
   Place it after the main explanation and examples, before the warnings/Python output/formula parts (see ch03).
6. **Markers in front**: in every paper-style box, put `<span class="mk">n</span>` immediately BEFORE the phrase or number it explains, and wrap that phrase in `<span class="hl">…</span>` if it is not already (e.g. `(<span class="mk">2</span> <span class="hl">mean change, −7.0 mmHg</span>; …)`). In journal tables put the marker at the start of the cell (`<td class="r"><span class="mk">2</span> −7.0 ± 11.0</td>`); in footnotes at the start of the relevant sentence. A helper already moved the easy cases (`tools/move_marks.py`, run `python3 tools/move_marks.py content/chNN.html` to list what is left — "unhandled excerpt marks"); fix all remaining ones by hand so each marker precedes the text it annotates. Do NOT move markers inside `<pre class="out">` program outputs (fixed-width text).
7. **Simplify wording where it is dense**: shorter sentences, more concrete; keep every correct statement that matters for reading papers or doing analysis, but you may move technical side notes (software defaults, rare variants) later in the section (e.g., into the Python output box explanation or the 수식으로 보기 part) or into `<div class="callout deep"><p class="ct">심화 · …</p>…</div>`.
8. Keep: all numbers (they are computed — do not change them; if you rephrase a calculation, re-check it against `gen/nums_chNN.py` output), figures (`<!--FIG:…-->`), citations (`<cite data-ref>`), lab links, 정리 (update bullets if the section's emphasis changed), section ids/titles, the `<span class="wrong">` convention. Don't delete content that the owner's goals need (paper reading, confounding/adjustment, correct interpretation); do trim repetition.

## Scale and tone
- Aim for a section that a beginner can read top-down: first 2–3 screens give the idea and how it looks in a paper; details and formulas come later for those who want them.
- Korean 합니다체, plain headings (no flashy titles, no colon subtitles), no emoji, no em-dash asides.
- Do not add R. Python mentions only where accurate.

## Mechanics
- Edit only your assigned `content/chNN.html` files (and, only if a figure caption must change, the matching `gen/fig_chNN.py`, then re-run it with `source /home/claude/pylibs/env.sh`). Other agents are editing other chapters and writing `realpapers/*.html` in parallel — do not touch those or shared files (shell.html, build.py, refs, svgplot, labkit, briefs).
- Verify: `python3 build.py --only chNN --out /tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<yourdir>/t.html` (no warnings); `python3 tools/move_marks.py content/chNN.html` shows 0 unhandled excerpt marks; `python3 tools/preview.py chNN <dir>` (and `--mobile --dark`) and look at a few PNGs; check that every `<ol class="marks">` number still has its marker and vice versa.
- Final reply (under 200 words): what you changed per chapter (sections restructured, easy/use boxes added, formulas moved), anything you removed, anything uncertain.
