# Brief: fold PART 1·2 for beginners, add guides and practice to PART 2 (2026-10-02)

The site (Korean statistics study notes for new social-pharmacy grad students; source in /home/claude/site, built by `python3 build.py`) is too long for its readers. The owner decided (see plan doc decisions):
- Readers are non-statistics grad students who must **read** claims-data cohort, pharmacoeconomic and policy-evaluation/meta-analysis papers and **run the analyses in Python**. Content needed for that stays visible; the rest is folded into collapsed '심화' boxes (never deleted).
- **Every section (절) of the original book TOC (chapters 1–13) stays, visible.** Only parts inside a section are folded: 심화 subsections, formulas, intermediate calculations.
- '수식으로 보기' is always folded; in '숫자로 따라가기' the setup and the final result stay visible, intermediate arithmetic is folded.
- PART 1 already has a '처음 읽을 때' route box per chapter and '스스로 확인하기' practice per section (made in an earlier revision; models: content/ch02.html, ch04.html, ch07.html). PART 2 (ch08–ch13) must get the same.

## Folding mechanism (build.py `fold_deep`, already implemented — do not edit build.py/shell.html)
1. An `<h3>` containing `<span class="lv deep">심화</span>` is folded with everything after it up to the next `<h3>` (or up to the section's trailing lab-link / 정리 / 스스로 확인하기). Same for `<h4>` (up to the next `<h4>`). Every `<h3>수식으로 보기…` is folded automatically.
2. `<div class="callout deep">` boxes, and paper/callout boxes whose title (paper-h / p.ct) contains the 심화 tag, are folded as boxes.
3. Any span: `<!--FOLD:summary text-->` … `<!--/FOLD-->` becomes a collapsed box titled "summary text" + 심화 tag. Use it (a) for intermediate steps of a worked example ("계산 과정 보기"), (b) when a long subsection should keep a 1–3 sentence visible summary and fold the rest ("자세히 보기: …").
Build `python3 build.py --only <ids> --out …/t.html`; `--nofold` shows everything.

## What to fold (plan/triage.py)
`plan/triage.py` lists, per section, `fold` = subsection (h3) titles to fold, `note` = what must stay visible, `move` = parts that will later move to new chapters (PART 3/4) — **fold them now** (tag 심화) and leave nothing else. `plan/measure.json` has each section's current and target visible length (`before`/`after`, characters; the target assumed '숫자로 따라가기' shrinks to ~60%). Aim near the target (±20%), but judge by content: never fold something the visible text needs to make sense; when a folded subsection contains one key message the reader needs (e.g., 2장 가 '가정과 확인 방법' → "Welch를 기본으로 쓰고, 표본이 크면 정규성은 덜 중요하며, 이상값은 따로 본다"), keep that message as a short visible summary paragraph and fold the rest with a FOLD marker or the 심화 tag. If an item in the plan should clearly stay visible for these readers, keep it and say so in your report.

After folding, read the visible text of each section top to bottom: references to folded material should say so ("아래 접힌 '심화' 상자", "(심화)"), and nothing visible may depend on an unexplained folded term.

## Route boxes and practice
- PART 1 (ch00–ch07, rv01): update each chapter's route box (`<div class="route">`: 먼저 읽을 부분 / 나중에 읽을 부분) to match the new folding. Practice blocks already exist; adjust a question only if it now depends on folded material without saying so.
- PART 2 (ch08–ch13): add a route box at the top of each chapter (after the lead paragraph; same markup and four items: 미리 알아 둘 것 with links like `<a href="#ch08-s2">8장 나 절</a>`, 먼저 읽을 부분, 나중에 읽을 부분, 이 장을 마치면 ①②③④). Add a `<div class="practice">` 스스로 확인하기 block at the end of every section (after the keypoints box, exactly like PART 1): 2–3 questions per section (single-section chapters 9 and 11: 5–6), tagged with `<span class="qt">` types used in PART 1 (계산, 해석 고치기, 분석 고르기, 논문 읽기 …), each with `<details><summary>답과 풀이</summary><div><p>…</p></div></details>`. Questions should practise what these readers need: reading a number in a paper correctly (OR vs RR, HR is not a risk ratio, adjusted vs crude, P for interaction, CI vs P), choosing the analysis for a claims-data/pharmacy study, and spotting a wrong interpretation. Compute every number in questions and answers with Python (`source /home/claude/pylibs/env.sh`); reuse the chapter's own examples where possible.

## Rules
Korean 합니다체; terminology per STYLE.md (상대위험도 = RR, 위험비 = HR, 위험률 = hazard, 오즈비 = OR); no R; do not change statistics content except to fix a real error (report it). Markers `<span class="mk">n</span>` stay before the phrase they explain. Edit only your chapters' content/chNN.html (and realpapers/ of your chapters only if their position text breaks). Do not edit build.py, shell.html, plan/, other chapters.

## Verify and report
`python3 build.py --only <your ids> --out /tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<dir>/t.html` (no warnings); visible length per chapter: build the same ids with `--nofold` to another file and run `python3 tools/visible.py <nofold.html> <t.html>`; `python3 tools/move_marks.py content/chNN.html` must print "unchanged" (it has a bug with adjacent markers — never keep its changes); preview with `python3 tools/preview.py <chid> <dir>` and look at a few PNGs (folded boxes closed). Final reply (under 250 words): per chapter visible length before → after vs target, what you kept visible against the plan and why, route boxes/practice added, any statistical errors found.
