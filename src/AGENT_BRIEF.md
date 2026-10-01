# Brief for chapter writers (Part 2)

You are writing chapter(s) of a Korean-language statistics learning website for new graduate students in social pharmacy (pharmacoepidemiology, pharmacoeconomics/HEOR). Project root: /home/claude/site. Part 1 (chapters 1–7) is finished and approved; you are writing Part 2.

## Read first, in full
1. `STYLE.md` — the binding style/format guide, including the **추가 규칙** section at the end (terminology table, common mistakes found in review).
2. `content/ch01.html` — the approved model chapter (structure, tone, density, components). Also skim `content/ch07.html` (the approved long, deep chapter) for how a heavily used method is covered from basics to advanced.
3. `PART1_OUTLINE.md` (what chapters 1–7 already explain — refer back instead of repeating) and `FORWARD_REFS.txt` (what chapters 1–7 promised later chapters would cover — fulfil those promises in your chapter).
4. `gen/fig_ch01.py`, `gen/nums_ch01.py`, `svgplot.py`, `build.py` (CHAPTERS list = exact section titles), `refs.py` and `refs_add/*.py` (existing reference keys).

## What the site owner values
- The site owner (a pharmacist and social-pharmacy grad student) said the '논문에서는 이렇게 보입니다' boxes are what they like most: "처음 공부하는 사람이 논문을 봐도 '아 이런 뜻이었구나' 할 수 있게 자세히 설명". Make these boxes the centerpiece: at least one per section and 2–4 in key sections. Every marked number gets a Korean explanation: where it comes from (with a check calculation where possible), how to read it, how it is commonly misread, and what else to check in the paper.
- Korean 합니다체. Intuitive explanation and concrete numeric examples (치료군/대조군, 신규 사용자 코호트, etc.) first; only a few key formulas and no long derivations. Statistics terms paired with English on first use. Plain, non-flashy headings. Accuracy over agreeableness: state uncertainty rather than guess.
- Errors the owner wants actively corrected: OR vs RR confusion, interpreting HR as a risk ratio, correlation = causation, believing PSM removes all bias, reading AUC as accuracy.
- Realistic social-pharmacy settings: Korean claims data (HIRA/NHIS), active-comparator new-user cohorts, medication adherence (PDC), pharmacist interventions, healthcare costs, adverse drug events, oncology cohorts (e.g., renal cell carcinoma). Label every example 가상의 예시.

## Mechanics
- Write only your own files: `content/chXX.html`, `gen/nums_chXX.py`, `gen/fig_chXX.py` (and optional `gen/lib_chXX.py`), `figs/chXX_*.html`, `refs_add/chXX.py`, optional `widgets/chXX.js` (max one widget). Other agents are writing other chapters in parallel in the same folder: **do not edit shared files** (`shell.html`, `build.py`, `refs.py`, `svgplot.py`, `STYLE.md`, other chapters' files). You may import (read-only) helpers from other gen/lib files, e.g. `gen/lib_ch07.py` has Kaplan–Meier, log-rank, RMST and a one-covariate Cox.
- Python: numpy, scipy, pandas, sklearn are available; **statsmodels and lifelines are not installed** (pip is blocked). Implement models yourself (Newton–Raphson logistic/Cox, IRLS for GLMs, etc.) and sanity-check them (e.g., sklearn LogisticRegression(penalty=None) for logistic; by-hand small cases; simulated data with known parameters).
- Compute EVERY statistic in `gen/nums_chXX.py`; text, tables, figures, captions, R-output boxes and mark explanations must agree exactly (watch rounding).
- Figure captions "그림 X-n." in order of appearance. Check figures in the preview for overlapping labels, clipped text and overflow.
- References: reuse existing keys; add new ones in `refs_add/chXX.py` only if you are certain of the bibliographic details (DOI only if sure). The same key in two files must have identical text; check existing files first.
- Verify: `python3 build.py --only chXX --out /tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<yourdir>/test.html` must print no warnings; then `python3 tools/preview.py chXX <scratch dir>` and again with `--mobile --dark`; open the PNGs with Read and fix problems. Re-check all statistical claims once more before finishing.

## Final reply (under 250 words)
Files created, number of figures and paper boxes, widget (if any), new ref keys, anything you were unsure about, and any change you need in shared files.
