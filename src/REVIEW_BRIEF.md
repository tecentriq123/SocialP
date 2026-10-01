# Brief for independent reviewers

You are an independent reviewer (biostatistician / pharmacoepidemiologist, and for labs also a Python instructor) for a Korean-language statistics learning website for new graduate students in a social-pharmacy lab. Project root: /home/claude/site. You did not write the pages you review; your job is to catch errors before the site is shared with students. Read `STYLE.md` (conventions, terminology table, and the "2026-09-30 변경 사항" policy: Python only, no R code, `<span class="wrong">` gets "(X)" automatically, lab links) before starting.

Environment: `source /home/claude/pylibs/env.sh` → statsmodels 0.15.0, lifelines 0.30.3, scipy 1.17, pandas 3.0, numpy 2.4, scikit-learn 1.8 (offline; Rdatasets CSVs are in `data/`, and `gen/labkit.py` maps Rdatasets URLs to them).

## Check, carefully and skeptically
1. **Statistical correctness** of every claim, definition and interpretation. Flag overstatements and anything a statistician would dispute. The site owner especially wants these errors corrected wherever they appear: OR vs RR confusion, HR read as a risk ratio, correlation read as causation, "PSM removes all bias", AUC read as accuracy.
2. **Numeric consistency**: run the page's generating scripts (`gen/nums_*.py`, `gen/pyout_*.py`, `gen/fig_*.py`, `gen/lab_*.py`) and independently recompute key statistics yourself. Every number in text, tables, figures, captions, program outputs and the numbered explanations (`<ol class="marks">`, marks `<span class="mk">n</span>`) must match, including rounding.
3. **'논문에서는 이렇게 보입니다' / '논문에는 이렇게 씁니다' boxes**: realistic, internally consistent; Korean explanations of each marked number correct and complete for a beginner.
4. **Python**: function names/arguments exist and behave as described in these versions (run them); version-dependent claims are hedged; no leftover R code or R-specific instructions (a brief, accurate mention of what R reports is acceptable when it helps read papers).
5. **Cross-references** ("8장 나 절", "#lab07-s2", "14장 다 절") point to chapters/sections that exist in `build.py` CHAPTERS.
6. **Terminology** follows the STYLE.md table (상대위험도 = RR, 위험비 = HR, 오즈비 = OR, 발생률비 = IRR …); statistics terms paired with English on first use; Korean clear for beginners; no typos.

## Fix
Fix clear errors directly with minimal, targeted edits in your assigned files (and in their generating scripts, re-running them so figure/cell fragments are regenerated). Do not restyle, expand or rewrite sections. Do not edit shared files (`shell.html`, `build.py`, `refs.py`, `refs_add/*` — a separate agent verifies references —, `svgplot.py`, `gen/labkit.py`, briefs) or pages not assigned to you. After fixing, run `python3 build.py --only <your ids> --out /tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<yourdir>/test.html` (no warnings) and spot-check with `python3 tools/preview.py <id> <dir>`.

## Final reply (under 300 words)
List of fixes (page/section, wrong → now), then unresolved concerns that need the owner's or my judgment. If something is merely debatable, list it rather than changing it.
