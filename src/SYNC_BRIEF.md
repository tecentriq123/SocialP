# Brief: port the GitHub PART 1 revision back into the source (2026-10-02)

The site (Korean statistics study site) is built from source in /home/claude/site by `python3 build.py` (shell.html + content/chNN.html + figs/ + refs.py + refs_add/*.py + realpapers/ + widgets/ + styles/ → dist/index.html). Read build.py first: it injects figures (`<!--FIG:name-->` → figs/name.html), numbers citations (`<cite data-ref="key"></cite>` → `<a class="cite" href="#ch15-ref-key" title="…">[n]</a>`, numbered by first use in chapter order), injects real-paper guides (realpapers/<sid>.html → `<details class="real-box">` after the Nth paper box), and wraps line-by-line code lists (`collapse_lines`).

Another session revised PART 1 by editing the BUILT file directly and deployed it. That version is the new truth:
`/tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/gh_latest.html` (also `git -C /home/claude/socialp show origin/main:index.html`). Its META (chapter list) is extracted in `…/scratchpad/gh_meta.json`. Note: gh_latest.html is wrapped in a small document skeleton before `<title>` and ends with `</body></html>`; compare only from `<title>` on.

Current differences between our build (`dist/index.html`) and gh_latest:
- templates `ch00` (new chapter "통계를 시작하기 전에", 6 sections) and `rv01` (new, `no: "종합"`, `review: true`, group p1, placed after ch07: "분석 고르기 연습", 4 sections);
- templates ch01–ch07 changed (guides "처음 읽을 때", `<span class="lv deep">심화</span>` tags, `div.practice` "스스로 확인하기", run cells, a checking widget in 1장 마 절, statistical fixes);
- ch12 tiny change; ch15 has 2 new references (238–239);
- the non-template part (CSS/JS of the shell) grew by ~4 KB;
- ch08–ch11, ch13, ch14 and all labs are already identical.

## Goal
Update the source so that `python3 build.py` reproduces gh_latest exactly: every `<template>` byte-identical, the META JSON identical, and the shell (non-template part) identical. Then our source is the single source of truth again.

## Steps
1. Back up first: copy content/ figs/ gen/ realpapers/ refs_add/ widgets/ styles/ shell.html build.py refs.py to `…/scratchpad/backup/before_sync_1002/`.
2. build.py: add ch00 and rv01 to CHAPTERS (titles/sections exactly as gh_meta.json) and support the `review`/`no` fields so `chapter_meta()` emits identical JSON (diff the JSON).
3. shell.html: diff the non-template part of our build vs gh_latest and port every change (CSS, JS, widgets). Chapter-specific widget JS may go to widgets/chNN.js or the shell — whatever reproduces the output.
4. For each changed chapter, write content/chNN.html by reversing the build transforms on the gh template:
   - remove each injected `<details class="real-box">…</details>`; if its text differs from realpapers/<sid>.html, update the realpapers file (and its data-after if the position changed);
   - citation links → `<cite data-ref="KEY"></cite>` (KEY is in the href);
   - injected figures → `<!--FIG:name-->` when identical to figs/name.html; if a figure changed, update figs/name.html AND the caption/text in gen/fig_chNN.py (so regeneration does not revert it; re-run the script with `source /home/claude/pylibs/env.sh` only if that reproduces the same file);
   - undo `collapse_lines` (read its code to invert exactly).
5. New references: their keys are in the `id="ch15-ref-KEY"` items; add them to a new `refs_add/sync1002.py` (`ADD = {...}`, same structure as other refs_add files) with text/doi exactly as rendered.
6. Iterate build → compare (per template, META, shell) until identical. A tiny script that prints the first differing offset per template is useful.
7. Run `python3 tools/move_marks.py content/chNN.html` for changed chapters only to confirm it prints "unchanged" (do not keep any change it makes).

Do not edit anything else (no content improvements). Scratch: `/tmp/claude-0/-home-claude-socialp/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/sync/`.
Final reply (under 200 words): what changed in the source, whether the rebuild is byte-identical (per template/META/shell), any remaining differences and why.
