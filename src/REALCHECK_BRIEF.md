# Brief: independent check of the "실제 논문으로 읽어 보기" guides

Files `realpapers/<section-id>.html` contain guides to real open-access papers (format in REAL_BRIEF.md). They are injected, collapsed, under the first (or `data-after`-th) paper box of each section at build time. Students will trust them, so every factual claim must be right. You did not write them.

For each guide in your scope:
1. **Paper exists and matches the citation**: authors, title, journal, year, volume/pages, DOI and PMCID. Use the PubMed tools (ToolSearch "PubMed": `mcp__PubMed__get_article_metadata`, `lookup_article_by_citation`, `get_full_text_article`, `convert_article_ids`) and WebFetch/WebSearch as fallback. Open-access status: the PMC link should resolve to full text.
2. **Every quoted number and short quote matches the paper's full text** (abstract/Results/tables). Values marked "직접 계산"/"이 사이트의 계산" must be recomputed from the paper's reported numbers. If a number cannot be verified, remove it or rephrase without it; if the whole guide is unreliable, tell me rather than deleting.
3. **Interpretation is correct** and consistent with the chapter section it sits in (read the section in content/): e.g., OR vs RR, HR vs risk, adjusted vs crude, one-sided vs two-sided, what a P for interaction means, criticism of the paper stated fairly and accurately.
4. **Placement**: build and look at where each guide lands (`python3 build.py --only chNN --out …/t.html`, then search the output for `real-box`) — it should sit under a paper box that teaches the same thing. Adjust `data-after` if a chapter's box order changed.
5. **Copyright**: no reproduced tables/figures, quotes ≤ 15 words.
6. Format: each block ends with `</div><!--/real-->` and has no other `</div>` inside.

Fix problems directly in realpapers/*.html (only). Do not edit content/, gen/, shared files. Final reply (under 250 words): per guide OK/fixed (what), anything removed or unverifiable.
