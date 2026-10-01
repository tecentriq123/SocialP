# Brief: "실제 논문으로 읽어 보기" guides (real, open-access papers)

The theory chapters (content/ch01–ch13.html) teach how to read numbers in papers using **hypothetical** example boxes ('논문에서는 이렇게 보입니다', labelled 가상의 예시). The site owner asked to add, under those boxes and **collapsed by default**, guides to **real published papers** so students can practise on the real thing: "which table/figure to open, which number to look at, and what it means".

## Rules
- **Real, verifiable, open-access papers only.** Prefer PubMed Central (PMC) open-access articles, BMJ/BMJ Open, PLOS Medicine/PLOS ONE, JAMA Network Open, BMC journals, and Korean open-access journals (Journal of Korean Medical Science, Epidemiology and Health, Diabetes & Metabolism Journal, Korean Journal of Internal Medicine, Yonsei Medical Journal, etc.). Studies using Korean claims data (HIRA/NHIS) or pharmacy/pharmacoepidemiology/clinical-trial topics are ideal for this audience (social-pharmacy grad students).
- **Verify everything from the full text** (use the PubMed tools: search ToolSearch for "PubMed", e.g. `mcp__PubMed__search_articles`, `get_article_metadata`, `get_full_text_article`; or WebFetch of the PMC page). Bibliographic details (authors, title, journal, year, volume, pages, DOI/PMCID) and **every number you quote** must match the paper exactly. Do not use a paper you could not open in full text. Never invent or approximate numbers.
- **Copyright**: do not reproduce tables, figures or long passages. Quote only a few numbers and at most a short phrase (≤ 15 words) per item; describe everything else in your own Korean words. Link to the paper so students read the original.
- One guide per section where the method appears in a suitable paper (target: every method section; for pure concept sections like 자유도 or 중심극한정리, a guide only if a paper shows the concept naturally, e.g. "t(58) = …" reporting, or a sample-size paragraph). Chapters 9 (logistic) and 11 (Cox) are single long sections: give 2–3 guides there, attached to different paper boxes via `data-after` (see below). Do not reuse the same paper more than twice across the whole site.
- Explanations must match what the chapter teaches (read the section first) and be correct; point out real-paper nuances (e.g., "this paper reports aOR, not RR; the outcome is common so do not read it as risk ratio"; "HR, not risk"; "P for interaction reported"; "they used Welch's test"). If the paper has a limitation relevant to the section (e.g., pre-post without control, immortal time risk), say so neutrally.

## File format
Write one file per section: `realpapers/<section-id>.html` (e.g. `realpapers/ch07-s2.html`), containing one or more blocks:
```html
<div class="real" data-after="1" data-title="Short Korean label, e.g. 청구자료 코호트의 Kaplan-Meier 곡선"><p class="real-cite">Author A, Author B, Author C, et al. Title. <i>Journal</i>. Year;Vol(Issue):pages. <a href="https://doi.org/..." target="_blank" rel="noopener">doi:...</a> · <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC.../" target="_blank" rel="noopener">PMC 전문</a></p>
<p>연구 개요 2–3문장 (누구를, 무엇을, 어떤 설계로). 이 절과 관련된 분석이 무엇인지.</p>
<ol class="marks">
<li><span class="mk">1</span><span><b>어디를 보나: Table 2의 … 행.</b> 거기 적힌 숫자(짧게 인용)가 무엇이고, 이 절에서 배운 대로 어떻게 읽는지.</span></li>
<li><span class="mk">2</span><span>…</span></li>
</ol>
<p class="real-note">숫자는 원문 Table 2와 Results에서 옮겼습니다. 표와 그림은 링크한 원문에서 직접 확인하세요.</p>
</div><!--/real-->
```
- The closing `</div><!--/real-->` is required exactly (the build uses it). No other `</div>` may appear inside a block — use `<p>`, `<ol>`, `<ul>`, `<span>` only.
- `data-after="1"` attaches the guide below the first paper box of the section (the main '논문에서는 이렇게 보입니다' example). Use 2, 3… only to attach to a later paper box in the same section (e.g., in ch09/ch11); otherwise always 1.
- 3–6 numbered items per guide. Korean 합니다체; statistics terms Korean + English on first use; terminology per STYLE.md (상대위험도 = RR, 위험비 = HR, 오즈비 = OR, …).
- Do not edit content/*.html, gen/, figs/, shell.html, build.py, refs. Other agents are rewriting chapter text in parallel; section ids and the rule "the first paper box is the main example" are stable.

## Verify and report
- `python3 build.py --only chNN --out /tmp/claude-0/-home-claude/e0b25f68-ca2e-5b14-8f3a-9898d4746d73/scratchpad/<yourdir>/t.html` must show no warnings from your files; open the output HTML's section in `python3 tools/preview.py chNN <dir>` to see the collapsed box.
- Final reply (under 250 words): for each section, the paper used (first author, journal, year, PMCID) and what the guide teaches; sections skipped and why; anything you could not verify.
