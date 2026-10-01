"""Build one chapter alone and screenshot its visual blocks.

usage: python3 tools/preview.py ch02 /path/to/outdir [--dark] [--mobile]
Writes outdir/preview.html and outdir/<ch>_<n>_<kind>.png for every figure, paper box,
widget and table; prints console errors and horizontal-overflow warnings.
(MathJax/Google Fonts are blocked in this sandbox, so formulas show as raw TeX here;
that is expected — they render in the published page.)
"""
import asyncio, os, subprocess, sys
from playwright.async_api import async_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main(ch, out, dark, mobile):
    os.makedirs(out, exist_ok=True)
    html = os.path.join(out, "preview.html")
    subprocess.run([sys.executable, os.path.join(ROOT, "build.py"), "--only", ch, "--out", html], check=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        vw = 390 if mobile else 1400
        ctx = await b.new_context(viewport={"width": vw, "height": 900}, color_scheme="dark" if dark else "light")
        await ctx.route("**/*", lambda r: r.abort() if r.request.url.startswith("http") else r.continue_())
        pg = await ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "ERR_FAILED" not in m.text else None)
        await pg.goto("file://" + html + "#" + ch)
        await pg.wait_for_timeout(600)
        sw = await pg.evaluate("document.documentElement.scrollWidth")
        print(f"page scrollWidth={sw} (viewport {vw})" + ("  <-- HORIZONTAL OVERFLOW" if sw > vw else ""))
        n = 0
        for sel, kind in (("#body figure", "fig"), ("#body .paper", "paper"), ("#body .widget", "widget"),
                          ("#body .tbl-wrap", "table"), ("#body .flow", "flow"), ("#body .matrix", "matrix"),
                          ("#body .tree", "tree"), ("#body .cell", "cell"), ("#body ol.lines", "lines")):
            for el in await pg.query_selector_all(sel):
                n += 1
                await el.screenshot(path=os.path.join(out, f"{ch}_{n:02d}_{kind}.png"))
        secs = await pg.evaluate("[...document.querySelectorAll('#body section.sec')].map(s => s.id + ':' + s.querySelector('h2').textContent.trim())")
        print("sections:", secs)
        print(f"{n} block screenshots written to {out}")
        print("errors:", errs or "none")
        await b.close()


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    asyncio.run(main(a[0], a[1], "--dark" in sys.argv, "--mobile" in sys.argv))
