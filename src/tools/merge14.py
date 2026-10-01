"""Assemble content/ch14.html from content/_ch14/lead.html + part files in section order."""
import glob, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "content", "_ch14")
parts = []
lead = os.path.join(D, "lead.html")
if os.path.exists(lead):
    parts.append(open(lead, encoding="utf-8").read().strip())
secs = []
for f in glob.glob(os.path.join(D, "s*.html")):
    txt = open(f, encoding="utf-8").read()
    for m in re.finditer(r'<section class="sec" id="ch14-s(\d+)">.*?</section>', txt, re.S):
        secs.append((int(m.group(1)), m.group(0)))
secs.sort()
out = "\n\n".join(parts + [s for _, s in secs]) + "\n"
open(os.path.join(ROOT, "content", "ch14.html"), "w", encoding="utf-8").write(out)
print("ch14 sections:", [n for n, _ in secs])
