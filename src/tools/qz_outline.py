"""Print a chapter's sections, anchored headings and keypoints (for writing quiz questions). Usage: python3 tools/qz_outline.py ch00"""
import re, sys
cid = sys.argv[1]
s = open(f"content/{cid}.html", encoding="utf-8").read()
print(cid, len(s), "chars")
for m in re.finditer(r'<section class="sec" id="(%s-s\d+)">' % cid, s):
    e = s.find("</section>", m.end()); sec = s[m.end():e]
    h2 = re.search(r"<h2>(.*?)</h2>", sec); print("\n==", m.group(1), re.sub("<[^>]+>", "", h2.group(1)) if h2 else "")
    for h in re.finditer(r'<h([34]) id="([^"]+)">(.*?)</h\1>', sec):
        print("  " * (int(h.group(1)) - 2), h.group(2), re.sub("<[^>]+>", "", h.group(3))[:80])
    k = sec.find('<div class="keypoints">')
    if k >= 0:
        kp = sec[k:sec.find("</ul></div>", k)]
        print("  [정리]", re.sub(r"\s+", " ", re.sub("<[^>]+>", " ", kp.replace("</li>", " ||")))[:1800])
