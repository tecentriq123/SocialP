"""Print option lengths per question (correct marked *) and the longest-correct count. Usage: python3 tools/qz_lens.py qzNN"""
import re, sys
s = open(f"content/{sys.argv[1]}.html", encoding="utf-8").read()
strip = lambda h: re.sub(r"<[^>]+>", "", re.sub(r'<div class="qz-why-o">.*?</div>', "", h, flags=re.S)).strip()
n = L = 0
for m in re.finditer(r'<div class="qz"[^>]*id="([^"]+)"(.*?)<div class="qz-fb">', s, re.S):
    for k, ol in enumerate(re.findall(r'<ol class="qz-opts">(.*?)</ol>', m.group(2), re.S)):
        lis = re.findall(r"<li( data-ok=\"1\")?>(.*?)</li>", ol, re.S)
        ok = [len(strip(t)) for a, t in lis if a][0]; bad = max(len(strip(t)) for a, t in lis if not a); n += 1; L += ok > bad
        print(m.group(1) + ("s2" if k else ""), [("*" if a else "") + str(len(strip(t))) for a, t in lis], "LONGEST" if ok > bad else "")
print(L, "/", n)
