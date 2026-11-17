"""Print plain text under given heading ids. Usage: python3 tools/qz_read.py ch00 s1-problem s2-bias ... [-n 900]"""
import re, sys
args = sys.argv[1:]; n = 900
if "-n" in args: i = args.index("-n"); n = int(args[i + 1]); del args[i:i + 2]
cid, keys = args[0], args[1:]
s = open(f"content/{cid}.html", encoding="utf-8").read()
for k in keys:
    hid = k if k.startswith(cid) else f"{cid}-{k}"
    i = s.find(f'id="{hid}"')
    if i < 0: print("###", hid, "NOT FOUND"); continue
    j = re.search(r"<h[234][ >]|</section>", s[i + 10:])
    t = s[i:i + 10 + (j.start() if j else 4000)]
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S); t = re.sub(r"<[^>]+>", " ", t); t = re.sub(r"\s+", " ", t)
    print("###", hid, t[:n], "\n")
