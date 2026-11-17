"""Print option texts (without per-option explanations) for given question ids.
usage: python3 tools/qz_opts.py qz00-s1-q02 qz00-s1-q04 ..."""
import re, sys
for qid in sys.argv[1:]:
    s = open(f"content/{qid[:4]}.html", encoding="utf-8").read()
    m = re.search(r'<div class="qz"[^>]*id="%s"(.*?)<div class="qz-fb">' % qid, s, re.S)
    print("##", qid)
    for n, ol in enumerate(re.findall(r'<ol class="qz-opts">(.*?)</ol>', m.group(1), re.S)):
        for k, t in re.findall(r"<li( data-ok=\"1\")?>(.*?)<div class=\"qz-why-o\">", ol, re.S):
            print(("  *" if k else "   ") + ("s2 " if n else "") + t.strip())
