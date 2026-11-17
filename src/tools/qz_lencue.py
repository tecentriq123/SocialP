"""Flag questions whose correct option is much longer than every distractor (a length cue).
usage: python3 tools/qz_lencue.py [qzNN ...]   (ratio threshold 1.35)"""
import re, sys, glob
files = [f"content/{a}.html" for a in sys.argv[1:]] or sorted(glob.glob("content/qz*.html"))
strip = lambda h: re.sub(r"<[^>]+>", "", re.sub(r'<div class="qz-why-o">.*?</div>', "", h, flags=re.S)).strip()
for f in files:
    s = open(f, encoding="utf-8").read(); tot = longest = 0
    for m in re.finditer(r'<div class="qz"[^>]*id="([^"]+)"(.*?)<div class="qz-fb">', s, re.S):
        qid, body = m.group(1), m.group(2)
        for ol in re.findall(r'<ol class="qz-opts">(.*?)</ol>', body, re.S):
            lis = re.findall(r"<li( data-ok=\"1\")?>(.*?)</li>", ol, re.S)
            ok = [len(strip(t)) for k, t in lis if k]; bad = [len(strip(t)) for k, t in lis if not k]
            if ok and bad:
                tot += 1; longest += ok[0] > max(bad)
            if ok and bad and ok[0] > 1.35 * max(bad):
                print(f"{qid}: correct {ok[0]} vs longest distractor {max(bad)} (x{ok[0]/max(bad):.2f})")
    print(f"# {f}: correct option is the longest in {longest}/{tot}")
