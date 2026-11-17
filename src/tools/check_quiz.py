"""Check quiz chapters (content/qz*.html): one correct option per list, ids, links into stats.html, answer spread.
Usage: python3 tools/check_quiz.py [qz00 ...]"""
import re, sys, glob, collections
ids = set()
for f in glob.glob("content/*.html"):
    if "/qz" in f: continue
    ids |= set(re.findall(r'\sid="([^"]+)"', open(f, encoding="utf-8").read()))
files = [f"content/{a}.html" for a in sys.argv[1:]] or sorted(glob.glob("content/qz*.html"))
bad = 0
for f in files:
    s = open(f, encoding="utf-8").read()
    qs = re.findall(r'<div class="qz" data-widget="qz" id="([^"]+)"(.*?)\n</div>\n(?=\n|<div class="qz"|</section>)', s, re.S)
    n_open = s.count('<div class="qz" data-widget="qz"')
    if len(qs) != n_open: print(f, "parse mismatch", len(qs), n_open); bad += 1
    letters = collections.Counter(); qids = [q[0] for q in qs]
    if len(set(qids)) != len(qids): print(f, "duplicate question ids"); bad += 1
    for qid, body in qs:
        if 'data-kind="' not in body or 'data-lv="' not in body: print(qid, "missing kind/lv"); bad += 1
        for ol in re.findall(r'<ol class="qz-opts">(.*?)</ol>', body, re.S):
            lis = re.findall(r"<li(?: data-ok=\"1\")?>", ol)
            oks = [i for i, x in enumerate(lis) if "data-ok" in x]
            if len(lis) < 3 or len(oks) != 1: print(qid, "options", len(lis), "correct", len(oks)); bad += 1
            elif lis: letters["ABCDEFG"[oks[0]]] += 1
            if ol.count('<div class="qz-why-o">') != len(lis): print(qid, "option explanations", ol.count('<div class="qz-why-o">'), "of", len(lis)); bad += 1
        if '<div class="qz-fb">' not in body or "다시 읽을 곳" not in body: print(qid, "feedback/read missing"); bad += 1
        for h in re.findall(r'href="stats\.html#([^"]+)"', body):
            if h not in ids and not re.fullmatch(r"ch\d\d", h): print(qid, "broken link", h); bad += 1
        if body.count("<div") != body.count("</div>") - 0 and body.count("<div") != body.count("</div>"): pass
    for h in re.findall(r'href="stats\.html#([^"]+)"', s):
        if h not in ids and not re.fullmatch(r"ch\d\d", h): print(f, "broken link", h); bad += 1
    print(f, len(qs), "questions; correct letters", dict(sorted(letters.items())))
print("OK" if not bad else f"{bad} problem(s)")
