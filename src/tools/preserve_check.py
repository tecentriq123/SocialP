"""Check that a rewritten chapter kept everything the old one had (content is moved, not deleted).

Usage: python3 tools/preserve_check.py <old content/chNN.html> <new content/chNN.html>

Checks (prints OK or the list of problems):
  sections   same <section> ids and <h2> titles, same order
  figs       same set of <!--FIG:...--> markers; order of figures inside each section kept
  papers     every old <div class="paper"> box appears verbatim in the new file (moved whole, marks untouched)
  practice   every old practice question (<li> inside <div class="practice">) still present (by its first 40 chars)
  cites      every old <cite data-ref> key still cited
  links      every old internal link target (#...) still linked somewhere
  numbers    decimal numbers of the old text missing from the new text (prints them for a manual check;
             a missing number is fine only if it was an exact duplicate or a rounding of a number that stays)
"""
import re, sys, html as H

def div_block(src, start):
    depth, k = 0, start
    tag = re.compile(r'<(/?)div\b[^>]*>')
    for m in tag.finditer(src, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return src[start:m.end()]
    return src[start:]

def papers(src):
    return [div_block(src, m.start()) for m in re.finditer(r'<div class="paper">', src)]

def practice_q(src):
    out = []
    for m in re.finditer(r'<div class="practice">', src):
        blk = div_block(src, m.start())
        for li in re.findall(r'<li\b[^>]*>(.*?)(?:<details|</li>)', blk, re.S):
            t = re.sub(r'\s+', ' ', H.unescape(re.sub(r'<[^>]+>', '', li))).strip()
            if t:
                out.append(t[:40])
    return out

def text(src):
    return re.sub(r'\s+', ' ', H.unescape(re.sub(r'<[^>]+>', ' ', src)))

def main(a, b):
    old, new = open(a, encoding='utf-8').read(), open(b, encoding='utf-8').read()
    probs = []
    so = re.findall(r'<section class="sec" id="([^"]+)">\s*<h2>(.*?)</h2>', old, re.S)
    sn = re.findall(r'<section class="sec" id="([^"]+)">\s*<h2>(.*?)</h2>', new, re.S)
    if so != sn:
        probs.append(f'sections differ: {so} vs {sn}')
    fo, fn = re.findall(r'<!--FIG:([^>]+)-->', old), re.findall(r'<!--FIG:([^>]+)-->', new)
    if sorted(fo) != sorted(fn):
        probs.append(f'figs missing {sorted(set(fo)-set(fn))} added {sorted(set(fn)-set(fo))}')
    else:
        for sid in [s for s, _ in so]:
            def order(src):
                m = re.search(r'<section class="sec" id="%s">(.*?)</section>' % sid, src, re.S)
                return re.findall(r'<!--FIG:([^>]+)-->', m.group(1)) if m else []
            if order(old) != order(new):
                probs.append(f'figure order changed in {sid}: {order(old)} -> {order(new)} (captions are numbered)')
    pn = new
    newp = papers(new)
    def head_of(p):
        h = re.search(r'<div class="paper-h">(.*?)</div>', p, re.S)
        return text(h.group(1)).strip() if h else '?'
    def sig(p):   # marks sequence + numbers: must not change even when position wording is updated
        return (re.findall(r'<span class="mk">(\d+)</span>', p), re.findall(r'\d+(?:\.\d+)?', text(p)))
    import difflib
    for i, p in enumerate(papers(old), 1):
        if p in pn:
            continue
        cand = [q for q in newp if head_of(q) == head_of(p)]
        if not cand:
            probs.append(f'paper box {i} not found: {head_of(p)[:70]}')
            continue
        q = max(cand, key=lambda q: difflib.SequenceMatcher(None, text(p), text(q)).ratio())
        if sig(q) != sig(p):
            probs.append(f'paper box {i} changed marks or numbers: {head_of(p)[:70]}')
            continue
        sm = difflib.SequenceMatcher(None, text(p), text(q))
        ch = [(text(p)[a1:a2], text(q)[b1:b2]) for op, a1, a2, b1, b2 in sm.get_opcodes() if op != 'equal']
        print(f'note: paper box {i} reworded ({head_of(p)[:50]}):', ' | '.join(f'"{x[:60]}" -> "{y[:60]}"' for x, y in ch[:6]))
    qn = set(practice_q(new))
    for q in practice_q(old):
        if q not in qn:
            probs.append(f'practice question missing: {q}')
    co, cn = set(re.findall(r'data-ref="([^"]+)"', old)), set(re.findall(r'data-ref="([^"]+)"', new))
    if co - cn:
        probs.append(f'cites missing: {sorted(co - cn)}')
    lo, ln = set(re.findall(r'href="(#[^"]+)"', old)), set(re.findall(r'href="(#[^"]+)"', new))
    if lo - ln:
        probs.append(f'link targets no longer linked: {sorted(lo - ln)}')
    num = re.compile(r'(?<![\d.])[−-]?\d+\.\d+')
    to, tn = text(old), text(new)
    miss = sorted(set(num.findall(to)) - set(num.findall(tn)))
    print('\n'.join(probs) if probs else 'structure OK (sections, figs, paper boxes, practice, cites, links)')
    print(f'numbers in old text not found in new ({len(miss)}):', ' '.join(miss[:200]))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
