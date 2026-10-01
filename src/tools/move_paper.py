"""Move the '논문에서 먼저 보기' block of each section to just before '수식으로 보기'
(or, without formulas, before the lab-link / keypoints at the end of the section).

Block = from the first <h3> whose text starts with '논문에서' up to the next <h3>, lab-link or keypoints.
The h3 is renamed to '논문에서 읽어 보기'. Real-paper guide anchors (realpapers/<sid>.html data-after)
are remapped so each guide stays under the same paper box.

usage: python3 tools/move_paper.py [--apply] [--override sid=Heading text ...] files...
  --override ch09-s1="통계분석 방법 문단"  → insert before that h3 instead of the default target.
"""
import re, sys, os, glob

H3 = re.compile(r'<h3[^>]*>(.*?)</h3>', re.S)
NEW_TITLE = '논문에서 읽어 보기'


def strip(t):
    return re.sub(r'<[^>]+>', '', t).strip()


def paper_keys(body):
    """fingerprint of each paper box in order: header text + first 120 chars after it"""
    keys = []
    for m in re.finditer(r'<div class="paper">', body):
        keys.append(body[m.start():m.start() + 400])
    return keys


def process_section(sid, body, override=None):
    heads = [(m.start(), m.end(), strip(m.group(1))) for m in H3.finditer(body)]
    first = [h for h in heads if h[2].startswith('논문에서')]
    if not first:
        return body, 'no paper h3'
    b0, b0e, title = first[0]
    # end of block
    cands = [h[0] for h in heads if h[0] > b0]
    for pat in ('<p class="lab-link"', '<div class="keypoints">'):
        k = body.find(pat, b0e)
        if k >= 0:
            cands.append(k)
    b1 = min(cands) if cands else len(body)
    block = body[b0:b1]
    rest = body[:b0] + body[b1:]
    # target in rest
    if override:
        m = [h for h in H3.finditer(rest) if strip(h.group(1)) == override]
        if not m:
            return body, f'override heading not found: {override}'
        t = m[0].start()
    else:
        m = [h for h in H3.finditer(rest) if strip(h.group(1)) == '수식으로 보기']
        if m:
            t = m[0].start()
        else:
            k = rest.find('<p class="lab-link"')
            if k < 0:
                k = rest.find('<div class="keypoints">')
            t = k if k >= 0 else len(rest)
    if t <= b0:
        return body, 'already in place'
    block = re.sub(r'(<h3[^>]*>)\s*논문에서[^<]*(</h3>)', r'\g<1>' + NEW_TITLE + r'\g<2>', block, count=1)
    if not block.endswith('\n\n'):
        block = block.rstrip('\n') + '\n\n'
    new = rest[:t] + block + rest[t:]
    return new, f'moved "{title}" ({len(block)} chars) → before pos {t}'


def main():
    args = sys.argv[1:]
    apply = '--apply' in args
    overrides = {}
    files = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == '--apply':
            pass
        elif a == '--override':
            k, v = args[i + 1].split('=', 1)
            overrides[k] = v
            i += 1
        else:
            files.append(a)
        i += 1
    for f in files:
        src = open(f, encoding='utf-8').read()
        out = src
        for m in list(re.finditer(r'<section class="sec" id="([^"]+)">(.*?)</section>', src, re.S)):
            sid, body = m.group(1), m.group(2)
            old_keys = paper_keys(body)
            new_body, msg = process_section(sid, body, overrides.get(sid))
            print(f'{sid}: {msg}')
            if new_body == body:
                continue
            new_keys = paper_keys(new_body.replace(NEW_TITLE, NEW_TITLE))
            out = out.replace(body, new_body, 1)
            # remap real-paper anchors
            rp = os.path.join('realpapers', sid + '.html')
            if os.path.exists(rp):
                t = open(rp, encoding='utf-8').read()

                def remap(mm):
                    old = int(mm.group(1))
                    key = old_keys[min(old, len(old_keys)) - 1]
                    new = new_keys.index(key) + 1
                    return f'data-after="{new}"'
                t2 = re.sub(r'data-after="(\d+)"', remap, t)
                if t2 != t:
                    print(f'   realpapers/{sid}.html anchors: ' + ','.join(re.findall(r'data-after="(\d+)"', t)) + ' → ' + ','.join(re.findall(r'data-after="(\d+)"', t2)))
                    if apply:
                        open(rp, 'w', encoding='utf-8').write(t2)
        if apply and out != src:
            open(f, 'w', encoding='utf-8').write(out)


if __name__ == '__main__':
    main()
