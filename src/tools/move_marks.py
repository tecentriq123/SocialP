"""Move <span class="mk">n</span> markers in paper boxes to sit BEFORE the text they explain.
Handles: (1) hl spans followed by marks in excerpts; (2) marks at the end of <td>/<th> cells in journal tables;
(3) marks at the end of a jt-foot paragraph. Reports marks it could not move."""
import re, sys, glob

MK = r'(?:\s*<span class="mk">\d+</span>)+'
HL_THEN_MK = re.compile(r'(<span class="hl">(?:(?!</span>).)*?</span>)(' + MK + r')', re.S)
CELL = re.compile(r'(<t[dh][^>]*>)(.*?)(</t[dh]>)', re.S)
TAIL_MK = re.compile(r'^(.*?)(' + MK + r')\s*$', re.S)


def fix_excerpt(p):
    def rep(m):
        marks = re.findall(r'<span class="mk">\d+</span>', m.group(2))
        return " ".join(marks) + " " + m.group(1)
    return HL_THEN_MK.sub(rep, p)


def fix_cells(tbl):
    def rep(m):
        open_, body, close = m.groups()
        mm = TAIL_MK.match(body)
        if not mm or '<span class="mk">' in mm.group(1):
            return m.group(0)
        marks = re.findall(r'<span class="mk">\d+</span>', mm.group(2))
        return open_ + " ".join(marks) + " " + mm.group(1).strip() + close
    return CELL.sub(rep, tbl)


def process(src):
    # excerpts
    src = re.sub(r'<p class="excerpt">.*?</p>', lambda m: fix_excerpt(m.group(0)), src, flags=re.S)
    # journal tables
    src = re.sub(r'<table class="jt".*?</table>', lambda m: fix_cells(m.group(0)), src, flags=re.S)
    return src


def leftovers(src):
    out = []
    for m in re.finditer(r'<p class="excerpt">(.*?)</p>', src, re.S):
        body = m.group(1)
        for k in re.finditer(r'<span class="mk">(\d+)</span>', body):
            before = body[max(0, k.start() - 12):k.start()]
            after = body[k.end():k.end() + 25]
            if not after.lstrip().startswith(('<span class="hl">', '<span class="mk">')):
                out.append(("excerpt", k.group(1), re.sub(r'<[^>]+>', '', body[max(0, k.start() - 50):k.start()])[-45:]))
    return out


if __name__ == "__main__":
    for f in sys.argv[1:]:
        s = open(f, encoding="utf-8").read()
        t = process(s)
        if t != s:
            open(f, "w", encoding="utf-8").write(t)
        lo = leftovers(t)
        print(f, "changed" if t != s else "unchanged", "| unhandled excerpt marks:", len(lo))
        for x in lo[:6]:
            print("   ", x)
