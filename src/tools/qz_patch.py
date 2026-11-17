"""Apply exact-text replacements inside single quiz questions.
usage (as module): from qz_patch import patch; patch({ "qz00-s1-q02": [(old, new), ...] })
Each old string must occur exactly once inside that question block."""
import re, sys
def patch(edits):
    by_file = {}
    for qid, pairs in edits.items():
        by_file.setdefault(qid[:4], []).append((qid, pairs))
    for f, items in by_file.items():
        p = f"content/{f}.html"; s = open(p, encoding="utf-8").read()
        for qid, pairs in items:
            a = s.index(f'id="{qid}"'); b = s.index('<div class="qz-fb">', a)
            blk = s[a:b]
            for old, new in pairs:
                n = blk.count(old)
                if n != 1: sys.exit(f"{qid}: {n} matches for {old[:40]!r}")
                blk = blk.replace(old, new)
            s = s[:a] + blk + s[b:]
        open(p, "w", encoding="utf-8").write(s)
        print("patched", p, len(items), "questions")
