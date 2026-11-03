import re

def split_units(lines, level):
    """level('###' 등) 제목 줄에서 나눈 (시작, 끝) 줄 범위 목록"""
    pat = re.compile(r'^' + re.escape(level) + r' ')
    cuts = [i for i, l in enumerate(lines) if pat.match(l)]
    return cuts

def chunk_chapter(text, limit=30000):
    """장 본문을 검토 단위로 나눔. 반환: [(시작 줄, 끝 줄(미포함), 제목)]"""
    lines = text.split('\n')
    n = len(lines)
    # 경계 후보: '## ', '### ', '#### ' 제목 줄
    def level(l):
        m = re.match(r'^(#{2,4}) ', l)
        return len(m.group(1)) if m else 0
    heads = [(i, level(l)) for i, l in enumerate(lines) if level(l)]
    first = heads[0][0] if heads else 0
    # 장 첫머리(첫 제목 전)는 첫 단위에 붙임
    bounds = [0] + [i for i, lv in heads if i > first] + [n]
    lens = [sum(len(x) + 1 for x in lines[bounds[k]:bounds[k + 1]]) for k in range(len(bounds) - 1)]
    lv_at = {i: lv for i, lv in heads}
    chunks, start, size = [], 0, 0
    for k in range(len(bounds) - 1):
        b = bounds[k]
        lv = lv_at.get(b, 9)
        # 새 '## '(절)에서는 항상 끊고, '### '·'#### '에서는 크기가 넘칠 때만 끊음
        must = (lv == 2 and size > 0)
        over = size > 0 and size + lens[k] > limit and lv in (2, 3, 4)
        if must or over:
            chunks.append((start, b))
            start, size = b, 0
        size += lens[k]
    chunks.append((start, n))
    out = []
    for s, e in chunks:
        title = next((lines[i].lstrip('# ').strip() for i in range(s, e) if level(lines[i])), lines[s][:40])
        out.append((s, e, title))
    return out

def mark_chapter(text, limit=30000):
    """검토 단위 표식을 넣은 장 본문과 단위 목록을 돌려줌"""
    lines = text.split('\n')
    chunks = chunk_chapter(text, limit)
    ins = {}
    for k, (s, e, t) in enumerate(chunks, 1):
        ins.setdefault(s, []).append(f'〔검토 단위 {k} 시작〕')
        ins.setdefault(e, []).insert(0, f'〔검토 단위 {k} 끝〕')
    out = []
    for i in range(len(lines) + 1):
        for m in ins.get(i, []):
            out += ['', m, '']
        if i < len(lines):
            out.append(lines[i])
    return '\n'.join(out), [(k, t, sum(len(x) + 1 for x in lines[s:e])) for k, (s, e, t) in enumerate(chunks, 1)]
