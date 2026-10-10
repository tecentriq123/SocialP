"""Readability measures of the VISIBLE (not collapsed) text, per section, from a built page.

Usage: python3 tools/readability.py <built.html> ch02 [ch03 ...]
Build first, e.g. python3 build.py --only ch02 --out <scratch>/t.html  (the default build folds 심화 parts).

Per section it prints:
  vis      visible characters (collapsed <details> removed, figures/SVG/scripts removed)
  sent     mean sentence length in characters (visible prose: <p>, <li>, <dd>, <td> excluded)
  paren    '(' per 1,000 visible characters of prose
  h3       visible <h3> subsections / collapsed ones
  boxes    visible boxes: paper, callout, easy, glance, flow, matrix, tree
"""
import re, sys, html as H

def strip_details(h):
    while True:
        n = re.sub(r'<details\b(?:(?!<details\b).)*?</details>', '', h, flags=re.S)
        if n == h:
            return n
        h = n

def text(h):
    return re.sub(r'\s+', ' ', H.unescape(re.sub(r'<[^>]+>', '', h))).strip()

def prose(h):
    h = re.sub(r'<(table|pre|svg|figure|script|style)\b.*?</\1>', '', h, flags=re.S)
    h = re.sub(r'<p class="excerpt".*?</p>', '', h, flags=re.S)   # English paper excerpts
    parts = re.findall(r'<(p|li|dd)\b[^>]*>(.*?)</\1>', h, flags=re.S)
    return [text(b) for _, b in parts if len(text(b)) > 15]

def sentences(paras):
    out = []
    for p in paras:
        # a sentence ends with 다/요/까 + '.', optionally followed by a citation [n] or a closing ')' before the period
        for s in re.split(r'(?:(?<=[다요까])|(?<=[다요까]\[\d\])|(?<=[다요까]\[\d\d\])|(?<=\)))\.(?:\s+|$)', p):
            s = s.strip()
            if len(s) > 8 and re.search(r'[가-힣]', s):
                out.append(s)
    return out

def measure(sec_html):
    vis_h = strip_details(sec_html)
    vis_h2 = re.sub(r'<script.*?</script>|<svg.*?</svg>|<style.*?</style>', '', vis_h, flags=re.S)
    vis = len(text(vis_h2))
    ps = prose(vis_h)
    ss = sentences(ps)
    ptxt = ' '.join(ps)
    sent = sum(len(s) for s in ss) / max(len(ss), 1)
    paren = ptxt.count('(') / max(len(ptxt), 1) * 1000
    h3v = len(re.findall(r'<h3\b', vis_h))
    h3all = len(re.findall(r'<h3\b', sec_html))
    boxes = len(re.findall(r'<div class="(paper|callout[^"]*|easy|flow|matrix|tree)"|<dl class="glance"', vis_h))
    return vis, sent, paren, h3v, h3all - h3v, boxes

if __name__ == '__main__':
    page = open(sys.argv[1], encoding='utf-8').read()
    tpl = {m.group(1): m.group(2) for m in re.finditer(r'<template id="tpl-([a-z0-9]+)">(.*?)</template>', page, re.S)}
    print(f'{"section":10} {"vis":>6} {"sent":>5} {"paren":>5} {"h3 vis/fold":>11} {"boxes":>5}')
    for cid in sys.argv[2:]:
        body = tpl.get(cid)
        if body is None:
            print(cid, 'not in page'); continue
        tot = [0, 0]
        for m in re.finditer(r'<section class="sec" id="([^"]+)">(.*?)</section>', body, re.S):
            v, s, p, h3v, h3f, b = measure(m.group(2))
            tot[0] += v
            print(f'{m.group(1):10} {v:6d} {s:5.0f} {p:5.1f} {h3v:6d}/{h3f:<4d} {b:5d}')
        print(f'{cid+" total":10} {tot[0]:6d}')
