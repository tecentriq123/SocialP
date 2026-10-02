"""Visible (not collapsed) text length per chapter in dist/index.html, vs the unfolded build."""
import re, sys, json
def strip_details(h):
    while True:
        n = re.sub(r'<details\b(?:(?!<details\b).)*?</details>', '', h, flags=re.S)
        if n == h: return n
        h = n
def vis(h):
    h = strip_details(h)
    h = re.sub(r'<script.*?</script>|<svg.*?</svg>|<style.*?</style>', '', h, flags=re.S)
    return len(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', h)))
def tpls(path):
    s = open(path, encoding='utf-8').read()
    return {m.group(1): m.group(2) for m in re.finditer(r'<template id="tpl-([a-z0-9]+)">(.*?)</template>', s, re.S)}
if __name__ == '__main__':
    a, b = tpls(sys.argv[1]), tpls(sys.argv[2])  # before (unfolded), after
    P = {'P1': ['ch00','ch01','ch02','ch03','ch04','ch05','ch06','ch07','rv01'], 'P2': ['ch08','ch09','ch10','ch11','ch12','ch13']}
    for k, ids in P.items():
        x = sum(vis(a[i]) for i in ids); y = sum(vis(b[i]) for i in ids)
        print(k, x, y, f'{y/x:.0%}')
        for i in ids: print('   ', i, vis(a[i]), vis(b[i]))
