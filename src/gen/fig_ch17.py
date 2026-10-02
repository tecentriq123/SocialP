"""Figures for chapter 17 (경쟁위험 분석).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch17.py && python3 gen/fig_ch17.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import pandas as pd
from svgplot import Plot, figure, panel_title

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch17_nums.json")))


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


d = pd.read_csv(os.path.join(HERE, "_ch17_cr.csv"))
grid = np.linspace(0, 5, 301)


def aj_curves(s):
    """Aalen-Johansen CIFs (dialysis, death) and 1 - KM for dialysis with deaths censored, on the grid"""
    t = s.t.values; e = s.ev.values
    o = np.argsort(t, kind="stable"); t, e = t[o], e[o]
    n = len(t); S = 1.0; c1 = c2 = 0.0; Skm = 1.0
    T, C1, C2, K = [0.0], [0.0], [0.0], [0.0]
    for i in range(len(t)):
        at = n - i
        if e[i] == 1:
            c1 += S / at; Skm *= (1 - 1 / at)
        elif e[i] == 2:
            c2 += S / at
        if e[i] in (1, 2):
            S *= (1 - 1 / at)
        T.append(t[i]); C1.append(c1); C2.append(c2); K.append(1 - Skm)
    T = np.array(T)
    idx = np.searchsorted(T, grid, side="right") - 1
    return np.array(C1)[idx], np.array(C2)[idx], np.array(K)[idx]


cA = aj_curves(d[d.grp == "A"])
cB = aj_curves(d[d.grp == "B"])
# consistency with the numbers script
assert abs(cB[0][-1] - N["B"]["y5"]["cif_d"]) < 1e-6 and abs(cB[2][-1] - N["B"]["y5"]["km_d"]) < 1e-6
assert abs(cA[0][-1] - N["A"]["y5"]["cif_d"]) < 1e-6 and abs(cA[1][-1] - N["A"]["y5"]["cif_m"]) < 1e-6
pc = lambda v: f"{v * 100:.0f}%"
P1 = lambda v: f"{v * 100:.1f}%"

# ============================================================ 그림 17-1: 1 - KM vs CIF, and where patients go
p1 = Plot((0, 5), (0, 0.6), w=420, h=330, ml=58, mr=88, xlabel="약물 시작 후 연수", ylabel="투석을 시작한 누적 비율",
          yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ytickfmt=pc, xticks=[0, 1, 2, 3, 4, 5])
p1.fill_between(grid, cB[0], cB[2], s=2)
p1.line(grid, cB[2], s=2, dash=True, w=2.2)
p1.line(grid, cB[0], s=1, w=2.4)
p1.text(5, cB[2][-1], f"1 − KM {P1(cB[2][-1])}", dx=6, dy=4, cls="lbl small")
p1.text(5, cB[0][-1], f"CIF {P1(cB[0][-1])}", dx=6, dy=4, cls="lbl small")
p1.legend([("1 − Kaplan-Meier (사망을 중도절단)", 2, "dash"), ("누적발생함수 (사망을 경쟁 사건)", 1, "line")])
panel_title(p1, "(가) 두 방법으로 구한 투석 누적 비율")

free = 1 - cB[0] - cB[1]
p2 = Plot((0, 5), (0, 1.0), w=420, h=330, ml=58, mr=88, xlabel="약물 시작 후 연수", ylabel="환자의 비율",
          yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=pc, xticks=[0, 1, 2, 3, 4, 5])
zero = np.zeros_like(grid)
p2.fill_between(grid, zero, cB[0], s=1)
p2.fill_between(grid, cB[0], cB[0] + cB[1], s=2)
p2.fill_between(grid, cB[0] + cB[1], np.ones_like(grid), s=4)
p2.line(grid, cB[0], s=1, w=2.2)
p2.line(grid, cB[0] + cB[1], s=2, w=2.2)
p2.text(5, cB[0][-1] / 2, f"투석 {P1(cB[0][-1])}", dx=6, dy=4, cls="lbl small")
p2.text(5, cB[0][-1] + cB[1][-1] / 2, f"사망 {P1(cB[1][-1])}", dx=6, dy=4, cls="lbl small")
p2.text(5, 1 - free[-1] / 2, f"무사건 {P1(free[-1])}", dx=6, dy=4, cls="lbl small")
p2.text(0.25, 0.86, "사건 없이 추적 중", cls="lbl small")
p2.text(2.6, 0.40, "투석 전 사망", cls="lbl small")
p2.text(3.2, 0.07, "투석 시작", cls="lbl small")
panel_title(p2, "(나) 세 상태로 나눈 환자의 비율")
save("ch17_cif", figure(
    [p1.svg("약물 B군에서 1-Kaplan-Meier와 누적발생함수 비교"), p2.svg("약물 B군 환자가 시간에 따라 투석, 투석 전 사망, 무사건 상태에 있는 비율")],
    "그림 17-1. 약물 B군 2,120명의 투석 시작(가상의 예시). (가) 사망을 중도절단으로 처리한 1 − Kaplan-Meier(점선)는 5년 투석 비율을 "
    f"{P1(cB[2][-1])}로, 사망을 경쟁 사건으로 다룬 누적발생함수(실선)는 {P1(cB[0][-1])}로 추정합니다. 색칠한 부분이 과대추정된 크기입니다. "
    f"(나) 누적발생함수를 쌓아 그리면 어느 시점에서나 투석, 투석 전 사망, 무사건의 합이 100%입니다. 5년에 {P1(cB[0][-1])} + {P1(cB[1][-1])} + {P1(free[-1])}입니다.",
    cols=2))

# ============================================================ 그림 17-2: two groups, and the two risk sets
p3 = Plot((0, 5), (0, 0.6), w=420, h=330, ml=58, mr=88, xlabel="약물 시작 후 연수", ylabel="누적발생함수",
          yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ytickfmt=pc, xticks=[0, 1, 2, 3, 4, 5])
p3.line(grid, cB[1], s=2, dash=True, w=2.0)
p3.line(grid, cA[1], s=1, dash=True, w=2.0)
p3.line(grid, cB[0], s=2, w=2.4)
p3.line(grid, cA[0], s=1, w=2.4)
p3.text(5, cB[1][-1], f"B 사망 {P1(cB[1][-1])}", dx=6, dy=4, cls="lbl small")
p3.text(5, cA[1][-1], f"A 사망 {P1(cA[1][-1])}", dx=6, dy=0, cls="lbl small")
p3.text(5, cA[0][-1], f"A 투석 {P1(cA[0][-1])}", dx=6, dy=8, cls="lbl small")
p3.text(5, cB[0][-1], f"B 투석 {P1(cB[0][-1])}", dx=6, dy=8, cls="lbl small")
p3.legend([("약물 A", 1, "line"), ("약물 B", 2, "line")])
p3.text(0.15, 0.44, "실선: 투석, 점선: 투석 전 사망", cls="lbl small")
panel_title(p3, "(가) 두 군의 투석과 사망")

# risk sets in group B over time
s = d[d.grp == "B"]
tg = np.linspace(0, 5, 251)
rs_cs = np.array([(s.t >= u).sum() for u in tg])
rs_sd = np.array([((s.t >= u) | ((s.ev == 2) & (s.C >= u))).sum() for u in tg])
for y in (1, 2, 3, 4, 5):
    i = int(np.argmin(np.abs(tg - y)))
    assert rs_cs[i] == N["B"][f"y{y}"]["rs_cs"] and rs_sd[i] == N["B"][f"y{y}"]["rs_sd"], (y, rs_cs[i], rs_sd[i])
p4 = Plot((0, 5), (0, 2200), w=420, h=330, ml=58, mr=88, xlabel="약물 시작 후 연수", ylabel="위험집합의 인원 (약물 B군)",
          yticks=[0, 500, 1000, 1500, 2000], ytickfmt=lambda v: f"{v:,.0f}", xticks=[0, 1, 2, 3, 4, 5])
p4.fill_between(tg, rs_cs, rs_sd, s=2)
p4.line(tg, rs_sd, s=2, w=2.2)
p4.line(tg, rs_cs, s=1, w=2.4)
i3 = int(np.argmin(np.abs(tg - 3)))
p4.vline(3, y0=0, y1=rs_sd[i3])
p4.text(3, rs_sd[i3], f"{rs_sd[i3]:,}명", dx=5, dy=-6, cls="lbl small")
p4.text(3, rs_cs[i3], f"{rs_cs[i3]:,}명", dx=5, dy=-6, cls="lbl small")
p4.text(5, rs_sd[-1], "하위분포", dx=6, dy=0, cls="lbl small")
p4.text(5, rs_sd[-1], "위험집합", dx=6, dy=14, cls="lbl small")
p4.text(5, rs_cs[-1], "원인별", dx=6, dy=0, cls="lbl small")
p4.text(5, rs_cs[-1], "위험집합", dx=6, dy=14, cls="lbl small")
p4.text(1.5, 1500, "남겨 둔 사망자", cls="lbl small")
panel_title(p4, "(나) 두 위험비가 쓰는 위험집합")
save("ch17_two", figure(
    [p3.svg("두 군의 투석과 투석 전 사망 누적발생함수"), p4.svg("약물 B군에서 원인별 위험집합과 하위분포 위험집합의 인원")],
    "그림 17-2. (가) 약물 A군은 투석 전 사망이 적고(점선) 투석까지 가는 환자가 더 많습니다(실선). 살아 있는 환자가 투석으로 넘어가는 속도는 "
    "두 군이 비슷합니다(원인별 위험비 1.02). (나) 원인별 위험비는 사망자를 그 시점에 빼고 계산하고, 하위분포 위험비는 사망자를 "
    f"추적 종료 예정일까지 남겨 두고 계산합니다. 약물 B군의 3년 시점 위험집합은 {rs_cs[i3]:,}명과 {rs_sd[i3]:,}명이고, 색칠한 부분이 남겨 둔 사망자입니다.",
    cols=2))
print("figures written")
