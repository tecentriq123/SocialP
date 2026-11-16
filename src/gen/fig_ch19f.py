"""Figure 19-10: network of the 19장 f 절 example (nodes = treatments, edges = direct comparisons).
Run: python3 gen/fig_ch19f.py   (writes figs/ch19_network.html)"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from svgplot import Plot, figure
from nums_ch19f import network

rows = network()
arm_n = {}
for r in rows:
    arm_n[r["t1"]] = arm_n.get(r["t1"], 0) + r["n1"]; arm_n[r["t0"]] = arm_n.get(r["t0"], 0) + r["n0"]
edges = {}
for r in rows:
    k = tuple(sorted((r["t1"], r["t0"]))); edges[k] = edges.get(k, 0) + 1
pos = {"X": (135, 78), "Y": (405, 78), "P": (270, 262), "Z": (85, 305)}
name = {"X": "약물 X", "Y": "약물 Y", "Z": "약물 Z", "P": "위약"}
cls = {"X": "1", "Y": "2", "Z": "3", "P": "4"}
p = Plot((0, 1), (0, 1), w=580, h=375, ml=0, mr=0, mt=0, mb=0, show_xaxis=False, show_yaxis=False, ygrid=False)
rad = {t: 9 + 26 * math.sqrt(n / max(arm_n.values())) for t, n in arm_n.items()}
for (a, b), k in edges.items():
    (x1, y1), (x2, y2) = pos[a], pos[b]
    p.els.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="ln" style="stroke:var(--muted)" stroke-width="{1.5 + 1.6 * k:.1f}"/>')
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = y2 - y1, -(x2 - x1); L = math.hypot(dx, dy); ox, oy = dx / L * 24, dy / L * 24
    cx, cy = (pos["X"][0] + pos["Y"][0] + pos["P"][0]) / 3, (pos["X"][1] + pos["Y"][1] + pos["P"][1]) / 3
    if (mx + ox - cx) ** 2 + (my + oy - cy) ** 2 < (mx - ox - cx) ** 2 + (my - oy - cy) ** 2: ox, oy = -ox, -oy
    if (a, b) == ("P", "Z"): ox, oy = abs(ox), abs(oy)
    p.text_px(mx + ox, my + oy + 4, f"시험 {k}개", anchor="middle", cls="lbl small")
for t, (x, y) in pos.items():
    p.els.append(f'<circle cx="{x}" cy="{y}" r="{rad[t]:.1f}" class="f{cls[t]}" stroke="var(--surface)" stroke-width="2"/>')
    ty = y - rad[t] - 10 if y < 150 else y + rad[t] + 18
    p.text_px(x, ty, name[t], anchor="middle", cls="lbl strong")
    p.text_px(x, ty + 15 if y >= 150 else ty - 15, f"{arm_n[t]:,}명", anchor="middle", cls="lbl mute small")
html = figure(p.svg("네트워크 그림: 약물 X, Y, Z와 위약을 직접 비교한 시험"),
              "그림 19-10. 이 절의 네트워크(가상 자료, 시험 19개, 54,285명). 원은 치료이고 크기는 그 치료에 배정된 환자 수, 선은 두 치료를 직접 비교한 시험이 있다는 뜻이며 굵기는 시험 수입니다. "
              "약물 X·약물 Y·위약은 세 변이 모두 이어진 고리를 이루고, 약물 Z는 위약하고만 이어져 있습니다.")
open(os.path.join(HERE, "..", "figs", "ch19_network.html"), "w", encoding="utf-8").write(html)
print("ok", arm_n, edges)
