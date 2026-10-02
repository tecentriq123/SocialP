"""Figures for chapter 22 (효용과 QALY).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch22.py && python3 gen/fig_ch22.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from svgplot import Plot, figure, panel_title

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch22_nums.json"), encoding="utf-8"))
f3 = lambda v: f"{v:.3f}"
f2 = lambda v: f"{v:.2f}"
f1 = lambda v: f"{v:.1f}"


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


VIS = [0, 3, 6, 12, 18, 24]
ex = N["ex"]

# ============================================================ 그림 22-1: 두 환자의 효용 변화
p = Plot((0, 24), (0, 1.0), w=600, h=340, mr=96, xlabel="치료 시작 후 개월", ylabel="효용 (EQ-5D-5L 지수)",
         xticks=VIS, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=f1)
p.vline(9, y0=0, y1=0.93)
p.text(9, 0.93, "환자 나: 9개월째 진행", dx=5, dy=-2, cls="lbl small")
for k, s in (("가", 1), ("나", 2)):
    p.line(ex[k]["t"], ex[k]["path_u"], s=s, w=2.4)
    p.points(ex[k]["t"], ex[k]["path_u"], s=s, r=4)
p.text(24, ex["가"]["u"][-1], "환자 가", dx=10, dy=-2, cls="lbl")
p.text(24, ex["가"]["u"][-1], f"24개월 {f3(ex['가']['u'][-1])}", dx=10, dy=14, cls="lbl small")
p.text(24, ex["나"]["u"][-1], "환자 나", dx=10, dy=-2, cls="lbl")
p.text(24, ex["나"]["u"][-1], f"24개월 {f3(ex['나']['u'][-1])}", dx=10, dy=14, cls="lbl small")
p.text(0.4, 0.06, "0 = 사망", cls="lbl small")
p.text(0.4, 0.96, "1 = 완전한 건강", cls="lbl small")
save("ch22_profile", figure(
    p.svg("두 환자의 24개월 동안의 효용 변화"),
    "그림 22-1. 같은 24개월을 생존한 두 환자의 효용 변화(가상의 예시). 환자 가는 무진행 상태로 0.84~0.86을 유지했고, "
    f"환자 나는 9개월째 질병이 진행한 뒤 {f3(ex['나']['u'][3])}에서 {f3(ex['나']['u'][-1])}까지 낮아졌습니다. "
    "생존기간만 보면 두 환자의 결과는 같습니다. 선의 높이가 삶의 질이고, 선 아래의 넓이가 다 절에서 계산할 QALY입니다."))

# ============================================================ 그림 22-2: 시간교환과 표준 도박
tto = N["tto"]
p1 = Plot((0, 11), (0, 1.15), w=420, h=320, ml=56, mr=18, xlabel="생존 연수", ylabel="효용",
          xticks=[0, 2, 4, 6, 7, 8, 10], yticks=[0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0], ytickfmt=f1)
p1.fill_between([0, 10], [0, 0], [tto["u"], tto["u"]], s=2)
p1.line([0, 10, 10], [tto["u"], tto["u"], 0], s=2, w=2.2)
p1.fill_between([0, 7], [0.7, 0.7], [1, 1], s=1)
p1.line([0, 7, 7], [1, 1, 0], s=1, w=2.2, dash=True)
p1.text(3.5, 1.0, "완전한 건강으로 7년", anchor="middle", dy=-8, cls="lbl small")
p1.text(8.5, 0.7, "상태 X로", anchor="middle", dy=-22, cls="lbl small")
p1.text(8.5, 0.7, "10년", anchor="middle", dy=-8, cls="lbl small")
p1.text(3.5, 0.35, "넓이 7 × 1 = 10 × 0.7", anchor="middle", cls="lbl small")
panel_title(p1, "(가) 시간교환")

sg = N["sg"]
p2 = Plot((0, 10), (0, 10), w=420, h=320, ml=14, mr=14, mt=22, mb=20, ygrid=False, show_yaxis=False, show_xaxis=False)
# 선택 마디
p2.points([1.0], [5.0], s=4, r=5)
p2.seg(1.0, 5.0, 3.2, 8.0, cls="ln s1", w=2)
p2.seg(1.0, 5.0, 3.2, 2.0, cls="ln s2", w=2)
p2.text(0.4, 6.1, "선택", cls="lbl small")
# 도박
p2.points([3.2], [8.0], s=1, r=5, hollow=True)
p2.seg(3.2, 8.0, 6.0, 9.2, cls="ln s1", w=2)
p2.seg(3.2, 8.0, 6.0, 6.8, cls="ln s1", w=2)
p2.text(2.9, 8.3, "도박", anchor="end", cls="lbl small")
p2.text(6.2, 9.2, "완전한 건강 (효용 1)", dy=4, cls="lbl small")
p2.text(6.2, 6.8, "즉시 사망 (효용 0)", dy=4, cls="lbl small")
p2.text(4.3, 9.2, "확률 p", anchor="middle", dy=-6, cls="lbl small")
p2.text(4.3, 6.9, "확률 1 − p", anchor="middle", dy=16, cls="lbl small")
# 확실한 상태
p2.seg(3.2, 2.0, 6.0, 2.0, cls="ln s2", w=2)
p2.text(4.6, 2.0, "확실", anchor="middle", dy=-8, cls="lbl small")
p2.text(6.2, 2.0, "상태 X로 계속 생활", dy=4, cls="lbl small")
p2.text(0.4, 0.2, f"p = {sg['p']:.2f}에서 둘이 같게 느껴지면 상태 X의 효용 = {sg['u']:.2f}", cls="lbl small")
panel_title(p2, "(나) 표준 도박")
save("ch22_direct", figure(
    [p1.svg("시간교환법: 상태 X로 10년과 완전한 건강으로 7년의 넓이가 같다"),
     p2.svg("표준 도박법: 확실한 상태 X와 완전한 건강 또는 즉시 사망의 도박 사이의 선택")],
    "그림 22-2. 효용을 직접 재는 두 가지 질문. (가) 시간교환: 두 직사각형의 넓이가 같다고 느끼는 지점을 찾습니다. "
    "10 × 효용 = 7 × 1이므로 효용은 0.7입니다. (나) 표준 도박: 확실한 상태 X와 도박이 같게 느껴지는 성공 확률 p가 상태 X의 효용입니다.", cols=2))

# ============================================================ 그림 22-3: 곡선 아래 면적
e = ex["다"]
t, u = np.array(e["t"]), np.array(e["path_u"])
p3 = Plot((0, 24), (0, 1.0), w=420, h=330, ml=56, mr=20, xlabel="치료 시작 후 개월", ylabel="효용",
          xticks=VIS + [15], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=f1)
p3.fill_between(t, np.zeros_like(t), u, s=2)
p3.line(t, u, s=2, w=2.4)
p3.points(t[:5], u[:5], s=2, r=4)
for i, s_ in enumerate(e["steps"][:4]):
    if i > 0:
        p3.vline(s_["t0"], y0=0, y1=s_["u0"])
    mid = (s_["t0"] + s_["t1"]) / 2
    p3.text(mid, 0.22 if i < 3 else 0.08, f3(s_["area"]), anchor="middle", cls="lbl small")
p3.text(15, 0.0, "15개월 사망", dx=6, dy=-30, cls="lbl small")
p3.text(15, 0.0, "이후 효용 0", dx=6, dy=-16, cls="lbl small")
p3.text(23.5, 0.93, f"합계 {f3(e['qaly'])} QALY", anchor="end", cls="lbl strong")
panel_title(p3, "(가) 환자 다의 QALY")

T = N["trial"]
g = np.array(T["grid"])
pa, pb = np.array(T["prof"]["A"]), np.array(T["prof"]["B"])
p4 = Plot((0, 24), (0, 1.0), w=420, h=330, ml=56, mr=20, xlabel="치료 시작 후 개월", ylabel="평균 효용 (사망자는 0)",
          xticks=VIS, yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=f1)
p4.fill_between(g, np.zeros_like(g), pb, s=2)
p4.fill_between(g, pb, pa, s=1)
p4.line(g, pb, s=2, w=2.2)
p4.line(g, pa, s=1, w=2.4)
iv = [int(np.argmin(np.abs(g - v))) for v in VIS]
p4.points(g[iv], pa[iv], s=1, r=3.5)
p4.points(g[iv], pb[iv], s=2, r=3.5)
p4.legend([(f"신약 A: 평균 {f3(T['qaly']['A'])} QALY", 1, "line"), (f"표준요법 B: 평균 {f3(T['qaly']['B'])} QALY", 2, "line")],
          X=p4.ml + 110, Y=p4.mt + 12)
p4.text(12, 0.25, "B군 곡선 아래 넓이", anchor="middle", cls="lbl small")
p4.text(18.5, 0.70, f"두 곡선 사이 = 차이 {f3(T['qaly']['unadj']['diff'])}", anchor="middle", cls="lbl small")
panel_title(p4, "(나) 두 군의 평균 효용 곡선")
save("ch22_auc", figure(
    [p3.svg("환자 다의 효용 곡선 아래 면적을 사다리꼴로 나눠 계산한 그림"),
     p4.svg("신약 A군과 표준요법 B군의 평균 효용 곡선과 그 아래 면적")],
    f"그림 22-3. QALY는 효용 곡선 아래의 넓이입니다(가상의 예시). (가) 환자 다의 측정값을 직선으로 잇고 사망 시점에 0으로 내린 뒤 "
    f"사다리꼴 네 개의 넓이(구간별 숫자)를 더하면 {f3(e['qaly'])} QALY입니다. (나) 시험 환자의 효용 경로를 군별로 평균한 곡선입니다. "
    f"사망한 환자가 0으로 들어가므로 곡선이 내려갑니다. 곡선 아래 넓이가 군의 평균 QALY이고, 두 곡선 사이의 넓이가 보정 전 차이 {f3(T['qaly']['unadj']['diff'])}입니다.", cols=2))

# ============================================================ 그림 22-4: 모형의 생존연수와 QALY를 상태별로 나누기
M = N["model"]
p5 = Plot((0, 6), (0, 4.0), w=600, h=340, mr=24, ylabel="년 또는 QALY (할인 후)", yticks=[0, 1, 2, 3, 4],
          xticklabels=[(1, "A 생존연수"), (2, "A QALY"), (4, "B 생존연수"), (5, "B QALY")], xticks=[1, 2, 4, 5])
def stack(pl, x, lo, hi, s_, w=0.7):
    """lo~hi 높이의 직사각형 한 칸(겹치지 않게 쌓기)"""
    xa, xb = x - w / 2, x + w / 2
    pl.fill_between([xa, xb], [lo, lo], [hi, hi], s=s_)
    pl.line([xa, xa, xb, xb, xa], [lo, hi, hi, lo, lo], s=s_, w=1.2)


for x0, arm in ((1, "A"), (4, "B")):
    m = M[arm]
    # 생존연수: 무진행 + 진행
    stack(p5, x0, 0, m["lyd_pf"], 1)
    stack(p5, x0, m["lyd_pf"], m["ly_d"], 2)
    p5.text(x0, m["lyd_pf"] / 2, f"무진행 {f3(m['lyd_pf'])}", anchor="middle", dy=4, cls="lbl small")
    p5.text(x0, m["lyd_pf"] + m["lyd_pd"] / 2, f"진행 {f3(m['lyd_pd'])}", anchor="middle", dy=4, cls="lbl small")
    p5.text(x0, m["ly_d"], f3(m["ly_d"]), anchor="middle", dy=-7, cls="lbl strong")
    # QALY
    tot = m["q_pf"] + m["q_pd"]
    stack(p5, x0 + 1, 0, m["q_pf"], 1)
    stack(p5, x0 + 1, m["q_pf"], tot, 2)
    p5.text(x0 + 1, m["q_pf"] / 2, f"무진행 {f3(m['q_pf'])}", anchor="middle", dy=4, cls="lbl small")
    p5.text(x0 + 1, m["q_pf"] + m["q_pd"] / 2, f"진행 {f3(m['q_pd'])}", anchor="middle", dy=4, cls="lbl small")
    p5.text(x0 + 1, tot, f3(m["qaly"]), anchor="middle", dy=-7, cls="lbl strong")
p5.text(1.5, 3.8, "신약 A", anchor="middle", cls="lbl")
p5.text(4.5, 3.8, "표준요법 B", anchor="middle", cls="lbl")
save("ch22_decomp", figure(
    p5.svg("공통 예시 모형에서 두 군의 할인 후 생존연수와 QALY를 무진행과 진행 상태로 나눈 막대그림"),
    f"그림 22-4. 공통 예시 모형의 생존연수와 QALY(할인 후). 무진행 기간에 {M['u_pf']:.2f}, 진행 기간에 {M['u_pd']:.2f}를 곱하므로 "
    f"QALY 막대가 생존연수 막대보다 낮습니다. 막대 위의 QALY는 두 부분의 합에서 이상반응 손실({M['A']['du']:.3f}, {M['B']['du']:.3f})을 뺀 값입니다."))
print("figures written")
