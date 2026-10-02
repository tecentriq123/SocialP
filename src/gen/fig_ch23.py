"""Figures for chapter 23 (결정분석 모형).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch23.py && python3 gen/fig_ch23.py
Writes figs/ch23_tree.html, ch23_states.html, ch23_trace.html, ch23_psm.html, ch23_compare.html, ch23_extrap.html
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from svgplot import Plot, figure, panel_title
import lib_p4 as L

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch23_nums.json"), encoding="utf-8"))
p = L.base_params()
T = np.arange(0, L.HORIZON_MONTHS + 1) * L.CYCLE_MONTHS


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def canvas(w, h):
    return Plot((0, 1), (0, 1), w=w, h=h, ml=0, mr=0, mt=0, mb=0, show_xaxis=False, show_yaxis=False, ygrid=False)


pc = lambda v: f"{v * 100:.0f}%"
won = lambda v: f"{v:,.0f}만원"

# ====================================================================== 그림 23-1: 결정수형
tp = N["tp"]
W, H = 700, 372
c = canvas(W, H)
XD, XC1, XC2, XT = 34, 196, 372, 500          # 결정 마디, 첫 확률 마디, 둘째 확률 마디, 종결 마디의 x


def branch(x1, y1, x2, y2, s):
    """꺾은선 가지: (x1, y1)에서 세로로 y2까지 간 뒤 가로로 x2까지"""
    c.els.append(f'<path d="M{x1:.1f},{y1:.1f} L{x1 + 14:.1f},{y2:.1f} L{x2:.1f},{y2:.1f}" class="ln s{s}" stroke-width="1.6" fill="none"/>')


def chance(x, y, s):
    c.top.append(f'<circle cx="{x}" cy="{y}" r="8" class="pth s{s}"/>')


def terminal(x, y, s):
    c.top.append(f'<polygon points="{x - 2},{y} {x + 12},{y - 8} {x + 12},{y + 8}" class="f{s}"/>')


def strategy(y0, s, key, title, p_ae, add):
    paths = N["tree"][key]["paths"]
    ys = [y0 - 52, y0 - 4, y0 + 48]             # 입원, 외래, 이상반응 없음
    ya = (ys[0] + ys[1]) / 2                    # 둘째 확률 마디
    branch(XD + 9, 186, XC1 - 8, y0, s)
    c.text_px(XD + 36, y0 - 8, title, cls="lbl strong")
    branch(XC1 + 8, y0, XC2 - 8, ya, s)
    c.text_px(XC1 + 30, ya - 7, f"중증 이상반응 {p_ae:.2f}", cls="lbl small")
    branch(XC1 + 8, y0, XT - 2, ys[2], s)
    c.text_px(XC1 + 30, ys[2] - 7, f"없음 {1 - p_ae:.2f}", cls="lbl small")
    branch(XC2 + 8, ya, XT - 2, ys[0], s)
    c.text_px(XC2 + 30, ys[0] - 7, f"입원 {tp['p_hosp']:.2f}", cls="lbl small")
    branch(XC2 + 8, ya, XT - 2, ys[1], s)
    c.text_px(XC2 + 30, ys[1] - 7, f"외래 치료 {1 - tp['p_hosp']:.2f}", cls="lbl small")
    chance(XC1, y0, s); chance(XC2, ya, s)
    for (names, pr, cost, q), y in zip(paths, ys):
        terminal(XT, y, s)
        qs = "0" if q == 0 else f"−{abs(q):.2f}"
        c.text_px(XT + 20, y - 2, f"{won(cost)}, QALY {qs}", cls="lbl small")
        c.text_px(XT + 20, y + 12, f"경로 확률 {pr:.3f}", cls="lbl mute small")
    ev = N["tree"][key]
    c.text_px(XD + 36, y0 + 17, f"기대비용 {won(ev['cost'])}", cls="lbl small")
    c.text_px(XD + 36, y0 + 31, f"QALY −{abs(ev['qaly']):.4f}", cls="lbl small")
    ae_c = add + tp["p_hosp"] * tp["c_hosp"] + (1 - tp["p_hosp"]) * tp["c_out"]
    c.text_px(XC2 - 14, ya + 16, f"기대 {won(ae_c)}", anchor="end", cls="lbl mute small")


strategy(96, 1, "none", "예방 요법 없음", tp["p_ae"], 0.0)
strategy(282, 2, "proph", "예방 요법", tp["p_ae"] * tp["rr_proph"], tp["c_proph"])
c.top.append(f'<rect x="{XD - 9}" y="177" width="18" height="18" rx="2" class="f4"/>')
# 범례
LX, LY = 14, 352
c.top.append(f'<rect x="{LX}" y="{LY - 7}" width="12" height="12" rx="2" class="f4"/>')
c.text_px(LX + 18, LY + 4, "결정 마디", cls="lbl small")
c.top.append(f'<circle cx="{LX + 92}" cy="{LY - 1}" r="6" class="pth s4"/>')
c.text_px(LX + 104, LY + 4, "확률 마디", cls="lbl small")
c.top.append(f'<polygon points="{LX + 170},{LY - 1} {LX + 182},{LY - 8} {LX + 182},{LY + 6}" class="f4"/>')
c.text_px(LX + 188, LY + 4, "종결 마디 (비용, QALY 변화)", cls="lbl small")
save("ch23_tree", figure(
    c.svg("중증 이상반응 예방 요법의 결정수형"),
    "그림 23-1. 신약 A 투여 초기의 중증 이상반응 예방 요법에 대한 결정수형(가상의 예시). 왼쪽 네모에서 전략을 고르고, 동그라미에서 확률에 따라 갈라지며, "
    "오른쪽 세모에 그 경로의 비용과 QALY 변화가 적혀 있습니다. 경로 확률은 지나온 가지의 확률을 곱한 값이고 한 전략 안에서 합이 1입니다. "
    f"전략 이름 아래의 값은 오른쪽에서 왼쪽으로 접어 올라가며 구한 기대값입니다(예방 요법 없음 {won(N['tree']['none']['cost'])}, 예방 요법 {won(N['tree']['proph']['cost'])})."))

# ====================================================================== 그림 23-2: 상태전이 그림
def state_panel(P, title, s):
    Wd, Hd = 420, 300
    q = canvas(Wd, Hd)
    pos = {"PF": (92, 112), "PD": (328, 112), "D": (210, 226)}
    names = {"PF": "무진행", "PD": "진행", "D": "사망"}
    bw, bh = 96, 40

    def box(k, cls):
        x, y = pos[k]
        q.els.append(f'<rect x="{x - bw / 2}" y="{y - bh / 2}" width="{bw}" height="{bh}" rx="10" class="pth {cls}"/>')
        q.text_px(x, y + 5, names[k], anchor="middle", cls="lbl strong")

    def arrow(a, b, label, lx, ly, anchor="middle"):
        (x1, y1), (x2, y2) = pos[a], pos[b]
        dx, dy = x2 - x1, y2 - y1

        def clip(dx, dy):
            tx = (bw / 2 + 3) / abs(dx) if dx else 1e9
            ty = (bh / 2 + 3) / abs(dy) if dy else 1e9
            return min(tx, ty)
        t1, t2 = clip(dx, dy), clip(-dx, -dy)
        sx, sy, ex, ey = x1 + dx * t1, y1 + dy * t1, x2 - dx * t2, y2 - dy * t2
        ln = math.hypot(ex - sx, ey - sy); ux, uy = (ex - sx) / ln, (ey - sy) / ln
        hx, hy = ex - ux * 9, ey - uy * 9
        q.els.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" class="ln s4" stroke-width="1.8"/>')
        q.els.append(f'<polygon points="{ex:.1f},{ey:.1f} {hx - uy * 5:.1f},{hy + ux * 5:.1f} {hx + uy * 5:.1f},{hy - ux * 5:.1f}" class="f4"/>')
        q.text_px(lx, ly, label, anchor=anchor, cls="lbl")

    def loop(k, label, below=False):
        x, y = pos[k]
        if below:
            y0 = y + bh / 2 + 2
            d = f"M{x - 16},{y0} C{x - 44},{y0 + 40} {x + 44},{y0 + 40} {x + 16},{y0 + 6}"
            head = f"{x + 13},{y0 + 1} {x + 22.5},{y0 + 9} {x + 12.5},{y0 + 12}"
            q.text_px(x + 34, y0 + 26, label, cls="lbl small")
        else:
            y0 = y - bh / 2 - 2
            d = f"M{x - 16},{y0} C{x - 44},{y0 - 40} {x + 44},{y0 - 40} {x + 16},{y0 - 6}"
            head = f"{x + 13},{y0 - 1} {x + 22.5},{y0 - 9} {x + 12.5},{y0 - 12}"
            q.text_px(x, y0 - 37, label, anchor="middle", cls="lbl small")
        q.els.append(f'<path d="{d}" class="ln s4" stroke-width="1.6" fill="none"/>')
        q.els.append(f'<polygon points="{head}" class="f4"/>')

    box("PF", f"s{s}"); box("PD", f"s{s}"); box("D", "s4")
    arrow("PF", "PD", f"{P[0][1]:.3f}", 210, 103)
    arrow("PF", "D", f"{P[0][2]:.3f}", 128, 182, anchor="end")
    arrow("PD", "D", f"{P[1][2]:.3f}", 292, 182, anchor="start")
    loop("PF", f"머무름 {P[0][0]:.3f}"); loop("PD", f"머무름 {P[1][1]:.3f}")
    loop("D", "머무름 1", below=True)
    q.text_px(10, 20, title, cls="ptitle")
    return q.svg(title + " 상태전이 그림")


PB, PA = N["markov"]["B"]["P"], N["markov"]["A"]["P"]
save("ch23_states", figure(
    [state_panel(PB, "(가) 표준요법 B", 2), state_panel(PA, "(나) 신약 A", 1)],
    "그림 23-2. 세 상태 마르코프 모형의 상태전이 그림과 한 달 전이확률(가상의 예시). 화살표는 한 주기(1개월) 동안 일어날 수 있는 이동이고, "
    f"한 상태에서 나가는 화살표와 머무름의 확률을 더하면 1입니다. 표준요법 B의 무진행 상태: {PB[0][0]:.3f} + {PB[0][1]:.3f} + {PB[0][2]:.3f} = 1. "
    "사망에서는 나가는 화살표가 없습니다. 신약 A는 무진행 상태를 떠나는 확률만 다릅니다.", cols=2))


# ====================================================================== 공통: 쌓은 면적 그림
def stacked(pf, pd_, title, aria, labels, xmax=120, top_line_label=None, mid_line_label=None):
    x = T[: xmax + 1]
    pf, pd_ = np.asarray(pf)[: xmax + 1], np.asarray(pd_)[: xmax + 1]
    alive = pf + pd_
    q = Plot((0, xmax), (0, 1.0), w=420, h=320, ml=56, mr=18, xlabel="치료 시작 후 개월", ylabel="코호트의 비율",
             yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=pc, xticks=list(range(0, xmax + 1, 24)))
    zero = np.zeros_like(x)
    q.fill_between(x, zero, pf, s=1)
    q.fill_between(x, pf, alive, s=2)
    q.fill_between(x, alive, np.ones_like(x), s=4)
    q.line(x, pf, s=1, w=2.2)
    q.line(x, alive, s=2, w=2.2)
    for (tx, ty, txt, cls) in labels:
        q.text(tx, ty, txt, cls=cls)
    panel_title(q, title)
    return q.svg(aria)


# ====================================================================== 그림 23-3: 마르코프 코호트 추적
mB, mA = N["markov"]["B"], N["markov"]["A"]
save("ch23_trace", figure(
    [stacked(mB["pf"], mB["pd"], "(가) 표준요법 B", "표준요법 B의 마르코프 코호트 추적",
             [(5, 0.10, "무진행", "lbl small"), (30, 0.22, "진행", "lbl small"), (66, 0.62, "사망", "lbl small")]),
     stacked(mA["pf"], mA["pd"], "(나) 신약 A", "신약 A의 마르코프 코호트 추적",
             [(5, 0.10, "무진행", "lbl small"), (40, 0.26, "진행", "lbl small"), (72, 0.62, "사망", "lbl small")])],
    "그림 23-3. 마르코프 코호트 추적(가상의 예시). 달마다 코호트가 세 상태에 어떻게 나뉘어 있는지를 쌓아 그린 것으로, 어느 시점에서나 세 띠의 합은 100%입니다. "
    f"24개월에 표준요법 B는 무진행 {mB['pf'][24] * 100:.1f}%, 진행 {mB['pd'][24] * 100:.1f}%, 사망 {(1 - mB['pf'][24] - mB['pd'][24]) * 100:.1f}%이고, "
    f"신약 A는 {mA['pf'][24] * 100:.1f}%, {mA['pd'][24] * 100:.1f}%, {(1 - mA['pf'][24] - mA['pd'][24]) * 100:.1f}%입니다. 띠의 넓이가 그 상태에서 보낸 평균 기간입니다.",
    cols=2))

# ====================================================================== 그림 23-4: 분할생존모형의 면적
R = L.run()
panels = []
for arm, title, pos in (("B", "(가) 표준요법 B", [(3, 0.07), (27, 0.21), (66, 0.56)]),
                        ("A", "(나) 신약 A", [(3, 0.07), (40, 0.24), (74, 0.56)])):
    tr = R[arm]["trace"]
    lab = [(pos[0][0], pos[0][1], f"무진행 {R[arm]['ly_pf']:.2f}년", "lbl small"),
           (pos[1][0], pos[1][1], f"진행 {R[arm]['ly_pd']:.2f}년", "lbl small"),
           (pos[2][0], pos[2][1], "사망", "lbl small")]
    x = T[:121]
    q = Plot((0, 120), (0, 1.0), w=420, h=320, ml=56, mr=18, xlabel="치료 시작 후 개월", ylabel="생존 비율",
             yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=pc, xticks=list(range(0, 121, 24)))
    pf, alive = tr["pf"][:121], (tr["pf"] + tr["pd"])[:121]
    q.fill_between(x, np.zeros_like(x), pf, s=1)
    q.fill_between(x, pf, alive, s=2)
    q.line(x, pf, s=1, w=2.4)
    q.line(x, alive, s=2, w=2.4)
    q.legend([("전체생존(OS) 곡선", 2, "line"), ("무진행생존(PFS) 곡선", 1, "line")], X=q.ml + 176, Y=q.mt + 12)
    for (tx, ty, txt, cls) in lab:
        q.text(tx, ty, txt, cls=cls)
    panel_title(q, title)
    panels.append(q.svg(f"{arm}군의 무진행생존 곡선과 전체생존 곡선, 그 사이의 면적"))
save("ch23_psm", figure(
    panels,
    "그림 23-4. 분할생존모형(가상의 예시). 아래 곡선이 무진행생존, 위 곡선이 전체생존입니다. 무진행생존 곡선 아래(파란 면적)가 무진행 상태, "
    "두 곡선 사이(주황 면적)가 진행 상태, 전체생존 곡선 위가 사망입니다. 면적이 그 상태에서 보낸 평균 기간이며, 적힌 값은 20년까지 더한 것입니다(할인 전). "
    f"합계는 표준요법 B {R['B']['ly']:.2f}년, 신약 A {R['A']['ly']:.2f}년입니다.",
    cols=2))

# ====================================================================== 그림 23-5: 마르코프 모형과 분할생존모형의 곡선
MK = L.run_markov()
panels = []
for arm, title in (("B", "(가) 표준요법 B"), ("A", "(나) 신약 A")):
    x = T[:121]
    tr, mk = R[arm]["trace"], MK[arm]["trace"]
    q = Plot((0, 120), (0, 1.0), w=420, h=320, ml=56, mr=18, xlabel="치료 시작 후 개월", ylabel="생존 비율",
             yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=pc, xticks=list(range(0, 121, 24)))
    q.line(x, (tr["pf"] + tr["pd"])[:121], s=2, w=2.4)
    q.line(x, (mk["pf"] + mk["pd"])[:121], s=2, w=2.0, dash=True)
    q.line(x, tr["pf"][:121], s=1, w=2.4)
    q.line(x, mk["pf"][:121], s=1, w=2.0, dash=True)
    q.legend([("분할생존모형 (와이블 곡선)", 4, "line"), ("마르코프 모형 (일정한 전이확률)", 4, "dash")], X=q.ml + 118, Y=q.mt + 14)
    q.text(44 if arm == "B" else 56, 0.36, "전체생존", cls="lbl small")
    q.text(3, 0.06, "무진행생존", cls="lbl small")
    panel_title(q, title)
    panels.append(q.svg(f"{arm}군에서 두 모형이 내는 무진행생존과 전체생존 곡선"))
save("ch23_compare", figure(
    panels,
    "그림 23-5. 같은 중앙값에 맞춘 두 모형의 곡선(가상의 예시). 실선은 분할생존모형이 쓰는 와이블 곡선, 점선은 전이확률이 일정한 마르코프 모형에서 나온 곡선입니다. "
    "표준요법 B의 전체생존은 두 모형이 중앙값(28개월)에서 만나지만 앞쪽에서는 마르코프 모형이, 꼬리에서는 분할생존모형이 더 높습니다. "
    "신약 A의 전체생존 꼬리에서 차이가 더 큽니다.", cols=2))

# ====================================================================== 그림 23-6: 외삽
trial = np.loadtxt(os.path.join(HERE, "_ch23_trial.csv"), delimiter=",", skiprows=1)
dur, ev = trial[:, 0], trial[:, 1].astype(int)
# Kaplan-Meier (직접 계산: 그림용 계단 좌표)
o = np.argsort(dur, kind="stable")
td, ed = dur[o], ev[o]
kt, ks, S = [0.0], [1.0], 1.0
cens_t, cens_s = [], []
n = len(td)
for i in range(n):
    if ed[i] == 1:
        S *= 1 - 1 / (n - i)
        kt.append(td[i]); ks.append(S)
    else:
        cens_t.append(td[i]); cens_s.append(S)
kt.append(td[-1]); ks.append(S)
assert abs(S - N["km"]["s_last"]) < 1e-9
F = N["fit"]
STYLE = {"exp": (1, True), "weib": (1, False), "lnorm": (2, False), "llog": (2, True), "ggam": (3, False)}

q1 = Plot((0, 30), (0.4, 1.0), w=420, h=330, ml=56, mr=18, xlabel="무작위배정 후 개월", ylabel="전체생존 비율",
          yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ytickfmt=pc, xticks=[0, 6, 12, 18, 24, 30])
for key, (s, dash) in STYLE.items():
    q1.line(T[:31], np.array(F[key]["S"])[:31], s=s, dash=dash, w=1.8)
q1.step(kt, ks, s=4, w=2.6)
q1.ticks_marks(cens_t[::6], cens_s[::6], s=4, size=4)
q1.legend([("Kaplan-Meier (관찰 자료)", 4, "line")], X=q1.ml + 150, Y=q1.mt + 12)
q1.text(1, 0.47, "가는 선: 다섯 모수 모형", cls="lbl small")
q1.text(1, 0.43, "(색과 선 모양은 오른쪽 그림과 같음)", cls="lbl small")
panel_title(q1, "(가) 관찰 기간 (30개월까지)")

q2 = Plot((0, 240), (0, 1.0), w=420, h=330, ml=56, mr=18, xlabel="무작위배정 후 개월", ylabel="전체생존 비율",
          yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=pc, xticks=[0, 60, 120, 180, 240])
q2.fill_between([0, 30], [0, 0], [1, 1], cls="a4")
for key, (s, dash) in STYLE.items():
    q2.line(T, F[key]["S"], s=s, dash=dash, w=2.0)
q2.step(kt, ks, s=4, w=2.6)
q2.text(34, 0.95, "← 관찰 기간", cls="lbl mute small")
q2.legend([("로그-정규", 2, "line"), ("로그-로지스틱", 2, "dash"), ("지수", 1, "dash"), ("와이블", 1, "line"), ("일반화 감마", 3, "line")],
          X=q2.ml + 196, Y=q2.mt + 46, gap=17)
panel_title(q2, "(나) 20년까지 외삽")
save("ch23_extrap", figure(
    [q1.svg("관찰 기간의 Kaplan-Meier 곡선과 다섯 모수 모형"), q2.svg("다섯 모수 모형을 20년까지 외삽한 전체생존 곡선")],
    "그림 23-6. 표준요법 B군 300명의 가상 시험 자료에 다섯 모수 모형을 적합한 결과. (가) 자료가 있는 30개월까지는 다섯 곡선이 Kaplan-Meier 곡선(회색 계단, 세로 눈금은 중도절단의 일부)을 "
    f"비슷하게 따라갑니다. (나) 같은 곡선을 20년까지 늘리면 10년 생존율이 {F['ggam']['s120'] * 100:.1f}%(일반화 감마)에서 {F['lnorm']['s120'] * 100:.1f}%(로그-정규)까지 갈립니다. "
    "회색으로 칠한 왼쪽 띠만 자료가 있는 구간입니다.", cols=2))
print("figures written")
