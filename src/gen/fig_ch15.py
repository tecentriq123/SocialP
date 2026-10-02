"""Figures for chapter 15 (성향점수).

Run:  source /home/claude/pylibs/env.sh && python3 gen/fig_ch15.py      (about 30 s; calls nums_ch15.compute())
Writes figs/ch15_dag.html, ch15_psdist.html, ch15_target.html, ch15_love.html, ch15_cuminc.html, ch15_forest.html
(captions 그림 15-1 … 15-6 in this order of appearance).
"""
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from html import escape
from svgplot import Plot, figure, panel_title, forest
from nums_ch15 import compute

OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)
R = compute()
s = R["sim"]
toy = R["toy"]
poor = R["poor"]


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


# ------------------------------------------------------------------ DAG helpers (same drawing rules as gen/fig_ch08.py)
def _tw(t, fs):
    w = 0.0
    for c in t:
        if "가" <= c <= "힣":
            w += fs * 1.0
        elif c == " ":
            w += fs * 0.3
        elif c in "%WMm":
            w += fs * 0.85
        else:
            w += fs * 0.6
    return w


def node(p, X, Y, label, cls="s4", sub=None, w=None, dash=False, fs=13):
    w = w or (max(_tw(label, fs), _tw(sub, 11.5) if sub else 0) + 24)
    h = 34 if not sub else 46
    da = ' stroke-dasharray="5 4"' if dash else ""
    p.els.append(f'<rect x="{X - w / 2:.1f}" y="{Y - h / 2:.1f}" width="{w:.1f}" height="{h}" rx="8" class="pth {cls}"{da}/>')
    if sub:
        p.text_px(X, Y - 3, label, anchor="middle", cls="lbl strong")
        p.text_px(X, Y + 13, sub, anchor="middle", cls="lbl mute small")
    else:
        p.text_px(X, Y + 4.5, label, anchor="middle", cls="lbl strong")
    return (X, Y, w, h)


def arrow(p, a, b, cls="s4", dash=False, w=1.6):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b

    def clip(xc, yc, wc, hc, dx, dy):
        tx = (wc / 2 + 3) / abs(dx) if dx else 1e9
        ty = (hc / 2 + 3) / abs(dy) if dy else 1e9
        t = min(tx, ty)
        return xc + dx * t, yc + dy * t
    dx, dy = x2 - x1, y2 - y1
    sx, sy = clip(x1, y1, w1, h1, dx, dy)
    ex, ey = clip(x2, y2, w2, h2, -dx, -dy)
    ln = math.hypot(ex - sx, ey - sy)
    ux, uy = (ex - sx) / ln, (ey - sy) / ln
    hx, hy = ex - ux * 9, ey - uy * 9
    da = ' stroke-dasharray="6 4"' if dash else ""
    p.els.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" class="ln {cls}" stroke-width="{w}"{da}/>')
    px, py = -uy, ux
    pts = f"{ex:.1f},{ey:.1f} {hx + px * 5:.1f},{hy + py * 5:.1f} {hx - px * 5:.1f},{hy - py * 5:.1f}"
    p.els.append(f'<polygon points="{pts}" class="f{cls[1:]}"/>')


def canvas(w, h):
    return Plot((0, 1), (0, 1), w=w, h=h, ml=0, mr=0, mt=0, mb=0, show_xaxis=False, show_yaxis=False, ygrid=False)


# ====================================================================== 15-1  which variables to adjust for
W, H = 420, 290
c1 = canvas(W, H)
C = node(c1, 210, 66, "교란변수", cls="s2", sub="예: 심부전, 과거 입원")
I = node(c1, 78, 138, "치료에만 영향", cls="s4", sub="예: 처방 의사의 선호", dash=True)
Rk = node(c1, 345, 138, "결과에만 영향", cls="s3", sub="예: 만성폐질환")
E = node(c1, 120, 228, "약물 A 처방", cls="s1")
Y = node(c1, 320, 228, "1년 안 입원", cls="s1")
arrow(c1, C, E, cls="s2"); arrow(c1, C, Y, cls="s2")
arrow(c1, I, E, cls="s4", dash=True)
arrow(c1, Rk, Y, cls="s3")
arrow(c1, E, Y, cls="s1", w=2.2)
c1.text_px(10, 20, "가. 치료 시작 전에 정해진 변수", cls="ptitle")
c1.text_px(210, 276, "교란변수는 보정, 치료에만 영향을 주는 변수는 넣지 않음", anchor="middle", cls="lbl small")

c2 = canvas(W, H)
E2 = node(c2, 78, 138, "약물 A 처방", cls="s1")
Y2 = node(c2, 342, 138, "1년 안 입원", cls="s1")
M2 = node(c2, 210, 62, "매개변수", cls="s3", sub="예: 시작 후 증상 조절")
C2 = node(c2, 210, 222, "충돌변수", cls="s2", sub="예: 시작 후 외래 방문 횟수", dash=True)
arrow(c2, E2, M2, cls="s3"); arrow(c2, M2, Y2, cls="s3")
arrow(c2, E2, Y2, cls="s1", w=2.2)
arrow(c2, E2, C2, cls="s4"); arrow(c2, Y2, C2, cls="s4")
c2.text_px(10, 20, "나. 치료 시작 뒤에 생긴 변수", cls="ptitle")
c2.text_px(210, 276, "둘 다 보정하지 않음", anchor="middle", cls="lbl small")
save("ch15_dag", figure(
    [c1.svg("치료 시작 전에 정해진 변수의 인과 도표"), c2.svg("치료 시작 뒤에 생긴 변수의 인과 도표")],
    "그림 15-1. 변수의 종류와 보정 여부. 화살표는 '원인 → 결과'입니다. (가) 교란변수는 약 선택과 입원 모두의 원인이므로 보정합니다. "
    "결과에만 영향을 주는 변수는 넣어도 좋고, 치료에만 영향을 주는 변수(점선)는 넣지 않습니다. "
    "(나) 매개변수는 약의 효과가 지나가는 길이고, 충돌변수는 약과 결과가 함께 영향을 주는 변수입니다. 둘 다 치료 시작 뒤에 생기며 보정하면 편향이 생깁니다.",
    cols=2))


# ====================================================================== 15-2  propensity score distributions
def mirror(ps, A, title, nA, nB, cstat, ymax=20):
    edges = np.linspace(0, 1, 26)
    hA, _ = np.histogram(ps[A == 1], edges)
    hB, _ = np.histogram(ps[A == 0], edges)
    fA = 100 * hA / hA.sum(); fB = 100 * hB / hB.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    p = Plot((0, 1), (-ymax, ymax), w=420, h=330, ml=54, mr=14, mt=30, mb=52,
             xlabel="성향점수 (약물 A를 받을 추정 확률)", ylabel="각 군 안에서의 비율(%)",
             xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xtickfmt=lambda v: f"{v:.1f}",
             yticks=[-ymax, -ymax // 2, 0, ymax // 2, ymax], ytickfmt=lambda v: f"{abs(v):.0f}")
    p.bars(mids, np.minimum(fA, ymax), 0.036, s=1, rounded=False)
    p.bars(mids, -np.minimum(fB, ymax), 0.036, s=2, rounded=False)
    p.hline(0, dash=False, cls="axis")
    p.text(0.99, ymax * 0.84, f"약물 A (n = {nA:,})", anchor="end", cls="lbl strong")
    p.text(0.99, -ymax * 0.90, f"약물 B (n = {nB:,})", anchor="end", cls="lbl strong")
    p.text(0.99, ymax * 0.84, f"C 통계량 {cstat:.2f}", anchor="end", dy=17, cls="lbl mute small")
    panel_title(p, title)
    return p, fA, fB


pa, fA1, fB1 = mirror(s["ps"], s["A"], "가. 이 장의 코호트", s["nA"], s["nB"], s["c_ps"])
pb, fA2, fB2 = mirror(poor["ps"], poor["A"], "나. 약 선택이 훨씬 엄격한 가상의 경우", poor["nA"], poor["nB"], poor["c"], ymax=20)
assert fA1.max() < 20 and fB1.max() < 20, (fA1.max(), fB1.max())
print("psdist max bar %:", fA1.max(), fB1.max(), fA2.max(), fB2.max())
save("ch15_psdist", figure(
    [pa.svg("이 장의 코호트에서 약물 A군과 B군의 성향점수 분포"), pb.svg("약 선택이 훨씬 엄격한 가상의 코호트에서 두 군의 성향점수 분포")],
    f"그림 15-2. 성향점수의 분포. 위(파랑)는 약물 A 사용자, 아래(주황)는 약물 B 사용자입니다. "
    f"(가) 이 장의 코호트는 A군의 점수가 조금 높지만(평균 {s['overlap']['meanA']:.2f} 대 {s['overlap']['meanB']:.2f}) 거의 모든 구간에 두 군이 함께 있습니다. "
    f"(나) 같은 변수가 약 선택에 세 배 강하게 작용하도록 만든 가상의 경우입니다. A군의 {100 * poor['gt95_A']:.0f}%가 0.95를 넘는 점수에 몰려 있고 그 구간에는 비교할 B 사용자가 거의 없습니다. "
    f"C 통계량은 {poor['c']:.2f}으로 높지만 비교 가능한 환자는 줄었습니다.", cols=2))


# ====================================================================== 15-3  target populations in the toy example
t = toy
groups = [
    ("보정 전", [(t["tot"]["nA"], t["hfA"]), (t["tot"]["nB"], t["hfB"])]),
    ("IPTW (ATE)", [(t["wA"], t["hf_all"]), (t["wB"], t["hf_all"])]),
    ("ATT 가중", [(t["tot"]["nA"], t["hfA"]), (t["wB_att"], t["hfA"])]),
    ("1:1 매칭", [(t["m"]["n"], t["m"]["hf"] / t["m"]["n"]), (t["m"]["n"], t["m"]["hf"] / t["m"]["n"])]),
]
p = Plot((0, 12), (0, 110), w=600, h=340, ml=58, mr=16, mt=14, mb=74, ylabel="군 안에서의 구성(%)",
         yticks=[0, 20, 40, 60, 80, 100], xticks=[], show_xaxis=True)
xs = []
for gi, (gname, bars) in enumerate(groups):
    for bi, (n, hf) in enumerate(bars):
        x = 1.5 + gi * 3 + (bi - 0.5) * 1.15
        p.bars([x], [100], 1.0, cls="a4", rounded=False)
        p.bars([x], [100 * hf], 1.0, s=2, rounded=False)
        p.text(x, 100, f"{n:.0f}명", anchor="middle", dy=-7, cls="lbl small")
        p.text(x, 100 * hf, f"{100 * hf:.0f}%", anchor="middle", dy=-6, cls="lbl strong")
        p.text_px(p.sx(x), p.h - p.mb + 18, "AB"[bi], anchor="middle", cls="tick")
    p.text_px(p.sx(1.5 + gi * 3), p.h - p.mb + 38, gname, anchor="middle", cls="lbl strong")
p.legend([("심부전 있음", 2, "box")], X=p.sx(0.1), Y=p.h - 14)
p.legend([("심부전 없음", 4, "soft")], X=p.sx(3.0), Y=p.h - 14)
svg3 = p.svg("보정 전, IPTW, ATT 가중, 1:1 매칭에서 두 군의 심부전 환자 비율")
save("ch15_target", figure(
    svg3,
    "그림 15-3. 심부전 하나만 교란변수인 작은 예에서 각 방법이 만드는 비교 집단. 막대 위 숫자는 (가중) 인원, 주황 부분은 심부전 환자의 비율입니다. "
    "보정 전에는 A군 60%, B군 11%로 다릅니다. 세 방법 모두 두 군의 구성을 같게 만들지만 맞춘 구성은 서로 다릅니다. "
    "IPTW는 코호트 전체(29%), ATT 가중은 A 사용자(60%), 1:1 매칭은 짝을 찾은 환자(33%)의 구성입니다."))


# ====================================================================== 15-4  Love plot (measured covariates only)
tab = s["tab"]
meas = [r for r in tab if r["key"] != "frail"]
order = sorted(meas, key=lambda r: abs(r["smd_pre"]))
nrow = len(order)
p = Plot((0, 0.42), (0.3, nrow + 1.5), w=600, h=40 + 32 * nrow + 86, ml=176, mr=24, mt=24, mb=52,
         xlabel="표준화 평균차의 절댓값 (|SMD|)", xticks=[0, 0.1, 0.2, 0.3, 0.4], xtickfmt=lambda v: f"{v:.1f}",
         yticks=[], ygrid=False, xgrid=True, show_yaxis=False)
p.vline(0.1, dash=True, y0=0.3, y1=nrow + 0.6)
p.text(0.1, nrow + 0.62, "0.1 기준", dx=5, dy=4, cls="lbl mute small")
for i, r in enumerate(order):
    y = i + 1
    b, a, w_ = abs(r["smd_pre"]), abs(r["smd_m"]), abs(r["smd_w"])
    p.seg(min(a, w_), y, b, y, cls="ref")
    p.points([b], [y], s=2, r=5, hollow=True)
    p.points([a], [y + 0.16], s=1, r=4.5)
    p.points([w_], [y - 0.16], s=3, r=4.5)
    p.text_px(p.ml - 10, p.sy(y) + 4.5, r["ko"], anchor="end", cls="lbl")
ly = p.sy(nrow + 1.15)
p.legend([("보정 전", 2, "dot")], X=p.sx(0.135), Y=ly)
p.legend([("매칭 후", 1, "dot")], X=p.sx(0.225), Y=ly)
p.legend([("IPTW 후", 3, "dot")], X=p.sx(0.315), Y=ly)
p.top = [x.replace('class="f2"', 'class="pth s2"') if 'r="4.5" class="f2"' in x else x for x in p.top]
mx_pre = max(abs(r["smd_pre"]) for r in meas); mx_m = max(abs(r["smd_m"]) for r in meas); mx_w = max(abs(r["smd_w"]) for r in meas)
print("love: max |SMD| pre %.3f matched %.3f iptw %.3f" % (mx_pre, mx_m, mx_w))
save("ch15_love", figure(
    p.svg("성향점수 적용 전후 공변량별 표준화 평균차 (Love plot)"),
    f"그림 15-4. 공변량 균형을 보여 주는 Love plot. 빈 원은 보정 전, 파란 점은 1:1 매칭 후, 초록 점은 IPTW 후의 |SMD|입니다. "
    f"보정 전에는 7개 변수 중 6개가 0.1 기준선 오른쪽에 있었고(최대 {mx_pre:.2f}), 적용 후에는 모두 왼쪽으로 옮겨 왔습니다"
    f"(매칭 후 최대 {mx_m:.2f}, IPTW 후 최대 {mx_w:.2f}). 이 그림은 성향점수 모형에 넣은 변수의 균형만 보여 줍니다."))


# ====================================================================== 15-5  crude and weighted cumulative incidence
km = s["km"]


def cum_panel(a, b, title, ra, rb):
    p = Plot((0, 365), (0, 15), w=420, h=320, ml=52, mr=18, mt=30, mb=52, xlabel="첫 처방일부터의 일수",
             ylabel="입원의 누적 발생률(%)", xticks=[0, 90, 180, 270, 365], yticks=[0, 5, 10, 15])
    p.step(np.r_[0, a[0]], np.r_[0, 100 * a[1]], s=1, w=2)
    p.step(np.r_[0, b[0]], np.r_[0, 100 * b[1]], s=2, w=2)
    p.legend([(f"약물 A (1년 {100 * ra:.1f}%)", 1, "line"), (f"약물 B (1년 {100 * rb:.1f}%)", 2, "line")], X=p.ml + 10, Y=p.mt + 12)
    panel_title(p, title)
    return p


e = s["km_end"]
pa = cum_panel(km["cA"], km["cB"], "가. 보정 전", e["cA"], e["cB"])
pb = cum_panel(km["wA"], km["wB"], "나. IPTW 가중 후", e["wA"], e["wB"])
save("ch15_cuminc", figure(
    [pa.svg("보정 전 두 군의 입원 누적 발생률 곡선"), pb.svg("IPTW 가중 후 두 군의 입원 누적 발생률 곡선")],
    f"그림 15-5. 1년 동안의 입원 누적 발생률(1 − Kaplan-Meier 생존확률). (가) 보정 전에는 A군의 곡선이 처음부터 위에 있습니다"
    f"(1년 {100 * e['cA']:.1f}% 대 {100 * e['cB']:.1f}%). (나) 성향점수 가중치를 준 곡선은 측정한 기저특성이 같아진 두 가상 집단의 곡선으로, "
    f"거의 겹치며 A가 조금 아래입니다({100 * e['wA']:.1f}% 대 {100 * e['wB']:.1f}%). 논문에서는 'weighted' 또는 'adjusted cumulative incidence curves'라고 부릅니다.",
    cols=2))


# ====================================================================== 15-6  estimates by method vs the truth
def row(label, t3, s_=1, indent=True):
    return dict(label=label, est=t3[0], lo=t3[1][0], hi=t3[1][1], s=s_, indent=indent)


rows = [
    dict(label="측정한 7개 변수만 사용", header=True),
    row("보정 전", s["crude"], 4),
    row("다변수 회귀", s["reg"]),
    row("성향점수 1:1 매칭", s["matched"]),
    row("IPTW (ATE)", s["iptw"]),
    row("ATT 가중", s["att"]),
    row("겹침 가중", s["owr"]),
    row("성향점수 5분위 층화", (s["strat"]["rr"], s["strat"]["ci"])),
    dict(label="허약까지 넣었다면 (실제로는 불가능)", header=True),
    row("다변수 회귀", s["reg_frail"], 3),
    row("성향점수 1:1 매칭", s["matched_o"], 3),
    row("IPTW (ATE)", s["oracle"], 3),
]
XL = (0.62, 1.62)
FW, LW, EW, RH = 640, 236, 132, 29
svg = forest(rows, XL, ref=1.0, log=True, w=FW, row_h=RH, label_w=LW, est_w=EW, xlabel="상대위험도 (95% CI). 1보다 작으면 약물 A군의 입원이 적음",
             xticks=[0.7, 0.8, 1, 1.25, 1.5])
# add the line for the true value (0.80) used to generate the data
xt = LW + (math.log(0.80) - math.log(XL[0])) / (math.log(XL[1]) - math.log(XL[0])) * (FW - EW - LW)
yb = 12 + RH * len(rows) + 6
extra = (f'<line x1="{xt:.1f}" y1="8" x2="{xt:.1f}" y2="{yb}" class="ln s3" stroke-width="1.6" stroke-dasharray="5 4"/>'
         f'<text x="{xt + 6:.1f}" y="{12 + RH * 8 + RH / 2 + 4.5:.1f}" class="lbl small">참값 0.80</text>')
svg = svg.replace("</svg>", extra + "</svg>").replace('aria-label="forest plot"', 'aria-label="보정 방법별 상대위험도 추정값과 참값의 비교"')
save("ch15_forest", figure(
    svg,
    "그림 15-6. 같은 코호트를 여러 방법으로 분석한 상대위험도. 초록 점선은 이 가상 자료를 만들 때 정한 참값 0.80입니다. "
    "측정한 7개 변수만 쓴 방법은 다변수 회귀든 성향점수든 0.91–1.01에 모여 있고 신뢰구간이 모두 1을 포함합니다. "
    "청구자료에 없는 허약을 넣을 수 있었다면(아래 세 줄, 초록) 어느 방법으로도 참값 근처가 나옵니다. 남은 편향은 방법이 아니라 측정하지 못한 변수에서 옵니다."))

print("written:", [f for f in sorted(os.listdir(OUT)) if f.startswith("ch15_")])
