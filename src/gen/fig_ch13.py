import sys, os, io, contextlib, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt

with contextlib.redirect_stdout(io.StringIO()):
    import nums_ch13 as N

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def m(v, nd=1):
    """number with a real minus sign"""
    s = f"{v:.{nd}f}"
    return s.replace("-", "−")


def ci_row(p, y, est, lo, hi, s=1, sz=5, w=2.4):
    p.seg(lo, y, hi, y, cls=f"ln s{s}", w=w)
    p.seg(lo, y - 0.12, lo, y + 0.12, cls=f"ln s{s}", w=w * 0.75)
    p.seg(hi, y - 0.12, hi, y + 0.12, cls=f"ln s{s}", w=w * 0.75)
    X, Y = p.sx(est), p.sy(y)
    p.els.append(f'<rect x="{X - sz:.1f}" y="{Y - sz:.1f}" width="{2 * sz}" height="{2 * sz}" class="f{s}"/>')


# ------------------------------------------------------------------ 13-1 scenarios
W, H = 660, 360
p = Plot((-26, 16), (0.3, 6.9), w=W, h=H, ml=34, mr=196, mt=40, mb=52, show_yaxis=False, ygrid=False,
         xlabel="완치율 차이 (신약 − 표준치료, %p)", xticks=[-20, -10, 0, 10],
         xticklabels=[(-20, "−20"), (-10, "−10 (−Δ)"), (0, "0"), (10, "10")])
xs = np.array([-26, -10])
p.fill_between(xs, [0.3, 0.3], [6.9, 6.9], cls="a2")
p.vline(-10, cls="ref strongref", dash=False, w=1.4)
p.vline(0, dash=True)
p.vline(10, dash=True, cls="ref mute", w=1.0)
p.text(10, 6.9, "+Δ", anchor="middle", dy=-10, cls="lbl small mute")
p.text(-18, 6.9, "열등 영역", anchor="middle", dy=-10, cls="lbl strong")
p.text(-10, 6.9, "비열등성 한계", anchor="middle", dy=-24, cls="lbl small")
p.text(-10, 6.9, "−Δ = −10", anchor="middle", dy=-10, cls="lbl small")
p.text(0, 6.9, "차이 없음", anchor="middle", dy=-10, cls="lbl small")
sub = {
    "A": "하한 > 0",
    "B": "하한 > −Δ, 0을 포함",
    "C": "하한 > −Δ, 상한 < 0",
    "D": "하한 < −Δ, 0을 포함",
    "E": "하한 < −Δ, 상한 < 0",
    "F": "상한 < −Δ",
}
head = {"A": "우월성 입증", "B": "비열등성 입증", "C": "비열등성 입증 *", "D": "결론 불가",
        "E": "결론 불가 *", "F": "열등"}
for i, (k, est, lo, hi, _) in enumerate(N.SCEN):
    y = 6.2 - i * 1.07
    ci_row(p, y, est, lo, hi, s=1 if k in "ABC" else (4 if k in "DE" else 2))
    p.text_px(14, p.sy(y) + 5, k, cls="lbl strong")
    p.text_px(W - 196 + 14, p.sy(y) - 1, head[k], cls="lbl strong")
    p.text_px(W - 196 + 14, p.sy(y) + 15, sub[k], cls="lbl small mute")
save("ch13_scen", figure(p.svg("비열등성 시험에서 95% 신뢰구간의 위치에 따른 결론"),
     "그림 13-1. 비열등성 시험의 가능한 결과 여섯 가지(가상의 완치율 차이와 양측 95% 신뢰구간, 비열등성 한계 −10%p). 판단은 점추정값이 아니라 <b>신뢰구간의 하한</b>이 −Δ보다 오른쪽에 있는지로 합니다. * C는 비열등성은 입증됐지만 신약이 통계적으로 유의하게 낮고, E는 통계적으로 낮으면서 비열등성도 입증하지 못한 경우입니다. 오른쪽의 +Δ는 동등성을 판단할 때만 쓰는 위쪽 한계입니다."))

# ------------------------------------------------------------------ 13-2 BE scenarios (log axis)
W, H = 680, 330
lg = math.log
p = Plot((lg(0.66), lg(1.40)), (0.4, 4.9), w=W, h=H, ml=168, mr=176, mt=40, mb=52, show_yaxis=False, ygrid=False,
         xlabel="시험약/대조약 기하평균비 (로그 눈금)",
         xticks=[lg(v) for v in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25)],
         xticklabels=[(lg(v), lab) for v, lab in ((0.7, "70%"), (0.8, "80%"), (0.9, "90%"), (1.0, "100%"), (1.1, "110%"), (1.25, "125%"))])
p.fill_between(np.array([lg(0.8), lg(1.25)]), [0.4, 0.4], [4.9, 4.9], cls="a3")
p.vline(lg(0.8), cls="ref strongref", dash=False, w=1.4)
p.vline(lg(1.25), cls="ref strongref", dash=False, w=1.4)
p.vline(0, dash=True)
p.text(0, 4.9, "동등성 인정 범위 80.00–125.00%", anchor="middle", dy=-12, cls="lbl strong")
for i, (k, c, lab) in enumerate(N.BECASES):
    y = 4.25 - i * 1.1
    ok = c["lo"] >= 0.8 and c["hi"] <= 1.25
    ci_row(p, y, lg(c["gmr"]), lg(c["lo"]), lg(c["hi"]), s=1 if ok else 2)
    p.text_px(10, p.sy(y) - 1, f"{k} {lab}", cls="lbl strong")
    p.text_px(10, p.sy(y) + 15, f"개체내 CV {c['cv'] * 100:.0f}%", cls="lbl small mute")
    pv = c["p"]
    ptxt = f"p = {pv:.3f}" if pv < 0.01 else f"p = {pv:.2f}"
    p.text_px(W - 176 + 12, p.sy(y) - 1, ("동등성 인정" if ok else "동등성 불인정"), cls="lbl strong")
    p.text_px(W - 176 + 12, p.sy(y) + 15, f"{c['gmr'] * 100:.1f} ({c['lo'] * 100:.1f}–{c['hi'] * 100:.1f})", cls="lbl small")
    p.text_px(W - 176 + 12, p.sy(y) + 30, f"차이 검정 {ptxt}", cls="lbl small mute")
save("ch13_be", figure(p.svg("생물학적 동등성 판정: 기하평균비의 90% 신뢰구간과 80–125% 범위"),
     "그림 13-3. 생물학적 동등성 판정의 네 가지 예. 막대는 기하평균비의 90% 신뢰구간이고, 초록 띠(80.00–125.00%) 안에 구간 전체가 들어가야 동등성을 인정합니다. ②는 \"차이가 유의하지 않은데도\"(p = 0.72) 동등성을 인정받지 못하고, ③은 \"차이가 유의한데도\"(p = 0.001) 동등성을 인정받습니다. 가로축은 로그 눈금이라 80%와 125%가 100%에서 같은 거리에 있습니다."))

# ------------------------------------------------------------------ 13-3 margin derivation
W, H = 640, 300
p = Plot((-0.02, 1.2), (0.3, 3.9), w=W, h=H, ml=24, mr=24, mt=24, mb=52, show_yaxis=False, ygrid=False,
         xlabel="위약 대비 HbA1c 감소 효과 (%p)", xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
p.vline(0, cls="ref strongref", dash=False)
hist = N.HIST
y1, y2, y3 = 3.25, 2.05, 0.85
bh = 0.24
# row 1: historical estimate (sign flipped to 'reduction')
p.text(0.0, y1 + 0.42, "① 과거 위약대조 시험의 메타분석: 표준약의 효과", dx=6, cls="lbl strong")
ci_row(p, y1, -hist["est"], -hist["hi"], -hist["lo"], s=1)
p.text(-hist["hi"], y1, f"{-hist['est']:.2f} (95% CI {-hist['hi']:.2f}–{-hist['lo']:.2f})", anchor="end", dx=-10, dy=5, cls="lbl")


def bar(p, x0, x1, y, cls, s):
    X0, X1 = p.sx(x0), p.sx(x1)
    Y0, Y1 = p.sy(y + bh), p.sy(y - bh)
    p.els.append(f'<rect x="{X0:.1f}" y="{Y0:.1f}" width="{X1 - X0:.1f}" height="{Y1 - Y0:.1f}" class="{cls} s{s}" stroke-width="1.2"/>')


p.text(0.0, y2 + 0.42, "② M1: 비열등성 시험에서도 표준약에 있다고 믿을 수 있는 효과 = 95% CI의 보수적인 끝", dx=6, cls="lbl strong")
bar(p, 0, N.M1, y2, "a1", 1)
p.text(N.M1 / 2, y2, f"M1 = {N.M1:.2f}", anchor="middle", dy=5, cls="lbl strong")
p.text(0.0, y3 + 0.42, "③ M2: M1의 50%를 보존하도록 정한 비열등성 한계", dx=6, cls="lbl strong")
bar(p, 0, N.M1 - N.M2, y3, "a3", 3)
bar(p, N.M1 - N.M2, N.M1, y3, "a2", 2)
p.text((N.M1 - N.M2) / 2, y3, "신약이 지켜야 할 효과 0.40", anchor="middle", dy=5, cls="lbl small")
p.text(N.M1 - N.M2 / 2, y3, f"잃어도 되는 최대 {N.M2:.2f} = Δ", anchor="middle", dy=5, cls="lbl small")
p.text(0, 0.3, "위약 수준", anchor="start", dx=4, dy=-4, cls="lbl small mute")
save("ch13_margin", figure(p.svg("M1과 M2로 비열등성 한계를 정하는 과정"),
     "그림 13-2. 비열등성 한계를 정하는 두 단계(가상의 예). 표준약이 위약보다 HbA1c를 0.95%p 더 낮춘다는 과거 자료에서 95% 신뢰구간의 보수적인 끝 0.80%p를 M1으로 잡고, 그 절반 0.40%p를 신약이 잃어도 되는 최대치(M2 = Δ)로 정했습니다. 신약 − 표준약 차이가 0.40%p보다 작다는 것을 보이면 신약은 위약 대비 적어도 0.40%p의 효과를 가진다고 간접적으로 추론할 수 있습니다."))

# ------------------------------------------------------------------ 13-4 PDC: shifted null + CI
P = N.PDC["ITT"]
se, crit, dobs = P["se"], P["crit"], P["d"]
x = np.linspace(-11, 6, 341)
p1 = Plot((-11, 6), (0, 0.33), w=640, h=270, ml=58, mr=24, mt=30, mb=50, xlabel="관측될 수 있는 평균 차이 (원격 − 대면, %p)",
          ylabel="확률밀도", yticks=[0, 0.1, 0.2, 0.3], xticks=[-10, -8, -6, -5, -4, -2, 0, 2, 4, 6],
          xticklabels=[(-10, "−10"), (-8, "−8"), (-6, "−6"), (-4, "−4"), (-2, "−2"), (0, "0"), (2, "2"), (4, "4"), (6, "6")])
dens0 = st.norm.pdf(x, 0, se)
densN = st.norm.pdf(x, -N.DP, se)
xr = np.linspace(crit, 6, 120)
p1.fill_between(xr, np.zeros_like(xr), st.norm.pdf(xr, -N.DP, se), s=2)
p1.line(x, dens0, s=4, dash=True)
p1.line(x, densN, s=2)
p1.vline(-N.DP, cls="ref strongref", dash=False, y1=0.33)
p1.vline(crit, dash=True, y1=0.300)
p1.vline(dobs, cls="ln s1", dash=False, w=2.2, y1=0.312)
p1.text(-N.DP, 0.305, "귀무가설 경계 (−Δ = −5)", anchor="end", dx=-6, cls="lbl")
p1.text(-8.6, 0.10, "실제 차이가 한계(−5)일 때", anchor="middle", cls="lbl small")
p1.text(-8.6, 0.10, "관측 차이의 분포", anchor="middle", dy=15, cls="lbl small")
p1.text(crit, 0.300, f"임계값 {m(crit, 2)}", anchor="end", dx=-4, dy=4, cls="lbl small")
p1.text(dobs, 0.312, f"관측된 차이 {m(dobs)}", anchor="start", dx=6, dy=5, cls="lbl strong")
p1.text(3.2, 0.21, "차이 = 0일 때의 분포", anchor="start", cls="lbl small mute")
p1.text(3.2, 0.21, "(우월성 검정의 H₀)", anchor="start", dy=15, cls="lbl small mute")
p1.text(-2.55, 0.045, "기각역 2.5%", anchor="end", dx=-4, cls="lbl small")
panel_title(p1, "가. 비열등성 검정: 귀무가설을 −5로 옮겨 놓고 검정")

p2 = Plot((-11, 6), (0.3, 2.7), w=640, h=170, ml=58, mr=24, mt=30, mb=46, show_yaxis=False, ygrid=False,
          xlabel="평균 PDC 차이 (원격 − 대면, %p)와 95% 신뢰구간", xticks=[-10, -5, 0, 5],
          xticklabels=[(-10, "−10"), (-5, "−5 (−Δ)"), (0, "0"), (5, "5")])
p2.fill_between(np.array([-11, -5]), [0.3, 0.3], [2.7, 2.7], cls="a2")
p2.vline(-5, cls="ref strongref", dash=False, w=1.4)
p2.vline(0, dash=True)
for i, lab in enumerate(("ITT", "PP")):
    q = N.PDC[lab]
    y = 2.0 - i * 1.0
    ci_row(p2, y, q["d"], q["lo"], q["hi"], s=1)
    p2.text_px(10, p2.sy(y) + 5, lab, cls="lbl strong")
    p2.text(q["hi"], y, f"{m(q['d'])} ({m(q['lo'], 2)} ~ {m(q['hi'], 2)})", dx=10, dy=5, cls="lbl small")
p2.text(-8, 2.7, "열등 영역", anchor="middle", dy=14, cls="lbl small")
panel_title(p2, "나. 같은 결론을 신뢰구간으로: 하한이 −5보다 오른쪽")
save("ch13_pdc", figure([p1.svg("비열등성 검정의 귀무가설 분포와 기각역"), p2.svg("평균 차이의 95% 신뢰구간과 비열등성 한계")],
     f"그림 13-4. 원격 복약상담 시험(ITT, 군당 304명, 차이의 표준오차 {se:.2f}). 가: 실제 차이가 한계 −5%p일 때 관측 차이가 어떻게 흩어지는지 보여 주는 분포(주황)에서, 관측값 −1.3이 오른쪽 2.5% 기각역(임계값 {m(crit, 2)}) 안에 있으므로 비열등성을 입증합니다(단측 p = {P['pni']:.3f}). 같은 −1.3은 차이 = 0을 귀무가설로 하는 회색 분포에서는 평범한 값입니다(양측 p = {P['psup']:.2f}). 나: 같은 판단을 신뢰구간으로 하면 95% CI 하한({m(P['lo'], 2)})이 −5보다 크므로 비열등성이 입증됩니다."))

# ------------------------------------------------------------------ 13-5 HbA1c direction
panels = []
for flip in (False, True):
    sg = -1 if flip else 1
    xl = (-0.7, 0.5) if flip else (-0.5, 0.7)
    pp = Plot(xl, (0.3, 2.8), w=420, h=250, ml=40, mr=16, mt=34, mb=76, show_yaxis=False, ygrid=False,
              xlabel=("대조약 − 신약 (%p)" if flip else "신약 − 대조약 (%p)"),
              xticks=[-0.4, -0.2, 0, 0.2, 0.4],
              xticklabels=[(v, m(v, 1) if v != 0 else "0") for v in (-0.4, -0.2, 0, 0.2, 0.4)])
    zone = np.array([-0.7, -0.4]) if flip else np.array([0.4, 0.7])
    pp.fill_between(zone, [0.3, 0.3], [2.8, 2.8], cls="a2")
    pp.vline(-0.4 if flip else 0.4, cls="ref strongref", dash=False, w=1.4)
    pp.vline(0, dash=True)
    for i, lab in enumerate(("FAS", "PP")):
        q = N.HB[lab]
        y = 2.05 - i * 1.0
        est, lo, hi = (sg * q["d"], -q["hi"], -q["lo"]) if flip else (q["d"], q["lo"], q["hi"])
        ci_row(pp, y, est, lo, hi, s=1)
        pp.text_px(6, pp.sy(y) + 5, lab, cls="lbl strong")
        pp.text(est, y, f"{m(est, 2)} ({m(lo, 2)} ~ {m(hi, 2)})", anchor="middle", dy=-12, cls="lbl small")
    if flip:
        pp.text(-0.55, 2.8, "열등 영역", anchor="middle", dy=14, cls="lbl small")
        pp.text(-0.4, 0.3, "한계 −0.4", anchor="end", dx=-4, dy=-4, cls="lbl small mute")
        pp.text_px(pp.sx(0) - 6, 213, "← 대조약이 더 낮춤", anchor="end", cls="lbl small mute")
        pp.text_px(pp.sx(0) + 6, 213, "신약이 더 낮춤 →", anchor="start", cls="lbl small mute")
        panel_title(pp, "나. 대조약 − 신약: 하한을 −0.4와 비교")
    else:
        pp.text(0.55, 2.8, "열등 영역", anchor="middle", dy=14, cls="lbl small")
        pp.text(0.4, 0.3, "한계 +0.4", anchor="start", dx=4, dy=-4, cls="lbl small mute")
        pp.text_px(pp.sx(0) - 6, 213, "← 신약이 더 낮춤", anchor="end", cls="lbl small mute")
        pp.text_px(pp.sx(0) + 6, 213, "대조약이 더 낮춤 →", anchor="start", cls="lbl small mute")
        panel_title(pp, "가. 신약 − 대조약: 상한을 +0.4와 비교")
    panels.append(pp.svg("HbA1c 변화량 차이의 95% 신뢰구간과 비열등성 한계"))
save("ch13_hba1c", figure(panels,
     "그림 13-5. 같은 HbA1c 시험 결과를 두 방향으로 그린 그림. HbA1c는 낮을수록 좋으므로, 가(신약 − 대조약)에서는 신뢰구간의 <b>상한</b>이 +0.4보다 작아야 하고, 나(대조약 − 신약)에서는 <b>하한</b>이 −0.4보다 커야 합니다. 두 그림은 좌우가 뒤집혔을 뿐 결론은 같습니다. 논문의 그림을 볼 때는 먼저 가로축이 어느 쪽에서 어느 쪽을 뺀 값인지, 열등 영역이 어느 쪽인지 확인하세요.", cols=2))

# ------------------------------------------------------------------ 13-6 Wald vs Newcombe
W, H = 660, 350
p = Plot((-17, 8), (0.3, 5.6), w=W, h=H, ml=112, mr=140, mt=36, mb=52, show_yaxis=False, ygrid=False,
         xlabel="제균율 차이 (신약 − 표준, %p)와 95% 신뢰구간", xticks=[-15, -10, -5, 0, 5],
         xticklabels=[(-15, "−15"), (-10, "−10 (−Δ)"), (-5, "−5"), (0, "0"), (5, "5")])
p.fill_between(np.array([-17, -10]), [0.3, 0.3], [5.6, 5.6], cls="a2")
p.vline(-10, cls="ref strongref", dash=False, w=1.4)
p.vline(0, dash=True)
p.text(-13.5, 5.6, "열등 영역", anchor="middle", dy=-10, cls="lbl small")
rows = [
    ("본 시험 ITT", "186/224 대 190/224", N.HP["ITT"]),
    ("소규모 시험", "48/50 대 50/50", N.SM["small"]),
]
yy = 4.55
for title, sub_, r in rows:
    p.text_px(10, p.sy(yy + 0.62) + 4, title, cls="lbl strong")
    p.text(0.6, yy + 0.62, sub_, anchor="start", dy=4, cls="lbl small mute")
    for meth, lo, hi, s in (("Wald", r["lo"], r["hi"], 2), ("Newcombe", r["nlo"], r["nhi"], 1)):
        ci_row(p, yy, r["d"] * 100, lo * 100, hi * 100, s=s)
        p.text_px(24, p.sy(yy) + 5, meth, cls="lbl")
        ok = lo > -0.10
        p.text_px(W - 140 + 12, p.sy(yy) + 1, f"{m(lo * 100)} ~ {m(hi * 100)}", cls="lbl small")
        p.text_px(W - 140 + 12, p.sy(yy) + 16, "비열등성 입증" if ok else "입증 못함", cls="lbl small strong" if ok else "lbl small mute")
        yy -= 0.9
    yy -= 0.75
save("ch13_ci_methods", figure(p.svg("Wald와 Newcombe 신뢰구간의 비교"),
     "그림 13-7. 같은 자료, 다른 신뢰구간 계산법. 비율이 중간 범위이고 표본이 큰 본 시험에서는 두 방법의 하한이 0.03%p밖에 차이 나지 않습니다. 반면 제균율이 100%에 가깝고 표본이 작은 시험에서는 Wald 하한(−9.4%p)이 한계를 넘지 않아 '비열등'이 되지만, Newcombe 하한(−13.5%p)으로는 입증하지 못합니다. 대조군 50명이 모두 성공해 Wald 표준오차에서 대조군 몫이 0이 된 것이 원인입니다."))

# ------------------------------------------------------------------ 13-7 meaning of a fixed absolute margin
pc = np.linspace(0.55, 0.9605, 200)
rr = (1 - pc + 0.10) / (1 - pc)
p = Plot((0.55, 0.97), (1, 3.6), w=600, h=320, ml=62, mr=24, mt=24, mb=54,
         xlabel="대조군(표준치료)의 성공률", ylabel="허용되는 실패 위험의 비(신약/표준)",
         xticks=[0.6, 0.7, 0.8, 0.9], xtickfmt=lambda v: f"{int(round(v * 100))}%", yticks=[1, 1.5, 2, 2.5, 3, 3.5])
p.line(pc, rr, s=1)
for v in (0.70, 0.85, 0.95):
    r_ = (1 - v + 0.10) / (1 - v)
    p.points([v], [r_], s=1, r=4.5)
    p.text(v, r_, f"{r_:.2f}배", anchor="end", dx=-8, dy=-6, cls="lbl strong")
p.text(0.70, (1 - 0.70 + 0.10) / 0.30, "실패 30% → 40%", anchor="start", dx=10, dy=16, cls="lbl small")
p.text(0.85, (0.25) / 0.15, "실패 15% → 25%", anchor="start", dx=10, dy=16, cls="lbl small")
p.text(0.95, 3.0, "실패 5% → 15%", anchor="end", dx=-8, dy=12, cls="lbl small")
save("ch13_abs_margin", figure(p.svg("대조군 성공률에 따라 달라지는 −10%p 한계의 의미"),
     "그림 13-6. 같은 −10%p 한계라도 대조군 성공률에 따라 허용되는 손실의 무게가 다릅니다. 성공률 70%에서는 실패 위험이 1.33배까지 늘어나는 것을 허용하지만, 95%에서는 3배(5% → 15%)까지 허용합니다. 표본크기 계산 때 가정한 성공률과 실제 성공률이 크게 다르면 한계의 임상적 의미를 다시 따져 봐야 합니다."))

# ------------------------------------------------------------------ 13-8 paper-style figure (English)
W, H = 660, 262
p = Plot((-14, 8), (0.3, 2.9), w=W, h=H, ml=130, mr=150, mt=40, mb=82, show_yaxis=False, ygrid=False,
         xlabel="Difference in eradication rate, percentage points (new minus standard)",
         xticks=[-12, -10, -8, -4, 0, 4, 8],
         xticklabels=[(-12, "−12"), (-8, "−8"), (-4, "−4"), (0, "0"), (4, "4"), (8, "8")])
p.vline(-10, cls="ref strongref", dash=True, w=1.4)
p.vline(0, cls="ref", dash=False, w=1.0)
p.text(-10, 2.9, "Noninferiority margin (−10)", anchor="middle", dy=-12, cls="lbl small")
p.text_px(W - 150 + 14, 26, "Difference (95% CI)", cls="lbl small strong")
for i, (lab, key) in enumerate((("Intention-to-treat", "ITT"), ("Per-protocol", "PP"))):
    q = N.HP[key]
    y = 2.15 - i * 1.05
    ci_row(p, y, q["d"] * 100, q["nlo"] * 100, q["nhi"] * 100, s=1)
    p.text_px(8, p.sy(y) - 1, lab, cls="lbl strong")
    p.text_px(8, p.sy(y) + 15, f"{q['x1']}/{q['n1']} vs {q['x2']}/{q['n2']}", cls="lbl small mute")
    p.text_px(W - 150 + 14, p.sy(y) + 5, f"{m(q['d'] * 100)} ({m(q['nlo'] * 100)} to {m(q['nhi'] * 100)})", cls="lbl small")
p.text_px(p.sx(0) - 8, H - 34, "← Favors standard therapy", anchor="end", cls="lbl small mute")
p.text_px(p.sx(0) + 8, H - 34, "Favors new therapy →", anchor="start", cls="lbl small mute")
save("ch13_paperforest", figure(p.svg("가상 논문의 비열등성 결과 그림"),
     "그림 13-8. 가상의 논문 그림. <i>Figure 2. Difference in eradication rates between the new dual therapy and bismuth quadruple therapy. Squares indicate point estimates and horizontal lines 95% confidence intervals (Newcombe hybrid score method). The dashed line indicates the prespecified noninferiority margin of −10 percentage points.</i>"))

print("figures written")
