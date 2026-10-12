import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt
import nums_ch06 as N

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def f2(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")


# ------------------------------------------------------------------ 6-1 quadrant scatter (8 patients)
P = N.pearson8()
x, y = N.AGE8, N.SBP8
p = Plot((38, 84), (118, 154), w=600, h=360, xlabel="나이 (세)", ylabel="수축기혈압 (mmHg)",
         xticks=[40, 50, 60, 70, 80], yticks=[120, 130, 140, 150])
p.fill_between([60, 84], [135, 135], [154, 154], s=1)
p.fill_between([38, 60], [118, 118], [135, 135], s=1)
p.vline(60, dash=True)
p.hline(135, dash=True)
p.text(60, 154, "평균 나이 60세", anchor="middle", dy=-6, cls="lbl mute small")
p.text(84, 135, "평균 SBP 135", anchor="end", dy=-6, cls="lbl mute small")
pos = x * 0 + 1
p.points(x, y, s=1, r=5.5)
offs = {0: (8, 4), 1: (8, 4), 2: (8, 16), 3: (-8, -8), 4: (8, 14), 5: (8, 14), 6: (8, 14), 7: (-8, -8)}
anch = {0: "start", 1: "start", 2: "start", 3: "end", 4: "start", 5: "start", 6: "start", 7: "end"}
for i, (a, b, pr) in enumerate(zip(x, y, P["prod"])):
    dx_, dy_ = offs[i]
    cls = "lbl strong" if pr > 0 else "lbl"
    p.text(a, b, f"{pr:+.0f}".replace("-", "−"), anchor=anch[i], dx=dx_, dy=dy_, cls=cls, size=12)
p.text(82.5, 152, "오른쪽 위: (+) × (+) = +", anchor="end", cls="lbl small")
p.text(39.5, 120.5, "왼쪽 아래: (−) × (−) = +", anchor="start", cls="lbl small")
p.text(39.5, 152, "왼쪽 위: (−) × (+) = −", anchor="start", cls="lbl mute small")
p.text(82.5, 120.5, "오른쪽 아래: (+) × (−) = −", anchor="end", cls="lbl mute small")
save("ch06_quadrant", figure(p.svg("환자 8명의 나이와 수축기혈압 산점도, 평균선으로 나눈 네 구역"),
     "그림 6-1. 환자 8명의 나이와 수축기혈압. 점선은 두 변수의 평균이고, 점 옆의 숫자는 (나이 − 60) × (SBP − 135), 즉 편차의 곱입니다. "
     "색칠된 두 구역(곱이 양수)과 색칠하지 않은 두 구역(곱이 음수)에 점이 각각 몇 개 있는지 봅니다."))

# ------------------------------------------------------------------ 6-2 gallery
G, sub, gr = N.gallery()


def gpanel(key, title, xlim, ylim):
    xs, ys = G[key]
    pl = Plot(xlim, ylim, w=300, h=240, ml=16, mr=12, mt=30, mb=18, xticks=[], yticks=[], ygrid=False,
              show_yaxis=False)
    pl.els.append(f'<line x1="{pl.ml}" y1="{pl.mt}" x2="{pl.ml}" y2="{pl.h - pl.mb}" class="axis"/>')
    return pl, xs, ys


panels = []
pl, xs, ys = gpanel("a", "", (-2.8, 2.8), (-2.8, 2.8))
pl.points(xs, ys, s=1, r=3.2)
panel_title(pl, f"A. r = {gr['a']:.2f} · 강한 양의 관계")
panels.append(pl)
pl, xs, ys = gpanel("b", "", (-2.8, 2.8), (-2.8, 2.8))
pl.points(xs, ys, s=1, r=3.2)
panel_title(pl, f"B. r = {gr['b']:.2f} · 중간 정도")
panels.append(pl)
pl, xs, ys = gpanel("c", "", (-2.8, 2.8), (-2.8, 2.8))
pl.points(xs, ys, s=1, r=3.2)
panel_title(pl, "C. r = 0.00 · 선형관계 없음")
panels.append(pl)
pl, xs, ys = gpanel("d", "", (-2.2, 2.2), (-0.8, 4.6))
pl.points(xs, ys, s=1, r=3.2)
panel_title(pl, "D. r = 0.00 · 강한 U자형 관계")
panels.append(pl)
pl, xs, ys = gpanel("e", "", (-3, 9), (-3, 9))
pl.points(xs[:-1], ys[:-1], s=1, r=3.2)
pl.points(xs[-1:], ys[-1:], s=2, r=4.5)
pl.text(xs[-1], ys[-1], "극단값 1개", anchor="end", dx=-9, dy=4, cls="lbl small")
pl.text(-2.6, 6.6, "이 점을 빼면 r = 0.00", anchor="start", cls="lbl mute small")
panel_title(pl, f"E. r = {gr['e']:.2f} · 극단값 하나가 만든 r")
panels.append(pl)
xs, ys = G["f"]
pl = Plot((-3, 3), (-3.2, 3.2), w=300, h=240, ml=16, mr=12, mt=30, mb=18, xticks=[], yticks=[], ygrid=False,
          show_yaxis=False)
pl.els.append(f'<line x1="{pl.ml}" y1="{pl.mt}" x2="{pl.ml}" y2="{pl.h - pl.mb}" class="axis"/>')
pl.fill_between([-0.6, 0.6], [-3.2, -3.2], [3.2, 3.2], s=1)
pl.points(xs[~sub], ys[~sub], s=4, r=2.8, hollow=True)
pl.points(xs[sub], ys[sub], s=1, r=3.2)
pl.text(2.9, -2.6, f"전체 r = {gr['f']:.2f}", anchor="end", cls="lbl mute small")
pl.text(2.9, -2.6, f"색칠한 범위만 r = {gr['f_sub']:.2f}", anchor="end", dy=15, cls="lbl small")
panel_title(pl, "F. 범위 제한")
panels.append(pl)
save("ch06_gallery", figure([q.svg("상관계수 예시 산점도") for q in panels],
     "그림 6-2. 여러 산점도와 Pearson 상관계수(각 n = 40, F는 n = 120). A–C에서는 r이 작아질수록 점이 직선 주위에 어떻게 흩어지는지 봅니다. "
     "D–F에서는 패널에 적힌 r과 점의 실제 모양이 어떻게 어긋나는지 봅니다. F의 색칠한 띠는 범위를 좁혀 모은 환자입니다.", cols=3))

# ------------------------------------------------------------------ 6-3 Pearson vs Spearman (2x2)
c, e = N.emax_data()
ps1 = N.pair_stats(c, e)
pA = Plot((0, 105), (0, 105), w=420, h=300, ml=52, xlabel="혈중 농도 (mg/L)", ylabel="효과 (최대효과 대비 %)",
          xticks=[0, 20, 40, 60, 80, 100], yticks=[0, 25, 50, 75, 100])
cc = np.linspace(0, 105, 200)
pA.line(cc, 100 * cc / (2 + cc), s=4, dash=True, w=1.5)
pA.points(c, e, s=1, r=4)
pA.text(100, 22, f"Pearson r = {ps1['r']:.2f}", anchor="end", cls="lbl strong")
pA.text(100, 22, f"Spearman ρ = {ps1['rho']:.2f}", anchor="end", dy=17, cls="lbl strong")
panel_title(pA, "A. 단조 증가하지만 곡선인 관계 (원래 값)")
rc, re_ = st.rankdata(c), st.rankdata(e)
pB = Plot((0, 21), (0, 21), w=420, h=300, ml=52, xlabel="농도의 순위", ylabel="효과의 순위",
          xticks=[1, 5, 10, 15, 20], yticks=[1, 5, 10, 15, 20])
pB.seg(1, 1, 20, 20, cls="ref", dash=True)
pB.points(rc, re_, s=1, r=4)
pB.text(20, 3, f"순위끼리의 Pearson r = {ps1['rho']:.2f}", anchor="end", cls="lbl strong")
pB.text(20, 3, "= Spearman ρ", anchor="end", dy=17, cls="lbl mute small")
panel_title(pB, "B. 같은 자료를 순위로 바꾼 것")
xo, yo = N.outlier_data()
ps2 = N.pair_stats(xo, yo)
ps2b = N.pair_stats(xo[:-1], yo[:-1])
pC = Plot((-2, 44), (0, 180), w=420, h=300, ml=52, xlabel="음주량 (잔/주)", ylabel="AST (U/L)",
          xticks=[0, 10, 20, 30, 40], yticks=[0, 40, 80, 120, 160])
pC.points(xo[:-1], yo[:-1], s=1, r=4)
pC.points(xo[-1:], yo[-1:], s=2, r=5)
pC.text(xo[-1], yo[-1], "1명", anchor="end", dx=-10, dy=4, cls="lbl small")
pC.text(42, 112, f"Pearson r = {ps2['r']:.2f}", anchor="end", cls="lbl strong")
pC.text(42, 112, f"Spearman ρ = {ps2['rho']:.2f}", anchor="end", dy=17, cls="lbl strong")
pC.text(42, 112, f"(이 1명 제외: r = {ps2b['r']:.2f})".replace("-", "−"), anchor="end", dy=34, cls="lbl mute small")
panel_title(pC, "C. 극단값 1명이 있는 자료 (원래 값)")
rx, ry = st.rankdata(xo), st.rankdata(yo)
pD = Plot((0, 17), (0, 17), w=420, h=300, ml=52, xlabel="음주량의 순위", ylabel="AST의 순위",
          xticks=[1, 4, 8, 12, 16], yticks=[1, 4, 8, 12, 16])
pD.points(rx[:-1], ry[:-1], s=1, r=4)
pD.points(rx[-1:], ry[-1:], s=2, r=5)
pD.text(rx[-1], ry[-1], "같은 1명", anchor="end", dx=-10, dy=4, cls="lbl small")
pD.text(0.8, 15.2, f"Spearman ρ = {ps2['rho']:.2f}", anchor="start", cls="lbl strong")
panel_title(pD, "D. 같은 자료를 순위로 바꾼 것")
save("ch06_spearman", figure([pA.svg("곡선 관계의 원래 값"), pB.svg("곡선 관계의 순위"),
                              pC.svg("극단값 자료의 원래 값"), pD.svg("극단값 자료의 순위")],
     "그림 6-3. 순위로 바꾸면 달라지는 것. A·B는 농도-효과 곡선(Emax 형태), C·D는 관계가 없는 15명에 극단적인 1명이 더해진 음주량-AST 자료입니다. "
     "원래 값(A·C)과 순위(B·D)에서 점의 모양과 패널에 적힌 계수가 어떻게 달라지는지 봅니다.", cols=2))

# ------------------------------------------------------------------ 6-4 least squares (mean line vs fitted line)
S8 = N.simple8()
b0, b1 = S8["b0"], S8["b1"]


def ls_panel(title, yhat, label, total, s):
    pl = Plot((38, 84), (118, 154), w=420, h=300, ml=52, xlabel="나이 (세)", ylabel="수축기혈압 (mmHg)",
              xticks=[40, 50, 60, 70, 80], yticks=[120, 130, 140, 150])
    for a, b, h in zip(x, y, yhat):
        pl.seg(a, b, a, h, cls=f"ln s2", w=2)
    xx = np.array([38, 84])
    pl.line(xx, (np.full(2, 135.0) if s == 4 else b0 + b1 * xx), s=s, w=2.2, dash=(s == 4))
    pl.points(x, y, s=1, r=5)
    pl.text(40, 151.5, label, anchor="start", cls="lbl strong")
    pl.text(40, 151.5, total, anchor="start", dy=17, cls="lbl")
    panel_title(pl, title)
    return pl


l1 = ls_panel("A. 나이를 모를 때: 평균으로 예측", np.full(8, 135.0), "예측값 = 평균 135", f"잔차 제곱합 = {S8['fit']['tss']:.0f} (SST)", 4)
l2 = ls_panel("B. 나이를 알 때: 최소제곱 회귀선", S8["fit"]["fit"], f"ŷ = {b0:.1f} + {b1:.3f} × 나이",
              f"잔차 제곱합 = {S8['fit']['rss']:.1f} (SSE)", 1)
save("ch06_ls", figure([l1.svg("평균선과 잔차"), l2.svg("회귀선과 잔차")],
     f"그림 6-4. 주황 세로선이 잔차(관측값 − 예측값)입니다. A(평균으로 예측)와 B(회귀직선으로 예측)에서 세로선의 길이와 패널에 적힌 잔차 제곱합을 비교합니다.",
     cols=2))

# ------------------------------------------------------------------ 6-5 residual patterns
S = N.simple100()
(x1, y1), (x2, y2) = N.resid_patterns()
f1 = N.ols(x1, y1)
f2_ = N.ols(x2, y2)


def rpanel(fit, title, xlim, ylim, yt, xt, xlab):
    pl = Plot(xlim, ylim, w=300, h=250, ml=44, mr=12, mt=30, mb=40, xlabel=xlab, ylabel="잔차",
              yticks=yt, xticks=xt)
    pl.hline(0, dash=True)
    pl.points(fit["fit"], fit["resid"], s=1, r=2.8)
    panel_title(pl, title)
    return pl


r1 = rpanel(S["fit"], "A. 문제 없음", (112, 142), (-45, 45), [-40, -20, 0, 20, 40], [115, 125, 135], "예측값 (mmHg)")
lo1, hi1 = f1["fit"].min(), f1["fit"].max()
r2 = rpanel(f1, "B. 부채꼴: 등분산 위반", (40, 420), (-300, 600), [-200, 0, 200, 400], [100, 200, 300, 400], "예측값 (만원)")
r3 = rpanel(f2_, "C. 휘어짐: 선형성 위반", (30, 142), (-48, 36), [-40, -20, 0, 20], [40, 70, 100, 130], "예측값 (%)")
save("ch06_resid", figure([r1.svg("정상 잔차 그림"), r2.svg("부채꼴 잔차"), r3.svg("휘어진 잔차")],
     "그림 6-5. 잔차 대 예측값 그림(residual vs fitted plot). A는 나이-혈압 자료(n = 100), B는 의료비처럼 오른쪽으로 치우친 자료, C는 곡선 관계(농도-효과)를 직선으로 적합한 자료입니다. "
     "0을 지나는 점선을 기준으로 예측값에 따라 잔차의 폭과 모양이 어떻게 변하는지 봅니다.",
     cols=3))
print("resid ranges", f1["fit"].min(), f1["fit"].max(), f1["resid"].min(), f1["resid"].max(), f2_["fit"].min(), f2_["fit"].max(), f2_["resid"].min(), f2_["resid"].max(),
      S["fit"]["fit"].min(), S["fit"]["fit"].max(), S["fit"]["resid"].min(), S["fit"]["resid"].max())

# ------------------------------------------------------------------ 6-6 CI vs PI
fit = S["fit"]
Xi = fit["XtX_inv"]
tc = fit["tcrit"]
ag = np.linspace(35, 92, 120)
X0 = np.column_stack([np.ones_like(ag), ag])
yh = X0 @ fit["b"]
sem = fit["sigma"] * np.sqrt(np.einsum("ij,jk,ik->i", X0, Xi, X0))
sep = fit["sigma"] * np.sqrt(1 + np.einsum("ij,jk,ik->i", X0, Xi, X0))
p = Plot((32, 94), (80, 190), w=600, h=380, xlabel="나이 (세)", ylabel="수축기혈압 (mmHg)",
         xticks=[40, 50, 60, 70, 80, 90], yticks=[80, 100, 120, 140, 160, 180])
p.fill_between([80, 94], [80, 80], [190, 190], cls="a4")
inr = ag <= 80
p.fill_between(ag, yh - tc * sem, yh + tc * sem, s=1)
p.line(ag[inr], (yh + tc * sep)[inr], s=2, dash=True, w=1.6)
p.line(ag[inr], (yh - tc * sep)[inr], s=2, dash=True, w=1.6)
p.line(ag[~inr | (ag >= 79.9)], (yh + tc * sep)[~inr | (ag >= 79.9)], s=4, dash=True, w=1.2)
p.line(ag[~inr | (ag >= 79.9)], (yh - tc * sep)[~inr | (ag >= 79.9)], s=4, dash=True, w=1.2)
p.points(S["age"], S["sbp"], s=4, r=3)
p.line(ag[inr], yh[inr], s=1, w=2.4)
p.line(ag[~inr | (ag >= 79.9)], yh[~inr | (ag >= 79.9)], s=1, w=2, dash=True)
v = S["pts"][70]
p.seg(70, v["pi"][0], 70, v["pi"][1], cls="ln s2", w=1.2)
p.seg(70, v["ci"][0], 70, v["ci"][1], cls="ln s1", w=3)
p.text(79, 87.5, f"굵은 세로선: 70세 평균의 95% 신뢰구간 {v['ci'][0]:.0f}–{v['ci'][1]:.0f}", anchor="end", cls="lbl small")
p.text(79, 82.3, f"가는 세로선: 70세 개인의 95% 예측구간 {v['pi'][0]:.0f}–{v['pi'][1]:.0f}", anchor="end", cls="lbl small")
p.text(87, 186, "관측 범위 밖", anchor="middle", dy=4, cls="lbl mute small")
p.text(87, 186, "(외삽)", anchor="middle", dy=19, cls="lbl mute small")
p.legend([("회귀선 (평균의 추정)", 1, "line"), ("평균의 95% 신뢰구간", 1, "soft"), ("개인의 95% 예측구간", 2, "dash")], X=p.ml + 12, Y=p.mt + 12)
save("ch06_cipi", figure(p.svg("회귀선과 신뢰구간, 예측구간"),
     f"그림 6-6. 지역약국 혈압측정 참여자 100명(가상 자료)의 나이와 수축기혈압. 파란 띠는 '그 나이 사람들의 평균 SBP'에 대한 95% 신뢰구간이고, "
     "주황 점선은 '그 나이의 새로운 한 사람'의 SBP가 들어갈 95% 예측구간입니다. 70세의 두 세로선에서 두 구간의 폭을 비교합니다. "
     "회색 영역(81세 이상)은 자료가 없어 직선을 연장한 구간입니다."))

# ------------------------------------------------------------------ 6-7 confounding by indication
M = N.multi240()
d = M["d"]
m2 = M["m2"]
m1 = M["m1"]
t1, t2 = np.percentile(d["base"], [100 / 3, 200 / 3])
grp = np.where(d["base"] < t1, 0, np.where(d["base"] < t2, 1, 2))
rng = np.random.default_rng(3)
jit = rng.uniform(-0.22, 0.22, d["n"])
p = Plot((0.5, 4.5), (100, 200), w=600, h=420, xlabel="복용 중인 항고혈압제 계열 수", ylabel="6개월 후 수축기혈압 (mmHg)",
         xticks=[1, 2, 3, 4], yticks=[100, 120, 140, 160, 180, 200])
labels = []
meds = []
for g_, s_ in ((0, 3), (1, 1), (2, 2)):
    m = grp == g_
    p.points(d["cls"][m] + jit[m], d["fu"][m], s=s_, r=2.6)
    bm = np.median(d["base"][m])
    meds.append(bm)
xx = np.array([0.8, 4.2])
for g_, s_ in ((0, 3), (1, 1), (2, 2)):
    bm = meds[g_]
    p.line(xx, m2["b"][0] + m2["b"][1] * xx + m2["b"][2] * bm, s=s_, w=2.6)
p.line(xx, m1["b"][0] + m1["b"][1] * xx, s=4, w=2.6, dash=True)
p.legend([(f"기저 SBP 하위 1/3 (중앙값 {meds[0]:.0f})", 3, "dot"), (f"기저 SBP 중간 1/3 (중앙값 {meds[1]:.0f})", 1, "dot"),
          (f"기저 SBP 상위 1/3 (중앙값 {meds[2]:.0f})", 2, "dot"), (f"보정 전 회귀선 (기울기 +{m1['b'][1]:.1f}/계열)", 4, "dash")], X=p.ml + 10, Y=p.mt + 10)
p.text(4.4, 104, f"실선: 기저 SBP를 보정한 회귀선 (기울기 {m2['b'][1]:.1f}/계열)".replace("-", "−"), anchor="end", cls="lbl small")
save("ch06_confound", figure(p.svg("항고혈압제 계열 수와 추적 수축기혈압, 기저 혈압에 따른 교란"),
     "그림 6-7. 고혈압 환자 240명(가상 자료). 점의 색은 기저 SBP 묶음입니다. 주황 점(기저 SBP 상위 1/3)이 오른쪽 위에 몰려 있는 것과, "
     "회색 점선(보정 전 회귀선)과 색 실선(기저 SBP를 보정한 회귀선)의 기울기 방향이 반대인 것을 봅니다. 점은 겹치지 않게 좌우로 약간 흩뜨렸습니다.",
     ))
print("tertiles", t1, t2, meds, "x-right label y", m1["b"][0] + m1["b"][1] * 4.2)
print("done")
