import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, scipy.stats as st
from itertools import product
from svgplot import Plot, figure, panel_title, fmt

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def errbar(p, x, lo, hi, cls="strongref", w=1.8, cap=0.05):
    p.seg(x, lo, x, hi, cls=cls, w=w)
    p.seg(x - cap, lo, x + cap, lo, cls=cls, w=w)
    p.seg(x - cap, hi, x + cap, hi, cls=cls, w=w)


M = lambda v, nd=1: f"{v:.{nd}f}".replace("-", "−")

# ------------------------------------------------------------------ data (see nums_ch03.py)
before = np.array([131, 146, 144, 154, 164, 159, 155, 164, 149, 161, 155, 142.])
after = np.array([126, 144, 131, 147, 162, 150, 154, 147, 142, 154, 139, 144.])
d = after - before
n = len(d)
md, sd = d.mean(), d.std(ddof=1)
se = sd / np.sqrt(n); tc = st.t.ppf(.975, n - 1)
lo, hi = md - tc * se, md + tc * se
print("paired", md, sd, lo, hi)

# ------------------------------------------------------------------ 3-1 slope plot + differences
p1 = Plot((-0.42, 1.42), (120, 170), w=420, h=330, ml=52, ylabel="수축기혈압 (mmHg)",
          yticks=[120, 130, 140, 150, 160, 170], xticks=[0, 1], xticklabels=[(0, "중재 전"), (1, "3개월 후")])
for b_, a_ in zip(before, after):
    s = 2 if a_ > b_ else 4
    p1.line([0, 1], [b_, a_], s=s, w=1.4 if s == 4 else 2)
    p1.points([0, 1], [b_, a_], s=s, r=3)
p1.line([0, 1], [before.mean(), after.mean()], s=1, w=3.2)
p1.points([0, 1], [before.mean(), after.mean()], s=1, r=5.5)
p1.text(-0.07, before.mean(), f"평균 {before.mean():.1f}", anchor="end", dy=4, cls="lbl strong", size=12)
p1.text(1.07, after.mean(), f"평균 {after.mean():.1f}", anchor="start", dy=4, cls="lbl strong", size=12)
p1.text(1.07, 144, "", anchor="start")
panel_title(p1, "(가) 환자 12명의 전후 값")

p2 = Plot((0.3, 1.95), (-20, 6), w=420, h=330, ml=52, ylabel="변화량 (후 − 전, mmHg)",
          yticks=[-20, -15, -10, -5, 0, 5], xticks=[0.75, 1.45],
          xticklabels=[(0.75, "환자별 변화"), (1.45, "평균과 95% CI")],
          ytickfmt=lambda v: ("+" if v > 0 else "") + fmt(v).replace("-", "−"))
p2.hline(0, dash=True)
p2.text(0.32, 0, "변화 없음", dy=-6, cls="lbl mute small")
# simple jitter for ties
xs = np.zeros(n)
seen = {}
for i in np.argsort(d, kind="stable"):
    k = seen.get(d[i], 0); seen[d[i]] = k + 1
    xs[i] = 0.75 + ((k + 1) // 2) * 0.07 * (1 if k % 2 else -1)
p2.points(xs, d, s=1, r=4)
errbar(p2, 1.45, lo, hi, cap=0.05)
p2.seg(1.36, md, 1.54, md, cls="strongref", w=2.6)
p2.text(1.58, md, M(md), dy=4, cls="lbl strong")
p2.text(1.45, hi, f"{M(lo)} ~ {M(hi)}", anchor="middle", dy=-10, cls="lbl small")
panel_title(p2, "(나) 차이 d = 후 − 전")
save("ch03_paired", figure([p1.svg("환자별 전후 수축기혈압"), p2.svg("전후 차이의 분포와 평균, 95% 신뢰구간")],
     "그림 3-1. 가정혈압 자가측정 교육을 받은 환자 12명의 수축기혈압. (가) 선 하나가 환자 한 명이며, 11명은 떨어지고 1명(주황)은 올랐습니다. 환자 사이의 차이(130–165 mmHg)는 크지만 각자의 변화는 비슷한 방향입니다. (나) 대응표본 t 검정은 오른쪽의 차이 12개만 분석합니다. 평균 변화 −7.0 mmHg의 95% 신뢰구간이 0을 포함하지 않습니다.", cols=2))

# ------------------------------------------------------------------ 3-2 SD of difference vs rho
sb, sa = before.std(ddof=1), after.std(ddof=1)
r = np.corrcoef(before, after)[0, 1]
rho = np.linspace(0, 0.98, 200)
sdd = np.sqrt(sb ** 2 + sa ** 2 - 2 * rho * sb * sa)
p = Plot((0, 1), (0, 16), w=600, h=320, xlabel="전후 측정값의 상관계수 ρ", ylabel="차이의 표준편차 (mmHg)",
         xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 4, 8, 12, 16], xtickfmt=lambda v: fmt(v, 1))
p.line(rho, sdd, s=1, w=2.4)
s0 = np.sqrt(sb ** 2 + sa ** 2); sr = np.sqrt(sb ** 2 + sa ** 2 - 2 * r * sb * sa)
p.points([0], [s0], s=2, r=6)
p.points([r], [sr], s=1, r=6)
p.vline(r, y1=sr, dash=True)
p.text(0, s0, f"ρ = 0 (독립표본처럼 분석): SD {s0:.1f}, SE {s0 / np.sqrt(n):.2f}", dx=12, dy=-10, cls="lbl strong")
p.text(r, sr, f"이 자료 ρ = {r:.2f}: SD {sr:.1f}, SE {sr / np.sqrt(n):.2f}", anchor="end", dx=-14, dy=22, cls="lbl strong")
p.text(0.5, np.sqrt(sb ** 2 + sa ** 2 - sb * sa), "상관이 클수록 차이가 덜 흩어짐", dx=10, dy=-10, cls="lbl mute small")
save("ch03_rho", figure(p.svg("전후 상관계수에 따른 차이의 표준편차"),
     "그림 3-2. 전후 값의 표준편차가 각각 약 10 mmHg일 때, 두 값의 상관계수 ρ에 따라 차이 d의 표준편차가 어떻게 줄어드는지 보여 줍니다. 같은 사람을 두 번 재면 ρ가 크므로(이 자료 0.82) 차이의 SD가 6.0으로 작아지고, 표준오차는 독립표본처럼 분석할 때(4.06)의 약 2.4분의 1(1.72)이 됩니다."))

# ------------------------------------------------------------------ 3-3 regression to the mean
mu, sbt, sem = 140.0, 12.0, 8.0
rel = sbt ** 2 / (sbt ** 2 + sem ** 2)
rng2 = np.random.default_rng(512)
Mn = 600
tr = rng2.normal(mu, sbt, Mn); a1 = tr + rng2.normal(0, sem, Mn); a2 = tr + rng2.normal(0, sem, Mn)
sel = a1 >= 160
m1s, m2s = a1[sel].mean(), a2[sel].mean()
print("RTM fig: n sel", sel.sum(), "mean x1 %.2f x2 %.2f drop %.2f" % (m1s, m2s, m1s - m2s))
p = Plot((95, 200), (95, 200), w=600, h=450, xlabel="첫 번째 측정: 선별 검사 수축기혈압 (mmHg)",
         ylabel="두 번째 측정 수축기혈압 (mmHg)", xticks=[100, 120, 140, 160, 180, 200],
         yticks=[100, 120, 140, 160, 180, 200], xgrid=True)
p.fill_between([160, 200], [95, 95], [200, 200], cls="a2")
p.points(a1[~sel], a2[~sel], s=4, r=2.3)
p.points(a1[sel], a2[sel], s=2, r=3)
p.line([95, 200], [95, 200], s=4, dash=True, w=1.6)
xx = np.array([100, 200])
p.line(xx, mu + rel * (xx - mu), s=1, w=2.2)
p.vline(160, dash=False, cls="ref", w=1.2)
p.text(158, 197, "등록 기준 ≥ 160 mmHg", anchor="end", dy=4, cls="lbl strong")
p.legend([("y = x (변화 없음)", 4, "dash"), ("기대되는 두 번째 측정값", 1, "line"),
          (f"첫 측정 ≥ 160으로 선정된 {sel.sum()}명", 2, "dot")], X=p.ml + 12, Y=p.sy(188))
p.seg(m1s, m1s, m1s, m2s, cls="strongref", w=2)
p.points([m1s], [m1s], s=4, r=4, hollow=True)
p.els.append(f'<circle cx="{p.sx(m1s):.1f}" cy="{p.sy(m2s):.1f}" r="6.5" class="pt f2" stroke-width="2"/>')
p.text(198, 110, f"선정군 평균: {m1s:.1f} → {m2s:.1f}", anchor="end", cls="lbl strong")
p.text(198, 110, f"치료 없이 {m1s - m2s:.1f} mmHg 하락", anchor="end", dy=18, cls="lbl strong")
p.text(198, 110, "(속이 빈 점 → 채운 점)", anchor="end", dy=34, cls="lbl small")
save("ch03_rtm", figure(p.svg("선별 검사 값으로 대상자를 뽑을 때의 평균으로의 회귀"),
     f"그림 3-3. 가상의 성인 600명에게 아무 치료 없이 혈압을 두 번 측정했습니다(평소 혈압 평균 140, 사람 간 SD 12, 측정할 때마다의 변동 SD 8 mmHg). 첫 측정이 160 mmHg 이상인 {sel.sum()}명(주황)만 뽑으면, 두 번째 측정 평균은 {m1s:.1f}에서 {m2s:.1f} mmHg로 {m1s - m2s:.1f} mmHg 낮아집니다. 첫 측정이 높았던 사람 중에는 그날 우연히 높게 나온 사람이 많기 때문입니다. 파란 선은 첫 측정값이 주어졌을 때 기대되는 두 번째 측정값입니다."))

# ------------------------------------------------------------------ 3-4 signed ranks (deprescribing)
b = np.array([12, 10, 14, 9, 11, 13, 10, 12, 15, 11, 9, 13, 10, 12])
a = np.array([9, 9, 11, 9, 10, 10, 8, 12, 11, 9, 10, 10, 12, 9])
dd = a - b
nz_idx = np.where(dd != 0)[0]
rk = st.rankdata(np.abs(dd[nz_idx]))
rank_of = {int(i): float(r_) for i, r_ in zip(nz_idx, rk)}
order = sorted(range(len(dd)), key=lambda i: (abs(dd[i]) if dd[i] != 0 else -1, dd[i] > 0, i))
rows = len(order)
p = Plot((-4.8, 2.8), (0.3, rows + 0.7), w=640, h=470, ml=76, mr=190, mt=36, mb=54,
         xlabel="변화량 d (후 − 전, 약물 수)", xticks=[-4, -3, -2, -1, 0, 1, 2], show_yaxis=False, ygrid=False, xgrid=True,
         xtickfmt=lambda v: ("+" if v > 0 else "") + fmt(v).replace("-", "−"))
groups = [(0, "d = 0 → 제외"), (1, "1–3위 → 평균 2"), (2, "4–6위 → 평균 5"),
          (3, "7–11위 → 평균 9"), (4, "12위")]
ypos = {}
for k, i in enumerate(order):
    ypos[i] = rows - k
# tie-group bands
for gi, (g, lab) in enumerate(groups):
    ys = [ypos[i] for i in order if abs(dd[i]) == g]
    if gi % 2 == 0:
        p.fill_between([-4.8, 2.8], [min(ys) - 0.5] * 2, [max(ys) + 0.5] * 2, cls="a4")
    p.text_px(p.w - p.mr + 60, p.sy((min(ys) + max(ys)) / 2) + 4, lab, cls="lbl small")
p.vline(0, dash=False, cls="strongref")
for i in order:
    y = ypos[i]
    p.text_px(p.ml - 10, p.sy(y) + 4, f"환자 {i + 1}", anchor="end", cls="lbl small")
    if dd[i] == 0:
        p.points([0], [y], s=4, r=4.5, hollow=True)
        p.text_px(p.w - p.mr + 26, p.sy(y) + 4, "–", anchor="middle", cls="lbl mute")
        continue
    s = 1 if dd[i] < 0 else 2
    p.seg(0, y, dd[i], y, cls=f"ln s{s}", w=2.2)
    p.points([dd[i]], [y], s=s, r=5)
    rv = rank_of[i]
    lab = ("+" if dd[i] > 0 else "−") + fmt(rv)
    p.text_px(p.w - p.mr + 26, p.sy(y) + 4, lab, anchor="middle", cls="lbl strong" if dd[i] > 0 else "lbl")
p.text_px(p.w - p.mr + 26, p.mt - 14, "부호순위", anchor="middle", cls="lbl small mute")
p.text_px(p.ml - 10, p.mt - 14, "", anchor="end")
p.text(-4.7, rows + 0.7, "감소", dy=-8, cls="lbl mute small")
p.text(2.7, rows + 0.7, "증가", anchor="end", dy=-8, cls="lbl mute small")
save("ch03_signrank", figure(p.svg("환자별 약물 수 변화와 부호순위"),
     "그림 3-4. 약물 정리 전후의 복용 약물 수 변화를 |d|가 작은 순서로 정렬했습니다. 변화가 0인 2명은 제외하고, 남은 12명의 |d|에 순위를 매깁니다(같은 |d|는 중간순위). 늘어난 2명(주황)의 순위 2와 5를 더한 W+ = 7이 검정통계량입니다."))

# ------------------------------------------------------------------ 3-5 permutation distribution of W+
dist = {}
for signs in product((0, 1), repeat=len(rk)):
    w = float(np.dot(signs, rk))
    dist[w] = dist.get(w, 0) + 1
vals = np.array(sorted(dist)); cnt = np.array([dist[v] for v in vals]); prob = cnt / cnt.sum()
Wp = 7.0
tail = prob[vals <= Wp].sum()
print("perm tail", tail, "two-sided", 2 * tail, "max prob", prob.max())
p = Plot((-2, 80), (0, 0.075), w=600, h=320, xlabel="W+ (늘어난 환자들의 순위합)", ylabel="확률",
         xticks=[0, 7, 39, 71, 78], yticks=[0, 0.02, 0.04, 0.06], ytickfmt=lambda v: fmt(v, 2))
inside = (vals > Wp) & (vals < 78 - Wp)
p.bars(vals[inside], prob[inside], 0.8, rounded=False, cls="a4")
p.bars(vals[~inside], prob[~inside], 0.8, s=2, rounded=False)
p.vline(Wp, y1=0.045)
p.text(Wp, 0.045, "관측 W+ = 7", anchor="middle", dy=-6, cls="lbl strong")
p.vline(78 - Wp, y1=0.045)
p.text(78 - Wp, 0.045, "W+ = 71 (반대쪽)", anchor="middle", dy=-6, cls="lbl")
p.text(6.2, 0.012, f"{tail:.4f}", anchor="end", cls="lbl small")
p.text(71.8, 0.012, f"{tail:.4f}", anchor="start", cls="lbl small")
p.text(39, 0.069, "기댓값 39 (= 78 ÷ 2)", anchor="middle", cls="lbl mute small")
save("ch03_wperm", figure(p.svg("부호를 무작위로 바꾼 4096가지 경우의 W+ 분포"),
     f"그림 3-5. 3개월 사이 약물 수의 변화가 어느 쪽으로도 치우치지 않는다면(귀무가설) 12명 각자의 변화가 +인지 −인지는 동전 던지기와 같습니다. 12개의 순위(중간순위 포함)에 부호를 붙이는 2¹² = 4,096가지 경우의 W+를 모두 계산한 분포에서, 관측값 7 이하일 확률은 {tail:.4f}, 반대쪽까지 더한 양측 p = {2 * tail:.4f}입니다."))
