import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np, scipy.stats as st
from html import escape
from svgplot import Plot, figure, panel_title, fmt
from nums_ch04 import running_data, TINY, anova, pairwise_all, kw, power_sim

OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def pct(v):
    return f"{int(round(v * 100))}%"


G = running_data()
A = anova(G)
NAMES = ["일반 복약지도", "1회 상담", "집중 상담"]

# ------------------------------------------------------------------ 4-1 strip plot of running example
rng = np.random.default_rng(3)
p = Plot((0.45, 4.15), (45, 110), w=600, h=360, ylabel="6개월 PDC (%)", yticks=[50, 60, 70, 80, 90, 100],
         xticks=[1, 2, 3], xticklabels=[(i + 1, f"{NAMES[i]} (n = 30)") for i in range(3)], mb=40)
gm = A["gm"]
p.hline(gm, x0=0.5, x1=4.12)
p.text(4.12, gm, f"전체 평균 {gm:.1f}", anchor="end", dy=15, cls="lbl mute small")
tc = st.t.ppf(0.975, 29)
for i, g in enumerate(G):
    x = i + 1
    jit = (rng.random(len(g)) - 0.5) * 0.36
    p.points(x - 0.08 + jit, g, s=1, r=3.4, hollow=True)
    m = g.mean(); se = g.std(ddof=1) / np.sqrt(len(g))
    lo, hi = m - tc * se, m + tc * se
    xc = x + 0.3
    p.seg(xc, lo, xc, hi, cls="ln s2", w=2.4)
    p.seg(xc - 0.05, lo, xc + 0.05, lo, cls="ln s2", w=2)
    p.seg(xc - 0.05, hi, xc + 0.05, hi, cls="ln s2", w=2)
    p.seg(x - 0.3, m, xc + 0.07, m, cls="ln s2", w=3)
    p.text(xc + 0.09, m, f"{m:.1f}", dy=4, cls="lbl strong")
p.legend([("환자 한 명", 1, "dot"), ("평균과 95% 신뢰구간", 2, "line")], X=76, Y=26)
save("ch04_strip", figure(p.svg("세 군의 PDC 분포"),
     "그림 4-1. 약사 중재 세 군의 6개월 PDC(가상 자료, 군당 30명). 점 하나가 환자 한 명이고, 주황 가로선과 세로선은 군별 평균과 그 95% 신뢰구간입니다. 평균은 중재 강도에 따라 높아지지만, 세 군의 분포는 크게 겹칩니다."))

# ------------------------------------------------------------------ 4-2 FWER by number of groups
ks = [3, 4, 5, 6]
indep, actual = [], []
for k in ks:
    m = k * (k - 1) // 2
    indep.append(1 - 0.95 ** m)
    df = 30 * k - k
    actual.append(st.studentized_range.sf(st.t.ppf(0.975, df) * np.sqrt(2), k, df))
p = Plot((0.4, 4.6), (0, 0.6), w=600, h=320, ylabel="우연히 하나라도 유의할 확률", yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
         ytickfmt=pct, xticks=[1, 2, 3, 4], xticklabels=[(i + 1, f"{k}군 ({k * (k - 1) // 2}쌍)") for i, k in enumerate(ks)],
         xlabel="비교하는 군의 수 (쌍별 비교 수)")
xs = np.arange(1, 5)
p.bars(xs - 0.18, indep, 0.32, s=4)
p.bars(xs + 0.18, actual, 0.32, s=1)
for x, a_, b_ in zip(xs, indep, actual):
    p.text(x - 0.18, a_, f"{a_ * 100:.1f}%", anchor="middle", dy=-6, cls="lbl small")
    p.text(x + 0.18, b_, f"{b_ * 100:.1f}%", anchor="middle", dy=-6, cls="lbl strong small")
p.hline(0.05, dash=True)
p.text(4.6, 0.05, "α = 5%", anchor="end", dy=-5, cls="lbl mute small")
p.legend([("1 − 0.95^m (비교가 서로 독립이라는 가정)", 4, "box"), ("실제 값 (군당 30명, 비교 사이 상관 반영)", 1, "box")], X=76, Y=30)
save("ch04_fwer", figure(p.svg("군 수에 따른 쌍별 비교의 가족단위 오류율"),
     "그림 4-2. 모든 평균이 실제로 같을 때, 보정 없이 모든 쌍을 t 검정(α = 0.05)으로 비교하면 적어도 하나가 우연히 유의할 확률. 회색은 1장 다 절의 공식 1 − 0.95<sup>m</sup>(m = 쌍의 수), 파랑은 쌍별 비교들이 같은 군을 공유해 서로 상관되어 있다는 점을 반영한 실제 값입니다. 어느 쪽이든 5%를 크게 넘습니다."))

# ------------------------------------------------------------------ 4-3 pairwise CIs: unadjusted vs Tukey
res, tcrit, qcrit = pairwise_all(G, A)
rows = []
for r in res:
    rows.append(dict(label=f"{NAMES[r['j']]} − {NAMES[r['i']]}", header=True))
    rows.append(dict(label="보정 없는 95% 신뢰구간", est=r["diff"], lo=r["lo"], hi=r["hi"], s=4))
    rows.append(dict(label="Tukey 동시 95% 신뢰구간", est=r["diff"], lo=r["tlo"], hi=r["thi"], s=1))
W, ROW, TOP = 660, 27, 14
LW, EW = 218, 150
H = TOP + ROW * len(rows) + 56
X0, X1 = LW, W - EW
xl = (-8, 18)


def sxf(v):
    return X0 + (v - xl[0]) / (xl[1] - xl[0]) * (X1 - X0)


o = []
yb = TOP + ROW * len(rows) + 4
for t in (-5, 0, 5, 10, 15):
    x = sxf(t)
    o.append(f'<line x1="{x:.1f}" y1="{TOP - 4}" x2="{x:.1f}" y2="{yb}" class="grid"/>')
    o.append(f'<text x="{x:.1f}" y="{yb + 18}" text-anchor="middle" class="tick">{fmt(t)}</text>')
o.append(f'<line x1="{X0}" y1="{yb}" x2="{X1}" y2="{yb}" class="axis"/>')
o.append(f'<line x1="{sxf(0):.1f}" y1="{TOP - 4}" x2="{sxf(0):.1f}" y2="{yb}" class="ref strongref" stroke-width="1.4"/>')
o.append(f'<text x="{(X0 + X1) / 2:.1f}" y="{H - 10}" text-anchor="middle" class="axlab">평균 PDC 차이 (%p)</text>')
o.append(f'<text x="{W - 4}" y="{TOP - 2}" text-anchor="end" class="lbl small mute">차이 (95% 신뢰구간)</text>')
for i, r in enumerate(rows):
    y = TOP + ROW * i + ROW / 2 + 4
    if r.get("header"):
        o.append(f'<text x="4" y="{y + 4.5:.1f}" class="lbl" font-weight="600">{escape(r["label"])}</text>')
        continue
    o.append(f'<text x="18" y="{y + 4.5:.1f}" class="lbl small">{escape(r["label"])}</text>')
    a_, b_ = sxf(r["lo"]), sxf(r["hi"])
    o.append(f'<line x1="{a_:.1f}" y1="{y:.1f}" x2="{b_:.1f}" y2="{y:.1f}" class="ln s{r["s"]}" stroke-width="2.2"/>')
    for xx in (a_, b_):
        o.append(f'<line x1="{xx:.1f}" y1="{y - 4:.1f}" x2="{xx:.1f}" y2="{y + 4:.1f}" class="ln s{r["s"]}" stroke-width="1.6"/>')
    xe = sxf(r["est"])
    o.append(f'<rect x="{xe - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" class="f{r["s"]}"/>')
    lo_s = f'{r["lo"]:.2f}'.replace("-", "−"); hi_s = f'{r["hi"]:.2f}'.replace("-", "−")
    o.append(f'<text x="{W - 4}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num small">{r["est"]:.2f} ({lo_s}, {hi_s})</text>')
svg = (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="쌍별 평균 차이의 신뢰구간" '
       f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')
save("ch04_ci", figure(svg,
     "그림 4-3. 세 쌍의 평균 차이와 95% 신뢰구간. 회색은 보정하지 않은 구간(±1.99 × SE), 파랑은 Tukey 방법의 동시 신뢰구간(±2.38 × SE)입니다. Tukey 구간은 세 구간이 모두 동시에 참값을 포함할 확률이 95%가 되도록 넓힌 것이며, 0(세로선)을 포함하지 않는 쌍은 집중 상담 − 일반 복약지도뿐입니다."))

# ------------------------------------------------------------------ 4-4 same means, different spread
rng = np.random.default_rng(11)
base = [rng.normal(0, 1, 10) for _ in range(3)]
base = [(b_ - b_.mean()) / b_.std(ddof=1) for b_ in base]
panels = []
jr = np.random.default_rng(8)
jits = [(jr.random(10) - 0.5) * 0.34 for _ in range(3)]
for sd, title in ((4.0, "군 안의 흩어짐이 작을 때 (SD 4)"), (12.0, "군 안의 흩어짐이 클 때 (SD 12)")):
    gs = [m + sd * b_ for m, b_ in zip((66, 72, 78), base)]
    aa = anova(gs)
    pl = Plot((0.45, 3.55), (40, 106), w=420, h=300, ml=56, mt=30, ylabel="PDC (%)", yticks=[40, 50, 60, 70, 80, 90, 100],
              xticks=[1, 2, 3], xticklabels=[(1, "일반"), (2, "1회 상담"), (3, "집중 상담")], mb=36)
    for i, g in enumerate(gs):
        pl.points(i + 1 + jits[i], g, s=1, r=3.4, hollow=True)
        pl.seg(i + 1 - 0.3, g.mean(), i + 1 + 0.3, g.mean(), cls="ln s2", w=3)
    ptxt = "P < 0.001" if aa["p"] < 0.001 else f"P = {aa['p']:.2f}"
    pl.text(0.55, 101, f"F = {aa['F']:.2f}, {ptxt}", cls="lbl strong")
    pl.text(0.55, 101, f"MSB {aa['msb']:.0f} ÷ MSW {aa['msw']:.0f}", dy=17, cls="lbl mute small")
    panel_title(pl, title)
    panels.append(pl.svg(title))
save("ch04_spread", figure(panels,
     "그림 4-4. 두 그림 모두 세 군의 평균(주황 선: 66, 72, 78)과 군당 인원(10명)이 같습니다. 군 안의 흩어짐이 작으면(왼쪽) 평균 차이가 흩어짐에 비해 커서 F가 크고, 흩어짐이 크면(오른쪽) 같은 평균 차이도 우연히 생길 수 있는 수준이라 F가 작습니다. 군간 평균제곱(MSB)은 360으로 같고 군내 평균제곱(MSW)만 16과 144로 다릅니다.", cols=2))

# ------------------------------------------------------------------ 4-5 sum of squares decomposition (tiny example)
p = Plot((-0.6, 18.2), (54, 90), w=620, h=340, ylabel="PDC (%)", yticks=[55, 60, 65, 70, 75, 80, 85, 90],
         xticks=[3, 9, 15], xticklabels=[(3, "일반 복약지도"), (9, "1회 상담"), (15, "집중 상담")], mb=40)
p.hline(72, x0=-0.6, x1=18.2)
starts = [1, 7, 13]
for gi, (g, s0) in enumerate(zip(TINY, starts)):
    m = g.mean()
    xs = np.arange(s0, s0 + 5)
    for x, v in zip(xs, g):
        p.seg(x, m, x, v, cls="ln s4", w=1.8)
    p.seg(s0 - 0.45, m, s0 + 4.45, m, cls="ln s1", w=2.2)
    p.points(xs, g, s=1, r=4.2)
    xb = s0 - 1.0
    if m != 72:
        p.seg(xb, 72, xb, m, cls="ln s2", w=4.5)
    p.text(s0 + 4.45, m, f"군 평균 {m:g}", dy=15, cls="lbl small", anchor="end")
p.text(18.2, 72, "전체 평균 72", anchor="end", dy=16, cls="lbl mute small")
p.legend([("군내 편차 = 값 − 군 평균", 4, "line"), ("군간 편차 = 군 평균 − 전체 평균", 2, "line")], X=76, Y=30)
save("ch04_ss", figure(p.svg("제곱합 분해"),
     "그림 4-5. 작은 예제(군당 5명)의 제곱합 분해. 회색 세로선은 각 환자가 자기 군 평균(파란 가로선)에서 떨어진 거리(군내 편차), 주황 막대는 군 평균이 전체 평균(점선)에서 떨어진 거리(군간 편차)입니다. 회색 선 길이를 제곱해 모두 더하면 군내 제곱합 528, 주황 막대 길이를 제곱해 군 인원(5명)만큼 더하면 군간 제곱합 360입니다. 1회 상담군은 군 평균이 전체 평균과 같아 주황 막대가 없습니다."))

# ------------------------------------------------------------------ 4-6 ranks on a number line (tiny example)
kt = kw(TINY)
allv = np.concatenate(TINY)
ranks = st.rankdata(allv)
p = Plot((55.5, 88.5), (0.4, 3.6), w=660, h=250, ml=110, mr=112, mt=16, mb=48, xlabel="PDC (%)",
         xticks=[60, 65, 70, 75, 80, 85], show_yaxis=False, ygrid=False, xgrid=True)
rows_y = {0: 3, 1: 2, 2: 1}
for gi, g in enumerate(TINY):
    y = rows_y[gi]
    p.hline(y, x0=55.5, x1=88.5, cls="grid", dash=False)
    p.points(g, [y] * len(g), s=1, r=5)
    for v in g:
        rk = ranks[list(allv).index(v)]
        p.text(v, y, f"{rk:g}", anchor="middle", dy=-10, cls="lbl strong small")
    p.text_px(p.ml - 12, p.sy(y) + 4.5, NAMES[gi], anchor="end")
    p.text_px(p.w - p.mr + 12, p.sy(y) + 4.5, f"평균 순위 {kt['mean_rank'][gi]:.1f}", anchor="start", cls="lbl strong")
save("ch04_ranks", figure(p.svg("세 군 15명의 값과 전체 순위"),
     "그림 4-6. 작은 예제 15명의 값을 한 줄에 세우고, 군을 무시한 채 가장 작은 값부터 1–15위를 매긴 모습입니다(점 위 숫자가 순위). 순위를 매긴 뒤 군별 평균 순위를 구합니다. 세 군의 분포가 같다면 평균 순위는 모두 전체 평균 순위 8 근처여야 합니다."))

# ------------------------------------------------------------------ 4-7 power: KW vs JT vs ANOVA
ps = power_sim()
d = ps["deltas"]
panels = []
for key, title in (("mono", "순서대로 커지는 경우 (평균 0, δ/2, δ)"), ("umbrella", "가운데 군만 높은 경우 (평균 0, δ, 0)")):
    pl = Plot((0, 1.5), (0, 1.0), w=420, h=300, ml=62, mt=30, xlabel="δ (표준편차 단위)", ylabel="검정력",
              xticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=pct)
    pl.hline(0.05, dash=True)
    pl.line(d, ps[key]["anova"], s=4, dash=True, w=1.8)
    pl.line(d, ps[key]["kw"], s=1)
    pl.line(d, ps[key]["jt"], s=2)
    pl.points(d, ps[key]["kw"], s=1, r=2.6)
    pl.points(d, ps[key]["jt"], s=2, r=2.6)
    if key == "mono":
        pl.legend([("Jonckheere-Terpstra", 2, "line"), ("Kruskal-Wallis", 1, "line"), ("분산분석", 4, "dash")], X=62, Y=44)
    else:
        pl.text(0.9, 0.17, "Jonckheere-Terpstra", anchor="start", cls="lbl small")
        pl.text(1.02, ps[key]["kw"][8], "Kruskal-Wallis", anchor="start", dx=4, dy=14, cls="lbl small")
        pl.text(0.97, ps[key]["anova"][8], "분산분석", anchor="end", dx=-4, dy=-6, cls="lbl small")
    pl.text(1.5, 0.05, "α = 5%", anchor="end", dy=-5, cls="lbl mute small")
    panel_title(pl, title)
    panels.append(pl.svg(title))
save("ch04_power", figure(panels,
     "그림 4-7. 정규분포 자료(군당 10명)에서 세 검정이 α = 0.05로 차이를 검출한 비율(모의실험, 조건마다 4,000회). 왼쪽처럼 군의 순서대로 커지는 경우에는 Jonckheere-Terpstra 검정의 검정력이 가장 높지만, 오른쪽처럼 가운데 군만 높은 경우에는 차이가 커져도 거의 검출하지 못합니다.", cols=2))

print("done", [f for f in os.listdir(OUT) if f.startswith("ch04")])
print("fwer", indep, actual)
print("power mono d=1", {k: ps["mono"][k][8] for k in ps["mono"]}, "umb", {k: ps["umbrella"][k][8] for k in ps["umbrella"]})
