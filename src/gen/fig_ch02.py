import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, scipy.stats as st
from itertools import combinations
from html import escape
from svgplot import Plot, figure, panel_title, fmt

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def exact_data(m, s, n, seed):
    z = np.random.default_rng(seed).normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return m + s * z


def swarm(vals, center, binw, step):
    """simple deterministic beeswarm: returns x positions"""
    order = np.argsort(vals)
    xs = np.zeros(len(vals))
    bins = {}
    for i in order:
        b = int(np.floor(vals[i] / binw))
        k = bins.get(b, 0)
        bins[b] = k + 1
        off = ((k + 1) // 2) * step * (1 if k % 2 else -1)
        xs[i] = center + off
    return xs


def errbar(p, x, lo, hi, cls="strongref", w=1.8, cap=0.05):
    p.seg(x, lo, x, hi, cls=cls, w=w)
    p.seg(x - cap, lo, x + cap, lo, cls=cls, w=w)
    p.seg(x - cap, hi, x + cap, hi, cls=cls, w=w)


def mk(p, x, y, n, dx=0, dy=0):
    X, Y = p.sx(x) + dx, p.sy(y) + dy
    p.top.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
                 f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11.5px;font-weight:600">{n}</text>')


# ------------------------------------------------------------------ data (see nums_ch02.py)
m1, s1, m2, s2, n = -0.90, 0.80, -0.40, 0.90, 36
x1 = exact_data(m1, s1, n, 11)
x2 = exact_data(m2, s2, n, 12)
tc35 = st.t.ppf(.975, 35)
ci1 = (m1 - tc35 * s1 / 6, m1 + tc35 * s1 / 6)
ci2 = (m2 - tc35 * s2 / 6, m2 + tc35 * s2 / 6)
print("data range", x1.min(), x1.max(), x2.min(), x2.max())

# ------------------------------------------------------------------ 2-1 strip plot
p = Plot((0.4, 3.45), (-3, 2), w=600, h=360, xlabel="", ylabel="6개월 HbA1c 변화량 (%)",
         yticks=[-3, -2, -1, 0, 1, 2], xticklabels=[(1, "복약상담군 (n = 36)"), (2, "일반 진료군 (n = 36)")],
         xticks=[1, 2], ytickfmt=lambda v: ("+" if v > 0 else "") + fmt(v).replace("-", "−"))
p.hline(0, dash=True)
p.text(0.42, 0, "변화 없음", dy=-6, cls="lbl mute small")
for vals, g, s in ((x1, 1, 1), (x2, 2, 2)):
    xs = swarm(vals, g - 0.06, 0.16, 0.045)
    p.points(xs, vals, s=s, r=3.6)
for g, m, ci in ((1, m1, ci1), (2, m2, ci2)):
    errbar(p, g + 0.3, ci[0], ci[1], cap=0.04)
    p.seg(g + 0.22, m, g + 0.38, m, cls="strongref", w=2.6)
    p.text(g + 0.42, m, f"{m:.2f}".replace("-", "−"), dy=4, cls="lbl strong")
# difference annotation (top-right, empty area)
p.text(2.4, 1.78, "평균 차이 (복약상담 − 일반)", cls="lbl small mute")
p.text(2.4, 1.78, "−0.50%p", dy=20, cls="lbl strong")
p.text(2.4, 1.78, "95% CI −0.90 ~ −0.10", dy=38, cls="lbl small")
p.text(2.4, 1.78, "p = 0.015", dy=54, cls="lbl small")
save("ch02_strip", figure(p.svg("두 군의 HbA1c 변화량 개별값과 평균, 95% 신뢰구간"),
     "그림 2-1. 가상의 약사 복약상담 임상시험(군당 36명)에서 환자 한 명 한 명의 6개월 HbA1c 변화량. 각 군 오른쪽의 굵은 가로선은 평균, 세로 막대는 평균의 95% 신뢰구간입니다. 점들이 위아래로 퍼진 폭과 세로 막대의 길이를 견주어 보세요."))

# ------------------------------------------------------------------ 2-2 SD vs SE vs CI error bars
p = Plot((0.45, 3.55), (-1.8, 0.9), w=600, h=340, ylabel="6개월 HbA1c 변화량 (%)",
         yticks=[-1.5, -1.0, -0.5, 0, 0.5], xticks=[1, 2, 3],
         xticklabels=[(1, "평균 ± SD"), (2, "평균 ± SE"), (3, "평균 ± 95% CI")],
         ytickfmt=lambda v: ("+" if v > 0 else "") + fmt(v, 1).replace("-", "−"))
p.hline(0, dash=True)
bars = {1: ((m1 - s1, m1 + s1), (m2 - s2, m2 + s2)),
        2: ((m1 - s1 / 6, m1 + s1 / 6), (m2 - s2 / 6, m2 + s2 / 6)),
        3: (ci1, ci2)}
for c, (b1, b2) in bars.items():
    errbar(p, c - 0.14, b1[0], b1[1], cls="ln s1", w=2, cap=0.05)
    errbar(p, c + 0.14, b2[0], b2[1], cls="ln s2", w=2, cap=0.05)
    p.points([c - 0.14], [m1], s=1, r=5)
    p.points([c + 0.14], [m2], s=2, r=5)
p.text(1, 0.78, "개인 값의 흩어짐", anchor="middle", dy=4, cls="lbl mute small")
p.text(2, 0.78, "평균의 불확실성", anchor="middle", dy=4, cls="lbl mute small")
p.text(3, 0.78, "SE × 2.03", anchor="middle", dy=4, cls="lbl mute small")
p.legend([("복약상담군", 1, "dot"), ("일반 진료군", 2, "dot")], X=p.sx(1.55), Y=p.sy(-1.42))
save("ch02_errbars", figure(p.svg("같은 자료를 SD, SE, 95% 신뢰구간 오차막대로 그린 비교"),
     "그림 2-2. 그림 2-1과 같은 자료를 세 가지 오차막대로 그렸습니다. 왼쪽부터 평균 ± SD, 평균 ± SE(SD ÷ √36), 평균 ± 95% 신뢰구간(SE × 2.03)입니다. 자료도 두 군의 차이도 세 묶음에서 같으니, 막대의 길이와 두 군의 막대가 겹치는지가 묶음마다 어떻게 달라지는지 보세요."))

# ------------------------------------------------------------------ 2-3 simulated type I error
rng = np.random.default_rng(2024)
R = 200000


def sim(n1, sd1, n2, sd2):
    x = rng.normal(0, sd1, (R, n1)); y = rng.normal(0, sd2, (R, n2))
    a_, b_ = x.mean(1), y.mean(1); v1, v2 = x.var(1, ddof=1), y.var(1, ddof=1)
    df = n1 + n2 - 2; sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
    ps = 2 * st.t.sf(abs((a_ - b_) / np.sqrt(sp2 * (1 / n1 + 1 / n2))), df)
    u, w = v1 / n1, v2 / n2
    dfw = (u + w) ** 2 / (u ** 2 / (n1 - 1) + w ** 2 / (n2 - 1))
    pw = 2 * st.t.sf(abs((a_ - b_) / np.sqrt(u + w)), dfw)
    return 100 * (ps < .05).mean(), 100 * (pw < .05).mean()


SIM = [(36, 1, 36, 1), (36, 1, 36, 2), (40, 1, 15, 1), (40, 1.2, 15, 0.6), (40, 0.6, 15, 1.2)]
LAB = [("36 : 36", "1 : 1"), ("36 : 36", "1 : 2"), ("40 : 15", "1 : 1"), ("40 : 15", "2 : 1"), ("40 : 15", "1 : 2")]
res = [sim(*sc) for sc in SIM]
print("type I error %", [tuple(round(v, 2) for v in r) for r in res])
p = Plot((0.4, 5.6), (0, 16), w=600, h=350, mb=84, ylabel="실제 1종 오류율 (%)",
         yticks=[0, 4, 8, 12, 16], xticks=[1, 2, 3, 4, 5], xticklabels=[(i, "") for i in range(1, 6)],
         xlabel="")
p.hline(5, dash=True)
for i, (rs, rw) in enumerate(res):
    g = i + 1
    p.bars([g - 0.16], [rs], 0.28, s=2)
    p.bars([g + 0.16], [rw], 0.28, s=1)
    p.text(g - 0.16, rs, f"{rs:.1f}", anchor="middle", dy=-5, cls="lbl small")
    p.text(g + 0.16, rw, f"{rw:.1f}", anchor="middle", dy=-5, cls="lbl small")
    p.text(g, 0, "n " + LAB[i][0], anchor="middle", dy=20)
    p.text(g, 0, "SD " + LAB[i][1], anchor="middle", dy=37, cls="lbl mute")
p.text_px(p.ml - 8, p.h - p.mb + 20, "", anchor="end")
p.text_px((p.ml + p.w - p.mr) / 2, p.h - 12, "시나리오 (두 군의 표본수 비, 표준편차 비)", anchor="middle", cls="axlab")
p.legend([("Student t 검정", 2, "box"), ("Welch t 검정", 1, "box"), ("명목 유의수준 5%", 4, "dash")])
save("ch02_typeI", figure(p.svg("시나리오별 Student와 Welch t 검정의 실제 1종 오류율"),
     "그림 2-3. 두 군의 모평균이 실제로 같은 상황(귀무가설이 참)에서 시나리오마다 20만 번씩 시뮬레이션해, p &lt; 0.05가 나온 비율을 센 결과입니다. 점선(명목 유의수준 5%)에서 막대가 얼마나 벗어나는지 보세요. 두 군의 크기가 같거나 SD가 같은 왼쪽 세 시나리오에서는 Student t 검정도 5% 근처이고, 크기와 SD가 모두 다른 오른쪽 두 시나리오에서 Student t 검정만 크게 벗어납니다."))

# ------------------------------------------------------------------ 2-4 paper-style bar chart with 95% CI
p = Plot((0.3, 2.7), (-1.4, 0.35), w=520, h=330, ml=66, ylabel="Change in HbA1c (%)", ygrid=False,
         yticks=[-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2], xticks=[1, 2],
         xticklabels=[(1, "Pharmacist counseling"), (2, "Usual care")],
         ytickfmt=lambda v: fmt(v, 1).replace("-", "−"))
p.els.append(f'<line x1="{p.ml}" y1="{p.mt}" x2="{p.ml}" y2="{p.h - p.mb}" class="axis"/>')
p.hline(0, dash=False, cls="axis")
p.bars([1], [m1], 0.5, s=1, rounded=False, cls="a1")
p.bars([2], [m2], 0.5, s=2, rounded=False, cls="a2")
for g, m, ci, s in ((1, m1, ci1, 1), (2, m2, ci2, 2)):
    p.seg(g - 0.25, m, g + 0.25, m, cls=f"ln s{s}", w=2)
    errbar(p, g, ci[0], ci[1], cls="strongref", w=1.6, cap=0.07)
# significance bracket
p.seg(1, 0.14, 1, 0.2, cls="strongref", w=1.2)
p.seg(2, 0.14, 2, 0.2, cls="strongref", w=1.2)
p.seg(1, 0.2, 2, 0.2, cls="strongref", w=1.2)
p.text(1.5, 0.2, "P = 0.015", anchor="middle", dy=-6, cls="lbl")
mk(p, 1, ci1[0], 1, dx=-22, dy=6)
mk(p, 1.5, (ci1[1] + ci2[0]) / 2, 2)
mk(p, 1.5, 0.2, 3, dx=52, dy=-10)
save("ch02_paperfig", figure(p.svg("가상 논문의 막대그래프: 평균 HbA1c 변화와 95% 신뢰구간"),
     "그림 2-4. 가상의 논문 그림. <i>Figure 2. Mean change in HbA1c from baseline to 6 months. Error bars indicate 95% confidence intervals.</i>"))

# ------------------------------------------------------------------ 2-5 values -> ranks
P = np.array([12, 15, 18, 20, 22, 26, 31, 68.])
U = np.array([24, 29, 36, 41, 45, 58, 115.])
allv = np.concatenate([P, U]); grp = np.array([1] * 8 + [2] * 7)
order = np.argsort(allv, kind="stable")
rank = st.rankdata(allv)
W, H = 640, 292
X0, X1 = 92, 620
YV, YR = 92, 214


def vx(v):
    return X0 + v / 120 * (X1 - X0)


def rx(r):
    return X0 + (r - 1) / 14 * (X1 - X0)


o = []
o.append(f'<line x1="{X0}" y1="{YV}" x2="{X1}" y2="{YV}" class="axis"/>')
for t in range(0, 121, 20):
    o.append(f'<line x1="{vx(t):.1f}" y1="{YV}" x2="{vx(t):.1f}" y2="{YV + 4}" class="axis"/>')
    o.append(f'<text x="{vx(t):.1f}" y="{YV + 18}" text-anchor="middle" class="tick">{t}</text>')
o.append(f'<text x="8" y="{YV + 4}" class="lbl strong">실제 값</text>')
o.append(f'<text x="8" y="{YV + 20}" class="lbl small mute">(MME)</text>')
o.append(f'<text x="8" y="{YR + 4}" class="lbl strong">순위</text>')
for i in range(len(allv)):
    s = grp[i]
    o.append(f'<line x1="{vx(allv[i]):.1f}" y1="{YV - 6}" x2="{vx(allv[i]):.1f}" y2="{YV - 6}" class="ref"/>')
    o.append(f'<path d="M{vx(allv[i]):.1f},{YV + 24} L{rx(rank[i]):.1f},{YR - 14}" class="ref" stroke-width="0.9" fill="none"/>')
for i in range(len(allv)):
    s = grp[i]
    cy = YV - 12 if s == 1 else YV - 26
    o.append(f'<line x1="{vx(allv[i]):.1f}" y1="{cy}" x2="{vx(allv[i]):.1f}" y2="{YV}" class="ln s{s}" stroke-width="1.2"/>')
    o.append(f'<circle cx="{vx(allv[i]):.1f}" cy="{cy}" r="4.5" class="pt f{s}"/>')
    o.append(f'<circle cx="{rx(rank[i]):.1f}" cy="{YR}" r="12" class="f{s}"/>')
    o.append(f'<text x="{rx(rank[i]):.1f}" y="{YR + 4.5}" text-anchor="middle" style="fill:#fff;font-size:12px;font-weight:600">{int(rank[i])}</text>')
    o.append(f'<text x="{rx(rank[i]):.1f}" y="{YR + 30}" text-anchor="middle" class="tick">{int(allv[i])}</text>')
o.append(f'<text x="{X0}" y="{YR + 52}" class="lbl small mute">원 아래 숫자 = 실제 값(MME)</text>')
# legend
o.append(f'<circle cx="{X0 + 6}" cy="20" r="5" class="f1"/><text x="{X0 + 16}" y="24.5" class="lbl">프로토콜군 (n = 8) · 순위합 45</text>')
o.append(f'<circle cx="{X0 + 276}" cy="20" r="5" class="f2"/><text x="{X0 + 286}" y="24.5" class="lbl">일반 관리군 (n = 7) · 순위합 75</text>')
svg = (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="실제 값과 순위의 대응" '
       f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')
save("ch02_ranks", figure(svg,
     "그림 2-5. 15명의 72시간 오피오이드 사용량(위)을 크기순으로 한 줄로 세워 순위(아래)로 바꾼 모습. 위쪽에서 다른 점들과 멀리 떨어진 맨 오른쪽 값이 아래쪽 순위에서는 어디에 놓이는지 보세요."))

# ------------------------------------------------------------------ 2-6 pair grid (U counting)
Ps, Us = np.sort(P), np.sort(U)
cs = 34
GX, GY = 178, 78
W, H = 640, GY + 8 * cs + 92
o = []
o.append(f'<text x="{GX + 7 * cs / 2:.1f}" y="22" text-anchor="middle" class="lbl strong">일반 관리군 환자의 값 (MME)</text>')
o.append(f'<text x="{GX - 12}" y="{GY - 12}" text-anchor="end" class="lbl strong">프로토콜군 값</text>')
o.append(f'<text x="{GX - 12}" y="{GY + 4}" text-anchor="end" class="lbl small mute">(MME)</text>')
for j, u in enumerate(Us):
    o.append(f'<text x="{GX + j * cs + cs / 2:.1f}" y="{GY - 12}" text-anchor="middle" class="tick">{int(u)}</text>')
cnt_row = []
for i, pv in enumerate(Ps):
    y = GY + i * cs
    o.append(f'<text x="{GX - 12}" y="{y + cs / 2 + 4.5:.1f}" text-anchor="end" class="tick">{int(pv)}</text>')
    k = 0
    for j, u in enumerate(Us):
        x = GX + j * cs
        big = pv > u
        k += big
        cls = "f1" if big else "a2"
        o.append(f'<rect x="{x + 1.5:.1f}" y="{y + 1.5:.1f}" width="{cs - 3}" height="{cs - 3}" rx="3" class="{cls}"/>')
        if big:
            o.append(f'<text x="{x + cs / 2:.1f}" y="{y + cs / 2 + 4.5:.1f}" text-anchor="middle" style="fill:#fff;font-size:12px;font-weight:600">1</text>')
        else:
            o.append(f'<text x="{x + cs / 2:.1f}" y="{y + cs / 2 + 4.5:.1f}" text-anchor="middle" class="lbl small mute">0</text>')
    cnt_row.append(k)
    o.append(f'<text x="{GX + 7 * cs + 34:.1f}" y="{y + cs / 2 + 4.5:.1f}" text-anchor="middle" class="lbl num">{k}</text>')
o.append(f'<text x="{GX + 7 * cs + 34:.1f}" y="{GY - 12}" text-anchor="middle" class="lbl small mute">행 합</text>')
yb = GY + 8 * cs
o.append(f'<line x1="{GX + 7 * cs + 16:.1f}" y1="{yb + 4}" x2="{GX + 7 * cs + 52:.1f}" y2="{yb + 4}" class="axis"/>')
o.append(f'<text x="{GX + 7 * cs + 34:.1f}" y="{yb + 22}" text-anchor="middle" class="lbl strong">9</text>')
o.append(f'<text x="{GX + 7 * cs + 58:.1f}" y="{yb + 22}" class="lbl strong">= U₁</text>')
o.append(f'<rect x="{GX}" y="{yb + 40}" width="14" height="14" rx="3" class="f1"/><text x="{GX + 22}" y="{yb + 51.5}" class="lbl">프로토콜군 값이 더 큼: 9쌍</text>')
o.append(f'<rect x="{GX}" y="{yb + 62}" width="14" height="14" rx="3" class="a2"/><text x="{GX + 22}" y="{yb + 73.5}" class="lbl">일반 관리군 값이 더 큼: 47쌍</text>')
svg = (f'<svg viewBox="0 0 {W} {H}" class="viz" role="img" aria-label="두 군 환자를 한 명씩 짝지은 56쌍의 비교" '
       f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')
save("ch02_pairs", figure(svg,
     "그림 2-6. 프로토콜군 8명(행)과 일반 관리군 7명(열)을 한 명씩 짝지은 56쌍입니다. 프로토콜군 값이 더 큰 칸에 1, 그렇지 않은 칸에 0을 적었습니다. 오른쪽 행 합은 그 프로토콜군 환자가 일반 관리군 몇 명보다 많이 썼는지이고, 행 합을 모두 더한 것이 U₁입니다."))

# ------------------------------------------------------------------ 2-7 exact null distribution of U
cnts = np.zeros(57, int)
for comb in combinations(range(1, 16), 8):
    cnts[sum(comb) - 36] += 1
prob = cnts / cnts.sum()
print("max prob", prob.max(), "tail", prob[:10].sum())
p = Plot((-1, 57), (0, 0.054), w=600, h=320, xlabel="U₁ (프로토콜군 값이 더 큰 쌍의 수)", ylabel="확률",
         yticks=[0, 0.01, 0.02, 0.03, 0.04, 0.05], xticks=[0, 9, 28, 47, 56],
         ytickfmt=lambda v: fmt(v, 2))
ks = np.arange(57)
mid = (ks > 9) & (ks < 47)
p.bars(ks[mid], prob[mid], 0.8, s=4, rounded=False, cls="a4")
p.bars(ks[~mid], prob[~mid], 0.8, s=2, rounded=False)
p.vline(9, y1=0.03)
p.text(9, 0.03, "관측 U₁ = 9", anchor="middle", dy=-6, cls="lbl strong")
p.vline(47, y1=0.03)
p.text(47, 0.03, "U₁ = 47 (반대쪽)", anchor="middle", dy=-6, cls="lbl")
p.text(4.5, 0.012, "0.0145", anchor="middle", cls="lbl small")
p.text(51.5, 0.012, "0.0145", anchor="middle", cls="lbl small")
p.text(28, 0.0475, "기댓값 28 (= 56 ÷ 2)", anchor="middle", cls="lbl mute small")
save("ch02_unull", figure(p.svg("귀무가설에서 U의 정확한 분포"),
     "그림 2-7. 두 군에 차이가 없을 때, 순위 1–15를 두 군에 나누는 방식마다 U₁을 세어 만든 정확한 분포입니다. 양 끝의 색칠한 막대가 관측값 9 이하와 반대쪽 47 이상인 부분이고, 두 부분의 확률을 더한 것이 양측 정확 p입니다."))
