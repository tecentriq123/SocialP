import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def pct(v, nd=0):
    return f"{v * 100:.{nd}f}%"


def _tw(s, fs):
    w = 0.0
    for c in s:
        if "\uac00" <= c <= "\ud7a3":
            w += fs * 1.0
        elif c == " ":
            w += fs * 0.3
        elif c in "%W":
            w += fs * 0.85
        else:
            w += fs * 0.62
    return w


def halo(p, x, y, s, anchor="middle", cls="lbl", dx=0, dy=0, fs=13):
    """text with a surface-colored box behind it, so lines passing under it stay readable"""
    w = _tw(s, fs)
    X, Y = p.sx(x) + dx, p.sy(y) + dy
    x0 = X - w / 2 if anchor == "middle" else (X - w if anchor == "end" else X)
    p.top.append(f'<rect x="{x0 - 3:.1f}" y="{Y - fs + 1:.1f}" width="{w + 6:.1f}" height="{fs + 4:.1f}" rx="3" class="pth"/>')
    p.text(x, y, s, anchor=anchor, cls=cls, dx=dx, dy=dy)


# ------------------------------------------------------------------ 5-1 observed vs expected, cell contributions
T = np.array([[48, 352], [102, 498]], float)
N = T.sum()
E = np.outer(T.sum(1), T.sum(0)) / N
contrib = (T - E) ** 2 / E
chi = contrib.sum()
pbar = T[:, 0].sum() / N

p1 = Plot((0.4, 2.9), (0, 0.25), w=420, h=300, ml=62, mr=16, xlabel="", ylabel="1년 내 입원 비율",
          yticks=[0, 0.05, 0.10, 0.15, 0.20, 0.25], ytickfmt=lambda v: pct(v),
          xticks=[1, 2], xticklabels=[(1, "약물 A (n = 400)"), (2, "약물 B (n = 600)")])
p1.bars([1], [T[0, 0] / 400], 0.44, s=1)
p1.bars([2], [T[1, 0] / 600], 0.44, s=2)
p1.hline(pbar, dash=True, cls="ref strongref", w=1.4)
p1.text(2.88, pbar, "전체 15.0%", anchor="end", dy=-6, cls="lbl mute small")
p1.text(1, T[0, 0] / 400, "12.0%", anchor="middle", dy=-8, cls="lbl strong")
p1.text(2, T[1, 0] / 600, "17.0%", anchor="middle", dy=-8, cls="lbl strong")
p1.text(1.26, 0.075, "관측 48명", cls="lbl small")
p1.text(1.26, 0.075, "기대 60명", dy=16, cls="lbl mute small")
p1.text(2.26, 0.075, "관측 102명", cls="lbl small")
p1.text(2.26, 0.075, "기대 90명", dy=16, cls="lbl mute small")
panel_title(p1, "관측 비율과 기대 비율")

labs = ["A · 입원", "B · 입원", "A · 비입원", "B · 비입원"]
vals = [contrib[0, 0], contrib[1, 0], contrib[0, 1], contrib[1, 1]]
Os = [T[0, 0], T[1, 0], T[0, 1], T[1, 1]]
Es = [E[0, 0], E[1, 0], E[0, 1], E[1, 1]]
p2 = Plot((0.4, 4.6), (0, 3.2), w=420, h=300, ml=44, mr=12, ylabel="(O − E)² ÷ E",
          yticks=[0, 1, 2, 3], xticks=[1, 2, 3, 4], xticklabels=list(zip([1, 2, 3, 4], labs)))
p2.bars([1, 2, 3, 4], vals, 0.56, s=1)
for i, (v, o, e) in enumerate(zip(vals, Os, Es)):
    d = int(round(o - e))
    sgn = "−" if d < 0 else "+"
    p2.text(i + 1, v, f"{v:.2f}", anchor="middle", dy=-24, cls="lbl strong")
    p2.text(i + 1, v, f"({sgn}{abs(d)})² ÷ {e:g}", anchor="middle", dy=-9, cls="lbl mute small")
p2.text(4.55, 2.95, f"합계 χ² = {chi:.2f}", anchor="end", cls="lbl strong")
p2.text(4.55, 2.95, "자유도 1, p = 0.030", anchor="end", dy=17, cls="lbl mute small")
panel_title(p2, "칸별 기여도")
save("ch05_obs_exp", figure([p1.svg("군별 입원 비율과 기대 비율"), p2.svg("칸별 카이제곱 기여도")],
     "그림 5-1. 왼쪽: 약물과 입원이 관련이 없다면 두 군 모두 전체 입원율 15.0%만큼 입원해야 합니다(점선). "
     "오른쪽: 네 칸의 관측값과 기대값의 차이는 모두 12명이지만, 기대빈도가 작은 '입원' 칸의 기여가 훨씬 큽니다. "
     "네 기여도를 더한 값이 카이제곱 통계량입니다.", cols=2))

# ------------------------------------------------------------------ 5-2 OR vs RR as baseline risk rises
L = np.log
p = Plot((0, 0.40), (L(0.25), L(8)), w=600, h=340, ml=66, mr=24, xlabel="대조군의 사건 위험 (p₀)",
         ylabel="오즈비 (로그 눈금)", xticks=[0, 0.1, 0.2, 0.3, 0.4], xtickfmt=lambda v: pct(v),
         yticks=[L(v) for v in (0.25, 0.5, 1, 2, 4, 8)],
         ytickfmt=lambda v: fmt(round(np.exp(v), 3)))
x = np.linspace(0.002, 0.40, 300)


def orf(rr, p0):
    p1_ = rr * p0
    return (p1_ / (1 - p1_)) / (p0 / (1 - p0))


p.hline(L(1), dash=False, cls="ref", w=1)
p.hline(L(2), dash=True, cls="ln s4", w=1.4)
p.hline(L(0.5), dash=True, cls="ln s4", w=1.4)
p.line(x, L(orf(2, x)), s=2, w=2.4)
p.line(x, L(orf(0.5, x)), s=1, w=2.4)
for p0 in (0.10, 0.30, 0.40):
    v = orf(2, p0)
    p.points([p0], [L(v)], s=2, r=4.5)
    p.text(p0, L(v), f"OR {v:.2f}", anchor="end", dx=-8, dy=-6, cls="lbl strong")
for p0 in (0.30,):
    v = orf(0.5, p0)
    p.points([p0], [L(v)], s=1, r=4.5)
    p.text(p0, L(v), f"OR {v:.2f}", anchor="middle", dy=20, cls="lbl strong")
p.text(0.398, L(2), "실제 RR = 2", anchor="end", dy=15, cls="lbl mute small")
p.text(0.398, L(0.5), "실제 RR = 0.5", anchor="end", dy=-7, cls="lbl mute small")
p.text(0.2, L(orf(2, 0.2)), "RR = 2일 때 계산되는 OR", anchor="end", dx=-10, dy=-12, cls="lbl")
p.text(0.2, L(orf(0.5, 0.2)), "RR = 0.5일 때 계산되는 OR", anchor="end", dx=-6, dy=20, cls="lbl")
p.text(0.398, L(1), "OR = 1 (차이 없음)", anchor="end", dy=-6, cls="lbl mute small")
save("ch05_or_rr", figure(p.svg("대조군 위험에 따른 오즈비와 상대위험도의 차이"),
     "그림 5-2. 상대위험도(RR)를 2 또는 0.5로 고정하고 대조군의 사건 위험만 바꿨을 때 같은 자료에서 계산되는 오즈비(OR). "
     "사건이 드물면(왼쪽 끝) OR과 RR이 거의 같지만, 사건이 흔해질수록 OR은 RR보다 1에서 더 멀어집니다. "
     "RR = 2인데 p₀가 30%이면 OR은 3.50입니다."))

# ------------------------------------------------------------------ 5-3 3x2 table: proportions + adjusted residuals
T3 = np.array([[12, 288], [15, 285], [33, 267]], float)
N3 = T3.sum()
E3 = np.outer(T3.sum(1), T3.sum(0)) / N3
rp = T3.sum(1, keepdims=True) / N3
cp = T3.sum(0, keepdims=True) / N3
R3 = (T3 - E3) / np.sqrt(E3 * (1 - rp) * (1 - cp))
names = ["SGLT2 억제제", "DPP-4 억제제", "설포닐우레아"]
props = T3[:, 0] / 300
p1 = Plot((0.4, 3.6), (0, 0.14), w=420, h=300, ml=62, mr=14, ylabel="1년 내 저혈당 비율",
          yticks=[0, 0.04, 0.08, 0.12], ytickfmt=lambda v: pct(v),
          xticks=[1, 2, 3], xticklabels=list(zip([1, 2, 3], names)))
p1.bars([1, 2], props[:2], 0.5, s=1)
p1.bars([3], props[2:], 0.5, s=2)
ov = T3[:, 0].sum() / N3
p1.hline(ov, dash=True, cls="ref strongref", w=1.4)
p1.text(0.45, ov, f"전체 {ov * 100:.1f}%", dy=-6, cls="lbl mute small")
for i, (pr, o) in enumerate(zip(props, T3[:, 0])):
    halo(p1, i + 1, pr, f"{pr * 100:.1f}%", anchor="middle", dy=-22, cls="lbl strong")
    halo(p1, i + 1, pr, f"{int(o)}/300", anchor="middle", dy=-8, cls="lbl mute small", fs=11.5)
panel_title(p1, "군별 저혈당 비율")

p2 = Plot((0.4, 3.6), (-3.2, 4.4), w=420, h=300, ml=44, mr=14, ylabel="수정 표준화 잔차",
          yticks=[-3, -2, -1, 0, 1, 2, 3, 4], xticks=[1, 2, 3], xticklabels=list(zip([1, 2, 3], names)))
res = R3[:, 0]
for i, v in enumerate(res):
    p2.bars([i + 1], [v], 0.5, s=(2 if v > 0 else 1), rounded=False)
p2.hline(0, dash=False, cls="axis", w=1)
for v in (1.96, -1.96):
    p2.hline(v, dash=True, cls="ref", w=1.1)
for v in (2.39, -2.39):
    p2.hline(v, dash=True, cls="ref strongref", w=1.1)
p2.text(3.58, 1.96, "±1.96", anchor="end", dy=13, cls="lbl mute small")
p2.text(3.58, -2.39, "±2.39 (Bonferroni)", anchor="end", dy=13, cls="lbl mute small")
for i, v in enumerate(res):
    lab = f"{v:+.2f}".replace("-", "−")
    if v > 0:
        p2.text(i + 1, v, lab, anchor="middle", dy=-7, cls="lbl strong")
    else:
        p2.text(i + 1, 0, lab, anchor="middle", dy=-7, cls="lbl strong")
panel_title(p2, "'저혈당' 칸의 수정 표준화 잔차")
save("ch05_resid", figure([p1.svg("약물 계열별 저혈당 비율"), p2.svg("수정 표준화 잔차")],
     "그림 5-3. 왼쪽: 세 군의 저혈당 비율과 전체 비율(점선). 오른쪽: '저혈당' 칸의 수정 표준화 잔차. "
     "0보다 크면 기대보다 많이, 작으면 적게 관측되었다는 뜻이며, 절댓값이 1.96(세 군을 함께 볼 때 Bonferroni 기준 2.39)을 넘는 칸을 차이의 출처로 봅니다.",
     cols=2))

# ------------------------------------------------------------------ 5-4 Fisher: all possible tables
m_, n_, k_ = 15, 14, 7
a = np.arange(0, 8)
pr = st.hypergeom.pmf(a, m_ + n_, m_, k_)
pobs = pr[1]
inc = pr <= pobs * (1 + 1e-7)
p = Plot((-0.6, 7.6), (0, 0.50), w=600, h=360, ml=62, mr=20, mb=84, xlabel="약물 X군(15명) 중 간독성 환자 수 a",
         ylabel="그 표가 나올 확률", yticks=[0, 0.1, 0.2, 0.3, 0.4], xticks=list(range(8)))
for ai, v, ok in zip(a, pr, inc):
    p.bars([ai], [v], 0.62, s=(2 if ok else 4))
p.hline(pobs, dash=True, cls="ref strongref", w=1.2)
B = p.h - p.mb
p.text_px(p.ml - 8, B + 38, "확률", anchor="end", cls="lbl small")
for ai, v, ok in zip(a, pr, inc):
    p.text_px(p.sx(ai), B + 38, f"{v:.4f}", anchor="middle", cls=("lbl strong small" if ok else "lbl mute small"))
p.text(1, pobs, "관측", anchor="middle", dy=-8, cls="lbl strong")
p.text(-0.45, 0.49, "주황 막대 = 관측된 표보다 확률이 같거나 작은 표", cls="lbl")
p.text(-0.45, 0.49, f"점선 = 관측된 표(a = 1)의 확률 {pobs:.4f}", dy=19, cls="lbl mute")
p.text(-0.45, 0.49, f"양측 p = {pr[0]:.4f} + {pr[1]:.4f} + {pr[7]:.4f} = {pr[inc].sum():.4f}", dy=40, cls="lbl strong")
save("ch05_fisher", figure(p.svg("여백을 고정했을 때 가능한 모든 표의 확률"),
     "그림 5-4. 행 합계(15명, 14명)와 열 합계(간독성 7명, 없음 22명)를 고정하면 가능한 표는 a = 0~7의 8개뿐이고, 각 표의 확률은 초기하분포로 정해집니다. "
     "관측된 표(a = 1)보다 확률이 같거나 작은 표(주황)의 확률을 모두 더한 값이 양측 p-value입니다. a = 6인 표(0.0449)는 관측된 표보다 확률이 커서 더하지 않습니다."))


# ------------------------------------------------------------------ 5-5 trend: monotone vs inverted-U
def wls_line(events, ns, scores):
    events, ns, scores = map(lambda v: np.asarray(v, float), (events, ns, scores))
    pr_ = events / ns
    sbar = (ns * scores).sum() / ns.sum()
    pbar_ = events.sum() / ns.sum()
    b = (ns * (scores - sbar) * (pr_ - pbar_)).sum() / (ns * (scores - sbar) ** 2).sum()
    return pbar_ - b * sbar, b


def trend(events, ns, scores):
    events, ns, scores = map(lambda v: np.asarray(v, float), (events, ns, scores))
    s = np.repeat(scores, ns.astype(int))
    y = np.concatenate([np.r_[np.ones(int(e)), np.zeros(int(n - e))] for e, n in zip(events, ns)])
    r = np.corrcoef(s, y)[0, 1]
    M2 = (ns.sum() - 1) * r ** 2
    tab = np.column_stack([events, ns - events])
    Ex = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
    chi_ = ((tab - Ex) ** 2 / Ex).sum()
    return chi_, st.chi2.sf(chi_, 2), M2, st.chi2.sf(M2, 1)


def pfmt(pv):
    return f"{pv:.3f}" if pv < 0.1 else f"{pv:.2f}"


cats = ["< 40%", "40–79%", "≥ 80%"]
ns = [200, 300, 500]
panels = []
for ev, title, sb in (([30, 36, 40], "한 방향으로 감소하는 경우", 1), ([16, 45, 45], "가운데가 높은 경우", 3)):
    pr_ = np.array(ev) / np.array(ns)
    c, pc, M2, pm = trend(ev, ns, [1, 2, 3])
    a0, b0 = wls_line(ev, ns, [1, 2, 3])
    pl = Plot((0.4, 3.6), (0, 0.24), w=420, h=300, ml=62, mr=14, ylabel="다음 해 입원 비율",
              yticks=[0, 0.05, 0.10, 0.15, 0.20], ytickfmt=lambda v: pct(v),
              xticks=[1, 2, 3], xticklabels=list(zip([1, 2, 3], cats)), xlabel="PDC 범주")
    pl.bars([1, 2, 3], pr_, 0.5, s=sb)
    xx = np.array([0.7, 3.3])
    pl.line(xx, a0 + b0 * xx, s=2 if sb != 2 else 1, dash=True, w=2)
    for i, v in enumerate(pr_):
        halo(pl, i + 1, v, f"{v * 100:.1f}%", anchor="middle", dy=-8, cls="lbl strong")
    pl.text(0.5, 0.225, f"Pearson χ²(2) = {c:.2f}, P = {pfmt(pc)}", cls="lbl")
    pl.text(0.5, 0.225, f"추세 M²(1) = {M2:.2f}, P = {pfmt(pm)}", dy=18, cls="lbl strong")
    panel_title(pl, title)
    panels.append(pl.svg(title))
save("ch05_trend", figure(panels,
     "그림 5-5. PDC 범주별 입원 비율(막대)과 범주 점수 1, 2, 3에 맞춘 가중 직선(주황 점선). "
     "왼쪽처럼 비율이 한 방향으로 변하면 선형 대 선형 결합(M²)이 Pearson χ²의 차이를 거의 전부 자유도 1개로 잡아내 p가 더 작아집니다. "
     "오른쪽처럼 가운데가 높은 모양이면 Pearson χ²는 유의하지만 직선 추세는 거의 없습니다.", cols=2))
print("figures written")
print("wls monotone", wls_line([30, 36, 40], ns, [1, 2, 3]), "wls U", wls_line([16, 45, 45], ns, [1, 2, 3]))
print("trend U", trend([16, 45, 45], ns, [1, 2, 3]))
