"""Figures for chapter 19 (메타분석).
run: source /home/claude/pylibs/env.sh && python3 gen/fig_ch19.py   (imports gen/nums_ch19.compute)"""
import math, os, sys
from html import escape
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
import scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt
from nums_ch19 import compute

OUT = os.path.join(HERE, "..", "figs")
R = compute()
T = R["T"]
Z = 1.959963984540054
L = math.log


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def badge(X, Y, n):
    return (f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
            f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def tw(s, size=13):
    """rough text width in viewBox units"""
    return sum(size * (0.95 if ord(ch) > 0x2000 else 0.56) for ch in s)


# ====================================================================== 19-1 PRISMA flow diagram (paper box, 가)
def prisma():
    w, h = 720, 448
    o = []
    bx, bw = 14, 372          # main column
    ex, ew = 420, 286         # exclusion column

    def box(x, y, wd, ht, lines, strong_first=False, cls="pth s4"):
        o.append(f'<rect x="{x}" y="{y}" width="{wd}" height="{ht}" rx="6" class="{cls}"/>')
        for i, ln in enumerate(lines):
            c = "lbl strong" if (strong_first and i == 0) else ("lbl" if i == 0 else "lbl small")
            o.append(f'<text x="{x + 12}" y="{y + 20 + i * 17}" class="{c}">{escape(ln)}</text>')

    def arrow_down(x, y0, y1):
        o.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y1 - 6}" class="ref" stroke-width="1.4"/>')
        o.append(f'<polygon points="{x - 5},{y1 - 7} {x + 5},{y1 - 7} {x},{y1}" class="f4"/>')

    def arrow_right(x0, x1, y):
        o.append(f'<line x1="{x0}" y1="{y}" x2="{x1 - 6}" y2="{y}" class="ref" stroke-width="1.4"/>')
        o.append(f'<polygon points="{x1 - 7},{y - 5} {x1 - 7},{y + 5} {x1},{y}" class="f4"/>')

    cx = bx + bw / 2
    # row 1
    box(bx, 14, bw, 62, ["1,248 Records identified through database searching",
                         "PubMed 412, Embase 655, CENTRAL 181"])
    arrow_down(cx, 76, 112); arrow_right(cx, ex, 94)
    box(ex, 76, ew, 36, ["316 Duplicates removed"])
    # row 2
    box(bx, 112, bw, 44, ["932 Records screened (title and abstract)"])
    arrow_down(cx, 156, 196); arrow_right(cx, ex, 176)
    box(ex, 158, ew, 36, ["884 Records excluded as not relevant"])
    # row 3
    box(bx, 196, bw, 44, ["48 Full-text articles assessed for eligibility"])
    arrow_down(cx, 240, 376); arrow_right(cx, ex, 300)
    box(ex, 240, ew, 124, ["38 Full-text articles excluded",
                           "14 Secondary reports of an included trial",
                           "  9 Not randomized",
                           "  8 Comparator not placebo or standard care",
                           "  4 Outcomes of interest not reported",
                           "  3 Ongoing, no results available"])
    # row 4
    box(bx, 376, bw, 58, ["10 Randomized trials included in the", "meta-analysis (31,725 participants)"], cls="a1 s1")
    o.append(f'<text x="{bx + 12}" y="{376 + 37}" class="lbl">{escape("meta-analysis (31,725 participants)")}</text>')
    o.append(badge(bx + bw, 14, 1))
    o.append(badge(ex + ew, 158, 2))
    o.append(badge(ex + ew, 240, 3))
    o.append(badge(bx + bw, 376, 4))
    return (f'<svg viewBox="0 0 {w} {h}" class="viz" role="img" aria-label="PRISMA flow diagram" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')


svg = prisma()
# the second line of the last box was drawn twice (small + normal); keep the normal one only
svg = svg.replace('<text x="26" y="413" class="lbl small">meta-analysis (31,725 participants)</text>', '')
save("ch19_prisma", figure(
    [svg],
    "그림 19-1. 가상의 체계적 문헌고찰의 문헌 선정 흐름도(PRISMA flow diagram). 위에서 아래로 남은 문헌 수를, 오른쪽에 단계마다 빠진 문헌 수와 이유를 적습니다. "
    "번호는 아래 해설과 짝을 이룹니다.", cols=1))


# ====================================================================== 19-2 fixed-effect vs random-effects (concept)
def concept():
    xt = [L(0.5), L(0.7), L(1.0), L(1.4)]
    xl = [(v, fmt(round(math.exp(v), 1))) for v in xt]
    mu = L(0.82)
    xlim = (L(0.36), L(2.1))
    kw = dict(w=420, h=310, ml=16, mr=16, mt=26, mb=48, xlabel="오즈비 (왼쪽일수록 약이 유리)", xticks=xt, xticklabels=xl, ygrid=False, show_yaxis=False)

    def study(p, yy, e, se):
        p.seg(e - Z * se, yy, e + Z * se, yy, cls="ln s1", w=2)
        sz = 3 + 0.06 / se * 4.5
        p.top.append(f'<rect x="{p.sx(e) - sz:.1f}" y="{p.sy(yy) - sz:.1f}" width="{2 * sz:.1f}" height="{2 * sz:.1f}" class="f1"/>')

    p1 = Plot(xlim, (0, 7.2), **kw)
    p1.vline(mu, dash=False, cls="strongref", w=1.6, y0=0.4, y1=5.9)
    p1.text(mu, 6.3, "모든 연구에 공통인 참값 θ", anchor="middle", cls="lbl strong small")
    for yy, e, se in [(5.1, mu + 0.17, 0.22), (4.2, mu - 0.05, 0.10), (3.3, mu - 0.24, 0.24), (2.4, mu + 0.04, 0.06), (1.5, mu + 0.12, 0.15)]:
        study(p1, yy, e, se)
    p1.text(L(1.12), 3.25, "추정값이 서로 다른", cls="lbl small")
    p1.text(L(1.12), 2.75, "이유는 표본오차뿐", cls="lbl small")
    panel_title(p1, "(가) 고정효과 모형")

    p2 = Plot(xlim, (0, 7.2), **kw)
    tau = 0.15
    xs = np.linspace(mu - 3.2 * tau, mu + 3.2 * tau, 121)
    dens = 5.95 + 0.95 * np.exp(-0.5 * ((xs - mu) / tau) ** 2)
    p2.fill_between(xs, np.full_like(xs, 5.95), dens, cls="a4")
    p2.line(xs, dens, s=4, w=1.6)
    p2.vline(mu, dash=True, y0=0.4, y1=5.95)
    p2.text(L(1.33), 6.55, "참값들의 분포", cls="lbl strong small")
    p2.text(L(1.33), 6.05, "평균 μ, 표준편차 τ", cls="lbl small")
    for yy, th, e, se in [(5.1, mu + 0.20, mu + 0.33, 0.22), (4.2, mu - 0.14, mu - 0.19, 0.10), (3.3, mu - 0.10, mu - 0.33, 0.24),
                          (2.4, mu + 0.16, mu + 0.19, 0.06), (1.5, mu - 0.02, mu + 0.10, 0.15)]:
        study(p2, yy, e, se)
        p2.top.append(f'<circle cx="{p2.sx(th):.1f}" cy="{p2.sy(yy) - 14:.1f}" r="4" class="pth s2"/>')
    p2.top.append(f'<circle cx="{p2.sx(L(1.33)) + 5:.1f}" cy="{p2.sy(3.25) - 4:.1f}" r="4" class="pth s2"/>')
    p2.text(L(1.33), 3.25, "각 연구의 참값", cls="lbl small", dx=14)
    p2.text(L(1.33), 2.75, "(연구마다 다름)", cls="lbl small", dx=14)
    panel_title(p2, "(나) 무작위효과 모형")
    return [p1.svg("고정효과 모형의 가정"), p2.svg("무작위효과 모형의 가정")]


save("ch19_models", figure(
    concept(),
    "그림 19-3. 두 모형이 가정하는 것. 파란 네모는 연구의 추정값(클수록 정밀한 연구), 가로선은 95% 신뢰구간입니다. (가) 고정효과 모형은 모든 연구가 같은 참값 하나를 추정한다고 봅니다. "
    "(나) 무작위효과 모형은 연구마다 참값(주황 원)이 조금씩 다르고 그 참값들이 어떤 분포를 이룬다고 보며, 그 분포의 평균 μ를 추정합니다.", cols=2))


# ====================================================================== forest plot builder
def forest_ma(rows, xlim, xticks, w=900, row_h=26, hdr=("Drug X", "Placebo"), xlabel="Odds ratio (95% CI)", left="← Favours drug X",
              right="Favours placebo →", wcol=True, foot=None, badges=None, c_plot=(386, 670)):
    top = 42
    nfoot = len(foot or [])
    yb = top + row_h * len(rows) + 4
    h = yb + 44 + 18 * nfoot + 6
    cA, cB = 262, 368                   # events/N columns (right aligned)
    X0, X1 = c_plot
    cE = w - (62 if wcol else 6)
    cW = w - 6
    lo_t, hi_t = L(xlim[0]), L(xlim[1])
    sx = lambda v: X0 + (L(v) - lo_t) / (hi_t - lo_t) * (X1 - X0)
    o = []
    for x_, an, t in [(6, "start", "Study"), (cA, "end", hdr[0]), (cB, "end", hdr[1]), ((X0 + X1) / 2, "middle", xlabel), (cE, "end", "OR (95% CI)")]:
        o.append(f'<text x="{x_:.1f}" y="16" text-anchor="{an}" class="axlab" font-weight="600">{escape(t)}</text>')
    if wcol:
        o.append(f'<text x="{cW}" y="16" text-anchor="end" class="axlab" font-weight="600">Weight</text>')
    o.append(f'<text x="{cA}" y="31" text-anchor="end" class="lbl mute small">events/N</text>')
    o.append(f'<text x="{cB}" y="31" text-anchor="end" class="lbl mute small">events/N</text>')
    for t in xticks:
        x_ = sx(t)
        o.append(f'<line x1="{x_:.1f}" y1="{top - 2}" x2="{x_:.1f}" y2="{yb}" class="grid"/>')
        o.append(f'<text x="{x_:.1f}" y="{yb + 17}" text-anchor="middle" class="tick">{fmt(t)}</text>')
    o.append(f'<line x1="{X0}" y1="{yb}" x2="{X1}" y2="{yb}" class="axis"/>')
    o.append(f'<line x1="{sx(1):.1f}" y1="{top - 2}" x2="{sx(1):.1f}" y2="{yb}" class="ref" stroke-width="1.4"/>')
    o.append(f'<text x="{sx(1) - 8:.1f}" y="{yb + 36}" text-anchor="end" class="lbl small">{escape(left)}</text>')
    o.append(f'<text x="{sx(1) + 8:.1f}" y="{yb + 36}" class="lbl small">{escape(right)}</text>')
    ypos = lambda i: top + row_h * i + row_h / 2
    for i, r in enumerate(rows):
        y = ypos(i)
        bold = ' font-weight="600"' if (r.get("head") or r.get("diamond")) else ""
        ind = 18 if r.get("ind") else 6
        cls = "lbl small" if r.get("note") else "lbl"
        o.append(f'<text x="{ind}" y="{y + 4.5:.1f}" class="{cls}"{bold}>{escape(r["label"])}</text>')
        if r.get("est") is None:
            continue
        if r.get("n1"):
            o.append(f'<text x="{cA}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num">{r["n1"]}</text>')
            o.append(f'<text x="{cB}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num">{r["n0"]}</text>')
        s = r.get("s", 1)
        a, b = sx(max(r["lo"], xlim[0])), sx(min(r["hi"], xlim[1]))
        xe = sx(r["est"])
        if r.get("diamond"):
            o.append(f'<polygon points="{a:.1f},{y:.1f} {xe:.1f},{y - 7:.1f} {b:.1f},{y:.1f} {xe:.1f},{y + 7:.1f}" class="f{s}"/>')
        else:
            o.append(f'<line x1="{a:.1f}" y1="{y:.1f}" x2="{b:.1f}" y2="{y:.1f}" class="ln s{s}" stroke-width="2"/>')
            if r["lo"] < xlim[0]:
                o.append(f'<polygon points="{a - 6:.1f},{y:.1f} {a + 1:.1f},{y - 4:.1f} {a + 1:.1f},{y + 4:.1f}" class="f{s}"/>')
            if r["hi"] > xlim[1]:
                o.append(f'<polygon points="{b + 6:.1f},{y:.1f} {b - 1:.1f},{y - 4:.1f} {b - 1:.1f},{y + 4:.1f}" class="f{s}"/>')
            sz = r.get("size", 5)
            if sz > 0.5:
                o.append(f'<rect x="{xe - sz:.1f}" y="{y - sz:.1f}" width="{2 * sz:.1f}" height="{2 * sz:.1f}" class="f{s}"/>')
        o.append(f'<text x="{cE}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num"{bold}>{r["est"]:.2f} ({r["lo"]:.2f}–{r["hi"]:.2f})</text>')
        if wcol and r.get("wt") is not None:
            o.append(f'<text x="{cW}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num"{bold}>{r["wt"]}</text>')
    for j, ln in enumerate(foot or []):
        o.append(f'<text x="6" y="{yb + 58 + 18 * j}" class="lbl small">{escape(ln)}</text>')
    for spec in (badges or []):
        kind, i, n = spec[:3]
        if kind == "label":
            r = rows[i]
            o.append(badge((18 if r.get("ind") else 6) + tw(r["label"]) * (1.12 if (r.get("head") or r.get("diamond")) else 1.0) + 18, ypos(i), n))
        elif kind == "note":
            o.append(badge(6 + tw(rows[i]["label"], 11.5) + 14, ypos(i), n))
        elif kind == "ev":
            o.append(badge(cB + 13, ypos(i), n))
        elif kind == "wt":
            o.append(badge(cW - 22, 31, n))
        elif kind == "axis":
            o.append(badge(sx(1) - tw(left, 11.5) - 22, yb + 32, n))
        elif kind == "est":
            o.append(badge(cE - 132, ypos(i), n))
        elif kind == "plot":
            o.append(badge(sx(spec[3]), ypos(i) + (spec[4] if len(spec) > 4 else 0), n))
        elif kind == "foot":
            o.append(badge(6 + tw(foot[i], 11.5) + 14, yb + 54 + 18 * i, n))
    return (f'<svg viewBox="0 0 {w} {h}" class="viz" role="img" aria-label="forest plot" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')


def pfmt(p):
    return "< .001" if p < 0.001 else ("= " + (f"{p:.2f}" if p >= 0.01 else f"{p:.3f}")).replace("0.", ".")


# ====================================================================== 19-3 forest plot, efficacy (나, body)
E = R["eff"]
rows = []
for i, t in enumerate(T):
    y, se = E["y"][i], E["se"][i]
    rows.append(dict(label=f'Trial {t["nm"]} ({t["yr"]})', n1=f'{t["eT"]:,}/{t["nT"]:,}', n0=f'{t["eC"]:,}/{t["nC"]:,}', est=math.exp(y), lo=math.exp(y - Z * se),
                     hi=math.exp(y + Z * se), size=2.2 + 13 * math.sqrt(E["wre"][i]), wt=f'{E["wre"][i] * 100:.1f}%'))
N = R["N"]
rows.append(dict(label="Random effects", diamond=True, est=E["re_or"][0], lo=E["re_or"][1], hi=E["re_or"][2], s=1,
                 n1=f'{N["eT"]:,}/{N["nT"]:,}', n0=f'{N["eC"]:,}/{N["nC"]:,}', wt="100%"))
rows.append(dict(label="Fixed effect", diamond=True, est=E["fe_or"][0], lo=E["fe_or"][1], hi=E["fe_or"][2], s=4))
rows.append(dict(label="95% prediction interval", est=E["re_or"][0], lo=E["pi_or"][0], hi=E["pi_or"][1], s=4, size=0, ind=True))
foot = [f'Heterogeneity: Q = {E["Q"]:.2f}, df = {E["df"]}, P {pfmt(E["pQ"])}; I² = {E["I2"] * 100:.0f}%; τ² = {E["t2"]:.4f}',
        f'Test for overall effect (random effects): z = {abs(E["re"] / E["re_se"]):.2f}, P {pfmt(E["re_p"])}']
save("ch19_forest", figure(
    [forest_ma(rows, (0.2, 5.5), (0.25, 0.5, 1, 2, 4), foot=foot)],
    "그림 19-2. 약물 X 추가의 유효성(심혈관 사망·심근경색·뇌졸중) 메타분석 포레스트 플롯(가상의 예시). 한 줄이 한 시험이고, 네모의 크기는 무작위효과 가중치(오른쪽 끝 열), "
    "가로선은 95% 신뢰구간입니다. 파란 마름모는 무작위효과 모형, 회색 마름모는 고정효과 모형으로 합친 값입니다. 맨 아래 회색 선(예측구간)은 다 절에서 설명합니다.",
    cols=1))

# ====================================================================== 19-4 forest plot, bleeding (나, paper box)
B = R["bleed"]
rows = []
for i, t in enumerate(T):
    y, se = B["y"][i], B["se"][i]
    rows.append(dict(label=f'Trial {t["nm"]} ({t["yr"]})', n1=f'{t["bT"]:,}/{t["nT"]:,}', n0=f'{t["bC"]:,}/{t["nC"]:,}', est=math.exp(y), lo=math.exp(y - Z * se),
                     hi=math.exp(y + Z * se), size=2.2 + 13 * math.sqrt(B["wre"][i]), wt=f'{B["wre"][i] * 100:.1f}%'))
rows.append(dict(label="Total (random effects)", diamond=True, est=B["re_or"][0], lo=B["re_or"][1], hi=B["re_or"][2], s=1,
                 n1=f'{N["bT"]:,}/{N["nT"]:,}', n0=f'{N["bC"]:,}/{N["nC"]:,}', wt="100%"))
foot = [f'Heterogeneity: Q = {B["Q"]:.2f}, df = {B["df"]}, P {pfmt(B["pQ"])}; I² = {B["I2"] * 100:.0f}%; τ² = {B["t2"]:.2f}',
        f'Test for overall effect: z = {abs(B["re"] / B["re_se"]):.2f}, P {pfmt(B["re_p"])}']
save("ch19_forest_bleed", figure(
    [forest_ma(rows, (0.1, 30), (0.1, 0.3, 1, 3, 10, 30), foot=foot, right="Favours placebo →",
               badges=[("ev", 3, 1), ("ev", 2, 2), ("wt", 0, 3), ("est", 10, 4), ("foot", 0, 5), ("axis", 0, 6)])],
    "그림 19-5. 약물 X 추가의 안전성(주요 출혈) 메타분석 포레스트 플롯(가상의 예시). 가로축은 로그 눈금이고, 신뢰구간이 그림 범위를 벗어나면 화살표로 표시했습니다. "
    "번호는 아래 해설과 짝을 이룹니다.", cols=1))

# ====================================================================== 19-5 subgroup forest (다, paper box)
SG = R["sub"]
rows = []
for key, lab, e1, n1, e0, n0 in (("S", "STEMI", "eTs", "nTs", "eCs", "nCs"), ("N", "NSTE-ACS", "eTn", "nTn", "eCn", "nCn")):
    M = SG[key]
    rows.append(dict(label=lab, head=True))
    for j, n in enumerate(SG["names"]):
        t = [x for x in T if x["nm"] == n][0]
        y, se = M["y"][j], M["se"][j]
        rows.append(dict(label=f'Trial {n}', ind=True, n1=f'{t[e1]:,}/{t[n1]:,}', n0=f'{t[e0]:,}/{t[n0]:,}', est=math.exp(y), lo=math.exp(y - Z * se),
                         hi=math.exp(y + Z * se), size=2.2 + 11 * math.sqrt(M["wre"][j]), wt=f'{M["wre"][j] * 100:.1f}%'))
    tt = SG["tot"][key]
    rows.append(dict(label="Subtotal", ind=True, diamond=True, est=M["re_or"][0], lo=M["re_or"][1], hi=M["re_or"][2], s=1,
                     n1=f'{tt[0]:,}/{tt[1]:,}', n0=f'{tt[2]:,}/{tt[3]:,}', wt="100%"))
    rows.append(dict(label=f'Heterogeneity: I² = {M["I2"] * 100:.0f}%, P {pfmt(M["pQ"])}; test for effect: P {pfmt(M["re_p"])}', ind=True, note=True))
A12 = SG["all12"]
rows.append(dict(label="Overall", diamond=True, est=A12["re_or"][0], lo=A12["re_or"][1], hi=A12["re_or"][2], s=4))
foot = [f'Test for subgroup differences: χ² = {SG["Qb"]:.2f}, df = 1, P for interaction {pfmt(SG["pint"])}',
        f'Overall heterogeneity: I² = {A12["I2"] * 100:.0f}%, P {pfmt(A12["pQ"])}. Weights are within-subgroup random-effects weights.']
save("ch19_subgroup", figure(
    [forest_ma(rows, (0.25, 4.2), (0.25, 0.5, 1, 2, 4), foot=foot,
               badges=[("label", 0, 1), ("est", 7, 2), ("est", 16, 3), ("note", 8, 4), ("foot", 0, 5)])],
    "그림 19-7. 기저 진단(STEMI 대 NSTE-ACS)에 따른 하위군 분석 포레스트 플롯(가상의 예시). 결과를 진단별로 나누어 보고한 6개 시험만 들어갑니다. "
    "하위군마다 소계(Subtotal) 마름모가 있고, 두 소계가 서로 다른지는 맨 아래 'Test for subgroup differences' 한 줄로 판단합니다.", cols=1))

# ====================================================================== 19-6 meta-regression bubble plot (다)
MR = R["mr"]
ps = R["mr_x"]
yt = [L(v) for v in (0.5, 0.7, 1.0, 1.4)]
p = Plot((30, 72), (L(0.42), L(1.75)), w=600, h=340, ml=62, mr=24, xlabel="시험 참여자 중 STEMI 환자의 비율 (%)", ylabel="오즈비 (로그 눈금)",
         yticks=yt, ytickfmt=lambda v: fmt(round(math.exp(v), 1)), xticks=[30, 40, 50, 60, 70])
p.hline(0, dash=True)
xs = np.array([32, 70.0])
p.line(xs, MR["b"][0] + MR["b"][1] * (xs - 50) / 10, s=2, w=2.2)
lab_off = {"A": (0, -12), "B": (17, 4), "C": (12, 4), "D": (0, -24), "E": (12, 4), "F": (0, -28), "G": (12, 4), "H": (0, 30), "I": (0, 28), "J": (-13, 4)}
for i, t in enumerate(T):
    rr = 3 + 34 * math.sqrt(MR["ws"][i] / MR["ws"].sum())
    p.els.append(f'<circle cx="{p.sx(ps[i]):.1f}" cy="{p.sy(E["y"][i]):.1f}" r="{rr:.1f}" class="a1 s1" stroke-width="1.4"/>')
    dx, dy = lab_off[t["nm"]]
    p.text(ps[i], E["y"][i], t["nm"], anchor="middle", cls="lbl small strong", dx=dx, dy=dy)
p.legend([("메타회귀 직선", 2, "line")], X=p.ml + 12, Y=p.mt + 12)
save("ch19_metareg", figure(
    [p.svg("메타회귀 버블 그림")],
    "그림 19-6. 메타회귀의 버블 그림(bubble plot). 원 하나가 시험 하나이고, 원이 클수록 가중치가 큰 시험입니다. 가로축은 시험 수준의 특성(STEMI 환자 비율), 세로축은 그 시험의 오즈비입니다. "
    f"직선의 기울기는 STEMI 비율이 10%p 높은 시험에서 오즈비가 {math.exp(MR['b'][1]):.2f}배라는 뜻이지만, 점이 10개뿐이어서 불확실합니다(P = {MR['p'][1]:.2f}).", cols=1))

# ====================================================================== 19-7 funnel plots (라)
EG = R["egger"]
smax = 0.75
xt = [L(v) for v in (0.25, 0.5, 1, 2, 4)]
xl = [(v, fmt(round(math.exp(v), 2))) for v in xt]
pa = Plot((L(0.2), L(5)), (-smax, 0), w=420, h=330, ml=58, mr=14, xlabel="오즈비 (로그 눈금)", ylabel="표준오차 (위로 갈수록 큰 시험)",
          yticks=[-0.6, -0.4, -0.2, 0], ytickfmt=lambda v: f"{abs(v):.1f}", xticks=xt, xticklabels=xl)
c = E["fe"]
pa.line([c - Z * smax, c, c + Z * smax], [-smax, 0, -smax], s=4, dash=True, w=1.4)
pa.vline(c, dash=False, cls="strongref", w=1.2)
pa.vline(0, dash=True)
pa.points(E["y"], -E["se"], s=1, r=5)
for nmx, dx, dy in (("C", 10, 4), ("G", -10, 4), ("E", -10, 4), ("A", 10, 4), ("F", -10, 2), ("D", 10, 5)):
    i = [t["nm"] for t in T].index(nmx)
    pa.text(E["y"][i], -E["se"][i], nmx, cls="lbl small", dx=dx, dy=dy, anchor="start" if dx > 0 else "end")
panel_title(pa, "(가) 이 장의 예: 시험 10개")

F = R["fun"]
smax2 = 0.6
pb = Plot((L(0.2), L(5)), (-smax2, 0), w=420, h=330, ml=58, mr=14, xlabel="오즈비 (로그 눈금)", ylabel="표준오차",
          yticks=[-0.6, -0.4, -0.2, 0], ytickfmt=lambda v: f"{abs(v):.1f}", xticks=xt, xticklabels=xl)
c2 = F["publ"]["fe"]
pb.line([c2 - Z * smax2, c2, c2 + Z * smax2], [-smax2, 0, -smax2], s=4, dash=True, w=1.4)
pb.vline(c2, dash=False, cls="strongref", w=1.2)
pb.vline(0, dash=True)
pub = np.array(F["pub"], bool)
pb.points(F["y"][~pub], -F["se"][~pub], s=2, r=5, hollow=True)
pb.points(F["y"][pub], -F["se"][pub], s=1, r=5)
X0, Y0 = pb.sx(L(1.25)), pb.mt + 12
pb.top.append(f'<circle cx="{X0:.1f}" cy="{Y0:.1f}" r="4.5" class="f1"/><text x="{X0 + 12:.1f}" y="{Y0 + 4.5:.1f}" class="lbl small">출판된 시험 17개</text>')
pb.top.append(f'<circle cx="{X0:.1f}" cy="{Y0 + 18:.1f}" r="4.5" class="pth s2"/><text x="{X0 + 12:.1f}" y="{Y0 + 22.5:.1f}" class="lbl small">출판되지 않은 13개</text>')
panel_title(pb, "(나) 출판 편향이 있을 때 (효과 없는 약)")
save("ch19_funnel", figure(
    [pa.svg("이 장 예제의 깔때기 그림"), pb.svg("출판 편향이 있는 깔때기 그림")],
    "그림 19-8. 깔때기 그림(funnel plot). 점 하나가 시험 하나이고, 위쪽이 크고 정밀한 시험입니다. 실선은 합친 값, 점선 삼각형은 편향과 이질성이 없을 때 시험의 95%가 들어오는 범위입니다. "
    "(가) 이 장의 10개 시험. 작은 시험(아래쪽)이 몇 개뿐이어서 대칭인지 아닌지 눈으로 가리기 어렵습니다. "
    "(나) 실제로는 효과가 없는 약(오즈비 1)의 시험 30개 가운데, 유리한 결과를 내지 못한 작은 시험 13개(빈 원)가 출판되지 않은 가상의 상황. 남은 점들은 아래쪽 오른편이 비어 있습니다.", cols=2))

# ====================================================================== 19-8 absolute effects by baseline risk (마)
lv = R["levels"]
Ab = R["abs_b"]
p = Plot((0, 3), (0, 40), w=600, h=330, ml=62, mr=24, ylabel="1,000명을 12개월 치료할 때의 사건 수 차이", xticks=[0.5, 1.5, 2.5],
         xticklabels=[(0.5, "기저 위험 4%"), (1.5, "기저 위험 8.4%"), (2.5, "기저 위험 15%")], yticks=[0, 10, 20, 30, 40],
         xlabel="대조군의 12개월 허혈 사건 위험 (주요 출혈의 기저 위험은 1.2%로 같다고 가정)")
for j, d in enumerate(lv):
    xc = j + 0.5
    p.bars([xc - 0.17], [d["per1000"]], 0.3, s=1)
    p.seg(xc - 0.17, d["lo"], xc - 0.17, d["hi"], cls="strongref", w=1.6)
    p.text(xc - 0.17, d["hi"], f'{d["per1000"]:.1f}', anchor="middle", cls="lbl strong small", dy=-6)
    p.bars([xc + 0.17], [Ab["per1000"][0]], 0.3, s=2)
    p.seg(xc + 0.17, Ab["per1000"][1], xc + 0.17, Ab["per1000"][2], cls="strongref", w=1.6)
    p.text(xc + 0.17, Ab["per1000"][2], f'{Ab["per1000"][0]:.1f}', anchor="middle", cls="lbl strong small", dy=-6)
p.legend([("막은 허혈 사건 (심혈관 사망·심근경색·뇌졸중)", 1, "box"), ("늘어난 주요 출혈", 2, "box")], X=p.ml + 10, Y=p.mt + 8)
save("ch19_absolute", figure(
    [p.svg("기저 위험에 따른 절대 효과")],
    "그림 19-9. 같은 오즈비(유효성 0.83, 주요 출혈 2.28)를 기저 위험이 다른 환자군에 적용했을 때 1,000명당 사건 수의 변화(가상의 예시). 세로선은 합친 오즈비의 95% 신뢰구간을 옮긴 범위입니다. "
    "상대적 효과가 같아도 막는 사건 수는 기저 위험이 높을수록 많아집니다.", cols=1))

# ====================================================================== 19-9 continuous-outcome forest (나, fold; from the old section)
H = R["hba"]
M_ = H["M"]
from svgplot import forest
import re
MM = lambda v, nd=2: f"{v:.{nd}f}".replace("-", "−")
rows = [{"label": "Study", "header": True}]
for t, y, se, wr in zip(H["trials"], H["y"], H["se"], M_["wre"]):
    rows.append({"label": f"{t[0]}  ({wr * 100:.1f}%)", "est": y, "lo": y - Z * se, "hi": y + Z * se, "size": 3 + 9 * wr ** 0.5 * 1.1, "indent": True})
rows.append({"label": "Fixed effect", "est": M_["fe"], "lo": M_["fe_ci"][0], "hi": M_["fe_ci"][1], "diamond": True, "bold": True, "s": 4})
rows.append({"label": "Random effects", "est": M_["re"], "lo": M_["re_ci"][0], "hi": M_["re_ci"][1], "diamond": True, "bold": True, "s": 1})
rows.append({"label": "  95% prediction interval", "est": M_["re"], "lo": M_["pi"][0], "hi": M_["pi"][1], "s": 4, "size": 0.01})
svg = forest(rows, xlim=(-1.8, 0.6), ref=0.0, log=False, w=680, row_h=28, label_w=220, est_w=196,
             xlabel="HbA1c 변화의 평균 차이, %p (중재 − 통상관리)", left_note="← 중재가 유리", right_note="통상관리가 유리 →",
             xticks=[-1.5, -1.0, -0.5, 0, 0.5], header=None)
svg = re.sub(r'>(-?\d\.\d\d) \((-?\d\.\d\d)–(-?\d\.\d\d)\)<',
             lambda m: f'>{MM(float(m.group(1)))} ({MM(float(m.group(2)))} to {MM(float(m.group(3)))})<', svg)
svg = svg.replace('class="tick">-', 'class="tick">−')
note = f"Heterogeneity: Q = {M_['Q']:.2f} (df = 5), P = {M_['pQ']:.4f}; I² = {M_['I2'] * 100:.0f}%; τ² = {M_['t2']:.3f}"
hgt = int(re.search(r'viewBox="0 0 \d+ (\d+)"', svg).group(1))
svg = svg.replace(f'viewBox="0 0 680 {hgt}"', f'viewBox="0 0 680 {hgt + 40}"', 1)
svg = svg.replace(f'y="{hgt - 10}" text-anchor="middle" class="axlab"', f'y="{hgt + 10}" text-anchor="middle" class="axlab"', 1)
svg = svg.replace('</svg>', f'<text x="4" y="{hgt + 34}" class="lbl small">{note}</text></svg>')
save("ch19_forest_md", figure(
    [svg],
    "그림 19-4. 약사 주도 중재가 6개월 HbA1c 변화에 미친 효과를 합친 가상의 메타분석(연속형 결과). 사각형 크기는 무작위효과 가중치(괄호 안 %)에 비례합니다. "
    "세로선 0이 '차이 없음'입니다. 맨 아래 회색 선은 95% 예측구간으로, 0을 넘어섭니다.", cols=1))
print("figures written")
