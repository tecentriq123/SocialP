import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt, forest
from nums_ch10 import example, MONTHS, TRUE_DIFF12, Z
from lib_ch10 import expit

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def mark(p, x, y, n, dx=0, dy=0):
    X, Y = p.sx(x) + dx, p.sy(y) + dy
    p.top.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
                 f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def errbar(p, x, lo, hi, s=1, cap=4, w=1.6):
    p.els.append(f'<line x1="{p.sx(x):.1f}" y1="{p.sy(lo):.1f}" x2="{p.sx(x):.1f}" y2="{p.sy(hi):.1f}" class="ln s{s}" stroke-width="{w}"/>')
    for v in (lo, hi):
        p.els.append(f'<line x1="{p.sx(x) - cap:.1f}" y1="{p.sy(v):.1f}" x2="{p.sx(x) + cap:.1f}" y2="{p.sy(v):.1f}" class="ln s{s}" stroke-width="{w}"/>')


def hbar(p, y, lo, hi, s=1, cap=4, w=1.8):
    p.els.append(f'<line x1="{p.sx(lo):.1f}" y1="{p.sy(y):.1f}" x2="{p.sx(hi):.1f}" y2="{p.sy(y):.1f}" class="ln s{s}" stroke-width="{w}"/>')
    for v in (lo, hi):
        p.els.append(f'<line x1="{p.sx(v):.1f}" y1="{p.sy(y) - cap:.1f}" x2="{p.sx(v):.1f}" y2="{p.sy(y) + cap:.1f}" class="ln s{s}" stroke-width="{w}"/>')


E = example()
D = E["D"]
g, y = D["g"], D["y"]
MX = [(0, "0"), (3, "3"), (6, "6"), (12, "12")]

# ------------------------------------------------------------------ 10-1 three patterns
pats = [
    ("시간 효과만", 8.5 - 0.04 * MONTHS, 8.5 - 0.04 * MONTHS),
    ("평행 (교호작용 없음)", 8.2 - 0.04 * MONTHS, 8.7 - 0.04 * MONTHS),
    ("교호작용", 8.5 - 0.06 * MONTHS, 8.5 - 0.015 * MONTHS),
]
svgs = []
for i, (title, a, b) in enumerate(pats):
    p = Plot((-0.8, 12.8), (7.5, 8.9), w=320, h=250, ml=46, mr=14, mt=28, mb=46, xlabel="개월",
             ylabel="HbA1c (%)" if i == 0 else "", xticklabels=MX, xticks=[0, 3, 6, 12], yticks=[7.6, 8.0, 8.4, 8.8],
             ytickfmt=lambda v: f"{v:.1f}")
    p.line(MONTHS, b, s=2, dash=True)
    p.points(MONTHS, b, s=2, r=3.5)
    p.line(MONTHS, a, s=1)
    p.points(MONTHS, a, s=1, r=3.5)
    if i == 0:
        p.legend([("중재군", 1, "line"), ("통상치료군", 2, "dash")], X=p.sx(4.2), Y=p.sy(8.78))
        p.text(12, a[-1], "두 선이 겹침", anchor="end", dy=22, cls="lbl mute small")
    if i == 1:
        p.seg(12.35, a[-1], 12.35, b[-1], cls="ref", w=1)
        p.text(12.2, (a[-1] + b[-1]) / 2, "차이 일정", anchor="end", cls="lbl mute small", dy=4)
    if i == 2:
        p.seg(12.35, a[-1], 12.35, b[-1], cls="ref", w=1)
        p.text(12.2, (a[-1] + b[-1]) / 2, "차이가 커짐", anchor="end", cls="lbl mute small", dy=4)
    panel_title(p, f"({'가나다'[i]}) {title}")
    svgs.append(p.svg(title))
save("ch10_patterns", figure(svgs,
     "그림 10-1. 두 군을 반복 측정했을 때 평균 추이의 세 가지 전형(가상의 예시). (가) 두 군이 함께 변하면 시간 효과만 있습니다. "
     "(나) 두 군의 차이가 처음부터 끝까지 같으면 군 효과는 있지만 교호작용은 없습니다. (다) 시간이 지날수록 두 군이 벌어지면 군 × 시간 교호작용이 있습니다. "
     "기저값이 같은 무작위배정 연구에서 중재 효과는 (다)의 모습으로 나타납니다.", cols=3))

# ------------------------------------------------------------------ 10-2 observed mean profiles with 95% CI and n
p = Plot((-1, 13), (7.3, 8.9), w=600, h=390, ml=62, mr=24, mt=24, mb=118, xlabel="",
         ylabel="HbA1c (%), 평균과 95% CI", xticklabels=MX, xticks=[0, 3, 6, 12],
         yticks=[7.4, 7.6, 7.8, 8.0, 8.2, 8.4, 8.6, 8.8], ytickfmt=lambda v: f"{v:.1f}")
for k, s, off, dash in ((1, 1, -0.22, False), (0, 2, 0.22, True)):
    d = E["desc"][k]
    xs = MONTHS + off
    p.line(xs, d["mean"], s=s, dash=dash)
    for x, lo, hi in zip(xs, d["lo"], d["hi"]):
        errbar(p, x, lo, hi, s=s)
    p.points(xs, d["mean"], s=s, r=4)
p.text(12.2, E["desc"][1]["mean"][3], "중재군", dx=10, dy=4, cls="lbl strong")
p.text(9.3, 8.3, "통상치료군", anchor="middle", cls="lbl strong")
B = p.h - p.mb
p.text_px((p.ml + p.w - p.mr) / 2, B + 40, "무작위배정 후 개월", anchor="middle", cls="axlab")
p.text_px(6, B + 66, "측정 인원 (n)", cls="lbl strong small")
for r_, (k, s, lab) in enumerate(((1, 1, "중재군"), (0, 2, "통상치료군"))):
    Y = B + 66 + 19 * (r_ + 1)
    p.top.append(f'<line x1="6" y1="{Y - 4}" x2="20" y2="{Y - 4}" class="ln s{s}" stroke-width="2.4"/>')
    p.text_px(25, Y, lab, cls="lbl small")
    for j, m in enumerate(MONTHS):
        p.text_px(p.sx(m), Y, str(E["desc"][k]["n"][j]), anchor="middle", cls="lbl small")
save("ch10_profile", figure(p.svg("군별 HbA1c 평균 추이"),
     "그림 10-2. 예제 연구(가상의 예시)에서 시점별로 측정된 환자의 HbA1c 평균과 95% 신뢰구간. 아래 숫자는 각 시점에 실제로 측정된 인원입니다. "
     "통상치료군은 12개월에 35명만 남았습니다. 반복측정 분산분석은 네 시점을 모두 측정한 75명만 분석에 씁니다."))

# ------------------------------------------------------------------ 10-3 variance of differences (sphericity)
vd = E["vardiff"]
p = Plot((0.4, 6.6), (0, 0.56), w=600, h=320, ml=62, mr=24, mt=24, mb=62, xlabel="두 시점의 쌍 (시점 사이 간격)",
         ylabel="차이값의 분산", xticks=[1, 2, 3, 4, 5, 6],
         xticklabels=[(i + 1, f"{a}–{b}개월") for i, (a, b, v) in enumerate(vd)], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
         ytickfmt=lambda v: f"{v:.1f}")
gaps = [b - a for a, b, v in vd]
p.bars([i + 1 for i in range(6)], [v for a, b, v in vd], 0.56, s=1)
for i, (a, b, v) in enumerate(vd):
    p.text(i + 1, v, f"{v:.2f}", anchor="middle", dy=-7, cls="lbl strong")
    p.text(i + 1, 0, f"({b - a}개월)", anchor="middle", dy=34, cls="lbl mute small")
mv = np.mean([v for a, b, v in vd])
p.hline(mv, x0=0.5, x1=6.5)
p.text(0.55, mv, f"평균 {mv:.2f}", dy=-6, cls="lbl mute small")
save("ch10_sphericity", figure(p.svg("시점 쌍별 차이값의 분산"),
     "그림 10-3. 네 시점을 모두 측정한 75명에서 두 시점 사이 차이값(예: 12개월 값 − 기저값)의 분산(군내 합동 분산). "
     "구형성 가정은 여섯 막대의 높이가 모두 같다는 가정입니다. 이 자료에서는 시점 사이 간격이 멀수록 분산이 커져 가정이 깨졌습니다."))

# ------------------------------------------------------------------ 10-4 spaghetti: observed vs fitted (random intercept + slope)
rs = E["rs"]
sel = [(3, 1), (23, 1), (8, 1), (36, 1), (1, 1), (9, 0), (38, 0), (15, 0), (22, 0), (5, 0)]
sv = []
for panel in (0, 1):
    p = Plot((-0.8, 12.8), (5.0, 10.6), w=420, h=330, ml=50, mr=16, mt=28, mb=50, xlabel="개월",
             ylabel="HbA1c (%)", xticklabels=MX, xticks=[0, 3, 6, 12], yticks=[5, 6, 7, 8, 9, 10])
    for pid, gg_ in sel:
        i = pid - 1
        s = 1 if gg_ == 1 else 2
        ob = ~np.isnan(y[i])
        if panel == 0:
            p.line(MONTHS[ob], y[i, ob], s=s, w=1.3, dash=(gg_ == 0))
            p.points(MONTHS[ob], y[i, ob], s=s, r=2.8)
        else:
            b0, b1 = E["blup_rs"][i]
            bt = rs.beta
            icp = bt[0] + bt[1] * gg_ + b0
            slp = bt[2] + bt[3] * gg_ + b1
            xx = np.array([0, 12.0])
            p.line(xx, icp + slp * xx, s=s, w=1.3, dash=(gg_ == 0))
            p.points(MONTHS[ob], y[i, ob], s=s, r=2.2, hollow=True)
    if panel == 1:
        bt = rs.beta
        xx = np.array([0, 12.0])
        p.line(xx, bt[0] + bt[2] * xx, s=2, w=4.5)
        p.line(xx, bt[0] + bt[1] + (bt[2] + bt[3]) * xx, s=1, w=4.5)
        p.text(12, bt[0] + bt[1] + (bt[2] + bt[3]) * 12, "중재군 평균", anchor="end", dy=20, cls="lbl strong small")
        p.text(12, bt[0] + bt[2] * 12, "통상치료군 평균", anchor="end", dy=-12, cls="lbl strong small")
        # patient 15: only two visits
        i = 14
        b0, b1 = E["blup_rs"][i]
        v12 = bt[0] + b0 + (bt[2] + b1) * 12
        p.text(12, v12, "#15 (2회 측정)", anchor="end", dy=-8, cls="lbl small")
        panel_title(p, "(나) 모형이 추정한 환자별 직선")
    else:
        p.legend([("중재군 환자", 1, "line"), ("통상치료군 환자", 2, "dash")], X=p.sx(-0.5), Y=p.sy(5.6))
        p.text(3, 10.1, "#15", anchor="middle", dy=-10, cls="lbl small")
        panel_title(p, "(가) 관측값 (환자 10명)")
    sv.append(p.svg("환자별 HbA1c 추이"))
save("ch10_spaghetti", figure(sv,
     "그림 10-4. 환자 10명의 HbA1c 추이(가상의 예시). (가) 관측값을 환자별로 이은 선. 환자마다 출발점(수준)과 기울기가 다릅니다. "
     "(나) 무작위 절편·기울기 모형이 추정한 환자별 직선(가는 선, 속이 빈 점은 관측값)과 군 평균 직선(굵은 선). "
     "두 번만 측정된 #15의 직선은 자기 관측값만 따르지 않고 군 평균 기울기 쪽으로 당겨져 있습니다.", cols=2))

# ------------------------------------------------------------------ 10-5 correlation structures
cf = E["covfits"]
pairs = [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (0, 3)]
labels = [(i + 1, f"{MONTHS[a]}–{MONTHS[b]}") for i, (a, b) in enumerate(pairs)]
p = Plot((0.4, 6.6), (0.5, 1.0), w=600, h=330, ml=62, mr=24, mt=24, mb=58, xlabel="두 시점의 쌍 (개월)",
         ylabel="두 시점 HbA1c의 상관계수", xticks=[1, 2, 3, 4, 5, 6], xticklabels=labels,
         yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ytickfmt=lambda v: f"{v:.1f}")
series = [("un", "비구조 (자료가 보여 주는 상관)", 1, -0.21), ("cs", "복합대칭 (= 무작위 절편)", 4, -0.07),
          ("ar1", "AR(1), 방문 순서 기준", 2, 0.07), ("rirs", "무작위 절편 + 기울기", 3, 0.21)]
leg = []
for kind, lab, s, off in series:
    Sg = cf[kind].Sigma
    sd = np.sqrt(np.diag(Sg))
    Rm = Sg / np.outer(sd, sd)
    vals = [Rm[a, b] for a, b in pairs]
    xs = [i + 1 + off for i in range(6)]
    if kind == "un":
        p.points(xs, vals, s=s, r=5.5)
    else:
        p.points(xs, vals, s=s, r=4.2, hollow=True)
    leg.append((f"{lab}, AIC {cf[kind].aic:.1f}", s, "dot"))
p.legend(leg, X=p.sx(0.55), Y=p.sy(0.64), gap=17)
p.text(3, 1.0, "시점 사이 간격: 3, 3, 6, 6, 9, 12개월", anchor="middle", dy=12, cls="lbl mute small")
save("ch10_corr", figure(p.svg("공분산 구조별 시점 간 상관"),
     "그림 10-5. 같은 자료(100명, 네 시점)에 네 가지 공분산 구조를 적합했을 때 모형이 가정하는 시점 간 상관. "
     "복합대칭은 모든 쌍의 상관을 0.80으로 같다고 보고, 방문 순서로 정의한 AR(1)은 3개월 간격인 0–3과 6개월 간격인 6–12를 같은 상관으로 봅니다. "
     "AIC(작을수록 좋음)는 무작위 절편 + 기울기 구조가 가장 작습니다."))

# ------------------------------------------------------------------ 10-6 usual-care means: observed vs full vs MAR-based model
un = cf["un"]
obs_m = E["desc"][0]["mean"]
full_m = E["full_mean"][0]
mod_m = un.beta[:4]
p = Plot((-0.8, 12.8), (7.9, 8.6), w=600, h=330, ml=62, mr=24, mt=24, mb=54, xlabel="무작위배정 후 개월",
         ylabel="통상치료군 평균 HbA1c (%)", xticklabels=MX, xticks=[0, 3, 6, 12],
         yticks=[7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6], ytickfmt=lambda v: f"{v:.1f}")
p.line(MONTHS, full_m, s=4, dash=True, w=2.4)
p.points(MONTHS, full_m, s=4, r=4)
p.line(MONTHS, mod_m, s=3, w=2.2)
p.points(MONTHS, mod_m, s=3, r=4)
p.line(MONTHS, obs_m, s=2, w=2.2)
p.points(MONTHS, obs_m, s=2, r=4)
for j in (1, 2, 3):
    p.text(MONTHS[j], obs_m[j], f"{obs_m[j]:.2f}", anchor="middle", dy=18, cls="lbl small")
    p.text(MONTHS[j], max(mod_m[j], full_m[j]), f"{mod_m[j]:.2f}", anchor="middle", dy=-10, cls="lbl small")
p.legend([("탈락자까지 모두 측정했다면 (실제로는 알 수 없음)", 4, "dash"), ("혼합모형 추정 (MAR 가정, 100명)", 3, "line"),
          ("관측된 사람만의 평균", 2, "line")], X=p.sx(-0.5), Y=p.sy(8.05), gap=17)
save("ch10_missing", figure(p.svg("탈락이 관측 평균에 미치는 영향"),
     "그림 10-6. 통상치료군의 시점별 평균 HbA1c. 혈당 조절이 나쁜 환자가 더 많이 탈락했기 때문에 남은 사람만의 평균(주황)은 실제보다 낮아 "
     f"12개월에 {full_m[3] - obs_m[3]:.2f}%p 차이가 납니다. 이 예제는 모의자료라서 탈락자의 실제 값(회색 점선)을 알 수 있는데, 탈락 전 값을 이용하는 혼합모형의 추정(초록, 숫자)은 이 값에 가깝습니다."))

# ------------------------------------------------------------------ 10-7 month-12 difference by method
comp = E["comp"]
rows = [("full", "탈락 없는 전체 자료 (n = 100)*", 4), ("mmrm", "MMRM, 관측 자료 모두 (n = 92)", 1),
        ("locf", "LOCF 대체 후 ANCOVA (n = 92)", 2), ("cc12", "12개월 측정자 ANCOVA (n = 79)", 2),
        ("ccall", "네 시점 모두 측정자 ANCOVA (n = 75)", 2)]
p = Plot((-0.95, 0.15), (0.4, len(rows) + 0.6), w=640, h=312, ml=236, mr=110, mt=30, mb=52,
         xlabel="12개월 HbA1c 변화량의 군간 차이 (%p), 95% CI", xticks=[-0.8, -0.6, -0.4, -0.2, 0.0],
         xtickfmt=lambda v: fmt(v, 1).replace("-", "−"), ygrid=False, xgrid=True, show_yaxis=False)
p.vline(0, dash=False, cls="ref strongref")
p.vline(TRUE_DIFF12, dash=True)
p.text(TRUE_DIFF12, len(rows) + 0.6, "모의자료의 참값 −0.48", anchor="middle", dy=-8, cls="lbl mute small")
for r_, (key, lab, s) in enumerate(rows):
    yv = len(rows) - r_
    c = comp[key]
    hbar(p, yv, c["lo"], c["hi"], s=s)
    p.points([c["est"]], [yv], s=s, r=5)
    p.text_px(8, p.sy(yv) + 4.5, lab, cls="lbl")
    p.text_px(p.w - 6, p.sy(yv) + 4.5, f"{c['est']:.2f} ({c['lo']:.2f} to {c['hi']:.2f})".replace("-", "−"), anchor="end", cls="lbl num small")
save("ch10_methods", figure(p.svg("방법별 12개월 군간 차이"),
     "그림 10-7. 같은 예제 자료를 결측 처리 방법만 바꿔 분석한 12개월 HbA1c 변화량의 군간 차이(중재군 − 통상치료군). "
     "*맨 위 줄은 모의자료라서 가능한 기준값입니다. 완전사례 분석은 차이를 작게 추정했고, 네 시점을 모두 측정한 사람만 쓰면 신뢰구간이 0을 포함합니다."))

# ------------------------------------------------------------------ 10-8 conditional vs marginal (GLMM estimates, 3-month visit)
gm = E["glmm"]
b0, b1, sig = gm["beta"][0], gm["beta"][1], gm["sigma"]
P0, P1, ORm = E["glmm_implied"][0]
uu = np.linspace(-4.5, 4.5, 241)
p = Plot((-4.5, 4.5), (0, 1.0), w=600, h=360, ml=62, mr=24, mt=24, mb=56,
         xlabel="환자 고유의 순응 성향 (무작위효과 u, 로짓 척도)", ylabel="3개월 PDC ≥ 80% 확률",
         xticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
         ytickfmt=lambda v: f"{int(round(v * 100))}%")
dens = st.norm.pdf(uu, 0, sig)
dens = dens / dens.max() * 0.16
p.fill_between(uu, np.zeros_like(uu), dens, cls="a4")
p.text(0, 0.16, f"환자 분포 (SD {sig:.2f})", anchor="middle", dy=-6, cls="lbl mute small")
p.line(uu, expit(b0 + b1 + uu), s=1)
p.line(uu, expit(b0 + uu), s=2, dash=True)
p.hline(P1, x0=-4.5, x1=4.5)
p.hline(P0, x0=-4.5, x1=4.5)
p.text(4.4, P1, f"중재군 모집단 평균 {P1 * 100:.1f}%", anchor="end", dy=-6, cls="lbl small")
p.text(4.4, P0, f"통상치료군 모집단 평균 {P0 * 100:.1f}%", anchor="end", dy=-6, cls="lbl small")
for u0 in (-2, 0, 2):
    pa, pb = expit(b0 + u0), expit(b0 + b1 + u0)
    p.seg(u0, pa, u0, pb, cls="ref", w=1.4)
p.text(0.12, (expit(b0) + expit(b0 + b1)) / 2, f"같은 u에서 OR {np.exp(b1):.2f}", dy=4, cls="lbl strong small")
p.legend([("중재군 (환자별 곡선)", 1, "line"), ("통상치료군 (환자별 곡선)", 2, "dash")], X=p.sx(1.3), Y=p.sy(0.3), gap=17)
save("ch10_margcond", figure(p.svg("조건부 오즈비와 주변 오즈비"),
     f"그림 10-9. 무작위 절편 로지스틱 모형(가상의 예시)이 추정한 3개월 순응 확률. 같은 순응 성향(u)을 가진 환자끼리 비교한 오즈비는 어느 u에서나 {np.exp(b1):.2f}입니다(조건부, 환자 단위). "
     f"그러나 환자 분포 전체에 대해 평균한 두 군의 순응률 {P1 * 100:.1f}%와 {P0 * 100:.1f}%로 계산한 오즈비는 {ORm:.2f}입니다(주변, 모집단 평균). GEE가 추정하는 것은 뒤의 값입니다."))

# ------------------------------------------------------------------ 10-9 OR forest by method
ge = E["gee"]
nl = E["naive_logit"]


def row(label, b, se, s=1, bold=False):
    return dict(label=label, est=np.exp(b), lo=np.exp(b - Z * se), hi=np.exp(b + Z * se), s=s, bold=bold)


frows = [row("일반 로지스틱 (독립 가정)", nl["beta"][1], nl["se"][1], s=2),
         row("GEE 독립 + 강건 SE", ge["independence"]["beta"][1], ge["independence"]["se_robust"][1]),
         row("GEE 교환가능 + 강건 SE", ge["exchangeable"]["beta"][1], ge["exchangeable"]["se_robust"][1], bold=True),
         row("GEE AR(1) + 강건 SE", ge["ar1"]["beta"][1], ge["ar1"]["se_robust"][1]),
         row("GEE 비구조 + 강건 SE", ge["unstructured"]["beta"][1], ge["unstructured"]["se_robust"][1]),
         row("혼합모형 (조건부 OR)", gm["beta"][1], gm["se"][1], s=3)]
save("ch10_orforest", figure(forest(frows, (0.8, 12), ref=1.0, log=True, w=640, label_w=220, est_w=130,
                                   xlabel="중재군의 순응(PDC ≥ 80%) 오즈비 (95% CI)", xticks=[1, 2, 4, 8],
                                   header=("분석 방법", "OR (95% CI)")),
     "그림 10-10. 같은 순응도 자료(93명, 259개 관측)를 여러 방법으로 분석한 중재군의 오즈비. 일반 로지스틱 회귀와 GEE는 점추정값이 거의 같지만 "
     "일반 로지스틱의 신뢰구간이 좁습니다(표준오차 과소추정). 작업상관 구조를 바꿔도 GEE 결과는 거의 같습니다. 혼합모형의 오즈비가 더 큰 것은 추정 대상(조건부)이 다르기 때문입니다."))

# ------------------------------------------------------------------ paper figure: LS mean change by visit (MMRM)
lsm = E["lsm"]
p = Plot((-1, 13), (-0.95, 0.3), w=600, h=360, ml=66, mr=24, mt=24, mb=56, xlabel="Month",
         ylabel="Adjusted mean change in HbA1c (%)", xticklabels=MX, xticks=[0, 3, 6, 12],
         yticks=[-0.8, -0.6, -0.4, -0.2, 0.0, 0.2], ytickfmt=lambda v: fmt(v, 1).replace("-", "−"))
p.hline(0, dash=False)
for k, s, off, dash in ((1, 1, -0.22, False), (0, 2, 0.22, True)):
    xs = [0 + off] + [m + off for m in (3, 6, 12)]
    ys = [0] + [lsm[(k, j)]["est"] for j in range(3)]
    p.line(xs, ys, s=s, dash=dash)
    for j in range(3):
        c = lsm[(k, j)]
        errbar(p, xs[j + 1], c["lo"], c["hi"], s=s)
    p.points(xs, ys, s=s, r=4.2)
d12 = lsm[("d", 2)]
d6 = lsm[("d", 1)]
p.text(6, lsm[(0, 1)]["hi"], "P = %.3f" % d6["p"], anchor="middle", dy=-8, cls="lbl small")
p.text(12, lsm[(0, 2)]["hi"], "P = %.3f" % d12["p"], anchor="middle", dy=-8, cls="lbl small")
p.legend([("Pharmacist intervention", 1, "line"), ("Usual care", 2, "dash")], X=p.sx(-0.6), Y=p.sy(-0.72), gap=17)
mark(p, 0.2, 0.0, 1, dx=-12, dy=-16)
mark(p, 12, lsm[(1, 2)]["lo"], 2, dx=18, dy=4)
mark(p, 12, lsm[(0, 2)]["est"], 3, dx=24, dy=0)
save("ch10_paperfig", figure(p.svg("Adjusted mean change in HbA1c by visit"),
     "그림 10-8. 학술지 형식의 MMRM 결과 그림(가상의 예시). 원문 캡션: Figure 2. Least-squares mean change from baseline in HbA1c by visit "
     "(mixed model for repeated measures). Error bars indicate 95% CIs; P values are for between-group differences at each visit."))
print("figures written")
