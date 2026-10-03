"""Figures for chapter 11 (Cox proportional-hazards model). run: python3 gen/fig_ch11.py"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
import scipy.stats as st
from svgplot import Plot, figure, panel_title, forest, fmt
from lib_ch07 import km, surv_at, small_arrays, logrank, scenario, n_risk, Z
from lib_ch11 import (coxph, cox_loglik, zph, ph_test_approx, martingale, survsplit, adjusted_curves, lowess,
                      rcc_cohort, design, concordance)

OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def km_xy(rows, xmax):
    xs, ys = [0.0], [1.0]
    for r in rows:
        if r["d"] > 0 and r["t"] <= xmax:
            xs.append(r["t"]); ys.append(r["S"])
    tend = min(max(r["t"] for r in rows), xmax)
    xs.append(tend); ys.append(ys[-1])
    return xs, ys


def censor_xy(rows, xmax):
    xs, ys = [], []
    S = 1.0
    for r in rows:
        if r["d"] > 0:
            S = r["S"]
        if r["c"] > 0 and r["t"] <= xmax:
            xs.append(r["t"]); ys.append(S)
    return xs, ys


def mark(p, x, y, n, dx=0, dy=0):
    X, Y = p.sx(x) + dx, p.sy(y) + dy
    p.top.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
                 f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def mark_px(p, X, Y, n):
    p.top.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
                 f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def risk_table(p, grid, rows_, y0, title="No. at risk", label_x=6, gap=19):
    B = p.h - p.mb
    p.text_px(label_x, B + y0, title, cls="lbl strong small")
    for i, (lab, s, counts) in enumerate(rows_):
        Y = B + y0 + gap * (i + 1)
        p.top.append(f'<line x1="{label_x:.1f}" y1="{Y - 4:.1f}" x2="{label_x + 14:.1f}" y2="{Y - 4:.1f}" class="ln s{s}" stroke-width="2.4"/>')
        p.text_px(label_x + 19, Y, lab, cls="lbl small")
        for x, c in zip(grid, counts):
            p.text_px(p.sx(x), Y, str(c), anchor="middle", cls="lbl small num")


# ---------------------------------------------------------------- data
D = rcc_cohort()
T, S_, G, IM = D["time"], D["status"], D["drug"], D["imdc"]
X, NM = design(D)
f1 = coxph(T, S_, X[:, :1])
fm = coxph(T, S_, X)
rA = km(T[G == 1], S_[G == 1]); rB = km(T[G == 0], S_[G == 0])

# ======================================================================
# 11-1 hazard shapes (Weibull, same median 20 months)
MED = 20.0
shapes = [(0.6, 2, "감소하는 위험률"), (1.0, 4, "일정한 위험률"), (1.6, 1, "증가하는 위험률")]
pa = Plot((0, 60), (0, 0.12), w=420, h=310, ml=56, mr=14, mt=26, mb=50, xlabel="개월", ylabel="위험률 h(t) (1개월당)",
          xticks=[0, 12, 24, 36, 48, 60], yticks=[0, 0.03, 0.06, 0.09, 0.12], ytickfmt=lambda v: f"{v:.2f}")
pb = Plot((0, 60), (0, 1.0), w=420, h=310, ml=52, mr=14, mt=26, mb=50, xlabel="개월", ylabel="생존 확률 S(t)",
          xticks=[0, 12, 24, 36, 48, 60], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
tg = np.linspace(0.05, 60, 300)
for k, s, lab in shapes:
    b = MED / np.log(2) ** (1 / k)
    h = (k / b) * (tg / b) ** (k - 1)
    m = h <= 0.12
    pa.line(tg[m], h[m], s=s, w=2.2, dash=(s == 4))
    pb.line(np.concatenate([[0], tg]), np.concatenate([[1], np.exp(-(tg / b) ** k)]), s=s, w=2.2, dash=(s == 4))
pa.text(9, 0.049, "감소 (k = 0.6)", cls="lbl small", dx=4)
pa.text(40, 0.0347, "일정 (k = 1)", cls="lbl small", dy=-8)
pa.text(47, 0.095, "증가 (k = 1.6)", cls="lbl small", anchor="end")
panel_title(pa, "(가) 위험률")
pb.hline(0.5, x1=20); pb.vline(20, y1=0.5)
pb.text(20, 0.5, "세 곡선 모두 중앙값 20개월", dx=6, dy=-8, cls="lbl small")
pb.text(58, 0.31, "감소", anchor="end", dy=-6, cls="lbl small")
pb.text(58, 0.19, "일정", anchor="end", dy=-6, cls="lbl small")
pb.text(31, 0.13, "증가", anchor="end", cls="lbl small")
panel_title(pb, "(나) 같은 위험률에서 나오는 생존곡선")
save("ch11_hazard", figure([pa.svg("세 가지 모양의 위험률"), pb.svg("위험률에 대응하는 생존곡선")],
     "그림 11-1. 모양이 다른 세 위험률(시간이 갈수록 줄어들 때, 일정할 때, 늘어날 때)과 그에 대응하는 생존곡선. 그림의 k는 모양을 정하는 값으로, 1보다 작으면 감소, 1이면 일정, 1보다 크면 증가입니다. 세 경우 모두 중앙생존기간은 20개월로 같습니다. "
     "위험률이 감소하는 경우(수술 직후처럼 초기에 위험이 몰림)는 초반에 곡선이 빨리 떨어지고 뒤에는 평평해지며, 위험률이 증가하는 경우(노화, 누적 독성)는 반대입니다. "
     "생존곡선에서는 이런 차이가 곡선이 휘는 모양으로만 간접적으로 보이므로 위험률이라는 척도가 따로 필요합니다.", cols=2))

# ======================================================================
# 11-9 partial likelihood curve (12 patients)
t12, s12, g12 = small_arrays()
xc = (g12 - g12.mean())[:, None]
bgrid = np.linspace(-4.0, 1.2, 260)
llg = np.array([cox_loglik([b], t12, s12, xc)[0] for b in bgrid])
fe = coxph(t12, s12, g12[:, None])
bh, ll_h, ll_0 = fe["coef"][0], fe["loglik"], fe["loglik0"]
U0 = cox_loglik([0.0], t12, s12, xc)[1][0]
p = Plot((-4.3, 1.3), (-16.5, -10.5), w=600, h=360, ml=62, mr=24, mt=20, mb=74, xlabel="",
         ylabel="로그 부분우도 log L(β)", xticks=[-4, -3, -2, -1, 0, 1], yticks=[-16, -15, -14, -13, -12, -11],
         xtickfmt=lambda v: fmt(v))
p.line(bgrid, llg, s=1, w=2.4)
# tangent at 0 (score)
xs_t = np.array([-1.0, 0.8])
p.line(xs_t, ll_0 + U0 * xs_t, s=2, w=1.6, dash=True)
p.points([0], [ll_0], s=2, r=4.5)
p.points([bh], [ll_h], s=1, r=4.5)
p.hline(ll_h, x0=bh, x1=0.35, dash=True)
p.seg(0.25, ll_0, 0.25, ll_h, cls="ln s3", w=2)
p.text(0.3, (ll_0 + ll_h) / 2, "우도비: 높이 차이", dx=4, dy=-2, cls="lbl small")
p.text(0.3, (ll_0 + ll_h) / 2, f"2 × {ll_h - ll_0:.3f} = {2 * (ll_h - ll_0):.2f}", dx=4, dy=14, cls="lbl small")
p.text(bh, ll_h, f"최댓값 β̂ = {bh:.3f}", anchor="middle", dy=-12, cls="lbl strong")
p.text(bh, ll_h, f"(HR = e^β̂ = {np.exp(bh):.3f})", anchor="middle", dy=4 + 18, cls="lbl small")
p.text(0, ll_0, "β = 0 (HR = 1)", anchor="end", dx=-8, dy=20, cls="lbl small")
p.text(-0.15, -13.75, "점수: β = 0에서 주황 점선의 기울기", anchor="end", cls="lbl small")
p.seg(bh, -16.2, 0, -16.2, cls="ln s4", w=2)
p.text((bh + 0) / 2, -16.2, f"Wald: β̂ − 0 = {bh:.2f}, SE {fe['se'][0]:.2f}", anchor="middle", dy=-7, cls="lbl small")
B = p.h - p.mb
for v in (-4, -3, -2, -1, 0, 1):
    p.text_px(p.sx(v), B + 34, f"{np.exp(v):.2f}" if v < 0 else f"{np.exp(v):.1f}" if v > 0 else "1", anchor="middle", cls="tick")
p.text_px(p.ml - 4, B + 18, "β", anchor="end", cls="axlab")
p.text_px(p.ml - 4, B + 34, "HR", anchor="end", cls="axlab")
p.text_px((p.ml + p.w - p.mr) / 2, B + 58, "약물 A의 회귀계수 β (아래 줄은 위험비 e^β)", anchor="middle", cls="axlab")
save("ch11_plik", figure(p.svg("12명 자료의 로그 부분우도 곡선"),
     f"그림 11-9. 표 7-1의 환자 12명 자료에서 β(약물 A = 1, 약물 B = 0)의 값을 바꿔 가며 계산한 로그 부분우도. 곡선이 가장 높은 β̂ = {bh:.3f}이 추정값입니다. "
     f"β = 0(두 약이 같음)과 비교하는 세 검정은 같은 곡선의 다른 부분을 봅니다. 우도비 검정은 두 점의 높이 차이, Wald 검정은 가로 거리(β̂ − 0)를 표준오차로 나눈 값, "
     f"점수 검정은 β = 0에서 곡선의 기울기(주황 점선)가 0에서 얼마나 먼지를 봅니다(8장 마 절)."))

# ======================================================================
# 11-2 HR vs RR over time (drug B KM baseline, crude HR)
HRc = f1["hr"][0]
ev = [r for r in rB if r["d"] > 0 and r["t"] <= 60]
xs = [0.0] + [r["t"] for r in ev] + [60.0]
SB = [1.0] + [r["S"] for r in ev] + [ev[-1]["S"]]
riskB = [1 - v for v in SB]
riskA = [1 - v ** HRc for v in SB]
pa = Plot((0, 60), (0, 1.0), w=420, h=320, ml=56, mr=14, mt=26, mb=50, xlabel="개월", ylabel="누적 사망위험 1 − S(t)",
          xticks=[0, 12, 24, 36, 48, 60], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{int(round(v * 100))}%")
pa.step(xs, riskB, s=2, w=2.2)
pa.step(xs, riskA, s=1, w=2.2)
for x in (12, 36):
    sb = surv_at(rB, x)[0]
    pa.vline(x, y1=1 - sb, dash=True)
    pa.points([x, x], [1 - sb, 1 - sb ** HRc], s=4, r=3.5)
    pa.text(x, 1 - sb, f"{(1 - sb) * 100:.1f}%", dx=-5, dy=-7, anchor="end", cls="lbl small")
    pa.text(x, 1 - sb ** HRc, f"{(1 - sb ** HRc) * 100:.1f}%", dx=5, dy=17, anchor="start", cls="lbl small")
pa.text(56, 0.87, "약물 B (관찰된 KM)", anchor="end", cls="lbl strong small")
pa.text(58, 0.42, "약물 A (HR 0.73 가정)", anchor="end", cls="lbl strong small")
panel_title(pa, "(가) 누적 사망위험")
tg = np.linspace(0.5, 60, 240)
rr = []
for x in tg:
    sb = surv_at(rB, x)[0]
    rr.append((1 - sb ** HRc) / (1 - sb) if sb < 1 else np.nan)
rr = np.array(rr)
pb = Plot((0, 60), (0.6, 1.0), w=420, h=320, ml=52, mr=14, mt=26, mb=50, xlabel="개월", ylabel="비 (약물 A ÷ 약물 B)",
          xticks=[0, 12, 24, 36, 48, 60], yticks=[0.6, 0.7, 0.8, 0.9, 1.0], ytickfmt=lambda v: f"{v:.1f}")
pb.hline(1.0, dash=False, cls="ref")
pb.line([0, 60], [HRc, HRc], s=1, w=2.2, dash=True)
m = ~np.isnan(rr)
pb.line(tg[m], rr[m], s=2, w=2.2)
pb.text(59, HRc, f"위험비 HR = {HRc:.2f} (시간에 따라 일정)", anchor="end", dy=16, cls="lbl small")
pb.text(44, 0.80, "누적위험의 비 (상대위험도)", anchor="middle", cls="lbl small")
for x in (12, 36, 60):
    sb = surv_at(rB, x)[0]
    v = (1 - sb ** HRc) / (1 - sb)
    pb.points([x], [v], s=2, r=3.5)
    pb.text(x, v, f"{v:.2f}", anchor="middle", dy=-9, cls="lbl small")
panel_title(pb, "(나) 위험비와 상대위험도")
save("ch11_hrrr", figure([pa.svg("누적 사망위험 곡선"), pb.svg("위험비와 시점별 상대위험도")],
     f"그림 11-2. 위험비가 전 기간 {HRc:.2f}로 일정하다고 가정하고, 약물 B의 Kaplan-Meier 곡선(7장 그림 7-6)으로부터 약물 A의 곡선을 S<sub>A</sub>(t) = S<sub>B</sub>(t)<sup>{HRc:.2f}</sup>로 계산했습니다(설명용 계산). "
     f"(가) 누적 사망위험. (나) 위험비는 일정하지만, 누적 사망위험의 비(상대위험도)는 사건이 쌓일수록 1 쪽으로 올라갑니다.", cols=2))

# ======================================================================
# 11-3 log(-log S) vs log t
def lml_xy(rows, tmin=0.5):
    xs, ys = [], []
    for r in rows:
        if r["d"] > 0 and 0 < r["S"] < 1 and r["t"] >= tmin:
            xs.append(np.log(r["t"])); ys.append(np.log(-np.log(r["S"])))
    return xs, ys


def lml_step(p, rows, s, xmax):
    xs, ys = lml_xy(rows)
    px, py = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        px += [xs[i], xs[i]]; py += [ys[i - 1], ys[i]]
    px.append(np.log(xmax)); py.append(ys[-1])
    p.line(px, py, s=s, w=2)


lt = [1, 3, 6, 12, 24, 48]
panels = []
fn1 = km(T[D["ncc"] == 1], S_[D["ncc"] == 1]); fn0 = km(T[D["ncc"] == 0], S_[D["ncc"] == 0])
for title, r1, r0, l1, l0, yl in (("(가) 약물 A와 약물 B", rA, rB, "약물 A", "약물 B", (-4.5, 1.0)),
                                  ("(나) 조직형", fn1, fn0, "비투명세포", "투명세포", (-4.5, 1.0))):
    pl = Plot((np.log(0.5), np.log(72)), yl, w=420, h=320, ml=52, mr=14, mt=26, mb=50,
              xlabel="개월 (로그 눈금)", ylabel="log(−log S(t))", xticks=[np.log(v) for v in lt],
              xticklabels=[(np.log(v), str(v)) for v in lt], yticks=[-4, -3, -2, -1, 0, 1], xgrid=True)
    lml_step(pl, r0, 2, 72)
    lml_step(pl, r1, 1, 72)
    panel_title(pl, title)
    panels.append(pl)
pA_, pB_ = panels
pA_.text(np.log(20), -0.35, "약물 B", anchor="end", dy=-10, cls="lbl strong small")
pA_.text(np.log(30), -0.55, "약물 A", anchor="start", dy=16, cls="lbl strong small")
pA_.text(np.log(0.6), 0.7, "두 선의 간격이 대체로 일정", cls="lbl small")
pB_.text(np.log(2.2), -1.0, "비투명세포", anchor="middle", dy=-10, cls="lbl strong small")
pB_.text(np.log(3.2), -2.9, "투명세포", anchor="start", dy=16, cls="lbl strong small")
pB_.text(np.log(0.6), 0.7, "초반 간격이 크고 뒤로 갈수록 좁아짐", cls="lbl small")
save("ch11_lml", figure([pA_.svg("약물별 로그-로그 생존 그림"), pB_.svg("조직형별 로그-로그 생존 그림")],
     "그림 11-3. 로그-로그 생존 그림: x축은 로그 시간, y축은 각 군 Kaplan-Meier 추정값의 log(−log S(t))입니다. 비례위험이 성립하면 두 선의 세로 간격이 log(HR)로 일정합니다. "
     "(가) 약물 A와 B는 간격이 −0.2에서 −0.5 사이에서 흔들리지만 뚜렷한 추세는 없습니다. (나) 비투명세포 조직형의 간격은 3개월 1.44에서 36개월 0.34로 줄어듭니다. 초반에만 위험이 크다는 뜻입니다.", cols=2))

# ======================================================================
# 11-4 scaled Schoenfeld residuals (km transform) for drug and histology
terms = {"drug": [0], "age": [1], "sex": [2], "imdc": [3, 4], "nephrectomy": [5], "histology": [6]}
zr, gt, sc = zph(fm, terms, "km")
scl = sc["scaled"]; tev = sc["time"]
PHT = ph_test_approx(fm, "km")
# km transform for tick labels
rows_all = km(T, S_)
kt = np.array([r["t"] for r in rows_all]); kS = np.array([r["S"] for r in rows_all])


def gk(t):
    i = np.searchsorted(kt, t, side="left") - 1
    return 1 - (kS[i] if i >= 0 else 1.0)


tick_t = [2, 6, 12, 24, 48]
panels = []
for j, nm_, title, yl, yt in ((0, "drug", "(가) 약물 A", (-4, 4), [-4, -2, 0, 2, 4]),
                              (6, "histology", "(나) 비투명세포 조직형", (-3, 9), [-2, 0, 2, 4, 6, 8])):
    pl = Plot((0, 0.8), yl, w=420, h=320, ml=52, mr=14, mt=26, mb=50, xlabel="사망 시점 (개월, KM 변환 눈금)",
              ylabel="β(t)의 추정 (척도화 잔차)", xticks=[gk(v) for v in tick_t],
              xticklabels=[(gk(v), str(v)) for v in tick_t], yticks=yt, xgrid=True)
    pl.hline(0, dash=False, cls="ref")
    pl.points(gt, np.clip(scl[:, j], yl[0], yl[1]), s=4, r=2.2)
    xs_, ys_ = lowess(gt, scl[:, j], frac=0.4)
    pl.line(xs_, ys_, s=1 if j == 0 else 2, w=2.6)
    pl.line([0, 0.8], [fm["coef"][j]] * 2, s=4, w=1.6, dash=True)
    chi, pv = PHT[0][j], PHT[1][j]
    pl.text(0.79, yl[1], f"비례위험 검정(km): χ² = {chi:.2f}, P = {pv:.3f}", anchor="end", dy=16, cls="lbl small")
    pl.text(0.79, fm["coef"][j], f"β̂ = {fm['coef'][j]:.2f}", anchor="end", dy=(16 if j == 0 else -6), cls="lbl small")
    panel_title(pl, title)
    panels.append(pl.svg(title))
save("ch11_schoen", figure(panels,
     "그림 11-4. 다변수 모형의 척도화 Schoenfeld 잔차(점: 사망 한 건당 하나)와 평활선. 세로축은 그 시점의 계수 β(t)를 추정한 값으로 읽고, 점선은 모형이 추정한 하나의 β̂입니다. "
     "가로축은 사망 시점을 전체 환자의 Kaplan-Meier 척도(1 − S(t))로 바꿔 사망이 고르게 퍼지게 했습니다. 오른쪽 위의 검정은 이 가로축(km 변환)을 쓴 비례위험 검정입니다. "
     "(가) 약물 변수의 평활선은 뚜렷한 추세 없이 β̂ 주위를 오르내립니다. (나) 조직형의 평활선은 초반에 높고(첫 8개월 척도화 잔차의 평균 1.78, 위험비 약 5.9) 뒤로 갈수록 0 근처로 내려옵니다. 그림 밖으로 나가는 점은 가장자리에 붙여 표시했습니다.", cols=2))

# ======================================================================
# 11-5 time-varying HR: histology (split at 12 and log t) and the crossing trial
def split_fit(time, status, Xbase, col_idx_varying, cut):
    ids, a, b, ev_, ep = survsplit(time, status, [cut])
    Xs = Xbase[ids]
    other = [c for c in range(Xs.shape[1]) if c != col_idx_varying]
    Xd = np.column_stack([Xs[:, other], Xs[:, col_idx_varying] * (ep == 0), Xs[:, col_idx_varying] * (ep == 1)])
    f = coxph(b, ev_, Xd, entry=a)
    k = len(other)
    return [(f["hr"][k], f["lo"][k], f["hi"][k]), (f["hr"][k + 1], f["lo"][k + 1], f["hi"][k + 1])]


def tv_panel(title, xmax, yl, yt, pw, tt_fit, jcol, ph, cut, labels):
    pl = Plot((0, xmax), (np.log(yl[0]), np.log(yl[1])), w=420, h=320, ml=52, mr=14, mt=26, mb=50, xlabel="개월",
              ylabel="위험비 (로그 눈금)", xticks=list(range(0, xmax + 1, 12)),
              yticks=[np.log(v) for v in yt], ytickfmt=lambda v: fmt(round(np.exp(v), 2)))
    pl.hline(0, dash=False, cls="ref")
    (h0, l0, u0), (h1, l1, u1) = pw
    pl.fill_between([0, cut], [np.log(l0)] * 2, [np.log(u0)] * 2, s=1)
    pl.fill_between([cut, xmax], [np.log(l1)] * 2, [np.log(u1)] * 2, s=1)
    pl.line([0, cut], [np.log(h0)] * 2, s=1, w=2.4)
    pl.line([cut, xmax], [np.log(h1)] * 2, s=1, w=2.4)
    tg_ = np.linspace(0.5, xmax, 200)
    pl.line(tg_, tt_fit["coef"][jcol] + tt_fit["coef"][-1] * np.log(tg_), s=2, w=2.2)
    pl.line([0, xmax], [np.log(ph)] * 2, s=4, w=1.8, dash=True)
    for (x, y, txt, anc, dy) in labels:
        pl.text(x, y, txt, anchor=anc, dy=dy, cls="lbl small")
    panel_title(pl, title)
    return pl.svg(title)


pw_h = split_fit(T, S_, X, 6, 12)
ftt = coxph(T, S_, X, tt=[(6, np.log)])
tt_c, ss_c, gg_c = scenario("late", 23)
pw_c = split_fit(tt_c, ss_c, gg_c[:, None].astype(float), 0, 4)
ftc = coxph(tt_c, ss_c, gg_c[:, None], tt=[(0, np.log)])
fc = coxph(tt_c, ss_c, gg_c[:, None])
pan1 = tv_panel("(가) 조직형 (비투명세포 vs 투명세포)", 60, (0.4, 8), [0.5, 1, 2, 4, 8], pw_h, ftt, 6, fm["hr"][6], 12,
                [(6, np.log(pw_h[0][0]), f"0–12개월 {pw_h[0][0]:.2f}", "middle", -8),
                 (59, np.log(pw_h[1][0]), f"12개월 이후 {pw_h[1][0]:.2f}", "end", -8),
                 (59, np.log(fm["hr"][6]), f"비례위험 가정 {fm['hr'][6]:.2f}", "end", -8),
                 (45, ftt["coef"][6] + ftt["coef"][-1] * np.log(45), "log t 교호작용", "middle", 16)])
pan2 = tv_panel("(나) 곡선이 교차하는 시험 (7장 그림 7-9 (나))", 48, (0.3, 3), [0.3, 0.5, 1, 2, 3], pw_c, ftc, 0, fc["hr"][0], 4,
                [(5, np.log(pw_c[0][0]), f"0–4개월 {pw_c[0][0]:.2f}", "start", 4),
                 (47, np.log(pw_c[1][0]), f"4개월 이후 {pw_c[1][0]:.2f}", "end", 16),
                 (47, np.log(fc["hr"][0]), f"비례위험 가정 {fc['hr'][0]:.2f}", "end", -6),
                 (30, ftc["coef"][0] + ftc["coef"][-1] * np.log(30), "log t 교호작용", "middle", 18)])
save("ch11_tvhr", figure([pan1, pan2],
     "그림 11-5. 시간에 따라 변하는 위험비를 추정한 두 가지 방법. 파란 계단과 음영은 추적 기간을 나눠 구간마다 따로 추정한 위험비와 95% 신뢰구간, 주황 곡선은 공변량 × log(t) 교호작용 항으로 추정한 연속적인 위험비, 회색 점선은 비례위험을 가정한 하나의 위험비입니다. "
     "(가) 조직형은 첫 1년에만 위험을 높입니다. (나) 7장의 가상 시험에서 치료 X는 처음 4개월에는 불리하고(HR &gt; 1) 그 뒤로 유리합니다(HR &lt; 1). 두 경우 모두 하나의 위험비는 서로 다른 시기를 섞은 평균입니다.", cols=2))

# ======================================================================
# 11-6 martingale residuals vs age (model without age)
f_noage = coxph(T, S_, X[:, [0, 2, 3, 4, 5, 6]])
M = martingale(f_noage)
age = D["age"].astype(float)
xs_, ys_ = lowess(age, M, frac=0.6)
p = Plot((36, 88), (-3.7, 1.2), w=600, h=330, ml=62, mr=24, mt=20, mb=54, xlabel="나이 (세)",
         ylabel="마팅게일 잔차", xticks=[40, 50, 60, 70, 80], yticks=[-3, -2, -1, 0, 1])
p.hline(0, dash=False, cls="ref")
p.points(age[S_ == 0] + 0.0, M[S_ == 0], s=4, r=2.6, hollow=True)
p.points(age[S_ == 1], M[S_ == 1], s=4, r=2.6)
p.line(xs_, ys_, s=1, w=2.8)
p.text(86, 0.99, "사망한 환자: 1 − 누적위험", anchor="end", dy=-6, cls="lbl small")
p.text(86, -3.3, "중도절단 환자: 0 − 누적위험 (항상 음수)", anchor="end", cls="lbl small")
i45 = np.argmin(np.abs(xs_ - 45)); i75 = np.argmin(np.abs(xs_ - 75))
p.text(40, -0.5, "평활선 (LOWESS)", cls="lbl strong small")
save("ch11_mart", figure(p.svg("나이에 따른 마팅게일 잔차"),
     "그림 11-6. 나이를 뺀 모형의 마팅게일 잔차를 나이에 대해 그린 그림. 채운 점은 사망, 빈 원은 중도절단 환자입니다. 평활선의 모양이 나이가 로그 위험률에 들어가야 할 모양을 대략 보여 줍니다. "
     "45세 −0.69에서 65세 +0.05까지는 거의 직선으로 오르고 75세 이후에는 평평해집니다. 끝부분은 환자가 적어 불확실하므로 제곱항이나 스플라인을 넣은 모형과 비교해 판단합니다."))

# ======================================================================
# 11-7 paper figure: unadjusted KM vs standardized (adjusted) curves
lr = logrank(T, S_, G, 1)
grid_t = np.linspace(0, 60, 241)
adj = adjusted_curves(fm, X, 0, [1, 0], grid_t)
pa = Plot((0, 60), (0, 1.0), w=420, h=380, ml=52, mr=14, mt=26, mb=112, xlabel="",
          ylabel="Overall survival (%)", xticks=[0, 12, 24, 36, 48, 60], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
          ytickfmt=lambda v: f"{int(round(v * 100))}")
xa, ya = km_xy(rA, 60); xb, yb = km_xy(rB, 60)
pa.step(xb, yb, s=2, w=2); pa.step(xa, ya, s=1, w=2)
cxa, cya = censor_xy(rA, 60); cxb, cyb = censor_xy(rB, 60)
pa.ticks_marks(cxa, cya, s=1, size=3.5); pa.ticks_marks(cxb, cyb, s=2, size=3.5)
pa.legend([("Drug A", 1, "line"), ("Drug B", 2, "line")], X=pa.sx(33), Y=pa.sy(0.97))
pa.text(1.5, 0.155, f"HR {f1['hr'][0]:.2f} (95% CI {f1['lo'][0]:.2f}–{f1['hi'][0]:.2f})", cls="lbl small")
pa.text(1.5, 0.155, f"Log-rank P = {lr['p']:.3f}", dy=16, cls="lbl small")
Bq = pa.h - pa.mb
pa.text_px((pa.ml + pa.w - pa.mr) / 2, Bq + 34, "Months", anchor="middle", cls="axlab")
grid = [0, 12, 24, 36, 48, 60]
risk_table(pa, grid, [("A", 1, [n_risk(T[G == 1], x) for x in grid]), ("B", 2, [n_risk(T[G == 0], x) for x in grid])], 58)
panel_title(pa, "A  Unadjusted (Kaplan–Meier)")
mark(pa, 33, 0.155, 1, dy=-4)
mark(pa, 24.5, 0.155, 2, dy=12)
mark(pa, 24, surv_at(rA, 24)[0], 3, dx=0, dy=-16)
pb = Plot((0, 60), (0, 1.0), w=420, h=380, ml=52, mr=14, mt=26, mb=112, xlabel="",
          ylabel="Overall survival (%)", xticks=[0, 12, 24, 36, 48, 60], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
          ytickfmt=lambda v: f"{int(round(v * 100))}")
pb.line(grid_t, adj[0], s=2, w=2); pb.line(grid_t, adj[1], s=1, w=2)
pb.legend([("Drug A", 1, "line"), ("Drug B", 2, "line")], X=pb.sx(33), Y=pb.sy(0.97))
pb.text(1.5, 0.155, f"Adjusted HR {fm['hr'][0]:.2f} (95% CI {fm['lo'][0]:.2f}–{fm['hi'][0]:.2f})", cls="lbl small")
pb.text(1.5, 0.155, f"P = {fm['p'][0]:.2f}", dy=16, cls="lbl small")
Bq = pb.h - pb.mb
pb.text_px((pb.ml + pb.w - pb.mr) / 2, Bq + 34, "Months", anchor="middle", cls="axlab")
pb.text_px(6, Bq + 58, "Curves standardized to the covariate", cls="lbl small")
pb.text_px(6, Bq + 76, "distribution of all 300 patients", cls="lbl small")
panel_title(pb, "B  Adjusted (standardized)")
i24 = np.argmin(np.abs(grid_t - 24))
mark(pb, 42, 0.155, 4, dy=-4)
mark(pb, 24, adj[1][i24], 5, dx=0, dy=-16)
mark_px(pb, 6 + 250, Bq + 56, 6)
save("ch11_paperadj", figure([pa.svg("보정하지 않은 Kaplan-Meier 곡선"), pb.svg("표준화한 보정 생존곡선")],
     "그림 11-7. 학술지 형식의 그림(가상의 예시). 원문 범례: “Figure 2. Overall survival according to first-line treatment. (A) Kaplan–Meier estimates; the hazard ratio was estimated with a univariable Cox model and the P value with the log-rank test. "
     "(B) Survival curves standardized to the covariate distribution of the whole cohort, based on the multivariable Cox model (age, sex, IMDC risk group, prior nephrectomy, and histologic type).”", cols=2))

# ======================================================================
# 11-8 subgroup forest plot
age65 = (D["age"] >= 65).astype(int)
sub_defs = [
    ("Age", [("< 65 yr", age65 == 0), ("≥ 65 yr", age65 == 1)], age65, None),
    ("Sex", [("Female", D["male"] == 0), ("Male", D["male"] == 1)], D["male"], None),
    ("IMDC risk group", [("Favorable", IM == 0), ("Intermediate", IM == 1), ("Poor", IM == 2)], None, [3, 4]),
    ("Prior nephrectomy", [("No", D["neph"] == 0), ("Yes", D["neph"] == 1)], D["neph"], None),
]
covcols = {"Age": [1, 2, 3, 4, 5, 6], "Sex": [1, 3, 4, 5, 6], "IMDC risk group": [1, 2, 5, 6], "Prior nephrectomy": [1, 2, 3, 4, 6]}
rows = [dict(label=f"All patients ({len(T)}/{int(S_.sum())})", est=fm["hr"][0], lo=fm["lo"][0], hi=fm["hi"][0], bold=True, diamond=True, s=1)]
for name, levels, var, dcols in sub_defs:
    inter = (X[:, 0] * var)[:, None] if dcols is None else X[:, [0]] * X[:, dcols]
    f_int = coxph(T, S_, np.column_stack([X, inter]))
    lrt = 2 * (f_int["loglik"] - fm["loglik"])
    p_int = st.chi2.sf(lrt, inter.shape[1])
    rows.append(dict(label=f"{name} (P for interaction = {p_int:.2f})", header=True))
    for lab, m in levels:
        cc = [0] + [c for c in covcols[name] if np.ptp(X[m][:, c]) > 0]
        fsg = coxph(T[m], S_[m], X[m][:, cc])
        rows.append(dict(label=f"{lab} ({int(m.sum())}/{int(S_[m].sum())})", est=fsg["hr"][0], lo=fsg["lo"][0], hi=fsg["hi"][0],
                         indent=True, s=1))
svg = forest(rows, (0.25, 4), ref=1.0, log=True, w=640, row_h=27, label_w=250, est_w=128,
             xlabel="Adjusted hazard ratio (95% CI)", left_note="Drug A better", right_note="Drug B better",
             header=("Subgroup (patients/deaths)", "HR (95% CI)"), xticks=[0.25, 0.5, 1, 2, 4])
import re as _re
_h = int(_re.search(r'viewBox="0 0 640 (\d+)"', svg).group(1))
svg = svg.replace(f'viewBox="0 0 640 {_h}"', f'viewBox="0 0 640 {_h + 20}"')
svg = svg.replace(f'y="{_h - 10}" text-anchor="middle" class="axlab"', f'y="{_h + 10}" text-anchor="middle" class="axlab"')
save("ch11_forest", figure(svg, "그림 11-8. 하위군별 위험비(가상의 예시). 각 하위군 안에서 다른 공변량을 보정한 약물 A 대 약물 B의 위험비와 95% 신뢰구간이며, 괄호 안은 환자 수/사망 수입니다. 네모의 크기는 모두 같게 그렸습니다."))
print("figures written")
