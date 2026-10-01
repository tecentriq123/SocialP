import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from html import escape
from svgplot import Plot, figure, panel_title, fmt
from lib_ch07 import (km, surv_at, quantile_ci, rmst, logrank, cox_efron, n_risk, ci_from,
                      small_arrays, SMALL, CUTOFF, cohort, scenario)

OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def km_xy(rows, xmax):
    """step coordinates for Plot.step: value ys[i] holds on [xs[i], xs[i+1])"""
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


def band_xy(rows, xmax, kind="loglog"):
    """step-shaped CI band polygons (lists) for fill_between"""
    segs = []
    S, gw = 1.0, 0.0
    knots = [0.0] + [r["t"] for r in rows if r["d"] > 0 and r["t"] <= xmax]
    vals = [(1.0, 1.0)]
    for r in rows:
        if r["d"] > 0 and r["t"] <= xmax:
            vals.append(ci_from(r["S"], r["gw"], kind))
    tend = min(max(r["t"] for r in rows), xmax)
    knots.append(tend)
    xs, lo, hi = [], [], []
    for i in range(len(vals)):
        a, b = knots[i], knots[i + 1]
        xs += [a, b]; lo += [vals[i][0]] * 2; hi += [vals[i][1]] * 2
    return xs, lo, hi


def mark(p, x, y, n, dx=0, dy=0):
    X, Y = p.sx(x) + dx, p.sy(y) + dy
    p.top.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
                 f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def risk_table(p, grid, rows_, y0, title="위험집합 크기 (number at risk)", label_x=6, gap=19):
    """rows_: list of (label, series, counts). draws under the x axis."""
    B = p.h - p.mb
    p.text_px(label_x, B + y0, title, cls="lbl strong small")
    for i, (lab, s, counts) in enumerate(rows_):
        Y = B + y0 + gap * (i + 1)
        if s:
            p.top.append(f'<line x1="{label_x:.1f}" y1="{Y - 4:.1f}" x2="{label_x + 14:.1f}" y2="{Y - 4:.1f}" class="ln s{s}" stroke-width="2.4"/>')
            p.text_px(label_x + 19, Y, lab, cls="lbl small")
        else:
            p.text_px(label_x, Y, lab, cls="lbl small")
        for x, c in zip(grid, counts):
            p.text_px(p.sx(x), Y, str(c), anchor="middle", cls="lbl small num")


pct = lambda v: f"{int(round(v * 100))}"

# ======================================================================
# 7-1 swimmer plot: calendar time vs follow-up time
t, s, g = small_arrays()
nP = len(SMALL)


def swimmer(calendar):
    w, h = 420, 400
    if calendar:
        p = Plot((0, 50), (0.3, nP + 0.8), w=w, h=h, ml=44, mr=16, mt=30, mb=56, xlabel="달력 시간 (진료 연월)",
                 xticks=[0, 12, 24, 36, 48], xticklabels=[(0, "2019.1"), (12, "2020.1"), (24, "2021.1"), (36, "2022.1"), (48, "2023.1")],
                 yticks=[], ygrid=False, show_yaxis=False)
    else:
        p = Plot((0, 50), (0.3, nP + 0.8), w=w, h=h, ml=44, mr=16, mt=30, mb=56, xlabel="치료 시작 후 개월 수 (추적 시간)",
                 xticks=[0, 12, 24, 36, 48], yticks=[], ygrid=False, show_yaxis=False, xgrid=True)
    for pid, entry, time, stat, reason, drug in SMALL:
        y = nP + 1 - pid
        a = entry if calendar else 0
        b = a + time
        p.seg(a, y, b, y, cls="ln s1", w=3)
        if stat == 1:
            p.points([b], [y], s=2, r=5)
        elif reason == "자료 마감":
            p.points([b], [y], s=1, r=4.6, hollow=True)
        else:
            p.points([b], [y], s=4, r=4.6, hollow=True)
            if calendar:
                p.text(b, y, "전원", dx=8, dy=4, cls="lbl mute small")
        p.text_px(p.ml - 8, p.sy(y) + 4, f"#{pid}", anchor="end", cls="tick")
    if calendar:
        p.vline(CUTOFF, y1=nP + 0.8, cls="ref strongref")
        p.text(CUTOFF, nP + 0.8, "자료 마감", anchor="end", dx=-4, dy=-4, cls="lbl small")
        panel_title(p, "(가) 달력 시간: 등록 시점이 제각각")
    else:
        panel_title(p, "(나) 추적 시간: 모두 0에서 출발")
    return p


p1 = swimmer(True)
p2 = swimmer(False)
p2.legend([("사망 (사건)", 2, "dot")], X=250, Y=p2.mt + 14)
p2.top.append(f'<circle cx="{250 + 11:.1f}" cy="{p2.mt + 32:.1f}" r="4.6" class="pth s1"/><text x="{280:.1f}" y="{p2.mt + 36.5:.1f}" class="lbl">중도절단 (자료 마감)</text>')
p2.top.append(f'<circle cx="{250 + 11:.1f}" cy="{p2.mt + 50:.1f}" r="4.6" class="pth s4"/><text x="{280:.1f}" y="{p2.mt + 54.5:.1f}" class="lbl">중도절단 (전원)</text>')
save("ch07_swimmer", figure([p1.svg("달력 시간 기준 환자별 추적 막대"), p2.svg("추적 시간 기준 환자별 추적 막대")],
     "그림 7-1. 가상의 신세포암 환자 12명의 추적 과정. (가) 실제 달력에서는 환자마다 치료를 시작한 시점이 다르고, 자료 마감일(2023년 1월)에 아직 살아 있던 환자는 그 시점에서 관찰이 끝납니다. (나) 생존분석은 각자의 치료 시작일을 0으로 맞춘 '추적 시간'으로 바꿔서 봅니다. 채운 점은 사망, 빈 원은 중도절단입니다.", cols=2))

# ======================================================================
# 7-2 claims new-user design timeline
p = Plot((-420, 600), (0, 5.2), w=640, h=300, ml=96, mr=18, mt=18, mb=46, xlabel="Index date 기준 일수",
         xticks=[-365, 0, 90, 180, 270, 360, 450, 540], yticks=[], ygrid=False, show_yaxis=False)
# washout band
p.fill_between([-365, 0], [0.3, 0.3], [5.0, 5.0], cls="a4")
p.text(-182, 4.75, "세척 기간 (washout) 365일", anchor="middle", cls="lbl small")
p.text(-182, 4.75, "비교 두 약 모두 처방 없음", anchor="middle", dy=15, cls="lbl mute small")
p.text(-182, 4.75, "= 기저 공변량 측정 구간", anchor="middle", dy=30, cls="lbl mute small")
p.vline(0, y0=0.3, y1=5.0, cls="ref strongref", dash=False, w=1.6)
p.text(0, 5.0, "Index date (첫 처방일)", anchor="start", dx=5, dy=12, cls="lbl strong small")
# row labels
rows_lab = [(3.6, "처방 (공급일수)"), (2.35, "as-treated 추적"), (1.1, "ITT 유사 추적")]
for y, lab in rows_lab:
    p.text_px(8, p.sy(y) + 4, lab, cls="lbl small")
# prescription bars
for a, b in [(0, 90), (95, 185), (205, 295)]:
    X0, X1 = p.sx(a), p.sx(b)
    p.els.append(f'<rect x="{X0:.1f}" y="{p.sy(3.85):.1f}" width="{X1 - X0:.1f}" height="{p.sy(3.35) - p.sy(3.85):.1f}" rx="3" class="f1"/>')
p.text(45, 3.85, "90일분", anchor="middle", dy=-5, cls="lbl mute small")
# grace period
X0, X1 = p.sx(295), p.sx(355)
p.els.append(f'<rect x="{X0:.1f}" y="{p.sy(3.85):.1f}" width="{X1 - X0:.1f}" height="{p.sy(3.35) - p.sy(3.85):.1f}" rx="3" class="a1 s1" stroke-width="1" stroke-dasharray="3 3"/>')
p.text(325, 3.85, "유예 60일", anchor="middle", dy=-5, cls="lbl small")
# as-treated line
p.seg(1, 2.35, 355, 2.35, cls="ln s1", w=3)
p.points([355], [2.35], s=1, r=5, hollow=True)
p.text(355, 2.35, "중단 → 중도절단", dx=10, dy=4, cls="lbl small")
# ITT-like
p.seg(1, 1.1, 470, 1.1, cls="ln s1", w=3)
p.points([470], [1.1], s=2, r=5.5)
p.text(470, 1.1, "심부전 입원", dx=10, dy=4, cls="lbl small")
p.text(1, 0.55, "추적 시작 = index date 다음 날", dx=4, cls="lbl mute small")
save("ch07_claims", figure(p.svg("청구자료 신규사용자 코호트의 시간축"),
     "그림 7-2. 청구자료 신규사용자 설계에서 한 환자의 시간축(가상의 예). 첫 처방일(index date) 이전 365일 동안 비교하는 두 약의 처방이 없어야 '신규' 사용자로 봅니다. as-treated 분석에서는 마지막 처방의 공급일수가 끝난 뒤 유예기간(grace period) 60일 안에 재처방이 없으면 그 시점에 중도절단합니다. ITT 유사 분석은 중단과 관계없이 계속 추적하므로 이 환자의 심부전 입원이 사건으로 잡힙니다."))

# ======================================================================
# 7-3 KM curve for 12 patients
rows = km(t, s)
p = Plot((0, 48), (0, 1.0), w=600, h=380, ml=62, mr=24, mt=20, mb=112, xlabel="",
         ylabel="생존 확률 S(t)", xticks=[0, 6, 12, 18, 24, 30, 36, 42, 48], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
         ytickfmt=lambda v: f"{v:.1f}")
xs, ys = km_xy(rows, 48)
p.hline(0.5, dash=True, x1=28)
p.vline(28, y1=0.5, dash=True)
S24 = surv_at(rows, 24)[0]
p.seg(24, 0, 24, S24, cls="ref", dash=True)
p.step(xs, ys, s=1, w=2.4)
cx_, cy_ = censor_xy(rows, 48)
p.ticks_marks(cx_, cy_, s=1, size=6)
p.points([24], [S24], s=2, r=4.5)
p.text(24, S24, f"S(24) = {S24:.3f}", dx=-6, dy=-10, anchor="end", cls="lbl strong")
p.text(28, 0.5, "중앙생존기간 28개월", dx=6, dy=-8, cls="lbl strong")
p.text(47.5, 0.2005, "38개월: 2명 중 1명 사망", anchor="end", dy=20, cls="lbl mute small")
p.text(11.4, 0.70, "12개월: 9명 중 2명 사망", anchor="end", cls="lbl mute small")
B = p.h - p.mb
p.text_px((p.ml + p.w - p.mr) / 2, B + 36, "치료 시작 후 개월 수", anchor="middle", cls="axlab")
risk_table(p, [0, 12, 24, 36, 48], [("전체", None, [n_risk(t, x) for x in (0, 12, 24, 36, 48)])], 62)
save("ch07_km12", figure(p.svg("12명 자료의 Kaplan-Meier 생존곡선"),
     "그림 7-3. 표 7-2의 Kaplan-Meier 곡선. 계단은 사망이 있었던 시점에서만 내려가고, 세로 눈금(|)은 중도절단 시점입니다. 사망 시점에서 떨어지는 폭은 '그 직전 위험집합 중 사망한 비율'이라, 남은 사람이 적은 후반부일수록 한 명의 사망이 큰 계단을 만듭니다. 아래 숫자는 각 시점에 아직 관찰 중인 환자 수입니다."))

# ======================================================================
# 7-4 CI bands: 12 patients vs cohort arm A
C = cohort(2)
T, S_, G, IM = C["time"], C["status"], C["drug"], C["imdc"]
pa = Plot((0, 48), (0, 1.0), w=420, h=300, ml=52, mr=14, mt=26, mb=50, xlabel="개월", ylabel="생존 확률",
          xticks=[0, 12, 24, 36, 48], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
bx, blo, bhi = band_xy(rows, 48)
pa.fill_between(bx, blo, bhi, s=1)
pa.step(xs, ys, s=1, w=2.2)
pa.ticks_marks(cx_, cy_, s=1, size=5)
pa.text(31, 0.9, "95% CI (log-log)", anchor="middle", cls="lbl small")
pa.text(40, 0.2005, "n = 2", dx=0, dy=24, anchor="middle", cls="lbl mute small")
panel_title(pa, "(가) 12명")
rA = km(T[G == 1], S_[G == 1])
pb = Plot((0, 84), (0, 1.0), w=420, h=300, ml=52, mr=14, mt=26, mb=50, xlabel="개월", ylabel="생존 확률",
          xticks=[0, 12, 24, 36, 48, 60, 72, 84], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
bx, blo, bhi = band_xy(rA, 84)
pb.fill_between(bx, blo, bhi, s=1)
xa, ya = km_xy(rA, 84)
pb.step(xa, ya, s=1, w=2.2)
for x in (12, 36, 60, 72):
    pb.text(x, 0.02, f"n={n_risk(T[G == 1], x)}", anchor="middle", dy=-4, cls="lbl mute small")
panel_title(pb, "(나) 약물 A 코호트 152명")
save("ch07_ci", figure([pa.svg("12명 자료의 신뢰구간 띠"), pb.svg("152명 자료의 신뢰구간 띠")],
     "그림 7-4. Kaplan-Meier 곡선의 95% 신뢰구간(log-log 변환, 음영). 같은 방법이라도 환자가 적으면 띠가 넓고, 한 코호트 안에서도 위험집합(n)이 줄어드는 후반부로 갈수록 띠가 넓어집니다. (나)의 아래쪽 숫자는 그 시점의 위험집합 크기입니다. (나)의 곡선은 마지막까지 남은 1명이 79.9개월에 사망해 17.2%에서 0으로 떨어집니다.", cols=2))

# ======================================================================
# 7-5 RMST area
a36, se36, parts = rmst(rows, 36)
p = Plot((0, 48), (0, 1.0), w=600, h=330, ml=62, mr=24, mt=20, mb=54, xlabel="치료 시작 후 개월 수",
         ylabel="생존 확률 S(t)", xticks=[0, 3, 8, 12, 20, 28, 36, 48], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
         ytickfmt=lambda v: f"{v:.1f}")
for a, b, sv in parts:
    p.fill_between([a, b], [0, 0], [sv, sv], s=1)
    p.vline(b, y1=sv, dash=False, cls="ref", w=0.8)
    p.text((a + b) / 2, sv / 2, f"{sv * (b - a):.2f}", anchor="middle", cls="lbl small num", dy=4)
p.step(xs, ys, s=1, w=2.2)
p.vline(36, y1=1.0, dash=True, cls="ref strongref")
p.text(36, 0.95, "τ = 36개월", dx=6, cls="lbl strong")
p.text(36, 0.80, f"음영 넓이 = RMST", dx=6, cls="lbl")
p.text(36, 0.80, f"= {a36:.1f}개월", dx=6, dy=17, cls="lbl strong")
save("ch07_rmst", figure(p.svg("제한평균생존시간: 생존곡선 아래 넓이"),
     f"그림 7-5. 제한평균생존시간(RMST)은 0부터 τ까지 생존곡선 아래의 넓이입니다. 계단 곡선이므로 직사각형 넓이(높이 S × 폭)의 합이 되고, 칸 안의 숫자가 각 직사각형의 넓이입니다. 합계 {a36:.1f}개월은 '처음 36개월 가운데 평균적으로 생존해 있던 기간'을 뜻합니다."))

# ======================================================================
# 7-6 journal-style two-arm KM with numbers at risk and marks
rB = km(T[G == 0], S_[G == 0])
mA = quantile_ci(rA)[0]; mB = quantile_ci(rB)[0]
cx = cox_efron(T, S_, G); lr = logrank(T, S_, G, 1)
p = Plot((0, 60), (0, 1.0), w=640, h=460, ml=100, mr=24, mt=24, mb=150, xlabel="",
         ylabel="Overall survival (%)", xticks=[0, 12, 24, 36, 48, 60], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
         ytickfmt=lambda v: pct(v))
p.hline(0.5, dash=True, x1=mA)
p.vline(mA, y1=0.5, dash=True); p.vline(mB, y1=0.5, dash=True)
xa, ya = km_xy(rA, 60); xb, yb = km_xy(rB, 60)
p.step(xa, ya, s=1, w=2.2); p.step(xb, yb, s=2, w=2.2)
cxa, cya = censor_xy(rA, 60); cxb, cyb = censor_xy(rB, 60)
p.ticks_marks(cxa, cya, s=1, size=4.5); p.ticks_marks(cxb, cyb, s=2, size=4.5)
p.legend([("Drug A", 1, "line"), ("Drug B", 2, "line")], X=p.sx(40), Y=p.sy(0.93))
p.text(38.5, 0.74, f"HR {cx['hr']:.2f} (95% CI {cx['lo']:.2f}–{cx['hi']:.2f})", cls="lbl")
p.text(38.5, 0.74, f"Log-rank P = {lr['p']:.3f}", dy=18, cls="lbl")
Bq = p.h - p.mb
p.text_px((p.ml + p.w - p.mr) / 2, Bq + 36, "Months since treatment initiation", anchor="middle", cls="axlab")
grid = [0, 12, 24, 36, 48, 60]
risk_table(p, grid, [("Drug A", 1, [n_risk(T[G == 1], x) for x in grid]), ("Drug B", 2, [n_risk(T[G == 0], x) for x in grid])],
           64, title="No. at risk")
mark(p, 0, 1.0, 1, dx=-50, dy=-2)
mark(p, cxa[3], cya[3], 2, dx=0, dy=-18)
mark(p, mB, 0.5, 3, dx=-12, dy=-16)
mark(p, 0, 0, 4, dx=-p.ml + 100, dy=58)
mark(p, 38.5, 0.74, 5, dx=-14, dy=-5)
mark(p, 38.5, 0.74, 6, dx=-14, dy=13)
mark(p, 57.5, 0.07, 7)
save("ch07_paperkm", figure(p.svg("두 군의 Kaplan-Meier 곡선과 위험집합 표"),
     "그림 7-6. 학술지 형식의 Kaplan-Meier 그림(가상의 예시). 원문 범례: “Figure 2. Kaplan–Meier estimates of overall survival according to first-line treatment. Tick marks indicate censored observations. The hazard ratio was estimated with a Cox proportional-hazards model, and the P value was calculated with the log-rank test.”"))

# ======================================================================
# 7-7 two small groups
fA = km(t[g == 1], s[g == 1]); fB = km(t[g == 0], s[g == 0])
lr12 = logrank(t, s, g, 1)
p = Plot((0, 48), (0, 1.0), w=600, h=320, ml=62, mr=24, mt=20, mb=54, xlabel="치료 시작 후 개월 수",
         ylabel="생존 확률 S(t)", xticks=[0, 6, 12, 18, 24, 30, 36, 42, 48], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
         ytickfmt=lambda v: f"{v:.1f}")
xa, ya = km_xy(fA, 48); xb, yb = km_xy(fB, 48)
p.step(xa, ya, s=1, w=2.4); p.step(xb, yb, s=2, w=2.4)
c1 = censor_xy(fA, 48); c2 = censor_xy(fB, 48)
p.ticks_marks(*c1, s=1, size=6); p.ticks_marks(*c2, s=2, size=6)
p.text(44, 0.5556, "약물 A (6명)", anchor="end", dy=-10, cls="lbl strong")
p.text(20, 0.2083, "약물 B (6명)", dx=6, dy=-8, cls="lbl strong")
p.text(46, 0.95, f"로그순위 χ² = {lr12['chi2']:.2f}, P = {lr12['p']:.3f}", anchor="end", cls="lbl")
save("ch07_km2", figure(p.svg("약물 A와 B의 Kaplan-Meier 곡선 (각 6명)"),
     "그림 7-7. 12명을 치료 약물로 나눈 Kaplan-Meier 곡선. 약물 A 곡선은 0.5 아래로 내려가지 않으므로 중앙생존기간에 도달하지 않았습니다(not reached). 두 곡선이 꽤 벌어져 보이지만 환자가 12명뿐이라 로그순위 검정은 유의수준 5%에 미치지 못합니다."))

# ======================================================================
# 7-8 weights over time (pooled data of the crossing scenario)
tt, ss, gg = scenario("late", 23)
et = np.unique(tt[ss == 1])
nj = np.array([(tt >= x).sum() for x in et]); dj = np.array([((tt == x) & (ss == 1)).sum() for x in et])
Sl = np.cumprod(np.concatenate([[1.0], (1 - dj / nj)[:-1]]))
p = Plot((0, 48), (0, 1.05), w=600, h=300, ml=62, mr=24, mt=20, mb=54, xlabel="사건 시점 (개월)",
         ylabel="가중치 (처음 값 = 1)", xticks=[0, 6, 12, 18, 24, 30, 36, 42, 48], yticks=[0, 0.25, 0.5, 0.75, 1.0])
p.line([0, 48], [1, 1], s=4, dash=True, w=2)
p.line(et, nj / nj[0], s=3, w=2)
p.line(et, Sl, s=1, w=2)
p.line(et, 1 - Sl, s=2, w=2)
p.text(47, 1.0, "로그순위: 모든 시점 1", anchor="end", dy=-8, cls="lbl small")
p.text(38, 0.12, "Gehan-Breslow: 위험집합 크기 n", anchor="end", cls="lbl small")
p.text(6.5, 0.87, "Peto-Peto: S(t−)", cls="lbl small")
p.text(46, 0.74, "Fleming-Harrington(0,1): 1 − S(t−)", anchor="end", cls="lbl small")
save("ch07_weights", figure(p.svg("가중 로그순위 검정의 시점별 가중치"),
     "그림 7-8. 검정별로 각 사건 시점의 (O − E)에 곱하는 가중치(그림 7-9 (나) 자료). Gehan-Breslow와 Peto-Peto는 초반 사건에, Fleming-Harrington(0,1)은 후반 사건에 무게를 둡니다. 로그순위 검정은 모든 시점을 똑같이 취급합니다."))

# ======================================================================
# 7-9 early vs crossing scenarios
panels = []
for kind, seed, title in (("early", 70, "(가) 초기에만 차이"), ("late", 23, "(나) 곡선이 교차")):
    tt, ss, gg = scenario(kind, seed)
    fx = km(tt[gg == 1], ss[gg == 1]); fy = km(tt[gg == 0], ss[gg == 0])
    pl = Plot((0, 48), (0, 1.0), w=420, h=310, ml=52, mr=14, mt=26, mb=50, xlabel="개월", ylabel="생존 확률",
              xticks=[0, 12, 24, 36, 48], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
    x1, y1 = km_xy(fx, 48); x2, y2 = km_xy(fy, 48)
    pl.step(x2, y2, s=2, w=2); pl.step(x1, y1, s=1, w=2)
    P = {w: logrank(tt, ss, gg, 1, weight=w)["p"] for w in ("logrank", "peto", "fh01")}
    ytxt = 0.97
    pl.text(47, ytxt, f"로그순위 P = {P['logrank']:.3f}", anchor="end", cls="lbl small")
    pl.text(47, ytxt, f"Peto-Peto P = {P['peto']:.3f}", anchor="end", dy=16, cls="lbl small")
    pl.text(47, ytxt, f"FH(0,1) P = {P['fh01']:.3f}", anchor="end", dy=32, cls="lbl small")
    if kind == "early":
        pl.text(10, surv_at(fx, 10)[0], "치료 X", dx=-4, dy=18, anchor="middle", cls="lbl strong small")
        pl.text(8, surv_at(fy, 8)[0], "치료 Y", dy=-10, anchor="middle", cls="lbl strong small")
    else:
        pl.text(40, surv_at(fx, 40)[0], "치료 X", dy=-10, anchor="middle", cls="lbl strong small")
        pl.text(40, surv_at(fy, 40)[0], "치료 Y", dy=18, anchor="middle", cls="lbl strong small")
    panel_title(pl, title)
    panels.append(pl.svg(title))
save("ch07_weighted", figure(panels,
     "그림 7-9. 가상의 두 무작위배정 연구(군당 200명). (가) 치료 X에서 처음 3개월에 사망이 몰리고 그 뒤 위험은 같습니다. 그림에 표시한 세 검정 중에서는 초반에 무게를 두는 Peto-Peto 검정만 유의합니다. (나) 치료 X가 처음 4개월은 불리하고 그 뒤 유리해 곡선이 교차합니다. 후반에 무게를 두는 FH(0,1) 검정이 가장 강하게 차이를 잡고, Peto-Peto 검정은 차이를 놓칩니다.", cols=2))
print("figures written")
