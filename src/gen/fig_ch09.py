import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import math
import numpy as np
import scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt, forest
from lib_ch09 import expit, logit, fit_logit, rcs_basis
from nums_ch09 import compute

OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)
R = compute()


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def mark_xy(X, Y, n):
    return (f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
            f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def mark(p, x, y, n, dx=0, dy=0):
    p.top.append(mark_xy(p.sx(x) + dx, p.sy(y) + dy, n))


def dec(v, d=2):
    return f"{v:.{d}f}"


# ======================================================================
# 9-1 linear probability model vs logistic (vancomycin)
V = R["vanco"]
t, a = V["t"], V["a"]
rng = np.random.default_rng(3)
p = Plot((0, 45), (-0.3, 1.3), w=600, h=380, ml=62, mr=24, mt=20, mb=54,
         xlabel="반코마이신 최저혈중농도 (trough, mg/L)", ylabel="AKI 확률",
         xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45], yticks=[-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
         ytickfmt=lambda v: f"{v:.1f}")
p.fill_between([0, 45], [-0.3, -0.3], [0, 0], cls="a4")
p.fill_between([0, 45], [1.0, 1.0], [1.3, 1.3], cls="a4")
p.hline(0, dash=False, cls="ref"); p.hline(1, dash=False, cls="ref")
jit = rng.uniform(-0.035, 0.035, len(t))
p.points(t[a == 0], (0.06 + jit[a == 0]), s=4, r=3, hollow=True)
p.points(t[a == 1], (0.94 + jit[a == 1]), s=4, r=3, hollow=True)
# observed proportion in trough groups
edges = [4, 10, 15, 20, 25, 30, 45]
bx, by = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (t >= lo) & (t < hi)
    bx.append(t[m].mean()); by.append(a[m].mean())
xs = np.linspace(0, 45, 300)
p.line(xs, V["lpm"][0] + V["lpm"][1] * xs, s=2, w=2.2, dash=True)
p.line(xs, expit(V["b0"] + V["b1"] * xs), s=1, w=2.4)
p.points(bx, by, s=3, r=5.5)
p.text(44.2, -0.24, "확률이 될 수 없는 영역 (0 미만)", anchor="end", cls="lbl mute small")
p.text(0.8, 1.23, "확률이 될 수 없는 영역 (1 초과)", cls="lbl mute small")
p.text(t.min(), V["lpm_at_min"], f"{V['lpm_at_min']:.2f}", anchor="start", dx=8, dy=14, cls="lbl small")
p.text(V["tmax"], V["lpm_at_max"], f"{V['lpm_at_max']:.2f}", anchor="end", dx=-8, dy=-6, cls="lbl small")
p.points([V["tmax"]], [V["lpm_at_max"]], s=2, r=4)
p.points([t.min()], [V["lpm_at_min"]], s=2, r=4)
p.legend([("선형확률모형 (직선)", 2, "dash"), ("로지스틱 회귀 (S자 곡선)", 1, "line"), ("농도 구간별 실제 AKI 비율", 3, "dot"),
          ("환자 한 명 (위 = AKI, 아래 = 없음)", 4, "dot")], X=p.sx(0.6), Y=p.sy(0.84))
save("ch09_lpm", figure(p.svg("반코마이신 농도와 AKI: 직선과 로지스틱 곡선"),
     f"그림 9-1. 반코마이신 최저혈중농도와 급성 신손상(AKI) 자료 150명(가상). 회색 빈 점은 환자 한 명씩이며 AKI가 있으면 위, 없으면 아래에 찍었습니다(겹치지 않게 위아래로 조금 흩뜨림). 0과 1만 있는 결과에 직선을 맞추면(주황 점선) 농도가 {V['lpm_x0']:.1f} mg/L보다 낮은 {V['lpm_neg']}명의 예측값이 음수이고, {V['lpm_x1']:.1f} mg/L를 넘는 {V['lpm_gt1']}명은 1을 넘습니다. 로지스틱 곡선(파랑)은 항상 0과 1 사이에 있고, 농도 구간별 실제 비율(초록 점)을 잘 따라갑니다."))

# ======================================================================
# 9-2 probability scale vs logit scale
b0, b1 = V["b0"], V["b1"]
grid = [10, 15, 20, 25, 30]
pa = Plot((0, 40), (0, 1), w=420, h=320, ml=52, mr=14, mt=26, mb=50, xlabel="trough (mg/L)", ylabel="AKI 확률 p",
          xticks=[0, 10, 20, 30, 40], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
xs = np.linspace(0, 40, 300)
pa.line(xs, expit(b0 + b1 * xs), s=1, w=2.4)
for x in grid:
    pv = expit(b0 + b1 * x)
    pa.vline(x, y1=pv, dash=True, w=0.9)
for x0, x1 in zip(grid[:-1], grid[1:]):
    p0_, p1_ = expit(b0 + b1 * x0), expit(b0 + b1 * x1)
    pa.seg(x1, p0_, x1, p1_, cls="ln s2", w=2.4)
    pa.seg(x0, p0_, x1, p0_, cls="ref", w=0.9, dash=True)
    pa.text(x1, (p0_ + p1_) / 2, f"+{(p1_ - p0_) * 100:.1f}%p", dx=5, dy=4, cls="lbl small")
pa.points(grid, [expit(b0 + b1 * x) for x in grid], s=1, r=4.5)
panel_title(pa, "(가) 확률 척도: S자 곡선")
pb = Plot((0, 40), (-4.5, 3.5), w=420, h=320, ml=52, mr=14, mt=26, mb=50, xlabel="trough (mg/L)", ylabel="로그 오즈 = 로짓(p)",
          xticks=[0, 10, 20, 30, 40], yticks=[-4, -3, -2, -1, 0, 1, 2, 3])
pb.hline(0, dash=True)
pb.line(xs, b0 + b1 * xs, s=1, w=2.4)
for x0, x1 in zip(grid[:-1], grid[1:]):
    l0, l1 = b0 + b1 * x0, b0 + b1 * x1
    pb.seg(x0, l0, x1, l0, cls="ref", w=0.9, dash=True)
    pb.seg(x1, l0, x1, l1, cls="ln s2", w=2.4)
    pb.text(x1, (l0 + l1) / 2, f"+{l1 - l0:.2f}", dx=5, dy=4, cls="lbl small")
pb.points(grid, [b0 + b1 * x for x in grid], s=1, r=4.5)
pb.text(39, -3.0, f"5 mg/L마다 로그 오즈 +{5 * b1:.2f}", anchor="end", cls="lbl small")
pb.text(39, -3.0, f"= 오즈는 {np.exp(5 * b1):.2f}배", anchor="end", dy=16, cls="lbl small")
pb.text(40, 0, "p = 0.5", anchor="end", dy=-5, cls="lbl mute small")
panel_title(pb, "(나) 로짓 척도: 직선")
save("ch09_scales", figure([pa.svg("확률 척도의 로지스틱 곡선"), pb.svg("로짓 척도의 직선")],
     f"그림 9-2. 그림 9-1의 로지스틱 회귀 결과를 두 척도로 그린 것입니다. (나) 로짓 척도에서는 직선이므로 농도가 5 mg/L 높아질 때마다 로그 오즈가 언제나 {5 * b1:.2f}씩 올라갑니다(오즈비 {np.exp(5 * b1):.2f}로 일정). (가) 확률 척도에서는 같은 5 mg/L라도 어느 구간이냐에 따라 확률 증가폭(주황 세로선)이 {(expit(b0 + 15 * b1) - expit(b0 + 10 * b1)) * 100:.1f}%p에서 {(expit(b0 + 25 * b1) - expit(b0 + 20 * b1)) * 100:.1f}%p까지 달라집니다.", cols=2))

# ======================================================================
# 9-3 profile log-likelihood for beta1
X = np.column_stack([np.ones(len(t)), t])
bs = np.linspace(0.06, 0.34, 141)
prof = []
for bb in bs:
    r = fit_logit(np.ones((len(t), 1)), a, offset=bb * t, beta0=[b0])
    prof.append(r["ll"] - V["ll"])
prof = np.array(prof)
se1 = V["se1"]
p = Plot((0.06, 0.34), (-6.2, 0.6), w=600, h=340, ml=62, mr=24, mt=20, mb=54,
         xlabel="β₁ (trough 1 mg/L당 로그 오즈비)", ylabel="로그 우도 − 최댓값",
         xticks=[0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32], xtickfmt=lambda v: f"{v:.2f}", yticks=[-6, -5, -4, -3, -2, -1, 0])
p.line(bs, -(bs - b1) ** 2 / (2 * se1 ** 2), s=2, dash=True, w=2)
p.line(bs, prof, s=1, w=2.4)
crit = st.chi2.ppf(0.95, 1) / 2
p.hline(-crit, dash=True)
p.text(0.064, -crit, f"최댓값 − {crit:.2f}", anchor="start", dy=-6, cls="lbl mute small")
lo, hi = V["prof_ci"]
wl, wh = V["wald_ci"]
for v in (lo, hi):
    p.vline(v, y0=-6.2, y1=-crit, dash=False, cls="ln s1", w=1.4)
for v in (wl, wh):
    p.vline(v, y0=-6.2, y1=-crit, dash=True, cls="ln s2", w=1.2)
p.vline(b1, y0=-crit, y1=0, dash=True)
p.text(b1, 0, f"β̂₁ = {b1:.3f}", anchor="middle", dy=-8, cls="lbl strong")
p.text(lo, -5.3, f"{lo:.3f}", anchor="start", dx=4, cls="lbl small")
p.text(hi, -5.3, f"{hi:.3f}", anchor="start", dx=4, cls="lbl small")
p.text(wl, -5.8, f"{wl:.3f}", anchor="end", dx=-4, cls="lbl small")
p.text(wh, -5.8, f"{wh:.3f}", anchor="end", dx=-4, cls="lbl small")
p.legend([("프로파일 로그 우도 (실제 모양)", 1, "line"), ("Wald 근사 (포물선)", 2, "dash")], X=p.sx(0.136), Y=p.sy(-3.2))
save("ch09_loglik", figure(p.svg("β1에 대한 프로파일 로그 우도"),
     f"그림 9-4. 반코마이신 자료에서 β₁ 값을 바꿔 가며(β₀는 그때마다 다시 최적화) 계산한 로그 우도. 꼭대기가 최대우도추정값 {b1:.3f}입니다. 로그 우도가 꼭대기보다 1.92 낮아지는 두 점이 프로파일 우도 95% 신뢰구간({lo:.3f}–{hi:.3f}, 파란 실선)이고, 꼭대기의 곡률로 그린 포물선(주황 점선)에서 같은 방식으로 읽은 것이 Wald 신뢰구간({wl:.3f}–{wh:.3f}, 주황 점선)입니다. 실제 곡선이 오른쪽으로 조금 더 완만해 두 구간이 약간 어긋납니다."))

# ======================================================================
# 9-4 forest plot of adjusted ORs (paper box)
E = R["etio"]
ad = E["adj"]
rows = [
    dict(label="SGLT2 inhibitor (vs DPP-4 inhibitor)", est=ad["sglt2"]["orci"][0], lo=ad["sglt2"]["orci"][1], hi=ad["sglt2"]["orci"][2], bold=True),
    dict(label="Age, per 10 years", est=ad["age10"]["orci"][0], lo=ad["age10"]["orci"][1], hi=ad["age10"]["orci"][2]),
    dict(label="Female (vs male)", est=ad["female"]["orci"][0], lo=ad["female"]["orci"][1], hi=ad["female"]["orci"][2]),
    dict(label="Charlson comorbidity index", header=True),
    dict(label="0 (reference)", est=None, indent=True),
    dict(label="1–2", est=ad["cci12"]["orci"][0], lo=ad["cci12"]["orci"][1], hi=ad["cci12"]["orci"][2], indent=True),
    dict(label="≥3", est=ad["cci3"]["orci"][0], lo=ad["cci3"]["orci"][1], hi=ad["cci3"]["orci"][2], indent=True),
    dict(label="Prior hospitalization", est=ad["prior"]["orci"][0], lo=ad["prior"]["orci"][1], hi=ad["prior"]["orci"][2]),
    dict(label="Heart failure", est=ad["hf"]["orci"][0], lo=ad["hf"]["orci"][1], hi=ad["hf"]["orci"][2]),
    dict(label="eGFR, mL/min/1.73 m²", header=True),
    dict(label="≥60 (reference)", est=None, indent=True),
    dict(label="45–59", est=ad["egfr4559"]["orci"][0], lo=ad["egfr4559"]["orci"][1], hi=ad["egfr4559"]["orci"][2], indent=True),
    dict(label="<45", est=ad["egfr45"]["orci"][0], lo=ad["egfr45"]["orci"][1], hi=ad["egfr45"]["orci"][2], indent=True),
]
W, rh, lw, ew = 660, 30, 250, 140
svg = forest(rows, (0.4, 5), ref=1.0, log=True, w=W, row_h=rh, label_w=lw, est_w=ew,
             xlabel="Adjusted odds ratio (95% CI, log scale)",
             xticks=[0.5, 1, 2, 4], header=("Variable", "Adjusted OR (95% CI)"))
top_pad = 34
X0, X1 = lw, W - ew
sxf = lambda v: X0 + (math.log(v) - math.log(0.4)) / (math.log(5) - math.log(0.4)) * (X1 - X0)
ry = lambda i: top_pad + rh * i + rh / 2
yb = top_pad + rh * len(rows) + 6
extra = (f'<text x="{sxf(1.0) - 7:.1f}" y="18" text-anchor="end" class="lbl small">← Lower odds</text>'
         f'<text x="{sxf(1.0) + 7:.1f}" y="18" class="lbl small">Higher odds →</text>')
mk = [mark_xy(sxf(1.0) - 14, ry(10), 1),
      mark_xy(sxf(ad["sglt2"]["orci"][2]) + 16, ry(0), 2),
      mark_xy(sxf(4) + 24, yb + 14, 3),
      mark_xy(118, ry(4), 4),
      mark_xy(158, ry(1), 5),
      mark_xy(sxf(ad["prior"]["orci"][2]) + 16, ry(7), 6)]
svg = svg.replace("</svg>", extra + "".join(mk) + "</svg>")
save("ch09_forest", figure(svg,
     "그림 9-8. 학술지 형식의 포레스트 플롯(가상의 예시). 원문 범례: “Figure 2. Adjusted odds ratios for 1-year all-cause hospitalization. Odds ratios were estimated with a multivariable logistic regression model including all variables shown. Squares indicate point estimates and horizontal lines indicate 95% confidence intervals; the x-axis is on a logarithmic scale.”"))

# ======================================================================
# 9-5 eGFR linearity
L = R["lin"]
cv = L["curve"]
p = Plot((15, 120), (math.log(0.45), math.log(6)), w=600, h=360, ml=62, mr=24, mt=20, mb=54,
         xlabel="eGFR (mL/min/1.73 m²)", ylabel="오즈비 (기준: eGFR 90, 로그 척도)",
         xticks=[15, 30, 45, 60, 75, 90, 105, 120], yticks=[math.log(v) for v in (0.5, 1, 2, 4)],
         ytickfmt=lambda v: fmt(round(math.exp(v), 2)))
p.fill_between(cv[:, 0], cv[:, 2], cv[:, 3], s=1)
p.hline(0, dash=False, cls="ref")
p.line(cv[:, 0], cv[:, 1], s=1, w=2.4)
xs = np.linspace(15, 120, 50)
p.line(xs, -L["lin_b"] * (90 - xs), s=2, dash=True, w=2)
# categories relative to >=60
c1, c2 = ad["egfr4559"]["b"], ad["egfr45"]["b"]
p.step([15, 45, 60, 120], [c2, c1, 0, 0], s=3, w=2.2)
for k in L["knots"]:
    p.seg(k, math.log(0.45), k, math.log(0.45) + 0.08, cls="ln s4", w=2)
p.text(L["knots"][0], math.log(0.45) + 0.1, "매듭", anchor="middle", dy=-4, cls="lbl mute small")
p.text_px(p.sx(62) + 30, p.sy(math.log(5.2)) + 3 * 18 + 4.5, f"범주별 오즈비: 45–59 {np.exp(c1):.2f}, <45 {np.exp(c2):.2f}", cls="lbl small")
p.legend([("제한 삼차 스플라인 (4개 매듭) + 95% CI", 1, "line"), ("직선 (eGFR 10 낮을 때마다 오즈비 %.2f)" % L["lin_or10"], 2, "dash"),
          ("세 범주 (기준 ≥60)", 3, "line")], X=p.sx(62), Y=p.sy(math.log(5.2)))
save("ch09_egfr", figure(p.svg("eGFR과 입원 로그 오즈의 관계"),
     f"그림 9-5. 다른 변수를 보정한 모형에서 eGFR과 1년 입원의 관계를 세 가지 방식으로 넣은 결과. 스플라인(파랑)은 eGFR 60 이상에서는 거의 평평하다가 60 아래에서 가파르게 올라갑니다. 직선(주황 점선)은 이 꺾임을 담지 못해 eGFR 60–90 구간의 차이는 실제보다 크게, 45 미만의 위험은 작게 보여 주고, 90을 넘으면 위험이 계속 줄어드는 것처럼 그립니다. 세 범주(초록 계단)는 꺾임을 대략 담지만 범주 안의 차이는 무시합니다. 비선형성 우도비 검정 P = {L['lr'][1]:.3f}. 아래쪽 회색 눈금은 스플라인의 매듭(eGFR의 5·35·65·95 백분위수 {', '.join(str(int(k)) for k in L['knots'])})입니다."))

# ======================================================================
# 9-6 OR vs RR
RR = R["rr"]
labs = [("sglt2", "SGLT2 억제제 (vs DPP-4 억제제)"), ("age10", "나이 10세 증가"), ("female", "여성"), ("cci12", "CCI 1–2 (vs 0)"),
        ("cci3", "CCI ≥3 (vs 0)"), ("prior", "이전 1년 입원"), ("hf", "심부전"), ("egfr4559", "eGFR 45–59 (vs ≥60)"),
        ("egfr45", "eGFR <45 (vs ≥60)")]
nr = len(labs)
p = Plot((math.log(0.5), math.log(4)), (0, nr + 0.6), w=620, h=70 + 34 * nr + 60, ml=200, mr=20, mt=46, mb=54,
         xlabel="비 (로그 척도)", xticks=[math.log(v) for v in (0.5, 1, 2, 4)], xtickfmt=lambda v: fmt(round(math.exp(v), 2)),
         ygrid=False, xgrid=True, show_yaxis=False)
p.vline(0, dash=False, cls="ref strongref", w=1.4)
for i, (k, lab) in enumerate(labs):
    yc = nr - i - 0.2
    o = ad[k]["orci"]; r_ = RR["mp"][k]["rrci"]
    p.seg(math.log(o[1]), yc + 0.17, math.log(o[2]), yc + 0.17, cls="ln s2", w=2)
    p.points([math.log(o[0])], [yc + 0.17], s=2, r=5)
    p.seg(math.log(r_[1]), yc - 0.17, math.log(r_[2]), yc - 0.17, cls="ln s1", w=2)
    p.points([math.log(r_[0])], [yc - 0.17], s=1, r=5)
    p.text_px(8, p.sy(yc) + 4.5, lab, cls="lbl")
p.legend([("오즈비 (로지스틱 회귀)", 2, "dot")], X=200, Y=18)
p.legend([("상대위험도 (수정 포아송 회귀)", 1, "dot")], X=392, Y=18)
save("ch09_orrr", figure(p.svg("같은 자료의 보정 오즈비와 보정 상대위험도"),
     f"그림 9-3. 같은 코호트(1년 입원 {R['desc']['ev'] / R['desc']['n'] * 100:.1f}%)에 같은 변수를 넣고 로지스틱 회귀로 구한 보정 오즈비(주황)와 수정 포아송 회귀로 구한 보정 상대위험도(파랑). 모든 변수에서 오즈비가 상대위험도보다 1에서 더 멀리 있고, 효과가 큰 변수일수록 차이가 큽니다(이전 1년 입원: 오즈비 {ad['prior']['orci'][0]:.2f}, 상대위험도 {RR['mp']['prior']['rrci'][0]:.2f})."))

# ======================================================================
# 9-7 predicted-probability distributions (mirrored)
P = R["pred"]
ph, yv = P["p"], P["y"]
edges = np.arange(0, 0.84, 0.04)
h1, _ = np.histogram(ph[yv == 1], edges); h0, _ = np.histogram(ph[yv == 0], edges)
f1 = h1 / h1.sum() * 100; f0 = h0 / h0.sum() * 100
ymax = 22
p = Plot((0, 0.8), (-ymax, ymax), w=600, h=380, ml=62, mr=24, mt=20, mb=54,
         xlabel="모형이 예측한 1년 입원 확률", ylabel="각 군 안에서의 비율 (%)",
         xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], xtickfmt=lambda v: f"{v:.1f}",
         yticks=[-20, -10, 0, 10, 20], ytickfmt=lambda v: fmt(abs(v)))
for i in range(len(f1)):
    xa, xb = edges[i], edges[i + 1]
    if f1[i] > 0:
        p.fill_between([xa + 0.002, xb - 0.002], [0, 0], [f1[i], f1[i]], cls="f2")
    if f0[i] > 0:
        p.fill_between([xa + 0.002, xb - 0.002], [-f0[i], -f0[i]], [0, 0], cls="f1")
p.hline(0, dash=False, cls="axis")
for tv, lab in ((0.2, "임계값 0.2"), (0.5, "임계값 0.5")):
    p.vline(tv, dash=True, cls="ref strongref", w=1.3)
    p.text(tv, ymax, lab, dx=5, dy=12, cls="lbl small")
p.text(0.79, 15, f"입원한 환자 {int(yv.sum()):,}명", anchor="end", cls="lbl strong")
p.text(0.79, 15, f"평균 예측확률 {P['p_by'][0] * 100:.1f}%", anchor="end", dy=17, cls="lbl small")
p.text(0.79, -12, f"입원하지 않은 환자 {int((1 - yv).sum()):,}명", anchor="end", cls="lbl strong")
p.text(0.79, -12, f"평균 예측확률 {P['p_by'][1] * 100:.1f}%", anchor="end", dy=17, cls="lbl small")
save("ch09_predhist", figure(p.svg("입원 여부별 예측확률 분포"),
     f"그림 9-6. 예측모형이 계산한 1년 입원 확률의 분포. 위(주황)는 실제로 입원한 환자, 아래(파랑)는 입원하지 않은 환자이며, 각 군 안에서의 비율로 그렸습니다. 입원한 환자의 예측확률이 전체적으로 오른쪽에 있지만 두 분포가 많이 겹칩니다. C 통계량 {P['auc'][0]:.2f}는 이 겹침의 정도를 한 숫자로 요약한 것이고, 임계값을 어디에 두느냐에 따라 민감도와 특이도가 달라집니다."))

# ======================================================================
# 9-8 journal-style ROC + calibration (paper box)
X_ = R["ext"]
fpr, tpr = P["roc"]
efpr, etpr, _ = X_["roc"]
pa = Plot((0, 1), (0, 1), w=420, h=380, ml=56, mr=14, mt=26, mb=50, xlabel="1 − Specificity", ylabel="Sensitivity",
          xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
          xtickfmt=lambda v: f"{v:.1f}", ytickfmt=lambda v: f"{v:.1f}", xgrid=True)
pa.seg(0, 0, 1, 1, cls="ref", dash=True)
pa.line(fpr, tpr, s=1, w=2.2)
pa.line(efpr, etpr, s=2, w=2.2)
c2 = P["thr"][0.2]
ce2 = X_["thr"][0.2]
pa.points([1 - c2["spec"]], [c2["sens"]], s=1, r=5)
pa.points([1 - ce2["spec"]], [ce2["sens"]], s=2, r=5)
pa.text(0.40, 0.20, f"Development: C = {P['auc'][0]:.2f} ({P['auc'][2]:.2f}–{P['auc'][3]:.2f})", cls="lbl small")
pa.text(0.40, 0.20, f"External: C = {X_['auc'][0]:.2f} ({X_['auc'][2]:.2f}–{X_['auc'][3]:.2f})", dy=17, cls="lbl small")
pa.legend([("Development", 1, "line"), ("External validation", 2, "line")], X=pa.sx(0.43), Y=pa.sy(0.44))
panel_title(pa, "A. Discrimination")
mark(pa, 0.40, 0.20, 1, dx=-14, dy=-4)
mark(pa, 0.62, 0.62, 2, dx=10, dy=-6)
mark(pa, 1 - c2["spec"], c2["sens"], 3, dx=-14, dy=-12)

pb = Plot((0, 0.6), (0, 0.6), w=420, h=380, ml=56, mr=14, mt=26, mb=50, xlabel="Predicted probability", ylabel="Observed proportion",
          xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
          xtickfmt=lambda v: f"{v:.1f}", ytickfmt=lambda v: f"{v:.1f}", xgrid=True)
pb.seg(0, 0, 0.6, 0.6, cls="ref", dash=True)


def wilson(k, n):
    z = 1.96
    ph_ = k / n
    den = 1 + z * z / n
    c = (ph_ + z * z / (2 * n)) / den
    h = z * math.sqrt(ph_ * (1 - ph_) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


for rows_, s_ in ((P["hl"][3], 1), (X_["hl"][3], 2)):
    for r in rows_:
        lo_, hi_ = wilson(r["obs"], r["n"])
        pb.seg(r["mean_p"], lo_, r["mean_p"], min(hi_, 0.6), cls=f"ln s{s_}", w=1.2)
    pb.points([r["mean_p"] for r in rows_], [r["obs_rate"] for r in rows_], s=s_, r=4.5)
gx, gy = P["calcurve"]
pb.line(gx, gy, s=1, w=1.6, dash=True)
gx, gy = X_["calcurve"]
pb.line(gx, gy, s=2, w=1.6, dash=True)
pb.text(0.012, 0.565, f"Development: slope {P['slope_corr']:.2f}*, H–L P = {P['hl'][2]:.2f}", cls="lbl small")
pb.text(0.012, 0.565, f"External: intercept {X_['cal']['citl']:.2f}, slope {X_['cal']['slope']:.2f}", dy=17, cls="lbl small")
pb.text(0.59, 0.03, "*bootstrap optimism-corrected", anchor="end", cls="lbl mute small")
panel_title(pb, "B. Calibration")
last_d = P["hl"][3][-1]; last_e = X_["hl"][3][-1]
mark(pb, last_d["mean_p"], last_d["obs_rate"], 4, dx=-16, dy=-12)
mark(pb, last_e["mean_p"], last_e["obs_rate"], 5, dx=14, dy=10)
mark(pb, 0.012, 0.565, 6, dx=-12, dy=-5)
save("ch09_paperroc", figure([pa.svg("ROC 곡선: 개발 자료와 외부 검증 자료"), pb.svg("교정 그림: 개발 자료와 외부 검증 자료")],
     "그림 9-7. 예측모형 논문의 전형적인 성능 그림(가상의 예시). 원문 범례: “Figure 3. Performance of the prediction model in the development cohort (blue) and the external validation cohort (orange). (A) Receiver operating characteristic curves; dots indicate a risk threshold of 0.2. (B) Calibration plots; dots show the mean predicted probability and the observed proportion (95% CI) in each decile of predicted risk, and dashed lines are smoothed calibration curves. The diagonal line represents perfect calibration.”", cols=2))

print("figures written:", [f for f in sorted(os.listdir(OUT)) if f.startswith("ch09_")])

# ======================================================================
# widget: threshold and prevalence (writes widgets/ch09.js from the same predicted probabilities)
import json
kb1 = np.bincount(np.floor(ph[yv == 1] * 100).astype(int), minlength=100)[:100]
kb0 = np.bincount(np.floor(ph[yv == 0] * 100).astype(int), minlength=100)[:100]
assert kb1.sum() == int(yv.sum()) and kb0.sum() == int((1 - yv).sum())
c20 = P["thr"][0.2]
assert int(kb1[20:].sum()) == c20["TP"] and int(kb0[20:].sum()) == c20["FP"]
WJS = r"""// generated by gen/fig_ch09.py — predicted probabilities of the chapter 9 prediction model (2,400 patients)
window.EXTRA_WIDGETS["ch09_thr"] = function (el, H) {
  const C1 = __C1__, C0 = __C0__;          // patients per 0.01-wide bin of predicted risk (hospitalized / not)
  const N1 = C1.reduce((a, b) => a + b, 0), N0 = C0.reduce((a, b) => a + b, 0);
  const PREV0 = N1 / (N1 + N0), AUC = __AUC__;
  let thr = 20, prev = "cohort";
  const f1 = v => (Math.round(v * 1000 + 1e-7) / 10).toFixed(1) + "%";
  const n0 = v => Math.round(v).toLocaleString("en-US");
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">계산기</span></p>
    <p class="wd">본문 예측모형의 예측확률(환자 2,400명)입니다. 임계값을 옮기면 "고위험"으로 분류되는 환자가 바뀌어 민감도·특이도·정확도가 달라집니다. 사건 비율을 바꾸면 같은 모형(같은 C 통계량)이라도 양성예측도와 정확도가 어떻게 변하는지 볼 수 있습니다.</p>
    <div class="wrow"><label for="c9thr">임계값 <b id="c9thrv"></b></label>
      <input type="range" id="c9thr" min="5" max="60" step="1" value="20" style="flex:1;min-width:180px"></div>
    <div class="wrow"><span>1년 입원 비율</span><div class="seg" role="group" aria-label="사건 비율">
      <button type="button" data-p="cohort" aria-pressed="true">이 코호트 (${f1(PREV0)})</button>
      <button type="button" data-p="0.05" aria-pressed="false">5%</button>
      <button type="button" data-p="0.5" aria-pressed="false">50%</button></div></div>
    <div class="wchart" id="c9chart"></div>
    <div class="stats">
      <div class="stat"><div class="sl">민감도</div><div class="sv" id="c9se"></div></div>
      <div class="stat"><div class="sl">특이도</div><div class="sv" id="c9sp"></div></div>
      <div class="stat"><div class="sl">양성예측도</div><div class="sv" id="c9ppv"></div></div>
      <div class="stat"><div class="sl">음성예측도</div><div class="sv" id="c9npv"></div></div>
      <div class="stat"><div class="sl">정확도</div><div class="sv" id="c9acc"></div></div>
      <div class="stat"><div class="sl">전원 "입원 안 함" 예측의 정확도</div><div class="sv" id="c9none"></div></div>
      <div class="stat"><div class="sl">C 통계량 (임계값과 무관)</div><div class="sv">${AUC.toFixed(2)}</div></div>
    </div>
    <p class="wd" id="c9note" style="margin-top:10px"></p>`;
  function draw() {
    const TP = C1.slice(thr).reduce((a, b) => a + b, 0), FP = C0.slice(thr).reduce((a, b) => a + b, 0);
    const se = TP / N1, sp = 1 - FP / N0;
    const pr = prev === "cohort" ? PREV0 : +prev;
    const ppv = se * pr / (se * pr + (1 - sp) * (1 - pr)), npv = sp * (1 - pr) / (sp * (1 - pr) + (1 - se) * pr);
    const acc = se * pr + sp * (1 - pr);
    el.querySelector("#c9thrv").textContent = (thr / 100).toFixed(2);
    el.querySelector("#c9se").textContent = f1(se);
    el.querySelector("#c9sp").textContent = f1(sp);
    el.querySelector("#c9ppv").textContent = f1(ppv);
    el.querySelector("#c9npv").textContent = f1(npv);
    el.querySelector("#c9acc").textContent = f1(acc);
    el.querySelector("#c9none").textContent = f1(1 - pr);
    const per = prev === "cohort"
      ? `이 코호트 2,400명 중 ${n0(TP + FP)}명을 고위험으로 분류하고, 그중 ${n0(TP)}명이 실제로 입원했습니다(입원한 ${n0(N1)}명 중 ${n0(N1 - TP)}명은 놓침).`
      : `입원 비율이 ${f1(pr)}인 환자 1,000명이라면 ${n0(1000 * (se * pr + (1 - sp) * (1 - pr)))}명을 고위험으로 분류하고, 그중 ${n0(1000 * se * pr)}명이 실제로 입원합니다.`;
    el.querySelector("#c9note").textContent = per + (acc < 1 - pr ? " 지금 임계값의 정확도는 아무도 입원하지 않는다고 예측할 때보다 낮습니다." : "");
    // chart: mirrored histogram of predicted risk, 0.02 bins, % within each group
    const W = 640, Hh = 300, L = 56, R = 16, T = 14, B = 50, xmax = 0.8, ymax = 12;
    const sx = x => L + x / xmax * (W - L - R), mid = (Hh - B + T) / 2, sy = v => mid - v / ymax * (mid - T);
    let s = "";
    for (const v of [-10, -5, 0, 5, 10]) s += `<line x1="${L}" y1="${sy(v)}" x2="${W - R}" y2="${sy(v)}" class="grid"/><text x="${L - 8}" y="${sy(v) + 4}" text-anchor="end" class="tick">${Math.abs(v)}</text>`;
    for (let k = 0; k < 40; k++) {
      const a = C1[2 * k] + C1[2 * k + 1], b = C0[2 * k] + C0[2 * k + 1];
      const x0 = sx(k * 0.02) + 1, w = sx(0.02) - sx(0) - 2, on = 2 * k >= thr;
      if (a) s += `<rect x="${x0}" y="${sy(a / N1 * 100)}" width="${w}" height="${sy(0) - sy(a / N1 * 100)}" class="f2" opacity="${on ? 1 : 0.35}"/>`;
      if (b) s += `<rect x="${x0}" y="${sy(0)}" width="${w}" height="${sy(-b / N0 * 100) - sy(0)}" class="f1" opacity="${on ? 1 : 0.35}"/>`;
    }
    s += `<line x1="${L}" y1="${sy(0)}" x2="${W - R}" y2="${sy(0)}" class="axis"/>`;
    for (const x of [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) s += `<text x="${sx(x)}" y="${Hh - B + 18}" text-anchor="middle" class="tick">${x.toFixed(1)}</text>`;
    s += `<text x="${(L + W - R) / 2}" y="${Hh - 10}" text-anchor="middle" class="axlab">예측확률 (진하게 칠한 막대 = 고위험으로 분류)</text>`;
    s += `<line x1="${sx(thr / 100)}" y1="${T}" x2="${sx(thr / 100)}" y2="${Hh - B}" class="ref strongref" stroke-width="1.6" stroke-dasharray="5 4"/>`;
    s += `<text x="${W - R - 4}" y="${sy(9)}" text-anchor="end" class="lbl strong">입원한 환자</text>`;
    s += `<text x="${W - R - 4}" y="${sy(-9)}" text-anchor="end" class="lbl strong">입원하지 않은 환자</text>`;
    s += `<text x="${sx(thr / 100) + 6}" y="${T + 12}" class="lbl small">임계값 ${(thr / 100).toFixed(2)}</text>`;
    el.querySelector("#c9chart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="입원 여부별 예측확률 분포와 임계값">${s}</svg>`;
  }
  el.querySelector("#c9thr").addEventListener("input", e => { thr = +e.target.value; draw(); });
  el.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => {
    prev = b.dataset.p; el.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", x === b)); draw();
  }));
  draw();
};
"""
WJS = (WJS.replace("__C1__", json.dumps([int(v) for v in kb1])).replace("__C0__", json.dumps([int(v) for v in kb0]))
       .replace("__AUC__", f"{P['auc'][0]:.4f}"))
with open(os.path.join(HERE, "..", "widgets", "ch09.js"), "w", encoding="utf-8") as f:
    f.write(WJS)
print("widget written")
