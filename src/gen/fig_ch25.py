"""Figures for chapter 25 (경제성 평가 논문 읽기).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch25.py && python3 gen/fig_ch25.py
Writes figs/ch25_extrap.html, ch25_papertornado.html, ch25_paperpsa.html, ch25_bia.html
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from svgplot import Plot, figure, panel_title
import lib_p4 as L

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch25_nums.json"), encoding="utf-8"))
won = lambda v: f"{v:,.0f}".replace("-", "−")
pct = lambda v: f"{v * 100:.1f}%"


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


# ====================================================================== 그림 25-1: 관찰 기간과 외삽 구간
E = N["extrap"]
FU = E["fu"]
p0 = L.base_params()
t = np.arange(0, 241)
sA, sB = L.curves(p0, "A", t)[1], L.curves(p0, "B", t)[1]
p1 = Plot((0, 240), (0, 1.0), w=430, h=330, ml=54, mr=20, mt=30, mb=48, xlabel="치료 시작 후 개월", ylabel="전체생존율",
          xticks=[0, 30, 60, 120, 180, 240], yticks=[0, 0.25, 0.5, 0.75, 1.0], ytickfmt=lambda v: f"{v * 100:.0f}%")
p1.fill_between([0, FU], [0, 0], [1, 1], cls="a4")
p1.vline(FU, dash=True)
p1.line(list(t), list(sA), s=1, w=2.4)
p1.line(list(t), list(sB), s=2, w=2.4)
p1.points([FU], [E["sA30"]], s=1, r=4)
p1.points([FU], [E["sB30"]], s=2, r=4)
p1.legend([(f"신약 A (30개월 생존율 {E['sA30'] * 100:.0f}%)", 1, "line"), (f"표준요법 B (30개월 생존율 {E['sB30'] * 100:.0f}%)", 2, "line")], X=p1.sx(78), Y=p1.sy(0.97))
p1.text(FU, 0.70, "← 시험이 관찰한 30개월", dx=6, cls="lbl small mute")
p1.text(140, 0.45, "모형이 외삽한 구간", anchor="middle", cls="lbl small strong")
p1.text(140, 0.45, "(30개월부터 20년까지)", anchor="middle", dy=15, cls="lbl small mute")
panel_title(p1, "(가) 전체생존 곡선")

hs, cum = np.array(E["hs"]), np.array(E["cum"])
tot = N["base"]["d_qaly"]
p2 = Plot((0, 240), (0, 0.6), w=430, h=330, ml=54, mr=20, mt=30, mb=48, xlabel="분석기간을 이 시점에서 끊는다면 (개월)", ylabel="증분 QALY (누적)",
          xticks=[0, 30, 60, 120, 180, 240], yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ytickfmt=lambda v: f"{v:.1f}")
p2.fill_between([0, FU], [0, 0], [0.6, 0.6], cls="a4")
p2.vline(FU, dash=True)
p2.hline(tot, dash=True)
p2.line(list(hs), list(np.clip(cum, 0, None)), s=1, w=2.4)
p2.points([FU], [E["dq30"]], s=1, r=4.5)
p2.text(FU, E["dq30"], f"30개월까지 {E['dq30']:.3f}", dx=9, dy=5, cls="lbl small strong")
p2.text(FU, E["dq30"], f"(전체의 {(1 - E['share_q']) * 100:.0f}%)", dx=9, dy=20, cls="lbl small")
p2.points([240], [tot], s=1, r=4.5)
p2.text(236, tot, f"20년 {tot:.3f}", anchor="end", dy=-9, cls="lbl small strong")
p2.text(135, 0.25, f"30개월 뒤에 쌓이는 몫 {tot - E['dq30']:.3f}", anchor="middle", cls="lbl small strong")
p2.text(135, 0.25, f"(전체의 {E['share_q'] * 100:.0f}%)", anchor="middle", dy=15, cls="lbl small")
panel_title(p2, "(나) 증분 QALY가 쌓이는 시기")
save("ch25_extrap", figure(
    [p1.svg("신약 A와 표준요법 B의 전체생존 곡선, 시험 관찰 기간 30개월과 외삽 구간"),
     p2.svg("분석기간에 따른 누적 증분 QALY, 30개월까지와 그 뒤의 몫")],
    f"그림 25-1. 관찰한 구간과 외삽한 구간(가상의 예시). 회색 띠가 시험이 관찰한 30개월입니다. (가) 그 뒤의 곡선은 모형이 그린 것입니다. "
    f"(나) 증분 QALY {tot:.3f} 가운데 30개월까지 쌓이는 것은 {E['dq30']:.3f}이고, 나머지 {E['share_q'] * 100:.0f}%는 시험이 관찰하지 못한 기간에서 계산됩니다.",
    cols=2))

# ====================================================================== 그림 25-2: 학술지 형식의 토네이도 그림
EN = {"hr_os": "HR for overall survival", "hr_pfs": "HR for progression-free survival", "c_drug_A": "Monthly price of drug A",
      "c_drug_B": "Monthly price of standard therapy B", "u_pf": "Utility, progression-free", "os_gam": "Weibull shape, overall survival",
      "c_pf": "Cost of progression-free state, per month"}


def rng_txt(r):
    if r["key"].startswith("c_"):
        return f"₩{r['low'] / 100:.2f}–{r['high'] / 100:.2f} million"
    return f"{r['low']:.2f}–{r['high']:.2f}"


rows = N["oneway"]
n = len(rows)
base = N["base"]["icer"] / 100
p = Plot((35, 85), (0, n), w=660, h=58 + 40 * n + 50, ml=236, mr=22, mt=58, mb=50, xlabel="ICER (₩ million per QALY gained)",
         xticks=[40, 50, 60, 70, 80], xtickfmt=lambda v: f"{v:.0f}", ygrid=False, xgrid=True, show_yaxis=False)
for i, r in enumerate(rows):
    yc = n - i - 0.5
    for end, s in (("low", 1), ("high", 2)):
        a, b = sorted((base, r["icer_" + end] / 100))
        if b - a < 1e-9:
            continue
        xa, xb = p.sx(a), p.sx(b)
        p.els.append(f'<rect x="{xa:.1f}" y="{p.sy(yc + 0.30):.1f}" width="{xb - xa:.1f}" height="{p.sy(yc - 0.30) - p.sy(yc + 0.30):.1f}" class="f{s}"/>')
    p.text_px(p.ml - 10, p.sy(yc) - 2, EN[r["key"]], anchor="end", cls="lbl small")
    p.text_px(p.ml - 10, p.sy(yc) + 12, f"({rng_txt(r)})", anchor="end", cls="lbl small mute")
    left, right = sorted((r["icer_low"] / 100, r["icer_high"] / 100))
    if abs(left - base) > 1e-6:
        p.text(left, yc, f"{left:.1f}", anchor="end", dx=-5, dy=4.5, cls="lbl small")
    if abs(right - base) > 1e-6:
        p.text(right, yc, f"{right:.1f}", anchor="start", dx=5, dy=4.5, cls="lbl small")
p.vline(base, cls="strongref", dash=False, w=1.4)
p.vline(50, dash=True)
p.text(base, n, f"Base case {base:.1f}", dx=5, dy=-8, cls="lbl small strong")
p.text(50, n, "Threshold 50", anchor="end", dx=-5, dy=-8, cls="lbl small mute")
p.legend([("Lower value of the parameter", 1, "box"), ("Upper value of the parameter", 2, "box")], X=p.ml + 170, Y=12, gap=16)
save("ch25_papertornado", figure(
    p.svg("Tornado diagram of the one-way sensitivity analysis"),
    "그림 25-2. 가상 논문의 Figure 2 (Tornado diagram of one-way sensitivity analyses). 괄호 안이 입력값의 범위, 점선이 ₩50 million per QALY, 막대 끝의 숫자가 그때의 ICER(백만 원/QALY)입니다."))

# ====================================================================== 그림 25-3: 학술지 형식의 비용효과평면과 수용곡선
PS = N["psa"]
sdq, sdc = np.array(PS["scatter_dq"]), np.array(PS["scatter_dc"]) / 100
q1 = Plot((-0.25, 1.5), (-10, 80), w=430, h=330, ml=54, mr=18, mt=30, mb=48, xlabel="Incremental QALYs", ylabel="Incremental costs (₩ million)",
          xticks=[0, 0.5, 1.0, 1.5], xtickfmt=lambda v: f"{v:g}", yticks=[0, 20, 40, 60, 80])
q1.hline(0, cls="strongref", dash=False)
q1.vline(0, cls="strongref", dash=False)
inside = (sdq > -0.25) & (sdq < 1.5) & (sdc > -10) & (sdc < 80)
for x, y in zip(sdq[inside], sdc[inside]):
    q1.els.append(f'<circle cx="{q1.sx(x):.1f}" cy="{q1.sy(y):.1f}" r="1.7" class="f1" fill-opacity="0.42"/>')
q1.line([-0.2, 1.5], [-10, 75], s=4, dash=True, w=1.8)
q1.points([N["base"]["d_qaly"]], [N["base"]["d_cost"] / 100], s=2, r=5)
q1.text(1.48, 4, "Dashed line: ₩50 million per QALY", anchor="end", cls="lbl small mute")
q1.text(N["base"]["d_qaly"], N["base"]["d_cost"] / 100, "Base case", anchor="end", dx=-10, dy=-16, cls="lbl small strong")
panel_title(q1, "(A) Cost-effectiveness plane")
lams, pc = np.array(PS["lams"]) / 100, np.array(PS["curve"])
q2 = Plot((0, 120), (0, 1.0), w=430, h=330, ml=54, mr=18, mt=30, mb=48, xlabel="Willingness-to-pay threshold (₩ million per QALY)",
          ylabel="Probability cost-effective", xticks=[0, 20, 40, 50, 60, 80, 100, 120], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
q2.vline(50, dash=True)
q2.line(list(lams), list(pc), s=1, w=2.4)
for lam, dx, dy, anc in ((50, -7, -5, "end"), (60, 8, 6, "start"), (80, 6, 18, "start")):
    v = PS["ceac"][str(lam * 100)]
    q2.points([lam], [v], s=1, r=4)
    q2.text(lam, v, f"{v * 100:.1f}%", anchor=anc, dx=dx, dy=dy, cls="lbl small")
panel_title(q2, "(B) Cost-effectiveness acceptability curve")
n_out = int((~inside).sum())
save("ch25_paperpsa", figure(
    [q1.svg("Cost-effectiveness plane with probabilistic sensitivity analysis results"), q2.svg("Cost-effectiveness acceptability curve of drug A")],
    "그림 25-3. 가상 논문의 Figure 3 (Results of the probabilistic sensitivity analysis, 5,000 simulations; 500 shown in A). "
    "(A)의 점선 아래에 있는 점의 비율이 (B)에서 가로축 50의 높이입니다." + (f" (A)에서 축 밖의 점 {n_out}개는 그리지 않았습니다." if n_out else ""),
    cols=2))

# ====================================================================== 그림 25-4: 5년 재정영향
BI = N["bia"]
eok = lambda v: np.array(v) / 10000.0
net, drug = eok(BI["net"]), eok(BI["drug_A"])
yrs = np.arange(1, 6)
b = Plot((0.4, 5.6), (0, 110), w=600, h=320, ml=60, mr=22, mt=40, mb=50, xlabel="급여 후 연차", ylabel="금액 (억 원)",
         xticks=list(yrs), xtickfmt=lambda v: f"{v:.0f}년차", yticks=[0, 20, 40, 60, 80, 100])
b.bars(list(yrs - 0.19), list(drug), 0.34, cls="f4")
b.bars(list(yrs + 0.19), list(net), 0.34, s=1)
for x, d, v in zip(yrs, drug, net):
    b.text(x - 0.19, d, f"{d:.1f}", anchor="middle", dy=-6, cls="lbl small mute")
    b.text(x + 0.19, v, f"{v:.1f}", anchor="middle", dy=-6, cls="lbl small strong")
b.legend([("신약 A의 약품비 (대체된 치료를 빼지 않음)", 4, "box"), ("재정영향 = 신약 A를 급여할 때의 지출 − 급여하지 않을 때의 지출", 1, "box")], X=b.ml + 8, Y=12, gap=16)
save("ch25_bia", figure(
    b.svg("급여 후 5년 동안의 연도별 신약 A 약품비와 재정영향"),
    f"그림 25-4. 연도별 재정영향(가상의 예시, 할인하지 않음). 파란 막대가 재정영향(5년 합 {net.sum():.1f}억 원), 회색 막대가 신약 A의 약품비(5년 합 {drug.sum():.1f}억 원)입니다."))
print("figures written")
