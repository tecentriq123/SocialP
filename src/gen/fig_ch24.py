"""Figures for chapter 24 (불확실성 분석).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch24.py && python3 gen/fig_ch24.py
Writes figs/ch24_tornado.html, ch24_dists.html, ch24_scatter.html, ch24_converge.html, ch24_inmb.html, ch24_ceac.html
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from scipy import stats
from svgplot import Plot, figure, panel_title

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch24_nums.json"), encoding="utf-8"))
LAM = 5000.0
B0 = N["base"]
won = lambda v: f"{v:,.0f}".replace("-", "−")
pct = lambda v: f"{v * 100:.1f}%"


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


# ====================================================================== 그림 24-1: 토네이도 그림
def fmt_val(key, v):
    if key.startswith("c_"):
        return f"{v:,.0f}"
    if key.endswith("_lam"):
        return f"{v:.4f}"
    return f"{v:.2f}"


rows = N["oneway"][:10]
n = len(rows)
p = Plot((3500, 8500), (0, n), w=660, h=64 + 31 * n + 46, ml=206, mr=22, mt=64, mb=46, xlabel="ICER (만원/QALY)",
         xticks=[4000, 5000, 6000, 7000, 8000], xtickfmt=won, ygrid=False, xgrid=True, show_yaxis=False)
for i, r in enumerate(rows):
    yc = n - i - 0.5
    for end, s in (("low", 1), ("high", 2)):
        a, b = sorted((B0["icer"], r["icer_" + end]))
        if b - a < 1e-9:
            continue
        xa, xb = p.sx(a), p.sx(b)
        p.els.append(f'<rect x="{xa:.1f}" y="{p.sy(yc + 0.31):.1f}" width="{xb - xa:.1f}" height="{p.sy(yc - 0.31) - p.sy(yc + 0.31):.1f}" class="f{s}"/>')
    p.text_px(p.ml - 10, p.sy(yc) + 4.5, r["label"], anchor="end", cls="lbl small")
    # 막대 양 끝에 입력값
    lo_end, hi_end = (r["icer_low"], r["low"]), (r["icer_high"], r["high"])
    left, right = sorted((lo_end, hi_end))
    if abs(left[0] - B0["icer"]) > 1e-9:
        p.text(left[0], yc, fmt_val(r["key"], left[1]), anchor="end", dx=-5, dy=4.5, cls="lbl small mute")
    if abs(right[0] - B0["icer"]) > 1e-9:
        p.text(right[0], yc, fmt_val(r["key"], right[1]), anchor="start", dx=5, dy=4.5, cls="lbl small mute")
p.vline(B0["icer"], cls="strongref", dash=False, w=1.4)
p.vline(LAM, dash=True)
p.text(B0["icer"], n, f"기준 분석 {won(B0['icer'])}", dx=5, dy=-8, cls="lbl small strong")
p.text(LAM, n, "가정한 임계값 5,000", anchor="end", dx=-5, dy=-8, cls="lbl small mute")
p.legend([("입력값이 범위의 낮은 쪽 끝일 때", 1, "box"), ("입력값이 범위의 높은 쪽 끝일 때", 2, "box")], X=p.ml + 150, Y=14, gap=17)
top3 = rows[:3]
save("ch24_tornado", figure(
    p.svg("일원 민감도 분석 결과를 ICER가 움직인 폭이 큰 순서로 쌓은 토네이도 그림"),
    "그림 24-1. 토네이도 그림(가상의 예시). 입력값을 하나씩 범위의 양 끝으로 바꿨을 때의 ICER를 막대로 그리고 긴 것부터 위에 쌓았습니다. "
    "막대 끝의 숫자는 그때의 입력값, 세로 실선은 기준 분석의 ICER입니다. 막대가 점선(가정한 임계값)을 넘는 입력값은 그것 하나로 결론이 바뀔 수 있습니다. "
    "17개 입력값 가운데 위 10개만 그렸습니다."))

# ====================================================================== 그림 24-2: 세 가지 분포
D = N["dist"]


def dist_panel(title, xs, pdf, d, xlabel, xticks, xfmt, note):
    q = Plot((xs[0], xs[-1]), (0, max(pdf) * 1.22), w=330, h=250, ml=18, mr=14, mt=30, mb=46, xlabel=xlabel,
             xticks=xticks, xtickfmt=xfmt, ygrid=False, show_yaxis=False)
    m = (xs >= d["lo"]) & (xs <= d["hi"])
    q.fill_between(list(xs[m]), [0] * int(m.sum()), list(pdf[m]), s=1)
    q.line(list(xs), list(pdf), s=1, w=2)
    q.vline(d["base"], cls="strongref", dash=False, y1=max(pdf) * 1.04)
    q.text(d["base"], max(pdf) * 1.04, f"기준값 {xfmt(d['base'])}", anchor="middle", dy=-6, cls="lbl small strong")
    q.text_px(q.w - q.mr - 2, q.mt + 30, note[0], anchor="end", cls="lbl small mute")
    q.text_px(q.w - q.mr - 2, q.mt + 45, note[1], anchor="end", cls="lbl small mute")
    panel_title(q, title)
    return q.svg(title)


x1 = np.linspace(0.66, 0.90, 241); d1 = D["u_pf"]
x2 = np.linspace(90, 460, 371); d2 = D["c_pd"]
x3 = np.linspace(0.42, 1.22, 321); d3 = D["hr_os"]
panels = [
    dist_panel("(가) 베타분포: 무진행 상태의 효용", x1, stats.beta.pdf(x1, d1["alpha"], d1["beta"]), d1, "효용", [0.70, 0.75, 0.80, 0.85, 0.90],
               lambda v: f"{v:.2f}", (f"95% 구간", f"{d1['lo']:.2f}–{d1['hi']:.2f}")),
    dist_panel("(나) 감마분포: 진행 상태 월 비용", x2, stats.gamma.pdf(x2, d2["shape"], scale=d2["scale"]), d2, "비용 (만원)", [100, 200, 300, 400],
               lambda v: f"{v:.0f}", (f"95% 구간", f"{d2['lo']:.0f}–{d2['hi']:.0f}")),
    dist_panel("(다) 로그정규분포: 전체생존 위험비", x3, stats.lognorm.pdf(x3, d3["sigma"], scale=d3["base"]), d3, "위험비", [0.5, 0.75, 1.0, 1.2],
               lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v != 0.75 else "0.75", (f"95% 구간", f"{d3['lo']:.2f}–{d3['hi']:.2f}")),
]
save("ch24_dists", figure(
    panels,
    "그림 24-2. 확률적 민감도 분석에서 입력값을 뽑는 분포의 예(가상의 예시). 색칠한 부분이 95% 구간입니다. "
    f"(가) 베타분포는 0과 1 사이의 값만 냅니다(α = {d1['alpha']:.1f}, β = {d1['beta']:.1f}). "
    f"(나) 감마분포는 양수만 내고 오른쪽 꼬리가 깁니다(모양 {d2['shape']:.0f}, 척도 {d2['scale']:.0f}). "
    f"(다) 로그정규분포는 비의 척도에서 대칭이어서 구간이 기준값의 위쪽으로 더 넓습니다(0.75에서 아래로 {d3['base'] - d3['lo']:.2f}, 위로 {d3['hi'] - d3['base']:.2f}).",
    cols=3))

# ====================================================================== 그림 24-3: 비용효과평면 위의 PSA 산점도
PS = N["psa"]
sdq, sdc = np.array(N["scatter"]["dq"]), np.array(N["scatter"]["dc"])
p = Plot((-0.3, 1.5), (-1000, 8000), w=620, h=410, ml=70, mr=22, mt=20, mb=50, xlabel="증분 QALY (신약 A − 표준요법 B)", ylabel="증분 비용 (만원)",
         xticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5], xtickfmt=lambda v: f"{v:g}", yticks=[0, 2000, 4000, 6000, 8000], ytickfmt=won)
p.fill_between([-0.2, 1.5], [-1000, -1000], [-1000, 7500], s=1)
p.hline(0, cls="strongref", dash=False)
p.vline(0, cls="strongref", dash=False)
inside = (sdq > -0.3) & (sdq < 1.5) & (sdc > -1000) & (sdc < 8000)
for x, y in zip(sdq[inside], sdc[inside]):
    p.els.append(f'<circle cx="{p.sx(x):.1f}" cy="{p.sy(y):.1f}" r="1.9" class="f1" fill-opacity="0.42"/>')
p.line([-0.2, 1.5], [-1000, 7500], s=4, dash=True, w=1.8)
p.points([B0["d_qaly"]], [B0["d_cost"]], s=2, r=5.5)
p.text(B0["d_qaly"], B0["d_cost"], "기준 분석", anchor="end", dx=-12, dy=-28, cls="lbl strong small")
p.text(B0["d_qaly"], B0["d_cost"], f"({B0['d_qaly']:.3f}, {won(B0['d_cost'])})", anchor="end", dx=-12, dy=-14, cls="lbl small")
p.text(1.02, 5100, "임계값 선 (기울기 5,000)", dx=10, dy=22, cls="lbl small mute")
p.text(1.48, 600, "선 아래: 비용효과적", anchor="end", cls="lbl small strong")
p.text(1.48, 600, f"5,000회 가운데 {pct(PS['p_ce'])}", anchor="end", dy=15, cls="lbl small")
p.text(-0.28, 7400, "선 위: 비용효과적이지 않음", cls="lbl small strong")
p.text(-0.28, 7400, f"{pct(1 - PS['p_ce'])}", dy=15, cls="lbl small")
n_out = int((~inside).sum())
save("ch24_scatter", figure(
    p.svg("확률적 민감도 분석의 증분 QALY와 증분 비용을 비용효과평면에 찍은 산점도"),
    "그림 24-3. 비용효과평면 위의 확률적 민감도 분석 결과(가상의 예시). 점 하나가 모의실험 한 번의 증분 QALY와 증분 비용입니다. "
    f"5,000회 가운데 처음 1,000회만 그렸습니다{'' if n_out == 0 else f'(축 밖의 {n_out}개 제외)'}. 점들이 오른쪽 위로 길게 늘어선 것은 생존 이득이 큰 모의실험일수록 투약과 진행 상태의 비용도 함께 늘기 때문입니다. "
    f"점선은 이 예시에서 가정한 임계값(5,000만원/QALY)이고, 5,000회 가운데 {pct(PS['p_ce'])}가 이 선 아래(색칠한 영역)에 있습니다. "
    f"주황색 점은 기준 분석입니다."))

# ====================================================================== 그림 24-4: 모의실험 횟수와 확률의 안정
cv = N["conv"]
ns = np.array(cv["run_n"]); rp = np.array(cv["run"])
pf = PS["p_ce"]
p = Plot((0, 5000), (0.08, 0.30), w=600, h=320, ml=62, mr=28, xlabel="모의실험 횟수", ylabel="비용효과적일 확률 (누적)",
         xticks=[0, 1000, 2000, 3000, 4000, 5000], xtickfmt=won, yticks=[0.10, 0.15, 0.20, 0.25, 0.30], ytickfmt=lambda v: f"{v * 100:.0f}%")
m = ns >= 50
band = 1.96 * np.sqrt(pf * (1 - pf) / ns[m])
p.fill_between(list(ns[m]), list(np.clip(pf - band, 0.08, 0.30)), list(np.clip(pf + band, 0.08, 0.30)), s=4)
p.hline(pf, dash=True)
p.line(list(ns[m]), list(np.clip(rp[m], 0.08, 0.30)), s=1, w=2)
for k in ("500", "1000", "5000"):
    v = cv["at"][k]
    p.points([int(k)], [v], s=1, r=4)
    p.text(int(k), v, f"{won(int(k))}회 {pct(v)}", anchor="end" if k == "5000" else "start", dx=-4 if k == "5000" else 4, dy=-10 if k != "500" else 18, cls="lbl small")
p.text(2600, 0.262, "회색 띠: 그 횟수에서 우연히 생길 수 있는 오차의 범위", anchor="middle", cls="lbl small mute")
p.text(2600, 0.262, "(최종값 ± 1.96 × 몬테카를로 표준오차)", anchor="middle", dy=15, cls="lbl small mute")
save("ch24_converge", figure(
    p.svg("모의실험 횟수가 늘어남에 따라 비용효과적일 확률의 누적 추정값이 안정되는 모습"),
    f"그림 24-4. 모의실험 횟수와 결과의 안정(가상의 예시). 처음부터 그 횟수까지의 모의실험으로 구한 '비용효과적일 확률'입니다. "
    f"수백 회까지는 크게 흔들리다가 점차 {pct(pf)} 근처에 머뭅니다. 5,000회에서 몬테카를로 표준오차는 {PS['mcse'] * 100:.1f}%포인트입니다."))

# ====================================================================== 그림 24-5: 증분 순금전편익의 분포
H = N["inmb"]
edges = np.arange(-2000, 2001, 100)
cnt = np.array(H["hist"])
p = Plot((-2000, 2000), (0, max(cnt) * 1.18), w=600, h=320, ml=62, mr=24, xlabel="증분 순금전편익 (만원, 임계값 5,000만원/QALY)", ylabel="모의실험 횟수",
         xticks=[-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000], xtickfmt=won)
neg = edges[:-1] < 0
p.hist(edges[: neg.sum() + 1], cnt[neg], cls="f4")
p.hist(edges[neg.sum():], cnt[~neg], s=1)
p.vline(0, cls="strongref", dash=False)
p.vline(H["mean"], dash=True, y1=max(cnt) * 1.06)
p.text(H["mean"], max(cnt) * 1.06, f"평균 {won(H['mean'])}", anchor="end", dx=-6, dy=4, cls="lbl small strong")
p.text(520, max(cnt) * 0.62, f"0보다 큼: {won(H['n_pos'])}회 ({pct(PS['p_ce'])})", cls="lbl small strong")
p.text(520, max(cnt) * 0.62, "신약 A가 비용효과적인 모의실험", dy=15, cls="lbl small")
p.text(-1950, max(cnt) * 0.62, f"0보다 작음: {won(5000 - H['n_pos'])}회 ({pct(1 - PS['p_ce'])})", cls="lbl small strong")
save("ch24_inmb", figure(
    p.svg("확률적 민감도 분석 5,000회의 증분 순금전편익 히스토그램"),
    f"그림 24-5. 모의실험 5,000회의 증분 순금전편익(가상의 예시, 임계값 5,000만원/QALY). 평균은 {won(H['mean'])}만원이고 "
    f"2.5–97.5 백분위수는 {won(H['lo'])}만원에서 {won(H['hi'])}만원입니다. 0보다 큰 부분(파란 막대)의 비율 {pct(PS['p_ce'])}가 이 임계값에서 신약 A가 비용효과적일 확률입니다."
    + (f" 가로축 범위 밖의 {H['below'] + H['above']}회는 그리지 않았습니다." if H["below"] + H["above"] else "")))

# ====================================================================== 그림 24-6: 비용효과 수용곡선과 EVPI
C = N["ceac"]
lams = np.array(C["lams"]); pc = np.array(C["p"]); ev = np.array(C["evpi"])
xt = [0, 2000, 4000, 6000, 8000, 10000, 12000]
p1 = Plot((0, 12000), (0, 1.0), w=430, h=340, ml=54, mr=28, mt=30, mb=48, xlabel="임계값 (만원/QALY)", ylabel="비용효과적일 확률",
          xticks=xt, xtickfmt=won, yticks=[0, 0.25, 0.5, 0.75, 1.0], ytickfmt=lambda v: f"{v * 100:.0f}%")
p1.hline(0.5, dash=True)
p1.vline(B0["icer"], dash=True, y1=0.5)
p1.line(list(lams), list(pc), s=1, w=2.4)
for lam, dx, dy, anc in ((5000, -8, -4, "end"), (6000, 9, 6, "start"), (8000, 6, 18, "start")):
    v = C["tab"][str(lam)]["p"]
    p1.points([lam], [v], s=1, r=4.5)
    p1.text(lam, v, f"{won(lam)}: {pct(v)}", anchor=anc, dx=dx, dy=dy, cls="lbl small")
p1.text(B0["icer"], 0.06, f"ICER {won(B0['icer'])}", anchor="start", dx=6, dy=4, cls="lbl small mute")
p1.text(11900, C["tab"]["12000"]["p"], pct(C["tab"]["12000"]["p"]), anchor="end", dy=16, cls="lbl small")
panel_title(p1, "(가) 비용효과 수용곡선")
p2 = Plot((0, 12000), (0, 200), w=430, h=340, ml=54, mr=28, mt=30, mb=48, xlabel="임계값 (만원/QALY)", ylabel="1인당 EVPI (만원)",
          xticks=xt, xtickfmt=won, yticks=[0, 50, 100, 150, 200])
p2.vline(B0["icer"], dash=True, y1=175)
p2.line(list(lams), list(ev), s=2, w=2.4)
v5 = C["tab"]["5000"]["evpi"]
p2.points([5000], [v5], s=2, r=4.5)
p2.text(5000, v5, f"5,000: {v5:.0f}만원", anchor="end", dx=-8, dy=-2, cls="lbl small")
p2.points([C["lam_evpi_max"]], [C["evpi_max"]], s=2, r=4.5)
p2.text(C["lam_evpi_max"], C["evpi_max"], f"최대 {C['evpi_max']:.0f}만원 (임계값 {won(C['lam_evpi_max'])})", dx=8, dy=-8, cls="lbl small")
panel_title(p2, "(나) 완전정보의 기대가치")
save("ch24_ceac", figure(
    [p1.svg("임계값에 따른 신약 A가 비용효과적일 확률, 비용효과 수용곡선"), p2.svg("임계값에 따른 환자 1인당 완전정보의 기대가치")],
    "그림 24-6. 비용효과 수용곡선과 완전정보의 기대가치(가상의 예시). (가) 임계값마다 모의실험 5,000회 가운데 증분 순금전편익이 양수인 비율을 이은 곡선입니다. "
    f"이 예시에서 가정한 임계값 5,000만원/QALY에서 {pct(C['tab']['5000']['p'])}, 6,000만원에서 {pct(C['tab']['6000']['p'])}이고, 곡선은 임계값 {won(C['lam50'])}만원에서 50%를 넘습니다(세로 점선은 기준 분석의 ICER). "
    f"{pct(C['tab']['5000']['p'])}와 {pct(C['tab']['6000']['p'])}도 무진행생존 곡선을 전체생존 곡선에 맞춰 자른 이 예제의 단순화를 포함한 값입니다(<a href=\"#ch24-s2\">나 절</a>). "
    f"(나) 환자 1인당 완전정보의 기대가치(EVPI)는 결정이 가장 아슬아슬한 임계값, 곧 ICER 근처에서 가장 큽니다.",
    cols=2))
# ====================================================================== 그림 24-7: 학술지 형식의 수용곡선(논문 상자)
pp = Plot((0, 120), (0, 1.0), w=600, h=330, ml=62, mr=26, mt=20, mb=52, xlabel="Willingness-to-pay threshold (₩ million per QALY gained)",
          ylabel="Probability cost-effective", xticks=[0, 20, 40, 50, 60, 80, 100, 120], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
pp.vline(50, dash=True)
pp.line(list(lams / 100), list(pc), s=1, w=2.4)
pp.line(list(lams / 100), list(1 - pc), s=2, w=2, dash=True)
pp.legend([("Drug A", 1, "line"), ("Standard therapy B", 2, "dash")], X=pp.sx(84), Y=pp.sy(0.56))
save("ch24_paperceac", figure(
    pp.svg("Cost-effectiveness acceptability curves of drug A and standard therapy B"),
    "그림 24-7. 학술지 형식의 그림(가상의 예시). 원문 범례: “Figure 3. Cost-effectiveness acceptability curves. The curves show the proportion of 5,000 probabilistic simulations in which "
    "each strategy had the higher net monetary benefit at a given willingness-to-pay threshold. The dashed vertical line indicates ₩50 million per QALY.” "
    "대안이 둘이면 두 곡선의 합은 언제나 1이어서 표준요법 B의 곡선은 신약 A의 곡선을 뒤집은 모양입니다."))
print("figures written")
