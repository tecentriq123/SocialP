import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.stats as st
from svgplot import Plot, figure, panel_title, forest
from lib_ch12 import (garwood, poisson_glm, sandwich_hc0, nb_glm, nb_pmf, cohort, design, Z)

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


# ------------------------------------------------------------------ 12-1 pmf, lambda = 1, 4, 10
panels = []
ks = np.arange(0, 21)
for i, (lam, s) in enumerate(((1, 1), (4, 2), (10, 3))):
    last = i == 2
    pm = st.poisson.pmf(ks, lam)
    ymax = 0.5 if lam == 1 else 0.27 if lam == 4 else 0.175
    yt = [0, 0.1, 0.2, 0.3, 0.4, 0.5] if lam == 1 else [0, 0.1, 0.2] if lam == 4 else [0, 0.05, 0.1, 0.15]
    p = Plot((-0.7, 20.7), (0, ymax), w=600, h=170 if not last else 204, mt=26, mb=16 if not last else 50,
             xlabel="한 달 동안의 보고 건수 k" if last else "", ylabel="P(X = k)", xticks=list(range(0, 21, 2)),
             yticks=yt, ytickfmt=lambda v: f"{v:.2f}".rstrip("0").rstrip(".") if v else "0")
    if not last:
        p.xticklabels = [(t, "") for t in range(0, 21, 2)]
    p.bars(ks, pm, 0.62, s=s)
    yb = ymax * 0.80
    p.vline(lam, dash=True, y1=yb)
    sd = np.sqrt(lam)
    p.seg(lam - sd, yb, lam + sd, yb, cls="ln s4", w=1.6)
    p.seg(lam - sd, yb - ymax * 0.03, lam - sd, yb + ymax * 0.03, cls="ln s4", w=1.6)
    p.seg(lam + sd, yb - ymax * 0.03, lam + sd, yb + ymax * 0.03, cls="ln s4", w=1.6)
    panel_title(p, f"λ = {lam}")
    p.text(20.5, ymax * 0.95, f"평균 = 분산 = {lam},  SD = √{lam} = {sd:.2f}", anchor="end", cls="lbl", dy=4)
    p.text(20.5, ymax * 0.95, f"P(X = 0) = {st.poisson.pmf(0, lam):.5f}" if lam == 10 else f"P(X = 0) = {st.poisson.pmf(0, lam):.4f}",
           anchor="end", cls="lbl mute small", dy=22)
    panels.append(p.svg(f"평균이 {lam}인 포아송 분포"))
save("ch12_pmf", figure(panels,
     "그림 12-1. 평균 λ가 1, 4, 10인 포아송 분포. 막대는 각 건수가 나올 확률, 점선은 평균 λ, 회색 가로선은 평균 ± 1 SD(= ±√λ)입니다. "
     "λ가 작으면 0 쪽으로 몰린 비대칭 분포이고, λ가 커질수록 평균을 중심으로 대칭에 가까워지면서 퍼짐(분산 = λ)도 커집니다."))

# ------------------------------------------------------------------ 12-2 coverage of 95% CIs
def coverage(mu, kind):
    kk = np.arange(0, int(mu + 12 * np.sqrt(mu) + 30))
    pk = st.poisson.pmf(kk, mu)
    tot = 0.0
    for k, pr in zip(kk, pk):
        lo, hi = garwood(k) if kind == "exact" else (k - Z * np.sqrt(k), k + Z * np.sqrt(k))
        if lo <= mu <= hi:
            tot += pr
    return tot


grid = np.round(np.arange(1.0, 30.001, 0.1), 2)
cw = [coverage(m, "wald") for m in grid]
ce = [coverage(m, "exact") for m in grid]
p = Plot((0, 30), (0.6, 1.0), w=600, h=320, xlabel="기대 사건 수 (λ × 인년)", ylabel="실제 포함 확률",
         xticks=[0, 5, 10, 15, 20, 25, 30], yticks=[0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
         ytickfmt=lambda v: f"{v * 100:.0f}%")
p.hline(0.95, dash=True)
p.line(grid, cw, s=2, w=1.6)
p.line(grid, ce, s=1, w=2)
p.text(13, 0.985, "정확(Garwood) 신뢰구간", anchor="start", cls="lbl strong", dy=-2)
p.text(4.2, 0.70, "정규근사 신뢰구간", anchor="start", cls="lbl strong")
p.text(4.2, 0.70, "사건 수 ± 1.96√사건 수", anchor="start", cls="lbl mute small", dy=16)
i1, i3 = int(np.argmin(abs(grid - 1))), int(np.argmin(abs(grid - 3)))
p.points([1, 3], [cw[i1], cw[i3]], s=2, r=4)
p.text(1, cw[i1], f"{cw[i1] * 100:.0f}%", anchor="start", dx=8, dy=4, cls="lbl small")
p.text(3, cw[i3], f"{cw[i3] * 100:.0f}%", anchor="start", dx=8, dy=10, cls="lbl small")
save("ch12_coverage", figure(p.svg("신뢰구간 방법별 실제 포함 확률"),
     "그림 12-2. '95% 신뢰구간'이 실제로 참값을 포함하는 비율. 같은 연구를 무한히 반복했을 때를 포아송 확률로 정확히 계산했습니다. "
     "점선이 목표 95%입니다. 정규근사 구간(주황)은 기대 사건 수가 작을수록 포함 확률이 95%에 크게 못 미치고(기대 1건에서 63%, 3건에서 80%) 들쭉날쭉합니다. 정확 구간(파랑)은 항상 95% 이상입니다."))

# ------------------------------------------------------------------ 12-3 offset: six patients
toy = [("#1", 1, 0.5, [0.3]), ("#2", 1, 2.0, [0.6, 1.5]), ("#3", 1, 1.2, []),
       ("#4", 0, 3.0, [0.4, 1.1, 1.9, 2.6]), ("#5", 0, 0.8, []), ("#6", 0, 2.5, [0.7, 1.6, 2.2])]
p = Plot((0, 4.3), (0.3, 7.2), w=620, h=300, ml=78, mr=16, mt=16, mb=50, xlabel="추적 기간 (년)",
         xticks=[0, 1, 2, 3], show_yaxis=False, ygrid=False, xgrid=True)
for i, (pid, a, t, evs) in enumerate(toy):
    yrow = 6.5 - i if i < 3 else 6.0 - i
    s = 1 if a else 2
    p.seg(0, yrow, t, yrow, cls=f"ln s{s}", w=3)
    if evs:
        p.points(evs, [yrow] * len(evs), s=s, r=5.5)
    p.seg(t, yrow - 0.18, t, yrow + 0.18, cls=f"ln s{s}", w=2)
    p.text(0, yrow, f"{pid} 약 {'A' if a else 'B'}", anchor="end", dx=-10, dy=4)
    p.text(3.2, yrow, f"{len(evs)}건 / {t:g}년", anchor="start", dy=4, cls="lbl small")
p.text(3.2, 7.0, "건수 / 추적", anchor="start", cls="lbl mute small", dy=4)
p.text(0.05, 3.55, "", anchor="start")
p.hline(3.5, cls="grid", dash=False)
p.text_px(606, 268, "점: 악화 발생 시점", anchor="end", cls="lbl mute small")
save("ch12_offset", figure(p.svg("추적 기간이 다른 환자 6명의 사건 발생"),
     "그림 12-3. 가상의 환자 6명. 선의 길이가 추적 기간, 점이 사건(악화) 발생 시점입니다. "
     "약 A군은 3명이 3.7인년 동안 3건(0.81건/인년), 약 B군은 3명이 6.3인년 동안 7건(1.11건/인년)을 겪었습니다. "
     "1인당 건수(1.0 대 2.3)만 비교하면 B군이 오래 관찰되었다는 사실이 차이처럼 보입니다."))

# ------------------------------------------------------------------ COPD cohort fits (12-4, 12-5)
c = cohort()
y, py = c["y"], c["py"]
X = design(c)
off = np.log(py)
pf = poisson_glm(X, y, off)
nb = nb_glm(X, y, off)
n = c["n"]
K = 7
obs = np.array([np.sum(y == k) for k in range(K)] + [np.sum(y >= K)], float)
ep = np.array([st.poisson.pmf(k, pf["mu"]).sum() for k in range(K)] + [st.poisson.sf(K - 1, pf["mu"]).sum()])
enb = np.array([nb_pmf(k, nb["mu"], nb["alpha"]).sum() for k in range(K)])
enb = np.r_[enb, n - enb.sum()]
labels = [str(k) for k in range(K)] + [f"{K}+"]


def obs_panel(idx, ymax, yt, title, show_legend):
    p = Plot((idx[0] - 0.6, idx[-1] + 0.6), (0, ymax), w=420, h=300, ml=56, mr=14, mt=26,
             xlabel="환자 1명의 악화 건수", ylabel="환자 수", xticks=list(idx),
             xticklabels=[(k, labels[k]) for k in idx], yticks=yt)
    p.bars(idx, obs[idx], 0.62, cls="a4 s4")
    p.points(np.array(idx) - 0.14, ep[idx], s=2, r=5)
    p.points(np.array(idx) + 0.14, enb[idx], s=1, r=5)
    panel_title(p, title)
    if show_legend:
        p.legend([("관측", 4, "soft"), ("포아송 기대", 2, "dot"), ("음이항 기대", 1, "dot")], X=250, Y=44)
    return p


p1 = obs_panel([0, 1, 2], 1800, [0, 400, 800, 1200, 1600], "0–2건", True)
for j, (lab, v) in enumerate((("0건 관측", obs[0]), ("포아송 예측", ep[0]), ("음이항 예측", enb[0]))):
    p1.text(0.42, 1180 - j * 150, f"{lab} {v:,.0f}명", anchor="start", cls="lbl small")
p2 = obs_panel([3, 4, 5, 6, 7], 100, [0, 20, 40, 60, 80, 100], "3건 이상", False)
for j, (lab, v) in enumerate((("7건 이상 관측", f"{obs[7]:.0f}"), ("포아송 예측", f"{ep[7]:.1f}"), ("음이항 예측", f"{enb[7]:.1f}"))):
    p2.text(7.55, 88 - j * 11, f"{lab} {v}명", anchor="end", cls="lbl small")
save("ch12_obsexp", figure([p1.svg("0-2건의 관측 빈도와 기대 빈도"), p2.svg("3건 이상의 관측 빈도와 기대 빈도")],
     f"그림 12-4. COPD 코호트 {n:,}명의 악화 건수 분포(회색 막대)와 두 모형이 예측한 환자 수. 기대 빈도는 환자마다 모형이 준 확률을 모두 더한 값입니다. "
     f"포아송 모형은 0건({ep[0]:,.0f}명 예측, 관측 {obs[0]:,.0f}명)과 4건 이상을 과소예측하고 1–2건을 과대예측합니다. "
     "음이항 모형은 양 끝을 거의 맞춥니다. 오른쪽 패널은 y축 범위가 다릅니다.", cols=2))

# ------------------------------------------------------------------ 12-5 forest: drug A IRR by method
Xc = design(c, ("drugA",))
nc = nb_glm(Xc, y, off)
phi = pf["pearson"] / pf["df"]
rob = np.sqrt(np.diag(sandwich_hc0(pf)))
b, s = pf["beta"][1], pf["se"][1]


def row(label, be, se, sser=1, bold=False):
    return dict(label=label, est=np.exp(be), lo=np.exp(be - Z * se), hi=np.exp(be + Z * se), s=sser, bold=bold)


rows = [
    row("보정 전, 음이항", nc["beta"][1], nc["se"][1], 4),
    row("포아송, 모형 기반 SE", b, s, 2),
    row(f"준포아송 (φ = {phi:.2f})", b, s * np.sqrt(phi), 1),
    row("포아송 + 강건 SE", b, rob[1], 1),
    row("음이항 회귀", nb["beta"][1], nb["se"][1], 1, bold=True),
]
svg = forest(rows, (0.6, 1.2), ref=1.0, log=True, w=640, label_w=200, est_w=140,
             xlabel="약 A 대 약 B의 발생률비 (로그 척도, 1보다 작으면 약 A에서 악화가 적음)", xticks=[0.6, 0.7, 0.8, 0.9, 1.0, 1.2],
             header=("분석 방법", "IRR (95% CI)"))
save("ch12_forest", figure(svg,
     "그림 12-5. 같은 COPD 자료에서 분석 방법에 따른 약 A의 발생률비. 아래 넷은 연령군·성별·CCI를 보정했습니다. "
     "포아송의 모형 기반 신뢰구간(주황)만 좁고, 과산포를 반영한 세 방법의 신뢰구간은 서로 비슷합니다. 점추정값은 네 방법 모두 0.79입니다."))
print("figures written")
