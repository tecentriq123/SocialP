"""Figures added with the '잠깐 확인' round (2026-10-24): 그림 9-4 (ch09_ordiff), 15-4 (ch15_matched), 21-4 (ch21_hetamt).
run: source /home/claude/pylibs/env.sh && cd gen && python3 fig_extra_qc.py
All numbers come from the chapters' own number scripts (nums_ch09.compute, nums_ch15 functions, _ch21_nums.json).
"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from svgplot import Plot, figure, panel_title  # noqa

OUT = os.path.join(HERE, "..", "figs")


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def expit(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ 그림 9-4: same OR, different RD and RR by baseline risk
import nums_ch09
R9 = nums_ch09.compute()
beta = float(R9["etio"]["fit"]["beta"][1])
OR = float(np.exp(beta))
pe = R9["pats_eff"]
A_, B_ = pe["A"], pe["B"]
p0 = np.linspace(0.005, 0.995, 397)
p1 = expit(logit(p0) + beta)
rd = 100 * (p1 - p0)
rr = p1 / p0
pmin = float(expit(-beta / 2))
print(f"[9-4] beta {beta:.4f} OR {OR:.3f}; RD min {rd.min():.2f}%p at p0 {100 * p0[rd.argmin()]:.1f}% (exact {100 * pmin:.1f}%); "
      f"A p0 {A_['p0']:.3f} rd {100 * A_['rd']:.1f} rr {A_['rr']:.2f}; B p0 {B_['p0']:.3f} rd {100 * B_['rd']:.1f} rr {B_['rr']:.2f}")
pa = Plot((0, 100), (-8, 0.5), h=320, mt=24, ylabel="위험차 (%p)", xlabel="DPP-4 억제제를 쓸 때의 1년 입원 확률 (%)",
          xticks=[0, 20, 40, 60, 80, 100], yticks=[-8, -6, -4, -2, 0],
          ytickfmt=lambda v: f"{v:.0f}".replace("-", "−"))
pa.line(100 * p0, rd, s=1)
pa.hline(0, dash=False, cls="axis")
ymin = 100 * (expit(logit(pmin) + beta) - pmin)
pa.points([100 * pmin], [ymin], s=4, r=4)
pa.text(100 * pmin, ymin, f"기저위험 {100 * pmin:.1f}%에서 가장 큼 ({ymin:.1f}%p)".replace("-", "−"), anchor="middle", dy=22, cls="lbl mute small")
for lab, d in (("환자 A", A_), ("환자 B", B_)):
    pa.points([100 * d["p0"]], [100 * d["rd"]], s=2, r=5)
    pa.text(100 * d["p0"], 100 * d["rd"], f"{lab} {100 * d['rd']:.1f}%p".replace("-", "−"), dx=9, dy=-9, cls="lbl strong")
save("ch09_ordiff", figure(pa.svg("오즈비가 같을 때 기저위험에 따른 위험차"),
     f"그림 9-4. 보정 오즈비 {OR:.2f}을 기저위험이 다른 환자에게 똑같이 적용했을 때의 위험차. 가로축은 그 환자가 DPP-4 억제제를 쓸 때의 1년 입원 확률이고, "
     f"세로축은 SGLT2 억제제로 바꿀 때 달라지는 확률(%p)입니다. 곡선 위의 모든 환자는 오즈비가 {OR:.2f}으로 같지만, 위험차는 기저위험 {100 * pmin:.1f}%에서 "
     f"{ymin:.1f}%p로 가장 크고 0이나 100%에 가까울수록 0에 가까워집니다.".replace("-6", "−6")))

# ------------------------------------------------------------------ 그림 15-4: who found a match in 1:1 matching
import nums_ch15 as N15
X, A, Y, risk, day, NC = N15.simulate_cohort()
ps, _ = N15.fit_ps(X, A, N15.KEYS)
pairs, cal = N15.greedy_match(ps, A)
pairs = np.asarray(pairs)
idxA = np.where(A == 1)[0]; idxB = np.where(A == 0)[0]
mA = np.zeros(len(A), bool); mA[pairs[:, 0]] = True
mB = np.zeros(len(A), bool); mB[pairs[:, 1]] = True
unA = (A == 1) & ~mA; unB = (A == 0) & ~mB
print(f"[15-4] pairs {len(pairs)}; unmatched A {unA.sum()} (min ps {ps[unA].min():.3f}); unused B {unB.sum()} (max ps {ps[unB].max():.3f})")
edges = np.linspace(0.10, 0.95, 18)
mids = (edges[:-1] + edges[1:]) / 2
hAm, _ = np.histogram(ps[(A == 1) & mA], edges); hAu, _ = np.histogram(ps[unA], edges)
hBm, _ = np.histogram(ps[(A == 0) & mB], edges); hBu, _ = np.histogram(ps[unB], edges)
assert hAm.sum() + hAu.sum() == (A == 1).sum() and hBm.sum() + hBu.sum() == (A == 0).sum()
ytop, ybot = 650, -1150
p = Plot((0.1, 0.95), (ybot, ytop), h=380, ml=62, mt=24, xlabel="성향점수 (약물 A를 받을 추정 확률)", ylabel="인원 (성향점수 0.05 간격)",
         xticks=[0.1, 0.3, 0.5, 0.7, 0.9], xtickfmt=lambda v: f"{v:.1f}",
         yticks=[-1000, -500, 0, 500], ytickfmt=lambda v: f"{abs(v):,.0f}")
assert (hAm + hAu).max() < ytop and (hBm + hBu).max() < -ybot
w = 0.043
p.bars(mids, hAm + hAu, w, s=4, rounded=False)
p.bars(mids, hAm, w, s=1, rounded=False)
p.bars(mids, -(hBm + hBu), w, cls="a2", rounded=False)
p.bars(mids, -hBm, w, s=2, rounded=False)
p.hline(0, dash=False, cls="axis")
p.vline(0.5)
p.text(0.94, ytop * 0.86, f"약물 A (짝을 찾음 {int(hAm.sum()):,}, 못 찾음 {int(hAu.sum()):,})", anchor="end", cls="lbl strong")
p.text(0.94, ybot * 0.93, f"약물 B (짝으로 쓰임 {int(hBm.sum()):,}, 쓰이지 않음 {int(hBu.sum()):,})", anchor="end", cls="lbl strong")
save("ch15_matched", figure(p.svg("1:1 매칭에서 짝을 찾은 사람과 못 찾은 사람의 성향점수"),
     f"그림 15-4. 1:1 매칭 결과를 성향점수대별로 나눈 그림입니다. 위는 약물 A 사용자로 파랑이 짝을 찾은 사람, 그 위에 쌓은 회색이 짝을 못 찾은 사람입니다. "
     f"아래는 약물 B 사용자로 진한 주황이 짝으로 쓰인 사람, 연한 주황이 쓰이지 않은 사람입니다. 짝을 못 찾은 A 사용자 {int(unA.sum()):,}명은 모두 점수 0.5 위쪽에, "
     f"쓰이지 않은 B 사용자 {int(unB.sum()):,}명은 모두 0.5 아래쪽에 있습니다. 세로 점선이 성향점수 0.5이고, 세로축은 성향점수 0.05 간격마다의 인원입니다."))

# ------------------------------------------------------------------ 그림 21-4: amount added by drug A across comorbidity scores
R21 = json.load(open(os.path.join(HERE, "_ch21_nums.json"), encoding="utf-8"))
het = R21["het"]; ratio = R21["reg"]["glm"]["ratio"]; ols = R21["reg"]["ols"]["diff"]
xs = [0, 1, 2]
truth = [het[k]["diff"] for k in ("cci0", "cci2", "cci4")]
yB = [het[k]["yB"] for k in ("cci0", "cci2", "cci4")]
gam = [y * (ratio - 1) for y in yB]
print(f"[21-4] truth {[round(v) for v in truth]}; B cost {[round(v) for v in yB]}; gamma {[round(v) for v in gam]} (ratio {ratio:.3f}); ols {ols:.0f}")
p = Plot((-0.4, 2.4), (300, 750), h=320, ml=64, mb=70, ylabel="A로 더해지는 1인당 1년 비용 (만원)",
         xticks=[0, 1, 2], xticklabels=[(0, "동반질환 0점"), (1, "2점"), (2, "4점")], yticks=[300, 400, 500, 600, 700])
p.line(xs, truth, s=1, w=2.5)
p.points(xs, truth, s=1, r=5)
p.line(xs, gam, s=2, dash=True, w=2)
p.points(xs, gam, s=2, r=5)
p.line([-0.3, 2.3], [ols, ols], s=4, dash=True, w=1.5)
for x, v, g in zip(xs, truth, gam):
    up = v > g
    p.text(x, v, f"{v:.0f}", dx=8, dy=-9 if up else 19, cls="lbl strong")
    p.text(x, g, f"{g:.0f}", dx=8, dy=19 if up else -9, cls="lbl")
for x, y in zip(xs, yB):
    p.text(x, 300, f"B 비용 {y:,.0f}", anchor="middle", dy=50, cls="lbl mute small")
p.legend([("참값 (모의 자료)", 1, "line"), (f"감마 GLM (B 비용 × {ratio - 1:.3f})", 2, "dash"), (f"선형회귀 (모두 {ols:.0f})", 4, "dash")], X=None, Y=None)
save("ch21_hetamt", figure(p.svg("동반질환 점수별로 신약 A가 더하는 1년 비용"),
     f"그림 21-4. 동반질환 점수별로 신약 A가 더하는 1인당 1년 비용(만원). 실선의 참값은 원래 비용이 큰 환자일수록 작아집니다. "
     f"비 하나({ratio:.2f})를 모든 환자에게 쓰는 감마 GLM은 원래 비용(B 비용)에 비례해 금액을 키우고(주황 파선), 선형회귀는 모든 환자에게 같은 금액을 더합니다(회색 파선). "
     f"감마 GLM의 점수별 금액은 B 비용 × ({ratio:.3f} − 1)로 어림한 값입니다."))
print("figs written")
