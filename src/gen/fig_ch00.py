"""Figures for 0장 (그림 0-1 spread, 0-2 sampling, 0-3 shuffle, 0-4 ci20 — numbered in order of appearance). Every number in the captions is computed and printed here.
run: source /home/claude/pylibs/env.sh && cd gen && python3 fig_ch00.py
Writes figs/ch00_spread.html, ch00_sampling.html, ch00_ci20.html, ch00_shuffle.html.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def kfmt(v, nd=2):
    return f"{v:.{nd}f}".replace("-", "−")


# ----------------------------------------------------------------------------------------------
# 그림 0-1. 평균은 같고 흩어짐이 다른 두 병동 (c 절): 값 다섯 개씩, 계산은 본문 표와 같음
A = np.array([8, 12, 9, 11, 10])
B = np.array([4, 16, 6, 14, 10])
sdA, sdB = A.std(ddof=1), B.std(ddof=1)
print(f"[0-1] A mean {A.mean():.1f} SD {sdA:.2f}; B mean {B.mean():.1f} SD {sdB:.2f}")
p = Plot((0, 20), (0, 3), h=230, ml=86, mb=50, xlabel="재원일수 (일)", xticks=[0, 4, 8, 12, 16, 20],
         yticks=[], ygrid=False, show_yaxis=False, xgrid=True)
for row, vals, sd, s, lab in ((2.1, A, sdA, 1, "A 병동"), (0.9, B, sdB, 2, "B 병동")):
    p.seg(10 - sd, row - 0.42, 10 + sd, row - 0.42, cls="ref", w=2)
    p.seg(10 - sd, row - 0.50, 10 - sd, row - 0.34, cls="ref", w=2)
    p.seg(10 + sd, row - 0.50, 10 + sd, row - 0.34, cls="ref", w=2)
    p.points(vals, [row] * len(vals), s=s, r=7)
    p.text(0, row, lab, anchor="end", dx=-10, dy=5, cls="lbl strong")
    p.text(10 - sd, row - 0.42, f"평균 ± SD ({kfmt(sd)}일)", dy=16, cls="lbl mute small")
p.vline(10, y0=0.2, y1=2.75)
p.text(10, 2.75, "평균 10일", anchor="middle", dy=-6, cls="lbl mute")
save("ch00_spread", figure(p.svg("평균이 같은 두 병동의 재원일수"),
     f"그림 0-1. 점 하나가 환자 한 명의 재원일수입니다. "
     f"점 아래 가로 막대가 평균 ± 표준편차로, A 병동은 ±{kfmt(sdA)}일, B 병동은 ±{kfmt(sdB)}일입니다."))

# ----------------------------------------------------------------------------------------------
# 그림 0-2. 환자 한 명의 값, 30명 평균, 120명 평균의 흩어짐 (e 절)
# 신약군의 참 평균 12.4, 참 SD 7.0으로 놓은 가상의 모집단에서 뽑는다.
rng = np.random.default_rng(2610)
MU, SIG, NREP = 12.4, 7.0, 10000
indiv = rng.normal(MU, SIG, NREP)
m30 = rng.normal(MU, SIG, (NREP, 30)).mean(axis=1)
m120 = rng.normal(MU, SIG, (NREP, 120)).mean(axis=1)
sd_i, sd30, sd120 = indiv.std(ddof=1), m30.std(ddof=1), m120.std(ddof=1)
print(f"[0-2] SD of 10,000 patients {sd_i:.2f}; SD of 10,000 means n=30 {sd30:.2f} (theory {SIG / np.sqrt(30):.2f}); "
      f"n=120 {sd120:.2f} (theory {SIG / np.sqrt(120):.2f}); ratio {sd30 / sd120:.2f}")
edges = np.arange(-10, 35.01, 0.5)
panels = []
for vals, sd, title in ((indiv, sd_i, "환자 한 명의 감소량 10,000개"),
                        (m30, sd30, "30명 평균 10,000개"),
                        (m120, sd120, "120명 평균 10,000개")):
    cnt, _ = np.histogram(vals, edges)
    ymax = cnt.max() * 1.08
    q = Plot((-10, 35), (0, ymax), h=150, ml=24, mt=30, mb=40, xlabel="", yticks=[], ygrid=False, show_yaxis=False,
             xticks=[-10, 0, 10, 20, 30])
    q.hist(edges, cnt, s=1)
    q.vline(MU, y1=ymax)
    panel_title(q, f"{title}: 흩어짐(SD) {kfmt(sd)} mmHg")
    panels.append(q)
panels[-1].xlabel = "8주 수축기혈압 감소량 (mmHg)"
panels[-1].h = 170
panels[-1].mb = 56
save("ch00_sampling", figure([q.svg(t) for q, t in zip(panels, ("환자 한 명", "30명 평균", "120명 평균"))],
     f"그림 0-2. 참 평균이 12.4 mmHg, 환자 사이의 표준편차가 7.0 mmHg인 가상의 모집단에서 뽑은 결과입니다. "
     f"위는 환자 한 명의 값 10,000개, 가운데는 30명씩 10,000번 뽑아 낸 평균, 아래는 120명씩 10,000번 뽑아 낸 평균입니다(막대의 높이는 패널마다 따로 맞춤). "
     f"평균들의 흩어짐이 표준오차이고, 30명이면 {kfmt(sd30)}, 120명이면 {kfmt(sd120)} mmHg입니다(식 7.0 ÷ √n으로는 1.28과 0.64). "
     f"환자 한 명의 흩어짐({kfmt(sd_i)} mmHg)은 인원과 관계없습니다."))

# ----------------------------------------------------------------------------------------------
# 그림 0-4. 같은 시험을 20번 되풀이했을 때 차이의 95% 신뢰구간 (f 절)
# 참 차이 4 mmHg(두 군 SD 7), 군당 30명. 본문과 같은 방식(합동 분산, 자유도 58)으로 구간을 만든다.
TRUE_D, NG = 4.0, 30
rng = np.random.default_rng(58)
rows = []
for k in range(20):
    a = rng.normal(12.0, 7.0, NG)
    b = rng.normal(12.0 - TRUE_D, 7.0, NG)
    r = st.ttest_ind(a, b)
    ci = r.confidence_interval()
    d = a.mean() - b.mean()
    rows.append((d, ci.low, ci.high, not (ci.low <= TRUE_D <= ci.high)))
miss = sum(r[3] for r in rows)
# coverage of the same procedure over many repetitions
rng2 = np.random.default_rng(1)
big = 20000
aa = rng2.normal(12.0, 7.0, (big, NG)); bb = rng2.normal(12.0 - TRUE_D, 7.0, (big, NG))
dd = aa.mean(1) - bb.mean(1)
sp = np.sqrt((aa.var(1, ddof=1) + bb.var(1, ddof=1)) / 2) * np.sqrt(2 / NG)
tc = st.t.ppf(0.975, 2 * NG - 2)
cover = np.mean((dd - tc * sp <= TRUE_D) & (TRUE_D <= dd + tc * sp))
print(f"[0-4] 20 trials: misses {miss}; diffs {min(r[0] for r in rows):.1f} to {max(r[0] for r in rows):.1f}; "
      f"coverage over {big} trials {cover:.3f}")
p = Plot((-4, 13), (0.3, 20.7), h=440, ml=70, mb=50, xlabel="두 군 평균 감소량의 차이 (mmHg)",
         xticks=[-4, 0, 4, 8, 12], yticks=[], ygrid=False, show_yaxis=False, xgrid=True)
p.vline(TRUE_D, y0=0.3, y1=20.7, dash=False, cls="ref strongref", w=2.4)
p.vline(0, y0=0.3, y1=20.7)
for i, (d, lo, hi, ms) in enumerate(rows):
    y = 20 - i
    s = 2 if ms else 1
    p.seg(lo, y, hi, y, cls=f"ln s{s}", w=3 if ms else 2)
    p.points([d], [y], s=s, r=4)
p.text(TRUE_D, 20.7, "참 차이 4", anchor="middle", dy=-6, cls="lbl strong")
p.text(-4, 10.5, "시험 20번", anchor="end", dx=-10, cls="lbl mute")
save("ch00_ci20", figure(p.svg("같은 시험을 20번 되풀이한 95% 신뢰구간"),
     f"그림 0-4. 참 차이가 4 mmHg인 가상의 시험(군당 30명)을 20번 되풀이하고, 시험마다 95% 신뢰구간을 만들었습니다. 구간을 만든 방법은 본문의 0.6–8.0 mmHg를 구할 때와 같습니다. "
     f"점은 그 시험의 평균 차이, 가로선은 신뢰구간, 굵은 세로선은 참 차이, 점선은 0입니다. 주황은 참 차이를 놓친 구간입니다."))

# ----------------------------------------------------------------------------------------------
# 그림 0-3. 효과가 없다면 생길 차이: 군 이름을 섞어 다시 나누기 (f 절)
# 본문 요약값(신약 12.4 ± 7.0, 위약 8.1 ± 7.2, 각 30명)을 그대로 갖도록 만든 가상의 60명.
rng = np.random.default_rng(30)


def exact(m, s, n):
    z = rng.normal(size=n)
    z = (z - z.mean()) / z.std(ddof=1)
    return m + s * z


drug, plac = exact(12.4, 7.0, 30), exact(8.1, 7.2, 30)
obs = drug.mean() - plac.mean()
allv = np.r_[drug, plac]
NP = 10000
diffs = np.empty(NP)
for k in range(NP):
    perm = rng.permutation(allv)
    diffs[k] = perm[:30].mean() - perm[30:].mean()
p_perm = np.mean(np.abs(diffs) >= obs - 1e-9)
tt = st.ttest_ind(drug, plac)
print(f"[0-3] check means {drug.mean():.2f} {plac.mean():.2f} SDs {drug.std(ddof=1):.2f} {plac.std(ddof=1):.2f}; obs {obs:.2f}; "
      f"shuffled SD {diffs.std(ddof=1):.2f}; share |diff| >= 4.3: {p_perm:.4f} ({int(round(p_perm * NP))} of {NP}); "
      f"t-test p {tt.pvalue:.4f}; shuffles between -2 and 2: {np.mean(np.abs(diffs) < 2):.3f}")
edges = np.linspace(-8.6, 8.6, 41)  # 폭 0.43: ±4.3이 막대 경계에 오도록
cnt, _ = np.histogram(diffs, edges)
ymax = 900
p = Plot((-8.6, 8.6), (0, ymax), h=300, xlabel="섞어서 다시 나눈 두 군의 평균 차이 (mmHg)", ylabel="횟수",
         xticks=[-8, -4, 0, 4, 8], yticks=[0, 300, 600, 900])
mid = (edges[:-1] + edges[1:]) / 2
inner = np.where(np.abs(mid) < obs, cnt, 0)
outer = np.where(np.abs(mid) >= obs, cnt, 0)
assert cnt.max() < ymax, cnt.max()
p.hist(edges, inner, s=1)
p.hist(edges, outer, s=2)
for v in (obs, -obs):
    p.vline(v, y1=ymax * 0.82)
p.text(obs, ymax * 0.82, "실제 차이 4.3", anchor="middle", dy=-6, cls="lbl strong")
p.text(-obs, ymax * 0.82, "−4.3", anchor="middle", dy=-6, cls="lbl mute")
p.text(6.2, ymax * 0.35, f"{p_perm * 100:.1f}%", anchor="middle", cls="lbl strong")
p.text(6.2, ymax * 0.35, "양쪽 꼬리", anchor="middle", dy=16, cls="lbl mute small")
save("ch00_shuffle", figure(p.svg("군 이름을 섞어 다시 나눈 평균 차이의 분포"),
     f"그림 0-3. 신약·위약 이름을 무작위로 섞어 30명씩 다시 나눌 때마다 구한 두 군의 평균 차이 {NP:,}개의 분포입니다. "
     f"주황 막대는 실제 차이 4.3 mmHg만큼 또는 그보다 더 벌어진 경우(양쪽 꼬리)입니다. "
     f"60명의 값은 본문 요약값(12.4 ± 7.0, 8.1 ± 7.2)을 그대로 갖도록 만든 가상의 자료입니다."))
print("figs written")
