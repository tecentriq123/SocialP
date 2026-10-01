import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, scipy.stats as st
from svgplot import Plot, figure, panel_title, fmt

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


# 1. null distribution t(58)
df = 58
x = np.linspace(-4, 4, 401)
y = st.t.pdf(x, df)
p = Plot((-4, 4), (0, 0.45), xlabel="t 값", ylabel="확률밀도", yticks=[0, 0.1, 0.2, 0.3, 0.4],
         xticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4])
tobs = 2.345
for side in (1, -1):
    xs = np.linspace(tobs, 4, 80) if side == 1 else np.linspace(-4, -tobs, 80)
    p.fill_between(xs, np.zeros_like(xs), st.t.pdf(xs, df), s=1)
p.line(x, y, s=4)
tc = st.t.ppf(0.975, df)
for v in (tc, -tc):
    p.vline(v, y1=0.30)
p.text(-tc, 0.31, "−2.00", anchor="middle", dy=-4)
p.text(tc, 0.31, "+2.00", anchor="middle", dy=-4)
p.text(0, 0.36, "기각역 경계 (α = 0.05, 양측)", anchor="middle", dy=-18, cls="lbl mute")
p.points([tobs], [0], s=1, r=5)
p.text(tobs, 0.0, "관측된 t = 2.35", dx=6, dy=-40)
p.seg(tobs + 0.25, 0.012, tobs + 0.1, 0.004, cls="ref", w=1)
p.text(3.1, 0.07, "p = 0.022", anchor="middle", cls="lbl strong")
p.text(3.1, 0.07, "양쪽 꼬리 넓이의 합", anchor="middle", dy=16, cls="lbl mute small")
save("ch01_null", figure(p.svg("귀무가설 하의 t 분포와 관측된 t 값"),
     "그림 1-1. 귀무가설이 참일 때 t 통계량의 분포(자유도 58). 파란 영역(양쪽 꼬리)의 넓이가 p-value입니다. 관측된 t가 점선(±2.00) 바깥에 있으므로 유의수준 5%에서 귀무가설을 기각합니다."))

# 2. power
ncp = 2.213
x = np.linspace(-3.5, 6, 500)
p = Plot((-3.5, 6), (0, 0.45), xlabel="검정통계량 (Z)", ylabel="확률밀도", yticks=[0, 0.1, 0.2, 0.3, 0.4],
         xticks=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
crit = 1.96
xs = np.linspace(crit, 6, 120)
p.fill_between(xs, np.zeros_like(xs), st.norm.pdf(xs, ncp), s=1)
xs2 = np.linspace(-3.5, crit, 200)
p.fill_between(xs2, np.zeros_like(xs2), st.norm.pdf(xs2, ncp), cls="a4")
xs3 = np.linspace(crit, 6, 80)
p.fill_between(xs3, np.zeros_like(xs3), st.norm.pdf(xs3, 0), s=2)
p.line(x, st.norm.pdf(x, 0), s=4, dash=True)
p.line(x, st.norm.pdf(x, ncp), s=1)
p.vline(crit, y1=0.44, dash=False, cls="ref strongref")
p.text(crit, 0.44, "임계값 1.96", dx=6, dy=4)
p.text(-1.25, 0.33, "귀무가설 분포", anchor="end")
p.text(-1.25, 0.33, "(실제 차이 = 0)", anchor="end", dy=16, cls="lbl mute small")
p.text(3.45, 0.33, "대립가설 분포", anchor="start")
p.text(3.45, 0.33, "(실제 차이 = 4 mmHg)", anchor="start", dy=16, cls="lbl mute small")
p.text(3.2, 0.14, "검정력 1−β ≈ 0.60", anchor="start", cls="lbl strong")
p.text(1.2, 0.05, "β ≈ 0.40", anchor="end", cls="lbl")
p.text(2.35, 0.012, "α/2", anchor="start", dy=-6, cls="lbl small")
save("ch01_power", figure(p.svg("귀무가설 분포와 대립가설 분포, 검정력"),
     "그림 1-2. 실제 차이가 4 mmHg(SD 7, 군당 30명)일 때의 검정력. 대립가설 분포 중 임계값(1.96) 오른쪽 넓이(파랑)가 검정력, 왼쪽 넓이(회색)가 2종 오류 β입니다. 주황은 1종 오류 α의 오른쪽 절반입니다(정규근사)."))

# 3. FWER
k = np.arange(1, 21)
fw = 1 - 0.95 ** k
p = Plot((0, 21), (0, 0.8), xlabel="독립적인 검정의 수 (k)", ylabel="1종 오류가 하나라도 생길 확률",
         yticks=[0, 0.2, 0.4, 0.6, 0.8], ytickfmt=lambda v: f"{int(round(v * 100))}%",
         xticks=[1, 5, 10, 15, 20])
p.hline(0.05, dash=True)
p.text(20.8, 0.05, "α = 5%", anchor="end", dy=-6, cls="lbl mute small")
p.line(k, fw, s=1)
sel = [1, 5, 10, 20]
p.points(sel, [1 - 0.95 ** i for i in sel], s=1, r=4.5)
for i in sel:
    v = 1 - 0.95 ** i
    p.text(i, v, f"{v * 100:.1f}%", anchor="middle", dy=-12, cls="lbl strong")
save("ch01_fwer", figure(p.svg("검정 수에 따른 가족단위 오류율"),
     "그림 1-3. 서로 독립인 검정을 각각 α = 0.05로 k번 할 때, 적어도 하나에서 우연히 '유의'가 나올 확률 \\(1-(1-0.05)^k\\)."))

# 4. t distributions by df
x = np.linspace(-5, 5, 501)
p = Plot((-5, 5), (0, 0.42), xlabel="값", ylabel="확률밀도", yticks=[0, 0.1, 0.2, 0.3, 0.4])
p.line(x, st.norm.pdf(x), s=4, dash=True)
p.line(x, st.t.pdf(x, 2), s=2)
p.line(x, st.t.pdf(x, 5), s=3)
p.line(x, st.t.pdf(x, 30), s=1)
p.legend([("표준정규분포", 4, "dash"), ("t 분포, 자유도 30", 1, "line"), ("t 분포, 자유도 5", 3, "line"), ("t 분포, 자유도 2", 2, "line")])
p.text(3.6, st.t.pdf(3.6, 2), "꼬리가 두꺼움", dx=4, dy=-10, cls="lbl mute small")
save("ch01_tdist", figure(p.svg("자유도에 따른 t 분포"),
     "그림 1-4. 자유도가 작을수록 t 분포는 꼬리가 두꺼워지고, 자유도가 커지면 표준정규분포에 거의 겹칩니다."))

# 5. normal 68-95
x = np.linspace(-4, 4, 401)
p = Plot((-4, 4), (0, 0.45), xlabel="평균으로부터 떨어진 거리 (표준편차 단위, z)", ylabel="확률밀도",
         yticks=[0, 0.1, 0.2, 0.3, 0.4], xticks=[-3, -2, -1, 0, 1, 2, 3],
         xtickfmt=lambda v: ("μ" if v == 0 else (f"μ{'+' if v > 0 else '−'}{abs(int(v))}σ")))
xs = np.linspace(-1.96, 1.96, 200)
p.fill_between(xs, np.zeros_like(xs), st.norm.pdf(xs), s=1)
xs = np.linspace(-1, 1, 120)
p.fill_between(xs, np.zeros_like(xs), st.norm.pdf(xs), s=1)
p.line(x, st.norm.pdf(x), s=1)
p.text(0, 0.16, "68.3%", anchor="middle", cls="lbl strong")
p.text(0, 0.16, "(±1σ)", anchor="middle", dy=16, cls="lbl mute small")
p.text(0, 0.03, "95% (±1.96σ)", anchor="middle", cls="lbl")
p.text(-3.0, 0.03, "2.5%", anchor="middle", cls="lbl mute small")
p.text(3.0, 0.03, "2.5%", anchor="middle", cls="lbl mute small")
save("ch01_normal", figure(p.svg("정규분포의 68-95 규칙"),
     "그림 1-5. 정규분포에서는 평균 ±1 표준편차 안에 약 68%, ±1.96 표준편차 안에 95%의 값이 들어갑니다."))

# 6. chi-square & F
x = np.linspace(0.01, 16, 400)
x1 = np.linspace(0.3, 16, 400)
p1 = Plot((0, 16), (0, 0.5), w=420, h=300, ml=52, xlabel="χ² 값", ylabel="확률밀도",
          yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], xticks=[0, 4, 8, 12, 16])
p1.line(x1, st.chi2.pdf(x1, 1), s=1)
p1.line(x, st.chi2.pdf(x, 4), s=2)
p1.line(x, st.chi2.pdf(x, 8), s=3)
p1.legend([("자유도 1", 1, "line"), ("자유도 4", 2, "line"), ("자유도 8", 3, "line")], X=250)
panel_title(p1, "카이제곱(χ²) 분포")
x = np.linspace(0.01, 6, 400)
p2 = Plot((0, 6), (0, 1.0), w=420, h=300, ml=52, xlabel="F 값", ylabel="확률밀도",
          yticks=[0, 0.25, 0.5, 0.75, 1.0], xticks=[0, 1, 2, 3, 4, 5, 6])
fc = st.f.ppf(0.95, 2, 57)
xs = np.linspace(fc, 6, 80)
p2.fill_between(xs, np.zeros_like(xs), st.f.pdf(xs, 2, 57), s=2)
p2.line(x, st.f.pdf(x, 2, 57), s=1)
p2.vline(fc, y1=0.45)
p2.text(fc, 0.45, f"F = {fc:.2f}", anchor="middle", dy=-6)
p2.text(fc + 0.3, 0.1, "상위 5%", cls="lbl small")
panel_title(p2, "F 분포 (자유도 2, 57)")
save("ch01_chisq_f", figure([p1.svg("자유도별 카이제곱 분포"), p2.svg("F 분포")],
     "그림 1-6. 카이제곱 분포와 F 분포는 0 이상의 값만 가지며 오른쪽으로 치우쳐 있습니다. 그래서 이 두 검정은 보통 '오른쪽 꼬리'만 봅니다.", cols=2))

# 7. exploration: histogram + QQ for normal SBP and skewed LOS
rng = np.random.default_rng(7)
sbp = np.round(rng.normal(135, 15, 60), 0)
los = np.round(np.exp(rng.normal(1.9, 0.7, 60)), 0).clip(1)
res = {}
for name, data in (("sbp", sbp), ("los", los)):
    w, pv = st.shapiro(data)
    ks = st.kstest((data - data.mean()) / data.std(ddof=1), "norm")
    res[name] = (w, pv, data.mean(), data.std(ddof=1), np.median(data), np.percentile(data, 25), np.percentile(data, 75), st.skew(data), ks.pvalue)
with open(os.path.join(OUT, "ch01_explore_stats.txt"), "w") as f:
    for k_, v in res.items():
        f.write(f"{k_}: W={v[0]:.3f} p={v[1]:.4f} mean={v[2]:.1f} sd={v[3]:.1f} median={v[4]} q1={v[5]} q3={v[6]} skew={v[7]:.2f} ks_p={v[8]:.3f}\n")
    f.write("los sorted: " + ",".join(str(int(v)) for v in sorted(los)) + "\n")


def hist_panel(data, lo, hi, width, title, xlabel, ymax):
    edges = np.arange(lo, hi + width, width)
    cnt, _ = np.histogram(data, edges)
    pl = Plot((lo, hi), (0, ymax), w=420, h=280, ml=48, xlabel=xlabel, ylabel="환자 수",
              yticks=list(range(0, ymax + 1, 5)))
    pl.hist(edges, cnt, s=1, gap=2)
    panel_title(pl, title)
    return pl


def qq_panel(data, title, ylabel):
    n = len(data)
    srt = np.sort(data)
    q = st.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))  # Blom
    lo, hi = srt.min(), srt.max()
    pad = (hi - lo) * 0.08
    yt = None
    pl = Plot((-2.6, 2.6), (lo - pad, hi + pad), w=420, h=280, ml=48, xlabel="이론적 정규분위수", ylabel=ylabel,
              xticks=[-2, -1, 0, 1, 2])
    q1, q3 = np.percentile(data, [25, 75])
    z1, z3 = st.norm.ppf([0.25, 0.75])
    slope = (q3 - q1) / (z3 - z1)
    icpt = q1 - slope * z1
    xx = np.array([-2.6, 2.6])
    yy = icpt + slope * xx
    # clip line to y range
    pl.line(xx, np.clip(yy, lo - pad, hi + pad), s=4, dash=True, w=1.5)
    pl.points(q, srt, s=1, r=3.2)
    panel_title(pl, title)
    return pl


h1 = hist_panel(sbp, 95, 175, 10, "수축기혈압 — 히스토그램", "수축기혈압 (mmHg)", 20)
q1p = qq_panel(sbp, "수축기혈압 — Q-Q plot", "관측값 (mmHg)")
h2 = hist_panel(los, 0, 40, 4, "재원일수 — 히스토그램", "재원일수 (일)", 25)
q2p = qq_panel(los, "재원일수 — Q-Q plot", "관측값 (일)")
save("ch01_explore", figure([h1.svg("수축기혈압 히스토그램"), q1p.svg("수축기혈압 Q-Q plot"),
                             h2.svg("재원일수 히스토그램"), q2p.svg("재원일수 Q-Q plot")],
     "그림 1-8. 가상의 환자 60명 자료. 위: 수축기혈압은 종 모양이고 Q-Q plot의 점이 기준선을 따라갑니다. 아래: 재원일수는 오른쪽 꼬리가 길고, Q-Q plot의 점이 위로 휘어 올라갑니다.", cols=2))

# 8. boxplot anatomy using los
data = los
q1, med, q3 = np.percentile(data, [25, 50, 75])
iqr = q3 - q1
lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
wl = data[data >= lf].min()
wu = data[data <= uf].max()
outs = np.sort(data[(data > uf) | (data < lf)])
p = Plot((0, 40), (0, 1), w=640, h=230, ml=24, mr=24, mt=20, mb=50, xlabel="재원일수 (일)",
         show_yaxis=False, ygrid=False, xticks=[0, 5, 10, 15, 20, 25, 30, 35, 40])
yc = 0.5
bh = 0.16
p.els.append(f'<rect x="{p.sx(q1):.1f}" y="{p.sy(yc + bh):.1f}" width="{p.sx(q3) - p.sx(q1):.1f}" height="{p.sy(yc - bh) - p.sy(yc + bh):.1f}" class="a1 s1" stroke-width="1.5"/>')
p.seg(med, yc - bh, med, yc + bh, cls="ln s1", w=2.5)
p.seg(wl, yc, q1, yc, cls="ln s1", w=1.5)
p.seg(q3, yc, wu, yc, cls="ln s1", w=1.5)
p.seg(wl, yc - 0.07, wl, yc + 0.07, cls="ln s1", w=1.5)
p.seg(wu, yc - 0.07, wu, yc + 0.07, cls="ln s1", w=1.5)
p.points(outs, [yc] * len(outs), s=1, r=3.5, hollow=True)
p.text(med, yc + bh, f"중앙값 {med:g}", anchor="middle", dy=-10, cls="lbl strong")
p.text(q1, yc - bh, f"Q1 {q1:g}", anchor="middle", dy=18)
p.text(q3, yc - bh, f"Q3 {q3:g}", anchor="middle", dy=18)
p.text((q1 + q3) / 2, yc - bh, "", anchor="middle")
p.text(wu, yc, f"수염 끝 {wu:g}", anchor="middle", dy=-24, cls="lbl mute small")
p.text(outs.mean() if len(outs) else 35, yc, "이상값 (Q3 + 1.5×IQR 초과)", anchor="middle", dy=-22, cls="lbl mute small")
p.seg(q1, 0.12, q3, 0.12, cls="ref", w=1)
p.text((q1 + q3) / 2, 0.12, f"IQR = {iqr:g}일 (가운데 50%)", anchor="middle", dy=14, cls="lbl small")
save("ch01_boxplot", figure(p.svg("상자그림의 구성"),
     f"그림 1-7. 상자그림(box plot) 읽는 법. 상자의 양 끝은 1사분위수(Q1)와 3사분위수(Q3), 가운데 선은 중앙값입니다. 수염은 Q1 − 1.5×IQR ~ Q3 + 1.5×IQR 범위 안의 가장 먼 관측값까지 그리고, 그 밖의 값은 점(이상값)으로 표시합니다."))
print(open(os.path.join(OUT, "ch01_explore_stats.txt")).read())
print("box", q1, med, q3, iqr, wl, wu, outs)
