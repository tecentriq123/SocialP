"""Figures for 14장 아·자·차·카 절 (content/_ch14/sc.html).
run: source /home/claude/pylibs/env.sh && python3 gen/fig_ch14c.py
Caption numbers (그림 14-n) are provisional; the chapter editor renumbers after merging.
"""
import sys, os, io, contextlib, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from svgplot import Plot, figure, panel_title

with contextlib.redirect_stdout(io.StringIO()):
    import nums_ch14c as N

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def m(v, nd=1):
    s = f"{v:.{nd}f}"
    return s.replace("-", "−")


def ci_row(p, y, est, lo, hi, s=1, sz=5, w=2.4):
    p.seg(lo, y, hi, y, cls=f"ln s{s}", w=w)
    p.seg(lo, y - 0.12, lo, y + 0.12, cls=f"ln s{s}", w=w * 0.75)
    p.seg(hi, y - 0.12, hi, y + 0.12, cls=f"ln s{s}", w=w * 0.75)
    X, Y = p.sx(est), p.sy(y)
    p.els.append(f'<rect x="{X - sz:.1f}" y="{Y - sz:.1f}" width="{2 * sz}" height="{2 * sz}" class="f{s}"/>')


# ------------------------------------------------------------------ 아: cost histograms
panels = []
YMAX = 40
for g, title, s in (("C", "가. 통상관리군 (500명)", 4), ("I", "나. 약물검토 중재군 (500명)", 1)):
    y = N.COST[g]
    cs = N.CS[g]
    pos = y[y > 0]
    edges = N.HIST_EDGES
    cnt, _ = np.histogram(np.clip(pos, 0, edges[-1] - 1e-6), bins=edges)
    over = int((pos >= edges[-1]).sum())
    p = Plot((-150, 3000), (0, YMAX), w=420, h=300, ml=44, mr=14, mt=40, mb=50,
             xlabel="12개월 입원 진료비 (만원)", ylabel="환자 수",
             xticks=[0, 500, 1000, 1500, 2000, 2500, 3000],
             xticklabels=[(0, "0"), (500, "500"), (1000, "1,000"), (1500, "1,500"), (2000, "2,000"), (2500, "2,500"), (3000, "3,000+")],
             yticks=[0, 10, 20, 30, 40])
    # zero bar (clipped)
    zero = cs["n"] - cs["npos"]
    p.bars([-60], [YMAX], 90, s=s, rounded=False)
    p.text(-15, YMAX * 0.92, f"← 0원 {zero}명({zero / cs['n'] * 100:.1f}%), 위를 자름", anchor="start", dx=4, cls="lbl small strong")
    p.hist(edges, cnt, s=s, gap=1.5)
    if over:
        p.text(2950, cnt[-1] if len(cnt) else 0, f"+{over}명", anchor="middle", dy=-8, cls="lbl small mute")
    p.vline(cs["mean"], cls="ref strongref", dash=False, w=1.6, y1=YMAX * 0.78)
    p.text(cs["mean"], YMAX * 0.78, f"평균 {cs['mean']:.1f}", anchor="start", dx=4, dy=-4, cls="lbl small strong")
    p.text(cs["mean"], YMAX * 0.78, "중앙값은 0", anchor="start", dx=4, dy=12, cls="lbl small mute")
    p.text(1500, YMAX * 0.45, f"입원한 {cs['npos']}명의 평균 {cs['pos_mean']:.1f}", anchor="start", cls="lbl small")
    p.text(1500, YMAX * 0.45, f"상위 10% 환자가 총비용의 {cs['top10'] * 100:.0f}%", anchor="start", dy=16, cls="lbl small")
    panel_title(p, title)
    panels.append(p.svg(f"{title} 입원 진료비 분포"))
save("ch14_h_cost", figure(panels,
     f"그림 14-10. 가상의 약물검토 시험에서 12개월 입원 진료비의 분포(100만원 간격, 0원 막대는 위를 잘라 표시). "
     f"두 군 모두 절반 이상이 0원이라 중앙값은 0으로 같고, 입원한 소수가 긴 오른쪽 꼬리를 만듭니다. "
     f"총비용과 예산을 정하는 것은 꼬리까지 모두 반영한 산술평균(굵은 세로선)입니다.", cols=2))

# ------------------------------------------------------------------ 자: method comparison
rows = N.MI_TAB
W, H = 640, 300
p = Plot((-8.5, 0.5), (0.3, len(rows) + 0.5), w=W, h=H, ml=250, mr=150, mt=26, mb=50, show_yaxis=False, ygrid=False,
         xlabel="12개월 수축기혈압의 군간 차이 (중재 − 통상, mmHg)", xticks=[-8, -6, -4, -2, 0],
         xticklabels=[(-8, "−8"), (-6, "−6"), (-4, "−4"), (-2, "−2"), (0, "0")])
p.vline(0, dash=False, cls="ref", w=1.0)
full = rows[0]["est"]
p.vline(full, dash=True, cls="ref strongref", w=1.2)
p.text(full, len(rows) + 0.5, "참값에 가까운 기준", anchor="middle", dy=-8, cls="lbl small mute")
labs = {"full": ("결측 없는 전체 자료", "모의자료라서 알 수 있음"),
        "cc": ("완전사례 분석", "12개월 측정자 223명만"),
        "mean": ("군별 평균으로 단일 대체", "빈칸을 한 값으로 채움"),
        "noaux": ("다중대체: 군 + 기저 SBP", "결측을 설명하는 정보 없음"),
        "aux": ("다중대체: + 3개월 SBP", "결측을 설명하는 정보 포함")}
col = {"full": 4, "cc": 2, "mean": 2, "noaux": 2, "aux": 1}
for i, r in enumerate(rows):
    yy = len(rows) - i
    ci_row(p, yy, r["est"], r["lo"], r["hi"], s=col[r["key"]])
    a, b = labs[r["key"]]
    p.text_px(12, p.sy(yy) - 1, a, cls="lbl strong")
    p.text_px(12, p.sy(yy) + 15, b, cls="lbl small mute")
    p.text_px(W - 150 + 12, p.sy(yy) + 5, f"{m(r['est'], 2)} ({m(r['lo'], 1)} ~ {m(r['hi'], 1)})", cls="lbl small")
save("ch14_i_mi", figure(p.svg("결측 처리 방법별 12개월 수축기혈압 군간 차이와 95% 신뢰구간"),
     f"그림 14-11. 같은 가상 시험 자료(300명, 12개월 결측 {N.NM['miss']}명)를 결측 처리 방법만 바꿔 분석한 결과. "
     f"탈락이 관측된 3개월 혈압에 따라 생겼기 때문에(MAR), 3개월 혈압을 대체모형에 넣은 다중대체만 결측이 없었을 때의 값(점선)에 가깝습니다. "
     f"결측을 설명하는 정보가 없는 다중대체는 완전사례 분석과 비슷하게 효과를 작게 추정합니다."))

# ------------------------------------------------------------------ 차: n per group vs difference
lg = math.log10
yt = [50, 100, 200, 500, 1000, 2000]
p = Plot((4, 26), (lg(50), lg(2500)), w=600, h=330, ml=76, mr=24, mt=24, mb=54,
         xlabel="검출하려는 순응률 차이 (%p, 통상관리군 50% 기준)", ylabel="군당 필요 인원 (로그 눈금)",
         xticks=[5, 10, 15, 20, 25], yticks=[lg(v) for v in yt],
         ytickfmt=lambda v: f"{round(10 ** v):,}")
xs = N.CURVE_D * 100
p.line(xs, np.log10(N.CURVE90), s=2, dash=True)
p.line(xs, np.log10(N.CURVE80), s=1)
for d, nn in ((15, N.n_two_prop(0.5, 0.65)), (10, N.n_two_prop(0.5, 0.60)), (7.5, N.n_two_prop(0.5, 0.575))):
    p.points([d], [lg(nn)], s=1, r=4.5)
    p.text(d, lg(nn), f"{d:g}%p → {math.ceil(nn)}명", anchor="start", dx=8, dy=-6, cls="lbl small strong")
p.legend([("검정력 80%", 1, "line"), ("검정력 90%", 2, "dash")], X=430, Y=44)
save("ch14_j_ss", figure(p.svg("검출하려는 차이에 따른 군당 필요 인원"),
     "그림 14-12. 두 비율 비교에서 검출하려는 차이와 필요한 인원(양측 α 0.05). 차이를 15%p에서 10%p로 줄이면 인원은 약 2.3배, "
     "7.5%p(절반)로 줄이면 약 4배가 됩니다. 필요 인원은 대략 차이의 제곱에 반비례합니다."))

# ------------------------------------------------------------------ 카: scatter + Bland-Altman
BA = N.BA
p1 = Plot((95, 180), (95, 180), w=420, h=340, ml=52, mr=16, mt=40, mb=52,
          xlabel="기준 혈압계 (mmHg)", ylabel="새 손목 혈압계 (mmHg)", xticks=[100, 120, 140, 160, 180], yticks=[100, 120, 140, 160, 180])
p1.line([95, 180], [95, 180], s=4, dash=True, w=1.4)
p1.points(N.REF, N.NEW, s=1, r=3.8)
p1.text(170, 176, "같은 값 선", anchor="end", dy=-6, cls="lbl small mute")
p1.text(100, 170, f"r = {BA['r']:.2f}", anchor="start", cls="lbl strong")
panel_title(p1, "가. 산점도: 함께 움직이는가")
p2 = Plot((95, 180), (-20, 25), w=420, h=340, ml=52, mr=74, mt=40, mb=52,
          xlabel="두 혈압계의 평균 (mmHg)", ylabel="차이: 손목 − 기준 (mmHg)", xticks=[100, 120, 140, 160, 180],
          yticks=[-20, -10, 0, 10, 20])
p2.hline(0, cls="ref", dash=False, w=1.0)
p2.hline(BA["mean"], cls="ref strongref", dash=False, w=1.6)
p2.hline(BA["lo"], cls="ref strongref", dash=True, w=1.3)
p2.hline(BA["hi"], cls="ref strongref", dash=True, w=1.3)
p2.points(N.avg, N.dif, s=1, r=3.8)
XR = 420 - 74 + 6
p2.text_px(XR, p2.sy(BA["hi"]) - 3, "+1.96 SD", cls="lbl small")
p2.text_px(XR, p2.sy(BA["hi"]) + 12, m(BA['hi']), cls="lbl small")
p2.text_px(XR, p2.sy(BA["mean"]) - 3, "평균 차이", cls="lbl small strong")
p2.text_px(XR, p2.sy(BA["mean"]) + 12, m(BA['mean']), cls="lbl small strong")
p2.text_px(XR, p2.sy(BA["lo"]) - 3, "−1.96 SD", cls="lbl small")
p2.text_px(XR, p2.sy(BA["lo"]) + 12, m(BA['lo']), cls="lbl small")
panel_title(p2, "나. Bland–Altman 그림: 얼마나 다른가")
save("ch14_k_ba", figure([p1.svg("두 혈압계 측정값의 산점도"), p2.svg("두 혈압계의 Bland–Altman 그림")],
     f"그림 14-13. 가상의 혈압계 비교 연구(50명). 가: 점들이 한 직선 근처에 모여 r = {BA['r']:.2f}로 높지만, 대부분 같은 값 선(점선)보다 위에 있습니다. "
     f"나: 가로축은 두 측정의 평균, 세로축은 차이입니다. 가운데 굵은 선이 평균 차이(치우침, {m(BA['mean'])} mmHg), "
     f"두 점선이 95% 일치 한계({m(BA['lo'])} ~ {m(BA['hi'])} mmHg)입니다. 한 사람에서 새 기기가 기준보다 9 mmHg 낮게부터 17 mmHg 높게까지 잴 수 있다는 뜻입니다.",
     cols=2))
print("figs written")
