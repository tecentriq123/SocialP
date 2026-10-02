"""Figures for chapter 21 (비용 자료 분석).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch21.py && python3 gen/fig_ch21.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import pandas as pd
from svgplot import Plot, figure, panel_title, forest

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch21_nums.json"), encoding="utf-8"))
d = pd.read_csv(os.path.join(HERE, "_ch21_cost.csv"))


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def c0(v):
    return f"{v:,.0f}"


# ------------------------------------------------------------------ 그림 21-1: 비용의 분포 (B군)
b = d[d.A == 0]
gB = N["grp"]["B"]
assert abs(b.cost.mean() - gB["mean"]) < 1e-6 and len(b) == gB["n"]
panels = []
# (가) 1년 총 의료비
edges = np.arange(0, 8250, 250)
y = b.cost.values
cnt, _ = np.histogram(np.clip(y, 0, edges[-1] - 1e-6), bins=edges)
over = int((y >= edges[-1] - 250).sum())
YM = int(np.ceil(cnt.max() / 100) * 100) + 100
p = Plot((0, 8000), (0, YM), w=420, h=310, ml=46, mr=14, mt=40, mb=50,
         xlabel="치료 시작 후 1년 의료비 (만원)", ylabel="환자 수",
         xticks=[0, 2000, 4000, 6000, 8000],
         xticklabels=[(0, "0"), (2000, "2,000"), (4000, "4,000"), (6000, "6,000"), (8000, "8,000+")],
         yticks=list(range(0, YM + 1, 100)))
p.hist(edges, cnt, s=4, gap=1.2)
p.vline(gB["med"], cls="ref", dash=True, w=1.4, y1=YM * 0.95)
p.vline(gB["mean"], cls="ref strongref", dash=False, w=1.8, y1=YM * 0.95)
p.text(gB["med"], YM * 0.95, f"중앙값 {c0(gB['med'])}", anchor="end", dx=-7, dy=8, cls="lbl small mute")
p.text(gB["mean"], YM * 0.95, f"평균 {c0(gB['mean'])}", anchor="start", dx=6, dy=8, cls="lbl small strong")
p.text(4300, YM * 0.52, f"5,000만원 초과 {gB['over5000']}명", anchor="start", cls="lbl small")
p.text(4300, YM * 0.52, f"최고 {c0(gB['max'])}만원", anchor="start", dy=16, cls="lbl small")
p.text(4300, YM * 0.52, f"상위 5% 환자가 총액의 {gB['top5'] * 100:.0f}%", anchor="start", dy=32, cls="lbl small")
panel_title(p, f"가. 총 의료비 (표준요법 B군 {c0(gB['n'])}명)")
panels.append(p.svg("표준요법 B군의 1년 총 의료비 분포"))
# (나) 입원비
cI = N["comp"]["inp"]
iB = N["inpB"]
yi = b.inp.values
pos = yi[yi > 0]
edges2 = np.arange(0, 4250, 250)
cnt2, _ = np.histogram(np.clip(pos, 0, edges2[-1] - 1e-6), bins=edges2)
YM2 = 300
p = Plot((-330, 4000), (0, YM2), w=420, h=310, ml=46, mr=14, mt=40, mb=50,
         xlabel="치료 시작 후 1년 입원 진료비 (만원)", ylabel="환자 수",
         xticks=[0, 1000, 2000, 3000, 4000],
         xticklabels=[(0, "0"), (1000, "1,000"), (2000, "2,000"), (3000, "3,000"), (4000, "4,000+")],
         yticks=[0, 100, 200, 300])
p.bars([-165], [YM2], 200, s=2, rounded=False)
p.text(-40, YM2 * 0.93, f"← 0원 {c0(cI['nzB'])}명({cI['zeroB'] * 100:.0f}%), 위를 자름", anchor="start", dx=4, cls="lbl small strong")
p.hist(edges2, cnt2, s=4, gap=1.2)
p.vline(cI["mB"], cls="ref strongref", dash=False, w=1.8, y1=YM2 * 0.70)
p.text(cI["mB"], YM2 * 0.70, f"평균 {c0(cI['mB'])}", anchor="start", dx=5, dy=4, cls="lbl small strong")
p.text(cI["mB"], YM2 * 0.70, f"중앙값 {c0(cI['medB'])}", anchor="start", dx=5, dy=20, cls="lbl small mute")
p.text(2000, YM2 * 0.40, f"입원한 {c0(iB['npos'])}명의 평균 {c0(iB['pos_mean'])}", anchor="start", cls="lbl small")
p.text(2000, YM2 * 0.40, f"상위 10% 환자가 입원비의 {iB['top10'] * 100:.0f}%", anchor="start", dy=16, cls="lbl small")
panel_title(p, "나. 그중 입원 진료비")
panels.append(p.svg("표준요법 B군의 1년 입원 진료비 분포"))
save("ch21_dist", figure(panels,
     f"그림 21-1. 표준요법 B군 {c0(gB['n'])}명의 1년 의료비 분포(250만원 간격, 가상의 자료). "
     f"가: 대부분은 1,000만–3,000만원에 몰려 있고 소수의 고액 환자가 오른쪽으로 긴 꼬리를 만들어, 평균(굵은 실선)이 중앙값(점선)보다 큽니다. "
     f"나: 입원 진료비만 보면 3명 중 1명이 0원이고 꼬리는 더 깁니다. 0원 막대는 위를 잘라 그렸습니다.", cols=2))

# ------------------------------------------------------------------ 그림 21-2: 부트스트랩 분포
dist = np.load(os.path.join(HERE, "_ch21_boot.npy"))
bt = N["boot"]
obs = N["comp"]["cost"]["diff"]
edges3 = np.arange(200, 601, 10)
cnt3, _ = np.histogram(dist, bins=edges3)
YM3 = int(np.ceil(cnt3.max() / 100) * 100) + 100
p = Plot((200, 600), (0, YM3), w=620, h=330, ml=58, mr=20, mt=24, mb=52,
         xlabel="복원추출한 표본의 평균 비용 차이, A군 − B군 (만원)", ylabel="횟수 (9,999번 중)",
         xticks=[200, 250, 300, 350, 400, 450, 500, 550, 600], yticks=list(range(0, YM3 + 1, 200)))
inside = (edges3[:-1] >= bt["pct"][0] - 5) & (edges3[1:] <= bt["pct"][1] + 5)
p.hist(edges3, np.where(inside, cnt3, 0), s=1, gap=1.0)
p.hist(edges3, np.where(inside, 0, cnt3), s=4, gap=1.0)
top = YM3 * 0.93
p.vline(obs, cls="ref strongref", dash=False, w=1.8, y1=top)
p.text(obs, top, f"관찰된 차이 {obs:.0f}", anchor="middle", dy=-6, cls="lbl small strong")
for v, anc, dx in ((bt["pct"][0], "end", -6), (bt["pct"][1], "start", 6)):
    p.vline(v, cls="ref", dash=True, w=1.3, y1=YM3 * 0.62)
    p.text(v, YM3 * 0.62, f"{v:.0f}", anchor=anc, dx=dx, dy=4, cls="lbl small strong")
p.text(bt["pct"][0], YM3 * 0.62, "2.5번째 백분위수", anchor="end", dx=-6, dy=20, cls="lbl small mute")
p.text(bt["pct"][1], YM3 * 0.62, "97.5번째 백분위수", anchor="start", dx=6, dy=20, cls="lbl small mute")
save("ch21_boot", figure(p.svg("평균 비용 차이의 부트스트랩 분포"),
     f"그림 21-2. 평균 비용 차이의 부트스트랩 분포. 두 군에서 환자를 복원추출해 평균 차이를 다시 계산하는 일을 9,999번 반복한 결과입니다. "
     f"가운데 95%(파란 막대)의 양 끝 {bt['pct'][0]:.0f}만원과 {bt['pct'][1]:.0f}만원이 백분위수 신뢰구간입니다. "
     f"환자 한 명 한 명의 비용은 치우쳐 있어도 평균 차이의 분포는 좌우대칭에 가깝습니다."))

# ------------------------------------------------------------------ 그림 21-3: 방법별 추정값과 참값
R = N["reg"]
truth = N["true"]["s_ate"]
rows = [
    dict(label="보정 전 평균 차이", est=R["crude"]["diff"], lo=R["crude"]["ci"][0], hi=R["crude"]["ci"][1], s=4, nd=0),
    dict(label="선형회귀 (원래 비용)", est=R["ols"]["diff"], lo=R["ols"]["ci"][0], hi=R["ols"]["ci"][1], s=1, nd=0),
    dict(label="로그 변환 회귀, 그대로 되돌림", est=R["logols"]["naive"], lo=R["logols"]["naive_ci"][0], hi=R["logols"]["naive_ci"][1], s=2, nd=0),
    dict(label="로그 변환 회귀 + smearing", est=R["logols"]["smear_diff"], lo=R["logols"]["smear_ci"][0], hi=R["logols"]["smear_ci"][1], s=2, nd=0),
    dict(label="감마 GLM (로그 연결)", est=R["glm"]["diff"], lo=R["glm"]["ci"][0], hi=R["glm"]["ci"][1], s=1, nd=0, bold=True),
]
svg = forest(rows, xlim=(250, 780), ref=truth, log=False, w=640, label_w=214, est_w=132,
             xlabel=f"1인당 1년 비용 차이, A − B (만원). 세로 실선은 참값 {truth:.0f}", xticks=[300, 400, 500, 600, 700],
             header=("방법", "차이 (95% CI)"))
save("ch21_methods", figure(svg,
     f"그림 21-3. 방법별로 추정한 1인당 1년 비용 차이와 95% 신뢰구간. 세로 실선은 모의 자료의 참값 {truth:.0f}만원입니다."))
print("figures saved")
