"""Figures for chapter 14, sections 나 (ROC) and 다 (propensity scores).

Run:  source /home/claude/pylibs/env.sh && python3 gen/fig_ch14a.py
Writes figs/ch14_b_roc.html, figs/ch14_c_psdist.html, figs/ch14_c_love.html.
Caption numbers are provisional (sections 가–다 come first in the chapter).
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from svgplot import Plot, figure, panel_title
from nums_ch14a import compute, SCORES, CASES, NONCASES

OUT = os.path.join(HERE, "..", "figs")
os.makedirs(OUT, exist_ok=True)
R = compute()


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


# ======================================================================
# 14-1  score distribution by group + ROC curve (section 나)
ro = R["roc"]
pc = 100 * CASES / CASES.sum()
pn = 100 * NONCASES / NONCASES.sum()
pa = Plot((-0.6, 8.6), (0, 25), w=420, h=360, ml=50, mr=14, mt=30, mb=52,
          xlabel="설문 점수", ylabel="각 군 안에서의 비율(%)", xticks=list(range(9)), yticks=[0, 5, 10, 15, 20, 25])
pa.bars(SCORES - 0.19, pn, 0.36, s=1)
pa.bars(SCORES + 0.19, pc, 0.36, s=2)
pa.vline(4.5, dash=True)
pa.text(4.62, 23.6, "≥5점이면 양성", cls="lbl small")
pa.legend([("순응 (n = 140)", 1, "box"), ("비순응 (n = 60)", 2, "box")], X=pa.sx(-0.5), Y=pa.sy(24.2))
panel_title(pa, "가. 두 군의 점수 분포")

rows = ro["rows"]
fpr = [1 - r["spec"] for r in rows][::-1]
tpr = [r["sens"] for r in rows][::-1]
pb = Plot((0, 1), (0, 1), w=420, h=360, ml=56, mr=14, mt=30, mb=52, xlabel="1 − 특이도 (위양성 비율)", ylabel="민감도",
          xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
          xtickfmt=lambda v: f"{v:.1f}", ytickfmt=lambda v: f"{v:.1f}", xgrid=True)
pb.fill_between(fpr, [0] * len(fpr), tpr, s=1)
pb.seg(0, 0, 1, 1, cls="ref", dash=True)
pb.line(fpr, tpr, s=1, w=2.2)
pb.points(fpr[1:-1], tpr[1:-1], s=1, r=3.5)
for k, dx, dy, anc in ((3, -8, -6, "end"), (4, -8, -8, "end"), (5, -8, -8, "end"), (6, -8, -8, "end"), (7, -8, -6, "end")):
    r = rows[k]
    pb.text(1 - r["spec"], r["sens"], f"≥{k}", anchor=anc, dx=dx, dy=dy, cls="lbl small")
r5 = rows[5]
pb.points([1 - r5["spec"]], [r5["sens"]], s=2, r=6)
pb.text(1 - r5["spec"], r5["sens"], f"민감도 {100*r5['sens']:.1f}%, 특이도 {100*r5['spec']:.1f}%", dx=12, dy=16, cls="lbl small")
pb.text(0.52, 0.30, f"AUC = {ro['auc']:.3f}", cls="lbl strong")
pb.text(0.52, 0.30, "(색칠한 넓이)", dy=17, cls="lbl mute small")
pb.text(0.98, 0.50, "대각선: AUC 0.5", anchor="end", cls="lbl mute small")
panel_title(pb, "나. ROC 곡선")
save("ch14_b_roc", figure(
    [pa.svg("순응군과 비순응군의 설문 점수 분포"), pb.svg("설문 점수의 ROC 곡선")],
    f"그림 14-1. 가상의 복약 비순응 선별 설문(0–8점). (가) 비순응 환자(주황)의 점수가 대체로 높지만 두 분포가 많이 겹칩니다. "
    f"(나) 기준점을 ≥8점에서 ≥1점으로 낮출 때마다 민감도와 위양성 비율이 함께 올라가며 곡선 위의 점이 오른쪽 위로 이동합니다. "
    f"주황 점이 ≥5점 기준이고, 곡선 아래 넓이가 AUC {ro['auc']:.3f}입니다. 점수가 정수라 같은 점수(동점)가 많아 점 사이를 직선으로 이었습니다.",
    cols=2))

# ======================================================================
# 14-2  propensity score distributions (mirror histogram, section 다)
s = R["sim"]
ps, A = s["ps"], s["A"]
edges = np.linspace(0, 1, 26)
hA, _ = np.histogram(ps[A == 1], edges)
hB, _ = np.histogram(ps[A == 0], edges)
fA = 100 * hA / hA.sum()
fB = 100 * hB / hB.sum()
mids = (edges[:-1] + edges[1:]) / 2
ymax = 20
p = Plot((0, 1), (-ymax, ymax), w=600, h=340, ml=62, mr=24, mt=24, mb=52,
         xlabel="성향점수 (약물 A를 받을 추정 확률)", ylabel="각 군 안에서의 비율(%)",
         xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], xtickfmt=lambda v: f"{v:.1f}",
         yticks=[-20, -10, 0, 10, 20], ytickfmt=lambda v: f"{abs(v):.0f}")
p.bars(mids, fA, 0.036, s=1, rounded=False)
p.bars(mids, -fB, 0.036, s=2, rounded=False)
p.hline(0, dash=False, cls="axis")
p.text(0.99, 16.5, f"약물 A (n = {s['nA']:,})", anchor="end", cls="lbl strong")
p.text(0.99, -17.5, f"약물 B (n = {s['nB']:,})", anchor="end", cls="lbl strong")
mA = ps[A == 1].mean(); mB = ps[A == 0].mean()
p.vline(mA, dash=True, y0=0, y1=ymax)
p.vline(mB, dash=True, y0=-ymax, y1=0)
p.text(mA, 18.5, f"평균 {mA:.2f}", dx=6, cls="lbl small")
p.text(mB, -19.0, f"평균 {mB:.2f}", dx=6, cls="lbl small")
save("ch14_c_psdist", figure(
    p.svg("약물 A군과 B군의 성향점수 분포"),
    f"그림 14-2. 가상의 청구자료 코호트(1만 명)에서 추정한 성향점수의 분포. 위(파랑)는 약물 A 사용자, 아래(주황)는 약물 B 사용자입니다. "
    f"A군의 점수가 오른쪽으로 치우쳐 있어(평균 {mA:.2f} 대 {mB:.2f}) 두 군의 기저특성이 다르다는 것을 보여 주지만, "
    f"거의 모든 구간에 두 군의 환자가 함께 있어 비교할 상대를 찾을 수 있습니다(겹침, overlap)."))

# ======================================================================
# 14-3  Love plot (section 다)
tab = s["tab"]
meas = [r for r in tab if r["key"] != "frail"]
frail = [r for r in tab if r["key"] == "frail"][0]
order = sorted(meas, key=lambda r: abs(r["smd_pre"]))
rows_ = order + [None, frail]  # gap row before frailty
nrow = len(rows_)
p = Plot((0, 0.45), (0.3, nrow + 0.7), w=600, h=40 + 30 * nrow + 60, ml=176, mr=24, mt=24, mb=52,
         xlabel="표준화 평균차의 절댓값 (|SMD|)", xticks=[0, 0.1, 0.2, 0.3, 0.4], xtickfmt=lambda v: f"{v:.1f}",
         yticks=[], ygrid=False, xgrid=True, show_yaxis=False)
p.vline(0.1, dash=True)
p.text(0.1, nrow + 0.55, "0.1 기준", dx=5, cls="lbl mute small")
for i, r in enumerate(rows_):
    if r is None:
        continue
    y = nrow - i
    b, a = abs(r["smd_pre"]), abs(r["smd_m"])
    is_frail = r["key"] == "frail"
    p.seg(a, y, b, y, cls="ref", dash=is_frail)
    p.points([b], [y], s=2 if not is_frail else 4, r=5, hollow=True)
    p.points([a], [y], s=1 if not is_frail else 4, r=5)
    p.text_px(p.ml - 10, p.sy(y) + 4.5, r["ko"], anchor="end", cls="lbl" if not is_frail else "lbl mute")
yf = nrow - (len(rows_) - 1)
p.text(abs(frail["smd_m"]), yf, "매칭해도 남는 불균형", dx=10, dy=-10, cls="lbl mute small")
p.legend([("매칭 전", 2, "dot")], X=p.sx(0.13), Y=p.sy(2.0))
p.legend([("매칭 후", 1, "dot")], X=p.sx(0.26), Y=p.sy(2.0))
# hollow legend marker for "before": overwrite legend dot with hollow circle look
p.top = [t.replace('class="f2"', 'class="pth s2"') if 'r="4.5" class="f2"' in t else t for t in p.top]
save("ch14_c_love", figure(
    p.svg("매칭 전후 공변량별 표준화 평균차 (Love plot)"),
    "그림 14-3. 성향점수 매칭 전후의 공변량 균형(Love plot). 빈 원은 매칭 전, 채운 원은 매칭 후의 |SMD|입니다. "
    "성향점수 모형에 넣은 7개 변수는 모두 0.1 기준선 왼쪽으로 옮겨 왔습니다. 맨 아래 회색 줄의 허약은 청구자료에 없어 모형에 넣지 못한 변수로, "
    "가상 자료라서 값을 알 수 있을 뿐 실제 연구에서는 이 줄을 그릴 수 없습니다. 매칭 후에도 0.34로 크게 남아 있습니다."))

print("written:", [f for f in sorted(os.listdir(OUT)) if f.startswith("ch14_b") or f.startswith("ch14_c")])
