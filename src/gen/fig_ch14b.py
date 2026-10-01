"""Figures for chapter 14 sections 라–사 (part b).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch14b.py && python3 gen/fig_ch14b.py"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import pandas as pd
from svgplot import Plot, figure, panel_title, forest, fmt

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch14b_nums.json")))
M = lambda v, nd=1: f"{v:.{nd}f}".replace("-", "−")


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================ 라. 1-KM vs CIF
d = pd.read_csv(os.path.join(HERE, "_ch14b_cr.csv"))
grid = np.linspace(0, 5, 301)


def aj_curves(s):
    t = s.t.values; e = s.ev.values
    o = np.argsort(t, kind="stable"); t, e = t[o], e[o]
    n = len(t); S = 1.0; c1 = c2 = 0.0; Skm = 1.0
    T, C1, C2, K = [0.0], [0.0], [0.0], [0.0]
    for i in range(len(t)):
        at = n - i
        if e[i] == 1:
            c1 += S / at; Skm *= (1 - 1 / at)
        elif e[i] == 2:
            c2 += S / at
        if e[i] in (1, 2):
            S *= (1 - 1 / at)
        T.append(t[i]); C1.append(c1); C2.append(c2); K.append(1 - Skm)
    T = np.array(T)
    idx = np.searchsorted(T, grid, side="right") - 1
    return np.array(C1)[idx], np.array(C2)[idx], np.array(K)[idx]


cA = aj_curves(d[d.grp == "A"])
cB = aj_curves(d[d.grp == "B"])
pc = lambda v: f"{v * 100:.0f}%"
p1 = Plot((0, 5), (0, 0.6), w=420, h=330, ml=58, mr=86, xlabel="약물 시작 후 연수", ylabel="누적 비율",
          yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ytickfmt=pc, xticks=[0, 1, 2, 3, 4, 5])
p1.line(grid, cB[2], s=2, dash=True, w=2.2)
p1.line(grid, cB[0], s=1, w=2.4)
p1.text(5, cB[2][-1], f"1 − KM {cB[2][-1] * 100:.1f}%", dx=6, dy=4, cls="lbl small")
p1.text(5, cB[0][-1], f"CIF {cB[0][-1] * 100:.1f}%", dx=6, dy=4, cls="lbl small")
p1.legend([("1 − Kaplan-Meier", 2, "dash"), ("누적발생함수", 1, "line")])
panel_title(p1, "(가) 약물 B군의 투석 누적 비율")

p2 = Plot((0, 5), (0, 0.6), w=420, h=330, ml=58, mr=86, xlabel="약물 시작 후 연수", ylabel="누적발생함수",
          yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ytickfmt=pc, xticks=[0, 1, 2, 3, 4, 5])
p2.line(grid, cB[1], s=2, dash=True, w=2.0)
p2.line(grid, cA[1], s=1, dash=True, w=2.0)
p2.line(grid, cB[0], s=2, w=2.4)
p2.line(grid, cA[0], s=1, w=2.4)
p2.text(5, cB[1][-1], f"B 사망 {cB[1][-1] * 100:.1f}%", dx=6, dy=4, cls="lbl small")
p2.text(5, cA[1][-1], f"A 사망 {cA[1][-1] * 100:.1f}%", dx=6, dy=4, cls="lbl small")
p2.text(5, cA[0][-1], f"A 투석 {cA[0][-1] * 100:.1f}%", dx=6, dy=-2, cls="lbl small")
p2.text(5, cB[0][-1], f"B 투석 {cB[0][-1] * 100:.1f}%", dx=6, dy=10, cls="lbl small")
p2.legend([("약물 A", 1, "line"), ("약물 B", 2, "line")])
p2.text(5, 0.03, "실선: 투석, 점선: 투석 전 사망", anchor="end", cls="lbl small")
panel_title(p2, "(나) 두 군의 투석과 사망")
save("ch14_d_cr", figure(
    [p1.svg("약물 B군에서 1-KM과 누적발생함수 비교"), p2.svg("두 군의 투석과 사망 누적발생함수")],
    "그림 14-4. 고령 CKD 4기 환자 가상 코호트의 투석 시작. (가) 같은 자료에서 사망을 중도절단으로 처리한 1 − Kaplan-Meier(점선)는 "
    f"5년 투석 비율을 {cB[2][-1] * 100:.1f}%로, 사망을 경쟁 사건으로 다룬 누적발생함수(실선)는 {cB[0][-1] * 100:.1f}%로 추정합니다. "
    "(나) 약물 A군은 투석 전 사망이 적어(점선) 투석까지 가는 환자가 더 많습니다(실선). 투석 속도 자체는 두 군이 비슷합니다(원인별 위험비 1.02).",
    cols=2))

# ============================================================ 마. immortal-time timeline
pts = [  # label, start drug (months) or None, end (months), died
    ("환자 1", 2, 12, False),
    ("환자 2", 5, 9, True),
    ("환자 3", None, 1.5, True),
    ("환자 4", 8, 12, False),
    ("환자 5", None, 12, False),
    ("환자 6", None, 4, True),
]


def death_mark(p, x, y):
    X, Y = p.sx(x), p.sy(y)
    for a, b in ((-5, -5), (-5, 5)):
        p.top.append(f'<line x1="{X + a:.1f}" y1="{Y + b:.1f}" x2="{X - a:.1f}" y2="{Y - b:.1f}" class="strongref" stroke-width="2.2"/>')


def start_mark(p, x, y):
    X, Y = p.sx(x), p.sy(y)
    p.top.append(f'<polygon points="{X - 5:.1f},{Y - 16:.1f} {X + 5:.1f},{Y - 16:.1f} {X:.1f},{Y - 8:.1f}" class="f1"/>')


def draw(p, correct):
    for i, (lab, s0, end, died) in enumerate(pts):
        y = 6 - i
        p.text_px(8, p.sy(y) + 4.5, lab, cls="lbl small")
        if s0 is None:
            p.seg(0, y, end, y, cls="ln s4", w=10)
            grp = "비사용군"
        else:
            p.seg(0, y, s0, y, cls="ln s4" if correct else "ln s2", w=10)
            p.seg(s0, y, end, y, cls="ln s1", w=10)
            start_mark(p, s0, y)
            grp = "사용군"
        if died:
            death_mark(p, end, y)
        if not correct:
            p.text_px(p.w - p.mr + 14, p.sy(y) + 4.5, grp, cls="lbl small")
    B = p.h - p.mb
    p.text_px((p.ml + p.w - p.mr) / 2, B + 34, "퇴원 후 개월", anchor="middle", cls="axlab")
    # legend row(s) under the axis
    if correct:
        items = [("비노출 시간", 4), ("노출 시간(약물 X 복용 기간)", 1)]
    else:
        items = [("비사용군으로 센 시간", 4), ("사용군으로 센 시간", 1), ("불멸시간", 2)]
    X = p.ml
    for lab, s in items:
        p.top.append(f'<rect x="{X:.1f}" y="{B + 50:.1f}" width="12" height="12" rx="2" class="f{s}"/>')
        p.text_px(X + 17, B + 60, lab, cls="lbl small")
        X += 17 + 13.2 * len(lab) + 18
    # symbols
    Y = B + 76
    p.top.append(f'<polygon points="{p.ml + 1:.1f},{Y - 5:.1f} {p.ml + 11:.1f},{Y - 5:.1f} {p.ml + 6:.1f},{Y + 3:.1f}" class="f1"/>')
    p.text_px(p.ml + 17, Y + 3, "약물 X 첫 처방", cls="lbl small")
    xx = p.ml + 140
    for a, b in ((-5, -5), (-5, 5)):
        p.top.append(f'<line x1="{xx + 6 + a:.1f}" y1="{Y - 1 + b:.1f}" x2="{xx + 6 - a:.1f}" y2="{Y - 1 - b:.1f}" class="strongref" stroke-width="2.2"/>')
    p.text_px(xx + 17, Y + 3, "사망", cls="lbl small")


pa = Plot((0, 12), (0.4, 6.9), w=600, h=350, ml=64, mr=90, mt=30, mb=98,
          xlabel="", xticks=[0, 2, 4, 6, 8, 10, 12], yticks=[], ygrid=False, show_yaxis=False, xgrid=True)
draw(pa, False)
panel_title(pa, "(가) 잘못된 분류: 추적 중 한 번이라도 처방받으면 처음부터 사용군")
pb = Plot((0, 12), (0.4, 6.9), w=600, h=350, ml=64, mr=90, mt=30, mb=98,
          xlabel="", xticks=[0, 2, 4, 6, 8, 10, 12], yticks=[], ygrid=False, show_yaxis=False, xgrid=True)
draw(pb, True)
panel_title(pb, "(나) 시간의존 분류: 처방 전은 비노출, 처방 후는 노출")
save("ch14_e_immortal", figure(
    [pa.svg("불멸시간이 사용군에 잘못 들어가는 분류"), pb.svg("처방 시점에서 노출이 바뀌는 시간의존 분류")],
    "그림 14-5. 퇴원 후 1년 동안 추적한 환자 6명. (가) 환자 1, 2, 4는 처방을 받기 전까지 사망할 수 없었는데(주황), "
    "이 기간이 사용군의 추적 시간으로 들어가 사용군의 사망률을 낮춥니다. 일찍 사망한 환자 3, 6은 약을 시작할 기회도 없이 비사용군이 됩니다. "
    "(나) 같은 환자들을 처방 시점에서 노출이 바뀌도록 나누면 처방 전 시간은 비노출 쪽으로 갑니다.", cols=1))

# ============================================================ 바. forest plot
ma = N["ma"]
z = 1.959963984540054
rows = [{"label": "Study", "header": True}]
for nm, y, se, wr in zip(ma["names"], ma["y"], ma["se"], ma["wre"]):
    rows.append({"label": f"{nm}  ({wr * 100:.1f}%)", "est": y, "lo": y - z * se, "hi": y + z * se,
                 "size": 3 + 9 * wr ** 0.5 * 1.1, "indent": True})
fe = ma["fe"]; re_ = ma["re"]
rows.append({"label": "Fixed effect", "est": fe[0], "lo": fe[2], "hi": fe[3], "diamond": True, "bold": True, "s": 4})
rows.append({"label": "Random effects", "est": re_[0], "lo": re_[2], "hi": re_[3], "diamond": True, "bold": True, "s": 1})
rows.append({"label": "  95% prediction interval", "est": re_[0], "lo": ma["pi"][0], "hi": ma["pi"][1], "s": 4, "size": 0.01})
svg = forest(rows, xlim=(-1.8, 0.6), ref=0.0, log=False, w=680, row_h=28, label_w=220, est_w=196,
             xlabel="HbA1c 변화의 평균 차이, %p (중재 − 통상관리)", left_note="← 중재가 유리",
             right_note="통상관리가 유리 →", xticks=[-1.5, -1.0, -0.5, 0, 0.5], header=None)
# nicer numbers: unicode minus and "to"
svg = re.sub(r'>(-?\d\.\d\d) \((-?\d\.\d\d)–(-?\d\.\d\d)\)<',
             lambda m: f'>{M(float(m.group(1)), 2)} ({M(float(m.group(2)), 2)} to {M(float(m.group(3)), 2)})<', svg)
svg = svg.replace('class="tick">-', 'class="tick">−')
note = (f"Heterogeneity: Q = {ma['Q']:.2f} (df = 5), P = {ma['pQ']:.4f}; I² = {ma['I2'] * 100:.0f}%; τ² = {ma['tau2']:.3f}")
hgt = int(re.search(r'viewBox="0 0 \d+ (\d+)"', svg).group(1))
svg = svg.replace(f'viewBox="0 0 680 {hgt}"', f'viewBox="0 0 680 {hgt + 40}"', 1)
# move the x-axis title below the direction notes, then add the heterogeneity line
svg = svg.replace(f'y="{hgt - 10}" text-anchor="middle" class="axlab"', f'y="{hgt + 10}" text-anchor="middle" class="axlab"', 1)
svg = svg.replace('</svg>', f'<text x="4" y="{hgt + 34}" class="lbl small">{note}</text></svg>')
save("ch14_f_forest", figure(
    [svg],
    "그림 14-6. 약사 주도 중재가 6개월 HbA1c 변화에 미친 효과를 합친 가상의 메타분석. 사각형 크기는 무작위효과 가중치(괄호 안 %)에 비례합니다. "
    "마름모의 가운데가 합친 추정값, 양 끝이 95% CI입니다. 맨 아래 회색 선은 새 연구에서 기대되는 효과의 범위(95% 예측구간)로, 0을 넘어섭니다.",
    cols=1))

# ============================================================ 바. funnel plot
fu = N["funnel"]
yf = np.array(fu["y"]); sef = np.array(fu["se"]); pub = np.array(fu["pub"], bool)
wf = 1 / sef ** 2
fe_pub = fu["fe_pub"]
smax = 0.36
p = Plot((-1.2, 0.7), (-smax, 0), w=600, h=340, xlabel="평균 차이 (중재 − 대조)", ylabel="표준오차 (위로 갈수록 큰 연구)",
         yticks=[-0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0], ytickfmt=lambda v: f"{abs(v):.2f}",
         xticks=[-1.0, -0.5, 0, 0.5], xtickfmt=lambda v: M(v, 1) if v else "0")
p.line([fe_pub - z * smax, fe_pub, fe_pub + z * smax], [-smax, 0, -smax], s=4, dash=True, w=1.4)
p.vline(fe_pub, dash=False, cls="strongref", w=1.2)
p.vline(0, dash=True)
p.points(yf[pub], -sef[pub], s=1, r=5)
p.points(yf[~pub], -sef[~pub], s=2, r=5, hollow=True)
p.legend([("출판된 연구", 1, "dot"), ("출판되지 않은 연구", 2, None)], X=p.sx(0.1), Y=p.mt + 14)
# hollow legend marker
p.top.append(f'<circle cx="{p.sx(0.1) + 11:.1f}" cy="{p.mt + 14 + 18:.1f}" r="4.5" class="pth s2"/>')
p.text(fe_pub, -smax, f"합친 추정값 {M(fe_pub, 2)}", anchor="end", dx=-6, dy=-8, cls="lbl small")
save("ch14_f_funnel", figure(
    [p.svg("출판 비뚤림이 있는 깔때기 그림")],
    "그림 14-7. 깔때기 그림(funnel plot). 위쪽은 큰 연구, 아래쪽은 작은 연구입니다. 비뚤림이 없으면 점들이 가운데 선을 중심으로 좌우 대칭인 깔때기 모양으로 퍼집니다. "
    "이 가상의 예에서는 유의한 효과를 보이지 못한 작은 연구(빈 원)가 출판되지 않아, 보이는 점들이 아래쪽 오른편이 비어 있는 비대칭 모양이 됩니다.",
    cols=1))

# ============================================================ 사. ITS
its = N["its"]
b = its["b"]; yts = np.array(its["y"])
t = np.arange(1, 49)
x = t - 24
p = Plot((-24, 24.5), (36, 56), w=600, h=340, xlabel="안전성 서한 발표 전후 개월 (0 = 발표 직전 달)",
         ylabel="처방률 (1,000명당, 월)", xticks=[-24, -18, -12, -6, 0, 6, 12, 18, 24],
         xtickfmt=lambda v: ("+" if v > 0 else "") + fmt(v).replace("-", "−"), yticks=[36, 40, 44, 48, 52, 56])
p.vline(0.5, dash=True)
p.points(x, yts, s=4, r=3.2)
pre_t = np.array([1, 24]); p.line(pre_t - 24, b[0] + b[1] * pre_t, s=1, w=2.6)
post_t = np.array([25, 48]); p.line(post_t - 24, b[0] + b[1] * post_t + b[2] + b[3] * (post_t - 24), s=1, w=2.6)
p.line(post_t - 24, b[0] + b[1] * post_t, s=2, dash=True, w=2.2)
p.text(0.5, 55.2, "안전성 서한 발표", anchor="start", dx=6, cls="lbl small")
# level change bracket at t=24.5 (x=0.5): from projected to post intercept
yp = b[0] + b[1] * 24.5
yq = b[0] + b[1] * 24.5 + b[2] + b[3] * 0.5
p.seg(1.6, yp, 1.6, yq, cls="strongref", w=1.6)
p.text(2.2, (yp + yq) / 2, f"수준 변화 {M(b[2], 1)}", dy=4, cls="lbl small")
# 12-month effect
pred, cf = N["its_12"][0], N["its_12"][1]
p.seg(12, cf, 12, pred, cls="strongref", w=1.6)
p.text(12.6, (pred + cf) / 2 - 1.2, f"12개월 뒤 {M(pred - cf, 1)}", dy=4, cls="lbl small")
p.legend([("월별 관측값", 4, "dot"), ("분절회귀 적합선", 1, "line"), ("발표 전 추세의 연장(반사실)", 2, "dash")],
         X=p.ml + 12, Y=p.h - p.mb - 58)
save("ch14_g_its", figure(
    [p.svg("중단시계열 분절회귀")],
    "그림 14-8. 안전성 서한 발표 전후 48개월 동안 65세 이상 환자 1,000명당 약물 X 월별 처방률(가상 자료). "
    "발표 전 추세를 연장한 선(주황 점선)이 '발표가 없었다면'의 추정입니다. 발표 직후 수준이 뚝 떨어졌고(수준 변화), 그 뒤 감소 기울기가 더해졌습니다(기울기 변화).",
    cols=1))

# ============================================================ 사. DID
dd = N["did"]
q_pre = np.array([-4, -3, -2, -1]); q_post = np.array([1, 2, 3, 4])
p = Plot((-4.6, 4.9), (10, 20), w=600, h=340, xlabel="정책 시행 전후 분기", ylabel="장기 처방 비율 (%)",
         xticks=[-4, -3, -2, -1, 1, 2, 3, 4], xtickfmt=lambda v: ("+" if v > 0 else "") + fmt(v).replace("-", "−"),
         yticks=[10, 12, 14, 16, 18, 20], mr=142)
p.vline(0, dash=True)
p.text(0, 19.6, "시범사업 시작", anchor="middle", dy=0, cls="lbl small")
xa = np.concatenate([q_pre, q_post])
p.line(xa, dd["tp"] + dd["tpost"], s=1, w=2.4)
p.points(xa, dd["tp"] + dd["tpost"], s=1, r=4)
p.line(xa, dd["cp"] + dd["cpost"], s=2, w=2.4)
p.points(xa, dd["cp"] + dd["cpost"], s=2, r=4)
p.line(np.concatenate([[-1], q_post]), [dd["tp"][-1]] + dd["tcf"], s=1, dash=True, w=2.0)
p.text(4, dd["tcf"][-1], "시범 지역의 반사실", anchor="start", dx=22, dy=4, cls="lbl small")
p.text(4, dd["tpost"][-1], "시범 지역", anchor="start", dx=22, dy=4, cls="lbl small")
p.text(4, dd["cpost"][-1], "비교 지역", anchor="start", dx=22, dy=4, cls="lbl small")
p.seg(4.25, dd["tcf"][-1], 4.25, dd["tpost"][-1], cls="strongref", w=1.6)
p.text(4, (dd["tcf"][-1] + dd["tpost"][-1]) / 2 - 0.9, f"차이의 차이 {M(dd['did'], 1)}%p", anchor="start", dx=22, dy=4, cls="lbl small strong")
save("ch14_g_did", figure(
    [p.svg("이중차분법")],
    "그림 14-9. 시범사업 지역(파랑)과 비교 지역(주황)의 분기별 벤조디아제핀 장기 처방 비율(가상 자료). 시행 전 두 선의 기울기가 같다(평행 추세)는 것을 근거로, "
    "비교 지역의 변화를 시범 지역에 옮겨 그린 파란 점선이 '사업이 없었다면'의 추정입니다. 실제 값과 이 점선의 간격이 이중차분 추정값입니다.",
    cols=1))
print("figures written")
