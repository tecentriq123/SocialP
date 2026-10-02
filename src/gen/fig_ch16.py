"""Figures for chapter 16 (시간과 관련된 편향).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch16.py && python3 gen/fig_ch16.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
import pandas as pd
from svgplot import Plot, figure, panel_title

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch16_nums.json")))
dd = pd.read_csv(os.path.join(HERE, "_ch16_cohort.csv"))


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


def death_mark(p, x, y):
    X, Y = p.sx(x), p.sy(y)
    for a, b in ((-5, -5), (-5, 5)):
        p.top.append(f'<line x1="{X + a:.1f}" y1="{Y + b:.1f}" x2="{X - a:.1f}" y2="{Y - b:.1f}" class="strongref" stroke-width="2.2"/>')


def start_mark(p, x, y):
    X, Y = p.sx(x), p.sy(y)
    p.top.append(f'<polygon points="{X - 5:.1f},{Y - 16:.1f} {X + 5:.1f},{Y - 16:.1f} {X:.1f},{Y - 8:.1f}" class="f1"/>')


def legend_row(p, items, Y, X=None):
    """items: (label, series) colour boxes in one row under the axis"""
    X = p.ml if X is None else X
    for lab, s in items:
        p.top.append(f'<rect x="{X:.1f}" y="{Y - 10:.1f}" width="12" height="12" rx="2" class="f{s}"/>')
        p.text_px(X + 17, Y, lab, cls="lbl small")
        X += 17 + 12.4 * len(lab) + 16
    return X


def symbol_row(p, Y):
    p.top.append(f'<polygon points="{p.ml + 1:.1f},{Y - 9:.1f} {p.ml + 11:.1f},{Y - 9:.1f} {p.ml + 6:.1f},{Y - 1:.1f}" class="f1"/>')
    p.text_px(p.ml + 17, Y, "약물 X 첫 처방", cls="lbl small")
    xx = p.ml + 132
    for a, b in ((-5, -5), (-5, 5)):
        p.top.append(f'<line x1="{xx + 6 + a:.1f}" y1="{Y - 5 + b:.1f}" x2="{xx + 6 - a:.1f}" y2="{Y - 5 - b:.1f}" class="strongref" stroke-width="2.2"/>')
    p.text_px(xx + 17, Y, "사망", cls="lbl small")


# ============================================================ 16-1 immortal-time timeline
pts = [  # label, start drug (months) or None, end (months), died
    ("환자 1", 2, 12, False),
    ("환자 2", 5, 9, True),
    ("환자 3", None, 1.5, True),
    ("환자 4", 8, 12, False),
    ("환자 5", None, 12, False),
    ("환자 6", None, 4, True),
]


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
    if correct:
        legend_row(p, [("비노출 시간", 4), ("노출 시간(약물 X 처방 이후)", 1)], B + 60)
    else:
        legend_row(p, [("비사용군으로 센 시간", 4), ("사용군으로 센 시간", 1), ("불멸시간", 2)], B + 60)
    symbol_row(p, B + 80)


kw = dict(w=600, h=350, ml=64, mr=90, mt=30, mb=98, xlabel="", xticks=[0, 2, 4, 6, 8, 10, 12], yticks=[],
          ygrid=False, show_yaxis=False, xgrid=True)
pa = Plot((0, 12), (0.4, 6.9), **kw)
draw(pa, False)
panel_title(pa, "(가) 잘못된 분류: 추적 중 한 번이라도 처방받으면 처음부터 사용군")
pb = Plot((0, 12), (0.4, 6.9), **kw)
draw(pb, True)
panel_title(pb, "(나) 시간의존 분류: 처방 전은 비노출, 처방 후는 노출")
save("ch16_immortal", figure(
    [pa.svg("불멸시간이 사용군에 잘못 들어가는 분류"), pb.svg("처방 시점에서 노출이 바뀌는 시간의존 분류")],
    "그림 16-1. 퇴원 후 1년 동안 추적한 환자 6명. (가) 환자 1, 2, 4는 처방을 받기 전까지 사망할 수 없었는데(주황), "
    "이 기간이 사용군의 추적 시간으로 들어가 사용군의 사망률을 낮춥니다. 일찍 사망한 환자 3, 6은 약을 시작할 기회도 없이 비사용군이 됩니다. "
    "(나) 같은 환자들을 처방 시점에서 노출이 바뀌도록 나누면 처방 전 시간은 비노출 쪽으로 갑니다.", cols=1))


# ============================================================ 16-2 naive Kaplan-Meier curves
def surv_curve(t, e, grid):
    """Kaplan-Meier on a grid (months); t in months"""
    o = np.argsort(t, kind="stable"); t, e = np.asarray(t)[o], np.asarray(e)[o]
    n = len(t); S = 1.0; out = []; j = 0
    for g in grid:
        while j < n and t[j] <= g:
            if e[j]:
                S *= 1 - 1 / (n - j)
            j += 1
        out.append(S)
    return np.array(out)


grid = np.linspace(0, 12, 241)
u, nu = dd[dd.user == 1], dd[dd.user == 0]
Su = surv_curve(u.end * 12, u.died, grid) * 100
Sn = surv_curve(nu.end * 12, nu.died, grid) * 100
Sa = surv_curve(dd.end * 12, dd.died, grid) * 100
p = Plot((0, 12), (65, 100), w=600, h=340, mr=30, mt=26, xlabel="퇴원 후 개월", ylabel="생존율 (%)",
         xticks=[0, 2, 4, 6, 8, 10, 12], yticks=[70, 80, 90, 100])
p.step(grid, Sa, s=4, dash=True, w=1.6)
p.step(grid, Sn, s=2)
p.step(grid, Su, s=1)
p.text(12, Su[-1], f"사용군 {Su[-1]:.1f}%", anchor="end", cls="lbl strong", dy=19)
p.text(12, Sn[-1], f"비사용군 {Sn[-1]:.1f}%", anchor="end", cls="lbl strong", dy=19)
p.text(12, Sa[-1], f"전체 {Sa[-1]:.1f}%", anchor="end", cls="lbl mute", dy=19)
i3 = int(np.searchsorted(grid, 3))
p.text(0.25, 100, f"사용군은 3개월까지 {100 - Su[i3]:.1f}%만 사망", cls="lbl small", dy=-8)
p.text(1.1, Sn[i3] - 6.5, f"비사용군은 3개월까지 {100 - Sn[i3]:.1f}% 사망", cls="lbl small")
OUT_NAIVE = dict(su3=float(100 - Su[i3]), sn3=float(100 - Sn[i3]), su12=float(Su[-1]), sn12=float(Sn[-1]), sa12=float(Sa[-1]))
print("naive KM: death by 3 mo users %.1f%% non-users %.1f%%; 12-mo survival %.1f / %.1f / all %.1f" %
      (OUT_NAIVE["su3"], OUT_NAIVE["sn3"], OUT_NAIVE["su12"], OUT_NAIVE["sn12"], OUT_NAIVE["sa12"]))
save("ch16_naivekm", figure(
    p.svg("추적 중 사용 여부로 나눈 잘못된 Kaplan-Meier 곡선"),
    "그림 16-2. 예제 코호트 6,000명을 '추적 중 한 번이라도 약물 X를 처방받았는가'로 나눠 퇴원일부터 그린 Kaplan-Meier 곡선(잘못된 분석). "
    "약물 X는 효과가 없는데도 두 곡선이 처음부터 크게 벌어집니다. 사용군 곡선이 초반에 거의 내려가지 않는 것이 불멸시간의 흔적입니다. 점선은 전체 환자의 곡선입니다."))

# ============================================================ 16-3 landmark analysis
LM = 3
lpts = [  # label, start drug, end, died, group at the 3-month landmark
    ("환자 1", 2, 12, False, "노출군"),
    ("환자 2", 5, 9, True, "비노출군"),
    ("환자 3", None, 1.5, True, "제외"),
    ("환자 4", 1, 7, True, "노출군"),
    ("환자 5", None, 12, False, "비노출군"),
    ("환자 6", None, 8, True, "비노출군"),
]
pl = Plot((0, 12), (0.0, 6.9), w=600, h=350, ml=64, mr=90, mt=30, mb=84, xlabel="", xticks=[0, 3, 6, 9, 12], yticks=[],
          ygrid=False, show_yaxis=False, xgrid=True)
for i, (lab, s0, end, died, grp) in enumerate(lpts):
    y = 6 - i
    pl.text_px(8, pl.sy(y) + 4.5, lab, cls="lbl small")
    pl.seg(0, y, min(end, LM), y, cls="ln s4", w=3, dash=True)
    if end > LM:
        pl.seg(LM, y, end, y, cls="ln s1" if grp == "노출군" else "ln s4", w=10)
    if s0 is not None:
        start_mark(pl, s0, y)
    if died:
        death_mark(pl, end, y)
    pl.text_px(pl.w - pl.mr + 14, pl.sy(y) + 4.5, grp, cls="lbl small")
pl.vline(LM, cls="strongref", dash=False, w=1.8)
pl.text(LM, 0.38, "랜드마크 (3개월)", cls="lbl strong", dx=7, dy=4)
B = pl.h - pl.mb
pl.text_px((pl.ml + pl.w - pl.mr) / 2, B + 34, "퇴원 후 개월", anchor="middle", cls="axlab")
X = legend_row(pl, [("랜드마크 시점의 노출군", 1), ("비노출군", 4)], B + 60)
pl.top.append(f'<line x1="{X:.1f}" y1="{B + 56:.1f}" x2="{X + 22:.1f}" y2="{B + 56:.1f}" class="ln s4" stroke-width="3" stroke-dasharray="4 4"/>')
pl.text_px(X + 28, B + 60, "분석에 쓰지 않는 시간", cls="lbl small")
panel_title(pl, "(가) 3개월에 살아 있는 사람만, 그때까지의 처방으로 나눔")

lm = dd[dd.end > 0.25].copy()
lm["xL"] = ((lm.user == 1) & (lm.tinit <= 0.25)).astype(int)
g2 = np.linspace(3, 12, 181)
S1 = surv_curve(lm[lm.xL == 1].end * 12, lm[lm.xL == 1].died, g2) * 100
S0 = surv_curve(lm[lm.xL == 0].end * 12, lm[lm.xL == 0].died, g2) * 100
pk = Plot((0, 12), (65, 100), w=600, h=300, mr=30, mt=30, xlabel="퇴원 후 개월", ylabel="생존율 (%)",
          xticks=[0, 3, 6, 9, 12], yticks=[70, 80, 90, 100])
pk.vline(LM, cls="strongref", dash=False, w=1.8)
pk.step(g2, S0, s=4)
pk.step(g2, S1, s=1)
v = N["null"]["lm"]["3"]
pk.legend([(f"노출군 (12개월 생존율 {S1[-1]:.1f}%)", 1, "line"), (f"비노출군 ({S0[-1]:.1f}%)", 4, "line")],
          X=pk.sx(3.4), Y=pk.sy(77))
pk.text(1.5, 82, "랜드마크 이전은", anchor="middle", cls="lbl mute small")
pk.text(1.5, 82, "그리지 않음", anchor="middle", cls="lbl mute small", dy=16)
pk.text(3.4, 68.2, f"3개월 생존자 {v['n']:,}명: 노출군 {v['exposed']:,}명, 비노출군 {v['unexposed']:,}명", cls="lbl small")
panel_title(pk, "(나) 예제 코호트의 3개월 랜드마크 곡선")
print("landmark KM at 12 mo: exposed %.1f unexposed %.1f (json risk %.1f / %.1f)" %
      (S1[-1], S0[-1], v["risk_1"] * 100, v["risk_0"] * 100))
save("ch16_landmark", figure(
    [pl.svg("랜드마크 분석에서 환자를 나누는 방법"), pk.svg("3개월 랜드마크 시점부터 그린 두 군의 Kaplan-Meier 곡선")],
    "그림 16-3. 랜드마크 분석. (가) 3개월 전에 사망한 환자 3은 분석에서 빠지고, 3개월 뒤에 약을 시작한 환자 2는 비노출군에 남습니다. "
    "(나) 예제 코호트에서 3개월 생존자를 그 시점까지의 처방 여부로 나눠 3개월부터 그리면, 그림 16-2에서 크게 벌어졌던 두 곡선이 거의 겹칩니다.", cols=1))

# ============================================================ 16-4 exposure definitions over time (schematic, one patient)
fills = [(40, 30), (70, 30), (100, 30), (190, 30), (220, 30)]      # (dispensing day, days supplied)
GRACE, LAG = 30, 30
pe = Plot((0, 365), (0.3, 4.9), w=600, h=300, ml=150, mr=20, mt=18, mb=74, xlabel="",
          xticks=[0, 60, 120, 180, 240, 300, 365], yticks=[], ygrid=False, show_yaxis=False, xgrid=True)
rows = ["처방 기록", "① 시작하면 계속 노출", "② 현재 사용", "③ ①에 시차 30일"]
for i, lab in enumerate(rows):
    pe.text_px(6, pe.sy(4 - i) + 4.5, lab, cls="lbl small")
for d, k in fills:
    pe.seg(d, 4, d + k, 4, cls="ln s3", w=10)
    pe.seg(d, 3.75, d, 4.25, cls="strongref", w=1.2)
first = fills[0][0]
pe.seg(0, 3, first, 3, cls="ln s4", w=10); pe.seg(first, 3, 365, 3, cls="ln s1", w=10)
# current use: on from a fill until end of supply + grace; merge overlapping episodes
ep = []
for d, k in fills:
    a, b = d, d + k + GRACE
    if ep and a <= ep[-1][1]:
        ep[-1][1] = max(ep[-1][1], b)
    else:
        ep.append([a, b])
prev = 0
for a, b in ep:
    pe.seg(prev, 2, a, 2, cls="ln s4", w=10); pe.seg(a, 2, min(b, 365), 2, cls="ln s1", w=10); prev = b
pe.seg(prev, 2, 365, 2, cls="ln s4", w=10)
pe.seg(0, 1, first + LAG, 1, cls="ln s4", w=10); pe.seg(first + LAG, 1, 365, 1, cls="ln s1", w=10)
pe.seg(first, 1, first + LAG, 1, cls="ln s2", w=10)
B = pe.h - pe.mb
pe.text_px((pe.ml + pe.w - pe.mr) / 2, B + 34, "퇴원 후 일수", anchor="middle", cls="axlab")
legend_row(pe, [("처방된 공급일수", 3), ("비노출", 4), ("노출", 1), ("시차 구간(비노출로 셈)", 2)], B + 60, X=40)
print("exposure episodes (current use, grace %d d):" % GRACE, ep, "; lagged start day", first + LAG)
save("ch16_expdef", figure(
    pe.svg("같은 처방 기록에 세 가지 노출 정의를 적용한 모습"),
    "그림 16-4. 한 환자의 처방 기록(40일째부터 30일분씩 세 번, 190일째부터 두 번)에 세 가지 노출 정의를 적용한 모습. "
    "① 첫 처방 뒤로는 계속 노출 ② 공급일수가 끝나고 유예기간 30일이 지나면 비노출로 돌아가는 현재 사용 "
    "③ 첫 처방 후 30일(시차 구간)까지는 비노출로 세는 정의. 어느 정의든 그 시점까지의 기록만으로 노출을 정합니다."))

# ============================================================ 16-5 hazard ratio by lag time
lags = [0, 7, 14, 30, 60, 90]
pos = list(range(len(lags)))
hr = [N["proto"]["lag"][str(l)]["hr"] for l in lags]
pg = Plot((-0.6, 5.6), (np.log(0.5), np.log(8)), w=600, h=320, mt=24, mr=24, xlabel="노출 정의에 둔 시차 (일)", ylabel="위험비 (95% CI, 로그 눈금)",
          xticks=pos, yticks=[np.log(v) for v in (0.5, 1, 2, 4, 8)], ytickfmt=lambda v: f"{np.exp(v):g}",
          xticklabels=[(i, str(l)) for i, l in zip(pos, lags)])
pg.hline(0, cls="strongref", dash=False, w=1.4)
for i, h in zip(pos, hr):
    pg.seg(i, np.log(h[1]), i, np.log(h[2]), cls="ln s1", w=2)
    pg.text(i, np.log(h[2]), f"{h[0]:.2f}", anchor="middle", cls="lbl small", dy=-8)
pg.points(pos, [np.log(h[0]) for h in hr], s=1, r=5)
pg.text(-0.5, 0, "위험비 1 (연관 없음)", cls="lbl mute small", dy=16)
save("ch16_lag", figure(
    pg.svg("시차를 늘릴수록 위험비가 1에 가까워지는 모습"),
    "그림 16-5. 예제 코호트에서 PPI와 위장관 출혈 입원의 위험비를 시차를 바꿔 가며 구한 결과. 실제로는 PPI가 출혈에 영향을 주지 않도록 만든 자료입니다. "
    "시차가 없으면 출혈의 초기 증상 때문에 나간 처방이 PPI의 위해처럼 보이고, 시차를 30일 이상 두면 위험비가 1 근처에서 더 변하지 않습니다."))
print("figures written")
