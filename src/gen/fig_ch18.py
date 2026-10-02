"""Figures for chapter 18 (정책 효과의 평가).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch18.py && python3 gen/fig_ch18.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from svgplot import Plot, figure, panel_title, fmt

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch18_nums.json")))
M = lambda v, nd=1: f"{v:.{nd}f}".replace("-", "−")


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


t = np.arange(1, 61)
T0 = 36
yp = np.array(N["series"]["yp"]); yc = np.array(N["series"]["yc"]); yd = np.array(N["series"]["yd"])
YEARS = [(1, "2015"), (13, "2016"), (25, "2017"), (37, "2018"), (49, "2019")]
XL = "연도 (눈금은 각 해의 1월)"
YL = "처방률 (1,000명당, 월)"

# ============================================================ 그림 18-1. 전후 평균 비교
nv = N["naive"]
panels = []
for key, y, title, s in (("pilot", yp, "(a) 시범 지역 (사업 시행)", 1), ("comp", yc, "(b) 비교 지역 (사업 없음)", 2)):
    p = Plot((0, 61), (62, 100), w=420, h=320, xlabel=XL, ylabel=YL, xticklabels=YEARS, yticks=[65, 70, 75, 80, 85, 90, 95, 100], mt=30)
    p.vline(36.5, dash=True)
    p.points(t, y, s=4, r=2.8)
    a, b = nv[key]["pre"], nv[key]["post"]
    p.seg(1, a, 36, a, cls=f"ln s{s}", w=2.6)
    p.seg(37, b, 60, b, cls=f"ln s{s}", w=2.6)
    p.text(59.5, 97.5, f"평균 {M(a)} → {M(b)}", anchor="end", cls="lbl small")
    p.text(59.5, 93.6, f"차이 {M(nv[key]['diff'])}", anchor="end", cls="lbl strong")
    p.text(36.5, 63.2, "2018년 1월", anchor="end", dx=-5, cls="lbl small")
    panel_title(p, title)
    panels.append(p.svg(title))
save("ch18_naive", figure(
    panels,
    "그림 18-1. 65세 이상 외래환자 1,000명당 수면진정제 처방 환자 수의 월별 값(회색 점)과 시행 전후 평균(굵은 선). 가상 자료입니다. "
    "시범 지역의 전후 평균 차이는 −17.5이지만, 사업이 없었던 비교 지역에서도 −10.1이 나옵니다. 두 지역 모두 2015년부터 이미 내려가고 있었기 때문입니다.",
    cols=2))

# ============================================================ 그림 18-2. ITS
b = N["its"]["b"]
p = Plot((0, 61), (62, 100), w=600, h=350, xlabel=XL, ylabel=YL, xticklabels=YEARS, yticks=[65, 70, 75, 80, 85, 90, 95, 100])
p.vline(36.5, dash=True)
p.points(t, yp, s=4, r=3.0)
pre_t = np.array([1, 36]); post_t = np.array([37, 60])
p.line(pre_t, b[0] + b[1] * pre_t, s=1, w=2.6)
p.line(post_t, b[0] + b[1] * post_t + b[2] + b[3] * (post_t - T0), s=1, w=2.6)
p.line(np.array([36, 60]), b[0] + b[1] * np.array([36, 60]), s=2, dash=True, w=2.2)
p.text(36.5, 98.3, "시범사업 시작 (2018년 1월)", anchor="start", dx=6, cls="lbl small")
# level change bracket just after the start
ya = b[0] + b[1] * 36.5
yb = ya + b[2]
p.seg(37.6, ya, 37.6, yb, cls="strongref", w=1.6)
p.text(36, (ya + yb) / 2, f"수준 변화 {M(b[2])}", anchor="end", dx=-3, dy=11, cls="lbl small")
e12 = N["its_eff"]["12"]
p.seg(48, e12["cf"], 48, e12["pred"], cls="strongref", w=1.6)
p.text(48.6, (e12["cf"] + e12["pred"]) / 2, f"12개월째 {M(e12['diff'])}", dy=4, cls="lbl small")
p.text(60, b[0] + b[1] * 60, "반사실", anchor="end", dy=-8, cls="lbl small")
p.legend([("월별 관측값", 4, "dot"), ("분절회귀 적합선", 1, "line"), ("시행 전 추세의 연장(반사실)", 2, "dash")],
         X=p.ml + 12, Y=p.h - p.mb - 58)
save("ch18_its", figure(
    [p.svg("중단시계열 분절회귀")],
    "그림 18-2. 시범 지역의 월별 처방률에 분절회귀를 적합한 결과(가상 자료). 시행 전 36개월의 추세를 연장한 주황 점선이 '사업이 없었다면'의 추정(반사실)입니다. "
    "시행 직후 적합선이 반사실보다 7.4 낮게 시작하고(수준 변화), 그 뒤 두 선의 간격이 매달 0.23씩 더 벌어집니다(기울기 변화).",
    cols=1))

# ============================================================ 그림 18-3. 잔차와 자기상관
res = np.array(N["its"]["resid"]); ac = np.array(N["its"]["acf"])
p1 = Plot((0, 61), (-4, 4), w=420, h=300, xlabel=XL, ylabel="잔차 (관측값 − 적합값)", xticklabels=YEARS, yticks=[-4, -2, 0, 2, 4], mt=30,
          ytickfmt=lambda v: fmt(v).replace("-", "−"))
p1.hline(0, dash=False)
p1.vline(36.5, dash=True)
p1.line(t, res, s=1, w=1.6)
p1.points(t, res, s=1, r=2.6)
panel_title(p1, "(a) 달마다의 잔차")
bd = 1.96 / np.sqrt(60)
p2 = Plot((0.2, 12.8), (-0.8, 0.8), w=420, h=300, xlabel="시차 (개월)", ylabel="잔차의 자기상관", xticks=list(range(1, 13)),
          yticks=[-0.8, -0.4, 0, 0.4, 0.8], mt=30, ytickfmt=lambda v: fmt(v).replace("-", "−"))
p2.fill_between([0.2, 12.8], [-bd, -bd], [bd, bd], cls="a4")
p2.hline(0, dash=False)
p2.bars(list(range(1, 13)), ac[1:], 0.6, s=1, rounded=False)
p2.text(1, ac[1], M(ac[1], 2), anchor="middle", dy=-6, cls="lbl small")
p2.text(0.5, -0.68, "회색 띠: 우연의 범위 (±0.25)", cls="lbl small")
panel_title(p2, "(b) 시차별 자기상관")
save("ch18_acf", figure(
    [p1.svg("분절회귀 잔차"), p2.svg("잔차의 자기상관함수")],
    "그림 18-3. 그림 18-2의 분절회귀에서 남은 잔차. (a) 양(+)인 달 뒤에는 양인 달이, 음(−)인 달 뒤에는 음인 달이 이어지는 구간이 보입니다. "
    "(b) 한 달 간격의 자기상관이 0.51로 회색 띠(자기상관이 없을 때 우연히 나올 수 있는 범위)를 벗어나고, 6–7개월 간격에서 음, 12개월 간격에서 다시 양이 되는 계절 무늬도 보입니다.",
    cols=2))

# ============================================================ 그림 18-4. 대조군이 있는 ITS
cp, cc, cd = N["cits"]["pilot"]["b"], N["cits"]["comp"]["b"], N["cits"]["diff"]["b"]
p1 = Plot((0, 61), (62, 100), w=420, h=320, xlabel=XL, ylabel=YL, xticklabels=YEARS, yticks=[65, 70, 75, 80, 85, 90, 95, 100], mt=30)
p1.vline(36.5, dash=True)
for y, bb, s in ((yp, cp, 1), (yc, cc, 2)):
    p1.points(t, y, s=s, r=2.3)
    p1.line(pre_t, bb[0] + bb[1] * pre_t, s=s, w=2.2)
    p1.line(post_t, bb[0] + bb[1] * post_t + bb[2] + bb[3] * (post_t - T0), s=s, w=2.2)
p1.text(17, 96.8, "시범 지역", cls="lbl small")
p1.text(2, 78.5, "비교 지역", cls="lbl small")
p1.text(38, 88.5, f"시범 {M(cp[2])}", cls="lbl small")
p1.text(38, 65.5, f"비교 {M(cc[2])}", cls="lbl small")
panel_title(p1, "(a) 두 지역의 처방률")
p2 = Plot((0, 61), (0, 16), w=420, h=320, xlabel=XL, ylabel="처방률의 차이 (시범 − 비교)", xticklabels=YEARS, yticks=[0, 4, 8, 12, 16], mt=30)
p2.vline(36.5, dash=True)
p2.points(t, yd, s=4, r=2.6)
p2.line(pre_t, cd[0] + cd[1] * pre_t, s=1, w=2.4)
p2.line(post_t, cd[0] + cd[1] * post_t + cd[2] + cd[3] * (post_t - T0), s=1, w=2.4)
p2.line(np.array([36, 60]), cd[0] + cd[1] * np.array([36, 60]), s=2, dash=True, w=2.0)
ya = cd[0] + cd[1] * 36.5
p2.seg(37.6, ya, 37.6, ya + cd[2], cls="strongref", w=1.6)
p2.text(36, ya + cd[2] / 2, f"수준 변화 {M(cd[2])}", anchor="end", dx=-3, dy=14, cls="lbl small")
d12 = N["cits"]["diff"]["e12"][0]
cf48 = cd[0] + cd[1] * 48
p2.seg(48, cf48, 48, cf48 + d12, cls="strongref", w=1.6)
p2.text(48.6, cf48 + d12 / 2, f"12개월째 {M(d12)}", dy=4, cls="lbl small")
p2.text(60, cd[0] + cd[1] * 60, "반사실", anchor="end", dy=-8, cls="lbl small")
panel_title(p2, "(b) 두 지역의 차이")
save("ch18_cits", figure(
    [p1.svg("시범 지역과 비교 지역의 분절회귀"), p2.svg("차이 계열의 분절회귀")],
    "그림 18-4. 대조군이 있는 중단시계열(가상 자료). (a) 사업이 없었던 비교 지역(주황)도 2018년 1월에 2.5 내려갔습니다. 같은 달 전국에 배포된 안전성 서한의 영향으로 볼 수 있는 부분입니다. "
    "(b) 달마다 두 지역의 차이를 구하면 두 지역이 함께 겪은 변화가 지워지고, 시범 지역에만 있었던 변화(수준 −4.8, 기울기 −0.20/월)가 남습니다. 점들이 (a)보다 선에 가깝게 모여 있습니다.",
    cols=2))

# ============================================================ 그림 18-5. DID
dd = N["did"]
p = Plot((0, 61), (62, 100), w=600, h=350, xlabel=XL, ylabel=YL, xticklabels=YEARS, yticks=[65, 70, 75, 80, 85, 90, 95, 100], mr=30)
p.vline(36.5, dash=True)
p.points(t, yp, s=1, r=2.0)
p.points(t, yc, s=2, r=2.0)
p.seg(1, dd["tp"], 36, dd["tp"], cls="ln s1", w=2.8)
p.seg(37, dd["tpost"], 60, dd["tpost"], cls="ln s1", w=2.8)
p.seg(1, dd["cp"], 36, dd["cp"], cls="ln s2", w=2.8)
p.seg(37, dd["cpost"], 60, dd["cpost"], cls="ln s2", w=2.8)
tcf = dd["tp"] + (dd["cpost"] - dd["cp"])
p.line([37, 60], [tcf, tcf], s=1, dash=True, w=2.2)
p.text(36, dd["tp"], f"시범 지역 {M(dd['tp'], 2)}", anchor="end", dy=-9, cls="lbl small")
p.text(36, dd["cp"], f"비교 지역 {M(dd['cp'], 2)}", anchor="end", dy=-9, cls="lbl small")
p.text(48.5, tcf, f"시범 지역의 반사실 {M(tcf, 2)}", anchor="middle", dy=-8, cls="lbl small")
p.text(54, dd["tpost"], M(dd["tpost"], 2), anchor="middle", dy=-7, cls="lbl small")
p.text(37.5, dd["cpost"], M(dd["cpost"], 2), anchor="start", dy=17, cls="lbl small")
p.seg(42, tcf, 42, dd["tpost"], cls="strongref", w=1.6)
p.text(42.6, (tcf + dd["tpost"]) / 2, f"이중차분 {M(dd['did'], 2)}", dy=4, cls="lbl small strong")
p.text(36.5, 98.3, "시범사업 시작", anchor="start", dx=6, cls="lbl small")
save("ch18_did", figure(
    [p.svg("이중차분법")],
    "그림 18-5. 이중차분법(가상 자료). 굵은 가로선은 시행 전 36개월과 시행 후 24개월의 평균입니다. 비교 지역의 변화(−10.06)를 시범 지역의 시행 전 평균에 그대로 옮긴 파란 점선이 "
    "'사업이 없었다면'의 추정이고, 실제 평균과 이 점선의 간격 −7.40이 이중차분 추정값입니다. 두 지역의 수준은 다르지만 시행 전 점들은 나란히 내려갑니다.",
    cols=1))

# ============================================================ 그림 18-6. event study
ev = N["event"]
xs = [e["h"] + (1 if e["h"] >= 0 else 0) for e in ev]      # -6..-1, 1..4
p = Plot((-6.7, 4.7), (-12, 3), w=600, h=330, xlabel="시행 전후 반기 (−1 = 시행 직전 반기, 기준)", ylabel="두 지역 차이의 변화 (기준 반기 대비)",
         xticks=[-6, -5, -4, -3, -2, -1, 1, 2, 3, 4], xtickfmt=lambda v: ("+" if v > 0 else "") + fmt(v).replace("-", "−"),
         yticks=[-12, -9, -6, -3, 0, 3], ytickfmt=lambda v: fmt(v).replace("-", "−"))
p.hline(0, dash=False)
p.vline(0, dash=True)
for e, x in zip(ev, xs):
    if e.get("ref"):
        p.points([x], [0], s=4, r=4.5, hollow=True)
        continue
    s = 1 if e["h"] >= 0 else 4
    p.seg(x, e["lo"], x, e["hi"], cls=f"ln s{s}", w=2)
    p.points([x], [e["b"]], s=s, r=4.5)
    if e["h"] >= 0:
        p.text(x, e["b"], M(e["b"]), dx=9, dy=4, cls="lbl small")
p.text(0, 2.2, "시범사업 시작", anchor="start", dx=6, cls="lbl small")
p.text(-6.5, -10.8, "시행 전: 0 근처 (평행 추세와 모순되지 않음)", cls="lbl small")
save("ch18_event", figure(
    [p.svg("사건연구 그림")],
    "그림 18-6. 사건연구(event study) 그림(가상 자료). 반기마다 '시범 지역과 비교 지역의 차이'가 시행 직전 반기에 비해 얼마나 달라졌는지와 95% 신뢰구간입니다. "
    "시행 전 다섯 반기(회색)는 모두 0 근처이고, 시행 후(파랑)에는 −6.0에서 −9.6으로 점점 커집니다. 이중차분 추정값 −7.40은 시행 후 기간 전체를 하나로 평균한 값입니다.",
    cols=1))
print("figures written")
