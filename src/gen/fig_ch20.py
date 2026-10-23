"""Figures for chapter 20 (경제성 평가의 틀).
run after nums: source /home/claude/pylibs/env.sh && python3 gen/nums_ch20.py && python3 gen/fig_ch20.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np
from svgplot import Plot, figure, panel_title

OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch20_nums.json"), encoding="utf-8"))
B0 = N["base"]
LAM = 5000.0
won = lambda v: f"{v:,.0f}"


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)


# ============================================================ 그림 20-1: ICER by time horizon
cv = [c for c in N["horizon_curve"] if c[0] >= 2]
xs = [c[0] for c in cv]; ys = [c[1] for c in cv]
p = Plot((0, 20), (4000, 12000), w=600, h=330, ml=70, mr=30, xlabel="분석기간 (년)", ylabel="ICER (만원/QALY)",
         xticks=[0, 3, 5, 10, 15, 20], yticks=[4000, 6000, 8000, 10000, 12000], ytickfmt=won)
p.hline(LAM, dash=True)
p.text(0.3, LAM, "이 예시에서 가정한 임계값 5,000만원/QALY", dy=16, cls="lbl small mute")
p.line(xs, ys, s=1, w=2.4)
hz = N["horizon"]
for y, (dx, dy, anc) in {"3": (10, 2, "start"), "5": (8, -8, "start"), "10": (0, -12, "middle"), "20": (-2, -12, "end")}.items():
    v = hz[y]["icer"]
    p.points([int(y)], [v], s=1, r=4.5)
    p.text(int(y), v, f"{y}년 {won(v)}", dx=dx, dy=dy, anchor=anc, cls="lbl small")
save("ch20_horizon", figure(
    p.svg("분석기간을 2년에서 20년까지 늘릴 때 신약 A 대 표준요법 B의 ICER"),
    "그림 20-1. 분석기간에 따른 신약 A 대 표준요법 B의 ICER(가상의 예시). 분석기간이 짧은 왼쪽에서 곡선이 가파르게 내려오다가 "
    "10년 무렵부터 거의 평평해지는 모양을 봅니다. 가로 점선은 이 예시에서 가정한 임계값입니다."))

# ============================================================ 그림 20-2: discount factor
t = np.linspace(0, 20, 201)
p = Plot((0, 20), (0, 1.0), w=600, h=320, ml=62, mr=70, xlabel="지금부터의 연수", ylabel="할인 계수 (현재가치 ÷ 원래 값)",
         xticks=[0, 1, 5, 10, 15, 20], yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], ytickfmt=lambda v: f"{v:.1f}")
for rate, s, dash in ((0.015, 3, True), (0.03, 2, True), (0.045, 1, False)):
    p.line(t, 1 / (1 + rate) ** t, s=s, dash=dash, w=2.4 if rate == 0.045 else 1.8)
    p.text(20, 1 / (1 + rate) ** 20, f"연 {rate * 100:g}%", dx=6, dy=4, cls="lbl small")
df = N["disc_factor"]["0.045"]
for y in ("1", "5", "10", "20"):
    p.points([int(y)], [df[y]], s=1, r=4.5)
    p.text(int(y), df[y], f"{df[y]:.3f}", dx=0 if y != "20" else -4, dy=18, anchor="middle" if y != "20" else "end", cls="lbl small")
save("ch20_discount", figure(
    p.svg("할인율 연 4.5%, 3%, 1.5%에서 시간에 따른 할인 계수"),
    "그림 20-2. 할인 계수. 실선과 점이 국내 지침의 연 4.5%이고, 점선은 국내 지침이 민감도 분석에 쓰는 3%와, 캐나다 CADTH(2017)의 기본값 1.5%입니다. "
    "세 선의 간격이 먼 미래로 갈수록 벌어지는 것, 곧 할인율의 차이가 늦게 생기는 비용과 건강에서 크게 나타나는 것을 봅니다."))

# ============================================================ 그림 20-3: cost-effectiveness plane
p1 = Plot((-1, 1), (-1, 1), w=420, h=360, ml=62, mr=16, mt=28, mb=46, xlabel="증분 효과 (QALY)", ylabel="증분 비용",
          xticklabels=[(-0.6, "덜 효과적"), (0, "0"), (0.6, "더 효과적")], yticks=[-0.6, 0, 0.6],
          ytickfmt=lambda v: {-0.6: "더 쌈", 0: "0", 0.6: "더 비쌈"}[round(v, 1)], ygrid=False)
p1.fill_between([-1, 1], [-1, -1], [-1, 1], s=1)
p1.vline(0, cls="strongref", dash=False)
p1.hline(0, cls="strongref", dash=False)
p1.line([-1, 1], [-1, 1], s=1, dash=True, w=1.8)
p1.text(0.36, 0.86, "더 비싸고 더 효과적", anchor="middle", cls="lbl strong small")
p1.text(0.36, 0.86, "ICER를 임계값과 비교", anchor="middle", cls="lbl small", dy=16)
p1.text(-0.5, 0.50, "더 비싸고 덜 효과적", anchor="middle", cls="lbl strong small")
p1.text(-0.5, 0.50, "열등: 채택하지 않음", anchor="middle", cls="lbl small", dy=16)
p1.text(0.5, -0.50, "더 싸고 더 효과적", anchor="middle", cls="lbl strong small")
p1.text(0.5, -0.50, "우월: 채택", anchor="middle", cls="lbl small", dy=16)
p1.text(-0.40, -0.80, "더 싸고 덜 효과적", anchor="middle", cls="lbl strong small")
p1.text(-0.40, -0.80, "절감액 대 잃는 건강", anchor="middle", cls="lbl small", dy=16)
p1.text(0.40, 0.24, "임계값 선", cls="lbl small mute")
panel_title(p1, "(가) 네 사분면")

dq, dc = B0["d_qaly"], B0["d_cost"]
p2 = Plot((0, 0.8), (0, 4000), w=420, h=360, ml=58, mr=16, mt=28, mb=46, xlabel="증분 QALY (신약 A − 표준요법 B)", ylabel="증분 비용 (만원)",
          xticks=[0, 0.2, 0.4, 0.6, 0.8], xtickfmt=lambda v: f"{v:.1f}", yticks=[0, 1000, 2000, 3000, 4000], ytickfmt=won)
p2.fill_between([0, 0.8], [0, 0], [0, 4000], s=1)
p2.line([0, 0.8], [0, 4000], s=1, dash=True, w=1.8)
p2.seg(0, 0, dq, dc, cls="ln s2", w=1.8)
p2.seg(dq, 0, dq, LAM * dq, cls="ref", dash=True)
p2.seg(0, dc, dq, dc, cls="ref", dash=True)
p2.seg(dq, LAM * dq, dq, dc, cls="ln s2", w=3)
p2.points([dq], [dc], s=2, r=5.5)
p2.text(dq, dc, "신약 A 대 표준요법 B", anchor="end", dx=-10, dy=-22, cls="lbl strong small")
p2.text(dq, dc, f"({dq:.3f}, {won(dc)})", anchor="end", dx=-10, dy=-8, cls="lbl small")
p2.text(dq, LAM * dq, f"차이 {won(-B0['nmb'])}만원", dx=9, dy=18, cls="lbl small")
p2.text(0.20, 1450, f"기울기 = ICER {won(B0['icer'])}", cls="lbl small", dx=-64, dy=-22)
p2.text(0.30, 1500, "임계값 선 (기울기 5,000)", cls="lbl small mute", dx=14, dy=30)
p2.text(0.60, 500, "비용효과적인 영역", anchor="middle", cls="lbl small mute")
panel_title(p2, "(나) 공통 예시의 기준 분석")
save("ch20_plane", figure(
    [p1.svg("비용효과평면의 네 사분면과 임계값 선"), p2.svg("공통 예시의 증분 QALY와 증분 비용을 비용효과평면에 찍은 그림")],
    "그림 20-3. 비용효과평면(가상의 예시). (가) 원점이 비교대안이고, 색칠한 부분(임계값 선의 오른쪽 아래)이 비용효과적인 영역입니다. "
    "(나) 공통 예시의 점이 임계값 선보다 위에 있는 것, 곧 원점과 점을 이은 선이 임계값 선보다 가파른 것을 봅니다. "
    "그 선의 기울기가 ICER이고, 점과 임계값 선 사이의 굵은 세로 선의 길이가 순금전편익의 크기입니다. 점이 선보다 위에 있으므로 순금전편익은 음수입니다.",
    cols=2))

# ============================================================ 그림 20-4: efficiency frontier
S = {d["name"]: d for d in N["strats"]}
b, c, d_, a = S["표준요법 B"], S["기존약 C"], S["병용요법 D"], S["신약 A"]
mix = N["mix"]
p = Plot((1.6, 2.5), (7000, 11000), w=600, h=360, ml=70, mr=30, xlabel="총 QALY", ylabel="총비용 (만원)",
         xticks=[1.6, 1.8, 2.0, 2.2, 2.4], xtickfmt=lambda v: f"{v:.1f}", yticks=[7000, 8000, 9000, 10000, 11000], ytickfmt=won)
p.line([b["eff"], d_["eff"], a["eff"]], [b["cost"], d_["cost"], a["cost"]], s=4, dash=True, w=1.6)
p.line([b["eff"], a["eff"]], [b["cost"], a["cost"]], s=1, w=2.6)
p.seg(d_["eff"], mix["cost"], d_["eff"], d_["cost"], cls="ref", dash=True)
p.points([d_["eff"]], [mix["cost"]], s=1, r=4, hollow=True)
p.points([b["eff"], a["eff"]], [b["cost"], a["cost"]], s=1, r=5.5)
p.points([c["eff"], d_["eff"]], [c["cost"], d_["cost"]], s=2, r=5.5)
p.text(b["eff"], b["cost"], "표준요법 B", dx=10, dy=16, cls="lbl strong small")
p.text(a["eff"], a["cost"], "신약 A", dx=-10, dy=-10, anchor="end", cls="lbl strong small")
p.text(c["eff"], c["cost"], "기존약 C (열등)", dx=0, dy=-12, anchor="middle", cls="lbl small")
p.text(d_["eff"], d_["cost"], "병용요법 D (확장 열등)", dx=-8, dy=-10, anchor="end", cls="lbl small")
p.text(d_["eff"], mix["cost"], f"B와 A를 섞으면 {won(mix['cost'])}", dx=10, dy=14, cls="lbl small")
p.text(2.22, 9650, f"효율 경계: {won(B0['icer'])}만원/QALY", dx=8, dy=14, cls="lbl small")
p.legend([("효율 경계 위의 대안", 1, "dot"), ("경계에서 빠지는 대안", 2, "dot")], X=82, Y=34)
save("ch20_frontier", figure(
    p.svg("네 대안의 총 QALY와 총비용, 효율 경계"),
    "그림 20-4. 네 대안의 효율 경계(가상의 예시). 기존약 C가 표준요법 B보다 위(더 비쌈)이면서 왼쪽(QALY가 적음)에 있는 것, "
    "병용요법 D가 B와 A를 잇는 실선(효율 경계)보다 위에 있는 것을 봅니다. D 아래의 빈 점은 환자 일부에게 A를, 나머지에게 B를 쓸 때의 평균으로, "
    "D와 QALY가 같으면서 더 쌉니다. 경계에 남은 B와 A 사이의 기울기가 A의 ICER입니다."))
print("figures written")
