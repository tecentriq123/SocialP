import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, scipy.stats as st
from html import escape
from svgplot import Plot, figure, panel_title, fmt
import nums_ch08 as N

R = N.R
OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
os.makedirs(OUT, exist_ok=True)
L = np.log


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def _tw(s, fs):
    w = 0.0
    for c in s:
        if "가" <= c <= "힣":
            w += fs * 1.0
        elif c == " ":
            w += fs * 0.3
        elif c in "%WMm":
            w += fs * 0.85
        else:
            w += fs * 0.6
    return w


def halo(p, x, y, s, anchor="middle", cls="lbl", dx=0, dy=0, fs=13):
    w = _tw(s, fs)
    X, Y = p.sx(x) + dx, p.sy(y) + dy
    x0 = X - w / 2 if anchor == "middle" else (X - w if anchor == "end" else X)
    p.top.append(f'<rect x="{x0 - 3:.1f}" y="{Y - fs + 1:.1f}" width="{w + 6:.1f}" height="{fs + 4:.1f}" rx="3" class="pth" stroke="none"/>')
    p.text(x, y, s, anchor=anchor, cls=cls, dx=dx, dy=dy)


def pct(v, nd=0):
    return f"{v * 100:.{nd}f}%"


# ------------------------------------------------------------------ 8-1 twenty 95% CIs
sim = R["sim20"]
ci = sim["ci"]
p = Plot((-1.45, 0.25), (0.3, 20.7), w=600, h=440, ml=56, mr=24, mt=30, mb=50,
         xlabel="표본에서 구한 HbA1c 평균 변화 (%p)와 95% 신뢰구간", ylabel="반복한 연구 번호",
         xticks=[-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2], xtickfmt=lambda v: fmt(round(v, 1)),
         yticks=[1, 5, 10, 15, 20], ygrid=False)
p.vline(sim["mu"], cls="strongref", dash=False, w=1.6)
for i, (mm, lo, hi) in enumerate(ci):
    yv = 20 - i
    s = 2 if i in sim["miss"] else 1
    p.seg(lo, yv, hi, yv, cls=f"ln s{s}", w=2.2 if s == 2 else 1.8)
    p.points([mm], [yv], s=s, r=3.6)
im = sim["miss"][0]
mm, lo, hi = ci[im]
p.text(lo, 20 - im, "참값을 놓친 구간", anchor="start", dy=-8, cls="lbl strong")
p.text(sim["mu"], 20.7, "참값 μ = −0.5 (고정)", anchor="middle", dy=-10, cls="lbl strong")
save("ch08_sim20", figure(p.svg("같은 모집단에서 뽑은 20개 표본의 95% 신뢰구간"),
     "그림 8-1. 참값(평균 HbA1c 변화 −0.5%p, SD 0.8)이 정해진 모집단에서 25명씩 20번 표본을 뽑아 각각 95% 신뢰구간을 구했습니다. "
     "구간은 표본마다 움직이고 참값은 움직이지 않습니다. 20개 중 19개(파랑)가 참값을 포함했고 1개(주황)는 놓쳤습니다. "
     "'95%'는 이 절차를 반복할 때 참값을 포함하는 구간의 비율입니다."))

# ------------------------------------------------------------------ 8-2 ratio CI on linear vs log axis
rr = R["rr5"]
p1 = Plot((0.4, 1.3), (0, 1), w=420, h=170, ml=20, mr=20, mt=30, mb=46, show_yaxis=False, ygrid=False,
          xticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], xtickfmt=lambda v: fmt(round(v, 1)),
          xlabel="상대위험도 (일반 눈금)")
p1.vline(1.0, cls="ref", dash=True, y0=0.05, y1=0.95)
p1.seg(rr["lo"], 0.45, rr["hi"], 0.45, cls="ln s1", w=2.4)
p1.points([rr["rr"]], [0.45], s=1, r=5.5)
p1.seg(rr["lo"], 0.75, rr["rr"], 0.75, cls="ln s2", w=1.4)
p1.seg(rr["rr"], 0.82, rr["hi"], 0.82, cls="ln s3", w=1.4)
p1.text((rr["lo"] + rr["rr"]) / 2, 0.75, f"{rr['dlo']:.2f}", anchor="middle", dy=-5, cls="lbl small")
p1.text((rr["rr"] + rr["hi"]) / 2, 0.82, f"{rr['dhi']:.2f}", anchor="middle", dy=-5, cls="lbl small")
p1.text(rr["lo"], 0.45, f"{rr['lo']:.2f}", anchor="end", dx=-6, dy=4, cls="lbl small")
p1.text(rr["hi"], 0.45, f"{rr['hi']:.2f}", anchor="start", dx=6, dy=4, cls="lbl small")
p1.text(rr["rr"], 0.45, f"{rr['rr']:.2f}", anchor="middle", dy=22, cls="lbl strong")
panel_title(p1, "가. 일반 눈금: 점추정값 양쪽 길이가 다름")
lx = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2]
p2 = Plot((L(0.4), L(1.3)), (0, 1), w=420, h=170, ml=20, mr=20, mt=30, mb=46, show_yaxis=False, ygrid=False,
          xticks=[L(v) for v in lx], xtickfmt=lambda v: fmt(round(math.exp(v), 2)), xlabel="상대위험도 (로그 눈금)")
p2.vline(0.0, cls="ref", dash=True, y0=0.05, y1=0.95)
p2.seg(L(rr["lo"]), 0.45, L(rr["hi"]), 0.45, cls="ln s1", w=2.4)
p2.points([L(rr["rr"])], [0.45], s=1, r=5.5)
p2.seg(L(rr["lo"]), 0.75, L(rr["rr"]), 0.75, cls="ln s2", w=1.4)
p2.seg(L(rr["rr"]), 0.82, L(rr["hi"]), 0.82, cls="ln s3", w=1.4)
d_ = Z_ = 1.959963984540054 * rr["se"]
p2.text((L(rr["lo"]) + L(rr["rr"])) / 2, 0.75, f"{d_:.3f}", anchor="middle", dy=-5, cls="lbl small")
p2.text((L(rr["rr"]) + L(rr["hi"])) / 2, 0.82, f"{d_:.3f}", anchor="middle", dy=-5, cls="lbl small")
p2.text(L(rr["rr"]), 0.45, f"ln {rr['rr']:.2f} = −{abs(rr['ln']):.3f}", anchor="middle", dy=22, cls="lbl strong")
panel_title(p2, "나. 로그 눈금: 양쪽 길이가 같음 (1.96 × SE)")
save("ch08_logci", figure([p1.svg("일반 눈금의 상대위험도 신뢰구간"), p2.svg("로그 눈금의 상대위험도 신뢰구간")],
     f"그림 8-2. 5장 약물 A·B 자료의 상대위험도 {rr['rr']:.2f}(95% CI {rr['lo']:.2f}–{rr['hi']:.2f}). 신뢰구간은 로그 척도에서 "
     f"ln RR ± 1.96 × SE로 대칭으로 만든 뒤 지수함수로 되돌리므로, 일반 눈금에서는 아래쪽이 {rr['dlo']:.2f}, 위쪽이 {rr['dhi']:.2f}로 비대칭입니다. "
     "비(ratio)를 그리는 포레스트 플롯이 로그 눈금을 쓰는 이유입니다.", cols=2))

# ------------------------------------------------------------------ 8-3 CI vs clinically important difference
sc = R["scen"]
txt = {"A": ("유의 + 임상적으로 중요", "구간 전체가 5 mmHg 위"),
       "B": ("유의, 중요성은 불확실", "0.6은 사소, 8.0은 중요"),
       "C": ("유의하지만 사소함", "구간 전체가 0과 5 사이"),
       "D": ("유의하지 않음, 판단 불가", "중요한 효과도 배제 못 함"),
       "E": ("유의하지 않음, 정밀한 무효", "5 mmHg 효과를 배제")}
p = Plot((-4, 14), (0.4, 5.6), w=660, h=330, ml=40, mr=222, mt=26, mb=50, show_yaxis=False, ygrid=False, xgrid=True,
         xticks=[-4, -2, 0, 2, 4, 6, 8, 10, 12, 14], xlabel="수축기혈압 감소량의 차이 (신약 − 대조, mmHg)")
p.els.append(f'<rect x="{p.sx(5):.1f}" y="{p.mt}" width="{p.sx(14) - p.sx(5):.1f}" height="{p.h - p.mt - p.mb}" class="a3"/>')
p.vline(0, cls="strongref", dash=False, w=1.4)
p.vline(5, cls="ln s3", dash=True, w=1.6)
p.text(0, 5.6, "효과 없음", anchor="middle", dy=-8, cls="lbl mute small")
p.text(5, 5.6, "최소 임상적 중요 차이 5", anchor="start", dx=4, dy=-8, cls="lbl small")
for i, r in enumerate(sc):
    yv = 5 - i
    p.seg(r["lo"], yv, r["hi"], yv, cls="ln s1", w=2.4)
    p.points([r["est"]], [yv], s=1, r=5)
    p.text_px(18, p.sy(yv) + 5, r["lab"], anchor="middle", cls="lbl strong")
    pv = "P < 0.001" if r["p"] < 0.001 else (f"P = {r['p']:.3f}" if r["p"] < 0.01 else f"P = {r['p']:.2f}")
    f1 = lambda v: f"{v:.1f}".replace("-", "−")
    X = p.w - p.mr + 14
    p.text_px(X, p.sy(yv) - 1, txt[r["lab"]][0], cls="lbl strong", size=12.5)
    p.text_px(X, p.sy(yv) + 14, f"{f1(r['est'])} ({f1(r['lo'])} to {f1(r['hi'])}), {pv}", cls="lbl mute small")
save("ch08_mcid", figure(p.svg("신뢰구간과 최소 임상적 중요 차이"),
     "그림 8-3. 같은 '유의함' 또는 '유의하지 않음'도 신뢰구간이 최소 임상적 중요 차이(초록 점선, 5 mmHg)와 어떻게 놓여 있는지에 따라 뜻이 다릅니다. "
     "B는 1장 나 절의 예(4.3 mmHg, 95% CI 0.6–8.0)입니다. P 값은 구간에서 역산한 근삿값입니다."))

# ------------------------------------------------------------------ 8-4 same RR, different baseline risk
RRc = 0.625
x = np.linspace(0.005, 0.5, 300)
orv = (RRc * x / (1 - RRc * x)) / (x / (1 - x))
pA = Plot((0, 0.5), (0.4, 1.05), w=420, h=300, ml=54, mr=16, xlabel="대조군의 위험 (p₀)", ylabel="비 (ratio)",
          xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], xtickfmt=lambda v: pct(v), yticks=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
pA.hline(1.0, cls="ref", dash=False, w=1)
pA.line(x, np.full_like(x, RRc), s=1, w=2.4)
pA.line(x, orv, s=2, w=2.4)
o24 = R["rct"]["readm"]["orr"][0]
pA.points([0.24], [RRc], s=1, r=4.5)
pA.points([0.24], [o24], s=2, r=4.5)
pA.text(0.49, RRc, f"상대위험도 {RRc} (고정)", anchor="end", dy=-8, cls="lbl strong")
pA.text(0.24, o24, f"p₀ 24%: OR {o24:.2f}", anchor="end", dx=-6, dy=20, cls="lbl strong")
pA.text(0.46, orv[-20], "오즈비", anchor="end", dy=16, cls="lbl")
panel_title(pA, "가. 오즈비는 p₀가 클수록 1에서 멀어짐")
nnt = 1 / (x * (1 - RRc))
pB = Plot((0, 0.5), (L(4), L(300)), w=420, h=300, ml=54, mr=16, xlabel="대조군의 위험 (p₀)", ylabel="치료필요수 NNT (로그 눈금)",
          xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], xtickfmt=lambda v: pct(v), yticks=[L(v) for v in (5, 10, 20, 50, 100, 200)],
          ytickfmt=lambda v: fmt(round(math.exp(v))))
msk = nnt <= 300
pB.line(x[msk], L(nnt[msk]), s=1, w=2.4)
for p0v, dy_, dx_, an in ((0.024, -2, 10, "start"), (0.10, -8, 8, "start"), (0.24, -12, 4, "start")):
    nv = 1 / (p0v * (1 - RRc))
    pB.points([p0v], [L(nv)], s=2, r=4.5)
    pB.text(p0v, L(nv), f"p₀ {p0v * 100:g}%: ARR {p0v * (1 - RRc) * 100:.1f}%p", anchor=an, dx=dx_, dy=dy_,
            cls="lbl strong" if p0v != 0.10 else "lbl")
    pB.text(p0v, L(nv), f"NNT {nv:.0f}", anchor=an, dx=dx_, dy=dy_ + 15, cls="lbl strong" if p0v != 0.10 else "lbl")
panel_title(pB, "나. 같은 RR 0.625에서 NNT")
save("ch08_baseline", figure([pA.svg("대조군 위험에 따른 오즈비"), pB.svg("대조군 위험에 따른 NNT")],
     "그림 8-4. 상대위험도를 0.625(상대위험 감소 37.5%)로 고정하고 대조군의 위험 p₀만 바꿨습니다. "
     "가: 사건이 흔할수록 같은 효과를 오즈비로 나타내면 1에서 더 멀어집니다(과장되어 보입니다). "
     "나: 상대적 효과가 같아도 기저위험이 낮으면 절대위험 감소(ARR)가 작아져 NNT가 커집니다. 기저위험 2.4%에서는 111명을 치료해야 1명의 재입원을 막습니다.",
     cols=2))

# ------------------------------------------------------------------ 8-5 exponential function and the two scales
xs = np.linspace(-1.6, 1.6, 300)
pE = Plot((-1.6, 1.6), (0, 5), w=420, h=320, ml=46, mr=16, xlabel="로그 척도의 값 β", ylabel="e^β",
          xticks=[-1.386, -0.693, 0, 0.693, 1.386], xtickfmt=lambda v: {-1.386: "−1.386", -0.693: "−0.693", 0: "0", 0.693: "0.693", 1.386: "1.386"}[round(v, 3)],
          yticks=[0, 0.5, 1, 2, 3, 4, 5], xgrid=True)
pE.line(xs, np.exp(xs), s=1, w=2.4)
for bv in (-1.386, -0.693, 0, 0.693, 1.386):
    ev = math.exp(bv)
    pE.seg(bv, 0, bv, ev, cls="ref", dash=True)
    pE.seg(-1.6, ev, bv, ev, cls="ref", dash=True)
    pE.points([bv], [ev], s=2, r=4.5)
for bv, lab, dx_, dy_ in ((-1.386, "0.25", 4, -10), (-0.693, "0.5", 0, -12), (0, "1", -6, -10), (0.693, "2", -8, -6), (1.386, "4", -10, -4)):
    pE.text(bv, math.exp(bv), lab, anchor="end", dx=dx_, dy=dy_, cls="lbl strong")
pE.text(0.35, 3.3, "β에 0.693을 더할 때마다", anchor="middle", cls="lbl small")
pE.text(0.35, 3.3, "e^β는 2배", anchor="middle", dy=16, cls="lbl small")
panel_title(pE, "가. e^β 곡선")
# panel B: two number lines
pN = Plot((-1.8, 4.6), (0, 1), w=420, h=320, ml=16, mr=16, mt=26, mb=30, show_yaxis=False, ygrid=False, show_xaxis=False)
yb, yr = 0.80, 0.34
pN.seg(-1.8, yb, 4.6, yb, cls="axis", w=1)
pN.seg(-1.8, yr, 4.6, yr, cls="axis", w=1)
bmap = lambda b: -1.6 + (b + 1.6) / 3.2 * 6.0
rmap = lambda r: -1.6 + r / 4.4 * 6.0
for bv, rv, lab_b, lab_r, dyr in ((-1.386, 0.25, "−1.386", "0.25", 34), (-0.693, 0.5, "−0.693", "0.5", 18), (0, 1, "0", "1", 18),
                                   (0.693, 2, "0.693", "2", 18), (1.386, 4, "1.386", "4", 18)):
    pN.seg(bmap(bv), yb - 0.03, bmap(bv), yb + 0.03, cls="axis")
    pN.seg(rmap(rv), yr - 0.03, rmap(rv), yr + 0.03, cls="axis")
    pN.seg(bmap(bv), yb - 0.04, rmap(rv), yr + 0.04, cls="ln s4", w=1.2, dash=True)
    pN.text(bmap(bv), yb, lab_b, anchor="middle", dy=-10, cls="lbl small")
    pN.text(rmap(rv), yr, lab_r, anchor="middle", dy=dyr, cls="lbl strong")
pN.seg(rmap(0.25), yr - 0.10, rmap(0.25), yr - 0.045, cls="ref", w=1)
pN.text(-1.8, yb, "로그 척도 (β, ln OR)", anchor="start", dy=-32, cls="lbl strong")
pN.text(-1.8, yr, "비 척도 (e^β, OR)", anchor="start", dy=58, cls="lbl strong")
pN.text(rmap(0.5), 0.02, "1 아래: 0~1 사이에 압축", anchor="start", cls="lbl mute small")
pN.text(rmap(2.4), 0.02, "1 위: 1~∞로 펼쳐짐", anchor="start", cls="lbl mute small")
panel_title(pN, "나. 로그 척도의 등간격 = 비 척도의 같은 배수")
save("ch08_exp", figure([pE.svg("지수함수 곡선"), pN.svg("로그 척도와 비 척도의 대응")],
     "그림 8-5. 회귀계수 β는 로그 척도에 있고, e^β로 바꾸면 비(ratio)가 됩니다. 로그 척도에서 같은 거리(±0.693)는 비 척도에서 같은 배수(×2, ÷2)가 됩니다. "
     "그래서 OR 2와 OR 0.5는 '1에서 같은 크기만큼 떨어진' 효과이며, 일반 눈금에서 1 아래 구간이 좁아 보이는 것은 착시입니다.", cols=2))

# ------------------------------------------------------------------ 8-6 link functions
eta = np.linspace(-6, 6, 300)
pl = Plot((-6, 6), (-0.05, 1.15), w=420, h=300, ml=50, mr=16, xlabel="선형예측값 η = β₀ + β₁x₁ + …", ylabel="평균 μ (사건 확률)",
          xticks=[-6, -4, -2, 0, 2, 4, 6], yticks=[0, 0.25, 0.5, 0.75, 1.0])
pl.hline(1.0, cls="ref", dash=True); pl.hline(0.0, cls="ref", dash=True)
pl.line(eta, 1 / (1 + np.exp(-eta)), s=1, w=2.6)
pl.points([0], [0.5], s=2, r=4.5)
pl.text(0, 0.5, "η = 0 → μ = 0.5", anchor="start", dx=10, dy=4, cls="lbl strong")
pl.text(-5.8, 1.0, "μ는 항상 0과 1 사이", anchor="start", dy=-8, cls="lbl mute small")
panel_title(pl, "가. 로짓 연결: μ = 1 / (1 + e^(−η))")
eta2 = np.linspace(-4, 1.2, 300)
pg = Plot((-4, 1.2), (-0.05, 3.4), w=420, h=300, ml=50, mr=16, xlabel="선형예측값 η", ylabel="평균 μ",
          xticks=[-4, -3, -2, -1, 0, 1], yticks=[0, 0.5, 1, 1.5, 2, 2.5, 3])
pg.els.append(f'<rect x="{pg.sx(-4):.1f}" y="{pg.sy(3.4):.1f}" width="{pg.sx(1.2) - pg.sx(-4):.1f}" height="{pg.sy(1) - pg.sy(3.4):.1f}" class="a2"/>')
pg.hline(1.0, cls="ref", dash=True)
pg.line(eta2, np.exp(eta2), s=1, w=2.6)
pg.points([0], [1], s=2, r=4.5)
pg.text(-3.9, 2.9, "μ > 1: 확률로는 불가능한 영역", anchor="start", cls="lbl small")
pg.text(-3.9, 2.9, "(발생률·비용에는 문제 없음)", anchor="start", dy=16, cls="lbl mute small")
pg.text(0, 1, "η = 0 → μ = 1", anchor="end", dx=-10, dy=-8, cls="lbl strong")
panel_title(pg, "나. 로그 연결: μ = e^η")
save("ch08_links", figure([pl.svg("로짓 연결함수"), pg.svg("로그 연결함수")],
     "그림 8-6. 연결함수는 −∞~∞ 범위의 선형예측값 η를 결과변수 평균 μ의 범위로 옮깁니다. 가: 로짓 연결은 어떤 η든 0과 1 사이의 확률로 바꿉니다. "
     "나: 로그 연결은 0보다 큰 값을 만들지만 상한이 없어, 이분형 결과에 쓰면(로그-이항 모형) η가 0을 넘는 순간 확률이 1을 넘게 되어 추정이 수렴하지 않기도 합니다."
     , cols=2))

# ------------------------------------------------------------------ 8-7 likelihood for 7/50
lk = R["lik"]
pv = np.linspace(0.005, 0.40, 400)
llv = 7 * np.log(pv) + 43 * np.log(1 - pv)
rel = np.exp(llv - lk["llmax"])
pa = Plot((0, 0.4), (0, 1.1), w=420, h=300, ml=50, mr=16, xlabel="이상반응 발생 확률 p", ylabel="상대 우도 L(p) / L(0.14)",
          xticks=[0, 0.1, 0.2, 0.3, 0.4], xtickfmt=lambda v: fmt(round(v, 2)), yticks=[0, 0.25, 0.5, 0.75, 1.0])
inside = (pv >= lk["lr_lo"]) & (pv <= lk["lr_hi"])
pa.fill_between(pv[inside], np.zeros(inside.sum()), rel[inside], s=1)
pa.line(pv, rel, s=1, w=2.4)
pa.vline(0.14, cls="strongref", dash=True, y1=1.0)
pa.text(0.14, 1.0, "최대우도추정값 p̂ = 7/50 = 0.14", anchor="start", dx=6, dy=-6, cls="lbl strong")
for q, lab in ((0.05, "p = 0.05: 0.054"), (0.30, "p = 0.30: 0.030")):
    rq = math.exp(7 * math.log(q) + 43 * math.log(1 - q) - lk["llmax"])
    pa.points([q], [rq], s=2, r=4)
pa.text(0.05, 0.054, "0.05에서 0.054", anchor="start", dx=-4, dy=-12, cls="lbl small")
pa.text(0.30, 0.03, "0.30에서 0.030", anchor="start", dx=2, dy=-12, cls="lbl small")
halo(pa, 0.155, 0.35, f"95% 구간 {lk['lr_lo']:.3f}–{lk['lr_hi']:.3f}", anchor="middle", cls="lbl small", fs=11.5)
panel_title(pa, "가. 우도 (최댓값을 1로)")
pb = Plot((0, 0.4), (-6, 0.6), w=420, h=300, ml=50, mr=16, xlabel="이상반응 발생 확률 p", ylabel="로그우도 ln L(p) − ln L(0.14)",
          xticks=[0, 0.1, 0.2, 0.3, 0.4], xtickfmt=lambda v: fmt(round(v, 2)), yticks=[-6, -5, -4, -3, -2, -1, 0])
mk_ = (llv - lk["llmax"]) >= -6
pb.line(pv[mk_], (llv - lk["llmax"])[mk_], s=1, w=2.4)
pb.hline(-lk["drop"], cls="ln s2", dash=True, w=1.4)
pb.seg(lk["lr_lo"], -lk["drop"], lk["lr_lo"], -6, cls="ref", dash=True)
pb.seg(lk["lr_hi"], -lk["drop"], lk["lr_hi"], -6, cls="ref", dash=True)
pb.points([lk["lr_lo"], lk["lr_hi"]], [-lk["drop"]] * 2, s=2, r=4)
pb.text(0.395, -lk["drop"], "최댓값보다 1.92 낮은 선", anchor="end", dy=20, cls="lbl small")
pb.text(lk["lr_lo"], -5.2, f"{lk['lr_lo']:.3f}", anchor="end", dx=-4, cls="lbl strong")
pb.text(lk["lr_hi"], -5.2, f"{lk['lr_hi']:.3f}", anchor="start", dx=4, cls="lbl strong")
panel_title(pb, "나. 로그우도와 우도비 신뢰구간")
save("ch08_lik", figure([pa.svg("우도 곡선"), pb.svg("로그우도 곡선")],
     f"그림 8-7. 환자 50명 중 7명에게 이상반응이 생겼을 때 발생 확률 p의 우도. 가: p = 0.14에서 우도가 가장 크고, p = 0.05와 0.30에서는 각각 최댓값의 5.4%와 3.0%에 불과합니다. "
     f"나: 로그우도가 최댓값보다 1.92(= χ²₁ 임계값 3.84 ÷ 2) 이상 떨어지지 않는 범위가 95% 우도비 신뢰구간({lk['lr_lo']:.3f}–{lk['lr_hi']:.3f})입니다. "
     "곡선이 비대칭이라 구간도 0.14를 중심으로 비대칭입니다.", cols=2))

# ------------------------------------------------------------------ 8-8 Wald / LR / score
T = R["tests"]
pv = np.linspace(0.02, 0.36, 500)
ll_rel = 7 * np.log(pv) + 43 * np.log(1 - pv) - lk["llmax"]
ph, p0 = 0.14, 0.05
wald_par = -0.5 * T["info_hat"] * (pv - ph) ** 2
U0, I0, l0 = T["score_slope"], T["info0"], T["ll0"] - lk["llmax"]
score_par = l0 + U0 * (pv - p0) - 0.5 * I0 * (pv - p0) ** 2
p = Plot((0.02, 0.36), (-4.5, 1.7), w=600, h=380, ml=56, mr=24, xlabel="이상반응 발생 확률 p",
         ylabel="로그우도 (최댓값 = 0)", xticks=[0.05, 0.10, 0.14, 0.20, 0.25, 0.30, 0.35], xtickfmt=lambda v: fmt(round(v, 2)),
         yticks=[-4, -3, -2, -1, 0, 1])
mk_ = ll_rel >= -4.5
p.line(pv[mk_], ll_rel[mk_], s=1, w=2.8)
mk_ = wald_par >= -4.5
p.line(pv[mk_], wald_par[mk_], s=2, w=1.8, dash=True)
msk = score_par > -4.5
p.line(pv[msk], score_par[msk], s=3, w=1.8, dash=True)
p.vline(p0, cls="ref", dash=True); p.vline(ph, cls="ref", dash=True)
p.points([p0], [l0], s=1, r=5)
p.points([ph], [0], s=1, r=5)
p.points([p0], [-0.5 * T["info_hat"] * (p0 - ph) ** 2], s=2, r=4.5)
smax = U0 ** 2 / (2 * I0)
psm = p0 + U0 / I0
p.points([psm], [l0 + smax], s=3, r=4.5)
# annotations
p.seg(p0 - 0.004, l0, p0 - 0.004, 0, cls="ln s1", w=1.4)
p.seg(p0 - 0.006, 0, ph, 0, cls="ref", dash=True, w=1)
halo(p, 0.058, -3.75, f"우도비: 2 × {-l0:.2f} = {T['lr']:.2f}", anchor="start", cls="lbl strong", fs=13)
halo(p, ph + 0.012, -0.5 * T["info_hat"] * (p0 - ph) ** 2 - 1.0, f"Wald: 2 × {0.5 * T['info_hat'] * (p0 - ph) ** 2:.2f} = {T['wz'] ** 2:.2f}", anchor="start", cls="lbl", fs=13)
halo(p, psm, l0 + smax, f"점수: 2 × {smax:.2f} = {T['sz'] ** 2:.2f}", anchor="middle", dy=-12, cls="lbl", fs=13)
p.text(p0, -4.5, "귀무가설 p₀ = 0.05", anchor="start", dx=4, dy=-8, cls="lbl mute small")
p.text(ph, -4.5, "p̂ = 0.14", anchor="start", dx=4, dy=-8, cls="lbl mute small")
p.legend([("실제 로그우도 (우도비 검정)", 1, "line"), ("p̂에서 맞춘 포물선 (Wald)", 2, "dash"), ("p₀에서 맞춘 포물선 (점수)", 3, "dash")], X=p.sx(0.205), Y=p.mt + 14)
save("ch08_tests", figure(p.svg("Wald, 우도비, 점수 검정의 비교"),
     f"그림 8-8. 50명 중 7명(p̂ = 0.14)에서 귀무가설 p₀ = 0.05를 검정하는 세 방법. 세 검정 모두 '로그우도의 높이 차이 × 2'를 χ²₁과 비교하지만, "
     f"우도비 검정은 실제 곡선을, Wald 검정은 p̂ 근처의 곡률로 그린 포물선을, 점수 검정은 p₀에서의 기울기와 그 지점의 정보량(기대 곡률)으로 그린 포물선을 씁니다. "
     f"표본이 작고 곡선이 비대칭이라 χ²가 {T['wz'] ** 2:.2f}(P = {T['p_w']:.3f}), {T['lr']:.2f}(P = {T['p_lr']:.3f}), {T['sz'] ** 2:.2f}(P = {T['p_s']:.4f})로 크게 다릅니다."))


# ------------------------------------------------------------------ DAG helpers
def node(p, X, Y, label, cls="s4", sub=None, w=None, dash=False, fs=13):
    w = w or (_tw(label, fs) + 24)
    h = 34 if not sub else 46
    da = ' stroke-dasharray="5 4"' if dash else ""
    p.els.append(f'<rect x="{X - w / 2:.1f}" y="{Y - h / 2:.1f}" width="{w:.1f}" height="{h}" rx="8" class="pth {cls}"{da}/>')
    if sub:
        p.text_px(X, Y - 3, label, anchor="middle", cls="lbl strong")
        p.text_px(X, Y + 13, sub, anchor="middle", cls="lbl mute small")
    else:
        p.text_px(X, Y + 4.5, label, anchor="middle", cls="lbl strong")
    return (X, Y, w, h)


def arrow(p, a, b, cls="s4", dash=False, w=1.6):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = a, b

    def clip(xc, yc, wc, hc, dx, dy):
        tx = (wc / 2 + 3) / abs(dx) if dx else 1e9
        ty = (hc / 2 + 3) / abs(dy) if dy else 1e9
        t = min(tx, ty)
        return xc + dx * t, yc + dy * t
    dx, dy = x2 - x1, y2 - y1
    sx, sy = clip(x1, y1, w1, h1, dx, dy)
    ex, ey = clip(x2, y2, w2, h2, -dx, -dy)
    ln = math.hypot(ex - sx, ey - sy)
    ux, uy = (ex - sx) / ln, (ey - sy) / ln
    hx, hy = ex - ux * 9, ey - uy * 9
    da = ' stroke-dasharray="6 4"' if dash else ""
    p.els.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" class="ln {cls}" stroke-width="{w}"{da}/>')
    px, py = -uy, ux
    pts = f"{ex:.1f},{ey:.1f} {hx + px * 5:.1f},{hy + py * 5:.1f} {hx - px * 5:.1f},{hy - py * 5:.1f}"
    p.els.append(f'<polygon points="{pts}" class="f{cls[1:]}"/>')


def canvas(w, h):
    return Plot((0, 1), (0, 1), w=w, h=h, ml=0, mr=0, mt=0, mb=0, show_xaxis=False, show_yaxis=False, ygrid=False)


# ------------------------------------------------------------------ 8-9 three DAGs
W, H = 330, 230
c1 = canvas(W, H)
C = node(c1, 165, 50, "질병 중증도", cls="s2")
E = node(c1, 70, 150, "약물 A 처방")
Y = node(c1, 260, 150, "입원")
arrow(c1, C, E, cls="s2"); arrow(c1, C, Y, cls="s2"); arrow(c1, E, Y, cls="s1", dash=True)
c1.text_px(165, 200, "E ← C → Y: 뒷문 경로", anchor="middle", cls="lbl small")
c1.text_px(165, 218, "C를 보정해 경로를 막음", anchor="middle", cls="lbl strong")
c1.text_px(10, 18, "가. 교란변수", cls="ptitle")
c2 = canvas(W, H)
E2 = node(c2, 60, 110, "스타틴")
M2 = node(c2, 165, 50, "LDL 감소", cls="s3")
Y2 = node(c2, 275, 110, "심근경색")
arrow(c2, E2, M2, cls="s3"); arrow(c2, M2, Y2, cls="s3"); arrow(c2, E2, Y2, cls="s1", dash=True)
c2.text_px(165, 200, "E → M → Y: 효과가 지나가는 길", anchor="middle", cls="lbl small")
c2.text_px(165, 218, "총효과를 보려면 M을 보정하지 않음", anchor="middle", cls="lbl strong")
c2.text_px(10, 18, "나. 매개변수", cls="ptitle")
c3 = canvas(W, H)
E3 = node(c3, 70, 60, "약물 A 사용")
Y3 = node(c3, 260, 60, "질환 X")
H3 = node(c3, 165, 150, "입원", cls="s2", dash=True)
arrow(c3, E3, H3, cls="s4"); arrow(c3, Y3, H3, cls="s4")
c3.text_px(165, 200, "E → H ← Y: 원래 막혀 있는 경로", anchor="middle", cls="lbl small")
c3.text_px(165, 218, "H로 보정·제한하면 경로가 열림", anchor="middle", cls="lbl strong")
c3.text_px(10, 18, "다. 충돌변수", cls="ptitle")
save("ch08_dag", figure([c1.svg("교란변수 DAG"), c2.svg("매개변수 DAG"), c3.svg("충돌변수 DAG")],
     "그림 8-10. 인과 도표(DAG)로 본 세 가지 변수. 화살표는 '원인 → 결과'입니다. 가: 중증도가 약물 선택과 입원 모두의 원인이면 교란변수이며 보정해야 합니다. "
     "나: LDL 감소는 스타틴 효과가 전달되는 통로(매개변수)이므로 보정하면 효과의 일부를 지웁니다. "
     "다: 입원은 약물 사용과 질환 X의 공통 결과(충돌변수)이므로, 입원 환자만 분석하면 원래 없던 연관이 생깁니다.", cols=3))

# ------------------------------------------------------------------ 8-10 Simpson-type reversal
cf = R["conf"]
groups = [("중증 환자", cf["severe"]), ("경증 환자", cf["mild"]), ("전체 (보정 전)", cf["crude"])]
p = Plot((0.4, 3.6), (0, 0.5), w=600, h=336, ml=56, mr=20, mb=70, ylabel="1년 내 입원 위험",
         yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], ytickfmt=lambda v: pct(v),
         xticks=[1, 2, 3], xticklabels=[(i + 1, g[0]) for i, g in enumerate(groups)])
for i, (g, v) in enumerate(groups):
    xa, xb = i + 1 - 0.18, i + 1 + 0.18
    p.bars([xa], [v["r1"]], 0.32, s=1)
    p.bars([xb], [v["r0"]], 0.32, s=2)
    p.text(xa, v["r1"], f"{v['r1'] * 100:.1f}%", anchor="middle", dy=-6, cls="lbl strong")
    p.text(xb, v["r0"], f"{v['r0'] * 100:.1f}%", anchor="middle", dy=-6, cls="lbl strong")
    p.text(i + 1, 0, f"A {v['a']}/{v['n1']} · B {v['c']}/{v['n0']}", anchor="middle", dy=38, cls="lbl mute small")
    rrv = v["rr"][0]
    p.text(i + 1, max(v["r1"], v["r0"]), f"RR {rrv:.2f}", anchor="middle", dy=-28, cls="lbl")
p.vline(2.5, cls="ref", dash=True)
p.legend([("약물 A (새 약)", 1, "box"), ("약물 B (기존 약)", 2, "box")], X=p.sx(2.55), Y=p.mt + 10)
save("ch08_simpson", figure(p.svg("중증도로 층화한 입원 위험"),
     f"그림 8-9. 적응증에 의한 교란의 가상 예. 중증 환자와 경증 환자 각각에서는 약물 A의 입원 위험이 B의 0.80배로 낮습니다. "
     f"그런데 약물 A는 중증 환자에게 주로 처방되어(A군의 75%, B군의 25%가 중증) 두 층을 합치면 A군의 위험이 26.0%로 B군(17.5%)보다 높아 보입니다(RR {cf['crude']['rr'][0]:.2f})."))

# ------------------------------------------------------------------ 8-11 effect modification depends on scale
def scale_panel(logy):
    if logy:
        pp = Plot((0.6, 2.4), (L(0.05), L(0.5)), w=420, h=300, ml=54, mr=90, ylabel="입원 위험 (로그 눈금)",
                  yticks=[L(v) for v in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)], ytickfmt=lambda v: pct(math.exp(v)),
                  xticks=[1, 2], xticklabels=[(1, "약물 B"), (2, "약물 A")])
        tf = L
    else:
        pp = Plot((0.6, 2.4), (0, 0.45), w=420, h=300, ml=54, mr=90, ylabel="입원 위험 (일반 눈금)",
                  yticks=[0, 0.1, 0.2, 0.3, 0.4], ytickfmt=lambda v: pct(v), xticks=[1, 2], xticklabels=[(1, "약물 B"), (2, "약물 A")])
        tf = lambda v: v
    for key, s, lab in (("severe", 2, "중증"), ("mild", 1, "경증")):
        v = cf[key]
        pp.line([1, 2], [tf(v["r0"]), tf(v["r1"])], s=s, w=2.4)
        pp.points([1, 2], [tf(v["r0"]), tf(v["r1"])], s=s, r=5)
        pp.text(1, tf(v["r0"]), f"{v['r0'] * 100:.0f}%", anchor="end", dx=-10, dy=4, cls="lbl")
        pp.text(2, tf(v["r1"]), f"{v['r1'] * 100:.0f}%", anchor="start", dx=10, dy=4, cls="lbl")
        eff = f"RR {v['rr'][0]:.2f}" if logy else f"RD −{abs(v['rd'][0]) * 100:.0f}%p"
        pp.text_px(pp.w - pp.mr + 8, pp.sy(tf(v["r1"])) - 8, lab, cls="lbl strong")
        pp.text_px(pp.w - pp.mr + 8, pp.sy(tf(v["r1"])) + 8, eff, cls="lbl small")
    return pp


pl_ = scale_panel(False); panel_title(pl_, "가. 위험차: 층마다 다름 (−8 vs −2%p)")
pg_ = scale_panel(True); panel_title(pg_, "나. 상대위험도: 층마다 같음 (0.80)")
save("ch08_scale", figure([pl_.svg("일반 눈금의 위험"), pg_.svg("로그 눈금의 위험")],
     "그림 8-11. 그림 8-9의 층별 위험을 두 척도로 그렸습니다. 로그 눈금(나)에서 두 선이 평행하므로 곱셈 척도의 효과수정은 없습니다(RR이 두 층에서 모두 0.80). "
     "일반 눈금(가)에서는 기울기가 달라, 중증 환자에서 100명당 8명, 경증 환자에서 100명당 2명이 줄어 덧셈 척도의 효과수정이 있습니다. "
     "'교호작용이 있다/없다'는 어느 척도에서 본 것인지와 함께 말해야 합니다.", cols=2))


# ------------------------------------------------------------------ 8-12 subgroup forest with P for interaction
def badge(X, Y, n):
    return (f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="9" class="f2"/>'
            f'<text x="{X:.1f}" y="{Y + 4:.1f}" text-anchor="middle" style="fill:#fff;font-size:11px;font-weight:600">{n}</text>')


def forest2(rows, xlim=(0.3, 2.0), w=860, row_h=28, xticks=(0.3, 0.5, 0.75, 1, 1.5, 2), badges=None):
    top = 40
    h = top + row_h * len(rows) + 62
    cL, cX, cP = 196, 290, 384        # n/N columns end (right aligned)
    X0, X1 = 400, 620
    cE, cI = 760, w - 6
    lo_t, hi_t = L(xlim[0]), L(xlim[1])
    sx = lambda v: X0 + (L(v) - lo_t) / (hi_t - lo_t) * (X1 - X0)
    o = []
    hdr = [(6, "start", "Subgroup"), (cX - 10, "end", "Drug X"), (cP - 10, "end", "Placebo"),
           ((X0 + X1) / 2, "middle", "Risk ratio (95% CI)"), (cE, "end", "RR (95% CI)"), (cI, "end", "P for")]
    for x_, an, t in hdr:
        o.append(f'<text x="{x_:.1f}" y="18" text-anchor="{an}" class="axlab" font-weight="600">{escape(t)}</text>')
    o.append(f'<text x="{cI:.1f}" y="33" text-anchor="end" class="axlab" font-weight="600">interaction</text>')
    o.append(f'<text x="{cX - 10:.1f}" y="33" text-anchor="end" class="lbl mute small">events/N</text>')
    o.append(f'<text x="{cP - 10:.1f}" y="33" text-anchor="end" class="lbl mute small">events/N</text>')
    yb = top + row_h * len(rows) + 4
    for t in xticks:
        x_ = sx(t)
        o.append(f'<line x1="{x_:.1f}" y1="{top - 2}" x2="{x_:.1f}" y2="{yb}" class="grid"/>')
        o.append(f'<text x="{x_:.1f}" y="{yb + 17}" text-anchor="middle" class="tick">{fmt(t)}</text>')
    o.append(f'<line x1="{X0}" y1="{yb}" x2="{X1}" y2="{yb}" class="axis"/>')
    o.append(f'<line x1="{sx(1):.1f}" y1="{top - 2}" x2="{sx(1):.1f}" y2="{yb}" class="ref" stroke-width="1.4"/>')
    o.append(f'<text x="{sx(1) - 8:.1f}" y="{yb + 36}" text-anchor="end" class="lbl small">← Drug X better</text>')
    o.append(f'<text x="{sx(1) + 8:.1f}" y="{yb + 36}" class="lbl small">Placebo better →</text>')
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        bold = ' font-weight="600"' if r.get("head") else ""
        ind = 20 if r.get("ind") else 6
        o.append(f'<text x="{ind}" y="{y + 4.5:.1f}" class="lbl"{bold}>{r["label"]}</text>')
        if r.get("pint"):
            o.append(f'<text x="{cI}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num"{bold}>{r["pint"]}</text>')
        if r.get("est") is None:
            continue
        o.append(f'<text x="{cX - 10:.1f}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num">{r["n1"]}</text>')
        o.append(f'<text x="{cP - 10:.1f}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num">{r["n0"]}</text>')
        a, b = sx(max(r["lo"], xlim[0])), sx(min(r["hi"], xlim[1]))
        s = r.get("s", 1)
        o.append(f'<line x1="{a:.1f}" y1="{y:.1f}" x2="{b:.1f}" y2="{y:.1f}" class="ln s{s}" stroke-width="2"/>')
        xe = sx(r["est"])
        if r.get("diamond"):
            o.append(f'<polygon points="{a:.1f},{y:.1f} {xe:.1f},{y - 8:.1f} {b:.1f},{y:.1f} {xe:.1f},{y + 8:.1f}" class="f{s}"/>')
        else:
            sz = r.get("size", 5)
            o.append(f'<rect x="{xe - sz:.1f}" y="{y - sz:.1f}" width="{2 * sz:.1f}" height="{2 * sz:.1f}" class="f{s}"/>')
        o.append(f'<text x="{cE}" y="{y + 4.5:.1f}" text-anchor="end" class="lbl num"{bold}>{r["est"]:.2f} ({r["lo"]:.2f}–{r["hi"]:.2f})</text>')
    ypos = lambda i: top + row_h * i + row_h / 2
    for spec in (badges or []):
        kind, i, n = spec
        if kind == "label":
            o.append(badge(spec_x(rows[i]), ypos(i), n))
        elif kind == "pint":
            o.append(badge(cI - 7.6 * len(rows[i]["pint"]) - 14, ypos(i), n))
        elif kind == "hdr":
            o.append(badge(cX - 88, 26, n))
        elif kind == "axis":
            o.append(badge(sx(1) - 128, yb + 32, n))
    return (f'<svg viewBox="0 0 {w} {h}" class="viz" role="img" aria-label="subgroup forest plot" '
            f'xmlns="http://www.w3.org/2000/svg">{"".join(o)}</svg>')


def spec_x(r):
    from html import unescape
    return (20 if r.get("ind") else 6) + _tw(unescape(r["label"]), 13) * (1.08 if r.get("head") else 1.0) + 16


rows = []
ov = R["sg_overall"]["rr"]
rows.append(dict(label="Overall", head=True, est=ov[0], lo=ov[1], hi=ov[2], n1="150/1000", n0="200/1000", diamond=True, s=4))
gl = {"Age": "Age", "Sex": "Sex", "eGFR": "eGFR, mL/min/1.73 m²", "Diabetes": "Diabetes"}
for sg in R["sg"]:
    pint = f"{sg['pint']:.2f}" if sg["pint"] >= 0.01 else f"{sg['pint']:.3f}"
    rows.append(dict(label=gl[sg["g"]], head=True, pint=pint))
    for lab, t, r_ in ((sg["l1"], sg["t1"], sg["rr1"]), (sg["l2"], sg["t2"], sg["rr2"])):
        lab_ = lab.replace(" mL/min/1.73 m²", "").replace("&lt;", "<")
        sz = 3 + 5 * math.sqrt((t[1] + t[3]) / 2000)
        rows.append(dict(label=escape(lab_), ind=True, est=r_[0], lo=r_[1], hi=r_[2], n1=f"{t[0]}/{t[1]}", n0=f"{t[2]}/{t[3]}", size=sz))
save("ch08_forest", figure(forest2(rows, badges=[("label", 0, 1), ("pint", 1, 2), ("hdr", 0, 4), ("pint", 7, 3), ("axis", 0, 5)]),
     "그림 8-12. 가상의 무작위배정 임상시험(약물 X 대 위약, 1년 내 입원)의 하위군 분석 포레스트 플롯. 네모의 크기는 하위군의 인원에 비례하고, 가로축은 로그 눈금입니다. 번호는 아래 해설과 짝을 이룹니다."))

# ------------------------------------------------------------------ 8-13 CCI: dummies vs linear trend
rows_d = [r for r in R["full_rows"] if r["name"].startswith("cci")]
lin = [r for r in R["cci_lin"]["rows"] if r["name"] == "cci_score"][0]
xs = [0, 1, 2, 3]
p = Plot((-0.5, 3.5), (L(0.7), L(5)), w=600, h=320, ml=60, mr=24, ylabel="보정 오즈비 (로그 눈금, CCI 0 = 1)",
         xlabel="Charlson 동반질환지수(CCI) 범주", xticks=xs, xticklabels=list(zip(xs, ["0 (기준범주)", "1–2", "3–4", "≥5"])),
         yticks=[L(v) for v in (0.75, 1, 1.5, 2, 3, 4, 5)], ytickfmt=lambda v: fmt(round(math.exp(v), 2)))
p.hline(0, cls="ref", dash=False, w=1)
fitted = [lin["b"] * k for k in xs]
p.line(xs, fitted, s=2, w=2, dash=True)
p.points([0], [0], s=1, r=5.5, hollow=True)
for k, r in zip([1, 2, 3], rows_d):
    p.seg(k, r["lo"], k, r["hi"], cls="ln s1", w=2)
    p.points([k], [r["b"]], s=1, r=5.5)
    p.text(k, r["b"], f"{math.exp(r['b']):.2f}", anchor="start", dx=9, dy=4, cls="lbl strong")
p.text(0, 0, "1 (기준)", anchor="start", dx=4, dy=20, cls="lbl strong")
p.legend([("가변수 3개로 추정한 OR (95% CI)", 1, "dot"), (f"선형 점수(0–3) 모형: 한 단계당 OR {math.exp(lin['b']):.2f}", 2, "dash")],
         X=p.ml + 12, Y=p.mt + 10)
save("ch08_dummy", figure(p.svg("CCI 범주별 보정 오즈비"),
     f"그림 8-13. 재입원 코호트에서 CCI 범주를 가변수로 넣은 보정 오즈비(파랑)와, 범주 번호 0–3을 연속형 점수로 넣었을 때의 추정값(주황 점선: 1, {math.exp(lin['b']):.2f}, "
     f"{math.exp(2 * lin['b']):.2f}, {math.exp(3 * lin['b']):.2f}). 이 자료에서는 두 방식이 거의 같아 선형 추세 가정이 잘 맞습니다(선형성에 대한 우도비 검정 P &gt; 0.99)."))

# ------------------------------------------------------------------ 8-14 DAG for the readmission cohort
c = canvas(640, 320)
Ea = node(c, 230, 200, "다제약물 (≥10개)", cls="s1")
Ya = node(c, 540, 200, "30일 재입원", cls="s1")
Ma = node(c, 70, 50, "의료급여", cls="s2")
Ag = node(c, 225, 50, "나이", cls="s2")
Cc = node(c, 385, 50, "동반질환 (CCI)", cls="s2")
Fe = node(c, 565, 50, "성별", cls="s2")
Md = node(c, 385, 285, "퇴원 후 약물 이상반응", cls="s3", dash=True)
arrow(c, Ag, Cc, cls="s4"); arrow(c, Ag, Ea, cls="s4"); arrow(c, Cc, Ea, cls="s4"); arrow(c, Ma, Ea, cls="s4")
arrow(c, Ag, Ya, cls="s4"); arrow(c, Cc, Ya, cls="s4"); arrow(c, Ma, Ya, cls="s4"); arrow(c, Fe, Ya, cls="s4")
arrow(c, Fe, Ea, cls="s4", dash=True, w=1.2)
arrow(c, Ea, Ya, cls="s1", w=2.2)
arrow(c, Ea, Md, cls="s3"); arrow(c, Md, Ya, cls="s3")
c.text_px(10, 118, "교란변수", cls="lbl strong")
c.text_px(10, 134, "노출 이전에 정해진", cls="lbl small")
c.text_px(10, 149, "공통 원인 → 보정", cls="lbl small")
c.text_px(10, 272, "매개변수", cls="lbl strong")
c.text_px(10, 288, "노출 이후에 생긴 변수", cls="lbl small")
c.text_px(10, 303, "→ 총효과 추정에서는 보정하지 않음", cls="lbl small")
save("ch08_dag2", figure(c.svg("재입원 코호트의 인과 도표"),
     "그림 8-14. 재입원 코호트에 대해 가정한 인과 도표. 나이, 동반질환, 의료급여, 성별은 퇴원 시점 이전에 정해진 변수로, 다제약물 처방과 재입원 모두에 영향을 주는 교란변수입니다"
     "(성별과 다제약물의 관련은 약해서 점선으로 그렸습니다). 퇴원 후 약물 이상반응은 다제약물이 재입원으로 이어지는 경로의 매개변수이므로 총효과를 추정할 때는 보정하지 않습니다."))
print("figures written")
