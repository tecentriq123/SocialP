"""Figures for chapter 14 (청구자료 코호트 연구의 설계).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch14.py && python3 gen/fig_ch14.py
Reads gen/_ch14_nums.json written by nums_ch14.py."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from svgplot import Plot, figure, forest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs")
N = json.load(open(os.path.join(HERE, "_ch14_nums.json")))


def save(name, html):
    with open(os.path.join(OUT, name + ".html"), "w") as f:
        f.write(html)


def bar(p, a, b, y, hh=0.26, cls="f1", extra=""):
    X0, X1 = p.sx(a), p.sx(b)
    p.els.append(f'<rect x="{X0:.1f}" y="{p.sy(y + hh):.1f}" width="{max(X1 - X0, 1.5):.1f}" '
                 f'height="{p.sy(y - hh) - p.sy(y + hh):.1f}" rx="3" class="{cls}"{extra}/>')


ex = N["ex"]
fills = ex["fills_day"]            # [(day, supply)]
hhf = ex["hhf_day"]                # 475

# ====================================================================== 14-1 what a claims record keeps
p = Plot((-25, 520), (0.2, 4.7), w=640, h=330, ml=150, mr=16, mt=14, mb=48, xlabel="첫 처방일(2021-03-02) 이후 일수",
         xticks=[0, 90, 180, 270, 360, 450], yticks=[], ygrid=False, show_yaxis=False)
labs = [(4.0, "외래 명세서", "상병코드 E11, I10"), (3.0, "원외처방 내역", "주성분코드, 공급일수"),
        (2.0, "입원 명세서", "주상병 I50"), (1.0, "자료에 없는 것", "")]
for y, a, b in labs:
    p.text_px(6, p.sy(y) + (-2 if b else 4), a, cls="lbl strong small")
    if b:
        p.text_px(6, p.sy(y) + 13, b, cls="lbl mute small")
    p.hline(y, cls="grid", dash=False, x0=-25, x1=520, w=1)
p.points([d for d, _ in fills], [4.0] * len(fills), s=1, r=4.5)
for d, s in fills:
    bar(p, d, d + s, 3.0)
p.text(fills[0][0] + 15, 3.0, "30일분", anchor="middle", dy=-20, cls="lbl mute small")
p.text(fills[3][0] + 45, 3.0, "90일분", anchor="middle", dy=-20, cls="lbl mute small")
p.text(207, 3.0, "45일 공백", anchor="middle", dy=28, cls="lbl mute small")
bar(p, hhf, hhf + 8, 2.0, cls="f2")
p.text(hhf, 2.0, "심부전 입원", anchor="end", dx=-8, dy=4, cls="lbl small")
for x, t, anc in ((0, "검사 결과값", "start"), (230, "실제 복용 여부, 일반약", "middle"),
                  (505, "흡연·체중, 병원 밖 사망", "end")):
    X = p.sx(x)
    p.els.append(f'<circle cx="{X:.1f}" cy="{p.sy(1.0):.1f}" r="4.5" class="pth s4" stroke-dasharray="2 2"/>')
    p.text(x, 1.0, t, anchor=anc, dy=-11, cls="lbl mute small")
save("ch14_record", figure(p.svg("한 환자의 청구 기록이 남는 모습"),
     "그림 14-1. 가상의 환자 한 명의 청구 기록을 시간 순서로 늘어놓은 모습. 외래 방문마다 명세서와 상병코드가, 처방마다 "
     "약의 주성분코드와 공급일수가, 입원에는 입원 명세서와 주상병이 남습니다. 맨 아래 줄은 환자에게 실제로 있었지만 "
     "청구자료에는 남지 않는 정보입니다."))

# ====================================================================== 14-2 prevalent vs new users
p = Plot((2013, 2020.6), (0.3, 8.7), w=640, h=330, ml=96, mr=16, mt=30, mb=46, xlabel="달력 연도",
         xticks=[2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020], yticks=[], ygrid=False, show_yaxis=False,
         xtickfmt=lambda v: f"{int(v)}")
p.fill_between([2016, 2020.6], [0.3, 0.3], [8.7, 8.7], cls="a4")
p.vline(2016, y0=0.3, y1=8.7, cls="ref strongref", dash=False, w=1.6)
p.text(2016, 8.7, "연구 등록 시작", dx=6, dy=-6, cls="lbl strong small")
p.text(2016, 8.7, "이 선 왼쪽은 연구에서 보지 못함", anchor="end", dx=-8, dy=-6, cls="lbl mute small")
pts = [  # (y, start, end, kind, end marker, note)
    (8, 2013.3, 2013.7, "gone", "stop", "초기 이상반응으로 중단"),
    (7, 2013.9, 2014.5, "gone", "event", "등록 전에 사건 발생"),
    (6, 2014.3, 2014.7, "gone", "stop", "초기에 중단"),
    (5, 2013.5, 2019.4, "prev", "cont", ""),
    (4, 2014.6, 2020.3, "prev", "cont", ""),
    (3, 2016.7, 2017.2, "new", "event", "초기 사건도 관찰됨"),
    (2, 2017.4, 2020.3, "new", "cont", ""),
    (1, 2018.3, 2018.7, "new", "stop", "초기 중단도 관찰됨"),
]
for y, a, b, kind, endm, note in pts:
    if kind == "gone":
        p.seg(a, y, b, y, cls="ln s4", w=3, dash=True)
    elif kind == "prev":
        p.seg(a, y, 2016, y, cls="ln s4", w=3, dash=True)
        p.seg(2016, y, b, y, cls="ln s2", w=3.2)
    else:
        p.seg(a, y, b, y, cls="ln s1", w=3.2)
    s = 4 if kind == "gone" else 2 if kind == "prev" else 1
    p.points([a], [y], s=s, r=3.2)
    if endm == "event":
        p.els.append(f'<path d="M{p.sx(b) - 5:.1f},{p.sy(y) - 5:.1f} l10,10 m0,-10 l-10,10" class="ln s{s}" stroke-width="2.4" fill="none"/>')
    elif endm == "stop":
        p.points([b], [y], s=s, r=4.5, hollow=True)
    if note:
        p.text(b, y, note, dx=14, dy=4, cls="lbl small" if kind == "new" else "lbl mute small")
p.text(2017.7, 5, "기존 사용자: 초기를 무사히 넘긴 사람만 남음", dy=-12, anchor="middle", cls="lbl small")
p.text(2019.0, 2, "신규 사용자: 첫 처방일부터 관찰", dy=-12, anchor="middle", cls="lbl small")
p.text_px(6, p.sy(7) + 4, "연구에 없음", cls="lbl mute small")
p.text_px(6, p.sy(4.5) + 4, "기존 사용자", cls="lbl strong small")
p.text_px(6, p.sy(2) + 4, "신규 사용자", cls="lbl strong small")
save("ch14_prevalent", figure(p.svg("기존 사용자와 신규 사용자"),
     "그림 14-2. 기존 사용자와 신규 사용자(가상의 환자 8명). 점은 약을 시작한 날, 빈 원은 중단, ×는 사건입니다. "
     "연구 등록이 시작될 때 이미 약을 쓰고 있던 기존 사용자(주황)는 초기에 중단하거나 사건을 겪은 사람(회색 점선)이 "
     "빠지고 남은 사람들입니다. 신규 사용자(파랑)는 첫 처방일부터 따라가므로 초기의 중단과 사건이 모두 자료에 들어옵니다."))

# ====================================================================== 14-3 design diagram (windows around the index date)
p = Plot((-420, 560), (0.3, 5.7), w=640, h=320, ml=104, mr=16, mt=30, mb=48, xlabel="Index date(첫 처방일) 기준 일수",
         xticks=[-365, -180, 0, 180, 365], yticks=[], ygrid=False, show_yaxis=False)
p.vline(0, y0=0.3, y1=5.7, cls="ref strongref", dash=False, w=1.6)
p.text(0, 5.7, "Index date = time 0", anchor="middle", dy=-8, cls="lbl strong small")
rows = [(5, "세척 기간"), (4, "제외 기준"), (3, "공변량"), (2, "노출군"), (1, "추적")]
for y, lab in rows:
    p.text_px(6, p.sy(y) + 4, lab, cls="lbl strong small")
bar(p, -365, -1, 5, cls="a4 s4", extra=' stroke-width="1"')
p.text(-183, 5, "두 계열 처방 없음 [−365, −1]", anchor="middle", dy=4, cls="lbl small")
bar(p, -365, 0, 4, cls="a4 s4", extra=' stroke-width="1"')
p.text(-183, 4, "심부전 입원, 말기신부전 [−365, 0]", anchor="middle", dy=4, cls="lbl small")
bar(p, -365, 0, 3, cls="a1 s1", extra=' stroke-width="1"')
p.text(-183, 3, "동반질환·병용약 [−365, 0]", anchor="middle", dy=4, cls="lbl small")
p.text(8, 3, "나이·성별 [0]", dy=4, cls="lbl small")
p.points([0], [2], s=1, r=6)
p.text(12, 2, "첫 처방이 SGLT2 억제제인가 DPP-4 억제제인가 [0]", dy=4, cls="lbl small")
bar(p, 1, 560, 1, cls="f1")
p.text(10, 1, "다음 날부터 사건·사망·중단·변경·자료 종료까지 [1, 종료]", dy=-17, cls="lbl small")
p.text(-183, 1, "index 이후 정보는", anchor="middle", dy=-3, cls="lbl mute small")
p.text(-183, 1, "선정·공변량에 쓰지 않음", anchor="middle", dy=12, cls="lbl mute small")
save("ch14_design", figure(p.svg("index date를 기준으로 한 측정 구간"),
     "그림 14-3. 예제 연구의 설계 그림. 대상자 선정(세척 기간, 제외 기준), 공변량 측정, 노출군 결정은 모두 index date "
     "당일까지의 기록만 쓰고, 추적은 그다음 날 시작합니다. 대괄호 안의 숫자는 index date를 0으로 한 일수입니다. "
     "논문의 Methods를 읽을 때 이런 그림을 직접 그려 보면 시점이 어긋난 곳을 찾기 쉽습니다."))

# ====================================================================== 14-4 grace period and the example patient
p = Plot((0, 545), (0.3, 6.0), w=640, h=330, ml=112, mr=14, mt=16, mb=48, xlabel="Index date 이후 일수",
         xticks=[0, 90, 180, 270, 360, 450, 540], yticks=[], ygrid=False, show_yaxis=False)
for y, lab in ((5, "처방 (공급일수)"), (4, "유예 30일"), (3, "유예 60일"), (2, "유예 90일"), (1, "ITT 유사")):
    p.text_px(6, p.sy(y) + 4, lab, cls="lbl strong small")
for d, s in fills:
    bar(p, d, d + s, 5)
last_run = fills[-1][0] + fills[-1][1]
p.text(207.5, 5, "45일 공백", anchor="middle", dy=-24, cls="lbl small")
p.seg(185, 5.5, 230, 5.5, cls="ln s4", w=1.4)
p.vline(hhf, y0=0.3, y1=5.9, cls="ref", dash=True)
p.text(hhf, 5.9, f"심부전 입원 ({hhf}일째)", anchor="end", dx=-6, dy=10, cls="lbl small")
for y, key in ((4, "g30"), (3, "g60"), (2, "g90")):
    g = ex[key]
    p.seg(1, y, g["days"], y, cls="ln s1", w=3.2)
    G = int(key[1:])
    # grace window that triggered the stop
    bar(p, g["stop_day"] - G, g["stop_day"], y, hh=0.2, cls="a1 s1", extra=' stroke-width="1" stroke-dasharray="3 3"')
    if g["status"]:
        p.points([g["days"]], [y], s=2, r=5.5)
        p.text(g["stop_day"], y, "사건", dx=8, dy=4, cls="lbl small")
    else:
        p.points([g["days"]], [y], s=1, r=5, hollow=True)
        p.text(g["days"], y, f"{g['days']}일째 중도절단", dx=10 if g["days"] < 300 else -10,
               anchor="start" if g["days"] < 300 else "end", dy=-12 if g["days"] >= 300 else 4, cls="lbl small")
p.seg(1, 1, ex["itt_days"], 1, cls="ln s1", w=3.2)
p.points([ex["itt_days"]], [1], s=2, r=5.5)
p.text(ex["itt_days"], 1, "사건", dx=10, dy=4, cls="lbl small")
save("ch14_grace", figure(p.svg("유예기간에 따라 달라지는 추적"),
     "그림 14-4. 같은 처방 기록에 유예기간을 다르게 적용한 결과(가상의 환자 한 명). 점선 상자는 중단을 판정한 유예기간입니다. "
     "유예 30일이면 45일 공백에서 중단으로 보아 215일째에 중도절단하고, 60일이면 마지막 처방이 끝난 뒤 470일째에 "
     "중도절단하며, 90일이면 475일째의 심부전 입원이 약을 쓰는 중의 사건으로 잡힙니다."))

# ====================================================================== 14-5 forest: analysis strategies
fu = N["fu"]
rows = [{"label": "As-treated (약을 쓰는 동안만 추적)", "header": True}]
for key, lab in (("at60", "유예 60일 (주 분석)"), ("at30", "유예 30일"), ("at90", "유예 90일")):
    r = fu[key]["adj"]
    rows.append({"label": lab, "est": r["hr"], "lo": r["lo"], "hi": r["hi"], "indent": True,
                 "bold": False, "s": 1})
rows.append({"label": "ITT 유사 (중단·변경 무시)", "header": True})
r = fu["itt"]["adj"]
rows.append({"label": "처음 시작한 약 기준", "est": r["hr"], "lo": r["lo"], "hi": r["hi"], "indent": True, "s": 2})
svg = forest(rows, xlim=(0.45, 1.15), ref=1.0, log=True, w=640, row_h=30, label_w=250, est_w=140,
             xlabel="보정 위험비 (95% CI). 1보다 작으면 SGLT2 억제제군의 위험률이 낮음", xticks=[0.5, 0.6, 0.7, 0.8, 1.0])
save("ch14_forest", figure(svg,
     "그림 14-5. 추적 정의에 따른 심부전 입원의 보정 위험비(예제 코호트, 나이·성별·동반질환·병용약 보정). 유예기간을 "
     "30일, 60일, 90일로 바꿔도 as-treated 결과는 거의 같고, 중단 뒤의 기간까지 포함하는 ITT 유사 분석은 1에 더 가깝습니다. "
     "네 분석이 같은 방향인지 보는 것이 민감도 분석의 목적입니다."))
print("figures written")
