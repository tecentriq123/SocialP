"""실습 20 · ICER와 순편익 계산 — cells run for real with labkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/lab_lab20.py
      then  python3 tools/nbcheck.py lab20   (runs the notebook alone in a fresh process and compares outputs)
      (NOMARK=1 을 앞에 붙이면 표식 없이 셀 출력만 화면에 찍는다. VERIFY=1 이면 본문 문장의 숫자를 더 찍는다)

자료 파일은 없다. 네 대안(표준요법 B, 기존약 C, 병용요법 D, 신약 A)의 총비용과 QALY 8개를 노트북에 직접 적는다.
B와 A는 20–25장 공통 예시의 기준 분석 결과(gen/lib_p4.py의 run(), 실습 23 셀 15–16)이고 C와 D는 20장 d 절이 지어낸 대안이다.
B와 A의 값은 순금전편익의 끝자리까지 20장 본문과 같게 나오도록 비용은 소수 셋째 자리, QALY는 여섯째 자리까지 적었다
(20장 '더 알아보기'의 효율 경계 코드처럼 7,689.5와 1.87364로 적으면 순금전편익이 −317.2로 나온다. 대조 블록이 확인한다).
학생 코드는 gen/lib_p4.py를 쓰지 않는다. 끝의 대조 블록이 실습 결과를 lib_p4의 기준 분석과 gen/_ch20_nums.json(20장의 숫자)에 맞춘다.
(2026-10-15 새로 만듦. 20장 작성자의 '실습 20에 넣을 셀'에서 골랐다.)
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from labkit import Notebook, _mark  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
nb = Notebook("lab20")


def render(c, marks=None, dfmarks=None):
    """text marks -> first text <pre> only; table marks -> DataFrame only."""
    h = nb.html(c)
    if marks:
        i = h.index('<div class="cell-out">')
        j = h.index("<pre>", i)
        k = h.index("</pre>", j)
        h = h[:j] + _mark(h[j:k], marks) + h[k:]
    if dfmarks:
        j = h.index('<div class="df-wrap">')
        k = h.index("</table>", j)
        h = h[:j] + _mark(h[j:k], dfmarks) + h[k:]
    return h


def save(name, c, marks=None, dfmarks=None):
    if os.environ.get("NOMARK"):
        print(f"===== {name} (cell {c.n})\n{c.stdout}{c.value_repr or ''}{c.error or ''}")
        for w in c.warns:
            print("WARN:", w)
        marks = dfmarks = None
    nb.save_fragment(name, render(c, marks, dfmarks))


# ---------------------------------------------------------------- a. 실습 데이터 준비
c = nb.cell('''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

print("numpy     ", np.__version__)
print("pandas    ", pd.__version__)
print("matplotlib", matplotlib.__version__)
''', title="패키지 불러오기와 버전 확인")
save("lab20_versions", c, marks={"2.5.3": 1, "3.0.5": 2, "3.11.2": 3})

c = nb.cell('''
alts = pd.DataFrame({
    "cost": [7689.485, 8100.0, 9400.0, 10561.778],   # 총비용(만원, 할인 후)
    "qaly": [1.873640, 1.700, 2.000, 2.384669],       # 총 QALY(할인 후)
}, index=["B", "C", "D", "A"])                         # B 표준요법, C 기존약, D 병용요법, A 신약
print(alts.shape)
alts
''', title="네 대안의 총비용과 QALY")
save("lab20_inputs", c, marks={"(4, 2)": 1}, dfmarks={"7689.485": 2, "1.700000": 3, "10561.778": 4})

# ---------------------------------------------------------------- b. 증분비용효과비와 지배
c = nb.cell('''
cost_A, qaly_A = alts.loc["A", "cost"], alts.loc["A", "qaly"]
cost_B, qaly_B = alts.loc["B", "cost"], alts.loc["B", "qaly"]

d_cost = cost_A - cost_B               # 증분비용
d_qaly = qaly_A - qaly_B               # 증분 QALY
icer = d_cost / d_qaly                 # 증분비용효과비
print("증분비용  :", round(d_cost, 1), "만원")
print("증분 QALY :", round(d_qaly, 3))
print("ICER      :", round(icer), "만원/QALY")
print("평균 비   : A", round(cost_A / qaly_A), " B", round(cost_B / qaly_B))

ly_A, ly_B = 3.398772, 2.728843        # 할인한 생존연수
print("생존연수당:", round(d_cost / (ly_A - ly_B)), "만원/생존연수")
''', title="신약 A 대 표준요법 B의 증분과 ICER")
save("lab20_icer", c, marks={"2872.3": 1, "0.511": 2, "5621": 3, "A 4429": 4, "B 4104": 5, "4287": 6})
C_ICER = c.n

c = nb.cell('''
def quadrant(dc, dq):
    """증분비용(dc)과 증분 QALY(dq)의 부호로 비용효과평면의 사분면을 판정"""
    if dq > 0 and dc <= 0:
        return "오른쪽 아래, 우월(dominant)"
    if dq <= 0 and dc > 0:
        return "왼쪽 위, 열등(dominated)"
    if dq > 0:
        return "오른쪽 위, ICER를 임계값과 견줌"
    return "왼쪽 아래, ICER가 임계값보다 높아야 받아들임"

cases = {"A vs B": (d_cost, d_qaly),
         "예 1  ": (-500, 0.2),       # 500만원 덜 들고 0.2 QALY를 더 얻음
         "예 2  ": (500, -0.2),       # 500만원 더 들고 0.2 QALY를 잃음
         "예 3  ": (-1000, -0.1)}     # 1,000만원 아끼고 0.1 QALY를 잃음
for name, (dc, dq) in cases.items():
    print(name, "ICER", round(dc / dq), "|", quadrant(dc, dq))
''', title="사분면 판정과 음수 ICER")
save("lab20_quad", c, marks={"ICER 5621 | 오른쪽 위": 1, "ICER -2500 | 오른쪽 아래": 2, "ICER -2500 | 왼쪽 위": 3,
                             "ICER 10000": 4})
C_QUAD = c.n

c = nb.cell('''
def icer_between(df, lo, hi):
    """대안 lo에서 대안 hi로 옮길 때의 ICER"""
    return ((df.loc[hi, "cost"] - df.loc[lo, "cost"])
            / (df.loc[hi, "qaly"] - df.loc[lo, "qaly"]))

def frontier(df):
    """비용순으로 놓고 열등과 확장 열등을 표시한 뒤, 효율 경계 위에서 이웃끼리 ICER를 구한 표"""
    df = df.sort_values("cost").copy()          # 비용이 낮은 순서
    df["status"], df["vs"], df["icer"] = "경계", "", np.nan
    keep = []                                   # 경계에 남은 대안의 이름
    for i in df.index:
        if any(df.loc[j, "qaly"] >= df.loc[i, "qaly"] for j in keep):
            df.loc[i, "status"] = "열등"         # 더 싸면서 효과가 같거나 큰 대안이 있음
            continue
        keep.append(i)
        while len(keep) >= 3:                   # 이어진 세 대안의 ICER 순서 확인
            a, b, c = keep[-3:]
            if icer_between(df, a, b) <= icer_between(df, b, c):
                break
            print(b, "확장 열등:", round(icer_between(df, a, b)), ">",
                  round(icer_between(df, b, c)))
            df.loc[b, "status"] = "확장 열등"
            keep.pop(-2)                        # 가운데 대안을 뺌
    for a, b in zip(keep[:-1], keep[1:]):       # 경계 위의 이웃끼리
        df.loc[b, "vs"] = a
        df.loc[b, "icer"] = icer_between(df, a, b)
    return df

front = frontier(alts)
front.round({"icer": 0})
''', title="열등, 확장 열등, 효율 경계")
save("lab20_front", c, marks={"D 확장 열등: 13537 > 3020": 1},
     dfmarks={"열등": 2, "확장 열등": 3, "5621.0": 4})
C_FRONT = c.n

c = nb.cell('''
f = (alts.loc["D", "qaly"] - qaly_B) / (qaly_A - qaly_B)   # A를 쓸 환자의 비율
mix_cost = cost_B + f * (cost_A - cost_B)
mix_qaly = qaly_B + f * (qaly_A - qaly_B)
print("A를 쓰는 비율:", round(f * 100, 1), "%")
print("섞은 전략 : QALY", round(mix_qaly, 3), " 비용", round(mix_cost, 1))
print("병용요법 D: QALY", alts.loc["D", "qaly"], " 비용", alts.loc["D", "cost"])
print("D보다 적게 드는 비용:", round(alts.loc["D", "cost"] - mix_cost, 1), "만원")
''', title="B와 A를 섞은 전략과 병용요법 D")
save("lab20_mix", c, marks={"24.7 %": 1, "QALY 2.0  비용 8399.7": 2, "1000.3": 3})
C_MIX = c.n

c = nb.cell('''
fig, ax = plt.subplots(figsize=(6, 4.2))
on = front[front["status"] == "경계"]
off = front[front["status"] != "경계"]
ax.plot(on["qaly"], on["cost"], "-o", color="tab:blue", label="Efficiency frontier")
ax.plot(off["qaly"], off["cost"], "s", color="tab:red",
        label="Dominated or extendedly dominated")
ax.plot(mix_qaly, mix_cost, "o", mfc="white", color="tab:blue",
        label="24.7% A + 75.3% B")
for name, row in front.iterrows():
    ax.annotate(name, (row["qaly"], row["cost"]),
                xytext=(6, -12), textcoords="offset points")
ax.margins(x=0.08, y=0.12)              # 가장자리의 이름이 잘리지 않게
ax.set_xlabel("Total QALYs")
ax.set_ylabel("Total cost (10,000 KRW)")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
''', title="네 대안과 효율 경계")
save("lab20_frontplot", c)
C_FRONTPLOT = c.n

# ---------------------------------------------------------------- c. 순편익과 비용효과평면
c = nb.cell('''
lam = 5000                             # 이 예시에서 가정한 임계값(만원/QALY)
nmb = lam * d_qaly - d_cost            # 증분 순금전편익(만원)
nhb = d_qaly - d_cost / lam            # 증분 순건강편익(QALY)
print("순금전편익:", round(nmb, 1), "만원")
print("순건강편익:", round(nhb, 3), "QALY")
print("환자 100명이면:", round(100 * nhb, 1), "QALY")
print("ICER < 임계값:", icer < lam, "  순금전편익 > 0:", nmb > 0)
''', title="임계값 5,000만원에서 순금전편익과 순건강편익")
save("lab20_nmb", c, marks={"-317.1": 1, "-0.063": 2, "-6.3": 3, "ICER < 임계값: False": 4})
C_NMB = c.n

c = nb.cell('''
tab = pd.DataFrame({"lam": [2500, 3000, 5000, 6000, 7000, 8000]})
tab["nmb"] = tab["lam"] * d_qaly - d_cost      # 순금전편익(만원)
tab["nhb"] = tab["nmb"] / tab["lam"]           # 순건강편익(QALY)
tab["ce"] = tab["nmb"] > 0                     # 비용효과적인가
print("부호가 바뀌는 임계값:", round(d_cost / d_qaly, 1))
tab.round({"nmb": 1, "nhb": 3})
''', title="임계값을 바꾼 순금전편익")
save("lab20_lams", c, marks={"5620.6": 1},
     dfmarks={"-1594.7": 2, "-1339.2": 3, "193.9": 4, "704.9": 5, "1215.9": 6})
C_LAMS = c.n

c = nb.cell('''
nmb4 = pd.DataFrame({lam_: lam_ * alts["qaly"] - alts["cost"]
                     for lam_ in [5000, 6000]})
print("순금전편익이 가장 큰 대안:", nmb4.idxmax().to_dict())
nmb4.round(1)
''', title="네 대안의 순금전편익")
save("lab20_nmb4", c, marks={"{5000: 'B', 6000: 'A'}": 1},
     dfmarks={"1678.7": 2, "400.0": 3, "600.0": 4, "1361.6": 5, "3746.2": 6})
C_NMB4 = c.n

c = nb.cell('''
grid = np.arange(0, 20001, 100)        # 임계값 0, 100, ..., 20,000
best = [(g * alts["qaly"] - alts["cost"]).idxmax() for g in grid]
print("가장 큰 대안이 된 횟수:", pd.Series(best).value_counts().to_dict())
print("A가 처음 가장 커지는 임계값:", grid[best.index("A")])
''', title="임계값마다 순금전편익이 가장 큰 대안")
save("lab20_best", c, marks={"{'A': 144, 'B': 57}": 1, "5700": 2})
C_BEST = c.n

c = nb.cell('''
fig, ax = plt.subplots(figsize=(6, 4.5))
x = np.array([-0.6, 0.8])
ax.fill_between(x, lam * x, -3500, color="tab:blue", alpha=0.08)   # 비용효과적인 영역
ax.plot(x, lam * x, color="gray", ls="--", label="Threshold 5,000 per QALY")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(0, color="black", lw=0.8)
ax.plot(d_qaly, d_cost, "o", ms=8, color="tab:blue", label="Drug A vs B")
ax.plot([d_qaly, d_qaly], [lam * d_qaly, d_cost], color="tab:red", lw=2,
        label="NMB = -317")
ax.plot(0.2, -500, "^", ms=8, color="tab:green", label="Example 1 (dominant)")
ax.plot(-0.2, 500, "v", ms=8, color="tab:orange", label="Example 2 (dominated)")
ax.set_xlim(-0.6, 0.8)
ax.set_ylim(-3500, 4500)
ax.set_xlabel("Incremental QALYs")
ax.set_ylabel("Incremental cost (10,000 KRW)")
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
''', title="비용효과평면")
save("lab20_plane", c)
C_PLANE = c.n

# ---------------------------------------------------------------- d. 과제
c = nb.cell('''
lam6 = 6000
value = lam6 * d_qaly                  # 늘어난 QALY를 임계값으로 돈으로 바꾼 값
print("늘어난 QALY의 값:", round(value, 1), "만원")
print("늘어난 비용     :", round(d_cost, 1), "만원")
print("순금전편익      :", round(value - d_cost, 1), "만원")
print("순건강편익      :", round(d_qaly - d_cost / lam6, 3), "QALY")
print("환자 100명이면  :", round(100 * (value - d_cost)), "만원,",
      round(100 * (d_qaly - d_cost / lam6), 1), "QALY")
''', title="과제 1 정답. 임계값 6,000만원 행 풀어 보기")
save("lab20_hw1", c, marks={"3066.2": 1, "2872.3": 2, "193.9": 3, "0.032": 4, "19388": 5, "3.2 QALY": 6})
C_HW1 = c.n

c = nb.cell('''
alts_E = alts.copy()
alts_E.loc["E"] = [cost_B + 600, qaly_B + 0.20]   # 신약 E: B보다 600만원 더, 0.20 QALY 더
front_E = frontier(alts_E)
print(front_E.round({"icer": 0}))
for lam_ in [5000, 8000]:
    nb_ = lam_ * alts_E["qaly"] - alts_E["cost"]
    print(lam_, "| 가장 큰 대안:", nb_.idxmax(),
          "| B 대비 E", round(nb_["E"] - nb_["B"], 1),
          " A", round(nb_["A"] - nb_["B"], 1))
''', title="과제 2 정답. 신약 E를 더한 효율 경계")
save("lab20_hw2", c, marks={"8289.485": 1, "9400.000  2.000000     열등": 2, "3000.0": 3, "7306.0": 4,
                            "5000 | 가장 큰 대안: E": 5, "8000 | 가장 큰 대안: A": 6})
C_HW2 = c.n

c = nb.cell('''
ratio = alts["cost"] / alts["qaly"]               # 동료가 구한 총비용 ÷ QALY
print("동료의 비:", ratio.round().to_dict())
print("임계값보다 낮은 대안:", list(ratio[ratio < lam].index))
print("동료의 A 대 D:", round(icer_between(alts, "D", "A")))
print("D의 판정:", front.loc["D", "status"])
print("경계 위 A 대", front.loc["A", "vs"], ":", round(front.loc["A", "icer"]))
print("임계값 5,000에서 순금전편익이 가장 큰 대안:",
      (lam * alts["qaly"] - alts["cost"]).idxmax())
''', title="과제 3 정답. 동료의 분석 다시 보기")
save("lab20_hw3", c, marks={"'C': 4765.0": 1, "['B', 'C', 'D', 'A']": 2, "동료의 A 대 D: 3020": 3,
                            "D의 판정: 확장 열등": 4, "B : 5621": 5, "가장 큰 대안: B": 6})
C_HW3 = c.n

print("lab20: cells", len(nb.cells))
for cc in nb.cells:
    if cc.warns:
        print("WARN cell", cc.n, cc.warns)

# ---------------------------------------------------------------- 대조 블록: lib_p4와 20장의 숫자에 맞추기
import lib_p4 as L  # noqa: E402  (읽기만 한다. 학생 코드는 이 파일을 쓰지 않는다)

ns = nb.ns
N = json.load(open(os.path.join(HERE, "_ch20_nums.json"), encoding="utf-8"))
REF = L.run()
CHECKS = []


def chk(name, got, want, tol):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    err = float(np.max(np.abs(got - want)))
    CHECKS.append((name, err))
    assert err < tol, f"{name}: lab {got} vs reference {want} (err {err:g})"


al = ns["alts"]
# 입력값: B와 A는 기준 분석의 결과를 반올림한 값(비용 0.0005만원, QALY 0.0000005 안쪽)
chk("inputs B, A cost = lib_p4.run()", [al.loc["B", "cost"], al.loc["A", "cost"]], [REF["B"]["cost"], REF["A"]["cost"]], 5e-4)
chk("inputs B, A QALY = lib_p4.run()", [al.loc["B", "qaly"], al.loc["A", "qaly"]], [REF["B"]["qaly"], REF["A"]["qaly"]], 5e-7)
chk("discounted LY A, B", [ns["ly_A"], ns["ly_B"]], [REF["A"]["ly_d"], REF["B"]["ly_d"]], 5e-7)
S4 = {d["name"]: d for d in N["strats"]}
chk("C, D inputs = 20장 표 20-12", [al.loc["C", "cost"], al.loc["C", "qaly"], al.loc["D", "cost"], al.loc["D", "qaly"]],
    [S4["기존약 C"]["cost"], S4["기존약 C"]["eff"], S4["병용요법 D"]["cost"], S4["병용요법 D"]["eff"]], 1e-12)
# b 절: 증분, ICER, 평균 비, 생존연수당
b = N["base"]
chk("d_cost, d_qaly", [ns["d_cost"], ns["d_qaly"]], [b["d_cost"], b["d_qaly"]], 1e-3)
chk("ICER, ICER per LY", [ns["icer"], ns["d_cost"] / (ns["ly_A"] - ns["ly_B"])], [b["icer"], b["icer_ly"]], 0.01)
chk("average ratios A, B", [ns["cost_A"] / ns["qaly_A"], ns["cost_B"] / ns["qaly_B"]], [b["acer_A"], b["acer_B"]], 0.01)
assert (round(ns["d_cost"], 1), round(ns["d_qaly"], 3), round(ns["icer"]), round(ns["icer"], 1)) == (2872.3, 0.511, 5621, 5620.6)
assert (round(b["d_cost"], 1), round(b["icer"], 1)) == (2872.3, 5620.6)
assert (round(ns["cost_A"] / ns["qaly_A"]), round(ns["cost_B"] / ns["qaly_B"]), round(ns["d_cost"] / (ns["ly_A"] - ns["ly_B"]))) == (4429, 4104, 4287)
# 사분면과 음수 ICER(20장 d 절 3단계, '덜 효과적이고 더 싼 대안')
q = ns["quadrant"]
for (dc, dq), want in (((ns["d_cost"], ns["d_qaly"]), "NE"), ((-500, 0.2), "SE"), ((500, -0.2), "NW"), ((-1000, -0.1), "SW")):
    got = L.quadrants([dc], [dq])
    assert got[want] == 1.0, (dc, dq, want, got)
assert q(-500, 0.2).startswith("오른쪽 아래") and q(500, -0.2).startswith("왼쪽 위") and q(-1000, -0.1).startswith("왼쪽 아래")
assert (N["neg"]["dominant"][2], N["neg"]["dominated"][2], N["neg"]["sw"][2]) == (-2500, -2500, 10000)
# 효율 경계(20장 표 20-12, '더 알아보기'의 코드)
fr = ns["front"]
want_status = {"표준요법 B": ("B", "경계"), "기존약 C": ("C", "열등"), "병용요법 D": ("D", "확장 열등"), "신약 A": ("A", "경계")}
STATUS_20 = {"": "경계", "열등": "열등", "확장 열등": "확장 열등"}
for d in N["strats"]:
    k, st = want_status[d["name"]]
    assert fr.loc[k, "status"] == st == STATUS_20[d["status"]], (k, fr.loc[k, "status"], d["status"])
chk("frontier ICER A vs B", fr.loc["A", "icer"], S4["신약 A"]["icer"], 0.01)
assert fr.loc["A", "vs"] == "B" and list(fr.index) == ["B", "C", "D", "A"]
chk("D vs B, A vs D", [ns["icer_between"](al, "B", "D"), ns["icer_between"](al, "D", "A")], [N["seq"]["D_vs_B"], N["seq"]["A_vs_D"]], 0.2)
assert (round(ns["icer_between"](al, "B", "D")), round(ns["icer_between"](al, "D", "A"))) == (13537, 3020)
chk("mix f, cost, saving", [ns["f"], ns["mix_cost"], ns["alts"].loc["D", "cost"] - ns["mix_cost"]], [N["mix"]["f"], N["mix"]["cost"], N["mix"]["saving"]], 0.02)
assert (round(ns["f"] * 100, 1), round(ns["mix_cost"], 1), round(9400 - ns["mix_cost"], 1)) == (24.7, 8399.7, 1000.3)
assert round(N["mix"]["cost"], 1) == 8399.7 and round(N["mix"]["saving"]) == 1000
# c 절: 순금전편익, 순건강편익, 임계값별, 네 대안
chk("NMB, NHB at 5,000", [ns["nmb"], ns["nhb"]], [b["nmb"], b["nhb"]], 0.01)
assert (round(ns["nmb"], 1), round(b["nmb"], 1), round(ns["nhb"], 3), round(100 * ns["nhb"], 1)) == (-317.1, -317.1, -0.063, -6.3)
assert round(-N["opp"]["net"], 1) == 6.3
tb = ns["tab"]
for lam_, want in zip(tb["lam"], tb["nmb"]):
    chk(f"NMB at {lam_}", want, L.nmb(REF, lam_), 0.01)
assert [round(x, 1) for x in tb["nmb"]] == [-1594.7, -1339.2, -317.1, 193.9, 704.9, 1215.9]
assert [round(L.nmb(REF, l_), 1) for l_ in (2500, 3000, 5000, 6000, 7000, 8000)] == [-1594.7, -1339.2, -317.1, 193.9, 704.9, 1215.9]
assert all(round(N["practice"]["q5"][k]) == round(L.nmb(REF, float(k))) for k in ("3000", "5000", "6000", "7000"))
n4 = ns["nmb4"]
for d in N["strats"]:
    k = want_status[d["name"]][0]
    chk(f"NMB at 5,000 of {k}", n4.loc[k, 5000], d["nmb"], 0.01)
assert [round(n4.loc[k, 5000], 1) for k in "BCDA"] == [1678.7, 400.0, 600.0, 1361.6]
assert n4[5000].idxmax() == "B" and n4[6000].idxmax() == "A"
best_ = ns["best"]
assert set(best_) == {"A", "B"} and ns["grid"][best_.index("A")] == 5700 and ns["grid"][best_.index("A") - 1] < b["icer"] < 5700
# 과제 1: 임계값 6,000만원 행
assert (round(6000 * ns["d_qaly"], 1), round(6000 * ns["d_qaly"] - ns["d_cost"], 1), round(ns["d_qaly"] - ns["d_cost"] / 6000, 3)) == (3066.2, 193.9, 0.032)
# 과제 2: 신약 E(20장 d 절 스스로 확인하기 2번)
fE = ns["front_E"]
assert list(fE.index) == ["B", "C", "E", "D", "A"]
assert [fE.loc[k, "status"] for k in ["B", "C", "E", "D", "A"]] == ["경계", "열등", "경계", "열등", "경계"]
E_ref = (REF["B"]["cost"] + 600, REF["B"]["qaly"] + 0.20)
chk("E vs B, A vs E", [fE.loc["E", "icer"], fE.loc["A", "icer"]],
    [600 / 0.20, (REF["A"]["cost"] - E_ref[0]) / (REF["A"]["qaly"] - E_ref[1])], 0.02)
assert (round(fE.loc["E", "icer"]), round(fE.loc["A", "icer"])) == (3000, 7306)
for lam_, want_best, want_e, want_a in ((5000, "E", 400.0, -317.1), (8000, "A", 1000.0, 1215.9)):
    nb_ = lam_ * ns["alts_E"]["qaly"] - ns["alts_E"]["cost"]
    assert nb_.idxmax() == want_best and round(nb_["E"] - nb_["B"], 1) == want_e and round(nb_["A"] - nb_["B"], 1) == want_a
# 과제 3: 평균 비와 확장 열등 대안과의 비교
ra = ns["ratio"]
assert [round(ra[k]) for k in "BCDA"] == [4104, 4765, 4700, 4429] and (ra < 5000).all()
# 본문 문장에 표식 없이 쓴 숫자
assert round(ns["alts"].loc["D", "qaly"] - ns["qaly_B"], 3) == 0.126 and round(ns["d_cost"] / 5000, 3) == 0.574 and round(5000 * ns["d_qaly"]) == 2555
assert [round(b_ - a_, 1) for a_, b_ in zip(tb["nmb"][3:-1], tb["nmb"][4:])] == [511.0, 511.0]
assert round(5000 * (2.38467 - 1.87364) - (10561.8 - 7689.5), 1) == -317.2 and round(2872 / 0.511) == 5620   # 참고 상자의 20장 코드 입력값, 셀 3의 표시값 계산
assert (round(b["nmb"], 4), round(L.nmb(REF), 4)) == (-317.1497, -317.1497)
assert round(100 * (6000 * ns["d_qaly"] - ns["d_cost"]) / 10000, 2) == 1.94
assert (round(ns["cost_A"] - ns["alts_E"].loc["E", "cost"], 1), round(ns["qaly_A"] - ns["alts_E"].loc["E", "qaly"], 3)) == (2272.3, 0.311)
assert (ns["best"].count("B"), ns["best"].count("A")) == (57, 144)
print(f"대조: {len(CHECKS)}개 항목 모두 허용 오차 안 (가장 큰 차이 {max(e for _, e in CHECKS):.2e}), 나머지 assert 통과")

# ---------------------------------------------------------------- 본문 문장에 쓴 그 밖의 숫자 확인
if os.environ.get("VERIFY") or os.environ.get("NOMARK"):
    cB20, qB20, cA20, qA20 = 7689.5, 1.87364, 10561.8, 2.38467     # 20장 '더 알아보기' 코드의 입력값
    print("VERIFY NMB with ch20 code inputs:", round(5000 * (qA20 - qB20) - (cA20 - cB20), 1))
    print("VERIFY ICER from rounded table values 2872/0.511:", round(2872 / 0.511))
    print("VERIFY exact lib ICER, NMB:", round(REF["icer"], 4), round(L.nmb(REF), 4), " lab:", round(ns["icer"], 4), round(ns["nmb"], 4))
    print("VERIFY hw1 per 100 patients (억 원):", round(100 * (6000 * ns["d_qaly"] - ns["d_cost"]) / 10000, 2))
    print("VERIFY hw2 E cost, QALY:", ns["alts_E"].loc["E"].tolist(), " NMB at 5000 E, A (abs):",
          round(5000 * ns["alts_E"].loc["E", "qaly"] - ns["alts_E"].loc["E", "cost"], 1))
    print("VERIFY hw2 A vs E pieces:", round(ns["cost_A"] - ns["alts_E"].loc["E", "cost"], 1), round(ns["qaly_A"] - ns["alts_E"].loc["E", "qaly"], 3))
    print("VERIFY best-count grid: B", ns["best"].count("B"), "A", ns["best"].count("A"), "C", ns["best"].count("C"), "D", ns["best"].count("D"))
    print("VERIFY D - B QALY:", round(ns["alts"].loc["D", "qaly"] - ns["qaly_B"], 3), " displaced QALY d_cost/5000:", round(ns["d_cost"] / 5000, 3),
          " money value 5000 x d_qaly:", round(5000 * ns["d_qaly"]), " NMB step per 1,000:", round(1000 * ns["d_qaly"], 1))

# ---------------------------------------------------------------- 노트북 (pub/notebooks/lab20.ipynb)
notes = {
    1: ("## a. 실습 데이터 준비\n\n출력을 만든 패키지의 버전을 확인하고, 네 대안(표준요법 B, 기존약 C, 병용요법 D, 신약 A)의 "
        "총비용과 QALY를 표로 만듭니다."),
    C_ICER: "## b. 증분비용효과비와 지배\n\n신약 A 대 표준요법 B의 ICER를 구하고, 사분면 판정과 효율 경계를 코드로 만듭니다.",
    C_NMB: ("## c. 순편익과 비용효과평면\n\n이 예시에서 가정한 임계값 5,000만원/QALY(우리나라에는 공식 임계값이 없음)에서 "
            "순금전편익과 순건강편익을 구하고, 임계값을 바꿔 봅니다."),
    C_HW1: ("## 과제 정답\n\n**과제 1.** 셀 %d 표의 임계값 6,000만원 행(순금전편익 +193.9만원)을 급여 평가 보고서에 쓰듯 "
            "한 문장으로 옮기고, 이 값이 뜻하지 않는 것을 두 가지 씁니다." % C_LAMS),
    C_HW2: ("**과제 2.** 표준요법 B보다 600만원 더 들고 0.20 QALY를 더 얻는 신약 E를 네 대안에 더해 효율 경계를 다시 만들고, "
            "임계값 5,000만원과 8,000만원에서 어느 대안을 고르는지 답합니다."),
    C_HW3: ("**과제 3.** 총비용 ÷ QALY와 병용요법 D와 견준 ICER로 신약 A가 비용효과적이라고 결론 낸 동료의 분석에서 "
            "틀린 곳 두 가지를 찾습니다."),
}
assert (C_ICER, C_NMB, C_HW1, C_HW2, C_HW3) == (3, 8, 13, 14, 15)
path = nb.save_ipynb(
    "실습 20. ICER와 순편익 계산",
    intro=("사회약학 연구방법 노트의 '실습 20 ICER와 순편익 계산'에 나오는 셀을 차례로 모은 노트북입니다. "
           "진행성 신세포암 1차 치료(신약 A 대 표준요법 B, 가상의 예시)의 기준 분석 결과와 20장이 지어낸 대안 두 개(기존약 C, "
           "병용요법 D)로 증분비용효과비, 지배와 확장 지배, 효율 경계, 순금전편익과 순건강편익, 비용효과평면을 계산해 "
           "20장의 숫자와 맞춰 봅니다. 자료 파일 없이 숫자를 직접 적으므로 인터넷 연결이 없어도 됩니다. "
           "설명과 출력 읽는 법은 사이트의 실습 20 쪽에 있습니다."),
    notes=notes)
print("notebook:", path, len(nb.cells), "cells")
