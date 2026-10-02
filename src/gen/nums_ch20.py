"""Numbers for chapter 20 (경제성 평가의 틀).
run: source /home/claude/pylibs/env.sh && python3 gen/nums_ch20.py
Every number in content/ch20.html comes from this script (printed and saved to gen/_ch20_nums.json).
The shared example is imported from gen/lib_p4.py and never retyped."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import lib_p4 as L

N = {}
LAM = L.THRESHOLD
r = L.run()
A, B = r["A"], r["B"]

# ------------------------------------------------------------ base case (가, 라)
base = {
    "cost_A": A["cost"], "cost_B": B["cost"], "d_cost": r["d_cost"],
    "ly_A": A["ly"], "ly_B": B["ly"], "d_ly_undisc": A["ly"] - B["ly"],
    "lyd_A": A["ly_d"], "lyd_B": B["ly_d"], "d_lyd": r["d_ly"],
    "qaly_A": A["qaly"], "qaly_B": B["qaly"], "d_qaly": r["d_qaly"],
    "icer": r["icer"], "icer_ly": r["icer_ly"], "nmb": L.nmb(r), "nhb": L.nmb(r) / LAM,
    "icer_from_rounded": round(r["d_cost"]) / round(r["d_qaly"], 3),
}
for k in ("c_drug", "c_pf", "c_pd", "c_death", "c_ae"):
    base[k + "_A"], base[k + "_B"], base["d_" + k] = A[k], B[k], A[k] - B[k]
# average cost-effectiveness ratios
base["acer_A"] = A["cost"] / A["qaly"]
base["acer_B"] = B["cost"] / B["qaly"]
# benefit side valued in money at the assumed threshold (CBA-like presentation)
base["benefit_money"] = LAM * r["d_qaly"]
# price at which ICER = threshold (monthly price of A)
p = L.base_params()
lo, hi = 100.0, 190.0
for _ in range(60):
    mid = (lo + hi) / 2
    q = dict(p); q["c_drug_A"] = mid
    if L.run(q)["icer"] > LAM: hi = mid
    else: lo = mid
base["price_at_thr"] = (lo + hi) / 2
N["base"] = base

# ------------------------------------------------------------ 가: opportunity cost illustrations
# (1) a fixed budget of 1억 원 and two programmes (가상의 숫자)
bud = 10000.0   # 만원
X = {"cost": 500.0, "gain": 0.10}    # 사업 X: 1인당 500만원, 0.10 QALY
Y = {"cost": 200.0, "gain": 0.05}    # 사업 Y: 1인당 200만원, 0.05 QALY
N["budget"] = {"bud": bud,
               "nX": bud / X["cost"], "qX": bud / X["cost"] * X["gain"], "perX": X["cost"] / X["gain"],
               "nY": bud / Y["cost"], "qY": bud / Y["cost"] * Y["gain"], "perY": Y["cost"] / Y["gain"]}
# (2) 100 patients on A instead of B: health gained vs health displaced at the assumed threshold
n_pat = 100
N["opp"] = {"n": n_pat, "extra_cost": n_pat * r["d_cost"], "gain": n_pat * r["d_qaly"],
            "displaced": n_pat * r["d_cost"] / LAM, "net": n_pat * (r["d_qaly"] - r["d_cost"] / LAM)}

# ------------------------------------------------------------ 다: time horizon
hz = {}
for yrs in (3, 5, 10, 20):
    x = L.run(horizon=12 * yrs)
    hz[yrs] = {"d_cost": x["d_cost"], "d_qaly": x["d_qaly"], "icer": x["icer"],
               "cost_A": x["A"]["cost"], "cost_B": x["B"]["cost"], "qaly_A": x["A"]["qaly"], "qaly_B": x["B"]["qaly"]}
N["horizon"] = hz
curve = []
for m in range(12, 241, 6):
    x = L.run(horizon=m)
    curve.append((m / 12.0, x["icer"], x["d_cost"], x["d_qaly"]))
N["horizon_curve"] = curve
t = np.array([36, 60, 120, 240])
sA = L.curves(p, "A", t); sB = L.curves(p, "B", t)
N["alive"] = {"t": t.tolist(), "os_A": sA[1].tolist(), "os_B": sB[1].tolist()}

# ------------------------------------------------------------ 다: discounting
disc = {}
for rate in (0.045, 0.03, 0.015):
    disc[str(rate)] = {str(y): 1 / (1 + rate) ** y for y in (1, 5, 10, 20)}
N["disc_factor"] = disc
dr = {}
for rate in (0.045, 0.03, 0.0):
    x = L.run(disc=rate)
    dr[str(rate)] = {"cost_A": x["A"]["cost"], "cost_B": x["B"]["cost"], "d_cost": x["d_cost"],
                     "qaly_A": x["A"]["qaly"], "qaly_B": x["B"]["qaly"], "d_qaly": x["d_qaly"], "icer": x["icer"]}
N["disc_run"] = dr
# differential discounting, shown only as a remark (costs 4.5%, effects 0%)
x0 = L.run(disc=0.0)
N["disc_costonly"] = {"icer": r["d_cost"] / x0["d_qaly"]}
# a single payment example: 1,000만원 paid 10 years from now
N["pv_example"] = {"pv10": 1000 / 1.045 ** 10, "pv20": 1000 / 1.045 ** 20}

# ------------------------------------------------------------ 라: four strategies, dominance and the frontier
# B and A are the shared example. C and D are invented around them (가상의 비교대안).
STRATS = [
    ("표준요법 B", B["cost"], B["qaly"]),
    ("기존약 C", 8100.0, 1.700),
    ("병용요법 D", 9400.0, 2.000),
    ("신약 A", A["cost"], A["qaly"]),
]


def frontier(strats):
    """strats: list of (name, cost, effect). Returns rows sorted by cost with status and ICER on the frontier."""
    rows = [{"name": n, "cost": c, "eff": e, "status": "", "icer": None, "vs": None} for n, c, e in strats]
    rows.sort(key=lambda d: (d["cost"], -d["eff"]))
    # 1) strong dominance: another strategy costs no more and is at least as effective
    for d in rows:
        for o in rows:
            if o is not d and o["cost"] <= d["cost"] and o["eff"] >= d["eff"]:
                d["status"] = "열등"
    # 2) extended dominance: repeat until ICERs increase along the list
    while True:
        live = [d for d in rows if d["status"] == ""]
        icers = [(live[i]["cost"] - live[i - 1]["cost"]) / (live[i]["eff"] - live[i - 1]["eff"]) for i in range(1, len(live))]
        bad = [i for i in range(len(icers) - 1) if icers[i] > icers[i + 1]]
        if not bad:
            break
        live[bad[0] + 1]["status"] = "확장 열등"
    for i in range(1, len(live)):
        live[i]["icer"], live[i]["vs"] = icers[i - 1], live[i - 1]["name"]
    return rows


rows = frontier(STRATS)
for d in rows:
    d["nmb"] = LAM * d["eff"] - d["cost"]
N["strats"] = rows
c = {n: (cc, e) for n, cc, e in STRATS}
# naive sequential ICERs before removing extended dominance
N["seq"] = {
    "D_vs_B": (c["병용요법 D"][0] - c["표준요법 B"][0]) / (c["병용요법 D"][1] - c["표준요법 B"][1]),
    "A_vs_D": (c["신약 A"][0] - c["병용요법 D"][0]) / (c["신약 A"][1] - c["병용요법 D"][1]),
    "A_vs_B": (c["신약 A"][0] - c["표준요법 B"][0]) / (c["신약 A"][1] - c["표준요법 B"][1]),
    "dD_cost": c["병용요법 D"][0] - c["표준요법 B"][0], "dD_eff": c["병용요법 D"][1] - c["표준요법 B"][1],
    "dAD_cost": c["신약 A"][0] - c["병용요법 D"][0], "dAD_eff": c["신약 A"][1] - c["병용요법 D"][1],
}
# the mix of B and A that gives D's QALYs
f = (c["병용요법 D"][1] - c["표준요법 B"][1]) / (c["신약 A"][1] - c["표준요법 B"][1])
N["mix"] = {"f": f, "cost": c["표준요법 B"][0] + f * (c["신약 A"][0] - c["표준요법 B"][0]),
            "saving": c["병용요법 D"][0] - (c["표준요법 B"][0] + f * (c["신약 A"][0] - c["표준요법 B"][0]))}

# ------------------------------------------------------------ 라: negative ICERs, ratios
N["neg"] = {"dominant": (-500, 0.2, -500 / 0.2), "dominated": (500, -0.2, 500 / -0.2),
            "sw": (-1000, -0.1, -1000 / -0.1), "sw_nmb": LAM * -0.1 - (-1000)}
# mean of ratios vs ratio of means (two subgroups of equal size, 가상의 숫자)
g = [(3000.0, 0.60), (2800.0, 0.10)]
N["ratio"] = {"icers": [a / b for a, b in g], "mean_of_ratios": float(np.mean([a / b for a, b in g])),
              "ratio_of_means": float(np.mean([a for a, _ in g]) / np.mean([b for _, b in g])),
              "mc": float(np.mean([a for a, _ in g])), "me": float(np.mean([b for _, b in g]))}

# ------------------------------------------------------------ practice questions
pr = {}
# 가-계산: 200 patients, extra cost and displaced QALYs under an assumed 3,000만원/QALY productivity of displaced care
pr["q1"] = {"n": 200, "extra": 200 * 1500, "gain": 200 * 0.25, "displaced": 200 * 1500 / 5000, "icer": 1500 / 0.25}
# 나-계산: CEA and CUA from the same data (hypothetical antihypertensive-like example avoided; use oncology numbers)
pr["q2"] = {"dc": 1800.0, "dly": 0.60, "dq": 0.40, "icer_ly": 1800 / 0.60, "icer_q": 1800 / 0.40}
# 다-계산: 500만원 a year for 3 years, paid at the end of years 1-3, discounted at 4.5%
pv = [500 / 1.045 ** k for k in (1, 2, 3)]
pr["q3"] = {"pv": pv, "sum": sum(pv), "undisc": 1500.0, "q_10y": 1 / 1.045 ** 10, "q_10y_3": 1 / 1.03 ** 10}
# 라-계산: three strategies
P3 = frontier([("P", 1000.0, 1.00), ("Q", 1600.0, 1.10), ("R", 2200.0, 1.50)])
pr["q4"] = {"rows": P3, "Q_vs_P": 600 / 0.10, "R_vs_Q": 600 / 0.40, "R_vs_P": 1200 / 0.50,
            "acer_R": 2200 / 1.5}
# 라-해석: NMB at different thresholds for the shared example
pr["q5"] = {str(k): k * r["d_qaly"] - r["d_cost"] for k in (3000, 5000, 6000, 7000)}
N["practice"] = pr

# ------------------------------------------------------------ paper box (라): rounded table, consistency check
tab = {"cost_A": round(A["cost"]), "cost_B": round(B["cost"]),
       "lyd_A": round(A["ly_d"], 2), "lyd_B": round(B["ly_d"], 2),
       "q_A": round(A["qaly"], 2), "q_B": round(B["qaly"], 2)}
N["paper"] = tab

# costs in thousand KRW for the journal-style table: check that the rounded table is internally consistent
KEYS = ("c_drug", "c_pf", "c_pd", "c_death", "c_ae", "cost")
tk = {k: (round(A[k] * 10), round(B[k] * 10), round((A[k] - B[k]) * 10)) for k in KEYS}
for k, (a, b, d) in tk.items():
    assert a - b == d, k
assert sum(tk[k][0] for k in KEYS[:-1]) == tk["cost"][0] and sum(tk[k][1] for k in KEYS[:-1]) == tk["cost"][1]
ly3 = (round(A["ly_d"], 3), round(B["ly_d"], 3), round(r["d_ly"], 3))
q3 = (round(A["qaly"], 3), round(B["qaly"], 3), round(r["d_qaly"], 3))
assert abs(ly3[0] - ly3[1] - ly3[2]) < 1e-9 and abs(q3[0] - q3[1] - q3[2]) < 1e-9
tab.update({"thousand": tk, "ly3": ly3, "q3": q3,
            "icer_q_million": r["icer"] / 100, "icer_ly_million": r["icer_ly"] / 100,
            "icer_q_from_table": tk["cost"][2] / q3[2] / 1000, "icer_ly_from_table": tk["cost"][2] / ly3[2] / 1000,
            "share_drug": (A["c_drug"] - B["c_drug"]) / r["d_cost"],
            "ly_pf": (A["ly_pf"], B["ly_pf"]), "ly_pd": (A["ly_pd"], B["ly_pd"])})

# ------------------------------------------------------------ code shown in the chapter (run here, output pasted)
DEMO_DISC = """import numpy as np

r = 0.045                                  # 연 할인율 4.5%
years = np.array([1, 5, 10, 20])           # 몇 년 뒤인가
factor = 1 / (1 + r) ** years              # 할인계수
print(np.round(factor, 3))
print(np.round(1000 * factor))             # 1,000만원의 현재가치

cost = np.array([500, 500, 500])           # 1, 2, 3년 뒤에 드는 비용(만원)
t = np.array([1, 2, 3])
pv = cost / (1 + r) ** t                   # 해마다 따로 할인
print(cost.sum(), np.round(pv, 1), round(pv.sum(), 1))
"""
DEMO_FRONT = """import numpy as np

names = ["표준요법 B", "기존약 C", "병용요법 D", "신약 A"]
cost = np.array([7689.5, 8100.0, 9400.0, 10561.8])   # 총비용(만원)
qaly = np.array([1.87364, 1.70000, 2.00000, 2.38467]) # 총 QALY

keep = []                                    # 효율 경계에 남는 대안의 번호
for i in np.argsort(cost):                   # 비용이 낮은 것부터 차례로
    if any(qaly[j] >= qaly[i] for j in keep):
        print(names[i], "-> 열등")           # 더 싸면서 효과가 같거나 큰 대안이 있음
        continue
    keep.append(i)
    while len(keep) >= 3:                    # 이어진 세 대안의 ICER 순서를 확인
        a, b, c = keep[-3:]
        icer_ab = (cost[b] - cost[a]) / (qaly[b] - qaly[a])
        icer_bc = (cost[c] - cost[b]) / (qaly[c] - qaly[b])
        if icer_ab <= icer_bc:
            break
        print(names[b], "-> 확장 열등", round(icer_ab), ">", round(icer_bc))
        keep.pop(-2)                         # 가운데 대안을 뺌

for a, b in zip(keep[:-1], keep[1:]):
    icer = (cost[b] - cost[a]) / (qaly[b] - qaly[a])
    print(names[b], "대", names[a], "ICER", round(icer), "만원/QALY")
"""
import io, contextlib
N["demo"] = {}
for nm, code in (("disc", DEMO_DISC), ("front", DEMO_FRONT)):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {})
    N["demo"][nm] = {"code": code, "out": buf.getvalue()}

# ------------------------------------------------------------ small derived values quoted in the text
N["text"] = {
    "qaly_per_ly": r["d_qaly"] / r["d_ly"],                       # 늘어난 생존 1년의 평균 QALY
    "hz3_vs_base": hz[3]["icer"] / r["icer"] - 1,                 # 3년 분석기간의 ICER가 기준 분석보다 몇 % 높은가
    "disc_dcost_drop": 1 - r["d_cost"] / dr["0.0"]["d_cost"],     # 할인으로 증분비용이 줄어드는 비율
    "disc_dqaly_drop": 1 - r["d_qaly"] / dr["0.0"]["d_qaly"],
    "cea_paper3": 31 / 0.05,                                      # 나 절 논문 상자: 31만원 ÷ 0.05건
    "abstract_icer": (105.6 - 76.9) / 0.511,                      # 가 절 초록: 백만 원 단위로 반올림한 값의 검산
    "q3_diff": 1500 - sum(pv),
    "q2_ratio": 0.40 / 0.60,
}

json.dump(N, open(os.path.join(HERE, "_ch20_nums.json"), "w"), ensure_ascii=False, indent=1, default=float)

if __name__ == "__main__":
    b = N["base"]
    print("== base case")
    print(f"cost A {b['cost_A']:.2f} B {b['cost_B']:.2f} d {b['d_cost']:.2f}")
    print(f"LY undisc A {b['ly_A']:.4f} B {b['ly_B']:.4f} d {b['d_ly_undisc']:.4f} | LY disc A {b['lyd_A']:.4f} B {b['lyd_B']:.4f} d {b['d_lyd']:.4f}")
    print(f"QALY A {b['qaly_A']:.4f} B {b['qaly_B']:.4f} d {b['d_qaly']:.4f}")
    print(f"ICER {b['icer']:.1f} /QALY, {b['icer_ly']:.1f} /LY, from rounded {b['icer_from_rounded']:.1f}; NMB {b['nmb']:.1f}, NHB {b['nhb']:.4f}")
    for k in ("c_drug", "c_pf", "c_pd", "c_death", "c_ae"):
        print(f"  {k}: A {b[k + '_A']:.1f} B {b[k + '_B']:.1f} d {b['d_' + k]:.1f}")
    print(f"average ratio A {b['acer_A']:.1f} B {b['acer_B']:.1f}; benefit in money {b['benefit_money']:.1f}; price at threshold {b['price_at_thr']:.1f}")
    print("== budget", N["budget"])
    print("== opportunity", N["opp"])
    print("== horizon")
    for y, v in N["horizon"].items():
        print(f"  {y}y: dC {v['d_cost']:.1f} dQ {v['d_qaly']:.4f} ICER {v['icer']:.0f} | A {v['cost_A']:.0f}/{v['qaly_A']:.3f} B {v['cost_B']:.0f}/{v['qaly_B']:.3f}")
    print("  alive", N["alive"])
    print("== discount factors")
    for k, v in N["disc_factor"].items():
        print(" ", k, {a: round(x, 4) for a, x in v.items()})
    for k, v in N["disc_run"].items():
        print(f"  r={k}: cost A {v['cost_A']:.1f} B {v['cost_B']:.1f} d {v['d_cost']:.1f} | QALY A {v['qaly_A']:.4f} B {v['qaly_B']:.4f} d {v['d_qaly']:.4f} | ICER {v['icer']:.1f}")
    print("  cost-only discount ICER", N["disc_costonly"], N["pv_example"])
    print("== strategies")
    for d in N["strats"]:
        print(f"  {d['name']}: cost {d['cost']:.1f} eff {d['eff']:.4f} status '{d['status']}' icer {d['icer']} vs {d['vs']} nmb {d['nmb']:.1f}")
    print("  seq", N["seq"]); print("  mix", N["mix"])
    print("== neg", N["neg"]); print("== ratio", N["ratio"])
    print("== practice")
    for k, v in N["practice"].items():
        print(" ", k, v)
    print("== paper", N["paper"])
    for nm in ("disc", "front"):
        print("== demo", nm); print(N["demo"][nm]["out"])


def check_html():
    """Every number quoted in content/ch20.html that comes from the model must match this script."""
    path = os.path.join(HERE, "..", "content", "ch20.html")
    if not os.path.exists(path):
        return
    h = open(path, encoding="utf-8").read()
    c = lambda v: f"{v:,.0f}"
    b = N["base"]; t = N["text"]; S4 = {d["name"]: d for d in N["strats"]}
    want = [
        c(b["cost_A"]), c(b["cost_B"]), c(b["d_cost"]), f"{b['ly_A']:.3f}", f"{b['ly_B']:.3f}", f"{b['qaly_A']:.3f}", f"{b['qaly_B']:.3f}",
        f"{b['d_qaly']:.3f}", c(b["icer"]), c(b["icer_ly"]), f"{b['d_lyd']:.3f}", f"−{c(-b['nmb'])}만원", f"−{-b['nhb']:.3f} QALY",
        c(b["acer_A"]), c(b["acer_B"]), c(b["benefit_money"]) + "만원", f"평균 {t['qaly_per_ly']:.2f} QALY",
        f"{N['budget']['qX']:.1f} QALY", f"{N['budget']['qY']:.1f} QALY",
        f"{N['opp']['gain']:.1f} QALY", f"{N['opp']['displaced']:.1f} QALY", f"{-N['opp']['net']:.1f} QALY", f"{N['opp']['extra_cost'] / 10000:.1f}억 원",
        f"{t['hz3_vs_base'] * 100:.0f}% 높습니다", f"증분비용은 {t['disc_dcost_drop'] * 100:.0f}%, 증분 QALY는 {t['disc_dqaly_drop'] * 100:.0f}%",
        f"{N['alive']['os_A'][0] * 100:.1f}%", f"{N['alive']['os_B'][0] * 100:.1f}%",
        c(N["disc_costonly"]["icer"]) + "만원/QALY", c(dr["0.0"]["cost_A"]) + "만원", f"{dr['0.0']['qaly_A']:.3f}",
        c(N["seq"]["D_vs_B"]), c(N["seq"]["A_vs_D"]), f"{N['mix']['f'] * 100:.1f}%", c(N["mix"]["cost"]) + "만원", c(N["mix"]["saving"]) + "만원",
        c(S4["표준요법 B"]["nmb"]), c(S4["신약 A"]["nmb"]), c(S4["기존약 C"]["nmb"]), c(S4["병용요법 D"]["nmb"]),
        c(N["ratio"]["mean_of_ratios"]), c(N["ratio"]["ratio_of_means"]),
        f"{tab['icer_q_million']:.1f}", f"{tab['icer_ly_million']:.1f}", c(tab["thousand"]["cost"][2] / tab["q3"][2]) + "천 원",
        f"{tab['share_drug'] * 100:.0f}%", f"{t['cea_paper3']:.0f}만원", f"{t['abstract_icer']:.1f}",
        f"{pr['q3']['sum']:,.1f}만원", f"{t['q3_diff']:.1f}만원", f"{pr['q3']['q_10y']:.3f}", f"{pr['q3']['q_10y_3']:.3f}", f"{t['q2_ratio']:.2f} QALY",
        c(pr["q4"]["Q_vs_P"]), c(pr["q4"]["R_vs_Q"]), c(pr["q4"]["R_vs_P"]), c(pr["q4"]["acer_R"]) + "만원",
        f"−{c(-pr['q5']['3000'])}만원", f"+{c(pr['q5']['6000'])}만원", f"+{c(pr['q5']['7000'])}만원",
        c(pr["q1"]["extra"]), f"{pr['q1']['displaced']:.0f} QALY", c(pr["q2"]["icer_ly"]), c(pr["q2"]["icer_q"]),
    ]
    for y in ("3", "5", "10", "20"):
        v = N["horizon"][y] if y in N["horizon"] else hz[int(y)]
        want += [f'<td class="r">{c(v["d_cost"])}</td><td class="r">{v["d_qaly"]:.3f}</td><td class="r">{c(v["icer"])}</td>']
    for k in ("0.0", "0.03", "0.045"):
        v = dr[k]
        want += [f'<td class="r">{c(v["d_cost"])}</td><td class="r">{v["d_qaly"]:.3f}</td><td class="r">{c(v["icer"])}</td>']
    for y in ("1", "5", "10", "20"):
        f = disc["0.045"][y]
        want += [f'<td class="r">{f:.3f}</td><td class="r">{1000 * f:.0f}만원</td><td class="r">{f:.3f}</td>']
    for k in KEYS:
        a_, b_, d_ = tab["thousand"][k]
        neg = lambda v: f"{v:,}".replace("-", "−")
        want += [f"{a_:,}", f"{b_:,}", neg(d_)]
        want += [f"{A[k]:,.1f}", f"{B[k]:,.1f}", f"{A[k] - B[k]:,.1f}".replace("-", "−")]
    for nm in ("disc", "front"):
        import html as _html
        out = _html.escape(N["demo"][nm]["out"].strip(), quote=False)
        assert out in h, ("demo output not in html", nm)
        for line in N["demo"][nm]["code"].strip().splitlines():
            assert _html.escape(line, quote=False).rstrip() in h, ("demo code line not in html", line)
    miss = [w for w in want if w not in h]
    print("html check:", len(want), "strings,", "missing:", miss if miss else "none")


if __name__ == "__main__":
    check_html()
