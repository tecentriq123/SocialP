"""Figure 14-6: reporting odds ratio of diabetic ketoacidosis by comparison background (14장 e 절, hypothetical database).
Run: python3 gen/fig_ch14e.py   (writes figs/ch14_disp.html)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from svgplot import figure, forest
from nums_ch14e import RES

rows = []
for bg, head in (("all", "전체 보고와 비교 (100만 건)"), ("dm", "당뇨병약 보고와 비교 (5만 건)")):
    rows.append({"label": head, "header": True})
    for drug, lab, s in (("SGLT2", "SGLT2 억제제", 1), ("DPP4", "DPP-4 억제제", 2)):
        m = RES[(drug, bg)]
        rows.append({"label": lab, "est": m["ror"], "lo": m["ror_lo"], "hi": m["ror_hi"], "indent": True, "s": s,
                     "nd": 1 if m["ror"] >= 10 else 2})
svg = forest(rows, xlim=(0.4, 60), ref=1.0, log=True, w=640, row_h=30, label_w=230, est_w=150,
             xlabel="당뇨병케톤산증의 보고 오즈비 (95% CI, 로그 척도)", xticks=[0.5, 1, 2, 5, 10, 20, 50])
html = figure(svg,
              "그림 14-6. 비교 배경에 따른 당뇨병케톤산증의 보고 오즈비(가상의 보고 자료). 위 두 줄은 자료의 모든 보고를, "
              "아래 두 줄은 당뇨병약이 의심약으로 적힌 보고만을 비교 배경으로 썼습니다. 케톤산증과 관련이 알려지지 않은 "
              "DPP-4 억제제도 전체 보고와 비교하면 11배로 나오고, 같은 병의 약끼리 비교하면 SGLT2 억제제가 35.5에서 3.02로 "
              "내려옵니다. 가로축은 로그 척도입니다.")
open(os.path.join(HERE, "..", "figs", "ch14_disp.html"), "w", encoding="utf-8").write(html)
print("ok")
