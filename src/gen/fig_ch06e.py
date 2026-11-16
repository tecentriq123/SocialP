"""Figure 6-8: ANCOVA in the pharmacist-counseling HbA1c trial (6장 e 절)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from svgplot import Plot, figure
import nums_ch06e as E

OUT = os.path.join(os.path.dirname(__file__), "..", "figs")
C = E.ols([E.arm, E.base], E.post, ['int', 'arm', 'base'])
b0, b1, b2 = C['_']['b']
mb = E.base.mean()
p = Plot((6.5, 10.75), (5.5, 9.75), w=600, h=420, xlabel="기저 HbA1c (%)", ylabel="6개월 HbA1c (%)",
         xticks=[7, 8, 9, 10], yticks=[6, 7, 8, 9])
p.points(E.base[E.arm == 0], E.post[E.arm == 0], s=2, r=3.4)
p.points(E.base[E.arm == 1], E.post[E.arm == 1], s=1, r=3.4)
xx = np.array([6.6, 10.6])
p.line(xx, b0 + b2 * xx, s=2, w=2.6)
p.line(xx, b0 + b1 + b2 * xx, s=1, w=2.6)
ls0, ls1 = b0 + b2 * mb, b0 + b1 + b2 * mb
p.vline(mb, dash=True)
p.seg(mb, ls1, mb, ls0, cls="ln s4", w=3)
p.points([mb, mb], [ls0, ls1], s=4, r=4.5)
p.text(mb, ls1, f"보정한 차이 {b1:.2f}%p".replace("-", "−"), anchor="start", dx=10, dy=22, cls="lbl strong small")
p.text(mb, 9.75, f"기저 평균 {mb:.2f}%", anchor="middle", dy=-6, cls="lbl mute small")
p.legend([("일반 진료군 (n = 36)", 2, "dot"), ("복약상담군 (n = 36)", 1, "dot")], X=p.ml + 10, Y=p.mt + 10)
html = figure(p.svg("기저 HbA1c와 6개월 HbA1c, 공분산분석의 두 평행선"),
              f"그림 6-8. 2장 a 절의 복약상담 시험 72명(가상 자료). 두 실선은 공분산분석으로 맞춘 군별 직선이며, 기울기 {b2:.2f}을 두 군이 함께 쓰므로 평행합니다. "
              f"두 직선의 세로 간격이 기저값을 보정한 차이이고, 기저 HbA1c 전체 평균({mb:.2f}%)에서 읽은 두 직선의 높이({ls1:.2f}%와 {ls0:.2f}%)가 군별 최소제곱 평균입니다.")
open(os.path.join(OUT, "ch06_ancova.html"), "w").write(html)
print("ok", round(b1, 4), round(b2, 4), round(mb, 3), round(ls1, 3), round(ls0, 3))
