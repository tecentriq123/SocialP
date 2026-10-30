"""R과 통계분석 5장 자료: pub/data/rcc_km_digitized.csv (논문 그림에서 읽어 낸 것으로 가정한 Kaplan-Meier 좌표).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_rc05.py

23장 d 절의 가상 시험 자료(pub/data/rcc_trial.csv, 표준요법 B군 300명)의 Kaplan-Meier 곡선을 '논문 그림'으로 보고,
디지타이저(예를 들어 WebPlotDigitizer)로 계단의 모서리를 찍어 좌표를 읽은 상황을 흉내 낸다.
  - 사건 시점마다 계단의 위 모서리(내려가기 직전)와 아래 모서리(내려간 뒤) 두 점
  - 그림에서 작은 계단을 놓친 것처럼 모서리 쌍의 25%를 뺀다
  - 읽기 오차: 시간 표준편차 0.08개월, 생존율 표준편차 0.003 (그림 한 칸의 1–2 픽셀 정도)
  - 시작점 (0, 1)은 그대로 둔다. 오차 때문에 순서가 뒤바뀐 점도 그대로 둔다(R의 전처리가 정리한다)
난수 시드를 고정했으므로 다시 돌리면 같은 파일이 나온다.
  - time : 개월 (소수 셋째 자리)
  - surv : 생존 비율 0–1 (소수 넷째 자리)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "pub", "data", "rcc_trial.csv")
OUT = os.path.join(ROOT, "pub", "data", "rcc_km_digitized.csv")


def km(time, event):
    """Kaplan-Meier: 사건 시점과 그 직후의 생존율"""
    t_ev = np.unique(time[event == 1])
    s, out = 1.0, []
    for t in t_ev:
        n = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        s *= 1 - d / n
        out.append((t, s))
    return np.array(out)


def build():
    d = pd.read_csv(SRC)
    steps = km(d["months"].to_numpy(), d["event"].to_numpy())
    assert len(steps) == 145
    rng = np.random.default_rng(2026)
    prev = np.concatenate([[1.0], steps[:-1, 1]])
    keep = rng.random(len(steps)) >= 0.25                      # 놓친 계단
    pts = []
    for (t, s), sp, k in zip(steps, prev, keep):
        if k:
            pts.append((t, sp))                                  # 위 모서리
            pts.append((t, s))                                   # 아래 모서리
    pts = np.array(pts)
    pts[:, 0] += rng.normal(0, 0.08, len(pts))
    pts[:, 1] += rng.normal(0, 0.003, len(pts))
    pts[:, 0] = np.clip(pts[:, 0], 0.0, None)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.0)
    end = d["months"].max()                                       # 곡선의 오른쪽 끝(마지막 중도절단)
    pts = np.vstack([[0.0, 1.0], pts, [end + rng.normal(0, 0.08), steps[-1, 1] + rng.normal(0, 0.003)]])
    out = pd.DataFrame({"time": pts[:, 0].round(3), "surv": pts[:, 1].round(4)})
    assert out.iloc[0].tolist() == [0.0, 1.0]
    assert len(out) == 240
    return out


if __name__ == "__main__":
    out = build()
    out.to_csv(OUT, index=False, encoding="utf-8", lineterminator="\n")
    print(OUT, out.shape)
    print(out.head(8).to_string(index=False))
    print("non-monotone pairs:", int((np.diff(out["surv"]) > 0).sum()), " time reversals:", int((np.diff(out["time"]) < 0).sum()))
