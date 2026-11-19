"""R과 파이썬으로 하는 통계분석 25장 · 입력값 표, 관찰 기간의 결과, 재정영향 — R cells run for real with rkit.

run:  python3 gen/r_rc25.py      (NOMARK=1 python3 gen/r_rc25.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 25장의 '직접 분석' 상자와 '장 마무리'에 있던 파이썬 코드(분석기간을 바꾼 분할생존모형, 연차별 비용과 재정영향,
입력값 표와 결과표, 반올림)를 2026-11-19 이 장으로 옮기며 R로 다시 돌린다. 자료 파일은 없고, 공통 예시 모형의 입력값을 코드에 적는다.
난수를 쓰는 셀은 없으므로 모든 숫자가 25장 본문과 같아야 한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc25")

# ------------------------------------------------------------------ a. 입력값 표와 결과표
c_model = nb.cell('''
p <- list(pfs_med = 10, pfs_gam = 0.95, os_med = 28, os_gam = 1.15,   # 표준요법 B의 곡선(중앙값 개월, 모양)
          hr_pfs = 0.65, hr_os = 0.75,                                  # 신약 A의 위험비
          u_pf = 0.78, u_pd = 0.62, du_ae_A = 0.012, du_ae_B = 0.008,   # 효용과 이상반응의 QALY 손실
          c_drug_A = 190, c_drug_B = 120, c_pf = 40, c_pd = 250,        # 월 비용(만원)
          c_death = 800, c_ae_A = 120, c_ae_B = 80, disc = 0.045)       # 1회 비용(만원), 연 할인율
model <- function(p, horizon = 240, disc = TRUE) {    # 분석기간(개월)과 할인 여부를 인자로
  t <- 0:horizon
  d <- if (disc) 1 / (1 + p$disc)^((head(t, -1) + 0.5) / 12) else 1   # 각 달의 한가운데에서 할인
  sapply(c(A = "A", B = "B"), function(arm) {
    hp <- if (arm == "A") p$hr_pfs else 1; ho <- if (arm == "A") p$hr_os else 1
    s_os  <- exp(-log(2) * ho * (t / p$os_med)^p$os_gam)
    s_pfs <- pmin(exp(-log(2) * hp * (t / p$pfs_med)^p$pfs_gam), s_os)
    pf <- (head(s_pfs, -1) + tail(s_pfs, -1)) / 2                     # 반주기 보정
    pd <- (head(s_os - s_pfs, -1) + tail(s_os - s_pfs, -1)) / 2
    died <- head(s_os, -1) - tail(s_os, -1)
    c_drug <- p[[paste0("c_drug_", arm)]]
    c(cost = sum((pf * (c_drug + p$c_pf) + pd * p$c_pd + died * p$c_death) * d) + p[[paste0("c_ae_", arm)]],
      qaly = sum((pf * p$u_pf + pd * p$u_pd) / 12 * d) - p[[paste0("du_ae_", arm)]],
      drug = sum(pf * c_drug * d),                    # 1차 약값
      pf_months = sum(pf), pd_months = sum(pd))       # 상태에 머문 기간(개월, 할인하지 않음)
  })
}
r <- model(p)
round(r, 3)
''', title="분석기간을 인자로 둔 공통 예시 모형")

c_tables = nb.cell('''
src <- c(pfs_med = "Pivotal trial, arm B", os_med = "Pivotal trial, arm B", hr_pfs = "Pivotal trial",
         hr_os = "Pivotal trial", u_pf = "Trial EQ-5D-5L, Korean value set", u_pd = "Trial EQ-5D-5L, Korean value set",
         c_drug_A = "List price", c_pd = "Claims data", disc = "HIRA guideline")
table1 <- data.frame(base_case = unlist(p[names(src)]), source = src)   # 사전의 값을 그대로 표로
table1
res <- data.frame(cost = r["cost", ], qaly = r["qaly", ])              # 결과표: 반올림하지 않은 값
res["A - B", ] <- res["A", ] - res["B", ]
res
c(icer = res["A - B", "cost"] / res["A - B", "qaly"],                  # 반올림 전 값으로 나눔
  from_table = round(res["A - B", "cost"]) / round(res["A - B", "qaly"], 3))   # 표에 적힌 2,872 / 0.511
''', title="입력값 표와 결과표를 모형의 값에서 만들기")

c_round = nb.cell('''
round(c(2.5, 3.5, 0.5))              # 정확히 반인 값은 짝수 쪽으로
round(56.25, 1); round(0.125, 2)
sprintf("%.1f", 56.25)               # 글자로 바꿀 때도 같음
half_up <- function(x, digits = 0) floor(x * 10^digits + 0.5) / 10^digits   # 엑셀 ROUND처럼 올림
half_up(c(2.5, 56.25, 0.125), c(0, 1, 2))
''', title="반올림 규칙 확인하기")

# ------------------------------------------------------------------ b. 관찰 기간의 결과
c_horizon = nb.cell('''
inc <- function(r) c(d_cost = r["cost", "A"] - r["cost", "B"], d_qaly = r["qaly", "A"] - r["qaly", "B"],
                     icer = (r["cost", "A"] - r["cost", "B"]) / (r["qaly", "A"] - r["qaly", "B"]))
r30 <- model(p, horizon = 30)                         # 시험이 관찰한 30개월에서 끊음
tab <- rbind(years20 = inc(r), months30 = inc(r30))
round(tab, 3)
round(1 - tab["months30", 1:2] / tab["years20", 1:2], 3)   # 30개월 뒤에서 생긴 비율
round(r30[c("pf_months", "pd_months"), ], 1)               # 30개월 안에 각 상태에 머문 평균 기간
''', title="분석기간 20년과 30개월의 증분 결과")

# ------------------------------------------------------------------ c. 재정영향
c_yearly = nb.cell('''
yearly <- function(p, what = "cost") {               # 치료 시작 후 1-5년차의 1인당 값(할인하지 않음)
  cum <- sapply(1:5, function(k) model(p, horizon = 12 * k, disc = FALSE)[what, ])   # 2행(A, B) x 5열
  out <- cum - cbind(0, cum[, -5])                    # k년 누적 - (k-1)년 누적
  colnames(out) <- paste0("year", 1:5); out
}
cy <- yearly(p)
round(rbind(cy, diff = cy["A", ] - cy["B", ]), 1)
''', title="연차별 1인당 비용")

c_bia = nb.cell('''
bia <- function(cA, cB, n = 600, share = c(0.1, 0.2, 0.3, 0.4, 0.5)) {
  start_A <- n * share                                # 급여 후 해마다 신약 A로 시작하는 환자 수
  out <- matrix(0, 3, 5, dimnames = list(c("without_A", "with_A", "net"), paste0("year", 1:5)))
  for (y in 1:5) for (s in 1:y) {                     # 급여 y년차에 치료 중인 s년 코호트
    k <- y - s + 1                                    # 그 코호트의 치료 연차
    out["without_A", y] <- out["without_A", y] + n * cB[k]
    out["with_A", y]    <- out["with_A", y] + start_A[s] * cA[k] + (n - start_A[s]) * cB[k]
    out["net", y]       <- out["net", y] + start_A[s] * (cA[k] - cB[k])
  }
  cbind(out, total = rowSums(out)) / 10000            # 억 원
}
round(bia(cy["A", ], cy["B", ]), 1)
dy <- yearly(p, "drug")                               # 1차 약값만
round(bia(dy["A", ], 0 * dy["B", ])["net", ], 1)      # 신약 A 약품비 총액
''', title="5년 재정영향 (표 25-8)")

c_scen = nb.cell('''
price <- uniroot(function(x) { r <- model(modifyList(p, list(c_drug_A = x)))
  inc(r)[["icer"]] - 5000 }, c(150, 190), tol = 1e-9)$root   # ICER가 5,000이 되는 월 약값
price
cy_p <- yearly(modifyList(p, list(c_drug_A = price)))
sc <- list(base        = bia(cy["A", ], cy["B", ]),
           share_half  = bia(cy["A", ], cy["B", ], share = c(0.05, 0.10, 0.15, 0.20, 0.25)),
           n_900       = bia(cy["A", ], cy["B", ], n = 900),
           price_5000  = bia(cy_p["A", ], cy_p["B", ]),
           drug_only   = bia(dy["A", ], dy["B", ]))
round(t(sapply(sc, function(b) b["net", c("year3", "total")])), 1)
round(cy_p["A", 1] - cy_p["B", 1], 1)                 # 그 가격에서 치료 1년차의 1인당 비용 차이
''', title="가정을 바꾼 재정영향")

# ------------------------------------------------------------------ d. 과제
c_t1 = nb.cell('''
refund <- c(0, 0.05, 1 - price / 190, 0.10)           # 환급률. 셋째는 ICER가 5,000이 되는 7.7%
real <- 190 * (1 - refund)                            # 실제로 내는 월 약값
icer_real <- sapply(real, function(x) inc(model(modifyList(p, list(c_drug_A = x))))[["icer"]])
data.frame(refund = round(refund, 3), real = round(real, 1), icer_list = round(inc(r)[["icer"]]), icer_real = round(icer_real))
''', title="과제 1 정답. 환급률에 따른 ICER (표 25-7)")

c_t2 = nb.cell('''
r18 <- model(p, horizon = 18)                         # 중앙 추적 기간 18개월에서 끊으면
tab2 <- rbind(tab, months18 = inc(r18))
round(tab2, 3)
round(1 - sweep(tab2[c("months30", "months18"), 1:2], 2, tab2["years20", 1:2], "/"), 3)   # 끊은 뒤에서 생긴 비율
''', title="과제 2 정답. 어디서 끊느냐에 따른 관찰 기간만의 결과")

nb.run()

if NOMARK:
    for c in nb.cells:
        print(f"===== cell {c.n} {c.title}")
        print(c.text)
        if c.warns: print("WARN", c.warns)
        if c.error: print("ERROR", c.error)
    sys.exit()

out = {c.n: c.text for c in nb.cells}
for c in nb.cells:
    assert not c.error, (c.n, c.error)
CHK = {
    c_model: ['10561.778', '7689.485', '2.385', '1.874'],
    c_tables: ['2872.29', '5620.61', '5620.35'],
    c_round: ['[1] 2 4 0', '[1] 56.2', '[1] 0.12', '"56.2"', '[1]  3.00 56.30  0.13'],
    c_horizon: ['2872.294', '0.511', '5620.610', '1156.855', '0.128', '9055.886', '0.597', '0.750'],
    c_yearly: ['2820.7', '2226.8', '1763.1', '1366.9', '1044.3', '2200.1', '1821.0', '1398.5', '1021.5', '723.8', '620.6', '320.4'],
    c_bia: ['132.0', '135.7', '1514.8', '1616.3', '3.7', '9.9', '18.2', '28.6', '41.0', '101.5', '10.5', '252.4'],
    c_scen: ['175.3897', '9.1', '50.7', '27.3', '152.2', '14.5', '82.1', '24.0', '129.2', '485.8'],
    c_t1: ['5217', '5000', '4814', '175.4'],
    c_t2: ['14311.648', '0.718', '0.889'],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])
print("checks ok")

S = nb.save_fragment
S("rc25_model", nb.html(c_model, marks={'cost      10561.778 7689.485': 1, 'pf_months    23.690   15.059': 2}))
S("rc25_tables", nb.html(c_tables, marks={'disc         0.045                   HIRA guideline': 1, 'A - B  2872.294 0.5110288': 2, '5620.610   5620.352': 3}))
S("rc25_round", nb.html(c_round, marks={'[1] 2 4 0': 1, '[1] 0.12': 2, '[1]  3.00 56.30  0.13': 3}))
S("rc25_horizon", nb.html(c_horizon, marks={'months30 1156.855  0.128 9055.886': 1, ' 0.597  0.750': 2, 'pd_months  6.8  8.9': 3}))
S("rc25_yearly", nb.html(c_yearly, marks={'B    2200.1 1821.0 1398.5 1021.5  723.8': 1, 'diff  620.6  405.8  364.6  345.4  320.4': 2}))
S("rc25_bia", nb.html(c_bia, marks={'net         3.7   9.9  18.2  28.6  41.0  101.5': 1, ' 10.5  27.3  48.0  71.0  95.5 252.4': 2}))
S("rc25_scen", nb.html(c_scen, marks={'[1] 175.3897': 1, 'price_5000  14.5  82.1': 2, 'drug_only   24.0 129.2': 3, '485.8': 4}))
S("rc25_t1", nb.html(c_t1))
S("rc25_t2", nb.html(c_t2))
nb.save_ipynb("25장 입력값 표, 관찰 기간의 결과, 재정영향 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 25장의 공통 예시(진행성 신세포암, 신약 A 대 표준요법 B)로 입력값 표와 결과표, 분석기간을 시험의 관찰 기간으로 줄인 결과, "
                    "연차별 1인당 비용과 5년 재정영향을 R로 계산합니다. 자료 파일은 없고 모형의 입력값을 코드에 적습니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. 추가 패키지는 쓰지 않습니다.",
              notes={c_model.n: "## a. 입력값 표와 결과표", c_horizon.n: "## b. 관찰 기간의 결과", c_yearly.n: "## c. 재정영향",
                     c_t1.n: "## d. 과제 정답"})
print("rc25 fragments and notebook written")
