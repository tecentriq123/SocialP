"""R과 파이썬으로 하는 통계분석 19장 · 할인, ICER, 순편익, 효율 경계 — R cells run for real with rkit.

run:  python3 gen/r_rc22.py      (NOMARK=1 python3 gen/r_rc22.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 20장의 '직접 분석' 상자와 본문에 있던 파이썬 코드, 파이썬 실습 20의 계산을 2026-11-19 이 장으로 옮기며 R로 다시 돌린다.
자료: 20장의 신세포암 공통 예시에서 모형이 낸 네 대안의 할인한 총비용과 총 QALY(실습 20과 같은 값, gen/_ch20_nums.json).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc22")

# ------------------------------------------------------------------ a. 할인
c_disc = nb.cell('''
r <- 0.045                                   # 연 할인율 4.5%
years  <- c(1, 5, 10, 20)                    # 몇 년 뒤인가
factor <- 1 / (1 + r)^years                  # 할인 계수
round(factor, 3)
round(1000 * factor)                         # 1,000만원의 현재가치
cost <- c(500, 500, 500); t <- 1:3           # 1, 2, 3년 뒤에 드는 비용(만원)
pv <- cost / (1 + r)^t                       # 해마다 따로 할인
c(total = sum(cost), round(pv, 1), pv_sum = round(sum(pv), 1))
''', title="할인 계수와 현재가치 (20장 c 절)")

# ------------------------------------------------------------------ b. ICER와 사분면
c_inputs = nb.cell('''
alts <- data.frame(cost = c(7689.485, 8100, 9400, 10561.778),   # 할인한 총비용(만원)
                   qaly = c(1.873640, 1.700, 2.000, 2.384669),   # 할인한 총 QALY
                   row.names = c("B", "C", "D", "A"))            # 표준요법, 기존약, 병용요법, 신약
alts
''', title="네 대안의 총비용과 총 QALY (20장 d 절)")

c_icer = nb.cell('''
d_cost <- alts["A", "cost"] - alts["B", "cost"]     # 증분비용
d_qaly <- alts["A", "qaly"] - alts["B", "qaly"]     # 증분 QALY
icer   <- d_cost / d_qaly                           # 증분비용효과비
round(c(d_cost = d_cost, d_qaly = d_qaly, icer = icer), 3)
round(alts$cost / alts$qaly)                        # 평균 비(총비용 ÷ 총 QALY): B, C, D, A
ly <- c(A = 3.398772, B = 2.728843)                 # 할인한 생존연수
round(d_cost / unname(ly["A"] - ly["B"]))           # 늘어난 생존연수 1년당 비용
''', title="신약 A 대 표준요법 B의 ICER (20장 d 절)")

c_quad = nb.cell('''
quadrant <- function(dc, dq) {                      # 증분의 부호로 사분면부터 정한다
  if (dq == 0) return("증분 QALY 0: ICER를 정의할 수 없음")
  if (dq > 0 && dc <= 0) return("오른쪽 아래: 우월(dominant)")
  if (dq < 0 && dc >= 0) return("왼쪽 위: 열등(dominated)")
  if (dq > 0) return(sprintf("오른쪽 위: ICER %.1f", dc / dq))
  sprintf("왼쪽 아래: 아낀 비용 ÷ 잃은 QALY %.1f", dc / dq)
}
quadrant(d_cost, d_qaly); quadrant(-500, 0.2); quadrant(500, -0.2); quadrant(-1000, -0.1)
c(dominant = -500 / 0.2, dominated = 500 / -0.2)    # 둘 다 −2500: 비만으로는 구분되지 않음
c(1000 / 0, 0 / 0)                                  # 증분 QALY가 0이면 경고 없이 Inf, NaN
''', title="사분면 판정과 음수 ICER (20장 d 절)")

c_front = nb.cell('''
frontier <- function(df) {
  df <- df[order(df$cost, -df$qaly), ]               # 비용이 낮은 순서, 같으면 효과가 큰 순서
  df$status <- "경계"; df$vs <- ""; df$icer <- NA
  ic <- function(a, b) (df[b, "cost"] - df[a, "cost"]) / (df[b, "qaly"] - df[a, "qaly"])
  keep <- character(0)                               # 경계에 남은 대안
  for (i in rownames(df)) {
    if (any(df[keep, "qaly"] >= df[i, "qaly"])) {    # 더 싸면서 효과가 같거나 큰 대안이 있음
      df[i, "status"] <- "열등"; next
    }
    keep <- c(keep, i)
    while (length(keep) >= 3) {                      # 이어진 세 대안의 ICER 순서 확인
      n <- length(keep); x <- keep[n - 2]; y <- keep[n - 1]; z <- keep[n]
      if (ic(x, y) <= ic(y, z)) break
      cat(y, "확장 열등:", round(ic(x, y)), ">", round(ic(y, z)), "\\n")
      df[y, "status"] <- "확장 열등"; keep <- keep[-(n - 1)]   # 가운데 대안을 뺌
    }
  }
  for (k in seq_along(keep)[-1]) {                   # 경계 위의 이웃끼리 ICER
    df[keep[k], "vs"] <- keep[k - 1]; df[keep[k], "icer"] <- ic(keep[k - 1], keep[k])
  }
  df
}
fr <- frontier(alts); fr$icer <- round(fr$icer); fr
''', title="열등, 확장 열등, 효율 경계 (20장 d 절)")

c_mix = nb.cell('''
f <- (alts["D", "qaly"] - alts["B", "qaly"]) / d_qaly      # D와 같은 QALY가 되도록 A를 쓸 비율
mix_cost <- alts["B", "cost"] + f * d_cost
round(c(share_A = f, mix_qaly = alts["B", "qaly"] + f * d_qaly,
        mix_cost = mix_cost, saving_vs_D = alts["D", "cost"] - mix_cost), 3)
''', title="B와 A를 섞은 전략과 병용요법 D (20장 d 절)")

# ------------------------------------------------------------------ c. 순편익
c_nmb = nb.cell('''
lam <- 5000                                      # 가정한 임계값(만원/QALY)
round(c(nmb = lam * d_qaly - d_cost,             # 증분 순금전편익(만원)
        nhb = d_qaly - d_cost / lam,             # 증분 순건강편익(QALY)
        per_100 = 100 * (d_qaly - d_cost / lam)), 3)
lams <- c(2500, 3000, 5000, 6000, 7000, 8000)
data.frame(lam = lams, nmb = round(lams * d_qaly - d_cost, 1),
           nhb = round(d_qaly - d_cost / lams, 3), ce = lams * d_qaly - d_cost > 0)
''', title="순금전편익과 순건강편익 (20장 d 절)")

c_nmb4 = nb.cell('''
nmb4 <- sapply(c(5000, 6000), function(l) l * alts$qaly - alts$cost)   # 대안마다 λ × QALY − 비용
dimnames(nmb4) <- list(rownames(alts), c("lam5000", "lam6000")); round(nmb4, 1)
rownames(alts)[apply(nmb4, 2, which.max)]               # 순금전편익이 가장 큰 대안
grid <- seq(0, 20000, by = 100)
best <- sapply(grid, function(l) rownames(alts)[which.max(l * alts$qaly - alts$cost)])
table(best); grid[match("A", best)]                     # A가 처음 가장 커지는 임계값
''', title="네 대안의 순금전편익 (20장 d 절)")

# ------------------------------------------------------------------ d. 과제
c_t1 = nb.cell('''
round(c(value = 6000 * d_qaly, cost = d_cost, nmb = 6000 * d_qaly - d_cost,
        nhb = d_qaly - d_cost / 6000), 3)
''', title="과제 1 정답. 임계값 6,000만원의 순편익")

c_t2 = nb.cell('''
alts_E <- rbind(alts, E = c(alts["B", "cost"] + 600, alts["B", "qaly"] + 0.20))
fe <- frontier(alts_E); fe$icer <- round(fe$icer); fe
''', title="과제 2 정답. 신약 E를 더해 다시 고르기")

nb.run()

if NOMARK:
    for c in nb.cells:
        print(f"===== cell {c.n} {c.title}\n{c.text}")
        for w in c.warns:
            print("WARN:", w)
        if c.error:
            print("ERR:", c.error)
    sys.exit(0)

out = {c.n: c.text for c in nb.cells}
for c in nb.cells:
    assert not c.error, (c.n, c.error)
CHK = {
    c_disc: ['0.957 0.802 0.644 0.415', '1374.5'],
    c_inputs: ['10561.778 2.384669'],
    c_icer: ['5620.607', '4104 4765 4700 4429', '4287'],
    c_quad: ['-2500     -2500', 'Inf NaN'],
    c_front: ['13537 > 3020', 'B 5621'],
    c_mix: ['8399.705    1000.295'],
    c_nmb: ['-317.148   -0.063   -6.343', '193.9  0.032  TRUE'],
    c_nmb4: ['"B" "A"', '[1] 5700'],
    c_t1: ['193.881', '0.032'],
    c_t2: ['E 7306', 'B 3000'],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc22_disc", nb.html(c_disc, marks={'0.957 0.802 0.644 0.415': 1, '1374.5': 2}))
S("rc22_inputs", nb.html(c_inputs))
S("rc22_icer", nb.html(c_icer, marks={'5620.607': 1, '4104 4765 4700 4429': 2, '4287': 3}))
S("rc22_quad", nb.html(c_quad, marks={'-2500     -2500': 1, 'Inf NaN': 2}))
S("rc22_front", nb.html(c_front, marks={'13537 > 3020': 1, 'B 5621': 2}))
S("rc22_mix", nb.html(c_mix, marks={'8399.705    1000.295': 1}))
S("rc22_nmb", nb.html(c_nmb, marks={'-317.148   -0.063   -6.343': 1, '193.9  0.032  TRUE': 2}))
S("rc22_nmb4", nb.html(c_nmb4, marks={'"B" "A"': 1, '[1] 5700': 2}))
S("rc22_t1", nb.html(c_t1))
S("rc22_t2", nb.html(c_t2))
nb.save_ipynb("19장 할인, ICER, 순편익, 효율 경계 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 20장의 신세포암 공통 예시(네 대안의 할인한 총비용과 총 QALY)로 할인 계수, ICER와 사분면, 효율 경계, "
                    "순금전편익과 순건강편익을 R로 계산합니다. Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_disc.n: "## a. 할인", c_inputs.n: "## b. ICER, 사분면, 효율 경계", c_nmb.n: "## c. 순편익", c_t1.n: "## d. 과제 정답"})
print("rc22 fragments and notebook written")
