"""R과 파이썬으로 하는 통계분석 21장 · 효용과 QALY — R cells run for real with rkit.

run:  python3 gen/r_rc24.py      (NOMARK=1 python3 gen/r_rc24.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 22장의 '직접 분석' 상자와 본문에 있던 파이썬 코드(가치 세트 함수, statsmodels의 군집 표준오차,
scipy의 trapezoid, 할인 계수)를 2026-11-19 이 장으로 옮기며 R로 다시 돌린다.
자료: 22장의 가상 시험 자료(EQ-5D-5L 응답 1,593건, 환자 320명; pub/data/lab22_eq5d.csv, lab22_patients.csv = 실습 22와 같은 파일).
난수를 쓰는 셀은 없으므로 모든 숫자가 22장 본문과 같아야 한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc24")

# ------------------------------------------------------------------ a. EQ-5D 응답을 효용으로
c_load = nb.cell('''
D <- "https://socialp-ajou.tecentriq12.workers.dev/data/"
df  <- read.csv(paste0(D, "lab22_eq5d.csv"), colClasses = c(eq5d = "character"))   # 응답 코드는 글자로
pts <- read.csv(paste0(D, "lab22_patients.csv"))
df  <- df[order(df$id, df$month), ]                  # 환자, 측정 시점 순서로 정렬
dim(df); dim(pts)
head(df, 3)
table(df$state)
sum(is.infinite(pts$t_prog))                         # 진행 없이 사망한 환자: t_prog가 inf
visits <- c(0, 3, 6, 12, 18, 24)
sum(sapply(pts$t_death, function(d) sum(visits < d)))   # 살아 있던 방문 수 = 응답 수
''', title="설문 응답과 환자 자료 불러오기")

c_kr5l = nb.cell('''
# 한국 EQ-5D-5L 가치 세트(Kim 등, 2016)의 감점. 행 = 영역, 열 = 수준 1-5
KR5 <- rbind(mobility = c(0, 0.046, 0.058, 0.133, 0.251),
             selfcare = c(0, 0.032, 0.050, 0.078, 0.122),
             activity = c(0, 0.021, 0.051, 0.100, 0.175),
             pain     = c(0, 0.042, 0.053, 0.166, 0.207),
             anxiety  = c(0, 0.033, 0.046, 0.102, 0.137))
kr5l <- function(code) {
  lv <- as.integer(strsplit(as.character(code), "")[[1]])   # "21222" -> 2 1 2 2 2
  if (all(lv == 1)) return(1)                                # 완전한 건강은 감점 없음
  u <- 1 - 0.096 - sum(KR5[cbind(1:5, lv)])                  # 상수 감점과 영역별 감점
  if (max(lv) >= 4) u <- u - 0.078                           # 수준 4나 5가 하나라도 있으면
  round(u, 3)
}
sapply(c("21222", "31342", "11211", "55555", "11111"), kr5l)
''', title="한국 EQ-5D-5L 가치 세트 함수")

c_apply = nb.cell('''
df$u_calc <- sapply(df$eq5d, kr5l)                   # 1,593건 모두
length(unique(df$eq5d))                              # 서로 다른 응답 코드
max(abs(df$u_calc - df$utility))                     # 자료의 utility 열과의 차이
sum(df$eq5d == "11111")                              # 효용 1인 응답
df[df$u_calc < 0, c("id", "arm", "month", "state", "eq5d", "u_calc")]
''', title="응답 1,593건을 효용으로 바꾸고 utility 열과 비교하기")

c_pkg = nb.cell('''
suppressPackageStartupMessages(library(eq5d))       # 여러 나라의 가치 세트를 담은 패키지
u_pkg <- eq5d(df$eq5d, country = "SouthKorea", version = "5L", type = "VT")
max(abs(u_pkg - df$u_calc))                          # 직접 짠 함수와 같은가
codes <- c("11121", "21222", "33333")                # 표 22-3. 같은 3L 응답, 두 나라의 가치 세트
t3 <- data.frame(korea = eq5d(codes, country = "SouthKorea", version = "3L", type = "TTO"),
                 uk    = eq5d(codes, country = "UK", version = "3L", type = "TTO"), row.names = codes)
t3$diff <- t3$korea - t3$uk
t3
''', title="eq5d 패키지로 검산하고 3L 가치 세트 두 개 비교하기")

# ------------------------------------------------------------------ b. 상태별 효용과 신뢰구간
c_state = nb.cell('''
suppressPackageStartupMessages({library(sandwich); library(lmtest)})
fit <- lm(utility ~ 0 + state, data = df)           # 절편을 빼면 계수 = 상태별 평균
V   <- vcovCL(fit, cluster = ~ id)                   # 같은 환자의 응답을 한 묶음으로
round(cbind(mean = coef(fit), se_naive = sqrt(diag(vcov(fit))),
            se_cluster = sqrt(diag(V)), coefci(fit, vcov. = V)), 4)
''', title="상태별 평균 효용과 환자 단위로 묶은 신뢰구간")

c_mixed = nb.cell('''
suppressPackageStartupMessages(library(lme4))
round(fixef(lmer(utility ~ 0 + state + (1 | id), data = df)), 3)   # 환자를 무작위효과로
f2 <- lm(utility ~ 0 + state + state:arm, data = df)                # 상태마다 B군 - A군
round(coeftest(f2, vcov. = vcovCL(f2, cluster = ~ id))[3:4, ], 3)
f3 <- lm(utility ~ state * arm, data = df)                          # 상호작용 항
round(coeftest(f3, vcov. = vcovCL(f3, cluster = ~ id))["statePF:armB", ], 3)
''', title="혼합모형과 군별 상태 효용")

c_vas = nb.cell('''
by <- aggregate(cbind(utility, vas_100 = vas / 100) ~ state, data = df, FUN = mean)
rownames(by) <- by$state; by <- by[c("PF", "PD"), ]
round(by[, -1], 3)
round(by[1, -1] - by[2, -1], 3)                      # 무진행 - 진행
''', title="EQ VAS ÷ 100과 효용 비교하기")

# ------------------------------------------------------------------ c. 환자별 QALY
c_one = nb.cell('''
path <- function(month, u, t_death, end = 24) {      # 사망 시점에 효용 0을 붙이고, 추적 끝까지 0
  if (t_death <= end) {
    month <- c(month, t_death); u <- c(u, 0)
    if (t_death < end) { month <- c(month, end); u <- c(u, 0) }
  }
  list(t = month, u = unname(u))
}
trap <- function(t, u) sum(diff(t) * (head(u, -1) + tail(u, -1)) / 2)   # 사다리꼴 넓이

# 환자 다 (22장 표 22-4). 0, 3, 6, 12개월에 응답하고 15개월에 사망
p <- path(c(0, 3, 6, 12), sapply(c("11221", "21222", "21232", "31241"), kr5l), t_death = 15)
p
round(diff(p$t) * (head(p$u, -1) + tail(p$u, -1)) / 2 / 12, 3)    # 구간별 QALY
trap(p$t, p$u) / 12                                                 # QALY
DescTools::AUC(p$t, p$u) / 12                                       # 패키지 함수로 같은 값
''', title="환자 한 명의 QALY (사다리꼴 공식)")

c_miss = nb.cell('''
ga <- c(0.862, 0.841, 0.862, 0.862, 0.841, 0.862)     # 환자 가: 24개월까지 살아서 여섯 번 응답
m  <- c(0, 3, 6, 12, 18, 24)
DescTools::AUC(m, ga) / 12                            # 1.708
g12 <- replace(ga, 4, NA)                             # 12개월 응답이 NA
DescTools::AUC(m, g12) / 12                           # 오류 없이 6개월과 18개월을 직선으로 이음
g24 <- replace(ga, 6, NA)                             # 24개월 응답이 NA
DescTools::AUC(m, g24) / 12                           # NA
DescTools::AUC(m[-6], ga[-6]) / 12                    # 그 행을 지우면 넓이가 18개월에서 끝남
''', title="빠진 방문이 있으면 넓이는 어떻게 되나")

c_all = nb.cell('''
P <- lapply(pts$id, function(i) {                     # 환자마다 (시점, 효용) 경로
  g <- df[df$id == i, ]
  path(g$month, g$utility, pts$t_death[pts$id == i])
})
pts$qaly <- sapply(P, function(p) trap(p$t, p$u) / 12)
pts$ly   <- pmin(pts$t_death, 24) / 12                # 24개월 안의 생존연수
pts$base <- sapply(pts$id, function(i) df$utility[df$id == i][1])   # 치료 시작 시점의 효용
print(head(pts, 3), digits = 3)
round(sapply(split(pts[, c("qaly", "ly", "base")], pts$arm), colMeans), 3)
''', title="환자 320명의 24개월 QALY와 생존연수")

c_reg = nb.cell('''
pts$A <- as.integer(pts$arm == "A")                   # 신약 A = 1, B = 0
m0 <- lm(qaly ~ A, data = pts)
m1 <- lm(qaly ~ A + base, data = pts)
round(rbind(unadjusted = c(coef(m0)["A"], confint(m0)["A", ]),
            adjusted   = c(coef(m1)["A"], confint(m1)["A", ]),
            adjusted_HC3 = c(coef(m1)["A"], coefci(m1, vcov. = vcovHC(m1, type = "HC3"))["A", ])), 3)
round(coef(m1)["base"], 3)
''', title="기저 효용 보정 전후의 QALY 차이")

c_surv = nb.cell('''
am <- tapply(df$utility, list(df$month, df$arm), mean)   # 그 시점에 살아서 답한 사람만의 평균
round(am, 3)
round(apply(am, 2, function(u) trap(visits, u)) / 12, 3) # 생존자만의 평균 곡선 아래 넓이
round(tapply(pts$qaly, pts$arm, mean), 3)                # 환자별 QALY의 평균
''', title="생존자만의 평균 효용으로 계산하면")

# ------------------------------------------------------------------ d. 할인
c_disc24 = nb.cell('''
split_q <- function(p, cut = 12) {                    # 경로를 12개월에서 나눠 (첫해, 둘째 해)의 QALY
  uc <- approx(p$t, p$u, xout = cut)$y                # 12개월의 효용(직선 위의 값)
  a <- p$t < cut; b <- p$t > cut
  c(trap(c(p$t[a], cut), c(p$u[a], uc)), trap(c(cut, p$t[b]), c(uc, p$u[b]))) / 12
}
Y <- t(sapply(P, split_q))
isTRUE(all.equal(rowSums(Y), pts$qaly))               # 첫해 + 둘째 해 = 할인 전 QALY
pts$qaly_d <- Y[, 1] + Y[, 2] / 1.045                 # 둘째 해만 할인
cont <- function(p, r = 0.045) {                      # 0개월부터 연속으로 할인
  g <- seq(0, 24, by = 0.001)
  trap(g, approx(p$t, p$u, xout = g)$y / (1 + r)^(g / 12)) / 12
}
pts$qaly_c <- sapply(P, cont)
round(sapply(c("qaly", "qaly_d", "qaly_c"), function(v)       # 군 차이: 보정 전, 기저 효용 보정 후
  c(unadjusted = coef(lm(pts[[v]] ~ A, pts))[["A"]], adjusted = coef(lm(pts[[v]] ~ A + base, pts))[["A"]])), 3)
''', title="시험 자료의 24개월 QALY 할인하기")

c_model = nb.cell('''
p0 <- list(pfs_med = 10, pfs_gam = 0.95, os_med = 28, os_gam = 1.15,   # 표준요법 B의 곡선
           hr_pfs = 0.65, hr_os = 0.75,                                  # 신약 A의 위험비
           u_pf = 0.78, u_pd = 0.62, du_ae_A = 0.012, du_ae_B = 0.008,   # 효용과 이상반응의 QALY 손실
           c_drug_A = 190, c_drug_B = 120, c_pf = 40, c_pd = 250,        # 월 비용(만원)
           c_death = 800, c_ae_A = 120, c_ae_B = 80, disc = 0.045)       # 1회 비용(만원), 연 할인율
model <- function(p, when = "mid", months = 240) {
  t <- 0:months; k <- 0:(months - 1)
  years <- switch(when, mid = (k + 0.5) / 12,         # 각 달의 한가운데 (기준 분석)
                  start = k / 12, end = (k + 1) / 12, # 각 달의 시작, 끝
                  annual = k %/% 12,                  # 연 단위 (첫해는 할인 없음)
                  wrong = k + 0.5)                    # 잘못: 개월 수를 연수 자리에
  disc <- 1 / (1 + p$disc)^years
  sapply(c(A = "A", B = "B"), function(arm) {
    hp <- if (arm == "A") p$hr_pfs else 1; ho <- if (arm == "A") p$hr_os else 1
    s_os  <- exp(-log(2) * ho * (t / p$os_med)^p$os_gam)
    s_pfs <- pmin(exp(-log(2) * hp * (t / p$pfs_med)^p$pfs_gam), s_os)
    pf <- (head(s_pfs, -1) + tail(s_pfs, -1)) / 2                  # 반주기 보정
    pd <- (head(s_os - s_pfs, -1) + tail(s_os - s_pfs, -1)) / 2
    died <- head(s_os, -1) - tail(s_os, -1)
    pf_y <- sum(pf * disc) / 12; pd_y <- sum(pd * disc) / 12        # 할인한 상태별 기간(년)
    c(pf_years = pf_y, pd_years = pd_y,
      qaly = p$u_pf * pf_y + p$u_pd * pd_y - p[[paste0("du_ae_", arm)]],
      cost = sum((pf * (p[[paste0("c_drug_", arm)]] + p$c_pf) + pd * p$c_pd + died * p$c_death) * disc)
             + p[[paste0("c_ae_", arm)]])
  })
}
r <- model(p0)
round(r, 3)
(r["cost", "A"] - r["cost", "B"]) / (r["qaly", "A"] - r["qaly", "B"])   # ICER
''', title="공통 예시 모형을 함수 하나로")

c_when = nb.cell('''
W <- t(sapply(c("mid", "start", "end", "annual", "wrong"), function(w) {
  r <- model(p0, w)
  d_q <- r["qaly", "A"] - r["qaly", "B"]; d_c <- r["cost", "A"] - r["cost", "B"]
  c(qaly_A = r["qaly", "A"], qaly_B = r["qaly", "B"], d_qaly = d_q, d_cost = d_c, icer = d_c / d_q)
}))
round(W, 3)
round(c(month12 = 1 / 1.045^(11.5 / 12), wrong = 1 / 1.045^11.5), 3)   # 12번째 달의 할인 계수
''', title="할인 계수를 매기는 시점 바꾸기")

# ------------------------------------------------------------------ e. 과제
c_t1 = nb.cell('''
path_locf <- function(month, u, t_death, end = 24) {  # 마지막 측정값이 사망 직전까지 유지
  if (t_death <= end) {
    month <- c(month, t_death, t_death); u <- c(u, tail(u, 1), 0)
    if (t_death < end) { month <- c(month, end); u <- c(u, 0) }
  }
  list(t = month, u = unname(u))
}
p <- path_locf(c(0, 3, 6, 12), sapply(c("11221", "21222", "21232", "31241"), kr5l), 15)
round(trap(p$t, p$u) / 12, 3)                         # 환자 다
pts$qaly_locf <- sapply(pts$id, function(i) {
  g <- df[df$id == i, ]
  p <- path_locf(g$month, g$utility, pts$t_death[pts$id == i]); trap(p$t, p$u) / 12
})
round(sapply(split(pts[, c("qaly", "qaly_locf")], pts$arm), colMeans), 3)
m2 <- lm(qaly_locf ~ A, pts); m3 <- lm(qaly_locf ~ A + base, pts)
round(rbind(unadjusted = c(coef(m2)["A"], confint(m2)["A", ]),
            adjusted   = c(coef(m3)["A"], confint(m3)["A", ])), 3)
''', title="과제 1 정답. 마지막 측정값을 사망 직전까지 유지")

c_t2 = nb.cell('''
pts$qaly_vas <- sapply(pts$id, function(i) {          # 동료의 코드: 효용 대신 EQ VAS / 100
  g <- df[df$id == i, ]
  p <- path(g$month, g$vas / 100, pts$t_death[pts$id == i]); trap(p$t, p$u) / 12
})
round(tapply(pts$qaly_vas, pts$arm, mean), 3)
m4 <- lm(qaly_vas ~ A, pts)
round(c(coef(m4)["A"], confint(m4)["A", ]), 3)
p_vas <- modifyList(p0, list(u_pf = 0.73, u_pd = 0.64))   # 상태별 VAS/100 평균
sapply(list(utility = p0, vas_100 = p_vas), function(pp) {
  r <- model(pp); d_q <- r["qaly", "A"] - r["qaly", "B"]
  round(c(d_qaly = d_q, icer = (r["cost", "A"] - r["cost", "B"]) / d_q), 3)
})
''', title="과제 2 정답. EQ VAS ÷ 100으로 계산한 동료의 QALY")

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
    c_load: ['[1] 1593    7', '[1] 320   4', '1184', '409', '[1] 47', '[1] 1593'],
    c_kr5l: ['0.762', '0.518', '0.883', '-0.066'],
    c_apply: ['[1] 227', '[1] 0', '[1] 180', '-0.031'],
    c_pkg: ['[1] 0', '0.913', '0.796', '0.423'],
    c_state: ['0.7795', '0.0045', '0.0074', '0.7651', '0.7939', '0.6213', '0.0133', '0.5952', '0.6474'],
    c_mixed: ['0.623', '0.778', '0.434', '0.586', '0.616'],
    c_vas: ['0.158', '0.096'],
    c_one: ['0.200 0.189 0.333 0.073', '0.795125'],
    c_miss: ['1.708', '1.703', 'NA', '1.2825'],
    c_all: ['1.204', '1.087', '1.640', '1.571'],
    c_reg: ['0.117', '0.001', '0.232', '0.094', '-0.017', '0.204', '1.186'],
    c_surv: ['1.496', '1.411', '0.719'],
    c_disc24: ['TRUE', '0.113', '0.110', '0.088', '0.117', '0.094'],
    c_model: ['2.385', '1.874', '10561.778', '7689.485', '5620.61'],
    c_when: ['5620.610', '9140.496', '0.523', '0.959', '0.603'],
    c_t1: ['0.868', '1.230', '1.118', '0.112', '0.089'],
    c_t2: ['1.153', '1.052', '0.101', '-0.004', '0.207', '0.481', '5973.7'],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])
print("checks ok")

S = nb.save_fragment
S("rc24_load", nb.html(c_load, marks={'[1] 1593    7': 1, ' 409 1184': 2, '[1] 47': 3, '\n[1] 1593': 4}))
S("rc24_kr5l", nb.html(c_kr5l, marks={' 0.762  0.518  0.883 -0.066  1.000': 1}))
S("rc24_apply", nb.html(c_apply, marks={'[1] 227': 1, '[1] 0': 2, '-0.031': 3}))
S("rc24_pkg", nb.html(c_pkg, marks={'[1] 0': 1, '0.913  0.796 0.117': 2, '-0.594 0.423': 3}))
S("rc24_state", nb.html(c_state, marks={'0.6213   0.0077     0.0133': 1, '0.7795   0.0045     0.0074 0.7651 0.7939': 2}))
S("rc24_mixed", nb.html(c_mixed, marks={'  0.623   0.778': 1, '-0.544    0.586': 2, '0.501      0.616': 3}))
S("rc24_vas", nb.html(c_vas, marks={'0.158   0.096': 1}))
S("rc24_one", nb.html(c_one, marks={'0.000 0.000': 1, '0.200 0.189 0.333 0.073 0.000': 2, '[1] 0.795125\n[1] 0.795125': 3}))
S("rc24_miss", nb.html(c_miss, marks={'[1] 1.703': 1, '[1] NA': 2, '[1] 1.2825': 3}))
S("rc24_all", nb.html(c_all, marks={'0.799 1.442 0.293': 1, 'qaly 1.204 1.087': 2}))
S("rc24_reg", nb.html(c_reg, marks={'unadjusted   0.117  0.001  0.232': 1, 'adjusted     0.094 -0.017  0.204': 2, 'adjusted_HC3 0.094 -0.017  0.204': 3}))
S("rc24_surv", nb.html(c_surv, marks={'24 0.719 0.676': 1, '1.496 1.411': 2}))
S("rc24_disc24", nb.html(c_disc24, marks={'[1] TRUE': 1, '0.117  0.113  0.110': 2, '0.094  0.090  0.088': 3}))
S("rc24_model", nb.html(c_model, marks={'qaly         2.385    1.874': 1, '[1] 5620.61': 2}))
S("rc24_when", nb.html(c_when, marks={'annual  2.435  1.913  0.523': 1, 'wrong   0.969  0.870  0.099  905.505 9140.496': 2, '  0.959   0.603': 3}))
S("rc24_t1", nb.html(c_t1))
S("rc24_t2", nb.html(c_t2))
nb.save_ipynb("21장 효용과 QALY (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 22장의 가상 시험 자료(EQ-5D-5L 응답 1,593건, 환자 320명)로 응답 코드를 효용으로 바꾸고, 상태별 효용과 그 신뢰구간, "
                    "환자별 24개월 QALY와 기저 효용 보정, 할인을 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"eq5d\", \"sandwich\", \"lmtest\", \"lme4\", \"DescTools\"))를 실행합니다.",
              notes={c_load.n: "## a. EQ-5D 응답을 효용으로", c_state.n: "## b. 상태별 효용과 신뢰구간", c_one.n: "## c. 환자별 QALY",
                     c_disc24.n: "## d. 할인", c_t1.n: "## e. 과제 정답"})
print("rc24 fragments and notebook written")
