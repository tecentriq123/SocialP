"""R과 파이썬으로 하는 통계분석 13장 · 경쟁위험 분석 — R cells run for real with rkit.

run:  python3 gen/r_rc20.py      (NOMARK=1 python3 gen/r_rc20.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 17장의 '직접 분석' 상자와 본문에 있던 파이썬 코드를 2026-11-19 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 17장의 가상 코호트(만성콩팥병 4기 고령 환자, 약물 A 1,380명 대 B 2,120명, 투석과 투석 전 사망;
pub/data/ckd_competing.csv)와 확인 문제의 치매 환자 고관절 골절 자료(pub/data/dementia_fracture.csv), 둘 다 gen/data_lab17.py
(= gen/nums_ch17.py의 _ch17_cr.csv, _ch17_fx.csv). 숫자는 gen/_ch17_nums.json과 대조한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc20")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 사건 열과 누적발생함수
c_load = nb.cell(f'''
suppressPackageStartupMessages({{library(survival); library(cmprsk)}})
d <- read.csv("{D}ckd_competing.csv")     # 한 줄 = 환자 한 명
str(d)
table(drug = d$drug, status = d$status)  # 0 = 중도절단, 1 = 투석, 2 = 투석 전 사망
d$ev <- factor(d$status, levels = 0:2,
               labels = c("censor", "dialysis", "death"))   # 첫 수준이 중도절단
''', title="자료 불러오기와 사건 열 (17장 a 절)")

c_event = nb.cell('''
x <- data.frame(id = 1:4, index = as.Date("2018-01-01"),
                dial  = as.Date(c("2019-03-01", NA, "2020-06-30", NA)),           # 투석 시작일
                death = as.Date(c(NA, "2018-11-20", "2020-06-30", NA)),           # 사망일
                cens  = as.Date(c("2022-12-31", "2022-12-31", "2022-12-31", "2021-05-31")))  # 중도절단일
end <- pmin(x$dial, x$death, x$cens, na.rm = TRUE)            # 가장 이른 날 = 추적 종료일
x$years  <- round(as.numeric(end - x$index) / 365.25, 3)
x$status <- ifelse(!is.na(x$dial) & x$dial == end, 1,         # 같은 날이면 투석을 먼저 확인
            ifelse(!is.na(x$death) & x$death == end, 2, 0))
x[, c("id", "years", "status")]
''', title="청구 날짜로 사건 열 만들기 (17장 b 절)")

c_aj = nb.cell('''
fit <- survfit(Surv(years, ev) ~ drug, data = d)      # 군마다 Aalen–Johansen 추정
s <- summary(fit, times = c(1, 3, 5))
p <- s$pstate; colnames(p) <- c("free", "dialysis", "death")
data.frame(drug = sub("drug=", "", s$strata), time = s$time, round(p, 4),
           dial_lo = round(s$lower[, 2], 4), dial_up = round(s$upper[, 2], 4),
           death_lo = round(s$lower[, 3], 4), death_up = round(s$upper[, 3], 4))
''', title="두 군의 누적발생함수 (17장 a 절)")

c_km = nb.cell('''
b  <- d[d$drug == "B", ]
km <- summary(survfit(Surv(years, status == 1) ~ 1, data = b), times = 1:5)   # 사망을 중도절단한 1 − KM
aj <- summary(survfit(Surv(years, ev) ~ 1, data = b), times = 1:5)
cbind(year = 1:5, round(cbind(one_minus_km = 1 - km$surv, dialysis = aj$pstate[, 2], death = aj$pstate[, 3],
                               free = aj$pstate[, 1], total = rowSums(aj$pstate)) * 100, 1))
bad <- survfit(Surv(years, status) ~ 1, data = b)    # 0/1/2를 숫자 그대로 넣으면
c(n = bad$n, events = sum(bad$n.event))
''', title="표 17-2: 1 − Kaplan-Meier와 누적발생함수 (17장 a 절)")

c_diff = nb.cell('''
s5 <- summary(fit, times = 5)
p  <- s5$pstate[, 2]; se <- s5$std.err[, 2]            # 5년 투석 누적발생률과 표준오차(A, B)
dif <- unname(p[1] - p[2]); sed <- sqrt(sum(se^2))
round(c(A = p[1], B = p[2], diff = dif, lower = dif - 1.96 * sed, upper = dif + 1.96 * sed) * 100, 1)
round(s5$pstate[, 3] * 100, 1)                           # 5년 투석 전 사망(A, B)
''', title="5년 누적발생률의 차이 (17장 a 절)")

c_gray = nb.cell('''
ci <- cuminc(d$years, d$status, group = d$drug, cencode = 0)
ci$Tests                                   # Gray 검정: 1행 = 투석, 2행 = 투석 전 사망
survdiff(Surv(years, status == 1) ~ drug, data = d)   # 사망을 중도절단한 로그순위법
''', title="Gray 검정과 로그순위법 (17장 a 절)")

# ------------------------------------------------------------------ b. 원인별 위험비와 하위분포 위험비
c_rates = nb.cell('''
agg <- aggregate(cbind(py = years, dialysis = status == 1, death = status == 2) ~ drug, data = d, sum)
cbind(agg, rate_dialysis = round(agg$dialysis / agg$py * 100, 2),
      rate_death = round(agg$death / agg$py * 100, 2))   # 100인년당
''', title="표 17-3: 군별 속도 (17장 b 절)")

c_cs = nb.cell('''
d$A <- as.integer(d$drug == "A")
ci3 <- function(fit) round(summary(fit)$conf.int[, c(1, 3, 4)], 3)
cs_dial  <- coxph(Surv(years, status == 1) ~ A, data = d)   # 투석만 사건, 사망은 중도절단
cs_death <- coxph(Surv(years, status == 2) ~ A, data = d)   # 사망만 사건, 투석은 중도절단
rbind(dialysis = ci3(cs_dial), death = ci3(cs_death))
summary(cs_dial)$coefficients
''', title="원인별 위험비 (17장 b 절)")

c_fg = nb.cell('''
d$t_sd <- ifelse(d$status == 2, d$max_years, d$years)       # 사망자를 추적 종료 예정 시점까지 남김
fg_date <- coxph(Surv(t_sd, status == 1) ~ A, data = d)
fgd  <- finegray(Surv(years, ev) ~ A + id, data = d, etype = "dialysis")   # 중도절단 확률로 가중해 남김
fg_w <- coxph(Surv(fgstart, fgstop, fgstatus) ~ A, weights = fgwt, cluster = id, data = fgd)
fg_crr <- crr(d$years, d$status, cov1 = cbind(A = d$A), failcode = 1, cencode = 0)
rbind(date_method = ci3(fg_date), finegray = ci3(fg_w),
      crr = round(summary(fg_crr)$conf.int[, c(1, 3, 4)], 3))
nrow(fgd)                                                    # finegray가 만든 줄 수
fg_row <- coxph(Surv(fgstart, fgstop, fgstatus) ~ A, weights = fgwt, data = fgd)   # cluster = id를 빼면
round(c(cluster_id = sqrt(fg_w$var[1, 1]), by_row = sqrt(fg_row$var[1, 1]), crr = sqrt(fg_crr$var[1, 1])), 4)
''', title="하위분포 위험비: 예정일 방법, finegray, crr (17장 b 절)")

c_zph = nb.cell('''
round(rbind(dialysis = cox.zph(cs_dial)$table["A", ], death = cox.zph(cs_death)$table["A", ],
            subdist_date = cox.zph(fg_date)$table["A", ]), 3)
''', title="비례위험 검정 (17장 b 절)")

c_w = nb.cell('''
b$w <- 2                                              # 모든 환자의 가중치를 2로
cif5 <- function(f) {
  s <- summary(f, times = 5)
  round(c(cif = s$pstate[2], lower = s$lower[2], upper = s$upper[2]), 4)
}
rbind(no_weight = cif5(survfit(Surv(years, ev) ~ 1, data = b)),
      weight_2  = cif5(survfit(Surv(years, ev) ~ 1, data = b, weights = w, id = id)))
''', title="가중 누적발생함수의 신뢰구간 (17장 b 절)")

# ------------------------------------------------------------------ c. 과제
c_t1 = nb.cell('''
small <- data.frame(month = 1:8, ev = factor(c(1, 2, 2, 1, 2, 1, 0, 0), 0:2, c("censor", "fracture", "death")))
km6 <- summary(survfit(Surv(month, ev == "fracture") ~ 1, data = small), times = 6)
aj6 <- summary(survfit(Surv(month, ev) ~ 1, data = small), times = 6)
round(c(one_minus_km = 1 - km6$surv, cif_fracture = aj6$pstate[2], cif_death = aj6$pstate[3], free = aj6$pstate[1]), 3)
''', title="과제 1 정답. 요양병원 입소자 8명")

c_t2 = nb.cell(f'''
fx <- read.csv("{D}dementia_fracture.csv")
fx$ev <- factor(fx$status, 0:2, c("censor", "fracture", "death")); fx$C <- as.integer(fx$drug == "C")
sf <- summary(survfit(Surv(years, ev) ~ drug, data = fx), times = 5)
km <- summary(survfit(Surv(years, status == 1) ~ drug, data = fx), times = 5)
round(cbind(fracture = sf$pstate[, 2], death = sf$pstate[, 3], one_minus_km = 1 - km$surv) * 100, 1)
fgx <- finegray(Surv(years, ev) ~ C, data = fx, etype = "fracture")
rbind(cs_fracture = ci3(coxph(Surv(years, status == 1) ~ C, data = fx)),
      cs_death    = ci3(coxph(Surv(years, status == 2) ~ C, data = fx)),
      subdist     = ci3(coxph(Surv(fgstart, fgstop, fgstatus) ~ C, weights = fgwt, data = fgx)))
''', title="과제 2 정답. 고관절 골절 자료로 동료의 분석 다시 보기")

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
    c_load: ["3500 obs. of  5 variables", "521  396  463", "536  496 1088"],
    c_event: ["1.161      1", "0.884      2", "2.494      1", "3.411      0"],
    c_aj: ["0.3021", "0.2778", "0.3284", "0.2455", "0.2272", "0.2652", "0.5296"],
    c_km: ["8.8      7.9  20.2 71.9", "39.2     24.5  53.0 22.5", "1584   1088"],
    c_diff: ["30.2  24.5   5.7   2.5   8.8", "35.2 53.0"],
    c_gray: ["11.51338 0.0006909696", "p= 0.8"],
    c_rates: ["4049.164", "5195.511", "9.78", "11.43", "9.55", "20.94"],
    c_cs: ["1.018     0.892     1.162", "0.551     0.494     0.614", "0.7873154"],
    c_fg: ["1.254     1.099     1.431", "1.257     1.102     1.435", "223496", "0.0673", "0.0651"],
    c_zph: ["0.568", "0.325", "0.126"],
    c_w: ["0.2455 0.2272 0.2652"],
    c_t1: ["0.533", "0.375", "0.250"],
    c_t2: ["8.3", "10.2", "14.1", "0.968     0.801     1.169", "1.648     1.526     1.779", "0.810     0.673     0.976"],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc20_load", nb.html(c_load, marks={"536  496 1088": 1}))
S("rc20_event", nb.html(c_event, marks={"2.494      1": 1}))
S("rc20_aj", nb.html(c_aj, marks={"0.3021": 1, "0.2455": 2, "0.5296": 3}))
S("rc20_km", nb.html(c_km, marks={"39.2     24.5  53.0 22.5": 1, "1584   1088": 2}))
S("rc20_diff", nb.html(c_diff, marks={"30.2  24.5   5.7   2.5   8.8": 1}))
S("rc20_gray", nb.html(c_gray, marks={"11.51338 0.0006909696": 1, "p= 0.8": 2}))
S("rc20_rates", nb.html(c_rates, marks={"9.55": 1, "20.94": 2}))
S("rc20_cs", nb.html(c_cs, marks={"1.018     0.892     1.162": 1, "0.551     0.494     0.614": 2}))
S("rc20_fg", nb.html(c_fg, marks={"1.254     1.099     1.431": 1, "1.257     1.102     1.435": 2, "0.0651": 3}))
S("rc20_zph", nb.html(c_zph))
S("rc20_w", nb.html(c_w))
S("rc20_t1", nb.html(c_t1))
S("rc20_t2", nb.html(c_t2))
nb.save_ipynb("13장 경쟁위험 분석 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 17장의 가상 코호트(만성콩팥병 4기 고령 환자 3,500명, 투석과 투석 전 사망)로 사건 열, Aalen–Johansen 누적발생함수, "
                    "1 − Kaplan-Meier와의 차이, Gray 검정, 원인별 위험비, Fine–Gray 하위분포 위험비, 비례위험 검정을 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(\"cmprsk\")를 실행합니다.",
              notes={c_load.n: "## a. 사건 열과 누적발생함수", c_rates.n: "## b. 원인별 위험비와 하위분포 위험비", c_t1.n: "## c. 과제 정답"})
print("rc20 fragments and notebook written")
