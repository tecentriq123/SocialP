"""R과 파이썬으로 하는 통계분석 12장 · 불멸시간 편향과 시간의존 노출 — R cells run for real with rkit.

run:  python3 gen/r_rc19.py      (NOMARK=1 python3 gen/r_rc19.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 16장의 '직접 분석' 상자와 본문에 있던 파이썬 코드를 2026-11-19 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 16장의 가상 코호트(급성 심근경색 퇴원 환자 6,000명, 약물 X는 사망에 효과 없음; pub/data/ami_cohort.csv,
gen/data_lab16.py = gen/nums_ch16.py와 같은 seed). 숫자는 gen/_ch16_nums.json과 대조한다
(환자 단위 부트스트랩만 난수가 달라 끝자리가 다를 수 있음).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc19")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 불멸시간
c_load = nb.cell(f'''
suppressPackageStartupMessages(library(survival))
d <- read.csv("{D}ami_cohort.csv")     # 한 줄 = 환자 한 명
str(d)
summary(d$rx_day)                       # NA = 추적 중 약물 X를 시작하지 않은 사람
''', title="자료 불러오기 (16장 a 절)")

c_imm = nb.cell('''
u <- d[!is.na(d$rx_day), ]                          # 추적 중 처방받은 사람
round(c(users = nrow(u),
        immortal_py = sum(u$rx_day) / 365,           # 첫 처방 전 시간(인년)
        share = sum(u$rx_day) / sum(u$fu_days),      # 사용군 추적 시간 가운데 그 비율
        median_months = median(u$rx_day) / 365 * 12), 3)   # 첫 처방까지 중앙값(개월)
early <- d[d$died == 1 & d$fu_days <= 91.25, ]      # 퇴원 후 3개월 안의 사망
table(started = !is.na(early$rx_day))               # 사망 전에 약을 시작했나
''', title="불멸시간의 크기 재기 (16장 a 절)")

c_wrong = nb.cell('''
d$ever <- as.integer(!is.na(d$rx_day))              # 추적 중 한 번이라도 처방 = 1
tab <- function(time, ev, g) {                      # 군별 사망, 인년, 100인년당 사망률
  r <- rbind(deaths = tapply(ev, g, sum), py = tapply(time, g, sum) / 365)
  round(rbind(r, rate = r["deaths", ] / r["py", ] * 100), 1)
}
tab(d$fu_days, d$died, d$ever)                      # ① 추적 중 사용하면 처음부터 사용군
f1 <- coxph(Surv(fu_days, died) ~ ever, data = d)
d$t2 <- ifelse(d$ever == 1, d$fu_days - d$rx_day, d$fu_days)
tab(d$t2, d$died, d$ever)                           # ② 사용군만 첫 처방일부터 추적
f2 <- coxph(Surv(t2, died) ~ ever, data = d)
round(rbind(f1 = summary(f1)$conf.int[, c(1, 3, 4)],
            f2 = summary(f2)$conf.int[, c(1, 3, 4)]), 3)
''', title="표 16-2의 ①과 ②: 시간을 잘못 나눈 두 분석 (16장 a 절)")

# ------------------------------------------------------------------ b. 랜드마크
c_lm = nb.cell('''
L  <- 365 * 3 / 12                                  # 랜드마크 91.25일
lm <- d[d$fu_days > L, ]                            # 그때 살아서 추적 중인 사람
lm$x <- as.integer(!is.na(lm$rx_day) & lm$rx_day <= L)   # 그때까지 처방 = 1
lm$t <- lm$fu_days - L                              # 랜드마크부터의 일수
c(n = nrow(lm), excluded = nrow(d) - nrow(lm), exposed = sum(lm$x),
  late_starters = sum(lm$x == 0 & !is.na(lm$rx_day)))
fl <- coxph(Surv(t, died) ~ x, data = lm)
round(summary(fl)$conf.int[, c(1, 3, 4)], 3)
lm$x_wrong <- lm$ever                               # 틀린 정의: 랜드마크 뒤의 처방까지 사용군
round(summary(coxph(Surv(t, died) ~ x_wrong, data = lm))$conf.int[, c(1, 3, 4)], 3)
''', title="3개월 랜드마크 분석 (16장 b 절)")

c_lm2 = nb.cell('''
landmark <- function(months) {
  L  <- 365 * months / 12
  lm <- d[d$fu_days > L, ]
  lm$x <- as.integer(!is.na(lm$rx_day) & lm$rx_day <= L)
  fit <- coxph(Surv(fu_days - L, died) ~ x, data = lm)
  c(months = months, n = nrow(lm), exposed = sum(lm$x),
    round(summary(fit)$conf.int[, c(1, 3, 4)], 3))
}
t(sapply(c(1, 3, 6), landmark))
''', title="랜드마크 시점 바꿔 보기 (16장 b 절)")

# ------------------------------------------------------------------ c. 시간의존 Cox
c_tm = nb.cell('''
base <- tmerge(d[, c("id", "fu_days", "died")], d[, c("id", "fu_days", "died")],
               id = id, death = event(fu_days, died))   # 환자당 한 줄: tstart = 0, tstop = fu_days
rx   <- d[!is.na(d$rx_day), c("id", "rx_day")]          # 약을 시작한 사람의 첫 처방일
long <- tmerge(base, rx, id = id, x = tdc(rx_day))       # 첫 처방일에서 줄을 나눔
long[long$id %in% 0:2, c("id", "tstart", "tstop", "death", "x")]
c(rows = nrow(long), events = sum(long$death))
dth <- tapply(long$death, long$x, sum)                   # 노출 상태별 사망
py  <- tapply(long$tstop - long$tstart, long$x, sum) / 365   # 노출 상태별 인년
round(rbind(deaths = dth, py = py, rate = dth / py * 100), 1)
''', title="긴 형식 자료 만들기와 노출 상태별 사망률 (16장 c 절)")

c_tv = nb.cell('''
f3 <- coxph(Surv(tstart, tstop, death) ~ x, data = long)
summary(f3)
''', title="시간의존 Cox 모형 (16장 c 절)")

c_period = nb.cell('''
rr <- glm(death ~ x + offset(log((tstop - tstart) / 365)), family = poisson, data = long)
round(exp(c(coef(rr)["x"], confint.default(rr)["x", ])), 3)     # 1년을 한데 섞은 발생률비
sp <- survSplit(Surv(tstart, tstop, death) ~ ., data = long,
                cut = c(91.25, 182.5), episode = "period")       # 0–3, 3–6, 6–12개월
sp$py <- (sp$tstop - sp$tstart) / 365
agg <- aggregate(cbind(death, py) ~ period + x, data = sp, sum)
agg$rate <- round(agg$death / agg$py * 100, 1); agg
mh <- sapply(split(agg, agg$period), function(a) {               # Mantel–Haenszel 발생률비
  T <- sum(a$py)
  c(num = a$death[a$x == 1] * a$py[a$x == 0] / T, den = a$death[a$x == 0] * a$py[a$x == 1] / T)
})
round(sum(mh["num", ]) / sum(mh["den", ]), 3)
''', title="사망률의 비와 위험비: 기간을 나눠 같은 시기끼리 비교 (16장 c 절)")

c_check = nb.cell('''
hr <- function(dat) round(summary(coxph(Surv(tstart, tstop, death) ~ x, data = dat))$conf.int[, c(1, 3, 4)], 3)
dup <- long                                   # 틀린 자료 1: 처방 뒤 사망을 처방 전 줄에도 1로
k <- dup$id[dup$x == 1 & dup$death == 1]
dup$death[dup$x == 0 & dup$id %in% k] <- 1
ov <- long                                    # 틀린 자료 2: 노출 줄의 시작을 10일 앞당겨 겹침
ov$tstart[ov$x == 1] <- ov$tstart[ov$x == 1] - 10
rbind(right = hr(long), duplicated = hr(dup), overlap = hr(ov))
c(right = sum(long$death), duplicated = sum(dup$death))
survcheck(Surv(tstart, tstop, death) ~ 1, data = ov, id = id)$flag
''', title="틀린 긴 형식 자료도 오류 없이 적합된다 (16장 c 절)")

c_rules = nb.cell('''
toy <- data.frame(id = 1:4, fu_days = c(100, 200, 150, 365), died = c(1, 1, 0, 0))
rx_toy <- data.frame(id = 1:4, rx_day = c(0, 200, 180, NA))   # 퇴원일, 사망일, 추적 종료 뒤, 기록 없음
b0 <- tmerge(toy, toy, id = id, death = event(fu_days, died))
tmerge(b0, rx_toy, id = id, x = tdc(rx_day))[, c("id", "tstart", "tstop", "death", "x")]
na_long <- tmerge(base, transform(rx, drugx = 1), id = id, x = tdc(rx_day, drugx))   # 값을 주고 init을 빼면
table(x = na_long$x, useNA = "ifany")                          # 처방 전 구간과 비사용자는 NA
fit_na <- coxph(Surv(tstart, tstop, death) ~ x, data = na_long)
c(rows_used = fit_na$n, events_used = fit_na$nevent)
''', title="tmerge가 경고 없이 정하는 동작 (16장 c 절)")

c_cur = nb.cell('''
one <- data.frame(id = 7, fu_days = 365, died = 0)            # 그림 16-4의 환자
b7  <- tmerge(one, one, id = id, death = event(fu_days, died))
cv  <- data.frame(id = 7, day = c(40, 160, 190, 280), drugx = c(1, 0, 1, 0))   # 노출이 시작·끝나는 날
tmerge(b7, cv, id = id, x = tdc(day, drugx, init = 0))[, c("tstart", "tstop", "x")]   # 첫 처방 전은 0
''', title="현재 사용으로 정한 노출 구간 (16장 c 절)")

# ------------------------------------------------------------------ d. 시차, 강건 표준오차, 날짜
c_lag = nb.cell('''
long30 <- tmerge(base, rx, id = id, x = tdc(rx_day),
                 options = list(delay = 30))          # 첫 처방 뒤 30일은 비노출로 셈
nrow(long30)
round(summary(coxph(Surv(tstart, tstop, death) ~ x, data = long30))$conf.int[, c(1, 3, 4)], 3)
''', title="시차 30일 (16장 d 절)")

c_rob = nb.cell('''
fr <- coxph(Surv(tstart, tstop, death) ~ x, data = long, cluster = id)   # 환자 단위 강건 표준오차
round(summary(fr)$coefficients, 4)
rows <- split(seq_len(nrow(long)), long$id)            # 환자별 줄 번호
set.seed(16)
bs <- replicate(1000, {                                 # 환자 단위 부트스트랩
  s <- sample(names(rows), replace = TRUE)
  coef(coxph(Surv(tstart, tstop, death) ~ x, data = long[unlist(rows[s]), ]))
})
round(sd(bs), 4)
''', title="환자 단위 강건 표준오차와 부트스트랩 (16장 d 절)")

c_dates = nb.cell('''
cohort <- data.frame(id = 1:3, dis = as.Date(c("2021-03-02", "2021-05-10", "2021-07-21")))   # 퇴원일
fills  <- data.frame(id = c(1, 1, 2, 3, 3),                                                  # 약물 X 조제일
                     fill = as.Date(c("2021-03-02", "2021-04-15", "2021-06-01", "2021-07-10", "2021-09-30")))
f <- merge(fills, cohort, by = "id")
f <- f[f$fill > f$dis, ]                         # 퇴원 다음 날 이후(퇴원약을 노출로 보면 >=)
first <- aggregate(fill ~ id + dis, data = f, FUN = min)
first$rx_day <- as.numeric(first$fill - first$dis)   # 퇴원 후 일수 = 처방 표의 rx_day
first
''', title="청구 날짜로 처방 표 만들기 (16장 d 절)")

# ------------------------------------------------------------------ e. 과제
c_t1 = nb.cell('''
p <- tapply(d$died, d$ever, mean)          # 1년 사망 비율(분모 = 사람 수)
round(c(p, ratio = unname(p["1"] / p["0"])), 3)
''', title="과제 1 정답. 사람 수를 분모로 한 비교")

c_t2 = nb.cell('''
t(sapply(c(0, 30, 60, 90), function(lag) {
  l <- tmerge(base, rx, id = id, x = tdc(rx_day), options = list(delay = lag))
  c(lag = lag, rows = nrow(l), round(summary(coxph(Surv(tstart, tstop, death) ~ x, data = l))$conf.int[, c(1, 3, 4)], 3))
}))
''', title="과제 2 정답. 시차의 길이 바꿔 보기")

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
    c_load: ["6000 obs. of  4 variables", "NA's", "3266"],
    c_imm: ["2734.000", "1141.768", "0.437", "4.548", "380", "37"],
    c_wrong: ["913.0  321.0", "2612.8", "12.3", "913.0", "2698.7", "33.8", "1471.0", "21.8", "0.366", "0.323", "0.416", "0.582", "0.511", "0.662"],
    c_lm: ["5583", "417", "934", "1763", "0.942     0.781     1.135", "0.535     0.463     0.618"],
    c_lm2: ["1.057", "0.824", "1.356", "1.056", "0.877", "1.271"],
    c_tm: ["8734", "1234", "3840.5", "1471.0", "23.8", "21.8"],
    c_tv: ["1.031", "0.9     1.182", "n= 8734, number of events= 1234"],
    c_period: ["0.918  0.808  1.042", "28.8", "30.0", "22.0", "20.6", "20.5", "21.2", "1.011"],
    c_check: ["0.810     0.711     0.923", "0.966     0.844     1.106", "1234", "1555", "2734"],
    c_rules: ["2734", "321"],
    c_cur: ["160", "190", "280"],
    c_lag: ["8555", "1.026     0.886     1.187"],
    c_rob: ["0.0695", "0.0696", "0.0688"],
    c_dates: ["44", "22", "71"],
    c_t1: ["0.280", "0.117", "0.420"],
    c_t2: ["8382", "8151", "0.969"],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc19_load", nb.html(c_load, marks={"6000 obs. of  4 variables": 1, "3266": 2}))
S("rc19_imm", nb.html(c_imm, marks={"0.437": 1, "4.548": 2, "380": 3}))
S("rc19_wrong", nb.html(c_wrong, marks={"0.366": 1, "0.582": 2}))
S("rc19_lm", nb.html(c_lm, marks={"1763": 1, "0.942     0.781     1.135": 2, "0.535     0.463     0.618": 3}))
S("rc19_lm2", nb.html(c_lm2))
S("rc19_tm", nb.html(c_tm, marks={"8734": 1, "23.8": 2}))
S("rc19_tv", nb.html(c_tv, marks={"n= 8734, number of events= 1234": 1, "1.182": 2}))
S("rc19_period", nb.html(c_period, marks={"0.918  0.808  1.042": 1, "1.011": 2}))
S("rc19_check", nb.html(c_check, marks={"0.810     0.711     0.923": 1, "0.966     0.844     1.106": 2, "1555": 3}))
S("rc19_rules", nb.html(c_rules))
S("rc19_cur", nb.html(c_cur))
S("rc19_lag", nb.html(c_lag, marks={"8555": 1, "1.026     0.886     1.187": 2}))
S("rc19_rob", nb.html(c_rob))
S("rc19_dates", nb.html(c_dates))
S("rc19_t1", nb.html(c_t1))
S("rc19_t2", nb.html(c_t2))
nb.save_ipynb("12장 불멸시간 편향과 시간의존 노출 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 16장의 가상 코호트(급성 심근경색 퇴원 환자 6,000명)로 불멸시간의 크기, 시간을 잘못 나눈 두 분석, "
                    "랜드마크 분석, tmerge로 만든 긴 형식 자료와 시간의존 Cox 모형, 기간을 나눈 발생률비, 시차, 환자 단위 강건 표준오차를 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 불멸시간의 크기 재기", c_lm.n: "## b. 랜드마크 분석", c_tm.n: "## c. 시간의존 Cox 모형",
                     c_lag.n: "## d. 시차, 강건 표준오차, 날짜로 처방 표 만들기", c_t1.n: "## e. 과제 정답"})
print("rc19 fragments and notebook written")
