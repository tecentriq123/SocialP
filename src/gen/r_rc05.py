"""R과 통계분석 5장 · 논문의 KM 곡선에서 환자 자료 재구성하기 — R cells run for real with rkit (IPDfromKM, survival, flexsurv).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_rc05.py && python3 gen/r_rc05.py   (NOMARK=1 … 로 모든 출력)

'논문 그림'은 23장 d 절의 가상 시험(rcc_trial.csv, 표준요법 B군 300명)의 Kaplan-Meier 곡선이고, 읽어 낸 좌표는
gen/data_rc05.py가 만든 pub/data/rcc_km_digitized.csv다. 원자료가 있으므로 재구성한 자료를 정답과 맞춰 볼 수 있다.
끝의 대조 블록은 원자료 쪽 숫자가 4장(23장 d 절)과 같은지, 재구성한 자료의 사건 수와 위험집합이 출력과 본문에 적은 값인지 확인한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc05")

# a. 읽어 낸 좌표와 위험집합 표
c_dig = nb.cell('''
options(HTTPUserAgent = "Mozilla/5.0")
BASE <- "https://socialp-ajou.tecentriq12.workers.dev/data/"
dig <- read.csv(paste0(BASE, "rcc_km_digitized.csv"))
dim(dig)
head(dig, 8)
c(time_reversed = sum(diff(dig$time) < 0), surv_up = sum(diff(dig$surv) > 0))
''', title="그림에서 읽어 낸 좌표")

c_dplot = nb.cell('''
plot(dig$time, dig$surv, pch = 20, cex = 0.5, ylim = c(0, 1), xaxt = "n",
     xlab = "Months", ylab = "Overall survival (read from figure)")
axis(1, at = seq(0, 30, by = 6))
''', title="읽어 낸 좌표 그려 보기", fig=(7, 4))

c_nar = nb.cell('''
trisk <- c(0, 6, 12, 18, 24)            # 그림 아래 위험집합 표의 시점(개월)
nrisk <- c(300, 265, 238, 202, 167)     # 그 시점에 위험집합에 남은 환자 수
tot_events <- 145                       # 본문에 적힌 총 사망 수
''', title="위험집합 표와 총 사건 수 옮겨 적기")

# b. 재구성하기
c_pre = nb.cell('''
install.packages("IPDfromKM")
library(IPDfromKM)
pre <- preprocess(dat = dig, trisk = trisk, nrisk = nrisk,
                  totalpts = 300, maxy = 1)
nrow(pre$preprocessdat)
head(pre$preprocessdat, 4)
pre$intervalIndex
''', title="preprocess()로 좌표 정리하기")

c_get = nb.cell('''
ipd <- getIPD(prep = pre, armID = 1, tot.events = tot_events)
summary(ipd)
''', title="getIPD()로 환자 자료 재구성하기")

c_rec = nb.cell('''
rec <- ipd$IPD
head(rec)
table(rec$status)
library(survival)
km_r <- survfit(Surv(time, status) ~ 1, data = rec)
summary(km_r, times = trisk)[c("n.risk", "surv")]
''', title="재구성한 자료 살펴보기")

# c. 원자료와 비교
c_cmp = nb.cell('''
orig <- read.csv(paste0(BASE, "rcc_trial.csv"))     # 정답 확인용 원자료
km_o <- survfit(Surv(months, event) ~ 1, data = orig)
at <- c(6, 12, 18, 24, 29)
round(cbind(months = at, original = summary(km_o, times = at)$surv,
            reconstructed = summary(km_r, times = at)$surv), 4)
after24 <- function(t, e)            # 24개월 뒤의 사망 수와 위험집합에 머문 기간(사람-월)
  c(deaths = sum(e[t > 24]), person_months = round(sum(pmax(0, t - 24)), 1))
rbind(original = after24(orig$months, orig$event), reconstructed = after24(rec$time, rec$status))
''', title="원자료와 생존율 비교")

c_cplot = nb.cell('''
plot(km_o, conf.int = FALSE, xlab = "Months", ylab = "Overall survival", xaxt = "n")
lines(km_r, conf.int = FALSE, col = "#c8553d", lty = 2)
points(dig$time, dig$surv, pch = 20, cex = 0.3, col = "gray50")
axis(1, at = seq(0, 30, by = 6))
legend("bottomleft", c("Original data", "Reconstructed", "Points read"),
       col = c("black", "#c8553d", "gray50"), lty = c(1, 2, NA),
       pch = c(NA, NA, 20), bty = "n")
''', title="두 곡선 겹쳐 그리기", fig=(7, 4.2))

c_fit = nb.cell('''
library(flexsurv)
dists <- c(exp = "exp", weibull = "weibull", gompertz = "gompertz",
           lnorm = "lnorm", llogis = "llogis", gengamma = "gengamma")
ext_tab <- function(formula, data) {          # 여섯 분포의 AIC 차이, 5년 생존율, 20년 평균 생존
  fits <- lapply(dists, function(k) flexsurvreg(formula, data = data, dist = k))
  aic  <- sapply(fits, AIC)
  data.frame(dAIC = round(aic - min(aic), 1),
             S5y  = round(100 * sapply(fits, function(f)
                      summary(f, t = 60, ci = FALSE, tidy = TRUE)$est), 1),
             RMST20 = round(sapply(fits, function(f)
                      summary(f, t = 240, type = "rmst", ci = FALSE, tidy = TRUE)$est) / 12, 2))
}
cbind(ext_tab(Surv(months, event) ~ 1, orig), ext_tab(Surv(time, status) ~ 1, rec))
''', title="여섯 분포로 외삽한 결과 비교")

# d. 위험집합 표가 촘촘하면
c_nar3 = nb.cell('''
trisk3 <- seq(0, 27, by = 3)
nrisk3 <- c(300, 283, 265, 251, 238, 223, 202, 185, 167, 80)
rec3 <- getIPD(preprocess(dig, trisk = trisk3, nrisk = nrisk3, totalpts = 300, maxy = 1),
               armID = 1, tot.events = tot_events)$IPD
c(events = sum(rec3$status), after24 = sum(rec3$status[rec3$time > 24]))
ext_tab(Surv(time, status) ~ 1, rec3)
''', title="3개월 간격의 위험집합 표로 다시")

# e. 과제
c_t1 = nb.cell('''
rec0 <- getIPD(preprocess(dig, trisk = 0, nrisk = 300, totalpts = 300, maxy = 1),
               armID = 1, tot.events = tot_events)$IPD
c(events = sum(rec0$status))
summary(survfit(Surv(time, status) ~ 1, data = rec0), times = trisk)[c("n.risk", "surv")]
''', title="과제 1 정답: 위험집합 표가 없으면")

c_t2 = nb.cell('''
step <- data.frame(time = c(0, rep(km_o$time[km_o$n.event > 0], each = 2)),
                   surv = c(1, as.vector(rbind(c(1, head(km_o$surv[km_o$n.event > 0], -1)),
                                               km_o$surv[km_o$n.event > 0]))))
step <- rbind(step, data.frame(time = max(orig$months), surv = tail(km_o$surv, 1)))
rec_x <- getIPD(preprocess(step, trisk = trisk, nrisk = nrisk, totalpts = 300, maxy = 1),
                armID = 1, tot.events = tot_events)$IPD
c(events = sum(rec_x$status), after24 = sum(rec_x$status[rec_x$time > 24]))
''', title="과제 2 정답: 읽기 오차가 없는 좌표로")

c_t3 = nb.cell('''
dig_pct <- transform(dig, surv = surv * 100)      # 세로축이 0-100%인 그림
rec_p <- getIPD(preprocess(dig_pct, trisk = trisk, nrisk = nrisk, totalpts = 300, maxy = 100),
                armID = 1, tot.events = tot_events)$IPD
all.equal(rec_p, rec)
''', title="과제 3 정답: 세로축이 백분율이면")

nb.run()

if NOMARK:
    for c in nb.cells:
        print(f"===== cell {c.n} {c.title}\n{c.text}")
        for w in c.warns:
            print("WARN:", w)
        if c.error:
            print("ERR:", c.error)
        print("images:", len(c.images))
    sys.exit(0)

# ---------------------------------------------------------------- 대조
out = {c.n: c.text for c in nb.cells}
assert "[1] 240   2" in out[c_dig.n]
assert "152 148" in out[c_rec.n]
for v in ("24.1   3.50", "19.5   3.10", "11.1   2.60", "32.6   5.19", "28.3   4.62", "12.1   2.64"):   # 원자료 = 23장 d 절
    assert v in out[c_fit.n], v

S = nb.save_fragment
S("rc05_dig", nb.html(c_dig, marks={"[1] 240   2": 1, "0.220 0.9960": 2, "time_reversed": 3}))
S("rc05_dplot", nb.html(c_dplot))
S("rc05_nar", nb.html(c_nar))
S("rc05_pre", nb.html(c_pre, marks={"Loading required package: ggplot2": 1, "[1] 233": 2, "lower upper": 3}))
S("rc05_get", nb.html(c_get, marks={"nrisk.hat": 1, "censor.hat": 2, "     5   213   233    24   164": 3,
                                    "max absolute error": 4, "D= 0.03433": 5}))
S("rc05_rec", nb.html(c_rec, marks={"time status treat": 1, "152 148": 2, "164": 3}))
S("rc05_cmp", nb.html(c_cmp, marks={"0.5567": 1, "469.6": 2, "908.0": 3}))
S("rc05_cplot", nb.html(c_cplot))
S("rc05_fit", nb.html(c_fit, marks={"dAIC": 1, "19.5   3.10": 2, "23.1   3.40": 3}))
S("rc05_nar3", nb.html(c_nar3, marks={"events after24": 1, "20.7   3.19": 2}))
S("rc05_t1", nb.html(c_t1))
S("rc05_t2", nb.html(c_t2))
S("rc05_t3", nb.html(c_t3, marks={"TRUE": 1}))
nb.save_ipynb("5장 논문의 KM 곡선에서 환자 자료 재구성하기 (R과 통계분석)",
              intro="논문의 Kaplan-Meier 그림에서 읽어 낸 좌표와 위험집합 표로 환자 단위 자료를 재구성하고(Guyot 방법, IPDfromKM 패키지), "
                    "원자료와 비교합니다. Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_dig.n: "## a. 그림에서 읽어 낸 것 준비하기", c_pre.n: "## b. 환자 자료 재구성하기",
                     c_cmp.n: "## c. 원자료와 비교하기", c_nar3.n: "## d. 위험집합 표가 촘촘하면",
                     c_t1.n: "## e. 과제 정답"})
print("rc05 fragments and notebook written")
