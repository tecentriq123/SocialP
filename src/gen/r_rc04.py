"""R과 통계분석 4장 · 모수적 생존모형과 외삽 — R cells run for real with rkit (R 4.5, survival, flexsurv).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc04.py      (NOMARK=1 … 로 돌리면 모든 셀의 출력을 찍는다)

자료는 보건통계학 기초 23장 d 절과 실습 23의 pub/data/rcc_trial.csv(표준요법 B군 300명, 사망 145명).
끝의 대조 블록에서 23장 d 절 표(AIC, 2·5·10·20년 생존율, 20년까지의 평균 생존)와 R 결과를 맞춘다.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc04")

# ---------------------------------------------------------------- a. 자료와 Kaplan-Meier
c_load = nb.cell('''
install.packages(c("survival", "flexsurv"))   # 처음 한 번만
library(survival)
library(flexsurv)

options(HTTPUserAgent = "Mozilla/5.0")   # 사이트가 프로그램의 기본 요청을 막을 때를 대비해 브라우저처럼 요청
BASE <- "https://socialp-ajou.tecentriq12.workers.dev/data/"
d <- read.csv(paste0(BASE, "rcc_trial.csv"))
str(d)
table(d$event)
''', title="자료 불러오기")

c_km = nb.cell('''
km <- survfit(Surv(months, event) ~ 1, data = d)
km
summary(km, times = c(6, 12, 24))
''', title="관찰한 생존곡선(Kaplan-Meier)")

# ---------------------------------------------------------------- b. 곡선 하나
c_wei = nb.cell('''
fit_w <- flexsurvreg(Surv(months, event) ~ 1, data = d, dist = "weibull")
fit_w
''', title="와이블 곡선 맞추기")

c_wsum = nb.cell('''
set.seed(2026)   # 신뢰구간을 모의실험으로 구하므로 고정
s_w <- summary(fit_w, t = c(24, 60, 120, 240), type = "survival", tidy = TRUE)
round(s_w, 4)
''', title="정한 시점의 생존율")

# ---------------------------------------------------------------- c. 여섯 분포
c_six = nb.cell('''
dists <- c(exp = "exp", weibull = "weibull", gompertz = "gompertz",
           lnorm = "lnorm", llogis = "llogis", gengamma = "gengamma")
fits <- lapply(dists, function(k)
  flexsurvreg(Surv(months, event) ~ 1, data = d, dist = k))

fit_tab <- data.frame(
  model = names(fits),
  k     = sapply(fits, function(f) f$npars),
  AIC   = sapply(fits, AIC),
  BIC   = sapply(fits, BIC))
fit_tab$dAIC <- fit_tab$AIC - min(fit_tab$AIC)
fit_tab[order(fit_tab$AIC), ]
''', title="여섯 분포를 맞추고 AIC·BIC 비교")

# ---------------------------------------------------------------- d. 20년 외삽
c_ext = nb.cell('''
times <- c(24, 60, 120, 240)            # 2, 5, 10, 20년 (개월)
surv_at <- function(f) summary(f, t = times, type = "survival",
                               ci = FALSE, tidy = TRUE)$est
ext <- t(sapply(fits, surv_at))
colnames(ext) <- paste0(times / 12, "y")
round(100 * ext, 1)
''', title="2·5·10·20년 생존율")

c_mean = nb.cell('''
rmst20 <- sapply(fits, function(f)
  summary(f, type = "rmst", t = 240, ci = FALSE, tidy = TRUE)$est)
mean_all <- sapply(fits, function(f)
  summary(f, type = "mean", ci = FALSE, tidy = TRUE)$est)
round(data.frame(rmst_20y = rmst20 / 12, mean_unrestricted = mean_all / 12), 2)
''', title="20년까지의 평균 생존과 제한 없는 평균 (년)")

# ---------------------------------------------------------------- e. 그림
c_plot = nb.cell('''
library(ggplot2)
grid <- seq(0, 240, by = 1)
curves <- do.call(rbind, lapply(names(fits), function(m)
  data.frame(model = m, months = grid,
             surv = summary(fits[[m]], t = grid, ci = FALSE, tidy = TRUE)$est)))
km_df <- data.frame(months = c(0, km$time), surv = c(1, km$surv))

ggplot(curves, aes(months / 12, surv, colour = model)) +
  geom_line(linewidth = 0.8) +
  geom_step(data = km_df, aes(months / 12, surv), colour = "black",
            inherit.aes = FALSE) +
  geom_vline(xintercept = 30 / 12, linetype = "dashed") +
  labs(x = "Years since randomisation", y = "Overall survival", colour = NULL) +
  theme_minimal(base_size = 13)
''', title="관찰 구간과 외삽 구간의 곡선", fig=(7.5, 4.6))

c_haz = nb.cell('''
library(muhaz)
emp <- muhaz(d$months, d$event, max.time = 28)      # 자료에서 본 위험률(평활)
haz <- do.call(rbind, lapply(names(fits), function(m)
  data.frame(model = m, months = grid[grid > 0],
             h = summary(fits[[m]], t = grid[grid > 0], type = "hazard",
                         ci = FALSE, tidy = TRUE)$est)))

ggplot(haz, aes(months / 12, h * 12, colour = model)) +
  geom_line(linewidth = 0.8) +
  geom_line(data = data.frame(months = emp$est.grid, h = emp$haz.est),
            aes(months / 12, h * 12), colour = "black", linewidth = 1.1,
            inherit.aes = FALSE) +
  coord_cartesian(ylim = c(0, 1.2)) +
  labs(x = "Years since randomisation", y = "Hazard (deaths per person-year)",
       colour = NULL) +
  theme_minimal(base_size = 13)
''', title="위험률: 자료에서 본 것과 모형이 그리는 것", fig=(7.5, 4.6))

# ---------------------------------------------------------------- f. 스플라인과 PH·AFT
c_spl = nb.cell('''
spl <- lapply(c(k1 = 1, k2 = 2), function(k)
  flexsurvspline(Surv(months, event) ~ 1, data = d, k = k, scale = "hazard"))
data.frame(
  model = names(spl),
  AIC   = sapply(spl, AIC),
  surv_10y = sapply(spl, function(f)
    summary(f, t = 120, ci = FALSE, tidy = TRUE)$est),
  rmst_20y = sapply(spl, function(f)
    summary(f, type = "rmst", t = 240, ci = FALSE, tidy = TRUE)$est) / 12)
''', title="스플라인 모형(매듭 1개, 2개)")

c_ph = nb.cell('''
fit_wph <- flexsurvreg(Surv(months, event) ~ 1, data = d, dist = "weibullPH")
rbind(AFT = fit_w$res[, "est"], PH = fit_wph$res[, "est"])
c(AFT = AIC(fit_w), PH = AIC(fit_wph))
''', title="같은 와이블의 두 가지 모수 표현")

# ---------------------------------------------------------------- g. 과제 (정답 셀)
c_t1 = nb.cell('''
d$years <- d$months / 12
fit_wy <- flexsurvreg(Surv(years, event) ~ 1, data = d, dist = "weibull")
fit_wy$res[, "est"]
c(months = AIC(fit_w), years = AIC(fit_wy), diff = AIC(fit_w) - AIC(fit_wy))
summary(fit_wy, t = 20, ci = FALSE, tidy = TRUE)$est
''', title="과제 1 정답: 시간 단위를 년으로")

c_t2 = nb.cell('''
d18 <- d
d18$event  <- ifelse(d$months > 18, 0, d$event)   # 18개월에 자료를 마감했다면
d18$months <- pmin(d$months, 18)
fits18 <- lapply(dists, function(k)
  flexsurvreg(Surv(months, event) ~ 1, data = d18, dist = k))
data.frame(
  events = sum(d18$event),
  AIC    = round(sapply(fits18, AIC), 1),
  surv_10y_18m = round(100 * sapply(fits18, function(f)
    summary(f, t = 120, ci = FALSE, tidy = TRUE)$est), 1),
  surv_10y_30m = round(100 * ext[, "10y"], 1))
''', title="과제 2 정답: 18개월에 마감한 자료로 다시")

c_t3 = nb.cell('''
ext5 <- 100 * ext[, "5y"]
round(sort(abs(ext5 - 20)), 1)     # 등록자료의 5년 생존율 20%(가상)와의 차이
''', title="과제 3 정답: 외부 자료와 비교")

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

# ---------------------------------------------------------------- 23장 d 절과 대조
out = {c.n: c.text for c in nb.cells}
for want in ("gompertz 2 1374.904", "weibull   weibull 2 1375.921", "exp           exp 1 1376.849",
             "lnorm       lnorm 2 1389.806", "llogis     llogis 2 1379.672", "gengamma gengamma 3 1376.882"):
    assert want in out[c_six.n], want
for row in ("exp      56.6 24.1  5.8 0.3", "weibull  56.4 19.5  2.7 0.0", "gompertz 56.7 11.1  0.0 0.0",
            "lnorm    56.9 32.6 17.7 8.1", "llogis   56.7 28.3 13.8 6.0", "gengamma 56.7 12.1  0.0 0.0"):
    assert row in out[c_ext.n], row
for m, v in (("exp", "3.50"), ("weibull", "3.10"), ("gompertz", "2.60"), ("lnorm", "5.19"), ("llogis", "4.62"), ("gengamma", "2.64")):
    assert re.search(rf"^{m}\s+{re.escape(v)}\s", out[c_mean.n], re.M), (m, v)
assert "shape   1.1454" in out[c_wei.n] and "scale  39.0593" in out[c_wei.n]      # 실습 23 lifelines rho_ 1.1454, lambda_ 39.059

S = nb.save_fragment
S("rc04_load", nb.html(c_load, marks={"300 obs. of  3 variables": 1, "155 145": 2}))
S("rc04_km", nb.html(c_km, marks={"145     NA": 1, "0.557": 2}))
S("rc04_wei", nb.html(c_wei, marks={"1.1454": 1, "39.0593": 2, "Log-likelihood = -685.9603": 3, "AIC = 1375.921": 4}))
S("rc04_wsum", nb.html(c_wsum, marks={"0.5641": 1, "0.0003": 2, "0.0044": 3}))
S("rc04_six", nb.html(c_six, marks={"1374.904": 1, "1380.552": 2, "14.902453": 3}))
S("rc04_ext", nb.html(c_ext, marks={"56.6": 1, "17.7": 2}))
S("rc04_mean", nb.html(c_mean, marks={"7.52": 1, "8.75": 2}))
S("rc04_plot", nb.html(c_plot))
S("rc04_haz", nb.html(c_haz))
S("rc04_spl", nb.html(c_spl, marks={"1377.120": 1, "0.016952009": 2}))
S("rc04_ph", nb.html(c_ph, marks={"0.01502429": 1, "1375.921 1375.921": 2}))
S("rc04_t1", nb.html(c_t1, marks={"3.254939": 1, "720.6229": 2}))
S("rc04_t2", nb.html(c_t2, marks={"949.6": 1, "25.1": 2}))
S("rc04_t3", nb.html(c_t3, marks={"0.5": 1}))
nb.save_ipynb("4장 모수적 생존모형과 외삽 (R과 통계분석)",
              intro="보건통계학 기초 23장 d 절의 가상 시험(표준요법군 300명)의 전체생존 자료로 여섯 분포와 스플라인 모형을 맞추고 20년까지 외삽합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={1: "## a. 자료와 관찰한 생존곡선", 3: "## b. 곡선 하나 맞추기", 5: "## c. 여섯 분포 비교",
                     6: "## d. 20년까지의 생존율과 평균 생존", 8: "## e. 그림으로 점검", 10: "## f. 스플라인과 모수 표현",
                     12: "## g. 과제 정답"})
print("rc04 fragments and notebook written")
