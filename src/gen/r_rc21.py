"""R과 파이썬으로 하는 통계분석 14장 · 중단시계열분석과 이중차분법 — R cells run for real with rkit.

run:  python3 gen/r_rc21.py      (NOMARK=1 python3 gen/r_rc21.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 18장의 '직접 분석' 상자와 본문에 있던 파이썬 코드를 2026-11-19 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 18장의 모의자료(시군구 60곳 × 60개월, 수면진정제 처방 점검 시범사업; pub/data/sedative_district_month.csv,
gen/data_lab18.py = gen/nums_ch18.py의 simulate(), seed 91). 숫자는 gen/_ch18_nums.json과 대조한다.
AR(1) 오차 모형은 파이썬(statsmodels GLSAR, 첫 관측값을 뺌)과 달리 prais 패키지의 Prais–Winsten 방법으로 적합하므로 끝자리가 다르다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc21")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 월별 계열과 전후 비교
c_load = nb.cell(f'''
suppressPackageStartupMessages({{library(sandwich); library(lmtest)}})
raw <- read.csv("{D}sedative_district_month.csv")    # 한 줄 = 시군구의 한 달
str(raw)
raw$time <- match(raw$month, sort(unique(raw$month)))       # 2015년 1월 = 1, …, 2019년 12월 = 60
s <- aggregate(cbind(n_sed, n_pat) ~ treat + time, data = raw, sum)   # 지역별 월별 분자와 분모
s$rate <- s$n_sed / s$n_pat * 1000                          # 65세 이상 외래환자 1,000명당 처방률
s$post <- as.integer(s$time >= 37)                          # 시행(2018년 1월) 뒤 = 1
s$time_after <- pmax(0, s$time - 36)                        # 시행 첫 달 = 1
its  <- s[s$treat == 1, ]                                   # 시범 지역 계열
comp <- s[s$treat == 0, ]                                   # 비교 지역 계열
its[its$time %in% c(1, 2, 36, 37, 38, 60), c("time", "post", "time_after", "n_sed", "n_pat", "rate")]
''', title="자료 불러오기와 지역별 월별 계열 (18장 a·b 절)")

c_naive = nb.cell('''
welch <- function(x, post) {                     # 시행 전후 평균의 Welch t 검정
  t <- t.test(x[post == 1], x[post == 0])
  round(c(pre = mean(x[post == 0]), post = mean(x[post == 1]),
          diff = unname(t$estimate[1] - t$estimate[2]), lower = t$conf.int[1], upper = t$conf.int[2]), 2)
}
rbind(pilot = welch(its$rate, its$post), comparison = welch(comp$rate, comp$post))
round(rbind(pilot = tapply(its$rate, (its$time - 1) %/% 12 + 2015, mean),
            comparison = tapply(comp$rate, (comp$time - 1) %/% 12 + 2015, mean)), 2)   # 연도별 평균
''', title="전후 평균 비교와 연도별 평균 (18장 a 절)")

c_claims = nb.cell('''
claims <- data.frame(patient = c(1, 1, 1, 1, 2, 2, 3),          # 명세서 한 줄 = 진료 한 건
                     date = as.Date(c("2018-01-05", "2018-01-20", "2018-03-02", "2018-03-16",
                                      "2018-01-11", "2018-02-08", "2018-03-30")),
                     sedative = c(1, 1, 1, 0, 0, 1, 0))             # 수면진정제 처방이 있으면 1
claims$ym <- format(claims$date, "%Y-%m")
n_unique <- function(x) length(unique(x))                          # 건수가 아니라 환자 수
den <- tapply(claims$patient, claims$ym, n_unique)                 # 분모: 외래 이용 환자
sed <- claims[claims$sedative == 1, ]
num <- tapply(sed$patient, sed$ym, n_unique)[names(den)]           # 분자: 처방받은 환자
num[is.na(num)] <- 0                                               # 처방이 한 건도 없는 달은 0
data.frame(ym = names(den), num = as.vector(num), den = as.vector(den))
''', title="명세서에서 월별 분자와 분모 세기 (18장 d 절)")

# ------------------------------------------------------------------ b. 분절회귀
c_its = nb.cell('''
fit <- lm(rate ~ time + post + time_after, data = its)    # 분절회귀
round(cbind(coef = coef(fit), se_ols = sqrt(diag(vcov(fit)))), 4)
dwtest(fit)                                                # Durbin–Watson(기본: 양의 자기상관을 검정)
round(acf(resid(fit), lag.max = 6, plot = FALSE)$acf[2:7], 3)   # 시차 1–6의 자기상관
''', title="분절회귀와 자기상관 확인 (18장 b 절)")

c_nw = nb.cell('''
V <- NeweyWest(fit, lag = 3, prewhite = FALSE)             # 시차 3, 사전 백색화 없음
round(cbind(se = sqrt(diag(V)), coefci(fit, vcov. = V, df = Inf)), 3)   # 정규분포로 구한 95% CI
round(sqrt(diag(NeweyWest(fit)))[c("post", "time_after")], 3)           # 기본값(시차 자동, 사전 백색화)
car::deltaMethod(fit, "post + 12*time_after", vcov. = V)   # 시행 12개월째의 효과
''', title="Newey–West 표준오차와 12개월째 효과 (18장 b 절)")

c_lags = nb.cell('''
se_lag <- sapply(c(1, 3, 6, 12), function(l)
  sqrt(diag(NeweyWest(fit, lag = l, prewhite = FALSE)))[c("post", "time_after")])
colnames(se_lag) <- paste0("lag", c(1, 3, 6, 12)); round(se_lag, 3)
round(sqrt(diag(NeweyWest(fit, lag = 3, prewhite = FALSE, adjust = TRUE)))[c("post", "time_after")], 3)   # 소표본 보정
''', title="시차를 바꾸면 표준오차는 (18장 b 절)")

c_ar1 = nb.cell('''
suppressPackageStartupMessages(library(prais))
pw <- suppressMessages(prais_winsten(rate ~ time + post + time_after, data = its, index = "time"))
sp <- summary(pw)
cf <- sp$coefficients[c("post", "time_after"), 1:2]
round(cbind(cf, lower = cf[, 1] - qt(0.975, 56) * cf[, 2], upper = cf[, 1] + qt(0.975, 56) * cf[, 2]), 3)
round(c(rho = pw$rho[nrow(pw$rho), 1], sp$dw), 3)          # ρ, 변환 전후 Durbin–Watson
X  <- as.matrix(its[, c("time", "post", "time_after")])
ar <- arima(its$rate, order = c(1, 0, 0), xreg = X, method = "ML")   # 같은 AR(1) 구조를 최대우도로
se <- sqrt(diag(ar$var.coef))
round(cbind(est = ar$coef, lower = ar$coef - 1.96 * se, upper = ar$coef + 1.96 * se)[c("ar1", "post", "time_after"), ], 3)
''', title="AR(1) 오차 모형: Prais–Winsten과 ARIMA (18장 b 절)")

c_season = nb.cell('''
its$s1 <- sin(2 * pi * its$time / 12); its$c1 <- cos(2 * pi * its$time / 12)   # time 1 = 1월
fs <- lm(rate ~ time + post + time_after + s1 + c1, data = its)
round(cbind(coef = coef(fs), coefci(fs, vcov. = NeweyWest(fs, lag = 3, prewhite = FALSE), df = Inf))[c("post", "time_after"), ], 3)
round(c(amplitude = sqrt(sum(coef(fs)[c("s1", "c1")]^2)), sigma = sigma(fs), dw = unname(dwtest(fs)$statistic)), 2)
fm <- lm(rate ~ time + post + time_after + factor((time - 1) %% 12), data = its)   # 달 가변수 11개
round(coef(fm)[c("post", "time_after")], 3)
''', title="계절 항 넣기 (18장 b 절)")

c_pois = nb.cell('''
pois <- glm(n_sed ~ time + post + time_after + offset(log(n_pat)), family = poisson, data = its)
Vp <- NeweyWest(pois, lag = 3, prewhite = FALSE)
round(c(dispersion = sum(residuals(pois, "pearson")^2) / pois$df.residual,
        se_default = sqrt(vcov(pois)["post", "post"]), se_nw = sqrt(Vp["post", "post"])), 4)
round(exp(c(coef(pois)["post"], coefci(pois, vcov. = Vp, df = Inf)["post", ])), 3)   # 수준 변화의 비
d12 <- car::deltaMethod(pois, "post + 12*time_after", vcov. = Vp)
round(exp(unlist(d12[c("Estimate", "2.5 %", "97.5 %")])), 3)                         # 12개월째의 비
''', title="결과가 건수이면: 포아송 분절회귀 (18장 b 절)")

# ------------------------------------------------------------------ c. 대조군이 있는 중단시계열
c_comp = nb.cell('''
stopifnot(all(its$time == comp$time))                       # 같은 행이 같은 달
fc <- lm(rate ~ time + post + time_after, data = comp)      # 비교 지역
dd <- its[, c("time", "post", "time_after")]
dd$rate <- its$rate - comp$rate                             # 달마다 시범 − 비교
fd <- lm(rate ~ time + post + time_after, data = dd)
nw <- function(f, L = c(0, 0, 1, 12)) {                     # 계수 조합과 Newey–West 95% CI
  V <- NeweyWest(f, lag = 3, prewhite = FALSE)
  e <- sum(L * coef(f)); se <- sqrt(drop(t(L) %*% V %*% L))
  round(c(est = e, lower = e - 1.96 * se, upper = e + 1.96 * se), 3)
}
rbind(pilot = nw(fit), comparison = nw(fc), difference = nw(fd))       # 12개월째 효과
round(cbind(coef = coef(fd), coefci(fd, vcov. = NeweyWest(fd, lag = 3, prewhite = FALSE), df = Inf)), 3)
''', title="세 계열의 분절회귀와 차이 계열 (18장 c 절)")

c_stack = nb.cell('''
keep <- c("rate", "time", "post", "time_after")
both <- rbind(cbind(its[, keep], group = 1), cbind(comp[, keep], group = 0))   # 120줄
fb <- lm(rate ~ (time + post + time_after) * group, data = both)
round(coef(fb)[c("time:group", "post:group", "time_after:group")], 3)
k <- "post:group"
round(c(default = sqrt(vcov(fb)[k, k]),
        cluster_month = sqrt(vcovCL(fb, cluster = ~ time)[k, k]),
        cluster_month_lag3 = sqrt(vcovPL(fb, cluster = ~ group, order.by = ~ time, lag = 3)[k, k]),
        difference_nw = sqrt(NeweyWest(fd, lag = 3, prewhite = FALSE)["post", "post"])), 3)
round(cor(resid(fit), resid(fc)), 2)                        # 두 지역 잔차의 상관
''', title="두 계열을 한 모형에 넣으면 (18장 c 절)")

# ------------------------------------------------------------------ d. 이중차분법
c_did = nb.cell('''
p <- raw                                                    # 시군구-월 3,600줄
p$rate <- p$n_sed / p$n_pat * 1000; p$post <- as.integer(p$time >= 37)
p$w <- ave(p$n_pat, p$unit)                                 # 시군구의 평균 환자 수(가중치)
sum(is.na(p$rate))
did <- lm(rate ~ treat * post, data = p, weights = w)
Vc  <- vcovCL(did, cluster = ~ unit)                        # 시군구 군집 강건 표준오차
round(cbind(coef = coef(did), se = sqrt(diag(Vc)), coefci(did, vcov. = Vc, df = Inf)), 3)
round(coefci(did, vcov. = Vc, df = 59)["treat:post", ], 3)  # 자유도 59(군집 수 − 1)의 t
did_u <- lm(rate ~ treat * post, data = p)                  # 가중치 없이
round(c(coef(did_u)["treat:post"], coefci(did_u, vcov. = vcovCL(did_u, cluster = ~ unit), df = Inf)["treat:post", ]), 2)
tw <- lm(rate ~ treat:post + factor(unit) + factor(time), data = p, weights = w)   # 이원 고정효과
round(coef(tw)["treat:post"], 3)
''', title="이중차분 회귀와 군집 강건 표준오차 (18장 d 절)")

c_60 = nb.cell('''
ch <- do.call(rbind, lapply(split(p, p$unit), function(u)
  data.frame(treat = u$treat[1], w = u$w[1],
             change = mean(u$rate[u$post == 1]) - mean(u$rate[u$post == 0]))))   # 시군구마다 전후 변화
round(rbind(mean = tapply(ch$change, ch$treat, mean), sd = tapply(ch$change, ch$treat, sd)), 2)
f60 <- lm(change ~ treat, data = ch, weights = w)
round(c(coef(f60)["treat"], coefci(f60, vcov. = vcovHC(f60, type = "HC1"), df = Inf)["treat", ]), 2)
''', title="시군구 60곳을 한 줄씩으로 (18장 d 절)")

c_es = nb.cell('''
ft <- lm(rate ~ treat * time, data = p[p$post == 0, ], weights = w)    # 시행 전 36개월의 기울기 차이
Vt <- vcovCL(ft, cluster = ~ unit)
round(c(coeftest(ft, vcov. = Vt, df = Inf)["treat:time", c(1, 4)], coefci(ft, vcov. = Vt, df = Inf)["treat:time", ]), 3)
p$half <- (p$time - 1) %/% 6 + 1                            # 반기 1–10, 6 = 시행 직전(기준)
hs <- setdiff(1:10, 6)
for (h in hs) p[[paste0("h", h)]] <- as.integer(p$half == h & p$treat == 1)
fes <- lm(reformulate(c(paste0("h", hs), "factor(unit)", "factor(time)"), "rate"), data = p, weights = w)
Ve  <- vcovCL(fes, cluster = ~ unit)
round(cbind(est = coef(fes)[paste0("h", hs)], coefci(fes, vcov. = Ve, df = Inf)[paste0("h", hs), ]), 2)
lh <- car::linearHypothesis(fes, paste0("h", 1:5, " = 0"), vcov. = Ve, test = "Chisq")
round(c(chisq = lh$Chisq[2], p = lh$`Pr(>Chisq)`[2]), 3)     # 시행 전 다섯 계수가 모두 0인가
''', title="평행 추세 점검과 사건연구 (18장 d 절)")

c_four = nb.cell('''
rbind(its = nw(fit, c(0, 0, 1, 12.5)),                      # 시행 후 24개월의 평균 효과
      controlled_its = nw(fd, c(0, 0, 1, 12.5)),
      did = round(c(est = unname(coef(did)["treat:post"]), coefci(did, vcov. = Vc, df = Inf)["treat:post", ]), 3))
''', title="같은 자료에서 네 분석 (18장 d 절)")

# ------------------------------------------------------------------ e. 과제
c_t1 = nb.cell('''
ad <- its[!(its$time %in% 37:39), ]                         # 시행 직후 3개월을 뺌
fa <- lm(rate ~ time + post + time_after, data = ad)
round(cbind(coef = coef(fa), coefci(fa, vcov. = NeweyWest(fa, lag = 3, prewhite = FALSE), df = Inf))[c("post", "time_after"), ], 3)
nw(fa)                                                      # 12개월째
''', title="과제 1 정답. 적응 기간을 뺀 분절회귀")

c_t2 = nb.cell('''
k <- "treat:post"
round(c(ignore_cluster = sqrt(vcov(tw)[k, k]), cluster_unit = sqrt(vcovCL(tw, cluster = ~ unit)[k, k])), 3)
round(c(ignore_cluster = sqrt(vcov(did)[k, k]), cluster_unit = sqrt(Vc[k, k])), 3)
''', title="과제 2 정답. 군집을 무시하면")

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
    c_load: ['3600 obs. of  5 variables', '81.67696'],
    c_naive: ['-17.46 -19.25 -15.67', '-10.06 -11.40  -8.72'],
    c_claims: ['2018-01   1   2'],
    c_its: ['-7.3707 0.8090', 'DW = 0.94505', '0.513'],
    c_nw: ['1.117 -9.559 -5.182', '1.293      0.088', '-10.0860   1.0559 -12.1554 -8.0166'],
    c_lags: ['1.156      0.065'],
    c_ar1: ['-7.131      1.128 -9.390 -4.871', '0.530       0.945       1.781', '-7.138 -9.272 -5.004'],
    c_season: ['-7.772 -9.168 -6.376'],
    c_pois: ['6.8840     0.0037     0.0137', '0.916  0.892  0.941', '0.877    0.856    0.898'],
    c_comp: ['-2.863  -4.824 -0.902', '-7.223  -8.031 -6.415', '-0.003 -0.027  0.022'],
    c_stack: ['1.089              0.429              0.416              0.402', '[1] 0.83'],
    c_did: ['-7.402 0.238  -7.868 -6.935', '-7.878 -6.925', '-7.26      -7.70      -6.82'],
    c_60: ['-7.40  -7.87  -6.93'],
    c_es: ['-0.003    0.881   -0.037    0.032', '2.598 0.762'],
    c_four: ['-10.199 -12.286 -8.112', '-7.322  -8.142 -6.502'],
    c_t1: ['-8.540 -10.604 -6.476', '-10.426'],
    c_t2: ['0.219          0.242', '0.469          0.238'],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc21_load", nb.html(c_load, marks={'3600 obs. of  5 variables': 1, '81.67696': 2}))
S("rc21_naive", nb.html(c_naive, marks={'-17.46 -19.25 -15.67': 1, '-10.06 -11.40  -8.72': 2}))
S("rc21_claims", nb.html(c_claims, marks={'2018-01   1   2': 1}))
S("rc21_its", nb.html(c_its, marks={'-7.3707 0.8090': 1, 'DW = 0.94505': 2, '0.513': 3}))
S("rc21_nw", nb.html(c_nw, marks={'1.117 -9.559 -5.182': 1, '1.293      0.088': 2, '-10.0860   1.0559 -12.1554 -8.0166': 3}))
S("rc21_lags", nb.html(c_lags, marks={'1.156      0.065': 1}))
S("rc21_ar1", nb.html(c_ar1, marks={'-7.131      1.128 -9.390 -4.871': 1, '0.530       0.945       1.781': 2, '-7.138 -9.272 -5.004': 3}))
S("rc21_season", nb.html(c_season, marks={'-7.772 -9.168 -6.376': 1}))
S("rc21_pois", nb.html(c_pois, marks={'6.8840     0.0037     0.0137': 1, '0.916  0.892  0.941': 2, '0.877    0.856    0.898': 3}))
S("rc21_comp", nb.html(c_comp, marks={'-2.863  -4.824 -0.902': 1, '-7.223  -8.031 -6.415': 2, '-0.003 -0.027  0.022': 3}))
S("rc21_stack", nb.html(c_stack, marks={'1.089              0.429              0.416              0.402': 1, '[1] 0.83': 2}))
S("rc21_did", nb.html(c_did, marks={'-7.402 0.238  -7.868 -6.935': 1, '-7.878 -6.925': 2, '-7.26      -7.70      -6.82': 3}))
S("rc21_60", nb.html(c_60, marks={'-7.40  -7.87  -6.93': 1}))
S("rc21_es", nb.html(c_es, marks={'-0.003    0.881   -0.037    0.032': 1, '2.598 0.762': 2}))
S("rc21_four", nb.html(c_four))
S("rc21_t1", nb.html(c_t1))
S("rc21_t2", nb.html(c_t2))
nb.save_ipynb("14장 중단시계열분석과 이중차분법 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 18장의 모의자료(시군구 60곳 × 60개월, 수면진정제 처방 점검 시범사업)로 전후 비교, 분절회귀와 Newey–West 표준오차, "
                    "AR(1) 오차 모형, 계절 항, 포아송 분절회귀, 대조군이 있는 중단시계열, 이중차분 회귀와 군집 강건 표준오차, 사건연구를 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"sandwich\", \"lmtest\", \"car\", \"prais\"))를 실행합니다.",
              notes={c_load.n: "## a. 월별 계열과 전후 비교", c_its.n: "## b. 분절회귀", c_comp.n: "## c. 대조군이 있는 중단시계열",
                     c_did.n: "## d. 이중차분법", c_t1.n: "## e. 과제 정답"})
print("rc21 fragments and notebook written")
