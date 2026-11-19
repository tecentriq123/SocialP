"""R과 파이썬으로 하는 통계분석 20장 · 비용 자료 분석 — R cells run for real with rkit.

run:  python3 gen/r_rc23.py      (NOMARK=1 python3 gen/r_rc23.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 21장의 '직접 분석' 상자와 본문에 있던 파이썬 코드(pandas, scipy, statsmodels)를 2026-11-19 이 장으로 옮기며 R로 다시 돌린다.
자료: 21장의 가상 1년 의료비 자료(신세포암 1차 치료, 3,000명; pub/data/rcc_cost.csv, gen/data_lab21.py = 실습 21과 같은 파일)와
a 절의 명세서 11건(표 21-2). 부트스트랩은 난수가 달라 21장(scipy, numpy)의 끝자리와 조금 다를 수 있다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc23")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 명세서에서 환자별 비용
c_claims = nb.cell('''
claims <- data.frame(                                    # 표 21-2의 명세서 11건(금액은 만원)
  pid   = c("P01", "P01", "P01", "P01", "P01", "P02", "P02", "P02", "P03", "P03", "P03"),
  date  = as.Date(c("2024-03-04", "2024-03-04", "2024-04-03", "2024-06-10", "2025-03-10", "2024-05-20",
                    "2024-05-20", "2024-06-19", "2024-02-15", "2024-07-01", "2024-07-01")),
  type  = c("outpt", "pharm", "pharm", "inpt", "outpt", "outpt", "pharm", "pharm", "inpt", "outpt", "pharm"),
  total = c(38, 190, 190, 412, 25, 41, 120, 120, 300, 36, 190),          # 요양급여비용
  oop   = c(1.9, 9.5, 9.5, 20.6, 1.3, 2.1, 6.0, 6.0, 15.0, 1.8, 9.5))    # 그중 본인부담금
cohort <- data.frame(pid = c("P01", "P02", "P03"),
                     index_date = as.Date(c("2024-03-04", "2024-05-20", "2024-07-01")))
m <- merge(claims, cohort, by = "pid")                   # 명세서마다 치료 시작일을 붙임
m$day <- as.numeric(m$date - m$index_date)
win <- m[m$day >= 0 & m$day < 365, ]                     # 관찰 창: 시작일부터 364일째까지
nrow(win)
win$pid  <- factor(win$pid, levels = cohort$pid)         # 대상자 목록의 모든 환자를 남김
win$type <- factor(win$type, levels = c("pharm", "outpt", "inpt"))
cost <- as.data.frame.matrix(xtabs(total ~ pid + type, data = win))   # 명세서가 없는 칸은 0
cost$total   <- rowSums(cost)                                         # 보건의료체계 관점
cost$insurer <- tapply(win$total - win$oop, win$pid, sum)             # 보험자 관점(공단 부담금)
cost
round(colMeans(cost), 1)                                              # 1인당 평균
mean(win$total[win$type == "inpt"])                                   # 입원한 환자만 평균하면
''', title="명세서에서 환자 한 명의 1년 비용 만들기 (21장 a 절)")

# ------------------------------------------------------------------ b. 분포와 평균 비교
c_load = nb.cell(f'''
suppressPackageStartupMessages({{library(sandwich); library(lmtest); library(boot)}})
d <- read.csv("{D}rcc_cost.csv")          # 한 줄 = 환자 한 명의 1년 비용(만원)
d$A <- as.integer(d$arm == "A")            # 신약 A = 1, 표준요법 B = 0
str(d)
sapply(split(d$cost, d$arm), function(x)
  round(c(n = length(x), mean = mean(x), sd = sd(x), median = median(x), max = max(x)), 1))
round(tapply(d$cost_inp == 0, d$arm, mean), 3)   # 입원비가 0원인 비율
''', title="자료 불러오기와 군별 요약 (21장 b 절)")

c_ttest = nb.cell('''
a <- d$cost[d$A == 1]; b <- d$cost[d$A == 0]
round(c(diff = mean(a) - mean(b),
        welch = t.test(a, b)$conf.int,                        # 기본: Welch
        student = t.test(a, b, var.equal = TRUE)$conf.int), 1)
''', title="평균 차이와 t 검정의 신뢰구간 (21장 b 절)")

c_boot = nb.cell('''
set.seed(21)
bt <- boot(d, function(x, i) mean(x$cost[i][x$A[i] == 1]) - mean(x$cost[i][x$A[i] == 0]),
           R = 9999, strata = d$A)                 # 군 안에서 따로 복원추출
round(sd(bt$t), 1)                                 # 부트스트랩 표준오차
ci <- boot.ci(bt, type = c("perc", "bca"))
iv <- rbind(percentile = ci$percent[4:5], bca = ci$bca[4:5]); colnames(iv) <- c("lower", "upper")
round(iv, 1)
''', title="부트스트랩 신뢰구간 (21장 b 절)")

c_skew = nb.cell('''
round(c(median_A = median(a), median_B = median(b), diff = median(a) - median(b)), 1)
w <- wilcox.test(a, b)                              # Mann-Whitney 검정
cat("P(A > B) =", round(unname(w$statistic) / (length(a) * length(b)), 3), "  P =", signif(w$p.value, 3), "\n")
lt <- t.test(log(a), log(b))                       # 로그 척도의 평균 차이
round(exp(c(ratio = unname(lt$estimate[1] - lt$estimate[2]), lt$conf.int)), 3)   # 기하평균의 비
''', title="중앙값, Mann-Whitney 검정, 기하평균의 비 (21장 b 절)")

# ------------------------------------------------------------------ c. 회귀분석
c_glm = nb.cell('''
g <- glm(cost ~ A + age + male + cci + stage4, data = d, family = Gamma(link = "log"))
round(coef(summary(g)), 4)
round(exp(c(ratio = coef(g)[["A"]], confint.default(g)["A", ])), 3)       # 비용의 비와 95% CI
round(summary(g)$dispersion, 3)                                            # 분산 비례 상수
round(exp(coefci(g, vcov. = vcovHC(g, type = "HC0"))["A", ]), 3)          # 강건 표준오차로 구한 구간
''', title="감마 GLM(로그 연결) (21장 c 절)")

c_std = nb.cell('''
d1 <- transform(d, A = 1); d0 <- transform(d, A = 0)      # 모두 A로, 모두 B로
std <- function(f, rows = rep(TRUE, nrow(d)))
  c(all_A = mean(predict(f, d1[rows, ], type = "response")),
    all_B = mean(predict(f, d0[rows, ], type = "response")))
r <- rbind(everyone = std(g), A_users = std(g, d$A == 1), B_users = std(g, d$A == 0))
round(cbind(r, diff = r[, 1] - r[, 2]), 1)
X1 <- model.matrix(formula(g), d1); X0 <- model.matrix(formula(g), d0)   # 델타 방법
mu1 <- exp(drop(X1 %*% coef(g))); mu0 <- exp(drop(X0 %*% coef(g)))
grad <- colMeans(X1 * mu1) - colMeans(X0 * mu0)
se <- sqrt(drop(t(grad) %*% vcov(g) %*% grad))
dif <- mean(mu1) - mean(mu0)
round(c(diff = dif, lower = dif - 1.96 * se, upper = dif + 1.96 * se), 1)
''', title="표준화: 금액 차이와 델타 방법 구간 (21장 c 절)")

c_stdboot = nb.cell('''
set.seed(21)
sb <- boot(d, function(x, i) {                     # 환자를 다시 뽑아 적합부터 표준화까지 되풀이
  y <- x[i, ]
  f <- glm(cost ~ A + age + male + cci + stage4, data = y, family = Gamma(link = "log"))
  mean(predict(f, transform(y, A = 1), type = "response")) -
    mean(predict(f, transform(y, A = 0), type = "response"))
}, R = 1000)
round(c(se = sd(sb$t), quantile(sb$t, c(0.025, 0.975))), 1)
''', title="금액 차이의 부트스트랩 구간 (21장 c 절)")

c_default = nb.cell('''
coef(glm(cost ~ A + age + male + cci + stage4, data = d, family = Gamma()))[["A"]]   # 연결함수를 빼면 역수 연결
coef(glm(cost ~ A + age + male + cci + stage4, data = d))[["A"]]                     # family를 빼면 선형회귀
''', title="기본값이 바꾸는 것 (21장 c 절)")

c_models = nb.cell('''
lm1 <- lm(cost ~ A + age + male + cci + stage4, data = d)                     # 원래 비용의 선형회귀
ll  <- lm(log(cost) ~ A + age + male + cci + stage4, data = d)                # 로그 변환 회귀
sm  <- mean(exp(resid(ll)))                                                    # smearing 보정 계수
gp  <- glm(cost ~ A + age + male + cci + stage4, data = d, family = quasipoisson(link = "log"))
gi  <- glm(cost ~ A * (age + male + cci + stage4), data = d, family = Gamma(link = "log"))
lg  <- c(mean(exp(predict(ll, d1))), mean(exp(predict(ll, d0))))
tab <- rbind(linear = c(mean(predict(lm1, d1)), mean(predict(lm1, d0))),
             log_naive = lg, log_smearing = sm * lg, gamma = std(g),
             poisson = std(gp), gamma_interaction = std(gi))
round(cbind(tab, diff = tab[, 1] - tab[, 2]), 1)
round(coefci(lm1, vcov. = vcovHC(lm1, type = "HC3"))["A", ], 1)             # 선형회귀의 HC3 구간
''', title="같은 자료, 여러 모형 (21장 c 절 표 21-7)")

c_check = nb.cell('''
chk <- data.frame(cost = d$cost, pred = fitted(g)); chk$ln <- log(chk$pred)
pk <- glm(I((cost - pred)^2) ~ ln, data = chk, family = Gamma(link = "log"))   # modified Park test
round(c(slope = coef(pk)[["ln"]], coefci(pk, vcov. = vcovHC(pk, type = "HC0"), df = Inf)["ln", ]), 2)
chk$tenth <- cut(rank(chk$pred, ties.method = "first"), 10, labels = FALSE)
round(aggregate(cbind(cost, pred) ~ tenth, data = chk, mean))               # 구간별 관측·예측 평균
lk <- glm(cost ~ ln + I(ln^2), data = chk, family = Gamma(link = "log"))     # 연결 검정
round(coeftest(lk, vcov. = vcovHC(lk, type = "HC0"), df = Inf)["I(ln^2)", ], 3)
''', title="모형 점검: Park test, 구간별 평균, 연결 검정 (21장 c 절)")

# ------------------------------------------------------------------ d. 0원이 많은 입원비
c_twopart = nb.cell('''
d$anyinp <- as.integer(d$cost_inp > 0); round(mean(d$anyinp == 0), 3)
rhs <- ~ A + age + male + cci + stage4
p1 <- glm(update(rhs, anyinp ~ .), data = d, family = binomial)                        # 1부: 입원비가 생겼나
p2 <- glm(update(rhs, cost_inp ~ .), data = d[d$cost_inp > 0, ], family = Gamma(link = "log"))   # 2부: 얼마인가
round(exp(c(odds_ratio = coef(p1)[["A"]], cost_ratio = coef(p2)[["A"]])), 3)
tp <- function(q1, q2, nd) mean(predict(q1, nd, type = "response") * predict(q2, nd, type = "response"))
e <- c(all_A = tp(p1, p2, d1), all_B = tp(p1, p2, d0))       # 환자마다 곱한 뒤 평균
round(c(e, diff = e[["all_A"]] - e[["all_B"]]), 1)
''', title="이단계 모형 (21장 c 절)")

c_tpboot = nb.cell('''
set.seed(21)
tb <- boot(d, function(x, i) {
  y  <- x[i, ]
  q1 <- glm(update(rhs, anyinp ~ .), data = y, family = binomial)
  q2 <- glm(update(rhs, cost_inp ~ .), data = y[y$cost_inp > 0, ], family = Gamma(link = "log"))
  tp(q1, q2, transform(y, A = 1)) - tp(q1, q2, transform(y, A = 0))
}, R = 1000)
round(quantile(tb$t, c(0.025, 0.975)), 1)
''', title="이단계 모형의 부트스트랩 구간 (21장 c 절)")

c_zero = nb.cell('''
r <- try(glm(update(rhs, cost_inp ~ .), data = d, family = Gamma(link = "log")), silent = TRUE)
cat(r)                                                          # 0원이 있으면 Gamma는 멈춘다
gq <- glm(update(rhs, cost_inp ~ .), data = d,
          family = quasi(link = "log", variance = "mu^2"),      # 감마와 같은 평균·분산 구조
          mustart = rep(mean(d$cost_inp), nrow(d)))             # 시작값을 주지 않으면 수렴하지 않음
round(c(ratio = exp(coef(gq)[["A"]]), diff = std(gq)[["all_A"]] - std(gq)[["all_B"]]), 3)
li <- lm(update(rhs, cost_inp ~ .), data = d)                   # 원래 입원비의 선형회귀
round(c(diff = coef(li)[["A"]], coefci(li, vcov. = vcovHC(li, type = "HC3"))["A", ]), 1)
r <- try(lm(update(rhs, log(cost_inp) ~ .), data = d), silent = TRUE)
cat(r)                                                          # log(0) = -Inf
''', title="0원을 넣은 채 적합하면 (21장 c 절)")

# ------------------------------------------------------------------ e. 과제
c_t1 = nb.cell('''
items <- c("cost_drug", "cost_op", "cost_inp", "cost")
t(sapply(items, function(v) {
  tt <- t.test(d[[v]][d$A == 1], d[[v]][d$A == 0])
  round(c(mean_A = mean(d[[v]][d$A == 1]), mean_B = mean(d[[v]][d$A == 0]),
          diff = unname(tt$estimate[1] - tt$estimate[2]), tt$conf.int), 1)
}))
''', title="과제 1 정답. 항목별 평균 차이")

c_t2 = nb.cell('''
do.call(rbind, lapply(c(0.01, 1, 100), function(k) {
  f <- lm(log(cost_inp + k) ~ A + age + male + cci + stage4, data = d)   # 0원에 k를 더해 로그
  data.frame(add = k, ratio = round(exp(coef(f)[["A"]]), 3), p = round(summary(f)$coefficients["A", 4], 3))
}))
''', title="과제 2 정답. 입원비에 1을 더해 로그를 취한 분석")

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
    c_claims: ['[1] 9', '137.3', '[1] 412'],
    c_load: ['3000 obs. of  11 variables', '21727.1 16462.9', '0.436 0.357'],
    c_ttest: ['296.1    496.1', '293.8    498.4'],
    c_boot: ['[1] 51.6', '296.5 496.3'],
    c_skew: ['625.9', '0.678', '1.244 1.202 1.288'],
    c_glm: [' 1.241  1.188  1.296', '[1] 0.318', ' 1.192  1.292'],
    c_std: ['2223.8 536.0', '547.1', '536.0 424.8 647.1'],
    c_stdboot: ['437.5 636.5'],
    c_default: ['-8.552332e-05', '512.5732'],
    c_models: ['643.2', '506.1', ' 412.7  612.5'],
    c_check: ['2.02   0.64   3.41', '10    10 2878 3019', '0.077'],
    c_twopart: ['0.935      0.923', '645.1 715.9 -70.7'],
    c_tpboot: ['-162.0   16.4'],
    c_zero: ['non-positive values not allowed', '0.907 -66.232', '-72.0 -161.8   17.7', "NA/NaN/Inf in 'y'"],
    c_t1: ['-175.0 -263.9  -86.1'],
    c_t2: ['0.829', '0.904'],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc23_claims", nb.html(c_claims, marks={'[1] 9': 1, '137.3': 2, '[1] 412': 3}))
S("rc23_load", nb.html(c_load, marks={'3000 obs. of  11 variables': 1, '21727.1 16462.9': 2, '0.436 0.357': 3}))
S("rc23_ttest", nb.html(c_ttest, marks={'296.1    496.1': 1, '293.8    498.4': 2}))
S("rc23_boot", nb.html(c_boot, marks={'[1] 51.6': 1, '296.5 496.3': 2}))
S("rc23_skew", nb.html(c_skew, marks={'625.9': 1, '0.678': 2, '1.244 1.202 1.288': 3}))
S("rc23_glm", nb.html(c_glm, marks={' 1.241  1.188  1.296': 1, '[1] 0.318': 2, ' 1.192  1.292': 3}))
S("rc23_std", nb.html(c_std, marks={'2223.8 536.0': 1, '547.1': 2, '536.0 424.8 647.1': 3}))
S("rc23_stdboot", nb.html(c_stdboot, marks={'437.5 636.5': 1}))
S("rc23_default", nb.html(c_default, marks={'-8.552332e-05': 1, '512.5732': 2}))
S("rc23_models", nb.html(c_models, marks={'643.2': 1, '506.1': 2, ' 412.7  612.5': 3}))
S("rc23_check", nb.html(c_check, marks={'2.02   0.64   3.41': 1, '10    10 2878 3019': 2, '0.077': 3}))
S("rc23_twopart", nb.html(c_twopart, marks={'0.935      0.923': 1, '645.1 715.9 -70.7': 2}))
S("rc23_tpboot", nb.html(c_tpboot, marks={'-162.0   16.4': 1}))
S("rc23_zero", nb.html(c_zero, marks={'non-positive values not allowed': 1, '0.907 -66.232': 2, '-72.0 -161.8   17.7': 3, "NA/NaN/Inf in 'y'": 4}))
S("rc23_t1", nb.html(c_t1))
S("rc23_t2", nb.html(c_t2))
nb.save_ipynb("20장 비용 자료 분석 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 21장의 가상 1년 의료비 자료(신세포암 1차 치료 3,000명)와 명세서 11건으로 환자별 비용 만들기, 평균 차이의 t 검정과 부트스트랩, "
                    "감마 GLM과 표준화, 여러 모형의 비교와 점검, 0원이 많은 입원비의 이단계 모형을 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"sandwich\", \"lmtest\", \"boot\"))를 실행합니다.",
              notes={c_claims.n: "## a. 명세서에서 환자별 비용", c_load.n: "## b. 비용 분포와 평균 비교", c_glm.n: "## c. 비용의 회귀분석",
                     c_twopart.n: "## d. 0원이 많은 입원비", c_t1.n: "## e. 과제 정답"})
print("rc23 fragments and notebook written")
