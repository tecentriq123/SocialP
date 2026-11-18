"""R과 파이썬으로 하는 통계분석 5장 · 로지스틱 회귀와 포아송 회귀 — R cells run for real with rkit.

run:  python3 gen/r_rc14.py      (NOMARK=1 python3 gen/r_rc14.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 8장(효과 지표, 일반화 선형모형, 우도, 교란, 가변수), 9장(로지스틱 회귀), 12장(발생률과 포아송 회귀)의
'직접 분석' 상자에 있던 파이썬 코드를 2026-11-18 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 8장의 퇴원 환자 코호트 1,500명(pub/data/ch08_readmit.csv), 6장 d 절의 의료비 400명(ch06_cost400.csv),
9장의 SGLT2 억제제 코호트 2,400명(ch09_cohort.csv), 반코마이신 150명(ch09_vanco.csv), 12장의 COPD 코호트 2,400명
(ch12_copd.csv). CSV는 gen/data_rc14.py가 gen/lib_ch08.py·lib_ch09.py·lib_ch12.py에서 만든다(같은 난수).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc14")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 비율의 구간과 세 검정 (8장 a·e 절)
c_prop = nb.cell('''
x <- 7; n <- 50                                     # 이상반응 7명 / 50명
p <- x / n
p + c(-1, 1) * qnorm(0.975) * sqrt(p * (1 - p) / n)   # Wald 구간 (식으로 직접)
prop.test(x, n, correct = FALSE)$conf.int           # Wilson 구간
binom.test(x, n)$conf.int                           # 정확(Clopper–Pearson) 구간
prop.test(x, n)$conf.int                            # 기본값: 연속성 보정을 한 Wilson 구간
''', title="50명 중 7명의 세 가지 95% 신뢰구간")

c_three = nb.cell('''
three <- function(x, n, p0) {                       # H0: p = p0 의 세 검정과 정확검정
  ph <- x / n
  wald  <- (ph - p0)^2 / (ph * (1 - ph) / n)        # 추정값에서의 곡률
  score <- (ph - p0)^2 / (p0 * (1 - p0) / n)        # 귀무가설 값에서의 기울기와 곡률
  lr    <- 2 * (dbinom(x, n, ph, log = TRUE) - dbinom(x, n, p0, log = TRUE))
  chi <- c(Wald = wald, LR = lr, score = score)
  rbind(chi2 = chi, P = pchisq(chi, 1, lower.tail = FALSE)) |>
    cbind(exact = c(NA, binom.test(x, n, p0)$p.value)) |> round(4)
}
three(7, 50, p0 = 0.05)
three(7, 50, p0 = 0.10)
''', title="같은 자료의 Wald, 우도비, 점수 검정")

# ------------------------------------------------------------------ b. 로지스틱 회귀의 계수와 오즈비 (8장 c·g 절)
c_fit8 = nb.cell(f'''
dc <- read.csv("{D}ch08_readmit.csv")
dc$cci <- factor(dc$cci, levels = c("0", "1-2", "3-4", "5+"))   # 범주 순서를 정함: 첫 수준이 기준범주
m <- glm(readm ~ poly + age + female + cci + medaid, family = binomial, data = dc)
summary(m)
''', title="퇴원 환자 1,500명의 로지스틱 회귀")

c_or8 = nb.cell('''
round(exp(cbind(OR = coef(m), confint.default(m))), 3)   # Wald 구간: β ± 1.96 × SE를 지수로
exp(confint(m, "poly"))                                  # 프로파일 우도 구간
''', title="오즈비와 두 가지 95% 신뢰구간")

c_joint8 = nb.cell('''
suppressPackageStartupMessages(library(car))
suppressWarnings(Anova(m, type = 3, test.statistic = "Wald"))   # 변수마다 Wald 결합 검정 (수준 이름의 "-" 때문에 나는 표시 경고는 숨김)
drop1(m, test = "LRT")                              # 변수마다 우도비 검정
''', title="CCI 전체의 P: Wald 결합 검정과 우도비 검정")

# ------------------------------------------------------------------ c. 우도비 검정과 AIC (8장 e 절)
c_ll8 = nb.cell('''
logLik(m)                                           # 로그우도와 추정한 계수 수(df)
c(m2LL = -2 * as.numeric(logLik(m)), deviance = deviance(m), null = m$null.deviance)
c(AIC = AIC(m), BIC = BIC(m))
m0 <- update(m, . ~ . - poly)                       # 다제약물만 뺀 모형
anova(m0, m, test = "LRT")                          # 우도비 검정
anova(m0, m, test = "Rao")                          # 같은 비교의 점수 검정
c(AIC0 = AIC(m0), BIC0 = BIC(m0))
c(iter = m$iter, converged = m$converged)           # 반복 횟수와 수렴
''', title="로그우도, 우도비 검정, AIC와 BIC")

c_nobs8 = nb.cell('''
dm <- dc; dm$age[1:5] <- NA                         # 다섯 명의 나이가 비었다고 하면
nobs(glm(readm ~ poly + age + female + cci + medaid, family = binomial, data = dm))
nobs(glm(readm ~ poly + female + cci + medaid, family = binomial, data = dm))
dd <- na.omit(dm[, c("readm", "poly", "age", "female", "cci", "medaid")])   # 큰 모형의 변수가 모두 있는 행만
f1 <- glm(readm ~ poly + age + female + cci + medaid, family = binomial, data = dd)
f0 <- update(f1, . ~ . - age)
c(nobs(f0), nobs(f1))                               # 두 값이 같아야 비교할 수 있다
''', title="결측이 있으면 분석 인원이 달라진다")

# ------------------------------------------------------------------ d. 상대위험도, 위험차, 의료비 (8장 d 절)
c_logbin = nb.cell('''
f <- readm ~ poly + age + female + cci + medaid
tryCatch(glm(f, family = binomial(link = "log"), data = dc),
         error = function(e) conditionMessage(e))   # 기본 시작값으로는 멈춘다
pois <- glm(f, family = poisson, data = dc)
lb <- glm(f, family = binomial(link = "log"), data = dc, start = coef(pois))   # 포아송 계수를 시작값으로
round(exp(c(RR = coef(lb)[["poly"]], confint.default(lb)["poly", ])), 2)
c(logLik = as.numeric(logLik(lb)), max_risk = max(fitted(lb)), converged = lb$converged)
''', title="로그-이항 모형: 시작값을 주어 상대위험도 구하기")

c_robust = nb.cell('''
suppressPackageStartupMessages({library(sandwich); library(lmtest)})
coeftest(pois, vcov = vcovHC(pois, type = "HC0"))["poly", ]          # 계수는 그대로, SE만 강건하게
round(exp(c(RR = coef(pois)[["poly"]],
            coefci(pois, vcov = vcovHC(pois, type = "HC0"))["poly", ])), 2)
''', title="수정 포아송 회귀: 포아송 회귀 + 강건 표준오차")

c_rd = nb.cell('''
tryCatch(glm(f, family = binomial(link = "identity"), data = dc),
         error = function(e) conditionMessage(e))   # 이항분포 + 항등 연결도 멈춘다
lpm <- lm(f, data = dc)                             # 선형 확률 모형
round(100 * c(RD = coef(lpm)[["poly"]],
              coefci(lpm, vcov = vcovHC(lpm, type = "HC0"))["poly", ]), 1)   # %p
sum(fitted(lpm) < 0)                                # 예측 확률이 음수인 환자 수
r1 <- mean(predict(m, transform(dc, poly = 1), type = "response"))   # 모두 다제약물로 놓고 예측
r0 <- mean(predict(m, transform(dc, poly = 0), type = "response"))   # 모두 아닌 것으로 놓고 예측
round(100 * c(r1 = r1, r0 = r0, RD = r1 - r0), 1)
''', title="보정한 위험차: 선형 확률 모형과 표준화")

c_rdboot = nb.cell('''
suppressPackageStartupMessages(library(boot))
std_rd <- function(d, i) {
  fit <- glm(f, family = binomial, data = d[i, ])
  mean(predict(fit, transform(d[i, ], poly = 1), type = "response")) -
    mean(predict(fit, transform(d[i, ], poly = 0), type = "response"))
}
set.seed(2026)
b <- boot(dc, std_rd, R = 1000)                     # 환자를 복원추출해 1,000번 다시 적합
round(100 * boot.ci(b, type = "perc")$percent[4:5], 1)   # 백분위수 95% 구간 (%p)
''', title="표준화 위험차의 부트스트랩 구간")

c_offset = nb.cell('''
er <- data.frame(group = factor(c("노출", "비교"), levels = c("비교", "노출")),   # 비교군이 기준
                 visits = c(120, 90), pyears = c(400, 450))
exp(coef(glm(visits ~ group, family = poisson, data = er)))[2]                       # 건수의 비
exp(coef(glm(visits ~ group + offset(log(pyears)), family = poisson, data = er)))[2] # 발생률비
''', title="추적기간이 다를 때의 offset")

c_gamma = nb.cell(f'''
co <- read.csv("{D}ch06_cost400.csv")              # 제2형 당뇨병 400명의 연간 의료비(만원)
exp(coef(glm(cost ~ adherent, family = Gamma(link = "log"), data = co)))[2]   # 보정 전: 산술평균의 비
g <- glm(cost ~ adherent + I(age / 10) + cci, family = Gamma(link = "log"), data = co)
round(exp(c(ratio = coef(g)[["adherent"]], confint.default(g)["adherent", ])), 3)
c1 <- mean(predict(g, transform(co, adherent = 1), type = "response"))   # 재활용 예측
c0 <- mean(predict(g, transform(co, adherent = 0), type = "response"))
round(c(adherent = c1, nonadherent = c0, diff = c1 - c0), 1)
''', title="의료비의 감마 GLM과 재활용 예측")

c_gboot = nb.cell('''
set.seed(2026)
bg <- boot(co, function(d, i) {
  fit <- glm(cost ~ adherent + I(age / 10) + cci, family = Gamma(link = "log"), data = d[i, ])
  mean(predict(fit, transform(d[i, ], adherent = 1), type = "response")) -
    mean(predict(fit, transform(d[i, ], adherent = 0), type = "response"))
}, R = 1000)
round(boot.ci(bg, type = "perc")$percent[4:5], 1)   # 1인당 금액 차이의 95% 구간 (만원)
''', title="금액 차이의 부트스트랩 구간")

# ------------------------------------------------------------------ e. 층화와 교호작용 (8장 f 절)
c_mh8 = nb.cell('''
tab <- with(dc, table(poly, readm, cci))[2:1, 2:1, ]   # 층마다 [노출 1행, 사건 1열]이 되게 순서를 뒤집음
round(apply(tab, 3, function(t) t[1, 1] * t[2, 2] / (t[1, 2] * t[2, 1])), 2)   # 층별 OR
mantelhaen.test(tab, correct = FALSE)               # MH 오즈비, 구간, CMH 검정
suppressPackageStartupMessages(library(DescTools))
BreslowDayTest(tab)$p.value                         # 층별 OR이 모두 같은가
WoolfTest(tab)$p.value
''', title="CCI 네 층의 Mantel–Haenszel 오즈비와 동질성 검정")

c_new8 = nb.cell('''
nd <- array(c(288, 120, 612, 180,                  # 중증: 새 약 288/900, 기존 약 120/300
              24, 90, 276, 810),                   # 경증: 새 약 24/300,  기존 약 90/900
            dim = c(2, 2, 2), dimnames = list(drug = c("새 약", "기존 약"),
                                              hosp = c("입원", "없음"), stratum = c("중증", "경증")))
mantelhaen.test(nd, correct = FALSE)$statistic      # CMH χ² (연속성 보정 없음)
mantelhaen.test(nd)$statistic                       # 기본값: 연속성 보정
n_k <- apply(nd, 3, sum)                            # MH 상대위험도 (식으로 직접)
sum(nd[1, 1, ] * colSums(nd[2, , ]) / n_k) / sum(nd[2, 1, ] * colSums(nd[1, , ]) / n_k)
''', title="새 약 자료: CMH 검정과 MH 상대위험도")

c_int8 = nb.cell('''
mi <- glm(readm ~ poly * cci + age + female + medaid, family = binomial, data = dc)
anova(m, mi, test = "LRT")                          # 곱셈항 3개의 우도비 검정 = P for interaction
''', title="다제약물 × CCI 교호작용의 우도비 검정")

# ------------------------------------------------------------------ f. 보정 모형과 결과 읽기 (9장)
c_fit9 = nb.cell(f'''
ch <- read.csv("{D}ch09_cohort.csv")
ch$cci  <- factor(ch$cci, levels = c("0", "1-2", ">=3"))
ch$egfr_c <- factor(cut(ch$egfr, c(-Inf, 45, 60, Inf), right = FALSE, labels = c("<45", "45-59", ">=60")),
                    levels = c(">=60", "45-59", "<45"))      # 기준범주 ≥60
ch$age10 <- (ch$age - 75) / 10
m9 <- glm(hosp ~ sglt2 + age10 + female + cci + prior + hf + egfr_c, family = binomial, data = ch)
summary(m9)
''', title="SGLT2 억제제 코호트 2,400명의 완전 보정 모형")

c_or9 = nb.cell('''
round(exp(cbind(OR = coef(m9), confint.default(m9)))[-1, ], 3)   # 보정 오즈비와 Wald 구간
round(exp(confint(m9, c("sglt2", "hf"))), 3)                    # 프로파일 우도 구간
plogis(coef(m9)[["(Intercept)"]])                                # 기준 환자의 입원 확률
''', title="보정 오즈비 표")

c_lr9 = nb.cell('''
red <- update(m9, . ~ . - cci)
anova(red, m9, test = "LRT")                        # CCI 전체 (자유도 2)
with(m9, c(LR = null.deviance - deviance, df = df.null - df.residual))   # 모형 전체
mt <- update(m9, . ~ . - cci + as.numeric(cci))     # 0, 1, 2 점수 하나로
exp(coef(mt)[["as.numeric(cci)"]])                  # 한 단계당 보정 오즈비 (P for trend는 이 행의 P)
''', title="CCI 전체의 우도비 검정과 P for trend")

c_int9 = nb.cell('''
mi9 <- update(m9, . ~ . + sglt2:hf)
anova(m9, mi9, test = "LRT")                        # P for interaction
round(coef(summary(mi9))["sglt2:hf", ], 4)
round(exp(c(coef(mi9)[["sglt2:hf"]], confint.default(mi9)["sglt2:hf", ])), 2)   # 두 오즈비의 비
round(exp(c(no_hf = coef(mi9)[["sglt2"]],                          # 심부전 없는 환자의 SGLT2 OR
            hf    = coef(mi9)[["sglt2"]] + coef(mi9)[["sglt2:hf"]])), 2)   # 심부전 환자의 SGLT2 OR
''', title="SGLT2 × 심부전 교호작용")

c_std9 = nb.cell('''
r1 <- mean(predict(m9, transform(ch, sglt2 = 1), type = "response"))
r0 <- mean(predict(m9, transform(ch, sglt2 = 0), type = "response"))
round(c(r1 = r1, r0 = r0, RD = r1 - r0, RR = r1 / r0), 3)
set.seed(2026)
b9 <- boot(ch, function(d, i) {
  fit <- update(m9, data = d[i, ])
  a <- mean(predict(fit, transform(d[i, ], sglt2 = 1), type = "response"))
  z <- mean(predict(fit, transform(d[i, ], sglt2 = 0), type = "response"))
  c(a - z, a / z)
}, R = 1000)
round(rbind(RD = boot.ci(b9, type = "perc", index = 1)$percent[4:5],
            RR = boot.ci(b9, type = "perc", index = 2)$percent[4:5]), 3)
''', title="표준화 위험, 위험차, 위험의 비")

c_rr9 = nb.cell('''
mp9 <- glm(hosp ~ sglt2 + age10 + female + cci + prior + hf + egfr_c, family = poisson, data = ch)
round(exp(cbind(RR = coef(mp9), coefci(mp9, vcov = vcovHC(mp9, type = "HC0"))))[c("prior", "sglt2"), ], 2)
round(c(model_SE = sqrt(vcov(mp9)["prior", "prior"]),
        robust_SE = sqrt(vcovHC(mp9, type = "HC0")["prior", "prior"])), 3)
tryCatch(glm(formula(m9), family = binomial(link = "log"), data = ch, start = coef(mp9)),
         error = function(e) conditionMessage(e))   # 이 자료에서는 로그-이항 모형이 시작값을 주어도 멈춘다
''', title="수정 포아송 회귀의 상대위험도")

c_cl9 = nb.cell('''
suppressPackageStartupMessages(library(survival))
pr <- data.frame(pair = rep(1:250, each = 2), case = rep(c(1, 0), 250),          # 짝마다 환자 1행, 대조군 1행
                 thiazide = c(rep(c(1, 1), 30), rep(c(1, 0), 50), rep(c(0, 1), 25), rep(c(0, 0), 145)))
fc <- clogit(case ~ thiazide + strata(pair), data = pr)                            # 조건부 로지스틱 회귀
round(exp(cbind(OR = coef(fc), confint(fc))), 2)
round(exp(coef(glm(case ~ thiazide, family = binomial, data = pr)))[["thiazide"]], 2)                 # 짝 무시
round(exp(coef(glm(case ~ thiazide + factor(pair), family = binomial, data = pr)))[["thiazide"]], 2)  # 짝마다 가변수
''', title="짝지은 환자-대조군 자료의 조건부 로지스틱 회귀")

# ------------------------------------------------------------------ g. 로지스틱 모형 점검 (9장)
c_spl9 = nb.cell('''
suppressPackageStartupMessages(library(rms))        # rcs(): Harrell의 제한 삼차 스플라인
base <- hosp ~ sglt2 + age10 + female + cci + prior + hf
ml <- glm(update(base, . ~ . + egfr), family = binomial, data = ch)          # 직선
ms <- glm(update(base, . ~ . + rcs(egfr, 4)), family = binomial, data = ch)  # 매듭 4개
anova(ml, ms, test = "LRT")                         # 비선형 항 2개의 우도비 검정
round(c(linear = AIC(ml), categories = AIC(m9), spline = AIC(ms)), 1)
attr(rcs(ch$egfr, 4), "parms")                      # 매듭 위치: 5, 35, 65, 95 백분위수
nd9 <- ch[rep(1, 5), ]; nd9$egfr <- c(90, 75, 60, 45, 30)
lp <- predict(ms, nd9); round(exp(lp - lp[1]), 2)   # eGFR 90 대비 오즈비
''', title="eGFR이 로짓과 직선 관계인가")

c_wald9 = nb.cell(f'''
va <- read.csv("{D}ch09_vanco.csv")                # 반코마이신 trough와 급성 신손상 150명
gv <- glm(aki ~ I(trough / 5), family = binomial, data = va)   # 5 mg/L당
round(exp(cbind(OR = coef(gv), Wald = confint.default(gv), profile = confint(gv)))[2, ], 2)
round(c(Wald = coef(summary(gv))[2, "z value"]^2,
        LR = anova(gv, test = "Rao")[2, "Deviance"], score = anova(gv, test = "Rao")[2, "Rao"]), 1)
''', title="Wald 구간과 프로파일 구간이 갈리는 자료")

c_sep9 = nb.cell('''
ici <- data.frame(autoimmune  = rep(c(1, 1, 0, 0), c(4, 2, 30, 44)),
                  combination = rep(c(1, 0, 1, 0), c(4, 2, 30, 44)),
                  irae = c(rep(1, 6), rep(c(1, 0), c(9, 21)), rep(c(1, 0), c(6, 38))))
with(ici, table(autoimmune, irae))                  # 자가면역질환 병력 6명 모두 사건
gs <- glm(irae ~ autoimmune + combination, family = binomial, data = ici)
round(coef(summary(gs)), 3)
c(iter = gs$iter, converged = gs$converged)
anova(update(gs, . ~ . - autoimmune), gs, test = "LRT")   # 우도비 검정은 작동한다
''', title="분리가 생긴 모형")

c_firth9 = nb.cell('''
suppressPackageStartupMessages(library(logistf))
ff <- logistf(irae ~ autoimmune + combination, data = ici)   # 기본: 프로파일 벌점 우도 구간과 검정
round(cbind(coef = coef(ff), OR = exp(coef(ff)), lower = exp(ff$ci.lower), upper = exp(ff$ci.upper),
            chisq = qchisq(1 - ff$prob, 1), p = ff$prob), 4)
''', title="Firth 방법")

c_vif9 = nb.cell('''
car::vif(m9)                                        # 범주형 변수는 묶음 단위의 GVIF
X <- model.matrix(m9)[, -1]                         # 가변수 하나하나의 VIF
round(sapply(colnames(X), function(j) 1 / (1 - summary(lm(X[, j] ~ X[, -match(j, colnames(X))]))$r.squared)), 2)
round(rms::vif(m9), 2)                              # 모형의 공분산 행렬로 계산한 값
summary(cooks.distance(m9))                         # 환자 한 명을 뺐을 때 전체 적합의 변화
max(abs(dfbeta(m9)[, "sglt2"]))                     # SGLT2 계수가 가장 크게 바뀌는 정도
''', title="VIF와 영향력이 큰 환자")

c_code9 = nb.cell('''
ch$hosp_s <- ifelse(ch$hosp == 1, "yes", "no")
coef(glm(factor(hosp_s) ~ sglt2, family = binomial, data = ch))   # 요인: 첫 수준(no)이 0, 나머지가 사건
coef(glm(hosp == 1 ~ sglt2, family = binomial, data = ch))        # 논리값: TRUE가 사건
tryCatch(glm(I(hosp + 1) ~ sglt2, family = binomial, data = ch),
         error = function(e) conditionMessage(e))                 # 1과 2로 넣으면 멈춘다
''', title="결과를 어떻게 부호화하나")

# ------------------------------------------------------------------ h. 발생률과 포아송 회귀 (12장)
c_rate12 = nb.cell('''
poisson.test(110, 11067)$conf.int * 1000            # SGLT2 억제제: 1,000인년당 정확 95% CI
poisson.test(447, 28002)$conf.int * 1000            # DPP-4 억제제
poisson.test(c(110, 447), c(11067, 28002))          # 발생률비, 조건부 정확검정과 정확 구간
r1 <- 110 / 11067; r0 <- 447 / 28002
exp(log(r1 / r0) + c(-1, 1) * 1.96 * sqrt(1 / 110 + 1 / 447))                 # 로그 척도 Wald 구간
1000 * (r1 - r0 + c(-1, 1) * 1.96 * sqrt(110 / 11067^2 + 447 / 28002^2))   # 발생률차의 Wald 구간
''', title="두 군의 발생률과 발생률비")

c_small12 = nb.cell('''
d <- 3; t <- 1500                                   # 혈관부종 3건 / 1,500인년
round(1000 * rbind(Wald  = d / t + c(-1, 1) * 1.96 * sqrt(d) / t,
                   log   = exp(log(d / t) + c(-1, 1) * 1.96 / sqrt(d)),
                   exact = poisson.test(d, t)$conf.int), 2)
''', title="사건이 적을 때 발생률의 세 구간")

c_pois12 = nb.cell(f'''
copd <- read.csv("{D}ch12_copd.csv")
copd$drug <- factor(copd$drug, levels = c("B", "A"))   # 약 B가 기준
fit <- glm(exac ~ drug + agegrp + female + cci + offset(log(py)), family = poisson, data = copd)
summary(fit)
round(exp(cbind(IRR = coef(fit), confint.default(fit))), 3)
''', title="COPD 코호트의 포아송 회귀")

c_zero12 = nb.cell('''
z <- data.frame(drug = factor(c("신약 X", "비교약"), levels = c("비교약", "신약 X")),
                events = c(3, 0), pyears = c(1500, 2300))
gz <- glm(events ~ drug + offset(log(pyears)), family = poisson, data = z)
round(coef(summary(gz)), 3)                         # 계수와 표준오차가 끝없이 커진 자리에서 멈춤
c(iter = gz$iter, converged = gz$converged)
poisson.test(c(3, 0), c(1500, 2300))$p.value        # 조건부 정확검정
''', title="한 군의 사건이 0건이면")

c_claims12 = nb.cell('''
cl <- data.frame(id = 1,
                 start = as.Date(c("2023-05-02", "2023-05-05", "2023-05-06", "2023-06-23")),
                 end   = as.Date(c("2023-05-02", "2023-05-05", "2023-05-12", "2023-06-23")))
cl <- cl[order(cl$id, cl$start), ]
prev_end <- ave(as.numeric(cl$end), cl$id,           # 같은 환자의 앞선 청구가 끝난 가장 늦은 날
                FUN = function(e) c(NA, head(cummax(e), -1)))
cl$new_ep <- is.na(prev_end) | (as.numeric(cl$start) - prev_end > 14)   # 14일 넘게 떨어지면 새 악화
cl
sum(cl$new_ep)                                      # 이 환자의 악화 건수 (환자가 여럿이면 tapply(cl$new_ep, cl$id, sum))
as.numeric(as.Date("2023-11-30") - as.Date("2023-03-01")) / 365.25   # 인년
''', title="청구 기록에서 악화 건수와 인년 만들기")

# ------------------------------------------------------------------ i. 과산포와 음이항 회귀 (12장 b 절)
c_disp12 = nb.cell('''
round(c(pearson = sum(residuals(fit, type = "pearson")^2), deviance = deviance(fit),
        df = df.residual(fit)), 1)
round(c(pearson_df = sum(residuals(fit, type = "pearson")^2) / df.residual(fit),
        deviance_df = deviance(fit) / df.residual(fit)), 2)
round(c(observed_0 = mean(copd$exac == 0), predicted_0 = mean(dpois(0, fitted(fit)))), 3)
''', title="과산포 확인")

c_three12 = nb.cell('''
qp <- glm(exac ~ drug + agegrp + female + cci + offset(log(py)), family = quasipoisson, data = copd)
suppressPackageStartupMessages(library(MASS))
nb2 <- glm.nb(exac ~ drug + agegrp + female + cci + offset(log(py)), data = copd)
round(c(poisson = sqrt(vcov(fit)["drugA", "drugA"]),
        quasi   = sqrt(vcov(qp)["drugA", "drugA"]),
        robust  = sqrt(vcovHC(fit, type = "HC0")["drugA", "drugA"]),
        negbin  = sqrt(vcov(nb2)["drugA", "drugA"])), 4)   # 약 A 계수의 표준오차
summary(qp)$dispersion                              # 준포아송의 φ
''', title="세 방법의 표준오차")

c_nb12 = nb.cell('''
summary(nb2)
round(exp(c(IRR = coef(nb2)[["drugA"]], confint.default(nb2)["drugA", ])), 3)
alpha <- 1 / nb2$theta; se_a <- nb2$SE.theta / nb2$theta^2      # α = 1/θ와 델타 방법 SE
round(c(theta = nb2$theta, alpha = alpha, lower = alpha - 1.96 * se_a, upper = alpha + 1.96 * se_a), 3)
lr <- 2 * (as.numeric(logLik(nb2)) - as.numeric(logLik(fit)))
c(LR = lr, P = pchisq(lr, 1, lower.tail = FALSE) / 2)          # α = 0이 경계라 P를 반으로
c(AIC_poisson = AIC(fit), AIC_negbin = AIC(nb2))
''', title="음이항 회귀")

# ------------------------------------------------------------------ j. 과제
c_t1 = nb.cell('''
pchisq(842.6 - 836.1, df = 1, lower.tail = FALSE)   # 약물 노출을 더한 단계
pchisq(836.1 - 829.0, df = 3, lower.tail = FALSE)   # 가변수 3개를 더한 단계
''', title="과제 1 정답")

c_t2 = nb.cell('''
m34 <- update(m, data = transform(dc, cci = relevel(cci, ref = "3-4")))   # 기준범주를 3–4점으로
round(exp(c(OR = coef(m34)[["cci5+"]], confint.default(m34)["cci5+", ])), 2)
round(coef(summary(m34))["cci5+", "Pr(>|z|)"], 3)
''', title="과제 2 정답")

c_t3 = nb.cell('''
young <- poisson.test(c(58, 162), c(8076, 16238))   # 65세 미만
old   <- poisson.test(c(52, 285), c(2991, 11764))   # 65세 이상
wl <- function(d1, t1, d0, t0) exp(log((d1 / t1) / (d0 / t0)) + c(0, -1, 1) * 1.96 * sqrt(1 / d1 + 1 / d0))
round(rbind(young_exact = c(young$estimate, young$conf.int), young_wald = wl(58, 8076, 162, 16238),
            old_exact = c(old$estimate, old$conf.int), old_wald = wl(52, 2991, 285, 11764)), 2)
''', title="과제 3 정답")

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

out = {c.n: c.text for c in nb.cells}
A = lambda c, *xs: [None for x in xs if (x in out[c.n]) or (_ for _ in ()).throw(AssertionError((c.title, x)))]
A(c_prop, "0.04382187 0.23617813", "0.06950833 0.26186194", "0.0581917 0.2673960")
A(c_three, "3.3638 5.8551 8.5263", "0.0666 0.0155 0.0035 0.0118", "0.6645 0.8008 0.8889", "0.4150 0.3708 0.3458 0.3415")
A(c_fit8, "0.38311    0.14850   2.580 0.009884", "Null deviance: 1383.4", "Residual deviance: 1306.9")
A(c_or8, "poly        1.467 1.096  1.962", "female      0.779 0.591  1.029", "cci3-4      1.881 1.227  2.884",
  "cci5+       2.609 1.519  4.481", "1.096124 1.962852")
A(c_joint8, "cci          3 14.5734   0.002220", "cci     3   1321.6 1331.6 14.6161  0.002176")
A(c_ll8, "-653.4729 (df=8)", "1306.946 1306.946 1383.397", "1322.946 1365.452", "1   6.6349     0.01", "1327.581 1364.773")
A(c_nobs8, "[1] 1495\n[1] 1500", "[1] 1495 1495")
A(c_logbin, "1.35   1.07   1.71", "-654.134248    0.614409")
A(c_robust, "1.36   1.08   1.72")
A(c_rd, "5.3    1.1    9.6", "[1] 11", "20.2 14.9  5.3")
A(c_offset, "1.333333", "      1.5")
A(c_gamma, "0.7676587", "0.790  0.644  0.971", "254.7       322.3       -67.6")
A(c_mh8, "2.39 1.64 1.48 1.06", "1.228754 2.157713", "1.628281", "[1] 0.4759352", "[1] 0.4802732")
A(c_new8, "7.243254", "6.924909", "[1] 0.8")
A(c_int8, "3   1.1914   0.7551")
A(c_fit9, "-0.27273    0.11944  -2.283 0.022406", "-2.08159", "0.38230    0.14725   2.596 0.009424", "Residual deviance: 2320.8")
A(c_or9, "sglt2       0.761 0.602  0.962", "hf          1.466 1.098  1.956", "prior       2.780 2.227  3.470",
  "sglt2 0.601  0.961", "hf    1.094  1.950", "[1] 0.1108986")
A(c_lr9, "2   42.494 5.922e-10", "220.9504   9.0000", "[1] 1.552537")
A(c_int9, "1   5.4114  0.02001", "-0.7203     0.3151    -2.2859     0.0223", "0.49   0.26   0.90", "0.86  0.42")
A(c_std9, "0.194  0.236 -0.042  0.822")
A(c_rr9, "prior 2.00  1.73   2.32", "sglt2 0.81  0.67   0.97", "0.091     0.074")
A(c_cl9, "thiazide  2  1.24   3.23", "[1] 1.67", "[1] 4")
A(c_spl9, "2   11.595 0.003034", "2344.6     2340.8     2337.0", "1.00 1.01 1.21 1.90 3.28")
A(c_wald9, "2.72   1.83   4.03   1.89   4.18", "24.7  38.3  37.6")
A(c_sep9, "1   16.038 6.207e-05")
A(c_firth9, "3.7952 44.4862 4.6658 5978.1845 13.6498 0.0002", "0.9540  2.5962")
A(c_vif9, "1.32        1.35")
A(c_rate12, "8.169027 11.979737", "14.51742 17.51387", "p-value = 3.804e-06", "0.6226504", "0.5054131 0.7670826", "-8.398595 -3.648777")
A(c_small12, "-0.26 4.26", "0.65 6.20", "0.41 5.84")
A(c_pois12, "-0.23466    0.05490  -4.274 1.92e-05", "-0.85767", "agegrp65-74  0.11627", "agegrp75+    0.52703",
  "cci          0.12812", "Null deviance: 3233.5", "Residual deviance: 3069.8  on 2394  degrees of freedom", "AIC: 5125.7",
  "drugA       0.791 0.710  0.881")
A(c_claims12, "[1] 2\n", "[1] 0.7501711")
A(c_disp12, "3784.2   3069.8   2394.0", "1.58        1.28", "0.640       0.578")
A(c_three12, "0.0549  0.0690  0.0705")
A(c_nb12, "-0.23524", "Theta:  1.167", "AIC: 4850.5", "1.167 0.857 0.691 1.024", "2.771664e+02", "4836.511")
A(c_t1, "[1] 0.01078745", "[1] 0.06877782")
A(c_t2, "1.39   0.87   2.21", "[1] 0.169")
A(c_t3, "young_exact       0.72 0.52 0.98", "young_wald        0.72 0.53 0.97", "old_wald          0.72 0.53 0.96")

S = nb.save_fragment
S("rc14_prop", nb.html(c_prop, marks={"0.04382187 0.23617813": 1, "0.06950833 0.26186194": 2, "0.0581917 0.2673960": 3, "0.06277009 0.27356376": 4}))
S("rc14_three", nb.html(c_three, marks={"3.3638 5.8551 8.5263": 1, "0.0666 0.0155 0.0035 0.0118": 2, "0.6645 0.8008 0.8889": 3}))
S("rc14_fit8", nb.html(c_fit8, marks={"0.38311    0.14850   2.580 0.009884": 1, "-6.78631": 2, "cci1-2       0.31074": 3,
                                      "0.06000    0.01284": 4, "Residual deviance: 1306.9": 5, "Number of Fisher Scoring iterations: 4": 6}))
S("rc14_or8", nb.html(c_or8, marks={"poly        1.467 1.096  1.962": 1, "female      0.779 0.591  1.029": 2, "1.096124 1.962852": 3}))
S("rc14_joint8", nb.html(c_joint8, marks={"cci          3 14.5734   0.002220": 1, "cci     3   1321.6 1331.6 14.6161  0.002176": 2}))
S("rc14_ll8", nb.html(c_ll8, marks={"-653.4729 (df=8)": 1, "1306.946 1306.946 1383.397": 2, "1322.946 1365.452": 3,
                                    "1   6.6349     0.01": 4, "6.6941 0.009673": 5, "1327.581 1364.773": 6}))
S("rc14_nobs8", nb.html(c_nobs8, marks={"[1] 1500": 1, "[1] 1495 1495": 2}))
S("rc14_logbin", nb.html(c_logbin, marks={"please supply starting values": 1, "1.35   1.07   1.71": 2, "0.614409": 3}))
S("rc14_robust", nb.html(c_robust, marks={"0.12023600": 1, "1.36   1.08   1.72": 2}))
S("rc14_rd", nb.html(c_rd, marks={"please supply starting values": 1, "5.3    1.1    9.6": 2, "[1] 11": 3, "20.2 14.9  5.3": 4}))
S("rc14_rdboot", nb.html(c_rdboot, marks={"[1] 1.2 9.3": 1}))
S("rc14_offset", nb.html(c_offset, marks={"1.333333": 1, "\n      1.5": 2}))
S("rc14_gamma", nb.html(c_gamma, marks={"0.7676587": 1, "0.790  0.644  0.971": 2, "254.7       322.3       -67.6": 3}))
S("rc14_gboot", nb.html(c_gboot, marks={"-130.7   -9.5": 1}))
S("rc14_mh8", nb.html(c_mh8, marks={"2.39 1.64 1.48 1.06": 1, "1.228754 2.157713": 2, "1.628281": 3, "X-squared = 11.863": 4,
                                    "[1] 0.4759352": 5, "[1] 0.4802732": 6}))
S("rc14_new8", nb.html(c_new8, marks={"7.243254": 1, "6.924909": 2, "[1] 0.8": 3}))
S("rc14_int8", nb.html(c_int8, marks={"3   1.1914   0.7551": 1}))
S("rc14_fit9", nb.html(c_fit9, marks={"-0.27273    0.11944  -2.283 0.022406": 1, "-2.08159": 2, "cci1-2       0.49873": 3,
                                      "Residual deviance: 2320.8": 4}))
S("rc14_or9", nb.html(c_or9, marks={"sglt2       0.761 0.602  0.962": 1, "sglt2 0.601  0.961": 2, "[1] 0.1108986": 3}))
S("rc14_lr9", nb.html(c_lr9, marks={"2   42.494 5.922e-10": 1, "220.9504   9.0000": 2, "[1] 1.552537": 3}))
S("rc14_int9", nb.html(c_int9, marks={"1   5.4114  0.02001": 1, "-0.7203     0.3151    -2.2859     0.0223": 2,
                                      "0.49   0.26   0.90": 3, "0.86  0.42": 4}))
S("rc14_std9", nb.html(c_std9, marks={"0.194  0.236 -0.042  0.822": 1, "-0.078 -0.004": 2, "0.687  0.981": 3}))
S("rc14_rr9", nb.html(c_rr9, marks={"prior 2.00  1.73   2.32": 1, "sglt2 0.81  0.67   0.97": 2, "0.091     0.074": 3,
                                    "cannot find valid starting values": 4}))
S("rc14_cl9", nb.html(c_cl9, marks={"thiazide  2  1.24   3.23": 1, "[1] 1.67": 2, "[1] 4": 3}))
S("rc14_spl9", nb.html(c_spl9, marks={"2   11.595 0.003034": 1, "2344.6     2340.8     2337.0": 2, "39  63  78 101": 3,
                                      "1.00 1.01 1.21 1.90 3.28": 4}))
S("rc14_wald9", nb.html(c_wald9, marks={"2.72   1.83   4.03   1.89   4.18": 1, "24.7  38.3  37.6": 2}))
S("rc14_sep9", nb.html(c_sep9, marks={"1  0  6": 1, "18.816   1575.855": 2, "16         1": 3, "1   16.038 6.207e-05": 4}))
S("rc14_firth9", nb.html(c_firth9, marks={"3.7952 44.4862 4.6658 5978.1845 13.6498 0.0002": 1, "0.9540  2.5962": 2}))
S("rc14_vif9", nb.html(c_vif9, marks={"cci    1.028644  2": 1, "1.32        1.35": 2, "1.54        1.55": 3, "[1] 0.009701652": 4}))
S("rc14_code9", nb.html(c_code9, marks={"-0.5861412": 1, "y values must be 0 <= y <= 1": 2}))
S("rc14_rate12", nb.html(c_rate12, marks={"8.169027 11.979737": 1, "14.51742 17.51387": 2, "p-value = 3.804e-06": 3,
                                          " 0.5007309 0.7686958": 4, "0.5054131 0.7670826": 5, "-8.398595 -3.648777": 6}))
S("rc14_small12", nb.html(c_small12, marks={"-0.26 4.26": 1, "0.65 6.20": 2, "0.41 5.84": 3}))
S("rc14_pois12", nb.html(c_pois12, marks={"-0.23466    0.05490  -4.274 1.92e-05": 1, "-0.85767": 2, "agegrp75+    0.52703": 3,
                                          "cci          0.12812": 4, "Residual deviance: 3069.8  on 2394  degrees of freedom": 5,
                                          "AIC: 5125.7": 6, "drugA       0.791 0.710  0.881": 7}))
S("rc14_zero12", nb.html(c_zero12, marks={"23.829   42247.17": 1, "20         1": 2, "[1] 0.06150678": 3}))
S("rc14_claims12", nb.html(c_claims12, marks={"2023-05-06 2023-05-12  FALSE": 1, "2023-06-23 2023-06-23   TRUE": 2, "[1] 2\n": 3,
                                              "[1] 0.7501711": 4}))
S("rc14_disp12", nb.html(c_disp12, marks={"3784.2   3069.8   2394.0": 1, "1.58        1.28": 2, "0.640       0.578": 3}))
S("rc14_three12", nb.html(c_three12, marks={"0.0549  0.0690  0.0705  0.0714": 1, "[1] 1.580697": 2}))
S("rc14_nb12", nb.html(c_nb12, marks={"-0.23524    0.07142  -3.294 0.000988": 1, "Theta:  1.167": 2, "AIC: 4850.5": 3,
                                      "0.790  0.687  0.909": 4, "1.167 0.857 0.691 1.024": 5, "2.771664e+02": 6}))
S("rc14_t1", nb.html(c_t1, marks={"[1] 0.01078745": 1, "[1] 0.06877782": 2}))
S("rc14_t2", nb.html(c_t2, marks={"1.39   0.87   2.21": 1, "[1] 0.169": 2}))
S("rc14_t3", nb.html(c_t3, marks={"young_exact       0.72 0.52 0.98": 1, "old_exact         0.72 0.52 0.97": 2}))
nb.save_ipynb("5장 로지스틱 회귀와 포아송 회귀 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 8장(효과 지표, 일반화 선형모형, 우도, 교란, 가변수), 9장(로지스틱 회귀), 12장(발생률과 포아송 회귀)의 예를 "
                    "R로 계산합니다. Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"car\", \"sandwich\", \"lmtest\", \"boot\", \"DescTools\", \"rms\", \"logistf\"))를 실행합니다.",
              notes={c_prop.n: "## a. 비율의 신뢰구간과 세 검정", c_fit8.n: "## b. 로지스틱 회귀의 계수와 오즈비",
                     c_ll8.n: "## c. 우도비 검정과 AIC", c_logbin.n: "## d. 상대위험도, 위험차, 의료비",
                     c_mh8.n: "## e. 층화와 교호작용", c_fit9.n: "## f. 보정 모형과 결과 읽기",
                     c_spl9.n: "## g. 로지스틱 모형 점검", c_rate12.n: "## h. 발생률과 포아송 회귀",
                     c_disp12.n: "## i. 과산포와 음이항 회귀", c_t1.n: "## j. 과제 정답"})
print("rc14 fragments and notebook written")
