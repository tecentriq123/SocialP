"""R과 파이썬으로 하는 통계분석 8장 · Cox 모형과 비례위험 점검 — R cells run for real with rkit (survival).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc03.py      (NOMARK=1 … 로 돌리면 모든 셀의 출력을 찍는다)

자료는 survival 패키지의 pbc(원발성 담즙성 담관염 무작위배정 312명). 보건통계학 기초 파이썬 실습 11이 Rdatasets의 pbc.csv로
읽는 것과 같은 자료라서 끝의 대조 블록에서 그 숫자(단변수 HR 1.059, 다변수 0.938·1.369·1.842·0.372·2.582, 일치도 0.846,
부분 AIC 1094.35·1124.7, 층화 0.922, 기간별 1.57·1.06·P 0.0234·dpca 0.913, 이식 포함 사건 144·HR 0.986)와 맞춘다.
비례위험 검정(cox.zph)은 lifelines와 계산 방식이 달라 숫자가 다르다(본문에서 설명).
e 절은 보건통계학 기초 11장의 예(pub/data/ch11_cohort.csv, ch11_bp.csv; gen/data_rc03e.py)를 R로 계산해 11장의 숫자와 맞춘다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc03")

# a. 자료
c_load = nb.cell('''
library(survival)
d <- subset(pbc, !is.na(trt))      # 무작위배정된 312명만
nrow(d)
table(status = d$status, trt = d$trt)
''', title="PBC 자료에서 무작위배정 환자 고르기")

c_vars = nb.cell('''
d$death  <- as.integer(d$status == 2)   # 사망 = 1, 간이식과 생존 = 0
d$years  <- d$time / 365.25
d$dpca   <- as.integer(d$trt == 1)      # D-penicillamine = 1, 위약 = 0
d$female <- as.integer(d$sex == "f")
d$age10  <- d$age / 10                  # 10세 단위
d$edema  <- factor(d$edema)             # 0, 0.5, 1을 범주로
colSums(is.na(d))[colSums(is.na(d)) > 0]
''', title="사건, 시간, 공변량 만들기")

# b. Cox
c_uni = nb.cell('''
fit1 <- coxph(Surv(years, death) ~ dpca, data = d)
summary(fit1)
survdiff(Surv(years, death) ~ dpca, data = d)$chisq
''', title="치료만 넣은 Cox 모형")

c_multi = nb.cell('''
fit <- coxph(Surv(years, death) ~ dpca + age10 + female + log2(bili) +
               albumin + edema + protime, data = d)
summary(fit)
''', title="다변수 Cox 모형")

c_na = nb.cell('''
coxph(Surv(years, death) ~ dpca + chol, data = d)
''', title="결측이 있는 변수를 넣으면")

c_tab = nb.cell('''
library(broom)
hr <- tidy(fit, exponentiate = TRUE, conf.int = TRUE)
data.frame(term = hr$term,
           `HR (95% CI)` = sprintf("%.2f (%.2f-%.2f)", hr$estimate,
                                   hr$conf.low, hr$conf.high),
           P = format.pval(hr$p.value, digits = 2, eps = 0.001),
           check.names = FALSE)
''', title="논문에 옮길 위험비 표")

c_aic = nb.cell('''
fit_lin <- update(fit, . ~ . - log2(bili) + bili)   # 빌리루빈을 그대로 넣은 모형
AIC(fit_lin, fit)
''', title="빌리루빈을 어떤 꼴로 넣을지")

# c. PH
c_zph = nb.cell('''
zph <- cox.zph(fit)
zph
''', title="cox.zph()로 비례위험 검정")

c_zph2 = nb.cell('''
cox.zph(fit, transform = "rank", terms = FALSE)
''', title="시간 변환과 가변수별 검정")

c_zplot = nb.cell('''
par(mfrow = c(1, 2))
plot(zph, var = "dpca")
abline(h = coef(fit)["dpca"], lty = 3, col = "gray40")
plot(zph, var = "protime")
abline(h = coef(fit)["protime"], lty = 3, col = "gray40")
''', title="척도화 Schoenfeld 잔차 그림", fig=(9, 3.8))

c_ll = nb.cell('''
km_ed <- survfit(Surv(years, death) ~ edema, data = d)
cols <- c("black", "#1f5f8b", "#c8553d")
plot(km_ed, fun = "cloglog", col = cols,
     xlab = "Years (log scale)", ylab = "log(-log S(t))")
legend("topleft", c("No edema", "Edema 0.5", "Edema 1.0"),
       col = cols, lty = 1, bty = "n")
''', title="부종 범주별 log(-log) 그림", fig=(7, 4.2))

# d. 대안과 보정 곡선
c_str = nb.cell('''
fit_s <- coxph(Surv(years, death) ~ dpca + age10 + female + log2(bili) +
                 albumin + protime + strata(edema), data = d)
fit_s
''', title="부종으로 층화한 Cox 모형")

c_split = nb.cell('''
long <- survSplit(Surv(years, death) ~ ., data = d, cut = 3, episode = "period")
c(nrow(d), nrow(long))
long[long$id %in% c(1, 4), c("id", "tstart", "years", "death", "period", "protime")]
''', title="추적 기간을 3년에서 나눈 긴 자료")

c_tsp = nb.cell('''
long$late <- as.integer(long$period == 2)
fit_t <- coxph(Surv(tstart, years, death) ~ dpca + age10 + female + log2(bili) +
                 albumin + edema + protime + protime:late, data = long)
round(summary(fit_t)$coefficients[c("dpca", "protime", "protime:late"), ], 4)
exp(coef(fit_t)["protime"] + coef(fit_t)["protime:late"])   # 3년 뒤의 위험비
''', title="기간별 위험비")

c_std = nb.cell('''
std_surv <- function(newd, times)      # 환자마다 예측한 곡선의 평균
  rowMeans(summary(survfit(fit, newdata = newd), times = times)$surv)
at <- c(1, 5, 10)
round(cbind(years   = at,
            dpca    = std_surv(transform(d, dpca = 1), at),
            placebo = std_surv(transform(d, dpca = 0), at)), 3)
grid <- seq(0, 12, by = 0.1)
plot(grid, std_surv(transform(d, dpca = 1), grid), type = "s", ylim = c(0, 1),
     col = "#1f5f8b", xlab = "Years since randomization",
     ylab = "Adjusted survival probability")
lines(grid, std_surv(transform(d, dpca = 0), grid), type = "s", col = "#c8553d")
legend("bottomleft", c("D-penicillamine", "Placebo"),
       col = c("#1f5f8b", "#c8553d"), lty = 1, bty = "n")
''', title="직접 표준화한 보정 생존곡선", fig=(7, 4.2))

# e. 보건통계학 기초 11장의 예
c11_fit1 = nb.cell('''
co <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch11_cohort.csv")
co$drug      <- factor(co$drug, levels = c("B", "A"))            # 약물 B가 기준
co$imdc      <- factor(co$imdc, levels = c("favorable", "intermediate", "poor"))
co$histology <- factor(co$histology, levels = c("clear", "nonclear"))
f1 <- coxph(Surv(time, status) ~ drug, data = co)
summary(f1)
f1$loglik                     # 계수가 모두 0일 때와 추정값에서의 로그 부분우도
''', title="약물만 넣은 Cox 모형 (11장 '위험률과 위험비')")

c11_ties = nb.cell('''
ties <- c("efron", "breslow", "exact")
hr_by <- function(tm) sapply(ties, function(m)
  exp(coef(coxph(Surv(tm, status) ~ drug, data = co, ties = m))))
co$t6 <- ceiling(co$time / 6) * 6          # 사망 시간을 6개월 단위로 뭉뚱그림
round(rbind(original = hr_by(co$time), by_6_months = hr_by(co$t6)), 4)
''', title="동점 처리 방법과 위험비 (표 11-3)")

c11_multi = nb.cell('''
fm <- coxph(Surv(time, status) ~ drug + I(age / 10) + male + imdc +
              nephrectomy + histology, data = co)
round(cbind(HR = exp(coef(fm)), exp(confint(fm)),
            P = summary(fm)$coefficients[, "Pr(>|z|)"]), 3)
anova(update(fm, . ~ . - imdc), fm)        # IMDC 위험군 전체(자유도 2)의 우도비 검정
fm_i <- update(fm, . ~ . - imdc + relevel(imdc, ref = "intermediate"))
round(exp(cbind(coef(fm_i), confint(fm_i)))[7, ], 2)   # poor 대 intermediate
''', title="다변수 모형 (11장 '논문에서 읽기'의 Table 2)")

c11_zph = nb.cell('''
cox.zph(fm)                                       # 기본 transform = "km"
rbind(identity = cox.zph(fm, transform = "identity")$table[c("male", "histology", "GLOBAL"), "p"],
      rank     = cox.zph(fm, transform = "rank")$table[c("male", "histology", "GLOBAL"), "p"],
      log      = cox.zph(fm, transform = "log")$table[c("male", "histology", "GLOBAL"), "p"])
''', title="비례위험 검정과 시간 변환")

c11_period = nb.cell('''
long <- survSplit(Surv(time, status) ~ ., data = co, cut = 12, episode = "period")
long$nc_0_12  <- as.integer(long$histology == "nonclear" & long$period == 1)
long$nc_after <- as.integer(long$histology == "nonclear" & long$period == 2)
fp <- coxph(Surv(tstart, time, status) ~ drug + I(age / 10) + male + imdc +
              nephrectomy + nc_0_12 + nc_after, data = long)
round(exp(cbind(HR = coef(fp), confint(fp)))[c("drugA", "nc_0_12", "nc_after"), ], 2)
anova(update(fp, . ~ . - nc_0_12 - nc_after + I(nc_0_12 + nc_after)), fp)   # 두 기간의 HR이 같은가
fs <- update(fm, . ~ . - histology + strata(histology))            # 조직형으로 층화
round(exp(cbind(coef(fs), confint(fs)))["drugA", ], 2)
round(cox.zph(fs)$table[, "p"], 3)                                  # 층화 뒤 나머지 변수의 비례위험 검정
co$nonclear <- as.integer(co$histology == "nonclear")
ft <- coxph(Surv(time, status) ~ drug + I(age / 10) + male + imdc + nephrectomy +
              histology + tt(nonclear), data = co,
            tt = function(x, t, ...) x * log(t))                   # 조직형 × log(개월)
round(summary(ft)$coefficients[c("histologynonclear", "tt(nonclear)"), 1:3], 4)
''', title="기간별 위험비, 층화, 시간과의 교호작용")

c11_std = nb.cell('''
std_surv <- function(g, times) {          # 모두 약물 g를 받았다고 놓은 곡선의 평균
  nd <- transform(co, drug = factor(g, levels = c("B", "A")))
  rowMeans(summary(survfit(fm, newdata = nd), times = times)$surv)
}
at <- c(12, 24, 48)
round(rbind(A = std_surv("A", at), B = std_surv("B", at)), 3)
summary(survfit(fm, newdata = data.frame(drug = c("B", "A"), age = 70, male = 1,
  imdc = "intermediate", nephrectomy = 1, histology = "clear")), times = c(12, 24))$surv
''', title="직접 표준화한 보정 생존율과 한 환자의 예측")

c11_iptw = nb.cell('''
ps <- glm(drug == "A" ~ I(age / 10) + male + imdc + nephrectomy + histology,
          family = binomial, data = co)$fitted.values      # 약물 A를 받을 확률
pA <- mean(co$drug == "A")
co$w_s <- ifelse(co$drug == "A", pA / ps, (1 - pA) / (1 - ps))   # 안정화 가중치
co$w_u <- ifelse(co$drug == "A", 1 / ps, 1 / (1 - ps))           # 안정화하지 않은 가중치
round(range(co$w_s), 2)
fw <- coxph(Surv(time, status) ~ drug, data = co, weights = w_s)
summary(fw)$coefficients
round(summary(fw)$conf.int, 3)
fu <- coxph(Surv(time, status) ~ drug, data = co, weights = w_u)
round(summary(fu)$coefficients[, c("se(coef)", "robust se")], 4)
''', title="역확률 가중 Cox 모형")

c11_infl = nb.cell('''
fr <- update(fm, robust = TRUE)           # 환자 한 명이 한 줄이면 id가 군집
round(summary(fr)$coefficients["drugA", c("se(coef)", "robust se")], 4)
loo <- sapply(seq_len(nrow(co)), function(i)         # 한 명씩 빼고 다시 적합
  coef(update(fm, data = co[-i, ]))["drugA"])
round(exp(range(loo)), 3)
se <- sqrt(fm$var[1, 1])
round(max(abs(coef(fm)["drugA"] - loo)) / se, 2)              # 가장 큰 변화 ÷ 표준오차
round(max(abs(residuals(fm, type = "dfbeta")[, 1])) / se, 2)  # 다시 적합하지 않은 근사
''', title="강건 표준오차와 한 명씩 빼고 다시 적합하기")

c11_bp = nb.cell('''
bp <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch11_bp.csv")
kn <- quantile(bp$sbp, c(0.05, 0.35, 0.65, 0.95))    # 매듭: 5·35·65·95번째 백분위수
round(kn, 1)
rcs4 <- function(x) Hmisc::rcspline.eval(x, knots = kn, inclx = TRUE)   # 제한 삼차 스플라인
f_lin <- coxph(Surv(time, status) ~ I(sbp / 10), data = bp)
f_rcs <- coxph(Surv(time, status) ~ rcs4(sbp), data = bp)
round(exp(coef(f_lin)), 3)                            # 10 mmHg당 위험비(직선)
anova(f_lin, f_rcs)                                   # 곡선이 필요한가(자유도 2)
at <- c(100, 140, 160, 180)
X  <- sweep(rcs4(at), 2, rcs4(120))                   # 120 mmHg와의 차이
lp <- drop(X %*% coef(f_rcs)); se <- sqrt(diag(X %*% vcov(f_rcs) %*% t(X)))
round(cbind(sbp = at, HR = exp(lp), lower = exp(lp - 1.96 * se), upper = exp(lp + 1.96 * se)), 2)
''', title="혈압을 스플라인으로 넣기 (그림 11-2)")

# f. 과제
c_t1 = nb.cell('''
b <- coef(fit)
exp(b["log2(bili)"] * log2(c(2 / 1, 10 / 5, 11 / 10)))   # 1→2, 5→10, 10→11 mg/dL
exp(2 * b["age10"])                                         # 20세 많을 때
''', title="과제 1 정답: 빌리루빈과 나이의 위험비")

c_t2 = nb.cell('''
fit_tx <- update(fit, Surv(years, status > 0) ~ .)    # 사망 또는 간이식
c(death = fit$nevent, death_or_tx = fit_tx$nevent)
round(cbind(death = exp(coef(fit)), death_or_tx = exp(coef(fit_tx))), 3)
''', title="과제 2 정답: 간이식도 사건으로 세면")

c_t3 = nb.cell('''
wrong <- coxph(Surv(years, death) ~ dpca + age10 + female + log2(bili) +
                 albumin + edema + protime + dpca:years, data = d)
round(summary(wrong)$coefficients[c("dpca", "dpca:years"), ], 4)
right <- coxph(Surv(years, death) ~ dpca + age10 + female + log2(bili) +
                 albumin + edema + protime + tt(dpca), data = d,
               tt = function(x, t, ...) x * t)
round(summary(right)$coefficients[c("dpca", "tt(dpca)"), ], 4)
''', title="과제 3 정답: 치료 × 추적 시간")

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

# ---------------------------------------------------------------- 파이썬 실습 11과 대조
out = {c.n: c.text for c in nb.cells}
assert "1.059" in out[c_uni.n] and "0.7453     1.504" in out[c_uni.n] and "0.1017" in out[c_uni.n]
for v in ("0.93752", "1.36907", "1.84188", "0.37212", "2.58203", "Concordance= 0.846", "201.6  on 8 df"):
    assert v in out[c_multi.n], v
assert "1124.717" in out[c_aic.n] and "1094.350" in out[c_aic.n]
assert "0.92162" in out[c_str.n] and "132.3  on 6 df" in out[c_str.n]
assert "312 552" in out[c_split.n]
assert "0.9127" in out[c_tsp.n] and "1.5678" in out[c_tsp.n] and "0.0234" in out[c_tsp.n] and "1.064" in out[c_tsp.n]
assert "144" in out[c_t2.n] and "0.986" in out[c_t2.n]
# 보건통계학 기초 11장의 숫자
for v in ['n= 300, number of events= 210', '-0.3163    0.7289   0.1391 -2.274    0.023', '0.555    0.9573', 'Concordance= 0.545', 'Likelihood ratio test= 5.18', '[1] -1055.689 -1053.100']:
    assert v in out[c11_fit1.n], ('c11_fit1', v)
for v in ['0.7289        0.7287      0.7278', '0.7243        0.7435      0.7059']:
    assert v in out[c11_ties.n], ('c11_ties', v)
for v in ['drugA             0.856 0.645  1.135 0.280', 'histologynonclear 1.833', '30.825  2  2.025e-07', '2.44   1.72   3.47']:
    assert v in out[c11_multi.n], ('c11_multi', v)
for v in ['drug         0.00112  1 0.9733', 'histology    9.70096  1 0.0018', 'GLOBAL      13.54944  7 0.0598', 'identity 0.01159328']:
    assert v in out[c11_zph.n], ('c11_zph', v)
for v in ['nc_0_12  3.30  2.03   5.35', '10.413  1   0.001251', '0.86   0.65   1.14', 'tt(nonclear)      -0.4833']:
    assert v in out[c11_period.n], ('c11_period', v)
for v in ['A 0.700 0.507 0.317', '[2,] 0.4862459 0.5395545']:
    assert v in out[c11_std.n], ('c11_std', v)
for v in ['[1] 0.67 2.35', '0.1377523 0.1434797', '0.87      1.149     0.657     1.152', '0.0974    0.1435']:
    assert v in out[c11_iptw.n], ('c11_iptw', v)
for v in ['0.1441    0.1474', '[1] 0.831 0.891', '[1] 0.28', '[1] 0.24']:
    assert v in out[c11_infl.n], ('c11_infl', v)
for v in ['102.4 128.2 142.8 167.9', '1.144', '45.173  2  1.552e-10', '[3,] 160 1.86  1.58  2.19']:
    assert v in out[c11_bp.n], ('c11_bp', v)
assert "8.1648" in out[c_t3.n] and "0.6787" in out[c_t3.n]

S = nb.save_fragment
S("rc03_load", nb.html(c_load, marks={"[1] 312": 1, "2 65 60": 2}))
S("rc03_vars", nb.html(c_vars, marks={"chol": 1}))
S("rc03_uni", nb.html(c_uni, marks={"n= 312, number of events= 125": 1, "1.05889": 2, "0.7453     1.504": 3,
                                    "Concordance= 0.499": 4, "Score (logrank) test = 0.1": 5, "[1] 0.1017055": 6}))
S("rc03_multi", nb.html(c_multi, marks={"log2(bili)  0.61079   1.84188": 1, "edema1      0.94858   2.58203": 2,
                                        "0.6416    1.3699": 3, "Concordance= 0.846": 4, "201.6  on 8 df": 5}))
S("rc03_na", nb.html(c_na, marks={"n= 284": 1, "(28 observations deleted due to missingness)": 2}))
S("rc03_tab", nb.html(c_tab, marks={"0.94 (0.64-1.37)": 1, "<0.001": 2}))
S("rc03_aic", nb.html(c_aic, marks={"1124.717": 1, "1094.350": 2}))
S("rc03_zph", nb.html(c_zph, marks={"4.7242  2 0.094": 1, "4.7670  1 0.029": 2, "12.2654  8 0.140": 3}))
S("rc03_zph2", nb.html(c_zph2, marks={"0.97053  1 0.325": 1, "3.86355  1 0.049": 2, "4.30850  1 0.038": 3}))
S("rc03_zplot", nb.html(c_zplot))
S("rc03_ll", nb.html(c_ll))
S("rc03_str", nb.html(c_str, marks={"0.92162": 1, "132.3  on 6 df": 2}))
S("rc03_split", nb.html(c_split, marks={"312 552": 1, "1.095140": 2, "3.000000": 3, "5.270363": 4}))
S("rc03_tsp", nb.html(c_tsp, marks={"1.5678": 1, "0.0234": 2, "1.064306": 3, "0.9127": 4}))
S("rc03_std", nb.html(c_std, marks={"0.714": 1}))
S('rc03_c11fit1', nb.html(c11_fit1, marks={'n= 300, number of events= 210': 1, '-0.3163    0.7289   0.1391 -2.274    0.023': 2, '0.555    0.9573': 3, 'Concordance= 0.545': 4, 'Likelihood ratio test= 5.18': 5, '[1] -1055.689 -1053.100': 6}))
S('rc03_c11ties', nb.html(c11_ties, marks={'0.7289        0.7287      0.7278': 1, '0.7243        0.7435      0.7059': 2}))
S('rc03_c11multi', nb.html(c11_multi, marks={'drugA             0.856 0.645  1.135 0.280': 1, 'histologynonclear 1.833': 2, '30.825  2  2.025e-07': 3, '2.44   1.72   3.47': 4}))
S('rc03_c11zph', nb.html(c11_zph, marks={'drug         0.00112  1 0.9733': 1, 'histology    9.70096  1 0.0018': 2, 'GLOBAL      13.54944  7 0.0598': 3, 'identity 0.01159328': 4}))
S('rc03_c11period', nb.html(c11_period, marks={'nc_0_12  3.30  2.03   5.35': 1, '10.413  1   0.001251': 2, '0.86   0.65   1.14': 3, 'tt(nonclear)      -0.4833': 4}))
S('rc03_c11std', nb.html(c11_std, marks={'A 0.700 0.507 0.317': 1, '[2,] 0.4862459 0.5395545': 2}))
S('rc03_c11iptw', nb.html(c11_iptw, marks={'[1] 0.67 2.35': 1, '0.1377523 0.1434797': 2, '0.87      1.149     0.657     1.152': 3, '0.0974    0.1435': 4}))
S('rc03_c11infl', nb.html(c11_infl, marks={'0.1441    0.1474': 1, '[1] 0.831 0.891': 2, '[1] 0.28': 3, '[1] 0.24': 4}))
S('rc03_c11bp', nb.html(c11_bp, marks={'102.4 128.2 142.8 167.9': 1, '1.144': 2, '45.173  2  1.552e-10': 3, '[3,] 160 1.86  1.58  2.19': 4}))
S("rc03_t1", nb.html(c_t1, marks={"1.087613": 1}))
S("rc03_t2", nb.html(c_t2, marks={"144": 1}))
S("rc03_t3", nb.html(c_t3, marks={"8.1648": 1}))
nb.save_ipynb("8장 Cox 모형과 비례위험 점검 (R과 파이썬으로 하는 통계분석)",
              intro="survival 패키지의 원발성 담즙성 담관염 자료(pbc)로 Cox 모형을 맞추고, 비례위험 가정을 점검하고, "
                    "보정 생존곡선을 그립니다. 보건통계학 기초의 파이썬 실습 11과 같은 자료입니다. e 절에서는 보건통계학 기초 11장의 예를 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 분석할 자료 만들기", c_uni.n: "## b. Cox 모형과 위험비",
                     c_zph.n: "## c. 비례위험 가정 점검", c_str.n: "## d. 가정이 어긋날 때와 보정 생존곡선",
                     c11_fit1.n: "## e. 보건통계학 기초 11장의 예", c_t1.n: "## f. 과제 정답"})
print("rc03 fragments and notebook written")
