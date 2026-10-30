"""R과 통계분석 3장 · Cox 모형과 비례위험 점검 — R cells run for real with rkit (survival).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc03.py      (NOMARK=1 … 로 돌리면 모든 셀의 출력을 찍는다)

자료는 survival 패키지의 pbc(원발성 담즙성 담관염 무작위배정 312명). 보건통계학 기초 파이썬 실습 11이 Rdatasets의 pbc.csv로
읽는 것과 같은 자료라서 끝의 대조 블록에서 그 숫자(단변수 HR 1.059, 다변수 0.938·1.369·1.842·0.372·2.582, 일치도 0.846,
부분 AIC 1094.35·1124.7, 층화 0.922, 기간별 1.57·1.06·P 0.0234·dpca 0.913, 이식 포함 사건 144·HR 0.986)와 맞춘다.
비례위험 검정(cox.zph)은 lifelines와 계산 방식이 달라 숫자가 다르다(본문에서 설명).
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

# e. 과제
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
S("rc03_t1", nb.html(c_t1, marks={"1.087613": 1}))
S("rc03_t2", nb.html(c_t2, marks={"144": 1}))
S("rc03_t3", nb.html(c_t3, marks={"8.1648": 1}))
nb.save_ipynb("3장 Cox 모형과 비례위험 점검 (R과 통계분석)",
              intro="survival 패키지의 원발성 담즙성 담관염 자료(pbc)로 Cox 모형을 맞추고, 비례위험 가정을 점검하고, "
                    "보정 생존곡선을 그립니다. 보건통계학 기초의 파이썬 실습 11과 같은 자료입니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 분석할 자료 만들기", c_uni.n: "## b. Cox 모형과 위험비",
                     c_zph.n: "## c. 비례위험 가정 점검", c_str.n: "## d. 가정이 어긋날 때와 보정 생존곡선",
                     c_t1.n: "## e. 과제 정답"})
print("rc03 fragments and notebook written")
