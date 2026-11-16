# 부록 A g 절 (복합표본 조사자료 분석): hypothetical KNHANES-like survey, all numbers in the section.
# Run: Rscript gen/svy_apg.R   (writes gen/apg_survey.csv; R 4.3.3, survey 4.2.1)
suppressMessages(library(survey))
simulate <- function(seed) {
  set.seed(seed)
  H <- 10; M <- 6                         # 10 strata (1-5 동, 6-10 읍면), 6 PSUs (조사구) each
  rural <- rep(c(0, 1), each = 5)
  rows <- list()
  for (h in 1:H) for (m in 1:M) {
    psu <- (h - 1) * M + m
    n <- rpois(1, 40)
    pold <- plogis(qlogis(ifelse(rural[h] == 1, .42, .28)) + rnorm(1, 0, .35))
    pyoung <- (1 - pold) * ifelse(rural[h] == 1, .30, .40)
    age <- sample(c("19-39", "40-64", "65+"), n, TRUE, prob = c(pyoung, 1 - pold - pyoung, pold))
    sex <- sample(c("M", "F"), n, TRUE)
    u <- rnorm(1, 0, .45)                 # PSU effect -> clustering
    lp <- c("19-39" = -4.4, "40-64" = -1.75, "65+" = 0.05)[age] + 0.08 * (sex == "M") + u
    y <- rbinom(n, 1, plogis(lp))
    base <- ifelse(rural[h] == 1, 9000, 21000)                  # 동 지역은 추출률이 낮음
    fac <- c("19-39" = 1.9, "40-64" = 1.0, "65+" = 0.62)[age]   # 젊은 층의 낮은 응답률 보정
    w <- base * fac * exp(rnorm(n, 0, .15))
    rows[[psu]] <- data.frame(stratum = h, psu = psu, rural = rural[h], age, sex, htn_med = y, w)
  }
  d <- do.call(rbind, rows)
  d$w <- d$w * 43e6 / sum(d$w)            # 성인 4,300만 명으로 맞춤
  d
}
d <- simulate(28)
set.seed(2806)                            # rare subgroup: 만성 신장병 진단 (drawn after, so nothing above changes)
d$ckd <- rbinom(nrow(d), 1, plogis(c("19-39" = -6.2, "40-64" = -4.6, "65+" = -3.6)[d$age] + 1.3 * d$htn_med))
write.csv(d, "gen/apg_survey.csv", row.names = FALSE)
d$sex <- relevel(factor(d$sex), "F")
f <- function(x, k = 4) format(round(x, k), nsmall = k)
cat("n", nrow(d), "strata", length(unique(d$stratum)), "PSU", length(unique(d$psu)), "sum w", sum(d$w), "\n")
cat("PSU size range", range(table(d$psu)), " weight range", range(d$w), " mean w", mean(d$w), "\n")
cat("age dist unweighted:", f(prop.table(table(d$age))), "\n")
cat("age dist weighted:  ", f(prop.table(xtabs(w ~ age, d))), "\n")
cat("prev by age (unw):", f(tapply(d$htn_med, d$age, mean)), "\n")

des   <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = d, nest = TRUE)
wonly <- svydesign(ids = ~1, weights = ~w, data = d)
cat("degf", degf(des), "\n")
p0 <- mean(d$htn_med); se0 <- sqrt(p0 * (1 - p0) / nrow(d))
cat("(1) unweighted", f(p0), "SE", f(se0), "CI", f(p0 + c(-1, 1) * qnorm(.975) * se0), "\n")
m1 <- svymean(~htn_med, wonly); cat("(2) weights only", f(coef(m1)), "SE", f(SE(m1)), "CI", f(confint(m1)), "\n")
m2 <- svymean(~htn_med, des, deff = TRUE)
cat("(3) design", f(coef(m2)), "SE", f(SE(m2)), "deff", f(deff(m2)), "Wald df CI", f(as.vector(coef(m2)) + c(-1, 1) * qt(.975, degf(des)) * as.vector(SE(m2))), "\n")
ci <- svyciprop(~htn_med, des, method = "logit"); cat("    svyciprop logit", f(attr(ci, "ci")), "\n")
cat("    effective n", f(nrow(d) / deff(m2), 1), "\n")
cat("    weighted N on meds", f(svytotal(~htn_med, des)[1], 0), "\n")
# sex comparison
pm <- svyby(~htn_med, ~sex, des, svymean, covmat = TRUE); print(pm)
cat("unw by sex", f(tapply(d$htn_med, d$sex, mean)), " n", table(d$sex), "\n")
tu <- table(d$sex, d$htn_med); print(tu)
cu <- chisq.test(tu, correct = FALSE); cat("chi unweighted", f(cu$statistic, 2), "P", f(cu$p.value), "\n")
tw <- xtabs(w ~ sex + htn_med, d); print(round(tw))
cw <- chisq.test(tw, correct = FALSE); cat("chi weighted counts", f(cw$statistic, 0), "P", format(cw$p.value), "\n")
rs <- svychisq(~sex + htn_med, des); print(rs)
rs1 <- svychisq(~sex + htn_med, des, statistic = "Chisq"); print(rs1)
rsw <- svychisq(~sex + htn_med, wonly); cat("weights-only RS P", f(rsw$p.value), "\n")
# design effect of the difference
dd <- svycontrast(pm, c(M = 1, F = -1)); print(dd)
# logistic regression: male OR adjusted for age group and rural
g0 <- glm(htn_med ~ sex + age + rural, family = binomial, data = d)
g1 <- svyglm(htn_med ~ sex + age + rural, design = des, family = quasibinomial())
for (g in list(g0, g1)) { b <- coef(g)["sexM"]; s <- sqrt(vcov(g)["sexM", "sexM"])
  dfq <- if (inherits(g, "svyglm")) g$df.residual else Inf
  cat("OR male", f(exp(b), 3), "CI", f(exp(b + c(-1, 1) * qt(.975, dfq) * s), 3), "P", f(2 * pt(-abs(b / s), dfq)), "df", dfq, "\n") }
# domain: 만성 신장병 진단자
dsub <- d[d$ckd == 1, ]
np <- tapply(dsub$psu, dsub$stratum, function(x) length(unique(x)))
cat("CKD n", nrow(dsub), "PSUs with members", length(unique(dsub$psu)), "strata", length(np), "strata with 1 PSU", names(np)[np == 1], "\n")
cat("CKD weighted prevalence", f(coef(svymean(~ckd, des))), "SE", f(SE(svymean(~ckd, des))), "\n")
sub_ok <- subset(des, ckd == 1); m_ok <- svymean(~htn_med, sub_ok)
cat("design subset", f(coef(m_ok)), "SE", f(SE(m_ok)), "degf", degf(sub_ok), "CI", f(attr(svyciprop(~htn_med, sub_ok), "ci")), "\n")
r <- try(svymean(~htn_med, svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = dsub, nest = TRUE)), silent = TRUE)
cat("data subset (default):", if (inherits(r, "try-error")) conditionMessage(attr(r, "condition")) else "no error", "\n")
options(survey.lonely.psu = "adjust")
db <- svydesign(ids = ~psu, strata = ~stratum, weights = ~w, data = dsub, nest = TRUE); mb <- svymean(~htn_med, db)
cat("data subset (adjust)", f(coef(mb)), "SE", f(SE(mb)), "degf", degf(db), "\n")
options(survey.lonely.psu = "fail")
