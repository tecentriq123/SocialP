"""R과 파이썬으로 하는 통계분석 11장 · 성향점수 — R cells run for real with rkit.

run:  python3 gen/r_rc18.py      (NOMARK=1 python3 gen/r_rc18.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 15장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-19 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 15장의 가상 코호트 10,000명(pub/data/ps_cohort.csv; gen/nums_ch15.py와 같은 seed), 1:1 매칭에서 약물 A 사용자를
짝짓는 순서(pub/data/ch15_match_order.csv; gen/data_rc18.py). 숫자는 gen/nums_ch15.py와 대조한다
(부트스트랩과 MatchIt은 난수가 달라 끝자리가 다를 수 있음).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc18")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 보정할 변수
c_adj = nb.cell(f'''
suppressPackageStartupMessages({{library(sandwich); library(survival)}})
d <- read.csv("{D}ps_cohort.csv")                       # 한 줄 = 환자 한 명
rr_ci <- function(fit, V) {{                             # 상대위험도와 95% 신뢰구간
  b <- coef(fit)["drugA"]; se <- sqrt(V["drugA", "drugA"])
  round(exp(c(RR = unname(b), lower = unname(b) - 1.96 * se, upper = unname(b) + 1.96 * se)), 3)
}}
sets <- list(none   = hosp_1y ~ drugA,
             agesex = hosp_1y ~ drugA + age + female,
             hf_ckd = hosp_1y ~ drugA + age + female + hf + ckd,
             all7   = hosp_1y ~ drugA + age + female + hf + ckd + prior_hosp + n_drug + tertiary,
             frail  = hosp_1y ~ drugA + age + female + hf + ckd + prior_hosp + n_drug + tertiary + frail)
t(sapply(sets, function(f) {{
  fit <- glm(f, family = poisson, data = d)              # 수정 포아송 회귀
  rr_ci(fit, vcovHC(fit, type = "HC0"))                  # 강건 분산
}}))
''', title="보정할 변수를 늘려 가며 상대위험도 구하기 (15장 a 절)")

# ------------------------------------------------------------------ b. 성향점수
c_ps = nb.cell('''
ps_fit <- glm(drugA ~ age + female + hf + ckd + prior_hosp + n_drug + tertiary,
              family = binomial, data = d)
summary(ps_fit)$coefficients
d$ps <- fitted(ps_fit)                                    # 환자마다의 성향점수
nobs(ps_fit)                                              # 분석에 들어간 인원
sapply(split(d$ps, d$drugA), range)                       # 군별 점수의 범위(0 = B, 1 = A)
r <- rank(d$ps); n1 <- sum(d$drugA); n0 <- sum(1 - d$drugA)
(sum(r[d$drugA == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)   # C 통계량
ex <- data.frame(age = c(78, 58), female = c(1, 0), hf = c(1, 0), ckd = c(1, 0),
                 prior_hosp = c(1, 0), n_drug = c(9, 4), tertiary = c(1, 0))
round(predict(ps_fit, newdata = ex, type = "response"), 4)   # 15장 b 절의 두 환자
''', title="성향점수 모형과 두 군의 겹침 (15장 b 절)")

# ------------------------------------------------------------------ c. 매칭과 가중
c_match = nb.cell(f'''
lg  <- qlogis(d$ps)                       # 성향점수의 로짓
cal <- 0.2 * sd(lg)                       # 허용 한계: 로짓 표준편차의 0.2배
ord <- read.csv("{D}ch15_match_order.csv")$id   # A 사용자를 짝짓는 순서(15장과 같은 무작위 순서)
ctrl  <- which(d$drugA == 0)
avail <- rep(TRUE, length(ctrl))          # 아직 짝이 없는 B 사용자
pt <- pc <- integer(0)
for (t in match(ord, d$id)) {{             # 비복원 1:1 최근접 매칭
  dist <- abs(lg[ctrl] - lg[t]); dist[!avail] <- Inf
  j <- which.min(dist)
  if (dist[j] <= cal) {{ pt <- c(pt, t); pc <- c(pc, ctrl[j]); avail[j] <- FALSE }}
}}
c(caliper = round(cal, 4), pairs = length(pt), unmatched_A = sum(d$drugA) - length(pt))
m <- rbind(cbind(d[pt, ], pair = seq_along(pt)), cbind(d[pc, ], pair = seq_along(pt)))   # 매칭 자료
un <- d$drugA == 1 & !(seq_len(nrow(d)) %in% pt)
round(rbind(unmatched = colMeans(d[un, c("ps", "age", "hf", "ckd")]),
            matched   = colMeans(d[pt, c("ps", "age", "hf", "ckd")])), 3)
''', title="로짓 성향점수로 1:1 매칭하기")

c_matchit = nb.cell('''
suppressPackageStartupMessages(library(MatchIt))
set.seed(7)
mi <- matchit(drugA ~ age + female + hf + ckd + prior_hosp + n_drug + tertiary, data = d,
              method = "nearest", distance = "glm", link = "linear.logit",   # 로짓 성향점수
              caliper = 0.2, m.order = "random")                            # 표준편차의 0.2배
summary(mi)$nn
md <- match.data(mi)                     # subclass 열이 짝 번호
fit <- glm(hosp_1y ~ drugA, family = poisson, data = md)
rr_ci(fit, vcovCL(fit, cluster = ~ subclass, type = "HC0"))
''', title="MatchIt으로 같은 설계 매칭하기")

c_w = nb.cell('''
pA <- mean(d$drugA)
d$w_ate <- ifelse(d$drugA == 1, 1 / d$ps, 1 / (1 - d$ps))               # 역확률 가중치
d$w_st  <- ifelse(d$drugA == 1, pA / d$ps, (1 - pA) / (1 - d$ps))       # 안정화 가중치
d$w_att <- ifelse(d$drugA == 1, 1, d$ps / (1 - d$ps))                   # A 사용자 기준(ATT)
d$w_ow  <- ifelse(d$drugA == 1, 1 - d$ps, d$ps)                         # 겹침 가중치
ess <- function(w) sum(w)^2 / sum(w^2)                                  # 유효 표본 크기
round(rbind(ate = range(d$w_ate), stabilized = range(d$w_st),
            att_B = range(d$w_att[d$drugA == 0])), 3)
round(c(sum_A = sum(d$w_ate[d$drugA == 1]), sum_B = sum(d$w_ate[d$drugA == 0]),
        ess_A = ess(d$w_ate[d$drugA == 1]), ess_B = ess(d$w_ate[d$drugA == 0])), 1)
''', title="역확률 가중치와 그 밖의 가중치")

# ------------------------------------------------------------------ d. 균형과 효과
c_smd = nb.cell('''
smd <- function(x, g, w = rep(1, length(x))) {        # 가중 평균과 가중 분산으로 구한 SMD
  m1 <- weighted.mean(x[g == 1], w[g == 1]); m0 <- weighted.mean(x[g == 0], w[g == 0])
  v1 <- weighted.mean((x[g == 1] - m1)^2, w[g == 1]); v0 <- weighted.mean((x[g == 0] - m0)^2, w[g == 0])
  (m1 - m0) / sqrt((v1 + v0) / 2)
}
vars <- c("age", "female", "hf", "ckd", "prior_hosp", "n_drug", "tertiary", "frail")
round(sapply(vars, function(v) c(before  = smd(d[[v]], d$drugA),
                                 matched = smd(m[[v]], m$drugA),
                                 iptw    = smd(d[[v]], d$drugA, d$w_ate))), 3)
''', title="표준화 평균차로 균형 보기")

c_rr = nb.cell('''
fm <- glm(hosp_1y ~ drugA, family = poisson, data = m)
fw <- glm(hosp_1y ~ drugA, family = poisson, data = d, weights = w_ate)
ft <- glm(hosp_1y ~ drugA, family = poisson, data = d, weights = w_att)
fo <- glm(hosp_1y ~ drugA, family = poisson, data = d, weights = w_ow)
rbind(matched = rr_ci(fm, vcovCL(fm, cluster = ~ pair, type = "HC0")),   # 짝 번호로 묶은 강건 분산
      iptw    = rr_ci(fw, vcovHC(fw, type = "HC0")),
      att     = rr_ci(ft, vcovHC(ft, type = "HC0")),
      overlap = rr_ci(fo, vcovHC(fo, type = "HC0")),
      iptw_model_based = rr_ci(fw, vcov(fw)))                             # 가중치를 사람 수로 셈
gw <- glm(hosp_1y ~ drugA, family = gaussian, data = d, weights = w_ate)   # 위험차
round(coef(gw)["drugA"] + c(0, -1.96, 1.96) * sqrt(vcovHC(gw, type = "HC0")["drugA", "drugA"]), 4)
''', title="매칭과 가중 자료의 상대위험도와 위험차")

c_cox = nb.cell('''
hr <- function(fit) round(exp(c(HR = unname(coef(fit)["drugA"]), confint(fit)["drugA", ])), 3)
rbind(crude   = hr(coxph(Surv(hosp_day, hosp_1y) ~ drugA, data = d)),
      iptw    = hr(coxph(Surv(hosp_day, hosp_1y) ~ drugA, data = d, weights = w_ate)),  # 강건 분산이 기본
      matched = hr(coxph(Surv(hosp_day, hosp_1y) ~ drugA, data = m, cluster = pair)),
      strata  = hr(coxph(Surv(hosp_day, hosp_1y) ~ drugA + strata(pair), data = m)))    # 짝을 층으로
''', title="매칭과 가중 자료의 위험비")

c_boot = nb.cell('''
set.seed(15)
boot_lrr <- replicate(500, {                       # 성향점수 추정까지 다시 하는 부트스트랩
  b <- d[sample(nrow(d), replace = TRUE), ]
  p <- fitted(glm(drugA ~ age + female + hf + ckd + prior_hosp + n_drug + tertiary,
                  family = binomial, data = b))
  w <- ifelse(b$drugA == 1, 1 / p, 1 / (1 - p))
  log(weighted.mean(b$hosp_1y[b$drugA == 1], w[b$drugA == 1]) /
      weighted.mean(b$hosp_1y[b$drugA == 0], w[b$drugA == 0]))
})
round(exp(quantile(boot_lrr, c(0.025, 0.975))), 3)
round(c(boot_se = sd(boot_lrr), robust_se = sqrt(vcovHC(fw, type = "HC0")["drugA", "drugA"])), 4)
''', title="부트스트랩 신뢰구간")

# ------------------------------------------------------------------ e. 민감도 분석
c_nc = nb.cell('''
fnc <- glm(injury_ed ~ drugA, family = poisson, data = d)                     # 음성 대조 결과
fnm <- glm(injury_ed ~ drugA, family = poisson, data = m)
fnw <- glm(injury_ed ~ drugA, family = poisson, data = d, weights = w_ate)
rbind(crude   = rr_ci(fnc, vcovHC(fnc, type = "HC0")),
      matched = rr_ci(fnm, vcovCL(fnm, cluster = ~ pair, type = "HC0")),
      iptw    = rr_ci(fnw, vcovHC(fnw, type = "HC0")))
''', title="음성 대조 결과로 남은 교란 찾기")

c_ev = nb.cell('''
evalue <- function(rr) { rr <- ifelse(rr < 1, 1 / rr, rr); rr + sqrt(rr * (rr - 1)) }
ev_ci  <- function(lo, hi) if (lo <= 1 & hi >= 1) 1 else if (hi < 1) evalue(hi) else evalue(lo)   # 1에 가까운 끝
est <- function(fit, V) {                     # 반올림하지 않은 추정값과 신뢰한계
  b <- unname(coef(fit)["drugA"]); se <- sqrt(V["drugA", "drugA"])
  exp(c(b, b - 1.96 * se, b + 1.96 * se))
}
f0 <- glm(hosp_1y ~ drugA, family = poisson, data = d)
e  <- rbind(crude   = est(f0, vcovHC(f0, type = "HC0")),
            matched = est(fm, vcovCL(fm, cluster = ~ pair, type = "HC0")),
            iptw    = est(fw, vcovHC(fw, type = "HC0")))
round(cbind(E_point = evalue(e[, 1]), E_limit = apply(e, 1, function(x) ev_ci(x[2], x[3]))), 3)
''', title="E-value 구하기")

# ------------------------------------------------------------------ f. 과제
c_t1 = nb.cell('''
round(c(point = evalue(0.60), limit = evalue(0.64)), 2)   # HR 0.60 (95% CI 0.53–0.64)
''', title="과제 1 정답: 논문 결과의 E-value")

c_t2 = nb.cell('''
q <- quantile(d$w_st, c(0.01, 0.99))
d$w_tr <- pmin(pmax(d$w_st, q[1]), q[2])             # 1·99번째 백분위수에서 자름
round(q, 3)
ftr <- glm(hosp_1y ~ drugA, family = poisson, data = d, weights = w_tr)
rr_ci(ftr, vcovHC(ftr, type = "HC0"))
''', title="과제 2 정답: 가중치 자르기")

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
    c_adj: ["1.328 1.191 1.481", "1.155 1.035 1.288", "1.035 0.926 1.158", "0.933 0.833 1.046", "0.800 0.714 0.897"],
    c_ps: ["0.02648388", "[1] 10000", "0.1296727 0.1492039", "0.8956509 0.9324756", "0.6575961", "0.8809 0.2526"],
    c_match: ["0.1232   3516.0000    537.0000", "0.687 75.747 0.527", "0.418 66.650 0.090"],
    c_matchit: ["Matched          3521    3521"],
    c_w: ["1.072 9.583", "0.435 5.699", "0.149 8.583", "10040.3  9974.6  3597.5  5470.6"],
    c_smd: ["0.346 -0.042  0.292  0.241      0.203  0.331    0.229 0.401", "0.004  0.006 -0.010 -0.010     -0.003  0.005    0.018 0.341"],
    c_rr: ["0.908 0.797 1.033", "0.953 0.848 1.072", "0.972 0.861 1.097", "0.942 0.839 1.058", "0.953 0.878 1.035", "-0.0054 -0.0186  0.0078"],
    c_cox: ["1.354 1.206  1.521", "0.952 0.840  1.078", "0.903 0.787  1.036", "0.897 0.778  1.035"],
    c_boot: ["0.0598"],
    c_nc: ["1.467 1.268 1.697", "1.289 1.086 1.530", "1.262 1.081 1.473"],
    c_ev: ["1.988   1.667", "1.437   1.000", "1.275   1.000"],
    c_t1: ["2.72  2.50"],
    c_t2: ["0.508 2.096", "0.971 0.865 1.090"],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc18_adj", nb.html(c_adj, marks={"1.328 1.191 1.481": 1, "0.933 0.833 1.046": 2, "0.800 0.714 0.897": 3}))
S("rc18_ps", nb.html(c_ps, marks={"0.02648388": 1, "[1] 10000": 2, "0.8956509 0.9324756": 3, "0.6575961": 4, "0.8809 0.2526": 5}))
S("rc18_match", nb.html(c_match, marks={"0.1232   3516.0000    537.0000": 1, "0.687 75.747 0.527": 2}))
S("rc18_matchit", nb.html(c_matchit, marks={"Matched          3521    3521": 1, "0.926 0.815 1.053": 2}))
S("rc18_w", nb.html(c_w, marks={"1.072 9.583": 1, "0.435 5.699": 2, "10040.3  9974.6  3597.5  5470.6": 3}))
S("rc18_smd", nb.html(c_smd, marks={"0.346 -0.042  0.292  0.241      0.203  0.331    0.229 0.401": 1, "0.004  0.006 -0.010 -0.010     -0.003  0.005    0.018 0.341": 2}))
S("rc18_rr", nb.html(c_rr, marks={"0.908 0.797 1.033": 1, "0.953 0.848 1.072": 2, "0.953 0.878 1.035": 3, "-0.0054 -0.0186  0.0078": 4}))
S("rc18_cox", nb.html(c_cox, marks={"0.952 0.840  1.078": 1, "0.897 0.778  1.035": 2}))
S("rc18_boot", nb.html(c_boot, marks={"0.0598": 1}))
S("rc18_nc", nb.html(c_nc, marks={"1.467 1.268 1.697": 1, "1.262 1.081 1.473": 2}))
S("rc18_ev", nb.html(c_ev, marks={"1.988   1.667": 1, "1.437   1.000": 2}))
S("rc18_t1", nb.html(c_t1, marks={"2.72  2.50": 1}))
S("rc18_t2", nb.html(c_t2, marks={"0.971 0.865 1.090": 1}))
nb.save_ipynb("11장 성향점수 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 15장의 가상 코호트 10,000명으로 보정할 변수 고르기, 성향점수 추정과 겹침, 1:1 매칭과 가중, "
                    "표준화 평균차로 본 균형, 상대위험도·위험차·위험비와 강건 분산, 부트스트랩, 음성 대조 결과와 E-value를 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"sandwich\", \"MatchIt\"))를 실행합니다.",
              notes={c_adj.n: "## a. 보정할 변수 고르기", c_ps.n: "## b. 성향점수의 추정과 겹침", c_match.n: "## c. 매칭과 가중",
                     c_smd.n: "## d. 균형 진단과 효과 추정", c_nc.n: "## e. 남는 교란과 민감도 분석", c_t1.n: "## f. 과제 정답"})
print("rc18 fragments and notebook written")
