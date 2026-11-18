"""R과 파이썬으로 하는 통계분석 7장 · 동등성과 비열등성 검정 — R cells run for real with rkit.

run:  python3 gen/r_rc16.py      (NOMARK=1 python3 gen/r_rc16.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 13장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-18 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 원격 상담 비열등성 시험의 환자별 PDC(pub/data/ch13_pdc.csv; gen/data_rc16.py, gen/pyout_ch13.py와 같은 seed 1302),
13장의 요약값(PDC PP, HbA1c, 제균 시험), 12명 AUC(13장 e 절), 2×2 교차설계 24명(pub/data/ch13_be24.csv).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc16")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 평균 차이
c_pdc = nb.cell(f'''
d <- read.csv("{D}ch13_pdc.csv")
remote   <- d$pdc[d$arm == "remote"]          # 원격 상담군 304명
inperson <- d$pdc[d$arm == "inperson"]        # 대면 상담군 304명
t.test(remote, inperson, mu = -5, alternative = "greater", conf.level = 0.975)
t.test(remote, inperson)$conf.int                                    # 양측 95% 구간
t.test(remote, inperson, mu = -5, alternative = "greater")$conf.int  # conf.level을 빼면
t.test(inperson, remote, mu = -5, alternative = "greater")$p.value   # 순서를 바꾸면
''', title="원격 상담 시험의 비열등성 검정 (13장 b 절)")

c_stats = nb.cell('''
ni_t <- function(m1, s1, n1, m2, s2, n2, margin, higher = TRUE) {   # 요약값으로 하는 Welch 비열등성 검정
  v1 <- s1^2 / n1; v2 <- s2^2 / n2; se <- sqrt(v1 + v2)
  df <- (v1 + v2)^2 / (v1^2 / (n1 - 1) + v2^2 / (n2 - 1))
  d  <- m1 - m2; q <- qt(0.975, df)
  t  <- if (higher) (d + margin) / se else (margin - d) / se     # 한계 쪽 끝을 검정
  round(c(diff = d, lower = d - q * se, upper = d + q * se, t = t, df = df,
          p_NI = pt(t, df, lower.tail = FALSE)), 5)
}
rbind(PDC_ITT  = ni_t(83.9, 16.8, 304, 85.2, 17.5, 304, margin = 5),
      PDC_PP   = ni_t(85.0, 15.9, 262, 85.6, 17.1, 275, margin = 5),
      PDC_ex   = ni_t(85.0, 18.0, 250, 87.5, 17.0, 250, margin = 5),
      HbA1c_FAS = ni_t(-0.74, 1.02, 185, -0.83, 1.06, 183, margin = 0.4, higher = FALSE),
      HbA1c_PP  = ni_t(-0.79, 1.00, 162, -0.85, 1.04, 165, margin = 0.4, higher = FALSE),
      SBP_ex    = ni_t(135.6, 10.6, 150, 132.2, 10.6, 150, margin = 5, higher = FALSE))
''', title="요약값으로 하는 비열등성 검정 (13장 b·c 절)")

# ------------------------------------------------------------------ b. 비율 차이
c_prop = nb.cell('''
suppressPackageStartupMessages(library(DescTools))
BinomDiffCI(186, 224, 190, 224, method = c("wald", "score", "mn"))   # ITT: 신약 186/224, 표준 190/224
BinomDiffCI(180, 205, 185, 208, method = c("wald", "score", "mn"))   # PP
''', title="제균율 차이의 세 가지 신뢰구간 (13장 d 절)")

c_fm = nb.cell('''
fm_test <- function(x1, n1, x2, n2, delta) {   # Farrington–Manning 검정, 귀무가설: p1 - p2 = delta
  nll <- function(p2) -(dbinom(x1, n1, p2 + delta, log = TRUE) + dbinom(x2, n2, p2, log = TRUE))
  p2 <- optimize(nll, c(max(0, -delta), min(1, 1 - delta)) + c(1e-9, -1e-9), tol = 1e-12)$minimum
  p1 <- p2 + delta                               # 귀무가설 아래에서 가장 그럴듯한 두 비율
  z  <- (x1 / n1 - x2 / n2 - delta) / sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
  round(c(p1_null = p1, p2_null = p2, z = z, p = pnorm(z, lower.tail = FALSE)), 5)
}
rbind(ITT = fm_test(186, 224, 190, 224, delta = -0.10),
      PP  = fm_test(180, 205, 185, 208, delta = -0.10))
''', title="Farrington–Manning 비열등성 검정")

c_small = nb.cell('''
BinomDiffCI(48, 50, 50, 50, method = c("wald", "score", "mn"))     # 소규모 시험 (13장 '결과는 어떻게 읽나?')
BinomDiffCI(94, 100, 98, 100, method = c("wald", "score", "mn"))   # 군당 100명
fm_test(48, 50, 50, 50, delta = -0.10)
''', title="성공률이 100%에 가까울 때")

# ------------------------------------------------------------------ c. 동등성
c_be12 = nb.cell('''
R <- c(412, 538, 365, 620, 488, 297, 554, 431, 702, 389, 515, 460)   # 대조약 AUC (13장 e 절의 12명)
T <- c(366, 455, 303, 548, 556, 268, 535, 510, 639, 429, 412, 525)   # 시험약 AUC
lo <- t.test(log(T), log(R), paired = TRUE, mu = log(0.8),  alternative = "greater")
up <- t.test(log(T), log(R), paired = TRUE, mu = log(1.25), alternative = "less")
round(c(t_L = unname(lo$statistic), p_L = lo$p.value, t_U = unname(up$statistic), p_U = up$p.value), 6)
exp(t.test(log(T), log(R), paired = TRUE, conf.level = 0.90)$conf.int)   # 기하평균비의 90% 구간
t.test(log(T), log(R), paired = TRUE)$p.value                           # 차이 = 0의 검정
c(t.test(log(T), log(R), var.equal = TRUE, mu = log(0.8),  alternative = "greater")$p.value,
  t.test(log(T), log(R), var.equal = TRUE, mu = log(1.25), alternative = "less")$p.value)   # 짝을 무시하면
''', title="12명 AUC의 두 단측 검정과 90% 신뢰구간")

c_cross = nb.cell(f'''
be <- read.csv("{D}ch13_be24.csv")              # 대상자 한 명의 한 기간이 한 줄
be$subject     <- factor(be$subject)
be$period      <- factor(be$period)
be$formulation <- factor(be$formulation, levels = c("R", "T"))   # 대조약이 기준
head(be, 4)
fit <- lm(log(auc) ~ subject + period + formulation, data = be)
ci <- confint(fit, "formulationT", level = 0.90)           # 90% 신뢰구간 (로그 척도)
round(100 * exp(c(GMR = unname(coef(fit)["formulationT"]), lower = ci[1], upper = ci[2])), 2)
round(100 * sqrt(exp(sigma(fit)^2) - 1), 1)              # 개체내 변동계수(%)
''', title="2×2 교차설계 분석 (13장 e 절 Table 2)")

# ------------------------------------------------------------------ d. 표본크기
c_ssm = nb.cell('''
z <- qnorm(0.975) + qnorm(0.90)                       # 단측 α 0.025, 검정력 90%
c(HbA1c = 2 * 1.1^2 * z^2 / 0.4^2, PDC = 2 * 18^2 * z^2 / 5^2)   # 정규분포 공식(군당)
power.t.test(delta = 0.4, sd = 1.1, sig.level = 0.025, power = 0.9, alternative = "one.sided")$n
power.t.test(delta = 5,   sd = 18,  sig.level = 0.025, power = 0.9, alternative = "one.sided")$n
ceiling(159 / 0.85)                                   # 탈락 15%를 감안한 인원
''', title="평균 차이의 비열등성 표본크기")

c_ssp = nb.cell('''
z <- qnorm(0.975) + qnorm(0.80)                       # 단측 α 0.025, 검정력 80%
n <- z^2 * 2 * 0.85 * 0.15 / 0.10^2                   # 두 군 모두 85%, 한계 −10%p
c(n = n, enrolled = ceiling(ceiling(n) / 0.9))
pw <- function(p, n = 201, margin = 0.10) {          # 실제 차이가 0일 때의 검정력
  se <- sqrt(2 * p * (1 - p) / n)
  pnorm((margin - qnorm(0.975) * se) / se)
}
round(sapply(c(0.70, 0.85, 0.95), pw), 3)
''', title="비율 차이의 비열등성 표본크기와 검정력")

# ------------------------------------------------------------------ e. 과제
c_t1 = nb.cell('''
ni_t(83.9, 16.8, 304, 85.2, 17.5, 304, margin = 4)        # 한계 4%p
power.t.test(delta = 4, sd = 18, sig.level = 0.025, power = 0.9, alternative = "one.sided")$n
''', title="과제 1 정답: 한계를 4%p로 좁히면")

c_t2 = nb.cell('''
fc <- lm(log(cmax) ~ subject + period + formulation, data = be)
ci <- confint(fc, "formulationT", level = 0.90)
round(100 * exp(c(GMR = unname(coef(fc)["formulationT"]), lower = ci[1], upper = ci[2])), 2)
round(100 * sqrt(exp(sigma(fc)^2) - 1), 1)
''', title="과제 2 정답: Cmax의 동등성")

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
    c_pdc: ["t = 2.6593, df = 604.99, p-value = 0.004019", "-4.03244      Inf", "-4.032440  1.432439", "-3.592058", "3.586489e-06"],
    c_stats: ["-1.30 -4.03244 1.43244 2.65931", "-0.60 -3.39764 2.19764 3.08954", "-2.50 -5.57658 0.57658 1.59654 496.38", "0.05550",
              "0.09 -0.12329 0.30329 2.85818", "0.00225", "0.00139", "0.09607"],
    c_prop: ["-0.08585463 0.05014034", "-0.08620709 0.05056550", "-0.08669073 0.05079652", "-0.07437873"],
    c_fm: ["0.77954 0.87954 2.33251 0.00984", "2.69370 0.00353"],
    c_small: ["-0.09431612 0.01431612", "-0.13460091", "-0.10637098", "1.41421"],
    c_be12: ["4.482221", "0.000464", "0.000018", "0.8909597 1.0289384", "0.3014489", "0.046501879"],
    c_cross: ["97.31", "91.20", "103.82", "13.1"],
    c_ssm: ["158.9248", "272.3524", "159.8913", "273.3163", "188"],
    c_ssp: ["200.1", "224", "0.590 0.802 0.996"],
    c_t1: ["1.94", "426.5"],
    c_t2: ["93.93", "84.42", "104.52", "21.8"],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc16_pdc", nb.html(c_pdc, marks={"t = 2.6593, df = 604.99, p-value = 0.004019": 1, "-4.03244      Inf": 2, "-4.032440  1.432439": 3,
                                    "-3.592058": 4, "3.586489e-06": 5}))
S("rc16_stats", nb.html(c_stats, marks={"-1.30 -4.03244 1.43244 2.65931": 1, "0.05550": 2, "0.09 -0.12329 0.30329 2.85818": 3, "0.09607": 4}))
S("rc16_prop", nb.html(c_prop, marks={"-0.08585463 0.05014034": 1, "-0.08620709 0.05056550": 2, "-0.08669073 0.05079652": 3}))
S("rc16_fm", nb.html(c_fm, marks={"0.77954 0.87954 2.33251 0.00984": 1, "2.69370 0.00353": 2}))
S("rc16_small", nb.html(c_small, marks={"-0.09431612 0.01431612": 1, "-0.13460091": 2, "-0.10637098": 3, "1.41421": 4}))
S("rc16_be12", nb.html(c_be12, marks={"4.482221": 1, "0.8909597 1.0289384": 2, "0.3014489": 3, "0.046501879": 4}))
S("rc16_cross", nb.html(c_cross, marks={"97.31": 1, "13.1": 2}))
S("rc16_ssm", nb.html(c_ssm, marks={"158.9248": 1, "159.8913": 2, "188": 3}))
S("rc16_ssp", nb.html(c_ssp, marks={"200.1": 1, "0.590 0.802 0.996": 2}))
S("rc16_t1", nb.html(c_t1, marks={"1.94": 1, "426.5": 2}))
S("rc16_t2", nb.html(c_t2, marks={"93.93": 1}))
nb.save_ipynb("7장 동등성과 비열등성 검정 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 13장의 예로 평균 차이와 비율 차이의 비열등성 검정, 두 단측 검정과 생물학적 동등성, "
                    "교차설계 분석, 비열등성 시험의 표본크기를 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(\"DescTools\")를 실행합니다.",
              notes={c_pdc.n: "## a. 평균 차이의 비열등성", c_prop.n: "## b. 비율 차이의 비열등성",
                     c_be12.n: "## c. 동등성과 생물학적 동등성", c_ssm.n: "## d. 표본크기와 검정력", c_t1.n: "## e. 과제 정답"})
print("rc16 fragments and notebook written")
