"""R과 파이썬으로 하는 통계분석 11장 · 요약통계, 분포, 검정의 기본 — R cells run for real with rkit.

run:  python3 gen/r_rc11.py      (NOMARK=1 python3 gen/r_rc11.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 0–2장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-18 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 0장의 처방 표와 A 병동 5명, 0장 혈압 시험의 요약값(12.4 ± 7.0, 30명 대 8.1 ± 7.2, 30명), 1장의 2 × 2 표(12/50 대 18/50),
1장 c 절의 p 다섯 개, 1장 h 절의 60명 자료(explore60.csv, gen/fig_ch01.py와 같은 난수로 만든 수축기혈압·재원일수),
1장 g 절의 7명씩 재원일수, 2장의 복약상담 요약값(−0.9 ± 0.8, 36명 대 −0.4 ± 0.9, 36명)과 예비연구(40명 대 15명),
2장 b 절의 오피오이드 사용량(8명 대 7명)과 만족도 리커트 점수(20명씩).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc11")
URL = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# a. 환자 단위로 모으고 요약하기 (0장 a·c 절)
c_rx = nb.cell('''
library(dplyr, warn.conflicts = FALSE)
rx <- data.frame(
  patient = c("A", "A", "A", "B", "C", "C"),
  date    = as.Date(c("2024-01-03", "2024-02-01", "2024-03-04",
                      "2024-01-10", "2024-01-15", "2024-03-20")),
  days    = c(30, 30, 30, 90, 30, 30))       # 공급일수
rx |>
  group_by(patient) |>
  summarise(first_date = min(date), n_rx = n(), total_days = sum(days))
''', title="처방 단위 자료를 환자 단위로 모으기")

c_sd = nb.cell('''
x <- c(8, 12, 9, 11, 10)                  # A 병동 5명의 재원일수(일)
c(mean = mean(x), var = var(x), sd = sd(x))      # var와 sd는 n − 1로 나눔
sqrt(sum((x - mean(x))^2) / length(x))           # n으로 나누면
''', title="평균, 분산, 표준편차")

c_na = nb.cell('''
x_na <- c(8, 12, NA, 11, 10)              # 셋째 환자의 값이 빔
mean(x_na)                                # NA가 하나라도 있으면 NA
c(mean = mean(x_na, na.rm = TRUE), sd = sd(x_na, na.rm = TRUE),
  n = sum(!is.na(x_na)))
cost <- c(0, 0, 12, 18, 20)               # 1인당 비용(만 원)
c(all = mean(cost), drop_zero = mean(cost[cost > 0]))
''', title="값이 빈 환자가 있으면")

# b. 분포에서 확률, 경계값, p 구하기 (0장 d·e 절, 1장 e 절, 2장 a 절)
c_norm = nb.cell('''
pnorm(-1.5)                        # P(Z < −1.5)
pnorm(2.0, lower.tail = FALSE)     # P(Z > 2.0)
qnorm(0.975)                       # 위쪽 2.5%를 자르는 z
''', title="정규분포의 확률과 경계값")

c_se = nb.cell('''
se1 <- 7.0 / sqrt(30)              # 신약군 평균의 SE
se2 <- 7.2 / sqrt(30)              # 위약군 평균의 SE
se_diff <- sqrt(se1^2 + se2^2)     # 차이의 SE
round(c(se1, se2, se_diff), 2)
round(qt(0.975, df = c(5, 29, 58, 1000)), 2)   # 자유도별 t 경계값
''', title="표준오차와 t 경계값")

c_p = nb.cell('''
qt(0.975, 58)                          # t(58)의 양측 5% 임계값
2 * pt(2.35, 58, lower.tail = FALSE)   # t(58) = 2.35의 양측 p
2 * pnorm(1.96, lower.tail = FALSE)    # z = 1.96의 양측 p
pnorm(0.25)                            # Φ(0.25): 검정력 계산
pchisq(5.31, 1, lower.tail = FALSE)    # χ²(1) = 5.31의 p
pf(4.12, 2, 87, lower.tail = FALSE)    # F(2, 87) = 4.12의 p
''', title="통계량에서 p, 확률에서 임계값")

# c. 요약값으로 하는 t 검정 (0장 f 절, 2장 a 절)
c_tfun = nb.cell('''
t_from_stats <- function(m1, s1, n1, m2, s2, n2, var.equal = FALSE) {
  if (var.equal) {                       # Student: 합동분산
    sp2 <- ((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2)
    se <- sqrt(sp2 * (1 / n1 + 1 / n2)); df <- n1 + n2 - 2
  } else {                               # Welch: 분산을 군마다 따로
    v1 <- s1^2 / n1; v2 <- s2^2 / n2
    se <- sqrt(v1 + v2); df <- (v1 + v2)^2 / (v1^2 / (n1 - 1) + v2^2 / (n2 - 1))
  }
  d <- m1 - m2; t <- d / se; crit <- qt(0.975, df)
  round(c(diff = d, se = se, t = t, df = df, p = 2 * pt(-abs(t), df),
          lower = d - crit * se, upper = d + crit * se), 4)
}
t_from_stats(12.4, 7.0, 30, 8.1, 7.2, 30, var.equal = TRUE)   # 0장의 혈압 시험
''', title="평균, SD, 인원만으로 t 검정")

c_welch = nb.cell('''
t_from_stats(-0.9, 0.8, 36, -0.4, 0.9, 36)                     # 2장 복약상담, Welch
rbind(Student = t_from_stats(-0.90, 0.60, 40, -0.40, 1.20, 15, var.equal = TRUE),
      Welch   = t_from_stats(-0.90, 0.60, 40, -0.40, 1.20, 15))   # 2장 예비연구
''', title="Welch와 Student가 다른 답을 낼 때")

# d. 같은 표의 χ²와 z, 회귀 출력의 z와 Wald χ² (1장 e 절)
c_chi = nb.cell('''
tab <- matrix(c(12, 38,
                18, 32), nrow = 2, byrow = TRUE,
              dimnames = list(group = c("treat", "control"), event = c("yes", "no")))
chisq.test(tab)$statistic                    # 기본값: Yates 보정
chisq.test(tab, correct = FALSE)$p.value
pt2 <- prop.test(c(12, 18), c(50, 50), correct = FALSE)
c(X2 = unname(pt2$statistic), z = -sqrt(unname(pt2$statistic)), p = pt2$p.value)
''', title="같은 2 × 2 표의 χ²와 z")

c_wald = nb.cell('''
b <- 0.6420; se <- 0.2110          # 로지스틱 회귀 출력의 계수와 표준오차
z <- b / se
round(c(z = z, wald = z^2), 3)
round(c(p_z = 2 * pnorm(-abs(z)), p_chisq = pchisq(z^2, 1, lower.tail = FALSE)), 4)
round(exp(c(OR = b, lower = b - 1.96 * se, upper = b + 1.96 * se)), 2)
''', title="z, Wald χ², 오즈비를 계수와 표준오차로")

# e. 다중비교 보정과 부트스트랩 (1장 c·f 절)
c_padj = nb.cell('''
p <- c(0.004, 0.012, 0.021, 0.038, 0.260)
rbind(holm = p.adjust(p),                    # 기본값이 Holm
      BH = p.adjust(p, method = "BH"),
      bonferroni = p.adjust(p, method = "bonferroni"))
''', title="다중비교 보정")

c_boot = nb.cell(f'''
d <- read.csv("{URL}explore60.csv")       # 1장 h 절의 60명
set.seed(2026)
m <- replicate(2000, mean(sample(d$los, replace = TRUE)))
round(quantile(m, c(0.025, 0.975)), 2)      # 백분위수 부트스트랩 95% CI
library(boot)
bt <- boot(d$los, function(x, i) mean(x[i]), R = 2000)
boot.ci(bt, type = c("perc", "bca"))
''', title="평균 재원일수의 부트스트랩 신뢰구간")

# f. 자료 탐색과 정규성 (1장 h 절)
c_quant = nb.cell('''
quantile(d$los, c(0.25, 0.75))                 # 기본값 type = 7
quantile(d$los, c(0.25, 0.75), type = 6)       # SPSS의 기본값과 같음
library(e1071)
round(c(type1 = skewness(d$los, type = 1),      # 표본 그대로(scipy 기본값)
        type2 = skewness(d$los, type = 2),      # 표본 크기 보정(SPSS, pandas)
        type3 = skewness(d$los)), 2)            # e1071의 기본값은 type 3
''', title="분위수와 왜도는 정의에 따라 다르다")

c_qq = nb.cell('''
par(mfrow = c(1, 2))
qqnorm(d$sbp, main = "Systolic BP"); qqline(d$sbp)
qqnorm(d$los, main = "Length of stay"); qqline(d$los)
''', title="Q-Q plot", fig=(8, 4))

c_normtest = nb.cell('''
shapiro.test(d$los)$p.value
ks.test(d$sbp, "pnorm")$p.value            # 표준정규분포와 비교해 버림
ks.test(as.numeric(scale(d$los)), "pnorm")$p.value   # 같은 자료의 평균·SD로 표준화
library(nortest)
lillie.test(d$los)$p.value                 # 이 점을 보정한 Lilliefors 검정
''', title="정규성 검정을 쓸 때의 함정")

# g. 순위합 검정의 세부 (1장 g 절, 2장 b 절)
c_mw7 = nb.cell('''
A <- c(1, 2, 3, 3, 4, 5, 47); B <- 6:12       # 1장 g 절의 재원일수 7명씩
wilcox.test(A, B, exact = FALSE)$p.value                    # 정규근사(동점 보정, 연속성 보정)
wilcox.test(A, B, exact = FALSE, correct = FALSE)$p.value   # 연속성 보정 없이
suppressPackageStartupMessages(library(coin))
g <- factor(rep(c("A", "B"), each = 7))
pvalue(wilcox_test(c(A, B) ~ g, distribution = "exact"))   # 중간순위 그대로 정확 p
''', title="같은 자료에서 p를 구하는 세 방법")

c_opioid = nb.cell('''
protocol <- c(12, 15, 18, 20, 22, 26, 31, 68)   # 프로토콜군 8명
usual    <- c(24, 29, 36, 41, 45, 58, 115)      # 일반 관리군 7명
wilcox.test(protocol, usual, conf.int = TRUE)
''', title="오피오이드 자료의 Wilcoxon 순위합 검정과 Hodges–Lehmann 추정치")

c_likert = nb.cell('''
counsel <- rep(1:5, times = c(0, 2, 5, 8, 5))   # 상담군 만족도 20명
control <- rep(1:5, times = c(2, 5, 7, 4, 2))   # 대조군 20명
wilcox.test(counsel, control, exact = FALSE)$p.value     # 동점 보정 + 연속성 보정
wilcox.test(counsel, control, exact = FALSE, correct = FALSE)$p.value
set.seed(2026)
g2 <- factor(rep(c("counsel", "control"), each = 20))
pvalue(wilcox_test(c(counsel, control) ~ g2,
                   distribution = approximate(nresample = 10000)))
''', title="동점이 많은 리커트 점수")

c_bm = nb.cell('''
library(brunnermunzel)
brunnermunzel.test(protocol, usual)
''', title="퍼진 정도가 다를 때: Brunner–Munzel 검정")

# h. 과제 정답
c_t1 = nb.cell('''
y <- c(132, NA, 128, 141, NA, 137, 125)
c(n_used = sum(!is.na(y)), mean = mean(y, na.rm = TRUE), sd = round(sd(y, na.rm = TRUE), 2))
''', title="과제 1 정답: 결측이 있는 열의 요약")

c_t2 = nb.cell('''
round(c(p_raw = shapiro.test(d$los)$p.value, p_log = shapiro.test(log(d$los))$p.value), 4)
round(exp(mean(log(d$los))), 2)            # 기하평균
''', title="과제 2 정답: 로그 변환한 재원일수")

c_t3 = nb.cell('''
t_from_stats(-0.90, 1.20, 40, -0.40, 0.60, 15, var.equal = TRUE)["p"]
t_from_stats(-0.90, 1.20, 40, -0.40, 0.60, 15)["p"]
''', title="과제 3 정답: 큰 군의 SD가 클 때")

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
# 보건통계학 기초 0–2장의 숫자와 대조
assert "1.581139" in out[c_sd.n] and "1.414214" in out[c_sd.n]
assert "10.250000  1.707825" in out[c_na.n] and "16.66667" in out[c_na.n]
assert "0.0668072" in out[c_norm.n] and "1.959964" in out[c_norm.n]
assert "1.28 1.31 1.83" in out[c_se.n] and "2.57 2.05 2.00 1.96" in out[c_se.n]
assert "0.02219754" in out[c_p.n] and "0.0195184" in out[c_p.n]
assert "2.3454 58.0000  0.0224  0.6301  7.9699" in out[c_tfun.n]
assert "-2.4914" in out[c_welch.n] and "0.0151" in out[c_welch.n] and "0.0447" in out[c_welch.n] and "16.6954 0.1416" in out[c_welch.n]
assert "1.190476" in out[c_chi.n] and "1.7142857" in out[c_chi.n]
assert "3.043 9.258" in out[c_wald.n] and "1.90  1.26  2.87" in out[c_wald.n]
assert "0.048 0.063 0.0760" in out[c_padj.n] and "0.0475" in out[c_padj.n]
assert "4.75 9.25" in out[c_quant.n] and "4.25 9.75" in out[c_quant.n] and "2.32  2.38" in out[c_quant.n]
assert "0.02966226" in out[c_mw7.n] and "0.02518656" in out[c_mw7.n] and "0.02447552" in out[c_mw7.n]
assert "W = 9, p-value = 0.0289" in out[c_opioid.n] and "-18.5" in out[c_opioid.n] and "-40  -3" in out[c_opioid.n]
assert "0.01882743" in out[c_likert.n] and "0.01813197" in out[c_likert.n]
assert "2.9452" in out[c_bm.n]

S = nb.save_fragment
S("rc11_rx", nb.html(c_rx, marks={"2024-01-03     3         90": 1}))
S("rc11_sd", nb.html(c_sd, marks={"1.581139": 1, "[1] 1.414214": 2}))
S("rc11_na", nb.html(c_na, marks={"[1] NA": 1, "4.000000": 2, "16.66667": 3}))
S("rc11_norm", nb.html(c_norm, marks={"0.0668072": 1, "0.02275013": 2, "1.959964": 3}))
S("rc11_se", nb.html(c_se, marks={"1.28 1.31 1.83": 1, "2.57 2.05 2.00 1.96": 2}))
S("rc11_p", nb.html(c_p, marks={"2.001717": 1, "0.02219754": 2, "0.04999579": 3, "0.5987063": 4, "0.02120336": 5, "0.0195184": 6}))
S("rc11_tfun", nb.html(c_tfun, marks={"2.3454 58.0000  0.0224  0.6301  7.9699": 1}))
S("rc11_welch", nb.html(c_welch, marks={"69.0509  0.0151": 1, "53.0000 0.0447": 2, "16.6954 0.1416": 3}))
S("rc11_chi", nb.html(c_chi, marks={"1.190476": 1, "[1] 0.1904303": 2, "1.7142857 -1.3093073": 3}))
S("rc11_wald", nb.html(c_wald, marks={"3.043 9.258": 1, "0.0023  0.0023": 2, "1.90  1.26  2.87": 3}))
S("rc11_padj", nb.html(c_padj, marks={"holm       0.02 0.048": 1, "BH         0.02 0.030": 2, "bonferroni 0.02": 3}))
S("rc11_boot", nb.html(c_boot, marks={"6.42  8.92": 1, "( 6.450,  8.850 )": 2, "( 6.550,  9.185 )": 3}))
S("rc11_quant", nb.html(c_quant, marks={"4.75 9.25": 1, "4.25 9.75": 2, "2.32  2.38  2.27": 3}))
S("rc11_qq", nb.html(c_qq))
S("rc11_normtest", nb.html(c_normtest, marks={"1.330931e-07": 1, "[1] 0\n": 2, "0.04344524": 3, "5.674424e-05": 4}))
S("rc11_mw7", nb.html(c_mw7, marks={"0.02966226": 1, "0.02518656": 2, "0.02447552": 3}))
S("rc11_opioid", nb.html(c_opioid, marks={"W = 9, p-value = 0.0289": 1, "-40  -3": 2, "-18.5": 3}))
S("rc11_likert", nb.html(c_likert, marks={"0.01882743": 1, "0.01813197": 2, "[1] 0.0185": 3}))
S("rc11_bm", nb.html(c_bm, marks={"2.9452": 1, "0.8392857": 2}))
S("rc11_t1", nb.html(c_t1))
S("rc11_t2", nb.html(c_t2, marks={"0.5451": 1}))
S("rc11_t3", nb.html(c_t3, marks={"0.1303": 1, "0.0467": 2}))
nb.save_ipynb("11장 요약통계, 분포, 검정의 기본 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 0–2장의 예로 요약통계와 결측, 정규분포·t 분포의 확률과 경계값, 요약값만으로 하는 t 검정, "
                    "같은 표의 χ²와 z, 다중비교 보정, 부트스트랩, 분위수·왜도·정규성 확인, 순위합 검정의 세부를 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"dplyr\", \"boot\", \"e1071\", \"nortest\", \"coin\", \"brunnermunzel\"))를 실행합니다.",
              notes={c_rx.n: "## a. 환자 단위로 모으고 요약하기", c_norm.n: "## b. 분포에서 확률과 경계값 구하기",
                     c_tfun.n: "## c. 요약값으로 하는 t 검정", c_chi.n: "## d. 같은 표의 χ²와 z, 회귀 출력의 z",
                     c_padj.n: "## e. 다중비교 보정과 부트스트랩", c_quant.n: "## f. 분위수, 왜도, 정규성",
                     c_mw7.n: "## g. 순위합 검정의 세부", c_t1.n: "## h. 과제 정답"})
print("rc11 fragments and notebook written")
