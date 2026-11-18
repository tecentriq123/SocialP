"""R과 파이썬으로 하는 통계분석 3장 · 짝지은 자료, 세 군 이상, 비율의 비교 — R cells run for real with rkit.

run:  python3 gen/r_rc12.py      (NOMARK=1 python3 gen/r_rc12.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 3–5장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-18 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 3장 a 절의 교육 전후 수축기혈압 12명, 3장 b 절의 약물 정리 전후 복용 약물 수 14명,
4장의 6개월 PDC 90명(pub/data/ch04_pdc.csv, gen/nums_ch04.py의 running_data와 knowledge_scores),
4장 d 절의 군당 5명 예비 자료, 5장 a 절의 약물 A·B 입원 표(48/400 대 102/600), 저혈당 3 × 2 표,
상담 전후 복약순응 짝지은 표, 5장 b 절의 간독성 표(1/15 대 6/14), 5장 c 절의 PDC 범주별 입원 표.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc12")
URL = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# a. 짝지은 두 값 (3장 a 절)
c_pt = nb.cell('''
before <- c(131, 146, 144, 154, 164, 159, 155, 164, 149, 161, 155, 142)   # 교육 전
after  <- c(126, 144, 131, 147, 162, 150, 154, 147, 142, 154, 139, 144)   # 교육 3개월 뒤
t.test(after, before, paired = TRUE)          # 후 − 전
''', title="대응표본 t 검정")

c_ptwrong = nb.cell('''
d <- after - before
c(mean = mean(d), sd = sd(d), n = length(d))
t.test(after, before, var.equal = TRUE)$p.value   # 잘못된 분석: 두 독립 군처럼
''', title="짝을 무시하면")

c_long = nb.cell('''
library(tidyr)
long <- data.frame(id = c(1, 1, 2, 2, 3),
                   time = c("base", "m3", "base", "m3", "base"),
                   sbp = c(131, 126, 146, 144, 144))      # 측정마다 한 줄, 3번은 3개월 값 없음
wide <- pivot_wider(long, names_from = time, values_from = sbp)
wide
sum(complete.cases(wide))                              # 짝이 맞는 환자 수
''', title="측정마다 한 줄인 자료를 환자당 한 줄로")

# b. 부호순위 검정과 부호 검정 (3장 b 절)
c_sr = nb.cell('''
before <- c(12, 10, 14, 9, 11, 13, 10, 12, 15, 11, 9, 13, 10, 12)   # 약물 정리 전 복용 약물 수
after  <- c( 9,  9, 11, 9, 10, 10,  8, 12, 11,  9, 10, 10, 12,  9)   # 3개월 뒤
d <- after - before
table(sign(d))                                                   # 감소, 변화 없음, 증가
wilcox.test(after, before, paired = TRUE, exact = FALSE, correct = FALSE)$p.value
wilcox.test(after, before, paired = TRUE, exact = FALSE)$p.value   # 연속성 보정
''', title="부호순위 검정의 정규근사")

c_srex = nb.cell('''
suppressPackageStartupMessages(library(coin))
dd <- data.frame(after, before)
pvalue(wilcoxsign_test(after ~ before, data = dd, zero.method = "Wilcoxon",
                       distribution = "exact"))                  # 0을 빼고 정확 p
binom.test(sum(d > 0), sum(d != 0))$p.value                      # 부호 검정
t.test(after, before, paired = TRUE)$p.value                     # 대응표본 t 검정
''', title="정확 p, 부호 검정, t 검정")

c_hl = nb.cell('''
d0 <- d[d != 0]                                  # 0을 뺀 차이 12개
walsh <- outer(d0, d0, "+") / 2                   # 두 차이의 평균
median(walsh[upper.tri(walsh, diag = TRUE)])      # Walsh 평균 78개의 중앙값
''', title="Hodges–Lehmann 추정치")

# c. 분산분석 (4장 b 절)
c_aov = nb.cell(f'''
pdc <- read.csv("{URL}ch04_pdc.csv")
pdc$arm <- factor(pdc$arm, levels = c("usual", "brief", "intensive"))   # 군의 순서와 기준군
tapply(pdc$pdc, pdc$arm, mean)
fit <- aov(pdc ~ arm, data = pdc)
summary(fit)
''', title="분산분석표")

c_welch = nb.cell('''
oneway.test(pdc ~ arm, data = pdc)                        # 기본값이 Welch
oneway.test(pdc ~ arm, data = pdc, var.equal = TRUE)$p.value
''', title="Welch 분산분석")

c_assume = nb.cell('''
suppressPackageStartupMessages(library(car))
leveneTest(pdc ~ arm, data = pdc)                         # 기본값: 중앙값 기준
leveneTest(pdc ~ arm, data = pdc, center = mean)$`Pr(>F)`[1]
shapiro.test(residuals(fit))
''', title="등분산과 잔차의 정규성")

c_lm = nb.cell('''
lmfit <- lm(pdc ~ arm, data = pdc)                        # 기준군은 factor의 첫 수준(usual)
round(summary(lmfit)$coefficients, 4)
summary(lmfit)$r.squared
''', title="분산분석을 회귀로")

# d. 사후분석 (4장 c 절)
c_tukey = nb.cell('''
TukeyHSD(fit)
''', title="Tukey 방법")

c_dunnett = nb.cell('''
suppressPackageStartupMessages(library(multcomp))
set.seed(1)
dn <- glht(fit, linfct = mcp(arm = "Dunnett"))          # 첫 수준(usual)이 대조군
summary(dn)
''', title="Dunnett 방법")

c_ph = nb.cell('''
pairwise.t.test(pdc$pdc, pdc$arm, p.adjust.method = "holm")   # 군내 평균제곱으로 t
suppressPackageStartupMessages(library(rstatix))
games_howell_test(pdc, pdc ~ arm)[, c("group1", "group2", "estimate", "p.adj")]
''', title="Holm과 Games-Howell")

c_contrast = nb.cell('''
ct <- glht(fit, linfct = mcp(arm = c("brief + intensive - 2 * usual = 0")))
s <- summary(ct)
c(diff = unname(coef(ct)) / 2, t = unname(s$test$tstat), p = unname(s$test$pvalues))
''', title="미리 정한 대비 하나")

# e. 순위 기반 비교 (4장 d·e 절)
c_kw5 = nb.cell('''
tiny <- data.frame(arm = factor(rep(c("usual", "brief", "intensive"), each = 5),
                                levels = c("usual", "brief", "intensive")),
                   pdc = c(58, 63, 67, 69, 73,  62, 68, 71, 77, 82,  70, 74, 79, 81, 86))
kruskal.test(pdc ~ arm, data = tiny)
set.seed(2026)
pvalue(coin::kruskal_test(pdc ~ arm, data = tiny,
                          distribution = approximate(nresample = 100000)))   # 순열검정 p
''', title="군당 5명의 Kruskal-Wallis 검정")

c_kw = nb.cell('''
kruskal.test(pdc ~ arm, data = pdc)
kruskal.test(knowledge ~ arm, data = pdc)$statistic      # 동점이 많은 지식 점수
dunn_test(pdc, pdc ~ arm, p.adjust.method = "holm")[, c("group1", "group2", "statistic", "p", "p.adj")]
''', title="Kruskal-Wallis 검정과 Dunn 검정")

c_jt = nb.cell('''
suppressPackageStartupMessages(library(DescTools))
suppressWarnings(JonckheereTerpstraTest(pdc ~ arm, data = pdc))   # 순서는 factor 수준 순서, 동점이 있어 정규근사
pdc$score <- as.numeric(pdc$arm)                          # usual 1, brief 2, intensive 3
round(confint(lm(pdc ~ score, data = pdc))["score", ], 2)
coef(summary(lm(pdc ~ score, data = pdc)))["score", ]
''', title="순서가 있는 세 군의 추세")

# f. 카이제곱 검정과 효과 지표 (5장 a 절)
c_chi = nb.cell('''
tab <- matrix(c(48, 352,
                102, 498), nrow = 2, byrow = TRUE,
              dimnames = list(drug = c("A", "B"), admit = c("yes", "no")))
ct <- chisq.test(tab, correct = FALSE)
ct
ct$expected
chisq.test(tab)$p.value                      # 기본값: Yates 보정
''', title="2 × 2 표의 카이제곱 검정")

c_eff = nb.cell('''
p1 <- 48 / 400; p2 <- 102 / 600
rd <- p1 - p2; se_rd <- sqrt(p1 * (1 - p1) / 400 + p2 * (1 - p2) / 600)
rr <- p1 / p2; se_lrr <- sqrt(1/48 - 1/400 + 1/102 - 1/600)
or <- (48 * 498) / (352 * 102); se_lor <- sqrt(1/48 + 1/352 + 1/102 + 1/498)
round(rbind(RD = rd + c(0, -1.96, 1.96) * se_rd,
            RR = exp(log(rr) + c(0, -1.96, 1.96) * se_lrr),
            OR = exp(log(or) + c(0, -1.96, 1.96) * se_lor)), 3)
''', title="위험차, 상대위험도, 오즈비")

c_3x2 = nb.cell('''
hypo <- matrix(c(12, 288, 15, 285, 33, 267), ncol = 2, byrow = TRUE,
               dimnames = list(class = c("SGLT2", "DPP4", "SU"), hypo = c("yes", "no")))
ch <- chisq.test(hypo)
c(X2 = unname(ch$statistic), df = unname(ch$parameter), p = ch$p.value)
round(ch$stdres[, "yes"], 2)                                  # 수정 표준화 잔차
pairwise.prop.test(hypo, correct = FALSE, p.adjust.method = "bonferroni")$p.value
''', title="세 군의 비율 비교")

c_mcn = nb.cell('''
pre_post <- matrix(c(47, 7,
                     25, 41), nrow = 2, byrow = TRUE,
                   dimnames = list(before = c("adh", "non"), after = c("adh", "non")))
mcnemar.test(pre_post)$p.value                     # 기본값: 연속성 보정
binom.test(7, 7 + 25)$p.value                      # 바뀐 32명만 보는 정확 검정
chisq.test(rbind(colSums(pre_post), rowSums(pre_post)), correct = FALSE)$p.value   # 잘못된 분석
''', title="같은 환자의 전후 비율")

# g. Fisher 정확 검정 (5장 b 절)
c_fisher = nb.cell('''
tabF <- matrix(c(1, 14,
                 6,  8), nrow = 2, byrow = TRUE,
               dimnames = list(drug = c("X", "Y"), liver = c("yes", "no")))
fisher.test(tabF)
fisher.test(tabF, alternative = "less")$p.value
(1 * 8) / (14 * 6)                                  # 표본 오즈비
''', title="Fisher의 정확한 검정")

c_midp = nb.cell('''
p_obs <- dhyper(1, 15, 14, 7)                       # 관측된 표의 확률
c(midp = fisher.test(tabF)$p.value - p_obs / 2,
  N1 = pchisq(chisq.test(tabF, correct = FALSE)$statistic * 28 / 29, 1, lower.tail = FALSE))
''', title="보수성을 줄이는 두 방법")

# h. 경향 검정 (5장 c 절)
c_trend = nb.cell('''
admit <- c(30, 36, 40); n <- c(200, 300, 500)          # PDC < 40%, 40–79%, ≥ 80%
chisq.test(cbind(admit, n - admit))$statistic          # 순서를 무시한 Pearson χ²
x <- rep(rep(1:3, 2), c(admit, n - admit))              # 점수
y <- rep(c(1, 0), c(sum(admit), sum(n - admit)))        # 입원 여부
r <- cor(x, y)
c(M2 = (length(x) - 1) * r^2, p = pchisq((length(x) - 1) * r^2, 1, lower.tail = FALSE))
prop.trend.test(admit, n)                               # Cochran–Armitage (N r²)
''', title="선형 대 선형 결합 검정과 Cochran–Armitage 검정")

# i. 과제
c_t1 = nb.cell('''
after2 <- after; after2[9] <- 3                      # 환자 9의 변화가 −4가 아니라 −12
c(t = t.test(after2, before, paired = TRUE)$p.value,
  signrank = pvalue(wilcoxsign_test(after2 ~ before, data = data.frame(after2, before),
                                    zero.method = "Wilcoxon", distribution = "exact")))
''', title="과제 1 정답: 극단값 하나")

c_t2 = nb.cell('''
TukeyHSD(aov(pdc ~ arm, data = pdc), conf.level = 0.99)$arm["intensive-usual", ]
''', title="과제 2 정답: 99% 동시 신뢰구간")

c_t3 = nb.cell('''
hypo2 <- hypo[c("DPP4", "SU"), ]
c(pearson = chisq.test(hypo2, correct = FALSE)$p.value, yates = chisq.test(hypo2)$p.value,
  fisher = fisher.test(hypo2)$p.value)
''', title="과제 3 정답: 두 군만 뽑은 표")

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
# 보건통계학 기초 3–5장의 숫자와 대조
A = lambda c, *xs: [None for x in xs if (x in out[c.n]) or (_ for _ in ()).throw(AssertionError((c.title, x)))]
A(c_pt, "t = -4.062, df = 11, p-value = 0.001877", "-10.792916  -3.207084")
A(c_ptwrong, "[1] 0.0988659")
A(c_sr, "10  2  2", "[1] 0.01115628", "[1] 0.01248566")
A(c_srex, "[1] 0.009765625", "[1] 0.03857422", "[1] 0.005818125")
A(c_hl, "[1] -2")
A(c_aov, "4.117 0.0196", "70.17000  75.97000  78.74333")
A(c_welch, "F = 3.7695, num df = 2.000, denom df = 57.808, p-value = 0.02892")
A(c_assume, "0.1771  0.838", "[1] 0.769526", "W = 0.97866, p-value = 0.1455")
A(c_lm, "5.8000     3.0493  1.9021   0.0605", "8.5733     3.0493  2.8116   0.0061", "0.08645667")
A(c_tukey, "0.1441782", "0.0166080", "0.6357816")
A(c_dunnett, "1.902   0.1085", "2.812   0.0116")
A(c_ph, "brief     0.121", "intensive 0.018 0.366", "0.147", "0.023", "0.616")
A(c_contrast, "7.186666667 2.721464302 0.007849552")
A(c_kw5, "chi-squared = 5.78, df = 2, p-value = 0.05558")
A(c_kw, "chi-squared = 7.4468, df = 2, p-value = 0.02415", "8.993514", "0.0968  0.194", "0.00682 0.0204")
A(c_jt, "p-value = 0.006557", "1.27   7.30", "4.286666667")
A(c_chi, "X-squared = 4.7059, df = 1, p-value = 0.03006", "60 340", "[1] 0.03762499")
A(c_eff, "-0.050 -0.094 -0.006", "0.706  0.513  0.971", "0.666  0.460  0.963")
A(c_3x2, "1.382143e+01", "-2.27 -1.42  3.69", "0.003402563 0.02026533")
A(c_mcn, "[1] 0.002654029", "[1] 0.002102402", "[1] 0.01998123")
A(c_fisher, "p-value = 0.03518", "0.001942289 1.082327356", "0.1033228", "[1] 0.03105947", "[1] 0.0952381")
A(c_midp, "0.02075212   0.02533464")
A(c_trend, "8.273184", "8.213064932 0.004158984", "X-squared = 8.2213, df = 1, p-value = 0.00414")

S = nb.save_fragment
S("rc12_pt", nb.html(c_pt, marks={"t = -4.062, df = 11, p-value = 0.001877": 1, "-10.792916  -3.207084": 2, "             -7": 3}))
S("rc12_ptwrong", nb.html(c_ptwrong, marks={"5.96962": 1, "[1] 0.0988659": 2}))
S("rc12_long", nb.html(c_long, marks={"3   144    NA": 1, "[1] 2": 2}))
S("rc12_sr", nb.html(c_sr, marks={"10  2  2": 1, "[1] 0.01115628": 2, "[1] 0.01248566": 3}))
S("rc12_srex", nb.html(c_srex, marks={"[1] 0.009765625": 1, "[1] 0.03857422": 2, "[1] 0.005818125": 3}))
S("rc12_hl", nb.html(c_hl, marks={"[1] -2": 1}))
S("rc12_aov", nb.html(c_aov, marks={"70.17000  75.97000  78.74333": 1, "4.117 0.0196": 2, "139.5": 3}))
S("rc12_welch", nb.html(c_welch, marks={"denom df = 57.808, p-value = 0.02892": 1, "[1] 0.01957574": 2}))
S("rc12_assume", nb.html(c_assume, marks={"0.1771  0.838": 1, "[1] 0.769526": 2, "W = 0.97866, p-value = 0.1455": 3}))
S("rc12_lm", nb.html(c_lm, marks={"70.1700     2.1562": 1, "8.5733     3.0493  2.8116   0.0061": 2, "0.08645667": 3}))
S("rc12_tukey", nb.html(c_tukey, marks={"0.1441782": 1, "1.302439 15.84423 0.0166080": 2}))
S("rc12_dunnett", nb.html(c_dunnett, marks={"1.902   0.1085": 1, "2.812   0.0116": 2}))
S("rc12_ph", nb.html(c_ph, marks={"brief     0.121": 1, "intensive 0.018 0.366": 2, "8.57 0.023": 3}))
S("rc12_contrast", nb.html(c_contrast, marks={"7.186666667 2.721464302 0.007849552": 1}))
S("rc12_kw5", nb.html(c_kw5, marks={"p-value = 0.05558": 1, "[1] 0.04899": 2}))
S("rc12_kw", nb.html(c_kw, marks={"p-value = 0.02415": 1, "8.993514": 2, "0.00682 0.0204": 3}))
S("rc12_jt", nb.html(c_jt, marks={"p-value = 0.006557": 1, "1.27   7.30": 2, "4.286666667": 3}))
S("rc12_chi", nb.html(c_chi, marks={"p-value = 0.03006": 1, "60 340": 2, "[1] 0.03762499": 3}))
S("rc12_eff", nb.html(c_eff, marks={"-0.050 -0.094 -0.006": 1, "0.706  0.513  0.971": 2, "0.666  0.460  0.963": 3}))
S("rc12_3x2", nb.html(c_3x2, marks={"9.970454e-04": 1, "-2.27 -1.42  3.69": 2, "0.003402563 0.02026533": 3}))
S("rc12_mcn", nb.html(c_mcn, marks={"[1] 0.002654029": 1, "[1] 0.002102402": 2, "[1] 0.01998123": 3}))
S("rc12_fisher", nb.html(c_fisher, marks={"p-value = 0.03518": 1, "0.001942289 1.082327356": 2, " 0.1033228": 3, "[1] 0.03105947": 4, "[1] 0.0952381": 5}))
S("rc12_midp", nb.html(c_midp, marks={"0.02075212   0.02533464": 1}))
S("rc12_trend", nb.html(c_trend, marks={" 8.273184": 1, "8.213064932 0.004158984": 2, "X-squared = 8.2213": 3}))
S("rc12_t1", nb.html(c_t1, marks={"0.029448288 0.009765625": 1}))
S("rc12_t2", nb.html(c_t2, marks={"-0.54897932 17.69564598": 1}))
S("rc12_t3", nb.html(c_t3, marks={"0.006755111 0.010521575 0.009880553": 1}))
nb.save_ipynb("3장 짝지은 자료, 세 군 이상, 비율의 비교 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 3–5장의 예로 대응표본 t 검정과 부호순위 검정, 분산분석과 Welch 분산분석, 사후분석, "
                    "Kruskal-Wallis 검정과 Dunn 검정, 추세 검정, 카이제곱 검정과 위험차·상대위험도·오즈비, McNemar 검정, "
                    "Fisher의 정확한 검정, 선형 대 선형 결합 검정을 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"tidyr\", \"coin\", \"car\", \"multcomp\", \"rstatix\", \"DescTools\"))를 실행합니다.",
              notes={c_pt.n: "## a. 짝지은 두 값의 평균 비교", c_sr.n: "## b. 부호순위 검정과 부호 검정",
                     c_aov.n: "## c. 분산분석과 Welch 분산분석", c_tukey.n: "## d. 사후분석",
                     c_kw5.n: "## e. 순위로 비교하기와 추세", c_chi.n: "## f. 카이제곱 검정과 효과 지표",
                     c_fisher.n: "## g. Fisher의 정확한 검정", c_trend.n: "## h. 비율의 추세",
                     c_t1.n: "## i. 과제 정답"})
print("rc12 fragments and notebook written")
