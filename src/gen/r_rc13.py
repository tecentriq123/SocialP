"""R과 파이썬으로 하는 통계분석 4장 · 상관과 회귀 — R cells run for real with rkit.

run:  python3 gen/r_rc13.py      (NOMARK=1 python3 gen/r_rc13.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 6장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-18 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 6장 a·b·c 절의 환자 8명(나이, 수축기혈압), c 절의 지역약국 참여자 100명(pub/data/ch06_bp100.csv),
d 절의 고혈압 코호트 240명(ch06_htn240.csv), 로그 의료비 400명(ch06_cost400.csv), e 절의 HbA1c 시험 72명
(ch06_hba1c72.csv). CSV는 gen/data_rc13.py가 gen/nums_ch06.py·nums_ch06e.py에서 만든다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc13")

# a. Pearson
c_r8 = nb.cell('''
age <- c(42, 48, 53, 58, 62, 66, 71, 80)            # 나이 (세)
sbp <- c(125, 127, 133, 141, 129, 139, 137, 149)    # 수축기혈압 (mmHg)
cor.test(age, sbp)                                  # Pearson r, t, 양측 p, Fisher z 95% CI
''', title="환자 8명의 Pearson 상관계수")

c_rna = nb.cell('''
sbp_na <- replace(sbp, 3, NA)                       # 셋째 환자의 혈압이 빠졌다면
cor(age, sbp_na)                                    # 결측이 있으면 NA
cor(age, sbp_na, use = "complete.obs")              # 두 값이 모두 있는 환자만
cor.test(age, sbp_na)$parameter                     # 자유도 5 = 분석한 쌍 7 − 2
''', title="결측값이 있으면")

c_boot = nb.cell('''
suppressPackageStartupMessages(library(boot))
set.seed(2026)
d8 <- data.frame(age, sbp)
b <- boot(d8, function(d, i) cor(d$age[i], d$sbp[i]), R = 4000)   # 환자를 짝지은 채 다시 뽑기
boot.ci(b, type = c("perc", "bca"))                 # 백분위수 구간과 BCa 구간
''', title="부트스트랩 신뢰구간")

# b. Spearman, Kendall
c_sp = nb.cell('''
cor.test(age, sbp, method = "spearman")                  # n < 1290이고 동점이 없으면 정확 p
cor.test(age, sbp, method = "spearman", exact = FALSE)   # t 근사 p
''', title="Spearman 상관계수와 두 가지 p")

c_kd = nb.cell('''
cor.test(age, sbp, method = "kendall")              # 동점이 없고 n < 50이면 정확 p
''', title="Kendall의 τ")

c_out = nb.cell('''
nrs   <- c(2, 4, 5, 7, 8, 9)                        # 통증 점수
doses <- c(1, 0, 3, 2, 5, 4)                        # 추가 복용 횟수
doses50 <- replace(doses, 5, 50)                    # 한 명이 50회였다면
round(c(pearson = cor(nrs, doses), pearson50 = cor(nrs, doses50),
        spearman = cor(nrs, doses, method = "spearman"),
        spearman50 = cor(nrs, doses50, method = "spearman")), 3)
''', title="극단값 한 명이 두 계수에 주는 영향")

# c. 단순회귀
c_lm = nb.cell('''
bp <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch06_bp100.csv")
fit <- lm(sbp ~ age, data = bp)
summary(fit)
confint(fit)
''', title="지역약국 참여자 100명의 단순회귀")

c_pred = nb.cell('''
new <- data.frame(age = 70)
predict(fit, new, interval = "confidence")          # 70세 평균의 95% 신뢰구간
predict(fit, new, interval = "prediction")          # 70세 한 사람의 95% 예측구간
''', title="평균의 신뢰구간과 개인의 예측구간")

c_diag = nb.cell('''
par(mfrow = c(1, 2))
plot(fit, which = 1)                                # 잔차 대 예측값: 휘어짐, 부채꼴
plot(fit, which = 2)                                # 잔차의 Q-Q plot
''', title="잔차 그림", fig=(8, 4))

c_hc3 = nb.cell('''
suppressPackageStartupMessages({library(sandwich); library(lmtest)})
coeftest(fit, vcov = vcovHC(fit, type = "HC3"))     # 이분산에 강건한 표준오차 (계수는 그대로)
''', title="이분산에 강건한 표준오차")

c_sum = nb.cell('''
n <- 100; b0 <- 97.2326; b1 <- 0.5296; se_b1 <- 0.1169; s <- 14.5008   # 출력의 요약값
tc <- qt(0.975, n - 2)
round(c(t_crit = tc, lo = b1 - tc * se_b1, hi = b1 + tc * se_b1), 3)    # 기울기의 95% CI
sxx <- (s / se_b1)^2                                # SE(b1) = s / sqrt(Sxx)를 거꾸로 풂
yhat <- b0 + b1 * 70
se_m <- s * sqrt(1 / n + (70 - 56)^2 / sxx)         # 70세 평균의 표준오차 (평균 나이 56세)
se_p <- s * sqrt(1 + 1 / n + (70 - 56)^2 / sxx)     # 70세 한 사람 예측의 표준오차
round(c(yhat = yhat, se_mean = se_m, se_pred = se_p), 2)
round(c(yhat - tc * se_m, yhat + tc * se_m, yhat - tc * se_p, yhat + tc * se_p), 1)
''', title="출력의 요약값만으로 두 구간 다시 계산하기")

# d. 다중회귀
c_load = nb.cell('''
d <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch06_htn240.csv")
d$inst <- factor(d$inst, levels = c("clinic", "hospital", "tertiary"))   # 첫 수준 = 기준범주
d$sbp0_10 <- d$sbp0 / 10                            # 기저 SBP, 10 mmHg 단위
d$age10 <- d$age / 10                               # 나이, 10세 단위
m1 <- lm(sbp6 ~ classes, data = d)                  # 보정 전
m2 <- lm(sbp6 ~ classes + sbp0, data = d)           # 기저 SBP만 보정
rbind(crude = coef(summary(m1))["classes", ], base_adjusted = coef(summary(m2))["classes", ])
''', title="고혈압 코호트 240명: 보정 전과 기저 SBP 보정")

c_full = nb.cell('''
full <- lm(sbp6 ~ classes + sbp0_10 + age10 + female + diabetes + bmi + adherent + inst, data = d)
summary(full)
confint(full)["classes", ]
nobs(full)                                          # 분석에 들어간 환자 수 (결측이 있으면 줄어듦)
''', title="전체 보정 모형")

c_inst = nb.cell('''
red <- update(full, . ~ . - inst)                   # 요양기관 종별을 뺀 모형
anova(red, full)                                    # 가변수 2개를 한꺼번에: P for type of institution
''', title="범주형 변수 전체의 F 검정")

c_ref = nb.cell('''
d$inst_h <- relevel(d$inst, ref = "hospital")       # 기준범주를 병원으로 바꾸면
coef(update(full, . ~ . - inst + inst_h))[c("inst_hclinic", "inst_htertiary")]
d$inst_code <- as.integer(d$inst)                   # 1, 2, 3 코드를 factor 없이 넣으면
coef(update(full, . ~ . - inst + inst_code))["inst_code"]   # 연속형으로 계수 하나
''', title="기준범주와 숫자 코드")

c_int = nb.cell('''
fi <- lm(sbp6 ~ classes * diabetes + sbp0_10 + age10 + female + bmi + adherent + inst, data = d)
coef(summary(fi))[c("classes", "diabetes", "classes:diabetes"), ]
suppressPackageStartupMessages(library(multcomp))
confint(glht(fi, linfct = "classes + `classes:diabetes` = 0"))$confint[1, ]   # 당뇨 환자의 기울기와 95% CI
fi2 <- lm(sbp6 ~ classes * inst + sbp0_10 + age10 + female + diabetes + bmi + adherent, data = d)
anova(full, fi2)                                    # 계열 수 × 요양기관 종별 (자유도 2)
''', title="교호작용")

# e. 모형 점검과 변환
c_spl = nb.cell('''
library(splines)
sp <- lm(sbp6 ~ classes + ns(sbp0_10, df = 3) + age10 + female + diabetes + bmi + adherent + inst, data = d)
anova(full, sp)                                     # 기저 SBP: 곡선 대 직선
coef(sp)["classes"]
fc <- lm(sbp6 ~ factor(classes) + sbp0_10 + age10 + female + diabetes + bmi + adherent + inst, data = d)
coef(summary(fc))[2:4, ]                            # 1계열과 비교한 2·3·4계열
confint(fc)["factor(classes)4", ]
anova(full, fc)                                     # 계열 수: 범주 대 직선
''', title="직선이 맞는지: 스플라인과 범주")

c_vif = nb.cell('''
suppressPackageStartupMessages(library(car))
round(vif(full), 2)                                 # 범주형 inst가 있어 GVIF 열로 나옴
fw <- update(full, . ~ . + weight)                  # BMI와 겹치는 체중을 더 넣으면
round(cor(d$bmi, d$weight), 2)
coef(summary(fw))[c("bmi", "classes"), ]
round(vif(fw)[c("bmi", "weight"), "GVIF"], 2)
''', title="VIF와 설명변수의 겹침")

c_std = nb.cell('''
b <- coef(full)
round(c(classes = b[["classes"]] * sd(d$classes), sbp0 = b[["sbp0_10"]] * sd(d$sbp0_10),
        adherent = b[["adherent"]] * sd(d$adherent)) / sd(d$sbp6), 2)   # 표준화 계수
''', title="표준화 계수")

c_log = nb.cell('''
cost <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch06_cost400.csv")
fl <- lm(log(cost) ~ adherent + I(age / 10) + cci, data = cost)
coef(summary(fl))
round(100 * (exp(cbind(estimate = coef(fl), confint(fl))[-1, ]) - 1), 1)   # 백분율 차이(%)
''', title="로그 의료비 모형")

# f. 공분산분석
c_anc = nb.cell('''
h <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch06_hba1c72.csv")
anc <- lm(hba1c_6m ~ arm + hba1c_0, data = h)      # arm: 복약상담 1, 일반 진료 0
summary(anc)
confint(anc)["arm", ]
''', title="HbA1c 시험 72명의 공분산분석")

c_three = nb.cell('''
rbind(post   = coef(summary(lm(hba1c_6m ~ arm, data = h)))["arm", ],
      change = coef(summary(lm(I(hba1c_6m - hba1c_0) ~ arm, data = h)))["arm", ],
      ancova = coef(summary(anc))["arm", ])
''', title="세 분석의 군 차이")

c_emm = nb.cell('''
suppressPackageStartupMessages(library(emmeans))
h$arm_f <- factor(h$arm, labels = c("usual", "counsel"))
emmeans(lm(hba1c_6m ~ arm_f + hba1c_0, data = h), ~ arm_f)   # 기저값을 전체 평균에 둔 군별 평균
anova(anc, lm(hba1c_6m ~ arm * hba1c_0, data = h))          # 두 군의 기울기가 같은가
''', title="군별 보정 평균과 기울기 점검")

# g. 과제
c_t1 = nb.cell('''
r <- 0.45; n <- 28
t <- r * sqrt(n - 2) / sqrt(1 - r^2)
c(t = t, p = 2 * pt(-abs(t), n - 2))
tanh(atanh(r) + c(-1, 1) * qnorm(0.975) / sqrt(n - 3))     # Fisher z 95% CI
''', title="과제 1 정답")

c_t2 = nb.cell('''
predict(fit, data.frame(age = c(40, 90)), interval = "confidence")
predict(fit, data.frame(age = c(40, 90)), interval = "prediction")
range(bp$age)
''', title="과제 2 정답")

c_t3 = nb.cell('''
fem <- lm(sbp6 ~ classes + sbp0_10 + age10 + diabetes + bmi + adherent + inst, data = subset(d, female == 1))
mal <- lm(sbp6 ~ classes + sbp0_10 + age10 + diabetes + bmi + adherent + inst, data = subset(d, female == 0))
rbind(women = coef(summary(fem))["classes", ], men = coef(summary(mal))["classes", ])
fx <- lm(sbp6 ~ classes * female + sbp0_10 + age10 + diabetes + bmi + adherent + inst, data = d)
coef(summary(fx))["classes:female", ]
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
# 보건통계학 기초 6장의 숫자와 대조
A = lambda c, *xs: [None for x in xs if (x in out[c.n]) or (_ for _ in ()).throw(AssertionError((c.title, x)))]
A(c_r8, "t = 3.8356, df = 6, p-value = 0.008602", "0.3401360 0.9708735", "0.8428022")
A(c_rna, "[1] NA", " 5 ")
A(c_sp, "S = 18, p-value = 0.02793", "S = 18, p-value = 0.02082", "0.7857143")
A(c_kd, "T = 23, p-value = 0.03115", "0.6428571")
A(c_out, "0.790      0.451      0.829      0.829")
A(c_lm, "0.5296     0.1169   4.531 1.66e-05", "Residual standard error: 14.5 on 98 degrees of freedom",
  "Multiple R-squared:  0.1732,\tAdjusted R-squared:  0.1648", "F-statistic: 20.53 on 1 and 98 DF", "0.2976755   0.7615399")
A(c_pred, "134.3051 129.9629 138.6473", "134.3051 105.203 163.4072")
A(c_sum, "1.984  0.298  0.762", "134.30    2.19   14.66", "130.0 138.6 105.2 163.4")
A(c_load, "3.688303  0.9308424", "-4.431045  1.1027896")
A(c_full, "-4.0424     1.0965  -3.687 0.000284", "2.1011     1.4773   1.422 0.156321", "4.2784     1.7030   2.512 0.012679",
  "-6.202918 -1.881860", "[1] 240", "Residual standard error: 9.647 on 230 degrees of freedom")
A(c_inst, "3.4132 0.03461")
A(c_ref, "-2.101074       2.177373", "2.132961")
A(c_int, "-3.524927   1.235234", "-1.421994   1.560662 -0.9111478 0.363175582", "-4.946921", "-7.862013", "-2.031829", "0.0205 0.9797")
A(c_spl, "2.8351 0.06079", "-4.106663", "-2.584264", "-7.324613", "-14.934934", "-23.167846  -6.702022", "1.0483 0.3522")
A(c_vif, "classes  2.33", "sbp0_10  2.21", "[1] 0.8", "-0.03508148   0.401017", "-4.03559635   1.097825", "4.56   6.13")
A(c_std, "-0.27     0.75    -0.26")
A(c_log, "-0.23747967 0.09127180", "-21.1 -34.1   -5.6", "18.3  10.6   26.5")
A(c_anc, "-0.3829     0.1785  -2.145 0.035482", "0.5121     0.1055", "0.7497 on 69 degrees of freedom", "-0.73902617 -0.02677408")
A(c_three, "-0.2597222  0.2031792", "-0.5002778  0.2007879", "-0.3829001  0.1785142")
A(c_emm, "7.78 0.126", "7.40 0.126", "0.0309  0.861")
A(c_t1, "2.56941255 0.01627228", "0.09244279 0.70475877")
A(c_t3, "-2.006383   1.648328", "-5.850290   1.513760", "0.7654543")

S = nb.save_fragment
S("rc13_r8", nb.html(c_r8, marks={"t = 3.8356, df = 6, p-value = 0.008602": 1, " 0.3401360 0.9708735": 2, "0.8428022": 3}))
S("rc13_rna", nb.html(c_rna, marks={"[1] NA": 1, "[1] 0.8463583": 2, "\n 5": 3}))
S("rc13_boot", nb.html(c_boot, marks={"( 0.3571,  0.9810 )": 1, "( 0.1785,  0.9724 )": 2}))
S("rc13_sp", nb.html(c_sp, marks={"S = 18, p-value = 0.02793": 1, "S = 18, p-value = 0.02082": 2}))
S("rc13_kd", nb.html(c_kd, marks={"T = 23, p-value = 0.03115": 1}))
S("rc13_out", nb.html(c_out, marks={"0.790      0.451      0.829      0.829": 1}))
S("rc13_lm", nb.html(c_lm, marks={"0.5296     0.1169   4.531 1.66e-05": 1, "Residual standard error: 14.5 on 98 degrees of freedom": 2,
                                  "Multiple R-squared:  0.1732": 3, "F-statistic: 20.53 on 1 and 98 DF,  p-value: 1.656e-05": 4,
                                  "age          0.2976755   0.7615399": 5}))
S("rc13_pred", nb.html(c_pred, marks={"134.3051 129.9629 138.6473": 1, "134.3051 105.203 163.4072": 2}))
S("rc13_diag", nb.html(c_diag))
S("rc13_hc3", nb.html(c_hc3, marks={"0.52961    0.11659  4.5423 1.587e-05": 1}))
S("rc13_sum", nb.html(c_sum, marks={"1.984  0.298  0.762": 1, "134.30    2.19   14.66": 2, "130.0 138.6 105.2 163.4": 3}))
S("rc13_load", nb.html(c_load, marks={"3.688303  0.9308424": 1, "-4.431045  1.1027896": 2}))
S("rc13_full", nb.html(c_full, marks={"-4.0424     1.0965  -3.687 0.000284": 1, "insthospital   2.1011": 2, "-6.202918 -1.881860": 3, "[1] 240": 4}))
S("rc13_inst", nb.html(c_inst, marks={"3.4132 0.03461": 1}))
S("rc13_ref", nb.html(c_ref, marks={"-2.101074       2.177373": 1, "2.132961": 2}))
S("rc13_int", nb.html(c_int, marks={"-3.524927   1.235234": 1, "-1.421994   1.560662 -0.9111478 0.363175582": 2, "-4.946921": 3, "0.0205 0.9797": 4}))
S("rc13_spl", nb.html(c_spl, marks={"2.8351 0.06079": 1, "-14.934934": 2, "1.0483 0.3522": 3}))
S("rc13_vif", nb.html(c_vif, marks={"classes  2.33": 1, "-0.03508148   0.401017": 2, "4.56   6.13": 3}))
S("rc13_std", nb.html(c_std, marks={"-0.27     0.75    -0.26": 1}))
S("rc13_log", nb.html(c_log, marks={"-0.23747967 0.09127180": 1, "-21.1 -34.1   -5.6": 2}))
S("rc13_anc", nb.html(c_anc, marks={"-0.3829     0.1785  -2.145 0.035482": 1, "0.5121     0.1055": 2, "0.7497 on 69 degrees of freedom": 3}))
S("rc13_three", nb.html(c_three, marks={"-0.2597222  0.2031792": 1, "-0.5002778  0.2007879": 2, "-0.3829001  0.1785142": 3}))
S("rc13_emm", nb.html(c_emm, marks={"7.78 0.126": 1, "7.40 0.126": 2, "0.0309  0.861": 3}))
S("rc13_t1", nb.html(c_t1, marks={"2.56941255 0.01627228": 1, "0.09244279 0.70475877": 2}))
S("rc13_t2", nb.html(c_t2, marks={"[1] 36 80": 1}))
S("rc13_t3", nb.html(c_t3, marks={"0.7654543": 1}))
nb.save_ipynb("4장 상관과 회귀 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 6장의 예로 Pearson·Spearman·Kendall 상관계수, 단순회귀와 두 구간, 다중회귀와 범주형 변수, "
                    "교호작용, 스플라인과 VIF, 로그 결과 모형, 공분산분석을 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"boot\", \"sandwich\", \"lmtest\", \"multcomp\", \"car\", \"emmeans\"))를 실행합니다.",
              notes={c_r8.n: "## a. Pearson 상관계수와 신뢰구간", c_sp.n: "## b. Spearman과 Kendall 상관계수",
                     c_lm.n: "## c. 단순회귀와 두 구간", c_load.n: "## d. 다중회귀와 범주형 변수",
                     c_spl.n: "## e. 모형 점검과 로그 변환", c_anc.n: "## f. 공분산분석", c_t1.n: "## g. 과제 정답"})
print("rc13 fragments and notebook written")
