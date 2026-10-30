"""R과 통계분석 1장 · Table 1, 군 비교와 회귀 — R cells run for real with rkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc01.py      (NOMARK=1 … 로 돌리면 모든 셀의 출력을 찍는다)

자료는 MASS 패키지의 birthwt(출생체중 189명). 보건통계학 기초의 파이썬 실습 2·5·9가 Rdatasets에서 읽는 것과 같은 자료라서
끝의 대조 블록에서 그 숫자(Welch t 2.730·df 170.1·p 0.0070, Yates p 0.0396, Pearson 4.9237, Fisher p 0.036,
로지스틱 회귀 흡연 OR 2.331·Wald 1.048–5.187, 체중 10 lb당 0.861)와 맞춘다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc01")

# a. 자료 다듬기
c_load = nb.cell('''
library(MASS)                 # R과 함께 설치되는 패키지. birthwt 자료가 들어 있다
bw <- birthwt
dim(bw)
head(bw, 3)
''', title="출생체중 자료 불러오기")

c_code = nb.cell('''
bw$race <- factor(bw$race, levels = 1:3,
                  labels = c("white", "black", "other"))   # 첫 수준이 기준범주
bw$smoke_f <- factor(bw$smoke, levels = c(0, 1),
                     labels = c("Nonsmoker", "Smoker"))
bw$ptd <- as.integer(bw$ptl > 0)       # 이전 조산 1회 이상 = 1
table(bw$race)
table(bw$smoke_f, bw$smoke)
''', title="범주형 변수에 이름표와 순서 붙이기")

# b. Table 1
c_t1 = nb.cell('''
install.packages("tableone")
library(tableone)
vars <- c("age", "lwt", "race", "ptd", "ht", "ui", "ftv", "bwt", "low")
t1 <- CreateTableOne(vars = vars, strata = "smoke_f", data = bw,
                     factorVars = c("ptd", "ht", "ui", "low"))
print(t1, smd = TRUE)
''', title="CreateTableOne()으로 Table 1 만들기")

c_t1b = nb.cell('''
tab1 <- print(t1, nonnormal = c("ftv", "bwt"), exact = "ht", smd = TRUE,
              quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
write.csv(tab1, "table1.csv")          # 엑셀에서 열어 논문 표로 옮긴다
noquote(tab1[c("ht = 1 (%)", "bwt (median [IQR])"), ])
''', title="중앙값과 정확검정으로 바꾸고 파일로 남기기")

# c. 두 군의 크기
c_tt = nb.cell('''
t.test(bwt ~ smoke_f, data = bw)
''', title="t.test()의 기본값은 Welch")

c_tt2 = nb.cell('''
library(broom)
rbind(tidy(t.test(bwt ~ smoke_f, data = bw)),
      tidy(t.test(bwt ~ smoke_f, data = bw, var.equal = TRUE))
      )[, c("method", "statistic", "parameter", "p.value", "conf.low", "conf.high")]
''', title="Student t 검정과 나란히 보기")

c_mw = nb.cell('''
aggregate(bwt ~ smoke_f, data = bw, FUN = quantile, probs = c(0.25, 0.5, 0.75))
wilcox.test(bwt ~ smoke_f, data = bw)
''', title="중앙값과 Wilcoxon 순위합 검정")

# d. 비율
c_tab = nb.cell('''
tab <- table(smoke = bw$smoke_f, low = bw$low)
addmargins(tab)
round(100 * prop.table(tab, margin = 1), 1)      # 행 백분율
''', title="교차표와 행 백분율")

c_chi = nb.cell('''
chisq.test(tab)
chisq.test(tab, correct = FALSE)
chisq.test(tab)$expected
''', title="chisq.test()의 기본값은 Yates 보정")

c_fis = nb.cell('''
fisher.test(tab)
(tab[2, 2] * tab[1, 1]) / (tab[2, 1] * tab[1, 2])    # 표에서 바로 구한 오즈비
''', title="Fisher의 정확한 검정과 두 가지 오즈비")

# e. 회귀
c_lm = nb.cell('''
fit_lm <- lm(bwt ~ age + lwt + race + smoke + ptd + ht + ui, data = bw)
round(cbind(estimate = coef(fit_lm), confint(fit_lm)), 1)
''', title="선형회귀로 보정한 체중 차이")

c_glm = nb.cell('''
fit_lr <- glm(low ~ age + lwt + race + smoke + ptd + ht + ui,
              family = binomial, data = bw)
summary(fit_lr)
''', title="glm()으로 로지스틱 회귀")

c_or = nb.cell('''
round(exp(cbind(OR = coef(fit_lr), confint.default(fit_lr))), 3)   # Wald 구간
round(exp(confint(fit_lr)), 3)                                      # 프로파일 우도 구간
''', title="오즈비와 두 가지 신뢰구간")

c_tidy = nb.cell('''
or_tab <- tidy(fit_lr, exponentiate = TRUE, conf.int = TRUE)[-1, ]   # 절편 행 빼기
data.frame(term = or_tab$term,
           `OR (95% CI)` = sprintf("%.2f (%.2f-%.2f)", or_tab$estimate,
                                   or_tab$conf.low, or_tab$conf.high),
           P = format.pval(or_tab$p.value, digits = 2, eps = 0.001),
           check.names = FALSE)
''', title="논문에 옮길 회귀 결과 표")

# f. 과제
c_t1x = nb.cell('''
t0 <- CreateTableOne(vars = vars, strata = "smoke_f", data = bw)   # factorVars 없이
print(t0)
''', title="과제 1 정답: factorVars를 빼면")

c_t2x = nb.cell('''
b  <- coef(fit_lr)["lwt"]
ci <- confint.default(fit_lr)["lwt", ]
round(exp(10 * c(OR = b, ci)), 3)        # 10 lb당 오즈비와 Wald 구간
''', title="과제 2 정답: 체중 10 lb당 오즈비")

c_t3x = nb.cell('''
crude <- glm(low ~ smoke, family = binomial, data = bw)
round(exp(cbind(OR = coef(crude), confint.default(crude))), 3)
''', title="과제 3 정답: 보정하지 않은 오즈비")

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

# ---------------------------------------------------------------- 파이썬 실습과 대조
out = {c.n: c.text for c in nb.cells}
assert "t = 2.7299, df = 170.1, p-value = 0.007003" in out[c_tt.n]                        # 실습 2 Welch
assert "187" in out[c_tt2.n] and "0.00867" in out[c_tt2.n]                                # 실습 2 Student df 187
assert "W = 5249.5, p-value = 0.006768" in out[c_mw.n] and "3100.00" in out[c_mw.n]       # 실습 2 U(nonsmoker), 중앙값
assert "X-squared = 4.2359, df = 1, p-value = 0.03958" in out[c_chi.n]                    # 실습 5 Yates p 0.0396
assert "X-squared = 4.9237, df = 1, p-value = 0.02649" in out[c_chi.n]                    # 실습 5 Pearson
assert "p-value = 0.03618" in out[c_fis.n]                                               # 실습 5 Fisher 0.036
assert "smoke       2.331 1.048  5.187" in out[c_or.n]                                     # 실습 9 Wald
assert "0.861" in out[c_t2x.n] and "0.989" in out[c_t2x.n]                                # 실습 9 10 lb당
assert "0.009" in out[c_t1.n] and "0.040" in out[c_t1.n]                                  # Student, Yates

S = nb.save_fragment
S("rc01_load", nb.html(c_load, marks={"189  10": 1, "2523": 2}))
S("rc01_code", nb.html(c_code, marks={"white black other": 1, "Nonsmoker 115": 2}))
S("rc01_t1", nb.html(c_t1, marks={"Nonsmoker        Smoker": 1, "23.43 (5.47)": 2, "ptd = 1 (%)": 3,
                                  " 0.009 ": 4, " 0.040 ": 5, "0.757": 6}))
S("rc01_t1b", nb.html(c_t1b, marks={"3100.00 [2509.00, 3621.50]": 1, "Stratified by smoke_f\n                     p": 2,
                                    "1.000 exact": 3, "0.007 nonnorm": 4}))
S("rc01_tt", nb.html(c_tt, marks={"Welch Two Sample t-test": 1, "df = 170.1": 2, "78.57486 488.97860": 3,
                                  "group Nonsmoker and group Smoker": 4}))
S("rc01_tt2", nb.html(c_tt2, marks={"# A tibble: 2 x 6": 1, "187  ": 2}))
S("rc01_mw", nb.html(c_mw, marks={"2775.50": 1, "W = 5249.5": 2, "with continuity correction": 3}))
S("rc01_tab", nb.html(c_tab, marks={"189": 1, "40.5": 2}))
S("rc01_chi", nb.html(c_chi, marks={"with Yates' continuity correction": 1, "p-value = 0.03958": 2,
                                    "p-value = 0.02649": 3, "23.10053": 4}))
S("rc01_fis", nb.html(c_fis, marks={"1.028780 3.964904": 1, "2.014137": 2, "2.021944": 3}))
S("rc01_lm", nb.html(c_lm, marks={"-328.1": 1, "-473.7": 2}))
S("rc01_glm", nb.html(c_glm, marks={"family = binomial": 1, "0.84640": 2, "Residual deviance: 196.83": 3,
                                    "Number of Fisher Scoring iterations: 4": 4}))
S("rc01_or", nb.html(c_or, marks={"smoke       2.331 1.048  5.187": 1, "Waiting for profiling to be done...": 2,
                                  "smoke       1.058  5.293": 3, "ht          1.640 27.258": 4}))
S("rc01_tidy", nb.html(c_tidy, marks={"2.33 (1.06-5.29)": 1, "0.99 (0.97-1.00)": 2, "0.0083": 3}))
S("rc01_t1x", nb.html(c_t1x, marks={"ptd (mean (SD))    0.10 (0.31)": 1, "0.011": 2}))
S("rc01_t2x", nb.html(c_t2x, marks={"0.861": 1}))
S("rc01_t3x", nb.html(c_t3x, marks={"2.022": 1}))
nb.save_ipynb("1장 Table 1, 군 비교와 회귀 (R과 통계분석)",
              intro="MASS 패키지의 출생체중 자료(birthwt)로 Table 1을 만들고, 두 군의 평균과 비율을 비교하고, "
                    "선형회귀와 로지스틱 회귀를 맞춥니다. 보건통계학 기초의 파이썬 실습 2·5·9와 같은 자료입니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 자료를 분석할 모양으로 다듬기", c_t1.n: "## b. Table 1 만들기",
                     c_tt.n: "## c. 두 군의 평균과 중앙값 비교", c_tab.n: "## d. 두 군의 비율 비교",
                     c_lm.n: "## e. 선형회귀와 로지스틱 회귀", c_t1x.n: "## f. 과제 정답"})
print("rc01 fragments and notebook written")
