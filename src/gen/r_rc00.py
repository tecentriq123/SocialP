"""R과 통계분석 0장 · R 시작하기 — R cells run for real with rkit.

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc00.py      (NOMARK=1 … 로 돌리면 모든 셀의 출력을 찍는다)

자료는 보건통계학 기초 21장의 pub/data/rcc_cost.csv(신세포암 환자 3,000명의 1년 의료비, 만원).
끝의 대조 블록에서 21장의 숫자(군별 평균 2,669·2,273만원, 중앙값 2,462·1,836만원, 선형회귀의 보정 차이 513만원)와 맞춘다.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc00")

# a. 계산기처럼
c_calc = nb.cell('''
1 + 2 * 3
surv5 <- c(exp = 24.1, weibull = 19.5, lnorm = 32.6)   # 5년 생존율(%) 세 개
surv5 / 100
mean(surv5)
surv5["weibull"]
''', title="값을 만들고 함수에 넣기")

# b. 패키지
c_pkg = nb.cell('''
install.packages("survival")     # 처음 한 번만 (Colab은 세션마다)
library(survival)
packageVersion("survival")
''', title="패키지 설치와 불러오기")

# c. 자료
c_read = nb.cell('''
options(HTTPUserAgent = "Mozilla/5.0")   # 사이트가 프로그램의 기본 요청을 막을 때를 대비
BASE <- "https://socialp-ajou.tecentriq12.workers.dev/data/"
cost <- read.csv(paste0(BASE, "rcc_cost.csv"))
dim(cost)
head(cost, 3)
''', title="자료 불러오기")

c_str = nb.cell('''
str(cost)
summary(cost$cost)
table(cost$arm)
sum(is.na(cost))
''', title="열의 형식과 요약")

c_fac = nb.cell('''
cost$arm <- factor(cost$arm, levels = c("B", "A"))   # 표준요법 B를 기준으로
levels(cost$arm)
table(cost$arm, cost$stage4)
''', title="범주형 변수(factor)와 교차표")

# d. 고르고 요약하기
c_base = nb.cell('''
a_only <- cost[cost$arm == "A", c("pid", "age", "cost")]
nrow(a_only)
cost$cost_m <- cost$cost / 100          # 만원 → 백만원
tapply(cost$cost, cost$arm, mean)
aggregate(cost ~ arm, data = cost, FUN = median)
''', title="행과 열 고르기, 군별 요약 (기본 R)")

c_dplyr = nb.cell('''
install.packages("dplyr")
library(dplyr)
cost |>
  group_by(arm) |>
  summarise(n = n(),
            mean_cost   = mean(cost),
            median_cost = median(cost),
            no_inpatient = mean(cost_inp == 0))
''', title="dplyr로 군별 요약")

# e. 결과 읽고 남기기
c_lm = nb.cell('''
fit <- lm(cost ~ arm + age + male + cci + stage4, data = cost)
fit
''', title="모형을 맞추고 이름만 쓰기")

c_sum = nb.cell('''
summary(fit)
''', title="summary()로 자세히 보기")

c_ci = nb.cell('''
round(confint(fit), 1)
library(sandwich)
library(lmtest)
rob <- vcovHC(fit, type = "HC3")         # 이분산에 강건한 공분산 (21장과 같은 HC3)
round(coeftest(fit, vcov = rob)["armA", ], 3)
round(coefci(fit, vcov = rob)["armA", ], 1)
''', title="신뢰구간과 강건 표준오차")

c_err = nb.cell('''
mean(cost$Cost)
''', title="이름을 틀리면", expect_error=False)

c_err2 = nb.cell('''
summary(fit2)
''', title="없는 객체를 부르면", expect_error=True)

c_save = nb.cell('''
res <- data.frame(term = names(coef(fit)), estimate = coef(fit), confint(fit))
write.csv(res, "rcc_cost_lm.csv", row.names = FALSE)
read.csv("rcc_cost_lm.csv")[2, ]
''', title="결과를 파일로 남기기")

# f. 과제
c_t1 = nb.cell('''
cost |>
  group_by(arm, stage4) |>
  summarise(n = n(), mean_cost = round(mean(cost), 1), .groups = "drop")
''', title="과제 1 정답: 병기별 평균 비용")

c_t2 = nb.cell('''
cost$high <- cost$cost > 5000            # 5,000만원 넘게 쓴 환자
tapply(cost$high, cost$arm, mean)
table(cost$arm, cost$high)
''', title="과제 2 정답: 고비용 환자의 비율")

c_t3 = nb.cell('''
fit_b <- lm(cost ~ arm + age + male + cci + stage4,
            data = transform(cost, arm = relevel(arm, ref = "A")))
coef(fit_b)["armB"]
''', title="과제 3 정답: 기준범주를 A로")

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

# ---------------------------------------------------------------- 21장과 대조
out = {c.n: c.text for c in nb.cells}
assert "2272.788 2668.862" in out[c_base.n] and "1836.4" in out[c_base.n] and "2462.3" in out[c_base.n]
assert "512.573" in out[c_lm.n] and "54.222" in out[c_sum.n]
assert "50.960" in out[c_ci.n] and "412.7" in out[c_ci.n] and "612.5" in out[c_ci.n]   # 21장: HC3 SE 50.96, CI 412.7–612.5

S = nb.save_fragment
S("rc00_calc", nb.html(c_calc, marks={"[1] 7": 1, "0.241": 2, "25.4": 3, "19.5": 4}))
S("rc00_pkg", nb.html(c_pkg, marks={"3.8.12": 1}))
S("rc00_read", nb.html(c_read, marks={"3000   10": 1, "R0001": 2}))
S("rc00_str", nb.html(c_str, marks={"$ arm      : chr": 1, "2218.7  2431.6": 2, "[1] 0": 3}))
S("rc00_fac", nb.html(c_fac, marks={'"B" "A"': 1, "636 1161": 2}))
S("rc00_base", nb.html(c_base, marks={"[1] 1203": 1, "2272.788 2668.862": 2, "1836.4": 3}))
S("rc00_dplyr", nb.html(c_dplyr, marks={"filter, lag": 1, "# A tibble: 2 x 5": 2, "0.357": 3}))
S("rc00_lm", nb.html(c_lm, marks={"512.573": 1}))
S("rc00_sum", nb.html(c_sum, marks={"54.222": 1, "9.453   <2e-16 ***": 2, "Multiple R-squared:  0.04681": 3}))
S("rc00_ci", nb.html(c_ci, marks={"406.3  618.9": 1, "as.Date, as.Date.numeric": 2, "50.960": 3, "412.7": 4}))
S("rc00_err", nb.html(c_err, marks={"[1] NA": 1}))
S("rc00_err2", nb.html(c_err2))
S("rc00_save", nb.html(c_save, marks={"512.5732": 1}))
S("rc00_t1", nb.html(c_t1, marks={"2655.": 1}))
S("rc00_t2", nb.html(c_t2, marks={"0.04897051": 1}))
S("rc00_t3", nb.html(c_t3, marks={"-512.5732": 1}))
nb.save_ipynb("0장 R 시작하기 (R과 통계분석)",
              intro="보건통계학 기초 21장의 신세포암 의료비 자료(rcc_cost.csv)로 R의 기본을 익힙니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={1: "## a. 값과 함수", 2: "## b. 패키지", 3: "## c. 자료 불러와 살펴보기", 6: "## d. 고르고 요약하기",
                     8: "## e. 결과를 읽고 남기기", 14: "## f. 과제 정답"}, skip=(12,))
print("rc00 fragments and notebook written")

