"""R과 통계분석 2장 · 생존 자료와 Kaplan-Meier — R cells run for real with rkit (survival, survminer).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc02.py      (NOMARK=1 … 로 돌리면 모든 셀의 출력을 찍는다)

자료는 survival 패키지의 lung(NCCTG 폐암 228명). 보건통계학 기초 파이썬 실습 7이 Rdatasets의 cancer.csv로 읽는 것과 같은 자료라서
끝의 대조 블록에서 그 숫자(중앙생존 10.18개월 9.33–11.86, 12개월 0.409 0.339–0.478, 남 8.9 6.9–10.1, 여 14.0 11.3–17.2,
로그순위 10.33, Peto 12.71, ECOG 18.98, 짝 비교 0.1890, RMST 차이 4.06 1.91–6.21, 층화 11.05)와 맞춘다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc02")

# a. 생존 자료
c_load = nb.cell('''
library(survival)
head(lung, 3)
table(lung$status)
''', title="폐암 자료 불러오기")

c_surv = nb.cell('''
lung$months <- lung$time / 30.4375      # 일 → 개월 (365.25 / 12)
s <- Surv(lung$months, lung$status)
head(s)
table(s[, "status"])
''', title="Surv()로 생존 자료 만들기")

c_fu = nb.cell('''
survfit(Surv(months, status == 1) ~ 1, data = lung)   # 중도절단을 사건으로 본 역 KM
''', title="중앙 추적기간 (역 Kaplan-Meier)")

# b. KM
c_km = nb.cell('''
km <- survfit(Surv(months, status) ~ 1, data = lung)
km
summary(km, times = c(6, 12, 24))
''', title="Kaplan-Meier 추정과 시점별 생존율")

c_ll = nb.cell('''
km_ll <- survfit(Surv(months, status) ~ 1, data = lung, conf.type = "log-log")
km_ll
summary(km_ll, times = c(6, 12, 24))
''', title="신뢰구간을 log-log 방식으로")

c_plot = nb.cell('''
plot(km_ll, xlab = "Months", ylab = "Survival probability",
     mark.time = TRUE, xaxt = "n")
axis(1, at = seq(0, 36, by = 6))
abline(h = 0.5, lty = 3, col = "gray50")
''', title="기본 그림으로 곡선 그리기", fig=(7, 4.2))

# c. 군 비교
c_sex = nb.cell('''
lung$sex_f <- factor(lung$sex, levels = 1:2, labels = c("Male", "Female"))
km_sex <- survfit(Surv(months, status) ~ sex_f, data = lung, conf.type = "log-log")
print(km_sex, rmean = 24)
''', title="성별 Kaplan-Meier")

c_tab = nb.cell('''
st  <- summary(km_sex)$table           # 군별 n, 사건 수, 중앙값과 구간
s12 <- summary(km_sex, times = 12)     # 12개월 생존율과 구간
data.frame(group = levels(lung$sex_f),
           n = st[, "n.max"], deaths = st[, "events"],
           `median (95% CI)` = sprintf("%.1f (%.1f-%.1f)",
                                       st[, "median"], st[, "0.95LCL"], st[, "0.95UCL"]),
           `1-yr % (95% CI)` = sprintf("%.1f (%.1f-%.1f)",
                                       100 * s12$surv, 100 * s12$lower, 100 * s12$upper),
           check.names = FALSE, row.names = NULL)
''', title="논문에 옮길 성별 요약표")

c_ggs = nb.cell('''
install.packages("survminer")
library(survminer)
ggsurvplot(km_sex, data = lung, conf.int = TRUE, pval = TRUE,
           risk.table = TRUE, break.time.by = 6,
           legend.title = "Sex", legend.labs = c("Male", "Female"),
           xlab = "Months", palette = c("#1f5f8b", "#c8553d"))
''', title="survminer로 곡선과 위험집합 표", fig=(7, 6))

c_lr = nb.cell('''
lr <- survdiff(Surv(months, status) ~ sex_f, data = lung)
lr
lr$pvalue
''', title="로그순위 검정")

c_rho = nb.cell('''
survdiff(Surv(months, status) ~ sex_f, data = lung, rho = 1)
''', title="초기 차이에 무게를 둔 검정 (rho = 1)")

c_ecog = nb.cell('''
lung$ecog <- factor(ifelse(lung$ph.ecog >= 2, "2+", lung$ph.ecog))
table(lung$ecog, useNA = "ifany")
survdiff(Surv(months, status) ~ ecog, data = lung)
pairwise_survdiff(Surv(months, status) ~ ecog, data = lung,
                  p.adjust.method = "bonferroni")
''', title="세 군 비교와 두 군씩 비교")

c_rmst = nb.cell('''
rm <- summary(km_sex, rmean = 24)$table[, c("rmean", "se(rmean)")]
rm
d  <- rm["sex_f=Female", "rmean"] - rm["sex_f=Male", "rmean"]
se <- sqrt(sum(rm[, "se(rmean)"]^2))
round(c(diff = d, lower = d - 1.96 * se, upper = d + 1.96 * se), 2)
''', title="24개월 제한평균생존시간(RMST)의 차이")

# d. 과제
c_t1 = nb.cell('''
km_ecog <- survfit(Surv(months, status) ~ ecog, data = lung, conf.type = "log-log")
st  <- summary(km_ecog)$table
s12 <- summary(km_ecog, times = 12)
data.frame(group = rownames(st), n = st[, "n.max"], deaths = st[, "events"],
           `median (95% CI)` = sprintf("%.1f (%.1f-%.1f)",
                                       st[, "median"], st[, "0.95LCL"], st[, "0.95UCL"]),
           `1-yr % (95% CI)` = sprintf("%.1f (%.1f-%.1f)",
                                       100 * s12$surv, 100 * s12$lower, 100 * s12$upper),
           check.names = FALSE, row.names = NULL)
''', title="과제 1 정답: ECOG 군별 요약표")

c_t2 = nb.cell('''
survdiff(Surv(months, status) ~ sex_f + strata(ecog), data = lung)
survdiff(Surv(months, status) ~ sex_f, data = lung, subset = !is.na(ecog))
''', title="과제 2 정답: ECOG로 층화한 로그순위 검정")

c_t3 = nb.cell('''
lung$dead_wrong <- as.integer(lung$status > 0)    # 실수: 모두 1이 된다
table(lung$dead_wrong)
survfit(Surv(months, dead_wrong) ~ 1, data = lung)
''', title="과제 3 정답: 사건 변수를 잘못 만들면")

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

# ---------------------------------------------------------------- 파이썬 실습 7과 대조
out = {c.n: c.text for c in nb.cells}
assert "228    165   10.2    9.33    11.9" in out[c_ll.n] and "0.3387        0.478" in out[c_ll.n]
assert "0.0676        0.178" in out[c_ll.n]
assert "19.3    16.8    33.2" in out[c_fu.n]
assert "8.9 (6.9-10.1)" in out[c_tab.n] and "14.0 (11.3-17.2)" in out[c_tab.n] and "33.6 (25.3-42.1)" in out[c_tab.n]
assert "Chisq= 10.3" in out[c_lr.n] and "Chisq= 12.7" in out[c_rho.n] and "Chisq= 19" in out[c_ecog.n]
assert "0.1890" in out[c_ecog.n]
assert "4.06" in out[c_rmst.n] and "1.91" in out[c_rmst.n] and "6.21" in out[c_rmst.n]
assert "Chisq= 11" in out[c_t2.n] and "Chisq= 10" in out[c_t2.n]

S = nb.save_fragment
S("rc02_load", nb.html(c_load, marks={"status": 1, "63 165": 2}))
S("rc02_surv", nb.html(c_surv, marks={"33.182752+": 1, "0   1": 2}))
S("rc02_fu", nb.html(c_fu, marks={"63   19.3": 1}))
S("rc02_km", nb.html(c_km, marks={"10.2    9.36    11.9": 1, "0.409": 2, "0.3447        0.486": 3}))
S("rc02_ll", nb.html(c_ll, marks={"9.33    11.9": 1, "0.3387        0.478": 2}))
S("rc02_plot", nb.html(c_plot))
S("rc02_sex", nb.html(c_sex, marks={"rmean*": 1, "8.87     6.9    10.1": 2}))
S("rc02_tab", nb.html(c_tab, marks={"8.9 (6.9-10.1)": 1, "33.6 (25.3-42.1)": 2}))
S("rc02_ggs", nb.html(c_ggs, marks={"myeloma": 1, '* colour : "Sex"': 2}))
S("rc02_lr", nb.html(c_lr, marks={"91.6": 1, "4.55      10.3": 2, "p= 0.001": 3, "0.001311165": 4}))
S("rc02_rho", nb.html(c_rho, marks={"70.4": 1, "Chisq= 12.7": 2}))
S("rc02_ecog", nb.html(c_ecog, marks={"<NA> \n  63  113   51    1": 1, "1 observation deleted due to missingness": 2,
                                      "Chisq= 19": 3, "0.1890": 4}))
S("rc02_rmst", nb.html(c_rmst, marks={"10.22281": 1, "4.06": 2}))
S("rc02_t1", nb.html(c_t1, marks={"6.0 (4.8-9.4)": 1}))
S("rc02_t2", nb.html(c_t2, marks={"Chisq= 11.1": 1, "Chisq= 10": 2}))
S("rc02_t3", nb.html(c_t3, marks={"8.39": 1}))
nb.save_ipynb("2장 생존 자료와 Kaplan-Meier (R과 통계분석)",
              intro="survival 패키지의 폐암 자료(lung)로 Kaplan-Meier 곡선을 그리고 군의 생존을 비교합니다. "
                    "보건통계학 기초의 파이썬 실습 7과 같은 자료입니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 생존 자료를 R에 알려 주기", c_km.n: "## b. Kaplan-Meier 곡선과 중앙생존기간",
                     c_sex.n: "## c. 군별 곡선과 요약표", c_lr.n: "## d. 로그순위 검정과 그 밖의 비교",
                     c_t1.n: "## e. 과제 정답"})
print("rc02 fragments and notebook written")
