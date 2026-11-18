"""R과 파이썬으로 하는 통계분석 5장 · 생존 자료와 Kaplan-Meier — R cells run for real with rkit (survival, survminer).

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
''', title="초기 차이에 가중치를 더 준 검정 (rho = 1)")

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
# e. 보건통계학 기초 7장의 예
c7_claims = nb.cell('''
d <- data.frame(pt = c("A", "B", "C", "D"),          # 가상의 환자 4명 (7장 a 절의 가·나·다·라)
  index   = as.Date(c("2021-02-15", "2021-06-01", "2022-01-10", "2022-03-20")),   # 첫 처방일
  last_rx = as.Date(c("2021-11-10", "2021-09-01", "2022-04-10", "2022-12-10")),   # 마지막 처방일
  supply  = c(90, 90, 30, 90),                                                     # 마지막 처방의 공급일수
  hhf     = as.Date(c("2022-01-20", "2022-04-18", NA, NA)),                        # 심부전 입원일
  death   = as.Date(c(NA, NA, "2022-05-02", NA)))                                  # 사망일
data_end <- as.Date("2022-12-31")                      # 자료 기간 종료
d$stop <- d$last_rx + d$supply + 60                    # 중단일 = 마지막 처방 + 공급일수 + 유예 60일
d$end_at  <- pmin(d$hhf, d$death, d$stop, data_end, na.rm = TRUE)   # as-treated: 가장 이른 날
d$end_itt <- pmin(d$hhf, d$death, data_end, na.rm = TRUE)           # ITT 유사: 중단은 끝이 아님
for (a in c("at", "itt")) {
  end <- d[[paste0("end_", a)]]
  d[[paste0("time_", a)]]   <- as.numeric(end - d$index)            # 날짜 차이(일)
  d[[paste0("status_", a)]] <- as.integer(!is.na(d$hhf) & end == d$hhf)   # 입원으로 끝나면 1
}
d[, c("pt", "stop", "end_at", "time_at", "status_at", "time_itt", "status_itt")]
round(colSums(d[, c("time_at", "time_itt")]) / 365.25, 2)           # 총 인년
''', title="청구 기록을 time과 status로 바꾸기")

c7_km = nb.cell('''
co <- read.csv("https://socialp-ajou.tecentriq12.workers.dev/data/ch07_cohort.csv")
km7 <- survfit(Surv(time, status) ~ drug, data = co, conf.type = "log-log")
km7
summary(km7, times = c(12, 24, 36))
''', title="약물 A·B 300명의 Kaplan-Meier (7장 b 절)")

c7_ci = nb.cell('''
survfit(Surv(time, status) ~ drug, data = co)          # 기본값 conf.type = "log"
survfit(Surv(time, status == 0) ~ 1, data = co)        # 역 Kaplan-Meier: 중앙 추적기간
''', title="신뢰구간 방법과 추적기간")

c7_lr12 = nb.cell('''
time12 <- c(3, 5, 8, 12, 12, 16, 20, 25, 28, 32, 38, 44)   # 7장 표 7-1의 12명 (개월)
st12   <- c(1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0)             # 1 = 사망
drugA  <- c(0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1)             # 약물 A = 1
lr12 <- survdiff(Surv(time12, st12) ~ drugA)
lr12
idx  <- combn(12, 6)                                        # A 이름표 6개를 붙이는 924가지
chis <- apply(idx, 2, function(i) {
  g <- seq_len(12) %in% i                                   # 이 배정에서 A인 환자
  survdiff(Surv(time12, st12) ~ g)$chisq
})
c(n = length(chis), n_ge = sum(chis >= lr12$chisq - 1e-9), p = mean(chis >= lr12$chisq - 1e-9))
''', title="12명의 로그순위 검정과 순열 P (7장 c 절)")

c7_lr300 = nb.cell('''
survdiff(Surv(time, status) ~ drug, data = co)                  # 층화하지 않음
survdiff(Surv(time, status) ~ drug + strata(imdc), data = co)   # IMDC 위험군으로 층화
''', title="300명의 로그순위 검정과 층화")

c7_rmst = nb.cell('''
k12 <- survfit(Surv(time12, st12) ~ 1)
print(k12, rmean = 36)                                  # 12명의 36개월 RMST (7장 b 절)
rm <- summary(km7, rmean = 48)$table[, c("rmean", "se(rmean)")]
dd <- rm["drug=A", "rmean"] - rm["drug=B", "rmean"]
se <- sqrt(sum(rm[, "se(rmean)"]^2))
round(c(diff = dd, lower = dd - 1.96 * se, upper = dd + 1.96 * se), 2)   # 48개월 RMST 차이
''', title="제한평균생존시간")

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

assert "339         1" in out[c7_claims.n] and "2.68     2.90" in out[c7_claims.n]
assert "32.4    21.7    39.7" in out[c7_km.n] and "19.3    15.7    23.7" in out[c7_km.n] and "0.718  0.0369        0.638        0.783" in out[c7_km.n]
assert "32.4      22    41.6" in out[c7_ci.n] and "51.4    48.5    56.2" in out[c7_ci.n]
assert "Chisq= 3.6" in out[c7_lr12.n] and "924" in out[c7_lr12.n] and "66" in out[c7_lr12.n] and "0.07142857" in out[c7_lr12.n]
assert "116.4      2.30      5.24" in out[c7_lr300.n] and "0.472      1.03" in out[c7_lr300.n]
assert "23.5      3.71" in out[c7_rmst.n] and "5.40" in out[c7_rmst.n] and "1.35" in out[c7_rmst.n] and "9.44" in out[c7_rmst.n]

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
S("rc02_c7claims", nb.html(c7_claims, marks={"339         1": 1, "2.68     2.90": 2}))
S("rc02_c7km", nb.html(c7_km, marks={"32.4    21.7    39.7": 1, "0.718  0.0369        0.638        0.783": 2}))
S("rc02_c7ci", nb.html(c7_ci, marks={"32.4      22    41.6": 1, "51.4    48.5    56.2": 2}))
S("rc02_c7lr12", nb.html(c7_lr12, marks={"Chisq= 3.6": 1, "0.07142857": 2}))
S("rc02_c7lr300", nb.html(c7_lr300, marks={"116.4      2.30      5.24": 1, "0.472      1.03": 2}))
S("rc02_c7rmst", nb.html(c7_rmst, marks={"23.5      3.71": 1, "9.44": 2}))
S("rc02_t1", nb.html(c_t1, marks={"6.0 (4.8-9.4)": 1}))
S("rc02_t2", nb.html(c_t2, marks={"Chisq= 11.1": 1, "Chisq= 10": 2}))
S("rc02_t3", nb.html(c_t3, marks={"8.39": 1}))
nb.save_ipynb("5장 생존 자료와 Kaplan-Meier (R과 파이썬으로 하는 통계분석)",
              intro="survival 패키지의 폐암 자료(lung)로 Kaplan-Meier 곡선을 그리고 군의 생존을 비교합니다. "
                    "보건통계학 기초의 파이썬 실습 7과 같은 자료입니다. e 절에서는 보건통계학 기초 7장의 예(청구 기록 4명, 12명, 약물 A·B 300명)를 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 생존 자료를 R에 알려 주기", c_km.n: "## b. Kaplan-Meier 곡선과 중앙생존기간",
                     c_sex.n: "## c. 군별 곡선과 요약표", c_lr.n: "## d. 로그순위 검정과 그 밖의 비교",
                     c7_claims.n: "## e. 보건통계학 기초 7장의 예", c_t1.n: "## f. 과제 정답"})
print("rc02 fragments and notebook written")
