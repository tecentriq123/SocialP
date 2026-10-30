"""R과 통계분석 6장 · 처치전환 보정 — R cells run for real with rkit (survival, rpsftm).

run:  source /home/claude/pylibs/env.sh && python3 gen/data_rc06.py && python3 gen/r_rc06.py   (NOMARK=1 … 로 모든 출력)

자료는 gen/data_rc06.py가 만든 pub/data/rcc_switch.csv(신약 A 대 표준요법 B, 군마다 300명, 대조군은 진행한 뒤 신약으로 전환 가능).
정답(전환이 없었을 때)은 gen/_rc06_truth.json: ψ = −0.3, 위험비 약 0.650, 28·36개월 RMST.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
TRUTH = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_rc06_truth.json")))
nb = RNotebook("rc06")

# a. 자료와 ITT
c_load = nb.cell('''
library(survival)
options(HTTPUserAgent = "Mozilla/5.0")
BASE <- "https://socialp-ajou.tecentriq12.workers.dev/data/"
d <- read.csv(paste0(BASE, "rcc_switch.csv"))
head(d, 3)
table(arm = d$arm, switch = d$switch)
''', title="전환이 있는 시험 자료")

c_who = nb.cell('''
prog_b <- subset(d, arm == 0 & pfs_event == 1)      # 진행한 대조군
table(ps_prog = prog_b$ps_prog, switch = prog_b$switch)
round(tapply(prog_b$switch, prog_b$ps_prog, mean), 2)
''', title="누가 전환했나")

c_itt = nb.cell('''
hr <- function(fit) round(exp(c(HR = unname(coef(fit)[1]), confint(fit)[1, ])), 3)
fit_itt <- coxph(Surv(os, os_event) ~ arm, data = d)
hr(fit_itt)
survdiff(Surv(os, os_event) ~ arm, data = d)$pvalue
''', title="배정대로 분석(ITT)")

# b. 단순한 방법
c_pp = nb.cell('''
d$os_c <- ifelse(d$switch == 1, d$switch_time, d$os)      # 전환한 시점에 중도절단
d$ev_c <- ifelse(d$switch == 1, 0, d$os_event)
hr(coxph(Surv(os_c, ev_c) ~ arm, data = d))
hr(coxph(Surv(os, os_event) ~ arm, data = subset(d, switch == 0)))   # 전환한 환자 빼기
''', title="전환한 환자를 중도절단하거나 빼면")

# c. RPSFTM
c_rx = nb.cell('''
d$rx <- ifelse(d$arm == 1, 1,
               ifelse(d$switch == 1, (d$os - d$switch_time) / d$os, 0))
round(tapply(d$rx, list(arm = d$arm, switch = d$switch), mean), 3)
''', title="신약을 받은 시간의 비율")

c_rp = nb.cell('''
install.packages("rpsftm")
library(rpsftm)
fit_rp <- rpsftm(Surv(os, os_event) ~ rand(arm, rx), data = d,
                 censor_time = cens_time)
fit_rp
fit_rp$CI
''', title="rpsftm()으로 ψ 추정")

c_rphr = nb.cell('''
d$os_rp <- ifelse(d$arm == 1, d$os, fit_rp$Sstar[, "time"])        # 대조군은 전환이 없었다면의 시간
d$ev_rp <- ifelse(d$arm == 1, d$os_event, fit_rp$Sstar[, "status"])
hr_rp <- hr(coxph(Surv(os_rp, ev_rp) ~ arm, data = d))
hr_rp
z_itt <- qnorm(1 - survdiff(Surv(os, os_event) ~ arm, data = d)$pvalue / 2)
se <- abs(log(hr_rp["HR"])) / z_itt                                # ITT의 p값을 지키는 구간
round(exp(log(hr_rp["HR"]) + c(-1.96, 1.96) * se), 3)
''', title="보정한 위험비와 신뢰구간")

c_rpkm = nb.cell('''
km_a  <- survfit(Surv(os, os_event) ~ 1, data = subset(d, arm == 1))
km_b  <- survfit(Surv(os, os_event) ~ 1, data = subset(d, arm == 0))
km_bc <- survfit(Surv(os_rp, ev_rp) ~ 1, data = subset(d, arm == 0))
plot(km_a, conf.int = FALSE, col = "#1f5f8b", xlab = "Months", ylab = "Overall survival",
     xlim = c(0, 42), xaxt = "n")
lines(km_b, conf.int = FALSE, col = "#c8553d")
lines(km_bc, conf.int = FALSE, col = "#c8553d", lty = 2)
axis(1, at = seq(0, 42, by = 6))
legend("bottomleft", c("Drug A", "Standard B (as randomised)", "Standard B (RPSFTM, no switching)"),
       col = c("#1f5f8b", "#c8553d", "#c8553d"), lty = c(1, 1, 2), bty = "n")
''', title="전환이 없었다면의 대조군 곡선", fig=(7, 4.3))

c_rpnr = nb.cell('''
fit_nr <- rpsftm(Surv(os, os_event) ~ rand(arm, rx), data = d)    # censor_time을 빼면
fit_nr$psi
d$os_nr <- ifelse(d$arm == 1, d$os, fit_nr$Sstar[, "time"])
d$ev_nr <- ifelse(d$arm == 1, d$os_event, fit_nr$Sstar[, "status"])
hr_nr <- hr(coxph(Surv(os_nr, ev_nr) ~ arm, data = d))
hr_nr
c(max_time_B = max(d$os_nr[d$arm == 0]),
  deaths_B_no_rc = sum(d$ev_nr[d$arm == 0]), deaths_B_rc = sum(d$ev_rp[d$arm == 0]))
''', title="다시 정한 중도절단 없이 맞추기")

# d. IPCW
c_ipw = nb.cell('''
d$id <- seq_len(nrow(d))
b <- subset(d, arm == 0 & pfs_event == 1)                         # 전환할 수 있었던 환자
sw_fit <- glm(switch ~ ps_prog + ecog + age, family = binomial, data = b)
round(summary(sw_fit)$coefficients, 3)
b$w <- 1 / (1 - predict(sw_fit, type = "response"))             # 전환하지 않을 확률의 역수
summary(b$w[b$switch == 0])
c(progressed = nrow(b), sum_of_weights = round(sum(b$w[b$switch == 0]), 1))
''', title="전환을 예측하는 모형과 무게")

c_ipwcox = nb.cell('''
cols <- c("id", "arm", "tstart", "tstop", "event", "w")
pre  <- transform(d, tstart = 0, tstop = pfs, w = 1,
                  event = ifelse(pfs_event == 1, 0L, os_event))      # 진행 전 구간
post <- transform(subset(d, pfs_event == 1 & switch == 0),
                  tstart = pfs, tstop = os, event = os_event, w = 1) # 진행 뒤, 전환한 환자는 없음
post$w[post$arm == 0] <- b$w[match(post$id[post$arm == 0], b$id)]
long <- rbind(pre[, cols], post[, cols])
fit_ipw <- coxph(Surv(tstart, tstop, event) ~ arm, data = long,
                 weights = w, cluster = id)
hr(fit_ipw)
''', title="무게를 준 Cox 모형")

# e. 2단계
c_ts = nb.cell('''
b$t_post <- b$os - b$pfs                                         # 진행 뒤 생존 시간
fit_ts <- survreg(Surv(t_post, os_event) ~ switch + ps_prog + ecog + age,
                  data = b, dist = "weibull")
round(summary(fit_ts)$table, 3)
af <- unname(exp(coef(fit_ts)["switch"]))                        # 전환이 진행 뒤 시간을 늘린 배수
af
''', title="두 단계 방법: 진행 뒤 생존의 AFT 모형")

c_ts2 = nb.cell('''
d$os_ts <- d$os; d$ev_ts <- d$os_event
i <- which(d$switch == 1)
d$os_ts[i] <- d$pfs[i] + (d$os[i] - d$pfs[i]) / af               # 전환하지 않았다면의 사망 시간
hr_ts <- hr(coxph(Surv(os_ts, ev_ts) ~ arm, data = d))
hr_ts
j <- which(d$arm == 0 & d$pfs_event == 1)                        # 진행한 B군 모두
c_star <- d$pfs[j] + (d$cens_time[j] - d$pfs[j]) / af            # 같은 방식으로 줄인 마감 시간
d$os_tr <- d$os_ts; d$ev_tr <- d$ev_ts
d$ev_tr[j] <- ifelse(d$os_ts[j] <= c_star, d$ev_ts[j], 0)
d$os_tr[j] <- pmin(d$os_ts[j], c_star)
hr_tr <- hr(coxph(Surv(os_tr, ev_tr) ~ arm, data = d))
hr_tr
c(events_lost = sum(d$ev_ts) - sum(d$ev_tr))
''', title="두 단계 방법: 대조군 시간 줄이기")

c_sum = nb.cell('''
res <- rbind(ITT            = hr(fit_itt),
             censor_switch  = hr(coxph(Surv(os_c, ev_c) ~ arm, data = d)),
             exclude_switch = hr(coxph(Surv(os, os_event) ~ arm, data = subset(d, switch == 0))),
             RPSFTM         = hr_rp,
             RPSFTM_no_rc   = hr_nr,
             IPCW          = hr(fit_ipw),
             two_stage      = hr_ts,
             two_stage_rc   = hr_tr)
res
''', title="방법별 위험비 모아 보기")

# f. 과제
c_t1 = nb.cell('''
fit_half <- rpsftm(Surv(os, os_event) ~ rand(arm, rx), data = d, censor_time = cens_time,
                   treat_modifier = ifelse(d$arm == 0, 0.5, 1))
fit_half$psi
d$os_h <- ifelse(d$arm == 1, d$os, fit_half$Sstar[, "time"])
d$ev_h <- ifelse(d$arm == 1, d$os_event, fit_half$Sstar[, "status"])
hr(coxph(Surv(os_h, ev_h) ~ arm, data = d))
''', title="과제 1 정답: 전환한 환자의 효과가 절반이라면")

c_t2 = nb.cell('''
sw_fit2 <- glm(switch ~ ecog + age, family = binomial, data = b)     # ps_prog를 빼면
b$w2 <- 1 / (1 - predict(sw_fit2, type = "response"))
long$w2 <- long$w
long$w2[long$arm == 0 & long$tstart > 0] <- b$w2[match(long$id[long$arm == 0 & long$tstart > 0], b$id)]
hr(coxph(Surv(tstart, tstop, event) ~ arm, data = long, weights = w2, cluster = id))
''', title="과제 2 정답: 전환의 이유를 모르면")

c_t3 = nb.cell('''
c(max_time_RPSFTM_B = max(d$os_rp[d$arm == 0]))
rm28 <- function(time, event, arm) {
  t <- summary(survfit(Surv(time, event) ~ arm), rmean = 28)$table[, "rmean"]
  round(c(B = unname(t[1]), A = unname(t[2]), diff = unname(t[2] - t[1])), 2)
}
rbind(ITT          = rm28(d$os, d$os_event, d$arm),
      RPSFTM       = rm28(d$os_rp, d$ev_rp, d$arm),
      RPSFTM_no_rc = rm28(d$os_nr, d$ev_nr, d$arm),
      two_stage    = rm28(d$os_ts, d$ev_ts, d$arm))
''', title="과제 3 정답: 28개월 평균 생존의 차이")

nb.run()

if NOMARK:
    for c in nb.cells:
        print(f"===== cell {c.n} {c.title}\n{c.text}")
        for w in c.warns:
            print("WARN:", w)
        if c.error:
            print("ERR:", c.error)
        print("images:", len(c.images))
    print(TRUTH)
    sys.exit(0)

out = {c.n: c.text for c in nb.cells}
assert "0.706" in out[c_itt.n] and "0.950" in out[c_pp.n] and "0.616" in out[c_pp.n]
assert "-0.3357825" in out[c_rp.n]
assert "0.592" in out[c_rphr.n] and "0.656" in out[c_ipwcox.n] and "0.604" in out[c_ts2.n]
assert abs(TRUTH["hr_no_switch"] - 0.650) < 0.001
assert "-0.3425335" in out[c_rpnr.n] and "0.610" in out[c_rpnr.n] and "194.000" in out[c_rpnr.n]
assert "20.56" in out[c_t3.n]

S = nb.save_fragment
S("rc06_load", nb.html(c_load, marks={"cens_time": 1, "0 126 174": 2}))
S("rc06_who", nb.html(c_who, marks={"31 145": 1, "0.82": 2}))
S("rc06_itt", nb.html(c_itt, marks={"0.706": 1, "0.001041796": 2}))
S("rc06_pp", nb.html(c_pp, marks={"0.950": 1, "0.616": 2}))
S("rc06_rx", nb.html(c_rx, marks={"0.54": 1}))
S("rc06_rp", nb.html(c_rp, marks={"cox.zph": 1, "rx.Mean": 2, "Chisq= 0": 3, "psi: -0.3357825": 4, "exp(psi): 0.7147786": 5, "-0.5823587": 6}))
S("rc06_rphr", nb.html(c_rphr, marks={"0.592": 1, "0.433 0.810": 2}))
S("rc06_rpkm", nb.html(c_rpkm))
S("rc06_rpnr", nb.html(c_rpnr, marks={"-0.3425335": 1, "0.610": 2, "41.732": 3, "156.000": 4}))
S("rc06_ipw", nb.html(c_ipw, marks={"-2.412": 1, "8.168": 2, "sum_of_weights": 3}))
S("rc06_ipwcox", nb.html(c_ipwcox, marks={"0.656": 1}))
S("rc06_ts", nb.html(c_ts, marks={"0.370": 1, "1.447836": 2}))
S("rc06_ts2", nb.html(c_ts2, marks={"0.604": 1, "events_lost": 2}))
S("rc06_sum", nb.html(c_sum))
S("rc06_t1", nb.html(c_t1))
S("rc06_t2", nb.html(c_t2))
S("rc06_t3", nb.html(c_t3, marks={"29.88489": 1}))
nb.save_ipynb("6장 처치전환 보정 (R과 통계분석)",
              intro="대조군의 처치전환을 허용한 가상 시험(rcc_switch.csv)으로 ITT 분석, 단순한 방법, RPSFTM, IPCW, 두 단계 방법의 "
                    "전체생존 위험비를 비교합니다. Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_load.n: "## a. 전환이 있는 시험과 ITT 분석", c_pp.n: "## b. 단순한 방법이 틀리는 이유",
                     c_rx.n: "## c. RPSFTM", c_ipw.n: "## d. IPCW", c_ts.n: "## e. 두 단계 방법과 결과 모으기",
                     c_t1.n: "## f. 과제 정답"})
print("rc06 fragments and notebook written")
