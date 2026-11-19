"""R과 파이썬으로 하는 통계분석 10장 · 청구자료로 코호트 만들기와 불균형 분석 — R cells run for real with rkit.

run:  python3 gen/r_rc17.py      (NOMARK=1 python3 gen/r_rc17.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 14장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-19 이 장으로 옮기며 같은 일을 R로 다시 한다.
자료: 실습 14의 청구자료 모양 표 세 개(claims_person·visit·rx.csv, 14장 코호트의 1/30 표본; gen/data_lab14.py),
14장의 분석용 전체 코호트(sglt2_cohort_full.csv, 166,548명), 14장 a 절의 양성예측도, c 절의 가상 환자 P1,
e 절의 가상 자발적 보고 자료(gen/nums_ch14e.py, gen/disp_ch14e.R).
선정 흐름과 인원은 실습 14(파이썬)와, 발생률·위험비·중앙 추적기간은 14장 본문(gen/nums_ch14.py)과 대조한다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc17")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 표 잇기
c_load = nb.cell(f'''
B <- "{D}"
person <- read.csv(paste0(B, "claims_person.csv"))   # 한 줄 = 한 사람(자격)
visit  <- read.csv(paste0(B, "claims_visit.csv"))    # 한 줄 = 명세서 한 건
rx     <- read.csv(paste0(B, "claims_rx.csv"))       # 한 줄 = 처방 한 건
str(rx)
for (v in c("elig_start", "elig_end", "death_date")) person[[v]] <- as.Date(person[[v]])
visit$visit_date <- as.Date(visit$visit_date)       # 글자 → 날짜
rx$rx_date       <- as.Date(rx$rx_date)
sapply(list(person = person, visit = visit, rx = rx),
       function(d) c(rows = nrow(d), people = length(unique(d$pid))))
table(rx$drug)
''', title="청구 표 세 개를 불러와 한 줄의 단위 확인하기")

c_merge = nb.cell('''
inp <- subset(visit, setting == "I")                    # 입원 명세서만
a <- merge(person, inp, by = "pid")                     # 기본값: 두 표 모두에 있는 행만
b <- merge(person, inp, by = "pid", all.x = TRUE)       # 입원이 없는 사람도 남김
c(rows_inner = nrow(a), people_inner = length(unique(a$pid)),
  rows_left = nrow(b), people_left = length(unique(b$pid)))
first_inp <- aggregate(visit_date ~ pid, data = inp, FUN = min)   # 여러 행을 한 사람 한 줄로
head(first_inp, 3)
''', title="표를 이을 때 행이 사라지지 않게 하기")

# ------------------------------------------------------------------ b. 신규 사용자
c_index = nb.cell('''
study  <- subset(rx, drug %in% c("SGLT2i", "DPP4i"))
enroll <- subset(study, rx_date >= as.Date("2016-01-01") & rx_date <= as.Date("2021-12-31"))
idx <- aggregate(rx_date ~ pid, data = enroll, FUN = min)       # 등록 기간의 첫 처방일
names(idx)[2] <- "index_date"
first <- enroll[enroll$rx_date == idx$index_date[match(enroll$pid, idx$pid)], ]   # 첫 처방일의 처방
n_cls <- tapply(first$drug, first$pid, function(x) length(unique(x)))
coh <- idx
coh$arm  <- first$drug[match(coh$pid, first$pid)]
coh$both <- n_cls[as.character(coh$pid)] > 1                    # 두 계열을 같은 날 시작
m <- merge(study, idx, by = "pid")
m$before <- as.numeric(m$index_date - m$rx_date)                # 첫 처방일 며칠 전의 처방인가
coh$prevalent <- coh$pid %in% m$pid[m$before >= 1 & m$before <= 365]   # 세척 기간 365일
c(people = nrow(coh), both = sum(coh$both), prevalent = sum(coh$prevalent))
table(coh$arm)
''', title="첫 처방일과 세척 기간으로 기존 사용자 가려내기")

c_flags = nb.cell('''
coh <- merge(coh, person[, c("pid", "sex", "birth_year", "death_date")], by = "pid", all.x = TRUE)
coh$age   <- as.numeric(format(coh$index_date, "%Y")) - coh$birth_year
coh$minor <- coh$age < 18
v <- merge(visit, idx, by = "pid")
v$before <- as.numeric(v$index_date - v$visit_date)
base <- subset(v, before >= 0 & before <= 365)                  # 첫 처방일까지 365일
has_dx <- function(d, codes, cols = c("dx_main", "dx_sub1", "dx_sub2")) {
  pat <- paste0("^(", paste(codes, collapse = "|"), ")")       # 코드의 앞부분이 같으면
  Reduce(`|`, lapply(cols, function(cl) grepl(pat, d[[cl]])))
}
coh$esrd      <- coh$pid %in% base$pid[has_dx(base, c("N185", "Z49", "Z992"))]
coh$prior_hhf <- coh$pid %in% base$pid[base$setting == "I" & has_dx(base, "I50", "dx_main")]
coh$no_t2dm   <- !(coh$pid %in% v$pid[v$before >= 0 & has_dx(v, "E11")])
colSums(coh[, c("minor", "esrd", "prior_hhf", "no_t2dm")])
''', title="나이와 기저 기간의 상병 표시 만들기")

c_flow = nb.cell('''
steps <- c(no_t2dm   = "No type 2 diabetes code (E11)",
           prevalent = "Prior use of either class (365 d)",
           both      = "Both classes on index date",
           minor     = "Age < 18 years",
           esrd      = "ESRD or dialysis (365 d)",
           prior_hhf = "HF hospitalization (365 d)")
flow_table <- function(coh) {
  keep <- rep(TRUE, nrow(coh))
  flow <- data.frame(step = "Prescribed in 2016-2021", excluded = NA, remaining = nrow(coh))
  for (s in names(steps)) {
    out  <- keep & coh[[s]]                   # 이 단계에서 처음 빠지는 사람
    keep <- keep & !coh[[s]]
    flow <- rbind(flow, data.frame(step = steps[[s]], excluded = sum(out), remaining = sum(keep)))
  }
  list(flow = flow, keep = keep)
}
ft <- flow_table(coh)
print(ft$flow, row.names = FALSE)
table(coh$arm[ft$keep])
''', title="선정 기준을 차례로 적용하며 인원 세기")

# ------------------------------------------------------------------ c. PDC와 양성예측도
c_gap = nb.cell('''
rx1 <- data.frame(pt = "P1",                     # 14장 c 절의 가상 환자 한 명
  date = as.Date(c("2021-03-02", "2021-04-01", "2021-05-06", "2021-06-05", "2021-10-18", "2022-01-16")),
  supply = c(30, 30, 30, 90, 90, 90))            # 공급일수
rx1 <- rx1[order(rx1$pt, rx1$date), ]
rx1$runout <- rx1$date + rx1$supply              # 약이 떨어지는 날
nxt <- ave(as.numeric(rx1$date), rx1$pt, FUN = function(x) c(x[-1], NA))   # 같은 환자의 다음 처방일
rx1$gap <- nxt - as.numeric(rx1$runout)          # 공백 일수
rx1[, c("date", "supply", "runout", "gap")]
stop_date <- function(d, grace) {                # 공백 > 유예기간인 첫 처방의 '떨어지는 날 + 유예기간'
  over <- is.na(d$gap) | d$gap > grace
  d$runout[over][1] + grace
}
sapply(c(grace30 = 30, grace60 = 60, grace90 = 90), function(g) format(stop_date(rx1, g)))
win_end <- min(rx1$date) + 365
covered <- sum(pmax(0, as.numeric(pmin(rx1$runout, win_end) - rx1$date)))
c(covered = covered, PDC = round(covered / 365, 3))
''', title="처방 기록에서 공백, 중단일, PDC 구하기")

c_ppv = nb.cell('''
prop.test(1391, 1438, correct = FALSE)$conf.int    # Wilson(점수) 구간: 정의 B의 양성예측도
binom.test(1391, 1438)$conf.int                    # Clopper–Pearson 정확 구간
p <- 1391 / 1438
p + c(-1, 1) * qnorm(0.975) * sqrt(p * (1 - p) / 1438)   # Wald 구간
prop.test(171, 200, correct = FALSE)$conf.int      # 심부전 입원 정의 171/200
''', title="양성예측도의 신뢰구간")

# ------------------------------------------------------------------ d. 추적과 분석
c_end = nb.cell('''
event    <- as.Date(c("2019-05-01", NA, NA))        # 결과 사건일
death    <- as.Date(c(NA, "2020-02-10", NA))        # 사망일
data_end <- as.Date("2022-12-31")                   # 자료 종료일
pmin(event, death, data_end)                        # 후보에 NA가 하나라도 있으면 NA
pmin(event, death, data_end, na.rm = TRUE)          # 있는 날짜 가운데 가장 이른 날
''', title="추적이 끝나는 날을 고를 때의 NA")

c_full = nb.cell(f'''
library(survival)
full <- read.csv("{D}sglt2_cohort_full.csv")   # 14장의 분석용 코호트(time: 일, 유예 60일 as-treated)
dim(full)
range(full$time)
stopifnot(all(full$time > 0))                   # 0일 이하의 time이 있으면 여기서 멈춤
agg <- aggregate(cbind(n = 1, events = event, py = time / 365.25) ~ sglt2, data = full, FUN = sum)
agg$rate <- 1000 * agg$events / agg$py           # 1,000인년당 발생률
agg
1000 * poisson.test(320, agg$py[agg$sglt2 == 1])$conf.int   # SGLT2 억제제군의 정확 구간
poisson.test(agg$events[2:1], agg$py[2:1])$conf.int          # 발생률비(SGLT2 ÷ DPP-4)
''', title="전체 코호트의 발생률과 발생률비")

c_cox = nb.cell('''
f0 <- coxph(Surv(time, event) ~ sglt2, data = full)
fa <- coxph(Surv(time, event) ~ sglt2 + age + female + htn + ihd + hf + ckd + met + ins, data = full)
round(rbind(crude    = exp(c(coef(f0), confint(f0))),
            adjusted = exp(c(coef(fa)[1], confint(fa)[1, ]))), 4)
round(exp(coef(update(fa, ties = "breslow")))["sglt2"], 4)   # 동점 처리를 Breslow로
''', title="as-treated 분석의 위험비 (표 14-5)")

c_medfu = nb.cell('''
km <- survfit(Surv(time / 365.25, 1 - event) ~ 1, data = full)   # 사건과 중도절단을 뒤집은 곡선
quantile(km, 0.5)$quantile                                        # 역 Kaplan-Meier 중앙 추적기간(년)
median(full$time) / 365.25                                        # time의 단순 중앙값
''', title="역 Kaplan-Meier로 중앙 추적기간 구하기")

# ------------------------------------------------------------------ e. 불균형 분석
c_disp = nb.cell('''
disp <- function(a, b, c, d) {          # a: 그 약·그 사건, b: 그 약·다른 사건, c·d: 다른 약
  ror <- (a * d) / (b * c); se <- sqrt(1/a + 1/b + 1/c + 1/d)
  E   <- (a + b) * (a + c) / (a + b + c + d)                     # 기대 보고 수
  round(c(obs = a, E = E, ROR = ror, lo = exp(log(ror) - 1.96 * se), hi = exp(log(ror) + 1.96 * se),
          IC = log2((a + 0.5) / (E + 0.5)),
          IC025 = log2(qgamma(0.025, shape = a + 0.5, rate = E + 0.5))), 2)
}
rbind(sglt2_all = disp(60, 4940, 340, 994660),    # 배경: 보고 전체
      sglt2_dm  = disp(60, 4940, 180, 44820),     # 배경: 당뇨병약 보고
      dpp4_dm   = disp(40, 9960, 200, 39800),
      first_yr  = disp(4, 296, 23, 5677))         # 시판 첫해 말
chisq.test(matrix(c(4, 296, 23, 5677), 2, byrow = TRUE))$statistic                    # Yates 보정(기본)
chisq.test(matrix(c(4, 296, 23, 5677), 2, byrow = TRUE), correct = FALSE)$statistic
''', title="2×2 표로 ROR과 IC 구하기")

c_da = nb.cell('''
suppressPackageStartupMessages(library(pvda))
n_rep <- c(SGLT2 = 5000, DPP4 = 10000, OTHER_DM = 35000)   # 당뇨병약 보고 5만 건
n_dka <- c(SGLT2 = 60,   DPP4 = 40,    OTHER_DM = 140)
rep <- do.call(rbind, lapply(names(n_rep), function(g)
  data.frame(drug = g, event = rep(c("DKA", "other"), c(n_dka[g], n_rep[g] - n_dka[g])))))
rep$report_id <- seq_len(nrow(rep))                        # 한 줄 = 보고 하나의 약 하나·사건 하나
head(rep, 3)
out <- da(rep)$da_df                                       # 모든 약–사건 조합을 한꺼번에
as.data.frame(subset(out, event == "DKA",
  select = c(drug, obs, exp_rrr, ic2.5, ic, ror2.5, ror, ror97.5, prr)))
''', title="보고 단위 자료로 pvda의 da() 돌리기")

# ------------------------------------------------------------------ f. 과제
c_t1 = nb.cell('''
coh180 <- coh
coh180$prevalent <- coh180$pid %in% m$pid[m$before >= 1 & m$before <= 180]   # 세척 기간 180일
ft180 <- flow_table(coh180)
print(ft180$flow, row.names = FALSE)
table(coh180$arm[ft180$keep])
''', title="과제 1 정답: 세척 기간을 180일로 줄이면")

c_t2 = nb.cell('''
full$time_itt2  <- ifelse(is.na(full$time_itt),  full$time,  full$time_itt)    # 빈칸은 as-treated와 같음
full$event_itt2 <- ifelse(is.na(full$event_itt), full$event, full$event_itt)
fi <- coxph(Surv(time_itt2, event_itt2) ~ sglt2 + age + female + htn + ihd + hf + ckd + met + ins, data = full)
round(exp(c(coef(fi)[1], confint(fi)[1, ])), 3)
quantile(survfit(Surv(time_itt2 / 365.25, 1 - event_itt2) ~ 1, data = full), 0.5)$quantile
''', title="과제 2 정답: ITT 유사 분석")

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
    c_load: ['$ rx_date: chr', "rows    10306 54370 111882", "people  10306 10306  10306"],
    c_merge: ["1277         1148        10435        10306"],
    c_index: ["10156       131      4103", "8846   1310"],
    c_flags: ["14       194       234       156"],
    c_flow: ["HF hospitalization (365 d)      136      5552", "4524   1028"],
    c_gap: ["45", '"2021-10-03" "2022-06-15" "2022-07-15"', "315.000   0.863"],
    c_ppv: ["0.9568087 0.9753326", "0.9581256 0.9765059", "0.7995133 0.8971065"],
    c_end: ["[1] NA NA NA", '"2019-05-01" "2020-02-10" "2022-12-31"'],
    c_full: ["166548     13", "30826    320  57781.48  5.538106", "4.947876 6.179367", "0.3802291 0.4792739"],
    c_cox: ["0.4228 0.3772 0.4740", "0.6092 0.5427 0.6838"],
    c_medfu: ["1.79603", "1.752225"],
    c_disp: ["60 24.00  3.02  2.25  4.06  1.30  0.92", "3.620581", "5.500385"],
    c_da: ["SGLT2  60      24  0.92  1.30   2.25 3.02    4.06"],
    c_t1: ["HF hospitalization (365 d)      141      5856", "4812   1044"],
    c_t2: ["0.726  0.665  0.791", "3.498973"],
}
for c, vs in CHK.items():
    for v in vs:
        assert v in out[c.n], (c.title, v, out[c.n])

S = nb.save_fragment
S("rc17_load", nb.html(c_load, marks={'$ rx_date: chr': 1, "rows    10306 54370 111882": 2, "people  10306 10306  10306": 3}))
S("rc17_merge", nb.html(c_merge, marks={"1277         1148        10435        10306": 1}))
S("rc17_index", nb.html(c_index, marks={"10156       131      4103": 1, "8846   1310": 2}))
S("rc17_flags", nb.html(c_flags, marks={"14       194       234       156": 1}))
S("rc17_flow", nb.html(c_flow, marks={"Prior use of either class (365 d)     4103      5897": 1, "4524   1028": 2}))
S("rc17_gap", nb.html(c_gap, marks={"2021-06-05     90 2021-09-03  45": 1, '"2021-10-03" "2022-06-15" "2022-07-15"': 2, "315.000   0.863": 3}))
S("rc17_ppv", nb.html(c_ppv, marks={"0.9568087 0.9753326": 1, "0.9581256 0.9765059": 2}))
S("rc17_end", nb.html(c_end, marks={"[1] NA NA NA": 1, '"2019-05-01" "2020-02-10" "2022-12-31"': 2}))
S("rc17_full", nb.html(c_full, marks={"[1]    1 2556": 1, "30826    320  57781.48  5.538106": 2, "4.947876 6.179367": 3, "0.3802291 0.4792739": 4}))
S("rc17_cox", nb.html(c_cox, marks={"0.4228 0.3772 0.4740": 1, "0.6092 0.5427 0.6838": 2}))
S("rc17_medfu", nb.html(c_medfu, marks={"1.79603": 1, "1.752225": 2}))
S("rc17_disp", nb.html(c_disp, marks={"60 24.00  3.02  2.25  4.06  1.30  0.92": 1, "first_yr    4  1.35  3.34  1.15  9.71  1.28 -0.45": 2, "3.620581": 3}))
S("rc17_da", nb.html(c_da, marks={"SGLT2  60      24  0.92  1.30   2.25 3.02    4.06": 1}))
S("rc17_t1", nb.html(c_t1, marks={"4812   1044": 1}))
S("rc17_t2", nb.html(c_t2, marks={"0.726  0.665  0.791": 1}))
nb.save_ipynb("10장 청구자료로 코호트 만들기와 불균형 분석 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 14장의 일을 R로 합니다. 청구자료 모양의 표 세 개(실습 14와 같은 1/30 표본)를 잇고, "
                    "신규 사용자 코호트의 선정 흐름을 만들고, 처방 기록에서 공백·중단일·PDC를 구한 뒤, 분석용 전체 코호트로 "
                    "발생률, 위험비, 중앙 추적기간을 구합니다. 마지막으로 자발적 보고 자료의 불균형 분석을 합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(\"pvda\")를 실행합니다(survival은 R에 들어 있습니다).",
              notes={c_load.n: "## a. 청구 표를 잇고 한 줄의 단위 확인하기", c_index.n: "## b. 신규 사용자 코호트와 선정 흐름",
                     c_gap.n: "## c. 공백, 중단일, PDC와 양성예측도", c_end.n: "## d. 추적 기간과 분석",
                     c_disp.n: "## e. 자발적 보고 자료의 불균형 분석", c_t1.n: "## f. 과제 정답"})
print("rc17 fragments and notebook written")
