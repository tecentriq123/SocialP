"""R과 통계분석 7장 · 분할생존모형 — R cells run for real with rkit (base R, flexsurv, ggplot2).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc07.py      (NOMARK=1 … 로 모든 출력)

보건통계학 기초 23장 c 절과 파이썬 실습 23의 공통 예시(신약 A 대 표준요법 B, 와이블 곡선과 위험비, 월 주기 240개, 반주기 보정, 할인 4.5%)를
R로 만든다. 끝의 대조 블록에서 실습 23의 숫자(B 생존연수 1.255·1.799·3.054, B 총비용 7,689.5·QALY 1.874, 증분 2,872.3·0.511,
ICER 5,621, 순금전편익 −317.1, 적합한 와이블 5,610, 여섯 곡선 5,281–6,015, 임계 약값 175.39)와 맞춘다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc07")

# a. 입력값과 곡선
c_inp = nb.cell('''
weib_lam <- function(median, gam) log(2) / median^gam   # S(중앙값) = 0.5가 되는 lam
p <- list(
  pfs_gam = 0.95, pfs_lam = weib_lam(10, 0.95),   # 표준요법 B: 중앙 PFS 10개월
  os_gam  = 1.15, os_lam  = weib_lam(28, 1.15),   #              중앙 OS 28개월
  hr_pfs = 0.65, hr_os = 0.75,                    # 신약 A의 위험비
  u_pf = 0.78, u_pd = 0.62, du_ae_A = 0.012, du_ae_B = 0.008,
  c_drug_A = 190, c_drug_B = 120, c_pf = 40, c_pd = 250, c_death = 800,
  c_ae_A = 120, c_ae_B = 80,                      # 만원
  disc = 0.045)                                   # 연 할인율
length(p)
round(c(pfs_lam = p$pfs_lam, os_lam = p$os_lam), 5)
''', title="입력값 목록")

c_curve = nb.cell('''
S <- function(lam, gam, t) exp(-lam * t^gam)       # 와이블 생존함수
t <- 0:240                                         # 주기 경계: 0, 1, ..., 240개월
pfs_B <- S(p$pfs_lam, p$pfs_gam, t)
os_B  <- S(p$os_lam,  p$os_gam,  t)
pfs_A <- S(p$pfs_lam * p$hr_pfs, p$pfs_gam, t)     # 위험비는 lam에 곱한다
os_A  <- S(p$os_lam  * p$hr_os,  p$os_gam,  t)
show <- c(0, 12, 24, 36, 60, 120, 240)
round(data.frame(month = show, PFS_B = pfs_B[show + 1], OS_B = os_B[show + 1],
                 PFS_A = pfs_A[show + 1], OS_A = os_A[show + 1]), 4)
''', title="군별 두 곡선")

# b. 상태 비율
c_occ = nb.cell('''
occupancy <- function(pfs, os) {
  pf <- pmin(pfs, os)                    # PFS가 OS를 넘지 않게
  data.frame(pf = pf, pd = os - pf, dead = 1 - os)
}
occ_B <- occupancy(pfs_B, os_B)
occ_A <- occupancy(pfs_A, os_A)
dim(occ_B)
round(unlist(occ_B[24 + 1, ]), 4)        # 24개월(25번째 행)
''', title="두 곡선에서 상태 비율 만들기")

c_area = nb.cell('''
library(ggplot2)
library(tidyr)
occ <- rbind(cbind(arm = "Standard therapy B", month = t, occ_B),
             cbind(arm = "Drug A", month = t, occ_A))
occ_long <- pivot_longer(occ, c(pf, pd, dead), names_to = "state", values_to = "prop")
occ_long$state <- factor(occ_long$state, levels = c("dead", "pd", "pf"),
                         labels = c("Dead", "Progressed", "Progression-free"))
ggplot(subset(occ_long, month <= 120), aes(month, prop, fill = state)) +
  geom_area() + facet_wrap(~ arm) +
  scale_fill_manual(values = c("grey85", "#e08a3c", "#1f5f8b")) +
  labs(x = "Months", y = "Proportion of cohort", fill = NULL) +
  theme_minimal()
''', title="상태 비율의 쌓은 면적 그림", fig=(9, 3.6))

c_half = nb.cell('''
mid <- function(x) (head(x, -1) + tail(x, -1)) / 2   # 주기 시작과 끝의 평균
pf_B <- mid(occ_B$pf); pd_B <- mid(occ_B$pd)
length(pf_B)
round(c(pf = sum(pf_B), pd = sum(pd_B), alive = sum(pf_B + pd_B)) / 12, 3)
round(sum(head(occ_B$pf + occ_B$pd, -1)) / 12, 3)    # 보정 없이 주기 시작 값으로
''', title="반주기 보정과 상태별 평균 기간 (표준요법 B)")

# c. 비용, QALY, ICER
c_cyc = nb.cell('''
died_B <- head(os_B, -1) - tail(os_B, -1)            # 그 주기에 사망한 비율
disc_f <- 1 / (1 + p$disc)^((0:239 + 0.5) / 12)      # 주기 가운데 시점의 할인 계수
cyc <- data.frame(cycle = 1:240, pf = pf_B, pd = pd_B, died = died_B,
                  c_drug = pf_B * p$c_drug_B, c_pf = pf_B * p$c_pf,
                  c_pd = pd_B * p$c_pd, c_death = died_B * p$c_death,
                  qaly = (pf_B * p$u_pf + pd_B * p$u_pd) / 12, disc = disc_f)
round(cyc[c(1, 2, 12, 120, 240), ], 4)
''', title="주기별 비용과 QALY, 할인 계수 (표준요법 B)")

c_sumB = nb.cell('''
items <- c("c_drug", "c_pf", "c_pd", "c_death")
round(cbind(undiscounted = colSums(cyc[items]),
            discounted   = colSums(cyc[items] * cyc$disc)), 1)
cost_B <- sum(cyc[items] * cyc$disc) + p$c_ae_B
qaly_B <- sum(cyc$qaly * cyc$disc) - p$du_ae_B
round(c(cost_B = cost_B, qaly_B = qaly_B, qaly_B_undisc = sum(cyc$qaly) - p$du_ae_B), 3)
''', title="240주기를 더해 총비용과 QALY 구하기")

c_fun = nb.cell('''
run_psm <- function(p, horizon = 240, os_B = NULL) {
  t <- 0:horizon
  disc <- 1 / (1 + p$disc)^((head(t, -1) + 0.5) / 12)
  if (is.null(os_B)) os_B <- S(p$os_lam, p$os_gam, t)
  arm <- function(a) {
    hr_pfs <- if (a == "A") p$hr_pfs else 1
    hr_os  <- if (a == "A") p$hr_os  else 1
    s_os  <- os_B[1:(horizon + 1)]^hr_os                 # 비례위험: S_A = S_B^HR
    s_pfs <- pmin(S(p$pfs_lam * hr_pfs, p$pfs_gam, t), s_os)
    pf <- mid(s_pfs); pd <- mid(s_os - s_pfs)
    died <- head(s_os, -1) - tail(s_os, -1)
    q <- (pf * p$u_pf + pd * p$u_pd) / 12
    cost <- sum((pf * (p[[paste0("c_drug_", a)]] + p$c_pf) + pd * p$c_pd +
                 died * p$c_death) * disc) + p[[paste0("c_ae_", a)]]
    c(ly_pf = sum(pf) / 12, ly_pd = sum(pd) / 12, ly = sum(pf + pd) / 12,
      ly_d = sum((pf + pd) * disc) / 12,
      qaly = sum(q * disc) - p[[paste0("du_ae_", a)]],
      c_drug = sum(pf * p[[paste0("c_drug_", a)]] * disc), cost = cost)
  }
  res <- cbind(A = arm("A"), B = arm("B"))
  res <- cbind(res, diff = res[, "A"] - res[, "B"])
  list(table = res, d_cost = res["cost", "diff"], d_qaly = res["qaly", "diff"],
       icer = res["cost", "diff"] / res["qaly", "diff"])
}
base <- run_psm(p)
round(c(func = base$table["cost", "B"], step = cost_B), 6)
''', title="모형을 함수 하나로 묶기")

c_res = nb.cell('''
round(base$table, 3)
threshold <- 5000                                   # 이 예시에서 가정한 임계값
round(c(d_cost = base$d_cost, d_qaly = base$d_qaly, icer = base$icer,
        icer_ly = base$d_cost / base$table["ly_d", "diff"],
        inmb = threshold * base$d_qaly - base$d_cost), 4)
''', title="군별 결과와 ICER")

# d. 외삽 곡선
c_fit = nb.cell('''
library(flexsurv)
options(HTTPUserAgent = "Mozilla/5.0")
BASE <- "https://socialp-ajou.tecentriq12.workers.dev/data/"
trial <- read.csv(paste0(BASE, "rcc_trial.csv"))      # 4장의 표준요법 B 시험 자료
dists <- c(exp = "exp", weibull = "weibull", gompertz = "gompertz",
           lnorm = "lnorm", llogis = "llogis", gengamma = "gengamma")
os_curves <- lapply(dists, function(k) {
  f <- flexsurvreg(Surv(months, event) ~ 1, data = trial, dist = k)
  c(1, summary(f, t = 1:240, ci = FALSE, tidy = TRUE)$est)   # 0-240개월의 생존율
})
res6 <- t(sapply(dists, function(k) {
  r <- run_psm(p, os_B = os_curves[[k]])
  c(ly_B = unname(r$table["ly", "B"]), ly_A = unname(r$table["ly", "A"]),
    d_cost = r$d_cost, d_qaly = r$d_qaly, icer = r$icer)
}))
res6 <- round(res6, 3); res6[, "icer"] <- round(res6[, "icer"])
res6
''', title="외삽 곡선 여섯 개로 돌린 결과")

c_chk = nb.cell('''
same <- modifyList(p, list(hr_pfs = 1, hr_os = 1, c_drug_A = p$c_drug_B,
                           du_ae_A = p$du_ae_B, c_ae_A = p$c_ae_B))
r0 <- run_psm(same)
round(c(d_cost = r0$d_cost, d_qaly = r0$d_qaly), 10)            # 같은 입력이면 증분 0
round(run_psm(modifyList(p, list(disc = 0)))$table["qaly", "B"], 3)   # 할인 0%
round(run_psm(modifyList(p, list(u_pf = 1, u_pd = 1, du_ae_B = 0)))$table[c("ly", "qaly"), "B"], 3)
''', title="모형 점검")

# e. 과제
c_t1 = nb.cell('''
inmb_at <- function(price) threshold * run_psm(modifyList(p, list(c_drug_A = price)))$d_qaly -
                           run_psm(modifyList(p, list(c_drug_A = price)))$d_cost
price <- uniroot(inmb_at, c(100, 190))$root
round(c(price = price, cut_pct = 100 * (1 - price / p$c_drug_A),
        icer = run_psm(modifyList(p, list(c_drug_A = price)))$icer), 2)
''', title="과제 1 정답: ICER가 임계값과 같아지는 약값")

c_t2 = nb.cell('''
round(t(sapply(c(30, 36, 60, 120, 240), function(m) {
  r <- run_psm(p, horizon = m)
  c(months = m, d_cost = r$d_cost, d_qaly = r$d_qaly, icer = r$icer,
    share = r$d_qaly / base$d_qaly)
})), 3)
''', title="과제 2 정답: 분석기간에 따른 결과")

c_t3 = nb.cell('''
round(sapply(c(0.75, 0.70, 0.65), function(h) {
  r <- run_psm(modifyList(p, list(hr_os = h)))
  c(hr_os = h, d_cost = r$d_cost, d_qaly = r$d_qaly, icer = r$icer)
}), 3)
''', title="과제 3 정답: 전체생존 위험비가 달라지면")

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
assert "0.07777" in out[c_inp.n] and "0.01502" in out[c_inp.n]
assert "1.255" in out[c_half.n] and "1.799" in out[c_half.n] and "3.054" in out[c_half.n] and "3.095" in out[c_half.n]
assert "7689.485" in out[c_sumB.n] and "1.874" in out[c_sumB.n]
for v in ("2872.2935", "0.5110", "5620.6102", "-317.1497"):
    assert v in out[c_res.n], v
for v in ("5610", "5281", "6013"):
    assert v in out[c_fit.n], v
assert "175.39" in out[c_t1.n]

S = nb.save_fragment
S("rc07_inp", nb.html(c_inp, marks={"[1] 18": 1, "0.07777": 2}))
S("rc07_curve", nb.html(c_curve, marks={"0.4386": 1, "0.0003": 2}))
S("rc07_occ", nb.html(c_occ, marks={"241   3": 1, "0.2035": 2}))
S("rc07_area", nb.html(c_area))
S("rc07_half", nb.html(c_half, marks={"[1] 240": 1, "1.255": 2, "3.054": 3, "3.095": 4}))
S("rc07_cyc", nb.html(c_cyc, marks={"0.9982": 1, "0.6451": 2}))
S("rc07_sumB", nb.html(c_sumB, marks={"4628.6": 1, "7689.485": 2, "1.874": 3}))
S("rc07_fun", nb.html(c_fun, marks={"7689.485": 1}))
S("rc07_res", nb.html(c_res, marks={"3.916": 1, "2872.2935": 2, "5620.6102": 3, "-317.1497": 4}))
S("rc07_fit", nb.html(c_fit, marks={"Loading required package: survival": 1, "5610": 2, "6013": 3, "5281": 4}))
S("rc07_chk", nb.html(c_chk))
S("rc07_t1", nb.html(c_t1, marks={"175.39": 1}))
S("rc07_t2", nb.html(c_t2))
S("rc07_t3", nb.html(c_t3))
nb.save_ipynb("7장 분할생존모형 (R과 통계분석)",
              intro="보건통계학 기초 23장의 공통 예시(신약 A 대 표준요법 B)를 R로 만듭니다. 와이블 곡선과 위험비로 상태 비율을 만들고, "
                    "반주기 보정과 할인으로 비용과 QALY를 더해 ICER를 구한 뒤, 4장의 외삽 곡선을 바꿔 넣습니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_inp.n: "## a. 입력값과 곡선", c_occ.n: "## b. 상태 비율과 반주기 보정",
                     c_cyc.n: "## c. 비용, QALY, ICER", c_fit.n: "## d. 외삽 곡선 바꿔 넣기와 점검",
                     c_t1.n: "## e. 과제 정답"})
print("rc07 fragments and notebook written")
