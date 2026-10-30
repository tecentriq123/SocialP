"""R과 통계분석 8장 · 상태전이(마르코프) 모형 — R cells run for real with rkit (base R 행렬 계산).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc08.py      (NOMARK=1 … 로 모든 출력)

보건통계학 기초 23장 b 절과 파이썬 실습 23 e 절의 마르코프 모형(무진행·진행·사망, 월 주기, 율에서 전이확률)을 R로 만든다.
끝의 대조 블록에서 실습 23의 숫자(월 율 0.0693, 표 23-7의 0.067·0.565·0.418, 행렬 0.933·0.0603·0.0067, 24주기 0.1895·0.3839·0.4266,
중앙 OS 28.0·34.8, ICER 5,858, 반주기 보정 없이 5,942, 군별 보정 5,723, 분할생존 5,621, 잘못된 변환 5,894, 결정수형 2,500)와 맞춘다.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc08")

# a. 준비
c_inp = nb.cell('''
weib_lam <- function(median, gam) log(2) / median^gam
p <- list(pfs_gam = 0.95, pfs_lam = weib_lam(10, 0.95), os_gam = 1.15, os_lam = weib_lam(28, 1.15),
          hr_pfs = 0.65, hr_os = 0.75, u_pf = 0.78, u_pd = 0.62, du_ae_A = 0.012, du_ae_B = 0.008,
          c_drug_A = 190, c_drug_B = 120, c_pf = 40, c_pd = 250, c_death = 800,
          c_ae_A = 120, c_ae_B = 80, disc = 0.045)            # 7장과 같은 입력값
mid <- function(x) (head(x, -1) + tail(x, -1)) / 2
arm_result <- function(p, a, pf_t, pd_t, half = TRUE) {   # 상태 비율(0-n개월) -> 한 군의 결과
  n <- length(pf_t) - 1
  pf <- if (half) mid(pf_t) else head(pf_t, -1)
  pd <- if (half) mid(pd_t) else head(pd_t, -1)
  alive <- pf_t + pd_t
  died <- head(alive, -1) - tail(alive, -1)
  disc <- 1 / (1 + p$disc)^((0:(n - 1) + 0.5) / 12)
  cost <- sum((pf * (p[[paste0("c_drug_", a)]] + p$c_pf) + pd * p$c_pd +
               died * p$c_death) * disc) + p[[paste0("c_ae_", a)]]
  qaly <- sum((pf * p$u_pf + pd * p$u_pd) / 12 * disc) - p[[paste0("du_ae_", a)]]
  c(ly_pf = sum(pf) / 12, ly_pd = sum(pd) / 12, ly = sum(pf + pd) / 12, qaly = qaly, cost = cost)
}
compare <- function(rA, rB) {
  tab <- cbind(A = rA, B = rB, diff = rA - rB)
  list(table = tab, icer = unname(tab["cost", "diff"] / tab["qaly", "diff"]))
}
t <- 0:240
S <- function(lam, gam, t) exp(-lam * t^gam)
psm_arm <- function(a) {                              # 7장의 분할생존모형
  h1 <- if (a == "A") p$hr_pfs else 1; h2 <- if (a == "A") p$hr_os else 1
  os <- S(p$os_lam * h2, p$os_gam, t); pf <- pmin(S(p$pfs_lam * h1, p$pfs_gam, t), os)
  arm_result(p, a, pf, os - pf)
}
psm <- compare(psm_arm("A"), psm_arm("B"))
round(psm$icer, 2)
''', title="입력값과 결과 계산 함수 (7장과 같음)")

# b. 율과 확률
c_rate = nb.cell('''
rate_to_prob <- function(r, t = 1) 1 - exp(-r * t)        # 율 -> 길이 t 기간의 확률
prob_to_rate <- function(pr, t = 1) -log(1 - pr) / t       # 확률 -> 율
r_exit <- log(2) / 10                                      # 중앙 무진행생존 10개월 -> 월 율
p_year  <- rate_to_prob(r_exit, 12)
p_month <- rate_to_prob(r_exit)
round(c(rate = r_exit, p_year = p_year, p_month = p_month), 4)
round(rbind(month_from_year = c(right = rate_to_prob(prob_to_rate(p_year, 12)), wrong = p_year / 12),
            year_from_month = c(right = 1 - (1 - p_month)^12, wrong = p_month * 12),
            apply_hr        = c(right = rate_to_prob(r_exit * p$hr_pfs, 12), wrong = p_year * p$hr_pfs)), 4)
''', title="율과 확률 바꾸기 (23장 표 23-7)")

# c. 전이확률 행렬과 코호트 추적
c_mat = nb.cell('''
tmat <- function(r_prog, r_dpf, r_dpd) {       # 월 율 세 개 -> 한 달 전이확률 행렬
  leave <- rate_to_prob(r_prog + r_dpf)          # 무진행 상태를 떠날 확률
  to_pd <- leave * r_prog / (r_prog + r_dpf)     # 그 가운데 진행으로 가는 몫
  q <- rate_to_prob(r_dpd)                       # 진행 상태에서 사망할 확률
  matrix(c(1 - leave, to_pd, leave - to_pd,
           0,         1 - q, q,
           0,         0,     1), nrow = 3, byrow = TRUE,
         dimnames = list(from = c("pf", "pd", "dead"), to = c("pf", "pd", "dead")))
}
r_dpd <- 0.0470602                               # 진행 -> 사망의 월 율 (23장)
P_B <- tmat(0.9 * r_exit, 0.1 * r_exit, r_dpd)
P_A <- tmat(0.9 * r_exit * p$hr_pfs, 0.1 * r_exit * p$hr_pfs, r_dpd)   # 위험비는 율에 곱한다
round(P_B, 4)
round(P_A, 4)
rbind(B = rowSums(P_B), A = rowSums(P_A))
''', title="율에서 전이확률 행렬 만들기")

c_trace = nb.cell('''
trace <- function(P, n = 240) {                  # 모두 무진행에서 출발해 n주기 추적
  x <- matrix(0, n + 1, 3, dimnames = list(NULL, colnames(P)))
  x[1, ] <- c(1, 0, 0)
  for (k in 1:n) x[k + 1, ] <- x[k, ] %*% P      # 행렬 곱 한 번 = 한 달
  x
}
tr_B <- trace(P_B); tr_A <- trace(P_A)
dim(tr_B)
round(tr_B[c(0, 1, 2, 3, 12, 24) + 1, ], 3)
P24 <- diag(3)
for (k in 1:24) P24 <- P24 %*% P_B               # 행렬을 24번 곱한 것
round(rbind(power = P24[1, ], trace = tr_B[24 + 1, ]), 4)
''', title="코호트 추적 (표준요법 B)")

c_med = nb.cell('''
median_os <- function(P, n = 1200) {            # 생존 비율이 0.5가 되는 시점(개월)
  alive <- rowSums(trace(P, n)[, c("pf", "pd")])
  k <- which(alive <= 0.5)[1]                     # 처음으로 0.5 이하가 된 행 (k행 = k - 1개월)
  (k - 2) + (alive[k - 1] - 0.5) / (alive[k - 1] - alive[k])
}
round(c(B = median_os(P_B), A = median_os(P_A)), 1)
''', title="모형 속 중앙 전체생존")

# d. 결과
c_res = nb.cell('''
mk <- compare(arm_result(p, "A", tr_A[, "pf"], tr_A[, "pd"]),
              arm_result(p, "B", tr_B[, "pf"], tr_B[, "pd"]))
round(mk$table, 3)
round(mk$icer)
nohalf <- compare(arm_result(p, "A", tr_A[, "pf"], tr_A[, "pd"], half = FALSE),
                  arm_result(p, "B", tr_B[, "pf"], tr_B[, "pd"], half = FALSE))
round(c(ly_B = unname(nohalf$table["ly", "B"]), icer = nohalf$icer), 3)
''', title="마르코프 모형의 결과")

c_plot = nb.cell('''
os_psm <- cbind(B = S(p$os_lam, p$os_gam, t), A = S(p$os_lam * p$hr_os, p$os_gam, t))
os_mk  <- cbind(B = rowSums(tr_B[, 1:2]), A = rowSums(tr_A[, 1:2]))
matplot(t / 12, cbind(os_psm, os_mk), type = "l", lty = c(1, 1, 2, 2),
        col = c("#c8553d", "#1f5f8b"), xlim = c(0, 15), ylim = c(0, 1),
        xlab = "Years", ylab = "Overall survival")
legend("topright", c("B, Weibull (partitioned)", "A, Weibull (partitioned)",
                     "B, Markov", "A, Markov"),
       col = c("#c8553d", "#1f5f8b"), lty = c(1, 1, 2, 2), bty = "n")
''', title="두 모형의 전체생존 곡선", fig=(7, 4.2))

# e. 맞추기와 비교
c_cal = nb.cell('''
gap <- function(r, hr, target)                   # 진행 -> 사망 율이 r일 때 중앙 OS - 목표
  median_os(tmat(0.9 * r_exit * hr, 0.1 * r_exit * hr, r)) - target
med_B <- (log(2) / p$os_lam)^(1 / p$os_gam)
med_A <- (log(2) / (p$os_lam * p$hr_os))^(1 / p$os_gam)
r_B <- uniroot(gap, c(0.01, 1), hr = 1, target = med_B, tol = 1e-10)$root
r_A <- uniroot(gap, c(0.01, 1), hr = p$hr_pfs, target = med_A, tol = 1e-10)$root
round(c(med_B = med_B, r_B = r_B, med_A = med_A, r_A = r_A), 5)
tr_A2 <- trace(tmat(0.9 * r_exit * p$hr_pfs, 0.1 * r_exit * p$hr_pfs, r_A))
mk2 <- compare(arm_result(p, "A", tr_A2[, "pf"], tr_A2[, "pd"]), mk$table[, "B"])
row <- function(r) c(d_pf = r$table["ly_pf", "diff"], d_pd = r$table["ly_pd", "diff"],
                     d_cost = r$table["cost", "diff"], d_qaly = r$table["qaly", "diff"], icer = r$icer)
cmp <- rbind(Markov = row(mk), Markov_by_arm = row(mk2), Partitioned = row(psm))
cmp <- round(cmp, 3); cmp[, "icer"] <- round(cmp[, "icer"]); cmp
''', title="진행 후 사망을 군별로 맞추기와 두 구조의 비교")

# f. 과제
c_t1 = nb.cell('''
p_wrong <- p_year / 12                           # 동료가 넣은 한 달 확률
r_w <- prob_to_rate(p_wrong)
tB <- trace(tmat(0.9 * r_w, 0.1 * r_w, r_dpd))
tA <- trace(tmat(0.9 * r_w * p$hr_pfs, 0.1 * r_w * p$hr_pfs, r_dpd))
wrong <- compare(arm_result(p, "A", tA[, "pf"], tA[, "pd"]), arm_result(p, "B", tB[, "pf"], tB[, "pd"]))
round(c(p_month_wrong = p_wrong, p_month_right = p_month, median_pfs_B = log(2) / r_w,
        median_os_B = median_os(tmat(0.9 * r_w, 0.1 * r_w, r_dpd))), 4)
round(c(ly_B = unname(wrong$table["ly", "B"]), d_cost = unname(wrong$table["cost", "diff"]),
        d_qaly = unname(wrong$table["qaly", "diff"]), icer = wrong$icer), 3)
''', title="과제 1 정답: 1년 확률을 12로 나누면")

c_t2 = nb.cell('''
trace_tv <- function(hr, n = 240) {             # 무진행을 떠날 확률이 주기마다 다른 모형
  s <- S(p$pfs_lam * hr, p$pfs_gam, 0:n)          # 와이블 PFS
  q <- rate_to_prob(r_dpd)
  x <- matrix(0, n + 1, 3, dimnames = list(NULL, c("pf", "pd", "dead")))
  x[1, ] <- c(1, 0, 0)
  for (k in 1:n) {
    leave <- 1 - s[k + 1] / s[k]                  # k번째 주기에 무진행을 떠날 확률
    P <- matrix(c(1 - leave, 0.9 * leave, 0.1 * leave,
                  0, 1 - q, q,
                  0, 0, 1), nrow = 3, byrow = TRUE)
    x[k + 1, ] <- x[k, ] %*% P
  }
  x
}
tvB <- trace_tv(1); tvA <- trace_tv(p$hr_pfs)
tv <- compare(arm_result(p, "A", tvA[, "pf"], tvA[, "pd"]), arm_result(p, "B", tvB[, "pf"], tvB[, "pd"]))
round(tv$table[c("ly_pf", "ly_pd", "ly", "qaly", "cost"), ], 3)
round(tv$icer)
''', title="과제 2 정답: 시간에 따라 변하는 전이확률")

c_t3 = nb.cell('''
ae_tree <- function(p_ae = 0.30, rr = 0.60, p_hosp = 0.40, c_hosp = 700, c_out = 200,
                    dq_hosp = 0.07, dq_out = 0.02, c_proph = 60) {
  one <- function(p_e, add) {
    prob <- c(hosp = p_e * p_hosp, out = p_e * (1 - p_hosp), none = 1 - p_e)
    c(cost = sum(prob * (add + c(c_hosp, c_out, 0))), qaly = -sum(prob * c(dq_hosp, dq_out, 0)))
  }
  r <- rbind(none = one(p_ae, 0), proph = one(p_ae * rr, c_proph))
  d <- r["proph", ] - r["none", ]
  list(table = r, icer = unname(d["cost"] / d["qaly"]), inmb_5000 = unname(5000 * d["qaly"] - d["cost"]))
}
ae_tree()
''', title="과제 3 정답: 이상반응 예방 요법의 결정수형")

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
assert "5620.61" in out[c_inp.n]
assert "0.0693" in out[c_rate.n] and "0.0670" in out[c_rate.n] and "0.5647" in out[c_rate.n] and "0.4176" in out[c_rate.n]
assert "0.933 0.0603 0.0067" in out[c_mat.n]
assert "0.1895 0.3839 0.4266" in out[c_trace.n]
assert "28.0" in out[c_med.n] and "34.8" in out[c_med.n]
assert "[1] 5858" in out[c_res.n] and "5942" in out[c_res.n]
for v in ("5858", "5723", "5621"):
    assert v in out[c_cal.n], v
assert "5894" in out[c_t1.n]
assert "2500" in out[c_t3.n]

S = nb.save_fragment
S("rc08_inp", nb.html(c_inp, marks={"5620.61": 1}))
S("rc08_rate", nb.html(c_rate, marks={"0.0693": 1, "0.0670 0.0471": 2, "0.5647 0.8036": 3, "0.4176": 4}))
S("rc08_mat", nb.html(c_mat, marks={"0.933 0.0603 0.0067": 1, "0.9559": 2, "B  1  1    1": 3}))
S("rc08_trace", nb.html(c_trace, marks={"241   3": 1, "0.427": 2, "0.1895 0.3839 0.4266": 3}))
S("rc08_med", nb.html(c_med, marks={"28.0": 1, "34.8": 2}))
S("rc08_res", nb.html(c_res, marks={"[1] 5858": 1, "5942.282": 2}))
S("rc08_plot", nb.html(c_plot))
S("rc08_cal", nb.html(c_cal, marks={"0.04706": 1, "5723": 2, "5621": 3}))
S("rc08_t1", nb.html(c_t1, marks={"5894.238": 1}))
S("rc08_t2", nb.html(c_t2))
S("rc08_t3", nb.html(c_t3, marks={"2500": 1}))
nb.save_ipynb("8장 상태전이(마르코프) 모형 (R과 통계분석)",
              intro="보건통계학 기초 23장 b 절의 마르코프 모형(무진행, 진행, 사망)을 R의 행렬 계산으로 만들고, 7장의 분할생존모형과 비교합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_inp.n: "## a. 준비", c_rate.n: "## b. 율과 확률", c_mat.n: "## c. 전이확률 행렬과 코호트 추적",
                     c_res.n: "## d. 비용, QALY, ICER", c_cal.n: "## e. 맞추기와 두 구조의 비교",
                     c_t1.n: "## f. 과제 정답"})
print("rc08 fragments and notebook written")
