"""R과 통계분석 9장 · 확률적 민감도 분석과 비용효과 수용곡선 — R cells run for real with rkit (base R, ggplot2).

run:  source /home/claude/pylibs/env.sh && python3 gen/r_rc09.py      (NOMARK=1 … 로 모든 출력)

보건통계학 기초 24장과 파이썬 실습 24의 공통 예시 민감도 분석을 R로 만든다. 결정론적 부분(입력값 범위, 일원 민감도 분석의 ICER,
적률법 모수)은 실습 24와 숫자가 같고, 확률적 부분은 난수 생성기가 달라 몬테카를로 오차 범위 안에서 비슷한지 본다
(실습 24: 비용효과적일 확률 0.1604, 표준오차 0.0052, 다시 뽑은 비율 2.7%).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc09")

# a. 모형과 입력값의 분포
c_model = nb.cell('''
weib_lam <- function(median, gam) log(2) / median^gam
S <- function(lam, gam, t) exp(-lam * t^gam)
mid <- function(x) (head(x, -1) + tail(x, -1)) / 2
p <- list(pfs_gam = 0.95, pfs_lam = weib_lam(10, 0.95), os_gam = 1.15, os_lam = weib_lam(28, 1.15),
          hr_pfs = 0.65, hr_os = 0.75, u_pf = 0.78, u_pd = 0.62, du_ae_A = 0.012, du_ae_B = 0.008,
          c_drug_A = 190, c_drug_B = 120, c_pf = 40, c_pd = 250, c_death = 800,
          c_ae_A = 120, c_ae_B = 80, disc = 0.045)
run_model <- function(p, horizon = 240) {           # 7장의 분할생존모형을 줄인 것
  t <- 0:horizon
  disc <- 1 / (1 + p$disc)^((head(t, -1) + 0.5) / 12)
  arm <- function(h_pfs, h_os, c_drug, du_ae, c_ae) {
    s_os <- S(p$os_lam * h_os, p$os_gam, t)
    s_pf <- pmin(S(p$pfs_lam * h_pfs, p$pfs_gam, t), s_os)
    pf <- mid(s_pf); pd <- mid(s_os - s_pf); died <- head(s_os, -1) - tail(s_os, -1)
    c(cost = sum((pf * (c_drug + p$c_pf) + pd * p$c_pd + died * p$c_death) * disc) + c_ae,
      qaly = sum((pf * p$u_pf + pd * p$u_pd) / 12 * disc) - du_ae)
  }
  d <- arm(p$hr_pfs, p$hr_os, p$c_drug_A, p$du_ae_A, p$c_ae_A) -
       arm(1, 1, p$c_drug_B, p$du_ae_B, p$c_ae_B)
  c(d_cost = unname(d["cost"]), d_qaly = unname(d["qaly"]), icer = unname(d["cost"] / d["qaly"]))
}
threshold <- 5000
base <- run_model(p)
round(c(base, inmb = threshold * base[["d_qaly"]] - base[["d_cost"]]), 4)
''', title="모형 함수와 기준 분석")

c_io = nb.cell('''
get_input <- function(p, key) {                 # 곡선은 중앙값(_med)으로 읽는다
  if (grepl("_med$", key)) {
    k <- sub("_med$", "", key)
    return((log(2) / p[[paste0(k, "_lam")]])^(1 / p[[paste0(k, "_gam")]]))
  }
  p[[key]]
}
set_input <- function(p, key, value) {          # 사본에서 값 하나를 바꾼다
  if (grepl("_med$", key)) {                     # 중앙값을 바꾸고 모양은 그대로
    k <- sub("_med$", "", key)
    p[[paste0(k, "_lam")]] <- weib_lam(value, p[[paste0(k, "_gam")]])
  } else if (grepl("_gam$", key)) {              # 모양을 바꾸고 중앙값은 그대로
    k <- sub("_gam$", "", key)
    med <- get_input(p, paste0(k, "_med"))
    p[[key]] <- value
    p[[paste0(k, "_lam")]] <- weib_lam(med, value)
  } else p[[key]] <- value
  p
}
round(c(pfs_med = get_input(p, "pfs_med"), os_med = get_input(p, "os_med")), 4)
q1 <- set_input(p, "os_gam", 1.29)               # 중앙값을 지키며 모양만 바꿈
q2 <- modifyList(p, list(os_gam = 1.29))         # 모양 값만 바꿈
round(rbind(set_input = c(med = get_input(q1, "os_med"), icer = run_model(q1)[["icer"]]),
            value_only = c(med = get_input(q2, "os_med"), icer = run_model(q2)[["icer"]])), 1)
''', title="곡선을 중앙값과 모양 모수로 읽고 바꾸기")

c_dist = nb.cell('''
spec <- data.frame(
  key  = c("hr_pfs", "hr_os", "pfs_med", "pfs_gam", "os_med", "os_gam", "u_pf", "u_pd",
           "du_ae_A", "du_ae_B", "c_pf", "c_pd", "c_death", "c_ae_A", "c_ae_B"),
  dist = c(rep("lognormal", 6), "beta", "beta", rep("gamma", 7)),
  u    = c(0.113, 0.131, 0.07, 0.05, 0.08, 0.06, 0.03, 0.05, 0.25, 0.25, 0.20, 0.20, 0.20, 0.25, 0.25))
beta_ab  <- function(m, se) { n <- m * (1 - m) / se^2 - 1; c(a = m * n, b = (1 - m) * n) }
gamma_ks <- function(m, se) { k <- (m / se)^2; c(shape = k, scale = m / k) }
nrow(spec)
round(beta_ab(0.78, 0.03), 2)          # 무진행 효용 0.78 (표준오차 0.03)
gamma_ks(250, 250 * 0.20)              # 진행 상태 비용 250만원 (표준오차 50만원)
''', title="입력값의 분포와 적률법")

# b. 일원 민감도 분석
c_rng = nb.cell('''
range95 <- function(p, key) {
  s <- spec[spec$key == key, ]; m <- get_input(p, key)
  switch(s$dist,
         lognormal = exp(log(m) + c(-1, 1) * qnorm(0.975) * s$u),
         beta  = { ab <- beta_ab(m, s$u); qbeta(c(0.025, 0.975), ab["a"], ab["b"]) },
         gamma = { ks <- gamma_ks(m, m * s$u); qgamma(c(0.025, 0.975), shape = ks["shape"], scale = ks["scale"]) })
}
ranges <- t(sapply(spec$key, function(k) c(base = get_input(p, k), range95(p, k))))
ranges <- rbind(ranges, c_drug_A = c(190, 190 * 0.8, 190),      # 20% 인하까지
                        c_drug_B = c(120, 120 * 0.8, 120 * 1.2))  # ±20%
colnames(ranges) <- c("base", "low", "high")
round(ranges, 3)
''', title="입력값별 범위 표")

c_ow = nb.cell('''
ow <- t(sapply(rownames(ranges), function(k)
  c(icer_low  = run_model(set_input(p, k, ranges[k, "low"]))[["icer"]],
    icer_high = run_model(set_input(p, k, ranges[k, "high"]))[["icer"]])))
ow <- cbind(ow, swing = abs(ow[, "icer_high"] - ow[, "icer_low"]))
ow <- ow[order(-ow[, "swing"]), ]
round(ow)
''', title="입력값을 하나씩 양 끝으로 바꿔 ICER 계산")

c_tor = nb.cell('''
top <- ow[10:1, ]                                 # 영향이 큰 10개, 큰 것이 위로
par(mar = c(4.5, 6, 1, 1))
plot(NA, xlim = range(top[, 1:2]), ylim = c(0.5, 10.5), yaxt = "n",
     xlab = "ICER (10,000 KRW per QALY)", ylab = "")
rect(pmin(top[, 1], base[["icer"]]), 1:10 - 0.35, pmax(top[, 1], base[["icer"]]), 1:10 + 0.35,
     col = "#9ec5e0", border = NA)                # 낮은 값으로 바꿨을 때
rect(pmin(top[, 2], base[["icer"]]), 1:10 - 0.35, pmax(top[, 2], base[["icer"]]), 1:10 + 0.35,
     col = "#e8a07c", border = NA)                # 높은 값으로 바꿨을 때
axis(2, at = 1:10, labels = rownames(top), las = 1, cex.axis = 0.85)
abline(v = c(base[["icer"]], threshold), lty = c(1, 2))
''', title="토네이도 그림", fig=(7, 4.6))

# c. 확률적 민감도 분석
c_draw = nb.cell('''
pairs <- list(c("pfs_med", "os_med"), c("pfs_gam", "os_gam"), c("hr_pfs", "hr_os"))
u <- setNames(spec$u, spec$key)
draw_curves <- function(p, rho = 0.5) {           # 짝마다 로그 척도에서 상관 rho로 함께 뽑기
  v <- list()
  for (pr in pairs) {
    z <- rnorm(2)
    z[2] <- rho * z[1] + sqrt(1 - rho^2) * z[2]
    for (j in 1:2) v[[pr[j]]] <- get_input(p, pr[j]) * exp(u[[pr[j]]] * z[j])
  }
  q <- modifyList(p, v[c("hr_pfs", "hr_os", "pfs_gam", "os_gam")])
  q$pfs_lam <- weib_lam(v$pfs_med, v$pfs_gam)
  q$os_lam  <- weib_lam(v$os_med,  v$os_gam)
  q
}
crossed <- function(q, t = 0:240)                 # 어느 군에서든 PFS가 OS를 넘으면 TRUE
  any(S(q$pfs_lam * q$hr_pfs, q$pfs_gam, t) > S(q$os_lam * q$hr_os, q$os_gam, t)) ||
  any(S(q$pfs_lam, q$pfs_gam, t) > S(q$os_lam, q$os_gam, t))
draw_inputs <- function(p, rho = 0.5, redraw = TRUE) {
  tries <- 0
  repeat {
    q <- draw_curves(p, rho); tries <- tries + 1
    if (!crossed(q) || !redraw) break
  }
  for (k in spec$key[spec$dist == "beta"]) {
    ab <- beta_ab(p[[k]], u[[k]]); q[[k]] <- rbeta(1, ab["a"], ab["b"])
  }
  for (k in spec$key[spec$dist == "gamma"]) {
    ks <- gamma_ks(p[[k]], p[[k]] * u[[k]]); q[[k]] <- rgamma(1, shape = ks["shape"], scale = ks["scale"])
  }
  attr(q, "tries") <- tries
  q
}
set.seed(20261002)
q <- draw_inputs(p)
round(unlist(q[c("pfs_lam", "os_lam", "hr_pfs", "hr_os", "u_pf", "c_pd", "c_drug_A")]), 4)
round(run_model(q), 3)
''', title="입력값 한 벌 뽑기")

c_psa = nb.cell('''
run_psa <- function(p, n = 5000, seed = 20261002, rho = 0.5, redraw = TRUE) {
  set.seed(seed)
  res <- matrix(NA, n, 2, dimnames = list(NULL, c("d_cost", "d_qaly")))
  tries <- 0
  for (i in 1:n) {
    q <- draw_inputs(p, rho, redraw)
    tries <- tries + attr(q, "tries")
    res[i, ] <- run_model(q)[c("d_cost", "d_qaly")]
  }
  list(dc = res[, "d_cost"], dq = res[, "d_qaly"], tries = tries)
}
elapsed <- system.time(psa <- run_psa(p))["elapsed"]
c(within_30s = unname(elapsed < 30))
c(tries = psa$tries, redrawn = psa$tries - 5000, share = round((psa$tries - 5000) / psa$tries, 4))
''', title="5,000벌 돌리기")

c_sum = nb.cell('''
dc <- psa$dc; dq <- psa$dq
inmb <- threshold * dq - dc
summ <- rbind(d_cost = c(mean(dc), quantile(dc, c(0.025, 0.975))),
              d_qaly = c(mean(dq), quantile(dq, c(0.025, 0.975))),
              inmb   = c(mean(inmb), quantile(inmb, c(0.025, 0.975))))
colnames(summ)[1] <- "mean"
round(summ, 3)
p_ce <- mean(inmb > 0)
round(c(icer_of_means = mean(dc) / mean(dq), p_ce = p_ce,
        mc_se = sqrt(p_ce * (1 - p_ce) / length(dc))), 4)
''', title="결과 요약")

c_plane = nb.cell('''
plot(dq, dc, pch = 16, cex = 0.3, col = rgb(0.12, 0.37, 0.55, 0.25),
     xlab = "Incremental QALYs", ylab = "Incremental cost (10,000 KRW)")
abline(a = 0, b = threshold, lty = 2)
abline(h = 0, v = 0, col = "gray60")
points(base[["d_qaly"]], base[["d_cost"]], pch = 4, col = "#c8553d", cex = 1.6, lwd = 2)
table(quadrant = ifelse(dq > 0, ifelse(dc > 0, "NE", "SE"), ifelse(dc > 0, "NW", "SW")))
''', title="비용효과평면", fig=(6.5, 4.6))

# d. 수용곡선과 EVPI
c_ceac = nb.cell('''
p_ce_at <- function(lam) mean(lam * dq - dc > 0)
evpi_at <- function(lam) { nb <- lam * dq - dc; mean(pmax(nb, 0)) - max(mean(nb), 0) }
lams <- c(3000, 4000, 5000, 5500, 6000, 7000, 8000, 10000, 12000)
round(data.frame(threshold = lams, mean_inmb = sapply(lams, function(l) mean(l * dq - dc)),
                 p_ce = sapply(lams, p_ce_at), evpi = sapply(lams, evpi_at)), 4)
''', title="임계값별 확률과 EVPI")

c_ceplot = nb.cell('''
grid <- seq(0, 12000, by = 100)
curve_ce <- sapply(grid, p_ce_at)
fine <- seq(0, 12000, by = 10)
lam50 <- fine[which(sapply(fine, p_ce_at) >= 0.5)[1]]
lam50
plot(grid, curve_ce, type = "l", lwd = 2, col = "#1f5f8b", ylim = c(0, 1),
     xlab = "Threshold (10,000 KRW per QALY)", ylab = "Probability drug A is cost-effective")
abline(v = threshold, lty = 2, col = "gray50"); abline(h = 0.5, lty = 3, col = "gray50")
''', title="비용효과 수용곡선", fig=(7, 4.2))

# e. 과제
c_t1 = nb.cell('''
p_cut <- modifyList(p, list(c_drug_A = 175.39))      # 7장 과제 1의 임계 약값
round(run_model(p_cut)[["icer"]])
psa_cut <- run_psa(p_cut)
inmb_cut <- threshold * psa_cut$dq - psa_cut$dc
round(c(mean_inmb = mean(inmb_cut), p_ce = mean(inmb_cut > 0)), 4)
''', title="과제 1 정답: 임계 약값에서의 확률")

c_t2 = nb.cell('''
ok <- inmb > 0
round(t(sapply(c(500, 1000, 5000), function(n) {
  pr <- mean(ok[1:n]); se <- sqrt(pr * (1 - pr) / n)
  c(n = n, p_ce = pr, se = se, low = pr - 1.96 * se, high = pr + 1.96 * se)
})), 4)
sapply(1:3, function(s) { r <- run_psa(p, seed = s); mean(threshold * r$dq - r$dc > 0) })
''', title="과제 2 정답: 모의실험 횟수와 몬테카를로 오차")

c_t3 = nb.cell('''
psa0 <- run_psa(p, rho = 0)
c(share_redrawn = round((psa0$tries - 5000) / psa0$tries, 4),
  p_ce = mean(threshold * psa0$dq - psa0$dc > 0))
''', title="과제 3 정답: 곡선을 상관 없이 뽑으면")

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
assert "5620.6102" in out[c_model.n]
assert "19.5 6020.7" in out[c_io.n] and "28.0 5731.3" in out[c_io.n]
assert "147.94" in out[c_dist.n] and "41.73" in out[c_dist.n]
for v in ("0.521", "8.718", "0.719", "161.787"):
    assert v in out[c_rng.n], v

S = nb.save_fragment
S("rc09_model", nb.html(c_model, marks={"2872.2935": 1, "5620.6102": 2, "-317.1497": 3}))
S("rc09_io", nb.html(c_io, marks={"28.0 5731.3": 1, "19.5 6020.7": 2}))
S("rc09_dist", nb.html(c_dist, marks={"[1] 15": 1, "147.94": 2, "25": 3}))
S("rc09_rng", nb.html(c_rng, marks={"0.521": 1, "161.787": 2}))
S("rc09_ow", nb.html(c_ow, marks={"hr_os        5232      7922": 1, "c_drug_A     4006": 2, "c_drug_B     6289      4952": 3}))
S("rc09_tor", nb.html(c_tor))
S("rc09_draw", nb.html(c_draw, marks={"0.6976": 1, "190.0000": 2}))
S("rc09_psa", nb.html(c_psa, marks={"TRUE": 1, "  126.0000": 2}))
S("rc09_sum", nb.html(c_sum, marks={"2956.722": 1, "0.531": 2, "-302.012": 3, "0.1628": 4, "0.0052": 5}))
S("rc09_plane", nb.html(c_plane, marks={"4972": 1, "21": 2}))
S("rc09_ceac", nb.html(c_ceac, marks={"0.1628  36.3905": 1, "0.4370 141.8953": 2, "0.9796": 3}))
S("rc09_ceplot", nb.html(c_ceplot, marks={"[1] 5620": 1}))
S("rc09_t1", nb.html(c_t1, marks={"0.4978": 1}))
S("rc09_t2", nb.html(c_t2))
S("rc09_t3", nb.html(c_t3, marks={"0.0869": 1}))
nb.save_ipynb("9장 확률적 민감도 분석과 비용효과 수용곡선 (R과 통계분석)",
              intro="보건통계학 기초 24장의 공통 예시로 일원 민감도 분석과 토네이도 그림, 확률적 민감도 분석, 비용효과 수용곡선과 EVPI를 R로 만듭니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요.",
              notes={c_model.n: "## a. 모형과 입력값의 분포", c_rng.n: "## b. 일원 민감도 분석",
                     c_draw.n: "## c. 확률적 민감도 분석", c_ceac.n: "## d. 수용곡선과 EVPI",
                     c_t1.n: "## e. 과제 정답"})
print("rc09 fragments and notebook written")
