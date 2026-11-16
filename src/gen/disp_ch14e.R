# 14장 e 절: R check of the disproportionality numbers (hypothetical database).
# Run: Rscript gen/disp_ch14e.R
suppressMessages(library(pvda))
disp <- function(a, b, c, d) {
  N <- a + b + c + d
  ror <- (a * d) / (b * c); se_ror <- sqrt(1/a + 1/b + 1/c + 1/d)
  prr <- (a / (a + b)) / (c / (c + d)); se_prr <- sqrt(1/a - 1/(a + b) + 1/c - 1/(c + d))
  E <- (a + b) * (a + c) / N
  ic <- log2((a + 0.5) / (E + 0.5))
  ic_ci <- log2(qgamma(c(0.025, 0.975), shape = a + 0.5, rate = E + 0.5))
  chi <- suppressWarnings(chisq.test(matrix(c(a, b, c, d), 2, byrow = TRUE)))$statistic   # Yates (default)
  chi0 <- suppressWarnings(chisq.test(matrix(c(a, b, c, d), 2, byrow = TRUE), correct = FALSE))$statistic
  round(c(E = E, ROR = ror, ROR_lo = exp(log(ror) - 1.96 * se_ror), ROR_hi = exp(log(ror) + 1.96 * se_ror),
          PRR = prr, PRR_lo = exp(log(prr) - 1.96 * se_prr), PRR_hi = exp(log(prr) + 1.96 * se_prr),
          chi2_yates = unname(chi), chi2 = unname(chi0), IC = ic, IC025 = ic_ci[1], IC975 = ic_ci[2]), 2)
}
tabs <- list(sglt2_all = c(60, 4940, 340, 994660), sglt2_dm = c(60, 4940, 180, 44820),
             dpp4_all = c(40, 9960, 360, 989640), dpp4_dm = c(40, 9960, 200, 39800),
             first_year = c(4, 296, 23, 5677))
print(t(sapply(tabs, function(x) do.call(disp, as.list(x)))))
# pvda cross-check for the main table
x <- tabs$sglt2_dm; E <- (x[1] + x[2]) * (x[1] + x[3]) / sum(x)
print(ror(a = x[1], b = x[2], c = x[3], d = x[4]))
print(prr(obs = x[1], n_drug = x[1] + x[2], n_event_prr = x[3], n_tot_prr = x[3] + x[4]))
print(ic(obs = x[1], exp = E))
y <- tabs$first_year; E1 <- (y[1] + y[2]) * (y[1] + y[3]) / sum(y)
print(ic(obs = y[1], exp = E1))
