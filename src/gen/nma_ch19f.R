# 19장 f 절 네트워크 메타분석 숫자 (netmeta 3.7-0, R 4.3.3). 먼저 python3 gen/nums_ch19f.py 로 ch19f_network.csv를 만든 뒤 Rscript gen/nma_ch19f.R
suppressMessages(library(netmeta))
d <- read.csv(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "ch19f_network.csv"))
p <- pairwise(treat = list(t1, t0), event = list(e1, e0), n = list(n1, n0), studlab = study, data = d, sm = "OR")
nm <- netmeta(p, reference.group = "P", method.tau = "REML", prediction = TRUE)
f <- function(te, lo, hi) sprintf("%.3f (%.3f-%.3f)", exp(te), exp(lo), exp(hi))
trt <- c("X","Y","Z")
cat("vs P random:\n"); for (t in trt) cat(t, f(nm$TE.random[t,"P"], nm$lower.random[t,"P"], nm$upper.random[t,"P"]), " PI", sprintf("%.3f-%.3f", exp(nm$lower.predict[t,"P"]), exp(nm$upper.predict[t,"P"])), " p", signif(nm$pval.random[t,"P"],3), "\n")
cat("vs P common:\n"); for (t in trt) cat(t, f(nm$TE.common[t,"P"], nm$lower.common[t,"P"], nm$upper.common[t,"P"]), "\n")
for (pr in list(c("X","Y"), c("Z","X"), c("Z","Y"))) cat(pr[1], "vs", pr[2], "random", f(nm$TE.random[pr[1],pr[2]], nm$lower.random[pr[1],pr[2]], nm$upper.random[pr[1],pr[2]]), " p", signif(nm$pval.random[pr[1],pr[2]],3), "\n")
cat("tau2", nm$tau2, "tau", nm$tau, "I2", nm$I2, "\n")
cat("Q", nm$Q, nm$df.Q, nm$pval.Q, " Qhet", nm$Q.heterogeneity, nm$df.Q.heterogeneity, nm$pval.Q.heterogeneity, " Qinc", nm$Q.inconsistency, nm$df.Q.inconsistency, nm$pval.Q.inconsistency, "\n")
ns <- netsplit(nm)
print(data.frame(comp = ns$random$comparison, k = ns$k, prop = round(ns$prop.random, 3), nma = round(exp(ns$random$TE),3),
  direct = round(exp(ns$direct.random$TE),3), dlo = round(exp(ns$direct.random$lower),3), dhi = round(exp(ns$direct.random$upper),3),
  indir = round(exp(ns$indirect.random$TE),3), ilo = round(exp(ns$indirect.random$lower),3), ihi = round(exp(ns$indirect.random$upper),3),
  RoR = round(exp(ns$compare.random$TE),3), p = round(ns$compare.random$p,3)))
r <- netrank(nm, small.values = "desirable"); print(round(r$ranking.random, 3))
set.seed(1); s <- netrank(nm, method = "SUCRA", nsim = 10000, small.values = "desirable"); print(round(s$ranking.random, 3))
rg <- rankogram(nm, nsim = 10000, small.values = "desirable"); print(round(rg$ranking.matrix.random, 3))
print(netleague(nm, common = FALSE, digits = 2, bracket = "(", separator = "-", seq = c("Z","X","Y","P")))
cat("n patients", sum(d$n1 + d$n0), " trials", nrow(d), "\n")
