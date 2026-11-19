"""R과 파이썬으로 하는 통계분석 12장 · 메타분석 — R cells run for real with rkit (meta 패키지).

run:  python3 gen/r_rc10.py      (NOMARK=1 python3 gen/r_rc10.py 로 돌리면 모든 셀의 출력을 찍는다)

자료는 보건통계학 기초 19장의 가상 메타분석(급성관상동맥증후군 시험 10개, pub/data/acs_trials.csv와
진단별 하위군 acs_trials_subgroup.csv)과 c 절의 가상 관찰연구 7개(보정 오즈비와 95% CI)다.
끝의 대조 블록에서 19장의 숫자와 맞춘다(gen/nums_ch19.py, 실습 19와 같은 값):
  고정효과 0.84 (0.78–0.91), 무작위효과 REML + HKSJ 0.82 (0.72–0.94) P 0.0102, 보통의 방법 P 0.003,
  예측구간 0.61–1.11(HKSJ 표준오차, 자유도 k − 1), Q 13.28 P 0.15, I² 32% (0–68%), τ² 0.0138 (0–0.128),
  τ² DL 0.0098 PM 0.0077, 하위군 STEMI 0.78 (0.69–0.88 보통의 방법) NSTE-ACS 0.92 (0.77–1.10),
  메타회귀 10%p당 0.91 (0.80–1.04) P 0.16, 관찰연구 7개 고정 1.48 (1.35–1.61) REML+HK 1.69 (1.17–2.43)
  예측구간 0.64–4.41, Harbord 0.45 Egger 0.47 Peters 0.54, 한 시험씩 빼기 0.79–0.85,
  편향 위험 낮음 8개 0.82 (0.71–0.96), 상대위험도 0.83 (0.73–0.95), 2상 0.73 3상 0.84 P 0.36.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc10")
URL = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# a. 자료
c_load = nb.cell(f'''
library(meta)
d <- read.csv("{URL}acs_trials.csv")
d$stemi10 <- d$stemi_pct / 10        # 메타회귀에 쓸 10%p 단위
d[, c("trial", "phase", "stemi_pct", "rob", "n_x", "mace_x", "n_pbo", "mace_pbo")]
''', title="시험 하나가 한 줄인 자료 불러오기")

c_obs = nb.cell('''
o <- data.frame(study = paste("Study", 1:7),
                or = c(1.15, 1.90, 1.45, 2.60, 1.05, 1.70, 3.10),
                lo = c(0.95, 1.55, 1.10, 1.90, 0.88, 1.30, 2.05),
                hi = c(1.39, 2.33, 1.91, 3.56, 1.25, 2.22, 4.69))
mo <- metagen(log(or), lower = log(lo), upper = log(hi), studlab = study,
              data = o, sm = "OR", method.tau = "REML",
              method.random.ci = "HK", prediction = TRUE, method.predict = "HK")
data.frame(study = o$study, logOR = round(mo$TE, 3), SE = round(mo$seTE, 3))
''', title="오즈비와 신뢰구간만 있는 연구 넣기")

# b. 두 모형과 방법
c_main = nb.cell('''
m <- metabin(mace_x, n_x, mace_pbo, n_pbo, studlab = trial, data = d,
             sm = "OR", method.tau = "REML", method.random.ci = "HK",
             prediction = TRUE, method.predict = "HK")
summary(m)
''', title="계획서대로 합치기: REML, HKSJ 구간, 예측구간")

c_forest = nb.cell('''
forest(m, sortvar = TE, prediction = TRUE, print.tau2 = TRUE,
       leftcols = c("studlab", "event.e", "n.e", "event.c", "n.c"),
       leftlabs = c("Trial", "Events", "Total", "Events", "Total"),
       label.e = "Drug X", label.c = "Placebo",
       label.left = "Favours drug X", label.right = "Favours placebo")
''', title="포레스트 플롯", fig=(10, 4.7))

c_default = nb.cell('''
m0 <- metabin(mace_x, n_x, mace_pbo, n_pbo, studlab = trial, data = d,
              sm = "OR", prediction = TRUE)            # 기본값 그대로
c(tau = m0$method.tau, ci = m0$method.random.ci, predict = m0$method.predict)
round(exp(c(lower = m0$lower.random, upper = m0$upper.random,
            pi.lower = m0$lower.predict, pi.upper = m0$upper.predict)), 3)
c(P.default = m0$pval.random, P.plan = m$pval.random)
''', title="기본값으로 돌리면 무엇이 달라지나")

c_combo = nb.cell('''
grid <- expand.grid(tau2 = c("DL", "PM", "REML"), ci = c("classic", "HK"),
                    stringsAsFactors = FALSE)
res <- t(sapply(seq_len(nrow(grid)), function(i) {
  x <- update(mo, method.tau = grid$tau2[i], method.random.ci = grid$ci[i])
  c(OR = exp(x$TE.random), lower = exp(x$lower.random),
    upper = exp(x$upper.random), P = x$pval.random, tau2 = x$tau2)
}))
cbind(grid, round(res, 4))
round(exp(c(common = mo$TE.common, lower = mo$lower.common, upper = mo$upper.common)), 2)
''', title="관찰연구 7개를 여러 방법으로 합치기")

c_adhoc = nb.cell('''
ma <- metabin(mace_x, n_x, mace_pbo, n_pbo, studlab = trial, data = d,
              sm = "OR", method.tau = "REML", method.random.ci = "HK",
              adhoc.hakn.ci = "se")      # q가 1보다 작으면 원래 표준오차를 씀
round(exp(c(OR = ma$TE.random, lower = ma$lower.random, upper = ma$upper.random)), 3)
round(ma$pval.random, 4)
''', title="q가 1보다 작을 때의 수정판")

# c. 이질성, 하위군, 메타회귀
c_het = nb.cell('''
round(c(Q = m$Q, df = m$df.Q, P = m$pval.Q), 4)
round(c(I2 = m$I2, lower = m$lower.I2, upper = m$upper.I2), 3)
round(c(tau2 = m$tau2, lower = m$lower.tau2, upper = m$upper.tau2, tau = m$tau), 4)
round(exp(c(pi.lower = m$lower.predict, pi.upper = m$upper.predict)), 3)
round(update(m, method.I2 = "tau2")$I2, 3)              # I²를 τ²로 구하면
sapply(c("DL", "PM", "REML"), function(t) round(update(m, method.tau = t)$tau2, 4))
''', title="이질성 지표를 꺼내 보기")

c_sub = nb.cell(f'''
sg <- read.csv("{URL}acs_trials_subgroup.csv")
ms <- metabin(mace_x, n_x, mace_pbo, n_pbo, studlab = trial, data = sg,
              sm = "OR", method.tau = "REML", method.random.ci = "HK",
              subgroup = subgroup, common = FALSE)
summary(ms)
''', title="진단별 하위군과 하위군 간 차이 검정")

c_rule = nb.cell('''
st <- metabin(mace_x, n_x, mace_pbo, n_pbo, studlab = trial,
              data = subset(sg, subgroup == "STEMI"), sm = "OR",
              method.tau = "REML", method.random.ci = "classic", common = FALSE)
round(c(tau2 = st$tau2, exp(c(OR = st$TE.random, lower = st$lower.random,
                               upper = st$upper.random))), 3)
''', title="τ²가 0인 하위군은 보통의 방법으로")

c_mr = nb.cell('''
mr <- metareg(m, ~ stemi10)
mr
round(exp(c(ratio = coef(mr)[["stemi10"]], lower = mr$ci.lb[2], upper = mr$ci.ub[2])), 3)
''', title="STEMI 비율로 메타회귀")

# d. 출판 편향과 민감도 분석
c_funnel = nb.cell('''
funnel(m)
''', title="깔때기 그림", fig=(7, 4.6))

c_bias = nb.cell('''
metabias(m)                     # 오즈비의 기본: Harbord 검정
round(sapply(c("Egger", "Harbord", "Peters"),
             function(b) metabias(m, method.bias = b)$pval), 3)
''', title="비대칭 검정")

c_sens = nb.cell('''
inf <- metainf(m, pooled = "random")
round(exp(range(inf$TE)), 3)                              # 한 시험씩 뺀 10번의 범위
low <- update(m, subset = rob == "low")                   # 편향 위험 '낮음'만
rr  <- update(m, sm = "RR")                               # 상대위험도로
est <- function(x) exp(c(OR = x$TE.random, lower = x$lower.random, upper = x$upper.random))
round(rbind(low_rob = est(low), risk_ratio = est(rr)), 3)
''', title="한 시험씩 빼기와 그 밖의 민감도 분석")

c_nnt = nb.cell('''
nnt(m, p.c = 0.084)             # p.c: 대조군의 위험(기저 위험)
''', title="NNT로 옮기기")

# e. 네트워크 메타분석 (보건통계학 기초 19장 f 절)
c_net_load = nb.cell(f'''
library(netmeta)
d2 <- read.csv("{URL}ch19f_network.csv")
d2[c(1, 11, 16, 18), ]                 # 비교마다 한 줄씩 보기
table(paste(d2$t1, "vs", d2$t0))      # 비교마다 시험 수
''', title="네트워크 자료 불러오기")

c_net = nb.cell('''
p  <- pairwise(treat = list(t1, t0), event = list(e1, e0), n = list(n1, n0),
               studlab = study, data = d2, sm = "OR")
nm <- netmeta(p, reference.group = "P", method.tau = "REML", common = FALSE)
nm
c(tau = netmeta(p, reference.group = "P")$method.tau)   # 적지 않으면
''', title="네트워크 전체를 한 번에 합치기")

c_split = nb.cell('''
ns <- netsplit(nm)
data.frame(comparison = ns$random$comparison, k = ns$k,
           direct.share = round(ns$prop.random, 2),
           direct = round(exp(ns$direct.random$TE), 2),
           indirect = round(exp(ns$indirect.random$TE), 2),
           P.diff = round(ns$compare.random$p, 2))
''', title="직접 근거와 간접 근거 나눠 보기")

c_rank = nb.cell('''
netrank(nm, small.values = "desirable")      # 작은 오즈비가 좋은 결과
''', title="순위(P-score)")

c_league = nb.cell('''
netleague(nm, common = FALSE, digits = 2, bracket = "(", separator = "-",
          seq = c("Z", "X", "Y", "P"))
''', title="리그 표")

c_graph = nb.cell('''
netgraph(nm, number.of.studies = TRUE, thickness = "number.of.studies",
         plastic = FALSE, col = "gray60", offset = 0.06)
''', title="네트워크 그림", fig=(6, 5))

# e. 과제 정답
c_t1 = nb.cell('''
mp <- update(m, subgroup = paste("Phase", phase), method.random.ci = "classic")
round(exp(c(phase2 = mp$TE.random.w[["Phase 2"]], phase3 = mp$TE.random.w[["Phase 3"]])), 2)
round(c(Q = mp$Q.b.random, P = mp$pval.Q.b.random), 3)
round(update(mp, method.random.ci = "HK")$pval.Q.b.random, 3)   # HKSJ 표준오차로 검정하면
''', title="과제 1 정답: 시험 단계로 나눈 하위군")

c_t2 = nb.cell('''
mf <- update(m, common = TRUE, random = FALSE)
round(exp(c(common = mf$TE.common, lower = mf$lower.common, upper = mf$upper.common)), 2)
round(exp(c(random = m$TE.random, lower = m$lower.random, upper = m$upper.random)), 2)
''', title="과제 2 정답: 고정효과로 바꾸면")

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

# ---------------------------------------------------------------- 19장, 실습 19와 대조
out = {c.n: c.text for c in nb.cells}
for v in ("0.8441 [0.7791; 0.9145]", "0.8208 [0.7151; 0.9421]", "0.0102", "[0.6086; 1.1070]",
          "tau^2 = 0.0138 [0.0000; 0.1278]", "I^2 = 32.3% [0.0%; 67.7%]", "13.28    9  0.1501"):
    assert v in out[c_main.n], v
assert "0.003" in out[c_default.n] and "0.605" in out[c_default.n]
for v in ("1.6864", "1.1715", "2.4277", "1.2807", "2.2096", "0.0127", "1.48", "1.35", "1.61"):
    assert v in out[c_combo.n], v
assert "0.706" in out[c_adhoc.n] and "0.954" in out[c_adhoc.n] and "0.0159" in out[c_adhoc.n]
assert "[1] 0.4" in out[c_het.n] and "0.0098" in out[c_het.n] and "0.0077" in out[c_het.n]
for v in ("0.7822", "0.9218 [0.7718; 1.1010]", "3.1%"):
    assert v in out[c_sub.n], v
assert "0.692" in out[c_rule.n] and "0.884" in out[c_rule.n]
assert "0.1573" in out[c_mr.n] and "0.912" in out[c_mr.n] and "0.796" in out[c_mr.n] and "1.045" in out[c_mr.n]
assert "0.4477" in out[c_bias.n] and "0.471" in out[c_bias.n] and "0.535" in out[c_bias.n]
assert "0.793" in out[c_sens.n] and "0.845" in out[c_sens.n] and "0.708" in out[c_sens.n] and "0.948" in out[c_sens.n]
assert "0.73" in out[c_t1.n] and "0.84" in out[c_t1.n] and "0.361" in out[c_t1.n] and "0.327" in out[c_t1.n]
assert "71.4451 [44.5324; 223.5188]" in out[c_nnt.n]
for v in ("X 0.8400 [0.7683; 0.9185]", "Z 0.7317 [0.4768; 1.1230]", "tau^2 = 0.0032", "0.03    1  0.8634", '"DL"'):
    assert v in out[c_net.n], v
assert "X:Y  2         0.42   0.96     0.93   0.81" in out[c_split.n]
assert "Z  0.8223" in out[c_rank.n] and "0.94 (0.83-1.06)" in out[c_league.n] and "0.87 (0.56-1.35)" in out[c_league.n]
print("all numbers match ch19")

S = nb.save_fragment
S("rc10_load", nb.html(c_load, marks={"version 8.5-0": 1, "1  Trial A": 2, "mace_pbo": 3}))
S("rc10_obs", nb.html(c_obs, marks={"0.140 0.097": 1, "1.131 0.211": 2}))
S("rc10_main", nb.html(c_main, marks={"43.0       25.1": 1, "0.8441 [0.7791; 0.9145]": 2,
                                      "0.8208 [0.7151; 0.9421] -3.24   0.0102": 3, "[0.6086; 1.1070]": 4,
                                      "tau^2 = 0.0138 [0.0000; 0.1278]": 5, "I^2 = 32.3% [0.0%; 67.7%]": 6,
                                      "13.28    9  0.1501": 7, "Hartung-Knapp adjustment": 8}))
S("rc10_forest", nb.html(c_forest))
S("rc10_default", nb.html(c_default, marks={'"V"': 1, "0.720    0.935": 2, "0.605    1.114": 3, "0.003048625": 4}))
S("rc10_combo", nb.html(c_combo, marks={"1   DL classic 1.6822 1.2807 2.2096 0.0002": 1,
                                        "6 REML      HK 1.6864 1.1715 2.4277 0.0127": 2, "1.48   1.35   1.61": 3}))
S("rc10_adhoc", nb.html(c_adhoc, marks={"0.706": 1, "0.0159": 2}))
S("rc10_het", nb.html(c_het, marks={"13.2849": 1, "0.323 0.000 0.677": 2, "0.0138 0.0000 0.1278 0.1174": 3,
                                    "0.609    1.107": 4, "[1] 0.4": 5, "0.0098 0.0077 0.0138": 6}))
S("rc10_sub", nb.html(c_sub, marks={"Trial A 0.8258": 1, "0.7822 [0.6649; 0.9201]": 2, "0.9218 [0.7718; 1.1010]": 3,
                                    "3.08    1  0.0794": 4, "0.8511 [0.7546; 0.9598]": 5}))
S("rc10_rule", nb.html(c_rule, marks={"0.000 0.782 0.692 0.884": 1}))
S("rc10_mr", nb.html(c_mr, marks={"0.0084": 1, "39.29%": 2, "p-val = 0.1573": 3, "0.912 0.796 1.045": 4}))
S("rc10_funnel", nb.html(c_funnel))
S("rc10_bias", nb.html(c_bias, marks={"p-value = 0.4477": 1, "0.471   0.448   0.535": 2}))
S("rc10_sens", nb.html(c_sens, marks={"0.793 0.845": 1, "0.823 0.708 0.958": 2, "0.834 0.733 0.948": 3}))
S("rc10_nnt", nb.html(c_nnt, marks={"71.4451 [44.5324; 223.5188]": 1}))
S("rc10_net_load", nb.html(c_net_load, marks={"16     Q 2017  X  Y": 1, "X vs P X vs Y Y vs P Z vs P": 2}))
S("rc10_net", nb.html(c_net, marks={"X 0.8400 [0.7683; 0.9185]": 1, "Z 0.7317 [0.4768; 1.1230]": 2, "tau^2 = 0.0032": 3,
                                    "Between designs  0.03    1  0.8634": 4, '"DL"': 5}))
S("rc10_split", nb.html(c_split, marks={"X:Y  2         0.42   0.96     0.93   0.81": 1, "X:Z  0         0.00": 2}))
S("rc10_rank", nb.html(c_rank, marks={"Z  0.8223": 1}))
S("rc10_league", nb.html(c_league, marks={"0.87 (0.56-1.35)": 1, "0.96 (0.79-1.16)": 2, "Lower triangle": 3}))
S("rc10_graph", nb.html(c_graph))
S("rc10_t1", nb.html(c_t1, marks={"0.73   0.84": 1, "0.833 0.361": 2, "[1] 0.327": 3}))
S("rc10_t2", nb.html(c_t2, marks={"0.84   0.78   0.91": 1, "0.82   0.72   0.94": 2}))
nb.save_ipynb("12장 메타분석 (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 19장의 가상 메타분석(급성관상동맥증후군 시험 10개)을 R의 meta 패키지로 합치고, "
                    "이질성, 하위군, 메타회귀, 출판 편향, 민감도 분석, NNT, 네트워크 메타분석까지 차례로 구합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "meta 패키지가 없으면 먼저 install.packages(\"meta\")를 실행합니다.",
              notes={c_load.n: "## a. 메타분석 자료 준비", c_main.n: "## b. 두 모형으로 합치기와 방법 고르기",
                     c_het.n: "## c. 이질성, 하위군, 메타회귀", c_funnel.n: "## d. 출판 편향, 민감도 분석, NNT", c_net_load.n: "## e. 네트워크 메타분석",
                     c_t1.n: "## f. 과제 정답"})
print("rc10 fragments and notebook written")
