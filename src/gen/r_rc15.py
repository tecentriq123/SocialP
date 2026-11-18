"""R과 파이썬으로 하는 통계분석 6장 · 반복측정 자료: 혼합모형과 GEE — R cells run for real with rkit.

run:  python3 gen/r_rc15.py      (NOMARK=1 python3 gen/r_rc15.py 로 돌리면 모든 셀의 출력을 찍는다)

보건통계학 기초 10장의 '직접 분석' 상자에 있던 파이썬 코드를 2026-11-18 이 장으로 옮기며 같은 예를 R로 다시 돌린다.
자료: 10장의 약사 주도 중재 시험 100명(pub/data/ch10_hba1c.csv, 넓은 형식; gen/data_rc15.py가 lib_ch10.simulate(1436)에서
만든다)과 a 절의 환자 4명 × 방문 3회 작은 예.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rkit import RNotebook  # noqa: E402

NOMARK = bool(os.environ.get("NOMARK"))
nb = RNotebook("rc15")
D = "https://socialp-ajou.tecentriq12.workers.dev/data/"

# ------------------------------------------------------------------ a. 반복측정 분산분석
c_toy = nb.cell('''
toy <- data.frame(id = factor(rep(1:4, each = 3)), visit = factor(rep(c(0, 3, 6), 4)),
                  hba1c = c(9.2, 9.0, 8.5,  7.4, 7.3, 6.9,  8.4, 7.8, 7.8,  10.2, 9.9, 9.6))
summary(aov(hba1c ~ visit + Error(id / visit), data = toy))   # 환자 간 변동을 떼어 낸 반복측정 분산분석
summary(aov(hba1c ~ visit, data = toy))                       # 환자를 무시한 일원분산분석
''', title="환자 4명 × 방문 3회의 반복측정 분산분석")

c_rm = nb.cell(f'''
suppressPackageStartupMessages({{library(tidyr); library(dplyr); library(afex)}})
w <- read.csv("{D}ch10_hba1c.csv")                        # 환자 한 명이 한 줄 (넓은 형식)
cc <- w |> filter(!is.na(hba1c_0), !is.na(hba1c_3), !is.na(hba1c_6), !is.na(hba1c_12)) |>
  pivot_longer(starts_with("hba1c_"), names_to = "month", names_prefix = "hba1c_", values_to = "hba1c") |>
  mutate(month = factor(month, levels = c("0", "3", "6", "12")), group = factor(group))
table(cc$group) / 4                                       # 네 번 모두 잰 환자 (군별)
a <- aov_ez("id", "hba1c", cc, between = "group", within = "month", type = 3)
summary(a)                                                # 구형성 가정, Mauchly 검정, GG·HF 보정
''', title="네 번 모두 잰 75명의 반복측정 분산분석")

c_eps = nb.cell('''
F <- 3.946; eps <- 0.809                                  # 교호작용의 F와 Greenhouse–Geisser ε
round(c(sphericity = pf(F, 3, 219, lower.tail = FALSE),
        GG = pf(F, 3 * eps, 219 * eps, lower.tail = FALSE),
        lower_bound = pf(F, 1, 73, lower.tail = FALSE)), 4)
nice(a, correction = "GG")                                # 보정한 자유도를 함께 적은 표
''', title="ε으로 자유도를 줄여 P 다시 읽기")

# ------------------------------------------------------------------ b. 넓은 형식과 긴 형식
c_long = nb.cell('''
L <- w |> pivot_longer(starts_with("hba1c_"), names_to = "month", names_prefix = "hba1c_",
                       values_to = "hba1c", names_transform = list(month = as.integer)) |>
  select(id, group, month, hba1c)
nrow(L); sum(is.na(L$hba1c))                              # 100명 × 4 = 400행, 그 가운데 빈 값
L <- filter(L, !is.na(hba1c))                             # 측정하지 못한 방문의 행은 지움
nrow(L); head(L, 6)
''', title="넓은 형식을 긴 형식으로")

c_pat = nb.cell('''
pat <- apply(!is.na(w[, c("hba1c_0", "hba1c_3", "hba1c_6", "hba1c_12")]), 1,
             function(o) paste(ifelse(o, "O", "-"), collapse = ""))
table(pattern = pat, group = w$group)                     # O = 잰 방문, - = 빈 방문
has12 <- !is.na(w$hba1c_12)
aggregate(hba1c_0 ~ group + has12, data = cbind(w, has12), FUN = function(x) round(c(n = length(x), mean = mean(x)), 2))
m12 <- glm(!has12 ~ group + I(hba1c_0 - mean(hba1c_0)), family = binomial, data = w)
round(exp(cbind(OR = coef(m12), confint.default(m12)))[-1, ], 2)   # 12개월 값이 빌 오즈
''', title="결측 양상과 탈락자의 기저값")

# ------------------------------------------------------------------ c. 선형 혼합모형
c_ri = nb.cell('''
suppressPackageStartupMessages(library(lmerTest))       # lme4의 lmer()에 Satterthwaite 자유도를 더함
ri <- lmer(hba1c ~ group * month + (1 | id), data = L)  # 무작위 절편, REML이 기본값
summary(ri)$coefficients
as.data.frame(VarCorr(ri))[, c("grp", "vcov", "sdcor")]
''', title="무작위 절편 모형")

c_rs = nb.cell('''
rs <- lmer(hba1c ~ group * month + (month | id), data = L,
           control = lmerControl(optimizer = "bobyqa"))  # 무작위 절편 + 기울기
summary(rs)$coefficients
as.data.frame(VarCorr(rs))[, c("var1", "var2", "vcov", "sdcor")]
contest(rs, c(0, 0, 0, 12), joint = FALSE)               # 12개월 군간 차이 = 12 × 군 × 개월
c(nobs = nobs(rs), patients = ngrps(rs))
''', title="무작위 절편 + 기울기 모형")

c_lrt = nb.cell('''
anova(ri, rs, refit = FALSE)                             # 같은 고정효과: REML 우도로 비교
c(REML_ri = REMLcrit(ri), REML_rs = REMLcrit(rs))        # −2 × REML 로그우도
''', title="무작위 기울기가 필요한가")

c_cov = nb.cell('''
suppressPackageStartupMessages(library(nlme))
L$fmonth <- factor(L$month); L$vn <- as.integer(L$fmonth)
cov_fit <- function(cor, w = NULL)
  gls(hba1c ~ fmonth * group, data = L, correlation = cor, weights = w, method = "REML")
fits <- list(CS  = cov_fit(corCompSymm(form = ~ 1 | id)),
             AR1 = cov_fit(corAR1(form = ~ vn | id)),
             UN  = cov_fit(corSymm(form = ~ vn | id), varIdent(form = ~ 1 | fmonth)))
t(sapply(fits, function(f) c(m2REML = -2 * as.numeric(logLik(f)), AIC = AIC(f),
                             diff12 = unname(coef(f)["fmonth12:group"]))))
''', title="공분산 구조 비교")

c_mmrm = nb.cell('''
suppressWarnings(suppressPackageStartupMessages({library(mmrm); library(emmeans)}))
C <- w |> pivot_longer(c(hba1c_3, hba1c_6, hba1c_12), names_to = "visit", names_prefix = "hba1c_",
                       values_to = "y") |>
  filter(!is.na(y)) |>
  mutate(chg = y - hba1c_0, visit = factor(visit, levels = c("3", "6", "12")),
         group = factor(group), id = factor(id))
C$cbase <- C$hba1c_0 - mean(w$hba1c_0[w$id %in% C$id])   # 기저값을 분석 환자 평균으로 중심화
c(patients = nlevels(droplevels(C$id)), rows = nrow(C))
m <- mmrm(chg ~ visit + group:visit + cbase:visit + us(visit | id), data = C,
          method = "Satterthwaite")                       # 비구조 공분산, REML
em <- emmeans(m, ~ group | visit, at = list(cbase = 0))
em
contrast(em, "revpairwise")                              # 방문마다 중재군 − 일반 진료군
''', title="MMRM: 비구조 공분산과 Satterthwaite 자유도")

c_comp = nb.cell('''
w$chg12 <- w$hba1c_12 - w$hba1c_0
w$cb <- w$hba1c_0 - mean(w$hba1c_0)
cc12 <- lm(chg12 ~ group + cb, data = w)                 # 12개월 값이 있는 환자만 (79명)
ccall <- lm(chg12 ~ group + cb, data = subset(w, !is.na(hba1c_3) & !is.na(hba1c_6)))  # 네 번 모두 잰 75명
locf <- w |> mutate(last = coalesce(hba1c_12, hba1c_6, hba1c_3)) |> filter(!is.na(last))
lo <- lm(I(last - hba1c_0) ~ group + cb, data = locf)    # 마지막 값을 끌어온 분석 (92명)
round(rbind(completers12 = coef(summary(cc12))["group", 1:2], completers_all = coef(summary(ccall))["group", 1:2],
            LOCF = coef(summary(lo))["group", 1:2]), 3)
''', title="결측을 다루는 방법에 따라 달라지는 12개월 차이")

# ------------------------------------------------------------------ d. GEE와 로지스틱 혼합모형
c_adh = nb.cell('''
A <- w |> pivot_longer(starts_with("adh_"), names_to = "visit", names_prefix = "adh_", values_to = "adh") |>
  filter(!is.na(adh)) |> mutate(visit = factor(visit, levels = c("3", "6", "12"))) |>
  arrange(id, visit)                                     # GEE는 같은 환자의 행이 붙어 있어야 함
c(rows = nrow(A), patients = length(unique(A$id)))
with(A, round(tapply(adh, list(group, visit), mean), 3)) # 기간별 순응 비율 (행: 일반 진료 0, 중재 1)
''', title="순응 자료를 긴 형식으로")

c_gee = nb.cell('''
suppressPackageStartupMessages(library(geepack))
ge <- geeglm(adh ~ group + visit, id = id, data = A, family = binomial, corstr = "exchangeable")
summary(ge)
b <- coef(ge)["group"]; se <- summary(ge)$coefficients["group", "Std.err"]
round(exp(c(OR = b, b - 1.96 * se, b + 1.96 * se)), 2)
''', title="교환가능 작업상관 GEE")

c_gee4 = nb.cell('''
t(sapply(c("independence", "exchangeable", "ar1", "unstructured"), function(cs) {
  g <- geeglm(adh ~ group + visit, id = id, waves = as.integer(visit), data = A,
              family = binomial, corstr = cs)
  s <- summary(g)$coefficients["group", "Std.err"]
  round(c(OR = exp(coef(g)[["group"]]), lower = exp(coef(g)[["group"]] - 1.96 * s),
          upper = exp(coef(g)[["group"]] + 1.96 * s), robust_SE = s, QIC = QIC(g)[["QIC"]]), 3)
}))
gl0 <- glm(adh ~ group + visit, family = binomial, data = A)   # 같은 환자의 상관을 무시하면
round(summary(gl0)$coefficients["group", 1:2], 4)
''', title="작업상관 네 가지와 상관을 무시한 표준오차")

c_glmm = nb.cell('''
suppressPackageStartupMessages(library(lme4))
lap <- glmer(adh ~ group + visit + (1 | id), data = A, family = binomial)              # 라플라스 근사
quad <- glmer(adh ~ group + visit + (1 | id), data = A, family = binomial, nAGQ = 40)   # 적분점 40개
se <- sqrt(vcov(quad)["group", "group"])
round(rbind(Laplace = c(OR = exp(fixef(lap)[["group"]]), SD = attr(VarCorr(lap)$id, "stddev")[[1]], NA, NA),
            quadrature = c(exp(fixef(quad)[["group"]]), attr(VarCorr(quad)$id, "stddev")[[1]],
                           exp(fixef(quad)[["group"]] + c(-1, 1) * 1.96 * se))), 2)
''', title="로지스틱 혼합모형의 조건부 오즈비")

# ------------------------------------------------------------------ e. 과제
c_t1 = nb.cell('''
pf(18.044, 3 * 0.8087, 219 * 0.8087, lower.tail = FALSE)   # 시간 주효과, GG 보정
pf(3.946, 1, 73, lower.tail = FALSE)                       # 교호작용, 하한 보정
''', title="과제 1 정답")

c_t2 = nb.cell('''
gi <- geeglm(adh ~ group * visit, id = id, data = A, family = binomial, corstr = "exchangeable")
anova(ge, gi)                                             # 군 × 기간 곱셈항 2개의 Wald 검정
''', title="과제 2 정답")

nb.run()

if NOMARK:
    for c in nb.cells:
        print(f"===== cell {c.n} {c.title}\n{c.text}")
        for wn in c.warns:
            print("WARN:", wn)
        if c.error:
            print("ERR:", c.error)
        print("images:", len(c.images))
    sys.exit(0)

out = {c.n: c.text for c in nb.cells}
A = lambda c, *xs: [None for x in xs if (x in out[c.n]) or (_ for _ in ()).throw(AssertionError((c.title, x)))]
A(c_toy, "visit      2   0.72  0.3600    21.6 0.00181", "0.264  0.774")
A(c_rm, "33 42", "group:month     1.8      3   34.224    219    3.9456  0.009078", "0.70302 0.00012366", "group:month 0.80874    0.01484", "0.8385514")
A(c_eps, "0.0091      0.0148      0.0507", "2.43, 177.11")
A(c_long, "[1] 400\n[1] 46\n[1] 354")
A(c_pat, "OOOO 33 42", "4.72  2.07  10.72", "0.26  0.08   0.83")
A(c_ri, "-0.03757659 0.009881633", "0.6281669", "0.1552926")
A(c_rs, "-0.04293037 0.012631169 85.76935", "0.538710593", "0.001911400", "-0.5151645   0.151574", "-0.8164952 -0.2138337")
A(c_lrt, "33.359  2  5.704e-08", "630.5848 597.2260")
A(c_cov, "630.8549", "-0.3975497", "633.3094", "-0.4065088", "594.2031", "-0.4662492")
A(c_mmrm, "92      254", "-0.4664 0.1557 83.8  -2.995  0.0036", "-0.2560 0.1140 84.6  -2.245  0.0274", "-0.6419 0.1041", "-0.1755 0.1162")
A(c_comp, "-0.375      0.161", "-0.301      0.165", "-0.449      0.142")
A(c_adh, "259       93", "0.651 0.575 0.429", "0.780 0.745 0.750")
A(c_gee, "0.9269  0.3402 7.422  0.00644", "alpha   0.3016", "Number of clusters:   93", "2.53     1.30     4.92")
A(c_gee4, "2.523 1.291 4.931", "324.0", "0.9253     0.2724")
A(c_glmm, "3.66 1.47", "3.77 1.64 1.4 10.19")
A(c_t1, "6.512e-09", "0.05074")
A(c_t2, "2 2.29      0.32")

S = nb.save_fragment
S("rc15_toy", nb.html(c_toy, marks={"Residuals  3  12.18    4.06": 1, "visit      2   0.72  0.3600    21.6 0.00181": 2, "0.264  0.774": 3}))
S("rc15_rm", nb.html(c_rm, marks={"33 42": 1, "group:month     1.8      3   34.224    219    3.9456  0.009078": 2, "month              0.70302 0.00012366": 3,
                                  "group:month 0.80874    0.01484": 4, "group:month 0.8385514": 5}))
S("rc15_eps", nb.html(c_eps, marks={"0.0091      0.0148      0.0507": 1, "group:month 2.43, 177.11": 2}))
S("rc15_long", nb.html(c_long, marks={"[1] 46": 1, "[1] 354": 2}))
S("rc15_pat", nb.html(c_pat, marks={"O---  8  0": 1, "OOOO 33 42": 2, "9.06": 3, "4.72  2.07  10.72": 4}))
S("rc15_ri", nb.html(c_ri, marks={"-0.03757659 0.009881633 255.7804": 1, "0.6281669 0.7925698": 2, "0.1552926 0.3940718": 3}))
S("rc15_rs", nb.html(c_rs, marks={"-0.04293037 0.012631169 85.76935": 1, "0.538710593 0.73396907": 2, "0.001911400 0.04371957": 3,
                                  "0.007751969 0.24157860": 4, "0.107562806 0.32796769": 5, "-0.8164952 -0.2138337": 6}))
S("rc15_lrt", nb.html(c_lrt, marks={"33.359  2  5.704e-08": 1, "613.23": 2}))
S("rc15_cov", nb.html(c_cov, marks={"CS  630.8549": 1, "AR1 633.3094": 2, "UN  594.2031": 3}))
S("rc15_mmrm", nb.html(c_mmrm, marks={"92      254": 1, " 1     -0.6419 0.1041 82.6": 2, "-0.4664 0.1557 83.8  -2.995  0.0036": 3, "-0.2560 0.1140 84.6  -2.245  0.0274": 4}))
S("rc15_comp", nb.html(c_comp, marks={"-0.375      0.161": 1, "-0.301      0.165": 2, "-0.449      0.142": 3}))
S("rc15_adh", nb.html(c_adh, marks={"259       93": 1, "0.651 0.575 0.429": 2}))
S("rc15_gee", nb.html(c_gee, marks={"0.9269  0.3402 7.422  0.00644": 1, "-0.5513  0.2583 4.556  0.03280": 2, "(Intercept)    1.006   0.111": 3,
                                    "alpha   0.3016   0.088": 4, "Number of clusters:   93  Maximum cluster size: 3": 5, "2.53     1.30     4.92": 6}))
S("rc15_gee4", nb.html(c_gee4, marks={"ar1          2.587 1.311 5.102": 1, "unstructured 2.500 1.286 4.859     0.339 323.9": 2, "0.9253     0.2724": 3}))
S("rc15_glmm", nb.html(c_glmm, marks={"3.66 1.47": 1, "3.77 1.64 1.4 10.19": 2}))
S("rc15_t1", nb.html(c_t1, marks={"6.512e-09": 1, "0.05074": 2}))
S("rc15_t2", nb.html(c_t2, marks={"2 2.29      0.32": 1}))
nb.save_ipynb("6장 반복측정 자료: 혼합모형과 GEE (R과 파이썬으로 하는 통계분석)",
              intro="보건통계학 기초 10장의 예로 반복측정 분산분석과 구형성 보정, 넓은 형식과 긴 형식, 무작위 절편·기울기 모형, "
                    "공분산 구조와 MMRM, GEE와 로지스틱 혼합모형을 R로 계산합니다. "
                    "Colab에서는 런타임 → 런타임 유형 변경에서 R을 고른 뒤 셀을 위에서부터 차례로 실행하세요. "
                    "패키지가 없으면 먼저 install.packages(c(\"tidyr\", \"dplyr\", \"afex\", \"lmerTest\", \"mmrm\", \"emmeans\", \"geepack\"))를 실행합니다.",
              notes={c_toy.n: "## a. 반복측정 분산분석", c_long.n: "## b. 넓은 형식과 긴 형식, 결측 양상",
                     c_ri.n: "## c. 선형 혼합모형과 MMRM", c_adh.n: "## d. GEE와 로지스틱 혼합모형", c_t1.n: "## e. 과제 정답"})
print("rc15 fragments and notebook written")
