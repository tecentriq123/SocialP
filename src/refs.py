# Master reference list. key -> {text (HTML allowed: <i>), doi?, url?, book?}
# Chapters cite with <cite data-ref="key"></cite>. Numbering is assigned at build time.
REFS = {
    # ---------- textbooks ----------
    "bae_book": {"text": "배정민. <i>그림으로 이해하는 닥터 배의 술술 보건의학통계</i>. 한나래출판사; 2012. ISBN 978-89-5566-128-6. (이 사이트의 목차 구성은 이 책을 따랐습니다.)", "book": True},
    "kirkwood2003": {"text": "Kirkwood BR, Sterne JAC. <i>Essential Medical Statistics</i>. 2nd ed. Oxford: Blackwell Science; 2003.", "book": True},
    "altman1991": {"text": "Altman DG. <i>Practical Statistics for Medical Research</i>. London: Chapman &amp; Hall; 1991.", "book": True},
    "rothman2008": {"text": "Rothman KJ, Greenland S, Lash TL. <i>Modern Epidemiology</i>. 3rd ed. Philadelphia: Lippincott Williams &amp; Wilkins; 2008.", "book": True},
    # ---------- chapter 1 ----------
    "fisher1925": {"text": "Fisher RA. <i>Statistical Methods for Research Workers</i>. Edinburgh: Oliver and Boyd; 1925."},
    "student1908": {"text": "Student. The probable error of a mean. <i>Biometrika</i>. 1908;6(1):1-25.", "doi": "10.1093/biomet/6.1.1"},
    "wasserstein2016": {"text": "Wasserstein RL, Lazar NA. The ASA statement on p-values: context, process, and purpose. <i>Am Stat</i>. 2016;70(2):129-133.", "doi": "10.1080/00031305.2016.1154108"},
    "greenland2016": {"text": "Greenland S, Senn SJ, Rothman KJ, et al. Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations. <i>Eur J Epidemiol</i>. 2016;31(4):337-350.", "doi": "10.1007/s10654-016-0149-3"},
    "altman1995absence": {"text": "Altman DG, Bland JM. Absence of evidence is not evidence of absence. <i>BMJ</i>. 1995;311(7003):485.", "doi": "10.1136/bmj.311.7003.485"},
    "altman2006dichot": {"text": "Altman DG, Royston P. The cost of dichotomising continuous variables. <i>BMJ</i>. 2006;332(7549):1080.", "doi": "10.1136/bmj.332.7549.1080"},
    "austin2009smd": {"text": "Austin PC. Balance diagnostics for comparing the distribution of baseline covariates between treatment groups in propensity-score matched samples. <i>Stat Med</i>. 2009;28(25):3083-3107.", "doi": "10.1002/sim.3697"},
    "moher2010consort": {"text": "Moher D, Hopewell S, Schulz KF, et al. CONSORT 2010 explanation and elaboration: updated guidelines for reporting parallel group randomised trials. <i>BMJ</i>. 2010;340:c869.", "doi": "10.1136/bmj.c869"},
    "senn1994": {"text": "Senn S. Testing for baseline balance in clinical trials. <i>Stat Med</i>. 1994;13(17):1715-1726.", "doi": "10.1002/sim.4780131703"},
    "bland1995bonf": {"text": "Bland JM, Altman DG. Multiple significance tests: the Bonferroni method. <i>BMJ</i>. 1995;310(6973):170.", "doi": "10.1136/bmj.310.6973.170"},
    "holm1979": {"text": "Holm S. A simple sequentially rejective multiple test procedure. <i>Scand J Stat</i>. 1979;6(2):65-70."},
    "bh1995": {"text": "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. <i>J R Stat Soc Series B Stat Methodol</i>. 1995;57(1):289-300.", "doi": "10.1111/j.2517-6161.1995.tb02031.x"},
    "rothman1990": {"text": "Rothman KJ. No adjustments are needed for multiple comparisons. <i>Epidemiology</i>. 1990;1(1):43-46."},
    "benjamin2018": {"text": "Benjamin DJ, Berger JO, Johannesson M, et al. Redefine statistical significance. <i>Nat Hum Behav</i>. 2018;2(1):6-10.", "doi": "10.1038/s41562-017-0189-z"},
    "shapiro1965": {"text": "Shapiro SS, Wilk MB. An analysis of variance test for normality (complete samples). <i>Biometrika</i>. 1965;52(3-4):591-611.", "doi": "10.1093/biomet/52.3-4.591"},
    "ghasemi2012": {"text": "Ghasemi A, Zahediasl S. Normality tests for statistical analysis: a guide for non-statisticians. <i>Int J Endocrinol Metab</i>. 2012;10(2):486-489.", "doi": "10.5812/ijem.3505"},
    "fagerland2012": {"text": "Fagerland MW. t-tests, non-parametric tests, and large studies—a paradox of statistical practice? <i>BMC Med Res Methodol</i>. 2012;12:78.", "doi": "10.1186/1471-2288-12-78"},
    "hart2001": {"text": "Hart A. Mann-Whitney test is not just a test of medians: differences in spread can be important. <i>BMJ</i>. 2001;323(7309):391-393.", "doi": "10.1136/bmj.323.7309.391"},
    "thompson2000cost": {"text": "Thompson SG, Barber JA. How should cost data in pragmatic randomised trials be analysed? <i>BMJ</i>. 2000;320(7243):1197-1200.", "doi": "10.1136/bmj.320.7243.1197"},
}

# ---------- merge per-chapter additions (refs_add/chXX.py each defining ADD = {...}) ----------
import glob as _glob, os as _os
for _f in sorted(_glob.glob(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "refs_add", "*.py"))):
    _ns = {}
    exec(open(_f, encoding="utf-8").read(), _ns)
    for _k, _v in _ns.get("ADD", {}).items():
        if _k in REFS and REFS[_k]["text"] != _v["text"]:
            print(f"WARNING: ref key {_k} redefined in {_os.path.basename(_f)}")
        REFS.setdefault(_k, _v)
