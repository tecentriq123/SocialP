# Chapter 19 references (meta-analysis). Bibliographic details checked against PubMed records (PMID in comment), 2026-10-02.
# Reused from other files: dersimonian1986, higgins2002, higgins2003, egger1997 (ch14b), altman2003int, altman1998nnt, sun2010 (ch08).
ADD = {
    # PMID 33782057
    "page2021": {"text": "Page MJ, McKenzie JE, Bossuyt PM, et al. The PRISMA 2020 statement: an updated guideline for reporting systematic reviews. <i>BMJ</i>. 2021;372:n71.",
                 "doi": "10.1136/bmj.n71"},
    # PMID 31462531
    "sterne2019rob": {"text": "Sterne JAC, Savović J, Page MJ, et al. RoB 2: a revised tool for assessing risk of bias in randomised trials. <i>BMJ</i>. 2019;366:l4898.",
                      "doi": "10.1136/bmj.l4898"},
    # PMID 27733354
    "sterne2016robins": {"text": "Sterne JA, Hernán MA, Reeves BC, et al. ROBINS-I: a tool for assessing risk of bias in non-randomised studies of interventions. <i>BMJ</i>. 2016;355:i4919.",
                         "doi": "10.1136/bmj.i4919"},
    # PMID 18436948
    "guyatt2008": {"text": "Guyatt GH, Oxman AD, Vist GE, et al; GRADE Working Group. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. <i>BMJ</i>. 2008;336(7650):924-926.",
                   "doi": "10.1136/bmj.39489.470347.AD"},
    # PMID 21310794
    "riley2011": {"text": "Riley RD, Higgins JPT, Deeks JJ. Interpretation of random effects meta-analyses. <i>BMJ</i>. 2011;342:d549.",
                  "doi": "10.1136/bmj.d549"},
    # PMID 12111920
    "thompson2002": {"text": "Thompson SG, Higgins JPT. How should meta-regression analyses be undertaken and interpreted? <i>Stat Med</i>. 2002;21(11):1559-1573.",
                     "doi": "10.1002/sim.1187"},
    # PMID 21784880
    "sterne2011funnel": {"text": "Sterne JAC, Sutton AJ, Ioannidis JPA, et al. Recommendations for examining and interpreting funnel plot asymmetry in meta-analyses of randomised controlled trials. <i>BMJ</i>. 2011;343:d4002.",
                         "doi": "10.1136/bmj.d4002"},
    # PMID 24548571
    "inthout2014": {"text": "IntHout J, Ioannidis JPA, Borm GF. The Hartung-Knapp-Sidik-Jonkman method for random effects meta-analysis is straightforward and considerably outperforms the standard DerSimonian-Laird method. <i>BMC Med Res Methodol</i>. 2014;14:25.",
                    "doi": "10.1186/1471-2288-14-25"},
    # PMID 15116347
    "sweeting2004": {"text": "Sweeting MJ, Sutton AJ, Lambert PC. What to add to nothing? Use and avoidance of continuity corrections in meta-analysis of sparse data. <i>Stat Med</i>. 2004;23(9):1351-1375.",
                     "doi": "10.1002/sim.1761"},
    # PMID 9250266 (added 2026-10-14 for the e 절 '간접비교' box; checked against the PubMed record)
    "bucher1997": {"text": "Bucher HC, Guyatt GH, Griffith LE, Walter SD. The results of direct and indirect treatment comparisons in meta-analysis of randomized controlled trials. <i>J Clin Epidemiol</i>. 1997;50(6):683-691.",
                   "doi": "10.1016/s0895-4356(97)00049-8"},
    # PMID 29417147 (PMC5885890)
    "chiarito2018": {"text": "Chiarito M, Cao D, Cannata F, et al. Direct oral anticoagulants in addition to antiplatelet therapy for secondary prevention after acute coronary syndromes: a systematic review and meta-analysis. <i>JAMA Cardiol</i>. 2018;3(3):234-241.",
                     "doi": "10.1001/jamacardio.2017.5306"},
    # PMID 26332144 (added 2026-10-15 for c 절 'τ²는 어떤 방법으로 구하나?'; checked against the PubMed record)
    "veroniki2016": {"text": "Veroniki AA, Jackson D, Viechtbauer W, et al. Methods to estimate the between-study variance and its uncertainty in meta-analysis. <i>Res Synth Methods</i>. 2016;7(1):55-79.",
                     "doi": "10.1002/jrsm.1164"},
    # PMID 30067315 (added 2026-10-15)
    "langan2019": {"text": "Langan D, Higgins JPT, Jackson D, et al. A comparison of heterogeneity variance estimators in simulated random-effects meta-analyses. <i>Res Synth Methods</i>. 2019;10(1):83-98.",
                   "doi": "10.1002/jrsm.1316"},
    # PMID 26573817 (added 2026-10-15)
    "rover2015": {"text": "Röver C, Knapp G, Friede T. Hartung-Knapp-Sidik-Jonkman approach and its modification for random-effects meta-analysis with few studies. <i>BMC Med Res Methodol</i>. 2015;15:99.",
                  "doi": "10.1186/s12874-015-0091-1"},
    # PMID 31948937 (added 2026-10-15)
    "campbell2020swim": {"text": "Campbell M, McKenzie JE, Sowden A, et al. Synthesis without meta-analysis (SWiM) in systematic reviews: reporting guideline. <i>BMJ</i>. 2020;368:l6890.",
                         "doi": "10.1136/bmj.l6890"},
    # Cochrane training presentation, dated 23 October 2024 (read 2026-10-15): REML default, HKSJ when k > 2 and tau2 > 0,
    # Wald when tau2 = 0, both when k = 2, PI with t(k-1) after HKSJ, I2 from the estimated tau2.
    "cochrane2024re": {"text": "Veroniki AA, McKenzie JE. Introduction to new random-effects methods in RevMan [webinar slides]. Cochrane Training; 23 October 2024.",
                       "url": "https://training-noproxy.cochrane.org/sites/training.cochrane.org/files/public/uploads/Introduction%20to%20new%20random-effects%20methods%20in%20RevMan.pdf"},
    # Cochrane RevMan release notes (read 2026-10-15): 8.15.0 (2025-01-23) REML, tau2 CI, HKSJ, prediction intervals added;
    # 8.19.0 (2025-03-20) "Default summary effect CI method changed to Wald in random-effects analyses".
    "revman2025notes": {"text": "Cochrane. RevMan release notes. Cochrane Documentation; versions 8.15.0 (23 January 2025) and 8.19.0 (20 March 2025). Accessed 15 October 2026.",
                        "url": "https://documentation.cochrane.org/revman-kb/revman-release-notes-342261851.html"},
    # Program defaults for tau2 (read 2026-10-15) for the c 절 3단계 표 '주요 프로그램의 τ² 기본값'.
    # Cochrane Methods explainer: "The default option in RevMan for calculating Tau2 is now REML"; DL still selectable.
    "cochrane_reexplainer": {"text": "Cochrane Methods. New random-effects methods in RevMan: explainer [PDF]. Cochrane; n.d. Accessed 15 October 2026.",
                             "url": "https://methods.cochrane.org/sites/methods.cochrane.org/files/uploads/news/final_-re-methods-explainer.pdf"},
    # R meta: "the current default in R package meta is the REML estimator"; DL was the default in version 4 and below.
    "meta_pkg": {"text": "Schwarzer G. meta: brief overview of methods and general hints (R package meta documentation). Accessed 15 October 2026.",
                 "url": "https://rdrr.io/cran/meta/man/meta-package.html"},
    # metafor misc-recs: rma.uni default method="REML"; test="knha" recommended; default test="z".
    "metafor_recs": {"text": "Viechtbauer W. Recommended practices (misc-recs), metafor package documentation. Accessed 15 October 2026.",
                     "url": "https://search.r-project.org/CRAN/refmans/metafor/html/misc-recs.html"},
    # Stata manual: "random is a synonym for random(reml)".
    "stata_meta": {"text": "StataCorp. meta summarize. In: Stata Meta-Analysis Reference Manual. College Station, TX: Stata Press. Accessed 15 October 2026.",
                   "url": "https://www.stata.com/manuals/metametasummarize.pdf"},
}
