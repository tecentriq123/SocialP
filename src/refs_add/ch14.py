# Chapter 14 (청구자료와 약물감시 자료의 연구 설계). Bibliographic details checked against PubMed records (2026-10-02); PMID in comments.
# Reused keys: ray2003, lund2015, hernan2016 (refs_add/ch07.py), hernan2016jce (refs_add/ch16.py).
ADD = {
    # PMID 15862718
    "schneeweiss2005": {"text": "Schneeweiss S, Avorn J. A review of uses of health care utilization databases for epidemiologic research on therapeutics. <i>J Clin Epidemiol</i>. 2005;58(4):323-337.",
                        "doi": "10.1016/j.jclinepi.2004.10.012"},
    # PMID 28378543
    "kimja2017": {"text": "Kim JA, Yoon S, Kim LY, Kim DS. Towards actualizing the value potential of Korea Health Insurance Review and Assessment (HIRA) data as a resource for health research: strengths, limitations, applications, and strategies for optimal use of HIRA data. <i>J Korean Med Sci</i>. 2017;32(5):718-728.",
                  "doi": "10.3346/jkms.2017.32.5.718"},
    # PMID 25078381 (table numbers 20/30/40/53 of the HIRA patient samples)
    "kiml2014": {"text": "Kim L, Kim JA, Kim S. A guide for the utilization of Health Insurance Review and Assessment Service National Patient Samples. <i>Epidemiol Health</i>. 2014;36:e2014008.",
                 "doi": "10.4178/epih/e2014008"},
    # PMID 27794523
    "seong2017": {"text": "Cheol Seong S, Kim YY, Khang YH, et al. Data resource profile: the National Health Information Database of the National Health Insurance Service in South Korea. <i>Int J Epidemiol</i>. 2017;46(3):799-800.",
                  "doi": "10.1093/ije/dyw253"},
    # PMID 26822938 (NHIS sample cohort profile; no PMC full text, content checked through the title and through citing papers)
    "lee2017nsc": {"text": "Lee J, Lee JS, Park SH, Shin SA, Kim K. Cohort profile: the National Health Insurance Service-National Sample Cohort (NHIS-NSC), South Korea. <i>Int J Epidemiol</i>. 2017;46(2):e15.",
                   "doi": "10.1093/ije/dyv319"},
    # PMID 35656154 (PMC9133780 full text read 2026-10-03: HIRA death only through the in-hospital treatment-result code and sometimes omitted;
    # NHIS has death records, screening, income in addition to the HIRA contents; NHIS sample cohorts are longitudinal, HIRA patient samples cross-sectional)
    "kyoung2022": {"text": "Kyoung DS, Kim HS. Understanding and utilizing claim data from the Korean National Health Insurance Service (NHIS) and Health Insurance Review &amp; Assessment (HIRA) database for research. <i>J Lipid Atheroscler</i>. 2022;11(2):103-110.",
                   "doi": "10.12997/jla.2022.11.2.103"},
    # PMID 30429167
    "langan2018": {"text": "Langan SM, Schmidt SA, Wing K, et al. The reporting of studies conducted using observational routinely collected health data statement for pharmacoepidemiology (RECORD-PE). <i>BMJ</i>. 2018;363:k3532.",
                   "doi": "10.1136/bmj.k3532"},
    # PMID 25800216
    "yoshida2015": {"text": "Yoshida K, Solomon DH, Kim SC. Active-comparator design and new-user design in observational studies. <i>Nat Rev Rheumatol</i>. 2015;11(7):437-441.",
                    "doi": "10.1038/nrrheum.2015.30"},
    # PMID 30856654
    "schneeweiss2019": {"text": "Schneeweiss S, Rassen JA, Brown JS, et al. Graphical depiction of longitudinal study designs in health care databases. <i>Ann Intern Med</i>. 2019;170(6):398-406.",
                        "doi": "10.7326/M18-3079"},
    # PMID 16514590
    "andrade2006": {"text": "Andrade SE, Kahler KH, Frech F, Chan KA. Methods for evaluation of medication adherence and persistence using automated databases. <i>Pharmacoepidemiol Drug Saf</i>. 2006;15(8):565-574.",
                    "doi": "10.1002/pds.1230"},
    # PMID 22363378
    "kimm2012": {"text": "Kimm H, Yun JE, Lee SH, Jang Y, Jee SH. Validity of the diagnosis of acute myocardial infarction in Korean national medical health insurance claims data: the Korean Heart Study (1). <i>Korean Circ J</i>. 2012;42(1):10-15.",
                 "doi": "10.4070/kcj.2012.42.1.10"},
    # PMID 18064739
    "vonelm2007": {"text": "von Elm E, Altman DG, Egger M, Pocock SJ, Gøtzsche PC, Vandenbroucke JP; STROBE Initiative. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. <i>Lancet</i>. 2007;370(9596):1453-1457.",
                   "doi": "10.1016/S0140-6736(07)61602-X"},
    # ---- 14장 e 절 (자발적 보고 자료와 불균형 분석). Checked against PubMed records (2026-11-16); PMID in comments.
    # PMID 15317031
    "rothman2004": {"text": "Rothman KJ, Lanes S, Sacks ST. The reporting odds ratio and its advantages over the proportional reporting ratio. <i>Pharmacoepidemiol Drug Saf</i>. 2004;13(8):519-523.",
                    "doi": "10.1002/pds.1001"},
    # PMID 11828828 (signal criteria: 3 or more cases, PRR at least 2, chi-squared at least 4; UK Yellow Card data)
    "evans2001": {"text": "Evans SJ, Waller PC, Davis S. Use of proportional reporting ratios (PRRs) for signal generation from spontaneous adverse drug reaction reports. <i>Pharmacoepidemiol Drug Saf</i>. 2001;10(6):483-486.",
                  "doi": "10.1002/pds.677"},
    # PMID 9696956 (BCPNN, WHO database)
    "bate1998": {"text": "Bate A, Lindquist M, Edwards IR, et al. A Bayesian neural network method for adverse drug reaction signal generation. <i>Eur J Clin Pharmacol</i>. 1998;54(4):315-321.",
                 "doi": "10.1007/s002280050466"},
    # PMID 21705438 (shrinkage observed-to-expected ratios; gamma credibility interval)
    "noren2013": {"text": "Norén GN, Hopstadius J, Bate A. Shrinkage observed-to-expected ratios for robust and transparent large-scale pattern discovery. <i>Stat Methods Med Res</i>. 2013;22(1):57-69.",
                  "doi": "10.1177/0962280211403604"},
    # American Statistician (not indexed in PubMed); JSTOR DOI
    "dumouchel1999": {"text": "DuMouchel W. Bayesian data mining in large frequency tables, with an application to the FDA spontaneous reporting system. <i>Am Stat</i>. 1999;53(3):177-190.",
                      "doi": "10.2307/2686093"},
    # PMID 12071774 (MGPS at FDA; signal = EB05 of 2 or more)
    "szarfman2002": {"text": "Szarfman A, Machado SG, O&#39;Neill RT. Use of screening algorithms and computer systems to efficiently signal higher-than-expected combinations of drugs and events in the US FDA&#39;s spontaneous reports database. <i>Drug Saf</i>. 2002;25(6):381-392.",
                     "doi": "10.2165/00002018-200225060-00001"},
    # PMID 34881404 (PMC8763735: ROR is the routine statistic in EudraVigilance; threshold lower 95% CI above 1 and 3 cases for additional-monitoring substances, 5 otherwise)
    "candore2022": {"text": "Candore G, Monzon S, Slattery J, et al. The impact of mandatory reporting of non-serious safety reports to EudraVigilance on the detection of adverse reactions. <i>Drug Saf</i>. 2022;45(1):83-95.",
                    "doi": "10.1007/s40264-021-01137-0"},
    # PMID 30773344
    "faillie2019": {"text": "Faillie JL. Case-non-case studies: principle, methods, bias and interpretation. <i>Therapie</i>. 2019;74(2):225-232.",
                    "doi": "10.1016/j.therap.2019.01.006"},
    # PMID 17867726 (statins-rhabdomyolysis ROR 5.8 before vs 9.4 after alert; pergolide valvulopathy 0 vs 63 reports)
    "pariente2007": {"text": "Pariente A, Gregoire F, Fourrier-Reglat A, Haramburu F, Moore N. Impact of safety alerts on measures of disproportionality in spontaneous reporting databases: the notoriety bias. <i>Drug Saf</i>. 2007;30(10):891-898.",
                     "doi": "10.2165/00002018-200730100-00007"},
    # PMID 16689555 (median under-reporting 94%, IQR 82-98%)
    "hazell2006": {"text": "Hazell L, Shakir SA. Under-reporting of adverse drug reactions: a systematic review. <i>Drug Saf</i>. 2006;29(5):385-396.",
                   "doi": "10.2165/00002018-200629050-00003"},
    # PMID 38713346 (32 recommendations in 14 items for the main text, 12 in 4 items for abstracts)
    "fusaroli2024": {"text": "Fusaroli M, Salvo F, Begaud B, et al. The reporting of a disproportionality analysis for drug safety signal detection using individual case safety reports in pharmacovigilance (READUS-PV): development and statement. <i>Drug Saf</i>. 2024;47(6):575-584.",
                     "doi": "10.1007/s40264-024-01421-9"},
    # FDA Drug Safety Communication: May 15, 2015 warning (20 FAERS cases, March 2013-June 6, 2014); label warnings added Dec 4, 2015 (73 cases to May 2015)
    "fda2015sglt2": {"text": "US Food and Drug Administration. FDA revises labels of SGLT2 inhibitors for diabetes to include warnings about too much acid in the blood and serious urinary tract infections. FDA Drug Safety Communication; 2015 Dec 4 (initial communication 2015 May 15).",
                     "url": "https://www.fda.gov/drugs/drug-safety-and-availability/fda-revises-labels-sglt2-inhibitors-diabetes-include-warnings-about-too-much-acid-blood-and-serious"},
    # PMID 28591538 (letter; MarketScan, new users, DPP-4 comparator, DKA hospitalization within 180 days, PS-matched HR about 2.2)
    "fralick2017": {"text": "Fralick M, Schneeweiss S, Patorno E. Risk of diabetic ketoacidosis after initiation of an SGLT2 inhibitor. <i>N Engl J Med</i>. 2017;376(23):2300-2302.",
                    "doi": "10.1056/NEJMc1701990"},
    # PMID 32716707 (CNODES; 208,757 matched pairs; HR 2.85, 1.99-4.08)
    "douros2020": {"text": "Douros A, Lix LM, Fralick M, et al. Sodium-glucose cotransporter-2 inhibitors and the risk for diabetic ketoacidosis: a multicenter cohort study. <i>Ann Intern Med</i>. 2020;173(6):417-425.",
                   "doi": "10.7326/M20-0289"},
}
