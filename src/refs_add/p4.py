# PART 4 (약물경제성 평가, 20–25장) 공통 참고문헌. P4_FACTS.md에서 key로 인용한다.
# 서지사항 확인(2026-10-02):
#   - 학술지 논문: PubMed 레코드(mcp PubMed get_article_metadata)로 저자·제목·학술지·연도·권(호)·쪽·DOI 확인.
#     저자가 많은 논문은 이 사이트의 기존 표기처럼 앞의 3명 + et al.로 적었다.
#   - husereau2022cheers: BMJ판(공개, PMC8749494 전문 확인). 같은 성명이 Value Health 2022;25(1):3-9
#     (doi 10.1016/j.jval.2021.11.1351) 등 여러 학술지에 동시 게재되었다.
#   - 책: Oxford University Press 제품 페이지(global.oup.com)에서 제목·저자·판·출판일 확인.
#     neumann2016book은 OUP 페이지의 출판일이 2016-11-01이다(판권 연도 2017로 인용하는 문헌도 많다).
#   - latimer2013tsd14, woods2017tsd19: NICE DSU(sheffield.ac.uk/nice-dsu) 페이지와 보고서 PDF 표지에서 제목·저자·날짜 확인.
#     woods2017tsd19의 저자·날짜는 York 대학 연구정보(pure.york.ac.uk) 기록과도 일치.
#   - nice2022manual: nice.org.uk/process/pmg36 (2022-01-31 발간, history 페이지 기준 최종 갱신 2026-03-31).
#   - cadth2017: cda-amc.ca 지침 페이지(4th Edition, 최종 갱신 2017-03-20).
#   - hira2021guideline: HIRA 공지(2021-02-24, 「의약품 경제성 평가 지침」 개정 안내(21.1.))와 HIRA OAK Repository 검색 기록
#     (발간등록번호 G000DB6-2021-25)으로 서지사항만 확인. 원문 파일은 이번 작업 환경에서 열지 못했다.
#   - hira2022icer: HIRA 공지사항(2022-12-16) 페이지 확인. 표의 숫자는 당일 보도(헬스코리아뉴스 등)로 확인.
#   - isporkorea2024: ISPOR 'Pharmacoeconomic Guidelines Around the World' 한국 페이지(최종 갱신 2024-05-21).
# 다른 파일에 이미 있어 여기서 다시 정의하지 않고 그대로 쓰는 key:
#   guyot2012, latimer2013 (Med Decis Making 논문), manning2001, mihaylova2011, duan1983, thompson2000cost,
#   matthews1990, briggs2001 (비용-최소화 분석), fleurence2007 (율과 확률), kimja2017, kiml2014, seong2017.
ADD = {
    # ---------- 국내 지침과 급여 제도 ----------
    "hira2021guideline": {"text": "건강보험심사평가원. <i>의약품 경제성 평가 지침</i>. 원주: 건강보험심사평가원; 2021. (2021년 1월 개정, 발간등록번호 G000DB6-2021-25)",
                          "url": "https://repository.hira.or.kr/handle/2019.oak/2541"},
    "bae2022kguideline": {"text": "Bae EY, Hong J, Bae S, Hahn S, An H, Hwang EJ, Lee SM, Lee TJ. Korean guidelines for pharmacoeconomic evaluations: updates in the third version. <i>Appl Health Econ Health Policy</i>. 2022;20(4):467-477.",
                          "doi": "10.1007/s40258-022-00721-4"},
    "bae2013kguideline": {"text": "Bae S, Lee S, Bae EY, Jang S. Korean guidelines for pharmacoeconomic evaluation (second and updated version): consensus and compromise. <i>Pharmacoeconomics</i>. 2013;31(4):257-267.",
                          "doi": "10.1007/s40273-012-0021-6"},
    "bae2008kguideline": {"text": "Bae EY. Guidelines for economic evaluation of pharmaceuticals in Korea [in Korean]. <i>J Prev Med Public Health</i>. 2008;41(2):80-83.",
                          "doi": "10.3961/jpmph.2008.41.2.80"},
    "isporkorea2024": {"text": "ISPOR. Pharmacoeconomic guidelines around the world: South Korea. Updated May 21, 2024.",
                       "url": "https://www.ispor.org/heor-resources/more-heor-resources/pharmacoeconomic-guidelines/pe-guideline-detail/south-korea"},
    "kim2022econ": {"text": "Kim Y, Kim Y, Lee HJ, et al. The primary process and key concepts of economic evaluation in healthcare. <i>J Prev Med Public Health</i>. 2022;55(5):415-423.",
                    "doi": "10.3961/jpmph.22.195"},
    "bae2018role": {"text": "Bae EY, Kim HJ, Lee HJ, et al. Role of economic evidence in coverage decision-making in South Korea. <i>PLoS One</i>. 2018;13(10):e0206121.",
                    "doi": "10.1371/journal.pone.0206121"},
    "bae2016hta": {"text": "Bae EY, Hong JM, Kwon HY, et al. Eight-year experience of using HTA in drug reimbursement: South Korea. <i>Health Policy</i>. 2016;120(6):612-620.",
                   "doi": "10.1016/j.healthpol.2016.03.013"},
    "bae2019hta": {"text": "Bae EY. Role of health technology assessment in drug policies: Korea. <i>Value Health Reg Issues</i>. 2019;18:24-29.",
                   "doi": "10.1016/j.vhri.2018.03.009"},
    "lee2021rsa": {"text": "Lee B, Bae EY, Bae S, et al. How can we improve patients' access to new drugs under uncertainties?: South Korea's experience with risk sharing arrangements. <i>BMC Health Serv Res</i>. 2021;21(1):967.",
                   "doi": "10.1186/s12913-021-06919-x"},
    "yoo2019access": {"text": "Yoo SL, Kim DJ, Lee SM, et al. Improving patient access to new drugs in South Korea: evaluation of the national drug formulary system. <i>Int J Environ Res Public Health</i>. 2019;16(2):288.",
                      "doi": "10.3390/ijerph16020288"},
    "yu2025waiver": {"text": "Yu SR. Improving the reimbursement process for new drugs: a case study of a two-waiver system in South Korea. <i>J Eval Clin Pract</i>. 2025;31(3):e70074.",
                     "doi": "10.1111/jep.70074"},
    "hong2025highpriced": {"text": "Hong J, Bae EY, Cha S, Lee J. The value-for-money assessment and funding arrangements for high-priced drugs in an era of uncertainty: a comparative analysis of national health technology assessment agencies in South Korea, England, Australia, and Canada. <i>BMC Health Serv Res</i>. 2025;25(1):74.",
                           "doi": "10.1186/s12913-025-12207-9"},
    "bae2022sens": {"text": "Bae S, Lee J, Bae EY. How sensitive is sensitivity analysis?: evaluation of pharmacoeconomic submissions in Korea. <i>Front Pharmacol</i>. 2022;13:884769.",
                    "doi": "10.3389/fphar.2022.884769"},
    "hong2021utility": {"text": "Hong J, Bae EY. A review of utility measurement methods used in pharmacoeconomic submissions to HIRA in South Korea: methodological consistency and areas for improvement. <i>Pharmacoeconomics</i>. 2021;39(10):1109-1121.",
                        "doi": "10.1007/s40273-021-01066-x"},
    "hira2022icer": {"text": "건강보험심사평가원. 경제성평가 제출 약제의 비용효과성 평가결과(ICER) 공개 [공지사항]. 2022년 12월 16일.",
                     "url": "https://www.hira.or.kr/bbsDummy.do?pgmid=HIRAA020002000100&brdScnBltNo=4&brdBltNo=10021&pageIndex=1"},
    # ---------- 외국 지침, 임계값 ----------
    "nice2022manual": {"text": "National Institute for Health and Care Excellence. <i>NICE health technology evaluations: the manual</i> (PMG36). London: NICE; 2022.",
                       "url": "https://www.nice.org.uk/process/pmg36"},
    "sanders2016panel": {"text": "Sanders GD, Neumann PJ, Basu A, et al. Recommendations for conduct, methodological practices, and reporting of cost-effectiveness analyses: Second Panel on Cost-Effectiveness in Health and Medicine. <i>JAMA</i>. 2016;316(10):1093-1103.",
                         "doi": "10.1001/jama.2016.12195"},
    "neumann2016book": {"text": "Neumann PJ, Sanders GD, Russell LB, Siegel JE, Ganiats TG, eds. <i>Cost-Effectiveness in Health and Medicine</i>. 2nd ed. New York: Oxford University Press; 2016.", "book": True},
    "cadth2017": {"text": "CADTH. <i>Guidelines for the Economic Evaluation of Health Technologies: Canada</i>. 4th ed. Ottawa: CADTH; 2017.",
                  "url": "https://www.cda-amc.ca/guidelines-economic-evaluation-health-technologies-canada-4th-edition"},
    "bertram2021choice": {"text": "Bertram MY, Lauer JA, Stenberg K, Edejer TTT. Methods for the economic evaluation of health care interventions for priority setting in the health system: an update from WHO CHOICE. <i>Int J Health Policy Manag</i>. 2021;10(11):673-677.",
                          "doi": "10.34172/ijhpm.2020.244"},
    "bertram2016threshold": {"text": "Bertram MY, Lauer JA, De Joncheere K, et al. Cost-effectiveness thresholds: pros and cons. <i>Bull World Health Organ</i>. 2016;94(12):925-930.",
                             "doi": "10.2471/BLT.15.164418"},
    "neumann2014threshold": {"text": "Neumann PJ, Cohen JT, Weinstein MC. Updating cost-effectiveness—the curious resilience of the $50,000-per-QALY threshold. <i>N Engl J Med</i>. 2014;371(9):796-797.",
                             "doi": "10.1056/NEJMp1405158"},
    # ---------- 보고 기준, 모형 모범 지침 ----------
    "husereau2022cheers": {"text": "Husereau D, Drummond M, Augustovski F, et al. Consolidated Health Economic Evaluation Reporting Standards 2022 (CHEERS 2022) statement: updated reporting guidance for health economic evaluations. <i>BMJ</i>. 2022;376:e067975.",
                           "doi": "10.1136/bmj-2021-067975"},
    "husereau2013cheers": {"text": "Husereau D, Drummond M, Petrou S, et al. Consolidated Health Economic Evaluation Reporting Standards (CHEERS) statement. <i>BMJ</i>. 2013;346:f1049.",
                           "doi": "10.1136/bmj.f1049"},
    "caro2012ispor": {"text": "Caro JJ, Briggs AH, Siebert U, Kuntz KM; ISPOR-SMDM Modeling Good Research Practices Task Force. Modeling good research practices—overview: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force-1. <i>Value Health</i>. 2012;15(6):796-803.",
                      "doi": "10.1016/j.jval.2012.06.012"},
    "roberts2012ispor": {"text": "Roberts M, Russell LB, Paltiel AD, et al. Conceptualizing a model: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force-2. <i>Value Health</i>. 2012;15(6):804-811.",
                         "doi": "10.1016/j.jval.2012.06.016"},
    "siebert2012ispor": {"text": "Siebert U, Alagoz O, Bayoumi AM, et al. State-transition modeling: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force-3. <i>Value Health</i>. 2012;15(6):812-820.",
                         "doi": "10.1016/j.jval.2012.06.014"},
    "briggs2012ispor": {"text": "Briggs AH, Weinstein MC, Fenwick EAL, et al. Model parameter estimation and uncertainty: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force-6. <i>Value Health</i>. 2012;15(6):835-842.",
                        "doi": "10.1016/j.jval.2012.04.014"},
    "eddy2012ispor": {"text": "Eddy DM, Hollingworth W, Caro JJ, et al. Model transparency and validation: a report of the ISPOR-SMDM Modeling Good Research Practices Task Force-7. <i>Value Health</i>. 2012;15(6):843-850.",
                      "doi": "10.1016/j.jval.2012.04.012"},
    "sullivan2014bia": {"text": "Sullivan SD, Mauskopf JA, Augustovski F, et al. Budget impact analysis—principles of good practice: report of the ISPOR 2012 Budget Impact Analysis Good Practice II Task Force. <i>Value Health</i>. 2014;17(1):5-14.",
                        "doi": "10.1016/j.jval.2013.08.2291"},
    "latimer2013tsd14": {"text": "Latimer N. <i>NICE DSU Technical Support Document 14: Survival analysis for economic evaluations alongside clinical trials – extrapolation with patient-level data</i>. Sheffield: NICE Decision Support Unit; 2011 (last updated March 2013).",
                         "url": "https://www.sheffield.ac.uk/nice-dsu/tsds/survival-analysis"},
    "woods2017tsd19": {"text": "Woods B, Sideris E, Palmer S, Latimer N, Soares M. <i>NICE DSU Technical Support Document 19: Partitioned survival analysis for decision modelling in health care: a critical review</i>. Sheffield: NICE Decision Support Unit; 2017.",
                       "url": "https://www.sheffield.ac.uk/nice-dsu/tsds/partitioned-survival-analysis"},
    "woods2020psm": {"text": "Woods BS, Sideris E, Palmer S, Latimer N, Soares M. Partitioned survival and state transition models for healthcare decision making in oncology: where are we now? <i>Value Health</i>. 2020;23(12):1613-1621.",
                     "doi": "10.1016/j.jval.2020.08.2094"},
    "sonnenberg1993": {"text": "Sonnenberg FA, Beck JR. Markov models in medical decision making: a practical guide. <i>Med Decis Making</i>. 1993;13(4):322-338.",
                       "doi": "10.1177/0272989X9301300409"},
    "briggs1998markov": {"text": "Briggs A, Sculpher M. An introduction to Markov modelling for economic evaluation. <i>Pharmacoeconomics</i>. 1998;13(4):397-409.",
                         "doi": "10.2165/00019053-199813040-00003"},
    # ---------- 교과서 ----------
    "briggs2006book": {"text": "Briggs A, Claxton K, Sculpher M. <i>Decision Modelling for Health Economic Evaluation</i>. Oxford: Oxford University Press; 2006.", "book": True},
    "drummond2015book": {"text": "Drummond MF, Sculpher MJ, Claxton K, Stoddart GL, Torrance GW. <i>Methods for the Economic Evaluation of Health Care Programmes</i>. 4th ed. Oxford: Oxford University Press; 2015.", "book": True},
    # ---------- 순편익, 수용곡선, 확률적 민감도 분석 ----------
    "stinnett1998": {"text": "Stinnett AA, Mullahy J. Net health benefits: a new framework for the analysis of uncertainty in cost-effectiveness analysis. <i>Med Decis Making</i>. 1998;18(2 Suppl):S68-S80.",
                     "doi": "10.1177/0272989X98018002S09"},
    "vanhout1994": {"text": "van Hout BA, Al MJ, Gordon GS, Rutten FF. Costs, effects and C/E-ratios alongside a clinical trial. <i>Health Econ</i>. 1994;3(5):309-319.",
                    "doi": "10.1002/hec.4730030505"},
    "fenwick2001": {"text": "Fenwick E, Claxton K, Sculpher M. Representing uncertainty: the role of cost-effectiveness acceptability curves. <i>Health Econ</i>. 2001;10(8):779-787.",
                    "doi": "10.1002/hec.635"},
    "fenwick2004": {"text": "Fenwick E, O'Brien BJ, Briggs A. Cost-effectiveness acceptability curves—facts, fallacies and frequently asked questions. <i>Health Econ</i>. 2004;13(5):405-415.",
                    "doi": "10.1002/hec.903"},
    "briggs2000uncertainty": {"text": "Briggs AH. Handling uncertainty in cost-effectiveness models. <i>Pharmacoeconomics</i>. 2000;17(5):479-500.",
                              "doi": "10.2165/00019053-200017050-00006"},
    "briggs2002psa": {"text": "Briggs AH, Goeree R, Blackhouse G, O'Brien BJ. Probabilistic analysis of cost-effectiveness models: choosing between treatment strategies for gastroesophageal reflux disease. <i>Med Decis Making</i>. 2002;22(4):290-308.",
                      "doi": "10.1177/0272989X0202200408"},
    "claxton2005psa": {"text": "Claxton K, Sculpher M, McCabe C, et al. Probabilistic sensitivity analysis for NICE technology assessment: not an optional extra. <i>Health Econ</i>. 2005;14(4):339-347.",
                       "doi": "10.1002/hec.985"},
    # ---------- 비용 자료 ----------
    "barber2004glm": {"text": "Barber J, Thompson S. Multiple regression of cost data: use of generalised linear models. <i>J Health Serv Res Policy</i>. 2004;9(4):197-204.",
                      "doi": "10.1258/1355819042250249"},
    # ---------- 효용, EQ-5D ----------
    "torrance1986": {"text": "Torrance GW. Measurement of health state utilities for economic appraisal. <i>J Health Econ</i>. 1986;5(1):1-30.",
                     "doi": "10.1016/0167-6296(86)90020-2"},
    "euroqol1990": {"text": "EuroQol Group. EuroQol—a new facility for the measurement of health-related quality of life. <i>Health Policy</i>. 1990;16(3):199-208.",
                    "doi": "10.1016/0168-8510(90)90421-9"},
    "dolan1997": {"text": "Dolan P. Modeling valuations for EuroQol health states. <i>Med Care</i>. 1997;35(11):1095-1108.",
                  "doi": "10.1097/00005650-199711000-00002"},
    "brazier2002": {"text": "Brazier J, Roberts J, Deverill M. The estimation of a preference-based measure of health from the SF-36. <i>J Health Econ</i>. 2002;21(2):271-292.",
                    "doi": "10.1016/S0167-6296(01)00130-8"},
    "herdman2011": {"text": "Herdman M, Gudex C, Lloyd A, et al. Development and preliminary testing of the new five-level version of EQ-5D (EQ-5D-5L). <i>Qual Life Res</i>. 2011;20(10):1727-1736.",
                    "doi": "10.1007/s11136-011-9903-x"},
    "lee2009eq5d3l": {"text": "Lee YK, Nam HS, Chuang LH, et al. South Korean time trade-off values for EQ-5D health states: modeling with observed values for 101 health states. <i>Value Health</i>. 2009;12(8):1187-1193.",
                      "doi": "10.1111/j.1524-4733.2009.00579.x"},
    "kim2016eq5d5l": {"text": "Kim SH, Ahn J, Ock M, et al. The EQ-5D-5L valuation study in Korea. <i>Qual Life Res</i>. 2016;25(7):1845-1852.",
                      "doi": "10.1007/s11136-015-1205-2"},
    "manca2005": {"text": "Manca A, Hawkins N, Sculpher MJ. Estimating mean QALYs in trial-based cost-effectiveness analysis: the importance of controlling for baseline utility. <i>Health Econ</i>. 2005;14(5):487-496.",
                  "doi": "10.1002/hec.944"},
    "wailoo2017mapping": {"text": "Wailoo AJ, Hernandez-Alava M, Manca A, et al. Mapping to estimate health-state utility from non-preference-based outcome measures: an ISPOR Good Practices for Outcomes Research Task Force report. <i>Value Health</i>. 2017;20(1):18-27.",
                          "doi": "10.1016/j.jval.2016.11.006"},
    "petrou2015maps": {"text": "Petrou S, Rivero-Arias O, Dakin H, et al. The MAPS reporting statement for studies mapping onto generic preference-based outcome measures: explanation and elaboration. <i>Pharmacoeconomics</i>. 2015;33(10):993-1011.",
                       "doi": "10.1007/s40273-015-0312-9"},
}
