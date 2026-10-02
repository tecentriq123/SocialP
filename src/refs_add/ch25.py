# 25장(경제성 평가 논문 읽기) 참고문헌. 서지사항은 2026-10-02 PubMed(mcp PubMed lookup_article_by_citation → get_article_metadata)로
# 저자·제목·학술지·연도·권(호)·쪽·DOI를 확인했다(PMID는 주석). evers2005는 PubMed 레코드에 DOI가 없어 넣지 않았다.
# 이 장에서 쓰는 나머지 key는 refs_add/p4.py(husereau2022cheers, husereau2013cheers, sullivan2014bia, bae2018role, bae2022kguideline,
# bae2022sens, isporkorea2024, yoo2019access, lee2021rsa, hira2022icer, kim2022econ, caro2012ispor, eddy2012ispor, woods2020psm,
# latimer2013tsd14, nice2022manual, drummond2015book), refs_add/ch20.py, refs_add/ch21.py(nhis_sanjeong)에 있다.
# 본문에 인용한 내용의 확인 경로:
#   husereau2022cheers  PMC 전문(PMC8749494): 28개 항목, 항목별 문구, Box 1(2013년판과 달라진 점), 점수화 금지, 재정영향분석은 범위 밖,
#                       "Not applicable"/"Not reported", CHEERS를 인용한 논문 50편 가운데 적절히 쓴 것 42%(95% CI 28–56).
#   sullivan2014bia     PubMed 초록(핵심 요소, 비용 계산기 방식, 시나리오 분석) + 학술지 전문 페이지(valueinhealthjournal.com)를
#                       요약 도구(WebFetch)로 읽음: 분석기간 1–5년이 흔함, 예산 기간마다 제시, 할인하지 않음, 예산 보유자 관점, 열린 코호트.
#   philips2006, evers2005, vemer2016, drummond2009transfer, bell2006, xie2022  PubMed 초록.
#   drummond1996        PubMed 레코드에 초록이 없어 제목과 서지사항만 확인(본문에는 제목이 말하는 범위까지만 적음).
ADD = {
    # PMID 8704542
    "drummond1996": {"text": "Drummond MF, Jefferson TO. Guidelines for authors and peer reviewers of economic submissions to the BMJ. <i>BMJ</i>. 1996;313(7052):275-283.",
                     "doi": "10.1136/bmj.313.7052.275"},
    # PMID 16605282
    "philips2006": {"text": "Philips Z, Bojke L, Sculpher M, Claxton K, Golder S. Good practice guidelines for decision-analytic modelling in health technology assessment: a review and consolidation of quality assessment. <i>Pharmacoeconomics</i>. 2006;24(4):355-371.",
                    "doi": "10.2165/00019053-200624040-00006"},
    # PMID 15921065
    "evers2005": {"text": "Evers S, Goossens M, de Vet H, van Tulder M, Ament A. Criteria list for assessment of methodological quality of economic evaluations: Consensus on Health Economic Criteria. <i>Int J Technol Assess Health Care</i>. 2005;21(2):240-245."},
    # PMID 26660529
    "vemer2016": {"text": "Vemer P, Corro Ramos I, van Voorn GAK, Al MJ, Feenstra TL. AdViSHE: a validation-assessment tool of health-economic models for decision makers and model users. <i>Pharmacoeconomics</i>. 2016;34(4):349-361.",
                  "doi": "10.1007/s40273-015-0327-2"},
    # PMID 19900249
    "drummond2009transfer": {"text": "Drummond M, Barbieri M, Cook J, et al. Transferability of economic evaluations across jurisdictions: ISPOR Good Research Practices Task Force report. <i>Value Health</i>. 2009;12(4):409-418.",
                             "doi": "10.1111/j.1524-4733.2008.00489.x"},
    # PMID 35031088
    "husereau2022ee": {"text": "Husereau D, Drummond M, Augustovski F, et al. Consolidated Health Economic Evaluation Reporting Standards (CHEERS) 2022 explanation and elaboration: a report of the ISPOR CHEERS II Good Practices Task Force. <i>Value Health</i>. 2022;25(1):10-31.",
                       "doi": "10.1016/j.jval.2021.10.008"},
    # PMID 16495332
    "bell2006": {"text": "Bell CM, Urbach DR, Ray JG, et al. Bias in published cost effectiveness studies: systematic review. <i>BMJ</i>. 2006;332(7543):699-703.",
                 "doi": "10.1136/bmj.38737.607558.80"},
    # PMID 35732297
    "xie2022": {"text": "Xie F, Zhou T. Industry sponsorship bias in cost effectiveness analysis: registry based analysis. <i>BMJ</i>. 2022;377:e069573.",
                "doi": "10.1136/bmj-2021-069573"},
}
