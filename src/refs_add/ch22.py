# Chapter 22 (효용과 QALY).
# 서지사항 확인(2026-10-02): 모두 PubMed 레코드(mcp PubMed lookup_article_by_citation → get_article_metadata)로
# 저자·제목·학술지·연도·권(호)·쪽·DOI를 확인했다.
#   weinstein2009 PMID 19250132, whitehead2010 PMID 21037243, brazier2005patients PMID 16466271,
#   wolowacz2016 PMID 27712695, brazier2019utility PMID 30832964, feeny2002 PMID 11802084, aaronson1993 PMID 8433390.
# 다른 파일에 이미 있어 그대로 쓰는 key(refs_add/p4.py): torrance1986, euroqol1990, dolan1997, brazier2002, herdman2011,
#   lee2009eq5d3l, kim2016eq5d5l, manca2005, wailoo2017mapping, petrou2015maps, hong2021utility, isporkorea2024,
#   bae2022kguideline, hira2022icer, nice2022manual, cadth2017, sanders2016panel, husereau2022cheers, bertram2021choice.
#   matthews1990은 refs_add/ch10.py.
ADD = {
    "weinstein2009": {"text": "Weinstein MC, Torrance G, McGuire A. QALYs: the basics. <i>Value Health</i>. 2009;12 Suppl 1:S5-S9.",
                      "doi": "10.1111/j.1524-4733.2009.00515.x"},
    "whitehead2010": {"text": "Whitehead SJ, Ali S. Health outcomes in economic evaluation: the QALY and utilities. <i>Br Med Bull</i>. 2010;96:5-21.",
                      "doi": "10.1093/bmb/ldq033"},
    "brazier2005patients": {"text": "Brazier J, Akehurst R, Brennan A, et al. Should patients have a greater role in valuing health states? <i>Appl Health Econ Health Policy</i>. 2005;4(4):201-208.",
                            "doi": "10.2165/00148365-200504040-00002"},
    "wolowacz2016": {"text": "Wolowacz SE, Briggs A, Belozeroff V, et al. Estimating health-state utility for economic models in clinical studies: an ISPOR Good Research Practices Task Force report. <i>Value Health</i>. 2016;19(6):704-719.",
                     "doi": "10.1016/j.jval.2016.06.001"},
    "brazier2019utility": {"text": "Brazier J, Ara R, Azzabi I, et al. Identification, review, and use of health state utilities in cost-effectiveness models: an ISPOR Good Practices for Outcomes Research Task Force report. <i>Value Health</i>. 2019;22(3):267-275.",
                           "doi": "10.1016/j.jval.2019.01.004"},
    "feeny2002": {"text": "Feeny D, Furlong W, Torrance GW, et al. Multiattribute and single-attribute utility functions for the Health Utilities Index Mark 3 system. <i>Med Care</i>. 2002;40(2):113-128.",
                  "doi": "10.1097/00005650-200202000-00006"},
    "aaronson1993": {"text": "Aaronson NK, Ahmedzai S, Bergman B, et al. The European Organization for Research and Treatment of Cancer QLQ-C30: a quality-of-life instrument for use in international clinical trials in oncology. <i>J Natl Cancer Inst</i>. 1993;85(5):365-376.",
                     "doi": "10.1093/jnci/85.5.365"},
}
