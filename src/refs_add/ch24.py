# 24장(불확실성 분석) 참고문헌. 서지사항은 2026-10-02 PubMed(mcp PubMed lookup_article_by_citation → get_article_metadata)로
# 저자·제목·학술지·연도·권(호)·쪽·DOI를 확인했다(PMID는 주석).
# 이 장에서 쓰는 나머지 key는 refs_add/p4.py(공통: briggs2012ispor, briggs2000uncertainty, briggs2006book, briggs2002psa,
# claxton2005psa, stinnett1998, vanhout1994, fenwick2001, fenwick2004, bae2022sens, bae2022kguideline, isporkorea2024,
# husereau2022cheers, lee2021rsa, yoo2019access, bae2018role, hira2022icer, cadth2017, bertram2021choice, sanders2016panel,
# nice2022manual)와 refs_add/ch20.py(stinnett1997)에 있다.
# bae2022sens(PMID 35652044)는 PMC 전문(PMC9149282)을 읽어 본문에 인용한 숫자를 확인했다: 50건 가운데 결정론적 민감도 분석 49건,
# 확률적 민감도 분석 18건(36%), 일원 분석 범위의 근거가 95% CI 14%·다른 자료원 38%·임의 값 48%, ICER 변화의 중앙값(약값 19.9% 등).
ADD = {
    # PMID 10537899
    "claxton1999": {"text": "Claxton K. The irrelevance of inference: a decision-making approach to the stochastic evaluation of health care technologies. <i>J Health Econ</i>. 1999;18(3):341-364.",
                    "doi": "10.1016/S0167-6296(98)00039-3"},
    # PMID 18489513 (제목의 'perfection'은 PubMed 레코드의 표기 그대로)
    "barton2008": {"text": "Barton GR, Briggs AH, Fenwick EAL. Optimal cost-effectiveness decisions: the role of the cost-effectiveness acceptability curve (CEAC), the cost-effectiveness acceptability frontier (CEAF), and the expected value of perfection information (EVPI). <i>Value Health</i>. 2008;11(5):886-897.",
                   "doi": "10.1111/j.1524-4733.2008.00358.x"},
    # PMID 30051268
    "hatswell2018": {"text": "Hatswell AJ, Bullement A, Briggs A, Paulden M, Stevenson MD. Probabilistic sensitivity analysis in cost-effectiveness models: determining model convergence in cohort models. <i>Pharmacoeconomics</i>. 2018;36(12):1421-1426.",
                     "doi": "10.1007/s40273-018-0697-3"},
    # PMID 19508655
    "bojke2009": {"text": "Bojke L, Claxton K, Sculpher M, Palmer S. Characterizing structural uncertainty in decision analytic models: a review and application of methods. <i>Value Health</i>. 2009;12(5):739-749.",
                  "doi": "10.1111/j.1524-4733.2008.00502.x"},
    # PMID 11910068
    "briggs2002box": {"text": "Briggs AH, O'Brien BJ, Blackhouse G. Thinking outside the box: recent advances in the analysis and presentation of uncertainty in cost-effectiveness studies. <i>Annu Rev Public Health</i>. 2002;23:377-401.",
                      "doi": "10.1146/annurev.publhealth.23.100901.140534"},
    # PMID 25336432
    "wilson2015": {"text": "Wilson ECF. A practical guide to value of information analysis. <i>Pharmacoeconomics</i>. 2015;33(2):105-121.",
                   "doi": "10.1007/s40273-014-0219-x"},
    # PMID 33313990 (PMC7790801)
    "vreman2021": {"text": "Vreman RA, Geenen JW, Knies S, Mantel-Teeuwisse AK, Leufkens HGM, Goettsch WG. The application and implications of novel deterministic sensitivity analysis methods. <i>Pharmacoeconomics</i>. 2021;39(1):1-17.",
                   "doi": "10.1007/s40273-020-00979-3"},
}
