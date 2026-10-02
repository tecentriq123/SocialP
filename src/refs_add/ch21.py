# Chapter 21 (비용 자료 분석) references.
# 확인(2026-10-02): 학술지 논문은 PubMed 레코드(PMID 주석)로 저자·제목·학술지·연도·권(호)·쪽·DOI를 확인했다.
#   bang2000은 PubMed에 없어 Oxford Academic의 논문 페이지(academic.oup.com/biomet/article/87/2/329/283692)로 확인했다.
#   nhis_sanjeong은 국민건강보험공단 누리집의 제도 안내 페이지를 직접 읽어 확인했다(암: 요양급여비용총액의 100분의 5, 등록일부터 5년, 비급여 제외).
# 검수(2026-10-02): barber2000, manning1998, lumley2002, lin1997은 PubMed 레코드로, bang2000은 Oxford Academic 페이지로,
#   nhis_sanjeong은 공단 페이지(요양급여비용총액의 100분의 5, 등록일로부터 5년간, 전액본인부담·선별급여·비급여 제외)로 다시 확인했다.
# 다른 파일에 이미 있어 그대로 쓰는 key: thompson2000cost (refs.py), duan1983, manning2001 (refs_add/ch06.py),
#   mihaylova2011 (refs_add/ch08.py), deb2018 (refs_add/ch14c.py), kimja2017 (refs_add/ch14.py),
#   barber2004glm, drummond2015book, kim2022econ, isporkorea2024, bae2022kguideline, husereau2022cheers,
#   nice2022manual, sanders2016panel, lee2021rsa (refs_add/p4.py).
ADD = {
    # PMID 11113956
    "barber2000": {"text": "Barber JA, Thompson SG. Analysis of cost data in randomized trials: an application of the non-parametric bootstrap. <i>Stat Med</i>. 2000;19(23):3219-3236.",
                   "doi": "10.1002/1097-0258(20001215)19:23&lt;3219::AID-SIM623&gt;3.0.CO;2-P"},
    # PMID 10180919
    "manning1998": {"text": "Manning WG. The logged dependent variable, heteroscedasticity, and the retransformation problem. <i>J Health Econ</i>. 1998;17(3):283-295.",
                    "doi": "10.1016/S0167-6296(98)00025-3"},
    # PMID 11910059
    "lumley2002": {"text": "Lumley T, Diehr P, Emerson S, Chen L. The importance of the normality assumption in large public health data sets. <i>Annu Rev Public Health</i>. 2002;23:151-169.",
                   "doi": "10.1146/annurev.publhealth.23.100901.140546"},
    # PMID 9192444 (DOI는 PubMed 레코드에 없음)
    "lin1997": {"text": "Lin DY, Feuer EJ, Etzioni R, Wax Y. Estimating medical costs from incomplete follow-up data. <i>Biometrics</i>. 1997;53(2):419-434."},
    # Oxford Academic 논문 페이지로 확인
    "bang2000": {"text": "Bang H, Tsiatis AA. Estimating medical costs with censored data. <i>Biometrika</i>. 2000;87(2):329-343.",
                 "doi": "10.1093/biomet/87.2.329"},
    "nhis_sanjeong": {"text": "국민건강보험공단. 본인일부부담금 산정특례 제도 [제도 안내]. 2026년 10월 2일 확인.",
                      "url": "https://www.nhis.or.kr/static/html/wbma/c/wbmac0215.html"},
}
