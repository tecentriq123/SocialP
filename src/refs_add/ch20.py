# Chapter 20 (경제성 평가의 틀) references. Bibliographic details checked against PubMed records
# (mcp PubMed lookup_article_by_citation + get_article_metadata, 2026-10-02; PMID in comment).
# Other keys used in the chapter come from refs_add/p4.py (shared PART 4 list) and refs_add/ch13.py (briggs2001).
ADD = {
    # PMID 402576
    "weinstein1977": {"text": "Weinstein MC, Stason WB. Foundations of cost-effectiveness analysis for health and medical practices. <i>N Engl J Med</i>. 1977;296(13):716-721.",
                      "doi": "10.1056/NEJM197703312961304"},
    # PMID 2115096
    "black1990": {"text": "Black WC. The CE plane: a graphic representation of cost-effectiveness. <i>Med Decis Making</i>. 1990;10(3):212-214.",
                  "doi": "10.1177/0272989X9001000308"},
    # PMID 7934713
    "cantor1994": {"text": "Cantor SB. Cost-effectiveness analysis, extended dominance, and ethics: a quantitative assessment. <i>Med Decis Making</i>. 1994;14(3):259-265.",
                   "doi": "10.1177/0272989X9401400308"},
    # PMID 10160090
    "karlsson1996": {"text": "Karlsson G, Johannesson M. The decision rules of cost-effectiveness analysis. <i>Pharmacoeconomics</i>. 1996;9(2):113-120.",
                     "doi": "10.2165/00019053-199609020-00003"},
    # PMID 9343807
    "stinnett1997": {"text": "Stinnett AA, Paltiel AD. Estimating CE ratios under second-order uncertainty: the mean ratio versus the ratio of means. <i>Med Decis Making</i>. 1997;17(4):483-489.",
                     "doi": "10.1177/0272989X9701700414"},
    # PMID 29779120 (PMC5999124)
    "attema2018": {"text": "Attema AE, Brouwer WBF, Claxton K. Discounting in economic evaluations. <i>Pharmacoeconomics</i>. 2018;36(7):745-758.",
                   "doi": "10.1007/s40273-018-0672-z"},
    # PMID 25692211 (PMC4781395). Checked with PubMed metadata 2026-10-15 (20장 d 절 '더 알아보기'):
    # central threshold estimate GBP 12,936 per QALY (2008 expenditure, 2008-10 mortality)
    "claxton2015threshold": {"text": "Claxton K, Martin S, Soares M, et al. Methods for the estimation of the National Institute for Health and Care Excellence cost-effectiveness threshold. <i>Health Technol Assess</i>. 2015;19(14):1-503.",
                             "doi": "10.3310/hta19140"},
    # PMID 27987642 (PMC5193154). Checked with PubMed metadata 2026-10-15: opportunity-cost-based thresholds for four
    # countries were 1%-59% of GDP per capita, lower than the 1-3 times GDP per capita suggested by WHO
    "woods2016threshold": {"text": "Woods B, Revill P, Sculpher M, Claxton K. Country-level cost-effectiveness thresholds: initial estimates and the need for further research. <i>Value Health</i>. 2016;19(8):929-935.",
                           "doi": "10.1016/j.jval.2016.02.017"},
    # NICE news article, read on nice.org.uk 2026-10-02 (dated 1 December 2025: standard threshold range raised from
    # £20,000-£30,000 to £25,000-£35,000 per QALY; applies from April 2026 to new and ongoing technology appraisals)
    "nice2025threshold": {"text": "National Institute for Health and Care Excellence. Changes to NICE's cost-effectiveness thresholds confirmed [news article]. 1 December 2025.",
                          "url": "https://www.nice.org.uk/news/articles/changes-to-nice-s-cost-effectiveness-thresholds-confirmed"},
}
