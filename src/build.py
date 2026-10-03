"""Assemble the site from shell.html + content/chXX.html + figs + refs:
dist/stats.html (the course page) and dist/index.html (home, lists the courses)."""
import glob, json, os, re, html, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from refs import REFS  # noqa: E402

COURSE = {"key": "stats", "title": "보건통계학 기초", "file": "stats.html"}

GROUPS = [
    {"key": "p1", "label": "PART 1", "title": "보건의학통계 시작하기"},
    {"key": "p2", "label": "PART 2", "title": "중급 보건의학통계 맛보기"},
    {"key": "p3", "label": "PART 3", "title": "약물역학 연구 설계"},
    {"key": "p4", "label": "PART 4", "title": "약물경제성 평가"},
    {"key": "p5", "label": "PART 5", "title": "파이썬 실습",
     # 목차와 장 목록에서 접어 두는 묶음. 실습 번호가 upto 이하이면 그 묶음에 들어간다
     "subs": [{"key": "p5a", "label": "A", "title": "검정별 실습", "note": "1–13장과 짝", "upto": 13},
              {"key": "p5b", "label": "B", "title": "연구 유형별 실습", "note": "약물역학(14–19장)과 약물경제성 평가(20–25장)", "upto": 99}]},
    {"key": "ap", "label": "부록", "title": "추가 검정과 참고문헌"},
]

CHAPTERS = [
    ("ch00", "p1", "통계를 시작하기 전에",
     ["자료표와 관측 단위", "모집단과 표본", "평균·분산·표준편차", "정규분포와 표준화", "표준오차와 t 분포", "효과크기·신뢰구간·p값"]),
    ("ch01", "p1", "이 여덟 가지만은 꼭 알고 통계를 시작하자",
     ["자료의 분류", "가설을 검정하는 방법", "5% 유의수준과 다중비교", "자유도", "분포와 검정통계량",
      "중심극한정리", "모수적 방법과 비모수적 방법", "자료의 탐색 및 정규성 검정"]),
    ("ch02", "p1", "두 군의 크기 비교", ["독립표본 T 검정", "Mann-Whitney test"]),
    ("ch03", "p1", "치료 전과 후의 크기 비교", ["대응표본 T 검정", "Wilcoxon signed rank test"]),
    ("ch04", "p1", "세 군 이상의 크기 비교",
     ["독립된 세 군 이상의 크기를 비교하는 방법", "사후분석", "일원배치 분산분석", "Kruskal-Wallis test", "Jonckheere-Terpstra test"]),
    ("ch05", "p1", "비율을 비교하는 방법", ["카이제곱 검정", "Fisher의 정확한 검정", "선형 대 선형 결합"]),
    ("ch06", "p1", "연속형 변수 사이의 선형관계 추정",
     ["Pearson의 상관분석", "Spearman의 순위상관분석", "단순회귀분석", "다중회귀분석"]),
    ("ch07", "p1", "생존율의 추정 및 군의 생존율 비교", ["생존 연구의 준비", "Kaplan-Meier 생존분석", "로그순위법"]),
    # PART 1 review: optional 5th element sets extra META fields ("no" label, "review" flag)
    ("rv01", "p1", "분석 고르기 연습", ["분석을 고르는 순서", "같은 주제, 다른 연구 상황", "틀린 해석 고쳐 쓰기", "가상 논문 한 편 읽기"],
     {"no": "종합", "review": True}),
    ("ch08", "p2", "이 여덟 가지만 더 알고 중급 통계에 들어가자",
     ["95% 신뢰구간", "상대위험도와 교차비", "지수함수", "일반화 선형모형", "우도", "교란변수와 교호작용", "가변수의 설정",
      "단변수 분석과 다변수 분석"]),
    ("ch09", "p2", "질병의 위험인자에 대한 연구", ["로지스틱 회귀분석"]),
    ("ch10", "p2", "동일 개체에서 반복적으로 측정된 자료를 분석하는 방법",
     ["반복측정 분산분석", "자료구조의 변환", "선형 혼합모형", "일반화 추정 방정식"]),
    ("ch11", "p2", "생존율에 영향을 미치는 위험인자에 대한 연구", ["Cox의 비례위험모형"]),
    ("ch12", "p2", "질병의 발생률에 대한 연구", ["포아송 분포", "포아송 회귀분석"]),
    ("ch13", "p2", "두 치료법의 동등성, 비열등성을 검정하는 방법",
     ["동등성 검정과 비열등성 검정", "크기의 비열등성 검정 (높을수록 좋은 경우)", "크기의 비열등성 검정 (낮을수록 좋은 경우)",
      "비율의 비열등성 검정"]),
    ("ch14", "p3", "청구자료 코호트 연구의 설계",
     ["청구자료의 구조와 한계", "신규 사용자·활성 비교군 설계", "노출·결과·공변량의 정의", "추적과 분석 전략"]),
    ("ch15", "p3", "성향점수",
     ["보정할 변수 고르기", "성향점수의 추정과 겹침", "매칭과 가중", "균형 진단과 효과 추정", "남는 교란과 민감도 분석"]),
    ("ch16", "p3", "시간과 관련된 편향",
     ["불멸시간 편향", "랜드마크 분석", "시간의존 노출과 시간의존 Cox 모형", "기존 사용자 편향과 그 밖의 편향"]),
    ("ch17", "p3", "경쟁위험 분석", ["경쟁위험과 누적발생함수", "원인별 위험비와 Fine–Gray 위험비"]),
    ("ch18", "p3", "정책 효과의 평가", ["전후 비교의 함정", "중단시계열분석", "대조군이 있는 중단시계열", "이중차분법"]),
    ("ch19", "p3", "메타분석",
     ["체계적 문헌고찰의 절차", "효과의 통합: 고정효과와 무작위효과", "이질성, 하위군 분석, 메타회귀", "출판 편향과 민감도 분석",
      "통합 결과의 활용: NNT·NNH와 경제성 평가"]),
    ("ch20", "p4", "경제성 평가의 틀",
     ["경제성 평가가 답하는 질문", "경제성 평가의 네 가지 유형", "관점, 비교대안, 분석기간, 할인", "증분비용효과비와 비용효과평면"]),
    ("ch21", "p4", "비용 자료 분석",
     ["비용의 종류와 측정", "비용 분포와 평균 비용의 비교", "비용의 회귀분석"]),
    ("ch22", "p4", "효용과 QALY",
     ["건강 관련 삶의 질과 효용", "효용을 재는 방법", "QALY의 계산"]),
    ("ch23", "p4", "결정분석 모형",
     ["결정수형", "마르코프 코호트 모형", "분할생존모형", "생존곡선의 외삽"]),
    ("ch24", "p4", "불확실성 분석",
     ["결정론적 민감도 분석", "확률적 민감도 분석", "비용효과 수용곡선과 순편익"]),
    ("ch25", "p4", "경제성 평가 논문 읽기",
     ["보고 기준 CHEERS 2022", "모형 기반 경제성 평가 논문 읽기", "국내 급여 평가와 재정영향분석"]),
    ("lab00", "p5", "실습 환경 준비",
     ["파이썬과 주피터 노트북", "Google Colab으로 시작하기", "내 컴퓨터(Windows 11)에 설치하기", "셀 실행과 패키지 설치",
      "데이터 불러오기", "pandas로 데이터 살펴보기", "오류 메시지 읽는 법"]),
    ("lab01", "p5", "자료 탐색과 정규성 검정", ["실습 데이터 준비", "요약통계와 그래프", "Q-Q plot과 정규성 검정", "다중비교 보정"]),
    ("lab02", "p5", "두 군의 크기 비교", ["실습 데이터 준비", "독립표본 t 검정", "Mann-Whitney 검정"]),
    ("lab03", "p5", "치료 전과 후의 크기 비교", ["실습 데이터 준비", "대응표본 t 검정", "Wilcoxon 부호순위 검정"]),
    ("lab04", "p5", "세 군 이상의 크기 비교", ["실습 데이터 준비", "분산분석과 사후분석", "Kruskal-Wallis 검정", "Jonckheere-Terpstra 검정"]),
    ("lab05", "p5", "비율 비교", ["실습 데이터 준비", "카이제곱 검정", "Fisher의 정확한 검정", "선형 대 선형 결합"]),
    ("lab06", "p5", "상관분석과 선형회귀", ["실습 데이터 준비", "상관분석", "단순회귀분석", "다중회귀분석"]),
    ("lab07", "p5", "Kaplan-Meier와 로그순위법", ["실습 데이터 준비", "Kaplan-Meier 생존곡선", "로그순위법"]),
    ("lab08", "p5", "신뢰구간, 효과크기, 교란", ["신뢰구간과 효과크기", "일반화 선형모형과 우도비 검정", "교란과 층화 분석", "가변수와 다변수 분석"]),
    ("lab09", "p5", "로지스틱 회귀분석", ["실습 데이터 준비", "다변수 로지스틱 회귀", "모형 점검", "예측모형과 성능 평가"]),
    ("lab10", "p5", "반복측정 자료 분석", ["실습 데이터와 자료구조 변환", "반복측정 분산분석", "선형 혼합모형", "일반화 추정 방정식"]),
    ("lab11", "p5", "Cox 비례위험모형", ["실습 데이터 준비", "Cox 모형과 위험비", "비례위험 가정 점검", "보정 생존곡선과 층화 Cox"]),
    ("lab12", "p5", "포아송 회귀와 음이항 회귀", ["실습 데이터 준비", "발생률과 발생률비", "포아송 회귀와 과산포", "음이항 회귀"]),
    ("lab13", "p5", "동등성·비열등성 검정", ["평균 차이의 비열등성", "비율 차이의 비열등성", "생물학적 동등성"]),
    # PART 5 B: 14장 이후와 짝인 실습 (절 제목은 쓸 때 정한다)
    ("lab14", "p5", "청구자료에서 코호트 만들기", ["청구자료의 표 구조", "신규 사용자 코호트 만들기", "노출·결과·공변량 만들기", "추적 기간과 발생률", "과제"]),
    ("lab15", "p5", "성향점수 분석", ["실습 데이터 준비", "성향점수 추정과 겹침", "매칭과 가중", "균형 진단과 효과 추정", "과제"]),
    ("lab16", "p5", "불멸시간 편향과 시간의존 Cox 모형", ["실습 데이터 준비", "불멸시간 편향 재현하기", "랜드마크 분석", "시간의존 Cox 모형", "과제"]),
    ("lab17", "p5", "경쟁위험 분석", ["실습 데이터 준비", "누적발생함수", "원인별 위험비와 Fine–Gray 위험비", "과제"]),
    ("lab18", "p5", "중단시계열분석과 이중차분법", ["실습 데이터 준비", "중단시계열분석", "대조군이 있는 중단시계열", "이중차분법", "과제"]),
    ("lab19", "p5", "메타분석", ["실습 데이터 준비", "효과의 통합", "이질성과 하위군 분석", "깔때기 그림과 민감도 분석", "과제"]),
    ("lab21", "p5", "비용 자료 분석", ["실습 데이터 준비", "평균 비용의 비교와 부트스트랩", "비용의 회귀분석", "과제"]),
    ("lab23", "p5", "결정분석 모형 만들기", ["생존곡선의 적합과 외삽", "분할생존모형 만들기", "비용과 QALY, ICER", "모형 점검", "과제"]),
    ("lab24", "p5", "민감도 분석", ["실습 준비", "일원 민감도 분석과 토네이도 그림", "확률적 민감도 분석", "비용효과 수용곡선", "과제"]),
    ("ap01", "ap", "추가로 알아야 할 검정",
     ["McNemar 검정", "진단검사 정확도와 ROC 곡선", "의료비용 자료의 분석", "결측자료와 다중대체", "표본크기와 검정력", "일치도 분석"],
     {"no": "A", "appendix": True}),
    ("ap02", "ap", "참고문헌", [], {"no": "B", "appendix": True, "refs": True}),
]

GUIDE = [
    ("연속형 · 두 독립군", "독립표본 t 검정 / Mann-Whitney", "2"),
    ("연속형 · 같은 대상 전후", "대응표본 t 검정 / Wilcoxon 부호순위", "3"),
    ("연속형 · 세 군 이상", "분산분석 / Kruskal-Wallis", "4"),
    ("연속형 · 순서 있는 군의 추세", "Jonckheere-Terpstra", "4"),
    ("범주형 · 군 간 비율", "카이제곱 / Fisher 정확검정", "5"),
    ("연속형 두 변수의 관계", "Pearson / Spearman 상관, 회귀", "6"),
    ("생존시간 · 군 비교", "Kaplan-Meier, 로그순위법", "7"),
    ("이분형 · 위험인자 보정", "로지스틱 회귀", "9"),
    ("연속형 · 반복측정", "반복측정 ANOVA, 혼합모형, GEE", "10"),
    ("생존시간 · 위험인자 보정", "Cox 비례위험모형", "11"),
    ("사건 발생 건수·발생률", "포아송 / 음이항 회귀", "12"),
    ("새 치료가 '못하지 않음'", "비열등성·동등성 검정", "13"),
    ("관찰연구의 교란 보정", "성향점수(매칭·IPTW)", "15"),
    ("노출 시점이 늦게 정해짐", "랜드마크, 시간의존 Cox", "16"),
    ("다른 사건이 먼저 일어남", "누적발생함수, Fine–Gray", "17"),
    ("정책 시행 전후", "중단시계열, 이중차분법", "18"),
    ("여러 연구의 결과 통합", "메타분석", "19"),
    ("비용 비교 (치우친 금액)", "평균 차이와 부트스트랩, 감마 GLM", "21"),
    ("비용과 효과를 함께", "ICER, 결정분석 모형", "20·23"),
]

KO = "가나다라마바사아자차카타파하"
REFS_ID = "ap02"   # the references page (부록 B)


def ch_label(cid):
    """Reader-facing name of a chapter id: 3장, 실습 3, 부록 A, 분석 고르기 연습."""
    if cid.startswith("lab"):
        return f"실습 {int(cid[-2:])}"
    if cid.startswith("ap"):
        return "부록 " + "AB"[int(cid[-2:]) - 1]
    if cid.startswith("rv"):
        return "분석 고르기 연습"
    return f"{int(cid[-2:])}장"



# ---------------------------------------------------------------- deep folding
DEEP_TAG = '<span class="lv deep">심화</span>'
_H3 = re.compile(r'<h3\b[^>]*>.*?</h3>', re.S)
_H4 = re.compile(r'<h4\b[^>]*>.*?</h4>', re.S)
_DIVTAG = re.compile(r'<(/?)div\b[^>]*>')
_TAIL = ('<div class="practice">', '<div class="keypoints">', '<p class="lab-link">')


def _div_end(src, start):
    depth = 0
    for m in _DIVTAG.finditer(src, start):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return m.end()
    return -1


def _plain(h):
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', h.replace(DEEP_TAG, ''))).strip()


def _is_deep_heading(h):
    return DEEP_TAG in h or _plain(h).startswith('수식으로 보기')


def _fold_h4(chunk):
    hs = list(_H4.finditer(chunk))
    if not any(DEEP_TAG in m.group(0) for m in hs):
        return chunk
    out, pos = [], 0
    for i, m in enumerate(hs):
        if DEEP_TAG not in m.group(0):
            continue
        end = hs[i + 1].start() if i + 1 < len(hs) else len(chunk)
        for t in _TAIL:
            k = chunk.find(t, m.end(), end)
            if k != -1:
                end = k
        out.append(chunk[pos:m.start()])
        h = m.group(0)
        if DEEP_TAG not in h:
            h = h.replace('</h4>', ' ' + DEEP_TAG + '</h4>')
        out.append(f'<details class="deep-sec deep-h4"><summary>{h}</summary><div class="deep-body">{chunk[m.end():end]}</div></details>')
        pos = end
    out.append(chunk[pos:])
    return ''.join(out)


def _fold_boxes(chunk):
    """Wrap paper / callout boxes whose title carries the 심화 tag."""
    out, pos = [], 0
    for m in re.finditer(r'<div class="(paper|callout[^"]*)">', chunk):
        if m.start() < pos:
            continue
        end = _div_end(chunk, m.start())
        if end < 0:
            continue
        box = chunk[m.start():end]
        head = re.search(r'<div class="paper-h">(.*?)</div>', box, re.S) if m.group(1) == 'paper' else re.search(r'<p class="ct">(.*?)</p>', box, re.S)
        if not head:
            continue
        title = _plain(head.group(1))
        is_deep = DEEP_TAG in head.group(1) or 'deep' in m.group(1).split() or title.startswith('심화')
        if not is_deep:
            continue
        title = re.sub(r'^심화\s*·\s*', '', title)
        out.append(chunk[pos:m.start()])
        out.append(f'<details class="deep-box"><summary><span class="ds-t">{html.escape(title)}</span> {DEEP_TAG}</summary>{box}</details>')
        pos = end
    out.append(chunk[pos:])
    return ''.join(out)


_FOLD = re.compile(r'<!--FOLD:(.*?)-->(.*?)<!--/FOLD-->', re.S)


def _fold_markers(src):
    """<!--FOLD:summary text--> ... <!--/FOLD--> → a collapsed '심화' box (used to fold part of a subsection,
    e.g. the intermediate steps of a worked example, while a short visible summary stays above)."""
    return _FOLD.sub(lambda m: f'<details class="deep-box"><summary><span class="ds-t">{html.escape(m.group(1).strip())}</span> '
                               f'{DEEP_TAG}</summary><div class="deep-body">{m.group(2)}</div></details>', src)


def fold_deep(src):
    src = _fold_markers(src)
    """Collapse '심화' subsections (h3/h4 with the tag, and every '수식으로 보기') and '심화' boxes into <details>."""
    def sec(m):
        body = m.group(2)
        hs = list(_H3.finditer(body))
        out, pos = [], 0
        for i, h in enumerate(hs):
            end = hs[i + 1].start() if i + 1 < len(hs) else len(body)
            tail = end
            for t in _TAIL:
                k = body.find(t, h.end(), end)
                if k != -1:
                    tail = min(tail, k)
            out.append(_fold_boxes(_fold_h4(body[pos:h.start()])))
            region = body[h.start():tail]
            if _is_deep_heading(h.group(0)):
                hh = h.group(0)
                if DEEP_TAG not in hh:
                    hh = hh.replace('</h3>', ' ' + DEEP_TAG + '</h3>')
                out.append(f'<details class="deep-sec"><summary>{hh}</summary><div class="deep-body">{region[len(h.group(0)):]}</div></details>')
            else:
                out.append(_fold_boxes(_fold_h4(region)))
            pos = tail
        out.append(_fold_boxes(_fold_h4(body[pos:])))
        return m.group(1) + ''.join(out) + '</section>'
    return re.sub(r'(<section class="sec" id="[^"]+">)(.*?)</section>', sec, src, flags=re.S)

ONLY = None


def chapter_meta():
    out = []
    for cid, grp, title, secs, *extra in CHAPTERS:
        extra = extra[0] if extra else {}
        path = os.path.join(ROOT, "content", cid + ".html")
        ready = (os.path.exists(path) and (ONLY is None or cid in ONLY)) or cid == REFS_ID
        d = {"id": cid, "no": extra.get("no", cid[-2:]), "lab": cid.startswith("lab")}
        subs = next((g.get("subs") for g in GROUPS if g["key"] == grp), None)
        if subs and cid[-2:].isdigit():
            d["sub"] = next(sb["key"] for sb in subs if int(cid[-2:]) <= sb["upto"])
        for flag in ("review", "appendix", "refs"):
            if extra.get(flag):
                d[flag] = True
        d.update({
            "group": grp, "title": title, "ready": ready,
            "sections": [{"id": f"{cid}-s{i + 1}", "no": KO[i], "title": t} for i, t in enumerate(secs)],
        })
        out.append(d)
    return out


FIG_RE = re.compile(r"<!--FIG:([A-Za-z0-9_\-]+)-->")
CITE_RE = re.compile(r'<cite data-ref="([^"]+)"></cite>')


LINES_OPEN = '<ol class="lines">'
H4_LINES = re.compile(r'<h4>\s*코드 한 줄씩\s*</h4>\s*$')


def collapse_lines(src):
    """Wrap every <ol class="lines"> (line-by-line code explanation) in a closed <details> box.
    A directly preceding <h4>코드 한 줄씩</h4> heading is folded into the summary."""
    out, i = [], 0
    while True:
        j = src.find(LINES_OPEN, i)
        if j < 0:
            out.append(src[i:])
            break
        # find the matching </ol>, allowing nested lists
        depth, k = 0, j
        while True:
            a = src.find("<ol", k)
            b = src.find("</ol>", k)
            if a != -1 and a < b:
                depth += 1
                k = a + 3
            else:
                depth -= 1
                k = b + 5
                if depth == 0:
                    break
        block = src[j:k]
        before = src[i:j]
        m = H4_LINES.search(before)
        if m:
            before = before[:m.start()]
        n = block.count("<li")
        out.append(before)
        out.append(f'<details class="lines-box"><summary>코드 한 줄씩 설명<span class="cnt">{n}개 항목 · 눌러서 펼치기</span></summary>{block}</details>')
        i = k
    return "".join(out)


REAL_DIR = os.path.join(ROOT, "realpapers")
REAL_RE = re.compile(r'<div class="real"([^>]*)>(.*?)</div><!--/real-->', re.S)


def _attr(attrs, name, default=None):
    m = re.search(name + r'="([^"]*)"', attrs)
    return m.group(1) if m else default


def _paper_end(src, start):
    """index just after the </div> closing the <div class="paper"> that starts at `start`"""
    depth, k = 0, start
    while True:
        a = src.find("<div", k)
        b = src.find("</div>", k)
        if a != -1 and a < b:
            depth += 1
            k = a + 4
        else:
            depth -= 1
            k = b + 6
            if depth == 0:
                return k


def inject_real(cid, src):
    """Insert collapsed real-paper reading guides from realpapers/<section-id>.html after the chosen paper box.
    File format: one or more <div class="real" data-after="1" data-title="…">…</div><!--/real--> blocks."""
    for sec in re.finditer(r'<section class="sec" id="(%s-s\d+)">' % cid, src):
        pass
    files = sorted(glob.glob(os.path.join(REAL_DIR, cid + "-s*.html")))
    for f in files:
        sid = os.path.basename(f)[:-5]
        text = open(f, encoding="utf-8").read()
        blocks = REAL_RE.findall(text)
        if not blocks:
            print(f"WARNING: no real blocks in {f}")
            continue
        s0 = src.find(f'<section class="sec" id="{sid}">')
        if s0 < 0:
            print(f"WARNING: section {sid} not found for {f}")
            continue
        s1 = src.find("</section>", s0)
        # insert from the last block backwards so indices stay valid
        inserts = []
        for attrs, body in blocks:
            after = int(_attr(attrs, "data-after", "1"))
            title = _attr(attrs, "data-title", "")
            papers = [m.start() for m in re.finditer(r'<div class="paper">', src[s0:s1])]
            if papers:
                idx = min(after, len(papers)) - 1
                pos = _paper_end(src, s0 + papers[idx])
            else:
                kp = src.find('<div class="keypoints">', s0, s1)
                pos = kp if kp >= 0 else s1
            html_block = (f'<details class="real-box"><summary>실제 논문으로 읽어 보기'
                          f'<span class="cnt">{title} · 눌러서 펼치기</span></summary><div class="real-b">{body}</div></details>')
            inserts.append((pos, html_block))
        for pos, hb in sorted(inserts, key=lambda x: -x[0]):
            src = src[:pos] + hb + src[pos:]
    return src


def build():
    meta = chapter_meta()
    ready = [c for c in meta if c["ready"] and c["id"] != REFS_ID]
    order, used_in = [], {}
    bodies = {}
    for c in ready:
        src = open(os.path.join(ROOT, "content", c["id"] + ".html"), encoding="utf-8").read()

        def fig(m):
            p = os.path.join(ROOT, "figs", m.group(1) + ".html")
            if not os.path.exists(p):
                raise SystemExit(f"missing figure {m.group(1)} in {c['id']}")
            return open(p, encoding="utf-8").read()

        src = FIG_RE.sub(fig, src)
        for key in CITE_RE.findall(src):
            if key not in REFS:
                raise SystemExit(f"unknown ref {key} in {c['id']}")
            if key not in order:
                order.append(key)
            used_in.setdefault(key, [])
            if c["id"] not in used_in[key]:
                used_in[key].append(c["id"])
        # check section ids
        for s in c["sections"]:
            if f'id="{s["id"]}"' not in src:
                print(f"WARNING: section {s['id']} missing in {c['id']}")
        body = collapse_lines(inject_real(c["id"], src))
        bodies[c["id"]] = body if NOFOLD else fold_deep(body)
    # refs flagged "append" are numbered after all others (keeps already published numbers stable)
    order = [k for k in order if not REFS[k].get("append")] + [k for k in order if REFS[k].get("append")]
    num = {k: i + 1 for i, k in enumerate(order)}

    def cite(m):
        k = m.group(1)
        title = re.sub(r"<[^>]+>", "", REFS[k]["text"])
        return f'<a class="cite" href="#{REFS_ID}-ref-{k}" title="{html.escape(title)}">[{num[k]}]</a>'

    for cid in bodies:
        bodies[cid] = CITE_RE.sub(cite, bodies[cid])
        # merge adjacent citations: [1][2] stays readable; fine as-is

    # references chapter
    items = []
    for k in order:
        r = REFS[k]
        link = ""
        if r.get("doi"):
            link = f' <a href="https://doi.org/{r["doi"]}" target="_blank" rel="noopener">doi:{r["doi"]}</a>'
        elif r.get("url"):
            link = f' <a href="{r["url"]}" target="_blank" rel="noopener">링크</a>'
        chs = ", ".join(f'<a href="#{i}">{ch_label(i)}</a>' for i in used_in[k])
        items.append(f'<li id="{REFS_ID}-ref-{k}"><span class="rn">{num[k]}.</span><span>{r["text"]}{link}'
                     f'<span class="used">인용: {chs}</span></span></li>')
    books = [k for k, r in REFS.items() if r.get("book") and k not in num]
    book_items = "".join(f'<li id="{REFS_ID}-ref-{k}"><span class="rn">·</span><span>{REFS[k]["text"]}</span></li>' for k in books)
    ref_body = (
        '<p class="lead">본문에서 [번호]로 인용한 문헌을 인용 순서대로 모았습니다. '
        + ('개정하면서 새로 인용한 문헌은 목록 끝에 번호를 이어 붙였습니다. ' if any(REFS[k].get("append") for k in order) else '')
        + '각 항목 아래의 장 번호를 누르면 해당 장으로 이동합니다.</p>'
        f'<section class="sec" id="{REFS_ID}-cited"><h2>본문 인용 문헌</h2><ol class="reflist">{"".join(items)}</ol></section>'
        + (f'<section class="sec" id="{REFS_ID}-books"><h2>함께 보면 좋은 교재</h2><ol class="reflist">{book_items}</ol></section>' if book_items else "")
    )
    bodies[REFS_ID] = ref_body

    tpls = "\n".join(f'<template id="tpl-{cid}">{b}</template>' for cid, b in bodies.items())
    shell = open(os.path.join(ROOT, "shell.html"), encoding="utf-8").read()
    meta_json = json.dumps({"mode": "course", "course": COURSE, "chapters": meta, "groups": GROUPS, "guide": GUIDE},
                           ensure_ascii=False)
    wjs, css = [], []
    for c in ready:
        wp = os.path.join(ROOT, "widgets", c["id"] + ".js")
        if os.path.exists(wp):
            wjs.append("<script>\n" + open(wp, encoding="utf-8").read() + "\n</script>")
        cp = os.path.join(ROOT, "styles", c["id"] + ".css")
        if os.path.exists(cp):
            css.append(open(cp, encoding="utf-8").read())
    out = (shell.replace("{{META}}", meta_json.replace("</", "<\\/")).replace("{{TEMPLATES}}", tpls)
           .replace("{{WIDGETS_JS}}", "\n".join(wjs)).replace("{{EXTRA_CSS}}", "\n".join(css)))
    dst = OUT or os.path.join(ROOT, "dist", COURSE["file"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(out)
    print(f"built {dst}: {len(out) / 1024:.0f} KB, chapters ready: {[c['id'] for c in ready]}, refs: {len(order)}")
    if OUT is None and ONLY is None:
        build_home(shell, meta)


def build_home(shell, meta):
    """dist/index.html: the small home page that lists the courses (same shell, META.mode = "home")."""
    theory = [c for c in meta if not c["lab"] and not c.get("refs")]
    labs = [c for c in meta if c["lab"]]
    courses = [
        {"tag": "과목 · 통계", "title": COURSE["title"], "href": COURSE["file"],
         "status": ("0–25장 공개" if all(c["ready"] for c in theory) else f"{sum(c['ready'] for c in theory)}/{len(theory)}장 공개"),
         "desc": "평균·표준편차와 표준오차에서 시작해 통계 검정과 회귀모형, 청구자료 연구의 설계와 성향점수, 약물경제성 평가까지. "
                 "논문의 표와 그림에서 수치를 읽는 법을 먼저 다루고, 직접 분석에 필요한 내용과 파이썬 실습을 따로 두었습니다.",
         "meta": ["이론 0–25장 + 종합 연습 + 부록",
                  f"파이썬 실습 {sum(c['ready'] for c in labs)}개 공개" + (f", {sum(not c['ready'] for c in labs)}개 준비 중" if any(not c['ready'] for c in labs) else "")]},
        {"tag": "과목 · 머신러닝", "title": "머신러닝 기초", "href": None, "status": "준비 중",
         "desc": "넘파이·판다스 기초에서 시작해 사이킷런으로 분류, 회귀, 평가, 군집화를 다루고, "
                 "의료 자료로 예측모형을 만들고 논문의 예측모형을 읽는 법까지 이어집니다.",
         "meta": ["파이썬 기초 · 분류 · 회귀 · 평가 · 군집화", "의료 예측모형"]},
    ]
    meta_json = json.dumps({"mode": "home", "courses": courses}, ensure_ascii=False)
    out = (shell.replace("{{META}}", meta_json.replace("</", "<\\/")).replace("{{TEMPLATES}}", "")
           .replace("{{WIDGETS_JS}}", "").replace("{{EXTRA_CSS}}", ""))
    # the home page shows no formulas or code: drop the MathJax / highlight.js CDN scripts
    out = re.sub(r'<script src="https://cdn[^"]*"[^>]*></script>\n?', "", out)
    dst = os.path.join(ROOT, "dist", "index.html")
    open(dst, "w", encoding="utf-8").write(out)
    print(f"built {dst}: {len(out) / 1024:.0f} KB (home)")


OUT = None
NOFOLD = False

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="chapter ids to include, e.g. ch02 ch03")
    ap.add_argument("--out", help="output html path (default dist/index.html)")
    ap.add_argument("--nofold", action="store_true", help="do not collapse 심화 parts")
    a = ap.parse_args()
    if a.only:
        ONLY = set(a.only)
    if a.out:
        OUT = a.out
    if a.nofold:
        NOFOLD = True
    build()
