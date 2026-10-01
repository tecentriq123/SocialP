"""Assemble index.html from shell.html + content/chXX.html + figs + refs."""
import glob, json, os, re, html, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from refs import REFS  # noqa: E402

GROUPS = [
    {"key": "p1", "label": "PART 1", "title": "보건의학통계 시작하기"},
    {"key": "p2", "label": "PART 2", "title": "중급 보건의학통계 맛보기"},
    {"key": "p3", "label": "PART 3", "title": "파이썬 실습"},
    {"key": "ap", "label": "부록", "title": "더 알아야 할 기법과 참고문헌"},
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
    ("lab00", "p3", "실습 환경 준비",
     ["파이썬과 주피터 노트북", "Google Colab으로 시작하기", "내 컴퓨터(Windows 11)에 설치하기", "셀 실행과 패키지 설치",
      "데이터 불러오기", "pandas로 데이터 살펴보기", "오류 메시지 읽는 법"]),
    ("lab01", "p3", "자료 탐색과 정규성 검정", ["실습 데이터 준비", "요약통계와 그래프", "Q-Q plot과 정규성 검정", "다중비교 보정"]),
    ("lab02", "p3", "두 군의 크기 비교", ["실습 데이터 준비", "독립표본 t 검정", "Mann-Whitney 검정"]),
    ("lab03", "p3", "치료 전과 후의 크기 비교", ["실습 데이터 준비", "대응표본 t 검정", "Wilcoxon 부호순위 검정"]),
    ("lab04", "p3", "세 군 이상의 크기 비교", ["실습 데이터 준비", "분산분석과 사후분석", "Kruskal-Wallis 검정", "Jonckheere-Terpstra 검정"]),
    ("lab05", "p3", "비율 비교", ["실습 데이터 준비", "카이제곱 검정", "Fisher의 정확한 검정", "선형 대 선형 결합"]),
    ("lab06", "p3", "상관분석과 선형회귀", ["실습 데이터 준비", "상관분석", "단순회귀분석", "다중회귀분석"]),
    ("lab07", "p3", "Kaplan-Meier와 로그순위법", ["실습 데이터 준비", "Kaplan-Meier 생존곡선", "로그순위법"]),
    ("lab08", "p3", "신뢰구간, 효과크기, 교란", ["신뢰구간과 효과크기", "일반화 선형모형과 우도비 검정", "교란과 층화 분석", "가변수와 다변수 분석"]),
    ("lab09", "p3", "로지스틱 회귀분석", ["실습 데이터 준비", "다변수 로지스틱 회귀", "모형 점검", "예측모형과 성능 평가"]),
    ("lab10", "p3", "반복측정 자료 분석", ["실습 데이터와 자료구조 변환", "반복측정 분산분석", "선형 혼합모형", "일반화 추정 방정식"]),
    ("lab11", "p3", "Cox 비례위험모형", ["실습 데이터 준비", "Cox 모형과 위험비", "비례위험 가정 점검", "보정 생존곡선과 층화 Cox"]),
    ("lab12", "p3", "포아송 회귀와 음이항 회귀", ["실습 데이터 준비", "발생률과 발생률비", "포아송 회귀와 과산포", "음이항 회귀"]),
    ("lab13", "p3", "동등성·비열등성 검정", ["평균 차이의 비열등성", "비율 차이의 비열등성", "생물학적 동등성"]),
    ("ch14", "ap", "추가로 알아야 할 검정",
     ["McNemar 검정", "진단검사 정확도와 ROC 곡선", "성향점수 방법", "경쟁위험 분석", "시간의존 변수와 불멸시간 편향",
      "메타분석", "중단시계열분석과 이중차분법", "의료비용 자료의 분석", "결측자료와 다중대체", "표본크기와 검정력",
      "일치도 분석"]),
    ("ch15", "ap", "참고문헌", []),
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
    ("관찰연구의 교란 보정", "성향점수(PSM·IPTW)", "14"),
]

KO = "가나다라마바사아자차카타파하"


ONLY = None


def chapter_meta():
    out = []
    for cid, grp, title, secs, *extra in CHAPTERS:
        extra = extra[0] if extra else {}
        path = os.path.join(ROOT, "content", cid + ".html")
        ready = (os.path.exists(path) and (ONLY is None or cid in ONLY)) or cid == "ch15"
        d = {"id": cid, "no": extra.get("no", cid[-2:]), "lab": cid.startswith("lab")}
        if extra.get("review"):
            d["review"] = True
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
    ready = [c for c in meta if c["ready"] and c["id"] != "ch15"]
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
        bodies[c["id"]] = collapse_lines(inject_real(c["id"], src))
    # refs flagged "append" are numbered after all others (keeps already published numbers stable)
    order = [k for k in order if not REFS[k].get("append")] + [k for k in order if REFS[k].get("append")]
    num = {k: i + 1 for i, k in enumerate(order)}

    def cite(m):
        k = m.group(1)
        title = re.sub(r"<[^>]+>", "", REFS[k]["text"])
        return f'<a class="cite" href="#ch15-ref-{k}" title="{html.escape(title)}">[{num[k]}]</a>'

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
        chs = ", ".join(f'<a href="#{i}">' + (f'실습 {int(i[-2:])}' if i.startswith("lab") else f'{int(i[-2:])}장') + '</a>' for i in used_in[k])
        items.append(f'<li id="ch15-ref-{k}"><span class="rn">{num[k]}.</span><span>{r["text"]}{link}'
                     f'<span class="used">인용: {chs}</span></span></li>')
    books = [k for k, r in REFS.items() if r.get("book") and k not in num]
    book_items = "".join(f'<li id="ch15-ref-{k}"><span class="rn">·</span><span>{REFS[k]["text"]}</span></li>' for k in books)
    ref_body = (
        '<p class="lead">본문에서 [번호]로 인용한 문헌을 인용 순서대로 모았습니다. '
        + ('개정하면서 새로 인용한 문헌은 목록 끝에 번호를 이어 붙였습니다. ' if any(REFS[k].get("append") for k in order) else '')
        + '각 항목 아래의 장 번호를 누르면 해당 장으로 이동합니다.</p>'
        f'<section class="sec" id="ch15-cited"><h2>본문 인용 문헌</h2><ol class="reflist">{"".join(items)}</ol></section>'
        + (f'<section class="sec" id="ch15-books"><h2>함께 보면 좋은 교재</h2><ol class="reflist">{book_items}</ol></section>' if book_items else "")
    )
    bodies["ch15"] = ref_body

    tpls = "\n".join(f'<template id="tpl-{cid}">{b}</template>' for cid, b in bodies.items())
    shell = open(os.path.join(ROOT, "shell.html"), encoding="utf-8").read()
    meta_json = json.dumps({"chapters": meta, "groups": GROUPS, "guide": GUIDE}, ensure_ascii=False)
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
    dst = OUT or os.path.join(ROOT, "dist", "index.html")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(out)
    print(f"built {dst}: {len(out) / 1024:.0f} KB, chapters ready: {[c['id'] for c in ready]}, refs: {len(order)}")


OUT = None

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="chapter ids to include, e.g. ch02 ch03")
    ap.add_argument("--out", help="output html path (default dist/index.html)")
    a = ap.parse_args()
    if a.only:
        ONLY = set(a.only)
    if a.out:
        OUT = a.out
    build()
