# 작성 가이드 — 사회약학 연구방법 노트 / 보건통계학 기초

이 사이트는 사회약학(약물역학·약물경제학·HEOR) 연구실에 새로 들어온 대학원생이 **통계 원리를 이해하고, 논문의 표·그림·문장에서 수치를 읽어 낼 수 있게** 하는 학습 자료입니다. 사이트 주인(약사, 사회약학 대학원생)이 1장을 보고 특히 마음에 들어 한 부분은 **'논문에서는 이렇게 보입니다' 상자**입니다. "처음 공부하는 사람이 논문을 봐도 '아 이런 뜻이었구나' 할 수 있게 자세히 설명"한 것이 핵심 가치입니다. 모든 장에서 이 상자를 가장 공들여 만드세요.

**완성된 모범 예시: `content/ch01.html`** — 반드시 먼저 전체를 읽고, 구조·문체·밀도·컴포넌트 사용법을 그대로 따르세요. 그림 생성 예시는 `gen/fig_ch01.py`.

## 파일과 빌드

- 본문: `content/chXX.html` (HTML 조각. `<html>`, `<head>`, `<style>`, `<script>` 금지)
- 그림 생성 스크립트: `gen/fig_chXX.py` → `figs/chXX_이름.html` 파일을 만들고, 본문에서 `<!--FIG:chXX_이름-->` 한 줄로 삽입
- 숫자 계산 스크립트: `gen/nums_chXX.py` (본문의 모든 통계량을 실제로 계산해 확인한 코드. 재현 가능하게 남겨 두세요)
- 참고문헌 추가: `refs_add/chXX.py` 에 `ADD = {"key": {"text": "...", "doi": "..."}}` (아래 '인용' 참고)
- (선택) 인터랙티브 위젯: `widgets/chXX.js` — 아래 '위젯' 참고
- (선택, 가급적 쓰지 말 것) 추가 CSS: `styles/chXX.css` — 모든 선택자는 `.x-chXX` 로 시작해야 함
- **수정 금지 파일**: `shell.html`, `build.py`, `refs.py`, `svgplot.py`, `content/ch01.html`, 다른 장의 파일. 공용 파일에 기능이 필요하면 최종 보고에 적으세요.
- 미리보기: `python3 tools/preview.py chXX <출력폴더>` (데스크톱), `--mobile --dark` 옵션으로 폰·다크모드. 그림·논문상자·표마다 PNG가 저장되니 **직접 Read로 열어 보고** 겹치는 라벨, 잘린 글자, 가로 넘침을 고치세요. MathJax와 웹폰트는 이 환경에서 막혀 있어 수식이 TeX 원문으로 보이는 것은 정상입니다.
- 사용 가능한 파이썬 패키지: numpy, scipy, pandas, matplotlib(그림은 svgplot 사용), sklearn. **statsmodels·lifelines는 설치할 수 없습니다** (pip 차단). 생존분석·회귀는 numpy/scipy로 직접 구현하세요 (Kaplan-Meier, log-rank, OLS, Newton-Raphson 등은 짧게 구현 가능).

## 장 구조

```html
<p class="lead">이 장에서 다루는 것 2–3문장 (페이지 머리로 옮겨짐)</p>

<section class="sec" id="chXX-s1">
<h2><span class="sec-no">가.</span>목차에 적힌 절 제목 그대로</h2>
...
</section>
```
- 절 id는 `chXX-s1`, `chXX-s2` … 순서대로. 절 제목은 목차 제목과 **글자 그대로 같게** (build.py의 CHAPTERS 참고).
- 절 안의 소제목은 `<h3>`, 그 아래는 `<h4>`.
- 소제목은 **담백하게**: "개념", "숫자로 따라가기", "가정과 확인 방법", "결과 읽기"처럼. "놀라운 ~의 비밀", 콜론으로 부제를 다는 과장된 제목, 이모지 금지.

## 한 절(방법 하나)의 권장 흐름

1. **한눈에 보기** (방법을 다루는 절에서 권장)
```html
<dl class="glance">
  <div><dt>언제 쓰나</dt><dd>...</dd></div>
  <div><dt>귀무가설</dt><dd>...</dd></div>
  <div><dt>가정</dt><dd>...</dd></div>
  <div><dt>논문 보고 형식</dt><dd>...</dd></div>
</dl>
```
2. **개념**: 비유 + 직관. 왜 이 방법이 필요한지.
3. **숫자로 따라가기**: 치료군/대조군, 환자군 A/B 같은 구체적 연구 상황의 작은 가상 자료로 손계산을 단계별로 보여 줍니다(표 + 계산식). 모든 수치는 `gen/nums_chXX.py` 로 실제 계산해 **본문·표·그림의 숫자가 서로 정확히 맞게** 하세요.
4. **그림**: 원리를 보여 주는 그림 1–3개 (svgplot 사용).
5. **가정과 확인 방법**, 가정이 깨졌을 때의 대안.
6. **논문에서는 이렇게 보입니다** 상자 (절마다 최소 1개, 핵심 절은 2–3개). 아래 상세 규칙.
7. **흔한 오해 / 주의**, **사회약학 연구에서는**, **심화** 상자를 필요한 곳에.
8. (선택) **R로 해 보기** 코드 접기 상자.
9. **정리** (keypoints) — 절의 마지막.

## 문체와 용어

- **합니다체**. 짧고 분명한 문장. 학술적이고 담백하게. 과장·감탄·수사 금지. em-dash(—)로 끼워 넣는 문장, "~가 아니라 ~다" 반복, "주목할 점은" 같은 상투어를 피하세요.
- 통계 용어는 **처음 나올 때 한글과 영어를 병기**: `<strong>표준오차</strong><span class="en">(standard error, SE)</span>`. 영어 이름이 그대로 쓰이는 검정(Mann-Whitney 등)은 영어 이름 + 한글 설명.
- 약학·의학 용어(HbA1c, eGFR, NYHA 등)는 병기 불필요.
- 수식은 **적게**: 직관적 설명과 숫자 예시가 중심이고, 핵심 수식만 곁들입니다. 긴 유도·증명 금지. 인라인 `\( ... \)`, 블록은 `<div class="eq">\[ ... \]<p>기호 설명</p></div>`. HTML이므로 `<`, `>`, `&` 는 `&lt;`, `&gt;`, `&amp;` 로. TeX 안의 `<` 도 `&lt;` 로 쓰거나 `\lt` 사용.
- 다른 장 참조: "(8장)", "(14장 다 절)". 존재하는 장·절만 참조 (build.py 목차 참고).
- 사이트 주인이 적극 교정을 원하는 오류: OR/RR 혼동, HR을 위험비(risk ratio)처럼 해석, 상관=인과, PSM 후 편향이 완전히 제거됐다는 생각, AUC를 정확도로 보는 것. 관련 장에서는 '흔한 오해' 상자로 다루세요.
- 사회약학 맥락 예시를 적극 활용: 건강보험 청구자료(HIRA·NHIS), 신규 사용자 설계, 약물 순응도(PDC/MPR), 약사 중재, 의료비용, 약물 이상반응, 신세포암 등 종양 코호트.

## '논문에서는 이렇게 보입니다' 상자 (가장 중요)

```html
<div class="paper">
<div class="paper-h"><b>논문에서는 이렇게 보입니다 · 무엇을 보여 주는지</b><span>가상의 예시</span></div>
<div class="paper-b">
<p>짧은 도입 (선택)</p>
<p class="excerpt">영어 논문 문장 ... <span class="hl">강조할 수치</span> <span class="mk">1</span> ...</p>
<!-- 또는 학술지 표 -->
<div class="tbl-wrap"><table class="jt">
<caption><b>Table 2.</b> ...</caption>
<thead><tr><th>...</th><th class="r">...</th></tr></thead>
<tbody><tr><td>...</td><td class="r">... <span class="mk">2</span></td></tr></tbody></table></div>
<p class="jt-foot">약어·각주</p>
<ol class="marks">
<li><span class="mk">1</span><span><b>한 줄 요지.</b> 이 숫자가 무엇이고, 어떻게 계산되며, 무엇을 뜻하고, 무엇을 뜻하지 않는지. 처음 보는 사람도 이해하도록 구체적으로.</span></li>
</ol>
</div></div>
```
- 문장·표는 **실제 논문 형식의 영어**로, 직접 만든 가상의 연구입니다. 실제 논문 문장을 옮겨 오지 마세요.
- 번호 표식(`<span class="mk">n</span>`)을 붙인 **모든 수치·표현을 번호 목록에서 한국어로 해석**합니다: 이 숫자가 어디서 나왔는지(가능하면 검산), 어떻게 읽는지, 흔히 잘못 읽는 방식, 확인해야 할 다른 정보.
- 다양한 유형을 섞으세요: Methods의 통계분석 문단, Results 문장, Table (baseline/outcome/regression), Figure 읽기(그림 + 번호 해설), 통계 프로그램 출력(`<pre class="out">`).
- 예시 연구의 숫자도 실제로 계산해 내적으로 일관되게 (평균·SD·n → t, p, CI가 맞아야 함).

## 컴포넌트 목록 (새 클래스를 만들지 말고 이것만 쓰세요)

| 용도 | 마크업 |
|---|---|
| 일반 표 | `<div class="tbl-wrap"><table class="tbl">…</table></div>`, 숫자 칸 `class="r"`, 가운데 `c`, 강조 행 `<tr class="hi">` |
| 학술지 표 | `<table class="jt">` + `<caption>`, 들여쓰기 셀 `class="ind"`, 소제목 행 `<tr class="grp">`, 아래 각주 `<p class="jt-foot">` |
| 논문 문장 | `<p class="excerpt">`, 강조 `<span class="hl">`, 번호 `<span class="mk">1</span>` |
| 번호 해설 | `<ol class="marks"><li><span class="mk">1</span><span>…</span></li></ol>` |
| 프로그램 출력 | `<pre class="out">…</pre>` (주석 `<span class="cm">`) |
| 수식 | `<div class="eq">\[…\]<p>설명</p></div>` |
| 콜아웃 | `<div class="callout note|warn|tip|deep"><p class="ct">라벨 · 제목</p><p>…</p></div>` — note=참고, warn=흔한 오해/주의, tip=사회약학 연구에서는, deep=심화 |
| 틀린 문장 표시 | 흔한 오해 상자 안에서 `<span class="wrong">틀린 주장</span>` 뒤에 올바른 설명 |
| 단계 흐름 | `<div class="flow"><div><div><b>단계</b><p>설명</p></div></div>…</div>` (자동 번호) |
| 2×2 판단표 | `<div class="matrix">` (ch01 1종/2종 오류 예 참고) |
| 두 갈래 분류 | `<div class="tree">` (ch01 가 절 참고) |
| 정리 | `<div class="keypoints"><p class="kt">정리</p><ul><li>…</li></ul></div>` |
| R 코드 | `<details class="code"><summary>R로 해 보기 · 주제</summary><pre><code>…</code></pre></details>` |
| 영어 병기 | `<span class="en">(English term)</span>` |
| 한눈에 보기 | `<dl class="glance"><div><dt>…</dt><dd>…</dd></div>…</dl>` |

R 코드: RStudio(Windows)에서 실행한다는 전제. 짧게(10–20줄), **각 줄에 주석**으로 역할과 출력에서 볼 부분을 설명 ("# 출력의 p-value 줄을 봅니다"). 존재하는 함수·인자만 쓰세요 (base R, survival 패키지 등 표준 패키지 위주).

## 그림 (svgplot)

`gen/fig_ch01.py` 를 참고하세요. 핵심 API (`svgplot.py`):
- `Plot(xlim, ylim, w=600, h=340, xlabel, ylabel, xticks, yticks, xtickfmt, ytickfmt, xticklabels=[(v,'라벨')], ygrid, xgrid, show_yaxis)`
- `.line(xs, ys, s=1, dash=False)`, `.step(xs, ys, s)` (KM 계단), `.fill_between(xs, y0s, y1s, s)` 또는 `cls="a4"`, `.points(xs, ys, s, r, hollow)`, `.ticks_marks(xs, ys, s)` (중도절단 표시), `.hist(edges, counts, s)`, `.bars(xs, heights, width, s)`, `.vline/.hline(v, dash, y0/y1 or x0/x1)`, `.seg(xa,ya,xb,yb)`, `.text(x, y, s, anchor, cls="lbl|lbl strong|lbl mute|lbl small", dx, dy)`, `.text_px(X, Y, …)`, `.legend([(라벨, s, "line|dash|box|soft|dot")], X, Y)`, `.svg(aria)`
- `figure([svg,...], caption, cols=1|2|3)` → `<figure>` HTML. `panel_title(plot, "제목")`. `forest(rows, xlim, ref, log, …)` 포레스트 플롯.
- 색은 계열 번호로만: s1 파랑(주 계열), s2 주황(대비 계열), s3 초록(세 번째), s4 회색(기준·귀무·참조). 색만으로 구분하지 말고 범례나 직접 라벨을 붙이세요.
- 그림 캡션: `그림 X-n. 설명` (장 번호-장 안의 순서, 등장 순서대로). 캡션에 무엇을 보면 되는지 한 문장.
- 단일 그림은 viewBox 폭 600 전후, 2단 패널은 420 전후. 라벨이 선·다른 라벨과 겹치지 않는지 미리보기로 확인. 곡선 점은 200–400개 정도면 충분 (파일 크기).
- 가로가 긴 수평 표(number at risk 등)는 SVG 안에 텍스트로 그리거나 HTML 표로.

## 인용

- 문장 끝에 `<cite data-ref="key"></cite>` 를 붙이면 빌드 시 [번호] 링크가 되고 15장 참고문헌에 모입니다.
- 먼저 `refs.py` 와 `refs_add/*.py` 에 이미 있는 키를 쓰세요. 없으면 `refs_add/chXX.py` 에 추가 (`ADD = {...}`, 키는 `저자연도` 형식 소문자, 예: `kaplan1958`). text 형식은 refs.py와 같게(Vancouver 비슷하게, 학술지명 `<i>`), 가능하면 `doi`.
- **서지정보에 확신이 있는 문헌만** 인용하세요 (저자, 제목, 학술지, 연도, 권(호), 쪽, DOI). 고전 방법론 논문, BMJ Statistics Notes, 표준 교과서 위주. 불확실하면 인용하지 마세요. 절당 0–3개면 충분합니다.

## 위젯 (선택, 장당 최대 1개)

정말 이해에 도움이 되는 경우에만. `widgets/chXX.js`:
```js
window.EXTRA_WIDGETS["chXX_name"] = function (el, H) {
  // H = { mulberry32(seed)->rng, gauss(rng), fmt(v, digits), esc(str) }
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">시뮬레이션</span></p><p class="wd">설명</p> ...`;
  // 컨트롤: <div class="wrow">, 버튼 묶음 <div class="seg"><button aria-pressed="true">, <button class="wbtn">, <input type="range">
  // 결과: <div class="stats"><div class="stat"><div class="sl">라벨</div><div class="sv">값</div></div></div>
  // 차트: <div class="wchart"><svg viewBox=… class="viz">…</svg></div>  (svgplot과 같은 클래스: grid, axis, tick, axlab, ln s1, f1, a1 …)
};
```
본문에는 `<div class="widget" data-widget="chXX_name"></div>`. 로드 직후 기본 상태가 그려져 있어야 합니다. `alert/confirm/localStorage` 금지. ch01의 clt·fwer 위젯(`shell.html` 안)을 참고.

## 최종 확인 체크리스트

- [ ] 절 id·제목이 목차와 일치, 모든 절에 '논문에서는 이렇게 보입니다' 상자와 '정리'
- [ ] 모든 수치를 스크립트로 계산·검산 (본문·표·그림·해설 번호 사이 불일치 없음)
- [ ] `python3 build.py --only chXX --out /tmp/...` 가 경고 없이 통과 (missing figure / unknown ref / section missing)
- [ ] preview PNG를 데스크톱·모바일로 직접 확인, 가로 넘침 없음
- [ ] 통계적으로 정확한가? 확신이 부족한 주장은 빼거나 불확실하다고 명시

## 용어 통일 (사이트 전체, 반드시 지킬 것)

| 영어 | 이 사이트의 한국어 표기 |
|---|---|
| risk ratio / relative risk (RR) | **상대위험도** |
| hazard ratio (HR) | **위험비** (hazard는 위험률) |
| odds ratio (OR) | **오즈비** (처음 나올 때 "교차비라고도 함") |
| incidence rate ratio (IRR) | **발생률비** |
| risk difference (RD) | 위험차 |
| confidence interval | 신뢰구간 |
| censoring / risk set | 중도절단 / 위험집합 |
| proportional hazards | 비례위험 |
| log-rank test | 로그순위법(로그순위 검정) |
| confounder / effect modification / interaction | 교란변수 / 효과수정 / 교호작용 |
| mediator / collider | 매개변수 / 충돌변수 |
| likelihood / maximum likelihood estimation | 우도 / 최대우도추정 |
| generalized linear model | 일반화 선형모형 |
| linear mixed model / GEE | 선형 혼합모형 / 일반화 추정 방정식 |
| fixed effect / random effect | 고정효과 / 무작위효과 |
| dummy variable / reference category | 가변수 / 기준범주 |
| non-inferiority margin | 비열등성 한계(마진) |

'위험비'를 RR의 뜻으로 쓰지 마세요. 누적 위험의 비는 상대위험도입니다.

## 이미 완성된 장 (Part 1) — 중복을 피하고 연결하세요

- 1장: 자료 유형, 가설검정, p-value, 1·2종 오류, 검정력·표본수 기초, 다중비교, 자유도, z·t·χ²·F, SD vs SE, 중심극한정리, 모수/비모수, 정규성. 로지스틱 계수의 R/SAS 출력 비교(β 0.642, OR 1.90) 예시가 마 절에 있음.
- 2·3장: t 검정, Welch, Mann-Whitney(AUC와의 관계), 대응 t, Wilcoxon, 평균으로의 회귀.
- 4장: 분산분석, 사후분석, Kruskal-Wallis, Jonckheere-Terpstra.
- 5장: 카이제곱, Fisher, 추세검정. 2×2 표에서 위험차·상대위험도·오즈비 계산과 "오즈비를 상대위험도처럼 읽기" 오해가 가 절에 있음(8장 나 절은 더 깊게).
- 6장: 상관, 단순·다중 선형회귀(출력 읽기, VIF, 로그 변환 결과, Table 2 오류, ANCOVA, 매개변수 보정 주의).
- 7장: 생존자료, 중도절단, 청구자료 코호트 설계, Kaplan-Meier, Greenwood, 중앙생존기간, RMST, 로그순위법과 가중 검정, "HR 0.73 ≠ 3년 사망위험 27% 감소" 오해(S₁ = S₀^HR 계산). Cox·비례위험 가정은 11장으로 미뤄 둠.
관련 장의 해당 부분을 읽고 용어·예시가 어긋나지 않게 하며, "(5장 가 절)"처럼 연결하세요.

## 추가 규칙 (Part 1 검수에서 나온 것 — Part 2부터 반드시 지킬 것)

### 용어 통일 (사이트 전체)
| 개념 | 표기 | 첫 등장 병기 |
|---|---|---|
| risk ratio / relative risk (RR) | **상대위험도** | `상대위험도<span class="en">(relative risk, risk ratio, RR)</span>` |
| hazard ratio (HR) | **위험비** | `위험비<span class="en">(hazard ratio, HR)</span>` — 절대로 RR을 위험비라고 부르지 마세요 |
| hazard | **위험률** | `위험률<span class="en">(hazard)</span>` |
| odds ratio (OR) | **오즈비** | `오즈비<span class="en">(odds ratio, OR; 교차비)</span>` — 8장 나 절 제목은 목차대로 '교차비'이므로 두 이름이 같다는 것을 밝힐 것 |
| incidence rate ratio | **발생률비** | `발생률비<span class="en">(incidence rate ratio, IRR)</span>` |
| risk difference | **위험차** | `위험차<span class="en">(risk difference, RD)</span>` |
| confounder / effect modification / interaction | 교란변수 / 효과수정 / 교호작용 | |
| mediator / collider | 매개변수 / 충돌변수 | |
| censoring | 중도절단 | |

### 검수에서 자주 나온 실수
- **반올림**: 표에 적힌 값은 소수점 반올림을 정확히 (예: 0.0605 → 0.060, 11.845 → 11.8). 신뢰구간 끝자리까지 스크립트 출력에서 옮기세요. 표시된 반올림 값으로 독자가 검산할 수 있는 식은 반올림 전 값을 쓰거나 "반올림 차이"를 밝히세요.
- **번호 표식 누락**: 해설 목록(`<ol class="marks">`)의 번호마다 본문·표·출력에 같은 번호 표식(`<span class="mk">n</span>`)이 있어야 합니다. `<pre class="out">` 안에도 표식을 넣을 수 있습니다.
- **소프트웨어 동작·출력 이름**: R/SAS/SPSS의 기본값, 출력 라벨, 버전별 동작을 단정하지 마세요. 확인하지 못했으면 "(버전에 따라 다를 수 있음)"처럼 누그러뜨리세요. R 출력은 형식을 흉내 낸 것이므로 숫자는 스크립트 계산값과 정확히 맞추세요.
- **보정 변수의 시점**: 회귀 예제에서 노출 이후에 측정된 변수(추적 중 순응도 등)를 교란변수로 보정하지 마세요. 공변량은 기저 시점(노출 이전) 값으로 정의하세요.
- **한국 자료원 서술**: HIRA 청구자료에는 자격 정보와 병원 밖 사망 정보가 없고, NHIS 자료에는 자격 DB(자격 상실, 사망 연계)가 있습니다. 확실하지 않은 제도·자료원 설명은 누그러뜨리세요.
- **과장 금지**: "항상", "모든", "가장 흔한" 같은 단정은 근거가 있을 때만.

### 이미 다룬 내용과 약속된 내용
- `PART1_OUTLINE.md`: 1–7장에서 이미 설명한 내용 목록. 반복하지 말고 "(2장 가 절)"처럼 참조하세요.
- `FORWARD_REFS.txt`: 1–7장이 "8장 X 절에서 다룹니다"처럼 뒤 장에 미룬 내용. 해당 장을 쓸 때 그 약속을 지키세요.


## 2026-09-30 변경 사항 (사이트 주인 요청 — 모든 장에 적용, 이전 규칙보다 우선)

1. **흔한 오해의 틀린 문장**: 계속 `<span class="wrong">틀린 주장</span>`으로 감싸세요. 이제 취소선 대신 CSS가 문장 뒤에 **"(X)"** 를 자동으로 붙입니다. 본문에 (X)를 직접 쓰지 마세요.
2. **R 금지, 파이썬 기준**: 연구실은 파이썬을 씁니다. Part 1·2 본문에는 `<details class="code">` 코드 상자(“R로 해 보기”)를 **넣지 않습니다**. 실습은 모두 Part 3(파이썬 실습)으로 모읍니다. 대신 각 절의 '정리' 바로 앞에 아래 한 줄을 넣으세요 (N = 장 번호와 같은 실습 번호):
   `<p class="lab-link">파이썬 실습 · <a href="#labNN">실습 N</a>에서 이 절의 분석을 공개 데이터로 직접 해 봅니다.</p>`
   실습 번호는 장 번호와 같습니다: 실습 1 ↔ 1장, …, 실습 13 ↔ 13장 (id: lab01 … lab13, 실습 0 = lab00 환경 준비).
3. **통계 프로그램 출력 상자**: R 출력 대신 **파이썬 출력 형식**으로 보여 주세요 — scipy 결과(`TtestResult(statistic=…, pvalue=…, df=…)` 등), statsmodels `summary()` (OLS/Logit/GLM/MixedLM/GEE), lifelines `print_summary()` (KaplanMeierFitter/CoxPHFitter), `logrank_test` 결과. 제목은 "파이썬 출력에서는 이렇게 보입니다". 숫자는 스크립트 계산값과 정확히 일치해야 하며, 출력 모양은 버전에 따라 조금 다를 수 있다는 점을 짧게 밝히세요. SAS 출력과의 비교는 꼭 필요할 때만.
4. 본문 문장에 "R에서는 …" 같은 R 전용 설명이 있으면 파이썬 기준(scipy/statsmodels/lifelines 함수)으로 바꾸세요. 함수·인자 이름은 확실한 것만 쓰세요.
