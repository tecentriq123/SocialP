# Claude Opus 5.5 가독성 피드백 — 아나콘다 실행 방법

## 폴더 구성

| 이름 | 내용 |
|---|---|
| chapters/ | 1차 수정을 반영한 28개 장 md (번호 시작값·번호 표식 보존) |
| prompt_system.txt | 공통 지시 (독자, 교과서 기준, 바꾸면 안 되는 것) |
| task_chapter.txt | 장 전체 요청: 총평·흐름·반복 버릇·잘 읽히는 곳 |
| task_unit.txt | 검토 단위 요청: 문장 수정표·문단 다시 쓰기 |
| chunker.py | 장을 절 단위(최대 3만 자)로 나누는 코드 |
| claude_feedback.py | 실행 스크립트 |

장마다 '장 전체 요청' 1건과 '검토 단위 요청' 여러 건을 보냅니다(28개 장, 모두 180건). 단위 요청에도 장 전체를 함께 보내 앞뒤 맥락을 보게 하고, 장 본문은 캐시에 올려 두 번째 요청부터는 싸게 읽습니다.

## 1. 아나콘다 환경 만들기 (처음 한 번)

1. 압축을 한글이 없는 경로에 풉니다. 예: `C:\work\claude_api`
2. 시작 메뉴에서 **Anaconda Prompt**를 엽니다.
3. 아래를 한 줄씩 입력합니다.

```
conda create -n claude python=3.12 -y
conda activate claude
pip install -U anthropic
```

`anthropic`은 2.4 이상이어야 합니다(`pip show anthropic`으로 확인).

## 2. API 키 넣기 (처음 한 번)

1. https://console.anthropic.com → **API Keys** → **Create Key**에서 키를 만들고 복사합니다. 키는 채팅에 붙여 넣지 마세요.
2. Anaconda Prompt에서:

```
setx ANTHROPIC_API_KEY "복사한_키"
```

3. **Anaconda Prompt를 닫고 새로 연 뒤** 다시 환경을 켭니다.

```
conda activate claude
python -c "import os; print(bool(os.environ.get('ANTHROPIC_API_KEY')))"
```

`True`가 나오면 됩니다.

## 3. 실행

```
cd /d C:\work\claude_api
```

**① 비용 확인 (무료)** — 실제 요청 없이 토큰 수만 셉니다.

```
python claude_feedback.py --count
```

마지막 줄의 '배치 예상 비용' 두 값(캐시가 안 맞을 때 / 잘 맞을 때) 사이에서 실제 비용이 나옵니다.

**② (선택) 한 장만 시험** — 0장(검토 단위 6개, 요청 7건)을 바로 돌립니다. 일반 요청 가격이라 3–5달러쯤 듭니다. 결과 모양을 먼저 보고 싶을 때만 쓰세요.

```
python claude_feedback.py --test 01_ch00
```

**③ 전체 배치 제출**

```
python claude_feedback.py --batch
```

배치 ID가 나오면 창을 닫아도 됩니다. 보통 1시간 안에, 길면 24시간 안에 끝납니다. 이미 받은 요청(raw 폴더에 있는 것)은 다시 보내지 않습니다.

**④ 결과 확인**

```
python claude_feedback.py --batch-check
```

- 끝났으면 `raw/`에 요청별 결과, `feedback/`에 장별 피드백 28개가 생깁니다.
- 실제 비용은 화면과 `usage_log.jsonl`에 남습니다.
- 오류나 만료가 있으면 `--batch`를 한 번 더 실행하세요. 빠진 요청만 다시 보냅니다.

**⑤ feedback 폴더의 md 28개를 Claude 대화에 올려 주세요.** 1차 때처럼 원문 대조와 판정을 하겠습니다.

## 옵션

| 옵션 | 뜻 |
|---|---|
| `--effort xhigh` | 추론을 더 깊게 (기본 high). 출력 토큰과 비용이 늘어납니다 |
| `--only 05_ch04,10_ch08` | 일부 장만 |
| `--merge` | raw 결과를 feedback 파일로 다시 합치기 (API 키 불필요) |

## 2차 이후 다시 돌릴 때

`chapters/`를 새 md로 바꾸고, `raw/`, `feedback/`, `batch_ids.txt`, `usage_log.jsonl`을 지운 뒤 ①부터 다시 합니다.

## Jupyter에서 돌리고 싶다면

Anaconda Navigator에서 `claude` 환경의 Jupyter를 열고, 셀에 아래처럼 씁니다(키는 2단계에서 넣은 값을 씁니다).

```
%cd C:\work\claude_api
!python claude_feedback.py --count
```

## Claude Code 클라우드 세션에서 돌릴 때

1. GitHub에 비공개 저장소(예: `socialp-feedback`)를 만들고, 이 폴더의 내용을 그대로 올립니다(**Add file → Upload files**).
2. https://claude.ai/code 에서 그 저장소를 고르고, 환경 선택 칸에서 **New environment**를 만듭니다.
   - Network access: **Trusted** (기본값, api.anthropic.com이 포함됨)
   - Environment variables: `FEEDBACK_API_KEY=발급받은_키`
     (`ANTHROPIC_API_KEY`라는 이름은 쓰지 않습니다. Claude Code 자체 로그인과 섞이지 않게 하려는 것입니다. 네트워크 시크릿은 api.anthropic.com에 적용되지 않으므로 환경 변수로 넣습니다.)
3. 세션에 이렇게 요청합니다.
   - "README대로 anthropic을 설치하고 `python claude_feedback.py --count`를 돌려서 결과를 알려줘"
   - (비용 확인 후) "`--batch`로 제출해줘"
   - (한두 시간 뒤) "`--batch-check`를 돌리고, 끝났으면 raw·feedback 폴더와 usage_log.jsonl을 커밋해서 푸시해줘"
