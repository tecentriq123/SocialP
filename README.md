# 사회약학 연구방법 노트

연구실 신입 대학원생을 위한 보건통계·연구방법 학습 사이트입니다.

- `index.html`: 홈(과목 목록), `stats.html`: '보건통계학 기초' 과목, `ml.html`: '머신러닝 기초' 과목. 모두 빌드 결과물입니다.
- `data/`: 실습 14 이후에 쓰는 가상 자료(CSV). 실습 코드가 사이트 주소에서 직접 읽습니다. 실제 환자 자료가 아닙니다.
- `notebooks/`: 실습 14 이후의 주피터 노트북(.ipynb)
- `src/`: 사이트 원본
  - `content/` 장별 본문, `figs/` 그림, `gen/` 숫자·그림을 계산하는 파이썬 스크립트
  - `refs.py`, `refs_add/` 참고문헌, `realpapers/` '실제 논문으로 읽어 보기' 안내
  - `shell.html` 화면 틀(CSS·JS), `build.py` 빌드 스크립트
  - `pub/` 사이트와 함께 내보내는 파일(`pub/data` → `data/`, `pub/notebooks` → `notebooks/`)
  - `STYLE.md` 용어·문체 규칙, `plan/` 개편 계획

빌드: `src`에서 `python3 build.py`를 실행하면 `src/dist/`에 홈과 두 과목 페이지가 만들어집니다(`--course stats|ml`로 한 과목만).
