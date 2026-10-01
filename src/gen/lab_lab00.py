"""실습 0 · 실습 환경 준비 — run with: source /home/claude/pylibs/env.sh && python3 gen/lab_lab00.py

Every executed cell runs for real through labkit (outputs are never typed by hand).
Code that cannot run in this offline sandbox (Colab-only helpers, Windows paths, Anaconda Prompt
commands) is shown with `static()` as a code-only block with its own label, not as a numbered cell.
"""
import html, os, re, sys, tempfile, traceback, linecache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labkit import Notebook, Cell  # noqa: E402

nb = Notebook("lab00")
# files the lab writes (my_patients.csv/.xlsx) go to a throw-away folder, like Colab's /content
os.chdir(tempfile.mkdtemp(prefix="lab00_"))


def mark_out(h, marks):
    """Put <span class="mk">n</span> after the first occurrence of each text in the cell OUTPUT.
    (labkit's own `marks` requires every mark to occur in both the printed text and the table;
    this version searches the whole output area, text nodes only, so tags/attributes are never touched.)"""
    if not marks:
        return h
    i0 = h.find('<div class="cell-out">')
    head, body = h[:i0], h[i0:]
    for sub, n in marks.items():
        parts = re.split(r"(<[^>]+>)", body)
        done = False
        for k, p in enumerate(parts):
            if p.startswith("<"):
                continue
            for e in (html.escape(sub), html.escape(sub, quote=False)):
                j = p.find(e)
                if j >= 0:
                    parts[k] = p[:j + len(e)] + f'<span class="mk">{n}</span>' + p[j + len(e):]
                    done = True
                    break
            if done:
                break
        if not done:
            raise ValueError(f"mark text not found in output: {sub!r}")
        body = "".join(parts)
    return head + body


def save(name, cell, marks=None, title=None):
    nb.save_fragment(name, mark_out(nb.html(cell, title=title), marks))


def static(name, code, label, title, lang="python"):
    """Code-only block (not executed, no output) with the same look as a cell."""
    code = code.strip("\n")
    h = (f'<div class="cell"><div class="cell-h"><span class="cell-n">{html.escape(label)}</span>'
         f'<span class="cell-t">{html.escape(title)}</span>'
         f'<button class="copy" type="button" aria-label="코드 복사">복사</button></div>'
         f'<pre class="cell-in"><code class="language-{lang}">{html.escape(code)}</code></pre></div>')
    nb.save_fragment(name, h)


def syntax_error_cell(code, title):
    """labkit parses a cell before running it, so a SyntaxError cannot use expect_error.
    Compile the code for real and attach the real message to a numbered cell."""
    code = code.strip("\n")
    c = Cell(len(nb.cells) + 1, code, title)
    nb.cells.append(c)
    try:
        compile(code, f"<lab00-cell{c.n}>", "exec")
        raise RuntimeError("expected a SyntaxError")
    except SyntaxError as e:
        c.error = "".join(traceback.format_exception_only(type(e), e)).strip().splitlines()[-1]
    return c


# ---------------------------------------------------------------- 가. 파이썬과 주피터 노트북
c = nb.cell('''
# 첫 번째 셀: 글자 출력과 계산
print("안녕하세요, 파이썬!")
bwt_g = 2523            # 출생체중(g)을 변수 bwt_g에 저장
bwt_kg = bwt_g / 1000   # g을 kg으로 바꾸어 새 변수에 저장
bwt_kg                  # 마지막 줄의 값은 셀 아래에 표시됨
''', title="첫 번째 셀")
save("lab00_first", c, marks={"안녕하세요, 파이썬!": 1, "2.523": 2})

# ---------------------------------------------------------------- 나. Google Colab
static("lab00_colab_upload", '''
from google.colab import files
uploaded = files.upload()   # 셀 아래에 파일 선택 버튼이 나타남
''', "Colab 전용", "코드로 파일 올리기 (이 사이트에서는 실행하지 않음)")

static("lab00_colab_drive", '''
from google.colab import drive
drive.mount("/content/drive")   # 구글 계정 접근 허용 창이 뜸

import pandas as pd
path = "/content/drive/MyDrive/lab/my_patients.csv"
df_my = pd.read_csv(path, encoding="cp949")
''', "Colab 전용", "Google Drive의 파일 읽기 (이 사이트에서는 실행하지 않음)")

# ---------------------------------------------------------------- 다. Windows 11 설치
static("lab00_prompt_start", r'''
cd C:\Users\hong\Documents\stats_lab
jupyter lab
''', "Anaconda Prompt", "작업 폴더로 이동한 뒤 JupyterLab 실행", lang="plaintext")

static("lab00_prompt_pip", '''
pip install lifelines
''', "Anaconda Prompt", "패키지 설치 (명령 창에서)", lang="plaintext")

static("lab00_nb_pip", '''
%pip install lifelines
''', "노트북 셀", "패키지 설치 (노트북 셀에서)")

static("lab00_prompt_ver", '''
python --version
conda list pandas
pip show lifelines
''', "Anaconda Prompt", "설치된 버전 확인", lang="plaintext")

# ---------------------------------------------------------------- 라. 셀 실행과 패키지 설치
c = nb.cell('''
!pip install lifelines
''', title="패키지 설치 (Colab)", shell_output="""Collecting lifelines
  Downloading lifelines-0.30.3-py3-none-any.whl.metadata
...
Installing collected packages: ..., lifelines
Successfully installed ... lifelines-0.30.3""")
save("lab00_pip", c, marks={"Collecting lifelines": 1, "Successfully installed": 2})

c = nb.cell('''
import pandas as pd   # pandas를 pd라는 짧은 이름으로 불러오기
import numpy as np    # numpy는 np로
''', title="패키지 불러오기")
save("lab00_import", c)

c = nb.cell('''
import sys
import scipy, statsmodels, sklearn, lifelines

print("Python      ", sys.version.split()[0])
print("pandas      ", pd.__version__)
print("numpy       ", np.__version__)
print("scipy       ", scipy.__version__)
print("statsmodels ", statsmodels.__version__)
print("scikit-learn", sklearn.__version__)
print("lifelines   ", lifelines.__version__)
''', title="버전 확인")
save("lab00_versions", c, marks={"3.11.15": 1, "3.0.2": 2, "0.15.0": 3})
VERS = c.stdout

# (removed: a "pip install without %/!" SyntaxError cell. In Jupyter/Colab, IPython's automagic
#  runs a bare `pip install x` as %pip, so a SyntaxError there would be misleading.)

c = nb.cell('''
# 산모 체중을 kg으로 바꾸기 (lwt_lb를 만드는 셀은 아직 실행 전)
lwt_kg = lwt_lb * 0.4536
''', title="순서를 건너뛰고 실행하면", expect_error=True)
save("lab00_order1", c)

c = nb.cell('''
lwt_lb = 182   # 산모 체중(파운드): birthwt 자료의 첫 번째 산모
''', title="빠뜨린 셀 실행")
save("lab00_order2", c)

c = nb.cell('''
lwt_kg = lwt_lb * 0.4536   # 1 lb = 0.4536 kg
round(lwt_kg, 1)
''', title="다시 실행")
save("lab00_order3", c)

c = nb.cell('''
bwt = pd.Series([2523, 2551, 2557])   # 출생체중(g) 3명
''', title="값 만들기")
save("lab00_twice1", c)

TWICE = '''
bwt = bwt / 1000   # g → kg (같은 이름에 덮어쓰기)
print(bwt)
'''
c = nb.cell(TWICE, title="단위 바꾸기")
save("lab00_twice2", c, marks={"2.523": 1})
c = nb.cell(TWICE, title="같은 셀을 한 번 더 실행하면")
save("lab00_twice3", c, marks={"0.002523": 2})

# ---------------------------------------------------------------- 마. 데이터 불러오기
c = nb.cell('''
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer(as_frame=True)   # 자료 꾸러미 받기
print(bc.frame.shape)        # (행 수, 열 수)
print(bc.target_names)       # target 0과 1의 이름
print(bc.feature_names[:3])  # 설명변수 이름 앞의 3개
''', title="scikit-learn 내장 데이터")
save("lab00_sk1", c, marks={"(569, 31)": 1, "['malignant' 'benign']": 2})

c = nb.cell('''
bc.frame[["mean radius", "mean area", "target"]].head()
''', title="앞의 5행 보기")
save("lab00_sk2", c, marks={"target": 1})

c = nb.cell('''
base = "https://vincentarelbundock.github.io/Rdatasets/csv/"
url = base + "MASS/birthwt.csv"   # <패키지>/<데이터 이름>.csv
df = pd.read_csv(url)             # 인터넷의 CSV를 표로 읽기
print(df.shape)
df.head()
''', title="Rdatasets에서 CSV 읽기")
save("lab00_rd1", c, marks={"(189, 11)": 1, "rownames": 2})

c = nb.cell('''
df = df.drop(columns="rownames")   # R의 행 이름 열 지우기
print(df.shape)
print(df.columns.tolist())         # 남은 열 이름 목록
''', title="필요 없는 열 지우기")
save("lab00_rd2", c, marks={"(189, 10)": 1})

c = nb.cell('''
# 연습용 파일 만들기 (엑셀에서 저장한 한글 CSV를 흉내)
demo = pd.DataFrame({"환자번호": [1, 2, 3],
                     "성별": ["남", "여", "여"],
                     "HbA1c": [7.2, 6.8, 8.1]})
demo.to_csv("my_patients.csv", index=False, encoding="cp949")
demo.to_excel("my_patients.xlsx", index=False)
''', title="연습용 파일 만들기")
save("lab00_own1", c)

c = nb.cell('''
pd.read_csv("my_patients.csv")   # encoding을 안 쓰면 utf-8로 읽음
''', title="한글 CSV를 그냥 읽으면", expect_error=True)
save("lab00_own2", c)

c = nb.cell('''
my = pd.read_csv("my_patients.csv", encoding="cp949")
my
''', title="인코딩을 지정해 읽기")
save("lab00_own3", c)

c = nb.cell('''
my_x = pd.read_excel("my_patients.xlsx")   # 첫 번째 시트를 읽음
my_x
''', title="엑셀 파일 읽기")
save("lab00_own4", c)

static("lab00_winpath", r'''
path = r"C:\Users\hong\Documents\lab\my_patients.csv"
my = pd.read_csv(path, encoding="cp949")
''', "Windows 예시", "내 컴퓨터의 파일 읽기 (이 사이트에서는 실행하지 않음)")

c = syntax_error_cell(r'''
path = "C:\Users\hong\Documents\lab\my_patients.csv"
''', title="경로 앞에 r을 빼먹으면")
save("lab00_winpath_err", c)

# ---------------------------------------------------------------- 바. pandas로 데이터 살펴보기
c = nb.cell('''
df.head()   # 앞의 5행 (괄호 안에 숫자를 넣으면 그만큼)
''', title="표의 구조")
save("lab00_pd_head", c, marks={"bwt": 1, "2523": 2})

c = nb.cell('''
df.info()   # 열마다 결측 아닌 값의 수와 자료형
''', title="info()")
save("lab00_pd_info", c, marks={"189 entries": 1, "189 non-null": 2, "int64": 3})

c = nb.cell('''
df.describe().round(2)   # 숫자 열의 요약통계 (소수 둘째 자리)
''', title="describe()")
save("lab00_pd_desc", c, marks={"2944.59": 1, "729.21": 2, "2977.00": 3, "1.85": 4})

c = nb.cell('''
print(type(df["bwt"]))    # 열 하나를 고르면 Series
print(df["bwt"].mean())   # 평균
print(df["bwt"].median()) # 중앙값
df[["age", "bwt"]].head(3)  # 대괄호 두 겹: 여러 열 → DataFrame
''', title="열 고르기")
save("lab00_pd_col", c, marks={"Series": 1, "2944.5873015873017": 2, "2977.0": 3})

c = nb.cell('''
is_smoker = df["smoke"] == 1   # 행마다 True / False
print(is_smoker.head())
smokers = df[is_smoker]        # True인 행만 남기기
print(smokers.shape)
''', title="조건에 맞는 행 고르기")
save("lab00_pd_filter", c, marks={"dtype: bool": 1, "(74, 10)": 2})

c = nb.cell('''
df["bwt_kg"] = df["bwt"] / 1000          # 새 열: g → kg
race_names = {1: "white", 2: "black", 3: "other"}
df["race_lab"] = df["race"].map(race_names)   # 숫자 코드 → 이름
df[["bwt", "bwt_kg", "race", "race_lab"]].head()
''', title="새 열 만들기")
save("lab00_pd_newcol", c, marks={"bwt_kg": 1, "race_lab": 2})

c = nb.cell('''
print(df["race_lab"].value_counts())                  # 범주별 빈도
print(df["smoke"].value_counts(normalize=True).round(3))  # 비율
''', title="value_counts()")
save("lab00_pd_vc", c, marks={"96": 1, "0.392": 2})

c = nb.cell('''
df.groupby("smoke")["bwt"].agg(["count", "mean", "std"]).round(1)
''', title="groupby()로 군별 요약")
save("lab00_pd_group", c, marks={"3055.7": 1, "2771.9": 2})

c = nb.cell('''
print(pd.crosstab(df["smoke"], df["low"], margins=True))
pd.crosstab(df["smoke"], df["low"], normalize="index").round(3)
''', title="crosstab()으로 교차표")
save("lab00_pd_ct", c, marks={"All": 1, "0.405": 2})

# ---------------------------------------------------------------- 사. 오류 메시지 읽는 법
TB_CODE = '''
def to_kg(grams):
    return grams / 1000

to_kg("2523")   # 숫자 대신 글자(문자열)를 넣음
'''
c = nb.cell(TB_CODE, title="오류가 나는 셀", expect_error=True)
save("lab00_tb_cell", c)

# the full traceback in Python's standard format (real, produced here)
src = TB_CODE.strip("\n") + "\n"
fname = "<cell>"
linecache.cache[fname] = (len(src), None, src.splitlines(True), fname)
try:
    exec(compile(src, fname, "exec"), {})
except TypeError:
    tb = traceback.format_exc()
lines = tb.rstrip("\n").split("\n")
# drop the frame of this generator script (the exec() call) so only the cell's frames remain
out, skip = [], False
for ln in lines:
    if ln.startswith('  File "') and fname not in ln:
        skip = True
        continue
    if skip and ln.startswith("    "):
        continue
    skip = False
    out.append(ln)
tb_txt = html.escape("\n".join(out))
for sub, n in (("Traceback (most recent call last):", 1), ('line 4, in &lt;module&gt;', 2),
               ("line 2, in to_kg", 3), ("TypeError: unsupported operand type(s) for /: &#x27;str&#x27; and &#x27;int&#x27;", 4)):
    i = tb_txt.find(sub)
    assert i >= 0, sub
    tb_txt = tb_txt[:i + len(sub)] + f'<span class="mk">{n}</span>' + tb_txt[i + len(sub):]
nb.save_fragment("lab00_tb_full", f'<pre class="out">{tb_txt}</pre>')

c = nb.cell('''
df["bwtt"]   # 열 이름 오타 (bwt가 맞음)
''', title="KeyError", expect_error=True)
save("lab00_err_key", c)

c = nb.cell('''
print("bwtt" in df.columns, "bwt" in df.columns)
''', title="열 이름이 있는지 확인")
save("lab00_err_key2", c)

c = nb.cell('''
import pingouin   # 설치하지 않은 패키지
''', title="ModuleNotFoundError", expect_error=True)
save("lab00_err_mod", c)

c = nb.cell('''
pd.read_csv("birthwt_2024.csv")   # 현재 폴더에 없는 파일
''', title="FileNotFoundError", expect_error=True)
save("lab00_err_file", c)

c = nb.cell('''
log_ptl = np.log(df["ptl"])   # ptl: 이전 조산 횟수 (0이 많음)
print(log_ptl.head())
''', title="경고(warning)")
save("lab00_warn", c, marks={"-inf": 1})

print(VERS)
print("cells:", len(nb.cells))
