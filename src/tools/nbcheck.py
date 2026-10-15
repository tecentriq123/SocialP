"""실습 노트북을 하나씩 새 프로세스에서 처음부터 돌려, 사이트에 실린 출력과 같은지 확인한다 (2026-10-12).

run:  source /home/claude/pylibs/env.sh && python3 tools/nbcheck.py lab09 [lab15 ...]
      python3 tools/nbcheck.py --all        (checks/*.expected.json 이 있는 노트북 모두)

- 노트북(pub/notebooks/<lab>.ipynb)의 코드 셀만, 위에서부터 차례로, 노트북마다 따로 띄운 파이썬 프로세스에서 실행한다.
  그래서 앞 실습에서 만든 변수가 뒤 실습의 오류를 가리지 않는다.
- 사이트 생성기(gen/lab_*.py, labkit)가 실제로 돌린 출력(checks/<lab>.expected.json)과 셀마다 비교한다.
  다르면 처음 달라진 줄을 보여 준다. 숫자가 다르면 핵심 결과값이 달라진 것이다.
- !pip, %matplotlib 같은 셸·매직 줄은 실행하지 않는다(Colab에서만 의미가 있다).
- 이 작업 환경은 인터넷이 막혀 있으므로 labkit을 불러 자료 주소를 사이트 폴더의 사본으로 돌린다
  (labkit을 import만 할 뿐 실습 셀은 하나도 미리 실행하지 않는다).
"""
import glob, json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNNER = r'''
import ast, json, sys, io, contextlib, warnings
sys.path.insert(0, {gen!r})
import labkit  # noqa: F401  (자료 주소를 오프라인 사본으로 돌리는 것만 씀)
import matplotlib.pyplot as plt
nb = json.load(open({nb!r}, encoding="utf-8"))
ns = {{"__name__": "__main__"}}
outs = []
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    code = "".join(cell["source"])
    code = "\n".join("pass" if ln.lstrip().startswith(("!", "%")) else ln for ln in code.split("\n"))
    tree = ast.parse(code)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    buf = io.StringIO()
    err = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, "<nb>", "exec"), ns)
                if last is not None:
                    v = eval(compile(last, "<nb>", "eval"), ns)
                    if v is not None:
                        t = buf.getvalue()
                        print(("" if not t or t.endswith("\n") else "\n") + repr(v), end="")
        except Exception as e:
            err = f"{{type(e).__name__}}: {{e}}"
    plt.close("all")
    outs.append({{"out": buf.getvalue(), "error": err}})
    if err:
        break
json.dump(outs, sys.stdout, ensure_ascii=False)
'''


_STAMP = re.compile(r"\b\d{2}:\d{2}:\d{2}\b|\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), \d{2} \w{3} \d{4}\b"
                    r"|(?<=time fit was run = )\d{4}-\d{2}-\d{2}\b")


def norm(t):
    """statsmodels summary와 lifelines print_summary의 날짜·시각은 실행할 때마다 달라지므로 가린다."""
    return [_STAMP.sub("#", ln).rstrip() for ln in t.strip("\n").split("\n")] if t.strip() else []


def check(lab, timeout=900):
    nbp = os.path.join(ROOT, "pub", "notebooks", f"{lab}.ipynb")
    exp_p = os.path.join(ROOT, "checks", f"{lab}.expected.json")
    if not os.path.exists(nbp):
        return False, f"{lab}: 노트북 없음 ({nbp})"
    if not os.path.exists(exp_p):
        return False, f"{lab}: 기대 출력 없음 ({exp_p}) — 생성기를 다시 돌리세요"
    exp = json.load(open(exp_p, encoding="utf-8"))
    code = RUNNER.format(gen=os.path.join(ROOT, "gen"), nb=nbp)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    if r.returncode != 0:
        return False, f"{lab}: 실행기 오류\n{r.stderr[-2000:]}"
    got = json.loads(r.stdout)
    problems = []
    for i, e in enumerate(exp):
        if i >= len(got):
            problems.append(f"  셀 {e['n']} ({e['title']}): 실행되지 않음(앞 셀에서 멈춤)")
            break
        g = got[i]
        if g["error"]:
            problems.append(f"  셀 {e['n']} ({e['title']}): 오류 — {g['error']}")
            break
        a, b = norm(e["out"]), norm(g["out"])
        if a != b:
            for k in range(max(len(a), len(b))):
                x = a[k] if k < len(a) else "(없음)"
                y = b[k] if k < len(b) else "(없음)"
                if x != y:
                    problems.append(f"  셀 {e['n']} ({e['title']}) {k + 1}번째 줄\n    사이트: {x}\n    실행:   {y}")
                    break
    if problems:
        return False, f"{lab}: 다름 {len(problems)}곳\n" + "\n".join(problems)
    return True, f"{lab}: 셀 {len(exp)}개 모두 같음"


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--all"]:
        args = sorted(os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(ROOT, "checks", "*.expected.json")))
    ok_all = True
    for lab in args:
        ok, msg = check(lab)
        ok_all &= ok
        print(("OK   " if ok else "FAIL ") + msg)
    sys.exit(0 if ok_all else 1)
