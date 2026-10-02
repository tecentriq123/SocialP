"""One-time remap (2026-10-02): old chapter 14 was split into chapters 15–19 and 부록 A; references → 부록 B.
Rewrites text references ("14장 다 절") and links (#ch14-s3) in content/, realpapers/, gen/, widgets/."""
import re, glob, sys
TEXT = {  # old section letter → new place
    "가": "부록 A 가 절", "나": "부록 A 나 절", "다": "15장", "라": "17장", "마": "16장", "바": "19장",
    "사": "18장", "아": "부록 A 다 절", "자": "부록 A 라 절", "차": "부록 A 마 절", "카": "부록 A 바 절",
}
HREF = {"1": "ap01-s1", "2": "ap01-s2", "3": "ch15", "4": "ch17", "5": "ch16", "6": "ch19", "7": "ch18",
        "8": "ap01-s3", "9": "ap01-s4", "10": "ap01-s5", "11": "ap01-s6"}
files = [f for pat in ("content/*.html", "realpapers/*.html", "gen/*.py", "widgets/*.js") for f in glob.glob(pat)]
apply = "--apply" in sys.argv
tot = 0; left = []
for f in sorted(files):
    s = open(f, encoding="utf-8").read(); o = s
    s = re.sub(r'#ch14-s(\d+)(?!\d)', lambda m: "#" + HREF[m.group(1)], s)
    s = re.sub(r'14장 ([가-카]) 절', lambda m: TEXT[m.group(1)], s)
    s = s.replace('href="#ch15-ref-', 'href="#ap02-ref-')
    if s != o:
        tot += 1
        if apply: open(f, "w", encoding="utf-8").write(s)
    for m in re.finditer(r'.{30}(?:14장|15장|ch14|ch15|[Pp][Aa][Rr][Tt] ?3).{30}', s):
        left.append((f, m.group(0).replace("\n", " ")))
print("files changed:", tot)
for f, t in left: print(f, "|", t)
