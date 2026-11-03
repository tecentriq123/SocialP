"""
보건통계학 기초 — Claude Opus 5.5 가독성 피드백 (배치)

사용법 (README.md 참고)
  python claude_feedback.py --count            토큰 수와 예상 비용 (무료, 실제 요청 없음)
  python claude_feedback.py --test 01_ch00     한 장만 바로 돌려 보기 (일반 요청 가격)
  python claude_feedback.py --batch            남은 요청을 배치로 제출
  python claude_feedback.py --batch-check      배치 상태 확인, 끝났으면 결과 저장·합치기
  python claude_feedback.py --merge            저장된 결과를 장별 피드백 파일로 다시 합치기
옵션
  --effort high|xhigh|max   추론 수준 (기본 high)
  --only 05_ch04,10_ch08    일부 장만
"""
import os, re, sys, json, glob, time, argparse, concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chunker import mark_chapter

import anthropic

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
UNIT_LIMIT = 30000          # 검토 단위 하나의 최대 글자 수
PRICE = dict(inp=4.0, out=20.0, cache_w1h=8.0, cache_r=0.20)   # USD / 100만 토큰 (일반 요청)

HERE = os.path.dirname(os.path.abspath(__file__))
CH_DIR = os.path.join(HERE, "chapters")
RAW_DIR = os.path.join(HERE, "raw")
FB_DIR = os.path.join(HERE, "feedback")
BATCH_FILE = os.path.join(HERE, "batch_ids.txt")
for d in (RAW_DIR, FB_DIR):
    os.makedirs(d, exist_ok=True)

def read(name):
    return open(os.path.join(HERE, name), encoding="utf-8").read()

SYSTEM = read("prompt_system.txt")
TASK_CH = read("task_chapter.txt")
TASK_UNIT = read("task_unit.txt")


def chapters(only=None):
    fs = sorted(glob.glob(os.path.join(CH_DIR, "*.md")))
    if only:
        keep = set(only.split(","))
        fs = [f for f in fs if os.path.basename(f)[:-3] in keep]
    out = []
    for f in fs:
        stem = os.path.basename(f)[:-3]
        text = open(f, encoding="utf-8").read()
        title = text.split("\n", 1)[0].lstrip("# ").strip()
        marked, units = mark_chapter(text, UNIT_LIMIT)
        out.append(dict(stem=stem, title=title, marked=marked, units=units))
    return out


def params(ch, unit=None, effort="high"):
    doc = f"[첨부 파일: {ch['stem']}.md]\n\n{ch['marked']}"
    if unit is None:
        task = TASK_CH.replace("{TITLE}", ch["title"])
    else:
        k, utitle, _ = unit
        task = (TASK_UNIT.replace("{TITLE}", ch["title"])
                .replace("{UNIT_TITLE}", f"'{utitle}'").replace("{UNIT}", str(k)))
    return dict(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        output_config={"effort": effort},
        # 시스템 프롬프트와 장 본문을 캐시에 올려, 같은 장의 다른 요청은 싸게 읽게 함
        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        messages=[{"role": "user", "content": [
            {"type": "text", "text": doc, "cache_control": {"type": "ephemeral", "ttl": "1h"}},
            {"type": "text", "text": task},
        ]}],
    )


def all_requests(chs, effort):
    reqs = []
    for ch in chs:
        reqs.append((f"{ch['stem']}__all", params(ch, None, effort)))
        for u in ch["units"]:
            reqs.append((f"{ch['stem']}__u{u[0]:02d}", params(ch, u, effort)))
    return reqs


def done(cid):
    return os.path.exists(os.path.join(RAW_DIR, cid + ".md"))


def text_of(msg_content):
    return "\n".join(b.text for b in msg_content if getattr(b, "type", "") == "text").strip()


def cost(u, batch):
    f = 0.5 if batch else 1.0
    w = getattr(u, "cache_creation_input_tokens", 0) or 0
    r = getattr(u, "cache_read_input_tokens", 0) or 0
    return f * (u.input_tokens * PRICE["inp"] + w * PRICE["cache_w1h"] + r * PRICE["cache_r"]
                + u.output_tokens * PRICE["out"]) / 1e6


def save(cid, text, usage, stop, batch):
    open(os.path.join(RAW_DIR, cid + ".md"), "w", encoding="utf-8").write(text)
    log = dict(cid=cid, stop=stop, input=usage.input_tokens,
               cache_write=getattr(usage, "cache_creation_input_tokens", 0) or 0,
               cache_read=getattr(usage, "cache_read_input_tokens", 0) or 0,
               output=usage.output_tokens, usd=round(cost(usage, batch), 4))
    with open(os.path.join(HERE, "usage_log.jsonl"), "a", encoding="utf-8") as fp:
        fp.write(json.dumps(log, ensure_ascii=False) + "\n")
    if stop == "max_tokens":
        print(f"  ! {cid}: 출력이 max_tokens에서 잘렸습니다")


# ---------------------------------------------------------------- 실행 모드
def do_count(client, chs, effort):
    tot_ch = tot_req = 0
    n_req = 0
    print(f"{'장':<10}{'단위':>5}{'장 토큰':>10}")
    for ch in chs:
        p = params(ch, None, effort)
        t = client.messages.count_tokens(model=MODEL, system=p["system"], messages=p["messages"]).input_tokens
        k = len(ch["units"])
        print(f"{ch['stem']:<10}{k:>5}{t:>10,}")
        tot_ch += t
        tot_req += t * (k + 1)
        n_req += k + 1
    # 출력: 장 요청 1만, 단위 요청 2만 토큰(추론 포함)으로 가정
    out = 10000 * len(chs) + 20000 * (n_req - len(chs))
    no_cache = 0.5 * (tot_req * PRICE["inp"] + out * PRICE["out"]) / 1e6
    good_cache = 0.5 * (tot_ch * PRICE["cache_w1h"] + (tot_req - tot_ch) * PRICE["cache_r"] + out * PRICE["out"]) / 1e6
    print(f"\n요청 {n_req}건 · 장 본문 {tot_ch:,} 토큰 · 요청 입력 합계 {tot_req:,} 토큰 · 예상 출력 {out:,} 토큰")
    print(f"배치 예상 비용: 캐시가 거의 안 맞으면 ${no_cache:.1f}, 잘 맞으면 ${good_cache:.1f}")
    print("(출력 토큰은 추정치입니다. 실제 비용은 --batch-check 뒤 usage_log.jsonl에 기록됩니다.)")


def do_test(client, chs, effort):
    reqs = [(cid, p) for cid, p in all_requests(chs, effort) if not done(cid)]
    print(f"요청 {len(reqs)}건을 일반 요청으로 보냅니다 (동시 3건)")

    def one(item):
        cid, p = item
        with client.messages.stream(**p) as s:
            m = s.get_final_message()
        save(cid, text_of(m.content), m.usage, m.stop_reason, batch=False)
        return cid, cost(m.usage, False)

    total = 0
    with cf.ThreadPoolExecutor(3) as ex:
        for cid, c in ex.map(one, reqs):
            total += c
            print(f"  {cid} 완료 (${c:.2f})")
    print(f"합계 ${total:.2f}")
    do_merge(chs)


def do_batch(client, chs, effort):
    reqs = [(cid, p) for cid, p in all_requests(chs, effort) if not done(cid)]
    if not reqs:
        print("남은 요청이 없습니다.")
        return
    b = client.messages.batches.create(requests=[{"custom_id": cid, "params": p} for cid, p in reqs])
    with open(BATCH_FILE, "a", encoding="utf-8") as fp:
        fp.write(b.id + "\n")
    print(f"배치 제출: {b.id} · 요청 {len(reqs)}건")
    print("보통 1시간 안에, 길면 24시간 안에 끝납니다. 나중에 --batch-check로 확인하세요.")


def do_check(client, chs):
    if not os.path.exists(BATCH_FILE):
        print("제출한 배치가 없습니다.")
        return
    ids = [l.strip() for l in open(BATCH_FILE, encoding="utf-8") if l.strip()]
    all_done = True
    for bid in ids:
        b = client.messages.batches.retrieve(bid)
        c = b.request_counts
        print(f"{bid}: {b.processing_status} · 처리 중 {c.processing}, 성공 {c.succeeded}, 오류 {c.errored}, 만료 {c.expired}")
        if b.processing_status != "ended":
            all_done = False
            continue
        total, bad = 0.0, []
        for r in client.messages.batches.results(bid):
            if r.result.type == "succeeded":
                m = r.result.message
                if not done(r.custom_id):
                    save(r.custom_id, text_of(m.content), m.usage, m.stop_reason, batch=True)
                total += cost(m.usage, True)
            else:
                bad.append((r.custom_id, r.result.type))
        print(f"  이 배치 비용 약 ${total:.2f}")
        for cid, t in bad:
            print(f"  ! {cid}: {t} → --batch로 다시 제출하면 이 요청만 다시 보냅니다")
    if all_done:
        do_merge(chs)
    else:
        print("아직 끝나지 않은 배치가 있습니다. 잠시 뒤 다시 확인하세요.")


def section(text, num):
    m = re.search(rf"^###\s*{num}\.[^\n]*\n(.*?)(?=^###\s*\d+\.|\Z)", text, re.S | re.M)
    return m.group(1).strip() if m else ""


def do_merge(chs):
    n = 0
    for ch in chs:
        allp = os.path.join(RAW_DIR, f"{ch['stem']}__all.md")
        units = [os.path.join(RAW_DIR, f"{ch['stem']}__u{u[0]:02d}.md") for u in ch["units"]]
        if not os.path.exists(allp) or not all(os.path.exists(p) for p in units):
            continue
        a = open(allp, encoding="utf-8").read()
        rows, rewrites, notes = [], [], []
        header = "| 위치(절·소제목) | 원문 | 문제 | 수정안 | 의미 변경 |\n|---|---|---|---|---|"
        for u, p in zip(ch["units"], units):
            t = open(p, encoding="utf-8").read()
            tbl = section(t, 3)
            for line in tbl.split("\n"):
                s = line.strip()
                if s.startswith("|"):
                    cells = [x.strip() for x in s.strip("|").split("|")]
                    is_sep = re.fullmatch(r"[\s:|-]+", s) is not None
                    is_head = len(cells) > 1 and cells[0].startswith("위치") and cells[1] == "원문"
                    if not is_sep and not is_head:
                        rows.append(s)
                elif s.startswith("같은 버릇"):
                    notes.append(f"- 단위 {u[0]} ({u[1]}): {s}")
            rw = section(t, 4)
            if rw and rw.strip() not in ("없음", "없음."):
                rewrites.append(f"#### 단위 {u[0]} · {u[1]}\n\n{rw}")
        out = [f"# Claude 가독성 피드백 — {ch['title']}\n",
               f"> 모델: {MODEL} (배치) · 검토 단위 {len(ch['units'])}개\n",
               "### 1. 총평\n\n" + section(a, 1),
               "### 2. 흐름 문제\n\n" + section(a, 2),
               "### 3. 문장 수정표\n\n" + header + "\n" + "\n".join(rows) + ("\n\n" + "\n".join(notes) if notes else ""),
               "### 4. 문단 다시 쓰기\n\n" + ("\n\n".join(rewrites) or "없음"),
               "### 5. 반복되는 버릇\n\n" + section(a, 5),
               "### 6. 잘 읽히는 곳\n\n" + section(a, 6)]
        open(os.path.join(FB_DIR, ch["stem"] + ".md"), "w", encoding="utf-8").write("\n\n".join(out) + "\n")
        n += 1
    print(f"feedback 폴더에 {n}개 장 피드백을 만들었습니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--count", action="store_true")
    g.add_argument("--test", metavar="장", help="예: 01_ch00")
    g.add_argument("--batch", action="store_true")
    g.add_argument("--batch-check", action="store_true")
    g.add_argument("--merge", action="store_true")
    ap.add_argument("--effort", default="high", choices=["medium", "high", "xhigh", "max"])
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    chs = chapters(a.test or a.only)
    if not chs:
        sys.exit("chapters 폴더에서 장을 찾지 못했습니다.")
    if a.merge:
        do_merge(chs)
        sys.exit()
    # 클라우드 세션에서는 Claude Code 자체 로그인과 섞이지 않게 FEEDBACK_API_KEY를 씀
    key = os.environ.get("FEEDBACK_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("API 키 환경변수(FEEDBACK_API_KEY 또는 ANTHROPIC_API_KEY)가 없습니다. README를 확인하세요.")
    client = anthropic.Anthropic(api_key=key, max_retries=5, timeout=3600)
    if a.count:
        do_count(client, chs, a.effort)
    elif a.test:
        do_test(client, chs, a.effort)
    elif a.batch:
        do_batch(client, chs, a.effort)
    else:
        do_check(client, chs)
