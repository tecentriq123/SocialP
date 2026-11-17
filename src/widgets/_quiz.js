// 보건통계학 퀴즈 과목 공통 엔진 (build.py가 quiz.html에만 넣는다).
// 문제 마크업: <div class="qz" data-widget="qz" id="…" data-kind="…" data-lv="…" [data-case="1"]>
//   <p class="qz-stem">…</p><ol class="qz-opts"><li data-ok="1">선택지<div class="qz-why-o">선택지 해설</div></li>…</ol>
//   [<div class="qz-step2"><p class="qz-stem">…</p><ol class="qz-opts">…</ol></div>]   (2단계 문항: 고른 것 + 그 이유)
//   <div class="qz-fb">해설(핵심 이유, 조건이 달라지면, 다시 읽을 곳)</div></div>
// 장 첫머리의 <div class="widget" data-widget="qz_player"></div>가 한 문제씩 보여 주는 진행판이다.
// 기록은 이 브라우저의 localStorage에만 둔다: KEY = { 문제 id: { ok, ok2, conf, at, n } }, POS = { 장 id: 마지막으로 본 문제 id }
(function () {
  const KEY = "socialp-quiz-v1", POS = "socialp-quiz-pos";
  const CONF = [["sure", "확실함"], ["unsure", "헷갈림"], ["guess", "추측"]];
  const CONF_LABEL = Object.fromEntries(CONF);
  const get = k => { try { return JSON.parse(localStorage.getItem(k) || "{}") || {}; } catch (e) { return {}; } };
  const put = (k, o) => { try { localStorage.setItem(k, JSON.stringify(o)); } catch (e) {} };
  const load = () => get(KEY), store = o => put(KEY, o);
  const right = r => !!r && r.ok && r.ok2 !== false;
  const review = r => !!r && (!right(r) || r.conf !== "sure");
  const day = iso => { const d = new Date(iso); return isNaN(d) ? "" : `${d.getMonth() + 1}월 ${d.getDate()}일`; };
  const L = "ABCDEFGH";

  if (!document.getElementById("qz-style")) {
    const st = document.createElement("style");
    st.id = "qz-style";
    st.textContent = `
.qz { margin: 22px 0 26px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); padding: 14px 18px 12px; scroll-margin-top: 72px; }
.qz-top { display: flex; flex-wrap: wrap; gap: 6px 8px; align-items: center; margin: 0 0 8px; font-size: 13px; }
.qz-no { font-weight: 700; color: var(--accent-ink); margin-right: 2px; }
.qz-chip { border: 1px solid var(--line); border-radius: 999px; padding: 1px 9px; color: var(--muted); background: var(--bg); }
.qz-last { margin-left: auto; color: var(--muted); }
.qz-last.bad { color: var(--warn); }
.qz-stem { margin: 0 0 10px; font-size: 16px; }
.qz-opts { list-style: none; margin: 0 0 8px; padding: 0; counter-reset: qzopt; }
.qz-opts > li { counter-increment: qzopt; position: relative; margin: 0 0 7px; padding: 9px 12px 9px 42px; border: 1px solid var(--line);
  border-radius: 9px; cursor: pointer; background: var(--surface); line-height: 1.55; }
.qz-opts > li::before { content: counter(qzopt, upper-alpha); position: absolute; left: 12px; top: 8px; width: 21px; height: 21px; border-radius: 50%;
  border: 1px solid var(--line); font-size: 12.5px; font-weight: 700; display: flex; align-items: center; justify-content: center; color: var(--muted); }
.qz-opts > li:hover { border-color: var(--accent); }
.qz-opts > li.sel { border-color: var(--accent); background: var(--accent-soft); }
.qz-opts > li.sel::before { border-color: var(--accent); color: var(--accent-ink); }
.qz.done .qz-opts > li, .qz-step2.done .qz-opts > li { cursor: default; }
.qz-opts > li.ok { border-color: var(--good); background: var(--good-soft); }
.qz-opts > li.ok::before { content: "✓"; border-color: var(--good); color: var(--good); }
.qz-opts > li.bad { border-color: var(--warn); background: var(--warn-soft); }
.qz-opts > li.bad::before { content: "✗"; border-color: var(--warn); color: var(--warn); }
.qz-why-o { display: none; margin-top: 6px; font-size: 14px; color: var(--ink-2, var(--ink)); }
.qz-opts.shown .qz-why-o { display: block; }
.qz-fb { display: none; }
.qz.done > .qz-fb { display: block; border-top: 1px dashed var(--line); margin-top: 10px; padding-top: 10px; font-size: 15px; }
.qz-fb p { margin: 0 0 8px; }
.qz-step2 { display: none; margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--line); }
.qz-step2.open { display: block; }
.qz-ask { display: none; flex-wrap: wrap; gap: 6px 8px; align-items: center; margin: 4px 0 6px; font-size: 14px; }
.qz-ask.open { display: flex; }
.qz-ask span { color: var(--muted); margin-right: 2px; }
.qz-btn { font: inherit; font-size: 14px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 8px; padding: 5px 12px; cursor: pointer; }
.qz-btn:hover { border-color: var(--accent); color: var(--accent-ink); }
.qz-btn:disabled { opacity: .4; cursor: default; border-color: var(--line); color: var(--ink); }
.qz-btn.main { border-color: var(--accent); background: var(--accent-soft); color: var(--accent-ink); font-weight: 600; }
.qz-btn.on { border-color: var(--accent); color: var(--accent-ink); font-weight: 600; }
.qz-btn.warn { border-color: var(--warn); color: var(--warn); }
.qz-res { display: none; margin: 8px 0 0; font-weight: 600; font-size: 15px; }
.qz.done .qz-res { display: block; }
.qz-res.good { color: var(--good); } .qz-res.badr { color: var(--warn); }
.qz-res small { display: block; font-weight: 400; color: var(--ink); margin-top: 3px; font-size: 14px; }
.qz-again { display: none; margin-top: 4px; }
.qz.done .qz-again { display: inline-block; }
.qz-pl { margin: 10px 0 18px; border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 10px; background: var(--surface); padding: 12px 16px; scroll-margin-top: 72px; }
.qz-pl-h { display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: baseline; margin: 0 0 8px; }
.qz-pl-h b { font-size: 14px; color: var(--accent-ink); }
.qz-pl-h span { font-size: 14px; color: var(--muted); }
.qz-strip { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; margin: 0 0 10px; }
.qz-strip .lab { font-size: 12.5px; color: var(--muted); margin: 0 2px 0 6px; }
.qz-strip .lab:first-child { margin-left: 0; }
.qz-strip button { font: inherit; font-size: 13px; width: 30px; height: 30px; border-radius: 7px; border: 1px solid var(--line); background: var(--surface); color: var(--ink); cursor: pointer; padding: 0; }
.qz-strip button.s-ok { background: var(--good-soft); border-color: var(--good); color: var(--good); }
.qz-strip button.s-rv { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); }
.qz-strip button.s-bad { background: var(--warn-soft); border-color: var(--warn); color: var(--warn); }
.qz-strip button.cur { outline: 2px solid var(--ink); outline-offset: 1px; font-weight: 700; }
.qz-strip button.dim { opacity: .3; }
.qz-pl .row { display: flex; flex-wrap: wrap; gap: 8px; }
.qz-pl .note { font-size: 13px; color: var(--muted); margin: 8px 0 0; }
.qz-nav { display: none; }
article.qz-paged .qz-nav { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin: -10px 0 26px; }
.qz-nav .qz-pos { font-size: 13.5px; color: var(--muted); }
article.qz-paged .qz:not(.qz-cur) { display: none; }
article.qz-paged section.sec:not(.qz-cursec) { display: none; }
article.qz-only:not(.qz-paged) .qz:not(.qz-rv) { display: none; }
`;
    document.head.appendChild(st);
  }

  const refreshAll = () => {
    const recs = load();
    document.querySelectorAll(".qz[id]").forEach(q => q.classList.toggle("qz-rv", review(recs[q.id])));
    if (window.__qzPlayer) window.__qzPlayer.render();
  };

  // ---------------- 문제 하나 ----------------
  window.EXTRA_WIDGETS["qz"] = function (el) {
    const sec = el.closest("section");
    const list = sec ? Array.from(sec.querySelectorAll(".qz")) : [el];
    const no = list.indexOf(el) + 1;
    const isCase = el.dataset.case === "1";
    el.dataset.no = (isCase ? "사례 질문 " : "문제 ") + no;
    el.dataset.n = no;
    const opts1 = el.querySelector(":scope > .qz-opts");
    const step2 = el.querySelector(":scope > .qz-step2");
    const opts2 = step2 ? step2.querySelector(".qz-opts") : null;
    const fb = el.querySelector(":scope > .qz-fb");

    const top = document.createElement("div");
    top.className = "qz-top";
    top.innerHTML = `<span class="qz-no">${el.dataset.no}</span>` +
      (el.dataset.kind ? `<span class="qz-chip">${el.dataset.kind}${step2 ? " · 2단계" : ""}</span>` : "") +
      (el.dataset.lv ? `<span class="qz-chip">${el.dataset.lv}</span>` : "") + `<span class="qz-last"></span>`;
    el.prepend(top);
    const last = top.querySelector(".qz-last");

    const ask = document.createElement("div");
    ask.className = "qz-ask";
    ask.innerHTML = `<span>얼마나 확신하나요?</span>` + CONF.map(([k, t]) => `<button type="button" class="qz-btn" data-c="${k}">${t}</button>`).join("");
    opts1.after(ask);
    const res = document.createElement("p");
    res.className = "qz-res";
    (fb || el.lastElementChild).before(res);
    const again = document.createElement("button");
    again.type = "button"; again.className = "qz-btn qz-again"; again.textContent = "이 문제 다시 풀기";
    el.appendChild(again);

    let pick1 = -1, pick2 = -1, conf = null;
    const items = ol => Array.from(ol.children).filter(x => x.tagName === "LI");
    const okIndex = ol => items(ol).findIndex(li => li.dataset.ok === "1");

    function showLast() {
      const r = load()[el.id];
      if (!r) { last.textContent = ""; last.className = "qz-last"; return; }
      last.className = "qz-last" + (right(r) ? "" : " bad");
      last.textContent = `지난번 ${right(r) ? "맞힘" : "틀림"} · ${CONF_LABEL[r.conf] || ""} · ${day(r.at)}`;
    }

    function arm(ol, onPick) {
      items(ol).forEach((li, i) => {
        li.tabIndex = 0; li.setAttribute("role", "button");
        const go = () => { if (ol.dataset.locked) return; items(ol).forEach(x => x.classList.remove("sel")); li.classList.add("sel"); onPick(i); };
        li.addEventListener("click", e => { if (e.target.closest("a")) return; go(); });
        li.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      });
    }
    arm(opts1, i => { pick1 = i; ask.classList.add("open"); });
    if (opts2) arm(opts2, i => { pick2 = i; finish(); });
    ask.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      if (pick1 < 0) return;
      conf = b.dataset.c;
      opts1.dataset.locked = "1";
      ask.classList.remove("open");
      if (step2) { step2.classList.add("open"); const f = items(opts2)[0]; if (f) f.focus({ preventScroll: true }); }
      else finish();
    }));

    function mark(ol, pick) {
      const k = okIndex(ol);
      items(ol).forEach((li, i) => { li.classList.remove("sel"); if (i === k) li.classList.add("ok"); else if (i === pick) li.classList.add("bad"); });
      ol.classList.add("shown"); ol.dataset.locked = "1";
      return pick === k;
    }

    function finish() {
      const ok = mark(opts1, pick1);
      let ok2 = null;
      if (opts2) { ok2 = mark(opts2, pick2); step2.classList.add("done"); }
      el.classList.add("done");
      const k1 = L[okIndex(opts1)];
      let head, sub = "";
      if (!opts2) head = ok ? "맞혔습니다." : `틀렸습니다. 정답은 ${k1}입니다.`;
      else {
        head = ok && ok2 ? "둘 다 맞혔습니다." : !ok && !ok2 ? "둘 다 틀렸습니다." : ok ? "고른 것은 맞혔지만 이유가 달랐습니다." : "이유는 맞혔지만 고른 것이 달랐습니다.";
        const k2 = L[okIndex(opts2)];
        sub = `첫 질문 정답 ${k1}, 이유 정답 ${k2}. ` + (ok && !ok2 ? "이름은 알지만 그 방법을 고르게 하는 조건을 다시 확인해 보세요." :
          !ok && ok2 ? "조건은 알고 있으니, 그 조건에 맞는 방법이 무엇인지 연결해 보세요." : "");
      }
      if (conf === "sure" && !(ok && ok2 !== false)) sub += " 확신하고 틀린 문제는 며칠 뒤에 다시 풀어 보면 좋습니다.";
      if (conf !== "sure" && ok && ok2 !== false) sub += " 맞혔지만 확신하지 못했으니 '다시 볼 문제'에 남겨 둡니다.";
      res.className = "qz-res " + (ok && ok2 !== false ? "good" : "badr");
      res.innerHTML = head + (sub.trim() ? `<small>${sub.trim()}</small>` : "");
      const recs = load(); const prev = recs[el.id];
      recs[el.id] = { ok, ok2, conf, at: new Date().toISOString(), n: (prev && prev.n || 0) + 1 };
      store(recs);
      showLast();
      refreshAll();
    }

    function reset(quiet) {
      pick1 = pick2 = -1; conf = null;
      el.classList.remove("done");
      [opts1, opts2].forEach(ol => { if (!ol) return; delete ol.dataset.locked; ol.classList.remove("shown"); items(ol).forEach(li => li.classList.remove("sel", "ok", "bad")); });
      if (step2) step2.classList.remove("open", "done");
      ask.classList.remove("open");
      res.className = "qz-res"; res.textContent = "";
      showLast();
      if (!quiet) el.scrollIntoView({ block: "start", behavior: "smooth" });
    }
    el.__qzReset = reset;
    again.addEventListener("click", () => reset(false));
    showLast();
    el.classList.toggle("qz-rv", review(load()[el.id]));
  };

  // ---------------- 진행판: 한 문제씩 보기, 번호 줄, 다시 볼 문제, 기록 초기화 ----------------
  window.EXTRA_WIDGETS["qz_player"] = function (el) {
    const art = el.closest("article") || document.getElementById("article");
    if (!art) return;
    const qs = Array.from(art.querySelectorAll(".qz[id]"));
    if (!qs.length) return;
    const chap = (qs[0].id.match(/^([a-z]+\d\d)/) || [])[1] || "quiz";
    let cur = qs[0], paged = true, onlyRv = false, armed = false, armTimer = null;
    art.classList.add("qz-paged");

    const nav = document.createElement("div");
    nav.className = "qz-nav";
    nav.innerHTML = `<button type="button" class="qz-btn" data-n="-1">← 이전 문제</button><span class="qz-pos"></span>` +
      `<button type="button" class="qz-btn main" data-n="1">다음 문제 →</button>`;
    nav.querySelectorAll("button").forEach(b => b.addEventListener("click", () => step(+b.dataset.n)));

    const pool = () => { const recs = load(); return onlyRv ? qs.filter(q => review(recs[q.id])) : qs; };
    const secOf = q => q.closest("section.sec");

    function show(q, how) {
      if (!q) return;
      const prevSec = cur && secOf(cur);
      cur = q;
      qs.forEach(x => x.classList.toggle("qz-cur", x === q));
      art.querySelectorAll("section.sec").forEach(s => s.classList.toggle("qz-cursec", s.contains(q)));
      q.after(nav);
      const pos = get(POS); pos[chap] = q.id; put(POS, pos);
      render();
      if (how === "none") return;
      // 사례 절의 첫 질문으로 들어갈 때는 사례 설명부터 보이도록 절 제목으로, 그 밖에는 문제 카드로 옮긴다
      const s = secOf(q);
      const target = s && s !== prevSec && q === s.querySelector(".qz") && s.querySelector("h2") ? s : q;
      target.scrollIntoView({ block: "start", behavior: how === "jump" ? "auto" : "smooth" });
    }

    function step(d) {
      const P = pool();
      if (!P.length) return;
      let i = P.indexOf(cur);
      if (i < 0) i = d > 0 ? -1 : P.length;
      const j = i + d;
      if (j < 0 || j >= P.length) return;
      show(P[j]);
    }

    function render() {
      if (!document.body.contains(el)) return;
      const recs = load();
      const done = qs.filter(q => recs[q.id]);
      const good = done.filter(q => right(recs[q.id]));
      const rv = done.filter(q => review(recs[q.id]));
      if (onlyRv && !rv.length) onlyRv = false;
      const P = pool();
      const idx = P.indexOf(cur);
      art.classList.toggle("qz-paged", paged);
      art.classList.toggle("qz-only", onlyRv);
      let strip = "", lastSec = null;
      qs.forEach(q => {
        const s = secOf(q);
        if (s !== lastSec) { strip += `<span class="lab">${q.dataset.case === "1" ? "사례" : "문제"}</span>`; lastSec = s; }
        const r = recs[q.id];
        const cls = !r ? "" : right(r) && r.conf === "sure" ? "s-ok" : right(r) ? "s-rv" : "s-bad";
        const dim = onlyRv && !review(r) ? " dim" : "";
        strip += `<button type="button" class="${cls}${dim}${q === cur && paged ? " cur" : ""}" data-go="${q.id}" title="${q.dataset.no}">${q.dataset.n}</button>`;
      });
      const head = !paged ? "모든 문제를 펼쳐 보는 중" : idx >= 0 ? `${onlyRv ? "다시 볼 문제 " : ""}${idx + 1} / ${P.length}` :
        onlyRv ? "지금 문제는 다시 볼 문제가 아닙니다" : "";
      el.innerHTML = `<div class="qz-pl">` +
        `<p class="qz-pl-h"><b>${head}</b>` +
        `<span>${qs.length}문제 가운데 ${done.length}문제 풀이 · ${good.length}문제 맞힘${rv.length ? ` · 다시 볼 문제 ${rv.length}` : ""}</span></p>` +
        `<div class="qz-strip">${strip}</div>` +
        `<div class="row">` +
        `<button type="button" class="qz-btn${onlyRv ? " on" : ""}" data-a="rv"${rv.length || onlyRv ? "" : " disabled"}>${onlyRv ? "모든 문제로 돌아가기" : `다시 볼 문제만 풀기${rv.length ? ` (${rv.length})` : ""}`}</button>` +
        `<button type="button" class="qz-btn" data-a="mode">${paged ? "모든 문제 펼쳐 보기" : "한 문제씩 보기"}</button>` +
        `<button type="button" class="qz-btn${armed ? " warn" : ""}" data-a="reset"${done.length ? "" : " disabled"}>${armed ? "한 번 더 누르면 이 장 기록을 지웁니다" : "기록 초기화"}</button>` +
        `</div>` +
        `<p class="note">번호 색: 초록은 확신하고 맞힘, 파랑은 맞혔지만 확신하지 못함, 주황은 틀림입니다. 틀린 문제와 확신하지 못하고 맞힌 문제가 '다시 볼 문제'가 됩니다. 기록은 이 브라우저에만 저장됩니다.</p></div>`;
      const n = nav.querySelector(".qz-pos");
      n.textContent = idx >= 0 ? `${idx + 1} / ${P.length}` : "";
      nav.querySelector('[data-n="-1"]').disabled = idx === 0 || !P.length;
      const nx = nav.querySelector('[data-n="1"]');
      const atEnd = idx >= 0 && idx >= P.length - 1;
      nx.disabled = atEnd || !P.length;
      nx.textContent = atEnd ? "마지막 문제입니다" : "다음 문제 →";
      el.querySelectorAll("[data-go]").forEach(b => b.addEventListener("click", () => {
        const q = document.getElementById(b.dataset.go);
        if (onlyRv && !review(load()[q.id])) onlyRv = false;
        if (paged) show(q); else q.scrollIntoView({ block: "start", behavior: "smooth" });
      }));
      el.querySelector('[data-a="rv"]').addEventListener("click", () => {
        onlyRv = !onlyRv;
        const P2 = pool();
        if (paged && P2.length) show(P2.includes(cur) ? cur : P2[0], "none"); else render();
        el.scrollIntoView({ block: "start", behavior: "smooth" });
      });
      el.querySelector('[data-a="mode"]').addEventListener("click", () => {
        paged = !paged;
        if (paged) show(cur, "none"); else { qs.forEach(x => x.classList.remove("qz-cur")); render(); }
        el.scrollIntoView({ block: "start", behavior: "smooth" });
      });
      el.querySelector('[data-a="reset"]').addEventListener("click", () => {
        if (!armed) { armed = true; clearTimeout(armTimer); armTimer = setTimeout(() => { armed = false; render(); }, 4000); render(); return; }
        armed = false; clearTimeout(armTimer);
        const all = load(); qs.forEach(q => delete all[q.id]); store(all);
        const pos = get(POS); delete pos[chap]; put(POS, pos);
        qs.forEach(q => { if (q.__qzReset) q.__qzReset(true); q.classList.remove("qz-rv"); });
        onlyRv = false;
        if (paged) show(qs[0], "none"); else render();
        el.scrollIntoView({ block: "start", behavior: "smooth" });
      });
    }

    function fromHash(how) {
      const h = decodeURIComponent(location.hash.slice(1));
      if (!h) return false;
      const q = qs.find(x => x.id === h);
      if (q) { if (onlyRv && !review(load()[q.id])) onlyRv = false; if (paged) show(q, how); return true; }
      const s = art.querySelector(`section.sec[id="${h}"]`);
      if (s && s.querySelector(".qz[id]")) { onlyRv = false; if (paged) show(s.querySelector(".qz[id]"), how); return true; }
      return false;
    }
    if (window.__qzHash) window.removeEventListener("hashchange", window.__qzHash);
    window.__qzHash = () => { if (document.body.contains(el)) fromHash("jump"); };
    window.addEventListener("hashchange", window.__qzHash);
    window.__qzPlayer = { render };

    // 문제 위젯이 모두 붙은 뒤에 시작: 주소의 문제 → 마지막으로 보던 문제 → 첫 문제
    setTimeout(() => {
      if (fromHash("jump")) return;
      const saved = get(POS)[chap];
      show(qs.find(x => x.id === saved) || qs[0], "none");
    }, 0);
  };
})();
