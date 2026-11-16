// 보건통계학 퀴즈 과목 공통 엔진 (build.py가 quiz.html에만 넣는다).
// 문제 마크업: <div class="qz" data-widget="qz" id="…" data-kind="…" data-lv="…">
//   <p class="qz-stem">…</p><ol class="qz-opts"><li data-ok="1">선택지<div class="qz-why-o">선택지 해설</div></li>…</ol>
//   [<div class="qz-step2"><p class="qz-stem">…</p><ol class="qz-opts">…</ol></div>]   (2단계 문항: 고른 것 + 그 이유)
//   <div class="qz-fb">해설(핵심 이유, 조건이 달라지면, 다시 읽을 곳)</div></div>
// 기록은 이 브라우저의 localStorage에만 둔다: { 문제 id: { ok, ok2, conf, at, n } }
(function () {
  const KEY = "socialp-quiz-v1";
  const CONF = [["sure", "확실함"], ["unsure", "헷갈림"], ["guess", "추측"]];
  const CONF_LABEL = Object.fromEntries(CONF);
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { return {}; } };
  const store = o => { try { localStorage.setItem(KEY, JSON.stringify(o)); } catch (e) {} };
  const right = r => !!r && r.ok && r.ok2 !== false;
  const review = r => !!r && (!right(r) || r.conf !== "sure");
  const day = iso => { const d = new Date(iso); return isNaN(d) ? "" : `${d.getMonth() + 1}월 ${d.getDate()}일`; };
  const L = "ABCDEFGH";

  if (!document.getElementById("qz-style")) {
    const st = document.createElement("style");
    st.id = "qz-style";
    st.textContent = `
.qz { margin: 22px 0 26px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); padding: 14px 18px 12px; }
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
.qz-btn.main { border-color: var(--accent); background: var(--accent-soft); color: var(--accent-ink); font-weight: 600; }
.qz-res { display: none; margin: 8px 0 0; font-weight: 600; font-size: 15px; }
.qz.done .qz-res { display: block; }
.qz-res.good { color: var(--good); } .qz-res.badr { color: var(--warn); }
.qz-res small { display: block; font-weight: 400; color: var(--ink); margin-top: 3px; font-size: 14px; }
.qz-again { display: none; margin-top: 4px; }
.qz.done .qz-again { display: inline-block; }
.qz-sum { margin: 10px 0 26px; border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: 10px; background: var(--surface); padding: 12px 16px; }
.qz-sum-h { margin: 0 0 4px; font-weight: 700; font-size: 14px; color: var(--accent-ink); }
.qz-sum p { margin: 0 0 8px; font-size: 15px; }
.qz-sum ul { margin: 0 0 10px; padding-left: 18px; font-size: 14.5px; }
.qz-sum li { margin: 2px 0; }
.qz-sum .tag { color: var(--warn); font-weight: 600; }
.qz-sum .row { display: flex; flex-wrap: wrap; gap: 8px; }
.qz-sum .note { font-size: 13.5px; color: var(--muted); margin: 8px 0 0; }
article.qz-only .qz:not(.qz-rv) { display: none; }
`;
    document.head.appendChild(st);
  }

  function refreshAll() {
    const recs = load();
    document.querySelectorAll(".qz[id]").forEach(q => q.classList.toggle("qz-rv", review(recs[q.id])));
    if (window.__qzSummary) window.__qzSummary();
  }

  window.EXTRA_WIDGETS["qz"] = function (el) {
    const sec = el.closest("section");
    const list = sec ? Array.from(sec.querySelectorAll(".qz")) : [el];
    const no = list.indexOf(el) + 1;
    const isCase = el.dataset.case === "1";
    el.dataset.no = (isCase ? "사례 질문 " : "문제 ") + no;
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
    again.type = "button"; again.className = "qz-btn qz-again"; again.textContent = "다시 풀기";
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

    again.addEventListener("click", () => {
      pick1 = pick2 = -1; conf = null;
      el.classList.remove("done");
      [opts1, opts2].forEach(ol => { if (!ol) return; delete ol.dataset.locked; ol.classList.remove("shown"); items(ol).forEach(li => li.classList.remove("sel", "ok", "bad")); });
      if (step2) step2.classList.remove("open", "done");
      ask.classList.remove("open");
      res.className = "qz-res"; res.textContent = "";
      el.scrollIntoView({ block: "start", behavior: "smooth" });
    });
    showLast();
    el.classList.toggle("qz-rv", review(load()[el.id]));
  };

  window.EXTRA_WIDGETS["qz_summary"] = function (el) {
    let armed = false;
    function render() {
      if (!document.body.contains(el)) return;
      const qs = Array.from(document.querySelectorAll(".qz[id]"));
      const recs = load();
      const done = qs.filter(q => recs[q.id]);
      const good = done.filter(q => right(recs[q.id]));
      const rv = done.filter(q => review(recs[q.id]))
        .sort((a, b) => (right(recs[a.id]) - right(recs[b.id])) || ((recs[b.id].conf === "sure") - (recs[a.id].conf === "sure")));
      const name = q => q.dataset.no || q.id;
      const line = q => {
        const r = recs[q.id];
        const sureWrong = !right(r) && r.conf === "sure";
        return `<li><a href="#${q.id}">${name(q)}</a> · ${right(r) ? "맞힘" : "틀림"} · ${CONF_LABEL[r.conf] || ""} · ${day(r.at)}` +
          (sureWrong ? ` <span class="tag">확신하고 틀림</span>` : "") + `</li>`;
      };
      const art = el.closest("article") || document.getElementById("article");
      const only = art && art.classList.contains("qz-only");
      el.innerHTML = `<div class="qz-sum"><p class="qz-sum-h">이 장의 기록</p>` +
        `<p>${qs.length}문제 가운데 ${done.length}문제를 풀었고 ${good.length}문제를 맞혔습니다.` +
        (rv.length ? ` 다시 볼 문제는 ${rv.length}개입니다.` : (done.length ? " 다시 볼 문제가 없습니다." : "")) + `</p>` +
        (rv.length ? `<ul>${rv.map(line).join("")}</ul>` : "") +
        `<div class="row">` + (rv.length ? `<button type="button" class="qz-btn main" data-a="only">${only ? "모든 문제 보기" : "다시 볼 문제만 보기"}</button>` : "") +
        (done.length ? `<button type="button" class="qz-btn" data-a="clear">${armed ? "한 번 더 누르면 지웁니다" : "이 장 기록 지우기"}</button>` : "") + `</div>` +
        `<p class="note">기록은 이 브라우저에만 저장됩니다. 틀린 문제와 확신하지 못하고 맞힌 문제가 '다시 볼 문제'에 남고, 확신하고 틀린 문제를 맨 위에 둡니다.</p></div>`;
      const ob = el.querySelector('[data-a="only"]');
      if (ob) ob.addEventListener("click", () => { if (art) art.classList.toggle("qz-only"); render(); });
      const cb = el.querySelector('[data-a="clear"]');
      if (cb) cb.addEventListener("click", () => {
        if (!armed) { armed = true; render(); setTimeout(() => { armed = false; render(); }, 4000); return; }
        const all = load(); qs.forEach(q => delete all[q.id]); store(all); armed = false;
        if (art) art.classList.remove("qz-only");
        qs.forEach(q => { const l = q.querySelector(".qz-last"); if (l) { l.textContent = ""; l.className = "qz-last"; } });
        refreshAll();
      });
    }
    window.__qzSummary = render;
    // 문제 위젯이 모두 붙은 뒤(번호가 정해진 뒤)에 그린다
    setTimeout(render, 0);
  };
})();
