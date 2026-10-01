window.EXTRA_WIDGETS["ch07_km"] = function (el, H) {
  // 12 patients of 표 7-1 (time in months, status 1 = death, 0 = censored)
  const T = [3, 5, 8, 12, 12, 16, 20, 25, 28, 32, 38, 44];
  const S0 = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0];
  let st = S0.slice();
  const btn = (i) => `<button type="button" data-i="${i}" aria-pressed="${st[i] === 1}" title="#${i + 1}, ${T[i]}개월">${T[i]}</button>`;
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">계산기</span></p>
    <p class="wd">표 7-1의 환자 12명입니다. 버튼의 숫자는 각 환자의 추적 시간(개월)이고, 색이 채워진 버튼은 사망, 빈 버튼은 중도절단입니다. 버튼을 눌러 사망과 중도절단을 바꾸면 Kaplan-Meier 곡선(파랑)이 원래 곡선(회색 점선)과 어떻게 달라지는지 봅니다.</p>
    <div class="wrow"><div class="seg" role="group" aria-label="환자 1-6">${[0, 1, 2, 3, 4, 5].map(btn).join("")}</div>
    <div class="seg" role="group" aria-label="환자 7-12">${[6, 7, 8, 9, 10, 11].map(btn).join("")}</div></div>
    <div class="wrow"><button type="button" class="wbtn" id="kmReset">원래 자료로</button>
    <button type="button" class="wbtn" id="kmAll">중도절단 5명을 모두 사망으로</button></div>
    <div class="wchart" id="kmChart"></div>
    <div class="stats">
      <div class="stat"><div class="sl">사망 수</div><div class="sv" id="kmD"></div></div>
      <div class="stat"><div class="sl">2년 생존율 S(24)</div><div class="sv" id="kmS24"></div></div>
      <div class="stat"><div class="sl">중앙생존기간</div><div class="sv" id="kmMed"></div></div>
      <div class="stat"><div class="sl">36개월 RMST</div><div class="sv" id="kmR"></div></div>
    </div>
    <p class="wd" id="kmNote" style="margin-top:10px"></p>`;

  function km(status) {
    const times = [...new Set(T)].sort((a, b) => a - b);
    let S = 1; const out = [];
    for (const t of times) {
      let n = 0, d = 0, c = 0;
      T.forEach((x, i) => { if (x >= t) n++; if (x === t) { if (status[i]) d++; else c++; } });
      if (d) S *= 1 - d / n;
      out.push({ t, n, d, c, S });
    }
    return out;
  }
  const at = (rows, x) => { let S = 1; for (const r of rows) if (r.t <= x) S = r.S; return S; };
  function median(rows) {
    const ev = rows.filter(r => r.d > 0);
    for (let i = 0; i < ev.length; i++) {
      if (ev[i].S <= 0.5 + 1e-12) {
        if (Math.abs(ev[i].S - 0.5) < 1e-12) {
          const nx = ev.slice(i + 1).find(r => r.S < 0.5 - 1e-12);
          // midpoint convention (as in R quantile.survfit): flat at exactly 0.5 until the end -> midpoint with the last follow-up time; lifelines returns the start of the flat stretch
          return nx ? { v: (ev[i].t + nx.t) / 2, mid: true } : { v: (ev[i].t + Math.max(...T)) / 2, mid: true };
        }
        return { v: ev[i].t, mid: false };
      }
    }
    return null;
  }
  function rmst(rows, tau) {
    let a = 0, prev = 0, S = 1;
    for (const r of rows) { if (r.d > 0 && r.t < tau) { a += S * (r.t - prev); prev = r.t; S = r.S; } }
    return a + S * (tau - prev);
  }
  function stepPath(rows, sx, sy, xmax) {
    let d = `M${sx(0)},${sy(1)}`, S = 1;
    for (const r of rows) if (r.d > 0) { d += ` L${sx(r.t)},${sy(S)} L${sx(r.t)},${sy(r.S)}`; S = r.S; }
    return d + ` L${sx(Math.min(Math.max(...T), xmax))},${sy(S)}`;
  }
  const orig = km(S0);

  function draw() {
    const rows = km(st);
    const W = 640, Hh = 300, L = 78, R = 20, Tp = 16, B = 70;
    const sx = x => L + x / 48 * (W - L - R), sy = y => Hh - B - y * (Hh - Tp - B);
    let s = "";
    for (const y of [0, 0.2, 0.4, 0.6, 0.8, 1]) s += `<line x1="${L}" y1="${sy(y)}" x2="${W - R}" y2="${sy(y)}" class="grid"/><text x="${L - 8}" y="${sy(y) + 4}" text-anchor="end" class="tick">${y.toFixed(1)}</text>`;
    s += `<line x1="${L}" y1="${sy(0)}" x2="${W - R}" y2="${sy(0)}" class="axis"/>`;
    for (const x of [0, 6, 12, 18, 24, 30, 36, 42, 48]) s += `<text x="${sx(x)}" y="${sy(0) + 18}" text-anchor="middle" class="tick">${x}</text>`;
    s += `<text x="${(L + W - R) / 2}" y="${sy(0) + 36}" text-anchor="middle" class="axlab">치료 시작 후 개월 수</text>`;
    s += `<line x1="${L}" y1="${sy(0.5)}" x2="${W - R}" y2="${sy(0.5)}" class="ref" stroke-dasharray="4 4"/>`;
    s += `<path d="${stepPath(orig, sx, sy, 48)}" class="ln s4" stroke-width="2" stroke-dasharray="6 5" fill="none"/>`;
    s += `<path d="${stepPath(rows, sx, sy, 48)}" class="ln s1" stroke-width="2.6" fill="none"/>`;
    let S = 1;
    for (const r of rows) { if (r.d > 0) S = r.S; if (r.c > 0) s += `<line x1="${sx(r.t)}" y1="${sy(S) - 6}" x2="${sx(r.t)}" y2="${sy(S) + 6}" class="ln s1" stroke-width="1.6"/>`; }
    const m = median(rows);
    if (m) s += `<line x1="${sx(m.v)}" y1="${sy(0.5)}" x2="${sx(m.v)}" y2="${sy(0)}" class="ref" stroke-dasharray="4 4"/>`;
    const s24 = at(rows, 24);
    s += `<circle cx="${sx(24)}" cy="${sy(s24)}" r="4.5" class="pt f2"/>`;
    // number at risk
    s += `<text x="6" y="${sy(0) + 60}" class="lbl small">위험집합</text>`;
    for (const x of [0, 12, 24, 36, 48]) s += `<text x="${sx(x)}" y="${sy(0) + 60}" text-anchor="middle" class="lbl small">${T.filter(v => v >= x).length}</text>`;
    s += `<line x1="${W - R - 200}" y1="${Tp + 8}" x2="${W - R - 178}" y2="${Tp + 8}" class="ln s4" stroke-width="2" stroke-dasharray="6 4"/><text x="${W - R - 172}" y="${Tp + 12}" class="lbl small">원래 자료</text>`;
    s += `<line x1="${W - R - 104}" y1="${Tp + 8}" x2="${W - R - 82}" y2="${Tp + 8}" class="ln s1" stroke-width="2.6"/><text x="${W - R - 76}" y="${Tp + 12}" class="lbl small">바꾼 자료</text>`;
    el.querySelector("#kmChart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="Kaplan-Meier 곡선">${s}</svg>`;
    const nd = st.reduce((a, b) => a + b, 0);
    el.querySelector("#kmD").textContent = `${nd} / 12명`;
    el.querySelector("#kmS24").textContent = H.fmt(s24 * 100, 1) + "%";
    el.querySelector("#kmMed").textContent = m ? H.fmt(m.v, m.v % 1 ? 1 : 0) + "개월" : "도달 안 함";
    el.querySelector("#kmR").textContent = H.fmt(rmst(rows, 36), 1) + "개월";
    const last = rows[rows.length - 1];
    let note = m && m.mid ? "곡선이 정확히 0.5인 구간이 있어, 그 구간의 가운데를 중앙값으로 표시했습니다. 이런 경우의 규칙은 프로그램마다 달라서, 파이썬 lifelines는 곡선이 처음 0.5가 되는 시점(구간의 시작)을 중앙값으로 출력합니다. " : "";
    if (!m) note += "곡선이 0.5 아래로 내려가지 않아 중앙생존기간을 추정할 수 없습니다(not reached). ";
    if (last.S === 0) note += "마지막 관찰 환자가 사망으로 바뀌어 곡선이 0까지 떨어졌습니다. 한 사람의 상태가 꼬리 모양 전체를 바꿉니다.";
    el.querySelector("#kmNote").textContent = note;
    el.querySelectorAll(".seg button").forEach(b => b.setAttribute("aria-pressed", st[+b.dataset.i] === 1));
  }
  el.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { const i = +b.dataset.i; st[i] = 1 - st[i]; draw(); }));
  el.querySelector("#kmReset").addEventListener("click", () => { st = S0.slice(); draw(); });
  el.querySelector("#kmAll").addEventListener("click", () => { st = T.map(() => 1); draw(); });
  draw();
};
