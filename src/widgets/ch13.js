window.EXTRA_WIDGETS["ch13_ni"] = function (el, H) {
  // difference = new - standard (higher is better), margin -D; normal approximation
  const PRE = {
    A: [8.0, 5.5], B: [1.0, 6.5], C: [-4.5, 4.0], D: [-2.5, 10.5], E: [-8.5, 6.0], F: [-17.0, 6.0],
  };
  let est = 1.0, hw = 6.5, D = 10;
  const mfmt = (v, d = 1) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d);
  function Phi(x) { // Abramowitz & Stegun 26.2.17
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327 * Math.exp(-x * x / 2);
    const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - p : p;
  }
  const pfmt = p => p < 0.001 ? "< 0.001" : p.toFixed(3);
  const peq = p => p < 0.001 ? "p < 0.001" : "p = " + p.toFixed(3);
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">계산기</span></p>
    <p class="wd">가로축은 완치율 차이(신약 − 표준, %p)이고 클수록 신약이 좋습니다. 점추정값, 95% 신뢰구간의 반폭(= 1.96 × 표준오차), 비열등성 한계 Δ를 바꾸거나 그림을 좌우로 끌어 보세요. 가는 선은 95% CI, 굵은 띠는 90% CI입니다. A–F는 그림 13-1의 여섯 경우입니다.</p>
    <div class="wrow"><label for="niEst">점추정값 <b id="niEstV"></b></label><input id="niEst" type="range" min="-20" max="15" step="0.5"></div>
    <div class="wrow"><label for="niHw">95% CI 반폭 <b id="niHwV"></b></label><input id="niHw" type="range" min="1" max="12" step="0.5"></div>
    <div class="wrow"><label for="niD">비열등성 한계 Δ <b id="niDV"></b></label><input id="niD" type="range" min="2" max="15" step="0.5"></div>
    <div class="wrow"><span>그림 13-1의 경우</span><div class="seg" role="group" aria-label="예시 경우">${Object.keys(PRE).map(k => `<button type="button" data-k="${k}" aria-pressed="false">${k}</button>`).join("")}</div></div>
    <div class="wchart" id="niChart"></div>
    <div class="stats">
      <div class="stat"><div class="sl">95% 신뢰구간</div><div class="sv" id="niCI"></div></div>
      <div class="stat"><div class="sl">비열등성 (95% 하한 &gt; −Δ)</div><div class="sv" id="niNI"></div></div>
      <div class="stat"><div class="sl">우월성 (95% 하한 &gt; 0)</div><div class="sv" id="niSup"></div></div>
      <div class="stat"><div class="sl">동등성 (90% CI가 ±Δ 안)</div><div class="sv" id="niEq"></div></div>
      <div class="stat"><div class="sl">p (차이 = 0, 양측)</div><div class="sv" id="niP0"></div></div>
      <div class="stat"><div class="sl">p (비열등성, 단측)</div><div class="sv" id="niP1"></div></div>
    </div>
    <p class="wd" id="niNote" style="margin-top:10px"></p>`;
  const $ = s => el.querySelector(s);
  const iE = $("#niEst"), iH = $("#niHw"), iD = $("#niD");
  const W = 640, Hh = 200, L = 24, R = 24, Tp = 40, B = 50, X0 = -30, X1 = 20;
  const sx = x => L + (x - X0) / (X1 - X0) * (W - L - R);
  const invx = X => X0 + (X - L) / (W - L - R) * (X1 - X0);

  function category(lo, hi) {
    if (lo > 0) return ["A", "우월성 입증 (비열등성도 입증)"];
    if (lo > -D && hi >= 0) return ["B", "비열등성 입증 (우월성은 입증 못함)"];
    if (lo > -D && hi < 0) return ["C", "비열등성 입증, 단 신약이 통계적으로 유의하게 낮음"];
    if (hi <= -D) return ["F", "열등 (구간 전체가 −Δ보다 왼쪽)"];
    if (hi >= 0) return ["D", "결론 불가 (차이도, 비열등성도 입증 못함)"];
    return ["E", "결론 불가 (신약이 통계적으로 낮고, 비열등성도 입증 못함)"];
  }

  function draw() {
    const se = hw / 1.959964, hw90 = 1.644854 * se;
    const lo = est - hw, hi = est + hw, lo9 = est - hw90, hi9 = est + hw90;
    const yc = 105, base = Hh - B;
    let s = "";
    s += `<rect x="${sx(X0)}" y="${Tp}" width="${sx(-D) - sx(X0)}" height="${base - Tp}" class="a2"/>`;
    for (const t of [-30, -20, -10, 0, 10, 20]) s += `<line x1="${sx(t)}" y1="${base}" x2="${sx(t)}" y2="${base + 4}" class="axis"/><text x="${sx(t)}" y="${base + 18}" text-anchor="middle" class="tick">${mfmt(t, 0)}</text>`;
    s += `<line x1="${L}" y1="${base}" x2="${W - R}" y2="${base}" class="axis"/>`;
    s += `<text x="${(L + W - R) / 2}" y="${Hh - 10}" text-anchor="middle" class="axlab">완치율 차이 (신약 − 표준, %p)</text>`;
    s += `<line x1="${sx(-D)}" y1="${Tp}" x2="${sx(-D)}" y2="${base}" class="ref strongref" stroke-width="1.4"/>`;
    s += `<line x1="${sx(D)}" y1="${Tp}" x2="${sx(D)}" y2="${base}" class="ref" stroke-width="1.2" stroke-dasharray="3 4"/>`;
    s += `<line x1="${sx(0)}" y1="${Tp}" x2="${sx(0)}" y2="${base}" class="ref" stroke-width="1.2" stroke-dasharray="4 4"/>`;
    s += `<text x="${sx(-D)}" y="${Tp - 8}" text-anchor="middle" class="lbl small">−Δ = ${mfmt(-D)}</text>`;
    s += `<text x="${sx(D)}" y="${Tp - 8}" text-anchor="middle" class="lbl small mute">+Δ</text>`;
    s += `<text x="${sx(0)}" y="${Tp - 22}" text-anchor="middle" class="lbl small mute">0</text>`;
    if (sx(-D) - sx(X0) > 70) s += `<text x="${(sx(X0) + sx(-D)) / 2}" y="${Tp + 16}" text-anchor="middle" class="lbl small">열등 영역</text>`;
    const cl = v => Math.max(X0, Math.min(X1, v));
    s += `<rect x="${sx(cl(lo9))}" y="${yc - 9}" width="${Math.max(0, sx(cl(hi9)) - sx(cl(lo9)))}" height="18" rx="3" class="a1"/>`;
    s += `<line x1="${sx(cl(lo))}" y1="${yc}" x2="${sx(cl(hi))}" y2="${yc}" class="ln s1" stroke-width="2.4"/>`;
    for (const v of [lo, hi]) if (v >= X0 && v <= X1) s += `<line x1="${sx(v)}" y1="${yc - 12}" x2="${sx(v)}" y2="${yc + 12}" class="ln s1" stroke-width="2"/>`;
    s += `<rect x="${sx(cl(est)) - 6}" y="${yc - 6}" width="12" height="12" class="f1"/>`;
    s += `<text x="${sx(cl(est))}" y="${yc + 32}" text-anchor="middle" class="lbl small">${mfmt(est)} (${mfmt(lo)} ~ ${mfmt(hi)})</text>`;
    $("#niChart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="신뢰구간과 비열등성 한계" style="touch-action:pan-y;cursor:ew-resize">${s}</svg>`;
    $("#niEstV").textContent = mfmt(est) + "%p";
    $("#niHwV").textContent = "±" + hw.toFixed(1) + "%p";
    $("#niDV").textContent = D.toFixed(1) + "%p";
    iE.value = est; iH.value = hw; iD.value = D;
    $("#niCI").textContent = `${mfmt(lo)} ~ ${mfmt(hi)}`;
    const ni = lo > -D, sup = lo > 0, eq = lo9 > -D && hi9 < D;
    $("#niNI").textContent = ni ? "입증" : "입증 못함";
    $("#niSup").textContent = sup ? "입증" : "입증 못함";
    $("#niEq").textContent = eq ? "입증" : "입증 못함";
    const p0 = 2 * (1 - Phi(Math.abs(est / se)));
    const p1 = 1 - Phi((est + D) / se);
    $("#niP0").textContent = pfmt(p0);
    $("#niP1").textContent = pfmt(p1);
    const [k, txt] = category(lo, hi);
    let note = `판정: ${k}. ${txt}. 표준오차 = ${hw.toFixed(1)} ÷ 1.96 = ${se.toFixed(2)}%p, 90% CI = ${mfmt(lo9)} ~ ${mfmt(hi9)}.`;
    if (!sup && p0 >= 0.05 && !ni) note += " 차이 검정은 '유의하지 않음'이지만 비열등성도 입증하지 못했습니다. \"유의한 차이가 없으므로 비슷하다\"고 쓰면 안 되는 경우입니다.";
    else if (ni && p0 >= 0.05) note += ` 차이 검정은 유의하지 않고(${peq(p0)}), 비열등성은 입증되었습니다(단측 ${peq(p1)}, 기준 0.025).`;
    if (ni && !eq) note += " 비열등성은 입증됐지만 90% CI가 ±Δ를 벗어나 동등성은 입증하지 못했습니다.";
    $("#niNote").textContent = note;
    el.querySelectorAll(".seg button").forEach(b => {
      const q = PRE[b.dataset.k];
      b.setAttribute("aria-pressed", Math.abs(q[0] - est) < 1e-9 && Math.abs(q[1] - hw) < 1e-9 && D === 10);
    });
  }
  iE.addEventListener("input", () => { est = +iE.value; draw(); });
  iH.addEventListener("input", () => { hw = +iH.value; draw(); });
  iD.addEventListener("input", () => { D = +iD.value; draw(); });
  el.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => { [est, hw] = PRE[b.dataset.k]; D = 10; draw(); }));
  const chart = $("#niChart");
  let drag = false;
  function moveTo(ev) {
    const svg = chart.querySelector("svg");
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const X = (ev.clientX - r.left) / r.width * W;
    est = Math.max(-20, Math.min(15, Math.round(invx(X) * 2) / 2));
    draw();
  }
  chart.addEventListener("pointerdown", ev => { drag = true; moveTo(ev); });
  window.addEventListener("pointermove", ev => { if (drag) moveTo(ev); });
  window.addEventListener("pointerup", () => { drag = false; });
  draw();
};
