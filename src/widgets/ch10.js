window.EXTRA_WIDGETS["ch10_margcond"] = function (el, H) {
  // defaults = random-intercept logistic fit of the chapter example (3-month visit)
  const D = { sig: 1.64, or: 3.77, p0: 0.674 };
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">계산기</span></p>
    <p class="wd">환자별(조건부) 오즈비는 고정한 채 환자 간 차이(무작위 절편의 SD)를 바꿔 보세요. 모든 환자에게서 오즈비가 같아도, 환자 분포 전체로 평균한 순응률로 계산한 모집단 평균(주변) 오즈비는 SD가 클수록 1 쪽으로 줄어듭니다. 처음 값은 이 장 예제의 혼합모형 추정값입니다.</p>
    <div class="wrow"><label for="w10s">환자 간 SD σ<sub>u</sub> = <b id="w10sv"></b></label><input id="w10s" type="range" min="0" max="3" step="0.01" value="${D.sig}"></div>
    <div class="wrow"><label for="w10o">환자별(조건부) OR = <b id="w10ov"></b></label><input id="w10o" type="range" min="1" max="6" step="0.01" value="${D.or}"></div>
    <div class="wrow"><label for="w10p">통상치료군 '중간' 환자(u = 0)의 순응 확률 = <b id="w10pv"></b></label><input id="w10p" type="range" min="0.1" max="0.9" step="0.001" value="${D.p0}"></div>
    <div class="wrow"><button type="button" class="wbtn" id="w10r">예제 값으로</button></div>
    <div class="wchart" id="w10c"></div>
    <div class="stats">
      <div class="stat"><div class="sl">조건부 OR (혼합모형)</div><div class="sv" id="w10a"></div></div>
      <div class="stat"><div class="sl">모집단 평균 순응률 (통상 / 중재)</div><div class="sv" id="w10b"></div></div>
      <div class="stat"><div class="sl">주변 OR (GEE가 추정하는 값)</div><div class="sv" id="w10m"></div></div>
      <div class="stat"><div class="sl">근사식 β / √(1 + 0.346σ²)</div><div class="sv" id="w10x"></div></div>
    </div>`;
  const $ = (id) => el.querySelector("#" + id);
  const expit = (x) => 1 / (1 + Math.exp(-x));
  const logit = (p) => Math.log(p / (1 - p));
  function marg(eta, sig) {
    if (sig < 1e-6) return expit(eta);
    // integrate expit(eta + u) over u ~ N(0, sig^2) on a fine grid (Simpson's rule, +-7 SD)
    const n = 400, a = -7 * sig, h = (14 * sig) / n;
    let s = 0;
    for (let i = 0; i <= n; i++) {
      const u = a + i * h, w = (i === 0 || i === n) ? 1 : (i % 2 ? 4 : 2);
      s += w * expit(eta + u) * Math.exp(-0.5 * (u / sig) ** 2);
    }
    return s * h / 3 / (sig * Math.sqrt(2 * Math.PI));
  }
  function draw() {
    const sig = +$("w10s").value, or = +$("w10o").value, p0 = +$("w10p").value;
    $("w10sv").textContent = H.fmt(sig, 2);
    $("w10ov").textContent = H.fmt(or, 2);
    $("w10pv").textContent = H.fmt(p0 * 100, 1) + "%";
    const b0 = logit(p0), b1 = Math.log(or);
    const P0 = marg(b0, sig), P1 = marg(b0 + b1, sig);
    const mor = (P1 / (1 - P1)) / (P0 / (1 - P0));
    const approx = Math.exp(b1 / Math.sqrt(1 + 0.346 * sig * sig));
    const W = 640, Hh = 320, L = 62, R = 20, T = 18, B = 52;
    const sx = (x) => L + (x + 4.5) / 9 * (W - L - R), sy = (y) => Hh - B - y * (Hh - T - B);
    let s = "";
    for (const y of [0, 0.2, 0.4, 0.6, 0.8, 1]) s += `<line x1="${L}" y1="${sy(y)}" x2="${W - R}" y2="${sy(y)}" class="grid"/><text x="${L - 8}" y="${sy(y) + 4}" text-anchor="end" class="tick">${Math.round(y * 100)}%</text>`;
    s += `<line x1="${L}" y1="${sy(0)}" x2="${W - R}" y2="${sy(0)}" class="axis"/>`;
    for (let x = -4; x <= 4; x++) s += `<text x="${sx(x)}" y="${sy(0) + 18}" text-anchor="middle" class="tick">${x}</text>`;
    s += `<text x="${(L + W - R) / 2}" y="${Hh - 10}" text-anchor="middle" class="axlab">환자 고유의 순응 성향 u (로짓 척도)</text>`;
    // density of u
    if (sig > 0.05) {
      let d = `M${sx(-4.5)},${sy(0)}`;
      const pk = 1 / (sig * Math.sqrt(2 * Math.PI));
      for (let i = 0; i <= 180; i++) { const u = -4.5 + 9 * i / 180; const f = Math.exp(-0.5 * (u / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI)); d += ` L${sx(u).toFixed(1)},${sy(0.16 * f / pk).toFixed(1)}`; }
      d += ` L${sx(4.5)},${sy(0)} Z`;
      s += `<path d="${d}" class="a4"/>`;
    }
    const curve = (b) => { let d = ""; for (let i = 0; i <= 180; i++) { const u = -4.5 + 9 * i / 180; d += (i ? " L" : "M") + sx(u).toFixed(1) + "," + sy(expit(b + u)).toFixed(1); } return d; };
    s += `<path d="${curve(b0 + b1)}" class="ln s1" stroke-width="2.4" fill="none"/>`;
    s += `<path d="${curve(b0)}" class="ln s2" stroke-width="2.4" stroke-dasharray="6 5" fill="none"/>`;
    for (const [P, lab] of [[P1, "중재군 평균"], [P0, "통상치료군 평균"]]) {
      s += `<line x1="${L}" y1="${sy(P)}" x2="${W - R}" y2="${sy(P)}" class="ref" stroke-dasharray="4 4"/>`;
      s += `<text x="${L + 6}" y="${sy(P) - 6}" class="lbl small">${lab} ${H.fmt(P * 100, 1)}%</text>`;
    }
    s += `<line x1="${L + 6}" y1="${sy(0.97)}" x2="${L + 28}" y2="${sy(0.97)}" class="ln s1" stroke-width="2.4"/><text x="${L + 34}" y="${sy(0.97) + 4}" class="lbl small">중재군 환자</text>`;
    s += `<line x1="${L + 6}" y1="${sy(0.97) + 18}" x2="${L + 28}" y2="${sy(0.97) + 18}" class="ln s2" stroke-width="2.4" stroke-dasharray="6 4"/><text x="${L + 34}" y="${sy(0.97) + 22}" class="lbl small">통상치료군 환자</text>`;
    $("w10c").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="조건부와 주변 순응 확률">${s}</svg>`;
    $("w10a").textContent = H.fmt(or, 2);
    $("w10b").textContent = H.fmt(P0 * 100, 1) + "% / " + H.fmt(P1 * 100, 1) + "%";
    $("w10m").textContent = H.fmt(mor, 2);
    $("w10x").textContent = H.fmt(approx, 2);
  }
  ["w10s", "w10o", "w10p"].forEach((id) => $(id).addEventListener("input", draw));
  $("w10r").addEventListener("click", () => { $("w10s").value = D.sig; $("w10o").value = D.or; $("w10p").value = D.p0; draw(); });
  draw();
};
