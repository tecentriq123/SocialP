window.EXTRA_WIDGETS["ch12_disp"] = function (el, H) {
  // Poisson(mu) vs negative binomial NB2(mu, alpha): same mean, variance mu + alpha * mu^2
  let mu = 4, alpha = 0.5;
  const AL = [0, 0.25, 0.5, 1, 2];
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">계산기</span></p>
    <p class="wd">평균 건수 μ와 과산포 모수 α를 바꿔 봅니다. 파란 막대는 평균이 μ이고 분산이 μ + αμ²인 음이항 분포, 주황 점은 평균이 같은 포아송 분포(분산 = μ)입니다. α = 0이면 두 분포가 같아집니다.</p>
    <div class="wrow"><label for="c12mu">평균 μ = <b id="c12muv"></b></label><input id="c12mu" type="range" min="0.5" max="12" step="0.5" value="4"></div>
    <div class="wrow"><span>과산포 α</span><div class="seg" role="group" aria-label="과산포 모수">${AL.map(a => `<button type="button" data-a="${a}" aria-pressed="${a === alpha}">${a}</button>`).join("")}</div></div>
    <div class="wchart" id="c12chart"></div>
    <div class="stats">
      <div class="stat"><div class="sl">분산: 포아송 / 음이항</div><div class="sv" id="c12v"></div></div>
      <div class="stat"><div class="sl">분산 ÷ 평균 (음이항)</div><div class="sv" id="c12r"></div></div>
      <div class="stat"><div class="sl">0건일 확률: 포아송 / 음이항</div><div class="sv" id="c12z"></div></div>
      <div class="stat"><div class="sl" id="c12tl"></div><div class="sv" id="c12t"></div></div>
    </div>
    <p class="wd" id="c12note" style="margin-top:10px"></p>`;

  function pois(m, K) { const p = [Math.exp(-m)]; for (let k = 1; k <= K; k++) p.push(p[k - 1] * m / k); return p; }
  function nb(m, a, K) {
    if (a === 0) return pois(m, K);
    const r = 1 / a, q = m / (m + r); const p = [Math.pow(1 + a * m, -r)];
    for (let k = 1; k <= K; k++) p.push(p[k - 1] * (k - 1 + r) / k * q);
    return p;
  }
  const pct = v => v >= 0.001 ? H.fmt(v * 100, 1) + "%" : "&lt; 0.1%";

  function draw() {
    const v = mu + alpha * mu * mu;
    const K = Math.min(40, Math.max(12, Math.ceil(mu + 4 * Math.sqrt(v))));
    const pp = pois(mu, 200), pn = nb(mu, alpha, 200);
    const W = 640, Hh = 270, L = 54, R = 16, T = 18, B = 46;
    const ymax = Math.max(0.1, Math.ceil(Math.max(...pp.slice(0, K + 1), ...pn.slice(0, K + 1)) * 1.12 * 20) / 20);
    const sx = x => L + (x + 0.6) / (K + 1.2) * (W - L - R), sy = y => Hh - B - y / ymax * (Hh - T - B);
    let s = "";
    const step = ymax > 0.5 ? 0.2 : ymax > 0.2 ? 0.1 : 0.05;
    for (let y = 0; y <= ymax + 1e-9; y += step) s += `<line x1="${L}" y1="${sy(y).toFixed(1)}" x2="${W - R}" y2="${sy(y).toFixed(1)}" class="grid"/><text x="${L - 8}" y="${(sy(y) + 4).toFixed(1)}" text-anchor="end" class="tick">${y.toFixed(2)}</text>`;
    s += `<line x1="${L}" y1="${sy(0)}" x2="${W - R}" y2="${sy(0)}" class="axis"/>`;
    const xstep = K > 24 ? 5 : K > 14 ? 2 : 1;
    for (let k = 0; k <= K; k += xstep) s += `<text x="${sx(k).toFixed(1)}" y="${sy(0) + 18}" text-anchor="middle" class="tick">${k}</text>`;
    s += `<text x="${(L + W - R) / 2}" y="${Hh - 8}" text-anchor="middle" class="axlab">건수 k</text>`;
    s += `<text x="14" y="${(T + Hh - B) / 2}" text-anchor="middle" transform="rotate(-90 14 ${(T + Hh - B) / 2})" class="axlab">P(X = k)</text>`;
    const bw = (sx(1) - sx(0)) * 0.62;
    for (let k = 0; k <= K; k++) {
      const x = sx(k) - bw / 2, y = sy(Math.min(pn[k], ymax));
      s += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${(sy(0) - y).toFixed(1)}" class="f1"/>`;
    }
    for (let k = 0; k <= K; k++) s += `<circle cx="${sx(k).toFixed(1)}" cy="${sy(Math.min(pp[k], ymax)).toFixed(1)}" r="4" class="pt f2"/>`;
    s += `<line x1="${sx(mu).toFixed(1)}" y1="${T}" x2="${sx(mu).toFixed(1)}" y2="${sy(0)}" class="ref" stroke-dasharray="4 4"/>`;
    s += `<text x="${(sx(mu) + 6).toFixed(1)}" y="${T + 10}" class="lbl mute small">평균 μ</text>`;
    s += `<rect x="${W - R - 200}" y="${T + 22}" width="12" height="12" rx="2" class="f1"/><text x="${W - R - 182}" y="${T + 32}" class="lbl small">음이항 (분산 μ + αμ²)</text>`;
    s += `<circle cx="${W - R - 194}" cy="${T + 50}" r="4" class="pt f2"/><text x="${W - R - 182}" y="${T + 54}" class="lbl small">포아송 (분산 μ)</text>`;
    el.querySelector("#c12chart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="포아송 분포와 음이항 분포 비교">${s}</svg>`;
    el.querySelector("#c12muv").textContent = H.fmt(mu, 1);
    el.querySelector("#c12v").textContent = `${H.fmt(mu, 1)} / ${H.fmt(v, 1)}`;
    el.querySelector("#c12r").textContent = H.fmt(1 + alpha * mu, 2) + "배";
    el.querySelector("#c12z").innerHTML = `${pct(pp[0])} / ${pct(pn[0])}`;
    const cut = Math.ceil(mu + 2 * Math.sqrt(mu));
    const tail = arr => 1 - arr.slice(0, cut).reduce((a, b) => a + b, 0);
    el.querySelector("#c12tl").textContent = `${cut}건 이상일 확률: 포아송 / 음이항`;
    el.querySelector("#c12t").innerHTML = `${pct(Math.max(0, tail(pp)))} / ${pct(Math.max(0, tail(pn)))}`;
    el.querySelector("#c12note").textContent = alpha === 0
      ? "α = 0이면 모든 환자의 발생률이 같다고 보는 포아송 분포 그대로입니다. 평균과 분산이 같습니다."
      : `평균은 그대로 ${H.fmt(mu, 1)}건인데 분산은 포아송의 ${H.fmt(1 + alpha * mu, 2)}배입니다. 사건이 전혀 없는 사람(0건)과 아주 많은 사람(오른쪽 꼬리)이 함께 늘고, 가운데 값은 줄어듭니다. 환자마다 발생률이 다를 때 생기는 모양입니다.`;
  }
  el.querySelector("#c12mu").addEventListener("input", e => { mu = +e.target.value; draw(); });
  el.querySelectorAll(".seg button").forEach(b => b.addEventListener("click", () => {
    alpha = +b.dataset.a;
    el.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", x === b));
    draw();
  }));
  draw();
};
