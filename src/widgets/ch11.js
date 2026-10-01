window.EXTRA_WIDGETS["ch11_hrrr"] = function (el, H) {
  // control group: Weibull S_B(t) = exp(-(t/b)^k) with median m -> b = m / (ln 2)^(1/k); treatment S_A = S_B^HR (PH)
  let hr = 0.73, k = 1.0, med = 20, tq = 36;
  const shapes = [[0.6, "감소 (k = 0.6)"], [1.0, "일정 (k = 1)"], [1.6, "증가 (k = 1.6)"]];
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">계산기</span></p>
    <p class="wd">대조군(약물 B)의 생존시간이 Weibull 분포를 따르고, 위험비가 전 기간 일정하다고(비례위험) 가정합니다. 이때 치료군의 생존곡선은 S<sub>A</sub>(t) = S<sub>B</sub>(t)<sup>HR</sup>입니다. 위험비, 대조군 위험률의 모양, 대조군 중앙생존기간, 비교 시점을 바꿔 가며 같은 위험비가 시점별 누적 사망위험의 비(상대위험도)와 중앙생존기간의 비로 어떻게 바뀌는지 봅니다.</p>
    <div class="wrow"><label for="w11hr">위험비 HR = <b id="w11hrv"></b></label><input id="w11hr" type="range" min="-1.2" max="1.2" step="0.01" value="${Math.log(hr).toFixed(2)}"></div>
    <div class="wrow"><span>대조군 위험률의 모양</span><div class="seg" role="group" aria-label="대조군 위험률의 모양">${shapes.map(([v, lab]) => `<button type="button" data-k="${v}" aria-pressed="${v === k}">${lab}</button>`).join("")}</div></div>
    <div class="wrow"><label for="w11m">대조군 중앙생존기간 = <b id="w11mv"></b>개월</label><input id="w11m" type="range" min="6" max="60" step="1" value="${med}"></div>
    <div class="wrow"><label for="w11t">비교 시점 t = <b id="w11tv"></b>개월</label><input id="w11t" type="range" min="3" max="60" step="1" value="${tq}"></div>
    <div class="wchart" id="w11chart"></div>
    <div class="stats">
      <div class="stat"><div class="sl">대조군 t까지 누적 사망위험</div><div class="sv" id="w11r0"></div></div>
      <div class="stat"><div class="sl">치료군 t까지 누적 사망위험</div><div class="sv" id="w11r1"></div></div>
      <div class="stat"><div class="sl">상대위험도 RR(t)</div><div class="sv" id="w11rr"></div></div>
      <div class="stat"><div class="sl">위험차 (치료 − 대조)</div><div class="sv" id="w11rd"></div></div>
      <div class="stat"><div class="sl">중앙생존기간의 비 (치료 ÷ 대조)</div><div class="sv" id="w11mr"></div></div>
    </div>
    <p class="wd" id="w11note" style="margin-top:10px"></p>`;

  const S0 = (t, b) => Math.exp(-Math.pow(t / b, k));
  function draw() {
    const b = med / Math.pow(Math.LN2, 1 / k);
    const W = 640, Hh = 300, L = 60, R = 20, Tp = 16, B = 50;
    const sx = x => L + x / 60 * (W - L - R), sy = y => Hh - B - y * (Hh - Tp - B);
    let s = "";
    for (const y of [0, 0.2, 0.4, 0.6, 0.8, 1]) s += `<line x1="${L}" y1="${sy(y)}" x2="${W - R}" y2="${sy(y)}" class="grid"/><text x="${L - 8}" y="${sy(y) + 4}" text-anchor="end" class="tick">${Math.round(y * 100)}%</text>`;
    s += `<line x1="${L}" y1="${sy(0)}" x2="${W - R}" y2="${sy(0)}" class="axis"/>`;
    for (const x of [0, 12, 24, 36, 48, 60]) s += `<text x="${sx(x)}" y="${sy(0) + 18}" text-anchor="middle" class="tick">${x}</text>`;
    s += `<text x="${(L + W - R) / 2}" y="${Hh - 8}" text-anchor="middle" class="axlab">개월</text>`;
    s += `<text x="14" y="${(Tp + Hh - B) / 2}" text-anchor="middle" transform="rotate(-90 14 ${(Tp + Hh - B) / 2})" class="axlab">누적 사망위험 1 − S(t)</text>`;
    let d0 = "", d1 = "";
    for (let i = 0; i <= 240; i++) {
      const x = i / 4, s0 = S0(x, b), s1 = Math.pow(s0, hr);
      d0 += (i ? " L" : "M") + sx(x).toFixed(1) + "," + sy(1 - s0).toFixed(1);
      d1 += (i ? " L" : "M") + sx(x).toFixed(1) + "," + sy(1 - s1).toFixed(1);
    }
    s += `<path d="${d0}" class="ln s2" stroke-width="2.4" fill="none"/><path d="${d1}" class="ln s1" stroke-width="2.4" fill="none"/>`;
    const s0t = S0(tq, b), s1t = Math.pow(s0t, hr), r0 = 1 - s0t, r1 = 1 - s1t;
    s += `<line x1="${sx(tq)}" y1="${sy(0)}" x2="${sx(tq)}" y2="${sy(1)}" class="ref" stroke-dasharray="4 4"/>`;
    s += `<circle cx="${sx(tq)}" cy="${sy(r0)}" r="4.5" class="pt f2"/><circle cx="${sx(tq)}" cy="${sy(r1)}" r="4.5" class="pt f1"/>`;
    s += `<line x1="${L + 14}" y1="${Tp + 10}" x2="${L + 36}" y2="${Tp + 10}" class="ln s2" stroke-width="2.4"/><text x="${L + 42}" y="${Tp + 14}" class="lbl">대조군 (약물 B)</text>`;
    s += `<line x1="${L + 14}" y1="${Tp + 30}" x2="${L + 36}" y2="${Tp + 30}" class="ln s1" stroke-width="2.4"/><text x="${L + 42}" y="${Tp + 34}" class="lbl">치료군 (약물 A)</text>`;
    el.querySelector("#w11chart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="두 군의 누적 사망위험 곡선">${s}</svg>`;
    const rr = r1 / r0, mr = Math.pow(hr, -1 / k);
    el.querySelector("#w11hrv").textContent = H.fmt(hr, 2);
    el.querySelector("#w11mv").textContent = med;
    el.querySelector("#w11tv").textContent = tq;
    el.querySelector("#w11r0").textContent = H.fmt(r0 * 100, 1) + "%";
    el.querySelector("#w11r1").textContent = H.fmt(r1 * 100, 1) + "%";
    el.querySelector("#w11rr").textContent = H.fmt(rr, 2);
    el.querySelector("#w11rd").textContent = (r1 - r0 >= 0 ? "+" : "−") + H.fmt(Math.abs(r1 - r0) * 100, 1) + "%p";
    el.querySelector("#w11mr").textContent = H.fmt(mr, 2) + "배";
    let note;
    if (Math.abs(hr - 1) < 0.005) note = "위험비가 1이면 두 곡선이 같습니다.";
    else if (hr < 1) note = `‘1 − HR’로 읽으면 ${H.fmt((1 - hr) * 100, 0)}% 감소이지만, ${tq}개월까지의 누적 사망위험은 ${H.fmt((1 - rr) * 100, 1)}% 줄어듭니다. 중앙생존기간은 ${H.fmt(mr, 2)}배로, 1/HR = ${H.fmt(1 / hr, 2)}배와 같아지는 것은 위험률이 일정할 때(k = 1)뿐입니다.`;
    else note = `‘HR − 1’로 읽으면 ${H.fmt((hr - 1) * 100, 0)}% 증가이지만, ${tq}개월까지의 누적 사망위험은 ${H.fmt((rr - 1) * 100, 1)}% 늘어납니다. 대조군의 누적위험이 클수록 상대위험도는 1 쪽으로 줄어듭니다.`;
    if (r0 < 0.1 && r1 < 0.1) note += " 두 군의 누적위험이 모두 10% 미만으로 작을 때는 상대위험도가 위험비와 거의 같습니다.";
    el.querySelector("#w11note").textContent = note;
  }
  el.querySelector("#w11hr").addEventListener("input", e => { hr = Math.exp(+e.target.value); draw(); });
  el.querySelector("#w11m").addEventListener("input", e => { med = +e.target.value; draw(); });
  el.querySelector("#w11t").addEventListener("input", e => { tq = +e.target.value; draw(); });
  el.querySelectorAll(".seg button").forEach(btn => btn.addEventListener("click", () => {
    k = +btn.dataset.k;
    el.querySelectorAll(".seg button").forEach(x => x.setAttribute("aria-pressed", x === btn));
    draw();
  }));
  draw();
};
