window.EXTRA_WIDGETS["ch08_ci"] = function (el, H) {
  // 100 confidence intervals for a mean from the same population (HbA1c change: mean -0.5, SD 0.8)
  const MU = -0.5, SD = 0.8, K = 100;
  // t quantiles (df = n - 1) and z quantiles for 90/95/99% (computed in gen/nums_ch08.py)
  const Q = {
    10: [1.8331, 2.2622, 3.2498], 25: [1.7109, 2.0639, 2.7969], 100: [1.6604, 1.9842, 2.6264],
    z: [1.6449, 1.9600, 2.5758]
  };
  const LV = [90, 95, 99];
  let n = 25, lv = 1, crit = "t", seed = 20, tot = 0, hit = 0;
  const segs = (name, items, cur) => `<div class="seg" role="group" aria-label="${name}">${items.map(([v, lab]) =>
    `<button type="button" data-k="${name}" data-v="${v}" aria-pressed="${String(v) === String(cur)}">${lab}</button>`).join("")}</div>`;
  el.innerHTML = `<p class="wt">직접 바꿔 보기 <span class="pill">시뮬레이션</span></p>
    <p class="wd">참값이 −0.5%p(표준편차 0.8)인 모집단에서 n명씩 뽑아 평균 HbA1c 변화의 신뢰구간을 100번 구합니다. 파란 구간은 참값(세로선)을 포함한 것, 주황 구간은 놓친 것입니다. '다시 뽑기'를 여러 번 누르면 누적 포함 비율이 신뢰수준에 가까워지는지 확인할 수 있습니다.</p>
    <div class="wrow"><span>표본 크기 n</span>${segs("n", [[10, "10"], [25, "25"], [100, "100"]], n)}
      <span>신뢰수준</span>${segs("lv", [[0, "90%"], [1, "95%"], [2, "99%"]], lv)}</div>
    <div class="wrow"><span>임계값</span>${segs("crit", [["t", "t 분포 (n − 1)"], ["z", "정규분포 z"]], crit)}
      <button type="button" class="wbtn" id="c8re">다시 뽑기</button></div>
    <div class="wchart" id="c8chart"></div>
    <div class="stats">
      <div class="stat"><div class="sl">이번 100개 중 참값을 포함한 구간</div><div class="sv" id="c8a"></div></div>
      <div class="stat"><div class="sl">누적 포함 비율 (같은 설정)</div><div class="sv" id="c8b"></div></div>
      <div class="stat"><div class="sl">구간 폭의 평균</div><div class="sv" id="c8c"></div></div>
      <div class="stat"><div class="sl">사용한 임계값</div><div class="sv" id="c8d"></div></div>
    </div>
    <p class="wd" id="c8note" style="margin-top:10px"></p>`;

  function draw() {
    const r = H.mulberry32(seed);
    const q = (crit === "t" ? Q[n] : Q.z)[lv];
    const W = 640, Hh = 400, L = 20, R = 20, T = 26, B = 44;
    const x0 = -1.9, x1 = 0.9;
    const sx = x => L + (x - x0) / (x1 - x0) * (W - L - R);
    const rowH = (Hh - T - B) / K;
    let s = "", cov = 0, wsum = 0;
    for (const tt of [-1.5, -1.0, -0.5, 0, 0.5]) {
      s += `<line x1="${sx(tt).toFixed(1)}" y1="${T}" x2="${sx(tt).toFixed(1)}" y2="${Hh - B}" class="grid"/>`;
      s += `<text x="${sx(tt).toFixed(1)}" y="${Hh - B + 18}" text-anchor="middle" class="tick">${tt.toFixed(1).replace("-", "−")}</text>`;
    }
    s += `<line x1="${L}" y1="${Hh - B}" x2="${W - R}" y2="${Hh - B}" class="axis"/>`;
    for (let i = 0; i < K; i++) {
      let sm = 0, ss = 0; const xs = [];
      for (let j = 0; j < n; j++) { const v = MU + SD * H.gauss(r); xs.push(v); sm += v; }
      const m = sm / n;
      for (const v of xs) ss += (v - m) * (v - m);
      const se = Math.sqrt(ss / (n - 1)) / Math.sqrt(n);
      const lo = m - q * se, hi = m + q * se;
      const ok = lo <= MU && MU <= hi;
      if (ok) cov++;
      wsum += hi - lo;
      const y = T + rowH * (i + 0.5);
      const a = sx(Math.max(lo, x0)), b = sx(Math.min(hi, x1));
      s += `<line x1="${a.toFixed(1)}" y1="${y.toFixed(1)}" x2="${b.toFixed(1)}" y2="${y.toFixed(1)}" class="ln ${ok ? "s1" : "s2"}" stroke-width="${ok ? 1.6 : 2.6}"/>`;
    }
    s += `<line x1="${sx(MU).toFixed(1)}" y1="${T - 6}" x2="${sx(MU).toFixed(1)}" y2="${Hh - B}" class="strongref" stroke-width="1.6"/>`;
    s += `<text x="${sx(MU).toFixed(1)}" y="${T - 10}" text-anchor="middle" class="lbl strong">참값 −0.5</text>`;
    s += `<text x="${(L + W - R) / 2}" y="${Hh - 8}" text-anchor="middle" class="axlab">평균 HbA1c 변화 (%p)의 ${LV[lv]}% 신뢰구간 100개</text>`;
    el.querySelector("#c8chart").innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" class="viz" role="img" aria-label="신뢰구간 100개 시뮬레이션">${s}</svg>`;
    tot += K; hit += cov;
    el.querySelector("#c8a").textContent = `${cov}개`;
    el.querySelector("#c8b").textContent = `${H.fmt(hit / tot * 100, 1)}% (${tot.toLocaleString("ko-KR")}개)`;
    el.querySelector("#c8c").textContent = `${H.fmt(wsum / K, 2)}%p`;
    el.querySelector("#c8d").textContent = `${crit === "t" ? "t" : "z"} = ${H.fmt(q, 3)}`;
    let note;
    if (crit === "z" && n === 10) note = `n = 10인데 t 분포 대신 정규분포 임계값을 쓰면 구간이 너무 좁아져, 여러 번 뽑아 보면 포함 비율이 ${LV[lv]}%보다 뚜렷하게 낮게 나옵니다(95%에서 약 92%). 표준편차를 표본에서 추정한 불확실성을 무시했기 때문입니다.`;
    else if (crit === "z") note = `n이 클수록 t와 z의 차이가 작아져 어느 쪽을 써도 포함 비율이 비슷합니다.`;
    else note = `n을 4배(25 → 100)로 늘리면 구간 폭은 약 절반이 됩니다. 신뢰수준을 높이면 구간이 넓어집니다. 어떤 설정에서든 한 번 뽑은 구간이 참값을 포함했는지는 알 수 없고, '포함하는 비율'만 정해져 있습니다.`;
    el.querySelector("#c8note").textContent = note;
  }
  el.querySelectorAll(".seg button").forEach(btn => btn.addEventListener("click", () => {
    const k = btn.dataset.k, v = btn.dataset.v;
    if (k === "n") n = +v; else if (k === "lv") lv = +v; else crit = v;
    el.querySelectorAll(`.seg button[data-k="${k}"]`).forEach(x => x.setAttribute("aria-pressed", x === btn));
    tot = 0; hit = 0; draw();
  }));
  el.querySelector("#c8re").addEventListener("click", () => { seed = (seed * 7 + 13) % 100003; draw(); });
  draw();
};
