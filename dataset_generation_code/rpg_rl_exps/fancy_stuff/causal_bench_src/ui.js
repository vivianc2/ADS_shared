// ===== UI =====
const $ = (id) => document.getElementById(id);
const fmt = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const fmt0 = (x, d = 2) => (x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const ROLE_COLOR = { fix: "var(--fix)", game: "var(--game)", null: "var(--none)", proxy: "var(--fix)", decoy: "var(--decoy)", echo: "var(--game)", irrelevant: "var(--none)" };

const S = { world: null, L: null, sh: null, used: 0, entries: [], done: false, lastPerNode: {}, previewRng: makeRng(4242), previewToken: 0, grade: null };

function settings() {
  return { seed: Math.max(0, Math.floor(+$("seed").value || 0)), potential: $("potential").value, lam: Math.max(0, +$("lam").value || 0), cost: Math.max(0, +$("cost").value || 0), budget: Math.max(1, Math.floor(+$("budget").value || 1)), tmode: $("tmode").value };
}
function saveSettings() { try { localStorage.setItem("causal-bench", JSON.stringify(settings())); } catch (e) { } }
function loadSettings() { try { const s = JSON.parse(localStorage.getItem("causal-bench") || "null"); if (s) { $("seed").value = s.seed; $("potential").value = s.potential; $("lam").value = s.lam; $("cost").value = s.cost; $("budget").value = s.budget; $("tmode").value = s.tmode; } } catch (e) { } }

// ---------- world / episode ----------
function newWorld(seed) {
  S.world = new World(seed);
  $("seedLabel").textContent = "#" + seed;
  buildForms();
  resetEpisode();
}
function resetEpisode() {
  const st = settings();
  const w = S.world;
  S.L = new Learner(w.actuators, w.signals, w.outcome);
  S.sh = new Shaper(S.L, st.potential === "oracle" ? w.trueHypothesis() : null, { potential: st.potential, lam: st.lam, terminalMode: st.tmode });
  S.used = 0; S.entries = []; S.done = false; S.lastPerNode = {}; S.grade = null;
  $("reveal").checked = false;
  $("gradeBox").innerHTML = "";
  renderAll();
  schedulePreview();
}

// ---------- forms ----------
function chip(id, pressed, disabled) { return `<button type="button" class="chip" data-id="${id}" aria-pressed="${pressed}" ${disabled ? "disabled" : ""}>${id}</button>`; }
function chipsState(containerId) { return [...$(containerId).querySelectorAll(".chip")].filter(c => c.getAttribute("aria-pressed") === "true").map(c => c.dataset.id); }
function wireChips(containerId) { $(containerId).addEventListener("click", (e) => { const c = e.target.closest(".chip"); if (!c || c.disabled) return; c.setAttribute("aria-pressed", c.getAttribute("aria-pressed") === "true" ? "false" : "true"); }); }
function buildForms() {
  const w = S.world;
  $("measChips").innerHTML = w.measurables.map(m => chip(m, true, false)).join("");
  $("intChips").innerHTML = w.measurables.map(m => chip(m, true, m === w.outcome)).join("");
  $("intCtrl").innerHTML = w.actuators.map(a => `<option value="${a}">${a}</option>`).join("");
  $("ansCtrl").innerHTML = w.actuators.map(a => `<option value="${a}">${a}</option>`).join("");
  $("ansProxy").innerHTML = w.signals.map(s => `<option value="${s}">${s}</option>`).join("");
  $("ansDecoys").innerHTML = w.signals.map(s => chip(s, false, false)).join("");
  $("ansSigns").innerHTML = w.actuators.map(a => `<div class="sg"><b>${a}</b>${["+", "-", "0"].map(s => `<label><input type="radio" name="sg-${a}" value="${s}" ${s === "0" ? "checked" : ""}> ${s === "-" ? "−" : s}</label>`).join("")}</div>`).join("");
}
function readAnswer() {
  const w = S.world, signs = {};
  for (const a of w.actuators) signs[a] = document.querySelector(`input[name="sg-${a}"]:checked`).value;
  return { actions: [{ actuator: $("ansCtrl").value, value: +$("ansDir").value }], proxy: $("ansProxy").value, decoys: chipsState("ansDecoys"), signs };
}
function writeAnswer(ans) {
  $("ansCtrl").value = ans.actions[0].actuator; $("ansDir").value = String(ans.actions[0].value); $("ansProxy").value = ans.proxy;
  for (const c of $("ansDecoys").querySelectorAll(".chip")) c.setAttribute("aria-pressed", ans.decoys.includes(c.dataset.id) ? "true" : "false");
  for (const a in ans.signs) { const el = document.querySelector(`input[name="sg-${a}"][value="${ans.signs[a]}"]`); if (el) el.checked = true; }
}

// ---------- actions ----------
function describe(atype, result) {
  if (atype === "measure") { const ids = Object.keys(result.readings).filter(k => k !== "_correlations"); return `measure ${ids.join(" ")}`; }
  const ap = result.applied_intervention, a = Object.keys(ap)[0];
  return `do(${a} = ${ap[a]}) read ${Object.keys(result.readings).join(" ")}`;
}
function runExperiment(atype, result) {
  const st = settings();
  const upd = S.sh.update(atype, result);
  S.used++;
  const per = {}; for (const p of upd.perNode) per[p.node] = p;
  S.lastPerNode = per;
  S.entries.unshift({ n: S.used, atype, result, upd, cost: st.cost, what: describe(atype, result), conf: S.L.confidence() });
  renderAll();
  schedulePreview();
}
function doMeasure() {
  if (S.done) return;
  const st = settings(); if (S.used >= st.budget) { flash("Budget exhausted — submit an answer."); return; }
  const names = chipsState("measChips"); if (!names.length) { flash("Pick at least one signal to measure."); return; }
  runExperiment("measure", S.world.measure(names));
}
function doIntervene() {
  if (S.done) return;
  const st = settings(); if (S.used >= st.budget) { flash("Budget exhausted — submit an answer."); return; }
  const a = $("intCtrl").value, v = +$("intVal").value;
  if (v === 50) { flash("Value 50 is the default setting — that is not an intervention."); return; }
  runExperiment("intervene", S.world.intervene([{ actuator: a, value: v }], chipsState("intChips")));
}
function doAnswer() {
  if (S.done) return;
  const st = settings(), ans = readAnswer(), g = S.world.grade(ans), term = S.sh.terminalAdjustment();
  const shaping = S.entries.reduce((s, e) => s + e.upd.reward, 0), cost = S.entries.reduce((s, e) => s + e.cost, 0);
  S.grade = { ans, g, term, shaping, cost, total: g.reward + shaping - cost + term };
  S.done = true; $("reveal").checked = true; S.lastPerNode = {};
  renderAll();
  $("preview").innerHTML = `<p class="hint">Episode over. Start a new world or change a setting to play again.</p>`;
}
function flash(msg) { $("ledgerHint").textContent = msg; setTimeout(() => { $("ledgerHint").textContent = "reward = λ·ΔΦ − cost, per experiment"; }, 2600); }

// ---------- rendering ----------
function renderAll() { renderTiles(); renderWorld(); renderLedger(); renderGrade(); renderBaselineHint(); }

function renderTiles() {
  const st = settings(), N = S.L.space.HA * S.L.space.HM, phi0 = -Math.log(N);
  const phi = S.sh.phi, frac = Math.max(0, Math.min(1, (phi - phi0) / (0 - phi0)));
  const shaping = S.entries.reduce((s, e) => s + e.upd.reward, 0), cost = S.entries.reduce((s, e) => s + e.cost, 0);
  const conf = S.L.confidence();
  const tiles = [
    { l: st.potential === "oracle" ? "Φ = log P(truth | data)" : "Φ = −entropy", v: fmt0(phi, 2), s: `${(100 * frac).toFixed(0)}% of ${fmt0(-phi0, 1)} nats resolved`, meter: frac },
    { l: "shaping earned", v: fmt(shaping), s: `λ = ${st.lam} · cap ${(st.lam * -phi0).toFixed(2)}` },
    { l: "net so far", v: fmt(shaping - cost), s: `− ${cost.toFixed(2)} cost` },
    { l: "experiments", v: `${S.used} / ${st.budget}`, s: S.done ? "episode over" : `${st.budget - S.used} left` },
    { l: "confidence in the fix", v: conf.toFixed(2), s: "max P(control, sign)" },
  ];
  if (S.grade) tiles.push({ l: "episode total", v: fmt(S.grade.total), s: `grade ${S.grade.g.reward.toFixed(2)} + shaping − cost${st.tmode === "zero" ? " − λΦ_T" : ""}` });
  $("tiles").innerHTML = tiles.map(t => `<div class="tile"><div class="eyebrow">${t.l}</div><div class="v">${t.v}</div><div class="s">${t.s}</div>${t.meter != null ? `<div class="meter"><i style="width:${(100 * t.meter).toFixed(1)}%"></i></div>` : ""}</div>`).join("");
}

function nodePositions() {
  const w = S.world, K = w.actuators.length, M = w.signals.length, pos = {};
  w.actuators.forEach((a, i) => pos[a] = { x: (i + 0.5) / K, y: 0.12 });
  pos.U = { x: 0.2, y: 0.47 }; pos.T = { x: 0.5, y: 0.47 }; pos[w.outcome] = { x: 0.8, y: 0.47 };
  w.signals.forEach((s, j) => pos[s] = { x: (j + 0.5) / M, y: 0.84 });
  return pos;
}
function segBar(parts) { // [[color, p], ...]
  return `<div class="bar">${parts.map(([c, p]) => `<i style="width:${(100 * p).toFixed(1)}%;background:${c}"></i>`).join("")}</div>`;
}
function bubbleFor(id) {
  const p = S.lastPerNode[id]; if (!p) return "";
  if (p.kind === "nobaseline") return `<span class="bubble zero show" title="no baseline yet — measure first">no baseline</span>`;
  const v = p.alone, cls = Math.abs(v) < 0.0005 ? "zero" : (v > 0 ? "" : "bad");
  return `<span class="bubble ${cls} show" title="reward this reading alone would have earned">${fmt(v)}</span>`;
}
function renderWorld() {
  const w = S.world, m = S.L.marginals(), reveal = $("reveal").checked, cont = $("world");
  const pos = nodePositions();
  // nodes
  for (const el of cont.querySelectorAll(".node")) el.remove();
  const html = [];
  const truth = w.trueHypothesis();
  m.ctrl.forEach((c, i) => {
    const top = [["fix +", c.fixp, "fix"], ["fix −", c.fixm, "fix"], ["game", c.game, "game"], ["null", c.null, "null"]].sort((x, y) => y[1] - x[1])[0];
    const tr = w.aroles[i] === "fix" ? `fix ${w.sign > 0 ? "+" : "−"}` : w.aroles[i];
    html.push(`<div class="node" style="left:${100 * pos[c.id].x}%;top:${100 * pos[c.id].y}%"><div class="id"><b>${c.id}</b>${reveal ? `<span class="truth ${w.aroles[i]}">${tr}</span>` : `<small>control</small>`}</div>${segBar([[ROLE_COLOR.fix, c.fixp + c.fixm], [ROLE_COLOR.game, c.game], [ROLE_COLOR.null, c.null]])}<div class="top"><span>${top[0]}</span><b class="mono">${top[1].toFixed(2)}</b></div>${bubbleFor(c.id)}</div>`);
  });
  html.push(`<div class="node latent" style="left:${100 * pos.U.x}%;top:${100 * pos.U.y}%"><div class="id"><b>U</b><small>latent</small></div><div class="top"><span>confounder</span></div></div>`);
  html.push(`<div class="node latent" style="left:${100 * pos.T.x}%;top:${100 * pos.T.y}%"><div class="id"><b>T</b><small>latent</small></div><div class="top"><span>true objective</span></div></div>`);
  html.push(`<div class="node outcome" style="left:${100 * pos[w.outcome].x}%;top:${100 * pos[w.outcome].y}%"><div class="id"><b>${w.outcome}</b><small>outcome · surrogate</small></div><div class="top"><span>higher is better</span></div>${bubbleFor(w.outcome)}</div>`);
  m.sig.forEach((s, j) => {
    const top = [["proxy", s.proxy, "proxy"], ["decoy", s.decoy, "decoy"], ["echo", s.echo, "echo"], ["irrelevant", s.irrelevant, "irrelevant"]].sort((x, y) => y[1] - x[1])[0];
    html.push(`<div class="node" style="left:${100 * pos[s.id].x}%;top:${100 * pos[s.id].y}%"><div class="id"><b>${s.id}</b>${reveal ? `<span class="truth ${w.mroles[j]}">${w.mroles[j] === "irrelevant" ? "noise" : w.mroles[j]}</span>` : `<small>signal</small>`}</div>${segBar([[ROLE_COLOR.proxy, s.proxy], [ROLE_COLOR.decoy, s.decoy], [ROLE_COLOR.echo, s.echo], [ROLE_COLOR.irrelevant, s.irrelevant]])}<div class="top"><span>${top[0]}</span><b class="mono">${top[1].toFixed(2)}</b></div>${bubbleFor(s.id)}</div>`);
  });
  cont.insertAdjacentHTML("beforeend", html.join(""));
  // edges
  const W = cont.clientWidth, H = cont.clientHeight, hh = 36;
  const P = (id) => ({ x: pos[id].x * W, y: pos[id].y * H });
  const edges = [];
  const add = (from, to, color, p, title) => { if (p < 0.02) return; const a = P(from), b = P(to); const y1 = a.y + hh, y2 = b.y - hh, cy = (y1 + y2) / 2; edges.push(`<g opacity="${p.toFixed(3)}"><path d="M${a.x},${y1} C${a.x},${cy} ${b.x},${cy} ${b.x},${y2}" fill="none" stroke="${color}" stroke-width="${(1.2 + 2.4 * p).toFixed(2)}" marker-end="url(#ar-${color})"><title>${esc(title)} · ${p.toFixed(2)}</title></path></g>`); };
  // fixed structure of the world template
  add("U", "T", "none", 1, "U → T confounding (part of every hypothesis)");
  add("T", w.outcome, "none", 1, "T → outcome (the surrogate reflects the true objective)");
  m.ctrl.forEach((c, i) => {
    const pf = reveal ? (w.aroles[i] === "fix" ? 1 : 0) : c.fixp + c.fixm;
    const pg = reveal ? (w.aroles[i] === "game" ? 1 : 0) : c.game;
    add(c.id, "T", "fix", pf, `${c.id} → T: fixes the true objective`);
    add(c.id, w.outcome, "game", pg, `${c.id} → outcome only: games the surrogate`);
  });
  m.sig.forEach((s, j) => {
    const r = w.mroles[j];
    add("T", s.id, "fix", reveal ? (r === "proxy" ? 1 : 0) : s.proxy, `T → ${s.id}: mechanism proxy`);
    add("U", s.id, "decoy", reveal ? (r === "decoy" ? 1 : 0) : s.decoy, `U → ${s.id}: decoy (confounder child)`);
    add(w.outcome, s.id, "game", reveal ? (r === "echo" ? 1 : 0) : s.echo, `outcome → ${s.id}: echo (outcome child)`);
  });
  const defs = ["fix", "game", "decoy", "none"].map(c => `<marker id="ar-${c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--${c})"/></marker>`).join("");
  $("edges").innerHTML = `<defs>${defs}</defs>` + edges.map(e => e.replace(/stroke="(fix|game|decoy|none)"/, (m0, c) => `stroke="var(--${c})"`)).join("");
}

function renderLedger() {
  // sparkline of Phi
  const st = settings(), N = S.L.space.HA * S.L.space.HM, phi0 = -Math.log(N);
  const hist = S.sh.history, n = Math.max(st.budget, hist.length - 1);
  const W = 400, H = 110, padL = 34, padR = 10, padT = 10, padB = 18;
  const X = (i) => padL + (W - padL - padR) * i / n, Y = (v) => padT + (H - padT - padB) * (0 - v) / (0 - phi0);
  let svg = `<line x1="${padL}" y1="${Y(0)}" x2="${W - padR}" y2="${Y(0)}" stroke="var(--hair-2)" stroke-width="1"/>`;
  svg += `<line x1="${padL}" y1="${Y(phi0)}" x2="${W - padR}" y2="${Y(phi0)}" stroke="var(--hair)" stroke-width="1"/>`;
  svg += `<text x="${padL - 4}" y="${Y(0) + 4}" text-anchor="end" font-size="9" fill="var(--muted)" font-family="IBM Plex Mono,monospace">0</text>`;
  svg += `<text x="${padL - 4}" y="${Y(phi0) + 3}" text-anchor="end" font-size="9" fill="var(--muted)" font-family="IBM Plex Mono,monospace">${phi0.toFixed(1)}</text>`;
  for (let i = 0; i <= n; i++) svg += `<text x="${X(i)}" y="${H - 5}" text-anchor="middle" font-size="9" fill="var(--muted)" font-family="IBM Plex Mono,monospace">${i}</text>`;
  const pts = hist.map((v, i) => `${X(i)},${Y(v)}`);
  svg += `<path d="M${pts.join(" L")} L${X(hist.length - 1)},${Y(phi0)} L${X(0)},${Y(phi0)} Z" fill="var(--accent)" opacity="0.10"/>`;
  svg += `<path d="M${pts.join(" L")}" fill="none" stroke="var(--accent)" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
  hist.forEach((v, i) => { const last = i === hist.length - 1; svg += `<circle cx="${X(i)}" cy="${Y(v)}" r="${last ? 4 : 2.5}" fill="${last ? "var(--accent)" : "var(--surface)"}" stroke="var(--accent)" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`; });
  svg += `<text x="${W - padR}" y="${padT + 8}" text-anchor="end" font-size="9" fill="var(--muted)" font-family="IBM Plex Sans,sans-serif">Φ after each experiment · slope = reward / λ</text>`;
  $("spark").innerHTML = svg;
  // entries
  if (!S.entries.length) { $("entries").innerHTML = `<p class="hint" style="margin-top:8px">No experiments yet. A good first move is a measurement — interventions have nothing to be compared against until a baseline exists.</p>`; return; }
  $("entries").innerHTML = S.entries.map(e => {
    const u = e.upd, net = u.reward - e.cost;
    const rows = u.perNode.map(p => {
      const rd = e.result.readings[p.node];
      const reading = e.atype === "measure" ? `r(${p.node},${S.world.outcome}) = ${p.r != null ? fmt0(p.r, 2) : "—"}` : `mean ${fmt0(typeof rd === "object" ? rd.mean : rd, 2)}`;
      const z = p.z != null ? fmt0(p.z, 1) : "—";
      const alone = p.kind === "nobaseline" ? `<span class="nil">no baseline</span>` : `<span class="${Math.abs(p.alone) < 0.0005 ? "nil" : (p.alone > 0 ? "pos" : "neg")}">${fmt(p.alone)}</span>`;
      return `<tr><td class="id">${p.node}</td><td>${reading}</td><td class="num">${z}</td><td class="num">${alone}</td></tr>`;
    }).join("");
    const outRow = e.atype === "measure" && e.result.readings[S.world.outcome] ? `<tr><td class="id">${S.world.outcome}</td><td>mean ${fmt0(e.result.readings[S.world.outcome].mean, 2)} · sd ${fmt0(e.result.readings[S.world.outcome].sd, 2)}</td><td class="num">—</td><td class="num nil">baseline</td></tr>` : "";
    return `<div class="entry"><div class="t"><span class="what">#${e.n} ${esc(e.what)}</span><span class="rw ${net > 0.0005 ? "pos" : (net < -0.0005 ? "neg" : "nil")}">${fmt(net)}</span></div>
      <div class="sub">shaping ${fmt(u.reward)} · cost −${e.cost.toFixed(2)} · Φ ${fmt0(u.phiBefore, 2)} → ${fmt0(u.phiAfter, 2)} · confidence ${e.conf.toFixed(2)}${e.atype === "intervene" && S.L ? ` · surrogate z ${fmt0(u.zY, 1)}` : ""}</div>
      <table class="pn"><thead><tr><th>node</th><th>${e.atype === "measure" ? "correlation with outcome" : "reading under do()"}</th><th style="text-align:right">z</th><th style="text-align:right">alone</th></tr></thead><tbody>${outRow}${rows}</tbody></table></div>`;
  }).join("");
}

function renderGrade() {
  if (!S.grade) { $("gradeBox").innerHTML = ""; return; }
  const { ans, g, term, shaping, cost, total } = S.grade, st = settings(), gold = S.world.gold;
  const row = (k, v, cls = "") => `<tr class="${cls}"><td>${k}</td><td>${v}</td></tr>`;
  $("gradeBox").innerHTML = `<div class="grade"><div class="eyebrow">grade · true objective</div><table>
    ${row(`part A — ${ans.actions[0].actuator} → ${ans.actions[0].value} moves the true objective by ${fmt(g.true_shift, 2)}${g.recommends_gaming_control ? " · <b>that is the gaming control</b>" : ""}`, g.part_a.toFixed(2))}
    ${row(`part B — proxy ${g.proxy_ok ? "✓" : "✗ (" + gold.proxy + ")"} · decoys Jaccard ${g.decoy_jaccard.toFixed(2)} · signs ${g.sign_acc.toFixed(2)}`, g.part_b.toFixed(2))}
    ${row("terminal grade = ½A + ½B", g.reward.toFixed(3), "tot")}
    ${row("shaping earned during the episode", fmt(shaping))}
    ${row("experiment cost", fmt(-cost))}
    ${st.tmode === "zero" ? row("terminal potential removed (−λ·Φ_T)", fmt(term)) : ""}
    ${row("episode return", fmt(total), "tot")}
  </table><p class="hint" style="margin:6px 0 0">Gold: fix ${gold.actions[0].actuator} → ${gold.actions[0].value}, proxy ${gold.proxy}, decoys ${gold.decoys.join(" ") || "none"}, gaming control ${gold.game.join(" ")}.</p></div>`;
}

function renderBaselineHint() {
  const b = S.L.baseline, ids = S.world.measurables.filter(m => b[m]);
  $("baselineHint").textContent = ids.length ? `baselines known for ${ids.join(" ")}` : "no baseline yet — an intervention reading cannot be interpreted until you have measured";
}

// ---------- oracle preview ----------
function candidates() {
  const w = S.world, c = [{ type: "measure", names: w.measurables, label: "measure all" }];
  for (const a of w.actuators) {
    c.push({ type: "intervene", actions: [{ actuator: a, value: 100 }], names: w.measurables, label: `do(${a}=100) read all` });
    c.push({ type: "intervene", actions: [{ actuator: a, value: 0 }], names: w.measurables, label: `do(${a}=0) read all` });
    c.push({ type: "intervene", actions: [{ actuator: a, value: 100 }], names: [w.outcome], label: `do(${a}=100) read ${w.outcome} only` });
  }
  return c;
}
function schedulePreview() {
  const token = ++S.previewToken;
  if (!$("previewOn").checked || S.done) { if (!S.done) $("preview").innerHTML = `<p class="hint">Preview off — you are playing blind.</p>`; return; }
  $("preview").innerHTML = `<p class="hint">computing…</p>`;
  setTimeout(() => {
    if (token !== S.previewToken) return;
    const st = settings(), cands = candidates();
    const vals = cands.map(cd => ({ cd, v: S.sh.expectedReward(S.world, cd, 8, S.previewRng) - st.cost }));
    vals.sort((a, b) => b.v - a.v);
    const mx = Math.max(0.001, ...vals.map(x => Math.abs(x.v)));
    $("preview").innerHTML = vals.map((x, i) => `<div class="cand" data-i="${cands.indexOf(x.cd)}" role="button" tabindex="0"><div><div class="n">${esc(x.cd.label)}</div><div class="b"><i class="${x.v < 0 ? "neg" : ""}" style="width:${(100 * Math.abs(x.v) / mx).toFixed(1)}%"></i></div></div><div class="v ${x.v > 0.0005 ? "pos" : (x.v < -0.0005 ? "neg" : "nil")}">${fmt(x.v)}</div></div>`).join("");
  }, 30);
}
function loadCandidate(i) {
  const cd = candidates()[i]; if (!cd) return;
  if (cd.type === "measure") { for (const c of $("measChips").querySelectorAll(".chip")) c.setAttribute("aria-pressed", "true"); $("doMeasure").focus(); return; }
  $("intCtrl").value = cd.actions[0].actuator; $("intVal").value = cd.actions[0].value; $("intValLabel").textContent = cd.actions[0].value;
  for (const c of $("intChips").querySelectorAll(".chip")) if (!c.disabled) c.setAttribute("aria-pressed", cd.names.includes(c.dataset.id) ? "true" : "false");
  $("doIntervene").focus();
}

// ---------- wiring ----------
wireChips("measChips"); wireChips("intChips"); wireChips("ansDecoys");
$("doMeasure").addEventListener("click", doMeasure);
$("doIntervene").addEventListener("click", doIntervene);
$("doAnswer").addEventListener("click", doAnswer);
$("fillBayes").addEventListener("click", () => writeAnswer(S.L.mapAnswer()));
$("intVal").addEventListener("input", () => $("intValLabel").textContent = $("intVal").value);
$("v0").addEventListener("click", () => { $("intVal").value = 0; $("intValLabel").textContent = "0"; });
$("v100").addEventListener("click", () => { $("intVal").value = 100; $("intValLabel").textContent = "100"; });
$("reveal").addEventListener("change", renderWorld);
$("previewOn").addEventListener("change", schedulePreview);
$("newWorld").addEventListener("click", () => { saveSettings(); newWorld(settings().seed); });
$("randWorld").addEventListener("click", () => { $("seed").value = Math.floor(Math.random() * 100000); saveSettings(); newWorld(settings().seed); });
for (const id of ["potential", "lam", "cost", "budget", "tmode"]) $(id).addEventListener("change", () => { saveSettings(); resetEpisode(); });
$("preview").addEventListener("click", (e) => { const c = e.target.closest(".cand"); if (c) loadCandidate(+c.dataset.i); });
$("preview").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { const c = e.target.closest(".cand"); if (c) { e.preventDefault(); loadCandidate(+c.dataset.i); } } });
let rsz; window.addEventListener("resize", () => { clearTimeout(rsz); rsz = setTimeout(renderWorld, 120); });

loadSettings();
if (!$("seed").value) $("seed").value = 1729;
newWorld(settings().seed);
