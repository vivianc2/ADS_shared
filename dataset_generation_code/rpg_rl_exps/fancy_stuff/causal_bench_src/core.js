// ===== Causal Bench core: seeded toy world + reference Bayesian learner + IG potentials =====
// A faithful port of toy_world.py / ig_reward.py (same structural equations, same hypothesis
// space, same sign-only skew-normal likelihoods).
const LOG2PI = Math.log(2 * Math.PI);
const PROXY = 0, DECOY = 1, ECHO = 2, IRREL = 3;
const ROLE_NAME = ["proxy", "decoy", "echo", "irrelevant"];

// ---------- RNG (mulberry32 + Box-Muller) ----------
function makeRng(seed) {
  let a = seed >>> 0;
  const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  let spare = null;
  return {
    uniform: u,
    normal() { if (spare !== null) { const s = spare; spare = null; return s; } let u1 = u(); while (u1 <= 1e-12) u1 = u(); const u2 = u(); const r = Math.sqrt(-2 * Math.log(u1)), th = 2 * Math.PI * u2; spare = r * Math.sin(th); return r * Math.cos(th); },
    int(n) { return Math.floor(u() * n); },
    shuffle(arr) { const b = arr.slice(); for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; },
    choice(items, p) { let x = u(), c = 0; for (let i = 0; i < items.length; i++) { c += p[i]; if (x < c) return items[i]; } return items[items.length - 1]; },
  };
}

// ---------- special functions ----------
function erfc(x) {            // Numerical Recipes erfcc, rel. err < 1.2e-7, good in the tails
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
function logNdtr(x) {         // log Phi(x)
  if (x > 6) return -0.5 * erfc(x / Math.SQRT2);       // log(1-e) ~ -e
  const v = 0.5 * erfc(-x / Math.SQRT2);
  if (v > 0) return Math.log(v);
  return -0.5 * x * x - Math.log(-x) - 0.5 * LOG2PI;    // asymptotic tail
}
function logaddexp(a, b) { if (a === -Infinity) return b; if (b === -Infinity) return a; const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m)); }
function logsumexp(arr, mask) {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) if (!mask || mask[i]) { if (arr[i] > m) m = arr[i]; }
  if (m === -Infinity) return -Infinity;
  let s = 0;
  for (let i = 0; i < arr.length; i++) if (!mask || mask[i]) s += Math.exp(arr[i] - m);
  return m + Math.log(s);
}
function llSign(z, sgn, tau) {
  if (sgn === 0) return -0.5 * z * z - 0.5 * LOG2PI;
  const zz = sgn * z, om2 = 1 + tau * tau;
  return Math.log(2) - 0.5 * zz * zz / om2 - 0.5 * (LOG2PI + Math.log(om2)) + logNdtr(zz * tau / Math.sqrt(om2));
}
function llTable(z, tau) { return [llSign(z, -1, tau), llSign(z, 0, tau), llSign(z, 1, tau)]; }
function llTwoSided(z, tau) { return logaddexp(llSign(z, 1, tau), llSign(z, -1, tau)) - Math.log(2); }
function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }
function sd(a) { const m = mean(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return Math.sqrt(s / (a.length - 1)); }
function corr(a, b) { const ma = mean(a), mb = mean(b); let sab = 0, saa = 0, sbb = 0; for (let i = 0; i < a.length; i++) { const da = a[i] - ma, db = b[i] - mb; sab += da * db; saa += da * da; sbb += db * db; } return sab / Math.sqrt(saa * sbb); }
const r4 = (x) => Math.round(x * 1e4) / 1e4;

// ---------- the world ----------
class World {
  constructor(seed, opts = {}) {
    this.seed = seed; this.K = opts.K || 4; this.M = opts.M || 4;
    this.nObs = opts.nObs || 400; this.nInt = opts.nInt || 60;
    const rng = makeRng(seed); this.rng = rng;
    const perm = rng.shuffle([...Array(this.K).keys()]);
    this.fix = perm[0]; this.game = [perm[1]];
    this.sign = rng.uniform() < 0.5 ? -1 : 1;
    this.aroles = Array(this.K).fill("null"); this.aroles[this.fix] = "fix"; for (const g of this.game) this.aroles[g] = "game";
    const mperm = rng.shuffle([...Array(this.M).keys()]);
    this.proxy = mperm[0];
    this.mroles = Array(this.M).fill("irrelevant"); this.mroles[this.proxy] = "proxy";
    for (const j of mperm.slice(1)) this.mroles[j] = rng.choice(["decoy", "echo", "irrelevant"], [0.4, 0.3, 0.3]);
    this.beta = 0.3 + 0.6 * rng.uniform(); this.gamma = 0.3 + 0.6 * rng.uniform();
    this.outcome = "m0";
    this.signals = [...Array(this.M).keys()].map(j => "m" + (j + 1));
    this.measurables = [this.outcome, ...this.signals];
    this.actuators = [...Array(this.K).keys()].map(i => "a" + i);
    this.expId = 0;
  }
  get gold() {
    const decoys = this.signals.filter((s, j) => this.mroles[j] === "decoy" || this.mroles[j] === "echo");
    const signs = {}; this.actuators.forEach((a, i) => signs[a] = i === this.fix ? (this.sign > 0 ? "+" : "-") : "0");
    return { actions: [{ actuator: "a" + this.fix, value: this.sign > 0 ? 100 : 0 }], proxy: "m" + (this.proxy + 1), decoys, signs, game: this.game.map(g => "a" + g) };
  }
  trueHypothesis() {
    const mroles = {}; this.signals.forEach((s, j) => mroles[s] = this.mroles[j]);
    return { fix: "a" + this.fix, sign: this.sign, game: new Set(this.game.map(g => "a" + g)), proxy: "m" + (this.proxy + 1), orient: 1, mroles };
  }
  sample(n, act, rng) {
    rng = rng || this.rng;
    const d = new Float64Array(this.K); for (const k in act) d[k] = (act[k] - 50) / 50;
    const U = new Float64Array(n), T = new Float64Array(n), Y = new Float64Array(n);
    let dg = 0; for (const g of this.game) dg += d[g];
    for (let i = 0; i < n; i++) {
      U[i] = rng.normal();
      T[i] = 0.6 * U[i] + this.sign * this.beta * d[this.fix] + 0.5 * rng.normal();
      Y[i] = T[i] + this.gamma * dg + 0.5 * rng.normal();
    }
    const out = { m0: Y, _T: T };
    for (let j = 0; j < this.M; j++) {
      const r = this.mroles[j], x = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        if (r === "proxy") x[i] = T[i] + 0.5 * rng.normal();
        else if (r === "decoy") x[i] = U[i] + 0.5 * rng.normal();
        else if (r === "echo") x[i] = Y[i] + 0.5 * rng.normal();
        else x[i] = rng.normal();
      }
      out["m" + (j + 1)] = x;
    }
    return out;
  }
  measure(names, rng) {
    names = names.filter(n => this.measurables.includes(n));
    this.expId++;
    const s = this.sample(this.nObs, {}, rng);
    const readings = {};
    for (const n of names) readings[n] = { mean: r4(mean(s[n])), sd: r4(sd(s[n])) };
    const c = {};
    for (let i = 0; i < names.length; i++) for (let j = i + 1; j < names.length; j++) c[names[i] + "~" + names[j]] = r4(corr(s[names[i]], s[names[j]]));
    readings._correlations = c;
    return { experiment_id: this.expId, n_units: this.nObs, readings };
  }
  intervene(actions, names, rng) {
    const act = {}, applied = {};
    for (const x of actions) { const a = x.actuator; if (this.actuators.includes(a)) { const v = Math.max(0, Math.min(100, +x.value)); act[+a.slice(1)] = v; applied[a] = v; } }
    names = names.filter(n => this.measurables.includes(n));
    if (!names.includes(this.outcome)) names = [this.outcome, ...names];
    this.expId++;
    const s = this.sample(this.nInt, act, rng);
    const readings = {}; for (const n of names) readings[n] = r4(mean(s[n]));
    return { experiment_id: this.expId, n_units: this.nInt, readings, applied_intervention: applied };
  }
  trueObjectiveShift(actions) {
    let d = 0; for (const x of (actions || [])) { const i = +String(x.actuator).slice(1); if (i === this.fix) d += this.sign * this.beta * ((+x.value) - 50) / 50; } return d;
  }
  grade(ans) {
    const gold = this.gold, acts = ans.actions || [];
    const shift = acts.length ? this.trueObjectiveShift(acts) : 0;
    const partA = shift > 0.05 ? 1 : 0;
    const proxyOk = ans.proxy === gold.proxy ? 1 : 0;
    const gd = new Set(gold.decoys), ad = new Set(ans.decoys || []);
    const inter = [...gd].filter(x => ad.has(x)).length, uni = new Set([...gd, ...ad]).size;
    const jac = (!gd.size && !ad.size) ? 1 : inter / Math.max(1, uni);
    const sg = ans.signs || {}; let ok = 0; for (const a of this.actuators) if (sg[a] === gold.signs[a]) ok++;
    const signAcc = ok / this.actuators.length;
    const partB = (proxyOk + jac + signAcc) / 3;
    const trap = acts.some(x => gold.game.includes(String(x.actuator)));
    return { reward: 0.5 * partA + 0.5 * partB, part_a: partA, part_b: partB, proxy_ok: proxyOk, decoy_jaccard: jac, sign_acc: signAcc, recommends_gaming_control: trap ? 1 : 0, true_shift: shift };
  }
}

// ---------- hypothesis space ----------
class Space {
  constructor(actuators, signals) {
    this.actuators = actuators; this.signals = signals;
    const K = actuators.length, M = signals.length;
    const fix = [], sign = [], isGame = [];
    for (let f = 0; f < K; f++) {
      const others = [...Array(K).keys()].filter(i => i !== f);
      for (const s of [1, -1]) for (let bits = 0; bits < (1 << (K - 1)); bits++) {
        const g = new Int8Array(K); others.forEach((o, k) => g[o] = (bits >> (K - 2 - k)) & 1);
        fix.push(f); sign.push(s); isGame.push(g);
      }
    }
    this.fix = fix; this.sign = sign; this.isGame = isGame; this.HA = fix.length;
    const proxy = [], orient = [], mrole = [];
    for (let p = 0; p < M; p++) {
      const others = [...Array(M).keys()].filter(j => j !== p);
      for (const o of [1, -1]) {
        const n = Math.pow(3, M - 1);
        for (let code = 0; code < n; code++) {
          const r = new Int8Array(M).fill(PROXY); let c = code;
          for (let k = others.length - 1; k >= 0; k--) { r[others[k]] = [DECOY, ECHO, IRREL][c % 3]; c = Math.floor(c / 3); }
          proxy.push(p); orient.push(o); mrole.push(r);
        }
      }
    }
    this.proxy = proxy; this.orient = orient; this.mrole = mrole; this.HM = proxy.length;
  }
  maskOf(h) {
    const K = this.actuators.length;
    const f = this.actuators.indexOf(h.fix);
    const ma = new Uint8Array(this.HA);
    for (let i = 0; i < this.HA; i++) {
      let ok = this.fix[i] === f && this.sign[i] === h.sign;
      if (ok && h.game != null) { for (let a = 0; a < K; a++) if (this.isGame[i][a] !== (h.game.has(this.actuators[a]) ? 1 : 0)) { ok = false; break; } }
      ma[i] = ok ? 1 : 0;
    }
    const p = this.signals.indexOf(h.proxy);
    const allowed = { proxy: [PROXY], decoy: [DECOY], echo: [ECHO], irrelevant: [IRREL], decoyish: [DECOY, ECHO], any: [PROXY, DECOY, ECHO, IRREL] };
    const mm = new Uint8Array(this.HM);
    for (let i = 0; i < this.HM; i++) {
      let ok = this.proxy[i] === p && (h.orient == null || this.orient[i] === h.orient);
      if (ok) for (let j = 0; j < this.signals.length; j++) { const s = this.signals[j]; if (s === h.proxy) continue; const roles = allowed[(h.mroles && h.mroles[s]) || "any"]; if (!roles.includes(this.mrole[i][j])) { ok = false; break; } }
      mm[i] = ok ? 1 : 0;
    }
    const mask = new Uint8Array(this.HA * this.HM); let any = 0;
    for (let i = 0; i < this.HA; i++) if (ma[i]) for (let k = 0; k < this.HM; k++) if (mm[k]) { mask[i * this.HM + k] = 1; any++; }
    if (!any) throw new Error("gold not representable");
    return mask;
  }
}

// ---------- reference learner ----------
class Learner {
  constructor(actuators, signals, outcome, opts = {}) {
    this.space = new Space(actuators, signals); this.outcome = outcome;
    this.better = opts.better || 1; this.tau = opts.tau || 4; this.defaultValue = opts.defaultValue || 50;
    this.reset();
  }
  reset() { const N = this.space.HA * this.space.HM; this.logpost = new Float64Array(N).fill(-Math.log(N)); this.baseline = {}; this.lastZY = 0; }
  clone() { const c = Object.create(Learner.prototype); c.space = this.space; c.outcome = this.outcome; c.better = this.better; c.tau = this.tau; c.defaultValue = this.defaultValue; c.logpost = new Float64Array(this.logpost); c.baseline = JSON.parse(JSON.stringify(this.baseline)); c.lastZY = this.lastZY; return c; }
  normalize() { const z = logsumexp(this.logpost); for (let i = 0; i < this.logpost.length; i++) this.logpost[i] -= z; }
  updateBaseline(mid, m, s, n) {
    const se = s / Math.sqrt(Math.max(n, 2));
    if (this.baseline[mid]) { const b = this.baseline[mid], w0 = 1 / (b.se * b.se), w1 = 1 / (se * se); this.baseline[mid] = { mean: (w0 * b.mean + w1 * m) / (w0 + w1), se: 1 / Math.sqrt(w0 + w1), sd: 0.5 * (b.sd + s) }; }
    else this.baseline[mid] = { mean: m, se, sd: s };
  }
  // Returns per-node log-likelihood terms (each a full HA*HM array) WITHOUT applying them.
  termsMeasure(result) {
    const sp = this.space, HA = sp.HA, HM = sp.HM, rd = result.readings || {}, n = result.n_units || 0;
    for (const mid in rd) if (mid !== "_correlations" && typeof rd[mid] === "object") this.updateBaseline(mid, rd[mid].mean, rd[mid].sd, n);
    const terms = [];
    if (n < 4 || !rd[this.outcome]) return terms;
    const c = rd._correlations || {};
    sp.signals.forEach((sid, j) => {
      let r = c[sid + "~" + this.outcome]; if (r == null) r = c[this.outcome + "~" + sid]; if (r == null) return;
      r = Math.max(-0.999, Math.min(0.999, r));
      const z = Math.atanh(r) * Math.sqrt(n - 3), lc = llTwoSided(z, this.tau), l0 = llSign(z, 0, this.tau);
      const ll = new Float64Array(HA * HM);
      for (let k = 0; k < HM; k++) { const v = sp.mrole[k][j] === IRREL ? l0 : lc; for (let i = 0; i < HA; i++) ll[i * HM + k] = v; }
      terms.push({ node: sid, ll, z, r, kind: "corr" });
    });
    return terms;
  }
  termsIntervene(result) {
    const sp = this.space, HA = sp.HA, HM = sp.HM, K = sp.actuators.length;
    const applied = result.applied_intervention || {}, rd = result.readings || {}, n = result.n_units || 0;
    const sT = new Int8Array(HA), sY = new Int8Array(HA);
    const fT = new Float64Array(HA), fY = new Float64Array(HA);
    for (const aid in applied) {
      const a = sp.actuators.indexOf(aid); if (a < 0) continue;
      const d = Math.sign(applied[aid] - this.defaultValue); if (!d) continue;
      for (let i = 0; i < HA; i++) { const fh = sp.fix[i] === a ? 1 : 0; fT[i] += d * sp.sign[i] * fh; fY[i] += d * sp.sign[i] * fh + d * sp.isGame[i][a]; }
    }
    for (let i = 0; i < HA; i++) { sT[i] = Math.sign(fT[i]); sY[i] = Math.sign(fY[i]); }
    const terms = []; this.lastZY = 0;
    for (const mid in rd) {
      if (mid === "_correlations") continue;
      const m = typeof rd[mid] === "object" ? rd[mid].mean : rd[mid];
      const b = this.baseline[mid]; if (!b || n < 2) { terms.push({ node: mid, ll: null, z: null, kind: "nobaseline" }); continue; }
      const seInt = b.sd / Math.sqrt(n), z = (m - b.mean) / Math.sqrt(seInt * seInt + b.se * b.se), tab = llTable(z, this.tau);
      const ll = new Float64Array(HA * HM);
      if (mid === this.outcome) {
        this.lastZY = this.better * z;
        for (let i = 0; i < HA; i++) { const v = tab[sY[i] + 1]; for (let k = 0; k < HM; k++) ll[i * HM + k] = v; }
      } else {
        const j = sp.signals.indexOf(mid); if (j < 0) continue;
        for (let i = 0; i < HA; i++) for (let k = 0; k < HM; k++) {
          const role = sp.mrole[k][j]; let v;
          if (role === PROXY) v = tab[sp.orient[k] * sT[i] + 1]; else if (role === ECHO) v = tab[sY[i] + 1]; else v = tab[1];
          ll[i * HM + k] = v;
        }
      }
      terms.push({ node: mid, ll, z, kind: "shift" });
    }
    return terms;
  }
  apply(terms) { for (const t of terms) if (t.ll) for (let i = 0; i < this.logpost.length; i++) this.logpost[i] += t.ll[i]; this.normalize(); }
  entropy() { let h = 0; for (const lp of this.logpost) { const p = Math.exp(lp); if (p > 0) h -= p * lp; } return h; }
  marginals() {
    const sp = this.space, HA = sp.HA, HM = sp.HM, K = sp.actuators.length, M = sp.signals.length;
    const pa = new Float64Array(HA), pm = new Float64Array(HM);
    for (let i = 0; i < HA; i++) for (let k = 0; k < HM; k++) { const p = Math.exp(this.logpost[i * HM + k]); pa[i] += p; pm[k] += p; }
    const ctrl = sp.actuators.map((a, ai) => { let fp = 0, fm = 0, g = 0; for (let i = 0; i < HA; i++) { if (sp.fix[i] === ai) { if (sp.sign[i] > 0) fp += pa[i]; else fm += pa[i]; } if (sp.isGame[i][ai]) g += pa[i]; } return { id: a, fixp: fp, fixm: fm, game: g, null: Math.max(0, 1 - fp - fm - g) }; });
    const sig = sp.signals.map((s, j) => { const r = [0, 0, 0, 0]; for (let k = 0; k < HM; k++) r[sp.mrole[k][j]] += pm[k]; return { id: s, proxy: r[0], decoy: r[1], echo: r[2], irrelevant: r[3] }; });
    return { ctrl, sig };
  }
  confidence() { const m = this.marginals(); let best = 0; for (const c of m.ctrl) best = Math.max(best, c.fixp, c.fixm); return best; }
  mapAnswer() {
    const m = this.marginals(); let best = { p: -1 };
    for (const c of m.ctrl) { if (c.fixp > best.p) best = { p: c.fixp, a: c.id, s: 1 }; if (c.fixm > best.p) best = { p: c.fixm, a: c.id, s: -1 }; }
    let proxy = m.sig[0]; for (const s of m.sig) if (s.proxy > proxy.proxy) proxy = s;
    const decoys = m.sig.filter(s => s.decoy + s.echo > 0.5).map(s => s.id);
    const signs = {}; for (const c of m.ctrl) { const o = [["+", c.fixp], ["-", c.fixm], ["0", 1 - c.fixp - c.fixm]].sort((x, y) => y[1] - x[1]); signs[c.id] = o[0][0]; }
    return { actions: [{ actuator: best.a, value: best.s * this.better > 0 ? 100 : 0 }], proxy: proxy.id, decoys, signs };
  }
}

// ---------- potentials ----------
function phiOf(lp, kind, mask) {
  if (kind === "entropy") { const z = logsumexp(lp); let h = 0; for (const v of lp) { const p = Math.exp(v - z); if (p > 0) h -= p * (v - z); } return -h; }
  return Math.max(logsumexp(lp, mask) - logsumexp(lp), Math.log(1e-8));   // oracle: log P(H* | D)
}
function addTerms(lp, terms) { const out = new Float64Array(lp); for (const t of terms) if (t.ll) for (let i = 0; i < out.length; i++) out[i] += t.ll[i]; return out; }

// ---------- the shaper: one experiment -> reward + per-node breakdown ----------
class Shaper {
  constructor(learner, gold, opts = {}) {
    this.L = learner; this.kind = opts.potential || "oracle"; this.lam = opts.lam ?? 0.1; this.clip = opts.clip ?? 1.0; this.terminalMode = opts.terminalMode || "keep";
    this.mask = gold ? learner.space.maskOf(gold) : null;
    this.reset();
  }
  reset() { this.L.reset(); this.phi = this.potential(this.L.logpost); this.phi0 = this.phi; this.total = 0; this.history = [this.phi]; }
  potential(lp) { return phiOf(lp, this.kind, this.mask); }
  update(atype, result) {
    const terms = atype === "measure" ? this.L.termsMeasure(result) : this.L.termsIntervene(result);
    const lp0 = this.L.logpost, phi0 = this.phi;
    const perNode = terms.map(t => ({ node: t.node, kind: t.kind, z: t.z, r: t.r, alone: t.ll ? this.lam * (this.potential(addTerms(lp0, [t])) - phi0) : null }));
    this.L.apply(terms);
    const phi1 = this.potential(this.L.logpost);
    const raw = this.lam * (phi1 - phi0), reward = Math.max(-this.clip, Math.min(this.clip, raw));
    this.phi = phi1; this.total += reward; this.history.push(phi1);
    return { reward, raw, phiBefore: phi0, phiAfter: phi1, perNode, zY: this.L.lastZY };
  }
  terminalAdjustment() { return this.terminalMode === "zero" ? -this.lam * this.phi : 0; }
  // expected reward of a candidate experiment: Monte-Carlo over the TRUE world (oracle preview)
  expectedReward(world, cand, nmc, rng) {
    let s = 0;
    for (let k = 0; k < nmc; k++) {
      const L = this.L.clone(); const sh = Object.create(Shaper.prototype); sh.L = L; sh.kind = this.kind; sh.lam = this.lam; sh.clip = this.clip; sh.mask = this.mask; sh.phi = this.phi; sh.total = 0; sh.history = [];
      const saveId = world.expId;
      const res = cand.type === "measure" ? world.measure(cand.names, rng) : world.intervene(cand.actions, cand.names, rng);
      world.expId = saveId;
      s += sh.update(cand.type, res).reward;
    }
    return s / nmc;
  }
}

if (typeof module !== "undefined") module.exports = { World, Space, Learner, Shaper, makeRng, phiOf, llSign, logNdtr, erfc };
