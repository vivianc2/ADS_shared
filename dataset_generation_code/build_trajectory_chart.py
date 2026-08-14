"""Render utility-trajectory JSON(s) into a self-contained interactive HTML chart.

Reads one or more trajectories.json (one per model) and emits a single HTML file
with the data embedded (opens in a browser, no server). Main panel: best-so-far
benefit_recovered vs query index, per model, with an IQR band and a gold=1.0
reference line. Below: per-archetype small multiples. Theme toggle + table view.

Run:  python3 build_trajectory_chart.py OUT.html traj1.json [traj2.json ...]
"""
import json, sys

OUT = sys.argv[1]
MODELS = [json.load(open(p)) for p in sys.argv[2:]]
if not MODELS:
    sys.exit("usage: build_trajectory_chart.py OUT.html traj1.json [traj2.json ...]")

# strip per_world to keep the file small; the chart uses aggregates only
payload = [{"model": m["model"], "kmax": m["kmax"], "n_worlds": m["n_worlds"],
            "overall": m["overall"], "by_arch": m["by_arch"]} for m in MODELS]
DATA = json.dumps(payload)

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RPG scientist — utility optimization speed</title>
<style>
:root[data-theme="dark"] { color-scheme: dark; }
.viz-root {
  color-scheme: light;
  --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b; --text-secondary:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --good:#006300;
  font-family: system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--page); color:var(--text-primary);
  min-height:100vh; margin:0; padding:24px 28px;
}
@media (prefers-color-scheme: dark){ :root:where(:not([data-theme="light"])) .viz-root{
  color-scheme:dark; --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --good:#0ca30c; } }
:root[data-theme="dark"] .viz-root{
  color-scheme:dark; --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
  --muted:#898781; --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --good:#0ca30c; }
.card{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:18px 20px; margin-bottom:18px; }
h1{ font-size:20px; margin:0 0 2px; } h2{ font-size:14px; margin:0 0 12px; color:var(--text-secondary); font-weight:500; }
.sub{ color:var(--text-secondary); font-size:13px; margin:0 0 16px; max-width:70ch; line-height:1.5; }
.toolbar{ display:flex; gap:8px; align-items:center; margin-bottom:14px; }
.btn{ font:inherit; font-size:12px; padding:5px 11px; border:1px solid var(--border); border-radius:6px;
  background:var(--surface-1); color:var(--text-secondary); cursor:pointer; }
.btn:hover{ color:var(--text-primary); }
.legend{ display:flex; gap:18px; align-items:center; margin:2px 0 10px; font-size:13px; color:var(--text-secondary); flex-wrap:wrap;}
.legend .k{ display:inline-flex; align-items:center; gap:7px; }
.swatch{ width:22px; height:3px; border-radius:2px; display:inline-block; }
.ref-swatch{ width:22px; height:0; border-top:2px dashed var(--muted); display:inline-block; }
.grid-sm{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.sm h3{ font-size:12px; margin:0 0 2px; font-weight:600; }
.sm .meta{ font-size:11px; color:var(--muted); margin:0 0 4px; }
svg{ display:block; width:100%; height:auto; overflow:visible; }
.tt{ position:fixed; pointer-events:none; background:var(--surface-1); border:1px solid var(--border);
  border-radius:7px; padding:8px 10px; font-size:12px; box-shadow:0 4px 14px rgba(0,0,0,.14); opacity:0; transition:opacity .08s; z-index:9; }
.tt b{ font-variant-numeric:tabular-nums; }
table{ border-collapse:collapse; font-size:12px; font-variant-numeric:tabular-nums; width:100%; }
th,td{ text-align:right; padding:3px 9px; border-bottom:1px solid var(--grid); }
th:first-child,td:first-child{ text-align:left; }
.hidden{ display:none; }
.dot{ cursor:pointer; }
</style>
</head>
<body>
<div class="viz-root">
  <div class="card">
    <h1>How fast does the scientist agent optimize utility?</h1>
    <p class="sub">Best-so-far <b>benefit recovered</b> = (utility of the best intervention run by query&nbsp;k − baseline) ÷ (gold utility − baseline),
    averaged over 72 worlds (8 × 9 archetypes) of the RL fast dataset. <b>1.0 = gold</b> (the oracle's optimal intervention); 0 = doing nothing.
    Utility of each executed intervention is recomputed on the true SCM. The curve shows optimization <i>speed</i>, not just final success within the 15-query budget.</p>
    <div class="toolbar">
      <button class="btn" id="theme">◐ Theme</button>
      <button class="btn" id="toggleTable">Table view</button>
      <span class="sub" style="margin:0;color:var(--muted)" id="counts"></span>
    </div>
    <div id="legend" class="legend"></div>
    <div id="mainChart"></div>
    <div id="tableWrap" class="hidden" style="margin-top:14px;overflow:auto;max-height:340px"></div>
  </div>
  <div class="card">
    <h2>Per-archetype — best-so-far benefit vs query (mean over 8 worlds each)</h2>
    <div id="smallMultiples" class="grid-sm"></div>
  </div>
</div>
<div class="tt" id="tt"></div>
<script>
const DATA = __DATA__;
const KMAX = DATA[0].kmax;
const SERIES_COLORS = ['var(--series-1)','var(--series-2)'];
const tt = document.getElementById('tt');

function lineChart(el, models, key, {w=760,h=380,m={t:16,r:18,b:40,l:46},title='',compact=false}={}){
  const M = compact ? {t:8,r:10,b:24,l:30} : m;
  const iw = w-M.l-M.r, ih = h-M.t-M.b;
  const x = q => M.l + (q-1)/(KMAX-1)*iw;
  const y = v => M.t + (1-(v)/1.05)*ih;   // 0..1.05 so gold=1 sits just below top
  const NS='http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox',`0 0 ${w} ${h}`); svg.setAttribute('role','img');
  const add=(t,a,p)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);else svg.appendChild(e);return e;};
  // gridlines + y ticks
  [0,0.25,0.5,0.75,1.0].forEach(v=>{
    add('line',{x1:M.l,x2:M.l+iw,y1:y(v),y2:y(v),stroke:'var(--grid)','stroke-width':1});
    if(!compact) add('text',{x:M.l-8,y:y(v)+3,'text-anchor':'end',fill:'var(--muted)','font-size':11},).textContent=v.toFixed(2);
  });
  // gold reference line at 1.0
  add('line',{x1:M.l,x2:M.l+iw,y1:y(1.0),y2:y(1.0),stroke:'var(--muted)','stroke-width':2,'stroke-dasharray':'5 4'});
  if(!compact) add('text',{x:M.l+iw,y:y(1.0)-6,'text-anchor':'end',fill:'var(--muted)','font-size':11,'font-weight':600}).textContent='gold = 1.0';
  // x ticks
  const xticks = compact?[1,5,10,15]:[1,3,5,7,9,11,13,15];
  xticks.forEach(q=>{ add('line',{x1:x(q),x2:x(q),y1:M.t+ih,y2:M.t+ih+4,stroke:'var(--axis)','stroke-width':1});
    add('text',{x:x(q),y:M.t+ih+16,'text-anchor':'middle',fill:'var(--muted)','font-size':compact?9:11}).textContent=q; });
  if(!compact){ add('text',{x:M.l+iw/2,y:h-2,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':12}).textContent='query index (interventions + measurements, budget 15)';
    const yl=add('text',{x:12,y:M.t+ih/2,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':12,transform:`rotate(-90 12 ${M.t+ih/2})`}); yl.textContent='best-so-far benefit recovered'; }
  // baseline axis
  add('line',{x1:M.l,x2:M.l+iw,y1:M.t+ih,y2:M.t+ih,stroke:'var(--axis)','stroke-width':1});
  models.forEach((mo,si)=>{
    const arr = mo[key]; if(!arr) return;
    const col = SERIES_COLORS[si%2];
    // IQR band (only in main, has p25/p75)
    if(!compact && arr[0].p25!=null){
      let d='M'+arr.map(p=>`${x(p.q)},${y(p.p75)}`).join('L');
      d+='L'+arr.slice().reverse().map(p=>`${x(p.q)},${y(p.p25)}`).join('L')+'Z';
      add('path',{d,fill:col,opacity:0.12,stroke:'none'});
    }
    // mean line
    const dl='M'+arr.map(p=>`${x(p.q)},${y(p.mean)}`).join('L');
    add('path',{d:dl,fill:'none',stroke:col,'stroke-width':compact?1.8:2.4,'stroke-linejoin':'round'});
    if(!compact) arr.forEach(p=>{ const c=add('circle',{cx:x(p.q),cy:y(p.mean),r:4,fill:col,stroke:'var(--surface-1)','stroke-width':1.5,class:'dot'});
      c.addEventListener('mousemove',ev=>{tt.style.opacity=1;tt.style.left=(ev.clientX+14)+'px';tt.style.top=(ev.clientY+14)+'px';
        tt.innerHTML=`<b>${mo.model}</b> · query ${p.q}<br>mean benefit <b>${p.mean.toFixed(3)}</b><br>IQR <b>${p.p25.toFixed(2)}–${p.p75.toFixed(2)}</b>`;});
      c.addEventListener('mouseleave',()=>tt.style.opacity=0); });
  });
  el.appendChild(svg);
}

// legend
const lg=document.getElementById('legend');
DATA.forEach((mo,i)=>{ const k=document.createElement('span'); k.className='k';
  k.innerHTML=`<span class="swatch" style="background:${SERIES_COLORS[i%2]}"></span>${mo.model} (n=${mo.n_worlds})`; lg.appendChild(k); });
const gk=document.createElement('span'); gk.className='k'; gk.innerHTML=`<span class="ref-swatch"></span>gold (oracle optimum)`; lg.appendChild(gk);
document.getElementById('counts').textContent = DATA.map(m=>`${m.model}: final mean ${m.overall[KMAX-1].mean.toFixed(2)}`).join('   ·   ');

lineChart(document.getElementById('mainChart'), DATA, 'overall', {});

// small multiples per archetype
const arches = Object.keys(DATA[0].by_arch).sort();
const sm=document.getElementById('smallMultiples');
arches.forEach(a=>{
  const cell=document.createElement('div'); cell.className='sm';
  const fin = DATA.map(m=> (m.by_arch[a]? m.by_arch[a][KMAX-1].mean:null));
  cell.innerHTML=`<h3>${a}</h3><div class="meta">final: ${DATA.map((m,i)=>fin[i]!=null?fin[i].toFixed(2):'—').join(' / ')}</div>`;
  const holder=document.createElement('div'); cell.appendChild(holder);
  const models = DATA.map(m=>({model:m.model, arr:m.by_arch[a]}));
  lineChart(holder, DATA.map(m=>({model:m.model, sm:m.by_arch[a]})), 'sm', {w:250,h:150,compact:true});
  sm.appendChild(cell);
});

// table
function buildTable(){
  let h='<table><thead><tr><th>query</th>'+DATA.map(m=>`<th>${m.model} mean</th><th>p25</th><th>p75</th>`).join('')+'</tr></thead><tbody>';
  for(let k=0;k<KMAX;k++){ h+=`<tr><td>${k+1}</td>`+DATA.map(m=>{const d=m.overall[k];return `<td>${d.mean.toFixed(3)}</td><td>${d.p25.toFixed(2)}</td><td>${d.p75.toFixed(2)}</td>`;}).join('')+'</tr>'; }
  h+='</tbody></table>'; return h;
}
document.getElementById('tableWrap').innerHTML=buildTable();
document.getElementById('toggleTable').onclick=()=>document.getElementById('tableWrap').classList.toggle('hidden');
document.getElementById('theme').onclick=()=>{const r=document.documentElement; r.dataset.theme = r.dataset.theme==='dark'?'light':'dark';};
</script>
</body></html>"""

open(OUT, "w").write(HTML.replace("__DATA__", DATA))
print(f"wrote {OUT}  ({len(MODELS)} model(s): {', '.join(m['model'] for m in MODELS)})")
