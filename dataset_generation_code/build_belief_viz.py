"""Render the reconstructed belief graph into a self-contained interactive HTML.

Two panels:
  1. Aggregate structural accuracy vs action-turn ordinal (graph_score + cause/proxy/
     decoy components), across all worlds — "how fast the reasoning graph converges to
     the true SCM." IQR band on graph_score.
  2. Belief-graph explorer: pick a world + drag a turn slider; an SVG causal graph
     builds up as the agent explores — actuators (left) linked to the OUTCOME (centre)
     when believed causal (edge colored by believed sign), measurables (right) linked as
     the mechanism proxy; decoys ringed red, ruled-out faded. Node borders show whether
     the belief matches the true SCM role (green ✓ / red ✗). An edit log lists the
     symbolic changes at each turn.

Run:  python3 build_belief_viz.py OUT.html beliefs_scored.json
"""
import json, sys

OUT = sys.argv[1]
SCORED = json.load(open(sys.argv[2]))
# trim per_world payload for size
for w in SCORED["per_world"]:
    for s in w["snapshots"]:
        s.pop("ruled_out", None)
DATA = json.dumps(SCORED)

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RPG scientist — reasoning-graph reconstruction</title>
<style>
:root[data-theme="dark"]{color-scheme:dark;}
.viz-root{color-scheme:light;
 --surface-1:#fcfcfb;--page:#f9f9f7;--text-primary:#0b0b0b;--text-secondary:#52514e;
 --muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,0.10);
 --series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--series-8:#e34948;--good:#006300;
 font-family:system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--page);color:var(--text-primary);
 min-height:100vh;margin:0;padding:24px 28px;}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-8:#e66767;--good:#0ca30c;}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
 --surface-1:#1a1a19;--page:#0d0d0d;--text-primary:#fff;--text-secondary:#c3c2b7;--muted:#898781;
 --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);
 --series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--series-8:#e66767;--good:#0ca30c;}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin-bottom:18px;}
h1{font-size:20px;margin:0 0 2px;}h2{font-size:14px;margin:0 0 10px;color:var(--text-secondary);font-weight:500;}
.sub{color:var(--text-secondary);font-size:13px;margin:0 0 14px;max-width:74ch;line-height:1.5;}
.btn{font:inherit;font-size:12px;padding:5px 11px;border:1px solid var(--border);border-radius:6px;background:var(--surface-1);color:var(--text-secondary);cursor:pointer;}
.btn:hover{color:var(--text-primary);}
.legend{display:flex;gap:16px;align-items:center;margin:2px 0 10px;font-size:12.5px;color:var(--text-secondary);flex-wrap:wrap;}
.legend .k{display:inline-flex;align-items:center;gap:6px;}
.swatch{width:20px;height:3px;border-radius:2px;display:inline-block;}
.ref-swatch{width:20px;height:0;border-top:2px dashed var(--muted);display:inline-block;}
svg{display:block;width:100%;height:auto;overflow:visible;}
.tt{position:fixed;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);border-radius:7px;padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.14);opacity:0;transition:opacity .08s;z-index:9;}
select,input[type=range]{font:inherit;}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:8px;}
.controls label{font-size:12px;color:var(--text-secondary);}
.node text{font-size:11px;}
.editlog{font-size:12px;color:var(--text-secondary);max-height:120px;overflow:auto;border-top:1px solid var(--grid);margin-top:8px;padding-top:8px;font-variant-numeric:tabular-nums;}
.editlog b{color:var(--text-primary);}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;border:1px solid var(--border);margin-left:6px;}
</style></head>
<body><div class="viz-root">
 <div class="card">
  <h1>Reasoning-graph reconstruction — how the agent's causal belief converges</h1>
  <p class="sub">A post-hoc LLM read each turn's scratchpad into a structured belief
  {cause, proxy, decoys, signs}, scored against the true SCM. <b>graph_score</b> = mean(cause correct,
  proxy correct, decoy-F1). Curves are means over all reconstructed worlds vs the action-turn ordinal
  (1st decision, 2nd, …).</p>
  <button class="btn" id="theme">◐ Theme</button>
  <div id="legend" class="legend"></div>
  <div id="accChart"></div>
 </div>
 <div class="card">
  <h2>Belief-graph explorer — watch the causal graph build up as the agent explores</h2>
  <div class="controls">
   <label>world <select id="worldSel"></select></label>
   <label>turn <input type="range" id="turnSlider" min="0" value="0"> <span id="turnLabel"></span></label>
   <span id="scoreLabel" class="sub" style="margin:0"></span>
  </div>
  <div id="beliefGraph"></div>
  <div id="editlog" class="editlog"></div>
 </div>
</div>
<div class="tt" id="tt"></div>
<script>
const S = __DATA__;
const KMAX = S.kmax, tt = document.getElementById('tt');
const NS='http://www.w3.org/2000/svg';
const add=(svg,t,a,txt)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;svg.appendChild(e);return e;};

/* ---------- panel 1: aggregate accuracy ---------- */
function accChart(){
 const el=document.getElementById('accChart');
 const w=780,h=360,M={t:14,r:20,b:40,l:46},iw=w-M.l-M.r,ih=h-M.t-M.b;
 const x=q=>M.l+(q-1)/(KMAX-1)*iw, y=v=>M.t+(1-v)*ih;
 const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);el.appendChild(svg);
 [0,.25,.5,.75,1].forEach(v=>{add(svg,'line',{x1:M.l,x2:M.l+iw,y1:y(v),y2:y(v),stroke:'var(--grid)','stroke-width':1});
   add(svg,'text',{x:M.l-8,y:y(v)+3,'text-anchor':'end',fill:'var(--muted)','font-size':11},v.toFixed(2));});
 [1,3,5,7,9,11,13,15].forEach(q=>{add(svg,'text',{x:x(q),y:M.t+ih+16,'text-anchor':'middle',fill:'var(--muted)','font-size':11},q);});
 add(svg,'text',{x:M.l+iw/2,y:h-2,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':12},'action-turn ordinal (agent decisions in order)');
 add(svg,'line',{x1:M.l,x2:M.l+iw,y1:M.t+ih,y2:M.t+ih,stroke:'var(--axis)','stroke-width':1});
 const series=[['graph','var(--series-1)','graph_score',true],['cause','var(--series-3)','cause correct',false],
   ['proxy','var(--series-2)','proxy correct',false],['decoy','var(--series-8)','decoy F1',false]];
 series.forEach(([c,col,lab,band])=>{
   const arr=S.overall[c].slice(0,KMAX);
   if(band){let d='M'+arr.map(p=>`${x(p.q)},${y(p.p75)}`).join('L')+'L'+arr.slice().reverse().map(p=>`${x(p.q)},${y(p.p25)}`).join('L')+'Z';
     add(svg,'path',{d,fill:col,opacity:.10,stroke:'none'});}
   const dl='M'+arr.map(p=>`${x(p.q)},${y(p.mean)}`).join('L');
   add(svg,'path',{d:dl,fill:'none',stroke:col,'stroke-width':band?2.6:1.8,'stroke-dasharray':band?'':'',opacity:band?1:.9,'stroke-linejoin':'round'});
   arr.forEach(p=>{const dot=add(svg,'circle',{cx:x(p.q),cy:y(p.mean),r:band?4:3,fill:col,stroke:'var(--surface-1)','stroke-width':1.2});
     dot.addEventListener('mousemove',ev=>{tt.style.opacity=1;tt.style.left=(ev.clientX+14)+'px';tt.style.top=(ev.clientY+14)+'px';
       tt.innerHTML=`<b>${lab}</b> · turn ${p.q}<br>mean <b>${p.mean.toFixed(3)}</b>`;});
     dot.addEventListener('mouseleave',()=>tt.style.opacity=0);});
 });
 const lg=document.getElementById('legend');
 series.forEach(([c,col,lab])=>{const k=document.createElement('span');k.className='k';
   k.innerHTML=`<span class="swatch" style="background:${col}"></span>${lab}`;lg.appendChild(k);});
}

/* ---------- panel 2: belief graph explorer ---------- */
const worlds=S.per_world;
const sel=document.getElementById('worldSel');
worlds.forEach((w,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${w.archetype} — ${w.world_id.slice(0,42)}`;sel.appendChild(o);});
const slider=document.getElementById('turnSlider');

function drawBelief(wi, ti){
 const w=worlds[wi], snap=w.snapshots[ti]||w.snapshots[w.snapshots.length-1];
 const gt=w.gt;
 // nodes to show: GT-relevant + anything the agent ever mentioned, capped
 const mentionedA=new Set(), mentionedM=new Set();
 w.snapshots.forEach(s=>{if(s.cause)mentionedA.add(s.cause);(s.decoys||[]).forEach(d=>{mentionedA.has(d)||mentionedM.add(d);});
   Object.keys(s.signs||{}).forEach(a=>mentionedA.add(a)); if(s.proxy)mentionedM.add(s.proxy);});
 [gt.cause,gt.trap].forEach(a=>a&&mentionedA.add(a)); gt.proxy&&mentionedM.add(gt.proxy);(gt.decoys||[]).forEach(d=>mentionedM.add(d));
 const acts=[...mentionedA].slice(0,8), meas=[...mentionedM].slice(0,8);
 const el=document.getElementById('beliefGraph');el.innerHTML='';
 const w2=780,h2=Math.max(300,40+Math.max(acts.length,meas.length)*46),M={t:30,l:150,r:150};
 const svg=document.createElementNS(NS,'svg');svg.setAttribute('viewBox',`0 0 ${w2} ${h2}`);el.appendChild(svg);
 const cx=w2/2, cy=h2/2;
 const ay=i=>M.t+ (acts.length>1? i*(h2-2*M.t)/(acts.length-1):(h2-2*M.t)/2), ax=110;
 const my=i=>M.t+ (meas.length>1? i*(h2-2*M.t)/(meas.length-1):(h2-2*M.t)/2), mx=w2-110;
 const cur={cause:snap.cause,proxy:snap.proxy,decoys:new Set(snap.decoys||[]),signs:snap.signs||{}};
 // edges first
 const signCol=s=> s==='+'?'var(--series-3)': s==='-'?'var(--series-8)':'var(--muted)';
 acts.forEach((a,i)=>{ if(a===cur.cause){ const s=cur.signs[a]||'+';
   add(svg,'line',{x1:ax+8,y1:ay(i),x2:cx-46,y2:cy,stroke:signCol(s),'stroke-width':3,'marker-end':'url(#arr)'});}});
 meas.forEach((m,i)=>{ if(m===cur.proxy){ add(svg,'line',{x1:mx-8,y1:my(i),x2:cx+46,y2:cy,stroke:'var(--series-3)','stroke-width':2.5,'stroke-dasharray':'5 4'});}});
 // arrow marker
 const defs=add(svg,'defs',{});const mk=document.createElementNS(NS,'marker');
 mk.setAttribute('id','arr');mk.setAttribute('markerWidth','9');mk.setAttribute('markerHeight','9');mk.setAttribute('refX','7');mk.setAttribute('refY','3');mk.setAttribute('orient','auto');
 const pth=document.createElementNS(NS,'path');pth.setAttribute('d','M0,0 L7,3 L0,6 Z');pth.setAttribute('fill','var(--muted)');mk.appendChild(pth);defs.appendChild(mk);
 // outcome node
 add(svg,'rect',{x:cx-44,y:cy-18,width:88,height:36,rx:8,fill:'var(--surface-1)',stroke:'var(--text-primary)','stroke-width':1.5});
 add(svg,'text',{x:cx,y:cy+4,'text-anchor':'middle',fill:'var(--text-primary)','font-size':12,'font-weight':600},'OUTCOME');
 // node renderer
 function node(name,nx,ny,kind){
   let ring='var(--border)', fill='var(--surface-1)', txt='var(--text-primary)', badge='';
   const isCause=name===cur.cause, isProxy=name===cur.proxy, isDecoy=cur.decoys.has(name);
   if(isCause){ring='var(--series-1)';badge=(name===gt.cause)?'✓':'✗';}
   else if(isProxy){ring='var(--series-3)';badge=(name===gt.proxy)?'✓':'✗';}
   else if(isDecoy){ring='var(--series-8)'; badge=(gt.decoys.includes(name))?'✓':((name===gt.proxy||name===gt.cause)?'✗ trap':'');}
   // GT role hint (subtle) when agent hasn't marked it
   const g=document.createElementNS(NS,'g');g.setAttribute('class','node');svg.appendChild(g);
   const rr=add(g,'rect',{x:nx-64,y:ny-15,width:128,height:30,rx:7,fill,stroke:ring,'stroke-width':(isCause||isProxy||isDecoy)?2.4:1});
   if(name===gt.cause) add(g,'circle',{cx:nx-64,cy:ny-15,r:3.5,fill:'var(--series-1)'});
   if(name===gt.proxy) add(g,'circle',{cx:nx+64,cy:ny-15,r:3.5,fill:'var(--series-3)'});
   add(g,'text',{x:nx,y:ny+4,'text-anchor':'middle',fill:txt,'font-size':11},name.length>18?name.slice(0,17)+'…':name);
   if(badge) add(g,'text',{x:nx+58,y:ny-6,'text-anchor':'end',fill:ring,'font-size':11,'font-weight':700},badge);
 }
 add(svg,'text',{x:ax,y:14,'text-anchor':'middle',fill:'var(--muted)','font-size':11,'font-weight':600},'ACTUATORS (levers)');
 add(svg,'text',{x:mx,y:14,'text-anchor':'middle',fill:'var(--muted)','font-size':11,'font-weight':600},'MEASURABLES');
 acts.forEach((a,i)=>node(a,ax,ay(i)));
 meas.forEach((m,i)=>node(m,mx,my(i)));
 // labels
 document.getElementById('turnLabel').textContent=`${ti+1}/${w.snapshots.length} (world turn ${snap.turn})`;
 document.getElementById('scoreLabel').innerHTML=
   `graph_score <b>${snap.graph_score.toFixed(2)}</b><span class="pill">cause ${snap.cause_ok?'✓':'✗'}</span>`+
   `<span class="pill">proxy ${snap.proxy_ok?'✓':'✗'}</span><span class="pill">decoy F1 ${snap.decoy_f1.toFixed(2)}</span>`+
   (snap.trap_ok?'':'<span class="pill" style="color:var(--series-8)">trap error</span>');
 // edit log
 const ed=document.getElementById('editlog');
 ed.innerHTML='<b>GT:</b> cause='+gt.cause+' · proxy='+gt.proxy+' · decoys=['+gt.decoys.join(', ')+']'+(gt.trap?' · trap='+gt.trap:'')+'<br>'+
   w.edits.filter((e,i)=>i<=ti && e.edits.length).map(e=>`t${e.turn}: ${e.edits.join(', ')}`).join('<br>');
}
function refresh(){const wi=+sel.value;slider.max=worlds[wi].snapshots.length-1;drawBelief(wi,+slider.value);}
sel.onchange=()=>{slider.value=0;refresh();};
slider.oninput=()=>drawBelief(+sel.value,+slider.value);
document.getElementById('theme').onclick=()=>{const r=document.documentElement;r.dataset.theme=r.dataset.theme==='dark'?'light':'dark';};
accChart(); sel.value=0; slider.max=worlds[0].snapshots.length-1; slider.value=worlds[0].snapshots.length-1; refresh();
</script></body></html>"""

open(OUT, "w").write(HTML.replace("__DATA__", DATA))
print(f"wrote {OUT}  ({SCORED['n_worlds']} worlds)")
