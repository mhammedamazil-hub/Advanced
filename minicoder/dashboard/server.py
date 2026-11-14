"""Minimal, dependency-free brain dashboard (``http.server`` based).

Routes
------
GET  /                     single-page dashboard UI
GET  /api/status           brain state + all memory stats
GET  /api/memory/working|episodic|semantic|procedural
GET  /api/graph            knowledge-graph nodes/edges
GET  /api/sleep/history    past sleep cycles
GET  /api/sleep/report     last consolidation report
GET  /api/learn            learning hierarchy history
GET  /api/benchmarks       benchmark runs
POST /api/sleep            enter sleep (async)
POST /api/wake             wake up
POST /api/eval             run Mini-Bench evaluation (async)
POST /api/export           export memory to file
POST /api/import           import memory export {path}
POST /api/perceive         {kind, content} → working memory
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from ..brain import Brain
from ..evals.benchmarks import run_benchmark_suite
from ..curriculum import Task

INDEX_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>MINICODER · Brain Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0e1117;--panel:#161b22;--border:#2b333d;--fg:#e6edf3;--mut:#8b949e;
--acc:#4cc2ff;--ok:#3fb950;--warn:#d29922;--bad:#f85149;--purp:#bc8cff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
header{display:flex;align-items:center;gap:12px;padding:12px 18px;border-bottom:1px solid var(--border)}
h1{font-size:16px;margin:0;letter-spacing:.5px}
.badge{padding:3px 10px;border-radius:12px;border:1px solid var(--border);font-weight:700}
.s-awake{background:#12361c;color:var(--ok)}
.s-thinking{background:#3b2f0b;color:var(--warn)}
.s-learning{background:#2d1f4e;color:var(--purp)}
.s-sleeping{background:#12233d;color:var(--acc)}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;padding:16px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px;min-height:120px}
.card h2{margin:0 0 10px;font-size:13px;color:var(--mut);text-transform:uppercase;letter-spacing:1px}
.big{font-size:26px;font-weight:700}
.bar{height:6px;background:#222;border-radius:3px;overflow:hidden;margin-top:6px}
.bar>div{height:100%;background:var(--acc)}
button{background:#1f6feb;color:#fff;border:0;border-radius:6px;padding:7px 12px;cursor:pointer;font:inherit;font-size:12px}
button:hover{filter:brightness(1.15)}button:disabled{opacity:.4;cursor:default}
button.ghost{background:#21262d;border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:4px 6px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mut{color:var(--mut)}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.controls{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.log{max-height:180px;overflow:auto;font-size:12px;white-space:pre-wrap}
.kv{display:flex;justify-content:space-between;padding:2px 0;font-size:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
</style></head><body>
<header>
  <h1>🧠 MINICODER</h1>
  <span id="state" class="badge s-awake">awake</span>
  <span class="mut" id="uptime"></span>
  <span style="flex:1"></span>
  <span class="mut" id="model"></span>
</header>
<main>
  <section class="card"><h2>Controls</h2>
    <div class="controls">
      <button onclick="act('sleep')">💤 Sleep</button>
      <button class="ghost" onclick="act('wake')">⏰ Wake</button>
      <button class="ghost" onclick="act('eval')">🧪 Run Evaluation</button>
      <button class="ghost" onclick="act('export')">📦 Export Memory</button>
    </div>
    <div class="controls" style="margin-top:8px">
      <input id="imp" placeholder="import path…" style="background:#0d1117;color:var(--fg);border:1px solid var(--border);border-radius:6px;padding:6px;font:inherit;font-size:12px;width:170px">
      <button class="ghost" onclick="imp()">📥 Import</button>
    </div>
    <div id="notice" class="mut" style="margin-top:8px;font-size:12px"></div>
  </section>
  <section class="card"><h2>Working Memory</h2>
    <div class="big"><span id="wm-items">0</span><span class="mut">/<span id="wm-cap">0</span></span></div>
    <div class="bar"><div id="wm-bar" style="width:0%"></div></div>
    <div class="mut" id="wm-extra" style="margin-top:6px"></div>
    <div class="log mut" id="wm-list" style="margin-top:8px"></div>
  </section>
  <section class="card"><h2>Long-Term Memory</h2>
    <div class="kv"><span>📼 episodes</span><b id="ep-c">0</b></div>
    <div class="kv"><span>&nbsp;&nbsp;verified</span><b class="ok" id="ep-s">0</b></div>
    <div class="kv"><span>&nbsp;&nbsp;failed</span><b class="bad" id="ep-f">0</b></div>
    <div class="kv"><span>&nbsp;&nbsp;consolidated</span><b id="ep-cons">0</b></div>
    <div class="kv"><span>🧩 concepts</span><b id="sem-c">0</b></div>
    <div class="kv"><span>&nbsp;&nbsp;facts / hypotheses</span><b><span class="ok" id="sem-f">0</span> / <span class="warn" id="sem-h">0</span></b></div>
    <div class="kv"><span>🕸 graph edges</span><b id="sem-e">0</b></div>
    <div class="kv"><span>🛠 skills (verified)</span><b id="sk-c">0</b></div>
    <div class="kv"><span>💾 db size</span><b id="db-size">0</b></div>
  </section>
  <section class="card"><h2>Metacognition</h2>
    <div class="kv"><span>current task</span><b id="mc-task" class="mut">—</b></div>
    <div class="kv"><span>confidence</span><b id="mc-conf">—</b></div>
    <div class="kv"><span>tasks completed</span><b id="mc-done">0</b></div>
    <div class="log mut" id="mc-gaps" style="margin-top:6px"></div>
  </section>
  <section class="card"><h2>Model & Benchmarks</h2>
    <div id="model-detail" class="mut" style="font-size:12px"></div>
    <div id="bench-list" style="margin-top:8px"></div>
  </section>
  <section class="card"><h2>Recent Experiences</h2><div id="recent"></div></section>
  <section class="card"><h2>Sleep History & Last Report</h2>
    <div id="sleep-hist"></div><div class="log mut" id="sleep-rep" style="margin-top:8px"></div>
  </section>
  <section class="card"><h2>Learning History (L1–L5)</h2><div id="learn"></div></section>
  <section class="card"><h2>Knowledge Graph (top nodes)</h2><div id="graph"></div></section>
</main>
<script>
const $=id=>document.getElementById(id);
function fmtKB(b){return b>1048576?(b/1048576).toFixed(1)+' MB':(b/1024).toFixed(1)+' KB';}
async function act(name,opts){
  try{
    const r=await fetch('/api/'+name,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(opts||{})});
    const j=await r.json();$('notice').textContent=j.message||JSON.stringify(j);refresh();
  }catch(e){$('notice').textContent='error: '+e;}
}
function imp(){const p=$('imp').value.trim();if(p)act('import',{path:p});}
async function refresh(){
  const s=await (await fetch('/api/status')).json();
  const st=$('state');st.textContent=s.state;st.className='badge s-'+s.state;
  $('uptime').textContent='uptime '+s.uptime_s+'s · experiences since sleep: '+s.experiences_since_sleep;
  const w=s.memory.working;$('wm-items').textContent=w.items;$('wm-cap').textContent=w.capacity;
  $('wm-bar').style.width=Math.round(100*w.usage)+'%';
  $('wm-extra').textContent=`pinned ${w.pinned} · compressions ${w.compressions}`;
  const e=s.memory.episodic;$('ep-c').textContent=e.count;$('ep-s').textContent=e.success;
  $('ep-f').textContent=e.failure;$('ep-cons').textContent=e.consolidated;
  const m=s.memory.semantic;$('sem-c').textContent=m.count;$('sem-f').textContent=m.facts;
  $('sem-h').textContent=m.hypotheses;$('sem-e').textContent=m.edges;
  $('sk-c').textContent=s.memory.procedural.count+' ('+s.memory.procedural.verified+'✓)';
  $('db-size').textContent=fmtKB(s.memory.db_bytes);
  const mc=s.metacognition;$('mc-task').textContent=mc.current_task||'—';
  $('mc-conf').textContent=mc.confidence==null?'—':(100*mc.confidence).toFixed(0)+'%';
  $('mc-done').textContent=mc.completed_tasks+' ('+mc.resolved+' resolved)';
  $('mc-gaps').textContent=(mc.gaps||[]).map(g=>'⚠ '+g).join('\\n');
  if(s.model.available){$('model').textContent=(s.model.params/1e6).toFixed(3)+'M params · '+s.model.checkpoint;
    $('model-detail').innerHTML=`params <b>${s.model.params.toLocaleString()}</b> · fp32 ${fmtKB(s.model.size.fp32_bytes)} → int8 ${fmtKB(s.model.size.int8_bytes)} (${s.model.size.compression_ratio}×) · d=${s.model.config.d_model} L=${s.model.config.n_layer} GQA ${s.model.config.n_head}→${s.model.config.n_kv_head}`;}
  else $('model-detail').textContent='model not loaded yet';
  const wm=await(await fetch('/api/memory/working')).json();
  $('wm-list').textContent=wm.map(i=>`[${i.kind}] ${i.content}`).slice(0,8).join('\\n');
  const rec=await(await fetch('/api/memory/episodic?n=6')).json();
  $('recent').innerHTML='<table>'+rec.map(r=>`<tr><td><span class="${r.status==='success'?'ok':r.status==='failure'?'bad':'warn'}">●</span> ${r.objective}</td><td class="mut">${r.status}</td></tr>`).join('')+'</table>';
  const sh=await(await fetch('/api/sleep/history')).json();
  $('sleep-hist').innerHTML='<table>'+sh.slice(-5).reverse().map(c=>`<tr><td>${c.id}</td><td class="${c.status==='completed'?'ok':'warn'}">${c.status}</td></tr>`).join('')+'</table>';
  if(s.last_sleep_report){const r=s.last_sleep_report;
    $('sleep-rep').textContent=`last cycle ${r.cycle_id} [${r.status}]\\nreplayed ${r.replayed_episodes.length} · consolidated ${r.consolidated_episodes.length} · new concepts ${r.new_concepts.length} · merged ${r.merged_concepts.length} · archived ${r.archived.length} · practice ${r.practice.map(p=>p.status).join(',')||'—'}\\nweaknesses: ${JSON.stringify(r.weaknesses)}\\nunresolved: ${r.unresolved.length}`;}
  const ln=await(await fetch('/api/learn')).json();
  $('learn').innerHTML='<table>'+ln.slice(0,6).map(l=>`<tr><td class="mut">L${l.level}</td><td>${l.summary}</td></tr>`).join('')+'</table>';
  const be=await(await fetch('/api/benchmarks')).json();
  $('bench-list').innerHTML='<table>'+be.slice(-5).reverse().map(b=>`<tr><td>${b.label}</td><td><b>${(100*b.score).toFixed(0)}%</b> <span class="mut">${new Date(b.timestamp*1000).toTimeString().slice(0,8)}</span></td></tr>`).join('')+'</table>';
  const g=await(await fetch('/api/graph')).json();
  const deg={};g.edges.forEach(e=>{deg[e.src]=(deg[e.src]||0)+1;deg[e.dst]=(deg[e.dst]||0)+1;});
  const top=g.nodes.map(n=>({n,d:deg[n.id]||0})).sort((a,b)=>b.d-a.d).slice(0,8);
  $('graph').innerHTML=top.map(t=>`<div class="kv"><span class="${t.n.status==='fact'?'ok':'warn'}"><span class="dot" style="background:${t.n.status==='fact'?'var(--ok)':'var(--warn)'}"></span>${t.n.name.slice(0,60)}</span><b>${t.d}</b></div>`).join('');
}
refresh();setInterval(refresh,2500);
</script></body></html>
"""


def make_app(brain: Brain):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):      # quiet logging
            pass

        # ------------------------------------------------------------- utils
        def _json(self, obj, code: int = 200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _html(self, text: str):
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        # --------------------------------------------------------------- GET
        def do_GET(self):               # noqa: N802
            path, _, query = self.path.partition("?")
            if path == "/":
                return self._html(INDEX_HTML)
            if path == "/api/status":
                return self._json(brain.status())
            if path == "/api/memory/working":
                return self._json([i.__dict__ for i in brain.memory.working.items])
            if path == "/api/memory/episodic":
                n = 10
                for part in query.split("&"):
                    if part.startswith("n="):
                        n = int(part[2:])
                return self._json([
                    {"id": e.id, "objective": e.objective, "status": e.status,
                     "domain": e.domain, "confidence": e.confidence,
                     "timestamp": e.timestamp}
                    for e in brain.memory.episodic.recent(n)])
            if path == "/api/memory/semantic":
                return self._json([
                    {"id": c.id, "name": c.name, "domain": c.domain,
                     "status": c.status, "confidence": c.confidence}
                    for c in brain.memory.semantic.all_active()][:50])
            if path == "/api/memory/procedural":
                return self._json([
                    {"id": s.id, "name": s.name, "trust": s.trust,
                     "success": s.success_count, "failure": s.failure_count}
                    for s in brain.memory.procedural.all_active()])
            if path == "/api/graph":
                return self._json(brain.memory.semantic.graph_snapshot())
            if path == "/api/sleep/history":
                rows = brain.memory.store.execute(
                    "SELECT id, started, ended, status FROM sleep_cycles "
                    "ORDER BY started DESC LIMIT 20").fetchall()
                return self._json([dict(r) for r in rows])
            if path == "/api/sleep/report":
                return self._json(brain.last_sleep_report or {})
            if path == "/api/learn":
                from ..learning.continual import ContinualLearner
                cl = ContinualLearner(brain.config, brain.memory, brain.sandbox,
                                      brain.checkpoints)
                return self._json(cl.history())
            if path == "/api/benchmarks":
                rows = brain.memory.store.execute(
                    "SELECT id, timestamp, label, score FROM benchmarks "
                    "ORDER BY timestamp DESC LIMIT 20").fetchall()
                return self._json([dict(r) for r in rows])
            return self._json({"error": "not found"}, 404)

        # -------------------------------------------------------------- POST
        def do_POST(self):              # noqa: N802
            path = self.path
            length = int(self.headers.get("Content-Length") or 0)
            payload = {}
            if length:
                try:
                    payload = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    payload = {}
            if path == "/api/sleep":
                msg = brain.sleep_async()
                return self._json({"message": msg})
            if path == "/api/wake":
                threading.Thread(target=brain.wake, daemon=True).start()
                return self._json({"message": "wake"})
            if path == "/api/eval":
                def _run():
                    try:
                        brain.ensure_model()
                        solver = _model_solver(brain)
                        run_benchmark_suite(brain.memory.store, "mini-bench",
                                            solver, brain.sandbox)
                    except Exception:
                        pass
                threading.Thread(target=_run, daemon=True).start()
                return self._json({"message": "evaluation started"})
            if path == "/api/export":
                out = brain.memory.export_all()
                return self._json({"message": f"exported to {out}"})
            if path == "/api/import":
                p = payload.get("path")
                if not p:
                    return self._json({"error": "path required"}, 400)
                try:
                    counts = brain.memory.import_all(p)
                except (OSError, ValueError) as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"message": f"imported {counts}"})
            if path == "/api/perceive":
                brain.perceive(payload.get("kind", "note"),
                               payload.get("content", ""))
                return self._json({"message": "perceived"})
            return self._json({"error": "not found"}, 404)

    return Handler


def _model_solver(brain: Brain):
    def solve(task: Task) -> str:
        result = brain.solve(task, record=False)
        return result.code
    return solve


def serve(brain: Brain, host: str = "0.0.0.0", port: int = 8000) -> None:
    handler = make_app(brain)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"🧠 MINICODER dashboard: http://{host}:{port}")
    try:
        httpd.serve_forever()
    finally:                      # pragma: no cover
        brain.close()
