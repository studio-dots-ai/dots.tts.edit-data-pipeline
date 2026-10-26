const content = document.querySelector('#content');
const errorBox = document.querySelector('#error');
let selected = null;
let refreshing = false;
const esc = value => String(value ?? '').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const number = value => Number(value || 0).toLocaleString();
async function api(path, options = {}) {
  const response = await fetch(path, {headers: {'Content-Type':'application/json'}, ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail));
  return body;
}
function runUrl(suffix) { return `/api/runs/${encodeURIComponent(selected.date)}/${encodeURIComponent(selected.name)}${suffix}`; }
function metric(label, value) { return `<article class="metric"><span class="subtle">${esc(label)}</span><strong>${esc(value)}</strong></article>`; }
async function renderRun() {
  if (!selected) return;
  const data = await api(runUrl('/dashboard'));
  const parts = data.parquet;
  const span = (parts.latest_created_at - parts.first_created_at) / 3600;
  const rate = span > 0 ? Math.round(parts.total_rows / span) : null;
  const path = data.run.config_path;
  const workerCommand = `python -m speech_edit.runtime.local_runner --run-config '${String(path).replaceAll("'", "'\\''")}'`;
  content.innerHTML = `<div class="section-row"><div><span class="badge">${esc(data.run.pipeline)}</span><h2>${esc(data.run.name)}</h2><div class="subtle">${esc(data.run.run_dir)}</div></div><div><span class="badge">${esc(data.status)}</span> <button id="stop">Stop</button> <button id="resume">Resume</button></div></div>
  <div class="metrics">${metric('Rows written',number(parts.total_rows))}${metric('Parquet parts',number(parts.total_parts))}${metric('Audio hours',parts.total_duration_sec == null ? '—' : (parts.total_duration_sec/3600).toFixed(2))}${metric('Rows / hour',rate == null ? '—' : number(rate))}${metric('Workers online',data.workers.filter(w=>w.active).length)}${metric('Output size',`${(parts.total_bytes/1048576).toFixed(1)} MB`)}</div>
  <h3>Stages</h3><div class="stages">${data.stages.map(s=>`<article class="stage"><h3>${esc(s.stage)}</h3><strong>${s.active_workers} online</strong><p class="subtle">${s.stale_workers} unresponsive · ${s.max_workers} configured</p><div class="subtle">${esc(s.input_queue || '—')}<br>↓<br>${esc(s.output_queues.join(', ') || (s.writes_parquet ? 'Parquet output' : '—'))}</div></article>`).join('')}</div>
  <h3>Queue backlog</h3><div class="card">${data.queues.map(q=>`<div class="queue"><span>${esc(q.queue_name)}</span><div class="bar"><i style="width:${q.total ? 100*(q.counts.done||0)/q.total : 0}%"></i></div><span class="subtle">${number(q.ready)} ready / ${number(q.leased)} leased / ${number(q.counts.done)} done / ${number(q.counts.dead)} failed</span></div>`).join('')}</div>
  <h3>Workers</h3><div class="card"><table><thead><tr><th>Worker</th><th>Stage</th><th>Status</th><th>Heartbeat age</th></tr></thead><tbody>${data.workers.map(w=>`<tr><td>${esc(w.worker_id)}</td><td>${esc(w.stage)}</td><td>${esc(w.stale ? 'unresponsive' : w.status)}</td><td>${Math.round(w.age_sec)} s</td></tr>`).join('')}</tbody></table></div>
  <h3>Preview Parquet data</h3><div class="card"><table><thead><tr><th>File</th><th>Rows</th><th>Size</th><th>Preview</th></tr></thead><tbody>${(parts.parts || []).map(p=>`<tr><td>${esc(p.output_path)}</td><td>${number(p.row_count)}</td><td>${(p.byte_count/1048576).toFixed(1)} MB</td><td><button data-file="${esc(p.output_path)}">Top row</button></td></tr>`).join('')}</tbody></table><div id="preview"></div></div>
  <details><summary>Start workers</summary><p>Run on an allocated machine using the configured Python environment. Resume only enables processing; it does not launch processes.</p><pre>${esc(workerCommand)}</pre><p>For multiple machines, start individual stages with <code>speech_edit.runtime.entry</code>.</p></details>`;
  document.querySelector('#stop').onclick = () => control('stop');
  document.querySelector('#resume').onclick = () => control('resume');
  document.querySelectorAll('[data-file]').forEach(button => button.onclick = async () => {
    try {
      const query = new URLSearchParams({file:button.dataset.file, mode:'top', limit:'1'});
      const result = await api(runUrl(`/parquet-preview?${query}`));
      const samples = result.samples.map(s => {
        const params = new URLSearchParams({date:selected.date,name:selected.name,file:s.file,row_index:s.row_index});
        return `<a href="/sample.html?${params}" target="_blank">${esc(s.text || s.audio_id || 'View sample')}</a>`;
      });
      document.querySelector('#preview').innerHTML = samples.join('<br>') || 'No rows';
    } catch(error) { errorBox.textContent = error.message; }
  });
}
async function control(action) { try { await api(runUrl('/'+action), {method:'POST'}); await renderRun(); } catch(error) { errorBox.textContent=error.message; } }
async function refresh() {
  if (refreshing) return;
  refreshing=true;
  try {
    errorBox.textContent='';
    const data=await api('/api/runs');
    const list=document.querySelector('#runs'); list.replaceChildren();
    for (const run of data.runs) {
      const button=document.createElement('button'); button.className='run'+(selected?.run_id===run.run_id?' selected':'');
      button.innerHTML=`<strong>${esc(run.name)}</strong><span class="subtle">${esc(run.pipeline)} · ${esc(run.status)}</span>`;
      button.onclick=()=>{selected=run;refresh();}; list.append(button);
    }
    if (!selected && data.runs.length) selected=data.runs[0];
    await renderRun();
  } catch(error) { errorBox.textContent=error.message; }
  finally {refreshing=false;}
}
document.querySelector('#refresh').onclick=refresh;
document.querySelector('#submit').onsubmit=async event=>{
  event.preventDefault();
  try {
    const result=await api('/api/runs/submit',{method:'POST',body:JSON.stringify({run_config:document.querySelector('#config').value,run_id:document.querySelector('#runId').value || undefined})});
    errorBox.textContent=`Run created: ${result.config_path}. Start workers to begin processing.`;
    await refresh();
  } catch(error) {errorBox.textContent=error.message;}
};
api('/api/configs').then(data=>{document.querySelector('#configOptions').innerHTML=data.configs.map(c=>`<option value="${esc(c.path)}">${esc(c.name)}</option>`).join('');}).catch(error=>{errorBox.textContent=error.message;});
refresh();
setInterval(refresh, 10000);
