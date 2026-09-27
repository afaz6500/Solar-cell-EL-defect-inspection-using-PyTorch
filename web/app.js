const $ = id => document.getElementById(id);
function applyTheme(dark) {
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  $('theme-toggle').setAttribute('aria-pressed', String(dark));
  $('theme-toggle').title = dark ? 'Switch to light mode' : 'Switch to dark mode';
  $('theme-icon').textContent = dark ? '☀' : '☾';
  $('theme-label').textContent = dark ? 'Light mode' : 'Dark mode';
}
let savedTheme = null;
try { savedTheme = localStorage.getItem('solar-theme'); } catch (_) {}
applyTheme(savedTheme === 'dark');
$('theme-toggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme !== 'dark';
  applyTheme(dark);
  try { localStorage.setItem('solar-theme', dark ? 'dark' : 'light'); } catch (_) {}
});
const state = {files: [], current: null, job: null, jobs: [], view:'dashboard', previewUrl: null, previewImage: null, ready: false, busy: false, lastDashboard:0};
const text = (id, value) => $(id).textContent = value;
const label = value => String(value || '—').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const terminal = status => ['completed', 'cancelled', 'failed', 'interrupted'].includes(status);
const asset = path => `/api/jobs/${state.current}/asset?path=${encodeURIComponent(path)}`;
function notice(message) { text('notice', message); $('notice').hidden = !message; }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || 'The request could not be completed.');
  return body;
}
function config() { return {title:$('inspection-name').value.trim(), mode:$('mode').value, rows:Number($('rows').value), cols:Number($('cols').value), margin:0, margins:Object.fromEntries(['left','top','right','bottom'].map(side=>[side,Number($('margin-'+side).value)])), gap:Number($('gap').value), visuals:$('visuals').checked}; }
function canStart() { $('start').disabled = !state.files.length || !state.ready || state.busy; }
function chooseFiles(files) {
  if (state.busy) return;
  const list = [...files];
  if (!list.length) return;
  if (list.length > 50 || list.some(f => f.size > 20*1024*1024) || list.reduce((n,f) => n+f.size,0) > 100*1024*1024) {
    notice('Choose up to 50 images, no more than 20 MB each or 100 MB in total.'); return;
  }
  state.files = list;
  notice('');
  $('preview-select').replaceChildren(...list.map((f,i) => new Option(f.name, i)));
  text('file-summary', `${list.length} image${list.length === 1 ? '' : 's'} selected · ${(list.reduce((n,f) => n+f.size,0)/1024/1024).toFixed(1)} MB`);
  canStart(); previewFile();
}
function previewFile() {
  const file = state.files[Number($('preview-select').value) || 0];
  if (!file) return;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewImage = null;
  state.previewUrl = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    state.previewImage = img;
    $('preview-empty').hidden = true; $('preview-content').hidden = false;
    text('dimensions', `${img.naturalWidth} × ${img.naturalHeight} px`);
    drawPreview();
  };
  img.onerror = () => {
    $('preview-empty').hidden = false; $('preview-content').hidden = true;
    notice('This browser cannot preview this image. TIFF uploads are supported by the server; use PNG or JPEG if you need to check the grid before starting.');
  };
  img.src = state.previewUrl;
}
function drawPreview() {
  $('grid-options').hidden = $('mode').value !== 'panel';
  const img = state.previewImage;
  if (!img) return;
  const canvas = $('preview');
  const scale = Math.min(1, 1200/img.naturalWidth, 850/img.naturalHeight);
  canvas.width = Math.round(img.naturalWidth*scale); canvas.height = Math.round(img.naturalHeight*scale);
  const ctx = canvas.getContext('2d'); ctx.drawImage(img,0,0,canvas.width,canvas.height);
  const cfg = config();
  let caption = 'Individual cell · the original image is preserved for review.';
  if (cfg.mode === 'panel' && cfg.rows > 0 && cfg.rows <= 24 && cfg.cols > 0 && cfg.cols <= 24) {
    const mx = cfg.margins.left, my = cfg.margins.top;
    const cw = (img.naturalWidth-mx-cfg.margins.right)/cfg.cols, ch = (img.naturalHeight-my-cfg.margins.bottom)/cfg.rows;
    if (Object.values(cfg.margins).some(v=>!Number.isInteger(v)||v<0) || Math.min(cw,ch)*(1-cfg.gap/100)<16) {
      text('preview-caption','Margins and grid must leave cells at least 16 pixels wide and high.'); return;
    }
    ctx.strokeStyle = '#72efc5'; ctx.lineWidth = Math.max(1,canvas.width/750);
    for(let r=0;r<cfg.rows;r++) for(let c=0;c<cfg.cols;c++) {
      const gx=cw*cfg.gap/200, gy=ch*cfg.gap/200;
      const left=Math.round(mx+c*cw+gx), top=Math.round(my+r*ch+gy);
      const right=Math.round(mx+(c+1)*cw-gx), bottom=Math.round(my+(r+1)*ch-gy);
      ctx.strokeRect(left*scale,top*scale,(right-left)*scale,(bottom-top)*scale);
    }
    caption = `${cfg.rows} rows × ${cfg.cols} columns · ${cfg.rows*cfg.cols} cells per panel. Green outlines show the crops that will be analysed.`;
  }
  text('preview-caption',caption);
}
async function history() {
  state.jobs = await api('/api/jobs');
  renderLibrary();
}
async function openJob(id) {
  if (state.busy) return;
  state.current=id; localStorage.setItem('el-inspect-current',id);
  showView('inspection'); notice('');
  const job = await api(`/api/jobs/${id}`); if (state.current === id) {state.job=job;renderJob();}
  await history();
}
function renderJob() {
  const job=state.job;
  text('job-id', `INSPECTION ${job.id.slice(0,8).toUpperCase()} · ${new Date(job.created).toLocaleString()}`);
  text('job-title', job.title || job.sources[0]?.name || 'New inspection');
  text('job-stage', `${label(job.status)} · ${job.stage}`);
  text('progress-label', `${job.done} of ${job.total} cells processed${job.cancel_requested && !terminal(job.status) ? ' · stopping after the current cell' : ''}`);
  $('progress-fill').style.width = `${job.total ? job.done/job.total*100 : 0}%`;
  $('cancel').hidden=terminal(job.status); $('cancel').disabled=Boolean(job.cancel_requested);
  $('delete-job').hidden=!terminal(job.status);
  $('retry-rejected').hidden=!terminal(job.status)||!job.results.some(r=>r.prediction==='REJECTED');
  for(const format of ['pdf','csv','json']) $('export-'+format).href=`/api/jobs/${job.id}/export?format=${format}`;
  $('export-pdf').hidden=!terminal(job.status)||!job.results.length;
  const errors=job.errors.map(e => `${e.source}: ${e.message}`);
  if(job.status==='failed') errors.push(job.stage);
  text('job-errors',errors.join(' · ')); $('job-errors').hidden=!errors.length;
  text('metric-total',job.results.length);
  text('metric-defects',job.results.filter(r => !['normal','UNCERTAIN','REJECTED','ERROR'].includes(r.prediction)).length);
  text('metric-review',job.results.filter(needsReview).length);
  text('metric-high',job.results.filter(r => r.dsi != null && r.dsi>=60).length);
  renderResults();
}
function needsReview(r) { return r.ood_override || ['UNCERTAIN','REJECTED','ERROR'].includes(r.prediction); }
function renderResults() {
  if(!state.job) return;
  const query=$('search').value.toLowerCase(), filter=$('filter').value;
  const rows=state.job.results.filter(r => {
    const matches=filter==='all'||filter==='review'&&needsReview(r)||filter==='high'&&r.dsi!=null&&r.dsi>=60||filter==='normal'&&r.prediction==='normal'||filter==='defects'&&!needsReview(r)&&r.prediction!=='normal';
    return matches && `${r.source} ${r.prediction} ${r.likely_class||''}`.toLowerCase().includes(query);
  });
  text('result-count',`${rows.length} of ${state.job.results.length} findings shown`);
  $('results-empty').hidden=rows.length>0;
  text('results-empty',state.job.results.length ? 'No findings match this filter.' : terminal(state.job.status) ? 'No cell results are available for this inspection.' : 'Results will appear as cells are processed.');
  $('results').innerHTML=rows.map(r=>`<tr><td><div class="source-cell"><img loading="lazy" src="${asset(r.image)}" alt="Cell crop"><div class="source-name">${escapeHTML(r.source)}<small>${r.row ? `Row ${r.row} · Column ${r.col}` : 'Individual cell'}</small></div></div></td><td><span class="pill ${String(r.prediction).toLowerCase()}">${escapeHTML(label(r.prediction))}</span>${r.ood_override?'<small class="amber">OOD override · review required</small>':''}</td><td>${r.confidence==null?'—':r.confidence.toFixed(1)+'%'}</td><td>${r.dsi==null?'—':r.dsi.toFixed(1)}</td><td><button class="review-button" data-index="${r.index}">Inspect ↗</button></td></tr>`).join('');
}
function detail(index) {
  const r=state.job.results.find(row=>row.index===index); if(!r)return;
  text('detail-title',`${r.source}${r.row ? ` · R${r.row} C${r.col}`:''}`);
  const probs=Object.entries(r.probabilities||{}).sort((a,b)=>b[1]-a[1]);
  $('detail-body').innerHTML=`<div class="detail-grid"><img class="detail-image" src="${asset(r.image)}" alt="Original cell crop"><div><span class="pill ${r.prediction.toLowerCase()}">${escapeHTML(label(r.prediction))}</span>${r.ood_override?'<small class="amber">OOD override · review required</small>':''}<div class="detail-values"><div><small>Calibrated confidence</small><strong>${r.confidence==null?'—':r.confidence.toFixed(1)+'%'}</strong></div><div><small>DSI / 100</small><strong>${r.dsi==null?'—':r.dsi.toFixed(1)}</strong></div></div><p class="detail-reason">${escapeHTML(r.error||r.ood_reason||'')}</p>${r.prediction==='UNCERTAIN'?`<p class="detail-reason">Leading class: ${escapeHTML(label(r.likely_class))}. This remains an uncertain prediction.</p>`:''}${probs.slice(0,5).map(([name,value])=>`<div class="prob-row"><span>${escapeHTML(label(name))}</span><strong>${(value*100).toFixed(1)}%</strong></div>`).join('')}</div></div>${r.visual?`<h3>Visual evidence</h3><p class="hint">Original · Grad-CAM · damage mask · skeleton · localization. These are supporting visualizations, not verified defect boundaries.</p><img class="visual-output" src="${asset(r.visual)}" alt="Five-part visual analysis">`:`<p class="hint">${state.job.settings.visuals?'No visual output was produced for this cell.':'Visual analysis was not enabled for this inspection.'}</p>`}`;
  $('detail').showModal();
}
async function start() {
  if(state.busy || !state.files.length) return;
  const cfg=config();
  if(cfg.mode==='panel' && cfg.rows*cfg.cols*state.files.length>2000) {notice('This batch would exceed 2,000 cells. Use fewer panel images.');return;}
  state.busy=true;canStart(); notice('');
  let id=null;
  try {
    const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});id=job.id;
    for(let i=0;i<state.files.length;i++) {
      text('start',`Uploading ${i+1} of ${state.files.length}…`);
      await api(`/api/jobs/${id}/files?name=${encodeURIComponent(state.files[i].name)}`,{method:'PUT',body:state.files[i]});
    }
    await api(`/api/jobs/${id}/start`,{method:'POST'});
    state.busy=false; await openJob(id);
  } catch(e) {
    if(id) await api(`/api/jobs/${id}/cancel`,{method:'POST'}).catch(()=>{});
    notice(e.message);
  } finally {state.busy=false; text('start','Run inspection →');canStart();history().catch(()=>{});}
}
$('files').addEventListener('change',e=>chooseFiles(e.target.files));
$('preview-select').addEventListener('change',previewFile);
for(const id of ['mode','rows','cols','margin-left','margin-top','margin-right','margin-bottom','gap']) $(id).addEventListener('input',drawPreview);
for(const type of ['dragenter','dragover']) $('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop']) $('dropzone').addEventListener(type,e=>{e.preventDefault();$('dropzone').classList.remove('dragging');if(type==='drop')chooseFiles(e.dataTransfer.files);});
$('start').addEventListener('click',start);
$('new-job').addEventListener('click',newInspection);
$('cancel').addEventListener('click',async()=>{try{await api(`/api/jobs/${state.current}/cancel`,{method:'POST'});text('job-stage','Stopping after the current cell…');$('cancel').disabled=true;}catch(e){notice(e.message);}});
$('delete-job').addEventListener('click',async()=>{if(!confirm('Delete this inspection and its uploaded images, crops, and results?'))return;try{await api(`/api/jobs/${state.current}`,{method:'DELETE'});$('new-job').click();}catch(e){notice(e.message);}});
$('search').addEventListener('input',renderResults);$('filter').addEventListener('change',renderResults);
$('results').addEventListener('click',e=>{const button=e.target.closest('[data-index]');if(button)detail(Number(button.dataset.index));});
$('close-detail').onclick=()=>$('detail').close();
async function poll() {
  try {
    const id=state.current;
    if(id && !state.busy && (!state.job||!terminal(state.job.status))) {
      const job=await api(`/api/jobs/${id}`);if(state.current===id){state.job=job;renderJob();if(terminal(job.status))await history();}
    }
    if(!state.busy && ['dashboard','library','reports','activity'].includes(state.view) && Date.now()-state.lastDashboard>8000) await refreshDashboard();
  } catch(e) {notice(`Connection interrupted: ${e.message} Keep the server terminal open.`);}
  setTimeout(poll,1800);
}
async function init() {
  try {
    const status=await api('/api/status');state.ready=status.ready;
    text('model-status',status.ready?'10-class model · ready':'Model setup needed');canStart();
    if(!status.ready)notice('The trained 10-class checkpoint and calibration are required before inspection.');
    const profile=await api('/api/profile');setProfile(profile);
    await refreshDashboard();
    showView('dashboard');
  }catch(e){notice(e.message);text('model-status','Server unavailable');}
  poll();
}

const views={dashboard:['Dashboard','A clear view of your inspections and findings.'],library:['Inspections','Manage your image batches and review cell-level findings.'],reports:['Reports','Download inspection results as clear, shareable PDF reports.'],activity:['Activity log','A record of what happened in your workspace.'],settings:['Business profile','Add the details you want on your inspection reports.'],setup:['New inspection','Upload images, check the grid, and let the model assist your review.'],inspection:['Inspection results','Review predictions and create your inspection report.']};
function showView(view){
  state.view=view;
  for(const key of Object.keys(views)) $(key).hidden=key!==view;
  text('page-title',views[view][0]);text('page-description',views[view][1]);text('breadcrumb-title',views[view][0]);
  document.querySelectorAll('.nav-item').forEach(button=>{const active=button.dataset.view===view||button.dataset.view==='library'&&['setup','inspection'].includes(view);button.classList.toggle('active',active);if(active)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
  document.body.classList.remove('sidebar-mobile-open');syncSidebar();
  $('new-job').hidden=view==='setup';
}
function newInspection(){
  if(state.busy)return;
  state.current=null;state.job=null;state.files=[];state.previewImage=null;
  if(state.previewUrl)URL.revokeObjectURL(state.previewUrl);
  $('files').value='';$('inspection-name').value='';$('preview-content').hidden=true;$('preview-empty').hidden=false;
  text('file-summary','Up to 50 images / 100 MB per inspection');canStart();
  localStorage.removeItem('el-inspect-current');showView('setup');notice('');
}
const dateText=value=>new Date(value).toLocaleDateString(undefined,{day:'numeric',month:'short',year:'numeric'});
const jobName=job=>job.title||job.sources?.[0]?.name||'New inspection';
function jobRow(job){return `<tr><td><div class="table-name">${escapeHTML(jobName(job))}</div><small>#${job.id.slice(0,8).toUpperCase()}</small></td><td>${dateText(job.created_at||job.created)}</td><td><span class="pill ${job.status}">${label(job.status)}</span></td><td>${job.cells??job.done}${job.total!=null?` / ${job.total}`:''}</td><td><button class="text-button" data-job="${job.id}">View details →</button></td></tr>`;}
function renderLibrary(){
  const query=$('job-search').value.toLowerCase();
  const jobs=state.jobs.filter(j=>jobName(j).toLowerCase().includes(query));
  $('all-jobs').innerHTML=jobs.map(jobRow).join('');$('library-empty').hidden=jobs.length>0;
  text('library-empty',state.jobs.length?'No inspections match your search.':'No inspections yet. Start by uploading your first image.');
  const reports=state.jobs.filter(j=>terminal(j.status)&&j.done>0);
  $('report-jobs').innerHTML=reports.map(j=>`<tr><td><div class="table-name">${escapeHTML(jobName(j))}</div><small>#${j.id.slice(0,8).toUpperCase()}</small></td><td>${dateText(j.created)}</td><td>${j.done} / ${j.total} cells${j.status!=='completed'?'<small>Partial report</small>':''}</td><td><a class="secondary" href="/api/jobs/${j.id}/export?format=pdf" download>Download PDF ↓</a></td></tr>`).join('');
  $('reports-empty').hidden=reports.length>0;
}
async function refreshDashboard(){
  const data=await api('/api/dashboard');state.lastDashboard=Date.now();
  for(const key of ['inspections','active','cells','review']) text('dash-'+key,data.totals[key].toLocaleString());
  const max=Math.max(1,...data.trend.map(day=>day.count));
  $('trend-chart').replaceChildren(...data.trend.map(day=>{
    const column=document.createElement('div');column.className='chart-column';
    const count=document.createElement('span');count.className='chart-value';count.textContent=day.count;
    const bar=document.createElement('div');bar.className='chart-bar';bar.style.height=`${day.count/max*75}%`;
    const date=document.createElement('span');date.className='chart-date';date.textContent=new Date(day.date+'T12:00:00Z').toLocaleDateString(undefined,{weekday:'short'});
    column.append(count,bar,date);return column;
  }));
  $('trend-chart').setAttribute('aria-label',data.trend.map(day=>`${day.date}: ${day.count} inspections`).join('; '));
  $('findings-overview').replaceChildren(...[['Defect predictions','defects',''],['Normal predictions','normal','normal'],['Flagged for review','review','review']].map(([name,key,style])=>{
    const line=document.createElement('div');line.className='finding-line';const title=document.createElement('span');title.textContent=name;
    const value=document.createElement('strong');value.textContent=data.totals[key];const track=document.createElement('div');track.className='finding-track';
    const bar=document.createElement('div');bar.className='finding-bar '+style;bar.style.width=`${data.totals.cells?data.totals[key]/data.totals.cells*100:0}%`;track.append(bar);line.append(title,value,track);return line;
  }));
  $('recent-jobs').innerHTML=data.recent.map(jobRow).join('');$('dashboard-empty').hidden=data.recent.length>0;
  $('activity-list').innerHTML=data.activity.map(a=>`<li><span class="activity-symbol" aria-hidden="true">${a.event==='report_downloaded'?'↓':'◷'}</span><div><strong>${escapeHTML(label(a.event))}</strong><p>${escapeHTML(a.detail)}</p>${a.inspection_id?`<button class="text-button" data-job="${a.inspection_id}">Open inspection →</button>`:''}</div><time datetime="${a.created_at}">${dateText(a.created_at)}<br>${new Date(a.created_at).toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'})}</time></li>`).join('')||'<li>No activity yet. Your first inspection will appear here.</li>';
  await history();
}
function setProfile(profile){text('sidebar-name',profile.name);text('sidebar-company',profile.company||'Local workspace');$('profile-name').value=profile.name;$('profile-company').value=profile.company;}
$('profile-form').addEventListener('submit',async e=>{e.preventDefault();const button=e.target.querySelector('button');button.disabled=true;try{const profile=await api('/api/profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:$('profile-name').value,company:$('profile-company').value})});setProfile(profile);text('profile-saved','Profile saved. These details will appear on new reports.');notice('');}catch(error){notice(error.message);}finally{button.disabled=false;}});
document.addEventListener('click',e=>{
  const navigation=e.target.closest('[data-view]');if(navigation){if(state.busy)return;showView(navigation.dataset.view);notice('');refreshDashboard().catch(error=>notice(error.message));}
  const job=e.target.closest('[data-job]');if(job)openJob(job.dataset.job).catch(error=>notice(error.message));
  if(e.target.closest('[data-new]'))newInspection();
});
$('job-search').addEventListener('input',renderLibrary);
function syncSidebar(){const small=window.matchMedia('(max-width:760px)').matches;const expanded=small?document.body.classList.contains('sidebar-mobile-open'):!document.body.classList.contains('sidebar-collapsed');$('toggle-sidebar').setAttribute('aria-expanded',String(expanded));$('toggle-sidebar').setAttribute('aria-label',expanded?'Collapse sidebar':'Expand sidebar');$('sidebar').inert=small&&!expanded;}
$('toggle-sidebar').addEventListener('click',()=>{if(window.matchMedia('(max-width:760px)').matches)document.body.classList.toggle('sidebar-mobile-open');else{document.body.classList.toggle('sidebar-collapsed');localStorage.setItem('solar-sidebar-collapsed',String(document.body.classList.contains('sidebar-collapsed')));}syncSidebar();});
if(localStorage.getItem('solar-sidebar-collapsed')==='true')document.body.classList.add('sidebar-collapsed');
window.addEventListener('resize',syncSidebar);document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.body.classList.remove('sidebar-mobile-open');syncSidebar();}});syncSidebar();
init();

let marginDrag = null;
function marginPoint(e) {
  const r=$('preview').getBoundingClientRect(),i=state.previewImage;
  return {x:Math.round(Math.max(0,Math.min(1,(e.clientX-r.left)/r.width))*i.naturalWidth),y:Math.round(Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))*i.naturalHeight)};
}
function setMargins(m) {
  for(const [side,value] of Object.entries(m)) $('margin-'+side).value=value;
  drawPreview();
}
$('preview').addEventListener('pointerdown',e=>{
  if($('mode').value!=='panel'||!state.previewImage||state.busy||e.button!==0)return;
  marginDrag={start:marginPoint(e),previous:config().margins};
  $('preview').setPointerCapture(e.pointerId);e.preventDefault();
});
$('preview').addEventListener('pointermove',e=>{
  if(!marginDrag)return;
  const a=marginDrag.start,b=marginPoint(e),i=state.previewImage;
  setMargins({left:Math.min(a.x,b.x),top:Math.min(a.y,b.y),right:i.naturalWidth-Math.max(a.x,b.x),bottom:i.naturalHeight-Math.max(a.y,b.y)});
});
function finishMarginDrag(e) {
  if(!marginDrag)return;
  const c=config(),m=c.margins,i=state.previewImage;
  if(e.type==='pointercancel'||Math.min((i.naturalWidth-m.left-m.right)/c.cols,(i.naturalHeight-m.top-m.bottom)/c.rows)*(1-c.gap/100)<16)setMargins(marginDrag.previous);
  marginDrag=null;
  if($('preview').hasPointerCapture(e.pointerId))$('preview').releasePointerCapture(e.pointerId);
}
$('preview').addEventListener('pointerup',finishMarginDrag);
$('preview').addEventListener('pointercancel',finishMarginDrag);

$('retry-rejected').addEventListener('click',async()=>{
  const button=$('retry-rejected');button.disabled=true;
  try {
    await api(`/api/jobs/${state.current}/retry-rejected`,{method:'POST'});
    await openJob(state.current);
    notice('Processing rejected cells. OOD warnings will remain visible with the results.');
  } catch(e) {notice(e.message);} finally {button.disabled=false;}
});
