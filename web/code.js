(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const statuses = {running:'正在分析',completed:'分析完成',failed:'未通过验证',cancelled:'已停止',interrupted:'执行已中断'};
  const stages = {generate:'生成分析代码',execute:'在隔离环境执行',verify:'独立核对计算结果'};
  let config, selected = '', timer, lastVersion = '', starting = false, busy = false, pollGeneration = 0;
  const node = (tag, text, cls) => { const el = document.createElement(tag); if (text !== undefined) el.textContent = text; if (cls) el.className = cls; return el; };
  async function api(url, options) {
    const response = await fetch(url, {cache:'no-store', ...options});
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '请求参数不正确，请检查后重试。');
    return body;
  }
  function error(message='') { $('error').textContent = message; $('error').hidden = !message; }
  function enable() { $('start').disabled = !config?.configured || !config?.sandbox.ready || busy || starting; }
  function table(target, headings, rows) {
    target.replaceChildren();
    const head = node('thead'), headRow = node('tr'); headings.forEach(value => headRow.append(node('th', value))); head.append(headRow); target.append(head);
    const body = node('tbody'); rows.forEach(values => { const tr = node('tr'); values.forEach(value => tr.append(node('td', String(value)))); body.append(tr); }); target.append(body);
  }
  function preview() {
    const item = config.datasets.find(item => item.id === $('dataset').value);
    table($('preview'), ['订单号','商品','渠道','状态'], item.preview.map(row => config.fields.map(key => row[key])));
    $('row-count').textContent = `共 ${item.rows} 行，展示前 ${item.preview.length} 行。完整数据由执行环境计算，不全部塞入模型请求。`;
  }
  function codeDetails(title, code) { const details = node('details'); details.append(node('summary', title), node('pre', code)); return details; }
  function render(record) {
    const expanded = new Set([...$('attempts').querySelectorAll('details[open]')].map(el => el.dataset.key));
    $('empty').hidden = true; $('detail').hidden = false;
    $('status').textContent = statuses[record.status] || record.status;
    $('description').textContent = `${record.dataset_title} · 按${record.config.group_by === 'product' ? '商品' : '渠道'}分组 · 关注阈值 ${Math.round(record.config.threshold * 100)}%`;
    $('cancel').hidden = record.status !== 'running'; $('cancel').disabled = false;
    const last = record.events.at(-1);
    $('progress').textContent = record.status === 'running' ? `${stages[last?.stage] || '准备数据'}${last?.attempt ? ` · 第 ${last.attempt} 轮` : ''}` : '可展开查看代码与执行反馈。';
    $('run-error').textContent = record.error; $('run-error').hidden = !record.error;
    const metrics = record.result?.metrics || [...record.events].reverse().find(event => event.metrics)?.metrics;
    $('metrics').replaceChildren();
    if (metrics) {
      const pairs = [['模型请求', metrics.model_requests],['代码执行',metrics.executions],['已报告 tokens',metrics.usage_status === 'unavailable' ? '未提供' : `${metrics.total_tokens}${metrics.usage_status === 'partial' ? '（不完整）' : ''}`]];
      if (record.status !== 'running') pairs.push(['耗时', record.elapsed_ms > 0 ? `${(record.elapsed_ms/1000).toFixed(1)} 秒` : '未记录']);
      pairs.forEach(([label,value]) => { const group=node('div'); group.append(node('dt',label),node('dd',String(value))); $('metrics').append(group); });
    }
    const attempts = new Map(); record.events.forEach(event => { if(event.record) attempts.set(event.attempt,event.record); });
    $('attempts').replaceChildren();
    for (const [attempt, item] of attempts) {
      const card=node('section',undefined,'attempt'); card.append(node('h3',`第 ${attempt} 轮`));
      if(item.note) card.append(node('p',item.note,'muted'));
      if(item.injected) card.append(node('p','练习模式：本轮由系统注入错误，用于观察恢复过程，不代表模型自行产生了该错误。','feedback'));
      if(item.generated_code) { const details=codeDetails('查看模型生成的代码',item.generated_code); details.dataset.key=`${attempt}-generated`; details.open=expanded.has(details.dataset.key); card.append(details); }
      if(item.injected) { const details=codeDetails('查看本轮实际执行的练习代码',item.code); details.dataset.key=`${attempt}-injected`; details.open=expanded.has(details.dataset.key); card.append(details); }
      if(item.execution) card.append(node('p',`${item.execution.ok ? '代码已运行' : '代码执行失败'} · ${item.execution.elapsed_ms} ms`,item.execution.ok?'pass':'failure'));
      if(item.execution?.stdout || item.execution?.stderr) card.append(codeDetails('程序输出',`${item.execution.stdout}\n${item.execution.stderr}`));
      if(item.feedback.length) card.append(node('p',item.feedback.join('\n'),'feedback'));
      if(item.verification?.passed) card.append(node('p','计算结果已通过独立校验。','pass'));
      $('attempts').append(card);
    }
    $('report').hidden = record.status !== 'completed' || !record.result?.verification?.passed;
    $('downloads').replaceChildren();
    if (!$('report').hidden) {
      const report=record.result.report;
      $('report-summary').textContent=`已核对全部 ${report.total_orders} 条模拟订单。`;
      table($('report-table'),['分组','订单数','退款数','退款率','需关注'],report.groups.map(row=>[row.name,row.orders,row.refunded,`${(row.refund_rate*100).toFixed(2)}%`,row.refund_rate>=record.config.threshold?'是':'否']));
      for (const [name,label] of [['report.md','下载分析报告'],['report.csv','下载统计表']]) {
        const link=node('a',label,'quiet-button'); link.href=`/api/code/runs/${record.run_id}/artifacts/${name}`; link.download=name; $('downloads').append(link);
      }
    }
  }
  async function refreshList(generation = pollGeneration) {
    const data = await api('/api/code/runs');
    if (generation !== pollGeneration) return [];
    busy=data.runs.some(run=>run.status==='running'); enable(); $('run-list').replaceChildren();
    if(!data.runs.length) $('run-list').append(node('li','还没有分析记录。','muted'));
    data.runs.forEach(run=>{const li=node('li'),button=node('button',run.dataset_title,`run-button${selected===run.run_id?' selected':''}`);button.type='button';button.append(node('small',`${statuses[run.status]} · ${new Date(run.created_at).toLocaleString()}`));button.onclick=()=>select(run.run_id);li.append(button);$('run-list').append(li);});
    return data.runs;
  }
  async function poll() {
    clearTimeout(timer); const id=selected, generation=++pollGeneration;
    try {
      if(id) { const record=await api(`/api/code/runs/${id}`); if(id!==selected || generation!==pollGeneration)return; if(record.updated_at!==lastVersion){render(record);lastVersion=record.updated_at;} }
      await refreshList(generation);
    } catch(e) { if(generation===pollGeneration)error(e.message); }
    if(generation===pollGeneration && !document.hidden) timer=setTimeout(poll,busy?1000:4000);
  }
  function select(id) { selected=id;lastVersion='';history.replaceState(null,'',`#${id}`);error();poll(); }
  $('dataset').onchange=()=>config&&preview();
  $('exercise').onchange=()=>{$('exercise-note').textContent=$('exercise').value==='normal'?'统计各分组的订单数与退款率，并标出达到关注阈值的分组。':'首次执行代码时故意注入错误；后续由模型根据真实反馈重新编写代码。记录中会明确标注注入。';};
  $('refresh').onclick=()=>{error();poll();};
  $('analysis-form').onsubmit=async event=>{
    event.preventDefault(); if(starting)return; starting=true;enable();error();
    try { const run=await api('/api/code/runs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_id:$('dataset').value,group_by:$('group-by').value,threshold:Number($('threshold').value)/100,exercise:$('exercise').value})});busy=true;select(run.run_id); }
    catch(e){error(e.message);} finally{starting=false;enable();}
  };
  $('cancel').onclick=async()=>{ $('cancel').disabled=true; try{await api(`/api/code/runs/${selected}/cancel`,{method:'POST'});lastVersion='';await poll();}catch(e){error(e.message);$('cancel').disabled=false;} };
  document.addEventListener('visibilitychange',()=>{if(document.hidden)clearTimeout(timer);else poll();});
  (async()=>{try{config=await api('/api/code/config');preview();$('ready').textContent=config.sandbox.ready?(config.configured?`运行环境就绪 · ${config.model}`:'请先配置模型服务。'):config.sandbox.message;const runs=await refreshList();const hash=location.hash.slice(1);selected=runs.some(run=>run.run_id===hash)?hash:runs[0]?.run_id||'';await poll();}catch(e){error(e.message);$('ready').textContent='连接失败，请刷新页面重试。';}})();
})();
