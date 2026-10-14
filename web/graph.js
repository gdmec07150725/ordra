/* 每次实验独立保存；候选答案与原始模型上下文不由网页读取。 */
const graphUI = (() => {
  const $ = (id) => document.getElementById(id);
  const labels = {running: '执行中', awaiting_approval: '等待人工审批', completed: '已完成', unverified: '校验未通过', failed: '执行失败', interrupted: '已中断'};
  const stops = {model_budget: '执行模型请求已达到上限', tool_budget: '工具调用已达到上限', knowledge_budget: '知识检索已达到上限', context_budget: '当前完整上下文超过输入预算', review_context_budget: '完整证据超过校验输入预算', verification_budget: '校验请求已达到上限', verification_failed: '结果经修正后仍未通过校验'};
  const nodes = {reserve_model: '预留模型请求额度', model: '模型决策', reserve_tools: '预留工具额度', approval: '人工审批', tools: '执行工具', reserve_review: '预留校验额度', verify: '结果校验', finish: '生成最终结果'};
  let selected = null, generation = 0, timer = null, posting = false, configured = false;
  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function error(message = '') { $('error').textContent = message; $('error').hidden = !message; }
  async function request(url, body) {
    const response = await fetch(url, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '请求未通过，请检查输入或刷新记录。');
    return data;
  }
  function saveSelection(id) { try { localStorage.setItem('langgraph.selectedRun', id); } catch {} }
  function renderList(runs) {
    $('experiment-list').replaceChildren();
    if (!runs.length) $('experiment-list').append(element('li', '还没有实验记录', 'muted'));
    for (const run of runs) {
      const item = element('li'), button = element('button');
      button.type = 'button'; button.setAttribute('aria-current', String(run.run_id === selected));
      button.append(element('strong', run.question.length > 65 ? run.question.slice(0, 65) + '…' : run.question), element('small', `${labels[run.status] || run.status} · ${new Date(run.created_at).toLocaleString('zh-CN')}`));
      button.addEventListener('click', () => select(run.run_id));
      item.append(button); $('experiment-list').append(item);
    }
  }
  async function refreshList() {
    const data = await request('/api/graph/runs');
    configured = data.configured;
    $('model-label').textContent = configured ? `模型：${data.model}` : '请先在服务端配置聊天模型';
    $('start').disabled = posting || !configured;
    renderList(data.runs);
    return data.runs;
  }
  function render(run) {
    $('empty').hidden = true; $('run').hidden = false;
    $('run-status').textContent = labels[run.status] || run.status;
    $('run-id').textContent = run.run_id.slice(0, 8);
    $('run-question').textContent = run.question;
    $('metrics').replaceChildren();
    for (const [key, title] of [['model_requests', '执行模型请求'], ['verification_requests', '校验模型请求'], ['tool_calls', '工具调度次数'], ['knowledge_calls', '知识检索次数'], ['corrections', '修正轮数'], ['total_tokens', '已报告 tokens']]) {
      const metric = element('div'); metric.append(element('dt', title), element('dd', String(run.metrics?.[key] ?? 0))); $('metrics').append(metric);
    }
    $('progress').textContent = run.busy ? `正在运行，刷新或关闭网页不会取消执行。${run.next?.length ? '待执行：' + run.next.map(n => nodes[n] || n).join('、') : ''}` : run.status === 'awaiting_approval' ? '流程已保存，处理下面的审批后继续。' : `已保存 · 故障恢复 ${run.resume_count} / 3 次`;
    if (run.status === 'unverified') $('progress').textContent = `${stops[run.stop_reason] || '本次结果未能核实'}。可新建实验重试。`;
    $('recovery').hidden = !run.error;
    $('run-error').textContent = run.error + (run.error && !run.can_resume && !run.approvals?.length ? ' 本次实验无法继续，请新建实验。' : '');
    $('resume').hidden = !run.can_resume; $('resume').disabled = posting;
    ApprovalCard.render($('approvals'), {approvals: (run.approvals || []).map(item => ({...item, description: '批准后生成实验退货预览；此页面不会提交退货申请。'}))}, (approvalId, decision) => resume({approval_id: approvalId, ...decision}), posting || run.busy);
    const events = $('events'), atBottom = events.scrollHeight - events.clientHeight - events.scrollTop < 35;
    const position = events.scrollTop;
    events.replaceChildren();
    for (const event of run.events || []) {
      const row = element('li');
      row.append(element('strong', event.title || nodes[event.node] || event.node || '执行事件'), element('small', event.status || ''));
      const detail = event.summary || event.message || event.detail;
      if (detail) row.append(element('p', typeof detail === 'string' ? detail : JSON.stringify(detail)));
      if (event.tool) row.append(element('p', `${event.tool.name} · ${JSON.stringify(event.tool.arguments || {})} · ${event.tool.status || ''} · ${JSON.stringify(event.tool.result || {})}`));
      events.append(row);
    }
    events.scrollTop = atBottom ? events.scrollHeight : position;
    $('result').hidden = !['completed', 'unverified'].includes(run.status);
    $('answer').textContent = run.answer || (run.status === 'unverified' ? '本次结果未通过校验，未发布候选答案。请新建实验。' : '');
    $('previews').replaceChildren();
    for (const preview of run.return_previews || []) {
      const card = element('div', undefined, 'preview');
      card.append(element('strong', '实验退货预览 · 尚未提交'), element('p', `订单：${preview.order_id}`), element('p', `退货原因：${preview.reason}`));
      $('previews').append(card);
    }
    $('sources').replaceChildren();
    for (const source of run.sources || []) {
      const label = `${source.title || source.source || '知识来源'}${source.section ? ' · ' + source.section : ''}`;
      if (/^[a-f0-9]{12}$/i.test(source.document_id || '') && /^[a-f0-9]{12}$/i.test(source.version || '')) {
        const link = element('a', label); link.href = `/api/knowledge/${source.document_id}?version=${encodeURIComponent(source.version)}`; link.target = '_blank'; link.rel = 'noopener noreferrer'; $('sources').append(link);
      } else $('sources').append(element('p', label, 'muted'));
    }
  }
  async function load(id, version) {
    try {
      const run = await request(`/api/graph/runs/${encodeURIComponent(id)}`);
      if (version !== generation || id !== selected) return;
      render(run);
      await refreshList();
      if (version !== generation || id !== selected) return;
      if (run.busy) timer = setTimeout(() => load(id, version), 1000);
    } catch (failure) {
      if (version === generation) error(failure.message);
    }
  }
  async function select(id) {
    clearTimeout(timer); selected = id; const version = ++generation;
    saveSelection(id); error(); await load(id, version);
  }
  async function resume(body = {}) {
    if (posting || !selected) return;
    posting = true; error(); const id = selected;
    for (const button of $('approvals').querySelectorAll('button')) button.disabled = true;
    $('resume').disabled = true; $('start').disabled = true;
    let message = '';
    try { await request(`/api/graph/runs/${encodeURIComponent(id)}/resume`, body); }
    catch (failure) { message = failure.message; }
    finally {
      posting = false; $('start').disabled = !configured;
      // 审批期间允许切换实验；结束时解除当前记录上因本次提交施加的锁。
      if (selected) await select(selected);
      if (selected === id && message) error(message);
    }
  }
  $('experiment-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (posting || !configured) return;
    const question = $('question').value.trim(); if (!question) return;
    posting = true; $('start').disabled = true; error();
    try { const run = await request('/api/graph/runs', {question}); posting = false; await select(run.run_id); }
    catch (failure) { error(failure.message); }
    finally { posting = false; $('start').disabled = !configured; }
  });
  $('resume').addEventListener('click', () => resume());
  $('refresh').addEventListener('click', async () => { error(); try { if (selected) await select(selected); else await refreshList(); } catch (failure) { error(failure.message); } });
  for (const button of document.querySelectorAll('[data-question]')) button.addEventListener('click', () => { $('question').value = button.dataset.question; $('question').focus(); });
  async function start() {
    try {
      const runs = await refreshList();
      let saved; try { saved = localStorage.getItem('langgraph.selectedRun'); } catch {}
      const run = runs.find(row => row.run_id === saved) || runs[0];
      if (run) await select(run.run_id);
    } catch (failure) { error(failure.message); }
  }
  start();
  return {render, select};
})();
