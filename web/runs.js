const picker = document.querySelector('#session-picker');
const list = document.querySelector('#run-list');
const detail = document.querySelector('#detail');
const statusLine = document.querySelector('#status');
const refresh = document.querySelector('#refresh');
const more = document.querySelector('#load-more');
let runs = [], runId = null, cursor = null, listVersion = 0, detailVersion = 0;
const statusNames = {running:'执行中', awaiting_approval:'等待人工决定', ready:'已决定，等待继续', waiting_input:'等待补充信息', completed:'已完成', unverified:'结果尚未核实', failed:'执行失败', interrupted:'执行中断', discarded:'已放弃'};
const eventNames = {started:'任务开始', resumed:'从检查点继续', model_request:'模型请求', model_response:'模型响应', summary_request:'整理历史摘要', summary_result:'摘要已保存', verification_request:'开始核对结果', verification_result:'结果核对完成', result_correction:'开始修正', unverified:'已保存未核实说明', plan:'执行计划更新', tool_result:'工具结果', approval_requested:'请求人工确认', human_decision:'人工决定已保存', completed:'答复与结果已提交', waiting_input:'已保存补充信息问题', failed:'执行失败', interrupted:'执行中断', discarded:'用户放弃执行', truncated:'事件详情已达保存上限'};
Object.assign(eventNames, {delegated_task:'委派子任务', specialist_request:'子 Agent 请求模型', specialist_done:'子任务结束'});
eventNames.context_budget = '模型上下文预算';
eventNames.tool_discovery = '工具按需加载';
eventNames.skill_load = '技能按需加载';
eventNames.skill_resource = '技能引用读取';
statusNames.stalled = '无新进展，已停止';
eventNames.recovery = '执行反馈与调整';
eventNames.stalled = '已停止自动尝试';
const recoveryNames = {feedback:'收到执行反馈', replanned:'已调整后续计划', resolved:'已恢复进展', waiting_input:'等待补充信息', stopped:'连续无进展，停止尝试'};
const agentNames = {orders:'订单查询 Agent', knowledge:'知识检索 Agent'};
const roleNames = {actor:'主管', specialist:'子 Agent', summary:'历史摘要', verification:'结果校验'};
const componentNames = {system:'系统提示词', tools:'工具定义', history_summary:'历史摘要', recent_messages:'近期完整消息', current_messages:'当前轮消息', tool_results:'工具结果', skills:'技能指引', request_data:'结构化任务资料', format:'请求格式余量'};
function element(tag, text, className) { const node = document.createElement(tag); if (text != null) node.textContent = text; if (className) node.className = className; return node; }
function renderBudget(data) {
  const box = element('div', null, 'budget');
  box.append(element('p', `${roleNames[data.role] || data.role} · ${data.status === 'blocked' ? '预算已拦截' : '请求准备就绪'} · 估算输入 ${data.estimated_input_tokens} / ${data.input_limit} tokens`, data.status === 'blocked' ? 'error' : ''));
  box.append(element('p', `整理前 ${data.before_estimated_tokens} → 整理后 ${data.estimated_input_tokens}，减少 ${data.saved_tokens}；输出预留 ${data.output_reserve}，安全余量 ${data.safety_tokens}。`));
  const parts = element('dl', null, 'budget-parts');
  for (const [key, value] of Object.entries(data.components || {})) {
    if (value > 0) { const part = element('div'); part.append(element('dt', componentNames[key] || key), element('dd', String(value))); parts.append(part); }
  }
  box.append(parts);
  if (data.tool_names) box.append(element('p', `本次提供的工具：${data.tool_names.length ? data.tool_names.join('、') : '无'}。`, 'budget-note'));
  if (data.components?.request_data) box.append(element('p', '结构化任务资料包含这次摘要或校验所需的历史、候选答复及证据，按实际请求包装合计。', 'budget-note'));
  return box;
}
async function read(url) { const response = await fetch(url); const data = await response.json(); if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '读取失败'); return data; }
function base() { return `/api/sessions/${encodeURIComponent(picker.value)}/runs`; }
function showError(error) { statusLine.textContent = error.message; statusLine.classList.add('error'); }
function renderList() {
  list.replaceChildren();
  for (const run of runs) {
    const row = element('li');
    const button = element('button', run.question);
    button.type = 'button'; button.setAttribute('aria-current', String(run.run_id === runId));
    button.append(element('small', `${statusNames[run.status] || run.status} · ${new Date(run.created_at).toLocaleString('zh-CN')}`));
    button.addEventListener('click', () => loadDetail(run.run_id)); row.append(button); list.append(row);
  }
  more.hidden = !cursor;
}
async function loadDetail(id) {
  const ticket = ++detailVersion, sid = picker.value;
  runId = id; renderList(); detail.replaceChildren(element('p', '正在读取执行详情…', 'empty'));
  try {
    const run = await read(`${base()}/${encodeURIComponent(id)}`);
    if (ticket !== detailVersion || sid !== picker.value) return;
    detail.replaceChildren(element('h2', run.question), element('p', `${statusNames[run.status] || run.status} · ${run.metrics.model || '模型未记录'}`));
    const metrics = element('div', null, 'metrics');
    for (const [label, value] of [['主管模型请求', run.metrics.model_requests || 0], ['子任务数', run.metrics.delegated_tasks || 0], ['子 Agent 模型请求', run.metrics.subagent_requests || 0], ['摘要请求', run.metrics.summary_requests || 0], ['结果核对', run.metrics.verification_requests || 0], ['结果修正', run.metrics.corrections || 0], ['已报告 tokens', run.metrics.reported_requests ? run.metrics.total_tokens : '未报告'], ['业务工具调用', run.metrics.tool_calls || 0], ['工具调度尝试', run.metrics.tool_attempts || 0], ['自动重试', run.metrics.tool_retries || 0]]) {
      const card = element('div', null, 'metric'); card.append(element('span', label), element('strong', String(value))); metrics.append(card);
    }
    detail.append(metrics);
    detail.append(element('p', `调整计划 ${run.metrics.replans || 0} 次；连续无进展 ${run.metrics.stalled_batches || 0} 次${run.metrics.recovery_stopped ? '，已停止自动尝试' : ''}。`, 'budget-note'));
    detail.append(element('p', `工具模式：${run.metrics.tool_mode === 'mcp' ? 'MCP · 按需加载' : '本地直接调用'}；发现工具 ${run.metrics.tool_discoveries || 0} 次。`, 'budget-note'));
    detail.append(element('p', `已加载 ${run.metrics.skills_loaded || 0} 个技能；尝试读取 ${run.metrics.skill_loads || 0} 次。`, 'budget-note'));
    detail.append(element('p', `已读取 ${run.metrics.skill_resources_loaded || 0} 份技能引用；尝试读取 ${run.metrics.skill_resource_reads || 0} 次。`, 'budget-note'));
    const budgets = run.events.filter(event => event.kind === 'context_budget');
    detail.append(element('p', budgets.length
      ? `已记录 ${budgets.length} 次上下文预检。以下为保守估算，实际计费用量以模型报告为准；准备就绪不代表请求已完成。`
      : '这条记录暂无上下文用量明细。新请求及继续执行后会记录，旧数据不补造。', 'budget-note'));
    const timeline = element('ol', null, 'timeline');
    for (const event of run.events) {
      const row = element('li'); const data = event.data;
      row.append(element('h3', `${eventNames[event.kind] || event.kind}${data.agent ? ` · ${agentNames[data.agent] || data.agent}` : ''}${data.name ? ` · ${data.name}` : ''}${data.round || data.number ? ` · 第 ${data.round || data.number} 次` : ''}`));
      const time = element('time', new Date(event.created_at).toLocaleString('zh-CN')); time.dateTime = event.created_at; row.append(time);
      if (data.message) row.append(element('p', data.message));
      if (event.kind === 'recovery') {
        row.append(element('p', `${recoveryNames[data.kind] || data.kind}：${data.reason || ''}`));
        if (data.changed_steps?.length) row.append(element('p', `调整步骤：${data.changed_steps.join('、')}。`));
      }
      if (event.kind === 'context_budget') row.append(renderBudget(data));
      if (event.kind === 'skill_load' && !data.result?.error) {
        const skill = data.result || {};
        row.append(element('p', `${skill.status === 'already_loaded' ? '本轮已加载，无需重复读取' : '完整指引已加载'}；版本 ${(skill.version || '').slice(0, 12)}；正文包装估算 ${skill.estimated_tokens || 0} tokens。`));
      }
      if (event.kind === 'skill_resource') {
        row.append(element('p', data.path || '未指定引用文件'));
        if (!data.result?.error) {
          const reference = data.result || {};
          row.append(element('p', `${reference.status === 'already_loaded' ? '本轮已读取，复用已有内容' : '引用已读取'}；版本 ${(reference.version || '').slice(0, 12)}；内容包装估算 ${reference.estimated_tokens || 0} tokens。`));
        }
      }
      if (event.kind === 'tool_discovery') {
        row.append(element('p', data.error || `当前已加载：${(data.loaded_names || []).join('、') || '无'}。`));
        if (data.tools) row.append(element('p', data.tools.map(tool => `${tool.name}：${tool.source === 'mcp' ? '本地 MCP 服务' : '应用内工具'}`).join('；')));
      }
      if (event.kind === 'delegated_task') row.append(element('p', data.task));
      if (event.kind === 'specialist_done') row.append(element('p', `状态：${({completed:'已完成', failed:'未完成', needs_input:'需要补充信息'})[data.status] || data.status}；模型请求 ${data.rounds} 次。`));
      if (event.kind === 'verification_result') row.append(element('p', data.verdict === 'pass' ? '核对通过。' : `发现 ${data.issue_count || 1} 项信息需要核对。`));
      if (event.kind === 'tool_result') row.append(element('p', `状态：${({ok:'成功', error:'失败', skipped:'跳过'})[data.status] || data.status}；调度尝试 ${data.attempts} 次；耗时 ${data.elapsed_ms} ms`));
      if (event.kind === 'tool_result' && data.source) row.append(element('p', `来源：${data.source === 'mcp' ? '本地 MCP 服务' : '应用内工具'}`));
      if (event.kind === 'plan') row.append(element('p', `状态：${({in_progress:'执行中', completed:'已完成', blocked:'等待补充信息'})[data.status] || data.status}${data.note ? `；${data.note}` : ''}`));
      if (data.result?.error) row.append(element('p', data.result.error, 'error'));
      const extra = element('details'); extra.append(element('summary', '查看记录详情'), element('pre', JSON.stringify(data, null, 2))); row.append(extra); timeline.append(row);
    }
    detail.append(timeline);
  } catch (error) { if (ticket === detailVersion && sid === picker.value) { detail.replaceChildren(element('p', error.message, 'error')); showError(error); } }
}
async function loadRuns(append = false) {
  const ticket = ++listVersion, sid = picker.value;
  if (!sid) return;
  refresh.disabled = true; more.disabled = true; statusLine.classList.remove('error');
  const selected = runId;
  if (!append) { ++detailVersion; runs = []; runId = null; detail.replaceChildren(element('p', '正在读取…', 'empty')); renderList(); }
  try {
    const data = await read(base() + (append && cursor ? `?before=${cursor}` : ''));
    if (ticket !== listVersion || sid !== picker.value) return;
    runs = append ? [...runs, ...data.runs] : data.runs; cursor = data.next_cursor;
    statusLine.textContent = runs.length ? `已读取 ${runs.length} 次执行。执行中的记录可点击刷新更新。` : '这段对话还没有执行记录。记录从本次升级后的请求开始保存。';
    renderList();
    if (!append && runs.length) await loadDetail(runs.some(r => r.run_id === selected) ? selected : runs[0].run_id);
    if (!runs.length) detail.replaceChildren(element('p', '发送一个新问题后，可在这里查看 Agent 的执行过程。', 'empty'));
  } catch (error) { if (ticket === listVersion && sid === picker.value) showError(error); }
  finally { if (ticket === listVersion) { refresh.disabled = false; more.disabled = false; } }
}
picker.addEventListener('change', () => { runId = null; cursor = null; loadRuns(); });
refresh.addEventListener('click', () => loadRuns()); more.addEventListener('click', () => loadRuns(true));
(async () => {
  try {
    const data = await read('/api/sessions');
    for (const session of data.sessions) { const option = element('option', session.title); option.value = session.session_id; picker.append(option); }
    const preferred = new URLSearchParams(location.search).get('session_id');
    if (data.sessions.some(s => s.session_id === preferred)) picker.value = preferred;
    if (data.sessions.length) await loadRuns();
    else { statusLine.textContent = '还没有对话，请先返回聊天页面发送问题。'; refresh.disabled = true; }
  } catch (error) { showError(error); }
})();
