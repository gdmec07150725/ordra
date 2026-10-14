// 人工审核入口。模型没有审核工具，结果与原因只能通过此表单提交。
const list = document.querySelector('#request-list');
const detail = document.querySelector('#request-detail');
const heading = document.querySelector('#detail-heading');
const form = document.querySelector('#review-form');
const reasonInput = document.querySelector('#review-reason');
const statusLine = document.querySelector('#review-status');
const empty = document.querySelector('#queue-empty');
const refreshButton = document.querySelector('#refresh-reviews');
const filters = [...document.querySelectorAll('[data-filter]')];
const drafts = new Map();
let requests = [];
let selectedId = null;
let filter = 'pending';
let loading = false;
let saving = false;
let ready = false;

function status(text, error = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle('error', error);
}

function controls() {
  const locked = loading || saving;
  refreshButton.disabled = locked;
  filters.forEach((button) => { button.disabled = locked; button.setAttribute('aria-pressed', String(button.dataset.filter === filter)); });
  list.querySelectorAll('button').forEach((button) => { button.disabled = locked; });
  [...form.elements].forEach((field) => { field.disabled = locked; });
  document.querySelector('#submit-review').textContent = saving ? '正在保存…' : '提交审核';
}

async function api(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '审核信息不完整或格式不正确。');
  return data;
}

function current() { return requests.find((request) => request.request_id === selectedId); }

function rememberDraft() {
  if (selectedId && !form.hidden) drafts.set(selectedId, { reason: reasonInput.value, decision: form.elements.decision.value });
}

function renderDetail(request = current()) {
  detail.replaceChildren();
  form.hidden = true;
  form.reset();
  reasonInput.setCustomValidity('');
  if (!request) {
    heading.textContent = '选择一份申请';
    const hint = document.createElement('p');
    hint.textContent = '选择申请后，可以查看订单、退货原因和审核记录。';
    detail.append(hint);
    return;
  }
  heading.textContent = `${request.order_id} · ${request.status}`;
  const fields = document.createElement('dl');
  const entries = [
    ['申请编号', request.request_id], ['订单号', request.order_id], ['商品', request.order?.product_name],
    ['订单状态', request.order?.status], ['退货原因', request.reason],
    ['提交时间', new Date(request.confirmed_at).toLocaleString('zh-CN')],
    ['审核原因', request.review?.reason],
    ['审核时间', request.review?.reviewed_at ? new Date(request.review.reviewed_at).toLocaleString('zh-CN') : null],
  ];
  for (const [label, value] of entries) {
    if (!value) continue;
    const term = document.createElement('dt');
    const text = document.createElement('dd');
    term.textContent = label;
    text.textContent = value;
    fields.append(term, text);
  }
  detail.append(fields);
  if (request.review) {
    const note = document.createElement('p');
    note.textContent = request.review.decision === 'approved'
      ? '本地模拟审核已通过。后续寄回安排仍需人工处理，尚未执行退款。'
      : '本地模拟审核已拒绝，原因已记录。';
    detail.append(note);
  } else {
    form.hidden = false;
    const draft = drafts.get(request.request_id);
    if (draft) {
      reasonInput.value = draft.reason;
      form.elements.decision.value = draft.decision;
    }
  }
}

function renderList() {
  list.replaceChildren();
  empty.hidden = requests.length > 0;
  for (const request of requests) {
    const item = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'review-item';
    button.setAttribute('aria-current', String(request.request_id === selectedId));
    const title = document.createElement('strong');
    title.textContent = `${request.order_id} · ${request.status}`;
    const reason = document.createElement('span');
    reason.className = 'queue-reason';
    reason.textContent = request.reason;
    const number = document.createElement('small');
    number.textContent = request.request_id;
    button.append(title, reason, number);
    button.addEventListener('click', () => {
      if (loading || saving) return;
      rememberDraft();
      selectedId = request.request_id;
      renderList(); renderDetail(); controls();
    });
    item.append(button);
    list.append(item);
  }
}

async function loadRequests(preferredId = selectedId, nextFilter = filter) {
  rememberDraft();
  loading = true;
  controls();
  try {
    const data = await api(`/api/returns?status=${encodeURIComponent(nextFilter)}`);
    requests = data.requests;
    filter = nextFilter;
    selectedId = requests.some((item) => item.request_id === preferredId) ? preferredId : requests[0]?.request_id || null;
    for (const node of document.querySelectorAll('[data-count]')) node.textContent = data.counts[node.dataset.count] ?? 0;
    renderList(); renderDetail();
    ready = true;
  } finally {
    loading = false;
    controls();
  }
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (loading || saving) return;
  const request = current();
  if (!request || request.review) return;
  const decision = form.elements.decision.value;
  const reason = reasonInput.value.trim();
  reasonInput.setCustomValidity(reason ? '' : '请填写审核原因。');
  if (!form.reportValidity()) return;
  rememberDraft();
  saving = true;
  controls();
  status('正在保存审核结果…');
  let committed = false;
  try {
    const data = await api(`/api/returns/${encodeURIComponent(request.request_id)}/review`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: request.session_id, decision, reason }),
    });
    committed = true;
    const saved = { ...request, ...data.request };
    requests = requests.map((item) => item.request_id === saved.request_id ? saved : item);
    drafts.delete(request.request_id);
    renderDetail(saved);
    await loadRequests(request.request_id, decision);
    status('审核结果已保存，返回聊天后可以查询最新结果。');
  } catch (error) {
    // 网络响应丢失可能发生在提交之后，先读取服务器状态再允许人工重试。
    if (!committed) {
      try { await loadRequests(request.request_id, 'all'); } catch { /* 保留草稿供重试。 */ }
    }
    status(committed ? `审核已保存，但列表暂时刷新失败：${error.message}` : `${error.message} 请核对最新状态后再操作。`, true);
  } finally {
    saving = false;
    controls();
  }
});

reasonInput.addEventListener('input', () => { reasonInput.setCustomValidity(''); rememberDraft(); });
form.addEventListener('change', rememberDraft);
for (const button of filters) button.addEventListener('click', async () => {
  if (loading || saving) return;
  try { await loadRequests(null, button.dataset.filter); status('列表已更新。'); }
  catch (error) { status(error.message, true); }
});
refreshButton.addEventListener('click', async () => {
  if (loading || saving) return;
  try { await loadRequests(); status('列表已更新。'); }
  catch (error) { status(error.message, true); }
});
window.addEventListener('focus', async () => {
  if (!ready || loading || saving) return;
  try { await loadRequests(); } catch { /* 手动刷新时显示错误。 */ }
});
loadRequests().then(() => status('请选择申请进行人工审核。')).catch((error) => status(error.message, true));
