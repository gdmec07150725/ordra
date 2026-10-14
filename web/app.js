// 故意只用原生 fetch：以后换成 React，接口和 SSE 数据格式仍然可以沿用。
const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send");
const newChatButton = document.querySelector("#new-chat");
const sessionList = document.querySelector("#session-list");
const sessionCount = document.querySelector("#session-count");
const activeSessionTitle = document.querySelector("#active-session-title");
const messages = document.querySelector("#messages");
const chatArea = document.querySelector(".chat-area");
const scrollBottomButton = document.querySelector("#scroll-bottom");
const deleteDialog = document.querySelector("#delete-dialog");
const deleteTitle = document.querySelector("#delete-title");
const welcome = document.querySelector("#welcome");
const status = document.querySelector("#status");
const connection = document.querySelector("#connection");
const suggestions = [...document.querySelectorAll("[data-question]")];
const recoveryPanel = document.querySelector('#recovery-panel');
const recoveryDescription = document.querySelector('#recovery-description');
const resumeButton = document.querySelector('#resume-run');
const discardButton = document.querySelector('#discard-run');
const storageKey = "order-agent-session";
let sessionId = readPreferredSession();
let sessions = [];
let returnPreviews = [];
let busy = false;
let ready = false;
let navigating = false;
let configured = false;
let busyTimer;
let followLatest = true;
let scrollFrame;
let checkpoint = null;
let recoveryReply = null;

function isAtBottom() {
  return chatArea.scrollHeight - chatArea.clientHeight - chatArea.scrollTop <= 48;
}

function updateChatScroll(force = false) {
  if (force) followLatest = true;
  if (scrollFrame) return;
  scrollFrame = requestAnimationFrame(() => {
    scrollFrame = null;
    // 只在原本跟随底部时追随流式内容，上翻阅读后保留用户的位置。
    if (followLatest) chatArea.scrollTop = chatArea.scrollHeight;
    scrollBottomButton.hidden = isAtBottom();
  });
}

chatArea.addEventListener("scroll", () => {
  followLatest = isAtBottom();
  scrollBottomButton.hidden = followLatest;
}, { passive: true });
scrollBottomButton.addEventListener("click", () => updateChatScroll(true));
// 流式正文、历史切换及窗口/输入框大小变化都会重新计算可滚动空间。
const chatResizeObserver = new ResizeObserver(() => updateChatScroll());
chatResizeObserver.observe(messages);
chatResizeObserver.observe(chatArea);

function readPreferredSession() {
  try {
    // localStorage 跨标签页关闭保留；兼容升级前还在打开的旧标签页编号。
    return localStorage.getItem(storageKey) || sessionStorage.getItem(storageKey);
  } catch {
    return null; // 浏览器禁止存储时，仍可从服务端列表选择对话。
  }
}

function rememberSession() {
  try {
    if (sessionId) localStorage.setItem(storageKey, sessionId);
    else localStorage.removeItem(storageKey);
    sessionStorage.removeItem(storageKey); // 成功迁移后不再保留旧的选择。
  } catch {
    // 历史始终在 SQLite 中；浏览器存储只影响下次默认选中哪个会话。
  }
}

function setStatus(text, isError = false) {
  status.textContent = text;
  status.classList.toggle("error", isError);
}

function updateControls() {
  const locked = !ready || busy || navigating;
  input.disabled = locked || !configured || !sessionId || !!checkpoint;
  sendButton.disabled = locked || !configured || !sessionId || !!checkpoint || !input.value.trim();
  newChatButton.disabled = locked;
  sendButton.firstChild.textContent = busy ? "生成中 " : "发送 ";
  suggestions.forEach((button) => { button.disabled = locked || !configured || !sessionId || !!checkpoint; });
  sessionList.querySelectorAll("button").forEach((button) => { button.disabled = locked; });
  messages.querySelectorAll('.return-card button').forEach((button) => { button.disabled = locked || !!checkpoint; });
  messages.querySelectorAll('.approval-card button, .approval-card input, .approval-card textarea').forEach((element) => { element.disabled = locked; });
  recoveryPanel.hidden = !checkpoint || busy;
  recoveryDescription.textContent = checkpoint?.can_resume === false
    ? checkpoint.resume_reason || '本次任务无法继续，可放弃本次执行后重新提问。'
    : '上次执行尚未完成，可以从已保存的进度继续。放弃只清除此任务进度。';
  resumeButton.disabled = locked || !configured || !checkpoint?.can_resume;
  discardButton.disabled = locked || !checkpoint;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 204) return null; // DELETE 成功没有 JSON 响应体。
  const result = await response.json();
  if (!response.ok) {
    const message = typeof result.detail === "string" ? result.detail : "输入格式不正确，请检查问题内容。";
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return result;
}

function loadSession(id) {
  return requestJson(`/api/sessions/${id}?include_checkpoint=true`);
}

function addMessage(role, text) {
  welcome.hidden = true;
  const row = document.createElement("article");
  row.className = `message ${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "你" : "订单助手";
  const body = document.createElement("div");
  body.className = "message-body";
  // 模型输出按普通文本显示，避免生成内容被浏览器当作 HTML 执行。
  body.textContent = text;
  const note = document.createElement("div");
  note.className = "message-note";
  row.append(label, body, note);
  messages.append(row);
  return { row, body, note };
}

function describeContext(context) {
  if (!context) return "";
  const summary = context.summarized_turns ? `，更早 ${context.summarized_turns} 轮以摘要带入` : "";
  const usage = context.summary_usage;
  const cost = context.summary_calls
    ? (usage?.reported_rounds
      ? `\n整理摘要额外用量（已报告 ${usage.reported_rounds}/${context.summary_calls} 次）：输入 ${usage.prompt_tokens}，输出 ${usage.completion_tokens}，合计 ${usage.total_tokens} tokens。`
      : "\n本轮整理过摘要，模型未报告摘要 token 用量。")
    : "";
  return `最近一次模型请求：估算输入 ${context.estimated_input_tokens}/${context.input_limit} tokens（已预留回答与安全余量）；保留近期 ${context.kept_turns} 轮原文${summary}。完整历史仍可查看。${cost}`;
}

function renderSources(row, sources) {
  row.querySelector(".message-sources")?.remove();
  const valid = new Map();
  for (const source of sources || []) {
    if (source && /^[0-9a-f]{12}$/.test(source.document_id) && /^[0-9a-f]{12}$/.test(source.version)
      && typeof source.text === "string") valid.set(`${source.source}:${source.version}`, source);
  }
  if (!valid.size) return;
  const details = document.createElement("details");
  details.className = "message-sources";
  const summary = document.createElement("summary");
  summary.textContent = `检索参考资料 · ${valid.size} 段`;
  details.append(summary);
  for (const source of valid.values()) {
    const item = document.createElement("div");
    item.className = "source-item";
    const link = document.createElement("a");
    link.textContent = `${source.title} · ${source.section}`;
    link.href = `/api/knowledge/${source.document_id}?version=${source.version}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const quote = document.createElement("blockquote");
    quote.textContent = source.text;
    item.append(link, quote);
    details.append(item);
  }
  row.append(details);
}

function renderReturnCards(row, ids) {
  row.dataset.returnIds = JSON.stringify([...new Set(ids)]);
  ReturnCards.render(row, returnPreviews.filter((preview) => ids.includes(preview.preview_id)), changeReturn, !ready || busy || navigating || !!checkpoint);
}

function refreshReturnCards() {
  for (const row of messages.querySelectorAll('[data-return-ids]')) {
    renderReturnCards(row, JSON.parse(row.dataset.returnIds));
  }
}

async function refreshReturnStatus() {
  if (!ready || busy || navigating || !sessionId || document.hidden) return;
  const currentId = sessionId;
  try {
    const saved = await loadSession(currentId);
    if (sessionId !== currentId || busy || navigating) return;
    if (JSON.stringify(saved.checkpoint || null) !== JSON.stringify(checkpoint) || saved.busy) {
      applySession(saved, true);
      return;
    }
    returnPreviews = saved.return_previews || [];
    refreshReturnCards(); // 回到聊天页只更新申请状态，保留输入草稿和聊天内容。
  } catch { /* 状态刷新失败时不打断输入，下次查询或刷新仍从服务器读取。 */ }
}

async function changeReturn(previewId, action) {
  if (!ready || busy || navigating || !sessionId || checkpoint) return;
  const currentId = sessionId;
  const draft = input.value;
  const position = chatArea.scrollTop;
  const following = followLatest;
  navigating = true;
  updateControls();
  setStatus(action === 'confirm' ? '正在提交模拟退货申请…' : '正在取消预览…');
  const restore = (saved) => {
    applySession(saved, true);
    input.value = draft;
    followLatest = following;
    if (!following) chatArea.scrollTop = position;
    updateChatScroll();
  };
  try {
    await requestJson(`/api/sessions/${currentId}/return-previews/${previewId}/${action}`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(action === 'confirm' ? { confirmed: true } : {}),
    });
    restore(await loadSession(currentId));
    setStatus(action === 'confirm' ? '模拟申请已提交，申请编号和状态已保存。' : '预览已取消。');
  } catch (error) {
    // 响应丢失也可能已经提交；先从服务器恢复，重试仍使用同一个预览 ID。
    try { restore(await loadSession(currentId)); } catch { /* 保留页面，允许刷新重试。 */ }
    setStatus(`${error.message} 请以卡片的已保存状态为准，也可以刷新确认。`, true);
  } finally {
    navigating = false;
    updateControls();
  }
}

function renderHistory(history, scrollToLatest = true) {
  messages.replaceChildren();
  welcome.hidden = false;
  let sources = [];
  const knowledgeCalls = new Set();
  const returnCalls = new Set();
  const planCalls = new Set();
  const delegationCalls = new Set();
  let savedPlan = null;
  let decisions = [];
  let returnIds = [];
  for (const message of history) {
    if (message.role === "user") {
      sources = [];
      knowledgeCalls.clear();
      returnCalls.clear();
      planCalls.clear();
      delegationCalls.clear();
      savedPlan = null;
      decisions = [];
      returnIds = [];
    }
    for (const call of message.tool_calls || []) {
      if (call.function?.name === "search_knowledge") knowledgeCalls.add(call.id);
      if (call.function?.name === 'update_plan') planCalls.add(call.id);
      if (call.function?.name === 'delegate_tasks') delegationCalls.add(call.id);
      if (['prepare_return_request', 'get_return_request'].includes(call.function?.name)) returnCalls.add(call.id);
    }
    if (message.role === "tool" && knowledgeCalls.has(message.tool_call_id)) {
      try {
        const result = JSON.parse(message.content);
        if (Array.isArray(result.results)) sources.push(...result.results);
      } catch { /* 旧版或损坏的工具内容不影响正文恢复。 */ }
    }
    if (message.role === 'tool' && delegationCalls.has(message.tool_call_id)) {
      try {
        const result = JSON.parse(message.content);
        for (const task of result.delegations || []) {
          for (const evidence of task.evidence || []) {
            if (evidence.name === 'search_knowledge' && !evidence.result?.error && Array.isArray(evidence.result?.results)) sources.push(...evidence.result.results);
          }
        }
      } catch { /* 子任务原始记录不影响问答正文恢复。 */ }
    }
    if (message.role === 'tool') {
      try {
        const result = JSON.parse(message.content);
        if (result._human_decision) decisions.push(result._human_decision);
      } catch { /* 旧记录中的非 JSON 内容不影响阅读。 */ }
    }
    if (message.role === 'tool' && returnCalls.has(message.tool_call_id)) {
      try {
        const result = JSON.parse(message.content);
        const preview = result.preview || result.request;
        if (preview?.preview_id) returnIds.push(preview.preview_id);
      } catch { /* 旧工具消息不影响历史显示。 */ }
    }
    // system、tool 和助手的工具请求留在服务端历史中，聊天区显示问答正文。
    if (message.role === 'tool' && planCalls.has(message.tool_call_id)) {
      try {
        const result = JSON.parse(message.content);
        if (result.plan) savedPlan = result.plan;
      } catch { /* 旧记录不影响聊天正文恢复。 */ }
    }
    if (message.role === 'tool') {
      try {
        const result = JSON.parse(message.content);
        if (result._recovery?.stopped && result._recovery.plan) savedPlan = result._recovery.plan;
      } catch { /* 旧记录不影响停止状态恢复。 */ }
    }
    if ((message.role === "user" || message.role === "assistant") && !message.tool_calls && message.content) {
      const reply = addMessage(message.role, message.content);
      if (message.role === "assistant") {
        renderSources(reply.row, sources);
        renderReturnCards(reply.row, returnIds);
        PlanCard.render(reply.row, savedPlan);
        ApprovalCard.renderAudit(reply.row, decisions);
      }
    }
  }
  updateChatScroll(scrollToLatest);
}

function renderSessions() {
  sessionList.replaceChildren();
  sessionCount.textContent = sessions.length;
  for (const session of sessions) {
    const row = document.createElement("li");
    row.className = "session-row";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-item";
    button.dataset.sessionId = session.session_id;
    button.setAttribute("aria-current", session.session_id === sessionId ? "true" : "false");
    button.title = session.title;
    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = session.title;
    const time = document.createElement("time");
    time.dateTime = session.created_at;
    time.textContent = new Date(session.created_at).toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
    button.append(title, time);
    button.addEventListener("click", () => openSession(session.session_id));
    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "delete-session";
    deleteButton.textContent = "删除";
    deleteButton.setAttribute("aria-label", `删除对话：${session.title}`);
    deleteButton.addEventListener("click", () => confirmDeleteSession(session));
    row.append(button, deleteButton);
    sessionList.append(row);
  }
  const selected = sessions.find((session) => session.session_id === sessionId);
  document.querySelector('#runs-link').href = '/runs' + (sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '');
  activeSessionTitle.textContent = selected?.title || "新对话";
  activeSessionTitle.title = activeSessionTitle.textContent;
  updateControls();
}

async function refreshSessions() {
  const result = await requestJson("/api/sessions");
  sessions = result.sessions;
  renderSessions();
}

function applySession(session, preserveScroll = false) {
  const preserveDraft = preserveScroll && session.session_id === sessionId;
  sessionId = session.session_id;
  rememberSession();
  busy = session.busy;
  checkpoint = session.checkpoint || null;
  recoveryReply = null;
  returnPreviews = session.return_previews || [];
  if (!preserveDraft) input.value = "";
  renderHistory(session.messages, !preserveScroll);
  if (checkpoint) {
    addMessage('user', checkpoint.question);
    recoveryReply = addMessage('assistant', busy ? '这项任务正在执行…' : '本次执行尚未完成，以下是已保存的进度。');
    PlanCard.render(recoveryReply.row, checkpoint.plan);
    ApprovalCard.render(recoveryReply.row, checkpoint, decideApproval, !ready || busy || navigating);
    updateChatScroll(!preserveScroll);
  }
  renderSessions();
  setStatus(!configured ? "请在服务端配置密钥并重启，再刷新此页面。" : busy ? "这段对话正在生成回复，请稍候…" : checkpoint ? checkpoint.resume_reason || "有未完成的任务，请选择继续执行或放弃。" : session.messages.length > 1 ? "已恢复保存的对话，可以继续追问。" : "可以开始对话了。", !configured);
  clearTimeout(busyTimer);
  if (busy) {
    // 另一标签页可能正在生成，等它结束后再恢复当前会话的输入。
    const pendingId = sessionId;
    busyTimer = setTimeout(async () => {
      try {
        const updated = await loadSession(pendingId);
        if (sessionId === pendingId && !navigating) applySession(updated, true);
      } catch (error) {
        if (sessionId !== pendingId || navigating) return;
        if (error.status === 404) {
          // 另一标签页可能在生成结束后删除会话，不能让旧 busy 永久锁住页面。
          busy = false;
          sessionId = null;
          rememberSession();
          renderHistory([]);
          navigating = true;
          try {
            await refreshSessions();
            if (sessions.length) applySession(await loadSession(sessions[0].session_id));
            else await addSession();
            setStatus("原对话已在其他页面删除，已切换到可用对话。");
          } catch (recoveryError) {
            setStatus(`原对话已删除，恢复列表失败：${recoveryError.message} 请新建对话或刷新页面。`, true);
          } finally {
            navigating = false;
          }
        } else {
          setStatus(`恢复会话失败：${error.message} 请刷新页面重试。`, true);
        }
      } finally {
        updateControls();
      }
    }, 700);
  }
}

async function addSession() {
  // “新对话”调用创建接口；只有创建成功后才切换，原会话不做任何清空。
  const created = await requestJson("/api/sessions", { method: "POST" });
  sessions.unshift({ ...created, title: "新对话" });
  applySession({ ...created, messages: [], busy: false });
}

async function openSession(nextId) {
  if (!ready || busy || navigating || nextId === sessionId) return;
  navigating = true;
  updateControls();
  try {
    // 先成功读取目标历史，再改变当前编号，避免加载失败后把问题发错会话。
    const session = await loadSession(nextId);
    applySession(session);
  } catch (error) {
    setStatus(`切换失败：${error.message}`, true);
  } finally {
    navigating = false;
    updateControls();
    input.focus({ preventScroll: true });
  }
}

function confirmDeleteSession(session) {
  if (!ready || busy || navigating || deleteDialog.open) return;
  deleteTitle.textContent = session.title;
  deleteDialog.returnValue = "cancel";
  deleteDialog.showModal();
  deleteDialog.addEventListener("close", () => {
    if (deleteDialog.returnValue === "delete") deleteSession(session);
  }, { once: true });
}

async function deleteSession(session) {
  if (!ready || busy || navigating) return;
  navigating = true;
  updateControls();
  let deleted = false;
  try {
    try {
      await requestJson(`/api/sessions/${session.session_id}`, { method: "DELETE" });
    } catch (error) {
      if (error.status !== 404) throw error; // 其他标签页已经删除时，同步移出列表。
    }
    deleted = true;
    sessions = sessions.filter((item) => item.session_id !== session.session_id);
    if (session.session_id === sessionId) {
      // 删除成功才清除当前选择；后续加载失败时也不能继续向旧编号发送。
      sessionId = null;
      rememberSession();
      clearTimeout(busyTimer);
      input.value = "";
      renderHistory([]);
      renderSessions();
      if (sessions.length) applySession(await loadSession(sessions[0].session_id));
      else await addSession();
    } else {
      renderSessions(); // 删除其他会话时，保留当前消息、草稿与滚动位置。
    }
    setStatus("对话已删除。" + (!configured ? "请在服务端配置密钥后再开始聊天。" : busy ? "当前对话正在生成回复，请稍候…" : ""));
  } catch (error) {
    setStatus(deleted ? `对话已删除，但加载下一段失败：${error.message} 请从列表选择或新建对话。` : `删除失败：${error.message}`, true);
  } finally {
    navigating = false;
    updateControls();
  }
}

async function initialize() {
  try {
    const health = await requestJson("/api/health");
    configured = health.configured;
    connection.textContent = configured ? "服务已连接 · 本机保存" : "模型尚未配置";
    await refreshSessions();
    // 编号被清理或已失效时，选中已有列表的最新会话；没有任何记录才新建。
    const selected = sessions.find((session) => session.session_id === sessionId) || sessions[0];
    if (selected) applySession(await loadSession(selected.session_id));
    else await addSession();
    ready = true;
  } catch (error) {
    setStatus(`连接失败：${error.message} 请确认服务已启动，再刷新页面。`, true);
    connection.textContent = "服务未连接";
  } finally {
    updateControls();
  }
}

// fetch 的一次 read 可能拿到半个事件，也可能拿到多个事件。
// TextDecoder 保留跨片段的 UTF-8 字节；buffer 保留尚未收齐的 SSE 文本。
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
      let separator;
      while ((separator = /\r?\n\r?\n/.exec(buffer))) {
        const block = buffer.slice(0, separator.index);
        buffer = buffer.slice(separator.index + separator[0].length);
        let name = "message";
        const data = [];
        for (const line of block.split(/\r?\n/)) {
          if (line.startsWith("event:")) name = line.slice(6).trim();
          if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
        }
        if (data.length) onEvent(name, JSON.parse(data.join("\n")));
      }
      if (done) break;
    }
  } finally {
    // error 事件或解析异常提前退出时，也通知浏览器关闭这条请求。
    await reader.cancel();
    reader.releaseLock();
  }
}

async function submitChat(resume = false) {
  const run = resume ? checkpoint : null;
  if (resume ? !run?.can_resume : checkpoint) return;
  const question = run ? run.question : input.value.trim();
  if (!question || busy || navigating || !ready || !configured || !sessionId) return;
  const requestSessionId = sessionId;
  busy = true;
  input.value = "";
  updateControls();
  if (!run) addMessage("user", question);
  const reply = run && recoveryReply ? recoveryReply : addMessage("assistant", "正在准备回复…");
  reply.row.querySelector('.approval-cards')?.remove();
  reply.body.textContent = run ? '正在恢复已保存的执行…' : '正在准备回复…';
  updateChatScroll(true);
  let text = "";
  let completed = false;
  let paused = false;
  const decisions = [];
  let currentPlan = run?.plan || null;
  const returnIds = [];
  setStatus("正在处理你的问题…");

  try {
    // 浏览器只发送问题和会话编号；密钥与模型完整历史由 Python 管理。
    const response = await fetch(run ? `/api/sessions/${requestSessionId}/resume/stream` : "/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(run ? {run_id: run.run_id} : { session_id: requestSessionId, message: question }),
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(typeof result.detail === "string" ? result.detail : "输入格式不正确。");
    }
    await readEvents(response, (name, data) => {
      if (name === 'resumed') {
        currentPlan = data.plan;
        PlanCard.render(reply.row, currentPlan);
        returnIds.push(...(data.return_ids || []));
        setStatus('已恢复保存的进度，正在继续执行…');
      } else if (name === "summarizing") {
        setStatus("正在整理较早的对话，保留后续追问所需信息…");
        reply.body.textContent = "正在整理对话记忆…";
      } else if (name === "context") {
        reply.note.textContent = describeContext(data);
      } else if (name === "round") {
        text = "";
        reply.body.textContent = "正在生成回复…";
      } else if (name === "delta") {
        text += data.text;
        reply.body.textContent = text;
        setStatus("正在回复…");
      } else if (name === 'plan' || name === 'plan_pending') {
        currentPlan = data.plan;
        PlanCard.render(reply.row, currentPlan);
        setStatus(name === 'plan_pending' ? '仍有步骤未完成，正在继续处理…'
          : currentPlan.status === 'completed' ? '计划步骤已完成，正在整理结果…'
          : currentPlan.status === 'blocked' ? '需要补充信息或确认，正在整理说明…' : '执行计划已更新，正在继续处理…');
      } else if (name === "tool_start") {
        setStatus(data.batch_size > 1 ? `正在处理 ${data.batch_size} 项信息查询…`
          : data.name === 'read_skill_resource' ? '正在读取处理细则…' : data.name === 'load_skill' ? '正在读取处理指引…' : data.name === 'discover_tools' ? '正在选择所需工具…' : data.name === 'update_plan' ? '正在安排执行步骤…' : data.name === "search_history" ? "正在查找之前的对话原文…" : data.name === "search_knowledge" ? "正在查阅业务资料…" : data.name === 'prepare_return_request' ? '正在准备退货申请预览…' : data.name === 'get_return_request' ? '正在查询退货申请…' : "正在查询订单…");
      } else if (name === 'tool_execution') {
        if (data.retries) setStatus('已完成查询重试，正在整理可确认的信息…');
      } else if (name === 'recovery') {
        setStatus(data.kind === 'stopped' ? '连续尝试没有新进展，正在保存停止说明…'
          : data.kind === 'replanned' ? '已根据查询结果调整后续步骤…'
          : data.kind === 'resolved' ? '已取得新信息，继续处理…'
          : data.kind === 'waiting_input' ? '需要你补充信息，正在整理问题…'
          : '正在根据执行反馈调整处理方式…');
      } else if (name === "tool_result") {
        if (data.result._human_decision) decisions.push(data.result._human_decision);
        if (data.name === 'update_plan') {
          if (data.result.error) setStatus('正在校正执行计划…');
        } else if (data.name === 'load_skill') {
          setStatus(data.result.error ? '处理指引暂不可用，正在调整处理方式…' : '处理指引已读取，正在继续处理…');
        } else if (data.name === "search_history") {
          const result = data.result;
          setStatus(result.error ? "历史检索未完成，正在处理…"
            : result.semantic_status === "unavailable" ? "已使用关键词查找历史，正在整理回复…"
            : result.budget_limited ? "历史片段受上下文预算限制，正在整理回复…"
            : result.results.length ? `已找到 ${result.results.length} 条历史片段，正在整理回复…`
            : "没有找到匹配的历史原文，正在整理回复…");
        } else if (data.name === "search_knowledge") {
          const result = data.result;
          setStatus(result.error ? "资料暂时无法查阅，正在处理…"
            : result.results.length ? "已找到相关资料，正在整理回复…"
            : "没有找到相关资料，正在整理回复…");
        } else if (['prepare_return_request', 'get_return_request'].includes(data.name)) {
          const preview = data.result.preview || data.result.request;
          if (preview?.preview_id) returnIds.push(preview.preview_id);
          setStatus(data.result.error ? '申请信息需要核实，正在整理回复…' : '申请信息已获取，正在整理回复…');
        } else {
          setStatus(data.result.error ? '部分信息暂时无法查询，正在整理回复…' : "查询完成，正在整理回复…");
        }
      } else if (name === 'delegation_start') {
        reply.body.textContent = `已委派 ${data.tasks.length} 项子任务，正在分别查询…`;
        setStatus('正在协调订单查询与资料检索…');
      } else if (name === 'specialist_start' || name === 'specialist_done') {
        const agent = {orders:'订单查询 Agent', knowledge:'知识检索 Agent'}[data.agent] || '子 Agent';
        setStatus(name === 'specialist_start' ? `${agent} 正在处理子任务…`
          : data.status === 'completed' ? `${agent} 已完成，正在汇总结果…` : `${agent} 尚未完成，正在核对可用结果…`);
      } else if (name === 'verifying') {
        reply.body.textContent = '正在核对处理结果…';
        setStatus(`正在第 ${data.number} 次核对目标与执行结果…`);
      } else if (name === 'correcting') {
        reply.body.textContent = '正在补查和修正回复…';
        setStatus(`正在第 ${data.number} 次修正，最多修正 2 次…`);
      } else if (name === 'verification') {
        setStatus(data.verdict === 'pass' ? '结果核对通过，正在保存回复…'
          : data.verdict === 'unverified' ? '结果尚未核实，正在保存说明…' : '发现需要核对的信息，正在继续处理…');
      } else if (name === 'paused') {
        paused = true;
        checkpoint = data.checkpoint;
        recoveryReply = reply;
        reply.body.textContent = '请核对下面的信息，决定是否执行。';
        PlanCard.render(reply.row, checkpoint.plan);
        ApprovalCard.render(reply.row, checkpoint, decideApproval, true);
        setStatus('正在等待你的决定，当前没有继续调用模型或执行工具。');
      } else if (name === "done") {
        completed = true;
        checkpoint = null;
        recoveryReply = null;
        reply.body.textContent = data.answer;
        currentPlan = data.plan;
        PlanCard.render(reply.row, currentPlan);
        reply.row.querySelector('.approval-cards')?.remove();
        ApprovalCard.renderAudit(reply.row, data.human_decisions || decisions);
        renderSources(reply.row, data.sources);
        returnPreviews = returnPreviews.map((preview) => preview.state === 'pending' ? { ...preview, state: 'expired', status: '预览已失效' } : preview);
        returnPreviews.push(...(data.return_previews || []));
        refreshReturnCards();
        renderReturnCards(reply.row, returnIds);
        const usage = data.usage.reported_rounds
          ? `回答阶段已报告 ${data.usage.reported_rounds} 次模型请求用量：输入 ${data.usage.prompt_tokens}，输出 ${data.usage.completion_tokens}，合计 ${data.usage.total_tokens} tokens。`
          : "模型未报告 token 用量。";
        const retrieval = data.retrieval?.calls
          ? `\n本轮检索历史 ${data.retrieval.calls} 次，带回 ${data.retrieval.returned_messages} 条原文片段。`
          : "";
        const knowledge = data.knowledge?.calls
          ? `\n本轮查阅业务资料 ${data.knowledge.calls} 次，带回 ${data.knowledge.returned_chunks} 个片段。` : "";
        const execution = data.execution?.calls
          ? `\n本轮工具调用 ${data.execution.calls} 项，最多同时处理 ${data.execution.max_parallel} 项，自动重试 ${data.execution.retries} 次。` : '';
        const verification = data.verification?.status === 'not_applicable' ? '\n本次仅保存系统停止说明，未发布未经核实的业务结论。' : data.verification?.enabled
          ? `\n结果核对 ${data.verification.checks} 次，修正 ${data.verification.corrections} 次；${data.verification.status === 'passed' ? '核对通过' : '尚未核实'}。`
            + (data.verification.usage?.reported_rounds ? `核对额外用量 ${data.verification.usage.total_tokens} tokens。` : '') : '';
        const delegation = data.delegation?.tasks?.length
          ? `\n已委派 ${data.delegation.tasks.length} 项子任务，子 Agent 共请求模型 ${data.delegation.requests} 次。`
            + (data.delegation.usage?.reported_rounds ? `已报告用量 ${data.delegation.usage.total_tokens} tokens。` : '') : '';
        reply.note.textContent = `${describeContext(data.context)}\n${usage}${retrieval}${knowledge}${execution}${delegation}${verification}`;
        setStatus(data.recovery?.stopped ? '自动尝试已停止，已有记录已保存。可以补充信息或重新提问。'
          : data.verification?.status === 'unverified' ? '结果尚未核实，说明已保存。可以补充信息后继续。' : '回复已保存，可以继续追问。');
      } else if (name === "error") {
        throw new Error(data.message);
      }
    });
    if (!completed && !paused) throw new Error("连接已结束，但没有收到完整回复。");
  } catch (error) {
    reply.row.classList.add("failed");
    busy = false;
    PlanCard.render(reply.row, currentPlan, true);
    if (!text) reply.body.textContent = "这次回复没有完成。";
    reply.note.textContent = "本轮未完成；刷新后以服务端已保存的对话为准。";
    try {
      applySession(await loadSession(requestSessionId));
      setStatus(checkpoint ? `${error.message} ${checkpoint.can_resume ? '已保存的进度可点击“继续执行”。' : checkpoint.resume_reason}`
        : `${error.message} 已重新读取服务端保存的内容。`, true);
    } catch {
      setStatus(`${error.message} 暂时无法读取执行进度，请刷新页面后继续。`, true);
    }
  } finally {
    if (paused) {
      try { await refreshSessions(); } catch { /* 检查点已保存，稍后刷新仍能恢复。 */ }
    }
    if (completed) {
      try {
        await refreshSessions(); // 第一条问题保存后，列表标题由服务端摘要更新。
        const saved = await loadSession(requestSessionId);
        returnPreviews = saved.return_previews || [];
        refreshReturnCards();
      } catch {
        setStatus("回复已保存，但历史列表暂时刷新失败。可以稍后刷新页面。", true);
      }
    }
    if (completed || paused) busy = false;
    updateControls();
    input.focus({ preventScroll: true });
  }
}

form.addEventListener('submit', (event) => { event.preventDefault(); submitChat(); });
async function decideApproval(approvalId, decision) {
  if (!checkpoint || busy || navigating || !ready) return;
  const currentId = sessionId;
  const runId = checkpoint.run_id;
  navigating = true;
  updateControls();
  let shouldResume = false;
  try {
    await requestJson(`/api/sessions/${currentId}/checkpoint/decision`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({run_id: runId, approval_id: approvalId, ...decision}),
    });
    applySession(await loadSession(currentId), true);
    shouldResume = checkpoint?.can_resume && !busy;
  } catch (error) {
    // 422 保留用户正在编辑的参数；冲突或网络状态不明时读取服务器决定。
    if (error.status !== 422) {
      try { applySession(await loadSession(currentId), true); } catch { /* 保留卡片，稍后刷新。 */ }
    }
    setStatus(`决定未确认：${error.message}`, true);
  } finally {
    navigating = false;
    updateControls();
  }
  if (shouldResume) await submitChat(true);
}
resumeButton.addEventListener('click', () => submitChat(true));
discardButton.addEventListener('click', async () => {
  if (busy || navigating || !checkpoint) return;
  const run = checkpoint;
  const currentId = sessionId;
  navigating = true;
  updateControls();
  try {
    await requestJson(`/api/sessions/${currentId}/checkpoint/discard`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({run_id: run.run_id}),
    });
    applySession(await loadSession(currentId));
    setStatus('本次未完成的执行已放弃，可以输入新问题。');
  } catch (error) {
    try { applySession(await loadSession(currentId)); } catch { /* 等待用户刷新。 */ }
    setStatus(`未能确认放弃：${error.message}`, true);
  } finally {
    navigating = false;
    updateControls();
  }
});

input.addEventListener("input", updateControls);
input.addEventListener("keydown", (event) => {
  // isComposing 避免中文输入法选字时按 Enter 意外发送。
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
suggestions.forEach((button) => button.addEventListener("click", () => {
  input.value = button.dataset.question;
  form.requestSubmit();
}));
newChatButton.addEventListener("click", async () => {
  if (busy || navigating || !ready) return;
  navigating = true;
  updateControls();
  try {
    await addSession();
    setStatus("已创建新对话，之前的对话可在历史列表中继续打开。");
  } catch (error) {
    setStatus(`创建失败：${error.message}`, true);
  } finally {
    navigating = false;
    updateControls();
    input.focus({ preventScroll: true });
  }
});

window.addEventListener('focus', refreshReturnStatus);
document.addEventListener('visibilitychange', refreshReturnStatus);
updateControls();
initialize();
