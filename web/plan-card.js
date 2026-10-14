// 展示公开的执行步骤与状态，历史恢复使用已保存的 update_plan 工具结果。
window.PlanCard = {
  render(row, plan, unsaved = false) {
    row.querySelector('.agent-plan')?.remove();
    if (!plan?.steps?.length) return;
    const labels = {pending: '待执行', in_progress: '执行中', completed: '已完成', blocked: '待处理'};
    const card = document.createElement('section');
    card.className = 'agent-plan';
    const heading = document.createElement('h3');
    heading.textContent = `执行计划 · ${unsaved ? '本轮未保存' : labels[plan.status] || '执行中'}`;
    const list = document.createElement('ol');
    for (const step of plan.steps) {
      const item = document.createElement('li');
      item.dataset.state = step.status;
      const title = document.createElement('span');
      title.textContent = step.title;
      const state = document.createElement('small');
      state.textContent = labels[step.status] || '待执行';
      item.append(title, state);
      list.append(item);
    }
    card.append(heading, list);
    if (plan.note || unsaved) {
      const note = document.createElement('p');
      note.textContent = unsaved ? '本轮没有完成，以下次成功保存的结果为准。' : plan.status === 'blocked' ? '处理已暂停，具体说明见最终回复。' : plan.note;
      card.append(note);
    }
    row.append(card);
  },
};
