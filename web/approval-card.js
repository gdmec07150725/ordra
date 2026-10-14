// 审批内容来自服务器检查点；网页只提交审批 ID、决定和允许修改的参数。
const ApprovalCard = {
  render(parent, checkpoint, onDecision, locked) {
    parent.querySelector('.approval-cards')?.remove();
    if (!checkpoint?.approvals?.length) return;
    const cards = document.createElement('div');
    cards.className = 'approval-cards';
    for (const approval of checkpoint.approvals) {
      const card = document.createElement('form');
      card.className = 'approval-card';
      card.addEventListener('submit', (event) => event.preventDefault());
      const heading = document.createElement('h3');
      heading.textContent = `${approval.decision ? '已处理' : '等待你确认'} · ${approval.title}`;
      const description = document.createElement('p');
      description.textContent = approval.description;
      card.append(heading, description);
      const fields = {};
      for (const field of approval.fields) {
        const label = document.createElement('label');
        label.textContent = field.label;
        const input = document.createElement(field.max_length > 100 ? 'textarea' : 'input');
        input.name = field.name;
        input.value = approval.arguments[field.name];
        input.maxLength = field.max_length;
        input.required = true;
        input.readOnly = true;
        if (input.tagName === 'TEXTAREA') input.rows = 3;
        label.append(input);
        card.append(label);
        fields[field.name] = input;
      }
      if (approval.decision) {
        const decision = document.createElement('p');
        decision.textContent = {approve: '已批准，等待继续执行。', edit: '已修改并批准，等待继续执行。', reject: '已拒绝，继续后助手会处理这个决定。'}[approval.decision];
        card.append(decision);
      } else {
        let editing = false;
        const actions = document.createElement('div');
        actions.className = 'approval-actions';
        const button = (text, action) => {
          const element = document.createElement('button');
          element.type = 'button';
          element.className = 'quiet-button';
          element.textContent = text;
          element.disabled = locked;
          element.addEventListener('click', action);
          actions.append(element);
          return element;
        };
        const approve = button('批准执行', () => {
          if (!card.reportValidity()) return;
          const decision = {decision: editing ? 'edit' : 'approve'};
          if (editing) decision.arguments = Object.fromEntries(Object.entries(fields).map(([key, value]) => [key, value.value.trim()]));
          onDecision(approval.approval_id, decision);
        });
        const edit = button('修改参数', () => {
          editing = !editing;
          for (const [key, input] of Object.entries(fields)) {
            input.readOnly = !editing;
            if (!editing) input.value = approval.arguments[key];
          }
          approve.textContent = editing ? '保存修改并执行' : '批准执行';
          edit.textContent = editing ? '取消修改' : '修改参数';
          if (editing) Object.values(fields)[0].focus({preventScroll: true});
        });
        button('拒绝执行', () => onDecision(approval.approval_id, {decision: 'reject'}));
        card.append(actions);
      }
      cards.append(card);
    }
    parent.append(cards);
  },
  renderAudit(parent, decisions) {
    parent.querySelector('.approval-audit')?.remove();
    if (!decisions.length) return;
    const audit = document.createElement('details');
    audit.className = 'approval-audit';
    const title = document.createElement('summary');
    title.textContent = '查看本次人工决定';
    audit.append(title);
    for (const decision of decisions) {
      const text = document.createElement('p');
      const label = {approve: '批准', edit: '修改并批准', reject: '拒绝'}[decision.decision];
      const describe = (arguments) => Object.entries(arguments).map(([name, value]) => `${{order_id: '订单号', reason: '退货原因'}[name] || name}：${value}`).join('；');
      text.textContent = `${label} · ${describe(decision.arguments)}${decision.decision === 'edit' ? `（修改前：${describe(decision.original_arguments)}）` : ''}${decision.reason ? `；说明：${decision.reason}` : ''}`;
      audit.append(text);
    }
    parent.append(audit);
  },
};
