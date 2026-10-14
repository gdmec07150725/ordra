// 卡片只展示服务器保存的预览；提交时只发送 ID，不回传可被修改的订单和原因。
window.ReturnCards = {
  render(row, previews, onAction, locked) {
    row.querySelector('.return-cards')?.remove();
    if (!previews.length) return;
    const group = document.createElement('div');
    group.className = 'return-cards';
    for (const preview of previews) {
      const card = document.createElement('section');
      card.className = 'return-card';
      const title = document.createElement('h3');
      title.textContent = preview.state === 'submitted' ? '退货申请 · 本地模拟' : '退货申请预览 · 本地模拟';
      const state = document.createElement('p');
      state.className = 'return-state';
      state.textContent = preview.status;
      const fields = document.createElement('dl');
      const entries = [['订单', preview.order_id], ['商品', preview.order?.product_name],
        ['订单状态', preview.order?.status], ['退货原因', preview.reason], ['申请编号', preview.request_id],
        ['审核原因', preview.review?.reason], ['审核时间', preview.review?.reviewed_at ? new Date(preview.review.reviewed_at).toLocaleString('zh-CN') : null]];
      for (const [label, value] of entries) {
        if (!value) continue;
        const term = document.createElement('dt');
        const detail = document.createElement('dd');
        term.textContent = label;
        detail.textContent = value;
        fields.append(term, detail);
      }
      const note = document.createElement('p');
      note.className = 'return-hint';
      note.textContent = preview.state === 'pending'
        ? '请核对订单和原因。确认后创建模拟申请，等待人工审核；不会执行退款。继续聊天会使这份预览失效。'
        : preview.review?.decision === 'approved' ? '模拟审核已通过。后续寄回安排需人工处理，尚未执行退款。'
        : preview.review?.decision === 'rejected' ? '模拟审核未通过，请查看审核原因。当前没有接入真实售后处理。'
        : preview.state === 'submitted' ? '模拟申请已提交，等待人工审核。当前没有接入真实售后处理。'
        : preview.state === 'expired' ? '后续对话已使这份预览失效。如需申请，请重新生成预览。'
        : '这份预览已取消，未创建申请。';
      card.append(title, state, fields, note);
      if (preview.state === 'pending') {
        const actions = document.createElement('div');
        actions.className = 'return-actions';
        for (const [action, label] of [['confirm', '确认提交'], ['cancel', '取消']]) {
          const button = document.createElement('button');
          button.type = 'button';
          button.className = action === 'confirm' ? 'confirm-return' : 'quiet-button';
          button.textContent = label;
          button.disabled = locked;
          button.addEventListener('click', () => onAction(preview.preview_id, action));
          actions.append(button);
        }
        card.append(actions);
      }
      group.append(card);
    }
    row.append(group);
  },
};
