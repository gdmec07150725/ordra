"""主管请求中的知识交接视图；原始证据仍由历史、检查点和校验器保管。"""
import json


def compact_delegated_knowledge(messages):
    """只修改传入的请求副本；合并相同来源/版本/原文，不截断政策内容。"""
    calls = {call['id']: call for message in messages for call in message.get('tool_calls', [])}
    for message in messages:
        call = calls.get(message.get('tool_call_id'))
        if message.get('role') != 'tool' or not call or call['function']['name'] != 'delegate_tasks':
            continue
        payload = json.loads(message['content'])
        sources, references = {}, {}
        for task in payload.get('delegations', []):
            cited = set(task.get('cited_evidence_ids', []))
            complete_knowledge = (task.get('agent') == 'knowledge' and task.get('status') == 'completed'
                                  and bool(task.get('evidence')))
            for evidence in task.get('evidence', []):
                result = evidence.get('result', {})
                if evidence.get('name') != 'search_knowledge' or 'error' in result or not isinstance(result.get('results'), list):
                    complete_knowledge = False
                    continue
                if evidence.get('status') != 'ok' or not result['results']:
                    complete_knowledge = False
                refs = []
                for hit in result['results']:
                    # 保留来源ID、版本、标题、章节和全文；排序分数与文件偏移留在原记录。
                    source = {key: value for key, value in hit.items()
                              if key not in ('document_id', 'start', 'end', 'score')}
                    identity = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
                    if identity not in references:
                        ref = f'K{len(references) + 1}'
                        references[identity], sources[ref] = ref, source
                    refs.append(references[identity])
                view = {key: value for key, value in result.items() if key not in (
                    'results', 'query', 'document_count', 'matched_chunks', 'search_mode',
                    'semantic_status', '_tool_call_id')}
                # 空资料库与未找到匹配仍可区分；不隐藏检索失败或预算截断标记。
                if result.get('document_count') == 0:
                    view['document_count'] = 0
                for flag in ('has_more', 'budget_limited'):
                    if view.get(flag) is False:
                        view.pop(flag)
                view['source_refs'] = refs
                evidence['result'] = view
                # 有命中时保留政策事实；无命中时保留查询词，说明哪一项未找到。
                if refs:
                    evidence.pop('arguments', None)
            # 引用关系贴在实际证据旁，避免再传一遍长ID列表。
            if sources:
                for evidence in task.get('evidence', []):
                    evidence['cited'] = evidence.get('call_id') in cited
                task.pop('cited_evidence_ids', None)
                if complete_knowledge:
                    # 成功检索的完整原文已经在 sources 中；主管可直接据此作答。
                    # 子模型的派生解释仍在原历史与校验输入中，不必在请求里重复。
                    task.pop('summary', None)
                    task['summary_omitted'] = True
        if sources:
            payload['knowledge_sources'] = sources
            message['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
