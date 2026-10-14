---
name: return-handling
description: 退货退款办理：核对条件、补齐原因、预览和确认。
---
# 退货处理

1. 获取明确订单号和具体原因。缺信息则询问；有计划则标记 blocked，结束等待，不反复检索。
2. 发现 get_order 和 search_knowledge，核对订单和适用政策原文。独立查询可并行或委派 orders、knowledge。
3. 条件满足后发现 prepare_return_request，遵守审批并生成预览。请用户核对订单、原因并通过页面确认；预览不等于提交成功。
4. 只有实际提交成功才告知申请编号；否则说明待确认或待审核，不承诺退款到账。

需要细化询问缺失信息时，单独调用 read_skill_resource 读取[缺失信息询问指引](references/missing-information.md)。遇到失败、拒绝或重复申请时，按需读取[异常处理流程](references/exception-handling.md)。已知道如何处理时不必读取；引用只补充操作流程，订单与政策仍需实际查询。
