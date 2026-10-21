"""
智能客服 - 售后工单处理进度查询工具
查询用户已提交工单的当前处理环节与人工客服备注，实现"答复回传"闭环：
- 客服机器人自动建单 → 人工客服在管理端推进业务环节并填写备注(note)
- 用户再次询问"我的退货办得怎么样"时，机器人按环节拼装完整答复

回答完整度标准（对齐企业客服话术四要素）：
当前环节 + 已完成哪些环节 + 下一步谁做什么 + 时效承诺 + 联系方式。
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_core.tools import tool

from logs.log_config import log
from infrastructure.mysql_store import query_ticket_progress
from agent.tools.tool_context import current_user_id
from shared.reply_templates import (
    NO_TICKET,
    RETURN_PROGRESS,
    RETURN_PROGRESS_ABNORMAL,
    RETURN_PROGRESS_NOTE,
    RETURN_STAGE_EXCEPTIONS,
    SERVICE_CONTACT,
    TICKET_NEED_LOGIN,
    TICKET_PROGRESS_NO_NOTE,
    TICKET_PROGRESS_WITH_NOTE,
    category_cn,
    render_stage_timeline,
    stage_cn,
    stage_def,
    stage_step,
    status_cn,
)

# 走阶段化流转的售后分类
_STAGED_CATEGORY = {"return", "refund"}
# 允许按业务过滤的分类白名单（防止模型传中文/乱值把查询收窄成空结果）
_VALID_QUERY_CATEGORY = {"return", "refund", "complaint", "consult", "technical",
                         "logistics", "other"}


def _render_staged_progress(ticket: dict, subject: str) -> str:
    """退货/退款工单：按业务环节拼装完整进度答复（含流转轨迹、下一步、时效、联系方式）。"""
    stage = (ticket.get("stage") or "").strip()
    # 分类决定轨迹长度：纯退款不退货，轨迹里不出现「寄回商品 / 商家验收」
    category = (ticket.get("category") or "").strip()
    note = (ticket.get("note") or "").strip()
    note_line = RETURN_PROGRESS_NOTE.format(note=note) if note else ""
    updated_at = ticket.get("stage_updated_at") or ticket.get("created_at") or ""
    d = stage_def(stage)
    common = dict(
        subject=subject,
        ticket_id=ticket["ticket_id"],
        order_id=ticket.get("order_id") or "未关联订单",
        stage_cn=d["cn"],
        updated_at=updated_at,
        next=d["next"],
        note_line=note_line,
        contact=SERVICE_CONTACT,
    )

    # 非顺序环节（如审核未通过）：流程已终止，不再展示进度轨迹
    if stage in RETURN_STAGE_EXCEPTIONS:
        return RETURN_PROGRESS_ABNORMAL.format(**common)

    step, total = stage_step(stage, category)
    return RETURN_PROGRESS.format(
        step=step,
        total=total,
        sla=d["sla"],
        timeline=render_stage_timeline(ticket.get("stage_history"), stage, category),
        **common,
    )


@tool(description="查询当前登录账号名下工单的处理进度：退货/退款类工单会返回当前业务环节、流转轨迹、下一步与时效；其他工单返回处理状态与人工客服备注。当用户询问'退货办到哪一步了/工单处理得怎么样/投诉有结果吗/转人工后续怎么处理'时调用。建议传入 category（return 退货 / refund 退款 / complaint 投诉）限定「问的是哪件业务的工单」，避免在同时有退货单和投诉单时查串。本工具没有账号参数也不需要账号：归属账号由服务端按当前登录账号自动注入。")
def query_ticket(order_id: str = "", category: str = "") -> str:
    """
    智能客服工单进度查询工具：
    - 输入：order_id 订单号（如 SO20241120005）。不带时按当前登录账号查最近一张工单。
    - 输入：category 要查的业务分类（return/refund/complaint/...），用户明确在问某件业务的
      进度时传入，避免「先建退货单、后建投诉单」时把投诉单当成退货进度返回。
    - 输出：最近一张工单的处理进度（阶段化售后为完整环节说明 + 人工客服答复回传）
    """
    # 身份只认服务端登录态：工具参数里已无账号字段，模型无从伪造（PROJECT_AUDIT P0-3）
    owner = current_user_id()
    if not owner:
        log.warning("query_ticket 缺少登录账号，拒绝查询")
        return TICKET_NEED_LOGIN

    query = (order_id or "").strip() or owner
    _category = (category or "").strip() if (category or "").strip() in _VALID_QUERY_CATEGORY else ""
    log.info(f"客服工具 query_ticket 被调用，查询依据：{query}，归属限定：{owner}，"
             f"业务分类：{_category or '不限'}")

    ticket = query_ticket_progress(query, owner=owner, category=_category)
    if not ticket:
        return NO_TICKET

    subject = "退货" if ticket.get("category") == "return" else "退款"
    # 退货/退款且已进入业务环节：给完整流程答复（用户能看懂"现在到哪、还要做什么"）
    if ticket.get("category") in _STAGED_CATEGORY and (ticket.get("stage") or "").strip():
        return _render_staged_progress(ticket, subject)

    note = (ticket.get("note") or "").strip()
    template = TICKET_PROGRESS_WITH_NOTE if note else TICKET_PROGRESS_NO_NOTE
    return template.format(
        subject=category_cn(ticket.get("category")),
        ticket_id=ticket["ticket_id"],
        status=status_cn(ticket.get("status")) or stage_cn(ticket.get("stage")),
        created_at=ticket.get("created_at", ""),
        note=note,
        contact=SERVICE_CONTACT,
    )


if __name__ == "__main__":
    # 身份改由服务端注入后，脚本里要先显式绑定账号才有归属上下文
    from agent.tools.tool_context import set_current_user
    set_current_user("demo-owner")
    print("--- 查询退货工单（按订单号）---")
    print(query_ticket.invoke({"order_id": "SO20241120005", "category": "return"}))
    print("--- 空参数 ---")
    print(query_ticket.invoke({}))