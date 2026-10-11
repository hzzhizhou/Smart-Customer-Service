"""确定性投诉建单：投诉场景直接建工单并开启人工会话，绕过 ReAct 工具循环。

背景：qwen 在投诉（含情绪化文本）场景可能陷入"识别藐视→再识别"的循环，
迟迟不调用 create_ticket 导致流式端点超时。这里由意图分类器确定性判定的
complaint 意图直接落单，保证永不卡死、工单必达。
"""
from typing import AsyncIterator

from logs.log_config import log
from service.handlers.context import TurnContext
from utils.metrics import handoff_total
from shared.reply_templates import (
    COMPLAINT_FALLBACK,
    SERVICE_CONTACT,
    TICKET_CREATED,
    TICKET_PROGRESS_NO_NOTE,
    category_cn,
    sla_cn,
    status_cn,
)


async def handle_complaint(ctx: TurnContext) -> AsyncIterator[str]:
    """产出投诉答复：已有在办投诉单则回报进度，否则新建工单并转入人工会话。

    调用前提（由调度方判定）：本轮意图为 complaint，且本轮不是在问工单进度
    —— 否则会「每问一次进度就多一张工单」（实测连问两次进度生成两张工单）。
    """
    try:
        from agent.tools.ticket_tool import create_ticket_record
        from service.handoff_service import start_handoff_session
        from infrastructure.mysql_store import query_ticket_progress

        user_id = ctx.user["user_id"]
        # 幂等：账号下已有未关闭的投诉工单时不再重复建单。转人工后 DST 会一直
        # 复用 complaint 意图，没有这道防护则用户每补一句话就多一张工单
        # （实测「我要投诉…」后再补「订单 SOxxx 有问题」又落一张新单）。
        open_ticket = query_ticket_progress(user_id, owner=user_id) if user_id else None
        if open_ticket and open_ticket.get("category") == "complaint" \
                and open_ticket.get("status") != "closed":
            yield TICKET_PROGRESS_NO_NOTE.format(
                subject=category_cn(open_ticket.get("category")),
                ticket_id=open_ticket["ticket_id"],
                status=status_cn(open_ticket.get("status")),
                created_at=open_ticket.get("created_at", ""),
                contact=SERVICE_CONTACT,
            )
            return

        description = ctx.question.strip() or "用户投诉，未填写具体描述"
        complaint_order = (ctx.dst.slots.get("order_id", "") if ctx.dst else "") or ""
        # 直接调 create_ticket_record 而非工具：需要拿到工单号来开启人工会话
        ticket = create_ticket_record(
            # 工单归属登录账号：换会话后用户查「我的工单」仍能查到
            user_id=user_id,
            order_id=complaint_order,
            description=f"[投诉] {description}",
            priority="high",
            category="complaint",
        )
        # 写入会话开场白：该工单由此成为一场人工会话，用户端切到人工客服模式
        start_handoff_session(ticket["ticket_id"])
        handoff_total.labels("complaint").inc()
        yield TICKET_CREATED.format(
            ticket_id=ticket["ticket_id"],
            created_at=ticket["created_at"],
            sla=sla_cn(ticket["priority"]),
            contact=SERVICE_CONTACT,
        )
    except Exception as e:
        log.error(f"确定性投诉建单失败: {e}", exc_info=True)
        yield COMPLAINT_FALLBACK