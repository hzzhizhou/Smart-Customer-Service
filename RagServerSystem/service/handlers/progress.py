"""
确定性工单进度查询：命中关键词直接读工单（含人工备注）回传，绕过 ReAct。

背景：纯提示词难以约束 LLM 在"订单号 + 工单/售后"场景优先选 query_ticket，
模型常被 order_id 拽回 query_order。改由规则确定性判定，保证"答复回传"闭环必达。

按业务定位（PROJECT_AUDIT §6.2(1)）：只按「最近一张」取工单是不够的 ——
用户先建退货单、后建投诉单，再问「退货进度」就会拿到投诉单。这里按问法里的
业务词推出 category 一并传给查询，把结果收窄到该业务。
"""
import asyncio
import re
from typing import AsyncIterator, Optional

from logs.log_config import log
from service.handlers.context import TurnContext
from shared.reply_templates import PROGRESS_FALLBACK

# 问法关键词 → 业务分类。顺序即优先级：用户说「退款」时不该被「退货」抢答
_CATEGORY_KEYWORDS = (
    ("退款", "refund"),
    ("退货", "return"),
    ("退换", "return"),
    ("投诉", "complaint"),
    ("转人工", "consult"),
)


def infer_business_category(question: str, dst) -> str:
    """从问法（其次从会话主线意图）推出「用户问的是哪件业务」的工单分类。

    问法优先于 DST：用户可能在退货进行中插一句「投诉处理得怎么样了」，
    这时要按本轮问法查投诉单，而不是沿用退货主线。
    """
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in (question or ""):
            return category
    intent = getattr(dst, "intent", "") if dst is not None else ""
    return intent if intent in ("return", "refund", "complaint", "consult") else ""


async def handle_ticket_progress(ctx: TurnContext) -> AsyncIterator[str]:
    try:
        from agent.tools.ticket_query_tool import query_ticket as _qt

        # 查询依据优先级：本句订单号 > 会话已收集的订单号 > 登录账号
        # （用户追问"退货进度"时常不带单号，靠 DST 槽位里的订单号兜底；
        #   槽位也没有时按账号查，保证换会话后仍能查到自己以前的工单）
        m = re.search(r"SO\d+", ctx.question)
        if m:
            order_id: Optional[str] = m.group(0)
        elif ctx.dst is not None and ctx.dst.slots.get("order_id"):
            order_id = ctx.dst.slots["order_id"]
        else:
            order_id = ""
        # 业务分类由问法确定性判定，不交给模型猜：模型漏传就会退回「最近一张」的旧行为
        category = infer_business_category(ctx.question, ctx.dst)
        # 归属账号由服务端上下文注入，调用方不再传 owner_id 参数。
        # 工具内部同步查库，放线程池执行
        yield await asyncio.to_thread(
            _qt.invoke, {"order_id": order_id, "category": category})
    except Exception as e:
        log.error(f"确定性工单进度查询失败: {e}", exc_info=True)
        yield PROGRESS_FALLBACK
