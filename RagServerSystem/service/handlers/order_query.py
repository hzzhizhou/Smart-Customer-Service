"""
确定性订单查询：问题里带订单号且意图为订单/物流，直接查库回传。
"""
import asyncio
import re
from typing import AsyncIterator

from logs.log_config import log
from service.handlers.context import TurnContext
from shared.reply_templates import AGENT_NO_ANSWER_FALLBACK, ORDER_FORBIDDEN, ORDER_TAIL


async def handle_order_query(ctx: TurnContext) -> AsyncIterator[str]:
    """调用前提（由调度方判定）：本句含订单号，且意图为 order/logistics。"""
    try:
        from agent.tools.order_query import query_order as _qo

        m = re.search(r"SO\d+", ctx.question)
        # 只传订单号：归属账号由服务端上下文注入（agent.tools.tool_context），
        # 调用方不再把身份当参数往下传，这条确定性路径与 Agent 路径口径一致。
        # 工具内部是同步查库（pymysql），放线程池，避免在事件循环里阻塞其他请求
        resp = await asyncio.to_thread(_qo.invoke, {"order_id": m.group(0)})
        # 被归属校验拒绝时不再追加拿单建议，避免答复自相矛盾
        yield resp if ORDER_FORBIDDEN in resp else f"{resp}\n\n{ORDER_TAIL}"
    except Exception as e:
        log.error(f"确定性订单查询失败: {e}", exc_info=True)
        yield AGENT_NO_ANSWER_FALLBACK