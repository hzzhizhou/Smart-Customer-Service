"""确定性分支处理器（服务层）：把「投诉/工单进度/订单查询/我名下订单/退货退款/知识咨询」
六条与 LLM 无关的确定性分支各自成文件。原先挂在接入层，因不含任何 HTTP 语义、
且只被调度方调用，已归入服务层。

约定：每个处理器都是异步生成器，入参只有 TurnContext，产出若干文本片段。
它们只负责「这一轮该回什么」，不负责写历史与推进 DST —— 那是调度方
（service/turn/agent_turn.py 的 AgentTurnService）统一收尾的职责，避免每条分支各写一遍。
"""
from service.handlers.context import TurnContext
from service.handlers.complaint import handle_complaint
from service.handlers.progress import handle_ticket_progress
from service.handlers.order_query import handle_order_query
from service.handlers.my_orders import handle_my_orders
from service.handlers.return_refund import handle_return_refund
from service.handlers.knowledge import handle_consult, stream_knowledge_answer

__all__ = [
    "TurnContext",
    "handle_complaint",
    "handle_ticket_progress",
    "handle_order_query",
    "handle_my_orders",
    "handle_return_refund",
    "handle_consult",
    "stream_knowledge_answer",
]