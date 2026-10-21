"""
# ====================== 智能售后客服工具指标 ======================
智能售后客服 - 订单查询工具
从 MySQL 持久化存储查询订单状态、物流、金额等信息（重启不丢失）。
面试演示场景：电商智能售后客服
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from langchain_core.tools import tool
from logs.log_config import log
from utils.metrics import order_queries_total, order_not_found_total
from infrastructure.mysql_store import can_access_order, get_order, list_orders_by_user
from agent.tools.tool_context import current_user_id
from shared.reply_templates import (
    MY_ORDERS_EMPTY,
    MY_ORDERS_HEADER,
    MY_ORDERS_LINE,
    MY_ORDERS_MORE,
    MY_ORDERS_NEED_LOGIN,
    ORDER_FORBIDDEN,
    ORDER_TAIL,
    order_status_cn,
)

# 「我名下订单」单次最多列出的张数：客服答复要保持简短，超出的让用户按单号单独查
MY_ORDERS_MAX = 10


@tool(description="根据订单号查询订单状态、物流轨迹、金额等信息（仅查询订单本身的物流/状态）。注意：若用户询问的是'工单/售后处理进度、处理结果、投诉结果'，这是查工单，请调用 query_ticket 而非本工具。本工具没有账号参数也不需要账号：订单归属由服务端按当前登录账号自动校验。")
def query_order(order_id: str) -> str:
    """
    智能售后客服订单查询工具：
    - 输入：order_id 订单号（如 SO20241120005）
    - 输出：订单详情（客户姓名已脱敏，合规要求）
    - 归属账号由服务端注入（agent.tools.tool_context），不由 LLM 传参决定
    """
    order_queries_total.inc()
    # 身份只来自服务端登录态：LLM 参数里已无该字段，提示注入无处可填（PROJECT_AUDIT P0-3）
    owner_id = current_user_id()
    log.info(f"客服工具 query_order 被调用，订单号：{order_id}，归属账号：{owner_id or '无'}")

    order = get_order(order_id)
    if not order:
        order_not_found_total.inc()
        log.warning(f"订单未找到：{order_id}")
        return f"未找到订单号为 {order_id} 的订单，请确认订单号是否正确。"

    # 归属校验：非本账号订单不返回任何字段（客户/金额/物流轨迹都属隐私）
    if not can_access_order(order, owner_id):
        log.warning(f"订单越权访问被拦截：订单={order_id} 归属={order.get('user_id')} "
                    f"请求账号={owner_id or '无'}")
        return ORDER_FORBIDDEN

    status_cn = order_status_cn(order["status"])
    lines = [
        f"订单号：{order['order_id']}",
        f"客户：{order['customer']}",
        f"商品：{order['product']}",
        f"金额：¥{order['amount']:.2f}",
        f"状态：{status_cn}",
    ]
    if order.get("logistics"):
        lines.append(f"物流：{order['logistics']}")
    if order.get("tracking"):
        lines.append(f"物流轨迹：{order['tracking']}")
    lines.append(f"下单时间：{order['created_at']}")
    return "\n".join(lines)


# ====================== 智能售后客服工具指标 ======================
@tool(description="列出「当前登录账号名下」的订单（最近的若干张，含订单号/状态/商品/金额/下单时间）。当用户没有提供订单号、问'我有哪些订单/我名下所有订单/我买过什么/我的订单列表'时调用本工具，不要因为缺订单号就只追问订单号。本工具没有账号参数也不需要账号：归属账号由服务端自动注入，你无法也不应该自行指定。")
def list_my_orders(limit: int = MY_ORDERS_MAX) -> str:
    """
    智能售后客服「我名下订单」查询工具：
    - 输入：limit 最多列出多少张（缺省/越界时取 MY_ORDERS_MAX）
    - 输出：订单列表（仅本账号；不含客户姓名；超出上限时提示可用订单号单独查）
    """
    order_queries_total.inc()
    uid = current_user_id()
    log.info(f"客服工具 list_my_orders 被调用，归属账号：{uid or '无'}")
    if not uid:
        # 没有账号上下文时不做任何猜测（例如按会话、按最近订单兜底都会越权）
        log.warning("list_my_orders 缺少归属账号，拒绝列出订单")
        return MY_ORDERS_NEED_LOGIN

    limit = limit if isinstance(limit, int) and 1 <= limit <= 20 else MY_ORDERS_MAX
    orders, total = list_orders_by_user(uid, limit=limit)
    if not orders:
        return MY_ORDERS_EMPTY

    lines = [MY_ORDERS_HEADER.format(total=total, shown=len(orders))]
    lines += [
        MY_ORDERS_LINE.format(
            order_id=o.get("order_id", ""),
            status=order_status_cn(o.get("status", "")) or "未知",
            product=o.get("product") or "未填写",
            amount=o.get("amount") or 0.0,
            created_at=o.get("created_at") or "未知",
        )
        for o in orders
    ]
    if total > len(orders):
        lines.append(MY_ORDERS_MORE)
    lines.append(ORDER_TAIL)
    return "\n".join(lines)
