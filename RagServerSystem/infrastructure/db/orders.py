"""订单存储。"""
from typing import Any, Dict, List, Optional

from . import connection
from . import users


def get_order(order_id: str) -> Optional[Dict[str, Any]]:
    with connection._tx() as cur:
        cur.execute("SELECT * FROM orders WHERE order_id=%s", (order_id,))
        return connection._row(cur.fetchone())


def can_access_order(order: Dict[str, Any], owner_id: str = "") -> bool:
    """订单归属校验：订单有归属时，只有归属账号本人或管理员可以访问。

    此前订单查询没有任何权限边界 —— 拿别人的订单号一查就能看到客户、金额、
    物流与收货信息。归属账号由 bind_ownerless_orders 在启动时绑定到演示买家。
    订单无归属（历史数据）时放行，避免旧数据把所有人都挡在门外。
    """
    holder = (order.get("user_id") or "").strip()
    if not holder:
        return True
    requester = (owner_id or "").strip()
    if not requester:
        # 没带账号上下文（如工具被无登录态调用）时按拒绝处理：宁可查不到，也不越权
        return False
    if requester == holder:
        return True
    admin = users.get_user_by_id(requester)
    return bool(admin and admin.get("role") == "admin")


def bind_ownerless_orders(user_id: str) -> int:
    """把尚无归属的订单绑到指定账号（演示买家），让归属校验有真实数据可用。幂等。

    种子订单是 mock 数据、原本谁都能查，启动时统一挂到演示买家账号下；
    已绑定的订单不会被改写，因此重启或换账号不会把归属挪走。
    """
    if not user_id:
        return 0
    with connection._tx() as cur:
        cur.execute(
            "UPDATE orders SET user_id=%s WHERE user_id IS NULL OR user_id=''",
            (user_id,),
        )
        return cur.rowcount or 0


def update_order_status(order_id: str, status: str) -> Optional[Dict[str, Any]]:
    """更新订单状态（后台管理员维护，如发货/签收/退款中）；返回更新后的订单，不存在返回 None。"""
    with connection._tx() as cur:
        cur.execute("SELECT * FROM orders WHERE order_id=%s", (order_id,))
        row = connection._row(cur.fetchone())
        if row is None:
            return None
        cur.execute("UPDATE orders SET status=%s WHERE order_id=%s", (status, order_id))
        row["status"] = status
        return row


def list_all_orders(status: Optional[str] = None, page: int = 1,
                    page_size: int = 50) -> tuple[List[Dict[str, Any]], int]:
    """后台订单列表（分页）。status 为空/'all' 返回全部，否则按状态过滤；
    返回 (items, total)。供管理端订单页使用。"""
    status = None if status in (None, "", "all") else status
    where, params = "", []
    if status:
        where = " WHERE status=%s"
        params = [status]
    with connection._tx() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM orders{where}", params)
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT * FROM orders{where} ORDER BY created_at DESC, id DESC "
            "LIMIT %s OFFSET %s",
            params + [page_size, max(0, (page - 1) * page_size)],
        )
        return connection._rows(cur.fetchall()), total


def list_orders_by_user(user_id: str, limit: int = 20) -> tuple[List[Dict[str, Any]], int]:
    """当前账号名下的订单列表（最近下单在前）。返回 (items, total)。

    「查我名下所有订单」的唯一入口：只按归属账号过滤，因此天然不会带出他人订单
    （不需要再逐单调 can_access_order）。user_id 为空直接返回空，避免没有账号上下文
    时把全库订单列出来。
    """
    uid = (user_id or "").strip()
    if not uid:
        return [], 0
    with connection._tx() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM orders WHERE user_id=%s", (uid,))
        total = cur.fetchone()["n"]
        cur.execute(
            "SELECT * FROM orders WHERE user_id=%s "
            "ORDER BY created_at DESC, id DESC LIMIT %s",
            (uid, limit),
        )
        return connection._rows(cur.fetchall()), total
