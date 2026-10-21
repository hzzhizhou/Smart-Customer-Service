"""工单、工单消息与人工会话存储。"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pymysql

from logs.log_config import log

from . import connection


def create_ticket(ticket_id: str, user_id: str, order_id: str, description: str,
                  priority: str, category: str, note: str = "",
                  stage: str = "", idempotency_key: str = "") -> Dict[str, Any]:
    """建单。stage 为业务环节（退货/退款类工单才有，如 submitted）。

    工单双轨状态：status 是工单生命周期（待处理/处理中/已解决/已关闭），
    stage 是业务环节（申请→审核→寄回→验收→退款→到账），用户查进度看的是 stage。

    idempotency_key（§6.2(2)）：传入后由 uk_tickets_idem 唯一键保证「同一次申请只落一张单」。
    并发下两个请求同时 INSERT，只有一个能成功；另一个捕获唯一键冲突后返回已存在的那张，
    不再依赖「先查后写」那种一有并发就失效的检查。键为空时不参与去重（历史调用方兼容）。
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    order_id = order_id or ""
    stage = stage or ""
    note = note or ""
    key = (idempotency_key or "").strip() or None
    history = [{"stage": stage, "at": created_at}] if stage else []
    try:
        with connection._tx() as cur:
            cur.execute(
                "INSERT INTO tickets (ticket_id, user_id, order_id, description, priority, "
                "category, status, created_at, note, stage, stage_history, stage_updated_at, "
                "idempotency_key) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (ticket_id, user_id, order_id, description, priority, category, "open",
                 created_at, note, stage, json.dumps(history, ensure_ascii=False),
                 created_at if stage else "", key),
            )
    except pymysql.err.IntegrityError:
        # 唯一键冲突 = 同一次申请已经建过单：返回已有工单，让调用方按 ticket_id 继续
        existing = get_ticket_by_idempotency_key(key or "")
        if existing is None:
            raise          # 不是幂等键冲突（例如理论上不可能的 ticket_id 撞车）就照常抛
        return existing
    return {
        "ticket_id": ticket_id, "user_id": user_id, "order_id": order_id,
        "description": description, "priority": priority, "category": category,
        "status": "open", "created_at": created_at, "note": note,
        "stage": stage, "stage_history": history,
        "stage_updated_at": created_at if stage else "",
    }


def list_tickets(status: str = "open") -> List[Dict[str, Any]]:
    """供人工客服侧查询工单"""
    with connection._tx() as cur:
        cur.execute(
            "SELECT * FROM tickets WHERE status=%s ORDER BY created_at DESC, id DESC",
            (status,),
        )
        return connection._rows(cur.fetchall())


def list_all_tickets(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """查询全部工单；status 为空/'all' 返回所有，否则按状态过滤。供管理端看板。"""
    with connection._tx() as cur:
        if status in (None, "", "all"):
            cur.execute("SELECT * FROM tickets ORDER BY created_at DESC, id DESC")
        else:
            cur.execute(
                "SELECT * FROM tickets WHERE status=%s ORDER BY created_at DESC, id DESC",
                (status,),
            )
        return connection._rows(cur.fetchall())


def list_tickets_paged(status: Optional[str], page: int, page_size: int
                       ) -> List[Dict[str, Any]]:
    """分页查询工单（排序：创建时间倒序，最新在前）。返回当页工单。"""
    offset = max(0, (page - 1) * page_size)
    with connection._tx() as cur:
        if status in (None, "", "all"):
            cur.execute(
                "SELECT * FROM tickets ORDER BY created_at DESC, id DESC "
                "LIMIT %s OFFSET %s",
                (page_size, offset),
            )
        else:
            cur.execute(
                "SELECT * FROM tickets WHERE status=%s ORDER BY created_at DESC, id DESC "
                "LIMIT %s OFFSET %s",
                (status, page_size, offset),
            )
        return connection._rows(cur.fetchall())


def count_tickets(status: Optional[str] = None) -> int:
    """统计工单数量；status 为空/'all' 统计全部，否则按状态过滤。"""
    with connection._tx() as cur:
        if status in (None, "", "all"):
            cur.execute("SELECT COUNT(*) AS n FROM tickets")
        else:
            cur.execute("SELECT COUNT(*) AS n FROM tickets WHERE status=%s", (status,))
        return cur.fetchone()["n"]


# 工单环节 → 工单生命周期状态的自动联动（人工只推环节，状态自动跟随）
_STAGE_TO_STATUS = {
    "submitted": "open",       # 待商家审核
    "completed": "resolved",   # 退款到账，售后办结
    "rejected": "resolved",    # 审核未通过，流程终止
}
# 工单环节 → 订单销售状态的联动（业务数据随售后结果变化）
_STAGE_TO_ORDER_STATUS = {
    "refunding": "refunding",  # 退款已发起
    "completed": "refunded",   # 退款已到账
}


def _parse_stage_history(raw: Any) -> List[Dict[str, str]]:
    """把 stage_history 字段（JSON 文本）解析为列表；脏数据降级为空列表。"""
    if isinstance(raw, list):
        return raw
    try:
        data = json.loads(raw) if raw else []
        return data if isinstance(data, list) else []
    except (ValueError, TypeError):
        return []


def update_ticket_status(
    ticket_id: str, status: str = None, note: str = None, stage: str = None
) -> Optional[Dict[str, Any]]:
    """更新工单处理状态/处理备注/业务环节；返回更新后的工单，不存在返回 None。

    企业做法：人工在后台只需推进「业务环节」，工单生命周期状态与订单销售状态自动跟随，
    并把每次环节变更追加进 stage_history，用户侧即可看到完整的流转轨迹。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute("SELECT * FROM tickets WHERE ticket_id=%s", (ticket_id,))
        row = connection._row(cur.fetchone())
        if row is None:
            return None
        d = dict(row)

        if note is not None:
            cur.execute(
                "UPDATE tickets SET note=%s WHERE ticket_id=%s", (note, ticket_id)
            )
            d["note"] = note

        if stage is not None and stage != (d.get("stage") or ""):
            history = _parse_stage_history(d.get("stage_history"))
            history.append({"stage": stage, "at": now})
            # 人工未显式指定状态时，按环节自动推进工单状态
            if status is None:
                status = _STAGE_TO_STATUS.get(stage, "processing")
            cur.execute(
                "UPDATE tickets SET stage=%s, stage_history=%s, stage_updated_at=%s "
                "WHERE ticket_id=%s",
                (stage, json.dumps(history, ensure_ascii=False), now, ticket_id),
            )
            d["stage"] = stage
            d["stage_history"] = history
            d["stage_updated_at"] = now
            # 业务数据联动：退款环节同步订单销售状态（订单号缺失时跳过）
            order_status = _STAGE_TO_ORDER_STATUS.get(stage)
            if order_status and (d.get("order_id") or ""):
                cur.execute(
                    "UPDATE orders SET status=%s WHERE order_id=%s",
                    (order_status, d["order_id"]),
                )
                log.info(f"工单 {ticket_id} 环节={stage} → 订单 {d['order_id']} 状态={order_status}")

        if status is not None:
            cur.execute(
                "UPDATE tickets SET status=%s WHERE ticket_id=%s", (status, ticket_id)
            )
            d["status"] = status

    d["stage_history"] = _parse_stage_history(d.get("stage_history"))
    return d


def get_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    """按工单号查询单条工单；不存在返回 None。stage_history 已解析为列表。"""
    with connection._tx() as cur:
        cur.execute("SELECT * FROM tickets WHERE ticket_id=%s", (ticket_id,))
        d = connection._row(cur.fetchone())
    if not d:
        return None
    d["stage_history"] = _parse_stage_history(d.get("stage_history"))
    return d


def get_ticket_by_idempotency_key(idempotency_key: str) -> Optional[Dict[str, Any]]:
    """按幂等键取工单。建单撞唯一键时用它把「已有工单」原样返回给调用方。"""
    key = (idempotency_key or "").strip()
    if not key:
        return None
    with connection._tx() as cur:
        cur.execute("SELECT * FROM tickets WHERE idempotency_key=%s", (key,))
        d = connection._row(cur.fetchone())
    if not d:
        return None
    d["stage_history"] = _parse_stage_history(d.get("stage_history"))
    return d


def query_ticket_progress(query: str, owner: str = "", category: str = "",
                          ticket_id: str = "") -> Optional[Dict[str, Any]]:
    """按工单号 / 用户标识 / 订单号查最近一张工单的处理进度（倒序最新一条）。

    工单的 user_id 存登录账号标识（工单归属账号，不随会话变化），
    order_id 列存关联订单号，描述里也常带订单号，故三者同时匹配，
    取最新一条即可，避免漏查。不存在返回 None。

    owner: 限定工单归属账号。传入后只看 owner 名下的工单 —— 种子订单号是 mock 数据，
    被多个测试账号共用（如 SO20241120005），不加这道收口时 B 账号拿 A 账号的单号一查，
    就会拿到 A 的工单（实测新账号确认退货时返回了 admin 几十分钟前建的工单）。

    category / ticket_id（§6.2(1)）：工单进度必须能定位到「问的是哪件业务」。
    此前只有 user_id/order_id/描述 LIKE 三个维度取最新一条，用户先建退货单、后建投诉单，
    再问「退货进度」就会拿到投诉单。现在：
      - ticket_id 是最稳的定位方式，给了就精确查一条（同时仍校验归属）；
      - category 把结果收窄到该业务分类，避免跨业务串单。
    """
    tid = (ticket_id or "").strip()
    if tid:
        with connection._tx() as cur:
            if owner:
                cur.execute("SELECT * FROM tickets WHERE ticket_id=%s AND user_id=%s",
                            (tid, owner))
            else:
                cur.execute("SELECT * FROM tickets WHERE ticket_id=%s", (tid,))
            d = connection._row(cur.fetchone())
        if not d:
            return None
        d["stage_history"] = _parse_stage_history(d.get("stage_history"))
        return d

    if not query and not owner:
        return None
    with connection._tx() as cur:
        # created_at 是秒级字符串，同一秒内建的多张工单会并列，排序结果不确定：
        # 实测「先建退货工单、同秒再建投诉工单」后查进度，返回的是退货那张。
        # 用自增主键 id 兜底（插入顺序），保证「最近一张」始终是最后建的那张。
        where = ["(user_id=%s OR order_id=%s OR description LIKE %s)"]
        params: List[Any] = [query, query, f"%{query}%"]
        if owner:
            where.append("user_id=%s")
            params.append(owner)
        if category:
            where.append("category=%s")
            params.append(category)
        cur.execute(
            "SELECT * FROM tickets WHERE " + " AND ".join(where) +
            " ORDER BY created_at DESC, id DESC LIMIT 1",
            params,
        )
        d = connection._row(cur.fetchone())
    if not d:
        return None
    d["stage_history"] = _parse_stage_history(d.get("stage_history"))
    return d


# ====================== 人工会话消息（转人工实时对话）======================
# 约定：某张工单一旦有了 ticket_messages 记录，它就代表一场「人工会话」。
# AI 自动建的退货/退款工单走阶段化流转，不写消息，因此不会被当成人工会话。

def add_ticket_message(ticket_id: str, sender: str, content: str,
                       sender_id: str = "", sender_name: str = "") -> Dict[str, Any]:
    """追加一条会话消息。sender 为 user（用户）/ agent（人工客服）/ system（系统提示）。"""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute(
            "INSERT INTO ticket_messages (ticket_id, sender, sender_id, sender_name, "
            "content, created_at) VALUES (%s,%s,%s,%s,%s,%s)",
            (ticket_id, sender, sender_id or "", sender_name or "", content, created_at),
        )
        msg_id = cur.lastrowid
    return {
        "msg_id": msg_id, "ticket_id": ticket_id, "sender": sender,
        "sender_id": sender_id or "", "sender_name": sender_name or "",
        "content": content, "created_at": created_at,
    }


def list_ticket_messages(ticket_id: str, after_id: int = 0,
                         limit: int = 200) -> List[Dict[str, Any]]:
    """按自增 msg_id 升序取消息。after_id 用于增量拉取（断线重连补消息）。"""
    with connection._tx() as cur:
        cur.execute(
            "SELECT * FROM ticket_messages WHERE ticket_id=%s AND msg_id>%s "
            "ORDER BY msg_id LIMIT %s",
            (ticket_id, after_id, limit),
        )
        return connection._rows(cur.fetchall())


def get_handoff_ticket(user_id: str) -> Optional[Dict[str, Any]]:
    """查该用户当前进行中的人工会话工单：未关闭且已有消息，取最新一条。

    用「是否已有消息」而非「是否未关闭」判定，避免把 AI 自动建的
    退货/退款工单误当成人工会话。不存在返回 None。
    """
    if not user_id:
        return None
    with connection._tx() as cur:
        cur.execute(
            "SELECT t.* FROM tickets t WHERE t.user_id=%s AND t.status<>'closed' "
            "AND EXISTS (SELECT 1 FROM ticket_messages m WHERE m.ticket_id=t.ticket_id) "
            "ORDER BY t.created_at DESC, t.id DESC LIMIT 1",
            (user_id,),
        )
        d = connection._row(cur.fetchone())
    if not d:
        return None
    d["stage_history"] = _parse_stage_history(d.get("stage_history"))
    return d


def list_handoff_sessions() -> List[Dict[str, Any]]:
    """客服工作台会话列表：所有存在消息的工单，带最后一条消息与消息条数。

    按最后一条消息时间倒序，让客服一眼看出谁刚发了新留言。
    关联 users 表带出用户名，客服看到的才不是一串账号 ID。
    """
    with connection._tx() as cur:
        cur.execute(
            """SELECT t.ticket_id, t.user_id, t.description, t.status, t.priority,
                      t.category, t.created_at,
                      u.username AS username, u.nickname AS nickname,
                      m.content AS last_content, m.sender AS last_sender,
                      m.created_at AS last_at,
                      (SELECT COUNT(*) FROM ticket_messages x
                        WHERE x.ticket_id=t.ticket_id) AS message_count
                 FROM tickets t
                 JOIN (SELECT ticket_id, MAX(msg_id) AS mid FROM ticket_messages
                        GROUP BY ticket_id) lm ON lm.ticket_id = t.ticket_id
                 JOIN ticket_messages m ON m.msg_id = lm.mid
                 LEFT JOIN users u ON u.user_id = t.user_id
                ORDER BY m.msg_id DESC"""
        )
        return connection._rows(cur.fetchall())
