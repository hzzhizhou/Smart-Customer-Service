"""对话历史存储（会话 + 消息）。

对话历史是业务数据：以 MySQL 为准（重启进程、Redis 被清空都不丢），
Redis 只作为热缓存（见 infrastructure/mysql_history.py）。
"""
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import connection

_TITLE_MAX = 24


def _owner_from_conversation_id(conversation_id: str) -> str:
    """从作用域会话 ID 反解归属账号：形如 u{user_id}_{session_id}。

    格式由 access/history.scoped_session_id 定义。user_id 是 32 位 hex、不含下划线，
    故按第一个下划线切分是安全的；两处格式若被改坏，
    tests/test_chat_history_store.py 的一致性断言会直接失败，不会静默错位。
    """
    cid = conversation_id or ""
    if cid.startswith("u") and "_" in cid:
        return cid[1:].split("_", 1)[0]
    return ""


def ensure_conversation(conversation_id: str, user_id: str = "") -> Dict[str, Any]:
    """确保会话行存在（幂等，已有则不改动）；返回会话行。"""
    cid = (conversation_id or "").strip()
    if not cid:
        raise ValueError("conversation_id 不能为空")
    uid = (user_id or "").strip() or _owner_from_conversation_id(cid)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute(
            "INSERT IGNORE INTO conversations (conversation_id, user_id, title, status, "
            "created_at, updated_at) VALUES (%s,%s,'','open',%s,%s)",
            (cid, uid, now, now),
        )
        cur.execute("SELECT * FROM conversations WHERE conversation_id=%s", (cid,))
        return connection._row(cur.fetchone()) or {}


def append_message(conversation_id: str, role: str, content: str,
                   user_id: str = "") -> Optional[Dict[str, Any]]:
    """追加一条会话消息（seq 自增；会话不存在则自动创建）；空内容返回 None。

    role: user / assistant。标题取首条用户消息，updated_at 每次刷新。
    seq 用「先锁会话行再取 MAX(seq)+1」计算：并发写同一会话时不会拿到同一个序号
    （唯一索引 uk_messages_conv_seq 是最后一道兜底）。
    """
    cid = (conversation_id or "").strip()
    text = (content or "").strip()
    if not cid or not text:
        return None
    role = "assistant" if role in ("ai", "assistant") else "user"
    uid = (user_id or "").strip() or _owner_from_conversation_id(cid)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = text.replace("\n", " ").strip()[:_TITLE_MAX]
    with connection._tx() as cur:
        cur.execute(
            "INSERT IGNORE INTO conversations (conversation_id, user_id, title, status, "
            "created_at, updated_at) VALUES (%s,%s,'','open',%s,%s)",
            (cid, uid, now, now),
        )
        cur.execute(
            "SELECT conversation_id FROM conversations WHERE conversation_id=%s FOR UPDATE",
            (cid,),
        )
        cur.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM messages "
            "WHERE conversation_id=%s",
            (cid,),
        )
        seq = cur.fetchone()["next_seq"]
        cur.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, created_at) "
            "VALUES (%s,%s,%s,%s,%s)",
            (cid, seq, role, text, now),
        )
        msg_id = cur.lastrowid
        cur.execute(
            "UPDATE conversations SET updated_at=%s, "
            "title=CASE WHEN title='' AND %s='user' THEN %s ELSE title END "
            "WHERE conversation_id=%s",
            (now, role, title, cid),
        )
    return {"msg_id": msg_id, "conversation_id": cid, "seq": seq,
            "role": role, "content": text, "created_at": now}


def list_messages(conversation_id: str, limit: int = 0, offset: int = 0
                  ) -> tuple[List[Dict[str, Any]], int]:
    """按 seq 升序取会话消息（limit<=0 取全部）。返回 (items, total)。"""
    cid = (conversation_id or "").strip()
    if not cid:
        return [], 0
    with connection._tx() as cur:
        cur.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=%s", (cid,)
        )
        total = cur.fetchone()["n"]
        if limit and limit > 0:
            cur.execute(
                "SELECT * FROM messages WHERE conversation_id=%s "
                "ORDER BY seq LIMIT %s OFFSET %s",
                (cid, limit, max(0, offset)),
            )
        else:
            cur.execute(
                "SELECT * FROM messages WHERE conversation_id=%s ORDER BY seq", (cid,)
            )
        return connection._rows(cur.fetchall()), total


def list_conversations_by_user(user_id: str, page: int = 1, page_size: int = 50
                               ) -> tuple[List[Dict[str, Any]], int]:
    """当前账号的会话列表（最近更新在前，带消息数）。返回 (items, total)。

    只按归属账号过滤，天然看不到他人会话；空会话（只有会话行、没有消息）不列出。
    """
    uid = (user_id or "").strip()
    if not uid:
        return [], 0
    cond = ("FROM conversations c WHERE c.user_id=%s "
            "AND EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id=c.conversation_id)")
    with connection._tx() as cur:
        cur.execute(f"SELECT COUNT(*) AS n {cond}", (uid,))
        total = cur.fetchone()["n"]
        cur.execute(
            "SELECT c.conversation_id, c.user_id, c.title, c.status, c.created_at, "
            "c.updated_at, (SELECT COUNT(*) FROM messages m "
            "WHERE m.conversation_id=c.conversation_id) AS message_count "
            f"{cond} ORDER BY c.updated_at DESC, c.conversation_id DESC "
            "LIMIT %s OFFSET %s",
            (uid, page_size, max(0, (page - 1) * page_size)),
        )
        return connection._rows(cur.fetchall()), total


def get_conversation(conversation_id: str, user_id: str = "") -> Optional[Dict[str, Any]]:
    """按归属账号取会话；不属于该账号（或不存在）返回 None —— 越权一律当作不存在。"""
    cid = (conversation_id or "").strip()
    uid = (user_id or "").strip()
    if not cid or not uid:
        return None
    with connection._tx() as cur:
        cur.execute(
            "SELECT * FROM conversations WHERE conversation_id=%s AND user_id=%s",
            (cid, uid),
        )
        return connection._row(cur.fetchone())


def get_latest_conversation_before(user_id: str, before_time: str
                                   ) -> Optional[Dict[str, Any]]:
    """取该账号在某一时刻之前最近活跃的一场会话（用于「转人工前的机器人记录」）。

    tickets 表与 conversations 表之间没有关联字段，所以这里按「归属账号 + 会话创建时间
    不晚于该工单创建时间」反查最近活跃的一场：用户总是从当前正在聊的那场会话点转人工，
    而那场会话正是 updated_at 最新的，因此反查结果稳定，不需要新增字段。

    过滤用 created_at 而不是 updated_at：用户转人工后若又回到 AI 模式继续聊，同一场会话的
    updated_at 会走到工单创建时间之后，用 updated_at 会把这唯一正确的会话漏掉。
    """
    uid = (user_id or "").strip()
    if not uid or not before_time:
        return None
    with connection._tx() as cur:
        cur.execute(
            "SELECT * FROM conversations WHERE user_id=%s AND created_at<=%s "
            "ORDER BY updated_at DESC, conversation_id DESC LIMIT 1",
            (uid, before_time),
        )
        return connection._row(cur.fetchone())


def close_conversation(conversation_id: str, user_id: str) -> bool:
    """结束会话（status=closed，历史仍可回看）；非本人会话/不存在/已结束返回 False。"""
    cid = (conversation_id or "").strip()
    uid = (user_id or "").strip()
    if not cid or not uid:
        return False
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute(
            "UPDATE conversations SET status='closed', updated_at=%s "
            "WHERE conversation_id=%s AND user_id=%s AND status='open'",
            (now, cid, uid),
        )
        return cur.rowcount > 0


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """删除会话及其消息；非本人会话/不存在返回 False。"""
    cid = (conversation_id or "").strip()
    uid = (user_id or "").strip()
    if not cid or not uid:
        return False
    with connection._tx() as cur:
        cur.execute(
            "DELETE FROM conversations WHERE conversation_id=%s AND user_id=%s",
            (cid, uid),
        )
        if cur.rowcount <= 0:
            return False
        cur.execute("DELETE FROM messages WHERE conversation_id=%s", (cid,))
        return True


def clear_messages(conversation_id: str) -> int:
    """清空某会话的消息（保留会话行）；返回删除条数。"""
    cid = (conversation_id or "").strip()
    if not cid:
        return 0
    with connection._tx() as cur:
        cur.execute("DELETE FROM messages WHERE conversation_id=%s", (cid,))
        return cur.rowcount or 0
