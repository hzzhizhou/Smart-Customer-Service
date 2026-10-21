"""
用户端对话历史 API：列出「当前登录用户」的历史会话 + 读取/删除/结束某会话。

数据来源：MySQL（infrastructure.mysql_store 的 conversations / messages 表）。
  - 历史是业务数据，以库为准：进程重启、Redis 被清空都不丢；
  - Redis 只是同一条历史的热缓存（见 infrastructure/mysql_history.py），
    因此这里读列表/读消息完全不依赖 Redis，Redis 挂了只是下一次对话慢一点。

隔离方式：会话行带归属账号 user_id，查询一律带 `WHERE user_id=%s`，
  - 列表只取自己名下的会话；
  - 读取/删除/结束时把 session_id 与自己的账号组合成作用域会话 ID 再查，
    越权传他人 session_id 只会查到自己名下不存在的会话，拿不到任何数据
    （也不泄露“该会话是否存在”）。
"""
from __future__ import annotations

import asyncio
import re
import uuid
from typing import List, Dict

from config.settings import REDIS_CONFIG
from infrastructure import mysql_store

# 前端会话 ID 只允许 URL 安全字符，避免拼接进作用域 ID 时引入分隔符/通配符
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

# 转人工时交给客服的机器人会话条数上限（最近 N 条，够看清「聊到哪了」即可）
_PRE_HANDOFF_LIMIT = 20


def scoped_session_id(user_id: str, session_id: str) -> str:
    """把「登录用户 + 前端会话 ID」组合成存储层使用的作用域会话 ID。

    格式 `u{user_id}_{session_id}` 与 mysql_store._owner_from_conversation_id
    互为逆运算（user_id 是 32 位 hex、不含下划线），两处必须一致。
    """
    return f"u{user_id}_{session_id}"


def resolve_session_id(user_id: str, session_id: str = "") -> str:
    """校验前端传来的会话 ID，缺失或非法则新生成，返回作用域会话 ID。"""
    if not session_id or not _SESSION_ID_RE.match(session_id):
        session_id = uuid.uuid4().hex
    return scoped_session_id(user_id, session_id)


def is_valid_session_id(session_id: str) -> bool:
    """判断前端传来的会话 ID 是否合法（用于反馈等只需可追溯的场景）。"""
    return bool(session_id) and bool(_SESSION_ID_RE.match(session_id))


def _cache_key(scoped_id: str) -> str:
    """该会话在 Redis 热缓存中的 key（与 MysqlChatHistory.key 保持一致）。"""
    return f"{REDIS_CONFIG['key_prefix']}{scoped_id}"


async def _drop_cache(scoped_id: str) -> None:
    """尽力而为地清掉热缓存：删失败只记日志，不影响「以库为准」的结果。"""
    try:
        from infrastructure.redis.connection import get_redis_async
        client = await get_redis_async()
        await client.delete(_cache_key(scoped_id))
    except Exception as e:
        import logging
        logging.getLogger("history").warning(f"清理会话热缓存失败（忽略）: {e}")


async def list_history_sessions(user_id: str) -> List[Dict]:
    """
    列出当前用户的历史会话，进行中的排在最前，组内按最后更新时间倒序。
    返回: [{session_id, title, message_count, updated_at, status}]
    status: open=进行中(可续接) / closed=已结束(仅可回看)
    session_id 为前端会话 ID（已剥掉归属账号前缀），可直接回传续接。
    """
    uid = (user_id or "").strip()
    if not uid:
        return []
    try:
        # pymysql 是同步阻塞调用，放线程池执行，避免读历史时卡住整个事件循环
        rows, _ = await asyncio.to_thread(mysql_store.list_conversations_by_user, uid)
    except Exception as e:
        import logging
        logging.getLogger("history").warning(f"读取历史会话失败（降级为空）: {e}")
        return []

    prefix = f"u{uid}_"
    sessions = []
    for r in rows:
        scoped_id = r.get("conversation_id") or ""
        # 防御：库里混入非本账号命名格式的行时跳过，避免把内部 ID 透给前端
        if not scoped_id.startswith(prefix):
            continue
        sessions.append({
            "session_id": scoped_id[len(prefix):],
            "title": (r.get("title") or "").strip() or "（新会话）",
            "message_count": r.get("message_count") or 0,
            "updated_at": r.get("updated_at") or "",
            "status": r.get("status") or "open",
        })
    # 稳定排序：进行中的会话置顶，组内仍保持更新时间倒序（前端「续接」即取第一条 open）
    sessions.sort(key=lambda x: x["status"] != "open")
    return sessions


async def get_history_messages(user_id: str, session_id: str) -> List[Dict]:
    """
    读取某会话的消息，返回 [{role: 'user'|'assistant', content, timestamp}]
    若会话不存在、session_id 非法或不属于当前账号，返回空列表。
    """
    if not session_id or not _SESSION_ID_RE.match(session_id):
        return []
    scoped_id = scoped_session_id(user_id, session_id)
    try:
        rows, _ = await asyncio.to_thread(mysql_store.list_messages, scoped_id)
    except Exception as e:
        import logging
        logging.getLogger("history").warning(f"读取会话历史失败（降级为空）: {e}")
        return []
    return [
        {
            "role": "user" if (r.get("role") == "user") else "assistant",
            "content": r.get("content") or "",
            "timestamp": r.get("created_at") or "",
        }
        for r in rows
        if (r.get("content") or "").strip()
    ]


async def delete_history_session(user_id: str, session_id: str) -> bool:
    """删除当前用户的某个会话（连同消息）；返回该会话是否真的存在并被删除。"""
    if not is_valid_session_id(session_id):
        return False
    scoped_id = scoped_session_id(user_id, session_id)
    try:
        # 归属校验在 SQL 里（WHERE user_id=%s）：非本用户的会话删除影响 0 行，返回 False
        deleted = await asyncio.to_thread(mysql_store.delete_conversation, scoped_id, user_id)
    except Exception as e:
        import logging
        logging.getLogger("history").warning(f"删除会话失败: {e}")
        return False
    if deleted:
        await _drop_cache(scoped_id)
    return deleted


async def get_pre_handoff_messages(user_id: str, before_time: str,
                                   limit: int = _PRE_HANDOFF_LIMIT) -> List[Dict]:
    """转人工那一刻之前，该账号与机器人那场对话的记录（客服接手时的上下文）。

    「转人工前的机器人会话」在存储上没有关联字段，按「归属账号 + 工单创建时间之前最近
    活跃的一场会话」反查（见 mysql_store.get_latest_conversation_before）。
    查不到（用户没聊过就直接转人工）返回空列表，调用方据此不展示该区域。
    只取最近 limit 条：客服需要的是「刚才聊到哪了」，不是完整档案。
    """
    uid = (user_id or "").strip()
    conv = await asyncio.to_thread(
        mysql_store.get_latest_conversation_before, uid, before_time)
    if not conv:
        return []
    scoped_id = conv.get("conversation_id") or ""
    prefix = f"u{uid}_"
    # 防御：库里混入非本账号命名格式的行时直接放弃，不把内部 ID 透出去
    if not uid or not scoped_id.startswith(prefix):
        return []
    msgs = await get_history_messages(uid, scoped_id[len(prefix):])
    return msgs[-limit:] if limit > 0 else msgs


async def close_history_session(user_id: str, session_id: str) -> bool:
    """把某个会话标记为「已结束」：之后不再被续接，但历史仍可回看。

    对应用户端「结束本次咨询」——结束后用户再发消息即开启新会话（现实客服的做法：
    会话单位跟着一次诉求走，用户端不提供随手「新建对话」）。
    """
    if not is_valid_session_id(session_id):
        return False
    scoped_id = scoped_session_id(user_id, session_id)
    try:
        # 只对「自己名下」的会话生效；从未发过消息的空会话无历史可回看，返回 False
        conv = await asyncio.to_thread(mysql_store.get_conversation, scoped_id, user_id)
        if not conv:
            return False
        if conv.get("status") == "closed":
            return True  # 幂等：重复结束同一会话仍算成功
        return await asyncio.to_thread(mysql_store.close_conversation, scoped_id, user_id)
    except Exception as e:
        import logging
        logging.getLogger("history").warning(f"结束会话失败: {e}")
        return False