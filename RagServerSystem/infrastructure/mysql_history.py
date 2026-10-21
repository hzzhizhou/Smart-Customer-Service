"""对话历史后端：MySQL 为准 + Redis 只做热缓存。

为什么这么改（此前历史只存在 Redis 的单个 JSON 数组里）：
  - Redis 一挂或 key 到期，历史就整段消失，用户看到的「历史会话」直接变空；
  - 全量 JSON 数组的 GET→改→SET 会随轮次线性放大读写量，同会话并发写还会丢消息。
现在：
  - 权威存储在 MySQL（infrastructure.mysql_store 的 conversations / messages 表），
    逐条 INSERT，进程重启、Redis 被清空都不丢，且能按会话分页查；
  - Redis 只缓存最近 CACHE_WINDOW 条消息（LLM 上下文窗口只用最近 6 条），
    命中则省一次查库，未命中/不可用就回源查库并回填，写缓存失败只记日志。
"""
import asyncio
import json
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config.settings import REDIS_CONFIG
from infrastructure import mysql_store
from infrastructure.base_chat_history import BaseChatHistory
from logs.log_config import chat_history_log as log

# Redis 热缓存保留的消息条数（够 LLM 上下文窗口用即可，避免缓存无界增长）
CACHE_WINDOW = 20


class MysqlChatHistory(BaseChatHistory):
    def __init__(self, session_id: str) -> None:
        if not session_id or not isinstance(session_id, str):
            raise ValueError("session_id 不能为空且必须为字符串")
        self.session_id = session_id
        self.key = f"{REDIS_CONFIG['key_prefix']}{self.session_id}"
        self.expire_seconds = REDIS_CONFIG["expire_days"] * 24 * 60 * 60

    # ===================== 权威存储：MySQL =====================
    @staticmethod
    def _to_messages(rows) -> List[BaseMessage]:
        messages = []
        for row in rows:
            content = (row.get("content") or "").strip()
            if not content:
                continue
            messages.append(
                HumanMessage(content=content) if row.get("role") == "user"
                else AIMessage(content=content)
            )
        return messages

    def _db_messages(self) -> List[BaseMessage]:
        try:
            rows, _ = mysql_store.list_messages(self.session_id)
            return self._to_messages(rows)
        except Exception as e:
            log.error(f"读取会话历史失败（MySQL）: {e}")
            return []

    def _db_tail(self, limit: int = CACHE_WINDOW) -> List[BaseMessage]:
        """取最近 limit 条（先拿总数再按偏移取尾页），用于缓存缺失时回填。"""
        try:
            _, total = mysql_store.list_messages(self.session_id, limit=1)
            rows, _ = mysql_store.list_messages(
                self.session_id, limit=limit, offset=max(0, total - limit)
            )
            return self._to_messages(rows)
        except Exception as e:
            log.error(f"读取会话历史尾部失败（MySQL）: {e}")
            return []

    def _db_append(self, message: BaseMessage) -> None:
        """权威写入：失败向上抛，确保「存不进库」能被上层发现，而不是静默丢历史。"""
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        mysql_store.append_message(self.session_id, role, message.content)
        log.info(f"会话[{self.session_id}] 历史已落库（{role}）")

    # ===================== 热缓存：Redis（尽力而为） =====================
    @staticmethod
    def _dump(messages: List[BaseMessage]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data = [
            {"type": "human" if isinstance(m, HumanMessage) else "ai",
             "content": m.content, "timestamp": now}
            for m in messages
        ]
        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse(history_str) -> Optional[List[BaseMessage]]:
        """解析缓存内容；空值或格式不认识返回 None（一律当作未命中，回源查库）。"""
        if not history_str:
            return None
        try:
            data = json.loads(history_str)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, list):
            return None
        messages = []
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content") or ""
            if not content:
                continue
            messages.append(
                HumanMessage(content=content) if item.get("type") == "human"
                else AIMessage(content=content)
            )
        return messages or None

    def _trim(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        return messages[-CACHE_WINDOW:]

    # ===================== 【异步 · 事件循环内使用】 =====================
    async def amessages(self) -> List[BaseMessage]:
        """【异步】读取历史消息：先查缓存，未命中回源查库并回填缓存。"""
        try:
            from infrastructure.redis.connection import get_redis_async
            client = await get_redis_async()
            messages = self._parse(await client.get(self.key))
            if messages is not None:
                log.debug(f"📖 会话[{self.session_id}] 命中热缓存 {len(messages)} 条")
                return messages
        except Exception as e:
            log.warning(f"读取历史缓存失败，回源查库: {e}")

        messages = await asyncio.to_thread(self._db_messages)
        log.debug(f"📖 会话[{self.session_id}] 从 MySQL 读取历史 {len(messages)} 条")
        await self._cache_set(self._trim(messages))
        return messages

    async def async_add_message(self, message: BaseMessage):
        """【异步】添加消息：先落库（权威），再更新热缓存（失败不影响对话）。"""
        if not isinstance(message, (HumanMessage, AIMessage)):
            raise ValueError(f"仅支持HumanMessage/AIMessage，当前类型：{type(message)}")
        # 落库是同步阻塞调用（pymysql），放线程池执行，避免写历史时卡住事件循环
        await asyncio.to_thread(self._db_append, message)
        await self._cache_append(message)

    async def aclear(self):
        """【异步】清空历史（库 + 缓存）。"""
        try:
            await asyncio.to_thread(mysql_store.clear_messages, self.session_id)
            from infrastructure.redis.connection import get_redis_async
            client = await get_redis_async()
            await client.delete(self.key)
            log.info(f"会话[{self.session_id}]会话历史已清空")
        except Exception as e:
            log.error(f"【异步】清空历史失败 {e}")
            raise

    async def _cache_set(self, messages: List[BaseMessage]) -> None:
        """把消息列表写进缓存（空列表不写，避免「空缓存=命中」的歧义）。"""
        if not messages:
            return
        try:
            from infrastructure.redis.connection import get_redis_async
            client = await get_redis_async()
            await client.set(self.key, self._dump(messages), ex=self.expire_seconds)
        except Exception as e:
            log.warning(f"历史热缓存写入失败（不影响对话）: {e}")

    async def _cache_append(self, message: BaseMessage) -> None:
        """在缓存尾部追加一条：缓存缺失时先回源库尾做基线，整体保持 O(窗口)。"""
        try:
            from infrastructure.redis.connection import get_redis_async
            client = await get_redis_async()
            messages = self._parse(await client.get(self.key))
            if messages is None:
                messages = await asyncio.to_thread(self._db_tail)
            await client.set(
                self.key, self._dump(self._trim(messages + [message])),
                ex=self.expire_seconds,
            )
        except Exception as e:
            log.warning(f"历史热缓存更新失败（不影响对话）: {e}")

    # ===================== 【同步 · 线程/后台任务使用】 =====================
    def messages(self) -> List[BaseMessage]:
        """
        读取历史消息（转换为LangChain的Message对象）
        ⚠️ 同步实现：仅供线程/后台任务调用；事件循环内请用 amessages()。
        """
        try:
            from infrastructure.redis.connection import get_redis_connection
            messages = self._parse(get_redis_connection().get(self.key))
            if messages is not None:
                return messages
        except Exception as e:
            log.warning(f"读取历史缓存失败，回源查库: {e}")
        return self._db_messages()

    def add_message(self, message: BaseMessage):
        """添加消息到历史（同步，线程/后台任务用；事件循环内请用 async_add_message）"""
        if not isinstance(message, (HumanMessage, AIMessage)):
            raise ValueError(f"仅支持HumanMessage/AIMessage，当前类型：{type(message)}")
        self._db_append(message)
        try:
            from infrastructure.redis.connection import get_redis_connection
            client = get_redis_connection()
            messages = self._parse(client.get(self.key))
            if messages is None:
                messages = self._db_tail()
            client.set(
                self.key, self._dump(self._trim(messages + [message])),
                ex=self.expire_seconds,
            )
        except Exception as e:
            log.warning(f"历史热缓存更新失败（不影响对话）: {e}")

    def clear(self):
        try:
            mysql_store.clear_messages(self.session_id)
            from infrastructure.redis.connection import get_redis_connection
            get_redis_connection().delete(self.key)
            log.info(f"会话[{self.session_id}]会话历史已清空")
        except Exception as e:
            log.error(f'清空历史失败 {e}')
            raise