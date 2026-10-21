"""D1 回归：同步 MySQL 调用必须挪到线程池，不得阻塞事件循环（PROJECT_AUDIT P0-4 / §2.4）。

做法：把底层同步调用换成「阻塞 0.3s」的替身，同时跑一个每 10ms 打一次点的
心跳协程。若该调用直接在事件循环里执行，阻塞期间心跳一个点都打不出来；
改走 asyncio.to_thread 后事件循环空闲，心跳应打出几十个点。

阈值取 5 而非 30：留出 CI 机器调度抖动的余量，同时与「阻塞期间 0~1 个点」
有数量级差距，判别力足够。
"""
import asyncio
import time

from langchain_core.messages import HumanMessage

from access.history import list_history_sessions
from infrastructure.mysql_history import MysqlChatHistory

_BLOCK_SECONDS = 0.3
_MIN_TICKS = 5


def _blocking_return(value):
    """构造一个「阻塞 0.3s 后返回 value」的同步替身，模拟慢 DB 调用。"""
    def _inner(*args, **kwargs):
        time.sleep(_BLOCK_SECONDS)
        return value
    return _inner


async def _ticks_during(coro):
    """执行 coro 并统计期间的心跳次数：越少说明事件循环被占得越死。"""
    ticks = []

    async def heartbeat():
        while True:
            await asyncio.sleep(0.01)
            ticks.append(1)

    hb = asyncio.create_task(heartbeat())
    try:
        await coro
    finally:
        hb.cancel()
    return len(ticks)


class TestHistoryReadOffLoop:
    def test_list_sessions_off_loop(self, monkeypatch):
        import infrastructure.mysql_store as ms
        monkeypatch.setattr(ms, "list_conversations_by_user",
                            _blocking_return(([], 0)))
        ticks = asyncio.run(_ticks_during(list_history_sessions("u1")))
        assert ticks >= _MIN_TICKS


class TestChatHistoryWriteOffLoop:
    def test_async_add_message_off_loop(self, monkeypatch):
        import infrastructure.mysql_store as ms
        monkeypatch.setattr(ms, "append_message", _blocking_return(None))

        # 屏蔽 Redis 热缓存写入，只留「落库」这一步可测（避免依赖本地 Redis）
        async def _noop(self, message):
            return None
        monkeypatch.setattr(MysqlChatHistory, "_cache_append", _noop)

        history = MysqlChatHistory("u1_abc12345")
        ticks = asyncio.run(
            _ticks_during(history.async_add_message(HumanMessage(content="你好"))))
        assert ticks >= _MIN_TICKS
