"""流式产出骨架：生产者任务、TTFT 打点、会话历史读写与收尾落库。

Agent 与 RAG 两条流式入口共用这一套骨架：
生成跑在独立任务里，与 HTTP 连接解耦；客户端断连只影响转发，该轮历史照常完整落库。
"""
import asyncio
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from logs.log_config import log
from utils.metrics import agent_ttft_seconds

# 后台生成任务集合：持有引用防止任务被 GC 回收（生成与 HTTP 连接解耦后需独立跑完）
_BG_TASKS: set = set()


class _TTFTRecorder:
    """首 token 延迟记录器：一轮只记一次（TTFT 是流式体验的核心指标）。

    起算点由调用方给出（Agent 取 start_turn 入口、RAG 取 stream 入口），
    因此包含意图识别、历史加载、DST 等「用户实际在等」的前置耗时，而不只是模型时间。
    """

    def __init__(self, started_at: float, path: str):
        self._started_at = started_at
        self._path = path
        self._done = False

    def mark(self) -> None:
        """首次调用时记一次耗时；后续调用直接返回（幂等）。"""
        if self._done:
            return
        self._done = True
        agent_ttft_seconds.labels(self._path).observe(
            max(time.perf_counter() - self._started_at, 0.0))


class _StreamQueue:
    """产出侧队列包装：首次写入用户可见内容时记录 TTFT，其余行为与原生队列一致。

    只包装生产者使用的 put；哨兵 None 由 _spawn_producer 用真实队列写入，
    因此不会把「结束标记」误算成首 token。
    """

    def __init__(self, queue: asyncio.Queue, recorder: _TTFTRecorder):
        self._queue = queue
        self._recorder = recorder

    async def put(self, item: Any) -> None:
        self._recorder.mark()
        await self._queue.put(item)


def _spawn_producer(coro, queue: asyncio.Queue) -> None:
    """启动后台生成任务：结束送 None 哨兵，并记录未捕获异常。

    生成与 HTTP 连接解耦的公共骨架（RAG 与 Agent 两条流式入口共用）：
    客户端中途断开时 Starlette 只关闭下游的转发生成器，上游生成任务照常跑完，
    历史照常落库——既不会把半截答复当完整答复写进历史，也不会断连丢整轮。
    """
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)

    def _on_done(task: asyncio.Task):
        _BG_TASKS.discard(task)
        if not task.cancelled() and task.exception():
            log.error(f"生成任务异常: {task.exception()}", exc_info=task.exception())
        queue.put_nowait(None)

    task.add_done_callback(_on_done)


# ========== 会话历史读写（两条流式入口共用）==========
async def load_chat_history(chat_history):
    """
    读取会话历史：优先异步（MysqlChatHistory.amessages，事件循环内非阻塞），
    兼容 InMemoryChatMessageHistory 兜底（sync messages 属性/方法）。
    """
    if hasattr(chat_history, "amessages"):
        return await chat_history.amessages()
    msgs = getattr(chat_history, "messages", None)
    return msgs() if callable(msgs) else (msgs or [])


async def append_history_message(chat_history, message):
    """追加一条会话历史：优先异步写库，内存兜底直接同步写。"""
    if hasattr(chat_history, "async_add_message"):
        await chat_history.async_add_message(message)
    else:
        chat_history.add_message(message)


async def finalize_turn(chat_history, question: str, answer: str):
    """落库本轮问答（提问 + 完整答复）。

    生成与 HTTP 连接解耦后必须由生成任务自己写完：响应结束时 Starlette 不会再执行
    BackgroundTasks，若仍挂在背景任务上，客户端一断开就会丢掉（或只留下半截）历史。
    """
    try:
        await append_history_message(chat_history, HumanMessage(content=question))
        await append_history_message(chat_history, AIMessage(content=answer))
    except Exception as e:
        log.error(f"存储对话历史失败: {e}", exc_info=True)
