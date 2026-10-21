"""轮次服务层（兼容 facade）：一次请求 = 一轮问答的编排入口。

实现已按职责拆分到 service/turn/ 包：
- service/turn/streaming.py  生产者任务、TTFT 打点、历史读写与收尾落库；
- service/turn/state.py      TurnState 与工具入参脱敏/结果摘要等纯工具；
- service/turn/agent_turn.py AgentTurnService（历史窗口 / 意图 / DST / 确定性分支 / ReAct 兜底）；
- service/turn/rag_turn.py   RagTurnService（无状态检索问答 + 缓存 + 门控响应头）。

本文件只做 re-export，保证 `from service.turn_service import ...` 的既有调用方无需改动。
注意：这里 re-export 的是对象引用，运行时补丁要打到真正的定义模块
（如 service.turn.agent_turn.DST_ENABLED），patch 本 facade 的模块属性不会生效。
"""
from config.settings import CHAT_HISTORY_WINDOW, DST_ENABLED
from infrastructure import mysql_store
from utils.metrics import agent_tool_calls_total, agent_tool_duration_seconds

from service.turn.agent_turn import AgentTurnService
from service.turn.rag_turn import RagTurnService, _gate_headers
from service.turn.state import (
    TurnState,
    _redact_args,
    _result_digest,
    _summarize_value,
)
from service.turn.streaming import (
    _BG_TASKS,
    _StreamQueue,
    _TTFTRecorder,
    _spawn_producer,
    append_history_message,
    finalize_turn,
    load_chat_history,
)

__all__ = [
    "AgentTurnService",
    "RagTurnService",
    "TurnState",
    "_summarize_value",
    "_redact_args",
    "_result_digest",
    "_TTFTRecorder",
    "_StreamQueue",
    "_spawn_producer",
    "_BG_TASKS",
    "load_chat_history",
    "append_history_message",
    "finalize_turn",
    "_gate_headers",
]
