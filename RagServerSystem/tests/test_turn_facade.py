"""重构守护：service/turn_service.py 是 re-export facade，实现住在 service/turn/ 各模块。

断言三件事：
  1) facade 暴露的对象与领域模块里的对象是同一个（is），不是副本；
  2) facade 覆盖拆分前对外可见的名字 —— 防止以后只改领域模块、漏了 re-export；
  3) facade 里没有实现（无 class / async def），新增逻辑不该再塞回这个文件。
"""
import inspect

import service.turn_service as turn_service
import service.turn.agent_turn as agent_turn
import service.turn.rag_turn as rag_turn
import service.turn.state as turn_state
import service.turn.streaming as streaming

# 拆分前 turn_service.py 的模块级名字（显式列出，新增/删除时同步维护）
_EXPORTED_NAMES = [
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


def test_facade_reexports_are_same_objects():
    assert turn_service.AgentTurnService is agent_turn.AgentTurnService
    assert turn_service.RagTurnService is rag_turn.RagTurnService
    assert turn_service.TurnState is turn_state.TurnState
    assert turn_service._redact_args is turn_state._redact_args
    assert turn_service._result_digest is turn_state._result_digest
    assert turn_service._TTFTRecorder is streaming._TTFTRecorder
    assert turn_service._StreamQueue is streaming._StreamQueue
    assert turn_service._spawn_producer is streaming._spawn_producer
    assert turn_service._gate_headers is rag_turn._gate_headers
    assert turn_service._BG_TASKS is streaming._BG_TASKS


def test_facade_covers_all_exported_names():
    missing = [name for name in _EXPORTED_NAMES
               if getattr(turn_service, name, None) is None]
    assert missing == []


def test_facade_has_no_implementation():
    # facade 只做搬运：出现 class/async def 就说明逻辑又被塞回来了
    source = inspect.getsource(turn_service)
    assert "class " not in source
    assert "async def" not in source
