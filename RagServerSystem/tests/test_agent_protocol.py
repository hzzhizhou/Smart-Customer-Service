"""B2 协议统一回归：只用原生 tool calling（无 ReAct 文本协议）。

对应 PROJECT_AUDIT §5.2。锁住三件事：
  1) 正文不再依赖 "Final Answer:" 标记：非工具内容块直接流式透出；
  2) 工具调用块（tool_call_chunks）与推理块（reasoning_content）绝不当正文透出；
  3) 空答复占位（「无相关信息」）改走兜底、无正文时转知识兜底；system prompt 不含 ReAct 段。

约定：纯内存桩件，不连 MySQL/Redis/LLM；同步用例内用 asyncio.run(drive()) 驱动。
"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import agent.service as ag_svc
import service.turn.agent_turn as agent_turn
from service.handlers import TurnContext
from service.turn_service import AgentTurnService, TurnState


# ---------- 桩件 ----------
class _ChunkMsg:
    """模拟 AIMessageChunk：正文 / 工具调用块 / 推理内容三选一。"""

    def __init__(self, content="", tool_call_chunks=None, reasoning=None):
        self.content = content
        self.tool_call_chunks = tool_call_chunks
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}


class _StubAgent:
    """按脚本吐 on_chat_model_stream 事件（chunk 为 _ChunkMsg）。"""

    def __init__(self, chunks=None):
        self._chunks = chunks or []

    async def astream_events(self, inputs, config=None, version=None):
        for c in self._chunks:
            yield {"event": "on_chat_model_stream", "name": "ChatModel",
                   "run_id": "m1", "data": {"chunk": c}}


class _HistoryStub:
    def __init__(self):
        self.messages = []

    async def async_add_message(self, message):
        self.messages.append(message)


class _RetrievalStub:
    async def retrieve(self, **kwargs):
        return ([Document(page_content="商品资料")], "vector",
                {"level": "high", "should_escalate": False, "top_score": 0.9})


class _GeneratorStub:
    async def stream_generate(self, question, docs, session_id):
        yield "知识兜底答复"


def _turn(question="订单到哪了", session_id="u1__t1"):
    ctx = TurnContext(
        retrieval_service=_RetrievalStub(), answer_generator=_GeneratorStub(),
        dst_manager=None, slot_extractor=None, question=question,
        session_id=session_id, user={"user_id": "u1"}, dst=None,
        intent_info={"intent": "consult"})
    return TurnState(ctx=ctx, messages=[], chat_history=_HistoryStub())


def _svc(agent):
    return AgentTurnService(
        retrieval_service=_RetrievalStub(), answer_generator=_GeneratorStub(),
        unified_agent=agent, intent_classifier=None, dst_manager=None,
        slot_extractor=None)


def _drive(svc, turn):
    async def drive():
        queue = asyncio.Queue()
        final_text = []
        await svc._produce_agent(turn, queue, final_text)
        return [queue.get_nowait() for _ in range(queue.qsize())]

    return asyncio.run(drive())


@pytest.fixture(autouse=True)
def _no_dst(monkeypatch):
    # DST_ENABLED 在编排模块内是模块级常量，必须 patch 真正的定义模块（facade 的副本不生效）
    monkeypatch.setattr(agent_turn, "DST_ENABLED", False)


@pytest.fixture(autouse=True)
def _reset_llm_breaker():
    from utils.circuit_breaker import llm_breaker
    llm_breaker.record_success()
    yield
    llm_breaker.record_success()


# ===================== 1) 正文透出 =====================
class TestNativeStreaming:
    def test_answer_streams_without_marker(self):
        svc = _svc(_StubAgent(chunks=[_ChunkMsg("您好，"), _ChunkMsg("订单已发货")]))
        pieces = _drive(svc, _turn())
        assert "".join(pieces) == "您好，订单已发货"

    def test_tool_call_chunks_not_leaked(self):
        svc = _svc(_StubAgent(chunks=[
            _ChunkMsg("", tool_call_chunks=[{"name": "query_order"}]),
            _ChunkMsg("已为您查到，正在配送中"),
        ]))
        assert _drive(svc, _turn()) == ["已为您查到，正在配送中"]

    def test_reasoning_content_not_leaked(self):
        svc = _svc(_StubAgent(chunks=[
            _ChunkMsg("我需要先查一下订单", reasoning="我需要先查一下订单"),
            _ChunkMsg("订单已于昨日发货"),
        ]))
        answer = "".join(_drive(svc, _turn()))
        assert answer == "订单已于昨日发货"
        assert "查一下订单" not in answer

    def test_empty_chunk_ignored(self):
        svc = _svc(_StubAgent(chunks=[_ChunkMsg(""), _ChunkMsg("正常答复")]))
        assert _drive(svc, _turn()) == ["正常答复"]


# ===================== 2) 空答复与兜底 =====================
class TestNativeFallbacks:
    def test_empty_placeholder_replaced(self):
        svc = _svc(_StubAgent(chunks=[_ChunkMsg("无相关信息")]))
        answer = "".join(_drive(svc, _turn("京东同款多少钱")))
        assert "无相关信息" not in answer
        assert "没有查到可靠的资料" in answer

    def test_no_text_falls_back_to_knowledge(self):
        # 只调了工具、没有正文块 → 走确定性知识兜底，用户不能拿到空白
        svc = _svc(_StubAgent(chunks=[]))
        assert _drive(svc, _turn()) == ["知识兜底答复"]


# ===================== 3) system prompt 无 ReAct 段 =====================
def test_system_prompt_has_no_react_section(monkeypatch):
    captured = {}
    monkeypatch.setattr(ag_svc, "create_agent",
                        lambda **kw: (captured.update(kw), object())[1])
    ag_svc.create_unified_agent(object(), llm=object())
    prompt = captured["system_prompt"]
    assert "ReAct 格式" not in prompt
    assert "Final Answer" not in prompt
    assert "结构化方式调用" in prompt
