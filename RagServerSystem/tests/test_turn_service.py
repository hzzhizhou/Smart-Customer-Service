"""轮次编排（turn_service）回归测试：流式生成与 HTTP 连接解耦后的两个关键行为。

回归防护（两条均为此前线上逻辑缺陷，2026-09-26 修复）：
1. Agent 流式中途异常时，旧代码重推 final_text[-1]——该段已流给客户端，
   用户会看到末尾内容重复一遍；且一段正文都没有时用户拿到空响应。
2. RAG 路径历史挂在 BackgroundTasks 上（Agent 路径已改为生成任务直写），
   客户端断连时 Starlette 不再执行背景任务，该轮历史直接丢失；
   现与 Agent 同一套生产者骨架：断连只影响转发，历史照常完整落库。

测试均为纯内存桩件（不连 MySQL/Redis/LLM），历史句柄用 _HistoryStub 记录写入。
"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import service.turn.agent_turn as agent_turn
import service.turn.rag_turn as rag_turn
from service.cache import ResponseCache
from service.handlers import TurnContext
from service.turn_service import AgentTurnService, RagTurnService, TurnState
from shared.reply_templates import AGENT_NO_ANSWER_FALLBACK
from shared.schemas import RAGRequest


# ---------- 桩件 ----------

class _HistoryStub:
    """会话历史桩：记录 finalize_turn 写入的消息。"""

    def __init__(self):
        self.messages = []

    async def async_add_message(self, message):
        self.messages.append(message)


class _StubGenerator:
    """答案生成器桩：stream_generate 按脚本吐块，可中途抛异常。"""

    def __init__(self, chunks=None, error=None):
        self._chunks = chunks or []
        self._error = error

    async def stream_generate(self, question, docs, session_id):
        for c in self._chunks:
            yield c
        if self._error is not None:
            raise self._error


class _StubAgent:
    """统一 Agent 桩：astream_events 吐 on_chat_model_stream 事件，可中途抛异常。"""

    def __init__(self, contents=None, error=None):
        self._contents = contents or []
        self._error = error

    async def astream_events(self, inputs, config=None, version=None):
        for c in self._contents:
            yield {"event": "on_chat_model_stream",
                   "data": {"chunk": SimpleNamespace(
                       message=SimpleNamespace(tool_call_chunks=None, content=c))}}
        if self._error is not None:
            raise self._error


class _StubRetrieval:
    """检索服务桩：retrieve 返回固定 (docs, retriever_type, gate_info)。"""

    def __init__(self, gate_level="high"):
        self.gate_info = {"level": gate_level, "should_escalate": False,
                          "top_score": 0.9}

    async def retrieve(self, **kwargs):
        return [], "vector", self.gate_info


def _agent_turn(question: str, history: _HistoryStub,
                agent: _StubAgent) -> TurnState:
    """手工组装一轮 Agent 上下文（绕开 _prepare，不触达 MySQL/DST/LLM）。"""
    ctx = TurnContext(
        retrieval_service=None, answer_generator=None, dst_manager=None,
        slot_extractor=None, question=question, session_id="u1__t1",
        user={"user_id": "u1"}, dst=None,
        intent_info={"intent": "consult"},
    )
    return TurnState(ctx=ctx, messages=[], chat_history=history)


@pytest.fixture(autouse=True)
def _no_dst(monkeypatch):
    """关闭 DST 更新（本文件不测 DST，只测流式产出与落库）。"""
    monkeypatch.setattr(agent_turn, "DST_ENABLED", False)


@pytest.fixture(autouse=True)
def _reset_llm_breaker():
    """熔断器是模块级单例：用例间须复位，避免用例触发的失败累计把状态推进 OPEN 而相互影响。"""
    from utils.circuit_breaker import llm_breaker
    llm_breaker.record_success()   # record_success 会把状态复位为 CLOSED 并清零计数
    yield
    llm_breaker.record_success()


@pytest.fixture
def history(monkeypatch):
    """init_chat_history 替换为内存桩：RAG 生产者经它落库，断言直接读桩。"""
    stub = _HistoryStub()
    # init_chat_history 在 RAG 编排模块内被调用，patch 必须打到该模块的全局名
    monkeypatch.setattr(rag_turn, "init_chat_history", lambda sid: stub)
    return stub


# ---------- Agent 路径：异常不重复、不空响应 ----------

def test_agent_exception_midstream_no_duplicate_tail():
    """中途异常：已流出的段不得重推（旧代码会重复末段一遍）。"""
    svc = AgentTurnService(
        retrieval_service=None, answer_generator=None,
        unified_agent=_StubAgent(
            contents=["您好，", "订单已发货"],
            error=RuntimeError("LLM 连接中断")),
        intent_classifier=None, dst_manager=None, slot_extractor=None)
    hist = _HistoryStub()
    turn = _agent_turn("订单到哪了", hist, svc.unified_agent)

    async def drive():
        queue = asyncio.Queue()
        await svc._produce_agent(turn, queue, [])
        out = []
        while not queue.empty():
            out.append(queue.get_nowait())
        return out

    pieces = asyncio.run(drive())
    # 正文各流一次：您好，、订单已发货——没有第三段重复
    assert pieces == ["您好，", "订单已发货"]
    answer = "".join(pieces)
    assert answer == "您好，订单已发货"
    # 落库一次、内容与流出一致
    assert [type(m).__name__ for m in hist.messages] == ["HumanMessage", "AIMessage"]
    assert hist.messages[1].content == answer


def test_agent_exception_before_any_content_gets_fallback():
    """一段正文都没产出就异常：必须补兜底话术，用户不能拿到空响应。"""
    svc = AgentTurnService(
        retrieval_service=None, answer_generator=None,
        unified_agent=_StubAgent(error=RuntimeError("LLM 连接中断")),
        intent_classifier=None, dst_manager=None, slot_extractor=None)
    hist = _HistoryStub()
    turn = _agent_turn("退货政策", hist, svc.unified_agent)

    async def drive():
        queue = asyncio.Queue()
        await svc._produce_agent(turn, queue, [])
        return [queue.get_nowait() for _ in range(queue.qsize())]

    pieces = asyncio.run(drive())
    assert pieces == [AGENT_NO_ANSWER_FALLBACK]
    assert hist.messages[-1].content == AGENT_NO_ANSWER_FALLBACK


def _drive_agent(svc, turn):
    async def drive():
        queue = asyncio.Queue()
        await svc._produce_agent(turn, queue, [])
        return [queue.get_nowait() for _ in range(queue.qsize())]

    return asyncio.run(drive())


def test_agent_empty_placeholder_replaced():
    """整段正文是「无相关信息」这类占位答复 → 换兜底话术，不能原样抛给用户。

    回归防护：模型无据可依时会吐「无相关信息」当正文；此前该段直接透传，
    用户拿到的回答只有四个字（实测 J2「京东上同款 iPhone 15 卖多少钱」）。
    """
    svc = AgentTurnService(
        retrieval_service=None, answer_generator=None,
        unified_agent=_StubAgent(contents=["无相关信息"]),
        intent_classifier=None, dst_manager=None, slot_extractor=None)
    hist = _HistoryStub()
    turn = _agent_turn("京东上同款 iPhone 15 卖多少钱", hist, svc.unified_agent)

    answer = "".join(_drive_agent(svc, turn))
    assert "无相关信息" not in answer
    assert "没有查到可靠的资料" in answer


def test_agent_normal_answer_passes_through():
    """正常正文（不是空答复候选）必须即时透出，不因缓冲而丢失。"""
    svc = AgentTurnService(
        retrieval_service=None, answer_generator=None,
        unified_agent=_StubAgent(contents=["iPhone 15 售价 4999 元"]),
        intent_classifier=None, dst_manager=None, slot_extractor=None)
    hist = _HistoryStub()
    turn = _agent_turn("iPhone 15 多少钱", hist, svc.unified_agent)

    answer = "".join(_drive_agent(svc, turn))
    assert answer.strip() == "iPhone 15 售价 4999 元"
    assert hist.messages[-1].content == answer


# ---------- 知识作答：空答复不得原样透出 ----------

class _RetrievalStub:
    """检索桩：命中若干文档且门控为高置信（保证走到生成阶段）。"""

    async def retrieve(self, **kwargs):
        return ([Document(page_content="商品资料")], "vector",
                {"level": "high", "should_escalate": False, "top_score": 0.9})


def test_knowledge_answer_empty_placeholder_replaced():
    """检索命中但资料答不到问题时，模型会把「无相关信息」当正文 → 必须换兜底话术。

    回归防护：这四字此前原样透出（实测 J2「京东上同款 iPhone 15 卖多少钱」）。
    """
    from service.handlers.knowledge import stream_knowledge_answer

    async def drive():
        return "".join([c async for c in stream_knowledge_answer(
            _RetrievalStub(), _StubGenerator(chunks=["无", "相关信息"]), "京东同款多少钱", "s1")])

    answer = asyncio.run(drive())
    assert "无相关信息" not in answer
    assert "没有查到可靠的资料" in answer


def test_knowledge_answer_normal_passes_through():
    from service.handlers.knowledge import stream_knowledge_answer

    async def drive():
        return "".join([c async for c in stream_knowledge_answer(
            _RetrievalStub(), _StubGenerator(chunks=["iPhone 15 售价 ", "4999 元"]), "iPhone 15 多少钱", "s1")])

    assert asyncio.run(drive()) == "iPhone 15 售价 4999 元"


# ---------- RAG 路径：历史与 HTTP 连接解耦 ----------

def test_rag_history_written_even_if_client_never_consumes(history):
    """断连回归：客户端完全不消费流（等价于立刻断开），历史仍完整落库。

    旧实现历史挂 BackgroundTasks：响应未完成时背景任务不执行，断连即丢整轮。
    """
    svc = RagTurnService(
        retrieval_service=_StubRetrieval(),
        answer_generator=_StubGenerator(chunks=["7天无理由退货", "需保持商品完好"]))
    request = RAGRequest(question="退货政策是什么", session_id=None)

    async def drive():
        relay, headers = await svc.stream(request, "u1__t2")
        # 模拟客户端断连：不消费 relay，轮询等待生产者独立跑完（上限 3s）
        for _ in range(300):
            if len(history.messages) >= 2:
                break
            await asyncio.sleep(0.01)
        return headers

    headers = asyncio.run(drive())
    assert headers["X-Gate-Level"] == "high"
    # 历史已由生产者直写：完整一问一答
    assert [type(m).__name__ for m in history.messages] == ["HumanMessage", "AIMessage"]
    assert history.messages[0].content == "退货政策是什么"
    assert history.messages[1].content == "7天无理由退货需保持商品完好"
    # 成功完成：同问缓存已写入（下一轮命中秒回）
    key = svc.cache.make_key(request.question, request.route_mode)
    assert svc.cache.get(key)["answer"] == "7天无理由退货需保持商品完好"


def test_rag_generation_failure_returns_fallback_and_skips_cache(history):
    """生成一开始就失败：用户拿到兜底话术而非空响应；失败结果不污染缓存。"""
    svc = RagTurnService(
        retrieval_service=_StubRetrieval(),
        answer_generator=_StubGenerator(error=RuntimeError("LLM 连接中断")))
    request = RAGRequest(question="保修多久", session_id=None)

    async def drive():
        relay, _ = await svc.stream(request, "u1__t3")
        out = []
        async for piece in relay:
            out.append(piece)
        return out

    pieces = asyncio.run(drive())
    assert pieces == [AGENT_NO_ANSWER_FALLBACK]
    assert history.messages[-1].content == AGENT_NO_ANSWER_FALLBACK
    key = svc.cache.make_key(request.question, request.route_mode)
    assert svc.cache.get(key) is None


def test_rag_midstream_failure_keeps_partial_answer_out_of_cache(history):
    """生成中途失败：已流出的部分照常透出与落库，但半截答复绝不入缓存。"""
    svc = RagTurnService(
        retrieval_service=_StubRetrieval(),
        answer_generator=_StubGenerator(
            chunks=["保修期为一年"], error=RuntimeError("LLM 连接中断")))
    request = RAGRequest(question="保修期多久", session_id=None)

    async def drive():
        relay, _ = await svc.stream(request, "u1__t4")
        out = []
        async for piece in relay:
            out.append(piece)
        return out

    pieces = asyncio.run(drive())
    assert pieces == ["保修期为一年"]
    assert history.messages[-1].content == "保修期为一年"
    key = svc.cache.make_key(request.question, request.route_mode)
    assert svc.cache.get(key) is None
