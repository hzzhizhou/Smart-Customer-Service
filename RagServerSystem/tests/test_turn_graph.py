"""主链路 LangGraph 编排回归：路由优先级、图分派与流式收尾。

迁移前 `_pick_handler` 的六条优先级分支没有单测覆盖；改为 StateGraph 后，
路由表收敛为 `route_turn` 纯函数，这里把它与「图只执行命中的分支」一并锁住。
纯内存桩件，不连 MySQL/Redis/LLM。
"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

import service.turn.agent_turn as agent_turn
from service.handlers import TurnContext
from service.turn.graph import BRANCHES, build_turn_graph, route_turn
from service.turn_service import AgentTurnService, TurnState


def _ctx(question, intent="consult", dst=None, slot_extractor=None):
    return SimpleNamespace(question=question, intent_info={"intent": intent},
                           dst=dst, slot_extractor=slot_extractor)


class _DstStub:
    def __init__(self, intent):
        self.intent = intent


# ===================== 1) 路由优先级（与原 _pick_handler 一致） =====================
class TestRoutePriority:
    def test_complaint(self):
        assert route_turn(_ctx("我要投诉", "complaint"), False) == "complaint"

    def test_progress_beats_complaint(self):
        """投诉后追问进度不得再次建单：进度关键词优先于 complaint。"""
        assert route_turn(_ctx("投诉处理得怎么样", "complaint"), False) == "ticket_progress"

    def test_order_query_needs_order_id_and_intent(self):
        assert route_turn(_ctx("帮我查订单 SO20241120005", "order"), False) == "order_query"
        # 有单号但意图不是 order/logistics：不抢订单查询分支
        assert route_turn(_ctx("订单 SO20241120005 怎么退", "return"), False) == "agent"

    def test_my_orders(self):
        assert route_turn(_ctx("我有哪些订单", "order"), False) == "my_orders"

    def test_return_refund_needs_dst_and_slot_extractor(self):
        dst = _DstStub("return")
        assert route_turn(_ctx("商品有问题", "chat", dst, object()), True) == "return_refund"
        # DST 关闭时不走退货主线
        assert route_turn(_ctx("商品有问题", "chat", dst, object()), False) == "agent"

    def test_consult_and_agent_fallback(self):
        assert route_turn(_ctx("退货政策是什么", "consult"), False) == "consult"
        assert route_turn(_ctx("你好", "chat"), False) == "agent"


# ===================== 2) 图执行：classify 先跑，条件边只读 branch =====================
def _graph_with_real_router():
    """classify 桩按生产口径计算 branch，其余节点只记录是否被执行。"""
    calls = []

    async def classify(state):
        calls.append("classify")
        return {"branch": route_turn(state["ctx"], state.get("dst_enabled", False))}

    nodes = {"classify": classify}
    for name in BRANCHES:
        async def run(state, name=name):
            calls.append(name)
            return {"branch": name}
        nodes[name] = run
    return build_turn_graph(nodes), calls


def test_graph_runs_classify_then_routed_branch():
    graph, calls = _graph_with_real_router()

    state = asyncio.run(graph.ainvoke({
        "ctx": _ctx("退货政策是什么", "consult"),
        "dst_enabled": False,
        "final_text": [],
    }))
    # classify 是固定第一步，之后只执行它判定的分支
    assert calls == ["classify", "consult"]
    assert state["branch"] == "consult"


def test_graph_executes_complaint_branch():
    """投诉意图：classify 判定 complaint，图只执行该节点，不落到 agent。"""
    graph, calls = _graph_with_real_router()

    state = asyncio.run(graph.ainvoke({
        "ctx": _ctx("我要投诉", "complaint"),
        "dst_enabled": False,
        "final_text": [],
    }))
    assert calls == ["classify", "complaint"]
    assert state["branch"] == "complaint"


def test_graph_executes_return_refund_branch():
    """DST 主线为 return 且已配置槽位抽取器：图执行 return_refund 业务节点。"""
    graph, calls = _graph_with_real_router()

    state = asyncio.run(graph.ainvoke({
        "ctx": _ctx("商品有问题", "chat", _DstStub("return"), object()),
        "dst_enabled": True,
        "final_text": [],
    }))
    assert calls == ["classify", "return_refund"]
    assert state["branch"] == "return_refund"


def test_graph_edge_reads_branch_instead_of_recomputing():
    """条件边只读 state["branch"]：classify 未写 branch 时兜底 agent，不再按 ctx 重算。"""
    calls = []

    async def classify(state):
        calls.append("classify")
        return {}          # 模拟异常/遗漏：没有写 branch

    async def agent(state):
        calls.append("agent")
        return {"branch": "agent"}

    nodes = {"classify": classify, "agent": agent}
    for name in BRANCHES:
        if name != "agent":
            nodes[name] = agent
    graph = build_turn_graph(nodes)

    state = asyncio.run(graph.ainvoke({
        # ctx 本身会被 route_turn 判成 complaint；若边重算就会走 complaint
        "ctx": _ctx("我要投诉", "complaint"),
        "dst_enabled": False,
        "final_text": [],
    }))
    assert calls == ["classify", "agent"]
    assert state["branch"] == "agent"


# ===================== 2.5) classify 节点：识别结果回填 =====================
def test_classify_node_fills_intent_and_messages(monkeypatch):
    """意图节点把识别结果回填 ctx.intent_info（路由输入）与 messages（system 注入）。"""
    monkeypatch.setattr(agent_turn, "DST_ENABLED", False)

    class _Classifier:
        async def classify(self, question, force_intent=None):
            return {"intent": "consult", "intent_cn": "知识咨询",
                    "confidence": 0.9, "primary_tool": "search_knowledge",
                    "source": "rule"}

    svc = AgentTurnService(
        retrieval_service=None, answer_generator=None, unified_agent=None,
        intent_classifier=_Classifier(), dst_manager=None, slot_extractor=None)
    ctx = TurnContext(
        retrieval_service=None, answer_generator=None, dst_manager=None,
        slot_extractor=None, question="退货政策是什么",
        session_id="u1__t1", user={"user_id": "u1"}, dst=None,
        intent_info={})
    turn = TurnState(ctx=ctx, messages=[("user", "退货政策是什么")],
                     chat_history=None)

    async def drive():
        return await svc._run_classify_node({"turn": turn})

    result = asyncio.run(drive())
    # 路由结果写入 state["branch"]，条件边只读它
    assert result == {"intent": "consult", "confidence": 0.9, "branch": "consult"}
    # 回填后的 intent_info 是下游提示词与路由的共用输入
    assert ctx.intent_info["intent"] == "consult"
    # system 意图指令注入消息列表头部，供 Agent 兜底节点使用
    assert turn.messages[0][0] == "system"
    assert "知识咨询" in turn.messages[0][1]
    # 原始用户消息保留在意图指令之后
    assert turn.messages[1] == ("user", "退货政策是什么")


def test_classify_node_without_classifier_still_writes_branch():
    """降级路径（未配置分类器）也必须写 branch，否则条件边无输入。"""
    svc = AgentTurnService(
        retrieval_service=None, answer_generator=None, unified_agent=None,
        intent_classifier=None, dst_manager=None, slot_extractor=None)
    ctx = TurnContext(
        retrieval_service=None, answer_generator=None, dst_manager=None,
        slot_extractor=None, question="我要投诉",
        session_id="u1__t1", user={"user_id": "u1"}, dst=None,
        intent_info={"intent": "complaint"})
    turn = TurnState(ctx=ctx, messages=[("user", "我要投诉")], chat_history=None)

    async def drive():
        return await svc._run_classify_node({"turn": turn, "dst_enabled": False})

    assert asyncio.run(drive()) == {
        "intent": "complaint", "confidence": 0.0, "branch": "complaint"}
    # 未配置分类器时不注入 system 指令，消息列表保持原样
    assert turn.messages == [("user", "我要投诉")]


# ===================== 3) _produce：图路由 + 流式产出 + 历史落库 =====================
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
        yield "退货需保持商品完好"


def test_produce_routes_consult_and_finalizes(monkeypatch):
    """consult 经图节点流式产出，结束后历史一问一答完整落库。"""
    monkeypatch.setattr(agent_turn, "DST_ENABLED", False)
    # 审计是旁路：单测不落库（handler 节点也会记一行 handler.consult）
    monkeypatch.setattr(agent_turn.mysql_store, "record_tool_call", lambda **kw: None)
    history = _HistoryStub()
    svc = AgentTurnService(
        retrieval_service=_RetrievalStub(), answer_generator=_GeneratorStub(),
        unified_agent=None, intent_classifier=None, dst_manager=None,
        slot_extractor=None)
    ctx = TurnContext(
        retrieval_service=_RetrievalStub(), answer_generator=_GeneratorStub(),
        dst_manager=None, slot_extractor=None, question="退货政策是什么",
        session_id="u1__t1", user={"user_id": "u1"}, dst=None,
        intent_info={"intent": "consult"})
    turn = TurnState(ctx=ctx, messages=[], chat_history=history)

    async def drive():
        queue = asyncio.Queue()
        await svc._produce(turn, queue)
        return [queue.get_nowait() for _ in range(queue.qsize())]

    pieces = asyncio.run(drive())
    assert pieces == ["退货需保持商品完好"]
    assert [type(m).__name__ for m in history.messages] == ["HumanMessage", "AIMessage"]
    assert history.messages[1].content == "退货需保持商品完好"
