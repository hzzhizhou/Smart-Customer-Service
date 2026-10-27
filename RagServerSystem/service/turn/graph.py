"""LangGraph 编排：一轮 Agent 问答的分支状态机。

图是无状态的（不挂 checkpointer）：跨轮业务状态仍由 DSTManager 持久化到 MySQL，
StateGraph 只回答「本轮走哪条分支」，节点内直接向轮次队列流式写片段。
"""
import re
from typing import Any, Awaitable, Callable, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from shared.constants import PROGRESS_KEYWORDS, is_my_orders_query


BRANCHES = (
    "complaint",
    "ticket_progress",
    "order_query",
    "my_orders",
    "return_refund",
    "consult",
    "agent",
)

# 意图节点：整轮的固定第一步，识别结果驱动后续条件路由
CLASSIFY = "classify"


class TurnGraphState(TypedDict, total=False):
    """图状态：本轮上下文与流式输出句柄（单轮使用，不序列化、不跨轮）。"""
    ctx: Any            # TurnContext：问题/会话/用户/DST/意图
    turn: Any           # TurnState：消息、历史句柄等编排素材
    queue: Any          # 下游转发队列
    final_text: list    # 已产出片段（收尾落库用）
    dst_enabled: bool   # 调用时传入，保证测试可 patch agent_turn.DST_ENABLED
    intent: str         # classify 识别的意图（可观测）
    confidence: float   # 识别置信度（可观测）
    branch: str         # classify 判定的分支，条件边唯一输入


def route_turn(ctx, dst_enabled: bool) -> str:
    """判定本轮分支"""
    question = ctx.question
    intent = (ctx.intent_info or {}).get("intent")
    progress_hit = any(k in question for k in PROGRESS_KEYWORDS)
    if intent == "complaint" and not progress_hit:
        return "complaint"
    if progress_hit:
        return "ticket_progress"
    if re.search(r"SO\d+", question) and intent in ("order", "logistics"):
        return "order_query"
    if is_my_orders_query(question):
        return "my_orders"
    main_intent = (ctx.dst.intent if (ctx.dst is not None and dst_enabled)
                   else (intent or ""))
    if (dst_enabled and ctx.dst is not None and ctx.slot_extractor is not None
            and main_intent in ("return", "refund")):
        return "return_refund"
    if intent == "consult":
        return "consult"
    return "agent"


Node = Callable[[TurnGraphState], Awaitable[dict]]


def route_after_classify(state: TurnGraphState) -> str:
    """条件边：只读 classify 写入的 branch，不重复计算；缺失时兜底 agent。"""
    return state.get("branch") or "agent"


def build_turn_graph(nodes: Dict[str, Node]):
    """组装并编译分支图：classify 意图节点先跑，条件边按它写入的 branch 分派业务分支。

    只编译一次、跨轮复用；节点回调由调用方注入（意图识别/handler 执行/Agent 兜底），
    图本身不持有任何会话数据。路由结果作为 state["branch"] 可见、可断言，
    条件边只做读取，避免「节点算一次、边再算一次」的口径漂移。
    """
    graph = StateGraph(TurnGraphState)
    # 固定入口边：classify 是整轮的第一步（读 DST → 意图识别 → 判定分支），
    # 条件路由只能发生在它之后——路由判定的输入正是 classify 的产出。
    graph.add_node(CLASSIFY, nodes[CLASSIFY])
    graph.add_edge(START, CLASSIFY)
    graph.add_conditional_edges(
        CLASSIFY,
        route_after_classify,
        {name: name for name in BRANCHES},
    )
    for name, node in nodes.items():
        if name == CLASSIFY:
            continue
        graph.add_node(name, node)
        graph.add_edge(name, END)
    return graph.compile()
