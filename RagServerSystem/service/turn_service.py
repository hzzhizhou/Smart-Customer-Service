"""轮次服务层：一次请求 = 一轮问答，本模块承载两条流式入口的编排逻辑。

分层约定（API 层只接客，编排逻辑住在这里）
- API 层（access/app.py 的路由函数）只做四件事：接客、参数校验、
  鉴权、会话 ID 作用域化，然后把「这一轮怎么答」交给本模块；
- 本模块负责：读历史窗口 → 识别意图 → 选确定性分支或 Agent 兜底 → 逐段产出
  → 收尾（落库 + 推进 DST）；
- Handlers（service/handlers/）只负责「这一轮回什么」，收尾动作统一在本模块做。

两条流式入口的差异
- AgentTurnService：有历史窗口、意图注入、DST 跨轮状态、确定性分支调度、ReAct 兜底；
- RagTurnService  ：无状态检索问答，带响应缓存与门控响应头。
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import CHAT_HISTORY_WINDOW, DST_ENABLED
from logs.log_config import log
from utils.circuit_breaker import llm_breaker
from utils.metrics import (
    agent_duration_seconds,
    agent_fallback_total,
    agent_ttft_seconds,
    agent_tool_calls_total,
    agent_tool_duration_seconds,
    branch_total,
    intent_total,
)
from shared.schemas import RAGRequest
from service.cache import ResponseCache
from infrastructure.chat_history_factory import init_chat_history
from shared.constants import (
    MAINLINE_SWITCH_HINTS,
    MAINLINE_SWITCH_SCORE,
    PROGRESS_KEYWORDS,
    could_be_empty_answer,
    is_empty_answer,
    is_my_orders_query,
    strip_react_preamble,
)
from service.handlers import (
    TurnContext,
    handle_complaint,
    handle_consult,
    handle_my_orders,
    handle_order_query,
    handle_return_refund,
    handle_ticket_progress,
    stream_knowledge_answer,
)
from shared.reply_templates import (
    AGENT_NO_ANSWER_FALLBACK,
    KNOWLEDGE_NO_ANSWER,
    SERVICE_CONTACT,
)

# 后台生成任务集合：持有引用防止任务被 GC 回收（生成与 HTTP 连接解耦后需独立跑完）
_BG_TASKS: set = set()


def _summarize_value(value: Any, limit: int = 200) -> str:
    """把工具入参/输出压成单行摘要（观测用，防日志刷屏：检索文档全文不进日志）。

    字符串原样折叠空白；dict/list 走 JSON；不可序列化对象经 default=str 兜底；
    超长截断并加省略号标记。
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


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


@dataclass
class TurnState:
    """一轮 Agent 问答的完整上下文（服务内部使用）。

    handlers 只看到 ctx（见 service/handlers/context.py），messages 与 chat_history
    属于服务自身的编排素材，不进处理器契约。
    """
    ctx: TurnContext             # 处理器入参：问题/会话/用户/DST/意图都在里面
    messages: list               # 注入意图指令后的消息列表（泛化 ReAct 兜底路径用）
    chat_history: Any            # 会话历史读写句柄


class AgentTurnService:
    """Agent 一轮问答的编排：读历史 → 识别意图 → 选分支 → 产出 → 收尾。"""

    def __init__(self, retrieval_service, answer_generator, unified_agent,
                 intent_classifier, dst_manager, slot_extractor=None):
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.unified_agent = unified_agent
        self.intent_classifier = intent_classifier
        self.dst_manager = dst_manager
        self.slot_extractor = slot_extractor

    # ---------- 对外入口 ----------
    async def start_turn(self, question: str, session_id: str,
                         user: Dict[str, Any]) -> asyncio.Queue:
        """启动一轮问答：立即返回转发队列，答复在独立任务里生成并直接落库。

        生成与 HTTP 连接解耦：客户端中途断开时，Starlette 只会关掉下游生成器，
        上游生成不受影响，从而避免「半截答复被当成完整答复写进历史」。
        """
        turn_started_at = time.perf_counter()   # TTFT 起算点（必须在 _prepare 之前）
        turn = await self._prepare(question, session_id, user)
        queue: asyncio.Queue = asyncio.Queue()
        # TTFT 起算于 start_turn 入口（含 _prepare 的意图识别/历史加载/DST 加载）
        stream_q = _StreamQueue(queue, _TTFTRecorder(turn_started_at, "agent"))
        _spawn_producer(self._produce(turn, stream_q), queue)
        return queue

    # ---------- 准备阶段 ----------
    async def _prepare(self, question: str, session_id: str,
                       user: Dict[str, Any]) -> TurnState:
        """取历史窗口 → 加载 DST → 注入业务意图，组装成一轮的上下文。"""
        chat_history = init_chat_history(session_id)
        history_msgs = await load_chat_history(chat_history)
        messages = self._build_messages(history_msgs, question)
        # DST：加载跨轮会话状态，注入已收集槽位/待补项
        dst = self.dst_manager.load(session_id) if DST_ENABLED else None
        messages, intent_info = await self._enrich_messages_with_intent(
            messages, question, dst=dst, user_id=user["user_id"])
        return TurnState(
            ctx=TurnContext(
                retrieval_service=self.retrieval_service,
                answer_generator=self.answer_generator,
                dst_manager=self.dst_manager,
                slot_extractor=self.slot_extractor,
                question=question,
                session_id=session_id,
                user=user,
                dst=dst,
                intent_info=intent_info,
            ),
            messages=messages,
            chat_history=chat_history,
        )

    @staticmethod
    def _build_messages(history_msgs: list, question: str) -> list:
        """取最近 CHAT_HISTORY_WINDOW 条历史转为 (role, content)，并接上本轮提问。"""
        messages = []
        for msg in history_msgs[-CHAT_HISTORY_WINDOW:]:
            if isinstance(msg, HumanMessage):
                messages.append(("user", msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(("assistant", msg.content))
        messages.append(("user", question))
        return messages

    async def _enrich_messages_with_intent(self, messages: list, question: str,dst: Optional[Any] = None,user_id: str = "") -> tuple:
        """
        在 Agent 触发前识别细粒度业务意图，作为首条 system 指令注入消息列表，
        让 Agent 在一开始就知道优先使用哪个业务工具。
        若传入 dst（DST 会话状态），则把跨轮已收集槽位/待补项合并进同一 system 段。
        若传入 user_id（登录账号），则告知 Agent 工单归属，使其建单/查单都挂在账号上。
        返回 (增强后的消息列表, 本轮意图识别结果)。
        """
        # 业务主线意图（有明确槽位诉求），后续追问直接复用，避免 LLM 兜底拖慢响应
        BUSINESS_MAINLINE = {"return", "refund", "logistics", "complaint"}
        force_intent = None
        if dst is not None and DST_ENABLED and dst.intent in BUSINESS_MAINLINE and dst.stage in ("COLLECTING", "CONFIRMING", "EXECUTING"):
            # 复用主线意图前，先免费跑一次规则打分，确认用户没有中途换话题：
            # 实测「退货进行中突然问『帮我查一下订单 SOxxx』」会被硬套成退货，答非所问。
            scores = self.intent_classifier.rule_scores(question)
            mainline_score = max([scores.get(k, 0.0) for k in BUSINESS_MAINLINE] + [0.0])
            others = {k: v for k, v in scores.items() if k not in BUSINESS_MAINLINE}
            switch_intent, switch_score = max(others.items(), key=lambda kv: kv[1]) if others else ("", 0.0)
            can_switch = switch_score >= MAINLINE_SWITCH_SCORE and switch_score > mainline_score
            # 例外：纯槽位回答不算换话题。用户在补主线正等着的信息（如只回「订单号 SO20241120005」
            # 或「质量问题」）时，把它当成 order 意图让位会让 dst.intent 变成 order、槽位丢失，
            # 主流程再也回不来 —— 实测接下来一句「商品有质量问题」被判成投诉直接建单，
            # 且 dst.intent 被永久改成 complaint，此后每问一次进度就重复建一张新单。
            # 只有带明确查询动作的问法（「帮我查一下订单 SOxxx」「物流到哪了」）才允许让位。
            if can_switch and switch_intent in ("order", "logistics") \
                    and not any(h in question for h in MAINLINE_SWITCH_HINTS):
                can_switch = False
                log.info(f"DST 主线不让位 | 主线={dst.intent} vs 本轮={switch_intent}"
                         f"({switch_score})：本轮为槽位补充，非查询诉求 | 问题={question[:40]}")
            if can_switch:
                # 让位时必须同时把会话意图切过去：下游分支判主线用的是 main_intent = dst.intent，
                # 只拦 force_intent 不够（实测仍会回退货引导）。
                stale = dst.intent
                dst.intent = switch_intent
                log.info(f"DST 主线意图让位 | 主线={stale} → 本轮命中={switch_intent}"
                         f"({switch_score} > 主线{mainline_score}) | 问题={question[:40]}")
            else:
                force_intent = dst.intent
        intent_info = await self.intent_classifier.classify(question, force_intent=force_intent)
        prompt = (
            f"[当前用户业务意图：{intent_info['intent_cn']}({intent_info['intent']}，"
            f"置信度{intent_info['confidence']})] "
            f"请优先使用工具 `{intent_info['primary_tool']}`。"
            f"（来源：{intent_info['source']}）"
        )
        if user_id:
            # 工单/订单归属登录账号（而非会话）：换会话后用户查「我的工单」仍能查到，
            # 同时把查询限定在本账号名下，避免拿别人的订单号查到他人订单与工单
            prompt += (f" [当前用户账号标识：{user_id}]"
                       f" 创建工单时 create_ticket 的 user_id 必须填这个账号标识。"
                       f" 查询订单时 query_order 的 owner_id、列出名下订单时"
                       f" list_my_orders 的 owner_id、查询工单时 query_ticket 的 owner_id"
                       f" 都必须填这个账号标识（限定只查本账号的订单/工单）；"
                       f" query_ticket 的 user_id 填订单号或该账号标识。")
        if intent_info["intent"] == "chat":
            prompt += " 这是闲聊，请直接礼貌回应，不要调用任何工具。"
        elif intent_info["intent"] == "complaint":
            prompt += " 这是投诉场景，不要自行承诺赔偿，优先创建工单转人工。"
        log.info(f"意图识别 | {intent_info} | 问题={question[:40]}")
        # 意图分布指标：与上面这行日志同点，回答「用户都在问什么」。
        # intent 取值来自分类器固定集合，标签基数有界。
        intent_total.labels(intent_info["intent"]).inc()
        # 若启用 DST 且存在跨轮状态：注入会话记忆 + 意图融合（保持业务主线）
        if dst is not None and DST_ENABLED:
            new_intent = intent_info["intent"]
            # 业务主线意图（有明确槽位诉求），遇到泛化意图(order/consult)时不覆盖主线
            BUSINESS_MAINLINE = {"return", "refund", "logistics", "complaint"}
            if dst.intent not in BUSINESS_MAINLINE or new_intent not in ("order", "consult", "chat"):
                dst.intent = new_intent
            dst_ctx = self.dst_manager.build_system_context(dst)
            if dst_ctx:
                prompt += dst_ctx
        # system 指令插到消息列表最前；同时把识别到的意图返回给上层（用于确定性动作决策）
        return [("system", prompt)] + messages, intent_info

    # ---------- 确定性分支调度 ----------
    @staticmethod
    def _pick_handler(ctx: TurnContext):
        """挑选本轮要走的确定性分支，返回 (处理器, 该分支应计入的工具调用次数)。

        判定顺序即业务优先级（与拆分前完全一致）：投诉 → 工单进度 → 带单号订单查询
        → 我名下订单 → 退货/退款 → 知识咨询；都不命中则返回 (None, 0)，交给 Agent 泛化处理。
        """
        question = ctx.question
        intent = (ctx.intent_info or {}).get("intent")
        # 本轮若在问「工单/处理进度」就不判投诉建单：用户提交投诉后 dst 会一直复用
        # complaint 主线意图，先判投诉就会「每问一次进度就多一张工单」（实测连问两次生成两张）。
        progress_hit = any(k in question for k in PROGRESS_KEYWORDS)
        if intent == "complaint" and not progress_hit:
            return handle_complaint, 1     # 建单相当于调了一次工具，用于推进 DST 阶段
        if progress_hit:
            return handle_ticket_progress, 0
        if re.search(r"SO\d+", question) and intent in ("order", "logistics"):
            return handle_order_query, 0
        if is_my_orders_query(question):
            return handle_my_orders, 0
        # 退货/退款以「融合后的主线意图」为准（跨轮保持 return/refund 不被 qwen 冲掉）
        main_intent = (ctx.dst.intent if (ctx.dst is not None and DST_ENABLED)
                       else (intent or ""))
        if (DST_ENABLED and ctx.dst is not None and ctx.slot_extractor is not None
                and main_intent in ("return", "refund")):
            return handle_return_refund, 0
        if intent == "consult":
            return handle_consult, 0
        return None, 0

    # ---------- 产出阶段 ----------
    async def _produce(self, turn: TurnState, queue: asyncio.Queue):
        """（独立任务）生成完整答复：逐段写入队列供下游转发，结束时直接落库。"""
        final_text: List[str] = []
        # 处理器只负责「这一轮回什么」，写历史与推进 DST 统一在这里收尾
        handler, dst_tool_count = self._pick_handler(turn.ctx)
        # 分支分布指标：确定性分支取其处理器名（去掉 handle_ 前缀），
        # 都不命中记为 agent（泛化 ReAct）。回答「请求实际走了哪条链路」。
        branch_total.labels(
            handler.__name__.removeprefix("handle_") if handler is not None else "agent"
        ).inc()
        if handler is not None:
            try:
                async for piece in handler(turn.ctx):
                    final_text.append(piece)
                    await queue.put(piece)
            finally:
                answer = "".join(final_text)
                if answer:
                    await finalize_turn(turn.chat_history, turn.ctx.question, answer)
                    if dst_tool_count and DST_ENABLED:
                        self._schedule_dst_update(turn.ctx.session_id, turn.ctx.question,
                                                  answer, dst_tool_count)
            return

        # ========== 泛化 ReAct 兜底：以上确定性分支都不命中时交给 Agent ==========
        await self._produce_agent(turn, queue, final_text)

    async def _produce_agent(self, turn: TurnState, queue: asyncio.Queue,
                             final_text: List[str]):
        """泛化 ReAct 兜底：只把面向用户的正文透出，推理与工具调用一律吞掉。

        astream_events 同时会吐出工具调用的结构化消息与 Thought/Action 推理文本，
        这里以「Final Answer:」标记为界，标记之前的内容全部丢弃。
        """
        question = turn.ctx.question
        session_id = turn.ctx.session_id
        raw = []           # 全量内容，用于「无 Final Answer 标记」时的回退
        marker = "Final Answer:"
        mlen = len(marker)
        buffer = ""        # 标记检测缓冲
        seen_final = False
        pending = ""       # Final Answer 之后的正文缓冲（空答复判定用，见 _produce_rag）
        passthrough = False
        tool_calls_this_round = {"n": 0}   # 统计本轮工具调用（推进 EXECUTING 阶段）
        tool_started: Dict[str, float] = {}  # run_id → 打点时刻（工具耗时观测）
        reasoning_chars = 0                 # Final Answer 前的 Thought/Action 推理字数（观测用）
        # 熔断：LLM 连续失败时不再进入 ReAct 循环（否则「多轮工具调用 + 超时」会雪崩），
        # 直接给确定性兜底话术。Agent 与 RAG 生成同用一个 DashScope LLM 后端，故共用 llm_breaker。
        if not llm_breaker.allow_request():
            log.warning(f"LLM 熔断中，Agent 跳过 ReAct 直接兜底 | 问题={question[:40]}")
            agent_fallback_total.labels("breaker").inc()
            final_text.append(AGENT_NO_ANSWER_FALLBACK)
            await queue.put(AGENT_NO_ANSWER_FALLBACK)
            return
        agent_t0 = time.perf_counter()
        try:
            async for event in self.unified_agent.astream_events(
                {"messages": turn.messages},
                config={"recursion_limit": 10},
                version="v1",
            ):
                # ---- 工具调用观测：开始 / 完成 / 失败（日志自动带 request_id）----
                if event["event"] == "on_tool_start":
                    tool_calls_this_round["n"] += 1
                    tool_started[event.get("run_id", "")] = time.perf_counter()
                    log.info(f"Agent 工具调用开始 | 第{tool_calls_this_round['n']}次 | "
                             f"tool={event.get('name')} | 入参={_summarize_value((event.get('data') or {}).get('input'))}")
                    continue
                if event["event"] == "on_tool_end":
                    tool = event.get("name") or "unknown"
                    start = tool_started.pop(event.get("run_id", ""), None)
                    if start is not None:
                        elapsed = time.perf_counter() - start
                        agent_tool_calls_total.labels(tool, "success").inc()
                        agent_tool_duration_seconds.labels(tool).observe(elapsed)
                        log.info(f"Agent 工具调用完成 | tool={tool} | 耗时={elapsed:.2f}s | "
                                 f"输出={_summarize_value((event.get('data') or {}).get('output'))}")
                    continue
                if event["event"] == "on_tool_error":
                    tool = event.get("name") or "unknown"
                    agent_tool_calls_total.labels(tool, "error").inc()
                    tool_started.pop(event.get("run_id", ""), None)
                    log.warning(f"Agent 工具调用失败 | tool={tool} | "
                                f"err={_summarize_value((event.get('data') or {}).get('error'))}")
                    continue
                if event["event"] != "on_chat_model_stream":
                    continue
                chunk = event["data"]["chunk"]
                msg = getattr(chunk, "message", None) or chunk
                # 工具调用阶段（含 Thought/Action 推理文本）跳过
                if getattr(msg, "tool_call_chunks", None):
                    continue
                content = msg.content if isinstance(msg.content, str) else ""
                if not content:
                    continue
                raw.append(content)

                if seen_final:
                    # 与 RAG 链路同一处理：正文可能是「无相关信息」这类占位答复，
                    # 要等整段收完才能判定，故先把「可能是空答复」的前缀押后，
                    # 一旦确认不是（偏离所有候选写法）立即 flush 转直通。
                    if passthrough:
                        final_text.append(content)
                        await queue.put(content)
                        continue
                    pending += content
                    if could_be_empty_answer(pending):
                        continue
                    passthrough = True
                    final_text.append(pending)
                    await queue.put(pending)
                    pending = ""
                    continue

                # 尚未遇到 Final Answer 标记：把推理文本吞进缓冲
                buffer += content
                reasoning_chars += len(content)
                idx = buffer.find(marker)
                if idx == -1:
                    # 丢弃推理内容，仅保留可能横跨分块的标记尾部
                    if len(buffer) > mlen:
                        buffer = buffer[-(mlen - 1):]
                    continue
                seen_final = True
                # 推理链留痕：Thought 长度 + 工具轮次（正文推理文本本身仍不落日志，防刷屏）
                log.info(f"Agent 结束推理进入作答 | Thought≈{reasoning_chars}字 | 工具调用={tool_calls_this_round['n']}次")
                tail = buffer[idx + mlen:]
                buffer = ""
                if tail:
                    pending += tail
                    if not could_be_empty_answer(pending):
                        passthrough = True
                        final_text.append(pending)
                        await queue.put(pending)
                        pending = ""

            # LLM 流式产出正常结束：记为一次成功，CLOSED 态复位失败计数
            llm_breaker.record_success()
            if seen_final:
                # Final Answer 之后整段就是空答复（模型无据可依时的占位答复）：
                # 对用户等于没答，换确定性兜底话术，不能把「无相关信息」原样抛给用户。
                if pending:
                    if is_empty_answer(pending):
                        log.info(f"Agent 空答复改走兜底 | 问题={question[:40]}")
                        agent_fallback_total.labels("empty_answer").inc()
                        pending = KNOWLEDGE_NO_ANSWER.format(contact=SERVICE_CONTACT)
                    final_text.append(pending)
                    await queue.put(pending)
            else:
                joined = "".join(raw)
                # raw 为空：模型连内容流都没有（空回复/只发工具调用后中断），
                # 此前这种情况直接返回空串，用户拿到一片空白；统一走知识兜底。
                answer_raw = strip_react_preamble(joined) if joined else ""
                if answer_raw and not is_empty_answer(answer_raw):
                    final_text.append(answer_raw)
                    await queue.put(answer_raw)
                else:
                    # 模型只输出推理、没给正文（qwen 偶发），或正文是「无」这类空答复：
                    # 一律转确定性知识兜底，走 RAG 检索作答，避免用户拿到无意义的答复
                    log.warning(f"Agent 未产出有效正文，转知识兜底 | 原始内容={joined[:300]!r}")
                    agent_fallback_total.labels("no_content").inc()
                    pieces = []
                    async for piece in stream_knowledge_answer(
                            self.retrieval_service, self.answer_generator,
                            question, session_id):
                        pieces.append(piece)
                        await queue.put(piece)
                    final_text.append("".join(pieces) or AGENT_NO_ANSWER_FALLBACK)
        except Exception as e:
            llm_breaker.record_failure()
            log.error(f"Agent 流式失败: {e}", exc_info=True)
            # 不重推 final_text[-1]：该段此前已流给客户端，重推会让末尾内容重复一遍；
            # 若异常发生在一段正文都没产出时，补一句兜底话术，避免用户拿到空响应
            if not final_text:
                agent_fallback_total.labels("exception").inc()
                final_text.append(AGENT_NO_ANSWER_FALLBACK)
                await queue.put(AGENT_NO_ANSWER_FALLBACK)
        finally:
            # Agent 整体耗时：覆盖本轮 ReAct 全流程（含多轮工具调用）。
            # 缺此观测则「Agent 整体耗时分位」面板恒为空（指标有声明无埋点）。
            agent_duration_seconds.observe(time.perf_counter() - agent_t0)
            answer = "".join(final_text)
            if answer:
                await finalize_turn(turn.chat_history, question, answer)
                # DST：更新会话状态（槽位提取→合并→推进阶段→持久化），不影响已返回的流
                if DST_ENABLED:
                    self._schedule_dst_update(session_id, question, answer,
                                              tool_calls_this_round["n"])

    # ---------- DST 后台更新 ----------
    def _schedule_dst_update(self, session_id: str, question: str,
                             answer: str, tool_count: int):
        """调度后台 DST 状态更新；优先挂到当前事件循环，失败则退化为同步后台任务。"""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._update_dst_state_async(session_id, question, answer, tool_count))
        except RuntimeError:
            import asyncio as _aio
            try:
                _aio.run(self._update_dst_state_async(
                    session_id, question, answer, tool_count))
            except Exception as e:
                log.error(f"DST 同步更新失败（不影响已返回响应）: {e}", exc_info=True)

    async def _update_dst_state_async(self, session_id: str, question: str,
                                      answer: str, tool_count: int):
        """（异步）LLM 提取槽位 → 合并 → 推进阶段 → save。全程 try/except，绝不影响主响应。"""
        if not DST_ENABLED:
            return
        try:
            dst = self.dst_manager.load(session_id)
            new_slots = {}
            if self.slot_extractor is not None:
                new_slots = await self.slot_extractor.extract(question, dst.intent or "", dst.slots)
            dst = self.dst_manager.merge_new_slots(dst, new_slots)
            dst.tool_call_count += tool_count
            dst.answer = answer or dst.answer
            stage = self.dst_manager.advance_stage(dst, tool_executed=tool_count > 0)
            self.dst_manager.save(dst)
            log.info(f"DST 更新 | session={session_id} | intent={dst.intent} | "
                     f"slots={dst.slots} | stage={stage}")
        except Exception as e:
            log.error(f"DST 更新失败（不影响已返回响应）: {e}", exc_info=True)


def _gate_headers(retriever_type: str, gate_info: dict) -> dict:
    """置信度门控结果转响应头。

    冷路径（新生成）与热路径（命中缓存）必须回填同一组头，否则客户端无法区分
    「命中缓存」与「检索降级」——所以两条分支共用这一个函数，不做两份。
    """
    return {
        "X-Retriever-Type": retriever_type,
        "X-Gate-Level": gate_info["level"],
        "X-Gate-Should-Escalate": str(gate_info["should_escalate"]),
        "X-Gate-Top-Score": str(gate_info["top_score"]),
    }


class RagTurnService:
    """RAG 一轮问答：缓存命中直接秒回；未命中检索 + 流式生成 + 独立任务落库。

    与 AgentTurnService 同一套生产者骨架：LLM 生成跑在独立任务里，客户端断连
    只影响转发，该轮历史仍完整落库——此前历史挂在 BackgroundTasks 上，客户端
    一断开 Starlette 不再执行背景任务，该轮历史直接丢失。

    返回 (文本流, 响应头) 而不是 HTTP 响应对象：包成 StreamingResponse 是 API 层的事。
    """

    def __init__(self, retrieval_service, answer_generator,
                 cache: Optional[ResponseCache] = None):
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.cache = cache if cache is not None else ResponseCache()

    async def stream(self, request: RAGRequest, session_id: str) -> tuple:
        """回答一个问题，返回 (异步文本流, 门控响应头)。"""
        stream_started_at = time.perf_counter()   # TTFT 起算点（含检索耗时）
        # 缓存：无会话场景下同一问题命中直接秒回（避免再次走 6s 的 LLM）
        cache_key = None
        if request.session_id is None:
            cache_key = self.cache.make_key(request.question, request.route_mode)
            cached = self.cache.get(cache_key)
            if cached:
                log.info(f"流式 RAG 命中缓存 | {request.question[:30]}")

                async def cached_stream():
                    # 缓存里存的是 dict（见 ResponseCache.set），必须按键取值：
                    # 此前写成 cached.answer 会在生成器里抛 AttributeError，
                    # 响应已返回 200，客户端只会收到空 body（表现为"某问题恒秒回空"）
                    # 缓存命中单独打一个 path 标签：与真实生成路径分开，避免把毫秒级
                    # 命中混进生成延迟直方图、把 P95 拉成失真值
                    agent_ttft_seconds.labels("rag_cached").observe(
                        max(time.perf_counter() - stream_started_at, 0.0))
                    yield cached["answer"]

                # 命中缓存也要回填门控头，否则客户端无法区分"命中"与"降级"
                return cached_stream(), _gate_headers(cached["retriever_type"],
                                                      cached["gate_info"])

        try:
            docs, retriever_type, gate_info = await self.retrieval_service.retrieve(
                question=request.question,
                route_mode=request.route_mode,
                use_context=request.use_context,
                use_hyde=request.use_hyde,
                use_multi=request.use_multi,
                session_id=session_id
            )
            headers = _gate_headers(retriever_type, gate_info)
            chat_history = init_chat_history(session_id)
            queue: asyncio.Queue = asyncio.Queue()
            stream_q = _StreamQueue(queue, _TTFTRecorder(stream_started_at, "rag"))
            _spawn_producer(
                self._produce_rag(request, session_id, docs, chat_history,
                                 cache_key, retriever_type, gate_info, stream_q),
                queue,
            )

            async def relay():
                """（HTTP 下游）只做转发：从队列取块推给客户端，收到哨兵即结束。"""
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item

            return relay(), headers
        except Exception as e:
            log.error(f"流式 RAG 处理失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="服务内部错误")

    async def _produce_rag(self, request: RAGRequest, session_id: str, docs,
                           chat_history, cache_key, retriever_type: str,
                           gate_info: dict, queue: asyncio.Queue):
        """（独立任务）RAG 生成：流式产出 → 空答复兜底 → 缓存 → 落库。

        成功跑完才写缓存：半截/异常答复入缓存会被长期秒回；历史在 finally 里
        落库，无论中途是否异常，已产出的正文都完整入库。
        """
        full_answer: List[str] = []
        pending = ""
        passthrough = False
        try:
            async for chunk in self.answer_generator.stream_generate(
                    request.question, docs, session_id):
                if passthrough:
                    full_answer.append(chunk)
                    await queue.put(chunk)
                    continue
                # 模型偶发把「无相关信息」当正文吐出来（无据可依时的占位答复），
                # 对用户等于没答。但这要等整段收完才能判定，所以先把「可能是空答复」
                # 的前缀押后；一旦确认不是（前缀偏离了所有候选写法）立即 flush 转直通，
                # 只比正常路径多等一两个 chunk，不动正常回答的首字体验。
                pending += chunk
                if could_be_empty_answer(pending):
                    continue
                passthrough = True
                full_answer.append(pending)
                await queue.put(pending)
                pending = ""
            if pending:
                # 走完整个流仍未直通：要么整段就是空答复 → 换确定性兜底话术，
                # 要么是「有」这类正常短答 → 原样透出
                if is_empty_answer(pending):
                    log.info(f"RAG 空答复改走兜底 | 问题={request.question[:40]}")
                    agent_fallback_total.labels("empty_answer").inc()
                    pending = KNOWLEDGE_NO_ANSWER.format(contact=SERVICE_CONTACT)
                full_answer.append(pending)
                await queue.put(pending)
            # 成功完成才写缓存：空答案（生成被中断/模型异常）绝不入缓存，
            # 否则这个问题会被长期秒回空响应
            answer_text = "".join(full_answer)
            if cache_key and answer_text.strip():
                self.cache.set(cache_key, {
                    "answer": answer_text,
                    "retriever_type": retriever_type,
                    "response_time": 0.0,
                    "session_id": session_id,
                    "gate_info": gate_info,
                })
        except Exception as e:
            log.error(f"RAG 流式生成失败: {e}", exc_info=True)
            # 一段正文都没流出就失败：补兜底话术，避免用户拿到空响应
            if not full_answer:
                agent_fallback_total.labels("exception").inc()
                full_answer.append(AGENT_NO_ANSWER_FALLBACK)
                await queue.put(AGENT_NO_ANSWER_FALLBACK)
        finally:
            answer = "".join(full_answer)
            if answer:
                await finalize_turn(chat_history, request.question, answer)