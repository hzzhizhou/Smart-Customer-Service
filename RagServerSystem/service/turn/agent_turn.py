"""Agent 一轮问答的编排：读历史 → 图内意图识别与路由 → 分支执行 → 收尾。
classify 意图节点 → route_turn 条件边 → 业务分支 / Agent 兜底；
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from agent.tools.tool_context import reset_current_user, set_current_user
from config.settings import CHAT_HISTORY_WINDOW, DST_ENABLED
from infrastructure import mysql_store
from infrastructure.chat_history_factory import init_chat_history
from logs.log_config import log
from shared.constants import (
    MAINLINE_SWITCH_HINTS,
    MAINLINE_SWITCH_SCORE,
    could_be_empty_answer,
    is_empty_answer,
    split_sub_questions,
    looks_like_question,
)
from shared.reply_templates import (
    AGENT_NO_ANSWER_FALLBACK,
    KNOWLEDGE_NO_ANSWER,
    NO_ANSWER_SIGNALS,
    SERVICE_CONTACT,
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
from utils.circuit_breaker import llm_breaker
from utils.metrics import (
    agent_duration_seconds,
    agent_fallback_total,
    agent_tool_calls_total,
    agent_tool_duration_seconds,
    branch_total,
    intent_total,
)
from utils.request_context import get_request_id, get_trace_id, span
from service.turn.context_resolver import merge_into_slots, resolve_question

from .graph import build_turn_graph, route_turn
from .state import TurnState, _redact_args, _result_digest, _summarize_value
from .streaming import (
    _StreamQueue,
    _TTFTRecorder,
    _spawn_producer,
    finalize_turn,
    load_chat_history,
)


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

    # ---------- 多轮补全 ----------
    @staticmethod
    def _resolve_multi_turn(question: str, dst, history_msgs) -> str:
        """补全跨轮实体并把继承到的槽位合并进 DST；返回原问题（不改写）。

        失败不影响主链路：补全只是「让下游少反问一次」，拿不到就照旧走反问。
        """
        try:
            texts = []
            for m in (history_msgs or []):
                content = getattr(m, "content", m)
                if isinstance(content, str) and content.strip():
                    texts.append(content)
            res = resolve_question(question, dst=dst, history_texts=texts)
            if not res.changed:
                return question
            log.info("多轮补全", extra={"event": "turn.context.resolve", **res.as_log_fields()})
            slots = getattr(dst, "slots", None)
            if isinstance(slots, dict):
                dst.slots = merge_into_slots(slots, res.inherited)
            return question
        except Exception as exc:                     # noqa: BLE001 补全是增强，不能拖垮主链路
            log.warning("多轮补全失败，按原问题继续",
                        extra={"event": "turn.context.resolve.error",
                               "err": f"{type(exc).__name__}: {exc}"})
            return question

    # ---------- 准备阶段 ----------
    async def _prepare(self, question: str, session_id: str,
                       user: Dict[str, Any]) -> TurnState:
        """取历史窗口 → 加载 DST，组装成一轮的上下文。

        意图识别已移入图 classify 节点（见 _run_classify_node）：路由判定的输入
        就是 classify 的产出，时序上必须先识别后路由。
        """
        chat_history = init_chat_history(session_id)
        history_msgs = await load_chat_history(chat_history)
        messages = self._build_messages(history_msgs, question)
        # DST：加载跨轮会话状态，注入已收集槽位/待补项
        dst = await self.dst_manager.load(session_id) if DST_ENABLED else None

        # 多轮补全：把「那能改地址吗」这类省略句依赖的实体从会话状态/最近历史里补出来。
        # 只补槽位、不改写原句（下游确定性分支用正则解析原句，加前缀会改变解析结果）。
        question = self._resolve_multi_turn(question, dst, history_msgs)

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
                # 由图 classify 节点回填（未配置意图分类器时保留预置值）
                intent_info={},
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
        with span("intent.classify", kind="internal"):
            intent_info = await self.intent_classifier.classify(
                question, force_intent=force_intent)
        prompt = (
            f"[当前用户业务意图：{intent_info['intent_cn']}({intent_info['intent']}，"
            f"置信度{intent_info['confidence']})] "
            f"请优先使用工具 `{intent_info['primary_tool']}`。"
            f"（来源：{intent_info['source']}）"
        )
        if user_id:
            # 身份不再写进提示让模型填参数（PROJECT_AUDIT P0-3）：账号由服务端注入工具，
            # 参数里根本没有 owner_id/user_id，模型填不了、注入文本也无处可填。
            # 这里只告知「已登录」这一事实，让模型知道查单/建单是可行的。
            prompt += (" [当前用户已登录：订单与工单的归属账号由服务端自动注入，"
                       "你不需要、也不要在工具参数里填写任何账号标识；"
                       "用户给出的订单号请原样传给工具。]")
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

    # ---------- LangGraph 编排 ----------
    def _build_turn_graph(self):
        """把意图节点 + 六条确定性分支 + Agent 兜底注册成图节点；节点名即分支指标/审计标签。"""

        def handler_node(handler, dst_tool_count: int):
            branch = handler.__name__.removeprefix("handle_")

            async def node(state):
                return await self._run_handler_node(
                    state, handler, dst_tool_count, branch)
            return node

        return build_turn_graph({
            # 意图节点：整轮第一步，识别结果驱动条件路由
            "classify": self._run_classify_node,
            # 建单相当于调了一次工具，用于推进 DST 阶段
            "complaint": handler_node(handle_complaint, 1),
            "ticket_progress": handler_node(handle_ticket_progress, 0),
            "order_query": handler_node(handle_order_query, 0),
            "my_orders": handler_node(handle_my_orders, 0),
            "return_refund": handler_node(handle_return_refund, 0),
            "consult": handler_node(handle_consult, 0),
            "agent": self._run_agent_node,
        })

    async def _run_classify_node(self, state) -> dict:
        """意图节点：DST 主线融合 → 规则/LLM 两级级联识别 → 判定分支。

        识别结果回填 ctx.intent_info（下游提示词与路由共用），system 意图指令
        同时注入 turn.messages；路由结果写入 state["branch"]，条件边只读不重算。
        意图分类器未配置时（测试降级路径）保留预置意图，但 branch 仍必须写入。
        """
        turn: TurnState = state["turn"]
        if self.intent_classifier is not None:
            messages, intent_info = await self._enrich_messages_with_intent(
                turn.messages, turn.ctx.question, dst=turn.ctx.dst,
                user_id=turn.ctx.user.get("user_id", ""))
            turn.messages = messages
            turn.ctx.intent_info = intent_info
        else:
            intent_info = turn.ctx.intent_info or {}
        branch = route_turn(turn.ctx, state.get("dst_enabled", DST_ENABLED))
        multi = self._analyze_multi_intent(turn.ctx.question)
        return {
            "intent": intent_info.get("intent", ""),
            "confidence": intent_info.get("confidence", 0.0),
            "branch": branch,
            **multi,
        }

    # 只读意图：作答不改变任何业务状态，可以在动作分支之前先答
    READ_INTENTS = frozenset({"consult", "order_query", "logistics"})
    # 动作意图：有副作用（建单/退款/改状态），不能与别的动作并行
    ACTION_INTENTS = frozenset({"return", "refund", "complaint", "ticket"})

    def _analyze_multi_intent(self, question: str) -> dict:
        """切分子句并判定「多意图」，结果写进 state 供路由与分支复用。

        为什么放在 classify 而不是检索层：动作分支（退货/投诉）根本不走检索，
        把判定放在检索层等于对这些分支失效——实测「能开专票吗，另外我要退货」
        只识别到 return，发票那半句被整段丢掉。

        判定「多意图」用的是意图集合去重后的大小，不是子句数量：
        「保修多久、范围包括屏幕吗」两个子句但同属保修，不该触发多意图处理。

        只读子句只在「同时存在动作意图」时才回传：全是只读的句子走 consult 分支，
        检索层本来就按子问拆开检索，不需要在这里重复处理。
        """
        empty = {"sub_questions": [], "intent_set": [], "read_questions": [],
                 "is_multi_intent": False}
        subs = split_sub_questions(question or "")
        if len(subs) <= 1 or self.intent_classifier is None:
            return empty

        labels = []
        for sub in subs:
            try:
                scores = self.intent_classifier.rule_scores(sub)
            except Exception:
                scores = {}
            intent = max(scores, key=scores.get) if scores and max(scores.values()) > 0 else ""
            labels.append(intent)

        intent_set = {i for i in labels if i}
        if len(intent_set) <= 1:
            return {**empty, "sub_questions": subs, "intent_set": sorted(intent_set)}

        has_action = any(i in self.ACTION_INTENTS for i in labels)
        read_questions = [s for s, i in zip(subs, labels) if i in self.READ_INTENTS]
        return {
            "sub_questions": subs,
            "intent_set": sorted(intent_set),
            "read_questions": read_questions if has_action else [],
            "is_multi_intent": True,
        }

    def _get_turn_graph(self):
        """懒加载并缓存编译后的图（object.__new__ 构造的测试桩同样可用）。"""
        graph = getattr(self, "_turn_graph", None)
        if graph is None:
            graph = self._build_turn_graph()
            self._turn_graph = graph
        return graph

    # 需要用户补槽位的动作分支：进入前先答「只读」子问
    ACTION_BRANCHES = frozenset({"return_refund", "complaint"})

    async def _answer_read_questions(self, state, questions) -> None:
        """复用 consult 分支回答只读子问，流式写进同一个队列（用户先看到咨询答案）。

        只改 ctx.question 临时换成子问，其余上下文（会话/用户/DST）不变；
        finally 里必须还原，否则后续动作分支会用子问去做槽位提取。
        """
        turn = state["turn"]
        queue: asyncio.Queue = state["queue"]
        final_text: List[str] = state["final_text"]
        ctx = turn.ctx
        origin = ctx.question
        try:
            ctx.question = "，".join(questions)
            # 先缓冲再输出：补答若命中「整句拒答」话术，用户会看到一句没头没脑的
            # "这个问题我暂时没有查到可靠的资料"，再接上动作引导，观感很糟。
            # 这里换成针对该子问的短说明——它只是补充信息，不该表现得像整轮失败。
            pieces = [piece async for piece in handle_consult(ctx)]
            answer = "".join(pieces)
            # 兜底话术并不总是与常量完全相等（生成层会再拼客服联系方式），
            # 所以按特征串判断。信号集中定义在 reply_templates.NO_ANSWER_SIGNALS，
            # 避免这里另写一份字符串、改文案时漏改。
            if answer.strip() == AGENT_NO_ANSWER_FALLBACK.strip() \
                    or any(sign in answer for sign in NO_ANSWER_SIGNALS):
                answer = f"关于「{questions[0]}」，我这边没查到确切的资料，建议转人工确认。"
            final_text.append(answer)
            await queue.put(answer)
            separator = "\n\n"
            final_text.append(separator)
            await queue.put(separator)
            log.info("多意图：动作分支前已补答只读子问", extra={
                "event": "turn.multi_intent.read_first",
                "read_questions": questions,
            })
        except Exception as exc:                        # 补答失败不能拖垮动作分支
            log.warning(f"旁路咨询作答失败，跳过：{exc}")
        finally:
            ctx.question = origin

    async def _run_handler_node(self, state, handler, dst_tool_count: int,
                                branch: str) -> dict:
        """执行一条确定性分支：流式写队列 → 落历史 → 推进 DST → 审计。"""
        turn: TurnState = state["turn"]
        queue: asyncio.Queue = state["queue"]
        final_text: List[str] = state["final_text"]
        # 分支分布指标：回答「请求实际走了哪条链路」
        branch_total.labels(branch).inc()
        # 多意图「读 + 写」：动作分支前先把只读子问答掉。
        # 动作回复会让用户去补槽位（"请提供订单号"），咨询一旦压在后面就永久丢失——
        # 实测「能开专票吗，另外我要退货」只回了退货引导，发票问题整段消失。
        read_questions = state.get("read_questions") or []
        if read_questions and branch in self.ACTION_BRANCHES:
            await self._answer_read_questions(state, read_questions)
        started = time.perf_counter()
        status, error_code = "success", ""
        with span(f"handler.{branch}", kind="internal"):
            try:
                async for piece in handler(turn.ctx):
                    final_text.append(piece)
                    await queue.put(piece)
            except Exception as e:
                status, error_code = "error", type(e).__name__
                raise
            finally:
                answer = "".join(final_text)
                if answer:
                    await finalize_turn(turn.chat_history, turn.ctx.question, answer)
                    if dst_tool_count and DST_ENABLED:
                        self._schedule_dst_update(turn.ctx.session_id, turn.ctx.question,
                                                  answer, dst_tool_count)
                # 确定性 handler 也进审计：Agent 路径按单工具记，handler 路径按分支记，
                # 这样「工具审计」不再只覆盖 ReAct 兜底（此前 query_order 等分支完全无审计）。
                await self._record_tool_audit(
                    turn, f"handler.{branch}", status, error_code,
                    int((time.perf_counter() - started) * 1000),
                    {"question": turn.ctx.question}, answer)
        return {"branch": branch}

    async def _run_agent_node(self, state) -> dict:
        """泛化 ReAct 兜底节点：确定性分支都不命中时交给 Agent。"""
        branch_total.labels("agent").inc()
        await self._produce_agent(state["turn"], state["queue"], state["final_text"])
        return {"branch": "agent"}

    # ---------- 产出阶段 ----------
    async def _produce(self, turn: TurnState, queue: asyncio.Queue):
        """（独立任务）经 LangGraph 状态机生成答复：路由 → 分支 → 流式产出 → 落库。"""
        final_text: List[str] = []
        # 身份注入点（PROJECT_AUDIT P0-3）：工具读的是服务端上下文，不是 LLM 填的参数。
        # 必须在这里 set —— 本方法是独立任务，其上下文在任务创建时复制，父任务里 set 不会
        # 追溯生效；set 在本任务内则天然按「一轮 = 一个任务」隔离，且随任务结束一起销毁。
        user_token = set_current_user(turn.ctx.user.get("user_id", ""))
        try:
            # 一轮的根 span：覆盖确定性 handler / Agent 兜底；其内再嵌 intent/retrieve/llm/tool
            with span("agent.turn", kind="internal"):
                await self._get_turn_graph().ainvoke({
                    "ctx": turn.ctx,
                    "turn": turn,
                    "queue": queue,
                    "final_text": final_text,
                    # 每轮读取，测试可 patch 模块常量
                    "dst_enabled": DST_ENABLED,
                })
        finally:
            reset_current_user(user_token)

    async def _produce_agent(self, turn: TurnState, queue: asyncio.Queue,
                             final_text: List[str]):
        """Agent 流式产出编排：只把面向用户的正文透出，工具调用与推理一律吞掉。

        B2 协议统一为**原生 tool calling**：工具调用是结构化 tool_calls，正文即最终
        AIMessage 内容，不依赖任何文本标记。带 tool_call_chunks 的块是工具调用、
        带 reasoning_content 的块是内部思维链，两者都不透出；其余非空内容块即正文，
        在「空答复占位」判定后直接流式透出。
        """
        question = turn.ctx.question
        session_id = turn.ctx.session_id
        pending = ""       # 正文缓冲：空答复前缀押后判定（见 _replace_empty_answer）
        # 思考过程埋点：每步模型调用的 TTFT / 总耗时 / 产出字数 / 是否决定调工具
        # tool_idx：本次工具调用的「序号集合」。流式下一个工具会被拆成多个
        # tool_call_chunks，早期用计数器累加会把 2 个工具记成 9（分片数），
        # 日志里就变成「模型在乱调工具」。按 index 去重才是真实工具个数。
        think = {"step": 0, "t0": 0.0, "first": 0.0, "chars": 0,
                 "tool_idx": set(), "tool_chunks": 0, "reasoning": 0}
        passthrough = False
        tool_calls_this_round = {"n": 0}   # 统计本轮工具调用（推进 EXECUTING 阶段）
        tool_started: Dict[str, tuple] = {}  # run_id → (打点时刻, 入参)：耗时观测 + 审计落库
        # 熔断：LLM 连续失败时不再进入 Agent 循环（否则「多轮工具调用 + 超时」会雪崩），
        # 直接给确定性兜底话术。Agent 与 RAG 生成同用一个 DashScope LLM 后端，故共用 llm_breaker。
        if not llm_breaker.allow_request():
            log.warning(f"LLM 熔断中，Agent 直接兜底 | 问题={question[:40]}")
            agent_fallback_total.labels("breaker").inc()
            final_text.append(AGENT_NO_ANSWER_FALLBACK)
            await queue.put(AGENT_NO_ANSWER_FALLBACK)
            return
        agent_t0 = time.perf_counter()
        try:
            async for event in self.unified_agent.astream_events(
                {"messages": turn.messages},
                config={"recursion_limit": 10},
                # v2 是当前稳定事件协议；v1 已废弃（B2 协议统一）
                version="v2",
            ):
                # ---- 工具调用观测：开始 / 完成 / 失败（日志自动带 request_id）----
                if event["event"] == "on_tool_start":
                    tool_calls_this_round["n"] += 1
                    args = (event.get("data") or {}).get("input")
                    tool_started[event.get("run_id", "")] = (time.perf_counter(), args)
                    log.info("工具调用开始", extra={
                        "event": "agent.tool.start",
                        "seq": tool_calls_this_round["n"],
                        "tool": event.get("name"),
                        "arguments": _summarize_value(args),
                    })
                    continue
                if event["event"] == "on_tool_end":
                    tool = event.get("name") or "unknown"
                    start, args = tool_started.pop(event.get("run_id", ""), (None, None))
                    if start is not None:
                        elapsed = time.perf_counter() - start
                        output = (event.get("data") or {}).get("output")
                        agent_tool_calls_total.labels(tool, "success").inc()
                        agent_tool_duration_seconds.labels(tool).observe(elapsed)
                        log.info("工具调用完成", extra={
                            "event": "agent.tool.done",
                            "tool": tool, "status": "ok",
                            "dur_ms": round(elapsed * 1000, 1),
                            "result": _summarize_value(output),
                        })
                        # 审计落库（C1）：一次调用一行，入参脱敏、结果只存摘要哈希
                        await self._record_tool_audit(
                            turn, tool, "success", "", int(elapsed * 1000), args, output)
                    continue
                if event["event"] == "on_tool_error":
                    tool = event.get("name") or "unknown"
                    err = (event.get("data") or {}).get("error")
                    started = tool_started.pop(event.get("run_id", ""), None)
                    duration_ms = int((time.perf_counter() - started[0]) * 1000) \
                        if started else 0
                    error_code = type(err).__name__ if err is not None else "tool_error"
                    agent_tool_calls_total.labels(tool, "error").inc()
                    log.warning("工具调用失败", extra={
                        "event": "agent.tool.error", "tool": tool,
                        "status": "error", "err": _summarize_value(err),
                        "dur_ms": duration_ms,
                    })
                    # 审计落库（C1）：失败也留痕（status=error + 异常类名），入参同样脱敏
                    await self._record_tool_audit(
                        turn, tool, "error", error_code, duration_ms,
                        started[1] if started else None)
                    continue
                # 每步模型调用：开始时清零计数，结束时落一条「思考完成」，
                # 于是日志里能读出一条思维链：第1步想了 X ms → 决定调 A → 第2步想了 Y ms → 出答案
                if event["event"] == "on_chat_model_start":
                    think.update(step=think["step"] + 1, t0=time.perf_counter(),
                                 first=0.0, chars=0, tool_idx=set(),
                                 tool_chunks=0, reasoning=0)
                    continue
                if event["event"] == "on_chat_model_end":
                    if think["t0"]:
                        ended = time.perf_counter()
                        log.info("思考完成", extra={
                            "event": "agent.think.done",
                            "step": think["step"],
                            "ttft_ms": round(((think["first"] or ended) - think["t0"]) * 1000, 1),
                            "dur_ms": round((ended - think["t0"]) * 1000, 1),
                            "chars": think["chars"],
                            # 只记「思维链有多长」，不记内容：原文可能含用户隐私与
                            # 内部提示词，体积又是正文数倍，不适合进日志。
                            "reasoning_chars": think["reasoning"],
                            "tool_calls": len(think["tool_idx"]),
                            "tool_chunks": think["tool_chunks"],
                        })
                        think["t0"] = 0.0
                    continue
                if event["event"] != "on_chat_model_stream":
                    continue
                chunk = event["data"]["chunk"]
                msg = getattr(chunk, "message", None) or chunk
                if think["t0"] and not think["first"]:
                    think["first"] = time.perf_counter()
                # 工具调用阶段的结构化块（带 tool_call_chunks）一律跳过：那是工具调用，
                # 不是面向用户的正文。原生协议下正文与工具调用是两类块，天然可区分。
                _tool_chunks = getattr(msg, "tool_call_chunks", None)
                if _tool_chunks:
                    think["tool_chunks"] += 1
                    for _tc in _tool_chunks:
                        _idx = _tc.get("index") if isinstance(_tc, dict) else None
                        if _idx is not None:
                            think["tool_idx"].add(_idx)
                    continue
                # 推理模型（如部分 qwen/deepseek 变体）把思维链放在 reasoning_content：
                # 那是内部推理，绝不能当正文透出。
                _reasoning = (getattr(msg, "additional_kwargs", None) or {}).get("reasoning_content")
                if _reasoning:
                    think["reasoning"] += len(str(_reasoning))
                    continue
                content = msg.content if isinstance(msg.content, str) else ""
                if not content:
                    continue
                think["chars"] += len(content)
                # 与 RAG 链路同一处理：正文可能是「无相关信息」这类占位答复，
                # 先把「可能是空答复」的前缀押后，一旦偏离所有候选写法立即 flush 转直通。
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

            # LLM 流式产出正常结束：记为一次成功，CLOSED 态复位失败计数
            llm_breaker.record_success()
            if pending:
                # 收尾只剩「可能是空答复」的 pending（正常正文早已透出）
                pending = self._replace_empty_answer(pending, question)
                final_text.append(pending)
                await queue.put(pending)
            elif not final_text:
                # 一个正文块都没有（只调了工具、或模型空回复）：走确定性知识兜底，
                # 否则用户拿到一片空白。
                log.warning(f"Agent 未产出有效正文，转知识兜底 | 问题={question[:40]}")
                await self._knowledge_fallback(turn, queue, final_text, "no_content")
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
            turn_ms = (time.perf_counter() - agent_t0) * 1000
            agent_duration_seconds.observe(time.perf_counter() - agent_t0)
            # 本轮汇总：一次 ReAct 的步数/工具数/总耗时，排障时看这一行就够
            log.info("Agent 本轮结束", extra={
                "event": "agent.turn.done",
                "steps": think["step"],
                "tools": tool_calls_this_round["n"],
                "dur_ms": round(turn_ms, 1),
                "answer_chars": len("".join(final_text)),
                "fallback": not bool(final_text),
            })
            answer = "".join(final_text)
            if answer:
                await finalize_turn(turn.chat_history, question, answer)
                # DST：更新会话状态（槽位提取→合并→推进阶段→持久化），不影响已返回的流
                if DST_ENABLED:
                    self._schedule_dst_update(session_id, question, answer,
                                              tool_calls_this_round["n"])

    # ---------- 正文收尾辅助 ----------
    @staticmethod
    def _replace_empty_answer(pending: str, question: str) -> str:
        """整段正文是「无相关信息」这类空答复 → 换确定性兜底话术，不原样抛给用户。"""
        if is_empty_answer(pending):
            log.info(f"Agent 空答复改走兜底 | 问题={question[:40]}")
            agent_fallback_total.labels("empty_answer").inc()
            return KNOWLEDGE_NO_ANSWER.format(contact=SERVICE_CONTACT)
        return pending

    async def _knowledge_fallback(self, turn: TurnState, queue: asyncio.Queue,
                                  final_text: List[str], reason: str) -> None:
        """Agent 没能产出有效正文时，改走确定性知识兜底（RAG 检索作答）。

        两条协议共用：都可能在「没有正文」时落到这里，统一按 no_content 兜底率计数。
        """
        agent_fallback_total.labels(reason).inc()
        pieces = []
        async for piece in stream_knowledge_answer(
                self.retrieval_service, self.answer_generator,
                turn.ctx.question, turn.ctx.session_id):
            pieces.append(piece)
            await queue.put(piece)
        final_text.append("".join(pieces) or AGENT_NO_ANSWER_FALLBACK)

    # ---------- 工具调用审计（C1）----------
    async def _record_tool_audit(self, turn: TurnState, tool_name: str, status: str,
                                 error_code: str, duration_ms: int, args: Any,
                                 result: Any = None) -> None:
        """把一次 Agent 工具调用落进审计表 agent_tool_calls。

        入参脱敏、结果只存摘要哈希（见 _redact_args/_result_digest）。
        同步 DB 调用经 asyncio.to_thread 下线程池，不阻塞事件循环。
        观测失败只记日志、不打断本轮答复 —— 审计是旁路，不能影响主链路可用性。

        诚实边界：埋点只覆盖 astream_events 路径上的 @tool 调用；确定性 handler 分支
        （order_query/my_orders/progress/complaint/return_refund/consult）直接调存储层、
        不经工具，故不会在此产生审计行。
        """
        try:
            user_id = (turn.ctx.user or {}).get("user_id", "")
            await asyncio.to_thread(
                mysql_store.record_tool_call,
                trace_id=get_trace_id(),
                request_id=get_request_id(),
                session_id=turn.ctx.session_id,
                user_id=user_id,
                tool_name=tool_name,
                args_json=_redact_args(args),
                result_digest=_result_digest(result),
                status=status,
                error_code=error_code,
                duration_ms=duration_ms,
            )
        except Exception as e:
            log.error(f"工具调用审计落库失败（不影响答复）: {e}", exc_info=True)

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
            dst = await self.dst_manager.load(session_id)
            new_slots = {}
            if self.slot_extractor is not None:
                new_slots = await self.slot_extractor.extract(question, dst.intent or "", dst.slots)
            dst = self.dst_manager.merge_new_slots(dst, new_slots)
            dst.tool_call_count += tool_count
            dst.answer = answer or dst.answer
            stage = self.dst_manager.advance_stage(dst, tool_executed=tool_count > 0)
            await self.dst_manager.save(dst)
            log.info(f"DST 更新 | session={session_id} | intent={dst.intent} | "
                     f"slots={dst.slots} | stage={stage}")
        except Exception as e:
            log.error(f"DST 更新失败（不影响已返回响应）: {e}", exc_info=True)
