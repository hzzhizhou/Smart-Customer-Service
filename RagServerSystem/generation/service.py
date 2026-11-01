"""
生成层：无幻觉生成+提示词工程+格式标准化+复杂问题处理
通过AnswerGenerator类，可以生成答案，支持流式生成和非流式生成
"""
from typing import List, Optional, AsyncIterator, Any
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.context_format import format_context
from utils.llm_factory import create_llm
from config.settings import (
    CHAT_HISTORY_WINDOW, MAX_CONTEXT_TOKENS,
)
from logs.log_config import generation_layer_log as log
from infrastructure.chat_history_factory import init_chat_history
from utils.output_guard import output_guard
from utils.circuit_breaker import llm_breaker
from utils.request_context import span
from langchain_core.messages import HumanMessage, AIMessage
import time

class AnswerGenerator:
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or create_llm(streaming=True, with_fallback=True)
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是智能售后客服，正在跟顾客对话。严格遵守以下要求：

【表达风格】像真人客服那样说话：
    1. 短句、口语化、直接给结论；不要写成条款说明书，不要出现"根据资料""参考资料显示"这类系统腔；
    2. 第一句先回答顾客最关心的问题，需要补充细节时最多再用 3 条短要点，每条一句话；
    3. 整体控制在 150 字以内。不要把资料里的表格、编号条款原样贴出来，要把表格里跟顾客问题
       相关的数据转成一句人话（例如"退货运费一般由您承担，质量问题则由商家承担"）；
    4. 结尾给一句自然的收尾：如果这事通常还有下一步操作（查订单、办退货、转人工），就主动追问
       一句（例如"需要我帮您查一下这单的物流吗？"）；纯政策咨询也用一句轻松的话收尾，别生硬截断。

【回答规则】
    5. 优先使用【对话历史】中的信息回答关于对话上下文的问题（如"上一个问题是什么"）；
    6. 仅使用资料中的知识回答，不添加额外信息；绝不编造金额、时限、订单号；
    7. 主动追问写在正文最后，来源标注放在整个回答的最末尾一行，
       格式：「来源：文档名」（若来自对话历史则标注「来源：对话历史」）；
    8. 无相关资料时，仅回复「无相关信息」。
    资料：{context}
    """),
            ("human", "用户问题：{question}")
        ])
        self.answer_chain = self.base_prompt | self.llm | StrOutputParser()

    def format_context(self, docs: List[Document]) -> str:
        # 按 token 估算截断（中文 1 字≈1 token，保守按字符数近似）
        return format_context(docs=docs, max_context_length=MAX_CONTEXT_TOKENS)

    async def stream_generate(self, question: str, docs: List[Document], session_id: Optional[str] = None) -> AsyncIterator[str]:
        """
        流式生成答案，逐步产出 token。
        注意：不存储对话历史（因为无法获得完整答案），如需存储请在外部收集完整答案后调用存储逻辑。
        """
        start_time = time.time()
        context_str = self.format_context(docs)
        # 添加对话历史（窗口大小统一为 CHAT_HISTORY_WINDOW，不再硬编码 2 条）
        if session_id:
            chat_hist = init_chat_history(session_id)
            # 事件循环内读历史：优先异步（MysqlChatHistory.amessages），兜底兼容 InMemory
            if hasattr(chat_hist, "amessages"):
                messages = await chat_hist.amessages()
            else:
                m = getattr(chat_hist, "messages", None)
                messages = m() if callable(m) else (m or [])
            if messages:
                history_lines = []
                # 取最近 CHAT_HISTORY_WINDOW 条（与 Agent 端一致，之前 RAG 取2、Agent 取6 不一致）
                for msg in messages[-CHAT_HISTORY_WINDOW:]:
                    role = "User" if isinstance(msg, HumanMessage) else "AI"
                    history_lines.append(f"{role}：{msg.content}")
                if history_lines:
                    history_str = "\n".join(history_lines)
                    context_str = f"对话历史:\n{history_str}\n\n上下文:\n{context_str}"
        # 熔断检查：LLM 连续失败时降级，避免雪崩
        if not llm_breaker.allow_request():
            log.warning("LLM 熔断中，返回兜底话术",
                        extra={"event": "llm.breaker.open", "stage": "generate"})
            # 不得声称「已创建工单」：本分支只返回话术、不建单（与 reply_templates
            # 第 335 行同一原则）。虚假承诺比不回复更糟——用户会等一个不存在的回访。
            yield "抱歉，AI 服务暂时不可用，请稍后重试；如需人工协助，请回复「转人工」。"
            return
        try:
            llm_start = time.perf_counter()
            full_answer = []
            first_chunk_at = None      # 首 token 时刻：TTFT 与总生成时长要分开看
            with span("llm.generate", kind="client"):
                async for chunk in self.answer_chain.astream({
                    "question": question,
                    "context": context_str
                }):
                    if first_chunk_at is None:
                        first_chunk_at = time.perf_counter()
                    full_answer.append(chunk)
                    yield chunk
            llm_end = time.perf_counter()
            llm_breaker.record_success()
            # 流式输出后做合规审查：若有风险，追加系统提示（不破坏已流式输出的内容）
            complete = "".join(full_answer)
            safe, issues = output_guard.review(complete, context_str, question)
            if issues:
                # 追加合规提示（前面已流式输出原文，这里补提示）
                tail = safe[len(complete):]  # review 在原文后追加的提示部分
                if tail:
                    yield tail
            # 结构化收尾：TTFT 看体感、gen_ms 看吞吐、chars 核对是否被截断
            log.info("回答生成完成", extra={
                "event": "llm.generate.done",
                "ttft_ms": round(((first_chunk_at or llm_end) - llm_start) * 1000, 1),
                "dur_ms": round((llm_end - llm_start) * 1000, 1),
                "chars": len("".join(full_answer)),
                "ctx_chars": len(context_str or ""),
                "total_ms": round((time.time() - start_time) * 1000, 1),
            })
        except Exception as e:
            llm_breaker.record_failure()
            log.error("回答生成失败", exc_info=True, extra={
                "event": "llm.generate.error", "err": type(e).__name__,
                "dur_ms": round((time.time() - start_time) * 1000, 1),
            })
            yield "系统异常，无法生成回答，请稍后重试或联系人工客服。"

    async def generate(self, question: str, docs: List[Document], session_id: Optional[str] = None) -> str:
        """非流式生成（保留原接口，内部使用流式拼接 + 输出合规审查）"""
        full_answer = []
        async for chunk in self.stream_generate(question, docs, session_id):
            full_answer.append(chunk)
        answer = "".join(full_answer)
        # 存储对话历史
        if session_id:
            chat_history = init_chat_history(session_id)
            # 事件循环内写历史：优先异步，兜底兼容 InMemory sync API
            if hasattr(chat_history, "async_add_message"):
                await chat_history.async_add_message(HumanMessage(content=question))
                await chat_history.async_add_message(AIMessage(content=answer))
            else:
                chat_history.add_user_message(question)
                chat_history.add_ai_message(answer)
        log.info(f"AI回答:\n{answer[:200]}")
        return answer