"""RAG 一轮问答：缓存命中直接秒回；未命中检索 + 流式生成 + 独立任务落库。

与 Agent 链路共用流式生产者骨架（见 service/turn/streaming.py），但不带历史意图、
DST 与工具调度，只做「检索 → 生成 → 缓存 → 落库」。
"""
import asyncio
import time
from typing import Optional

from fastapi import HTTPException

from infrastructure.chat_history_factory import init_chat_history
from logs.log_config import log
from service.cache import ResponseCache
from shared.constants import could_be_empty_answer, is_empty_answer
from shared.reply_templates import (
    AGENT_NO_ANSWER_FALLBACK,
    KNOWLEDGE_NO_ANSWER,
    SERVICE_CONTACT,
)
from shared.schemas import RAGRequest
from utils.metrics import agent_fallback_total, agent_ttft_seconds

from .streaming import (
    _StreamQueue,
    _TTFTRecorder,
    _spawn_producer,
    finalize_turn,
)


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
        full_answer: list = []
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
