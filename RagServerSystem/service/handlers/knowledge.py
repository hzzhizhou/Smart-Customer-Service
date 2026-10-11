"""知识咨询：确定性 RAG 直答，绕过 ReAct 工具循环。

背景：consult 交给 Agent 时，qwen 有一定概率不调 search_knowledge 就直接凭参数
记忆作答，编出并不存在的商品（实测"你们店里有哪些手机可以选"5 次里 2 次编出
"星辰"系列机型，检索日志中无对应记录，可确认是幻觉）。
改由规则确定性判定后直答：直接走 RetrievalService（含路由/rerank/置信度门控）
→ answer_generator，保证答复有据可依，响应也更快。

stream_knowledge_answer 同时被 ReAct 兜底路径复用：模型只吐推理、没给正文时，
统一走这里检索作答，避免用户拿到空白答复。
"""
from typing import AsyncIterator

from logs.log_config import log
from service.handlers.context import TurnContext
from shared.constants import could_be_empty_answer, is_empty_answer
from utils.metrics import agent_fallback_total
from shared.reply_templates import (
    AGENT_NO_ANSWER_FALLBACK,
    KNOWLEDGE_NO_ANSWER,
    SERVICE_CONTACT,
)


async def stream_knowledge_answer(retrieval_service, answer_generator,
                                  question: str, session_id: str) -> AsyncIterator[str]:
    """确定性知识作答（异步生成器）：检索 → 置信度门控 → 流式生成。

    低置信度时不硬答，改引导转人工（与 RAG 主链路的门控口径一致）。

    生成阶段还要挡一层空答复：检索命中但资料答不到问题时（如问平台外信息，
    用户问题指向的外部信息不在语料里），模型会按提示词把「无相关信息」当正文吐出来。
    这四个字对用户等于没答，且此前会原样透出（实测 J2「京东上同款 iPhone 15 卖多少钱」），
    故与 RAG 主链路同口径——先押后「可能是空答复」的前缀，整段确认为空答复即换兜底话术。
    """
    docs, _retriever_type, gate_info = await retrieval_service.retrieve(
        question=question,
        route_mode="rule",
        use_context=True,
        use_hyde=False,
        use_multi=False,
        session_id=session_id,
        auto_filter=True,   # 按问题关键词推断文档类别，先在 metadata 上缩小范围再检索
    )
    if gate_info.get("should_escalate") or not docs:
        log.info(f"知识兜底：检索置信度不足，转人工 | level={gate_info.get('level')}")
        agent_fallback_total.labels("low_confidence").inc()
        yield KNOWLEDGE_NO_ANSWER.format(contact=SERVICE_CONTACT)
        return
    produced = False
    pending = ""
    passthrough = False
    async for chunk in answer_generator.stream_generate(question, docs, session_id):
        produced = True
        if passthrough:
            yield chunk
            continue
        pending += chunk
        if could_be_empty_answer(pending):
            continue
        passthrough = True
        yield pending
        pending = ""
    if pending:
        # 整段收完仍没直通：是空答复 → 换兜底话术；否则（如「好的」这类短答）原样透出
        if is_empty_answer(pending):
            log.info(f"知识作答空答复改走兜底 | 问题={question[:40]}")
            agent_fallback_total.labels("empty_answer").inc()
            yield KNOWLEDGE_NO_ANSWER.format(contact=SERVICE_CONTACT)
        else:
            yield pending
    elif not produced:
        agent_fallback_total.labels("empty_answer").inc()
        yield KNOWLEDGE_NO_ANSWER.format(contact=SERVICE_CONTACT)


async def handle_consult(ctx: TurnContext) -> AsyncIterator[str]:
    """产出知识咨询答复；生成过程异常时转人工兜底话术。"""
    try:
        async for piece in stream_knowledge_answer(
                ctx.retrieval_service, ctx.answer_generator,
                ctx.question, ctx.session_id):
            yield piece
    except Exception as e:
        log.error(f"确定性知识作答失败: {e}", exc_info=True)
        yield AGENT_NO_ANSWER_FALLBACK