"""
检索层:Bm25检索,相似度检索,混合检索
通过路由实现智能检索，选择合适的检索方式
实现HyDE,重排序
"""
from functools import partial
from infrastructure.EmbeddingService.embedding_service import EmbeddingService
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from config.settings import (
    ROUTE_MODE, CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_MARGIN_THRESHOLD, DOC_CATEGORY_PRIORITY, DOC_CATEGORY_KEYWORDS,
    RERANK_MODE, RERANK_TOP_N
)
from logs.log_config import retrieval_layer_log as log
from retrieval.bm25_retriever import Bm25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.async_hybrid_retriever import HybridRetriever
from retrieval.router import SmartRouter
from retrieval.reranker import CrossEncoderReranker
from retrieval.fusion import dedupe_docs
from retrieval.query_enhancer import QueryEnhancer
from shared.constants import looks_like_multi_question, split_sub_questions
from utils.metrics import retrieval_stage_duration_seconds
import time
import asyncio


def infer_doc_categories(question: str) -> List[str]:
    """按问题关键词推断文档类别（检索前的 metadata 预过滤），返回所有命中的类别。
    """
    text = (question or "").lower()
    return [category for category in DOC_CATEGORY_PRIORITY
            if any(kw in text for kw in DOC_CATEGORY_KEYWORDS.get(category, ()))]


class RetrievalService:
    def __init__(self, vector_store, llm, chat_history):
        self.llm = llm
        self.vector_store = vector_store
        self.chat_history = chat_history
        self.embedding_service = EmbeddingService(vector_store.embedding_model)
        self.bm25_retriever = Bm25Retriever(self.vector_store)
        self.vector_retriever = VectorRetriever(self.vector_store,self.embedding_service)
        self.hybrid_retriever = HybridRetriever(self.bm25_retriever, self.vector_retriever,self.embedding_service)
        self.router = SmartRouter(self.bm25_retriever, self.vector_retriever, self.hybrid_retriever, self.llm)
        # 重排器（Cross-Encoder）单独一层：融合负责把多路结果并成一个排序，重排再精排一次
        self.reranker = CrossEncoderReranker() if RERANK_MODE == "cross_encoder" else None
        if self.reranker is None:
            log.info(f"RERANK_MODE={RERANK_MODE}，已关闭重排（直接返回融合序）")
        self.query_enhancer = QueryEnhancer(self.llm)

    async def _merge_results(self, docs: List[Document], original_question: str) -> List[Document]:
        """融合结果 → 重排。关闭重排时仅去重并截断，顺序沿用融合结果。"""
        if self.reranker is None:
            return dedupe_docs(docs)[:RERANK_TOP_N]
        return await self.reranker.rerank(docs, original_question)

    async def retrieve(self, question: str, route_mode: str = ROUTE_MODE,
                 use_context: bool = True,
                 use_hyde: bool = False,
                 use_multi: bool = False,
                 session_id: str = None,
                 filter_metadata: Optional[dict] = None,
                 auto_filter: bool = False) -> Tuple[List[Document], str, dict]:
        start_time = time.time()
        original_question = question

        # 1. 预取 embedding（基于原始问题）
        #    仅用于与“查询改写”并行抢占 embedding 计算；包装成安全任务吞掉异常，
        #    避免成为无人 await 的 fire-and-forget 导致 “Task exception was never retrieved”。
        async def _safe_prefetch(q: str):
            try:
                await self.embedding_service.embed(q)
            except Exception as e:
                log.warning(f"预取 embedding 失败（不影响主链路，下游会自行计算）: {e}")
        prefetch_task = asyncio.create_task(_safe_prefetch(original_question))

        # 2. 异步执行查询改写
        #    注意：必须用"当前会话"的历史做改写。若沿用构造时传入的全局共享 chat_history，
        #    历史会随所有请求累积，导致每条单轮请求都触发改写 LLM（约 +8s）。
        #    新会话历史为空时，rewrite_question 内部直接返回原问题，不调 LLM。
        rewrite_start = time.time()
        if use_context:
            loop = asyncio.get_running_loop()
            from infrastructure.chat_history_factory import init_chat_history
            current_hist = init_chat_history(session_id) if session_id else None
            if current_hist is not None:
                rewrite_func = partial(current_hist.rewrite_question, original_question, self.llm)
                rewritten_question = await loop.run_in_executor(None, rewrite_func)
                if rewritten_question != original_question:
                    log.info(f"查询改写: {original_question} -> {rewritten_question}")
            else:
                rewritten_question = original_question
        else:
            rewritten_question = original_question
        retrieval_stage_duration_seconds.labels("rewrite").observe(time.time() - rewrite_start)

        # 3. 查询分解：一次提出多个问题时逐子问检索。
        #    整段拼成一个查询会让各子问题的候选互相挤占 RERANK_TOP_N 名额，
        #    靠后的子问题被挤出结果 → 生成层无据可依，该问静默消失（实测复现）。
        decompose_start = time.time()
        sub_questions = await self._decompose_queries(rewritten_question)
        retrieval_stage_duration_seconds.labels("decompose").observe(time.time() - decompose_start)

        log.debug(f"查询改写耗时:{time.time() - start_time}")
        retriever_time = time.time()

        if len(sub_questions) > 1:
            final_docs, retriever_type, gate_info = await self._retrieve_per_sub_question(
                sub_questions, route_mode, filter_metadata, auto_filter)
        else:
            final_docs, retriever_type, gate_info = await self._retrieve_one(
                rewritten_question, original_question, route_mode,
                filter_metadata, auto_filter, use_hyde, use_multi)

        end_time = time.time() - retriever_time
        log.info(f"检索完成，检索器={retriever_type}，返回文档数={len(final_docs)},"
                 f"置信度={gate_info['level']}(top={gate_info['top_score']}),"
                 f"子问数={len(sub_questions)},检索总耗时：{end_time:.2f}秒")
        # 回收预取任务（其内部已吞异常；这里确保不遗留未完成/未回收的后台任务）
        await asyncio.gather(prefetch_task, return_exceptions=True)
        # 整段检索耗时（含预取回收）：与 rewrite/decompose/search/rerank 各阶段对账
        retrieval_stage_duration_seconds.labels("total").observe(time.time() - start_time)
        return final_docs, retriever_type, gate_info

    async def _decompose_queries(self, question: str) -> List[str]:
        """查询分解：把一次提出的多个问题拆成独立子问题。

        先用确定性规则（免费、可单测）；规则拆不出多段、但文本看起来像多问题时，
        才让 LLM 兜底拆一次，避免每轮都多付一次模型调用。
        """
        parts = split_sub_questions(question)
        if len(parts) > 1:
            log.info(f"查询分解（规则）：{question} -> {parts}")
            return parts
        if looks_like_multi_question(question):
            parts = await self.query_enhancer.decompose_query(question)
            if len(parts) > 1:
                log.info(f"查询分解（LLM）：{question} -> {parts}")
            return parts
        return [question]

    @staticmethod
    def _infer_filter(search_question: str, filter_metadata: Optional[dict],
                      auto_filter: bool) -> Optional[dict]:
        """metadata 预过滤：按问题关键词推断文档类别，先在范围上缩小再检索。

        类别来自文件名派生（loader 写入 doc_category），无把握时不过滤。
        """
        if auto_filter and not filter_metadata:
            categories = infer_doc_categories(search_question)
            if categories:
                # 单类 → 等值过滤；跨域（多类）→ $in 取并集，避免把答案所在的那一类整类排除
                where = {"doc_category": categories[0]} if len(categories) == 1 \
                    else {"doc_category": {"$in": categories}}
                log.info(f"检索预过滤：按文档类别缩小范围 doc_category={categories}")
                return where
        return filter_metadata

    async def _search_and_rerank(self, queries: List[str], retriever,
                                 rerank_query: str,
                                 filter_arg: Optional[dict]) -> List[Document]:
        """按给定 metadata 过滤条件执行检索（可多查询）并重排。

        两段分开打点：召回（多查询并行）与精排（融合去重 + Cross-Encoder）。
        CPU 环境下 Cross-Encoder 是检索侧主要耗时来源，拆开才能看出占比。
        """
        search_start = time.time()
        results_list = await asyncio.gather(
            *[retriever.ainvoke(q, filter=filter_arg) for q in queries])
        docs = [doc for sublist in results_list for doc in sublist]
        retrieval_stage_duration_seconds.labels("search").observe(time.time() - search_start)

        rerank_start = time.time()
        merged = await self._merge_results(docs, rerank_query)
        retrieval_stage_duration_seconds.labels("rerank").observe(time.time() - rerank_start)
        return merged

    async def _retrieve_one(self, search_question: str, rerank_question: str,
                            route_mode: str, filter_metadata: Optional[dict],
                            auto_filter: bool, use_hyde: bool,
                            use_multi: bool) -> Tuple[List[Document], str, dict]:
        """单问题检索：路由 → 预过滤 → （HyDE/多查询）→ 检索重排 → 门控。"""
        retriever, retriever_type = await self.router.route(search_question, route_mode)
        filter_arg = self._infer_filter(search_question, filter_metadata, auto_filter)

        queries = [search_question]
        if use_multi:
            queries = await self.query_enhancer.multi_query(search_question)
            log.info(f"生成的多个问题{queries}")
        elif use_hyde:
            queries = [await self.query_enhancer.hyde(search_question)]

        docs = await self._search_and_rerank(queries, retriever, rerank_question, filter_arg)
        if filter_arg and not docs:
            # 过滤后一条都不剩（关键词猜错类别 / 该类目无内容）→ 回退全库，宁可多检索也不漏答
            log.info(f"预过滤后无命中，回退全库检索 | filter={filter_arg}")
            docs = await self._search_and_rerank(queries, retriever, rerank_question, None)
        return docs, retriever_type, self._compute_confidence_gate(docs)

    async def _retrieve_per_sub_question(self, sub_questions: List[str], route_mode: str,
                                         filter_metadata: Optional[dict],
                                         auto_filter: bool) -> Tuple[List[Document], str, dict]:
        """多子问检索：每个子问独立路由/预过滤/重排，各自独占 RERANK_TOP_N 名额。

        子问并行执行；门控逐子问计算：先剔除「答不上来」的子问，只在能答的子问之间
        取最弱（保证答不了就转人工，同时不让一个查不到的子问把整轮拖去转人工）。

        此前直接对全部子问取最弱，实测「你们有哪些在售手机？iPhone 15 多少钱？」因为前一个
        子问没有直接语料（top 0.54 → low），连能答的「iPhone 15 4999 元」也一并被放弃，
        整轮回"没查到可靠的资料"（G3）——用户明明有一半问题是可以答的。
        """
        async def _one(sub_q: str) -> Tuple[List[Document], str]:
            retriever, retriever_type = await self.router.route(sub_q, route_mode)
            filter_arg = self._infer_filter(sub_q, filter_metadata, auto_filter)
            docs = await self._search_and_rerank([sub_q], retriever, sub_q, filter_arg)
            if filter_arg and not docs:
                log.info(f"预过滤后无命中，回退全库检索 | 子问={sub_q} filter={filter_arg}")
                docs = await self._search_and_rerank([sub_q], retriever, sub_q, None)
            return docs, retriever_type

        results = await asyncio.gather(*[_one(q) for q in sub_questions])
        doc_lists = [docs for docs, _ in results]
        retriever_type = "+".join(dict.fromkeys(t for _, t in results))
        gates = [self._compute_confidence_gate(docs) for docs in doc_lists]

        answerable = [(docs, gate) for docs, gate in zip(doc_lists, gates)
                      if gate["level"] != "low"]
        if answerable:
            if len(answerable) != len(doc_lists):
                log.info(f"多子问门控：{len(doc_lists) - len(answerable)} 个子问置信度不足已剔除，"
                         f"仅用其余 {len(answerable)} 个作答")
            return (self._merge_sub_results([d for d, _ in answerable]), retriever_type,
                    self._combine_gates([g for _, g in answerable]))
        # 所有子问都答不了 → 保持整体转人工
        return self._merge_sub_results(doc_lists), retriever_type, self._combine_gates(gates)

    @staticmethod
    def _merge_sub_results(doc_lists: List[List[Document]]) -> List[Document]:
        """合并各子问结果：按名次交错（轮转）排列，并按文档去重。

        交错而非首尾相接：生成层会把上下文按顺序截断到 MAX_CONTEXT_TOKENS，
        串行拼接会让后一个子问的资料整段被截掉 —— 正是本轮要修的问题。
        """
        merged: List[Document] = []
        seen = set()
        depth = max((len(d) for d in doc_lists), default=0)
        for i in range(depth):
            for docs in doc_lists:
                if i >= len(docs):
                    continue
                doc = docs[i]
                key = (doc.metadata.get("file_name", ""), doc.page_content)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
        return merged

    @staticmethod
    def _combine_gates(gates: List[dict]) -> dict:
        """多子问门控合并：整体取最弱子问（top_score 取 min、should_escalate 取 any）。

        只按最强子问判 high 会让弱子问被硬答成幻觉；取最弱才能保证"答不了就转人工"。
        """
        rank = {"low": 0, "medium": 1, "high": 2}
        weakest = min(gates, key=lambda g: (rank.get(g["level"], 0), g["top_score"]))
        return {
            "top_score": round(min(g["top_score"] for g in gates), 4),
            "mean_score": round(min(g["mean_score"] for g in gates), 4),
            "margin": weakest["margin"],
            "level": weakest["level"],
            "should_escalate": any(g["should_escalate"] for g in gates),
            "reason": weakest["reason"],
        }

    def _compute_confidence_gate(self, docs: List[Document]) -> dict:
        """
        置信度门控：基于绝对相似度判断是否需要转人工

        双信号判断：
          1. 主信号 = vector_score（=1-distance/2，即余弦相似度，绝对分、范围 [0,1]，语义稳定）
             - >= CONFIDENCE_HIGH_THRESHOLD → high 候选，直接回答
             - < CONFIDENCE_LOW_THRESHOLD    → low，转人工
             - 其余 → medium，回答但提示可转人工
          2. 辅信号 = margin（top - 第二名），衡量排序稳定性
             - high 但 margin < 阈值 → 降级 medium（多个候选分数接近，排序不稳健）

        """
        if not docs:
            return {
                "top_score": 0.0,
                "mean_score": 0.0,
                "margin": 0.0,
                "level": "low",
                "should_escalate": True,
                "reason": "no_results"
            }

        # 取绝对相似度：只认 vector_score
        # vector_score 是绝对分（余弦，[0,1]），而 rerank_score 是另一个模型的相关性打分、
        # fusion_score 是批次内相对分，三者的量纲与 cos 阈值都不可比，所以不做降级，
        # 缺失即记 0 分。
        def _abs_score(doc: Document) -> float:
            return float(doc.metadata.get("vector_score", 0.0))

        scores = sorted([_abs_score(doc) for doc in docs], reverse=True)
        top_score = scores[0]
        mean_score = sum(scores) / len(scores)
        margin = scores[0] - scores[1] if len(scores) > 1 else top_score

        # 主信号判断
        if top_score >= CONFIDENCE_HIGH_THRESHOLD:
            level, should_escalate, reason = "high", False, "confident"
        elif top_score >= CONFIDENCE_LOW_THRESHOLD:
            level, should_escalate, reason = "medium", False, "low_confidence"
        else:
            level, should_escalate, reason = "low", True, "below_threshold"

        # 辅信号：排序不稳定 → high 降级为 medium
        if level == "high" and margin < CONFIDENCE_MARGIN_THRESHOLD:
            level, should_escalate, reason = "medium", False, "unstable_ranking"

        return {
            "top_score": round(float(top_score), 4),
            "mean_score": round(float(mean_score), 4),
            "margin": round(float(margin), 4),
            "level": level,
            "should_escalate": should_escalate,
            "reason": reason
        }