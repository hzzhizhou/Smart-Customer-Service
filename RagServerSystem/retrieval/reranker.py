"""
重排：对「融合排序」后的候选做 Cross-Encoder 精排。

与融合的分工：融合只按名次/分数把多路结果并到一起（便宜、粗），重排让 query 与
每篇候选真正过一遍模型（贵、准），只在这一步做精排。因此候选要先截断再打分，
CPU 下这是延迟的主要来源。
"""
import asyncio
import time
from typing import List

from langchain_core.documents import Document

from config.settings import RERANK_CANDIDATES, RERANK_MODEL_PATH, RERANK_TOP_N
from logs.log_config import retrieval_layer_log as log
from retrieval.fusion import dedupe_docs


class CrossEncoderReranker:
    """基于本地 Cross-Encoder 的重排器（BAAI/bge-reranker-base）。

    四个要点：
      - 懒加载：首次调用才载入模型，且放进线程池执行 —— 否则 278M 权重会在事件循环
        里同步加载数秒，把并发请求全卡住
      - 去重：多查询/多子问的结果是 concat 起来的，同一文档可能出现多次，重复打分纯属浪费
      - 截断：只对候选上限内的文档打分，控制 CPU 耗时
      - 降级：模型缺失或推理失败时保持融合序返回并告警，不让整条检索链路失败
    """

    def __init__(self, model_path: str = RERANK_MODEL_PATH,
                 top_n: int = RERANK_TOP_N,
                 candidates: int = RERANK_CANDIDATES):
        self.model_path = model_path
        self.top_n = top_n
        self.candidates = candidates
        self._model = None
        self._load_failed = False
        self._load_lock = asyncio.Lock()

    def _load_model(self):
        """在线程池中调用：载入本地模型（不访问网络）。"""
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_path)
        log.info(f"Cross-Encoder 模型加载完成: {self.model_path}")

    async def _ensure_model(self) -> bool:
        """确保模型可用；返回 False 表示应降级为不重排。"""
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        # 加锁避免并发请求同时加载两份权重（各自约 1.1GB）
        async with self._load_lock:
            if self._model is not None:
                return True
            if self._load_failed:
                return False
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self._load_model)
            except Exception as e:
                self._load_failed = True
                log.error(f"Cross-Encoder 模型加载失败（{self.model_path}），"
                          f"降级为不重排、保持融合序: {e}")
                return False
        return True

    async def warmup(self) -> bool:
        """启动预热：提前加载模型并跑一次小推理。

        懒加载的代价实测约 18s（278M 权重加载 + torch 首次推理的惰性初始化），
        若不预热会全部落在重启后的第一个用户请求上（首字延迟 25s+，
        前端长时间转圈像"没反应"）。启动后在后台跑这里预热；
        预热完成前若有请求进来，rerank 会经 _load_lock 等待同一份加载结果。
        """
        if not await self._ensure_model():
            return False
        # 单对小样本推理：把 torch 首次推理的惰性开销（kernel 选择/内存池）
        # 也一并热掉，否则这部分仍会留在首个真实请求里
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, self._model.predict, [("预热", "模型")])
        except Exception as e:
            log.warning(f"Cross-Encoder 预热推理失败（不影响后续正常使用）: {e}")
        return True

    async def rerank(self, docs: List[Document], query: str) -> List[Document]:
        """按 query 与文档的相关性精排，返回 top_n 个文档。

        降级路径（无模型/推理失败）同样会去重并截断，只是顺序沿用融合序。
        """
        if not docs:
            return []
        candidates = dedupe_docs(docs)[:self.candidates]
        if len(candidates) <= 1 or not await self._ensure_model():
            return candidates[:self.top_n]

        start_time = time.time()
        pairs = [(query, doc.page_content) for doc in candidates]
        loop = asyncio.get_running_loop()
        try:
            scores = await loop.run_in_executor(None, self._model.predict, pairs)
        except Exception as e:
            log.error(f"Cross-Encoder 推理失败，降级为不重排、保持融合序: {e}")
            return candidates[:self.top_n]

        for doc, score in zip(candidates, scores):
            doc.metadata["rerank_score"] = float(score)
        candidates.sort(key=lambda doc: doc.metadata["rerank_score"], reverse=True)
        reranked_docs = candidates[:self.top_n]
        log.info(f"Cross-Encoder 重排序完成，候选 {len(pairs)} → 返回 {len(reranked_docs)} 个文档，"
                 f"耗时 {time.time() - start_time:.3f}s")
        return reranked_docs
