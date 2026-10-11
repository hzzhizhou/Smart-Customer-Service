"""
嵌入服务类，用于异步计算文本的嵌入向量。
"""
import hashlib
import asyncio
from typing import List, Dict
from functools import partial
from logs.log_config import retrieval_layer_log as log
from utils.metrics import cache_requests_total

class EmbeddingService:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self._cache: Dict[str, List[float]] = {}
        self._pending: Dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()          # 新增锁
        self.hit_count = 0
        self.miss_count = 0

    def _get_cache_key(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()

    async def embed(self, query: str) -> List[float]:
        cache_key = self._get_cache_key(query)

        async with self._lock:
            # 1. 检查缓存
            if cache_key in self._cache:
                self.hit_count += 1
                cache_requests_total.labels("embedding", "hit").inc()
                log.info(f"Embedding 缓存命中 (hit={self.hit_count}, miss={self.miss_count}): {query[:30]}...")
                return self._cache[cache_key]

            # 2. 检查是否有正在进行的计算
            if cache_key in self._pending:
                # 单航班合并：同一文本正在算，等它的结果即可。既非命中也不是新计算，
                # 故不计入 hit/miss（否则会把一次真实 embed 算两遍、污染命中率）。
                log.info(f"Embedding 等待进行中的计算: {query[:30]}...")  # 提高日志级别以便观察
                future = self._pending[cache_key]
                # 释放锁后等待结果
            else:
                self.miss_count += 1
                cache_requests_total.labels("embedding", "miss").inc()
                log.info(f"Embedding 缓存未命中，新计算 (hit={self.hit_count}, miss={self.miss_count}): {query[:30]}...")
                future = asyncio.Future()
                self._pending[cache_key] = future
                # 启动后台计算任务（不阻塞锁）
                asyncio.create_task(self._compute_and_cache(cache_key, query, future))

        # 在锁外等待结果，避免死锁
        return await future

    async def _compute_and_cache(self, cache_key: str, query: str, future: asyncio.Future):
        """实际计算 embedding 并缓存结果"""
        try:
            loop = asyncio.get_running_loop()
            func = partial(self.embedding_model.embed_query, query)
            embedding = await loop.run_in_executor(None, func)
            async with self._lock:
                self._cache[cache_key] = embedding
                future.set_result(embedding)
        except Exception as e:
            future.set_exception(e)
        finally:
            async with self._lock:
                self._pending.pop(cache_key, None)

    async def embed_batch(self, queries: List[str]) -> List[List[float]]:
        tasks = [self.embed(q) for q in queries]
        return await asyncio.gather(*tasks)

