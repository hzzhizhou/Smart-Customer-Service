"""RAG 响应缓存：进程内 LRU 淘汰 + TTL 过期。

只服务于「无会话」的同一问题秒回（命中即跳过检索与 LLM，实测 6s → 0.02s）。
多 worker 部署时需换成 Redis 才能跨进程共享，这里保持无依赖的内存实现。
"""
import hashlib
import threading
import time
from typing import Dict, Optional, Tuple

from config.settings import CACHE_MAXSIZE, CACHE_TTL
from utils.metrics import cache_requests_total


class ResponseCache:
    def __init__(self, maxsize: int = CACHE_MAXSIZE, ttl: float = CACHE_TTL):
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: Dict[str, Tuple[float, dict]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def make_key(question: str, route_mode: str) -> str:
        q_hash = hashlib.md5(question.encode()).hexdigest()
        return f"{q_hash}_{route_mode}"

    def set(self, key: str, result: dict) -> None:
        """写入缓存；超容量时按插入顺序淘汰最旧一条。"""
        with self._lock:
            if len(self._store) >= self._maxsize:
                oldest = next(iter(self._store.keys()))
                del self._store[oldest]
            self._store[key] = (time.time(), result)

    def get(self, key: str) -> Optional[dict]:
        """读缓存；过期即删除并返回 None。命中/未命中计入指标用于算命中率。

        过期视为未命中（它确实触发了一次真实检索），避免把「已失效但还没被淘汰」
        的条目算成命中、虚高命中率。
        """
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                cache_requests_total.labels("response", "miss").inc()
                return None
            ts, result = hit
            if time.time() - ts > self._ttl:
                del self._store[key]
                cache_requests_total.labels("response", "miss").inc()
                return None
            cache_requests_total.labels("response", "hit").inc()
            return result