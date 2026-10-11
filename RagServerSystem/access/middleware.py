"""HTTP 中间件：request_id 贯穿 + 滑动窗口限流 + Prometheus 请求指标。

单进程实现：限流桶按 key（X-API-KEY 或客户端 IP）各自维护最近一分钟的时间戳队列，
多 worker 部署时需换成 Redis 计数才能全局生效。
"""
import time
from collections import defaultdict, deque
from typing import Dict

from fastapi import Request
from fastapi.responses import JSONResponse

from config.settings import RATE_LIMIT_PER_MINUTE
from utils.metrics import http_requests_total, http_request_duration_seconds
from utils.request_context import clear_request_id, normalize_request_id, set_request_id

# 监控端点放行：健康检查与指标抓取被限流会直接影响告警，不能计入配额
_PASS_PATHS = ("/health", "/metrics")


class RateLimiter:
    """按 key 的滑动窗口限流器（1 分钟窗口，内存字典）。"""

    def __init__(self, per_minute: int = RATE_LIMIT_PER_MINUTE):
        self.per_minute = per_minute
        self._bucket: Dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        """返回 True 表示放行，False 表示超限。"""
        now = time.time()
        bucket = self._bucket[key]
        # 清理一分钟前的记录
        while bucket and bucket[0] < now - 60:
            bucket.popleft()
        if len(bucket) >= self.per_minute:
            return False
        bucket.append(now)
        return True


def register_middleware(app, limiter: RateLimiter) -> None:
    """注册限流 + 请求计速中间件。"""

    @app.middleware("http")
    async def rate_limit_and_metrics(request: Request, call_next):
        # 健康检查与指标端点放行
        if request.url.path in _PASS_PATHS:
            return await call_next(request)
        # 限流 key：优先 X-API-KEY header，降级用客户端 IP（不读 body 以免破坏下游路由）
        limit_key = request.headers.get("X-API-KEY", "") or request.client.host
        if not limiter.allow(limit_key):
            return JSONResponse(
                {"detail": f"请求过于频繁，限流 {limiter.per_minute}/分钟"},
                status_code=429,
            )
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        # Prometheus 指标
        http_requests_total.labels(
            request.method, request.url.path, str(response.status_code)
        ).inc()
        http_request_duration_seconds.labels(
            request.method, request.url.path
        ).observe(elapsed)
        return response


def register_request_context_middleware(app) -> None:
    """注册 request_id 贯穿中间件。

    - 入口处确定 request_id：优先透传外部 X-Request-ID（便于网关/前端关联），
      无效则生成 32 位十六进制新值；写入 contextvars，该请求内的所有
      await 链路（含 SSE 生成器、create_task 衍生的历史落库任务）自动继承；
    - 结束后在响应头回写 X-Request-ID，客户端可据其报障，服务端 grep 即得全链路日志；
    - finally 中清理上下文，防止协程上下文泄漏到请求外的任务。

    注册顺序说明：Starlette 后注册的中间件在更外层、先执行。
    本中间件必须在 register_middleware（限流）之后注册，保证 429 限流响应也带 request_id。
    """

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = normalize_request_id(request.headers.get("X-Request-ID", ""))
        set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            clear_request_id()
        response.headers["X-Request-ID"] = request_id
        return response