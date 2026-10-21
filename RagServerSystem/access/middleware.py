"""HTTP 中间件：request_id 贯穿 + 滑动窗口限流 + 请求体上限 + Prometheus 请求指标。

单进程实现：限流桶按客户端 IP 维护最近一分钟的时间戳队列，多 worker 部署时需换成
Redis 计数才能全局生效。
"""
import time
from collections import defaultdict, deque
from typing import Dict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.routing import Match

from config.settings import MAX_REQUEST_BODY_SIZE, RATE_LIMIT_PER_MINUTE
from utils.metrics import http_requests_total, http_request_duration_seconds
from utils.request_context import (
    clear_request_id,
    clear_trace_id,
    normalize_request_id,
    normalize_trace_id,
    set_request_id,
    set_trace_id,
    span,
)

# 监控端点放行：健康检查与指标抓取被限流会直接影响告警，不能计入配额
_PASS_PATHS = ("/health", "/metrics")


def _resolve_endpoint(app, request: Request) -> str:
    """解析指标 endpoint 标签，尽量还原成路由模板。

    直接用 request.url.path 作标签会让 /items/{item_id} 这类带参路由每来一个 ID
    就多出一条时间序列（基数无界增长，PROJECT_AUDIT §7.2(1)）。正常请求在 call_next
    之后 scope["route"] 已就绪，取模板即可；限流/超限在路由匹配前提前返回，此时用
    路由表手动匹配还原模板；仍未命中（404 等）才退回原始 path。
    """
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    if path:
        return path
    for candidate in getattr(app, "routes", []):
        try:
            matched, child = candidate.matches(request.scope)
        except Exception:
            continue
        if matched == Match.FULL:
            inner = child.get("route")
            return getattr(inner, "path", None) or getattr(candidate, "path", None) \
                or request.url.path
    return request.url.path


def _record_request_metrics(app, request: Request, status: int, elapsed: float) -> None:
    """统一落 HTTP 指标：请求数按 (method, 路由模板, 状态) 计，耗时按 (method, 路由模板) 记。"""
    endpoint = _resolve_endpoint(app, request)
    http_requests_total.labels(request.method, endpoint, str(status)).inc()
    http_request_duration_seconds.labels(request.method, endpoint).observe(
        max(elapsed, 0.0))


def make_limit_key(request: Request) -> str:
    """推导限流身份：只认客户端 IP。

    为什么不再叠加 Authorization 令牌：中间件在鉴权之前执行，拿到的 Bearer 只是
    客户端可任意填写的字符串，并没有被验证。若按令牌哈希分桶，攻击者每次换一个
    随机 token 就得到一个新桶，等于没有限流（PROJECT_AUDIT P0-6）。这里只按 IP
    计量；登录用户的更细粒度配额应放在 require_user 之后按 user_id 做，不能把
    “未验证的 token”当身份。

    已知边界：同一 NAT/代理后的用户共享配额；攻击者轮换出口 IP 仍可绕过。
    生产环境应由网关做全局限流，并只信任可信代理写入的客户端 IP。
    """
    ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


class RateLimiter:
    """按 key 的滑动窗口限流器（1 分钟窗口，内存字典）。"""

    def __init__(self, per_minute: int = RATE_LIMIT_PER_MINUTE):
        self.per_minute = per_minute
        self._bucket: Dict[str, deque] = defaultdict(deque)
        self._calls = 0

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
        # 低频清理空桶：避免每个 IP/令牌留下一条永久 dict 项（内存泄漏）。
        # 每 1000 次调用扫一遍，成本可忽略；活跃桶的最后一个时间戳 >= cutoff 会保留。
        self._calls += 1
        if self._calls % 1000 == 0:
            cutoff = now - 60
            stale = [k for k, b in self._bucket.items() if not b or b[-1] < cutoff]
            for k in stale:
                self._bucket.pop(k, None)
        return True


def register_middleware(app, limiter: RateLimiter,
                        max_body_size: int = MAX_REQUEST_BODY_SIZE) -> None:
    """注册限流 + 请求体上限 + 请求计速中间件（max_body_size 可注入，便于单测）。"""

    @app.middleware("http")
    async def rate_limit_and_metrics(request: Request, call_next):
        # 健康检查与指标端点放行
        if request.url.path in _PASS_PATHS:
            return await call_next(request)
        start = time.time()
        # 请求体上限：带 Content-Length 的超限请求直接 413（不读 body、不进业务路由）。
        # 边界：chunked 传输不带 Content-Length，此处拦不住；上传路由内另有逐块读的硬上限。
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > max_body_size:
            response = JSONResponse(
                {"detail": f"请求体过大，上限 {max_body_size} 字节"},
                status_code=413,
            )
            # 被拒的请求同样计入指标，否则看板的错误率/流量会被系统性低估
            _record_request_metrics(app, request, 413, time.time() - start)
            return response
        if not limiter.allow(make_limit_key(request)):
            response = JSONResponse(
                {"detail": f"请求过于频繁，限流 {limiter.per_minute}/分钟"},
                status_code=429,
            )
            _record_request_metrics(app, request, 429, time.time() - start)
            return response
        try:
            response = await call_next(request)
        except Exception:
            # 未捕获异常由外层 ServerErrorMiddleware 转 500：这里先按 500 计数再抛出，
            # 保证异常请求也进错误率（此前直接漏掉，看板错误率偏低）
            _record_request_metrics(app, request, 500, time.time() - start)
            raise
        _record_request_metrics(app, request, response.status_code, time.time() - start)
        return response


def register_request_context_middleware(app) -> None:
    """注册 request_id + trace_id 贯穿中间件。

    - 入口处确定 request_id：优先透传外部 X-Request-ID（便于网关/前端关联），
      无效则生成 32 位十六进制新值；写入 contextvars，该请求内的所有
      await 链路（含 SSE 生成器、create_task 衍生的历史落库任务）自动继承；
    - 入口处确定 trace_id（§7.2(7)）：优先透传外部 X-Trace-ID，未带则回退到
      request_id，使「单请求」与「调用链」默认对齐、可互相定位；
    - 用根 span 包住 call_next：整条 HTTP 请求的耗时/状态进 trace 缓冲，
      该请求内后续 span() 也以它为父（contextvars 嵌套）；
    - 结束后在响应头回写 X-Request-ID / X-Trace-ID，客户端可据其报障，
      服务端 grep 即得全链路日志；
    - finally 中清理上下文，防止协程上下文泄漏到请求外的任务。

    注册顺序说明：Starlette 后注册的中间件在更外层、先执行。
    本中间件必须在 register_middleware（限流）之后注册，保证 429 限流响应也带 request_id。
    """

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = normalize_request_id(request.headers.get("X-Request-ID", ""))
        set_request_id(request_id)
        trace_id = normalize_trace_id(request.headers.get("X-Trace-ID", ""), request_id)
        set_trace_id(trace_id)
        try:
            # 根 span：request_id/trace_id 已就绪，其内嵌套 span 的 parent_id 指向它
            with span("http_request", kind="server"):
                response = await call_next(request)
        finally:
            clear_request_id()
            clear_trace_id()
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        return response
