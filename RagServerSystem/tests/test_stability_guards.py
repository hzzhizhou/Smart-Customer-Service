"""D 批稳定性回归：限流不可绕过 + 请求体上限（对应 PROJECT_AUDIT P0-6 / §2.6）。

锁住四件事：
  1) 限流身份不再取客户端可伪造的 X-API-KEY 或未验证 Bearer —— 换头/换 token 不再换桶
     （此前换一个随机头就得到一个新桶，等于没有限流）；
  2) 同一 IP 上的请求共用配额，不同 IP 各自计量；
  3) 超过请求体上限直接 413，不进入业务路由；
  4) /health、/metrics 放行，不被限流影响（否则告警链路会把自己掐断）。
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from access.middleware import RateLimiter, make_limit_key, register_middleware


def _request(*, ip: str = "10.0.0.1", headers: dict | None = None) -> Request:
    """构造一个最小 Request，只为验证限流身份推导。"""
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http", "method": "GET", "path": "/", "headers": raw,
        "client": (ip, 12345), "server": ("testserver", 80), "query_string": b"",
    }
    return Request(scope)


def _app(per_minute: int = 3, max_body_size: int = 64) -> FastAPI:
    app = FastAPI()
    register_middleware(app, RateLimiter(per_minute=per_minute),
                        max_body_size=max_body_size)

    @app.post("/echo")
    async def echo():
        return {"ok": True}

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app


# ===================== 1) 限流身份推导 =====================
class TestLimitKey:
    def test_x_api_key_no_longer_identifies_client(self):
        # 同 IP、每次换一个 X-API-KEY，身份必须完全一致（否则可无限绕过）
        keys = {
            make_limit_key(_request(headers={"X-API-KEY": f"rotated-{i}"}))
            for i in range(5)
        }
        assert len(keys) == 1

    def test_same_ip_different_token_does_not_change_identity(self):
        # 中间件在鉴权之前执行，Bearer 只是未验证的字符串：不能按它分桶，
        # 否则每次换一个随机 token 就能绕过限流。
        a = make_limit_key(_request(headers={"Authorization": "Bearer token-a"}))
        b = make_limit_key(_request(headers={"Authorization": "Bearer token-b"}))
        assert a == b

    def test_same_token_different_ip_gets_separate_bucket(self):
        a = make_limit_key(_request(ip="10.0.0.1",
                                    headers={"Authorization": "Bearer token-a"}))
        b = make_limit_key(_request(ip="10.0.0.2",
                                    headers={"Authorization": "Bearer token-a"}))
        assert a != b

    def test_anonymous_request_keyed_by_ip(self):
        assert make_limit_key(_request()) == make_limit_key(_request())
        assert "10.0.0.1" in make_limit_key(_request())


# ===================== 2) 中间件行为 =====================
class TestRateLimitMiddleware:
    def test_rotating_x_api_key_does_not_bypass(self):
        client = TestClient(_app(per_minute=3))
        codes = [
            client.post("/echo", headers={"X-API-KEY": f"rotated-{i}"}).status_code
            for i in range(4)
        ]
        assert codes == [200, 200, 200, 429]

    def test_rotating_bearer_token_does_not_bypass(self):
        client = TestClient(_app(per_minute=1))
        first = client.post("/echo", headers={"Authorization": "Bearer token-a"})
        second = client.post("/echo", headers={"Authorization": "Bearer token-b"})
        assert first.status_code == 200
        assert second.status_code == 429  # 换 token 不能换桶

    def test_health_is_exempt(self):
        client = TestClient(_app(per_minute=1))
        assert [client.get("/health").status_code for _ in range(3)] == [200, 200, 200]


class TestBodySizeLimit:
    def test_oversized_body_rejected_with_413(self):
        client = TestClient(_app(max_body_size=64))
        resp = client.post("/echo", content=b"x" * 200)
        assert resp.status_code == 413

    def test_within_limit_passes(self):
        client = TestClient(_app(max_body_size=64))
        assert client.post("/echo", content=b"x" * 32).status_code == 200
