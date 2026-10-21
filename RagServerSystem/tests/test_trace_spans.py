"""C2 轻量 Trace/Span 回归：contextvars 链路 + 结构化记录 + 查询接口。

对应 PROJECT_AUDIT §7.2(7)/§7.3。锁住四件事：
  1) span() 记录 duration_ms/status，嵌套时 parent_id 指向外层（contextvars 链）；
  2) 异常 span 记 status=error 且原样抛出，span_id 上下文正确复位；
  3) 记录为固定字段的结构化条目，进内存环缓冲，可按 trace_id 取回；
  4) 中间件生成/透传 X-Trace-ID，并用根 span 包住请求。

约定：与项目其它测试一致；span 缓冲为进程内共享，用例开头先 clear_spans 隔离。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from access.app import register_observability_routes
from access.middleware import register_request_context_middleware
from access.routes.auth import require_admin
from utils.request_context import (
    clear_request_id,
    clear_spans,
    clear_trace_id,
    get_request_id,
    get_span_id,
    get_trace_id,
    get_trace_spans,
    list_spans,
    normalize_trace_id,
    set_request_id,
    set_trace_id,
    span,
)

_ENTRY_KEYS = {"trace_id", "span_id", "parent_id", "name", "kind",
               "duration_ms", "status", "error"}


# ===================== 1) span 记录 =====================
class TestSpanRecord:
    def test_ok_span_fields(self):
        clear_spans()
        set_trace_id("t-1")
        with span("retrieve", kind="internal") as sp:
            pass
        spans = get_trace_spans("t-1")
        assert len(spans) == 1
        s = spans[0]
        assert set(s.keys()) == _ENTRY_KEYS        # 结构化字段固定
        assert s["name"] == "retrieve" and s["kind"] == "internal"
        assert s["status"] == "ok" and s["error"] == ""
        assert s["parent_id"] == ""                # 顶层 span 无父
        assert s["duration_ms"] >= 0
        assert sp.duration_ms == s["duration_ms"]

    def test_span_id_reset_after_exit(self):
        clear_spans()
        set_trace_id("t-2")
        with span("outer"):
            assert get_span_id() != ""
        assert get_span_id() == ""                 # 退出后上下文复位

    def test_nested_parent_chain(self):
        clear_spans()
        set_trace_id("t-3")
        with span("outer") as outer:
            assert get_span_id() == outer.span_id
            with span("inner") as inner:
                assert get_span_id() == inner.span_id
            assert get_span_id() == outer.span_id   # 内层退出恢复外层
        spans = {s["name"]: s for s in get_trace_spans("t-3")}
        assert spans["inner"]["parent_id"] == outer.span_id
        assert spans["outer"]["parent_id"] == ""

    def test_error_span_recorded_and_reraised(self):
        clear_spans()
        set_trace_id("t-4")
        with pytest.raises(ValueError):
            with span("boom"):
                raise ValueError("单元异常")
        s = get_trace_spans("t-4")[0]
        assert s["status"] == "error"
        assert "ValueError" in s["error"]
        assert get_span_id() == ""                 # 异常路径也要复位

    def test_falls_back_to_request_id(self):
        clear_spans()
        clear_trace_id()
        set_request_id("rq-9")
        try:
            with span("fallback"):
                pass
        finally:
            clear_request_id()
        assert get_trace_spans("rq-9")[0]["name"] == "fallback"


# ===================== 2) trace_id 清洗 =====================
class TestNormalizeTraceId:
    def test_keeps_valid(self):
        assert normalize_trace_id("ext-trace-1", "rq") == "ext-trace-1"

    def test_falls_back_when_empty(self):
        assert normalize_trace_id("", "rq-1") == "rq-1"
        assert normalize_trace_id(None, "rq-1") == "rq-1"

    def test_truncates_oversized(self):
        assert len(normalize_trace_id("x" * 500, "rq")) == 64


# ===================== 3) 缓冲与查询 =====================
class TestSpanBuffer:
    def test_get_trace_spans_filters_by_trace(self):
        clear_spans()
        set_trace_id("tA")
        with span("a1"):
            pass
        set_trace_id("tB")
        with span("b1"):
            pass
        assert [s["name"] for s in get_trace_spans("tA")] == ["a1"]
        assert get_trace_spans("nope") == []

    def test_list_spans_returns_recent(self):
        clear_spans()
        set_trace_id("tC")
        for i in range(5):
            with span(f"s{i}"):
                pass
        assert [s["name"] for s in list_spans(3)] == ["s2", "s3", "s4"]
        assert len(list_spans()) == 5

    def test_returns_copies(self):
        clear_spans()
        set_trace_id("tD")
        with span("c"):
            pass
        got = get_trace_spans("tD")[0]
        got["name"] = "tampered"                   # 改副本不应污染缓冲
        assert get_trace_spans("tD")[0]["name"] == "c"


# ===================== 4) 中间件 =====================
def _app() -> FastAPI:
    app = FastAPI()
    register_request_context_middleware(app)

    @app.get("/echo")
    async def echo():
        return {"request_id": get_request_id(), "trace_id": get_trace_id()}

    return app


class TestTraceMiddleware:
    def test_generates_and_echoes_trace_id(self):
        clear_spans()
        resp = TestClient(_app()).get("/echo")
        tid = resp.headers["X-Trace-ID"]
        assert tid != ""
        assert resp.json()["trace_id"] == tid
        # 根 span 以同一 trace_id 落进缓冲，供查询接口按 trace 取回
        assert [s["name"] for s in get_trace_spans(tid)] == ["http_request"]
        assert get_trace_spans(tid)[0]["status"] == "ok"

    def test_passthrough_external_trace_id(self):
        clear_spans()
        resp = TestClient(_app()).get("/echo", headers={"X-Trace-ID": "ext-trace-9"})
        assert resp.headers["X-Trace-ID"] == "ext-trace-9"
        assert resp.json()["trace_id"] == "ext-trace-9"
        assert get_trace_spans("ext-trace-9")[0]["name"] == "http_request"

    def test_consecutive_requests_distinct_trace(self):
        clear_spans()
        client = TestClient(_app())
        first = client.get("/echo").headers["X-Trace-ID"]
        second = client.get("/echo").headers["X-Trace-ID"]
        assert first != second


# ===================== 5) 查询路由 =====================
class TestTraceRoute:
    def test_returns_spans_of_trace(self):
        clear_spans()
        set_trace_id("t-route")
        with span("step"):
            pass
        app = FastAPI()
        register_observability_routes(app)
        app.dependency_overrides[require_admin] = lambda: {"role": "admin"}

        resp = TestClient(app).get("/observability/traces/t-route")

        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1 and body["spans"][0]["name"] == "step"

    def test_requires_admin(self):
        app = FastAPI()
        register_observability_routes(app)
        resp = TestClient(app).get("/observability/traces/t-route")
        assert resp.status_code in (401, 403)
