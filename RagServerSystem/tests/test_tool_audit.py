"""C1 工具调用审计回归：agent_tool_calls 写入 + 查询接口 + 入参脱敏。

对应 PROJECT_AUDIT §7.2(5)/§7.4。锁住四件事：
  1) Agent 工具成功/失败各写一行审计：status/error_code/duration_ms/user/session 正确；
  2) 入参脱敏：口令类明文绝不入库，结果只存 sha256 摘要（不落原文）；
  3) list_tool_calls 的过滤/分页：条件为空不过滤，limit 被夹到 [1,500]；
  4) 查询路由只需管理员，按 query 参数透传给存储层。

约定：与项目其它测试一致，同步用例内用 asyncio.run(drive()) 驱动异步逻辑；
无 MySQL 环境，故用假 cursor/monkeypatch 替掉存储层真调用。
"""
import asyncio
import hashlib
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import service.turn.agent_turn as agent_turn
import service.turn.state as turn_state
from access.app import register_observability_routes
from access.routes.auth import require_admin
from infrastructure import mysql_store
import infrastructure.db.connection as db_connection


# ===================== 工具事件脚本 =====================
class _Chunk:
    def __init__(self, content):
        self.content = content


class _StubAgent:
    """按脚本吐 astream_events 事件，替代真实 ReAct Agent。"""

    def __init__(self, events):
        self._events = events

    async def astream_events(self, *args, **kwargs):
        for e in self._events:
            yield e


def _tool_start(run_id, name, args):
    return {"event": "on_tool_start", "name": name, "run_id": run_id,
            "data": {"input": args}}


def _tool_end(run_id, name, output):
    return {"event": "on_tool_end", "name": name, "run_id": run_id,
            "data": {"output": output}}


def _tool_error(run_id, name, err):
    return {"event": "on_tool_error", "name": name, "run_id": run_id,
            "data": {"error": err}}


def _final_answer(text="已为你处理"):
    return {"event": "on_chat_model_stream", "name": "ChatModel", "run_id": "m1",
            "data": {"chunk": _Chunk(text)}}


def _turn(question="帮我查订单", session_id="s1", user_id="u1"):
    return SimpleNamespace(
        ctx=SimpleNamespace(question=question, session_id=session_id,
                            user={"user_id": user_id}),
        messages=[("user", question)],
        chat_history=SimpleNamespace(),
    )


def _run_agent(monkeypatch, events):
    """跑一遍 _produce_agent，返回被捕获的审计写入参数列表。"""
    captured = []

    def _fake_record(**kw):
        captured.append(kw)

    monkeypatch.setattr(agent_turn.mysql_store, "record_tool_call", _fake_record)
    monkeypatch.setattr(agent_turn, "DST_ENABLED", False)   # 不触发后台 DST 更新
    svc = object.__new__(agent_turn.AgentTurnService)
    svc.unified_agent = _StubAgent(events)

    async def drive():
        queue = asyncio.Queue()
        final_text = []
        await svc._produce_agent(_turn(), queue, final_text)
        return final_text

    asyncio.run(drive())
    return captured


# ===================== 1) 成功/失败写入 =====================
class TestAuditWrite:
    def test_success_writes_one_row(self, monkeypatch):
        events = [
            _tool_start("r1", "query_order", {"order_id": "SO1", "password": "p@ss"}),
            _tool_end("r1", "query_order", {"status": "shipped"}),
            _final_answer(),
        ]
        captured = _run_agent(monkeypatch, events)

        assert len(captured) == 1
        row = captured[0]
        assert row["tool_name"] == "query_order"
        assert row["status"] == "success"
        assert row["error_code"] == ""
        assert isinstance(row["duration_ms"], int) and row["duration_ms"] >= 0
        assert row["user_id"] == "u1"
        assert row["session_id"] == "s1"

    def test_error_writes_row_with_error_code(self, monkeypatch):
        events = [
            _tool_start("r1", "create_ticket", {"description": "退货"}),
            _tool_error("r1", "create_ticket", ValueError("库存不足")),
            _final_answer(),
        ]
        captured = _run_agent(monkeypatch, events)

        assert len(captured) == 1
        assert captured[0]["status"] == "error"
        assert captured[0]["error_code"] == "ValueError"

    def test_error_without_start_still_audited(self, monkeypatch):
        # 工具未 start 就报错（如被拒绝）：duration 记 0，仍要留痕
        events = [_tool_error("r9", "query_order", PermissionError("越权")), _final_answer()]
        captured = _run_agent(monkeypatch, events)

        assert captured[0]["status"] == "error"
        assert captured[0]["duration_ms"] == 0


# ===================== 2) 脱敏与摘要 =====================
class TestRedactAndDigest:
    def test_masks_sensitive_keys_recursively(self):
        out = turn_state._redact_args({"a": {"secret": "s3cr3t", "b": 1}, "list": [{"token": "tk"}]})
        assert "s3cr3t" not in out and "tk" not in out
        assert out.count("***") == 2
        assert '"b": 1' in out

    def test_plain_args_kept(self):
        out = turn_state._redact_args({"order_id": "SO123"})
        assert "SO123" in out

    def test_masks_pii_in_text_values(self):
        # 审计库只应留脱敏后的描述，不能把手机号原样写进去
        out = turn_state._redact_args({"description": "联系我 13800138000 处理退款"})
        assert "13800138000" not in out
        assert "138****8000" in out

    def test_long_args_truncated(self):
        out = turn_state._redact_args({"x": "y" * 5000})
        assert len(out) == 1001 and out.endswith("…")

    def test_digest_is_sha256_of_summary(self):
        assert turn_state._result_digest("hi") == hashlib.sha256(b"hi").hexdigest()

    def test_digest_empty_for_none(self):
        assert turn_state._result_digest(None) == ""

    def test_success_row_masks_password(self, monkeypatch):
        events = [
            _tool_start("r1", "query_order", {"order_id": "SO1", "password": "p@ss"}),
            _tool_end("r1", "query_order", "done"),
            _final_answer(),
        ]
        row = _run_agent(monkeypatch, events)[0]
        assert "p@ss" not in row["args_json"]
        assert "***" in row["args_json"]
        assert len(row["result_digest"]) == 64


# ===================== 3) 存储层过滤/分页 =====================
class _FakeCursor:
    def __init__(self, sink):
        self._sink = sink

    def execute(self, sql, params=None):
        self._sink.append((sql, params))

    def fetchall(self):
        return []


def _patch_tx(monkeypatch):
    sink = []

    @contextmanager
    def _fake_tx():
        yield _FakeCursor(sink)

    monkeypatch.setattr(db_connection, "_tx", _fake_tx)
    return sink


class TestListToolCalls:
    def test_filters_bound_by_params(self, monkeypatch):
        sink = _patch_tx(monkeypatch)
        mysql_store.list_tool_calls(tool_name="query_order", user_id="u1", limit=50)
        sql, params = sink[0]
        assert "tool_name=%s" in sql and "user_id=%s" in sql
        assert sql.endswith("ORDER BY id DESC LIMIT %s")
        assert params == ("query_order", "u1", 50)

    def test_no_condition_has_no_where(self, monkeypatch):
        sink = _patch_tx(monkeypatch)
        mysql_store.list_tool_calls()
        sql, params = sink[0]
        assert "WHERE" not in sql
        assert params == (100,)          # 默认 limit

    def test_limit_clamped(self, monkeypatch):
        sink = _patch_tx(monkeypatch)
        mysql_store.list_tool_calls(limit=9999)
        assert sink[0][1] == (500,)
        mysql_store.list_tool_calls(limit=0)
        assert sink[1][1] == (100,)      # 0/None 回落到默认

    def test_trace_filter(self, monkeypatch):
        sink = _patch_tx(monkeypatch)
        mysql_store.list_tool_calls(trace_id="tr-1", status="error")
        sql, params = sink[0]
        assert "trace_id=%s" in sql and "status=%s" in sql
        assert params == ("tr-1", "error", 100)


class TestRecordToolCall:
    def test_insert_params_ordered(self, monkeypatch):
        sink = _patch_tx(monkeypatch)
        mysql_store.record_tool_call(
            trace_id="tr-1", request_id="rq-1", session_id="s1", user_id="u1",
            tool_name="query_order", args_json="{}", result_digest="abc",
            status="error", error_code="ValueError", duration_ms=5)
        sql, params = sink[0]
        assert sql.startswith("INSERT INTO agent_tool_calls")
        assert params[:10] == ("tr-1", "rq-1", "s1", "u1", "query_order",
                              "{}", "abc", "error", "ValueError", 5)

    def test_defaults_are_safe(self, monkeypatch):
        sink = _patch_tx(monkeypatch)
        mysql_store.record_tool_call(tool_name="t")
        _sql, params = sink[0]
        # 缺省 status 必须落 success，duration/error_code 落 0/''（NOT NULL 列不允许 None）
        assert params[7] == "success" and params[9] == 0 and params[8] == ""


# ===================== 4) 查询路由 =====================
def _route_app(monkeypatch, rows):
    seen = {}

    def _fake_list(tool_name="", user_id="", trace_id="", status="", limit=100):
        seen.update(tool_name=tool_name, user_id=user_id, trace_id=trace_id,
                    status=status, limit=limit)
        return rows

    monkeypatch.setattr(mysql_store, "list_tool_calls", _fake_list)
    app = FastAPI()
    register_observability_routes(app)
    app.dependency_overrides[require_admin] = lambda: {"user_id": "admin1", "role": "admin"}
    return app, seen


class TestToolCallsRoute:
    def test_forwards_filters_to_store(self, monkeypatch):
        app, seen = _route_app(monkeypatch, [{"id": 1, "tool_name": "query_order"}])
        resp = TestClient(app).get(
            "/observability/tool-calls",
            params={"tool_name": "query_order", "user_id": "u1", "limit": 20})
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert seen == {"tool_name": "query_order", "user_id": "u1",
                        "trace_id": "", "status": "", "limit": 20}

    def test_requires_admin(self, monkeypatch):
        app, _seen = _route_app(monkeypatch, [])
        # 去掉管理员覆盖：真实 require_admin 无令牌应 401/403
        app.dependency_overrides.clear()
        resp = TestClient(app).get("/observability/tool-calls")
        assert resp.status_code in (401, 403)
