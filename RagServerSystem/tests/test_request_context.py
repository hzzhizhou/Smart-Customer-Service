"""request_id 全链路贯穿 + 结构化日志测试。

锁住四件事：
  1) 上下文读写：set/get/clear 正确，请求外读到空串；
  2) 日志注入：JSON 行含 request_id 字段（请求内为真实值、请求外为 "-"），
     异常栈随 exc_info 进入 JSON，控制台纯文本格式也能渲染 request_id；
  3) 中间件：生成并回写 X-Request-ID、透传外部传入值、相邻请求 id 互不串扰；
  4) 落盘集成：setup_module_logger 产生的文件日志是合法 JSON Lines。
"""
import json
import logging
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from access.middleware import register_request_context_middleware
from logs.log_config import (
    JSONFormatter,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    MODULE_LOGGERS,
    RequestContextFilter,
    setup_module_logger,
)
from utils.request_context import (
    clear_request_id,
    get_request_id,
    new_request_id,
    normalize_request_id,
    set_request_id,
)


def _make_record(msg: str = "hello", exc_info=None) -> logging.LogRecord:
    return logging.LogRecord(
        name="rag_server.unit_test", level=logging.INFO, pathname="x.py",
        lineno=1, msg=msg, args=(), exc_info=exc_info,
    )


def _format_with_filter(record: logging.LogRecord, formatter) -> str:
    """复刻 handler 的处理顺序：Filter 先注入 request_id，再格式化。"""
    RequestContextFilter().filter(record)
    return formatter.format(record)


# ===================== 1) 上下文读写 =====================
class TestRequestContext:
    def test_roundtrip(self):
        rid = new_request_id()
        set_request_id(rid)
        try:
            assert get_request_id() == rid
        finally:
            clear_request_id()
        assert get_request_id() == ""

    def test_new_id_is_32bit_hex(self):
        rid = new_request_id()
        assert len(rid) == 32
        int(rid, 16)  # 不是十六进制会抛 ValueError

    def test_normalize_keeps_valid_external_id(self):
        assert normalize_request_id("ext-req-001") == "ext-req-001"

    def test_normalize_generates_when_empty(self):
        assert normalize_request_id("") != ""
        assert normalize_request_id(None) != ""

    def test_normalize_truncates_oversized(self):
        assert len(normalize_request_id("x" * 500)) == 64


# ===================== 2) 日志注入与格式 =====================
class TestLoggingFormat:
    def test_json_line_contains_request_id(self):
        rid = set_request_id(new_request_id())
        try:
            line = _format_with_filter(
                _make_record("带请求的日志"), JSONFormatter(datefmt=LOG_DATE_FORMAT)
            )
        finally:
            clear_request_id()
        payload = json.loads(line)  # 不是合法 JSON 会直接失败
        assert payload["request_id"] == rid
        assert payload["message"] == "带请求的日志"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "rag_server.unit_test"
        assert payload["location"] == "x.py:1"

    def test_json_line_dash_outside_request(self):
        clear_request_id()
        line = _format_with_filter(
            _make_record("请求外的日志"), JSONFormatter(datefmt=LOG_DATE_FORMAT)
        )
        assert json.loads(line)["request_id"] == "-"

    def test_json_line_includes_exc_info(self):
        try:
            raise ValueError("模拟异常")
        except ValueError:
            import sys
            record = _make_record("出错了", exc_info=sys.exc_info())
        line = _format_with_filter(record, JSONFormatter(datefmt=LOG_DATE_FORMAT))
        payload = json.loads(line)
        assert "ValueError: 模拟异常" in payload["exc_info"]
        assert "\n" not in payload["exc_info"].splitlines()[0]  # 栈整体在单条 JSON 内

    def test_console_plain_format_renders_request_id(self):
        rid = set_request_id(new_request_id())
        try:
            text = _format_with_filter(_make_record("控制台"), logging.Formatter(LOG_FORMAT))
        finally:
            clear_request_id()
        assert f"[{rid}]" in text
        assert "控制台" in text


# ===================== 3) 中间件行为 =====================
def _app() -> FastAPI:
    app = FastAPI()
    register_request_context_middleware(app)

    @app.get("/echo")
    async def echo():
        """端点内读上下文，验证中间件 set 的值能传进业务代码。"""
        return {"request_id": get_request_id()}

    return app


class TestRequestContextMiddleware:
    def test_generates_and_echoes_header(self):
        resp = TestClient(_app()).get("/echo")
        rid = resp.headers["X-Request-ID"]
        assert rid != ""
        assert resp.json()["request_id"] == rid  # 中间件 set → 端点读到同一值

    def test_passthrough_external_header(self):
        resp = TestClient(_app()).get("/echo", headers={"X-Request-ID": "ext-req-001"})
        assert resp.headers["X-Request-ID"] == "ext-req-001"
        assert resp.json()["request_id"] == "ext-req-001"

    def test_consecutive_requests_get_distinct_ids(self):
        client = TestClient(_app())
        first = client.get("/echo").json()["request_id"]
        second = client.get("/echo").json()["request_id"]
        assert first != second  # finally 清理生效，下一请求重新生成

    def test_oversized_external_header_truncated(self):
        resp = TestClient(_app()).get("/echo", headers={"X-Request-ID": "y" * 500})
        assert len(resp.headers["X-Request-ID"]) == 64


# ===================== 4) 落盘集成 =====================
class TestModuleLoggerJSONFile:
    def test_file_output_is_json_lines(self, tmp_path):
        name = f"unit_json_{uuid.uuid4().hex[:8]}"  # 唯一名，避开 MODULE_LOGGERS 缓存
        try:
            logger = setup_module_logger(
                name, root_log_dir=tmp_path,
                console_output=False, file_output=True,
            )
            rid = set_request_id(new_request_id())
            logger.info("结构化落盘测试")
            clear_request_id()
            log_file = tmp_path / "Log_server" / name / f"{name}.log"
            lines = log_file.read_text(encoding="utf-8").strip().splitlines()
            payload = json.loads(lines[-1])
            assert payload["message"] == "结构化落盘测试"
            assert payload["request_id"] == rid
        finally:
            clear_request_id()
            MODULE_LOGGERS.pop(f"Log_server.{name}", None)

    def test_plain_file_output_when_disabled(self, tmp_path):
        name = f"unit_plain_{uuid.uuid4().hex[:8]}"
        try:
            logger = setup_module_logger(
                name, root_log_dir=tmp_path,
                console_output=False, file_output=True, json_file_output=False,
            )
            logger.info("纯文本落盘测试")
            log_file = tmp_path / "Log_server" / name / f"{name}.log"
            line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
            with pytest.raises(json.JSONDecodeError):  # 关闭开关时不应再是 JSON
                json.loads(line)
            assert "纯文本落盘测试" in line
        finally:
            MODULE_LOGGERS.pop(f"Log_server.{name}", None)
