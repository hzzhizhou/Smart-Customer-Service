"""请求上下文：request_id 全链路贯穿 + 轻量 Trace/Span。

用 contextvars 保存当前请求的 request_id，日志 Filter 每条日志自动注入该值，
不改任何业务代码的 log 调用点即可实现「一个请求一条链」：
grep 同一个 request_id 可跨模块日志文件（retrieval/generation/chat_history…）串起完整链路。

为什么用 contextvars 而不是全局变量：
- 单进程 uvicorn 并发处理多个请求，全局变量会互相覆盖；
- contextvars 是协程级隔离的，asyncio 任务创建时会快照当前上下文，
  因此中间件里 set 一次，该请求内的所有 await 链路（含 create_task 衍生的
  历史落库任务、SSE 流式生成器）都能读到同一个 request_id。

Trace/Span（PROJECT_AUDIT §7.2(7)/§7.3）：一个 trace_id + 若干 span 组成一次请求的调用链。
- trace_id 由中间件按 X-Trace-ID 透传或回退到 request_id 生成，贯穿整条链；
- span() 上下文管理器产出 span_id/parent_id（按 contextvars 嵌套取父），
  退出时记录 duration_ms/status，写一条结构化 JSON 日志并进内存环形缓冲，
  供查询接口按 trace_id 取回。刻意不接 OpenTelemetry：单进程演示定位「刚发生的
  慢/失败链路」用不着整套 SDK，内存环缓冲 + JSON 日志已够用。
"""
import contextvars
import json
import re
import time
import uuid
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List

# default=""：请求外（启动脚本、离线评估等场景）读到空串，日志 Filter 会降级填 "-"
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# 外部传入的 request_id 上限长度（防止恶意超长 header 撑爆日志行）
MAX_REQUEST_ID_LEN = 64
# 只允许 URL 安全字符：外部 ID 会回写响应头，换行/控制字符可造成 header 注入。
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def new_request_id() -> str:
    """生成一个新的 request_id（32 位十六进制，与工单 owner_id 同风格）。"""
    return uuid.uuid4().hex


def normalize_request_id(raw: str) -> str:
    """清洗外部传入的 X-Request-ID：限长、只接受 URL 安全字符；无效则生成新的。"""
    cleaned = (raw or "").strip()[:MAX_REQUEST_ID_LEN]
    return cleaned if _ID_RE.fullmatch(cleaned) else new_request_id()


def set_request_id(request_id: str) -> str:
    """设置当前上下文的 request_id（中间件入口调用一次）。"""
    _request_id.set(request_id)
    return request_id


def get_request_id() -> str:
    """读取当前上下文的 request_id（日志 Filter 调用，热路径保持零开销）。"""
    return _request_id.get()


def clear_request_id() -> None:
    """清空当前上下文的 request_id（中间件 finally 中调用，防上下文泄漏）。"""
    _request_id.set("")


# ====================== Trace / Span ======================
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_span_id: contextvars.ContextVar[str] = contextvars.ContextVar("span_id", default="")

# 外部传入的 trace_id 上限长度
MAX_TRACE_ID_LEN = 64

# span 记录的内存环形缓冲：只留最近 N 条（非持久化，重启即清空）。
# 定位「刚发生的慢/失败链路」足够，且不会随运行时长无限增长。
_SPAN_BUFFER_MAX = 2000
_span_buffer: "deque[Dict[str, Any]]" = deque(maxlen=_SPAN_BUFFER_MAX)


def new_span_id() -> str:
    """生成一个新的 span_id（16 位十六进制，短且够区分同链内各段）。"""
    return uuid.uuid4().hex[:16]


def normalize_trace_id(raw: str, fallback: str = "") -> str:
    """清洗外部传入的 X-Trace-ID：限长、只接受 URL 安全字符；无效时回退 fallback。"""
    cleaned = (raw or "").strip()[:MAX_TRACE_ID_LEN]
    if cleaned and _ID_RE.fullmatch(cleaned):
        return cleaned
    safe_fallback = (fallback or "").strip()
    if safe_fallback and _ID_RE.fullmatch(safe_fallback):
        return safe_fallback
    return new_request_id()


def set_trace_id(trace_id: str) -> str:
    _trace_id.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    return _trace_id.get()


def clear_trace_id() -> None:
    _trace_id.set("")


def get_span_id() -> str:
    """读取当前 span_id（嵌套 span 的父 id 来源）。"""
    return _span_id.get()


class Span:
    """一个 span 的记录：span() 上下文管理器 yield 的对象，退出后 duration/status 可用。"""

    __slots__ = ("span_id", "parent_id", "name", "kind", "duration_ms", "status", "error")

    def __init__(self, span_id: str, parent_id: str, name: str, kind: str):
        self.span_id = span_id
        self.parent_id = parent_id
        self.name = name
        self.kind = kind
        self.duration_ms = 0.0
        self.status = "ok"
        self.error = ""


@contextmanager
def span(name: str, kind: str = "internal"):
    """记录一段执行的耗时与状态，并作为其内部嵌套 span 的父。

    用法::

        with span("retrieve") as sp:      # 退出后 sp.duration_ms / sp.status 可用
            ...

    退出时写一条结构化 JSON 日志（grep trace_id 即可串起一条链），同时进内存环形缓冲，
    供 get_trace_spans 查询。异常时记 status=error 并原样抛出，不吞异常。
    """
    trace_id = get_trace_id() or get_request_id()
    parent_id = get_span_id()
    sp = Span(new_span_id(), parent_id, name, kind)
    token = _span_id.set(sp.span_id)
    started = time.perf_counter()
    try:
        yield sp
    except Exception as e:                       # noqa: BLE001 —— 记录后原样抛出
        sp.status = "error"
        sp.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        sp.duration_ms = round(max(time.perf_counter() - started, 0.0) * 1000, 3)
        _span_id.reset(token)
        _record_span(trace_id, sp)


def _record_span(trace_id: str, sp: Span) -> None:
    """把 span 落进内存环缓冲 + 结构化 JSON 日志（延迟导入 log，避免与 log_config 循环依赖）。"""
    entry: Dict[str, Any] = {
        "trace_id": trace_id,
        "span_id": sp.span_id,
        "parent_id": sp.parent_id,
        "name": sp.name,
        "kind": sp.kind,
        "duration_ms": sp.duration_ms,
        "status": sp.status,
        "error": sp.error,
    }
    _span_buffer.append(entry)
    from logs.log_config import log          # 延迟导入：log_config 反向依赖本模块
    log.info("trace_span " + json.dumps(entry, ensure_ascii=False))


def get_trace_spans(trace_id: str) -> List[Dict[str, Any]]:
    """按 trace_id 取内存缓冲里的 span 列表（按记录先后顺序，同 trace 的各段成链）。"""
    if not trace_id:
        return []
    return [dict(s) for s in _span_buffer if s["trace_id"] == trace_id]


def list_spans(limit: int = 100) -> List[Dict[str, Any]]:
    """取最近若干条 span（跨 trace，按记录顺序），用于概览。"""
    items = list(_span_buffer)[-max(int(limit or 100), 1):]
    return [dict(s) for s in items]


def clear_spans() -> None:
    """清空 span 环缓冲（测试隔离用）。"""
    _span_buffer.clear()
