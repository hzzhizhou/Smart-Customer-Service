"""请求上下文：request_id 全链路贯穿。

用 contextvars 保存当前请求的 request_id，日志 Filter 每条日志自动注入该值，
不改任何业务代码的 log 调用点即可实现「一个请求一条链」：
grep 同一个 request_id 可跨模块日志文件（retrieval/generation/chat_history…）串起完整链路。

为什么用 contextvars 而不是全局变量：
- 单进程 uvicorn 并发处理多个请求，全局变量会互相覆盖；
- contextvars 是协程级隔离的，asyncio 任务创建时会快照当前上下文，
  因此中间件里 set 一次，该请求内的所有 await 链路（含 create_task 衍生的
  历史落库任务、SSE 流式生成器）都能读到同一个 request_id。
"""
import contextvars
import uuid

# default=""：请求外（启动脚本、离线评估等场景）读到空串，日志 Filter 会降级填 "-"
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# 外部传入的 request_id 上限长度（防止恶意超长 header 撑爆日志行）
MAX_REQUEST_ID_LEN = 64


def new_request_id() -> str:
    """生成一个新的 request_id（32 位十六进制，与工单 owner_id 同风格）。"""
    return uuid.uuid4().hex


def normalize_request_id(raw: str) -> str:
    """清洗外部传入的 X-Request-ID：去空白、限长；无效则生成新的。"""
    cleaned = (raw or "").strip()[:MAX_REQUEST_ID_LEN]
    return cleaned or new_request_id()


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
