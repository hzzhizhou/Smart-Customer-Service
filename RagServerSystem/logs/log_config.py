"""
日志配置模块（结构化重构版）

设计目标：一条日志 = 一个可查询的事件，而不是一句给人看的话。

三层输出，各司其职：
1. 模块文件  logs/Log_server/<module>/<module>.log  —— 按模块分，JSON Lines
2. 全链路文件 logs/Log_server/trace.log              —— 跨模块合流，一个 trace_id 一条链
3. 控制台                                            —— 人眼可读的单行彩色输出

与旧版的关键差异（都是为了「可查询」）：
- trace_id / span_id / parent_id 从 contextvars 直接注入顶层字段，
  不再塞进 message 的嵌套 JSON 里（旧版 log.info("trace_span " + json.dumps(entry))）；
- 业务数据一律走 extra，Formatter 把 extra 原样上浮成顶层字段，
  于是 event="retrieval.done" + 按 dur_ms 聚合这类统计才成立；
- 时间戳带毫秒与时区，和 span 的 duration_ms 可对齐；
- event 是显式语义（缺省回退到函数名），message 只承载人话描述。

向后兼容：setup_module_logger / get_logger / log / *_log 全部保留，
老的 log.info("...") 调用点不改也能跑（文本进 message 字段）。

用法对比：
    旧：log.info(f"检索完成，返回文档数={n},耗时={t:.2f}秒")
    新：log.info("检索完成", extra={
            "event": "retrieval.done", "hits": n, "dur_ms": round(t * 1000, 1)})
"""
import json
import logging
import os
import re
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

from utils.request_context import get_request_id, get_span_id, get_trace_id

# ===================== 基础配置常量 =====================
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

DEFAULT_ROOT_LOG_DIR = Path(os.getenv("RAG_LOG_DIR", "./logs"))
DEFAULT_LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO")
DEFAULT_ROOT_LOG_NAME = "Log_server"

LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 项目根目录：把异常栈里的绝对路径压成相对路径（避免泄漏部署目录）
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_PATH_RE = re.compile(re.escape(_PROJECT_ROOT) + r"[\\/]?")

# LogRecord 内置属性：这些不进 extra 区，其余全部上浮为业务字段
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime", "request_id", "trace_id", "span_id", "parent_id", "event",
}

# 控制台字段渲染顺序：耗时排最前，人眼最先要看的就是它
_FIELD_ORDER = ("dur_ms", "ttft_ms", "status", "tool", "intent", "branch",
                "hits", "k", "top_score", "cost", "tokens_in", "tokens_out")

MODULE_LOGGERS: Dict[str, logging.Logger] = {}


# ===================== 上下文注入 =====================
class ContextFilter(logging.Filter):
    """把当前请求的 trace_id / span_id / request_id 注入 record。

    必须挂在 handler 上而不是 logger 上：handler 级过滤先于格式化执行，
    而 logger 级 Filter 对已被 handler 直接接收的记录不生效（历史踩坑）。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        # 显式传入的 trace_id/span_id 优先（span.done 事件要标自己而非父 span）
        if not hasattr(record, "trace_id"):
            record.trace_id = get_trace_id() or record.request_id
        if not hasattr(record, "span_id"):
            record.span_id = get_span_id() or "-"
        if not hasattr(record, "parent_id"):
            record.parent_id = "-"
        return True


# ===================== 通用工具 =====================
def _iso_ts(record: logging.LogRecord) -> str:
    """ISO8601 + 毫秒 + 本地时区偏移（与 span 的 duration_ms 可对齐）。"""
    dt = datetime.fromtimestamp(record.created).astimezone()
    off = dt.strftime("%z") or "+0000"
    return (dt.strftime("%Y-%m-%dT%H:%M:%S.")
            + f"{int(record.msecs):03d}"
            + f"{off[:-2]}:{off[-2:]}")


def _extra_fields(record: logging.LogRecord) -> Dict[str, Any]:
    """收集 extra 里上浮的业务字段（排除 logging 内置与下划线私有键）。"""
    return {k: v for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_")}


def _scrub(text: str) -> str:
    """异常栈里的项目绝对路径换成相对路径（Windows 反斜杠一并规范）。"""
    return _PATH_RE.sub("", text or "").replace("\\", "/")


# ===================== 格式化器 =====================
class JSONFormatter(logging.Formatter):
    """文件侧：JSON Lines，一行一条，字段固定 + 业务字段上浮。

    字段顺序刻意稳定（ts/level/event/trace/... → extra → message），
    因为人眼扫日志是按位置扫的，顺序一变就要重新读。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": _iso_ts(record),
            "level": record.levelname,
            "event": getattr(record, "event", None) or record.funcName,
            "trace_id": getattr(record, "trace_id", "-"),
            "span_id": getattr(record, "span_id", "-"),
            "parent_id": getattr(record, "parent_id", "-"),
            "svc": record.name.split(".")[-1],
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "loc": f"{record.filename}:{record.lineno}",
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exc"] = _scrub(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    """控制台：人眼可读的单行，固定列宽，事件名在前，字段紧凑平铺。

    形如：
        23:14:07.382  INFO   [8f3a5c1e]  retrieval.done   dur=200.4ms hits=8 top_score=0.836
    没有 extra 的老式调用直接显示 message，不会因为改造而看不到内容。
    """

    COLORS = {
        "DEBUG": "\033[38;5;244m", "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m", "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[1;38;5;196m",
    }
    RESET = "\033[0m"
    DIM = "\033[38;5;244m"

    def __init__(self, color: bool = True, datefmt: str = "%H:%M:%S"):
        super().__init__(datefmt=datefmt)
        self.color = color

    @staticmethod
    def _render_value(key: str, value: Any) -> str:
        if key.endswith("_ms") and isinstance(value, (int, float)):
            return f"{key[:-3]}={round(float(value), 1)}ms"
        if isinstance(value, float):
            return f"{key}={round(value, 3)}"
        if isinstance(value, (dict, list)):
            try:
                s = json.dumps(value, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                s = str(value)
            return f"{key}={s if len(s) <= 48 else s[:48] + '…'}"
        s = str(value)
        return f"{key}={s if len(s) <= 60 else s[:60] + '…'}"

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created)
        ts = dt.strftime("%H:%M:%S.") + f"{int(record.msecs):03d}"
        extra = _extra_fields(record)
        event = getattr(record, "event", None) or ""
        trace = (getattr(record, "trace_id", "-") or "-")[:8]
        level = record.levelname

        ordered = [k for k in _FIELD_ORDER if k in extra] + \
                  [k for k in extra if k not in _FIELD_ORDER]
        # 空值不上控制台（JSON 里保留，便于 schema 稳定），否则会看到 status=ok error=
        body = " ".join(self._render_value(k, extra[k]) for k in ordered
                        if extra[k] not in ("", None))
        message = record.getMessage()
        # message 已在字段里表达过就不再重复（如 span.done 的 op、tool 事件）
        dup = message in (event, extra.get("op"), extra.get("tool"))
        if message and not dup:
            body = f"{body}  {message}" if body else message

        event_col = f"{event:<24}" if event else ""
        if self.color:
            c = self.COLORS.get(level, "")
            line = (f"{self.DIM}{ts}{self.RESET} {c}{level:<7}{self.RESET} "
                    f"{self.DIM}[{trace}]{self.RESET} {event_col} {body}")
        else:
            line = f"{ts} {level:<7} [{trace}] {event_col} {body}"
        if record.exc_info:
            line += "\n" + _scrub(self.formatException(record.exc_info))
        return line.rstrip()

class ConsoleNoiseFilter(logging.Filter):
    """控制台降噪：span.done 是链路骨架，正常完成的只进 trace.log。

    否则一次请求十几行里有一半是 span 结束事件，人眼读不到「业务发生了什么」。
    只有慢于阈值或失败的 span 才上控制台 —— 那才是需要立刻看见的。
    """

    def __init__(self, slow_ms: float = 1000.0):
        super().__init__()
        self.slow_ms = slow_ms

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "event", "") != "span.done":
            return True
        if getattr(record, "status", "ok") != "ok":
            return True
        try:
            return float(getattr(record, "dur_ms", 0) or 0) >= self.slow_ms
        except (TypeError, ValueError):
            return True


# ===================== 处理器工厂 =====================
def _console_color_enabled() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    try:
        return bool(sys.stdout.isatty())
    except (AttributeError, ValueError):
        return False


def _build_file_handler(path: Path, formatter: logging.Formatter,
                        level: int, backup_days: int) -> TimedRotatingFileHandler:
    """按天滚动的文件处理器（跨平台 UTF-8）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=str(path), when="midnight", interval=1,
        backupCount=backup_days, encoding="utf-8", utc=False,
    )
    handler.suffix = "%Y-%m-%d"
    # 后缀含前导点（.2026-10-29），extMatch 必须匹配，否则历史文件不会被清理
    handler.extMatch = re.compile(r"^\.\d{4}-\d{2}-\d{2}$")
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(ContextFilter())
    return handler


def _get_app_root_logger(root_dir: Path, level: int, backup_days: int,
                        console_output: bool, json_file_output: bool) -> logging.Logger:
    """应用根日志器：承载「控制台」与「全链路 trace.log」两个共享出口。

    子 logger 一律 propagate=True 汇到这里，于是：
    - 每个模块只写自己的文件（不重复写）
    - trace.log 天然跨模块合流（一条命令按 trace_id 取回整条链）
    """
    root = logging.getLogger(DEFAULT_ROOT_LOG_NAME)
    root.setLevel(logging.DEBUG)      # 真正的级别过滤在子 logger，这里只做汇聚
    root.propagate = False            # 不冒泡到标准库 root，避免重复输出

    trace_path = Path(root_dir) / DEFAULT_ROOT_LOG_NAME / "trace.log"
    has_trace = any(isinstance(h, TimedRotatingFileHandler)
                    and Path(getattr(h, "baseFilename", "")).name == "trace.log"
                    for h in root.handlers)
    if not has_trace:
        formatter = JSONFormatter() if json_file_output else ConsoleFormatter(color=False)
        root.addHandler(_build_file_handler(trace_path, formatter, logging.DEBUG, backup_days))

    has_console = any(isinstance(h, logging.StreamHandler)
                      and not isinstance(h, logging.FileHandler) for h in root.handlers)
    if console_output and not has_console:
        try:
            sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台中文
        except (AttributeError, ValueError, OSError):
            pass
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(level)
        console.setFormatter(ConsoleFormatter(color=_console_color_enabled()))
        console.addFilter(ContextFilter())
        console.addFilter(ConsoleNoiseFilter())
        root.addHandler(console)

    return root


# ===================== 核心配置函数 =====================
def setup_module_logger(
    module_name: str,
    root_log_dir: Optional[Path] = None,
    log_level: str = DEFAULT_LOG_LEVEL,
    console_output: bool = True,
    file_output: bool = True,
    backup_days: int = 7,
    json_file_output: bool = True,
) -> logging.Logger:
    """初始化/获取模块日志器（单例）。落盘路径 logs/Log_server/<module>/<module>.log。"""
    full_logger_name = f"{DEFAULT_ROOT_LOG_NAME}.{module_name}"
    if full_logger_name in MODULE_LOGGERS:
        return MODULE_LOGGERS[full_logger_name]

    level = LOG_LEVEL_MAP.get(str(log_level).upper(), logging.INFO)
    target_root = Path(root_log_dir or DEFAULT_ROOT_LOG_DIR)

    logger = logging.getLogger(full_logger_name)
    logger.setLevel(level)
    logger.propagate = True       # 汇入应用根日志器（trace.log + 控制台）

    if file_output:
        log_file = target_root / DEFAULT_ROOT_LOG_NAME / module_name / f"{module_name}.log"
        already = any(isinstance(h, TimedRotatingFileHandler)
                      and Path(getattr(h, "baseFilename", "")).name == f"{module_name}.log"
                      for h in logger.handlers)
        if not already:
            formatter = JSONFormatter() if json_file_output else ConsoleFormatter(color=False)
            logger.addHandler(_build_file_handler(log_file, formatter, level, backup_days))

    # 根日志器负责控制台与全链路文件；即使本模块不落文件也要保证它被初始化
    _get_app_root_logger(target_root, level, backup_days, console_output, json_file_output)

    MODULE_LOGGERS[full_logger_name] = logger
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """获取模块专属日志器（读 config.settings，缺省降级到环境变量/默认值）。"""
    try:
        import config.settings as config
        return setup_module_logger(
            module_name=module_name,
            root_log_dir=getattr(config, "LOG_DIR", DEFAULT_ROOT_LOG_DIR),
            log_level=getattr(config, "LOG_LEVEL", DEFAULT_LOG_LEVEL),
            backup_days=getattr(config, "LOG_BACKUP_DAYS", 7),
            json_file_output=getattr(config, "LOG_FORMAT_JSON", True),
        )
    except (ImportError, AttributeError):
        return setup_module_logger(module_name=module_name, json_file_output=True)


# ===================== 结构化事件便捷入口 =====================
def log_event(event: str, level: str = "INFO", logger: Optional[logging.Logger] = None,
              message: str = "", **fields: Any) -> None:
    """打一条结构化事件，等价于 log.info(message or event, extra={"event": event, **fields})。

    单独提供这个入口，是为了让「事件名」始终显式，避免在两百多个调用点里漏写。
    """
    target = logger or log
    target.log(LOG_LEVEL_MAP.get(str(level).upper(), logging.INFO),
               message or event, extra={"event": event, **fields})


# ===================== 全局默认日志器 =====================
log = get_logger("default")
data_layer_log = get_logger("data_layer")
retrieval_layer_log = get_logger("retrieval_layer")
generation_layer_log = get_logger("generation_layer")
chat_history_log = get_logger("chat_history_layer")
agent_log = get_logger("agent")
evaluation_layer_log = get_logger("evaluation_layer")


def log_exception(module_name: str, exc: Exception, level: str = "ERROR") -> None:
    """打印模块专属异常栈（保留旧接口）。"""
    logger = get_logger(module_name)
    log_func = getattr(logger, level.lower(), logger.error)
    log_func(f"异常信息：{exc}", exc_info=True, extra={"event": "exception"})
