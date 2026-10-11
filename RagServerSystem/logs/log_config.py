"""
日志配置模块：全局统一日志配置（按模块拆分文件）
功能特性：
1. 分级输出（DEBUG/INFO/WARNING/ERROR/CRITICAL）
2. 同时输出到控制台和文件
3. 按日期滚动切割日志文件（保留7天）
4. 兼容Windows/Linux路径
5. 全局单例Logger，避免重复输出
6. 自动降级配置（无config模块时可用）
7. 支持异常栈完整打印
8. 按模块拆分日志文件（每个模块一个独立日志文件）
9. 结构化输出：文件侧 JSON（每行一条，便于检索），控制台保持纯文本便于开发阅读
10. request_id 贯穿：日志 Filter 自动注入当前请求的 request_id（见 utils/request_context.py），
    业务代码的 log 调用点无需任何改动；grep 同一 request_id 即可跨模块日志文件串起全链路
"""
import json
import logging
import os
import sys
import re
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, Dict
from concurrent_log_handler import ConcurrentRotatingFileHandler

from utils.request_context import get_request_id
# ===================== 基础配置常量 =====================
# 日志级别映射（兼容大小写）
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# 默认日志配置（降级使用）
DEFAULT_ROOT_LOG_DIR = Path("./logs")  # 根日志目录
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_ROOT_LOG_NAME = "Log_server"

# 日志格式（包含更多调试信息；request_id 由 RequestContextFilter 注入 record，请求外显示 "-"）
LOG_FORMAT = "%(asctime)s [%(request_id)s] %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 模块日志器缓存（避免重复创建）
MODULE_LOGGERS: Dict[str, logging.Logger] = {}


# ===================== request_id 注入与结构化格式 =====================
class RequestContextFilter(logging.Filter):
    """把当前请求的 request_id 注入每条日志记录（挂 handler 上，对该 handler 的所有日志生效）。

    请求外（启动/离线任务）读到空串，降级填 "-"，保证格式串不缺字段。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class JSONFormatter(logging.Formatter):
    """文件侧结构化输出：每行一个 JSON 对象（JSON Lines），字段固定便于检索。

    - 中文不转义（ensure_ascii=False），保持日志可读；
    - exc_info 时附完整异常栈字符串，异常排查仍在一行内完成。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "location": f"{record.filename}:{record.lineno}",
            "func": record.funcName,
            "request_id": getattr(record, "request_id", "-"),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

# ===================== 核心日志配置函数 =====================
def setup_module_logger(
    module_name: str,
    root_log_dir: Optional[Path] = None,
    log_level: str = DEFAULT_LOG_LEVEL,
    console_output: bool = True,
    file_output: bool = True,
    backup_days: int = 7,
    json_file_output: bool = True
) -> logging.Logger:
    """
    初始化/获取模块专属日志器（单例模式，按模块拆分文件）
    :param module_name: 模块名称（如data_layer/retrieval_layer/generation_layer）
    :param root_log_dir: 日志根目录
    :param log_level: 日志级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）
    :param console_output: 是否输出到控制台
    :param file_output: 是否输出到文件
    :param backup_days: 日志保留天数
    :param json_file_output: 文件侧是否用 JSON 结构化格式（False 则沿用纯文本格式）
    :return: 配置好的模块专属Logger对象
    """
    # 1. 拼接完整日志器名称（如 rag_server.data_layer）
    full_logger_name = f"{DEFAULT_ROOT_LOG_NAME}.{module_name}"

    # 检查缓存，避免重复创建
    if full_logger_name in MODULE_LOGGERS:
        return MODULE_LOGGERS[full_logger_name]

    # 2. 创建/获取日志器
    logger = logging.getLogger(full_logger_name)
    logger.setLevel(LOG_LEVEL_MAP.get(log_level.upper(), logging.INFO))
    logger.propagate = False  # 禁用向上传播，避免根日志器重复输出

    # request_id 注入过滤器：挂在 handler 上（handler 级过滤先于格式化，
    # 保证该 handler 的每条日志都带 request_id 字段；logger 级过滤会被 handler 级绕过）
    request_filter = RequestContextFilter()

    # 3. 定义格式器：控制台纯文本（含 request_id）；文件侧按开关用 JSON
    console_formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )

    # 4. 控制台处理器（所有模块共享控制台输出，兼容Windows/Linux编码）
    if console_output and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(LOG_LEVEL_MAP.get(log_level.upper(), logging.INFO))
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(request_filter)
        # 修复Windows控制台中文乱码问题
        if sys.platform == "win32":
            console_handler.stream.reconfigure(encoding="utf-8")
        logger.addHandler(console_handler)

    # 5. 文件处理器（按模块拆分文件，按日期滚动；JSON 结构化输出）
    if file_output and not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        # 日志根目录（优先配置，否则默认）
        target_root_dir = root_log_dir or DEFAULT_ROOT_LOG_DIR
        # 模块专属日志目录（logs/rag_server/data_layer/）
        module_log_dir = target_root_dir / DEFAULT_ROOT_LOG_NAME / module_name
        # 确保目录存在（兼容跨平台权限）
        module_log_dir.mkdir(parents=True, exist_ok=True)

        # 模块专属日志文件路径（如 logs/rag_server/data_layer/data_layer.log）
        log_file_path = module_log_dir / f"{module_name}.log"

        # 按天滚动的文件处理器（每天午夜切割一次，按日期轮转）
        # 修复BUG：此前误用 ConcurrentRotatingFileHandler（仅按 maxBytes=100MB 大小轮转，
        #          suffix/extMatch 对它是无效的），导致日志全部堆在同一文件。
        # 说明：ConcurrentRotatingFileHandler 不支持按天切割；本项目为单进程 uvicorn，
        #       改用 TimedRotatingFileHandler 即可安全按天切分 + 按 backup_days 清理。
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file_path),
            when="midnight",              # 每天午夜切割
            interval=1,
            backupCount=backup_days,      # 保留天数
            encoding="utf-8",             # 强制UTF-8编码
            utc=False                     # 使用本地时间
        )
        # 时间轮转：切割后日志文件名自动加日期后缀（如 data_layer.log.2026-09-07）
        file_handler.suffix = "%Y-%m-%d"
        # 修复日志切割后缀匹配问题（后缀含前导点，如 ".2026-09-07"）
        file_handler.extMatch = re.compile(r"^\.\d{4}-\d{2}-\d{2}$")
        file_handler.setLevel(LOG_LEVEL_MAP.get(log_level.upper(), logging.INFO))
        file_handler.setFormatter(
            JSONFormatter(datefmt=LOG_DATE_FORMAT) if json_file_output else console_formatter
        )
        file_handler.addFilter(request_filter)
        logger.addHandler(file_handler)

    # 6. 加入缓存
    MODULE_LOGGERS[full_logger_name] = logger
    return logger

# ===================== 全局日志器快捷函数 =====================
def get_logger(module_name: str) -> logging.Logger:
    """
    获取模块专属日志器（最简调用方式）
    :param module_name: 模块名称（如data_layer/retrieval_layer/generation_layer/chat_history）
    :return: 配置好的模块专属Logger对象
    """
    try:
        # 优先从配置文件读取配置
        import config.settings as config
        return setup_module_logger(
            module_name=module_name,
            root_log_dir=getattr(config, "LOG_DIR", DEFAULT_ROOT_LOG_DIR),
            log_level=getattr(config, "LOG_LEVEL", DEFAULT_LOG_LEVEL),
            backup_days=getattr(config, "LOG_BACKUP_DAYS", 7),
            json_file_output=getattr(config, "LOG_FORMAT_JSON", True)
        )
    except (ImportError, AttributeError):
        # 降级使用默认配置（无config模块/配置项缺失时）
        return setup_module_logger(
            module_name=module_name,
            root_log_dir=DEFAULT_ROOT_LOG_DIR,
            log_level=DEFAULT_LOG_LEVEL,
            backup_days=7,
            json_file_output=True
        )

# ===================== 全局默认日志器（兼容原有调用） =====================
# 全局默认日志器（无模块名称时使用）
log = get_logger("default")
data_layer_log = get_logger("data_layer")
retrieval_layer_log = get_logger("retrieval_layer")
generation_layer_log = get_logger("generation_layer")
chat_history_log = get_logger("chat_history_layer")
evaluation_layer_log  = get_logger("evalution_layer")
# ===================== 工具函数（可选） =====================
def log_exception(module_name: str, exc: Exception, level: str = "ERROR") -> None:
    """
    打印模块专属的异常栈信息
    :param module_name: 模块名称
    :param exc: 异常对象
    :param level: 日志级别
    """
    logger = get_logger(module_name)
    log_func = getattr(logger, level.lower(), logger.error)
    log_func(f"异常信息：{str(exc)}", exc_info=True)