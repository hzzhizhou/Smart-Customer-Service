"""轮次状态与审计辅助：TurnState 以及工具入参/结果的压缩、脱敏与摘要。

这些函数是 Agent 编排的纯工具（无 I/O、无模块状态），单独成文件便于测试与复用。
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from utils.security import DataSecurity
from service.handlers import TurnContext


def _summarize_value(value: Any, limit: int = 200) -> str:
    """把工具入参/输出压成单行摘要（观测用，防日志刷屏：检索文档全文不进日志）。

    字符串原样折叠空白；dict/list 走 JSON；不可序列化对象经 default=str 兜底；
    超长截断并加省略号标记。
    """
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = " ".join(text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


# 工具入参里需要脱敏的键名（小写比对）：审计库不该留存口令/令牌类明文
_SENSITIVE_ARG_KEYS = {
    "password", "passwd", "pwd", "token", "secret", "authorization",
    "api_key", "apikey", "access_token", "refresh_token",
}


def _redact_args(value: Any, limit: int = 1000) -> str:
    """工具入参脱敏后压成 JSON 串（审计落库用）。

    命中敏感键名的值统一替换为 "***"，嵌套 dict/list 递归处理；
    超长截断，避免一次调用把审计行撑大。
    """
    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: ("***" if str(k).lower() in _SENSITIVE_ARG_KEYS else _walk(v))
                    for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            return [_walk(v) for v in node]
        if isinstance(node, str):
            # 订单号/描述里可能夹带手机号、身份证、邮箱；审计库只留脱敏后的文本
            return DataSecurity.desensitize_text(node)
        return node

    try:
        text = json.dumps(_walk(value), ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text[:limit] + ("…" if len(text) > limit else "")


def _result_digest(value: Any) -> str:
    """工具结果的 sha256 摘要（只存哈希不存原文）：既防隐私外泄，也控审计表体积。"""
    if value is None:
        return ""
    return hashlib.sha256(_summarize_value(value).encode("utf-8")).hexdigest()


@dataclass
class TurnState:
    """一轮 Agent 问答的完整上下文（服务内部使用）。

    handlers 只看到 ctx（见 service/handlers/context.py），messages 与 chat_history
    属于服务自身的编排素材，不进处理器契约。
    """
    ctx: TurnContext             # 处理器入参：问题/会话/用户/DST/意图都在里面
    messages: list               # 注入意图指令后的消息列表（泛化 ReAct 兜底路径用）
    chat_history: Any            # 会话历史读写句柄
