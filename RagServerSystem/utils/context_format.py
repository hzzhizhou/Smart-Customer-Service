"""
上下文格式化工具类，用于将检索到的文档格式化为符合要求的字符串。
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from logs.log_config import generation_layer_log as log


def format_context(docs: List[Document], max_context_length: int = 4000) -> str:
    """把检索结果拼成喂给模型的上下文。

    Args:
        docs: 检索返回的文档列表（顺序由检索层决定，这里不重排）
        max_context_length: 最大上下文长度（字符数），超出则截断

    Returns:
        格式化后的上下文字符串
    """
    if not docs:
        return "无参考资料"

    context_parts = []
    for idx, doc in enumerate(docs):
        file_name = doc.metadata.get("file_name") or f"未知文档_{idx}"
        if "\\" in file_name or "/" in file_name:
            file_name = Path(file_name).name
        context_parts.append(f"来源：{file_name}\n{doc.page_content}")

    full_context = "\n\n".join(context_parts)
    if len(full_context) > max_context_length:
        log.warning(f"上下文过长 ({len(full_context)} 字符)，截断至 {max_context_length}")
        full_context = full_context[:max_context_length]

    log.debug(f"上下文预览：{full_context[:200]}")
    return full_context
