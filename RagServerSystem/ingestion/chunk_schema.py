"""
入库块的元数据契约与校验。
"""
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

# 必备字段 → 期望类型。这里是「干净 schema」的唯一事实来源。
REQUIRED_FIELDS: Dict[str, type] = {
    # 文档级
    "doc_id": str,
    "file_name": str,
    "file_path": str,
    "file_type": str,
    "doc_category": str,
    "file_size": int,
    # 块级
    "id": str,
    "chunk_index": int,
    "chunk_text_hash": str,
    "chunk_type": str,      # 内容类型：paragraph / faq_pair / list / table_row
    "splitter": str,        # 用哪个分块器产出：recursive / parent / child / normal
    # 结构级（无结构来源给空串；Word 读标题样式、PDF 读书签，都能填）
    "section_path": str,
    # 表格
    "is_table": bool,
    # 治理
    "load_time": str,
}

# 允许的内容类型。加新类型必须同时改这里，避免出现 "recursive" 这种「分块器名
# 混进内容类型」的历史遗留。
CHUNK_TYPES = frozenset({"paragraph", "faq_pair", "list", "table_row", "other"})


def validate_chunks(chunks: List[Document]) -> Tuple[List[str], int]:
    """校验块元数据，返回 (问题列表, 合规块数)。

    只做结构校验，不做业务判断：缺字段、类型不符、chunk_type 不在枚举内。
    """
    problems: List[str] = []
    ok = 0
    for i, chunk in enumerate(chunks):
        meta: Dict[str, Any] = chunk.metadata or {}
        missing = [k for k, t in REQUIRED_FIELDS.items() if k not in meta]
        wrong = [(k, type(meta[k]).__name__, t.__name__)
                 for k, t in REQUIRED_FIELDS.items()
                 if k in meta and not isinstance(meta[k], t)]
        if missing or wrong:
            detail = []
            if missing:
                detail.append(f"缺字段 {missing}")
            if wrong:
                detail.append("类型不符 " + ", ".join(f"{k}:{got}≠{exp}" for k, got, exp in wrong))
            problems.append(f"块#{i}({meta.get('file_name', '?')}): " + "; ".join(detail))
            continue
        if meta["chunk_type"] not in CHUNK_TYPES:
            problems.append(f"块#{i}: chunk_type={meta['chunk_type']} 不在枚举内")
            continue
        ok += 1
    return problems, ok
