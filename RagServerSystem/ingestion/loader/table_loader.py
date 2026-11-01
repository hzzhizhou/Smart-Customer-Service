"""
表格文档解析：CSV / DOCX / XLSX 的表格统一转成「带表头的 Markdown 表格」文本块。
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence

from langchain_core.documents import Document

from config.settings import TABLE_CHUNK_SIZE
from logs.log_config import data_layer_log as log

# 单块字符预算：走独立的 TABLE_CHUNK_SIZE，不沿用子块大小。表格每块都要重复表头，
# 宽表的表头本身就很长（9 列约 144 字符），借用 400 的小预算会把整表碎成
# 大量「表头 + 一行」的块（实测 40 行 FAQ 切成 28 块）。
DEFAULT_TABLE_BUDGET = TABLE_CHUNK_SIZE


def _cell(value) -> str:
    """单元格文本：压平换行、去掉竖线（竖线会破坏 Markdown 表格结构）"""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)          # xlsx 里的 599 会读成 599.0
    text = str(value).replace("\r", " ").replace("\n", " ")
    return " ".join(text.split()).replace("|", "/")


def render_markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """渲染 Markdown 表格（含表头分隔行）"""
    head = "| " + " | ".join(header) + " |"
    separator = "| " + " | ".join(["---"] * len(header)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, separator] + body)


def _normalize_rows(header: Sequence[str], rows: List[List[str]]) -> List[List[str]]:
    """列数对齐表头：短行补空、长行截断，避免 Markdown 表格列错位"""
    width = len(header)
    fixed = []
    for row in rows:
        if len(row) < width:
            fixed.append(list(row) + [""] * (width - len(row)))
        elif len(row) > width:
            fixed.append(list(row[:width]))
        else:
            fixed.append(list(row))
    return fixed


def _table_documents(header: List[str], rows: List[List[str]],
                     metadata: dict, budget: int) -> List[Document]:
    """
    按字符预算把整表切成多个「表头 + 整行」块
    :param header: 表头单元格
    :param rows: 数据行（已对齐列数）
    :param metadata: 附加元数据（除 is_table 外）
    :return: 表格块列表（无数据行时返回空，避免生成只有列名的无效块）
    """
    if not header or not rows:
        return []
    docs = []
    prefix_len = len(render_markdown_table(header, []))          # 表头 + 分隔行
    group: List[List[str]] = []
    size = prefix_len
    groups: List[List[List[str]]] = []
    for row in rows:
        row_len = len("| " + " | ".join(row) + " |") + 1
        if group and size + row_len > budget:
            groups.append(group)
            group, size = [], prefix_len
        group.append(row)
        size += row_len
    if group:
        groups.append(group)

    for part in groups:
        meta = dict(metadata)
        meta["is_table"] = True
        meta["table_rows"] = len(part)
        docs.append(Document(page_content=render_markdown_table(header, part), metadata=meta))
    return docs


def csv_to_documents(file_path: Path, budget: int = DEFAULT_TABLE_BUDGET) -> List[Document]:
    """CSV → 带表头的 Markdown 表格块（首行视为表头）"""
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        raw_rows = [[_cell(c) for c in row] for row in csv.reader(f)]
    raw_rows = [row for row in raw_rows if any(row)]
    if len(raw_rows) < 2:
        log.warning(f"CSV 无有效数据行，跳过: {file_path.name}")
        return []
    header = raw_rows[0]
    # 表格也要有归属：否则 section_path 为空，引用时只能说「出自某个表」
    return _table_documents(header, _normalize_rows(header, raw_rows[1:]),
                            {"Header1": file_path.stem}, budget)


_HEADING_RE = re.compile(r"(?:heading|标题)\s*([1-9])$", re.I)


def _heading_level(paragraph) -> int:
    """段落是几级标题？非标题返回 0，标题最多认到 3 级。

    中文 Word 的样式名是「标题 1」，英文模板是「Heading 1」，另有一些模板把
    style_id 写成 Heading1。三种都要认——企业文档大多来自中文模板，
    只匹配 "Heading" 会全部落空，等于没做标题识别。
    """
    style = getattr(paragraph, "style", None)
    if style is None:
        return 0
    for raw in (getattr(style, "name", "") or "", getattr(style, "style_id", "") or ""):
        m = _HEADING_RE.search(str(raw).strip())
        if m:
            return min(int(m.group(1)), 3)
    return 0


def docx_to_documents(file_path: Path, budget: int = DEFAULT_TABLE_BUDGET) -> List[Document]:
    """DOCX → 按标题样式切分的元素块（段落按节、表格为节内原子块）。

    为什么不再「遇到表格切一刀」：docx 没有 Markdown 那种显式标记，结构只能从
    paragraph.style 推断。旧的按表格切分有三个实测后果：

      1) Header1/2/3 全空 → section_path 为空 → 相邻小节合并、标题前缀、引用定位
         对 Word 文档全部失效，只能按 768 字硬切；
      2) 表格成了章节边界：「1.1 七天无理由退货」的正文被夹在表格前后，切成两半；
      3) 段落段没有章节切分，短小节（实测「1.2 退货流程」41 字）会因 MIN_CHUNK_SIZE
         被整段丢弃，日志里只有一行「过滤过短块」。

    现在：标题段落只更新层级上下文（自身不进正文，由前缀承担），正文按节累积；
    表格作为「当前节内的一张表」输出，并继承当前章节信息，不打断章节。
    """
    from docx import Document as _Docx
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    docx = _Docx(str(file_path))
    docs: List[Document] = []
    headers: Dict[int, str] = {1: "", 2: "", 3: ""}
    buffer: List[str] = []
    table_index = 0

    def current_meta() -> Dict[str, object]:
        meta: Dict[str, object] = {"is_table": False}
        for lvl in (1, 2, 3):
            if headers[lvl]:
                meta[f"Header{lvl}"] = headers[lvl]
        return meta

    def flush() -> None:
        nonlocal buffer
        if buffer:
            docs.append(Document(page_content="\n".join(buffer), metadata=current_meta()))
            buffer = []

    for child in docx.element.body.iterchildren():
        if child.tag == qn("w:p"):
            para = Paragraph(child, docx)
            text = para.text.strip()
            if not text:
                continue
            level = _heading_level(para)
            if level:
                flush()                       # 上一节收口
                headers[level] = text         # 开新节
                for deeper in range(level + 1, 4):
                    headers[deeper] = ""      # 更深的层级失效
            else:
                buffer.append(text)
        elif child.tag == qn("w:tbl"):
            flush()
            table_rows = [[_cell(cell.text) for cell in row.cells]
                          for row in Table(child, docx).rows]
            table_rows = [row for row in table_rows if any(row)]
            if len(table_rows) < 2:
                # 单行表（多为排版用表）：按普通文本保内容，不足两行无法构成表头+数据
                if table_rows:
                    buffer.append(" | ".join(table_rows[0]))
                continue
            header = table_rows[0]
            docs.extend(_table_documents(
                header, _normalize_rows(header, table_rows[1:]),
                {**current_meta(), "table_index": table_index}, budget))
            table_index += 1

    flush()
    return docs


def xlsx_to_documents(file_path: Path, budget: int = DEFAULT_TABLE_BUDGET) -> List[Document]:
    """XLSX → 每个工作表一张 Markdown 表（首行视为表头，按预算分成多块）"""
    from openpyxl import load_workbook

    workbook = load_workbook(str(file_path), data_only=True, read_only=True)
    docs: List[Document] = []
    try:
        for sheet in workbook.worksheets:
            raw_rows = [[_cell(c) for c in row] for row in sheet.iter_rows(values_only=True)]
            raw_rows = [row for row in raw_rows if any(row)]
            if len(raw_rows) < 2:
                log.warning(f"工作表 {sheet.title} 无有效数据行，跳过: {file_path.name}")
                continue
            header = raw_rows[0]
            docs.extend(_table_documents(
                header, _normalize_rows(header, raw_rows[1:]),
                {"Header1": file_path.stem, "Header2": str(sheet.title),
                 "sheet_name": str(sheet.title)}, budget))
    finally:
        workbook.close()
    return docs