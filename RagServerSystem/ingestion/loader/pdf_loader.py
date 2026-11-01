"""PDF → 按编号伪标题切节的元素块（表格按「表头 + 整行」还原）。

PDF 没有 Markdown 的显式标记，也没有 Word 的标题样式，结构只能靠版面线索推断：
  1. 编号伪标题：`一、` / `第X章` 为一级，`1.1` / `（一）` 为二级，`1.1.1` / `（1）` 为三级；
  2. 重复行：页眉页脚在多页的页首/页尾重复出现，按频次剔除；
  3. 硬换行：句子被折断（「自 2026 年 1 月 1」/「日起施行。」），上一行无句末标点且
     下一行不像标题时接续；
  4. 表格：用 pdfplumber 的 `find_tables()` 拿到**单元格二维数组**，按 y 坐标插回阅读顺序
     中的原位置，再走与 Word / Excel 相同的「表头 + 整行」分组。

为什么必须换掉 pypdf：pypdf 只输出字符流，不给坐标、也不识别表格——一张 5 行 7 列的表会被
拆成 35 行文字，行列关系彻底丢失（检索命中「20 元」却不知道它属于哪一行哪一列）。
PDF 里没有「表格对象」，只有字符和线条；能还原表格的前提是解析器读得到坐标。

已知限制：
  - **扫描件不做 OCR**：整份无文字层时明确拒收并告警，不往库里塞空块；
  - 无线条的「假表格」（纯空格对齐）识别不到，仍按文本处理；
  - 跨页表格会被拆成两张（不做跨页合并）。
"""
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document

from logs.log_config import data_layer_log as log

# 章节编号线索。注意顺序：先判三级，再判二级（1.1.1 也会命中 1.1 的前缀）
_L3_RE = re.compile(r"^(?:\d+\.\d+\.\d+|[（(]\d+[）)])")
_L2_RE = re.compile(r"^(?:第[一二三四五六七八九十百]+条|\d+\.\d+(?!\.)|[（(][一二三四五六七八九十]+[）)])")
_L1_RE = re.compile(r"^(?:第[一二三四五六七八九十百]+[章编部]|[一二三四五六七八九十]+、|第[一二三四五六七八九十百]+部分)")

# 页码形态：「第 2 页 / 共 3 页」「- 2 -」「2/3」「第2页」
_PAGE_NO_RE = re.compile(
    r"^(?:第\s*\d+\s*页(?:\s*[/／]\s*共?\s*\d+\s*页)?|[-—–]\s*\d+\s*[-—–]|\d+\s*[/／]\s*\d+)$"
)

# 句末标点：这些结尾说明句子已结束，不需要和下一行接续
_TERMINAL = ("。", "！", "？", "；", "：", "」", "』", "”", "）", ")", "】", "…")

_HEADING_MAX_LEN = 40        # 超过这个长度的不可能是标题
_MIN_SENTENCE_LEN = 10       # 上一行至少这么长才值得接续（避免把表格单元格连成句）
_MIN_CONTINUATION = 8        # 下一行至少这么长才可能是正文接续
_SCANNED_MIN_CHARS = 50      # 全文字符数低于此值视为无文字层

# 单元格内部常有软换行，统一压成空格
_WS_RE = re.compile(r"\s+")
_NEWLINE_RE = re.compile(r"\s*\n\s*")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _clean_cell(value) -> str:
    """清洗单元格文本。

    PDF 单元格内常有软换行（「升级门槛（累\n计消费）」），直接替换成空格会把中文词
    拆开（「累 计消费」）。两侧都是中日韩字符时直接拼接，否则保留一个空格。
    """
    text = str(value or "")

    def _join(match):
        left = text[match.start() - 1] if match.start() > 0 else ""
        right = text[match.end()] if match.end() < len(text) else ""
        if _CJK_RE.match(left) and _CJK_RE.match(right):
            return ""
        return " "

    return _WS_RE.sub(" ", _NEWLINE_RE.sub(_join, text)).strip()


def _heading_level(line: str) -> int:
    """按编号线索判断标题层级，非标题返回 0。

    编号本身不足以判定：政策正文常用「（一）（二）」做条款序号，实测
    「（一）本政策适用于智选商城（英文名…」会被误判成二级标题，把一段正文切成独立小节。
    所以括号序号额外要求短——真标题不会写到 24 字。
    """
    text = line.strip()
    if not text or len(text) > _HEADING_MAX_LEN:
        return 0
    # 标题不会以句读结尾：实测断行接续会拼出「第七章、第六章相关规则处理。」这种
    # 伪标题，把后续内容错误地挂到一个不存在的小节下（问号保留，FAQ 式标题常用）
    if text[-1] in "。，；、":
        return 0
    if _L3_RE.match(text):
        return 3 if len(text) <= 30 else 0
    if _L2_RE.match(text):
        # 括号序号（一）（二）在政策文档里几乎总是**条款序号**，不是标题。
        # 实测「（二）用户完成商品评价的，每单额外奖励 10」被当成标题，把一句话
        # 劈成「标题 + 积分;…」，既产生了伪小节，又把句子切断了。
        # 只有「短且无句读」的括号行才可能是标题（如「（一）总则」）。
        if text[0] in "（(":
            if len(text) > 12 or any(ch in text for ch in "，；。、,;"):
                return 0
        return 2
    if _L1_RE.match(text):
        return 1 if len(text) <= 30 else 0
    return 0


def _strip_running_heads(pages: List[List[str]]) -> List[List[str]]:
    """剔除页眉页脚：在多页的页首或页尾重复出现即视为页眉页脚。

    单页无法判断重复，只剔除页码形态的行——只按「第一行永远是页眉」处理会误删正文首行。
    """
    def drop_page_numbers(lines: List[str]) -> List[str]:
        return [l for l in lines if not _PAGE_NO_RE.match(l.strip())]

    if len(pages) < 2:
        return [drop_page_numbers(p) for p in pages]

    edge_counter: Counter = Counter()
    for lines in pages:
        for line in set(lines[:2] + lines[-2:]):
            if line.strip():
                edge_counter[line.strip()] += 1
    threshold = max(2, int(len(pages) * 0.5))
    running = {text for text, count in edge_counter.items() if count >= threshold}
    return [[l for l in lines
             if l.strip() not in running and not _PAGE_NO_RE.match(l.strip())]
            for lines in pages]


def _page_units(page) -> List[Tuple[str, object, float]]:
    """把一页拆成按阅读顺序排列的单元：("text", 行, top) 与 ("table", 二维数组, top)。

    表格区域内的文字行会被丢弃——它们属于表格，不能再作为正文重复出现。
    """
    try:
        tables = page.find_tables()
    except Exception as exc:                      # 复杂版面可能让识别失败
        log.warning(f"表格识别失败，该页按纯文本处理: {exc}")
        tables = []

    boxes = [t.bbox for t in tables]
    units: List[Tuple[str, object, float]] = []

    try:
        text_lines = page.extract_text_lines()
    except Exception as exc:
        log.warning(f"文本行抽取失败: {exc}")
        text_lines = []

    for line in text_lines:
        text = (line.get("text") or "").strip()
        if not text:
            continue
        center = (line.get("top", 0) + line.get("bottom", 0)) / 2
        if any(box[1] <= center <= box[3] for box in boxes):
            continue                              # 落在表格框内，由表格单元负责
        units.append(("text", text, float(line.get("top", 0))))

    for table in tables:
        try:
            rows = [[_clean_cell(c) for c in row] for row in table.extract()]
        except Exception as exc:
            log.warning(f"表格内容抽取失败: {exc}")
            continue
        rows = [r for r in rows if any(cell for cell in r)]
        if len(rows) < 2:                         # 单行表多半是排版用表，退回正文处理
            continue
        units.append(("table", rows, float(table.bbox[1])))

    units.sort(key=lambda u: u[2])
    return units


def _join_lines(units: List[Tuple[str, object, int]]) -> List[Tuple[str, object, int]]:
    """接续被硬换行折断的句子（跨页也接）；表格单元不参与。"""
    out: List[Tuple[str, object, int]] = []
    for kind, payload, page in units:
        if kind == "table":
            out.append((kind, payload, page))
            continue
        text = str(payload)
        if out and out[-1][0] == "text":
            prev_text, prev_page = str(out[-1][1]), out[-1][2]
            joinable = (
                not prev_text.endswith(_TERMINAL)
                and len(prev_text) >= _MIN_SENTENCE_LEN
                and len(text) >= _MIN_CONTINUATION
                and _heading_level(text) == 0
                and _heading_level(prev_text) == 0
                # 首行不接续：PDF 第一行常是文档标题（无标点、无编号），
                # 拼上正文后会超长，标题识别彻底失效
                and not (len(out) == 1 and len(prev_text) <= 20)
            )
            if joinable:
                out[-1] = ("text", prev_text + text, prev_page)
                continue
        out.append((kind, text, page))
    return out


def _build_sections(units, doc_title: str) -> List[Document]:
    """按伪标题切节；表格作为「节内原子块」输出，并继承当前章节。"""
    from ingestion.loader.table_loader import _table_documents

    docs: List[Document] = []
    headers = {1: "", 2: "", 3: ""}
    buffer: List[str] = []
    start_page = end_page = 0

    def current_meta(extra: dict) -> dict:
        meta = {"is_table": False, "page_start": start_page, "page_end": end_page}
        for level in (1, 2, 3):
            if headers[level]:
                meta[f"Header{level}"] = headers[level]
        if not headers[1] and doc_title:
            meta["Header1"] = doc_title
        meta.update(extra)
        return meta

    def flush():
        nonlocal buffer
        if buffer:
            docs.append(Document(page_content="\n".join(buffer), metadata=current_meta({})))
            buffer = []

    table_index = 0
    for kind, payload, page in units:
        if kind == "table":
            flush()
            rows = payload
            docs.extend(_table_documents(
                rows[0], rows[1:],
                current_meta({"table_index": table_index,
                              "page_start": page, "page_end": page}),
                budget=1000))
            table_index += 1
            continue

        text = str(payload)
        level = _heading_level(text)
        if level:
            flush()
            headers[level] = text
            for deeper in range(level + 1, 4):
                headers[deeper] = ""
            start_page = end_page = page
            continue
        if not buffer:
            start_page = page
        buffer.append(text)
        end_page = page
    flush()
    return docs


def pdf_to_documents(file_path: Path) -> List[Document]:
    """PDF → 元素块：编号切节、剔除页眉页脚、接续硬换行、表格按行列还原。

    返回空列表表示「拿不到文字」（扫描件/加密），调用方据此跳过——不要用空块占位，
    空块会进索引，让检索命中一段什么都没有的内容。
    """
    import pdfplumber

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            page_units = [_page_units(page) for page in pdf.pages]
    except Exception as exc:
        log.warning(f"PDF 打开失败，跳过: {file_path.name} | {exc}")
        return []

    total_chars = sum(len(str(payload)) for units in page_units
                      for kind, payload, _ in units if kind == "text")
    if total_chars < _SCANNED_MIN_CHARS:
        log.warning(f"PDF 无可用文字层（疑似扫描件），已跳过: {file_path.name}"
                    f"（共 {len(page_units)} 页，{total_chars} 字符）｜需要 OCR 才能入库")
        return []

    # 页眉页脚：按页算重复行，再把被剔除的行从单元序列里去掉
    per_page_text = [[str(p) for k, p, _ in units if k == "text"] for units in page_units]
    kept = _strip_running_heads(per_page_text)
    units: List[Tuple[str, object, int]] = []
    for page_no, (page_unit, page_kept) in enumerate(zip(page_units, kept), start=1):
        quota = Counter(page_kept)
        for kind, payload, _ in page_unit:
            if kind == "text":
                text = str(payload)
                if quota.get(text, 0) > 0:
                    quota[text] -= 1
                    units.append(("text", text, page_no))
            else:
                units.append(("table", payload, page_no))

    units = _join_lines(units)

    # 首个标题之前的第一行通常是文档标题（无编号，匹配不到任何规则），抽出来当兜底
    doc_title = ""
    if units and units[0][0] == "text":
        first = str(units[0][1])
        if _heading_level(first) == 0 and len(first) <= _HEADING_MAX_LEN:
            doc_title = first
            units = units[1:]

    docs = _build_sections(units, doc_title)
    if not docs:
        log.warning(f"PDF 未解析出正文，跳过: {file_path.name}")
        return []
    tables = sum(1 for kind, _, _ in units if kind == "table")
    log.info(f"PDF 解析完成: {file_path.name} | {len(page_units)} 页 | "
             f"{tables} 张表 → {len(docs)} 个元素块")
    return docs
