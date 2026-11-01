"""分块公共工具函数"""
import re
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from config.settings import MIN_CHUNK_SIZE
from logs.log_config import data_layer_log as log
from utils.doc_identity import make_doc_id, make_chunk_text_hash, normalize_doc_path

# Markdown 标题层级配置
HEADERS_TO_SPLIT_ON = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]

# ====================== Markdown 表格保护 ======================
# 表格行必须整行保真：按标点/换行切分会把「表头与数据行」拆开、把一行里的多列拆到
# 不同的块，检索命中 "599" 这类单元格时无法还原它属于哪一行哪一列。
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def is_markdown_table_row(line: str) -> bool:
    """判断单行是否为 Markdown 表格行（形如 | a | b |）"""
    return bool(TABLE_ROW_RE.match(line))


def is_table_doc(chunk: Document) -> bool:
    """判断块是否由表格解析器生成：这类块已按「表头 + 整行」分组，长度受控，无需再切"""
    return str(chunk.metadata.get("is_table", "")).lower() == "true"


def split_table_segments(text: str) -> List[Tuple[bool, str]]:
    """
    把文本按「连续表格段 / 普通文本段」切开，返回 [(是否表格段, 文本)]，保持原始顺序
    """
    segments: List[Tuple[bool, str]] = []
    buffer: List[str] = []
    in_table = False
    for line in text.split("\n"):
        line_is_table = is_markdown_table_row(line)
        if line_is_table != in_table and buffer:
            segments.append((in_table, "\n".join(buffer)))
            buffer = []
        in_table = line_is_table
        buffer.append(line)
    if buffer:
        segments.append((in_table, "\n".join(buffer)))
    return segments


def markdown_table_chunks(text: str, budget: int) -> List[str]:
    """
    把一段 Markdown 表格按「表头 + 若干完整数据行」分组，每组字符数不超过 budget。
    单行本身超过 budget 时该行独占一组：宁可略超预算，也不切开一行。
    :param text: 表格段文本（首行为表头，允许紧跟分隔行 |---|）
    :param budget: 单组字符预算
    :return: 每组都自带表头的 Markdown 表格文本列表
    """
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return []
    header = lines[0]
    has_separator = len(lines) > 1 and bool(TABLE_SEPARATOR_RE.match(lines[1]))
    body = lines[2:] if has_separator else lines[1:]
    prefix = header + ("\n" + lines[1] if has_separator else "")
    if not body:
        return [prefix]

    groups: List[List[str]] = []
    current: List[str] = []
    size = len(prefix)
    for row in body:
        if current and size + len(row) + 1 > budget:
            groups.append(current)
            current, size = [], len(prefix)
        current.append(row)
        size += len(row) + 1
    if current:
        groups.append(current)
    return [prefix + "\n" + "\n".join(group) for group in groups]


def split_by_markdown_headers(docs: List[Document]) -> List[Document]:
    """
    对文档列表按 Markdown 标题结构进行预切分（保留原始元数据）
    :param docs: 原始文档列表
    :return: 按标题切分后的文档列表（未进行大小控制）
    """
    result = []
    try:
        md_splitter = MarkdownHeaderTextSplitter(HEADERS_TO_SPLIT_ON)
        for doc in docs:
            # 仅对 markdown 文件进行结构切分
            if doc.metadata.get("file_type") == "md":
                chunks = md_splitter.split_text(doc.page_content)
                # 将原始元数据复制到每个子块
                for chunk in chunks:
                    chunk.metadata.update(doc.metadata.copy())
                result.extend(chunks)
            else:
                result.append(doc)
    except Exception as e:
        log.error(f"Markdown 结构切分失败: {e}")
        return docs
    return result


def resolve_doc_source(chunk: Document) -> str:
    """从块元数据中解析文档来源路径（优先 file_path，兼容 source/file_name）。"""
    return (chunk.metadata.get("file_path")
            or chunk.metadata.get("source")
            or chunk.metadata.get("file_name")
            or "")


def make_chunk_id(chunk: Document, index: int) -> str:
    """
    生成企业规范的确定性块 ID：<doc_id>::chunk::<序号>
    - doc_id      = sha256(归一化绝对路径)  → 稳定、路径非主键、跨模块同源
    - 序号(seq)   = 块在文档内的顺序        → 可定位、可替换
    同一文件重新入库时 doc_id 稳定，向量库 upsert 可直接覆盖旧块，
    增量更新时状态库(MySQL doc_chunks)记录的 chunk_id 与向量库一致，可对齐删除。
    """
    doc_id = make_doc_id(resolve_doc_source(chunk))
    return f"{doc_id}::chunk::{index:04d}"


def make_parent_id(parent: Document, index: int) -> str:
    """
    生成父块映射的唯一键：<doc_id>::parent::<序号>
    用 doc_id（路径哈希）替代原始文件名，避免"不同目录下同名文件"的父块键碰撞；
    用 ::parent:: 前缀与子块 id(::chunk::) 区分。
    父块不写向量库，仅作检索后回查完整上下文的映射键。
    """
    doc_id = make_doc_id(resolve_doc_source(parent))
    return f"{doc_id}::parent::{index:04d}"


def prepend_section_path(chunk: Document) -> None:
    """
    把 Markdown 标题路径回填到块正文开头（就地修改）。

    MarkdownHeaderTextSplitter 默认 strip_headers=True：标题行会被剥离出 page_content，
    只留在 metadata（Header1/2/3），而全项目没有任何地方读这几个键。后果是向量与 BM25
    都索引不到章节名，「第十五章」「附则」这类章节指代型提问召回不到；段落也失去了所属
    小节的上下文（如「为统一…特制定本流程」脱离「1.1 制定目的」标题后可读性变差）。

    这里按「文档标题 > 章 > 节」拼成前缀写回正文，让标题重新参与检索与生成。
    非 Markdown 块没有 Header 元数据，直接跳过；已带前缀的块也跳过——组合分块的子块会
    被 add_chunk_metadata 处理两次，必须保证幂等，否则前缀会叠加两遍。
    """
    parts = [str(chunk.metadata.get(k) or "").strip()
             for k in ("Header1", "Header2", "Header3")]
    parts = [p for p in parts if p]
    if not parts:
        return
    prefix = f"【{' > '.join(parts)}】"
    if not chunk.page_content.lstrip().startswith(prefix):
        chunk.page_content = f"{prefix}\n{chunk.page_content}"


def classify_block(chunk: Document) -> str:
    """判定块的内容类型（而不是「用哪个分块器切出来的」）。

    历史字段 chunk_type 恒为 "recursive"，那是分块器名，不是内容类型，
    于是下游想「表格块转成字段:值 再送模型」时无从判断。分块器名改为存 splitter。
    """
    if is_table_doc(chunk):
        return "table_row"
    text = chunk.page_content or ""
    if re.search(r"^\s*(Q[:：]|问[:：])", text, re.M):
        return "faq_pair"
    if re.search(r"^\s*[-*]\s+", text, re.M):
        return "list"
    return "paragraph"


def build_section_path(chunk: Document) -> str:
    """把 Header1/2/3 拼成一条可读路径，替代三列分散取值。"""
    parts = [str(chunk.metadata.get(k) or "").strip()
             for k in ("Header1", "Header2", "Header3")]
    return " > ".join(p for p in parts if p)


def add_chunk_metadata(chunks: List[Document], chunk_type: str = "normal",
                       min_chunk_size: int = MIN_CHUNK_SIZE) -> List[Document]:
    """
    过滤过短块，并为分块添加标准元数据（id, doc_id, chunk_index, chunk_type,
    chunk_text_hash），同时把 Markdown 标题路径回填进正文（见 prepend_section_path）
    :param chunks: 分块列表
    :param chunk_type: 块类型标识（normal, child, parent, recursive 等），以传入值为准
    :param min_chunk_size: 最小块长度，低于该长度的块被丢弃（避免无效 embedding）
    :return: 处理后的列表
    """
    # keep_short：已确认无宿主可归的短块。丢弃比"块小"更伤人（用户直接查不到），
    # 所以这类块带标记穿透，只有真正无信息量的碎片才被丢掉。
    valid_chunks = [c for c in chunks
                    if len(c.page_content.strip()) >= min_chunk_size
                    or c.metadata.get("keep_short")]
    dropped = len(chunks) - len(valid_chunks)
    if dropped:
        log.info(f"过滤过短块: 丢弃 {dropped} 个（阈值 {min_chunk_size} 字符）")
    for i, chunk in enumerate(valid_chunks):
        # 标题路径放在 chunk_text_hash 之前回填：指纹要反映真正入库的最终内容
        prepend_section_path(chunk)
        # 统一归一化 sources：无论来自哪个 loader，入库/比对都用正斜杠绝对路径
        for _key in ("file_path", "source"):
            if chunk.metadata.get(_key):
                chunk.metadata[_key] = normalize_doc_path(chunk.metadata[_key])
        chunk.metadata["chunk_index"] = i
        # chunk_type 以调用方传入的为准（覆盖继承值）：子块会从父块继承元数据，若这里
        # "已有就不覆盖"，父块的 semantic 标签会被 RecursiveCharacterTextSplitter 复制到
        # 所有子块上，导致子块类型全部误标为 semantic（实测 29/29 中招）。
        # 分块器名与内容类型分开：chunk_type 是「这块是什么」，splitter 是「谁切的」
        chunk.metadata["splitter"] = chunk_type
        chunk.metadata["chunk_type"] = classify_block(chunk)
        # 企业规范：doc_id 稳定可锚定；块内容指纹用于去重/溯源
        chunk.metadata["doc_id"] = make_doc_id(resolve_doc_source(chunk))
        chunk.metadata["chunk_text_hash"] = make_chunk_text_hash(chunk.page_content)
        chunk.metadata["id"] = make_chunk_id(chunk, i)
        # —— 类型归一：下游要能直接过滤/排序，不能每次自己转 ——
        try:
            chunk.metadata["file_size"] = int(chunk.metadata.get("file_size") or 0)
        except (TypeError, ValueError):
            chunk.metadata["file_size"] = 0
        chunk.metadata["is_table"] = is_table_doc(chunk)
        # —— 覆盖归一：所有块都有 section_path ——
        # 无结构的来源（CSV 表格、无标题文档）给空串：Chroma 接受空字符串，
        # 但不接受**空列表**，所以这里刻意不用列表型字段装结构信息。
        chunk.metadata.setdefault("section_path", build_section_path(chunk))
        # —— 去掉冗余字段 ——
        # source 与 file_path 内容完全相同（且表格块压根没有 source），保留一个即可
        chunk.metadata.pop("source", None)
        # 历史上还有 merged_sections（合并了几节）：纯派生值，已废弃，这里顺手清掉旧块残留
        chunk.metadata.pop("merged_sections", None)
        chunk.metadata.pop("section_ids", None)

    # 入库前按契约自检：缺字段/类型不符/枚举外，一次全报出来。
    # 不抛异常——线上宁可入库，也不因一个字段名写错整批失败；但绝不允许静默通过。
    from ingestion.chunk_schema import validate_chunks
    problems, ok_count = validate_chunks(valid_chunks)
    if problems:
        log.warning(f"块元数据不符合契约：{len(problems)} 个块有问题"
                    f"（合规 {ok_count}/{len(valid_chunks)}），前 3 条：{problems[:3]}")
    return valid_chunks


# ====================== 相邻小节合并（预算驱动） ======================
# 结构切分只按标题下切，粒度由「文档有多少个 ###」决定，而不是由内容量决定：
# 服务政策 56 个 ### → 57 块，而 1.1（265 字）+ 1.2（149 字）合计才 414 字，仍各自成块。
# 后果有两条：答案跨小节时召不全（问「退货运费谁承担」要同时看 1.1 与 1.2）；
# 短块里标题前缀占比高达 30%，向量被导航信息主导。所以补上「太小就向上合并」这一半。
#
# 不做成配置开关：它是切分质量的内建规则，不是业务选择——留开关只会多一种
# 「配错了却没人发现」的失败模式。
MERGE_BUDGET = 600


def _section_key(chunk: Document):
    """分组合并的键：同一章（Header2）之下的相邻小节才允许合并。"""
    return (chunk.metadata.get("Header1"), chunk.metadata.get("Header2"))


def _body_with_heading(chunk: Document) -> str:
    """把被剥离的三级标题作为一行插回正文。

    MarkdownHeaderTextSplitter 的 strip_headers=True 会把标题从正文剥离、只留在 metadata。
    单块时靠前缀能补回来；但合并后一个块含多个小节，必须在正文里保留各小节的标题，
    否则模型看到的只是几段无标题条款接在一起，不知道自己读到了哪一节。
    """
    h3 = str(chunk.metadata.get("Header3") or "").strip()
    body = chunk.page_content.strip()
    # 不写 "### "：模型读纯文本标题已经够用，Markdown 标记只在渲染时才有意义，
    # 而生成层是纯文本拼接；这 4 个字符对 embedding 是纯噪声，还要乘上每次召回。
    return f"{h3}\n{body}" if h3 else body


def merge_adjacent_sections(chunks: List[Document],
                            budget: int = MERGE_BUDGET) -> List[Document]:
    """把同一章下的相邻小节，累计未超预算的合并成一块。

    三条规则：
      - 只合并同一 (Header1, Header2) 分组内的相邻块，不跨章；
      - 表格块不参与（is_table 已是「表头 + 整行」原子单元，合并会破坏结构）；
      - 完全没有标题的块直接放行，交给后面的长度控制处理——那类文档本来就没有
        章节语义，按预算硬合并反而会把无关段落粘在一起。
    """
    out: List[Document] = []
    buf: List[Document] = []
    buf_key = None

    def flush():
        nonlocal buf, buf_key
        if not buf:
            return
        if len(buf) == 1:
            out.append(buf[0])
        else:
            merged = Document(
                page_content="\n\n".join(_body_with_heading(c) for c in buf),
                metadata=dict(buf[0].metadata),
            )
            # 合并块跨多个三级小节：前缀只能写「章」，否则会把第一小节的标题
            # 当成本块标题，指向错误的位置
            merged.metadata.pop("Header3", None)
            out.append(merged)
        buf, buf_key = [], None

    for c in chunks:
        if is_table_doc(c):
            flush()
            out.append(c)
            continue
        key = _section_key(c)
        if not any(key):
            flush()
            out.append(c)
            continue
        if buf and key != buf_key:
            flush()
        # 预算按「合并后真实正文长度」算：插回的三级标题行也要计入，
        # 否则每合并一节就悄悄多出 10~20 字，块会超出预算却不自知
        merged_len = sum(len(_body_with_heading(x)) + 2 for x in buf) \
            + len(_body_with_heading(c)) + 2
        if buf and merged_len > budget:
            flush()
        buf.append(c)
        buf_key = key
    flush()
    return out


def _own_heading(chunk: Document) -> str:
    """块自己最具体的一级标题（三级优先，其次二级）。用于归并时保留标签。"""
    for key in ("Header3", "Header2"):
        val = str(chunk.metadata.get(key) or "").strip()
        if val:
            return val
    return ""


def absorb_short_sections(chunks: List[Document],
                          min_size: int = MIN_CHUNK_SIZE) -> List[Document]:
    """把短于最小块长度的**文本**残块并到邻块，而不是直接丢弃。

    为什么需要：`add_chunk_metadata` 会把 < MIN_CHUNK_SIZE 的块当噪声丢掉。Markdown
    里少见（一节常几百字），Word/PDF 里极常见——实测「2.1 领取与使用」正文只有 37 字、
    「1.3 退保规则」29 字，一节就是一句话；丢了它，用户问「优惠券怎么领」永远查不到。

    宿主优先级（顺序就是正确性的关键）：
      1. **同一段内**（不跨表格）最近的长块；
      2. 段内全是短块时，并入**紧邻的同节表格**（文本按原位置放在表格前或后）——
         实测「2.1 正文 + 2.1 表格」本就属于同一节，合成一块既保内容又保顺序；
      3. 都没有 → **原样保留并打 keep_short**，交给过滤层豁免。

    为什么不再跨表格合并：早期版本为救孤块而跨表格并入同章邻块，结果把表格**后面**的
    「2.2 不可用场景」提前到了「2.1 表格」之前——内容没丢，但阅读顺序错乱，看块列表会以为
    2.2 被切没了。顺序错乱比块小更伤人。
    """
    def chapter_of(chunk: Document) -> str:
        return chunk.metadata.get("Header1") or ""

    def section_of(chunk: Document) -> str:
        return chunk.metadata.get("Header2") or ""

    def body_of(chunk: Document) -> str:
        label = _own_heading(chunk)
        body = chunk.page_content.strip()
        return f"{label}\n{body}" if label else body

    # 1) 切「章内连续文本段」：表格与换章都切断（表格是屏障，不参与归并）
    runs: List[List[int]] = []
    current: List[int] = []
    for i, chunk in enumerate(chunks):
        if is_table_doc(chunk):
            if current:
                runs.append(current)
                current = []
            continue
        if current and chapter_of(chunks[current[-1]]) != chapter_of(chunk):
            runs.append(current)
            current = []
        current.append(i)
    if current:
        runs.append(current)

    # 2) 段内分组；段内无长块时尝试并入紧邻的同节表格
    groups: List[List[int]] = []          # 每组第一个元素是宿主（文本块）
    table_hosts: Dict[int, List[int]] = {}   # 表格下标 → 并入的短文本下标
    keep_short: set = set()
    for run in runs:
        longs = [i for i in run if len(chunks[i].page_content) >= min_size]
        if longs:
            gmap = {i: [i] for i in longs}
            for i in run:
                if i in gmap:
                    continue
                nearest = min(longs, key=lambda j: abs(j - i))
                gmap[nearest].append(i)
            groups.extend(gmap.values())
            continue
        # 整段都是短块：先看紧邻的表格是不是同一节
        host = None
        for cand in (run[0] - 1, run[-1] + 1):
            if (0 <= cand < len(chunks) and is_table_doc(chunks[cand])
                    and section_of(chunks[cand]) == section_of(chunks[run[0]])):
                host = cand
                break
        if host is not None:
            table_hosts.setdefault(host, []).extend(run)
        elif len(run) > 1:
            # 一段里全是短块（如「章前引言 + 一句话小节」）：它们本就首尾相接，
            # 合成一块即可，没必要各自成小块（实测合前 35/37 字两块，合后 72 字一块）
            groups.append(run)
        else:
            keep_short.update(run)
            groups.append(run)

    # 3) 合并文本组，按组内最早位置输出
    placed = []
    consumed = set()
    for group in groups:
        members = sorted(group)
        consumed.update(members)
        host = members[0]
        meta = dict(chunks[host].metadata)
        labels = {_own_heading(chunks[j]) for j in members if _own_heading(chunks[j])}
        if any(lbl != _own_heading(chunks[host]) for lbl in labels):
            meta.pop("Header3", None)
            meta.pop("Header2", None)     # 跨小节 → 前缀收缩到章
        if any(j in keep_short for j in members):
            meta["keep_short"] = True
        parts = [chunks[j].page_content if j == host else body_of(chunks[j]) for j in members]
        placed.append((members[0], Document(
            page_content="\n\n".join(p for p in parts if p), metadata=meta)))

    # 4) 表格：有短文本并入时，按原位置把文本拼在表格前 / 后
    for i, chunk in enumerate(chunks):
        if not is_table_doc(chunk):
            continue
        guests = table_hosts.get(i)
        if not guests:
            placed.append((i, chunk))
            continue
        before = [j for j in sorted(guests) if j < i]
        after = [j for j in sorted(guests) if j > i]
        parts = [body_of(chunks[j]) for j in before] + [chunk.page_content] \
            + [body_of(chunks[j]) for j in after]
        meta = dict(chunk.metadata)
        placed.append((i, Document(page_content="\n\n".join(p for p in parts if p),
                                   metadata=meta)))

    placed.sort(key=lambda x: x[0])
    return [doc for _, doc in placed]


def split_sentences(text: str) -> List[str]:
    """
    简单的中英文句子分割（基于标点）
    表格段不参与标点切分：整段表格作为整体返回，避免表格被切散
    :param text: 原始文本
    :return: 句子列表
    """
    # 匹配句号、感叹号、问号、分号等结尾的句子
    pattern = r'(?<=[。！？!?;；])\s*'
    sentences: List[str] = []
    for is_table, segment in split_table_segments(text):
        if not segment.strip():
            continue
        if is_table:
            sentences.append(segment.strip())
        else:
            sentences.extend(s.strip() for s in re.split(pattern, segment) if s.strip())
    return sentences