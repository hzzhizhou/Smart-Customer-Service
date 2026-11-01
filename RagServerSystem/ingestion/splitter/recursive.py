"""递归字符分块（原 DocumentSplitter 逻辑）"""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from logs.log_config import data_layer_log as log
from .base import (
    split_by_markdown_headers, add_chunk_metadata, merge_adjacent_sections, MERGE_BUDGET,
    absorb_short_sections,
    is_table_doc, is_markdown_table_row, split_table_segments, markdown_table_chunks
)


class RecursiveChunker():
    """
    递归字符分块器（默认策略）
    先对 Markdown 文件按标题切分，再对超长块递归切分
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        # 1. 对 Markdown 文件进行结构切分
        structure_chunks = split_by_markdown_headers(docs)

        #相邻小节按预算合并。结构切分只「向下切」，粒度由文档有多少个三级标题决定。这里补上「太小就向上合并」，让块大小由内容量决定
        before_merge = len(structure_chunks)
        structure_chunks = merge_adjacent_sections(structure_chunks)
        # 仍短于最小块长度的残块（Word 的「一句话小节」）并回同章相邻块，不直接丢
        structure_chunks = absorb_short_sections(structure_chunks)
        if len(structure_chunks) != before_merge:
            log.info(f"相邻小节合并：{before_merge} → {len(structure_chunks)} 块"
                     f"（预算 {MERGE_BUDGET} 字）")

        # 2. 递归控制长度
        final_chunks = []
        for chunk in structure_chunks:
            # 表格解析器产出的块已按「表头 + 整行」分组，长度受控，不再二次切分
            if is_table_doc(chunk) or len(chunk.page_content) <= self.chunk_size:
                final_chunks.append(chunk)
            elif any(is_markdown_table_row(line) for line in chunk.page_content.split("\n")):
                # 含表格的超长块：表格段单独走「表头 + 整行」分组，避免被 sep 切散
                final_chunks.extend(self._split_with_tables(chunk))
            else:
                sub_chunks = self.text_splitter.split_documents([chunk])
                final_chunks.extend(sub_chunks)

        # 3. 添加块级元数据
        final_chunks = add_chunk_metadata(final_chunks, chunk_type="recursive")
        log.info(f"递归分块完成，共 {len(final_chunks)} 块")
        return final_chunks

    def _split_with_tables(self, chunk: Document) -> List[Document]:
        """普通文本段沿用递归切分，表格段按「表头 + 整行」分组（不切开任何一行）"""
        result = []
        for is_table, segment in split_table_segments(chunk.page_content):
            if not segment.strip():
                continue
            if is_table:
                for table_text in markdown_table_chunks(segment, self.chunk_size):
                    result.append(Document(page_content=table_text,
                                           metadata=chunk.metadata.copy()))
            else:
                result.extend(self.text_splitter.split_documents(
                    [Document(page_content=segment, metadata=chunk.metadata.copy())]))
        return result