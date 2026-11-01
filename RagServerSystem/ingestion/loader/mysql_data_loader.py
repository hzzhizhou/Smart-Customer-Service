from infrastructure.sql.mysql_state_manager import MySQLStateManager
from pathlib import Path
from datetime import datetime
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CHUNKING_STRATEGY
from ingestion.loader.pdf_loader import pdf_to_documents
from ingestion.loader.table_loader import (
    csv_to_documents, docx_to_documents, xlsx_to_documents
)
from ingestion.splitter.splitter_factory import get_chunker
from logs.log_config import data_layer_log as log


MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = [".txt", ".pdf", ".docx", ".xlsx", ".md", ".csv"]
THREAD_POOL = ThreadPoolExecutor(max_workers=8)


def derive_doc_category(file_name: str) -> str:
    """
    按文件名派生文档类别（检索预过滤维度，写入块元数据 doc_category）
    约定：品类词优先于通用词，例如 商品FAQ.csv 归为 product 而非 faq
    中英文关键词都要认：语料文件名已中文化（手机总览.md / 服务政策.md …），
    而 retrieval.infer_doc_categories 是用中文问题推断类别的，两侧必须能对上类别
    """
    name = file_name.lower()
    if any(kw in name for kw in ("phone", "手机", "机型", "苹果", "iphone", "华为", "小米")):
        return "phone"
    if any(kw in name for kw in ("product", "商品")):
        return "product"
    # policy 并入 faq：服务规则/售后政策与 FAQ 同属"政策知识"，若各成一类，
    # 按问题关键词预过滤时会把这类文档排除在候选之外（制度文件往往正是答案所在）。
    # sop/售后 同样并入：售后 SOP 与政策、FAQ 回答的是同一批问题，问题侧推断出的
    # 类别也是 faq，若此处归为 general，预过滤会把 SOP 挡在候选之外（只剩全库回退兜底）。
    if any(kw in name for kw in ("faq", "客服", "policy", "政策", "sop", "售后")):
        return "faq"
    return "general"


def _is_office_temp(path: Path) -> bool:
    """Office 的锁文件 / 隐藏临时文件：以 ~$ 或 .~ 开头，读取会 PermanError。"""
    name = path.name
    return name.startswith("~$") or name.startswith(".~") or name.startswith(".")


class StreamDocumentLoader:
    def __init__(self):
        self.splitter = get_chunker(CHUNKING_STRATEGY)
        self.state_mgr: MySQLStateManager = None   # 改为 MySQL
        # MySQL 状态写入串行化：入库用 asyncio.gather 并发处理多个文件，并发事务在
        # documents/doc_chunks 上互相等待会触发 1213 死锁（全量重建时实测 6 个文件里
        # 3 个中招）。死锁后该文件“向量已入 Chroma、状态未落 MySQL”，管理端列表会缺
        # 文档、增量去重也会误判，故写状态这一步加锁串行（耗时以嵌入计算为主，影响很小）。
        self._state_lock = asyncio.Lock()

    async def initialize(self):
        """初始化状态管理器（必须在 stream_dir_loader 前调用）"""
        self.state_mgr = MySQLStateManager()
        await self.state_mgr.initialize()
    # 同步文件加载（供线程池调用）
    def _load_sync(self, file_path: Path):
        try:
            if file_path.suffix == ".txt":
                docs = TextLoader(file_path, encoding="utf-8").load()
            elif file_path.suffix == ".pdf":
                # 结构化解析：编号伪标题切节、剔除页眉页脚、接续硬换行。
                # 不用 PyPDFLoader——它只给「每页一段纯文本」，章节结构全丢，
                # 切分层只能按字数硬切，条款会被从中间切断（见 pdf_loader 模块注释）。
                docs = pdf_to_documents(file_path)
            elif file_path.suffix == ".csv":
                # CSV 是表：转成带表头的 Markdown 表格块，而不是退化成无结构的行文本
                docs = csv_to_documents(file_path)
            elif file_path.suffix == ".docx":
                # 段落 + 表格：表格转 Markdown，避免 Docx2txtLoader 把表格拼成单元格文字
                docs = docx_to_documents(file_path)
            elif file_path.suffix == ".xlsx":
                # 每个工作表转一张带表头的 Markdown 表
                docs = xlsx_to_documents(file_path)
            elif file_path.suffix == ".md":
                # 用 TextLoader 读原文而非 UnstructuredMarkdownLoader：后者把 Markdown 表格
                # 解析成 Table 元素后不带回表格文本，实测 16904 字符的文档只读出 15322，
                # 10 张表格全部丢失（分块后一个表格行都不剩），使 splitter 里的表格保护
                # （split_table_segments / markdown_table_chunks）永远没机会执行。
                # 标题结构由 MarkdownHeaderTextSplitter 负责，不需要 loader 再做解析。
                docs = TextLoader(file_path, encoding="utf-8").load()
            else:
                raise ValueError("不支持的格式")

            norm_path = file_path.absolute().as_posix()
            doc_category = derive_doc_category(file_path.name)
            file_size = str(file_path.stat().st_size)
            load_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for doc in docs:
                doc.metadata.update({
                    "file_name": file_path.name,
                    "file_path": norm_path,
                    "file_type": file_path.suffix.strip("."),
                    "doc_category": doc_category,
                    "file_size": file_size,
                    "load_time": load_time
                })
                # 转换 Path 对象，并统一为正斜杠（与 doc_id 派生的归一化路径一致）
                if 'source' in doc.metadata and isinstance(doc.metadata['source'], Path):
                    doc.metadata['source'] = doc.metadata['source'].as_posix()
            from utils.text_clean import clean_documents
            return clean_documents(docs)
        except Exception as e:
            log.error(f"加载失败 {file_path.name}: {e}")
            return []

    # 异步单文件加载
    async def single_file_loader(self, file_path: Path) -> list[Document]:
        if not file_path.exists() or file_path.suffix not in ALLOWED_EXTENSIONS:
            return []
        if file_path.stat().st_size > MAX_FILE_SIZE:
            log.warning(f"文件过大跳过: {file_path.name}")
            return []
        log.info(f"⏳ 流式加载: {file_path.name}")
        docs = await asyncio.get_event_loop().run_in_executor(
            THREAD_POOL, self._load_sync, file_path
        )
        return docs
    async def stream_load_one_file(self, file_path: Path, vector_store) -> bool:
        """处理单个文件。返回 True=成功或无需更新，False=失败（旧向量可能已被清空）"""
        # 检查是否需要更新
        if not await self.state_mgr.need_update(file_path):
            log.info(f"文件无变化，跳过：{file_path.name}")
            return True

        try:
            # 1. 删除旧向量（基于 MySQL 中记录的旧 chunk_id）
            old_chunk_ids = await self.state_mgr.get_old_chunk_ids(file_path)
            if old_chunk_ids:
                await vector_store.delete_by_ids(old_chunk_ids)

            # 2. 加载、分块
            docs = await self.single_file_loader(file_path)
            if not docs:
                log.warning(f"未解析出内容，跳过：{file_path.name}")
                return True
            # 分块器统一返回 List[Document]。历史上这里还要处理「(chunks, parent_map)」
            # 二元组（父子分块的父块映射），父子分块移除后不再是可能形态。
            chunks = self.splitter.split_documents(docs)
            # 3. 入库（id 已由分块器统一写入 metadata，见 ingestion/splitter/base.py 的 add_chunk_metadata）
            await vector_store.aadd_documents(chunks)

            # 4. 更新 MySQL 状态（与向量库使用同一套 chunk_id，保证下次增量删除能对齐）
            chunk_ids = [chunk.metadata["id"] for chunk in chunks]
            async with self._state_lock:
                await self.state_mgr.update_state(file_path, chunk_ids)

            log.info(f"✅ 增量更新完成: {file_path.name}，共 {len(chunks)} 个块")
            return True
        except Exception as e:
            # 注意：步骤 1 已删除该文件的旧向量，此处失败会让它在向量库中「写空」。
            # 好在 MySQL 状态尚未更新（update_state 在 aadd_documents 之后），
            # 所以重跑本服务仍会判定为待更新并自动补齐，不需要人工介入。
            log.error(f"处理失败 {file_path.name}: {e}", exc_info=True)
            return False

    async def stream_dir_loader(self, dir_path: Path, vector_store):
        # 确保状态管理器已初始化
        if self.state_mgr is None:
            await self.initialize()

        # 获取当前文件列表（归一化为正斜杠，与 MySQL 存储的 file_path 保持一致）
        # 跳过 Office 的锁/临时文件（~$xxx.docx、.~xxx.xlsx）：用户开着 Word/Excel 时
        # 目录里就会出现它们，且被系统独占锁住——读一个就抛 PermissionError，整批入库失败
        current_files = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix in ALLOWED_EXTENSIONS and not _is_office_temp(f)
        ]
        current_paths = {f.absolute().as_posix() for f in current_files}

        # 获取 MySQL 中记录的文件路径
        recorded_paths = await self.state_mgr.get_all_file_paths()

        # 失败文件清单：per-file 异常不中断整批，但必须如实汇总，
        # 避免个别文件失败后仍打印「✅ 入库完成」掩盖问题
        failures = []

        # 处理已删除的文件
        deleted_paths = recorded_paths - current_paths
        for del_path_str in deleted_paths:
            del_path = Path(del_path_str)
            try:
                old_ids = await self.state_mgr.get_old_chunk_ids(del_path)
                if old_ids:
                    await vector_store.delete_by_ids(old_ids)
                await self.state_mgr.remove_document(del_path)
                log.info(f"🗑️ 已清理被删除文件: {del_path.name}")
            except Exception as e:
                log.error(f"清理失败 {del_path.name}: {e}")
                failures.append(f"{del_path.name}（清理已删除文件）")

        # 处理新增或修改的文件
        tasks = [self.stream_load_one_file(f, vector_store) for f in current_files]
        results = await asyncio.gather(*tasks)
        failures.extend(f.name for f, ok in zip(current_files, results) if not ok)

        if failures:
            log.warning(
                f"⚠️ 入库流程结束，但有 {len(failures)} 个文件失败：{'、'.join(failures)}；"
                f"这些文件的向量可能未更新，详见上文 ERROR 日志，重跑本服务可自动补齐"
            )
        else:
            log.info("MySQL 增量更新全流程完成")

        return failures