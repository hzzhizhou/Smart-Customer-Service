"""
异步 Chroma 向量数据库实现，用于存储和检索向量数据。
"""
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.thread_pool_manager import get_thread_pool
from config.settings import DASHSCOPE_EMBEDDING_MODEL, EMBEDDING_BACKEND, LOCAL_EMBEDDING_MODEL, VECTOR_DB_DIR
from langchain_community.embeddings import DashScopeEmbeddings,HuggingFaceEmbeddings
from infrastructure.vector_store.base_store import BaseVectorStore
from logs.log_config import data_layer_log as log
from utils.doc_identity import normalize_doc_path
import asyncio

class ChromaVector(BaseVectorStore):
    def __init__(self, persist_dir: str = VECTOR_DB_DIR, embedding_model: str =LOCAL_EMBEDDING_MODEL) -> None:
        super().__init__()
        self.persist_dir = persist_dir
        # 嵌入模型同步/异步通用
        try:
            self._executor = get_thread_pool("vector")
        except:
            # 若还未初始化，则创建一个默认大小的线程池
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chroma")
        if EMBEDDING_BACKEND == "local":
            model_name = embedding_model or LOCAL_EMBEDDING_MODEL
            log.info(f"初始化本地嵌入模型: {model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            # 云端后端时固定用 DASHSCOPE_EMBEDDING_MODEL，忽略本地 embedding_model 参数
            # （否则 embedding_model 默认为本地 bge 路径，会被误当作云端模型名）
            model_name = DASHSCOPE_EMBEDDING_MODEL
            log.info(f"初始化云端嵌入模型: {model_name}")
            self.embedding_model = DashScopeEmbeddings(model=model_name)
        self.vector_store: Optional[Chroma] = None
        self._init_vector()

    def _init_vector(self):
        """初始化向量库（同步）"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embedding_model,
                collection_name=self.COLLECTION_NAME
            )
            log.info(f"Chroma向量数据库加载成功,持久化目录{self.persist_dir}")
        except Exception as e:
            log.error(f"Chroma向量数据库加载失败{e}", exc_info=True)
            raise

    # ===================== 【索引版本号：让 BM25 感知向量库变更】 =====================
    # BM25 索引是检索器构造时一次性构建的内存快照。若向量库之后被写入（上传/全量重建/
    # 删除），BM25 仍会检索旧快照：新块召不回、已删块还在，与向量分支的数据不一致，
    # 必须重启服务才恢复。这里把「写入次数」持久化到 Chroma 集合元数据，检索器每次查询前
    # 比对版本号，变化就重建索引。
    # 用集合元数据而不是进程内变量：全量重建常由另一个进程执行
    # （python -m ingestion.service），进程内计数器看不到。
    INDEX_VERSION_KEY = "index_version"
    COLLECTION_NAME = "Chroma_db"

    def _read_index_version(self) -> Optional[int]:
        """读取索引版本号；读不到时返回 None（调用方据此跳过重建判断）

        必须用 client.get_collection 重新取集合对象：chromadb 本地模式下
        Collection.metadata 是进程内缓存，另一个进程（python -m ingestion.service）改过之后
        本进程从旧对象上读到的仍是旧值（已实测），只有重新 get_collection 才是库里的现值。
        """
        try:
            collection = self.vector_store._client.get_collection(self.COLLECTION_NAME)
            metadata = collection.metadata or {}
            return int(metadata.get(self.INDEX_VERSION_KEY, 0))
        except Exception as e:
            log.warning(f"读取索引版本号失败: {e}")
            return None

    def _write_index_version(self, version: int) -> None:
        try:
            collection = self.vector_store._collection
            collection.modify(metadata={
                **(collection.metadata or {}),
                self.INDEX_VERSION_KEY: int(version)
            })
        except Exception as e:
            log.warning(f"写入索引版本号失败（BM25 需重启后才会刷新）: {e}")

    def _bump_index_version(self) -> None:
        """写入成功后自增版本号；失败只告警，不影响入库主流程"""
        current = self._read_index_version()
        self._write_index_version((current or 0) + 1)

    def get_index_version(self) -> str:
        """对外暴露的索引版本号；返回空串表示不可读，调用方应跳过重建判断"""
        version = self._read_index_version()
        return "" if version is None else str(version)

    # ===================== 【原有同步方法 · 保留】 =====================
    def add_documents(self, docs: List[Document], batch_size: int = 30) -> None:
        """同步批量添加文档（本地用）"""
        if not docs:
            log.warning("无文档需要加载，跳过")
            return
        total = len(docs)
        for i in range(0, total, batch_size):
            batch = docs[i:i+batch_size]
            try:
                self.vector_store.add_documents(batch)
                log.info(f"Chroma 入库批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}，共 {len(batch)} 个块")
            except Exception as e:
                log.error(f"批次 {i//batch_size + 1} 入库失败: {e}", exc_info=True)
                raise
        log.info(f"Chroma 入库完成，总计 {total} 个块")
        self._bump_index_version()      # 通知 BM25 索引：快照已过期

    def _get_all_documents(self) -> List[Document]:
        """获取所有文档（构建BM25用）"""
        try:
            result = self.vector_store.get()
            return [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(result['documents'], result['metadatas'])
            ]
        except Exception as e:
            log.error(f"Chroma 获取所有文档失败: {e}", exc_info=True)
            return []

    def similarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Document]:
        """【修复BUG】同步相似度检索"""
        try:
            return self.vector_store.similarity_search(query, k=k, filter=filter)
        except Exception as e:
            log.error(f"同步相似度检索失败: {e}", exc_info=True)
            return []

    def similarity_search_with_score(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:
        """【修复BUG】同步带分数检索"""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k, filter=filter)
        except Exception as e:
            log.error(f"同步带分数检索失败: {e}", exc_info=True)
            return []

    def as_retriever(self, **kwargs):
        return self.vector_store.as_retriever(**kwargs)

    def clear(self) -> None:
        """同步清空向量库"""
        try:
            old_version = self._read_index_version() or 0
            if self.vector_store:
                self.vector_store.delete_collection()
            self._init_vector()
            # delete_collection 会连集合元数据一起删掉，版本号需显式推进到旧值+1，
            # 否则已构建的 BM25 索引看不到这次清空重建、继续检索旧快照
            self._write_index_version(old_version + 1)
            log.warning("Chroma 向量库已清空并重建")
        except Exception as e:
            log.error(f"向量数据库清除失败{e}", exc_info=True)
            raise

    # ===================== 【新增异步方法 · API高并发专用】 =====================
    async def aadd_documents(self, docs: List[Document], batch_size: int = 10) -> None:
        """
        ✅ 异步批量添加文档（通过线程池执行同步的 add_documents）
        用于：API 高并发、批量文档异步入库
        """
        if not docs:
            return

        loop = asyncio.get_running_loop()
        total = len(docs)
        
        # 准备数据：id 必须来自分块器写入的 metadata["id"]（确定性，可覆盖/去重）
        # 缺 id 说明绕过了 add_chunk_metadata，属 bug —— 直接 fail-fast，不做 uuid 兜底
        missing_id = [i for i, doc in enumerate(docs) if not doc.metadata.get("id")]
        if missing_id:
            raise ValueError(
                f"存在 {len(missing_id)} 个缺少 metadata['id'] 的块（第 {missing_id[:5]} 个）。"
                "请确保入库前经过 ingestion/splitter/base.add_chunk_metadata 生成确定性 id。"
            )
        ids = [doc.metadata["id"] for doc in docs]
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        
        # 生成 embeddings（同步方法，放入线程池）
        # 注意：DashScope 云端 embedding 单次请求最多 20 条，需按 16 分片调用
        EMBED_BATCH = 16
        embeddings = []
        for i in range(0, total, EMBED_BATCH):
            chunk = await loop.run_in_executor(
                None,
                self.embedding_model.embed_documents,
                texts[i:i + EMBED_BATCH],
            )
            embeddings.extend(chunk)

        # 维度自检：换嵌入模型（如 qwen3.7-text-embedding → ...-flash）后，
        # 旧集合的维度可能不同；直接 upsert 会被 Chroma 以含糊异常拒绝。
        # 这里先读一条旧向量，给出「请全量重建索引」的明确错误。
        if embeddings:
            try:
                probe = await loop.run_in_executor(
                    self._executor,
                    lambda: self.vector_store._collection.get(
                        limit=1, include=["embeddings"]),
                )
                old_vectors = (probe or {}).get("embeddings")
                # 空集合时 chromadb 返回 numpy 空数组：对它做 `x or []` 会触发
                # ValueError: The truth value of an empty array is ambiguous
                # （全量重建时集合刚被清空，正是这个场景），必须先判 None、再看 size。
                if old_vectors is None:
                    old_vectors = []
                old_dim = len(old_vectors[0]) if len(old_vectors) else 0
                new_dim = len(embeddings[0])
                if old_dim and old_dim != new_dim:
                    raise ValueError(
                        f"嵌入维度不一致：集合中已有向量 {old_dim} 维，当前模型输出 {new_dim} 维。"
                        f"请全量重建索引（ingestion.service full）或切回原嵌入模型。"
                    )
            except ValueError:
                raise
            except Exception as e:
                # 读不到旧向量（空集合/接口差异）不应阻断入库，交给后续 upsert 自己校验
                log.warning(f"嵌入维度自检跳过（读取旧向量失败）: {e}")

        # 分批 upsert
        for i in range(0, total, batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            await loop.run_in_executor(
                self._executor,
                lambda: self.vector_store._collection.upsert(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
            )
            log.info(f"【异步】入库批次 {i//batch_size + 1}/{(total-1)//batch_size + 1} 完成，共 {len(batch_ids)} 个块")
        
        log.info(f"【异步】Chroma upsert 完成，总计 {total} 个块")
        self._bump_index_version()      # 通知 BM25 索引：快照已过期

    async def asimilarity_search(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Document]:
        """✅ 异步相似度检索（通过线程池执行同步的 similarity_search）"""
        loop = asyncio.get_running_loop()
        try:
            # run_in_executor 只接收位置参数，k/filter 必须先用 partial 绑定，
            # 否则会抛 TypeError: run_in_executor() got an unexpected keyword argument
            func = partial(self.vector_store.similarity_search, query, k=k, filter=filter)
            return await loop.run_in_executor(self._executor, func)
        except Exception as e:
            log.error(f"【异步】相似度检索失败: {e}", exc_info=True)
            return []

    async def asimilarity_search_with_score(self, query: str, k: int = 3, filter: Optional[dict] = None) -> List[Tuple[Document, float]]:
        """✅ 异步带分数检索（通过线程池执行同步的 similarity_search_with_score）"""
        loop = asyncio.get_running_loop()
        try:
            func = partial(self.vector_store.similarity_search_with_score, query, k=k, filter=filter)
            return await loop.run_in_executor(self._executor, func)
        except Exception as e:
            log.error(f"【异步】带分数检索失败: {e}", exc_info=True)
            return []

    async def asimilarity_search_by_vector(self, embedding: List[float], k: int = 3, filter: Optional[dict] = None):
        """✅ 异步根据预计算向量进行相似度检索（已经正确实现，无需修改）"""
        loop = asyncio.get_running_loop()
        func = partial(self.vector_store.similarity_search_by_vector_with_relevance_scores, embedding, k=k, filter=filter)
        return await loop.run_in_executor(self._executor, func)

    async def aclear(self) -> None:
        """✅ 异步清空向量库（通过线程池执行同步的 delete_collection）"""
        loop = asyncio.get_running_loop()
        try:
            old_version = self._read_index_version() or 0
            if self.vector_store:
                # delete_collection 是同步方法，放入线程池
                await loop.run_in_executor(self._executor, self.vector_store.delete_collection)
            # _init_vector 是同步方法，也放入线程池执行（或者直接调用，但为了避免阻塞，也放线程池）
            await loop.run_in_executor(self._executor, self._init_vector)
            # 集合元数据随 delete_collection 丢失，显式推进版本号让 BM25 感知重建
            await loop.run_in_executor(self._executor, self._write_index_version, old_version + 1)
            log.info("【异步】Chroma 向量库已清空并重建")
        except Exception as e:
            log.error(f"【异步】向量数据库清除失败{e}", exc_info=True)
            raise

    async def delete_by_file(self, file_path: str):
        """异步删除指定文件的所有向量（基于 metadata 中的 file_path）"""
        loop = asyncio.get_running_loop()
        try:
            # 归一化路径后再比对：入库时 file_path 已统一为正斜杠绝对路径
            norm_path = normalize_doc_path(file_path)
            if not norm_path:
                log.warning("删除文件路径为空，跳过: %s", file_path)
                return
            # Chroma 的 delete 支持 where 条件
            await loop.run_in_executor(
                self._executor,
                lambda: self.vector_store._collection.delete(where={"file_path": norm_path})
            )
            log.info(f"已删除文件 {file_path} 的旧向量")
            self._bump_index_version()      # 通知 BM25 索引：快照已过期
        except Exception as e:
            log.error(f"删除文件 {file_path} 向量失败: {e}")

    async def delete_by_ids(self, ids: List[str]):
        """批量删除指定 ID 的向量（Chroma 原生支持）"""
        if not ids:
            return

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: self.vector_store._collection.delete(ids=ids)
            )
            log.info(f"已删除 {len(ids)} 个向量")
            self._bump_index_version()      # 通知 BM25 索引：快照已过期
        except Exception as e:
            log.error(f"批量删除向量失败: {e}")
            # 不抛出，避免中断整个流程
