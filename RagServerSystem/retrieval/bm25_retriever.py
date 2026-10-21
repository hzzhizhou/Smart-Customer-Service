from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi
import asyncio
import threading
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from utils.thread_pool_manager import get_thread_pool, init_thread_pools
from config.settings import RETRIEVER_K, BM25_ASYNC_TIMEOUT, BM25_SCORE_THRESHOLD  # 新增阈值导入
from utils.tokenizer import tokenizer
from logs.log_config import retrieval_layer_log as log
from retrieval.filters import matches_filter
import time
from functools import partial

class Bm25Retriever(BaseRetriever):
    vector_store: Any = Field(description="向量库实例")
    k: int = Field(default=RETRIEVER_K, description="检索返回条数")
    score_threshold: float = Field(default=BM25_SCORE_THRESHOLD, description="BM25分数过滤阈值")  # 新增阈值参数

    _bm25: Any = PrivateAttr()
    _docs: List[str] = PrivateAttr()
    _metadatas: List[dict] = PrivateAttr()
    _built_version: str = PrivateAttr()
    _refresh_lock: Any = PrivateAttr()

    def __init__(self, vector_store, k: int = RETRIEVER_K, score_threshold: float = BM25_SCORE_THRESHOLD):  # 新增参数
        super().__init__(vector_store=vector_store, k=k, score_threshold=score_threshold)  # 传递阈值
        bm25, docs, metadatas = self._init_bm25()
        object.__setattr__(self, '_bm25', bm25)
        object.__setattr__(self, '_docs', docs)
        object.__setattr__(self, '_metadatas', metadatas)
        object.__setattr__(self, '_built_version', self.vector_store.get_index_version())
        object.__setattr__(self, '_refresh_lock', threading.Lock())

    def _init_bm25(self):
        try:
            documents = self.vector_store._get_all_documents()
            docs = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            tokenized_docs = [tokenizer.tokenize_document(doc) for doc in docs]
            bm25 = BM25Okapi(tokenized_docs)
            log.info(f"bm25索引构建完成,总共{len(docs)}条文档")
            return bm25, docs, metadatas
        except Exception as e:
            log.error(f"bm25索引构建失败: {e}", exc_info=True)
            raise

    def _maybe_refresh_index(self) -> None:
        """
        向量库写入过（版本号变化）就重建 BM25 索引

        BM25 索引是构造时的一次性内存快照。服务运行期间上传/全量重建/删除文档后，
        不重建就会继续检索旧快照：新块召不回、已删块仍被召回，与向量分支数据不一致，
        必须重启服务才能恢复。版本号持久化在 Chroma 集合元数据里，因此另一个进程
        （如 python -m ingestion.service）发起的重建也能被感知。
        版本号不可读（返回空串）时跳过判断，退化为原行为。
        """
        current = self.vector_store.get_index_version()
        if not current or current == self._built_version:
            return
        with self._refresh_lock:                      # 异步检索在线程池里跑，需串行化重建
            latest = self.vector_store.get_index_version()
            if not latest or latest == self._built_version:
                return                                # 双检：可能已被其它线程重建
            try:
                bm25, docs, metadatas = self._init_bm25()
            except Exception as e:
                log.error(f"BM25 索引重建失败，继续使用旧索引: {e}", exc_info=True)
                return
            object.__setattr__(self, '_bm25', bm25)
            object.__setattr__(self, '_docs', docs)
            object.__setattr__(self, '_metadatas', metadatas)
            object.__setattr__(self, '_built_version', latest)
            log.info(f"检测到向量库变更，BM25 索引已重建（版本 {latest}，共 {len(docs)} 块）")

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        start_time = time.time()
        self._maybe_refresh_index()      # 同步/异步检索都走这里，统一在此校验索引新鲜度
        tokens = tokenizer.tokenize_query(query)
        log.debug(f"分词结果：{tokens}")
        if not tokens:
            log.warning(f"查询分词为空 | 查询：{query[:50]}...")
            return []
        
        try:
            # 1. 计算所有文档的BM25分数
            scores = self._bm25.get_scores(tokens)
            
            # 2. 筛选：分数≥阈值 且 索引有效
            valid_indices = [
                idx for idx, score in enumerate(scores)
                if score >= self.score_threshold
            ]
            
            # 3. 按分数降序取前 k 条（分数相同则保持库内原顺序，sorted 稳定）
            sorted_valid_indices = sorted(
                valid_indices,
                key=lambda i: scores[i],
                reverse=True
            )[:self.k]  # 取前k条（最多k条）
            
            # 4. 构建返回文档列表（支持 metadata filter 后置过滤）
            #    过滤语义与向量库共用 matches_filter：必须认 $in，否则多类别预过滤
            #    会把 BM25 路整个清空，混合检索静默退化成纯向量（PROJECT_AUDIT §4.2(1)）
            filter_condition = kwargs.get('filter')
            filtered_docs = []
            for idx in sorted_valid_indices:
                metadata = {**self._metadatas[idx], "bm25_score": float(scores[idx])}
                if filter_condition and not matches_filter(metadata, filter_condition):
                    continue
                filtered_docs.append(Document(page_content=self._docs[idx], metadata=metadata))
            
            # 5. 日志输出过滤结果
            total_valid = len(valid_indices)
            final_count = len(filtered_docs)
            log.info(
                f"Bm25检索完成 | "
                f"原始有效文档数：{total_valid} | "
                f"阈值过滤后返回：{final_count}条 | "
                f"阈值：{self.score_threshold} | "
                f"耗时：{time.time()-start_time:.3f}秒"
            )
            return filtered_docs
        
        except Exception as e:
            log.error(f"Bm25检索失败: {e}", exc_info=True)
            return []

    async def _aget_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        try:
            start_time = time.time()
            loop = asyncio.get_running_loop()
            func = partial(self._get_relevant_documents, query, **kwargs)
            docs = await asyncio.wait_for(
                loop.run_in_executor(get_thread_pool("bm25"), func),
                timeout=BM25_ASYNC_TIMEOUT
            )
            log.info(
                f"异步BM25检索完成 | "
                f"查询：{query[:50]}... | "
                f"返回文档数：{len(docs)} | "
                f"耗时：{time.time()-start_time:.3f}秒"
            )
            return docs
        except asyncio.TimeoutError:
            log.error(f"异步BM25检索超时 | 查询：{query[:50]}... | 超时时间：{BM25_ASYNC_TIMEOUT}秒")
            raise
        except Exception as e:
            log.error(f"异步BM25检索失败: {e}", exc_info=True)
            raise
if __name__ =='__main__':
    # 实例化时自定义阈值（覆盖全局配置）
    bm25_retriever = Bm25Retriever(
        vector_store=ChromaVector(),
        k=10,
        score_threshold=0.2 # 仅返回分数≥0.2的文档
    )

    # 同步检索
    # docs = bm25_retriever.invoke("大模型私有化部署")
    questions = [
        "大模型私有化部署",
        "RAG"
    ]
    # ✅ 正确打印每一条的分数
    # print("=== 检索结果分数 ===")
    init_thread_pools()
    async def async_retrieve():
        tasks = [ bm25_retriever.ainvoke(q) for q in questions]
        result =await asyncio.gather(*tasks)
        return result
    result =  asyncio.run(async_retrieve())
    # for i, doc in enumerate(result[1]):
    #     print(f"第{i+1}条 | 分数：{doc.metadata['bm25_score']:.4f}")
    # # 异步检索