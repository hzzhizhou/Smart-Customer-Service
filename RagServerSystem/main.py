from pathlib import Path
import sys
import atexit
import threading
from utils.thread_pool_manager import init_thread_pools
from utils.thread_pool_manager import shutdown_thread_pools
from utils.llm_factory import create_llm
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from infrastructure.vector_store.async_chroma_vector import ChromaVector
from retrieval.service import RetrievalService
from generation.service import AnswerGenerator
from infrastructure.chat_history_factory import init_chat_history
from access.app import RAGService
from config.settings import SESSION_ID, validate_chunk_config
from logs.log_config import log


def _warmup_reranker(retrieval_service: RetrievalService) -> None:
    """后台预热 Cross-Encoder 重排器（daemon 线程，不阻塞启动）。

    重排器是懒加载：278M 权重 + torch 首次推理惰性开销实测约 18s，
    不预热的话全部落在重启后的第一个用户请求上（首字 25s+，前端像"没反应"）。
    BM25 索引启动即构建，重排模型也应在启动期就绪，口径一致。
    预热完成前的请求会经 reranker 内部的加载锁等待同一份结果，不会重复加载。
    """
    reranker = getattr(retrieval_service, "reranker", None)
    if reranker is None:
        return

    def _run():
        import asyncio
        try:
            ready = asyncio.run(reranker.warmup())
            log.info("重排器预热完成" if ready else "重排器预热失败（已降级为不重排）")
        except Exception as e:
            log.warning(f"重排器预热异常（不影响服务启动）: {e}")

    threading.Thread(target=_run, name="reranker-warmup", daemon=True).start()


def main():
    # 启动时校验分块配置（只告警不阻断：切本地 bge 是显式选择，但必须让人看见截断风险）
    for warning in validate_chunk_config():
        log.warning(f"分块配置校验：{warning}")
    # 初始化各层
    init_thread_pools()
    atexit.register(shutdown_thread_pools)
    vector_store = ChromaVector()
    chat_history = init_chat_history(session_id=SESSION_ID)
    # 双实例分流：Agent 工具链必须用原始 ChatOpenAI（bind_tools），
    # 其余纯文本链路（查询改写/生成/意图/槽位）用带供应商降级的实例：
    # 主 DeepSeek 失败自动切备用通义，两道全挂才触发熔断兜底话术。
    llm = create_llm(streaming=True)
    text_llm = create_llm(streaming=True, with_fallback=True)
    retrieval_service = RetrievalService(vector_store, text_llm, chat_history)
    answer_generator = AnswerGenerator(text_llm)
    service = RAGService(retrieval_service, answer_generator,llm)
    # 重排模型预热：与 RAGService 装配/端口监听并行进行，启动后即处于"热"状态
    _warmup_reranker(retrieval_service)
    # 启动服务
    service.run()

if __name__ == "__main__":
    main()