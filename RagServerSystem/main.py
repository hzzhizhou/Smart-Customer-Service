from pathlib import Path
import sys
import atexit
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


def main():
    # 启动时校验分块配置（只告警不阻断：切本地 bge 是显式选择，但必须让人看见截断风险）
    for warning in validate_chunk_config():
        log.warning(f"分块配置校验：{warning}")
    # 初始化各层
    init_thread_pools()
    atexit.register(shutdown_thread_pools)
    vector_store = ChromaVector()
    chat_history = init_chat_history(session_id=SESSION_ID)
    llm = create_llm(streaming=True)
    retrieval_service = RetrievalService(vector_store, llm, chat_history)
    answer_generator = AnswerGenerator(llm)
    service = RAGService(retrieval_service, answer_generator,llm)
    # 启动服务
    service.run()

if __name__ == "__main__":
    main()