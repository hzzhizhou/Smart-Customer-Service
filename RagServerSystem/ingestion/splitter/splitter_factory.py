"""分块器工厂：根据配置返回对应的分块器实例

为什么这里没有父子分块：父子分块的收益前提是「文档多、单篇长、答案常跨小节」，
本项目语料（12 篇、单篇最长 1.5 万字）不满足；而且它需要 parent_cache.json 单独
维护父块映射，全项目从未生成过——生成层每轮启动都要打一条「父子分块功能不可用」
的 WARNING。一个从不生效的策略留在选项里，只会让后来者以为它在跑。
跨小节召回改用「同章相邻小节按预算合并」（见 base.merge_adjacent_sections），
成本约为父子分块的十分之一。
"""
from config.settings import (
    CHUNKING_STRATEGY,
    CHUNK_SIZE, CHUNK_OVERLAP,
    SEMANTIC_CHUNK_THRESHOLD, SEMANTIC_BUFFER_SIZE,
    SEMANTIC_EMBEDDING_MODEL,
    SLIDING_WINDOW_STEP,
)
from logs.log_config import data_layer_log as log
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sliding_window import SlidingWindowChunker


def get_chunker(strategy: str = None):
    """
    获取分块器实例
    :param strategy: 可选值: recursive, semantic, sliding_window
                     默认从 settings.CHUNKING_STRATEGY 读取
    :return: BaseChunker 子类实例
    """
    strategy = strategy or CHUNKING_STRATEGY
    log.info(f"初始化分块器，策略={strategy}")

    if strategy == "semantic":
        return SemanticChunker(
            threshold=SEMANTIC_CHUNK_THRESHOLD,
            buffer_size=SEMANTIC_BUFFER_SIZE,
            embedding_model=SEMANTIC_EMBEDDING_MODEL
        )
    elif strategy == "sliding_window":
        return SlidingWindowChunker(
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
    else:  # 默认递归分块（结构优先 + 相邻小节按预算合并）
        return RecursiveChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
