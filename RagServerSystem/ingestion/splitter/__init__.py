"""分块模块：提供多种文档分块策略"""
from .splitter_factory import get_chunker
from .recursive import RecursiveChunker
from .semantic import SemanticChunker
from .sliding_window import SlidingWindowChunker

# 注意：这里原本列了 "BaseChunker"，但 base.py 里并没有这个类——`import *` 时会
# 抛 AttributeError。它从来没被触发过（没人用星号导入），顺手删掉不留假承诺。
__all__ = [
    "get_chunker",
    "RecursiveChunker",
    "SemanticChunker",
    "SlidingWindowChunker",
]
