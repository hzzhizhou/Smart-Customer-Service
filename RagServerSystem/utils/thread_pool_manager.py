"""
线程池管理工具类，用于创建、获取和关闭检索线程池。

三类池的定位不同，别混用：
  - bm25  ：CPU 密集（分词 + 打分），池大小 = 核数
  - vector：等待远端 embedding / 查库，属 I/O 等待，池可以放大
  - rerank：CPU 密集且会再往下开 torch 线程，池大小必须小（见 RERANK_CONCURRENCY）

重排单独建池的原因：它原来用 loop.run_in_executor(None, ...)，即 asyncio 的默认
线程池，max_workers = min(32, 核数+4)。默认池是「给所有阻塞调用共用的一张网」，
在上面跑 CPU 密集任务等于没有并发上限——两个子问并行时会开出 2×12 个 torch 线程
抢 16 个核，实测单请求 3.6s 恶化到 8.4~9.0s。
"""
from concurrent.futures import ThreadPoolExecutor

from config.settings import (
    BM25_THREAD_POOL_SIZE, BM25_THREAD_POOL_NAME,
    VECTOR_THREAD_POOL_SIZE, VECTOR_THREAD_POOL_NAME,
    RERANK_CONCURRENCY, RERANK_TORCH_THREADS,
    CPU_CORES,
)
from logs.log_config import log

# 池规格表：初始化与惰性创建共用一份，避免两处写死不同的大小
_POOL_SPECS = {
    "bm25": (BM25_THREAD_POOL_SIZE, BM25_THREAD_POOL_NAME),
    "vector": (VECTOR_THREAD_POOL_SIZE, VECTOR_THREAD_POOL_NAME),
    "rerank": (RERANK_CONCURRENCY, "rerank-pool"),
}

# 全局线程池字典（单例，避免重复创建）
THREAD_POOLS = {}

_torch_limited = False


def _limit_torch_threads() -> None:
    """
    固定 torch 的算子内并行线程数（可选，默认不干预）。

    为什么默认不设：16 核机器上实测线程数 4/6/8/12 的吞吐差异落在噪声里
    （0.99~1.13 题/s），重排已被算力本身饱和；把线程压到 4 反而让单请求
    从 ~800ms 涨到 ~1140ms。既然没有数据支撑，就不要去改一个影响全局、
    又只能设一次的参数——RERANK_TORCH_THREADS=0 表示沿用默认。

    真需要设的场景是小核数机器（4~8 核）跑多实例：那时
    `并发数 × 每请求线程数 > 核数` 会真正拖慢整体，可通过环境变量覆盖。
    """
    global _torch_limited
    if _torch_limited:
        return
    _torch_limited = True
    if RERANK_TORCH_THREADS <= 0:
        log.info(
            f"torch 算子线程数保持默认（核数 {CPU_CORES}，重排并发上限 {RERANK_CONCURRENCY}；"
            f"实测调到更小的值无收益，如需限制可设 RERANK_TORCH_THREADS）"
        )
        return
    import torch                                 # 延迟导入：本模块被多处 import，不想每次导入都背 torch 的启动开销
    try:
        torch.set_num_threads(RERANK_TORCH_THREADS)
        log.info(
            f"torch 算子线程数已固定为 {RERANK_TORCH_THREADS}"
            f"（核数 {CPU_CORES}，重排并发上限 {RERANK_CONCURRENCY}，"
            f"预算 {RERANK_CONCURRENCY}×{RERANK_TORCH_THREADS}≤{CPU_CORES}）"
        )
    except Exception as exc:                     # 设不上不该拦启动，只是回到默认值
        log.warning(f"设置 torch 线程数失败（沿用默认）：{exc}")


def _ensure_pool(pool_type: str) -> ThreadPoolExecutor:
    """
    按需创建并返回线程池。
    """
    if pool_type not in _POOL_SPECS:
        raise ValueError(f"未知的线程池类型：{pool_type}（可选：{', '.join(_POOL_SPECS)}）")
    if pool_type not in THREAD_POOLS:
        size, prefix = _POOL_SPECS[pool_type]
        THREAD_POOLS[pool_type] = ThreadPoolExecutor(
            max_workers=size, thread_name_prefix=prefix
        )
        if pool_type == "rerank":
            _limit_torch_threads()
        log.info(f"{pool_type} 线程池已创建（max_workers={size}）")
    return THREAD_POOLS[pool_type]


def init_thread_pools():
    """服务启动时初始化所有线程池（预热，避免首个请求承担创建开销）"""
    try:
        for pool_type in _POOL_SPECS:
            _ensure_pool(pool_type)
        log.info("所有检索线程池初始化完成")
    except Exception as e:
        log.error(f"线程池初始化失败：{e}")
        raise


def get_thread_pool(pool_type: str) -> ThreadPoolExecutor:
    """获取指定类型的线程池（不存在则创建）"""
    return _ensure_pool(pool_type)


def shutdown_thread_pools():
    """服务优雅关闭时销毁线程池"""
    for pool_type, pool in THREAD_POOLS.items():
        pool.shutdown(wait=True)
        log.info(f"{pool_type}线程池已关闭")
