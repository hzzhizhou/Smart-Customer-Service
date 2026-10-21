"""
统一入库服务：全量 + 增量，共用同一条"加载 → 分块 → 入库 → 写 MySQL 状态"链路。

设计目标：
- 只有一个入库入口，消除"全量/增量两个不同实现 + 状态不一致"的问题。
- mode="incremental"：基于 MySQL 状态比对，只处理新增/修改/删除的文件。
- mode="full"      ：先清空向量库 + MySQL 状态，再全量重建（重建同样写完状态）。
- 无论哪种方式，MySQL 的 documents/doc_chunks 都与 Chroma 内容保持一致，
  从而增量去重、删除对齐始终准确。

运行时：RAG 服务(main.py)只加载 Chroma 不重建；
需要维护知识库时调用本服务（增量由 ingestion.loader.mysql_data_layer 的长驻监听驱动）。
"""
import asyncio
import time
from pathlib import Path

from config.settings import BASE_DIR, validate_chunk_config
from logs.log_config import data_layer_log as log
from infrastructure.vector_store.async_chroma_vector import ChromaVector
from ingestion.loader.mysql_data_loader import StreamDocumentLoader


class DataLayer:
    """统一入库服务入口"""

    @staticmethod
    async def ingest(dir_path=None, mode: str = "incremental", total_timeout: int = 600):
        """
        统一入库（全量 / 增量二合一）。

        :param dir_path:       数据目录，默认 BASE_DIR/"data"
        :param mode:           "incremental"(默认) / "full"
        :param total_timeout:  整条流程超时秒数
        """
        dir_path = Path(dir_path) if dir_path is not None else BASE_DIR / "data"
        if not dir_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {dir_path}")
        if mode not in ("incremental", "full"):
            raise ValueError(f"不支持的 mode: {mode}（可选 incremental/full）")

        # 启动校验分块配置（PROJECT_AUDIT §4.2(3)）：分块超长会被嵌入模型静默截断，
        # 检索质量下降却没有任何报错。只告警不阻断（切回本地 bge 是使用者的显式选择），
        # 也不重建索引 —— 分块参数变更须重新入库才生效。
        for warning in validate_chunk_config():
            log.warning(f"分块配置校验：{warning}")

        vector = ChromaVector()
        loader = StreamDocumentLoader()

        try:
            await asyncio.wait_for(
                DataLayer._run(dir_path, vector, loader, mode),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError:
            log.error(f"入库超时（{total_timeout}秒）")
            raise
        except Exception as e:
            log.error(f"入库失败: {e}", exc_info=True)
            raise
        finally:
            # 关闭不再需要的 loader，避免 sql 连接残留
            if loader.state_mgr is not None:
                await loader.state_mgr.close()

    @staticmethod
    async def _run(dir_path: Path, vector, loader, mode: str):
        if mode == "full":
            start = time.time()
            await vector.aclear()                      # 清空向量库
            if loader.state_mgr is None:
                await loader.initialize()
            await loader.state_mgr.reset()             # 清空 MySQL 状态（doc_chunks 级联）
            log.info(f"全量模式：已清空向量库 + MySQL 状态（{time.time()-start:.2f}s）")

        t0 = time.time()
        failures = await loader.stream_dir_loader(dir_path, vector)
        if failures:
            log.warning(
                f"⚠️ 入库结束（mode={mode}），但有 {len(failures)} 个文件失败："
                f"{'、'.join(failures)}；总耗时 {time.time()-t0:.2f}s"
            )
        else:
            log.info(f"✅ 入库完成（mode={mode}），总耗时 {time.time()-t0:.2f}s")

    # --------------------兼容/便捷入口--------------------
    @staticmethod
    async def data_loader(dir_path=None, mode: str = "full", total_timeout: int = 600):
        """兼容入口：默认全量重建（verify_dataset 等历史调用方使用）。"""
        await DataLayer.ingest(dir_path, mode=mode, total_timeout=total_timeout)


if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(DataLayer.ingest(BASE_DIR / "data", mode="full"))
    print(f"入库耗时: {time.time() - start_time:.2f}s")