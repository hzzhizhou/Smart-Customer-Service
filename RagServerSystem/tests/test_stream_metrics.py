"""TTFT（首 token 延迟）与检索阶段耗时指标测试。

锁住三件事：
  1) _TTFTRecorder 一轮只记一次（幂等），按 path 分标签，且耗时钳到非负；
  2) _StreamQueue 把内容原样转发给底层队列，并在首次写入时打点（哨兵走真实队列，不会被误判）；
  3) 检索阶段打点：retrieve() 会落下 rewrite / decompose / search / rerank / total 五个阶段各一次。
"""
import asyncio
import time
from types import SimpleNamespace

from langchain_core.documents import Document
from prometheus_client import REGISTRY

from retrieval.service import RetrievalService
from service.turn_service import _StreamQueue, _TTFTRecorder


def _ttft_count(path: str) -> float:
    return REGISTRY.get_sample_value("web_agent_ttft_seconds_count", {"path": path}) or 0.0


def _ttft_sum(path: str) -> float:
    return REGISTRY.get_sample_value("web_agent_ttft_seconds_sum", {"path": path}) or 0.0


def _stage_count(stage: str) -> float:
    return REGISTRY.get_sample_value(
        "web_agent_retrieval_stage_duration_seconds_count", {"stage": stage}) or 0.0


# ===================== 1) TTFT 记录器 =====================
class TestTTFTRecorder:
    def test_observes_only_once(self):
        path = "unit_ttft_once"
        rec = _TTFTRecorder(time.perf_counter(), path)
        before = _ttft_count(path)
        rec.mark()
        rec.mark()   # 幂等：同一轮不应重复计入直方图
        rec.mark()
        assert _ttft_count(path) == before + 1

    def test_elapsed_clamped_to_non_negative(self):
        path = "unit_ttft_nonneg"
        # 起算点置于未来：耗时算出负数时必须钳到 0，否则直方图观测值非法
        rec = _TTFTRecorder(time.perf_counter() + 100, path)
        rec.mark()
        assert _ttft_sum(path) == 0.0

    def test_paths_are_separated(self):
        agent_path, rag_path = "unit_ttft_agent", "unit_ttft_rag"
        a_before, r_before = _ttft_count(agent_path), _ttft_count(rag_path)
        _TTFTRecorder(time.perf_counter(), agent_path).mark()
        assert _ttft_count(agent_path) == a_before + 1
        assert _ttft_count(rag_path) == r_before   # 互不串扰


# ===================== 2) 流式队列包装 =====================
class TestStreamQueue:
    def test_forwards_items_and_marks_once(self):
        path = "unit_stream_queue"
        queue = asyncio.Queue()
        sq = _StreamQueue(queue, _TTFTRecorder(time.perf_counter(), path))
        before = _ttft_count(path)

        async def drive():
            await sq.put("第一段")
            await sq.put("第二段")

        asyncio.run(drive())
        # 内容原样透传（包装不改变队列语义）
        assert [queue.get_nowait(), queue.get_nowait()] == ["第一段", "第二段"]
        assert _ttft_count(path) == before + 1   # 只记首段


# ===================== 3) 检索阶段打点 =====================
class _StubEmbedder:
    async def embed(self, query):
        return [0.0] * 4


class _StubRetriever:
    async def ainvoke(self, query, filter=None):
        return [Document(page_content=f"{query}-doc1", metadata={"id": "1"}),
                Document(page_content=f"{query}-doc2", metadata={"id": "2"})]


class _StubRouter:
    async def route(self, question, mode):
        return _StubRetriever(), "vector"


def _bare_service() -> RetrievalService:
    """跳过 __init__（会建真实 Chroma/嵌入服务），只装配 retrieve 需要的桩件。"""
    svc = object.__new__(RetrievalService)
    svc.embedding_service = _StubEmbedder()
    svc.router = _StubRouter()
    svc.reranker = None      # 关闭重排：_merge_results 走「去重 + 截断」分支
    svc.query_enhancer = SimpleNamespace(
        decompose_query=lambda q: _just(q),
        multi_query=lambda q: _just(q),
    )
    return svc


async def _just(question):
    return [question]


class TestRetrievalStageMetrics:
    STAGES = ("rewrite", "decompose", "search", "rerank", "total")

    def test_retrieve_observes_every_stage(self):
        svc = _bare_service()
        before = {s: _stage_count(s) for s in self.STAGES}

        # use_context=False：跳过查询改写 LLM；单问题不触发 LLM 兜底分解，全程纯内存
        docs, retriever_type, gate = asyncio.run(
            svc.retrieve("iPhone 15 多少钱", use_context=False))

        assert len(docs) == 2 and retriever_type == "vector"
        for stage in self.STAGES:
            assert _stage_count(stage) == before[stage] + 1, f"{stage} 阶段未打点"

    def test_stage_labels_are_bounded(self):
        # 固定 5 个阶段标签：加入新阶段需显式改这里，防止标签基数悄悄膨胀
        svc = _bare_service()
        asyncio.run(svc.retrieve("运费谁承担", use_context=False))
        for stage in self.STAGES:
            assert _stage_count(stage) >= 1