"""业务观测指标测试：兜底率 / 缓存命中率 / 意图分布 / 转人工率。

锁住四件事：
  1) agent_fallback_total：知识兜底的两条路径（置信度不足、无产出）各按 reason 计数；
  2) cache_requests_total：响应缓存与 embedding 缓存各自 hit/miss 计数（过期算 miss）；
  3) intent_total：意图识别结果落进对应 intent 标签；
  4) handoff_total：用户主动 / 投诉 / Agent 工具三条转人工来源分别计数，复用不重复计。

约定：与项目其它测试一致，同步用例内用 asyncio.run(drive()) 驱动异步逻辑。
"""
import asyncio

from prometheus_client import REGISTRY

from service.cache import ResponseCache
from service.handlers.knowledge import stream_knowledge_answer
from service.turn_service import AgentTurnService
from infrastructure.EmbeddingService.embedding_service import EmbeddingService


def _counter(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ===================== 1) 兜底率 =====================
class _StubRetrieval:
    def __init__(self, docs, gate):
        self._docs, self._gate = docs, gate

    async def retrieve(self, **kwargs):
        return self._docs, "vector", self._gate


class _EmptyAnswerGen:
    """stream_generate 一个块都不产出（模拟模型空回复）。"""

    async def stream_generate(self, question, docs, session_id):
        if False:            # 保持异步生成器语义，但永不 yield
            yield ""


class TestFallbackMetrics:
    def test_low_confidence_counts(self):
        svc = _StubRetrieval(docs=[], gate={"should_escalate": True, "level": "low"})
        before = _counter("web_agent_fallback_total", {"reason": "low_confidence"})

        async def drive():
            return [p async for p in stream_knowledge_answer(
                svc, _EmptyAnswerGen(), "平台外问题", "s1")]

        pieces = asyncio.run(drive())
        assert pieces and "人工" in "".join(pieces)          # 走的是转人工兜底话术
        assert _counter("web_agent_fallback_total",
                        {"reason": "low_confidence"}) == before + 1

    def test_no_output_counts_as_empty_answer(self):
        # 检索命中、置信度足够，但生成器一个块都没产出 → 空答复兜底
        svc = _StubRetrieval(docs=[object()], gate={"should_escalate": False, "level": "high"})
        before = _counter("web_agent_fallback_total", {"reason": "empty_answer"})

        async def drive():
            return [p async for p in stream_knowledge_answer(
                svc, _EmptyAnswerGen(), "正常问题", "s1")]

        asyncio.run(drive())
        assert _counter("web_agent_fallback_total",
                        {"reason": "empty_answer"}) == before + 1

    def test_reasons_are_bounded(self):
        # 兜底原因只有固定 5 个，防止业务代码里写错标签悄悄膨胀基数
        import utils.metrics as m
        assert set(m.agent_fallback_total._labelnames) == {"reason"}


# ===================== 2) 缓存命中率 =====================
class TestResponseCacheMetrics:
    def test_hit_and_miss_counted(self):
        cache = ResponseCache(maxsize=10, ttl=100)
        hit_before = _counter("web_agent_cache_requests_total",
                              {"cache": "response", "result": "hit"})
        miss_before = _counter("web_agent_cache_requests_total",
                               {"cache": "response", "result": "miss"})

        assert cache.get("unit_key_missing") is None          # 从未写入 → miss
        cache.set("unit_key", {"answer": "hi"})
        assert cache.get("unit_key") == {"answer": "hi"}      # 命中 → hit

        assert _counter("web_agent_cache_requests_total",
                        {"cache": "response", "result": "miss"}) == miss_before + 1
        assert _counter("web_agent_cache_requests_total",
                        {"cache": "response", "result": "hit"}) == hit_before + 1

    def test_expired_entry_counts_as_miss(self):
        cache = ResponseCache(maxsize=10, ttl=100)
        miss_before = _counter("web_agent_cache_requests_total",
                               {"cache": "response", "result": "miss"})
        cache.set("unit_key_expired", {"answer": "hi"})
        # 手动把写入时间戳改到很久之前：模拟「已失效但还没被淘汰」的条目，
        # 它必须算 miss（真实检索会被触发），不能算命中虚高命中率
        cache._store["unit_key_expired"] = (0.0, {"answer": "hi"})
        assert cache.get("unit_key_expired") is None
        assert _counter("web_agent_cache_requests_total",
                        {"cache": "response", "result": "miss"}) == miss_before + 1


class _StubEmbeddingModel:
    def embed_query(self, query):
        return [0.1] * 4


class TestEmbeddingCacheMetrics:
    def test_second_call_hits(self):
        svc = EmbeddingService(_StubEmbeddingModel())
        hit_before = _counter("web_agent_cache_requests_total",
                              {"cache": "embedding", "result": "hit"})
        miss_before = _counter("web_agent_cache_requests_total",
                               {"cache": "embedding", "result": "miss"})

        async def drive():
            first = await svc.embed("同一个问题")
            second = await svc.embed("同一个问题")
            return first, second

        first, second = asyncio.run(drive())
        assert first == second                                # 第二次直接命中，结果一致
        assert _counter("web_agent_cache_requests_total",
                        {"cache": "embedding", "result": "miss"}) == miss_before + 1
        assert _counter("web_agent_cache_requests_total",
                        {"cache": "embedding", "result": "hit"}) == hit_before + 1


# ===================== 3) 意图分布 =====================
class _StubIntentClassifier:
    async def classify(self, question, force_intent=None):
        return {"intent": "unit_intent_probe", "intent_cn": "单测探针",
                "confidence": 0.9, "primary_tool": "search_knowledge",
                "source": "rule"}


class TestIntentMetrics:
    def test_classify_result_counted(self):
        svc = object.__new__(AgentTurnService)               # 只装 classify 需要的桩件
        svc.intent_classifier = _StubIntentClassifier()
        before = _counter("web_agent_intent_total", {"intent": "unit_intent_probe"})

        async def drive():
            # dst=None：不触发规则打分/主线让位逻辑，只走 classify 与埋点
            return await svc._enrich_messages_with_intent([("user", "你好")], "你好")

        _messages, info = asyncio.run(drive())
        assert info["intent"] == "unit_intent_probe"
        assert _counter("web_agent_intent_total",
                        {"intent": "unit_intent_probe"}) == before + 1


# ===================== 4) 转人工率 =====================
def _fake_ticket(category: str) -> dict:
    return {"ticket_id": "TK-UNIT-1", "user_id": "u1", "status": "open",
            "priority": "normal", "category": category,
            "created_at": "2026-01-01 00:00:00"}


class TestHandoffMetrics:
    def test_user_request_new_session_counted(self, monkeypatch):
        import service.handoff_service as hs
        monkeypatch.setattr(hs.mysql_store, "get_handoff_ticket", lambda uid: None)
        monkeypatch.setattr(hs.mysql_store, "add_ticket_message", lambda *a, **k: {})
        monkeypatch.setattr(hs, "create_ticket_record", lambda **k: _fake_ticket("consult"))
        before = _counter("web_agent_handoff_total", {"reason": "user_request"})

        ticket, reused = hs.ensure_handoff_session("u1", "我要找人工")

        assert reused is False and ticket["ticket_id"] == "TK-UNIT-1"
        assert _counter("web_agent_handoff_total",
                        {"reason": "user_request"}) == before + 1

    def test_user_request_reuse_not_counted(self, monkeypatch):
        import service.handoff_service as hs
        monkeypatch.setattr(hs.mysql_store, "get_handoff_ticket",
                            lambda uid: _fake_ticket("consult"))
        before = _counter("web_agent_handoff_total", {"reason": "user_request"})

        _ticket, reused = hs.ensure_handoff_session("u1")

        assert reused is True                                 # 复用既有会话不重复计数
        assert _counter("web_agent_handoff_total",
                        {"reason": "user_request"}) == before

    def test_agent_tool_counted(self, monkeypatch, tool_identity):
        import agent.tools.ticket_tool as tt
        import service.handoff_service as hs
        monkeypatch.setattr(tt, "create_ticket_record", lambda *a, **k: _fake_ticket("other"))
        # 新会话：start_handoff_session 返回开场白记录，才计一次 agent_tool
        monkeypatch.setattr(hs, "start_handoff_session", lambda tid: {"msg_id": 1})
        before = _counter("web_agent_handoff_total", {"reason": "agent_tool"})

        # 账号由服务端上下文注入（工具参数里已无 user_id）
        tool_identity("u1")
        tt.create_ticket.invoke({"description": "明天天气怎么样"})

        assert _counter("web_agent_handoff_total",
                        {"reason": "agent_tool"}) == before + 1

    def test_agent_tool_reused_session_not_counted(self, monkeypatch, tool_identity):
        import agent.tools.ticket_tool as tt
        import service.handoff_service as hs
        monkeypatch.setattr(tt, "create_ticket_record", lambda *a, **k: _fake_ticket("other"))
        # 幂等命中已有会话：没有写新开场白（None），不应把转人工率灌水
        monkeypatch.setattr(hs, "start_handoff_session", lambda tid: None)
        before = _counter("web_agent_handoff_total", {"reason": "agent_tool"})

        tool_identity("u1")
        tt.create_ticket.invoke({"description": "明天天气怎么样"})

        assert _counter("web_agent_handoff_total",
                        {"reason": "agent_tool"}) == before

    def test_reasons_are_bounded(self):
        import utils.metrics as m
        assert set(m.handoff_total._labelnames) == {"reason"}