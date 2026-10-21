"""F 批回归测试：RAG 正确性四项（BM25 $in / 门控前置 / 分块上限 / auto_filter 统一）。

背景（PROJECT_AUDIT §4.2(1)(2)(3)(4)）：
  (1) 多类别预过滤构造 `{"doc_category": {"$in": [...]}}`，而 BM25 只做等值比较 →
      BM25 路被整个清空，混合检索静默退化成纯向量（日志上仍写 hybrid）。
  (2) 置信度门控取的是「重排后 TOP5」的 vector_score；重排把向量分最高的文档挤出后，
      门控看到的分不再代表召回置信度；BM25-only 文档没有 vector_score，默认为 0。
  (3) CHUNK_SIZE=768 / TABLE_CHUNK_SIZE=1000 在切回本地 bge（512 token）时会被静默截断。
  (4) auto_filter 在 /rag/stream 与 Agent 两条入口不一致，召回范围不同、评测不可比。

约定：门控与过滤都是纯函数式逻辑，用 object.__new__ 直接造服务壳 + 桩件即可，
不起真检索（不连向量库 / 不调模型）。
"""
import asyncio
import inspect

import pytest
from langchain_core.documents import Document

import config.settings as settings
from retrieval.filters import matches_filter
from retrieval.service import RetrievalService


def _doc(tag: str, **meta) -> Document:
    base = {"file_name": f"{tag}.md", "id": tag}
    base.update(meta)
    return Document(page_content=f"内容-{tag}", metadata=base)


# ===================== F1：BM25 支持 $in =====================
class TestMatchesFilter:
    def test_plain_equality(self):
        assert matches_filter({"doc_category": "faq"}, {"doc_category": "faq"})
        assert not matches_filter({"doc_category": "faq"}, {"doc_category": "phone"})

    def test_dollar_in(self):
        """多类别预过滤用的就是 $in —— BM25 路必须认，否则整路被清空。"""
        cond = {"doc_category": {"$in": ["phone", "product"]}}
        assert matches_filter({"doc_category": "phone"}, cond)
        assert matches_filter({"doc_category": "product"}, cond)
        assert not matches_filter({"doc_category": "faq"}, cond)

    def test_ne_and_eq_operators(self):
        assert matches_filter({"x": "a"}, {"x": {"$ne": "b"}})
        assert not matches_filter({"x": "a"}, {"x": {"$ne": "a"}})
        assert matches_filter({"x": "a"}, {"x": {"$eq": "a"}})

    def test_missing_key_does_not_match(self):
        assert not matches_filter({}, {"doc_category": {"$in": ["phone"]}})

    def test_empty_condition_passes(self):
        assert matches_filter({"a": 1}, None)
        assert matches_filter({"a": 1}, {})

    def test_unknown_operator_fails_closed(self):
        """不认识的算子不能当「全通过」（那会静默放大召回范围）。"""
        assert not matches_filter({"doc_category": "faq"}, {"doc_category": {"$regex": ".*"}})

    def test_multi_condition_all_must_match(self):
        cond = {"doc_category": {"$in": ["phone"]}, "layer": "public"}
        assert matches_filter({"doc_category": "phone", "layer": "public"}, cond)
        assert not matches_filter({"doc_category": "phone", "layer": "private"}, cond)

    def test_infer_filter_multi_category_is_dollar_in(self):
        """多类别问题必须产出 $in 形式，且该条件能被 matches_filter 接受。"""
        # 「iPhone 15 多少钱」同时命中 phone(iphone) 与 product(多少钱)
        where = RetrievalService._infer_filter("iPhone 15 多少钱", None, True)
        assert where == {"doc_category": {"$in": ["phone", "product"]}}
        assert matches_filter({"doc_category": "phone"}, where)
        assert not matches_filter({"doc_category": "faq"}, where)

    def test_infer_filter_single_category_is_equality(self):
        where = RetrievalService._infer_filter("退货流程怎么走", None, True)
        assert where == {"doc_category": "faq"}

    def test_auto_filter_off_keeps_no_filter(self):
        assert RetrievalService._infer_filter("iPhone 15 多少钱", None, False) is None


# ===================== F2：门控用重排前的召回候选 =====================
class _StubRouter:
    async def route(self, question, mode):
        return "stub-retriever", "hybrid_retriever"


class TestGateUsesRecallCandidates:
    def _svc(self, monkeypatch, merged, candidates):
        svc = object.__new__(RetrievalService)
        svc.router = _StubRouter()

        async def _fake_search(queries, retriever, rerank_query, filter_arg):
            return merged, candidates

        monkeypatch.setattr(svc, "_search_and_rerank", _fake_search)
        return svc

    def test_gate_sees_doc_pushed_out_by_rerank(self, monkeypatch):
        """重排把向量分最高的文档挤出 TOP5，门控仍按召回候选判 high。"""
        strong = _doc("strong", vector_score=0.90)
        weak = _doc("weak", vector_score=0.20)
        svc = self._svc(monkeypatch, merged=[weak], candidates=[strong, weak])

        docs, _rtype, gate = asyncio.run(
            svc._retrieve_one("q", "q", "rule", None, False, False, False))

        assert docs == [weak]                 # 返回给生成层的仍是重排后的 TOP-N
        assert gate["top_score"] == 0.9       # 但门控看的是召回候选
        assert gate["level"] == "high"
        assert gate["should_escalate"] is False

    def test_bm25_only_top_doc_does_not_zero_the_gate(self, monkeypatch):
        """BM25-only 文档排在第一位时，不能因为缺 vector_score 就判 low 转人工。"""
        vec = _doc("vec", vector_score=0.85)
        bm_only = _doc("bm25only", bm25_score=9.0)      # 无 vector_score
        svc = self._svc(monkeypatch, merged=[bm_only], candidates=[vec, bm_only])

        _docs, _rtype, gate = asyncio.run(
            svc._retrieve_one("q", "q", "rule", None, False, False, False))

        assert gate["top_score"] == 0.85
        assert gate["level"] == "high"

    def test_empty_candidates_still_escalates(self, monkeypatch):
        svc = self._svc(monkeypatch, merged=[], candidates=[])

        _docs, _rtype, gate = asyncio.run(
            svc._retrieve_one("q", "q", "rule", None, False, False, False))

        assert gate["level"] == "low" and gate["should_escalate"] is True

    def test_gate_ignores_rerank_only_change(self, monkeypatch):
        """同一批召回候选，重排顺序变化不影响门控结果（两者解耦）。"""
        a, b = _doc("a", vector_score=0.90), _doc("b", vector_score=0.60)
        g1 = asyncio.run(self._svc(monkeypatch, [a, b], [a, b])._retrieve_one(
            "q", "q", "rule", None, False, False, False))[2]
        g2 = asyncio.run(self._svc(monkeypatch, [b, a], [b, a])._retrieve_one(
            "q", "q", "rule", None, False, False, False))[2]
        assert g1 == g2


# ===================== F3a：分块长度 ≤ 嵌入模型上限 =====================
class TestChunkSizeGuard:
    def test_warns_when_chunk_exceeds_model_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "CHUNK_SIZE", 768)
        monkeypatch.setattr(settings, "TABLE_CHUNK_SIZE", 1000)
        monkeypatch.setattr(settings, "EMBEDDING_MODEL_MAX_TOKENS", 512)
        monkeypatch.setattr(settings, "CHUNK_PREFIX_OVERHEAD", 60)

        warnings = settings.validate_chunk_config()

        joined = " ".join(warnings)
        assert "CHUNK_SIZE" in joined and "512" in joined

    def test_no_warning_when_within_limit(self, monkeypatch):
        monkeypatch.setattr(settings, "CHUNK_SIZE", 400)
        monkeypatch.setattr(settings, "TABLE_CHUNK_SIZE", 400)
        monkeypatch.setattr(settings, "PARENT_CHUNK_SIZE", 400)
        monkeypatch.setattr(settings, "EMBEDDING_MODEL_MAX_TOKENS", 8192)
        monkeypatch.setattr(settings, "CHUNK_PREFIX_OVERHEAD", 60)

        assert settings.validate_chunk_config() == []

    def test_prefix_overhead_counts(self, monkeypatch):
        """标题前缀也算进块长度 —— 只比 CHUNK_SIZE 会漏判。"""
        monkeypatch.setattr(settings, "CHUNK_SIZE", 500)
        monkeypatch.setattr(settings, "TABLE_CHUNK_SIZE", 500)
        monkeypatch.setattr(settings, "PARENT_CHUNK_SIZE", 500)
        monkeypatch.setattr(settings, "EMBEDDING_MODEL_MAX_TOKENS", 512)
        monkeypatch.setattr(settings, "CHUNK_PREFIX_OVERHEAD", 60)

        assert settings.validate_chunk_config()      # 500+60 > 512 → 告警

    def test_current_config_is_consistent(self):
        """默认配置（dashscope 8k）当下必须是自洽的，不能一启动就告警。"""
        assert settings.validate_chunk_config() == []


# ===================== F4：auto_filter 单一策略 =====================
class TestAutoFilterUnified:
    def test_default_comes_from_settings(self):
        assert settings.AUTO_FILTER_DEFAULT is True
        default = inspect.signature(
            RetrievalService.retrieve).parameters["auto_filter"].default
        assert default is settings.AUTO_FILTER_DEFAULT

    def test_entry_points_do_not_override_with_false(self):
        """三条入口都不能再各写各的 auto_filter（含 False 的分叉）。"""
        import agent.tools.search_knowledge as sk
        import service.handlers.knowledge as kh
        import service.turn.agent_turn as agent_turn
        import service.turn.rag_turn as rag_turn

        for mod in (sk, kh, agent_turn, rag_turn):
            src = inspect.getsource(mod)
            assert "auto_filter=False" not in src, mod.__name__
