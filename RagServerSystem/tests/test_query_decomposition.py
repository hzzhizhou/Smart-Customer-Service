"""查询分解单元测试：多问题拆分器 + 多子问结果合并/门控（全部为纯函数，无 LLM、无 IO）。"""
import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from retrieval.service import RetrievalService, infer_doc_categories
from shared.constants import looks_like_multi_question, split_sub_questions


@pytest.mark.parametrize("question,expected", [
    ("iPhone15 电池容量多大？另外运费怎么算？",
     ["iPhone15 电池容量多大", "运费怎么算"]),
    ("华为 Mate60Pro 支持卫星通话吗？退货要几天到账？",
     ["华为 Mate60Pro 支持卫星通话吗", "退货要几天到账"]),
    ("退货流程是怎样的；保修多久", ["退货流程是怎样的", "保修多久"]),
    ("我想退货，还有运费谁出", ["我想退货", "运费谁出"]),
    ("iPhone 15 有降噪吗？以及续航多久", ["iPhone 15 有降噪吗", "续航多久"]),
    ("iPhone 15 多少钱\n支持花呗吗", ["iPhone 15 多少钱", "支持花呗吗"]),
])
def test_split_sub_questions_splits_multi(question, expected):
    assert split_sub_questions(question) == expected


@pytest.mark.parametrize("question", [
    "退货运费由谁承担",
    "退款多久到账",
    "我想退货",
    "你好",
    "这款手机有降噪吗，还有没有别的颜色",   # 「还有没有」不是新问题的分界
    "介绍一下你们的 iPhone 15",
])
def test_split_sub_questions_keeps_single(question):
    """拆不出 ≥2 个有效片段时必须原样返回单问，避免把一句话切碎、多打一次检索。"""
    assert split_sub_questions(question) == [question]


def test_split_sub_questions_empty():
    assert split_sub_questions("") == []
    assert split_sub_questions(None) == []


@pytest.mark.parametrize("question,expected", [
    ("电池多大？运费呢？", True),          # 多问号 → 值得让 LLM 兜底再拆一次
    ("运费怎么算", False),
    ("这款手机有降噪吗，还有没有别的颜色", False),
])
def test_looks_like_multi_question(question, expected):
    assert looks_like_multi_question(question) is expected


def _gate(level, top, escalate, reason="confident"):
    return {"top_score": top, "mean_score": top, "margin": 0.2,
            "level": level, "should_escalate": escalate, "reason": reason}


def test_combine_gates_takes_weakest():
    """强子问不得掩盖弱子问：整体档位/分数取最弱，转人工标记取 any。"""
    combined = RetrievalService._combine_gates([
        _gate("high", 0.91, False),
        _gate("low", 0.42, True, "below_threshold"),
    ])
    assert combined["level"] == "low"
    assert combined["top_score"] == 0.42
    assert combined["should_escalate"] is True
    assert combined["reason"] == "below_threshold"


def test_combine_gates_all_good_stays_high():
    combined = RetrievalService._combine_gates([
        _gate("high", 0.88, False), _gate("high", 0.85, False)])
    assert combined["level"] == "high"
    assert combined["should_escalate"] is False
    assert combined["top_score"] == 0.85


def _doc(name, content):
    return Document(page_content=content, metadata={"file_name": name})


def test_merge_sub_results_interleaves_and_dedups():
    """交错而非首尾相接：生成层按顺序截断上下文时，两个子问都能保住头部资料。"""
    a1, a2 = _doc("a.md", "A1"), _doc("a.md", "A2")
    b1, b2 = _doc("b.md", "B1"), _doc("b.md", "B2")
    merged = RetrievalService._merge_sub_results([[a1, a2], [b1, b2, a1]])
    assert [d.page_content for d in merged] == ["A1", "B1", "A2", "B2"]


# ---------- 检索预过滤：跨域问题取类别并集 ----------

@pytest.mark.parametrize("question,expected", [
    ("运费险能赔多少钱", ["product", "faq"]),        # 多少钱(product) + 运费(faq)
    ("手机多少钱，保修多久", ["phone", "product", "faq"]),
    ("退货运费由谁承担", ["faq"]),
    ("iPhone15 电池容量多大", ["phone"]),
    ("你好", []),
])
def test_infer_doc_categories_returns_all_matches(question, expected):
    """返回全部命中类别，而不是优先级最高的一个。

    回归防护：只留最高优先级会把答案所在的那一类整类排除——「运费险能赔多少钱」
    同时命中 product 与 faq，只留 product 时 平台FAQ(PLAT-013) 不在候选内，
    置信度被压到 0.53（low）→ 用户得到"没查到资料"。
    """
    assert infer_doc_categories(question) == expected


def test_infer_filter_union_and_passthrough():
    """跨域 → $in 并集；单类 → 等值；无命中 → 不过滤；显式 filter 优先。"""
    assert RetrievalService._infer_filter("运费险能赔多少钱", None, True) == \
        {"doc_category": {"$in": ["product", "faq"]}}
    assert RetrievalService._infer_filter("退货运费由谁承担", None, True) == \
        {"doc_category": "faq"}
    assert RetrievalService._infer_filter("你好", None, True) is None
    assert RetrievalService._infer_filter("运费险能赔多少钱", {"merchant_id": "ALL"}, True) == \
        {"merchant_id": "ALL"}


# ---------- 多子问门控：答不上来的子问不拖累能答的子问 ----------

class _SubQRetrieval(RetrievalService):
    """桩：按子问返回预置文档，绕开真实检索/路由/rerank。"""

    def __init__(self, docs_by_query):
        self.docs_by_query = docs_by_query
        self.router = SimpleNamespace(route=self._route)

    @staticmethod
    async def _route(question, mode):
        return SimpleNamespace(), "hybrid_retriever"

    async def _search_and_rerank(self, queries, retriever, rerank_query, filter_arg):
        # 契约：(重排后的 TOP-N, 重排前的召回候选)。本桩不重排，两者相同。
        docs = list(self.docs_by_query.get(queries[0], []))
        return docs, docs


def _scored_doc(content, score):
    return Document(page_content=content,
                    metadata={"file_name": "f.md", "vector_score": score})


def test_sub_question_gate_drops_unanswerable_sub_question():
    """一个子问答不上来，不能把能答的子问一起拖去转人工。

    回归防护：「你们有哪些在售手机？iPhone 15 多少钱？」此前整体取最弱子问（0.54→low），
    连能答的「iPhone 15 4999 元」也一并放弃，整轮回"没查到可靠的资料"。
    """
    svc = _SubQRetrieval({
        "你们有哪些在售手机": [_scored_doc("手机目录", 0.53)],
        "iPhone 15 多少钱": [_scored_doc("iPhone 15 售价 4999 元", 0.72)],
    })
    docs, _, gate = asyncio.run(svc._retrieve_per_sub_question(
        ["你们有哪些在售手机", "iPhone 15 多少钱"], "rule", None, False))
    assert gate["level"] == "medium"          # 只按能答的子问判定
    assert gate["should_escalate"] is False
    assert [d.page_content for d in docs] == ["iPhone 15 售价 4999 元"]


def test_sub_question_gate_all_low_still_escalates():
    """全部子问都答不上来 → 仍整体转人工，剔除逻辑不能把该转的漏掉。"""
    svc = _SubQRetrieval({
        "甲": [_scored_doc("无关", 0.40)],
        "乙": [_scored_doc("无关二", 0.42)],
    })
    _, _, gate = asyncio.run(svc._retrieve_per_sub_question(
        ["甲", "乙"], "rule", None, False))
    assert gate["level"] == "low"
    assert gate["should_escalate"] is True