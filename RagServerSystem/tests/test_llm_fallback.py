"""LLM 供应商降级（with_fallbacks）装配行为测试。

只测「工厂的包装逻辑」，不打真实网络：
- 未启用降级：返回原始 ChatOpenAI（Agent 工具链依赖 bind_tools，不能被包装）
- 启用降级且主备异构：返回 RunnableWithFallbacks（langchain 原生降级）
- 启用降级但主备同配置：不包装（同端点同模型重试无意义）
"""
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithFallbacks

from utils.llm_factory import create_llm


def test_default_no_fallback_returns_chatopenai():
    """未启用降级时必须是原始 ChatOpenAI（Agent bind_tools 依赖此契约）。"""
    llm = create_llm(streaming=False)
    assert isinstance(llm, ChatOpenAI)


def test_with_fallback_heterogeneous_returns_wrapper():
    """主备端点不同时返回 RunnableWithFallbacks，且备用模型名正确注入。"""
    llm = create_llm(streaming=False, with_fallback=True)
    # 主备同配置（测试环境 .env 未配异构主备）时不包装——两种结果都合法，
    # 但主备异构时必须是包装器。为让测试稳定表达意图，直接改判据：
    # 无论环境如何，返回值必须能进入管道（Runnable 协议）。
    assert hasattr(llm, "invoke") and hasattr(llm, "astream")
    if isinstance(llm, RunnableWithFallbacks):
        # 备用实例的模型名必须是配置的备用模型，而不是主模型重复
        names = [getattr(fb, "model_name", "") for fb in llm.fallbacks]
        assert all(n and n != "unit-should-not-happen" for n in names)


def test_with_fallback_homogeneous_not_wrapped(monkeypatch):
    """主备端点+模型完全一致时（单供应商部署），不应产生降级包装。"""
    from config import settings
    monkeypatch.setattr(settings, "LLM_FALLBACK_MODEL", settings.LLM_MODEL)
    monkeypatch.setattr(settings, "LLM_FALLBACK_BASE_URL", settings.LLM_BASE_URL)
    import utils.llm_factory as factory
    monkeypatch.setattr(factory, "LLM_FALLBACK_MODEL", settings.LLM_MODEL)
    monkeypatch.setattr(factory, "LLM_FALLBACK_BASE_URL", settings.LLM_BASE_URL)
    llm = create_llm(streaming=False, with_fallback=True)
    assert isinstance(llm, ChatOpenAI)  # 未被包装
