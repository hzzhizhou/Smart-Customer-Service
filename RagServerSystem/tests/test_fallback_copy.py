"""兜底话术不得空口承诺转人工（实测 21:08 的知识问答 SSL 异常）。

异常/空答复分支并不建单，前端靠轮询 /handoff/active 切换人工模式，
拿不到工单就等于“已转接”但无人接入。兜底只能如实告知，并询问是否转人工。
"""
import asyncio

from shared.reply_templates import AGENT_NO_ANSWER_FALLBACK
from service.handlers.return_refund import handle_return_refund


def test_agent_fallback_does_not_claim_handoff():
    assert "已为您转接" not in AGENT_NO_ANSWER_FALLBACK
    assert "知识库" in AGENT_NO_ANSWER_FALLBACK
    assert "转人工" in AGENT_NO_ANSWER_FALLBACK


def test_return_exception_copy_is_honest():
    class _Ctx:
        question = "我要退货"
        session_id = "s1"
        user = {"user_id": "u1"}
        dst = None           # 触发主流程异常
        dst_manager = None

    async def run():
        return "".join([p async for p in handle_return_refund(_Ctx())])

    answer = asyncio.run(run())
    assert "我这就转人工" not in answer
    assert "转人工" in answer
    assert "暂时无法确认" in answer
