"""AI 自动转人工必须「开启人工会话」的回归测试。

背景：客服工作台以「工单有没有 ticket_messages」判定一场人工会话（见
access/routes/handoff.py 的判据）。`create_ticket` 工具原先建完单只返回话术，
没有写开场白，于是 AI 自己转人工的工单只进工单列表、不进工作台，
客服根本看不到 —— 也导致用户端的「人工客服中」状态判断不成立。

这两条用例把这个缺口钉住：能开人工会话的分类必须开，带阶段流转的退货/退款不能开
（它们的 stage 进度查询会被会话消息污染）。
"""
import agent.tools.ticket_tool as ticket_tool
import service.handoff_service as handoff_service


def _fake_ticket(category: str) -> dict:
    return {
        "ticket_id": "TK-AI-1",
        "user_id": "u1",
        "status": "open",
        "priority": "normal",
        "category": category,
        "created_at": "2026-01-01 00:00:00",
    }


def test_create_ticket_tool_opens_handoff_session(monkeypatch, tool_identity):
    """超范围问题（分类 other）建单后，必须开启人工会话让工作台可见。"""
    opened = []
    monkeypatch.setattr(ticket_tool, "create_ticket_record",
                        lambda *a, **k: _fake_ticket("other"))
    monkeypatch.setattr(handoff_service, "start_handoff_session",
                        lambda tid: opened.append(tid))

    tool_identity("u1")
    ticket_tool.create_ticket.invoke({"description": "明天天气怎么样"})

    assert opened == ["TK-AI-1"]


def test_create_ticket_tool_skips_handoff_for_return(monkeypatch, tool_identity):
    """退货/退款有 stage 阶段流转，不能当成人工会话开启。"""
    opened = []
    monkeypatch.setattr(ticket_tool, "create_ticket_record",
                        lambda *a, **k: _fake_ticket("return"))
    monkeypatch.setattr(handoff_service, "start_handoff_session",
                        lambda tid: opened.append(tid))

    tool_identity("u1")
    ticket_tool.create_ticket.invoke({"description": "我要退货"})

    assert opened == []
