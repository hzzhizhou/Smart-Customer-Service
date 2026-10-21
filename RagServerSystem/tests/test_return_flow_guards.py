"""退货主线两道闸门回归（对应实测 4 轮对话里的第 3、4 轮）。

1) 用户说「我不退货啊」必须能退出主线：DST 会把 return 当主线锁住，
   且规则里「退货」3.0 分压过任何其他意图，靠分类器让位永远不成立，
   因此取消闸门必须独立于意图判定。
2) DONE 之后用户再说「我要退货」，文案不得让用户以为本次已登记完成
   （旧文案是「您之前提交的退货申请已经登记完成」，用户还没给单号就"被完成"了）。
另附商品清单问法的规则覆盖：这批词此前整句零命中，在主线下直接补槽。
"""
import asyncio

from agent.dst.dst_manager import ConversationState
from agent.dst.slot_schema import Stage
from agent.intent_classifier import IntentClassifier
from service.handlers.return_refund import handle_return_refund


class _SaveStub:
    def __init__(self):
        self.saved = 0

    async def save(self, state):
        self.saved += 1


class _Ctx:
    """最小 TurnContext：取消/完成分支都不触达 slot_extractor 与检索。"""

    def __init__(self, question, stage, slots=None, intent="return"):
        self.question = question
        self.session_id = "s1"
        self.user = {"user_id": "u1"}
        self.dst = ConversationState(session_id="s1", intent=intent, stage=stage,
                                     slots=dict(slots or {}))
        self.dst_manager = _SaveStub()
        self.slot_extractor = None


def _collect(ctx) -> str:
    async def run():
        return "".join([piece async for piece in handle_return_refund(ctx)])
    return asyncio.run(run())


def test_cancel_hint_exits_return_mainline():
    ctx = _Ctx("我不退货啊", Stage.COLLECTING,
               {"order_id": "SO20241120001", "reason": "破损"})
    answer = _collect(ctx)
    assert "已取消本次退货申请" in answer
    assert ctx.dst.intent == "chat"
    assert ctx.dst.stage == Stage.COLLECTING
    assert ctx.dst.slots == {}
    assert ctx.dst_manager.saved == 1


def test_cancel_hint_also_works_while_confirming():
    ctx = _Ctx("算了先不办了", Stage.CONFIRMING,
               {"order_id": "SO20241120001", "reason": "破损"})
    answer = _collect(ctx)
    assert "已取消" in answer
    assert ctx.dst.intent == "chat"


def test_done_copy_does_not_claim_new_request_registered():
    ctx = _Ctx("我要退货", Stage.DONE,
               {"order_id": "SO20241120001", "reason": "破损"})
    answer = _collect(ctx)
    assert "上一笔退货申请已登记完成" in answer
    assert "尚未创建工单" in answer
    assert ctx.dst.stage == Stage.COLLECTING
    assert ctx.dst.slots == {}


def test_product_listing_question_scores_as_consult():
    scores = IntentClassifier(llm=None, mode="rule").rule_scores(
        "你们有哪几种产品在卖，推荐一下")
    assert scores.get("consult", 0.0) >= 2.0
