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

    def merge_new_slots(self, dst, new_slots):
        dst.slots.update(new_slots)
        return dst


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


def test_fabricated_order_rejected_at_slot_fill(monkeypatch):
    """虚构订单号必须在填充阶段就被退回，不得进入确认环节。

    实测（2026-10-21）："订单号1233423112" 曾一路走到「已为您核对退货信息：
    回复"确认"即可提交」——填充期只拦归属（forbidden），not_found 全放行。
    修复后：拿到订单号立即全量校验（存在性/归属/可退），不通过当场清槽退回。
    """
    import agent.dst.order_validator as ov

    def _fake_validate(order_id, owner_id=""):
        return {"valid": False, "status": "not_found", "order_id": order_id,
                "status_raw": None,
                "message": f"很抱歉，未查询到订单号 **{order_id}** 的订单，"
                           f"请核对后重新提供正确的订单号。"}

    monkeypatch.setattr(ov, "validate_return_order", _fake_validate)

    class _Extractor:
        async def extract(self, question, intent, slots):
            return {"order_id": "1233423112"}

    ctx = _Ctx("订单号1233423112", Stage.COLLECTING)
    ctx.slot_extractor = _Extractor()
    answer = _collect(ctx)
    assert "未查询到订单号" in answer
    assert "已为您核对" not in answer          # 不得再进入确认话术
    assert "order_id" not in ctx.dst.slots      # 无效单号被清掉
    assert ctx.dst.stage == Stage.COLLECTING    # 阶段退回继续收集
    assert ctx.dst_manager.saved == 1


def test_foreign_order_rejected_at_slot_fill(monkeypatch):
    """他人订单同样在填充阶段拦截（该分支此前已存在，锁住防回归）。"""
    import agent.dst.order_validator as ov

    def _fake_validate(order_id, owner_id=""):
        return {"valid": False, "status": "forbidden", "order_id": order_id,
                "status_raw": "paid",
                "message": f"订单 **{order_id}** 不在当前登录账号名下，"
                           f"无法为您办理退货/退款。"}

    monkeypatch.setattr(ov, "validate_return_order", _fake_validate)

    class _Extractor:
        async def extract(self, question, intent, slots):
            return {"order_id": "SO20241120001"}

    ctx = _Ctx("订单号SO20241120001", Stage.COLLECTING)
    ctx.slot_extractor = _Extractor()
    answer = _collect(ctx)
    assert "不在当前登录账号名下" in answer
    assert "order_id" not in ctx.dst.slots
    assert ctx.dst.stage == Stage.COLLECTING
