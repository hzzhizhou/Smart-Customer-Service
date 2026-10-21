"""B3 回归测试：建单幂等（唯一键）与工单进度按业务定位。

背景（PROJECT_AUDIT §6.2(1)(2)）：
  (1) 工单进度此前只按 user_id/order_id/描述 LIKE 取「最近一张」。用户先建退货单、
      后建投诉单，再问「退货办到哪了」会拿到投诉单 —— 跨业务串单。
  (2) 防重复是「先查后写」：两个并发请求（双击确认、客户端重试）会同时查到
      「还没有在办工单」，然后各建一张 —— 检查在并发下形同虚设。

本组用例把修复后的四条性质钉住：
  1) 幂等键只由「账号 + 订单 + 分类 + 时间窗」决定，窗口内稳定可复现；
  2) 连续两次建单用的是同一个幂等键（去重交给数据库唯一键，而非先查后写）；
  3) 唯一键冲突时返回「已有工单」而不是报错，调用方按 ticket_id 继续；
  4) 进度查询把业务分类收进 SQL，问法能确定性推出分类，handler 会把它传给工具。

约定：与项目其它测试一致，只 monkeypatch 存储层的 `_tx`（假 cursor），不连真库。
"""
import asyncio
from contextlib import contextmanager
from datetime import datetime

import pymysql
import pytest

import agent.tools.ticket_tool as tt
import infrastructure.db.connection as db_connection
import infrastructure.db.tickets as db_tickets
import infrastructure.mysql_store as ms
from service.handlers.progress import handle_ticket_progress, infer_business_category
from service.handlers.context import TurnContext


# ===================== 假 cursor / 假事务 =====================
class _RecordingCursor:
    """记录 execute 的 SQL 与参数，并按需返回一行。"""

    def __init__(self, row=None):
        self.row = row
        self.executed = []          # [(sql, params), ...]

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.row

    def fetchall(self):
        return [self.row] if self.row else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_tx(monkeypatch, cursor):
    """把 connection._tx 换成「直接产出假 cursor」的事务，避免连真库。"""

    @contextmanager
    def _fake_tx():
        yield cursor

    monkeypatch.setattr(db_connection, "_tx", _fake_tx)


def _ticket(ticket_id="TK-EXISTING", **over):
    base = {
        "ticket_id": ticket_id, "user_id": "u1", "order_id": "SO20241120005",
        "description": "退货", "priority": "normal", "category": "return",
        "status": "open", "created_at": "2026-01-01 00:00:00", "note": "",
        "stage": "submitted", "stage_history": "[]", "stage_updated_at": "",
    }
    base.update(over)
    return base


# ===================== 1) 幂等键的可复现性 =====================
class TestIdempotencyKey:
    def test_same_business_same_key(self):
        """同账号 + 同订单 + 同分类 + 同时间窗 → 同一个键（并发去重的依据）。"""
        now = datetime(2026, 1, 1, 12, 0, 0)
        a = tt._idempotency_key("u1", "SO20241120005", "return", now)
        b = tt._idempotency_key("u1", "SO20241120005", "return", now)
        assert a == b

    @pytest.mark.parametrize("field,value", [
        ("user_id", "u2"),          # 换账号 = 另一件申请
        ("order_id", "SO20241121006"),
        ("category", "refund"),     # 退货与退款是两件业务
    ])
    def test_any_dimension_change_gives_new_key(self, field, value):
        now = datetime(2026, 1, 1, 12, 0, 0)
        args = {"user_id": "u1", "order_id": "SO20241120005", "category": "return"}
        args[field] = value
        base = tt._idempotency_key("u1", "SO20241120005", "return", now)
        assert tt._idempotency_key(**args, now=now) != base

    def test_key_changes_outside_window(self, monkeypatch):
        """跨出时间窗的再次提交要能建新单，不能把该业务永久锁死。"""
        monkeypatch.setattr(tt, "TICKET_IDEMPOTENCY_WINDOW", 300)
        early = datetime(2026, 1, 1, 12, 0, 0)
        late = datetime(2026, 1, 1, 13, 0, 0)      # 一小时后再申请
        assert tt._idempotency_key("u1", "SO20241120005", "return", early) != \
            tt._idempotency_key("u1", "SO20241120005", "return", late)

    def test_empty_order_is_still_deduplicated(self):
        """无订单号的工单也要能去重（同一句「我要投诉」双击不能建两张）。"""
        now = datetime(2026, 1, 1, 12, 0, 0)
        assert tt._idempotency_key("u1", "", "complaint", now) == \
            tt._idempotency_key("u1", "", "complaint", now)


# ===================== 2) 两次建单共用同一个幂等键 =====================
class TestCreateTicketRecordUsesKey:
    def test_two_calls_persist_same_key(self, monkeypatch):
        """连续两次建单传给存储层的幂等键一致 —— 去重责任落在数据库唯一键上。"""
        monkeypatch.setattr(tt, "TICKET_IDEMPOTENCY_WINDOW", 3600)   # 排除跨窗边界抖动
        seen_keys = []

        def _fake_persist(ticket_id, user_id, order_id, description, priority,
                          category, note="", stage="", idempotency_key=""):
            seen_keys.append(idempotency_key)
            return _ticket(ticket_id, user_id=user_id, category=category, stage=stage)

        monkeypatch.setattr(tt, "_persist_ticket", _fake_persist)
        tt.create_ticket_record("u1", "我要退货", "normal", "return", "SO20241120005")
        tt.create_ticket_record("u1", "我要退货", "normal", "return", "SO20241120005")

        assert len(seen_keys) == 2
        assert seen_keys[0] and seen_keys[0] == seen_keys[1]

    def test_reused_ticket_is_returned_unchanged(self, monkeypatch):
        """存储层返回已存在的工单（不同 ticket_id）时，原样透传，不报错、不覆盖。"""
        monkeypatch.setattr(
            tt, "_persist_ticket",
            lambda *a, **k: _ticket("TK-EXISTING", user_id="u1", category="return"))

        out = tt.create_ticket_record("u1", "我要退货", "normal", "return", "SO20241120005")

        assert out["ticket_id"] == "TK-EXISTING"      # 拿到的是已有工单，而非新号


# ===================== 3) 唯一键冲突 → 返回已有工单 =====================
class TestUniqueKeyCollision:
    def _colliding_cursor(self):
        class _C(_RecordingCursor):
            def execute(self, sql, params=None):
                super().execute(sql, params)
                raise pymysql.err.IntegrityError(
                    1062, "Duplicate entry for key 'uk_tickets_idem'")
        return _C()

    def test_collision_returns_existing(self, monkeypatch):
        """并发下第二个 INSERT 撞唯一键：不能抛给调用方，要返回已有工单。"""
        _patch_tx(monkeypatch, self._colliding_cursor())
        monkeypatch.setattr(db_tickets, "get_ticket_by_idempotency_key",
                            lambda key: _ticket("TK-WINNER", user_id="u1"))

        out = ms.create_ticket("TK-LOSER", "u1", "SO20241120005", "退货", "normal",
                               "return", stage="submitted", idempotency_key="k-abc")

        assert out["ticket_id"] == "TK-WINNER"

    def test_non_idempotency_collision_still_raises(self, monkeypatch):
        """撞的不是幂等键（如 ticket_id 撞车）时照常抛错，不掩盖真实冲突。"""
        _patch_tx(monkeypatch, self._colliding_cursor())
        monkeypatch.setattr(db_tickets, "get_ticket_by_idempotency_key", lambda key: None)

        with pytest.raises(pymysql.err.IntegrityError):
            ms.create_ticket("TK-LOSER", "u1", "SO20241120005", "退货", "normal",
                             "return", idempotency_key="k-abc")


# ===================== 4) 进度查询按业务收窄 =====================
class TestProgressQueryScope:
    def test_category_narrows_sql(self, monkeypatch):
        """传了 category，SQL 必须带 category 条件 —— 「不串单」落到查询条件上。"""
        cur = _RecordingCursor(row=None)
        _patch_tx(monkeypatch, cur)

        ms.query_ticket_progress("SO20241120005", owner="u1", category="return")

        sql, params = cur.executed[-1]
        assert "category=%s" in sql
        assert "return" in params

    def test_owner_scopes_query(self, monkeypatch):
        """归属账号始终参与收口，避免拿别人订单号查出别人工单。"""
        cur = _RecordingCursor(row=None)
        _patch_tx(monkeypatch, cur)

        ms.query_ticket_progress("SO20241120005", owner="u1")

        sql, params = cur.executed[-1]
        assert "user_id=%s" in sql and "u1" in params

    def test_ticket_id_lookup_checks_ownership(self, monkeypatch):
        """按 ticket_id 精确查也要带归属校验，不能靠单号就能看别人的单。"""
        cur = _RecordingCursor(row=None)
        _patch_tx(monkeypatch, cur)

        ms.query_ticket_progress("", owner="u1", ticket_id="TK-1")

        sql, params = cur.executed[-1]
        assert "ticket_id=%s" in sql and "user_id=%s" in sql
        assert ("TK-1", "u1") == tuple(params)


class TestInferBusinessCategory:
    @pytest.mark.parametrize("question,expected", [
        ("我的退货办得怎么样了", "return"),
        ("退款到账了吗", "refund"),          # 退款优先于退货，不被「退」字抢答
        ("投诉有结果了吗", "complaint"),
        ("帮我转人工吧", "consult"),
        ("今天天气怎么样", ""),               # 与工单无关 → 不限定分类
    ])
    def test_question_keywords(self, question, expected):
        assert infer_business_category(question, None) == expected

    def test_falls_back_to_dst_intent(self):
        """问法里没有业务词时，沿用会话主线意图（退货进行中的追问）。"""

        class _DST:
            intent = "return"

        assert infer_business_category("办到哪一步了", _DST()) == "return"

    def test_question_beats_dst(self):
        """用户中途换业务：问法优先于 DST，否则会拿退货单回答投诉进度。"""

        class _DST:
            intent = "return"

        assert infer_business_category("我的投诉处理得怎么样", _DST()) == "complaint"


class TestProgressHandlerPassesCategory:
    def test_handler_forwards_business_category(self, monkeypatch):
        """handler 把问法推出的分类传给 query_ticket（模型漏传也不会退回旧行为）。"""
        captured = {}

        class _FakeTool:
            def invoke(self, payload):
                captured.update(payload)
                return "工单进度"

        import agent.tools.ticket_query_tool as tq
        monkeypatch.setattr(tq, "query_ticket", _FakeTool())

        ctx = TurnContext(
            retrieval_service=None, answer_generator=None, dst_manager=None,
            slot_extractor=None, question="我的退货办到哪一步了",
            session_id="u1_s1", user={"user_id": "u1"}, dst=None,
        )

        async def drive():
            return [p async for p in handle_ticket_progress(ctx)]

        assert asyncio.run(drive()) == ["工单进度"]
        assert captured.get("category") == "return"
        assert captured.get("order_id") == ""      # 无订单号时留给工具按账号兜底
