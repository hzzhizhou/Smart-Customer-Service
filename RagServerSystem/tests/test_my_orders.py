"""「查我名下所有订单」功能单元测试。

背景：此前客服只支持「按订单号查单」，用户不带订单号问"我有哪些订单"时只会反复
追问订单号。新增 list_my_orders 后，本组用例锁住三件事：
  1) 只列本登录账号名下订单（不串到他人）；
  2) 拿不到账号时拒答，而不是按会话/最近订单去猜；
  3) 「列出订单」与「办理业务 / 问方法」的问法能区分开，避免误拦。

注：账号已改为服务端上下文注入（agent.tools.tool_context），用例用 tool_identity
夹具绑定登录账号，工具参数里不再有 owner_id。
"""
import pytest  # noqa: F401  （pytest 注入 monkeypatch，导入以声明测试依赖）

from infrastructure import mysql_store
from shared.constants import is_my_orders_query
import agent.tools.order_query as oq


def _order(oid: str, owner: str, status: str = "shipped") -> dict:
    return {
        "order_id": oid, "user_id": owner, "customer": "陈*杰",
        "product": "iPhone 15", "amount": 4999.0, "status": status,
        "logistics": None, "tracking": None, "created_at": "2024-11-20 10:15:00",
    }


# ---------- 问法判定：该拦的拦、不该拦的不拦 ----------
def test_listing_questions_are_detected():
    for q in ("我有哪些订单", "查一下我名下所有订单", "我的订单", "我买过什么",
              "我的订单列表", "看看我的历史订单"):
        assert is_my_orders_query(q) is True, q


def test_specific_order_query_is_not_listing():
    # 带订单号属于精确查询，走上面那个分支
    assert is_my_orders_query("我的订单 SO20241120005 到哪了") is False


def test_method_questions_are_not_listing():
    # "怎么查订单"属知识咨询，不能被当成"帮我列出来"
    assert is_my_orders_query("我的订单怎么查") is False


def test_business_requests_are_not_listing():
    # 句中出现"我的订单"只是在说明对象，诉求是退货/查进度，不能只回一个列表
    for q in ("我的订单有质量问题想退货", "我的订单退款进度", "我名下订单要开发票"):
        assert is_my_orders_query(q) is False, q


# ---------- list_orders_by_user ----------
def test_blank_account_returns_nothing():
    # 没有账号上下文时直接返回空，不能把全库订单列出来
    assert mysql_store.list_orders_by_user("") == ([], 0)


# ---------- list_my_orders 工具 ----------
def test_lists_only_own_orders(monkeypatch, tool_identity):
    monkeypatch.setattr(oq, "list_orders_by_user",
                        lambda uid, limit=20: ([_order("SO20241121006", uid),
                                               _order("SO20241120005", uid)], 2))
    tool_identity("owner1")
    out = oq.list_my_orders.invoke({})
    assert "您名下共 2 张订单" in out
    assert "SO20241120005" in out and "SO20241121006" in out
    assert "已发货" in out and "¥4999.00" in out


def test_empty_account_gets_hint_not_blank(monkeypatch, tool_identity):
    monkeypatch.setattr(oq, "list_orders_by_user", lambda uid, limit=20: ([], 0))
    tool_identity("owner1")
    out = oq.list_my_orders.invoke({})
    assert "还没有订单" in out


def test_missing_owner_is_refused(monkeypatch):
    called = []
    monkeypatch.setattr(oq, "list_orders_by_user",
                        lambda uid, limit=20: called.append(uid) or ([_order("SO1", "owner1")], 1))
    # 未绑定身份（上下文里没有账号）：连库都不查，更不猜账号
    out = oq.list_my_orders.invoke({})
    assert "没能确认您的登录账号" in out
    assert called == []      # 缺账号时连库都不查（不猜账号）
    assert "SO1" not in out


def test_limit_is_clamped_to_max(monkeypatch, tool_identity):
    seen = {}

    def fake(uid, limit=20):
        seen["limit"] = limit
        return [], 0

    monkeypatch.setattr(oq, "list_orders_by_user", fake)
    tool_identity("owner1")
    oq.list_my_orders.invoke({"limit": 999})
    assert seen["limit"] == oq.MY_ORDERS_MAX


def test_more_than_limit_hints_single_query(monkeypatch, tool_identity):
    monkeypatch.setattr(oq, "list_orders_by_user",
                        lambda uid, limit=20: ([_order("SO1", uid)], 5))
    tool_identity("owner1")
    out = oq.list_my_orders.invoke({})
    assert "共 5 张订单" in out and "以下是最新的 1 张" in out
    assert "请告诉我订单号" in out