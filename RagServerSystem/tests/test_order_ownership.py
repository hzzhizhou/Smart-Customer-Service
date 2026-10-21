"""订单归属校验单元测试：本人 / 管理员 / 他人 / 未绑定 / 无账号上下文五种情形。

背景：订单此前没有任何权限边界 —— 拿到订单号就能查到客户、金额、物流轨迹。
本组用例锁住修复后的行为，防止校验被改回「无条件放行」。

注：工具层身份已改为服务端上下文注入（agent.tools.tool_context），
用例用 tool_identity 夹具绑账号，工具参数里不再有 owner_id。
"""
import pytest  # noqa: F401  （pytest 注入 monkeypatch，导入以声明测试依赖）

from infrastructure import mysql_store
import infrastructure.db.users as db_users
from agent.dst.order_validator import validate_return_order
import agent.tools.order_query as oq


def _order(owner: str) -> dict:
    return {
        "order_id": "SO20241120005", "user_id": owner, "customer": "陈*杰",
        "product": "iPhone 15", "amount": 4999.0, "status": "shipped",
        "logistics": "顺丰速运 SF2345678901", "tracking": "已揽收",
        "created_at": "2024-11-20 10:15:00",
    }


# ---------- can_access_order ----------
def test_owner_can_access_own_order():
    assert mysql_store.can_access_order(_order("owner1"), "owner1") is True


def test_other_account_is_denied(monkeypatch):
    monkeypatch.setattr(db_users, "get_user_by_id",
                        lambda uid: {"user_id": uid, "role": "user"})
    assert mysql_store.can_access_order(_order("owner1"), "owner2") is False


def test_admin_can_access_any_order(monkeypatch):
    monkeypatch.setattr(db_users, "get_user_by_id",
                        lambda uid: {"user_id": uid, "role": "admin"})
    assert mysql_store.can_access_order(_order("owner1"), "admin1") is True


def test_ownerless_order_is_open_to_all():
    # 历史数据没有归属账号：不能因为缺字段就把所有人都挡在门外
    assert mysql_store.can_access_order(_order(""), "anyone") is True


def test_missing_account_context_is_denied():
    # 无登录态调用（owner_id 缺失）时宁可查不到，也不越权
    assert mysql_store.can_access_order(_order("owner1"), "") is False


def test_unknown_account_id_is_denied(monkeypatch):
    monkeypatch.setattr(db_users, "get_user_by_id", lambda uid: None)
    assert mysql_store.can_access_order(_order("owner1"), "ghost") is False


# ---------- query_order 工具 ----------
def test_query_order_returns_detail_for_owner(monkeypatch, tool_identity):
    monkeypatch.setattr(oq, "get_order", lambda oid: _order("owner1"))
    tool_identity("owner1")
    out = oq.query_order.invoke({"order_id": "SO20241120005"})
    assert "订单号：SO20241120005" in out
    assert "商品：iPhone 15" in out


def test_query_order_hides_detail_from_other_account(monkeypatch, tool_identity):
    monkeypatch.setattr(oq, "get_order", lambda oid: _order("owner1"))
    monkeypatch.setattr(db_users, "get_user_by_id",
                        lambda uid: {"user_id": uid, "role": "user"})
    tool_identity("owner2")
    out = oq.query_order.invoke({"order_id": "SO20241120005"})
    assert "不在当前登录账号名下" in out
    # 拒绝时必须一个订单字段都不给（客户/商品/金额/物流都属隐私）
    for leaked in ("客户", "商品", "金额", "物流轨迹", "陈*杰", "4999"):
        assert leaked not in out


def test_query_order_not_found_still_reported(monkeypatch, tool_identity):
    monkeypatch.setattr(oq, "get_order", lambda oid: None)
    tool_identity("owner1")
    out = oq.query_order.invoke({"order_id": "SO00000000"})
    assert "未找到订单号" in out


# ---------- 退货槽位校验 ----------
def test_return_validator_forbids_other_account(monkeypatch):
    monkeypatch.setattr(mysql_store, "get_order", lambda oid: _order("owner1"))
    monkeypatch.setattr(db_users, "get_user_by_id",
                        lambda uid: {"user_id": uid, "role": "user"})
    res = validate_return_order("SO20241120005", owner_id="owner2")
    assert res["valid"] is False
    assert res["status"] == "forbidden"


def test_return_validator_allows_owner(monkeypatch):
    monkeypatch.setattr(mysql_store, "get_order", lambda oid: _order("owner1"))
    res = validate_return_order("SO20241120005", owner_id="owner1")
    assert res["valid"] is True
    assert res["status"] == "ok"


def test_return_validator_keeps_status_check(monkeypatch):
    # 归属通过后仍要走「可退状态」校验，别把原有分支覆盖掉
    monkeypatch.setattr(mysql_store, "get_order",
                        lambda oid: {**_order("owner1"), "status": "cancelled"})
    res = validate_return_order("SO20241120005", owner_id="owner1")
    assert res["valid"] is False
    assert res["status"] == "not_returnable"