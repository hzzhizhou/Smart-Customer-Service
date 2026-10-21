"""重构守护：mysql_store 是 re-export facade，不能是复制粘贴的实现。

断言两件事：
  1) facade 暴露的对象与各领域模块里的对象是同一个（is），不是副本；
  2) facade 覆盖拆分前全部公共函数名 —— 防止以后新增函数只加到领域模块、漏了 re-export。
"""
import infrastructure.mysql_store as mysql_store
import infrastructure.db.connection as connection
import infrastructure.db.orders as orders
import infrastructure.db.tickets as tickets
import infrastructure.db.users as users
import infrastructure.db.conversations as conversations
import infrastructure.db.dst as dst
import infrastructure.db.tool_audit as tool_audit

# 拆分前 mysql_store.py 的全部公共函数（显式列出，新增/删除时同步维护）
_PUBLIC_FUNCTIONS = [
    "init_db",
    # 订单
    "get_order", "can_access_order", "bind_ownerless_orders",
    "update_order_status", "list_all_orders", "list_orders_by_user",
    # 工单
    "create_ticket", "list_tickets", "list_all_tickets", "list_tickets_paged",
    "count_tickets", "update_ticket_status", "get_ticket",
    "get_ticket_by_idempotency_key", "query_ticket_progress",
    "add_ticket_message", "list_ticket_messages", "get_handoff_ticket",
    "list_handoff_sessions",
    # 用户
    "create_user", "get_user_by_id", "get_user_by_username", "get_user_by_token",
    "update_user_login", "update_user_password", "clear_user_token",
    "list_users", "update_user_status", "delete_user", "count_users",
    # 对话历史
    "ensure_conversation", "append_message", "list_messages",
    "list_conversations_by_user", "get_conversation",
    "get_latest_conversation_before", "close_conversation",
    "delete_conversation", "clear_messages",
    # DST
    "get_conversation_state", "upsert_conversation_state",
    # 工具审计
    "record_tool_call", "list_tool_calls",
]


def test_facade_reexports_are_same_objects():
    assert mysql_store.get_order is orders.get_order
    assert mysql_store.get_ticket is tickets.get_ticket
    assert mysql_store.get_user_by_id is users.get_user_by_id
    assert mysql_store.record_tool_call is tool_audit.record_tool_call
    assert mysql_store._tx is connection._tx
    assert mysql_store.append_message is conversations.append_message
    assert mysql_store.get_conversation_state is dst.get_conversation_state


def test_facade_covers_all_public_functions():
    missing = [name for name in _PUBLIC_FUNCTIONS
               if not callable(getattr(mysql_store, name, None))]
    assert missing == []


def test_domain_modules_access_connection_via_module():
    # 领域模块必须用 connection._tx()（运行时查属性），否则 monkeypatch 不生效
    import inspect
    for module in (orders, tickets, users, conversations, dst, tool_audit):
        source = inspect.getsource(module)
        assert "from .connection import" not in source
        assert "connection._tx" in source
