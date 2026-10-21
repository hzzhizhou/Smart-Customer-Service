"""客服业务数据持久化存储（兼容 facade）。

原 god module 已按领域拆分到 infrastructure/db/：
  - connection    : _tx / _row / _rows（连接 + 事务 + 行类型归一）
  - schema        : _DDL / _ensure_column / _ensure_index / init_db（建表 + 增量迁移）
  - orders        : 订单
  - tickets       : 工单、工单消息、人工会话
  - users         : 用户账号与登录令牌
  - conversations : 会话与消息（对话历史）
  - dst           : DST 会话状态
  - tool_audit    : 工具调用审计

存储：MySQL（pymysql 同步驱动，短连接 + 显式事务），数据库 rag_db。
表结构：
  - orders             : 订单主表（首次启动导入种子数据）
  - tickets            : 工单表（Agent 转人工时写入）
  - ticket_messages    : 工单消息（某张工单一旦有消息，它就代表一场人工会话）
  - conversation_state : DST 会话状态（意图/槽位/阶段/护栏计数，多轮跨请求持久化）
  - conversations      : 会话表（对话历史的归属/标题/状态/时间，历史列表的数据源）
  - messages           : 会话消息表（逐条追加，Redis 只是它的热缓存）
  - users              : 用户账号（登录令牌只存 sha256 哈希 + 有效期，库中不留明文）

本模块只做 re-export 与初始化调用，不包含任何函数实现；对外 API 与拆分前完全一致。
"""
from logs.log_config import log

from .db.connection import _tx, _row, _rows
from .db.schema import _DDL, _ensure_column, _ensure_index, init_db
from .db.orders import (
    get_order,
    can_access_order,
    bind_ownerless_orders,
    update_order_status,
    list_all_orders,
    list_orders_by_user,
)
from .db.tickets import (
    create_ticket,
    list_tickets,
    list_all_tickets,
    list_tickets_paged,
    count_tickets,
    _parse_stage_history,
    _STAGE_TO_STATUS,
    _STAGE_TO_ORDER_STATUS,
    update_ticket_status,
    get_ticket,
    get_ticket_by_idempotency_key,
    query_ticket_progress,
    add_ticket_message,
    list_ticket_messages,
    get_handoff_ticket,
    list_handoff_sessions,
)
from .db.users import (
    _USER_PUBLIC,
    _public_user,
    create_user,
    get_user_by_id,
    get_user_by_username,
    _hash_token,
    get_user_by_token,
    update_user_login,
    update_user_password,
    clear_user_token,
    list_users,
    update_user_status,
    delete_user,
    count_users,
)
from .db.conversations import (
    _TITLE_MAX,
    _owner_from_conversation_id,
    ensure_conversation,
    append_message,
    list_messages,
    list_conversations_by_user,
    get_conversation,
    get_latest_conversation_before,
    close_conversation,
    delete_conversation,
    clear_messages,
)
from .db.dst import get_conversation_state, upsert_conversation_state
from .db.tool_audit import record_tool_call, list_tool_calls


# 模块加载时自动初始化（幂等）
try:
    init_db()
except Exception as e:
    log.error(f"业务库(MySQL)初始化失败: {e}", exc_info=True)
