"""建表语句与增量迁移。"""
from config.settings import MYSQL_DATABASE
from logs.log_config import log

from . import connection

# 建表语句（MySQL 8 方言）。索引写在建表里：MySQL 不支持 CREATE INDEX IF NOT EXISTS。
_DDL = [
    """CREATE TABLE IF NOT EXISTS orders (
        id         BIGINT AUTO_INCREMENT PRIMARY KEY,
        order_id   VARCHAR(64)  NOT NULL UNIQUE,
        customer   VARCHAR(64)  NOT NULL DEFAULT '',
        product    VARCHAR(255) NOT NULL DEFAULT '',
        amount     DECIMAL(10,2) NOT NULL DEFAULT 0,
        status     VARCHAR(32)  NOT NULL DEFAULT '',
        logistics  VARCHAR(255) NULL,
        tracking   TEXT         NULL,
        created_at DATETIME     NULL,
        user_id    VARCHAR(64)  NOT NULL DEFAULT '',
        KEY idx_orders_user (user_id, created_at),
        KEY idx_orders_status (status, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS tickets (
        id               BIGINT AUTO_INCREMENT PRIMARY KEY,
        ticket_id        VARCHAR(64)  NOT NULL UNIQUE,
        user_id          VARCHAR(64)  NOT NULL DEFAULT '',
        order_id         VARCHAR(64)  NOT NULL DEFAULT '',
        description      TEXT         NOT NULL,
        priority         VARCHAR(16)  NOT NULL DEFAULT '',
        category         VARCHAR(32)  NOT NULL DEFAULT '',
        status           VARCHAR(16)  NOT NULL DEFAULT 'open',
        created_at       DATETIME     NULL,
        note             TEXT         NOT NULL,
        stage            VARCHAR(32)  NOT NULL DEFAULT '',
        stage_history    TEXT         NOT NULL,
        stage_updated_at VARCHAR(32)  NOT NULL DEFAULT '',
        idempotency_key  VARCHAR(128) NULL,
        KEY idx_tickets_user (user_id, created_at),
        KEY idx_tickets_status (status, created_at),
        UNIQUE KEY uk_tickets_idem (idempotency_key)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS ticket_messages (
        msg_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
        ticket_id   VARCHAR(64) NOT NULL,
        sender      VARCHAR(16) NOT NULL DEFAULT '',
        sender_id   VARCHAR(64) NOT NULL DEFAULT '',
        sender_name VARCHAR(64) NOT NULL DEFAULT '',
        content     TEXT        NOT NULL,
        created_at  DATETIME    NULL,
        KEY idx_ticket_messages_ticket (ticket_id, msg_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS conversation_state (
        session_id        VARCHAR(255) PRIMARY KEY,
        intent            VARCHAR(32)  NOT NULL DEFAULT '',
        slots             TEXT         NOT NULL,
        stage             VARCHAR(32)  NOT NULL DEFAULT 'COLLECTING',
        tool_call_count   INT          NOT NULL DEFAULT 0,
        escalation_reason TEXT         NOT NULL,
        answer            TEXT         NOT NULL,
        updated_at        DATETIME     NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS conversations (
        conversation_id VARCHAR(255) PRIMARY KEY,   -- 作用域会话 ID：u{user_id}_{session_id}
        user_id         VARCHAR(64)  NOT NULL DEFAULT '',  -- 归属账号（隔离唯一依据）
        title           VARCHAR(255) NOT NULL DEFAULT '',  -- 取首条用户消息
        status          VARCHAR(16)  NOT NULL DEFAULT 'open',  -- open=进行中 / closed=已结束
        created_at      DATETIME     NULL,
        updated_at      DATETIME     NULL,
        KEY idx_conversations_user (user_id, updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS messages (
        msg_id          BIGINT AUTO_INCREMENT PRIMARY KEY,
        conversation_id VARCHAR(255) NOT NULL,
        seq             INT          NOT NULL,   -- 会话内序号（从 1 开始，稳定排序与分页）
        role            VARCHAR(16)  NOT NULL,   -- user / assistant
        content         TEXT         NOT NULL,
        created_at      DATETIME     NULL,
        UNIQUE KEY uk_messages_conv_seq (conversation_id, seq)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS users (
        user_id       VARCHAR(64)  PRIMARY KEY,
        username      VARCHAR(64)  NOT NULL UNIQUE,
        password_hash VARCHAR(255) NOT NULL DEFAULT '',
        nickname      VARCHAR(64)  NOT NULL DEFAULT '',
        role          VARCHAR(16)  NOT NULL DEFAULT 'user',
        status        VARCHAR(16)  NOT NULL DEFAULT 'active',
        token         VARCHAR(128) NOT NULL DEFAULT '',
        token_expires_at DATETIME  NULL,
        created_at    DATETIME     NULL,
        last_login_at VARCHAR(32)  NOT NULL DEFAULT '',
        KEY idx_users_token (token)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",

    """CREATE TABLE IF NOT EXISTS agent_tool_calls (
        id            BIGINT AUTO_INCREMENT PRIMARY KEY,
        trace_id      VARCHAR(64)  NOT NULL DEFAULT '',
        request_id    VARCHAR(64)  NOT NULL DEFAULT '',
        session_id    VARCHAR(255) NOT NULL DEFAULT '',
        user_id       VARCHAR(64)  NOT NULL DEFAULT '',
        tool_name     VARCHAR(64)  NOT NULL DEFAULT '',
        args_json     TEXT         NOT NULL,          -- 工具入参（脱敏后的 JSON 串）
        result_digest VARCHAR(64)  NOT NULL DEFAULT '', -- 结果摘要的 sha256（结果原文不入库，防泄露/膨胀）
        status        VARCHAR(16)  NOT NULL DEFAULT 'success',  -- success / error / timeout / denied
        error_code    VARCHAR(64)  NOT NULL DEFAULT '',
        duration_ms   INT          NOT NULL DEFAULT 0,
        created_at    DATETIME     NULL,
        KEY idx_tool_calls_trace (trace_id),
        KEY idx_tool_calls_user_time (user_id, created_at),
        KEY idx_tool_calls_tool_time (tool_name, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def _ensure_column(cur, table: str, column: str, column_ddl: str) -> None:
    """给已存在的老表补列（MySQL 不支持 ADD COLUMN IF NOT EXISTS）。

    CREATE TABLE IF NOT EXISTS 对已建好的表是空操作，老库不会自动获得新列，
    因此迁移必须显式检查 information_schema 后再 ALTER。
    """
    cur.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s AND column_name=%s",
        (MYSQL_DATABASE, table, column),
    )
    if not cur.fetchone():
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column_ddl}")
        log.info(f"迁移：{table} 新增列 {column}")


def _ensure_index(cur, table: str, index: str, ddl: str) -> None:
    """给已存在的老表补索引（CREATE TABLE IF NOT EXISTS 不会动老表的结构）。"""
    cur.execute(
        "SELECT 1 FROM information_schema.statistics "
        "WHERE table_schema=%s AND table_name=%s AND index_name=%s",
        (MYSQL_DATABASE, table, index),
    )
    if not cur.fetchone():
        cur.execute(ddl)
        log.info(f"迁移：{table} 新增索引 {index}")


def init_db():
    """建表 + 增量迁移 + 导入种子订单（幂等）。模块加载时执行一次。"""
    with connection._tx() as cur:
        for ddl in _DDL:
            cur.execute(ddl)

        # 增量迁移：老库 users 表补 token_expires_at（新库建表语句已带该列）。
        # 补列后历史上已存在的令牌因 token_expires_at 为 NULL 而全部失效，
        # 需重新登录换发带有效期的新令牌 —— 这正是「旧 token 过期后 401」的落地方式。
        _ensure_column(cur, "users", "token_expires_at", "token_expires_at DATETIME NULL")
        # 旧版本 token 是明文；补列后虽然已失效，但明文仍留在库里（拖库即可见）。
        # 只清理“没有有效期”的旧行，不碰新登录写入的哈希+有效期令牌。
        cur.execute(
            "UPDATE users SET token='', token_expires_at=NULL "
            "WHERE token<>'' AND token_expires_at IS NULL"
        )

        # 增量迁移：老库 tickets 表补建单幂等键（新库建表语句已带该列与唯一索引）。
        # 唯一索引落在 idempotency_key 上：MySQL 允许多行 NULL，因此历史工单（键为空）
        # 不会互相冲突；只有新写的幂等键才参与去重 —— 并发建单的硬保证就来自这条索引。
        _ensure_column(cur, "tickets", "idempotency_key", "idempotency_key VARCHAR(128) NULL")
        _ensure_index(cur, "tickets", "uk_tickets_idem",
                      "ALTER TABLE tickets ADD UNIQUE KEY uk_tickets_idem (idempotency_key)")

        cur.execute("SELECT COUNT(*) AS n FROM orders")
        if cur.fetchone()["n"] == 0:
            # 平台只经营智能手机一个品类（iPhone 15 / 小米 15 / 华为 Mate 60 Pro），
            # 种子订单与知识库商品清单保持一致，便于演示时按订单号直接核对机型。
            seeds = [
                ("SO20241120005", "陈*杰", "iPhone 15", 4999.00, "delivered",
                 "顺丰速运 SF2345678901",
                 "2024-11-21 已揽收 → 2024-11-22 北京转运中心 → 2024-11-23 已签收，签收人：本人",
                 "2024-11-20 10:15:00"),
                ("SO20241121006", "林*婷", "小米 15", 4499.00, "shipped",
                 "京东物流 JD8765432109",
                 "2024-11-22 已出库 → 2024-11-23 武汉转运中心 → 预计 2024-11-24 送达",
                 "2024-11-21 15:30:00"),
                ("SO20241122007", "周*豪", "华为 Mate 60 Pro", 6499.00, "paid",
                 None, "预计 24 小时内发货", "2024-11-22 09:05:00"),
            ]
            cur.executemany(
                "INSERT INTO orders (order_id, customer, product, amount, status, "
                "logistics, tracking, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                seeds,
            )
            log.info(f"种子订单导入完成：{len(seeds)} 条")
