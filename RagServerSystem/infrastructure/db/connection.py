"""MySQL 连接与事务（pymysql 同步驱动，短连接 + 显式事务）。"""
from contextlib import contextmanager
from datetime import date, datetime, time as _time
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor

from config.settings import (
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_USER,
)


@contextmanager
def _tx():
    """一次业务操作 = 一次连接 + 一个事务：正常提交，异常回滚，最后关连接。

    FastAPI 线程池与后台线程都会调进来，pymysql 连接不是线程安全的，
    因此不做跨请求的连接复用（每次短连接），用数据库自身的事务和行锁保证一致性。
    """
    conn = pymysql.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
        password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
        charset="utf8mb4", cursorclass=DictCursor,
        autocommit=False, connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _row(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """把 DB 行的类型转成与迁移前一致的 Python 类型，避免调用方感知存储变更：

    - DATETIME → 'YYYY-MM-DD HH:MM:SS' 字符串（前端展示与日志都按字符串用）
    - DECIMAL  → float（金额按数值用，避免出现 '599.00' 这类字符串差异）
    """
    if row is None:
        return None
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, datetime):
            value = value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(value, (date, _time)):
            value = value.isoformat()
        elif isinstance(value, Decimal):
            value = float(value)
        out[key] = value
    return out


def _rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_row(r) for r in rows]
