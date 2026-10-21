"""用户账号存储（登录令牌只存 sha256 哈希 + 有效期，库中不留明文）。"""
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import TOKEN_TTL_HOURS

from . import connection

# 对外返回的用户字段（不包含 password_hash / token）
_USER_PUBLIC = ("user_id", "username", "nickname", "role", "status",
                "created_at", "last_login_at")


def _public_user(row: Any) -> Dict[str, Any]:
    d = connection._row(dict(row)) or {}
    return {k: d.get(k) for k in _USER_PUBLIC}


def create_user(user_id: str, username: str, password_hash: str,
                nickname: str = "", role: str = "user") -> Optional[Dict[str, Any]]:
    """新建用户；用户名已存在返回 None（由调用方转 409）。"""
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute("SELECT 1 AS ok FROM users WHERE username=%s", (username,))
        if cur.fetchone():
            return None
        cur.execute(
            "INSERT INTO users (user_id, username, password_hash, nickname, role, "
            "status, token, created_at, last_login_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (user_id, username, password_hash, nickname or username, role,
             "active", "", created_at, ""),
        )
    return {"user_id": user_id, "username": username,
            "nickname": nickname or username, "role": role,
            "status": "active", "created_at": created_at, "last_login_at": ""}


def get_user_by_id(user_id: str, with_secret: bool = False) -> Optional[Dict[str, Any]]:
    with connection._tx() as cur:
        cur.execute("SELECT * FROM users WHERE user_id=%s", (user_id,))
        row = connection._row(cur.fetchone())
    if row is None:
        return None
    return row if with_secret else _public_user(row)


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """含 password_hash，仅供登录校验使用。"""
    with connection._tx() as cur:
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        return connection._row(cur.fetchone())


def _hash_token(token: str) -> str:
    """令牌入库存哈希（sha256 十六进制），库中不留明文。

    数据库一旦被拖走，明文令牌等于可直接冒充任意在线用户；存哈希后攻击者
    拿到的只是摘要，无法反推出可用的 Bearer 令牌。verify 时对来值做同样哈希再比对，
    无需加盐（令牌本身是 256 位随机串，不存在字典爆破空间）。
    """
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def get_user_by_token(token: str) -> Optional[Dict[str, Any]]:
    """按登录令牌取用户；令牌无效或已过期返回 None。

    令牌以哈希形式存库，故先对来值取哈希再查；过期判断放在应用侧用同一个时钟做，
    避免应用与 MySQL 服务器时区不一致时出现偏差（token_expires_at 由本进程写入）。
    """
    if not token:
        return None
    with connection._tx() as cur:
        cur.execute("SELECT * FROM users WHERE token=%s", (_hash_token(token),))
        row = connection._row(cur.fetchone())
    if not row:
        return None
    expires = row.get("token_expires_at") or ""
    if not expires or expires <= datetime.now().strftime("%Y-%m-%d %H:%M:%S"):
        return None
    return _public_user(row)


def update_user_login(user_id: str, token: str) -> None:
    """登录成功：写入令牌哈希、有效期与最近登录时间（明文令牌只回给客户端）。"""
    now = datetime.now()
    expires = (now + timedelta(hours=TOKEN_TTL_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute(
            "UPDATE users SET token=%s, token_expires_at=%s, last_login_at=%s WHERE user_id=%s",
            (_hash_token(token), expires, now.strftime("%Y-%m-%d %H:%M:%S"), user_id),
        )


def update_user_password(user_id: str, password_hash: str) -> None:
    """重置口令哈希（默认账号按环境变量对账时使用）。"""
    with connection._tx() as cur:
        cur.execute(
            "UPDATE users SET password_hash=%s WHERE user_id=%s", (password_hash, user_id)
        )


def clear_user_token(user_id: str) -> None:
    """登出：作废令牌（清空哈希与有效期）。"""
    with connection._tx() as cur:
        cur.execute(
            "UPDATE users SET token='', token_expires_at=NULL WHERE user_id=%s", (user_id,)
        )


def list_users(page: int = 1, page_size: int = 10,
               keyword: str = "") -> tuple[List[Dict[str, Any]], int]:
    """后台用户列表（分页，可按用户名/昵称模糊搜索）。返回 (items, total)。"""
    where, params = "", []
    if keyword:
        where = " WHERE username LIKE %s OR nickname LIKE %s"
        params = [f"%{keyword}%", f"%{keyword}%"]
    with connection._tx() as cur:
        cur.execute(f"SELECT COUNT(*) AS n FROM users{where}", params)
        total = cur.fetchone()["n"]
        cur.execute(
            f"SELECT * FROM users{where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
            params + [page_size, max(0, (page - 1) * page_size)],
        )
        return [_public_user(r) for r in cur.fetchall()], total


def update_user_status(user_id: str, status: str) -> Optional[Dict[str, Any]]:
    """启用/禁用用户；被禁用时同时作废令牌（立即踢下线）。"""
    with connection._tx() as cur:
        cur.execute("SELECT 1 AS ok FROM users WHERE user_id=%s", (user_id,))
        if not cur.fetchone():
            return None
        if status == "disabled":
            cur.execute(
                "UPDATE users SET status=%s, token='', token_expires_at=NULL WHERE user_id=%s",
                (status, user_id),
            )
        else:
            cur.execute("UPDATE users SET status=%s WHERE user_id=%s", (status, user_id))
    return get_user_by_id(user_id)


def delete_user(user_id: str) -> bool:
    """删除用户；不存在返回 False。"""
    with connection._tx() as cur:
        cur.execute("DELETE FROM users WHERE user_id=%s", (user_id,))
        return cur.rowcount > 0


def count_users() -> int:
    with connection._tx() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM users")
        return cur.fetchone()["n"]
