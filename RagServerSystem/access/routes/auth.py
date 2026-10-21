"""
账号体系 API：注册 / 登录 / 登出 / 当前用户 + 后台用户管理。

实现取舍（面试项目宁简勿繁）：
- 存储复用 infrastructure.mysql_store 的 users 表，不引入新数据层。
- 口令用标准库 hashlib.pbkdf2_hmac 加盐存储，不引入 bcrypt 等依赖。
- 会话用随机 token（存库为 sha256 哈希 + 有效期），不引入 JWT 库；登出/禁用即作废。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import uuid

from fastapi import Header, HTTPException
from pydantic import BaseModel

from infrastructure import mysql_store

logger = logging.getLogger("auth")

_PBKDF2_ROUNDS = 100_000
_USERNAME_MIN, _USERNAME_MAX = 3, 20
_PASSWORD_MIN = 6

# 默认账号（空库时自动创建，保证首次能进后台 / 跑演示）：
# 口令不再写死进代码 —— 此前默认管理员/买家用的是公开弱口令，随仓库分发等于把后台入口公开。
# 现在优先取环境变量，未配置则生成一次性强随机口令并只在创建时打印一次到日志；
# 口令随机不可预测，故默认账号在未显式配置口令时无法被猜解登录。
DEFAULT_ADMIN_USERNAME = (os.getenv("DEFAULT_ADMIN_USERNAME") or "admin").strip()
DEFAULT_BUYER_USERNAME = (os.getenv("DEFAULT_BUYER_USERNAME") or "customer").strip()


def _bootstrap_password(env_key: str) -> str:
    """默认账号口令：优先读环境变量；未配置则生成一次性强随机口令。"""
    return (os.getenv(env_key) or "").strip() or secrets.token_urlsafe(16)


# ---------- 口令 ----------

def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ROUNDS
    )
    return f"pbkdf2_sha256${_PBKDF2_ROUNDS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(rounds)
        )
        return secrets.compare_digest(got.hex(), digest)
    except (ValueError, AttributeError):
        return False


# ---------- 请求体 ----------

class RegisterReq(BaseModel):
    username: str
    password: str
    nickname: str | None = None


class LoginReq(BaseModel):
    username: str
    password: str


class UserStatusReq(BaseModel):
    status: str   # active | disabled


# ---------- 令牌 ----------

def _extract_token(authorization: str) -> str:
    """从 Authorization: Bearer <token> 中取令牌；缺失返回空串。"""
    if not authorization:
        return ""
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization.strip()


def require_user(authorization: str = Header(default="")) -> dict:
    """取当前登录用户；无效/禁用则 401。"""
    user = mysql_store.get_user_by_token(_extract_token(authorization))
    if not user:
        raise HTTPException(status_code=401, detail="未登录或登录已失效，请重新登录")
    if user.get("status") != "active":
        raise HTTPException(status_code=403, detail="账号已被禁用")
    return user


def require_admin(authorization: str = Header(default="")) -> dict:
    """取当前登录用户并校验管理员角色；非管理员 403。

    后台管理接口（工单/订单/知识库/用户）统一挂这个依赖，
    普通用户即使已登录也无法读写后台数据。
    """
    user = require_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="仅管理员可访问后台管理接口")
    return user


def _validate(username: str, password: str) -> None:
    username = (username or "").strip()
    if not (_USERNAME_MIN <= len(username) <= _USERNAME_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"用户名长度需在 {_USERNAME_MIN}-{_USERNAME_MAX} 个字符之间",
        )
    if len(password or "") < _PASSWORD_MIN:
        raise HTTPException(status_code=400, detail=f"密码至少 {_PASSWORD_MIN} 位")


def _sync_default_password(existing: dict, env_key: str, username: str,
                           legacy_password: str = "") -> None:
    """对账已存在的默认账号：配置了环境变量口令就重置库中口令，否则提示轮换。

    代码只在空库时新建账号，因此老库里的历史默认账号（口令可能是旧弱口令）不会自动变强。
    这里给两条显式路径：
      1) 在 .env 设 <ENV_KEY>：重启即把口令重置为该值；
      2) 未配置但检测到历史弱口令（admin123/customer123）：直接换成随机口令并打印一次，
         避免“代码已改、老库仍是弱口令”的升级盲区。
    """
    configured = (os.getenv(env_key) or "").strip()
    if configured:
        if verify_password(configured, existing.get("password_hash", "")):
            return
        mysql_store.update_user_password(existing["user_id"], hash_password(configured))
        logger.info(f"已按 {env_key} 重置默认账号口令：{username}")
        return

    if legacy_password and verify_password(legacy_password, existing.get("password_hash", "")):
        new_password = _bootstrap_password(env_key)
        mysql_store.update_user_password(existing["user_id"], hash_password(new_password))
        logger.warning(
            f"检测到默认账号 {username} 仍在使用历史弱口令，已重置为随机口令：{new_password} "
            f"—— 请立即登录修改，或在 .env 设置 {env_key} 固定口令"
        )
        return

    logger.warning(
        f"默认账号 {username} 已存在且未配置 {env_key}：当前口令保持不变，"
        f"若仍是历史弱口令请尽快轮换（在 .env 设置 {env_key} 后重启即自动重置）"
    )


def ensure_default_admin() -> None:
    """空库时创建默认管理员（幂等），便于首次演示直接登录后台。

    口令取 DEFAULT_ADMIN_PASSWORD，未配置则随机生成（只在此处打印一次）。
    """
    existing = mysql_store.get_user_by_username(DEFAULT_ADMIN_USERNAME)
    if existing:
        _sync_default_password(existing, "DEFAULT_ADMIN_PASSWORD",
                               DEFAULT_ADMIN_USERNAME, legacy_password="admin123")
        return
    password = _bootstrap_password("DEFAULT_ADMIN_PASSWORD")
    mysql_store.create_user(
        uuid.uuid4().hex, DEFAULT_ADMIN_USERNAME, hash_password(password), "管理员", role="admin"
    )
    logger.info(
        f"已创建默认管理员：{DEFAULT_ADMIN_USERNAME}（口令：{password}）"
        f" —— 请登录后立即修改，或用 DEFAULT_ADMIN_PASSWORD 固定口令"
    )


def ensure_default_buyer() -> None:
    """演示买家账号（幂等）：种子订单挂它名下，归属校验才有真实数据可校验。

    不建这个账号的话，种子订单没有归属账号，任何登录账号都能查到别人的订单；
    建了之后「用别的账号查同一张订单」会被拒，正好演示归属校验生效。
    """
    user = mysql_store.get_user_by_username(DEFAULT_BUYER_USERNAME)
    if user:
        _sync_default_password(user, "DEFAULT_BUYER_PASSWORD",
                               DEFAULT_BUYER_USERNAME, legacy_password="customer123")
    else:
        password = _bootstrap_password("DEFAULT_BUYER_PASSWORD")
        user = mysql_store.create_user(
            uuid.uuid4().hex, DEFAULT_BUYER_USERNAME, hash_password(password), "用户", role="user"
        )
        if user:
            logger.info(
                f"已创建演示买家：{DEFAULT_BUYER_USERNAME}（口令：{password}）"
                f" —— 可用 DEFAULT_BUYER_PASSWORD 固定口令"
            )
    if not user:
        return
    bound = mysql_store.bind_ownerless_orders(user["user_id"])
    if bound:
        logger.info(f"已把 {bound} 张无归属订单绑定到演示买家 {DEFAULT_BUYER_USERNAME}")


def register_auth_routes(app) -> None:
    """把账号相关接口挂到 FastAPI app。"""
    try:
        ensure_default_admin()
        ensure_default_buyer()
    except Exception as e:  # 种子失败不影响服务启动
        logger.error(f"默认账号创建失败: {e}")

    # ---------- 注册 / 登录 ----------
    @app.post("/auth/register")
    async def register(req: RegisterReq):
        _validate(req.username, req.password)
        username = req.username.strip()
        # 口令哈希（PBKDF2 10 万轮，CPU 密集）与写库都是阻塞操作，一律放线程池，
        # 否则注册/登录这类热路径会在事件循环里串行排队
        pwd_hash = await asyncio.to_thread(hash_password, req.password)
        user = await asyncio.to_thread(
            mysql_store.create_user,
            uuid.uuid4().hex, username, pwd_hash,
            (req.nickname or "").strip() or username,
        )
        if user is None:
            raise HTTPException(status_code=409, detail=f"用户名已存在: {username}")
        # 注册即登录，直接下发令牌，前端无需再跳一次登录
        token = secrets.token_urlsafe(32)
        await asyncio.to_thread(mysql_store.update_user_login, user["user_id"], token)
        return {"token": token,
                "user": await asyncio.to_thread(mysql_store.get_user_by_id, user["user_id"])}

    @app.post("/auth/login")
    async def login(req: LoginReq):
        row = await asyncio.to_thread(
            mysql_store.get_user_by_username, (req.username or "").strip())
        if not row or not await asyncio.to_thread(
                verify_password, req.password, row.get("password_hash", "")):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        if row.get("status") != "active":
            raise HTTPException(status_code=403, detail="账号已被禁用，请联系管理员")
        token = secrets.token_urlsafe(32)
        await asyncio.to_thread(mysql_store.update_user_login, row["user_id"], token)
        return {"token": token,
                "user": await asyncio.to_thread(mysql_store.get_user_by_id, row["user_id"])}

    @app.post("/auth/logout")
    async def logout(authorization: str = Header(default="")):
        user = await asyncio.to_thread(mysql_store.get_user_by_token,
                                       _extract_token(authorization))
        if user:
            await asyncio.to_thread(mysql_store.clear_user_token, user["user_id"])
        return {"ok": True}

    @app.get("/auth/me")
    async def me(authorization: str = Header(default="")):
        return {"user": await asyncio.to_thread(require_user, authorization)}

    # ---------- 用户管理（后台，仅管理员） ----------
    @app.get("/users")
    async def list_users(keyword: str = "", page: int = 1, page_size: int = 10,
                         authorization: str = Header(default="")):
        # require_admin 内部要查库校验令牌，必须放线程池（这里是直接调用而非 Depends，
        # 不会自动被 FastAPI 调度到线程池）
        await asyncio.to_thread(require_admin, authorization)
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        items, total = await asyncio.to_thread(
            mysql_store.list_users, page, page_size, keyword.strip())
        return {"items": items, "total": total, "page": page, "page_size": page_size}

    @app.patch("/users/{user_id}")
    async def update_user(user_id: str, req: UserStatusReq,
                          authorization: str = Header(default="")):
        current = await asyncio.to_thread(require_admin, authorization)
        if req.status not in ("active", "disabled"):
            raise HTTPException(status_code=400, detail="非法状态，可选：active, disabled")
        if user_id == current["user_id"] and req.status == "disabled":
            raise HTTPException(status_code=400, detail="不能禁用当前登录账号")
        row = await asyncio.to_thread(mysql_store.update_user_status, user_id, req.status)
        if row is None:
            raise HTTPException(status_code=404, detail=f"用户不存在: {user_id}")
        return row

    @app.delete("/users/{user_id}")
    async def delete_user(user_id: str, authorization: str = Header(default="")):
        current = await asyncio.to_thread(require_admin, authorization)
        if user_id == current["user_id"]:
            raise HTTPException(status_code=400, detail="不能删除当前登录账号")
        if not await asyncio.to_thread(mysql_store.delete_user, user_id):
            raise HTTPException(status_code=404, detail=f"用户不存在: {user_id}")
        return {"deleted": user_id}