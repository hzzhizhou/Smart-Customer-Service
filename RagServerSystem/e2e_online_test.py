"""
智能售后客服 RAG 客服 · 在线端到端验证
前置：服务已在 8000 运行、MySQL/Redis 可用、DashScope key 有效；
会话类接口需登录态，脚本用环境变量里的管理员账号登录
（首次启动时后端自动建号；账号名取 DEFAULT_ADMIN_USERNAME，口令取 DEFAULT_ADMIN_PASSWORD）。
覆盖场景：
  S1 健康检查
  S2 RAG 知识库问答
  S3 Agent 订单查询（业务工具链路）
  S4 多轮对话 / DST 状态跟踪
  S5 投诉转人工（create_ticket 工具）
"""
import json
import os
import sys
import time

import urllib.request

BASE = "http://127.0.0.1:8000"

# 会话类接口（/rag/stream、/agent/stream、/history/*）都要登录态，
# 统一用管理员账号登录：管理接口（如 /health 之外的看板接口）也只认管理员。
# 默认账号口令不再写死在代码里（见 access/routes/auth.py），故从环境变量读取：
#   DEFAULT_ADMIN_USERNAME / DEFAULT_ADMIN_PASSWORD（未设账号名则回退 admin）
TOKEN = ""
UID = ""
ADMIN = (os.getenv("DEFAULT_ADMIN_USERNAME", "admin"),
         os.getenv("DEFAULT_ADMIN_PASSWORD", ""))


def _login(username: str, password: str) -> dict:
    req = urllib.request.Request(
        BASE + "/auth/login",
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


def _call(method: str, path: str, payload: dict = None, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=_headers(),
        method=method,
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    data["_latency"] = round(time.time() - t0, 2)
    return data


def _stream(path: str, payload: dict = None, timeout: int = 180) -> tuple:
    """企业统一流式输出：POST 流式端点，返回 (完整文本, 耗时)。"""
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=_headers(),
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        text = resp.read().decode()
    return text.strip(), round(time.time() - t0, 2)


def _read_dst_state(sid: str, timeout: float = 15.0):
    """轮询读取 MySQL 中的 DST 会话状态（后台异步保存，故轮询等待）。"""
    import pymysql
    from config.settings import (
        MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = pymysql.connect(
                host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                password=MYSQL_PASSWORD, database=MYSQL_DATABASE,
                charset="utf8mb4", connect_timeout=5,
            )
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT intent, slots, stage FROM conversation_state "
                    "WHERE session_id=%s", (sid,))
                row = cur.fetchone()
            conn.close()
            if row:
                return {"intent": row[0], "slots": json.loads(row[1]), "stage": row[2]}
        except (pymysql.MySQLError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    return None


def step(no: str, title: str):
    print(f"\n===== S{no} | {title} =====")


def main():
    global TOKEN, UID
    failed = []

    # S0 登录（后续所有会话类/管理类接口都带这个令牌）
    step("0", "登录 admin")
    try:
        login = _login(*ADMIN)
        TOKEN, UID = login["token"], login["user"]["user_id"]
        print(f"  login -> {login['user']['username']} / {login['user']['role']}")
    except Exception as e:
        print(f"  login ERR: {e}")
        sys.exit(1)

    # S1 健康检查
    step("1", "健康检查")
    try:
        h = _call("GET", "/health", timeout=30)
        comps = h.get("components", {})
        detail = ", ".join(f"{k}={v}" for k, v in comps.items())
        print(f"  health -> {h.get('status')} ({detail})")
        # 关键路径（vector/mysql/llm）任一不 ok 才算失败；redis 只是缓存层，单独展示
        if h.get("status") != "healthy" or any(
                not str(v).startswith("ok") for v in comps.values()):
            failed.append("S1")
    except Exception as e:
        print(f"  health ERR: {e}")
        failed.append("S1")

    # S2 RAG 知识库问答（流式）
    step("2", "RAG 知识库问答 (退货流程)")
    try:
        ans, lat = _stream("/rag/stream", {"question": "退货流程是怎么样的？"})
        print(f"  latency={lat}s")
        print(f"  answer: {ans[:120]}")
        # 「无相关信息」与空答复兜底话术都代表"这题没答上来"，S2 必须判失败：
        # 正常链路应给出有据可依的答案，否则说明检索/生成退化（兜底只用于无据可答的场景）。
        if not ans or "无相关信息" in ans or "没能查到可靠的资料" in ans:
            failed.append("S2")
    except Exception as e:
        print(f"  RAG ERR: {e}")
        failed.append("S2")

    # S3 Agent 订单查询（流式）
    step("3", "Agent 订单查询 (SO20241120005)")
    try:
        ans, lat = _stream("/agent/stream", {"question": "帮我查一下订单 SO20241120005 现在到哪了"})
        print(f"  latency={lat}s")
        print(f"  answer: {ans[:160]}")
        if not ans:
            failed.append("S3")
    except Exception as e:
        print(f"  Agent ERR: {e}")
        failed.append("S3")

    # S4 多轮对话 / DST（连续两问，验证状态跟踪 + MySQL 槽位持久化）
    step("4", "多轮对话 + DST 状态跟踪")
    try:
        sid = "e2e_dst_test_001"
        a1, _ = _stream("/agent/stream", {"question": "我要退货", "session_id": sid})
        a2, _ = _stream("/agent/stream", {"question": "订单号 SO20241120005，商品破损", "session_id": sid})
        print(f"  第1轮: {a1[:80]}")
        print(f"  第2轮: {a2[:80]}")
        if not a1 or not a2:
            failed.append("S4")
        # 灰盒断言：读 MySQL 会话状态（DST 按作用域会话 ID 存：u{user_id}_{session_id}）
        dst = _read_dst_state(f"u{UID}_{sid}")
        if dst is None:
            print("  DST 状态未落库")
            failed.append("S4")
        else:
            intent, slots, stage = dst["intent"], dst["slots"], dst["stage"]
            print(f"  DST -> intent={intent} | slots={slots} | stage={stage}")
            # 第2轮仍保持退货主线（融合后不被"订单号"重置为 order）
            if intent != "return":
                print(f"  意图不是 return: {intent}")
                failed.append("S4")
            if slots.get("order_id") != "SO20241120005":
                print(f"  order_id 槽位未正确填充: {slots}")
                failed.append("S4")
            if slots.get("reason") != "商品破损":
                print(f"  reason 槽位未正确填充: {slots}")
                failed.append("S4")
            if stage not in ("CONFIRMING", "EXECUTING", "DONE"):
                print(f"  阶段未推进: {stage}")
                failed.append("S4")
    except Exception as e:
        print(f"  DST ERR: {e}")
        failed.append("S4")

    # S5 投诉转人工（create_ticket）
    step("5", "投诉转人工 (create_ticket)")
    try:
        ans, lat = _stream("/agent/stream",
                           {"question": "我要投诉！商品质量太差了，客服也不接电话，我要投诉"})
        print(f"  latency={lat}s")
        print(f"  answer: {ans[:160]}")
        if not ans:
            failed.append("S5")
    except Exception as e:
        print(f"  投诉 ERR: {e}")
        failed.append("S5")

    print("\n" + "=" * 40)
    if failed:
        print(f"结论：FAILED 场景 {failed}")
        sys.exit(1)
    print("结论：全部场景通过 ✅")


if __name__ == "__main__":
    main()