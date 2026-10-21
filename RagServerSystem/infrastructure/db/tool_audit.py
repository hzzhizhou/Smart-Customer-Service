"""工具调用审计存储。"""
from datetime import datetime
from typing import Any, Dict, List

from . import connection


def record_tool_call(trace_id: str = "", request_id: str = "", session_id: str = "",
                     user_id: str = "", tool_name: str = "", args_json: str = "",
                     result_digest: str = "", status: str = "success",
                     error_code: str = "", duration_ms: int = 0) -> None:
    """写入一条工具调用审计（一次调用一行）。

    只落「摘要」不落原文：入参经脱敏后再存（见 service/turn/state.py 的 _redact_args），
    结果只存摘要的 sha256，避免把客户隐私/长文本写进审计库。
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connection._tx() as cur:
        cur.execute(
            "INSERT INTO agent_tool_calls (trace_id, request_id, session_id, user_id, "
            "tool_name, args_json, result_digest, status, error_code, duration_ms, created_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (trace_id, request_id, session_id, user_id, tool_name, args_json or "",
             result_digest or "", status or "success", error_code or "",
             int(duration_ms or 0), created_at),
        )


def list_tool_calls(tool_name: str = "", user_id: str = "", trace_id: str = "",
                    status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
    """按维度查询工具调用审计（排障/审计用），最新在前。

    过滤条件全部为固定子句（用户输入仍走 %s 占位符绑定），WHERE 部分只拼接白名单片段，
    不存在 SQL 注入面；空条件表示不过滤该维度。
    """
    clauses: List[str] = []
    params: List[Any] = []
    if tool_name:
        clauses.append("tool_name=%s")
        params.append(tool_name)
    if user_id:
        clauses.append("user_id=%s")
        params.append(user_id)
    if trace_id:
        clauses.append("trace_id=%s")
        params.append(trace_id)
    if status:
        clauses.append("status=%s")
        params.append(status)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(min(max(int(limit or 100), 1), 500))
    with connection._tx() as cur:
        cur.execute(
            f"SELECT * FROM agent_tool_calls{where} ORDER BY id DESC LIMIT %s",
            tuple(params),
        )
        return connection._rows(cur.fetchall())
