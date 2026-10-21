"""DST 会话状态存储。"""
import json
from datetime import datetime
from typing import Any, Dict, Optional

from . import connection


def get_conversation_state(session_id: str) -> Optional[Dict[str, Any]]:
    """读取会话状态；slots 字段解析为 dict；不存在返回 None。"""
    with connection._tx() as cur:
        cur.execute(
            "SELECT * FROM conversation_state WHERE session_id=%s", (session_id,)
        )
        d = connection._row(cur.fetchone())
    if not d:
        return None
    try:
        d["slots"] = json.loads(d["slots"]) if d["slots"] else {}
    except ValueError:
        d["slots"] = {}
    return d


def upsert_conversation_state(state: Dict[str, Any]) -> None:
    """不存在则 INSERT，存在则整体覆盖（session 级 upsert），幂等。"""
    with connection._tx() as cur:
        cur.execute(
            """INSERT INTO conversation_state
               (session_id, intent, slots, stage, tool_call_count,
                escalation_reason, answer, updated_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE
                 intent=VALUES(intent), slots=VALUES(slots), stage=VALUES(stage),
                 tool_call_count=VALUES(tool_call_count),
                 escalation_reason=VALUES(escalation_reason),
                 answer=VALUES(answer), updated_at=VALUES(updated_at)""",
            (state["session_id"], state.get("intent", ""),
             json.dumps(state.get("slots", {}), ensure_ascii=False),
             state.get("stage", "COLLECTING"), state.get("tool_call_count", 0),
             state.get("escalation_reason", "") or "", state.get("answer", "") or "",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
