"""人工会话生命周期（服务层）：建单/复用人工会话、写入开场白。
判据：某工单「存在 ticket_messages 记录」即视为一场人工会话（见 access/routes/handoff.py）。
真正写开场白的只有本模块，两处调用方共用：
- 用户主动转人工 → access/routes/handoff.py 的 POST /handoff/request
- AI 判定投诉后确定性建单 → service/handlers/complaint.py
"""
from typing import Any, Dict, Tuple

from logs.log_config import log
from infrastructure import mysql_store
from agent.tools.ticket_tool import create_ticket_record
from shared.reply_templates import HANDOFF_OPENING
from utils.metrics import handoff_total


def _write_opening(ticket_id: str) -> Dict[str, Any]:
    """写入人工会话开场白。这条 system 消息同时是该工单「已进入人工会话」的标记。"""
    return mysql_store.add_ticket_message(
        ticket_id, "system", HANDOFF_OPENING.format(ticket_id=ticket_id),
    )


def ensure_handoff_session(user_id: str, question: str = "") -> Tuple[Dict[str, Any], bool]:
    """开启或复用该用户的人工会话，返回 (工单记录, 是否复用)。

    已有进行中会话则直接复用，双击「转人工」不会重复建单。
    新建时用 consult 分类，不复用退货/退款工单——那类工单有 stage 阶段流转，
    拿来当会话会污染用户查到的退货进度。
    """
    existing = mysql_store.get_handoff_ticket(user_id)
    if existing:
        return existing, True

    ticket = create_ticket_record(
        user_id=user_id,
        description=(question or "").strip() or "用户请求转人工",
        priority="high",
        category="consult",
    )
    _write_opening(ticket["ticket_id"])
    # 只在「新建」时计数：复用既有会话不算一次新的转人工，否则命中率会被重复请求灌水
    handoff_total.labels("user_request").inc()
    log.info(f"人工会话已开启：{ticket['ticket_id']} | 用户：{user_id}")
    return ticket, False


def start_handoff_session(ticket_id: str) -> Dict[str, Any]:
    """对已存在的工单开启人工会话（AI 自动转人工建单后调用）。"""
    msg = _write_opening(ticket_id)
    log.info(f"人工会话已开启（AI 转人工）：{ticket_id}")
    return msg