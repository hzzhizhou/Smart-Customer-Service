"""
# 智能售后客服 - 工单创建/转人工工具
当 RAG/订单系统无法解决用户问题时，创建工单转人工客服跟进。
适用场景：用户投诉、需要人工介入的复杂售后问题、RAG 无法解决的咨询。
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import hashlib
import uuid
from datetime import datetime
from langchain_core.tools import tool
from logs.log_config import log
from utils.metrics import handoff_total, tickets_created_total
from infrastructure.mysql_store import create_ticket as _persist_ticket
from agent.tools.tool_context import current_user_id
from config.settings import TICKET_IDEMPOTENCY_WINDOW
from shared.reply_templates import SERVICE_CONTACT, TICKET_CREATED, TICKET_NEED_LOGIN, sla_cn


# 优先级白名单
_VALID_PRIORITY = {"urgent", "high", "normal", "low"}
# 分类白名单
_VALID_CATEGORY = {
    "complaint",   # 投诉
    "return",      # 退货
    "refund",      # 退款
    "consult",     # 咨询
    "technical",   # 技术问题
    "logistics",   # 物流问题
    "other",       # 其他
}
# 带业务环节流转的售后分类（建单即进入第 1 环节：提交申请，等待商家审核）
_STAGED_CATEGORY = {"return", "refund"}
# 建单后同时开启人工会话的分类：客服工作台以「工单有没有 ticket_messages」判定会话，
# 不写开场白的话工单只进工单列表、不进工作台，客服根本看不到（AI 转人工的漏接点）。
# 退货/退款排除在外：它们有 stage 阶段流转，拿来当会话会污染用户的退货进度查询。
_HANDOFF_CATEGORY = {"complaint", "consult", "technical", "logistics", "other"}
# 分类中文别名 → 枚举（模型偶尔直接回中文分类，先归一化再校验，避免被降级成 other）
_CATEGORY_ALIAS = {
    "退货": "return", "退款": "refund", "投诉": "complaint", "咨询": "consult",
    "技术": "technical", "技术支持": "technical", "物流": "logistics", "其他": "other",
}


def _idempotency_key(user_id: str, order_id: str, category: str,
                     now: datetime) -> str:
    """按「账号 + 订单 + 分类 + 时间窗」生成幂等键（PROJECT_AUDIT §6.2(2)）。

    此前防重复是「先查后写」：两个并发请求（用户双击确认、客户端重试）会同时查到
    「还没有在办工单」，然后各建一张 —— 检查形同虚设。现在把去重交给数据库唯一键：
    同键并发插入只有一个能成功，另一个捕获冲突后返回已有工单。

    时间窗（TICKET_IDEMPOTENCY_WINDOW，默认 5 分钟）的取舍：窗口内同账号同订单同分类
    视为同一次申请；超出窗口再提交会新建工单，避免「一次误操作把该业务永久锁死」。
    反过来的已知边界：上一张单已关闭、用户仍在窗口内再次申请时会被判为重复，
    需等窗口过后再提交（或由客服手动建单）。
    """
    bucket = int(now.timestamp()) // max(TICKET_IDEMPOTENCY_WINDOW, 1)
    raw = f"{user_id}|{order_id or ''}|{category}|{bucket}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]


def create_ticket_record(user_id: str, description: str, priority: str = "normal",
                         category: str = "other", order_id: str = "") -> dict:
    """建单并落库，返回工单记录（dict）。

    与 @tool 分离：工具返回给 LLM 的是给用户看的话术，本函数返回结构化记录，
    供确定性主流程（如退货登记成功后拼装完整流程说明）复用同一套校验与落库逻辑。
    返回的可能是**已存在的工单**（幂等键冲突），调用方按 ticket_id 继续即可。
    """
    # 优先级/分类合法性校验（边界输入保护）
    if priority not in _VALID_PRIORITY:
        log.warning(f"非法优先级 {priority}，降级为 normal")
        priority = "normal"
    category = _CATEGORY_ALIAS.get(str(category).strip(), category)
    if category not in _VALID_CATEGORY:
        log.warning(f"非法分类 {category}，降级为 other")
        category = "other"

    now = datetime.now()
    # 生成工单号：TK + 时间戳 + 4位随机
    ticket_id = f"TK{now.strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4].upper()}"
    # 退货/退款走阶段化流转，建单即处于第 1 环节；其他分类无业务环节
    stage = "submitted" if category in _STAGED_CATEGORY else ""

    # 持久化到 MySQL：user_id 存登录账号标识（工单归属账号，换会话也能查到），
    # order_id 单独存关联订单（无则空串）
    ticket = _persist_ticket(ticket_id, user_id, order_id, description, priority,
                             category, stage=stage,
                             idempotency_key=_idempotency_key(user_id, order_id, category, now))
    if ticket["ticket_id"] != ticket_id:
        log.info(f"建单命中幂等键，复用已有工单：{ticket['ticket_id']} | 分类：{category} "
                 f"| 用户：{user_id} | 订单：{order_id}")
        return ticket
    # 只有真正新建才计数：幂等复用不是一次新工单，否则“工单创建数”会被双击/重试灌水
    tickets_created_total.inc()
    log.info(f"工单已创建：{ticket_id} | 优先级：{priority} | 分类：{category} "
             f"| 用户：{user_id} | 订单：{order_id} | 环节：{stage or '-'}")
    return ticket


@tool(description=(
    "创建工单并转交人工客服处理（同时开启人工会话）。适用于：用户投诉、需要人工介入"
    "的复杂问题、知识库或订单系统无法解决的咨询。参数：description 问题描述（尽量完整"
    "记录用户诉求）；priority 优先级取 urgent/high/normal/low（默认 normal）；category "
    "分类取 complaint/return/refund/consult/technical/logistics/other（默认 other）；"
    "order_id 关联订单号（可选，能提取到订单号时务必传入，便于客服按单号跟进）。"
    "本工具没有账号参数也不需要账号：工单归属由服务端按当前登录账号自动写入。"
))
def create_ticket(
    description: str,
    priority: str = "normal",
    category: str = "other",
    order_id: str = "",
) -> str:
    """
    智能客服工单工具：
    - description: 问题描述（尽量完整记录用户诉求）
    - priority: 优先级 urgent/high/normal/low（默认 normal）
    - category: 分类 complaint/return/refund/consult/technical/logistics/other（填英文枚举值，默认 other）
    - order_id: 关联的订单号（可选；能提取到订单号时务必传入，便于客服按单号跟进）
    返回：工单号 + 处理承诺 + 后续查询方式与联系方式
    归属账号由服务端注入（agent.tools.tool_context），LLM 无法指定。
    """
    # 身份来自服务端登录态：读不到就是真的没有登录态，拒绝建单
    # （旧实现里 user_id 是模型填的参数，模型填错/被注入就会建出无归属或他人的工单）
    user_id = current_user_id()
    if not user_id:
        log.warning("create_ticket 缺少登录账号，拒绝建单")
        return TICKET_NEED_LOGIN

    ticket = create_ticket_record(user_id, description, priority, category, order_id)
    # AI 建单等于「转人工」，必须同时开启人工会话（写开场白），否则客服工作台看不到。
    # 局部导入：handoff_service 反过来依赖本模块的 create_ticket_record，模块级导入会成环。
    if ticket["category"] in _HANDOFF_CATEGORY:
        from service.handoff_service import start_handoff_session
        opening = start_handoff_session(ticket["ticket_id"])
        # 幂等命中已有工单时 start_handoff_session 会跳过开场白（返回 None），
        # 此时不应再计一次“转人工”，否则转人工率会被重复请求灌水。
        if opening:
            handoff_total.labels("agent_tool").inc()
    response_time = sla_cn(ticket["priority"])
    return TICKET_CREATED.format(
        ticket_id=ticket["ticket_id"],
        created_at=ticket["created_at"],
        sla=response_time,
        contact=SERVICE_CONTACT,
    )

