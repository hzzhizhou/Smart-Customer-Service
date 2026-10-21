"""
用户侧话术模板集中管理（话术与代码分离）。
企业做法：面向用户的回复只说人话，不暴露内部字段（优先级/分类编码/流转状态）；
运营调整文案只改本文件，无需翻业务逻辑。变量用 {} 占位符运行时填充。

注：用户端为纯文本渲染（<pre>），模板内不使用 Markdown 标记（避免星号外泄）。
"""

# ---------------- 枚举中文映射（用户侧展示用）----------------
CATEGORY_CN = {
    "complaint": "投诉",
    "return": "退货",
    "refund": "退款",
    "consult": "咨询",
    "technical": "技术支持",
    "logistics": "物流",
    "other": "其他",
}

PRIORITY_CN = {
    "urgent": "紧急",
    "high": "较紧急",
    "normal": "普通",
    "low": "较低",
}

STATUS_CN = {
    "open": "待处理",
    "processing": "处理中",
    "resolved": "已解决",
    "closed": "已关闭",
}

# 各优先级对应的人工响应预计时限（SLA，与工单优先级绑定）
# 对外只作「预计」表述：不承诺确定的完成时间，避免说到做不到。
SLA_CN = {
    "urgent": "30 分钟",
    "high": "2 小时",
    "normal": "24 小时",
    "low": "48 小时",
}


# 订单销售状态（业务数据，客服只读）
ORDER_STATUS_CN = {
    "pending": "待付款",
    "paid": "已付款",
    "shipped": "已发货",
    "delivered": "已签收",
    "refunding": "退款中",
    "refunded": "已退款",
    "cancelled": "已取消",
}

# 服务联系方式（所有对外回复统一附上，用户任何环节都能找到人）
SERVICE_CONTACT = (
    "客服热线：400-800-1234（24 小时）\n"
    "在线客服：每天 9:00-22:00，直接回复“转人工”，我们会安排人工客服接入"
)


# ---------------- 退货/退款业务环节（阶段化流转，对齐现实企业售后流程）----------------
# 现实退货不是“一步到位”，而是多环节跨多天：申请 → 商家审核 → 用户寄回 → 商家验收 → 发起退款 → 到账。
# 每个环节定义：中文名/简称、由谁推进、用户下一步要做什么、时效承诺、是否必经。
# key 与 tickets.stage 一一对应；运营调整环节文案只改本表，不动业务代码。
# required=False 表示「只有退货（要寄回实物）才走」：纯退款无货可寄，渲染轨迹与总步数时
# 必须跳过，否则会把压根不会发生的环节画成“待办”，等于替商家与支付渠道做承诺。
RETURN_STAGES = [
    {"key": "submitted", "required": True, "cn": "已提交申请，等待商家审核", "short": "提交申请", "owner": "商家",
     "next": "商家会在承诺时效内完成审核，审核结果我会在这里同步给您，请留意消息。",
     "sla": "24 小时内完成审核"},
    {"key": "approved", "required": True, "cn": "审核通过，等待您寄回商品", "short": "商家审核", "owner": "您",
     "next": "请保持商品及原包装、配件、赠品完整，在 7 天内寄回；寄出后把快递单号告诉我，方便为您跟踪。",
     "sla": "请您 7 天内寄出"},
    {"key": "shipped", "required": False, "cn": "商品已寄回，等待商家收货", "short": "寄回商品", "owner": "物流",
     "next": "商家签收后会立即验收，此环节您无需操作。",
     "sla": "签收后 48 小时内完成验收"},
    {"key": "received", "required": False, "cn": "商家已收货，正在验收", "short": "商家验收", "owner": "商家",
     "next": "验收通过后会立即为您发起退款，退款原路退回支付账户。",
     "sla": "1-3 个工作日内发起退款"},
    {"key": "refunding", "required": True, "cn": "退款已发起，正在原路退回", "short": "发起退款", "owner": "支付渠道",
     "next": "请留意原支付账户的到账通知，此环节您无需操作。",
     "sla": "1-3 个工作日到账"},
    {"key": "completed", "required": True, "cn": "退款已到账，本次售后已完成", "short": "退款到账", "owner": "—",
     "next": "无需其他操作。如仍未收到款项，请回复“转人工”，我们为您核实。",
     "sla": "已办结"},
]

# 非顺序流转的异常环节（不在进度条内，单独告知用户）
RETURN_STAGE_EXCEPTIONS = {
    "rejected": {
        "cn": "审核未通过", "short": "审核未通过",
        "next": "如需了解未通过原因或申请复核，请回复“转人工”，由人工客服为您核实处理。",
        "sla": "流程已终止",
    },
}

_STAGE_MAP = {s["key"]: s for s in RETURN_STAGES}


def stage_flow(category: str = "") -> list:
    """某售后分类实际要走的环节序列。

    纯退款（refund）不退货、无实物可寄，跳过 required=False 的「寄回商品 / 商家验收」；
    其余分类（含分类缺失）按完整退货流程走，与改动前行为一致。
    """
    if category == "refund":
        return [s for s in RETURN_STAGES if s["required"]]
    return list(RETURN_STAGES)


def stage_def(key: str) -> dict:
    """取环节定义（含异常环节）；未知环节返回带 key 的兜底定义。"""
    if key in _STAGE_MAP:
        return _STAGE_MAP[key]
    if key in RETURN_STAGE_EXCEPTIONS:
        return RETURN_STAGE_EXCEPTIONS[key]
    return {"key": key, "cn": key or "处理中", "short": key or "处理中", "owner": "商家",
            "next": "如需了解详情，请回复“转人工”，由人工客服为您跟进。", "sla": "以人工答复为准"}


def stage_cn(key: str) -> str:
    return stage_def(key)["cn"]


def stage_step(key: str, category: str = ""):
    """返回 (第几步, 该分类的总步数)；非顺序环节（如审核未通过）返回 (0, 总步数)。"""
    flow = stage_flow(category)
    idx = next((i for i, s in enumerate(flow) if s["key"] == key), None)
    return (idx + 1 if idx is not None else 0), len(flow)


def return_flow_brief(category: str = "") -> str:
    """全流程一句话概览，如「① 提交申请 → ② 商家审核 → …」；纯退款只列实际要走的环节。"""
    marks = "①②③④⑤⑥⑦⑧⑨"
    return " → ".join(f"{marks[i]} {s['short']}" for i, s in enumerate(stage_flow(category)))


def render_stage_timeline(history: list, current: str, category: str = "") -> str:
    """把环节流转记录渲染成用户可读的进度轨迹：✔ 已完成 / ▶ 当前 / ○ 待办。

    history 为 [{"stage": key, "at": 时间}] 列表（工单流转轨迹），
    时间缺失的已完成环节用「—」占位，未到的环节统一显示「待办」。
    category 决定轨迹长度：纯退款不退货，跳过「寄回商品 / 商家验收」，
    避免把不会发生的环节画成“待办”当成对用户的承诺。
    """
    time_map = {}   # 环节 key → 进入该环节的时间（同一环节多次流转取首次）
    for h in history or []:
        if isinstance(h, dict):
            time_map.setdefault(h.get("stage"), h.get("at", ""))
    flow = stage_flow(category)
    cur_idx = next((i for i, s in enumerate(flow) if s["key"] == current), -1)
    lines = []
    for i, s in enumerate(flow):
        at = time_map.get(s["key"], "")
        if i < cur_idx:
            mark = "✔"
        elif i == cur_idx:
            mark = "▶"
        else:
            mark, at = "○", "待办"
        lines.append(RETURN_PROGRESS_LINE.format(
            mark=mark, no=i + 1, short=s["short"], at=at or "—"))
    return "\n".join(lines)


def category_cn(v: str) -> str:
    return CATEGORY_CN.get(v, v or "")


def order_status_cn(v: str) -> str:
    return ORDER_STATUS_CN.get(v, v or "")


def priority_cn(v: str) -> str:
    return PRIORITY_CN.get(v, v or "")


def status_cn(v: str) -> str:
    return STATUS_CN.get(v, v or "")


def sla_cn(v: str) -> str:
    return SLA_CN.get(v, "24 小时")


# ---------------- 场景话术模板 ----------------
# 完整度标准（企业客服话术四要素）：当前环节 + 已完成什么 + 下一步做什么 + 时效与联系方式。
# 1) 工单创建成功（用户侧只给工单号/时间/处理时效，不显示优先级、分类等内部字段）
TICKET_CREATED = (
    "已为您提交人工处理，工单号：{ticket_id}\n"
    "提交时间：{created_at}\n"
    "预计 {sla}内会有客服与您联系，请保持电话或在线畅通。\n"
    "您可以随时回复“工单进度”了解处理情况。\n"
    "\n"
    "{contact}"
)

# 2) 工单进度（答复回传：状态 + 人工处理意见）
TICKET_PROGRESS_WITH_NOTE = (
    "您提交的{subject}申请（工单号：{ticket_id}）当前状态：{status}\n"
    "提交时间：{created_at}\n"
    "人工客服处理意见：{note}\n"
    "\n"
    "{contact}"
)
TICKET_PROGRESS_NO_NOTE = (
    "您提交的{subject}申请（工单号：{ticket_id}）当前状态：{status}\n"
    "提交时间：{created_at}\n"
    "人工客服正在跟进处理中，暂未填写处理意见，请您耐心等待。\n"
    "\n"
    "{contact}"
)

# 3) 查询类提示
NEED_ORDER_NO = "请提供订单号，以便为您查询工单处理进度。"
NO_TICKET = (
    "暂时没有查询到您名下的工单记录。如需退货、退款或转人工，"
    "请先说明您遇到的问题，我帮您登记后即可随时查看进度。"
)

# 4) 退货/退款主流程
RETURN_ASK_MISSING = "好的，我来帮您办理{subject}。\n还需要您补充：{ask}。{hint}"
RETURN_CONFIRM = (
    "已为您核对{subject}信息：\n"
    "订单号：{order_id}\n"
    "{subject}原因：{reason}\n"
    "\n"
    "确认后我会立即为您登记工单，并写明后续每一步流程、由谁处理和时效承诺。\n"
    "回复“确认”即可提交；如需修改，直接告诉我即可。"
)
RETURN_CLARIFY = (
    "我理解您想办理{subject}。请确认上面的订单号与原因是否正确，"
    "如需修改，直接告诉我即可。"
)
RETURN_RECORD_FALLBACK = "已为您记录{subject}需求，人工客服稍后会与您核实，请留意消息。"
RETURN_ALREADY_DONE = (
    "您上一笔{subject}申请已登记完成；本次按新申请处理，尚未创建工单。\n"
    "如需再次{subject}，请告知订单号和{subject}原因；"
    "查询上一笔进度可直接回复“{subject}进度”。"
)
RETURN_DUPLICATED = (
    "该订单的{subject}申请已登记（工单号：{ticket_id}），无需重复提交。\n"
    "回复“{subject}进度”即可查看当前办到哪一步。"
)

# 登记成功（完整告知：单号 / 当前环节 / 全流程 / 下一步 / 时效 / 联系方式）
RETURN_SUBMITTED = (
    "已完成登记，请您放心，后续每个环节我都会同步给您。\n"
    "\n"
    "工单号：{ticket_id}\n"
    "订单号：{order_id}\n"
    "提交时间：{created_at}\n"
    "当前环节：{step}/{total} · {stage_cn}\n"
    "\n"
    "{subject}完整流程（共 {total} 步）：{flow}\n"
    "\n"
    "下一步：{next}\n"
    "时效承诺：{sla}\n"
    "\n"
    "随时回复“{subject}进度”，即可查看办到哪一步、还剩哪些环节。\n"
    "\n"
    "{contact}"
)

# 进度查询（完整告知：当前环节 / 流转轨迹 / 下一步 / 时效 / 人工意见 / 联系方式）
RETURN_PROGRESS = (
    "【{subject}进度查询】\n"
    "工单号：{ticket_id}\n"
    "订单号：{order_id}\n"
    "当前环节：{step}/{total} · {stage_cn}\n"
    "最近更新：{updated_at}\n"
    "\n"
    "流转轨迹：\n"
    "{timeline}\n"
    "\n"
    "下一步：{next}\n"
    "时效承诺：{sla}\n"
    "{note_line}"
    "\n"
    "{contact}"
)
# 进度轨迹单行：{mark} 序号 环节简称 时间（{mark} 为 ✔ 已完成 / ▶ 当前 / ○ 待办）
RETURN_PROGRESS_LINE = "{mark} {no}. {short}　{at}"
RETURN_PROGRESS_NOTE = "人工客服处理意见：{note}\n"
# 非顺序环节（审核未通过等）：不展示进度轨迹
RETURN_PROGRESS_ABNORMAL = (
    "【{subject}进度查询】\n"
    "工单号：{ticket_id}\n"
    "订单号：{order_id}\n"
    "当前环节：{stage_cn}\n"
    "最近更新：{updated_at}\n"
    "\n"
    "{subject}流程已在「{stage_cn}」终止，未完成后续环节。\n"
    "下一步：{next}\n"
    "{note_line}"
    "\n"
    "{contact}"
)

# 5) 异常兜底
COMPLAINT_FALLBACK = "非常抱歉给您带来不好的体验，我这就为您转接人工客服，请稍候。"
PROGRESS_FALLBACK = "抱歉，暂时没有查到您的工单进度。您可以提供订单号再试一次。"
# 确定性订单查询答复的结尾提示（订单详情块由 query_order 返回，自带字段标签，无需另加开场）
ORDER_TAIL = "如需退货、退款或有其他问题，请随时告诉我。"
# 订单不属于当前登录账号时的答复：不透露任何订单字段（客户/金额/物流都算隐私），
# 只说明查不到，并给出「换账号登录 / 转人工核实」两条出路
ORDER_FORBIDDEN = (
    "该订单不在当前登录账号名下，我无法为您查询，也不能透露订单信息。\n"
    "如果这是您的订单，请用下单时使用的账号登录后再查；"
    "如果您确认订单归属无误，回复“转人工”，我为您转接人工客服核实。"
)
# 「查我名下订单」列表答复：只列本账号订单，不含客户姓名（自己名下的单无需再公示姓名）
MY_ORDERS_HEADER = "您名下共 {total} 张订单，以下是最新的 {shown} 张："
MY_ORDERS_LINE = "{order_id}｜{status}｜{product}｜¥{amount:.2f}｜下单时间 {created_at}"
MY_ORDERS_MORE = "如需查看更早的订单，请告诉我订单号，我为您单独查询。"
MY_ORDERS_EMPTY = (
    "当前登录账号名下还没有订单。\n"
    "如果订单是用其他账号下的，请换用下单账号登录后再查；"
    "也可以直接把订单号告诉我，我帮您核实。"
)
# 拿不到账号标识（工具被无登录态调用）时的安全侧失败：宁可列不出来，也不猜账号
MY_ORDERS_NEED_LOGIN = (
    "抱歉，我没能确认您的登录账号，无法列出名下订单。\n"
    "请重新登录后再试，或直接把订单号告诉我，我帮您查询。"
)
# 拿不到登录账号时的安全侧失败：不建单、不查工单。
# 身份改由服务端注入后，工具不再接受账号参数，因此这里读不到账号就是真的没有登录态
# （后台任务、脚本、离线评测），一律拒绝 —— 否则会建出无归属工单、或退化成全库匹配。
TICKET_NEED_LOGIN = (
    "抱歉，我没能确认您的登录账号，无法为您创建或查询工单。\n"
    "请重新登录后再试；如果是紧急问题，也可以直接联系客服。"
)
# Agent 只输出了推理过程（无面向用户的正文）/检索不可用时的兜底。
# 这里不得声称“已为您转接”：该分支不建单，前端轮询 /handoff/active 拿不到工单，
# 用户会在无人接听的状态下一直等。改为如实告知 + 把转人工与其他出路作为选项问出来。
AGENT_NO_ANSWER_FALLBACK = (
    "抱歉，根据当前知识库我暂时无法回答这个问题，不想给您不准确的答复。\n"
    "需要我为您转接人工客服吗？直接回复“转人工”即可；"
    "也可以换个问法，或说明您要办理的业务（查订单、退货、退款），我继续帮您。"
)
# 知识库未检索到可靠依据时的兜底：不硬答，明确告知并给出求助渠道
KNOWLEDGE_NO_ANSWER = (
    "抱歉，这个问题我暂时没有查到可靠的资料，不敢给您不准确的答复。\n"
    "{contact}"
)

# ---------------- 人工会话（转人工实时对话）----------------
# 会话开场白：写入 ticket_messages 的第一条 system 消息。
# 「工单是否已有消息」就是「是否处于人工会话」的判据，因此这条文案同时也是人工会话的起点标记。
HANDOFF_OPENING = (
    "正在为您转接人工客服（工单号 {ticket_id}），请稍候……\n"
    "请补充您的问题细节，人工客服看到后会尽快回复。"
)
# 人工客服结束会话后，推给用户端的系统提示
HANDOFF_CLOSED = "本次人工服务已结束。您仍可以继续向智能客服提问。"