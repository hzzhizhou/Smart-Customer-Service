"""
用户侧话术模板集中管理（话术与代码分离）。
三条写作规则（企业客服文案的通用约束）：
  1. 只说人话：不暴露优先级、分类编码、内部状态等技术字段；
  2. 一句一个意思：能用短句就不用长句，去掉「请您放心」「我们会尽快」这类无信息量的垫话；
  3. 事实不可编：单号、时间、金额、时效只能来自模板占位符，不允许由模型自由生成。
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

# 各优先级对应的人工首次响应时限（SLA）。对外一律说「预计」：
# 承诺确定的完成时间，等于替客服和支付渠道做保证，做不到就是失信。
SLA_CN = {
    "urgent": "30 分钟内",
    "high": "2 小时内",
    "normal": "24 小时内",
    "low": "48 小时内",
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

# 服务联系方式：两行足够，不必每次都写成完整说明（用户只需要知道怎么找到人）
SERVICE_CONTACT = "客服热线 400-800-1234（24 小时）\n在线客服 9:00-22:00，回复“转人工”"


# ---------------- 售后环节（阶段化流转）----------------
# 现实退货不是一步到位：申请 → 审核 → 寄回 → 验收 → 退款 → 到账。每个环节定义
RETURN_STAGES = [
    # 各场景共用
    {"key": "submitted", "cn": "待商家确认", "short": "提交申请",
     "next": "商家确认结果会在这里同步给您，请留意消息。", "sla": "24 小时内确认"},
    {"key": "refunding", "cn": "退款中", "short": "发起退款",
     "next": "请留意原支付账户的到账通知，您无需操作。", "sla": "1-3 个工作日到账"},
    {"key": "completed", "cn": "已到账", "short": "退款到账",
     "next": "无需其他操作。未收到款项请回复“转人工”。", "sla": "已办结"},
    # 已签收（货在用户手上）：寄回 → 验收
    {"key": "approved", "cn": "待您寄回", "short": "商家审核",
     "next": "请保持商品、配件、赠品完整，7 天内寄回；寄出后把快递单号告诉我。", "sla": "7 天内寄出"},
    {"key": "shipped", "cn": "退回中", "short": "寄回商品",
     "next": "商家签收后会立即验收，您无需操作。", "sla": "签收后 48 小时内验收"},
    {"key": "received", "cn": "验收中", "short": "商家验收",
     "next": "验收通过后立即发起退款，原路退回。", "sla": "1-3 个工作日发起退款"},
    # 已付款未发货（无货可寄）
    {"key": "intercepting", "cn": "待商家取消", "short": "商家拦截",
     "next": "商家确认后直接原路退款，未发货不需要寄回商品。", "sla": "1 个工作日内确认"},
    # 已发货未签收（在途）
    {"key": "rejecting", "cn": "待确认拒收", "short": "拒收包裹",
     "next": "拒收包裹或等待物流退回即可，不用您自行寄回。", "sla": "1 个工作日内确认"},
    {"key": "returning", "cn": "退回中", "short": "物流退回",
     "next": "包裹退回商家后立即验收，您无需操作。", "sla": "3-7 天退回"},
]

# 非顺序流转的异常环节（不在进度条内，单独告知）
RETURN_STAGE_EXCEPTIONS = {
    "rejected": {
        "cn": "审核未通过", "short": "审核未通过",
        "next": "了解原因或申请复核，请回复“转人工”。",
        "sla": "流程已终止",
    },
}

_STAGE_MAP = {s["key"]: s for s in RETURN_STAGES}

# 四种场景的环节序列。顺序即用户看到的进度条，也是「第几步/共几步」的依据。
FLOW_BY_SCENARIO = {
    "undelivered": ["submitted", "intercepting", "refunding", "completed"],
    "in_transit": ["submitted", "rejecting", "returning", "received", "refunding", "completed"],
    "received": ["submitted", "approved", "shipped", "received", "refunding", "completed"],
    "refund_only": ["submitted", "approved", "refunding", "completed"],
}


def scenario_of(category: str = "", order_status: str = "") -> str:
    """该走哪套流程：先看商品在哪，再看是否只退款不退货。

    顺序不能反：货还在商家仓库时，「要不要寄回」这个问题根本不成立，
    此时即使分类是 refund，流程也应落在 undelivered 而不是 refund_only。
    """
    status = (order_status or "").strip().lower()
    if status in ("pending", "paid"):        # 未发货：无货可寄
        return "undelivered"
    if status == "shipped":                  # 已发货未签收：在途
        return "in_transit"
    if (category or "").strip().lower() == "refund":
        return "refund_only"                 # 已签收但只退款不退货
    return "received"                        # 已签收；状态未知时按保守流程


def stage_flow(category: str = "", order_status: str = "") -> list:
    """某次售后实际要走的环节列表（未发货 / 在途 / 已签收 / 仅退款各不相同）。"""
    keys = FLOW_BY_SCENARIO[scenario_of(category, order_status)]
    return [_STAGE_MAP[k] for k in keys]


def stage_def(key: str) -> dict:
    """取环节定义（含异常环节）；未知环节返回兜底定义。"""
    if key in _STAGE_MAP:
        return _STAGE_MAP[key]
    if key in RETURN_STAGE_EXCEPTIONS:
        return RETURN_STAGE_EXCEPTIONS[key]
    return {"key": key, "cn": key or "处理中", "short": key or "处理中", "owner": "商家",
            "next": "了解详情请回复“转人工”。", "sla": "以人工答复为准"}


def stage_cn(key: str) -> str:
    return stage_def(key)["cn"]


def stage_step(key: str, category: str = "", order_status: str = ""):
    """返回 (第几步, 总步数)；非顺序环节（如审核未通过）返回 (0, 总步数)。"""
    flow = stage_flow(category, order_status)
    idx = next((i for i, s in enumerate(flow) if s["key"] == key), None)
    return (idx + 1 if idx is not None else 0), len(flow)


def return_flow_brief(category: str = "", order_status: str = "") -> str:
    """全流程一句话概览，如「① 提交申请 → ② 商家拦截 → …」。"""
    marks = "①②③④⑤⑥⑦⑧⑨"
    return " → ".join(f"{marks[i]} {s['short']}"
                      for i, s in enumerate(stage_flow(category, order_status)))


def render_stage_timeline(history: list, current: str,
                         category: str = "", order_status: str = "") -> str:
    """把环节流转记录渲染成用户可读的进度轨迹：✔ 已完成 / ▶ 当前 / ○ 待办。

    history 为 [{"stage": key, "at": 时间}]；时间缺失的已完成环节用「—」占位，
    未到的环节统一显示「待办」。category 决定轨迹长度（纯退款不显示寄回/验收）。
    """
    time_map = {}   # 环节 key → 进入时间（同一环节多次流转取首次）
    for h in history or []:
        if isinstance(h, dict):
            time_map.setdefault(h.get("stage"), h.get("at", ""))
    flow = stage_flow(category, order_status)
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
    return CATEGORY_CN.get((v or "").strip().lower(), v or "其他")


def order_status_cn(v: str) -> str:
    return ORDER_STATUS_CN.get((v or "").strip().lower(), v or "未知")


def priority_cn(v: str) -> str:
    return PRIORITY_CN.get((v or "").strip().lower(), v or "普通")


def status_cn(v: str) -> str:
    return STATUS_CN.get((v or "").strip().lower(), v or "处理中")


def sla_cn(priority: str) -> str:
    return SLA_CN.get((priority or "").strip().lower(), SLA_CN["normal"])


# ---------------- 场景话术 ----------------
# 通用骨架：当前状态 + 下一步 + 时效（或求助方式）。字段标签用「｜」分隔，一行一个意思。
# 1) 工单创建成功（用户侧只给工单号/时间/时效，不显示优先级、分类等内部字段）
TICKET_CREATED = (
    "已提交人工，工单号：{ticket_id}\n"
    "提交时间：{created_at}｜预计 {sla}联系您\n"
    "回复“工单进度”可随时查看。\n"
    "\n"
    "{contact}"
)

# 2) 工单进度（状态 + 人工处理意见）
TICKET_PROGRESS_WITH_NOTE = (
    "工单号：{ticket_id}｜{subject}\n"
    "当前状态：{status}\n"
    "处理意见：{note}\n"
    "\n"
    "{contact}"
)
TICKET_PROGRESS_NO_NOTE = (
    "工单号：{ticket_id}｜{subject}\n"
    "当前状态：{status}\n"
    "人工客服正在跟进，暂未填写处理意见。\n"
    "\n"
    "{contact}"
)

# 3) 查询类提示
NEED_ORDER_NO = "请提供订单号，我帮您查进度。"
NO_TICKET = (
    "没查到您名下的工单。\n"
    "如需退货、退款或转人工，说明一下问题，我帮您登记。"
)

# 4) 退货/退款主流程
# 「已进入办理流程」的标志话术，改文案要同步改那个常量，否则评测会静默失准。
RETURN_ASK_MISSING = "好的，为您办理{subject}。\n还需要您补充：{ask}。{hint}"
RETURN_CONFIRM = (
    "请确认以下信息：\n"
    "订单号：{order_id}\n"
    "原因：{reason}\n"
    "\n"
    "回复“确认”即提交；需修改直接告诉我。"
)
RETURN_CLARIFY = "请确认上面的订单号与原因是否正确，需修改直接告诉我。"
RETURN_RECORD_FALLBACK = "已记录{subject}需求，人工客服稍后与您核实。"
RETURN_ALREADY_DONE = (
    "上一笔{subject}申请已登记完成；本次按新申请处理，尚未创建工单。\n"
    "再办一次请给订单号和原因；查上一笔回复“{subject}进度”。"
)
RETURN_DUPLICATED = (
    "该订单的{subject}已登记（工单号：{ticket_id}），无需重复提交。\n"
    "回复“{subject}进度”可查看办到哪一步。"
)

# 登记成功：单号 / 当前环节 / 全流程 / 下一步 / 时效
RETURN_SUBMITTED = (
    "已登记，工单号：{ticket_id}\n"
    "订单号：{order_id}｜提交时间：{created_at}\n"
    "当前：{step}/{total} · {stage_cn}\n"
    "\n"
    "流程：{flow}\n"
    "下一步：{next}\n"
    "时效：{sla}\n"
    "\n"
    "回复“{subject}进度”可随时查看。\n"
    "\n"
    "{contact}"
)

# 进度查询：只保留「当前在哪、走过哪些、下一步」，不重复提交时间等已知信息
RETURN_PROGRESS = (
    "【{subject}进度】\n"
    "工单号：{ticket_id}｜订单号：{order_id}\n"
    "当前：{step}/{total} · {stage_cn}\n"
    "最近更新：{updated_at}\n"
    "\n"
    "{timeline}\n"
    "\n"
    "下一步：{next}\n"
    "时效：{sla}\n"
    "{note_line}"
    "\n"
    "{contact}"
)
# 轨迹单行：{mark} 序号 环节简称 时间（✔ 已完成 / ▶ 当前 / ○ 待办）
RETURN_PROGRESS_LINE = "{mark} {no}. {short}　{at}"
RETURN_PROGRESS_NOTE = "处理意见：{note}\n"
# 非顺序环节（审核未通过等）：不展示轨迹
RETURN_PROGRESS_ABNORMAL = (
    "【{subject}进度】\n"
    "工单号：{ticket_id}｜订单号：{order_id}\n"
    "当前：{stage_cn}\n"
    "最近更新：{updated_at}\n"
    "\n"
    "流程已在「{stage_cn}」终止，后续环节不再进行。\n"
    "下一步：{next}\n"
    "{note_line}"
    "\n"
    "{contact}"
)

# 5) 异常兜底
COMPLAINT_FALLBACK = "抱歉给您带来不好的体验，正在为您转接人工，请稍候。"
# 主流程异常：不能空口承诺转接（异常分支没有建单），只如实说明 + 给核实路径
RETURN_SYSTEM_ERROR = (
    "抱歉，办理退货/退款时遇到系统问题，暂时无法确认本次操作的结果。\n"
    "请稍后重试，或回复“退货进度”核实；需要人工协助请回复“转人工”。"
)
PROGRESS_FALLBACK = "没查到工单进度，可以提供订单号再试一次。"
# 确定性订单查询答复的结尾提示（订单详情块自带字段标签，无需加开场）
ORDER_TAIL = "还需要其他帮助，随时告诉我。"
# 订单不属于当前登录账号：不透露任何字段（客户/金额/物流都算隐私），只给两条出路
ORDER_FORBIDDEN = (
    "该订单不在当前登录账号名下，无法查询，也不能透露订单信息。\n"
    "如果这是您的订单，请用下单账号登录后再查；"
    "确认归属无误，回复“转人工”由人工核实。"
)
# 「查我名下订单」列表：只列本账号订单，不含客户姓名
MY_ORDERS_HEADER = "您名下共 {total} 张订单，以下是最新的 {shown} 张："
MY_ORDERS_LINE = "{order_id}｜{status}｜{product}｜¥{amount:.2f}｜{created_at}"
MY_ORDERS_MORE = "查看更早的订单，请告诉我订单号。"
MY_ORDERS_EMPTY = (
    "当前账号名下还没有订单。\n"
    "如果订单是用其他账号下的，请换下单账号登录；"
    "也可以把订单号告诉我，我帮您核实。"
)
# 拿不到账号标识（无登录态调用）时的安全侧失败：宁可列不出来，也不猜账号
MY_ORDERS_NEED_LOGIN = (
    "没能确认您的登录账号，无法列出名下订单。\n"
    "请重新登录后再试，或直接把订单号告诉我。"
)
# 无登录态一律拒绝建单/查工单：否则会建出无归属工单、或退化成全库匹配
TICKET_NEED_LOGIN = (
    "没能确认您的登录账号，无法创建或查询工单。\n"
    "请重新登录后再试；紧急问题可直接联系客服。"
)
# Agent 只输出推理过程（无用户正文）/检索不可用时的兜底。
# 不得声称“已为您转接”：该分支不建单，前端拿不到工单，用户会一直等——
# 所以把转人工作为选项问出来，而不是替用户决定。
AGENT_NO_ANSWER_FALLBACK = (
    "抱歉，这个问题我在知识库里没找到依据，不想给您不准确的信息。\n"
    "需要转人工吗？回复“转人工”即可；也可以换个问法，"
    "或说明要办的业务（查订单、退货、退款），我继续帮您。"
)
# 知识库未检索到可靠依据：不硬答，明确告知并给出求助渠道
KNOWLEDGE_NO_ANSWER = (
    "抱歉，这个问题我暂时没有查到可靠的资料，不敢给您不准确的答复。\n"
    "需要转人工吗？回复“转人工”即可。\n"
    "{contact}"
)

# 「未作答」判定信号：回答命中任一片段，即视为没答上来（用于多意图补齐等场景）。
# 集中在这里，避免调用方各写一份字符串、改文案时漏改。
NO_ANSWER_SIGNALS = (
    "没查到可靠依据",
    "答不上来",
    "没有查到可靠的资料",
    "不敢给您不准确的答复",
    "无相关信息",
)

# ---------------- 人工会话（转人工实时对话）----------------
# 会话开场白：写入 ticket_messages 的第一条 system 消息。
# 「工单是否已有消息」就是「是否处于人工会话」的判据，因此这条文案也是会话起点标记。
HANDOFF_OPENING = (
    "正在转接人工客服（工单号 {ticket_id}），请稍候……\n"
    "请补充问题细节，人工客服看到后会尽快回复。"
)
# 人工客服结束会话后，推给用户端的系统提示
HANDOFF_CLOSED = "人工客服已结束本次会话，您仍可继续向智能客服提问。"
