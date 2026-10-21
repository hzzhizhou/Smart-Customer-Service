"""确定性业务规则：关键词、阈值、正则与纯判定函数。

只放「业务规则」，不放环境配置（配置统一在 config/settings.py）。
这些规则用于在 LLM 之前做确定性判定（是否列我名下订单、ReAct 推理文本清洗等），
全部是纯函数、无副作用，单测可直接覆盖，改动前建议先跑 tests/。
"""
import re
from typing import List

# ---------- 确定性工单进度查询 ----------
# 触发关键词（命中即读工单人工备注回传，绕过 LLM 工具选择）
PROGRESS_KEYWORDS = (
    "工单", "售后处理", "处理进度", "处理结果", "处理得", "处理了没",
    "投诉结果", "投诉处理", "人工处理", "跟进", "办得怎么样", "办好了吗",
    "转人工后", "有没有结果", "结果出来", "批复",
    # 退货/退款进度类追问（用户提交后最常问法）
    "退货进度", "退款进度", "退货的进度", "退款的进度",
    "退货怎么样", "退款怎么样", "退货处理", "退款处理",
    "退款到账", "退到哪", "退货没", "退了没", "退了吗", "退款吗",
)

# ---------- DST 主线换话题判定 ----------
# DST 主线（退货/退款等）进行中，仍要允许用户中途换话题：当本轮问题命中其他业务线的
# 强关键词、且原始得分达此阈值时，不再复用主线意图。用原始得分而非归一化置信度，
# 是为了让槽位补答（如回答「质量问题」仅 0.5 分）继续留在主线里，不被打断。
MAINLINE_SWITCH_SCORE = 2.0
# 允许让位给 order/logistics 的问法特征：带明确"去查一下"的动作。缺了它，一句纯订单号
# 也会被当成换话题，把退货主线的意图与槽位一起冲掉（详见 RAGService._enrich_messages_with_intent）。
MAINLINE_SWITCH_HINTS = ("查", "看看", "看下", "看一下", "到哪", "到哪儿", "什么状态",
                         "进度", "怎么样", "物流")

# ---------- 确定性「我名下订单」列表查询 ----------
# 触发关键词（命中即按归属账号查库回传，绕过 LLM）。
# 背景：用户不带订单号问"我有哪些订单"时，模型倾向于反复追问订单号，答不到点上；
# 而"列出本账号订单"本身是可确定性完成、且必须只查本账号的动作，改由规则保证必达。
_MY_ORDERS_KEYWORDS = (
    "我的订单", "我名下", "名下订单", "我的所有订单", "我的全部订单",
    "所有订单", "全部订单", "历史订单", "订单列表", "有哪些订单", "有哪几个订单",
    "买过什么", "买了哪些", "买过哪些", "都买了什么", "我的单子",
)
# 这些问法问的是「怎么查」而不是「帮我查」，属知识咨询，不能触发列表查询
_MY_ORDERS_METHOD_HINTS = ("怎么查", "如何查", "在哪查", "哪里查", "怎么查看", "如何查看")
# 句中含这些「办理诉求」词时按业务处理，不做列表查询：
# 例如"我的订单有质量问题"是要退货、"我的订单退款进度"是要查进度，都不该被当成长列表诉求。
_MY_ORDERS_BLOCK_HINTS = (
    "退", "换", "投诉", "坏", "破损", "质量", "物流", "到哪", "发货", "取消",
    "改地址", "发票", "保修", "催", "丢", "少", "错了",
)


def is_my_orders_query(question: str) -> bool:
    """判断是否为「列出我名下订单」的诉求（四个条件缺一不可）。

    - 命中列表类关键词；
    - 句中不含具体订单号（有单号属于精确查询，走带单号的分支）；
    - 不是在问「怎么查」（属知识咨询）；
    - 句中不含办理诉求词（"我的订单有质量问题"是要退货，不是要看列表）。
    """
    q = question or ""
    if re.search(r"SO\d+", q):
        return False
    if not any(k in q for k in _MY_ORDERS_KEYWORDS):
        return False
    if any(h in q for h in _MY_ORDERS_METHOD_HINTS):
        return False
    return not any(b in q for b in _MY_ORDERS_BLOCK_HINTS)


# ---------- 多问题拆分（一次提问包含多个子问题） ----------
# 用户常一次抛多个问题（「iPhone15 电池容量多大？另外运费怎么算？」）。若把整段拼成
# 一个查询做检索 + 重排，各子问题的候选会互相挤占 RERANK_TOP_N 名额，靠后的子问题被
# 整段挤出结果，生成层拿不到它的资料 → 该问静默消失（实测 C1/C2 端到端复现）。
# 故检索前先做查询分解：按强分隔符与显式连接词切成独立子问题，各自检索后再合并
# （接入点见 retrieval/service.py 的 _decompose_queries）。
_SUBQ_SEPARATOR_RE = re.compile(r"[?？;；\n]+")
# 连接词必须出现在句首或标点/空白之后才算分界：「还有没有别的颜色」里的"还有"不算
_SUBQ_CONNECTOR_RE = re.compile(
    r"(?:^|[，,。.\s]+)(?P<c>另外|还有|以及|顺便|同时|再有|再问|还想问|另外就是)")
SUBQ_MIN_LEN = 4   # 短于此长度的片段视为切分噪声（如「呢」「谢谢」），丢弃
_SUBQ_TRIM = " \t\r\n,，。.、:："


def _iter_split_connectors(text: str):
    """遍历真正起分界作用的连接词（排除「还有没有别的颜色」这类非分界用法）。"""
    for m in _SUBQ_CONNECTOR_RE.finditer(text):
        # 「还有没有别的颜色」= 是否还有，不是新问题的开头
        if m.group("c") == "还有" and text[m.end():].startswith("没"):
            continue
        yield m


def _split_by_connectors(text: str) -> List[str]:
    """按显式连接词切分；只有连接词之前已有实质内容时才切。"""
    parts, start = [], 0
    for m in _iter_split_connectors(text):
        head = text[start:m.start()]          # m.start() 指向连接词前的标点/句首，正好一并切掉
        stripped = head.strip(_SUBQ_TRIM)
        if len(stripped) >= SUBQ_MIN_LEN:
            parts.append(head)
            start = m.end()
        elif not stripped:
            # 连接词落在片段开头（上一级分隔符已把句子切开）→ 只需丢掉连接词本身
            start = m.end()
    parts.append(text[start:])
    return parts


def split_sub_questions(question: str) -> List[str]:
    """把一次提问切成若干独立子问题（纯函数、无副作用、零 LLM 成本）。

    - 切分依据：强分隔符（？/?/；/换行）+ 显式连接词（另外/还有/以及/顺便…）；
    - 逗号不算分隔符（句内停顿太常见，切了会把一个问句切碎）；
    - 拆不出 ≥2 个有效片段（含空串、纯标点、过短噪声）时原样返回 [question]，
      调用方据此判断"本轮只有一个问题"，不做任何额外开销。
    """
    text = (question or "").strip()
    if not text:
        return []
    raw: List[str] = []
    for chunk in _SUBQ_SEPARATOR_RE.split(text):
        raw.extend(_split_by_connectors(chunk))
    cleaned: List[str] = []
    for part in raw:
        part = part.strip(_SUBQ_TRIM)
        if len(part) >= SUBQ_MIN_LEN and part not in cleaned:
            cleaned.append(part)
    return cleaned if len(cleaned) >= 2 else [text]


def looks_like_multi_question(question: str) -> bool:
    """规则拆不出多段时，判断文本「看起来像多问题」（决定是否值得花一次 LLM 拆分）。

    两个问号及以上、或出现显式连接词，都说明用户想一次问好几件事；
    只是规则没切干净（如用分号/括号混排），这时才让 LLM 兜底拆一次。
    """
    text = question or ""
    if len(_SUBQ_SEPARATOR_RE.findall(text)) >= 2:
        return True
    return any(_iter_split_connectors(text))


# 模型偶发把「无 / 无相关信息」这类空答复当成正文吐出来（多见于没带订单号的查询类问题，
# 如实测的「如何查询订单？」），对用户毫无价值，按「没答上来」处理，转确定性知识兜底。
_EMPTY_ANSWER_RE = re.compile(
    r"^(无|没有|暂无|无相关信息|暂无相关信息|未找到|未找到相关信息|none|n/?a)[。.!！?？\s]*$",
    re.I,
)


def is_empty_answer(text: str) -> bool:
    """判断模型正文是否为「什么都没答」的空答复。"""
    return bool(_EMPTY_ANSWER_RE.match(text.strip()))


# 空答复的候选写法：用于流式场景下判断「已收到的前缀还有可能是空答复吗」。
# 必须在收到整段之后才能判定是不是空答复，所以这几个词开头的片段要押后输出；
# 一旦前缀偏离所有候选，说明这是正常回答，立刻 flush 转入直通（只多等一两个 chunk）。
_EMPTY_ANSWER_CANDIDATES = (
    "无相关信息", "暂无相关信息", "未找到相关信息", "无", "没有", "暂无", "未找到",
    "none", "n/a", "na",
)


def could_be_empty_answer(text: str) -> bool:
    """判断已收到的前缀是否仍「有可能」是空答复（流式缓冲用；空串视为可能）。"""
    t = text.strip()
    if not t:
        return True
    t = t.rstrip("。.!！?？ \t\r\n")
    return any(cand.startswith(t) for cand in _EMPTY_ANSWER_CANDIDATES)