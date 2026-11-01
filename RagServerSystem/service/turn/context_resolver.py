"""多轮上下文补全：把「那能改地址吗」补成「订单 ORD123 能不能改地址」。

问题
----
单轮链路只处理当前这句话。多轮场景里第二句往往省略了实体：

    用户：我的订单到哪了
    用户：那能改地址吗          ← 这句话里没有任何订单号

此时订单查询分支拿不到 order_id，只能反问「请提供订单号」，用户体验直接断掉。
而订单号其实就在上一轮的会话状态里。

为什么用规则而不是让模型做
--------------------------
1. **确定性**：订单号/工单号是强格式实体，正则比模型可靠，且不会把上一单的号码带到新问题上；
2. **成本与延迟**：每轮多一次模型调用，多轮对话下成本会乘 2 以上；
3. **可解释**：命中哪条规则、继承了什么，能直接写进日志，出问题可复现。

模型补全留作下一层：规则没命中且有明确指代词时，才值得付一次调用。当前不接入。

边界（明确不做的事）
--------------------
- **不改写原始问题**：`resolve()` 只产出「可继承的槽位」，不改 `question`。
  原因是下游确定性分支用正则解析原句，前面加实体前缀会改变解析结果。
  需要增强检索时用 `augment_query=True` 显式开启，且应先做 A/B 对照再上。
- **不做跨会话记忆**：只继承当前会话最近若干轮的实体，换会话即失效。
- **不做冲突消解**：新问题里出现新的订单号时一律以新值为准，不与旧值比对（用户换单是常态）。
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# ---- 实体识别 ----
# 带前缀的订单号最可靠（ORD20250101001 / SO-20250101-001）
_ORDER_PREFIX_RE = re.compile(r"\b(?:ORD|SO|DD|PO)[-_]?\d{6,20}\b", re.IGNORECASE)
# 上下文词引导：「订单号 12345678」「工单是 TK123456」
_ORDER_CTX_RE = re.compile(r"(?:订单号?|单号)\s*(?:是|为|:|：)?\s*([A-Za-z0-9][A-Za-z0-9\-]{3,31})")
_TICKET_PREFIX_RE = re.compile(r"\b(?:TK|GD|WO|WO|T)\d{6,20}\b", re.IGNORECASE)
_TICKET_CTX_RE = re.compile(r"(?:工单号?|服务单号?)\s*(?:是|为|:|：)?\s*([A-Za-z0-9][A-Za-z0-9\-]{3,31})")

# ---- 指代/省略信号：出现这些词，说明这句话依赖上文 ----
REFERENTIAL = ("它", "这个", "那个", "这单", "那单", "该订单", "该工单", "这笔", "此单",
               "刚才", "刚刚", "上次", "前面", "上面", "所说", "说的那", "同一个", "还是那个",
               "同上", "也是", "继续", "然后呢")

# 明确自带业务语义的短句，不需要上文也能路由（避免把「转人工」也去继承订单号）
_SELF_CONTAINED = ("转人工", "人工", "投诉", "退货", "退款", "保修", "发票", "物流", "订单")


@dataclass
class Resolution:
    """一次上下文补全的结果。"""

    question: str                       # 原始问题（默认不改写）
    inherited: Dict[str, str] = field(default_factory=dict)   # 从上下文继承到的实体
    reason: str = ""                    # 命中原因（写日志用）
    changed: bool = False               # 是否发生继承

    def as_log_fields(self) -> Dict[str, Any]:
        return {"inherited": self.inherited, "reason": self.reason,
                "changed": self.changed, "raw_len": len(self.question or "")}


def extract_entities(text: str) -> Dict[str, str]:
    """从一句话里抽订单号/工单号。抽不到返回空 dict（不猜）。"""
    text = text or ""
    out: Dict[str, str] = {}
    m = _ORDER_PREFIX_RE.search(text) or _ORDER_CTX_RE.search(text)
    if m:
        val = (m.group(1) if m.re.groups else m.group(0)).strip()
        if any(ch.isdigit() for ch in val):     # 单号必然含数字，纯字母不算
            out["order_id"] = val
    m = _TICKET_PREFIX_RE.search(text) or _TICKET_CTX_RE.search(text)
    if m:
        out["ticket_id"] = (m.group(1) if m.re.groups else m.group(0)).strip()
    return out


def needs_context(text: str) -> bool:
    """这句话是否依赖上文（含指代词，或短到不含任何业务关键词）。"""
    t = (text or "").strip()
    if not t:
        return False
    if any(w in t for w in REFERENTIAL):
        return True
    # 短句且不带业务词：大概率是「那能改地址吗」「多久到账」这类接续问句
    return len(t) <= 12 and not any(w in t for w in _SELF_CONTAINED)


def slots_from_dst(dst: Any) -> Dict[str, str]:
    """从跨轮会话状态里取已知槽位。

    DST 的实现可能是 dict 也可能是对象，字段名可能是 slots / entities，这里做容错，
    取不到就返回空 dict —— 补全失败不应影响主链路。
    """
    if dst is None:
        return {}
    raw: Optional[dict] = None
    if isinstance(dst, dict):
        raw = dst.get("slots") or dst.get("entities") or dst
    else:
        for name in ("slots", "entities"):
            val = getattr(dst, name, None)
            if isinstance(val, dict):
                raw = val
                break
    if not isinstance(raw, dict):
        return {}
    keys = ("order_id", "ticket_id")
    return {k: str(raw[k]).strip() for k in keys
            if raw.get(k) not in (None, "", 0)}


def slots_from_history(history_texts: Iterable[str], lookback: int = 4) -> Dict[str, str]:
    """从最近若干轮用户话里回捞实体（越近越优先）。"""
    texts: List[str] = [t for t in (history_texts or []) if t]
    found: Dict[str, str] = {}
    for t in reversed(texts[-lookback:]):
        for k, v in extract_entities(t).items():
            found.setdefault(k, v)      # 从新往旧找，先命中的更新，不被旧值覆盖
    return found


def resolve(question: str, *, dst: Any = None,
            history_texts: Optional[Iterable[str]] = None,
            augment_query: bool = False) -> Resolution:
    """补全当前问题依赖的上文实体。

    只在该句确实依赖上文、且自身不含新实体时，才从「会话槽位 → 最近历史」依次继承。
    自身带了新实体则一律以新值优先（用户换单是常态，不是错误）。
    """
    q = question or ""
    own = extract_entities(q)
    res = Resolution(question=q)

    if own:
        res.reason = "自带实体，不继承"
        return res

    if not needs_context(q):
        res.reason = "无指代信号"
        return res

    candidates: Dict[str, str] = {}
    candidates.update(slots_from_history(history_texts or []))
    candidates.update(slots_from_dst(dst))      # 会话槽位比历史更权威，覆盖之

    if not candidates:
        res.reason = "有指代但上下文无可用实体"
        return res

    res.inherited = candidates
    res.reason = "指代/省略，继承 " + "/".join(sorted(candidates))
    res.changed = True

    if augment_query:
        # 仅在显式开启时改写检索用的问句；默认关闭（下游有基于原句的正则解析）
        prefix = "，".join(
            f"{'订单' if k == 'order_id' else '工单'} {v}" for k, v in candidates.items())
        res.question = f"（{prefix}）{q}"
    return res


def merge_into_slots(dst_slots: Dict[str, str], inherited: Dict[str, str]) -> Dict[str, str]:
    """把继承到的实体合并进会话槽位（不覆盖已有值）。供 DST 落库前调用。"""
    merged = dict(dst_slots or {})
    for k, v in (inherited or {}).items():
        merged.setdefault(k, v)
    return merged


# 对外别名：调用方（service/turn/agent_turn.py）按「补全问题」的语义导入这个名字，
# 比 resolve 更明确；保持单一实现，避免两处逻辑漂移。
resolve_question = resolve
