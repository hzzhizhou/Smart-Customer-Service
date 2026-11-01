"""
工单环节状态机：把「能怎么流转」从口头约定变成可校验的规则。
"""
from typing import Dict, Optional, Tuple

# 售后流程
STAGE_SEQUENCE: Tuple[str, ...] = (
    "submitted",   # 用户提交申请，待商家审核
    "approved",    # 审核通过，待用户寄回
    "shipped",     # 用户已寄回，在途
    "received",    # 商家已签收，验收中
    "refunding",   # 已发起退款，待到账
    "completed",   # 退款到账，办结
)

# 允许整段跳过的环节：线下寄回、来不及登记单号的场景，业务上允许直接从 approved 到 refunding
OPTIONAL_STAGES = frozenset({"shipped", "received"})

# 终态：进入后不可再流转（不论是自动还是人工）
TERMINAL_STAGES = frozenset({"completed", "rejected", "cancelled"})

# 异常终止环节：可从任意非终态进入（审核不通过、用户撤单）
EXCEPTION_STAGES = frozenset({"rejected", "cancelled"})

# 环节 → 生命周期状态。与原 DB 层 _STAGE_TO_STATUS 保持一致，并补齐异常环节：
# 原实现把 rejected 映射为 resolved（已解决），语义上「被拒绝」不是解决，应为 closed。
STAGE_TO_STATUS: Dict[str, str] = {
    "": "open",            # 无环节（投诉/咨询类工单）
    "submitted": "open",
    "approved": "processing",
    "shipped": "processing",
    "received": "processing",
    "refunding": "processing",
    "completed": "resolved",
    "rejected": "closed",
    "cancelled": "closed",
}

# 允许人工显式指定的生命周期状态；与环节不一致时由 normalize_status 纠正
VALID_STATUS = ("open", "processing", "resolved", "closed")


class IllegalTransition(Exception):
    """非法环节流转。消息面向运维，包含当前环节、目标环节与原因。"""

    def __init__(self, message: str, current: str, target: str):
        super().__init__(message)
        self.current = current
        self.target = target


def is_terminal(stage: str) -> bool:
    return (stage or "") in TERMINAL_STAGES


def status_of(stage: str) -> str:
    """环节对应的生命周期状态（未知环节按 processing 处理，与原实现一致）。"""
    if stage in STAGE_TO_STATUS:
        return STAGE_TO_STATUS[stage]
    return "processing"


def normalize_status(current_status: Optional[str], stage: str) -> str:
    """给出与环节一致的生命周期状态。

    人工显式指定与环节冲突时**以环节为准**并记录告警——因为用户看到的是环节进度，
    状态只是内部看板字段；两者不一致会让看板统计与用户感知对不上。
    """
    expected = status_of(stage)
    if current_status is None or current_status == expected:
        return expected
    if current_status in VALID_STATUS:
        return expected
    return expected


def validate_transition(current: str, target: str, *, force: bool = False) -> None:
    """校验 current → target 是否合法，不合法抛 IllegalTransition。

    合法情形：
    - 原地不动（幂等重放，如人工重复点同一个环节）
    - 沿流程前进，且被跳过的环节必须全部是可选的（shipped / received）
    - 从任意非终态进入异常终止环节（rejected / cancelled）
    - force=True 时放行回退（管理员纠错用，调用方需自行记录原因）

    非法情形：
    - 目标环节不在流程定义里（拼写错误、前端传了旧值）
    - 跳过必填环节（submitted → completed）
    - 从终态再流转（completed → refunding）
    - 回退（refunding → approved），除非 force
    """
    cur = current or ""
    tgt = target or ""

    if tgt == cur:
        return

    if tgt not in STAGE_SEQUENCE and tgt not in EXCEPTION_STAGES:
        raise IllegalTransition(
            f"未知环节 {tgt!r}，可选：{', '.join(STAGE_SEQUENCE + tuple(EXCEPTION_STAGES))}",
            cur, tgt)

    if cur and cur not in STAGE_SEQUENCE and cur not in EXCEPTION_STAGES:
        raise IllegalTransition(f"当前环节 {cur!r} 不在流程定义内，拒绝流转", cur, tgt)

    # 终态封口：已完成/已拒绝/已取消不能再动（否则会造成「退款到账后又回到待审核」）
    if is_terminal(cur):
        if force:
            return
        raise IllegalTransition(
            f"环节 {cur} 已是终态，不可再流转到 {tgt}（确需纠正请用 force 并记录原因）", cur, tgt)

    # 异常终止：任意非终态均可进入
    if tgt in EXCEPTION_STAGES:
        return

    cur_idx = STAGE_SEQUENCE.index(cur) if cur in STAGE_SEQUENCE else -1
    tgt_idx = STAGE_SEQUENCE.index(tgt)

    if tgt_idx < cur_idx:
        if force:
            return
        raise IllegalTransition(f"不允许回退：{cur} → {tgt}（确需纠正请用 force）", cur, tgt)

    # 前进：中间被跳过的环节必须都是可选的
    skipped = [s for s in STAGE_SEQUENCE[cur_idx + 1:tgt_idx] if s not in OPTIONAL_STAGES]
    if skipped:
        raise IllegalTransition(
            f"跳过必填环节 {skipped}：{cur or '（无）'} → {tgt}", cur, tgt)


def next_stages(current: str, *, force: bool = False) -> Tuple[str, ...]:
    """当前环节下所有合法目标（给后台下拉框用，避免人工手填出非法值）。"""
    cur = current or ""
    out = []
    for tgt in STAGE_SEQUENCE + tuple(EXCEPTION_STAGES):
        try:
            validate_transition(cur, tgt, force=force)
        except IllegalTransition:
            continue
        out.append(tgt)
    return tuple(out)
