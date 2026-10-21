"""
DST 会话状态管理器（对话状态跟踪）

职责：
  - ConversationState：跨轮会话的结构化状态对象（意图/槽位/阶段/护栏计数/答复）
  - load / save：MySQL 持久化读写（重启不丢）
  - merge_new_slots / advance_stage：状态更新
  - build_system_context：生成注入主 Agent 的"会话记忆"提示段

存储策略：MySQL（infrastructure/mysql_store.conversation_state 表），
不依赖 Redis；模块内热缓存加速高频读。
"""
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from infrastructure.mysql_store import (
    get_conversation_state as _get_state,
    upsert_conversation_state as _upsert_state,
)
from agent.dst.slot_schema import Stage, required_slots, SLOT_SCHEMA
from logs.log_config import log


@dataclass
class ConversationState:
    """跨轮会话状态（对应设计图中的中期上下文）"""
    session_id: str
    intent: str = ""                     # 融合后的业务意图（return/refund/...）
    slots: Dict[str, str] = field(default_factory=dict)   # 跨轮累积槽位
    stage: str = Stage.COLLECTING        # 会话阶段
    tool_call_count: int = 0             # 护栏：累计工具调用次数
    escalation_reason: str = ""          # 转人工原因（complaint/置信度低）
    answer: str = ""                     # 上一轮最终答复
    updated_at: str = ""                 # 最后更新时间


class DSTManager:
    def __init__(self):
        # 热缓存：加速同会话高频读；重启后重建、回源 MySQL
        self._mem_cache: Dict[str, ConversationState] = {}

    async def load(self, session_id: str) -> ConversationState:
        """读取会话状态；无则返回默认 COLLECTING 空状态。

        读写走线程池：DST 在每轮问答的入口与收尾都会被调用，而底层 pymysql 是同步
        阻塞调用，直接在事件循环里查库会让并发请求排队。
        """
        cached = self._mem_cache.get(session_id)
        if cached is not None:
            return cached
        row = await asyncio.to_thread(_get_state, session_id)
        if row is None:
            st = ConversationState(session_id=session_id)
        else:
            st = ConversationState(
                session_id=row["session_id"],
                intent=row.get("intent", ""),
                slots=row.get("slots", {}) or {},
                stage=row.get("stage", Stage.COLLECTING),
                tool_call_count=row.get("tool_call_count", 0),
                escalation_reason=row.get("escalation_reason", ""),
                answer=row.get("answer", ""),
                updated_at=row.get("updated_at", ""),
            )
        self._mem_cache[session_id] = st
        return st

    async def save(self, state: ConversationState) -> None:
        """写入 MySQL + 更新热缓存"""
        state.updated_at = datetime.now().isoformat(timespec="seconds")
        await asyncio.to_thread(_upsert_state, {
            "session_id": state.session_id,
            "intent": state.intent,
            "slots": state.slots,
            "stage": state.stage,
            "tool_call_count": state.tool_call_count,
            "escalation_reason": state.escalation_reason,
            "answer": state.answer,
        })
        self._mem_cache[state.session_id] = state

    @staticmethod
    def merge_new_slots(state: ConversationState,
                        new_slots: Dict[str, str]) -> ConversationState:
        """合并本轮增量槽位（覆盖已有值，用户纠正可生效）"""
        if new_slots:
            state.slots.update(new_slots)
        # 自愈：清理不满足格式正则的历史槽位（防编辑时残留幻觉值，
        # 例如先前把订单号数字误存成 amount），意图未识别时跳过
        schema = SLOT_SCHEMA.get(state.intent, {})
        if schema:
            for k in list(state.slots):
                pat = schema[k].pattern if k in schema else None
                if pat is not None and not pat.search(str(state.slots[k])):
                    log.info(f"DST 清理格式不符槽位[{k}]={state.slots[k]}")
                    state.slots.pop(k, None)
        return state

    @staticmethod
    def advance_stage(state: ConversationState, tool_executed: bool) -> str:
        """按槽位完备度 + 工具执行推进阶段：
        缺关键槽位 → COLLECTING；已执行工具 → EXECUTING；
        槽位齐全且产出答复 → DONE；其余 → CONFIRMING
        """
        need = required_slots(state.intent)
        missing = [k for k in need if not state.slots.get(k)]
        if missing:
            state.stage = Stage.COLLECTING
        elif tool_executed:
            state.stage = Stage.EXECUTING
        elif state.answer:
            state.stage = Stage.DONE
        else:
            state.stage = Stage.CONFIRMING
        return state.stage

    @staticmethod
    def build_system_context(state: ConversationState) -> str:
        """生成注入主 Agent 的 DST 提示段（空状态/无槽位时返回空串，保持兼容）"""
        if not state.slots and not state.intent:
            return ""
        lines = ["", "## 会话记忆（跨轮已收集的业务信息，优先复用，勿重复询问）"]
        if state.intent:
            lines.append(f"- 当前业务意图：{state.intent}")
        if state.slots:
            lines.append(f"- 已确认信息：{json.dumps(state.slots, ensure_ascii=False)}")
        if state.stage == Stage.COLLECTING:
            missing = sorted(required_slots(state.intent) - set(state.slots))
            if missing:
                lines.append(f"- 尚缺信息：{', '.join(missing)}，请礼貌地向用户询问")
        elif state.stage in (Stage.EXECUTING, Stage.DONE) and state.answer:
            lines.append(f"- 上一轮结论：{state.answer[:200]}")
        return "\n".join(lines)
