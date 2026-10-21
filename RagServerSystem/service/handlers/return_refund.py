"""退货/退款主流程引导（DST 驱动、确定性执行）。

本质：把 return/refund 这类"有明确状态机"的意图，与泛化 ReAct 解耦。
栈位收集（Collect）→ 确认（Confirm）→ 执行（Execute）→ 完成（Done）
由 DST 阶段驱动，避免被 agent 泛化成"查物流/破损鉴定"，且全程不依赖 LLM 工具循环。
"""
import asyncio
from typing import AsyncIterator

from logs.log_config import log
from service.handlers.context import TurnContext
from shared.reply_templates import (
    RETURN_ALREADY_DONE,
    RETURN_ASK_MISSING,
    RETURN_CLARIFY,
    RETURN_CONFIRM,
    RETURN_DUPLICATED,
    RETURN_RECORD_FALLBACK,
    RETURN_SUBMITTED,
    SERVICE_CONTACT,
    return_flow_brief,
    stage_cn,
    stage_def,
    stage_step,
)


def _is_confirmation(question: str) -> bool:
    """判断用户是否在确认办理（用于 CONFIRMING → EXECUTING 阶段推进）。"""
    return any(k in question for k in
               ("确认", "是的", "对的", "可以", "确定", "同意", "提交", "办理", "就这么办"))


CANCEL_HINTS = ("不退货", "不退了", "不想退", "不退款", "算了", "取消", "不用了", "先不办")


async def handle_return_refund(ctx: TurnContext) -> AsyncIterator[str]:
    """产出退货/退款引导答复（收集槽位 → 确认 → 登记工单）。"""
    try:
        yield await _run_mainline(ctx)
    except Exception as e:
        log.error(f"退货/退款主流程异常: {e}", exc_info=True)
        # 同样不能空口承诺转接：异常分支没有建单，用户等不到人工接入。
        yield ("抱歉，办理退货/退款时遇到系统问题，暂时无法确认本次操作的结果。\n"
               "请稍后重试，或回复“退货进度”核实；如需人工客服协助，直接回复“转人工”。")


async def _run_mainline(ctx: TurnContext) -> str:
    """执行退货/退款主线，返回本轮答复文本。"""
    from agent.dst.slot_schema import Stage, required_slots

    dst = ctx.dst
    intent = dst.intent
    user_id = ctx.user["user_id"]
    subject = "退货" if intent == "return" else "退款"

    # 取消闸门必须先于一切分支：它不依赖意图分类，也不依赖 DONE 文案，
    # 用户在任何收集/确认阶段说"不办了"都能立刻退出主线。
    if dst.stage in (Stage.COLLECTING, Stage.CONFIRMING) \
            and any(k in ctx.question for k in CANCEL_HINTS):
        dst.intent, dst.stage, dst.slots = "chat", Stage.COLLECTING, {}
        await ctx.dst_manager.save(dst)
        log.info(f"退货/退款取消 | session={ctx.session_id} | 问题={ctx.question[:40]}")
        return f"好的，已取消本次{subject}申请，不会创建工单。您还有其他问题吗？"

    # 已完成流程防护：上一轮已登记工单（DONE），用户再提 return/refund 时
    # 不应拿旧槽位重复走"确认提交"（否则每轮都复读确认语，且会重复建单）。
    # 进度类追问已由前置关键词确定性分流；走到这里视为"发起新申请"，清空旧槽位重新收集。
    if dst.stage == Stage.DONE:
        dst.stage = Stage.COLLECTING
        dst.slots = {}
        await ctx.dst_manager.save(dst)
        return RETURN_ALREADY_DONE.format(subject=subject)

    # 0) 快速路径：用户在 CONFIRMING 阶段明确确认办理 → 直接登记，跳过槽位提取 LLM。
    #    确认轮本身不会再提供新槽位，再跑一次 LLM 提取属于浪费（每轮约 +4~7s）。
    if dst.stage == Stage.CONFIRMING and _is_confirmation(ctx.question):
        dst.stage = Stage.EXECUTING
        try:
            # 槽位验证层：统一校验订单归属、存在性与状态可退性（与 DST 状态机解耦）
            # 校验内部同步查库，放线程池执行
            from agent.dst.order_validator import validate_return_order
            _check = await asyncio.to_thread(
                validate_return_order, dst.slots.get("order_id"), owner_id=user_id)
            if not _check["valid"]:
                dst.stage = Stage.COLLECTING
                dst.slots.pop("order_id", None)  # 清掉无效单号，让用户重新提供
                await ctx.dst_manager.save(dst)
                return _check["message"]
            # 幂等防护：同一订单若已有在办的退货/退款工单，不再重复建单
            # （避免 DST 状态被新话题改写后，用户再次确认导致重复落单）
            # 归属限定到当前账号：种子订单号是 mock 数据、被多账号共用，
            # 不加限定会让 B 账号确认时命中 A 账号早先建的工单而误报「已登记」。
            from infrastructure.mysql_store import query_ticket_progress
            # 同一订单可能同时有退货单/退款单，且最近一张可能是投诉单。
            # 只查“最近一张”会把早先的退货单漏掉，因此按 return/refund 各查一次取较新者。
            _category = "return" if intent == "return" else "refund"
            _existing = None
            for _cat in ("return", "refund"):
                _candidate = await asyncio.to_thread(
                    query_ticket_progress, dst.slots.get("order_id") or "",
                    owner=user_id, category=_cat)
                if _candidate and (
                        _existing is None
                        or (_candidate.get("created_at") or "") > (_existing.get("created_at") or "")):
                    _existing = _candidate
            if _existing and _existing.get("status") != "closed":
                dst.stage = Stage.DONE
                await ctx.dst_manager.save(dst)
                return RETURN_DUPLICATED.format(
                    subject=subject, ticket_id=_existing["ticket_id"])
            from agent.tools.ticket_tool import create_ticket_record
            # 把槽位拼装成人类可读的中文描述，避免直接把 JSON dump 塞给客服
            _parts = []
            if dst.slots.get("order_id"):
                _parts.append(f"订单号：{dst.slots['order_id']}")
            if dst.slots.get("reason"):
                _parts.append(f"原因：{dst.slots['reason']}")
            if dst.slots.get("amount"):
                _parts.append(f"金额：{dst.slots['amount']}")
            if dst.slots.get("product"):
                _parts.append(f"商品：{dst.slots['product']}")
            _desc = f"{subject}申请（" + "；".join(_parts) + "）" if _parts else f"{subject}申请"
            # 建单即进入阶段化流转的第 1 环节（提交申请，等待商家审核）
            # 分类同时决定轨迹长度：纯退款不退货，轨迹里不出现「寄回商品 / 商家验收」
            _ticket = await asyncio.to_thread(
                create_ticket_record,
                user_id=user_id or ctx.session_id,
                order_id=dst.slots.get("order_id", ""),
                description=_desc,
                priority="normal",
                category=_category,
            )
            # 登记成功告知"全流程 + 当前环节 + 下一步 + 时效 + 联系方式"，
            # 用户一开始就能看懂接下来会发生什么、自己需要做什么。
            _stage = _ticket.get("stage", "")
            _sdef = stage_def(_stage)
            _step, _total = stage_step(_stage, _category)
            resp = RETURN_SUBMITTED.format(
                subject=subject,
                ticket_id=_ticket["ticket_id"],
                order_id=dst.slots.get("order_id", "") or "未关联订单",
                created_at=_ticket["created_at"],
                step=_step,
                total=_total,
                stage_cn=stage_cn(_stage),
                flow=return_flow_brief(_category),
                next=_sdef["next"],
                sla=_sdef["sla"],
                contact=SERVICE_CONTACT,
            )
        except Exception as e:
            log.error(f"退货/退款登记失败: {e}", exc_info=True)
            resp = RETURN_RECORD_FALLBACK.format(subject=subject)
        dst.answer = resp
        dst.tool_call_count += 1
        dst.stage = Stage.DONE
        await ctx.dst_manager.save(dst)
        log.info(f"退货/退款完成 | session={ctx.session_id} | slots={dst.slots} | stage={Stage.DONE}")
        return resp

    # 1) LLM 提取本轮槽位并合并（含格式校验与自愈清理）
    new_slots = await ctx.slot_extractor.extract(ctx.question, intent, dst.slots)
    dst = ctx.dst_manager.merge_new_slots(dst, new_slots)

    # 1.5) 归属前置校验：一拿到订单号就校验，别等用户确认后才说「这不是您的订单」
    #      （只拦归属，存在性/可退状态仍在确认时校验，保持原有分支不变）
    _oid = dst.slots.get("order_id")
    if _oid:
        from agent.dst.order_validator import validate_return_order
        _owner_check = await asyncio.to_thread(
            validate_return_order, _oid, owner_id=user_id)
        if _owner_check["status"] == "forbidden":
            dst.stage = Stage.COLLECTING
            dst.slots.pop("order_id", None)  # 清掉非本人单号，免得下一轮继续复用
            await ctx.dst_manager.save(dst)
            log.warning(f"退货/退款归属校验拦截 | session={ctx.session_id} | 订单={_oid}")
            return _owner_check["message"]

    # 2) COLLECTING：必要槽位缺失 → 礼貌补齐提问
    missing = [k for k in required_slots(intent) if not dst.slots.get(k)]
    if missing:
        dst.stage = Stage.COLLECTING
        await ctx.dst_manager.save(dst)
        names = {"order_id": "订单号", "reason": f"{subject}原因"}
        ask = "、".join(names.get(k, k) for k in missing)
        hint = "（请一并告知，便于尽快为您处理）" if "order_id" in missing else ""
        log.info(f"退货/退款引导 | session={ctx.session_id} | 待补={missing}")
        return RETURN_ASK_MISSING.format(subject=subject, ask=ask, hint=hint)

    # 3) 槽位已齐全：进入 CONFIRMING，向用户确认
    if dst.stage != Stage.CONFIRMING:
        dst.stage = Stage.CONFIRMING
        await ctx.dst_manager.save(dst)
        return RETURN_CONFIRM.format(
            order_id=dst.slots.get("order_id", ""),
            subject=subject,
            reason=dst.slots.get("reason", ""),
        )

    # 4) 已在 CONFIRMING 但用户未明确确认（补充/修正了信息）→ 回到 COLLECTING 继续澄清
    dst.stage = Stage.COLLECTING
    await ctx.dst_manager.save(dst)
    return RETURN_CLARIFY.format(subject=subject)