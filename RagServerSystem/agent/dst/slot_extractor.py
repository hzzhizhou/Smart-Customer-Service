"""
LLM 槽位提取器（DST 的"槽位填充"环节）

输入：当前用户问题 + 已识别意图 + 历史已收集槽位
输出：本轮**增量**槽位 dict（只含用户本轮明确给出的值）

- 白名单过滤：只允许 SLOT_SCHEMA[intent] 中定义的槽位
- 失败兜底：LLM 调用失败 / 返回非法 JSON → 返回 {}，保留旧槽位，绝不抛出
- 规则兜底：订单号/金额用正则先行提取，降低对 LLM 的单点依赖
"""
import json
import re
from typing import Dict, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_factory import create_llm
from utils.circuit_breaker import llm_breaker
from logs.log_config import log
from agent.dst.slot_schema import SLOT_SCHEMA, slot_prompt_desc, ORDER_ID_RE, AMOUNT_RE

# 正则兜底：命中直接并入（快速、可测），作为 LLM 提取的补充通道
# 说明：amount 严格匹配 ¥/元（避免把订单号数字当金额）；reason 用常见诉求关键词兜底，
#       防止 LLM 漏提必填槽位导致阶段卡在 COLLECTING。
# reason 词表按平台定位「智能手机及配件」重写：原表的「尺码不合适/太大/太小/偏大/偏小」
# 是服装品类残留（手机没有尺码），裸「破」会误命中「突破/打破」等无关词，均已删除。
ENHANCEMENT_PATTERNS: Dict[str, re.Pattern] = {
    "order_id": ORDER_ID_RE,
    "amount": AMOUNT_RE,
    "reason": re.compile(
        # ① 商品本身的问题：与描述不符、发错、漏发、破损
        r"商品(?:出现|有|是)?(?:破损|损坏|质量问题|瑕疵|发错?.{0,2}货|漏发|少件|缺件|颜色不对|款式不符)"
        #    「描述不符」常常不带「商品」前缀（「收到的和描述不符」），单独放一条
        r"|(?:与|和|跟)?描述不符"
        # ② 手机常见故障：这批句子此前一条规则都不命中，reason 只能靠 LLM 提，
        #    LLM 一旦漏提就缺必填槽位、流程卡在 COLLECTING，故补齐三类高频硬件/系统故障
        r"|碎屏|屏幕(?:碎|裂|坏|有坏点)|进水|进液|浸水|无法开机|开不了机|死机|黑屏|花屏|闪屏"
        r"|闪退|自动关机|电池(?:鼓包|不耐用|掉电|健康度)|续航(?:差|不行|太差)"
        r"|充不上电|充不进电|充电(?:故障|异常)|摄像头(?:坏|故障|模糊|进灰)|信号(?:差|不好)"
        r"|发热|卡顿|按键失灵|听筒(?:坏|故障|没声)|扬声器(?:坏|故障|没声|杂音)"
        # ③ 主观意愿与交付类原因：无理由退货 / 拒收 / 送错
        #    中文数字「七天」与阿拉伯数字「7天」都要认（规则表里只写了 7 天那种写法）
        r"|不(?:想要|喜欢|满意|合适|愿换|愿修)|(?:七|7)天无理由|无理由退货|拒收|送错"
        # ④ 通用同义词兜底
        r"|破损|碎裂|损坏|质量问题",
        re.I,
    ),
}

SLOT_EXTRACT_SYSTEM = """你是一个电商客服的槽位提取器。从用户本次发言中提取业务槽位，只输出 JSON。

## 可提取的槽位（意图：{intent}）
{slot_desc}

## 要求
1. 只输出一个 JSON 对象，不要任何解释、前后缀或 markdown 代码块标记。
2. 只填用户本轮**明确给出**的槽位值；未提到的槽位不要输出。
3. 不要编造任何值；不要重复已有槽位（已有槽位只用于理解语境，不输出）。
4. 值统一为字符串，去掉多余空白。

## 格式要求（不符合格式的值会被系统丢弃，等同于没提）
- order_id：平台订单号，形如 SO20241120005（SO 开头）；不要把手机号、物流单号当订单号。
- amount：必须带「¥」前缀或「元」后缀，如 ¥199、199.5元；不要把订单号里的数字当金额。
- reason：用短语概括用户给出的原因，如「商品破损」「屏幕碎裂」「进水」；不要照抄整句原话。

示例输出：{{"order_id": "SO20241120005", "reason": "商品破损"}}"""


class SlotExtractor:
    def __init__(self, llm=None):
        self.llm = llm or create_llm(streaming=False, with_fallback=True)
        self._chain = self._init_chain()

    def _init_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SLOT_EXTRACT_SYSTEM),
            ("human", "用户问题：{question}\n已收集槽位：{existing}"),
        ])
        return prompt | self.llm | StrOutputParser()

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """解析 LLM 输出为 dict，容忍 ```json 包裹与前后空白"""
        text = raw.strip()
        if text.startswith("```"):
            # 去掉 markdown 代码块围栏
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # 尝试截取第一个 { ... } 片段
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                return {}
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
        return obj if isinstance(obj, dict) else {}

    @staticmethod
    def _rule_extract(question: str, intent: str) -> Dict[str, str]:
        """正则兜底：对 schema 内槽位做快速规则提取"""
        slots: Dict[str, str] = {}
        schema = SLOT_SCHEMA.get(intent, {})
        for name, pattern in ENHANCEMENT_PATTERNS.items():
            if name not in schema:
                continue
            m = pattern.search(question)
            if m:
                slots[name] = m.group(0).strip()
        return slots

    async def extract(self, question: str, intent: str,
                      existing: Dict[str, str]) -> Dict[str, str]:
        """返回本轮**增量**槽位。LLM 失败或返回非法 JSON → 返回 {}（保留旧槽位）。

        :param question: 当前用户问题
        :param intent:   当前业务意图（空则直接返回 {}）
        :param existing: 历史已收集槽位（仅用于提示 LLM 勿重复）
        """
        schema = SLOT_SCHEMA.get(intent)
        if not schema:
            # 无业务槽位的意图（consult/chat）或意图未知 → 无增量
            return {}

        merged: Dict[str, str] = {}
        # 熔断：LLM 连续失败时跳过槽位提取，仅走正则兜底（关键槽位有正则通道，
        # 缺槽位只是让流程多问一句，不会报错），避免雪崩
        if not llm_breaker.allow_request():
            log.warning("LLM 熔断中，槽位提取跳过 LLM，仅用正则兜底")
        else:
            try:
                raw = await self._chain.ainvoke({
                    "intent": intent,
                    "slot_desc": slot_prompt_desc(intent),
                    "question": question,
                    "existing": json.dumps(existing, ensure_ascii=False),
                })
                llm_breaker.record_success()
                new = self._parse_json(raw)
                # 白名单过滤 + 非空字符串化 + 格式校验
                for k, v in new.items():
                    if k not in schema or not isinstance(v, (str, int, float)):
                        continue
                    s = str(v).strip()
                    if not s:
                        continue
                    # 若该槽位在 schema 中定义了格式正则，则必须匹配才接受（防 LLM 幻觉，
                    # 例如把订单号数字识别成 amount）。无正则的（如自由文本 reason）直接接受。
                    pat = schema[k].pattern
                    if pat is not None and not getattr(pat, "search", None)(s):
                        log.info(f"槽位[{k}]值不匹配格式，丢弃: {s}")
                        continue
                    merged[k] = s
            except Exception as e:
                llm_breaker.record_failure()
                log.warning(f"LLM 槽位提取失败，保留原槽位: {e}")

        # 正则兜底补充：仅在 LLM 未提取到该槽位时填充。
        # 不能整体覆盖——正则只能截到关键词（如把“商品有破损”截成“破”），
        # 覆盖会破坏 LLM 给出的完整语义值，用户看到的退货原因就不成句了。
        for k, v in self._rule_extract(question, intent).items():
            merged.setdefault(k, v)
        if not merged:
            log.info(f"槽位提取无有效增量 | intent={intent} | 问题={question[:30]}")
        return merged
