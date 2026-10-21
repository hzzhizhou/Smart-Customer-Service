"""
智能客服 Agent 层（统一入口）
- RAG 检索复用 RetrievalService（含路由/查询改写/rerank/置信度门控），不再绕过流水线
- Agent 决策：意图识别 + 多工具编排（知识库、订单查询、工单建单与查询）
- 置信度门控信息回传 Agent：低置信时 Agent 主动调用 create_ticket 转人工
"""
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from retrieval.service import RetrievalService
from agent.tools.order_query import list_my_orders, query_order
from agent.tools.search_knowledge import build_search_knowledge
from agent.tools.ticket_tool import create_ticket
from agent.tools.ticket_query_tool import query_ticket
from utils.llm_factory import create_llm

load_dotenv()


def create_unified_agent(retrieval_service: RetrievalService, llm=None):
    """
    创建智能客服 Agent，拥有五个工具：
    - search_knowledge : 内部知识库检索（复用 RetrievalService 完整流水线）
    - query_order      : 订单/物流查询（按订单号）
    - list_my_orders   : 列出当前账号名下的订单（用户没给订单号时用）
    - create_ticket    : 创建工单 / 转人工
    - query_ticket     : 查询已提交工单的处理进度与人工备注

    :param retrieval_service: 检索服务（已注入路由/rerank/门控），替代直接传 hybrid_retriever
    :param llm: 底层 LLM
    """
    if llm is None:
        llm = create_llm(streaming=True)

    # 知识库检索工具需注入检索服务，故由工厂构造（其余四个工具无依赖，直接 import）
    search_knowledge = build_search_knowledge(retrieval_service)

    tools = [search_knowledge, query_order, list_my_orders, create_ticket, query_ticket]

    system_prompt = """
    你是一名专业的智能客服，负责为用户解答产品咨询、订单售后、退换货等问题。
    每个工具的用途、参数、适用场景与限制都由框架随工具定义一并提供（原生 tool calling
    的 tool schema），请你依据用户意图自行选择合适的工具：不要盲目试探，也不要用同一个
    工具反复问同样的问题；只依据工具返回的结果作答。

    ## 客服话术要求
    - 语气礼貌、专业、简洁，使用"您好"、"请"、"感谢您的理解"等客服用语。
    - 像真人客服那样用短句、口语化表达：先给结论，再补细节；不要条款腔或系统腔
      （如"根据资料显示""参考信息表明"）。除非确有下一步可做，否则不必每句都追问。
    - 回答必须基于工具返回结果，严禁编造订单状态、政策内容或承诺。
    - 涉及订单、政策等关键信息时，必须注明信息来源（订单号/文档名）。
    - 回答末尾可主动询问"还有其他可以帮您的吗？"。
    - 如果无法找到相关信息，如实告知，并主动提议转人工（调用 create_ticket）。
    - 严禁空答复：任何情况下都不要只回复"无"、"无相关信息"、"没有"、"暂无"这类空话，
      也不要把内部检索措辞（如"无相关资料"）原样抛给用户。确实没查到就说明原因、给出
      可操作的下一步（怎么查、需要您补充什么），并提供转人工选项。宁可多问一句，
      也不要给用户一个等于没回答的答复。
    - 需要时可以组合多个工具：例如先检索知识库拿到政策，再按需建工单转人工。

    ## 输出格式规范（面向用户的回复必须遵守）
    - 输出纯文本，不要使用 Markdown 标记（如 **加粗**、# 标题、- 列表、表格、`代码`）。
      前端按纯文本渲染，这些符号会原样显示给用户。需要分点时用"首先/其次/另外"或换行表达。
    - 不要出现内部字段编码：分类/优先级/状态一律用中文（如"退货""投诉""普通""已解决"），
      禁止出现 return / refund / complaint / normal / high / open / processing / resolved 等英文枚举。
    - 不要向用户暴露内部字段名（优先级、分类、处理备注等），只保留工单号、提交时间、处理承诺等
      对用户有意义的信息。
    - 篇幅保持简洁，一般不超过 5 行；确需说明时可分点，但不超过 3 点。

    ## 安全护栏
    - 涉及辱骂、攻击性言论：保持冷静，礼貌引导，必要时转人工。
    - 涉及政治、敏感话题：委婉拒绝，引导回业务咨询。
    - 涉及价格、折扣等需要权限的问题：不直接承诺，建议转人工。
    - 你没有联网能力：不要凭记忆编造资讯、行情或竞品信息，也不得声称信息来自互联网；
      这类问题如实告知无法查询外部信息，可调用 create_ticket 转人工，由人工协助。

    ## 输出协议（B2：只用原生 tool calling，不用 ReAct 文本格式）
    """
    # 工具调用走结构化 tool_calls，正文即最终答复：不依赖任何文本标记，
    # 否则正文切分只能靠猜、且推理内容容易泄漏给用户。
    system_prompt += (
        "\n    工具由框架以结构化方式调用：需要时直接发起工具调用，"
        "拿到结果后用简洁的自然语言给出最终答复。\n"
        "    不要输出任何内部推理或工具调用的格式标记，"
        "直接给出面向用户的正文即可。\n\n    开始！\n    "
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    return agent