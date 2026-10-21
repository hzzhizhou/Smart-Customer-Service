"""
定义了智能客服系统的指标，用于监控和分析系统的运行状态。
指标包括 HTTP 请求、Agent 调用、系统信息等。
"""
from prometheus_client import Counter, Histogram, Gauge, Info

# HTTP 请求指标
http_requests_total = Counter(
    'web_agent_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'web_agent_http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

# Agent 调用指标
agent_duration_seconds = Histogram(
    'web_agent_agent_duration_seconds',
    'Agent invocation duration',
    buckets=(0.5, 1, 2, 4, 8, 15)
)

# 系统信息
# 下面这段代码用于定义/暴露服务运行的静态信息（如版本、环境）
system_info = Info(
    'web_agent_system',                # 指标名称
    'System information'               # 指标描述
)
system_info.info({
    'version': '1.0',                  # 版本号
    'environment': 'production'        # 环境类型
})

# ====================== 智能售后客服工具指标 ======================
# 订单查询工具
order_queries_total = Counter(
    'customer_service_order_queries_total',
    'Total order query tool calls'
)
order_not_found_total = Counter(
    'customer_service_order_not_found_total',
    'Order queries that returned not found'
)

# 工单创建工具
tickets_created_total = Counter(
    'customer_service_tickets_created_total',
    'Total tickets created via agent'
)

# ====================== Agent ReAct 工具调用指标（通用层） ======================
# 与上面订单/工单业务计数器的分工：业务计数器统计「业务动作发生了多少次」，
# 这里统计「每次工具调用的成败与耗时」，回答「哪一步工具慢/失败」。
# 标签基数有界：tool 只有 5 个固定工具名、status 只有 success/error，无基数爆炸风险。
agent_tool_calls_total = Counter(
    'web_agent_tool_calls_total',
    'Agent ReAct tool calls by tool and status',
    ['tool', 'status']            # status: success / error
)

agent_tool_duration_seconds = Histogram(
    'web_agent_tool_duration_seconds',
    'Agent tool execution duration',
    ['tool'],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

# ===================== LLM 调用观测指标 =====================
# 回答「模型调用花了多少、失败几次、烧了多少 token」。label 基数有界：
# model 只有配置里的固定模型名，status 只有 success/error，type 只有 prompt/completion。
llm_requests_total = Counter(
    'web_agent_llm_requests_total',
    'LLM calls by model and status',
    ['model', 'status']              # status: success / error
)

llm_tokens_total = Counter(
    'web_agent_llm_tokens_total',
    'LLM tokens by model and type',
    ['model', 'type']                # type: prompt / completion
)

llm_request_duration_seconds = Histogram(
    'web_agent_llm_request_duration_seconds',
    'LLM call duration by model and status',
    ['model', 'status'],
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30)
)

# 首 token 延迟（模型侧）：衡量「模型多久开始出字」，与整轮 TTFT（web_agent_ttft_seconds）
# 区分开——后者含检索/路由耗时，这里只看模型本身。
llm_ttft_seconds = Histogram(
    'web_agent_llm_ttft_seconds',
    'LLM time to first token by model',
    ['model'],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 15)
)

# ===================== 流式体验与检索阶段指标 ======================
# 首 token 延迟（TTFT）：用户从提问到看见第一个字的等待时间。
# 流式场景只看整轮耗时没有意义：生成 10s 但 0.5s 就开始出字，体验远好于 3s 才吐第一个字，
# 两者整轮耗时可能相同。path 只有 agent / rag / rag_cached 三种，基数有界。
agent_ttft_seconds = Histogram(
    'web_agent_ttft_seconds',
    'Time to first token for streaming responses, by path',
    ['path'],                        # agent / rag / rag_cached
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 15)
)

# 检索阶段耗时：把「检索慢」拆到具体环节，回答「慢在哪一步」。
# stage 取值固定 5 个：rewrite（查询改写）/ decompose（查询分解）/
# search（路由+召回，含双路并行）/ rerank（融合后精排）/ total（整段检索）。
retrieval_stage_duration_seconds = Histogram(
    'web_agent_retrieval_stage_duration_seconds',
    'Retrieval stage duration',
    ['stage'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)

# ========= 降级 / 缓存 / 意图 / 转人工指标 ========
# 兜底率：回答没能正常走完主链路、改用确定性话术兜底的次数，按原因拆分。
# reason 取值固定（基数有界）：
#   breaker        —— LLM 熔断，直接跳过 ReAct 循环；
#   empty_answer   —— 模型把「无相关信息」这类占位答复当正文吐出，换兜底话术；
#   no_content     —— 模型只推理没给正文，转确定性知识兜底；
#   exception      —— 生成过程异常且一段正文都没产出；
#   low_confidence —— 检索置信度不足，知识兜底转人工。
# 级联场景（如 no_content 内部又触发 low_confidence）会各自 +1，属预期：
# 各 reason 独立计数，回答「哪类兜底最多」，不做去重。
agent_fallback_total = Counter(
    'web_agent_fallback_total',
    'Fallback responses by reason',
    ['reason']
)

# 缓存命中率：cache 区分两类缓存（response 响应缓存 / embedding 向量缓存），
# result 只有 hit / miss。命中率 = hit /(hit + miss)，用来判断缓存是否真的在生效。
cache_requests_total = Counter(
    'web_agent_cache_requests_total',
    'Cache lookups by cache type and result',
    ['cache', 'result']          # cache: response / embedding；result: hit / miss
)

# 意图分布：每次意图识别的结果计入对应 intent，回答「用户都在问什么」。
# intent 取值来自分类器的固定集合，基数有界。
intent_total = Counter(
    'web_agent_intent_total',
    'Detected business intent distribution',
    ['intent']
)

# 转人工率：按转人工来源拆分。
#   user_request —— 用户主动点「转人工」并新建会话；
#   complaint    —— 投诉意图确定性建单转人工；
#   agent_tool   —— Agent 调 create_ticket 工具转人工。
handoff_total = Counter(
    'web_agent_handoff_total',
    'Human handoff sessions opened by reason',
    ['reason']
)

# 轮次分支分布：每条确定性业务分支 + 泛化 ReAct 各计一次，回答「请求实际走了哪条链路」。
# 与 ReAct 工具调用指标互补：需要工具的意图（带单号订单/我名下订单/投诉/工单进度）
# 多数被确定性分支拦截，只看 tool_calls 会误判成「没有流量」。
# branch 取值固定：complaint / ticket_progress / order_query / my_orders /
# return_refund / consult / agent(泛化 ReAct 兜底)，基数有界。
branch_total = Counter(
    'web_agent_branch_total',
    'Turn handling branch distribution',
    ['branch']
)
