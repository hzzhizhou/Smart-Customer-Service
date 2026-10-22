
# 设计说明

本文记录各模块的设计取舍、关键参数与实测数据，供代码评审与排障使用。
面向使用者的安装与接口说明见仓库根 `README.md`，接口清单见 `docs/api.md`。

## 1. 检索流水线

实现：`retrieval/service.py`（编排）+ `retrieval/*`（各阶段）。

### 1.1 分块

- 默认策略 `recursive`：按 Markdown 标题结构递归切分，块大小 768 / 重叠 128，
  短于 50 字符的块不入库。
- `semantic`、`sliding_window`、`parent_child`、`combined_splitter` 保留为可选策略，通过
  `CHUNKING_STRATEGY` 切换；`combined_splitter`（语义父块 2000 + 递归子块 400）的父子回填
  仅在 `/rag/stream` 生效。
- 表格类内容（`.csv` 整表、`.docx/.xlsx` 内表格）统一转为带表头的 Markdown 表格块入库
  （`TABLE_CHUNK_SIZE=1000`，表头 + 约 5~6 行），避免表头与数据行被拆进不同块。

### 1.2 混合召回与融合

- BM25（`rank_bm25` + jieba 分词）与稠密向量双路召回。
- 融合默认 RRF（rank-based）：按名次累加 `1/(RRF_K + rank)`，`RRF_K=60`。与两路分数量纲无关，
  无需调权重。
- `weighted` 为备选策略：两路分数各自 Min-Max 归一化后按 `BM25_WEIGHT=0.4 / VECTOR_WEIGHT=0.6`
  加权。融合层与重排层不串联，二选一。

### 1.3 重排

- 融合后取前 `RERANK_CANDIDATES=20` 条送本地 `bge-reranker-base`（`RERANK_MODE=cross_encoder`）
  逐条打分，返回前 `RERANK_TOP_N=5` 条。
- 模型缺失或推理失败时降级为融合序并告警，不影响检索整体可用性。
- 纯 CPU 环境下重排是主要耗时项：单题检索由约 0.2s 升至约 4.4s（20 候选真实分块实测，
  耗时随候选文本长度增长；首次懒加载 20~25s）。
  `RERANK_MODE=none` 可关闭重排，便于灰度与排障。

### 1.4 置信度门控

- `vector_score = 1 - distance/2`，即余弦相似度（Chroma 默认返回平方 L2 距离且库内向量已单位化）。
- 阈值：`high ≥ 0.80`、`low < 0.55`；当 top1 与 top2 差距小于 `0.08` 时 high 降级为 medium。
- 低置信不强行作答，直接转人工；不使用批次内归一化分数（全差批次也会被拉成 1.0）。

### 1.5 查询分解与多子问门控

- `split_sub_questions()` 按 `？/?/；/换行` 与显式连接词（另外 / 还有 / 以及 / 顺便）做确定性切分；
  拆不出多段但形似多问时，才由 LLM 兜底拆一次。
- 逐子问独立检索（各占 `RERANK_TOP_N` 名额），结果交错合并去重，避免生成层截断上下文时
  丢掉后一个问题。
- 多子问门控：先剔除 `low` 子问，再在剩余可答子问间取最弱作为整轮门控；全部 `low` 才整轮转人工。
  实测「你们有哪些在售手机？iPhone 15 多少钱？」修复前取最弱得 0.54(low) 连能答的问题一并放弃，
  修复后 0.62(medium) 正常作答。
- 取舍：被剔除子问的资料一并丢弃，不向用户显式说明「这部分没查到」。

### 1.6 文档类别预过滤

- 检索前用 `infer_doc_categories()` 按关键词推断 `doc_category`（类别由文件名派生），
  命中多个类别时取并集而非常量优先级。
- 原因：「运费险能赔多少钱」同时命中 `product` 与 `faq`，只保留 `product` 会把真正答到问题的
  `平台FAQ(PLAT-013)` 整类排除，置信度被压到 0.53(low)。

### 1.7 空答复兜底

- 流式产出先缓冲，判定为占位答复（`is_empty_answer` / `could_be_empty_answer`）即整段替换为
  确定性话术「没有查到可靠的资料……」，内部串「无相关信息」不会透给用户。

## 2. 意图识别

实现：`agent/intent_classifier.py`（算法）+ `agent/intent_rules.py`（词表与正则）。

- 7 类业务子意图：`consult / return / refund / logistics / order / complaint / chat`。
- 规则优先 + LLM 兜底（`INTENT_MODE=hybrid`）。置信度 `conf = best_score / all_score`；
  `conf ≥ 0.5` 时规则直判，不触发 LLM（`INTENT_LLM_FALLBACK_CONFIDENCE=0.5`）。
- 「问政策」与「要办理」按「先过动作闸门、再查问法守卫」判定：政策词表 + 合并正则
  （条件类 / 评价类 / 疑问句式三个互斥分支）。命中即把 `return/refund` 降级为 `consult`，
  沿用原置信度，不再多花一次 LLM 兜底。
- 规则表内任何不自洽的权重都会造成永久误判（因为高置信时不兜底），修一处就要跑
  `intent_eval` 回归。已修复两例：「这个订单发货了吗」误判为 `order`、「上次退货的钱去哪了」
  误判为 `return`。
- 评测（81 条标注集：A 组 46 条与词表重合 / B 组 35 条同义改写）：
  `rule` 模式整体 76.5%（A 100% / B 45.7%），`hybrid` 模式两轮均 81/81；
  LLM 兜底占比 25%；单条平均 0.24~0.35s。校准分析见 `intent_calibration.py`（ECE 约 0.047）。
- 单测只断言「某个句式必须判对」，不替代评测集——用例是为规则量身写的，只反映回归，不反映泛化。

## 3. 对话状态与确定性主流程

实现：`agent/dst/`（状态机与槽位）+ `service/handlers/return_refund.py`（主流程）。

- 状态结构 `{intent, slots, stage, tool_call_count, answer}`，持久化在 MySQL
  `conversation_state` 表，重启不丢。
- 阶段：`COLLECTING → CONFIRMING → EXECUTING → DONE`。
- 退货 / 退款走确定性主流程：收集缺漏槽位 → 核对确认 → 建单，全程不依赖 LLM 工具循环，
  避免泛化成「查物流 / 破损鉴定」，也避免投诉场景在多轮工具调用中卡死。
- 槽位提取 = LLM 主 + 规则辅：
  1. `_rule_extract` 用正则补漏（`setdefault`，不覆盖 LLM 结果）；
  2. pattern 校验拦截幻觉（订单号里的数字当金额、手机号当订单号）；
  3. `merge_new_slots` 合并后自愈清理。
- 归属校验前置：拿到 `order_id` 立即校验归属，非本人订单返回 `ORDER_FORBIDDEN` 且不泄露任何
  订单字段。金额与订单号用正则强校验。
- 幂等：同账号 + 同订单 + 同分类在窗口内只落一张工单（`TICKET_IDEMPOTENCY_WINDOW`），
  库层由 `idempotency_key` 唯一键兜底。
- 工单轨迹按类别渲染：`return` 6 步 / `refund` 4 步，退款类工单不再显示「寄回商品」等无关待办。

## 4. Agent 与工具

- 单 Agent 编排 5 个工具：`search_knowledge`、`query_order`、`list_my_orders`、
  `query_ticket`、`create_ticket`。
- 工具身份（`user_id`）由服务端 `contextvars` 注入（`agent/tools/tool_context.py`），
  不作为 LLM 参数传递，提示注入无法伪造身份。
- `search_knowledge` 复用 `RetrievalService.retrieve()` 完整流水线，路由、重排、门控同样生效。
- 不接入联网搜索工具：模型凭参数记忆「假装联网」会编造资讯与来源，外部资讯类问题统一如实告知
  并转人工。
- 投诉意图不进入 ReAct 工具循环，直接建单转人工，避免多轮工具调用超时与情绪激化。
- 工具调用写入 `agent_tool_calls` 审计表；每轮生成轻量 span，可按 `trace_id` 查询。

## 5. 人工会话

- AI 主动升级转人工时必须建单并写入开场白，工单才会出现在客服工作台待处理列表。
- 话术统一为「正在为您转接人工客服（工单号 {ticket_id}），请稍候……」；客服实际接入前
  不声称「已接入人工客服」。
- 客服接手时可拉取转人工前的机器人会话记录（`/tickets/{id}/context`），无需用户重复描述。
- 下行通道 `WS /ws/handoff/{ticket_id}`（令牌放 query，浏览器 WS 构造器不支持自定义 header）；
  上行仍走 HTTP POST。断线重连用 `after_id` 增量补消息。

## 6. 生成与输出护栏

- 流式输出（`text/plain`），生产者任务与 HTTP 连接解耦：客户端中途断开，该轮历史仍完整落库。
- 合规护栏（`utils/output_guard.py`）：
  - 敏感词过滤；
  - 未授权承诺拦截（全额退款 / 免费换新 / 双倍赔偿 / 保证 X 天到货）；
  - 金额事实核查（`¥\d+` 必须有检索依据）。

## 7. 健壮性与性能

- 三态熔断器（`CLOSED / OPEN / HALF_OPEN`）分别用于 LLM 与 Redis：
  失败阈值 5 次、恢复 60s、半开探测 1 次，熔断后自动降级。
- 限流：按客户端 IP 60 次/分（`RATE_LIMIT_PER_MINUTE`）；请求体上限 12MB，上传文件 10MB。
- 超时：LLM 单次调用 30s（流式下为两个 chunk 之间的读超时）。
- 缓存：RAG 同问结果内存 LRU，`CACHE_MAXSIZE=1000` / `CACHE_TTL=300s`；
  缓存命中分支同样返回门控响应头。
- 线程池：BM25 为 CPU 密集（`cores`），向量检索为 IO 密集（`cores×2`）。
- 同步数据库调用经 `asyncio.to_thread` 卸载，避免阻塞事件循环。
- 意图分类 `force_intent` 复用 + 确认轮跳过槽位提取 LLM，确认轮由约 7s 降至约 0.2s。

## 8. 已知限制与取舍

**架构 / 部署**

- 订单、工单、用户为自建 MySQL 表的 mock 数据，未对接真实 ERP/OMS。
- Chroma 为单机持久化，生产建议迁移 Milvus / Elasticsearch。
- 单进程 uvicorn，未配置多 worker 与压测指标。
- 内存限流与缓存在多实例下不共享；分布式部署需换 Redis 计数与共享缓存。
- 业务库用 pymysql（同步短连接）、文档状态库用 aiomysql（连接池）两套驱动并存，未统一。
- 本地模型路径为绝对路径（`config/settings.py`），换机器需同步修改。

**准确性 / 性能**

- CPU 环境下 Cross-Encoder 重排是主要耗时项（真实分块约 4s/题，并发下事实串行构成吞吐上限），
  已用流式输出与同问缓存缓解。
- 门控阈值（0.80 / 0.55 / 0.08）基于当前语料选取，需更多业务样本持续调优。
- 查询分解只按强分隔符与显式连接词切分，逗号分隔的多问题不拆；子问过多时上下文仍会被
  `MAX_CONTEXT_TOKENS` 截断（已用交错合并保证每个子问先保头部）。
- 多子问门控会丢弃 `low` 子问的资料，用户拿不到「这部分没查到」的显式说明。
- 文档类别预过滤取并集会让候选池略微变大；索引继续增长后建议按类别分 collection。
- `combined_splitter` 未作默认：当前语料为 8 篇短文 + 2 份表格（约 40KB），父块平均 425、
  子块上限 400，18 个父块中 9 个只切出 1 个子块，分层已退化为重复存储。
- 「问政策 vs 要办理」的剩余边界：主线流程进行中（`COLLECTING/CONFIRMING`）会复用主线意图，
  评价类问题不会切走；词表外表述仍可能落进退货主流程。

## 9. 评测口径

| 工具 | 数据 | 说明 |
| --- | --- | --- |
| `pytest tests -q` | 339 例 | 单测，核心纯逻辑；历史持久化用例真连 MySQL，连不上自动 skip |
| `intent_eval.py rule` | 81 条 | 规则层准确率，秒级、零 LLM 费用 |
| `intent_eval.py` | 81 条 | 追加 hybrid 模式，输出兜底救回 / 改错明细 |
| `intent_calibration.py` | 81 条 | 可靠性表、ECE、兜底阈值扫描 |
| `verify_dataset.py` | 10 题检索用例 | 离线检索命中验证；`fresh` 会清空重建向量库 |
| `e2e_business_eval.py` | 53 题 | 在线业务评测，需服务在 8000 运行 |
| `evaluation/service.py` | 30 题 | RAGAS 双臂对比（开 / 关重排），输出四项指标与平均检索耗时 |

覆盖率：仓库尚未固定 `--cov` 的 source / omit 口径，不同口径结果差异较大。对外引用前应先写入
`[tool.coverage.run]`（`pyproject.toml`）并重新测量，避免与实测不符。

> 单测与评测集的分工：单测保证「回归没坏」，评测集反映泛化能力，两者的数字不能互相替代。

## 10. 后端分层与模块边界

### 10.1 目录与职责

| 目录 | 关键文件 | 职责 |
| --- | --- | --- |
| `access/` | `app.py`、`routes/*.py` | 应用装配、路由、鉴权依赖、限流中间件；只接客与参数校验，不写编排 |
| `service/` | `turn_service.py`（facade）、`turn/`、`handlers/`、`handoff_service.py`、`cache.py` | 一轮问答的编排、确定性分支、人工会话、RAG 响应缓存 |
| `agent/` | `service.py`、`intent_classifier.py`、`intent_rules.py`、`dst/`、`tools/` | 统一 Agent（5 工具）、意图分类、DST 状态机、工具上下文注入 |
| `retrieval/` | `service.py`、`router.py`、`query_enhancer.py`、`fusion.py`、`reranker.py`、`filters.py` | 路由、改写、混合召回、RRF / 加权融合、重排、置信度门控 |
| `generation/` | `service.py` | 拼 prompt、流式产出、敏感词与承诺校验 |
| `ingestion/` | `service.py`、`loader/`、`splitter/` | 加载、清洗、分块、增量 / 全量入库 |
| `infrastructure/` | `db/`、`mysql_store.py`（facade）、`mysql_history.py`、Chroma / Redis 封装 | 领域读写、历史持久化、向量库与缓存 |
| `shared/` | `constants.py`、`schemas.py`、`reply_templates.py` | 业务常量、DTO、话术模板 |
| `utils/` | `llm_factory.py`、`circuit_breaker.py`、`security.py`、`metrics.py` | LLM 工厂、熔断、脱敏、Prometheus 指标 |

### 10.2 轮次编排

拆分前的 `service/turn_service.py`（以仓库 HEAD 版本计约 730 行）按职责拆为四个模块，
原文件退化为纯 re-export 的 facade：

| 模块 | 内容 |
| --- | --- |
| `turn/streaming.py` | `_TTFTRecorder`、`_StreamQueue`、`_spawn_producer`、历史读写与收尾落库 |
| `turn/state.py` | `TurnState`、`_summarize_value`、`_redact_args`、`_result_digest` |
| `turn/agent_turn.py` | `AgentTurnService`：历史窗口、意图注入、分支调度、Agent 流式解析、工具审计、DST 更新 |
| `turn/rag_turn.py` | `RagTurnService`：无状态检索问答、响应缓存、`_gate_headers` |

调用方仍从 `service.turn_service` 导入，拆分对上层无感。测试若要 patch `DST_ENABLED`、
`init_chat_history` 等模块级对象，必须指向 `service.turn.agent_turn` / `service.turn.rag_turn`，
patch facade 不生效。

### 10.3 存储层

拆分前的 `infrastructure/mysql_store.py`（约 980 行）按领域拆为八个模块：

| 模块 | 内容 |
| --- | --- |
| `connection.py` | 连接、事务上下文 `_tx()`、行类型归一 |
| `schema.py` | DDL、`init_db`、增量迁移 |
| `orders.py` | 订单查询、归属校验 |
| `tickets.py` | 工单、工单消息、人工会话、幂等建单 |
| `users.py` | 用户、令牌 |
| `conversations.py` | 会话与消息 |
| `dst.py` | `conversation_state` |
| `tool_audit.py` | `agent_tool_calls` 审计 |

约定：域模块统一 `from . import connection` 并以 `connection._tx()` 调用，事务实现只有一份；
跨域调用走模块属性，不复制实现。`mysql_store.py` 仅 re-export，现有 30 余处
`from infrastructure import mysql_store` 无需改动；测试 patch `_tx` 时要指向
`infrastructure.db.connection._tx`。

### 10.4 依赖方向

```
access → service → agent / retrieval / generation → infrastructure
```

反向依赖只允许通过 `shared` 的常量与 DTO；facade 只承担兼容边界，不新增业务逻辑。
新增表或字段时只动对应领域模块与 `schema.py`，不再回到单个 god module。

