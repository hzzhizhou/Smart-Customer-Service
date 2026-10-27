# 智能售后客服系统

面向「智能手机及配件」电商售后场景的 **RAG + 对话式 Agent** 客服系统，覆盖知识问答、订单与物流查询、退货退款办理、投诉转人工四条主链路，包含买家端对话界面与客服工作台。

- 后端：Python 3.12 / FastAPI / LangGraph / LangChain / Chroma / MySQL 8 / Redis（可选）
- 前端：Vue 3 / Element Plus / Vite
- 部署：Docker Compose（app / MySQL / Redis / Prometheus / Grafana）

| 指标 | 数值 |
| --- | --- |
| HTTP / WebSocket 端点 | 35（32 条路径） |
| Agent 工具 | 5 |
| 单元测试 | 363 个全部通过（本机约 23s） |
| 知识库语料 | 10 个文件（8 篇文档 + 2 份表格） |
| 评测集 | 意图 81 条 · RAGAS 30 题 · 业务端到端 53 题 |
| 并发压测（10 并发 × 40 请求） | P50 40.9s · P95 47.6s · P99 48.4s · 0 失败 |

## 目录

- [1. 核心能力](#1-核心能力)
- [2. 系统架构](#2-系统架构)
- [3. 技术栈](#3-技术栈)
- [4. 快速开始](#4-快速开始)
- [5. 配置说明](#5-配置说明)
- [6. 接口概览](#6-接口概览)
- [7. 目录结构](#7-目录结构)
- [8. 测试与评估](#8-测试与评估)
- [9. 运维](#9-运维)
- [10. 已知限制](#10-已知限制)
- [11. 文档索引](#11-文档索引)

## 1. 核心能力

- **LangGraph 编排**：StateGraph 接管主链路，`START → classify 意图节点 → 条件边 → 6 条业务分支 / Agent 兜底 → END`；图不挂 checkpointer，跨轮状态由 DST 落 MySQL。
- **统一 Agent 兜底**：单 Agent 编排 5 个工具；工具身份由服务端上下文注入，LLM 只负责选择工具与业务参数。
- **确定性主流程**：退货 / 退款走「收集槽位 → 确认 → 建单」状态机，不依赖 LLM 工具循环；
  建单带幂等键，重复确认不会重复建单。
- **混合检索**：BM25 + 向量双路召回，RRF 融合，Cross-Encoder 重排，绝对余弦相似度门控
  （high ≥ 0.80 / low < 0.55），低置信自动转人工。
- **意图识别**：规则优先 + LLM 兜底，区分「问政策」与「要办理」；规则层直判占比约 75%。
- **人工会话**：转人工自动建单；WebSocket 下行、HTTP 上行；断线按 `after_id` 增量补消息。
- **工程化**：单测、Prometheus 指标、请求级 trace、工具调用审计、Grafana 看板、一键编排。

## 2. 系统架构

```
┌──────────────────────────────────────────────────────┐
│ 交互层    Vue 3（买家端 / 客服工作台)
├──────────────────────────────────────────────────────┤
│ 接入层    FastAPI、鉴权、限流、请求上下文
├──────────────────────────────────────────────────────┤
│ 编排层    LangGraph 状态机、确定性业务分支、人工会话
├──────────────────────────────────────────────────────┤
│ 智能体    统一 Agent（5 工具）、意图分类、DST 状态机
├──────────────────────────────────────────────────────|
│ 流式生成 流式生成、输出合规护栏
├──────────────────────────────────────────────────────┤
│ 检索层    路由、改写、混合检索、RRF 融合、重排、门控
├──────────────────────────────────────────────────────┤
│ 数据层    加载、清洗、分块、向量入库
├──────────────────────────────────────────────────────┤
│ 基础设施  Chroma、MySQL、Redis
└──────────────────────────────────────────────────────┘
```

| 层 | 目录 | 职责 |
| --- | --- | --- |
| 交互层 | `frontend/` | 买家端对话与订单页，客服工作台与后台管理 |
| 接入层 | `access/` | 应用装配、路由、鉴权、限流、请求上下文 |
| 编排层 | `service/` | LangGraph 图编排、确定性业务分支、人工会话、响应缓存 |
| 智能体 | `agent/` | 统一 Agent、意图分类、DST 状态机、工具实现 |
| 生成层 | `generation/` | 流式生成、输出合规护栏 |
| 检索层 | `retrieval/` | 路由、改写、混合检索、RRF、重排、置信度门控 |
| 数据层 | `ingestion/` | 文档加载、清洗、分块、向量入库 |
| 基础设施 | `infrastructure/` | Chroma、MySQL、Redis |

## 3. 技术栈

| 类别 | 选型 |
| --- | --- |
| 后端框架 | FastAPI（单进程 uvicorn）+ LangGraph + LangChain |
| 模型接入 | DashScope OpenAI 兼容端点，模型名见 `config/settings.py` |
| 嵌入 | 默认 DashScope 云端（1024 维）；可切本地 `bge-base-zh-v1.5`（768 维） |
| 向量库 | Chroma（异步封装，本地持久化） |
| 检索 | BM25 + 向量混合召回，RRF 融合，Cross-Encoder 重排（`bge-reranker-base`） |
| 持久化 | MySQL 8：订单 / 工单 / 用户 / 会话历史 / DST 状态 / 文档入库状态 |
| 缓存 | Redis：会话历史热缓存，不可用时回源 MySQL（fail-open） |
| 可观测 | Prometheus 指标、Grafana 看板、请求级 trace、工具调用审计 |
| 前端 | Vue 3（script setup）、Element Plus、Vue Router、Pinia、Vite |
| 测试 | pytest（363 个用例） |

## 4. 快速开始

### 4.1 前置条件

| 组件 | 要求 |
| --- | --- |
| Python | 3.12 |
| Node.js | `^22.18.0` 或 `>=24.12.0` |
| MySQL | 8.x（**必需**，无降级） |
| Redis | 7.x（可选，仅作热缓存） |
| DashScope | API Key（LLM 与云端嵌入） |
| 本地模型 | 约 656 MB（重排默认启用；本地嵌入可选），不入库 |

### 4.2 安装

```powershell
cd "d:\文件路径"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r RagServerSystem\requirements.txt
```

也可以不激活环境，直接用解释器绝对路径运行脚本。

### 4.3 配置

```powershell
cd RagServerSystem
Copy-Item .env.example .env
# 填写 DASHSCOPE_API_KEY 与 MYSQL_*，其余项按需
```

配置项见 (#5-配置说明) 与 `RagServerSystem/.env.example`。

### 4.4 初始化数据库与知识库

```sql
CREATE DATABASE IF NOT EXISTS rag_db DEFAULT CHARACTER SET utf8mb4;
CREATE USER IF NOT EXISTS 'rag_user'@'%' IDENTIFIED BY '******';
GRANT ALL PRIVILEGES ON rag_db.* TO 'rag_user'@'%';
FLUSH PRIVILEGES;
```

业务表与文档状态表在后端首次启动时自动创建。

```powershell
cd RagServerSystem
python -m ingestion.service        # 全量重建：清空向量库与入库状态后重建
```

该命令是全量重建，首次初始化使用。日常新增或修改语料走管理端上传接口
（`POST /knowledge/upload`，增量入库，服务运行中即时生效）。

### 4.5 启动

```powershell
cd RagServerSystem
python main.py                    # API → http://127.0.0.1:8000
```

```powershell
cd frontend
npm install
npm run dev                       # 买家端 → http://127.0.0.1:5173
npm run dev:admin                 # 管理端 → http://127.0.0.1:5174（可选）
```

| 端口 | 用途 |
| --- | --- |
| 8000 | 后端 API |
| 5173 | 买家端 |
| 5174 | 管理端（第二个 dev server，可选） |
| 3307 | Docker 中 MySQL 映射到宿主机的端口 |
| 9090 | Prometheus |
| 3000 | Grafana |

### 4.6 Docker Compose

```bash
cd RagServerSystem
cp .env.example .env              # 填 DASHSCOPE_API_KEY
docker compose up -d --build
```

| 服务 | 说明 |
| --- | --- |
| `app` | 后端，构建自本仓库 `Dockerfile`，宿主机 8000 |
| `mysql` | MySQL 8.4，宿主机 3307，named volume 持久化 |
| `redis` | Redis 7，口令由 `Redis_password` 注入 |
| `prometheus` | 抓取 `app:8000/metrics`，宿主机 9090 |
| `grafana` | 自动加载 `deploy/observability` 下的数据源与看板，宿主机 3000 |

首次启动时若向量库为空，容器入口会自动入库（DashScope 嵌入，约 1~2 分钟）；
更新 `data/` 语料后全量重建索引：

```bash
docker compose run --rm app python -m ingestion.service
```

### 4.7 演示账号

首次启动自动创建 `admin`（管理员）与 `customer`（用户）。
口令优先读 `DEFAULT_ADMIN_PASSWORD` / `DEFAULT_BUYER_PASSWORD`；未配置则生成随机口令并打印到
后端日志（容器内为 `docker compose logs app`）。本地长期演示建议在 `.env` 显式固定。

### 4.8 常见问题

- **同时登录买家与管理员**：登录态存 `localStorage`，按 origin 隔离。分别用 5173 / 5174 两个端口
  即可各持一份登录态；`dev:admin` 带 `--strictPort`，端口占用会直接报错。
- **PowerShell 运行虚拟环境中的解释器**：路径以 `.` 开头时必须加调用运算符，
  如 `& .\.venv\Scripts\python.exe`，否则会被当作模块名。
- **全量重建向量库后**：需要重启服务，进程持有的集合句柄不会自动刷新（增量入库不需要重启）。
- **`/docs` 打不开**：默认关闭，本地演示时在 `.env` 设 `ENABLE_API_DOCS=true` 后重启。

## 5. 配置说明

配置集中在 `config/settings.py`，环境变量通过 `.env` 注入（以 `BASE_DIR` 显式绑定路径加载，
与进程工作目录无关）。

| 变量 | 必填 | 说明 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 是 | LLM 与云端嵌入 |
| `MYSQL_HOST` / `PORT` / `USER` / `PASSWORD` / `DATABASE` | 是 | 业务库连接 |
| `Redis_password` | 否 | 为空则不启用 Redis 鉴权 |
| `ENABLE_API_DOCS` | 否 | 默认 `false`，关闭 `/docs` 与 `/openapi.json` |
| `DEFAULT_ADMIN_PASSWORD` / `DEFAULT_BUYER_PASSWORD` | 否 | 为空则启动时生成随机口令并打印 |
| `TOKEN_TTL_HOURS` | 否 | 登录令牌有效期，默认 168 |
| `LLM_TIMEOUT` | 否 | 单次 LLM 超时，默认 30s |
| `MAX_REQUEST_BODY_SIZE` | 否 | 请求体上限，默认 12 MB |
| `AUTO_FILTER` | 否 | 检索前按问题推断类别做预过滤，默认开启 |
| `TICKET_IDEMPOTENCY_WINDOW` | 否 | 建单幂等窗口，默认 300s |

检索、分块、门控、熔断等参数见 `config/settings.py` 与 [设计说明](RagServerSystem/docs/design-notes.md)。

## 6. 接口概览

| 分组 | 鉴权 | 端点 | 说明 |
| --- | --- | --- | --- |
| 公开 | 无 | 2 | `/health`、`/metrics` |
| 认证 | 无 / 登录 | 4 | `/auth/*` |
| 会话 | 登录 | 14 | 流式对话、历史、我的订单、转人工、WebSocket |
| 管理 | 管理员 | 15 | 工单、订单、用户、知识库、审计与 trace |

端点按 HTTP 方法计数（同一路径的 GET/PATCH 分开计），共 35 个；按路径计为 32 条。

```json
// POST /agent/stream     Authorization: Bearer <token>
{ "question": "订单 SO20241120005 到哪了", "session_id": "u_001" }
```

完整清单、响应头与错误码见 [接口说明](RagServerSystem/docs/api.md)。

## 7. 目录结构

```
AI大模型与智能体开发/
├── RagServerSystem/                后端服务
│   ├── access/                     接入层：应用装配、路由、鉴权、限流
│   ├── service/                    编排层：LangGraph 图编排、业务分支、人工会话、缓存
│   ├── agent/                      智能体：统一 Agent、意图分类、DST、工具
│   ├── generation/                 生成层：流式生成、输出护栏
│   ├── retrieval/                  检索层：路由、改写、混合检索、重排、门控
│   ├── ingestion/                  数据层：加载、清洗、分块、向量入库
│   ├── infrastructure/             基础设施：Chroma、MySQL、Redis
│   ├── shared/                     跨层共享：常量、话术模板、DTO
│   ├── config/settings.py          配置项
│   ├── utils/                      LLM 工厂、熔断器、指标、请求上下文
│   ├── evaluation/                 RAGAS 离线评测
│   ├── deploy/                     Prometheus / Grafana / Redis 编排
│   ├── tests/                      单元测试
│   ├── data/                       知识库语料
│   ├── docs/                       接口与设计文档
│   ├── logs/                       日志配置与运行日志
│   └── main.py                     服务入口
├── frontend/                       Vue 3 前端
│   └── src/                        api / router / stores / views
├── models/                         本地模型（不入库）
└── .venv/                          Python 虚拟环境（不入库）
```

前端页面：买家端 `/`（智能对话）、`/my/orders`；管理端 `/admin/workbench`、
`/admin/tickets`、`/admin/orders`、`/admin/kb`、`/admin/users`、`/admin/monitor`；登录 `/login`。

## 8. 测试与评估

```powershell
cd RagServerSystem

python -m pytest tests -q          # 单测：363 passed
python intent_eval.py rule         # 意图规则层评测：81 条，秒级、零 LLM 费用
python intent_calibration.py       # 置信度校准：可靠性表 + ECE + 阈值扫描
python verify_dataset.py           # 离线检索命中验证（不带参数不会改数据）
python e2e_online_test.py          # 在线端到端 S0-S5，需服务在 8000 运行
python e2e_business_eval.py        # 业务评测集 53 题，需服务在 8000 运行
python -m evaluation.service       # RAGAS 双臂对比（开 / 关重排），30 题
python benchmark.py -c 10 -n 40    # 并发压测：/agent/stream 全链路，P50/P95/P99
```

### 8.1 RAGAS 双臂对比：重排的价值

同一 30 题分别关闭 / 开启重排各跑一遍 RAGAS，其余链路完全一致（实测结果）：

| 指标 | 关闭重排 | 开启重排 | 差值 |
| --- | --- | --- | --- |
| context_precision | 0.6867 | 0.8933 | **+0.207（+30%）** |
| context_recall | 0.6200 | 0.7667 | **+0.147（+24%）** |
| faithfulness | 0.6209 | 0.6277 | +0.007 |
| answer_relevancy | 0.7254 | 0.7229 | -0.003 |
| 平均单题检索耗时 | 0.21s | 4.38s | +4.17s |
| 无答案题子集 faithfulness | 0.3659 | 0.4338 | 两臂都在编 |

- 重排的价值钉在**检索层**：precision +30%、recall +24%；生成端指标基本饱和，重排不动它们。
- 无答案题两臂都低——该拒答时仍在编答案，**拒答门控是两臂共同的短板**，重排解决不了。
- 口径备注：判分模型与生成模型同为 deepseek-chat（同源判分，绝对分数仅参考、两臂差值有效）；
  判分送检索前 3 条，生成用全部 5 条候选，两臂口径一致。

### 8.2 并发压测

`python benchmark.py -c 10 -n 40`：10 并发 × 40 请求打 `/agent/stream` 全链路
（意图 → 检索 → 重排 → LLM 流式生成），每请求独立 session，知识类问句轮转，
本机实测（CPU 重排 + DeepSeek 生成）：

| 指标 | 数值 |
| --- | --- |
| 成功 / 失败 | 40 / 0 |
| P50 | 40.9s |
| P95 | 47.6s |
| P99 | 48.4s |
| MIN / MAX / MEAN | 26.4s / 48.4s / 40.2s |
| 吞吐 | 0.24 req/s |

单并发基线 P50 约 6.5s（其中重排约 4.2s）；10 并发下 P50 抬至 40.9s，吞吐
0.24 req/s ≈ 1 / 4.2s，墙钟 166s ≈ 40 题 × 4.2s 的重排串行总工时——瓶颈被钉在
CPU 重排单点（并发下事实串行），LLM 生成（API 并发）被排队完全掩盖。
全程无超时、无失败、无雪崩；生产优化路径明确：GPU 重排、独立重排服务、或批量重排。

评测口径、各项指标的实测结果与已知偏差见 [设计说明](RagServerSystem/docs/design-notes.md#9-评测口径)。

## 9. 运维

- **健康检查**：`GET /health` 探测 Redis / 向量库 / MySQL / LLM，全部可用返回 200，否则 503。
- **指标**：`GET /metrics` 暴露请求量、延迟、TTFT、检索阶段耗时、工具调用与兜底次数。
- **看板**：Grafana 数据源与看板由 `deploy/observability` 自动装载。
- **日志**：按层分文件写入 `logs/Log_server/<layer>/`，每条带 `request_id` 与 `trace_id`；
  支持按 `trace_id` 查询单次请求的 span 链路。
- **限流**：按客户端 IP 60 次/分，超限 429。
- **熔断**：LLM 与 Redis 各自独立熔断，失败 5 次断开、60s 后半开探测，熔断后自动降级。
- **生产部署**：后端不托管前端产物。用 Nginx 托管 `frontend/dist`，把 `/api` 去前缀反代到后端 8000，并为 WebSocket 开启 upgrade。

## 10. 已知限制

- 订单、工单、用户为自建 MySQL 表的 **mock 数据**，未对接真实 ERP / OMS。
- Chroma 为单机向量库，生产建议迁移 Milvus / Elasticsearch。
- 单进程 uvicorn，未配置多 worker；10 并发压测与瓶颈分析见 [§8.2](#82-并发压测)，内存限流与缓存不跨实例共享。
- 业务库（pymysql）与文档状态库（aiomysql）两套驱动并存，未统一连接池。
- 本地模型路径为绝对路径，换机器需修改 `config/settings.py`。
- CPU 环境下 Cross-Encoder 重排约 4s/题（20 候选真实分块；首次加载 20~25s），可用 `RERANK_MODE=none` 临时关闭。
- 门控阈值与意图词表依赖评测集持续回归。
- **单意图路由**：一张图只走一个分支，逗号分隔的多问题不会被拆分，第二个诉求可能被忽略；完整多意图需 plan 节点 + Send 子图 + 结果合并。
- **编排层无 checkpointer**：图为单轮短事务，跨轮状态由 DST / MySQL 承担；引入人工中断或断点恢复时再加。
- `combined_splitter` 未作默认：当前语料较短，父子分层会退化为重复存储。

完整取舍说明见 [设计说明](RagServerSystem/docs/design-notes.md#8-已知限制与取舍)。

## 11. 文档索引

| 文档 | 内容 |
| --- | --- |
| [docs/api.md](RagServerSystem/docs/api.md) | 全部端点、鉴权方式、响应头、错误码、WebSocket 约定 |
| [docs/design-notes.md](RagServerSystem/docs/design-notes.md) | 检索 / 意图 / DST / Agent / 护栏的设计取舍，后端分层与实测数据 |
| `RagServerSystem/.env.example` | 环境变量模板与逐项注释 |
| `RagServerSystem/deploy/` | Prometheus、Grafana、Redis 的编排配置 |
