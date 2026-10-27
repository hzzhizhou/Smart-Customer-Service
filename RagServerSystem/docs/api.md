
# 接口说明

后端由 FastAPI 提供，共 35 个端点（含 1 个 WebSocket，按路径计 32 条）。路由**不带** `/api` 前缀：
前端由 Vite 代理把 `/api` 去前缀后转发到 `http://127.0.0.1:8000`，生产环境由网关做同样的转发。

## 鉴权

- 除 `/health`、`/metrics` 与登录注册外，所有接口要求 `Authorization: Bearer <token>`。
- 令牌来自 `POST /auth/login`，默认有效期 168 小时（`TOKEN_TTL_HOURS`）。
- 浏览器 WebSocket 构造器不支持自定义 header，因此 WS 的令牌放在 URL query（`?token=...`）。
- 对话类接口按登录账号隔离：订单、工单、会话一律带 `user_id` 过滤，越权访问返回 403 且不返回资源是否存在。

## 公开端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 组件健康检查：Redis / 向量库 / MySQL / LLM，全部可用返回 200，否则 503 |
| GET | `/metrics` | Prometheus 指标（文本格式） |

## 认证

| 方法 | 路径 | 鉴权 | 说明 |
| --- | --- | --- | --- |
| POST | `/auth/register` | 无 | 注册（用户名 3-20 字符，口令至少 6 位） |
| POST | `/auth/login` | 无 | 登录，返回 `token` + `user` |
| POST | `/auth/logout` | 登录 | 注销并失效当前令牌 |
| GET | `/auth/me` | 登录 | 当前登录用户信息 |

## 会话类

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/rag/stream` | RAG 流式问答（无历史意图、无 DST，带响应缓存） |
| POST | `/agent/stream` | Agent 流式对话，前端唯一使用的对话入口（含意图、DST、工具调度） |
| GET | `/history/sessions` | 当前用户的会话列表 |
| GET | `/history/messages` | 指定会话的历史消息（`?session_id=`） |
| DELETE | `/history/sessions/{session_id}` | 删除会话 |
| POST | `/history/sessions/{session_id}/close` | 结束会话（幂等） |
| GET | `/my/orders` | 当前用户名下订单（只读，最近下单在前） |
| POST | `/handoff/request` | 用户主动转人工；已有进行中会话则复用，否则建单 |
| GET | `/handoff/active` | 当前进行中的人工会话（无则返回 `ticket: null`） |
| GET | `/tickets/{ticket_id}/messages` | 人工会话消息，`after_id` 支持断线增量补消息 |
| GET | `/tickets/{ticket_id}/context` | 转人工前的机器人会话记录 |
| POST | `/tickets/{ticket_id}/messages` | 人工发送消息（上行） |
| POST | `/handoff/{ticket_id}/close` | 关闭人工会话 |
| WS | `/ws/handoff/{ticket_id}?token=…` | 人工会话下行通道 |

## 管理类

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/tickets` | 工单列表（分页，可按状态筛选） |
| GET | `/tickets/{ticket_id}` | 工单详情（含处理备注） |
| PATCH | `/tickets/{ticket_id}` | 更新工单：推进环节 / 填写备注 / 强制改状态 |
| GET | `/orders` | 订单列表（分页，可按状态筛选） |
| PATCH | `/orders/{order_id}` | 更新订单状态（发货 / 签收 / 退款中 / 取消等） |
| GET | `/users` | 用户列表（关键词 + 分页） |
| PATCH | `/users/{user_id}` | 启用 / 禁用用户（不能禁用当前登录账号） |
| DELETE | `/users/{user_id}` | 删除用户（不能删除自己） |
| POST | `/knowledge/upload` | 上传文件到 `data/` 并触发增量入库 |
| GET | `/knowledge/documents` | 知识库文档列表（入库状态） |
| GET | `/knowledge/documents/{doc_id}/chunks` | 文档分块列表，含从向量库取回的块正文 |
| DELETE | `/knowledge/documents/{doc_id}` | 删除文档（清向量分块 + 删状态 + 移除源文件） |
| GET | `/handoff/sessions` | 客服工作台会话列表（带最后一条消息，按最新倒序） |
| GET | `/observability/tool-calls` | 工具调用审计（按工具 / 用户 / trace / 状态过滤） |
| GET | `/observability/traces/{trace_id}` | 按 `trace_id` 查询内存中的 span 列表 |

## 对话接口约定

### 请求

```json
// POST /agent/stream
{
  "question": "订单 SO20241120005 到哪了",
  "session_id": "u_001"
}
```

`session_id` 由前端生成并复用；后端会拼成 `u{user_id}_{session_id}` 的作用域 ID。缺失或非法时
服务端新建，响应不回传，前端下次从 `/history/sessions` 续接。

### 响应

- 类型 `text/plain`，逐段流式返回，不包 JSON 信封。
- 生成在独立任务中执行：客户端中途断开，本轮历史仍会完整落库。
- 响应头：

| 头 | 含义 |
| --- | --- |
| `X-Retriever-Type` | 实际使用的召回方式 |
| `X-Gate-Level` | 置信度等级：`high / medium / low` |
| `X-Gate-Should-Escalate` | 是否建议转人工（`True / False`） |
| `X-Gate-Top-Score` | 最高余弦相似度 |

`/rag/stream` 在缓存命中时同样返回上述响应头，便于客户端区分「缓存命中」与「检索降级」。

### 状态码

| 码 | 场景 |
| --- | --- |
| 401 | 未登录、令牌失效或已注销 |
| 403 | 非管理员访问管理接口，或访问他人订单 / 工单 |
| 404 | 资源不存在，或无权访问（不泄露是否存在） |
| 413 | 请求体或上传文件超限 |
| 429 | 触发限流（默认 60 次/分/IP） |
| 503 | `/health` 检测到依赖不可用 |

## WebSocket

```js
const ws = new WebSocket(
  "ws://127.0.0.1:8000/ws/handoff/123?token=" + encodeURIComponent(token)
)
```

- 下行：客服发送的消息与状态变更。
- 上行：仍走 `POST /tickets/{ticket_id}/messages`。
- 重连：带 `after_id` 调用 `GET /tickets/{ticket_id}/messages` 补齐断线期间的消息。

## 调试

接口文档默认关闭（`ENABLE_API_DOCS=false`）。本地需要 Swagger / ReDoc 时在 `.env` 设
`ENABLE_API_DOCS=true`，重启后访问 `/docs`、`/redoc`、`/openapi.json`。生产环境应保持关闭。
