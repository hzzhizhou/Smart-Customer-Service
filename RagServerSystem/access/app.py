"""接入层：FastAPI 应用装配与启动入口。

分层约定（企业惯例：路由装配与业务实现分离，一个文件只承担一个职责）
- 请求/响应模型        → shared/schemas.py
- 限流与指标中间件      → access/middleware.py
- 确定性业务规则        → shared/constants.py
- 响应缓存             → service/cache.py
- 六条确定性业务分支    → service/handlers/（投诉、工单进度、订单查询、我名下订单、
                          退货/退款、知识咨询）
- 轮次编排（服务层）    → service/turn_service.py（读历史 → 识别意图 → 选分支 →
                          产出 → 收尾落库/推进 DST；两条流式入口各一个服务）

本文件只保留三件事：应用装配、路由定义（接客 + 参数校验 + 鉴权）、运维端点探测。
路由函数不写业务逻辑，一律交给 service/turn_service.py 或对应子路由模块。
"""
import asyncio
import os

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from config.settings import API_HOST, API_PORT, RESPONSE_TIMEOUT, ENABLE_API_DOCS
from logs.log_config import log
from agent.service import create_unified_agent
from utils.llm_factory import create_llm
from access.middleware import RateLimiter, register_middleware, register_request_context_middleware
from shared.schemas import AgentRequest, RAGRequest
from access.routes.auth import require_admin, require_user
from service.cache import ResponseCache
from agent.dst.dst_manager import DSTManager
from agent.dst.slot_extractor import SlotExtractor
from access.history import (
    close_history_session,
    delete_history_session,
    get_history_messages,
    list_history_sessions,
    resolve_session_id,
)
from agent.intent_classifier import IntentClassifier
from service.turn_service import AgentTurnService, RagTurnService


def register_observability_routes(app) -> None:
    """可观测性查询端点（C1 工具调用审计 / C2 调用链 span）；仅管理员可访问。

    独立成模块级函数（与 admin/auth/handoff 子路由同风格）便于单独装配与测试。
    """

    @app.get("/observability/tool-calls", dependencies=[Depends(require_admin)])
    async def tool_call_audit(tool_name: str = "", user_id: str = "",
                              trace_id: str = "", status: str = "",
                              limit: int = 100):
        """工具调用审计查询（C1）：按工具/用户/链路/状态过滤，最新在前。

        数据来自 agent_tool_calls 表（Agent 工具路径写入，见 service/turn/agent_turn.py 的 _record_tool_audit）。
        """
        from infrastructure import mysql_store
        items = await asyncio.to_thread(
            mysql_store.list_tool_calls, tool_name, user_id, trace_id, status, limit)
        return {"items": items, "total": len(items)}

    @app.get("/observability/traces/{trace_id}", dependencies=[Depends(require_admin)])
    async def trace_spans(trace_id: str):
        """按 trace_id 查调用链 span（C2）：内存环形缓冲中的 span 列表。

        数据非持久化（重启即清空），用于定位「刚发生的慢/失败链路」。
        """
        from utils.request_context import get_trace_spans
        spans = get_trace_spans(trace_id)
        return {"trace_id": trace_id, "count": len(spans), "spans": spans}


class RAGService:
    def __init__(self, retrieval_service, answer_generator, llm=None):
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.llm = llm
        self.hybrid_retriever = retrieval_service.hybrid_retriever   # 兼容旧引用
        # OpenAPI/交互文档默认关闭（ENABLE_API_DOCS=false），避免接口面与内部模型暴露到公网；
        # 本地演示在 .env 设 ENABLE_API_DOCS=true 即可恢复 /docs。
        self.app = FastAPI(
            title="企业级RAG服务",
            version="1.0",
            docs_url="/docs" if ENABLE_API_DOCS else None,
            redoc_url="/redoc" if ENABLE_API_DOCS else None,
            openapi_url="/openapi.json" if ENABLE_API_DOCS else None,
        )
        # 无会话场景的响应缓存：同一问题命中即秒回（实现见 service/cache.py）
        self.cache = ResponseCache()
        # 速率限制：按客户端 IP 维护最近一分钟请求时间戳队列（实现见 access/middleware.py）
        self.rate_limiter = RateLimiter()

        # ========== 服务层装配：两条流式入口的轮次编排（实现见 service/turn_service.py）==========
        self.rag_turn_service = RagTurnService(retrieval_service, answer_generator, self.cache)
        self.turn_service = AgentTurnService(
            retrieval_service=retrieval_service,
            answer_generator=answer_generator,
            # 统一 Agent：复用 RetrievalService（含路由/rerank/置信度门控）
            unified_agent=create_unified_agent(retrieval_service, llm),
            # 细粒度业务意图分类器：识别退货/退款/物流/投诉等子意图，引导 Agent 优先选工具
            # （意图 LLM 兜底为纯文本管道，用带供应商降级的独立实例，与 Agent 的 bind_tools 实例解耦）
            intent_classifier=IntentClassifier(llm=create_llm(streaming=False, with_fallback=True)),
            # DST：跨轮会话状态管理（意图/槽位/阶段持久化）
            dst_manager=DSTManager(),
            # 槽位提取器：LLM 从用户发言中抽取本轮增量槽位（同样走供应商降级实例）
            slot_extractor=SlotExtractor(llm=create_llm(streaming=False, with_fallback=True)),
        )

        self._register_routes()
        # 账号 / 后台管理 / 转人工 / 我的订单：各自独立的子路由模块，装配时挂载
        from access.routes.admin import register_admin_routes
        from access.routes.auth import register_auth_routes
        from access.routes.handoff import register_handoff_routes
        from access.routes.user_orders import register_user_order_routes
        register_admin_routes(self.app, self.retrieval_service)
        register_auth_routes(self.app)
        register_handoff_routes(self.app)
        register_user_order_routes(self.app)
        register_observability_routes(self.app)
        self._register_middleware()

    def _register_routes(self):
        # ========== 会话历史（归属校验在 SQL 的 WHERE user_id 条件里）==========
        @self.app.get("/history/sessions")
        async def history_sessions(user: dict = Depends(require_user)):
            """列出当前登录用户自己的历史会话（查询按归属账号过滤，看不到他人会话）。"""
            return {"sessions": await list_history_sessions(user["user_id"])}

        @self.app.get("/history/messages")
        async def history_messages(session_id: str = "",
                                   user: dict = Depends(require_user)):
            return {
                "session_id": session_id,
                "messages": await get_history_messages(user["user_id"], session_id),
            }

        @self.app.delete("/history/sessions/{session_id}")
        async def remove_history_session(session_id: str,
                                         user: dict = Depends(require_user)):
            """删除当前用户自己的某个会话（归属校验在库的 WHERE user_id 条件里）。"""
            if not await delete_history_session(user["user_id"], session_id):
                raise HTTPException(status_code=404, detail="会话不存在或已删除")
            return {"status": "success", "session_id": session_id}

        @self.app.post("/history/sessions/{session_id}/close")
        async def close_session(session_id: str, user: dict = Depends(require_user)):
            """结束本次咨询：该会话标记为已结束，用户再发消息即开启新会话，历史仍可回看。

            幂等：重复结束同一会话仍返回成功，避免用户重复点击时报错。
            """
            if not await close_history_session(user["user_id"], session_id):
                raise HTTPException(status_code=404, detail="会话不存在")
            return {"status": "success", "session_id": session_id}

        # ========== 流式 RAG 路由（企业统一流式输出）==========
        @self.app.post("/rag/stream")
        async def rag_stream(request: RAGRequest,
                             user: dict = Depends(require_user)):
            """RAG 流式问答：检索与产出交给 RagTurnService，本层只做鉴权与响应包装。

            生成与 HTTP 连接解耦（与 /agent/stream 同一套骨架）：答复在服务层的
            独立任务里跑完并直接落库，这里的生成器只从队列转发，
            客户端断开不会丢掉该轮历史。
            """
            # 鉴权由 require_user（Bearer 令牌）完成；会话 ID 按登录用户作用域化，
            # 历史/DST 都落在自己的命名空间下
            session_id = resolve_session_id(user["user_id"], request.session_id or "")
            chunks, headers = await self.rag_turn_service.stream(request, session_id)
            return StreamingResponse(chunks, media_type="text/plain", headers=headers)

        # ========== Agent 路由（企业统一流式输出）==========
        @self.app.post("/agent/stream")
        async def unified_ask_stream(request: AgentRequest,
                                     user: dict = Depends(require_user)):
            """Agent 流式输出：编排交给 AgentTurnService，本层只做鉴权与响应包装。

            生成与 HTTP 连接解耦：答复在服务层的独立任务里跑完并直接落库，
            这里的生成器只从队列转发，客户端断开不会写坏历史。
            """
            # 鉴权由 require_user（Bearer 令牌）完成；会话 ID 按登录用户作用域化，
            # 历史/DST/工单归属都落在自己的命名空间下
            session_id = resolve_session_id(user["user_id"], request.session_id or "")
            queue = await self.turn_service.start_turn(request.question, session_id, user)

            async def event_source():
                """（HTTP 下游）只做转发：从队列取块推给客户端，收到哨兵即结束。"""
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item

            return StreamingResponse(event_source(), media_type="text/plain")

        # ========== 运维端点 ==========
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus 指标暴露端点"""
            from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
            return PlainTextResponse(
                generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )

        @self.app.get("/health")
        async def health():
            """健康检查：VectorDB / LLM / MySQL（关键路径）+ Redis（非关键缓存层）。
            关键路径不可用时返回 503；Redis 仅是缓存（降级不影响服务可用性），
            单独报告但不拉垮整体健康状态。"""
            checks = {"status": "healthy", "components": {}}
            # 1. Redis（非关键缓存层：不可用时服务仍可用，仅记录不降级）
            #    同步 ping 放到线程池执行，避免 /health 被频繁轮询时阻塞事件循环
            try:
                from infrastructure.redis.connection import get_redis_connection
                client = get_redis_connection()
                await asyncio.to_thread(client.ping)
                checks["components"]["redis"] = "ok"
            except Exception as e:
                checks["components"]["redis"] = f"unavailable(cache, service ok): {type(e).__name__}"
            # 2. 向量库（关键路径）：确认 Chroma 可访问并读取 collection 名
            #    同步 DB 调用放到线程池执行，避免 /health 被频繁轮询时阻塞单事件循环
            try:
                inner = self.retrieval_service.vector_store
                chroma = inner.vector_store                 # langchain_chroma.Chroma 实例
                await asyncio.to_thread(lambda: chroma.get(limit=1))  # 触发惰性建库，验证可达
                name = getattr(chroma, "_collection_name", None) \
                       or getattr(getattr(chroma, "_collection", None), "name", "Chroma_db")
                checks["components"]["vector_store"] = f"ok ({name})"
            except Exception as e:
                checks["components"]["vector_store"] = f"fail: {type(e).__name__}"
                checks["status"] = "degraded"
            # 3. MySQL（关键路径：订单/工单/用户/会话历史/DST 状态都落这张库，
            #    连不上时这些接口全部不可用，故计入健康状态而非仅提示）
            try:
                from infrastructure import mysql_store
                await asyncio.to_thread(mysql_store.count_users)
                checks["components"]["mysql"] = "ok"
            except Exception as e:
                checks["components"]["mysql"] = f"fail: {type(e).__name__}"
                checks["status"] = "degraded"
            # 4. LLM 配置（关键路径：只检查 API key 是否就绪，不发实际请求）
            checks["components"]["llm"] = "ok" if os.getenv("DASHSCOPE_API_KEY") else "fail: no api key"
            if checks["components"]["llm"] != "ok":
                checks["status"] = "degraded"
            code = 200 if checks["status"] == "healthy" else 503
            return JSONResponse(checks, status_code=code)

    def _register_middleware(self):
        """注册中间件（实现见 access/middleware.py）。

        Starlette 后注册的在更外层、先执行：request_id 中间件放最后（最外层），
        保证包括限流 429 在内的所有响应都带 X-Request-ID、所有日志都注入 request_id。
        """
        register_middleware(self.app, self.rate_limiter)          # 限流 + 请求计速（内层）
        register_request_context_middleware(self.app)             # request_id 贯穿（外层）

    def run(self):
        log.info(f"启动RAG API服务：http://{API_HOST}:{API_PORT}")
        uvicorn.run(
            self.app,
            host=API_HOST,
            port=API_PORT,
            timeout_keep_alive=RESPONSE_TIMEOUT
        )