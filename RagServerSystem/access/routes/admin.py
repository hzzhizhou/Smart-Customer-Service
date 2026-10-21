"""
管理端 API：工单管理 + 知识库管理。
注册到现有 FastAPI app 上，复用已有存储层：
优势：不另造数据层，管理端看到的正是对话/入库侧写入的真实数据。

权限：全部接口挂 require_admin 依赖，仅管理员角色可访问（普通用户 403）。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from config.settings import ALLOWED_EXTENSIONS, BASE_DIR, MAX_FILE_SIZE
from access.routes.auth import require_admin
from infrastructure import mysql_store
from shared.reply_templates import RETURN_STAGES, RETURN_STAGE_EXCEPTIONS
from utils.security import DataSecurity


class TicketStatusReq(BaseModel):
    status: str | None = None   # 要更新的工单状态（可为空）
    note: str | None = None     # 处理备注（可为空）
    stage: str | None = None    # 要推进到的业务环节（退货/退款类工单，可为空）


class OrderStatusReq(BaseModel):
    status: str  # 订单要更新到的状态


# MySQL 状态管理器（懒加载单例，避免每个请求重建连接池）
_mysql_mgr = None
_mysql_mgr_lock = asyncio.Lock()


async def _get_mysql_mgr():
    global _mysql_mgr
    if _mysql_mgr is None:
        async with _mysql_mgr_lock:
            if _mysql_mgr is None:
                from infrastructure.sql.mysql_state_manager import MySQLStateManager
                mgr = MySQLStateManager()
                try:
                    await mgr.initialize()
                except Exception as e:
                    raise HTTPException(
                        status_code=503,
                        detail=f"MySQL 不可用，知识库管理不可用: {type(e).__name__}",
                    )
                _mysql_mgr = mgr
    return _mysql_mgr


# 合法的工单处理状态
_TICKET_STATUS = {"open", "processing", "resolved", "closed"}

# 合法的工单业务环节（退货/退款阶段化流转；顺序环节 + 异常环节）
_TICKET_STAGE = {s["key"] for s in RETURN_STAGES} | set(RETURN_STAGE_EXCEPTIONS)

# 合法订单销售状态（后台管理员可在订单页维护）
_ORDER_STATUS = {"pending", "paid", "shipped", "delivered", "refunding", "refunded", "cancelled"}


def register_admin_routes(app, retrieval_service):
    """把管理端接口挂到 FastAPI app，retrieval_service 用于读取向量库分块内容。"""

    # ---------- 工单管理 ----------
    @app.get("/tickets", dependencies=[Depends(require_admin)])
    async def list_tickets(status: str = "", page: int = 1, page_size: int = 10):
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        items = await asyncio.to_thread(mysql_store.list_tickets_paged, status or None, page, page_size)
        total = await asyncio.to_thread(mysql_store.count_tickets, status or None)
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    # ---------- 订单管理 ----------
    @app.get("/orders", dependencies=[Depends(require_admin)])
    async def list_orders(status: str = "", page: int = 1, page_size: int = 10):
        """后台订单列表（分页，可按状态筛选）。"""
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        items, total = await asyncio.to_thread(mysql_store.list_all_orders, status or None, page, page_size)
        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        }

    @app.patch("/orders/{order_id}", dependencies=[Depends(require_admin)])
    async def update_order(order_id: str, req: OrderStatusReq):
        """后台更新订单状态（发货/签收/退款中/取消等）。"""
        if req.status not in _ORDER_STATUS:
            raise HTTPException(status_code=400, detail=f"非法状态，可选：{', '.join(sorted(_ORDER_STATUS))}")
        row = await asyncio.to_thread(mysql_store.update_order_status, order_id, req.status)
        if row is None:
            raise HTTPException(status_code=404, detail=f"订单不存在: {order_id}")
        return row

    @app.get("/tickets/{ticket_id}", dependencies=[Depends(require_admin)])
    async def ticket_detail(ticket_id: str):
        """工单详情：单条工单全字段（含处理备注）。"""
        row = await asyncio.to_thread(mysql_store.get_ticket, ticket_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {ticket_id}")
        return row

    @app.patch("/tickets/{ticket_id}", dependencies=[Depends(require_admin)])
    async def update_ticket(ticket_id: str, req: TicketStatusReq):
        """更新工单：推进业务环节 / 填写处理备注 / 人工强制改状态。

        只推环节时，工单状态与订单退款状态自动联动（见 mysql_store.update_ticket_status）。
        """
        if req.status is not None and req.status not in _TICKET_STATUS:
            raise HTTPException(status_code=400, detail=f"非法状态，可选：{', '.join(sorted(_TICKET_STATUS))}")
        if req.stage is not None and req.stage not in _TICKET_STAGE:
            raise HTTPException(status_code=400, detail=f"非法环节，可选：{', '.join(sorted(_TICKET_STAGE))}")
        if req.status is None and req.note is None and req.stage is None:
            raise HTTPException(status_code=400, detail="至少提供 status、note 或 stage 之一")
        row = await asyncio.to_thread(
            mysql_store.update_ticket_status, ticket_id, req.status, req.note, req.stage)
        if row is None:
            raise HTTPException(status_code=404, detail=f"工单不存在: {ticket_id}")
        return row

    # ---------- 知识库管理 ----------
    @app.post("/knowledge/upload", dependencies=[Depends(require_admin)])
    async def upload_knowledge(file: UploadFile = File(...)):
        """上传文件到 data/ 后后台触发统一增量入库。"""
        data_dir = BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        # 仅取文件名，剔除路径分隔符，防路径穿越
        filename = Path(file.filename or "unnamed").name
        if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型，仅支持: {ALLOWED_EXTENSIONS}",
            )
        target = data_dir / filename
        # 逐块读并计量：中间件按 Content-Length 预检，但 chunked 传输不带该头，
        # 所以这里必须真的边读边卡，否则超限文件仍会把内存撑爆
        buf = bytearray()
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buf.extend(chunk)
            if len(buf) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"文件超过上限 {MAX_FILE_SIZE} 字节",
                )
        await asyncio.to_thread(target.write_bytes, bytes(buf))

        # 后台任务：上传返回后异步走统一入库（增量，写 MySQL 状态）
        def run_ingest():
            try:
                from ingestion.service import DataLayer
                asyncio.run(DataLayer.ingest(data_dir, mode="incremental"))
            except Exception as e:  # 后台异常记日志即可，不抛回请求
                import logging
                logging.getLogger("admin").error(f"上传后入库失败: {e}", exc_info=True)

        import threading
        threading.Thread(target=run_ingest, daemon=True).start()

        return {"filename": filename, "saved_to": str(target), "ingest": "started"}

    @app.get("/knowledge/documents", dependencies=[Depends(require_admin)])
    async def list_documents():
        mgr = await _get_mysql_mgr()
        return {"documents": await mgr.list_documents()}

    @app.get("/knowledge/documents/{doc_id}/chunks", dependencies=[Depends(require_admin)])
    async def doc_chunks(doc_id: str):
        """某文档的分块列表，并从向量库取回每块正文内容。"""
        mgr = await _get_mysql_mgr()
        chunk_list = await mgr.get_doc_chunks(doc_id)
        if not chunk_list:
            return {"doc_id": doc_id, "chunks": []}

        ids = [c["chunk_id"] for c in chunk_list]
        content_map = {}
        try:
            # 向量库按 id 取文本（同步 Chroma 调用放线程池，避免阻塞事件循环）
            chroma_vector = retrieval_service.vector_store  # ChromaVector
            def _fetch(_ids):
                return chroma_vector.vector_store.get(
                    ids=_ids, include=["documents", "metadatas"]
                )
            got = await asyncio.to_thread(_fetch, ids)
            g_ids = got.get("ids", []) or []
            g_docs = got.get("documents", []) or []
            for i, cid in enumerate(g_ids):
                content_map[cid] = g_docs[i] if i < len(g_docs) else ""
        except Exception as e:
            import logging
            logging.getLogger("admin").warning(f"读取分块正文失败（仅返回元数据）: {e}")

        for c in chunk_list:
            c["content"] = content_map.get(c["chunk_id"], "")
        return {"doc_id": doc_id, "chunks": chunk_list}

    @app.delete("/knowledge/documents/{doc_id}", dependencies=[Depends(require_admin)])
    async def delete_document(doc_id: str):
        """删除文档：清向量库分块 + 删 MySQL 状态(级联 doc_chunks) + 移除 data 源文件。"""
        import logging
        logger = logging.getLogger("admin")
        mgr = await _get_mysql_mgr()
        docs = await mgr.list_documents()
        doc = next((d for d in docs if d.get("id") == doc_id), None)
        if not doc:
            raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")

        file_path = (doc.get("file_path") or "").strip()
        norm_path = Path(file_path).as_posix() if file_path else ""

        # 路径白名单：只允许删除受管 data/ 目录内的源文件。file_path 虽由入库侧派生，
        # 但一旦库里被污染（手工改库 / 历史脏数据 / 有人用 data 之外的目录入库），
        # 下面的 unlink 就会删掉任意文件，所以动手前先卡住（fail-closed），
        # 不再往下走向量库与状态清理，避免"记录已删、文件还在"的半成品状态。
        if norm_path and not DataSecurity.validate_file_permission(
                norm_path, str(BASE_DIR / "data")):
            raise HTTPException(
                status_code=403,
                detail="源文件路径不在受管的 data/ 目录内，已拒绝删除",
            )

        # 1. 清理向量库分块（失败仅告警，不中断状态清理）
        try:
            if norm_path:
                chroma_vector = retrieval_service.vector_store
                await chroma_vector.delete_by_file(norm_path)
        except Exception as e:
            logger.warning(f"删除向量分块失败（继续清理状态）: {e}")

        # 2. 删 MySQL 记录（doc_chunks 通过外键级联删除）
        if norm_path:
            await mgr.remove_document(Path(norm_path))

        # 3. 移除 data/ 下的源文件
        file_removed = False
        try:
            if norm_path:
                src = Path(norm_path)
                if src.is_file():
                    src.unlink()
                    file_removed = True
        except Exception as e:
            logger.warning(f"删除源文件失败: {e}")

        return {
            "deleted": doc_id,
            "file_name": doc.get("file_name"),
            "file_removed": file_removed,
        }