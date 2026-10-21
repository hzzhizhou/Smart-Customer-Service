"""HTTP 请求/响应模型（Pydantic）。

鉴权走 Authorization: Bearer <token>（见 access/routes/auth.py 的 require_user），
请求体不再携带 api_key —— 避免把服务端密钥当作字段默认值写进 OpenAPI Schema。
"""
from typing import Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    question: str
    route_mode: str = "rule"
    session_id: Optional[str] = None
    use_context: bool = True
    use_hyde: bool = False
    use_multi: bool = False


class AgentRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
