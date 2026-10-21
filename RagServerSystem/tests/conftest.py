"""pytest 公共配置：把项目根目录加入 sys.path，保证可直接导入项目模块。"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tool_identity():
    """绑定工具身份上下文（agent.tools.tool_context）。

    身份改由服务端注入后，用例要像真实请求那样先把登录账号绑进上下文，工具才认。
    这里统一 set + 用例结束复位：ContextVar 是「当前上下文」级别的，不复位会把身份
    泄漏给同一线程里后续的用例，让越权类断言出现假阳性/假阴性。
    """
    from agent.tools.tool_context import reset_current_user, set_current_user

    tokens = []

    def bind(user_id: str) -> str:
        tokens.append(set_current_user(user_id))
        return user_id

    yield bind
    for token in reversed(tokens):
        reset_current_user(token)