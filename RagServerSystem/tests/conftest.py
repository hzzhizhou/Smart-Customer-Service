"""pytest 公共配置：把项目根目录加入 sys.path，保证可直接导入项目模块。"""
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 日志隔离：测试日志写到「系统临时目录」，不落项目内。
# 两个原因：
#   1) pytest 故意触发的异常路径会把 ERROR 写进真实日志，排障时全是假错误
#      （改造前 default.log 里 request_id="-" 的堆栈刷屏就是这个原因）；
#   2) 落项目内会让目录里出现两棵 Log_server（真实 + 测试），看起来像重复写日志。
# 临时目录位置：%TEMP%\rag_test_logs\Log_server\...（只在排错时需要时才去看）。
# 必须在 import 任何业务模块之前设置 —— log_config 在 import 期就会读环境变量。
os.environ.setdefault("RAG_LOG_DIR", str(Path(tempfile.gettempdir()) / "rag_test_logs"))
# 注意：不要再设 RAG_LOG_LEVEL。
# log_config.DEFAULT_LOG_LEVEL 在 import 期读取该变量，并作为 setup_module_logger 的
# 默认参数值——一旦压到 WARNING，落盘测试里的 logger.info 会被静默丢弃，
# 表现为「文件存在但为空」。目录隔离已足够避免污染真实日志，级别保持 INFO。


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