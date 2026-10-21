"""
企业级数据安全工具：脱敏、权限校验、数据加密
"""
import re
from pathlib import Path
from typing import List
from logs.log_config import log

class DataSecurity:
    """数据安全处理：脱敏、权限校验"""

    @staticmethod
    def desensitize_text(text: str) -> str:
        """
        敏感信息脱敏：身份证、手机号、邮箱
        """
        if not text:
            return ""
        # 身份证脱敏（110101199001011234 → 110101********1234）
        text = re.sub(r'(\d{6})\d{8}(\d{4})', r'\1********\2', text)
        # 手机号脱敏（13800138000 → 138****8000）
        text = re.sub(r'(\d{3})\d{4}(\d{4})', r'\1****\2', text)
        # 邮箱脱敏（test@example.com → t***t@example.com）
        text = re.sub(r'(\w)\w+(\w)@(\w+\.\w+)', r'\1***\2@\3', text)
        return text

    @staticmethod
    def validate_file_permission(file_path: str, allowed_dir: str) -> bool:
        """
        文件权限校验（规避越权访问）
        :param file_path: 要访问的文件路径
        :param allowed_dir: 允许访问的根目录
        判定用 resolve() + is_relative_to()，而不是字符串 startswith：
        后者会被同前缀的兄弟目录骗过（.../data_evil/x 也以 .../data 开头），
        且 Windows 盘符/目录大小写不一致时会误判。resolve() 同时消解 ".."，
        防止 data/../../secret 这类遍历写法绕过前缀比对。
        """
        try:
            real_path = Path(file_path).resolve()
            real_allowed = Path(allowed_dir).resolve()
        except (OSError, ValueError):
            log.warning(f"越权访问文件（路径无法解析）：{file_path}")
            return False
        if not real_path.is_relative_to(real_allowed):
            log.warning(f"越权访问文件：{real_path}")
            return False
        return True