"""Agent 思考链可观测性测试：工具调用日志摘要函数 + 工具级 Prometheus 指标。

锁住三件事：
  1) _summarize_value：字符串/对象都能压成单行短摘要，超长截断不炸日志；
  2) 工具级指标：web_agent_tool_calls_total（按 tool/status 分维度）与
     web_agent_tool_duration_seconds 已注册且 inc/observe 生效；
  3) service.turn_service 导入链完好（观测埋点随模块引入不报错）。
"""
import pytest
from prometheus_client import REGISTRY

from service.turn_service import _summarize_value
from utils.metrics import agent_tool_calls_total, agent_tool_duration_seconds


class TestSummarizeValue:
    def test_string_collapses_whitespace(self):
        assert _summarize_value("订单  SO123\n 状态  已发货") == "订单 SO123 状态 已发货"

    def test_dict_becomes_compact_json(self):
        out = _summarize_value({"order_id": "SO123"})
        assert out == '{"order_id": "SO123"}'

    def test_truncates_with_ellipsis(self):
        out = _summarize_value("x" * 500, limit=50)
        assert len(out) == 51
        assert out.endswith("…")

    def test_within_limit_no_ellipsis(self):
        assert _summarize_value("短文本") == "短文本"

    def test_non_serializable_falls_back_to_str(self):
        class Obj:
            def __str__(self):
                return "对象兜底"
        # 不可序列化对象经 default=str 转字符串后仍走 JSON 路径（产出带引号的 JSON 字符串）
        assert _summarize_value(Obj()) == '"对象兜底"'

    def test_none_is_empty(self):
        assert _summarize_value(None) == "null"


class TestAgentToolMetrics:
    # Counter 命名以 _total 结尾时 prometheus_client 剥掉再补回：样本名即 'web_agent_tool_calls_total'
    def test_counter_registered_and_incremented(self):
        labels = {"tool": "unit_test_tool", "status": "success"}
        before = REGISTRY.get_sample_value("web_agent_tool_calls_total", labels) or 0
        agent_tool_calls_total.labels(**labels).inc()
        after = REGISTRY.get_sample_value("web_agent_tool_calls_total", labels)
        assert after == before + 1

    def test_error_status_counted_independently(self):
        err_labels = {"tool": "unit_test_tool", "status": "error"}
        before = REGISTRY.get_sample_value("web_agent_tool_calls_total", err_labels) or 0
        agent_tool_calls_total.labels(**err_labels).inc()
        assert REGISTRY.get_sample_value(
            "web_agent_tool_calls_total", err_labels) == before + 1

    def test_histogram_observes(self):
        labels = {"tool": "unit_test_tool"}
        before = REGISTRY.get_sample_value("web_agent_tool_duration_seconds_count", labels) or 0
        agent_tool_duration_seconds.labels(**labels).observe(0.42)
        after = REGISTRY.get_sample_value("web_agent_tool_duration_seconds_count", labels)
        assert after == before + 1


class TestTurnServiceImport:
    def test_module_imports_with_observability(self):
        # 埋点在 astream_events 事件循环内，模块级验证导入链与引用无拼写错误
        import service.turn_service as ts
        assert callable(ts._summarize_value)
        assert ts.agent_tool_calls_total is agent_tool_calls_total
        assert ts.agent_tool_duration_seconds is agent_tool_duration_seconds
