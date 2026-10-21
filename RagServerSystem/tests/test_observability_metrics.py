"""C 批可观测性回归：HTTP 指标口径修正（C3）+ LLM 调用观测（C4）。

对应 PROJECT_AUDIT §7.2（1）（2）（4）（6）。锁住四件事：
  1) HTTP 指标 endpoint 标签用**路由模板**，带参路径不再一个 ID 一条时间序列；
  2) 被限流的 429、未捕获异常的 500 也计入指标（此前看板错误率偏低）；
  3) LLM 调用按模型/状态记录请求数与耗时，成功时记录 token（有 usage 用真值，
     无 usage 按字符数估算），失败记录 error 状态；
  4) create_llm 造出的实例挂上了观测回调，埋点随链路生效。

约定：与项目其它测试一致，同步用例内用 asyncio.run(drive()) 驱动异步逻辑。
"""
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from prometheus_client import REGISTRY

from access.middleware import RateLimiter, register_middleware


def _counter(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _hist_count(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _obs_app(per_minute: int = 100, max_body_size: int = 1024) -> FastAPI:
    app = FastAPI()
    register_middleware(app, RateLimiter(per_minute=per_minute),
                        max_body_size=max_body_size)

    @app.get("/items/{item_id}")
    async def get_item(item_id: str):
        return {"id": item_id}

    @app.get("/boom")
    async def boom():
        raise RuntimeError("unit boom")

    return app


# ===================== C3-1) endpoint 用路由模板 =====================
class TestEndpointLabelUsesRouteTemplate:
    def test_path_param_collapses_to_template(self):
        client = TestClient(_obs_app())
        client.get("/items/123")
        client.get("/items/999")
        assert _counter("web_agent_http_requests_total",
                        {"method": "GET", "endpoint": "/items/{item_id}",
                         "status": "200"}) >= 2
        # 原始带参路径不得成为独立标签（否则每来一个 ID 就多一条时间序列）
        assert _counter("web_agent_http_requests_total",
                        {"method": "GET", "endpoint": "/items/123",
                         "status": "200"}) == 0

    def test_duration_histogram_also_templated(self):
        client = TestClient(_obs_app())
        client.get("/items/777")
        assert _hist_count("web_agent_http_request_duration_seconds_count",
                           {"method": "GET", "endpoint": "/items/{item_id}"}) >= 1
        assert _hist_count("web_agent_http_request_duration_seconds_count",
                           {"method": "GET", "endpoint": "/items/777"}) == 0


# ===================== C3-2) 429 / 500 也计数 =====================
class TestErrorResponsesAreCounted:
    def test_rate_limited_429_counted(self):
        client = TestClient(_obs_app(per_minute=1))
        assert client.get("/items/1").status_code == 200
        assert client.get("/items/2").status_code == 429
        # 限流在路由匹配前返回：label 仍应还原为路由模板，而不是原始 path
        assert _counter("web_agent_http_requests_total",
                        {"method": "GET", "endpoint": "/items/{item_id}",
                         "status": "429"}) >= 1

    def test_unhandled_exception_counted_as_500(self):
        client = TestClient(_obs_app(), raise_server_exceptions=False)
        assert client.get("/boom").status_code == 500
        assert _counter("web_agent_http_requests_total",
                        {"method": "GET", "endpoint": "/boom",
                         "status": "500"}) >= 1


# ===================== C4) LLM 调用观测 =====================
def _llm_result(text: str = "回答内容", usage: dict | None = None,
                model: str = "unit-llm") -> LLMResult:
    result = LLMResult(generations=[[ChatGeneration(message=AIMessage(content=text))]],
                       llm_output={"model_name": model})
    if usage is not None:
        result.llm_output["token_usage"] = usage
    return result


class TestLLMMetricsDefinition:
    def test_labels_are_bounded(self):
        import utils.metrics as m
        assert set(m.llm_requests_total._labelnames) == {"model", "status"}
        assert set(m.llm_tokens_total._labelnames) == {"model", "type"}
        assert set(m.llm_request_duration_seconds._labelnames) == {"model", "status"}
        assert set(m.llm_ttft_seconds._labelnames) == {"model"}


class TestLLMMetricsCallback:
    def test_success_records_request_ttft_and_real_tokens(self):
        from utils.llm_factory import LLMMetricsCallback
        h = LLMMetricsCallback("unit-llm-ok")
        h.on_llm_start({}, ["用户问题"])
        req_before = _counter("web_agent_llm_requests_total",
                              {"model": "unit-llm-ok", "status": "success"})
        prompt_before = _counter("web_agent_llm_tokens_total",
                                 {"model": "unit-llm-ok", "type": "prompt"})
        comp_before = _counter("web_agent_llm_tokens_total",
                               {"model": "unit-llm-ok", "type": "completion"})
        ttft_before = _hist_count("web_agent_llm_ttft_seconds_count",
                                  {"model": "unit-llm-ok"})

        h.on_llm_new_token("答")            # 首 token 打点应在 new_token 时发生
        assert _hist_count("web_agent_llm_ttft_seconds_count",
                           {"model": "unit-llm-ok"}) == ttft_before + 1
        h.on_llm_end(_llm_result(usage={"prompt_tokens": 12, "completion_tokens": 34},
                                 model="unit-llm-ok"))

        assert _counter("web_agent_llm_requests_total",
                        {"model": "unit-llm-ok", "status": "success"}) == req_before + 1
        assert _counter("web_agent_llm_tokens_total",
                        {"model": "unit-llm-ok", "type": "prompt"}) == prompt_before + 12
        assert _counter("web_agent_llm_tokens_total",
                        {"model": "unit-llm-ok", "type": "completion"}) == comp_before + 34

    def test_ttft_observed_only_once_per_call(self):
        from utils.llm_factory import LLMMetricsCallback
        h = LLMMetricsCallback("unit-llm-once")
        h.on_llm_start({}, ["p"])
        before = _hist_count("web_agent_llm_ttft_seconds_count", {"model": "unit-llm-once"})
        h.on_llm_new_token("第")
        h.on_llm_new_token("二")
        h.on_llm_new_token("三")
        assert _hist_count("web_agent_llm_ttft_seconds_count",
                           {"model": "unit-llm-once"}) == before + 1

    def test_estimates_tokens_when_usage_absent(self):
        from utils.llm_factory import LLMMetricsCallback
        h = LLMMetricsCallback("unit-llm-est")
        h.on_llm_start({}, ["四个字的问题"])          # 6 字符
        h.on_llm_end(_llm_result(text="答案", usage=None, model="unit-llm-est"))
        # 流式常拿不到 usage：此时按字符数估算，绝不能记 0（看板会误判为无流量）
        assert _counter("web_agent_llm_tokens_total",
                        {"model": "unit-llm-est", "type": "prompt"}) == 6
        assert _counter("web_agent_llm_tokens_total",
                        {"model": "unit-llm-est", "type": "completion"}) == 2

    def test_error_records_error_status_only(self):
        from utils.llm_factory import LLMMetricsCallback
        h = LLMMetricsCallback("unit-llm-err")
        h.on_llm_start({}, ["p"])
        h.on_llm_error(RuntimeError("unit boom"))
        assert _counter("web_agent_llm_requests_total",
                        {"model": "unit-llm-err", "status": "error"}) == 1
        assert _hist_count("web_agent_llm_request_duration_seconds_count",
                           {"model": "unit-llm-err", "status": "error"}) == 1
        # 失败调用不产生 token 计数
        assert _counter("web_agent_llm_tokens_total",
                        {"model": "unit-llm-err", "type": "completion"}) == 0

    def test_create_llm_attaches_callback(self):
        from utils.llm_factory import LLMMetricsCallback, create_llm
        llm = create_llm(model="unit-model-wire", streaming=False)
        handlers = [c for c in (llm.callbacks or [])
                    if isinstance(c, LLMMetricsCallback)]
        assert handlers and handlers[0].model == "unit-model-wire"

    def test_concurrent_runs_do_not_share_state(self):
        """两个 run 交错时，prompt token / 首 token / 耗时必须各算各的。

        回归防护：旧实现把 _start/_prompt_chars 放在 callback 实例字段里，
        共享 LLM 并发调用时后一个 run 会覆盖前一个 run 的起点。
        """
        from utils.llm_factory import LLMMetricsCallback
        h = LLMMetricsCallback("unit-llm-concurrent")
        prompt_before = _counter("web_agent_llm_tokens_total",
                                 {"model": "unit-llm-concurrent", "type": "prompt"})

        h.on_llm_start({}, ["a" * 6], run_id="run-a")
        h.on_llm_start({}, ["b" * 20], run_id="run-b")
        h.on_llm_new_token("x", run_id="run-a")
        h.on_llm_end(_llm_result(usage={"prompt_tokens": 6, "completion_tokens": 2},
                                 model="unit-llm-concurrent"), run_id="run-a")
        h.on_llm_end(_llm_result(usage={"prompt_tokens": 20, "completion_tokens": 4},
                                 model="unit-llm-concurrent"), run_id="run-b")

        # run-a 的 6 不能被 run-b 的 20 覆盖（旧实现会记录成 20+20 或 20+6）
        assert _counter("web_agent_llm_tokens_total",
                        {"model": "unit-llm-concurrent", "type": "prompt"}) == prompt_before + 26
