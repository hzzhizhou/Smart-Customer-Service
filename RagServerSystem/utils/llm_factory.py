"""
统一 LLM 工厂：DashScope OpenAI 兼容端点（OpenAI 兼容协议调用通义模型）。
背景：通义原生 ChatTongyi/dashscope 无法识别 qwen3.7-flash 模型名（url error），
统一改用 OpenAI 兼容端点。key、模型均为通义账号，仅调用协议不同。
"""
import time
from dataclasses import dataclass
from typing import Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from config.settings import (
    LLM_MODEL, LLM_TEMPERATURE,
    DASHSCOPE_API_KEY, LLM_BASE_URL, LLM_TIMEOUT,
)
from utils.metrics import (
    llm_requests_total, llm_tokens_total,
    llm_request_duration_seconds, llm_ttft_seconds,
)


@dataclass
class _RunMetrics:
    """一次 LLM run 的观测状态。

    必须按 run_id 隔离：create_llm 创建的 callback 实例挂在共享的 ChatOpenAI 上，
    并发请求会同时触发 start/new_token/end。若把 start/first_token 放在 callback
    实例字段里，后一个请求会覆盖前一个请求的起点，TTFT/耗时/token 全部串账。
    """

    started: float
    prompt_chars: int
    first_token_at: Optional[float] = None


class LLMMetricsCallback(BaseCallbackHandler):
    """按次观测 LLM 调用：耗时、首 token 延迟、token 用量、成败（PROJECT_AUDIT §7.2(6)）。

    挂在 LLM 实例上随链路自动触发：start → new_token（可多次）→ end/error。
    每次 run 的状态按 run_id 存在字典里，支持同一 LLM 实例上的并发调用。
    token 口径：有 usage（非流式或已开启 include_usage）用真值；流式常拿不到 usage，
    此时按字符数估算（中文≈1 字 1 token 的保守近似），只是趋势观测，不等于计费口径。
    """

    def __init__(self, model: str):
        self.model = model
        self._runs: Dict[str, _RunMetrics] = {}

    # 普通 LLM 走 on_llm_start；Chat 模型走 on_chat_model_start，两者都要接住
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs):
        self._begin("\n".join(prompts or []), run_id)

    def on_chat_model_start(self, serialized, messages, *, run_id=None, **kwargs):
        text = "\n".join(str(getattr(m, "content", m)) for batch in messages for m in batch)
        self._begin(text, run_id)

    def _begin(self, prompt_text: str, run_id=None) -> None:
        # 极端情况下上游没配对 end/error，最多保留 512 条，避免无界增长
        if len(self._runs) >= 512:
            self._runs.pop(next(iter(self._runs)), None)
        self._runs[str(run_id or "")] = _RunMetrics(
            started=time.perf_counter(),
            prompt_chars=len(prompt_text or ""),
        )

    def on_llm_new_token(self, token, *, run_id=None, **kwargs):
        state = self._runs.get(str(run_id or ""))
        if state is None or state.first_token_at is not None:
            return
        state.first_token_at = time.perf_counter()
        llm_ttft_seconds.labels(self.model).observe(
            max(state.first_token_at - state.started, 0.0))

    def on_llm_end(self, response, *, run_id=None, **kwargs):
        state = self._runs.pop(str(run_id or ""), None)
        model = self._model_of(response)
        llm_requests_total.labels(model, "success").inc()
        llm_request_duration_seconds.labels(model, "success").observe(
            self._elapsed(state))
        prompt_tokens, completion_tokens = self._usage(response, state)
        llm_tokens_total.labels(model, "prompt").inc(prompt_tokens)
        llm_tokens_total.labels(model, "completion").inc(completion_tokens)

    def on_llm_error(self, error, *, run_id=None, **kwargs):
        state = self._runs.pop(str(run_id or ""), None)
        llm_requests_total.labels(self.model, "error").inc()
        llm_request_duration_seconds.labels(self.model, "error").observe(
            self._elapsed(state))

    @staticmethod
    def _elapsed(state: Optional[_RunMetrics]) -> float:
        if state is None:
            return 0.0
        return max(time.perf_counter() - state.started, 0.0)

    @staticmethod
    def _model_of(response) -> str:
        llm_output = getattr(response, "llm_output", None) or {}
        return llm_output.get("model_name") or LLM_MODEL

    def _usage(self, response, state: Optional[_RunMetrics]) -> tuple:
        """优先取真实 usage，缺失则按字符数估算。"""
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if usage:
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            return prompt, completion
        completion_chars = 0
        for batch in getattr(response, "generations", []) or []:
            for gen in batch:
                completion_chars += len(getattr(gen, "text", "") or "")
        prompt_chars = state.prompt_chars if state is not None else 0
        return self._estimate(prompt_chars), self._estimate(completion_chars)

    @staticmethod
    def _estimate(chars: int) -> int:
        # 中文≈1 字 1 token；英文偏保守（会高估），仅用于看板趋势，非计费口径
        return max(int(chars), 0)


def create_llm(model: Optional[str] = None,
               temperature: Optional[float] = None,
               streaming: bool = True,
               top_p: Optional[float] = None,
               seed: Optional[int] = None) -> ChatOpenAI:
    """创建指向 DashScope OpenAI 兼容端点的 LLM（自带调用观测回调）。"""
    model_name = model or LLM_MODEL
    kwargs = dict(api_key=DASHSCOPE_API_KEY, base_url=LLM_BASE_URL, streaming=streaming,
                  # 超时 + 重试上限一起设：只设超时不够，langchain-openai 默认重试 2 次，
                  # 最坏 3×30s 才走到降级；限为 1 次让「模型挂住」的最坏耗时可控。
                  timeout=LLM_TIMEOUT, max_retries=1,
                  callbacks=[LLMMetricsCallback(model_name)])
    if top_p is not None:
        kwargs['top_p'] = top_p
    if seed is not None:
        kwargs['seed'] = seed
    return ChatOpenAI(
        model=model_name,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        **kwargs,
    )